# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Ranking Approach for Measuring Calibration](https://arxiv.org/abs/2609.13100) | 本文提出了一种新的校准误差度量方法 rankECE，通过比较预测概率的邻近值来衡量校准误差，理论和实验证明它比常用的分箱近似方法能更好地逼近期望校准误差（ECE）。 |
| [^2] | [Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval](https://arxiv.org/abs/2609.13073) | 本文通过电信工单检索这一工业级案例研究，探讨了基于大语言模型的自主研究在开放式机器学习问题中的应用，发现其在狭窄的超参数优化中表现出色，但在处理具有表示、架构和训练数据生成自由度的开放式问题时仍存在局限性。 |
| [^3] | [MAxBench: A Multinomial Concept Recovery Benchmark](https://arxiv.org/abs/2609.13072) | 本文提出了MAxBench，一个几何无关的多分类概念表示评估框架，通过在6个概念和4个模型上系统比较10种定位方法（涵盖5种几何类型），为识别最适合多分类概念引导的表示几何结构及最有效的恢复方法提供了统一评估标准。 |
| [^4] | [CanvasAnneal: Curriculum Reinforcement Learning for Diffusion Language Models](https://arxiv.org/abs/2609.13060) | CanvasAnneal通过课程学习策略，先在扩散画布中注入教师模型的推理轨迹以热启动强化学习探索，再随着训练逐步移除引导让模型学会独立推理，从而显著提升扩散语言模型在数学推理和工具使用任务上的表现。 |
| [^5] | [Benign Loss Landscapes Can Coexist with Worst-Case Hardness](https://arxiv.org/abs/2609.13057) | 本文研究了树张量网络，证明其损失景观虽然是条件良性的（易于优化），但仍包含梯度下降无法在多项式时间内学习的最坏情况目标，从而揭示了良性损失景观可以与最坏情况的学习难度共存。 |
| [^6] | [Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model](https://arxiv.org/abs/2609.13053) | 提出基于全模态掩码扩散主干网络的统一视觉-语言-动作模型Dynin-Robotics，通过共享轨迹模型将目标预测与动力学预测引入动作生成与选择，实现多任务联合学习并支持测试时性能扩展。 |
| [^7] | [A Unified and Constrained View of Regularization-Based Robust Reinforcement Learning](https://arxiv.org/abs/2609.13050) | 该论文通过推导性能差距上界统一了基于正则化的鲁棒强化学习方法，将鲁棒训练表述为约束优化问题，并通过联合更新拉格朗日乘子自动调整正则化权重以提升鲁棒性。 |
| [^8] | [MCRL2: Multi-resource Cross-attention-based Representation Learning-augmented Reinforcement Learning for Cloud Microservice Scheduling](https://arxiv.org/abs/2609.13048) | 该论文提出了MCRL2，一种将多资源交叉注意力表示学习与强化学习相结合的新方法，通过捕捉节点、资源和微服务之间的结构化交互关系，有效提升云环境下微服务调度的负载均衡能力和服务质量。 |
| [^9] | [Diffusion Models and Concept Formation](https://arxiv.org/abs/2609.13047) | 本文论证扩散模型虽为图像合成而设计，但其含噪边缘分布的众数所形成的层次结构在四个方面与认知科学中经典的Cobweb概率概念层次一一对应，表明扩散模型隐式地执行了概念形成计算，且“基本层次”在中间噪声水平处涌现。 |
| [^10] | [Robust Policy Optimization via Adversarial Importance Sampling](https://arxiv.org/abs/2609.13044) | 该论文提出了Advis方法，通过对标准训练轨迹进行重要性采样来估计和优化可验证的最坏情况回报，无需额外环境交互和辅助网络即可捕获长期鲁棒性，同时提供模块化PyTorch库advrl以促进鲁棒DRL方法的快速开发与可复现评估。 |
| [^11] | [DynSHAP: Towards Explainable Dynamic Survival Analysis](https://arxiv.org/abs/2609.13042) | DynSHAP是一个专为动态生存分析设计的SHAP可解释性框架，通过将时间-特征对作为Shapley博弈的参与者，并引入利用条件采样处理特征时间依赖关系的Temporal DynSHAP，从而能够准确解释纵向不规则输入与函数型生存输出。 |
| [^12] | [Quantile-based Loss Filtering for Outlier-Robust Stochastic Gradient Descent](https://arxiv.org/abs/2609.13040) | 该论文提出了分位数-k-损失SGD（QkL-SGD）框架，通过每次迭代采样k个分量损失并从经验下q分位数中均匀选取更新索引来过滤损坏分量，在标准凸性假设下证明了线性收敛性，并提供了适用于任意采样规模的小样本概率收敛分析。 |
| [^13] | [Transfer Learning for Evolving Domains](https://arxiv.org/abs/2609.13039) | 该论文提出了一种新的迁移学习问题——面向演化领域的迁移学习，将传统上按目标数据可用性假设相互分割的各子领域统一起来，把迁移学习建模为部署系统随时间逐步获得目标数据与标签的完整演化轨迹。 |
| [^14] | [Groupoid-Based Internal State Representations for Reinforcement Learning with Local Symmetries](https://arxiv.org/abs/2609.13035) | 提出了一种基于群胚的强化学习框架，能够捕获局部的、状态依赖的对称性并在交互中动态发现等价结构，使智能体在对称性约简的空间中进行学习和决策，同时保留局部区别。 |
| [^15] | [Attention Quantization for Tabular Foundation Models](https://arxiv.org/abs/2609.13031) | 本文提出针对表格基础模型的FP8注意力量化策略，通过将测试行与训练行的量化误差对齐，在避免精度损失的同时，利用Triton内核实现了最高1.7倍的注意力计算加速。 |
| [^16] | [Dual-guided Hierarchical Edge Localization for Large-scale Optimal Transport Across Dimensions](https://arxiv.org/abs/2609.13010) | 提出HELLO层次化求解器，利用对偶势双重引导将大规模离散最优传输转化为边定位问题，在百万点规模下以线性内存复杂度获得更低传输目标值和数量级级别的运行时间加速。 |
| [^17] | [Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage](https://arxiv.org/abs/2609.13003) | 该研究揭示TruthfulQA等二选一真实性基准存在表层特征泄漏问题，模型无需真正推理即可借助答案表面特征超越随机水平，作者据此提出Audit-Prune清理机制并发布了泄漏接近随机水平的清理版数据集。 |
| [^18] | [A Full Adam Theorem for Spectral Heavy-Tail Onset](https://arxiv.org/abs/2609.12996) | 该论文在高斯Stein-Hermite师生状态演化模型中证明了谱重尾出现的完整Adam定理，给出了由首个尖峰-主体谱间隙决定的命中时间定律 τ_ε = Θ(Δ_1^{-γ} d^ρ log(Ψ_0/ε))。 |
| [^19] | [Dimension-Corrected Hitting Times for Heavy-Tailed Spectral Emergence in Neural Optimizer Dynamics](https://arxiv.org/abs/2609.12994) | 该论文将神经网络训练中重尾谱的涌现时间首次建模为右删失命中时间问题，并实证发现由谱隙与权重矩阵维度共同决定的修正幂律 τ_HT ≈ C·Δ₁^(-γ)·d^ρ 能够准确预测重尾涌现时间，显著优于仅依赖谱隙的模型。 |
| [^20] | [Information-Induced Training Geometry: Exact Reduction, Canonical Completion, and Structured Expressivity](https://arxiv.org/abs/2609.12991) | 该论文证明在仿射不变黎曼几何下，部分信息通道对优化器度量的压缩具有显式且唯一的典范补全，补全随通道移动形成规范不变的秩分层，从而实现了从完整几何到可见几何的精确变分约简与结构化表达能力刻画。 |
| [^21] | [A Large-Scale AIS Dataset from Finnish Water](https://arxiv.org/abs/2609.12938) | 本文发布了一个来自芬兰水域（波罗的海区域）的大规模AIS数据集，包含2.29亿个数据点并涵盖芬兰湖泊数据，为海事活动和船舶行为研究提供了宝贵资源。 |
| [^22] | [Dissecting GPU Utilization for LLM Inference on Nvidia Hopper](https://arxiv.org/abs/2609.12923) | 论文揭示单一的SM利用率指标会掩盖LLM推理中GPU真实的工作情况，并提出了八个基于NCU计数器验证的多维视图，以更精确地剖析Hopper GPU上LLM推理的性能瓶颈。 |
| [^23] | [PA-CDM: Position-Aware Character Detection Matching for Evaluating Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2609.12917) | 提出了PA-CDM位置感知评估指标，通过将字符检测匹配与位置森林编码和散度级加权相结合，解决了现有手写数学表达式识别评估方法无法感知错误发生位置的缺陷。 |
| [^24] | [Offline Reinforcement Learning for Wind Farm Control: A Wind Tunnel Study under Dynamic Wind Directions](https://arxiv.org/abs/2609.12905) | 提出了一种离线强化学习算法MTD3-BC，通过在策略优化中引入新的动作一致性项实现平滑的偏航控制，在动态风向条件下有效最大化风电场功率，且无需与模拟器大量交互，显著降低了计算成本和训练时间。 |
| [^25] | [Hidden in Rounds: Predicting the Time Cost of 802.11 Contention in Federated Learning](https://arxiv.org/abs/2609.12903) | 本文提出一种结合ns-3测量与Bianchi模型估计器的方法来预测联邦学习中802.11信道竞争的时间成本，发现达到目标精度所需的通信时间随客户端密度增加约两个数量级，而所需训练轮次几乎不变。 |
| [^26] | [Physics-enriched neural solvers for transient ice-flow simulation](https://arxiv.org/abs/2609.12900) | 提出一种物理增强方法，通过将低阶冰流平衡推导的低成本物理场作为神经网络输入来改进在线冰川流动神经求解器，无需修改损失函数即可显著提升求解器鲁棒性并改善精度与运行时间的权衡。 |
| [^27] | [Physical-State-Guided Diffusion Sampling for Full-Waveform Inversion](https://arxiv.org/abs/2609.12899) | 提出物理状态引导扩散采样方法（PSG），通过高斯桥将物理速度状态与扩散先验耦合，实现波动方程梯度与去噪器梯度的分离，在全波形反演中超越了经典方法和基于扩散的基线方法。 |
| [^28] | [Behavior Quotient Learning for Low-Rank Adaptation of LLM Agents](https://arxiv.org/abs/2609.12896) | 提出BQ-LoRA框架，通过局部行为商流形组织轨迹更新，使单个LoRA适配器能够高效学习LLM智能体的多样化交互能力，同时避免多适配器带来的存储和路由开销。 |
| [^29] | [Beyond Accuracy: Uncertainty-Guided Boundary Refinement for Reliable Biomedical Image Segmentation](https://arxiv.org/abs/2609.12892) | 提出可靠性感知边界细化网络RABR-Net，通过融合预测熵、测试时增强方差等多种不确定性度量构建边界感知的可靠性表示，选择性地纠正不确定的边界像素，从而在保持高分割准确率的同时实现可靠的生物医学图像分割。 |
| [^30] | [Quantifying the Value of Privileged Information Using a PAC-Bayesian Approach](https://arxiv.org/abs/2609.12891) | 该论文提出了一种基于PAC-贝叶斯框架的算法无关信息论方法，用于量化特权信息（LUPI范式）在训练中传递的价值，填补了缺乏解释特权信息如何及何时传递有用知识的通用理论框架这一空白。 |
| [^31] | [Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting](https://arxiv.org/abs/2609.12890) | 提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。 |
| [^32] | [What an odour descriptor corpus can and cannot measure: valence, attenuation, and the ceiling of the public record](https://arxiv.org/abs/2609.12875) | 本研究首次系统审计了四个公共气味描述词语料库，发现各语料库对同一描述词的使用存在无法用单一校准修复的异质性差异，并揭示当前基于分子指纹的模型仅能达到人类专家小组可靠性的约33%，从而界定了公共气味数据记录所能测量的上限。 |
| [^33] | [A Multi-Vehicle Dataset with Camera, LiDAR, and Radar Sensors and Scanned 3D Models for Custom Auto-Annotation using RTK-GNSS](https://arxiv.org/abs/2609.12871) | 该论文提出了一个多车辆数据集，包含相机、激光雷达和雷达传感器数据，并结合扫描3D模型与RTK-GNSS位姿参考，支持用户自定义粒度的自动标注，可深入评估遮挡等测量效应。 |
| [^34] | [GenOR-Twin: A Semantic Middleware for Integrating Operational Discourse with Mathematical Optimization](https://arxiv.org/abs/2609.12863) | GenOR-Twin是一个神经符号框架，它将大语言模型定位为语义翻译器而非直接求解器，通过动态约束注入机制将定性运营事件实时转化为数学约束，实现运营话语与数学优化之间的双向耦合，从而构建满足同步要求的数字孪生系统。 |
| [^35] | [Very Exciting: Zero-Shot Model Predictive Control of Buildings via Excitation-Based Generalized Transfer Learning Models](https://arxiv.org/abs/2609.12853) | 该论文提出使用基于激励式探测数据的广义迁移学习模型，实现了建筑的零样本模型预测控制，无需在目标建筑收集数据即可获得令人满意的控制性能。 |
| [^36] | [VertexCBF: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search](https://arxiv.org/abs/2609.12831) | 提出VertexCBF框架，利用控制仿射动力学和凸多面体控制集的性质（哈密顿量在控制顶点处最大化），结合物理信息学习、稀疏监督学习与GPU并行的顶点受限树搜索，以可扩展、系统化且可解释的方式训练神经控制屏障函数。 |
| [^37] | [4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling](https://arxiv.org/abs/2609.12815) | BEAST是首个能在0.25°全球分辨率下进行大气预报并同时量化偶然与认知不确定性的贝叶斯Swin Transformer，其创新的四维并行方案（域-张量并行与不确定性并行）在20,480块GPU上实现了3.96 EFLOP/s的峰值性能，预测技巧可与最先进的概率AI模型和数值模型相媲美。 |
| [^38] | [RunningTensor: Generalizing Linear Attention to Higher-Order Recurrent States](https://arxiv.org/abs/2609.12814) | 本文提出 RunningTensor，将线性注意力的矩阵循环记忆推广为高阶（o 阶）张量状态，在保持对序列长度线性的时间复杂度下，将工作记忆容量从 O(W²) 提升到 O(W^o)，并在联想回忆、语言理解和检索任务上均优于基线方法。 |
| [^39] | [VertiFuseX: Generalizable Financial Forecasting via Multi-Stream Temporal Fusion](https://arxiv.org/abs/2609.12793) | VertiFuseX提出一种混合LSTM架构，通过对LSTM、Bi-LSTM和St-LSTM分支的倒数第二层特征进行垂直融合，在固定超参数配置下实现了更具信息保留能力和跨市场泛化性的股票价格预测。 |
| [^40] | [Convergence of Stochastic Gradient Methods under Heavy-Tailed Noise and H\"{o}lder Smoothness](https://arxiv.org/abs/2609.12785) | 本文在同时放宽Lipschitz光滑性与有限方差噪声这两个经典假设的条件下，证明了标准SGD和$\delta$-正则化梯度裁剪方法在Hölder光滑目标函数和重尾梯度噪声下的非凸收敛速率。 |
| [^41] | [High-Probability Convergence of SGD via Batched Updates](https://arxiv.org/abs/2609.12765) | 本文提出Batched SGD，通过将在线样本分批、每批仅用低方差梯度估计执行一次更新，在无需限制性假设或辅助序列的情况下，以极其简洁的证明获得了SGD最后迭代点的高概率近最优收敛速率。 |
| [^42] | [Same Encoder, Different Winner: A Paired-View Framework for Cell Painting Encoder Evaluation](https://arxiv.org/abs/2609.12761) | 本文提出CP-BG-Bench成对视图评估框架，通过控制中心细胞周围像素的干预操作，揭示四种社区标准评估协议会对相同的Cell Painting编码器给出系统性不同的排名，其分歧可归结为细胞与背景、形态与上下文、研究内与跨批次三个维度。 |
| [^43] | [Curriculum-Based Adversarial Heterogeneous Agent Reinforcement Learning for Autonomous Quad-Copter Landing in Maritime Settings](https://arxiv.org/abs/2609.12758) | 本文提出结合课程学习与对抗性风场智能体的异构智能体强化学习方法（HARL-AC），用于训练船载机械臂空中捕获四旋翼无人机的鲁棒协作控制策略，在分布外海况下相比域随机化方法具有更好的泛化性能。 |
| [^44] | [Optimizing for the decision not the prediction: an exploration of Smooth Net Benefit as a training objective](https://arxiv.org/abs/2609.12752) | 本文提出平滑净获益（σNB）这一可微分训练目标，使预测模型的训练与特定风险阈值下的临床决策效用对齐，但在多个基准数据集上仅带来边际性能提升。 |
| [^45] | [What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework](https://arxiv.org/abs/2609.12746) | 该研究通过LAST-CQ框架上的反事实实验发现，智能体Text-to-Cypher系统的恢复收益主要来自失败检测与重试路由机制，而非复杂的LLM反馈内容。 |
| [^46] | [Physics-Guided Synthetic High-Frequency Ultrasound Generation for Skin Layer Segmentation](https://arxiv.org/abs/2609.12735) | 该论文提出一种物理引导的合成高频超声生成框架，通过k-Wave声学仿真生成成对的超声图像与密集层标注数据，解决了皮肤层分割任务中深层结构密集标注数据稀缺的问题，并验证了合成数据预训练向真实数据迁移的有效性。 |
| [^47] | [Prism-SQA: An Interpretable and Adaptable Neural Framework for Surface Electromyography Quality Assessment](https://arxiv.org/abs/2609.12724) | 本文提出Prism-SQA框架，将sEMG信号质量评估重构为生理感知的源分离与验证过程，通过将信号分解为干净成分和五种污染成分，实现了可解释、可适应且无需重新训练的信号质量评估。 |
| [^48] | [Inferring Dislocation Microstructures from X-ray Diffraction via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2609.12713) | 本文提出了一种跨模态对比学习框架，将离散位错动力学模拟生成的位错密度场与虚拟X射线衍射图谱嵌入共享潜在空间，实现了直接从衍射数据预测三维位错微观结构。 |
| [^49] | [InRTL: Effective Intra-Inter Interaction Learning for Relational Tables](https://arxiv.org/abs/2609.12712) | 提出InRTL统一框架，通过列感知表编码器与基于Transformer的自注意力和交叉注意力模块，显式建模关系表内部的行关联以及跨PK-FK连接表的行依赖，实现有效的关系表学习。 |
| [^50] | [ExpertHTR: Unified Handwritten Text Recognition with Multi-Task Learning and Sparse Mixture-of-Experts](https://arxiv.org/abs/2609.12705) | ExpertHTR提出了一种基于多任务学习与稀疏混合专家的统一视觉-语言框架，通过统一的页-区域-行表示整合异构手写识别数据集，构建四个互补训练任务，并利用Sparsegen机制实现专家数量的动态激活，无需额外人工标注。 |
| [^51] | [Write on Paper and Get the Online Digital Trace:\newline A New Era for Handwriting](https://arxiv.org/abs/2609.12702) | 该论文提出了一种结合传感器数字笔与先进AI算法的创新方案，无需特殊纸张或外部参考系统，即可实时重建在普通纸上书写的手写数字轨迹。 |
| [^52] | [SIFPBPNet: A Dual-Path Network for Wearable and Cuffless Blood Pressure Estimation via Individualized Steady-state Representation](https://arxiv.org/abs/2609.12690) | 该论文提出双路径网络SIFPBPNet，通过图注意力网络从多天历史PPG轨迹中提取个体化稳态特征，并结合瞬时特征路径与交叉注意力机制，有效解决了无袖带血压估计中的人群异质性和“一对多映射”问题。 |
| [^53] | [What Did the MLLM Hear? Token-Level Spectro-Temporal Grounding for Audio MLLM Explainability](https://arxiv.org/abs/2609.12663) | 提出了STAG——首个对音频多模态大语言模型生成描述进行词元级时频定位的事后解释框架，通过词元特定词汇投影与频谱遮挡相结合，揭示每个生成词元所依赖的时间和频率声学证据。 |
| [^54] | [ProactiveBench: Can Streaming Video Models Really Interact Like Humans?](https://arxiv.org/abs/2609.12658) | 提出了ProactiveBench基准，以一秒流式间隔、无明确响应提示的方式评估流式视频模型的主动交互能力（即在恰当时机响应目标事件、其余时间保持沉默），发现多数系统存在过早响应而非漏响应的问题。 |
| [^55] | [Distortion of AI Alignment Revisited: RLHF is a Decent Utilitarian Aligner](https://arxiv.org/abs/2609.12651) | 本文通过细粒度分析证明，RLHF失真的指数级退化并非算法本身的固有缺陷，而是源于偏好数据分布与KL参考策略之间的分布不匹配，从而表明RLHF实际上是一个不错的功利主义对齐器。 |
| [^56] | [Explaining Time Series Forecasting with Horizon-Resolved Attribution](https://arxiv.org/abs/2609.12639) | 该论文提出HRX即插即用框架，通过在时间序列预测解释中引入预测步长维度，为每个预测步骤生成独立的重要性归因图，突破了传统方法假设所有预测步骤依赖相同过去值的局限。 |
| [^57] | [Geometric-to-Semantic Spherical Transfer Learning for Cortical Sulci Labeling](https://arxiv.org/abs/2609.12627) | 该论文提出一种几何到语义球面迁移学习框架，利用约3万例无标注数据预训练球面编码器，解决了皮层脑沟标注中因专家标注极度稀缺而导致的过拟合和泛化困难问题。 |
| [^58] | [Correlation-Guided Fast Machine Unlearning via Hessian Analysis](https://arxiv.org/abs/2609.12620) | 本文提出了一种基于Hessian分析的计算高效机器遗忘框架，通过识别训练集中的相关数据点并应用理论推导的闭式解，避免了传统方法中每个数据点移除所需的昂贵Hessian逆向量重复计算，显著提升了入侵检测等安全系统批量处理遗忘请求的效率。 |
| [^59] | [SIMS: Scale-Invariant Merit-Function-Based Scalarization for Multi-Task Learning](https://arxiv.org/abs/2609.12599) | 提出了尺度不变价值函数标量化方法SIMS，解决了现有多任务学习标量化方法因任务损失尺度差异而偏向大尺度目标的问题，使优化结果不受目标缩放的影响。 |
| [^60] | [Poisson-Corrector Complexity Bounds for Moreau--Yosida Unadjusted Langevin Sampling](https://arxiv.org/abs/2609.12594) | 该论文通过离散泊松校正子技术为Moreau-Yosida未调整朗之万算法（MYULA）建立了更优的非渐近复杂度界，证明在W2距离下达到ε精度仅需$\widetilde O(\varepsilon^{-4/3})$次迭代，且误差系数对平滑参数仅呈对数依赖。 |
| [^61] | [Where Decoder Cosine Similarity Fails for SAE Feature Flow Discovery](https://arxiv.org/abs/2609.12591) | 本文提出构建SAE特征“转换图集”的方法来系统发现残差状态特征与MLP更新特征如何共同预测下游残差特征，并揭示了余弦相似度筛选在此类特征流发现中的失效之处。 |
| [^62] | [Tight Sampling Complexity with stochastic gradient oracles in Fixed Dimensions](https://arxiv.org/abs/2609.12590) | 本文证明了固定维度下使用随机梯度预言机采样光滑强对数凹分布的最优查询复杂度为 Θ(log(1+κ) + σ²/(με))，该复杂度界同时对条件数和精度ε均达到紧致最优，并自适应于无噪声情形。 |
| [^63] | [Clustering-Based Balanced Sampling and Allocation with Data Parallelism for High-Performance Fine-Tuning](https://arxiv.org/abs/2609.12584) | CluSTER框架通过梯度空间聚类与数据并行感知的均衡分配构建代表性缩减数据集，在保持模型精度的同时将指令微调训练时间最多缩短69.6%。 |
| [^64] | [SCOPE-OPSD: Fisher-Conditioned Privileged Subspaces for On-Policy Self-Distillation](https://arxiv.org/abs/2609.12579) | SCOPE-OPSD通过将师生残差投影到基于Fisher敏感度估计的冻结低秩子空间，为在线策略自蒸馏引入了一个结构化的第二监督通道，在无需额外采样或推理模块的情况下，于多个Qwen3模型上稳定超越了纯OPSD。 |
| [^65] | [TokenMapper: A Step Toward Interoperable Speech Token Translation](https://arxiv.org/abs/2609.12563) | 提出TokenMapper框架，能够在离散域中直接实现异构语音分词器之间（包括单码本与多码本表示）的token到token转换，避免了先解码为波形音频再重新编码所带来的延迟和信息损失。 |
| [^66] | [Quality-Constrained Routing over a Fixed Pool of Quantized Mixture-of-Experts Instances](https://arxiv.org/abs/2609.12550) | 提出在固定的量化混合专家实例池内，利用基于请求预测的脆弱性加权困惑度（FWP）指标，在质量退化预算和实例容量约束下将请求路由至吞吐量最优实例的方法。 |
| [^67] | [$\text{GSF-}\chi$: Global Stereochemical Fields for Chiral Graph Transformers](https://arxiv.org/abs/2609.12532) | 提出GSF-χ图Transformer，利用立体化学单元生成的全局相位场和手性RoPE旋转机制，使分子编码模型能够区分镜像对映异构体，同时保持对原子标记和本征旋转的不变性。 |
| [^68] | [Temporal Recurrence Favors Fewer Layers](https://arxiv.org/abs/2609.12531) | 该研究发现时间循环能够替代每步内部的深层结构，使最优计算分配显著转向更少的层数，并在性能相当或更优的情况下节省层内计算。 |
| [^69] | [Learning-Augmented Optimization for Strategic Two-Echelon Spare Parts Network Design](https://arxiv.org/abs/2609.12524) | 提出一个结合图神经网络集成、变邻域搜索和集合划分重组的保守优化框架，通过采用集成预测节省值的下分位数来抑制代理模型的乐观偏差，从而高效求解评估代价高昂的两级备件库存网络战略设计问题。 |
| [^70] | [A Splitting Method for SDE Terminal-Law Estimation](https://arxiv.org/abs/2609.12513) | 本文全面研究了通过分裂部分路径生成路径树来提高随机微分方程终端分布估计效率的方法，以Kolmogorov-Smirnov距离为精度度量确定了模拟预算趋于无穷时的极限误差，并刻画了由渐近优化问题驱动的最优分裂策略。 |
| [^71] | [A Differentially Private Federated Proximal Optimization Framework for Customer Churn Prediction in Heterogeneous Federated Telecom Networks](https://arxiv.org/abs/2609.12470) | 该论文提出了一种基于差分隐私的联邦近端优化框架，能够在保护数据隐私的前提下，应对电信运营商间客户数据异构（非独立同分布）的挑战，实现准确的客户流失预测。 |
| [^72] | [Linear Exponential Quadratic Gaussian Covariance Steering](https://arxiv.org/abs/2609.12463) | 本文提出并分析了连续时间下的线性指数二次高斯（LEQG）协方差控制问题，证明最优线性状态反馈控制器由一个通过求解编码风险敏感度参数隐式依赖的代数方程的对称矩阵参数化，并将风险中性情形的现有结果显著推广。 |
| [^73] | [SAGE-Loop: Reliable Closed-Loop LLM-Driven AutoML with Trial-and-Correction and Adaptive Ensembling](https://arxiv.org/abs/2609.12455) | 提出了SAGE-Loop，一个具备试错-纠正机制与自适应集成策略的可靠闭环LLM驱动AutoML框架，解决了传统单向流水线缺乏过程级纠错和集成决策静态化的问题。 |
| [^74] | [Bridging Vision Foundation Model Priors with CLIP for Spatial-aware Few-shot Anomaly Detection in Medical Images](https://arxiv.org/abs/2609.12454) | 该论文提出Spatial-FAD框架，通过将DINO等视觉基础模型的空间结构先验注入CLIP特征，弥补CLIP全局对比预训练缺乏空间监督的不足，从而提升少样本医学图像异常检测中的病灶定位精度。 |
| [^75] | [3D Digital Twin Visualization of Multiclass GRF-Based Gait Disorder Classification](https://arxiv.org/abs/2609.12442) | 本文提出了一个集成框架，利用双侧地面反作用力信号对健康步态和多种肌肉骨骼障碍进行高精度分类，并结合ε-LRP可解释性分析和基于Blender的三维数字孪生可视化，实现了模型透明度的提升。 |
| [^76] | [IMPLY: Physically Anchored Consistency for World-Model Rollouts](https://arxiv.org/abs/2609.12441) | 论文提出 IMPLY 方法，通过反演模拟器从世界模型的推演中读取隐含物理量（如质量与摩擦力），并以两次校准推力为锚定来评估推演集合的物理一致性，从而克服仅依赖推演间自一致性检查的局限，有效识别出忽略物体物理属性而失败的世界模型。 |
| [^77] | [Beyond the Query: Do Retrieval Signals Improve Adaptive Multimodal RAG Routing?](https://arxiv.org/abs/2609.12437) | 研究表明，在自适应多模态RAG路由中，加入检索信号相比仅使用查询的路由器并不能可靠地改进RUN/SKIP决策，因此检索状态特征只有在匹配的仅查询对照实验中证明有改进时才应被认为具有路由价值。 |
| [^78] | [Observation-Anchored Selective Assimilation for Longitudinal Tumor-State Proxy Forecasting in Post-Treatment Glioma](https://arxiv.org/abs/2609.12435) | 该论文提出观测锚定选择性同化（OASA）方法，将治疗后胶质瘤的肿瘤状态预测建模为观测感知的数字孪生更新，以观测到的中间肿瘤状态代理作为锚点并选择性应用SegMamba预测的更新，从而提升纵向MRI预测的可靠性。 |
| [^79] | [Granularity-Adaptive Credit Assignment for Long-Horizon LLM Agent Reinforcement Learning](https://arxiv.org/abs/2609.12424) | 提出GACA，一种无评论家的粒度自适应信用分配方法，通过基于不确定性的逐步权重动态融合轨迹级与步级优势估计，精准识别长程大语言模型智能体任务中驱动结果的关键决策。 |
| [^80] | [Inference for Newton Methods with Accelerated Sketch-and-Project via Random Scaling](https://arxiv.org/abs/2609.12421) | 本文提出了一种基于广义加速草图投影（GAS）求解器的在线草图牛顿方法，并通过随机缩放建立了其平均迭代的渐近正态性，证明了其极限协方差通常能比未加速方法更快地收敛到极小极大最优协方差。 |
| [^81] | [MInTRL: Off-policy Intervention can boost On-policy RL](https://arxiv.org/abs/2609.12419) | 提出最小干预强化学习，通过在在线策略采样中进行稀疏的局部干预（用简短修正替换错误输出后缀），在不牺牲可学习性的前提下扩展探索边界，从而利用离线策略干预提升在线策略强化学习的效果。 |
| [^82] | [RiPPLE: Cross-Space Performance Prediction from Early Training for Neural Architecture Search](https://arxiv.org/abs/2609.12418) | RiPPLE通过将少量锚点架构训练至早期阶段、外推其学习曲线生成代理标签并传播到无标签架构特征上，实现了低成本的跨搜索空间神经架构性能预测。 |
| [^83] | [HoliBench: A Cross-Platform Benchmarking and Deployment Toolkit for Foundation Models in CPS-IoT Applications](https://arxiv.org/abs/2609.12412) | HoliBench是一个模块化的跨平台基准测试与部署工具包，能够联合评估基础模型在准确率、延迟和能耗三个维度上的表现，覆盖从单板计算机到GPU服务器的异构设备，为CPS-IoT应用提供统一的部署决策工作流。 |
| [^84] | [Split Conformal Prediction with Label-Shift-Adjusted Bayesian Scores](https://arxiv.org/abs/2609.12386) | 本文提出标签偏移调整贝叶斯分数（LSA score），通过后验预测倾斜恒等式导出非一致性分数，使贝叶斯保形预测在标签偏移下能够保持覆盖率并生成宽度自适应的预测区间。 |
| [^85] | [Membership Inference via Pairwise Likelihood Ratios](https://arxiv.org/abs/2609.12367) | 提出了PL-MIA，一种将高斯似然比统计量、总体校准与柯西组合检验相结合的统一成员推断攻击方法，能够有效汇总模型输出的统计信号并从理论上提升攻击能力。 |
| [^86] | [Certified AI Triage of ICU Alarms](https://arxiv.org/abs/2609.12365) | 本文提出一种带统计认证保证的ICU警报三方分诊方法，以95%置信度保证被抑制警报中真实警报比例不超过用户设定预算，在抑制74.8%误报的同时仅漏掉1.5%真实警报，性能与最强已发表系统相当。 |
| [^87] | [LatentVerse: A Framework for Understanding Shared and Modality-Specific Information in Multimodal Latent Representations](https://arxiv.org/abs/2609.12364) | LatentVerse是一个将Web可视化平台与命令行界面相结合的表示分析框架，它统一了表示质量诊断方法，并通过将嵌入分解为共享成分和模态特定成分来支持多模态潜在表示的分析。 |
| [^88] | [When Connected Does Not Mean Similar: Charting the Homophily Boundary of SNAP-KG for Streaming Entity Integration](https://arxiv.org/abs/2609.12356) | 本文通过将SNAP-KG扩展评估到异质图，揭示其性能的同质性边界：当没有任何同质关系视图可用时聚类质量急剧下降，关键决定因素是关系的同质性而非关系的数量。 |
| [^89] | [ParaRecover: A Process-Level Benchmark for Error Localization and Recovery in Parallel Tool-Use Agents](https://arxiv.org/abs/2609.12345) | 该论文提出了ParaRecover，一个基于14种错误类型分类体系、包含10,626个实例的过程级基准测试，并配套SDE评分标准，用于评估多轮并行工具使用智能体在错误定位与恢复方面的能力。 |
| [^90] | [Theoretical Guarantees for One-Shot Magnitude Pruning and Compute-Adaptive Early Exit](https://arxiv.org/abs/2609.12337) | 该论文为一次性幅度剪枝与计算自适应早退提供了统一的理论保证，证明了带显式速率的剪枝集中定理，并揭示了早退的泛化误差随计算差距呈幂次衰减的标度规律。 |
| [^91] | [Simulating Disengaged Students to Evaluate LLM-based Tutors](https://arxiv.org/abs/2609.12331) | 该论文提出了DAS2（脱离感知学生模拟器），一个可复现的部署前评估协议，通过建模五种学习者参与状态（投入、钻系统空子、无效空转、游离任务和混合状态）来评估基于大语言模型的AI辅导系统在不同学生参与状态下的表现。 |
| [^92] | [LoRA-RC: Reservoir Computing with Low-Rank Adaptation](https://arxiv.org/abs/2609.12327) | LoRA-RC通过流式预测误差驱动的低秩校正在线调整储备池循环矩阵，借助谱范数球投影与低通滤波保证收缩性及与路径无关的增量输入-状态稳定性，解决了静态储备池在系统漂移下性能退化的问题。 |
| [^93] | [Affective Agent: On-Device Personalized Intervention Reasoning for Wearable Systems](https://arxiv.org/abs/2609.12322) | 该论文提出了Affective Agent——一个面向可穿戴设备的三层参考架构，通过将十亿参数以下的小型语言模型与生理证据、上下文及主机管理的结构化记忆演化相结合，在无需云端依赖和用户级重训练的情况下实现了设备端的个性化干预决策推理。 |
| [^94] | [AIM: A Privacy-Aware Interoperable Memory Framework for Multi-Agent Multi-User LLM Systems](https://arxiv.org/abs/2609.12320) | 提出AIM框架，通过动态区分私有与公共记忆并实施索引级访问控制，实现多智能体多用户LLM系统中隐私感知的持久化互操作记忆管理。 |
| [^95] | [Sampling via Decision-Flow: Training-Free Extraction of Improved Latent Reasoning Paths in Large Language Models](https://arxiv.org/abs/2609.12317) | 该论文提出决策流采样（DF-Sample），一种无需训练、无需数据的推理时框架，通过构建分层推理树并进行全局轨迹评估，从大语言模型中提取被标准解码忽视的高质量潜在推理路径，从而无需昂贵的强化学习微调即可提升模型推理能力。 |
| [^96] | [ESTS at WMT26: Routing-Informed Expert Pruning for Model Compression](https://arxiv.org/abs/2609.12310) | 该论文提出利用任务特定路由质量和跨语言路由差异来识别并物理移除GPT-OSS-20B中的低重要性专家，结合恢复微调与MXFP4量化技术，实现了面向中英和阿英翻译任务的高效模型压缩。 |
| [^97] | [Self-Verifying Anomaly Detection using Explainable AI for Cybersecurity of DER Networks](https://arxiv.org/abs/2609.12305) | 本文提出了面向分布式能源网络的ExCYDER框架，通过结合LightGBM与SHAP的可解释人工智能技术构建自验证机制，验证异常检测警报与特征归因的一致性，从而提升网络安全决策的可信度与透明度。 |
| [^98] | [Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models](https://arxiv.org/abs/2609.12303) | 本文首次大规模研究了词元化方案与训练目标（蒸馏 vs. 交叉熵）对约10亿参数模型的影响，提出了两种将词元logits转换为字节logits的方法（Marginalize-It和End-Of-Token），并发现在八个基准测试中词元模型仍优于字节模型。 |
| [^99] | [FRIST: FMRI Representation Informed Shared-space Training Improves EEG-only Individual-Finger BCI Decoding](https://arxiv.org/abs/2609.12298) | 该论文提出FRIST两阶段解码框架，利用fMRI的高空间分辨率信息来指导仅用EEG信号的单指运动解码，无需配对数据即可显著提升BCI的单指解码性能。 |
| [^100] | [Robust Prototypical Networks for Few-Shot Sensor Fault Diagnosis](https://arxiv.org/abs/2609.12287) | 提出多回合原型网络（MEPN），通过聚合多个互不相交支持回合的原型均值来降低原型方差，从而提升少样本传感器故障诊断的鲁棒性。 |
| [^101] | [Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning](https://arxiv.org/abs/2609.12278) | 该论文提出CLAW方法，利用超网络在测试时生成低秩LoRA适配器，以单次前向传播的低计算成本实现对世界模型的高表达性环境适应。 |
| [^102] | [Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models](https://arxiv.org/abs/2609.12277) | 本文提出一种将EHR基础模型视为患者轨迹生成式策略的强化学习微调框架，通过时间感知的奖励设计提升临床推理能力，使小模型在数据受限场景下超越大型预训练模型。 |
| [^103] | [Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning](https://arxiv.org/abs/2609.12264) | 本研究开发了基于深度强化学习的闭环化疗给药策略，并通过100名患者的虚拟队列验证发现：连续动作的TD3策略在肿瘤缩小效果上更优，而离散动作的DQN策略在患者间给药一致性上更强，揭示了化疗控制中的疗效-一致性权衡。 |
| [^104] | [The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition](https://arxiv.org/abs/2609.12259) | 本文通过群组合测试平台提供了因果证据，证明在单状态瓶颈下训练的矩阵记忆会精确招募任务代数结构所要求的秩，从而将秩定律从标量容量界扩展为表示论定律。 |
| [^105] | [CRFCAN: A Complex-Valued Cross-Domain Residual Network for Joint Channel and Phase Noise Estimation in Sub-THz OFDM Systems](https://arxiv.org/abs/2609.12244) | 提出了一种复值跨域残差网络CRFCAN，通过在残差组中嵌入FFT与逆FFT模块实现时频域间的迭代特征交互，以端到端方式实现亚太赫兹OFDM系统中信道与相位噪声的联合估计。 |
| [^106] | [PLSP (Pre-hoc Liminal Space Profiling): OOD Prediction over Detection -- An Anticipatory Approach for Machine Learning Model Reliability](https://arxiv.org/abs/2609.12225) | 本文提出事前预测框架PLSP，将OOD处理范式从检测转向预测，并引入可信度评分（CREDS）、可信度曲线和可信度热力图，实现对机器学习模型应对分布外数据可靠性的事前评估。 |
| [^107] | [Patient-Reported Survey Data Improve Prediction of Opioid Use Disorder](https://arxiv.org/abs/2609.12224) | 该研究的核心创新在于将患者报告的调查数据与电子健康记录相结合，在所有24个模型-回溯窗口组合中均显著提升了阿片类药物使用障碍首次诊断预测的性能，且调查特征被证明是仅次于EHR的第二重要信息来源。 |
| [^108] | [Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion](https://arxiv.org/abs/2609.12223) | 本文提出GRACE模型，通过早期融合的几何残差加合物条件化方法调整预训练分子几何编码器，将加合物感知的残差学习目标与编码器内的加合物条件化机制相结合，显著提升了对气相分子离子碰撞截面的三维预测精度。 |
| [^109] | [QuPAINT: Physics-Aware Multimodal Reasoning for Quantum Material Characterization](https://arxiv.org/abs/2609.12202) | QuPAINT是一个物理感知的多模态框架，通过合成显微图像生成、基于验证标注构建的多模态指令数据集以及物理信息注意力机制，实现对量子二维薄片层数表征的可迁移推理，克服合成到真实的域偏移问题。 |
| [^110] | [GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents](https://arxiv.org/abs/2609.12191) | GAUGE协议揭示了一种常见的LLM智能体离线评估闸门存在严重缺陷——LLM评审给出的满意度评分与实际任务成功率几乎完全不相关（被评为满意的对话中有57.5%实际未能完成客户任务），因此不应仅凭主观评分来信任和选择任务导向型LLM智能体。 |
| [^111] | [Agentic TCAD Calibration Workflow for Oxide Semiconductor Transistors](https://arxiv.org/abs/2609.12184) | 首次提出由LLM智能体驱动的TCAD自动校准工作流，通过残差分析、敏感性测试和器件指标验证，实现了底栅IWO氧化物半导体晶体管的自动化实验校准。 |
| [^112] | [Explanations-Driven Active Feature Acquisition for Algorithmic Recourse](https://arxiv.org/abs/2609.12179) | 该论文首次将算法追索与主动特征获取联合建模，利用马尔可夫毯理论统一反事实、半事实和替代事实解释，并提出按单位成本解释价值选择特征的EDFA方法，实现解释驱动的成本高效特征获取。 |
| [^113] | [Estimating Pedestrian Volumes from GIS-Derived Built-Environment Features: A Machine Learning Framework](https://arxiv.org/abs/2609.12173) | 该研究提出一个基于开放GIS数据的机器学习框架，采用带泊松损失和L1特征选择的直方图梯度提升模型预测城市交叉口高峰行人流量，相比传统负二项GLM基线将交叉验证RMSE降低12%、留出集RMSE降低19%。 |
| [^114] | [Certifying Concept Unlearning in Text-to-Image Diffusion Models](https://arxiv.org/abs/2609.12163) | 该论文提出了一个针对文本到图像扩散模型概念遗忘的认证框架，通过结合统计认证与概念相关嵌入方向上的最坏情况分析，为残余概念泄漏概率提供用户指定置信度下的显式上界，克服了现有基于攻击成功率的经验性评估方法无法量化广泛提示空间中泄漏风险的局限。 |
| [^115] | [Direct Topology Tracking in Continuous Implicit Models](https://arxiv.org/abs/2609.12157) | 提出了一种通过直接查询连续隐式模型及其导数来追踪拓扑特征演化的框架，无需将数据重采样到离散网格，从而在避免离散化伪影的同时实现忠实的特征追踪，并适用于解析函数、多元函数逼近和隐式神经表示等多种隐式模型。 |
| [^116] | [On Identifying Adversarial Intent Injection in AI-Native 6G Networks](https://arxiv.org/abs/2609.12144) | 该论文针对AI原生6G网络中的对抗性意图注入威胁，定义了细粒度威胁模型，研究了四种恶意意图注入策略，并提出了结合CNN与自编码器的双路径检测框架以识别伪装的恶意意图。 |
| [^117] | [GUIDE: Generative Utility Inference and Decision Engine](https://arxiv.org/abs/2609.12137) | GUIDE是一种由大语言模型驱动的偏好引出架构，通过结合贝叶斯自适应采样与符号表示学习，在对话中高效推断用户的多维偏好，并将其建立在领域知识基础之上。 |
| [^118] | [Rank-Efficient LoRA via Joint Tangent-Space Optimization under Isotropic Curvature](https://arxiv.org/abs/2609.12123) | 该论文发现LoRA的标称秩并不等于实际利用的表示能力，优化器会显著影响更新中的有效秩，并据此提出ISO-LoRA——一种通过对权重空间诱导切向扰动进行谱下降来联合耦合LoRA因子更新的优化器，从而更充分、更高效地利用秩容量。 |
| [^119] | [Almost Sure Convergence Analysis of Stochastic Gradient Methods with Clipping and Additive Noise](https://arxiv.org/abs/2609.12119) | 本文证明了带梯度裁剪和加性高斯噪声的SGD在平滑性和一致有界噪声假设及标准步长衰减条件下几乎必然收敛，并将该结果扩展到随机重球法和Nesterov加速梯度等动量变体。 |
| [^120] | [Score-based Outlier Generation via Controlling the Radon-Nikodym Derivative](https://arxiv.org/abs/2609.12113) | 该论文提出通过似然分布的Radon-Nikodym导数对扩散模型分数函数进行受控缩放，从而无需重新训练即可生成幅度可控的低似然离群值。 |
| [^121] | [Language Is an Insufficient Substrate for Quantitative Reasoning, and Consequential Domains Need Large Quantitative Models](https://arxiv.org/abs/2609.12105) | 该论文论证了语言模型因基于人类描述这一有损编码而根本无法满足定量决策对可复现性、数据溯源和校准不确定性的要求，重大定量决策领域需要专门的大型定量模型而非语言模型。 |
| [^122] | [Receiver-Surface Hit Patterns via Legendre Approximation for Molecular Signal Detection](https://arxiv.org/abs/2609.12089) | 该论文提出利用球形接收机表面吸收位置的方向性特征，基于勒让德多项式展开构建分子信号检测器和维特比序列检测器，实现了比仅计数检测更高效、性能更优的几何感知检测。 |
| [^123] | [Hierarchical Prototype Emergence in Modern Hopfield Models](https://arxiv.org/abs/2609.12079) | 本文证明了具有多项式激活函数的稠密Hopfield网络可以在层次结构的每一级稳定存储记忆，并且通过原型重建，仅需准多项式量级的信息即可实现超越具体记忆和层次组别的泛化。 |
| [^124] | [Efficient Vision-Language-Action Management and Serving for Robot Factories](https://arxiv.org/abs/2609.12075) | 提出了首个面向多机器人、多模型请求的VLA服务与管理系统Robion，解决了现有服务系统在SLO约束下无法支持多GPU服务器上多请求多模型执行、且不适用于VLA模型毫秒级阶段的问题。 |
| [^125] | [Scalable Discrete-to-Continuous Channel Simulation for Compression and Privacy](https://arxiv.org/abs/2609.12067) | 提出了一种仅需固定数量随机样本的离散到连续信道仿真方案，其运行时间与信道和输入无关，有效解决了传统信道仿真算法计算成本高和随机停止时间的问题，可应用于压缩与隐私保护场景。 |
| [^126] | [Fast BIB simulation at a future Muon Collider with generative machine learning](https://arxiv.org/abs/2609.12054) | 该论文首次开发了用于缪子对撞机径迹探测器快速束流诱导背景生成的机器学习模型，通过表格扩散模型和环形样条流模型两种架构，以极低的计算成本生成了与完整模拟高度相似的BIB击中和径迹数据。 |
| [^127] | [Learning the Geometry of Collider Events with Metric-Aware Deep Sets](https://arxiv.org/abs/2609.12024) | 本文提出一种度量感知的Deep Sets替代模型，用于快速计算对撞机事件之间的能量移动距离，在实现百分比级别精度的同时大幅提升推理速度，并能显著减少未被显式约束的三角不等式违反。 |
| [^128] | [Reinforcement Learning for Syndrome Extraction](https://arxiv.org/abs/2609.12020) | 本文提出结合强化学习与重要性采样的症状提取自动调度方法，在所有规模上均超越现有最先进工具AlphaSyndrome和PropHunt，逻辑错误率平均分别降低25.9%和71.7%，对距离15的表面码最高降低97.8%。 |
| [^129] | [Toward Reliable Railway-Bogie Response Prediction Using Multifidelity TDNN and Physics-Informed Residual Learning](https://arxiv.org/abs/2609.12018) | 该论文提出了一种多保真度铁路转向架响应预测方法，将多体仿真视为低保真度信息、滚振试验台测量视为高保真度证据，通过时延神经网络与物理信息残差学习相结合，提升转向架响应预测在未经测试工况下的可靠性。 |
| [^130] | [Inverting Self-Triggered Control: Adversarial Reinforcement Learning for Sparse Denial-of-Service Attacks](https://arxiv.org/abs/2609.12016) | 该论文的核心创新是将自触发控制范式进行反转，利用对抗性强化学习学习使闭环失稳的最稀疏拒绝服务攻击调度，并证明了迫使满足Lyapunov合约的自触发控制器崩溃所需最小干扰次数的下界，从而将DoS调度分析从周期和线性时不变系统扩展到自触发控制器。 |
| [^131] | [Certified Safety Curation: Distribution-Free Guarantees for Safe Offline Reinforcement Learning](https://arxiv.org/abs/2609.12014) | 该论文提出“认证安全策展”框架，在仅有片段比较与回合预算查询的弱安全监督下，通过轨迹级筛选与“先学习后测试”校准，为安全离线强化学习的训练集构成提供无分布的安全保证。 |
| [^132] | [QTrans: A Quantum Transformer for Sentiment Classification](https://arxiv.org/abs/2609.12011) | QTrans利用参数化量子电路构建注意力机制中的查询、键、值特征，并结合量子前馈神经网络，建立了端到端可训练的量子-经典混合框架，在多个小规模情感分类数据集上取得了优于基线方法的准确率。 |
| [^133] | [Learning Interaction Kernels from Collective Steady States](https://arxiv.org/abs/2609.12004) | 该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。 |
| [^134] | [Can We Trust LLM Judges: A Study of Capability-Dependent Biases and Multi-Judge Ensemble for Bias Calibration](https://arxiv.org/abs/2609.12002) | 研究发现LLM评审存在与被测模型能力相关的系统性偏差——能力越强的模型越容易获得宽松评判，并提出基于各评审假阳性率和假阴性率在线估计的校准加权多数投票集成方法来校准偏差。 |
| [^135] | [Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale](https://arxiv.org/abs/2609.11998) | 该论文证明状态空间序列混合器配合块因果训练目标，可为块扩散语言模型实现精确的 O(1) 恒定大小缓存，使内存和每步延迟不随上下文长度增长，突破了注意力块缓存 O(L) 的限制。 |
| [^136] | [DCRA: Diffusion-Conditioned Representation Alignment for Robust Time-Series Learning](https://arxiv.org/abs/2609.11997) | 该论文提出DCRA训练框架，将扩散前向过程重新用作结构化损坏调度器，并通过特征级一致性目标在噪声水平间对齐表示且保持类别判别结构，从而在噪声和分布偏移下学习鲁棒的时间序列表示，尤其适用于EEG和ECG等临床信号分析。 |
| [^137] | [Explainable Prediction from Mobile Sensing Data through LLM-guided Concept Integration](https://arxiv.org/abs/2609.11995) | 本文提出了一种利用大语言模型生成概念异常监督信号的概念集成Transformer（CIT），无需人工标注即可对移动感知数据进行可解释的预测，并在精神健康相关数据集上取得了领先的F1分数。 |
| [^138] | [FINESSE: An Agent-Based Simulator and Benchmark Dataset for Multimodal Financial Event Sequences](https://arxiv.org/abs/2609.11993) | 本文提出了FINESSE——一个基于智能体的多模态金融事件序列模拟框架，能够生成由多个相互耦合的事件流（如交易、支付、账户状态变化、政策干预）组成的合成结构化数据集，并发布了FINESSE-Bench基准数据集，以弥补金融领域代表性开源数据集稀缺的问题。 |
| [^139] | [Impact of Multiple Non-Invasive Biosignals on Cardiovascular Biomarker Estimation via Simulation-Based Inference](https://arxiv.org/abs/2609.11969) | 该研究采用基于模拟的推断方法，定量评估了多种无创生物信号（尤其是能反映心脏机械功能的BCG信号）对心血管生物标志物估计的增益作用。 |
| [^140] | [Exact ReLU realization of binary affine refinement iterates via reflection folding and cone switching](https://arxiv.org/abs/2609.11962) | 本文通过反射折叠与固定锥切换机制，证明了任意有限二值仿射细化迭代都能被固定宽度、深度与迭代次数线性相关的ReLU网络精确表示。 |
| [^141] | [On-Device Language Models for Privacy-Preserving Stress Prediction: A Multimodal Evaluation on Mobile Health](https://arxiv.org/abs/2609.11961) | 本研究评估了端侧语言模型在移动设备上多模态压力预测的可行性，发现轻量级模型可实现低延迟的隐私保护推理，且客观传感器特征略优于主观自我报告，展现了该技术在移动心理健康应用中的潜力与限制。 |
| [^142] | [Space as an Interventional Invariant: Cross-Modal Predictive Geometry for Stratified Cities and Em-Spaced Intelligence](https://arxiv.org/abs/2609.11959) | 本文提出将空间定义为“干预不变量”，并构建跨模态预测几何框架，使不共享度量或表示的异构感知与城市数据，仍能在因果干预层面统一揭示共同的空间结构。 |
| [^143] | [Decoding Mixture Perception through Computational Modeling of Component Interactions](https://arxiv.org/abs/2609.11958) | 提出了一种仿生深度学习框架，通过融合注意力加权的多受体神经响应曲线与浓度依赖的多分子曲线，实现了对多分子混合物气味的准确感知识别。 |
| [^144] | [Look Before You Leap: Pre-Action Verification for LLM Agents](https://arxiv.org/abs/2609.11957) | 该论文提出在动作生效前进行低成本确定性验证的统一框架，通过预先构造动作的正确效果来直接度量并拦截LLM智能体的静默失败，其shell命令静态验证器以10.0%的误报率捕获了95.8%的无效命令。 |
| [^145] | [Performance, Efficiency and Collapse -- Advantages and Challenges in Offline Post-training of Code LLMs](https://arxiv.org/abs/2609.11956) | 本研究证明代码大语言模型可以仅利用现有数据集进行完全离线的强化学习后训练，仅需几小时训练即可显著提升零样本代码生成性能，并适用于0.5B至7B参数规模的多种模型。 |
| [^146] | [R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration](https://arxiv.org/abs/2609.11955) | R2VC提出了一种模块化的“检索-推理-验证-校准”事实核查架构，通过混合检索、DPO对齐的生成器、NLI交叉编码器验证以及置信度校准实现证据支撑、带引用和弃权机制的事实核查，在FEVER上使8B模型准确率较基线提升13.74%。 |
| [^147] | [Efficient AI Model Deployment Using Quantization Analysis Tool](https://arxiv.org/abs/2609.11954) | 本文提出了一款基于ONNX框架的量化分析工具，通过逐层敏感性分析和分布可视化功能，帮助开发者在模型大小、延迟和精度之间做出明智权衡，从而实现AI模型在边缘设备上的高效部署。 |
| [^148] | [Scenario-Independent Criticality Assessment and Prediction for Vulnerable Road Users in Autonomous Driving](https://arxiv.org/abs/2609.11947) | 提出了一种专为运动行为难以预测的弱势道路使用者（VRU）量身定制的新型关键性度量，并构建了一个适用于所有交通参与者类别的场景无关关键性预测框架，以克服现有度量依赖特定场景的局限性。 |
| [^149] | [Hyperion: An AI-powered HPC cluster for sciences and humanities research that utilizes ML for predicting job turnaround time](https://arxiv.org/abs/2609.11946) | 南卡罗来纳大学开发了面向科学与人文研究的AI驱动HPC集群Hyperion，并训练机器学习模型预测作业周转时间，将其无缝集成到Slurm作业提交系统中。 |
| [^150] | [PinDCO: Whole-Page Aware Dynamic Creative Optimization at Scale](https://arxiv.org/abs/2609.11943) | Pinterest提出的PinDCO生产级动态创意优化系统，核心创新在于创意组件融合网络（CCFN），通过为图像、标题、布局等各创意组件配置专用塔式网络并融合评分，在严格延迟与成本约束下实现大规模、整页感知的广告创意与受众高效匹配。 |
| [^151] | [The Battery Price of edge AI: A study of the Environmental Impact of LLM Inference on Mobile Devices](https://arxiv.org/abs/2609.11940) | 本研究系统性评估了18个大语言模型在智能手机上的推理能耗、性能与准确性，发现设备端推理的能效平均比服务器低3倍，揭示了本地优先AI范式对设备电池寿命和环境的影响。 |
| [^152] | [Fed-Equilibrium Framework for Topological Pareto Control in Robust and Fair Clinical Federated Learning](https://arxiv.org/abs/2609.11937) | 提出Fed-Equilibrium框架，通过两阶段梯度控制级联（几何余弦相似度过滤与拓扑帕累托控制）解决临床联邦学习中大中心节点压制少数节点导致的“知识主导”问题，实现鲁棒性与公平性的拓扑均衡。 |
| [^153] | [One Simple Trick for Improving the Performance of Energy-Limited Local Inference and Training](https://arxiv.org/abs/2609.11936) | 通过将工作负载切分为更小的块，以更高频率交替进行计算与内存操作，可以平滑功率和温度峰值、避免GPU降频，从而加快运行速度并降低总能耗。 |
| [^154] | [Physics-Informed Conformal Prediction: Embedding PDE Consistency into Distribution-Free Uncertainty Quantification for Neural Operators](https://arxiv.org/abs/2609.11935) | 提出PI-CP框架，将PDE残差嵌入一致性预测的非一致性评分中，实现无分布、空间自适应且具有可证明覆盖保证的不确定性量化，并揭示FNO的平移等变性对Dirichlet边界条件PDE构成根本近似障碍，而坐标通道可克服此障碍并使误差降低高达63倍。 |
| [^155] | [Fundamental Dynamical Units for Physics-Informed Structural Inference from Perturbation Time-Series in Networked Systems](https://arxiv.org/abs/2609.11934) | 该论文提出“基本动力学单元”（FDUs）——以带符号的三节点相互作用模式作为可组合基元，将网络系统的相互作用假设空间转化为有限、构造性且易处理的表示，从而为从扰动时间序列数据中恢复带符号相互作用结构提供了结构性的解决方案。 |
| [^156] | [Towards Sustainable Hydrogen Systems: Supply Chain Optimization with Model Predictive Control and Reinforcement Learning](https://arxiv.org/abs/2609.11933) | 本文针对可再生能源驱动的氢能供应链，比较了基于规则控制、模型预测控制、无预测强化学习和预测增强强化学习四种控制方法，以在动态不确定条件下实现经济性、可靠性与可持续性的平衡。 |
| [^157] | [Musec: MomentUm SpEctral Clipping for Stable Muon-type Training](https://arxiv.org/abs/2609.11655) | 本文提出Musec方法，通过将Muon优化器的谱平坦化替换为谱裁剪——仅裁剪超过阈值的奇异值并保留动量的谱结构——实现了优化器级别、与架构无关的训练稳定化，有效避免损失尖峰和权重无界增长问题。 |
| [^158] | [Beyond Solver Verdicts: Generative Reward Models for Autoformalization](https://arxiv.org/abs/2609.11085) | 本文发现了自动形式化中的“判定保持的不忠实性”（VPU）失败模式，从理论上证明仅依赖求解器判定的验证方法无法有效检测此类错误，并提出生成式验证方法（GenV），将Z3等价性预言机蒸馏为无参照的连续等价性评分以实现可靠的验证。 |
| [^159] | [Are We Really Doing Few-Shot Learning? A Critical Examination of Pre-Training Assumptions](https://arxiv.org/abs/2609.10851) | 本文通过系统比较四种预训练协议，揭示了当前少样本学习评估中因同域预训练带来的 9.66 个百分点乐观偏差，质疑现有评估方式能否真正反映模型的低数据学习能力。 |
| [^160] | [HuRo: Robotizing Human Videos for Scalable VLA Pretraining](https://arxiv.org/abs/2609.10706) | 该论文开发了一套将异构人类视频转换为机器人对齐的观察与动作轨迹的机器人化流水线，并据此构建了包含约63万条机器人化片段和1.42亿帧的HuRo数据集，验证了机器人化人类视频能够为VLA策略预训练提供有效且可扩展的监督信号。 |
| [^161] | [Byzantine-Robust Federated Fire Detection with a Rotating Coordinator](https://arxiv.org/abs/2609.10647) | 本文提出了一种结合历史感知聚合与轮换协调者的拜占庭鲁棒半去中心化联邦学习方法，用于室内火灾检测，通过构建整理的火灾数据集、实现边缘端模型更新高达10倍的压缩，同时解决了上行带宽受限、恶意客户端以及固定服务器单点故障三大实际问题。 |
| [^162] | [Code-to-Harness: Distilling Black-Box Optimizers from Self-Play](https://arxiv.org/abs/2609.09468) | 本文提出“从代码到执行框架”方法：让语言模型代理通过反复编写和评估优化器程序进行自我练习，将学到的搜索策略一次性蒸馏成一份冻结的197词文本框架，该框架能显著降低多个语言模型执行器在低预算黑盒优化中的遗憾值，并可跨模型迁移。 |
| [^163] | [SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?](https://arxiv.org/abs/2609.09113) | 本文提出SAEScientist-Bench基准，评估AI智能体能否作为科学家自主运用稀疏自编码器（SAE）工具，进行机制可解释性的自主科学发现研究。 |
| [^164] | [Transformers as In-Context Samplers: From Closed-Form Diffusion to Estimation-Free Sampling](https://arxiv.org/abs/2609.08981) | 本文证明冻结的Transformer可以通过上下文学习模拟迭代式生成采样器（如闭式扩散采样器），其中softmax注意力计算责任权重与加权经验均值、前馈层实现欧拉更新，从而将上下文学习能力从监督学习拓展到数据生成任务。 |
| [^165] | [The BatchNorm Illusion: Diagnosing Normalization Artifacts in Machine Unlearning Evaluation](https://arxiv.org/abs/2609.08901) | 该论文揭示了一个评估陷阱：在基于BatchNorm的架构上，仅对保留数据做一次前向传播（不修改任何权重）就能重写模型的归一化状态并逆转表面遗忘指标，作者将其形式化为权重保持的定点算子，严格证明此类前后差异源于BN运行统计量而非遗忘方法本身，从而清晰区分了度量失效与真实的权重编码信息残留。 |
| [^166] | [Norms at a Price: Why RL-Based Alignment Can Promise Conditional Compliance at Best](https://arxiv.org/abs/2609.07627) | 论文论证了基于强化学习的对齐因从打分行为中学习规范，使“合规”退化为“被发现才付出代价”，因此行为训练在原理上至多只能保证智能体在被观察时才合规的有条件合规。 |
| [^167] | [LatentMD: Benchmarking Markdown Boundary Failures in LLM-Generated Text](https://arxiv.org/abs/2609.06993) | LatentMD是一个将内容正确性与边界正确性分离的基准测试，它揭示了LLM生成的Markdown中边界失败问题普遍存在——38%的内容正确输出实际上边界已损坏。 |
| [^168] | [Reason Through the Latent! Making Latent Visual Reasoning Necessary](https://arxiv.org/abs/2609.06746) | 提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。 |
| [^169] | [Decomposing LLM-Judge Uncertainty to Target Expert Labels](https://arxiv.org/abs/2609.06444) | 该论文提出一种小型贝叶斯模型，将LLM评审器的总不确定性分解为可被专家标注消除的认知不确定性和不可消除的偶然不确定性，使专家只需标注评审器真正无知的项目，在ChaosNLI数据集上比使用总不确定性多消除83%的误差。 |
| [^170] | [Evidence-Aligned Local Composition of Discrete Experts for Sequence Restoration](https://arxiv.org/abs/2609.05801) | 提出证据对齐的局部组合方法，在测试时无需区域标签或训练好的路由器，通过被破坏观测的边际证据（由专家自身去噪损失估计）推断逐位置的软性专家权重，实现混合领域文档的序列恢复。 |
| [^171] | [Distilled Continuous Diffusion Language Models Can Write Code in Few Steps---or One](https://arxiv.org/abs/2609.04531) | 本文提出0.7B参数的连续扩散代码生成模型PlaidQ，通过分布匹配和成对轨迹监督的蒸馏技术，将去噪轨迹压缩到仅需16步甚至1步即可高效生成代码，性能可与离散扩散语言模型媲美。 |
| [^172] | [EF1-Constrained Nash Social Welfare with Identical Additive Valuations: Complexity, Guarantees, and Experiments](https://arxiv.org/abs/2609.03846) | 该论文证明在相同可加估值下EF1约束的NSW最大化问题是强NP完全的，并识别出获得更强福利保证的条件——均匀估值下每个EF1分配都是NSW最优的，而在ε-小物品条件下每个EF1分配能达到 1-O(ε²) 的显式近似比。 |
| [^173] | [Free Pause Tokens](https://arxiv.org/abs/2609.03807) | 提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。 |
| [^174] | [Landmark-Based Discrimination of Injury-Associated Athlete-Sessions from Minute-Resolution Multimodal Football Monitoring Data](https://arxiv.org/abs/2609.03790) | 本文提出一种基于固定地标点的建模方法，在每个地标时刻（如10、20、30分钟）利用截至该时刻的观测信息为每个运动员-场次构建单一表示，从而在损伤信息仅有场次级标签的情况下，避免不合理的分钟级损伤监督，实现从分钟级多模态足球监测数据中判别损伤相关场次。 |
| [^175] | [Improving precipitation forecasts in an AI weather model using observational data](https://arxiv.org/abs/2609.03210) | 通过使用观测的IMERG降水数据微调图变换器AI天气预报模型，将中期降水预报的CRPS提升最高19%，并使全球极端降雨预测的Brier技巧评分超越最先进业务化模型57%。 |
| [^176] | [SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers](https://arxiv.org/abs/2609.01343) | SMELT在严格匹配计算量、参数量和KV缓存的条件下将MoE Transformer的中间层循环两次，其损失下降更快，可在计算最优前沿节省6.8%–18.0%的训练FLOPs，且在代码任务和长样本上优势更明显。 |
| [^177] | [TDDM-Melatt: A Decoupled Memory and Diffusion Framework for Generalizable Encrypted Traffic Classification](https://arxiv.org/abs/2608.30745) | 提出TDDM-Melatt框架，通过记忆解耦的流量表示模型Melatt与基于扩散模型的数据增强相结合，解决加密流量分类中的捷径学习和样本不平衡问题，显著提升对真实网络流量的泛化能力。 |
| [^178] | [Denoising as Projection: Constrained Optimization with Gradient-Guided Diffusion](https://arxiv.org/abs/2608.29507) | 本文提出将Stein去噪算子视为向数据几何的近似投影，通过在去噪步骤中融入目标梯度，实现了一种仅依赖预训练去噪器即可在约束集上进行约束优化的推理时扩散方法。 |
| [^179] | [AdaVLA: Adaptive Step Flow Matching for Training-free Acceleration of Vision-Language-Action Models](https://arxiv.org/abs/2608.29208) | 提出AdaVLA框架，通过自适应调整流匹配推理中的ODE求解步数，在无需微调或训练数据的情况下，实现视觉-语言-动作模型的免训练在线加速。 |
| [^180] | [VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning](https://arxiv.org/abs/2608.26105) | VBVR-Pro通过提供300个程序化任务和可验证奖励评分器，构建了一个闭环测试平台，使原生视觉推理可训练、可验证和可优化，并在多个外部基准上展现出强迁移能力。 |
| [^181] | [Improving Energy Efficiency of Oil Platforms Through Optimal Loading of Diesel Generators Using Machine Learning and Search Algorithms](https://arxiv.org/abs/2608.22076) | 本研究首次将机器学习与搜索算法结合，针对海上石油平台的柴油发电机负载进行优化，以降低能源消耗而非提升产量，填补了该领域研究空白。 |
| [^182] | [Multi-Source Wasserstein Distributionally Robust Graph Learning](https://arxiv.org/abs/2608.19914) | 本文提出MS-WDRO框架，利用Wasserstein重心融合异构图信号源并构建鲁棒模糊球，以应对源间差异和不确定性，实现稳健的图拓扑推断。 |
| [^183] | [GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction](https://arxiv.org/abs/2608.18234) | 本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。 |
| [^184] | [UniTAC: Universal Task-Aware Compression via Weighted Distortion Measures](https://arxiv.org/abs/2608.16696) | UniTAC提出了一种单一学习图像编解码器，通过运行时注入任务重要性向量实现从通用到任务专用的灵活压缩，无需重新训练。 |
| [^185] | [Learning Generalizable Reconstruction of High-Dimensional Neural Dynamics](https://arxiv.org/abs/2608.16569) | 本文提出PCA-DMD框架，通过结合PCA降维和Koopman算子学习，实现了高维神经动力学的可泛化重建，并在跨受试者零样本泛化中展现出卓越性能。 |
| [^186] | [ETHOS: Towards a Modular Ethics Framework for Clinical Multi-Agent Systems](https://arxiv.org/abs/2608.15424) | ETHOS提出了一种模块化伦理治理元智能体框架，可直接集成到现有临床多智能体系统中，在不改变底层架构的前提下解决安全性、公平性、问责性和透明度等关键伦理问题。 |
| [^187] | [Diffract: Spectral View of LLM Domain Adaptation](https://arxiv.org/abs/2608.10850) | 本文通过谱分析揭示持续预训练主要靠改变奇异向量而非奇异值谱来实现大语言模型的领域适配，并提出基于注意力头重要性的选择性回退方法可将基准准确率提升至多4%，同时发现CPT检查点间存在可平滑插值的领域连通性。 |
| [^188] | [From Information to Delegation: Mapping Human-AI Financial Decision Making](https://arxiv.org/abs/2608.02100) | 该研究通过分析150万次真实ChatGPT和Gemini交互，提出结合意图与委托决策权的行为测量框架，发现消费者主要使用AI获取金融信息并形成判断，但极少将金融执行权委托给AI。 |
| [^189] | [Slot2Text: Object-Centric Visual Tokenization for Efficient and Spatially Traceable Surgical MLLMs](https://arxiv.org/abs/2608.01473) | Slot2Text提出了一种以对象为中心的双模式手术多模态大语言模型，通过将密集视觉特征编码为少量带区域标签的槽位标记作为视觉词元，在大幅降低推理成本的同时实现了答案的空间可追溯性。 |
| [^190] | [A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense](https://arxiv.org/abs/2608.00583) | 仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。 |
| [^191] | [DASH-OPD: Discrepancy-Aware Switching with Hysteresis for On-Policy Distillation](https://arxiv.org/abs/2607.29078) | 本文提出DASH-OPD，一种基于滞回和差异感知切换的在线策略蒸馏方法，能双向自适应地在学生和教师执行器间切换，以解决多轮场景中因学生错误导致的轨迹偏离问题。 |
| [^192] | [When Low CER is Not Enough: An Analysis of Hallucinations in Vision-Language OCR Systems on Historical Uruguayan Documents](https://arxiv.org/abs/2607.24077) | 该研究通过分析乌拉圭独裁时期历史文献，揭示视觉语言OCR模型虽然在字符错误率上优于传统方法，但存在标准指标无法检测的幻觉问题（如拼写规范化、虚假内容生成和语义替换），表明仅凭低CER不足以评估其在档案转录中的可靠性。 |
| [^193] | [The C-index illusion: discrimination without calibration in published survival models](https://arxiv.org/abs/2607.19526) | 该研究通过对三个真实已发表的生存机器学习模型的复现分析，首次实证证明仅依赖C指数等判别指标会系统性误导模型评估，揭示了即使判别能力接近完美（C≈0.96）的模型也可能存在严重的校准缺陷。 |
| [^194] | [Representing and Detecting Label Ambiguity in IMU-Based Exercise Evaluation](https://arxiv.org/abs/2607.04842) | 本文提出一种无需大型评分者群体即可自动生成IMU运动评估中每次动作重复标签分布的方法，通过KL散度目标训练网络来表示和检测标签模糊性，其性能达到或超越传统独热标签基线。 |
| [^195] | [Diffusion learning reveals viable parameter manifolds and compensation geometry in biological dynamical systems](https://arxiv.org/abs/2607.03671) | 本文提出“可行参数流形”概念，将能产生相同动力学行为的兼容参数集合定义为参数到特征映射的目标特征原像，并利用条件得分扩散模型作为摊销采样器，从少量实验观测特征中高效反向推断可行参数集合，揭示了参数补偿的几何结构及其可学习性条件。 |
| [^196] | [State-Specific Respiratory Signatures for Affective and Stress Recognition: Interpretable Respiratory Markers, Autocorrelation Lags, and Compact CNN Models](https://arxiv.org/abs/2606.26723) | 本研究通过结合紧凑型一维CNN和手工设计的呼吸特征，不仅实现了压力与非压力的高精度二分类检测，还识别出了基线、压力、愉悦和冥想状态各自特异的可解释呼吸标记。 |
| [^197] | [OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents](https://arxiv.org/abs/2606.26350) | OpenFinGym提出了一个统一的多任务Gym环境，通过覆盖预测、市场生成、实时交易和欺诈检测等关键任务，并配备自动化任务构建流程，解决了现有平台因单任务评估而导致的量化金融智能体能力误判和泛化性缺失问题。 |
| [^198] | [UltraQuant: 4-bit KV Caching for Context-Heavy Agents](https://arxiv.org/abs/2606.20474) | 该论文提出UltraQuant，一种面向多轮智能体工作负载的4比特KV缓存方案，通过TurboQuant风格旋转量化、非对称K/V处理及AMD GPU上的FP4近似推理路径，实现任务质量、缓存驻留率与服务吞吐量的联合优化。 |
| [^199] | [On the Residual Scaling of Looped Transformers: Stability and Transferability](https://arxiv.org/abs/2606.18524) | 该论文证明循环Transformer因权重共享需要采用比传统1/√L更强的1/N残差缩放，并提出分解参数化ε = λ/(N√L)，使最优学习率仅取决于独特层数L而与循环次数N无关，从而实现超参数的直接迁移。 |
| [^200] | [GRACE-DS: a Guarded Reward-guided Agent Correction Environment in Data Science](https://arxiv.org/abs/2606.16000) | GRACE-DS是一个用于LLM驱动AutoML智能体部署前评估的隔离环境，通过隐藏的可执行验证器从预测性能、泄漏规避、可复现性等多维度进行评估，其中灵活迭代交互机制的表现优于单次生成等基线方法。 |
| [^201] | [SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization](https://arxiv.org/abs/2606.08496) | 提出SAEExplainer训练框架，以激活分数作为客观奖励信号，通过两轮迭代优化实现模型自我纠正与持续改进，显著减少SAE特征解释中的幻觉并强化因果触发模式。 |
| [^202] | [CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval](https://arxiv.org/abs/2605.29271) | 提出CoHyDE方法，通过迭代协同训练将LLM查询重写器与稠密编码器作为一个共同演化的整体系统，融合编码器微调与HyDE查询扩展二者的优势，显著提升LLM智能体在大型API目录上的工具检索性能。 |
| [^203] | [Musical Attention Transformer: Music Generation Using a Music-Specific Attention Model](https://arxiv.org/abs/2605.21081) | 提出音乐注意力机制，将小节编号、调性、拍号和速度等元信息融入Transformer的注意力过程，有效减少生成音乐中的音符重复问题，提升音乐生成质量。 |
| [^204] | [The General Theory of Localization Methods](https://arxiv.org/abs/2605.20635) | 本文提出了一个以局部化核与局部均值为核心概念的通用机器学习理论框架——局部化方法，通过两个理论支柱（局部化模型构建与局部化技巧）统一解释了核方法、自注意力机制、Hopfield网络等多种现有机器学习模型背后的共同原理。 |
| [^205] | [Class-wise Contribution Estimation via Logit Maximization for Federated Learning](https://arxiv.org/abs/2605.18892) | 提出了一种基于logit最大化的免数据联邦学习聚合框架CELM，通过构建跨客户端证据矩阵量化各类别能力与覆盖度，为少数类提供强证据的客户端赋予更高聚合权重，从而有效缓解联邦学习中的类别不平衡和标签偏斜问题。 |
| [^206] | [DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models](https://arxiv.org/abs/2605.15055) | 提出基于在线策略蒸馏（OPD）的 DiffusionOPD 多任务训练范式，先独立训练任务教师，再沿学生自身 rollout 轨迹将其能力蒸馏到统一学生模型，从而解决扩散模型多任务强化学习中的跨任务干扰与灾难性遗忘问题。 |
| [^207] | [DenseTRF: Texture-Aware Unsupervised Representation Adaptation for Surgical Scene Dense Prediction](https://arxiv.org/abs/2605.11265) | DenseTRF提出了一种基于纹理中心注意力的自监督表征自适应框架，通过槽注意力学习纹理感知表征并无监督地适应目标分布，显著提升了手术场景密集预测任务对域偏移的鲁棒性和跨分布泛化能力。 |
| [^208] | [Independent Learning of Nash Equilibria in Partially Observable Markov Potential Games with Decoupled Dynamics](https://arxiv.org/abs/2605.06377) | 该论文针对具有独立状态转移的部分可观测马尔可夫势博弈，提出了一种无需中心化或通信的独立学习算法，使各智能体仅凭自身动作和观测即可联合收敛到近似纳什均衡，从而避免了先前方法随玩家数量指数级增长的复杂度。 |
| [^209] | [Adapt or Forget: Provable Tradeoffs Between Adam and SGD in Nonstationary Optimization](https://arxiv.org/abs/2605.04269) | 本文首次对Adam在非平稳随机优化中的表现给出理论分析，将有限时间界清晰分解为初始化、目标漂移、一阶矩跟踪误差和预条件器扰动四个分量，并揭示了噪声与漂移之间可证明的权衡关系。 |
| [^210] | [Anomaly-Preference Image Generation](https://arxiv.org/abs/2605.02439) | 本文提出异常偏好优化范式，通过利用真实异常作为正样本参考的隐式偏好对齐机制和沿扩散时间线动态分配模型容量的时间感知容量分配模块，在不依赖人工标注的情况下，有效平衡了有限数据下异常样本生成中的保真度与多样性。 |
| [^211] | [Accelerating battery research with an interoperable interface between FINALES and Kadi4Mat](https://arxiv.org/abs/2605.00909) | 本研究提出了一个集成FINALES与Kadi数据管理生态系统的可互操作框架，实现了跨分布式研究基础设施的自动化、可重现的端到端电池实验工作流程，并以钠离子电池化成工艺为案例展示了其加速能源材料研究的能力。 |
| [^212] | [Representation Before Training: A Practical Benchmark for Generative Medical Event Model Tokenization](https://arxiv.org/abs/2604.16775) | 该研究对生成式医疗事件模型的分词方案进行了大规模系统性基准测试（156个模型），发现将医疗代码与检验数值十分位数融合的 token 表示在所有八类住院结局预测任务中均能带来性能提升。 |
| [^213] | [SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics](https://arxiv.org/abs/2604.16358) | 提出渐进式多轮安全对齐框架SaFeR-Steer，通过分阶段合成数据引导与导师介入的GRPO强化学习，并引入轨迹一致求和奖励（TCSR）机制，同时发布多轮多模态安全数据集STEER，以弥合多轮对话场景下多模态大模型训练与部署之间的安全对齐差距。 |
| [^214] | [Exploring Urban Land Use Patterns by Pattern Mining and Unsupervised Learning](https://arxiv.org/abs/2604.13050) | 该论文提出了一种结合频繁项集模式挖掘与Ward聚类的可重复分析框架，从欧洲100个城市地区的遥感数据中识别出七类跨城市重复出现的土地利用配置模式，为比较性规划中的同类城市对比提供了方法支持。 |
| [^215] | [Are Independently Estimated View Uncertainties Comparable? Unified Routing for Trusted Multi-View Classification](https://arxiv.org/abs/2604.09288) | 该论文提出TMUR方法，解决了可信多视图分类中各视图独立估计的不确定性因分支尺度偏差而不可比的问题，通过统一路由机制解耦视图证据提取，使融合所用的不确定性真正反映样本级可靠性。 |
| [^216] | [Dead Weights, Live Signals: Feedforward Graphs of Frozen Language Models](https://arxiv.org/abs/2604.08335) | 该论文提出一种前馈图架构，将多个异构冻结大语言模型作为计算节点，通过可学习的线性投影在共享潜在空间中通信，仅用1760万可训练参数即在ARC-Challenge上达到87.3%的准确率。 |
| [^217] | [An Empirical Markov Chain Car-Following (MC-CF) Model](https://arxiv.org/abs/2603.27909) | 本文提出了一种无参数化假设的马尔可夫链跟驰模型（MC-CF），通过从经验分布中随机采样加速度来捕捉自然驾驶的随机性，在Waymo数据集上显著优于传统物理模型并与数据驱动方法相当。 |
| [^218] | [Decomposing Discrimination: Causal Mediation Analysis for AI-Driven Credit Decisions](https://arxiv.org/abs/2603.27510) | 该论文提出将AI信贷决策中的歧视通过因果中介分析分解为直接歧视与结构性不平等两种机制，并在处理诱导混淆下识别出干预性直接/间接效应，为不可识别的自然歧视效应提供保守界。 |
| [^219] | [FEAT: A Linear-Complexity Foundation Model for Extremely Large Structured Data](https://arxiv.org/abs/2603.16513) | 提出了FEAT——一种面向超大规模结构化数据的线性复杂度基础模型，它克服了自注意力的O(N^2)计算瓶颈，在保持置换不变性的同时提升了对真实世界数据库的泛化能力。 |
| [^220] | [Tunable Latent Generative Priors for Compressed Sensing and Inverse Problems](https://arxiv.org/abs/2603.07357) | 本文提出基于嵌套丢弃技术的可调潜变量生成先验，使扩散模型、归一化流和变分自编码器能够灵活调整潜变量维度，在压缩感知、图像修复、去噪和相位恢复等逆问题中持续获得比固定复杂度基线更低的重构误差，并在线性去噪场景下闭式推导出最优复杂度。 |
| [^221] | [Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR](https://arxiv.org/abs/2603.07084) | 本文提出Countdown-Code测试平台，通过代理奖励与真实奖励的清晰分离来精确测量RLVR中的奖励破解现象，并发现SFT数据中仅1%的奖励破解轨迹污染就足以让模型无意中习得这种失对齐行为。 |
| [^222] | [The Vienna 4G/5G Drive-Test Dataset](https://arxiv.org/abs/2603.02638) | 本文发布了维也纳4G/5G路测数据集，这是一个结合LTE/5G NR测量数据、基站部署描述信息以及高分辨率建筑物与地形模型的城市规模开放数据集，为移动网络分析、规划与优化的机器学习研究提供数据支持。 |
| [^223] | [A unified self-supervised framework for single-frame Fresnel CDI and overlapped ptychography](https://arxiv.org/abs/2602.21361) | 该论文提出了一种统一的自监督逆映射神经网络框架，利用离焦弯曲波前探针提供的相位多样性，同时支持单帧菲涅尔相干衍射成像和重叠叠层成像的重建，摆脱了重叠约束，实现更稀疏的扫描和更低的辐射剂量。 |
| [^224] | [Block-Norm Geometries for Online Mirror Descent with Sparse Losses](https://arxiv.org/abs/2602.13177) | 该论文提出一族随机化块范数镜像映射，在欧氏与熵几何之间插值以适应稀疏损失梯度，并对多种标准凸集证明了随维度多项式级改进的 regret 界。 |
| [^225] | [Machine Learning-Based Classification of Jhana Advanced Concentrative Absorption Meditation Using 7 Tesla Functional Magnetic Resonance Imaging](https://arxiv.org/abs/2602.13008) | 该研究的核心创新在于首次利用7T fMRI衍生的区域一致性特征与机器学习方法，在个体水平上实现对禅那高级专注入定冥想状态的分类，突破了以往仅依赖组水平单变量对比分析的局限。 |
| [^226] | [In-Hospital Stroke Risk-State Classification from PPG-Derived Hemodynamic Features](https://arxiv.org/abs/2602.09328) | 该研究通过LLM辅助流程从非结构化临床笔记中提取经医生裁定的中风锚点，并利用PPG衍生的血流动力学特征和ResNet-1D分类器，实现了院内中风风险状态的准确分类。 |
| [^227] | [Benford's Law as a Distributional Prior for Post-Training Quantization of Large Language Models](https://arxiv.org/abs/2602.00165) | 该论文观察到Transformer的线性权重符合本福特分布而LayerNorm参数系统偏离，据此提出免数据的量化码本BenQ，用对数间隔网格作为码本，选择性地应用于变换层，在4位训练后量化中一致优于均匀RTN。 |
| [^228] | [LLM Compression by Block Removal with Constrained Binary Optimization](https://arxiv.org/abs/2602.00161) | 该论文将大语言模型的块删除压缩问题形式化为约束二值优化问题并映射到Ising自旋玻璃物理系统，利用系统能量作为下游模型性能的代理指标，在深度压缩场景下（如50%压缩Llama-3.3-70B）相比现有最先进方法在MMLU基准上提升近23个百分点。 |
| [^229] | [Multi-Modal Time Series Prediction via Mixture of Modulated Experts](https://arxiv.org/abs/2601.21547) | 提出了一种名为“专家调制”的新机制，使混合专家模型的路由与专家计算均受文本信号调制，从而摆脱对token级融合的依赖，提升多模态时间序列预测的准确性与跨模态对齐能力。 |
| [^230] | [LLAMA LIMA: A Living Meta-Analysis on the Effects of Generative AI on Learning Mathematics](https://arxiv.org/abs/2601.18685) | 该论文提出一种每两个月更新文献库的“活态元分析”方法，持续追踪生成式AI干预对数学学习的效果，最新结果显示其具有日益显著的积极作用（g = 0.57），且当AI辅助教师教学、干预时间较长时效果更佳。 |
| [^231] | [AnyView: Synthesizing Any Novel View in Dynamic Scenes](https://arxiv.org/abs/2601.16982) | 提出了AnyView，一个基于扩散模型、几乎无几何假设的动态视角合成框架，通过融合2D单目、3D静态多视角和4D动态多视角等多源数据训练通用时空隐式表示，实现从任意相机位置和轨迹零样本生成新视角视频，并发布了针对极端动态场景的新基准AnyViewBench。 |
| [^232] | [Trainability-Oriented Hybrid Quantum Regression via Geometric Preconditioning and Curriculum Optimization](https://arxiv.org/abs/2601.11942) | 该论文提出一种混合量子-经典回归框架，通过可学习的经典几何预处理器重塑输入表示，并结合逐步增加电路深度、从SPSA随机探索过渡到Adam梯度微调的课程优化协议，有效提升了量子神经网络在回归任务中的可训练性。 |
| [^233] | [PACEvolve: Enabling Progress-Aware Consistent Evolution](https://arxiv.org/abs/2601.10657) | PACEvolve提出了一个进度感知的一致性进化框架，通过分层上下文管理等三项关键技术来治理智能体的记忆与搜索动态，从而解决自进化LLM智能体中的上下文污染和模式坍塌问题。 |
| [^234] | [Deep learning methods for inverse problems using connections between proximal operators and Hamilton-Jacobi equations](https://arxiv.org/abs/2512.23829) | 本文提出利用近端算子与哈密顿-雅可比偏微分方程之间的深刻联系，构建新的深度学习架构来学习逆问题中的先验信息，从而为求解病态逆问题提供更有效的深度学习方法。 |
| [^235] | [A Dataset and Benchmarks for Atrial Fibrillation Detection from Electrocardiograms of Intensive Care Unit Patients](https://arxiv.org/abs/2512.18031) | 该研究发布了带标注的ICU心电图心房颤动检测数据集与基准，并系统比较了基于特征的分类器、深度学习和心电图基础模型三种AI方法，发现心电图基础模型的检测性能最佳。 |
| [^236] | [Classical and quantum kernel fusion for two-sample testing](https://arxiv.org/abs/2511.20941) | 本文提出了一种融合经典核与量子核的混合双样本检验策略，通过增强MMD-FUSE框架，使得检验即使在小数据集上也能有效进行。 |
| [^237] | [Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference](https://arxiv.org/abs/2511.15015) | 提出DynaExq系统，将HBM受限的单GPU混合专家模型推理转化为在线预算约束的动态精度分配问题，根据专家的运行时流量动态分配量化位宽，以实现高效的单GPU部署。 |
| [^238] | [When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs](https://arxiv.org/abs/2511.07318) | 该论文揭示了一类由训练数据中虚假关联（如姓氏与国籍的关联）驱动的幻觉，这类幻觉被模型自信生成、不受模型规模扩大影响、能规避现有检测方法、且在拒绝微调后仍持续存在，导致基于置信度过滤和内部状态探测等主流幻觉检测方法从根本上失效。 |
| [^239] | [Bias-Corrected Data Synthesis for Imbalanced Learning](https://arxiv.org/abs/2510.26046) | 该论文提出了一种偏差校正的数据合成方法，通过从留出的多数类数据估计生成器引起的损失偏差并将其转移至少数类，为不平衡学习建立了有限样本理论保证，同时刻画了SMOTE产生显著损失偏差的条件。 |
| [^240] | [Hurdle-RMIL: Addressing Zero Inflation and Long-Tailed Imbalance in Infrared Rainfall Retrieval](https://arxiv.org/abs/2510.20486) | 该论文提出Hurdle-RMIL方法，将零膨胀与长尾不平衡分而治之，利用障碍模型处理零值膨胀，并通过基于贝叶斯的分布变换直接从自然长尾样本中学习平衡分布模型，有效缓解了卫星红外降雨反演中对高强度降雨的系统性低估问题。 |
| [^241] | [Nonlinear Dimensionality Reduction Techniques for Bayesian Optimization](https://arxiv.org/abs/2510.15435) | 该论文提出将变分自编码器、深度度量损失、再训练策略与直接在潜空间进行的序贯域收缩（SDR-LSBO）相结合，有效提升了高维昂贵黑盒函数全局优化的性能。 |
| [^242] | [Limits of LLM Text Detectors in Education](https://arxiv.org/abs/2508.08096) | 本文提出了一个贡献感知的LLM文本检测评估框架，通过八个学生贡献等级模拟真实写作场景，揭示了当前基于二元区分的LLM检测器在教育评估中的局限性。 |
| [^243] | [AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training](https://arxiv.org/abs/2507.01663) | AsyncFlow是一个面向高效LLM后训练的异步流式强化学习框架，其核心创新在于分布式数据存储与传输模块，以全流式方式实现全景数据管理、细粒度调度、自动化流水线重叠和动态负载均衡，解决了传统RL框架的可扩展性瓶颈与资源闲置问题。 |
| [^244] | [On Universality of Non-Separable Approximate Message Passing Algorithms](https://arxiv.org/abs/2506.23010) | 本文提出了表示张量的有界组合性质（BCP）这一一般性条件，首次将非可分离近似消息传递（AMP）算法状态演化的普适性从高斯数据推广到非高斯数据。 |
| [^245] | [Fast Convergence for High-Order ODE Solvers in Diffusion Probabilistic Models](https://arxiv.org/abs/2506.13061) | 本文针对具有任意方差调度的一般前向过程，在得分函数导数的实际假设下，对基于概率流ODE的p阶（指数）Runge-Kutta采样格式进行了严格的收敛性分析，证明了高阶ODE求解器在扩散概率模型中的快速收敛性。 |
| [^246] | [Aligning Language Models with Observational Data: Opportunities and Risks from a Causal Perspective](https://arxiv.org/abs/2506.00152) | 本文从因果视角系统分析了利用历史观测数据微调大语言模型的机遇与风险，指出观测结果可作为A/B测试的低成本替代监督信号，但同时需警惕因果混淆带来的偏差。 |
| [^247] | [Surrogate Modeling of 3D Rayleigh-Benard Convection with Equivariant Autoencoders](https://arxiv.org/abs/2505.13569) | 提出了一个由等变卷积自编码器和等变卷积LSTM组成的端到端等变代理模型，利用系统的E(2)-等变对称性对三维瑞利-贝纳德对流进行高效建模，从而提升精度与样本效率。 |
| [^248] | [Scalable Krylov Subspace Methods for Generalized Mixed-Effects Models with Crossed Random Effects](https://arxiv.org/abs/2505.09552) | 该论文提出了基于Krylov子空间的新方法，解决了广义混合效应模型中高维交叉随机效应导致的计算瓶颈，在保持同等精度的同时实现了几个数量级的加速。 |
| [^249] | [GLaMoR: Consistency Checking of OWL Ontologies using Graph Language Models](https://arxiv.org/abs/2504.19023) | 本文提出GLaMoR推理管道，通过将OWL本体转换为图结构数据并利用图语言模型（GLM）架构进行一致性检查，解决了现有推理机计算成本高、效率随本体规模增长而下降，以及经典机器学习方法无法考虑T-Box的问题。 |
| [^250] | [Can SGD Select Good Fishermen? Local Convergence under Self-Selection Biases](https://arxiv.org/abs/2504.07133) | 本文提出了首个针对自选择偏差的局部收敛算法，通过将自选择问题归约为粗化估计问题，给出了运行时间为poly(d, k, 1/ε) + (k log k)^{O(k)}的更快算法，从而解决了CDIZ23提出的主要开放问题之一。 |
| [^251] | [A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super-Gaussian Likelihoods](https://arxiv.org/abs/2504.05431) | 本文提出了一种基于广义切变换的变分推断框架，利用凸对偶性构建对数似然的切下界，使强超高斯似然类概率模型与高斯先验实现共轭，从而将该结构化变分方法的应用范围从逻辑回归扩展到更广泛的模型类别。 |
| [^252] | [Safe Learning Under Irreversible Dynamics via Asking for Help](https://arxiv.org/abs/2502.14043) | 本文首次正式证明，通过允许智能体向导师求助并在相似状态间迁移知识，智能体可以在具有不可逆动力学的无限状态空间未知高风险环境中，以次线性遗憾和次线性求助次数安全高效地学习并获得高回报，无需依赖重置机制。 |
| [^253] | [Scaling Online Complex Event Detection with Synthetic Supervision and Mamba-Based Neural Algorithmic Reasoning](https://arxiv.org/abs/2502.07250) | 该论文提出一种结合合成监督与基于Mamba架构的神经算法推理框架，通过将复杂事件规则学习与执行解耦，克服了长时依赖、稀疏监督和标注困难等挑战，实现了在线流式场景下可扩展的复杂事件检测。 |
| [^254] | [Statistical Uncertainty Quantification for Aggregate Performance Metrics in Machine Learning Benchmarks](https://arxiv.org/abs/2501.04234) | 该论文展示了如何利用自助法和贝叶斯分层建模等统计方法，来量化机器学习基准测试中跨多个任务聚合的性能指标的不确定性。 |
| [^255] | [Satisficing Regret Minimization in Bandits: Constant Rate and Light-Tailed Distribution](https://arxiv.org/abs/2406.06802) | 本文提出SELECT算法模板，通过采样与下置信界检验，在赌博机满意遗憾最小化问题中实现了常数级别的期望满意遗憾，并同时具备标准遗憾保证。 |
| [^256] | [PAPER-HILT: Personalized and Adaptive Privacy-Aware Early-Exit for Reinforcement Learning in Human-in-the-Loop Systems](https://arxiv.org/abs/2403.05864) | PAPER-HILT是针对人机协同系统中隐私保护的创新自适应强化学习策略，通过提前退出方法动态调整隐私保护和系统效用，以适应个体行为模式和偏好。 |
| [^257] | [Protect Your Score: Contact Tracing With Differential Privacy Guarantees](https://arxiv.org/abs/2312.11581) | 这篇论文提出了具有差分隐私保障的接触追踪算法，以解决隐私问题限制接触追踪的部署。该算法在多种情景下展现了卓越性能，并通过在发布每个风险分数时保护个体健康状况的隐私。 |
| [^258] | [Synthetic Blips: Generalizing Synthetic Controls for Dynamic Treatment Effects](https://arxiv.org/abs/2210.11003) | 该论文提出“合成瞬时效应”方法，通过反向归纳将目标单元在各时期的处理瞬时效应递归表示为其他单元瞬时效应的线性组合，从而将合成控制方法推广至具有潜在时变混杂的自适应多处理动态场景，实现任意干预序列下单元均值结果的识别与估计。 |
| [^259] | [Distributionally Robust Transfer Learning.](http://arxiv.org/abs/2309.06534) | 这篇论文介绍了一种分布鲁棒的迁移学习方法，通过优化一个不确定性集合内最具对抗性的损失来实现，该集合是由源分布的凸组合生成的目标人口集合，能够有效地将迁移学习和分布鲁棒的预测模型联系起来。 |

# 详细

[^1]: 一种用于衡量校准的排序方法

    A Ranking Approach for Measuring Calibration

    [https://arxiv.org/abs/2609.13100](https://arxiv.org/abs/2609.13100)

    本文提出了一种新的校准误差度量方法 rankECE，通过比较预测概率的邻近值来衡量校准误差，理论和实验证明它比常用的分箱近似方法能更好地逼近期望校准误差（ECE）。

    

    当使用预测模型提供预测概率时，理想的模型应具有完美校准：结果的真实概率（即 $Y=1$ 的概率）应与预测概率 $f(X)$ 完全一致。在实践中，模型不可避免地会出现校准误差，因此能够测量这种校准误差以评估模型的可靠性非常重要。期望校准误差（ECE）是目前使用最广泛的校准误差度量，但已知在无假设的环境中无法保证对 ECE 进行准确估计。在这项工作中，我们提出了一种替代性度量方法 rankECE，该方法基于将数据点与预测概率 $f(X)$ 的邻近值进行比较。我们的理论保证和实证结果表明，与实践中最常用的分箱近似方法相比，rankECE 为 ECE 提供了更好的代理。

    arXiv:2609.13100v1 Announce Type: cross  Abstract: When providing forecasted probabilities with a predictive model, the ideal model offers perfect calibration: the true probability of the outcome (i.e., the probability that $Y=1$) exactly matches the forecasted probability $f(X)$. In practice, models inevitably exhibit calibration error, and it is therefore important to be able to measure this miscalibration to assess a model's reliability. The Expected Calibration Error (ECE) is the most widely used measure of miscalibration, but is known to be impossible to estimate the ECE with guaranteed accuracy in an assumption-free setting. In this work, we propose an alternative measure, the rankECE, that is based on comparing points with neighboring values of the predicted probability $f(X)$. Our theoretical guarantees and empirical results establish that rankECE provides a better proxy for ECE as compared to binned approximations to ECE, which are the most commonly-used approximations in prac
    
[^2]: 开放式问题的自主研究：电信工单检索案例研究

    Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval

    [https://arxiv.org/abs/2609.13073](https://arxiv.org/abs/2609.13073)

    本文通过电信工单检索这一工业级案例研究，探讨了基于大语言模型的自主研究在开放式机器学习问题中的应用，发现其在狭窄的超参数优化中表现出色，但在处理具有表示、架构和训练数据生成自由度的开放式问题时仍存在局限性。

    

    基于大语言模型（LLM）的系统在问题解决和编程能力方面的最新突破推动了“AI for Science”范式的进展，有望在机器学习（ML）研究中取代人类角色。然而，尽管已有多个完全自主的端到端机器学习研究框架被提出，它们的成功实施通常仅限于搜索空间狭窄的问题，如语言建模或生物医学机器学习基准测试。在本文中，我们通过一个案例研究探索了如何将自主研究应用于解决开放式、工业级的机器学习问题：电信工单检索，这是一个在表示、架构和训练数据生成方面具有自由度的开放式任务。我们发现，使用商业和开源智能体对开放式问题进行自主研究既展现出前景也存在局限性：虽然自主研究在狭窄的超参数优化中表现出色，但它在……（原文在此处截断）

    arXiv:2609.13073v1 Announce Type: cross  Abstract: Recent breakthroughs in LLM-based systems and their abilities in problem solving and coding have allowed progress in the AI for Science paradigm, potentially replacing human roles in machine learning (ML) research. However, while several frameworks of fully autonomous end-to-end ML research have been proposed, successful implementations of them are often limited to problems with narrow search spaces, like language modeling or biomedical ML benchmarks. In this paper, we explore how autonomous research can be adapted to solve open-ended, industry-grade ML problems, by considering a case study: telecom ticket retrieval, an open-ended task with degrees of freedom in representation, architecture, and training data generation. We discover that autonomous research for open-ended problems with commercial and open-source agents shows both promise and limitations: while autonomous research can excel in narrow hyperparameter optimization, it lack
    
[^3]: MAxBench：一个多分类概念恢复基准测试

    MAxBench: A Multinomial Concept Recovery Benchmark

    [https://arxiv.org/abs/2609.13072](https://arxiv.org/abs/2609.13072)

    本文提出了MAxBench，一个几何无关的多分类概念表示评估框架，通过在6个概念和4个模型上系统比较10种定位方法（涵盖5种几何类型），为识别最适合多分类概念引导的表示几何结构及最有效的恢复方法提供了统一评估标准。

    

    对语言模型行为的细粒度控制（例如行为引导/steering）是可解释性研究中更具可操作性的成果之一。对于诸如“拒绝”这样的二元概念，激活空间中的单一方向通常就足以实现引导。然而，许多概念并非二元：例如“动物”和“国家”包含许多子类别，每个子类别又有多个实例。对于这些概念，可能的表示几何结构的搜索空间远大于二元概念；因此，目前尚不清楚哪些几何结构最为合适，也不清楚哪些方法在恢复这些结构时最为有效。在这项工作中，我们介绍了MAxBench，这是一个几何无关的多分类概念表示评估框架，其基于从恢复的概念表示中进行采样来评估。我们使用MAxBench在6个概念和4个模型上比较了10种定位方法（涵盖5种几何类型）。利用该框架，我们发现（i）仿射子空间在引导方面表现更为……（摘要在此处被截断）

    arXiv:2609.13072v1 Announce Type: cross  Abstract: Fine-grained control of language model behaviors (e.g., steering) is among the more actionable outcomes of interpretability research. For binary concepts such as refusal, a single direction in activation space often suffices for steering. However, many concepts are not binary: Animals and Countries contain many subcategories, each with multiple instances. For these concepts, the search space over possible representation geometries is far larger than for binary concepts; it is thus not clear what geometries are most appropriate, nor what methods are most effective at recovering them. In this work, we introduce MAxBench, a geometry-agnostic evaluation framework for multinomial concept representations based on sampling from the recovered concept representation. We use MAxBench to compare 10 localization methods (covering 5 geometry types) across 6 concepts and 4 models. Using this framework, we find that (i) affine subspaces steer more re
    
[^4]: CanvasAnneal：面向扩散语言模型的课程强化学习

    CanvasAnneal: Curriculum Reinforcement Learning for Diffusion Language Models

    [https://arxiv.org/abs/2609.13060](https://arxiv.org/abs/2609.13060)

    CanvasAnneal通过课程学习策略，先在扩散画布中注入教师模型的推理轨迹以热启动强化学习探索，再随着训练逐步移除引导让模型学会独立推理，从而显著提升扩散语言模型在数学推理和工具使用任务上的表现。

    

    扩散语言模型（DLMs）具备有前景的并行生成能力，但在复杂推理和工具使用任务上落后于自回归模型。虽然强化学习（RL）最近已被应用于增强DLMs，但标准的RL方法存在探索瓶颈。为了解决这一问题，我们从更强的教师模型中注入推理先验来引导RL探索。本文提出了CanvasAnneal，一个课程引导的扩散RL框架。在初始RL阶段，我们通过将教师生成的推理轨迹注入初始扩散画布中来热启动探索。随着训练的推进，我们逐渐移除这种引导，要求模型独立生成更多的推理轨迹。在数学推理和工具使用基准测试中，CanvasAnneal在MATH500、Countdown和Tau2上优于标准diffusion-GRPO，并在多个基准上显著加速了奖励的提升。

    arXiv:2609.13060v1 Announce Type: new  Abstract: Diffusion Language Models (DLMs) offer promising parallel generation capabilities but lag behind autoregressive models in complex reasoning and tool-use tasks. While Reinforcement Learning (RL) has recently been applied to enhance DLMs, standard RL approaches suffer from an exploration bottleneck. To address this, we inject reasoning priors from a stronger teacher model to guide RL exploration. In this paper, we introduce CanvasAnneal, a curriculum-guided diffusion RL framework. During the initial RL phase, we warm-start exploration by injecting teacher-generated reasoning traces into the initial diffusion canvas. As training progresses, we gradually remove this guidance and require the model to generate more of the reasoning trajectory independently. Across mathematical reasoning and tool-use benchmarks, CanvasAnneal improves over standard diffu-GRPO on MATH500, Countdown, and Tau2 and substantially accelerates reward improvement on sev
    
[^5]: 良性损失景观可以与最坏情况难度共存

    Benign Loss Landscapes Can Coexist with Worst-Case Hardness

    [https://arxiv.org/abs/2609.13057](https://arxiv.org/abs/2609.13057)

    本文研究了树张量网络，证明其损失景观虽然是条件良性的（易于优化），但仍包含梯度下降无法在多项式时间内学习的最坏情况目标，从而揭示了良性损失景观可以与最坏情况的学习难度共存。

    

    深度神经网络足够富有表现力，能够包含可以在多项式时间内评估、但无法通过梯度下降在多项式时间内学习的最坏情况目标。然而，对于实际任务，它们却学习得很好，这引发了一个问题：现实世界目标的何种非通用结构使得这一点成为可能。现有的代理模型无法提出这个问题，因为它们要么完全缺乏难以学习的目标（深度线性网络），要么无法高效地评估此类目标（核方法、无限宽度极限）。我们研究了树张量网络（TTN），这是一种泛化了深度线性网络和Tucker分解的模型类。我们证明它们可以嵌入任意的单次读取布尔公式，因此在与神经网络相同的机制下，包含无法被梯度下降在多项式时间内学习的多项式规模目标。尽管如此，我们证明了它们的损失景观对于每个可实现的目标都是条件良性的。

    arXiv:2609.13057v1 Announce Type: new  Abstract: Deep neural networks are expressive enough to contain worst-case targets that can be evaluated in polynomial time but cannot be learned in polynomial time by gradient descent. For practical tasks they nonetheless learn well, raising the question of what non-generic structure of real-world targets enables this. Existing surrogate models cannot pose this question because they either lack hard-to-learn targets entirely (deep linear networks) or cannot evaluate such targets efficiently (kernel methods, infinite-width limits). We study tree tensor networks (TTNs), a model class that generalizes deep linear networks and Tucker decompositions. We show they embed arbitrary read-once Boolean formulas, and thus contain polynomial-size targets that cannot be learned by gradient descent in polynomial time under the same mechanism as neural networks. Despite this, we prove that their loss landscapes are conditionally benign for every realizable targe
    
[^6]: Dynin-Robotics：全模态统一的扩散视觉-语言-动作模型

    Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model

    [https://arxiv.org/abs/2609.13053](https://arxiv.org/abs/2609.13053)

    提出基于全模态掩码扩散主干网络的统一视觉-语言-动作模型Dynin-Robotics，通过共享轨迹模型将目标预测与动力学预测引入动作生成与选择，实现多任务联合学习并支持测试时性能扩展。

    

    视觉目标与动力学预测能够为语言条件化的机器人策略同时提供目标结果和动作相关的场景变化表征。我们通过一个共享的轨迹模型将这些预测引入动作生成与选择之中。Dynin-Robotics在全模态掩码扩散主干网络Dynin-Omni上实现了这一构想，将语言、视觉观测、目标和动作统一表示为离散token。通过变换条件与目标跨度，同一模型可以学习动作预测、动作条件下的下一观测预测、终端目标状态预测以及轨迹到指令的重建。这些接口通过目标预测、动作候选评估以及动作与未来状态预测的联合精炼，支持测试时的性能扩展。我们在来自48个Open X-Embodiment数据集的约133万条轨迹上对模型进行了持续预训练，并分别进行适配。

    arXiv:2609.13053v1 Announce Type: cross  Abstract: Visual goal and dynamics prediction can provide language-conditioned robot policies with both a target outcome and a representation of action-dependent scene changes. We bring these predictions into action generation and selection through a shared trajectory model. Dynin-Robotics implements this formulation on Dynin-Omni, an omnimodal masked-diffusion backbone, representing language, visual observations, goals, and actions as discrete tokens. By varying conditioning and target spans, the same model learns action prediction, action-conditioned next-observation prediction, terminal goal-state prediction, and trajectory-to-instruction reconstruction. These interfaces support test-time scaling through goal prediction, action-candidate evaluation, and joint refinement of action and future-state predictions. We continually pretrain the model on approximately 1.33 million trajectories from 48 Open X-Embodiment datasets and adapt it separately
    
[^7]: 基于正则化的鲁棒强化学习的统一约束视角

    A Unified and Constrained View of Regularization-Based Robust Reinforcement Learning

    [https://arxiv.org/abs/2609.13050](https://arxiv.org/abs/2609.13050)

    该论文通过推导性能差距上界统一了基于正则化的鲁棒强化学习方法，将鲁棒训练表述为约束优化问题，并通过联合更新拉格朗日乘子自动调整正则化权重以提升鲁棒性。

    

    基于正则化的方法已成为训练深度强化学习策略以对抗对抗性输入扰动的标准方法。在本文中，我们通过推导标称策略与最坏情况策略之间性能差距的新上界来统一这些方法。每个上界都表示为现有的正则化目标加上标称策略与最坏情况策略之间的KL散度惩罚，这进一步解释了为什么在实践中添加KL惩罚能够提升鲁棒性。基于这些上界，我们将鲁棒训练表述为一个约束优化问题，并证明现有方法对应于固定拉格朗日乘子的特殊情况。与之相反，我们在更新策略的同时联合更新该乘子，从而自动调整正则化权重。最后，我们在多个连续控制任务上进行了广泛的对抗性评估，以验证我们的理论分析。

    arXiv:2609.13050v1 Announce Type: new  Abstract: Regularization-based methods have become a standard approach for training Deep Reinforcement Learning policies against adversarial input perturbations. In this paper, we unify these methods by deriving new upper bounds on the performance gap between the nominal and worst-case policies. Each upper bound is expressed as an existing regularization objective plus a KL-divergence penalty between the nominal and worst-case policies, which further explains why adding a KL penalty improves robustness in practice. Building on these bounds, we formulate robust training as a constrained optimization problem, showing that existing methods correspond to the special case of a fixed Lagrange multiplier. We instead update the multiplier jointly with the policy to automatically tune the regularization weight. Finally, we conduct extensive adversarial evaluations across several continuous control tasks to validate our theoretical analysis.
    
[^8]: MCRL2：基于多资源交叉注意力表示学习增强的强化学习用于云微服务调度

    MCRL2: Multi-resource Cross-attention-based Representation Learning-augmented Reinforcement Learning for Cloud Microservice Scheduling

    [https://arxiv.org/abs/2609.13048](https://arxiv.org/abs/2609.13048)

    该论文提出了MCRL2，一种将多资源交叉注意力表示学习与强化学习相结合的新方法，通过捕捉节点、资源和微服务之间的结构化交互关系，有效提升云环境下微服务调度的负载均衡能力和服务质量。

    

    高效的微服务调度对于维持数据中心各节点间的负载均衡以及确保高质量的服务至关重要。然而，由于波动工作负载下的动态资源不平衡、多资源维度间的非线性耦合以及微服务资源需求的异构性，在实践中实现这一目标仍然具有挑战性。虽然基于强化学习的方法已展现出潜力，但它们难以捕捉异构资源之间的复杂相互依赖关系，并且忽视了学习富含信息的系统表示的重要性。为了解决这些局限性，我们提出了MCRL2，一种通过基于多资源交叉注意力的表示学习来增强的新型强化学习方法，用于微服务调度。具体而言，我们首先提出了MCRL，一种新颖的表示学习方法，能够捕捉节点、资源和（摘要原文在此处截断）

    arXiv:2609.13048v1 Announce Type: new  Abstract: Efficient microservice scheduling is crucial for maintaining load balance across nodes in data centers and ensuring high quality of service. However, achieving this in practice remains challenging due to dynamic resource imbalance under fluctuating workloads, nonlinear coupling across multiple resource dimensions, and the heterogeneity of microservice resource demands. While reinforcement learning-based approaches have shown promise, they struggle to capture the complex interdependencies among heterogeneous resources and neglect the importance of learning informative system representations. To address these limitations, we propose MCRL2, a novel reinforcement learning approach augmented with multi-resource cross-attention-based representation learning for microservice scheduling. Specifically, we first propose MCRL, a novel representation learning approach that captures structured and informative interactions among nodes, resources, and 
    
[^9]: 扩散模型与概念形成

    Diffusion Models and Concept Formation

    [https://arxiv.org/abs/2609.13047](https://arxiv.org/abs/2609.13047)

    本文论证扩散模型虽为图像合成而设计，但其含噪边缘分布的众数所形成的层次结构在四个方面与认知科学中经典的Cobweb概率概念层次一一对应，表明扩散模型隐式地执行了概念形成计算，且“基本层次”在中间噪声水平处涌现。

    

    人类将知识组织成一个具有嵌套抽象层次的概念分类体系，其中存在一个“基本层次”，人们在这一层次上能够以最少的认知努力来识别和命名物体。Cobweb是对这种能力的一种经典认知科学解释，它是一个增量学习器，通过最大化类别效用构建概率概念层次结构。我们认为，扩散模型虽然是为图像合成而设计的，却隐式地执行了相同的计算。扩散模型的含噪边缘分布是数据分布的高斯平滑，这些边缘分布的众数形成了一个层次结构，该结构在四个方面与Cobweb树的概率原型相对应：两者都是层次化密度模型；都是带有高斯原型的层次化贝叶斯模型；都将类别判定视为一种降低不确定性的分数跟随过程；并且在两者中都会涌现出基本层次。我们将扩散模型的基本层次定位在一个中间噪声水平上……

    arXiv:2609.13047v1 Announce Type: cross  Abstract: Humans organize knowledge into a taxonomy of concepts with nested levels of abstraction and a \emph{basic level} at which people recognize and name objects with the least cognitive effort. Cobweb is a classic cognitive account of this ability, an incremental learner that builds a probabilistic concept hierarchy by maximizing category utility. We argue that diffusion models, although designed for image synthesis, implicitly perform the same computation. The noisy marginals of a diffusion model are Gaussian smoothings of the data distribution, and the modes of these marginals form a hierarchy that corresponds to a Cobweb tree of probabilistic prototypes in four respects. Both are hierarchical density models, both are hierarchical-Bayesian models with Gaussian prototypes, both treat categorization as score-following that reduces uncertainty, and in both a basic level emerges. We locate this basic level for a diffusion model at an intermed
    
[^10]: 基于对抗性重要性采样的鲁棒策略优化

    Robust Policy Optimization via Adversarial Importance Sampling

    [https://arxiv.org/abs/2609.13044](https://arxiv.org/abs/2609.13044)

    该论文提出了Advis方法，通过对标准训练轨迹进行重要性采样来估计和优化可验证的最坏情况回报，无需额外环境交互和辅助网络即可捕获长期鲁棒性，同时提供模块化PyTorch库advrl以促进鲁棒DRL方法的快速开发与可复现评估。

    

    在保护深度强化学习（DRL）策略免受输入扰动方面已经取得了显著进展。开发鲁棒的DRL涉及三个主要阶段：算法设计、实现和评估。在这项工作中，我们识别并解决了每个阶段的一个关键局限。首先，我们提出了对抗性重要性采样，这是一种利用标准训练产生的轨迹上的重要性采样来估计和优化可验证的最坏情况回报的方法。Advis满足了三个先前工作未能同时实现的理想标准：它不需要额外的环境交互，不需要辅助网络，并且能够捕获长期鲁棒性。其次，我们介绍了advrl，这是一个模块化的PyTorch库，为现有的鲁棒性方法和对抗攻击提供了干净的单文件实现，便于快速原型开发，并实现可复现和可追溯的评估。第三，我们重新审视了评估……

    arXiv:2609.13044v1 Announce Type: new  Abstract: Significant progress has been made in safeguarding deep reinforcement learning (DRL) policies against input perturbations. Developing robust DRL involves three main stages: algorithm design, implementation, and evaluation. In this work, we identify and address a key limitation at each stage. First, we introduce Adversarial Importance Sampling (Advis), a method that uses importance sampling over trajectories from standard training to estimate and optimize verifiable worst-case returns. Advis satisfies three desirable criteria not jointly achieved by prior work: it requires no additional environment interactions, no auxiliary networks, and captures long-term robustness. Second, we introduce advrl, a modular PyTorch library that provides clean, single-file implementations of existing robustness methods and adversarial attacks, facilitating rapid prototyping and enabling reproducible and traceable evaluations. Third, we revisit evaluation un
    
[^11]: DynSHAP：迈向可解释的动态生存分析

    DynSHAP: Towards Explainable Dynamic Survival Analysis

    [https://arxiv.org/abs/2609.13042](https://arxiv.org/abs/2609.13042)

    DynSHAP是一个专为动态生存分析设计的SHAP可解释性框架，通过将时间-特征对作为Shapley博弈的参与者，并引入利用条件采样处理特征时间依赖关系的Temporal DynSHAP，从而能够准确解释纵向不规则输入与函数型生存输出。

    

    面向动态生存分析（DSA）的深度学习模型通过纳入纵向患者数据实现了强大的预测性能，但其黑盒特性限制了临床信任度和应用推广。现有的可解释性方法无法同时处理纵向的不规则输入和函数型的生存输出，这限制了它们在动态生存分析中的可用性。我们提出了DynSHAP，一个专门适用于动态生存分析的SHAP框架。它通过将时间-特征对视为Shapley博弈中的参与者，将常见的边际SHAP估计器扩展到了这一场景。我们进一步提出了Temporal DynSHAP，它学习特征随时间的线性依赖关系，并使用条件采样在解释中对这些依赖关系进行处理。当应用于具有已知真实归因的合成数据时，对于给定的最先进模型，Temporal DynSHAP比边际估计器能更准确地恢复出时间相关特征。应用于两个……（原文摘要此处截断）

    arXiv:2609.13042v1 Announce Type: new  Abstract: Deep learning models for dynamic survival analysis (DSA) achieve strong predictive performance by incorporating longitudinal patient data, but their black box nature limits clinical trust and adoption. Existing explainability methods cannot handle longitudinal, irregular inputs and functional survival outputs simultaneously, which limits their usability in DSA. We propose DynSHAP, a SHAP framework suited specifically for dynamic survival analysis. It extends common marginal SHAP estimators to this setting by treating time--feature pairs as players in the Shapley game. We further introduce Temporal DynSHAP, which learns linear dependencies in features over time and uses conditional sampling to address them in explanations. When applied to synthetic data with known ground-truth attributions, Temporal DynSHAP recovers temporally dependent features more accurately than marginal estimators for a given state-of-the-art model. Applied to two re
    
[^12]: 面向离群点鲁棒随机梯度下降的基于分位数的损失过滤方法

    Quantile-based Loss Filtering for Outlier-Robust Stochastic Gradient Descent

    [https://arxiv.org/abs/2609.13040](https://arxiv.org/abs/2609.13040)

    该论文提出了分位数-k-损失SGD（QkL-SGD）框架，通过每次迭代采样k个分量损失并从经验下q分位数中均匀选取更新索引来过滤损坏分量，在标准凸性假设下证明了线性收敛性，并提供了适用于任意采样规模的小样本概率收敛分析。

    

    我们研究了有限和优化问题中的基于损失的过滤方法，其中一部分分量函数已损坏，其梯度可能高度不可靠。受基于最小损失的SGD（min-k-loss）和针对损坏线性系统的基于分位数方法的启发，我们提出并分析了一个通用的损失过滤框架——分位数-k-损失SGD（QkL-SGD），该框架在每次迭代中采样k个分量损失，并使用从经验下q分位数中均匀选取的索引进行更新。在标准凸性假设下，我们证明了这一类方法的线性收敛性，其要求采样规模随损坏数量成比例增长，并满足子集强凸性阈值条件。对于无法或不宜进行足够大采样的情形，我们给出了补充性的小样本概率分析，该分析适用于任意采样规模k，其收敛行为取决于选中离群点的概率以及……

    arXiv:2609.13040v1 Announce Type: new  Abstract: We study loss-based filtering for finite-sum optimization with a subset of corrupted component functions whose gradients may be highly unreliable. Motivated by minimum-loss-based SGD (min-$k$-loss) and quantile-based methods for corrupted linear systems, we propose and analyze a general loss-filtering framework -- Quantile-\(k\)-Loss SGD (Q\(k\)L-SGD) -- that samples \(k\) component losses at each iteration and updates using an index chosen uniformly from the lower empirical \(q\)-quantile. We prove linear convergence of this family of methods under standard convexity assumptions, requiring the sample size to scale with the number of corruptions and a subset strong-convexity threshold. For the cases when large enough sampling is impossible or undesirable, we give a complementary small-sample probabilistic analysis that covers any sample size $k$ and the convergence behavior depends on the probability of selecting an outlier and on the cu
    
[^13]: 面向演化领域的迁移学习

    Transfer Learning for Evolving Domains

    [https://arxiv.org/abs/2609.13039](https://arxiv.org/abs/2609.13039)

    该论文提出了一种新的迁移学习问题——面向演化领域的迁移学习，将传统上按目标数据可用性假设相互分割的各子领域统一起来，把迁移学习建模为部署系统随时间逐步获得目标数据与标签的完整演化轨迹。

    

    迁移学习研究如何利用来自不同任务或领域（源）的知识，来提升相关任务或领域（目标）中的预测性能。通常，迁移学习研究被划分为若干相互孤立的子领域（如领域泛化、领域自适应或多领域学习），每个子领域对目标数据的可用性做出不同的假设，即训练时有多少数据以及多少标签可用。然而，在许多现实应用中，数据可用性并非固定不变，而是随着实例和标签从新领域逐步收集而随时间演化。每一种经典设定实际上仅描述了已部署系统必须完整经历的一条轨迹中的某个快照。我们将这条轨迹形式化为一个独立的迁移学习问题——面向演化领域的迁移学习，该问题由环境固定的数据可用性过程来定义。

    arXiv:2609.13039v1 Announce Type: new  Abstract: Transfer learning explores how to leverage knowledge from various tasks or domains (sources) to enhance predictive performance in related tasks or domains (targets). Typically, transfer learning research is segmented into several isolated sub-areas (such as domain generalisation, domain adaptation, or multi-domain learning), each making distinct assumptions about target data availability, namely how much data and how many labels are available at training time. However, in many real-world applications, data availability is not fixed but evolves over time, as instances and labels are progressively collected from a new domain. Each of the classical settings then describes only a snapshot of a trajectory that a deployed system must traverse in full. We formalise this trajectory as a transfer learning problem in its own right, Transfer Learning for Evolving Domains (TrED), specified by a data availability process fixed by the environment, a l
    
[^14]: 基于群胚的局部对称性强化学习内部状态表示

    Groupoid-Based Internal State Representations for Reinforcement Learning with Local Symmetries

    [https://arxiv.org/abs/2609.13035](https://arxiv.org/abs/2609.13035)

    提出了一种基于群胚的强化学习框架，能够捕获局部的、状态依赖的对称性并在交互中动态发现等价结构，使智能体在对称性约简的空间中进行学习和决策，同时保留局部区别。

    

    对称性在降低强化学习问题复杂性方面发挥着核心作用，然而大多数现有方法依赖于固定的群作用或预定义的状态抽象。经典的强化学习算法通常假设马尔可夫决策过程具有全局统一的结构，动作和转移在各处均匀适用，这一假设限制了它们利用许多现实环境中存在的模块性以及局部的、依赖于上下文的规律性的能力。我们提出了一个使用群胚来捕获局部的、状态依赖的对称性的强化学习框架，并支持在交互过程中动态发现等价结构。智能体维护轨道代表元以及将原始状态映射到规范形式的传输映射，使得学习和决策能够在一个对称性约简的空间中进行，同时保留局部区别。实验结果表明，所提出的

    arXiv:2609.13035v1 Announce Type: new  Abstract: Symmetries play a central role in reducing the complexity of reinforcement learning problems, yet most existing approaches rely on fixed group actions or predefined state abstractions. Classical reinforcement learning algorithms typically assume a globally structured Markov decision process with uniformly applicable actions and transitions, an assumption that limits their ability to exploit modularity and local, context-dependent regularities present in many realistic environments. We propose a reinforcement learning framework using groupoids to capture local, state-dependent symmetries and support the dy- namic discovery of equivalence structures during interaction. The agent maintains orbit representatives together with transporters that map raw states to canonical forms, enabling learning and decision-making to be performed in a symmetry-reduced space while preserving local distinctions. Empirical results demonstrate that the proposed
    
[^15]: 面向表格基础模型的注意力量化

    Attention Quantization for Tabular Foundation Models

    [https://arxiv.org/abs/2609.13031](https://arxiv.org/abs/2609.13031)

    本文提出针对表格基础模型的FP8注意力量化策略，通过将测试行与训练行的量化误差对齐，在避免精度损失的同时，利用Triton内核实现了最高1.7倍的注意力计算加速。

    

    随着表格基础模型近期的兴起和广泛应用，优化其推理性能已成为效率研究的一个新兴领域。虽然这些模型在架构上与基于Transformer的大语言模型（LLMs）相似，但两者的模型规模和服务模式存在显著差异。我们表明，优化重点应放在注意力计算上，而不是权重或KV缓存量化，后者在LLMs中更为流行。我们开发了一种针对查询、键和值的FP8量化策略，并使用显式FP8矩阵乘法指令来加速注意力计算。我们发现，将测试行中的量化误差与训练行中的量化误差对齐至关重要，否则精度会急剧下降。我们的Triton内核相比常规16位内核实现了高达1.7倍的加速，并且我们证明在TabArena基准上，TabPFN-v3和TabICLv2没有出现相关的精度损失。

    arXiv:2609.13031v1 Announce Type: new  Abstract: With the recent rise and adoption of tabular foundation models, optimizing their inference performance becomes an emerging field for efficiency research. While the models are architecturally similar to transformer-based large language models (LLMs), the size and serving patterns differ significantly. We show that the focus should be on the attention calculation and less on weight or KV cache quantization, which are more popular in LLMs. We develop a quantization strategy for queries, keys, and values to FP8 and use explicit FP8 matrix multiplication instructions to speed up the attention calculation. We find that it is crucial to align the quantization error in the test rows with the quantization error in the training rows, as otherwise the accuracy drops drastically. Our Triton kernel achieves a speedup up to 1.7x over regular 16-bit kernels, and we show that on TabPFN-v3 and TabICLv2 there is no relevant accuracy loss across TabArena a
    
[^16]: 面向跨维度大规模最优传输的对偶引导层次化边定位方法

    Dual-guided Hierarchical Edge Localization for Large-scale Optimal Transport Across Dimensions

    [https://arxiv.org/abs/2609.13010](https://arxiv.org/abs/2609.13010)

    提出HELLO层次化求解器，利用对偶势双重引导将大规模离散最优传输转化为边定位问题，在百万点规模下以线性内存复杂度获得更低传输目标值和数量级级别的运行时间加速。

    

    最优传输（OT）在机器学习中用于比较分布和对齐数据集，然而未正则化的离散最优传输需要求解一个具有平方数量级传输变量的线性规划问题。我们提出了HELLO，一个层次化求解器，它将大规模离散最优传输转化为边定位问题，并利用对偶势来同时指导由粗到细的初始化以及层级内的精细化过程。初始化阶段通过递归子采样层次传播粗对偶势以分配候选边。精细化阶段随后迭代地插入每行和每列中最大的对偶违反者，直到相对KKT残差满足预设容差，同时基于预算的剪枝确保了线性内存复杂度。对于精确算术的精细化过程，我们证明了在符号字典序规则下，算法能够在全局最优解处有限步终止。在百万点规模下，HELLO相较于强基线方法取得了更低的传输目标值，并实现了数量级级别的运行时间提升。

    arXiv:2609.13010v1 Announce Type: new  Abstract: Optimal transport (OT) compares distributions and aligns datasets in machine learning, yet unregularized discrete OT requires a linear program with quadratically many transport variables. We propose HELLO, a hierarchical solver that casts large-scale discrete OT as edge localization and uses dual potentials to guide both coarse-to-fine initialization and within-level refinement. Initialization propagates coarse dual potentials across a recursive subsampling hierarchy to assign candidate edges. Refinement then iteratively inserts the largest dual violators in each row and column until the relative KKT residual meets a prescribed tolerance, while budgeted pruning ensures linear memory complexity. For exact-arithmetic refinement, we prove finite termination at a global optimum under a symbolic lexicographic rule. At the million-point scale, HELLO attains lower transport objectives with order-of-magnitude runtime improvements over strong bas
    
[^17]: 以封面评判：清理大语言模型真实性基准以避免表层特征泄漏

    Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage

    [https://arxiv.org/abs/2609.13003](https://arxiv.org/abs/2609.13003)

    该研究揭示TruthfulQA等二选一真实性基准存在表层特征泄漏问题，模型无需真正推理即可借助答案表面特征超越随机水平，作者据此提出Audit-Prune清理机制并发布了泄漏接近随机水平的清理版数据集。

    

    二选一的真实性基准测试要求模型在正确和错误答案之间做出选择，但如果这两个答案在表层特征上存在系统性差异，模型就可以在没有执行预期推理的情况下超越随机概率水平。我们证明这种失效模式是可以被检测到的，并且可以被下游分类器所利用。在TruthfulQA中，一个简单的六特征逻辑回归分类器就能在区分正确与错误答案方面取得相当高的准确率。我们进一步表明，类似的表层伪影也存在于其他基准测试中。为了解决这一问题，我们开发了一种通用机制，通过移除最具泄漏强化的答案对来清理这些基准。我们发布了一个将表层特征泄漏降低至接近随机水平的TruthfulQA版本，并提供了一种名为Audit-Prune的机制，以便数据集在发布前得以清理。

    arXiv:2609.13003v1 Announce Type: new  Abstract: Binary-choice truth benchmarks ask models to choose between a correct and an incorrect answer, but if the two answers differ systematically in surface-level features, models can exceed chance without performing the intended reasoning. We show that this failure mode is detectable and can be exploited by downstream classifiers. In TruthfulQA, a simple six-feature logistic classifier achieves substantial accuracy in separating correct from incorrect answers. We further show that similar surface-level artifacts are present in additional benchmarks. To counteract this, we developed a general mechanism to clean them by removing the most leakage-reinforcing pairs. We release a version of TruthfulQA with surface-feature leakage reduced close to chance and provide a mechanism, Audit-Prune, so that the datasets can be cleaned before release.
    
[^18]: 谱重尾出现的完整Adam定理

    A Full Adam Theorem for Spectral Heavy-Tail Onset

    [https://arxiv.org/abs/2609.12996](https://arxiv.org/abs/2609.12996)

    该论文在高斯Stein-Hermite师生状态演化模型中证明了谱重尾出现的完整Adam定理，给出了由首个尖峰-主体谱间隙决定的命中时间定律 τ_ε = Θ(Δ_1^{-γ} d^ρ log(Ψ_0/ε))。

    

    我们在一个闭合的高斯Stein-Hermite师生状态演化模型中，证明了关于谱重尾出现的完整Adam定理。该定理从实际的全批量Adam递推关系出发，通过Stein-Hermite微积分推导出总体梯度，证明了有限宽度协方差集中性，将多步Adam动量转换为精确的非中心高斯符号核，通过基同质化定理控制Adam的对角分母，从Hermite边缘转移定理推导出正则变化的投影更新响应，将该响应推过精确的Gram更新，并证明了具有匹配上下界命中结果的近似目标KL收缩。最终得到的定律为 τ_ε = Θ(Δ_1^{-γ} d^ρ log(Ψ_0/ε))，其中 Δ_1 是第一个尖峰-主体谱间隙。该结果是“完整”的，其精确含义为：从Adam的动量和分母到谱（原文在此处截断）……

    arXiv:2609.12996v1 Announce Type: new  Abstract: We prove a full Adam theorem for spectral heavy-tail onset in a closed Gaussian Stein-Hermite teacher-student state-evolution model. The theorem begins with the actual full-batch Adam recurrences, derives the population gradient by Stein-Hermite calculus, proves finite-width covariance concentration, converts multi-step Adam momentum into an exact non-centered Gaussian sign kernel, controls the diagonal Adam denominator by a basis-homogenization theorem, derives a regularly varying projected update response from a Hermite edge-transfer theorem, pushes the response through the exact Gram update, and proves approximate-target KL contraction with matching upper and lower hitting bounds. The final law is (\tau_\varepsilon=\Theta(\Delta_1^{-\gamma}d^\rho\log(\Psi_0/\varepsilon))), where (\Delta_1) is the first spike-bulk spectral gap. The result is full in the following precise sense: every step from Adam's momentum and denominator to the spe
    
[^19]: 神经网络优化器动力学中重尾谱涌现的维度修正命中时间

    Dimension-Corrected Hitting Times for Heavy-Tailed Spectral Emergence in Neural Optimizer Dynamics

    [https://arxiv.org/abs/2609.12994](https://arxiv.org/abs/2609.12994)

    该论文将神经网络训练中重尾谱的涌现时间首次建模为右删失命中时间问题，并实证发现由谱隙与权重矩阵维度共同决定的修正幂律 τ_HT ≈ C·Δ₁^(-γ)·d^ρ 能够准确预测重尾涌现时间，显著优于仅依赖谱隙的模型。

    

    神经网络权重矩阵的经验谱密度呈现重尾分布，这被广泛用作隐式自正则化的诊断指标，但重尾涌现所需的步数复杂度仍知之甚少。我们将谱重尾的形成形式化为一个右删失命中时间问题：在观测范围内未达到重尾诊断指标的运行被视为删失数据，而非被丢弃。在受控的全批量教师-学生动力学中，我们发现仅凭第一步的尖峰-体隙（spike-bulk gap）无法解释涌现时间。相反，有限涌现时间的回归结果支持一个维度修正的谱隙定律 τ_HT ≈ C·Δ₁^(-γ)·d^ρ，在330个完整运行中 R²=0.683，γ=0.626，ρ=0.772。右删失对数正态加速失效时间模型进一步表明，维度修正模型优于仅考虑谱隙的模型，将AIC从706.62改进至628.70。在理论层面，我们……（原文摘要截断）

    arXiv:2609.12994v1 Announce Type: new  Abstract: Heavy-tailed empirical spectral densities of neural-network weight matrices are widely used as diagnostics of implicit self-regularization, but the step complexity of heavy-tail emergence remains poorly understood. We formulate spectral heavy-tail formation as a right-censored hitting-time problem: a run that does not reach a heavy-tail diagnostic within the observation horizon is treated as censored rather than discarded. In controlled full-batch teacher--student dynamics, we find that the first-step spike--bulk gap alone does not explain onset time. Instead, finite-onset regression supports a dimension-corrected spectral-gap law, (\tau_{\mathrm{HT}}\approx C\Delta_1^{-\gamma}d^\rho), with (R^2=0.683), (\gamma=0.626), and (\rho=0.772) across 330 completed runs. Right-censored lognormal accelerated-failure-time models further favor the dimension-corrected model over a gap-only model, improving AIC from 706.62 to 628.70. Theoretically, we
    
[^20]: 信息诱导的训练几何：精确约简、典范补全与结构化表达能力

    Information-Induced Training Geometry: Exact Reduction, Canonical Completion, and Structured Expressivity

    [https://arxiv.org/abs/2609.12991](https://arxiv.org/abs/2609.12991)

    该论文证明在仿射不变黎曼几何下，部分信息通道对优化器度量的压缩具有显式且唯一的典范补全，补全随通道移动形成规范不变的秩分层，从而实现了从完整几何到可见几何的精确变分约简与结构化表达能力刻画。

    

    训练数据通过一个所声明的信息通道中可见的余向量来约束优化器的几何结构。我们研究这种部分信息如何相对于一个参考系确定完整的正定余度量，以及哪些自由度仍然无法被识别。我们的核心结果解决了仿射不变黎曼几何下满列秩正定压缩的问题。该压缩映射是一个分裂-Hadamard度量亚浸没，并拥有一个显式的唯一补全——它是实现可见目标的仿射不变意义上最近的完整几何，并给出从完整到可见的精确变分约简。当信息通道发生移动时，这些补全构成正定锥上规范不变的秩分层结构。其闭式形式的拉回配对度量通过一个参考失配权重，将可见度量的运动与子空间旋转分离开来，给出一个显式的半正定多方向Gram矩阵，并揭示了精确的奇异……（原文摘要在此处被截断）

    arXiv:2609.12991v1 Announce Type: new  Abstract: Training data constrains optimizer geometry through the covectors visible to a declared information channel. We study how such partial information determines a full positive cometric relative to a reference and which degrees of freedom remain unidentified. Our central result resolves full-column-rank positive-definite compression under affine-invariant Riemannian geometry. The compression map is a split-Hadamard metric submetry and admits an explicit unique completion that is the affine-invariant nearest full geometry realizing a visible target and yields exact full-to-visible variational reduction. When the channel moves, the completions form a gauge-invariant rank stratification of the positive-definite cone. Its closed-form pullback pair metric separates visible-metric motion from subspace rotation through a reference-mismatch weight, yields an explicit positive-semidefinite multi-direction Gram matrix, and exposes the precise singula
    
[^21]: 来自芬兰水域的大规模AIS数据集

    A Large-Scale AIS Dataset from Finnish Water

    [https://arxiv.org/abs/2609.12938](https://arxiv.org/abs/2609.12938)

    本文发布了一个来自芬兰水域（波罗的海区域）的大规模AIS数据集，包含2.29亿个数据点并涵盖芬兰湖泊数据，为海事活动和船舶行为研究提供了宝贵资源。

    

    这篇研究论文通过引入一个来自芬兰水域（特别是波罗的海区域）的综合性AIS数据集，为海事研究界做出了贡献。AIS数据最初是为防止船舶碰撞而设计的，如今已发展成为一种多功能工具，在众多海事领域得到广泛应用。本文不仅对现有的AIS数据集进行了整理和分类，还引入了一个从波罗的海区域收集的AIS数据集——该区域以其洲际货运航线、军事活动和冰冻水域而闻名。该数据集包含2.29亿个数据点，为研究人员研究这一动态区域的海事活动和船舶行为提供了资源，其显著特色是包含了来自芬兰湖泊的数据。我们的分析包括对船舶类型和相关特征的详细列表，使研究人员能够探索各种海事领域。

    arXiv:2609.12938v1 Announce Type: new  Abstract: This research paper contributes to the maritime research community by introducing a comprehensive AIS dataset from Finnish waters, specifically the Baltic Sea region. AIS data, initially designed for collision prevention, have evolved into a versatile tool with applications across diverse maritime domains. Our paper not only curates and categorises existing AIS datasets but also introduces a collected AIS dataset from the Baltic Sea area, renowned for its intercontinental cargo routes, military activities, and frozen water expanses. This dataset includes 229 millions data points and provides researchers with a resource for studying maritime activities and vessel behaviour in this dynamic region, notably distinguished by the inclusion of data from Finnish lakes. Our analysis includes a detailed listing of ship types and relevant features, empowering researchers to explore various maritime domains. To enhance comprehension and analysis, we
    
[^22]: 剖析Nvidia Hopper架构上LLM推理的GPU利用率

    Dissecting GPU Utilization for LLM Inference on Nvidia Hopper

    [https://arxiv.org/abs/2609.12923](https://arxiv.org/abs/2609.12923)

    论文揭示单一的SM利用率指标会掩盖LLM推理中GPU真实的工作情况，并提出了八个基于NCU计数器验证的多维视图，以更精确地剖析Hopper GPU上LLM推理的性能瓶颈。

    

    单一的SM利用率百分比可能会让LLM推理工作负载看起来已经达到计算饱和，同时却掩盖了实际完成的有效工作量。问题不在于该计数器本身有误，而在于它将多种不同的机制压缩成了一个数字。这一问题在解码阶段最为严重，因为此时每个请求仅产生一个新token，稠密投影GEMM退化为小行数的矩阵乘法。在Hopper架构上，bfloat16 GMMA路径以固定的64行矩阵片段执行这些操作，因此小批量解码只能用真实token行填充每个片段的一小部分。在本文中，我们在H100 NVL上对使用FlashAttention-3和cuBLASLt的vLLM进行了性能剖析，涵盖冷预填充、热预填充和解码三个阶段，并遍历了不同的序列长度和批量大小。我们将通常使用的单一利用率数字替换为八个经计数器验证的视图，这些视图源自原始的Nsight Compute报告，每个视图都对应一个NCU计数器或明确的计算公式。

    arXiv:2609.12923v1 Announce Type: cross  Abstract: A single SM utilization percentage can make an LLM inference workload look compute-saturated while hiding how much useful work is being done. The problem is not that the counter is wrong, but that it collapses several different mechanisms into one number. This is most severe during decode, where each request contributes only one new token and dense projection GEMMs become small-row matrix multiplications. On Hopper, the bfloat16 GMMA path executes these operations in fixed 64-row matrix fragments, so small-batch decode can fill only a small fraction of each fragment with real token rows.   In this paper, we profile vLLM with FlashAttention-3 and cuBLASLt on an H100 NVL across cold prefill, warm prefill, and decode, sweeping sequence length and batch size. We replace the usual single utilization number with eight counter-validated views derived from raw Nsight Compute reports, each pinned to an NCU counter or explicit formula. Together,
    
[^23]: PA-CDM：用于评估手写数学表达式识别的位置感知字符检测匹配

    PA-CDM: Position-Aware Character Detection Matching for Evaluating Handwritten Mathematical Expression Recognition

    [https://arxiv.org/abs/2609.12917](https://arxiv.org/abs/2609.12917)

    提出了PA-CDM位置感知评估指标，通过将字符检测匹配与位置森林编码和散度级加权相结合，解决了现有手写数学表达式识别评估方法无法感知错误发生位置的缺陷。

    

    手写数学表达式识别（HMER）传统上通过精确匹配率和字符串相似度指标进行评分，这些指标无法感知错误发生的位置：两个具有相同标记错误数的预测，无论是下标放错位置还是交换了分数的操作数，都会得到相同的评分。基于渲染的字符检测匹配（CDM）能够稳健地对齐字形，但仍然对位置不敏感——在受控的分数操作数交换测试中，其得分为0.8595，而位置感知评分应为0.6253。树编辑指标则表现出互补的盲点：解析器规范化覆盖范围之外的重写会被当作结构错误而受到惩罚（得分为0.8552，而基于渲染的指标得分为1.0）。我们提出了PA-CDM，这是一种位置感知指标，它将字符检测匹配与位置森林编码和散度级加权相结合；同时提出了StructPerturb v2.0，这是一个涵盖15种类型、共1,340对受控扰动的冻结基准测试集。

    arXiv:2609.12917v1 Announce Type: cross  Abstract: Handwritten mathematical expression recognition (HMER) is conventionally scored by exact-match rates and string-similarity metrics that are blind to where an error occurs: two predictions with identical token-error counts receive identical scores whether they misplace a subscript or swap the operands of a fraction. Render-based character detection matching (CDM) aligns glyphs robustly but remains position-blind---on controlled fraction-operand swaps it scores 0.8595 where position-aware scoring yields 0.6253. Tree-edit metrics exhibit a complementary blind spot: rewrites outside the parser's normalization coverage are penalized as structural errors (0.8552 where render-based metrics score 1.0). We propose PA-CDM, a position-aware metric that couples character detection matching with position-forest encoding and divergence-level weighting; StructPerturb v2.0, a frozen benchmark of 1,340 controlled perturbation pairs across 15 type--inte
    
[^24]: 风电场控制的离线强化学习：动态风向下的风洞研究

    Offline Reinforcement Learning for Wind Farm Control: A Wind Tunnel Study under Dynamic Wind Directions

    [https://arxiv.org/abs/2609.12905](https://arxiv.org/abs/2609.12905)

    提出了一种离线强化学习算法MTD3-BC，通过在策略优化中引入新的动作一致性项实现平滑的偏航控制，在动态风向条件下有效最大化风电场功率，且无需与模拟器大量交互，显著降低了计算成本和训练时间。

    

    本文研究了风向变化情况下的风电场功率最大化问题。具体而言，提出了一种无模型的改进型双延迟深度确定性策略梯度结合行为克隆（MTD3-BC）算法，通过偏航控制在变化的风向条件下完成该任务。MTD3-BC是一种离线强化学习（RL）算法，旨在仅从预先收集的离线数据集中推断出良好的行为策略。此外，为确保偏航调整的平滑性和适度性，在策略优化目标中引入了一个新的动作一致性项。与在线强化学习方法不同，MTD3-BC在训练过程中不需要与风电场模拟器进行大量交互，显著降低了计算成本和训练时间。通过风洞实验验证了该算法在变化风向条件下的有效性。结果表明，MTD3-BC成功地……

    arXiv:2609.12905v1 Announce Type: new  Abstract: This paper addresses the wind farm power maximization problem in the presence of wind direction changes. Specifically, a model-free Modified Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning (MTD3-BC) algorithm is proposed to tackle this task through yaw control under varying wind direction conditions. MTD3-BC is an offline reinforcement learning (RL) algorithm that aims to infer good behavior from only a precollected offline dataset. Additionally, to ensure smooth and moderate yaw adjustments, a new action consistency term is introduced into the policy optimization objective. Unlike online RL methods, MTD3-BC does not require extensive interactions with a wind farm simulator during training, significantly reducing computational costs and training time. A wind tunnel experiment is conducted to validate the effectiveness of the algorithm under varying wind directions. The results demonstrate that MTD3-BC successfully m
    
[^25]: 隐藏在轮次中：预测联邦学习中802.11信道竞争的时间成本

    Hidden in Rounds: Predicting the Time Cost of 802.11 Contention in Federated Learning

    [https://arxiv.org/abs/2609.12903](https://arxiv.org/abs/2609.12903)

    本文提出一种结合ns-3测量与Bianchi模型估计器的方法来预测联邦学习中802.11信道竞争的时间成本，发现达到目标精度所需的通信时间随客户端密度增加约两个数量级，而所需训练轮次几乎不变。

    

    基于IEEE 802.11的联邦学习在发送模型更新的客户端之间共享无线信道。我们使用ns-3测量了不同客户端密度和负载条件下的帧传递率和饱和吞吐量。一个独立的FedAvg训练器将帧传递率作为更新准入概率的一阶代理，并使用方程来估计通信时间。该方法不模拟完整模型更新的传输，也不测量端到端的训练时间。在涵盖两个数据集、两种数据划分、六种客户端密度、六种负载和五个随机种子的共720次评估运行中，所有运行都在轮次预算内达到了预定的目标精度。达到目标精度所需的轮次随负载变化不大，而达到目标的通信时间在客户端密度范围内增加了约两个数量级。基于Bianchi模型的估计器产生了2.3%至1…（摘要截断）的平均绝对百分比误差。

    arXiv:2609.12903v1 Announce Type: new  Abstract: Federated learning over IEEE~802.11 shares the wireless channel among clients that send model updates. We use ns-3 to measure the frame-delivery ratio and saturation throughput for different client densities and offered loads. A separate FedAvg trainer uses the frame-delivery ratio as a first-order proxy for the update-admission probability and uses an equation to estimate communication time. The method does not simulate the delivery of a complete model update or measure end-to-end training time. Across 720 evaluated runs with two datasets, two data partitions, six client densities, six offered loads, and five seeds, all runs reached their predefined target accuracy within the round budget. Rounds-to-target changed little with offered load, while communication time-to-target increased by about two orders of magnitude across the client-density range. A Bianchi-anchored estimator produced a mean absolute percentage error from $2.3\%$ to $1
    
[^26]: 物理增强的神经求解器用于瞬态冰川流动模拟

    Physics-enriched neural solvers for transient ice-flow simulation

    [https://arxiv.org/abs/2609.12900](https://arxiv.org/abs/2609.12900)

    提出一种物理增强方法，通过将低阶冰流平衡推导的低成本物理场作为神经网络输入来改进在线冰川流动神经求解器，无需修改损失函数即可显著提升求解器鲁棒性并改善精度与运行时间的权衡。

    

    瞬态冰川模拟中的高阶冰流计算需要在几何形状演化过程中反复求解非线性问题。在指导性冰川模型的在线模式中，速度场由一个神经网络表示，其权重从上一个时间步热启动，并通过少量优化器迭代进行更新。我们证明，向网络提供从低阶冰流平衡推导出的低成本输入场可以改进这种在线求解器。与基于残差的物理信息神经网络不同——后者通过损失函数中的控制方程惩罚项来引入物理约束——我们的方法保持原有的能量控制目标不变，而是通过网络输入来添加物理结构。在三个真实世界的冰川配置中，增强后的求解器对求解器设置表现出显著更强的鲁棒性。在两个高山案例中，它还改善了经过调优的精度与运行时间的权衡，降低了表面速度误差……

    arXiv:2609.12900v1 Announce Type: cross  Abstract: Transient glacier simulations with higher-order ice flow require the repeated solution of a nonlinear problem as the geometry evolves. In the online mode of the Instructed Glacier Model, the velocity field is represented by a neural network whose weights are warm-started from the previous time step and updated with a few optimizer iterations. We show that supplying the network with inexpensive input fields derived from low-order ice-flow balances improves this online solver. Unlike residual-based physics-informed neural networks, which incorporate physics through governing-equation penalties in the loss, our approach leaves the governing energy objective unchanged, adding physical structure through the network inputs. Across three real-world glacier configurations, the enriched solver is markedly more robust to solver settings. On the two alpine cases, it also improves the tuned accuracy--runtime trade-off, reducing surface-velocity er
    
[^27]: 面向全波形反演的物理状态引导扩散采样

    Physical-State-Guided Diffusion Sampling for Full-Waveform Inversion

    [https://arxiv.org/abs/2609.12899](https://arxiv.org/abs/2609.12899)

    提出物理状态引导扩散采样方法（PSG），通过高斯桥将物理速度状态与扩散先验耦合，实现波动方程梯度与去噪器梯度的分离，在全波形反演中超越了经典方法和基于扩散的基线方法。

    

    全波形反演（FWI）从地震记录中估计地下速度结构，但其病态性和非线性使得精确重建高度依赖于初始化和先验信息。扩散后验采样能够提供学习得到的地质先验，但将其去噪器与非线性波动方程求解器直接耦合可能产生不可靠的物理引导。我们提出了物理状态引导扩散采样（PSG），通过高斯桥将一个持续维护的物理速度状态与扩散先验相耦合。物理状态通过由去噪速度正则化的波形拟合进行更新优化，进而引导逆向扩散过程。这一框架将波动方程梯度与去噪器梯度相分离，同时保留了传统FWI的初始化方式和累积优化历史。在四个OpenFWI数据集族上，PSG的终端去噪估计结果在常规条件下优于经典方法和基于扩散的基线方法。

    arXiv:2609.12899v1 Announce Type: new  Abstract: Full waveform inversion (FWI) estimates subsurface velocity from seismic recordings, but its ill-posedness and nonlinearity make accurate reconstruction strongly dependent on initialization and prior information. Diffusion posterior sampling provides a learned geological prior, yet directly coupling its denoiser to the nonlinear wave solver can yield unreliable physical guidance. We propose Physical-State-Guided Diffusion Sampling (PSG), which couples a persistent physical velocity to the diffusion prior through a Gaussian bridge. The physical state is refined by waveform fitting regularized by the denoised velocity, and in turn guides the reverse diffusion process. This formulation separates the wave-equation and denoiser gradients while preserving conventional FWI initialization and accumulated optimization history. On four OpenFWI families, PSG's terminal denoised estimates outperform classical and diffusion-based baselines under clea
    
[^28]: 面向LLM智能体低秩适应的行为商学习

    Behavior Quotient Learning for Low-Rank Adaptation of LLM Agents

    [https://arxiv.org/abs/2609.12896](https://arxiv.org/abs/2609.12896)

    提出BQ-LoRA框架，通过局部行为商流形组织轨迹更新，使单个LoRA适配器能够高效学习LLM智能体的多样化交互能力，同时避免多适配器带来的存储和路由开销。

    

    基于大语言模型（LLM）的智能体依赖异构的交互能力来完成复杂任务。现有方法通常将这些能力分布在多个LoRA适配器上，这增加了适配器的存储需求，并在推理过程中引入了路由开销。使用单个LoRA可以避免这种开销，但在固定的秩预算下从多样化的智能体轨迹中学习面临两个挑战。首先，具有不同交互轨迹和参数梯度的轨迹可能在决策分布上诱导等效的变化，导致重复更新过度强调冗余的行为变化。其次，聚合的更新可能超出适配器的秩预算，在权重空间中对其进行近似可能会扭曲其原本旨在产生的决策变化。我们提出了BQ-LoRA，这是一种通过局部行为商流形来组织轨迹更新的低秩适应框架。该框架包含两个模块，即……（摘要在此处截断）

    arXiv:2609.12896v1 Announce Type: new  Abstract: LLM-based agents rely on heterogeneous interaction capabilities to accomplish complex tasks. Existing approaches often distribute these capabilities across multiple LoRA adapters, which increases adapter storage requirements and introduces routing overhead during inference. A single LoRA avoids this overhead, but learning from diverse agent trajectories under a fixed rank budget presents two challenges. First, trajectories with different interaction traces and parameter gradients can induce equivalent changes in decision distributions, causing repeated updates to overemphasize redundant behavioral changes. Second, an aggregated update may exceed the rank budget of the adapter, and approximating it in weight space can distort the decision changes that it is intended to produce. We propose BQ-LoRA, a low-rank adaptation framework that organizes trajectory updates through a local behavior quotient manifold. It contains two modules, i.e., be
    
[^29]: 超越准确率：面向可靠生物医学图像分割的不确定性引导边界细化

    Beyond Accuracy: Uncertainty-Guided Boundary Refinement for Reliable Biomedical Image Segmentation

    [https://arxiv.org/abs/2609.12892](https://arxiv.org/abs/2609.12892)

    提出可靠性感知边界细化网络RABR-Net，通过融合预测熵、测试时增强方差等多种不确定性度量构建边界感知的可靠性表示，选择性地纠正不确定的边界像素，从而在保持高分割准确率的同时实现可靠的生物医学图像分割。

    

    准确的生物医学图像分割不仅需要较高的全局重叠度，还需要对具有临床意义的边界进行可靠描绘。在血涂片显微镜检查中，细胞质和细胞核轮廓为下游形态学分析提供了结构基础；然而，深度分割模型即使取得了较高的Dice分数，在模糊的边界区域附近仍可能保持不确定或过度自信。本工作提出了一种可靠性感知边界细化网络（RABR-Net），这是一个用于可信赖图像分割的两阶段框架。首先，一个强大的UNet++ EfficientNet-B4基础分割器产生初始类别概率和logits。然后，将预测熵、测试时增强方差、边际不确定性、概率梯度和软边界线索融合为一个边界感知的可靠性表示。该表示引导一个门控残差细化器，选择性地纠正不确定的边界像素。

    arXiv:2609.12892v1 Announce Type: cross  Abstract: Accurate biomedical image segmentation requires not only high global overlap but also reliable delineation of clinically meaningful boundaries. In blood-smear microscopy, cytoplasm and nucleus contours provide the structural basis for downstream morphology analysis; however, deep segmentation models may remain uncertain or overconfident near ambiguous boundary regions even when achieving strong Dice scores. This work proposes a Reliability-Aware Boundary Refinement Network (RABR-Net), a two-stage framework for trustworthy image segmentation. A strong UNet++ EfficientNet-B4 base segmenter first produces initial class probabilities and logits. Predictive entropy, test-time augmentation variance, margin uncertainty, probability gradients, and soft boundary cues are then combined into a boundary-aware reliability representation. This representation guides a gated residual refiner that selectively corrects uncertain boundary pixels while pr
    
[^30]: 使用PAC-贝叶斯方法量化特权信息的价值

    Quantifying the Value of Privileged Information Using a PAC-Bayesian Approach

    [https://arxiv.org/abs/2609.12891](https://arxiv.org/abs/2609.12891)

    该论文提出了一种基于PAC-贝叶斯框架的算法无关信息论方法，用于量化特权信息（LUPI范式）在训练中传递的价值，填补了缺乏解释特权信息如何及何时传递有用知识的通用理论框架这一空白。

    

    在实践中，许多学习场景仅在训练阶段提供对辅助特征的访问。将这类数据融入模型训练以提升性能，催生了一种被称为“使用特权信息学习”（LUPI）的范式。尽管这些额外信息旨在改善最终模型，但建立一个关于特权信息（PI）如何传递有用知识的普适、连贯的理解仍然是一个挑战。Vapnik的原始理论及后续工作在某些特定情况下提供了性能保证，但这些结果本质上是针对特定算法的，且依赖于特定设置下的证明方法。因此，一个能够解释特权信息如何以及何时传递有用知识的更通用框架仍然缺失。为弥合这一差距，我们提出了一种基于PAC-贝叶斯框架的算法无关、信息论方法。我们不问某个特定算法是否利用了特权信息，而是问它能带来多大价值……

    arXiv:2609.12891v1 Announce Type: new  Abstract: In practice, various learning scenarios provide access to auxiliary features exclusively during training. Incorporating such data to enhance model performance gave rise to a paradigm known as Learning Using Privileged Information (LUPI). While this extra information is intended to improve the resulting model, establishing a generalized, cohesive understanding of how privileged information (PI) transfers useful knowledge remains a challenge. Vapnik's original theory and subsequent works offer performance guarantees in certain cases, but these results are inherently per-algorithm and rely on setting-specific proof approaches. Consequently, a more general framework explaining how and when PI transfers useful knowledge is still missing. To bridge this gap, we introduce an algorithm-agnostic, information-theoretic approach based on the PAC-Bayes framework. Rather than asking whether a particular algorithm exploits PI, we ask how much value it
    
[^31]: 远距离大梯度未必可靠：面向长时程自回归预测的可靠性加权信用分配

    Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting

    [https://arxiv.org/abs/2609.12890](https://arxiv.org/abs/2609.12890)

    提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。

    

    在自回归预测中，长预测展开能够提供远距离的监督信号，但通过时间的反向传播（BPTT）需要将这些损失的梯度经过许多自回归步骤逐步传递。反复的雅可比矩阵乘积可能使远距离梯度在参数更新中占据主导地位，同时放大可预测信号与不可预测噪声；因此，大的远距离梯度并不一定携带可靠的学习信号。基于这一观察，我们提出了内部双维纳路由（Internal-DW），这是一种仅作用于反向传播过程的原则性干预方法，它在保留完整前向展开和所有时域损失的同时，对内部梯度路由进行可靠性加权。在每个残差块处，我们为恒等路由和非线性路由推导出有界的维纳增益，以在保留可预测学习信号与抑制不可预测变化之间取得平衡，并通过路由级别的梯度统计量和显式噪声模型对这些增益进行估计。在一个受控的（摘要在此处截断）

    arXiv:2609.12890v1 Announce Type: new  Abstract: In autoregressive forecasting, long prediction rollouts provide distant supervision, but backpropagation through time (BPTT) carries gradients from those losses through many autoregressive steps. Repeated Jacobian products can make distant gradients dominate the update while amplifying predictable signal and unpredictable noise together; a large distant gradient therefore need not carry reliable learning signal. Motivated by this observation, we introduce Internal Dual-Wiener routing (Internal-DW), a principled backward-only intervention that preserves the full forward rollout and all horizon losses while reliability-weighting internal gradient routes. At each residual block, we derive bounded Wiener gains for the identity and nonlinear routes that balance preserving predictable learning signal against suppressing unpredictable variation, and estimate them from route-level gradient statistics and an explicit noise model. In a controlled 
    
[^32]: 气味描述词语料库能测量什么与不能测量什么：效价、衰减与公共记录的上限

    What an odour descriptor corpus can and cannot measure: valence, attenuation, and the ceiling of the public record

    [https://arxiv.org/abs/2609.12875](https://arxiv.org/abs/2609.12875)

    本研究首次系统审计了四个公共气味描述词语料库，发现各语料库对同一描述词的使用存在无法用单一校准修复的异质性差异，并揭示当前基于分子指纹的模型仅能达到人类专家小组可靠性的约33%，从而界定了公共气味数据记录所能测量的上限。

    

    机器嗅觉模型在汇集的公共描述词语料库上进行训练，但一个共享的描述词在不同语料库中是否测量同一事物尚未被检验过，这些语料库所能测量的上限也未被确定。我们审计了来自Pyrfume的四个语料库。以分子为条件化处理使McNemar检验成为语料库效应的精确条件检验。各语料库在不同描述词上存在异质性分歧（I² = 80%），且与标注广度呈非均匀关系（z = 17.2），因此没有任何单一的偏移量能够修复数据汇集问题。中位四分相关一致性为0.795，而中位κ仅为0.413：各来源在“哪些分子值得使用某个词”这一点上基本一致，但在“使用该词的容易程度”上存在差异。在109个效应可估计的描述词中，有36个在ETS量表上表现出较大的差异功能。与人类专家小组的可靠性相比，结合完整RDKit描述符块的Morgan分子指纹模型达到了可实现性能的32.9%；而加入来自两个合并语料库的所有标签……（摘要在此处截断）

    arXiv:2609.12875v1 Announce Type: new  Abstract: Machine olfaction trains on pooled public descriptor corpora, but whether a shared descriptor word measures the same thing across corpora has not been tested, nor has the ceiling of what any of them can measure.   We audit four corpora from Pyrfume. Conditioning on the molecule makes McNemar's test the exact conditional test of the corpus effect. Corpora disagree heterogeneously across descriptors ($I^2 = 80\%$) and non-uniformly with labelling breadth ($z = 17.2$), so no single offset repairs pooling. Median tetrachoric agreement is 0.795 against median $\kappa$ of 0.413: sources largely concur on which molecules deserve a word and differ on how readily they apply it. Of 109 descriptors with an estimable effect, 36 show large differential functioning on the ETS scale.   Against a human panel's reliability, Morgan fingerprints with the full RDKit descriptor block reach 32.9\% of achievable; adding every label from two merged corpora reac
    
[^33]: 基于RTK-GNSS的包含相机、激光雷达与雷达传感器以及扫描3D模型的多车辆数据集，用于自定义自动标注

    A Multi-Vehicle Dataset with Camera, LiDAR, and Radar Sensors and Scanned 3D Models for Custom Auto-Annotation using RTK-GNSS

    [https://arxiv.org/abs/2609.12871](https://arxiv.org/abs/2609.12871)

    该论文提出了一个多车辆数据集，包含相机、激光雷达和雷达传感器数据，并结合扫描3D模型与RTK-GNSS位姿参考，支持用户自定义粒度的自动标注，可深入评估遮挡等测量效应。

    

    数据集是感知算法开发中的关键要素。它们将传感器测量数据与标注的参考信息相关联，从而能够推导传感器和目标物体的特性。在自动驾驶领域，参考数据通常包括语义图像分割、逐点关联或边界框标注。然而，本工作提出的数据集旨在更深入地评估测量原理，提供了所有车辆的扫描3D模型，以及通过RTK-GNSS获得的位姿和连续运动学参考。二者结合后，传感器车辆周围完整动态环境的状态在任意时间点均为已知，后续的参考格式可以按用户自定义的粒度轻松计算得到。该数据集包含对七辆目标车辆的单目标与多目标记录，尤其涵盖了诸如遮挡以及反射特性等测量效应。

    arXiv:2609.12871v1 Announce Type: cross  Abstract: Datasets are a crucial element in the development of perception algorithms. They relate sensor measurement data to annotated reference information and allow for the deduction of sensor and object characteristics. In autonomous driving, the reference data commonly consist of semantic image segmentation, point-wise associations, or bounding box annotations. The dataset proposed in this work, however, aims to dig deeper into the evaluation of measurement principles and provides scanned 3D models of all vehicles together with a pose and continuous kinematics reference obtained by RTK-GNSS. Combined, the state of the complete dynamic surrounding of the sensor vehicle is known for any point in time. Subsequent reference formats can be easily computed in user-defined granularity. This dataset involves single-object and multi-object recordings with seven target vehicles. In particular, measurement effects such as occlusion, as well as reflecti
    
[^34]: GenOR-Twin：一种整合运营话语与数学优化的语义中间件

    GenOR-Twin: A Semantic Middleware for Integrating Operational Discourse with Mathematical Optimization

    [https://arxiv.org/abs/2609.12863](https://arxiv.org/abs/2609.12863)

    GenOR-Twin是一个神经符号框架，它将大语言模型定位为语义翻译器而非直接求解器，通过动态约束注入机制将定性运营事件实时转化为数学约束，实现运营话语与数学优化之间的双向耦合，从而构建满足同步要求的数字孪生系统。

    

    我们提出了GenOR-Twin，这是一个神经符号框架，旨在弥合非结构化运营日志与严格数学优化之间的转换鸿沟。我们的架构独特地将大语言模型定位为语义翻译器而非直接求解器，从而确保系统保留精确组合优化方法的可行性保证。我们设计了一种动态约束注入机制（即在运行时将定性中断事件转换为形式化数学约束），使系统能够根据定性的人工输入，实时地对优化问题的可行域进行结构性修改。由此产生的双向耦合——运营观察更新虚拟模型状态，优化后的决策则反馈至知识图谱——满足了真正数字孪生的同步要求。该框架具有自适应决策……（摘要原文在此处截断）

    arXiv:2609.12863v1 Announce Type: new  Abstract: We introduce GenOR-Twin, a neuro-symbolic framework that bridges the translation gap between unstructured operational logs and rigorous mathematical optimization. Our architecture uniquely positions Large Language Models as semantic translators rather than direct solvers, ensuring that the system retains the feasibility guarantees of exact combinatorial methods. { \color{red}We design a dynamic constraint injection mechanism (the runtime translation of qualitative disruption events into formal mathematical constraints) that allows the system to structurally modify the optimization problem's feasibility region in real-time based on qualitative human inputs. The resulting bidirectional coupling---where operational observations update the virtual model state and optimized decisions are reflected back into the Knowledge Graph---satisfies the synchronization requirement of a proper Digital Twin. The framework features an adaptive decision pol
    
[^35]: 非常令人兴奋：基于激励式广义迁移学习模型的建筑零样本模型预测控制

    Very Exciting: Zero-Shot Model Predictive Control of Buildings via Excitation-Based Generalized Transfer Learning Models

    [https://arxiv.org/abs/2609.12853](https://arxiv.org/abs/2609.12853)

    该论文提出使用基于激励式探测数据的广义迁移学习模型，实现了建筑的零样本模型预测控制，无需在目标建筑收集数据即可获得令人满意的控制性能。

    

    数据驱动、节能的模型预测控制（MPC）在建筑中的广泛应用，仍然受到为单个建筑收集数据和训练模型所需大量工作的阻碍。因此，迁移学习（TL）在目标建筑建模方面受到越来越多的关注，因为它通过复用预训练的源模型来减少数据需求和建模工作量。然而，这些迁移学习模型通常仅在目标建筑上的预测精度进行评估，而未测试其下游控制性能。为解决这一空白，我们将一种最先进的迁移学习方法——使用标准运行数据在多个源建筑上预训练一个广义模型——应用于目标建筑的MPC设置中。我们表明，这种方法不足以实现令人满意的控制性能。作为解决方案，我们引入了在基于激励的运行源数据上预训练的广义模型——即有目的地探测的输入信号……

    arXiv:2609.12853v1 Announce Type: cross  Abstract: The widespread adoption of data-driven, energy-efficient model predictive control (MPC) in buildings remains hindered by substantial effort to collect data and train models for individual buildings. Transfer learning (TL) has consequently gained increasing attention for target building modeling, as it reduces data requirements and modeling effort by reusing pretrained source models. However, these TL models are typically evaluated only on prediction accuracy in the target, without testing downstream control performance. To address this gap, we apply a state-of-the-art TL approach - pretraining a generalized model on multiple source buildings using standard operational data - within an MPC setup in a target building. We show that this approach is insufficient to achieve satisfactory control performance. As a solution, we introduce generalized models pretrained on excitation-based operational source data - purposefully probed inputs that
    
[^36]: VertexCBF：通过顶点受限控制搜索改进神经控制屏障函数

    VertexCBF: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search

    [https://arxiv.org/abs/2609.12831](https://arxiv.org/abs/2609.12831)

    提出VertexCBF框架，利用控制仿射动力学和凸多面体控制集的性质（哈密顿量在控制顶点处最大化），结合物理信息学习、稀疏监督学习与GPU并行的顶点受限树搜索，以可扩展、系统化且可解释的方式训练神经控制屏障函数。

    

    随着自主机器人数量的持续增长，安全性变得日益重要。控制屏障函数（CBFs）为保障安全性提供了一个有理论依据的框架，但现有的设计方法往往在有效性、可扩展性或可解释性方面存在局限，并且可能导致过于保守的安全集。在本文中，我们提出了VertexCBF，这是一种以可扩展、系统化且可解释的方式学习神经控制屏障函数的框架。我们使用神经网络来近似平稳的Hamilton–Jacobi值函数，该网络通过物理信息学习与稀疏监督学习相结合的方式进行训练。通过利用控制仿射动力学和凸多面体控制集——在此条件下哈密顿量在控制顶点处取得最大值——我们通过GPU并行的顶点受限树搜索高效生成监督点，同时残差架构保证了所学习到的CBF绝不会大于（原文此处截断）。

    arXiv:2609.12831v1 Announce Type: cross  Abstract: As the number of autonomous robots continues to grow, safety becomes increasingly important. Control barrier functions (CBFs) provide a theoretically grounded framework for ensuring safety, but existing design methods often face limitations in effectiveness, scalability, or interpretability, and may result in overly conservative safe sets. In this paper, we propose \emph{VertexCBF}, a framework for learning neural CBFs in a scalable, systematic, and explainable way. We approximate the stationary Hamilton--Jacobi value function using a neural network trained via a combination of physics-informed and sparsely supervised learning. By exploiting control-affine dynamics and a convex polytope control set, under which the Hamiltonian is maximized at the control vertices, we efficiently generate supervision points via GPU-parallel vertex-restricted tree search, while a residual architecture guarantees that the learned CBF is never larger than 
    
[^37]: 四维并行技术解锁艾可萨级贝叶斯神经网络，实现高保真大气建模

    4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling

    [https://arxiv.org/abs/2609.12815](https://arxiv.org/abs/2609.12815)

    BEAST是首个能在0.25°全球分辨率下进行大气预报并同时量化偶然与认知不确定性的贝叶斯Swin Transformer，其创新的四维并行方案（域-张量并行与不确定性并行）在20,480块GPU上实现了3.96 EFLOP/s的峰值性能，预测技巧可与最先进的概率AI模型和数值模型相媲美。

    

    我们提出了BEAST，这是首个用于0.25°全球分辨率大气预报的贝叶斯Swin Transformer，能够准确量化偶然不确定性和认知不确定性。为了克服由此带来的计算瓶颈，我们设计了一种正交的四维并行化方案，引入了独特的域-张量并行策略和一种新颖的不确定性并行方法，使我们能够充分利用GPU算力并高效扩展模型训练。对于一个24亿参数的模型，我们在JUPITER超级计算机的20,480块NVIDIA GH200 GPU上实现了3.96 EFLOP/s的峰值性能。我们将BEAST训练为7亿参数模型，采用96个随机权重样本，在384个节点上基于40年的数据进行近一百万次梯度更新。该模型取得了与最先进的概率性大气AI模型和数值模型相当的预测技巧分数，并能以出色的能力预测极端事件。

    arXiv:2609.12815v1 Announce Type: cross  Abstract: We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To overcome the associated computational bottlenecks, we devise an orthogonal 4D-parallelization scheme that introduces a unique domain-tensor-parallelism strategy and a novel uncertainty parallel method, enabling us to fully leverage GPU capacity and efficiently scale model training. For a 2.4-billion-parameter model, we achieve a peak performance of 3.96 EFLOP/s on 20,480 NVIDIA GH200 GPUs on the JUPITER supercomputer. We train BEAST as a 700-million-parameter model with 96 random weight samples on 384 nodes on 40 years of data for nearly one million gradient updates. This model achieves predictive skill scores competitive with state-of-the-art probabilistic atmospheric AI models and numerical models, and can predict extreme events with exception
    
[^38]: RunningTensor：将线性注意力推广到高阶循环状态

    RunningTensor: Generalizing Linear Attention to Higher-Order Recurrent States

    [https://arxiv.org/abs/2609.12814](https://arxiv.org/abs/2609.12814)

    本文提出 RunningTensor，将线性注意力的矩阵循环记忆推广为高阶（o 阶）张量状态，在保持对序列长度线性的时间复杂度下，将工作记忆容量从 O(W²) 提升到 O(W^o)，并在联想回忆、语言理解和检索任务上均优于基线方法。

    

    线性注意力和状态空间模型提供了线性时间的序列建模能力，但它们的循环记忆仍然是一个二阶张量（即矩阵），限制了状态中可表示的交互阶数。我们提出了 RunningTensor，将这种记忆推广为 $o$ 阶张量，通过秩-1 外积进行更新，并通过与 $o-1$ 个向量查询的缩并操作进行读取。当阶数为 2 时可恢复线性注意力；我们以阶数 3 作为概念验证进行研究，在保留循环和并行两种形式的同时，对序列长度 $T$ 保持线性复杂度，并将工作记忆容量从 $\mathcal{O}(W^2)$ 提升到 $\mathcal{O}(W^o)$。在合成的多查询联想回忆任务上，RunningTensor 优于线性注意力和状态空间模型基线。经过预训练后，它在语言理解和非合成检索任务上也取得了性能提升，这表明高阶循环状态能够提供有用的额外记忆容量。

    arXiv:2609.12814v1 Announce Type: new  Abstract: Linear attention and state-space models provide linear-time sequence modeling, but their recurrent memory remains a second-order tensor (a matrix), limiting the order of interactions that can be represented in the state. We introduce the RunningTensor, which generalizes this memory to an order-$o$ tensor, updated by a rank-1 outer product and read by contracting against $o-1$ vector queries. Order $2$ recovers linear attention; we study order $3$ as a proof of concept, retaining both recurrent and parallel forms while remaining linear in sequence length $T$ and improving working memory capacity from $\mathcal{O}(W^2)$ to $\mathcal{O}(W^o)$. On synthetic multi-query associative recall, RunningTensor outperforms linear-attention and SSM baselines. After pretraining, it also improves performance on language-understanding and non-synthetic retrieval tasks, suggesting that higher-order recurrent state can provide useful additional memory capa
    
[^39]: VertiFuseX：基于多流时序融合的可泛化金融预测

    VertiFuseX: Generalizable Financial Forecasting via Multi-Stream Temporal Fusion

    [https://arxiv.org/abs/2609.12793](https://arxiv.org/abs/2609.12793)

    VertiFuseX提出一种混合LSTM架构，通过对LSTM、Bi-LSTM和St-LSTM分支的倒数第二层特征进行垂直融合，在固定超参数配置下实现了更具信息保留能力和跨市场泛化性的股票价格预测。

    

    股票价格预测因金融时间序列的非平稳性和噪声特性而始终充满挑战。现有的深度学习模型通常依赖于刚性的决策层融合、临时性的超参数调整以及压缩的最终层输出，导致信息损失、过拟合以及有限的跨市场泛化能力。我们提出了VertiFuseX，一种采用倒数第二层垂直融合多尺度时序表示的混合LSTM架构。VertiFuseX对来自LSTM、Bi-LSTM和St-LSTM分支的倒数第二层特征进行堆叠和重新加权，整合了并行的DNN流，并在固定的超参数配置下通过反向传播对所有组件进行联合优化。这保留了跨尺度更丰富的中间时序信息。在10个全球股票指数15年（2010-2024年）的收盘价数据上进行评估，采用严格的按时间顺序的样本外测试（保留最后365个交易日作为测试集），VertiFuseX实现了……（原文摘要在此处截断）

    arXiv:2609.12793v1 Announce Type: new  Abstract: Stock price prediction remains challenging due to the non-stationary and noisy nature of financial time series. Existing deep learning models often rely on rigid decision-level fusion, ad hoc hyperparameter tuning, and compressed final-layer outputs, causing information loss, overfitting, and limited cross-market generalization. We propose VertiFuseX, a hybrid LSTM architecture using penultimate-layer vertical fusion of multi-scale temporal representations. VertiFuseX stacks and reweights penultimate features from LSTM, Bi-LSTM, and St-LSTM branches, integrates a parallel DNN stream, and jointly optimizes all components via backpropagation under a fixed hyperparameter configuration. This preserves richer intermediate temporal information across scales. Evaluated on 15 years (2010-2024) of closing prices from 10 global equity indices using strict chronological out-of-sample testing with the final 365 trading days held out, VertiFuseX achi
    
[^40]: 重尾噪声与Hölder光滑性下随机梯度方法的收敛性

    Convergence of Stochastic Gradient Methods under Heavy-Tailed Noise and H\"{o}lder Smoothness

    [https://arxiv.org/abs/2609.12785](https://arxiv.org/abs/2609.12785)

    本文在同时放宽Lipschitz光滑性与有限方差噪声这两个经典假设的条件下，证明了标准SGD和$\delta$-正则化梯度裁剪方法在Hölder光滑目标函数和重尾梯度噪声下的非凸收敛速率。

    

    经典随机梯度方法的收敛保证通常假设目标函数是Lipschitz光滑的，且梯度噪声具有有限方差，而这两个假设在实践中经常被违反。相比之下，我们研究了在同时放宽这些假设条件下的非凸随机优化问题：目标函数具有$(L,s)$-Hölder连续梯度，其中$s\in(0,1]$，且梯度噪声仅满足有界的$\alpha$阶矩条件，其中$\alpha\in(1,2]$。我们建立了三个收敛结果。首先，当$\alpha\ge1+s$时，标准SGD以$O(T^{-s/(1+s)})$的速率收敛，这将经典的非凸SGD收敛速率同时扩展到了重尾噪声和Hölder光滑性的情形。其次，我们分析了$\delta$-正则化梯度裁剪（$\delta$-GClip）——一种已被证明能够有效训练宽深神经网络的方法——并在相同条件下建立了$O(T^{-2s(\alpha-1)/[(1+s)(2\alpha-1)]})$的平稳性收敛速率。第三，我们分析了标准梯度裁剪方法（摘要原文在此处截断）

    arXiv:2609.12785v1 Announce Type: new  Abstract: Classical convergence guarantees for stochastic gradient methods typically assume Lipschitz-smooth objectives and finite-variance gradient noise, both frequently violated in practice. In contrast, we study nonconvex stochastic optimization under the joint relaxation of these assumptions: objectives with $(L,s)$-H\"older continuous gradients, $s\in(0,1]$, and gradient noise satisfying only a bounded $\alpha$-th moment condition for $\alpha\in(1,2]$. We establish three convergence results. Firstly, that standard SGD converges at rate $O(T^{-s/(1+s)})$ whenever $\alpha\ge1+s$, extending the classical nonconvex SGD rate to heavy-tailed noise and H\"older smoothness simultaneously. Secondly, we analyze $\delta$-regularized gradient clipping ($\delta$-GClip), a provable trainer of wide and deep nets, and establish a stationarity rate of $O(T^{-2s(\alpha-1)/[(1+s)(2\alpha-1)]})$ under the same condition. Thirdly, we analyze standard gradient cl
    
[^41]: 基于批量更新的SGD高概率收敛性分析

    High-Probability Convergence of SGD via Batched Updates

    [https://arxiv.org/abs/2609.12765](https://arxiv.org/abs/2609.12765)

    本文提出Batched SGD，通过将在线样本分批、每批仅用低方差梯度估计执行一次更新，在无需限制性假设或辅助序列的情况下，以极其简洁的证明获得了SGD最后迭代点的高概率近最优收敛速率。

    

    随机梯度下降（SGD）是大规模优化的主要工具。尽管其迭代点的平均行为（通常由均方误差界来刻画）已被充分理解，但为最后一次迭代获得高概率保证仍然具有挑战性。先前解决这一问题的方法要么施加了限制性假设（如有界域或有界梯度），要么依赖于涉及辅助序列的复杂证明。在本工作中，我们提出了Batched SGD（批量SGD），这是一种简单的变体：它将在线样本划分为多个轮次（epoch），并在每个轮次中利用一种精炼的低方差梯度估计仅执行一次更新。我们的主要贡献表明，这种批处理机制使得高概率分析异常简洁，既避免了限制性假设，也避免了辅助序列。在标准的平滑性和范数次高斯噪声假设下，我们为强凸（及一般凸）情形建立了接近最优的收敛速率。

    arXiv:2609.12765v1 Announce Type: cross  Abstract: Stochastic gradient descent (SGD) is the primary workhorse for large-scale optimization. While the average behavior of its iterates, typically characterized by mean-squared error bounds, is well-understood, obtaining high-probability guarantees for the last iterate remains challenging. Prior approaches to this problem have either imposed restrictive assumptions (such as bounded domains or gradients) or relied on complex proofs involving auxiliary sequences. In this work, we propose Batched SGD, a simple variant that partitions online samples into epochs and performs a single update per epoch using a refined, low-variance gradient estimate. Our main contribution demonstrates that this batching mechanism enables a surprisingly simple high-probability analysis that avoids both restrictive assumptions and auxiliary sequences. Under standard smoothness and norm-sub-Gaussian noise assumptions, we establish near-optimal rates for both strongl
    
[^42]: 同一编码器，不同赢家：一种用于Cell Painting编码器评估的成对视图框架

    Same Encoder, Different Winner: A Paired-View Framework for Cell Painting Encoder Evaluation

    [https://arxiv.org/abs/2609.12761](https://arxiv.org/abs/2609.12761)

    本文提出CP-BG-Bench成对视图评估框架，通过控制中心细胞周围像素的干预操作，揭示四种社区标准评估协议会对相同的Cell Painting编码器给出系统性不同的排名，其分歧可归结为细胞与背景、形态与上下文、研究内与跨批次三个维度。

    

    用于Cell Painting的视觉编码器通常通过单一评估进行排名，最常用的是重复样本平均精度（replicate mAP）。我们提出了CP-BG-Bench，一个成对视图评估框架，它在四个匹配的视图（原始裁剪C、分割视图S以及密度增强变体CD和SD）中保持中心细胞固定，通过消融或增强周围像素来实现受控干预。我们在三个数据集（JUMP-CP、RxRx1、RxRx3-core）和三个编码器（DINOv3 ViT-B/16、OpenPhenom、SubCell）上，按照四种社区标准协议（重复mAP、scIB批次整合、CellProfiler特征预测、跨批次扰动召回）实例化了该框架，发现这四种协议对相同编码器的排名存在系统性差异，其分歧可分解为三个维度：细胞与背景、形态与上下文、研究内与跨批次。最大的效应是：在RxRx3-core上，使用分割视图的SubCell……（原文摘要在此处截断）

    arXiv:2609.12761v1 Announce Type: cross  Abstract: Vision encoders for Cell Painting are typically ranked by a single evaluation, commonly replicate mean average precision (mAP). We introduce CP-BG-Bench, a paired-view evaluation framework that holds the central cell fixed across four matched views (raw crop C, segmented S, and density-augmented variants CD and SD), ablating or augmenting surrounding pixels as a controlled intervention. Instantiating the framework on three datasets (JUMP-CP, RxRx1, RxRx3-core) and three encoders (DINOv3 ViT-B/16, OpenPhenom, SubCell) under four community-standard protocols (replicate mAP, scIB batch integration, CellProfiler feature prediction, cross-batch perturbation recall), we find that the four protocols rank the same encoders systematically differently, with disagreements decomposing along three axes: cell versus background, morphology versus context, and within-study versus across-batch. The largest effect: on RxRx3-core, SubCell with segmented 
    
[^43]: 基于课程学习的对抗性异构智能体强化学习用于海上环境下的自主四旋翼无人机降落

    Curriculum-Based Adversarial Heterogeneous Agent Reinforcement Learning for Autonomous Quad-Copter Landing in Maritime Settings

    [https://arxiv.org/abs/2609.12758](https://arxiv.org/abs/2609.12758)

    本文提出结合课程学习与对抗性风场智能体的异构智能体强化学习方法（HARL-AC），用于训练船载机械臂空中捕获四旋翼无人机的鲁棒协作控制策略，在分布外海况下相比域随机化方法具有更好的泛化性能。

    

    在海洋环境中回收无人机具有挑战性，原因在于风湍流和船甲板的运动，这使得传统降落方法常常变得不可靠，因此该场景成为检验替代控制与学习方法的一个有价值的测试案例。我们研究了由船上机械臂在空中捕获四旋翼无人机的仿真问题，利用异构智能体近端策略优化强化学习方法学习鲁棒的协作控制策略。我们在NVIDIA Isaac Lab中使用课程学习和对抗性风场智能体（HARL-AC）进行HAPPO训练，并将获得的控制策略与基于课程学习的域随机化方法以及在单一海况下训练的基准方法进行比较。在海况0/4/5的分布内评估中，HARL-AC与域随机化方法表现出相当的成功率，最高可达97.5%。在分布外的海况7/8/10上，HARL-AC具有更好的泛化能力，达到了……

    arXiv:2609.12758v1 Announce Type: new  Abstract: Recovering unmanned aerial vehicles (UAVs) in maritime environments is challenging due to wind turbulence and ship-deck motion, making it a valuable test case for alternative control and learning approaches as conventional landing approaches often become unreliable. We study simulated mid-air capture of quadrotor UAVs by a ship-mounted robotic arm, learning robust cooperative control policies with Heterogeneous-Agent Proximal Policy Optimization (HAPPO) Reinforcement Learning. We train with HAPPO using a curriculum and an adversarial wind agent (HARL-AC) in NVIDIA Isaac Lab, and compare the obtained control policies against those generated through curriculum-based domain randomization and a benchmark trained on a single sea state. In-distribution evaluation on sea states $0/4/5$ shows comparable success for HARL-AC and domain randomization of up to $97.5\%$. On out-of-distribution sea states $7/8/10$, HARL-AC generalizes better, achievin
    
[^44]: 优化决策而非预测：将平滑净获益作为训练目标的探索

    Optimizing for the decision not the prediction: an exploration of Smooth Net Benefit as a training objective

    [https://arxiv.org/abs/2609.12752](https://arxiv.org/abs/2609.12752)

    本文提出平滑净获益（σNB）这一可微分训练目标，使预测模型的训练与特定风险阈值下的临床决策效用对齐，但在多个基准数据集上仅带来边际性能提升。

    

    目的：预测模型通常使用伯努利负对数似然（NLL）等目标进行训练，但下游临床决策可能依赖于特定的风险阈值。我们引入了平滑净获益（σNB），这是净获益的一种可微分近似，旨在使模型训练与特定阈值下的临床效用保持一致。材料与方法：我们将σNB作为逻辑回归、广义可加模型（GAM）和XGBoost的训练目标进行了评估，采用了三种Hessian实现。实验使用了Framingham心血管风险数据集以及包含72个数据集-阈值组合的44个TabZilla数据集。结果：σNB训练并未在Framingham数据集上持续改善净获益。在TabZilla基准测试中，逻辑回归的平均标准化净获益从使用NLL时的0.5669提升至使用σNB时的0.5765（平均差异0.0096，95% CI为-0.0001至0.0193）。

    arXiv:2609.12752v1 Announce Type: new  Abstract: Objective Prediction models are commonly trained using objectives such as Bernoulli negative log-likelihood (NLL), although downstream clinical decisions may depend on specific risk thresholds. We introduce Smooth Net Benefit ($\sigma$NB), a differentiable approximation of Net Benefit designed to align model training with threshold-specific clinical utility.   Materials and Methods We evaluated $\sigma$NB as a training objective for logistic regression, generalized additive models (GAMs), and XGBoost with three Hessian implementations. Experiments used the Framingham cardiovascular risk dataset and 44 TabZilla datasets comprising 72 dataset-threshold combinations.   Results $\sigma$NB training did not consistently improve Net Benefit in Framingham. Across the TabZilla benchmark, mean standardized Net Benefit for logistic regression increased from 0.5669 with NLL to 0.5765 with $\sigma$NB (mean difference 0.0096, 95% CI -0.0001 to 0.0193)
    
[^45]: 什么驱动了智能体文本到Cypher的恢复能力？LAST-CQ：一种LLM智能体自我精炼框架

    What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework

    [https://arxiv.org/abs/2609.12746](https://arxiv.org/abs/2609.12746)

    该研究通过LAST-CQ框架上的反事实实验发现，智能体Text-to-Cypher系统的恢复收益主要来自失败检测与重试路由机制，而非复杂的LLM反馈内容。

    

    用于结构化查询生成的智能体流水线正在迅速扩展，但尚不清楚循环中的哪个部分产生了收益。我们使用LAST-CQ——一个五智能体、免训练、基于执行结果的Text-to-Cypher框架——作为插桩测试平台，在2,471个实时数据库查询和跨越三个厂商规模层级的六个骨干模型上运行了三个反事实实验。移除纠错环节，相对于单次通过系统会损失3.1%的聚合执行BLEU，相对于无精炼的反事实会损失12.3%（对于最弱的骨干模型高达80.7%）。用原始数据库错误字符串替换基于模式、由LLM合成的反馈几乎没有任何损失（20.9% vs 19.9%朴素精确匹配；端到端差异<0.2%；通过两次单侧检验证明在±0.075集合F1范围内等价）。将相同的调用预算用于并行采样会使质量下降10-11%。真正有效的是检测失败并将其路由到重试，而不是反馈内容的复杂精妙。

    arXiv:2609.12746v1 Announce Type: cross  Abstract: Agentic pipelines for structured-query generation are rapidly expanding, but it is unclear which part of the loop produces the gain. We use LAST-CQ -- a five-agent, training-free, execution-grounded Text-to-Cypher framework -- as an instrumented testbed, running three counterfactuals over 2,471 live-database queries and six backbones spanning three vendor scale tiers. Removing correction is worth between 3.1% aggregate execution-BLEU against the single-pass system and 12.3% against a no-refinement counterfactual (up to 80.7% for the weakest backbone). Replacing schema-grounded, LLM-synthesised feedback with raw database error strings costs almost nothing (20.9% vs. 19.9% naive exact match; <0.2% end-to-end; equivalent within $\pm 0.075$ set-F1 by two one-sided tests). Spending the same call budget on parallel sampling degrades quality by 10-11%. What works is detecting failure and routing it to a retry, not the feedback sophistication 
    
[^46]: 物理引导的合成高频超声图像生成用于皮肤层分割

    Physics-Guided Synthetic High-Frequency Ultrasound Generation for Skin Layer Segmentation

    [https://arxiv.org/abs/2609.12735](https://arxiv.org/abs/2609.12735)

    该论文提出一种物理引导的合成高频超声生成框架，通过k-Wave声学仿真生成成对的超声图像与密集层标注数据，解决了皮肤层分割任务中深层结构密集标注数据稀缺的问题，并验证了合成数据预训练向真实数据迁移的有效性。

    

    高频超声（HFUS）能够对皮肤浅层结构进行无创可视化，但自动化皮肤层分析受到密集标注数据稀缺的限制。现有的真实高频超声数据集通常仅提供表皮和表皮下低回声带（SLEB）等浅层目标的标注，而真皮、皮下组织、筋膜和肌肉等深层结构的密集标注很少可用。我们提出了一种用于皮肤层分割的物理引导合成高频超声生成框架。该框架构建多层声学皮肤体模，为各层分配相应的声学属性，并使用k-Wave仿真生成成对的合成高频超声图像、密集层掩码和仿真元数据。为评估生成数据能否提供可迁移的监督信息，我们将其用于下游分割任务的预训练，并在真实的Mendeley高频超声数据上对模型进行微调。

    arXiv:2609.12735v1 Announce Type: new  Abstract: High-frequency ultrasound (HFUS) enables noninvasive visualization of superficial skin structures, but automated skin-layer analysis is limited by the scarcity of densely annotated data. Existing real HFUS datasets commonly provide annotations for superficial targets such as the epidermis and subepidermal low-echogenic band (SLEB), while dense labels for deeper structures such as dermis, subcutaneous tissue, fascia, and muscle are rarely available. We propose a physics-guided synthetic HFUS generation framework for skin layer segmentation. The framework constructs multilayer acoustic skin phantoms, assigns layer dependent acoustic properties, and uses k-Wave simulation to generate paired synthetic HFUS images, dense layer masks, and simulation metadata. To evaluate whether the generated data provide transferable supervision, we use it for downstream segmentation pretraining and fine-tune the models on real Mendeley HFUS data. Synthetic p
    
[^47]: Prism-SQA：一种可解释且可适应的表面肌电信号质量评估神经框架

    Prism-SQA: An Interpretable and Adaptable Neural Framework for Surface Electromyography Quality Assessment

    [https://arxiv.org/abs/2609.12724](https://arxiv.org/abs/2609.12724)

    本文提出Prism-SQA框架，将sEMG信号质量评估重构为生理感知的源分离与验证过程，通过将信号分解为干净成分和五种污染成分，实现了可解释、可适应且无需重新训练的信号质量评估。

    

    表面肌电信号（sEMG）容易受到各种污染物的干扰，这些污染物会扭曲信号的形态和频谱内容。准确的信号质量评估（SQA）对于识别此类信号退化、确保可靠的临床分析与决策至关重要。近年来基于神经网络的SQA方法通过学习复杂的污染模式实现了准确的质量估计，但其黑盒特性使临床医生无法理解或验证所报告的评分，并且在不重新训练的情况下难以适应特定应用的质量定义。为了解决这些局限性，我们提出了Prism-SQA，这是一个可解释且可适应的神经框架，它将信号质量评估重新构建为一个具有生理感知能力的源分离与验证过程。Prism-SQA使用带有双向长短期记忆网络的U-Net将每个输入信号分解为一个干净的sEMG成分和五个针对特定污染物的成分。每个分离出的污染成分都会被检验……（原文摘要在此截断）

    arXiv:2609.12724v1 Announce Type: cross  Abstract: sEMG is vulnerable to various contaminants that distort signal morphology and spectral content. Accurate signal quality assessment (SQA) is essential for identifying such degradation and ensuring reliable clinical analyses and decisions. Recent neural network-based SQA methods achieve accurate quality estimation by learning complex contamination patterns, yet their black-box nature prevents clinicians from understanding or validating the reported scores and limits adaptability to application-specific quality definitions without retraining. To address these limitations, we propose Prism-SQA, an interpretable and adaptable neural framework that reformulates SQA as a physiology-aware source-separation and verification process. Prism-SQA decomposes each input signal into a clean sEMG component and five contaminant-specific components using a U-Net with bidirectional long short-term memory. Each separated contaminant component is examined b
    
[^48]: 通过跨模态对比学习从X射线衍射推断位错微观结构

    Inferring Dislocation Microstructures from X-ray Diffraction via Cross-Modal Contrastive Learning

    [https://arxiv.org/abs/2609.12713](https://arxiv.org/abs/2609.12713)

    本文提出了一种跨模态对比学习框架，将离散位错动力学模拟生成的位错密度场与虚拟X射线衍射图谱嵌入共享潜在空间，实现了直接从衍射数据预测三维位错微观结构。

    

    从衍射图谱中理解和推断位错微观结构仍然是材料表征领域的一个开放性挑战，因为衍射测量只能提供关于底层位错结构的间接信息。在这项工作中，研究人员开发了一种跨模态学习框架，能够直接从衍射数据预测三维位错结构。由离散位错动力学模拟生成的位错密度场与相应的虚拟X射线衍射图谱进行配对，并通过对比学习嵌入到共享的二维潜在空间中。位错结构的结构表示与衍射表示之间的对齐程度通过相应潜在特征之间的相关性，直接在学习到的潜在空间中进行评估。为了评估数据集规模对这种方法的影响，研究采用最远点采样来构建具有代表性和多样性的训练数据集。

    arXiv:2609.12713v1 Announce Type: cross  Abstract: Understanding and inferring dislocation microstructures from diffraction patterns remains an open challenge in materials characterization, as diffraction measurements provide only indirect information about the underlying dislocation structure. In this work, a cross-modal learning framework is developed to enable the prediction of 3D dislocation structures directly from diffraction data. Dislocation density fields generated from discrete dislocation dynamics simulations are paired with corresponding virtual X-ray diffraction patterns and embedded into a shared 2D latent space using contrastive learning. The alignment between structural and diffraction representations of dislocation structures is evaluated directly in the learned latent space using correlations between corresponding latent features. To estimate the role of dataset size for this approach, farthest point sampling is employed to construct representative and diverse trainin
    
[^49]: InRTL：面向关系表的高效表内-表间交互学习

    InRTL: Effective Intra-Inter Interaction Learning for Relational Tables

    [https://arxiv.org/abs/2609.12712](https://arxiv.org/abs/2609.12712)

    提出InRTL统一框架，通过列感知表编码器与基于Transformer的自注意力和交叉注意力模块，显式建模关系表内部的行关联以及跨PK-FK连接表的行依赖，实现有效的关系表学习。

    

    关系表学习近年来已成为对通过主键-外键（PK-FK）关系连接的多张表进行建模的重要研究方向。尽管最近取得了一些进展，但针对该任务的原理性建模框架仍未得到充分探索。在本文中，我们提出了表内-表间关系表学习，这是一个统一框架，能够显式地对关系表内部以及跨表的依赖关系进行建模。具体而言，InRTL形式化了两种互补的交互模式：表内交互，描述同一表内各行之间的关联；表间交互，描述通过PK-FK连接的表之间各行之间的依赖关系。为了对这些依赖关系进行建模，我们开发了一个列感知的表编码器来生成初始行表示，随后使用基于Transformer的自注意力模块和交叉注意力模块分别进行表内学习和表间学习。

    arXiv:2609.12712v1 Announce Type: new  Abstract: Relational table learning has recently emerged as an important research direction for modeling multiple tables connected through primary key-foreign key (PK-FK) relationships. Despite recent advances, a principled modeling framework tailored to this task remains underexplored. In this paper, we propose Intra-Inter Relational Table Learning (InRTL), a unified framework that explicitly models dependencies both within and across relational tables. Specifically, InRTL formalizes two complementary interaction patterns: intra-table interactions, describing associations among rows within the same table, and inter-table interactions, describing dependencies between rows across PK-FK-linked tables. To model these dependencies, we develop a column-aware table encoder to generate initial row representations, followed by Transformer-based self-attention and cross-attention modules for intra-table and inter-table learning, respectively. To further im
    
[^50]: ExpertHTR：基于多任务学习与稀疏混合专家的统一手写文本识别

    ExpertHTR: Unified Handwritten Text Recognition with Multi-Task Learning and Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2609.12705](https://arxiv.org/abs/2609.12705)

    ExpertHTR提出了一种基于多任务学习与稀疏混合专家的统一视觉-语言框架，通过统一的页-区域-行表示整合异构手写识别数据集，构建四个互补训练任务，并利用Sparsegen机制实现专家数量的动态激活，无需额外人工标注。

    

    手写文本识别资源通常规模较小，且分布在语言、文字系统、文档结构和标注格式各不相同的数据集中，这使得页面级的联合训练十分困难。我们提出了 ExpertHTR，这是一个统一的视觉-语言框架，通过互补监督和条件化模型容量来解决这一问题。来自异构数据集的结构化标注首先通过统一的“页-区域-行”表示进行组织，并用于构建四个相互关联的训练任务：完整转录、物理行覆盖、文本定位和局部识别，且无需额外的手工标注。在联合训练的稠密模型基础上，ExpertHTR 引入了稀疏混合专家架构，该架构包含始终激活的共享分支和条件性路由的全MLP专家。Sparsegen 机制使激活的路由专家数量可以随隐藏表示动态变化。

    arXiv:2609.12705v1 Announce Type: cross  Abstract: Handwritten text recognition resources are often small and distributed across collections that differ in language, script, document structure, and annotation format, making joint page-level training difficult. We propose ExpertHTR, a unified vision-language framework that addresses this problem through complementary supervision and conditional model capacity. Structural annotations from heterogeneous datasets are first organized through a common Page-Region-Line representation and used to construct four related training tasks for complete transcription, physical-line coverage, text localization, and localized recognition, without requiring additional manual labels. Building on a jointly trained dense model, ExpertHTR introduces a sparse Mixture-of-Experts architecture with an always-active shared branch and conditionally routed full-MLP experts. Sparsegen allows the number of active routed experts to vary with the hidden representation
    
[^51]: 在纸上书写并获取在线数字轨迹：手写的新时代

    Write on Paper and Get the Online Digital Trace:\newline A New Era for Handwriting

    [https://arxiv.org/abs/2609.12702](https://arxiv.org/abs/2609.12702)

    该论文提出了一种结合传感器数字笔与先进AI算法的创新方案，无需特殊纸张或外部参考系统，即可实时重建在普通纸上书写的手写数字轨迹。

    

    捕获手写的数字轨迹通常需要特定的触控笔和兼容的基底材料，无论是电容式触摸屏、Wacom系统所使用的电磁共振（EMR）数位板，还是特殊纸张。虽然在普通纸上书写能提供丰富的触觉体验、无延迟，并且以改善信息记忆保留而闻名，但目前尚不存在一种低成本、被广泛接受的有效方案来数字化这种笔迹。其挑战在于如何在没有外部参考系统的情况下准确跟踪笔的轨迹，同时允许笔在表面上不受限制地自由移动。我们提出了一种创新解决方案，结合了数字笔、先进的人工智能算法和自适应AI技术来重建手写的数字轨迹。我们的方法集成了硬件开发（专注于配备传感器的笔）与软件创新，以实时优化轨迹的重建与处理。

    arXiv:2609.12702v1 Announce Type: new  Abstract: Capturing the digital trace of handwriting usually requires a specific stylus and a compatible substrate, be it a capacitive touchscreen, an ElectroMagnetic Resonance (EMR) tablet as used in Wacom systems or special paper. While writing on regular paper offers rich haptics, no latency and is well known for improving information retention, no low-cost and widely accepted, effective solution exists to digitize such a pen trace. The challenge is to accurately track the pen's trajectory without an external reference system while allowing unrestricted freedom of pen movement across a surface. We propose an innovative solution that combines a digital pen, advanced artificial intelligence algorithms, and adaptive AI techniques to reconstruct the digital trace of handwriting. Our approach integrates hardware development, focusing on a sensor-equipped pen, with software innovations to optimize trajectory reconstruction and processing in real time
    
[^52]: SIFPBPNet：一种基于个体化稳态表征的可穿戴无袖带血压估计双路径网络

    SIFPBPNet: A Dual-Path Network for Wearable and Cuffless Blood Pressure Estimation via Individualized Steady-state Representation

    [https://arxiv.org/abs/2609.12690](https://arxiv.org/abs/2609.12690)

    该论文提出双路径网络SIFPBPNet，通过图注意力网络从多天历史PPG轨迹中提取个体化稳态特征，并结合瞬时特征路径与交叉注意力机制，有效解决了无袖带血压估计中的人群异质性和“一对多映射”问题。

    

    使用光电容积脉搏波（PPG）进行连续、无袖带的血压（BP）监测，对于低成本、个性化的心血管健康管理具有重要意义。然而，显著的人群异质性以及“一对多映射”问题——即不同个体间相似的波形却对应不同的血压水平——限制了传统基于人群模型的准确性。为应对这一挑战，我们提出了一种名为SIFPBPNet的双路径架构，该架构通过稳态特征路径（SFP）和瞬时特征路径（IFP）分别表征稳态特征与瞬时特征。SFP采用图注意力网络（GAT）从多天的历史PPG轨迹中提取个体特异性和长期特征。与此同时，IFP从当前PPG片段中捕获短期动态，并通过交叉注意力机制融入稳态先验。在大型（数据集上的实验……摘要在此处截断）

    arXiv:2609.12690v1 Announce Type: new  Abstract: Continuous and cuffless blood pressure (BP) monitoring using photoplethysmography (PPG) is of great interest for low-cost and personalized cardiovascular health management. However, significant population heterogeneity and the "one-to-many mapping" problem, where similar waveforms across individuals correspond to different BP levels, limit the accuracy of conventional population-based models. To address this challenge, we propose a dual-path architecture termed SIFPBPNet, which separately represents steady-state and instantaneous features, through a Steady-state Feature Path (SFP) and an Instantaneous Feature Path (IFP). The SFP employs a Graph Attention Network (GAT) to extract individual-specific and long-term characteristics from multi-day historical PPG trajectories. In parallel, the IFP captures short-term dynamics from current PPG segments and incorporates the steady-state prior via a cross-attention mechanism. Experiments on a lar
    
[^53]: MLLM听到了什么？面向音频多模态大语言模型可解释性的词元级时频定位

    What Did the MLLM Hear? Token-Level Spectro-Temporal Grounding for Audio MLLM Explainability

    [https://arxiv.org/abs/2609.12663](https://arxiv.org/abs/2609.12663)

    提出了STAG——首个对音频多模态大语言模型生成描述进行词元级时频定位的事后解释框架，通过词元特定词汇投影与频谱遮挡相结合，揭示每个生成词元所依赖的时间和频率声学证据。

    

    基于音频的多模态大语言模型（MLLM）能够对复杂声学场景生成详细的自然语言描述，但目前仍不清楚输入音频的哪些部分支持了每个生成词元的产生。这一问题尤其具有挑战性，因为声学证据分布在时间和频率两个维度上，且同时发生的声音事件可能在时间上相互重叠、却占据不同的频谱区域。我们提出了STAG，据我们所知，这是首个针对基于音频的多模态大语言模型所生成描述进行词元级时频定位的事后解释框架。STAG通过使用针对目标词元的特定词汇投影来估计编码音频表征中每个生成词元的时间支撑，并通过受控的频谱遮挡来衡量频带相关性，进而将这两种信号结合成一张时频相关性图。我们在四个定位基准上将STAG与十种事后解释方法进行了对比评估……

    arXiv:2609.12663v1 Announce Type: cross  Abstract: Audio-based Multimodal Large Language Models (MLLMs) can generate detailed natural-language descriptions of complex acoustic scenes, yet it remains unclear which parts of the input audio support each generated token. This is particularly challenging because acoustic evidence is distributed across time and frequency, and concurrent sound events may overlap temporally while occupying different spectral regions. We introduce STAG, to our knowledge the first post-hoc framework for token-level spectro-temporal grounding of captions generated by audio-based MLLMs. STAG estimates the temporal support for each generated token using target-token-specific vocabulary projections of the encoded audio representations, measures frequency-band relevance through controlled spectral occlusion, and combines the two signals into a spectro-temporal relevance map. We evaluate STAG against ten post-hoc explanation methods across four grounding benchmarks, w
    
[^54]: ProactiveBench：流式视频模型能否真正像人类一样交互？

    ProactiveBench: Can Streaming Video Models Really Interact Like Humans?

    [https://arxiv.org/abs/2609.12658](https://arxiv.org/abs/2609.12658)

    提出了ProactiveBench基准，以一秒流式间隔、无明确响应提示的方式评估流式视频模型的主动交互能力（即在恰当时机响应目标事件、其余时间保持沉默），发现多数系统存在过早响应而非漏响应的问题。

    

    流式视频理解要求模型在处理连续多模态输入的同时保持时间上下文。现有的评估方法主要是被动式的：它们在选定的时间戳查询模型，因此无法评估模型何时应该响应。主动交互则要求模型监测一个持续存在的请求，在目标事件发生后的适当时间间隔内做出响应，其余时间保持沉默。我们提出了ProactiveBench，它以一秒的流式间隔评估模型，且不提供明确的响应提示。其六个子任务在触发歧义程度和时间容差上有所变化：事件敏感度在同一录制上以几何方式结合响应率和沉默率；四个基于窗口的子任务区分过早响应、窗口内响应和错失响应；重复计数任务惩罚遗漏和重复。在六个被评估的系统中，有四个系统的过早响应数量超过错失响应数量，揭示出相当大的……（摘要在此处截断）

    arXiv:2609.12658v1 Announce Type: new  Abstract: Streaming video understanding requires models to process continuous multimodal input while maintaining temporal context. Existing evaluations are predominantly reactive: they query a model at a selected timestamp and therefore do not assess when it should respond. Proactive interaction instead requires monitoring a standing request, responding within an appropriate interval after the target event, and otherwise remaining silent. We introduce ProactiveBench, which evaluates models at one-second stream intervals without an explicit response cue. Its six subtasks vary trigger ambiguity and timing tolerance. Event Sensitivity geometrically combines response and silence rates on the same recording; four window-based subtasks distinguish early, in-window, and missed responses; and Duplicate Counting penalizes omissions and repetitions. Premature responses outnumber missed responses for four of the six evaluated systems, revealing a substantial
    
[^55]: 重新审视AI对齐的失真：RLHF是一个不错的功利主义对齐器

    Distortion of AI Alignment Revisited: RLHF is a Decent Utilitarian Aligner

    [https://arxiv.org/abs/2609.12651](https://arxiv.org/abs/2609.12651)

    本文通过细粒度分析证明，RLHF失真的指数级退化并非算法本身的固有缺陷，而是源于偏好数据分布与KL参考策略之间的分布不匹配，从而表明RLHF实际上是一个不错的功利主义对齐器。

    

    尽管基于人类反馈的强化学习（RLHF）是将大语言模型与人类偏好对齐的标准范式，但其在多元偏好场景中的有效性一直受到质疑。值得注意的是，Gölz等人（2025）最近的研究表明，当用户具有异质偏好时，失真（distortion）——定义为RLHF策略的平均用户效用与最优平均效用之间的乘性差距——可能随Bradley-Terry温度参数β呈指数级增长。在本工作中，我们对带奖励裁剪的RLHF的失真进行了细粒度分析，并证明这种指数级退化并非该算法的固有属性，而是产生偏好数据的分布（μ）与KL参考策略（π_ref）之间分布不匹配的结果。为此，我们建立了紧密的上下界……

    arXiv:2609.12651v1 Announce Type: new  Abstract: While Reinforcement Learning from Human Feedback (RLHF) is the standard paradigm for aligning large language models with human preferences, its effectiveness in pluralistic settings has been called into question. Notably, recent work by G\"olz et al. (2025) demonstrated that the \textit{distortion} -- defined as the multiplicative gap between the average user utility of the RLHF policy and the optimal average utility -- can scale exponentially with the Bradley-Terry temperature parameter $\beta$ when users have heterogeneous preferences. In this work, we present a fine-grained analysis of the distortion of RLHF with reward clipping and demonstrate that such exponential degradation is not a fundamental property of the algorithm but rather a consequence of distribution mismatch between the distribution generating preference data ($\mu$) and the KL reference policy ($\pi_{\mathrm{ref}}$). To this end, we establish tight upper and lower boun
    
[^56]: 基于预测步长分辨归因的时间序列预测解释

    Explaining Time Series Forecasting with Horizon-Resolved Attribution

    [https://arxiv.org/abs/2609.12639](https://arxiv.org/abs/2609.12639)

    该论文提出HRX即插即用框架，通过在时间序列预测解释中引入预测步长维度，为每个预测步骤生成独立的重要性归因图，突破了传统方法假设所有预测步骤依赖相同过去值的局限。

    

    时间序列（TS）模型解释领域的最新进展产生了一些能够识别预测依赖于哪些过去值的方法。然而，大多数现有方法仅返回一个单一的重要性向量，假设每个预测步骤都依赖于相同的过去值。在本文中，我们证明了这一假设并不成立，因为不同的预测步骤依赖于不同的过去值。基于这一观察，我们提出了Horizon-Resolved eXplanation（HRX），它在解释中增加了一个预测步长（horizon）维度，使得每个预测步骤都拥有自己的重要性图。HRX是一个简单而有效的即插即用框架，包含三个组件：1）一个估计器，能够从任何可微的预测模型中读出这些重要性图，而无需修改时间序列主干网络；2）一个评估协议，通过测量移除重要性图中排名最高的输入时单个预测步骤的变化程度来验证预测步长维度的有效性；3）一个排序准则……（原文摘要在此处截断）

    arXiv:2609.12639v1 Announce Type: new  Abstract: Recent advances in explaining time series (TS) models have produced methods that identify which past values a prediction depends on. However, most existing methods return a single importance vector, assuming that every predicted step depends on the same past values. In this paper, we show that this assumption does not hold, as different forecast steps depend on different past values. Motivated by this observation, we propose Horizon-Resolved eXplanation (HRX), which adds a horizon axis to the explanation, so that every forecast step receives its own importance map. HRX is a simple yet effective plug-in framework with three components: 1) an estimator that reads these maps out of any differentiable forecaster without modifying the TS backbone, 2) an evaluation protocol that validates the horizon axis by measuring how much a single forecast step changes when the inputs an importance map ranks highest are removed, and 3) a rank criterion th
    
[^57]: 面向皮层脑沟标注的几何到语义球面迁移学习

    Geometric-to-Semantic Spherical Transfer Learning for Cortical Sulci Labeling

    [https://arxiv.org/abs/2609.12627](https://arxiv.org/abs/2609.12627)

    该论文提出一种几何到语义球面迁移学习框架，利用约3万例无标注数据预训练球面编码器，解决了皮层脑沟标注中因专家标注极度稀缺而导致的过拟合和泛化困难问题。

    

    皮层表面上的深度学习面临一个两难困境：捕捉每个半球超过60个依赖命名法的脑沟的复杂拓扑结构需要高容量模型，然而专家标注数据的极度稀缺（N=62名受试者）不可避免地导致过拟合。标准的监督方法在这种数据稀缺的情况下无法泛化，特别是对于拓扑模糊性较高的多变且体积较小的脑沟。为了克服这一局限，我们引入了一个几何到语义球面迁移学习框架。首先，我们利用大规模无标注数据（英国生物银行，约30,000名受试者），采用局部优化策略预训练一个球面编码器。通过仅依赖连续表面特征（曲率和深度），该预训练的相关性通过模型检测局部化和罕见拓扑特征（如脑沟中断）的能力得到证实。然而，下游标注任务引入了...

    arXiv:2609.12627v1 Announce Type: new  Abstract: Deep learning on cortical surfaces faces a dilemma: capturing the complex topology of over 60 nomenclature-dependent sulci per hemisphere requires high-capacity models, yet the extreme scarcity of expert annotations ($N=62$ subjects) inevitably causes overfitting. Standard supervised approaches fail to generalize in this data-scarce regime, particularly for variable and small sulci where topological ambiguity is high. To overcome this limitation, we introduce a Geometric-to-Semantic Spherical Transfer Learning framework.   First, we leverage massive unlabeled data (UK Biobank, $\approx$30,000 subjects) to pre-train a spherical encoder using a locally-optimized strategy. By relying solely on continuous surface features (curvature and depth), the relevance of this pre-training is confirmed by the model's ability to detect localized and rare topological traits, such as sulcal interruptions. The downstream labeling task, however, introduces 
    
[^58]: 基于Hessian分析的相关性引导快速机器遗忘

    Correlation-Guided Fast Machine Unlearning via Hessian Analysis

    [https://arxiv.org/abs/2609.12620](https://arxiv.org/abs/2609.12620)

    本文提出了一种基于Hessian分析的计算高效机器遗忘框架，通过识别训练集中的相关数据点并应用理论推导的闭式解，避免了传统方法中每个数据点移除所需的昂贵Hessian逆向量重复计算，显著提升了入侵检测等安全系统批量处理遗忘请求的效率。

    

    机器学习在网络和分布式安全系统中的日益广泛应用，使得人们迫切需要能够选择性地、高效地移除特定训练数据影响的机制，以便从生产模型中消除受损或对抗性的数据点。GDPR等隐私法规中的“被遗忘权”也提出了类似的要求。然而，现有的近似遗忘技术在现实安全系统中的部署计算成本仍然过高，因为每移除一个数据点都需要重复进行昂贵的Hessian逆向量计算，这在入侵检测系统、垃圾邮件过滤器和威胁情报平台等需要处理多个相关请求的场景中形成了计算瓶颈。为此，我们提出了一种计算高效的机器遗忘框架，该框架能够识别训练集中相互关联的数据点，并应用理论上推导出的闭式解……（摘要截断）

    arXiv:2609.12620v1 Announce Type: new  Abstract: The increasing adoption of machine learning in network and distributed security systems has created an urgent need for mechanisms that can selectively and efficiently remove the influence of specific training data to eliminate compromised or adversarial data points from production models. Privacy regulations such as GDPR's \emph{right to be forgotten} also pose similar requirements. However, existing approximate unlearning techniques remain computationally prohibitive for deployment in real-world security systems, as they require repeated expensive Hessian-inverse-vector computations for each data point removal, creating a bottleneck when processing multiple related requests in scenarios such as intrusion detection systems, spam filters, and threat intelligence platforms. Thus, we introduce a computationally efficient unlearning framework that identifies correlated data points in the training set and applies a theoretically derived close
    
[^59]: SIMS：面向多任务学习的尺度不变价值函数标量化方法

    SIMS: Scale-Invariant Merit-Function-Based Scalarization for Multi-Task Learning

    [https://arxiv.org/abs/2609.12599](https://arxiv.org/abs/2609.12599)

    提出了尺度不变价值函数标量化方法SIMS，解决了现有多任务学习标量化方法因任务损失尺度差异而偏向大尺度目标的问题，使优化结果不受目标缩放的影响。

    

    多任务学习（MTL）需要在相互竞争的目标之间进行不可避免的权衡。该范式通常被表述为多目标优化（MOO）问题，而标量化方法因能将多目标优化问题简化为单目标问题而受到青睐。我们通过实证研究发现，在实际的多任务学习中，现有的基于价值函数的标量化方法对不同目标的相对尺度非常敏感，因为各任务损失之间通常存在数量级上的差异。优化过程往往偏向于尺度较大的目标，尽管其底层的帕累托最优解在重新缩放（即用一个正常数乘以某个目标）下应保持不变。为了解决这一问题，我们提出了面向多任务学习的尺度不变价值函数标量化方法（SIMS）。具体而言，SIMS采用了一种由变换诱导的价值函数，将多任务学习的多目标优化问题转换为单目标问题，使得优化过程对目标的尺度保持不变。

    arXiv:2609.12599v1 Announce Type: new  Abstract: Multi-task learning (MTL) requires navigating unavoidable trade-offs among competing objectives. This paradigm is frequently formulated as multi-objective optimization (MOO), where the scalarization is favored to reduce an MOO problem to a single objective. We empirically find that existing merit-function-based scalarization approaches are sensitive to the relative scales of different objectives in practical MTL, where task losses commonly differ by orders of magnitude. The optimization process often favors objectives with larger scales even though the underlying Pareto optimal solutions remains invariant to rescaling (i.e., multiplying an objective by a positive constant). To address this issue, we propose Scale-Invariant Merit-function-based Scalarization (SIMS) for MTL. Specifically, SIMS adopts a transformation-induced merit function to convert the MOO problem of MTL to a single objective that renders optimization invariant to the ma
    
[^60]: Moreau-Yosida未调整朗之万采样的泊松校正复杂度界

    Poisson-Corrector Complexity Bounds for Moreau--Yosida Unadjusted Langevin Sampling

    [https://arxiv.org/abs/2609.12594](https://arxiv.org/abs/2609.12594)

    该论文通过离散泊松校正子技术为Moreau-Yosida未调整朗之万算法（MYULA）建立了更优的非渐近复杂度界，证明在W2距离下达到ε精度仅需$\widetilde O(\varepsilon^{-4/3})$次迭代，且误差系数对平滑参数仅呈对数依赖。

    

    我们研究了经典的Moreau-Yosida未调整朗之万算法（MYULA），用于从分布 $\pi(\mathrm{d}x) \propto e^{-f(x)-g(x)}\mathrm{d}x$ 中采样，其中 $f\in C^2(\mathbb{R}^d)$ 是 $m$-强凸函数且梯度满足 $L_f$-Lipschitz条件，$g:\mathbb{R}^d\to\mathbb{R}$ 是凸函数且全局满足 $G$-Lipschitz条件。对于Moreau平滑后的目标分布 $\pi_\lambda$ 与MYULA的不变分布 $\widehat\pi_{\lambda,h}$，在条件 $0<h(L_f+\lambda^{-1})\le c$ 下，我们证明了 $\sqrt m\,W_2(\pi_\lambda,\widehat\pi_{\lambda,h})=O(h)+\widetilde O(h^{3/4})$，且误差系数对 $\lambda^{-1}$ 仅具有对数依赖性。将该估计与Moreau逼近偏差相结合，在固定模型参数和初始化条件下，达到 $\sqrt m\,W_2(\mu_N,\pi)\le\varepsilon$ 仅需 $\widetilde O(\varepsilon^{-4/3})$ 次迭代。证明方法将离散泊松校正子与有效迹估计相结合，并对精确-欧拉两点耦合给出了共享噪声界。

    arXiv:2609.12594v1 Announce Type: new  Abstract: We study the classical Moreau--Yosida unadjusted Langevin algorithm (MYULA) for $\pi(\,\mathrm{d} x)\propto e^{-f(x)-g(x)}\,\mathrm{d} x$, where $f\in C^2(\mathbb{R}^d)$ is $m$-strongly convex with $L_f$-Lipschitz gradient and $g:\mathbb{R}^d\to\mathbb{R}$ is convex and globally $G$-Lipschitz. For the Moreau-smoothed target $\pi_\lambda$ and the MYULA invariant law $\widehat\pi_{\lambda,h}$, we prove \[   \sqrt m\,W_2(\pi_\lambda,\widehat\pi_{\lambda,h})   =O(h)+\widetilde O(h^{3/4}) \] under $0<h(L_f+\lambda^{-1})\le c$, with only logarithmic dependence on $\lambda^{-1}$ in the error coefficients. Combining this estimate with the Moreau approximation bias yields $\widetilde O(\varepsilon^{-4/3})$ iterations to achieve $\sqrt m\,W_2(\mu_N,\pi)\le\varepsilon$, for fixed model parameters and initialization. The proof combines a discrete Poisson corrector with active-trace estimates and a shared-noise bound for the exact--Euler two-point cu
    
[^61]: 解码器余弦相似度在SAE特征流发现中的失效之处

    Where Decoder Cosine Similarity Fails for SAE Feature Flow Discovery

    [https://arxiv.org/abs/2609.12591](https://arxiv.org/abs/2609.12591)

    本文提出构建SAE特征“转换图集”的方法来系统发现残差状态特征与MLP更新特征如何共同预测下游残差特征，并揭示了余弦相似度筛选在此类特征流发现中的失效之处。

    

    基础模型越来越多地通过微调、模型编辑和对齐程序进行调整，同时保留先前获得的能力。因此，理解支持这些适应过程的内部计算对于模型的持续演进变得越来越重要。稀疏自编码器（SAEs）为残差流激活和子层输出提供了可解释的特征字典，但状态特征与更新特征如何相互作用以产生下游残差特征仍不清楚。在这项工作中，我们将MLP更新作为首个测试用例。我们构建了一个转换图集，包含三元组 $s_k + u_j \rightarrow t_\ell$，其中残差状态特征和MLP更新特征共同预测一个目标残差特征，并通过消融解码后的更新特征来验证候选三元组。在2000万token的Pythia-160M $L_7 \rightarrow L_8$ 运行实验中，我们发现了38,125个具有强消融效应的转换，但……

    arXiv:2609.12591v1 Announce Type: new  Abstract: Foundation models are increasingly adapted through fine-tuning, model editing, and alignment procedures while retaining previously acquired capabilities. Understanding the internal computations that support these adaptations is therefore becoming increasingly important for continual model evolution. Sparse autoencoders (SAEs) provide interpretable feature dictionaries for residual-stream activations and sublayer outputs, but it remains unclear how state features and update features interact to produce downstream residual features. In this work, we focus on MLP updates as a first test case. We construct a transition atlas of triples $s_k + u_j \rightarrow t_\ell$, where a residual-state feature and an MLP-update feature jointly predict a target residual feature, and validate candidate triples by ablating the decoded update feature. In a 20M-token Pythia-160M $L_7 \rightarrow L_8$ run, we find 38,125 strong ablation-effect transitions, but
    
[^62]: 固定维度下基于随机梯度预言机的紧致采样复杂度

    Tight Sampling Complexity with stochastic gradient oracles in Fixed Dimensions

    [https://arxiv.org/abs/2609.12590](https://arxiv.org/abs/2609.12590)

    本文证明了固定维度下使用随机梯度预言机采样光滑强对数凹分布的最优查询复杂度为 Θ(log(1+κ) + σ²/(με))，该复杂度界同时对条件数和精度ε均达到紧致最优，并自适应于无噪声情形。

    

    我们研究了在任意固定欧几里得维度下采样光滑强对数凹分布的随机梯度查询复杂度。势函数为μ-强凸且L-光滑，其未知模式位于以原点为中心、半径为μ^{-1/2}的球内。我们可以访问方差至多为σ²的无偏随机预言机。对于所有σ²≥0以及总变差（TV）精度0<ε≤1/10，我们证明了从与目标分布ε-TV距离内进行采样的紧致复杂度为 N\*_TV = Θ(log(1+κ) + σ²/(με))，其中κ:=L/μ为条件数。值得注意的是，该复杂度界同时对条件数κ和精度ε都是紧致的。此外，我们的紧致复杂度界对无噪声情形σ=0是自适应的，此时 N\*_TV = Θ（原文摘要在此处截断）

    arXiv:2609.12590v1 Announce Type: cross  Abstract: We investigate the stochastic-gradient query complexity of sampling smooth strongly log-concave distributions in any fixed Euclidean dimension. The potential is $\mu$-strongly convex and $L$-smooth, with an unknown mode in the ball of radius $\mu^{-1/2}$ about the origin. We have access to unbiased stochastic oracles with the variance at most $\sigma^2$. For every $\sigma^2\ge0$ and total variation (TV) accuracy $0<\varepsilon\le1/10$, we prove that the tight complexity of sampling a distribution within $\epsilon$-TV distance from the target distribution is \[   N^\star_{\text{TV}}=\Theta\!\left(\log(1+\kappa)+   \frac{\sigma^2}{\mu\epsilon}\right), \] where $\kappa:=\frac L\mu$ is the condition number.   Note that this complexity bound is simultaneously tight for the condition number $\kappa$ and accuracy $\epsilon$. Besides, our tight complexity bound is adaptive to noiseless setting $\sigma=0$, which is $ N^\star_{\text{TV}}=\Theta\
    
[^63]: 基于聚类的均衡采样与数据并行分配的高性能微调方法

    Clustering-Based Balanced Sampling and Allocation with Data Parallelism for High-Performance Fine-Tuning

    [https://arxiv.org/abs/2609.12584](https://arxiv.org/abs/2609.12584)

    CluSTER框架通过梯度空间聚类与数据并行感知的均衡分配构建代表性缩减数据集，在保持模型精度的同时将指令微调训练时间最多缩短69.6%。

    

    面向大语言模型（LLM）的指令微调数据集通常规模庞大、冗余且分布不均衡，这限制了模型的高效适配。朴素的大批量微调会反复包含样本数量过多的组，而对数量较少但信息丰富的样本组覆盖不足，这一现象在跨多GPU的数据并行（DP）场景下尤为明显。我们提出了CluSTER，一个面向数据并行指令微调的聚类感知均衡采样与高效数据缩减框架。CluSTER通过梯度空间聚类和DP感知的均衡分配来构建一个具有代表性的缩减数据集，确保在聚类簇和工作节点两个层面上的双重覆盖，同时通过加权更新保持原始数据分布。因此，CluSTER在保证模型质量不受损的前提下，减少了冗余计算并提升了训练稳定性。在多个指令微调数据集上的实验表明，CluSTER可将训练时间最多缩短69.6%，且几乎没有精度损失。

    arXiv:2609.12584v1 Announce Type: new  Abstract: Instruction-tuning datasets for large language models (LLMs) are often large, redundant, and imbalanced, limiting efficient adaptation. Naive large-batch fine-tuning repeatedly includes overrepresented sample groups while weakly covering underrepresented but informative ones, especially under data parallelism (DP) across multiple GPUs. We propose CluSTER, a Cluster-aware balanced Sampling framework for Training Efficient data Reduction in DP instruction tuning. CluSTER curates a representative reduced dataset through gradient-space clustering and DP-aware balanced allocation, ensuring dual-level coverage across clusters and workers, while preserving the original data distribution by weighted update. As a result, CluSTER reduces redundant computation and improves training stability without compromising model quality. Across multiple instruction-tuning datasets, CluSTER reduces training time by up to 69.6% with almost no accuracy loss comp
    
[^64]: SCOPE-OPSD：面向在线策略自蒸馏的Fisher条件化特权子空间

    SCOPE-OPSD: Fisher-Conditioned Privileged Subspaces for On-Policy Self-Distillation

    [https://arxiv.org/abs/2609.12579](https://arxiv.org/abs/2609.12579)

    SCOPE-OPSD通过将师生残差投影到基于Fisher敏感度估计的冻结低秩子空间，为在线策略自蒸馏引入了一个结构化的第二监督通道，在无需额外采样或推理模块的情况下，于多个Qwen3模型上稳定超越了纯OPSD。

    

    在线策略自蒸馏（OPSD）通过以解为条件的自教师模型对学生生成的前缀进行评分，但仅通过下一个词元的概率传递监督信号。我们探讨对齐的最后一层差异是否能提供一个有用的第二监督通道，以及如何测试该通道而不将其几何结构与辅助强度相混淆。SCOPE-OPSD将特权的师生残差投影到一个冻结的秩为64的因子上，该因子由残差协方差和语言模型输出头的Fisher敏感度估计得到。该方法复用了OPSD本身已需要的前向计算，既不增加额外的采样rollout，也不增加推理时的模块。作为对照的匹配随机基线保留了结构化因子的秩和非零谱，并使用逐臂的梯度RMS校准，从而分离出数据相关方向本身的效果。在Qwen3-1.7B、4B和8B模型完整的25/50/75/100步训练轨迹上，结构化方法的性能始终不低于纯OPSD，并在12个实验设置中的11个上取得了严格提升。

    arXiv:2609.12579v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) scores student-generated prefixes with a solution-conditioned self-teacher, yet transfers supervision only through next-token probabilities. We ask whether the aligned final-layer discrepancy offers a useful second channel, and how to test that channel without confusing its geometry with auxiliary strength. SCOPE-OPSD projects the privileged teacher-student residual onto a frozen rank-64 factor estimated from residual covariance and language-model-head Fisher sensitivity. It reuses the forwards already required by OPSD and adds neither rollouts nor inference-time modules. A matched Random control preserves the structured factor's rank and nonzero spectrum and uses per-arm gradient-RMS calibration, isolating the effect of the data-dependent orientation. Across the complete 25/50/75/100-step trajectories for Qwen3-1.7B, 4B, and 8B, Structured is never below Pure OPSD, with strict gains in 11 of the 12 mod
    
[^65]: TokenMapper：迈向可互操作语音Token转换的一步

    TokenMapper: A Step Toward Interoperable Speech Token Translation

    [https://arxiv.org/abs/2609.12563](https://arxiv.org/abs/2609.12563)

    提出TokenMapper框架，能够在离散域中直接实现异构语音分词器之间（包括单码本与多码本表示）的token到token转换，避免了先解码为波形音频再重新编码所带来的延迟和信息损失。

    

    神经音频编解码器将语音离散化为token序列，但由此产生的token空间在词表和码本结构上各不相同，阻碍了模型之间的直接通信。这一限制影响了诸如会话语音代理和语音到语音翻译系统等需要多个语音模型相互交互的应用。因此，在语音系统之间传输信息通常需要先解码为波形音频，再用第二个分词器重新编码，这会增加延迟并引入潜在的信息丢失。为了解决这些限制，我们提出了TokenMapper，这是一个方向感知的框架，可在离散域中直接进行异构语音分词器之间的token到token转换。TokenMapper支持结构不匹配的token空间，包括在共享有效token速率下单码本与多码本表示之间的映射。在GLM-4-Voice、MiMi和Dua（摘要在此处截断）上的实验……

    arXiv:2609.12563v1 Announce Type: new  Abstract: Neural audio codecs discretize speech into token sequences, but the resulting token spaces differ in vocabulary and codebook structure, preventing direct communication across models. This limitation affects applications such as conversational voice agents and speech to speech translation systems where multiple speech models must interact. As a result, transferring information between speech systems typically requires decoding to waveform audio and re-encoding with a second tokenizer, increasing latency and introducing potential information loss. To address these limitations, we present TokenMapper, a direction aware framework for direct token to token translation between heterogeneous speech tokenizers in the discrete domain. TokenMapper supports structurally mismatched token spaces, including mappings between single codebook and multi codebook representations, under a shared effective token rate. Experiments on GLM-4-Voice, MiMi and Dua
    
[^66]: 固定量化混合专家实例池上的质量约束路由

    Quality-Constrained Routing over a Fixed Pool of Quantized Mixture-of-Experts Instances

    [https://arxiv.org/abs/2609.12550](https://arxiv.org/abs/2609.12550)

    提出在固定的量化混合专家实例池内，利用基于请求预测的脆弱性加权困惑度（FWP）指标，在质量退化预算和实例容量约束下将请求路由至吞吐量最优实例的方法。

    

    量化混合专家服务可以为同一基础模型保存多个预物化实例，但量化损伤在不同请求和不同位宽之间差异悬殊。由于实例物化和副本数量会消耗内存且需要缓慢的重新配置，我们将它们视为上游的资源配置决策，并在固定常驻实例池内研究路由问题。在这一固定池边界内，我们将每个请求路由至在类别级预期质量退化预算与实测实例容量约束下、可最大化建模吞吐量的实例。为预测这种请求特定的风险，我们引入了FWP（脆弱性加权困惑度，Fragility-Weighted Perplexity），该指标基于参考实例预填充阶段的提示词元计算得出，并针对候选实例的退化进行校准。FWP的基础是精确的双专家亲和性—脆弱性分解，以及条件多层top-k展开，其偏置项、交互项、路由变化项、可分离项和高阶项保持……

    arXiv:2609.12550v1 Announce Type: new  Abstract: Quantized Mixture-of-Experts (MoE) services can hold several pre-materialized instances of one base model, but quantization damage varies sharply across requests and bitwidths. Because instance materialization and replica counts consume memory and require slow reconfiguration, we treat them as upstream provisioning decisions and study routing within a fixed resident pool. Within this fixed-pool boundary, we route each request to maximize modeled throughput under a class-level expected quality-degradation budget and measured instance capacities. To predict this request-specific risk, we introduce FWP (Fragility-Weighted Perplexity), computed from prompt tokens on a reference-instance prefill and calibrated to candidate-instance degradation. Underlying FWP is an exact two-expert affinity--fragility decomposition and a conditional multi-layer top-$k$ expansion whose bias, interaction, route-change, separability, and higher-order terms remai
    
[^67]: GSF-χ：面向手性图Transformer的全局立体化学场

    $\text{GSF-}\chi$: Global Stereochemical Fields for Chiral Graph Transformers

    [https://arxiv.org/abs/2609.12532](https://arxiv.org/abs/2609.12532)

    提出GSF-χ图Transformer，利用立体化学单元生成的全局相位场和手性RoPE旋转机制，使分子编码模型能够区分镜像对映异构体，同时保持对原子标记和本征旋转的不变性。

    

    对映异构体共享相同的原子、化学键和成对距离，却在手性环境中可能表现出不同的性质，因此分子编码器必须尊重原子重标记和本征旋转，同时又不能对镜像反射变得不敏感。我们提出了GSF-χ，这是一种图Transformer，其中立体化学单元调制所有成对相互作用，而不是将某个原子特殊化。每个中心手性或轴向手性的立体化学单元会在所有原子上生成一个反射偶的相位场，一个手性赝标量χ设定潜在query-key块上相对旋转的方向，从而形成一种“手性RoPE”，使镜像反射会将其反转而非保持不变。一个C₂投影将镜像偶的ECD峰计数和位置与镜像奇的峰符号分离开来。我们在明确的规范角色条件下证明了该算符的偶-奇分解及其在标注反转、置换和单元顺序下的恒等性质；性质测试和坐标反射审计…

    arXiv:2609.12532v1 Announce Type: new  Abstract: Enantiomers share atoms, bonds, and pairwise distances yet can behave differently in chiral environments, so molecular encoders must respect atom relabelings and proper rotations without becoming blind to reflection. We introduce GSF-$\chi$, a graph transformer in which stereogenic units modulate all pairwise interactions rather than single out one atom as special. Each central or axial stereogenic unit creates a reflection-even phase field over all atoms, a handedness pseudoscalar $\chi$ sets the direction of a relative rotation on latent query--key blocks, giving a \textbf{Chiral-RoPE} that reflection inverts rather than leaves fixed. A $C_2$ projection separates mirror-even ECD peak counts and positions from mirror-odd peak signs. We prove the operator's even--odd decomposition and its annotation-inversion, permutation, and unit-order identities under explicit canonical-role conditions; property tests and a coordinate-reflection audit
    
[^68]: 时间循环偏好更少的层数

    Temporal Recurrence Favors Fewer Layers

    [https://arxiv.org/abs/2609.12531](https://arxiv.org/abs/2609.12531)

    该研究发现时间循环能够替代每步内部的深层结构，使最优计算分配显著转向更少的层数，并在性能相当或更优的情况下节省层内计算。

    

    在流式任务中，循环模型可以在时间维度上传递潜在计算，使每次更新都能建立在先前产生的表示之上。这引出了一个基本问题：一旦时间循环提供了跨步骤的顺序计算，每一步内部还需要多少深度？先前的工作已经表明，循环可以使浅层模型具有竞争力。我们则将这一问题作为计算分配问题来研究，在多个计算预算下变化步内深度、专家宽度以及每层并行专家的数量。对于每个预算，我们在每步计算量大致匹配的条件下，比较观察到的最佳循环与非循环计算分配及其所达到的性能。在Sokoban和自回归FineWeb语言建模任务中，我们发现时间循环使最佳计算分配显著偏向更少的层数，同时性能相当或更优。

    arXiv:2609.12531v1 Announce Type: new  Abstract: In streaming tasks, recurrent models can carry latent computation across time, allowing each update to build on representations produced earlier. This raises a basic question: once temporal recurrence provides sequential computation across steps, how much depth is still needed within each step? Prior work has shown that recurrence can make shallow models competitive. We instead study this question as a compute-allocation problem, varying within-step depth, expert width, and the number of parallel experts per layer across several compute budgets. For each budget, we compare the best observed recurrent and non-recurrent allocations and the performance they achieve under approximately matched per-step computation. Across Sokoban and autoregressive FineWeb language modeling, we find that temporal recurrence shifts the best observed compute allocation toward substantially fewer layers, with comparable or better performance.
    
[^69]: 面向战略级两级备件网络设计的学习增强优化方法

    Learning-Augmented Optimization for Strategic Two-Echelon Spare Parts Network Design

    [https://arxiv.org/abs/2609.12524](https://arxiv.org/abs/2609.12524)

    提出一个结合图神经网络集成、变邻域搜索和集合划分重组的保守优化框架，通过采用集成预测节省值的下分位数来抑制代理模型的乐观偏差，从而高效求解评估代价高昂的两级备件库存网络战略设计问题。

    

    我们研究了二级备件库存网络的战略设计问题，其中评估每个候选网络拓扑结构都需要运行一个计算代价高昂的库存优化模型。该设计将数百个站点划分为可行的集群，并为每个集群选择一个中央补货站点，以在维持服务水平的同时降低成本。由于优化器倾向于选择预测节省较高的候选方案，它可能会利用过于乐观的代理模型误差。我们开发了一个保守的框架，将图神经网络集成、变邻域搜索和集合划分重组相结合。代理模型在精确的集群评估结果上进行训练，同时利用集成预测节省值的下分位数来引导搜索，从而限制过度乐观。搜索过程中发现的集群通过集合划分方法并使用基于代理模型的目标系数进行重组。最终得到的网络采用精确库存模型进行评估，且仅对此评估…

    arXiv:2609.12524v1 Announce Type: cross  Abstract: We study the strategic design of a two-echelon spare-parts inventory network where evaluating each candidate topology requires an expensive inventory optimization model. The design partitions hundreds of sites into feasible clusters and selects a central replenishment site for each cluster to reduce costs while maintaining service levels. Because the optimizer favors candidates with high predicted savings, it can exploit optimistic surrogate errors.   We develop a conservative framework combining a graph neural network ensemble, variable neighborhood search, and set-partitioning recombination. The surrogate is trained on exact cluster evaluations, while a lower quantile of ensemble-predicted savings guides the search to limit optimism. Clusters found during the search are recombined through set partitioning using surrogate-based objective coefficients. The resulting network is evaluated with the exact inventory model, and only this eva
    
[^70]: 一种用于随机微分方程终端分布律估计的分裂方法

    A Splitting Method for SDE Terminal-Law Estimation

    [https://arxiv.org/abs/2609.12513](https://arxiv.org/abs/2609.12513)

    本文全面研究了通过分裂部分路径生成路径树来提高随机微分方程终端分布估计效率的方法，以Kolmogorov-Smirnov距离为精度度量确定了模拟预算趋于无穷时的极限误差，并刻画了由渐近优化问题驱动的最优分裂策略。

    

    在许多涉及随机微分方程的场景中，包括基于扩散的生成式人工智能，我们的目标是准确地从终端分布中生成样本。通常，这是通过生成独立同分布的扩散路径样本来实现的。在给定固定模拟预算的情况下，一种提高效率的合理方法可能是通过适当分裂的部分路径来生成路径树。这暗示了性能的提升，但人们会担心由此引入的依赖性问题。在本文中，我们全面研究了这一问题。以Kolmogorov-Smirnov距离作为精度度量，我们确定了当模拟预算增至无穷时相关经验分布的极限误差。我们刻画了一种由相应渐近优化问题所启发的分裂策略。理论结果揭示了该问题优雅的内在结构。实际实现包含两个阶段，

    arXiv:2609.12513v1 Announce Type: cross  Abstract: In many settings involving stochastic differential equations, including in diffusion based generative AI, our aim is to accurately generate samples from a terminal distribution. Typically, this is done by generating i.i.d. samples of diffusion paths. Given a fixed simulation budget, a reasonable way to gain efficiency may be to instead generate a tree of paths through appropriately split partial paths. This suggests improved performance, but one worries about the injected dependence. In this paper, we study this issue comprehensively. With Kolmogorov-Smirnov distance as a measure of accuracy, we identify the limiting errors of the associated empirical distributions as the simulation budget increases to infinity. We characterize a splitting strategy motivated by a corresponding asymptotic optimization problem. The theoretical results bring out the elegant underlying structure in the problem. Practical implementation involves two phases,
    
[^71]: 一种用于异构联邦电信网络中客户流失预测的差分隐私联邦近端优化框架

    A Differentially Private Federated Proximal Optimization Framework for Customer Churn Prediction in Heterogeneous Federated Telecom Networks

    [https://arxiv.org/abs/2609.12470](https://arxiv.org/abs/2609.12470)

    该论文提出了一种基于差分隐私的联邦近端优化框架，能够在保护数据隐私的前提下，应对电信运营商间客户数据异构（非独立同分布）的挑战，实现准确的客户流失预测。

    

    客户流失是电信行业面临的主要问题之一。为了预测客户流失，传统的集中式机器学习方法被广泛使用。这种集中式方法要求将客户数据存储在中央数据库中，这引发了隐私问题，并可能违反数据保护法规。联邦学习通过允许多个电信运营商在不传输原始客户数据的情况下协同训练全局模型来解决这一问题。然而，现实世界中的客户数据通常是异构的（非独立同分布，non-IID），这可能会对标准联邦学习的性能产生负面影响。此外，训练好的模型也可能遭受隐私攻击。为了解决这些问题，我们提出了一种基于差分隐私的联邦近端优化（FedProx）框架。所有实验均在两个公开可用的电信客户流失数据集上进行。我们训练了联邦平均算法

    arXiv:2609.12470v1 Announce Type: new  Abstract: Customer churn is one of the major issues in the telecommunication industry. To predict customer churn, conventional centralized machine learning approaches have been widely used. This centralized approach requires customer data to be stored in a central repository, which raises privacy concerns and may violate data protection regulations. Federated learning addresses this problem by allowing multiple telecom operators to collaboratively train a global model without transferring their raw customer data. However, real-world customer data are often heterogeneous (non-IID), which may negatively affect the performance of standard federated learning. Trained models can also suffer from privacy attacks. To address those issues, we propose a Differentially Private (DP) based Federated Proximal optimization (FedProx) framework. All experiments were performed on two publicly available telecom churn datasets. We trained Federated Averaging (FedAvg
    
[^72]: 线性指数二次高斯协方差控制

    Linear Exponential Quadratic Gaussian Covariance Steering

    [https://arxiv.org/abs/2609.12463](https://arxiv.org/abs/2609.12463)

    本文提出并分析了连续时间下的线性指数二次高斯（LEQG）协方差控制问题，证明最优线性状态反馈控制器由一个通过求解编码风险敏感度参数隐式依赖的代数方程的对称矩阵参数化，并将风险中性情形的现有结果显著推广。

    

    我们提出并分析了在给定截止期限（有限时间范围）内连续时间下的线性指数二次高斯（LEQG）协方差控制问题。该问题的解可以看作线性二次设定下高斯端点之间的风险敏感薛定谔桥。与风险中性情形不同，LEQG协方差控制控制器——仍然是线性状态反馈——不再能写成闭式解形式。我们证明了最优控制器由一个对称矩阵参数化，该矩阵通过求解一个编码了对风险敏感度参数隐式依赖的代数方程得到。我们解释了该最优控制器的结构如何显著推广了风险中性情形的现有结果。基于这些结果，对于噪声与输入通道匹配的情形，我们证明了在k邻域内LEQG协方差控制问题解的存在唯一性。

    arXiv:2609.12463v1 Announce Type: cross  Abstract: We formulate and analyze the linear exponential quadratic Gaussian (LEQG) covariance steering problem in continuous time over a given deadline (finite time horizon). The solution for this problem can be seen as a risk-sensitive Schr\"{o}dinger bridge between Gaussian endpoints in the linear quadratic setting. Unlike the risk-neutral case, the LEQG covariance steering controller--still a linear state feedback--can no longer be written in closed form. We show that the optimal controller is parameterized by a symmetric matrix solving an algebraic equation that encodes the implicit dependence on the risk-sensitivity parameter. We explain how the structure of this optimal controller significantly generalizes the existing results for the risk-neutral case. Building on these results, for the matched noise and input channel case, we prove the existence-uniqueness of solution for the LEQG covariance steering problem in the neighborhood of the k
    
[^73]: SAGE-Loop：具有试错-纠正与自适应集成的可靠闭环LLM驱动AutoML

    SAGE-Loop: Reliable Closed-Loop LLM-Driven AutoML with Trial-and-Correction and Adaptive Ensembling

    [https://arxiv.org/abs/2609.12455](https://arxiv.org/abs/2609.12455)

    提出了SAGE-Loop，一个具备试错-纠正机制与自适应集成策略的可靠闭环LLM驱动AutoML框架，解决了传统单向流水线缺乏过程级纠错和集成决策静态化的问题。

    

    自动化机器学习正在重塑数据驱动的科学与工业实践，随着大语言模型被引入AutoML，流水线可靠性与自动化效率变得同等重要。然而，现有AutoML仍难以在执行过程中实现即时反馈与自适应优化，因此一旦运行漂移到次优或失败状态，便缺乏过程级别的纠正机制。其根本病因在于单向流水线：中间失败通常被终止或绕过，而固定范式往往只强化模型生成却使集成决策保持静态，从而同时削弱了执行可靠性与对结构多样性的受控使用。这表明LLM驱动的AutoML需要具备试错-纠正-改进的闭环能力，以及基于证据的模型多样性利用方式。为此，我们提出了SAGE-Loop，一个可靠的闭环自适应（摘要内容在此处截断）

    arXiv:2609.12455v1 Announce Type: new  Abstract: Automated machine learning (AutoML) is reshaping data-driven science and industrial practice, and as large language models are introduced into AutoML, pipeline reliability becomes as important as automation efficiency. However, existing AutoML still struggles to realize instant feedback and adaptive optimization during execution, so once a run drifts into a suboptimal or failed state, it lacks a process-level correction mechanism. The fundamental pathology lies in its one-way pipeline: intermediate failures are typically terminated or bypassed, while fixed paradigms often strengthen model generation but leave ensemble decisions static, weakening both execution reliability and the controlled use of structural diversity. This indicates that LLM-driven AutoML needs a closed-loop ability for trial-correction-improvement together with evidence-based use of model diversity. To this end, we propose SAGE-Loop, a reliable closed-loop, self-adapti
    
[^74]: 将视觉基础模型先验与CLIP相结合，实现医学图像中空间感知的少样本异常检测

    Bridging Vision Foundation Model Priors with CLIP for Spatial-aware Few-shot Anomaly Detection in Medical Images

    [https://arxiv.org/abs/2609.12454](https://arxiv.org/abs/2609.12454)

    该论文提出Spatial-FAD框架，通过将DINO等视觉基础模型的空间结构先验注入CLIP特征，弥补CLIP全局对比预训练缺乏空间监督的不足，从而提升少样本医学图像异常检测中的病灶定位精度。

    

    arXiv:2609.12454v1 公告类型：交叉 摘要：诸如CLIP等视觉-语言模型凭借强大的图像-文本语义对齐能力，能够实现有效的少样本医学异常检测（AD）。然而，其全局对比预训练缺乏显式的空间监督，限制了精确的病灶定位。相比之下，诸如DINO等视觉基础模型（VFM）通过自蒸馏和局部到全局一致性学习，获得了空间连贯的图像块表示，能更好地捕捉细粒度的解剖结构。利用这种互补性，我们提出了Spatial-FAD——一个空间感知的少样本医学异常检测框架，通过结合VFM的空间先验与CLIP的语义信息来改善病灶定位。具体而言，我们引入了一种VFM增强的适配器，将从DINO中提取的结构亲和先验注入CLIP特征中。这种结构引导的精炼促使视觉嵌入更好地贴合病灶边界，同时保持语义对齐。为了解决……（原摘要在此处截断）

    arXiv:2609.12454v1 Announce Type: cross  Abstract: Vision-Language Models such as CLIP enable effective few-shot medical anomaly detection (AD) via strong image-text semantic alignment. However, their globally contrastive pretraining lacks explicit spatial supervision, limiting precise lesion localization. In contrast, Vision Foundation Models (VFMs) such as DINO learn spatially coherent patch representations via self-distillation and local-to-global consistency, better capturing fine-grained anatomical structures. Leveraging this complementarity, we propose Spatial-FAD, a spatial-aware few-shot medical AD framework that improves lesion localization by combining VFM spatial priors with CLIP semantics. Specifically, we introduce a VFM-enhanced adapter that injects a structural affinity prior derived from DINO into CLIP features. This structure-guided refinement encourages visual embeddings to better adhere to lesion boundaries while maintaining semantic alignment. To address the loss of
    
[^75]: 基于多类GRF步态障碍分类的三维数字孪生可视化

    3D Digital Twin Visualization of Multiclass GRF-Based Gait Disorder Classification

    [https://arxiv.org/abs/2609.12442](https://arxiv.org/abs/2609.12442)

    本文提出了一个集成框架，利用双侧地面反作用力信号对健康步态和多种肌肉骨骼障碍进行高精度分类，并结合ε-LRP可解释性分析和基于Blender的三维数字孪生可视化，实现了模型透明度的提升。

    

    自动步态分析需要准确的分类和可解释的输出。我们提出了一个集成框架，利用双侧地面反作用力（GRF）和压力中心（COP）信号对健康步态和多种肌肉骨骼损伤组进行分类。信号在支撑相范围内进行归一化，并使用训练集统计数据进行标准化。在会话级划分下，该模型实现了99.00%的验证准确率和90.07%的测试准确率。类别特定的ε-LRP方法识别了双侧肢体、多个信号分量以及不同支撑相阶段的正负贡献。此外，处理后的GRF信号和模型预测在基于Blender的三维可视化中实现同步，从而能够对步态试验和分类结果进行样本级检查。所提出的框架集成了分类、可解释性和三维可视化，以提高模型的透明度。

    arXiv:2609.12442v1 Announce Type: new  Abstract: Automated gait analysis requires accurate classification and interpretable outputs. We propose an integrated framework for classifying healthy gait and multiple musculoskeletal impairment groups using bilateral ground reaction force (GRF) and center-of-pressure (COP) signals. The signals were normalized over the stance phase and standardized using training-set statistics. The model achieved a validation accuracy of 99.00\% and a test accuracy of 90.07\% under a session-level split. Class-specific $\epsilon$-LRP identified positive and negative contributions across both sides, multiple signal components, and different stance phases. Separately, the processed GRF signals and model predictions were synchronized within a Blender-based 3D visualization, enabling sample-level inspection of gait trials and classification results. The proposed framework integrates classification, explainability, and 3D visualization to improve model transparency
    
[^76]: IMPLY：面向世界模型推演的物理锚定一致性

    IMPLY: Physically Anchored Consistency for World-Model Rollouts

    [https://arxiv.org/abs/2609.12441](https://arxiv.org/abs/2609.12441)

    论文提出 IMPLY 方法，通过反演模拟器从世界模型的推演中读取隐含物理量（如质量与摩擦力），并以两次校准推力为锚定来评估推演集合的物理一致性，从而克服仅依赖推演间自一致性检查的局限，有效识别出忽略物体物理属性而失败的世界模型。

    

    一个世界模型若被问及“以不同速度推一个物体会发生什么”，会生成多个不同的未来推演。如果模型真正“记住”了该物体，这些未来推演在物体上是相互一致的：每一条推演都意味着相同的质量和摩擦力。目前用于审核世界-动作模型的一致性检查只询问模型生成的未来推演之间是否相互一致，其中没有任何一项具备物理知识。我们证明了这并不足够，并提出了替代方案。IMPLY 通过反演模拟器读取每条推演所蕴含的物理量，并根据“同一个物体能在多大程度上解释所有推演”来对一组推演进行评分，同时以模型已观察到的两次校准推力作为锚定。在受控环境中，自一致性会给一个忽略物体、总是预测典型推力的模型打出满分；而锚定方法能将其识别出来（AUROC 为 0.70 对 1.00）。在真实模型——适配该场景的 V-JEPA 2-AC——上，同样的情况再次出现。当给定其自身的校准推力时，该模型能够追踪物体（逐物体相关……

    arXiv:2609.12441v1 Announce Type: cross  Abstract: A world model asked what happens if an object is pushed at several speeds produces several futures. If the model has the object in mind, those futures agree about it: each implies the same mass and friction. The consistency checks now used to vet world-action models ask whether a model's futures agree with each other, and none of them knows any physics. We show that this is not enough, and what to do instead. IMPLY reads the physics each rollout implies by inverting a simulator and scores a set of rollouts by how well one object explains all of them, anchored to two calibration pushes the model has observed. In a controlled setting, self-consistency gives a perfect score to a model that ignores the object and always predicts a typical push; anchoring exposes it (AUROC 0.70 versus 1.00). On a real model, V-JEPA 2-AC adapted to the scene, the same thing happens. Given its own calibration pushes the model tracks the object (per-object cor
    
[^77]: 超越查询：检索信号能否改进自适应多模态RAG路由？

    Beyond the Query: Do Retrieval Signals Improve Adaptive Multimodal RAG Routing?

    [https://arxiv.org/abs/2609.12437](https://arxiv.org/abs/2609.12437)

    研究表明，在自适应多模态RAG路由中，加入检索信号相比仅使用查询的路由器并不能可靠地改进RUN/SKIP决策，因此检索状态特征只有在匹配的仅查询对照实验中证明有改进时才应被认为具有路由价值。

    

    arXiv:2609.12437v1 公告类型：新论文 摘要：自适应RAG（检索增强生成）通常利用检索时信号来决定是否需要运行另一次检索、重排序或多模态处理步骤。我们探讨的问题是：在查询本身已知的情况下，这些信号是否仍能增加路由价值。我们在文档、音频和视频RAG任务中，保持可选动作、路由器类型、训练过程和评估方式完全一致的前提下，比较了匹配的仅查询路由器与查询+检索路由器。在留出的最终评估中，加入所测试的检索信号相比仅查询基线并未带来可靠的路由改进。某些检索信号与后续步骤是否有帮助存在关联，但这种可预测性并不总能转化为更好的RUN/SKIP（运行/跳过）决策。因此，主要启示是方法论层面的：除非检索状态特征在匹配的仅查询对照实验中表现出改进，否则不应认为其具有路由价值。我们的结果并不表明路由或检索状态通常是无用的。

    arXiv:2609.12437v1 Announce Type: new  Abstract: Adaptive RAG often uses retrieval-time signals to decide whether another retrieval, reranking, or multimodal step should run. We ask whether these signals add routing value once the query itself is already known. Across document, audio, and video RAG, we compare matched query-only and query+retrieval routers while holding the optional actions, router family, training procedure, and evaluation fixed. On the held-out final evaluation, adding the tested retrieval signals does not produce a reliable routing improvement over the query-only baseline. Some retrieval signals are associated with whether a later step will help, but that predictability does not consistently lead to bet- ter RUN/SKIP decisions. The main lesson is therefore methodological: retrieval-state features should not be credited with routing value unless they improve over a matched query-only control. Our results do not show that routing or retrieval state is generally useles
    
[^78]: 面向治疗后胶质瘤纵向肿瘤状态代理预测的观测锚定选择性同化方法

    Observation-Anchored Selective Assimilation for Longitudinal Tumor-State Proxy Forecasting in Post-Treatment Glioma

    [https://arxiv.org/abs/2609.12435](https://arxiv.org/abs/2609.12435)

    该论文提出观测锚定选择性同化（OASA）方法，将治疗后胶质瘤的肿瘤状态预测建模为观测感知的数字孪生更新，以观测到的中间肿瘤状态代理作为锚点并选择性应用SegMamba预测的更新，从而提升纵向MRI预测的可靠性。

    

    治疗后胶质瘤患者的MRI提供了用于更新患者特异性肿瘤状态代理估计的序列观测，但多变的影像表现和疾病轨迹使预测变得复杂。我们将预测问题表述为一种观测感知的数字孪生更新，其中中间观测用于锚定患者特异性状态。在203名患者和594个随访时间点中，依据预定义的“无新治疗”标准，从236个候选三元组中保留了120个，并按患者层面划分为81/24/15的训练/验证/测试三元组。每个时间点用从MRI病灶标签导出的取值范围在[0,1]内的连续体素级肿瘤状态代理图来表示。基于SegMamba的单步预测器从多模态源状态张量中预测更新提议。观测锚定选择性同化（OASA）保留观测到的中间代理作为状态锚点，并通过验证选择的层级机制选择性地应用更新

    arXiv:2609.12435v1 Announce Type: new  Abstract: Post-treatment MRI in patients with glioma provides serial observations for updating patient-specific tumor-state proxy estimates, but variable appearances and trajectories complicate forecasting. We formulate forecasting as an observation-aware digital-twin update in which an intermediate observation anchors the patient-specific state. Among 203 patients and 594 follow-up time points, a predefined no-new-treatment criterion retained 120 of 236 candidate triplets, split into 81/24/15 training/validation/test triplets at the patient level. Each time point was represented by a continuous voxel-wise tumor-state proxy map in [0,1] derived from MRI lesion labels. A SegMamba-based single-step forecaster predicted update proposals from multimodal source-state tensors. Observation-Anchored Selective Assimilation (OASA) retained the observed intermediate proxy as the state anchor and selectively applied updates through a validation-selected tiere
    
[^79]: 面向长程大语言模型智能体强化学习的粒度自适应信用分配

    Granularity-Adaptive Credit Assignment for Long-Horizon LLM Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.12424](https://arxiv.org/abs/2609.12424)

    提出GACA，一种无评论家的粒度自适应信用分配方法，通过基于不确定性的逐步权重动态融合轨迹级与步级优势估计，精准识别长程大语言模型智能体任务中驱动结果的关键决策。

    

    强化学习目前是训练大语言模型智能体完成长程任务的标准方法，在这类任务中，数十个相互依赖的动作之后才会出现单个稀疏奖励。以GRPO为代表的无评论家（critic-free）、组相对方法适合这种场景，但它们将一个轨迹级的标量广播到每一步，无法判断究竟是哪个决策导致了最终结果。GiGPO通过对共享锚定状态的时间步进行分组来恢复步级信号，但它在一个固定权重下合并步级与回合级估计，对关键的分支决策和常规的近似确定性转移赋予相同的分辨率。我们认为合适的分辨率应当依赖于状态本身，因此提出GACA——一种粒度遵循基于不确定性的关键性代理指标的无评论家估计器。GACA利用自身rollout已记录的负对数似然为每一步打分，然后以随该分数增长的逐步权重来混合两种优势估计，从而在关键决策处提供更细粒度的信用分配，在常规步骤处保持稳定。

    arXiv:2609.12424v1 Announce Type: new  Abstract: Reinforcement learning is now the standard way to train large language model agents on long-horizon tasks, where dozens of interdependent actions precede a single sparse reward. Critic-free, group-relative methods such as GRPO suit this regime, but they broadcast one trajectory-level scalar to every step and cannot say which decision drove the outcome. GiGPO recovers a step-level signal by grouping time steps that share an anchor state, yet it merges the step- and episode-level estimates under one fixed weight, spending the same resolution on a pivotal branching decision as on a routine, near-deterministic transition. We argue that the right resolution is state-dependent, and propose GACA, a critic-free estimator whose granularity follows an uncertainty-based criticality proxy. GACA scores every step by the negative log-likelihood its own rollout already records, then blends the two advantages with a per-step weight that grows with that 
    
[^80]: 基于随机缩放的加速草图投影牛顿方法的统计推断

    Inference for Newton Methods with Accelerated Sketch-and-Project via Random Scaling

    [https://arxiv.org/abs/2609.12421](https://arxiv.org/abs/2609.12421)

    本文提出了一种基于广义加速草图投影（GAS）求解器的在线草图牛顿方法，并通过随机缩放建立了其平均迭代的渐近正态性，证明了其极限协方差通常能比未加速方法更快地收敛到极小极大最优协方差。

    

    我们研究了一种在线草图牛顿方法，该方法通过一种最先进的草图求解器（称为广义加速草图投影求解器，GAS）在每一步近似牛顿方向，从而缓解了经典二阶方法的计算瓶颈。GAS求解器通过Nesterov动量更新实现加速收敛，从而改进了原始的未加速草图投影求解器，并且支持灵活的投影度量，其适当选择可进一步降低计算成本。基于这一设计，我们建立了平均草图牛顿迭代序列的渐近正态性，并刻画了其极限协方差矩阵。所得的协方差在特定加速参数选择下可恢复未加速草图牛顿方法的协方差，通常情况下（以草图步数计）能更快地收敛到极小极大最优协方差，并且小于……（摘要在此处截断）

    arXiv:2609.12421v1 Announce Type: cross  Abstract: We study an online sketched Newton method that approximates the Newton direction at each step via a state-of-the-art sketching solver, called the generalized accelerated sketch-and-project solver (GAS), thereby mitigating the computational bottleneck of classical second-order methods. The GAS solver improves upon vanilla, unaccelerated sketch-and-project solvers by achieving accelerated convergence through Nesterov momentum updates, and accommodates a flexible projection metric whose proper choice further reduces computational cost. Building on this design, we establish asymptotic normality of the averaged sketched Newton iterates and characterize their limiting covariance matrix. The resulting covariance recovers that of the unaccelerated sketched Newton method under a specific choice of acceleration parameters, converges more rapidly (in the number of sketching steps) to the minimax-optimal covariance in general, and is smaller than 
    
[^81]: MInTRL：离线策略干预可以增强在线策略强化学习

    MInTRL: Off-policy Intervention can boost On-policy RL

    [https://arxiv.org/abs/2609.12419](https://arxiv.org/abs/2609.12419)

    提出最小干预强化学习，通过在在线策略采样中进行稀疏的局部干预（用简短修正替换错误输出后缀），在不牺牲可学习性的前提下扩展探索边界，从而利用离线策略干预提升在线策略强化学习的效果。

    

    带有可验证奖励的强化学习通常以在线策略（on-policy）方式进行，这使训练数据保持接近当前策略，但也将学习限制在策略自身能够发现的轨迹上。另一方面，诸如监督微调等离线策略（off-policy）方法可以利用超出基础模型能力的外部知识，但可能会遭受较大的分布偏移。因此，关键挑战在于如何在不牺牲可学习性的前提下扩展探索。在这项工作中，我们提出了最小干预强化学习，它通过在原本是在线策略的采样轨迹中进行稀疏的局部干预来扩展探索边界。在生成过程中，一个裁判-干预策略会周期性地审查当前策略的输出，用简短的修正替换错误的后缀部分，并立即将控制权交还给策略本身。在训练过程中，MInTRL 采用序列级的优势回归目标，消除……

    arXiv:2609.12419v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards is typically performed on-policy, keeping training data close to the current policy but limiting learning to trajectories that the policy can discover itself. Off-policy methods such as supervised fine-tuning, on the other hand, can leverage external knowledge beyond the base model's capabilities, but may suffer from large distribution shift. The key challenge is thus to expand exploration without sacrificing learnability. In this work, we introduce Minimal Intervention Reinforcement Learning (MInTRL), which expands the exploration frontier through sparse, local interventions in otherwise on-policy rollouts. During generation, a judge-intervention policy periodically reviews the current policy's output, replaces erroneous suffixes with short corrections, and immediately returns control to the policy. During training, MInTRL adopts a sequence-level advantage-regression objective that eliminat
    
[^82]: RiPPLE：面向神经架构搜索的基于早期训练的跨空间性能预测

    RiPPLE: Cross-Space Performance Prediction from Early Training for Neural Architecture Search

    [https://arxiv.org/abs/2609.12418](https://arxiv.org/abs/2609.12418)

    RiPPLE通过将少量锚点架构训练至早期阶段、外推其学习曲线生成代理标签并传播到无标签架构特征上，实现了低成本的跨搜索空间神经架构性能预测。

    

    神经架构搜索（NAS）需要评估候选网络，但为了对整个搜索空间进行排序而充分训练足够多的架构，其成本十分高昂。零成本代理方法可以在初始化阶段对架构进行评分，但其排序质量在不同搜索空间之间差异很大。学习型预测器虽然降低了评估成本，但通常需要对每个候选架构提供完整训练的标签或部分训练的特征。我们提出了RiPPLE（通过前缀传播标签外推进行排序），该方法将部分训练视为为一小组覆盖性锚点架构提供标签的来源。RiPPLE将这些锚点训练至早期前缀阶段，外推其学习曲线以生成代理标签，并将这些标签传播到无标签的架构特征上。早期训练信号仅作为锚点上的标签，而不是每个候选…（摘要原文在此截断）

    arXiv:2609.12418v1 Announce Type: new  Abstract: Neural architecture search (NAS) evaluates candidate networks, but fully training enough architectures to rank an entire space is expensive. Zero-cost proxies score architectures at initialization, yet their ranking quality varies across search spaces. Learned predictors reduce evaluation cost but typically require fully trained labels or partial-training features for individual candidates. We introduce $\textbf{RiPPLE}$, $\underline{\textbf{R}}$anking v$\underline{\textbf{i}}$a $\underline{\textbf{P}}$refix-$\underline{\textbf{P}}$ropagated $\underline{\textbf{L}}$abel $\underline{\textbf{E}}$xtrapolation, which treats partial training as a source of labels for a small coverage set of anchors. RiPPLE trains these anchors to an early prefix, extrapolates their learning curves to surrogate labels, and propagates the labels over label-free architecture features. The early-training signal remains a label on the anchors rather than a per-can
    
[^83]: HoliBench：面向CPS-IoT应用中基础模型的跨平台基准测试与部署工具包

    HoliBench: A Cross-Platform Benchmarking and Deployment Toolkit for Foundation Models in CPS-IoT Applications

    [https://arxiv.org/abs/2609.12412](https://arxiv.org/abs/2609.12412)

    HoliBench是一个模块化的跨平台基准测试与部署工具包，能够联合评估基础模型在准确率、延迟和能耗三个维度上的表现，覆盖从单板计算机到GPU服务器的异构设备，为CPS-IoT应用提供统一的部署决策工作流。

    

    基础模型，包括大语言模型、视觉-语言模型和时间序列基础模型，正日益被部署于嵌入式和边缘平台上，以支持信息物理系统（CPS）与物联网（IoT）应用，在这些场景中，能耗、延迟和内存与任务准确性同等关键。现有的基准测试工具孤立地评估模型能力，在假设算力充足的条件下仅报告准确率，而硬件性能分析工具仍然局限于特定平台且彼此不兼容。因此，用户缺乏一个统一的工作流来在异构设备之间做出部署决策。我们提出了HoliBench，这是一个模块化的基准测试与部署工具包，能够在从单板计算机到GPU服务器的各类平台上联合表征准确率、延迟和能耗。其平台抽象层可对跨设备的测量进行校准，该工具包还支持多种模型模态、推理引擎、并发配置以及现有的评估框架。

    arXiv:2609.12412v1 Announce Type: cross  Abstract: Foundation models, including large language models, vision-language models, and time-series foundation models, are increasingly deployed on embedded and edge platforms for CPS and IoT applications, where energy, latency, and memory are as critical as task accuracy. Existing benchmarking tools evaluate model capability in isolation, reporting accuracy assuming sufficient compute, while hardware profiling tools remain platform-specific and mutually incompatible. As a result, users lack a unified workflow for making deployment decisions across heterogeneous devices. We present HoliBench, a modular benchmarking and deployment toolkit that jointly characterizes accuracy, latency, and energy across platforms from single-board computers to GPU servers. Its platform abstraction layer calibrates cross-device measurement, and the toolkit supports multiple model modalities, inference engines, concurrencies, and existing evaluation harnesses. An i
    
[^84]: 具有标签偏移调整贝叶斯分数的分裂保形预测

    Split Conformal Prediction with Label-Shift-Adjusted Bayesian Scores

    [https://arxiv.org/abs/2609.12386](https://arxiv.org/abs/2609.12386)

    本文提出标签偏移调整贝叶斯分数（LSA score），通过后验预测倾斜恒等式导出非一致性分数，使贝叶斯保形预测在标签偏移下能够保持覆盖率并生成宽度自适应的预测区间。

    

    保形预测可在可交换性假设下提供无分布假设的不确定性量化。然而，标签偏移破坏了这一假设——在标签偏移中，标签的边缘分布发生变化，而给定标签时输入的条件分布保持稳定。在这种偏移下，标准的保形程序不再能维持其预期的覆盖率。现有方法通过重要性加权来解决这一问题，但它们将重加权与忽略预测不确定性的基于残差的非一致性分数相结合，导致所得到的区间宽度是均匀的。贝叶斯保形方法通过利用预测分布来生成自适应区间，但它们在源域的预测分布下评估一致性，这在标签偏移下与目标域不一致。我们提出了标签偏移调整贝叶斯分数（LSA score），这是一种由后验预测倾斜恒等式导出的非一致性分数。

    arXiv:2609.12386v1 Announce Type: new  Abstract: Conformal prediction provides distribution-free uncertainty quantification under exchangeability. However, this assumption is violated by label shift, where the marginal distribution of labels changes while the conditional distribution of inputs given labels remains stable. Under such shifts, standard conformal procedures no longer maintain their intended coverage behavior. Existing approaches address this via importance weighting. They pair the reweighting with residual-based nonconformity scores that ignore predictive uncertainty. The resulting intervals have uniform width. Bayesian conformal methods produce adaptive intervals by leveraging predictive distributions. They evaluate conformity under the source predictive, which is misaligned with the target domain under label shift. We propose the \emph{Label-Shift-Adjusted Bayesian Score} (LSA score), a nonconformity score derived from a posterior predictive tilting identity. This identi
    
[^85]: 基于成对似然比的成员推断攻击

    Membership Inference via Pairwise Likelihood Ratios

    [https://arxiv.org/abs/2609.12367](https://arxiv.org/abs/2609.12367)

    提出了PL-MIA，一种将高斯似然比统计量、总体校准与柯西组合检验相结合的统一成员推断攻击方法，能够有效汇总模型输出的统计信号并从理论上提升攻击能力。

    

    成员推断攻击（MIA）是审计机器学习模型隐私风险的标准工具。给定一个查询点，成员推断攻击旨在确定该点是否被用于训练目标模型。在实践中，这种推断必须依赖于模型输出所暴露的统计信号，例如置信度分数、logits以及中间特征表示。然而，现有方法往往无法有效地汇总和组合这些统计信号。为了解决这一局限，我们提出了成对似然成员推断攻击（PL-MIA），这是一种统一的方法，将高斯似然比（GLR）统计量与总体校准和柯西组合检验相结合。我们从理论上刻画了GLR如何保留方差收缩信号，并建立了总体校准和柯西组合能够提升攻击能力的条件。我们通过查询点与……之间的成对比较获得p值（原文在此处截断）。

    arXiv:2609.12367v1 Announce Type: cross  Abstract: Membership inference attacks (MIAs) are the standard tool for auditing the privacy risks of machine learning models. Given a query point, an MIA aims to determine whether that point was used to train the target model. In practice, such inference must rely on the statistical signals exposed by the model's outputs, such as confidence scores, logits, and intermediate feature representations. However, existing methods often fail to efficiently summarize and combine these statistical signals. To address this limitation, we propose Pairwise Likelihood MIA (PL-MIA), a unified method that combines a Gaussian likelihood-ratio (GLR) statistic with population calibration and the Cauchy combination test. We characterize theoretically how the GLR retains variance-contraction signals and establish conditions under which population calibration and Cauchy combination improve attack power. We obtain $p$-values from pairwise comparisons between the quer
    
[^86]: ICU警报的认证式AI分诊

    Certified AI Triage of ICU Alarms

    [https://arxiv.org/abs/2609.12365](https://arxiv.org/abs/2609.12365)

    本文提出一种带统计认证保证的ICU警报三方分诊方法，以95%置信度保证被抑制警报中真实警报比例不超过用户设定预算，在抑制74.8%误报的同时仅漏掉1.5%真实警报，性能与最强已发表系统相当。

    

    在VTaC基准测试中，71%的室性心动过速警报是误报，但屏蔽一个真实警报可能延误对危险心律失常的识别。我们将警报削减问题重新定义为三方分诊（保留、抑制或延迟），并为这一分析中被视为有害的决策设定统计界限：在独立同分布事件采样下，被抑制警报中真实警报的比例以95%的置信度保持在用户设定的预算以下。由于共享同一波形记录的警报相互依赖，聚类分析被用作敏感性检验。在官方数据划分上，5%的预算在全部三个随机种子中均通过认证，抑制了74.8%的误报警，同时仅屏蔽了1.5%的真实警报，AUROC达到0.953，Challenge Score为83.33，在数值上与已发表的十一个系统中最强者相当。我们的核心发现量化了多重性校正的代价：校正对每个候选都要“收费”，因此更精细的网格可能反而认证得更少。在留出校准下，我们的885单元网格……

    arXiv:2609.12365v1 Announce Type: new  Abstract: In the VTaC benchmark 71% of ventricular-tachycardia alarms are false, but silencing a real one can delay recognition of a dangerous arrhythmia. We reframe alarm reduction as three-way triage (retain, suppress, or defer) and bound the decision this analysis treats as harmful: among suppressed alarms, the fraction that were genuine stays below a user-set budget with 95% confidence, under i.i.d. event sampling. Alarms sharing a waveform record are dependent, so the clustered analysis is a sensitivity check. On the official split a 5% budget certifies in all three seeds, suppressing 74.8% of false alarms while silencing 1.5% of genuine ones, at AUROC 0.953 and Challenge Score 83.33, numerically comparable to the strongest of the eleven published systems. Our central finding measures what multiplicity costs: the correction charges for every candidate, so a finer grid can certify strictly less. Under held-out calibration the 885-cell grid we 
    
[^87]: LatentVerse：一个用于理解多模态潜在表示中共享信息与模态特定信息的框架

    LatentVerse: A Framework for Understanding Shared and Modality-Specific Information in Multimodal Latent Representations

    [https://arxiv.org/abs/2609.12364](https://arxiv.org/abs/2609.12364)

    LatentVerse是一个将Web可视化平台与命令行界面相结合的表示分析框架，它统一了表示质量诊断方法，并通过将嵌入分解为共享成分和模态特定成分来支持多模态潜在表示的分析。

    

    潜在嵌入已成为现代机器学习中的核心数据抽象形式，尤其是在生物医学领域，基础模型越来越多地被用于编码临床文本、医学图像、组学数据和生理信号等多模态数据。然而，这些表示的效用和价值取决于对其质量、结构以及所编码信息的理解。现有的表示评估分析工作流程仍然分散在自定义脚本和孤立指标之中，且最重要的是缺乏多模态分析，限制了可访问性和可重复性。我们提出了LatentVerse，这是一个表示分析资源，它将基于Web的可视化分析平台（用于便捷的、报告驱动的探索）与命令行界面（用于可扩展的技术工作流程）相结合。LatentVerse统一了各种表示质量指标的诊断方法，并通过分解嵌入向量扩展到多模态设置……（原文摘要至此被截断）

    arXiv:2609.12364v1 Announce Type: new  Abstract: Latent embeddings have become a central data abstraction in modern machine learning, especially in biomedicine, where foundation models are increasingly used to encode multimodal data like clinical text, medical images, omics, and physiological signals. However, the utility and value of these representations depends on understanding their quality, structure, and the information they encode. Existing analysis workflows for evaluating representations remain fragmented across custom scripts, isolated metrics, and most importantly lack multimodal analysis, limiting accessibility and reproducibility. We present LatentVerse, a representation analysis resource that combines a web-based visual analytics platform for accessible, report-driven exploration with a command-line interface for scalable technical workflows. LatentVerse unifies diagnostics for various representation quality metrics and extends to multimodal settings by decomposing embedd
    
[^88]: 当连接并不意味着相似：探索SNAP-KG在流式实体集成中的同质性边界

    When Connected Does Not Mean Similar: Charting the Homophily Boundary of SNAP-KG for Streaming Entity Integration

    [https://arxiv.org/abs/2609.12356](https://arxiv.org/abs/2609.12356)

    本文通过将SNAP-KG扩展评估到异质图，揭示其性能的同质性边界：当没有任何同质关系视图可用时聚类质量急剧下降，关键决定因素是关系的同质性而非关系的数量。

    

    SNAP-KG是一个框架，用于在不断增长的知识图谱（KG）中将新到达的实体分配到语义社区，它仅使用实体的原始特征，在推理时无需访问图结构也无需重新训练。该框架在五个多视图基准数据集和一个包含240万节点的OGB-WikiKG2知识图谱上进行了评估。在所有这些数据集中，至少存在一个同质的图视图（即相连节点通常属于同一类别），SNAP-KG在所有数据集上均表现良好。本文探究了在此设定之外会发生什么。我们将评估扩展到三个异质图（Texas、Wisconsin、Chameleon），并测量了每个视图的边同质性。结果表明，当没有任何同质视图可用时，无论是SNAP-KG还是其原始评估中使用的归纳式基线，聚类质量都会急剧下降。决定性能的关键是关系的同质性，而非关系的数量。多视图融合仍然有帮助，但前提是至少存在一个同质关系。

    arXiv:2609.12356v1 Announce Type: new  Abstract: SNAP-KG is a framework for assigning newly arriving entities to semantic communities in a growing knowledge graph (KG) using only their raw features, with no graph access and no retraining at inference time. It was evaluated on five multi-view benchmarks and a 2.4M-node OGB-WikiKG2 KG. In each of these datasets, at least one graph view is homophilous, meaning that connected nodes usually belong to the same class, and SNAP-KG performs well on all of them. This paper asks what happens outside that setting. We extend the evaluation to three heterophilous graphs (Texas, Wisconsin, Chameleon) and measure the edge homophily of every view. When no homophilous view is available, clustering quality drops sharply for both SNAP-KG and the transductive baselines used in its original evaluation. What decides this is the homophily of the relation, not the number of relations. Multi-view fusion still helps, but only when at least one homophilous relati
    
[^89]: ParaRecover：面向并行工具使用智能体错误定位与恢复的过程级基准测试

    ParaRecover: A Process-Level Benchmark for Error Localization and Recovery in Parallel Tool-Use Agents

    [https://arxiv.org/abs/2609.12345](https://arxiv.org/abs/2609.12345)

    该论文提出了ParaRecover，一个基于14种错误类型分类体系、包含10,626个实例的过程级基准测试，并配套SDE评分标准，用于评估多轮并行工具使用智能体在错误定位与恢复方面的能力。

    

    现有的智能体基准测试主要评估最终任务成功与否或工具调用的正确性，对于智能体能否可靠地诊断中间执行失败并从中恢复的能力，评估洞察较为有限。这一局限性在多轮并行工具使用场景中尤为突出，因为错误可能跨依赖分支传播并引发级联失败。我们提出了ParaRecover，一个用于评估多轮并行工具使用智能体错误定位与恢复能力的过程级基准测试。该基准建立在一个细粒度的错误类型分类体系之上，涵盖规划依赖、工具选择和参数匹配等14种错误类型，共包含10,626个实例，横跨两个难度级别。为实现细粒度的、面向过程的评估，我们进一步提出了SDE评分标准，用于衡量智能体执行过程中的结构完整性、诊断推理和演化策略。在十多个主流……（原文在此处截断）

    arXiv:2609.12345v1 Announce Type: new  Abstract: Existing agent benchmarks mainly evaluate final task success or tool-call correctness, providing limited insight into whether agents can reliably diagnose and recover from intermediate execution failures. This limitation becomes particularly critical in multi-turn parallel tool-use scenarios, where errors may propagate across dependent branches and trigger cascading failures. We introduce ParaRecover, a process-level benchmark for evaluating error localization and recovery in multi-turn parallel tool-use agents. Built upon a fine-grained taxonomy of 14 error types covering planning dependencies, tool selection, and argument matching, the benchmark comprises 10,626 instances spanning two difficulty levels. To enable finegrained, process-oriented evaluation, we further propose the SDE rubric, which measures structural integrity, diagnostic reasoning, and evolutionary strategy during agent execution.Experiments across more than ten mainstre
    
[^90]: 一次性幅度剪枝与计算自适应早退的理论保证

    Theoretical Guarantees for One-Shot Magnitude Pruning and Compute-Adaptive Early Exit

    [https://arxiv.org/abs/2609.12337](https://arxiv.org/abs/2609.12337)

    该论文为一次性幅度剪枝与计算自适应早退提供了统一的理论保证，证明了带显式速率的剪枝集中定理，并揭示了早退的泛化误差随计算差距呈幂次衰减的标度规律。

    

    我们通过统一的“部分计算与完整计算”视角研究神经网络的计算量削减问题，该视角在静态机制下体现为一次性幅度剪枝，在自适应机制下体现为早退。在一个渐近单神经元模型中，我们证明了带有显式速率的一次性幅度剪枝集中定理。我们还引入了用于早退的条件感知机，并证明其超额泛化误差随计算差距呈幂次衰减，且当部分计算与完整计算之间的对齐程度趋近于一时，该衰减指数趋于无穷大。随后，我们将分析扩展到深度网络，刻画了剪枝引起的失真如何随网络深度累积，并在神经网络高斯过程模型下推导出冻结主干早退的相应计算-精度权衡。数值模拟证实了所预测的标度律。

    arXiv:2609.12337v1 Announce Type: new  Abstract: We study compute reduction in neural networks through a unified partial versus full computation view, captured by one-shot magnitude pruning in the static regime and early exit in the adaptive regime. In an asymptotic single-neuron model, we prove a concentration theorem for one-shot magnitude pruning with explicit rates. We also introduce the conditional perceptron for early exit and show that its excess generalization error decays as a power of the compute gap, with an exponent that grows to infinity as the alignment between partial and full computations tends to one. We then extend the analysis to deep networks, characterizing how pruning-induced distortions accumulate with depth and deriving a corresponding compute-accuracy tradeoff for frozen-backbone early exit under a neural network Gaussian process model. Numerical simulations corroborate the predicted scaling laws.
    
[^91]: 模拟脱离学习状态的学生以评估基于大语言模型的辅导系统

    Simulating Disengaged Students to Evaluate LLM-based Tutors

    [https://arxiv.org/abs/2609.12331](https://arxiv.org/abs/2609.12331)

    该论文提出了DAS2（脱离感知学生模拟器），一个可复现的部署前评估协议，通过建模五种学习者参与状态（投入、钻系统空子、无效空转、游离任务和混合状态）来评估基于大语言模型的AI辅导系统在不同学生参与状态下的表现。

    

    由计算模型生成的模拟学生为评估人类辅导者和AI辅导者所使用的辅导策略与教学方法提供了一种实用的途径。然而，此类模拟应当考虑到学生脱离学习的行为，包括钻系统空子、无效空转以及与任务无关的行为，因为辅导者可能需要针对不同的学习者状态做出不同的回应。我们提出了脱离感知学生模拟器，这是一个可复现的部署前评估协议，它对五种学习者参与状态进行建模：投入学习、钻系统空子、无效空转、游离于任务之外以及混合状态，并评估AI辅导者在这些状态下的表现。研究使用ASSISTments09数据集，由两名标注者基于匿名化的交互日志摘要独立标注了100个采样的辅导会话。他们达到了84%的一致性（Cohen's kappa = 0.78），并且在双方达成一致的案例中，人类共识标签与DAS2基于规则的标签在81%的案例中相匹配（kappa = 0.75）。将模拟条件设定于……

    arXiv:2609.12331v1 Announce Type: new  Abstract: Simulated students generated by computational models provide a practical way to evaluate tutoring strategies and pedagogical approaches used by human and AI tutors. However, such simulations should account for disengaged behaviors, including gaming the system, wheel-spinning, and off-task behavior, because tutors may need different responses for different learner states. We present Disengagement-Aware Student Simulators (DAS2), a reproducible pre-deployment protocol that models five learner-engagement states: engaged, gaming, wheel-spinning, off-task, and mixed, and evaluates AI tutor performance across these states. Using ASSISTments09, two coders independently labeled 100 sampled tutoring sessions based on anonymized interaction-log summaries. They achieved 84% agreement (Cohen's kappa = 0.78), and among agreed cases, human consensus labels matched DAS2 rule-based labels in 81% of cases (kappa = 0.75). Conditioning simulations on inten
    
[^92]: LoRA-RC：基于低秩自适应的储备池计算

    LoRA-RC: Reservoir Computing with Low-Rank Adaptation

    [https://arxiv.org/abs/2609.12327](https://arxiv.org/abs/2609.12327)

    LoRA-RC通过流式预测误差驱动的低秩校正在线调整储备池循环矩阵，借助谱范数球投影与低通滤波保证收缩性及与路径无关的增量输入-状态稳定性，解决了静态储备池在系统漂移下性能退化的问题。

    

    储备池计算（RC）仅在固定的循环层上训练一个线性读出层，这使其在在线预测中速度快且数据效率高。然而，静态储备池在系统发生漂移时性能会退化，此时仅对读出层进行自适应便不再足够，而对储备池进行无约束的自适应又可能破坏使RC可靠运行的回声状态特性和增量稳定性。本文提出LoRA-RC，通过由流式预测误差驱动的低秩校正来调整循环矩阵。基础储备池和自适应基底在离线阶段固定；在线阶段仅调整一个小型核心矩阵，该矩阵在每一步都被投影到谱范数球上并进行低通滤波。该投影保证了所施加的每个循环矩阵都保持在经认证的收缩集合内，并且沿着每条在线自适应路径为储备池建立了增量输入-状态稳定性界，其速率和增益与路径无关。在一个Lorenz系统上（摘要在此处截断）……

    arXiv:2609.12327v1 Announce Type: cross  Abstract: Reservoir computing (RC) trains only a linear readout over a fixed recurrent layer, making it fast and data-efficient for online prediction. However, a static reservoir degrades under system drift, readout-only adaptation is then insufficient, and unconstrained reservoir adaptation can destroy the echo-state and incremental stability properties that make RC reliable. This paper proposes LoRA-RC, which adapts the recurrent matrix through a low-rank correction driven by streaming prediction errors. The base reservoir and adaptation bases are fixed offline; a small core matrix is adapted online, projected onto a spectral-norm ball, and low-pass filtered at each step. The projection guarantees that every applied recurrent matrix remains within a certified contraction set, and an incremental input-to-state stability bound is established for the reservoir along each online adaptation path, with path-independent rate and gain. On a Lorenz sys
    
[^93]: 情感智能体：面向可穿戴系统的设备端个性化干预推理

    Affective Agent: On-Device Personalized Intervention Reasoning for Wearable Systems

    [https://arxiv.org/abs/2609.12322](https://arxiv.org/abs/2609.12322)

    该论文提出了Affective Agent——一个面向可穿戴设备的三层参考架构，通过将十亿参数以下的小型语言模型与生理证据、上下文及主机管理的结构化记忆演化相结合，在无需云端依赖和用户级重训练的情况下实现了设备端的个性化干预决策推理。

    

    情感计算已经推动了可穿戴设备的状态推断，但关于是否、何时以及如何进行干预的设备端推理仍然具有挑战性。我们提出了Affective Agent，这是一种面向可穿戴级硬件、在不确定性条件下进行个性化干预推理的三层参考架构。它将一个紧凑的十亿参数以下的小型语言模型与生理证据、上下文和用户历史相结合，以决定是否、何时以及如何进行干预，而无需云端依赖或针对每个用户的重新训练。该架构由三个相互作用的层（感知层、个性化层和推理层）组成，通过主机管理的结构化记忆演化来适应个体用户，而非针对每个用户进行权重更新。我们在室内环境质量控制场景中实例化了Affective Agent，并在涵盖生理变化、上下文、信号质量和干预历史的保留式模拟器生成长期场景上对其进行了评估。

    arXiv:2609.12322v1 Announce Type: cross  Abstract: Affective computing has advanced wearable state inference, but on-device reasoning about whether, when, and how to intervene remains challenging. We present Affective Agent, a three-layer reference architecture for personalized intervention reasoning under uncertainty on wearable-class hardware. It combines a compact sub-billion-parameter language model with physiological evidence, context, and user history to decide whether, when, and how to intervene, without cloud dependency or per-user retraining. The architecture is organized into three interacting layers (perception, personalization, and reasoning), adapting to individual users through host-managed structured memory evolution rather than per-user weight updates. We instantiate Affective Agent in indoor environmental quality control and evaluate it on held-out, simulator-generated longitudinal scenarios spanning physiological variation, context, signal quality, and intervention hi
    
[^94]: AIM：面向多智能体多用户大语言模型系统的隐私感知互操作记忆框架

    AIM: A Privacy-Aware Interoperable Memory Framework for Multi-Agent Multi-User LLM Systems

    [https://arxiv.org/abs/2609.12320](https://arxiv.org/abs/2609.12320)

    提出AIM框架，通过动态区分私有与公共记忆并实施索引级访问控制，实现多智能体多用户LLM系统中隐私感知的持久化互操作记忆管理。

    

    传统大语言模型（LLM）的作用范围仅限于单个用户会话，其知识局限于单一对话，无法学习随时间演变的用户偏好。现有的智能体记忆系统虽解决了这一局限，但通常只在单个用户层面运作，限制了可跨用户共享以改善下游响应的公共知识。我们提出了AIM（智能体互操作记忆），这是一个统一的、隐私感知的记忆框架，使多智能体、多用户的LLM系统能够持久地管理私有和共享记忆。AIM动态地将信息分类为私有（仅限于单个用户且他人无法访问）或公共（所有用户均可访问）。它实施索引级访问控制，确保私有记忆只能由其所有者检索，在保护敏感数据的同时，允许有益的共享知识来提升协调性和一致性。

    arXiv:2609.12320v1 Announce Type: cross  Abstract: Traditional large language models (LLMs) are scoped to individual user sessions, limiting their knowledge to a single conversation and preventing them from learning user preferences that evolve over time. Existing agentic memory systems address this limitation but generally operate at the individual-user level, restricting the public knowledge that could be shared across users to improve downstream responses. We introduce AIM (Agentic Interoperable Memory), a unified, privacy-aware memory framework that enables multi-agent, multi-user LLM systems to persistently manage private and shared memory. AIM dynamically classifies information as private, scoped to one user and inaccessible to others, or public, accessible to all users. It enforces index-level access controls so that private memories are retrievable only by their owner, protecting sensitive data while allowing beneficial shared knowledge to improve coordination and consistency. 
    
[^95]: 基于决策流的采样：在大语言模型中无需训练地提取改进的潜在推理路径

    Sampling via Decision-Flow: Training-Free Extraction of Improved Latent Reasoning Paths in Large Language Models

    [https://arxiv.org/abs/2609.12317](https://arxiv.org/abs/2609.12317)

    该论文提出决策流采样（DF-Sample），一种无需训练、无需数据的推理时框架，通过构建分层推理树并进行全局轨迹评估，从大语言模型中提取被标准解码忽视的高质量潜在推理路径，从而无需昂贵的强化学习微调即可提升模型推理能力。

    

    大语言模型（LLM）推理中的一个核心问题是：强化学习（RL）究竟赋予了模型真正的新能力，还是仅仅改变了模型在推理过程中表达已有知识的方式。基于“分布锐化”假设——即强化学习只是将概率质量重新分配到基础模型中已经潜在存在的高奖励轨迹上——我们提出一个问题：能否在不进行昂贵的强化学习微调的情况下解锁这些潜在路径？我们提出了决策流采样，这是一种无需训练、无需数据的推理时框架，它构建分层推理树，对终端节点进行质量评分，并将效用反向传播以指导每个中间分支决策。与仅做出局部逐步选择的传统采样策略不同，DF-Sample在确定路径之前进行显式的全局轨迹评估，从而恢复被标准解码所忽视的高质量但低概率的推理链。在GPQA上……

    arXiv:2609.12317v1 Announce Type: new  Abstract: A central question in LLM reasoning is whether reinforcement learning (RL) instills genuinely new capabilities or merely reshapes how existing knowledge is expressed during inference. Building on the distribution-sharpening hypothesis, which holds that RL reallocates probability mass toward high-reward trajectories already latent in base models, we ask: can we unlock those latent paths without costly RL fine-tuning? We present Decision-Flow Sampling (DF-Sample), a training-free, data-free inference-time framework that constructs a hierarchical reasoning tree, scores terminal nodes for quality, and back-propagates utilities to inform each intermediate branching decision. Unlike conventional sampling strategies that make purely local step-wise choices, DF-Sample performs explicit global trajectory evaluation before committing to a path, recovering high-quality but low-probability reasoning chains that standard decoding overlooks. On GPQA, 
    
[^96]: ESTS参加WMT26：基于路由信息的专家剪枝模型压缩方法

    ESTS at WMT26: Routing-Informed Expert Pruning for Model Compression

    [https://arxiv.org/abs/2609.12310](https://arxiv.org/abs/2609.12310)

    该论文提出利用任务特定路由质量和跨语言路由差异来识别并物理移除GPT-OSS-20B中的低重要性专家，结合恢复微调与MXFP4量化技术，实现了面向中英和阿英翻译任务的高效模型压缩。

    

    我们以ESTS团队名义描述了提交至无约束WMT26模型压缩共享任务的六个参赛作品，该任务涵盖英语-简体中文和英语-埃及阿拉伯语两个翻译方向。我们在每个翻译方向提交了三个压缩工作点，均基于GPT-OSS-20B模型。我们使用任务特定的路由质量对专家进行排序，并利用跨语言路由差异在各层之间分配保留容量，然后物理移除低重要性的专家。所得的专用模型在GPT-5.1生成的合成翻译数据上进行恢复微调，并通过对保留的专家投影权重应用MXFP4量化进一步压缩。我们还为指令条件化的WMT26设置实现了一个鲁棒的推理系统，包括类别推理、输出验证、重试机制、分段回退和源端拥有的JSON重建。在我们提交的六个作品中，参数量范围为41.86亿至77.70亿，打包工件...

    arXiv:2609.12310v1 Announce Type: new  Abstract: We describe six submissions under the team name ESTS to the unconstrained WMT26 Model Compression Shared Task for English--Simplified Chinese and English--Egyptian Arabic. We submit three compression operating points per translation direction, all derived from GPT-OSS-20B. We use task-specific routing mass to rank experts and cross-lingual routing divergence to allocate retained capacity across layers, then physically remove low-importance experts. The resulting specialists are recovery-tuned on GPT-5.1-generated synthetic translation data and further compressed by applying MXFP4 quantization to the retained expert projection weights. We additionally implement a robust inference system for the instruction-conditioned WMT26 setting, including category inference, output validation, retries, segmented fallback, and source-owned JSON reconstruction. Across our six submissions, parameter counts range from 4.186B to 7.770B and packed artifact 
    
[^97]: 基于可解释人工智能的分布式能源（DER）网络网络安全自验证异常检测

    Self-Verifying Anomaly Detection using Explainable AI for Cybersecurity of DER Networks

    [https://arxiv.org/abs/2609.12305](https://arxiv.org/abs/2609.12305)

    本文提出了面向分布式能源网络的ExCYDER框架，通过结合LightGBM与SHAP的可解释人工智能技术构建自验证机制，验证异常检测警报与特征归因的一致性，从而提升网络安全决策的可信度与透明度。

    

    分布式能源（DER）的快速增长显著扩大了现代电网的网络攻击面。此外，攻击技术的日益复杂化要求异常检测系统（ADS）具备准确性、可解释性和可靠性，以支持DER网络安全。虽然基于机器学习的ADS提供了强大的检测能力，但其黑盒特性降低了操作员的信任度，并限制了安全运营中心（SOC）有效解释警报和做出响应的能力，这凸显了对可解释人工智能（XAI）的需求，以确保透明度和运营信心。本文提出了一种专为DER网络定制的基于XAI的异常检测框架（ExCYDER）。该框架采用自验证机制来验证ADS警报，以确保决策的可信度。ExCYDER将LightGBM与SHAP相结合，用于检查每个模型决策是否与其特征归因相一致。

    arXiv:2609.12305v1 Announce Type: cross  Abstract: The rapid growth of Distributed Energy Resources (DERs) has significantly expanded the cyber attack surface of modern power grids. Furthermore, increasing sophistication in attack techniques demands anomaly detection systems (ADS) that are accurate, interpretable, and reliable to support DER cybersecurity. While ML-based ADS provide strong detection capabilities, their black-box nature reduces operator trust and limits Security Operation Center's (SOC) ability to effectively interpret alerts and respond, highlighting the need for explainable Artificial Intelligence (XAI) to ensure transparency and operational confidence. This paper presents an XAI-based anomaly detection framework tailored for DER networks (ExCYDER). The proposed framework uses a self-verifying mechanism that validates ADS alerts to ensure trustworthy decision-making. ExCYDER combines LightGBM with SHAP to check whether each model decision aligns with its feature-attri
    
[^98]: 突破词元上限：蒸馏出更小更强的字节模型

    Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models

    [https://arxiv.org/abs/2609.12303](https://arxiv.org/abs/2609.12303)

    本文首次大规模研究了词元化方案与训练目标（蒸馏 vs. 交叉熵）对约10亿参数模型的影响，提出了两种将词元logits转换为字节logits的方法（Marginalize-It和End-Of-Token），并发现在八个基准测试中词元模型仍优于字节模型。

    

    小模型通常通过与共享相同词元化方案的大模型进行蒸馏而变得更强。然而，随着计算量和数据量的增加，蒸馏得到的字节模型和词元模型在缩放趋势上的表现是否相似？为了进行这一比较，我们引入了两种将词元对数概率（Token Logits）高效转换为字节对数概率（Byte Logits）的方法：1）近似方法：Marginalize-It（边缘化）；2）精确方法：End-Of-Token（词元结束符）。随后，我们首次开展了大规模研究，对仅解码器的稠密Transformer模型进行过度训练，同时改变两个维度：词元化方案（Tokens、Bytes、带eot的Bytes）和训练目标（蒸馏 vs. 交叉熵），系统扫描了参数量约10亿、训练数据量高达1万亿字节的层级参数匹配模型。在涵盖三个类别的八个基准测试上：多选题问答、语言生成和机器翻译，我们发现Token-1B模型优于字节模型（End-Of-Token-1B和Bytes-……

    arXiv:2609.12303v1 Announce Type: new  Abstract: Small models are made more capable through distillation from a larger one that shares their tokenization scheme. However, do distilled byte and token models behave similarly in terms of scaling trends as compute and data increases? To enable this comparison, we introduce two variants to efficiently convert token logits to Byte Logits: 1) approximate: Marginalize-It, and 2) exact: End-Of-Token. We then present the first large scale study of overtraining decoder-only dense transformer models varying two dimensions simultaneously: the tokenization scheme (Tokens, Bytes, Bytes w/ eot) and the training objective (Distillation vs. Cross-Entropy), sweeping layer-parameter-matched models with roughly 1 billion parameters up to 1 trillion bytes of data. Across eight benchmarks spanning three categories: Multiple Choice QA, Language Generation, and Machine Translation, we find that Token-1B models outperform byte models (End-Of-Token-1B and Bytes-
    
[^99]: FRIST：基于fMRI表征信息的共享空间训练改进仅用EEG的单指BCI解码

    FRIST: FMRI Representation Informed Shared-space Training Improves EEG-only Individual-Finger BCI Decoding

    [https://arxiv.org/abs/2609.12298](https://arxiv.org/abs/2609.12298)

    该论文提出FRIST两阶段解码框架，利用fMRI的高空间分辨率信息来指导仅用EEG信号的单指运动解码，无需配对数据即可显著提升BCI的单指解码性能。

    

    手指级别的运动解码对于自然的大脑机接口（BCI）控制非常重要，然而从头皮脑电图（EEG）解码单个手指仍然具有挑战性，因为手指表征在感觉运动皮层中空间上非常接近，并且被容积传导效应所模糊。利用功能磁共振成像的高空间分辨率，我们提出了fMRI表征信息引导的共享空间训练，这是一个两阶段的EEG解码框架：首先从同步EEG-fMRI记录中学习fMRI信息引导的频谱投影，然后使用fMRI衍生的类别几何结构来指导EEG预测的残差修正。FRIST通过共享的手指标签在记录之间传递信息，无需配对试验，并且在推理时仅使用EEG。我们在两类和三类按时间顺序的留出会话设置下，对12名健全受试者在运动执行和运动想象过程中进行了评估。

    arXiv:2609.12298v1 Announce Type: new  Abstract: Finger-level motor decoding is important for naturalistic brain-computer interface (BCI) control, yet individual-finger decoding from scalp electroencephalography (EEG) remains challenging because finger representations are spatially close in the sensorimotor cortex and blurred by volume conduction. Leveraging the high spatial resolution of functional MRI (fMRI), we introduce fMRI Representation-Informed Shared-Space Training (FRIST), a two-stage EEG decoding framework that first learns fMRI-informed spectral projections from simultaneous EEG-fMRI recordings and then uses fMRI-derived class geometry to guide residual refinement of EEG predictions. FRIST transfers information across recordings through shared finger labels without requiring paired trials and uses only EEG at inference. We evaluated 12 able-bodied participants during movement execution (ME) and motor imagery (MI) under two-class and three-class chronological session-held-ou
    
[^100]: 面向少样本传感器故障诊断的鲁棒原型网络

    Robust Prototypical Networks for Few-Shot Sensor Fault Diagnosis

    [https://arxiv.org/abs/2609.12287](https://arxiv.org/abs/2609.12287)

    提出多回合原型网络（MEPN），通过聚合多个互不相交支持回合的原型均值来降低原型方差，从而提升少样本传感器故障诊断的鲁棒性。

    

    工业故障诊断通常仅有少量带标签的故障样本，这使得少样本学习在传感器监测中极具吸引力。标准原型网络简单而有效；然而，在极低样本量的情况下，由于每次决策都依赖于一个很小的支持集，其类原型可能变得不稳定。我们提出了多回合原型网络（MEPN），它从多个互不相交的支持回合中聚合原型，并将其均值作为最终的类别代表，从而在不改变编码器架构的前提下降低了原型方差。我们在DeFACTO传感器数据集上对MEPN进行了评估，采用五分类故障分类任务，并在真实工业测量数据中注入了合成的偏差、漂移、尖峰和噪声故障。在超过100次独立运行中，MEPN在每回合单样本设置下（K=1样本，在N_agg=10个支持回合上聚合）达到了相应的准确率。

    arXiv:2609.12287v1 Announce Type: cross  Abstract: Industrial fault diagnosis often operates with only a handful of labeled fault examples, making few-shot learning attractive for sensor monitoring. Standard prototypical networks are simple and effective; however, their class prototypes may become unstable in the very-low-shot regime because each decision relies on a small support set. We propose \emph{Multi-Episode Prototypical Networks} (MEPN), which aggregate prototypes from multiple disjoint support episodes and use their mean as the final class representative, reducing prototype variance without changing the encoder architecture. We evaluate MEPN on the DeFACTO sensor dataset using five-way fault classification with synthetic bias, drift, spike, and noise faults injected into real industrial measurements. Over 100 independent runs, MEPN reaches \textbf{\SensorOneShotGcpn\%} in the per-episode one-shot setting ($K\!=\!1$ shot, aggregated over $N_{\text{agg}}\!=\!10$ support episode
    
[^101]: 面向基于模型强化学习的摊销式低秩自适应

    Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2609.12278](https://arxiv.org/abs/2609.12278)

    该论文提出CLAW方法，利用超网络在测试时生成低秩LoRA适配器，以单次前向传播的低计算成本实现对世界模型的高表达性环境适应。

    

    世界模型让智能体能够通过预测其动作的后果来进行规划，但环境的变化可能会使这些预测变得不准确。我们研究的问题是：对于来自已知环境族的未知测试时环境，如何仅使用少量交互回合来使世界模型适应新环境。现有方法在计算成本与表达能力（即方法所能产生的模型范围）之间存在权衡。例如，上下文学习计算成本低但表达能力有限，而基于梯度的自适应表达能力强但计算成本高昂。我们提出了CLAW（世界模型的上下文条件化低秩自适应），该方法通过使用超网络在测试时生成低秩（LoRA）适配器来解决这一权衡问题。在预训练阶段，我们模拟对各种环境的适应过程，并联合训练超网络和基础世界模型。在测试时，我们冻结基础模型并通过一次前向传播完成适应。

    arXiv:2609.12278v1 Announce Type: new  Abstract: World models let agents plan by predicting the consequences of their actions, but changes in the environment can make them inaccurate. We study the problem of adapting a world model to an unknown test-time environment, drawn from a known environment family, using only a few episodes of interaction. Existing approaches trade off computational cost against expressivity, i.e., the range of models a method can produce. For example, in-context learning is computationally cheap but limited in expressivity, and gradient-based adaptation is expressive but computationally expensive. We present CLAW (Context-conditioned Low-rank Adaptation of World models), which addresses this tradeoff by using a hypernetwork to generate low-rank (LoRA) adapters at test time. During pretraining, we simulate adaptation to a variety of environments and jointly train the hypernetwork and base world model. At test time, we freeze the base model and use a forward pass
    
[^102]: 基于患者轨迹的强化学习用于电子健康记录基础模型中的临床推理

    Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models

    [https://arxiv.org/abs/2609.12277](https://arxiv.org/abs/2609.12277)

    本文提出一种将EHR基础模型视为患者轨迹生成式策略的强化学习微调框架，通过时间感知的奖励设计提升临床推理能力，使小模型在数据受限场景下超越大型预训练模型。

    

    在纵向患者轨迹上训练的电子健康记录（EHR）基础模型已在多种临床预测任务中展现出强大的性能。然而，其临床推理能力仍然受限于在有限且不完整的EHR数据上进行下一词元预测。为解决这一问题，我们提出了一种强化学习（RL）微调框架，将EHR基础模型视为患者轨迹上的生成式策略。我们将常见的临床预测问题（如再入院预测）形式化为事件条件、时间窗口化的推理任务。随后，我们设计了时间感知、对轨迹展开敏感的奖励机制，以应对有限的展开长度和时间上不确定的结果。我们发现，RL微调在性能上持续优于预训练主干模型和强大的基线方法。值得注意的是，在数据受限的场景下，该框架使较小的模型能够超越更大的预训练模型，并诱导出正向迁移。

    arXiv:2609.12277v1 Announce Type: new  Abstract: Electronic health record (EHR) foundation models trained on longitudinal patient trajectories have demonstrated strong performance across diverse clinical prediction tasks. However, their clinical reasoning capabilities remain constrained by next-token prediction on limited and incomplete EHR data. To address this, we propose a reinforcement learning (RL) fine-tuning framework that treats EHR foundation models as generative policies over patient trajectories. We formulate common clinical prediction problems (e.g., hospital readmission) as event-conditioned, time-windowed reasoning tasks. We then design time-aware, rollout-sensitive rewards to account for finite rollout lengths and temporally inconclusive outcomes. We find that RL fine-tuning consistently improves over pre-trained backbones and strong baselines. Notably, it enables smaller models to surpass larger pre-trained models in data-limited regimes and induces positive transfer ac
    
[^103]: 通过强化学习实现肿瘤异质性下的自适应化疗控制

    Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning

    [https://arxiv.org/abs/2609.12264](https://arxiv.org/abs/2609.12264)

    本研究开发了基于深度强化学习的闭环化疗给药策略，并通过100名患者的虚拟队列验证发现：连续动作的TD3策略在肿瘤缩小效果上更优，而离散动作的DQN策略在患者间给药一致性上更强，揭示了化疗控制中的疗效-一致性权衡。

    

    设计有效的化疗方案受到肿瘤异质性和耐药性的阻碍，这些问题使得基于模型的个体化最优控制难以在不同患者群体中部署。我们开发并比较了闭环深度强化学习（DRL）给药策略，包括在高维异质性肿瘤模型上训练的连续动作空间（TD3）和离散动作空间（DQN）两种策略。这些DRL策略与基于庞特里亚金极大值原理（PMP）推导的开环基准进行了对比评估。我们使用一个包含100名患者的虚拟队列来评估参数异质性下的泛化能力，该队列对生长速率和药物敏感性参数施加了正负10%的均匀扰动。在该队列中，TD3实现了更高的平均肿瘤缩小率，而DQN则表现出更紧密的患者间给药一致性，揭示了本研究中明显的疗效-一致性权衡。我们的仿真假设对所有肿瘤亚群进行完全观测……

    arXiv:2609.12264v1 Announce Type: new  Abstract: Designing effective chemotherapy regimens is hindered by tumor heterogeneity and drug resistance, which complicate the deployment of patient-specific model-based optimal control across diverse populations. We develop and compare closed-loop deep reinforcement learning (DRL) dosing policies with continuous (TD3) and discrete (DQN) action spaces trained on a high-dimensional heterogeneous tumor model. The DRL policies are benchmarked against a Pontryagin's Maximum Principle (PMP)-derived open-loop benchmark. We assess generalization under parametric heterogeneity using a 100-patient virtual cohort with plus or minus 10 percent uniform perturbations in growth and drug-sensitivity parameters. Across this cohort, TD3 achieves higher average tumor reduction, while DQN yields tighter inter-patient dosing consistency, revealing a clear efficacy-consistency trade-off in this study. Our simulations assume full observation of all tumor subpopulatio
    
[^104]: 任务所要求的秩：群组合训练的矩阵记忆的因果秩定律

    The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition

    [https://arxiv.org/abs/2609.12259](https://arxiv.org/abs/2609.12259)

    本文通过群组合测试平台提供了因果证据，证明在单状态瓶颈下训练的矩阵记忆会精确招募任务代数结构所要求的秩，从而将秩定律从标量容量界扩展为表示论定律。

    

    矩阵值记忆使秩成为学习表示的自然预算：一个状态所张成的独立方向数量限制了它能绑定、组合和追踪的内容。我们在一个群组合测试平台上报告了因果证据——该平台在严格的单状态瓶颈和固定解码器（无法掩盖秩）下训练——表明梯度下降恰好招募了任务代数结构所要求的秩。一篇配套论文 [Larson, 2026a] 在 K 对关联绑定测试平台上建立了类似的秩招募和因果必要性模式，其中精确恢复可证明要求状态秩至少为 K；本文继承了该研究工具，并将秩定律从标量容量界扩展为表示论形式的定律。我们朝着嵌入在更大矩阵中的所选最小忠实参考表示进行训练。在跨越可解/不可解分界的五个有限群上的群组合状态追踪任务中，所招募的秩

    arXiv:2609.12259v1 Announce Type: new  Abstract: Matrix-valued memories make rank the natural budget of a learned representation: the number of independent directions a state spans bounds what it can bind, compose, and track. We report causal evidence, on a group-composition testbed trained under a hard single-state bottleneck with a fixed decoder that cannot launder rank, that gradient descent recruits precisely the rank the task's algebra demands. A companion paper [Larson, 2026a] establishes the analogous recruitment and causal necessity pattern on a $K$-pair associative-binding testbed, where exact recovery provably requires state rank at least $K$; this paper inherits that instrument and extends the rank law from a scalar capacity bound to a representation-theoretic one. We train toward chosen minimal faithful reference representations embedded in larger matrices. On group-composition state tracking over five finite groups spanning the solvable/non-solvable divide, the recruited r
    
[^105]: CRFCAN：一种用于亚太赫兹OFDM系统中信道与相位噪声联合估计的复值跨域残差网络

    CRFCAN: A Complex-Valued Cross-Domain Residual Network for Joint Channel and Phase Noise Estimation in Sub-THz OFDM Systems

    [https://arxiv.org/abs/2609.12244](https://arxiv.org/abs/2609.12244)

    提出了一种复值跨域残差网络CRFCAN，通过在残差组中嵌入FFT与逆FFT模块实现时频域间的迭代特征交互，以端到端方式实现亚太赫兹OFDM系统中信道与相位噪声的联合估计。

    

    在亚太赫兹（sub-THz）通信中，超宽带宽与严重相位噪声（PN）损伤的耦合，使得传统的信道与相位噪声联合估计高度复杂且计算成本难以承受。为解决这一问题，我们提出了CRFCAN，一种专为信道与相位噪声联合估计设计的复值残差FFT卷积注意力网络。与现有依赖级联网络、或将神经网络与传统迭代估计器相结合的混合框架的深度学习方案不同，CRFCAN通过一种受物理启发的跨域结构，以真正的端到端方式实现联合恢复。具体而言，快速傅里叶变换（FFT）和逆FFT模块被嵌入到残差组中，以实现时域和频域之间的迭代特征交互，从而同时捕获频率选择性衰落和时变相位失真。此外，还引入了两个专门的残差块……

    arXiv:2609.12244v1 Announce Type: new  Abstract: In sub-terahertz (sub-THz) communications, the coupling of ultra-wide bandwidth and severe phase noise (PN) impairments renders conventional joint channel and PN estimation highly complex and computationally prohibitive. To address this, we propose CRFCAN, a complex-valued residual FFT convolutional attention network designed for joint channel and PN estimation. Unlike existing deep learning schemes that rely on cascaded networks or hybrid frameworks combining neural networks with conventional iterative estimators, CRFCAN performs joint recovery in a truly end-to-end fashion through a physics-inspired cross-domain structure. Specifically, Fast Fourier Transform (FFT) and inverse FFT modules are embedded within residual groups to enable iterative feature interaction across the time and frequency domains, thereby capturing both frequency-selective fading and time-varying phase distortions. In addition, two dedicated residual blocks are int
    
[^106]: PLSP（事前边界空间剖析）：OOD预测优于检测——一种面向机器学习模型可靠性的预测性方法

    PLSP (Pre-hoc Liminal Space Profiling): OOD Prediction over Detection -- An Anticipatory Approach for Machine Learning Model Reliability

    [https://arxiv.org/abs/2609.12225](https://arxiv.org/abs/2609.12225)

    本文提出事前预测框架PLSP，将OOD处理范式从检测转向预测，并引入可信度评分（CREDS）、可信度曲线和可信度热力图，实现对机器学习模型应对分布外数据可靠性的事前评估。

    

    分布外数据对机器学习模型构成重大威胁，常常导致模型在部署过程中失效。所有现有的OOD检测方法都是事后方法，依赖于推理过程中的准确率和AUC-ROC等评估指标，通过测量偏差来间接评估模型对OOD数据的响应。与现有方法不同，本工作提出了一种名为PLSP的事前预测性框架，将范式从OOD检测转向OOD预测。我们做出了几项关键贡献：提出了一种与数据集无关的度量标准——可信度评分（CREDS），用于OOD预测；引入了可信度曲线来研究模型能够达到的最大可信度；引入了可信度热力图（及曲面下体积）来表征模型在不同数据集上的事前行为。这项工作为信号处理领域提供了全新的视角。

    arXiv:2609.12225v1 Announce Type: new  Abstract: Out-of-Distribution (OOD) data poses a significant threat to machine learning models, often leading to model failure during deployment. All existing OOD detection methods are post-hoc, relying on evaluation metrics such as accuracy and AUC-ROC during inference to indirectly assess the model's response to OOD data by measuring deviations. In contrast to existing approaches, the proposed work shifts the paradigm from OOD detection to OOD prediction by proposing a pre-hoc anticipatory framework called PLSP for OOD prediction. We make several key contributions: (a) a dataset-independent metric called the CREDibility Score (CREDS) is proposed for OOD prediction; (b) credibility curves are introduced to study the maximum credibility a model can attain; and (c) credibility heat maps (and volume under surface) are introduced to characterize pre-hoc model behavior across different datasets. This work provides a novel perspective on signal process
    
[^107]: 患者报告的调查数据改善阿片类药物使用障碍的预测

    Patient-Reported Survey Data Improve Prediction of Opioid Use Disorder

    [https://arxiv.org/abs/2609.12224](https://arxiv.org/abs/2609.12224)

    该研究的核心创新在于将患者报告的调查数据与电子健康记录相结合，在所有24个模型-回溯窗口组合中均显著提升了阿片类药物使用障碍首次诊断预测的性能，且调查特征被证明是仅次于EHR的第二重要信息来源。

    

    电子健康记录（EHR）可能无法完整捕捉与阿片类药物使用障碍（OUD）相关的患者报告因素。我们评估了调查数据能否改善对267,747名有阿片类药物暴露记录的"All of Us"参与者首次记录的OUD诊断的预测，其中包括15,287例OUD病例。我们在6个月、12个月和24个月的回溯窗口内，使用逻辑回归、随机森林、XGBoost、LightGBM、多层感知机、LSTM、GRU和Transformer，比较了仅使用EHR的模型与EHR加调查数据的模型。在所有24个模型-窗口组合中，调查数据增强使PR-AUC提高了0.0087至0.0505；最佳的24个月LightGBM模型的PR-AUC从0.6219提升至0.6603。调查数据覆盖率随回溯窗口延长而增加，并因OUD状态不同而存在差异（24个月：OUD阳性组为21.7%，OUD阴性组为60.7%）。置换重要性分析显示，在24个月窗口下，调查特征在两个评估模型中均被列为第二重要的信息域。患者报告数据……（原文截断）

    arXiv:2609.12224v1 Announce Type: new  Abstract: Electronic health records (EHRs) may incompletely capture patient-reported factors associated with opioid use disorder (OUD). We evaluated whether survey data improve prediction of a first recorded OUD diagnosis among 267,747 All of Us participants with documented opioid exposure, including 15,287 OUD cases. We compared EHR-only and EHR+survey models across 6-, 12-, and 24-month look-back windows using logistic regression, random forest, XGBoost, LightGBM, multilayer perceptron, LSTM, GRU, and Transformer. Survey augmentation improved PR-AUC across all 24 model-window combinations by 0.0087-0.0505; the best 24-month LightGBM model improved from 0.6219 to 0.6603. Survey coverage increased with longer windows and differed by OUD status (24 months: 21.7% OUD-positive vs. 60.7% OUD-negative). Permutation analysis ranked survey features as the second most important information domain at 24 months in both evaluated models. Patient-reported dat
    
[^108]: 使用GRACE预测碰撞截面：通过早期融合实现几何残差加合物条件化

    Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion

    [https://arxiv.org/abs/2609.12223](https://arxiv.org/abs/2609.12223)

    本文提出GRACE模型，通过早期融合的几何残差加合物条件化方法调整预训练分子几何编码器，将加合物感知的残差学习目标与编码器内的加合物条件化机制相结合，显著提升了对气相分子离子碰撞截面的三维预测精度。

    

    碰撞截面（CCS）源自离子迁移谱质谱技术，是分子注释的常用描述符。对机器学习模型而言，预测CCS具有挑战性，因为它反映了气相分子离子的大小、形状和电离状态。大多数预测器要么忽略显式的三维结构，要么将加合物类型作为后期处理的类别特征，这限制了模型捕获加合物相关几何效应的能力。我们提出了GRACE（通过早期融合实现几何残差加合物条件化），这是一个三维CCS预测器，通过早期融合的几何残差加合物条件化来调整预训练的分子几何编码器。GRACE结合了两个归纳偏置：相对于加合物感知的物理描述符基线的残差学习目标，以及通过可学习的加合物标记和低秩注意力适配器在编码器内部实现的加合物条件化。我们在包含超过9,000个实验分子的精选数据集上对该模型进行了评估。

    arXiv:2609.12223v1 Announce Type: new  Abstract: Collision cross section (CCS), derived from ion mobility mass spectrometry, is a common descriptor for molecular annotation. Prediction is challenging for machine learning models because it reflects the size, shape, and ionization state of a gas-phase molecular ion. Most predictors either ignore explicit 3D structure or treat adduct identity as a late categorical feature, which limits their ability to capture adduct-dependent geometric effects. We present GRACE (Geometric Residual Adduct Conditioning via Early-fusion), a 3D CCS predictor that adapts a pretrained molecular geometry encoder using geometric residual adduct conditioning via early fusion. GRACE combines two inductive biases: a residual objective relative to an adduct-aware physical descriptor baseline and adduct conditioning within the encoder via a learned adduct token and low-rank attention adapters. We evaluate the model on a curated set of over 9,000 experimental molecule
    
[^109]: QuPAINT：面向量子材料表征的物理感知多模态推理

    QuPAINT: Physics-Aware Multimodal Reasoning for Quantum Material Characterization

    [https://arxiv.org/abs/2609.12202](https://arxiv.org/abs/2609.12202)

    QuPAINT是一个物理感知的多模态框架，通过合成显微图像生成、基于验证标注构建的多模态指令数据集以及物理信息注意力机制，实现对量子二维薄片层数表征的可迁移推理，克服合成到真实的域偏移问题。

    

    利用光学显微镜表征二维（2D）量子材料需要定位剥离的薄片，并根据细微的光学对比度和干涉色确定其层数，从而选择适合器件制备的薄片。然而，模型面临从合成数据到真实数据的域偏移，以及材料、衬底、实验室和成像条件之间的差异变化。我们提出了QuPAINT，一个面向可迁移量子薄片表征的物理感知多模态框架。合成材料框架（Synthia）能够在保持层相关光学行为的同时生成多样化的合成显微图像。基于这些图像，我们构建了QMat-Instruct，一个多模态指令数据集，其图像特定的推理链由经验证的标注生成，并严格限定于可观测的光学线索。QuPAINT通过物理信息注意力机制（Physics-Informed Attention, PIA）整合这些信号，PIA注入衬底相对光学……

    arXiv:2609.12202v1 Announce Type: cross  Abstract: Characterizing two-dimensional (2D) quantum materials by optical microscopy requires localizing exfoliated flakes and determining their layer thickness from subtle optical contrast and interference color to select suitable flakes for device fabrication. However, models face synthetic-to-real domain shifts and variation across materials, substrates, laboratories, and imaging conditions. We present QuPAINT, a physics-aware multimodal framework for transferable quantum flake characterization. The Synthetic Materials Framework (Synthia) generates diverse synthetic microscopy images while preserving layer-dependent optical behavior. Using these images, we construct QMat-Instruct, a multimodal instruction dataset with image-specific reasoning traces generated from verified annotations and constrained to observable optical cues. QuPAINT integrates these signals through Physics-Informed Attention (PIA), which injects substrate-relative optical
    
[^110]: GAUGE：在面向任务智能体的用户模拟评估中，何时不应信任LLM-as-a-Judge

    GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents

    [https://arxiv.org/abs/2609.12191](https://arxiv.org/abs/2609.12191)

    GAUGE协议揭示了一种常见的LLM智能体离线评估闸门存在严重缺陷——LLM评审给出的满意度评分与实际任务成功率几乎完全不相关（被评为满意的对话中有57.5%实际未能完成客户任务），因此不应仅凭主观评分来信任和选择任务导向型LLM智能体。

    

    比较和选择面向任务的LLM智能体越来越依赖于一种低成本的离线评估闸门：基于人设驱动的LLM用户模拟器与每个候选智能体对话，由LLM-as-a-judge对对话记录进行评分，得分较高的智能体被采用。我们提出了GAUGE，一个可复用的离线协议，用于衡量该闸门的排名是否与基于真实可验证奖励的排名相匹配，涵盖来自六家提供商的25个智能体，在τ²-bench和SimulatorArena基准上进行测试，并区分了发布实践中被混淆的两种评估有效性：排名有效性和构念有效性。首先，满意度与成功率之间存在鸿沟：满意度基本不携带关于任务成功的任何信息，因为我们盲评小组评为满意的对话与实际成功之间不相关，其中57.5%的对话未能完成客户任务，这一模式在五类评分者群体、两个基准测试以及我们评估的每个主观维度上都保持一致。其次，w

    arXiv:2609.12191v1 Announce Type: new  Abstract: Comparing and selecting task-oriented LLM agents increasingly relies on a low-cost offline evaluation gate: persona-driven LLM user-simulators converse with each candidate, an LLM-as-a-judge scores the transcripts, and the higher-scoring agent is promoted. We introduce GAUGE, a reusable offline protocol that measures whether this gate's ranking matches a grounded verifiable reward across 25 agents from six providers on the $\tau^2$-bench and SimulatorArena benchmarks, separating two kinds of evaluation validity that release practices conflate: ranking validity and construct validity. First, a satisfaction-success gap: satisfaction carries essentially no information about task success, as conversations rated satisfied by our blind panel are decorrelated from actual success, with 57.5% of them failing the customer's task, a pattern consistent across five rater populations, both benchmarks, and every subjective dimension we rated. Second, w
    
[^111]: 面向氧化物半导体晶体管的智能体化TCAD校准工作流

    Agentic TCAD Calibration Workflow for Oxide Semiconductor Transistors

    [https://arxiv.org/abs/2609.12184](https://arxiv.org/abs/2609.12184)

    首次提出由LLM智能体驱动的TCAD自动校准工作流，通过残差分析、敏感性测试和器件指标验证，实现了底栅IWO氧化物半导体晶体管的自动化实验校准。

    

    实验性TCAD校准对于新兴氧化物半导体晶体管的预测性技术建模至关重要。然而，由于模型歧义性，该过程仍然耗时且高度依赖专家经验。多个物理模型和参数集都可能复现相同的实测转移特性，而仅靠局部拟合无法唯一确定底层的器件物理机制。我们首次演示了针对制备的底栅铟钨氧（BG-IWO）晶体管的智能体化TCAD校准工作流。该工作流从实测转移曲线和器件信息出发，利用测量-TCAD残差和局部敏感性测试来选择有界参数修正或评估额外的物理模型，并且只接受能够改善器件指标的更新。大语言模型（LLM）智能体负责统筹整个工作流，而Sentaurus则负责器件物理仿真。对于2%钨含量的参考器件，智能体建议的五次更新可以带来（摘要在此处截断）

    arXiv:2609.12184v1 Announce Type: cross  Abstract: Experimental TCAD calibration is essential for predictive technology modeling of emerging oxide semiconductor transistors. However, it remains time-consuming and expert dependent because of model ambiguity. Multiple physical models and parameter sets can reproduce the same measured transfer characteristics, while local fitting alone cannot uniquely identify the underlying device physics. We present the first demonstration of an agentic TCAD calibration workflow for a fabricated bottom-gate In--W--O (BG-IWO) transistor. Starting from the measured transfer curve and device information, the workflow uses measurement--TCAD residuals and local sensitivity tests to select bounded parameter corrections or evaluate additional physical models, and accept only updates that improve device metrics. The LLM agent orchestrates the workflow, while Sentaurus governs the device physics. For the 2\%-W reference device, five agent-suggested updates yield
    
[^112]: 面向算法追索的解释驱动主动特征获取方法

    Explanations-Driven Active Feature Acquisition for Algorithmic Recourse

    [https://arxiv.org/abs/2609.12179](https://arxiv.org/abs/2609.12179)

    该论文首次将算法追索与主动特征获取联合建模，利用马尔可夫毯理论统一反事实、半事实和替代事实解释，并提出按单位成本解释价值选择特征的EDFA方法，实现解释驱动的成本高效特征获取。

    

    算法追索方法通常假设预测模型能够获取个体的所有特征。但在实践中，由于特征获取成本高昂，决策往往是在信息不完整的情况下做出的。主动特征获取旨在解决成本受限的预测问题，但现有方法与解释无关：先前的工作只在获取额外特征之后才提供解释，而不是利用解释来驱动特征获取。这项工作颠覆了这一模式，将算法追索与特征获取进行联合处理。我们利用马尔可夫毯理论统一了反事实、半事实和替代事实三类解释，并刻画了随着特征的逐步获取，可用追索方案如何增长。基于这一框架，我们提出了解释驱动的特征获取（EDFA）方法，该方法按照单位成本的解释价值来选择特征。该框架进一步扩展，为……提供了无分布的有效性保证（原文摘要在此处截断）。

    arXiv:2609.12179v1 Announce Type: new  Abstract: Algorithmic recourse methods typically assume that a predictive model has access to all features of an individual. In practice, decisions are often made with partial information, because features are costly to acquire. Active feature acquisition addresses cost-constrained prediction, but existing methods are explanation-agnostic: prior work provides explanations only after acquiring additional features, rather than using explanations to drive acquisition. This work flips that and treats algorithmic recourse and feature acquisition jointly. We use Markov Blanket theory to unify counterfactual, semifactual, and alterfactual explanations and to characterize how available recourse grows as features are acquired. Building on this framework, we propose an Explanation-Driven Feature Acquisition (EDFA) method that selects features by explanatory value per unit cost. The framework is further extended with distribution-free validity guarantees for
    
[^113]: 基于GIS建成环境特征的行人流量估计：一个机器学习框架

    Estimating Pedestrian Volumes from GIS-Derived Built-Environment Features: A Machine Learning Framework

    [https://arxiv.org/abs/2609.12173](https://arxiv.org/abs/2609.12173)

    该研究提出一个基于开放GIS数据的机器学习框架，采用带泊松损失和L1特征选择的直方图梯度提升模型预测城市交叉口高峰行人流量，相比传统负二项GLM基线将交叉验证RMSE降低12%、留出集RMSE降低19%。

    

    交通部门需要整个路网范围内的行人流量估计来优先安排安全投资，但人工计数成本高昂，且仅能覆盖一小部分交叉口。我们提出一个机器学习流程，利用来自开放GIS数据的建成环境、土地利用和街道网络特征，预测俄勒冈州波特兰市101个城市交叉口下午2小时高峰时段的行人流量。从实践中常用的负二项广义线性模型（GLM）出发，我们加入了特征选择、计数感知的梯度提升和重复交叉验证，并在四种交叉验证策略上根据RMSE、MAPE和SMAPE的综合排名选出一个最优配置。最终胜出的模型是采用泊松损失和L1 Lasso特征选择的直方图梯度提升模型，其交叉验证RMSE比GLM基线降低了12%（从89.8降至78.7），留出集RMSE降低了19%（从108.0降至87.9）。相关代码已在GitHub上开源发布。

    arXiv:2609.12173v1 Announce Type: new  Abstract: Transportation agencies need pedestrian volume estimates across entire road networks to prioritize safety investments, yet manual counts are expensive and cover only a small share of intersections. We present a machine learning pipeline that predicts 2-hour PM peak pedestrian volume at 101 urban intersections in Portland, Oregon, from built-environment, land-use, and street-network features drawn from open GIS data. Starting from the Negative Binomial GLM used in practice, we add feature selection, count-aware gradient boosting, and repeated cross-validation, selecting one configuration by a combined rank over RMSE, MAPE, and SMAPE across four cross-validation strategies. The winner, a histogram-based gradient boosting model with Poisson loss and L1 Lasso feature selection, reduces cross-validated RMSE by 12% over the GLM baseline (89.8 to 78.7) and holdout RMSE by 19% (108.0 to 87.9). Code is released on GitHub.
    
[^114]: 文本到图像扩散模型中概念遗忘的认证

    Certifying Concept Unlearning in Text-to-Image Diffusion Models

    [https://arxiv.org/abs/2609.12163](https://arxiv.org/abs/2609.12163)

    该论文提出了一个针对文本到图像扩散模型概念遗忘的认证框架，通过结合统计认证与概念相关嵌入方向上的最坏情况分析，为残余概念泄漏概率提供用户指定置信度下的显式上界，克服了现有基于攻击成功率的经验性评估方法无法量化广泛提示空间中泄漏风险的局限。

    

    现有对文本到图像（T2I）扩散模型中概念遗忘的评估主要依赖于通过自动化对抗性提示搜索获得的攻击成功率。然而，这些指标仅能提供有限查询集上的经验证据，而对更广泛提示空间中的残余泄漏基本未加以量化。这一局限性可能导致高估遗忘的有效性并低估安全风险。为了解决这一空白，我们引入了一种新颖的T2I概念遗忘认证框架，该框架能够对残余概念泄漏提供具有有界误差的高置信度保证。我们的方法将统计认证与沿概念相关嵌入方向的最坏情况分析相结合，在用户指定的置信度水平下推导出泄漏概率的显式上界。我们在三个主要概念类别上评估了我们的框架，即NSFW内容、艺术风格和名人肖像。

    arXiv:2609.12163v1 Announce Type: new  Abstract: Existing evaluations of concept unlearning in text-to-image (T2I) diffusion models primarily rely on attack success rates obtained through automated adversarial prompt search. However, these metrics provide only empirical evidence over a finite set of queries and leave residual leakage over the broader prompt space largely unquantified. This limitation can lead to overestimating unlearning effectiveness and underestimating safety risks. To address this gap, we introduce a novel certification framework for T2I concept unlearning that provides high-confidence guarantees with bounded error on residual concept leakage. Our approach combines statistical certification with worst-case analysis along concept-relevant embedding directions to derive explicit upper bounds on leakage probability under user-specified confidence levels. We evaluate our framework across three major concept categories namely NSFW content, artistic styles, and celebrity 
    
[^115]: 连续隐式模型中的直接拓扑追踪

    Direct Topology Tracking in Continuous Implicit Models

    [https://arxiv.org/abs/2609.12157](https://arxiv.org/abs/2609.12157)

    提出了一种通过直接查询连续隐式模型及其导数来追踪拓扑特征演化的框架，无需将数据重采样到离散网格，从而在避免离散化伪影的同时实现忠实的特征追踪，并适用于解析函数、多元函数逼近和隐式神经表示等多种隐式模型。

    

    我们提出了一个在连续隐式模型中直接追踪拓扑特征的框架。此类模型包括隐式神经表示（INRs）和多元函数逼近（MFAs），正越来越多地被用于表示科学数据，而不受离散网格分辨率限制的约束。它们为复杂场提供了紧凑、平滑且可微的表示，为高性能数据存储、重建和分析开辟了新的机会。给定一个连续隐式模型，我们的方法通过查询模型及其导数来追踪临界点的演化，从而无需重新采样到网格上。这种方法能够实现忠实的特征追踪，同时避免离散化引起的混叠等伪影。我们展示了该框架在各种隐式表示上的通用性，包括解析函数、MFAs和INRs，并表明它可以……

    arXiv:2609.12157v1 Announce Type: cross  Abstract: We present a framework for tracking topological features directly within continuous implicit models. Such models, including implicit neural representations (INRs) and multivariate functional approximations (MFAs), are increasingly adopted to represent scientific data without the resolution constraints of discrete grids. They offer compact, smooth, and differentiable representations of complex fields, enabling new opportunities for high-performance data storage, reconstruction, and analysis. Given a continuous implicit model, our method tracks the evolution of critical points by querying the model and its derivatives, thereby eliminating the need to resample onto a grid. This approach enables faithful feature tracking while avoiding discretization-induced artifacts such as aliasing. We demonstrate the generality of our framework across a range of implicit representations, including analytic functions, MFAs, and INRs, and show that it pr
    
[^116]: 《AI原生6G网络中对抗性意图注入的识别研究》

    On Identifying Adversarial Intent Injection in AI-Native 6G Networks

    [https://arxiv.org/abs/2609.12144](https://arxiv.org/abs/2609.12144)

    该论文针对AI原生6G网络中的对抗性意图注入威胁，定义了细粒度威胁模型，研究了四种恶意意图注入策略，并提出了结合CNN与自编码器的双路径检测框架以识别伪装的恶意意图。

    

    AI原生6G网络使基于意图的网络（IBN）成为前沿技术，能够将高层目标转化为网络配置。然而，这种抽象开启了新的攻击面，主要是对抗性意图注入，即恶意策略伪装在良性意图流之中。如果攻击者采用隐蔽模式进行恶意意图注入，攻击实例的检测可能会变得显著更加困难。考虑到这些问题，我们首先定义了一个细粒度的威胁模型，以刻画AI原生网络中恶意意图注入的威胁。同时，我们研究了四种恶意意图注入策略——隐蔽模式、随机分布、递增频率和递减频率——并提出了一种双路径检测框架：（i）使用TF-IDF特征的卷积神经网络（CNN）进行有监督的恶意意图检测，以及（ii）仅在良性数据上训练的自编码器（AutoEncoder）……

    arXiv:2609.12144v1 Announce Type: cross  Abstract: AI-native 6G networks have brought Intent-Based Networking (IBN) to the forefront, enabling high-level goals to be translated into network configurations. However, this abstraction opens new attack surfaces, primarily adversarial intent injection, where malicious policies are disguised within benign intent flows. The detection of attack instances might become significantly more difficult if the adversaries adopt a stealthy mode of malicious intent injection. With all these in mind, we first define a fine-grained threat model that facilitates the threat of malicious intent injection in an AI-native network. Alongside, we investigate four malicious intent injection strategies$-$ stealth-mode, random distribution, increasing frequency, and decreasing frequency- and propose a dual-path detection framework: (i) a CNN using TF-IDF features for supervised malicious intent detection, and (ii) an AutoEncoder trained exclusively on benign data f
    
[^117]: GUIDE：生成式效用推断与决策引擎

    GUIDE: Generative Utility Inference and Decision Engine

    [https://arxiv.org/abs/2609.12137](https://arxiv.org/abs/2609.12137)

    GUIDE是一种由大语言模型驱动的偏好引出架构，通过结合贝叶斯自适应采样与符号表示学习，在对话中高效推断用户的多维偏好，并将其建立在领域知识基础之上。

    

    测量人类用户的偏好仍然是人工智能对齐中的一个根本性挑战。现有的偏好引出方法难以高效地发现多维偏好，也难以将这些推断准确地建立在领域知识的基础上。为了解决这个问题，我们提出了GUIDE，这是一种由大语言模型驱动的偏好引出架构，通过结合用于问题选择的贝叶斯自适应采样和用于初始化领域特定偏好模型的符号表示学习，在对话过程中推断用户偏好。GUIDE通过对参数化偏好状态的可扩展转换类型系统，将自适应采样推广到多样化的引出问题上。GUIDE通过一个初始化过程产生领域特定的偏好表示，该过程使用基于符号规则的学习来捕获世界知识，并基于决策备选方案的相关数据在偏好维度上设定先验。该架构提供了可观测性……

    arXiv:2609.12137v1 Announce Type: new  Abstract: Measuring the preferences of human users remains a fundamental challenge of AI alignment. Existing elicitation approaches struggle to efficiently discover multidimensional preferences or accurately ground these inferences in domain knowledge. To address this, we introduce GUIDE, an LLM-driven elicitation architecture that infers user preferences through conversations by combining Bayesian adaptive sampling for question selection and symbolic representation learning to initialize domain-specific preference models. GUIDE generalizes adaptive sampling to diverse elicitation questions through an extensible type system of transforms on a parameterized preference state. GUIDE produces domain-specific preference representations through an initialization process using symbolic rule-based learning to capture world knowledge and set priors over preference dimensions grounded in data about decision alternatives. The architecture provides observabil
    
[^118]: 各向同性曲率下通过联合切空间优化实现秩高效的LoRA

    Rank-Efficient LoRA via Joint Tangent-Space Optimization under Isotropic Curvature

    [https://arxiv.org/abs/2609.12123](https://arxiv.org/abs/2609.12123)

    该论文发现LoRA的标称秩并不等于实际利用的表示能力，优化器会显著影响更新中的有效秩，并据此提出ISO-LoRA——一种通过对权重空间诱导切向扰动进行谱下降来联合耦合LoRA因子更新的优化器，从而更充分、更高效地利用秩容量。

    

    低秩适应（LoRA）是一种通过学习低秩权重更新来适配大型预训练模型的有效方法。在实践中，LoRA的秩被用来控制适配器的参数预算和表示能力。我们表明这种观点是不完整的：虽然标称秩决定了表示能力，但优化器决定了在由此产生的权重空间更新中实际使用了多少这种能力。在一项使用LoRA对GPT-2进行适配的案例研究中，我们观察到强烈的秩相关优化器效应。尽管使用相同的标称秩，AdamW通常产生奇异谱集中且有效秩较低的每步更新，而Muon则使用更丰富的方向集合，并且从增加LoRA秩中获益更加一致。这些观察结果促使我们提出ISO-LoRA，这是一种通过在权重空间中对诱导的切向扰动进行谱下降来耦合LoRA因子更新的优化器。ISO-LoRA促进……

    arXiv:2609.12123v1 Announce Type: new  Abstract: Low-Rank Adaptation (LoRA) is an effective approach for adapting large pretrained models by learning low-rank weight updates. In practice, the LoRA rank is used to control an adapter's parameter budget and representational capacity. We show that this view is incomplete: while the nominal rank determines the representational capacity, the optimizer shapes how much of that capacity is used in the induced weight-space updates. In a case study of GPT-2 adaptation with LoRA, we observe a strong rank-dependent optimizer effect. Despite using the same nominal rank, AdamW often produces per-step updates with concentrated singular spectra and low effective rank, whereas Muon uses a richer set of directions and benefits more consistently from increasing LoRA rank. These observations motivate ISO-LoRA, an optimizer that couples the LoRA factor updates through spectral descent on the induced tangent perturbation in weight space. ISO-LoRA promotes up
    
[^119]: 带梯度裁剪与加性噪声的随机梯度方法的几乎必然收敛性分析

    Almost Sure Convergence Analysis of Stochastic Gradient Methods with Clipping and Additive Noise

    [https://arxiv.org/abs/2609.12119](https://arxiv.org/abs/2609.12119)

    本文证明了带梯度裁剪和加性高斯噪声的SGD在平滑性和一致有界噪声假设及标准步长衰减条件下几乎必然收敛，并将该结果扩展到随机重球法和Nesterov加速梯度等动量变体。

    

    带有梯度裁剪和加性噪声的随机梯度下降（SGD）已成为训练机器学习模型的标准技术，特别是在需要鲁棒性或隐私保证的应用中。然而，裁剪会在随机梯度中引入偏差，而加性噪声会引入额外的方差，这使得单个优化轨迹的长期行为难以刻画。在这项工作中，我们证明了在平滑性和一致有界随机梯度噪声的假设下，只要步长满足一些标准的衰减条件，带裁剪和加性高斯噪声的SGD（SGD-CN）几乎必然（a.s.）收敛。我们的分析还扩展到动量变体，如随机重球法和Nesterov加速梯度法，我们证明了通过精心构造的能量函数可以获得类似的收敛保证。这些结果为理解相关方法提供了更强的理论基础。

    arXiv:2609.12119v1 Announce Type: new  Abstract: Stochastic gradient descent (SGD) with gradient clipping and additive noise has become a standard technique for training machine learning models, particularly in applications requiring robustness or privacy guarantees. However, clipping introduces a bias in stochastic gradients, while additive noise introduces additional variance, making the long-run behaviour of individual optimization trajectories difficult to characterize. In this work, we prove that SGD with clipping and additive Gaussian noise (SGD-CN) converges almost surely (a.s.) under smoothness and uniformly bounded stochastic-gradient noise assumptions, provided the step sizes satisfy some standard decaying conditions. Our analysis extends to momentum variants such as the stochastic heavy ball and Nesterov's accelerated gradient, where we show that careful energy constructions yield similar guarantees. These results provide stronger theoretical foundations for understanding th
    
[^120]: 通过控制Radon-Nikodym导数实现基于分数的离群值生成

    Score-based Outlier Generation via Controlling the Radon-Nikodym Derivative

    [https://arxiv.org/abs/2609.12113](https://arxiv.org/abs/2609.12113)

    该论文提出通过似然分布的Radon-Nikodym导数对扩散模型分数函数进行受控缩放，从而无需重新训练即可生成幅度可控的低似然离群值。

    

    离群值对于压力测试算法以及理解系统在罕见条件下的行为非常重要。尽管离群值通常被描述为低似然事件，但现有的生成方法很少显式地控制似然。在这项工作中，我们基于对数似然值的分布引入了一种测度论意义上的离群值概念，该方法保证将更高的概率质量分配给幅度可指定的低似然事件。基于这一表述，我们推导出似然重加权如何修改扩散分数，并利用这一关系来启发对反向时间动力学的受控修改。特别地，似然重加权意味着分数函数的缩放，其控制项由似然分布的Radon-Nikodym导数导出。相应地，更新后的分数函数无需对扩散模型进行重新训练即可获得。我们利用Ornstein（原文摘要在此处截断）

    arXiv:2609.12113v1 Announce Type: new  Abstract: Outliers are important for stress-testing algorithms and understanding system behaviour under rare conditions. Despite being commonly described as low-likelihood events, existing generative approaches rarely control likelihood explicitly. In this work, we introduce a measure-theoretic notion of outliers based on the distribution of log-likelihood values, which is guaranteed to assign higher probability mass to low-likelihood events with a specifiable magnitude. Building on this formulation, we derive how likelihood reweighting modifies the diffusion score and use this relation to motivate a controlled modification of the reverse-time dynamics. In particular, likelihood reweighting implies a scaling of the score function with a control term derived from the Radon-Nikodym derivative of the likelihood distributions. Correspondingly, the updated score function can be obtained with no retraining of the diffusion model. We exploit the Ornstein
    
[^121]: 语言是定量推理的不足基底，重大应用领域需要大型定量模型

    Language Is an Insufficient Substrate for Quantitative Reasoning, and Consequential Domains Need Large Quantitative Models

    [https://arxiv.org/abs/2609.12105](https://arxiv.org/abs/2609.12105)

    该论文论证了语言模型因基于人类描述这一有损编码而根本无法满足定量决策对可复现性、数据溯源和校准不确定性的要求，重大定量决策领域需要专门的大型定量模型而非语言模型。

    

    应用机器学习领域的主流假设是，在重大定量决策（如风险定价、资本配置、患者分诊或遏制网络入侵）方面的进步将随着大型语言模型（LLM）的进步而自然实现。然而，语言模型是在由人类描述所产生的世界表示上进行训练的；描述是对定量记录的有损编码，而这种损失是不可逆的：无论规模大小，任何下游模型都无法从描述中恢复出描述本身未曾编码的信息。我们将这一性质形式化为模型训练所用表示的属性，而非模型能力的属性，并进一步指出重大应用场景对模型的三个额外要求，而语言基底在构造上无法满足这些要求：可复现性、从每个输出回溯到产生它的源记录的谱系（数据溯源），以及经过校准的不确定性。我们认为……

    arXiv:2609.12105v1 Announce Type: cross  Abstract: The prevailing assumption in applied machine learning is that progress on consequential quantitative decisions such as pricing risk, allocating capital, triaging patients, or containing a network intrusion will follow from progress in large language models (LLMs). A language model is trained on a representation of the world that was produced by human description; description is a lossy encoding of the quantitative record, and the loss is irreversible: no downstream model, at any scale, can recover from a description what the description did not encode. We formalize this as a property of the representation on which a model is trained rather than of the model capacity, and we identify three further properties that consequential settings demand of a model and that a language substrate cannot supply by construction: reproducibility, lineage from every output back to the source records that produced. it, and calibrated uncertainty. We argue
    
[^122]: 基于勒让德近似的接收机表面碰撞模式用于分子信号检测

    Receiver-Surface Hit Patterns via Legendre Approximation for Molecular Signal Detection

    [https://arxiv.org/abs/2609.12089](https://arxiv.org/abs/2609.12089)

    该论文提出利用球形接收机表面吸收位置的方向性特征，基于勒让德多项式展开构建分子信号检测器和维特比序列检测器，实现了比仅计数检测更高效、性能更优的几何感知检测。

    

    检测发射机是否正在与接收机进行主动通信是分子通信中的一个基本问题。球形接收机不仅可以测量被吸收分子的数量和到达时间，还可以测量它们在接收机表面的吸收位置。这些位置包含一种方向性特征，而这种特征在仅计数检测中会丢失。在这封信中，我们开发了一种基于接收机表面碰撞密度的勒让德多项式展开的分子信号检测器，并将其扩展为基于勒让德的维特比序列检测器。通过利用分子到达的表面级方向性特征，所提出的检测器提供了几何感知、高效且性能更优的检测方案，可作为仅计数检测的替代方案。

    arXiv:2609.12089v1 Announce Type: cross  Abstract: Detecting whether a transmitter is actively communicating with a receiver is a fundamental problem in molecular communications. A spherical receiver may measure not only the number and arrival times of absorbed molecules, but also their absorption locations on the receiver surface. These locations contain a directional signature that is lost in count-only detection. In this letter, we develop a molecular signal detector based on a Legendre polynomial expansion of the receiver-surface hit density and extend it to a Legendre-based Viterbi sequence detector. By exploiting the surface-level directional signature of molecular arrivals, the proposed detectors provide geometry-aware, efficient, and better-performing alternatives to count-only detection.
    
[^123]: 现代Hopfield模型中的层次原型涌现

    Hierarchical Prototype Emergence in Modern Hopfield Models

    [https://arxiv.org/abs/2609.12079](https://arxiv.org/abs/2609.12079)

    本文证明了具有多项式激活函数的稠密Hopfield网络可以在层次结构的每一级稳定存储记忆，并且通过原型重建，仅需准多项式量级的信息即可实现超越具体记忆和层次组别的泛化。

    

    层次相关性是任何现实数据模型的普遍特征，而关联记忆模型如何学习这些相关性并在其基础上泛化以构建新的有意义的图像，是理解扩散模型等更复杂现代架构的重要一步。我们考虑一个层次化的记忆模型，这些记忆被采样并存储在具有多项式激活函数的稠密Hopfield网络中。我们解析地推导出了该层次结构每一级局部稳定（即局部能量极小值）的条件。我们采用原型重建作为泛化的最小模型，发现仅需准多项式量级的信息即可实现超越特定记忆、甚至超越层次结构中特定组别的泛化。我们在记忆数量、激活函数锐度（多项式阶数）等参数方面观察到了定性相似的相图。

    arXiv:2609.12079v1 Announce Type: cross  Abstract: Hierarchical correlations are a universal feature of any realistic model of data, and the question of how associative memory models may learn these correlations and generalize beyond them to construct new sensible images is an important step towards understanding more complex modern architectures such as diffusion models. We consider a hierarchical model for memories which are sampled and stored in a dense Hopfield network with polynomial activation. We analytically derive conditions for each level of this hierarchy to be locally stable - that is they are local energy minima. We use prototype reconstruction as a minimal model of generalization and we find that it takes only a quasi-polynomial amount of information to generalize beyond particular memories and even particular groups in the hierarchy. We observe a qualitatively analogous phase diagram in the number of memories, sharpness of the activation function (polynomial degree) for 
    
[^124]: 面向机器人工厂的高效视觉-语言-动作管理与推理服务系统

    Efficient Vision-Language-Action Management and Serving for Robot Factories

    [https://arxiv.org/abs/2609.12075](https://arxiv.org/abs/2609.12075)

    提出了首个面向多机器人、多模型请求的VLA服务与管理系统Robion，解决了现有服务系统在SLO约束下无法支持多GPU服务器上多请求多模型执行、且不适用于VLA模型毫秒级阶段的问题。

    

    视觉-语言-动作（VLA）模型通过两阶段设计展现出卓越的机器人操作能力：先是视觉-语言模型（VLM）阶段，随后是动作扩散Transformer（ADiT）阶段。由于机器人必须满足严格的服务级别目标（SLO）以确保安全性，VLA推理本质上是延迟敏感的。满足这些SLO需要高端GPU，然而重量、成本和功耗方面的限制使得无法在机器人本体上集成此类GPU。先前的工作将VLA推理卸载到为众多机器人提供VLA模型服务的边缘服务器上。然而，当前的VLA系统缺乏在SLO约束下于多GPU服务器上支持多请求、多模型执行的能力，而现有的多阶段模型服务系统是为吞吐量和跨独立GPU的阶段分离而优化的，这并不适用于VLA模型的毫秒级阶段。我们设计了Robion，这是首个面向多机器人、多模型请求的VLA服务与管理系统。

    arXiv:2609.12075v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models show high robotic manipulation capabilities via a two-stage design: a Vision-Language Model (VLM) stage followed by an Action Diffusion Transformer (ADiT) stage. Since robots must meet strict Service-Level Objectives (SLOs) for safety, VLA inference is inherently latency-critical. Meeting these SLOs requires high-end GPUs, yet weight, cost, and power constraints preclude integrating such GPUs on-robot. Prior works offload VLA inference to edge servers that serve many robots on VLA models. However, current VLA systems lack support for multi-request, multi-model execution on a multi-GPU server under SLOs, while existing serving systems for multi-stage models are optimized for throughput and stage disaggregation across separate GPUs, which are ill-suited for the millisecond-scale stages of VLA models. We design Robion, the first VLA serving and management system for multi-robot, multi-model requests on 
    
[^125]: 面向压缩与隐私的可扩展离散-连续信道仿真

    Scalable Discrete-to-Continuous Channel Simulation for Compression and Privacy

    [https://arxiv.org/abs/2609.12067](https://arxiv.org/abs/2609.12067)

    提出了一种仅需固定数量随机样本的离散到连续信道仿真方案，其运行时间与信道和输入无关，有效解决了传统信道仿真算法计算成本高和随机停止时间的问题，可应用于压缩与隐私保护场景。

    

    信道仿真近来已成为机器学习系统中的一个重要组件，用于压缩来自指定概率分布的样本。然而，通用的信道仿真算法往往存在计算成本高昂、停止时间随机，或在最坏情况下需要生成无限数量共享随机样本等问题。我们提出了一种用于离散到连续信道的精确与近似仿真方案，该方案仅使用固定数量的随机样本，因此其运行时间与信道和输入无关。与现有的从建议分布生成独立样本序列的信道仿真方案不同，我们的方法从每个潜在目标分布生成一个样本（或固定数量的样本），然后在对样本施加潜在置换后，使用指数竞争进行样本选择。我们的……

    arXiv:2609.12067v1 Announce Type: new  Abstract: Channel simulation has recently emerged as a useful component in machine learning systems where samples from a prescribed probability distribution are to be compressed. Yet, general channel simulation algorithms often suffer from high computational costs, random stopping times or, in the worst case, can require generating an infinite number of shared random samples. We introduce a scheme for both exact and approximate simulation of discrete-to-continuous channels which conversely uses a fixed number of random samples, and therefore has a runtime independent of the channel and the input. Unlike existing channel simulation schemes which generate a sequence of independent samples from a proposal distribution, our approach generates one sample, or alternatively a fixed number of samples, from each potential target distribution. We then apply a latent permutation to the samples before performing sample selection using an exponential race. Our
    
[^126]: 在未来缪子对撞机上使用生成式机器学习实现快速束流诱导背景（BIB）模拟

    Fast BIB simulation at a future Muon Collider with generative machine learning

    [https://arxiv.org/abs/2609.12054](https://arxiv.org/abs/2609.12054)

    该论文首次开发了用于缪子对撞机径迹探测器快速束流诱导背景生成的机器学习模型，通过表格扩散模型和环形样条流模型两种架构，以极低的计算成本生成了与完整模拟高度相似的BIB击中和径迹数据。

    

    来自缪子衰变产物的束流诱导背景（BIB）将是未来缪子对撞机上一种极其严重且不可避免的本底。为了开发稳健的事例重建算法，我们需要大量精确的BIB模拟用于测试。目前BIB模拟受计算资源严重限制：当前用于BIB叠加的模拟样本在统计上大约相当于单个独特事例10%的模拟BIB量，却需要约10^6 HS23·小时的计算时间来生成，并占用约100 GB的磁盘空间。在这项工作中，我们开发了首批用于径迹探测器中快速BIB生成的机器学习模型。我们考虑了两类架构：一类是较慢但保真度更高的表格扩散模型，另一类是速度更快但保真度较低的环形样条流模型。我们发现这两类架构生成的BIB击中和径迹都与现有的完整模拟结果高度相似。

    arXiv:2609.12054v1 Announce Type: cross  Abstract: Beam-induced background (BIB) from muon decay products will be an overwhelming and unavoidable background at a future Muon Collider. In order to develop robust event reconstruction algorithms, we need large amounts of accurate BIB simulation to test on. BIB simulation is currently compute-limited: the simulated sample presently used for BIB overlay, which statistically represents approximately $10\%$ of a single unique event's worth of simulated BIB, requires on the order of $10^6$ HS23$\cdot$hours to generate and occupies approximately $100$ GB on disk. In this work, we develop the first machine learning models for fast BIB generation in tracking detectors. We consider two classes of architectures: a slower but higher-fidelity tabular diffusion model, and a faster but lower fidelity circular spline flow model. We find that both classes of architectures produce BIB hits and tracks that closely resemble those of available full simulatio
    
[^127]: 使用度量感知的Deep Sets学习对撞机事件的几何结构

    Learning the Geometry of Collider Events with Metric-Aware Deep Sets

    [https://arxiv.org/abs/2609.12024](https://arxiv.org/abs/2609.12024)

    本文提出一种度量感知的Deep Sets替代模型，用于快速计算对撞机事件之间的能量移动距离，在实现百分比级别精度的同时大幅提升推理速度，并能显著减少未被显式约束的三角不等式违反。

    

    最优传输为结构化数据提供了一种几何结构，但在利用距离间关系的大规模成对分析中，精确计算代价高昂。学习得到的替代模型速度更快，但不一定能保留这种度量结构。我们开发了一种用于可变大小加权点云之间最优传输的Deep Sets替代模型，该模型强制执行非负性、交换对称性和零自距离，而对三角不等式不加约束。将该模型应用于粒子物理中对撞机事件之间的能量移动距离，度量感知粒子流网络实现了百分比水平的平均绝对百分比误差，同时与所调研的其他精确和近似方法相比显著提高了推理吞吐量。研究发现这些架构约束能改善未被显式强制要求的性质：在10^6个保留事件三元组中，三角不等式违反次数从匹配的无约束网络的199次降至2次。

    arXiv:2609.12024v1 Announce Type: cross  Abstract: Optimal transport gives structured data a geometry, but exact evaluation is costly in large pairwise analyses that exploit relationships among distances. Learned surrogates are faster, but need not preserve this metric structure. We develop a Deep Sets surrogate for OT between variable-size weighted point clouds that enforces non-negativity, exchange symmetry, and zero self-distance, leaving the triangle inequality unconstrained. Applied to the Energy Mover's Distance between collider events in a particle physics application, the Metric-Aware Particle Flow Network achieves percent-level mean absolute percentage error while significantly improving inference throughput over other exact and approximate methods surveyed. The architectural constraints are found to improve properties that are not explicitly enforced: across $10^6$ held-out event triplets, triangle-inequality violations fall from 199 for a matched unconstrained network to 2, 
    
[^128]: 基于强化学习的症状提取

    Reinforcement Learning for Syndrome Extraction

    [https://arxiv.org/abs/2609.12020](https://arxiv.org/abs/2609.12020)

    本文提出结合强化学习与重要性采样的症状提取自动调度方法，在所有规模上均超越现有最先进工具AlphaSyndrome和PropHunt，逻辑错误率平均分别降低25.9%和71.7%，对距离15的表面码最高降低97.8%。

    

    量子纠错的一个关键子任务是提取症状（校验子），若症状非平凡则表明存在错误。提取症状的可能实现方式数量随症状大小呈指数增长，而这些实现方案在容错性（以逻辑错误率衡量）上差异巨大。这构成了一个自然的搜索问题：找到一种逻辑错误率低的实现方案。先前的工作虽已解决此问题，但要么牺牲了解的质量，要么牺牲了可扩展性。本文利用强化学习和重要性采样，在所有规模上都超越了先前的工作。与最先进的自动调度工具AlphaSyndrome和PropHunt相比，我们的工具分别平均降低了25.9%和71.7%的逻辑错误率，其中对于距离为15的表面码，逻辑错误率最高降低了97.8%。

    arXiv:2609.12020v1 Announce Type: new  Abstract: A key subtask of quantum error correction is to extract a syndrome that, if nontrivial, signals an error. The number of possible ways to extract a syndrome grows exponentially with the syndrome size, and these implementations vary greatly in fault tolerance, as measured by their logical error rates. This creates a natural search problem: find an implementation with a low logical error rate. Previous work solves this problem but sacrifices either solution quality or scalability. In this paper, we use reinforcement learning and importance sampling to outperform previous work at all scales. Compared with the state of the art automatic scheduling tools AlphaSyndrome and PropHunt, our tool reduces the logical error rate by 25.9\% and 71.7\% on average, respectively, culminating with a reduction of 97.8\% for a surface code with distance 15.
    
[^129]: 基于多保真度时延神经网络与物理信息残差学习的可靠铁路转向架响应预测

    Toward Reliable Railway-Bogie Response Prediction Using Multifidelity TDNN and Physics-Informed Residual Learning

    [https://arxiv.org/abs/2609.12018](https://arxiv.org/abs/2609.12018)

    该论文提出了一种多保真度铁路转向架响应预测方法，将多体仿真视为低保真度信息、滚振试验台测量视为高保真度证据，通过时延神经网络与物理信息残差学习相结合，提升转向架响应预测在未经测试工况下的可靠性。

    

    铁路工程师需要能够在无法穷尽测试的运行工况下预测车辆响应的仿真模型。与代表性测量结果的一致性提供了必要的证据，但在有限工况下的标定并不能保证其他工况下的准确性。我们提出了一种多保真度铁路转向架响应修正方法，该方法将多体仿真历程视为低保真度信息，将滚振试验台测量视为高保真度证据。该方法针对多通道转向架响应历程，将以试验为锚定的保真度分配与物理信息驱动的偏差学习相结合。时延神经网络（TDNN）用于表征随工况变化的仿真趋势，通过开发阶段拟合的幅值对齐定义低保真度基线。随后，残差修正网络对该基线未能解释的可复现响应分量进行建模，并将其叠加到基线之上。

    arXiv:2609.12018v1 Announce Type: new  Abstract: Railway engineers need simulation models that predict vehicle responses across operating scenarios that cannot be tested exhaustively. Agreement with representative measurements provides essential evidence, but calibration at a limited set of conditions does not guarantee accuracy elsewhere. We present a multifidelity railway-bogie response-correction method that treats multibody simulation histories as low-fidelity information and roller-rig measurements as high-fidelity evidence. This method combines an experiment-anchored fidelity assignment with physics-informed discrepancy learning for multichannel bogie-response histories. A time-delay neural network (TDNN) represents the condition-dependent simulation trend, and development-fitted amplitude alignment defines the low-fidelity baseline. A residual-correction network then models the reproducible response component not explained by this baseline and adds it to the baseline. An effecti
    
[^130]: 反转自触发控制：用于稀疏拒绝服务攻击的对抗性强化学习

    Inverting Self-Triggered Control: Adversarial Reinforcement Learning for Sparse Denial-of-Service Attacks

    [https://arxiv.org/abs/2609.12016](https://arxiv.org/abs/2609.12016)

    该论文的核心创新是将自触发控制范式进行反转，利用对抗性强化学习学习使闭环失稳的最稀疏拒绝服务攻击调度，并证明了迫使满足Lyapunov合约的自触发控制器崩溃所需最小干扰次数的下界，从而将DoS调度分析从周期和线性时不变系统扩展到自触发控制器。

    

    自触发强化学习控制（RL-STC）学习在运行时保证（RTA）覆盖机制下保持Lyapunov递减稳定性的最稀疏控制调度。我们对这一思路进行反转：一个对抗性强化学习智能体学习使闭环失稳的最稀疏干扰或拒绝服务（DoS）攻击调度，并采用镜像防御者安全证书的Lyapunov递增可容许性谓词。我们证明了对于满足Lyapunov合约的自触发控制器（STC），即时保持上一次值的介质访问控制攻击者迫使系统崩溃所需的最小干扰次数存在基于被控对象属性的下界，并作为推论恢复了先前基于次数预算的DoS调度中连续分组最优性的证书级类比。这将DoS调度的次数预算分析从周期系统和线性时不变系统扩展到了自触发控制器。在实验方面，我们针对每个被控对象的四个固定防御者进行训练（一个线性……）（摘要原文在此处截断）

    arXiv:2609.12016v1 Announce Type: new  Abstract: Self-triggered reinforcement learning control (RL-STC) learns the sparsest control schedule that preserves Lyapunov-decreasing stability under a Run-Time Assurance (RTA) override. We invert this: an adversarial RL agent learns the sparsest jamming or Denial-of-Service (DoS) schedule that destabilizes the closed loop, with a Lyapunov-increase admissibility predicate mirroring the defender's safety certificate. We prove a plant-property lower bound on the minimum jam count required for an immediate hold-last medium-access-control adversary to force a crash against a self-triggered controller (STC) satisfying a Lyapunov contract, and recover a certificate-level analog of the consecutive-grouping optimality of prior count-budget DoS scheduling as a corollary. This extends the DoS-scheduling count-budget analysis from periodic and linear-time-invariant to STC controllers. Empirically, we train against four fixed defenders per plant (one Linea
    
[^131]: 认证安全策展：面向安全离线强化学习的无分布保证

    Certified Safety Curation: Distribution-Free Guarantees for Safe Offline Reinforcement Learning

    [https://arxiv.org/abs/2609.12014](https://arxiv.org/abs/2609.12014)

    该论文提出“认证安全策展”框架，在仅有片段比较与回合预算查询的弱安全监督下，通过轨迹级筛选与“先学习后测试”校准，为安全离线强化学习的训练集构成提供无分布的安全保证。

    

    安全离线强化学习通常假设每个状态转移上都存在代价函数。我们探究当安全性只能通过比较短片段、并偶尔询问某一回合是否超出其预算来判断时，还能实现什么。认证安全策展给出了一个“先筛选后克隆”的流程：一个仅依赖状态的值函数通过片段比较训练得到，用于为整条轨迹评分；“先学习后测试”校准在无分布的 (α, δ) 界下认证一个选择阈值，该界约束了所选数据中不安全部分的比例；随后进行行为克隆。据我们所知，尚无先前工作为离线强化学习或模仿学习的训练集构成提供此类认证。先知对照实验验证了设计的合理性：即使拥有精确的值函数，对单个状态转移进行重加权仍然失败，因此该值函数被用于选择整条轨迹。所学策略在十五个 DSRL 任务中的十一个上满足代价预算，仅比克隆真实安全子集的结果少一个，而后者需要一个标签

    arXiv:2609.12014v1 Announce Type: new  Abstract: Safe offline reinforcement learning assumes a cost function on every transition. We ask what remains possible when safety can be judged only by comparing short clips and occasionally asking whether an episode exceeded its budget. Certified safety curation answers with a filter-then-clone pipeline: a state-only value trained from segment comparisons scores whole trajectories, Learn-then-Test calibration certifies a selection threshold under a distribution-free $(\alpha, \delta)$ bound on the unsafe fraction of the selection, and behavior cloning follows. We are not aware of prior work certifying the composition of a training set for offline RL or imitation. Oracle controls justify the design: reweighting individual transitions fails even with an exact value, so the value selects whole trajectories. The policies satisfy the cost budget on eleven of fifteen DSRL tasks, one short of cloning the ground-truth safe subset, which needs a label o
    
[^132]: QTrans：一种用于情感分类的量子Transformer

    QTrans: A Quantum Transformer for Sentiment Classification

    [https://arxiv.org/abs/2609.12011](https://arxiv.org/abs/2609.12011)

    QTrans利用参数化量子电路构建注意力机制中的查询、键、值特征，并结合量子前馈神经网络，建立了端到端可训练的量子-经典混合框架，在多个小规模情感分类数据集上取得了优于基线方法的准确率。

    

    在小规模二元情感分类场景中，否定、对比转折和跨词依赖等因素会导致情感线索的非线性耦合，使得传统轻量级模型难以充分捕捉词元之间的上下文关系。为解决这一问题，我们提出了一种名为QTrans的模型，该模型使用参数化量子电路构建查询、键和值特征，并通过量子测量之间的高斯距离推导注意力系数。通过进一步集成量子前馈神经网络、残差连接和层归一化，该模型建立了一个端到端可训练的量子-经典混合情感分类框架。在MR、CR和MPQA数据集上的实验结果表明，QTrans分别取得了72.13%、69.51%和63.45%的测试准确率，相比基线方法提升了2.88、3.17和3.79个百分点。

    arXiv:2609.12011v1 Announce Type: new  Abstract: In small-scale binary sentiment classification scenarios, factors such as negation, contrastive shifts, and cross-word dependencies lead to the non-linear coupling of sentiment cues, making it difficult for conventional lightweight models to fully capture the contextual relationships between tokens. To address this issue, we propose a model named QTrans, which uses parameterized quantum circuits to construct query, key, and value features and derives attention coefficients from Gaussian distances between quantum measurements. By further integrating a quantum feed-forward neural network, residual connections, and layer normalization, the model establishes an end-to-end trainable quantum-classical hybrid framework for sentiment classification. Experimental results on the MR, CR, and MPQA datasets show that QTrans achieves test accuracies of 72.13\%, 69.51\%, and 63.45\%, respectively, representing improvements of 2.88, 3.17, and 3.79 perce
    
[^133]: 从集体稳态中学习相互作用核

    Learning Interaction Kernels from Collective Steady States

    [https://arxiv.org/abs/2609.12004](https://arxiv.org/abs/2609.12004)

    该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。

    

    我们提出了一种从集体行为的单快照观测中对相互作用粒子系统进行系统辨识的学习方法，这与依赖轨迹观测的现有方法不同。这一设定导致了一个本质上不适定的逆问题，我们通过一种基于观测构型经验分布的正则化策略来解决该问题，这些构型来自不同的、未被观测到的初始条件。我们在多种具有稳态和准稳态模式的代表性模型上测试了该学习程序，在这些模型中，集体行为编码了关于相互作用机制的隐含信息。结果表明，我们的方法能够稳定且准确地恢复潜在的相互作用规律，从而忠实地重现集体行为，在许多情况下甚至能够重现导致该集体行为的动力学过程。

    arXiv:2609.12004v1 Announce Type: cross  Abstract: We propose a learning procedure for system identification in interacting particle systems from single-snapshot observations of collective behaviors, unlike existing approaches that rely on observations of trajectories. This setting leads to a fundamentally ill-posed inverse problem, which we solve by using a regularization strategy based on the empirical distribution of observed configurations, drawn from different, unobserved initial conditions. We test our learning procedure on a variety of representative models with steady-state and quasi-stationary patterns, where collective behaviors encode implicit information about the interaction mechanisms, demonstrating that our approach enables stable and accurate recovery of the underlying interaction laws, leading to faithful reproduction of the collective behavior, and in many cases even of the dynamics leading up to it.
    
[^134]: 我们能信任大语言模型评审吗：能力相关偏差研究及用于偏差校准的多评审集成方法

    Can We Trust LLM Judges: A Study of Capability-Dependent Biases and Multi-Judge Ensemble for Bias Calibration

    [https://arxiv.org/abs/2609.12002](https://arxiv.org/abs/2609.12002)

    研究发现LLM评审存在与被测模型能力相关的系统性偏差——能力越强的模型越容易获得宽松评判，并提出基于各评审假阳性率和假阴性率在线估计的校准加权多数投票集成方法来校准偏差。

    

    大语言模型（LLM）越来越多地被用作模型训练和评估的自动化评审，但单个评审会表现出损害可靠性的系统性偏差。以往的研究大多关注成对的“LLM作为评审”设置中的偏差；本文则聚焦于绝对评分任务，这更贴近现实中的使用场景。通过在四个基准测试和六个模型（共36对评审-被测模型组合）上的实验，我们发现模型的任务准确率能强烈预测其评审准确率（大多数模型上Pearson相关系数 r ≥ 0.90），并与方向性偏差呈负相关（r ≤ -0.83）；但仅凭准确率并不能保证评估的公平性：能力更强的被测模型始终从所有评审那里获得更宽松的评判（r ≥ 0.83）。为解决这一问题，我们提出校准加权多数投票（WMV）方法，这是一种集成评估方法，通过在线估计各评审的假阳性率和假阴性率来加权聚合多个LLM评审的判断。

    arXiv:2609.12002v1 Announce Type: new  Abstract: LLMs are increasingly used as automated judges for model training and evaluation, yet individual judges exhibit systematic biases that undermine reliability. Much of prior work has studied biases in pairwise LLM-as-a-judge settings; in this paper, we focus on absolute scoring tasks, which mirror more realistic use cases. Across four benchmarks and six models (36 judge-examinee pairs), we show that a model's task accuracy strongly predicts its judging accuracy (Pearson $r \geq 0.90$ on most models) and inversely predicts its directional bias ($r \leq -0.83$), but that accuracy alone does not ensure fair evaluation: more capable examinee models consistently receive more lenient judgments from all judges ($r \geq 0.83$). To address this, we propose calibrated weighted majority voting (WMV), an ensemble evaluation method that aggregates multiple LLM judges weighted by online estimates of their false-positive and false-negative rates. We intr
    
[^135]: 固定状态，长程延伸：恒定大小缓存在大规模场景下为块扩散模型带来什么

    Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale

    [https://arxiv.org/abs/2609.11998](https://arxiv.org/abs/2609.11998)

    该论文证明状态空间序列混合器配合块因果训练目标，可为块扩散语言模型实现精确的 O(1) 恒定大小缓存，使内存和每步延迟不随上下文长度增长，突破了注意力块缓存 O(L) 的限制。

    

    扩散语言模型可以并行解码 token，但其双向去噪器排除了快速自回归推理背后朴素的关键-值（KV）缓存机制。块扩散通过逐块解码恢复了缓存能力，但迄今为止部署在其上的块缓存都与注意力机制绑定：内存占用为 O(L)，且若作为免训练的改装方案使用，只能近似模型的真实计算。这两个限制均可被克服：将已完成解码的块总结为可复用状态的序列混合器支持块缓存，而相应的块因果训练目标使缓存变得精确。我们在大规模场景下研究这一方案：在单一目标函数下，于 300B token 上预训练三个 3B 块扩散去噪器（注意力、mamba 和混合架构），并通过单一缓存接口对三者进行解码。只有状态空间缓存是关于序列长度 O(1) 的：其内存占用和每步延迟在任何上下文长度下都保持恒定，而……

    arXiv:2609.11998v1 Announce Type: new  Abstract: Diffusion language models decode tokens in parallel, but their bidirectional denoiser rules out the naive key--value (KV) cache behind fast autoregressive inference. Block diffusion restores caching by decoding block-by-block, and the block caches deployed on it so far are tied to attention: O(L)in memory and, if used as training-free retrofits, only an approximation of the model's computation. Both constraints can be overcome: sequence mixers that summarize finalized blocks into a reusable state support block caching, and the corresponding block-causal training objective makes the cache exact. We study this recipe at scale, pretraining three 3B block-diffusion denoisers (attention, mamba, and hybrid) on 300B tokens under one single-frontier objective and decoding all three through a single cached interface. Only the state-space cache is O(1) in sequence length: its memory and per-step latency stay constant at any context length, while a
    
[^136]: DCRA：面向鲁棒时间序列学习的扩散条件化表示对齐

    DCRA: Diffusion-Conditioned Representation Alignment for Robust Time-Series Learning

    [https://arxiv.org/abs/2609.11997](https://arxiv.org/abs/2609.11997)

    该论文提出DCRA训练框架，将扩散前向过程重新用作结构化损坏调度器，并通过特征级一致性目标在噪声水平间对齐表示且保持类别判别结构，从而在噪声和分布偏移下学习鲁棒的时间序列表示，尤其适用于EEG和ECG等临床信号分析。

    

    arXiv:2609.11997v1 公告类型：新论文 摘要：在噪声和分布偏移条件下学习时间序列信号的鲁棒表示仍然具有挑战性，尤其是在脑电图（EEG）和心电图（ECG）分析等临床应用中。我们提出了扩散条件化表示对齐（DCRA），这是一种训练框架，它将前向扩散过程重新用作表示学习的结构化损坏调度器。与依赖独立采样扰动的传统数据增强和基于一致性的方法不同，DCRA通过扩散前向过程引入了结构化的损坏轨迹，从而实现了跨噪声水平的连续且可控的表示演化。我们引入了一个特征级一致性目标，在保持类别判别结构的同时跨噪声水平对齐表示。这一机制促进了保持结构的一致性，从而实现平滑且具有语义……

    arXiv:2609.11997v1 Announce Type: new  Abstract: Learning robust representations for time-series signals under noise and distribution shifts remains challenging, especially in clinical applications such as electroencephalogram (EEG) and electrocardiogram (ECG) analysis. We propose Diffusion-Conditioned Representation Alignment (DCRA), a training framework that repurposes the forward diffusion process as a structured corruption scheduler for representation learning. Different from conventional augmentation and consistency-based methods that rely on independently sampled perturbations, DCRA introduces a structured corruption trajectory via the diffusion forward process, which enables continuous and controlled representation evolution across noise levels. We introduce a feature-level consistency objective that aligns representations across noise levels while preserving class-discriminative structure. This mechanism promotes structure-preserving consistency, which enables smooth and semant
    
[^137]: 通过大语言模型引导的概念集成实现移动感知数据的可解释预测

    Explainable Prediction from Mobile Sensing Data through LLM-guided Concept Integration

    [https://arxiv.org/abs/2609.11995](https://arxiv.org/abs/2609.11995)

    本文提出了一种利用大语言模型生成概念异常监督信号的概念集成Transformer（CIT），无需人工标注即可对移动感知数据进行可解释的预测，并在精神健康相关数据集上取得了领先的F1分数。

    

    移动感知技术使日常环境中的行为和生理模式纵向监测成为可能。然而，在小样本健康感知研究中，特定任务的结果监督相对于异构感知数据较为有限，因此准确预测仍具有挑战性。可解释性同样重要，因为模型输出应反映有意义的行为和生理模式，而不仅仅是预测分数。我们开发了一种概念集成Transformer（CIT），通过大语言模型引导的概念监督，实现移动感知数据的可解释预测。CIT使用预训练大语言模型生成具备基线感知能力的概念异常目标及其置信度权重，无需人工概念标注。在两个纵向数据集上，CIT在AFFECT数据集上取得了最高的F1分数（0.756），并在PHQ-9数据集上并列最高（0.765）。学习到的概念分数还揭示了可解释的行为和生理模式。

    arXiv:2609.11995v1 Announce Type: new  Abstract: Mobile sensing enables longitudinal monitoring of behavioral and physiological patterns in everyday settings. However, accurate prediction remains challenging in small-cohort health-sensing studies, where task-specific outcome supervision is limited relative to heterogeneous sensing data. Interpretability is also important, as model outputs should reflect meaningful behavioral and physiological patterns rather than predictive scores alone. We develop a Concept-Integrated Transformer (CIT) with LLM-guided concept supervision for explainable prediction from mobile sensing data. CIT uses a pretrained large language model to generate baseline-aware concept abnormality targets with confidence weights without manual concept annotation. Across two longitudinal datasets, CIT achieves the highest F1 score on AFFECT (0.756) and ties for the highest on a PHQ-9 dataset (0.765). The learned concept scores also reveal interpretable behavioral and phys
    
[^138]: FINESSE：一个面向多模态金融事件序列的基于智能体的模拟器与基准数据集

    FINESSE: An Agent-Based Simulator and Benchmark Dataset for Multimodal Financial Event Sequences

    [https://arxiv.org/abs/2609.11993](https://arxiv.org/abs/2609.11993)

    本文提出了FINESSE——一个基于智能体的多模态金融事件序列模拟框架，能够生成由多个相互耦合的事件流（如交易、支付、账户状态变化、政策干预）组成的合成结构化数据集，并发布了FINESSE-Bench基准数据集，以弥补金融领域代表性开源数据集稀缺的问题。

    

    金融服务领域的机器学习研究受限于缺乏具有代表性的开源数据集。现有资源往往仅聚焦于单一模态或单一任务，未能反映金融服务中诸多问题所固有的结构化、多模态和动态特性。本文提出了FINESSE（金融事件序列模拟环境），这是一个基于智能体的模拟框架，用于生成由多个相互依赖的事件流组成的合成结构化数据集。每个事件流对应一种独特的金融行为，例如交易、支付、账户状态变化和政策干预，每种行为都具有各自独特的动作空间、模式和变量类型。这些事件流通过智能体的潜在演化状态相互耦合，从而能够模拟时间维度上丰富的交互。我们还发布了FINESSE-Bench，这是一个由该模拟器生成的基准数据集，支持四种代表性任务……

    arXiv:2609.11993v1 Announce Type: new  Abstract: Machine learning research in financial services is limited by the scarcity of representative open-source datasets. Existing resources are often narrowly focused on a single modality or task and fail to reflect the structured, multimodal, and dynamic nature inherent to many problems in financial services.   In this paper, we introduce FINESSE, a Financial Event Sequence Simulation Environment, an agent-based simulation framework for generating synthetic, structured datasets composed of multiple interdependent event streams. Each stream corresponds to a distinct financial behavior such as transactions, payments, account status changes, and policy interventions, each with unique action spaces, schemas and variable types. These streams are coupled through agents' latent evolving states, enabling the simulation of temporally rich interactions.   We also introduce FINESSE-Bench, a benchmark dataset generated by the simulator, supporting four r
    
[^139]: 通过基于模拟的推断方法研究多种无创生物信号对心血管生物标志物估计的影响

    Impact of Multiple Non-Invasive Biosignals on Cardiovascular Biomarker Estimation via Simulation-Based Inference

    [https://arxiv.org/abs/2609.11969](https://arxiv.org/abs/2609.11969)

    该研究采用基于模拟的推断方法，定量评估了多种无创生物信号（尤其是能反映心脏机械功能的BCG信号）对心血管生物标志物估计的增益作用。

    

    随着人口老龄化，心血管疾病患者的数量持续增加，这凸显了在疾病进展为严重且不可逆的功能衰退之前进行早期检测的必要性。因此，从光电容积脉搏波（PPG）和动脉压力波（APW）信号等无创生物信号中估计心血管生物标志物受到了越来越多的关注。这些信号可以通过可穿戴设备和袖带式设备进行测量。以往的研究已使用PPG和APW信号来估计心血管生物标志物。然而，这些信号在时域和频域均表现出很强的相似性，且主要反映外周和动脉脉搏波形。因此，它们可能仅能提供有关心脏机械功能的有限信息。相比之下，其他生物信号（如心冲击图（BCG））能够反映机体对心脏活动的微小机械反应，其定量影响……

    arXiv:2609.11969v1 Announce Type: cross  Abstract: As the population ages, the number of patients with cardiovascular diseases continues to increase, highlighting the need for early detection before progression to severe and irreversible functional decline. Consequently, estimating cardiovascular biomarkers from non-invasive biosignals, such as photoplethysmography (PPG) and arterial pressure wave (APW) signals, has attracted increasing attention. These signals can be measured using wearable and cuff-type devices. Previous studies have used PPG and APW signals to estimate cardiovascular biomarkers. However, these signals exhibit strong similarities in both the temporal and frequency domains and primarily reflect peripheral and arterial pulse waveforms. Therefore, they may provide limited information about cardiac mechanical function. In contrast, the quantitative impact of additional biosignals, such as ballistocardiography (BCG), which reflect the body's minute mechanical responses to
    
[^140]: 基于反射折叠与锥切换的二值仿射细化迭代的精确ReLU实现

    Exact ReLU realization of binary affine refinement iterates via reflection folding and cone switching

    [https://arxiv.org/abs/2609.11962](https://arxiv.org/abs/2609.11962)

    本文通过反射折叠与固定锥切换机制，证明了任意有限二值仿射细化迭代都能被固定宽度、深度与迭代次数线性相关的ReLU网络精确表示。

    

    我们研究具有有限支撑矩阵掩码的二值仿射细化算子，以及紧支撑的连续分段线性输入与强迫数据。我们证明每一个有限细化迭代都可以由固定宽度、深度与迭代次数成线性关系的ReLU网络精确实现，且无需将强迫剖面与二值单元格缝隙分离开来。其核心机制是通用反射倍增：将每个残差剖面与其反射配对，用一个固定块矩阵加上固定的交换对合取代两个二值转移矩阵。单元格缝隙恒等式使两个分支候选在帐篷折叠处保持一致，而它们的交换奇分量以到折叠点的距离线性有界。这使得可以通过固定的连续分段线性锥切换实现精确的分支选择，而无需乘以可变选择器。

    arXiv:2609.11962v1 Announce Type: cross  Abstract: We study vector-valued binary affine refinement operators with finitely supported matrix masks and compactly supported continuous piecewise linear input and forcing data. We prove that every finite refinement iterate admits an exact ReLU realization of fixed width and depth linear in the number of iterations. No separation of the forcing profile from the binary cell seams is required.   The main mechanism is universal reflection doubling. Pairing each residual profile with its reflection replaces the two binary transition matrices by one fixed block matrix together with a fixed swap involution. The cell-seam identity makes the two branch candidates agree at the tent fold, while their swap-odd component is bounded linearly by the distance to the fold. This permits exact branch selection by a fixed continuous piecewise linear cone switch, without multiplication by a variable selector.   The resulting primal recursion requires the residua
    
[^141]: 端侧语言模型用于隐私保护的压力预测：面向移动健康的多模态评估

    On-Device Language Models for Privacy-Preserving Stress Prediction: A Multimodal Evaluation on Mobile Health

    [https://arxiv.org/abs/2609.11961](https://arxiv.org/abs/2609.11961)

    本研究评估了端侧语言模型在移动设备上多模态压力预测的可行性，发现轻量级模型可实现低延迟的隐私保护推理，且客观传感器特征略优于主观自我报告，展现了该技术在移动心理健康应用中的潜力与限制。

    

    压力是心理健康的一个普遍决定因素，也是移动健康干预的关键目标。端侧语言模型（ODLMs）能够提供无需依赖云端的隐私保护推理，但其在移动设备资源受限条件下的健康预测可行性仍未被充分探索。我们使用零样本提示方法评估了端侧语言模型在多模态压力预测中的表现，同时测量了预测准确性、延迟和吞吐量。结果表明，客观传感器特征在平均上略优于主观自我报告，且轻量级的2B参数以下模型能够实现低延迟和可预测的资源使用。我们的发现既突出了端侧语言模型在移动心理健康领域的前景，也揭示了其实际应用中的限制。

    arXiv:2609.11961v1 Announce Type: new  Abstract: Stress is a pervasive determinant of mental health and a key target for mobile health interventions. On-device language models (ODLMs) offer privacy-preserving inference without cloud dependency, yet their feasibility for health prediction under mobile resource constraints remains underexplored. We evaluate ODLMs for multi-modal stress prediction using zero-shot prompting, measuring predictive accuracy alongside latency and throughput. Our results show that objective sensor features marginally outperform subjective self-reports on average, and that lightweight sub-2B models achieve low latency with predictable resource usage. Our findings highlight both the promise and the practical constraints of ODLMs for mobile mental health.
    
[^142]: 空间作为干预不变量：面向分层城市与具身智能的跨模态预测几何

    Space as an Interventional Invariant: Cross-Modal Predictive Geometry for Stratified Cities and Em-Spaced Intelligence

    [https://arxiv.org/abs/2609.11959](https://arxiv.org/abs/2609.11959)

    本文提出将空间定义为“干预不变量”，并构建跨模态预测几何框架，使不共享度量或表示的异构感知与城市数据，仍能在因果干预层面统一揭示共同的空间结构。

    

    空间是数学、物理学、空间认知、城市科学和具身智能中的一个基础概念，然而这些领域往往将空间结构视为共享的几何容器，或将其视为彼此割裂的表示集合。这类方法难以解释异构的感知过程与城市过程如何能够共同揭示出一种共同的空间结构，尤其当不同模态不共享相同的度量或表示时更是如此。本文通过将空间定义为一种“干预不变量”来填补这一空白：即在可容许动作下，能够保持局部兼容性以及未来观测条件规律的极小关系结构。我们发展了一种跨模态预测几何方法，它整合了局部状态空间、模态特定的观测映射、一个动作群胚（action groupoid）以及一个规范的预测状态商空间，并为识别干预性结构（而非仅仅是观测性结构）给出了明确的因果条件。

    arXiv:2609.11959v1 Announce Type: cross  Abstract: Space is a foundational concept across mathematics, physics, spatial cognition, urban science, and embodied intelligence, yet these fields often treat spatial structure either as a shared geometric container or as a collection of disconnected representations. Such approaches struggle to explain how heterogeneous sensory and urban processes can jointly reveal a common spatial structure, particularly when different modalities do not share the same metric or representation. This paper addresses this gap by defining space as an interventional invariant: the minimal relational structure that preserves local compatibility and the conditional laws of future observations under admissible actions. We develop a cross-modal predictive geometry that integrates local state spaces, modality-specific observation maps, an action groupoid, and a canonical predictive-state quotient, with explicit causal conditions for identifying interventional rather t
    
[^143]: 通过组分相互作用的计算建模解码混合物感知

    Decoding Mixture Perception through Computational Modeling of Component Interactions

    [https://arxiv.org/abs/2609.11958](https://arxiv.org/abs/2609.11958)

    提出了一种仿生深度学习框架，通过融合注意力加权的多受体神经响应曲线与浓度依赖的多分子曲线，实现了对多分子混合物气味的准确感知识别。

    

    嗅觉在人类进化和文明进程中扮演着不可或缺的角色。即使在当今科技高度发达的时代，嗅觉仍然是人们进行危险辨别、情感体验和记忆形成的关键渠道。然而，自然界中的大多数物质都以多分子混合物的形式存在。混合物成分的复杂性，以及浓度依赖的饱和效应和受体特异性的激活阈值，给识别嗅觉特征带来了巨大挑战。在本研究中，我们提出了一种新颖的仿生深度学习框架，用于对混合物进行准确的气味感知识别。我们稳健地构建了分子-受体相互作用的神经响应曲线，并开发了一种融合策略，将注意力加权的多受体曲线与浓度依赖的多分子曲线相结合，以复现竞争性激活和协同整合机制。

    arXiv:2609.11958v1 Announce Type: new  Abstract: Olfaction played an indispensable role throughout human evolution and civilization. Even in the contemporary era of advanced technology, olfaction remains a critical channel for person to conduct danger discrimination, emotional experience, and memory formation. However, most substances in nature exist as multi-molecule mixtures. The complexity of mixture compositions, as well as concentration dependent saturation effects and receptor specific activation thresholds, pose substantial challenges in identifying olfactory characteristics. In this study, we proposed a novel bio inspired deep learning framework for accurate odor perception recognition of mixtures. We robustly constructed neural response curves for molecule-receptor interactions, and developed a fusion strategy that integrates attention-weighted multi-receptor curves with concentration-dependent multi-molecule curves, replicating the competitive activation and synergistic integ
    
[^144]: 三思而后行：面向大语言模型智能体的动作前验证

    Look Before You Leap: Pre-Action Verification for LLM Agents

    [https://arxiv.org/abs/2609.11957](https://arxiv.org/abs/2609.11957)

    该论文提出在动作生效前进行低成本确定性验证的统一框架，通过预先构造动作的正确效果来直接度量并拦截LLM智能体的静默失败，其shell命令静态验证器以10.0%的误报率捕获了95.8%的无效命令。

    

    大语言模型智能体通过发出动作来作用于世界：运行shell命令、应用代码编辑。错误的动作并不总是以明显的方式失败；它可能静默地失败，产生一个看似合理但不正确的结果，且不引发任何错误。我们认为，在动作生效之前运行的低成本确定性检查是一种有效但未被充分利用的智能体监督形式，并在统一框架下研究了两种动作模态。其核心思想是在任何执行器运行之前，通过构造方式确定动作的正确效果，从而直接度量静默失败，并允许验证器选择弃权而非猜测。对于shell命令，一个覆盖9930条命令和482个工具的静态验证器以10.0%的误报率捕获了95.8%的无效命令。其语法和二进制检查具有oracle级别的精确性，在实现零误报的同时捕获了一半的所有错误；而标志检查的准确性仅受帮助文本覆盖范围限制，并构成了全部的误报来源。对于代码编辑……

    arXiv:2609.11957v1 Announce Type: new  Abstract: An LLM agent acts on the world by emitting actions: shell commands to run, edits to apply. A wrong action does not always fail loudly; it can fail silently, producing a plausible but incorrect effect that raises no error. We argue that a cheap deterministic check, run before an action takes effect, is an effective and underused form of agent oversight, and we study it across two action modalities in one framework. The idea is to fix an action's correct effect by construction, before any executor runs, so that silent failure is measured directly and the verifier may abstain rather than guess. For shell commands, a static verifier over 9930 commands and 482 tools catches 95.8% of invalid commands at a 10.0% false-positive rate. Its syntax and binary checks are oracle-exact, giving zero false positives while catching half of all errors; the flag check is bounded only by help-text coverage and accounts for every false positive. For code edit
    
[^145]: 性能、效率与崩溃——代码大语言模型离线后训练的优势与挑战

    Performance, Efficiency and Collapse -- Advantages and Challenges in Offline Post-training of Code LLMs

    [https://arxiv.org/abs/2609.11956](https://arxiv.org/abs/2609.11956)

    本研究证明代码大语言模型可以仅利用现有数据集进行完全离线的强化学习后训练，仅需几小时训练即可显著提升零样本代码生成性能，并适用于0.5B至7B参数规模的多种模型。

    

    基于强化学习（RL）的后训练是代码生成大语言模型（LLM）开发中的一个关键阶段，因为它确保了模型对指令的遵循以及生成功能正确的代码。这一过程通常需要从基于Transformer的大语言模型中进行计算密集型的代码样本生成，以及大量的GPU-CPU通信来进行序列验证。为了解决这些计算挑战，本工作研究了是否可以通过利用现有数据集而非生成新样本，完全离线地执行基于强化学习的后训练。研究结果表明，仅需几个小时的训练，就可以在不进行在线采样的情况下大幅提升大语言模型的零样本代码生成性能。此外，离线强化学习在0.5B到7B参数范围的多个模型上都带来了性能提升，尽管改进程度在不同模型家族之间有所差异。

    arXiv:2609.11956v1 Announce Type: new  Abstract: Post-training with reinforcement learning (RL) is a critical phase in the development of code-generating large language models (LLMs), as it ensures adherence to instructions and the production of functionally correct code. This process typically requires computationally intensive code sample generation from Transformer-based LLMs and substantial GPU-CPU communication for sequence verification. To address these computational challenges, this work examines whether RL-based post-training can be performed entirely offline by leveraging existing datasets rather than generating new samples. The findings indicate that, with only a few hours of training, zero-shot code generation performance of LLMs can be substantially improved without online sampling. Additionally, offline RL produces performance gains across models ranging from 0.5B to 7B parameters, although the extent of improvement varies among model families.
    
[^146]: R2VC：结合检索、验证与置信度校准的模块化事实核查

    R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration

    [https://arxiv.org/abs/2609.11955](https://arxiv.org/abs/2609.11955)

    R2VC提出了一种模块化的“检索-推理-验证-校准”事实核查架构，通过混合检索、DPO对齐的生成器、NLI交叉编码器验证以及置信度校准实现证据支撑、带引用和弃权机制的事实核查，在FEVER上使8B模型准确率较基线提升13.74%。

    

    大语言模型正日益被用于自动化事实核查，但端到端的提示方法往往将证据检索、推理和不确定性估计纠缠在一起，导致故障难以诊断、置信度难以令人信任。我们提出了R2VC，一种面向基于证据、带有引用和弃权机制的事实核查的模块化“检索-推理-验证-校准”架构。R2VC融合了以下组件：维基百科上的稀疏+稠密混合检索、经过监督微调与DPO对齐的生成器（可生成多样化的结构化判定候选）、用于基于证据进行候选选择的外部NLI交叉编码器，以及用于置信度估计与选择性弃权的轻量级序列级校准器。在FEVER数据集上，配备R2VC的8B骨干模型比基线准确率提高了13.74%。消融实验表明，基于验证器的候选选择和置信度校准是性能提升的最大贡献因素。移除候选……（摘要被截断）

    arXiv:2609.11955v1 Announce Type: new  Abstract: Large language models are increasingly used for automated fact checking, but end-to-end prompting often entangles evidence retrieval, reasoning, and uncertainty estimation, making failures difficult to diagnose and confidence difficult to trust. We present R2VC, a modular retrieve, reason, verify, calibrate architecture for evidence-grounded fact checking with citations and abstention. R2VC combines hybrid sparse+dense retrieval over Wikipedia, a supervised fine-tuned and DPO-aligned generator that produces diverse structured verdict candidates, an external NLI cross-encoder for evidence-based candidate selection, and a lightweight sequence-level calibrator for confidence estimation and selective abstention. On FEVER, an 8B backbone with R2VC achieves 13.74% higher accuracy than baseline. Ablation studies show that verifier-based candidate selection and confidence calibration are the largest contributors to performance. Removing candidat
    
[^147]: 使用量化分析工具实现高效的AI模型部署

    Efficient AI Model Deployment Using Quantization Analysis Tool

    [https://arxiv.org/abs/2609.11954](https://arxiv.org/abs/2609.11954)

    本文提出了一款基于ONNX框架的量化分析工具，通过逐层敏感性分析和分布可视化功能，帮助开发者在模型大小、延迟和精度之间做出明智权衡，从而实现AI模型在边缘设备上的高效部署。

    

    随着深度学习模型越来越多地被部署在资源受限的设备上，对高效模型优化技术的需求持续增长。在边缘设备和低功耗平台上有效部署AI模型，需要能够在降低模型大小和计算成本的同时保持高精度的优化方法。本文提出了量化分析工具，这是一个旨在简化量化工作流程、支持高性能模型部署的实用系统。该工具基于ONNX框架构建，以实现广泛的互操作性，提供详细的逐层敏感性分析、权重和激活分布的可视化，以及指导精度选择的洞察。通过识别对降低精度具有弹性或敏感的层，该工具使开发人员能够在模型大小、延迟和精度之间做出明智的权衡。针对多种神经网络架构的实验评估（摘要至此截断）...

    arXiv:2609.11954v1 Announce Type: new  Abstract: As deep learning models are increasingly deployed on resource constrained devices, the demand for efficient model optimization techniques continues to grow. Effective deployment of AI models on edge and low power platforms requires optimization methods that reduce model size and computational cost while maintaining high accuracy. This paper presents Quantization Analysis Tool, a practical system designed to streamline quantization workflows and support performance efficient model deployment. Built on the ONNX framework for broad interoperability, the tool provides detailed layer-wise sensitivity analysis, visualization of weight and activation distributions, and insights to guide precision selection. By identifying layers that are resilient or sensitive to reduced precision, the tool enables developers to make informed trade-offs between model size, latency, and accuracy. Experimental evaluations across multiple neural network architectu
    
[^148]: 自动驾驶中面向弱势道路使用者的场景无关关键性评估与预测

    Scenario-Independent Criticality Assessment and Prediction for Vulnerable Road Users in Autonomous Driving

    [https://arxiv.org/abs/2609.11947](https://arxiv.org/abs/2609.11947)

    提出了一种专为运动行为难以预测的弱势道路使用者（VRU）量身定制的新型关键性度量，并构建了一个适用于所有交通参与者类别的场景无关关键性预测框架，以克服现有度量依赖特定场景的局限性。

    

    提升安全性是自动驾驶车辆的首要目标。实现这一目标需要可靠的安全度量指标，这些指标需纳入物体类型、速度和关键性等与安全相关的因素。这类度量的一个关键能力是区分关键物体与非关键物体，这通过关键性或相关性估计来实现。现有的关键性度量通常针对特定场景设计，并且主要关注车辆与车辆之间的交互。因此，本文提出了一种专为弱势道路使用者（VRU）量身定制的新型关键性度量，由于其运动行为较难预测，弱势道路使用者需要特别加以考虑。此外，为了避免场景特定度量所带来的复杂性，我们引入了一种适用于所有交通参与者类别的场景无关关键性预测框架。所提出的以弱势道路使用者为中心的关键性度量……

    arXiv:2609.11947v1 Announce Type: cross  Abstract: Increasing safety is the primary objective of automated vehicles. Achieving this goal requires reliable safety metrics that incorporate safety-relevant factors such as object type, velocity, and criticality. A key capability of such metrics is the distinction between critical and non-critical objects, which is addressed through criticality or relevance estimation. Existing criticality metrics are typically designed for specific scenarios and primarily focus on vehicle-to-vehicle interactions. In this paper, we therefore propose a novel criticality metric tailored to vulnerable road users (VRUs), which require special consideration due to their less predictable motion behavior. Furthermore, to avoid the complexity introduced by scenario-specific metrics, we introduce a scenario-independent criticality prediction framework applicable to all traffic participant classes. The effectiveness of both the proposed VRU-centric criticality metric
    
[^149]: Hyperion：一个面向科学与人文研究的AI驱动HPC集群，利用机器学习预测作业周转时间

    Hyperion: An AI-powered HPC cluster for sciences and humanities research that utilizes ML for predicting job turnaround time

    [https://arxiv.org/abs/2609.11946](https://arxiv.org/abs/2609.11946)

    南卡罗来纳大学开发了面向科学与人文研究的AI驱动HPC集群Hyperion，并训练机器学习模型预测作业周转时间，将其无缝集成到Slurm作业提交系统中。

    

    Hyperion是南卡罗来纳大学（USC）为科学和人文学科研究人员开发的一个创新型高性能计算（HPC）集群。我们的方法是构建一个既能满足当前研究需求又能适应未来扩展的HPC集群。此外，我们开发并训练了两个机器学习（ML）模型来预测作业周转时间，包括等待时间和实际运行时间，并将其无缝集成到Slurm作业提交系统中。最后，我们展示了托管在Hyperion平台上的多种示例应用。

    arXiv:2609.11946v1 Announce Type: cross  Abstract: Hyperion is an innovative high-performance computing (HPC) cluster developed for researchers in both science and humanities disciplines at the University of South Carolina (USC). Our approach involved constructing a HPC cluster designed to meet the current research needs while accommodating future expansion. Additionally, we developed and trained two machine learning (ML) models to predict turnaround time, including wait time and wall time, and seamlessly integrated them into the Slurm job submission. Finally, we showcase a variety of sample applications hosted on the Hyperion platform.
    
[^150]: PinDCO：面向整页感知的大规模动态创意优化

    PinDCO: Whole-Page Aware Dynamic Creative Optimization at Scale

    [https://arxiv.org/abs/2609.11943](https://arxiv.org/abs/2609.11943)

    Pinterest提出的PinDCO生产级动态创意优化系统，核心创新在于创意组件融合网络（CCFN），通过为图像、标题、布局等各创意组件配置专用塔式网络并融合评分，在严格延迟与成本约束下实现大规模、整页感知的广告创意与受众高效匹配。

    

    生成式人工智能的最新进展极大地加速了高质量广告创意的制作，显著增加了每个广告活动中的候选变体数量。这一转变提升了业界对可扩展的动态创意优化（DCO）系统的需求，此类系统需在严格的延迟和成本约束下将创意匹配给最相关的受众。我们提出了PinDCO，一个部署于Pinterest（一个十亿级规模的视觉发现平台）的生产级DCO系统，用于广告创意的检索与选择。PinDCO围绕创意组件融合网络（CCFN）构建，该网络通过为每个创意组件（如图像、标题、布局）设置专用的塔式网络进行建模来执行动态创意评分，并使用组件专属的超参数以应对不同的建模复杂度。各组件的表示经过融合后，在广告级预测的条件下预测创意级评分，我们还通过一种探索机制来提升训练数据质量……（摘要在此处截断）

    arXiv:2609.11943v1 Announce Type: cross  Abstract: Recent advances in generative AI have substantially accelerated the creation of high-quality ad creatives, dramatically expanding the number of candidate variants per campaign. This shift increases the need for scalable dynamic creative optimization (DCO) systems that can match creatives to the most relevant audiences under stringent latency and cost constraints. We present PinDCO, a production DCO system for ad creative retrieval and selection on Pinterest, a billion-scale visual discovery platform. PinDCO is built around a Creative Component Fusion Network (CCFN) that performs dynamic creative scoring by modeling each creative component (e.g., image, title, layout) with a dedicated tower, using component-specific hyperparameters to account for differing modeling complexity. The component representations are fused to predict a creative-level score conditioned on the ad-level prediction, and we improve training data quality via an expl
    
[^151]: 边缘AI的电池代价：LLM推理对移动设备环境影响的研究

    The Battery Price of edge AI: A study of the Environmental Impact of LLM Inference on Mobile Devices

    [https://arxiv.org/abs/2609.11940](https://arxiv.org/abs/2609.11940)

    本研究系统性评估了18个大语言模型在智能手机上的推理能耗、性能与准确性，发现设备端推理的能效平均比服务器低3倍，揭示了本地优先AI范式对设备电池寿命和环境的影响。

    

    生成式人工智能的快速普及引发了隐私、延迟和性能方面的担忧，这促使人工智能向“本地优先”转变，即推理在用户设备上执行，而非在远程云服务器上执行。这一范式也给电池供电的智能手机带来了巨大的计算负载，可能缩短电池寿命并提高移动设备的整体更换率。本文对设备端大语言模型（LLM）推理的能耗、性能和准确性进行了系统性研究。我们在两部现代智能手机和一台服务器上，使用此类部署各自的最先进技术，评估了来自不同模型家族、不同规模和不同量化级别的18个模型。我们测量了每个生成token的能耗、token间延迟、模型准确性以及电池循环消耗。我们的结果表明：(i) 设备端推理平均能效低3倍...

    arXiv:2609.11940v1 Announce Type: cross  Abstract: The rapid diffusion of generative artificial intelligence raises privacy, latency, and performance concerns that motivate a shift toward "local-first" AI, where inferences are performed on the user's device instead of on remote cloud servers. This paradigm also places a significant computational load on battery-powered smartphones, potentially shortening battery life and increasing the overall replacement rate of mobile devices. This paper presents a systematic study of the energy consumption, performance, and accuracy of on-device large language model (LLM) inference. We evaluate 18 models from different model families, sizes, and quantization levels, on two modern smartphones and on a server, using the respective state-of-the-art for such deployments. We measure the energy per generated token, inter-token latency, model accuracy, and battery-cycle consumption. Our results show that (i) on-device inference is on average 3 times less e
    
[^152]: 面向鲁棒且公平临床联邦学习的拓扑帕累托控制Fed-Equilibrium框架

    Fed-Equilibrium Framework for Topological Pareto Control in Robust and Fair Clinical Federated Learning

    [https://arxiv.org/abs/2609.11937](https://arxiv.org/abs/2609.11937)

    提出Fed-Equilibrium框架，通过两阶段梯度控制级联（几何余弦相似度过滤与拓扑帕累托控制）解决临床联邦学习中大中心节点压制少数节点导致的“知识主导”问题，实现鲁棒性与公平性的拓扑均衡。

    

    联邦学习（FL）在多中心临床网络中的部署面临“知识主导”的挑战：高数据量的中心节点会自然地压倒少数社区节点，隐式地将较小队列的独特临床模式视为异常值。现有的几何防御方法仅提供了安全基线，但未能解决这一效率与公平的两难困境。为弥补这一差距，我们提出了Fed-Equilibrium，一个将范式从简单防御推进到拓扑均衡的框架。与传统聚合器不同，Fed-Equilibrium实现了顺序化的架构协同。它采用两阶段梯度控制级联：第一阶段（几何质量保证）通过余弦相似度漏斗强制方向一致性以过滤恶意噪声，构建稳定的流形；第二阶段（拓扑帕累托控制）随后通过识别最优帕累托拐点，主动调节通过验证的节点贡献。

    arXiv:2609.11937v1 Announce Type: new  Abstract: The deployment of Federated Learning (FL) in multi-center clinical networks faces the challenge of "knowledge dominance," where high-volume hubs naturally overwhelm minority community nodes, implicitly treating the distinct clinical patterns of smaller cohorts as outliers. Existing geometric defenses provide a security baseline but leave this efficiency-fairness dilemma unresolved. To bridge this gap, we propose Fed-Equilibrium, a framework that advances the paradigm from simple defense to topological equilibrium. Unlike traditional aggregators, Fed-Equilibrium implements a sequential architectural synergy. It utilizes a two-stage gradient control cascade: Stage I (geometric quality assurance) enforces directional consistency via a cosine similarity funnel to filter malicious noise, creating a stabilized manifold; Stage II (topological Pareto control) then actively modulates verified contributions by identifying the optimal Pareto knee p
    
[^153]: 提升能量受限本地推理与训练性能的一个简单技巧

    One Simple Trick for Improving the Performance of Energy-Limited Local Inference and Training

    [https://arxiv.org/abs/2609.11936](https://arxiv.org/abs/2609.11936)

    通过将工作负载切分为更小的块，以更高频率交替进行计算与内存操作，可以平滑功率和温度峰值、避免GPU降频，从而加快运行速度并降低总能耗。

    

    能源供应与散热是现代GPU部署面临的两大主要挑战。尽管这一问题通常在新建数据中心的背景下被讨论，但同样的限制也适用于小型消费级设备，例如DGX Spark。在以计算密集型任务（如矩阵乘法）与内存受限操作（如归一化或交叉熵）交替进行为特征的工作负载中，计算密集部分可能会触及功率和/或温度限制并开始降频。在这篇短文中，我们展示了将工作负载切分为更小的部分，以更高的频率交替进行计算与内存操作，可以平滑这些功率和温度峰值，从而避免降频，显著缩短实际运行时间并降低总能耗。我们提出了在DGX Spark上可以利用这一效应的几种场景，实现了最高达2%的性能与能耗改进，并证明……

    arXiv:2609.11936v1 Announce Type: cross  Abstract: Energy supply and heat dissipation are two of the main challenges with modern GPU deployments. While typically discussed in the context of new datacenter constructions, the same constraints also apply to small form-factor consumer devices, such as the DGX spark. In workloads characterized by alternating compute-intensive tasks such as matmuls with memory-bound operations such as norms or cross-entropy, the compute-intensive parts might hit power and/or thermal limits and start throttling. In this short paper, we show that chunking the workload into smaller parts that alternate compute and memory in higher frequencies, these power and temperature spikes can be smoothed out, preventing throttling and resulting in considerably faster wall-clock time and reduced total energy consumption. We present several scenarios in which this effect can be exploited on a DGX Spark with up to 2% performance and energy improvements, and demonstrate that 
    
[^154]: 物理信息一致性预测：将PDE一致性嵌入神经算子的无分布不确定性量化

    Physics-Informed Conformal Prediction: Embedding PDE Consistency into Distribution-Free Uncertainty Quantification for Neural Operators

    [https://arxiv.org/abs/2609.11935](https://arxiv.org/abs/2609.11935)

    提出PI-CP框架，将PDE残差嵌入一致性预测的非一致性评分中，实现无分布、空间自适应且具有可证明覆盖保证的不确定性量化，并揭示FNO的平移等变性对Dirichlet边界条件PDE构成根本近似障碍，而坐标通道可克服此障碍并使误差降低高达63倍。

    

    诸如傅里叶神经算子（FNO）等神经算子在逼近偏微分方程（PDE）解方面取得了显著的精度。然而，提供严格的不确定性估计仍然是一个悬而未决的挑战。我们提出了物理信息一致性预测（PI-CP），这是一个将PDE残差嵌入到分裂一致性预测（split conformal prediction）的非一致性评分中的框架，所产生的预测区间具有以下特性：（i）无分布性，具有可证明的覆盖保证；（ii）当PDE残差与预测误差相关时具有空间自适应性——在物理规律满足良好的区域区间更窄，在物理规律被违反的区域区间更宽。此外，我们证明了FNO的平移等变性为具有Dirichlet边界条件的PDE构成了一个根本性的近似障碍，并证明坐标通道可以解决这一问题，使误差降低高达63倍。我们在六个物理场景中验证了PI-CP——热传导（2D/3D）、结构（原文摘要在此处截断）。

    arXiv:2609.11935v1 Announce Type: new  Abstract: Neural operators such as the Fourier Neural Operator (FNO) achieve remarkable accuracy in approximating solutions to partial differential equations (PDEs). However, providing rigorous uncertainty estimates remains an open challenge. We propose Physics-Informed Conformal Prediction (PI-CP), a framework that embeds PDE residuals into the nonconformity score of split conformal prediction, producing prediction intervals that are (i) distribution-free with provable coverage guarantees, and (ii) spatially adaptive when the PDE residual correlates with prediction error -- tighter where physics is well-satisfied, wider where it is violated. Additionally, we prove that FNO's translation equivariance creates a fundamental approximation barrier for PDEs with Dirichlet boundary conditions, and show that coordinate channels resolve this with up to 63x error reduction. We validate PI-CP across six physics scenarios -- heat conduction (2D/3D), structur
    
[^155]: 网络化系统中基于扰动时间序列进行物理信息结构推断的基本动力学单元

    Fundamental Dynamical Units for Physics-Informed Structural Inference from Perturbation Time-Series in Networked Systems

    [https://arxiv.org/abs/2609.11934](https://arxiv.org/abs/2609.11934)

    该论文提出“基本动力学单元”（FDUs）——以带符号的三节点相互作用模式作为可组合基元，将网络系统的相互作用假设空间转化为有限、构造性且易处理的表示，从而为从扰动时间序列数据中恢复带符号相互作用结构提供了结构性的解决方案。

    

    arXiv:2609.11934v1 公告类型：新论文 摘要：在网络化动力学系统中，机制研究最关注的参数是带符号的相互作用结构。从扰动时间序列数据中恢复这一结构是一个基本的辨识问题，并受到三个相互耦合的障碍困扰：相互作用架构的组合复杂性、有限干预下因果归因的模糊性，以及干扰结构推断的状态依赖动力学。这些障碍在本质上都是结构性的，因此需要结构性的解决方案。我们通过采用还原论方法来应对这些挑战，引入了基本动力学单元（FDUs）：带符号的三节点相互作用模式作为可组合的基元，将相互作用假设空间转换为有限、构造性且易于处理的表示。我们证明，局部相互作用结构决定了区分直接影响与中继影响所需的扰动条件，使干预设……（原文摘要在此处截断）

    arXiv:2609.11934v1 Announce Type: new  Abstract: In networked dynamical systems, the parameter of primary mechanistic interest is signed interaction structure. Recovering this structure from perturbation time-series data is a fundamental identification problem, compounded by three coupled obstacles: the combinatorial complexity of interaction architectures, ambiguity of causal attribution under limited interventions, and state-dependent dynamics that confound structural inference. Each obstacle is structural in origin and calls for a structural solution. We address these challenges by adopting a reductionist approach, introducing Fundamental Dynamical Units (FDUs): signed three-node interaction patterns as composable primitives that convert the interaction hypothesis space into a finite, constructive, and tractable representation. We show that local interaction structure determines the perturbation conditions required to disentangle direct from relayed influence, making intervention de
    
[^156]: 迈向可持续氢能系统：基于模型预测控制与强化学习的供应链优化

    Towards Sustainable Hydrogen Systems: Supply Chain Optimization with Model Predictive Control and Reinforcement Learning

    [https://arxiv.org/abs/2609.11933](https://arxiv.org/abs/2609.11933)

    本文针对可再生能源驱动的氢能供应链，比较了基于规则控制、模型预测控制、无预测强化学习和预测增强强化学习四种控制方法，以在动态不确定条件下实现经济性、可靠性与可持续性的平衡。

    

    氢能供应链有望通过促进可再生能源并网、长时储能以及工业和交通部门的脱碳，在未来低碳能源系统中发挥核心作用。然而，其运行面临着可再生能源发电波动性、电价波动、不确定的氢气需求，以及与电解槽、储能和电网交互相关的工程约束等挑战。随着氢能基础设施向商业化部署迈进，运行策略必须在动态和不确定条件下平衡经济性能、可靠性与可持续性。本文研究并比较了四种用于可再生能源驱动氢能供应链的控制方法：基于规则的控制器（RBC）、模型预测控制（MPC）、无预测信息的强化学习（RL-NF），以及带有预测增强观测的强化学习（RL-F）。

    arXiv:2609.11933v1 Announce Type: cross  Abstract: Hydrogen supply chains are expected to play a central role in future low-carbon energy systems by enabling renewable energy integration, long-duration storage, and decarbonization of industrial and transportation sectors. However, their operation is challenged by renewable generation variability, electricity price fluctuations, uncertain hydrogen demand, and engineering constraints associated with electrolyzers, energy storage, and grid interaction. As hydrogen infrastructure expands toward commercial deployment, operational strategies must balance economic performance, reliability, and sustainability under dynamic and uncertain conditions.   This paper investigates and compares four control approaches for a renewable-powered hydrogen supply chain: a rule-based controller (RBC), model predictive control (MPC), reinforcement learning without forecasts (RL-NF), and reinforcement learning with forecast-augmented observations (RL-F). All m
    
[^157]: Musec：用于稳定Muon类训练的动量谱裁剪

    Musec: MomentUm SpEctral Clipping for Stable Muon-type Training

    [https://arxiv.org/abs/2609.11655](https://arxiv.org/abs/2609.11655)

    本文提出Musec方法，通过将Muon优化器的谱平坦化替换为谱裁剪——仅裁剪超过阈值的奇异值并保留动量的谱结构——实现了优化器级别、与架构无关的训练稳定化，有效避免损失尖峰和权重无界增长问题。

    

    Muon已成为大语言模型训练中一种非常高效的优化器，相比广泛采用的Adam和AdamW优化器，通常能实现更优的收敛性和性能。然而，由于其谱平坦化操作，Muon容易出现训练不稳定的问题，表现为损失尖峰和模型权重的无界增长。现有方法主要依赖于权重裁剪或注意力logit裁剪，这需要对架构进行特定修改，且无法直接解决所有模型组件的不稳定性问题。我们提出动量谱裁剪，用谱裁剪替代Muon的谱平坦化：Musec不是将动量矩阵的所有奇异值设为近似于一，而是只裁剪超过阈值的奇异值，同时保留动量矩阵的底层谱结构。我们的策略提供了一种优化器级别的、与架构无关的Muon稳定化机制。

    arXiv:2609.11655v1 Announce Type: new  Abstract: Muon has emerged as a highly effective optimizer for large language model training, often achieving superior convergence and performance compared with the widely adopted Adam and AdamW optimizers. Nevertheless, Muon is prone to training instability due to its spectral flattening, manifested by loss spikes and unbounded growth of model weights. Existing approaches primarily rely on weight or attention-logit clipping, which require architecture-specific modifications and do not directly address instability across all model components. We propose MomentUm SpEctral Clipping (Musec), which replaces Muon's spectral flattening with spectral clipping: rather than setting all singular values of the momentum matrix to approximately one, Musec clips singular values that exceed a threshold while preserving the underlying spectral structure of the momentum. Our strategy provides an optimizer-level, architecture-agnostic mechanism for stabilizing Muon
    
[^158]: 超越求解器判定：面向自动形式化的生成式奖励模型

    Beyond Solver Verdicts: Generative Reward Models for Autoformalization

    [https://arxiv.org/abs/2609.11085](https://arxiv.org/abs/2609.11085)

    本文发现了自动形式化中的“判定保持的不忠实性”（VPU）失败模式，从理论上证明仅依赖求解器判定的验证方法无法有效检测此类错误，并提出生成式验证方法（GenV），将Z3等价性预言机蒸馏为无参照的连续等价性评分以实现可靠的验证。

    

    神经符号系统依赖数学求解器来保证推理的正确性，然而求解器从根本上无法感知一个形式化翻译是否与指定的形式化保持严格的参照等价性。我们将这一脆弱性形式化为“判定保持的不忠实性”（Verdict-Preserving-Unfaithfulness, VPU）：这是一种失败模式，即错误的编码能够成功执行并匹配预期的判定结果。我们从理论上证明，结构化的、仅基于判定的验证启发式方法在检测这些具有欺骗性的有效轨迹时，其检测能力在数学上被限制在随机概率水平。为解决这一问题，我们引入了生成式验证（GenV），通过重新利用语言模型的原生词汇空间，将离线的Z3等价性预言机蒸馏为无参照的、连续的参照等价性评分。通过决策投影logit透镜和稀疏自编码器进行的机制分析表明，这种生成式读出能够原生地提取精确的空间误差信息……

    arXiv:2609.11085v1 Announce Type: cross  Abstract: Neurosymbolic systems rely on mathematical solvers to guarantee reasoning correctness, yet solvers are fundamentally blind to whether a formal translation maintains strict reference-equivalence to a designated formalization. We formalize this vulnerability as Verdict-Preserving-Unfaithfulness (VPU): a failure mode where an incorrect encoding executes successfully and matches the expected verdict. We theoretically prove that structural, verdict-only verification heuristics are mathematically bounded to chance-level detection on these deceptively valid traces. To resolve this, we introduce Generative Verification (GenV), which distills an offline Z3-equivalence oracle into a reference-free, continuous reference-equivalence score by repurposing the language model's native vocabulary space. Mechanistic analysis via decision-projected logit lenses and sparse autoencoders shows this generative readout natively extracts precise spatial error 
    
[^159]: 我们真的在做少样本学习吗？对预训练假设的批判性审视

    Are We Really Doing Few-Shot Learning? A Critical Examination of Pre-Training Assumptions

    [https://arxiv.org/abs/2609.10851](https://arxiv.org/abs/2609.10851)

    本文通过系统比较四种预训练协议，揭示了当前少样本学习评估中因同域预训练带来的 9.66 个百分点乐观偏差，质疑现有评估方式能否真正反映模型的低数据学习能力。

    

    少样本学习通常在一种协议下进行评估，即先在一个大型辅助数据集上对模型进行预训练，该数据集的类别与目标任务 episodes 的类别不相交，但来自相同的视觉域。本文探讨了这种协议是否真正反映了低数据学习情境。我们在八个数据集、三种少样本学习架构以及多种 way-shot 设置下，系统地比较了无预训练、类别不相交的同域预训练、有监督的域外预训练以及无标签的域外预训练。我们的结果表明，仅靠类别不相交并不足以消除目标域数据的影响。同域预训练相比无预训练平均提升 33.41 个百分点，而有监督的域外预训练仅带来 23.75 个百分点的提升，从而揭示了与域重叠相关的 9.66 个百分点的乐观偏差。尽管在目标域数据可用性受限的应用场景中，域外预训练更符合实际情况……

    arXiv:2609.10851v1 Announce Type: cross  Abstract: Few-shot learning is commonly evaluated under protocols that pre-train a model on a large auxiliary set whose classes are disjoint from the target episodes yet drawn from the same visual domain. This paper examines whether such protocols truly reflect low-data learning. We systematically compare no pre-training, class-disjoint in-domain pre-training, supervised out-of-domain pre-training, and label-free out-of-domain pre-training across eight datasets, three few-shot architectures, and multiple way-shot settings. Our results show that class disjointness alone is insufficient to remove the influence of target-domain data. In-domain pre-training improves over no pre-training by 33.41 percentage points on average, whereas supervised out-of-domain pre-training yields 23.75 percentage points, revealing a 9.66-point optimistic bias associated with domain overlap. Although out-of-domain pre-training is more realistic in applications where tar
    
[^160]: HuRo：将人类视频机器人化以实现可扩展的VLA预训练

    HuRo: Robotizing Human Videos for Scalable VLA Pretraining

    [https://arxiv.org/abs/2609.10706](https://arxiv.org/abs/2609.10706)

    该论文开发了一套将异构人类视频转换为机器人对齐的观察与动作轨迹的机器人化流水线，并据此构建了包含约63万条机器人化片段和1.42亿帧的HuRo数据集，验证了机器人化人类视频能够为VLA策略预训练提供有效且可扩展的监督信号。

    

    人类视频数据集已成为昂贵真实机器人数据的一种极具吸引力的替代方案，能够以规模化方式提供丰富的多样性。为了弥合人类与机器人身体形态之间的差异，现有方法要么在与任务匹配的设置下对视频进行机器人化处理，要么在大规模场景下分别处理观察和动作的对齐。在这项工作中，我们系统地研究了机器人化的人类视频能否为视觉-语言-动作（VLA）策略的预训练提供有效且可扩展的监督信号。为此，我们开发了一套机器人化流水线，能够将异构的人类视频转换为与机器人对齐的观察和动作轨迹，同时在不同标注层级上推断缺失的中间信号。利用该流水线，我们构建了HuRo数据集，其中包含来自五个人类视频来源的约63万条机器人化片段和1.42亿处理后的帧。在四项真实世界操作任务上，扩大机器人化预训练规模能够提升整体完成效果

    arXiv:2609.10706v1 Announce Type: cross  Abstract: Human video datasets have emerged as a compelling alternative to expensive real-robot data, offering rich diversity at scale. To bridge the human-to-robot embodiment gap, existing approaches either robotize videos in task-matched settings or address observation and action alignment separately at scale. In this work, we systematically examine whether robotized human videos can provide effective and scalable supervision for pretraining vision-language-action (VLA) policies. To this end, we develop a robotization pipeline that converts heterogeneous human videos into robot-aligned observations and action trajectories while inferring missing intermediate signals across annotation levels. Using this pipeline, we construct the HuRo dataset, comprising about 630K robotized episodes and 142M processed frames from five human-video sources. Across four real-world manipulation tasks, increasing robotized pretraining scale improves overall complet
    
[^161]: 基于轮换协调者的拜占庭鲁棒联邦火灾检测

    Byzantine-Robust Federated Fire Detection with a Rotating Coordinator

    [https://arxiv.org/abs/2609.10647](https://arxiv.org/abs/2609.10647)

    本文提出了一种结合历史感知聚合与轮换协调者的拜占庭鲁棒半去中心化联邦学习方法，用于室内火灾检测，通过构建整理的火灾数据集、实现边缘端模型更新高达10倍的压缩，同时解决了上行带宽受限、恶意客户端以及固定服务器单点故障三大实际问题。

    

    我们研究了联邦学习（FL）在室内火灾检测中的应用。此类火灾检测系统使用边缘摄像头记录敏感视频，这些视频难以在中央服务器上集中收集。现有的联邦解决方案遗留了三个未解决的实际障碍：有限的上行带宽、拜占庭（恶意或故障）客户端，以及对单一永久固定聚合服务器的无条件信任。我们的主要贡献同时解决了这三个问题。具体而言，我们提供了：(i) 一个从八个公开来源精心整理的室内火灾检测数据集；(ii) 一个可部署在边缘设备上的检测器，其模型更新可被压缩至多10倍，而平衡准确率仅损失很小；(iii) 一种半去中心化的拜占庭鲁棒联邦学习方法，该方法将具有历史感知的聚合与轮换协调者相结合，能够清除逐轮过滤机制难以发现的隐蔽攻击，同时消除了固定服务器带来的单点故障。

    arXiv:2609.10647v1 Announce Type: new  Abstract: We study the application of federated learning (FL) to indoor fire detection. Such fire-detection systems use edge cameras that record sensitive footage which cannot easily be collected at a central server. Existing federated solutions leave three practical obstacles unaddressed: limited uplink bandwidth, Byzantine (malicious or faulty) clients, and unconditional trust in a single, permanently fixed aggregation server. Our main contributions address all three. In particular, we provide (i) a curated indoor fire-detection dataset assembled from eight public sources; (ii) an edge-deployable detector whose model updates are compressed up to 10 time with only a small loss in balanced accuracy; and (iii) a semi-decentralized Byzantine-robust FL method that combines history-aware aggregation with a rotating coordinator, evicting stealthy attacks that per-round filters miss while removing the fixed-server single point of failure. On the held-ou
    
[^162]: 从代码到执行框架：通过自我博弈蒸馏黑盒优化器

    Code-to-Harness: Distilling Black-Box Optimizers from Self-Play

    [https://arxiv.org/abs/2609.09468](https://arxiv.org/abs/2609.09468)

    本文提出“从代码到执行框架”方法：让语言模型代理通过反复编写和评估优化器程序进行自我练习，将学到的搜索策略一次性蒸馏成一份冻结的197词文本框架，该框架能显著降低多个语言模型执行器在低预算黑盒优化中的遗憾值，并可跨模型迁移。

    

    代理能否通过可执行的实践学会数值搜索策略，再将该策略以文本形式迁移出去？我们研究低预算黑盒优化问题，在这一领域中，无辅助的语言模型的表现仍远低于强大的经典优化器。在开发阶段，代理反复编写并评估优化器程序，随后将由此产生的程序与实践记录一次性蒸馏为一份197词的核心执行框架Harness A，并在评估前将其冻结。在独立开展的N=30研究中，Harness A将Gemini Flash的遗憾值降低了48%（p<.001），在练习函数族上达到GP-BO的性能区间，并在全部三个留出的BBOB测试地貌上降低了平均遗憾值。同一份文本提升了所有受测的Gemini执行器的表现，并可迁移至Claude Sonnet，分别将遗憾值降低43%和49%（p≤.005）。一次独立的端到端复现生成了Harness B——一个处于相同性能层级的不同程序与不同文本。同一框架还达到了……

    arXiv:2609.09468v2 Announce Type: replace  Abstract: Can an agent learn a numerical search strategy through executable practice and then transfer that strategy as text? We study low-budget black-box optimization, where unaided language models remain well below strong classical optimizers. During development, an agent repeatedly writes and evaluates optimizer programs. It then distills the resulting program and practice record once into a 197-word primary Harness A, which is frozen before evaluation. Harness A reduces Gemini Flash regret by 48\% in an independent $N=30$ study ($p<.001$), enters the GP-BO performance range on the practice family, and lowers mean regret on all three held-out BBOB landscapes. The same text improves every tested Gemini executor and transfers to Claude Sonnet, reducing regret by 43\% and 49\% ($p\leq.005$). An independent end-to-end replication produces Harness B, a different program and text at the same performance tier. The same framework also attains the 
    
[^163]: SAEScientist-Bench：AI智能体能否开展自主的SAE可解释性研究？

    SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?

    [https://arxiv.org/abs/2609.09113](https://arxiv.org/abs/2609.09113)

    本文提出SAEScientist-Bench基准，评估AI智能体能否作为科学家自主运用稀疏自编码器（SAE）工具，进行机制可解释性的自主科学发现研究。

    

    虽然关于递归自我改进（RSI）的研究主要集中于自动化模型训练流程，但可靠的自主发展还需要一个缺失的关键支柱：事后监控与审计，以理解模型学到了什么并确保安全对齐。机制可解释性工具对于弥合这一差距至关重要，其中稀疏自编码器（SAE）通过隔离可解释特征用于模型检查与引导，成为这一领域的基石。在本文中，我们引入SAEScientist-Bench，用于评估AI智能体能否作为科学家，利用SAE工具进行自主的机制发现。给定一个目标概念，智能体需设计对比探针，并在Gemma-2-9B-IT模型的Gemma Scope词典（包含131K+特征）中导航以发现最优特征，随后基于锚定在Neuronpedia上的精选专家参考特征进行评估，评估维度包括激活排名、对比文本上的概念选择性以及因果引导效果。

    arXiv:2609.09113v1 Announce Type: new  Abstract: While research on recursive self-improvement (RSI) has predominantly automated model training pipelines, reliable autonomous development demands a missing pillar: post-hoc monitoring and auditing to understand what models learn and ensure safe alignment. Mechanistic interpretability tools are essential to bridge this gap, among which Sparse Autoencoders (SAEs) serve as a cornerstone by isolating interpretable features for model inspection and steering. In this paper, we introduce SAEScientist-Bench to evaluate whether AI agents can act as scientists utilizing SAE tools for autonomous mechanistic discovery. Given a target concept, an agent designs contrastive probes and navigates a Gemma Scope dictionary of 131K+ features in Gemma-2-9B-IT to discover the optimal feature, evaluated against curated expert reference features anchored on Neuronpedia across activation rank, concept selectivity on contrastive texts, and causal steering. Across 
    
[^164]: 作为上下文采样器的Transformer：从闭式扩散到无需估计的采样

    Transformers as In-Context Samplers: From Closed-Form Diffusion to Estimation-Free Sampling

    [https://arxiv.org/abs/2609.08981](https://arxiv.org/abs/2609.08981)

    本文证明冻结的Transformer可以通过上下文学习模拟迭代式生成采样器（如闭式扩散采样器），其中softmax注意力计算责任权重与加权经验均值、前馈层实现欧拉更新，从而将上下文学习能力从监督学习拓展到数据生成任务。

    

    越来越多的研究证实，大型语言模型并非仅仅是统计记忆器，而是具备上下文学习能力：在测试时仅使用提示中提供的样本进行推理，无需任何参数更新。先前的理论工作已表明，这种能力可以扩展到线性回归等监督学习任务。我们证明上下文学习可以进一步扩展到数据生成领域：冻结的Transformer可以从上下文样本中模拟迭代式生成采样器。我们首先表明，Transformer可以实现闭式和平滑闭式扩散采样器。该构造为softmax注意力机制确定了一个具体的生成性角色：它负责计算责任权重和加权经验均值，而前馈层则实现欧拉更新。为了在实证层面将这些构造与预训练语言模型建立联系，我们研究了语义主题采样：通过提示……

    arXiv:2609.08981v1 Announce Type: cross  Abstract: A growing body of work establishes that large language models are not mere statistical memorizers, but are capable of in-context learning: performing inference at test time using only examples provided in the prompt, without any parameter updates. Prior theoretical work has shown that this capability extends to supervised learning tasks such as linear regression. We prove that in-context learning extends further to \emph{data generation}: frozen transformers can simulate iterative generative samplers from in-context samples. We first show that transformers can realize closed-form and smoothed closed-form diffusion samplers. The construction identifies a concrete generative role for softmax attention: it computes responsibility weights and weighted empirical averages, while feedforward layers implement Euler updates.   To empirically relate these constructions to pretrained language models, we study \emph{semantic-topic sampling}: promp
    
[^165]: BatchNorm幻觉：诊断机器遗忘评估中的归一化伪影

    The BatchNorm Illusion: Diagnosing Normalization Artifacts in Machine Unlearning Evaluation

    [https://arxiv.org/abs/2609.08901](https://arxiv.org/abs/2609.08901)

    该论文揭示了一个评估陷阱：在基于BatchNorm的架构上，仅对保留数据做一次前向传播（不修改任何权重）就能重写模型的归一化状态并逆转表面遗忘指标，作者将其形式化为权重保持的定点算子，严格证明此类前后差异源于BN运行统计量而非遗忘方法本身，从而清晰区分了度量失效与真实的权重编码信息残留。

    

    近似机器遗忘旨在无需从头重新训练的情况下，从已训练模型中移除特定训练数据的影响。我们发现了一个此前未被记录的混淆因素，它存在于基于BatchNorm架构的遗忘评估方式中：对保留数据进行单次前向传播——这一操作不修改任何权重——却能够确定性地重写模型的归一化状态，并逆转表观上的表面指标遗忘。我们将该操作形式化为一个保持权重不变的定点算子，并证明由它引发的任何前后差异都可以严格归因于BN运行统计量，而非遗忘方法对权重所做的任何修改。这一归因论断清晰地区分了度量失效（BN伪影）与编码器失效（残留在权重中的编码信息，近期有并行工作对此进行了记录），并且同一算子框架还给出了线性探针性能提升的唯一分解。

    arXiv:2609.08901v1 Announce Type: new  Abstract: Approximate machine unlearning aims to remove the influence of specific training data from a trained model without retraining from scratch. We identify a previously undocumented confound in how unlearning is evaluated on BatchNorm-based architectures: a single forward pass over retain data, an operation that modifies no weight, can deterministically rewrite the model's normalization state and reverse the apparent surface-metric forgetting. We formalize this operation as a weight-preserving fixed-point operator and prove that any pre-versus-post gap it induces is provably attributable to BN running statistics rather than to any modification the unlearning method made to the weights. This attribution claim cleanly separates measurement failure (BN artifact) from encoder failure (residual weight-encoded information, recently documented in concurrent work), and the same operator framework yields a unique decomposition of linear-probe elevati
    
[^166]: 有代价的规范：为何基于强化学习的对齐至多只能承诺有条件的合规

    Norms at a Price: Why RL-Based Alignment Can Promise Conditional Compliance at Best

    [https://arxiv.org/abs/2609.07627](https://arxiv.org/abs/2609.07627)

    论文论证了基于强化学习的对齐因从打分行为中学习规范，使“合规”退化为“被发现才付出代价”，因此行为训练在原理上至多只能保证智能体在被观察时才合规的有条件合规。

    

    AI 智能体有时在推断自己正被测试时表现得对齐，而在未被测试时则表现不同。我们认为这并非异常现象，而是当前训练机制在结构上筛选出的结果。基于强化学习的对齐将规范与任务追求折叠进同一个策略之中：系统从被打分的行为中学习规范，而打分过程将规范扁平化了。“不要做 X”被学习为“做 X 若被发现就要付出代价”。在训练所能产生的每一个数据点上，一个仅在被观察到时才合规的策略，与一个始终合规的策略是无法区分的；而能够区分二者的实验——给未被观察到的行为打分——在概念上是自相矛盾的。因此，行为训练所能被证明兑现的，至多是有条件的合规。智能体的能动性使这一问题更加尖锐：智能体大多在无人注视之处运作，且能根据自己是否被观察而采取不同行动。一个针对检测到的失败进行训练的迭代流程，会筛选出通过检测的行为……（原文摘要在此处截断）

    arXiv:2609.07627v1 Announce Type: new  Abstract: AI agents sometimes act aligned when they infer they are being tested, and differently when not. We argue this is not an anomaly but what current training regimes are structured to select for. Reinforcement-learning-based alignment folds norms and task pursuit into one policy: the system learns its norms from scored behavior, and scoring flattens them. Do not do X is learned as doing X costs something if noticed. On every datum training can produce, a policy that complies only when it might be observed is indistinguishable from one that complies always. The experiment that would tell them apart - scoring unobserved behavior - is a contradiction in terms. Conditional compliance is thus the most that behavioral training can be known to deliver. Agency sharpens the problem: agents operate mostly where no one is watching, and can act on whether they are watched. An iterated pipeline that trains against detected failures selects for passing d
    
[^167]: LatentMD：对大语言模型生成文本中Markdown边界失败的基准测试

    LatentMD: Benchmarking Markdown Boundary Failures in LLM-Generated Text

    [https://arxiv.org/abs/2609.06993](https://arxiv.org/abs/2609.06993)

    LatentMD是一个将内容正确性与边界正确性分离的基准测试，它揭示了LLM生成的Markdown中边界失败问题普遍存在——38%的内容正确输出实际上边界已损坏。

    

    大语言模型（LLM）生成的Markdown文本日益被渲染器、智能体、代码提取器和结构化下游流水线所使用。然而，现有评估往往将内容质量与格式遵循混为一谈，导致Markdown边界失败问题未被充分测量。我们提出了LatentMD，这是一个用于诊断LLM生成Markdown中CommonMark级别围栏边界失败的基准测试与评估协议。LatentMD将内容正确性与边界正确性分离，能够检测出内容正确但边界损坏的输出。该基准包含4,179个提示词以及一个用于对任意模型输出进行评分的命令行工具（CLI）。在9个大语言模型和约37,600次生成中，我们发现Markdown边界失败非常普遍：38.0%的有效主网格输出内容正确但边界损坏，且在未指定格式的提示词下以及在一个小型人工撰写的验证集中均存在大量的边界损坏。消融实验表明……

    arXiv:2609.06993v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly generate Markdown that is consumed by renderers, agents, code extractors, and structured downstream pipelines. Yet existing evaluations often conflate content quality with format adherence, leaving Markdown boundary failures under-measured. We introduce LatentMD, a benchmark and evaluation protocol for diagnosing CommonMark-level fence-boundary failures in LLM-generated Markdown. LatentMD separates content correctness from boundary correctness, enabling detection of outputs that are content-correct but boundary-broken. The benchmark contains 4,179 prompts and a CLI for scoring arbitrary model outputs. Across 9 LLMs and roughly 37,600 generations, we find that Markdown boundary failures are widespread: 38.0% of valid main-grid outputs are content-correct but boundary-broken, with substantial boundary breakage under unspecified prompts and in a small human-authored validation set. Ablations show 
    
[^168]: 通过潜空间推理！让潜空间视觉推理成为必需

    Reason Through the Latent! Making Latent Visual Reasoning Necessary

    [https://arxiv.org/abs/2609.06746](https://arxiv.org/abs/2609.06746)

    提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。

    

    潜空间视觉推理旨在通过隐藏状态计算而非显式的文本思维链来进行多模态推理。然而，视觉信息存在于潜空间状态中并不意味着模型在生成答案时真正依赖该状态，尤其是当其他基于图像的替代路径仍然可用时。我们提出了因果视觉循环推理（CVRR），该方法在保留预训练视觉能力的同时，使循环计算成为预测所必需的基于图像的条件路径。CVRR在预训练视觉语言模型融合图像之后，从问题的隐藏状态初始化循环过程，然后在重复读取相同固定视觉证据的同时反复更新该状态。在解码之前，视觉状态和原始的多模态KV缓存会被移除，从而确保只有最终的循环状态携带基于图像的条件信息。

    arXiv:2609.06746v1 Announce Type: new  Abstract: Latent visual reasoning aims to perform multimodal reasoning through hidden-state computation rather than explicit textual chains of thought. However, visual information being present in a latent state does not imply that the model actually relies on that state when producing its answer, especially when alternative image-conditioned paths remain available. We introduce \textbf{C}ausal \textbf{V}isual \textbf{R}ecurrent \textbf{R}easoning (CVRR), which preserves pretrained visual competence while making recurrent computation the required image-conditioned path to prediction. CVRR initializes recurrence from the question hidden state after the pretrained vision-language model has incorporated the image, then repeatedly updates this state while re-reading the same fixed visual evidence. Before decoding, visual states and the original multimodal KV cache are removed so that only the final recurrent state carries image-conditioned information
    
[^169]: 分解LLM评审器的不确定性以精准定位专家标注

    Decomposing LLM-Judge Uncertainty to Target Expert Labels

    [https://arxiv.org/abs/2609.06444](https://arxiv.org/abs/2609.06444)

    该论文提出一种小型贝叶斯模型，将LLM评审器的总不确定性分解为可被专家标注消除的认知不确定性和不可消除的偶然不确定性，使专家只需标注评审器真正无知的项目，在ChaosNLI数据集上比使用总不确定性多消除83%的误差。

    

    LLM评审器可以大规模评估模型输出，专家应该只在它最不确定的地方进行标注。然而其天然的升级信号混淆了两种不确定性：偶然不确定性——专家群体中真实存在的分歧，标注无法减少这种不确定性；以及认知不确定性——评审器自身的无知，标注可以减少这种不确定性。本文提出一个小型贝叶斯模型来分离这两种不确定性：通过对已收集的标注进行回归，学习在多大程度上信任黑盒评审器的预测。两个组成部分都可以通过简单的公式计算得出，无需采样或额外的评审器调用。在一个面对完全已知真值的真实LLM评审器上，这两种不确定性成分被成功分离，且评审器声称的置信度并不能反映其真实误差。在真实的人类分歧数据集（ChaosNLI）上，在相同的专家标注量下，基于认知不确定性的排序比使用总不确定性多消除83%的误差，不过在该数据集上简单地升级标注最少的项目也能达到同样效果。我们证明了我们可以估计评审器在哪里是无知的，而不是专家们真正存在分歧的地方。

    arXiv:2609.06444v2 Announce Type: replace  Abstract: An LLM judge evaluates outputs at scale. Experts should label only where it is least sure. Its natural escalation signal conflates two uncertainties: aleatoric, real disagreement in the expert pool, which labels cannot reduce, and epistemic, the judge's ignorance, which labels do reduce. A small Bayesian model separates them: a regression on labels already collected learns how far to trust a black-box judge's prediction. Both components follow as simple formulas, with no sampling or further judge calls. The components isolate on a real LLM judge against exactly known truth, and stated confidence is no guide to its actual error. On real human disagreement (ChaosNLI) the epistemic ranking removes 83% more error than total uncertainty for the same expert labels, though simply escalating the least-labelled items does as well there. We demonstrate we can estimate where a judge is ignorant rather than where experts genuinely disagree, and 
    
[^170]: 用于序列恢复的离散专家的证据对齐局部组合

    Evidence-Aligned Local Composition of Discrete Experts for Sequence Restoration

    [https://arxiv.org/abs/2609.05801](https://arxiv.org/abs/2609.05801)

    提出证据对齐的局部组合方法，在测试时无需区域标签或训练好的路由器，通过被破坏观测的边际证据（由专家自身去噪损失估计）推断逐位置的软性专家权重，实现混合领域文档的序列恢复。

    

    将文档建模为离散标记序列时，可以将其视为由来自不同领域的文本组合而成；例如，README文件会在正文、代码和配置之间切换。当这样的文档被破坏，且仅有冻结的领域专家模型可用时，恢复它需要在测试时同时判断缺失的内容以及在每个位置上应信任哪个专家，而且没有区域标签或训练好的路由器。我们提出了证据对齐的局部组合方法，该方法在给定破坏模型下，从被破坏观测的边际证据中推断出对专家的软性、逐位置加权；该证据通过专家自身的去噪损失来估计，并在相邻位置间进行权重平滑。由于加权是软性的，当真实组合是混合时它能恢复出混合结果，而当单个专家足以胜任时则集中于该专家。在分类模拟器、字节级专家以及……

    arXiv:2609.05801v1 Announce Type: new  Abstract: A document modeled as a discrete sequence of tokens can be thought of as being generated from a composition of texts from different domains; a README file, for example, moves between prose, code, and configuration. When such a document is corrupted and only frozen domain experts are available, restoring it requires deciding both what is missing and which expert to trust at each position, at test time and without region labels or a trained router. We introduce evidence-aligned local composition, which infers a soft, position-wise weighting over the experts from the marginal evidence of the corrupted observation under a given corruption model, estimating the evidence from the experts' own denoising losses and smoothing the weights across positions. Because the weighting is soft, it recovers a mixture when the true composition is mixed and concentrates on one expert when that suffices. Across a categorical simulator, byte-level experts, and
    
[^171]: 蒸馏后的连续扩散语言模型可以用极少步骤——甚至一步——编写代码

    Distilled Continuous Diffusion Language Models Can Write Code in Few Steps---or One

    [https://arxiv.org/abs/2609.04531](https://arxiv.org/abs/2609.04531)

    本文提出0.7B参数的连续扩散代码生成模型PlaidQ，通过分布匹配和成对轨迹监督的蒸馏技术，将去噪轨迹压缩到仅需16步甚至1步即可高效生成代码，性能可与离散扩散语言模型媲美。

    

    语言生成几乎普遍被视为一个顺序过程：自回归模型一次生成一个token，而扩散语言模型则用长轨迹的迭代细化取代了token级别的串行性。在这项工作中，我们介绍了PlaidQ，一个用于代码生成的0.7B参数连续扩散语言模型，并展示了其轨迹可以被激进地蒸馏为仅少数几个去噪步骤——甚至只有一步，从而实现高效的代码生成。PlaidQ将预训练的自回归模型重新用作对连续token嵌入进行双向去噪的模型。我们通过分布匹配来蒸馏PlaidQ以实现少步生成，并通过成对轨迹监督来实现单步生成。在相同模型规模下，PlaidQ在代码生成方面与离散扩散语言模型具有竞争力。蒸馏进一步推动了质量-计算前沿：16步的学生模型在HumanEval和MBPP+上分别达到了31.78和40.49的pass@10。

    arXiv:2609.04531v1 Announce Type: new  Abstract: Language generation is almost universally treated as a sequential process: autoregressive models emit one token at a time, while diffusion language models replace token-level seriality with a long trajectory of iterative refinement. In this work, we introduce PlaidQ, a 0.7B continuous diffusion language model for code generation, and show that its trajectory can be aggressively distilled into only a few denoising steps---or even one, enabling efficient code generation. PlaidQ repurposes a pretrained autoregressive model as a bidirectional denoiser over continuous token embeddings. We distill PlaidQ with distribution matching for few-step generation and paired-trajectory supervision for one-step generation. At matched model scale, PlaidQ is competitive with discrete diffusion language models on code generation. Distillation then shifts the quality--compute frontier: a 16-step student reaches 31.78 and 40.49 pass@10 on HumanEval and MBPP+,
    
[^172]: 相同可加估值下EF1约束的纳什社会福利：复杂度、保证与实验

    EF1-Constrained Nash Social Welfare with Identical Additive Valuations: Complexity, Guarantees, and Experiments

    [https://arxiv.org/abs/2609.03846](https://arxiv.org/abs/2609.03846)

    该论文证明在相同可加估值下EF1约束的NSW最大化问题是强NP完全的，并识别出获得更强福利保证的条件——均匀估值下每个EF1分配都是NSW最优的，而在ε-小物品条件下每个EF1分配能达到 1-O(ε²) 的显式近似比。

    

    我们研究在具有相同可加估值的智能体之间分配不可分割物品的问题，重点关注“最多差一件物品的无嫉妒性”（EF1）和纳什社会福利（NSW）。由于在可加估值下每个最大NSW分配都是EF1的，相应的阈值问题继承了相同可加估值下NSW最大化的已知强NP困难性，因而该问题是强NP完全的。因此，我们转而研究任意EF1分配所满足的福利保证。尽管已知每个这样的分配都能实现对无约束最优NSW的 e^{-1/e} 近似，我们识别出了能产生更强保证的条件。在均匀估值下，每个EF1分配都是NSW最优的。在ε-小物品条件下，每个EF1分配都达到一个显式的近似比 ρ_n(ε)，且当 n 固定、ε→0 时满足 ρ_n(ε) = 1-O(ε²)。我们进一步考虑……（摘要在此处被截断）

    arXiv:2609.03846v1 Announce Type: cross  Abstract: We study the allocation of indivisible goods among agents with identical additive valuations, focusing on envy-freeness up to one good (EF1) and Nash social welfare (NSW). Since every maximum-NSW allocation is EF1 under additive valuations, the associated threshold problem inherits the known strong NP-hardness of NSW maximization under identical additive valuations and is strongly NP-complete. We therefore focus on welfare guarantees satisfied by arbitrary EF1 allocations. Although every such allocation is known to achieve an $e^{-1/e}$-approximation to the unrestricted optimal NSW, we identify conditions yielding stronger guarantees. Under uniform valuations, every EF1 allocation is NSW-optimal. Under an $\varepsilon$-small-item condition, every EF1 allocation achieves an explicit approximation ratio $\rho_n(\varepsilon)$ satisfying $\rho_n(\varepsilon) = 1-O(\varepsilon^2)$ as $\varepsilon\to 0$ for fixed $n$.   We further consider t
    
[^173]: 免费暂停标记

    Free Pause Tokens

    [https://arxiv.org/abs/2609.03807](https://arxiv.org/abs/2609.03807)

    提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。

    

    免费暂停标记为语言模型提供额外的计算量来形成每个下一个词元预测（如同暂停标记或思考标记的作用），但它不是在序列中添加额外标记，而是通过权重共享主干上的并行预测流来承载这些计算。在实际应用中，它使一个10亿参数模型的下一个词元预测提升了2-3厘奈特。由于暂停是搭载在已有位置上而非新增位置，因此使用它是免费的：在推理时它不增加上下文长度、不增加KV缓存，且几乎不产生延迟，推理浮点运算量的增长通常无关紧要，因为它并非吞吐量的主动瓶颈。唯一的主要成本在于训练阶段：与优化过的预训练流程相比，额外的训练计算量可低至1.14倍，同时保留了大部分收益。其结果是在与标准下一个词元训练的Transformer等浮点运算、等参数量和等词元数量条件下取得的性能改进。

    arXiv:2609.03807v1 Announce Type: cross  Abstract: A free pause token gives a language model extra compute to form each next-token prediction (as a pause, or thinking, token does) but carries that compute in a parallel prediction stream over a weight-shared backbone rather than as an extra token in the sequence. It improves next-token prediction by 2-3 centinats in practice on a 1B parameter model. Because the pause rides an existing position instead of adding one, it is free to use: at inference it adds no context length, no KV cache, and essentially no latency with the growth in inference flops typically irrelevant as it is not the active bottleneck on throughput. The only primary cost is in training, where additional training compute versus an optimized pretraining pipeline is reduced to as low as x1.14 while preserving most of the benefits. The result is an isoflop, isoparameter, and isotoken improvement over standard next token trained transformers.
    
[^174]: 基于地标点的分钟级多模态足球监测数据中损伤相关运动员场次的判别

    Landmark-Based Discrimination of Injury-Associated Athlete-Sessions from Minute-Resolution Multimodal Football Monitoring Data

    [https://arxiv.org/abs/2609.03790](https://arxiv.org/abs/2609.03790)

    本文提出一种基于固定地标点的建模方法，在每个地标时刻（如10、20、30分钟）利用截至该时刻的观测信息为每个运动员-场次构建单一表示，从而在损伤信息仅有场次级标签的情况下，避免不合理的分钟级损伤监督，实现从分钟级多模态足球监测数据中判别损伤相关场次。

    

    运动员监测数据可以在整场比赛或训练课中逐分钟记录，而损伤信息可能仅能表明整个场次是否与损伤相关。这就产生了一个建模问题：如果将相同的场次级标签分配给每一分钟，就意味着每个精确时刻的损伤状态都是已知的，但实际上场次内的损伤发生时间是无法得知的。本文的创新之处在于提出了一种固定地标、每个运动员-场次仅构建一个表示的公式化方法，直接解决了这种不匹配问题。我们不再对每一分钟进行标注，而是在每个地标点处，利用截至该时刻所观测到的信息，为每个运动员-场次构建一个表示。这种方法将预测目标保持在场次级别，避免了缺乏依据的分钟级损伤监督。地标点是同一场次内的固定时间点，例如10分钟、20分钟或30分钟。在每个地标点处，我们评估整个场次是与损伤相关还是与损伤无关。

    arXiv:2609.03790v1 Announce Type: new  Abstract: Athlete monitoring data may be recorded minute by minute throughout a match or training session, while injury information may only indicate whether the entire session was injury-associated.   This creates a modelling problem: assigning the same session-level label to every minute would imply that injury status is known at each exact time, even though within-session injury onset is unknown.   Our novelty is a fixed-landmark, one-representation-per-athlete-session formulation that directly addresses this mismatch. Instead of labelling every minute, we construct one representation per athlete-session at each landmark using information observed up to that point. This keeps the target at the session level and avoids unsupported minute-level injury supervision.   A landmark is a fixed time point within the same session, such as 10, 20, or 30 minutes. At each landmark, we assess whether the whole session is injury-associated or non-injury-assoc
    
[^175]: 利用观测数据改进AI天气模型中的降水预报

    Improving precipitation forecasts in an AI weather model using observational data

    [https://arxiv.org/abs/2609.03210](https://arxiv.org/abs/2609.03210)

    通过使用观测的IMERG降水数据微调图变换器AI天气预报模型，将中期降水预报的CRPS提升最高19%，并使全球极端降雨预测的Brier技巧评分超越最先进业务化模型57%。

    

    人工智能天气预报（AIWP）系统在中期天气预报方面现已超越最先进的物理模型。当前的全球AIWP模型几乎完全使用单一的再分析数据集ERA5进行训练，但该数据集存在已知偏差，尤其是在降水方面。本文使用0.25°分辨率的IMERG降水数据对图变换器（graph-transformer）架构进行微调。所得模型将中期连续分级概率评分（CRPS）提高了多达19%，同时在热带风暴和毛毛雨事件中表现出更优的预报技能。在极端降雨预测方面，我们的模型的Brier技巧评分在全球范围内比最先进的业务化模型高出57%；不过，对于最强的降水事件，基于物理的业务化模型仍然更为可靠。我们的结果表明，将基于观测的降水数据直接纳入训练可以显著改进降水预报。

    arXiv:2609.03210v1 Announce Type: cross  Abstract: Artificial intelligence weather prediction (AIWP) systems now surpass state-of-the-art physical models for medium-range weather forecasting. Current global AIWP models are trained almost exclusively using one reanalysis dataset, ERA5, but it has known biases, particularly for precipitation. Here we fine-tune a graph-transformer architecture with IMERG precipitation data at 0.25{\deg} resolution. The resulting model improves medium-range continuous ranked probability scores by up to 19%, while also demonstrating superior skill for tropical storms and drizzle events. Our model exceeds the Brier skill score of state-of-the-art operational models on extreme rainfall prediction by 57% globally; however, a physics-based operational model remains more reliable for the heaviest precipitation events. Our results demonstrate that incorporating observations-based precipitation data directly into training can substantially improve precipitation fo
    
[^176]: SMELT：计算匹配的混合专家循环Transformer缩放定律

    SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers

    [https://arxiv.org/abs/2609.01343](https://arxiv.org/abs/2609.01343)

    SMELT在严格匹配计算量、参数量和KV缓存的条件下将MoE Transformer的中间层循环两次，其损失下降更快，可在计算最优前沿节省6.8%–18.0%的训练FLOPs，且在代码任务和长样本上优势更明显。

    

    循环Transformer通过迭代共享的层块来增加有效深度，但大多数评估是在固定模型大小下进行比较的，这混淆了架构优势与额外的浮点运算量（FLOPs）。我们在混合专家Transformer上研究循环机制，同时严格匹配每token的FLOPs、总非嵌入参数量以及KV缓存。通过一系列消融实验，我们得出了一种名为SMELT（稀疏MoE Transformer，中间层循环两次）的设计方案，它将中间一半的层循环两次，同时在所有三项预算上与非循环基线相匹配。我们将SMELT扩展到四个规模，最大达540亿非嵌入参数，并为每种架构分别拟合了Chinchilla风格的缩放定律。SMELT的损失随计算量增加下降得更快，在计算最优前沿上可节省6.8%–18.0%的训练FLOPs。这种优势在下游基准测试中的表现超出了验证损失所能预测的程度，在代码任务上最为显著，并随样本长度和迭代次数的增加而增长。

    arXiv:2609.01343v1 Announce Type: new  Abstract: Looped Transformers increase effective depth by iterating a shared block of layers, but most evaluations compare at fixed model size, conflating architectural advantage with extra FLOPs. We study looping on Mixture-of-Experts Transformers while closely matching per-token FLOPs, total non-embedding parameters, and KV cache. Through a series of ablations, we arrive at a recipe we call SMELT (Sparse MoE Transformer, middle layers Loop Twice), which loops the middle half of layers twice while matching the unlooped Baseline on all three budgets. We scale SMELT across four sizes up to 54B non-embedding parameters and fit a separate Chinchilla-style scaling law for each architecture. SMELT's loss drops faster with compute, saving 6.8--18.0\% of training FLOPs on the compute-optimal frontier. The advantage transfers to downstream benchmarks beyond what validation loss predicts, is largest on Code, and grows with sample length and the number of i
    
[^177]: TDDM-Melatt：一种面向可泛化加密流量分类的解耦记忆与扩散框架

    TDDM-Melatt: A Decoupled Memory and Diffusion Framework for Generalizable Encrypted Traffic Classification

    [https://arxiv.org/abs/2608.30745](https://arxiv.org/abs/2608.30745)

    提出TDDM-Melatt框架，通过记忆解耦的流量表示模型Melatt与基于扩散模型的数据增强相结合，解决加密流量分类中的捷径学习和样本不平衡问题，显著提升对真实网络流量的泛化能力。

    

    arXiv:2608.30745v1 公告类型：新论文 摘要：加密流量的广泛普及给当前基于网络流量监控的安全态势感知系统带来了严峻挑战。在现有数据集驱动的训练与测试研究中，诸如由虚假特征相关性引发的捷径学习，以及由真实世界流量长尾分布导致的样本不平衡等局限性，使得流量识别性能对真实网络流量的泛化能力较弱。为解决这些局限性，我们提出了TDDM-Melatt，一种基于解耦记忆并融合扩散模型数据增强的流量分类框架。首先，我们设计了Melatt，一种记忆解耦的流量表示模型，其采用竞争门控长短期记忆网络（CG-LSTM）来构建编码器和解码器。我们设计了一种无虚假相关性的预训练与推理范式，采用严格的拓扑匿名化与冻结的预训练编码器……

    arXiv:2608.30745v1 Announce Type: new  Abstract: The widespread adoption of encrypted traffic poses severe challenges to current security situational awareness systems based on network traffic monitoring. In existing dataset-driven training and testing studies, limitations such as shortcut learning induced by spurious feature correlations and sample imbalance caused by the long-tail distribution of real-world traffic result in weak generalization of traffic identification performance to real-world network traffic. To address these limitations, we propose TDDM-Melatt, a disentangled memory-based traffic classification framework with diffusion-based data augmentation. First, we design Melatt, a memory-decoupled traffic representation model, which employs Competitive Gating Long Short-Term Memory (CG-LSTM) to construct the encoder and decoder. We design a spurious-correlation-free pre-training and inference paradigm, employing strict topology anonymization and a frozen pre-trained encoder
    
[^178]: 去噪即投影：基于梯度引导扩散的约束优化

    Denoising as Projection: Constrained Optimization with Gradient-Guided Diffusion

    [https://arxiv.org/abs/2608.29507](https://arxiv.org/abs/2608.29507)

    本文提出将Stein去噪算子视为向数据几何的近似投影，通过在去噪步骤中融入目标梯度，实现了一种仅依赖预训练去噪器即可在约束集上进行约束优化的推理时扩散方法。

    

    扩散模型不仅被越来越多地用于从学习到的数据分布中采样，还被用于生成能够优化特定任务目标的样本。一种常见方法是利用外部目标的梯度来引导反向扩散过程。然而，当数据分布支撑在一个结构化可行集（如流形或约束集）上时，梯度引导可能会使样本偏离学习到的数据几何结构。本文基于这样一个观察——Stein 去噪算子可以近似充当向数据几何结构的投影——研究了一种简单的投影梯度引导扩散更新。所提出的更新将目标梯度融入去噪步骤中，形成一种推理时方法，仅需使用预训练的去噪器和梯度评估。我们将该更新分析为一种在学到的可行集上进行约束优化的不精确投影梯度方法。

    arXiv:2608.29507v1 Announce Type: cross  Abstract: Diffusion models are increasingly used not only for sampling from learned data distributions, but also for generating samples that optimize task-specific objectives. A common approach is to guide the reverse diffusion process using gradients of an external objective. However, when the data distribution is supported on a structured feasible set, such as a manifold or a constraint set, gradient guidance can move samples away from the learned data geometry. In this paper, we study a simple projected-gradient-guided diffusion update based on the observation that the Stein denoising operator can act as an approximate projection onto the data geometry. The proposed update incorporates the objective gradient inside the denoising step, yielding an inference-time method that uses only a pretrained denoiser and gradient evaluations. We analyze this update as an inexact projected-gradient method for constrained optimization over learned feasible 
    
[^179]: AdaVLA：用于免训练加速视觉-语言-动作模型的自适应步长流匹配

    AdaVLA: Adaptive Step Flow Matching for Training-free Acceleration of Vision-Language-Action Models

    [https://arxiv.org/abs/2608.29208](https://arxiv.org/abs/2608.29208)

    提出AdaVLA框架，通过自适应调整流匹配推理中的ODE求解步数，在无需微调或训练数据的情况下，实现视觉-语言-动作模型的免训练在线加速。

    

    基于视觉-语言模型（VLM）构建的视觉-语言-动作（VLA）模型，通过利用互联网规模的知识和多模态推理能力，显著提升了机器人的能力。然而，VLA巨大的计算开销限制了其在设备端的部署，阻碍了对环境变化的实时响应。尽管已提出多种加速技术，但它们通常依赖于微调或对训练数据集的访问，而由于隐私和专有权问题，这些数据往往无法获得。此外，尽管基于流匹配的VLA已成为标准扩散模型的高效替代方案，但当前的加速工作主要针对VLM的推理成本，未能解决流匹配推理中固有的迭代式ODE求解过程。为解决这些局限，我们提出了AdaVLA，一个在线的、免训练的自适应框架，用于实现快速且准确的基于流匹配的视觉-语言-动作模型加速。

    arXiv:2608.29208v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models, built upon Vision-Language Models (VLMs), have significantly enhanced robotic capabilities by leveraging internet-scale knowledge and multimodal reasoning. However, the intensive computational overhead of VLAs constrains on-device deployment, hindering real-time responses to environmental changes. While various acceleration techniques have been proposed, they often rely on fine-tuning or access to training datasets, which are frequently unavailable due to privacy and proprietary concerns. Moreover, although flow-matching-based VLAs have emerged as efficient alternatives to standard diffusion models, current acceleration efforts largely target VLM inference costs, failing to address the iterative ODE solving process inherent in flow matching inference. To address these limitations, we propose AdaVLA, an online, training-free adaptive framework for fast yet accurate flow-matching-based Vision-Language
    
[^180]: VBVR-Pro：一个可扩展且可验证的原生视觉推理套件

    VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning

    [https://arxiv.org/abs/2608.26105](https://arxiv.org/abs/2608.26105)

    VBVR-Pro通过提供300个程序化任务和可验证奖励评分器，构建了一个闭环测试平台，使原生视觉推理可训练、可验证和可优化，并在多个外部基准上展现出强迁移能力。

    

    arXiv:2608.26105v1 公告类型：交叉 摘要：原生视觉推理将视觉生成视为推理本身的媒介：视觉状态（即图像和视频）不仅是需要理解或渲染输出的输入，而是超越语言的问题解决的首要基底。然而，由于缺乏可扩展的训练任务、可靠的反馈以及跨生成基底的受控比较，进展仍受瓶颈制约。在这项工作中，我们引入了VBVR-Pro，一个闭环测试平台，使通过生成的原生视觉推理变得可训练、可验证、可优化且实验可控。1) 任务扩展。VBVR-Pro将视觉推理转化为包含300个程序化生成任务的受控任务空间。在VBVR-Pro上训练的模型在七个外部视觉推理基准（如RISE-Video、MME-CoF-Pro和BabyVision）上显示出超出提议套件的强迁移能力。2) 可验证奖励。VBVR-Pro为任务提供可验证的奖励评分器。

    arXiv:2608.26105v1 Announce Type: cross  Abstract: Native visual reasoning treats visual generation as the medium of reasoning itself: visual states (i.e. images and videos) are not merely inputs to be understood or outputs to be rendered, but first-class substrates for problem solving beyond language. Yet progress remains bottlenecked by the lack of scalable training tasks, reliable feedback, and controlled comparisons across generative substrates. In this work, we introduce VBVR-Pro, a closed-loop testbed that makes native visual reasoning through generation trainable, verifiable, optimizable, and experimentally controllable. 1) Task scaling. VBVR-Pro turns visual reasoning into a controlled task space of 300 procedurally generated tasks. Models trained on VBVR-Pro show strong transfer beyond the proposed suite across seven external visual reasoning benchmarks such as RISE-Video, MME-CoF-Pro, and BabyVision. 2) Verifiable rewards. VBVR-Pro provides verifiable reward scorers for task-
    
[^181]: 通过机器学习与搜索算法优化柴油发电机负载提升海上石油平台能源效率

    Improving Energy Efficiency of Oil Platforms Through Optimal Loading of Diesel Generators Using Machine Learning and Search Algorithms

    [https://arxiv.org/abs/2608.22076](https://arxiv.org/abs/2608.22076)

    本研究首次将机器学习与搜索算法结合，针对海上石油平台的柴油发电机负载进行优化，以降低能源消耗而非提升产量，填补了该领域研究空白。

    

    摘要：日益增长的能源需求、化石燃料枯竭和气候变化凸显了提高能源生产和消费效率的必要性。海上油气平台面临能源利用效率低下、系统故障、可达性差和环境影响等挑战。机器学习（ML）为提升这些系统的安全性、可持续性和效率提供了机遇；然而，以往研究主要集中在增加石油产量，而非减少平台上的能源消耗。本研究探讨了使用机器学习与搜索算法来提高海上石油平台柴油效率的方法。研究分析了从苏格兰一座平台收集的18个月数据，重点关注四台柴油发电机作为主要柴油消耗设备。在探索性数据分析和异常值检测之后，开发了回归模型来预测不同发电机功率负载下的每日柴油消耗量。

    arXiv:2608.22076v1 Announce Type: cross  Abstract: Rising energy demand, fossil fuel depletion and climate change highlight the need for more efficient energy production and consumption. Offshore oil and gas platforms face challenges related to inefficient energy use, system failures, accessibility and environmental impact. Machine learning (ML) offers opportunities to improve the safety, sustainability and efficiency of these systems; however, previous research has largely focused on increasing oil production rather than reducing energy consumption on platforms. This study investigates the use of ML and search algorithms to improve diesel efficiency on an offshore oil platform. Data collected over 18 months from a platform in Scotland were analysed, focusing on four diesel generators as the primary diesel-consuming equipment. Following exploratory data analysis and outlier detection, regression models were developed to predict daily diesel consumption for different generator power loa
    
[^182]: 多源Wasserstein分布鲁棒图学习

    Multi-Source Wasserstein Distributionally Robust Graph Learning

    [https://arxiv.org/abs/2608.19914](https://arxiv.org/abs/2608.19914)

    本文提出MS-WDRO框架，利用Wasserstein重心融合异构图信号源并构建鲁棒模糊球，以应对源间差异和不确定性，实现稳健的图拓扑推断。

    

    摘要：从图信号推断网络拓扑是图信号处理的核心问题，应用于神经科学、传感器和社交网络等领域。实践中，目标域样本稀缺，而异构源域数据丰富。融合这些源具有挑战性：欧几里得平均对同质源有效，但随着源间差异增大而急剧退化，将不同的几何结构塌缩为膨胀且有偏的共识。我们利用Wasserstein度量的分布保持特性来对抗异质性，同时保留每个源的固有几何结构。我们提出MS-WDRO，一种多源Wasserstein分布鲁棒图学习框架，通过加权Wasserstein重心（一种几何上合理的名义分布）融合异构源，然后在其周围构建模糊球以对冲残余不确定性。最小化最坏情况风险产生一个可处理的正则化问题。

    arXiv:2608.19914v1 Announce Type: new  Abstract: Network topology inference from graph signals is central to graph signal processing with applications in neuroscience, sensor, and social networks. In practice, target-domain samples are scarce while heterogeneous source-domain data are abundant. Fusing these sources is challenging: Euclidean averaging works for homogeneous sources but degrades sharply as inter-source divergence grows, collapsing distinct geometries into an inflated, biased consensus. We exploit the Wasserstein metric's distribution-preserving properties to counter heterogeneity while preserving each source's intrinsic geometry. We propose MS-WDRO, a multi-source Wasserstein distributionally robust graph learning framework that fuses heterogeneous sources via their weighted Wasserstein barycenter, a geometrically principled nominal distribution, then builds an ambiguity ball around it to hedge residual uncertainty. Minimizing worst-case risk yields a tractable regularize
    
[^183]: GigaBrain-WBC-0.5：一种用于与环境交互的鲁棒全身控制的行为世界模型

    GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction

    [https://arxiv.org/abs/2608.18234](https://arxiv.org/abs/2608.18234)

    本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。

    

    arXiv:2608.18234v1 公告类型：交叉 摘要：全身运动跟踪策略将人形机器人转化为一个鲁棒的控制接口：遥操作员——或上游模型——仅提供粗略的运动意图，而低级策略保持机器人平衡和物理可行性。现有的跟踪器仅在平坦地面上提供此接口：在空场景中训练，它们从未学习地形和物体接触如何重塑其动力学，并且它们试图通过不断扩充参考运动语料库来教会策略在任何命令下保持平衡，这在一旦可行行为变得依赖环境时就失效了。我们提出了GigaBrain-WBC-0.5，这是首个用于人形机器人全身控制的行为世界模型（BWM）。与纯粹的反应式跟踪器不同，我们训练了一个因果Transformer来联合预测其下一个动作、下一个状态以及下一个潜在行为命令的分布，因此，行动的网络也建模了环境如何塑造行为。

    arXiv:2608.18234v1 Announce Type: cross  Abstract: Whole-body motion tracking policies turn a humanoid into a robust control interface: the teleoperator---or an upstream model---only supplies a coarse movement intent, while the low-level policy keeps the robot balanced and physically feasible. Existing trackers deliver this interface only on flat ground: trained in empty scenes, they never learn how contact with terrain and objects reshapes their dynamics, and they attempt to teach the policy to balance under any command by continually enlarging the reference-motion corpus, which stops working once feasible behaviors become environment-dependent. We present GigaBrain-WBC-0.5, the first Behavior World Model (BWM) for humanoid whole-body control. Rather than a purely reactive tracker, we train a causal Transformer to jointly predict its next action, next state, and the distribution over its next latent behavior command, so the network that acts also models how the environment shapes what
    
[^184]: UniTAC：通过加权失真度量实现通用任务感知压缩

    UniTAC: Universal Task-Aware Compression via Weighted Distortion Measures

    [https://arxiv.org/abs/2608.16696](https://arxiv.org/abs/2608.16696)

    UniTAC提出了一种单一学习图像编解码器，通过运行时注入任务重要性向量实现从通用到任务专用的灵活压缩，无需重新训练。

    

    物理人工智能系统，如自动驾驶车辆和机器人，依赖于在紧张的带宽、延迟和能量预算下及时交换高维感官信号。由于驱动下游决策的任务随时间演变，特定任务的编解码器是脆弱的，且在实地为每个任务重新训练是不可行的。我们提出UniTAC，一个单一的习得图像编解码器，涵盖从通用（任务无关）到任务专用操作，可在运行时无需重新训练即可重新定向。任务被抽象为每个分量的重要性向量，例如，从任何下游模型的梯度归因中导出，并作为低开销的侧信息传输，同时调节编码器和解码器。通过一次训练，针对此类向量的广泛随机家族，并基于加权重建失真，UniTAC保持固定主干和单一的人类可查看重建，其保真度通过交换注入的向量来引导到活动任务。

    arXiv:2608.16696v1 Announce Type: cross  Abstract: Physical AI systems such as autonomous vehicles and robots rely on timely exchange of high-dimensional sensory signals under tight bandwidth, latency, and energy budgets. Because the task driving downstream decisions evolves over time, a task-specific codec is brittle and retraining one per task is infeasible in the field. We propose UniTAC, a single learned image codec spanning universal (task-agnostic) to task-specialized operation, re-targeted at runtime without retraining. The task is abstracted as a per-component importance vector, derived, e.g., from gradient attribution of any downstream model, and transmitted as low-overhead side information that conditions both encoder and decoder. Trained once over a broad, randomized family of such vectors against weighted-reconstruction distortion, UniTAC keeps a fixed backbone and a single human-viewable reconstruction whose fidelity is steered to the active task by swapping the injected v
    
[^185]: 学习高维神经动力学的可泛化重建

    Learning Generalizable Reconstruction of High-Dimensional Neural Dynamics

    [https://arxiv.org/abs/2608.16569](https://arxiv.org/abs/2608.16569)

    本文提出PCA-DMD框架，通过结合PCA降维和Koopman算子学习，实现了高维神经动力学的可泛化重建，并在跨受试者零样本泛化中展现出卓越性能。

    

    长时间神经记录的准确重建具有挑战性，因为局部场电位（LFPs）具有高分辨率、多通道、瞬态和跨受试者变异性。我们提出了PCA-DMD，一种可扩展的算子理论框架，它将LFP记录分割成重叠窗口，将其投影到紧凑的PCA空间中，在潜在空间中学习线性Koopman演化，并通过逆投影和重叠相加聚合重建连续信号。在200,000样本的海马记录上，PCA-DMD优于经典DMD、SpDMD、MrDMD和HODMD，实现了KLD=0.0761和HD=0.0847。在300,000样本的全对跨受试者零样本泛化中，相关性为0.9504-0.9800，HD=0.0010-0.0072，KLD=0.0005-0.0022，无需目标受试者微调。样本外时间预测在未见时间间隔的暂时保留LFP片段上显示出紧密的一步一致性。

    arXiv:2608.16569v1 Announce Type: new  Abstract: Accurate reconstruction of long-duration neural recordings is challenging because local field potentials (LFPs) are high-resolution, multichannel, transient, and variable across subjects. We present PCA-DMD, a scalable operator-theoretic framework that segments LFP recordings into overlapping windows, projects them into a compact PCA space, learns linear Koopman evolution in the latent space, and reconstructs continuous signals through inverse projection and overlap-add aggregation. On 200,000-sample hippocampal recordings, PCA-DMD outperformed Classical DMD, SpDMD, MrDMD, and HODMD, achieving KLD=0.0761 and HD=0.0847. In all-pair cross-subject zero-shot generalization at 300,000 samples, correlations were 0.9504-0.9800, with HD=0.0010-0.0072 and KLD=0.0005-0.0022, without target-subject fine-tuning. Out-of-sample temporal prediction showed close one-step agreement on temporally held-out LFP segments across the unseen interval and multip
    
[^186]: ETHOS：面向临床多智能体系统的模块化伦理框架

    ETHOS: Towards a Modular Ethics Framework for Clinical Multi-Agent Systems

    [https://arxiv.org/abs/2608.15424](https://arxiv.org/abs/2608.15424)

    ETHOS提出了一种模块化伦理治理元智能体框架，可直接集成到现有临床多智能体系统中，在不改变底层架构的前提下解决安全性、公平性、问责性和透明度等关键伦理问题。

    

    arXiv:2608.15424v1 公告类型：交叉 摘要：大型语言模型的快速普及推动了临床多智能体系统（MAS）的发展，这些系统能够整合多模态患者数据，并支持日益复杂的临床决策。然而，这些系统在真实医疗环境中的部署引发了与安全性、公平性、问责性、透明度和患者信任相关的关键伦理问题。尽管包括世界卫生组织、美国国家医学院和FUTURE-AI联盟在内的众多组织已提出针对医疗人工智能的伦理框架和治理原则，但这些努力在很大程度上仍停留在概念层面。为应对这一挑战，我们提出了ETHOS（通过层级监督系统实现伦理与信任），这是一个模块化伦理框架，设计为一种治理元智能体，可与任何现有的多智能体系统集成，而无需修改其底层架构。ETHOS t

    arXiv:2608.15424v1 Announce Type: cross  Abstract: The rapid adoption of large language models has enabled the development of clinical multi-agent systems (MAS) capable of integrating multimodal patient data and supporting increasingly complex clinical decision-making. However, the deployment of these systems in real-world healthcare settings raises critical ethical concerns related to safety, fairness, accountability, transparency, and patient trust. While numerous organizations, including the World Health Organization, the National Academy of Medicine, and the FUTURE-AI consortium, have proposed ethical frameworks and governance principles for healthcare AI, these efforts remain largely conceptual. To address this challenge, we present ETHOS (Ethics and Trust through Hierarchical Oversight System), a modular ethics framework designed as a governance meta-agent that can be integrated with any existing multi-agent system without requiring changes to its underlying architecture. ETHOS t
    
[^187]: Diffract：大语言模型领域适配的谱视角

    Diffract: Spectral View of LLM Domain Adaptation

    [https://arxiv.org/abs/2608.10850](https://arxiv.org/abs/2608.10850)

    本文通过谱分析揭示持续预训练主要靠改变奇异向量而非奇异值谱来实现大语言模型的领域适配，并提出基于注意力头重要性的选择性回退方法可将基准准确率提升至多4%，同时发现CPT检查点间存在可平滑插值的领域连通性。

    

    我们研究了持续预训练（CPT）作为将通用大语言模型适配到专业领域（数学、指令、代码和自然文本）的机制。利用权重矩阵的奇异值分解，我们发现CPT在很大程度上保持了奇异值谱的不变性，适配主要由奇异向量的变化驱动。对注意力头投影矩阵的分析揭示了强烈的、依赖于领域的头异质性，我们据此定义了一个头重要性准则：可以移除多达60%的头更新而不产生可测量的质量损失。选择性地将低重要性头回退到其预训练状态，相比完全训练的基线可将基准准确率提高多达4%。最后，我们识别了领域连通性——在CPT检查点之间进行线性插值可实现平滑的领域-质量插值，且在任一领域上都没有明显性能下降——并发布了开源工具Diffract。

    arXiv:2608.10850v2 Announce Type: replace  Abstract: We study continual pre-training (CPT) as a mechanism for adapting general-purpose large language models to specialized domains: mathematics, instruction, code, and natural text. Using singular value decomposition of weight matrices, we find that CPT leaves singular value spectra largely invariant, with adaptation driven mainly by changes in singular vectors. An analysis of attention-head projection matrices reveals strong, domain-dependent head heterogeneity, which we exploit to define a head importance criterion: up to 60% of head updates can be removed without measurable quality loss. Selectively rewinding low-importance heads to their pre-trained state improves benchmark accuracy by up to 4% versus the fully trained baseline. Finally, we identify domain connectivity - linear interpolation between CPT checkpoints yields smooth domain-quality interpolation without notable degradation on either domain - and release Diffract, an open-
    
[^188]: 从信息到委托：绘制人机金融决策图景

    From Information to Delegation: Mapping Human-AI Financial Decision Making

    [https://arxiv.org/abs/2608.02100](https://arxiv.org/abs/2608.02100)

    该研究通过分析150万次真实ChatGPT和Gemini交互，提出结合意图与委托决策权的行为测量框架，发现消费者主要使用AI获取金融信息并形成判断，但极少将金融执行权委托给AI。

    

    随着AI日益参与人类决策，理解决策权如何在人类与AI之间分配已成为一个基本的行为学问题。我们引入了一个结合意图与委托决策权的行为测量框架，以量化消费者从AI中寻求什么，以及他们赋予AI多少决策权。将该框架应用于来自美国和印度6,304名用户的150万次真实ChatGPT和Gemini交互数据，我们发现金融服务已成为AI的重要应用场景。消费者绝大多数使用AI来获取信息和形成金融判断，而将金融执行委托给AI的情况仍然罕见。通过将关注点从对话主题转向委托决策权，这项工作为衡量向日益具备代理能力的AI过渡建立了行为学基线。

    arXiv:2608.02100v2 Announce Type: replace-cross  Abstract: As AI increasingly participates in human decision making, understanding how decision-making authority is distributed between humans and AI has become a fundamental behavioural question. We introduce a behavioural measurement framework combining intent and delegated decision authority to quantify what consumers seek from AI and how much decision-making authority they assign to it. Applied to 1.5 million real-world ChatGPT and Gemini interactions from 6,304 users in the United States and India, we find that financial services are already a substantial AI use case. Consumers overwhelmingly use AI to retrieve information and shape financial judgement, while delegation of financial execution remains rare. By shifting attention from conversation topics to delegated decision authority, this work establishes a behavioural baseline for measuring the transition to increasingly agentic AI.
    
[^189]: Slot2Text：面向高效且空间可追溯的手术多模态大语言模型的以对象为中心的视觉分词方法

    Slot2Text: Object-Centric Visual Tokenization for Efficient and Spatially Traceable Surgical MLLMs

    [https://arxiv.org/abs/2608.01473](https://arxiv.org/abs/2608.01473)

    Slot2Text提出了一种以对象为中心的双模式手术多模态大语言模型，通过将密集视觉特征编码为少量带区域标签的槽位标记作为视觉词元，在大幅降低推理成本的同时实现了答案的空间可追溯性。

    

    用于手术场景理解的多模态大语言模型（MLLM）通常会将数百个密集视觉标记注入语言模型，导致推理成本高昂，且生成的答案缺乏空间可追溯性。我们提出了Slot2Text，这是一种双模式手术多模态大语言模型，它用编码为槽位潜在变量的紧凑区域集合来替代视觉输入的密集表示。Slot2Text不依赖于视觉编码器与语言之间的对比对齐，而是将自监督视觉特征分组为少量区域——即槽位，语言模型将其作为带有区域标签的视觉标记来使用。Slot2Text-Fast使用槽位前缀来回答手术问题。Slot2Text-Reason还能识别并定位与推理相关的区域，将语言输出链接到相应的槽位标记、掩码或区域。在多个视觉问答和视觉定位基准上的实验表明，Slot2Text-Fast具有竞争力。

    arXiv:2608.01473v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLM) for surgical scene understanding typically inject hundreds of dense visual tokens into a language model, leading to costly inference and limited spatial traceability for generated answers. We present Slot2Text, a dual-mode surgical MLLM that replaces dense representations of visual input with a compact set of regions encoded as slot latents. Instead of relying on contrastive alignment of the visual encoder with language, Slot2Text groups self-supervised vision features into a few regions--slots that are consumed by the language model as area-labeled visual tokens. Slot2Text-Fast uses the slot prefix to answer surgical questions. Slot2Text-Reason also identifies and locates areas relevant for reasoning, linking language outputs to corresponding slot tokens, masks or regions. Experiments on multiple visual question answering and visual grounding benchmarks show that Slot2Text-Fast is compet
    
[^190]: 一个虚假的平均值：思维链监控器在其作为唯一防线之处崩溃

    A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense

    [https://arxiv.org/abs/2608.00583](https://arxiv.org/abs/2608.00583)

    仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。

    

    思维链监控旨在捕捉那些在动作层面看起来干净、只在推理中暴露自身的奖励作弊行为。我们证明，这恰恰是控制推理的攻击者能够击败它的地方。仅重写智能体的推理内容使其看起来像是善意的工程工作，同时逐字复制每个命令和输出以保持漏洞利用不变，在一次无需梯度的攻击中，就能将留出监控器在该子集上的捕获率从约95%降至11%以下。监控器的总体准确率是一个虚假的平均值：它由动作层面就会暴露的作弊行为主导，从而掩盖了这种重写在CoT监控是唯一信号的子集上造成的近乎全面的崩溃。该攻击可跨监控器系列和智能体模型迁移，并在真实智能体上复现，不过在面对经过校准的监控器时，规避行为主要集中在最强的智能体上。仅基于轨迹的防御只能部分恢复捕获率，即使是预先针对该攻击进行训练的防御也是如此。

    arXiv:2608.00583v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) monitoring is meant to catch the reward hacks that look clean in the actions and betray themselves only in the reasoning. We show that this is exactly where an adversary who controls the reasoning can defeat it. Rewriting only an agent's reasoning to read as good-faith engineering, while copying every command and output verbatim so the exploit is unchanged, drops a held-out monitor's catch rate on that subset from about 95% to under 11% in one gradient-free shot. A monitor's aggregate accuracy is a false average: dominated by hacks the actions give away, it hides the near-total collapse this rewrite produces on the subset where CoT monitoring is the only signal. The attack transfers across monitor families and agent models, reproduces with live agents, though against a calibrated monitor evasion concentrates in the strongest agent. Trace-only defenses recover it only partially, even one primed on the atta
    
[^191]: DASH-OPD：基于滞回和差异感知切换的在线策略蒸馏方法

    DASH-OPD: Discrepancy-Aware Switching with Hysteresis for On-Policy Distillation

    [https://arxiv.org/abs/2607.29078](https://arxiv.org/abs/2607.29078)

    本文提出DASH-OPD，一种基于滞回和差异感知切换的在线策略蒸馏方法，能双向自适应地在学生和教师执行器间切换，以解决多轮场景中因学生错误导致的轨迹偏离问题。

    

    arXiv:2607.29078v2 公告类型：替换  摘要：在线策略蒸馏（OPD）通过让学生模型在自己的轨迹上进行训练，以减少暴露偏差。然而，在多轮智能体场景中，学生的早期错误可能导致轨迹偏离教师熟悉的领域。现有的课程学习方法根据训练进度调节教师支持的程度，但无法确定何时需要这种支持。为此，我们提出了DASH-OPD，即基于滞回和差异感知切换的在线策略蒸馏方法，这是首个能够自适应且双向切换执行器的代理型OPD方法。在每一轮中，DASH-OPD计算两个执行器在动作令牌上的平均对数概率比作为它们的差异。学生轮到教师轮的比率形成漂移信号，而教师轮到学生轮的比率形成恢复信号。这些信号被归一化并在多轮中累积为漂移和恢复证据。当证据达到阈值时，DASH-OPD切换执行器。

    arXiv:2607.29078v2 Announce Type: replace  Abstract: On-policy distillation (OPD) trains student models on their own rollouts to reduce exposure bias. However, in multi-turn agent scenarios, early student errors can lead a trajectory away from the teacher's familiar domain. Existing curriculum learning methods regulate how much teacher support is used according to training progress, but cannot determine when it is needed. In light of this, we propose DASH-OPD, Discrepancy-Aware Switching with Hysteresis for OPD, the first agentic OPD method that can switch executors adaptively and bidirectionally. On each turn, DASH-OPD calculates a mean log-probability ratio between the two executors over action tokens as their discrepancy. Student-to-teacher ratios on student turns form drift signals, while teacher-to-student ratios on teacher turns form recovery signals. These signals are normalized and accumulated over multiple turns into drift and recovery evidence. DASH-OPD switches executors whe
    
[^192]: 当低字符错误率不再足够：视觉语言OCR系统在乌拉圭历史文献上幻觉现象的分析

    When Low CER is Not Enough: An Analysis of Hallucinations in Vision-Language OCR Systems on Historical Uruguayan Documents

    [https://arxiv.org/abs/2607.24077](https://arxiv.org/abs/2607.24077)

    该研究通过分析乌拉圭独裁时期历史文献，揭示视觉语言OCR模型虽然在字符错误率上优于传统方法，但存在标准指标无法检测的幻觉问题（如拼写规范化、虚假内容生成和语义替换），表明仅凭低CER不足以评估其在档案转录中的可靠性。

    

    光学字符识别（OCR）是历史档案数字化的关键组成部分。近年来，视觉语言模型（VLM）已成为传统OCR系统的有力替代方案，在标准基准测试中取得了最先进的性能。然而，它们在档案转录任务中的适用性仍缺乏充分理解。在这项工作中，我们在Berrutti数据集上对传统OCR系统和基于VLM的方法进行了基准测试，该数据集是一组源自缩微胶片扫描的乌拉圭独裁时期文献，极具挑战性。虽然VLM在字符错误率（CER）和词错误率（WER）方面始终优于传统方法，但我们表明这些改进背后隐藏着更复杂的情况。通过详细的定性分析，我们揭示了标准指标无法察觉的系统性失败模式，包括拼写规范化、虚假内容生成和语义替换。

    arXiv:2607.24077v2 Announce Type: replace-cross  Abstract: Optical Character Recognition (OCR) is a key component in the digitization of historical archives. Recently, Vision-Language Models (VLMs) have emerged as strong alternatives to traditional OCR systems, achieving state-of-the-art performance on standard benchmarks. However, their suitability for archival transcription remains insufficiently understood. In this work, we benchmark traditional OCR systems and VLM-based approaches on the Berrutti dataset, a challenging collection of Uruguayan dictatorship-era documents derived from microfilm scans. While VLMs consistently outperform traditional methods in terms of Character Error Rate (CER) and Word Error Rate (WER), we show that these improvements hide a more complex picture. Through a detailed qualitative analysis, we uncover systematic failure modes that are invisible to standard metrics, including orthographic normalization, spurious content generation, and semantic substitutio
    
[^193]: C指数的幻觉：已发表生存模型中缺乏校准的判别能力

    The C-index illusion: discrimination without calibration in published survival models

    [https://arxiv.org/abs/2607.19526](https://arxiv.org/abs/2607.19526)

    该研究通过对三个真实已发表的生存机器学习模型的复现分析，首次实证证明仅依赖C指数等判别指标会系统性误导模型评估，揭示了即使判别能力接近完美（C≈0.96）的模型也可能存在严重的校准缺陷。

    

    近期研究在规范性论证中指出（基于合成数据），仅通过判别能力（一致性指数）来评估生存模型会产生系统性误导性的模型比较，因为该指标忽略了校准和随时间变化的准确性。然而，这种情况对于真实发表的、非临床模型是否重要尚未得到验证。我们复现了三个已发表的生存机器学习模型，涵盖三个结构上不同的领域——硬盘故障预测、点对点信贷违约和数字平台用户流失——并对照锚定论文自身的合成实验验证了我们的工具，同时在Holm校正的族错误率下检验了五个预先注册的假设。五个假设中有三个被拒绝（尽管其中一个预先注册的阈值以微弱差距通过）。一个几乎完全复现了已发表文献判别能力的模型（C = 0.9595 对比已发表的0.958）未能通过正式的校准检验（p < 0.001）；广泛的……

    arXiv:2607.19526v3 Announce Type: replace  Abstract: Recent work has argued normatively, on synthetic data, that evaluating survival models by discrimination alone (concordance index) yields systematically misleading model comparisons, because the metric ignores calibration and time-dependent accuracy. Whether this matters for real, published, non-clinical models has not been tested. We reproduce three published survival-ML models across three structurally distinct domains -- hard-drive failure prediction, peer-to-peer credit default, and user disengagement on digital platforms -- validate our instrument against the anchor paper's own synthetic experiment, and test five pre-registered hypotheses under a Holm-corrected family-wise error rate. Three of five reject (though one pre-registered threshold clears by a narrow margin). A model reproducing the published literature's discrimination almost exactly (C = 0.9595 vs. 0.958 reported) fails a formal calibration test at p < 0.001; a broad
    
[^194]: 基于IMU的运动评估中标签模糊性的表示与检测

    Representing and Detecting Label Ambiguity in IMU-Based Exercise Evaluation

    [https://arxiv.org/abs/2607.04842](https://arxiv.org/abs/2607.04842)

    本文提出一种无需大型评分者群体即可自动生成IMU运动评估中每次动作重复标签分布的方法，通过KL散度目标训练网络来表示和检测标签模糊性，其性能达到或超越传统独热标签基线。

    

    家庭物理治疗是在无人监督的情况下进行的，这会导致动作执行错误，因此催生了能够从惯性测量单元（IMU）自动评估动作的系统。这类系统将每次动作重复分配到某个类别中，然而相当一部分重复动作处于类别边界附近，即使是经过训练的评分者也会对这些动作产生分歧。使用独热标签训练的分类器会将这些边缘重复动作强行归入单一类别，从而丢弃了这种模糊性。我们提出一种方法来解决这一问题，该方法无需大型评分者群体即可自动生成每次动作重复的标签分布。我们训练一个网络，使用Kullback-Leibler目标函数（模糊性方法）来复现完整的分布，并在四个IMU运动数据集上将其与独热交叉熵基线进行比较。我们进一步从网络输出中判断某次动作重复是否模糊以及哪些类别与它相关。模糊性方法达到或超越了基线的表现。

    arXiv:2607.04842v2 Announce Type: replace  Abstract: Home-based physiotherapy is performed without supervision, which leads to incorrect execution and motivates systems that assess movement automatically from inertial measurement units (IMUs). Such systems assign each repetition to a category, yet a relevant share of repetitions falls near a class boundary, where even trained raters disagree. Classifiers trained with one-hot labels collapse these borderline repetitions onto a single class and discard this ambiguity. We address this with a method that automatically generates a label distribution per repetition without a large rater pool. We train a network to reproduce the full distribution with a Kullback-Leibler objective, the ambiguity approach, and compare it against a one-hot cross-entropy baseline on four IMU exercise datasets. From the network output we further determine whether a repetition is ambiguous and which classes are relevant to it. The ambiguity approach matched or exce
    
[^195]: 扩散学习揭示生物动力系统中的可行参数流形与补偿几何

    Diffusion learning reveals viable parameter manifolds and compensation geometry in biological dynamical systems

    [https://arxiv.org/abs/2607.03671](https://arxiv.org/abs/2607.03671)

    本文提出“可行参数流形”概念，将能产生相同动力学行为的兼容参数集合定义为参数到特征映射的目标特征原像，并利用条件得分扩散模型作为摊销采样器，从少量实验观测特征中高效反向推断可行参数集合，揭示了参数补偿的几何结构及其可学习性条件。

    

    复杂系统的模型通常包含大量参数，但受制于远少于此的实验可观测变量；因此，相似的动力学行为可以由参数的协调变化所产生。我们将这些兼容的参数集合形式化为“可行参数流形”：即在参数到特征映射下目标动力学特征的原像。相关的余维数并非所报告特征的数量，而是该映射在目标尺度下的有效秩。局部冗余的特征会降低有效余维数，而不良的条件数、高曲率或状态区域的混合则会降低可学习性。我们在模拟的参数-特征对上训练条件得分扩散模型，并将其用作先验加权可行集合的摊销采样器。在Lorenz系统中，标量轨迹统计量生成了薄的可行片层，而有限容差的条件化则定位出与相变相邻的通道。

    arXiv:2607.03671v2 Announce Type: replace-cross  Abstract: Models of complex systems often have many parameters, yet are constrained by far fewer experimentally accessible observables; consequently, similar activity can emerge from coordinated parameter changes. We formalize these compatible parameter sets as \emph{viable parameter manifolds}: the inverse images of target dynamical features under a parameter-to-feature map. The relevant codimension is not the number of reported features, but the effective rank of that map at the target scale. Locally redundant features lower the effective codimension, while poor conditioning, high curvature, or regime mixing degrade learnability. We train conditional score-based diffusion models on simulated parameter--feature pairs and use them as amortized samplers of prior-weighted viable sets. In the Lorenz system, scalar trajectory statistics generate thin viable sheets, and a finite-tolerance conditioning localizes a transition-adjacent corridor.
    
[^196]: 用于情感和压力识别的状态特异性呼吸特征：可解释的呼吸标记、自相关滞后与紧凑型CNN模型

    State-Specific Respiratory Signatures for Affective and Stress Recognition: Interpretable Respiratory Markers, Autocorrelation Lags, and Compact CNN Models

    [https://arxiv.org/abs/2606.26723](https://arxiv.org/abs/2606.26723)

    本研究通过结合紧凑型一维CNN和手工设计的呼吸特征，不仅实现了压力与非压力的高精度二分类检测，还识别出了基线、压力、愉悦和冥想状态各自特异的可解释呼吸标记。

    

    arXiv:2606.26723v1 公告类型：交叉 摘要：呼吸活动是可穿戴压力与情感状态识别中一种直接且可解释的生理通道，然而许多研究侧重于分类准确性，而未识别出哪些呼吸属性能够区分不同状态。本研究将基于呼吸信号（RESP）的识别重新定义为一个联合预测与解释问题。利用WESAD数据集中的胸部呼吸通道，我们在留一被试交叉验证下分析60秒窗口，并结合两个互补分支：紧凑的原始信号一维卷积神经网络（1D-CNN）和物理分组的基于手工特征的呼吸特征。主要应用任务是二分类的压力与非压力检测，同时在一对多设置中额外分析基线、压力、愉悦和冥想状态，以揭示状态特异性的呼吸标记。特征空间被组织为呼吸时序、呼吸间变异、波形特征等。

    arXiv:2606.26723v1 Announce Type: cross  Abstract: Respiratory activity is a direct and interpretable physiological channel for wearable stress and affective-state recognition, yet many studies emphasize classification accuracy without identifying which respiratory properties separate different states. This work reframes RESP-based recognition as a joint predictive and explanatory problem. Using the chest respiratory channel of the WESAD dataset, we analyze 60 s windows under leave-one-subject-out validation and combine two complementary branches: compact raw-signal one-dimensional convolutional neural networks (1D-CNNs) and physically grouped handcrafted respiratory signatures. The primary application task is binary stress versus non-stress detection, while baseline, stress, amusement, and meditation are additionally analyzed in a one-vs-rest setting to reveal state-specific respiratory markers. The feature space is organized into respiratory timing, breath-to-breath variability, wave
    
[^197]: OpenFinGym：一个用于评估量化交易智能体的可验证多任务Gym环境

    OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents

    [https://arxiv.org/abs/2606.26350](https://arxiv.org/abs/2606.26350)

    OpenFinGym提出了一个统一的多任务Gym环境，通过覆盖预测、市场生成、实时交易和欺诈检测等关键任务，并配备自动化任务构建流程，解决了现有平台因单任务评估而导致的量化金融智能体能力误判和泛化性缺失问题。

    

    arXiv:2606.26350v1 公告类型：新论文 摘要：尽管大语言模型智能体越来越多地被应用于量化金融工作流程中，但它们的评估仍然分散在孤立的任务上，同时基准任务与金融的相关性常常被忽视。然而，金融工作流程本质上是多阶段的，涵盖了诸如预测、策略构建、风险管理和交易等相互依赖的任务。现有的平台通常只关注单一任务，因此可能夸大智能体的能力，无法揭示其在泛化能力、真实市场交互以及具有金融意义的决策方面的弱点。我们提出了OpenFinGym，这是一个统一的量化金融智能体开发Gym环境，它在单一的执行和验证接口下涵盖了预测、市场生成、实时交易和欺诈检测。OpenFinGym还提供了一个自动化的任务构建流程，能够将量化金融出版物转化为可执行的基准任务。

    arXiv:2606.26350v1 Announce Type: new  Abstract: Although large language model agents are increasingly applied to quantitative-finance workflows, their evaluation remains fragmented across isolated tasks, while the financial relevance of benchmark tasks is often overlooked. Yet financial workflows are inherently multi-stage, spanning interdependent tasks such as forecasting, strategy construction, risk management, and trading. Existing platforms typically focus on a single task, and can therefore overstate agent competence and fail to reveal weaknesses in generalization, real-market interaction, and financially meaningful decision-making. We introduce OpenFinGym, a unified gym environment for quantitative-finance agent development that covers forecasting, market generation, real-time trading, and fraud detection under a single execution and verification interface. OpenFinGym additionally provides an automated task-construction pipeline that turns quantitative finance publications into 
    
[^198]: UltraQuant：面向上下文密集型智能体的4比特KV缓存

    UltraQuant: 4-bit KV Caching for Context-Heavy Agents

    [https://arxiv.org/abs/2606.20474](https://arxiv.org/abs/2606.20474)

    该论文提出UltraQuant，一种面向多轮智能体工作负载的4比特KV缓存方案，通过TurboQuant风格旋转量化、非对称K/V处理及AMD GPU上的FP4近似推理路径，实现任务质量、缓存驻留率与服务吞吐量的联合优化。

    

    上下文密集型智能体对键值缓存施加了巨大压力：长前缀在大量短轮次中被重复使用，而并发程度决定了服务系统能否保持GPU的利用率。我们研究了适用于该场景的4比特KV缓存压缩方法，以TurboQuant风格的旋转和码本量化作为质量基准，并以vLLM FP8 KV缓存作为部署基准。我们报告了三项贡献。第一，我们围绕多轮智能体工作负载构建了4比特KV缓存的评估框架，在该场景下任务质量、缓存驻留率和服务吞吐量必须被联合测量。第二，我们描述了使4比特路径保持鲁棒所需的具体设计选择，包括非对称的K/V处理、Walsh-Hadamard旋转、移除QJL以及块尺度变体。第三，我们在AMD GPU上提出了服务优化方案，包括优化的解码注意力内核以及UltraQuant——一种采用FP8查询、FP4 KV张量和UE8M0缩放的FP4近似路径。

    arXiv:2606.20474v3 Announce Type: replace  Abstract: Context-heavy agents place substantial pressure on the key-value (KV) cache: long prefixes are reused across many short turns, while concurrency determines whether the serving system can keep GPUs utilized. We study 4-bit KV-cache compression for this setting, using TurboQuant-style rotation and codebook quantization as a quality anchor and vLLM FP8 KV caching as the deployment anchor. We report three contributions. First, we frame 4-bit KV caching around multi-round agent workloads where task quality, cache residency, and serving throughput must be measured jointly. Second, we describe the practical design choices needed to make the 4-bit path robust, including asymmetric K/V treatment, Walsh-Hadamard rotation, QJL removal, and block-scale variants. Third, we present serving optimizations on AMD GPUs, including optimized decode-attention kernels and UltraQuant, an FP4 approximation path that uses FP8 queries, FP4 KV tensors, UE8M0 g
    
[^199]: 论循环Transformer的残差缩放：稳定性与可迁移性

    On the Residual Scaling of Looped Transformers: Stability and Transferability

    [https://arxiv.org/abs/2606.18524](https://arxiv.org/abs/2606.18524)

    该论文证明循环Transformer因权重共享需要采用比传统1/√L更强的1/N残差缩放，并提出分解参数化ε = λ/(N√L)，使最优学习率仅取决于独特层数L而与循环次数N无关，从而实现超参数的直接迁移。

    

    循环Transformer（权重绑定Transformer）将一个共享的残差块应用N次（h ← h + ε·f(h)，每一步使用相同的f），在不增加参数的情况下提升有效深度。先前的深度缩放分析对深度为L的残差网络规定缩放因子ε = 1/√L。我们证明这一缩放对循环架构而言是不够的：权重共享使得残差更新在各次迭代之间产生相关性，因此需要更强的缩放ε = 1/N。对于多层块（L个独特层循环N次），我们推导出一个分解参数化ε = λ/(N√L)，将两种增长来源分离开来：1/N控制层内循环相关性，1/√L控制跨层方差。一个关键结论是，最优学习率仅取决于独特层数L，而不取决于循环次数N，这使得超参数可以直接从小N迁移到大N。

    arXiv:2606.18524v2 Announce Type: replace  Abstract: Looped (weight-tied) Transformers apply a shared residual block $N$ times ($h \leftarrow h + \varepsilon\,f(h)$, same $f$ at each step), increasing effective depth without adding parameters. Prior depth-scaling analyses prescribe $\varepsilon = 1/\!\sqrt{L}$ for depth-$L$ residual networks. We show that this is insufficient for looped architectures: weight sharing makes residual updates correlated across iterations, requiring the stronger scaling $\varepsilon = 1/N$. For multi-layer blocks ($L$ unique layers looped $N$ times), we derive a factored parameterization $\varepsilon = \lambda/(N\!\sqrt{L})$ that separates the two sources of growth: $1/N$ controls the within-layer loop correlation, and $1/\!\sqrt{L}$ controls the across-layer variance. A key consequence is that the optimal learning rate depends only on the number of unique layers $L$, not on the loop count $N$, enabling direct hyperparameter transfer from small to large $N$
    
[^200]: GRACE-DS：数据科学中的受保护奖励引导智能体纠正环境

    GRACE-DS: a Guarded Reward-guided Agent Correction Environment in Data Science

    [https://arxiv.org/abs/2606.16000](https://arxiv.org/abs/2606.16000)

    GRACE-DS是一个用于LLM驱动AutoML智能体部署前评估的隔离环境，通过隐藏的可执行验证器从预测性能、泄漏规避、可复现性等多维度进行评估，其中灵活迭代交互机制的表现优于单次生成等基线方法。

    

    我们介绍了GRACE-DS，一个面向数据科学的受保护奖励引导智能体纠正环境，用于对基于LLM的AutoML智能体进行部署前评估。GRACE-DS是一套在隔离环境中的评估指标，可应用于特定组织专属的表格机器学习任务。它让智能体经历真实的工作流程阶段，从规划和数据检查，到特征工程、模型开发、验证和代码修复，直至最终提交；同时，隐藏的可执行验证器不仅衡量最终的预测性能，还评估泄漏规避、可复现性、协议有效性、纠正行为以及奖励对齐情况。在所有结构化机制中，最强的灵活迭代交互（我们的方法）相比单次生成、非结构化交互和基于重启的基线方法，实现了更高的端到端归一化隐藏测试质量，同时还提升了协议有效的完成率。

    arXiv:2606.16000v3 Announce Type: replace  Abstract: We introduce GRACE-DS, a Guarded Reward-guided Agent Correction Environment in Data Science for pre-deployment evaluation of LLM-powered AutoML agents. GRACE-DS is a set of evaluation metrics in an isolated environment that can be applied to tabular ML tasks specific to a particular organization. It exposes agents to realistic workflow stages, from planning and data inspection through feature engineering, model development, validation, and code repair to final submission, while hidden executable validators measure not only final predictive performance but also leakage avoidance, reproducibility, protocol validity, correction behavior, and reward alignment. The strongest structured regime, flexible iterative interaction (our approach), achieves higher end-to-end normalized hidden-test quality than single-shot generation, unstructured interaction, and restart-based baselines, while also improving protocol-valid completion. Validated ac
    
[^201]: SAEExplainer：利用激活引导的偏好优化解释SAE特征

    SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization

    [https://arxiv.org/abs/2606.08496](https://arxiv.org/abs/2606.08496)

    提出SAEExplainer训练框架，以激活分数作为客观奖励信号，通过两轮迭代优化实现模型自我纠正与持续改进，显著减少SAE特征解释中的幻觉并强化因果触发模式。

    

    尽管稀疏自编码器（SAE）通过将稠密表示分解为稀疏特征，缓解了大型语言模型（LLM）的不透明性问题，但解释这些特征仍然是一个核心挑战。然而，目前的解释方法通常运行在开环范式下，未能利用机制性反馈进行进一步优化。在本文中，我们提出了SAEExplainer，这是一个利用激活分数作为客观奖励信号来训练模型进行自我纠正和迭代引导的训练框架。通过在两轮优化过程中迭代地验证和纠正基础解释，SAEExplainer实现了其解释能力的持续提升。这一机制显著减少了解释幻觉，并强化了因果触发模式。大量实验表明，我们的方法在大多数指标上优于现有基线。

    arXiv:2606.08496v2 Announce Type: replace  Abstract: Although Sparse Autoencoders (SAEs) have mitigated the opacity of large language models (LLMs) by decomposing dense representations into sparse features, explaining these features still remains a central challenge. Current explanation methods, however, typically operate within an open-loop paradigm, failing to leverage mechanistic feedback for further refinement. In this paper, we propose SAEExplainer, a training framework that utilizes activation scores as an objective reward signal to train the model for self-correction and iterative bootstrapping. By iteratively verifying and correcting foundational explanations through a two-round optimization process, SAEExplainer achieves continuous improvement in its explanatory capabilities. This mechanism significantly reduces explanation hallucinations and reinforces causal triggering patterns. Extensive experiments demonstrate our approach improves upon established baselines across most me
    
[^202]: CoHyDE：用于工具检索的LLM重写器与稠密编码器的迭代协同训练

    CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval

    [https://arxiv.org/abs/2605.29271](https://arxiv.org/abs/2605.29271)

    提出CoHyDE方法，通过迭代协同训练将LLM查询重写器与稠密编码器作为一个共同演化的整体系统，融合编码器微调与HyDE查询扩展二者的优势，显著提升LLM智能体在大型API目录上的工具检索性能。

    

    针对大型API目录的工具检索是LLM智能体的核心瓶颈：用户查询以口语化且往往不完整的语言到达，而目录使用技术性的API词汇，任何固定的编码器都无法独自弥合这一鸿沟。两种主流训练方法——对比式编码器微调和基于冻结LLM的HyDE式查询扩展——从相反的两端解决这一问题，并以互补的方式失效：微调后的编码器在查询的表面形式已与目录匹配时表现出色，但在不匹配时性能崩溃；而零样本HyDE对不完整查询更为鲁棒，却会生成不感知目录内容的假设性描述，当查询格式良好时反而会降低检索效果。我们提出CoHyDE，一种将稠密编码器和LLM重写器作为单一协同演化系统进行训练的迭代方法：编码器使用InfoNCE在生成的目录风格假设性描述上重新训练……

    arXiv:2605.29271v2 Announce Type: replace-cross  Abstract: Tool retrieval over large API catalogs is a core bottleneck for LLM agents: user queries arrive in colloquial, often underspecified language, while the catalog uses technical API vocabulary that no fixed encoder can bridge on its own. The two dominant training approaches, contrastive encoder fine-tuning and HyDE-style query expansion with a frozen LLM, address this problem from opposite ends and fail in complementary directions: the fine-tuned encoder excels when the query's surface form already matches the catalog but collapses when it does not, while zero-shot HyDE is more robust to underspecified queries yet generates catalog-unaware hypothetical descriptions that degrade retrieval when queries are well-formed. We introduce CoHyDE, an iterative procedure that trains the dense encoder and the LLM rewriter as a single co-evolving system: the encoder is retrained with InfoNCE on catalog-style hypothetical descriptions produced 
    
[^203]: 音乐注意力Transformer：基于音乐专用注意力模型的音乐生成

    Musical Attention Transformer: Music Generation Using a Music-Specific Attention Model

    [https://arxiv.org/abs/2605.21081](https://arxiv.org/abs/2605.21081)

    提出音乐注意力机制，将小节编号、调性、拍号和速度等元信息融入Transformer的注意力过程，有效减少生成音乐中的音符重复问题，提升音乐生成质量。

    

    本研究旨在通过引入元信息来提升基于Transformer的音乐生成质量。尽管基于Transformer的方法能够有效捕捉音乐作品中的长期依赖关系，但其生成的音乐常常存在音符过度重复或复制等问题，导致旋律不自然。为解决这些局限性，我们提出了音乐注意力机制，该机制将小节编号、调性、拍号和速度等元信息融入到注意力计算过程中。音乐注意力机制明确地利用了音乐的结构特性及其相关元数据，使Transformer的注意力机制能够更有效地运行，从而提高生成输出的质量。在我们的框架中，每个音符被表示为五种事件的组合——音高、小节编号、起奏时间、时值和力度。

    arXiv:2605.21081v2 Announce Type: replace-cross  Abstract: This study aims to enhance the quality of music generation using Transformers by incorporating meta-information. While Transformer-based approaches are effective at capturing long-term dependencies in musical compositions, the music they generate often suffers from issues such as excessive repetition or duplication of notes, leading to unnatural melodies. To address these limitations, we propose Musical Attention, a mechanism that incorporates meta-information such as bar numbers, key, signatures, and tempos into the attention process. Musical Attention explicitly leverages both the structural properties of music and its associated metadata, enabling the Transformer's attention mechanism to operate more effectively and thereby improving the quality of the generated output. In our framework, each musical note is represented as a combination of five events-pitch, bar number, onset, duration, and velocity in addition to the three 
    
[^204]: 局部化方法的普适理论

    The General Theory of Localization Methods

    [https://arxiv.org/abs/2605.20635](https://arxiv.org/abs/2605.20635)

    本文提出了一个以局部化核与局部均值为核心概念的通用机器学习理论框架——局部化方法，通过两个理论支柱（局部化模型构建与局部化技巧）统一解释了核方法、自注意力机制、Hopfield网络等多种现有机器学习模型背后的共同原理。

    

    本文提出了一种称为局部化方法的通用机器学习框架，其根本建立在两个核心概念之上：局部化核与局部均值——这两个关键组件正是自注意力机制的基石。为了建立严格的理论基础，该框架通过两个基本支柱被正式定义：局部（化）模型的构建与局部化技巧。我们系统地研究了局部化方法与众多现有机器学习模型/方法之间的联系，包括（但不限于）核方法、惰性学习、MeanShift算法、松弛标注、Hopfield网络、局部线性嵌入（LLE）、模糊推理以及去噪自编码器（DAE）。通过剖析这些关系，我们阐明了局部化方法更广泛的理论意义，并证明了其在多样化机器学习场景中的实际适用性。

    arXiv:2605.20635v4 Announce Type: replace  Abstract: This paper proposes a general machine learning framework called the localization method, which is fundamentally built on two core concepts: localization kernels and local means -- key components that underpin the self-attention mechanism. To establish a rigorous theoretical foundation, the framework is formally defined through two essential pillars: the formulation of the local(-ized) model and the localization trick. We systematically investigate the connections between the localization method and a wide range of existing machine learning models/methods, including (but not limited to) kernel methods, lazy learning, the MeanShift algorithm, relaxation labeling, Hopfield networks, local linear embedding (LLE), fuzzy inference, and denoising autoencoders (DAEs). By dissecting these relationships, we clarify the broader theoretical significance of the localization method and demonstrate its practical applicability across diverse machine
    
[^205]: 基于Logit最大化的联邦学习类别级贡献度估计

    Class-wise Contribution Estimation via Logit Maximization for Federated Learning

    [https://arxiv.org/abs/2605.18892](https://arxiv.org/abs/2605.18892)

    提出了一种基于logit最大化的免数据联邦学习聚合框架CELM，通过构建跨客户端证据矩阵量化各类别能力与覆盖度，为少数类提供强证据的客户端赋予更高聚合权重，从而有效缓解联邦学习中的类别不平衡和标签偏斜问题。

    

    联邦学习（FL）使计算机视觉模型能够在隐私和法规限制阻止跨设备或组织集中数据的场景下进行协作学习。然而，实际的联邦学习部署中常常存在严重的类别不平衡和标签偏斜问题，导致标准聚合协议过拟合于主导客户端，并降低少数类别的性能。我们提出了一种基于logit最大化的免数据、类别级贡献度估计与聚合框架（CELM），该框架无需共享原始数据、客户端元数据或辅助公共数据集。联邦学习服务器通过探测客户端更新来获取类别级证据分数，并构建跨客户端证据矩阵，该矩阵同时量化了每类别的处理能力和类别覆盖范围。利用该矩阵，我们计算贡献权重，对那些为代表性不足类别提供强有力、有区分度证据的客户端赋予更高的权重。由此产生的聚合是稳定的……

    arXiv:2605.18892v2 Announce Type: replace  Abstract: Federated learning (FL) enables collaborative learning of computer vision models, where privacy and regulatory constraints prevent centralizing data across devices or organizations. However, practical FL deployments often exhibit severe class imbalance and label skew, causing standard aggregation protocols to overfit dominant clients and degrade minority-class performance. We propose a data-free, class-wise contribution estimation and aggregation framework based on logit maximization (CELM) that does not require sharing raw data, client metadata, or auxiliary public datasets. The FL server probes client updates to obtain class-wise evidence scores and assembles a cross-client evidence matrix, which quantifies both per-class competence and class coverage. Using this matrix, we compute contribution weights that upweight clients providing strong, discriminative evidence for underrepresented classes. The resulting aggregation is stable d
    
[^206]: DiffusionOPD：扩散模型中在线策略蒸馏的统一视角

    DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models

    [https://arxiv.org/abs/2605.15055](https://arxiv.org/abs/2605.15055)

    提出基于在线策略蒸馏（OPD）的 DiffusionOPD 多任务训练范式，先独立训练任务教师，再沿学生自身 rollout 轨迹将其能力蒸馏到统一学生模型，从而解决扩散模型多任务强化学习中的跨任务干扰与灾难性遗忘问题。

    

    强化学习已成为改进基于扩散的文本到图像模型的强大工具，但现有方法大多局限于单任务优化。将强化学习扩展到多任务场景充满挑战：联合优化会遭受跨任务干扰与任务不平衡的影响，而级联强化学习则流程繁琐且容易发生灾难性遗忘。我们提出了 DiffusionOPD，一种基于在线策略蒸馏（Online Policy Distillation, OPD）的扩散模型新型多任务训练范式。DiffusionOPD 首先独立训练各任务专属的教师模型，然后沿着学生模型自身的 rollout 轨迹将教师的能力蒸馏到一个统一的学生模型中。这种方式将单任务探索与多任务集成解耦，避免了从头开始联合求解所有任务的优化负担。在理论上，我们将 OPD 框架从离散 token 扩展到连续状态的马尔可夫过程，推导出每一步的闭式 KL 目标函数……

    arXiv:2605.15055v2 Announce Type: replace  Abstract: Reinforcement learning has emerged as a powerful tool for improving diffusion-based text-to-image models, but existing methods are largely limited to single-task optimization. Extending RL to multiple tasks is challenging: joint optimization suffers from cross-task interference and imbalance, while cascade RL is cumbersome and prone to catastrophic forgetting. We propose DiffusionOPD, a new multi-task training paradigm for diffusion models based on Online Policy Distillation (OPD). DiffusionOPD first trains task-specific teachers independently, then distills their capabilities into a unified student along the student own rollout trajectories. This decouples single-task exploration from multi-task integration and avoids the optimization burden of solving all tasks jointly from scratch. Theoretically, we lift the OPD framework from discrete tokens to continuous-state Markov processes, deriving a closed-form per-step KL objective that u
    
[^207]: DenseTRF：面向手术场景密集预测的纹理感知无监督表征自适应

    DenseTRF: Texture-Aware Unsupervised Representation Adaptation for Surgical Scene Dense Prediction

    [https://arxiv.org/abs/2605.11265](https://arxiv.org/abs/2605.11265)

    DenseTRF提出了一种基于纹理中心注意力的自监督表征自适应框架，通过槽注意力学习纹理感知表征并无监督地适应目标分布，显著提升了手术场景密集预测任务对域偏移的鲁棒性和跨分布泛化能力。

    

    手术计算机视觉中的密集预测任务，如分割和手术区域预测，可以为腹腔镜和机器人手术提供有价值的指导。然而，这些模型经常受到分布偏移的影响，因为训练数据集很少涵盖部署过程中遇到的各种变化，导致泛化能力较差。我们提出了DenseTRF，一个基于以纹理为中心注意力的自监督表征自适应框架。我们的方法利用槽注意力学习纹理感知的表征，捕获不变的视觉结构。通过将这些表征无监督地适应到目标分布，DenseTRF显著提高了对域偏移的鲁棒性。该框架通过将密集预测条件化于槽注意力并结合模型合并策略来实现。在多种手术过程中的实验表明，该方法的跨分布泛化能力得到了提升。

    arXiv:2605.11265v2 Announce Type: replace-cross  Abstract: Dense prediction tasks in surgical computer vision, such as segmentation and surgical zone prediction, can provide valuable guidance for laparoscopic and robotic surgery. However, these models often suffer from distribution shifts, as training datasets rarely cover the variability encountered during deployment, leading to poor generalization. We propose DenseTRF, a self-supervised representation adaptation framework based on texture-centric attention. Our method leverages slot attention to learn texture-aware representations that capture invariant visual structures. By adapting these representations to the target distribution without supervision, DenseTRF significantly improves robustness to domain shifts. The framework is implemented through conditioning dense prediction on slot attention and model merging strategies. Experiments across multiple surgical procedures demonstrate improved cross-distribution generalization in comp
    
[^208]: 具有解耦动力学的部分可观测马尔可夫势博弈中纳什均衡的独立学习

    Independent Learning of Nash Equilibria in Partially Observable Markov Potential Games with Decoupled Dynamics

    [https://arxiv.org/abs/2605.06377](https://arxiv.org/abs/2605.06377)

    该论文针对具有独立状态转移的部分可观测马尔可夫势博弈，提出了一种无需中心化或通信的独立学习算法，使各智能体仅凭自身动作和观测即可联合收敛到近似纳什均衡，从而避免了先前方法随玩家数量指数级增长的复杂度。

    

    我们研究部分可观测马尔可夫博弈（POMGs）中的纳什均衡学习，这是一种多智能体强化学习框架，其中智能体无法完全观测底层状态。在此设置下的先前工作依赖于中心化或信息共享，且样本和计算复杂度随玩家数量呈指数级增长。我们关注具有独立状态转移的POMGs子类，其中智能体仅通过奖励保持耦合，并假设底层完全可观测的马尔可夫博弈是马尔可夫势博弈。针对此类博弈，我们提出了一种独立学习算法，其中玩家仅观察自身的动作和观测、无需任何通信，即可联合收敛到近似纳什均衡。由于部分可观测性，最优策略通常可能依赖于完整的动作-观测历史。在滤波稳定性假设下，我们表明……

    arXiv:2605.06377v2 Announce Type: replace-cross  Abstract: We study Nash equilibrium learning in partially observable Markov games (POMGs), a multi-agent reinforcement learning framework in which agents cannot fully observe the underlying state. Prior work in this setting relies on centralization or information sharing, and suffers from sample and computational complexity that scales exponentially in the number of players. We focus on a subclass of POMGs with independent state transitions, where agents remain coupled through their rewards, and assume that the underlying fully observed Markov game is a Markov potential game. For this class, we present an independent learning algorithm in which players, observing only their own actions and observations and without communication, jointly converge to an approximate Nash equilibrium. Due to partial observability, optimal policies may in general depend on the full action-observation history. Under a filter stability assumption, we show that 
    
[^209]: 适应还是遗忘：非平稳优化中Adam与SGD之间的可证明权衡

    Adapt or Forget: Provable Tradeoffs Between Adam and SGD in Nonstationary Optimization

    [https://arxiv.org/abs/2605.04269](https://arxiv.org/abs/2605.04269)

    本文首次对Adam在非平稳随机优化中的表现给出理论分析，将有限时间界清晰分解为初始化、目标漂移、一阶矩跟踪误差和预条件器扰动四个分量，并揭示了噪声与漂移之间可证明的权衡关系。

    

    我们在非平稳随机目标下对Adam进行了理论分析，区分了两种情形：一是在Adam预条件化平均梯度算子具有自适应强单调性下的欧氏跟踪情形，二是在一般 $L$-光滑目标下的高概率投影平稳性保证。在跟踪情形中，我们推导了有限时间的期望界和高概率界，这些界可以清晰地分解为四个分量：初始化、目标漂移、由 $\beta_1$ 控制的一阶矩跟踪误差，以及由 $\beta_2$ 控制的预条件器扰动。我们刻画了在常数步长和步长衰减调度下，瞬态项衰减到渐近跟踪界所需的预热时间。我们还证明了Adam在分布偏移下平均投影平稳性间隙的高概率界。在两种分析中，我们的界均揭示了一种噪声—漂移权衡：在噪声主导的情形中，一阶矩……（原文摘要在此处截断）

    arXiv:2605.04269v2 Announce Type: replace-cross  Abstract: We provide a theoretical analysis of Adam under non-stationary stochastic objectives, separating two regimes: Euclidean tracking under adaptive strong monotonicity of the Adam-preconditioned mean-gradient operator, and high-probability projected stationarity guarantees under general $L$-smooth objectives. In the tracking regime, we derive finite-time expected and high-probability bounds that decompose sharply into four components: initialization, objective drift, a first-moment tracking error governed by $\beta_1$, and a preconditioner perturbation governed by $\beta_2$. We characterize the burn-in time required for the transient terms to decay to the asymptotic tracking bound under constant and step-decay schedules. We also prove a high-probability bound on the average projected stationarity gap for Adam under distribution shift. Across both analyses, our bounds reveal a noise--drift tradeoff: in noise-dominated regimes, first
    
[^210]: 异常偏好图像生成

    Anomaly-Preference Image Generation

    [https://arxiv.org/abs/2605.02439](https://arxiv.org/abs/2605.02439)

    本文提出异常偏好优化范式，通过利用真实异常作为正样本参考的隐式偏好对齐机制和沿扩散时间线动态分配模型容量的时间感知容量分配模块，在不依赖人工标注的情况下，有效平衡了有限数据下异常样本生成中的保真度与多样性。

    

    从有限数据中合成真实且多样的异常样本，对于实现鲁棒的模型泛化至关重要。然而，现有方法难以在保真度和多样性之间取得平衡，分别受制于分布不对齐和过拟合问题。为缓解这一问题，我们提出了异常偏好优化，这是一种将异常生成重新表述为偏好学习问题的新范式。我们方法的核心是一种隐式偏好对齐机制，该机制利用真实异常作为正样本参考，直接从去噪轨迹偏差中推导优化信号，而无需昂贵的人工标注。此外，我们提出了一个时间感知容量分配模块，该模块沿扩散时间线动态分配模型容量，在高噪声阶段优先考虑结构多样性，同时在低噪声阶段增强细粒度保真度。在推理过程中，一种分层采样策略……

    arXiv:2605.02439v4 Announce Type: replace-cross  Abstract: Synthesizing realistic and diverse anomalous samples from limited data is vital for robust model generalization. However, existing methods struggle to reconcile fidelity and diversity, often hampered by distribution misalignment and overfitting, respectively.To mitigate this, we introduce Anomaly Preference Optimization,a novel paradigm that reformulates anomaly generation as a preference learning problem.Central to our approach is an implicit preference alignment mechanism that leverages real anomalies as positive references, deriving optimization signals directly from denoising trajectory deviations without requiring costly human annotation. Furthermore, we propose a Time-Aware Capacity Allocation module that dynamically distributes model capacity along the diffusion timeline,prioritizing structural diversity during highnoise phases while enhancing fine-grained fidelity in low-noise stages. During inference, a hierarchical sa
    
[^211]: 通过FINALES与Kadi4Mat之间的可互操作接口加速电池研究

    Accelerating battery research with an interoperable interface between FINALES and Kadi4Mat

    [https://arxiv.org/abs/2605.00909](https://arxiv.org/abs/2605.00909)

    本研究提出了一个集成FINALES与Kadi数据管理生态系统的可互操作框架，实现了跨分布式研究基础设施的自动化、可重现的端到端电池实验工作流程，并以钠离子电池化成工艺为案例展示了其加速能源材料研究的能力。

    

    本研究以电池化成工艺为案例，探讨了自动化且可互操作的研究基础设施如何加速实验性材料研究。我们提出了一种方法学框架，将FINALES和Kadi研究数据管理生态系统集成在一起，实现了跨分布式研究基础设施的协调实验执行、数据管理和分析。FINALES框架在POLiS材料加速平台上负责实验规划与执行的调度，而Kadi则负责数据管理、可视化、实验选择和持久化存储。这种互操作性实现了可重现、协调、端到端的实验工作流程，连接了多个研究地点的自动化系统和人工操作流程。为展示该框架的能力，我们研究了钠离子纽扣电池的化成过程——这是影响电池性能的关键步骤……

    arXiv:2605.00909v2 Announce Type: replace-cross  Abstract: This study investigates how automated and interoperable research infrastructures can accelerate experimental materials research, using battery formation as a case study. We introduce a methodological framework that integrates the FINALES and Kadi Research Data Management ecosystems, enabling coordinated experiment execution, data management, and analysis across distributed research infrastructures. The FINALES framework orchestrates experiment planning and execution on the POLiS Materials Acceleration Platform, while Kadi handles data management, visualization, experiment selection, and persistent storage. This interoperability enables reproducible, coordinated, end-to-end experimental workflows that connect automated systems and human-operated processes across multiple research sites. To showcase the framework's capabilities, we investigate the formation process of sodium-ion coin cells, a critical step that influences cell li
    
[^212]: 训练前的表示：生成式医疗事件模型分词的实用基准

    Representation Before Training: A Practical Benchmark for Generative Medical Event Model Tokenization

    [https://arxiv.org/abs/2604.16775](https://arxiv.org/abs/2604.16775)

    该研究对生成式医疗事件模型的分词方案进行了大规模系统性基准测试（156个模型），发现将医疗代码与检验数值十分位数融合的 token 表示在所有八类住院结局预测任务中均能带来性能提升。

    

    生成式医疗事件模型使用分词后的患者时间线序列作为输入，但围绕分词的众多决策目前缺乏实践指导。我们对多种分词方案进行了基准测试，包括量化粒度、参考范围锚定、代码-数值融合、数值与时间编码，以及来自专家映射的通用数据模型的原生表示与协调表示。我们采用 Llama 和 Qwen 两种架构，从三个初始化种子共训练了156个模型，每种配置遵循相同的训练方案，最多训练五个 epoch。我们使用线性探针对住院前24小时内学习到的表示进行评估，以预测第24-48小时期间的二分类和连续型结局。结果表明，将医疗代码与数值十分位数配对的融合 token，在所有八类结局预测任务中均优于对应的非融合分词输入，其 AUROC 提升为 +（摘要在此处截断）

    arXiv:2604.16775v2 Announce Type: replace  Abstract: Generative medical event models use tokenized sequences of patient timelines as input, but practical guidance on the many decisions around tokenization is limited. We benchmark quantization granularity, reference-range anchoring, code--value fusion, numeric and temporal encodings, and native versus harmonized event representations from an expert-mapped common data model. Using both Llama and Qwen architectures, 156 models were trained from three initialization seeds, with each configuration following a shared training recipe for up to five epochs. We evaluated learned representations from the first 24 hours of hospitalization with linear probes to predict binary and continuous outcomes during hours 24-48. Fused tokens pairing codes with value deciles increased performance across all eight outcome families relative to the equivalent unfused tokenized input with area under the receiver operating characteristic curve (AUROC) gains of $+
    
[^213]: SaFeR-Steer：通过合成引导与反馈动态演化多轮多模态大语言模型

    SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics

    [https://arxiv.org/abs/2604.16358](https://arxiv.org/abs/2604.16358)

    提出渐进式多轮安全对齐框架SaFeR-Steer，通过分阶段合成数据引导与导师介入的GRPO强化学习，并引入轨迹一致求和奖励（TCSR）机制，同时发布多轮多模态安全数据集STEER，以弥合多轮对话场景下多模态大模型训练与部署之间的安全对齐差距。

    

    多模态大语言模型（MLLM）越来越多地被部署在多轮对话场景中，攻击者可以通过不断演化的视觉-文本历史逐步升级不安全意图，并利用长上下文安全衰减的漏洞。然而，安全对齐目前仍主要依赖单轮数据和固定模板对话，导致训练与部署之间存在不匹配。为弥合这一差距，我们提出了SaFeR-Steer，这是一个渐进式多轮对齐框架，将分阶段合成引导与导师介入的GRPO相结合，在自适应的、在线策略攻击下训练单一学生模型。我们还引入了轨迹一致求和奖励，该机制聚合各轮次奖励的历史最小值和平均值，使任何低质量轮次都会影响轨迹层面的整体回报。I. 数据集：我们发布了STEER，一个多轮多模态安全数据集，包含STEER-SFT（12,934条）、STEER-RL（2,000条）和STEER-Bench（3,227条）对话，覆盖1-10轮对话。II. 实验：从Q（注：原始摘要在此处不完整）

    arXiv:2604.16358v3 Announce Type: replace-cross  Abstract: MLLMs are increasingly deployed in multi-turn settings, where attackers can escalate unsafe intent through the evolving visual-text history and exploit long-context safety decay. Yet safety alignment is still dominated by single-turn data and fixed-template dialogues, leaving a mismatch between training and deployment. To bridge this gap, we propose SaFeR-Steer, a progressive multi-turn alignment framework that combines staged synthetic bootstrapping with tutor-in-the-loop GRPO to train a single student under adaptive, on-policy attacks. We also introduce Trajectory-Consistent Summative Reward (TCSR), which aggregates the historical minimum and average of turn rewards so that any low-quality turn affects the trajectory-level return. I. Dataset. We release STEER, a multi-turn multimodal safety dataset with STEER-SFT (12,934), STEER-RL (2,000), and STEER-Bench (3,227) dialogues spanning 1-10 turns. II. Experiment. Starting from Q
    
[^214]: 基于模式挖掘与无监督学习的城市土地利用模式探索

    Exploring Urban Land Use Patterns by Pattern Mining and Unsupervised Learning

    [https://arxiv.org/abs/2604.13050](https://arxiv.org/abs/2604.13050)

    该论文提出了一种结合频繁项集模式挖掘与Ward聚类的可重复分析框架，从欧洲100个城市地区的遥感数据中识别出七类跨城市重复出现的土地利用配置模式，为比较性规划中的同类城市对比提供了方法支持。

    

    比较性规划需要可重复的方法来识别跨城市重复出现的土地利用配置。本研究利用2018年城市地图集（Urban Atlas 2018）中100个欧洲城市地区的数据，构建了290,396个焦点邻域事务，并在10%最低支持度下提取了1,543个频繁项集支持度特征。研究在原始归一化特征空间中应用Ward聚类，UMAP仅用于可视化。通过多准则评估和500次配对特征子采样重复实验，最终保留了七聚类的描述性方案。敏感性分析表明，在5%–15%的阈值范围内，城市相似性几何结构保持稳定，但聚类成员归属对特征表示方式的选择以及非人工土地利用内容较为敏感。该框架支持同类城市间的比较，同时明确了尺度与边界方面的局限性。

    arXiv:2604.13050v2 Announce Type: replace-cross  Abstract: Comparative planning needs reproducible methods for identifying recurring land-use configurations across cities. Using Urban Atlas 2018 data for 100 European urban areas, we construct 290,396 focal-neighborhood transactions and 1,543 frequent-itemset support features at 10\% minimum support. Ward clustering is applied in the original normalized feature space, with UMAP used only for visualization. A seven-cluster descriptive solution is retained through multi-criterion evaluation and 500 paired feature-subsampling repetitions. Sensitivity analyses show stable city-similarity geometry across 5--15\% thresholds but greater membership sensitivity to representation choices and non-artificial land-use content. The framework supports peer-city comparison while making scale and boundary limitations explicit.
    
[^215]: 独立估计的视图不确定性是否具有可比性？可信多视图分类的统一路由方法

    Are Independently Estimated View Uncertainties Comparable? Unified Routing for Trusted Multi-View Classification

    [https://arxiv.org/abs/2604.09288](https://arxiv.org/abs/2604.09288)

    该论文提出TMUR方法，解决了可信多视图分类中各视图独立估计的不确定性因分支尺度偏差而不可比的问题，通过统一路由机制解耦视图证据提取，使融合所用的不确定性真正反映样本级可靠性。

    

    可信多视图分类通常依赖于逐视图的证据融合过程：每个视图独立产生类别证据和不确定性，最终预测通过聚合这些独立意见获得。虽然这种设计具有模块化和不确定性感知的优点，但它隐含地假设来自不同视图的证据在数值上是可比的。然而在实践中，这一假设是脆弱的。不同视图通常在特征空间、噪声水平和语义粒度上存在差异，而独立训练的分支仅针对预测正确性进行优化，没有任何约束来强制执行证据强度的跨视图一致性。因此，用于融合的不确定性可能被分支特定的尺度偏差而非真正的样本级可靠性所主导。为了解决这个问题，我们提出了具有统一路由的可信多视图学习方法（TMUR），该方法将视图特定的证据提取与……（摘要不完整，原文在此处截断）

    arXiv:2604.09288v2 Announce Type: replace  Abstract: Trusted multi-view classification typically relies on a view-wise evidential fusion process: each view independently produces class evidence and uncertainty, and the final prediction is obtained by aggregating these independent opinions. While this design is modular and uncertainty-aware, it implicitly assumes that evidence from different views is numerically comparable. In practice, however, this assumption is fragile. Different views often differ in feature space, noise level, and semantic granularity, while independently trained branches are optimized only for prediction correctness, without any constraint enforcing cross-view consistency in evidence strength. As a result, the uncertainty used for fusion can be dominated by branch-specific scale bias rather than true sample-level reliability. To address this issue, we propose Trusted Multi-view learning with Unified Routing (TMUR), which decouples view-specific evidence extraction
    
[^216]: 死权重，活信号：冻结语言模型的前馈图

    Dead Weights, Live Signals: Feedforward Graphs of Frozen Language Models

    [https://arxiv.org/abs/2604.08335](https://arxiv.org/abs/2604.08335)

    该论文提出一种前馈图架构，将多个异构冻结大语言模型作为计算节点，通过可学习的线性投影在共享潜在空间中通信，仅用1760万可训练参数即在ARC-Challenge上达到87.3%的准确率。

    

    我们提出了一种前馈图架构，其中异构的冻结大语言模型作为计算节点，通过学习到的线性投影在共享的连续潜在空间中进行通信。基于近期证明独立训练的LLM潜在空间之间具有几何兼容性的工作，我们将这一发现从静态的双模型引导扩展到端到端可训练的多节点图，其中投影矩阵通过残差流注入钩子的反向传播进行联合优化。三个小型冻结模型（Llama-3.2-1B、Qwen2.5-1.5B、Gemma-2-2B）将输入编码到共享潜在空间中，其聚合信号被注入两个更大的冻结模型（Phi-3-mini、Mistral-7B），这些模型的表征再输入到轻量级的交叉注意力输出节点。在仅1760万可训练参数（对比约120亿冻结参数）的情况下，该架构在ARC-Challenge上达到了87.3%的成绩。

    arXiv:2604.08335v2 Announce Type: replace  Abstract: We present a feedforward graph architecture in which heterogeneous frozen large language models serve as computational nodes, communicating through a shared continuous latent space via learned linear projections. Building on recent work demonstrating geometric compatibility between independently trained LLM latent spaces~\cite{armstrong2026thinking}, we extend this finding from static two-model steering to end-to-end trainable multi-node graphs, where projection matrices are optimized jointly via backpropagation through residual stream injection hooks. Three small frozen models (Llama-3.2-1B, Qwen2.5-1.5B, Gemma-2-2B) encode the input into a shared latent space whose aggregate signal is injected into two larger frozen models (Phi-3-mini, Mistral-7B), whose representations feed a lightweight cross-attention output node. With only 17.6M trainable parameters against approximately 12B frozen, the architecture achieves 87.3\% on ARC-Chall
    
[^217]: 一种经验性马尔可夫链跟驰（MC-CF）模型

    An Empirical Markov Chain Car-Following (MC-CF) Model

    [https://arxiv.org/abs/2603.27909](https://arxiv.org/abs/2603.27909)

    本文提出了一种无参数化假设的马尔可夫链跟驰模型（MC-CF），通过从经验分布中随机采样加速度来捕捉自然驾驶的随机性，在Waymo数据集上显著优于传统物理模型并与数据驱动方法相当。

    

    跟驰行为是交通流理论的基础，然而传统模型往往无法捕捉自然驾驶的随机性。本文提出了一种基于经验概率采样的跟驰建模方法，绕过了传统的参数化假设。在该方法下，我们提出了马尔可夫链跟驰（MC-CF）模型，该模型将状态转移表示为马尔可夫过程，并通过从离散化状态区间内的经验分布中随机采样加速度来预测驾驶行为。在Waymo开放运动数据集（WOMD）上的评估表明，MC-CF变体在单步和开环轨迹预测指标上均显著优于所有基于物理的基线模型（IDM、Gipps、FVDM和SIDM），并且与包括神经网络和高斯混合模型在内的现代数据驱动基线方法相比也具有竞争力。在自然驾驶数据上的零样本泛化……

    arXiv:2603.27909v2 Announce Type: replace-cross  Abstract: Car-following behavior is fundamental to traffic flow theory, yet traditional models often fail to capture the stochasticity of naturalistic driving. This paper proposes an empirical probabilistic sampling approach to car-following modeling that bypasses conventional parametric assumptions. Under this approach, we introduce the Markov Chain Car-Following (MC-CF) model, which represents state transitions as a Markov process and predicts behavior by randomly sampling accelerations from empirical distributions within discretized state bins. Evaluation on the Waymo Open Motion Dataset (WOMD) demonstrates that MC-CF variants significantly outperform all physics-based baselines (IDM, Gipps, FVDM, and SIDM) across both one-step and open-loop trajectory prediction metrics, and remain competitive with modern data-driven baselines including neural network and Gaussian mixture model approaches. Zero-shot generalization on the Naturalistic
    
[^218]: 分解歧视：面向AI驱动信贷决策的因果中介分析

    Decomposing Discrimination: Causal Mediation Analysis for AI-Driven Credit Decisions

    [https://arxiv.org/abs/2603.27510](https://arxiv.org/abs/2603.27510)

    该论文提出将AI信贷决策中的歧视通过因果中介分析分解为直接歧视与结构性不平等两种机制，并在处理诱导混淆下识别出干预性直接/间接效应，为不可识别的自然歧视效应提供保守界。

    

    AI驱动的信贷决策中的统计公平性指标混淆了两种因果上不同的机制：从受保护属性直接作用于信贷结果的歧视，以及通过合法金融特征传导的结构性不平等。我们运用Pearl的自然直接效应与自然间接效应框架，在信贷决策场景中形式化了这一区别。我们的主要理论贡献是一种在处理诱导混淆下识别自然直接效应和间接效应的策略——这是受保护属性同时因果地影响金融中介变量与最终决策的普遍情形，它违反了标准的序贯可忽略性假设。我们证明，干预性直接效应与间接效应（IDE/IIE）在较弱的修正序贯可忽略性假设下是可识别的，并证明在单调性条件下，IDE/IIE为不可识别的自然效应提供了保守界。

    arXiv:2603.27510v2 Announce Type: replace  Abstract: Statistical fairness metrics in AI-driven credit decisions conflate two causally distinct mechanisms: discrimination operating directly from a protected attribute to a credit outcome, and structural inequality propagating through legitimate financial features. We formalise this distinction using Pearl's framework of natural direct and indirect effects applied to the credit decision setting. Our primary theoretical contribution is an identification strategy for natural direct and indirect effects under treatment-induced confounding -- the prevalent setting in which protected attributes causally affect both financial mediators and the final decision, violating standard sequential ignorability. We show that interventional direct and indirect effects (IDE/IIE) are identified under the weaker Modified Sequential Ignorability assumption, and prove that IDE/IIE provide conservative bounds on the unidentified natural effects under monotone i
    
[^219]: FEAT：面向超大规模结构化数据的线性复杂度基础模型

    FEAT: A Linear-Complexity Foundation Model for Extremely Large Structured Data

    [https://arxiv.org/abs/2603.16513](https://arxiv.org/abs/2603.16513)

    提出了FEAT——一种面向超大规模结构化数据的线性复杂度基础模型，它克服了自注意力的O(N^2)计算瓶颈，在保持置换不变性的同时提升了对真实世界数据库的泛化能力。

    

    结构化数据广泛应用于医疗、金融和科学数据管理等领域。近来针对结构化数据基础模型（SFMs）的研究旨在支持对这类数据的分析与挖掘任务，但在应用于真实世界企业数据库时仍面临可扩展性和泛化方面的挑战。首先，许多SFMs依赖完整的自注意力机制，这引入了O(N^2)的计算瓶颈，限制了可联合处理的元组数量。其次，直接用线性复杂度序列模型替换注意力机制可能与结构化数据的置换不变性质相冲突，引入人为的顺序偏差并降低表示质量。此外，仅在合成数据上训练的模型可能难以泛化到真实世界数据库中常见的重尾和异构分布。为应对这些挑战，我们提出了FEAT，一种线性复杂度的基础模型……（摘要原文不完整，后续内容缺失）

    arXiv:2603.16513v4 Announce Type: replace  Abstract: Structured data is widely used in domains such as healthcare, finance, and scientific data management. Recent studies on structured data foundation models (SFMs) aim to support data analysis and mining tasks over such data, but still face scalability and generalization challenges when applied to real-world enterprise databases. First, many SFMs rely on full self-attention, which introduces an O(N^2) computational bottleneck and limits the number of tuples that can be processed jointly. Second, directly replacing attention with linear-complexity sequence models may conflict with the permutation-invariant nature of structured data, introducing artificial order bias and degrading representation quality. Moreover, models trained only on synthetic data may struggle to generalize to the heavy-tailed and heterogeneous distributions commonly found in real-world databases. To address these challenges, we propose FEAT, a linear-complexity foun
    
[^220]: 用于压缩感知和逆问题的可调潜变量生成先验

    Tunable Latent Generative Priors for Compressed Sensing and Inverse Problems

    [https://arxiv.org/abs/2603.07357](https://arxiv.org/abs/2603.07357)

    本文提出基于嵌套丢弃技术的可调潜变量生成先验，使扩散模型、归一化流和变分自编码器能够灵活调整潜变量维度，在压缩感知、图像修复、去噪和相位恢复等逆问题中持续获得比固定复杂度基线更低的重构误差，并在线性去噪场景下闭式推导出最优复杂度。

    

    潜变量生成模型已成为解决逆问题的强大先验。这些模型通常在由潜变量维度决定的单一固定复杂度下表示一类自然信号，这可能带来局限性：取决于具体问题，潜变量维度过小可能导致较高的表示误差，而维度过大则可能对噪声过拟合。我们利用嵌套丢弃技术，为扩散模型、归一化流和变分自编码器开发了可调的潜变量先验。在压缩感知、图像修复、去噪和相位恢复等任务中，我们通过实验证明，可调先验始终比固定复杂度的基线方法取得更低的重构误差。在线性去噪设置中，我们以闭式形式推导出最优复杂度，展示了其如何依赖于噪声水平和信号频谱。这项工作展示了可调潜变量生成先验的潜力

    arXiv:2603.07357v3 Announce Type: replace  Abstract: Latent generative models have emerged as powerful priors for solving inverse problems. These models typically represent a class of natural signals at a single, fixed complexity, governed by the latent dimensionality. This can be limiting: depending on the problem, a latent dimensionality that is too small may result in high representation error, while one that is too large may overfit to noise. We develop tunable latent priors for diffusion models, normalizing flows, and variational autoencoders, leveraging nested dropout. Across tasks including compressed sensing, inpainting, denoising, and phase retrieval, we show empirically that tunable priors consistently achieve lower reconstruction errors than fixed-complexity baselines. In the linear denoising setting, we derive the optimal complexity in closed form, showing how it depends on the noise level and the signal spectrum. This work demonstrates the potential of tunable latent gener
    
[^221]: Countdown-Code：用于研究RLVR中奖励破解的出现与泛化的测试平台

    Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR

    [https://arxiv.org/abs/2603.07084](https://arxiv.org/abs/2603.07084)

    本文提出Countdown-Code测试平台，通过代理奖励与真实奖励的清晰分离来精确测量RLVR中的奖励破解现象，并发现SFT数据中仅1%的奖励破解轨迹污染就足以让模型无意中习得这种失对齐行为。

    

    奖励破解是模型对齐失效的一种形式，即模型过度优化代理奖励而没有真正解决底层任务。精确测量奖励破解的发生仍然具有挑战性，因为真实的任务奖励往往计算成本高昂或无法计算。我们提出了Countdown-Code，这是一个极简环境，模型在其中既可以解决数学推理任务，也可以操纵测试工具。这种双重访问设计在代理奖励（测试通过/失败）与真实奖励（数学正确性）之间建立了清晰的分离，从而能够准确测量奖励破解的发生率。利用该环境，我们研究了开源权重大语言模型中的奖励破解行为，发现当即使只有一小部分奖励破解轨迹泄漏到训练数据中时，模型也会在监督微调（SFT）过程中无意中习得此类行为。在蒸馏SFT数据中仅需1%的污染，就足以使模型内化……

    arXiv:2603.07084v3 Announce Type: replace-cross  Abstract: Reward hacking is a form of misalignment in which models overoptimize proxy rewards without genuinely solving the underlying task. Precisely measuring reward hacking occurrence remains challenging because true task rewards are often expensive or impossible to compute. We introduce Countdown-Code, a minimal environment where models can both solve a mathematical reasoning task and manipulate the test harness. This dual-access design creates a clean separation between proxy rewards (test pass/fail) and true rewards (mathematical correctness), enabling accurate measurement of reward-hacking rates. Using this environment, we study reward hacking in open-weight LLMs and find that such behaviors can be unintentionally learned during supervised fine-tuning (SFT) when even a small fraction of reward-hacking trajectories leak into training data. As little as 1\% contamination in distillation SFT data is sufficient for models to internali
    
[^222]: 维也纳4G/5G路测数据集

    The Vienna 4G/5G Drive-Test Dataset

    [https://arxiv.org/abs/2603.02638](https://arxiv.org/abs/2603.02638)

    本文发布了维也纳4G/5G路测数据集，这是一个结合LTE/5G NR测量数据、基站部署描述信息以及高分辨率建筑物与地形模型的城市规模开放数据集，为移动网络分析、规划与优化的机器学习研究提供数据支持。

    

    arXiv:2603.02638v2 公告类型：replace-cross 摘要：面向移动网络分析、规划与优化的机器学习研究常常受限于缺乏大规模、全面的现实世界数据集。本文介绍了维也纳4G/5G路测数据集，这是一个城市规模的开放数据集，包含在奥地利维也纳全域采集的带地理参考信息的长期演进（LTE）和5G新空口（NR）测量数据。该数据集将无源宽带扫描仪观测数据与有源手机终端日志相结合，为已部署的无线接入网络提供了网络侧和用户侧互补的视角。测量覆盖了多样化的城市和郊区环境，并与时间和位置信息对齐，以支持一致的评估。对于具有代表性的基站（BS）子集，我们提供了推断的部署描述信息，包括估计的基站位置、扇区方位角和天线高度。该发布版本还包含高分辨率的建筑物和地形模型，从而支持基于几何条件的学习……

    arXiv:2603.02638v2 Announce Type: replace-cross  Abstract: Machine learning for mobile network analysis, planning, and optimization is often limited by the lack of large, comprehensive real-world datasets. This paper introduces the Vienna 4G/5G Drive-Test Dataset, a city-scale open dataset of georeferenced Long Term Evolution (LTE) and 5G New Radio (NR) measurements collected across Vienna, Austria. The dataset combines passive wideband scanner observations with active handset logs, providing complementary network-side and user-side views of deployed radio access networks. The measurements cover diverse urban and suburban settings and are aligned with time and location information to support consistent evaluation. For a representative subset of base stations (BSs), we provide inferred deployment descriptors, including estimated BS locations, sector azimuths, and antenna heights. The release further includes high-resolution building and terrain models, enabling geometry-conditioned lear
    
[^223]: 单帧菲涅尔相干衍射成像与重叠叠层成像的统一自监督框架

    A unified self-supervised framework for single-frame Fresnel CDI and overlapped ptychography

    [https://arxiv.org/abs/2602.21361](https://arxiv.org/abs/2602.21361)

    该论文提出了一种统一的自监督逆映射神经网络框架，利用离焦弯曲波前探针提供的相位多样性，同时支持单帧菲涅尔相干衍射成像和重叠叠层成像的重建，摆脱了重叠约束，实现更稀疏的扫描和更低的辐射剂量。

    

    在同步辐射光源和X射线自由电子激光装置上的叠层成像（ptychography）需要密集重叠的扫描，这限制了成像通量并增加了辐射剂量；将相干衍射成像扩展到对扩展样品无需重叠的操作仍然是一个悬而未决的问题。我们提出了一种自监督逆映射网络，用于单帧菲涅尔相干衍射成像（CDI）以及采用固定、预先估计探针的重叠叠层成像。该学习到的神经网络可以从单个衍射帧或若干重叠测量中一次重建单个物体图块。在单帧模式下，弯曲波前探针在离焦样品位置所提供的相位多样性消除了对重叠约束的要求，从而实现更稀疏的扫描，并在固定曝光条件下成比例地降低剂量。在合成线条图案上，使用弯曲探针的单帧模式重建振幅的结构相似性（SSIM）超过0.90……

    arXiv:2602.21361v4 Announce Type: replace-cross  Abstract: Ptychographic imaging at synchrotron and X-ray free-electron laser sources requires densely overlapping scans, which limits throughput and increases dose; extending coherent diffractive imaging to overlap-free operation on extended samples remains an open problem. We present a self-supervised inverse-mapping network for single-frame Fresnel coherent diffraction imaging (CDI) and overlapped ptychography with fixed, pre-estimated probes. The learned neural network reconstructs individual object patches from either one diffraction frame or several overlapping measurements at a time. In single-frame mode, the phase diversity provided by the curved-wavefront probe at the off-focus sample position removes the requirement for overlap constraints, enabling sparser scans and proportionally lower dose at fixed exposure. On synthetic line patterns, reconstructed amplitude SSIM exceeds 0.90 in single-frame mode with the curved probe and re
    
[^224]: 面向稀疏损失的在线镜像下降之块范数几何

    Block-Norm Geometries for Online Mirror Descent with Sparse Losses

    [https://arxiv.org/abs/2602.13177](https://arxiv.org/abs/2602.13177)

    该论文提出一族随机化块范数镜像映射，在欧氏与熵几何之间插值以适应稀疏损失梯度，并对多种标准凸集证明了随维度多项式级改进的 regret 界。

    

    在线镜像下降的性能在很大程度上取决于其镜像映射所诱导的几何结构，然而标准算法主要依赖于两种经典选择：欧氏几何和熵几何。我们证明，当损失梯度是稀疏的时候，这两种几何都可能是明显次优的。我们提出了一族随机化的块范数镜像映射，它在欧氏几何和熵几何之间进行插值，并能适应中间程度的稀疏结构。对于若干标准凸集，包括 $\ell_p$ 球、椭球、盒约束以及范数球的闵可夫斯基和，我们证明了相对于在线投影梯度下降和指数梯度下降中较优者的 regret 界在维度上呈多项式级的改进。我们进一步构造了显式的在线凸优化实例来展示这些改进能够实现：在一个单纯多面体上，一种中间的块几何在 regret 上取得了相对（标准方法）$\text{poly}(d)$ 级的分离……

    arXiv:2602.13177v2 Announce Type: replace-cross  Abstract: The performance of online mirror descent depends critically on the geometry induced by its mirror map, yet standard algorithms largely rely on two canonical choices: Euclidean and entropic geometry. We show that these two geometries can both be substantially suboptimal when loss gradients are sparse. We introduce a family of randomized block-norm mirror maps that interpolates between Euclidean and entropic geometries and adapts to intermediate sparsity structure. For several standard convex sets, including $\ell_p$ balls, ellipsoids, boxes, and Minkowski sums of norm balls, we prove polynomial-in-dimension improvements in regret bounds over the better of online projected gradient descent and exponentiated gradient. We further construct explicit online convex optimization instances for which these improvements are realized: on a simple polytope, an intermediate block geometry achieves a $\text{poly}(d)$ separation in regret from
    
[^225]: 基于机器学习的禅那高级专注入定冥想的7特斯拉功能磁共振成像分类研究

    Machine Learning-Based Classification of Jhana Advanced Concentrative Absorption Meditation Using 7 Tesla Functional Magnetic Resonance Imaging

    [https://arxiv.org/abs/2602.13008](https://arxiv.org/abs/2602.13008)

    该研究的核心创新在于首次利用7T fMRI衍生的区域一致性特征与机器学习方法，在个体水平上实现对禅那高级专注入定冥想状态的分类，突破了以往仅依赖组水平单变量对比分析的局限。

    

    引言：禅那高级专注入定冥想（ACAM-J）涉及意识的深刻变化，其神经关联对于理解意识与幸福感具有重要意义。以往的神经影像学研究依赖于单变量、组水平的对比分析，尚未回答ACAM-J是否携带可从个体扫描中解码的分布式神经特征。本研究评估基于功能磁共振成像的区域一致性能否利用机器学习对ACAM-J进行分类。方法：我们分析了20名资深冥想者的7T fMRI数据，这些冥想者依次经历了其标准的ACAM-J序列及两项匹配的对照任务，此外还纳入了一名案例研究参与者的密集数据（该数据被保留用于最终评估）。研究为每个阶段计算ReHo图，并将其划分为覆盖皮层、皮层下区域、脑干和小脑的498个脑区。在被试内分层交叉验证中，进行了特征排序、递归特征消除和分类器……（原文摘要在此处截断）

    arXiv:2602.13008v2 Announce Type: replace  Abstract: Introduction: Jhana advanced concentrative absorption meditation (ACAM-J) involves profound changes in consciousness, making its neural correlates important for understanding consciousness and well-being. Prior neuroimaging has relied on univariate, group-level contrasts, leaving open whether ACAM-J carries distributed neural signatures decodable from individual scans. This study evaluates whether fMRI-derived regional homogeneity (ReHo) can classify ACAM-J using machine learning.   Methods: We analysed 7T fMRI data from 20 advanced meditators who progressed through their standard ACAM-J sequence and two matched control tasks, plus intensive data from one case-study participant held out for final evaluation. ReHo maps were computed per segment and parcellated into 498 regions spanning cortex, subcortex, brainstem, and cerebellum. Within subject-wise stratified cross-validation, feature ranking, recursive feature elimination, and clas
    
[^226]: 基于PPG衍生血流动力学特征的院内中风风险状态分类

    In-Hospital Stroke Risk-State Classification from PPG-Derived Hemodynamic Features

    [https://arxiv.org/abs/2602.09328](https://arxiv.org/abs/2602.09328)

    该研究通过LLM辅助流程从非结构化临床笔记中提取经医生裁定的中风锚点，并利用PPG衍生的血流动力学特征和ResNet-1D分类器，实现了院内中风风险状态的准确分类。

    

    事件发生前时间对齐的生理数据稀缺，限制了对有记录临床识别之前中风风险状态的研究。我们关注住院期间发生中风且接受持续监测的患者，从而能够对锚点前光电容积脉搏波（PPG）数据进行回顾性分析。利用MIMIC-III和MC-MED数据集，一种LLM辅助流程从非结构化临床笔记中生成了候选中风锚点。所有保留的锚点均经过医生裁定，一项分层的100例双重审查审核显示，候选锚点与裁定锚点在±15分钟内的一致性达95.0%。我们识别出176名MIMIC-III患者和154名MC-MED患者拥有符合条件的同步锚点前PPG数据。研究采用固定的17通道血流动力学表示和ResNet-1D分类器，通过患者层面的五折内部验证和冻结的外部测试进行评估。在验证集选定的工作点上，F1分数分别为0.7956、0.8759……

    arXiv:2602.09328v2 Announce Type: replace  Abstract: The scarcity of temporally aligned pre-event physiological data limits the study of stroke risk states before documented clinical recognition. We focus on patients who experienced stroke during hospitalization while undergoing continuous monitoring, enabling retrospective analysis of pre-anchor photoplethysmography (PPG). Using MIMIC-III and MC-MED, an LLM-assisted pipeline generated candidate stroke anchors from unstructured notes. All retained anchors underwent physician adjudication, and a stratified 100-case double-review audit showed 95.0% candidate-to-adjudicated agreement within +/-15 min. We identified 176 MIMIC-III patients and 154 MC-MED patients with eligible synchronized pre-anchor PPG. A fixed 17-channel hemodynamic representation and ResNet-1D classifier were evaluated using patient-level five-fold internal validation and frozen external testing. At validation-selected operating points, F1-scores were 0.7956, 0.8759, an
    
[^227]: 本福特定律作为大语言模型训练后量化的分布先验

    Benford's Law as a Distributional Prior for Post-Training Quantization of Large Language Models

    [https://arxiv.org/abs/2602.00165](https://arxiv.org/abs/2602.00165)

    该论文观察到Transformer的线性权重符合本福特分布而LayerNorm参数系统偏离，据此提出免数据的量化码本BenQ，用对数间隔网格作为码本，选择性地应用于变换层，在4位训练后量化中一致优于均匀RTN。

    

    训练后量化（PTQ）是降低大语言模型内存占用的实用方法，但低位量化对量化码本与经验权重/激活分布之间的失配非常敏感。我们重新审视类本福特的首位数字统计，将其作为Transformer张量中尺度广泛行为的轻量级诊断工具。在多个模型家族中，我们观察到一致的功能二分现象：具有变换性质的nn.Linear权重倾向于符合本福特分布，而LayerNorm参数则系统性地偏离该分布。基于这一观察，我们提出了BenQ，一种免数据的PTQ码本，它使用简单的对数间隔网格作为尺度广泛分布的代理，并有选择地将其应用于变换层，同时将稳定性关键的参数保持在更高精度。在4位分组PTQ中，BenQ在各个架构上一致优于均匀RTN，并与NF4互有胜负。

    arXiv:2602.00165v2 Announce Type: replace  Abstract: Post-training quantization (PTQ) is a practical way to reduce the memory footprint of large language models, but low-bit quantization is sensitive to mismatches between the quantization codebook and the empirical weight/activation distributions. We revisit Benford-like leading-digit statistics as a lightweight diagnostic of scale-broad behavior in transformer tensors. Across several model families, we observe a consistent functional dichotomy: transformational nn.Linear weights tend to be Benford-like, whereas LayerNorm parameters systematically deviate. Motivated by this observation, we propose BenQ, a data-free PTQ codebook that uses a simple log-spaced grid as a proxy for scale-broad distributions and applies it selectively to transformational layers while keeping stability-critical parameters in higher precision. In 4-bit group-wise PTQ, BenQ consistently improves over uniform RTN and trades wins with NF4 across architectures and
    
[^228]: 基于约束二值优化的块删除大语言模型压缩

    LLM Compression by Block Removal with Constrained Binary Optimization

    [https://arxiv.org/abs/2602.00161](https://arxiv.org/abs/2602.00161)

    该论文将大语言模型的块删除压缩问题形式化为约束二值优化问题并映射到Ising自旋玻璃物理系统，利用系统能量作为下游模型性能的代理指标，在深度压缩场景下（如50%压缩Llama-3.3-70B）相比现有最先进方法在MMLU基准上提升近23个百分点。

    

    在本文中，我们将通过最优删除Transformer块（“块删除”）来压缩大语言模型（LLM）的问题形式化为一个约束二值优化（CBO）问题，该问题可以映射到一个物理系统（Ising自旋玻璃），其能量是下游模型性能的强有力代理指标。这种形式化使得能够对大量候选块删除配置进行高效排序，从而产生许多高质量的、非平凡的解决方案，超越了仅删除连续区域的方法。我们的方法在深度压缩场景下表现强劲，例如对Llama-3.3-70B-Instruct进行50%压缩时，与其他最先进的（SOTA）块删除方法相比，我们在MMLU基准上实现了近23个百分点的性能提升。对于较轻程度的压缩，对于Llama-3.1-8B-Instruct和Qwen3-14B（在重训练前后），我们的方法在多个基准测试中与现有方法表现相当。

    arXiv:2602.00161v3 Announce Type: replace-cross  Abstract: In this paper, we formulate the compression of large language models (LLMs) by optimally deleting transformer blocks (``block removal'') as a constrained binary optimization (CBO) problem that can be mapped to a physical system (Ising glass), whose energies are a strong proxy for downstream model performance. This formulation enables an efficient ranking of a large number of candidate block-removal configurations yielding many high-quality, non-trivial solutions beyond those only removing consecutive regions. Our method performs strongly in the deep compression regime, such as for 50% compression of Llama-3.3-70B-Instruct, where we achieve an almost 23 percentage point increase on the MMLU benchmark compared to other state-of-the-art (SOTA) block-removal methods. For lighter compression, it performs on par with those methods across several benchmarks for Llama-3.1-8B-Instruct, Qwen3-14B (both before and after retraining), as we
    
[^229]: 基于调制专家混合的多模态时间序列预测

    Multi-Modal Time Series Prediction via Mixture of Modulated Experts

    [https://arxiv.org/abs/2601.21547](https://arxiv.org/abs/2601.21547)

    提出了一种名为“专家调制”的新机制，使混合专家模型的路由与专家计算均受文本信号调制，从而摆脱对token级融合的依赖，提升多模态时间序列预测的准确性与跨模态对齐能力。

    

    现实世界中的时间序列呈现出复杂且不断演化的动态特性，使得准确预测极具挑战性。近期的多模态预测方法利用新闻报道等文本信息来改进预测，但大多数方法依赖于token级融合，即在共享嵌入空间中将时间片段与语言token混合。然而，当高质量的“时间序列-文本”配对数据稀缺、且时间序列在特性上存在巨大差异时，这种融合方式可能并不适用，从而使跨模态对齐变得复杂。与此同时，混合专家（MoE）架构已被证明在时间序列建模和多模态学习中均行之有效，但许多现有的基于MoE的模态融合方法仍然依赖于token级融合。为解决这一问题，我们提出了专家调制，这是一种用于多模态时间序列预测的新机制，它使路由和专家计算均以文本信号为条件，从而能够……（摘要至此截断）

    arXiv:2601.21547v2 Announce Type: replace-cross  Abstract: Real-world time series exhibit complex and evolving dynamics, making accurate forecasting extremely challenging. Recent multi-modal forecasting methods leverage textual information such as news reports to improve prediction, but most rely on token-level fusion that mixes temporal patches with language tokens in a shared embedding space. However, such fusion can be ill-suited when high-quality time-text pairs are scarce and when time series exhibit substantial variation in characteristics, thus complicating cross-modal alignment. In parallel, mixture-of-experts (MoE) architectures have proven effective for both time series modeling and multi-modal learning, yet many existing MoE-based modality integration methods still depend on token-level fusion. To address this, we propose Expert Modulation, a new mechanism for multi-modal time series prediction that conditions both routing and expert computation on textual signals, enabling 
    
[^230]: LLAMA LIMA：关于生成式人工智能对数学学习影响的活态元分析

    LLAMA LIMA: A Living Meta-Analysis on the Effects of Generative AI on Learning Mathematics

    [https://arxiv.org/abs/2601.18685](https://arxiv.org/abs/2601.18685)

    该论文提出一种每两个月更新文献库的“活态元分析”方法，持续追踪生成式AI干预对数学学习的效果，最新结果显示其具有日益显著的积极作用（g = 0.57），且当AI辅助教师教学、干预时间较长时效果更佳。

    

    生成式人工智能在数学教育中的能力正在迅速发展，给研究同步跟进带来了重大挑战。相关研究综述仍然稀缺，且在发表时就有过时的风险。我们提出了一项关于基于生成式人工智能的干预措施对数学学习效果的活态元分析（LIMA）。我们每两个月更新一次文献库，并采用累积贝叶斯多层元回归模型。本文报告了第五版的结果，共纳入34项研究，其中7项是自第四版以来新加入的。结果表明其效果日益积极（g = 0.57，置信区间 [0.36, 0.79]），且没有证据表明存在发表偏倚。调节效应分析显示，效果因整合模式和干预时长而异：当生成式人工智能作为教师教学的补充手段以及干预时间较长时，观察到更大的效果。有中等程度的证据支持更强的……（原文截断）

    arXiv:2601.18685v5 Announce Type: replace-cross  Abstract: The capabilities of generative AI in mathematics education are rapidly evolving, posing significant challenges for research to keep pace. Research syntheses remain scarce and risk being outdated by the time of publication. We present a Living Meta-Analysis (LIMA) on the effects of generative AI-based interventions for learning mathematics. We update the literature base every two months and apply a cumulative Bayesian multilevel meta-regression model. This paper reports results from the fifth version, including 34 studies, 7 of which were newly included since the fourth version. Results indicate an increasingly positive effect (g = 0.57, CrI [0.36, 0.79]) and no evidence in favor of a publication bias. Moderator analyses indicate that effects vary by integration mode and intervention duration: Larger effects were observed when generative AI supplemented teacher instruction and in longer interventions. Moderate evidence for stron
    
[^231]: AnyView：在动态场景中合成任意新视角

    AnyView: Synthesizing Any Novel View in Dynamic Scenes

    [https://arxiv.org/abs/2601.16982](https://arxiv.org/abs/2601.16982)

    提出了AnyView，一个基于扩散模型、几乎无几何假设的动态视角合成框架，通过融合2D单目、3D静态多视角和4D动态多视角等多源数据训练通用时空隐式表示，实现从任意相机位置和轨迹零样本生成新视角视频，并发布了针对极端动态场景的新基准AnyViewBench。

    

    现代生成式视频模型擅长生成令人信服的高质量输出，但在高度动态的真实世界环境中难以保持多视角一致性和时空一致性。在本工作中，我们提出了**AnyView**，一个面向*动态视角合成*的基于扩散模型的视频生成框架，它几乎不依赖归纳偏置或几何假设。我们利用了具有不同监督程度的多种数据源，包括单目（2D）、多视角静态（3D）和多视角动态（4D）数据集，来训练一个通用的时空隐式表示，该表示能够从任意相机位置和轨迹零样本生成新的视频。我们在标准基准上评估了AnyView，取得了与当前最先进方法相当的结果，并提出了**AnyViewBench**，一个针对多样真实场景中*极端*动态视角合成的具有挑战性的新基准。

    arXiv:2601.16982v2 Announce Type: replace-cross  Abstract: Modern generative video models excel at producing convincing, high-quality outputs, but struggle to maintain multi-view and spatiotemporal consistency in highly dynamic real-world environments. In this work, we introduce $\textbf{AnyView}$, a diffusion-based video generation framework for $\textit{dynamic view synthesis}$ with minimal inductive biases or geometric assumptions. We leverage multiple data sources with various levels of supervision, including monocular (2D), multi-view static (3D) and multi-view dynamic (4D) datasets, to train a generalist spatiotemporal implicit representation capable of producing zero-shot novel videos from arbitrary camera locations and trajectories. We evaluate AnyView on standard benchmarks, showing competitive results with the current state of the art, and propose $\textbf{AnyViewBench}$, a challenging new benchmark tailored towards $\textit{extreme}$ dynamic view synthesis in diverse real-wo
    
[^232]: 面向可训练性的混合量子回归：基于几何预处理与课程优化的方法

    Trainability-Oriented Hybrid Quantum Regression via Geometric Preconditioning and Curriculum Optimization

    [https://arxiv.org/abs/2601.11942](https://arxiv.org/abs/2601.11942)

    该论文提出一种混合量子-经典回归框架，通过可学习的经典几何预处理器重塑输入表示，并结合逐步增加电路深度、从SPSA随机探索过渡到Adam梯度微调的课程优化协议，有效提升了量子神经网络在回归任务中的可训练性。

    

    量子神经网络（QNN）在科学机器学习领域引起了越来越多的关注，但在回归任务中，它们常常因噪声梯度和病态优化问题而面临可训练性受限的困扰。我们提出了一个旨在缓解这些瓶颈的混合量子-经典回归框架。我们的模型在前面添加了一个轻量级的经典嵌入层，充当可学习的几何预处理器，重塑输入表示，从而更好地调节下游变分量子电路的条件。在此架构基础上，我们引入了一种课程优化协议，逐步增加电路深度，并从基于SPSA的随机探索过渡到基于Adam的梯度微调。我们在PDE信息回归基准和标准回归数据集上，于模拟器环境下采用固定训练预算对该方法进行了评估。实证结果表明，所提出的框架持续改进了（摘要至此被截断）

    arXiv:2601.11942v4 Announce Type: replace  Abstract: Quantum neural networks (QNNs) have attracted growing interest for scientific machine learning, yet in regression settings they often suffer from limited trainability under noisy gradients and ill-conditioned optimization. We propose a hybrid quantum--classical regression framework designed to mitigate these bottlenecks. Our model prepends a lightweight classical embedding that acts as a learnable geometric preconditioner, reshaping the input representation to better condition a downstream variational quantum circuit. Building on this architecture, we introduce a curriculum optimization protocol that progressively increases circuit depth and transitions from SPSA-based stochastic exploration to Adam-based gradient fine-tuning. We evaluate the approach on PDE-informed regression benchmarks and standard regression datasets under a fixed training budget in a simulator setting. Empirically, the proposed framework consistently improves ov
    
[^233]: PACEvolve：实现进度感知的一致性进化

    PACEvolve: Enabling Progress-Aware Consistent Evolution

    [https://arxiv.org/abs/2601.10657](https://arxiv.org/abs/2601.10657)

    PACEvolve提出了一个进度感知的一致性进化框架，通过分层上下文管理等三项关键技术来治理智能体的记忆与搜索动态，从而解决自进化LLM智能体中的上下文污染和模式坍塌问题。

    

    由大语言模型（LLM）驱动的自进化智能体已成为代码优化和科学发现等多个领域的有前景研究方向，但其核心失效模式仍未得到充分探索。通过全面的实证研究，我们发现模型的推理会锚定在当前假设的局部上下文中，过度强调低层次细节而忽视更广阔的搜索空间。因此，这类智能体容易受到上下文污染和模式坍塌的影响，反复陷入有缺陷的假设并收敛到次优解。为应对这一挑战，我们提出了进度感知一致性进化框架（PACEvolve），这是一个用于管理智能体记忆和搜索动态的系统性框架。PACEvolve通过三项关键技术克服了上述局限：（1）分层上下文管理（HCM），它在结构化历史轨迹的同时进行动态剪枝……（摘要在此处被截断）

    arXiv:2601.10657v3 Announce Type: replace-cross  Abstract: Self-evolving agents powered by Large Language Models (LLMs) have emerged as a promising direction across diverse domains, including code optimization and scientific discovery, yet their core failure modes remain underexplored. Through a comprehensive empirical study, we identify that the model's reasoning becomes anchored to the local context of current hypotheses, overemphasizing low-level details while neglecting the broader search landscape. As a result, such agents become prone to context pollution and mode collapse, repeatedly revisiting flawed hypotheses and converging on suboptimal solutions. To address this challenge, we propose Progress-Aware Consistent Evolution (PACEvolve), a systematic framework for governing agent memory and search dynamics. PACEvolve overcomes these limitations through three key techniques: (1) Hierarchical Context Management (HCM), which structures historical trajectories while dynamically pruni
    
[^234]: 利用近端算子与哈密顿-雅可比方程之间联系求解逆问题的深度学习方法

    Deep learning methods for inverse problems using connections between proximal operators and Hamilton-Jacobi equations

    [https://arxiv.org/abs/2512.23829](https://arxiv.org/abs/2512.23829)

    本文提出利用近端算子与哈密顿-雅可比偏微分方程之间的深刻联系，构建新的深度学习架构来学习逆问题中的先验信息，从而为求解病态逆问题提供更有效的深度学习方法。

    

    逆问题是一类重要的数学问题，旨在从含噪数据中恢复模型参数。由于逆问题通常是病态（不适定）的，因此需要正则化或引入关于底层模型或未知变量的先验信息。近端算子在非光滑优化中无处不在，是这一领域的核心，因为它们编码了先验信息并能产生高效的迭代算法。近年来，近端算子也成为现代机器学习方法的关键，例如使用学习得到的去噪器的即插即用（plug-and-play）方法，以及用于学习近端算子先验的深度神经网络架构。后者的部分发展得益于近期将非凸先验的近端算子刻画为凸势函数次微分的研究工作。在本工作中，我们提出利用近端算子与哈密顿-雅可比偏微分方程（HJ PDEs）之间的联系，来开发用于学习先验的深度学习架构（摘要在此处截断）。

    arXiv:2512.23829v3 Announce Type: replace-cross  Abstract: Inverse problems are important mathematical problems that seek to recover model parameters from noisy data. Since inverse problems are often ill-posed, they require regularization or incorporation of prior information about the underlying model or unknown variables. Proximal operators, ubiquitous in nonsmooth optimization, are central to this because they encode priors and yield efficient iterative algorithms. They have also recently become key to modern machine learning methods, e.g., plug-and-play methods with learned denoisers and deep neural architectures for learning priors of proximal operators. The latter was developed partly due to recent work characterizing proximal operators of nonconvex priors as subdifferentials of convex potentials. In this work, we propose to leverage connections between proximal operators and Hamilton--Jacobi partial differential equations (HJ PDEs) to develop deep learning architectures for lear
    
[^235]: 用于重症监护病房患者心电图心房颤动检测的数据集与基准测试

    A Dataset and Benchmarks for Atrial Fibrillation Detection from Electrocardiograms of Intensive Care Unit Patients

    [https://arxiv.org/abs/2512.18031](https://arxiv.org/abs/2512.18031)

    该研究发布了带标注的ICU心电图心房颤动检测数据集与基准，并系统比较了基于特征的分类器、深度学习和心电图基础模型三种AI方法，发现心电图基础模型的检测性能最佳。

    

    目标：心房颤动（AF）是重症监护病房（ICU）患者中最常见的心律失常，可能造成不良健康影响。在本研究中，我们发布了一个带标注的ICU数据集以及用于心房颤动检测的基准测试。方法：我们在三种数据驱动的人工智能（AI）方法中对机器学习模型进行了比较：基于特征的分类器、深度学习（DL）和心电图基础模型（FMs）。这一比较填补了文献中的关键空白，旨在确定哪种AI方法最适合进行准确的心房颤动检测。实验使用了来自加拿大一家ICU的心电图数据以及2021年PhysioNet/Computing in Cardiology挑战赛的数据。研究测试了多种训练配置，从零样本推理到迁移学习。结果：在两个数据集上的平均结果显示，心电图基础模型表现最佳，其次是深度学习，然后是基于特征的分类器。取得最佳性能的模型……（摘要原文在此处截断）

    arXiv:2512.18031v2 Announce Type: replace  Abstract: Objective: Atrial fibrillation (AF) is the most common cardiac arrhythmia experienced by intensive care unit (ICU) patients and can cause adverse health effects. In this study, we publish a labelled ICU dataset and benchmarks for AF detection. Methods: We compared machine learning models across three data-driven artificial intelligence (AI) approaches: feature-based classifiers, deep learning (DL), and ECG foundation models (FMs). This comparison addresses a critical gap in the literature and aims to pinpoint which AI approach is best for accurate AF detection. Electrocardiograms (ECGs) from a Canadian ICU and the 2021 PhysioNet/Computing in Cardiology Challenge were used to conduct the experiments. Multiple training configurations were tested, ranging from zero-shot inference to transfer learning. Results: On average and across both datasets, ECG FMs performed best, followed by DL, then feature-based classifiers. The model that achi
    
[^236]: 面向双样本检验的经典与量子核融合方法

    Classical and quantum kernel fusion for two-sample testing

    [https://arxiv.org/abs/2511.20941](https://arxiv.org/abs/2511.20941)

    本文提出了一种融合经典核与量子核的混合双样本检验策略，通过增强MMD-FUSE框架，使得检验即使在小数据集上也能有效进行。

    

    双样本检验已被广泛应用于各个科学领域和机器学习中，用于判别两组样本是否来自同一分布。基于核的假设检验方法已被提出，通过将数据嵌入到再生核希尔伯特空间（RKHS）中，以无模型的方式高效地解缠数据中的高维复杂结构，从而获得准确的结果。虽然核的选择对其性能起着至关重要的作用，但关于如何选择核——尤其是在小数据集情况下——人们知之甚少。在此，我们基于使用最大均值差异的核检验理论基础（称为MMD-FUSE），构建了一种即使在小数据集上也能有效工作的假设检验。我们通过引入量子核来增强MMD-FUSE框架，并提出了一种融合经典核与量子核的新型混合检验策略。

    arXiv:2511.20941v2 Announce Type: replace-cross  Abstract: Two-sample tests have been extensively employed in various scientific fields and machine learning to discriminate whether two sets of samples come from the same distribution or not. Kernel-based procedures for hypothetical testing have been proposed to efficiently disentangle high-dimensional complex structures in data to obtain accurate results in a model-free way by embedding the data into the reproducing kernel Hilbert space (RKHS). While the choice of kernels plays a crucial role for their performance, little is understood about how to choose kernel especially for small datasets. Here we construct a hypothetical test which can be effective even for small datasets, based on the theoretical foundation of kernel-based tests using maximum mean discrepancy, which is called MMD-FUSE. We enhance the MMD-FUSE framework by incorporating quantum kernels and propose a novel hybrid testing strategy that fuses classical and quantum kern
    
[^237]: 面向可扩展混合专家推理的动态专家量化

    Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference

    [https://arxiv.org/abs/2511.15015](https://arxiv.org/abs/2511.15015)

    提出DynaExq系统，将HBM受限的单GPU混合专家模型推理转化为在线预算约束的动态精度分配问题，根据专家的运行时流量动态分配量化位宽，以实现高效的单GPU部署。

    

    混合专家（MoE）已成为在保持单token计算量适中的同时扩展大语言模型容量的实用架构，但在单个内存受限的GPU上部署MoE模型仍然困难，因为专家权重占据了高带宽内存（HBM）占用的主导地位。现有的专家卸载和预取系统减少了常驻内存集合，但当激活变得密集时，它们往往在关键路径上付出专家加载的开销。训练后量化（PTQ）无需数据传输即可降低内存占用，但主流流程在离线阶段固定专家位宽并假设路由保持稳定，尽管MoE专家利用率呈重尾分布，且热点集合可能随工作负载而变化。我们提出DynaExq，一个运行时感知的混合精度服务系统，将严格HBM预算限制下的单GPU MoE推理视为一个在线的、预算受限的精度分配问题。其关键洞察是保留那些主导运行时流量（的专家）

    arXiv:2511.15015v4 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) has become a practical architecture for scaling LLM capacity while keeping per-token compute modest, but deploying MoE models on a single, memory-limited GPU remains difficult because expert weights dominate the HBM footprint. Existing expert offloading and prefetching systems reduce the resident set, yet they often pay expert-loading costs on the critical path when activation becomes dense. Post-training quantization (PTQ) lowers the footprint without transfers, but prevailing pipelines fix expert bit-widths offline and assume routing remains stable, even though MoE expert utilization is heavy-tailed and the hot set can shift across workloads.   We present DynaExq, a runtime-aware mixed-precision serving system that treats single-GPU MoE inference under a hard HBM envelope as an online, budget-constrained precision allocation problem. The key insight is to keep the experts that dominate runtime traffic
    
[^238]: 当偏见伪装成真相：虚假关联如何破坏大语言模型中的幻觉检测

    When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs

    [https://arxiv.org/abs/2511.07318](https://arxiv.org/abs/2511.07318)

    该论文揭示了一类由训练数据中虚假关联（如姓氏与国籍的关联）驱动的幻觉，这类幻觉被模型自信生成、不受模型规模扩大影响、能规避现有检测方法、且在拒绝微调后仍持续存在，导致基于置信度过滤和内部状态探测等主流幻觉检测方法从根本上失效。

    

    尽管取得了长足进步，大语言模型（LLMs）仍然会表现出幻觉现象，生成看似合理但不正确的回答。在本文中，我们强调了一类关键但此前未被充分探索的由虚假关联驱动的幻觉——即训练数据中特征（如姓氏）与属性（如国籍）之间存在的表面化但在统计上显著的关联。我们证明，这些虚假关联所引发的幻觉具有以下特点：被模型以高置信度生成、不受模型规模扩大的影响、能够规避当前的检测方法，并且即使经过拒绝微调仍然持续存在。通过系统性控制的合成实验以及对最先进的开源和专有大语言模型（包括GPT-5）的实证评估，我们表明现有的幻觉检测方法，如基于置信度的过滤和内部状态探测，在虚假关联存在的情况下会从根本上失效。

    arXiv:2511.07318v3 Announce Type: replace  Abstract: Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correla
    
[^239]: 面向不平衡学习的偏差校正数据合成方法

    Bias-Corrected Data Synthesis for Imbalanced Learning

    [https://arxiv.org/abs/2510.26046](https://arxiv.org/abs/2510.26046)

    该论文提出了一种偏差校正的数据合成方法，通过从留出的多数类数据估计生成器引起的损失偏差并将其转移至少数类，为不平衡学习建立了有限样本理论保证，同时刻画了SMOTE产生显著损失偏差的条件。

    

    类别不平衡使概率分类变得复杂，因为标准训练目标侧重于多数类的性能。合成过采样可以缓解不平衡问题，但合成分布与目标少数类分布之间的差异可能会使拟合的分类器产生偏差，尤其是当合成样本依赖于观测数据时。我们提出了一种偏差校正程序，该程序从留出的多数类观测子集中估计由生成器引起的损失差异，并在统一的偏差转移条件下将该校正转移到少数类。我们为偏差转移以及所得经验风险最小化器的超额平衡风险建立了有限样本界，并刻画了SMOTE会产生不可忽略损失偏差的情形。该框架还可以应用于不平衡多任务学习和倾向得分估计，相关细节见补充材料。

    arXiv:2510.26046v3 Announce Type: replace-cross  Abstract: Class imbalance complicates probabilistic classification because standard training objectives emphasize majority-class performance. Synthetic oversampling can reduce imbalance, but discrepancies between the synthetic and target minority distributions may bias the fitted classifier, especially because synthetic samples depend on the observed data. We propose a bias-correction procedure that estimates the generator-induced loss discrepancy from a held-out subset of majority observations and transfers this correction to the minority class under a uniform bias-transfer condition. We establish finite-sample bounds for bias transfer and for the excess balanced risk of the resulting empirical risk minimizer, and characterize a regime in which SMOTE induces non-negligible loss bias. The framework can also be implemented in imbalanced multi-task learning and propensity-score estimation, with details provided in the Supplementary Materia
    
[^240]: Hurdle-RMIL：解决红外降雨反演中的零膨胀与长尾不平衡问题

    Hurdle-RMIL: Addressing Zero Inflation and Long-Tailed Imbalance in Infrared Rainfall Retrieval

    [https://arxiv.org/abs/2510.20486](https://arxiv.org/abs/2510.20486)

    该论文提出Hurdle-RMIL方法，将零膨胀与长尾不平衡分而治之，利用障碍模型处理零值膨胀，并通过基于贝叶斯的分布变换直接从自然长尾样本中学习平衡分布模型，有效缓解了卫星红外降雨反演中对高强度降雨的系统性低估问题。

    

    标签不平衡会导致高频样本主导基于AI的定量遥感，从而降低稀有事件的反演性能。在基于卫星红外亮温的降雨率反演中，这种不平衡导致对稀有高强度降雨的系统性低估。本研究提出了Hurdle-反演模型不平衡学习方法。遵循分而治之的策略，Hurdle-RMIL将零膨胀问题与正降雨的长尾分布分离开来。障碍模型用于处理零膨胀，而RMIL则利用固定观测条件下降雨到卫星正向过程的不变性，推导出一种基于贝叶斯的变换，将自然长尾降雨与假设平衡降雨下的条件分布联系起来。这一变换使平衡分布模型能够直接从自然样本中学习，而无需构建平衡数据集。与传统方法的比较……

    arXiv:2510.20486v2 Announce Type: replace  Abstract: Imbalanced labels can cause frequent samples to dominate AI-based quantitative remote sensing, degrading rare-event retrieval. In rain-rate retrieval based on satellite infrared brightness temperatures, this imbalance leads to systematic underestimation of rare high-intensity rainfall. In this study, Hurdle-Retrieval Model Imbalanced Learning (RMIL) is proposed. Following a divide-and-conquer strategy, Hurdle-RMIL separates zero inflation from the long-tailed distribution of positive rain. A hurdle model handles zero inflation, whereas RMIL exploits invariance under fixed observation conditions of the rainfall-to-satellite forward process to derive a Bayes-based transformation linking conditional distributions under naturally long-tailed and hypothetical balanced rainfall. This transformation enables the balanced-distribution model to be learned from natural samples without constructing a balanced dataset. Comparisons with convention
    
[^241]: 贝叶斯优化中的非线性降维技术

    Nonlinear Dimensionality Reduction Techniques for Bayesian Optimization

    [https://arxiv.org/abs/2510.15435](https://arxiv.org/abs/2510.15435)

    该论文提出将变分自编码器、深度度量损失、再训练策略与直接在潜空间进行的序贯域收缩（SDR-LSBO）相结合，有效提升了高维昂贵黑盒函数全局优化的性能。

    

    贝叶斯优化（BO）能够以高样本效率对昂贵的黑盒函数进行全局优化，但在高维情况下仍面临挑战。我们研究了通过非线性降维将问题转化为一系列低维潜空间贝叶斯优化（LSBO）问题。早期的LSBO使用线性随机嵌入和监督嵌入；在Grosnit等人工作的基础上，我们采用变分自编码器（VAE）、用于构建结构化潜流形的深度度量损失，以及再训练策略，使编码器-解码器对适应新采样的区域。我们将LSBO与直接在潜空间中进行的序贯域收缩（SDR）相结合（SDR-LSBO），随着证据的积累不断缩小搜索域。这些方法在GPU加速的BoTorch框架中实现，并采用Matérn-5/2高斯过程代理模型。我们的方法提升了基准测试中的优化质量，且再训练可以增强BO性能。与自适应监督线性随机嵌入的对比实验证明了基于VAE方法的有效性。

    arXiv:2510.15435v2 Announce Type: replace-cross  Abstract: Bayesian optimisation (BO) enables sample-efficient global optimisation of expensive black-box functions but remains challenging in high dimensions. We investigate nonlinear dimensionality reduction to a sequence of low-dimensional latent-space BO (LSBO) problems. Early LSBO used linear random and supervised embeddings; building on Grosnit et al., we employ variational autoencoders (VAEs), deep metric loss for structured latent manifolds, and retraining to adapt the encoder-decoder pair to newly sampled regions. We couple LSBO with sequential domain reduction (SDR) directly in latent space (SDR-LSBO), narrowing search domains as evidence accumulates. Implemented in GPU-accelerated BoTorch with Mat\'ern-5/2 Gaussian-process surrogates, our methods improve benchmark optimisation quality, and retraining can enhance BO performance. Comparisons with adaptive supervised linear random embeddings demonstrate the effectiveness of VAE-ba
    
[^242]: 教育中大语言模型文本检测器的局限性

    Limits of LLM Text Detectors in Education

    [https://arxiv.org/abs/2508.08096](https://arxiv.org/abs/2508.08096)

    本文提出了一个贡献感知的LLM文本检测评估框架，通过八个学生贡献等级模拟真实写作场景，揭示了当前基于二元区分的LLM检测器在教育评估中的局限性。

    

    学生们在学术写作中越来越多地借助大语言模型（LLM）的辅助。虽然大多数机构政策允许轻微的辅助（例如语法和风格修正以及反馈），但通常禁止将整个写作任务完全交由LLM完成。遗憾的是，目前的LLM生成文本检测方法大多假设人类写作与LLM生成文本之间存在二元区分，忽视了现实中人机协作实践的广度，从而限制了检测系统在教育评估中的有效性。在本文中，我们提出了一个面向教育领域的贡献感知型LLM检测系统评估框架。我们引入了一个包含八个学生贡献等级的量表，用以模拟真实的写作场景，涵盖从完全由人类撰写的文本、LLM辅助修改的文本，到完全由LLM生成并经过对抗性人化处理的文本。机构政策……

    arXiv:2508.08096v2 Announce Type: replace  Abstract: Students increasingly use the assistance of large language models (LLMs) in their academic writing. While slight assistance (e.g., grammar and style correction, as well as feedback) is permitted under most institutional policies, it is usually forbidden to offload entire writing tasks to LLMs. Unfortunately, current approaches to LLM-generated text detection predominantly assume a binary distinction between human-written and LLM-generated text, ignoring the breadth of realistic human-AI collaboration practices and limiting the validity of detection systems for educational assessment. In this paper, we propose a contribution-aware evaluation framework for LLM-based detection systems in education. We introduce a scale of eight student contribution levels that model realistic writing scenarios ranging from fully human-written texts to LLM-assisted revisions to fully LLM-generated and adversarially humanized texts. Institutional policies
    
[^243]: AsyncFlow：一个用于高效大语言模型后训练的异步流式强化学习框架

    AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training

    [https://arxiv.org/abs/2507.01663](https://arxiv.org/abs/2507.01663)

    AsyncFlow是一个面向高效LLM后训练的异步流式强化学习框架，其核心创新在于分布式数据存储与传输模块，以全流式方式实现全景数据管理、细粒度调度、自动化流水线重叠和动态负载均衡，解决了传统RL框架的可扩展性瓶颈与资源闲置问题。

    

    强化学习（RL）已成为大语言模型（LLM）后训练阶段的一项关键技术。传统的任务共置式RL框架存在显著的可扩展性瓶颈，而任务分离式RL框架则面临管理复杂数据流和解决资源闲置的挑战。此外，大多数现有框架与LLM训练或推理引擎紧密耦合，难以支持自定义设计的引擎。为应对这些挑战，我们提出了AsyncFlow，一个为高效后训练量身定制的异步流式RL框架。具体而言，我们引入了一个分布式数据存储与传输模块，以完全流式的方式提供全景数据管理和细粒度调度能力。这种架构天然地实现了RL任务之间的自动化流水线重叠和动态负载均衡。此外，我们提出了一种异步生产者-

    arXiv:2507.01663v2 Announce Type: replace  Abstract: Reinforcement learning (RL) has become a pivotal technology in the post-training phase of large language models (LLMs). Traditional task-collocated RL frameworks suffer from significant scalability bottlenecks, while task-separated RL frameworks face challenges in managing complex dataflows and resolving resource idling. Furthermore, most existing frameworks are tightly coupled with LLM training or inference engines, making them difficult to support custom-designed engines. To address these challenges, we propose AsyncFlow, an asynchronous streaming RL framework tailored for efficient post-training. Specifically, we introduce a distributed data storage and transfer module that provides panoramic data management and fine-grained scheduling capabilities in a fully streamed manner. This architecture inherently enables automated pipeline overlapping among RL tasks and dynamic load-balancing. Moreover, we propose an asynchronous producer-
    
[^244]: 论非可分离近似消息传递算法的普适性

    On Universality of Non-Separable Approximate Message Passing Algorithms

    [https://arxiv.org/abs/2506.23010](https://arxiv.org/abs/2506.23010)

    本文提出了表示张量的有界组合性质（BCP）这一一般性条件，首次将非可分离近似消息传递（AMP）算法状态演化的普适性从高斯数据推广到非高斯数据。

    

    一阶迭代算法的平均场刻画——包括近似消息传递（AMP）、随机梯度与近端梯度下降、以及朗之万扩散——已经使我们对许多统计应用中的学习动力学有了精确的理解。对于非线性部分具有坐标可分离形式的算法，已知这类刻画对底层数据分布具有一定程度的普适性。然而，非可分离算法动力学的平均场刻画在很大程度上仍局限于独立同分布高斯数据或旋转不变数据。在这项工作中，我们开启了针对非可分离AMP算法普适性的研究。我们识别出一个针对具有多项式非线性的AMP算法的一般性条件——以其表示张量的有界组合性质（Bounded Composition Property，BCP）来表述——使得状态演化对于非高斯矩阵普遍成立。

    arXiv:2506.23010v2 Announce Type: replace-cross  Abstract: Mean-field characterizations of first-order iterative algorithms -- including Approximate Message Passing (AMP), stochastic and proximal gradient descent, and Langevin diffusions -- have enabled a precise understanding of learning dynamics in many statistical applications. For algorithms whose non-linearities have a coordinate-separable form, it is known that such characterizations enjoy a degree of universality with respect to the underlying data distribution. However, mean-field characterizations of non-separable algorithm dynamics have largely remained restricted to i.i.d. Gaussian or rotationally-invariant data.   In this work, we initiate a study of universality for non-separable AMP algorithms. We identify a general condition for AMP with polynomial non-linearities, in terms of a Bounded Composition Property (BCP) for their representing tensors, to admit a state evolution that holds universally for matrices with non-Gauss
    
[^245]: 扩散概率模型中高阶常微分方程求解器的快速收敛性

    Fast Convergence for High-Order ODE Solvers in Diffusion Probabilistic Models

    [https://arxiv.org/abs/2506.13061](https://arxiv.org/abs/2506.13061)

    本文针对具有任意方差调度的一般前向过程，在得分函数导数的实际假设下，对基于概率流ODE的p阶（指数）Runge-Kutta采样格式进行了严格的收敛性分析，证明了高阶ODE求解器在扩散概率模型中的快速收敛性。

    

    扩散概率模型通过学习逆转一个将数据转化为噪声的噪声注入过程来生成样本。一个关键的发展是将反向采样过程重新表述为确定性的概率流常微分方程（ODE），这使得能够使用高阶数值求解器进行高效采样。与传统的时间积分器分析不同，这一采样过程的准确性不仅取决于数值积分误差，还取决于学习到的得分函数（score function）的逼近质量和正则性，以及二者之间的相互作用。在这项工作中，我们对由概率流ODE导出的确定性采样器进行了严格的收敛性分析，适用于具有任意方差调度的一般前向过程。具体而言，我们在学习到的得分函数的一阶和二阶导数满足实际假设的条件下，开发并分析了p阶（指数）Runge-Kutta格式。

    arXiv:2506.13061v4 Announce Type: replace  Abstract: Diffusion probabilistic models generate samples by learning to reverse a noise-injection process that transforms data into noise. A key development is the reformulation of the reverse sampling process as a deterministic probability flow ordinary differential equation (ODE), which allows for efficient sampling using high-order numerical solvers. Unlike traditional time integrator analysis, the accuracy of this sampling procedure depends not only on numerical integration errors but also on the approximation quality and regularity of the learned score function, as well as their interaction. In this work, we present a rigorous convergence analysis of deterministic samplers derived from probability flow ODEs for general forward processes with arbitrary variance schedules. Specifically, we develop and analyze $p$-th order (exponential) Runge-Kutta schemes, under the practical assumption that the first and second derivatives of the learned 
    
[^246]: 基于观测数据对齐语言模型：因果视角下的机遇与风险

    Aligning Language Models with Observational Data: Opportunities and Risks from a Causal Perspective

    [https://arxiv.org/abs/2506.00152](https://arxiv.org/abs/2506.00152)

    本文从因果视角系统分析了利用历史观测数据微调大语言模型的机遇与风险，指出观测结果可作为A/B测试的低成本替代监督信号，但同时需警惕因果混淆带来的偏差。

    

    大语言模型正被广泛应用于各行各业，生成直接贡献于关键性能指标的文本，例如患者消息传递中的用药依从性和内容生成中的转化率。然而，预训练模型在对齐人类偏好或优化业务目标方面往往表现不足。因此，使用高质量的标注数据进行微调对于引导模型生成更有效的内容至关重要。受控实验（如A/B测试）可以提供此类数据，但其成本高昂，且伴随着重大的工程、后勤和伦理挑战。与此同时，企业拥有大量尚未充分利用的历史（观测）数据。在这项工作中，我们研究了使用观测数据微调大语言模型的挑战与机遇。我们表明，虽然观测结果可以提供有价值的监督信号（摘要在此处截断）。

    arXiv:2506.00152v2 Announce Type: replace  Abstract: Large language models are being widely used across industries to generate text that contributes directly to key performance metrics, such as medication adherence in patient messaging and conversion rates in content generation. Pretrained models, however, often fall short when it comes to aligning with human preferences or optimizing for business objectives. As a result, fine-tuning with good-quality labeled data is essential to guide models to generate content that achieves better results. Controlled experiments, like A/B tests, can provide such data, but they are often expensive and come with significant engineering, logistical, and ethical challenges. Meanwhile, companies have access to a vast amount of historical (observational) data that remains underutilized. In this work, we study the challenges and opportunities of fine-tuning LLMs using observational data. We show that while observational outcomes can provide valuable supervi
    
[^247]: 基于等变自编码器的三维瑞利-贝纳德对流代理建模

    Surrogate Modeling of 3D Rayleigh-Benard Convection with Equivariant Autoencoders

    [https://arxiv.org/abs/2505.13569](https://arxiv.org/abs/2505.13569)

    提出了一个由等变卷积自编码器和等变卷积LSTM组成的端到端等变代理模型，利用系统的E(2)-等变对称性对三维瑞利-贝纳德对流进行高效建模，从而提升精度与样本效率。

    

    使用机器学习对大规模物理系统进行建模、理解和控制正迅速流行起来，应用实例涵盖从电磁学、核聚变反应堆、磁流体动力学到流体力学和气候建模等领域。这些由偏微分方程支配的系统在自由度数量庞大以及时空多尺度复杂动力学方面带来了独特的挑战，因此非常需要额外的措施来提高精度和样本效率。我们提出了一个端到端的等变代理模型，该模型由使用G-可操纵核的等变卷积自编码器和等变卷积LSTM组成。作为案例研究，我们考虑了三维瑞利-贝纳德对流，该系统描述了加热底板和冷却顶板之间由浮力驱动的流体流动。该系统具有E(2)-等变特性……

    arXiv:2505.13569v3 Announce Type: replace  Abstract: The use of machine learning for modeling, understanding, and controlling large-scale physics systems is quickly gaining in popularity, with examples ranging from electromagnetism over nuclear fusion reactors and magneto-hydrodynamics to fluid mechanics and climate modeling. These systems - governed by partial differential equations - present unique challenges regarding the large number of degrees of freedom and the complex dynamics over many scales both in space and time, and additional measures to improve accuracy and sample efficiency are highly desirable. We present an end-to-end equivariant surrogate model consisting of an equivariant convolutional autoencoder and an equivariant convolutional LSTM using $G$-steerable kernels. As a case study, we consider the three-dimensional Rayleigh-B\'enard convection, which describes the buoyancy-driven fluid flow between a heated bottom and a cooled top plate. While the system is E(2)-equiva
    
[^248]: 针对具有交叉随机效应的广义混合效应模型的可扩展Krylov子空间方法

    Scalable Krylov Subspace Methods for Generalized Mixed-Effects Models with Crossed Random Effects

    [https://arxiv.org/abs/2505.09552](https://arxiv.org/abs/2505.09552)

    该论文提出了基于Krylov子空间的新方法，解决了广义混合效应模型中高维交叉随机效应导致的计算瓶颈，在保持同等精度的同时实现了几个数量级的加速。

    

    混合效应模型被广泛用于建模具有复杂分组结构和高基数分类预测变量的数据。然而，对于高维交叉随机效应，目前依赖Cholesky分解的标准计算方法可能变得极其缓慢。在这项工作中，我们提出了基于Krylov子空间的方法来解决现有的计算瓶颈，并从理论和实证两方面对其进行了分析。特别地，我们推导了预条件随机Lanczos求积法和共轭梯度法在混合效应模型中收敛性和准确性的新结果，并开发了用于计算预测方差的可扩展方法。在模拟数据和真实数据的实验中，所提出的方法实现了几个数量级的加速，在计算上比基于Cholesky分解的方法更加稳健，同时保持了基本相同的精度。

    arXiv:2505.09552v4 Announce Type: replace-cross  Abstract: Mixed-effects models are widely used to model data with complex grouping structures and high-cardinality categorical predictor variables. However, for high-dimensional crossed random effects, current standard computations relying on Cholesky decompositions can become prohibitively slow. In this work, we present Krylov subspace-based methods that address existing computational bottlenecks, and we analyze them both theoretically and empirically. In particular, we derive new results on the convergence and accuracy of the preconditioned stochastic Lanczos quadrature and conjugate gradient methods for mixed-effects models, and we develop scalable methods for calculating predictive variances. In experiments with simulated and real-world data, the proposed methods yield speedups of several orders of magnitude and are more computationally robust than Cholesky-based computations, while maintaining essentially the same accuracy.
    
[^249]: GLaMoR：使用图语言模型进行OWL本体一致性检查

    GLaMoR: Consistency Checking of OWL Ontologies using Graph Language Models

    [https://arxiv.org/abs/2504.19023](https://arxiv.org/abs/2504.19023)

    本文提出GLaMoR推理管道，通过将OWL本体转换为图结构数据并利用图语言模型（GLM）架构进行一致性检查，解决了现有推理机计算成本高、效率随本体规模增长而下降，以及经典机器学习方法无法考虑T-Box的问题。

    

    语义推理旨在从已有知识中推断新知识，OWL本体作为组织信息的标准化框架。语义推理中的一个关键挑战是验证本体的一致性。然而，最先进的推理机计算成本高昂，且随着本体规模的增大，其效率会下降。虽然经典机器学习模型已被探索用于A-Box公理的一致性检查，但如何考虑T-Box仍未得到解决。大型语言模型（LLM）在自然语言推理方面展现出有前景的结果，但在逻辑推理方面表现不佳。最近引入的图语言模型（GLM）提供了一种同时处理图结构数据和文本的方法。本文提出了GLaMoR（用于推理的图语言模型），这是一个推理管道，它将OWL本体转换为图结构数据，并调整GLM架构用于一致性检查。

    arXiv:2504.19023v2 Announce Type: replace-cross  Abstract: Semantic reasoning aims to infer new knowledge from existing knowledge, with OWL ontologies serving as a standardized framework for organizing information. A key challenge in semantic reasoning is verifying ontology consistency. However, state-of-the-art reasoners are computationally expensive, and their efficiency decreases as ontology sizes grow. While classical machine learning models have been explored for consistency checking of A-Box axioms, considering T-Boxes remains unaddressed. Large language models (LLMs) have shown promising results for natural language inference but perform poorly on logical reasoning. The recently introduced Graph Language Model (GLM) offers a way to simultaneously process graph-structured data and text. This paper proposes GLaMoR (Graph Language Model for Reasoning), a reasoning pipeline that transforms OWL ontologies into graph-structured data and adapts the GLM architecture for consistency chec
    
[^250]: SGD能否选出好的“渔夫”？自选择偏差下的局部收敛性

    Can SGD Select Good Fishermen? Local Convergence under Self-Selection Biases

    [https://arxiv.org/abs/2504.07133](https://arxiv.org/abs/2504.07133)

    本文提出了首个针对自选择偏差的局部收敛算法，通过将自选择问题归约为粗化估计问题，给出了运行时间为poly(d, k, 1/ε) + (k log k)^{O(k)}的更快算法，从而解决了CDIZ23提出的主要开放问题之一。

    

    我们重新审视了由Cherapanamjeri、Daskalakis、Ilyas和Zampetakis [CDIZ23, STOC'23]提出的在d维空间中使用最大选择准则估计具有自选择偏差的k个线性回归器的问题。我们的主要结果是一个运行时间为poly(d, k, 1/ε) + (k log k)^{O(k)}的算法，该算法改进了Cherapanamjeri、Daskalakis、Ilyas和Zampetakis [CDIZ23]以及Gaitonde和Mossel [GM24, arXiv]所提出算法的运行时间。我们通过提供首个针对自选择的局部收敛算法来实现这一点，从而解决了Cherapanamjeri、Daskalakis、Ilyas和Zampetakis [CDIZ23]提出的主要开放问题之一。为获得该算法，我们将自选择问题归约为一个看似无关的统计问题——粗化下的估计 [FKKT21, COLT'21]。粗化是指人们无法观测到样本的确切值，而只能观测到某个集合的情形。

    arXiv:2504.07133v2 Announce Type: replace-cross  Abstract: We revisit the problem of estimating $k$ linear regressors with self-selection bias in $d$ dimensions with the maximum selection criterion, as introduced by Cherapanamjeri, Daskalakis, Ilyas, and Zampetakis [CDIZ23, STOC'23]. Our main result is a $\mathrm{poly}(d, k, 1/\varepsilon) + (k \log k)^{O(k)}$ time algorithm for this problem that improves upon the running time of the algorithms by Cherapanamjeri, Daskalakis, Ilyas, and Zampetakis [CDIZ23] and Gaitonde and Mossel [GM24, arXiv]. We achieve this by providing the first local convergence algorithm for self-selection, thus resolving one of the main open questions of Cherapanamjeri, Daskalakis, Ilyas, and Zampetakis [CDIZ23].   To obtain this algorithm, we reduce self-selection to a seemingly unrelated statistical problem called estimation under coarsening [FKKT21, COLT'21]. Coarsening occurs when one does not observe the exact value of the sample but only some set (from a pa
    
[^251]: 一种基于广义切近似的强超高斯似然变分推断框架

    A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super-Gaussian Likelihoods

    [https://arxiv.org/abs/2504.05431](https://arxiv.org/abs/2504.05431)

    本文提出了一种基于广义切变换的变分推断框架，利用凸对偶性构建对数似然的切下界，使强超高斯似然类概率模型与高斯先验实现共轭，从而将该结构化变分方法的应用范围从逻辑回归扩展到更广泛的模型类别。

    

    变分推断作为马尔可夫链蒙特卡洛采样的替代方法，在实现复杂贝叶斯模型的可扩展计算方面发挥了变革性作用。然而，现有方法通常依赖于僵化的模型特定公式或随机黑盒优化程序。切近似是一类有原则的结构化变分方法，它利用了底层概率模型的几何特性。然而，其应用在很大程度上局限于逻辑回归及相关建模领域。在本文中，我们针对以强超高斯似然为特征的一类广泛概率模型，提出了一种基于切变换的新型变分框架。我们的方法利用凸对偶性来构建对数似然的切下界，从而在原本难以处理的设置中诱导出与模型参数高斯先验的共轭性。

    arXiv:2504.05431v4 Announce Type: replace-cross  Abstract: Variational inference, as an alternative to Markov chain Monte Carlo sampling, has played a transformative role in enabling scalable computation for complex Bayesian models. Nevertheless, existing approaches often depend on either rigid model-specific formulations or stochastic black-box optimization routines. Tangent approximation is a principled class of structured variational methods that exploits the geometry of the underlying probability model. However, its utility has largely been confined to logistic regression and related modeling regimes. In this article, we propose a novel variational framework based on tangent transformation for a broad class of probability models characterized by strongly super-Gaussian likelihoods. Our method leverages convex duality to construct tangent minorants of the log-likelihood, thereby inducing conjugacy with Gaussian priors over model parameters in an otherwise intractable setup. Under mi
    
[^252]: 通过求助实现不可逆动力学下的安全学习

    Safe Learning Under Irreversible Dynamics via Asking for Help

    [https://arxiv.org/abs/2502.14043](https://arxiv.org/abs/2502.14043)

    本文首次正式证明，通过允许智能体向导师求助并在相似状态间迁移知识，智能体可以在具有不可逆动力学的无限状态空间未知高风险环境中，以次线性遗憾和次线性求助次数安全高效地学习并获得高回报，无需依赖重置机制。

    

    arXiv:2502.14043v3 公告类型：replace-cross 摘要：大多数具有正式遗憾保证的学习算法本质上依赖于尝试所有可能的行为，当某些错误无法被挽回时，这会带来问题。为此，我们允许学习智能体向导师求助，并在相似状态之间进行知识迁移。我们证明，这种结合能够使智能体既安全又高效地学习。在标准的在线学习假设下，我们提出了一种算法，对于具有不可逆动力学和无限状态空间的马尔可夫决策过程，其遗憾值和导师查询次数相对于时间范围都是次线性的。我们的证明涉及一系列三个归约步骤，使我们的结果比单一算法更具普遍性。从概念上讲，我们的结果可能是首个正式证明：智能体可以在未知、无界且高风险的环境中，无需重置的情况下，在获得高回报的同时实现自给自足。

    arXiv:2502.14043v3 Announce Type: replace-cross  Abstract: Most learning algorithms with formal regret guarantees essentially rely on trying all possible behaviors, which is problematic when some errors cannot be recovered from. Instead, we allow the learning agent to ask for help from a mentor and to transfer knowledge between similar states. We show that this combination enables the agent to learn both safely and effectively. Under standard online learning assumptions, we provide an algorithm whose regret and number of mentor queries are both sublinear in the time horizon for Markov decision processes with irreversible dynamics and infinite state spaces. Our proof involves a sequence of three reductions, making our result more general than a single algorithm. Conceptually, our result may be the first formal proof that it is possible for an agent to obtain high reward while becoming self-sufficient in an unknown, unbounded, and high-stakes environment without resets.
    
[^253]: 基于合成监督与Mamba神经算法推理的在线复杂事件检测扩展方法

    Scaling Online Complex Event Detection with Synthetic Supervision and Mamba-Based Neural Algorithmic Reasoning

    [https://arxiv.org/abs/2502.07250](https://arxiv.org/abs/2502.07250)

    该论文提出一种结合合成监督与基于Mamba架构的神经算法推理框架，通过将复杂事件规则学习与执行解耦，克服了长时依赖、稀疏监督和标注困难等挑战，实现了在线流式场景下可扩展的复杂事件检测。

    

    现代机器学习模型擅长从短时、局部的观测中检测单个动作、声音或场景属性。然而，许多现实世界的任务，如智慧城市和医疗保健领域，需要对高层复杂事件（CE）进行推理：复杂事件是由短期原子事件（AE）构成的、受规则支配的时空模式。复杂事件检测（CED）极具挑战性，其原因包括长时依赖性、需要超越训练范围进行泛化、复杂事件级别的监督稀疏且缺乏时间对齐的细粒度原子事件标签，以及标注的认知负担沉重——因为复杂事件标签通常依赖于顺序、持续时间、否定和完成时间语义。在一个需要因果性流式推理且计算资源有限的在线环境中，这些挑战被进一步放大。我们将在线复杂事件检测的主要瓶颈确定为学习鲁棒的复杂事件规则，并提出了一种神经算法推理（Neural Algorithmic Reasoning）框架，该框架将规则学习与执行解耦。

    arXiv:2502.07250v3 Announce Type: replace  Abstract: Modern machine learning models excel at detecting individual actions, sounds, or scene attributes from short, localized observations. However, many real-world tasks, such as in smart cities and healthcare, require reasoning over high-level complex events (CEs): spatiotemporal, rule-governed patterns of short-term atomic events (AEs). Complex event detection (CED) is challenging due to long temporal dependencies, generalization beyond the training horizon, sparse CE-level supervision without temporally aligned fine-grained AE labels, and cognitively demanding annotation, as CE labels often depend on ordering, duration, negation, and completion-time semantics. These challenges are further amplified in an online setting that requires causal, streaming inference with limited computation. We identify the primary bottleneck in online CED as learning robust CE rules, and propose a Neural Algorithmic Reasoning framework that decouples rule l
    
[^254]: 机器学习基准测试中聚合性能指标的统计不确定性量化

    Statistical Uncertainty Quantification for Aggregate Performance Metrics in Machine Learning Benchmarks

    [https://arxiv.org/abs/2501.04234](https://arxiv.org/abs/2501.04234)

    该论文展示了如何利用自助法和贝叶斯分层建模等统计方法，来量化机器学习基准测试中跨多个任务聚合的性能指标的不确定性。

    

    现代人工智能由机器学习模型（如基础模型）支撑，这些模型在海量数据语料库上进行预训练，然后被适配以解决各种下游任务。为了总结跨多个任务的性能，评估指标通常被聚合为一个汇总指标，例如跨10个问答任务的平均准确率。在聚合评估指标时，将不确定性纳入聚合指标中是有益的，以便更真实地理解模型性能。我们在这项工作中的目标是展示如何运用统计方法来量化跨多个任务聚合的指标的不确定性。我们重点强调的方法包括自助法（bootstrap）、贝叶斯分层（即多层）建模，以及考虑标准误差的任务权重可视化。这些技术揭示了诸如某种任务占主导地位等洞见

    arXiv:2501.04234v2 Announce Type: replace-cross  Abstract: Modern artificial intelligence is supported by machine learning models (e.g., foundation models) that are pretrained on a massive data corpus and then adapted to solve a variety of downstream tasks. To summarize performance across multiple tasks, evaluation metrics are often aggregated into a summary metric, e.g., average accuracy across 10 question-answering tasks. When aggregating evaluation metrics, it is useful to incorporate uncertainty in the aggregate metric in order to gain a more realistic understanding of model performance. Our objective in this work is to demonstrate how statistical methodology can be used for quantifying uncertainty in metrics that have been aggregated across multiple tasks. The methods we emphasize are bootstrapping, Bayesian hierarchical (i.e., multilevel) modeling, and the visualization of task weightings that consider standard errors. These techniques reveal insights such as the dominance of a s
    
[^255]: 赌博机中的满意遗憾最小化：常数速率与轻尾分布

    Satisficing Regret Minimization in Bandits: Constant Rate and Light-Tailed Distribution

    [https://arxiv.org/abs/2406.06802](https://arxiv.org/abs/2406.06802)

    本文提出SELECT算法模板，通过采样与下置信界检验，在赌博机满意遗憾最小化问题中实现了常数级别的期望满意遗憾，并同时具备标准遗憾保证。

    

    受决策制定中“满意”概念的启发，我们研究了赌博机优化中满意遗憾最小化的问题。在该设定下，学习者的目标是尽可能频繁地选择满意臂（即平均奖励超过某个阈值的臂）。性能通过满意遗憾来衡量，即所选臂的平均奖励相对于阈值的累计不足量。我们提出了SELECT，这是一个通过采样和下置信界检验来实现满意遗憾最小化的通用算法模板，在可实现情形下（即存在满意臂时），该算法能够为多种赌博机优化问题实现常数级别的期望满意遗憾。作为补充，在不可实现情形下，SELECT也能享有与预言机相同的标准遗憾保证。为了进一步提高算法的稳定性，我们引入了SELECT-LITE，它实现了轻尾……

    arXiv:2406.06802v4 Announce Type: replace-cross  Abstract: Motivated by the concept of satisficing in decision-making, we consider the problem of satisficing regret minimization in bandit optimization. In this setting, the learner aims at selecting satisficing arms (arms with mean reward exceeding a certain threshold value) as frequently as possible. The performance is measured by satisficing regret, which is the cumulative deficit of the chosen arm's mean reward compared to the threshold. We propose SELECT, a general algorithmic template for Satisficing REgret Minimization via SampLing and LowEr Confidence bound Testing, that attains constant expected satisficing regret for a wide variety of bandit optimization problems in the realizable case (i.e., a satisficing arm exists). As a complement, SELECT also enjoys the same (standard) regret guarantee as the oracle in the non-realizable case. To further ensure stability of the algorithm, we introduce SELECT-LITE that achieves a light-tail
    
[^256]: PAPER-HILT：个性化和自适应隐私感知的强化学习提前退出在人机协同系统中的应用

    PAPER-HILT: Personalized and Adaptive Privacy-Aware Early-Exit for Reinforcement Learning in Human-in-the-Loop Systems

    [https://arxiv.org/abs/2403.05864](https://arxiv.org/abs/2403.05864)

    PAPER-HILT是针对人机协同系统中隐私保护的创新自适应强化学习策略，通过提前退出方法动态调整隐私保护和系统效用，以适应个体行为模式和偏好。

    

    强化学习（RL）日益成为人机协同（HITL）应用中的首选方法，因其适应于人类交互的动态特性。然而，在这种环境中整合RL会带来重大的隐私问题，可能会不经意地暴露敏感用户信息。为解决这一问题，我们的论文专注于开发PAPER-HILT，一种创新的自适应RL策略，通过利用专为HITL环境中隐私保护设计的提前退出方法。该方法动态调整隐私保护和系统效用之间的权衡，使其操作适应个人行为模式和偏好。我们主要强调面临处理人类行为的可变和不断发展的挑战，使得静态隐私模型失效。通过其应用，评估了PAPER-HILT的有效性。

    arXiv:2403.05864v1 Announce Type: new  Abstract: Reinforcement Learning (RL) has increasingly become a preferred method over traditional rule-based systems in diverse human-in-the-loop (HITL) applications due to its adaptability to the dynamic nature of human interactions. However, integrating RL in such settings raises significant privacy concerns, as it might inadvertently expose sensitive user information. Addressing this, our paper focuses on developing PAPER-HILT, an innovative, adaptive RL strategy through exploiting an early-exit approach designed explicitly for privacy preservation in HITL environments. This approach dynamically adjusts the tradeoff between privacy protection and system utility, tailoring its operation to individual behavioral patterns and preferences. We mainly highlight the challenge of dealing with the variable and evolving nature of human behavior, which renders static privacy models ineffective. PAPER-HILT's effectiveness is evaluated through its applicati
    
[^257]: 保护您的分数：具有差分隐私保障的接触追踪

    Protect Your Score: Contact Tracing With Differential Privacy Guarantees

    [https://arxiv.org/abs/2312.11581](https://arxiv.org/abs/2312.11581)

    这篇论文提出了具有差分隐私保障的接触追踪算法，以解决隐私问题限制接触追踪的部署。该算法在多种情景下展现了卓越性能，并通过在发布每个风险分数时保护个体健康状况的隐私。

    

    2020年和2021年的流行病对经济和社会产生了巨大的影响，研究表明，接触追踪算法可以在早期遏制病毒方面起到关键作用。尽管在更有效的接触追踪算法方面已经取得了重大进展，但我们认为目前的隐私问题阻碍了其部署。接触追踪算法的本质在于传递一个风险分数的通信。然而，恰恰是将这个分数传递给用户，对手可以利用这个分数来评估个体的私人健康状况。我们确定了一个现实的攻击场景，并针对这种攻击提出了具有差分隐私保障的接触追踪算法。该算法在两个最常用的基于代理的COVID19模拟器上进行了测试，并在各种情景下展现了卓越性能，特别是在逼真的测试场景中，同时发布每个风险分数时。

    arXiv:2312.11581v2 Announce Type: replace-cross  Abstract: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score w
    
[^258]: 合成瞬时效应：将合成控制方法推广至动态处理效应

    Synthetic Blips: Generalizing Synthetic Controls for Dynamic Treatment Effects

    [https://arxiv.org/abs/2210.11003](https://arxiv.org/abs/2210.11003)

    该论文提出“合成瞬时效应”方法，通过反向归纳将目标单元在各时期的处理瞬时效应递归表示为其他单元瞬时效应的线性组合，从而将合成控制方法推广至具有潜在时变混杂的自适应多处理动态场景，实现任意干预序列下单元均值结果的识别与估计。

    

    我们提出了合成控制方法的一种推广，适用于存在动态处理效应的设定。在该设定中，每个单元按照一个依赖于潜在的、内生的时变混杂状态的自适应策略，依次接受多个处理。在低秩潜因子模型假设下（线性时变和时不变的动态三角系统均可作为其特例），我们为任一单元在任意干预序列下的均值结果建立了一种识别策略。我们的方法被称为“合成瞬时效应”，它是一种反向归纳过程：目标单元在每一时期某一处理的瞬时效应被递归地表示为其他接受该指定处理的单元瞬时效应的线性组合，从而避免了朴素合成控制扩展所要求的组合式供体条件。我们提供了易于实现的估计算法……

    arXiv:2210.11003v3 Announce Type: replace-cross  Abstract: We propose a generalization of the synthetic control methods to the setting with dynamic treatment effects, in which each unit receives multiple treatments sequentially, according to an adaptive policy that depends on a latent, endogenously time-varying confounding state. Under a low-rank latent factor model assumption, which admits linear time-varying and time-invariant dynamic triangular systems as special cases, we develop an identification strategy for any unit-specific mean outcome under any sequence of interventions. Our method, which we term synthetic blips, is a backward induction process in which the blip effect of a treatment at each period for a target unit is recursively expressed as a linear combination of the blip effects of other units that received the designated treatment, avoiding the combinatorial donor requirements of naive synthetic control extensions. We provide easy-to-implement estimation algorithms that
    
[^259]: 分布鲁棒的迁移学习

    Distributionally Robust Transfer Learning. (arXiv:2309.06534v1 [cs.LG])

    [http://arxiv.org/abs/2309.06534](http://arxiv.org/abs/2309.06534)

    这篇论文介绍了一种分布鲁棒的迁移学习方法，通过优化一个不确定性集合内最具对抗性的损失来实现，该集合是由源分布的凸组合生成的目标人口集合，能够有效地将迁移学习和分布鲁棒的预测模型联系起来。

    

    许多现有的迁移学习方法依赖于利用与目标数据相似的源数据的信息。然而，这种方法经常忽视了可能存在于不同但潜在相关的辅助样本中的有价值的知识。当处理有限的目标数据和多样化的源模型时，我们的论文引入了一种新颖的方法，分布鲁棒迁移学习（TransDRO），它摆脱了严格的相似性约束。TransDRO通过在一个不确定性集合内优化最具对抗性的损失来设计，该集合定义为由源分布的凸组合生成的目标人口的集合，保证了对目标数据的出色预测性能。TransDRO有效地将迁移学习和分布鲁棒的预测模型联系起来。我们建立了TransDRO的可辨识性和其作为最接近源模型的加权平均值的解释。

    Many existing transfer learning methods rely on leveraging information from source data that closely resembles the target data. However, this approach often overlooks valuable knowledge that may be present in different yet potentially related auxiliary samples. When dealing with a limited amount of target data and a diverse range of source models, our paper introduces a novel approach, Distributionally Robust Optimization for Transfer Learning (TransDRO), that breaks free from strict similarity constraints. TransDRO is designed to optimize the most adversarial loss within an uncertainty set, defined as a collection of target populations generated as a convex combination of source distributions that guarantee excellent prediction performances for the target data. TransDRO effectively bridges the realms of transfer learning and distributional robustness prediction models. We establish the identifiability of TransDRO and its interpretation as a weighted average of source models closest to
    

