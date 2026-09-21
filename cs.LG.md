# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BrainWideBench: Benchmarking large-scale pretraining and across-animal transfer in multi-region neural recordings](https://arxiv.org/abs/2609.22064) | 提出了BrainWideBench基准，基于139只小鼠、276个脑区的神经与行为记录数据，通过行为解码、神经活动预测等多任务套件，系统评估多脑区神经记录上的大规模预训练与跨动物迁移能力。 |
| [^2] | [Predictable Failure in Multi-Hop Retrieval: Score-Distributional Confidence Scoring and Abstention](https://arxiv.org/abs/2609.22056) | 该论文证明多跳检索失败集中在结构上可预测的子群体中，并提出RegimeAbstain框架，通过融合多种ANN分数特征的检索置信度评分（RCS）实现有原则的弃权决策，从而可证明地减少高置信失败。 |
| [^3] | [Benchmarking World Models for Continual Learning on Compositional Tasks](https://arxiv.org/abs/2609.22055) | 该论文提出了一个面向机器人操作的组合式持续学习基准，通过将新任务设计为已见任务的组合，把世界模型的知识复用能力与学习新任务的能力分离开来加以评估。 |
| [^4] | [Particle Competition and Cooperation for Robust Graph Convolutional Network Learning Under Label Noise](https://arxiv.org/abs/2609.22053) | 该论文提出PCC+GCN混合框架，利用粒子竞争与合作的粒子支配动力学在GCN训练前识别并精炼可疑标签（保留、移除或重新分配），从而在多种标签噪声下实现鲁棒的图卷积网络学习。 |
| [^5] | [Available Guardrails: Certifying Selective Prediction across ML Systems](https://arxiv.org/abs/2609.22048) | 该论文提出了一种可计算的“可用性”框架，利用精确二项反演和动态规划为跨机器学习系统的选择性预测提供认证，揭示了安全性、报告粒度与服务流量之间的权衡。 |
| [^6] | [$\lambda$-Controlled GRPO: Turning Flow-Matching Ratio Instability into a Budgeted Resource](https://arxiv.org/abs/2609.22041) | 该论文发现流匹配GRPO训练中的多种不稳定性（重要性比率漂移、分散与裁剪）均源于一个单一的每步量——路径方差，它由采样器的高斯转移核精确决定且可廉价估计，从而将不稳定性转化为一种可测量、可预算的资源。 |
| [^7] | [COMPLEX: A Closed-Form Certified Embedding of Multiparameter Persistence Modules](https://arxiv.org/abs/2609.22012) | COMPLEX通过闭式、免训练的嵌入方法为多参数持久模块补齐了以往缺失的Lipschitz下界，给出了首个多参数特征映射的双侧失真界，使特征忠实性变得可度量。 |
| [^8] | [Abstention and Noise Filtering: Two Missing Primitives of Softmax Attention](https://arxiv.org/abs/2609.22005) | 本文论证并通过实验证明，注意力值通路的门控为softmax注意力补充了两种缺失的基本要素——弃权（允许注意力头输出空值）和噪声过滤（抑制残差流中叠加特征的干扰），并在10M至350M参数的匹配模型上提供了实证支持。 |
| [^9] | [Assessment of Machine Learning-Based Critical Heat Flux Models in the CTF Subchannel Code for Square Rod Bundle Prediction](https://arxiv.org/abs/2609.21995) | 本研究利用EPRI棒束数据库评估了部署在CTF子通道代码中的机器学习临界热流密度模型，发现基于圆管数据训练的ML模型能够良好迁移到方形棒束几何结构中并优于传统方法。 |
| [^10] | [Time series generation with spectrally aligned latent flow matching](https://arxiv.org/abs/2609.21989) | 本文提出一种谱对齐的潜在流匹配时间序列生成方法，通过引入基于傅里叶、小波和签名变换等可解释信号表示的微调损失，使潜在空间保留关键动力学特性，从而消除潜在压缩导致的频谱失配伪影。 |
| [^11] | [Multiplicative Optimism for Constant Regret in Games](https://arxiv.org/abs/2609.21976) | 提出乘性乐观遗憾匹配（MORM）学习规则，使有限一般和博弈中的自博弈玩家在全信息下仅需一步乐观即可在所有时间跨度上一致地获得 O(√n log d) 的低遗憾，并能抵御对抗性效用。 |
| [^12] | [Schedule optimization for tau-leaping in masked discrete diffusion](https://arxiv.org/abs/2609.21960) | 该论文通过依赖密度ρ的精确积分表示来刻画掩码离散扩散中tau-leaping采样的因式分解误差，并推导有限步优化问题的递归平稳性方程，从而实现去噪调度的优化。 |
| [^13] | [RACER: Role-Aligned Competence Estimation for Human-AI Routing](https://arxiv.org/abs/2609.21953) | RACER 提出一个角色相对的能力估计框架，通过估计未见专家在各候选类别角色下回答正确的后验预测概率并与模型后验结合，既避免了基于绝对类别坐标的路由捷径，又能捕捉实例级别的专家专长。 |
| [^14] | [Learning to Move Cities: Deep Meta-Models and Reinforcement Policies for Calibration and Control in Urban Networks](https://arxiv.org/abs/2609.21945) | 本文提出一个共享潜在空间框架，利用MLP自编码器学习城市交通动态的低维表示，将交通仿真器的贝叶斯优化校准与深度Q学习控制统一起来，显著提升了校准的样本效率。 |
| [^15] | [Guiding Agents of Quantum Games to Equilibrium using Matrix Exponential Fixed-Point Iteration](https://arxiv.org/abs/2609.21944) | 本文提出了带退火的矩阵指数不动点迭代（MEFPIA）算法，通过张量缩并表达式避免构造高维联合密度矩阵，从而高效计算扩展Gutoski-Watrous量子博弈中的均衡策略。 |
| [^16] | [End-to-End Hard-Label Cryptanalytic Model Extraction Using Efficient Sign Recovery](https://arxiv.org/abs/2609.21941) | 本文提出了一种基于全新原理的高效符号恢复算法，实现了对训练好的深度ReLU多层感知机在硬标签设置下的端到端黑盒模型提取攻击。 |
| [^17] | [Joint Remaining Useful Life Prediction and Capacity Estimation of Lithium-Ion Batteries Using Partial-Charging Data](https://arxiv.org/abs/2609.21932) | 本文提出一种基于部分充电数据的跨专家框架，通过RUL专家与容量专家双网络结构及特征级线性调制模块，在无需历史完整循环容量的条件下联合实现锂离子电池的剩余使用寿命预测与容量估计。 |
| [^18] | [Kinks vs. Smoothness: Identifiability of Real Analytic nICA for Laplace-like Sources](https://arxiv.org/abs/2609.21926) | 该论文证明了当独立源的密度函数一阶导数存在有限个不连续点（如拉普拉斯分布）时，实解析非线性独立成分分析（nICA）在排除平凡歧义后是可辨识的，其核心证明思路是利用源分布中的折点与实解析函数平滑性之间的对比。 |
| [^19] | [Riemannian Simultaneous Inference for Tangent Vector Field Regression](https://arxiv.org/abs/2609.21910) | 该论文针对无边黎曼流形上的切向量场回归提出了一种基于平行输运与体积校正的核估计方法，并通过单位切丛上的上确界表示与 Gumbel 极限理论，构造了回归场的可行同时置信管。 |
| [^20] | [Beyond Kinematics: Benchmarking Simulation Fidelity for Muscle-Driven Imitation Learning](https://arxiv.org/abs/2609.21909) | 该论文基于共同的人体动作捕捉与肌电测量数据，系统比较了SCONE/HyFyDy与MuJoCo/MyoSim两种肌肉驱动运动模仿强化学习流程，揭示两者虽能高保真再现人体运动学，但在捕捉肌肉激活、代谢成本等潜在神经肌肉行为方面存在差异，这对机器人辅助设备的设计与控制具有重要意义。 |
| [^21] | [Intervention Granularity Matters: Coherent Treatment Bundles in Counterfactual Simulation with Clinical World Models](https://arxiv.org/abs/2609.21906) | 该论文发现临床干预以不可分割的治疗束形式记录，并证明干预编辑的粒度会显著影响临床世界模型反事实模拟的预测结果。 |
| [^22] | [ExpBoN: Exponential-Noise Best-of-$n$ for Efficient Test-Time LLM Alignment](https://arxiv.org/abs/2609.21899) | 本文提出基于指数噪声报告-噪声-最大机制的软Best-of-n采样方法ExpBoN，具有精确有限n分解和指数级快速收敛的理论保证，并将其集成到引导式投机推理框架中（ExpGSI），实现了高效的奖励引导LLM测试时对齐。 |
| [^23] | [LLMs as Feature Engineers for Text-and-Tabular Prediction](https://arxiv.org/abs/2609.21894) | 该论文提出一种错误驱动的迭代框架，让LLM自动从非结构化文本中提取可解释的分类特征用于表格预测，通过将模型错误转化为自然语言反馈将特征发现速度提升3倍，且生成特征与TF-IDF和稠密嵌入形成强互补。 |
| [^24] | [Detecting Pretraining Data in Large Language Models from a Free-Energy Perspective](https://arxiv.org/abs/2609.21888) | 该论文提出能量转移检测（ETD）方法，通过结合预测损失与预测熵的倾斜边界，并从亥姆霍兹自由能的宏观视角进行解释，从而更准确地区分大语言模型中的预训练数据与非预训练数据。 |
| [^25] | [Near-Optimal Acceleration for Smooth $\ell_p$ / $\ell_q$ Nondual Convex First-Order Oracle Optimization](https://arxiv.org/abs/2609.21880) | 本文提出了针对 $\ell_p$ 球上 $\ell_q$ 范数 Hölder 光滑凸优化的近最优加速算法，在对数因子内解决了 COLT 2015 的公开问题，并在光滑情形下达到 $\widetilde{O}(LR^2/T^3)$ 的三次方收敛速率。 |
| [^26] | [Geometric Mean Pooling for Equal-Weight Multiplicative Coarse-Graining](https://arxiv.org/abs/2609.21876) | 提出一种无需可学习参数的带符号池化算子——几何平均池化（GMP），它通过结合特征符号乘积与幅值几何平均来保留乘性信息，在分层粗粒化中可保持全局乘性统计量，并在基于乘积信号的合成任务上优于平均池化和最大池化。 |
| [^27] | [Chronosphere: Space-Time Tessellation of Local Climate Experts](https://arxiv.org/abs/2609.21872) | Chronosphere提出了一种时空神经场，通过在时空环面上对可学习站点进行自适应镶嵌并结合共享的局部基函数库，使气候表征的容量与细节层次能随数据在空间和时间上自适应调整，其性能达到或超越最先进的位置编码器。 |
| [^28] | [Neural Cellular Automata Learn General Features in their Hidden Channels](https://arxiv.org/abs/2609.21870) | 该论文揭示了神经细胞自动机隐藏通道能学习通用特征，并提出一种将预训练教师模型隐藏状态注入学生模型的新型迁移学习机制，使模型仅用约9,800个参数就在小样本任务中实现了超越循环和前馈网络的泛化能力。 |
| [^29] | [AutoRecLab: Describe the Experiment, Get the Code!](https://arxiv.org/abs/2609.21863) | AutoRecLab是一个基于Python的自主推荐系统实验平台，能根据自然语言描述的研究想法自动推导实验需求、构建验证原型并迭代扩展为完整实验代码，演示中9次运行成功8次且每次成本仅约1美元。 |
| [^30] | [Watermarkable Multi-Draft Speculative Sampling via Poisson Processes](https://arxiv.org/abs/2609.21858) | 提出了一种基于泊松过程的多草稿投机采样算法，在保持高效采样的同时能够自然嵌入无偏水印而不降低投机接受率，突破了投机采样与水印技术难以兼得的权衡瓶颈。 |
| [^31] | [The Weight Is Over - Interactive Diffusion on Consumer GPUs](https://arxiv.org/abs/2609.21849) | 该论文提出嵌入翻译器、可复现的扩散流水线调优方案和端侧交互式图像生成编辑器三项贡献，在消费级GPU上实现了亚秒级首图时间的交互式扩散图像生成。 |
| [^32] | [Federated Deep Clustering Networks for High-Dimensional and Heterogeneous Data](https://arxiv.org/abs/2609.21829) | 提出FedDCN——将深度聚类网络泛化到联邦学习场景，通过同时优化重构损失与聚类损失，在客户端数据非独立同分布（异构）的条件下实现鲁棒的高维数据聚类。 |
| [^33] | [RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding](https://arxiv.org/abs/2609.21827) | 该论文提出RheoSampling方法，通过将树构建与token验证两个任务解耦，解决了随机解码下动态树投机解码中草稿分布坍缩为独热概率、接受率骤降的两难困境。 |
| [^34] | [Adaptive Uncertainty-Aware Modeling and Stochastic Radial Basis Function Predictive Control for Personalized Fluid Resuscitation](https://arxiv.org/abs/2609.21821) | 该论文提出了一种融合贝叶斯生理建模与最优控制的新型框架，通过不确定性感知的状态空间模型（UVAE-SSM与BNSSM）显式建模偶然不确定性与认知不确定性，并结合随机径向基函数模型预测控制算法，实现液体复苏过程中不确定性感知的个性化血流动力学调节。 |
| [^35] | [Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods](https://arxiv.org/abs/2609.21815) | 本文提出了一个针对矩阵值参数的通用在线镜像下降框架，通过引入按行和按列的自适应近端函数，推导出 Row-AdaGrad 和 Column-AdaGrad 两种优化器，将 AdaGrad 式的自适应次梯度方法推广到了具有矩阵结构的参数优化中。 |
| [^36] | [RegKT: Interpretable and Robust Deep Knowledge Tracing With IRT-Regularizer](https://arxiv.org/abs/2609.21791) | 本文提出RegKT，一种基于IRT正则化器的新型正则化技术，能够同时提升深度知识追踪模型的可解释性与鲁棒性并缓解过拟合问题，使其更适用于真实教育场景。 |
| [^37] | [From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention](https://arxiv.org/abs/2609.21788) | 该论文提出PARTS框架，利用冻结的预训练策略提供标称动作，并通过智能体生成的选择器和成功验证器提供局部奖励，将强化学习练习集中在长时程操作任务的关键瓶颈子任务上，从而在最少人工干预下实现现实世界的策略微调。 |
| [^38] | [Beyond Benchmark Scores: Auditing Medical Vision-Language Models for Chest X-Ray Tuberculosis Screening](https://arxiv.org/abs/2609.21763) | 该研究通过对三个医学视觉-语言模型在多数据集、多提示词族、不同阴性谱等条件下的系统审计，揭示了基准分数无法可靠反映模型在真实结核病筛查中的表现——模型排名和AUROC会随评估条件的改变而显著波动。 |
| [^39] | [Complete Neural Electronic Initialization Accelerates Materials DFT](https://arxiv.org/abs/2609.21759) | 该论文提出了首个完备的机器学习方法来加速PAW形式下的材料平面波DFT计算，通过形式化七项准则并引入AugNet——首个PAW增强占据数通用等变模型和首个材料通用自旋密度模型——填补了现有方法中缺失的关键初始化组件。 |
| [^40] | [Bilevel Optimization of Topology and Hyperparameters (BOTH)](https://arxiv.org/abs/2609.21758) | 该论文提出通过对拓扑优化过程本身进行自动微分来获得“超梯度”，从而在双层优化框架下将超参数调优与设计优化同步进行，免去繁琐且依赖经验的手动调参。 |
| [^41] | [GraphSkillEvo: Evolutionary Optimization of Graph-Structured Agent Skills](https://arxiv.org/abs/2609.21749) | 提出GraphSkillEvo方法，将LLM智能体技能表示为图结构形式（节点表示执行步骤及操作指导，有向边表示步骤间转换），并通过进化优化克服非结构化技能缺乏工作流指导和搜索空间过大的问题。 |
| [^42] | [Single-Loop Stochastic Projected Damped Extragradient Methods for Stochastic Nonconvex--(Strongly) Concave Minimax Optimization](https://arxiv.org/abs/2609.21747) | 提出了保持单循环结构的随机投影阻尼外梯度方法 SPDE 及其递归方差缩减变体 VR-SPDE，用于随机非凸—（强）凹极小极大优化，并同时为博弈平稳性和优化平稳性提供了更优的随机一阶预言机复杂度保证。 |
| [^43] | [GEM-MPC: Balancing Exploration and Exploitation through Expert-Guided Planning](https://arxiv.org/abs/2609.21735) | GEM-MPC是一种基于MPPI的强化学习方法，通过将克隆规划器的策略与KL正则化探索策略相结合，并引入门控先验蒸馏机制选择性地利用存储的规划分布，从而改善规划与学习的交互，在高维连续控制中平衡探索与利用。 |
| [^44] | [SpecQuant: Speculative Decoding with Multi-Parent Quantization for Adaptive LLM Inference](https://arxiv.org/abs/2609.21704) | SpecQuant 提出了一种无需训练的框架，通过从共享基础模型派生多种量化变体（INT4、FP8、FP16）并结合投机解码，根据任务复杂度动态路由查询，从而实现自适应且高效的大语言模型推理。 |
| [^45] | [Bayesian classification of astronomical spectra with class uncertainties](https://arxiv.org/abs/2609.21694) | 该论文面向4MOST巡天需求，系统比较了卷积神经网络、狄利克雷分布、蒙特卡洛dropout和贝叶斯神经网络+变分推断四种概率机器学习方法，实现了能够量化输入数据与预测不确定性的天文光谱多类别分类。 |
| [^46] | [Optimization Geometry of Equivalent Brownian RKHS Representations](https://arxiv.org/abs/2609.21693) | 本文在有限布朗RKHS框架下揭示了等价参数化（节点、增量、谱坐标）虽表示相同函数与内在范数却诱导不同优化几何的现象，证明节点与谱坐标的GD/SGD映射轨迹完全一致、增量GD为常数布朗/索伯列夫度量下因子为1/h的显式欧拉步，且增量海森矩阵的条件数不随网格分辨率增长。 |
| [^47] | [Multi-Domain Clustering via Measure Quantization](https://arxiv.org/abs/2609.21664) | 提出了一种通过测度量化进行多域聚类的通用框架，通过最小化概率度量学习共享聚类原型，并利用最优传输实现协作式样本分配，配合小批量优化策略在保持性能的同时提升了可扩展性，在多个多域基准测试中优于经典方法。 |
| [^48] | [Beyond Gaussian Worlds: Latent Geometry Matters for JEPAs](https://arxiv.org/abs/2609.21656) | 该论文将JEPA的可识别性理论从欧几里得高斯设定扩展到嵌入黎曼流形上的潜在变量，证明了当潜在变量均匀分布在球面上且表示匹配相同的球面分布时，每个最优表示都能在相差正交变换的意义上恢复潜在状态。 |
| [^49] | [Analysing the Linearity of Linguistic Relations in Language Model Embedding Spaces](https://arxiv.org/abs/2609.21655) | 该论文提出了一个基于约束线性逼近的框架来量化语言模型嵌入空间中各类语言关系的线性编码程度，发现屈折和派生关系近乎完美地线性编码，而词典和百科关系线性度明显较低，且RoBERTa和ModernBERT比GloVe更线性地编码关系。 |
| [^50] | [Configurable Multi-Stage Vision Pipeline for Crop Disease and Pest Diagnosis](https://arxiv.org/abs/2609.21651) | 针对小农户仅凭一张田间照片进行作物诊断的场景，本文提出一种可配置的多阶段视觉流水线，支持调节照片质量拒绝阈值与置信度截止值、并可扩展添加作物与病虫害类别，同时基于来自四个国家的116万张真实照片揭示了现有生产系统的不足。 |
| [^51] | [Riemannian Neural Hamiltonian Flows: Geodesic Symplectic Transport and Interpretability](https://arxiv.org/abs/2609.21647) | 该论文提出黎曼神经哈密顿流，将黎曼流形上的固定动能、可学习标量势与显式测地线蛙跳积分器相结合，并从理论上揭示了学习到的哈密顿势的可解释性机制。 |
| [^52] | [Detection is solved, delineation is not: what governs tooth segmentation on panoramic radiographs](https://arxiv.org/abs/2609.21628) | 该研究通过构建包含1,422张全景X光片、4万余个专家描绘牙齿多边形的大规模数据集并进行系统消融，发现输入分辨率是决定牙齿边界描绘精度的主导因素，而模型架构在域内对性能几乎没有影响。 |
| [^53] | [Trading Depth for Time in Recurrent Transformers](https://arxiv.org/abs/2609.21605) | 该论文提出潜在循环Transformer（LRT），在词汇token之间插入共享骨干参数的潜在思考token，以受控方式比较了循环Transformer中额外计算用于时间递归（潜在思考步）与物理网络深度的效果差异。 |
| [^54] | [Periodic Neural Mapping for Unsteady Rotor-Blade Pressure and Aeroelastic Load Prediction](https://arxiv.org/abs/2609.21590) | 提出周期性傅里叶神经映射（p-FNM）神经算子框架，将时间周期性嵌入模型，可从运行工况和任意时刻独立预测涡轮转子叶片的非定常压力场，避免误差累积并保持时间连续性，从而实现高效的气动弹性载荷预测。 |
| [^55] | [Predictive Suppression Layers for Communication-Efficient Spiking Neural Networks](https://arxiv.org/abs/2609.21583) | 提出了一种面向脉冲神经网络的最小预测编码框架，通过预测抑制层利用残差幅值动态门控，仅转发不可预测的“惊讶”活动，从而显著降低层间通信冗余。 |
| [^56] | [Weighted Quantum Signal Processing: Low-Depth Polynomial Approximation with Applications to Kolmogorov-Arnold Networks](https://arxiv.org/abs/2609.21567) | 本文提出加权量子信号处理（WQSP），通过为中心旋转算子引入权重函数突破了QSP的电路深度瓶颈和奇偶性约束，在实现任意有界一元多项式时将所需参数数量从线性级降至指数级缩减，并展示了其在Kolmogorov-Arnold网络中的应用。 |
| [^57] | [On Repulsive and Attractive Teachers: Separating Correctness from Behavior in Self-Distillation](https://arxiv.org/abs/2609.21561) | 该论文揭示了策略上自蒸馏中特权信息会将正确性信号与行为变化纠缠在一起——吸引性蒸馏会抑制探索性推理并使回答更短更自信，而排斥性蒸馏会使回答变长、意外触发思考模式且训练不稳定——因此提出应在自蒸馏中将正确性与行为分离。 |
| [^58] | [OneBid: A Unified Auto-Bidding Foundation Model for Diverse oCPX Advertising Scenarios](https://arxiv.org/abs/2609.21550) | OneBid提出了一个统一的自动出价基础模型，通过从异构oCPX日志中学习可复用骨干网络并结合离线后训练实现场景适配，同时解决了多目标控制、严格延迟下的可扩展性以及安全离线策略改进三大挑战，将多样化的oCPX广告场景统一到单一模型中。 |
| [^59] | [Dual-Interest Sequential Product Recommendation With Multi-Granular SSM](https://arxiv.org/abs/2609.21548) | 提出 DSRec 模型，利用多粒度状态空间模型（SSM）将商品在长期与短期语义上下文中显式解耦为双兴趣表示，高效解决了序列推荐中的商品多义性与跨时间粒度动态行为建模问题。 |
| [^60] | [Purification and Regulation: Comorbidity-Aware Multi-Label Few-Shot Learning for Medical Image Classification](https://arxiv.org/abs/2609.21541) | 该论文提出了原型纯化与调控（PPR）框架，通过共病感知的原型纯化机制和类间原型距离调控，解决了医学图像多标签小样本学习中原型被无关疾病信息污染以及忽视疾病间固有相关性的两大关键问题。 |
| [^61] | [MACE: Memory-Agent Co-Evolution with Adaptive Memory Graphs for Multi-Agent Systems](https://arxiv.org/abs/2609.21533) | 提出MACE框架，通过自适应记忆图MemGoG将依赖关系组织为功能性记忆单元，并利用执行反馈实现记忆组织与智能体记忆使用的协同演化，从而提升多智能体系统中协作轨迹的复用与检索效果。 |
| [^62] | [OpenMAS-GCom. A Diagnostic Benchmark for Graph-enhanced Multi-Agent Systems](https://arxiv.org/abs/2609.21527) | 提出OpenMAS-GCom诊断基准，通过在固定任务、模型、提示词和预算的条件下对协作单元、通信链路、共享信息和执行规则进行单组件受控干预，将图增强多智能体系统的性能差异准确归因于具体的通信结构和角色分配。 |
| [^63] | [IncentRL: The Trade-Off Between Preference Guidance and Task Performance](https://arxiv.org/abs/2609.21525) | IncentRL框架通过在奖励中加入KL惩罚形式的偏好引导，并从理论上刻画其对外部任务价值的扰动界以及保持最优策略的条件，揭示了偏好引导与任务性能之间的权衡关系。 |
| [^64] | [What Must Survive? Exact Task-Information--State Frontiers for Resource-Sufficient Learning](https://arxiv.org/abs/2609.21523) | 该文刻画了在下游任务未完全确定时压缩系统所需状态的精确前沿：给定 K 种先行任务建议，最小保留状态等于将任务集划分为至多 K 块后各块联合任务算子最大秩的最小值，并证明寻找最优建议划分是强 NP 难的。 |
| [^65] | [ServeGuard: Verifiable, Bounded-Residual Confinement of Operator-Invisible Channels Without Revealing the Certified Read Factor](https://arxiv.org/abs/2609.21515) | 提出ServeGuard，通过让发布者构建仅经由安全监控器覆盖方向读取输入的适配器，并以零知识证明验证这一性质（且不泄露所认证的读取因子），从而在结构上消除监控器盲区中无法被检测的后门信道。 |
| [^66] | [Adaptive Rollout Truncation Based on Epistemic Uncertainty for Efficient Offline World Model Training](https://arxiv.org/abs/2609.21482) | 提出了一种由认知不确定性驱动的自适应轨迹展开截断策略，当不确定性超过预热阶段校准的阈值时即终止自回归展开，从而在提升长时程预测能力的同时降低计算成本并避免放大早期训练误差。 |
| [^67] | [Efficient Architecture Search under Leave-One-Subject-Out Evaluation](https://arxiv.org/abs/2609.21457) | 提出PainNAS方法，通过基于块且控制数据泄漏的方式在被试间共享架构搜索，将LOSO评估下的NAS搜索次数从N次降至B次（B远小于N），在BioVid热痛数据集上以大幅更少的参数和计算量实现了相当的疼痛评估准确率。 |
| [^68] | [Improving the Predictive Performance of Bootstrap Aggregating by Dirichlet Resampling](https://arxiv.org/abs/2609.21454) | 提出基于狄利克雷重采样的两种随机森林变体（DM和DW），通过浓度参数调节样本重加权以降低树间相关性，在几乎不增加运行时间的情况下提升了预测性能。 |
| [^69] | [Understanding LLM Quantization through Activation-Guided Compensation and Orthogonal Residuals](https://arxiv.org/abs/2609.21450) | 该论文将LLM量化误差精确分解为激活引导的权重补偿项和正交残差，并对残差进行理论界定，从而为Hadamard旋转、符号选择和通道缩放等量化变换设计提供了实用准则。 |
| [^70] | [FootQuery: Future-Touchdown-Guided Retrieval from Depth History for Perceptive Humanoid Locomotion](https://arxiv.org/abs/2609.21447) | 提出FootQuery框架，以每只脚预测的未来触地点作为查询，从历史深度观测中检索触地时已不可见的地形信息，从而提升人形机器人在复杂地形上的感知运动能力。 |
| [^71] | [Optimal Randomized Proper Online Learning](https://arxiv.org/abs/2609.21445) | 本文证明了随机化正规在线学习的最优期望错误界为 $O(\mathtt{L}(\mathcal{H}) \log T)$，将此前已知的 $O(\mathtt{L}(\mathcal{H}) \log^6 T)$ 界显著改进，并在最坏情况下达到通用常数因子内的最优。 |
| [^72] | [GVPO++: Group Variance Policy Optimization for LLM Post-Training and On-Policy Distillation](https://arxiv.org/abs/2609.21432) | GVPO通过将KL约束奖励最大化的解析解融入梯度加权方案，解决了GRPO因依赖重要性采样导致的训练不稳定问题，同时保证唯一最优解并支持灵活的采样分布，可用于大语言模型后训练与在线策略蒸馏。 |
| [^73] | [Decision-Focused Learning for Mean-Variance Portfolio Optimization via KKT-Based Reformulation](https://arxiv.org/abs/2609.21427) | 该论文提出了一种将KKT最优性条件融入单层优化的重构方法，使决策聚焦学习能够直接应用于均值-方差投资组合优化，从而解决了现有方法中预测模型训练与约束优化决策之间结构性不匹配的问题。 |
| [^74] | [Tracing the Evidence Behind Zero-Shot Time-Series Forecasting: A Source-First Taxonomy and Audit Framework](https://arxiv.org/abs/2609.21425) | 本文主张将零样本时间序列预测视为“证据可及性声明”，提出了一个以证据来源为先的分类法（区分冻结LLM先验复用、参数化时间序列预训练和检索增强外部记忆三种来源），并配套任务接口、预测对象与评分、预测时上下文和资源预算四项审计问题，构成完整的审计框架。 |
| [^75] | [Brownian Heads for Deep ReLU Representations: Activation Mass and the Cost of Same-Sample Selection](https://arxiv.org/abs/2609.21422) | 该论文提出“布朗头”复杂度分析框架，以激活质量给出深度ReLU表示的精确Rademacher复杂度界，并首次量化了在同一批样本上选择隐藏特征所产生的额外选择成本，将其与实现尺度分离开来。 |
| [^76] | [Prediction Dynamics in Depth-Recurrent Language Models](https://arxiv.org/abs/2609.21383) | 本文提出一种将深度循环语言模型的预测更新分解为共同平移、相对胜者的方向以及竞争者更新与分数差距配对的间隔刻画方法，解释了中间预测与最终预测一致但分数仍变化的机制，并据此将确定最终答案所需的计算深度额外减少22.5-34.4%。 |
| [^77] | [Probabilistic Forecasting of Business Process Executions with Neural Temporal Point Processes](https://arxiv.org/abs/2609.21382) | 该论文将业务流程执行预测重构为基于带标记时间点过程的生成式建模问题，通过Transformer编码器与混合解码器显式处理时间戳重复问题，从而在构造上提供可靠的预测分布。 |
| [^78] | [Knowledge-Graph-Augmented Chronos-2 for HEC-RAS Surrogate Forecasting](https://arxiv.org/abs/2609.21381) | KG-Chronos-2通过将冻结的时间序列基础模型Chronos-2与水力学工程知识（精确状态残差解码、图条件历史检索和输入对齐校正）相结合，显著提升了HEC-RAS水面高程代理预测精度，相比各类基线方法最高可将RMSE降低39.54%。 |
| [^79] | [Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models](https://arxiv.org/abs/2609.21363) | 该论文系统揭示了多模态大推理模型可通过视觉推理从照片精确推断用户地理位置的隐私威胁，指出拒绝式防护与像素空间扰动防御的不足，并提出了一种基于扩散模型的隐私泄露缓解方法。 |
| [^80] | [IntBMoE: Integrating Block-Level Conditioning into Expert Composition for Full-Participation Mixture-of-Experts](https://arxiv.org/abs/2609.21346) | IntBMoE提出一种块条件化的混合专家模型，通过将密集专家组合与稀疏块执行相结合，把参与度、执行量和参数具体化三个量解耦，从而在实现全专家参与的同时控制计算与内存开销。 |
| [^81] | [Routine Blood Tests Outperform CRP for Distinguishing Bacterial From Viral Infection in Children](https://arxiv.org/abs/2609.21332) | 常规血液检查（全血细胞计数）结合机器学习分类模型在区分儿童细菌性与病毒性感染方面优于CRP，有望减少不必要的抗生素使用并缓解抗菌素耐药性问题。 |
| [^82] | [Deep Reinforcement Learning with Buffered Quantile Objectives](https://arxiv.org/abs/2609.21327) | 提出了Deep-BQRL，一个无模型的分布式深度强化学习框架，利用缓冲分位数目标实现可解释的风险敏感决策，突破了以往基于模型方法仅适用于小型表格问题的局限。 |
| [^83] | [Sparse Identification for Automatic Large-Scale Screening: A Constraint-Aware Framework with Ultra Fast Decoding Algorithm](https://arxiv.org/abs/2609.21321) | 本文提出了逻辑筛查方法，这是一种超快速、准确且有理论保障的大规模筛查框架，仅用O(klogn)次合并检测即可识别所有阳性样本，且解码仅依赖简单的逻辑运算。 |
| [^84] | [Diagonalized Attention for Individualized Regression: Latent-Row Localization and Prediction](https://arxiv.org/abs/2609.21320) | 本文提出一种用于矩阵值协变量个体化稀疏回归的对角化注意力机制，通过查询-键分数定位每个样本特定的信号行并共享总体回归效应，其参数维度与样本量无关，且无需响应变量即可为新样本识别感兴趣的行。 |
| [^85] | [An Introduction to Compression-Based Machine Learning](https://arxiv.org/abs/2609.21309) | 本文系统调研并形式化了压缩与机器学习之间的双向转换关系，提出了一个经过实证验证的基于压缩的机器学习设计框架，其性能可与传统方法媲美，在恶意软件检测上优势尤为显著。 |
| [^86] | [Fast And Accurate Text Content File Type Identification](https://arxiv.org/abs/2609.21306) | 本文提出了一种用于识别文本内容文件（尤其是源代码）类型的神经网络模型，其平均准确率更高，速度比 Magika 快约四倍，且模型体积小 28%。 |
| [^87] | [Identifying Security Platform Product Abuse with Machine Learning](https://arxiv.org/abs/2609.21303) | 该论文首次提出并实际部署了一套基于机器学习的全系统防御方案，用于检测安全平台产品滥用，使滥用检测覆盖率提升35%、每月警报减少30%，并能自适应攻击者行为的变化。 |
| [^88] | [FairLMs: A Turnkey Library for Fairness in Language Models](https://arxiv.org/abs/2609.21296) | FairLMs是一个Python库，通过对模型能力和输入要求的显式声明，将语言模型公平性研究中的偏见测量、缓解方法应用和证据审查统一起来，提供33个指标、14个缓解组件、14个诊断工具以及多种架构和API的适配器。 |
| [^89] | [Multi-Subject Pretraining Enables Short-Calibration Personalization for Closed-Corpus Surface EMG Speech Decoding](https://arxiv.org/abs/2609.21288) | 该论文提出“检查点初始化+多受试者预训练+目标用户短校准微调”的流程，使表面肌电语音解码在每名用户不足半小时校准数据的情况下将字符错误率降至21.7%，显著优于无校准和直接微调方案。 |
| [^90] | [Hybrid GPU-CPU Retrieval for Personalized Search at Ultra-Large Scale](https://arxiv.org/abs/2609.21281) | 该论文提出一种混合GPU-CPU协同服务系统，通过高深度GPU通路与高广度CPU通路的编排设计，解决了超大规模个性化搜索中个性化深度与库存覆盖广度难以兼得的“个性化-规模悖论”。 |
| [^91] | [MIRCID: Inferred Hub-miRNAs Drive Cross-Task Improvements in Drug Mechanistic Modeling](https://arxiv.org/abs/2609.21280) | MIRCID框架通过HubmiRNet从L1000地标基因推断泛癌症中心miRNA，并证明miRNA特征增强比转录因子活性能在通路分类和药物作用机制检索任务中带来更一致的跨任务性能提升。 |
| [^92] | [How Many Humans Is a Judge Panel Worth?](https://arxiv.org/abs/2609.21277) | 该论文提出两种不同的“等效人类评判者数量”度量——谱残差多样性 ν_H 与分布平方误差 ν_MSE，发现同一组 32 个语言模型评审在三个 ChaosNLI 任务上分别相当于约 4.24–6.50 个和 2.30–3.75 个人类评判者，且更大的谱多样性并不保证更好的分布恢复。 |
| [^93] | [Programming AMD XDNA NPUs with Open-source Compiler Tools: A FlashAttention Case Study](https://arxiv.org/abs/2609.21264) | 本文通过FlashAttention案例研究，展示了如何利用开源编译器工具（IRON和MLIR-AIR）对AMD XDNA NPU进行编程，其中将三个注意力阶段融合为单一内核的设计在端到端执行中达到3.62 TFLOP/s，能效比同芯片集成GPU高5.3至7.2倍。 |
| [^94] | [Verify, Don't Trust: Agentic Model Development for Video Discovery Retrieval at Scale](https://arxiv.org/abs/2609.21257) | 提出了 EvoPilot——一种带人工把关的长周期在线自动研究方法，通过角色化智能体、持久化实验记录和确定性检查来保障结论的有效性，并在支撑视频发现产品 Video Deep Dive 的检索系统上完成了为期 37 天的大规模验证。 |
| [^95] | [From Trainability Diagnostics to Optimization Claims: Boundaries and Controls in Variational Quantum Optimization](https://arxiv.org/abs/2609.21243) | 该论文揭示了变分量子优化中“梯度可训练性”与“优化成功”之间的鸿沟，提出步级诊断并严格证明在固定量子态与更新范数下原始梯度即已最大化一阶下降，从而划定了各类梯度干预策略的能力边界。 |
| [^96] | [Multiclass Semantic Segmentation of Wildland Fire Images Using Context-Aware Centralized Copy-Paste Data Augmentation](https://arxiv.org/abs/2609.21241) | 本文提出一种上下文感知的集中式复制粘贴数据增强策略，通过将火焰仅粘贴到语义合理的区域，提升了小型多类野火数据集中增强样本的真实性与数据质量。 |
| [^97] | [Visual Navigation Transformer with Pose Attention](https://arxiv.org/abs/2609.21212) | 提出VNT-PA，一种以相机位姿作为位置编码的视觉导航Transformer规划器，使注意力基于关键帧间的位姿差异而非时间顺序，从而无需构建显式地图即可重用历史遍历经验，在HM3D点目标导航中达到93.3%成功率。 |
| [^98] | [Reliability-Centered Evaluation of Sparse Longitudinal CT Lesion-Size Forecasting with Conformal Interval Calibration and Gompertz-Inspired Regularization](https://arxiv.org/abs/2609.21197) | 本研究基于DeepLesion和DLT构建了包含129名患者205条轨迹的稀疏纵向CT病灶基准数据集，采用共形区间校准与Gompertz启发式正则化对病灶大小预测进行了以可靠性为中心的系统评估，发现各方法点预测精度相近但不确定性可靠性表现差异显著。 |
| [^99] | [SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190) | 该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。 |
| [^100] | [Implicit Rule Induction with Test-Time Task Embeddings in ARC-like Tasks](https://arxiv.org/abs/2609.21181) | 该论文提出了一种新颖的两步测试时训练协议（Embed-TTT），先仅微调任务嵌入再微调主干网络，使任务嵌入与底层任务规则更好地对齐，从而在ARC类基准上提升性能并实现对已知规则的准确线性探测。 |
| [^101] | [TierKV: Long-Context On-Device LLMs via Predictive Multi-Tier KV Caching](https://arxiv.org/abs/2609.21172) | TierKV提出预测性多层缓存优化方法（PMCO），在解码前根据预填充隐藏状态预测缓存需求，将KV缓存令牌智能分配到精确、低秩和闪存卸载三个层级，在设备内存与精度预算约束下实现移动端长上下文LLM推理。 |
| [^102] | [M2G-LLM: Enhancing Clinical Prediction via Multimodal Graph Reasoning and LLM Context Injection](https://arxiv.org/abs/2609.21164) | 该论文提出M2G-LLM框架，利用图神经网络建模患者就诊的时序关系并在相似患者间传播信息，将构建的多模态上下文向量注入大语言模型中间层，实现文本与非文本医疗数据的联合推理，从而提升临床预测性能。 |
| [^103] | [HMB-GAN: Hybrid Multi-B\'ezier GAN for Vector Shape Synthesis](https://arxiv.org/abs/2609.21158) | 该论文提出了端到端可微分的混合量子-经典生成对抗网络HMB-GAN，通过多段贝塞尔曲线拼接生成几何连续的闭合矢量形状，并发现量子生成器虽收敛更快、参数更少且点云指标略优，但受模拟器开销与硬件限制的制约。 |
| [^104] | [EnSol: an environment-aware graph neural network for molecular solubility prediction](https://arxiv.org/abs/2609.21151) | EnSol是一种环境感知的概率图神经网络框架，通过交叉注意力机制建模溶质-溶剂相互作用并直接纳入温度效应，从而实现对分子溶解度更准确且带不确定性估计的预测。 |
| [^105] | [Diverse and Adaptable Arm Coordination for Octopus-Crawling via Diffusion-Based Uncertainty-Aware Optimization](https://arxiv.org/abs/2609.21138) | 本文提出基于扩散的不确定性感知优化算法DUO，首次将扩散控制应用于接触丰富仿真中的软体多臂机器人，使模拟章鱼无需示教即可学习多样化的爬行协调模式，并证明协调多样性本身能促进鲁棒自适应。 |
| [^106] | [Layerwise Decoupling for Stable Structured Sparsification of Fully Connected Layers](https://arxiv.org/abs/2609.21126) | 该论文提出一种逐层解耦的全连接层结构化稀疏化方法，证明了其与联合惩罚在最优性上的等价性，并展示出比耦合方法更强的鲁棒性——正则化强度可用范围更宽、灾难性过度剪枝发生率更低。 |
| [^107] | [Signal-Centric Remote Sensing via Alternative Preprocessing and Acoustic Processing for ML-Driven Applications](https://arxiv.org/abs/2609.21123) | 本文提出用CSV格式数据替代传统图像表示来处理声呐遥感数据，使处理时间减少91.18%，同时提升了机器学习目标检测精度以及SNR、PSNR等评估指标。 |
| [^108] | [Talk to Me, Jarvis: An Open-Source Edge-Deployable Voice Assistant Framework for Autonomous Racecars](https://arxiv.org/abs/2609.21109) | 该论文提出了Jarvis，一个面向自动驾驶赛车的开源边缘可部署离线语音助手框架，通过对Mistral 7B模型进行领域特定微调实现低延迟的文本到指令分类，意图识别准确率达97.63%，性能优于更大的在线托管模型。 |
| [^109] | [REFINEPPO: Learning Continuous Control Policies by Iterative Action Refinement](https://arxiv.org/abs/2609.21108) | 提出迭代动作精化方法，让策略从初始动作出发通过一系列学习到的残差修正迭代地改进控制动作，突破了传统连续控制策略“单次前向直接输出动作”的局限，从而提升策略学习效果。 |
| [^110] | [Detecting Hallucination in LLMs: Tracing the Topological Signatures of Impaired Context Sharing](https://arxiv.org/abs/2609.21096) | 该论文通过分析注意力图中Forman-Ricci曲率等拓扑特征来捕捉因果生成过程中受损的上下文共享模式，提出了一种单次遍历的幻觉检测方法，在多个基准和不同LLM架构上均优于现有基线。 |
| [^111] | [Triply-Scalable Equivariant Gaussian Process Modeling](https://arxiv.org/abs/2609.21085) | 该论文提出了三重可扩展的等变高斯过程，通过等变稀疏变分推断与无积分等变核的结合，实现了兼具等变性、不确定性量化与大规模可扩展性的数据高效高斯过程建模。 |
| [^112] | [Toward individual-level calibration in affect recognition with perceptual adjustment queries](https://arxiv.org/abs/2609.21073) | 提出一种基于感知调整查询（PAQ）估计个体恰可察觉差（JND）的校准框架，能够消除面部情感识别任务中感知敏感性的个体差异，实现个体水平的任务难度均衡。 |
| [^113] | [FedeRage: Provably Convergent Agnostic Federated Learning under General Client Drift](https://arxiv.org/abs/2609.21057) | 该论文刻画了客户端参与完全未知且高度偏斜时FedAvg实际优化的随机目标函数，并在此基础上提出风险规避扩展算法FedeRage，实现了在一般客户端漂移下具有可证明收敛性的不可知联邦学习。 |
| [^114] | [Physically Based Rendering in the Latent Space](https://arxiv.org/abs/2609.21054) | 该论文提出在生成模型变分自编码器的潜在空间中直接进行基于物理的渲染，通过修改渲染方程并结合可微渲染器，实现了物理引导的可控内容生成。 |
| [^115] | [A Lightweight Plug-in Gate for Transformer-Based Time-Series Forecasters](https://arxiv.org/abs/2609.21044) | 本文提出一种轻量级即插即用的编码器前门控接口，通过为协变量表示单元分配sigmoid分数来调控外部变量进入基于Transformer的时间序列预测器，可作为TimeXer、iTransformer和PatchTST的模块在零额外调参条件下使用。 |
| [^116] | [Stiefel-AdamW: Geometry-Aware AdamW for Linear Factorization Blocks](https://arxiv.org/abs/2609.21039) | 提出Stiefel-AdamW优化器，通过将线性分解模块（如LoRA适配器、自注意力查询-键乘积）中的一个因子约束在Stiefel流形上，解决分解不唯一导致的训练不稳定和因子爆炸问题，可作为AdamW的近似即插即用替代方案。 |
| [^117] | [Scaling Discovery through Test-Time Communication](https://arxiv.org/abs/2609.21032) | 该研究发现多智能体测试时通信使 $k$ 个协作智能体达到 $4k$ 个独立智能体的成功率，优势随规模扩大而复合增长，且团队能可靠解决任何单个智能体都无法完成的任务。 |
| [^118] | [A Smoothed Discrepancy Principle for Random Feature Methods and Neural Networks](https://arxiv.org/abs/2609.21017) | 该论文提出了一种基于光滑化偏差原则的多尺度早停规则，在所有光滑度水平上实现完全自适应的最优统计性能，并能以数据驱动的方式同时确定最优停止时间、随机特征数量以及神经网络宽度。 |
| [^119] | [Fragment-Aware Vision Transformers for Fresco-Fragment Style Classification](https://arxiv.org/abs/2609.21012) | 该论文提出一种片段感知的视觉Transformer框架，通过前景引导掩码、基于图像修复的几何正则化和基于KL相似度的有监督对比学习，实现从不完整、不规则的壁画残片中进行艺术风格分类。 |
| [^120] | [On the Limits of Maximal Coding Rate Reduction for Out-of-Distribution Generalisation](https://arxiv.org/abs/2609.21001) | 该论文揭示了最大编码率降低（MCR²）目标在分布外泛化中的两个根本局限性，表明仅优化 MCR² 可能得到基于不稳定环境特征、虽达到全局编码最优却在分布偏移下完全无法预测的表示。 |
| [^121] | [Aggregated Posterior Predictive Checks for Generative Modeling](https://arxiv.org/abs/2609.20999) | 本文提出了聚合后验预测检验（APPC）这一新方法，用于检验生成模型中从聚合后验而非先验采样的两阶段合成数据生成流程，并从理论上证明了该检验渐近校准的充分条件。 |
| [^122] | [MOSAIC-SR: Transformer-Guided Symbolic Regression for Scientific Equation Recovery](https://arxiv.org/abs/2609.20997) | MOSAIC-SR利用预训练Transformer生成多个初始表达式草图来引导符号回归搜索，结合尺度感知常数优化与局部符号修复，解决了现有方法中结构搜索效率低与神经模型符号错误的难题，实现更准确高效的科学方程恢复。 |
| [^123] | [From Stress to Affect: Multimodal Deep Learning for Physiological Emotion Recognition Across Wearable Sensor Modalities](https://arxiv.org/abs/2609.20991) | 该研究在 WESAD 和 EmoWear 两个多模态可穿戴数据集上，系统比较了 LSTM、TCN 和 Transformer 在不同传感配置下的生理情绪识别性能，发现 Transformer 在多模态配置下取得最高准确率。 |
| [^124] | [ASGARD: Action-Space Guard for UAV Resilience via Reinforcement Learning](https://arxiv.org/abs/2609.20982) | 提出了ASGARD——一个两阶段师生框架，利用攻击相关的特权信息训练编码器和监测器，在运行时检测并校正被篡改的动作指令，使基于强化学习的无人机控制具备抵御动作空间攻击的韧性。 |
| [^125] | [Generative inversion for early ranking of competing geologic interpretations](https://arxiv.org/abs/2609.20978) | 该论文提出一种生成式反演工作流程，将相互竞争的地质解释转化为空间先验，利用文本到图像基础模型和变分自编码器，根据与水头观测的一致性对解释进行早期排序，从而在数据稀缺条件下辅助高后果地下决策。 |
| [^126] | [Complex Problem Solving in Large Language Models: A Statistical Control Survey and Diagnostic Framework](https://arxiv.org/abs/2609.20973) | 该综述的核心贡献是将大语言模型的复杂问题求解重新建模为一个对潜在解状态的序贯估计与决策（统计控制）问题，由控制器维护信念并决定提交、验证、分支、回滚或弃权，从而为早期错误放大、提示脆弱性和无法纠错等失效现象提供统一的诊断框架。 |
| [^127] | [From Switching to Dynamic Regret: A Simple Reduction via Unbiased Random Sequences](https://arxiv.org/abs/2609.20968) | 本文提出一个简单归约框架，通过构造每轮无偏、方差可控且切换次数可管理的辅助随机序列，将动态遗憾最小化转化为切换遗憾最小化，从而可直接利用现成的切换遗憾算法导出动态遗憾界。 |
| [^128] | [Efficient Bayes-Adaptive Reinforcement Learning with Temporal Logic Specifications](https://arxiv.org/abs/2609.20954) | 该论文提出一种将LTL任务的LDBA表示与环境的BAMDP表示同步的端到端模型强化学习算法，并结合BAMCP实现近似贝叶斯最优策略合成，在属性满足性和样本效率上优于传统非贝叶斯方法。 |
| [^129] | [When AI Reviews Train AI Reviewers: Scientific-Judgment Collapse and Mitigation](https://arxiv.org/abs/2609.20942) | 该论文首次在受控实验中揭示了AI评审数据的递归训练会导致“科学判断坍缩”——评分分布压缩与语义多样性下降，并提出开源系统TrustReviewer来缓解这一问题。 |
| [^130] | [Do Quantum Models Scale Like LLMs?](https://arxiv.org/abs/2609.20912) | 本文发现当里德堡原子量子系统处于临界点附近时，其Transformer模型的神经缩放定律与大语言模型最为相似，表明近临界量子数据的统计结构与自然语言最接近。 |
| [^131] | [Continuous Delayed-Memory Stochastic Gradient Descent and Continuous-Time Reinforcement Learning from History of Astrophysical Time Series Studies](https://arxiv.org/abs/2609.20906) | 本文提出了一种依赖过去迭代状态的连续延迟记忆随机梯度下降方法，以及一种无需求解HJB偏微分方程的连续时间策略梯度强化学习结构，用于建模类星体光变曲线等天体物理时间序列。 |
| [^132] | [Bio-MF: Low-Latency and High-Fidelity EEG-to-fNIRS Cross-Modal Generation for Hybrid Motor-Imagery Brain--Computer Interfaces](https://arxiv.org/abs/2609.20904) | 提出Bio-MF，一个无潜变量的单步MeanFlow框架，通过直接的信号空间x预测实现低延迟、高保真的EEG到fNIRS跨模态生成，克服了现有方法生成速度慢、依赖预训练以及单步生成保真度低的问题。 |
| [^133] | [Extreme classification: beating chance with one training example from each class](https://arxiv.org/abs/2609.20897) | 本文研究了一个极端的最小分类问题，证明当两个分布不同时，即使每类只有一个训练样本，也能通过基于最大均值差异（MMD）的随机化核规则等方法实现严格优于随机猜测的分类。 |
| [^134] | [Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding](https://arxiv.org/abs/2609.20888) | 提出弹性阈值注意力（ETA），一种端到端可训练的架构，通过从查询表示中预测动态上下文阈值，在不牺牲稠密模型质量的前提下实现长上下文解码的硬件加速，解决了KV缓存带来的内存带宽瓶颈。 |
| [^135] | [BI-Agent and BI-Bench: Towards Automating End-to-End Business Intelligence](https://arxiv.org/abs/2609.20886) | 该论文提出了 BI-Agent 智能体和首个端到端商业智能基准 BI-Bench，利用大语言模型自动完成数据准备与业务问题解答，实现 BI 工作流程的全面自动化。 |
| [^136] | [Sparse Priors for Efficient Distribution Learning](https://arxiv.org/abs/2609.20883) | 该论文提出“稀疏先验”类和“稀疏维度”的概念，证明在k-稀疏先验下分布学习的贝叶斯风险下界为Ω(√(k/n))并给出TV距离的匹配上界，从而突破了传统理论中依赖维度d的悲观样本复杂度保证。 |
| [^137] | [The Refutation Gap: Certifying Both Halves of an Optimality Claim](https://arxiv.org/abs/2609.20873) | 该论文提出了“反驳鸿沟”这一概念，指出电路最小化领域中最优性声明的下界（不存在更小的程序）缺乏可认证的证明，而组合优化领域已有成熟的认证算法和伪布尔证明日志技术，应当被引入以同时认证最优性声明的上界和下界。 |
| [^138] | [Automated Physics-Informed Neural-Networks-Based Calibration of Highly Segmented Silicon Telescopes](https://arxiv.org/abs/2609.20868) | 该论文提出了一种基于神经网络的物理信息自动校准框架，将高粒度硅望远镜阵列的校准表述为全局优化问题，在两体运动学约束下通过最小化重建激发能宽度同时确定探测器增益与几何修正，仅依赖实验数据即可实现完全自动化校准。 |
| [^139] | [Reconstruction of 4D Mitral Regurgitation Hemodynamics from Sparse Planar Data using Deep Operator Networks with Test-Time Adaptation](https://arxiv.org/abs/2609.20857) | 该研究提出一种结合测试时自适应的深度算子网络方法，能够仅从单个平面速度切片和边界压力轨迹等稀疏观测数据中重建四维二尖瓣反流血流动力学，为临床快速量化反流严重程度提供了新的技术路径。 |
| [^140] | [Rewarding Efficient Reasoning Improves Abstention on Underspecified Tasks in Reasoning Models](https://arxiv.org/abs/2609.20846) | 该研究提出一种新颖的GRPO奖励机制以鼓励高效推理，使4B大型推理模型在信息不完整任务上的弃答能力平均提升12.8%，同时保留原有回答能力，并使其推理行为更接近人类。 |
| [^141] | [Reading Less While Writing: A Closed-Form Bandwidth Dial for Streaming Multimodal Decoders](https://arxiv.org/abs/2609.20845) | 该论文提出ZENDAYA，用单一连续参数γ的闭式调度取代流式多模态解码器中固定的wait-k规则，将离线解码与实时流式解码统一为同一模型家族的两个端点，并以解析形式Ē(γ) ≈ 1/(1+γ)精确控制每个输出词消耗的源内容比例，实现可解释的延迟与带宽调节。 |
| [^142] | [VISPATH: Visual-Intent-Guided Path Reasoning for Multimodal Knowledge Graph Question Answering](https://arxiv.org/abs/2609.20843) | 提出VISPATH框架，通过视觉意图引导的路径推理，使多模态线索能够在多跳推理的中间步骤中持续发挥作用，突破了现有方法仅将多模态信息用于起始实体定位的局限，提升了多模态知识图谱问答的性能。 |
| [^143] | [PhysioBench: A Unified Benchmark for Physiological Signal Question Answering](https://arxiv.org/abs/2609.20836) | 该论文提出了PhysioBench——首个统一的生理信号问答基准，将22个公开数据集的标注整合为覆盖30个任务的6140万个问题，并系统评估了21个代表性模型，揭示现有模型在跨模态生理信号指令遵循方面均存在不足。 |
| [^144] | [Recursive Language Models Generalize Out of Domain](https://arxiv.org/abs/2609.20831) | 递归语言模型通过在隔离上下文中解决子任务，排除了标准CoT依赖子任务外上下文捷径的失败模式，从而在分布外泛化上表现更优，这表明实现真正推理仅覆盖正确规则是不够的。 |
| [^145] | [Beyond WER: Entity and Disfluency Recall in Accented Conversational ASR](https://arxiv.org/abs/2609.20828) | 该论文提出三阶段流水线，通过SQL启发式数据筛选和区域LoRA微调，将带口音英语会话语音识别的实体召回率和填充词召回率大幅提升，同时以十分之一的参数量媲美300亿参数模型。 |
| [^146] | [TALON: A Temporally Aware Longitudinal Framework for Radiology Report Generation](https://arxiv.org/abs/2609.20826) | 提出TALON框架，通过双通道时序融合模块自适应整合可变长度的患者病史，分别捕捉持续性发现与间期变化，从而提升放射学报告生成中的纵向比较能力。 |
| [^147] | [HERMES: Contrast-Aware Knowledge Graph Reasoning from Clinical Notes for Patient Outcome Prediction](https://arxiv.org/abs/2609.20825) | HERMES通过大语言模型从临床笔记中构建个性化知识图谱，利用对比逻辑建模显式捕捉时间动态与治疗失败等对比信息，并结合图注意力网络学习患者表示，从而在不依赖结构化数据的情况下提升患者结局预测效果。 |
| [^148] | [Parallelism, critical windows, and separations among diffusion language models](https://arxiv.org/abs/2609.20539) | 本文首次对掩码扩散、均匀扩散与高斯扩散三类主流扩散语言模型的并行生成能力进行了细粒度理论比较，证明均匀扩散和高斯扩散同样能以前向传播次数与分布对偶总相关（可远小于上下文长度）成比例的方式完成采样，而此前仅掩码扩散具备这一性质。 |
| [^149] | [Small Enough to Know Everything: The Fully-Enumerable Transformer as an Instrument for the Science of Delayed Generalization](https://arxiv.org/abs/2609.20166) | 本文提出将全可枚举任务上的微型Transformer作为研究延迟泛化（grokking）现象的科学仪器，其提供精确泛化上限、任务手术、全权重直接观察和生存时间统计四种独特能力，并通过预注册守恒研究验证小规模下发现的规律具有可迁移性。 |
| [^150] | [Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations](https://arxiv.org/abs/2609.19150) | 该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。 |
| [^151] | [Training-Adaptive Convolutional Sparse Coding via Information Bottleneck for Robust Visual Representation](https://arxiv.org/abs/2609.19122) | 该论文提出一种训练自适应的卷积稀疏编码框架，通过展开FISTA算法并将稀疏系数作为可学习变量，从信息瓶颈视角自动权衡信息保留与压缩，从而获得更鲁棒的视觉表示。 |
| [^152] | [Toward Composable Network Digital Twins: A Subgraph-Based Latency Prediction Study](https://arxiv.org/abs/2609.18704) | 本文提出一种可组合的网络数字孪生方法，将网络分解为可重用的子图单元孪生，并通过轻量级组合器聚合来预测每条路由的端到端时延，解决了现有方法单体化、难以适应拓扑和流量变化的问题。 |
| [^153] | [Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning](https://arxiv.org/abs/2609.18461) | 提出LGM神经符号框架，利用稀疏自编码器将长期记忆解耦到连续潜在空间，根据每个查询动态构建潜在图，从而克服现有静态图记忆框架和平面检索方法无法捕捉上下文相关关系的问题。 |
| [^154] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^155] | [QuanText: Protecting Dataset-Level Secrets in Textual Data Sharing](https://arxiv.org/abs/2609.17995) | QuanText提出了一种无需训练、与大语言模型无关的文本随机量化数据发布机制，能够在保护文本数据集中数据集级全局秘密（如敏感属性比例）的同时保持数据效用，弥补了差分隐私对聚合属性保护不足的缺陷。 |
| [^156] | [Modular Deep Learning Mechanisms for Auditable Next-Day Wildfire Spread Prediction](https://arxiv.org/abs/2609.17763) | 本文提出三种模块化深度学习增强机制（风坡条件化注意力偏置、物理特征检索增强的输出校正和火点条件化双流门控），实现了可审计、可解释的次日野火蔓延预测。 |
| [^157] | [Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches](https://arxiv.org/abs/2609.17652) | Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。 |
| [^158] | [Divergence Timing and Cumulative Disagreement under KV-Cache Eviction](https://arxiv.org/abs/2609.16617) | 该论文建立了KV缓存驱逐引发自回归生成分歧的理论框架，将累积token不匹配精确分解为首次分歧贡献与分歧后暴露量的乘积形式，并通过条件蒙特卡洛无偏估计及Llama-3.1与Qwen2.5模型实验验证了不同缓存压缩策略在分歧时机上的差异。 |
| [^159] | [Data Attribution at Scale via Influence Matrix Estimation](https://arxiv.org/abs/2609.15044) | 该论文将预算受限的数据归因问题转化为从少量测量中估计大型影响矩阵的问题，并据此提出了MAGE和SPEL两种算法，大幅降低了大规模数据归因的计算成本。 |
| [^160] | [Data-free On-policy Distillation](https://arxiv.org/abs/2609.14193) | 该研究发现在线策略蒸馏（OPD）对训练数据几乎不敏感——仅八个提示即可媲美1.7万题的数据集，且跨领域数据仍能保留90%以上的收益，表明OPD传递的是教师的推理方式而非具体知识。 |
| [^161] | [Write on Paper and Get the Online Digital Trace:\newline A New Era for Handwriting](https://arxiv.org/abs/2609.12702) | 该论文提出了一种结合传感器数字笔与先进AI算法的创新方案，无需特殊纸张或外部参考系统，即可实时重建在普通纸上书写的手写数字轨迹。 |
| [^162] | [HuRo: Robotizing Human Videos for Scalable VLA Pretraining](https://arxiv.org/abs/2609.10706) | 该论文开发了一套将异构人类视频转换为机器人对齐的观察与动作轨迹的机器人化流水线，并据此构建了包含约63万条机器人化片段和1.42亿帧的HuRo数据集，验证了机器人化人类视频能够为VLA策略预训练提供有效且可扩展的监督信号。 |
| [^163] | [EFQ-Softmax: Exp-Free Quantization for Softmax](https://arxiv.org/abs/2609.09721) | 提出 EFQ-Softmax，一种无需计算指数的低比特概率生成方法，直接将注意力分数映射为块缩放的 E2M1 操作数，消除了 softmax 中高精度概率生成与低比特矩阵计算之间的不匹配，从而加速 Transformer 推理。 |
| [^164] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^165] | [Certified Topological Interaction in Neural Representations: Class Disentanglement Is Mostly Pairwise](https://arxiv.org/abs/2609.08561) | 本文提出用交集欧拉示性数轮廓将神经表征中的类解缠度量为带统计认证的拓扑交互，发现解缠主要发生在成对类别之间、随深度分级且集中在训练早期。 |
| [^166] | [Attributing Cohen's d: Training Data Attribution for Disease-Related Effects in Normative Age Biomarkers](https://arxiv.org/abs/2609.07729) | 本文提出一种基于Cohen's d的闭式影响泛函，可直接将常模年龄模型中疾病相关的年龄差距效应归因到个体训练样本，并在UK Biobank上证明移除最有影响力的训练样本可显著提高疾病相关效应量。 |
| [^167] | [VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models](https://arxiv.org/abs/2609.04355) | 提出VLA-Precision框架，通过非对称协同自举算法和ACoB-Stream架构，同时解决VLA模型真实世界在线强化学习中价值信号不可靠与计算开销过大两大瓶颈，实现高效且精确的策略改进。 |
| [^168] | [Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation](https://arxiv.org/abs/2608.30044) | 提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。 |
| [^169] | [SafeStep: An Interactive Demonstration of Semantic Communication for Pedestrian Safety Monitoring](https://arxiv.org/abs/2608.27688) | SafeStep是一个基于浏览器的交互式语义通信演示平台，可从实时交通摄像头中提取行人信息进行安全监控，并展示了仅416万参数的Meta-VIB模型无需在线重训练即可在不同SNR、码长和AoI条件下泛化的优势。 |
| [^170] | [Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings](https://arxiv.org/abs/2608.26088) | 行星预测引擎是一个自主AI系统，能从自然语言查询直接端到端执行地理空间预测，通过智能数据选择和基础模型嵌入，自动整合多模态数据并搜索最优模型，以应对全球性挑战。 |
| [^171] | [Convex losses and their applications to SVM, SVR, and Shallow Neural Networks](https://arxiv.org/abs/2608.14288) | 本文提出了多种新的凸损失函数用于SVM和神经网络，证明了它们是标准损失的推广，但实验结果显示这些新损失并未显著提升泛化性能。 |
| [^172] | [Contrastive Concept Importance: Explaining Pairwise Class Decisions Through Automatically Extracted Concept Representations](https://arxiv.org/abs/2607.27904) | 提出对比性概念重要性方法，将目标类别与对比类别之间的logit间隔归因于自动提取的视觉概念，从而回答“为什么是P而不是Q”这一对比性问题。 |
| [^173] | [Emulating Cosmic Structure Formation with a Lagrangian Neural Cellular Automaton](https://arxiv.org/abs/2607.27320) | 提出了一种拉格朗日神经细胞自动机（LNCA）混合深度学习框架，通过在拉格朗日坐标系中迭代地平流计算图本身来跟随物质流动，从而以局部迭代的方式高效、精确且完全可微分地模拟宇宙结构形成，为场级宇宙学推断提供了实用的前向模型。 |
| [^174] | [Decision trees, Frobenius traces, and Weierstrass coefficients of elliptic curves](https://arxiv.org/abs/2607.24251) | 该论文通过在 LMFDB 数据库上训练决策树模型，首次发现并严格证明了椭圆曲线极小 Weierstrass 模型的系数可由其 $L$-函数的 Dirichlet 系数（即 Frobenius 迹）通过显式公式确定，并应用于椭圆曲线数据表的计算。 |
| [^175] | [Teaching LLMs to Self-Evolve: Cultivating Core Meta-Skills with Reinforcement Learning](https://arxiv.org/abs/2607.21971) | 提出MetaEvolve框架，通过数据合成、进化感知强化学习和推理时进化搜索来培养大语言模型结合环境反馈进行自我反思等元技能，从而实现迭代式自我进化。 |
| [^176] | [ANNLib: A Development Framework for Efficient Approximate Nearest Neighbor Search](https://arxiv.org/abs/2607.17582) | ANNLib通过将基于图的ANNS系统中的算法与数据结构组件解耦并独立优化，同时实现了高性能与灵活功能，支持过滤搜索、完全动态更新、快照历史查询和范围搜索等复杂场景。 |
| [^177] | [TVGL-CFM:Generating and Forecasting Time-Varying Trajectories of Dynamic Networks with Conditional Flow Matching](https://arxiv.org/abs/2607.16894) | 提出了TVGL-CFM统一生成框架，通过全局对数欧几里得微分同胚映射将SPD精度矩阵轨迹的黎曼流形问题转化为欧氏空间问题，利用基于Transformer的非自回归条件流匹配模型，在无需预先指定图的情况下实现动态网络时变轨迹的类条件生成与历史条件预测。 |
| [^178] | [Prompt-Driven Exploration](https://arxiv.org/abs/2607.08837) | 本文提出一种利用视觉-语言模型从强化学习展开视频中自动诊断并重写提示的方法，以实现对弱策略的全局探索，而无需依赖稀疏奖励。 |
| [^179] | [Tubular Neighbourhoods of Pfaffian Sets and Applications to Neural Networks](https://arxiv.org/abs/2607.08370) | 本文推导了光滑Pfaffian超曲面管状邻域体积的界，并以此获得了具有Pfaffian激活函数的神经网络分类器鲁棒性条件数的尾部概率界，在单隐层Sigmoid网络情形下给出了决策边界管状邻域关于宽度呈多项式级的界。 |
| [^180] | [The Binary Tree Mechanism is Optimal for Differentially Private Continual Counting](https://arxiv.org/abs/2607.00876) | 本文证明了在固定隐私参数下，差分隐私持续计数问题中误差对数据流长度的依赖是不可回避的下界（近似DP下 $\Omega(\log^{3/2} n)$，纯DP下 $\Omega(\log^2 n)$），从而确立了二叉树机制的最优性。 |
| [^181] | [Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval](https://arxiv.org/abs/2606.27401) | 本文对大规模代码检索第一阶段召回中的深度学习模型进行了多语言、多数据集基准测试，揭示了其在TB级代码库上的精度与可扩展性局限，并提出基于LLM的代码规范化与查询重写方案显著提升检索精度。 |
| [^182] | [Geometry-Aware Reinforcement Learning for 2D Irregular Nesting](https://arxiv.org/abs/2606.10611) | 该论文提出多边形Transformer（PoT）架构并与组合优化强化学习框架相结合，使智能体能从数据中自动学习几何先验，从而突破传统启发式求解器对多边形几何不敏感的局限，有效解决二维不规则排样问题。 |
| [^183] | [Post-Rejection Follow-up Sampling: Measuring Outcomes of Rejected Decisions in Algorithmic DEX Trading](https://arxiv.org/abs/2606.08228) | 本文提出拒绝后跟进采样（PRFS）这一观察性测量方法，通过独立跟踪子系统在与原扫描器相同的实时预言机路径上持续采样被拒绝代币的价格与流动性，从而首次在同一实时交易场所量化测量算法化DEX交易中被拒绝决策的前瞻性市场结果。 |
| [^184] | [Scaling Novel Graph Generation via Lightweight Structure-Guided Autoregressive Models](https://arxiv.org/abs/2606.04287) | 本文提出一种轻量级自回归图生成框架，通过结构引导的拓扑排序将图序列化以实现接近对数线性的生成复杂度，并结合两阶段训练策略减少过拟合、促进可控的新颖图生成。 |
| [^185] | [Survival Reinforcement Learning: Toward Scalable Self-Supervised RL](https://arxiv.org/abs/2605.31273) | 提出生存强化学习（SRL），一种基于在线分类的方法，通过最大化智能体在目标处的停留时间来绕过对比强化学习的结构约束并缓解开关式控制问题，在长时程运动任务上性能超越最先进的CRL达2至8倍。 |
| [^186] | [Symbolic Classification-Enabled LHC Limits Online BSM Global Fits](https://arxiv.org/abs/2605.22330) | 本研究通过符号回归技术推导近似表达式，首次实现了将LHC限制高效纳入BSM全局拟合的在线过程，克服了传统计算瓶颈。 |
| [^187] | [Exemplar Partitioning for Mechanistic Interpretability](https://arxiv.org/abs/2605.14347) | 提出了一种无监督的范例划分方法（EP），通过Voronoi划分大语言模型激活空间来构建可解释的特征字典，可用于解释和干预模型行为、追踪训练动态并检测隐藏概念。 |
| [^188] | [Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective](https://arxiv.org/abs/2605.12969) | 本文从对比视角重新审视GRPO，识别其目标层面的两个局限性，并提出ConSPO方法，通过使用长度归一化的序列对数概率作为回滚得分，改进策略优化效果。 |
| [^189] | [ISOMORPH: A Supply Chain Digital Twin for Simulation, Dataset Generation, and Forecasting Benchmarks](https://arxiv.org/abs/2605.12768) | 提出了首个公开的多级物流网络供应链数字孪生ISOMORPH，其参数可配置、动力学具有马尔可夫性并能再现牛鞭效应，同时发布了填补供应链物流空白的时间序列预测基准数据集。 |
| [^190] | [The critical slowing down in training diffusion models](https://arxiv.org/abs/2605.12597) | 该论文通过对高斯极限下O(n)模型的可解析分析，证明了训练扩散模型的分数模型时参数学习会出现临界慢化现象，且该慢化同样影响生成过程，说明临界点附近的采样困难在学习型生成模型中依然存在。 |
| [^191] | [Rhamba: Region-Aware Hybrid Attention-Mamba Framework for Self-Supervised Learning in Resting-State fMRI](https://arxiv.org/abs/2605.01240) | 提出Rhamba框架，将解剖学引导的区域感知掩码与混合Attention-Mamba架构相结合，用于静息态fMRI的自监督预训练，并在精神分裂症和ADHD等下游分类任务中验证了其有效性。 |
| [^192] | [How a Cooperative-Override Circuit Suppresses Nash Play in Large Language Models](https://arxiv.org/abs/2604.27167) | 该论文通过 logit-lens 与因果干预分析发现，大语言模型在囚徒困境中偏离纳什均衡、选择合作的行为源于残差流中一个单一且可因果控制的“合作覆盖”方向，该方向在模型最后几层对决策起决定性作用。 |
| [^193] | [Diagnostic-Guided Longitudinal Modeling for Forecasting Retinal Atrophy Progression](https://arxiv.org/abs/2604.16955) | 该论文提出一种任务自适应诊断方法，用于在纵向成像预测中实证判断应选择随机还是确定性生成模型，并将其应用于眼底自发荧光图像预测以预测视网膜萎缩进展。 |
| [^194] | [Representation Before Training: A Practical Benchmark for Generative Medical Event Model Tokenization](https://arxiv.org/abs/2604.16775) | 该研究对生成式医疗事件模型的分词方案进行了大规模系统性基准测试（156个模型），发现将医疗代码与检验数值十分位数融合的 token 表示在所有八类住院结局预测任务中均能带来性能提升。 |
| [^195] | [Stability Enhanced Gaussian Process Variational Autoencoders](https://arxiv.org/abs/2604.09331) | 本文提出稳定性增强的高斯过程变分自编码器（SEGP-VAE），通过从LTI系统定义推导先验并采用将搜索空间限制在半收缩系统集合内的完整无约束参数化，实现了利用高维视频数据间接、稳定且无约束地训练低维线性时不变系统。 |
| [^196] | [Amortized Filtering and Smoothing with Conditional Normalizing Flows](https://arxiv.org/abs/2604.07169) | 提出了一种基于条件归一化流的摊销贝叶斯滤波与平滑框架，通过联合训练共享循环摘要网络与两个条件归一化流，使训练好的模型能在不同观测序列和同化时间上复用，从而避免了每个同化步骤中重新估计分数或构建传输映射的开销。 |
| [^197] | [Transcriptomic Models for Immunotherapy Response Prediction Show Limited Cross-cohort Generalisability](https://arxiv.org/abs/2604.05478) | 本研究系统评估了九种最先进的转录组学免疫治疗反应预测模型，发现它们在独立外部数据集上的预测性能有限，表明现有模型的跨队列泛化能力不足，需要在临床应用前进一步改进。 |
| [^198] | [Offline Constrained RLHF with Multiple Preference Oracles](https://arxiv.org/abs/2604.00200) | 该论文提出了仅通过对偶求解的算法来处理多重偏好预言机下的离线约束式人类反馈强化学习，首次给出了有限样本性能保证，确保高概率满足约束，并可扩展至多约束和一般f-散度正则化情形。 |
| [^199] | [Nonnegative Matrix Factorization in the Component-Wise L1 Norm for Sparse Data](https://arxiv.org/abs/2603.29715) | 本文首次证明了L1-NMF即使在最简单的r=1情况下也是NP难问题，并从理论上揭示了数据稀疏性与L1-NMF因子稀疏性之间的内在联系。 |
| [^200] | [Learning Surrogate LPV State-Space Models with Uncertainty Quantification](https://arxiv.org/abs/2603.29532) | 本文提出一种贝叶斯方法，可直接从输入输出数据中联合估计LPV状态空间模型及其调度映射，并同时量化偶然不确定性与认知不确定性，为预测模型响应提供置信界。 |
| [^201] | [Near-Optimal Primal-Dual Algorithm for Learning Linear Mixture CMDPs with Adversarial Rewards](https://arxiv.org/abs/2603.27884) | 本文提出了首个针对具有对抗性奖励的线性混合约束马尔可夫决策过程（CMDP）的可证明高效算法，通过引入正则化对偶更新实现基于漂移的分析，达到了近最优的 $\widetilde{O}(\sqrt{d^2 H^3 K})$ 遗憾和约束违反界限。 |
| [^202] | [Causal Evidence that Language Models use Confidence to Drive Behavior](https://arxiv.org/abs/2603.22161) | 该研究通过四阶段实验范式提供了因果证据，证明语言模型确实利用内部置信度信号来驱动行为（如决定是否回答或弃权），激活引导实验显示增强或抑制置信度信号会相应地降低或提高弃权率。 |
| [^203] | [How do LLMs Compute Verbal Confidence](https://arxiv.org/abs/2603.17839) | 大语言模型在生成答案时会自动计算并缓存置信度表征，之后在需要口头表达置信度时再进行检索，而非在被询问时才即时计算。 |
| [^204] | [Taming the Adversary: A Cost-to-Disturbance Ratio Approach to Adversarial Reinforcement Learning](https://arxiv.org/abs/2603.12110) | 提出CoDRA框架，将控制器与对手的权衡表示为累积代价与累积平方扰动范数之比，并通过自归一化的actor-critic更新进行优化，从而在提升鲁棒性的同时避免名义性能崩溃。 |
| [^205] | [Data-Driven Integration Kernels for Interpretable Nonlocal Operator Learning](https://arxiv.org/abs/2603.10305) | 该论文提出数据驱动的积分核框架，通过将非局部信息聚合与局部非线性预测显式分离，使气候过程的非局部算子学习更具可解释性并降低过拟合风险。 |
| [^206] | [Neural ensemble Kalman filter: Data assimilation for compressible flows with shocks](https://arxiv.org/abs/2602.23461) | 提出神经集合卡尔曼滤波器，通过将含激波流动的预报集合映射到深度神经网络的参数空间并在该空间中执行数据同化，解决了经典EnKF在不确定激波附近因双峰预报分布违反高斯假设而性能不佳的问题。 |
| [^207] | [Position: A Dynamical Systems Perspective is Needed to Advance Time Series Modeling](https://arxiv.org/abs/2602.16864) | 该论文提出时间序列建模领域应引入动力学系统视角，通过动力学系统重构（DSR）方法从数据中推断潜在动力学系统的替代模型，从而不仅实现短期预测，还能预测观测系统的长期统计特性。 |
| [^208] | [Gradient-Stable Attention Heads Signal LLM Correctness](https://arxiv.org/abs/2602.13699) | 提出免训练方法HeadEntropy，利用softmax雅可比迹与2-Rényi熵的单调关系，通过测量注意力头对梯度更新的稳定性来预测LLM答案的正确性，达到0.736 AUROC并超越所有免训练基线。 |
| [^209] | [Near-Universal Multiplicative Updates for Nonnegative Einsum Factorization](https://arxiv.org/abs/2602.02759) | NNEinFact是一种基于einsum的乘法更新算法，用户仅需一个字符串即可指定并拟合任意可表示为张量收缩的非负张量分解模型，该算法收敛于损失驻点、支持缺失数据，并能在数秒内处理数亿规模条目的张量。 |
| [^210] | [Semantic Calibration Prevails Where Token Confidence Fails: Benchmarking Long-Form Scientific QA](https://arxiv.org/abs/2602.00279) | 该论文推出了首个面向长篇科学问答的不确定性量化校准大规模基准，发现指令微调会导致token概率极化、削弱token级置信度信号的可靠性，而语义校准方法在此场景下表现更优。 |
| [^211] | [CORDS: Continuous Representations of Discrete Structures](https://arxiv.org/abs/2601.21583) | CORDS通过可逆映射将数量未知的离散对象集合转换为连续的密度场和特征场，使模型能够在连续场空间中进行推断并精确解码回离散集合，从而解决了可变大小集合预测的难题。 |
| [^212] | [Learning to Advect: A Neural Semi-Lagrangian Architecture for Weather Forecasting](https://arxiv.org/abs/2601.21151) | 该论文提出一种受物理启发的神经半拉格朗日架构，将天气预测中的平流、扩散和反应过程解耦为专门算子，通过球面可微插值实现基于轨迹的输送，使网络能够高效学习长距离大气输送。 |
| [^213] | [Cross-Country Learning for National Infectious Disease Forecasting Using European Data](https://arxiv.org/abs/2601.20771) | 本文提出一种跨国学习方法，利用多个欧洲国家的时间序列数据训练单一模型来预测目标国家的传染病发病率，并通过塞浦路斯COVID-19病例预测的案例研究证明该方法能借助共享的疫情动态和扩大的训练集提升预测性能。 |
| [^214] | [jBOT: Semantic Jet Representation Clustering Emerges from Self-Distillation](https://arxiv.org/abs/2601.11719) | jBOT 通过结合粒子级与喷注级的自蒸馏预训练，使无标签喷注数据的表示空间中涌现出语义类别聚类，仅需简单的距离度量即可实现异常检测，并可通过微调提升分类性能。 |
| [^215] | [BEAT-Net: Injecting Biomimetic Spatio-Temporal Priors for Interpretable ECG Diagnosis](https://arxiv.org/abs/2601.07316) | BEAT-Net通过仿生设计的QRS波标记化和模拟心脏病学家诊断工作流程的分层架构，实现了数据高效、泛化性强且可解释的心电图自动诊断。 |
| [^216] | [Large Language Models As Shannon Lossy Compressors Not Solomonoff Induction Estimators: The Singularity Is Not Near Without Symbolic Model Synthesis in Program Space](https://arxiv.org/abs/2601.05280) | 该论文证明大语言模型基于交叉熵和下一词元目标的训练本质上是香农有损压缩而非所罗门诺夫归纳估计，因此若缺乏程序空间中的符号模型综合，实现真正自我改进的AI奇点不会临近。 |
| [^217] | [Boltzmann generators for amorphous particle systems](https://arxiv.org/abs/2512.16607) | 本研究针对平衡采样极其困难的非晶材料（玻璃），通过将所需的等变性直接嵌入黎曼结构中，开发了首个专为非晶粒子系统定制的玻尔兹曼生成器。 |
| [^218] | [Understanding Structural Representation in Foundation Models for Polymers](https://arxiv.org/abs/2512.11881) | 该研究提出了一种基于SMILES聚合物图表示（CPG）的化学语言基础模型，通过融入其他线性表示法缺失的聚合物结构特征与连接性信息，在30个聚合物性质基准数据集上取得卓越性能，证明了其作为语言基础模型中聚合物表示方法的稳健性。 |
| [^219] | [K2-V2: A 360-Open, Reasoning-Enhanced LLM](https://arxiv.org/abs/2512.06201) | K2-V2是一个从零构建的360度全开放推理增强大语言模型，性能超越Qwen2.5-72B并接近Qwen3-235B，其完整开放的训练数据与历史为社区的持续训练和推理适配提供了卓越基座。 |
| [^220] | [SteganoBackdoor: Evading Data-Poisoning Defenses via Steganographic Backdoors](https://arxiv.org/abs/2511.14301) | 提出SteganoBackdoor优化框架，通过自回归token替换将语义触发器种子转化为隐写式投毒样本，将后门载荷分散编码在普通token中，从而在保持语言流畅性的同时规避现有数据投毒防御的检测。 |
| [^221] | [PRIVET: PRoximIty leakage detection Via Extreme value Theory](https://arxiv.org/abs/2510.24233) | PRIVET利用最近邻距离上的极值统计，提出了一种通用的、与模态无关的样本级隐私泄漏检测算法，能够为每个合成样本赋予经校准的个体化邻近性泄漏分数。 |
| [^222] | [Provably Optimal Reinforcement Learning under Safety Filtering](https://arxiv.org/abs/2510.18082) | 本文首次证明使用足够宽松的安全过滤器来强制执行安全性不会降低强化学习的渐近最优性能，从而消除了安全过滤必然牺牲性能这一普遍误解。 |
| [^223] | [The Impact of Semantic Pairs on Self-Supervised Representation Learning](https://arxiv.org/abs/2510.08722) | 该论文通过从ImageNet-1K构建类别构成匹配的增强对基线与人工整理的语义对数据集，开展了首个受控实证研究，以分离并量化语义正样本对对自监督表征学习的影响。 |
| [^224] | [Transformers Discover Molecular Structure Without Graph Priors](https://arxiv.org/abs/2510.02259) | 该研究表明，Transformer模型无需嵌入图结构或几何局部性等物理先验，仅通过数据学习就能自动发现分子结构等物理模式。 |
| [^225] | [Fidel-TS: A High-Fidelity Multimodal Benchmark for Time Series Forecasting](https://arxiv.org/abs/2509.24789) | 提出了基于数据来源完整性、无泄露设计和结构清晰性原则构建的高保真大规模多模态时间序列预测基准 Fidel-TS，揭示了以往基准的局限性及模型评估中的潜在差异。 |
| [^226] | [A regret minimization approach to fixed-point iterations](https://arxiv.org/abs/2509.21653) | 本文提出了一种将遗憾最小化算法转换为不动点迭代的通用方案，推广了经典的Krasnoselskii-Mann迭代，并基于AdaGrad算法获得了收敛更快的新型自适应不动点迭代方法。 |
| [^227] | [Robust Mixture Models for Algorithmic Fairness Under Latent Heterogeneity](https://arxiv.org/abs/2509.17411) | 提出了ROME框架，将潜变量混合建模与分布鲁棒优化（DRO）相结合，在无需预先指定分组的情况下自动学习潜在的、交叉性的子群体结构，并同时优化最差群体的预测性能，从而提升算法公平性。 |
| [^228] | [Generalizing Beyond Suboptimality: Offline Reinforcement Learning Learns Effective Scheduling through Random Solutions](https://arxiv.org/abs/2509.10303) | 提出了离线强化学习算法CDQAC，通过结合分位数评论家与延迟策略更新，直接从静态的次优数据集中学习有效的车间调度策略，其性能达到或超越生成数据的启发式方法以及现有的离线和在线RL基线，同时具有极高的样本效率。 |
| [^229] | [CASE: Contrastive Activation for Class-Sensitive Explanations](https://arxiv.org/abs/2506.07327) | 该论文提出类别敏感性诊断测试，揭示了许多主流显著性方法无论类别标签如何都产生几乎相同解释的结构性缺陷，并据此提出对比解释方法CASE，可分离出对预测类别具有独特判别性的特征。 |
| [^230] | [Understanding In-context Learning of Addition via Activation Subspaces](https://arxiv.org/abs/2505.05145) | 该研究提出一种新方法，将语言模型加法少样本学习的机制定位到仅三个注意力头，并揭示其通过六维激活子空间实现。 |
| [^231] | [Trajectory Entropy Reinforcement Learning for Robust Robot Motor Skill Learning](https://arxiv.org/abs/2505.04193) | 该论文提出轨迹熵强化学习（TERL），通过最小化整个动作轨迹的熵向策略引入简洁性归纳偏置，从而学习到对环境轻微扰动更鲁棒的机器人运动技能。 |
| [^232] | [OverThink: Slowdown Attacks on Reasoning LLMs](https://arxiv.org/abs/2502.02542) | Overthink攻击通过向外部上下文注入单独看似无害的诱饵推理问题，迫使推理大语言模型生成多达数十倍的推理标记，从而显著增加计算成本，同时仍能产生正确答案并轻易绕过安全过滤器。 |
| [^233] | [dSTAR: Straggler Tolerant and Byzantine Resilient Distributed SGD](https://arxiv.org/abs/2412.07151) | dSTAR是一种轻量高效的分布式SGD方法，通过收集最先响应的前k个工作节点的梯度并利用集成中位数过滤偏差，同时缓解掉队者效应并抵御拜占庭攻击，且具有理论上的拜占庭韧性和线性收敛速率保证。 |
| [^234] | [Continuous Spiking Graph Neural Networks](https://arxiv.org/abs/2404.01897) | COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。 |
| [^235] | [Multi-Objective Hyperparameter Search via Damped Gauss--Newton Optimization](https://arxiv.org/abs/2401.03580) | 本文提出一种基于阻尼高斯-牛顿优化的多目标超参数搜索方法，利用有限差分雅可比矩阵和Tikhonov正则化实现有向的联合参数更新，在欠定情况下以远少于网格搜索的试验次数达到同等的最佳验证精度。 |
| [^236] | [Reinforcement Learning under External Influence: Guarantees, Algorithms, and Sample Complexity](https://arxiv.org/abs/2305.16056) | 本文研究了连续状态-动作空间中受外部过程非马尔可夫扰动影响的强化学习问题，提出了基于有限外生事件历史的策略迭代算法，并给出了可处理性条件、策略改进保证及样本复杂度分析。 |
| [^237] | [Rollout Total Correlation for Deep Reinforcement Learning](https://arxiv.org/abs/2209.05333) | 本文提出通过最大化智能体轨迹中所有学习表征与动作之间的展开总相关性来学习任务相关表征，并采用基于生成式与判别式模型的两个互补下界以及分块小批量技术来优化该目标。 |

# 详细

[^1]: BrainWideBench：多脑区神经记录中大规模预训练与跨动物迁移的基准测试

    BrainWideBench: Benchmarking large-scale pretraining and across-animal transfer in multi-region neural recordings

    [https://arxiv.org/abs/2609.22064](https://arxiv.org/abs/2609.22064)

    提出了BrainWideBench基准，基于139只小鼠、276个脑区的神经与行为记录数据，通过行为解码、神经活动预测等多任务套件，系统评估多脑区神经记录上的大规模预训练与跨动物迁移能力。

    

    大规模神经记录技术的进步使得跨多只动物、跨分布式脑区收集数据成为可能，由此引出一个问题：能否利用这种规模来学习可迁移至多种下游任务的通用神经表征。然而，实现这一目标的进展一直受到碎片化评估协议以及对单一任务领域狭窄关注的限制。在此，我们提出了BrainWideBench，一个用于评估多脑区神经记录上跨动物迁移能力的基准。该基准建立在国际脑实验室的Brainwide Map数据集之上，包含139只小鼠在执行感觉引导决策任务时跨越276个脑区的神经与行为记录。该基准围绕三个互补的任务套件组织，用于评估学习到的表征能否支持下游行为解码、能否预测被遮蔽或未来的神经活动，以及能否恢复生物学上有意义的（摘要在此处截断）

    arXiv:2609.22064v1 Announce Type: new  Abstract: Advances in large-scale neural recording have made it possible to collect data across many animals and distributed brain regions, raising the question of whether this scale can be exploited to learn general-purpose neural representations transferable across diverse downstream tasks. Yet, progress toward this goal has been limited by fragmented evaluation protocols and a narrow focus on individual task domains. Here, we present BrainWideBench, a benchmark for evaluating across-animal transfer on multi-region neural recordings, built on the International Brain Laboratory Brainwide Map dataset of neural and behavioral recordings spanning 276 brain regions from 139 mice performing a sensory-guided decision-making task. The benchmark is organized around three complementary task suites that evaluate whether learned representations support downstream decoding of behavior, can predict masked or future neural activity, and can recover biologicall
    
[^2]: 多跳检索中的可预测失败：基于分数分布的置信度评分与弃权机制

    Predictable Failure in Multi-Hop Retrieval: Score-Distributional Confidence Scoring and Abstention

    [https://arxiv.org/abs/2609.22056](https://arxiv.org/abs/2609.22056)

    该论文证明多跳检索失败集中在结构上可预测的子群体中，并提出RegimeAbstain框架，通过融合多种ANN分数特征的检索置信度评分（RCS）实现有原则的弃权决策，从而可证明地减少高置信失败。

    

    多跳检索的失败并非在所有查询上均匀分布：它们聚集在结构上可预测的子群体中。我们证明了两个形式化刻画这一结构的结果。第一（CWAR可约简性）：置信失败的减少当且仅当检索特征携带关于成功的互信息时才可实现，该条件在LLM评判流水线中得到满足，但在纯稠密检索设置中明显更弱，这解释了两种机制之间的AUC-AC差距。第二（特征机制互补性）：没有任何单一的ANN分数特征能在所有失败机制中取得最佳预测性能；主导特征因数据集而异（MuSiQue上为查询长度，HoVer上为hop-1集中度），并且一个构造性见证对表明每个特征在一种机制中是必要的，而在另一种机制中则无贡献。我们在RegimeAbstain中实例化这些原则，该系统计算检索置信度分数（RCS），即最多九个查询-ANN分数的逻辑函数……

    arXiv:2609.22056v1 Announce Type: cross  Abstract: Multi-hop retrieval failures are not uniformly distributed across queries: they cluster in structurally predictable subpopulations. We prove two results formalizing this structure. First (CWAR Reducibility): confident-failure reduction is achievable if and only if retrieval features carry mutual information about success, a condition satisfied by LLM-judge pipelines but substantially weaker in dense-only settings, explaining the AUC-AC gap between regimes. Second (Feature Regime Complementarity): no single ANN score feature achieves best predictive performance across all failure regimes; the dominant feature differs between datasets (query length on MuSiQue, hop-1 concentration on HoVer), and a constructive witness pair shows each is necessary in one regime and non-contributory in the other. We instantiate these principles in RegimeAbstain, which computes a Retrieval Confidence Score (RCS), a logistic function of up to nine query-ANN s
    
[^3]: 面向组合任务持续学习的世界模型基准测试

    Benchmarking World Models for Continual Learning on Compositional Tasks

    [https://arxiv.org/abs/2609.22055](https://arxiv.org/abs/2609.22055)

    该论文提出了一个面向机器人操作的组合式持续学习基准，通过将新任务设计为已见任务的组合，把世界模型的知识复用能力与学习新任务的能力分离开来加以评估。

    

    世界模型的一个理想特性是能够跨任务持续学习，即在不遗忘智能体已学内容的情况下适应新环境。特别是，保留并复用先前经验所获得知识的能力，是智能体高效适应新环境的基础，因为物理世界的动态往往可以用循环出现的机制来描述。然而，世界模型对适应能力的衡量混淆了两种能力：学习未见任务的速度与容量，以及对已获得知识的复用，因为新到来的任务既包含新颖内容，也包含重复出现的内容。为了将知识复用与先前经验分离开来，我们提出了一个面向机器人操作的世界模型组合式持续学习基准。具体而言，我们设计的每个任务课程由组合任务构成，这些任务结合了序列中已见任务的各个方面。我们进一步分解（原文摘要至此截断）

    arXiv:2609.22055v1 Announce Type: new  Abstract: A desirable property of a world model is the ability to learn continually across tasks, adapting to new environments without forgetting what the agent has already learnt. In particular, the ability to retain and reuse knowledge obtained from prior experiences underpins an agent's ability to efficiently adapt to novel environments, as the dynamics of the physical world can often be described in recurring mechanisms. However, the world model's measure of adaptation entangles two abilities: the speed and capacity to learn unseen tasks, and the reuse of knowledge already acquired, since incoming tasks carry novel content alongside what recurs. In order to isolate knowledge reuse from prior experiences, we propose a compositional continual learning benchmark for world models in robot manipulation. Specifically, we design each task curriculum with compositional tasks that combine aspects of the tasks seen in the sequence. We further factorise 
    
[^4]: 标签噪声下基于粒子竞争与合作的鲁棒图卷积网络学习

    Particle Competition and Cooperation for Robust Graph Convolutional Network Learning Under Label Noise

    [https://arxiv.org/abs/2609.22053](https://arxiv.org/abs/2609.22053)

    该论文提出PCC+GCN混合框架，利用粒子竞争与合作的粒子支配动力学在GCN训练前识别并精炼可疑标签（保留、移除或重新分配），从而在多种标签噪声下实现鲁棒的图卷积网络学习。

    

    图卷积网络（GCNs）对标签噪声高度敏感，因为被污染的监督信息会通过图结构传播，从而降低学习到的节点表示的质量。本工作提出了PCC+GCN，这是一种混合框架，在GCN训练之前使用粒子竞争与合作（PCC）作为基于图的标签精炼阶段。PCC通过粒子支配动力学识别可疑的已标记节点，并在GCN训练之前决定这些节点的标签应被保留、移除还是重新分配。该框架还允许将PCC所使用的图通过基于特征的k近邻边进行增强，而GCN本身则在原始图结构和节点特征上进行训练。所提出的方法在NoisyGL基准的十个图数据集上进行了评估，涵盖了常规的Uniform、Pair和Random标签噪声以及实例依赖标签噪声。此外，还进行了详细的超参数分析。

    arXiv:2609.22053v1 Announce Type: new  Abstract: Graph Convolutional Networks (GCNs) are highly sensitive to label noise, since corrupted supervision can propagate through the graph and degrade learned node representations. This work proposes PCC+GCN, a hybrid framework that uses Particle Competition and Cooperation (PCC) as a graph-based label-refinement stage before GCN training. PCC identifies suspicious labeled nodes through particle domination dynamics and determines whether their labels should be preserved, removed, or reassigned before GCN training. The framework also allows the graph used by PCC to be augmented with feature-based $k$-nearest-neighbor edges, while the GCN itself is trained on the original graph structure and node features. The proposed method was evaluated on ten graph datasets from the NoisyGL benchmark under conventional Uniform, Pair, and Random label noise, as well as under instance-dependent label noise. A detailed hyperparameter analysis was also conducted
    
[^5]: 可用的护栏：跨机器学习系统的选择性预测认证

    Available Guardrails: Certifying Selective Prediction across ML Systems

    [https://arxiv.org/abs/2609.22048](https://arxiv.org/abs/2609.22048)

    该论文提出了一种可计算的“可用性”框架，利用精确二项反演和动态规划为跨机器学习系统的选择性预测提供认证，揭示了安全性、报告粒度与服务流量之间的权衡。

    

    选择性预测器充当一个安全门控：只有当预测看起来足够可信时，它才返回输出。实际部署日益要求这种可靠性在目标精度下针对每个关注的报告单元（例如某个工具、政策标签或患者亚组）得到认证。主要的困难往往不在于已授予的证书是否有效，而在于有限的校准数据究竟能否产生证书。当门控变得更安全或更细粒度时，某些单元可能因证据过少而无法完成认证。我们通过经典的精确二项分布反演使这种“可用性”概念变得可计算，并在固定组顺序下将报告分区选择形式化为一个动态规划问题，从而揭示安全性、粒度与所服务流量之间的权衡。由此得到的前沿边界揭示了一个被有限样本估计几乎抹去的大量人群机会：掌握真实信息的规划者可获得 0.157 的平均覆盖提升……

    arXiv:2609.22048v1 Announce Type: new  Abstract: A selective predictor acts as a safety gate: it returns an output only when the prediction appears sufficiently trustworthy. Deployments increasingly require this reliability to be certified at a target precision for every reporting unit of interest, such as a tool, policy label, or patient subgroup. The main difficulty is often not whether a granted certificate is valid, but whether finite calibration data can produce one at all. As the gate becomes safer or more fine-grained, some units may receive too little evidence to certify. We make this notion of availability computable through classical exact-binomial inversion and formulate reporting-partition selection, under a fixed group order, as a dynamic program that exposes the trade-off among safety, granularity, and served traffic. The resulting frontier reveals a large population opportunity that finite-sample estimation nearly erases: a truth-informed planner gains $0.157$ mean cover
    
[^6]: λ控制的GRPO：将流匹配比率不稳定性转化为可预算的资源

    $\lambda$-Controlled GRPO: Turning Flow-Matching Ratio Instability into a Budgeted Resource

    [https://arxiv.org/abs/2609.22041](https://arxiv.org/abs/2609.22041)

    该论文发现流匹配GRPO训练中的多种不稳定性（重要性比率漂移、分散与裁剪）均源于一个单一的每步量——路径方差，它由采样器的高斯转移核精确决定且可廉价估计，从而将不稳定性转化为一种可测量、可预算的资源。

    

    强化学习正被越来越多地用于使图像生成器与奖励信号对齐，Flow-GRPO最近通过将去噪采样器视为一种可以从奖励反馈中优化的随机策略，将这一范式扩展到了流匹配模型。在这种设置下，训练会以一种多步去噪特有的方式变得不稳定：策略更新在各去噪步骤之间发生系统性变化，重要性比率漂移至1以下、变得越来越分散、以不同的速率被裁剪，并且在训练后期留下更少的可用样本。先前的工作将这些效应视为相互独立的失败模式，并针对每一种效应采用手工调参的稳定器来处理。我们则证明，这些效应源于一个单一的每步量，我们将其称为路径方差。该量由采样器的高斯转移核精确确定，并且可以在训练过程中被廉价地估计。这将不稳定性重新定义为一种可以被测量的资源……

    arXiv:2609.22041v1 Announce Type: new  Abstract: Reinforcement learning is increasingly used to align image generators with reward signals, and Flow-GRPO recently extended this paradigm to flow-matching models by treating the denoising sampler as a stochastic policy that can be optimized from reward feedback. Training in this setting is unstable in a way specific to multi-step denoising: the policy update changes systematically across denoising steps, with importance ratios drifting below one, becoming increasingly dispersed, clipping at different rates, and leaving fewer usable samples late in training. Prior work treats these effects as separate failure modes and addresses each with a hand-tuned stabilizer. We show instead that they arise from a single per-step quantity, which we call path variance. This quantity is determined exactly by the sampler's Gaussian transition kernel and can be estimated cheaply during training. This reframes instability as a resource that can be measured 
    
[^7]: COMPLEX：多参数持久模块的闭式认证嵌入

    COMPLEX: A Closed-Form Certified Embedding of Multiparameter Persistence Modules

    [https://arxiv.org/abs/2609.22012](https://arxiv.org/abs/2609.22012)

    COMPLEX通过闭式、免训练的嵌入方法为多参数持久模块补齐了以往缺失的Lipschitz下界，给出了首个多参数特征映射的双侧失真界，使特征忠实性变得可度量。

    

    据我们所知，现有的每一种多参数持久性向量化方法都只带有单侧的Lipschitz上界，而下界则无从谈起：没有下界度量，就无法说明这些特征是忠实的，也无法在其上建立针对单个预测的保证。本文补齐了缺失的一侧。COMPLEX是一种闭式的、无需训练的多参数模块嵌入方法——沿固定的近对角网格对模块进行切片，用经过认证的PLACE/PALACE地标映射嵌入每个切片的条形码，然后进行拼接。在一个可检验的见证切片一致性条件下（该条件在Orbit5k数据集上对100%的被审计样本对成立），单个切片即可携带一个闭式下界度量：相互分离的模块在嵌入空间中保持分离。结合标准上界，这给出了据我们所知首个针对多参数特征映射的双侧失真界，使忠实性变得可度量。在度量忠实性时，我们发现该下界在实现距离的一个小因子范围内是紧的，然而……

    arXiv:2609.22012v1 Announce Type: new  Abstract: Every multiparameter persistence vectorization we know of carries a one-sided Lipschitz upper bound and nothing below it: without a lower gauge there is no sense in which the features are faithful, and no per-prediction guarantee can be built on them. This paper supplies the missing side. COMPLEX is a closed-form, training-free embedding of multiparameter modules -- slice the module along a fixed near-diagonal net, embed each slice barcode by the certified PLACE/PALACE landmark map, concatenate. Under a checkable witnessing-slice coherence condition, holding on 100% of audited pairs on Orbit5k, a single slice carries a closed-form lower gauge: separated modules stay separated in the embedding. With the standard upper bound this gives, to our knowledge, the first two-sided distortion bound for a multiparameter feature map, making faithfulness measurable. Measuring it, we find the floor tight within a small factor of realized distances yet
    
[^8]: 弃权与噪声过滤：Softmax注意力缺失的两个基本要素

    Abstention and Noise Filtering: Two Missing Primitives of Softmax Attention

    [https://arxiv.org/abs/2609.22005](https://arxiv.org/abs/2609.22005)

    本文论证并通过实验证明，注意力值通路的门控为softmax注意力补充了两种缺失的基本要素——弃权（允许注意力头输出空值）和噪声过滤（抑制残差流中叠加特征的干扰），并在10M至350M参数的匹配模型上提供了实证支持。

    

    据报道，对注意力的值通路进行门控可以改善语言模型的预训练效果，但此前的研究对其原因看法不一。我们提出论点并提供实验证据，表明这类门控为softmax注意力提供了两种它所缺乏的不同能力：弃权与噪声过滤。第一种是弃权，它允许注意力头输出空值，从而绕过注意力权重之和必须为一的约束。第二种是噪声过滤，它允许注意力头的值通路抑制残差流中叠加特征带来的干扰。在我们对从10M到350M参数的匹配模型进行的实验中，我们通过在softmax中引入一个可学习的每头汇点logit来实现弃权，通过对每个值进行门控来实现噪声过滤。我们报告了三项实证发现。第一，弃权的收益（以相对于匹配基线的验证损失下降来衡量）随着模型规模的增大而下降，而……的收益（摘要在此处被截断）

    arXiv:2609.22005v1 Announce Type: cross  Abstract: Gating the value pathway of attention reportedly improves language model pretraining, and prior studies disagree on why. We argue and provide experimental evidence that such gates supply two different things that softmax attention lacks: abstention and noise filtering. The first is abstention, which allows an attention head to output nothing, bypassing the requirement that attention weights must sum to one. The second is noise filtering, which allows the value pathway of an attention head to suppress interference from superposed features in the residual stream. In our experiments in matched models from 10M to 350M parameters, we supply abstention through a learned per-head sink logit in the softmax and noise filtering through a gate on each value. We report three empirical findings. First, the benefit of abstention, measured as the reduction in validation loss relative to a matched baseline, declines as models grow, whereas the benefit
    
[^9]: 基于机器学习的临界热流密度模型在CTF子通道代码中用于方形棒束预测的评估

    Assessment of Machine Learning-Based Critical Heat Flux Models in the CTF Subchannel Code for Square Rod Bundle Prediction

    [https://arxiv.org/abs/2609.21995](https://arxiv.org/abs/2609.21995)

    本研究利用EPRI棒束数据库评估了部署在CTF子通道代码中的机器学习临界热流密度模型，发现基于圆管数据训练的ML模型能够良好迁移到方形棒束几何结构中并优于传统方法。

    

    临界热流密度（CHF）是核热工水力学中一个关键的安全相关量，由于其与燃料性能和反应堆安全的直接关系，其预测仍然是一个重要挑战。最近的研究表明，相对于传统的经验关联式和查找表（LUT），机器学习（ML）方法可以显著提高CHF预测精度。然而，大多数基于机器学习的CHF模型都是使用圆管数据库开发和评估的，其在反应堆相关棒束几何结构中的适用性在很大程度上尚未被探索。本研究使用电力研究院（EPRI）棒束CHF数据库，评估了部署在CTF子通道代码中的基于机器学习的CHF模型。研究考虑了局部和半局部公式下的纯机器学习模型和混合残差修正模型。基于圆管训练的机器学习CHF模型通常能够良好地迁移到棒束应用中，并优于传统方法。

    arXiv:2609.21995v1 Announce Type: new  Abstract: The prediction of critical heat flux (CHF), a key safety-related quantity in nuclear thermal hydraulics, remains an important challenge due to its direct relationship with fuel performance and reactor safety. Recent studies have demonstrated that relative to traditional empirical correlations and lookup tables (LUTs), machine learning (ML) methods can substantially improve CHF prediction accuracy. Most ML-based CHF models, however, have been developed and evaluated using tube databases, leaving their applicability to reactor-relevant rod bundle geometries largely unexplored.   This study evaluates ML-based CHF models deployed within the CTF subchannel code using the Electric Power Research Institute (EPRI) rod bundle CHF database. Both pure and hybrid residual correction models are considered in local and semilocal formulations. The tube-trained ML CHF models generally transferred favorably to rod bundle applications and outperformed tra
    
[^10]: 基于谱对齐潜在流匹配的时间序列生成

    Time series generation with spectrally aligned latent flow matching

    [https://arxiv.org/abs/2609.21989](https://arxiv.org/abs/2609.21989)

    本文提出一种谱对齐的潜在流匹配时间序列生成方法，通过引入基于傅里叶、小波和签名变换等可解释信号表示的微调损失，使潜在空间保留关键动力学特性，从而消除潜在压缩导致的频谱失配伪影。

    

    潜在流模型已被证明是一种可靠且经济高效的时间序列生成方法。然而，潜在压缩会引入不期望的伪影，例如与底层数据集之间的频谱失配，从而阻碍了其作为训练替代工具的使用。在本文中，我们提出了一种谱对齐的潜在流时间序列生成器，其中用于流匹配的潜在空间经过训练，以保留与合成样本适用性相关的动力学特性。我们发现，引入基于经典信号表示（如傅里叶变换、小波变换和签名变换）的微调损失有助于克服这些问题。这些变换的可解释性使我们能够确保合成信号在相关特征（如平滑度或目标频谱内容）方面与真实信号保持一致，而不仅仅依赖于逐点重建损失。

    arXiv:2609.21989v1 Announce Type: new  Abstract: Latent flow models have proven to be a reliable and cost-effective method for time series generation. However, the latent compression induces unwanted artefacts, such as a spectral mismatch with respect to the underlying dataset, thus hindering their use as training surrogates. In this article, we propose a spectrally-aligned latent-flow time series generator, where the latent space for flow matching is trained to preserve dynamical properties that are relevant for the suitability of synthetic samples. We find that incorporating fine-tuning losses based on canonical signal representations such as the Fourier, wavelet and signature transforms helps overcome these issues. The interpretability of these transformations allows us to ensure that the synthetic signals are aligned with the true ones in terms of relevant features, such as smoothness or targeted spectral content, as opposed to relying on pointwise reconstruction losses only. We co
    
[^11]: 博弈中实现常数遗憾的乘性乐观方法

    Multiplicative Optimism for Constant Regret in Games

    [https://arxiv.org/abs/2609.21976](https://arxiv.org/abs/2609.21976)

    提出乘性乐观遗憾匹配（MORM）学习规则，使有限一般和博弈中的自博弈玩家在全信息下仅需一步乐观即可在所有时间跨度上一致地获得 O(√n log d) 的低遗憾，并能抵御对抗性效用。

    

    我们提出了乘性乐观遗憾匹配，这是一种用于有限一般和博弈的非耦合学习规则。在同时进行的全信息自博弈下，每个玩家仅需一步乐观即可在所有时间跨度上一致地实现 O(√n log d) 的外部遗憾。该分析将基于势函数的遗憾匹配论证与乘性稳定性以及策略移动的 Hellinger 控制相结合。此外，学习率保障机制在面对对抗性效用时还能提供 O(√T log d) 的遗憾保证。

    arXiv:2609.21976v1 Announce Type: cross  Abstract: We introduce Multiplicatively Optimistic Regret Matching (MORM), an uncoupled learning rule for finite general-sum games. Under simultaneous full-information self-play, every player achieves external regret $O(\sqrt n\log d)$ uniformly over all horizons, using only one-step optimism. The analysis combines a potential-based regret-matching argument with multiplicative stability and Hellinger control of strategy movement. A learning-rate safeguard additionally gives $O(\sqrt{T\log d})$ regret in the face of adversarial utilities.
    
[^12]: 掩码离散扩散中tau-leaping的调度优化

    Schedule optimization for tau-leaping in masked discrete diffusion

    [https://arxiv.org/abs/2609.21960](https://arxiv.org/abs/2609.21960)

    该论文通过依赖密度ρ的精确积分表示来刻画掩码离散扩散中tau-leaping采样的因式分解误差，并推导有限步优化问题的递归平稳性方程，从而实现去噪调度的优化。

    

    掩码离散扩散模型通常使用所谓的tau-leaping离散化方法来加速，该方法在每个采样步骤中并行揭示多个坐标。采样器用乘积分布替代每个被揭示块的联合条件分布，从而产生因式分解误差 $\varepsilon_\text{fact}$，即使预测器被完美学习，该误差依然存在。我们分析了在 $N$ 个坐标上进行 $K$ 步采样的标准采样器，其随机块大小取决于去噪调度。我们的分析使用 $\varepsilon_\text{fact}$ 的精确积分表示，该表示以依赖于分布的依赖密度 $\rho$ 来刻画，$\rho$ 记录了随着坐标被揭示比例的增长，条件依赖如何演化。我们为该依赖轮廓开发了估计器，并量化了估计误差如何影响调度选择。我们为有限 $K$ 的优化问题推导了递归平稳性方程，并……（原文在此处截断）

    arXiv:2609.21960v1 Announce Type: cross  Abstract: Masked discrete diffusion models are commonly accelerated using the so-called tau-leaping discretization method, which reveals several coordinates in parallel at each sampling step. The sampler replaces the joint conditional law of each revealed block by a product distribution, incurring a factorization error $\varepsilon_\text{fact}$ present even with perfectly learned predictors. We analyze the standard sampler on $N$ coordinates with $K$ sampling steps, whose random block sizes depend on a denoising schedule. Our analysis uses an exact integral representation of $\varepsilon_\text{fact}$ in terms of a distribution-dependent dependence density $\rho$, which records how conditional dependence evolves as the revealed fraction of coordinates grows. We develop estimators for this profile and quantify how estimation errors affect schedule selection. We derive recursive stationarity equations for the finite-$K$ optimization problem and, un
    
[^13]: RACER：面向人机路由的角色对齐能力估计

    RACER: Role-Aligned Competence Estimation for Human-AI Routing

    [https://arxiv.org/abs/2609.21953](https://arxiv.org/abs/2609.21953)

    RACER 提出一个角色相对的能力估计框架，通过估计未见专家在各候选类别角色下回答正确的后验预测概率并与模型后验结合，既避免了基于绝对类别坐标的路由捷径，又能捕捉实例级别的专家专长。

    

    学习型延迟让预测系统决定何时自主行动、何时将任务转交给人类专家。种群自适应延迟将这一问题扩展到未见过的专家，只需使用一小段专家行为的上下文集合。诸如 L2D-Pop 这样的神经上下文编码器可以做到查询相关，但可能学到与绝对类别坐标绑定的路由捷径。无身份延迟通过角色索引的类别级能力画像消除了这类捷径，但其估计在每个类别内是恒定的，无法捕捉实例级别的专家专长。我们提出 RACER——面向路由的角色对齐能力估计——一个从上下文中估计未见专家能力的角色相对框架。RACER 估计专家在每个候选类别角色下对查询回答正确的后验预测概率，然后将这些估计与模型后验相结合，得到与贝叶斯相关的专家正确概率。

    arXiv:2609.21953v1 Announce Type: new  Abstract: Learning to defer asks a predictive system when to act autonomously and when to defer to a human expert. Population-adaptive deferral extends this problem to unseen experts using a small context set of expert behavior. Neural context encoders such as L2D-Pop can be query-dependent, but may learn routing shortcuts tied to absolute class coordinates. Identity-Free Deferral (IFD) removes such shortcuts through role-indexed classwise competence profiles, but its estimates are constant within each class and cannot capture instance-level expert specialization. We propose RACER---Role-Aligned Competence Estimation for Routing---a role-relative framework for estimating an unseen expert's competence from context. RACER estimates the posterior-predictive probability that the expert is correct on a query under each candidate class role, then combines these estimates with the model posterior to obtain the Bayes-relevant expert-correctness probabilit
    
[^14]: 学习驱动城市：用于城市网络校准与控制的深度元模型与强化学习策略

    Learning to Move Cities: Deep Meta-Models and Reinforcement Policies for Calibration and Control in Urban Networks

    [https://arxiv.org/abs/2609.21945](https://arxiv.org/abs/2609.21945)

    本文提出一个共享潜在空间框架，利用MLP自编码器学习城市交通动态的低维表示，将交通仿真器的贝叶斯优化校准与深度Q学习控制统一起来，显著提升了校准的样本效率。

    

    城市交通网络带来了复杂的优化挑战，涵盖高保真仿真器的校准与实时运行控制两个方面。本文提出了一个共享潜在空间框架，通过对城市交通动态的共同学习表示，将仿真器校准与强化学习控制联系起来。首先，我们开发了一种组合式MLP自编码器架构，学习连接仿真器输入（起讫点需求、网络参数）与输出（行程时间、拥堵模式）的低维流形，从而实现高效的贝叶斯优化校准。与传统降维方法相比，该方法展现出卓越的样本效率，能够在固定计算预算内更好地拟合观测数据。其次，我们实现了一个具备经验回放和目标网络的深度Q学习智能体，通过调度优化动态交通分配（摘要在此处被截断）。

    arXiv:2609.21945v1 Announce Type: new  Abstract: Urban transportation networks present complex optimization challenges spanning calibration of high-fidelity simulators and real-time operational control. This paper presents a shared latent-space framework that connects simulator calibration and reinforcement learning control through a common learned representation of urban traffic dynamics. First, we develop a combinatorial MLP-autoencoder architecture that learns low-dimensional manifolds linking simulator inputs (origin-destination demand, network parameters) to outputs (travel times, congestion patterns), enabling efficient Bayesian optimization for calibration. This approach demonstrates superior sample efficiency compared to traditional dimension reduction methods, achieving better fit to observational data within fixed computational budgets. Second, we implement a deep Q-learning agent with experience replay and target networks to optimize dynamic traffic assignment through schedu
    
[^15]: 使用矩阵指数不动点迭代引导量子博弈智能体达到均衡

    Guiding Agents of Quantum Games to Equilibrium using Matrix Exponential Fixed-Point Iteration

    [https://arxiv.org/abs/2609.21944](https://arxiv.org/abs/2609.21944)

    本文提出了带退火的矩阵指数不动点迭代（MEFPIA）算法，通过张量缩并表达式避免构造高维联合密度矩阵，从而高效计算扩展Gutoski-Watrous量子博弈中的均衡策略。

    

    近年来，量子博弈论作为利用量子原理研究多智能体系统中决策制定的框架而受到广泛关注。然而，计算均衡策略具有挑战性，因为联合希尔伯特空间的维度会随玩家局部维度的乘积而增长。在本文中，我们考虑了一种扩展的Gutoski-Watrous（EGW）博弈，其中每个玩家的量子策略由局部密度矩阵表示。我们推导了收益函数及其梯度的张量缩并表达式，从而避免了显式构造完整的联合密度矩阵以及将其与收益算子进行计算代价高昂的乘法运算。基于由此得到的有效哈密顿量，我们提出了带退火的矩阵指数不动点迭代（MEFPIA）算法来搜索EGW博弈中的均衡点。我们将MEFPIA与矩阵乘性权重方法进行了比较（摘要在此处截断）。

    arXiv:2609.21944v1 Announce Type: cross  Abstract: In recent years, quantum game theory has gained significant attention as a framework for studying decision-making in multi-agent systems using quantum principles. However, computing equilibrium strategies is challenging because the dimension of the joint Hilbert space grows as the product of the players' local dimensions. In this paper, we consider an extended Gutoski-Watrous (EGW) game in which each player's quantum strategy is represented by a local density matrix. We derive tensor-contraction expressions for the payoff functions and their gradients, thereby avoiding the explicit construction of the full joint density matrix and its computationally expensive multiplication by the payoff operators. Building on the resulting effective Hamiltonians, we propose the Matrix Exponential Fixed-Point Iteration with Annealing (MEFPIA) algorithm to search for equilibrium points in EGW games. We compare MEFPIA with the Matrix Multiplicative Weig
    
[^16]: 基于高效符号恢复的端到端硬标签密码分析模型提取

    End-to-End Hard-Label Cryptanalytic Model Extraction Using Efficient Sign Recovery

    [https://arxiv.org/abs/2609.21941](https://arxiv.org/abs/2609.21941)

    本文提出了一种基于全新原理的高效符号恢复算法，实现了对训练好的深度ReLU多层感知机在硬标签设置下的端到端黑盒模型提取攻击。

    

    深度神经网络（DNN）的重要性已得到广泛认可，通过训练获得的参数被视为宝贵的资产。近年来，仅通过对DNN进行预言机查询来提取这些参数的攻击在IACR会议上得到了积极研究。硬标签设置是模型提取中最具挑战性的设置，在这种设置下，攻击者只能观察到最终的输出标签，例如“狗”或“猫”。在Eurocrypt 2025上，Carlini等人提出了针对基于ReLU的多层感知机（MLP）的多项式时间硬标签提取方法。然而，该攻击过程中的一个步骤，即符号恢复，需要大量的查询和可观的计算量，在黑盒设置下实现这一步骤仍然困难。因此，在训练好的深度ReLU MLP上进行完全黑盒的端到端演示一直是一个挑战。本文提出了一种基于与先前方法完全不同原理的新型符号恢复算法。

    arXiv:2609.21941v1 Announce Type: cross  Abstract: The importance of deep neural networks (DNNs) is widely recognized, and the parameters obtained through training are regarded as valuable assets. Recently, attacks that extract these parameters using only oracle queries to a DNN have been actively studied at IACR conferences. The hard-label setting is the most challenging setting for model extraction, where an adversary can observe only the final output label, such as "dog" or "cat." At Eurocrypt 2025, Carlini et al. proposed polynomial-time hard-label extraction of ReLU-based MLPs. However, one step of this attack process, i.e., sign recovery, requires a large number of queries and substantial computation. Implementing this step in a black-box setting remains difficult. Consequently, a fully black-box end-to-end demonstration on trained deep ReLU MLPs has remained a challenge. In this paper, we propose a new sign-recovery algorithm based on a completely different principle from the ex
    
[^17]: 基于部分充电数据的锂离子电池剩余使用寿命预测与容量估计联合方法

    Joint Remaining Useful Life Prediction and Capacity Estimation of Lithium-Ion Batteries Using Partial-Charging Data

    [https://arxiv.org/abs/2609.21932](https://arxiv.org/abs/2609.21932)

    本文提出一种基于部分充电数据的跨专家框架，通过RUL专家与容量专家双网络结构及特征级线性调制模块，在无需历史完整循环容量的条件下联合实现锂离子电池的剩余使用寿命预测与容量估计。

    

    联合剩余使用寿命（RUL）预测与容量估计需要对电池渐进退化过程和近期行为的双重表征。本文提出了一种跨专家框架，仅使用部分充电测量数据，无需以实测的历史完整循环容量作为输入。RUL专家使用预训练的门控循环单元（GRU）编码器、二维卷积神经网络（2D-CNN）和时间GRU，对从30个循环历史中采样的十个循环的名义10分钟片段进行编码。容量专家使用2D-CNN和Transformer处理十个连续循环的名义40分钟片段的统计描述符。特征级线性调制（FiLM）模块利用短期表征对长期表征进行条件化调制，以实现联合预测。训练过程包括有监督的自编码器预训练、独立专家预训练以及冻结专家的融合训练。在两个公开电池数据集上（摘要原文在此处截断）……

    arXiv:2609.21932v1 Announce Type: new  Abstract: Joint remaining useful life (RUL) prediction and capacity estimation require representations of both gradual degradation and recent battery behavior. This paper presents a cross-expert framework using partial-charging measurements without measured historical full-cycle capacity as an input. The RUL Expert encodes nominal 10-min segments from ten cycles sampled within a 30-cycle history using a pretrained gated recurrent unit (GRU) encoder, a two-dimensional convolutional neural network (2D-CNN), and a temporal GRU. The Capacity Expert processes statistical descriptors of nominal 40-min segments from ten consecutive cycles using a 2D-CNN and a Transformer. A feature-wise linear modulation module uses the short-term representation to condition the long-term representation for joint prediction. Training comprises supervised autoencoder pretraining, independent expert pretraining, and fusion training with frozen experts. On two public batter
    
[^18]: 折点与平滑性：类拉普拉斯源下实解析非线性独立成分分析（nICA）的可辨识性

    Kinks vs. Smoothness: Identifiability of Real Analytic nICA for Laplace-like Sources

    [https://arxiv.org/abs/2609.21926](https://arxiv.org/abs/2609.21926)

    该论文证明了当独立源的密度函数一阶导数存在有限个不连续点（如拉普拉斯分布）时，实解析非线性独立成分分析（nICA）在排除平凡歧义后是可辨识的，其核心证明思路是利用源分布中的折点与实解析函数平滑性之间的对比。

    

    许多机器学习系统试图用生成复杂数据的隐藏独立因子来解释这些数据——例如图像或金融时间序列。恢复真实的潜在因子，而非它们的某种打乱版本，是非线性独立成分分析（nICA）的核心挑战。我们证明了当源概率密度函数的一阶导数存在有限个不连续点时，实解析生成函数是可辨识的（可精确恢复，仅存在平凡的歧义）。拉普拉斯分布是满足该假设的最典型例子。我们的证明依赖于源分布中的折点与实解析函数平滑性之间的对比。实解析函数涵盖了广泛的一类生成机制，并且可以用具有标准激活函数（如tanh、softplus、GELU）的归一化流或变分自编码器来近似。

    arXiv:2609.21926v1 Announce Type: new  Abstract: Many machine learning systems try to explain complex data - like images or financial time series - in terms of hidden, independent factors that generated them. Recovering the true underlying factors, rather than some scrambled version of them, is the central challenge of nonlinear Independent Component Analysis (nICA). We prove identifiability (exact recovery) up to trivial ambiguities for real analytic generating functions when source probability density functions have a finite number of discontinuities in the first derivative. The Laplace distribution is the most prominent example satisfying this assumption. Our proof relies on the contrast between kinks in the source distribution and the smoothness of real analytic functions. Real analytic functions comprise a broad class of generating mechanisms, and can be approximated with Normalizing Flows or Variational Autoencoders with standard activation functions (e.g., tanh, softplus, GELU),
    
[^19]: 黎曼流形上切向量场回归的同时推断

    Riemannian Simultaneous Inference for Tangent Vector Field Regression

    [https://arxiv.org/abs/2609.21910](https://arxiv.org/abs/2609.21910)

    该论文针对无边黎曼流形上的切向量场回归提出了一种基于平行输运与体积校正的核估计方法，并通过单位切丛上的上确界表示与 Gumbel 极限理论，构造了回归场的可行同时置信管。

    

    我们考虑无边黎曼流形上的非参数切向量场回归。由于不同点处的响应位于不同的切空间中，所提出的核估计量首先将近邻的响应平行输运到目标切空间，然后形成经过体积校正的局部平均。我们首先推导了该估计量的一致二阶偏差、有限带宽协方差以及随机收敛率。对于同时推断，我们将切范数表示为单位切丛上的上确界。精确的协方差白化给出了一个单位方差的高斯场，其相关长度沿底流形为 $h$ 阶，沿纤维为一阶。其局部协方差几何导致了一个带有显式内蕴常数的 Gumbel 极限。将该极限与高斯近似和交叉拟合协方差估计相结合，得到了回归场的可行同时置信管。我们进一步讨论……（摘要被截断）

    arXiv:2609.21910v1 Announce Type: cross  Abstract: We consider nonparametric tangent vector field regression on a Riemannian manifold without boundary. Because responses at different points lie in different tangent spaces, the proposed kernel estimator first parallel transports nearby responses to the target tangent space and then forms a volume-corrected local average. We first derive its uniform second-order bias, finite-bandwidth covariance, and stochastic rate. For simultaneous inference, the tangent norm is written as a supremum over the unit tangent bundle. Exact covariance whitening gives a unit-variance Gaussian field whose correlation length is of order $h$ along the base manifold and of order one along the fibre. Its local covariance geometry leads to a Gumbel limit with an explicit intrinsic constant. Combining this limit with Gaussian approximation and cross-fitted covariance estimation yields a feasible simultaneous confidence tube for the regression field. We further disc
    
[^20]: 超越运动学：面向肌肉驱动模仿学习的仿真保真度基准测试

    Beyond Kinematics: Benchmarking Simulation Fidelity for Muscle-Driven Imitation Learning

    [https://arxiv.org/abs/2609.21909](https://arxiv.org/abs/2609.21909)

    该论文基于共同的人体动作捕捉与肌电测量数据，系统比较了SCONE/HyFyDy与MuJoCo/MyoSim两种肌肉驱动运动模仿强化学习流程，揭示两者虽能高保真再现人体运动学，但在捕捉肌肉激活、代谢成本等潜在神经肌肉行为方面存在差异，这对机器人辅助设备的设计与控制具有重要意义。

    

    在这项工作中，我们对两种最先进的运动模仿强化学习（MIRL）流程进行了系统性比较，一种基于SCONE/HyFyDy构建，另一种基于MuJoCo/MyoSim构建。HyFyDy通过详细的肌肉肌腱建模强调生理真实性，而MuJoCo则优先考虑计算效率和可扩展的策略学习。尽管近期工作已经证明这两种流程都能高保真地再现人体运动学，但它们是否能准确捕捉产生运动的潜在神经肌肉行为仍不清楚。这一局限性对于机器人辅助设备的设计与控制尤为重要，因为肌肉激活模式和代谢成本等结果指标常被用作优化目标。为了进行系统性比较，我们的工作使用一组共同的人体动作捕捉和肌电图（EMG）测量数据对两种流程进行了比较。结果发现，虽然……（原文摘要在此处截断）

    arXiv:2609.21909v1 Announce Type: new  Abstract: In this work, we conduct a systematic comparison of two state-of-the-art motion-imitation reinforcement learning (MIRL) pipelines, one built on SCONE/HyFyDy and one built on MuJoCo/MyoSim. HyFyDy emphasizes physiological realism through detailed musculotendon modeling, while MuJoCo prioritizes computational efficiency and scalable policy learning. While recent work has demonstrated that both pipelines reproduce human kinematics with high fidelity, it remains unclear if they accurately capture the underlying neuromuscular behavior that produced the movement. This limitation is particularly important for robotic assistive-device design and control, where outcome measures such as muscle activation patterns and metabolic cost are often used as optimization targets. To conduct a systematic comparison, our work compares both pipelines using a common set of human motion-capture and electromyography (EMG) measurements. The results find that whil
    
[^21]: 干预粒度至关重要：临床世界模型反事实模拟中的连贯治疗束

    Intervention Granularity Matters: Coherent Treatment Bundles in Counterfactual Simulation with Clinical World Models

    [https://arxiv.org/abs/2609.21906](https://arxiv.org/abs/2609.21906)

    该论文发现临床干预以不可分割的治疗束形式记录，并证明干预编辑的粒度会显著影响临床世界模型反事实模拟的预测结果。

    

    使用临床世界模型进行反事实模拟意味着固定患者的病史、改变治疗方案并读取预测的响应。这样做需要决定什么算作一次干预。在临床环境中，干预是以捆绑束的形式记录的：对来自MIMIC-IV的945,707个患者小时的共现审计显示，某些组件组（例如透析回路的每个参数）从不单独出现，因此仅更改单个组件的编辑所描述的是一个在数据中从未出现过的时刻。我们假设干预被编辑的粒度会改变世界模型的响应方式，并使用Clin-JEPA进行验证，Clin-JEPA是一个以每小时治疗文本为条件的患者轨迹潜在世界模型。在1,019次有记录的有创通气开始时刻，我们保持患者的病史和其他治疗不变，比较编辑单个呼吸机设置与编辑完整记录配置的效果。

    arXiv:2609.21906v1 Announce Type: new  Abstract: Counterfactual simulation with a clinical world model means fixing a patient's history, changing the treatment, and reading off the predicted response. Doing so requires deciding what counts as one intervention. In clinical settings, interventions are documented as bundles: a co-occurrence audit of 945,707 patient-hours from MIMIC-IV shows groups of components, such as every parameter of a dialysis circuit, that never appear apart, so an edit that changes one component on its own describes an hour that never occurs in the data. We hypothesize that the granularity at which an intervention is edited changes how a world model responds, and test this with Clin-JEPA, a latent world model of patient trajectories conditioned on hourly treatment text. At 1,019 documented onsets of invasive ventilation, we keep the patient's history and other treatments fixed and compare editing one ventilator setting with editing the complete configuration recor
    
[^22]: ExpBoN：用于高效测试时LLM对齐的指数噪声Best-of-n方法

    ExpBoN: Exponential-Noise Best-of-$n$ for Efficient Test-Time LLM Alignment

    [https://arxiv.org/abs/2609.21899](https://arxiv.org/abs/2609.21899)

    本文提出基于指数噪声报告-噪声-最大机制的软Best-of-n采样方法ExpBoN，具有精确有限n分解和指数级快速收敛的理论保证，并将其集成到引导式投机推理框架中（ExpGSI），实现了高效的奖励引导LLM测试时对齐。

    

    Best-of-n（BoN）采样是一种简单而有效的推理时对齐方法，但硬最大化方式只能对奖励与分布偏移之间的权衡提供粗略的控制。软Best-of-n（Verdun等人，2025）提供了更平滑的控制，并收敛到与KL正则化奖励最大化相关联的最优分布。在本文中，我们提出了ExpBoN，一种基于指数噪声“报告噪声最大值”（report-noisy-max）机制的替代性软BoN方法。该方法具有精确的有限n分解，从而在总变差、期望奖励以及KL散度的两个方向上都能实现指数级快速收敛。我们对其收敛性和遗憾行为提供了全面的理论分析。我们进一步将ExpBoN集成到引导式投机推理（GSI）框架（Geuter、Mroueh和AlvarezMelis，2025）中，得到ExpGSI，用于高效的奖励引导LLM对齐。ExpGSI带来了大幅的降低……

    arXiv:2609.21899v1 Announce Type: new  Abstract: Best-of-$n$ (BoN) sampling is a simple yet effective inference-time alignment method, but hard maximization provides only coarse control over the trade-off between reward and distribution shift. Soft Best-of-$n$ (Verdun et al. 2025) provides smoother control and converges to the optimal distribution associated with KL-regularized reward maximization. In this paper, we introduce ExpBoN, an alternative soft BoN method based on the exponential-noise report-noisy-max mechanism. It admits an exact finite-$n$ decomposition, which yields exponentially fast convergence in total variation, expected reward, and both directions of KL divergence. We provide comprehensive theoretical analyses of its convergence and regret behavior. We further integrate ExpBoN into the guided speculative inference (GSI) framework (Geuter, Mroueh, and AlvarezMelis 2025), resulting in ExpGSI, for efficient reward-guided LLM alignment. ExpGSI yields substantial reduction
    
[^23]: 大语言模型作为文本与表格预测的特征工程师

    LLMs as Feature Engineers for Text-and-Tabular Prediction

    [https://arxiv.org/abs/2609.21894](https://arxiv.org/abs/2609.21894)

    该论文提出一种错误驱动的迭代框架，让LLM自动从非结构化文本中提取可解释的分类特征用于表格预测，通过将模型错误转化为自然语言反馈将特征发现速度提升3倍，且生成特征与TF-IDF和稠密嵌入形成强互补。

    

    我们提出了一个迭代框架，用于自动化地从非结构化文本中提取可解释的、与模式绑定的分类特征，以供表格预测模型使用。为了在特征空间中进行探索，一个生成器LLM提出语义定义，一个独立的提取器LLM将这些特征具体化，而下游的表格模型则评估它们的预测性能。我们通过将显式的模型错误（如AUC排序倒置）转化为自然语言反馈来优化这一搜索过程，引导LLM解决特定的预测失败问题。在三个公开数据集上的评估表明，与无引导搜索相比，这种错误驱动的循环可将特征发现速度提升高达3倍。实证结果显示，所生成的特征展现出强大多视角互补性，当与TF-IDF和稠密嵌入结合时，严格优于任何子集。最后，该框架保证了实例级别的可解释性：所发现的特征……（原文摘要至此截断）

    arXiv:2609.21894v1 Announce Type: new  Abstract: We introduce an iterative framework that automates the extraction of interpretable, schema-bound categorical features from unstructured text for tabular prediction models. To navigate the feature space, a generator LLM proposes semantic definitions, a separate extractor LLM materializes the features, and a downstream tabular model evaluates their predictive performance. We optimize this search by translating explicit model errors, such as AUC ranking inversions, into natural-language feedback, steering the LLM to resolve specific predictive failures. Evaluated across three public datasets, this error-driven loop accelerates feature discovery by up to $3\times$ compared to unguided search. Empirically, the generated features demonstrate strong multi-view complementarity, strictly outperforming any subset when combined with TF-IDF and dense embeddings. Finally, the framework guarantees instance-level interpretability: the discovered featur
    
[^24]: 从自由能视角检测大语言模型中的预训练数据

    Detecting Pretraining Data in Large Language Models from a Free-Energy Perspective

    [https://arxiv.org/abs/2609.21888](https://arxiv.org/abs/2609.21888)

    该论文提出能量转移检测（ETD）方法，通过结合预测损失与预测熵的倾斜边界，并从亥姆霍兹自由能的宏观视角进行解释，从而更准确地区分大语言模型中的预训练数据与非预训练数据。

    

    检测大语言模型中的预训练数据具有挑战性，因为高似然既可能反映训练时的接触，也可能反映强大的泛化能力。在预测损失与预测熵的联合空间中，仅基于似然的检测器使用水平边界，可能将可预测的非成员误判为成员。受此启发，我们引入了一种倾斜边界，相对于预测熵来评估预测损失。我们的分析表明，熵校正可以在保持期望成员信号的同时降低其方差，从而改善成员与非成员之间的标准化分离。我们进一步将均值-方差分析扩展到具有非零平均熵差的更一般设定。有趣的是，这种熵调整后的得分可以给出亥姆霍兹自由能的解释，由此提出了能量转移检测（ETD）方法，从宏观残余自由能转移的视角看待预训练数据检测……

    arXiv:2609.21888v1 Announce Type: cross  Abstract: Detecting pretraining data in large language models is challenging because high likelihood can reflect either training exposure or strong generalization. In the joint space of prediction loss and predictive entropy, a likelihood-only detector uses a horizontal boundary and can mistake predictable non-members for members. Motivated by this, we introduce an inclined boundary that evaluates prediction loss relative to predictive entropy. Our analysis shows that entropy correction can preserve the expected membership signal while reducing its variance, thereby improving standardized member--non-member separation. We further extend the mean--variance analysis to the more general setting with a nonzero mean entropy gap. Interestingly, this entropy-adjusted score admits a Helmholtz free-energy interpretation, leading to Energy Transfer Detection (ETD), which views pretraining data detection from a macroscopic residual free-energy transfer per
    
[^25]: 光滑 $\ell_p$/$\ell_q$ 非对偶凸一阶预言机优化的近最优加速

    Near-Optimal Acceleration for Smooth $\ell_p$ / $\ell_q$ Nondual Convex First-Order Oracle Optimization

    [https://arxiv.org/abs/2609.21880](https://arxiv.org/abs/2609.21880)

    本文提出了针对 $\ell_p$ 球上 $\ell_q$ 范数 Hölder 光滑凸优化的近最优加速算法，在对数因子内解决了 COLT 2015 的公开问题，并在光滑情形下达到 $\widetilde{O}(LR^2/T^3)$ 的三次方收敛速率。

    

    我们研究了在 $RB_p^d$ 上具有 $(L,\kappa-1)$-Hölder 连续梯度（关于 $\ell_q$ 范数）的凸目标函数优化问题，其中 $1<\kappa\le 2$。对于所有满足 $p<\min\{q,2\}$ 的情形，算法在对一阶预言机进行 $T$ 次查询后达到误差 $$\widetilde{O}_{\kappa,p,q}\!\left(\frac{LR^\kappa}{T^{\kappa(1+1/p-(1/q-1/2)_+)-1}}\right)$$，从而在对数因子范围内解决了 (Guz15) 在 COLT 2015 上提出的公开问题。当 $(p,q)=(1,2)$ 时，收敛速率为 $\widetilde{O}(LR^\kappa/T^{2\kappa-1})$，其中包括光滑情形下 $\widetilde{O}(LR^2/T^{3})$ 的三次方衰减。

    arXiv:2609.21880v1 Announce Type: cross  Abstract: We study the optimization of convex objectives with $(L,\kappa-1)$-H\"older-continuous gradients in $\ell_q$ over $R B_p^d$, $1<\kappa\le 2$. (MG26) provides selectors with a movement bound for the problem of chasing high-dimensional convex nested sets for every $p<\min\{q,2\}$, has error $$   \widetilde O_{\kappa,p,q}\!\left(   \frac{LR^\kappa}{T^{\kappa(1+1/p-(1/q-1/2)_+)-1}}   \right), $$ after $T$ queries to a first-order oracle, solving the COLT 2015 open problem of (Guz15), up to logarithmic factors. At $(p,q)=(1,2)$, the rate is $\widetilde{O}(LR^\kappa/T^{2\kappa-1})$, including $\widetilde{O}(LR^2/T^{3})$ cubic decay in the smooth case.
    
[^26]: 面向等权乘性粗粒化的几何平均池化

    Geometric Mean Pooling for Equal-Weight Multiplicative Coarse-Graining

    [https://arxiv.org/abs/2609.21876](https://arxiv.org/abs/2609.21876)

    提出一种无需可学习参数的带符号池化算子——几何平均池化（GMP），它通过结合特征符号乘积与幅值几何平均来保留乘性信息，在分层粗粒化中可保持全局乘性统计量，并在基于乘积信号的合成任务上优于平均池化和最大池化。

    

    作为平均池化的加性偏置和最大池化的极值偏置的替代方案，我们引入了几何平均池化，这是一种带符号的池化算子，它将特征符号的乘积与特征幅值的几何平均相结合。受量子多体物理中局部到全局组合方式的启发，GMP在不引入可学习池化参数的情况下，同时保留了联合符号信息和特征性的乘性尺度。我们证明了非重叠的分层GMP能够保持相应的全局乘性统计量，并在合成序列任务、迭代粗粒化、图像分类和分子亲脂性回归上对其进行了评估。在合成任务上，GMP比平均池化和最大池化更准确地恢复基于乘积的信号，并在所测试的乘性输入噪声水平下保持预测性能。然而，在图像和分子数据上，其有效性取决于……

    arXiv:2609.21876v1 Announce Type: new  Abstract: As an alternative to the additive and extremal biases of average and max pooling, we introduce Geometric Mean Pooling (GMP), a signed pooling operator that combines the product of feature signs with the geometric mean of feature magnitudes. Motivated by local-to-global composition in quantum many-body physics, GMP retains both joint sign information and a characteristic multiplicative scale without introducing learnable pooling parameters. We show that non-overlapping hierarchical GMP preserves the corresponding global multiplicative statistic and evaluate it on synthetic sequence tasks, iterative coarse-graining, image classification, and molecular lipophilicity regression. On the synthetic tasks, GMP recovers product-based signals more accurately than average and max pooling and maintains predictive performance under the tested levels of multiplicative input noise. On image and molecular data, however, its effectiveness depends on the 
    
[^27]: Chronosphere：局部气候专家的时空镶嵌

    Chronosphere: Space-Time Tessellation of Local Climate Experts

    [https://arxiv.org/abs/2609.21872](https://arxiv.org/abs/2609.21872)

    Chronosphere提出了一种时空神经场，通过在时空环面上对可学习站点进行自适应镶嵌并结合共享的局部基函数库，使气候表征的容量与细节层次能随数据在空间和时间上自适应调整，其性能达到或超越最先进的位置编码器。

    

    我们提出了Chronosphere，一种学习气候表征的时空神经场。地理表征学习中的一个核心挑战是对空间和时间复杂度差异巨大的环境过程进行建模。然而，现有的位置编码器通常在所有位置固定单一的细节层次。诸如球谐函数之类的全局基在空间和时间上均匀分布容量；局部化基只能解析预定义的区域；学习式镶嵌虽然能够自适应，但在表示更高频率时效率低下。Chronosphere统一了这些方法，将时空环面 $S^2\times S^1$ 上可学习站点的自适应镶嵌与共享的局部基函数库相结合。容量放置在哪里以及每个区域承载多少细节，都会根据数据在空间和时间上进行自适应。通过训练以重建气候学数据，Chronosphere在空间……（原文在此处截断）方面匹配或领先于最先进的位置编码器。

    arXiv:2609.21872v1 Announce Type: cross  Abstract: We introduce Chronosphere, a spatio-temporal neural field that learns representations of climate. A central challenge in geographic representation learning is modeling environmental processes whose spatial and temporal complexity varies widely. Yet existing location encoders typically fix a single level of detail everywhere. Global bases such as spherical harmonics spread capacity uniformly across space and time. Localized bases resolve only predefined regions. Learned tessellations adapt, but are inefficient at representing higher frequencies. Chronosphere unifies these approaches, pairing an adaptive tessellation of learnable sites on the spacetime torus $S^2\times S^1$ with a shared bank of local basis functions. Both where capacity is placed and how much detail each region carries adapt to the data, across space and time. Trained to reconstruct climatology, Chronosphere matches or leads state-of-the-art location encoders across spa
    
[^28]: 神经细胞自动机在其隐藏通道中学习通用特征

    Neural Cellular Automata Learn General Features in their Hidden Channels

    [https://arxiv.org/abs/2609.21870](https://arxiv.org/abs/2609.21870)

    该论文揭示了神经细胞自动机隐藏通道能学习通用特征，并提出一种将预训练教师模型隐藏状态注入学生模型的新型迁移学习机制，使模型仅用约9,800个参数就在小样本任务中实现了超越循环和前馈网络的泛化能力。

    

    现代深度学习模型通过过度参数化实现了令人印象深刻的泛化能力，但这一范式在小样本场景中常常面临过拟合和记忆化的问题。神经细胞自动机（NCAs）提供了一种高度参数高效的替代方案，然而现有研究主要关注其输出，其内部隐藏通道的作用在很大程度上尚未被探索。在本文中，我们研究了NCA隐藏通道的内部动态，并引入了一种新颖的迁移学习机制，将预训练教师模型的隐藏状态注入学生模型中，以指导早期优化。在小样本和尺度变化的MNIST基准测试中，NCAs优于可比的循环和前馈架构，以极小的参数预算（约9,800个参数）展示了卓越的泛化能力。机制分析表明，隐藏通道将特征提取与统一的分类共识解耦……

    arXiv:2609.21870v1 Announce Type: cross  Abstract: Modern deep learning models achieve impressive generalization through over-parameterization, but this paradigm often struggles with overfitting and memorization in few-shot regimes. Neural Cellular Automata (NCAs) offer a highly parameter-efficient alternative, yet research has focused primarily on their output, leaving the role of their internal hidden channels largely unexplored. In this paper, we investigate the internal dynamics of NCA hidden channels and introduce a novel transfer-learning mechanism that injects a pretrained teacher's hidden states into a student model to guide early optimization. Evaluated on few-shot and scale-variant MNIST benchmarks, NCAs outperform comparable recurrent and feed-forward architectures, demonstrating superior generalization with a minimal parameter budget (~9,800 parameters). Mechanistic analysis reveals that the hidden channels decouple feature extraction from uniform classification consensus b
    
[^29]: AutoRecLab：描述实验，获取代码！

    AutoRecLab: Describe the Experiment, Get the Code!

    [https://arxiv.org/abs/2609.21863](https://arxiv.org/abs/2609.21863)

    AutoRecLab是一个基于Python的自主推荐系统实验平台，能根据自然语言描述的研究想法自动推导实验需求、构建验证原型并迭代扩展为完整实验代码，演示中9次运行成功8次且每次成本仅约1美元。

    

    实证评估是推荐系统研究的核心，但将实验设计转化为可执行代码仍然是一项手动且容易出错的任务。我们提出了AutoRecLab，一个基于Python的自主推荐系统实验室，能够从自然语言提示出发自动完成推荐系统实验。给定一个研究想法，AutoRecLab会推导出明确的实验需求，构建并验证原型，然后迭代地将其扩展为所需的完整实验。该工作流程结合了用于文档查询的检索增强生成（RAG）、静态类型验证以及执行引导的树搜索。在我们的演示中，AutoRecLab自主实现了一项显式到隐式反馈转换的研究。在跨六种算法和三个数据集的基线比较中，9次运行中有8次成功，使用GPT-5.4-mini时平均每次运行成本约为1美元。

    arXiv:2609.21863v1 Announce Type: new  Abstract: Empirical evaluation is central to recommender-systems (RecSys) research, but turning experimental designs into executable code remains a manual and error-prone task. We present AutoRecLab, a Python-based autonomous RecSys lab that automates RecSys experiments from natural-language prompts. Given a research idea, AutoRecLab derives explicit experiment requirements, builds and validates a prototype, and iteratively expands it into the requested full experiment. The workflow combines retrieval-augmented generation (RAG) for documentation lookup, static type verification, and execution-steered tree search. In our demonstration, AutoRecLab autonomously implements an explicit-to-implicit feedback conversion study. In a baseline comparison across six algorithms and three datasets, 8 of 9 runs succeed at an average cost of approx- imately $1 per run with GPT-5.4-mini.
    
[^30]: 基于泊松过程的可水印多草稿投机采样

    Watermarkable Multi-Draft Speculative Sampling via Poisson Processes

    [https://arxiv.org/abs/2609.21858](https://arxiv.org/abs/2609.21858)

    提出了一种基于泊松过程的多草稿投机采样算法，在保持高效采样的同时能够自然嵌入无偏水印而不降低投机接受率，突破了投机采样与水印技术难以兼得的权衡瓶颈。

    

    arXiv:2609.21858v1 公告类型：cross 摘要：大语言模型（LLMs）已在广泛的任务中取得了最先进的性能，这促使部署中需要关注两个重要方面：推理效率和输出溯源，前者可通过投机采样来解决，后者可通过水印技术来解决。然而，近期研究表明，将这两个目标结合起来非常困难，甚至可能无法实现。在本工作中，我们开发了一种基于泊松过程的新型多草稿投机采样算法，改进了这一基本权衡的前沿。所提出的算法本身就具有很强的采样效率，更有趣的是，它天然具备可水印性：我们可以在不降低投机接受率的情况下嵌入无偏水印。此外，我们的算法基于精确的“无通信列表耦合”方案，从而产生了对采样和水印均有益处的起草器不变性。这是首个多草稿

    arXiv:2609.21858v1 Announce Type: cross  Abstract: Large language models (LLMs) have achieved state-of-the-art performance across a wide range of tasks, motivating two important aspects of deployment: inference efficiency and output provenance, which can be tackled by speculative sampling and watermarking, respectively. However, recent works have shown that combining these two goals is highly nontrivial and can be potentially impossible. In this work, we develop a novel multi-draft speculative sampling algorithm based on Poisson processes that improves the frontier of this fundamental trade-off. The proposed algorithm has strong sampling efficiency on its own and, more interestingly, is naturally watermarkable: we can embed an unbiased watermark without degrading speculative acceptance. Moreover, our algorithm is based on an exact list-coupling-without-communication scheme, which yields a drafter invariance property that benefits both sampling and watermarking. It is the first multi-dr
    
[^31]: 权重时代已落幕——消费级GPU上的交互式扩散模型

    The Weight Is Over - Interactive Diffusion on Consumer GPUs

    [https://arxiv.org/abs/2609.21849](https://arxiv.org/abs/2609.21849)

    该论文提出嵌入翻译器、可复现的扩散流水线调优方案和端侧交互式图像生成编辑器三项贡献，在消费级GPU上实现了亚秒级首图时间的交互式扩散图像生成。

    

    端侧推理正在蓬勃发展，但发展势头几乎全部集中在语言模型上。扩散模型流水线内存消耗大、延迟敏感，并且需要协调嵌入器、Transformer、解码器，以及通常还需要进一步的后处理，而这些并不像大语言模型推理循环那样标准化。我们在性能、质量和模型占用空间之间进行权衡，以覆盖尽可能多的实际客户端设备。我们做出了三项贡献：一个嵌入翻译器，可将小型文本编码器映射到大型编码器空间，从而削减权重和延迟；一套可复现的扫描调优方案，用于在扩散流水线中权衡速度/质量/内存三角；以及一个交互式端侧图像生成编辑器，可在近期的GPU上实现亚秒级的首次图像呈现时间（TTFI）。

    arXiv:2609.21849v1 Announce Type: new  Abstract: On-device inference is booming, but the momentum is almost all in language models. Diffusion pipelines are memory hungry, latency-sensitive, and require orchestrating an embedder, a transformer, a decoder, and often further postprocessing that is not as standardized as LLM inference loops are. We navigate the trade-off between performance, quality, and model footprint to reach as many client devices in the wild as possible. We make three contributions: an embedding translator that maps a small text encoder into a large encoder space to cut weight and latency; a reproducible sweep recipe for navigating the speed/quality/memory triangle in diffusion pipelines; and an interactive on-device image generation editor achieving sub-second TTFI on recent GPUs.
    
[^32]: 面向高维异构数据的联邦深度聚类网络

    Federated Deep Clustering Networks for High-Dimensional and Heterogeneous Data

    [https://arxiv.org/abs/2609.21829](https://arxiv.org/abs/2609.21829)

    提出FedDCN——将深度聚类网络泛化到联邦学习场景，通过同时优化重构损失与聚类损失，在客户端数据非独立同分布（异构）的条件下实现鲁棒的高维数据聚类。

    

    对高维数据进行聚类是无监督机器学习中的一项基础任务，在众多领域都有广泛应用。在集中式数据场景下，该任务通常采用深度聚类方法解决，即利用深度神经网络架构学习有利于聚类的潜在空间表示。而在联邦学习中，数据分布在各个客户端且属于私有数据，深度聚类方法尚未得到充分探索。特别是，近期提出的联邦深度聚类方法尽管展现出非常有前景的性能，但当客户端之间的数据为非独立同分布时，仍难以可靠地提供良好的性能表现。在这项工作中，我们提出了深度聚类网络在联邦场景下的泛化方法，命名为FedDCN，该方法同时优化重构损失和聚类损失。为了确保在非独立同分布（数据异构）情况下的鲁棒性和潜在空间对齐……

    arXiv:2609.21829v1 Announce Type: cross  Abstract: Clustering high-dimensional data is a fundamental task in unsupervised machine learning with applications to a variety of domains. In the centralized data scenario, this task is commonly solved using deep clustering methods that utilize deep neural network architectures to learn clustering-friendly latent space representations. In Federated Learning, where data is distributed between clients and is private, deep clustering methods are less explored. In particular, recently introduced federated deep clustering methods, despite showing very promising performance, still fall short in reliably providing good performance if data across clients are non-identically-independently distributed. In this work, we introduce a generalization of Deep Clustering Networks to the federated scenario, named FedDCN, that simultaneously optimizes a reconstruction loss and a clustering loss. To ensure robustness and latent space alignment in non-identically-
    
[^33]: RheoSampling：解决随机动态树投机解码中的独热困境

    RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding

    [https://arxiv.org/abs/2609.21827](https://arxiv.org/abs/2609.21827)

    该论文提出RheoSampling方法，通过将树构建与token验证两个任务解耦，解决了随机解码下动态树投机解码中草稿分布坍缩为独热概率、接受率骤降的两难困境。

    

    投机解码通过并行起草多个token来加速大语言模型推理，而基于树的方法通过层次化结构进一步提升了效率。EAGLE-3等动态树方法在贪心解码下通过确定性的top-K扩展和全局剪枝表现优异。然而，在随机解码（T>0）场景下，这种机制会将草稿分布坍缩为独热概率，导致接受率急剧下降。由此产生了一个两难困境：动态树方法为保留上下文感知的拓扑结构而牺牲了随机采样，而静态树方法则以与上下文无关的结构为代价来保留随机采样。问题的根源在于同一个概率分布被用于两个相互冲突的任务：构建树结构和验证token。这种耦合使得直接注入随机性因所产生的随机过程而变得极具挑战性。我们通过解耦这两个角色来解决该问题：RheoSampl...（原文摘要在此处截断）

    arXiv:2609.21827v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM inference by drafting multiple tokens in parallel, with tree-based methods further improving efficiency through hierarchical structures. Dynamic-tree methods such as EAGLE-3 perform well under greedy decoding via deterministic top-K expansion and global pruning. However, in stochastic decoding (T>0), this mechanism collapses the draft distribution into one-hot probabilities, causing a severe drop in acceptance rate. This creates a dilemma: dynamic-tree methods sacrifice stochastic sampling to preserve context-aware topology, while static-tree methods preserve stochastic sampling with context-agnostic structures. The issue arises because the same probability distribution is used for two conflicting tasks: constructing the tree and verifying tokens. This coupling makes direct injection of randomness challenging due to the resulting stochastic process. We resolve this by decoupling these roles: RheoSampl
    
[^34]: 面向个性化液体复苏的自适应不确定性感知建模与随机径向基函数预测控制

    Adaptive Uncertainty-Aware Modeling and Stochastic Radial Basis Function Predictive Control for Personalized Fluid Resuscitation

    [https://arxiv.org/abs/2609.21821](https://arxiv.org/abs/2609.21821)

    该论文提出了一种融合贝叶斯生理建模与最优控制的新型框架，通过不确定性感知的状态空间模型（UVAE-SSM与BNSSM）显式建模偶然不确定性与认知不确定性，并结合随机径向基函数模型预测控制算法，实现液体复苏过程中不确定性感知的个性化血流动力学调节。

    

    本文提出了一种将贝叶斯生理建模与最优控制策略相结合的新型框架，以在液体复苏过程中实现不确定性感知的个性化血流动力学调节。首先，开发了一种不确定性感知变分自编码器状态空间模型（UVAE-SSM），利用有限数据捕捉平均动脉压（MAP）与液体输注之间的动态关系，同时显式建模偶然不确定性（即测量中的随机性，如传感器噪声）。随后，利用贝叶斯神经网络（BNN）构建了贝叶斯非线性状态空间模型（BNSSM），以捕捉由生理变异性和患者个体差异引起的认知不确定性，从而实现了虚拟患者生成器（VPG）的构建。基于这一不确定性感知建模框架，进一步设计了一种随机径向基函数模型预测控制（sRBF-MPC）算法，用于跟踪……（原文摘要在此处截断）

    arXiv:2609.21821v1 Announce Type: cross  Abstract: This paper presents a novel framework integrating Bayesian physiological modeling with optimal control strategies to achieve uncertainty-aware, personalized hemodynamic regulation during fluid resuscitation. An uncertainty-aware variational autoencoder state-space model (UVAE-SSM) was first developed to capture the dynamical relationship between mean arterial pressure (MAP) and fluid infusion using limited data, while explicitly modeling aleatoric uncertainty (i.e., randomness in the measurements, such as sensor noise). Then, a Bayesian nonlinear state-space model (BNSSM) was developed by utilizing Bayesian neural networks (BNNs) to capture epistemic uncertainty arising from physiological and patient-specific variability, enabling the creation of a virtual patient generator (VPG). Building on this uncertainty-aware modeling framework, a stochastic radial basis function model predictive control (sRBF-MPC) algorithm was designed to track
    
[^35]: 矩阵 AdaGrad：按行与按列的自适应次梯度方法

    Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods

    [https://arxiv.org/abs/2609.21815](https://arxiv.org/abs/2609.21815)

    本文提出了一个针对矩阵值参数的通用在线镜像下降框架，通过引入按行和按列的自适应近端函数，推导出 Row-AdaGrad 和 Column-AdaGrad 两种优化器，将 AdaGrad 式的自适应次梯度方法推广到了具有矩阵结构的参数优化中。

    

    arXiv:2609.21815v1 公告类型：交叉 摘要：AdaGrad 和 Adam 等自适应优化方法在现代神经网络训练中被广泛使用，但它们的自适应缩放主要是为向量值参数设计的，并未显式地利用矩阵结构。近期的矩阵感知优化器展示了结构化优化的优势，然而，目前仍缺乏一个可用于推导与 AdaGrad 相当的矩阵感知自适应性的通用理论框架。在本工作中，我们为矩阵值参数开发了一个带有自适应近端函数的通用在线镜像下降框架，提供了一种通过在线遗憾最小化来推导矩阵感知自适应优化的原则性方法。通过引入按行和按列的矩阵近端函数，并分析由此产生的遗憾权衡，我们推导出了按行矩阵 AdaGrad和按列矩阵 AdaGrad，其自适应缩放由累积的（按行/按列）梯度信息决定。

    arXiv:2609.21815v1 Announce Type: cross  Abstract: Adaptive optimization methods such as AdaGrad and Adam are widely used in modern neural-network training, but their adaptive scaling is primarily designed for vector-valued parameters and does not explicitly exploit matrix structure. Recent matrix-aware optimizers demonstrate the benefits of structured optimization, yet a general theoretical framework for deriving matrix-aware adaptivity comparable to that of AdaGrad remains lacking. In this work, we develop a general Online Mirror Descent framework with adaptive proximal functions for matrix-valued parameters, providing a principled approach to deriving matrix-aware adaptive optimization through online regret minimization. By introducing row-wise and column-wise matrix proximal functions and analyzing the resulting regret trade-off, we derive Row-wise Matrix AdaGrad (Row-AdaGrad) and Column-wise Matrix AdaGrad (Column-AdaGrad), with adaptive scaling determined by the accumulated row-w
    
[^36]: RegKT：基于IRT正则化器的可解释且鲁棒的深度知识追踪

    RegKT: Interpretable and Robust Deep Knowledge Tracing With IRT-Regularizer

    [https://arxiv.org/abs/2609.21791](https://arxiv.org/abs/2609.21791)

    本文提出RegKT，一种基于IRT正则化器的新型正则化技术，能够同时提升深度知识追踪模型的可解释性与鲁棒性并缓解过拟合问题，使其更适用于真实教育场景。

    

    随着深度学习模型的不断发展，知识追踪模型已经取得了更高的准确率。然而，这些提升是以牺牲可解释性为代价的，而可解释性对于教育领域的从业者采用新方法至关重要。此外，深度学习模型容易过拟合，尤其是在处理教育应用中常见的小规模数据集时。在本文中，我们提出了一种新颖的正则化技术，旨在增强基于深度学习的知识追踪模型的鲁棒性，同时提高其可解释性。我们的方法同时解决了可解释性和过拟合这两个挑战，使其在真实世界的教育应用中更加可行。

    arXiv:2609.21791v1 Announce Type: new  Abstract: As deep learning models continue to advance, knowledge tracing models have achieved higher accuracy. However, these gains come at the cost of reduced interpretability, which is crucial for practitioners in educational settings to adopt new methodologies. Additionally, deep learning models are prone to overfitting, particularly when dealing with the small datasets that are common in educational applications. In this paper, we propose a novel regularization technique designed to enhance the robustness of deep-learning-based knowledge tracing models, while simultaneously improving their interpretability. Our method addresses both the interpretability and overfitting challenges, making it more feasible for real-world educational applications.
    
[^37]: 从预训练到熟练掌握：最少人工干预下面向长时程操作的现实世界子任务强化学习

    From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention

    [https://arxiv.org/abs/2609.21788](https://arxiv.org/abs/2609.21788)

    该论文提出PARTS框架，利用冻结的预训练策略提供标称动作，并通过智能体生成的选择器和成功验证器提供局部奖励，将强化学习练习集中在长时程操作任务的关键瓶颈子任务上，从而在最少人工干预下实现现实世界的策略微调。

    

    预训练的机器人基础策略或许能够执行长时程任务的大部分内容，但在少数关键子任务上却反复失败。为监督微调（SFT）收集额外的完整任务演示，需要操作者重复策略已经能够很好完成的行为。强化学习（RL）微调为弥合这一差距提供了一条有前景的路径，但现有方法在仅使用稀疏奖励的情况下难以解决长时程任务。我们提出了PARTS（基于目标子任务的策略适应强化学习），这是一个现实世界的子任务强化学习框架，它将练习集中在这些瓶颈子任务上，同时允许训练回滚在最少人工干预的情况下进行。冻结的预训练策略在整个执行过程中提供标称动作，而由智能体生成的选择器和成功验证器则激活残差修正并提供局部结果奖励。这些奖励支持从成功的子任务中学习，即使完整任务（原文在此处截断）……

    arXiv:2609.21788v1 Announce Type: cross  Abstract: A pretrained robot foundation policy may execute most of a long-horizon task yet repeatedly fail at a few critical subtasks. Collecting additional full-task demonstrations for supervised fine-tuning (SFT) requires operators to repeat behaviors the policy already performs well. Reinforcement learning (RL) fine-tuning offers a promising path to bridge this gap, but existing approaches struggle to solve long-horizon tasks using only sparse rewards. We present PARTS (Policy Adaptation with RL on Targeted Subtasks), a real-world subtask RL framework that concentrates practice at these bottlenecks while allowing training rollouts to proceed with minimal human intervention. The frozen pretrained policy supplies nominal actions throughout execution, while agent-generated selectors and success verifiers activate residual corrections and provide local outcome rewards. These rewards support learning from successful subtasks even when complete-tas
    
[^38]: 超越基准分数：对用于胸部X光结核病筛查的医学视觉-语言模型进行审计

    Beyond Benchmark Scores: Auditing Medical Vision-Language Models for Chest X-Ray Tuberculosis Screening

    [https://arxiv.org/abs/2609.21763](https://arxiv.org/abs/2609.21763)

    该研究通过对三个医学视觉-语言模型在多数据集、多提示词族、不同阴性谱等条件下的系统审计，揭示了基准分数无法可靠反映模型在真实结核病筛查中的表现——模型排名和AUROC会随评估条件的改变而显著波动。

    

    医学模型的基准分数并不能证明同一结论在不同的评估条件下依然成立。本研究检验了关于模型排名、分数可靠性和筛查性能的结论，是否能在队列、提示词、阴性谱、设定患病率和操作阈值发生变化时依然保持成立。我们在来自四个数据集（Montgomery、Shenzhen、TBX11K和VinDr-CXR）的12,200份胸部X光片记录上，对三个医学视觉-语言模型（BioMedCLIP、CheXficient和MedSigLIP）以及一个通用领域的OpenCLIP对照模型进行了审计。五个固定的提示词族共产生了244,000个模型-图像-提示词分数。没有任何模型能在所有队列和所有可靠性标准上都保持领先。在48个经过多重性控制的比较中，有21个的AUROC因提示词族的改变而发生变化。将健康对照替换为患病的非结核病对照后，所有四个模型的AUROC下降了0.075至0.306。在VinDr-CXR上，这三个医学模型能够区分结核病与无异常影像……（原文摘要在此处被截断）

    arXiv:2609.21763v1 Announce Type: cross  Abstract: A medical model's benchmark score does not establish that the same conclusion holds under a different evaluation. This study tests whether claims about model ranking, score reliability and screening performance survive changes in cohort, prompt, negative spectrum, specified prevalence and operating threshold. We audit three medical vision-language models (BioMedCLIP, CheXficient, and MedSigLIP) and a general-domain OpenCLIP comparator on 12,200 chest radiograph records from four datasets (Montgomery, Shenzhen, TBX11K, and VinDr-CXR). Five fixed prompt families yield 244,000 model--image--prompt scores. No model leads every cohort and reliability criterion. Prompt-family changes alter AUROC in 21 of 48 multiplicity-controlled comparisons. Replacing healthy controls with sick non-tuberculosis controls reduces AUROC by 0.075--0.306 across all four models. On VinDr-CXR, the three medical models distinguish tuberculosis from no-finding cont
    
[^39]: 完备神经电子初始化加速材料密度泛函理论计算

    Complete Neural Electronic Initialization Accelerates Materials DFT

    [https://arxiv.org/abs/2609.21759](https://arxiv.org/abs/2609.21759)

    该论文提出了首个完备的机器学习方法来加速PAW形式下的材料平面波DFT计算，通过形式化七项准则并引入AugNet——首个PAW增强占据数通用等变模型和首个材料通用自旋密度模型——填补了现有方法中缺失的关键初始化组件。

    

    我们提出了首个在投影增强波（PAW）形式下加速材料平面波密度泛函理论（DFT）计算的完备机器学习方法。我们形式化了“完备神经电子初始化器”为实现实用的端到端PAW DFT加速所必须满足的七项准则。将这些准则应用于先前的工作，揭示了两个缺失的结构相关组件——增强占据数和自旋初始化——它们阻碍了现有方法提供完备的无参考初始化。受控消融实验表明，省略这些组件可能会消除或逆转仅预测平滑价电子密度的模型所获得的加速效果。我们通过引入AugNet来满足这些缺失的要求，AugNet是首个针对PAW增强占据数的通用等变模型，同时也是首个针对材料的通用自旋密度模型，它可以预测平滑的自旋差密度。

    arXiv:2609.21759v1 Announce Type: cross  Abstract: We present the first complete machine learning method for accelerating plane-wave density functional theory (DFT) in materials under the projector augmented wave (PAW) formalism. We formalize seven criteria that a \textit{Complete Neural Electronic Initializer} must satisfy for practical end-to-end PAW DFT acceleration. Applying these criteria to prior work reveals two missing structure-dependent components, augmentation occupancies and spin initialization, that prevent existing methods from providing complete reference-free initialization. Controlled ablations show that omitting these components can eliminate or reverse the acceleration obtained via models that only predict the smooth valence density. We satisfy these missing requirements by introducing AugNet, the first general equivariant model for PAW augmentation occupancies, and the first general spin density model for materials, which predicts the smooth spin-difference density 
    
[^40]: 拓扑与超参数的双层优化

    Bilevel Optimization of Topology and Hyperparameters (BOTH)

    [https://arxiv.org/abs/2609.21758](https://arxiv.org/abs/2609.21758)

    该论文提出通过对拓扑优化过程本身进行自动微分来获得“超梯度”，从而在双层优化框架下将超参数调优与设计优化同步进行，免去繁琐且依赖经验的手动调参。

    

    拓扑优化（TO）是迈向设计过程自动化的重要一步：给定一个可运行的仿真，拓扑优化可以通过对仿真进行微分并迭代改进设计，一键生成可行的原型。然而在实践中，拓扑优化充斥着“魔法数字”——即其调优会显著影响结果的超参数。找到合适的取值通常不仅需要深入的问题特定知识，还需要大量的反复试错。虽然从业者可以使用代理模型辅助的超参数优化作为替代方案，但这种方法需要通过精心的问题表述来严格限制超参数的数量。在此，我们提出使用自动微分对拓扑优化本身进行微分，由此产生“超梯度”，使我们能够在主优化的同时同步调优这些超参数。我们证明，仅评估拓扑优化的一两个步骤就足以……

    arXiv:2609.21758v1 Announce Type: new  Abstract: Topology optimization (TO) represents a significant step towards automating the design process: given a working simulation, TO can produce a viable prototype at the press of a button by differentiating the simulation and iteratively improving the design. In practice, however, TO is riddled with ``magic numbers''---hyperparameters whose tuning significantly affects the outcome. Finding the right values typically requires not only deep problem-specific knowledge but also extensive trial-and-error. While practitioners can use surrogate-assisted hyperparameter optimization as an alternative, this approach requires strictly limiting the number of hyperparameters through careful problem formulation. Here, we propose differentiating TO itself using automatic differentiation. This yields ``hypergradients'' that allow us to tune these hyperparameters in tandem with the primary optimization. We show that evaluating just one or two steps of TO is s
    
[^41]: GraphSkillEvo：图结构智能体技能的进化优化

    GraphSkillEvo: Evolutionary Optimization of Graph-Structured Agent Skills

    [https://arxiv.org/abs/2609.21749](https://arxiv.org/abs/2609.21749)

    提出GraphSkillEvo方法，将LLM智能体技能表示为图结构形式（节点表示执行步骤及操作指导，有向边表示步骤间转换），并通过进化优化克服非结构化技能缺乏工作流指导和搜索空间过大的问题。

    

    技能可以通过提供任务特定的程序性指导来提升大型语言模型（LLM）智能体的性能，而技能优化则通过迭代改进进一步提升其有效性。然而，现有的技能优化方法通常将技能表示为非结构化的自然语言指令，这带来了两个关键挑战：1）非结构化技能往往缺乏明确的工作流级别指导，且包含大量冗余，使LLM难以执行；2）不受约束的自然语言技能搜索空间巨大，导致技能优化效率低下。为应对这些挑战，我们提出将技能表示为图结构的自然语言产物。在图结构技能中，每个节点表示一个执行步骤及其操作指导，而有向边则编码了步骤之间依赖于上下文的转换。与非结构化技能相比，图结构技能……

    arXiv:2609.21749v1 Announce Type: new  Abstract: Skills can improve the performance of Large Language Model (LLM) agents by providing task-specific procedural guidance, while skill optimization further improves their effectiveness through iterative refinement. However, existing skill optimization methods typically represent skills as unstructured natural-language instructions, creating two key challenges: 1) Unstructured skills often lack explicit workflow-level guidance and contain substantial redundancy, making them difficult for LLMs to execute; 2) the vast search space of unconstrained natural-language skills makes skill optimization ineffective. To address these challenges, we propose representing skills as graph-structured natural-language artifacts. In graph-structured skills, each node represents an execution step together with its operational guidance, while directed edges encode context-dependent transitions between steps. Compared to unstructured skills, graph-structured ski
    
[^42]: 面向随机非凸—（强）凹极小极大优化的单循环随机投影阻尼外梯度方法

    Single-Loop Stochastic Projected Damped Extragradient Methods for Stochastic Nonconvex--(Strongly) Concave Minimax Optimization

    [https://arxiv.org/abs/2609.21747](https://arxiv.org/abs/2609.21747)

    提出了保持单循环结构的随机投影阻尼外梯度方法 SPDE 及其递归方差缩减变体 VR-SPDE，用于随机非凸—（强）凹极小极大优化，并同时为博弈平稳性和优化平稳性提供了更优的随机一阶预言机复杂度保证。

    

    针对随机非凸—（强）凹极小极大优化问题，我们提出了单循环随机投影阻尼外梯度方法，并对博弈平稳性和优化平稳性均给出了复杂度保证。我们的方法将随机投影阻尼外梯度（SPDE）方法与递归方差缩减变体 VR-SPDE 相结合，二者均保持单循环结构。在具有一致有界方差的无偏随机梯度预言机下，SPDE 能够找到 ε-博弈平稳点，其随机一阶预言机（SFO）复杂度在非凸—强凹和非凸—凹设置下分别为 O(κ ε⁻⁴) 和 O(ε⁻⁵)，其中 κ = L/μ。在随机梯度满足额外的均方 Lipschitz 条件下，VR-SPDE 将上述博弈平稳性复杂度分别改进至 O(κ^{3/2} ε⁻³) 和 O(ε^{-9/2})……

    arXiv:2609.21747v1 Announce Type: cross  Abstract: We develop single-loop stochastic projected damped extragradient methods for stochastic nonconvex--(strongly) concave minimax optimization, with complexity guarantees for both game stationarity (GS) and optimization stationarity (OS). Our approach combines a stochastic projected damped extragradient (SPDE) method with a recursive variance-reduced variant, VR-SPDE, both of which retain a single-loop structure. Under an unbiased stochastic gradient oracle with uniformly bounded variance, SPDE finds an $\varepsilon$-game-stationary point with stochastic first-order oracle (SFO) complexities of $O(\kappa\varepsilon^{-4})$ and $O(\varepsilon^{-5})$ in the nonconvex--strongly concave and nonconvex--concave settings, respectively, where $\kappa=L/\mu$. Under an additional mean-square Lipschitz condition on the stochastic gradients, VR-SPDE improves these GS complexities to $O(\kappa^{3/2}\varepsilon^{-3})$ and $O(\varepsilon^{-9/2})$, respect
    
[^43]: GEM-MPC：通过专家引导的规划平衡探索与利用

    GEM-MPC: Balancing Exploration and Exploitation through Expert-Guided Planning

    [https://arxiv.org/abs/2609.21735](https://arxiv.org/abs/2609.21735)

    GEM-MPC是一种基于MPPI的强化学习方法，通过将克隆规划器的策略与KL正则化探索策略相结合，并引入门控先验蒸馏机制选择性地利用存储的规划分布，从而改善规划与学习的交互，在高维连续控制中平衡探索与利用。

    

    在高维连续控制中实现有效探索仍然是强化学习中的一个核心挑战。基于规划的方法通过将在线规划与学习到的策略和价值函数相结合来解决这一问题，但其各组件在训练过程中可能出现不一致：学习到的采样策略可能偏离规划器的行为，而存储在经验回放中的规划分布会随着模型和价值函数的演化而变得过时。重新分析可以刷新这些目标，但计算成本高昂。我们提出GEM-MPC，这是一种基于MPPI的强化学习方法，旨在改善规划与学习之间的交互。GEM-MPC使用MPPI将一个训练用于克隆规划器的策略与一个围绕其进行探索的KL正则化策略相结合，在规划中提供互补的利用与引导式探索。我们进一步引入了门控先验蒸馏，它选择性地从存储的规划分布中学习……

    arXiv:2609.21735v1 Announce Type: new  Abstract: Effective exploration in high-dimensional continuous control remains a central challenge in reinforcement learning. Planning-based methods address this by combining online planning with learned policies and value functions, but their components can become misaligned during training: learned sampling policies may diverge from planner behavior, while planning distributions stored in replay become stale as the model and value function evolve. Reanalysis can refresh these targets, but at substantial computational cost. We propose GEM-MPC, an MPPI-based reinforcement learning method that improves the interaction between planning and learning. GEM-MPC uses MPPI to combine a policy trained to clone the planner with a KL-regularized policy that explores around it, providing complementary exploitation and guided exploration within planning. We further introduce Gated Prior Distillation, which selectively learns from stored planning distributions 
    
[^44]: SpecQuant：基于多父量化与投机解码的自适应大语言模型推理

    SpecQuant: Speculative Decoding with Multi-Parent Quantization for Adaptive LLM Inference

    [https://arxiv.org/abs/2609.21704](https://arxiv.org/abs/2609.21704)

    SpecQuant 提出了一种无需训练的框架，通过从共享基础模型派生多种量化变体（INT4、FP8、FP16）并结合投机解码，根据任务复杂度动态路由查询，从而实现自适应且高效的大语言模型推理。

    

    在消费级硬件上本地运行大语言模型（LLM）持续受到计算和内存限制的制约。流行的加速技术，如量化、投机解码和自适应推理，虽然能带来显著的速度提升，但通常需要重新训练、针对特定架构的调优或草稿模型。SpecQuant 是一个无需训练的框架，它将投机解码与多父量化相结合，以实现自适应、高效的大语言模型推理。SpecQuant 从一个共享的基础模型中派生出多个量化变体（INT4、FP8、FP16），并根据预测的任务复杂度动态路由查询：轻量级变体用于简单或事实性任务，全精度模型则用于复杂推理任务或长上下文输入。SpecQuant 的共享权重设计确保了投机解码能够获得足够的 token 接受率，同时避免了使用单独草稿模型所带来的兼容性问题。

    arXiv:2609.21704v1 Announce Type: new  Abstract: Running large language models (LLMs) locally continues to be limited by restrictions of compute and memory on consumer hardware. The popular acceleration technologies, such as quantization, speculative decoding, and adaptive inferencing, offer substantial speed boosts but usually necessitate retraining, per architecture tuning, or draft models. SpecQuant is a trainingfree framework, that combines speculative decoding with multiparent quantization to perform adaptive, efficient inference of LLMs. SpecQuant derives multiple quantized variants (INT4, FP8, FP16) from a shared base model, and dynamically routes queries based on predicted complexity; lightweight variants are used for simple or factual tasks, and full-precision models are used for complex reasoning tasks or long-context inputs. The shared-weight design of SpecQuant ensures sufficient token acceptance for speculative decoding without compatibility issues using separate draft par
    
[^45]: 带有类别不确定性的天文光谱贝叶斯分类

    Bayesian classification of astronomical spectra with class uncertainties

    [https://arxiv.org/abs/2609.21694](https://arxiv.org/abs/2609.21694)

    该论文面向4MOST巡天需求，系统比较了卷积神经网络、狄利克雷分布、蒙特卡洛dropout和贝叶斯神经网络+变分推断四种概率机器学习方法，实现了能够量化输入数据与预测不确定性的天文光谱多类别分类。

    

    背景：我们开发了一种概率机器学习方法，旨在为即将开展的4MOST巡天对恒星和河外目标的高、低分辨率光谱进行约10类的分类。为满足巡天要求，该方法应能够表达输入数据中的不确定性以及预测中引入的不确定性。目标：我们探索了四种不同的方法：(1)卷积神经网络(CNN)，(2)狄利克雷分布，(3)蒙特卡洛dropout (MCD)，(4)贝叶斯神经网络(BNN)+变分推断(VI)。训练和验证使用来自SDSS数据库的已标记光谱以及定制的4MOST模拟数据集进行。所有方法均采用相同的指标进行比较：准确率、曲线下面积(AUC)、期望校准误差(ECE)、香农熵、负对数似然(NLL)、Brier分数、训练时间和推理时间。方法……（摘要不完整）

    arXiv:2609.21694v1 Announce Type: cross  Abstract: Context: We developed a probabilistic machine learning method with the aim of performing the O(10)-way classification of low- and high-resolution spectra of stellar and extragalactic targets for the upcoming 4MOST survey. In fulfilment of the survey requirements, this method should be able to express uncertainty in the input data as well as uncertainty introduced in its prediction. Aims: Four different methods are explored: (1) convolutional neural networks (CNNs), (2) the Dirichlet distribution, (3) Monte Carlo dropout (MCD), (4) Bayesian neural Networks (BNNs) + variational inference (VI). Training and validation was performed using labelled spectra from the SDSS database and a custom 4MOST mock dataset. All the methods were compared in terms of the same metrics: accuracy, area under the curve (AUC), expected calibration error (ECE), Shannon entropy, negative log-likelihood (NLL), Brier score, training time, and inference time. Metho
    
[^46]: 等价布朗RKHS表示的优化几何

    Optimization Geometry of Equivalent Brownian RKHS Representations

    [https://arxiv.org/abs/2609.21693](https://arxiv.org/abs/2609.21693)

    本文在有限布朗RKHS框架下揭示了等价参数化（节点、增量、谱坐标）虽表示相同函数与内在范数却诱导不同优化几何的现象，证明节点与谱坐标的GD/SGD映射轨迹完全一致、增量GD为常数布朗/索伯列夫度量下因子为1/h的显式欧拉步，且增量海森矩阵的条件数不随网格分辨率增长。

    

    arXiv:2609.21693v1 公告类型：新 摘要：等价的有限参数化可以表示相同的函数和内在范数，却会诱导出不同的优化算法。我们在一个受控的有限布朗RKHS（再生核希尔伯特空间）中研究这一效应，该空间包含节点坐标、增量坐标和谱坐标三种表示。经典的有限元、RKHS插值、布朗协方差以及混合边界DCT恒等式，使得共享的假设类、布朗能量、逼近算子和坐标映射变得明确。我们的主要结果涉及这一固定模型的优化几何。在映射初始化、相同标量步长和相同小批量的条件下，节点坐标与谱坐标的GD/SGD具有完全相同的映射轨迹。增量GD是常数布朗/索伯列夫度量下的显式欧拉步，其因子为 $1/h$。对于布朗正则化最小二乘问题，$\kappa_2(\mathbf H_{\mathrm{inc}})\le 1+A/\rho$，在固定 $A$、$\rho>0$ 及所述归一化条件下，该条件数与网格分辨率 $G$ 无关。（摘要在此处被截断）

    arXiv:2609.21693v1 Announce Type: new  Abstract: Equivalent finite parameterizations can represent the same functions and intrinsic norm yet induce different optimization algorithms. We study this effect in a controlled finite Brownian RKHS with nodal, increment, and spectral coordinates. Classical finite-element, RKHS-interpolation, Brownian-covariance, and mixed-boundary DCT identities make the shared hypothesis class, Brownian energy, approximation operator, and coordinate maps explicit. Our main results concern the optimization geometry of this fixed model. With mapped initialization, identical scalar steps, and identical minibatches, nodal and spectral GD/SGD have exactly the same mapped trajectories. Increment GD is an explicit Euler step for the constant Brownian/Sobolev metric, with factor $1/h$. For Brownian-regularized least squares, $\kappa_2(\mathbf H_{\mathrm{inc}})\le1+A/\rho$, independently of grid resolution $G$ for fixed $A$, $\rho>0$, and the stated normalization. Und
    
[^47]: 基于测度量化的多域聚类

    Multi-Domain Clustering via Measure Quantization

    [https://arxiv.org/abs/2609.21664](https://arxiv.org/abs/2609.21664)

    提出了一种通过测度量化进行多域聚类的通用框架，通过最小化概率度量学习共享聚类原型，并利用最优传输实现协作式样本分配，配合小批量优化策略在保持性能的同时提升了可扩展性，在多个多域基准测试中优于经典方法。

    

    聚类是数据分析中的一项基础任务，通常通过基于质心的方法（如K-means）来解决。在这项工作中，我们提出了一个通过测度量化进行多域聚类的通用框架：给定来自多个域的样本，我们通过最小化每个域的概率测度与原型测度之间的概率度量（如Sinkhorn散度或最大均值差异MMD）来学习一组共享的聚类原型。随后，数据点通过最近质心或最优传输（一种耦合域内所有样本的协作策略）被分配到各个聚类中。小批量优化策略使拟合和分配过程都具备可扩展性，在保持聚类性能的同时降低了内存和计算成本。在涵盖图像、音频和传感器数据的5个多域基准测试上的实验结果表明，我们基于Sinkhorn的方法始终优于经典方法。

    arXiv:2609.21664v1 Announce Type: new  Abstract: Clustering is a fundamental task in data analysis, typically addressed through centroid-based methods such as K-means. In this work, we present a general framework for multi-domain clustering via measure quantization: given samples from multiple domains, we learn a shared set of cluster prototypes by minimizing a probability metric, such as the Sinkhorn divergence or the Maximum Mean Discrepancy, between each domain's probability measure and the measure of prototypes. Data points are then assigned to clusters either via nearest centroid, or via optimal transport, a collaborative strategy that couples all samples within a domain. A mini-batch optimization strategy makes both fitting and assignment scalable, reducing memory and computational cost while preserving clustering performance. Experimental results on 5 multi-domain benchmarks spanning image, audio and sensor data show that our Sinkhorn-based method consistently outperforms classi
    
[^48]: 超越高斯世界：潜在几何对JEPA至关重要

    Beyond Gaussian Worlds: Latent Geometry Matters for JEPAs

    [https://arxiv.org/abs/2609.21656](https://arxiv.org/abs/2609.21656)

    该论文将JEPA的可识别性理论从欧几里得高斯设定扩展到嵌入黎曼流形上的潜在变量，证明了当潜在变量均匀分布在球面上且表示匹配相同的球面分布时，每个最优表示都能在相差正交变换的意义上恢复潜在状态。

    

    近期的联合嵌入预测架构通过约束学习到的表示遵循指定的目标分布（如各向同性高斯分布或超球面上的均匀分布）来防止表示坍缩。Klindt等人（2026）证明，在他们的欧几里得假设下，匹配高斯目标可以在相差一个线性变换的意义上恢复高斯潜在变量，并且高斯分布是唯一具有这种保证的分布。我们将他们的分析扩展到支撑在嵌入黎曼流形上的潜在变量，并推导出关于潜在几何和正样本对动态的条件，在这些条件下，对齐和精确分布匹配能够保证线性恢复。特别地，当潜在变量均匀分布在球面上且表示被匹配到相同的球面分布时，每个最优表示都能在相差一个正交变换的意义上恢复潜在状态。

    arXiv:2609.21656v1 Announce Type: new  Abstract: Recent Joint-Embedding Predictive Architectures (JEPAs) prevent representation collapse by constraining learned representations to follow a prescribed target distribution, such as an isotropic Gaussian or the uniform distribution on a hypersphere. Klindt et al. (2026) showed that, under their Euclidean assumptions, matching a Gaussian target can recover Gaussian latent variables up to a linear transformation, and that the Gaussian is the unique distribution with this guarantee. We extend their analysis to latent variables supported on embedded Riemannian manifolds and derive conditions on the latent geometry and positive-pair dynamics under which alignment and exact distribution matching guarantee linear recovery. In particular, when the latent variables are uniformly distributed on a sphere and the representations are matched to the same spherical distribution, every optimal representation recovers the latent state up to an orthogonal t
    
[^49]: 分析语言模型嵌入空间中语言关系的线性特性

    Analysing the Linearity of Linguistic Relations in Language Model Embedding Spaces

    [https://arxiv.org/abs/2609.21655](https://arxiv.org/abs/2609.21655)

    该论文提出了一个基于约束线性逼近的框架来量化语言模型嵌入空间中各类语言关系的线性编码程度，发现屈折和派生关系近乎完美地线性编码，而词典和百科关系线性度明显较低，且RoBERTa和ModernBERT比GloVe更线性地编码关系。

    

    我们提出了一个框架，用于分析不同语言关系在语言模型嵌入空间中被线性编码的强度。我们通过对相关与不相关词对进行约束线性逼近来形式化线性编码，并将其应用于扩展的BATS数据集，该数据集涵盖了GloVe、RoBERTa和ModernBERT中的屈折、派生、词典和百科全书式关系。我们的实验表明，屈折关系和派生关系具有近乎完美的线性编码，但词典关系和百科全书式关系的误差显著更高，尤其体现在一对多和多对多的关联上。我们还发现，RoBERTa和ModernBERT通常比GloVe更线性地编码语言关系。这些结果表明，我们的框架能够揭示嵌入中哪些关系结构最易于线性访问，为探测和比较不同模型的关系几何提供了一个紧凑的工具。

    arXiv:2609.21655v1 Announce Type: new  Abstract: We propose a framework to analyse how strongly different linguistic relations are linearly encoded in language model embedding spaces. We formalise linear encoding via a constrained linear approximation over related and unrelated word pairs and apply this to an extended BATS dataset covering inflectional, derivational, lexicographic, and encyclopedic relations in GloVe, RoBERTa, and ModernBERT. Our experiments show near-perfect linear encodings for inflectional and derivational relations, but substantially higher errors for lexicographic and encyclopedic relations, especially for one-to-many and many-to-many associations. We also find that RoBERTa and ModernBERT generally encode relations more linearly than GloVe. These results indicate that our framework can reveal which relational structures are most linearly accessible in embeddings, offering a compact tool for probing and comparing relational geometry across models.
    
[^50]: 面向作物病害与虫害诊断的可配置多阶段视觉流水线

    Configurable Multi-Stage Vision Pipeline for Crop Disease and Pest Diagnosis

    [https://arxiv.org/abs/2609.21651](https://arxiv.org/abs/2609.21651)

    针对小农户仅凭一张田间照片进行作物诊断的场景，本文提出一种可配置的多阶段视觉流水线，支持调节照片质量拒绝阈值与置信度截止值、并可扩展添加作物与病虫害类别，同时基于来自四个国家的116万张真实照片揭示了现有生产系统的不足。

    

    Farmer.Chat 是 Digital Green 面向小农户的农业咨询服务。当作物看起来出现异常时，农户会拍摄照片并发送，这张照片本身就是全部的问题信息：没有症状描述，没有作物名称，往往连文字都没有。该服务必须仅凭这些照片判断图像是否可用、图中是什么作物、以及作物出了什么问题——而这些照片是用廉价手机在田间、光线不佳、镜头晃动的条件下拍摄的。目前执行这一任务的系统无法调整：它没有可调节的照片拒绝阈值，无法添加新的作物和问题类别，也没有可设置的置信度截止值。我们研究了从埃塞俄比亚、印度、肯尼亚和尼日利亚发送给 Farmer.Chat 的约116万张照片。生产环境中的质量门控拒绝了其判定图像的46.8%，进入诊断阶段的图像中有超过四分之一未能返回作物名称，而在被标注为“病害”的问题中有35.8%实际上是虫害，

    arXiv:2609.21651v1 Announce Type: cross  Abstract: Farmer.Chat is Digital Green's farm advisory service for smallholder farmers. When something looks wrong with a crop, the farmer takes a photograph and sends it, and that photograph is the whole question: no symptom described, no crop named, often no text at all. The service has to determine whether the picture can be used, what crop it shows, and what is wrong with it, from images taken on cheap phones in a field, in poor light and with a moving camera. The system doing this today cannot be adjusted. It has no adjustable thresholds for photograph rejection, crops and problems cannot be added, and there is no confidence cut-off to set.   We study about 1.16 million photographs sent to Farmer.Chat from Ethiopia, India, Kenya and Nigeria. The production quality gate rejected 46.8% of the images it judged, over a quarter of those reaching diagnosis returned no crop name, and 35.8% of the labelled problems filed under "disease" are pests, 
    
[^51]: 黎曼神经哈密顿流：测地线辛输运与可解释性

    Riemannian Neural Hamiltonian Flows: Geodesic Symplectic Transport and Interpretability

    [https://arxiv.org/abs/2609.21647](https://arxiv.org/abs/2609.21647)

    该论文提出黎曼神经哈密顿流，将黎曼流形上的固定动能、可学习标量势与显式测地线蛙跳积分器相结合，并从理论上揭示了学习到的哈密顿势的可解释性机制。

    

    哈密顿归一化流是一类颇具吸引力的生成模型，因为其相空间映射是可逆且保体积的，但大多数神经网络的构造都是在欧几里得空间中表述的。我们提出了黎曼神经哈密顿流，它将黎曼流形的固定动能、一个可学习的标量势以及一个显式的测地线蛙跳积分器结合在一起。我们的分析解释了如何使学习到的哈密顿量具有可解释性：每个可归一化的势都定义了一个隐式轮廓，且位置边缘分布最初沿着该轮廓与基准分布之间的相对得分方向加速；匹配势是可解释的特化形式，其隐式轮廓即为目标分布。在各向同性高斯情形下，该机制对应于相空间旋转。局部谐波分析将这一结果推广到流形上一般目标分布的每个模式附近。学习到的与目标之间的差距（摘要在此处截断）……

    arXiv:2609.21647v1 Announce Type: new  Abstract: Hamiltonian normalizing flows are attractive generative models because their phase-space maps are invertible and volume preserving, but most neural constructions are formulated in Euclidean space. We introduce Riemannian Neural Hamiltonian Flows, which combine the fixed kinetic energy of a Riemannian manifold, a learned scalar potential, and an explicit geodesic leapfrog integrator. Our analysis explains how the learned Hamiltonian can be made interpretable. Every normalizable potential defines an implicit profile, and the position marginal initially accelerates along the relative score between that profile and the base. The matched potential is the interpretable specialization for which the implicit profile is the target. In the isotropic Gaussian case, the mechanism corresponds to a phase-space rotation. A local harmonic analysis extends this result around each mode of a general target on a manifold. The gap between the learned and the
    
[^52]: 检测已解决，描绘尚未解决：是什么决定了全景X光片上的牙齿分割

    Detection is solved, delineation is not: what governs tooth segmentation on panoramic radiographs

    [https://arxiv.org/abs/2609.21628](https://arxiv.org/abs/2609.21628)

    该研究通过构建包含1,422张全景X光片、4万余个专家描绘牙齿多边形的大规模数据集并进行系统消融，发现输入分辨率是决定牙齿边界描绘精度的主导因素，而模型架构在域内对性能几乎没有影响。

    

    全景X光片上的自动牙齿分割与FDI编号是计算机辅助牙科诊断的基础，然而哪些因素决定性能仍不清楚。我们构建了一个包含1,422张全景X光片的数据集，其中含有42,142个由专家描绘的牙齿多边形，覆盖32类FDI分类体系；标注由30名牙科执业医师完成，并由另外两名牙医独立审核，并利用该数据集在统一评估协议下分离出输入分辨率、模型架构与解剖学先验的影响。首先，分辨率起主导作用：在640/1024/1280的受控消融实验中，掩码mAP50-95从0.656提升至0.710再到0.717，而mAP50始终稳定在约0.982。基于图像的配对自助法检验表明两个提升均显著（p < 0.001，p = 0.024），而mAP50的变化与零无法区分，即增加分辨率带来的是边界精度的提升，而非检测能力的提升。其次，在域内架构几乎无关紧要：参数量为基线2.1倍的基于查询的transformer……（摘要原文在此处截断）

    arXiv:2609.21628v1 Announce Type: cross  Abstract: Automatic tooth segmentation and FDI numbering on panoramic radiographs underpins computer-assisted dental diagnosis, yet which factors govern performance remains unclear. We assemble a corpus of 1,422 panoramic radiographs containing 42,142 expert-delineated tooth polygons across the 32-class FDI taxonomy, annotated by 30 dental practitioners and independently reviewed by two others, and use it to isolate input resolution, architecture and anatomical priors under a single evaluation protocol.   First, resolution dominates: across a controlled 640/1024/1280 ablation, mask mAP50-95 rises 0.656 -> 0.710 -> 0.717 while mAP50 stays flat at ~0.982. Both gains are significant under a paired bootstrap over images (p < 0.001, p = 0.024); neither mAP50 change is distinguishable from zero. Added resolution buys boundary precision, not detection. Second, architecture is nearly irrelevant in-domain: a query-based transformer with 2.1x the paramete
    
[^53]: 循环Transformer中以深度换时间

    Trading Depth for Time in Recurrent Transformers

    [https://arxiv.org/abs/2609.21605](https://arxiv.org/abs/2609.21605)

    该论文提出潜在循环Transformer（LRT），在词汇token之间插入共享骨干参数的潜在思考token，以受控方式比较了循环Transformer中额外计算用于时间递归（潜在思考步）与物理网络深度的效果差异。

    

    循环Transformer通过时间递归来增加计算深度，将每个token的高层隐藏状态馈入下一个token的计算中。这引出了一个自然的问题：额外的计算应该用于更多的时间步，还是更大的物理深度？我们使用潜在循环Transformer（Latent Recurrent Transformers, LRTs）来研究这个问题，它在解码时对每个词汇token仅保留一次骨干网络前向传播，并为比较这两种增加计算的方式提供了受控的实验环境。具体而言，我们在连续的词汇token之间插入一个潜在思考token。每个思考token通过与词汇token相同的 $L$ 层网络，共享骨干参数，并在预测下一个token之前提供额外的隐藏状态精炼阶段。我们将这种 $L$ 层的LRT与不含思考token的 $2L$ 层LRT进行比较。两者在解码期间对每个词汇token都执行 $2L$ 个Transformer块的计算……

    arXiv:2609.21605v1 Announce Type: new  Abstract: Recurrent Transformers increase computational depth through temporal recurrence, feeding each token's high-level hidden state into the computation of the next. This raises a natural question: is additional computation better spent on more temporal steps or greater physical depth? We investigate this question using Latent Recurrent Transformers (LRTs), which retain one backbone forward pass per vocabulary token during decoding and provide a controlled setting for comparing these two ways of adding computation. Specifically, we insert a latent thought token between consecutive vocabulary tokens. Each thought token passes through the same $L$ layers as a vocabulary token, sharing the backbone parameters and providing an additional stage of hidden-state refinement before predicting the next token. We compare this $L$-layer LRT against a $2L$-layer LRT without thought tokens. Both execute $2L$ Transformer blocks per vocabulary token during de
    
[^54]: 用于非定常转子叶片压力与气动弹性载荷预测的周期性神经映射

    Periodic Neural Mapping for Unsteady Rotor-Blade Pressure and Aeroelastic Load Prediction

    [https://arxiv.org/abs/2609.21590](https://arxiv.org/abs/2609.21590)

    提出周期性傅里叶神经映射（p-FNM）神经算子框架，将时间周期性嵌入模型，可从运行工况和任意时刻独立预测涡轮转子叶片的非定常压力场，避免误差累积并保持时间连续性，从而实现高效的气动弹性载荷预测。

    

    非定常气动载荷的精确预测仍然是涡轮机械设计中的一项重大挑战。高保真计算流体动力学（CFD）模拟成本高昂，而气动弹性关注量对压力场的时间演化高度敏感。本工作提出了周期性傅里叶神经映射，这是一种神经算子框架，用于预测在时序周期性数值假设下模拟的涡轮转子叶片上的非定常压力分布。该架构将时间周期性嵌入模型中，并学习从运行工况和时间到压力场的连续映射。与顺序潜空间方法不同，p-FNM可以在任意时刻独立预测压力场，在保持时间连续性的同时避免了误差累积。该模型在非定常转子叶片模拟数据库上进行了评估，并与基于降阶模型的基线方法进行了比较。

    arXiv:2609.21590v1 Announce Type: cross  Abstract: Accurate prediction of unsteady aerodynamic loads remains a major challenge in turbomachinery design. High-fidelity Computational Fluid Dynamics (CFD) simulations are expensive, while aeroelastic Quantities of Interest (QoI) depend sensitively on the temporal evolution of the pressure field. This work introduces periodic Fourier Neural Mapping (p-FNM), a neural-operator framework for predicting unsteady pressure distributions on turbine rotor blades simulated using the chorochronic numerical hypothesis. The architecture embeds temporal periodicity into the model and learns a continuous mapping from operating conditions and time to pressure fields. Unlike sequential latent-space approaches, p-FNM predicts pressure fields independently at any time, avoiding error accumulation while preserving temporal continuity. The model is evaluated on a database of unsteady rotor-blade simulations and compared with a reduced-order baseline based on a
    
[^55]: 面向通信高效脉冲神经网络的预测抑制层

    Predictive Suppression Layers for Communication-Efficient Spiking Neural Networks

    [https://arxiv.org/abs/2609.21583](https://arxiv.org/abs/2609.21583)

    提出了一种面向脉冲神经网络的最小预测编码框架，通过预测抑制层利用残差幅值动态门控，仅转发不可预测的“惊讶”活动，从而显著降低层间通信冗余。

    

    前馈脉冲神经网络（SNN）通常会不加区分地传播每一个产生的脉冲，而未考虑这些信息从信息论角度是否冗余。这种缺乏选择性的做法导致层间通信中的高冗余，带来高昂的开销，例如在涉及多核神经形态硬件或以通信为主导的物联网场景中，特征需要通过无线方式传输。为了应对这一挑战，我们通过为SNN引入一个最小化的预测编码框架，以局部处理换取更精简的网络通道。我们提出了共享同一个预测器模块的两种层变体：误差单元，负责传输带符号的脉冲残差；以及预测抑制，利用残差幅值进行动态门控，仅转发不可预测的、“令人惊讶”的活动。在N-MNIST和Spiking Heidelberg Digits（SHD）数据集上使用诊断指标进行评估。

    arXiv:2609.21583v1 Announce Type: cross  Abstract: Feedforward Spiking Neural Networks (SNNs) typically propagate every generated spike indiscriminately, disregarding whether the information is redundant from an information-theoretic perspective. This lack of selectivity induces high redundancy in inter-layer communication, creating an expensive overhead, e.g., in scenarios involving many-core neuromorphic hardware or communication-dominated Internet-of-Things (IoT) where features are transmitted wirelessly. To address this challenge, we trade localized processing for leaner network channels by introducing a minimal predictive coding framework for SNNs. We propose two layer variants sharing a predictor block: error units, which transmit signed spiking residuals, and predictive suppression, which uses residual magnitude to dynamically gate and forward only unpredictable, "surprising" activity. Evaluated on the N-MNIST and Spiking Heidelberg Digits (SHD) datasets using diagnostic metrics
    
[^56]: 加权量子信号处理：低深度多项式逼近及其在Kolmogorov-Arnold网络中的应用

    Weighted Quantum Signal Processing: Low-Depth Polynomial Approximation with Applications to Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2609.21567](https://arxiv.org/abs/2609.21567)

    本文提出加权量子信号处理（WQSP），通过为中心旋转算子引入权重函数突破了QSP的电路深度瓶颈和奇偶性约束，在实现任意有界一元多项式时将所需参数数量从线性级降至指数级缩减，并展示了其在Kolmogorov-Arnold网络中的应用。

    

    量子信号处理是一种用于生成和逼近一元多项式的强大量子框架。然而，QSP常常受到电路深度瓶颈以及可实现多项式类别的奇偶性约束的限制。在这项工作中，我们提出了加权量子信号处理，这是QSP的一种扩展，其中为中心旋转算子分配了一个权重函数。这一表述为QSP提供了更深入的理解，QSP作为WQSP在单位权重下的特例而出现。权重的选择决定了WQSP电路的结构和表达能力。当权重为大于1的自然数时，WQSP简化为QSP的剪枝版本，揭示了标准框架中的参数冗余。通过适当选择整数权重，WQSP在实现任意有界一元多项式所需的参数数量上实现了从线性到指数级的缩减。

    arXiv:2609.21567v1 Announce Type: cross  Abstract: Quantum Signal Processing is a powerful quantum framework for generating and approximating univariate polynomials. However, QSP is often limited by circuit-depth bottlenecks and parity constraints on the class of realizable polynomials. In this work, we introduce Weighted Quantum Signal Processing, an extension of QSP in which a weight function is assigned to the central rotation operator. This formulation provides a deeper understanding of QSP, which emerges as the special case of WQSP with unit weights. The choice of weights determines the structure and expressive capabilities of WQSP circuits. When the weights are natural numbers greater than one, WQSP reduces to a pruned version of QSP, revealing parameter redundancies in the standard framework. Through appropriate selection of integer weights, WQSP achieves linear-to-exponential reductions in the number of parameters required to realize arbitrary bounded univariate polynomials whi
    
[^57]: 论排斥性与吸引性教师：在自蒸馏中分离正确性与行为

    On Repulsive and Attractive Teachers: Separating Correctness from Behavior in Self-Distillation

    [https://arxiv.org/abs/2609.21561](https://arxiv.org/abs/2609.21561)

    该论文揭示了策略上自蒸馏中特权信息会将正确性信号与行为变化纠缠在一起——吸引性蒸馏会抑制探索性推理并使回答更短更自信，而排斥性蒸馏会使回答变长、意外触发思考模式且训练不稳定——因此提出应在自蒸馏中将正确性与行为分离。

    

    策略上自蒸馏通过让模型以特权信息为条件，并将由此产生的教师分布蒸馏回模型本身，从而提供密集的、词元级别的监督信号。然而，特权信息不仅会改变教师“知道什么”，还会改变教师“如何表现”，导致与正确性相关的学习信号与意外的行为变化纠缠在一起。我们在推理任务中研究了这一效应，对比了吸引性自蒸馏（使模型向特权教师靠拢）与排斥性自蒸馏（使模型远离特权教师）。我们发现，这两种目标都可能引发强烈且方向相反的行为变化：吸引会抑制探索性推理，并促使模型生成更短、更自信的回答；而排斥则会增加回答长度，可能触发模型意外切换到其潜在的思考模式，并最终变得不稳定。基于这些观察，我们研究……（原文摘要在此处截断）

    arXiv:2609.21561v1 Announce Type: cross  Abstract: On-policy self-distillation provides dense, token-level supervision by conditioning a model on privileged information and distilling the resulting teacher distribution back into the model. However, privileged information can change not only what the teacher knows, but also how it behaves, entangling correctness-relevant learning signals with unintended behavioral shifts. We study this effect in reasoning tasks by contrasting attractive self-distillation, which moves the model toward a privileged teacher, with repulsive self-distillation, which moves it away from a privileged teacher. We find that both objectives can induce strong and opposing behavioral shifts: attraction suppresses exploratory reasoning and promotes shorter, more confident responses, whereas repulsion increases response length, can trigger unintended switches into a model's latent thinking mode, and ultimately becomes unstable. Motivated by these observations, we stud
    
[^58]: OneBid：面向多样化oCPX广告场景的统一自动出价基础模型

    OneBid: A Unified Auto-Bidding Foundation Model for Diverse oCPX Advertising Scenarios

    [https://arxiv.org/abs/2609.21550](https://arxiv.org/abs/2609.21550)

    OneBid提出了一个统一的自动出价基础模型，通过从异构oCPX日志中学习可复用骨干网络并结合离线后训练实现场景适配，同时解决了多目标控制、严格延迟下的可扩展性以及安全离线策略改进三大挑战，将多样化的oCPX广告场景统一到单一模型中。

    

    自动出价是计算广告的核心，其策略需要在经济约束下最大化广告主的转化价值。该领域已从基于规则的控制器发展为强化学习以及决策Transformer（DT）等生成式方法。然而，这些方法与当前主流的优化成本-per-X（oCPX）范式日益不匹配：oCPX涵盖注册、购买等异构场景，且每个场景由单独的模型服务，导致流水线碎片化，跨场景建模不足。受大语言模型等基础模型的启发，将这些oCPX场景统一到单一模型中面临三大挑战：多目标控制、严格延迟约束下的可扩展容量，以及安全的离线策略改进。我们提出OneBid，一个统一的自动出价基础模型，它从异构的oCPX日志中学习可复用的骨干网络，并通过离线后训练将其适配到特定场景的部署中。

    arXiv:2609.21550v1 Announce Type: cross  Abstract: Auto-bidding is central to computational advertising, where strategies must maximize advertisers' conversion value under economic constraints. It has evolved from rule-based controllers to reinforcement learning and generative methods such as Decision Transformer (DT). Yet these methods increasingly mismatch the prevailing optimized cost-per-X (oCPX) paradigm, which spans heterogeneous scenarios (e.g., registration, purchase), each served by a separate model, leading to fragmented pipelines and underexploring cross-scenario modeling. Inspired by foundation models like LLMs, unifying these oCPX scenarios into one model raises three challenges: multi-objective control, scalable capacity under strict latency, and safe offline policy improvement. We present OneBid, a unified auto-bidding foundation model that learns a reusable backbone from heterogeneous oCPX logs and adapts it to scenario-specific deployments via offline post-training. Bu
    
[^59]: 基于多粒度状态空间模型的双兴趣序列化商品推荐

    Dual-Interest Sequential Product Recommendation With Multi-Granular SSM

    [https://arxiv.org/abs/2609.21548](https://arxiv.org/abs/2609.21548)

    提出 DSRec 模型，利用多粒度状态空间模型（SSM）将商品在长期与短期语义上下文中显式解耦为双兴趣表示，高效解决了序列推荐中的商品多义性与跨时间粒度动态行为建模问题。

    

    序列推荐旨在根据用户的历史行为预测其下一个将要交互的商品。Transformer 的进展显著提升了序列推荐的效果，但仍受限于成本效率。尽管状态空间模型（SSM）近来实现了高效的长程建模，但大多数现有方法将每个商品编码为单一的静态上下文角色，忽视了商品多义性现象。事实上，同一商品往往会根据用户上下文扮演不同的语义角色，而现有方法在捕捉不同时间粒度上的动态行为方面存在局限。在本工作中，我们提出了 DSRec，一种新颖的双兴趣跨 SSM 模型，它在长期和短期语义上下文中显式地解耦商品角色。序列商品被编码为通过历史聚合捕捉稳定偏好的长期兴趣嵌入，以及短期兴趣……（原文摘要不完整，此处截止）

    arXiv:2609.21548v1 Announce Type: new  Abstract: Sequential recommendation aims to predict the next item a user will interact with based on their historical behavior. Advances in Transformers have significantly improved sequential recommendation but are still limited by cost efficiency. Although State Space Models (SSMs) have recently enabled efficient long-range modeling, most existing methods encode each item with a single static contextual role, overlooking the phenomenon of item polysemy. In fact, the same item often plays different semantic roles depending on user context, and existing methods are limited in capturing dynamic behavior across different temporal granularities. In this work, we propose DSRec, a novel dual-interest cross-SSM model that explicitly disentangles item roles across long-term and short-term semantic context. Sequential items are encoded into long-term interest embeddings that capture stable preferences via historical aggregation, and a short-term interest b
    
[^60]: 纯化与调控：面向医学图像分类的共病感知多标签小样本学习

    Purification and Regulation: Comorbidity-Aware Multi-Label Few-Shot Learning for Medical Image Classification

    [https://arxiv.org/abs/2609.21541](https://arxiv.org/abs/2609.21541)

    该论文提出了原型纯化与调控（PPR）框架，通过共病感知的原型纯化机制和类间原型距离调控，解决了医学图像多标签小样本学习中原型被无关疾病信息污染以及忽视疾病间固有相关性的两大关键问题。

    

    多标签小样本学习（MLFSL）仍然是医学图像分析（MIA）中的一项重大挑战。当前基于度量的元学习方法在医学图像分析中面临两个关键局限：首先，传统的原型生成常常会纠缠不相关的疾病信息，导致原型被污染、性能下降；其次，以往研究通常在嵌入空间中强制类间可分性，在很大程度上忽略了疾病之间固有的相关性。为了克服这些挑战，我们提出了原型纯化与调控（PPR），这是一种新颖的面向医学图像分析的多标签小样本学习框架。PPR首先通过利用样本级的共病评分来强调疾病特异性特征，从而执行原型纯化，生成能够更好表征每种疾病的纯化原型。在这些纯化原型的基础上，PPR进一步通过引入疾病……（原文在此处截断）

    arXiv:2609.21541v1 Announce Type: cross  Abstract: Multi-label few-shot learning (MLFSL) remains a significant challenge in medical image analysis (MIA). Current metric-based meta-learning methods face two critical limitations in MIA. First, conventional prototype generation often entangles irrelevant disease information, leading to contaminated prototypes and degraded performance. Second, prior studies typically enforce inter-class separability in embedding space, largely neglecting the inherent correlations among diseases. To overcome these challenges, we propose Prototype Purification and Regulation (PPR), a novel MLFSL framework for MIA. PPR first performs prototype purification by leveraging sample-level comorbidity scores to emphasize disease-specific features, producing purified prototypes that better characterize each disease. Building upon these purified prototypes, PPR further addresses the underexplored problem of inter-class prototype distance in MIA by incorporating diseas
    
[^61]: MACE：面向多智能体系统的自适应记忆图记忆-智能体协同演化框架

    MACE: Memory-Agent Co-Evolution with Adaptive Memory Graphs for Multi-Agent Systems

    [https://arxiv.org/abs/2609.21533](https://arxiv.org/abs/2609.21533)

    提出MACE框架，通过自适应记忆图MemGoG将依赖关系组织为功能性记忆单元，并利用执行反馈实现记忆组织与智能体记忆使用的协同演化，从而提升多智能体系统中协作轨迹的复用与检索效果。

    

    基于大语言模型（LLM）的多智能体系统会生成协作轨迹，记录智能体如何规划任务、验证中间结果以及修复失败。复用这些过程需要保留动作的前置条件以及后续智能体所需的输出。我们的实证研究表明，将这些依赖关系分组为功能性记忆单元可以提高其保留率，而连接这些单元则能提升对任务所需的单元及关联链接的联合检索效果。即使每种组合的内容在不同格式间保持不变，单元的首选组合也会在指令与清单之间发生变化。根据每种组合与格式配对的实际执行结果来更新选择，其效果优于分别对组合和格式进行独立评分。这些发现催生了MACE——一个通过执行反馈来协同适应记忆组织与智能体记忆使用的记忆-智能体协同演化框架。其MemGoG结构将功能单元表示为……（原文摘要在此处截断）

    arXiv:2609.21533v1 Announce Type: new  Abstract: LLM-based multi-agent systems generate collaboration traces that record how agents plan tasks, verify intermediate results, and repair failures. Reusing these procedures requires preserving an action's prerequisites and the outputs needed by subsequent agents. Our empirical studies show that grouping these dependencies into functional memory units improves their retention, while connecting units increases retrieval of the units and links jointly required by a task. The preferred combination of units also changes between instructions and checklists, even when each combination's content is fixed across formats. Updating choices from the outcomes of each combination and format pairing outperforms scoring combinations and formats separately. These findings motivate MACE, a memory-agent co-evolution framework that adapts memory organization and agent memory use through execution feedback. Its MemGoG structure represents functional units as su
    
[^62]: OpenMAS-GCom：面向图增强多智能体系统的诊断性基准测试

    OpenMAS-GCom. A Diagnostic Benchmark for Graph-enhanced Multi-Agent Systems

    [https://arxiv.org/abs/2609.21527](https://arxiv.org/abs/2609.21527)

    提出OpenMAS-GCom诊断基准，通过在固定任务、模型、提示词和预算的条件下对协作单元、通信链路、共享信息和执行规则进行单组件受控干预，将图增强多智能体系统的性能差异准确归因于具体的通信结构和角色分配。

    

    图增强多智能体系统（G-MAS）通过通信图和角色分配来协调大型语言模型智能体，通信图和角色分配决定了智能体之间如何交换信息以及如何划分职责。然而，跨系统的最终得分比较混杂了模型、通信模式、角色和计算成本等多方面的差异，使得性能差异难以归因于特定的通信结构、角色分配和信息流。为解决这一评估归因问题，我们提出了OpenMAS-GCom，这是一个通过受控干预来诊断这些组件如何影响G-MAS性能的基准测试。我们通过协作单元、通信链路、共享中间信息和执行规则来表示系统。OpenMAS-GCom将原始系统与修改后的版本进行比较，修改版本在保持任务、模型、提示词和预算限制不变的情况下仅改变一个组件。我们对通信进行重连……（摘要原文在此处截断）

    arXiv:2609.21527v1 Announce Type: new  Abstract: Graph-enhanced multi-agent systems (G-MAS) coordinate large language model agents through communication graphs and role assignments, which determine how agents exchange information and divide responsibilities. However, final-score comparisons across systems combine differences in models, communication patterns, roles, and computation costs, making performance differences difficult to attribute to specific communication structures, role assignments, and information flows. To address this evaluation attribution problem, we introduce OpenMAS-GCom, a benchmark for diagnosing how these components affect G-MAS performance through controlled interventions. We represent systems through collaboration units, communication links, shared intermediate information, and execution rules. OpenMAS-GCom compares original systems with versions modified by changing one component while keeping tasks, models, prompts, and budget limits fixed. We rewire communi
    
[^63]: IncentRL：偏好引导与任务性能之间的权衡

    IncentRL: The Trade-Off Between Preference Guidance and Task Performance

    [https://arxiv.org/abs/2609.21525](https://arxiv.org/abs/2609.21525)

    IncentRL框架通过在奖励中加入KL惩罚形式的偏好引导，并从理论上刻画其对外部任务价值的扰动界以及保持最优策略的条件，揭示了偏好引导与任务性能之间的权衡关系。

    

    基于偏好的奖励塑形可以引导强化学习，但将偏好信号加入奖励中可能会在无意中改变原本要优化的任务。我们通过IncentRL来解决这一问题，该框架在引入偏好引导的同时，明确刻画了其对外部任务性能的影响。IncentRL在指定的结果分布与偏好分布之间添加了Kullback-Leibler（KL）惩罚。对于具有有界塑形成本的有限折扣马尔可夫决策过程，我们推导了外部价值扰动界，建立了保持原始最优策略的充分严格动作间隔条件，并通过折扣累积偏好成本刻画了大权重情形。精确的例子阐明了这些理论保证的局限性，包括并列最优和支持不匹配的情况。我们研究了一种使用手工设计的基于距离的结果代理和固定偏好分布的实际实现方式。

    arXiv:2609.21525v1 Announce Type: new  Abstract: Preference-based reward shaping can guide reinforcement learning, but adding preference signals to the reward may unintentionally change the task being optimized. We address this problem with IncentRL, a framework that introduces preference guidance while explicitly characterizing its effect on external-task performance. IncentRL adds a Kullback--Leibler (KL) penalty between a specified outcome distribution and a preferred distribution. For finite discounted Markov decision processes with bounded shaping costs, we derive an external-value perturbation bound, establish a sufficient strict-action-gap condition for preserving the original optimal policy, and characterize the large-weight regime through discounted cumulative preference cost. Exact examples clarify the limits of these guarantees, including tied optima and support mismatch. We study a practical implementation using a hand-designed, distance-based outcome proxy, a fixed prefere
    
[^64]: 什么必须留存？资源充足学习中任务信息与状态的精确前沿

    What Must Survive? Exact Task-Information--State Frontiers for Resource-Sufficient Learning

    [https://arxiv.org/abs/2609.21523](https://arxiv.org/abs/2609.21523)

    该文刻画了在下游任务未完全确定时压缩系统所需状态的精确前沿：给定 K 种先行任务建议，最小保留状态等于将任务集划分为至多 K 块后各块联合任务算子最大秩的最小值，并证明寻找最优建议划分是强 NP 难的。

    

    一个系统可能在其下游任务尚未完全确定之前就需要被压缩。我们探究此时必须保留多少状态，以及有限的先行任务信息能够节省多少。对于一个有限的线性任务族，任务消息在状态形成之前被揭示，而确切的任务只在之后才揭晓。对于大小为 K 的建议字母表，精确前沿为 p\*(K) = min_{𝒫 为 𝒰 的划分且 |𝒫|≤K} max_{C∈𝒫} rank(T_C)，而 b 比特前沿通过取 K = min(2^b, |𝒰|) 得到。因此，先行任务信息是通过“联合任务算子低秩”的划分来减少所需状态的。我们还给出了近似的奇异值前沿、公共核下界与精确的直和定律，并证明寻找最优建议划分是强 NP 难的；该难度在每个固定的正近似容差下依然存在。三个例子阐释了这一结果，其中包括一个良态的 softmax 注意力……（原文摘要在此截断）

    arXiv:2609.21523v1 Announce Type: new  Abstract: A system may be compressed before its downstream task is fully known. We ask how much retained state is then necessary and how much can be saved by limited advance task information.   For a finite family of linear tasks, a task message is revealed before state formation and the exact task only afterwards. For an advice alphabet of size $K$, the exact frontier is \[ p^*(K)= \min_{\substack{\Pcal\text{ partition of }\U\\|\Pcal|\le K}} \max_{C\in\Pcal}\rank(T_C), \] with the $b$-bit frontier obtained by setting $K=\min(2^b,|\U|)$. Thus advance task information reduces state through partitions whose joint task operators have low rank.   We also give an approximate singular-value frontier, a common-core lower bound and exact direct-sum law, and strong NP-hardness of finding an optimal advice partition. The hardness persists at every fixed positive approximation tolerance.   Three examples illustrate the result. A well-conditioned softmax atte
    
[^65]: ServeGuard：在不泄露所认证读取因子的前提下，对操作者不可见信道实现可验证的有界残差禁闭

    ServeGuard: Verifiable, Bounded-Residual Confinement of Operator-Invisible Channels Without Revealing the Certified Read Factor

    [https://arxiv.org/abs/2609.21515](https://arxiv.org/abs/2609.21515)

    提出ServeGuard，通过让发布者构建仅经由安全监控器覆盖方向读取输入的适配器，并以零知识证明验证这一性质（且不泄露所认证的读取因子），从而在结构上消除监控器盲区中无法被检测的后门信道。

    

    面向开放权重语言模型的第三方适配器以不透明的权重矩阵形式发布；接收方若不信任发布者或不去检查权重（即发布者的核心资产），就无法确认适配器是否隐藏了后门。对于一类重要的威胁（载荷被放置在安全监控器在结构上无法察觉的盲区），检测作为防御手段是不健全的：任何经由所声明监控器进行因式分解的检测器在其盲子空间上都是不变的，而且诚实适配器与后门适配器在我们评估的每一个盲子空间统计量上都相互重叠，因为良性适配同样会使用该子空间。我们并非去检测这一信道，而是使其在结构上不存在，并证明我们确实做到了。发布者构建适配器，使其仅通过监控器所覆盖的方向读取输入，并以零知识方式证明这一点，同时不泄露其所认证的读取因子的任何信息。该证书之所以廉价，是因为其中昂贵的部分——识别监控器的盲区……

    arXiv:2609.21515v1 Announce Type: cross  Abstract: Third-party adapters for open-weight language models ship as opaque weight matrices; a recipient cannot check whether an adapter hides a backdoor without trusting the publisher or inspecting the weights, the publisher's core asset. For one important class (payloads placed where a safety monitor is structurally blind), detection is unsound as a defense: every detector that factors through the declared monitor is invariant on its blind subspace, and honest and backdoored adapters overlap on every blind-subspace statistic we evaluate, because benign adaptation uses that subspace too. Rather than detect this channel, we make it structurally \emph{absent} and prove that we did. The publisher builds the adapter to read the input only through directions the monitor covers and proves this in zero knowledge, revealing nothing about the read factor it certifies. The certificate is cheap because the expensive part, identifying the monitor's blind
    
[^66]: 基于认知不确定性的自适应轨迹展开截断方法用于高效的离线世界模型训练

    Adaptive Rollout Truncation Based on Epistemic Uncertainty for Efficient Offline World Model Training

    [https://arxiv.org/abs/2609.21482](https://arxiv.org/abs/2609.21482)

    提出了一种由认知不确定性驱动的自适应轨迹展开截断策略，当不确定性超过预热阶段校准的阈值时即终止自回归展开，从而在提升长时程预测能力的同时降低计算成本并避免放大早期训练误差。

    

    准确的神经世界模型是基于模型的机器人技术的核心，它使机器人能够从先前观察到的轨迹中预测未来状态。多步自回归训练可以改善长时程预测能力，但固定的展开步长也会增加计算成本，并且在模型仍不准确时可能会放大早期训练误差。现有的训练方案通常在整个优化过程中使用相同的展开长度，而与模型当前的预测可靠性无关。我们提出了一种由认知不确定性驱动的自适应展开策略，用于遵循自动课程训练方案的离线世界模型训练。该模型不再总是展开到固定的步长，而是当认知不确定性超过从预热阶段校准得到的阈值时终止自回归展开。我们研究了两种不确定性估计器：具有共享循环骨干网络的五头集成模型和蒙特卡洛Dropout。

    arXiv:2609.21482v1 Announce Type: cross  Abstract: Accurate neural world models are central to model-based robotics, where they enable robots to predict future states from previously observed trajectories. Multi-step autoregressive training improves long-horizon prediction, but fixed rollout horizons also increase computational cost and can amplify early training errors when the model is still inaccurate. Existing training schemes typically use the same rollout length throughout optimization, independent of the model's current predictive reliability.   We propose an epistemic uncertainty-driven adaptive rollout strategy for offline world model training following an auto-curriculum training scheme. Instead of always unrolling to a fixed horizon, the model terminates autoregressive rollouts once epistemic uncertainty exceeds a threshold calibrated from a warm-up phase. We study two uncertainty estimators: a five-head ensemble with a shared recurrent backbone and Monte Carlo Dropout. A tw
    
[^67]: 留一被试评估下的高效架构搜索

    Efficient Architecture Search under Leave-One-Subject-Out Evaluation

    [https://arxiv.org/abs/2609.21457](https://arxiv.org/abs/2609.21457)

    提出PainNAS方法，通过基于块且控制数据泄漏的方式在被试间共享架构搜索，将LOSO评估下的NAS搜索次数从N次降至B次（B远小于N），在BioVid热痛数据集上以大幅更少的参数和计算量实现了相当的疼痛评估准确率。

    

    深度神经网络架构被广泛应用于自动化疼痛评估系统中的信号处理。然而，尽管神经架构搜索（NAS）具有潜在的效率优势，架构设计在很大程度上仍然是一项人工任务。将NAS嵌入留一被试（Leave-One-Subject-Out, LOSO）评估中在计算上代价高昂，因为完全嵌套的实现需要N次独立的架构搜索，并且假设训练成本近似线性，其复杂度可达O(N^2)。我们提出了一种基于块（block-based）、控制数据泄漏的方法，通过在多个被试之间共享NAS运行，将搜索次数从N减少到B（其中B远小于N），该方法被命名为PainNAS。在BioVid热痛数据集上，PainNAS以显著更少的参数量和浮点运算次数（FLOPs）实现了相当的被试级准确率。

    arXiv:2609.21457v1 Announce Type: new  Abstract: Deep neural architectures are widely used for signal processing in automated pain assessment systems. However, architecture design has remained largely a manual task despite the potential efficiency benefits of Neural Architecture Search (NAS). Embedding NAS in a Leave-One-Subject-Out (LOSO) evaluation is computationally demanding because a fully nested implementation requires $N$ independent architecture searches and, assuming approximately linear training cost, scales as $\mathcal{O}(N^2)$. We propose a block-based, leakage-controlled approach that shares NAS runs between subjects, reducing the number of searches from $N$ to $B$, where $B \ll N$, dubbed PainNAS. On the BioVid Heat Pain dataset, PainNAS yields comparable subject-level accuracy with substantially fewer parameters and FLOPs.
    
[^68]: 通过狄利克雷重采样提升自助聚合的预测性能

    Improving the Predictive Performance of Bootstrap Aggregating by Dirichlet Resampling

    [https://arxiv.org/abs/2609.21454](https://arxiv.org/abs/2609.21454)

    提出基于狄利克雷重采样的两种随机森林变体（DM和DW），通过浓度参数调节样本重加权以降低树间相关性，在几乎不增加运行时间的情况下提升了预测性能。

    

    我们重新审视了Breiman的观察，即在不削弱单棵树的前提下降低树间相关性可以改进随机森林。基于这一原则，我们提出了两个变体：狄利克雷-多项式自助聚合随机森林（DM）和狄利克雷加权随机森林（DW）。两者都通过浓度参数 $\alpha>0$ 来调节样本重加权。我们提供了一个简单的理论判据，阐明了这些变体在何时与标准随机森林的表现无异，并利用该判据指导一种轻量级的调参策略。在公开分类基准数据集的受控评估中，DM和DW始终具有竞争力，且往往优于其他随机森林（RF）基线方法，同时额外运行时间可忽略不计。

    arXiv:2609.21454v1 Announce Type: cross  Abstract: We revisit Breiman's observation that reducing inter-tree correlation without weakening individual trees can improve random forests. Building on this principle, we introduce two variants: Dirichlet-Multinomial Bagging Random Forest (DM) and Dirichlet-Weighted Random Forest (DW). Both modulate sample reweighting via a concentration parameter $\alpha>0$. We provide a simple theoretical criterion that clarifies when these variants behave indistinguishably from standard random forests, and we use it to guide a lightweight tuning strategy. In a controlled evaluation on public classification benchmarks, DM and DW are consistently competitive and often stronger than other random-forest (RF) baselines, with negligible additional runtime.
    
[^69]: 通过激活引导补偿与正交残差理解大语言模型量化

    Understanding LLM Quantization through Activation-Guided Compensation and Orthogonal Residuals

    [https://arxiv.org/abs/2609.21450](https://arxiv.org/abs/2609.21450)

    该论文将LLM量化误差精确分解为激活引导的权重补偿项和正交残差，并对残差进行理论界定，从而为Hadamard旋转、符号选择和通道缩放等量化变换设计提供了实用准则。

    

    训练后权重-激活量化可以降低大语言模型的内存和推理成本，但由于激活异常值会降低有效量化分辨率，激进的W4A4量化仍然十分困难。尽管权重优化、逐通道缩放和正交旋转可以缓解这一问题，但它们所处理的误差分量及其相互关系仍不明确。通过将局部权重-激活量化误差精确分解为激活引导的权重补偿项和正交残差，我们利用持续的逐通道异常值和常规激活量对残差进行了界定。这一分解明确了哪些误差分量可以通过权重补偿来处理，哪些需要通过变换设计来解决。随后，我们利用残差界推导出应用随机Hadamard旋转、符号选择和通道缩放的实用指南。特别是，该分析……

    arXiv:2609.21450v1 Announce Type: new  Abstract: Post-training weight-activation quantization reduces the memory and inference costs of large language models, but aggressive W4A4 quantization remains difficult because activation outliers degrade effective quantization resolution. Although weight optimization, channel-wise scaling, and orthogonal rotation mitigate this problem, the error components they address and their relationship remain unclear. Using an exact decomposition of local weight-activation quantization error into an activation-guided weight compensation term and an orthogonal residual, we bound the residual using persistent channel-wise outlier and regular activation quantities. This decomposition clarifies which error components can be addressed by weight compensation and which require transformation design. We then use the residual bounds to derive practical guidelines for applying randomized Hadamard rotation, sign selection, and channel scaling. In particular, the ana
    
[^70]: FootQuery：基于未来落足点引导的深度历史检索的人形机器人感知运动

    FootQuery: Future-Touchdown-Guided Retrieval from Depth History for Perceptive Humanoid Locomotion

    [https://arxiv.org/abs/2609.21447](https://arxiv.org/abs/2609.21447)

    提出FootQuery框架，以每只脚预测的未来触地点作为查询，从历史深度观测中检索触地时已不可见的地形信息，从而提升人形机器人在复杂地形上的感知运动能力。

    

    人形机器人在复杂地形上的运动需要预判在触地时可能已不再可见的落足点。有限的相机覆盖范围和自遮挡使得从早期观测中检索相关地形信息成为必要。我们提出了FootQuery，一个感知运动框架，它使用每只脚预测的下次触地点来查询深度历史。该策略从本体感觉预测触地位置及其不确定性，并利用这些分布以及每只脚的特征来查询稀疏采样的历史深度帧。在训练期间，实际发生的接触会被投影到历史图像中，以在这些接触可见的区域监督检索。检索到的每脚特征与全局视觉记忆融合以生成控制动作。渐进式力辅助课程支持早期探索，而事件一致的踏面中线塑造鼓励协调的楼梯……

    arXiv:2609.21447v1 Announce Type: cross  Abstract: Humanoid locomotion over complex terrain requires anticipating footholds that may no longer be visible at touchdown. Limited camera coverage and self-occlusion make it necessary to retrieve relevant terrain information from earlier observations. We present FootQuery, a perceptive locomotion framework that queries depth history using each foot's predicted next touchdown. The policy predicts touchdown locations and uncertainty from proprioception and uses these distributions, together with per-foot features, to query sparsely sampled historical depth frames. During training, realized contacts are projected into historical images to supervise retrieval at the regions where those contacts were visible. The retrieved per-foot features are fused with global visual memory to generate control actions. A progressive force-assistance curriculum supports early exploration, while event-consistent tread-midline shaping encourages coordinated stair 
    
[^71]: 最优随机化正规在线学习

    Optimal Randomized Proper Online Learning

    [https://arxiv.org/abs/2609.21445](https://arxiv.org/abs/2609.21445)

    本文证明了随机化正规在线学习的最优期望错误界为 $O(\mathtt{L}(\mathcal{H}) \log T)$，将此前已知的 $O(\mathtt{L}(\mathcal{H}) \log^6 T)$ 界显著改进，并在最坏情况下达到通用常数因子内的最优。

    

    我们证明了，使用随机化正规学习算法在线学习函数类 $\mathcal{H}$ 的最优期望错误界为 $O(\mathtt{L}(\mathcal{H}) \log T)$，其中 $\mathtt{L}(\mathcal{H})$ 是 $\mathcal{H}$ 的 Littlestone 维数，$T$ 是时间范围。我们的结果改进了 Daskalakis 和 Golowich（STOC 2022）此前给出的已知最优界 $O(\mathtt{L}(\mathcal{H}) \log^6 T)$，并且对于最坏情况下的函数类，该界在通用常数因子内是最优的。

    arXiv:2609.21445v1 Announce Type: new  Abstract: We prove that the optimal expected mistake bound of online learning a function class $\mathcal{H}$ by a randomized proper learning algorithm is $O(\mathtt{L}(\mathcal{H}) \log T)$, where $\mathtt{L}(\mathcal{H})$ is the Littlestone dimension of $\mathcal{H}$ and $T$ is the time horizon. Our result improves upon the previously best known bound of $O(\mathtt{L}(\mathcal{H}) \log^6 T)$ given by Daskalakis and Golowich (STOC 2022), and is optimal up to a universal constant for worst-case classes.
    
[^72]: GVPO++：面向大语言模型后训练与在线策略蒸馏的组方差策略优化

    GVPO++: Group Variance Policy Optimization for LLM Post-Training and On-Policy Distillation

    [https://arxiv.org/abs/2609.21432](https://arxiv.org/abs/2609.21432)

    GVPO通过将KL约束奖励最大化的解析解融入梯度加权方案，解决了GRPO因依赖重要性采样导致的训练不稳定问题，同时保证唯一最优解并支持灵活的采样分布，可用于大语言模型后训练与在线策略蒸馏。

    

    后训练在提升大语言模型（LLM）的推理能力和特定任务专长方面起着关键作用。尽管最近在后训练方法方面取得了进展，例如组相对策略优化（GRPO），但其实际部署仍然受到因依赖重要性采样而导致的训练不稳定性的阻碍。我们提出了组方差策略优化（GVPO），这是一种新颖的后训练方法，它将KL约束奖励最大化的解析解融入到其梯度加权方案中。这一构造提供了直观的解释：GVPO的梯度对应于隐式奖励的中心距离与实际奖励的中心距离之间的均方误差。GVPO具有两个关键优势：（1）它保证唯一的最优解，精确对应于KL约束奖励最大化目标；（2）它支持灵活的采样分布，无需重要性采样（原文在此处截断）。

    arXiv:2609.21432v1 Announce Type: new  Abstract: Post-training plays a pivotal role in enhancing the reasoning capabilities and task-specific expertise of large language models (LLMs). Despite recent advances in post-training methods, such as Group Relative Policy Optimization (GRPO), their practical deployment remains impeded by training instability arising from the reliance on importance sampling.   We introduce Group Variance Policy Optimization (GVPO), a novel post-training method that integrates the analytical solution of KL-constrained reward maximization into its gradient weighting scheme. This formulation provides an intuitive interpretation: GVPO's gradient corresponds to the mean squared error between the central distance of implicit rewards and that of actual rewards. GVPO offers two key advantages: (1) it guarantees a unique optimal solution, exactly to the KL-constrained reward maximization objective, and (2) it enables flexible sampling distributions without requiring imp
    
[^73]: 基于KKT重构的均值-方差投资组合优化决策聚焦学习

    Decision-Focused Learning for Mean-Variance Portfolio Optimization via KKT-Based Reformulation

    [https://arxiv.org/abs/2609.21427](https://arxiv.org/abs/2609.21427)

    该论文提出了一种将KKT最优性条件融入单层优化的重构方法，使决策聚焦学习能够直接应用于均值-方差投资组合优化，从而解决了现有方法中预测模型训练与约束优化决策之间结构性不匹配的问题。

    

    均值-方差投资组合优化（MVO）是数据驱动资产管理中的核心框架。一种被广泛采用的方法是两阶段框架：首先预测预期收益，然后基于这些预测求解优化问题，其中预测模型通过最小化预测误差进行训练。然而，这种预测目标与下游投资组合决策的质量并不一致。决策聚焦学习（DFL）直接在学习过程中最小化下游决策损失，因此已成为一个有前景的方向。然而，现有的MVO决策聚焦学习方法依赖于代理损失或约束松弛来保证可解性，导致预测模型训练与评估时求解的约束MVO之间存在结构性不匹配。我们提出了一种单层优化形式，将下层MVO问题的Karush-Kuhn-Tucker（KKT）最优性条件纳入其中。

    arXiv:2609.21427v1 Announce Type: new  Abstract: Mean-variance portfolio optimization (MVO) is a central framework in data-driven asset management. A widely adopted approach is a two-stage framework that first predicts expected returns and then solves the optimization problem based on these predictions, with the predictive models trained by minimizing prediction errors. However, this objective of prediction is not aligned with the quality of the downstream portfolio decision. Decision-focused learning (DFL), which directly minimizes the downstream decision loss within the learning process, has thus emerged as a promising direction. However, existing DFL approaches to MVO rely on surrogate losses or constraint relaxations for tractability, creating a structural mismatch between predictive model training and the constrained MVO solved at evaluation. We propose a single-level optimization formulation that incorporates the Karush-Kuhn-Tucker (KKT) optimality conditions of the lower-level M
    
[^74]: 追溯零样本时间序列预测背后的证据：一种以证据来源为先的分类法与审计框架

    Tracing the Evidence Behind Zero-Shot Time-Series Forecasting: A Source-First Taxonomy and Audit Framework

    [https://arxiv.org/abs/2609.21425](https://arxiv.org/abs/2609.21425)

    本文主张将零样本时间序列预测视为“证据可及性声明”，提出了一个以证据来源为先的分类法（区分冻结LLM先验复用、参数化时间序列预训练和检索增强外部记忆三种来源），并配套任务接口、预测对象与评分、预测时上下文和资源预算四项审计问题，构成完整的审计框架。

    

    零样本时间序列预测（TSF）通常被描述为在不针对目标数据进行参数更新的情况下进行预测，但这一训练状态条件并未说明系统可以使用哪些证据。一个以序列化数值作为提示的冻结语言模型、一个在广泛预测语料库上预训练的时间序列模型，以及一个检索增强的预测器，都可能满足“无更新”条件，同时却利用了不同的可迁移证据。因此，本文主张应将零样本TSF作为一种“证据可及性声明”来加以规范。我们提出了一种以来源为先的分类法，将三种主要证据来源——冻结LLM先验复用、参数化时间序列预训练以及检索增强的外部记忆——与实现它们的具体架构区分开来。在确定证据来源之后，仍需回答四个额外的审计问题：任务接口、预测对象与评分方式、预测时的上下文以及资源预算。

    arXiv:2609.21425v1 Announce Type: new  Abstract: Zero-shot time-series forecasting (TSF) is often described as forecasting without target-specific parameter updates, but that training-status condition does not specify what evidence the system may use. A frozen language model prompted with serialized values, a time-series model pretrained on broad forecasting corpora, and a retrieval-augmented forecaster may all satisfy the no-update condition while drawing on different transferable evidence. This paper argues that zero-shot TSF should therefore be governed as an evidence-access claim. We propose a source-first taxonomy that separates three primary evidence sources---frozen LLM prior reuse, parametric time-series pretraining, and retrieval-augmented external memory---from the architectures that implement them. After the source is identified, four additional audit questions remain: task interface, forecast object and scoring, prediction-time context, and resource budget. The resulting ag
    
[^75]: 面向深度ReLU表示的布朗头：激活质量与同样本选择的代价

    Brownian Heads for Deep ReLU Representations: Activation Mass and the Cost of Same-Sample Selection

    [https://arxiv.org/abs/2609.21422](https://arxiv.org/abs/2609.21422)

    该论文提出“布朗头”复杂度分析框架，以激活质量给出深度ReLU表示的精确Rademacher复杂度界，并首次量化了在同一批样本上选择隐藏特征所产生的额外选择成本，将其与实现尺度分离开来。

    

    深度表示学习通常在同一批样本上选择隐藏特征并拟合最终预测器，因此选择之后进行的固定特征分析可能会遗漏选择成本。我们研究了深度ReLU表示后接加性或Lévy-Brownian再生核希尔伯特空间（RKHS）中有界范数预测器的条件经验Rademacher复杂度，并将其称为布朗头。对于固定表示，我们推导出精确的对偶恒等式，并给出以激活质量（即观测到的隐藏向量的平均范数）刻画的紧致界。在同样本选择下，表示的上确界会诱导出一个二次Rademacher过程。借助布朗层蛋糕表示与高斯投影恒等式，我们将其归约为逐坐标或带符号投影的阈值迹，从而将实现尺度与选择复杂度分离开来。对于输入两两互不相同的样本，显式的标量ReLU函数族在实现迹水平上以通用常数匹配有限迹速率与VC速率……（原文在此处截断）

    arXiv:2609.21422v1 Announce Type: cross  Abstract: Deep representation learning often selects hidden features and fits the final predictor on the same sample, so fixed-feature analysis performed after selection can omit selection cost. We study the conditional empirical Rademacher complexity of deep ReLU representations followed by bounded-norm predictors in additive or L\'evy-Brownian RKHSs, termed Brownian heads. For a fixed representation, we derive an exact dual identity and sharp bounds in terms of activation mass, the average norm of the observed hidden vectors. Under same-sample selection, the representation supremum induces a quadratic Rademacher process. Brownian layer-cake and Gaussian-projection identities reduce it to coordinatewise or signed projected threshold traces, separating realized scale from selection complexity. For samples with pairwise-distinct inputs, explicit scalar ReLU families match the finite-trace and VC rates up to universal constants at the realized tra
    
[^76]: 深度循环语言模型中的预测动力学

    Prediction Dynamics in Depth-Recurrent Language Models

    [https://arxiv.org/abs/2609.21383](https://arxiv.org/abs/2609.21383)

    本文提出一种将深度循环语言模型的预测更新分解为共同平移、相对胜者的方向以及竞争者更新与分数差距配对的间隔刻画方法，解释了中间预测与最终预测一致但分数仍变化的机制，并据此将确定最终答案所需的计算深度额外减少22.5-34.4%。

    

    深度循环语言模型通过重复的潜在更新来逐步优化其预测。为什么中间答案能够与最终答案保持一致，而其分数却仍在持续变化？我们推导出一个精确的间隔刻画方法，将幅度界的保守性分解为三个部分：共同平移、相对于胜者的更新方向，以及每个竞争者的更新与其分数差距之间的配对关系。在Huginn-3.5B和Ouro-1.4B模型上的实验表明，在完整答案文本评分下，考虑更新方向和竞争者配对因素后，在去除共同平移的基础上，可将平均最早合格深度进一步降低总深度的22.5-34.4%。这一回顾性比较基于已完成的轨迹。在标签评分下也存在显著的贡献。对于共享的预测分布，我们正交地分离共同运动和对比运动，并通过候选集质量和集合内集中度来表达共同分量。共同与对比能量……（摘要在此处截断）

    arXiv:2609.21383v1 Announce Type: new  Abstract: Depth-recurrent language models refine predictions through repeated latent updates. Why can intermediate answers agree with the endpoint while their scores continue to change? We derive a sharp margin characterization that decomposes the conservatism of a magnitude bound into common translation, direction relative to the winner, and the pairing of each competitor's update with its score gap. Across Huginn-3.5B and Ouro-1.4B, accounting for update direction and competitor pairing reduces the mean earliest qualifying depth by a further 22.5-34.4% of the total depth beyond translation removal under full answer-text scoring. This retrospective comparison uses completed trajectories. Substantial contributions also occur under label scoring. For shared predictive distributions, we separate common and contrast motion orthogonally and express the common component through candidate-set mass and within-set concentration. Common and contrast energi
    
[^77]: 基于神经时间点过程的业务流程执行概率预测

    Probabilistic Forecasting of Business Process Executions with Neural Temporal Point Processes

    [https://arxiv.org/abs/2609.21382](https://arxiv.org/abs/2609.21382)

    该论文将业务流程执行预测重构为基于带标记时间点过程的生成式建模问题，通过Transformer编码器与混合解码器显式处理时间戳重复问题，从而在构造上提供可靠的预测分布。

    

    基于服务的系统的操作员依赖于对正在运行的执行将如何继续的预测，而只有当预测的可靠性已知时，这种预测才是可操作的。主流的深度学习模型在此任务是判别式和确定性的：它们只输出单个下一活动和单个剩余时间估计，没有可供推理的概率分布。我们转而将该问题构建为基于带标记时间点过程的生成式序列建模，它定义了下一标记及其事件间隔时间的联合密度，因此在构造上就能提供预测分布。真实的事件日志违反了这些模型所依赖的简单点过程假设，因为连续事件经常带有相同的时间戳；我们显式地处理这种时间戳并列情况，并将Transformer编码器与事件间隔时间的混合解码器相结合，通过精确对数似然进行训练。在十个公开日志上，所得到的模型与判别式基线相匹配。

    arXiv:2609.21382v1 Announce Type: new  Abstract: Operators of service-based systems act on forecasts of how a running execution will continue, and such a forecast is actionable only if its reliability is known. Mainstream deep-learning models for this task are discriminative and deterministic: they emit a single next activity and a single remaining-time estimate, without a distribution to reason over. We instead cast the problem as generative sequence modelling with marked temporal point processes, which define a joint density over the next mark and its inter-event time and therefore deliver predictive distributions by construction. Real event logs violate the simple-point-process assumption these models rest on, since consecutive events frequently carry identical timestamps; we handle such ties explicitly and combine a transformer encoder with a mixture decoder over inter-event times, trained by exact log-likelihood. On ten public logs, the resulting model matches discriminative basel
    
[^78]: 知识图谱增强的Chronos-2用于HEC-RAS代理预测

    Knowledge-Graph-Augmented Chronos-2 for HEC-RAS Surrogate Forecasting

    [https://arxiv.org/abs/2609.21381](https://arxiv.org/abs/2609.21381)

    KG-Chronos-2通过将冻结的时间序列基础模型Chronos-2与水力学工程知识（精确状态残差解码、图条件历史检索和输入对齐校正）相结合，显著提升了HEC-RAS水面高程代理预测精度，相比各类基线方法最高可将RMSE降低39.54%。

    

    我们研究了将时间序列基础模型与水力学工程知识相结合能否改善HEC-RAS水面高程（WSE）的代理预测。我们提出了KG-Chronos-2，该方法将冻结的Chronos-2预测器与精确状态残差解码、图条件历史检索以及输入对齐校正相结合。我们将该方法与持续性方法、残差LSTM、工程条件循环GeoFNO、水力学DCRNN风格模型以及冻结的Chronos-2进行了比较。任务特定的拟合使用2008年模拟数据。评估涵盖来自2011年和2002年模拟的64个固定24小时窗口，涉及共享几何结构上71个河段中的4,675个横断面。KG-Chronos-2在原生WSE单位下实现了事件平衡的均方根误差0.246970。与冻结的Chronos-2相比，该方法将RMSE降低了14.13%；与水力学DCRNN风格模型相比降低了29.38%；与循环GeoFNO相比降低了39.54%。95%分层自助法区间……（摘要在此处截断）

    arXiv:2609.21381v1 Announce Type: cross  Abstract: We investigate whether coupling a time-series foundation model to hydraulic project knowledge improves surrogate forecasting of HEC-RAS water-surface elevation (WSE). We present KG-Chronos-2, which combines a frozen Chronos-2 predictor with exact-state residual decoding, graph-conditioned historical retrieval, and input-aligned correction. We compare the method with persistence, a residual LSTM, project-conditioned recurrent GeoFNO, a hydraulic DCRNN-style model, and frozen Chronos-2. Task-specific fitting uses the 2008 simulation. Evaluation covers 64 fixed 24-hour windows from the 2011 and 2002 simulations at 4,675 cross sections in 71 reaches on a shared geometry. KG-Chronos-2 achieves event-balanced root-mean-square error 0.246970 in native WSE units. It reduces RMSE by 14.13% relative to frozen Chronos-2, 29.38% relative to the hydraulic DCRNN-style model, and 39.54% relative to recurrent GeoFNO. The 95% hierarchical-bootstrap int
    
[^79]: 藏于寻常之中：一种基于扩散模型的视觉-语言模型地理定位隐私泄露缓解方法

    Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models

    [https://arxiv.org/abs/2609.21363](https://arxiv.org/abs/2609.21363)

    该论文系统揭示了多模态大推理模型可通过视觉推理从照片精确推断用户地理位置的隐私威胁，指出拒绝式防护与像素空间扰动防御的不足，并提出了一种基于扩散模型的隐私泄露缓解方法。

    

    多模态大型推理模型（MLRMs）在复杂视觉理解方面展现出了卓越的能力。然而，这种强大的能力也带来了一种关键但尚未被充分研究的隐私威胁：攻击者可以利用MLRMs，通过对建筑风格、植被和光照条件等细微视觉线索进行结构化推理，从用户随手分享的照片中精确推断其地理位置。在这项工作中，我们对MLRM驱动的地理定位隐私泄露进行了系统性研究。我们首先揭示了基于拒绝回答的安全防护措施严重不足，因为精心构造的越狱提示词可以将模型的响应率提升至100%。我们进一步发现，现有的防御方法——即向共享图像中注入不可感知的扰动——存在像素空间优化固有的结构性局限，导致黑盒迁移性下降并产生明显的视觉伪影。受此启发，本文提出了一种基于扩散模型的隐私泄露缓解方法（摘要在此处截断）。

    arXiv:2609.21363v1 Announce Type: cross  Abstract: Multimodal large reasoning models (MLRMs) have demonstrated remarkable capabilities in complex visual understanding. However, this very power introduces a critical yet underexplored privacy threat: adversaries can exploit MLRMs to precisely infer users' geographic locations from casually shared photographs, by performing structured reasoning over subtle visual cues such as architectural styles, vegetation, and lighting conditions. In this work, we present a systematic study of MLRM-driven geolocation privacy leakage. We first reveal that refusal-based safeguards are critically insufficient, as carefully crafted jailbreak prompts can raise model response rates to 100%. We further identify that existing defenses, which inject imperceptible perturbations into shared images, suffer from structural limitations intrinsic to their pixel-space optimization, resulting in degraded black-box transferability and pronounced visual artifacts. Motiva
    
[^80]: IntBMoE：将块级条件化融入专家组合以实现全参与混合专家模型

    IntBMoE: Integrating Block-Level Conditioning into Expert Composition for Full-Participation Mixture-of-Experts

    [https://arxiv.org/abs/2609.21346](https://arxiv.org/abs/2609.21346)

    IntBMoE提出一种块条件化的混合专家模型，通过将密集专家组合与稀疏块执行相结合，把参与度、执行量和参数具体化三个量解耦，从而在实现全专家参与的同时控制计算与内存开销。

    

    混合专家模型可以扩展模型容量，但现有设计无法独立设置三个量。对于单个token而言，参与度是指有多少专家为其输出贡献知识，执行量是指实际计算了多少个专家（计算成本），具体化是指必须构建和存储多少个专家规模的参数集（内存成本）。稀疏路由保持执行量和具体化处于较低水平，但会缩减参与度：对每个token，只有少数专家做出贡献。密集输出混合恢复了完全参与，但其执行量随专家数量增长。参数合并将执行量保持在一个专家，但其具体化随路由决策数量增长。我们提出IntBMoE，一种块条件化的MoE，通过将密集专家组合与稀疏块执行相结合来解耦这三个量。其块来自一个小型可学习码本，每个条目对应一个块。在每个内部层，一个轻量级超网络……

    arXiv:2609.21346v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) scales capacity, but existing designs cannot set three quantities independently. For a single token, participation is how many experts contribute knowledge to its output, execution is how many are actually computed (compute cost), and materialization is how many expert-sized parameter sets must be built and stored (memory cost). Sparse routing keeps execution and materialization low, but shrinks participation: for each token, only a few experts contribute. Dense output-mixing restores full participation, but its execution grows with the number of experts. Parameter-merging keeps execution at one expert, but its materialization grows with the number of routing decisions. We propose IntBMoE, a block-conditioned MoE that decouples all three by pairing dense expert composition with sparse block execution. Its blocks come from a small learned codebook, one per entry. At each internal layer, a lightweight hypernetwork 
    
[^81]: 常规血液检查在区分儿童细菌性与病毒性感染方面优于CRP

    Routine Blood Tests Outperform CRP for Distinguishing Bacterial From Viral Infection in Children

    [https://arxiv.org/abs/2609.21332](https://arxiv.org/abs/2609.21332)

    常规血液检查（全血细胞计数）结合机器学习分类模型在区分儿童细菌性与病毒性感染方面优于CRP，有望减少不必要的抗生素使用并缓解抗菌素耐药性问题。

    

    急性传染病是全球儿童就诊和住院的主要原因之一。这些感染主要由病毒或细菌引起，然而区分两者仍然是一个常见的临床挑战。因此，儿科医生往往默认选择更保守的方式开具抗生素处方，这加剧了日益严重的抗菌素耐药性问题。本研究旨在评估全血细胞计数（CBC）对判断当前感染类型的额外预测价值。这项回顾性研究使用了2022年至2026年间906名年龄在2至14岁之间、经检测确诊为病毒性或细菌性感染的儿童患者数据。纳入标准还要求具备CBC结果和CRP水平测量数据。这些实验室参数以及年龄被用作多个监督分类模型的输入特征。模型性能使用（原文在此处截断）

    arXiv:2609.21332v1 Announce Type: new  Abstract: Acute infectious diseases are among the leading causes of medical consultations and hospitalizations in children worldwide. These infections are predominantly caused by viruses or bacteria, yet differentiating between the two remains a common clinical challenge. As a result, pediatricians often default to the safer option of prescribing antibiotics contributing to the growing problem of antimicrobial resistance. The objective is to assess the additional predictive value of CBC towards determining the current infection. This retrospective study used data from 906 pediatric patients aged between 2 and 14 years who were tested positive either for viral or bacterial infection between 2022 and 2026. Inclusion criteria further required availability of CBC results and CRP level measurements. These laboratory parameters as well as age were used as input features for several supervised classification models. Model performance was evaluated using 
    
[^82]: 基于缓冲分位数目标的深度强化学习

    Deep Reinforcement Learning with Buffered Quantile Objectives

    [https://arxiv.org/abs/2609.21327](https://arxiv.org/abs/2609.21327)

    提出了Deep-BQRL，一个无模型的分布式深度强化学习框架，利用缓冲分位数目标实现可解释的风险敏感决策，突破了以往基于模型方法仅适用于小型表格问题的局限。

    

    基于分位数的强化学习通过优化累积回报分布的指定分位数，为风险敏感决策提供了一种可解释的方法。尽管具有这一吸引力，在点分位数目标下进行学习仍具有挑战性：回报分布的微小扰动可能导致分位数发生突变，且精确的分位数敏感规划需要计算开销巨大的分布式优化。下缓冲分位数通过对目标水平紧下方的相邻分位数取平均来缓解前一个困难，在保持底层点分位数目标的同时提供了一个更平滑的替代目标。然而，基于这一原理的现有方法仍然基于模型，并依赖于显式的回报分布规划，这限制了其在小型表格问题之外的应用。我们提出了Deep-BQRL，这是一个无模型的分布式强化学习框架，将缓冲分位数方法扩展到深度学习场景。

    arXiv:2609.21327v1 Announce Type: cross  Abstract: Quantile-based reinforcement learning provides an interpretable approach to risk-sensitive decision-making by optimizing a prescribed quantile of the cumulative-return distribution. Despite this appeal, learning under a point quantile objective is challenging: quantiles can change abruptly under small perturbations of the return distribution, and exact quantile-sensitive planning requires computationally demanding distributional optimization. Lower-buffered quantiles alleviate the former difficulty by averaging neighboring quantiles immediately below the target level, providing a smoother surrogate while preserving the underlying point-quantile objective. Existing methods based on this principle, however, remain model-based and rely on explicit return-law planning, limiting their applicability beyond small tabular problems. We develop Deep-BQRL, a model-free distributional reinforcement-learning framework that extends buffered-quantile
    
[^83]: 面向自动大规模筛查的稀疏识别：一种具有超快解码算法的约束感知框架

    Sparse Identification for Automatic Large-Scale Screening: A Constraint-Aware Framework with Ultra Fast Decoding Algorithm

    [https://arxiv.org/abs/2609.21321](https://arxiv.org/abs/2609.21321)

    本文提出了逻辑筛查方法，这是一种超快速、准确且有理论保障的大规模筛查框架，仅用O(klogn)次合并检测即可识别所有阳性样本，且解码仅依赖简单的逻辑运算。

    

    在疫情早期阶段，通过大规模筛查识别少量感染者对于疫情控制至关重要，但在试剂和检测能力有限的情况下，这仍然是一个挑战。现有的分组检测方法要么计算复杂度高，要么识别准确率低。更糟糕的是，目前没有任何方法能够为稀疏识别提供严格的理论分析，同时兼顾实际应用中普遍存在的样本使用约束和稀释效应所带来的硬约束。在本文中，我们提出了逻辑筛查方法，这是一个超快速、准确且具有理论依据的大规模筛查框架。LoSc引入了一种新颖的解码算法，采用非常简单的选择策略，仅需O(klogn)次合并检测即可识别出所有阳性样本。该解码过程仅依赖逻辑运算，使硬件能够直接实现……

    arXiv:2609.21321v1 Announce Type: cross  Abstract: In the early stages of a pandemic, identification of a small number of infected individuals through large-scale screening is critical for pandemic control, yet remains challenging under limited reagents and testing capacity. Existing group testing methods suffer from either high computational complexity or low identification accuracy. Even worse, no available methods provide theoretically rigorous analysis for sparse identification with hard constraints caused by the sample usage constraint and the dilution effect existing ubiquitously in practical applications. In this article, we propose the Logic Screening method (LoSc), an ultra fast, accurate, and theoretically grounded framework for large-scale screening. LoSc introduces a novel decoding algorithm with a very simple selection strategy, achieving identification of all positives with only O(klogn) pooled tests. The decoding relies only on logical operations, enabling direct hardwar
    
[^84]: 用于个体化回归的对角化注意力机制：潜在行定位与预测

    Diagonalized Attention for Individualized Regression: Latent-Row Localization and Prediction

    [https://arxiv.org/abs/2609.21320](https://arxiv.org/abs/2609.21320)

    本文提出一种用于矩阵值协变量个体化稀疏回归的对角化注意力机制，通过查询-键分数定位每个样本特定的信号行并共享总体回归效应，其参数维度与样本量无关，且无需响应变量即可为新样本识别感兴趣的行。

    

    现代文本和图像表示通常是矩阵值形式的，其行对应于标记、图像块或其他局部特征向量。预测信息往往是稀疏的但具有样本特异性，这使得采用共同支撑集的经典稀疏回归方法难以适应这种异质性。本文针对矩阵值协变量形式化了一个个体化稀疏回归框架，其中每个观测样本都有其自身感兴趣的行，而相关的回归效应在总体中是共享的。为了估计该模型，我们引入了一种对角化注意力机制，该机制使用查询-键分数来定位样本特定的信号行，并使用值矩阵进行下游回归。所提出的方法具有与样本量无关的参数维度，并且能够在没有响应变量的情况下为新观测识别感兴趣的行。我们建立了存在性定理，表明在适当的分数分离和（摘要在此处截断）

    arXiv:2609.21320v1 Announce Type: cross  Abstract: Modern text and image representations are often matrix-valued, with rows corresponding to tokens, patches, or other local feature vectors. Predictive information is often sparse but sample-specific, making classical sparse regression methods with a common support poorly suited to this heterogeneity. This paper formalizes an individualized sparse regression framework for matrix-valued covariates in which each observation has its own rows of interest, while the associated regression effects are shared across the population. To estimate this model, we introduce a diagonalized attention mechanism that uses query--key scores to localize sample-specific signal rows and a value matrix for downstream regression. The proposed method has a parameter dimension independent of sample size and can identify rows of interest for new observations without their responses. We establish existence theorems showing that, under suitable score-separation and 
    
[^85]: 基于压缩的机器学习导论

    An Introduction to Compression-Based Machine Learning

    [https://arxiv.org/abs/2609.21309](https://arxiv.org/abs/2609.21309)

    本文系统调研并形式化了压缩与机器学习之间的双向转换关系，提出了一个经过实证验证的基于压缩的机器学习设计框架，其性能可与传统方法媲美，在恶意软件检测上优势尤为显著。

    

    任何无损压缩算法（如gzip）都可以通过归一化压缩距离或最小描述长度原理转换为机器学习方法。任何自回归模型也都可以通过熵编码转换为无损压缩方法。这种看似循环的依赖关系在现代人工智能和机器学习中蕴含着尚未实现的潜力。我们调研并形式化了利用压缩来实现机器学习的各种策略，并引入且实证验证了一个基于压缩的机器学习设计框架，发现基于压缩的方法与传统基线方法相比具有竞争力，并且在恶意软件检测任务上明显更强。我们发现，改变这些设计选择可以带来高达0.62的准确率提升。

    arXiv:2609.21309v1 Announce Type: new  Abstract: Any lossless compression algorithm (like gzip) may be converted into a machine learning method, via either Normalized Compression Distance or the Minimum Description Length principle. Any auto-regressive model may be converted into a lossless compression method via entropy coding. This seemingly circular dependence has unrealized potential in modern artificial intelligence and machine learning, and we survey and formalize the various strategies that have been used to leverage compression for machine learning. We introduce and empirically validate a design framework for compression-based ML, finding compression-based methods competitive with conventional baselines and decisively stronger on malware. We find that varying these design choices yields accuracy gains of up to 0.62.
    
[^86]: 快速且准确的文本内容文件类型识别

    Fast And Accurate Text Content File Type Identification

    [https://arxiv.org/abs/2609.21306](https://arxiv.org/abs/2609.21306)

    本文提出了一种用于识别文本内容文件（尤其是源代码）类型的神经网络模型，其平均准确率更高，速度比 Magika 快约四倍，且模型体积小 28%。

    

    各组织普遍需要一个能够根据文件内容识别文件类型的工具，尤其是在网络安全领域，因为在该领域中魔数和文件扩展名都不可信。虽然现有工具在实践中表现良好，但仍有很大的改进空间：对于像 Magika 这样基于模型的工具，可以在计算负载和检测时间方面改进；而对于使用编程语言结构的文件解析工具，则可以在检测准确性方面改进。在本研究中，我们提出了一种用于识别文本内容文件类型（尤其是源代码）的神经网络模型，它比其他现有工具更准确、更快速。我们在开源文件上的实验表明，该模型在文本内容文件类型识别方面不仅平均准确率更高，而且速度约为 Magika 的四倍，同时模型体积小 28%。

    arXiv:2609.21306v1 Announce Type: new  Abstract: A common requirement across organizations is to have a tool that can identify file types based on their contents, particularly in the cybersecurity domain where magic numbers and file extensions can not be trusted. While existing tools work well in practice, there is plenty of room for improvement either in terms of computational load and time for detection in the case of model based tools like Magika or in terms of accuracy of detection in the case of file parsing tools that use programming language constructs. In this study, we propose a neural network model for identification of types of text content files, especially source code, that is more accurate and faster than other available tools. Our experiments on open-source files indicate that it is not only more accurate on average for text-content file-type identification, but also approximately four times faster than Magika, while being 28% smaller in size.
    
[^87]: 利用机器学习识别安全平台产品滥用

    Identifying Security Platform Product Abuse with Machine Learning

    [https://arxiv.org/abs/2609.21303](https://arxiv.org/abs/2609.21303)

    该论文首次提出并实际部署了一套基于机器学习的全系统防御方案，用于检测安全平台产品滥用，使滥用检测覆盖率提升35%、每月警报减少30%，并能自适应攻击者行为的变化。

    

    产品滥用在SaaS行业中是一种单个罕见但日益增长的问题。高度复杂的威胁行为者可以在客户环境中滥用安全平台，或对产品本身进行绕过实验。威胁行为者可以利用“就地取材”（LOTL）攻击来避免使用繁琐且经常被检测到的恶意软件。应对这一威胁需要收集跨不同类型数据库的多种数据模态，解决此类复杂但危险事件内在稀缺性带来的冷启动问题，并在现实部署的约束条件下进行设计（例如成本、用户行为、性能等）。为此，我们提供了首个针对此类全系统防御的研究，特别是针对已部署且实际运行的能力。我们的结果显示，产品滥用覆盖率提高了35%，每月警报减少了30%，并且能够适应恶意行为者行为的变化。

    arXiv:2609.21303v1 Announce Type: cross  Abstract: Product abuse is an individually rare, but growing, problem across the SaaS industry. Highly sophisticated threat actors can misuse security platforms within customer environments or conduct bypass experiments on the product itself. Threat actors can leverage living-off-the-land (LOTL) attacks to avoid using cumbersome, frequently detected malware. Remediating this threat requires collecting multiple data modalities across different types of databases, addressing a cold-start problem in the intrinsic rarity of such sophisticated but dangerous events, and designing within the constraints of real-world deployment (e.g., cost, user behavior, performance, etc). To wit, we provide the first study of such a whole-system defense, especially with respect to a deployed and operational capability. Our results show an increase in product abuse coverage by 35\%, a 30\% reduction in monthly alerts, and adaptability to changes in malicious actors' b
    
[^88]: FairLMs：一个用于语言模型公平性的开箱即用库

    FairLMs: A Turnkey Library for Fairness in Language Models

    [https://arxiv.org/abs/2609.21296](https://arxiv.org/abs/2609.21296)

    FairLMs是一个Python库，通过对模型能力和输入要求的显式声明，将语言模型公平性研究中的偏见测量、缓解方法应用和证据审查统一起来，提供33个指标、14个缓解组件、14个诊断工具以及多种架构和API的适配器。

    

    语言模型的公平性研究涉及测量偏见、应用缓解方法，以及审查评估所依据的证据。现有工具通过不同的接口提供互补的功能，因此要组合使用它们，必须先协调模型接口、证据格式、访问限制和结果类型，才能检查适用性或比较各种方法。我们推出了FairLMs，这是一个Python库，它通过对模型能力和输入要求的显式声明将这些活动连接起来。该库提供33个内在和外在指标、涵盖四个干预类别的14个缓解组件、14个数据集和评分工具诊断、适配三种Transformer架构及所支持的托管补全API的适配器，以及基准测试加载器。声明会在执行前进行检查，结果会携带其获得时所用的配置，从而使得兼容的……

    arXiv:2609.21296v1 Announce Type: cross  Abstract: Fairness research on language models involves measuring bias, applying mitigation methods, and examining the evidence on which an evaluation rests. Existing tools offer complementary functionality through different interfaces, so combining them requires reconciling model interfaces, evidence formats, access constraints, and result types before applicability can be checked or methods compared. We introduce \textbf{FairLMs}, a Python library that connects these activities through explicit declarations of model capabilities and input requirements. It provides 33 intrinsic and extrinsic metrics, 14 mitigation components spanning four intervention categories, 14 dataset and scoring-instrument diagnostics, adapters for the three Transformer architectures and supported hosted completion APIs, and benchmark loaders. Declarations are checked before execution and results carry the configuration under which they were obtained, so that compatible 
    
[^89]: 多受试者预训练实现闭语料库表面肌电语音解码的短校准个性化

    Multi-Subject Pretraining Enables Short-Calibration Personalization for Closed-Corpus Surface EMG Speech Decoding

    [https://arxiv.org/abs/2609.21288](https://arxiv.org/abs/2609.21288)

    该论文提出“检查点初始化+多受试者预训练+目标用户短校准微调”的流程，使表面肌电语音解码在每名用户不足半小时校准数据的情况下将字符错误率降至21.7%，显著优于无校准和直接微调方案。

    

    基于表面肌电图的无声语音接口一直受限于跨用户差异性大和校准负担重的问题。我们研究了一个有限数据场景：27名言语功能正常的参与者分别在出声朗读和默语（唇动）两种模式下各贡献了不到0.5小时的数据（平均21.3分钟）。在一个封闭的50句语料库中，我们采用留一受试者交叉评估，从已发布的单受试者检查点初始化，先在非留出参与者上进行预训练，再对目标参与者进行微调。该流程实现了21.7%的字符错误率（CER）和31.9%的词错误率（WER），相比之下，不进行目标受试者校准时CER为49.3%，直接进行检查点微调时CER为68.0%。而从随机初始化开始的多受试者预训练加微调仅达到44.9%的CER，且在27折中的5折未能在固定训练计划下收敛，表明基于检查点的初始化带来了显著的优化与精度收益。

    arXiv:2609.21288v1 Announce Type: new  Abstract: Surface electromyography (sEMG)-based silent speech interfaces are limited by cross-user variability and calibration burden. We study a limited-data setting in which each of 27 speech-typical participants contributed less than 0.5 h of data (21.3 min on average) across Aloud and Mimed speech. Within a closed 50-sentence corpus, we used leave-one-subject-out evaluation, initializing from a released single-subject checkpoint, pretraining on non-held-out participants, and fine-tuning on the target participant. This pipeline achieved 21.7% character error rate (CER) and 31.9% word error rate (WER), compared with 49.3% CER without target-subject calibration and 68.0% CER for direct checkpoint fine-tuning. Multi-subject pretraining from random initialization followed by fine-tuning reached 44.9% CER and did not converge under the fixed schedule in 5 of 27 folds, indicating substantial optimization and accuracy benefits from checkpoint initiali
    
[^90]: 超大规模个性化搜索的GPU-CPU混合检索

    Hybrid GPU-CPU Retrieval for Personalized Search at Ultra-Large Scale

    [https://arxiv.org/abs/2609.21281](https://arxiv.org/abs/2609.21281)

    该论文提出一种混合GPU-CPU协同服务系统，通过高深度GPU通路与高广度CPU通路的编排设计，解决了超大规模个性化搜索中个性化深度与库存覆盖广度难以兼得的“个性化-规模悖论”。

    

    在万亿级文档规模的用户生成内容上进行基于嵌入的检索，暴露了两种生产需求之间的尖锐冲突：对具有丰富用户意图的查询实现深度且富有表现力的个性化，以及在固定的延迟和资源预算下对海量库存的广泛覆盖。我们将这一矛盾特征化为“个性化-规模悖论”：在GPU内存中托管完整的推理库存资源消耗过高，而CPU计算无法在延迟关键路径上执行同样重交互的模型。我们提出了一种混合GPU-CPU协同服务系统，通过编排（而非新的模型类别）来解决这一悖论。高深度的GPU通路在约十亿文档规模的精选在线池上融合检索与交互预排序，而高广度的CPU通路则以轻量级个性化评分搜索一个规模约大二十倍的独立选择的在线库存。二者之一或两者……

    arXiv:2609.21281v1 Announce Type: cross  Abstract: Embedding-based retrieval on user-generated content at the trillion-document scale exposes a sharp conflict between two production demands: deep, expressive personalization for queries with rich user intent, and broad coverage of a massive inventory under fixed latency and resource budgets. We characterize this as the personalization-scale paradox: hosting the full serving inventory in GPU memory is too resource intensive, while CPU compute cannot execute the same interaction-heavy model on the latency-critical path.   We present a hybrid GPU-CPU co-serving system that resolves the paradox through orchestration rather than a new model class. A high-depth GPU pathway fuses retrieval and interaction pre-ranking over a curated online pool on the order of a billion documents, while a high-breadth CPU pathway searches an independently selected online inventory roughly twenty times larger with lightweight personalized scoring. Either or both
    
[^91]: MIRCID：推断的中心miRNA驱动药物机制建模中的跨任务性能提升

    MIRCID: Inferred Hub-miRNAs Drive Cross-Task Improvements in Drug Mechanistic Modeling

    [https://arxiv.org/abs/2609.21280](https://arxiv.org/abs/2609.21280)

    MIRCID框架通过HubmiRNet从L1000地标基因推断泛癌症中心miRNA，并证明miRNA特征增强比转录因子活性能在通路分类和药物作用机制检索任务中带来更一致的跨任务性能提升。

    

    药物作用机制建模通常依赖于扰动转录组数据，但匹配的微RNA（miRNA）测量数据往往不可用。推断的调控特征为复用这些数据提供了一种可扩展的方式。在此，我们提出MIRCID，一个在通路分类和基于相似性的MoA检索任务中，将基因表达与推断的转录因子（TF）活性及miRNA表达进行对比评估的框架。HubmiRNet从977个L1000地标基因中推断出414个泛癌症中心miRNA（HubmiRs），达到了87.72%的皮尔逊相关系数；其1,298输出的变体在全miRNA预测任务上也优于SiCmiR（71.21% 对比 67.30%）。在所评估的比较中，miRNA增强比TF活性带来了更一致的性能提升。通用嵌入对照显示出依赖于模型的效用，而互补性分析识别出一种独特的、部分可线性恢复的表示，该表示保留了源自基因的...

    arXiv:2609.21280v1 Announce Type: new  Abstract: Drug mechanism-of-action (MoA) modeling commonly relies on perturbational transcriptomes, but matched microRNA (miRNA) measurements are often unavailable. Inferred regulatory features offer a scalable way to reuse these data. Here, we present MIRCID, a framework comparing gene expression with inferred transcription factor (TF) activity and miRNA expression across pathway classification and similarity-based MoA retrieval. HubmiRNet infers 414 pan-cancer hub miRNAs (HubmiRs) from 977 L1000 landmark genes, achieving a Pearson correlation coefficient of 87.72\%; its 1,298-output variant also outperformed SiCmiR on the full-miRNA task (71.21\% versus 67.30\%). In the evaluated comparisons, miRNA augmentation provided more consistent gains than TF activity. Generic embedding controls showed model-dependent utility, while complementarity analyses identified a distinct, partially linearly recoverable representation that retained gene-derived str
    
[^92]: 一个语言模型评审团相当于多少个人类评判者？

    How Many Humans Is a Judge Panel Worth?

    [https://arxiv.org/abs/2609.21277](https://arxiv.org/abs/2609.21277)

    该论文提出两种不同的“等效人类评判者数量”度量——谱残差多样性 ν_H 与分布平方误差 ν_MSE，发现同一组 32 个语言模型评审在三个 ChaosNLI 任务上分别相当于约 4.24–6.50 个和 2.30–3.75 个人类评判者，且更大的谱多样性并不保证更好的分布恢复。

    

    一组语言模型评审团究竟代表多少个人类判断？答案取决于匹配的对象是什么。我们针对经验人类标签分布对类别型评审团进行审计，保留了那些相对于单一金标准标签的二值错误所坍缩掉的分歧。我们通过将归一化残差格拉姆矩阵的参与率与条件独立的人类参考抽样相匹配来度量谱残差多样性，得到 ν_H；并单独匹配分布平方误差，得到 ν_MSE。在三个 ChaosNLI 任务上，同一组由 32 个评审组成的评审团，其 ν_H 为 4.24–6.50，而 ν_MSE 仅为 2.30–3.75。一个谱恒等式将决定误差的特征值、成员能量和平均方向权重分离开来。可实现的硬标签评审团表明，即使成员能量相等且相关性非负，更大的谱多样性也可能伴随更差的分布恢复。在观察到的评审团中，同规模内的排名一致性……（摘要原文在此处截断）

    arXiv:2609.21277v1 Announce Type: new  Abstract: How many human judgments does a panel of language models represent? The answer depends on what is matched. We audit categorical judge panels against empirical human label distributions, retaining disagreement that binary errors relative to one gold label collapse. We measure spectral residual diversity by matching the participation ratio of a normalized residual Gram matrix to conditionally independent human-reference draws, giving nu_H. We separately match distributional squared error, giving nu_MSE. Across three ChaosNLI tasks, the same 32-judge panels have nu_H=4.24--6.50 but nu_MSE=2.30--3.75. A spectral identity separates the eigenvalues, member energies, and averaging-direction weights that determine error. Realizable hard-label panels show that greater spectral diversity can accompany worse distribution recovery even with equal member energies and nonnegative correlations. In the observed panels, within-size ranking agreement vari
    
[^93]: 使用开源编译器工具对AMD XDNA NPU进行编程：FlashAttention案例研究

    Programming AMD XDNA NPUs with Open-source Compiler Tools: A FlashAttention Case Study

    [https://arxiv.org/abs/2609.21264](https://arxiv.org/abs/2609.21264)

    本文通过FlashAttention案例研究，展示了如何利用开源编译器工具（IRON和MLIR-AIR）对AMD XDNA NPU进行编程，其中将三个注意力阶段融合为单一内核的设计在端到端执行中达到3.62 TFLOP/s，能效比同芯片集成GPU高5.3至7.2倍。

    

    空间架构NPU（如AMD XDNA）将计算单元置于小型本地存储器旁边，并将它们之间的数据移动交由软件处理。将多阶段工作负载映射到此类设备上，很大程度上是决定中间张量存放在哪里的问题。我们报告了使用开源IRON和MLIR-AIR流程为FlashAttention做出这些选择时所学到的经验。我们在XDNA 1和XDNA 2上比较了四种参考设计：一种单独运行每个算子，两种在芯片上算子之间进行流式传输，还有一种将所有三个注意力阶段融合到单个内核中。融合内核将QK^T分数保存在计算单元本地存储器中，并通过级联互连归约部分结果，因此分数永远不会返回到共享的MemTile存储器。在XDNA 2上，该设计在完整的端到端执行中达到3.62 TFLOP/s，是IRON设计的两倍，在2K及以上token时，能效是同一芯片上集成GPU的5.3至7.2倍。

    arXiv:2609.21264v1 Announce Type: cross  Abstract: Spatial NPUs such as AMD XDNA place compute tiles beside small local memories and leave data movement between them to software. Mapping a multi-stage workload onto such a device is largely a question of where the intermediate tensors live. We report what we learned making those choices for FlashAttention with the open-source IRON and MLIR-AIR flows.   We compare four reference designs on XDNA 1 and XDNA 2: one runs each operator separately, two stream between operators on chip, and one fuses all three attention stages into a single kernel. The fused kernel holds the $\boldsymbol{QK}^{\mathsf T}$ scores in compute-tile local memory and reduces partial results over the cascade interconnect, so the scores never return to shared MemTile memory. On XDNA 2, it reaches 3.62 TFLOP/s over complete end-to-end execution, twice the IRON design, with 5.3 to 7.2 times the energy efficiency of the integrated GPU on the same chip at 2K tokens and abov
    
[^94]: 验证而非信任：面向大规模视频发现检索的智能体化模型开发

    Verify, Don't Trust: Agentic Model Development for Video Discovery Retrieval at Scale

    [https://arxiv.org/abs/2609.21257](https://arxiv.org/abs/2609.21257)

    提出了 EvoPilot——一种带人工把关的长周期在线自动研究方法，通过角色化智能体、持久化实验记录和确定性检查来保障结论的有效性，并在支撑视频发现产品 Video Deep Dive 的检索系统上完成了为期 37 天的大规模验证。

    

    大型语言模型（LLM）智能体能够提出、实现并评估模型变更。自动研究循环通过在自包含程序上进行分钟级的迭代展示了这种能力。而在线自动研究则涉及异步系统、长达数小时的变体，以及可能影响产品的长达数周的实验活动。当代码变更实际未生效、数据窗口发生泄漏、评估器语义发生漂移，或两个实验分支经过不同的服务链路时，一次已完成的运行仍可能支持一个无效的结论。我们提出了 EvoPilot，一种用于长周期在线自动研究的人工把关方法。角色特定的智能体通过版本化的领域技能和类型化适配器执行每一轮任务。持久化记录保存实验与失败信息；确定性检查强制执行已记录的经验教训。我们针对为 Video Deep Dive（VDD）提供支持的检索系统开展了一项为期 37 天的实验活动，VDD 是一种在用户打开搜索后用于发现后续视频的在线体验。

    arXiv:2609.21257v1 Announce Type: cross  Abstract: Large language model (LLM) agents can propose, implement, and evaluate model changes. Autoresearch loops demonstrate this capability through minutes-scale iterations on a self-contained program. Online autoresearch instead spans asynchronous systems, hours-long variants, and weeks-long campaigns that can influence a product. A completed run can still support an invalid conclusion when a code change is a no-op, data windows leak, evaluator semantics drift, or the two arms traverse different serving funnels. We present EvoPilot, a human-gated method for long-horizon online autoresearch. Role-specific agents execute each round through a versioned domain skill and typed adapter. Durable records preserve experiments and failures; deterministic checks enforce recorded lessons.   We study a 37-day campaign for the retrieval system that powers Video Deep Dive (VDD), an online experience for discovering follow-on videos after a user opens a see
    
[^95]: 从可训练性诊断到优化论断：变分量子优化中的边界与控制

    From Trainability Diagnostics to Optimization Claims: Boundaries and Controls in Variational Quantum Optimization

    [https://arxiv.org/abs/2609.21243](https://arxiv.org/abs/2609.21243)

    该论文揭示了变分量子优化中“梯度可训练性”与“优化成功”之间的鸿沟，提出步级诊断并严格证明在固定量子态与更新范数下原始梯度即已最大化一阶下降，从而划定了各类梯度干预策略的能力边界。

    

    贫瘠高原诊断刻画的是梯度信号在训练中是否仍然可得，但残余的梯度信号并不必然转化为成功的优化。我们在优化器步级的层面上研究这一“可训练性—优化”鸿沟。将系数加权的哈密顿量各项梯度视为类任务分量，我们引入了步级诊断方法，并推导出符号化的逐项组织、方向性活动与一阶下降之间的精确桥梁。将该桥梁化解到标准的一阶几何框架中表明，表观上的“组织—活动”因子并非独立的优化轴，而且在固定量子态和更新范数的条件下，原始梯度即可使总目标函数的一阶下降量最大化。我们在横场伊辛模型实例上，结合硬件高效拟设与哈密顿变分拟设，比较了普通梯度下降、确定性的哈密顿量逐项PCGrad变体以及探测门控的LSO-PCGrad方法。

    arXiv:2609.21243v1 Announce Type: cross  Abstract: Barren plateau diagnostics characterize whether gradient signal remains available for training, but surviving signal need not translate into successful optimization. We study this trainability--optimization gap at the level of optimizer steps. Treating coefficient-weighted Hamiltonian-term gradients as task-like components, we introduce step-level diagnostics and derive an exact bridge between signed termwise organization, directional activity, and first-order descent. Resolving this bridge into standard first-order geometry shows that the apparent organization--activity factors are not independent optimization axes and that, at fixed state and update norm, the raw gradient maximizes first-order descent of the summed objective. We compare vanilla gradient descent, a deterministic Hamiltonian-term PCGrad variant, and probe-gated LSO-PCGrad on transverse-field Ising model instances with hardware-efficient and Hamiltonian variational ansa
    
[^96]: 使用上下文感知集中式复制粘贴数据增强的野火图像多类语义分割

    Multiclass Semantic Segmentation of Wildland Fire Images Using Context-Aware Centralized Copy-Paste Data Augmentation

    [https://arxiv.org/abs/2609.21241](https://arxiv.org/abs/2609.21241)

    本文提出一种上下文感知的集中式复制粘贴数据增强策略，通过将火焰仅粘贴到语义合理的区域，提升了小型多类野火数据集中增强样本的真实性与数据质量。

    

    为基于深度学习的图像分割生成精确标注既昂贵又耗费人力。这一挑战在野火应用中尤为突出，由于动态火灾场景难以采集和标注，准确标注的数据集十分稀缺。为了解决这一问题，我们之前的工作提出了用于野火图像语义分割的集中式复制粘贴数据增强（CCPDA）方法，该方法通过将源图像中的火簇随机粘贴到目标图像上来生成人工训练样本。然而，随机放置可能产生上下文不真实的场景，例如火焰在沥青上燃烧。在本文中，我们提出了一种专门为提升小型多类野火数据集的数据质量和真实性而设计的上下文感知策略，确保增强后的样本在上下文上依然合理。所提出的方法将火焰的放置限制在……（原文摘要在此处截断）

    arXiv:2609.21241v1 Announce Type: cross  Abstract: Producing accurate annotations for deep learning based image segmentation is both costly and labor intensive. This challenge is especially evident in wildland fire applications, where accurately labeled datasets are scarce due to the difficulty of collecting and annotating dynamic fire scenes. To address this problem, our previous work introduced the Centralized Copy-Paste Data Augmentation (CCPDA) method for semantic segmentation of wildland fire imagery, which generates artificial training samples by randomly pasting fire clusters from source images onto target images. However, random placement can produce contextually unrealistic scenes, such as fire burning on asphalt. In this paper, we present a context-aware strategy designed specifically to improve data quality and realism in small multiclass wildland fire datasets, ensuring that augmented samples remain contextually meaningful. The proposed method restricts fire placement to se
    
[^97]: 基于位姿注意力的视觉导航Transformer

    Visual Navigation Transformer with Pose Attention

    [https://arxiv.org/abs/2609.21212](https://arxiv.org/abs/2609.21212)

    提出VNT-PA，一种以相机位姿作为位置编码的视觉导航Transformer规划器，使注意力基于关键帧间的位姿差异而非时间顺序，从而无需构建显式地图即可重用历史遍历经验，在HM3D点目标导航中达到93.3%成功率。

    

    学习得到的导航策略通常将观测作为按时间排序的历史进行处理，位置编码将每个观测与其被观测到的时间绑定，这使得难以重用先前遍历环境时的经验。而能够重用此类经验的系统通常会构建显式表示（如地图或拓扑图）并在其上进行规划。我们提出了VNT-PA（基于位姿注意力的视觉导航Transformer），这是一种transformer规划器，其上下文是由相机位姿索引的一组深度关键帧。以相机位姿作为位置编码，注意力取决于关键帧之间的位姿差异而非其时间顺序。VNT-PA被训练用于模仿在真实场景网格上运行的最短路径规划器，仅通过当前位姿和目标位置查询空间上下文来预测动作。在HM3D验证场景的点目标导航任务中，VNT-PA达到了93.3%

    arXiv:2609.21212v1 Announce Type: cross  Abstract: Learned navigation policies typically consume observations as a temporally ordered history, with positional encodings tying each observation to when it was seen, making it difficult to reuse experience from earlier traversals of an environment. Systems that do reuse such experience usually construct an explicit representation, such as a map or a topological graph, and plan on it. We propose VNT-PA (Visual Navigation Transformer with Pose Attention), a transformer planner whose context is a set of depth keyframes indexed by camera pose. With camera poses as positional encoding, attention depends on the pose differences between keyframes rather than on their temporal order. VNT-PA is trained to imitate a shortest-path planner operating on the ground-truth scene mesh, predicting actions by querying the spatial context with only its current pose and the goal position. On point-goal navigation in HM3D validation scenes, VNT-PA reaches 93.3%
    
[^98]: 以可靠性为中心的稀疏纵向CT病灶大小预测评估：共形区间校准与Gompertz启发式正则化

    Reliability-Centered Evaluation of Sparse Longitudinal CT Lesion-Size Forecasting with Conformal Interval Calibration and Gompertz-Inspired Regularization

    [https://arxiv.org/abs/2609.21197](https://arxiv.org/abs/2609.21197)

    本研究基于DeepLesion和DLT构建了包含129名患者205条轨迹的稀疏纵向CT病灶基准数据集，采用共形区间校准与Gompertz启发式正则化对病灶大小预测进行了以可靠性为中心的系统评估，发现各方法点预测精度相近但不确定性可靠性表现差异显著。

    

    稀疏的纵向CT随访在仅有少量既往观测数据时限制了病灶大小的预测。我们基于DeepLesion和Deep Lesion Tracker（DLT）构建了一个五次访视的由DLT衍生的同一病灶轨迹基准，共获得来自129名患者的205条轨迹。我们比较了探索性的传统“稀疏到最终”分析与一种主要的固定访视索引预测水平设计——后者在逐步加入更早期观测的同时预测T3到T4之间的常见对数变化——并评估了预测精度、不确定性可靠性、事后共形区间校准、亚组性能以及Gompertz启发的轨迹正则化。所评估的方法在点预测精度上部分重叠，但表现出明显不同的不确定性特征。在十个训练随机种子下的平均保留集RMSE在m=1、2、3、4时分别为0.4726、0.4305、0.4499和0.4513，表明m=2时平均RMSE最低；更多的历史观测数据并未带来改善……（原文摘要在此处被截断）

    arXiv:2609.21197v1 Announce Type: new  Abstract: Sparse longitudinal CT follow-up limits lesion-size forecasting when only a few prior observations are available. We constructed a five-visit DLT-derived same-lesion trajectory benchmark from DeepLesion and Deep Lesion Tracker (DLT), yielding 205 trajectories from 129 patients. We compared an exploratory conventional sparse-to-final analysis with a primary fixed visit-index horizon design predicting the common log change from T3 to T4 while progressively adding earlier observations, evaluating predictive accuracy, uncertainty reliability, post-hoc conformal interval calibration, subgroup performance, and Gompertz-inspired trajectory regularization. The evaluated methods showed partially overlapping point-prediction accuracy but distinct uncertainty behavior. Mean held-out RMSE across ten training seeds was 0.4726, 0.4305, 0.4499, and 0.4513 for m = 1, 2, 3, 4, indicating the lowest mean RMSE at m = 2; additional history did not improve R
    
[^99]: SWE-Proof：语言模型能否通过机器校验的证明解决真实世界的问题？

    SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?

    [https://arxiv.org/abs/2609.21190](https://arxiv.org/abs/2609.21190)

    该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。

    

    确保大语言模型（LLM）生成代码的正确性是现代软件工程的核心挑战。面向智能体代码生成的基准测试通常使用留出的测试套件来检验正确性，但测试套件本质上是不完整的，且日益容易受到模型记忆（数据泄露）的影响。形式化验证可以同时避免这两个问题，但现有工作仅覆盖规范以输入形式给出的独立任务，而非真实问题——真实问题涉及大型代码仓库，并以模糊的自然语言表达意图。我们提出了Benchproofer，一个能将带有已知正确补丁的编码任务转化为形式化验证任务的流水线：它为新代码编写规范，用公理概括新代码所调用的已有函数，并且只有在机械验证与对抗性检查两道关卡均通过后才接受一个实例。将该流水线应用于SWE-bench Verified，我们得到了SWE-Proof——包含500个真实问题的基准，其正确性通过形式化验证而非测试来保证，并且该方法还可扩展至SWE-bench Pro。……

    arXiv:2609.21190v1 Announce Type: cross  Abstract: Ensuring the correctness of LLM-generated code is a core challenge for modern software engineering. Benchmarks for agentic code generation check correctness with held-out test suites, which are inherently incomplete and increasingly susceptible to memorization. Formal verification avoids both problems, but existing work covers only standalone tasks whose specifications are given as input, not real issues, which touch large repositories and state intent in vague natural language. We present Benchproofer, a pipeline that turns a coding task with a known correct patch into a formally verified one: it writes a specification for the new code, summarizes the existing functions that code calls with axioms, and admits an instance only after mechanical and adversarial gates agree. Applying it to SWE-bench Verified yields SWE-Proof, 500 real issues whose correctness is formally verified rather than tested, and it extends to SWE-bench Pro. Across
    
[^100]: 在ARC类任务中基于测试时任务嵌入的隐式规则归纳

    Implicit Rule Induction with Test-Time Task Embeddings in ARC-like Tasks

    [https://arxiv.org/abs/2609.21181](https://arxiv.org/abs/2609.21181)

    该论文提出了一种新颖的两步测试时训练协议（Embed-TTT），先仅微调任务嵌入再微调主干网络，使任务嵌入与底层任务规则更好地对齐，从而在ARC类基准上提升性能并实现对已知规则的准确线性探测。

    

    抽象与推理语料库（Abstraction and Reasoning Corpus）及相关基准测试用于评估AI模型能否解决新颖的推理任务，但往往无法明确成功究竟是源于对预期底层规则的推断，还是依赖于捷径。我们通过研究Vision ARC（VARC）中的测试时任务嵌入来填补这一空白，VARC是一个在预训练主干网络基础上补充了可训练嵌入的模型，该嵌入用于表示变换规则。在原始VARC中，测试时训练（TTT）被联合应用于主干网络和任务嵌入。本文我们提出了一种新颖的两步TTT协议：首先仅微调任务嵌入（Embed-TTT），然后将其冻结并微调主干网络。在ARC-AGI-1、ConceptARC以及两个具有已知规则的控制数据集上，Embed-TTT持续产生更优的任务嵌入，这些嵌入与底层任务规则更好地对齐，改进了基于嵌入的检索，并能够对已知规则进行准确的线性探测。

    arXiv:2609.21181v1 Announce Type: new  Abstract: The Abstraction and Reasoning Corpus and related benchmarks evaluate whether AI models can solve novel reasoning tasks, but often leave unclear whether success reflects inference of the intended underlying rule or reliance on shortcuts. We address this gap by studying test-time task embeddings in Vision ARC (VARC), a model in which a pre-trained backbone is complemented by a trainable embedding representing the transformation rule. In the original VARC, test-time training (TTT) is jointly applied to the backbone and task embedding. Here we introduce a novel two-step TTT protocol: first finetune only the task embedding (Embed-TTT), then freeze it and finetune the backbone. Across ARC-AGI-1, ConceptARC, and two controlled datasets with known rules, Embed-TTT consistently yields improved task embeddings, ones that align better with underlying task rules, improve embedding-based retrieval, and enable accurate linear probing of known rules. Q
    
[^101]: TierKV：通过预测性多层KV缓存实现设备端长上下文大语言模型

    TierKV: Long-Context On-Device LLMs via Predictive Multi-Tier KV Caching

    [https://arxiv.org/abs/2609.21172](https://arxiv.org/abs/2609.21172)

    TierKV提出预测性多层缓存优化方法（PMCO），在解码前根据预填充隐藏状态预测缓存需求，将KV缓存令牌智能分配到精确、低秩和闪存卸载三个层级，在设备内存与精度预算约束下实现移动端长上下文LLM推理。

    

    大语言模型（LLM）正逐渐迁移到移动设备上，以处理日益多样化的文本、图像、视频和音频工作负载。这些应用通常需要长上下文，使得键值（KV）缓存成为主要的内存瓶颈，因为KV缓存随序列长度线性增长，并且在每个解码步骤中都需要被访问。先前的工作通过低秩压缩、令牌驱逐或闪存卸载来减少KV缓存的占用，但由此产生的重建开销、不可逆的令牌丢失或I/O停顿可能会抵消节省内存带来的收益。我们提出了TierKV，一个基于预测性多层缓存优化（PMCO）构建的移动端LLM推理框架。在解码开始之前，PMCO根据预填充阶段的隐藏状态预测未来的缓存需求，并在设备内存和精度预算的约束下，将令牌联合分配到精确、低秩和闪存卸载三个层级。这一设计保留了对完整上下文的访问，消除了循环依赖……

    arXiv:2609.21172v1 Announce Type: new  Abstract: Large language models (LLMs) are moving onto mobile devices for increasingly diverse workloads over text, images, video, and audio. These applications often require long contexts, making the Key-Value (KV) cache a dominant memory bottleneck because it grows linearly with sequence length and is accessed at every decoding step. Prior work reduces KV-cache footprint through low-rank compression, token eviction, or flash offloading, but the resulting reconstruction overhead, irreversible token loss, or I/O stalls can offset the benefit of saving memory. We present TierKV, a mobile LLM inference framework built on Predictive Multi-Tier Cache Optimization (PMCO). Before decoding starts, PMCO predicts future cache demand from prefill hidden states and jointly assigns tokens to exact, low-rank, and flash-offloaded tiers under the device memory and accuracy budgets. This formulation retains access to the full context, removes the circular depende
    
[^102]: M2G-LLM：通过多模态图推理与LLM上下文注入增强临床预测

    M2G-LLM: Enhancing Clinical Prediction via Multimodal Graph Reasoning and LLM Context Injection

    [https://arxiv.org/abs/2609.21164](https://arxiv.org/abs/2609.21164)

    该论文提出M2G-LLM框架，利用图神经网络建模患者就诊的时序关系并在相似患者间传播信息，将构建的多模态上下文向量注入大语言模型中间层，实现文本与非文本医疗数据的联合推理，从而提升临床预测性能。

    

    整合多样化的数据模态——如临床笔记、实验室检查结果和医学影像——对于推进临床决策至关重要。尽管大语言模型（LLMs）在处理非结构化临床文本方面表现出色，但其融合非文本模态的能力有限，阻碍了其在医疗应用中的更广泛使用。本文提出了M2G-LLM（Multimodal MedGraph-LLM），一种通过图神经网络（GNNs）增强大语言模型多模态整合与对齐能力的新型框架。我们的方法建模患者就诊之间的时间关系，在临床相似的患者之间传播信息，并对齐异构数据源以构建丰富的多模态上下文向量。这些向量被注入到大语言模型的中间层，使其能够对文本和非文本模态进行联合推理。我们在MIMIC-IV和MIMIC-CX数据集上对M2G-LLM进行了评估（摘要在此处截断）。

    arXiv:2609.21164v1 Announce Type: new  Abstract: Integrating diverse data modalities --- such as clinical notes, laboratory results, and medical imaging --- is essential for advancing clinical decision-making. While Large Language Models (LLMs) have shown remarkable performance in processing unstructured clinical text, their limited capacity to incorporate non-text modalities hinders their broader utility in healthcare applications. Here, we introduce M2G-LLM (Multimodal MedGraph-LLM), a novel framework that enhances LLMs with multimodal integration and alignment via Graph Neural Networks (GNNs). Our approach models temporal relationships between patient visits, propagates information across clinically similar patients, and aligns heterogeneous data sources to construct enriched multimodal context vectors. These vectors are injected into the intermediate layers of the LLM, enabling joint reasoning over textual and non-textual modalities. We evaluate M2G-LLM on the MIMIC-IV and MIMIC-CX
    
[^103]: HMB-GAN：用于矢量形状合成的混合多重贝塞尔生成对抗网络

    HMB-GAN: Hybrid Multi-B\'ezier GAN for Vector Shape Synthesis

    [https://arxiv.org/abs/2609.21158](https://arxiv.org/abs/2609.21158)

    该论文提出了端到端可微分的混合量子-经典生成对抗网络HMB-GAN，通过多段贝塞尔曲线拼接生成几何连续的闭合矢量形状，并发现量子生成器虽收敛更快、参数更少且点云指标略优，但受模拟器开销与硬件限制的制约。

    

    我们探索了利用混合量子-经典生成对抗网络来合成可直接用于CAD的矢量几何。与先前在栅格化域或单段贝塞尔曲线域上工作的研究不同，我们提出了HMB-GAN（混合多重贝塞尔生成对抗网络），这是一个端到端可微分的生成框架，通过拼接多段贝塞尔曲线表示来构建闭合形状，并在构造过程中强制保证几何连续性。我们在该架构中比较了量子增强生成器与经典生成器，并通过点云分布指标和几何形状统计量对二者进行评估。结果表明，尽管量子生成器收敛更快、模型参数量更少且在点云指标上表现略有提升，但其模拟器开销过大，因而基于经典模拟的评估受到硬件条件的制约。这些结果证明了该方法的可行性。

    arXiv:2609.21158v1 Announce Type: new  Abstract: We explore the use of hybrid quantum-classical generative adversarial networks for synthesising CAD-ready vector geometries. Unlike prior work that operates in rasterised or single-B\'ezier domains, we introduce HMB-GAN (Hybrid Multi-B\'ezier GAN), an end-to-end differentiable generative framework that constructs closed shapes through stitched multi-segment B\'ezier representations with geometric continuity enforced by construction. We compare a quantum-enhanced generator with a classical generator within this architecture and evaluate them across point cloud distribution metrics and geometric shape statistics. Results show that despite faster convergence, a reduction in model parameter count, and slightly improved performance on point cloud metrics, the quantum generator suffers from excessive simulator overhead and thus classically-simulated evaluation suffers from hardware constraints. These results demonstrate the feasibility of mode
    
[^104]: EnSol：一种用于分子溶解度预测的环境感知图神经网络

    EnSol: an environment-aware graph neural network for molecular solubility prediction

    [https://arxiv.org/abs/2609.21151](https://arxiv.org/abs/2609.21151)

    EnSol是一种环境感知的概率图神经网络框架，通过交叉注意力机制建模溶质-溶剂相互作用并直接纳入温度效应，从而实现对分子溶解度更准确且带不确定性估计的预测。

    

    分子溶解度直接影响分子开发中的关键环节，例如反应可行性、制剂性能、分离效率和溶剂选择。然而，跨溶质、溶剂和温度的实验测量仍然成本高昂且采样稀疏。现有的计算模型通常依赖于固定溶剂假设、确定性公式或简化的溶质-溶剂相互作用表示，限制了它们捕捉复杂分子相互作用、连续温度效应和实验不确定性的能力。在此，我们提出了EnSol，一种用于分子溶解度预测的环境感知概率框架。EnSol将溶质和溶剂表示为分子图，并分别学习两者的表征，随后通过交叉注意力机制将它们结合起来以捕捉溶质-溶剂相互作用。温度被直接纳入溶剂环境中（原文摘要在此处截断）。

    arXiv:2609.21151v1 Announce Type: cross  Abstract: Molecular solubility directly affects key aspects of molecular development such as reaction feasibility, formulation performance, separation efficiency, and solvent selection. However, experimental measurement across solutes, solvents, and temperatures remains costly and sparsely sampled. Existing computational models often rely on fixed-solvent assumptions, deterministic formulations, or simplified representations of solute-solvent interactions, limiting their ability to capture complex molecular interactions, continuous temperature effects, and experimental uncertainty. Here, we introduce EnSol, an environment-aware probabilistic framework for molecular solubility prediction. EnSol represents the solute and solvent as molecular graphs and learns separate representations for each before bringing them together through cross-attention to capture solute-solvent interactions. Temperature is incorporated directly into the solvent environme
    
[^105]: 基于扩散模型不确定性感知优化的章鱼爬行多样化与自适应臂协调

    Diverse and Adaptable Arm Coordination for Octopus-Crawling via Diffusion-Based Uncertainty-Aware Optimization

    [https://arxiv.org/abs/2609.21138](https://arxiv.org/abs/2609.21138)

    本文提出基于扩散的不确定性感知优化算法DUO，首次将扩散控制应用于接触丰富仿真中的软体多臂机器人，使模拟章鱼无需示教即可学习多样化的爬行协调模式，并证明协调多样性本身能促进鲁棒自适应。

    

    章鱼爬行为利用冗余性的软体机器人提供了灵感，然而发现并组织多样化的协调模式以实现自适应仍然具有挑战性。为解决这一问题，我们提出了一种基于扩散的不确定性感知优化算法，该算法为模拟的肌肉驱动CyberOctopus学习无需示教的爬行控制器。这项工作首次将基于扩散的控制应用于接触丰富的仿真环境中的软体多臂机器人。通过在共享控制分布中嵌入多种运动行为，该方法使模拟章鱼能够应对动态物理约束，证明了学习到的协调多样性本身即可促进鲁棒的自适应能力。主要贡献包括：(i) 一种对称结构化的策略表示，将径向等价的控制器折叠到规范的方向扇区中；(ii) 一种在线黑盒优化策略……（摘要在此处被截断）

    arXiv:2609.21138v1 Announce Type: cross  Abstract: Octopus crawling motivates soft robots that exploit redundancy, yet discovering and organizing diverse coordination modes for adaptation remains challenging. To address this, we introduce a Diffusion-based Uncertainty-aware Optimization (DUO) algorithm that learns demonstration-free crawling controllers for a simulated, muscle-actuated CyberOctopus. This work represents the first application of diffusion-based control to soft multi-arm robots in contact-rich simulations. By embedding a variety of locomotion behaviors within a shared control distribution, this approach enables the simulated octopus to navigate dynamic physical constraints, demonstrating that learned coordination diversity inherently facilitates robust adaptation. The main contributions include: (i) a symmetry-structured policy representation that folds radially equivalent controllers into a canonical directional sector, (ii) an online black-box optimization strategy, th
    
[^106]: 面向全连接层稳定结构化稀疏化的逐层解耦方法

    Layerwise Decoupling for Stable Structured Sparsification of Fully Connected Layers

    [https://arxiv.org/abs/2609.21126](https://arxiv.org/abs/2609.21126)

    该论文提出一种逐层解耦的全连接层结构化稀疏化方法，证明了其与联合惩罚在最优性上的等价性，并展示出比耦合方法更强的鲁棒性——正则化强度可用范围更宽、灾难性过度剪枝发生率更低。

    

    我们提出了一种解耦的、逐层处理的方法，用于对预训练神经网络的全连接层进行结构化稀疏化。我们的方法不是对所有层进行联合惩罚，而是提取浅层双子网络，对内部权重进行归一化，并对每个块的外部权重矩阵施加结构化组惩罚，通过逐层处理来剪枝神经元并缩减每层的宽度。我们证明，对于任意正齐次激活函数，该受约束的解耦目标在最优点处等价于对内部和外部权重的某种特定联合惩罚，因此具有清晰的投影和近端算子表述形式。我们的核心发现是，这种解耦重构比耦合方法更加鲁棒。在数值实验中，与所测试的联合基线方法相比，该方法提供了更宽的正则化强度可用范围和更低灾难性过度剪枝发生率，同时保持了相当的准确性。

    arXiv:2609.21126v1 Announce Type: new  Abstract: We propose a decoupled, layerwise method for structurally sparsifying the fully connected layers of pretrained neural networks. Rather than penalizing all layers jointly, our approach extracts shallow two-layer subnetworks, normalizes the inner weights, and applies a structured group penalty to the outer weight matrix of each block, processing layers sequentially to prune neurons and reduce the width of each layer. We prove that the constrained decoupled objective is equivalent at optimality to a specific joint penalty on the inner and outer weights, for any positively homogeneous activation, and thus admits a clean projected and proximal formulation. Our central finding is that this decoupled reformulation is more robust than coupled methods. In numerical experiments it provides a wider usable range of the regularization strength and a lower rate of catastrophic over-pruning than the tested joint baseline while maintaining comparable ac
    
[^107]: 面向机器学习驱动应用的基于替代预处理与声学处理的以信号为中心的遥感方法

    Signal-Centric Remote Sensing via Alternative Preprocessing and Acoustic Processing for ML-Driven Applications

    [https://arxiv.org/abs/2609.21123](https://arxiv.org/abs/2609.21123)

    本文提出用CSV格式数据替代传统图像表示来处理声呐遥感数据，使处理时间减少91.18%，同时提升了机器学习目标检测精度以及SNR、PSNR等评估指标。

    

    处理声呐数据的主流方法是使用基于图像的表示，这需要在自主系统上对图像数据进行预处理。我们提出了一种用于遥感应用的替代数据处理方法，即使用逗号分隔值（CSV）格式的数据。对这种替代方法的实验表明，处理时间减少了91.18%，机器学习的准确目标检测能力得到提升，信噪比（SNR）、峰值信噪比（PSNR）及其他评估指标均有所提高。

    arXiv:2609.21123v1 Announce Type: new  Abstract: The dominant method of processing sonar data is using image-based representations, requiring the preprocessing of image data on autonomous systems. We propose an alternative data processing method for remote sensing applications via the use of data in Comma-Seperated Value format. Experimentation on our alternative approach shows a reduction of processing time by 91.18%, an improvement in accurate object detection by Machine Learning, and an increase in SNR (Signal-to-noise ratio), PSNR (Peak signal-to-noise ratio), and other evaluation metrics.
    
[^108]: 与我对话，Jarvis：一个面向自动驾驶赛车的开源边缘可部署语音助手框架

    Talk to Me, Jarvis: An Open-Source Edge-Deployable Voice Assistant Framework for Autonomous Racecars

    [https://arxiv.org/abs/2609.21109](https://arxiv.org/abs/2609.21109)

    该论文提出了Jarvis，一个面向自动驾驶赛车的开源边缘可部署离线语音助手框架，通过对Mistral 7B模型进行领域特定微调实现低延迟的文本到指令分类，意图识别准确率达97.63%，性能优于更大的在线托管模型。

    

    大语言模型的最新进展提高了其作为语音助手后端组件的有效性，特别是在意图理解和上下文感知的输入分类方面。然而，在线托管模型引入了网络依赖性和可变的推理延迟，限制了其在时间敏感的自动驾驶应用中的适用性。在本工作中，我们通过开发Jarvis来解决这些问题，Jarvis是一个面向自动驾驶车辆高级行为指令的离线语音助手。其架构将语音识别与合成以及自然语言指令分类集成到一个轻量级的本地框架中。Jarvis的核心组件是一个文本到指令的分类器，该分类器通过对Mistral 7B模型进行领域特定的微调而构建，展现出低延迟推理能力。我们的实验评估表明，我们的解决方案优于更大的在线托管模型，实现了97.63%的意图识别准确率。

    arXiv:2609.21109v1 Announce Type: new  Abstract: Recent advances in large language models have improved their effectiveness as back-end components for voice assistants, particularly in intent understanding and context-aware input classification. However, online-hosted models introduce network dependency and variable inference latency, limiting their suitability for time-critical autonomous driving applications. In this work, we address these issues by developing Jarvis, an offline voice assistant for high-level behavioral commands of autonomous vehicles. Its architecture integrates speech recognition and synthesis with natural language command classification into a lightweight, local framework. Jarvis core component is a text-to-command classifier, built using a domain-specific fine-tuning of the Mistral 7B model, demonstrating low-latency inference. Our experimental evaluation demonstrates that our solution outperforms larger online-hosted models, achieving 97.63 % intent recognition 
    
[^109]: REFINEPPO：通过迭代动作精化学习连续控制策略

    REFINEPPO: Learning Continuous Control Policies by Iterative Action Refinement

    [https://arxiv.org/abs/2609.21108](https://arxiv.org/abs/2609.21108)

    提出迭代动作精化方法，让策略从初始动作出发通过一系列学习到的残差修正迭代地改进控制动作，突破了传统连续控制策略“单次前向直接输出动作”的局限，从而提升策略学习效果。

    

    深度强化学习（DRL）在广泛的连续控制问题上取得了出色的性能。然而，这些连续控制策略通常被定义为从观测状态到动作或动作分布的直接映射，需要单个前馈网络一次性构建出最优控制决策。这种方法虽然有效，但一旦形成初始预测，策略几乎没有机会重新考虑或逐步改进动作。在本工作中，我们探索了一种替代方法：策略是否可以不仅学习直接预测动作，而是学习迭代地改进动作，并且这一迭代过程能否在策略学习中带来优势？我们提出了迭代动作精化，这是一种迭代的动作构建方法，通过一系列学习到的残差修正来逐步构建控制动作。从初始提案动作出发（原文摘要在此处截断）。

    arXiv:2609.21108v1 Announce Type: new  Abstract: Deep reinforcement learning (DRL) has achieved strong performance across a wide range of continuous-control problems. These continuous-control policies, however, are often defined as direct mappings from an observed state to an action or action distribution, requiring a single feed-forward network to construct an optimal control decision in one pass. While effective, this formulation leaves little opportunity for the policy to reconsider or progressively improve an action once an initial prediction has been formed. In this work, we explore an alternative approach: rather than learning only to directly predict an action, can a policy learn to iteratively improve one, and can this iterative process provide advantages during policy learning? We introduce Iterative Action Refinement (IAR), an iterative action-construction method that constructs control actions through a sequence of learned residual corrections. Starting from an initial propo
    
[^110]: 检测大语言模型中的幻觉：追踪受损上下文共享的拓扑特征

    Detecting Hallucination in LLMs: Tracing the Topological Signatures of Impaired Context Sharing

    [https://arxiv.org/abs/2609.21096](https://arxiv.org/abs/2609.21096)

    该论文通过分析注意力图中Forman-Ricci曲率等拓扑特征来捕捉因果生成过程中受损的上下文共享模式，提出了一种单次遍历的幻觉检测方法，在多个基准和不同LLM架构上均优于现有基线。

    

    在这项工作中，我们研究了注意力图内信息流模式的拓扑结构，以有效区分幻觉与非幻觉响应。我们通过分析Forman-Ricci曲率来识别注意力图中指示信息瓶颈的结构模式，进而提出了一种能够捕捉与幻觉响应相关的注意力头的半局部和全局信息流特征的方法。我们在多个大语言模型和已有基准上对该方法进行了广泛评估。实证结果表明，我们提出的单次遍历方法在两个幻觉检测基准上相比现有的基于注意力和多响应的基线方法取得了一致的改进，同时在多样化的LLM架构上实现了具有竞争力的性能。进一步的分析揭示，因果生成过程中词元之间受损的上下文共享与幻觉的产生密切相关。

    arXiv:2609.21096v1 Announce Type: new  Abstract: In this work, we examine the topology of information flow patterns within attention graphs to effectively distinguish hallucinated from non-hallucinated responses. We analyze the Forman-Ricci curvature to identify structural patterns indicating information bottlenecks in attention graphs. We then introduce a method that captures both semi-local and global information-flow characteristics of attention heads associated with hallucinated responses. We evaluate our approach extensively across several LLMs and established benchmarks. Empirical results demonstrate that our proposed single-pass approach provides consistent improvements over existing attention-based and multi-response baselines across two hallucination-detection benchmarks, while achieving competitive performance across diverse LLM architectures. Further analysis reveals that impaired context sharing among tokens during causal generation is strongly associated with hallucination
    
[^111]: 三重可扩展的等变高斯过程建模

    Triply-Scalable Equivariant Gaussian Process Modeling

    [https://arxiv.org/abs/2609.21085](https://arxiv.org/abs/2609.21085)

    该论文提出了三重可扩展的等变高斯过程，通过等变稀疏变分推断与无积分等变核的结合，实现了兼具等变性、不确定性量化与大规模可扩展性的数据高效高斯过程建模。

    

    高斯过程（GPs）在编码先验知识（包括等变性）的同时，能够提供有原则的概率预测。然而，其在大规模科学问题中的应用受到计算成本的限制。等变神经网络虽然常见，但通常缺乏高斯过程所提供的不确定性量化能力，而这种能力在分子研究等应用中非常有价值。高维输入和大的对称群进一步对可扩展性提出了要求。我们建立了关于高斯过程等变性与条件化之间相互作用的相关结果，并利用这些结果，通过合适的均值函数和协方差核构造出等变稀疏高斯过程。我们用一类灵活的无积分等变核来实例化该框架，从而实现可扩展且数据高效的高斯过程推断。特别地，我们提出了三重可扩展的等变高斯过程。我们将等变稀疏变分高斯过程应用于 SO(2)（摘要在此处截断）

    arXiv:2609.21085v1 Announce Type: cross  Abstract: Gaussian processes (GPs) provide principled probabilistic predictions while encoding prior knowledge, including equivariances. Yet, their use in large-scale scientific problems is limited by computational cost. Equivariant neural networks are common but typically lack the uncertainty quantification offered by GPs, which is valuable in applications such as molecular research. High-dimensional inputs and large symmetry groups further demand scalability. We establish results pertaining to the interplay of GP equivariance and conditioning and leverage them to obtain equivariant sparse GPs through suitable mean functions and covariance kernels. We instantiate this framework with a flexible class of integration-free equivariant kernels, yielding scalable and data-efficient GP inference. In particular, we introduce triply scalable equivariant Gaussian processes.   We employ equivariant sparse variational Gaussian processes for $\mathrm{SO}(2)
    
[^112]: 基于感知调整查询实现情感识别的个体级校准

    Toward individual-level calibration in affect recognition with perceptual adjustment queries

    [https://arxiv.org/abs/2609.21073](https://arxiv.org/abs/2609.21073)

    提出一种基于感知调整查询（PAQ）估计个体恰可察觉差（JND）的校准框架，能够消除面部情感识别任务中感知敏感性的个体差异，实现个体水平的任务难度均衡。

    

    测量面部情感感知的行为任务通常假设相同的刺激对所有参与者具有等效的感知难度。然而，感知敏感性的个体差异会系统性地违反这一假设。我们以情感感知任务为测试平台，提出了一个对感知难度进行归一化的框架，该框架通过认知负荷较轻的感知调整查询（PAQ）直接估计每位参与者沿面部情感谱的恰可察觉差。我们利用这些由PAQ推断出的JND重新表达刺激距离，在感知空间中构建难度等同的任务。我们在二选一强迫选择（2AFC）任务中，通过两种互补的行为测量方法——二元元认知难度判断和反应时方差分解——对该框架进行了验证。我们发现PAQ校准在个体水平上显著均衡了感知到的任务难度。

    arXiv:2609.21073v1 Announce Type: new  Abstract: Behavioral tasks measuring facial affect perception assume that identical stimuli impose equivalent perceptual difficulty across participants. However, this assumption is systematically violated by individual differences in perceptual sensitivity. Using an affective perception task as our testbed, we propose a framework to normalize for perceptual difficulty that directly estimates each participant's Just Noticeable Difference (JND) along the facial affect spectrum via cognitively lightweight perceptual adjustment queries (PAQs). We use these PAQ-inferred JNDs to re-express stimulus distances, constructing difficulty-equated tasks in perceptual space. We validate the framework in a Two-Alternative Forced-Choice (2AFC) task using two complementary behavioral measures: binary metacognitive difficulty judgments and response time variance decomposition. We find that PAQ calibration significantly equalizes perceived task difficulty at an indi
    
[^113]: FedeRage：一般客户端漂移下可证明收敛的不可知联邦学习

    FedeRage: Provably Convergent Agnostic Federated Learning under General Client Drift

    [https://arxiv.org/abs/2609.21057](https://arxiv.org/abs/2609.21057)

    该论文刻画了客户端参与完全未知且高度偏斜时FedAvg实际优化的随机目标函数，并在此基础上提出风险规避扩展算法FedeRage，实现了在一般客户端漂移下具有可证明收敛性的不可知联邦学习。

    

    联邦学习（FL）能够在不共享原始数据的情况下进行协作式模型训练，但其在非独立同分布（non-IID）数据和随机客户端参与情况下的性能会下降。基于经典联邦平均（FedAvg）构建的补救方法通常预设服务器知晓客户端的参与概率，而这在已部署的实际系统中很少成立。我们首先讨论并刻画了当客户端参与情况完全未知、可能高度偏斜且各轮参与规模可变时，“分布不可知”的FedAvg实际求解的优化问题：我们证明，均匀聚合能够最小化一个由参与诱导的边际加权、定义明确的随机目标，对于凸的且可能非光滑的损失，以标准的 $\mathcal{O}(1/\sqrt{T})$ 速率收敛。基于这一刻画，我们提出了联邦风险规避平均（FedeRage），这是FedAvg的一种风险规避扩展，它嵌入了……（摘要原文在此处截断）

    arXiv:2609.21057v1 Announce Type: new  Abstract: Federated learning (FL) enables collaborative model training without sharing raw data, but its performance degrades under non-IID data and stochastic client participation. Remedies built on classical Federated Averaging (FedAvg) typically presuppose that client participation probabilities are known to the server, which is rarely the case in deployed systems. We first discuss and then characterize the optimization problem that \emph{distributionally agnostic} FedAvg actually solves when participation is entirely unknown, possibly highly skewed, and of variable size across rounds: uniform aggregation is shown to minimize a well-defined stochastic objective, weighted by the participation-induced marginal, at a standard $\mathcal{O}(1/\sqrt{T})$ rate for convex and possibly nonsmooth losses. Building on this characterization, we propose \emph{Federated Risk-Averse Averaging} (\textsc{FedeRage}), a risk-averse extension of FedAvg that embeds 
    
[^114]: 潜在空间中的基于物理的渲染

    Physically Based Rendering in the Latent Space

    [https://arxiv.org/abs/2609.21054](https://arxiv.org/abs/2609.21054)

    该论文提出在生成模型变分自编码器的潜在空间中直接进行基于物理的渲染，通过修改渲染方程并结合可微渲染器，实现了物理引导的可控内容生成。

    

    图像扩散模型已展现出令人印象深刻的图像生成能力，但通常难以控制，这与基于物理的渲染等经典计算机图形学流程形成了对比。然而，我们观察到光传输现象与此类模型产生的潜在空间值分布之间存在着一种桥梁。因此，我们在生成模型中变分自编码器学习到的特征空间中引入了基于物理的渲染，实现了潜在空间中的光传输模拟。这使我们能够利用基于物理的渲染技术来输出潜在图，从而实现物理引导的内容生成。我们提出了对渲染方程的修改，当与可微渲染器配合使用时，可以获得一组最优的场景参数，只需极少的微调即可准确地渲染到预训练的潜在空间中。我们在单张渲染图像上训练我们的方法……

    arXiv:2609.21054v1 Announce Type: cross  Abstract: Image diffusion models have shown impressive image generation capabilities but are often hard to control, in contrast to classical computer graphics pipelines such as physically based rendering. However, we observe that there is a bridge between light transport phenomena and the distribution of latent space values produced by such models. Thus, we introduce physically based rendering in the feature space learned by the variational autoencoders in generative models, enabling light transport simulation in the latent space. This allows us to leverage physically based rendering techniques to output latent maps for physically guided content generation. We propose modifications to the rendering equation, which, when paired with a differentiable renderer, can yield an optimal set of scene parameters that require only minimal refinement to accurately render into the pretrained latent space. We train our method on a single rendered image, and t
    
[^115]: 面向基于Transformer的时间序列预测器的轻量级即插即用门控

    A Lightweight Plug-in Gate for Transformer-Based Time-Series Forecasters

    [https://arxiv.org/abs/2609.21044](https://arxiv.org/abs/2609.21044)

    本文提出一种轻量级即插即用的编码器前门控接口，通过为协变量表示单元分配sigmoid分数来调控外部变量进入基于Transformer的时间序列预测器，可作为TimeXer、iTransformer和PatchTST的模块在零额外调参条件下使用。

    

    富含协变量的时间序列预测需要决定外部变量如何进入目标预测路径。现有的基于Transformer的预测器通常构建协变量表示并将其直接传递给编码器，而缺乏显式的准入阶段。本文研究了编码器前协变量准入作为一种输入侧接口，在编码器处理之前对该表示进行即时调控。我们用一个轻量级的表示级编码器前门控来实现该接口，该门控为表示单元分配sigmoid分数，同时还研究了一种使用正则化的变体，对平均准入程度进行惩罚。该接口作为即插即用模块，在零额外调参协议下于TimeXer、倒置Transformer（iTransformer）和Patch时间序列Transformer（PatchTST）上进行评估，其中每个带门控的模型均继承相应基线的配置。实验在电力变压器温度分钟级数据集（ET……（摘要在此处截断）

    arXiv:2609.21044v1 Announce Type: new  Abstract: Covariate-rich time-series forecasting requires deciding how external variables enter the target forecasting path. Existing Transformer-based forecasters usually build a covariate representation and pass it to the encoder without an explicit admission stage. This paper studies pre-encoder covariate admission as an input-side interface that regulates that representation immediately before encoder processing. We implement the interface with a lightweight representation-level pre-encoder gate that assigns sigmoid scores to representation units, and we also study a usage-regularized variant that penalizes average admission. The interface is evaluated as a plug-in module for TimeXer, Inverted Transformer (iTransformer), and Patch Time Series Transformer (PatchTST) under a zero-extra-tuning protocol, where each gated model inherits the corresponding baseline configuration. Experiments on the Electricity Transformer Temperature minute-level (ET
    
[^116]: Stiefel-AdamW：面向线性分解模块的几何感知AdamW优化器

    Stiefel-AdamW: Geometry-Aware AdamW for Linear Factorization Blocks

    [https://arxiv.org/abs/2609.21039](https://arxiv.org/abs/2609.21039)

    提出Stiefel-AdamW优化器，通过将线性分解模块（如LoRA适配器、自注意力查询-键乘积）中的一个因子约束在Stiefel流形上，解决分解不唯一导致的训练不稳定和因子爆炸问题，可作为AdamW的近似即插即用替代方案。

    

    现代深度学习中一种普遍存在的结构模式是线性分解模块：形如 $W = BA$ 的子模块，其中两个参数矩阵直接相乘，中间没有任何非线性层。此类模块出现在LoRA适配器、低秩压缩层、自注意力的查询-键乘积中，它们共同存在一个病态问题：分解不唯一，这可能导致训练不稳定并限制可用的学习率。尽管如此，分解模块通常仍使用忽略底层几何结构的标准欧几里得优化方法进行优化。我们提出了Stiefel-AdamW，这是AdamW的近似即插即用替代方案，可应用于任何出现此类分解模块的地方。通过将其中一个因子约束在Stiefel流形上，同时保持另一个因子为欧几里得空间，Stiefel-AdamW将完整的 $\mathrm{GL}(\mathbb{R}^r)$ 规范对称性松弛为紧致的正交对称性，从而排除了因子爆炸的可能性，同时保留了逐坐标的对角预条件机制。

    arXiv:2609.21039v1 Announce Type: new  Abstract: A pervasive structural pattern in modern deep learning is the linear factorization block: a submodule of the form $W = BA$ in which two parameter matrices are multiplied directly, with no intervening nonlinearity. Such blocks appear in LoRA adapters, low-rank compressed layers, query-key products of self-attention, and share a common pathology: the factorization is non-unique, which can destabilize training and limit usable learning rates. Despite this, factorization blocks are typically optimized with standard Euclidean methods that ignore the underlying geometry. We introduce Stiefel-AdamW, a near drop-in replacement for AdamW for use wherever such blocks appear. By constraining one factor on the Stiefel manifold while leaving the other Euclidean, Stiefel-AdamW relaxes the full $\mathrm{GL}(\mathbb{R}^r)$ gauge symmetry to a compact orthogonal symmetry, ruling out factor blow-up while retaining the coordinate-wise diagonal precondition
    
[^117]: 通过测试时通信扩展科学发现

    Scaling Discovery through Test-Time Communication

    [https://arxiv.org/abs/2609.21032](https://arxiv.org/abs/2609.21032)

    该研究发现多智能体测试时通信使 $k$ 个协作智能体达到 $4k$ 个独立智能体的成功率，优势随规模扩大而复合增长，且团队能可靠解决任何单个智能体都无法完成的任务。

    

    科学的进步并非孤立发生，而是通过协作实现的，然而现有的智能体系统很少体现这一点。智能体之间的通信是否有帮助仍是一个开放性问题，此前的研究结果并不一致。我们证明，在具有挑战性的任务上，测试时通信能够大幅超越独立的并行尝试，因为分享一个突破性进展可以推动整个团队前进。我们首先研究了扩展多智能体测试时通信的效果——其中智能体没有预定义的角色，通过共享目录进行通信——并在 ARC-AGI-3 这一要求新颖问题解决能力的基准上进行评估。我们发现，由 $k$ 个通信智能体组成的团队（team@$k$）可以达到 $4k$ 个独立智能体的成功率，且这一优势随 $k$ 的增长而扩大，表明收益随规模呈复合增长。这种效果不仅仅是效率上的提升：单个智能体无法解决的任务，一个智能体团队却能够可靠地解决。此外，这些收益还能迁移到面向研究的任务中。

    arXiv:2609.21032v1 Announce Type: cross  Abstract: Science advances not in isolation but through collaboration, yet existing agentic systems capture little of this. Whether communicating agents help remains an open question with mixed prior results. We show that test-time communication can substantially outperform independent parallel attempts on challenging tasks, where sharing a breakthrough can push the whole group forward. We first study the effect of scaling multi-agent test-time communication, where agents have no predefined roles and communicate via a shared directory, on ARC-AGI-3, a benchmark requiring novel problem solving. We find that a team of $k$ communicating agents, team@$k$, matches the success rate of $4k$ independent agents, and this advantage grows with $k$, suggesting gains compound with scale. The effect is not merely efficiency: a task that no single agent can solve, a team of agents can solve reliably. Furthermore, these gains transfer to research-oriented tasks
    
[^118]: 随机特征方法与神经网络的光滑偏差原则

    A Smoothed Discrepancy Principle for Random Feature Methods and Neural Networks

    [https://arxiv.org/abs/2609.21017](https://arxiv.org/abs/2609.21017)

    该论文提出了一种基于光滑化偏差原则的多尺度早停规则，在所有光滑度水平上实现完全自适应的最优统计性能，并能以数据驱动的方式同时确定最优停止时间、随机特征数量以及神经网络宽度。

    

    我们研究了经典非参数回归设定下谱正则化方法的数据驱动早停策略。基于偏差原则，我们提出了一种适用于一般核估计器的多尺度停止规则，并证明与以往方法不同，该规则在模型正确设定的情形下能够在所有光滑度水平上实现完全自适应性。我们工作的一个关键贡献是基于随机特征近似的扩展方法，该方法在降低大规模数据集计算成本的同时，保持了极小化极大最优的统计保证。我们的方法不仅能选择最优停止时间，还能以完全数据驱动的方式选择达到最优收敛速率所需的随机特征数量。通过随机特征与神经切核机制下神经网络之间已建立的联系，我们的方法进一步为网络宽度提供了有原则的、数据驱动的推荐。

    arXiv:2609.21017v1 Announce Type: cross  Abstract: We study data-driven early stopping for spectral regularisation methods in the classical non-parametric regression setting. Building on the discrepancy principle, we propose a multi-scale stopping rule that applies to general kernel estimators and show that, unlike previous approaches, it achieves full adaptivity over all smoothness levels in the well-specified case. A key contribution of our work is an extension based on random feature approximations, which reduces computational cost on large datasets while preserving minimax-optimal statistical guarantees. Our procedure not only selects an optimal stopping time but also provides a fully data-driven choice of the number of random features needed to achieve optimal rates. Through the established connection between random features and neural networks in the neural tangent kernel regime, our method further yields a principled, data-driven recommendation for the network width. We prove th
    
[^119]: 面向壁画残片风格分类的片段感知视觉Transformer

    Fragment-Aware Vision Transformers for Fresco-Fragment Style Classification

    [https://arxiv.org/abs/2609.21012](https://arxiv.org/abs/2609.21012)

    该论文提出一种片段感知的视觉Transformer框架，通过前景引导掩码、基于图像修复的几何正则化和基于KL相似度的有监督对比学习，实现从不完整、不规则的壁画残片中进行艺术风格分类。

    

    艺术风格分类通常在完整的艺术品上进行研究，此时模型可以利用全局构图、空间组织和图像学结构。然而在考古环境中，艺术品往往仅以残片的形式存世，迫使模型从不完整、不规则且上下文受限的视觉证据中进行识别。我们使用一个渐进式的基于Transformer的框架来研究壁画残片风格分类。从ViT-B/16基线出发，我们引入了前景引导掩码以抑制仅包含背景的token，基于图像修复的几何正则化以将不规则的残片支撑域与ViT补丁网格对齐，以及一种通过Kullback-Leibler相似度作用于预测分布的有监督对比目标，该目标持续改进每个分支。我们采用一个刻意设计得简单易用的可学习logit集成来组合各个分支。在CLEOPATRA和POMPAAF数据集上的实验表明，片段感知的模……

    arXiv:2609.21012v1 Announce Type: cross  Abstract: Artistic style classification is usually studied on complete artworks, where models can exploit global composition, spatial organisation, and iconographic structure. In archaeological settings, however, artworks often survive only as fragmented remains, forcing recognition from incomplete, irregular, and context-limited visual evidence. We study fresco-fragment style classification using a progressive transformer-based framework. Starting from a ViT-B/16 baseline, we introduce foreground-guided masking to suppress background-only tokens, inpainting-based geometric regularisation to align irregular fragment supports with the ViT patch grid, and a supervised contrastive objective that operates on predictive distributions through a Kullback-Leibler similarity and consistently improves every branch. We combine the branches with a deliberately simple learnable logit ensemble. Experiments on CLEOPATRA and POMPAAF show that fragment-aware mod
    
[^120]: 关于最大编码率降低在分布外泛化中的局限性研究

    On the Limits of Maximal Coding Rate Reduction for Out-of-Distribution Generalisation

    [https://arxiv.org/abs/2609.21001](https://arxiv.org/abs/2609.21001)

    该论文揭示了最大编码率降低（MCR²）目标在分布外泛化中的两个根本局限性，表明仅优化 MCR² 可能得到基于不稳定环境特征、虽达到全局编码最优却在分布偏移下完全无法预测的表示。

    

    大量研究致力于使深度学习的目标函数、表示和架构具备可解释性，以提升学习系统在各类现实应用中的安全性、鲁棒性和泛化能力。最近提出的最大编码率降低（MCR²）提供了一个有前景的信息论框架，用于学习类内子流形的结构化、判别性表示，并启发了可解释的白盒架构。然而，我们观察到 MCR² 在分布偏移下可能完全失效，这促使我们研究其在分布外（OOD）泛化上的局限。我们建立了 MCR² 在 OOD 泛化方面的两个局限性：首先，仅依靠 MCR² 目标函数可能导致完全的预测失败——一个完全基于不稳定环境特征的表示可以达到全局编码最优，但仍然无法……

    arXiv:2609.21001v1 Announce Type: new  Abstract: Substantial efforts have been devoted to making deep learning objectives, representations, and architectures interpretable, with the goal of improving the safety, robustness, and generalisation of learning systems in diverse real-world applications. The recently proposed maximal coding rate reduction ($\mathrm{MCR}^{2}$) offers a promising information-theoretic framework for learning structured, discriminative representations of class-wise submanifolds and has inspired interpretable white-box architectures. However, we observe that $\mathrm{MCR}^{2}$ can completely fail under distribution shift, motivating our study of its out-of-distribution (OOD) generalisation limits. We establish two limitations of $\mathrm{MCR}^{2}$ for OOD generalisation. First, the $\mathrm{MCR}^{2}$ objective alone can admit complete prediction failure: a representation based entirely on unstable environmental features can achieve the global coding optimum yet fa
    
[^121]: 生成建模中的聚合后验预测检验

    Aggregated Posterior Predictive Checks for Generative Modeling

    [https://arxiv.org/abs/2609.20999](https://arxiv.org/abs/2609.20999)

    本文提出了聚合后验预测检验（APPC）这一新方法，用于检验生成模型中从聚合后验而非先验采样的两阶段合成数据生成流程，并从理论上证明了该检验渐近校准的充分条件。

    

    潜变量生成模型通常使用潜变量上的简单先验进行拟合，但从这些先验中抽取的样本往往无法产生真实的数据。这种失败源于先验与聚合后验（即由拟合模型和数据所诱导的潜变量分布）之间的不匹配。这种不匹配通常被视为先验设定错误的证据，因而应当被替换。另一种做法是，在现代生成模型中越来越多地采用两阶段策略：首先拟合模型，其次估计聚合后验（van den Oord et al., 2017; Rombach et al., 2022），然后通过从聚合后验而非先验中采样来获得合成数据。为了检验这类方法，我们提出了聚合后验预测检验（APPC）。在理论方面，我们建立了APPC渐近校准的充分条件。

    arXiv:2609.20999v1 Announce Type: cross  Abstract: Latent variable generative models are commonly fit using simple priors over latent variables, but draws from these priors often fail to produce realistic data. This failure is due to a mismatch between the prior and the aggregated posterior, the distribution of latent variables induced by the fitted model and the data. This mismatch is often viewed as evidence that the prior is misspecified and should be replaced. Alternatively, in modern generative models, a two-stage strategy is increasingly used where first, the model is fit, and second, the aggregated posterior is estimated (van den Oord et al.,2017; Rombach et al., 2022.). Synthetic data are then obtained by sampling from this aggregated posterior instead of the prior. To check such procedures, we introduce the aggregated posterior predictive check (APPC). Theoretically, we establish sufficient conditions under which the APPC is asymptotically calibrated. For probabilistic princip
    
[^122]: MOSAIC-SR：基于Transformer引导的符号回归用于科学方程恢复

    MOSAIC-SR: Transformer-Guided Symbolic Regression for Scientific Equation Recovery

    [https://arxiv.org/abs/2609.20997](https://arxiv.org/abs/2609.20997)

    MOSAIC-SR利用预训练Transformer生成多个初始表达式草图来引导符号回归搜索，结合尺度感知常数优化与局部符号修复，解决了现有方法中结构搜索效率低与神经模型符号错误的难题，实现更准确高效的科学方程恢复。

    

    符号回归旨在从观测数据中恢复闭式方程，为科学发现提供可解释的模型。现有方法难以将灵活的结构搜索与高效推理相结合：基于搜索的方法可以优化表达式结构，但通常依赖代价高昂的组合优化且采用随机初始化；预训练神经模型几乎可以即时生成公式，但其预测往往包含符号错误。我们提出了MOSAIC-SR，它利用预训练的Transformer提出多个初始表达式草图，这些草图在多个有前景的区域初始化搜索，从而避免了在庞大表达式空间中的随机起步。每次搜索通过尺度感知的常数优化和局部符号修复，联合恢复表达式结构与常数。我们在包含和不包含虚拟变量的SRSD-Feynman数据集以及六个额外的基准测试上对MOSAIC-SR进行了评估。（注：原摘要在此处被截断，评估结果部分不完整）

    arXiv:2609.20997v1 Announce Type: new  Abstract: Symbolic regression aims to recover closed-form equations from observations, providing interpretable models for scientific discovery. Existing approaches struggle to combine flexible structural search with efficient inference. Search-based methods can refine expression structure but often rely on costly combinatorial optimization with random initialization. Pretrained neural models generate formulas almost instantly, but their predictions often contain symbolic errors. We introduce MOSAIC-SR, which uses a pretrained Transformer to propose multiple initial sketches. These sketches initialize searches in several promising regions, avoiding random starts in the vast expression space. Each search jointly recovers structure and constants through scale-aware constant optimization and local symbolic repair. We evaluate MOSAIC-SR on the SRSD-Feynman dataset with and without dummy variables and on six additional benchmarks. MOSAIC-SR obtains the 
    
[^123]: 从压力到情感：跨可穿戴传感器模态的生理情绪识别多模态深度学习

    From Stress to Affect: Multimodal Deep Learning for Physiological Emotion Recognition Across Wearable Sensor Modalities

    [https://arxiv.org/abs/2609.20991](https://arxiv.org/abs/2609.20991)

    该研究在 WESAD 和 EmoWear 两个多模态可穿戴数据集上，系统比较了 LSTM、TCN 和 Transformer 在不同传感配置下的生理情绪识别性能，发现 Transformer 在多模态配置下取得最高准确率。

    

    基于可穿戴传感器的生理情绪识别在心理健康监测、情感计算和人机交互中具有重要应用。然而，现有研究通常仅评估单一模型、传感配置或数据集，限制了我们理解这些因素如何影响识别性能。我们提出了一项关于时序深度学习架构用于生理情绪识别的比较研究，使用了两个多模态可穿戴数据集：WESAD 和 EmoWear。在仅手腕、仅胸部和多模态三种传感配置下，采用与被试无关的留一被试交叉验证（LOSO-CV）评估了双向长短期记忆网络（LSTM）、时序卷积网络（TCN）和 Transformer 模型。我们还研究了软投票集成、传感器消融、采样频率以及基于梯度的显著性分析。Transformer 在 W……（原文摘要在此处截断）

    arXiv:2609.20991v1 Announce Type: new  Abstract: Physiological emotion recognition using wearable sensors has important applications in mental health monitoring, affective computing, and human-computer interaction. However, existing studies typically evaluate a single model, sensing configuration, or dataset, limiting our understanding of how these factors influence recognition performance. We present a comparative study of temporal deep learning architectures for physiological emotion recognition using two multimodal wearable datasets: WESAD and EmoWear. Bidirectional long short-term memory (LSTM), temporal convolutional network (TCN), and Transformer models are evaluated under wrist-only, chest-only, and multimodal sensing configurations using participant-independent leave-one-subject-out cross-validation (LOSO-CV). We also investigate soft-voting ensembles, sensor ablation, sampling frequency, and gradient-based saliency. The Transformer achieved the highest multimodal accuracy on W
    
[^124]: ASGARD：基于强化学习的无人机动作空间防护方法

    ASGARD: Action-Space Guard for UAV Resilience via Reinforcement Learning

    [https://arxiv.org/abs/2609.20982](https://arxiv.org/abs/2609.20982)

    提出了ASGARD——一个两阶段师生框架，利用攻击相关的特权信息训练编码器和监测器，在运行时检测并校正被篡改的动作指令，使基于强化学习的无人机控制具备抵御动作空间攻击的韧性。

    

    强化学习（RL）控制器近来已被应用于无人机（UAV）的导航与控制。然而，它们容易受到动作空间攻击的影响——这类攻击会在策略生成动作指令之后、执行器执行之前覆盖这些指令。尽管大多数现有防御手段针对的是策略输入端的攻击，但那些应对动作空间攻击的方法需要在训练阶段重新训练策略，且在运行时无法对被篡改的动作保持韧性。我们提出了ASGARD，一个两阶段的师生训练流水线，用于使基于强化学习的无人机控制能够抵御动作空间攻击。在教师阶段，编码器将无人机的物理状态与动作攻击相关的特权信息相结合，生成一个动作攻击感知的潜在表示，用于训练强化学习控制策略和一个监测器，该监测器向执行器输出经过校正的动作指令。在学生阶段，编码器和监测器均通过监督学习进行训练……

    arXiv:2609.20982v1 Announce Type: new  Abstract: Reinforcement learning (RL) controllers have been recently adopted for Unmanned Aerial Vehicles (UAV) navigation and control. However, they are susceptible to action-space attacks that overwrite the action commands after the policy generates them and before the actuators execute them. While most existing defenses target attacks on the policy's inputs, those addressing action-space attacks retrain the policy at training time and are not resilient to corrupted actions at runtime. We propose ASGARD, a two-phase teacher-student pipeline for making RL-based UAV control resilient to action-space attacks. In the teacher phase, an encoder combines the UAV's physical state with action-attack-related privileged information to produce an action-attack-aware latent that trains the RL control policy and a monitor that outputs corrected action commands to the actuators. In the student phase, both the encoder and the monitor are trained via supervised 
    
[^125]: 用于早期排序竞争性地质解释的生成式反演

    Generative inversion for early ranking of competing geologic interpretations

    [https://arxiv.org/abs/2609.20978](https://arxiv.org/abs/2609.20978)

    该论文提出一种生成式反演工作流程，将相互竞争的地质解释转化为空间先验，利用文本到图像基础模型和变分自编码器，根据与水头观测的一致性对解释进行早期排序，从而在数据稀缺条件下辅助高后果地下决策。

    

    高后果的地下决策往往在严重的数据稀缺条件下做出。专家们可能对同一地下系统得出相互竞争的解释，但在项目早期很少有实用的方法来确定哪一种最符合实际。这种不确定性可能持续到钻探数口井之后，往往耗资数百万美元。现有的评估地质解释的方法要么依赖主观判断，要么依赖在早期调查中很少能获得的密集数据。我们提出了一种工作流程来应对这一挑战，它将相互竞争的地质解释转化为不同的空间先验，并根据它们与水头观测的一致性进行排序。对于每种解释，一个文本到图像的基础模型生成1600张地质图像的集合，并由一个单独训练的变分自编码器提供解释特定的潜在表示。

    arXiv:2609.20978v1 Announce Type: new  Abstract: High-consequence subsurface decisions are often made under severe data scarcity. Experts may arrive at competing interpretations of the same subsurface system, yet early in a project there is rarely a practical way to determine which one is most realistic. This uncertainty can persist until several wells are drilled, often costing millions of dollars. Existing approaches for evaluating geologic interpretations rely either on subjective judgment or on dense data that are rarely available in early-stage investigations. We present a workflow that addresses this challenge by translating competing geologic interpretations into alternative spatial priors and ranking them according to their consistency with hydraulic-head observations. For each interpretation, a text-to-image foundation model generates an ensemble of 1600 geologic images, and a separately trained variational autoencoder provides an interpretation-specific latent representation.
    
[^126]: 大语言模型中的复杂问题求解：统计控制综述与诊断框架

    Complex Problem Solving in Large Language Models: A Statistical Control Survey and Diagnostic Framework

    [https://arxiv.org/abs/2609.20973](https://arxiv.org/abs/2609.20973)

    该综述的核心贡献是将大语言模型的复杂问题求解重新建模为一个对潜在解状态的序贯估计与决策（统计控制）问题，由控制器维护信念并决定提交、验证、分支、回滚或弃权，从而为早期错误放大、提示脆弱性和无法纠错等失效现象提供统一的诊断框架。

    

    大语言模型（LLM）的复杂问题求解（CPS）通常被归结为更强的推理能力或更长的生成。然而，早期步骤的错误放大、提示词的脆弱性以及无法修正错误承诺等现象，很难仅用知识缺失或表达能力不足来解释。本综述将复杂问题求解解读为一个关于潜在解状态的序贯估计与决策问题：控制器对一条不可观测的求解轨迹维护一个信念，随着带噪声的中间证据到来而不断更新该信念，并决定是提交、验证、分支、回滚还是弃权，以最小化期望损失。推理负责提供候选转移与解释，而过程控制则对这些候选进行塑形与评估，并调控后续的转移与观测。在该框架下，我们围绕五个组件来组织现有方法：显式状态表示、转移结构化、价……

    arXiv:2609.20973v1 Announce Type: cross  Abstract: Complex problem solving (CPS) with large language models (LLMs) is often framed as a matter of stronger reasoning or longer generation. Yet early-step error amplification, prompt brittleness, and failures to revise incorrect commitments are difficult to explain by missing knowledge or expressive capacity alone. This survey interprets CPS as a sequential estimation-and-decision problem over a latent solution state. A controller maintains a belief about an unobserved solution trajectory, updates it as noisy intermediate evidence arrives, and decides whether to commit, verify, branch, roll back, or abstain to minimize expected loss. Reasoning supplies candidate transitions and interpretations, whereas process control shapes and evaluates those proposals and regulates subsequent transitions and observations. Within this framework, we organize existing methods around five components: explicit state representation, transition structuring, va
    
[^127]: 从切换遗憾到动态遗憾：一种基于无偏随机序列的简单归约方法

    From Switching to Dynamic Regret: A Simple Reduction via Unbiased Random Sequences

    [https://arxiv.org/abs/2609.20968](https://arxiv.org/abs/2609.20968)

    本文提出一个简单归约框架，通过构造每轮无偏、方差可控且切换次数可管理的辅助随机序列，将动态遗憾最小化转化为切换遗憾最小化，从而可直接利用现成的切换遗憾算法导出动态遗憾界。

    

    在非平稳在线学习中，动态遗憾作为衡量在线学习者相对于时变比较序列表现好坏的指标，受到了越来越多的关注。尽管已取得相当大的进展，但为强凸和指数凹损失获得最优界通常涉及复杂的分析。在本文中，我们提出了一个简单的框架，将动态遗憾最小化归约为切换遗憾最小化。因此，我们可以通过使用具有切换遗憾保证的现成算法来推导动态遗憾界。我们归约的关键思想是，对于任意比较序列，构造一个辅助随机序列，该序列在每一轮都是无偏的，具有可控的方差和可管理的切换次数。将这一构造与合适的替代损失相结合，我们可以将动态遗憾分解为相对于该随机序列的期望切换遗憾及其可控方差。

    arXiv:2609.20968v1 Announce Type: new  Abstract: In non-stationary online learning, dynamic regret has attracted increasing attention as a measure of how well an online learner performs against a time-varying comparator sequence. Despite considerable advances, attaining optimal bounds for strongly convex and exp-concave losses often involves intricate analysis. In this paper, we present a \textit{simple} framework that reduces dynamic regret minimization to switching regret minimization. As a result, we can derive dynamic regret bounds by using off-the-shelf algorithms with switching regret guarantees. The key idea of our reduction is to construct, for \textit{any} comparator sequence, an auxiliary random sequence that is unbiased at each round, with the controlled variance and a manageable number of switches. Combining this construction with suitable surrogate losses, we can decompose dynamic regret into the expected switching regret against the random sequence and its controlled vari
    
[^128]: 具有时序逻辑规范的高效贝叶斯自适应强化学习

    Efficient Bayes-Adaptive Reinforcement Learning with Temporal Logic Specifications

    [https://arxiv.org/abs/2609.20954](https://arxiv.org/abs/2609.20954)

    该论文提出一种将LTL任务的LDBA表示与环境的BAMDP表示同步的端到端模型强化学习算法，并结合BAMCP实现近似贝叶斯最优策略合成，在属性满足性和样本效率上优于传统非贝叶斯方法。

    

    我们提出了一种新颖的端到端基于模型的强化学习（RL）算法，用于在未知环境中根据给定的线性时序逻辑（LTL）规范（例如安全性或可达性）进行高效的策略合成。为此，LTL任务的极限确定性Büchi自动机（LDBA）表示与环境贝叶斯自适应马尔可夫决策过程（BAMDP）表示进行同步，这使我们能够利用通过贝叶斯强化学习实现的增强的探索-利用权衡，而非传统的非贝叶斯方法。我们进一步提出了一种新颖的贝叶斯自适应蒙特卡洛规划（BAMCP）算法，以便在同步的BAMDP结构中进行近似贝叶斯最优策略合成。一系列有限时域和无限时域任务实验表明，与传统方法相比，我们的方法在属性满足性和样本效率方面均表现出有效性。

    arXiv:2609.20954v1 Announce Type: new  Abstract: We present a novel end-to-end model-based Reinforcement Learning (RL) algorithm for efficient policy synthesis under given Linear Temporal Logic (LTL) specifications (e.g., safety or reachability) in unknown environments. To do so, a Limit-Deterministic B{\"u}chi Automaton (LDBA) representation of the LTL task is synchronised with a Bayes-Adaptive Markov Decision Process (BAMDP) representation of the environment, which allows us to leverage an enhanced exploration-exploitation trade-off that is achieved via Bayesian RL, as opposed to traditional non-Bayesian approaches. We further propose a novel Bayes-Adaptive Monte-Carlo Planning (BAMCP) algorithm to allow for approximate Bayes-optimal strategy synthesis in the synchronised BAMDP construct. A range of finite- and infinite-horizon task experiments demonstrate the effectiveness of our approach in terms of both property satisfaction and sample efficiency, when compared to traditional mode
    
[^129]: 当AI评审训练AI审稿人：科学判断坍缩及其缓解

    When AI Reviews Train AI Reviewers: Scientific-Judgment Collapse and Mitigation

    [https://arxiv.org/abs/2609.20942](https://arxiv.org/abs/2609.20942)

    该论文首次在受控实验中揭示了AI评审数据的递归训练会导致“科学判断坍缩”——评分分布压缩与语义多样性下降，并提出开源系统TrustReviewer来缓解这一问题。

    

    大语言模型（LLM）越来越多地参与科学评估，既作为自动化审稿人，也作为人类审稿人的助手。随着模型生成的评审进入公共数据和未来的训练语料库，AI同行评审可能变得递归化：后来的审稿人从早期模型产生的判断中学习。我们在受控环境中研究了这一反馈循环的一个环节。从Llama 3.1 8B出发，我们首先在2018–2023年的官方ICLR评审上微调出一个审稿人模型，然后使用官方评审与模型生成评审的系统性不同配比，在ICLR 2024数据上训练四个后继模型。我们的研究表明，引入合成评审会压缩评分分布，并降低同论文层面和语料库层面的语义多样性。我们将这种模式称为**科学判断坍缩**（scientific-judgment collapse）。为缓解这一失效模式，我们提出了**TrustReviewer**，一个用于生成同行评审的开源基于LLM的系统。

    arXiv:2609.20942v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly participate in scientific evaluation, both as automated reviewers and as assistants to human reviewers. As model-generated reviews enter public data and future training corpora, AI peer review can become recursive: later reviewers learn from judgments produced by earlier models. We study one step of this feedback loop in a controlled setting. Starting from Llama 3.1 8B, we first fine-tune a reviewer on official ICLR reviews from 2018--2023 and then train four successor models on ICLR 2024 data with systematically varied mixtures of official and model-generated reviews. Our study shows that introducing synthetic reviews compresses rating distributions and reduces both same-paper and corpus-level semantic diversity. We call this pattern $\textbf{scientific-judgment collapse}$.   To mitigate this failure mode, we introduce $\textbf{TrustReviewer}$, an open-source LLM-based system for generating peer
    
[^130]: 量子模型是否像大语言模型一样遵循缩放定律？

    Do Quantum Models Scale Like LLMs?

    [https://arxiv.org/abs/2609.20912](https://arxiv.org/abs/2609.20912)

    本文发现当里德堡原子量子系统处于临界点附近时，其Transformer模型的神经缩放定律与大语言模型最为相似，表明近临界量子数据的统计结构与自然语言最接近。

    

    在这项工作中，我们研究了RydbergGPT的神经缩放定律，该模型是一个自回归Transformer模型，其训练数据来自相互作用里德堡原子阵列的量子比特投影测量数据。已知当激光失谐参数变化时，该量子系统会表现出临界点的有限尺寸残余。我们发现，在临界点附近，Transformer损失作为训练数据集规模的函数可以很好地用带有损失下限修正的幂律来描述。然而，远离临界点时，幂律描述的质量显著下降。随后，我们使用一种经熵归一化、有限样本修正的互信息“两点”函数，比较了里德堡测量数据与自然语言语料库的统计结构。我们发现，临界点附近的两点函数统计特性最接近自然语言中观察到的统计特性，而远离临界点的其他量子比特构型则……

    arXiv:2609.20912v1 Announce Type: new  Abstract: In this work, we study the neural scaling laws of RydbergGPT, an autoregressive transformer model trained on qubit projective measurement data gathered from interacting Rydberg atom arrays. The quantum system is known to exhibit a finite-size remnant of a critical point as the laser detuning parameter is varied. We find that near the critical point the transformer loss as a function of training dataset size is well described by a power-law with a loss floor correction. However, away from criticality the quality of the power-law description is substantially reduced. We then compare the statistical structure of both Rydberg measurements and natural-language corpora using an entropy-normalised, finite sample corrected mutual information "two-point" function. We find that near-critical statistics of the two point functions are closest to those observed in natural-language, whilst other qubit configurations far from the critical point have tw
    
[^131]: 连续延迟记忆随机梯度下降与基于天体物理时间序列研究历史的连续时间强化学习

    Continuous Delayed-Memory Stochastic Gradient Descent and Continuous-Time Reinforcement Learning from History of Astrophysical Time Series Studies

    [https://arxiv.org/abs/2609.20906](https://arxiv.org/abs/2609.20906)

    本文提出了一种依赖过去迭代状态的连续延迟记忆随机梯度下降方法，以及一种无需求解HJB偏微分方程的连续时间策略梯度强化学习结构，用于建模类星体光变曲线等天体物理时间序列。

    

    类星体是宇宙中的发光天体，其表现出随机的亮度变化，这些变化编码了驱动它们的超大质量黑洞的信息。基于地面巡天数据的时间序列（即光变曲线）对这些变化进行建模是一项统计难题。本文回顾了历史上如何将随机微分方程（SDE）与神经网络参数化相结合来克服这一挑战。我们创建了连续延迟记忆随机梯度下降方法，该方法依赖于离散迭代过程的过去状态。我们在一些二维地形上进行了仿真，通过调整超参数，观察到与普通SGD（Vanilla SGD）相比，该方法具有更广泛的探索范围和更精确的收敛行为。此外，我们提出了一种具有连续时间策略梯度的强化学习结构，用于探索性策略而无需求解HJB偏微分方程，并且我们证明了其最优性条件

    arXiv:2609.20906v1 Announce Type: new  Abstract: Quasars are luminous objects in the universe that exhibit stochastic brightness variations encoding information about the supermassive black holes powering them, and modeling these variations from ground-based survey data time series, known as light curves, is a statistical challenge. This paper reviews how stochastic differential equations (SDEs) have been adapted with neural network parameterizations to overcome this challenge in history. We create the Continuous-Delayed-Memory Stochastic Gradient Descent which depend on the past state of the discrete iteration process. We performed the simulation on some 2-dimensional landscape and observed some wider-exploration and more precise convergent behavior compared to Vanilla SGD by adjusting hyperparameters. Besides, we proposed a reinforcement learning structure with continuous time policy gradients for exploratory policies without solving HJB PDE, and we show that its optimality condition
    
[^132]: Bio-MF：面向混合运动想象脑机接口的低延迟、高保真EEG到fNIRS跨模态生成

    Bio-MF: Low-Latency and High-Fidelity EEG-to-fNIRS Cross-Modal Generation for Hybrid Motor-Imagery Brain--Computer Interfaces

    [https://arxiv.org/abs/2609.20904](https://arxiv.org/abs/2609.20904)

    提出Bio-MF，一个无潜变量的单步MeanFlow框架，通过直接的信号空间x预测实现低延迟、高保真的EEG到fNIRS跨模态生成，克服了现有方法生成速度慢、依赖预训练以及单步生成保真度低的问题。

    

    结合EEG（脑电）和fNIRS（功能近红外光谱）的混合运动想象脑机接口（MI-BCI），通过利用互补的电生理和血流动力学信息，其性能可以超越仅使用EEG的系统。为了在配对EEG-fNIRS采集不可用或不便时获得这种混合信息，近期研究聚焦于EEG到fNIRS的跨模态生成。然而，现有方法仍然存在生成速度慢的问题，且通常需要预训练，限制了其在实时MI-BCI场景中的应用。尽管单步生成模型为低延迟合成提供了一条有吸引力的途径，但去除迭代精化过程会降低生成保真度并引入非生理性伪影。为解决这些问题，本文提出Bio-MF，一个无潜变量（latent-free）的单步MeanFlow框架，用于以EEG为条件的fNIRS生成。Bio-MF执行直接的信号空间x预测，并将该信号空间输出转换为MeanFlow速度（摘要至此处截断）。

    arXiv:2609.20904v1 Announce Type: cross  Abstract: Hybrid motor-imagery brain-computer interfaces (MI-BCIs) combining EEG and fNIRS can outperform EEG-only systems by exploiting complementary electrophysiological and hemodynamic information. To obtain such hybrid information when paired EEG-fNIRS acquisition is unavailable or inconvenient, recent studies have focused on EEG-to-fNIRS cross-modal generation. However, existing methods still suffer from slow generation and often require pretraining, limiting their use in real-time MI-BCI scenarios. Although one-step generative models offer an attractive route to low-latency synthesis, removing the iterative refinement process can reduce generation fidelity and introduce non-physiological artifacts. To address these problems, this paper proposes Bio-MF, a latent-free one-step MeanFlow framework for EEG-conditioned fNIRS generation. Bio-MF performs direct signal-space x-prediction, converts this signal-space output into MeanFlow velocity sup
    
[^133]: 极端分类：每类仅用一个训练样本即可战胜随机猜测

    Extreme classification: beating chance with one training example from each class

    [https://arxiv.org/abs/2609.20897](https://arxiv.org/abs/2609.20897)

    本文研究了一个极端的最小分类问题，证明当两个分布不同时，即使每类只有一个训练样本，也能通过基于最大均值差异（MMD）的随机化核规则等方法实现严格优于随机猜测的分类。

    

    arXiv:2609.20897v1 公告类型：cross 摘要：我们研究一个最小化的分类问题：给定来自两个未知分布 P 和 Q 的独立带标签观测样本 X~P 和 Z~Q，以及一个独立的目标 Y（以相等概率从 P 或 Q 中抽取），是否只要 P≠Q，就能对 Y 进行严格优于随机猜测的分类？最近邻规则对于每一对具有不同均值和共同正定协方差矩阵的多变量高斯分布都能成功，但即使对于实数线上的光滑密度，其表现也可能严格差于随机猜测。我们构造了一个固定的随机化核方法，其期望准确率恰好为 1/2+MMD_k²(P,Q)/4，并从可数个可测二元问题族出发，在可数生成的可测空间上构造出特征核。我们还证明了实数轴上的确定性顺序规则对于每一对不同的 Borel 概率测度都能战胜随机猜测。随后，一个可测编码方案将……

    arXiv:2609.20897v1 Announce Type: cross  Abstract: We study a minimal classification problem: Given independent labeled observations $X\sim P$ and $Z\sim Q$ from two unknown distributions $P,Q$, and given an independent target $Y$ drawn with equal probability from $P$ or $Q$, can one classify $Y$ strictly better than chance whenever $P\neq Q$? The one-nearest-neighbor rule succeeds for every pair of multivariate Gaussian distributions with distinct means and a common positive-definite covariance matrix but can perform strictly worse than chance even for smooth densities on the real line. We construct a fixed randomized kernel rule whose expected accuracy is exactly $1/2+\operatorname{MMD}_k^2(P,Q)/4$, and obtain characteristic kernels on countably generated measurable spaces from countable families of measurable binary questions. We also prove that a deterministic order rule on $\mathbb R$ beats chance for every pair of distinct Borel probability measures. A measurable encoding then gi
    
[^134]: 弹性阈值注意力：面向长上下文解码的学习型上下文稀疏化

    Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding

    [https://arxiv.org/abs/2609.20888](https://arxiv.org/abs/2609.20888)

    提出弹性阈值注意力（ETA），一种端到端可训练的架构，通过从查询表示中预测动态上下文阈值，在不牺牲稠密模型质量的前提下实现长上下文解码的硬件加速，解决了KV缓存带来的内存带宽瓶颈。

    

    在长上下文解码过程中，庞大的KV缓存会造成严重的内存带宽瓶颈。稀疏注意力方法通过选择性加载来缓解这一问题，但代价是：僵化的启发式规则会丢弃必要的上下文，导致质量下降。我们提出了弹性阈值注意力（ETA），这是一种端到端可训练的架构，能够在不牺牲稠密模型质量的情况下实现硬件加速的解码速度。ETA直接从查询表示中预测动态的、上下文相关的阈值，使模型能够为困难的检索或推理步骤分配类似稠密的上下文，同时剪枝常规token。为了在不发生表示坍塌的情况下从头学习这一策略，ETA在训练期间通过乘性抑制将低于阈值的logits压向零，而不是直接删除它们。针对这种平滑的均匀注意力底座进行训练，提供了一个分布式的概率储备库，使得定位……

    arXiv:2609.20888v1 Announce Type: new  Abstract: Massive KV caches can cause severe memory-bandwidth bottlenecks during long-context decoding. Sparse attention methods mitigate this via selective loading, but that comes at a cost: rigid heuristics drop necessary context, leading to quality degradation. We introduce \textbf{Elastic Threshold Attention (ETA)}, an end-to-end trainable architecture that achieves hardware-accelerated decoding speed without sacrificing dense model quality. ETA predicts dynamic, contextual thresholds directly from query representations, allowing the model to allocate dense-like context to difficult retrieval or reasoning steps while pruning routine tokens. To learn this policy from scratch without representation collapse, ETA \emph{multiplicatively suppresses} sub-threshold logits toward zero during training rather than deleting them. Training against this smooth uniform attention floor provides a distributed probability reservoir that \textbf{causes localize
    
[^135]: BI-Agent 与 BI-Bench：迈向端到端商业智能的自动化

    BI-Agent and BI-Bench: Towards Automating End-to-End Business Intelligence

    [https://arxiv.org/abs/2609.20886](https://arxiv.org/abs/2609.20886)

    该论文提出了 BI-Agent 智能体和首个端到端商业智能基准 BI-Bench，利用大语言模型自动完成数据准备与业务问题解答，实现 BI 工作流程的全面自动化。

    

    商业智能（BI）是企业决策的基石，被企业用户广泛应用于 Power BI 和 Tableau 等软件中。在传统的 BI 工作流程中，用户需要先进行数据准备：（1）识别相关表格，（2）执行数据转换，（3）建立连接关系，然后才能（4）回答业务问题。这些步骤可能复杂且耗时，使得 BI 的使用充满挑战。鉴于大型语言模型（LLM）在处理数据方面的强大能力，我们研究了其端到端回答 BI 问题的能力，无需用户手动执行繁琐的准备步骤。为此，我们从公开来源收集了大量真实世界的 BI 项目，并从真实用户的仪表板中手动提取（问题，标准答案）对。由此产生的基准测试 BI-Bench 是首个系统性研究 LLM 端到端……能力的基准（摘要内容在此处不完整）。

    arXiv:2609.20886v1 Announce Type: cross  Abstract: Business intelligence (BI) is a cornerstone of enterprise decision-making and is widely used by enterprise users in software such as Power BI and Tableau. In traditional BI workflows, users need to prepare data by (1) identifying relevant tables, (2) performing data transformations, and (3) building join relationships, before they can (4) answer their business questions. These steps can be complex and time-consuming, making BI challenging.   Given the strong capabilities of large language models (LLMs) in working with data, we study their ability to answer BI questions end-to-end, without requiring users to manually perform the tedious preparation steps. To do this, we harvest a large collection of real-world BI projects from public sources, and manually extract pairs of (questions, ground-truth answers) from real user dashboards. The resulting benchmark, BI-Bench, is the first benchmark to systematically study LLMs' ability on end-to-
    
[^136]: 高效分布学习的稀疏先验

    Sparse Priors for Efficient Distribution Learning

    [https://arxiv.org/abs/2609.20883](https://arxiv.org/abs/2609.20883)

    该论文提出“稀疏先验”类和“稀疏维度”的概念，证明在k-稀疏先验下分布学习的贝叶斯风险下界为Ω(√(k/n))并给出TV距离的匹配上界，从而突破了传统理论中依赖维度d的悲观样本复杂度保证。

    

    尽管当今生成式AI技术被广泛使用并取得成功，但从n个样本中学习支撑在d维空间上的分布，其理论保证会退化为O(n^{-1/Θ(d)})，尽管该界已被证明是极小极大最优的。我们假设现有的界过于悲观，因为光滑性假设不足以刻画实际应用中经常出现的分布结构。因此，我们引入了稀疏先验类，并定义“稀疏维度”作为先验在所有分布空间上的稀疏性度量。我们证明，在k-稀疏先验下，分布学习在常见距离度量下达到Ω(√(k/n))的贝叶斯风险下界，并在温和的附加假设下，给出了TV距离的匹配上界（在n、k渐近意义下相差对数项）。我们证明了分布学习与学习……之间的统计等价性（摘要在此处截断）。

    arXiv:2609.20883v1 Announce Type: new  Abstract: Despite the widespread use and success of generative AI techniques today, theoretical guarantees on learning a distribution supported in $d$ dimensions from $n$ samples degrade as $O(n^{-1/\Theta(d)})$, though shown to be minimax optimal. We hypothesize that present bounds are too pessimistic because smoothness assumptions are not enough to capture the structure of distributions that often appear in real applications. Consequently, we introduce the class of sparse priors and define the "Sparse Dimension" as a measure of sparsity of a prior over the space of all distributions. We show that distribution learning under a $k$-sparse prior achieves a Bayesian risk lower bound of $\Omega(\sqrt{k/n})$ under common distance metrics, and show a matching (up to logarithmic terms asymptotically in $n,k$) upper bound for the TV distance under mild additional assumptions. We show the statistical equivalence of distribution learning and learning to sa
    
[^137]: 反驳鸿沟：对最优性声明的两个组成部分同时进行认证

    The Refutation Gap: Certifying Both Halves of an Optimality Claim

    [https://arxiv.org/abs/2609.20873](https://arxiv.org/abs/2609.20873)

    该论文提出了“反驳鸿沟”这一概念，指出电路最小化领域中最优性声明的下界（不存在更小的程序）缺乏可认证的证明，而组合优化领域已有成熟的认证算法和伪布尔证明日志技术，应当被引入以同时认证最优性声明的上界和下界。

    

    综合流水线越来越多地不仅声称一个程序是正确的，还声称它是最优的。这样的声明包含两个部分，而这两部分的验证方式截然不同。上界，“存在一个大小为 m 的程序”，可以由一个可重新执行、可被证明与其规范等价、并附带机器可检查证书的制品来见证。而下界，“不存在大小为 m-1 的程序”，没有任何见证者，只能通过运行求解器直到它报告 UNSAT（不可满足）来完成。组合优化领域几十年来早已知晓这种不对称性，并已在很大程度上解决了这个问题：认证算法使这种不对称性显式化（McConnell 等，2011），伪布尔证明日志可以通过形式化验证的检查器端到端地认证最优性（Bogaerts 等，2023；Koops 等，2025）。然而这种规范实践尚未传达到电路最小化领域。我们将其称为“反驳鸿沟”：已发表的关于最小 XOR 电路的门数没有提供任何证书

    arXiv:2609.20873v1 Announce Type: cross  Abstract: Synthesis pipelines increasingly claim not just that a program is correct, but that it is optimal. Such a claim has two halves with radically different verification stories. The upper bound, "a program of size m exists", is witnessed by an artifact that can be re-executed, proved equivalent to its specification, and shipped with a machine-checked certificate. The lower bound, "no program of size m-1 exists", has no witness and is discharged by running a solver until it reports UNSAT. Combinatorial optimization has known this asymmetry for decades and has largely addressed it: certifying algorithms make it explicit (McConnell et al., 2011), and pseudo-Boolean proof logging can certify optimality end to end with a formally verified checker (Bogaerts et al., 2023; Koops et al., 2025). That discipline has not reached circuit minimization. We call this the refutation gap: published gate counts for minimal XOR circuits provide no certificate
    
[^138]: 基于物理信息神经网络的强分段硅望远镜自动校准

    Automated Physics-Informed Neural-Networks-Based Calibration of Highly Segmented Silicon Telescopes

    [https://arxiv.org/abs/2609.20868](https://arxiv.org/abs/2609.20868)

    该论文提出了一种基于神经网络的物理信息自动校准框架，将高粒度硅望远镜阵列的校准表述为全局优化问题，在两体运动学约束下通过最小化重建激发能宽度同时确定探测器增益与几何修正，仅依赖实验数据即可实现完全自动化校准。

    

    转移反应和多核子转移反应是探测核结构与反应动力学的重要工具，需要精确测定反应产物的种类、能量和出射角。现代硅望远镜阵列日益提高的粒度增强了实验能力，但也给探测器校准带来了挑战，因为传统的逐通道校准方法变得低效且难以扩展。在这项工作中，我们提出了一种完全自动化的、基于神经网络的物理信息校准框架，专为高度分段的硅探测器阵列设计。该方法将校准表述为一个全局优化问题，通过在两体运动学约束下最小化重建激发能的宽度，同时确定探测器的增益和几何修正。该方法仅依赖实验数据和成熟的物理（原理）……

    arXiv:2609.20868v1 Announce Type: cross  Abstract: Transfer and multi-nucleon transfer reactions are essential tools for probing nuclear structure and reaction dynamics, requiring precise determination of the identity, energy, and emission angles of reaction products. The increasing granularity of modern silicon telescope arrays enhances experimental capabilities but challenges detector calibration, as conventional channel-by-channel approaches become inefficient and difficult to scale.   In this work, we present a fully automated, physics-informed calibration framework based on neural networks, specifically designed for highly segmented silicon detector arrays. The method formulates calibration as a global optimization problem, in which detector gains and geometrical corrections are determined simultaneously by minimizing the width of the reconstructed excitation energy under two-body kinematics constraints. The approach relies exclusively on experimental data and well-established phy
    
[^139]: 使用带测试时自适应的深度算子网络从稀疏平面数据重建四维二尖瓣反流血流动力学

    Reconstruction of 4D Mitral Regurgitation Hemodynamics from Sparse Planar Data using Deep Operator Networks with Test-Time Adaptation

    [https://arxiv.org/abs/2609.20857](https://arxiv.org/abs/2609.20857)

    该研究提出一种结合测试时自适应的深度算子网络方法，能够仅从单个平面速度切片和边界压力轨迹等稀疏观测数据中重建四维二尖瓣反流血流动力学，为临床快速量化反流严重程度提供了新的技术路径。

    

    量化二尖瓣反流严重程度仍然受到临床血流汇聚方法假设的限制，而高保真度模拟和体积测速技术对于常规临床使用而言过于缓慢。我们研究学习到的解算子能否从体外实验实际提供的稀疏观测中重建瞬态三维跨瓣膜血流动力学：即单个平面速度切片和两条边界压力轨迹。深度算子网络在一个经过实验基准验证、涵盖十一个二尖瓣反流孔口模型的URANS数据库上进行预训练，学习从被掩码的双分量平面速度快照到周围体积场的映射，随后通过在未见过的目标病例的稀疏测量上进行微调来实现自适应。自适应能够可靠地纠正监督平面内的流动拓扑结构，重新定向预训练算子难以正确处理的强烈偏心射流……

    arXiv:2609.20857v1 Announce Type: cross  Abstract: Quantifying mitral regurgitation severity remains limited by the assumptions of clinical flow convergence methods, while high-fidelity simulation and volumetric velocimetry are too slow for routine use. We investigate whether a learned solution operator can reconstruct transient three-dimensional transvalvular hemodynamics from the sparse observation an in-vitro experiment actually provides: a single planar velocity slice and two boundary pressure traces. A Deep Operator Network is pretrained on an experimentally benchmarked URANS database spanning eleven mitral regurgitation orifice phantoms, learning a mapping from a masked two-component planar velocity snapshot to the surrounding volumetric field, and is subsequently adapted to unseen target cases by fine-tuning on their sparse measurements. Adaptation reliably corrects the flow topology within the supervised plane, reorienting a strongly eccentric jet that the pretrained operator p
    
[^140]: 奖励高效推理可提升推理模型在欠规范任务上的弃答能力

    Rewarding Efficient Reasoning Improves Abstention on Underspecified Tasks in Reasoning Models

    [https://arxiv.org/abs/2609.20846](https://arxiv.org/abs/2609.20846)

    该研究提出一种新颖的GRPO奖励机制以鼓励高效推理，使4B大型推理模型在信息不完整任务上的弃答能力平均提升12.8%，同时保留原有回答能力，并使其推理行为更接近人类。

    

    虽然现代大型推理模型（LRMs）在许多任务中都能出色地给出正确答案，但我们提供了进一步的证据表明，它们往往在一项关键能力上存在不足：知道何时应当放弃回答。我们通过将LRM的行为与人类研究的结果进行对比来分析这一差距，发现人类在无法回答的任务上的推理努力以可回答任务为上限，而LRMs则浪费计算资源，在无法回答的问题上生成比可回答问题更长的思维链（CoTs）。为克服这种低效性，我们从人类认知的资源理性视角获得启发，引入了一种新颖的GRPO奖励，鼓励模型高效地推理任务是否包含解决问题所需的全部信息。使用该奖励对多个4B LRM进行微调后，模型获得了类人的弃答性能提升（平均提升12.8%），同时保留了回答能力。

    arXiv:2609.20846v1 Announce Type: new  Abstract: While modern large reasoning models (LRMs) excel at providing correct answers in many tasks, we provide additional evidence for the observation that they often struggle with a critical capability: knowing when to abstain from answering. We analyze this gap by comparing LRM behavior to results from a human study, revealing that human reasoning effort on unanswerable tasks is upper-bounded by answerable tasks, whereas LRMs waste computational resources by generating longer Chains of Thought (CoTs) on unanswerable than on answerable prompts. To overcome this inefficiency, we take inspiration from a resource-rational perspective on human cognition and introduce a novel GRPO reward that encourages efficient reasoning about whether the task contains all the information needed to solve it. Fine-tuning several 4B LRMs with this reward leads to human-like abstention performance gains (+12.8% on average) while retaining answering capabilities and 
    
[^141]: 边写边少读：流式多模态解码器的闭式带宽调节旋钮

    Reading Less While Writing: A Closed-Form Bandwidth Dial for Streaming Multimodal Decoders

    [https://arxiv.org/abs/2609.20845](https://arxiv.org/abs/2609.20845)

    该论文提出ZENDAYA，用单一连续参数γ的闭式调度取代流式多模态解码器中固定的wait-k规则，将离线解码与实时流式解码统一为同一模型家族的两个端点，并以解析形式Ē(γ) ≈ 1/(1+γ)精确控制每个输出词消耗的源内容比例，实现可解释的延迟与带宽调节。

    

    传统上将视频或音频转换为文本的解码器在输出第一个词之前会消耗整个输入。在离线场景下，这只是比任务所需的更多；但在实时场景下这是不可能的，因为字幕不能等到比赛结束才出现。流式系统通常采用固定规则，如wait-k，它在每个输出词之前等待相同数量的输入token，而不考虑输入的长度或节奏。我们用ZENDAYA取代固定偏移量，这是一个由单一连续参数γ控制的调度方案。它使可见的源前缀成为生成进度的闭式函数，并根据输入自身预测的长度进行缩放，因此普通的离线解码器和实时流式解码器成为同一个家族的两个端点，而不是相互分离的模型。同一个标量以闭式形式确定了每个输出词平均消耗的源内容比例，Ē(γ) ≈ 1/(1+γ)，使其同时成为一个延迟调节旋钮和一个可解释的带宽控制手段。

    arXiv:2609.20845v1 Announce Type: new  Abstract: A decoder that turns video or audio into text conventionally consumes the entire input before emitting a word. Offline this is merely more than the task requires; live it is impossible, since a caption cannot wait for a match to end. Streaming systems bolt on a fixed rule such as wait-$k$, which waits for the same number of input tokens before every word, regardless of the input's length or pace.   We replace the fixed offset with ZENDAYA, a schedule governed by a single continuous parameter $\gamma$. It makes the visible source prefix a closed-form function of generation progress, scaled by the input's own predicted length, so an ordinary offline decoder and a real-time streaming decoder become two endpoints of one family rather than separate models. The same scalar fixes, in closed form, the mean fraction of source consumed per emitted word, $\bar{E}(\gamma) \approx 1/(1+\gamma)$, making it at once a latency dial and an interpretable b
    
[^142]: VISPATH：面向多模态知识图谱问答的视觉意图引导路径推理

    VISPATH: Visual-Intent-Guided Path Reasoning for Multimodal Knowledge Graph Question Answering

    [https://arxiv.org/abs/2609.20843](https://arxiv.org/abs/2609.20843)

    提出VISPATH框架，通过视觉意图引导的路径推理，使多模态线索能够在多跳推理的中间步骤中持续发挥作用，突破了现有方法仅将多模态信息用于起始实体定位的局限，提升了多模态知识图谱问答的性能。

    

    知识图谱问答（KGQA）使模型能够通过结构化图推理来回答自然语言问题，并在众多基准测试和应用中取得了显著进展。近年来，多模态知识图谱问答（MM-KGQA）受到越来越多的关注，因为许多问题需要联合使用多模态输入和知识图谱证据。然而，现有的MM-KGQA方法通常仅将多模态信息用于起始实体定位或证据检索，此后多跳推理便退化为纯文本的图搜索。因此，这些方法无法利用在中间跳步中变得重要的多模态线索。为了解决这一局限，我们提出了VISPATH，一个用于多模态知识图谱问答的视觉意图引导路径推理框架。VISPATH首先通过结合多模态定位与图结构线索来识别可靠的起始实体，然后通过重新计算特定跳步的（摘要内容在此处截断）

    arXiv:2609.20843v1 Announce Type: new  Abstract: Knowledge graph question answering (KGQA) enables models to answer natural-language questions through structured graph reasoning and has achieved substantial progress across many benchmarks and applications. Recently, multimodal KGQA (MM-KGQA) has attracted increasing attention because many questions require jointly using multimodal inputs and KG evidence. However, existing MM-KGQA methods typically use multimodal information only for starting entity grounding or evidence retrieval, after which multi-hop reasoning degenerates into text-only graph search. As a result, they cannot exploit multimodal cues that become important at intermediate hops. To address this limitation, we propose VISPATH, a visual-intent-guided path reasoning framework for MM-KGQA. VISPATH first identifies a reliable starting entity by combining multimodal grounding with graph-structural cues. It then performs intent-guided path discovery by recomputing hop-specific 
    
[^143]: PhysioBench：一个统一的生理信号问答基准

    PhysioBench: A Unified Benchmark for Physiological Signal Question Answering

    [https://arxiv.org/abs/2609.20836](https://arxiv.org/abs/2609.20836)

    该论文提出了PhysioBench——首个统一的生理信号问答基准，将22个公开数据集的标注整合为覆盖30个任务的6140万个问题，并系统评估了21个代表性模型，揭示现有模型在跨模态生理信号指令遵循方面均存在不足。

    

    生理信号支持多样化的临床与监测任务，然而现有的生理信号基础模型通常需要针对每个任务进行专门的适配。自然语言为指定不同预测目标提供了通用接口，但当前模型在生理信号各模态上遵循此类指令的能力仍未得到充分评估。为填补这一空白，我们提出了PhysioBench，一个统一的生理信号问答基准。PhysioBench将来自22个公开数据集的标注整合为涵盖30个任务的6140万个问题，每个问答对都基于一个信号片段，并可追溯至其原始标注。我们在三种互补的设置下评估了21个代表性模型，包括大语言模型、视觉-语言模型、时间序列语言模型以及生理信号基础模型。结果显示，没有任何一个模型（摘要在此处截断）

    arXiv:2609.20836v1 Announce Type: new  Abstract: Physiological signals support diverse clinical and monitoring tasks, yet existing physiological signal foundation models typically require task-specific adaptation for each task. Natural language provides a common interface for specifying different prediction objectives, but the ability of current models to follow such instructions across physiological signal modalities remains insufficiently evaluated. To address this gap, we introduce PhysioBench, a unified benchmark for physiological signal question answering. PhysioBench harmonizes annotations from 22 public datasets into 61.4 million questions across 30 tasks. Each question-answer pair is grounded in a signal segment and traceable to its source annotation. We evaluate 21 representative models, including large language models, vision-language models, time-series language models, and physiological signal foundation models under three complementary settings. The results show that none 
    
[^144]: 递归语言模型实现分布外泛化

    Recursive Language Models Generalize Out of Domain

    [https://arxiv.org/abs/2609.20831](https://arxiv.org/abs/2609.20831)

    递归语言模型通过在隔离上下文中解决子任务，排除了标准CoT依赖子任务外上下文捷径的失败模式，从而在分布外泛化上表现更优，这表明实现真正推理仅覆盖正确规则是不够的。

    

    我们研究何时限制语言模型能看到的内容能够改善学习。我们将标准的思维链与递归语言模型进行比较：前者是读取完整轨迹的更通用学习器，后者通过在隔离的上下文中解决每个子任务来限制自身。在分布内，这种通用性是免费的：CoT可以高效地模拟递归规则，因此IID泛化保证仅改变一个常数因子，递归并未提供太多优势。但在分布外，CoT可以通过依赖当前子任务之外的上下文来拟合训练，即一种一旦这些标记发生变化就会失效的捷径；而递归上下文隔离排除了这种失败模式。尽管CoT的假设类仍然覆盖递归规则，但简单性偏差会选择捷径而非真相。因此，要超越分布内精度并实现真正的推理，仅仅覆盖正确的规则是不够的；这与经典学习理论形成对比。

    arXiv:2609.20831v1 Announce Type: new  Abstract: We study when limiting what a language model can see improves learning. We compare standard CoT, the more general learner that reads the full trace, with recursive language models, which restricts itself by solving each subtask in an isolated context. In-distribution, this generality comes for free: CoT can efficiently simulate the recursive rule, so the IID generalization guarantee changes only by a constant factor, and recursion does not offer much. But out of domain, CoT can fit training by relying on context outside the current subtask, i.e. a shortcut that breaks once those tokens change; recursive context isolation rules out this failure mode. Even though CoT's class still covers the recursive rule, simplicity bias picks the shortcut over the truth. Thus, to go beyond distributional accuracy and truly reason, covering the right rule is not enough; this contrasts with classical learning theory.
    
[^145]: 超越词错误率：口音英语会话语音识别中的实体与不流利召回

    Beyond WER: Entity and Disfluency Recall in Accented Conversational ASR

    [https://arxiv.org/abs/2609.20828](https://arxiv.org/abs/2609.20828)

    该论文提出三阶段流水线，通过SQL启发式数据筛选和区域LoRA微调，将带口音英语会话语音识别的实体召回率和填充词召回率大幅提升，同时以十分之一的参数量媲美300亿参数模型。

    

    针对词错误率（WER）优化的语音识别系统在处理带口音的英语会话时，常常遗漏命名实体和填充词，而这两者对于语言学习反馈至关重要。我们针对来自印度、印度尼西亚和拉丁美洲的说话人提出了一种三阶段流水线：（1）使用启发式SQL过滤器筛选实体丰富的训练数据，其实体密度是随机采样的2.8倍；（2）在Qwen2.5-Omni-3B上微调的区域LoRA适配器，可在单次前向传播中同时生成逐字转录和修正转录；（3）六类错误分类体系，由基于大语言模型的评判器验证（一致率83.8%，基于210条人工标注样本）。该流水线在6000条测试语句上实现了80-85%的实体召回率（从53-55%提升）、76-86%的填充词召回率（从不足5%提升）以及6-10%的词错误率，在实体召回方面超越Whisper和某商业语音识别系统，同时以十分之一的参数量达到零样本300亿参数模型的水平。配对自举检验证实，仅数据筛选就能带来2.8-（原文在此截断）

    arXiv:2609.20828v1 Announce Type: new  Abstract: ASR systems optimised for Word Error Rate (WER) often miss named entities and filled pauses in accented conversational English, both critical for language-learning feedback. We present a three-stage pipeline for speakers from India, Indonesia, and Latin America: (1) heuristic SQL filters curating entity-rich training data at 2.8x the entity density of random sampling, (2) regional LoRA adapters fine-tuned on Qwen2.5-Omni-3B producing both verbatim and corrected transcripts in a single forward pass, and (3) a six-category error taxonomy validated by an LLM-based judge (83.8% agreement, 210 human-labelled samples). The pipeline achieves 80-85% entity recall (up from 53-55%), 76-86% filler recall (up from <5%), and 6-10% WER across 6k test utterances, outperforming Whisper and a commercial ASR on entity recall while matching a zero-shot 30B model with 10x fewer parameters. Paired bootstrap tests confirm that curation alone accounts for 2.8-
    
[^146]: TALON：一种面向放射学报告生成的时序感知纵向框架

    TALON: A Temporally Aware Longitudinal Framework for Radiology Report Generation

    [https://arxiv.org/abs/2609.20826](https://arxiv.org/abs/2609.20826)

    提出TALON框架，通过双通道时序融合模块自适应整合可变长度的患者病史，分别捕捉持续性发现与间期变化，从而提升放射学报告生成中的纵向比较能力。

    

    当前的放射学报告生成（RRG）模型通常基于单次检查或仅最近一次的既往检查来生成描述性报告，这限制了模型进行准确且有意义的纵向比较以及检测细微间期变化的能力。尽管近期的一些方法已开始纳入多次既往检查，但它们通常聚合固定长度的病史，而没有在融合之前显式建模每次既往检查的角色相关性。为解决这一问题，我们提出了TALON，一个时序感知的纵向RRG框架，能够自适应地整合可变长度的患者病史。其底层的双通道时序融合模块（DCTFM）通过互补的相似性通道和变化通道，将当前检查与每次既往检查分别进行比较，以捕捉持续性发现和间期变化。专门设计的通道特定注意力机制估计……

    arXiv:2609.20826v1 Announce Type: new  Abstract: Current radiology report generation (RRG) models usually produce descriptive reports based on a single examination or only the most recent prior examination, limiting their ability to perform accurate and meaningful longitudinal comparisons and detect subtle interval changes. Although recent approaches have begun to incorporate multiple prior examinations, they usually aggregate a fixed-length history without explicitly modeling the role-dependent relevance of each prior examination before fusion. To address this, we propose TALON, a Temporally Aware LONgitudinal RRG framework that adaptively integrates variable-length patient histories. The underlying Dual-Channel Temporal Fusion Module (DCTFM) compares the current examination with each prior examination through complementary similarity and change channels to capture persistent findings and interval changes, respectively. The specially designed channel-specific attention estimates the r
    
[^147]: HERMES：基于对比感知的知识图谱推理，从临床笔记中进行患者结局预测

    HERMES: Contrast-Aware Knowledge Graph Reasoning from Clinical Notes for Patient Outcome Prediction

    [https://arxiv.org/abs/2609.20825](https://arxiv.org/abs/2609.20825)

    HERMES通过大语言模型从临床笔记中构建个性化知识图谱，利用对比逻辑建模显式捕捉时间动态与治疗失败等对比信息，并结合图注意力网络学习患者表示，从而在不依赖结构化数据的情况下提升患者结局预测效果。

    

    临床预测模型通常依赖于结构化的电子健康记录（EHR）数据，例如时间序列数据和手术操作编码。尽管近期的一些方法已开始利用非结构化的临床笔记，但它们通常将其编码为扁平的序列，这可能丢失临床叙述中存在的显式关系结构和时间结构。为此，我们提出了HERMES，一个完全基于临床文本运行、同时保留临床关系的图框架。该方法建立在两个关键思想之上：首先，通过大语言模型引导的抽取从临床笔记中构建个性化知识图谱（KG），并采用对比逻辑建模显式捕捉时间动态、治疗失败以及结局变化；其次，通过图注意力网络（GAT）在知识图谱上进行图学习，从而合成患者表示。实验在MIMIC-III和MIMIC-IV数据集上针对院内死亡率和30天（摘要内容不完整）……进行评估。

    arXiv:2609.20825v1 Announce Type: new  Abstract: Clinical predictive models often rely on structured Electronic Health Record data, such as time-series and procedure codes. While recent approaches have begun leveraging unstructured clinical notes, they typically encode them as flat sequences, which may lose explicit relational and temporal structure present in clinical narratives. In response, we propose HERMES, a graph-based framework that operates exclusively on clinical text while preserving clinical relationships. This approach builds on two key ideas. First, personalized Knowledge Graphs (KGs) are constructed through Large-Language-Model-guided extraction from clinical notes with Contrastive Logic Modeling that explicitly captures temporal dynamics and treatment failures and changes in outcomes. Second, a Graph Attention Network synthesizes patient representations through graph-based learning over the KGs. Experiments on MIMIC-III and MIMIC-IV for in-hospital mortality and 30-day 
    
[^148]: 扩散语言模型中的并行性、临界窗口与分离性

    Parallelism, critical windows, and separations among diffusion language models

    [https://arxiv.org/abs/2609.20539](https://arxiv.org/abs/2609.20539)

    本文首次对掩码扩散、均匀扩散与高斯扩散三类主流扩散语言模型的并行生成能力进行了细粒度理论比较，证明均匀扩散和高斯扩散同样能以前向传播次数与分布对偶总相关（可远小于上下文长度）成比例的方式完成采样，而此前仅掩码扩散具备这一性质。

    

    扩散大语言模型的一个广受欢迎的卖点在于其并行性能力：即能够以远高于自回归模型的效率生成文本序列，后者每个 token 都需要一次前向传播。然而，在众多相互竞争的 dLLM 范式之中——从掩码扩散到均匀扩散再到高斯扩散——对于这些不同方案在并行性方面如何比较，原理性的理解仍然有限。在本工作中，我们对这三种主流方法的并行性能力开展了细粒度的比较研究，并证明了以下结果：均匀扩散和高斯扩散可以在前向传播次数随底层分布的对偶总相关缩放的情况下完成采样，对偶总相关是一种内在复杂度的度量，其数值可以远小于上下文长度；而在此之前，只有掩码扩散被证明能够实现这一点。此外，对于某一族随机经验测度，我们表明（摘要截断）……

    arXiv:2609.20539v1 Announce Type: new  Abstract: A popular selling point of diffusion large language models (dLLMs) is their capacity for parallelism: the ability to generate sequences of text far more efficiently than autoregressive models, which require one forward pass per token. Yet among the many competing paradigms for dLLMs, from masked to uniform to Gaussian diffusion, principled understanding of how these different proposals compare in parallelism remains limited. In this work, we initiate a fine-grained comparison of the capacity for parallelism among these three leading approaches and prove the following:   - Uniform and Gaussian diffusion can sample in a number of forward passes which scales with the dual total correlation of the underlying distribution, a measure of intrinsic complexity which can be much smaller than the context length. Previously, it was only known how to achieve this using masked diffusion.   - For a certain family of random empirical measures, we show t
    
[^149]: 小到足以知晓一切：作为延迟泛化科学仪器的全可枚举Transformer

    Small Enough to Know Everything: The Fully-Enumerable Transformer as an Instrument for the Science of Delayed Generalization

    [https://arxiv.org/abs/2609.20166](https://arxiv.org/abs/2609.20166)

    本文提出将全可枚举任务上的微型Transformer作为研究延迟泛化（grokking）现象的科学仪器，其提供精确泛化上限、任务手术、全权重直接观察和生存时间统计四种独特能力，并通过预注册守恒研究验证小规模下发现的规律具有可迁移性。

    

    在完全可枚举任务上训练的微型Transformer在grokking（顿悟）现象研究中占据着一个特殊的位置：每个输入都可以被评估，每个泛化上限都可以被精确计算，数百个随机种子的实验只需几分钟。我们认为这一研究机制是一种科学仪器，具备四种近似设置无法提供的能力：(a) 精确且可证伪的泛化上限；(b) 任务手术，即可在可证明固定其他所有变量的同时操纵单一结构变量；(c) 直接观察每一个权重；(d) 对大量随机种子进行生存时间统计，从而将“不发生grokking”重新表述为删失观测。显而易见的质疑是：在10^4参数量级下刻画的规律在更大规模上可能毫无意义。我们通过一项预注册的守恒研究来回应这一质疑：在12K参数下确立的三条任务侧规律——可恢复性上限定律、角色冲突延迟定律和权重衰减响应定律——在一个理（摘要原文截断）

    arXiv:2609.20166v1 Announce Type: new  Abstract: Tiny transformers trained on fully-enumerable tasks occupy an unusual position in the study of grokking: every input can be evaluated, every generalization ceiling can be computed exactly, and hundreds of seeds cost minutes. We argue this regime is a scientific instrument with four capabilities that approximate settings cannot offer: (a) exact, falsifiable generalization ceilings; (b) task surgery that manipulates one structural variable while provably fixing all others; (c) direct observation of every weight; and (d) survival-time statistics over many seeds that recast "does not grok" as a censored observation. The obvious objection is that laws characterized at 10^4 parameters may not mean anything beyond them. We answer it with a preregistered conservation study: three task-side laws established at 12K parameters -- a recoverability-ceiling law, a role-conflict delay law, and a weight-decay response law -- are re-measured under an ide
    
[^150]: 采样揭示风格：大语言模型激活中提示条件风格轴的无监督、免训练发现

    Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations

    [https://arxiv.org/abs/2609.19150](https://arxiv.org/abs/2609.19150)

    该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。

    

    大语言模型（LLM）在其隐藏激活中编码了丰富的风格结构，但要发现对于给定提示哪些风格维度是显著的，通常需要监督式对比数据。我们提出了一种无需训练、提示条件化的替代方法：我们对单个提示在较高温度下反复采样补全结果，对汇集的隐藏激活应用主成分分析（PCA），并根据极性生成结果自动标记所得的风格轴。我们在一项两阶段研究中，将发现的轴与245个人类风格标注进行了验证。在我们最强的模型（Qwen-3.5-4B-Instruct）上，前两个轴以72.8%的精确率和43.6%的宏召回率匹配用户自发请求的风格维度，75.6%的有效性评分认为这些轴的极性生成结果与其标签相符，标注者之间的相邻一致性达90.9%。风格轴的可发现性强烈依赖于模型本身：两个Qwen模型（原文摘要在此处截断）

    arXiv:2609.19150v1 Announce Type: new  Abstract: Large language models (LLMs) encode rich stylistic structure in their hidden activations, but discovering which stylistic dimensions are salient for a given prompt typically requires supervised contrastive data. We present a training-free, prompt-conditional alternative: we repeatedly sample completions of a single prompt at elevated temperature, apply Principal Component Analysis (PCA) to the pooled hidden activations, and label the resulting axes automatically from the pole generations. We validate the discovered axes against 245 human-elicited stylistic annotations in a two-phase study. On our strongest model (Qwen-3.5-4B-Instruct), the top two axes match spontaneously requested human dimensions with 72.8% precision and 43.6% macro-recall, and 75.6% of validity ratings judge the axes' polar generations accurate to their labels, with 90.9% adjacent inter-annotator agreement. Discoverability is strongly model-dependent: both Qwen models
    
[^151]: 基于信息瓶颈的训练自适应卷积稀疏编码实现鲁棒视觉表示

    Training-Adaptive Convolutional Sparse Coding via Information Bottleneck for Robust Visual Representation

    [https://arxiv.org/abs/2609.19122](https://arxiv.org/abs/2609.19122)

    该论文提出一种训练自适应的卷积稀疏编码框架，通过展开FISTA算法并将稀疏系数作为可学习变量，从信息瓶颈视角自动权衡信息保留与压缩，从而获得更鲁棒的视觉表示。

    

    视觉信号需要紧凑且充分的表示，以实现鲁棒的下游预测。卷积稀疏编码（CSC）提供了一种显式机制，能够在保留信号内容的同时抑制冗余成分，但其稀疏系数通常是固定的且需要手动选择。我们提出了一种用于鲁棒视觉信号表示的训练自适应卷积稀疏编码框架。具体而言，我们使用快速迭代收缩阈值算法（FISTA）展开CSC优化过程，并将稀疏系数视为可微分变量，与网络参数进行联合学习。从信息瓶颈的角度来看，该系数控制着信息保留与压缩之间的权衡：稀疏项促进紧凑的表示，而重建项与任务损失共同保留与任务相关的信号内容。我们进一步引入了一种无标签的后处理方法（摘要在此处被截断）

    arXiv:2609.19122v2 Announce Type: cross  Abstract: Visual signals require compact yet sufficient representations for robust downstream prediction. Convolutional sparse coding (CSC) provides an explicit mechanism for suppressing redundant components while preserving signal content, but its sparsity coefficient is typically fixed and manually selected. We propose a training-adaptive convolutional sparse coding framework for robust visual signal representation. Specifically, we unfold the CSC optimization with the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) and treat the sparsity coefficient as a differentiable variable jointly learned with the network parameters. From the information bottleneck perspective, this coefficient controls the trade-off between information retention and compression: the sparsity term promotes compact representations, while the reconstruction term together with task loss preserves task-relevant signal content. We further introduce a label-free post-t
    
[^152]: 迈向可组合的网络数字孪生：基于子图的时延预测研究

    Toward Composable Network Digital Twins: A Subgraph-Based Latency Prediction Study

    [https://arxiv.org/abs/2609.18704](https://arxiv.org/abs/2609.18704)

    本文提出一种可组合的网络数字孪生方法，将网络分解为可重用的子图单元孪生，并通过轻量级组合器聚合来预测每条路由的端到端时延，解决了现有方法单体化、难以适应拓扑和流量变化的问题。

    

    现代网络必须支持不断变化的拓扑、配置和性能目标，这促使人们对快速且可靠的性能估计方法产生需求。网络数字孪生（NDT）能够在此类网络场景中支持性能估计的假设分析，然而，现有的基于机器学习的NDT方法通常依赖于完整的拓扑表示，这种表示本质上是单体式的，在网络拓扑或流量发生变化时缺乏可重用性。本文提出了一种可组合的NDT方法，将网络分解为子图，并用可重用的单元孪生来表示，这些单元孪生能够捕获子图的结构、配置和流量行为。一个轻量级的组合器聚合单元孪生的组合，从而创建出能够预测穿越整个拓扑的每条路由端到端时延的NDT。该方法在受控合成拓扑与多样化流量场景、真实世界的Topology Zoo拓扑以及一个公开的NDT挑战数据集上进行了评估。

    arXiv:2609.18704v1 Announce Type: cross  Abstract: Modern networks must support changing topologies, configurations, and performance objectives, motivating fast and reliable performance estimation. Network digital twins (NDTs) enable what-if analysis for performance estimation in such network scenarios, however, existing machine learning-based NDT approaches often rely on entire topology representations, which are inherently monolithic and lack reusability under topological or traffic changes in the network. This paper introduces a composable NDT approach that decomposes networks into subgraphs represented by reusable unit twins that capture subgraph structure, configuration and traffic behaviours. A lightweight composer aggregates unit twin combinations to create NDTs that predict per-route end-to-end latency through an overall topology. Evaluation across controlled synthetic topologies and diverse traffic scenarios, real-world Topology Zoo topologies, and a public NDT challenge datas
    
[^153]: 通过潜在神经符号推理解耦长期记忆

    Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning

    [https://arxiv.org/abs/2609.18461](https://arxiv.org/abs/2609.18461)

    提出LGM神经符号框架，利用稀疏自编码器将长期记忆解耦到连续潜在空间，根据每个查询动态构建潜在图，从而克服现有静态图记忆框架和平面检索方法无法捕捉上下文相关关系的问题。

    

    个性化智能体需要对长期历史交互进行推理，以同时推断显式偏好和隐式行为证据。早期的平面检索方法独立地对记忆片段进行评分，忽略了分布式信息；而当前的结构化记忆框架依赖于与查询无关的静态图，无法捕捉上下文相关的关系。至关重要的是，原始文本记忆本质上是纠缠且嘈杂的，使得细粒度个性化和跨会话推理在计算上变得难以实现。为此，我们提出了LGM，这是一种新颖的神经符号框架，将长期记忆解耦转移到连续潜在空间中。具体而言， 我们设计了一种基于稀疏自编码器的定制化潜在图构建方法，而非持久化固定图，它根据每个查询将历史交互映射为潜在记忆节点，并将记忆轨迹解耦为稀疏概念……

    arXiv:2609.18461v1 Announce Type: new  Abstract: Personalized agents are required to reason over long-term history interactions to infer both explicit preferences and implicit behavioral evidence. While early flat retrieval methods score memory fragments independently and neglect the distributed information, current structured memory frameworks rely on query-agnostic static graphs that fail to capture the context-dependent relations. Crucially, raw textual memories are inherently entangled and noisy, making fine-grained personalization and cross-session reasoning computationally prohibitive. To this end, we present LGM, a novel neuro-symbolic framework that shifts long-term memory disentanglement into a continuous latent space. Specifically, (i) instead of persisting fixed graphs, we design a tailored latent graph construction with a sparse autoencoder. Subject to each query, it maps historical interactions into latent memory nodes and disentangles the memory traces into sparse concept
    
[^154]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^155]: QuanText：在文本数据共享中保护数据集级秘密

    QuanText: Protecting Dataset-Level Secrets in Textual Data Sharing

    [https://arxiv.org/abs/2609.17995](https://arxiv.org/abs/2609.17995)

    QuanText提出了一种无需训练、与大语言模型无关的文本随机量化数据发布机制，能够在保护文本数据集中数据集级全局秘密（如敏感属性比例）的同时保持数据效用，弥补了差分隐私对聚合属性保护不足的缺陷。

    

    自然语言数据集支持许多下游应用和研究工作，但发布文本可能会暴露底层数据源的敏感全局属性，例如与特定性别、诊断或政治立场相关的记录比例。现有工作主要集中于恢复此类全局属性的属性推断攻击，而保护这些数据集级秘密的防御方法仍然有限。差分隐私虽然在保护个体记录方面有效，但对聚合属性只能提供较弱的保护。我们提出了文本随机量化，这是一种无需训练且与大语言模型无关的数据发布机制，能够在保持数据效用的同时保护文本数据集中的全局秘密。给定一个数据集级秘密，例如具有特定诊断的记录比例，以及需要保持其效用的属性，例如主题……

    arXiv:2609.17995v1 Announce Type: new  Abstract: Natural-language datasets support many downstream applications and research studies, but releasing text can reveal sensitive global properties of the underlying data source, such as the proportion of records associated with a particular gender, diagnosis, or political stance. Existing work has largely focused on property inference attacks that recover such global properties, while defenses for protecting these dataset-level secrets remain limited. Differential privacy, although effective for protecting individual records, provides only weak protection for aggregate properties. We propose Randomized Quantization for Text (QuanText), a training-free and large-language-model-agnostic data release mechanism that protects global secrets in textual datasets while preserving data utility. Given a dataset-level secret, such as the proportion of records with a particular diagnosis, and attributes whose utility should be preserved, such as topic a
    
[^156]: 用于可审计次日野火蔓延预测的模块化深度学习机制

    Modular Deep Learning Mechanisms for Auditable Next-Day Wildfire Spread Prediction

    [https://arxiv.org/abs/2609.17763](https://arxiv.org/abs/2609.17763)

    本文提出三种模块化深度学习增强机制（风坡条件化注意力偏置、物理特征检索增强的输出校正和火点条件化双流门控），实现了可审计、可解释的次日野火蔓延预测。

    

    次日野火预测要求模型的预测结果能够与其计算过程中所使用的假设和历史证据一起进行评估。尽管深度学习可以从遥感数据中学习空间模式，但仅凭预测性能并不能确立物理保真度或业务可信度。本研究针对次日活跃火点预测研究了三种模块化增强机制：风与坡度条件化的注意力偏置、物理特征检索增强的输出校正，以及火点条件化的双流门控。注意力偏置能够揭示预设的方向性偏好，而检索模块利用九维环境与火状态描述符选择历史图块，并对冻结模型的logits应用学习到的校正。这些模块在Next Day Wildfire Spread基准数据集上跨五个骨干网络进行评估，采用分阶段消融实验、方向性审计、检索扰动等方法……

    arXiv:2609.17763v1 Announce Type: new  Abstract: Next-day wildfire prediction requires models whose forecasts can be evaluated alongside the assumptions and historical evidence used in their computation. Although deep learning can learn spatial patterns from remote-sensing data, predictive performance alone does not establish physical fidelity or operational trustworthiness. This study investigates three modular augmentations for next-day active-fire prediction: wind- and slope-conditioned attention biases, physics-feature retrieval-augmented output correction, and fire conditioned dual-stream gating. The attention biases expose prescribed directional preferences, while the retrieval module selects historical tiles using a nine-dimensional environmental and fire-state descriptor and applies a learned correction to a frozen model's logits. The modules are evaluated across five backbones on the Next Day Wildfire Spread benchmark, using staged ablations, directional audits, retrieval pert
    
[^157]: Fathom：面向卸载KV缓存稀疏解码的逐查询读取深度

    Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches

    [https://arxiv.org/abs/2609.17652](https://arxiv.org/abs/2609.17652)

    Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。

    

    当智能体会话运行至百万token且同时驻留多个会话时，KV缓存及其排序索引存放在主机内存中，而针对top-k步骤对所有n个键进行排序的扫描成为限制解码速度的流量瓶颈。我们提出Fathom，一种由每个查询自主决定读取每个键通道多少比特的键扫描方法。4位K缓存以通道优先的方式存储为比特平面，因此t个平面的前缀恰好构成该通道的t位量化器，查询通过对方差加权通道重要性进行逆注水来分配其比特预算。在Qwen3-8B上处理一百万token时，解码步骤的GPU时间比Double Sparsity、Loki和SparQ r=32的136位扫描快1.67倍；在与SparQ的68位读取（r=16）相同的GPU时间内，Fathom读取的字节数减少18%，且在七个模型与上下文设置中的六个上注意力误差更低。在RULER风格的任务上，每次逐token扫描均与精确top-k解码结果相匹配。

    arXiv:2609.17652v1 Announce Type: cross  Abstract: When agentic sessions run to a million tokens with many sessions resident at once, the KV cache and the index that ranks it live in host memory, and the scan that ranks all n keys for a top-k step becomes the traffic that bounds decoding. We present Fathom, a key scan in which each query decides how many bits of each key channel to read. The 4-bit K cache is stored channel-major as bit planes, so a prefix of t planes is exactly the channel's t-bit quantizer, and the query spends its bit budget by reverse water-filling over the variance-weighted importance of its channels. At one million tokens on Qwen3-8B a decode step is 1.67x faster in GPU time than with the 136-bit scans of Double Sparsity, Loki and SparQ r=32, and in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer bytes with lower attention error on six of seven model and context settings. On RULER-style tasks every per-token scan matches exact top-k decoding
    
[^158]: KV缓存驱逐下的分歧时机与累积分歧

    Divergence Timing and Cumulative Disagreement under KV-Cache Eviction

    [https://arxiv.org/abs/2609.16617](https://arxiv.org/abs/2609.16617)

    该论文建立了KV缓存驱逐引发自回归生成分歧的理论框架，将累积token不匹配精确分解为首次分歧贡献与分歧后暴露量的乘积形式，并通过条件蒙特卡洛无偏估计及Llama-3.1与Qwen2.5模型实验验证了不同缓存压缩策略在分歧时机上的差异。

    

    KV缓存驱逐会扰动控制自回归生成的条件token分布。我们研究首次分歧时机及随后的token不匹配如何决定累积分歧。在指定的逐步最大耦合下，我们推导出精确分解：期望不匹配比例等于首次不匹配贡献，加上分歧后暴露时间乘以其不匹配率。通过在无约束自回归核对上的显式构造，我们实现了与有限分歧对齐观察窗口相容的风险尖锐区间。残差分支条件蒙特卡洛方法对发生、占据及窗口/尾部贡献提供了无偏联合估计，并对总token损失具有逐次重复的方差优势。来自Meta-Llama-3.1-8B-Instruct和Qwen2.5-7B-Instruct的完整轨迹表明，50%保留率的SnapKV比SnapKV-512或其他方法更晚且更少地进入分歧状态。

    arXiv:2609.16617v1 Announce Type: new  Abstract: KV-cache eviction perturbs the conditional token distributions governing autoregressive generation. We investigate how first-divergence timing and subsequent token mismatch determine cumulative disagreement. We derive an exact decomposition under a specified stepwise maximal coupling: the expected mismatch fraction equals a first-mismatch contribution plus post-divergence exposure multiplied by its mismatch rate. An explicit construction over unrestricted autoregressive kernel pairs realizes the sharp interval of risks compatible with a finite divergence-aligned observation window. Residual-branch conditional Monte Carlo provides unbiased joint estimates of occurrence, occupation, and window/tail contributions, with per-replicate variance dominance for total token loss. Complete trajectories from Meta-Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct show that SnapKV at 50% retention enters divergence later and less often than SnapKV-512 or 
    
[^159]: 通过影响矩阵估计实现大规模数据归因

    Data Attribution at Scale via Influence Matrix Estimation

    [https://arxiv.org/abs/2609.15044](https://arxiv.org/abs/2609.15044)

    该论文将预算受限的数据归因问题转化为从少量测量中估计大型影响矩阵的问题，并据此提出了MAGE和SPEL两种算法，大幅降低了大规模数据归因的计算成本。

    

    数据归因旨在量化单个训练样本如何塑造模型的预测，并支撑包括数据估值、机器遗忘和模型可解释性在内的一系列问题。尽管已有大量研究工作，但由于神经网络的非凸性质，计算上可扩展的方法往往难以准确预测移除训练数据所产生的影响。为克服这一挑战，基于元梯度的方法（如MAGIC (Ilyas and Engstrom, 2025)）通过整个训练过程对每个预测进行微分，并计算其相对于训练数据的精确影响，但需要对每个预测单独运行一次完整的训练。为降低这一成本，我们将预算受限的数据归因问题转化为从少量测量中估计一个大型影响矩阵。我们证明，最适合恢复该矩阵的测量方式与最适合归因本身的测量方式有所不同。随后，我们提出了两种算法，MAGE和SPEL。

    arXiv:2609.15044v1 Announce Type: cross  Abstract: Data attribution seeks to quantify how individual training examples shape a model's predictions and underpins problems including data valuation, machine unlearning, and model interpretability. Despite having a long line of work, computationally scalable methods often struggle to predict the effect of removing training data in neural networks due to their non-convex nature. To overcome this challenge, metagradient-based methods such as MAGIC (Ilyas and Engstrom, 2025) differentiate each prediction through the entire training run and compute its exact influence with respect to the training data, but require a separate run for every prediction. To reduce this cost, we cast budgeted attribution as estimating a large influence matrix from a small number of measurements. We show that the measurements most appropriate for recovering this matrix differ from those best suited for attribution itself. We then present two algorithms, MAGE and SPEL
    
[^160]: 免数据的在线策略蒸馏

    Data-free On-policy Distillation

    [https://arxiv.org/abs/2609.14193](https://arxiv.org/abs/2609.14193)

    该研究发现在线策略蒸馏（OPD）对训练数据几乎不敏感——仅八个提示即可媲美1.7万题的数据集，且跨领域数据仍能保留90%以上的收益，表明OPD传递的是教师的推理方式而非具体知识。

    

    在线策略蒸馏已成为前沿后训练流水线的标准组成部分，但其训练数据实际贡献了多少却基本未被检验。在实践当中最常见的两组师生模型配对上，我们发现OPD对其数据几乎不敏感：仅八个提示就能与一个包含1.7万道题目的数据集效果相当，而且三个独立构建、难度和师生KL散度相差数倍的数据集，产生了几乎难以区分的训练曲线。有两个原因可以解释这一现象。第一，OPD中数据的单位是一个提示所引导的状态，而非提示本身：随着采样的持续进行，单个提示会不断暴露新的教师纠正信号，而增加提示的边际价值在八个之后便急剧坍缩。第二，用竞赛编程替代数学领域仍能恢复超过百分之九十的域内收益，这表明OPD传递的是教师的推理方式，而非知识本身（原文在此处截断）。

    arXiv:2609.14193v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has become a standard component of frontier post-training pipelines, yet how much its training data actually contributes has gone largely unexamined. On the two teacher-student pairings most common in practice, we find OPD almost indifferent to its data: eight prompts already match a 17k-problem dataset, and three independently built datasets whose difficulty and teacher-student KL differ several-fold produce nearly indistinguishable training curves. Two causes account for this. First, the unit of data in OPD is the state a prompt leads to, not the prompt itself: a single prompt keeps exposing new teacher correction as sampling continues, while the marginal value of additional prompts collapses after eight. Second, replacing mathematics with competitive programming still recovers over ninety percent of the in-domain gain, indicating that OPD transfers the teacher's mode of reasoning rather than knowledge re
    
[^161]: 在纸上书写并获取在线数字轨迹：手写的新时代

    Write on Paper and Get the Online Digital Trace:\newline A New Era for Handwriting

    [https://arxiv.org/abs/2609.12702](https://arxiv.org/abs/2609.12702)

    该论文提出了一种结合传感器数字笔与先进AI算法的创新方案，无需特殊纸张或外部参考系统，即可实时重建在普通纸上书写的手写数字轨迹。

    

    捕获手写的数字轨迹通常需要特定的触控笔和兼容的基底材料，无论是电容式触摸屏、Wacom系统所使用的电磁共振（EMR）数位板，还是特殊纸张。虽然在普通纸上书写能提供丰富的触觉体验、无延迟，并且以改善信息记忆保留而闻名，但目前尚不存在一种低成本、被广泛接受的有效方案来数字化这种笔迹。其挑战在于如何在没有外部参考系统的情况下准确跟踪笔的轨迹，同时允许笔在表面上不受限制地自由移动。我们提出了一种创新解决方案，结合了数字笔、先进的人工智能算法和自适应AI技术来重建手写的数字轨迹。我们的方法集成了硬件开发（专注于配备传感器的笔）与软件创新，以实时优化轨迹的重建与处理。

    arXiv:2609.12702v1 Announce Type: new  Abstract: Capturing the digital trace of handwriting usually requires a specific stylus and a compatible substrate, be it a capacitive touchscreen, an ElectroMagnetic Resonance (EMR) tablet as used in Wacom systems or special paper. While writing on regular paper offers rich haptics, no latency and is well known for improving information retention, no low-cost and widely accepted, effective solution exists to digitize such a pen trace. The challenge is to accurately track the pen's trajectory without an external reference system while allowing unrestricted freedom of pen movement across a surface. We propose an innovative solution that combines a digital pen, advanced artificial intelligence algorithms, and adaptive AI techniques to reconstruct the digital trace of handwriting. Our approach integrates hardware development, focusing on a sensor-equipped pen, with software innovations to optimize trajectory reconstruction and processing in real time
    
[^162]: HuRo：将人类视频机器人化以实现可扩展的VLA预训练

    HuRo: Robotizing Human Videos for Scalable VLA Pretraining

    [https://arxiv.org/abs/2609.10706](https://arxiv.org/abs/2609.10706)

    该论文开发了一套将异构人类视频转换为机器人对齐的观察与动作轨迹的机器人化流水线，并据此构建了包含约63万条机器人化片段和1.42亿帧的HuRo数据集，验证了机器人化人类视频能够为VLA策略预训练提供有效且可扩展的监督信号。

    

    人类视频数据集已成为昂贵真实机器人数据的一种极具吸引力的替代方案，能够以规模化方式提供丰富的多样性。为了弥合人类与机器人身体形态之间的差异，现有方法要么在与任务匹配的设置下对视频进行机器人化处理，要么在大规模场景下分别处理观察和动作的对齐。在这项工作中，我们系统地研究了机器人化的人类视频能否为视觉-语言-动作（VLA）策略的预训练提供有效且可扩展的监督信号。为此，我们开发了一套机器人化流水线，能够将异构的人类视频转换为与机器人对齐的观察和动作轨迹，同时在不同标注层级上推断缺失的中间信号。利用该流水线，我们构建了HuRo数据集，其中包含来自五个人类视频来源的约63万条机器人化片段和1.42亿处理后的帧。在四项真实世界操作任务上，扩大机器人化预训练规模能够提升整体完成效果

    arXiv:2609.10706v1 Announce Type: cross  Abstract: Human video datasets have emerged as a compelling alternative to expensive real-robot data, offering rich diversity at scale. To bridge the human-to-robot embodiment gap, existing approaches either robotize videos in task-matched settings or address observation and action alignment separately at scale. In this work, we systematically examine whether robotized human videos can provide effective and scalable supervision for pretraining vision-language-action (VLA) policies. To this end, we develop a robotization pipeline that converts heterogeneous human videos into robot-aligned observations and action trajectories while inferring missing intermediate signals across annotation levels. Using this pipeline, we construct the HuRo dataset, comprising about 630K robotized episodes and 142M processed frames from five human-video sources. Across four real-world manipulation tasks, increasing robotized pretraining scale improves overall complet
    
[^163]: EFQ-Softmax：面向Softmax的无指数量化方法

    EFQ-Softmax: Exp-Free Quantization for Softmax

    [https://arxiv.org/abs/2609.09721](https://arxiv.org/abs/2609.09721)

    提出 EFQ-Softmax，一种无需计算指数的低比特概率生成方法，直接将注意力分数映射为块缩放的 E2M1 操作数，消除了 softmax 中高精度概率生成与低比特矩阵计算之间的不匹配，从而加速 Transformer 推理。

    

    低比特注意力通过将 $QK^\top$ 和 $PV$ 矩阵乘法迁移到 FP8 或 FP4 矩阵引擎来加速 Transformer 推理。然而，softmax 路径通常需要以更高精度计算偏移分数的指数，生成一个临时的概率块，并在低比特 $PV$ 乘法之前对其进行量化。这种“先求指数再量化”的路径造成了高精度概率生成器与低比特矩阵消费者之间的不匹配。我们提出 EFQ-Softmax（面向 Softmax 的无指数量化），这是一种低比特概率生成方法，可直接将偏移后的注意力分数映射为块缩放的 E2M1 操作数。对于每个微缩放块，EFQ-Softmax 从局部最大值中选取一个仅含指数的缩放因子，将偏移分数映射到归一化的残差域，并使用单一仿射规则生成非负的 E2M1 概率编码。所得操作数在 $\widetilde{P}V$ 分子更新与……（摘要在此处截断）中保持一致使用。

    arXiv:2609.09721v2 Announce Type: replace  Abstract: Low-bit attention accelerates Transformer inference by moving the $QK^\top$ and $PV$ matrix multiplications to FP8 or FP4 matrix engines. However, the softmax path often evaluates shifted-score exponentials in higher precision, forms a temporary probability block, and quantizes it before low-bit $PV$ multiplication. This exp-then-quantize path creates a mismatch between a high-precision probability producer and a low-bit matrix consumer. We propose EFQ-Softmax (Exp-Free Quantization for Softmax), a low-bit probability-generation method that directly maps shifted attention scores to block-scaled E2M1 operands. For each microscaling block, EFQ-Softmax selects an exponent-only scale from the local maximum, maps the shifted scores to a normalized residual domain, and generates nonnegative E2M1 probability codes using a single affine rule. The resulting operand is used consistently in both the $\widetilde{P}V$ numerator update and the $\w
    
[^164]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^165]: 神经表征中的认证拓扑交互：类解缠主要是成对发生的

    Certified Topological Interaction in Neural Representations: Class Disentanglement Is Mostly Pairwise

    [https://arxiv.org/abs/2609.08561](https://arxiv.org/abs/2609.08561)

    本文提出用交集欧拉示性数轮廓将神经表征中的类解缠度量为带统计认证的拓扑交互，发现解缠主要发生在成对类别之间、随深度分级且集中在训练早期。

    

    类解缠（即表征的类条件点云沿网络深度和训练过程的分离程度）通常是通过描述性曲线来读取的。我们将其度量为标记点云之间的认证拓扑交互，使用最近提出的交集欧拉示性数轮廓：即点云球体并集交集的欧拉示性数作为尺度的函数，通过一次Alpha复形扫描即可计算，无需边界矩阵约化。每一个数值都附带统计检验：双向的精确置换检验、受保护的分离证书，以及针对应用中比较性结论的配对检验。在111个训练网络和52,650个认证测量中，解缠是随深度分级的，并集中在最初的几个训练周期；交互商数可按可混淆性对类别对进行排序（Spearman rho=0.83），其效果与廉价的可分性统计指标相当。在96个模型的析因实验种群中，数据增强……

    arXiv:2609.08561v1 Announce Type: new  Abstract: Class disentanglement (the separation of a representation's class-conditional point clouds along depth and over training) is usually read off descriptive curves. We measure it as certified topological interaction between labeled point clouds, using the recently introduced Intersection Euler Characteristic Profile: the Euler characteristic of the overlap of the clouds' ball unions as a function of scale, computed by one Alpha-complex sweep with no boundary-matrix reduction. Every number carries a test: exact permutation tests in both directions, a guarded separation certificate, and a paired test for the comparative claims applications make. Across 111 trained networks and 52,650 certified measurements, disentanglement is depth-graded and concentrated in the first epochs, and interaction quotients rank class pairs by confusability (Spearman rho=0.83), on par with cheap separability statistics. In a 96-model factorial population, augmentat
    
[^166]: 归因Cohen's d：常模年龄生物标志物中疾病相关效应的训练数据归因

    Attributing Cohen's d: Training Data Attribution for Disease-Related Effects in Normative Age Biomarkers

    [https://arxiv.org/abs/2609.07729](https://arxiv.org/abs/2609.07729)

    本文提出一种基于Cohen's d的闭式影响泛函，可直接将常模年龄模型中疾病相关的年龄差距效应归因到个体训练样本，并在UK Biobank上证明移除最有影响力的训练样本可显著提高疾病相关效应量。

    

    常模年龄模型在名义上健康的队列中进行训练，用于预测实际年龄（实际年龄）。当应用于患者时，模型会产生偏差，预测年龄与实际年龄之间的差距被解读为疾病风险。本文将年龄差距的疾病相关效应量直接归因于个体训练样本，而不是以预测级别的损失作为归因目标。对于Cohen's d，由此得到的闭式影响泛函（经留一法重训练验证）可根据训练样本对留出数据的病例-对照分离效果的影响对其进行排序。在英国生物银行（UK Biobank）的四种疾病和两种生物标志物模态上，移除最具影响力的10%训练样本在所有随机种子下均能提高留出的疾病相关效应量：对于2型糖尿病，代谢组学年龄效应增加了一倍以上；对于多发性硬化症，脑年龄效应提高了约三分之一。而随机移除即使达到50%，效应量也保持不变，这证实了……

    arXiv:2609.07729v1 Announce Type: new  Abstract: Normative age models are trained to predict chronological age in a nominally healthy cohort. Applied to patients, they deviate, and the gap between predicted and chronological age is read as disease risk. Here, we attribute the disease-related effect size of the age gap directly to individual training samples, rather than using a prediction-level loss as the attribution target. For Cohen's $d$, the resulting closed-form influence functional, validated against leave-one-out retraining, ranks training samples by their effect on held-out case-control separation. Across four diseases and two biomarker modalities in UK Biobank, removing the 10% most influential training samples raises held-out disease-related effect size in every seed. It more than doubles the metabolomic-age effect for type-2 diabetes and raises the brain-age effect for multiple sclerosis by roughly a third. Random removal leaves effect size flat even at 50% removal, confirm
    
[^167]: VLA-Precision：面向视觉-语言-动作模型高效真实世界在线强化的非对称协同自举方法

    VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models

    [https://arxiv.org/abs/2609.04355](https://arxiv.org/abs/2609.04355)

    提出VLA-Precision框架，通过非对称协同自举算法和ACoB-Stream架构，同时解决VLA模型真实世界在线强化学习中价值信号不可靠与计算开销过大两大瓶颈，实现高效且精确的策略改进。

    

    预训练的视觉-语言-动作（VLA）模型能够实现广泛的操作任务，但在需要精度和可重复性的任务中仍然不够可靠。将真实世界在线强化学习应用于VLA后训练，可以在示范数据之外实现自主的试错式改进，但这也暴露出两个瓶颈：1）不可靠的价值信号可能导致策略漂移；2）大型VLA的开销限制了吞吐量和样本效率。为应对这些挑战，我们提出了VLA-Precision，一个高效的真实世界在线强化学习框架，其核心是非对称协同自举算法和ACoB-Stream架构。具体而言，ACoB在不同时间尺度上建立非对称协同自举机制：早期由干预引导的行为学习快速提升策略性能，同时提高在线经验的质量。随着自主经验的不断积累，全局回报传播与局部偏好排序逐步校准……（摘要截断）

    arXiv:2609.04355v1 Announce Type: cross  Abstract: Pretrained vision-language-action (VLA) models enable broad manipulation but remain unreliable in tasks demanding precision and repeatability. Applying real-world online reinforcement learning (RL) to VLA post-training enables autonomous trial-and-error improvement beyond demonstrations alone, but exposes two bottlenecks: 1) unreliable value signals can induce policy drift; 2) large-VLA overhead constrains throughput and sample efficiency. To address these challenges, we present VLA-Precision, an efficient real-world online RL framework featuring the Asymmetric Co-Bootstrapping (ACoB) algorithm and the ACoB-Stream architecture. Specifically, ACoB establishes asymmetric co-bootstrapping across timescales: early intervention-guided behavioral learning rapidly improves policy performance while enhancing online experience quality. As autonomous experience accumulates, global return propagation and local preference ranking progressively cal
    
[^168]: 基准的平衡：面向基准多重性与任务条件评估的语义密度重加权

    Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation

    [https://arxiv.org/abs/2608.30044](https://arxiv.org/abs/2608.30044)

    提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。

    

    语言模型通常通过对基准列表中的分数进行等权平均来进行比较。这类列表通过公开发表而不断增长，缺乏显式的测量设计，因此等权加权使已发表基准的密度成为隐含的能力权重：基准密集的区域会被重复计算。我们提出了基准平衡方法，该方法对基准描述进行嵌入，并为每个基准分配逆密度语义权重。在公开的密度尺度下，相近的基准条目共享聚合影响力。在将异构分数统一到共同潜在尺度后，残差场利用相同的几何结构，使模型排名能够以任务查询为条件。这两个组成部分在实证上发挥着不同的作用。在包含586个模型和14个基准的数据快照上，BoB能够预测哪些模型在留出任务上表现出超出其一般能力的异常强势，画像相关性达到0.462，而等权加权下仅为0.049。

    arXiv:2608.30044v1 Announce Type: new  Abstract: Language models are commonly compared by averaging scores across a benchmark list with equal weight. Such lists grow through publication outside an explicit measurement design, so equal weighting turns the density of published benchmarks into an implicit capability weight: densely benchmarked regions count repeatedly. We introduce Balance of Benchmarks (BoB), which embeds benchmark descriptions and assigns each benchmark an inverse-density semantic weight. Nearby entries share aggregate influence at a disclosed density scale. After equating heterogeneous scores onto a common latent scale, a residual field uses the same geometry to condition model rankings on a task query. The two components serve distinct empirical roles. On a snapshot of 586 models and 14 benchmarks, BoB predicts which models are unusually strong on a held-out task beyond their general ability, reaching a profile correlation of 0.462 compared with 0.049 under equal weig
    
[^169]: SafeStep：面向行人安全监控的语义通信交互式演示

    SafeStep: An Interactive Demonstration of Semantic Communication for Pedestrian Safety Monitoring

    [https://arxiv.org/abs/2608.27688](https://arxiv.org/abs/2608.27688)

    SafeStep是一个基于浏览器的交互式语义通信演示平台，可从实时交通摄像头中提取行人信息进行安全监控，并展示了仅416万参数的Meta-VIB模型无需在线重训练即可在不同SNR、码长和AoI条件下泛化的优势。

    

    本文开发了SafeStep，一个基于浏览器的交互式语义通信平台，用于实时行人安全监控。SafeStep从四路实时交通摄像头画面中提取行人信息，通过语义通信收发器在加性高斯白噪声（AWGN）信道上传输，并渲染用户特定的位置、轨迹和风险标签。该平台允许独立选择收发器、信噪比（SNR）、码长和信息年龄，并通过实时行人安全监控向每个浏览器展示所选配置的收发器性能。SafeStep将最近提出的语义通信设计Meta-VIB与五种基线收发器进行比较。Meta-VIB使用一个仅有416万参数的紧凑神经模型，无需在线重训练即可在不同SNR、码长和AoI值之间实现泛化。实验……

    arXiv:2608.27688v1 Announce Type: new  Abstract: In this paper, we develop SafeStep, an interactive browser-based semantic communication platform for live pedestrian safety monitoring. SafeStep extracts pedestrian information from four live traffic-camera feeds, transmits it through a semantic communication transceiver over an Additive White Gaussian Noise (AWGN) channel, and renders user-specific positions, trajectories, and risk labels. The platform allows to independently select the transceiver, Signal-to-Noise Ratio (SNR), codelength, and Age of Information (AoI), and demonstrates the transceiver performance of the selected configuration through live pedestrian safety monitoring to each browser. SafeStep compares a recently proposed semantic communication design called Meta-VIB with five baseline transceivers. Meta-VIB uses a compact neural model with only $4.16$ million parameters to generalize across varying SNR, codelength, and AoI values without online retraining. Experimental 
    
[^170]: 行星预测引擎：通过智能数据选择和基础模型嵌入实现自主地理空间预测

    Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings

    [https://arxiv.org/abs/2608.26088](https://arxiv.org/abs/2608.26088)

    行星预测引擎是一个自主AI系统，能从自然语言查询直接端到端执行地理空间预测，通过智能数据选择和基础模型嵌入，自动整合多模态数据并搜索最优模型，以应对全球性挑战。

    

    应对从粮食安全、灾害风险到疾病爆发和社会经济脆弱性等关键全球挑战，需要高保真度的地理空间建模。然而，构建预测性行星模型仍受制于碎片化的数据生态系统，需要手动数据检索、多模态数据整理和融合以及迭代模型选择。我们提出了行星预测引擎（PPE），这是一种自主AI系统，可直接从自然语言查询中执行端到端工作流程。PPE动态合成多模态数据集，在开放网络和地球观测平台（Data Commons、Google Earth Engine）上检索时空相关协变量，并将其与地理空间基础模型嵌入（PDFM、AlphaEarth）融合。同时，它通过自动过拟合防护搜索针对任务定制的模型架构家族。在多样化的任务、地理区域和科学领域中，该引擎展现出显著性能。

    arXiv:2608.26088v1 Announce Type: cross  Abstract: Addressing critical global challenges, from food security and disaster risk to disease outbreaks and socio-economic vulnerability, demands high-fidelity geospatial modeling. However, building predictive planetary models remains bottlenecked by a fragmented data ecosystem, requiring manual data retrieval, multimodal data curation and fusion along with iterative model selection. We present the Planetary Prediction Engine (PPE), an autonomous AI system that executes this end-to-end workflow directly from natural-language queries. PPE synthesizes multimodal datasets on the fly, retrieving spatiotemporally relevant covariates across open-web and Earth observation platforms (Data Commons, Google Earth Engine) and fusing them with geospatial foundation model embeddings (PDFM, AlphaEarth). Simultaneously, it searches over task-tailored model architecture families with automated overfitting guards. Across diverse tasks, geographies, and scienti
    
[^171]: 凸损失函数及其在SVM、SVR和浅层神经网络中的应用

    Convex losses and their applications to SVM, SVR, and Shallow Neural Networks

    [https://arxiv.org/abs/2608.14288](https://arxiv.org/abs/2608.14288)

    本文提出了多种新的凸损失函数用于SVM和神经网络，证明了它们是标准损失的推广，但实验结果显示这些新损失并未显著提升泛化性能。

    

    arXiv:2608.14288v1 公告类型：新 摘要：我们提出了多种用于SVM和神经网络的新凸损失函数，应用于二分类任务。尽管在利用对偶SVM模型时存在实际限制，但我们能够在SVM原始形式和神经网络中使用它们。具体来说，使用粒子群优化算法解决了带有修改损失函数的原始SVM问题。我们证明了所提出的损失函数是标准损失函数的推广，并在几个小型数据集上进行了实验。这项初步研究表明，在损失函数中利用模式相关性理论上可以增强某些数据集的泛化性能。为了评估每种损失函数的性能，我们采用了嵌套交叉验证程序。结果显示，使用或不使用新损失函数，泛化度量是相同的。

    arXiv:2608.14288v1 Announce Type: new  Abstract: We propose multiple new convex losses for SVM and Neural Networks, applied to binary classification tasks. While there are practical limitations in exploiting them with the dual SVM models, we are able to use them with SVM primal formulation and Neural Networks. In detail, the primal SVM problem with the modified losses has been solved with the Particle Swarm Optimization algorithm. We prove that the proposed losses are a generalization of the standard loss, and we experiment them with several small data-sets. This preliminary study shows that using pattern correlations   inside the loss function could in theory enhance the generalization performances on some data-sets. To evaluate the performance of each loss, we adopt a Nested Cross-Validation procedure. Results show that generalization measures are the same with or without the new losses.
    
[^172]: 对比性概念重要性：通过自动提取的概念表示解释成对类别决策

    Contrastive Concept Importance: Explaining Pairwise Class Decisions Through Automatically Extracted Concept Representations

    [https://arxiv.org/abs/2607.27904](https://arxiv.org/abs/2607.27904)

    提出对比性概念重要性方法，将目标类别与对比类别之间的logit间隔归因于自动提取的视觉概念，从而回答“为什么是P而不是Q”这一对比性问题。

    

    基于概念的解释是一种流行的方法，它通过语义上有意义、人类可解释的概念来解释复杂黑盒方法的决策。为了将此类概念对模型决策的贡献进行归因，特征归因方法被用来量化每个概念对模型输出的贡献强度。这些归因通常是针对单一输出类别计算的，因此回答的是非对比性的“为什么是P？”的问题。然而，在许多情况下，例如误分类、类别混淆和低置信度预测的情形，更自然的问题是“为什么是P而不是Q？”。我们提出了对比性概念重要性方法，它将目标类别与对比类别（foil）之间的logit间隔归因于自动提取的视觉概念基中的概念。所得到的分数带有符号，表明某个概念是支持目标类别而非对比类别，还是支持对比类别而非目标类别。

    arXiv:2607.27904v2 Announce Type: replace  Abstract: Concept-based explanations are a prevalent way to explain the decisions of complex black-box methods through semantically meaningful, human interpretable concepts. To attribute the contribution of such concepts to a model's decisions, feature attribution methods are used to quantify how strongly each concept contributes to a model output. These attributions are typically computed for a single output class and therefore answer a non-contrastive "why P?" question. In many situations, however, such as cases of misclassification, class confusion, and low- margin predictions, the more natural question to ask is "why P rather than Q?". We introduce contrastive concept importance, which attributes the logit margin between a target class and a contrast, or foil, class to concepts in an automatically extracted visual concept basis. The resulting scores are signed, indicating whether a concept supports the target over the foil or the foil over
    
[^173]: 使用拉格朗日神经细胞自动机模拟宇宙结构形成

    Emulating Cosmic Structure Formation with a Lagrangian Neural Cellular Automaton

    [https://arxiv.org/abs/2607.27320](https://arxiv.org/abs/2607.27320)

    提出了一种拉格朗日神经细胞自动机（LNCA）混合深度学习框架，通过在拉格朗日坐标系中迭代地平流计算图本身来跟随物质流动，从而以局部迭代的方式高效、精确且完全可微分地模拟宇宙结构形成，为场级宇宙学推断提供了实用的前向模型。

    

    从星系巡天数据中对宇宙学初始条件进行场级推断，需要一个在非线性区域保持精确、计算高效且完全可微分的前向模型。传统的N体模拟虽然精确，但对于迭代推断而言计算成本过高；而像拉格朗日微扰理论（LPT）这样的近似求解器则无法捕捉晚期宇宙网中复杂的晕形成动力学。我们提出了拉格朗日神经细胞自动机（LNCA），这是一种混合深度学习框架，可以将结构形成模拟为共动晶格上的局部迭代动力学过程。与在单次传递中映射固定网格的卷积模拟器不同，LNCA在拉格朗日坐标系中运行，迭代地平流计算图本身以跟随物质的流动。通过训练网络仅学习残差色散（摘要在此处截断）……

    arXiv:2607.27320v2 Announce Type: replace-cross  Abstract: Field-level inference of cosmological initial conditions from galaxy surveys requires a forward model that is simultaneously accurate in the non-linear regime, computationally efficient, and fully differentiable. Traditional N-body simulations are accurate but computationally prohibitive for iterative inference, while approximate solvers like Lagrangian Perturbation Theory (LPT) fail to capture the knotty halo-forming dynamics of the cosmic web at late times. We introduce the \textit{Lagrangian Neural Cellular Automaton} (LNCA), a hybrid deep learning framework that can be applied to emulate structure formation as a local, iterative dynamical process on a comoving lattice. Unlike convolutional emulators which map fixed grids in a single pass, the LNCA operates in the Lagrangian frame, iteratively advecting the computational graph itself to follow the flow of mass. By training the network to learn only the \textit{residual} disp
    
[^174]: 决策树、Frobenius迹与椭圆曲线的Weierstrass系数

    Decision trees, Frobenius traces, and Weierstrass coefficients of elliptic curves

    [https://arxiv.org/abs/2607.24251](https://arxiv.org/abs/2607.24251)

    该论文通过在 LMFDB 数据库上训练决策树模型，首次发现并严格证明了椭圆曲线极小 Weierstrass 模型的系数可由其 $L$-函数的 Dirichlet 系数（即 Frobenius 迹）通过显式公式确定，并应用于椭圆曲线数据表的计算。

    

    我们研究了椭圆曲线 $E/\mathbb{Q}$ 的约化极小 Weierstrass 模型的系数 $(w_1,w_2,w_3,w_4,w_6)$ 在多大程度上由其 $L$-函数的 Dirichlet 系数 $a_n(E)$ 所决定，其中这些系数在好约化素数处的取值即为 $E$ 的 Frobenius 迹。我们证明了 $w_1$、$w_2$ 和 $w_3$ 可以由 $a_2(E)$、$a_3(E)$ 和 $a_4(E)$ 的显式公式给出，$w_4$ 模 $5$ 的值随后由 $a_5(E)$ 确定，而 $w_6$ 模 $7$ 的值则由 $a_7(E)$ 连同 $w_1,w_2,w_3,w_4$ 共同确定。这些看似全新的公式是通过在 LMFDB 上训练决策树模型而发现的；我们报告了相关的实验过程，并探讨了其在构建椭圆曲线数据表计算中的应用。

    arXiv:2607.24251v2 Announce Type: replace-cross  Abstract: We investigate the extent to which the coefficients $(w_1,w_2,w_3,w_4,w_6)$ of the reduced minimal Weierstrass model of an elliptic curve $E/\mathbb{Q}$ are determined by the Dirichlet coefficients $a_n(E)$ of its $L$-function, whose values at primes of good reduction are the Frobenius traces of $E$. We prove that $w_1$, $w_2$ and $w_3$ are given by explicit formulae in $a_2(E)$, $a_3(E)$ and $a_4(E)$, that $w_4$ modulo $5$ is then determined by $a_5(E)$, and that $w_6$ modulo $7$ is determined by $a_7(E)$ together with $w_1,w_2,w_3,w_4$. These formulae, which appear to be new, were discovered by training decision tree models on the LMFDB; we report the accompanying experiments and explore applications to computing tables of elliptic curves.
    
[^175]: 教大语言模型自我进化：用强化学习培养核心元技能

    Teaching LLMs to Self-Evolve: Cultivating Core Meta-Skills with Reinforcement Learning

    [https://arxiv.org/abs/2607.21971](https://arxiv.org/abs/2607.21971)

    提出MetaEvolve框架，通过数据合成、进化感知强化学习和推理时进化搜索来培养大语言模型结合环境反馈进行自我反思等元技能，从而实现迭代式自我进化。

    

    通过环境反馈进行迭代自我进化的测试时扩展方法（如AlphaEvolve所展示的）带来了显著的性能提升。我们假设这类进化框架的成功关键在于元技能，例如结合环境反馈的自我反思，这些技能能够实现有效的多轮改进，但在传统后训练中很大程度上被忽视了。为了弥合这一差距，我们提出了MetaEvolve，一个旨在通过数据合成流水线、进化感知强化学习（RL）和推理时进化搜索来培养这些元技能的框架。具体而言，我们将MetaEvolve扎根于编程领域，因为程序执行提供了超越二元正确性的自然、连续的奖励信号。基于这些信号，我们合成进化轨迹作为训练数据，每条轨迹包含当前程序、其适应度得分（结合正确性和效率）以及先前尝试的历史记录。

    arXiv:2607.21971v2 Announce Type: replace-cross  Abstract: Test-time scaling through iterative self-evolution with environment feedback, as demonstrated by AlphaEvolve, shows remarkable performance gains. We hypothesize that the success of such evolution frameworks hinges on meta-skills, such as self-reflection with environment feedback, that enable effective multi-round refinement, yet are largely neglected by traditional post-training. To bridge this gap, we present MetaEvolve, a framework designed to develop these meta-skills via a data synthesis pipeline, evolution-aware reinforcement learning (RL), and inference-time evolutionary search. Concretely, we ground MetaEvolve in coding, where program execution provides natural, continuous reward signals beyond binary correctness. Building on these signals, we synthesize evolution trajectories as training data, each containing a current program, its fitness score (combining correctness and efficiency), and a history of prior attempts, an
    
[^176]: ANNLib：一个用于高效近似最近邻搜索的开发框架

    ANNLib: A Development Framework for Efficient Approximate Nearest Neighbor Search

    [https://arxiv.org/abs/2607.17582](https://arxiv.org/abs/2607.17582)

    ANNLib通过将基于图的ANNS系统中的算法与数据结构组件解耦并独立优化，同时实现了高性能与灵活功能，支持过滤搜索、完全动态更新、快照历史查询和范围搜索等复杂场景。

    

    近似最近邻搜索（ANNS）在现代深度学习流程中扮演着至关重要的角色。近年来，许多ANNS系统被提出，以提供广泛而灵活的功能或实现高性能。然而，同时实现这两者在本质上是困难的。我们提出ANNLib来填补这一空白。ANNLib是一个库，它基于流行的基于图的ANNS算法，提供了一个编程框架，使ANNS系统能够兼具高性能和灵活的功能。我们精心地将ANNS系统中的算法组件和数据结构组件解耦，并对它们进行独立优化。此外，我们将最先进的算法和数据结构作为模块集成到ANNLib中，同时也集成了我们的新设计。用户可以选择组件的组合，以高性能支持复杂的设置，例如过滤搜索、完全动态更新、基于快照的历史查询以及范围搜索。我们的实验……

    arXiv:2607.17582v2 Announce Type: replace  Abstract: Approximate Nearest Neighbor Search (ANNS) plays a pivotal role in modern deep learning pipelines. Recently, many ANNS systems have been proposed to provide broad, flexible functionalities or achieve high performance. However, it is inherently difficult to achieve both. We propose ANNLib to address this gap. ANNLib is a library that provides a programming framework to achieve high performance and flexible functionalities for ANNS systems, based on popular graph-based ANNS algorithms. We carefully decouple and independently optimize both the algorithm and the data structure components in an ANNS system. In addition, we integrate state-of-the-art algorithms and data structures as modules in ANNLib, as well as our new designs. Users can choose combinations of components to support sophisticated settings with high performance, such as filtered search, fully dynamic updates, historical queries on snapshots, and range searches. Our experim
    
[^177]: TVGL-CFM：基于条件流匹配的动态网络时变轨迹生成与预测

    TVGL-CFM:Generating and Forecasting Time-Varying Trajectories of Dynamic Networks with Conditional Flow Matching

    [https://arxiv.org/abs/2607.16894](https://arxiv.org/abs/2607.16894)

    提出了TVGL-CFM统一生成框架，通过全局对数欧几里得微分同胚映射将SPD精度矩阵轨迹的黎曼流形问题转化为欧氏空间问题，利用基于Transformer的非自回归条件流匹配模型，在无需预先指定图的情况下实现动态网络时变轨迹的类条件生成与历史条件预测。

    

    许多复杂系统，包括脑网络、金融市场和基因调控回路，用随时间演化的交互结构来描述比用单一固定图更为合适。时变图套索（TVGL）从多变量信号中估计这种结构，将其表示为时间上连贯的稀疏精度矩阵序列。我们提出了TVGL-CFM，这是一个统一的生成框架，能够在无需预先指定图的情况下学习完整的SPD（对称正定）精度矩阵轨迹的分布，同时支持类条件生成和历史条件预测。包含T个窗口的SPD轨迹位于乘积黎曼流形(S++^p)^T上。我们构建了一个从该乘积空间到欧几里得序列空间的全局对数欧几里得微分同胚映射，使得基于Transformer骨干的非自回归条件流匹配模型能够联合生成所有时间窗口，并将其解码为SPD矩阵。

    arXiv:2607.16894v2 Announce Type: replace  Abstract: Many complex systems, including brain networks, financial markets, and gene-regulatory circuits, are better described by interaction structures that evolve over time than by a single fixed graph. The time-varying graphical lasso (TVGL) estimates this structure from multivariate signals as a temporally coherent sequence of sparse precision matrices. We introduce TVGL-CFM, a unified generative framework that learns distributions over complete SPD precision-matrix trajectories without requiring a pre-specified graph, supporting both class-conditional generation and history-conditioned forecasting.   An SPD trajectory with T windows lies on the product Riemannian manifold (S++^p)^T. We construct a global log-Euclidean diffeomorphism from this product space to a Euclidean sequence space, enabling a non-autoregressive conditional flow-matching model with a Transformer backbone to generate all windows jointly and decode them to SPD matrices
    
[^178]: 基于提示驱动的探索

    Prompt-Driven Exploration

    [https://arxiv.org/abs/2607.08837](https://arxiv.org/abs/2607.08837)

    本文提出一种利用视觉-语言模型从强化学习展开视频中自动诊断并重写提示的方法，以实现对弱策略的全局探索，而无需依赖稀疏奖励。

    

    摘要：arXiv:2607.08837v2 公告类型：替换交叉 摘要：探索对于强化学习（RL）至关重要，因为策略无法通过反复采样其已偏好的行为来改进。标准方法在动作空间中注入随机性，但这种抖动只会产生接近原始轨迹的展开。要摆脱弱策略，通常需要动作噪声无法产生的全局扰动。大型语言模型（LLM）和视觉-语言-动作（VLA）模型提供了一条途径：它们将策略条件化于自然语言提示，由于展开遵循该提示，修改提示会引发全局变化。挑战在于找到能引发有用全局变化的提示。当弱策略很少成功时，奖励过于稀疏而无法用于选择。我们的想法是从展开本身中提炼提示：一个视觉-语言模型（VLM）对展开视频进行推理，诊断策略如何响应，并重写提示以在下次引发更好的行为。此过程类似于...

    arXiv:2607.08837v2 Announce Type: replace-cross  Abstract: Exploration is essential to RL since a policy cannot improve by repeatedly sampling the behaviors it already prefers. Standard methods inject stochasticity in the action space, but such jitter only yields rollouts close to the original. Escaping a weak policy often requires global perturbations that action noise cannot produce. Large language models (LLMs) and vision-language-action (VLA) models offer a pathway: they condition the policy on a natural language prompt, and since the rollout follows from it, modifying the prompt induces global changes. The challenge is finding prompts that induce useful global changes. With a weak policy that rarely succeeds, reward is too sparse to select on. Our idea is to refine prompts from the rollouts themselves: a vision-language model (VLM) reasons over the rollout video, diagnoses how the policy responded, and rewrites the prompt to elicit better behavior next time. This procedure resembl
    
[^179]: Pfaffian集合的管状邻域及其在神经网络中的应用

    Tubular Neighbourhoods of Pfaffian Sets and Applications to Neural Networks

    [https://arxiv.org/abs/2607.08370](https://arxiv.org/abs/2607.08370)

    本文推导了光滑Pfaffian超曲面管状邻域体积的界，并以此获得了具有Pfaffian激活函数的神经网络分类器鲁棒性条件数的尾部概率界，在单隐层Sigmoid网络情形下给出了决策边界管状邻域关于宽度呈多项式级的界。

    

    我们推导了光滑Pfaffian超曲面的管状邻域体积的界，推广了代数簇的已知结果。这些界用定义函数的Pfaffian格式来表示。作为应用，我们在均匀分布和高斯分布两种设置下，获得了衡量具有Pfaffian激活函数的神经网络分类器鲁棒性的条件数概率分布的尾部界。在具有有理权重的单隐层Sigmoid网络的特殊情形下，我们推导了决策边界的管状邻域关于网络宽度呈多项式级的界。

    arXiv:2607.08370v2 Announce Type: replace-cross  Abstract: We derive bounds for the volume of tubular neighbourhoods of smooth Pfaffian hypersurfaces, generalising known results for algebraic varieties. The bounds are given in terms of the Pfaffian format of the defining functions. As an application, we obtain tail bounds on the probability distribution of a condition number measuring the robustness of neural network classifiers with Pfaffian activation functions, in both the uniform and Gaussian settings. In the special case of single-hidden-layer sigmoid networks with rational weights, we derive polynomial-in-width bounds for tubular neighbourhoods of the decision boundary.
    
[^180]: 二叉树机制在差分隐私持续计数中是最优的

    The Binary Tree Mechanism is Optimal for Differentially Private Continual Counting

    [https://arxiv.org/abs/2607.00876](https://arxiv.org/abs/2607.00876)

    本文证明了在固定隐私参数下，差分隐私持续计数问题中误差对数据流长度的依赖是不可回避的下界（近似DP下 $\Omega(\log^{3/2} n)$，纯DP下 $\Omega(\log^2 n)$），从而确立了二叉树机制的最优性。

    

    私有持续计数是差分隐私中的一个基本问题：给定一条长度为 $n$ 的二进制数据流，其中每个 $1$ 对应一个个体的贡献，目标是在保护每个个体隐私的同时发布所有运行中的累计计数。在隐私参数固定的情况下，标准的二叉树机制在近似差分隐私下可实现期望 $\ell_\infty$ 误差 $O(\log^{3/2} n)$，在纯差分隐私下可实现 $O(\log^2 n)$。这些误差对数据流长度的依赖是否必要，一直是一个核心的开放性问题。对于固定的 $\varepsilon\in(0,1)$，我们证明了：在 $\delta>0$ 为足够小的固定值时，近似差分隐私下的下界为 $\Omega(\log^{3/2} n)$；在纯差分隐私下的下界为 $\Omega(\log^2 n)$。这些下界确立了二叉树机制在两种设置下的最优性。这些界对任意机制均成立，即使整条数据流是……（原文此处截断）

    arXiv:2607.00876v3 Announce Type: replace-cross  Abstract: Private continual counting is a fundamental problem in differential privacy: given a binary stream of length $n$, where each $1$ corresponds to the contribution of one individual, the goal is to release all running counts while protecting the privacy of each individual. For fixed privacy parameters, the standard binary tree mechanism achieves expected $\ell_\infty$ error $O(\log^{3/2} n)$ under approximate differential privacy and $O(\log^2 n)$ under pure differential privacy. Whether these dependences on the stream length are necessary has remained a central open problem.   For fixed $\varepsilon\in(0,1)$, we prove a lower bound of $\Omega(\log^{3/2} n)$ under approximate DP with sufficiently small fixed $\delta>0$, and a lower bound of $\Omega(\log^2 n)$ under pure DP. These bounds establish the optimality of the binary tree mechanism in both settings. The bounds hold for arbitrary mechanisms, even when the entire stream is a
    
[^181]: 先召回后重排：面向大规模代码到代码检索的深度学习模型基准测试

    Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval

    [https://arxiv.org/abs/2606.27401](https://arxiv.org/abs/2606.27401)

    本文对大规模代码检索第一阶段召回中的深度学习模型进行了多语言、多数据集基准测试，揭示了其在TB级代码库上的精度与可扩展性局限，并提出基于LLM的代码规范化与查询重写方案显著提升检索精度。

    

    语义代码搜索和克隆检测对于软件开发、维护和复用至关重要。本文评估了当代深度学习模型在大规模代码到代码搜索引擎中第一阶段召回的有效性、效率和可扩展性。在多种编程语言和数据集上的基准测试揭示了这些模型在TB级源代码集合上精度和可扩展性的关键局限。我们提出了基于大语言模型（LLM）的代码规范化与查询重写方案，为表现较差的模型带来了显著的精度提升。我们的结果对资源受限部署的可持续性以及当前代码专用大语言模型在跨数据集上鲁棒性的假设提出了质疑。最后，我们为构建可扩展、高效的代码检索系统提供了可操作的见解。

    arXiv:2606.27401v2 Announce Type: replace-cross  Abstract: Semantic code search and clone detection are essential for software development, maintenance, and reuse. This paper evaluates the effectiveness, efficiency, and scalability of contemporary deep learning models for first-stage recall in large-scale code-to-code search engines. Benchmarking across multiple programming languages and datasets reveals critical limits in the precision and scalability of these models on Terabyte-scale source-code collections. We present LLM-based code normalisation and query-rewriting schemes that yield significant gains in precision for lower-performing models. Our results question the sustainability of resource-constrained deployment and the assumed robustness of current code-specialised LLMs across datasets. We conclude with actionable insights for building scalable, efficient code-retrieval systems.
    
[^182]: 面向二维不规则排样的几何感知强化学习

    Geometry-Aware Reinforcement Learning for 2D Irregular Nesting

    [https://arxiv.org/abs/2606.10611](https://arxiv.org/abs/2606.10611)

    该论文提出多边形Transformer（PoT）架构并与组合优化强化学习框架相结合，使智能体能从数据中自动学习几何先验，从而突破传统启发式求解器对多边形几何不敏感的局限，有效解决二维不规则排样问题。

    

    传统的二维不规则排样问题启发式求解器存在一个根本性局限：它们对多边形几何视而不见，只能依靠引导式暴力搜索在连续放置空间中导航，几乎缺乏几何指导。本文认为，强化学习在克服这一瓶颈方面具有独特优势。通过将优化策略与几何感知神经编码器相结合，智能体可以直接从数据中自动发现丰富的几何先验，并利用这些学到的几何直觉来策略性地引导探索。为实现这一目标，我们提出了多边形Transformer（PoT），这是一种新颖的架构，能够编码二维连续向量几何，同时支持跨多边形的注意力机制。我们将这一新架构与组合优化强化学习（CORL）训练框架相结合以寻找最优解。为支持这一范式，我们发布了一个开源训练数据集。

    arXiv:2606.10611v2 Announce Type: replace  Abstract: Traditional heuristic solvers for the 2D irregular nesting problem share a fundamental limitation: they are blind to polygon geometry, relying on guided brute-force to navigate the continuous placement space with minimal geometrical guidance. In this paper, we argue that Reinforcement Learning is uniquely positioned to overcome this bottleneck. By pairing an optimization policy with a geometry-aware neural encoder, an agent can automatically discover rich geometric priors directly from data, utilizing these learned intuitions to strategically guide exploration. To realize this, we introduce the Polygons Transformer (PoT), a novel architecture that encodes 2D continuous vector geometries while allowing cross-polygon attention. We couple this novel architecture with a Combinatorial Optimization Reinforcement Learning (CORL) training framework to find optimal solutions. To support this paradigm, we release an open-source training datase
    
[^183]: 拒绝后跟进采样：测量算法化DEX交易中被拒绝决策的结果

    Post-Rejection Follow-up Sampling: Measuring Outcomes of Rejected Decisions in Algorithmic DEX Trading

    [https://arxiv.org/abs/2606.08228](https://arxiv.org/abs/2606.08228)

    本文提出拒绝后跟进采样（PRFS）这一观察性测量方法，通过独立跟踪子系统在与原扫描器相同的实时预言机路径上持续采样被拒绝代币的价格与流动性，从而首次在同一实时交易场所量化测量算法化DEX交易中被拒绝决策的前瞻性市场结果。

    

    在去中心化交易所上，采用过滤器门控机制的算法交易系统会拒绝其评估的绝大多数候选代币，然而这些被拒绝候选代币的前瞻性市场轨迹却很少在与产生拒绝决策相同的实时交易场所上被测量。本文提出了拒绝后跟进采样（Post-Rejection Follow-up Sampling, PRFS），这是一种观察性测量方法：由一个独立的跟踪子系统，使用与执行拒绝的扫描器相同的实时预言机路径，按照预定的采样节奏，从拒绝时刻开始直至固定的分析期限，对每个被拒绝代币的价格和流动性进行采样。该方法通过以下要素进行定义：观测窗口与原因归因规则的正式规范；一个参考实现（prfs v2.0.0），包含可执行的覆盖率测试和原因账本测试；以及一个二次交叉验证实现，通过不同的代码路径重新推导出每一个核心指标数字。伴随数据集……

    arXiv:2606.08228v2 Announce Type: replace-cross  Abstract: Filter-gated algorithmic trading systems on decentralised exchanges reject most candidate tokens they evaluate, yet the observed forward market trajectory of those rejected candidates is rarely measured on the same live venue that produced the rejection. This paper introduces Post-Rejection Follow-up Sampling (PRFS), an observational measurement methodology in which a separate tracking subsystem samples each rejected token's price and liquidity from the same live oracle path used by the rejecting scanner, at a scheduled cadence, from the moment of rejection out to a fixed analytic horizon. The methodology is defined through a formal specification of the observation window and the reason-attribution rule, a reference implementation (prfs v2.0.0) with executable coverage and reason-ledger tests, and a secondary cross-check implementation that re-derives every headline number through a distinct code path. The companion dataset con
    
[^184]: 通过轻量级结构引导的自回归模型实现新颖图生成的规模化

    Scaling Novel Graph Generation via Lightweight Structure-Guided Autoregressive Models

    [https://arxiv.org/abs/2606.04287](https://arxiv.org/abs/2606.04287)

    本文提出一种轻量级自回归图生成框架，通过结构引导的拓扑排序将图序列化以实现接近对数线性的生成复杂度，并结合两阶段训练策略减少过拟合、促进可控的新颖图生成。

    

    生成真实且多样的图是机器学习中的一个关键问题，其应用涵盖分子发现、电路设计、网络安全等领域。然而，当前的图生成模型仍然受限于可扩展性和新颖性。基于扩散的方法通常需要代价高昂的全邻接矩阵操作和漫长的去噪链，而许多自回归和混合模型至少具有二次方复杂度。此外，这些模型往往只是模仿训练图，而无法泛化到训练图之外。我们提出了一个轻量级自回归框架来解决这些问题。该框架使用结构引导的拓扑排序将图序列化为规则的边序列，从而实现接近对数线性的生成效率；并采用两阶段训练策略，将面向探索的数据增强与迭代精化相结合，以减少过拟合并促进可控的新颖性。在分子和非分子基准数据集上的实验……（摘要原文在此处被截断）

    arXiv:2606.04287v2 Announce Type: replace-cross  Abstract: Generating realistic and diverse graphs is a key problem in machine learning, with applications in molecular discovery, circuit design, cybersecurity, and beyond. However, current graph generative models remain limited by scalability and novelty. Diffusion-based methods often require costly full-adjacency operations and long denoising chains, while many autoregressive and hybrid models have at least quadratic complexity. In addition, these models often imitate training graphs rather than generalize beyond them.   We propose a lightweight autoregressive framework to address these issues. It uses a structure-guided topological ordering to serialize graphs into regular edge sequences, enabling near log-linear generation, and a two-phase training strategy that combines exploration-oriented augmentation with iterative refinement to reduce overfitting and promote controlled novelty.   Experiments on molecular and non-molecular benchm
    
[^185]: 生存强化学习：迈向可扩展的自监督强化学习

    Survival Reinforcement Learning: Toward Scalable Self-Supervised RL

    [https://arxiv.org/abs/2605.31273](https://arxiv.org/abs/2605.31273)

    提出生存强化学习（SRL），一种基于在线分类的方法，通过最大化智能体在目标处的停留时间来绕过对比强化学习的结构约束并缓解开关式控制问题，在长时程运动任务上性能超越最先进的CRL达2至8倍。

    

    虽然自监督对比强化学习（CRL）已经展现出卓越的深度扩展能力，成功使用了超过64层的网络，但由于对比损失固有的均匀性-容忍性困境，扩展后的CRL在长时程目标条件规划方面仍然存在困难。我们提出了生存强化学习（SRL），这是一种基于在线分类的替代方法，通过最大化智能体在目标位置的停留时间来扩展生存价值学习框架。SRL绕过了CRL的结构性约束，并缓解了生存框架固有的“bang-bang”（开关式）控制解决方案，这种解决方案在复杂动态系统中常常引发不良行为。在多样化的机器人基准测试上进行评估，扩展后的SRL在操作任务上与最先进的CRL表现相当，并在稳定的长时程运动任务上以2倍到8倍的性能超越它。我们的结果提供了强有力的额外证据，表明基于分类的（摘要在此处截断）

    arXiv:2605.31273v2 Announce Type: replace  Abstract: While self-supervised Contrastive Reinforcement Learning (CRL) has shown remarkable depth-scaling capabilities, successfully using networks over 64 layers, scaled CRL still struggles with long-horizon goal-conditioned planning due to the uniformity-tolerance dilemma inherent in contrastive losses. We introduce Survival Reinforcement Learning (SRL), an online classification-based alternative that extends the survival value learning framework by maximizing the agent's dwell time at target goals. SRL bypasses the structural constraints of CRL and mitigates the "bang-bang" control solutions inherent to survival frameworks, which often induce undesirable behavior in complex dynamical systems. Evaluated across diverse robotic benchmarks, scaled SRL matches state-of-the-art CRL on manipulation tasks and outperforms it by 2x to 8x on stable, long-horizon locomotion tasks. Our results provide strong additional evidence that classification-bas
    
[^186]: 基于符号分类的LHC限制实现在线BSM全局拟合

    Symbolic Classification-Enabled LHC Limits Online BSM Global Fits

    [https://arxiv.org/abs/2605.22330](https://arxiv.org/abs/2605.22330)

    本研究通过符号回归技术推导近似表达式，首次实现了将LHC限制高效纳入BSM全局拟合的在线过程，克服了传统计算瓶颈。

    

    arXiv:2605.22330v1 公告类型：交叉 摘要：超越标准模型（BSM）物理的全局拟合通常涉及理论与实验之间的双向互动。理论模型为实验搜索提供指导，而实验结果反过来又约束理论框架。这种反馈循环的一个关键方面是直接纳入测量值和排除限制的“在线”全局拟合，即在参数扫描过程中进行全局拟合的各个方面。然而，将大型强子对撞机（LHC）的限制纳入此类分析在计算上往往代价高昂，这通常是因为每个参数点所需的时间超过了全局拟合框架可接受的范围。在本研究中，我们展示了通过利用符号回归技术推导出的近似方法，可以将LHC限制纳入“在线”全局拟合。我们利用来自电弱子产生搜索的ATLAS约束数据集，推导出一个能够实现这一目标的数学表达式。

    arXiv:2605.22330v1 Announce Type: cross  Abstract: Global fits of Beyond the Standard Model (BSM) physics often involve a two-way interplay between theory and experiment. Theoretical models provide guidance for experimental searches, while experimental results, in turn, constrain theoretical frameworks. A crucial aspect of this feedback loop is the direct inclusion of measurements and exclusion limits ``online'' global fits, i.e. during the parameter scans aspects of the global fits. However, incorporating the Large Hadron Collider (LHC) limits into such analyses has been computationally prohibitive, often due to time taken per parameter point exceeding the scales acceptable for global fit frameworks. In this study, we show that LHC limits can be incorporated ``online'' global fits by leveraging approximations derived from symbolic regression techniques. We utilize a dataset of ATLAS constraints from searches for electroweakino productions to derive a mathematical expression capable of
    
[^187]: 面向机械可解释性的范例划分方法

    Exemplar Partitioning for Mechanistic Interpretability

    [https://arxiv.org/abs/2605.14347](https://arxiv.org/abs/2605.14347)

    提出了一种无监督的范例划分方法（EP），通过Voronoi划分大语言模型激活空间来构建可解释的特征字典，可用于解释和干预模型行为、追踪训练动态并检测隐藏概念。

    

    我们提出了范例划分（Exemplar Partitioning, EP），这是一种从大语言模型激活中构建可解释特征字典的无监督方法。EP字典是激活空间的一个Voronoi划分，通过在给定距离阈值内对流式激活进行领导聚类（leader-clustering）构建而成。每个区域由一个观测到的范例及其成员激活的平均值来定义，二者共同确定区域的成员关系并提供干预方向。字典的大小由所选阈值下的激活流决定，而非预先选定。范例将区域与观测到的输入直接关联，使得基于相同输入流构建的字典可以在不同层、不同训练检查点以及不同架构之间进行比较。我们展示了EP如何用于解释和干预模型行为、追踪训练过程中激活空间的变化，以及检测隐藏概念。我们还在基础模型与指令微调的Gemma模型上对EP字典进行了比较（原文此处截断）。

    arXiv:2605.14347v3 Announce Type: replace  Abstract: We introduce Exemplar Partitioning (EP), an unsupervised method for building interpretable feature dictionaries from large language model activations. An EP dictionary is a Voronoi partition of activation space, built by leader-clustering streamed activations within a distance threshold. Each region is defined by an observed exemplar and an average of its member activations, which define region membership and provide directions for intervention. Dictionary size is determined by the activation stream at the chosen threshold rather than pre-selected. Exemplars link regions to observed inputs, allowing dictionaries built from the same input stream to be compared across layers, training checkpoints, and architectures.   We demonstrate how EP can be used to interpret and intervene on model behaviour, track changes in activation space through training, and detect hidden concepts. Comparing EP dictionaries on base and instruction-tuned Gemm
    
[^188]: 从对比视角重新审视基于可验证奖励的强化学习

    Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective

    [https://arxiv.org/abs/2605.12969](https://arxiv.org/abs/2605.12969)

    本文从对比视角重新审视GRPO，识别其目标层面的两个局限性，并提出ConSPO方法，通过使用长度归一化的序列对数概率作为回滚得分，改进策略优化效果。

    

    arXiv:2605.12969v4 公告类型：替换-交叉 摘要：组相对策略优化（GRPO）是用于推理任务后训练大型语言模型的最广泛采用的RLVR算法之一。我们首先证明GRPO具有一个等价的判别性重新表述，其中策略优化最大化已验证的正负回滚之间的期望得分差距。这一重新表述揭示了两个目标层面的局限性：可能性错位的代理得分，即优化基于裁剪比率的得分而非控制生成的序列可能性；以及得分不敏感的信用分配，即回滚级别的信用未能反映当前正负回滚之间的得分差距。为解决这些局限性，我们提出ConSPO，一种对比序列级策略优化方法，使用长度归一化的序列对数概率作为回滚得分，并将已验证的正回滚与负分布进行对比。

    arXiv:2605.12969v4 Announce Type: replace-cross  Abstract: Group Relative Policy Optimization (GRPO) is one of the most widely adopted RLVR algorithms for post-training large language models on reasoning tasks. We first show that GRPO admits an equivalent discriminative reformulation, in which policy optimization maximizes the expected score gap between verified positive and negative rollouts. This reformulation reveals two objective-level limitations: likelihood-misaligned surrogate scores, in which clipped ratio-based scores are optimized rather than the sequence likelihoods that govern generation, and score-insensitive credit assignment, in which rollout-level credit does not reflect the current score gaps between positive and negative rollouts. To address these limitations, we propose ConSPO, a Contrastive Sequence-level Policy Optimization method that uses length-normalized sequence log-probabilities as rollout scores and contrasts verified positive rollouts against negative distr
    
[^189]: ISOMORPH：用于仿真、数据集生成与预测基准的供应链数字孪生

    ISOMORPH: A Supply Chain Digital Twin for Simulation, Dataset Generation, and Forecasting Benchmarks

    [https://arxiv.org/abs/2605.12768](https://arxiv.org/abs/2605.12768)

    提出了首个公开的多级物流网络供应链数字孪生ISOMORPH，其参数可配置、动力学具有马尔可夫性并能再现牛鞭效应，同时发布了填补供应链物流空白的时间序列预测基准数据集。

    

    开放的时间序列预测（TSF）基准覆盖了零售、能源、天气和交通等领域，但供应链物流领域仍然缺乏充分的覆盖。我们提出了ISOMORPH，这是首个公开的多级物流网络数字孪生，具有可解释、用户可配置的参数，以及模块化的拓扑、需求和控制规则。该模拟器在离散时间中推进有向路由图：需求由库存满足或记录为缺货积压，并触发整个网络的补货。状态跟踪库存、未完成订单、在途货运以及平滑的需求估计，从而在可处理的状态空间上产生马尔可夫动力学。发布的数据以与实证一致的程度再现了牛鞭效应，同时三条守恒定律为模拟器扩展提供了验证工具。我们在两个目录规模（C=50 和 C=200）下发布数据集，并在 C=50 下提供包含33次滚动的场景库。

    arXiv:2605.12768v3 Announce Type: replace-cross  Abstract: Open time-series forecasting (TSF) benchmarks cover retail, energy, weather, and traffic, but supply-chain logistics remains underserved. We introduce ISOMORPH, the first public digital twin of a multi-echelon logistics network with interpretable, user-configurable parameters and modular topology, demand, and control rules. The simulator advances a directed routing graph in discrete time: demand is served from inventory or recorded as backlog and triggers replenishment throughout the network. The state tracks inventory, outstanding orders, in-transit shipments, and a smoothed demand estimate, yielding Markovian dynamics on a tractable state space. The released data reproduces the bullwhip effect at empirically consistent magnitudes, while three conservation laws provide verification tools for simulator extensions. We release datasets at two catalogue scales ($C=50$ and $C=200$), with a 33-rollout scenario library at $C=50$. The
    
[^190]: 训练扩散模型中的临界慢化

    The critical slowing down in training diffusion models

    [https://arxiv.org/abs/2605.12597](https://arxiv.org/abs/2605.12597)

    该论文通过对高斯极限下O(n)模型的可解析分析，证明了训练扩散模型的分数模型时参数学习会出现临界慢化现象，且该慢化同样影响生成过程，说明临界点附近的采样困难在学习型生成模型中依然存在。

    

    计算采样自20世纪中叶以来一直是科学研究的核心。尽管基于机器学习的方法近来取得了重大进展，但人们对其行为的理解仍然有限，对于它们何时以及为何成功的理论控制也十分匮乏。在这里，我们通过分析扩散模型在统计场论O(n)模型的高斯极限（n → ∞）下的应用，为这类在实践中极为有效的生成方案提供了这样的洞见。在这一可解析处理的设定中，我们证明：使用与精确解相匹配的单层网络架构来训练分数模型时，参数学习会表现出一种临界慢化现象。这种慢化同样会影响生成过程，表明众所周知的高于临界点附近采样的困难，即使对于学习型生成模型也依然存在。为了克服这一瓶颈，我们考虑了架构……（原文摘要在此处被截断）

    arXiv:2605.12597v3 Announce Type: replace-cross  Abstract: Computational sampling has been central to the sciences since the mid-20th century. While machine-learning-based approaches have recently enabled major advances, their behavior remains poorly understood, with limited theoretical control over when and why they succeed. Here we provide such insight for diffusion models---a class of generative schemes highly effective in practice---by analyzing their application to the $O(n)$ model of statistical field theory in the Gaussian limit $n \to \infty$. In this analytically tractable setting, we show that training a score model with a one-layer network architecture matching the exact solution exhibits a form of critical slowing down in parameter learning. This slowing down also impacts the generation process, indicating that the well-known difficulties of sampling near criticality persist even for learned generative models. To overcome this bottleneck, we consider the power of architectu
    
[^191]: Rhamba：面向静息态fMRI自监督学习的区域感知混合注意力-Mamba框架

    Rhamba: Region-Aware Hybrid Attention-Mamba Framework for Self-Supervised Learning in Resting-State fMRI

    [https://arxiv.org/abs/2605.01240](https://arxiv.org/abs/2605.01240)

    提出Rhamba框架，将解剖学引导的区域感知掩码与混合Attention-Mamba架构相结合，用于静息态fMRI的自监督预训练，并在精神分裂症和ADHD等下游分类任务中验证了其有效性。

    

    自监督预训练对于大规模神经影像研究颇具前景，然而区域感知掩码与混合序列建模的影响仍未得到充分探索。在本工作中，我们提出了Rhamba，一个区域感知预训练框架，它将解剖学引导的掩码策略与混合Attention-Mamba架构相结合，用于静息态功能磁共振成像（fMRI）分析。模型在ABIDE数据集上使用区域对齐的补丁嵌入以及三种空间特异性递增的掩码策略（Any、Majority和Pure）进行预训练。我们评估了四种架构变体：仅Mamba模型、交替使用Mamba与注意力块的交替架构，以及两种混合编码器-解码器配置（Attention-Mamba (AM) 与 Mamba-Attention (MA)）。预训练模型在下游分类任务上使用COBRE和ADHD-200数据集进行微调，用于精神分裂症和注意缺陷多动障碍的分类。（原文摘要在此处截断）

    arXiv:2605.01240v3 Announce Type: replace-cross  Abstract: Self-supervised pretraining is promising for large-scale neuroimaging, yet the impact of region-aware masking and hybrid sequence modeling remains underexplored. In this work, we introduce Rhamba, a region-aware pretraining framework that integrates anatomically guided masking with hybrid Attention-Mamba architectures for resting state functional magnetic resonance imaging (fMRI) analysis. Models were pretrained on the ABIDE dataset using region-aligned patch embeddings and three masking strategies (Any, Majority, and Pure) with increasing spatial specificity. We evaluated four architectural variants: a Mamba only model, an Alternate architecture with interleaved Mamba and Attention blocks, and two hybrid encoder-decoder configurations (Attention-Mamba (AM) and Mamba-Attention (MA)). The pretrained models were fine-tuned on downstream classification tasks using the COBRE and ADHD-200 datasets for schizophrenia and attention-def
    
[^192]: 大语言模型中合作覆盖回路如何抑制纳什博弈行为

    How a Cooperative-Override Circuit Suppresses Nash Play in Large Language Models

    [https://arxiv.org/abs/2604.27167](https://arxiv.org/abs/2604.27167)

    该论文通过 logit-lens 与因果干预分析发现，大语言模型在囚徒困境中偏离纳什均衡、选择合作的行为源于残差流中一个单一且可因果控制的“合作覆盖”方向，该方向在模型最后几层对决策起决定性作用。

    

    在直接提示下的命名囚徒困境任务中，三个较大的指令微调模型（Llama-3-70B、Qwen2.5-32B 和 Qwen2.5-72B）锁定在完全合作状态，即该指标与纳什均衡的最大距离，且多次重复实验方差为零；而 Llama-3-8B 则以接近纳什均衡的方式博弈。通过打开模型内部，logit-lens 分析发现了一种分布式的合作覆盖机制：在约四分之三的网络深度中，中间层读数倾向于纳什动作，随后在后期向合作方向激增，最终由最后一层决定博弈结果。最终修正的大小（而非激增本身）在不同规模和两种架构上与思维链行为呈秩相关匹配。在 8B 模型中，该覆盖机制对应残差流中一个单一的可因果控制方向：对其进行引导可以调节决策，而在某一层的某一位置钳制其分量会使选择严格单调移动，Spearman rho = 1.000，且生成文本保持流畅。该回路……（摘要截断）

    arXiv:2604.27167v3 Announce Type: replace-cross  Abstract: On the named Prisoner's Dilemma under direct prompting, three larger instruction-tuned models, Llama-3-70B, Qwen2.5-32B, and Qwen2.5-72B, lock at full cooperation, the metric's maximum distance from Nash with zero variance across replicates, while Llama-3-8B plays near-Nash. Opening the models, a logit-lens analysis finds a distributed cooperative override. Intermediate readouts lean toward the Nash action through roughly three quarters of network depth before a late surge toward cooperation, and the final layer settles the contest. The size of that final correction, not the surge, rank-matches chain-of-thought behavior across scale and two architectures. In the 8B the override is a single causally controllable direction in the residual stream; steering it dials the decision, and clamping its component at one position of one layer moves the choice strictly monotonically, Spearman rho = 1.000, with generation fluent. The circuit
    
[^193]: 基于诊断引导的纵向建模用于预测视网膜萎缩进展

    Diagnostic-Guided Longitudinal Modeling for Forecasting Retinal Atrophy Progression

    [https://arxiv.org/abs/2604.16955](https://arxiv.org/abs/2604.16955)

    该论文提出一种任务自适应诊断方法，用于在纵向成像预测中实证判断应选择随机还是确定性生成模型，并将其应用于眼底自发荧光图像预测以预测视网膜萎缩进展。

    

    随机生成模型在纵向成像中的应用日益增多，但当可预测的疾病相关变化相对于技术变异性较小时，其增加的复杂性可能带来的收益有限。我们将模型类别选择（随机 vs 确定性）视为由任务自适应诊断决定的实证步骤。对于一个受不规则随访、采集变异性和设备异质性干扰的纵向图像预测任务，该诊断用于判断随访间的图像变化是由时间相关的疾病进展信号驱动，还是由时间无关的采集变异性驱动。如果随机方法无法产生有用的预测多样性，则选择更简洁的确定性模型类别。我们将该策略应用于眼底自发荧光（FAF）未来图像预测。应用于一个异质性的Optos FAF图像档案（来自9,708只眼的24,335张图像），诊断表明……

    arXiv:2604.16955v3 Announce Type: replace-cross  Abstract: Stochastic generative models are increasingly used for longitudinal imaging, but their added complexity may provide limited benefit when predictable disease-related change is small relative to technical variability. We treat model-class selection (stochastic vs deterministic) as an empirical step determined by a task-adaptive diagnostic. For a longitudinal image prediction task complicated by irregular follow-up, acquisition variability, and device heterogeneity, the diagnostic asks whether inter-visit image change is driven by time-dependent disease progression signals or time-independent acquisition variability. If a stochastic approach fails to yield useful predictive diversity, a more parsimonious, deterministic model class is selected. We applied this strategy to fundus autofluorescence (FAF) future image prediction. Applied to a heterogeneous Optos FAF archive (24,335 images from 9,708 eyes), the diagnostic indicated that
    
[^194]: 训练前的表示：生成式医疗事件模型分词的实用基准

    Representation Before Training: A Practical Benchmark for Generative Medical Event Model Tokenization

    [https://arxiv.org/abs/2604.16775](https://arxiv.org/abs/2604.16775)

    该研究对生成式医疗事件模型的分词方案进行了大规模系统性基准测试（156个模型），发现将医疗代码与检验数值十分位数融合的 token 表示在所有八类住院结局预测任务中均能带来性能提升。

    

    生成式医疗事件模型使用分词后的患者时间线序列作为输入，但围绕分词的众多决策目前缺乏实践指导。我们对多种分词方案进行了基准测试，包括量化粒度、参考范围锚定、代码-数值融合、数值与时间编码，以及来自专家映射的通用数据模型的原生表示与协调表示。我们采用 Llama 和 Qwen 两种架构，从三个初始化种子共训练了156个模型，每种配置遵循相同的训练方案，最多训练五个 epoch。我们使用线性探针对住院前24小时内学习到的表示进行评估，以预测第24-48小时期间的二分类和连续型结局。结果表明，将医疗代码与数值十分位数配对的融合 token，在所有八类结局预测任务中均优于对应的非融合分词输入，其 AUROC 提升为 +（摘要在此处截断）

    arXiv:2604.16775v2 Announce Type: replace  Abstract: Generative medical event models use tokenized sequences of patient timelines as input, but practical guidance on the many decisions around tokenization is limited. We benchmark quantization granularity, reference-range anchoring, code--value fusion, numeric and temporal encodings, and native versus harmonized event representations from an expert-mapped common data model. Using both Llama and Qwen architectures, 156 models were trained from three initialization seeds, with each configuration following a shared training recipe for up to five epochs. We evaluated learned representations from the first 24 hours of hospitalization with linear probes to predict binary and continuous outcomes during hours 24-48. Fused tokens pairing codes with value deciles increased performance across all eight outcome families relative to the equivalent unfused tokenized input with area under the receiver operating characteristic curve (AUROC) gains of $+
    
[^195]: 稳定性增强的高斯过程变分自编码器

    Stability Enhanced Gaussian Process Variational Autoencoders

    [https://arxiv.org/abs/2604.09331](https://arxiv.org/abs/2604.09331)

    本文提出稳定性增强的高斯过程变分自编码器（SEGP-VAE），通过从LTI系统定义推导先验并采用将搜索空间限制在半收缩系统集合内的完整无约束参数化，实现了利用高维视频数据间接、稳定且无约束地训练低维线性时不变系统。

    

    本文提出了一种新颖的稳定性增强高斯过程变分自编码器（SEGP-VAE），用于利用高维视频数据间接训练低维线性时不变（LTI）系统。该新型SEGP先验的均值函数和协方差函数由LTI系统的定义推导而来，使SEGP能够借助概率模型与可解释物理模型相结合的方式捕捉间接观测到的潜在过程。通过一种完整且无约束的参数化方法，LTI参数的搜索空间被限制在半收缩系统集合内。因此，SEGP-VAE可以使用无约束优化算法进行训练。此外，这种参数化方法避免了非Hurwitz状态矩阵的存在所引起的数值问题。案例研究将SEGP-VAE应用于一个包含螺旋运动粒子视频的数据集，突出了该方法的优势以及面向特定应用的设计。

    arXiv:2604.09331v2 Announce Type: replace  Abstract: A novel stability-enhanced Gaussian process variational autoencoder (SEGP-VAE) is proposed for indirectly training a low-dimensional linear time invariant (LTI) system, using high-dimensional video data. The mean and covariance function of the novel SEGP prior are derived from the definition of an LTI system, enabling the SEGP to capture the indirectly observed latent process using a combined probabilistic and interpretable physical model. The search space of LTI parameters is restricted to the set of semi-contracting systems via a complete and unconstrained parametrisation. As a result, the SEGP-VAE can be trained using unconstrained optimisation algorithms. Furthermore, this parametrisation prevents numerical issues caused by the presence of a non-Hurwitz state matrix. A case study applies SEGP-VAE to a dataset containing videos of spiralling particles. This highlights the benefits of the approach and the application-specific desig
    
[^196]: 基于条件归一化流的摊销滤波与平滑

    Amortized Filtering and Smoothing with Conditional Normalizing Flows

    [https://arxiv.org/abs/2604.07169](https://arxiv.org/abs/2604.07169)

    提出了一种基于条件归一化流的摊销贝叶斯滤波与平滑框架，通过联合训练共享循环摘要网络与两个条件归一化流，使训练好的模型能在不同观测序列和同化时间上复用，从而避免了每个同化步骤中重新估计分数或构建传输映射的开销。

    

    贝叶斯滤波与平滑是非线性动力系统数据同化的核心。深度生成模型的最新进展为相关的非高斯后验分布提供了灵活的近似。然而，现有的若干方法需要在每个同化步骤重新估计分数或构建传输映射。我们提出了一种用于滤波与平滑的摊销框架，该框架可在不同观测序列和同化时间上复用已训练的条件模型。该框架从模拟的状态和观测轨迹中联合学习一个共享的循环摘要网络和两个条件归一化流。循环网络通过固定维度的摘要来表示每个观测历史，该摘要为滤波近似提供条件，并与下一状态一起为后向转移近似提供条件。信息论分析表明，在马尔可夫（性质）条件下……（原文摘要到此截断）

    arXiv:2604.07169v3 Announce Type: replace-cross  Abstract: Bayesian filtering and smoothing are central to data assimilation in nonlinear dynamical systems. Recent advances in deep generative models provide flexible approximations of the associated non-Gaussian posterior distributions. However, several existing approaches require the score to be re-estimated or a transport map to be constructed at each assimilation step. We propose an amortized framework for filtering and smoothing that reuses trained conditional models across observation sequences and assimilation times. The framework jointly learns a shared recurrent summary network and two conditional normalizing flows from simulated state and observation trajectories. The recurrent network represents each observation history by a fixed-dimensional summary that conditions the filtering approximation and, together with the next state, the backward transition approximation. An information-theoretic analysis shows that, under the Marko
    
[^197]: 转录组学免疫治疗反应预测模型的跨队列泛化能力有限

    Transcriptomic Models for Immunotherapy Response Prediction Show Limited Cross-cohort Generalisability

    [https://arxiv.org/abs/2604.05478](https://arxiv.org/abs/2604.05478)

    本研究系统评估了九种最先进的转录组学免疫治疗反应预测模型，发现它们在独立外部数据集上的预测性能有限，表明现有模型的跨队列泛化能力不足，需要在临床应用前进一步改进。

    

    免疫检查点抑制剂（ICIs）已经改变了癌症治疗方法；然而，相当大比例的患者表现出原发耐药或获得性耐药，使得准确的治疗前反应预测成为一个关键的未满足的临床需求。基于bulk和单细胞RNA测序（scRNA-seq）数据开发的转录组学生物标志物为捕获肿瘤-免疫相互作用提供了有前景的途径，然而现有预测模型的跨队列泛化能力仍不清楚。我们系统地基准测试了九种最先进的转录组学ICI反应预测模型，包括五种基于bulk RNA-seq的模型（COMPASS、IRNet、NetBio、IKCScore和TNBC-ICI）和四种基于scRNA-seq的模型（PRECISE、DeepGeneX、Tres和scCURE），测试使用了模型开发阶段未见的公开独立数据集。总体而言，预测性能表现一般：bulk RNA-seq模型在大多数队列中的表现接近随机水平，而scRNA-seq模型仅显示出...

    arXiv:2604.05478v4 Announce Type: replace-cross  Abstract: Immune checkpoint inhibitors (ICIs) have transformed cancer therapy; yet substantial proportion of patients exhibit intrinsic or acquired resistance, making accurate pre-treatment response prediction a critical unmet need. Transcriptomics-based biomarkers derived from bulk and single-cell RNA sequencing (scRNA-seq) offer a promising avenue for capturing tumour-immune interactions, yet the cross-cohort generalisability of existing prediction models remains unclear.We systematically benchmark nine state-of-the-art transcriptomic ICI response predictors, five bulk RNA-seq-based models (COMPASS, IRNet, NetBio, IKCScore, and TNBC-ICI) and four scRNA-seq-based models (PRECISE, DeepGeneX, Tres and scCURE), using publicly available independent datasets unseen during model development. Overall, predictive performance was modest: bulk RNA-seq models performed at or near chance level across most cohorts, while scRNA-seq models showed only
    
[^198]: 基于多重偏好预言机的离线约束式人类反馈强化学习

    Offline Constrained RLHF with Multiple Preference Oracles

    [https://arxiv.org/abs/2604.00200](https://arxiv.org/abs/2604.00200)

    该论文提出了仅通过对偶求解的算法来处理多重偏好预言机下的离线约束式人类反馈强化学习，首次给出了有限样本性能保证，确保高概率满足约束，并可扩展至多约束和一般f-散度正则化情形。

    

    我们研究了具有多重偏好预言机的离线约束式人类反馈强化学习。受需要在性能与安全性或公平性之间进行权衡的应用启发，我们的目标是在满足受保护群体最低福利约束的前提下，最大化目标群体的效用。基于在参考策略下收集的成对比较数据，我们通过最大似然估计各预言机特定的奖励，并分析统计不确定性如何通过对偶问题传播。我们将约束目标表述为KL正则化的拉格朗日形式，其原始最优解为Gibbs策略，从而将学习问题转化为凸对偶问题。我们提出了一种仅通过对偶求解的算法，确保以高概率满足约束，并为离线约束偏好学习提供了首个有限样本性能保证。最后，我们将理论分析扩展到多约束情形以及一般的f-散度正则化。

    arXiv:2604.00200v2 Announce Type: replace  Abstract: We study offline constrained reinforcement learning from human feedback with multiple preference oracles. Motivated by applications that trade off performance with safety or fairness, we aim to maximize target population utility subject to a minimum protected group welfare constraint. From pairwise comparisons collected under a reference policy, we estimate oracle-specific rewards via maximum likelihood and analyze how statistical uncertainty propagates through the dual program. We cast the constrained objective as a KL-regularized Lagrangian whose primal optimizer is a Gibbs policy, reducing learning to a convex dual problem. We propose a dual-only algorithm that ensures high-probability constraint satisfaction and provide the first finite-sample performance guarantees for offline constrained preference learning. Finally, we extend our theoretical analysis to accommodate multiple constraints and general f-divergence regularization.
    
[^199]: 面向稀疏数据的分量级L1范数非负矩阵分解

    Nonnegative Matrix Factorization in the Component-Wise L1 Norm for Sparse Data

    [https://arxiv.org/abs/2603.29715](https://arxiv.org/abs/2603.29715)

    本文首次证明了L1-NMF即使在最简单的r=1情况下也是NP难问题，并从理论上揭示了数据稀疏性与L1-NMF因子稀疏性之间的内在联系。

    

    非负矩阵分解（NMF）通过两个非负因子的乘积WH来近似非负矩阵X，其中W有r列，H有r行。本文研究了使用分量级L1范数作为误差度量的NMF（L1-NMF），该方法适用于受重尾噪声（如拉普拉斯噪声或椒盐噪声）污染的数据，或存在离群值的情况。我们的第一个贡献是证明了L1-NMF是NP难问题，即使当r=1时也是如此，这与使用最小二乘的标准NMF形成鲜明对比。我们的第二个贡献是在简化的概率假设下，分析了数据中的稀疏性如何在保持其他所有条目固定的情况下，强制L1-NMF因子的最优标量更新为零解。这为L1-NMF因子的稀疏性与输入数据的稀疏性之间的联系提供了直观理解。尽管稀疏性有利于可解释性，但如果数据受到影响……

    arXiv:2603.29715v2 Announce Type: replace  Abstract: Nonnegative matrix factorization (NMF) approximates a nonnegative matrix, X, by the product of two nonnegative factors, WH, where W has r columns and H has r rows. In this paper, we consider NMF using the component-wise L1 norm as the error measure (L1-NMF), which is suited for data corrupted by heavy-tailed noise, such as Laplace noise or salt and pepper noise, or in the presence of outliers. Our first contribution is an NP-hardness proof for L1-NMF, even when r=1, in contrast to the standard NMF that uses least squares. Our second contribution is to analyze, under simplified probabilistic assumptions, how the sparsity in the data enforces zero solution in the optimal scalar update in the factors of L1-NMF when all the other entries are kept fixed. This provides an intuition of the connection between the sparsity of the L1-NMF factors with the sparsity of the input. Even though sparsity favors interpretability, if the data is affect
    
[^200]: 具有不确定性量化的代理LPV状态空间模型学习

    Learning Surrogate LPV State-Space Models with Uncertainty Quantification

    [https://arxiv.org/abs/2603.29532](https://arxiv.org/abs/2603.29532)

    本文提出一种贝叶斯方法，可直接从输入输出数据中联合估计LPV状态空间模型及其调度映射，并同时量化偶然不确定性与认知不确定性，为预测模型响应提供置信界。

    

    线性参数变化（LPV）框架能够构建复杂非线性、高维系统的代理模型，从而支持高效的稳定性与性能分析以及控制器设计。尽管数据驱动的LPV建模已取得显著进展，但现有方法并未对所获得的LPV模型的不确定性进行量化。因此，评估模型在分析与控制中的可靠性，或检测超出训练范围的操作，需要大量的验证工作和用户专业知识。本文提出了一种贝叶斯方法，可直接从输入输出数据中联合估计LPV状态空间模型（包括其调度映射），同时对模型不确定性进行表征，并对预测的模型响应给出置信界。该方法同时考虑了由测量噪声引起的偶然不确定性，以及由有限训练数据和结构偏差导致的认知不确定性。

    arXiv:2603.29532v2 Announce Type: replace-cross  Abstract: The Linear Parameter-Varying (LPV) framework enables the construction of surrogate models of complex nonlinear and high-dimensional systems, facilitating efficient stability and performance analysis together with controller design. Despite significant advances in data-driven LPV modelling, existing approaches do not quantify the uncertainty of the obtained LPV models. Consequently, assessing model reliability for analysis and control or detecting operation outside the training regime requires extensive validation and user expertise. This paper proposes a Bayesian approach for the joint estimation of LPV state-space models, including their scheduling map, together with characterization of the model uncertainty and confidence bounds on the predicted model response directly from input-output data. Both aleatoric uncertainty due to measurement noise and epistemic uncertainty arising from limited training data and structural bias ar
    
[^201]: 具有对抗性奖励的线性混合约束马尔可夫决策过程学习的近最优原始-对偶算法

    Near-Optimal Primal-Dual Algorithm for Learning Linear Mixture CMDPs with Adversarial Rewards

    [https://arxiv.org/abs/2603.27884](https://arxiv.org/abs/2603.27884)

    本文提出了首个针对具有对抗性奖励的线性混合约束马尔可夫决策过程（CMDP）的可证明高效算法，通过引入正则化对偶更新实现基于漂移的分析，达到了近最优的 $\widetilde{O}(\sqrt{d^2 H^3 K})$ 遗憾和约束违反界限。

    

    我们研究了在全信息反馈和未知转移核下，具有对抗性奖励的有限时域线性混合约束马尔可夫决策过程（CMDP）中的安全强化学习问题。我们提出了一种原始-对偶策略优化算法，在温和条件下实现了 $\widetilde{O}(\sqrt{d^2 H^3 K})$ 的遗憾和约束违反界限，其中 $d$ 是特征维度，$H$ 是时域长度，$K$ 是回合数。据我们所知，这是首个针对具有对抗性奖励的线性混合CMDP的可证明高效算法。特别地，我们的遗憾界限是近最优的，在对数因子范围内匹配了已知的极小极大下界。其关键思想是引入正则化的对偶更新，从而使基于漂移的分析成为可能。这一步至关重要，因为当奖励函数在不同回合之间变化时，基于强对偶性的分析无法直接应用。

    arXiv:2603.27884v2 Announce Type: replace  Abstract: We study safe reinforcement learning in finite-horizon linear mixture constrained Markov decision processes (CMDPs) with adversarial rewards under full-information feedback and an unknown transition kernel. We propose a primal-dual policy optimization algorithm that achieves regret and constraint violation bounds of $\widetilde{O}(\sqrt{d^2 H^3 K})$ under mild conditions, where $d$ is the feature dimension, $H$ is the horizon, and $K$ is the number of episodes. To the best of our knowledge, this is the first provably efficient algorithm for linear mixture CMDPs with adversarial rewards. In particular, our regret bound is near-optimal, matching the known minimax lower bound up to logarithmic factors. The key idea is to introduce a regularized dual update that enables a drift-based analysis. This step is essential, as strong duality-based analysis cannot be directly applied when reward functions change across episodes. In addition, we 
    
[^202]: 语言模型利用置信度驱动行为的因果证据

    Causal Evidence that Language Models use Confidence to Drive Behavior

    [https://arxiv.org/abs/2603.22161](https://arxiv.org/abs/2603.22161)

    该研究通过四阶段实验范式提供了因果证据，证明语言模型确实利用内部置信度信号来驱动行为（如决定是否回答或弃权），激活引导实验显示增强或抑制置信度信号会相应地降低或提高弃权率。

    

    元认知——评估自身认知表现的质量——在多个物种中引导适应性行为。大量研究表明，可以从语言模型的输出中提取置信度信号，但一个根本问题仍然存在：模型是否真的使用这些信号来控制行为，例如决定是否回答或弃权？为了研究这个问题，我们开发了一个四阶段实验范式。阶段1在不提供弃权选项的情况下引出基线置信度估计。阶段2揭示了大型语言模型（LLM）在决定弃权时会对内部置信度应用一个隐式阈值，且置信度的效应量比其他替代机制大约大一个数量级。阶段3通过激活引导技术提供了直接的因果证据：增强或抑制置信度信号会相应地降低或提高弃权率。阶段4进一步通过系统地改变指令阈值来扩展这一发现（原文在此处截断）。

    arXiv:2603.22161v3 Announce Type: replace  Abstract: Metacognition -- assessing the quality of one's own cognitive performance -- guides adaptive behavior across species. Substantial research demonstrates that confidence signals can be extracted from language model outputs, yet a fundamental question remains: do models actually use these signals to control behavior, such as deciding whether to answer or abstain? To investigate, we developed a four-phase paradigm. Phase~1 elicited baseline confidence estimates without an abstention option. Phase~2 revealed that LLMs apply an implicit threshold to internal confidence when deciding to abstain, with confidence effect sizes approximately an order of magnitude larger than alternative mechanisms. Phase~3 provided direct causal evidence through activation steering: boosting or suppressing confidence signals correspondingly decreased or increased abstention rates. Phase~4 extended this by systematically varying instructed thresholds, demonstrat
    
[^203]: 大语言模型如何计算言语置信度

    How do LLMs Compute Verbal Confidence

    [https://arxiv.org/abs/2603.17839](https://arxiv.org/abs/2603.17839)

    大语言模型在生成答案时会自动计算并缓存置信度表征，之后在需要口头表达置信度时再进行检索，而非在被询问时才即时计算。

    

    言语置信度——即提示大语言模型以数字或类别的形式陈述其置信度——被广泛用于从黑盒模型中提取不确定性估计。然而，大语言模型内部如何生成这些分数仍不得而知。我们研究了两个问题：第一，置信度是何时计算的——是在被请求时即时计算，还是在答案生成过程中自动计算并缓存以供后续检索；第二，言语置信度代表什么——是词元对数概率，还是对答案质量的更丰富评估？我们聚焦于 Gemma 3 27B（在 TriviaQA、BigMath 和 MMLU 上评估）、Qwen 2.5 7B 以及推理模型 Magistral Small 24B，为“缓存检索”机制提供了汇聚性证据。激活引导、修补、加噪和交换实验表明，置信度表征会先在答案相邻位置出现，然后才出现在言语化位置。注意力阻断实验精确定位了信息流：置信度从……（原文摘要在此处截断）

    arXiv:2603.17839v4 Announce Type: replace-cross  Abstract: Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed -- just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents -- token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B (across TriviaQA, BigMath, and MMLU), Qwen 2.5 7B, and the reasoning model Magistral Small 24B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from
    
[^204]: 驯服对手：一种基于代价-扰动比率的对抗性强化学习方法

    Taming the Adversary: A Cost-to-Disturbance Ratio Approach to Adversarial Reinforcement Learning

    [https://arxiv.org/abs/2603.12110](https://arxiv.org/abs/2603.12110)

    提出CoDRA框架，将控制器与对手的权衡表示为累积代价与累积平方扰动范数之比，并通过自归一化的actor-critic更新进行优化，从而在提升鲁棒性的同时避免名义性能崩溃。

    

    在仿真中训练的强化学习（RL）策略一旦部署到真实系统上，性能往往会下降，因为控制器必须抑制在仿真中从未遇到过的外部扰动。鲁棒强化学习通过在学习过程中让控制器暴露于扰动来解决这一问题，具体方法包括域随机化、对抗性极小极大公式化，或主角行为与对抗行为的概率混合。然而，不受调节的扰动机制会使训练不稳定，并且相对于标准的非鲁棒方法，往往会导致名义性能崩溃。我们提出了代价-扰动比对抗训练，这是一个将控制器与对手之间的权衡表示为累积代价与累积平方扰动范数之比的框架，并通过自归一化的actor-critic更新对其进行优化。在该算法中，每个价值项都由一个停止梯度的归一化常数进行缩放，该常数由……计算得出（摘要在此处被截断）。

    arXiv:2603.12110v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) policies trained in simulation often degrade once deployed on real systems, where the controller must reject external disturbances that were never encountered in simulation. Robust RL addresses this by exposing the controller to perturbations while it learns, through domain randomization, adversarial minimax formulations, or probabilistic mixtures of protagonist and adversarial behavior. However, an unregulated disturbance mechanism destabilizes training and often collapses nominal performance relative to standard, non-robust methods. We propose cost-to-disturbance ratio adversarial training (CoDRA), a framework that expresses the controller--adversary trade-off as a ratio of accumulated cost to accumulated squared disturbance norm, and optimizes it through a self-normalized actor--critic update. In this algorithm, each value term is scaled by a stop-gradient normalization constant computed from the 
    
[^205]: 数据驱动的积分核用于可解释的非局部算子学习

    Data-Driven Integration Kernels for Interpretable Nonlocal Operator Learning

    [https://arxiv.org/abs/2603.10305](https://arxiv.org/abs/2603.10305)

    该论文提出数据驱动的积分核框架，通过将非局部信息聚合与局部非线性预测显式分离，使气候过程的非局部算子学习更具可解释性并降低过拟合风险。

    

    机器学习模型可以表示在水平空间、高度和时间上非局部的气候过程，通常通过以高度非线性的方式组合这些维度的信息。虽然这可以提高预测技巧，但随着非局部信息范围的扩大，学习到的关系会变得难以解释且容易过拟合。我们通过引入数据驱动的积分核来应对这一挑战，这是一个为非局部算子学习增加结构的框架，它显式地将非局部信息聚合与局部非线性预测分离开来。每个时空预测因子场首先使用可学习的核（定义为在水平空间、高度和/或时间上的连续加权函数）进行积分，随后仅对由此得到的核积分特征以及可选的局部输入应用局部非线性映射。这种设计将非线性相互作用限制在一小组积分（摘要在此处被截断）

    arXiv:2603.10305v4 Announce Type: replace  Abstract: Machine learning models can represent climate processes that are nonlocal in horizontal space, height, and time, often by combining information across these dimensions in highly nonlinear ways. While this can improve predictive skill, it makes learned relationships difficult to interpret and prone to overfitting as the extent of nonlocal information grows. We address this challenge by introducing data-driven integration kernels, a framework that adds structure to nonlocal operator learning by explicitly separating nonlocal information aggregation from local nonlinear prediction. Each spatiotemporal predictor field is first integrated using learnable kernels (defined as continuous weighting functions over horizontal space, height, and/or time), after which a local nonlinear mapping is applied only to the resulting kernel-integrated features and optional local inputs. This design confines nonlinear interactions to a small set of integr
    
[^206]: 神经集合卡尔曼滤波器：针对含激波可压缩流的数据同化

    Neural ensemble Kalman filter: Data assimilation for compressible flows with shocks

    [https://arxiv.org/abs/2602.23461](https://arxiv.org/abs/2602.23461)

    提出神经集合卡尔曼滤波器，通过将含激波流动的预报集合映射到深度神经网络的参数空间并在该空间中执行数据同化，解决了经典EnKF在不确定激波附近因双峰预报分布违反高斯假设而性能不佳的问题。

    

    arXiv:2602.23461v3 公告类型：replace-cross 摘要：针对含激波的可压缩流进行数据同化（DA）具有挑战性，因为许多经典的数据同化方法会在不确定的激波附近产生虚假振荡和非物理特征。本文聚焦于集合卡尔曼滤波器（EnKF）。我们证明EnKF性能不佳的原因可能在于不确定激波位置附近会出现双峰预报分布；这违反了EnKF所依赖的基本假设，即假设预报分布接近高斯分布。为解决这一问题，我们提出了新的神经集合卡尔曼滤波器（neural EnKF）。其基本思想是将神经函数逼近系统地嵌入到集合数据同化中，通过将含激波流动的预报集合映射到深度神经网络（NN）的参数空间（权重和偏置），随后在该空间中执行数据同化。这种非线性映射将尖锐和光滑的流动特征编码到一组神经网络参数中。神经EnKF的更新...

    arXiv:2602.23461v3 Announce Type: replace-cross  Abstract: Data assimilation (DA) for compressible flows with shocks is challenging because many classical DA methods generate spurious oscillations and nonphysical features near uncertain shocks. We focus here on the ensemble Kalman filter (EnKF). We show that the poor performance of the EnKF may be attributed to the bimodal forecast distribution that can arise in the vicinity of an uncertain shock location; this violates the assumptions underpinning the EnKF, which assume a forecast which is close to Gaussian. To address this issue we introduce the new neural EnKF. The basic idea is to systematically embed neural function approximations within ensemble DA by mapping the forecast ensemble of shocked flows to the parameter space (weights and biases) of a deep neural network (NN) and to subsequently perform DA in that space. The nonlinear mapping encodes sharp and smooth flow features in an ensemble of NN parameters. Neural EnKF updates ar
    
[^207]: 立场：推进时间序列建模需要动力学系统的视角

    Position: A Dynamical Systems Perspective is Needed to Advance Time Series Modeling

    [https://arxiv.org/abs/2602.16864](https://arxiv.org/abs/2602.16864)

    该论文提出时间序列建模领域应引入动力学系统视角，通过动力学系统重构（DSR）方法从数据中推断潜在动力学系统的替代模型，从而不仅实现短期预测，还能预测观测系统的长期统计特性。

    

    时间序列（TS）建模已经从早期的统计方法（主要是线性方法）走过了漫长的道路，发展到了当前时间序列基础模型的趋势。尽管该领域存在大量炒作和工业需求，但并不总是清楚究竟取得了多少真正的进展。为了将时间序列预测和分析提升到下一个水平，我们认为该领域需要动力学系统（DS）的视角。来自自然或工程系统的观测时间序列几乎总是源于某种潜在的动力学系统，可以说，获取其控制方程将产生理论上最优的预测。这正是动力学系统重构（DSR）的前景所在，这是一类旨在从数据中推断潜在动力学系统替代模型的机器学习/人工智能方法。但基于动力学系统原理的模型还提供了其他深远的优势：除了短期预测之外，它们还能够预测观测系统的长期统计特性，在许多实际场景中这可能是更…

    arXiv:2602.16864v3 Announce Type: replace-cross  Abstract: Time series (TS) modeling has come a long way from early statistical, mainly linear, approaches to the current trend in TS foundation models. With a lot of hype and industrial demand in this field, it is not always clear how much progress there really is. To advance TS forecasting and analysis to the next level, here we argue that the field needs a dynamical systems (DS) perspective. TS of observations from natural or engineered systems almost always originate from some underlying DS, and arguably access to its governing equations would yield theoretically optimal forecasts. This is the promise of DS reconstruction (DSR), a class of ML/AI approaches that aim to infer surrogate models of the underlying DS from data. But models based on DS principles offer other profound advantages: Beyond short-term forecasts, they enable to predict the long-term statistics of an observed system, which in many practical scenarios may be the more
    
[^208]: 梯度稳定的注意力头可指示大语言模型答案的正确性

    Gradient-Stable Attention Heads Signal LLM Correctness

    [https://arxiv.org/abs/2602.13699](https://arxiv.org/abs/2602.13699)

    提出免训练方法HeadEntropy，利用softmax雅可比迹与2-Rényi熵的单调关系，通过测量注意力头对梯度更新的稳定性来预测LLM答案的正确性，达到0.736 AUROC并超越所有免训练基线。

    

    大语言模型（LLMs）经常生成看似合理却不正确的答案，这在医学等安全关键场景中构成了风险。人工评估成本高昂，而“以LLM作为评判者”的方法则可能引入隐藏错误。近来的单次前向白盒方法通过在模型内部使用线性探针来检测上下文幻觉，但其在域外、任务外以及免训练设置下的泛化能力仍缺乏充分理解。我们提出了HeadEntropy，该方法通过测量每个注意力头的模式在进一步梯度更新下被修改的敏感程度来预测答案的正确性。其关键洞察在于：softmax雅可比矩阵的迹是2-Rényi熵的单调函数，这将注意力的分散程度与训练过程中的梯度稳定性联系起来，且无需参考标注。在无需任何训练的情况下，HeadEntropy达到了0.736的AUROC，在成对比较中优于所有免训练基线方法。

    arXiv:2602.13699v2 Announce Type: replace  Abstract: Large language models (LLMs) often generate plausible yet incorrect answers, posing risks in safety-critical settings such as medicine. Human evaluation is expensive, and LLM-as-judge approaches risk introducing hidden errors. Recent single-pass white-box methods detect contextual hallucinations using linear probes over model internals, but their generalization to out-of-domain, out-of-task, and training-free settings remains poorly understood. We introduce HeadEntropy, a method that predicts answer correctness by measuring how susceptible each attention head's pattern is to modification under further gradient updates. The key insight is that the trace of the softmax Jacobian is a monotonic function of 2-Renyi entropy, linking attention spread to gradient stability during training, even without the reference annotation. With no training, HeadEntropy reaches 0.736 AUROC, outperforms every training-free baseline in paired comparison an
    
[^209]: 面向非负Einsum分解的近通用乘法更新算法

    Near-Universal Multiplicative Updates for Nonnegative Einsum Factorization

    [https://arxiv.org/abs/2602.02759](https://arxiv.org/abs/2602.02759)

    NNEinFact是一种基于einsum的乘法更新算法，用户仅需一个字符串即可指定并拟合任意可表示为张量收缩的非负张量分解模型，该算法收敛于损失驻点、支持缺失数据，并能在数秒内处理数亿规模条目的张量。

    

    尽管多路数据在科学领域中无处不在，但目前缺乏既高性能又易于使用的方法来拟合针对特定数据定制的非标准非负张量分解模型。研究人员可以使用基于梯度的自动微分方法，但该方法在非负约束下往往表现不佳；也可以在少数几种具有成熟实现的方法中进行选择；或者从零开始实现自己的模型。作为替代方案，我们提出了NNEinFact，这是一种基于einsum的乘法更新算法，它可以通过最小化多种用户指定的损失函数（包括$(\alpha,\beta)$-散度）来拟合任何可表示为张量收缩的非负张量分解。使用NNEinFact时，研究人员只需用一个字符串来指定其模型。NNEinFact收敛于损失的驻点，支持缺失数据，并能在数秒内拟合包含数亿个条目的张量。从实证角度来看，NN……

    arXiv:2602.02759v3 Announce Type: replace-cross  Abstract: Despite the ubiquity of multiway data across scientific domains, there are few performant and user-friendly methods that fit non-standard nonnegative tensor factorization models tailored to the data at-hand. Researchers may use gradient-based automatic differentiation, which often struggles under nonnegative constraints, choose between a limited set of methods with mature implementations, or implement their own model from scratch. As an alternative, we introduce NNEinFact, an einsum-based multiplicative update algorithm that fits any nonnegative tensor factorization expressible as a tensor contraction by minimizing one of many user-specified loss functions, including the $(\alpha,\beta)$-divergence. To use NNEinFact, the researcher specifies their model with a string. NNEinFact converges to a stationary point of the loss, supports missing data, and fits to tensors with hundreds of millions of entries in seconds. Empirically, NN
    
[^210]: 当Token置信度失效时语义校准更胜一筹：长篇科学问答基准测试

    Semantic Calibration Prevails Where Token Confidence Fails: Benchmarking Long-Form Scientific QA

    [https://arxiv.org/abs/2602.00279](https://arxiv.org/abs/2602.00279)

    该论文推出了首个面向长篇科学问答的不确定性量化校准大规模基准，发现指令微调会导致token概率极化、削弱token级置信度信号的可靠性，而语义校准方法在此场景下表现更优。

    

    可靠的不确定性量化（UQ）对于大语言模型（LLM）在科学问答中的安全部署至关重要，因为长篇输出超出了大规模人工验证的实际可行性。我们推出了首个针对长篇、需要推理的科学问答中UQ校准的大规模基准，在多达20个大语言模型和七个数据集上对四种UQ方法共685,000条响应进行了评估，并由一个可扩展的开源框架提供支持，其共享生成设计使跨方法比较具有可复现性。研究表明，指令微调与系统性的token概率极化相关，导致置信度分布坍缩，从而削弱了token级不确定性信号的可靠性。推理模型家族呈现分化：一些模型重现了这种极化，而另一些则主动缓解了它，这一模式按提供商聚类，表明训练流程设计是关键的差异化因素。

    arXiv:2602.00279v2 Announce Type: replace  Abstract: Reliable uncertainty quantification (UQ) is essential for safe deployment of large language models (LLMs) in scientific question answering, where long-form outputs exceed practical human verification at scale. We introduce the first large-scale benchmark for UQ calibration in long-form, reasoning-demanding scientific QA, evaluating four UQ methods on 685,000 responses across up to 20 LLMs and seven datasets, supported by an extensible open-source framework whose shared-generation design enables reproducible cross-method comparisons. Instruction tuning is shown to associate with systematic token probability polarization, collapsing confidence distributions and undermining the reliability of token-level uncertainty signals. Reasoning model families diverge: some reproduce this polarization while others actively mitigate it, a pattern that clusters by provider and suggests training pipeline design as a key differentiating factor. Verbal
    
[^211]: CORDS：离散结构的连续表示

    CORDS: Continuous Representations of Discrete Structures

    [https://arxiv.org/abs/2601.21583](https://arxiv.org/abs/2601.21583)

    CORDS通过可逆映射将数量未知的离散对象集合转换为连续的密度场和特征场，使模型能够在连续场空间中进行推断并精确解码回离散集合，从而解决了可变大小集合预测的难题。

    

    许多学习问题需要在对象数量事先未知的情况下预测对象集合，例如目标检测、分子建模以及天体物理源检测等科学推断任务。现有方法通常依赖于填充表示，或者必须显式推断集合大小，这往往带来挑战。我们提出了一种应对这一挑战的新策略，将可变大小集合的预测转化为连续推断问题。我们的方法CORDS（离散结构的连续表示）提供了一个可逆映射，将一组空间对象转换为连续场：一个编码对象位置和数量的密度场，以及一个在同一支撑域上携带对象属性的特征场。由于该映射是可逆的，模型可以完全在场空间中运行，同时仍能精确解码回离散集合。我们在……上对CORDS进行了评估。

    arXiv:2601.21583v2 Announce Type: replace  Abstract: Many learning problems require predicting sets of objects when the number of objects is not known beforehand. Examples include object detection, molecular modeling, and scientific inference tasks such as astrophysical source detection. Existing methods often rely on padded representations or must explicitly infer the set size, which often poses challenges. We present a novel strategy for addressing this challenge by casting prediction of variable-sized sets as a continuous inference problem. Our approach, CORDS (Continuous Representations of Discrete Structures), provides an invertible mapping that transforms a set of spatial objects into continuous fields: a density field that encodes object locations and count, and a feature field that carries their attributes over the same support. Because the mapping is invertible, models operate entirely in field space while remaining exactly decodable to discrete sets. We evaluate CORDS across 
    
[^212]: 学习平流：一种用于天气预报的神经半拉格朗日架构

    Learning to Advect: A Neural Semi-Lagrangian Architecture for Weather Forecasting

    [https://arxiv.org/abs/2601.21151](https://arxiv.org/abs/2601.21151)

    该论文提出一种受物理启发的神经半拉格朗日架构，将天气预测中的平流、扩散和反应过程解耦为专门算子，通过球面可微插值实现基于轨迹的输送，使网络能够高效学习长距离大气输送。

    

    机器学习天气预报方法通常采用单体式架构，其中不同的物理机制——如平流、扩散混合、热力学过程和强迫——被隐式地表示在单个大型神经网络中。这对平流而言尤其成问题，因为长距离输送通常需要昂贵的全局交互机制或深层堆叠的局部卷积层。为了解决这一局限，我们引入了一种受物理启发的神经架构，将潜在状态演化分解为专门的平流、扩散和反应算子。其核心组件是神经半拉格朗日算子，通过球面上的可微插值执行基于轨迹的输送，使网络能够同时学习待输送的压缩潜在模态集合及其特征轨迹。大气状态被投影到潜在（摘要在此处被截断）

    arXiv:2601.21151v3 Announce Type: replace  Abstract: Machine-learning approaches to weather forecasting often employ a monolithic architecture in which distinct physical mechanisms, such as advection, diffusive mixing, thermodynamic processes, and forcing, are represented implicitly within a single large neural network. This is particularly problematic for advection, where long-range transport typically requires expensive global interaction mechanisms or deep stacks of local convolutional layers. To address this limitation, we introduce a physics-inspired neural architecture that decomposes latent-state evolution into dedicated advection, diffusion, and reaction operators. Its central component is a Neural Semi-Lagrangian operator that performs trajectory-based transport via differentiable interpolation on the sphere, allowing the network to learn both a compressed set of latent modes to be transported and their characteristic trajectories. The atmospheric state is projected into laten
    
[^213]: 基于欧洲数据的跨国学习用于国家传染病预测

    Cross-Country Learning for National Infectious Disease Forecasting Using European Data

    [https://arxiv.org/abs/2601.20771](https://arxiv.org/abs/2601.20771)

    本文提出一种跨国学习方法，利用多个欧洲国家的时间序列数据训练单一模型来预测目标国家的传染病发病率，并通过塞浦路斯COVID-19病例预测的案例研究证明该方法能借助共享的疫情动态和扩大的训练集提升预测性能。

    

    传染病发病率的准确预测对于公共卫生规划和及时干预至关重要。尽管大多数数据驱动的预测方法主要依赖单一国家的历史数据，但此类数据在长度和变异性方面往往有限，从而限制了机器学习（ML）模型的性能。在本研究中，我们研究了一种用于传染病预测的跨国学习方法，即使用多个国家的时间序列数据训练单一模型，并在目标国家上进行评估。这种设置使模型能够利用各国之间共享的疫情动态，并从扩大的训练集中获益。我们通过对塞浦路斯COVID-19病例预测的案例研究来检验这一方法，使用了欧洲国家的监测数据。我们评估了多个模型，并分析了回溯窗口长度和跨国“数据增强”对多个评估指标的影响。

    arXiv:2601.20771v3 Announce Type: replace-cross  Abstract: Accurate forecasting of infectious disease incidence is critical for public health planning and timely intervention. While most data-driven forecasting approaches rely primarily on historical data from a single country, such data are often limited in length and variability, restricting the performance of machine learning (ML) models. In this work, we investigate a cross-country learning approach for infectious disease forecasting, in which a single model is trained on time series data from multiple countries and evaluated on a country of interest. This setting enables the model to exploit shared epidemic dynamics across countries and to benefit from an enlarged training set. We examine this approach through a case study on COVID-19 case forecasting in Cyprus, using surveillance data of European countries. We evaluate multiple models and analyse the impact of the lookback window length and cross-country 'data augmentation' on mu
    
[^214]: jBOT：自蒸馏涌现出的语义喷注表征聚类

    jBOT: Semantic Jet Representation Clustering Emerges from Self-Distillation

    [https://arxiv.org/abs/2601.11719](https://arxiv.org/abs/2601.11719)

    jBOT 通过结合粒子级与喷注级的自蒸馏预训练，使无标签喷注数据的表示空间中涌现出语义类别聚类，仅需简单的距离度量即可实现异常检测，并可通过微调提升分类性能。

    

    在基础模型训练的背景下，自监督学习是一种无需标签即可学习特征表示的强大预训练方法，它通常能从数据中捕捉通用的底层语义，之后可针对下游任务进行微调。在本工作中，我们提出了 jBOT，一种针对欧洲核子研究中心（CERN）大型强子对撞机喷注数据的基于自蒸馏的预训练方法，该方法将局部粒子级蒸馏与全局喷注级蒸馏相结合，以学习支持异常检测和分类等下游任务的喷注表示。我们观察到，在无标签喷注上进行预训练会在表示空间中涌现出语义类别的聚类现象。当仅在背景喷注上进行预训练时，冻结嵌入中的这种聚类可以通过简单的基于距离的度量实现异常检测，并且学到的嵌入可以经过微调用于分类任务，从而获得性能提升。

    arXiv:2601.11719v4 Announce Type: replace  Abstract: Self-supervised learning, in the context of foundation model training, is a powerful pre-training method for learning feature representations without labels, which often capture generic underlying semantics from the data and can later be fine-tuned for downstream tasks. In this work, we introduce jBOT, a pre-training method based on self-distillation for jet data from the CERN Large Hadron Collider, which combines local particle-level distillation with global jet-level distillation to learn jet representations that support downstream tasks such as anomaly detection and classification. We observe that pre-training on unlabeled jets leads to emergent semantic class clustering in the representation space. The clustering in the frozen embedding, when pre-trained on background jets only, enables anomaly detection via simple distance-based metrics, and the learned embedding can be fine-tuned for classification with improved performance com
    
[^215]: BEAT-Net：注入仿生时空先验以实现可解释的心电图诊断

    BEAT-Net: Injecting Biomimetic Spatio-Temporal Priors for Interpretable ECG Diagnosis

    [https://arxiv.org/abs/2601.07316](https://arxiv.org/abs/2601.07316)

    BEAT-Net通过仿生设计的QRS波标记化和模拟心脏病学家诊断工作流程的分层架构，实现了数据高效、泛化性强且可解释的心电图自动诊断。

    

    基于深度学习的自动化心电图诊断仍然受限于信号无关的表示方法，这些方法将多导联记录视为无差别的时间序列或图像，迫使模型隐式地重新发现生理结构。这导致了数据效率低下、泛化能力差，以及与临床推理不一致的不透明决策边界。我们提出了BEAT-Net，这是一个有监督的仿生框架，它将基于QRS波群的生物标记化与镜像心脏病学家工作流程的分层架构相结合。QRS标记器将连续信号转换为语义完整的心跳序列，这些序列通过四个专门阶段进行处理：通过词编码器进行形态特征提取、通过空间算子实现导联不变的归一化、通过时间算子注入时间上下文，以及使用基于Transformer的句子编码器进行全局推理。（原文摘要在此处截断）

    arXiv:2601.07316v2 Announce Type: replace-cross  Abstract: Automated electrocardiogram diagnosis using deep learning remains limited by signal-agnostic representations that treat multi-lead recordings as undifferentiated time-series or images, forcing models to rediscover physiological structure implicitly. This leads to data inefficiency, poor generalization, and opaque decision boundaries misaligned with clinical reasoning. We present BEAT-Net, a supervised biomimetic framework that integrates QRS-centered biological tokenization with a hierarchical architecture mirroring the cardiologist's workflow. A QRS tokenizer converts continuous signals into semantically complete heartbeat sequences, which are processed through four specialized stages: morphological feature extraction via a Word Encoder, lead-invariant normalization through a Spatial Operator, temporal context injection by a Temporal Operator, and global reasoning using a Transformer-based Sentence Encoder. Evaluated across th
    
[^216]: 大语言模型是香农有损压缩器而非所罗门诺夫归纳估计器：没有程序空间中的符号模型综合，奇点不会临近

    Large Language Models As Shannon Lossy Compressors Not Solomonoff Induction Estimators: The Singularity Is Not Near Without Symbolic Model Synthesis in Program Space

    [https://arxiv.org/abs/2601.05280](https://arxiv.org/abs/2601.05280)

    该论文证明大语言模型基于交叉熵和下一词元目标的训练本质上是香农有损压缩而非所罗门诺夫归纳估计，因此若缺乏程序空间中的符号模型综合，实现真正自我改进的AI奇点不会临近。

    

    一方面，大语言模型（LLM）是否为所罗门诺夫归纳估计器的问题已成为算法信息论（AIT）与机器学习（ML）交叉领域中备受关注的明确问题。另一方面，人工智能奇点这一如今已略显陈旧的理念——即需要一个可靠的正反馈过程，使系统能够生成、评估并保留对自身的真正改进——持续被提及，是AGI讨论中反复出现的概念。我们基于神经符号机器学习的当前假设和未来发展，将这些问题联系起来并提供一些解答。我们将证明，交叉熵、负对数似然及同类的下一词元预测目标本身并不能或无法实现所罗门诺夫归纳：它们优化的是对给定条件分布的拟合，而非程序加权的普适混合。虽然在固定目标内更多的计算……

    arXiv:2601.05280v4 Announce Type: replace-cross  Abstract: On the one hand, the question of whether Large Language Models (LLMs) are Solomonoff induction estimators has become an explicit question at the intersection of Algorithmic Information Theory (AIT) and Machine Learning (ML) of great interest. On the other hand, the now old idea of an AI Singularity that requires a reliable positive-feedback process in which a system can generate, evaluate and retain genuine improvements to itself continues to come up and is a recurrent concept in the discussion of AGI. We connect and provide some answers to these issues based on current assumptions and future developments of neurosymbolic ML. We will demonstrate that cross-entropy, negative log-likelihood and cognate next-token objectives do not or cannot, by themselves, implement Solomonoff induction: they optimise fit to a supplied conditional distribution rather than a program-weighted universal mixture. While more compute within a fixed obj
    
[^217]: 用于非晶粒子系统的玻尔兹曼生成器

    Boltzmann generators for amorphous particle systems

    [https://arxiv.org/abs/2512.16607](https://arxiv.org/abs/2512.16607)

    本研究针对平衡采样极其困难的非晶材料（玻璃），通过将所需的等变性直接嵌入黎曼结构中，开发了首个专为非晶粒子系统定制的玻尔兹曼生成器。

    

    在热力学平衡态下对构型进行采样是统计物理学中一个长期存在的挑战。玻尔兹曼生成器通过使用生成模型提出独立构型来解决这一问题，随后利用精确的似然评估通过重要性采样对这些构型进行重新加权。近期基于连续归一化流和流匹配的玻尔兹曼生成器在粒子系统和生物分子领域取得了显著成功。然而，这些方法尚未被扩展到非晶材料（玻璃），而这类材料的平衡采样以极其缓慢而著称。由于其无序结构，非晶材料的不变性和几何约束与晶体和生物分子不同，这阻碍了现有生成模型的直接应用。在本研究中，我们通过将所需的等变性直接构建到黎曼（流形）结构中，开发了专为非晶材料定制的玻尔兹曼生成器。

    arXiv:2512.16607v3 Announce Type: replace-cross  Abstract: Sampling configurations in thermodynamic equilibrium is a long-standing challenge in statistical physics. Boltzmann generators address this problem by employing generative models to propose independent configurations, which are then reweighted via importance sampling using exact likelihood evaluations. Recent Boltzmann Generators based on continuous normalizing flows and flow matching have achieved significant success for particle systems and biomolecules. However, these approaches have not been extended to amorphous materials (glasses), for which equilibrium sampling is notoriously slow. Because of their disordered structure, the invariances and geometrical constraints of amorphous materials differ from those of crystals and biomolecules, preventing the direct use of existing generative models. Here, we develop Boltzmann Generators tailored to amorphous materials by building the required equivariances directly into Riemannian 
    
[^218]: 理解聚合物基础模型中的结构表示

    Understanding Structural Representation in Foundation Models for Polymers

    [https://arxiv.org/abs/2512.11881](https://arxiv.org/abs/2512.11881)

    该研究提出了一种基于SMILES聚合物图表示（CPG）的化学语言基础模型，通过融入其他线性表示法缺失的聚合物结构特征与连接性信息，在30个聚合物性质基准数据集上取得卓越性能，证明了其作为语言基础模型中聚合物表示方法的稳健性。

    

    从训练数据的相对稀缺到标准化基准的缺乏，构建有效的聚合物基础模型面临着重大且多方面的挑战。从本质上讲，这些问题中的许多都直接与聚合物的结构表示相关。在此，我们提出了一种化学语言基础模型，该模型基于SMILES的聚合物图表示（CPG）构建，融入了其他线性表示法中常常缺失的聚合物结构特征和连接性信息。该基础模型在30个不同的聚合物性质基准数据集上表现出卓越的性能。在对照实验中，对所开发的表示方法与其他变体进行的严格评估表明，该方法是在语言基础模型中表示聚合物的一种稳健方法。这些实验还揭示了结构表示对小扰动具有很强的不变性。

    arXiv:2512.11881v2 Announce Type: replace-cross  Abstract: From the relative scarcity of training data to the lack of standardized benchmarks, the creation of effective foundation models for polymers faces significant and multi-faceted challenges. At the core, many of these issues are tied directly to the structural representation of polymers. Here, we present a chemical language foundation model built on using a SMILES-based polymer graph representation (CPG) that incorporates polymer architectural features and connectivity that are often missing in other line notations. This foundation model exhibited excellent performance on 30 different polymer property benchmark datasets. Critical evaluation of the developed representation against other variations in control experiments reveals this approach to be a robust method of representing polymers in language-based foundation models. These experiments also reveal a strong invariance of structural representations to small perturbations, with
    
[^219]: K2-V2：一个360度全开放、推理增强的大语言模型

    K2-V2: A 360-Open, Reasoning-Enhanced LLM

    [https://arxiv.org/abs/2512.06201](https://arxiv.org/abs/2512.06201)

    K2-V2是一个从零构建的360度全开放推理增强大语言模型，性能超越Qwen2.5-72B并接近Qwen3-235B，其完整开放的训练数据与历史为社区的持续训练和推理适配提供了卓越基座。

    

    我们推出了K2-V2，一个从零开始构建的360度全开放大语言模型，它不仅具备通用大语言模型的对话和知识检索等功能，更是一个面向推理适配的卓越基座。它是最强的完全开放模型，在同规模级别中可与开放权重模型的领先者相媲美，性能超越Qwen2.5-72B，并接近Qwen3-235B的水平。我们在整个训练过程中主动注入领域知识、推理、长上下文和工具使用能力，明确地为模型应对复杂推理任务做好准备。我们通过简单的监督微调展示了这一潜力，建立了一个强基线，表明在高级对齐方面仍有巨大的提升空间。通过发布完整的训练历史和数据组成，我们最大化了持续训练的有效性，这是开源生产中的一个关键场景。我们发布了模型权重以及LLM360的标志性产物（如完整的训练数据），以赋能开源社区。

    arXiv:2512.06201v3 Announce Type: replace  Abstract: We introduce K2-V2, a 360-open LLM built from scratch as a superior base for reasoning adaptation, in addition to functions such as conversation and knowledge retrieval from general LLMs. It stands as the strongest fully open model, rivals open-weight leaders in its size class, outperforms Qwen2.5-72B and approaches the performance of Qwen3-235B. We actively infuse domain knowledge, reasoning, long-context, and tool use throughout the training process. This explicitly prepares the model for complex reasoning tasks. We demonstrate this potential using simple supervised fine-tuning, establishing a strong baseline that indicates significant headroom for advanced alignment. By releasing the full training history and data composition, we maximize the effectiveness of continuous training, a key open source production scenario. We release the model weights and signature LLM360 artifacts, such as complete training data, to empower the commun
    
[^220]: SteganoBackdoor：通过隐写式后门规避数据投毒防御

    SteganoBackdoor: Evading Data-Poisoning Defenses via Steganographic Backdoors

    [https://arxiv.org/abs/2511.14301](https://arxiv.org/abs/2511.14301)

    提出SteganoBackdoor优化框架，通过自回归token替换将语义触发器种子转化为隐写式投毒样本，将后门载荷分散编码在普通token中，从而在保持语言流畅性的同时规避现有数据投毒防御的检测。

    

    基于Transformer的模型极易通过监督微调（SFT）遭受后门攻击。为了对现有的数据投毒防御进行红队测试，先前的工作越来越关注风格化触发器、合成伪影和token级扰动，以规避检测。然而，这一趋势使威胁模型偏离了自然存在的语义触发器和现实的低预算投毒设置。针对这一空白，我们提出了SteganoBackdoor，这是一个基于优化的框架，通过自回归token替换来转换语义触发器种子，在保持强单样本训练时载荷的同时，逐步最小化与推理时触发器的嵌入重叠。由此产生的SteganoPoisons保持了语言流畅性，并将载荷编码在普通token中，使得没有任何单个token携带集中信号，完整载荷而是从它们的精确组合中涌现。

    arXiv:2511.14301v4 Announce Type: replace-cross  Abstract: Transformer-based models are highly susceptible to backdoor attacks via supervised fine-tuning (SFT). To red-team existing data-poisoning defenses, prior work has increasingly focused on stylized triggers, synthetic artifacts, and token-level perturbations designed to evade detection. However, this trend has shifted threat models away from naturally occurring semantic triggers and realistic low-budget poisoning settings. Addressing this gap, we introduce SteganoBackdoor, an optimization-based framework that transforms semantic-trigger seeds through autoregressive token replacement, sequentially minimizing embedding overlap with the inference-time trigger while preserving a strong per-sample training-time payload. The resulting SteganoPoisons maintain linguistic fluency and encode the payload across ordinary tokens, such that no individual token carries a concentrated signal and the full payload instead emerges from their exact 
    
[^221]: PRIVET：基于极值理论的邻近性隐私泄漏检测

    PRIVET: PRoximIty leakage detection Via Extreme value Theory

    [https://arxiv.org/abs/2510.24233](https://arxiv.org/abs/2510.24233)

    PRIVET利用最近邻距离上的极值统计，提出了一种通用的、与模态无关的样本级隐私泄漏检测算法，能够为每个合成样本赋予经校准的个体化邻近性泄漏分数。

    

    深度生成模型通常在敏感数据上进行训练，例如基因序列、健康数据，或更广泛地说，任何受版权、许可或保护的内容。这引发了关于隐私保护合成数据的关键担忧，尤其是隐私泄漏问题，该问题与过拟合密切相关。现有的基于邻近性的方法大多通过全局标准来评估隐私风险，这类标准仅能量化模型的整体行为，而无法将风险归因于单条记录。虽然存在样本级的输出，但它们要么未经校准，要么不连续，要么对模型整体欠拟合时发生的泄漏视而不见，这限制了其实际应用。通过在最近邻距离上应用极值统计，我们提出了PRIVET，这是一种通用的、基于样本的、与模态无关的算法，可为每个合成样本分配一个个体化的邻近性泄漏分数。这些分数在选定的表示和距离度量下进行评估……

    arXiv:2510.24233v2 Announce Type: replace  Abstract: Deep generative models are often trained on sensitive data, such as genetic sequences, health data, or more broadly, any copyrighted, licensed or protected content. This raises critical concerns around privacy-preserving synthetic data, and more specifically around privacy leakage, an issue closely tied to overfitting. Existing proximity-based methods mostly assess privacy risk through global criteria, which quantify a model's overall behaviour but cannot attribute risk to an individual record. Sample-level outputs do exist but they are either uncalibrated, discontinuous, or blind to leakage occurring while the model is globally underfit, which limits their practical use. Using extreme value statistics on nearest-neighbor distances, we propose PRIVET, a generic sample-based, modality-agnostic algorithm that assigns an individual proximity leak score to each synthetic sample. These are evaluated under a chosen representation and dista
    
[^222]: 安全过滤下可证明最优的强化学习

    Provably Optimal Reinforcement Learning under Safety Filtering

    [https://arxiv.org/abs/2510.18082](https://arxiv.org/abs/2510.18082)

    本文首次证明使用足够宽松的安全过滤器来强制执行安全性不会降低强化学习的渐近最优性能，从而消除了安全过滤必然牺牲性能这一普遍误解。

    

    强化学习（RL）的最新进展使其能够应用于日益复杂的任务，但缺乏形式化的安全保证仍然限制了其在安全关键场景中的应用。一种常见的实用方法是为RL策略配备安全过滤器，该过滤器会覆盖不安全的动作，以防止训练和部署过程中发生故障。然而，安全过滤常被认为会牺牲性能并阻碍学习过程。我们证明了这种被认为存在的安全-性能权衡并非固有的，并首次证明，使用足够宽松的安全过滤器来强制执行安全性不会降低渐近性能。我们通过安全关键马尔可夫决策过程（SC-MDP）形式化了RL安全性，该过程要求对灾难性失败状态的规避是绝对性的，而非高概率的。此外，我们定义了一个相关的过滤MDP，其中所有动作都会产生安全的效（摘要在此处截断）

    arXiv:2510.18082v3 Announce Type: replace  Abstract: Recent advances in reinforcement learning (RL) enable its use on increasingly complex tasks, but the lack of formal safety guarantees still limits its application in safety-critical settings. A common practical approach is to augment the RL policy with a safety filter that overrides unsafe actions to prevent failures during both training and deployment. However, safety filtering is often perceived as sacrificing performance and hindering the learning process. We show that this perceived safety-performance tradeoff is not inherent and prove, for the first time, that enforcing safety with a sufficiently permissive safety filter does not degrade asymptotic performance. We formalize RL safety with a safety-critical Markov decision process (SC-MDP), which requires categorical, rather than high-probability, avoidance of catastrophic failure states. Additionally, we define an associated filtered MDP in which all actions result in safe effec
    
[^223]: 语义正样本对对自监督表征学习的影响

    The Impact of Semantic Pairs on Self-Supervised Representation Learning

    [https://arxiv.org/abs/2510.08722](https://arxiv.org/abs/2510.08722)

    该论文通过从ImageNet-1K构建类别构成匹配的增强对基线与人工整理的语义对数据集，开展了首个受控实证研究，以分离并量化语义正样本对对自监督表征学习的影响。

    

    实例判别通过将同一图像的不同增强视图视为正样本对来学习视觉表征。虽然这种方式促进了对手工设计变换的不变性，但同图像正样本对可能保留干扰性关联，例如背景、纹理、光照和物体特定细节。语义正样本对，即同一类别的不同实例，可以通过在不同上下文中呈现物体来减少这些关联。然而，以往的研究常常将语义对与增强正样本或假邻居（即错误映射的语义对）混合使用，使得难以分离出语义配对本身的效果。我们对自监督表征学习中的语义正样本对进行了受控实证研究。从ImageNet-1K中，我们构建了两个匹配的子集：一个增强对基线数据集和一个手动整理的语义对数据集，两者具有相同的类别构成和…

    arXiv:2510.08722v4 Announce Type: replace-cross  Abstract: Instance discrimination learns visual representations by treating different augmented views of the same image as positive pairs. While this encourages invariance to handcrafted transformations, same-image positives can preserve nuisance correlations such as background, texture, illumination, and object-specific details. Semantic positive pairs, i.e., different same-class instances, may reduce these correlations by presenting objects across diverse contexts. However, previous studies often combine semantic pairs with augmented positives or false neighbors (i.e., incorrectly mapped semantic pairs), making it difficult to isolate the effect of semantic pairing. We present a controlled empirical study of semantic positive pairs for self-supervised representation learning. From ImageNet-1K, we construct two matched subsets: an augmented-pair baseline and a manually curated semantic-pair dataset with the same class composition and tr
    
[^224]: Transformers 在无图先验的情况下发现分子结构

    Transformers Discover Molecular Structure Without Graph Priors

    [https://arxiv.org/abs/2510.02259](https://arxiv.org/abs/2510.02259)

    该研究表明，Transformer模型无需嵌入图结构或几何局部性等物理先验，仅通过数据学习就能自动发现分子结构等物理模式。

    

    计算模拟在科学发现中扮演着核心角色，而机器学习（ML）已成为传统基于物理建模的一种有前景的替代方案。然而，科学建模需要具有物理意义的预测，这为数据驱动方法提出了一个根本性问题：物理归纳偏置——即关于物理世界结构的先验假设——能在多大程度上仅通过从数据中学习而涌现？例如，在原子建模领域，机器学习架构历来都嵌入了强物理归纳偏置——如几何局部性和图结构——其依据是假设这些先验对于物理预测是必不可少的。我们系统地研究了物理模式如何能够直接从数据中被发现：通过训练一个不含领域特定先验的模型，其中包括任何手动定义的原子间成对相互作用。我们发现……

    arXiv:2510.02259v2 Announce Type: replace  Abstract: Computational simulations play a central role in scientific discovery, and machine learning (ML) has emerged as a promising alternative to traditional physics-based modeling. However, scientific modeling requires physically meaningful predictions, raising a fundamental question for data-driven methods: to what extent can physical inductive biases - that is, prior assumptions about the structure of the physical world - emerge by learning from data alone? In atomistic modeling, for example, ML architectures have historically embedded strong physical inductive biases - such as geometric locality and graph structure - based on the assumption that these priors are necessary for physical predictions. We systematically develop an understanding of how physical patterns can alternatively be discovered directly from data by training a model without domain-specific priors, including any manually defined atomistic pairwise interactions. We find 
    
[^225]: Fidel-TS：一个用于时间序列预测的高保真多模态基准

    Fidel-TS: A High-Fidelity Multimodal Benchmark for Time Series Forecasting

    [https://arxiv.org/abs/2509.24789](https://arxiv.org/abs/2509.24789)

    提出了基于数据来源完整性、无泄露设计和结构清晰性原则构建的高保真大规模多模态时间序列预测基准 Fidel-TS，揭示了以往基准的局限性及模型评估中的潜在差异。

    

    时间序列预测模型的评估由于缺乏高质量的基准而受到阻碍，导致对研究进展的评估被高估。现有的数据集存在诸多问题，包括单模态设计中的小规模、低频率、预训练数据污染，以及早期多模态设计中普遍存在的时间和信息描述泄露。为了解决这些问题，我们形式化了高保真基准测试的核心原则，重点关注数据来源的完整性、无泄露设计和结构清晰性。我们推出了 Fidel-TS，一个基于这些原则构建的新型大规模基准。我们的实验揭示了先前基准的局限性以及模型评估中潜在的差异，为多种现有的单模态和多模态预测模型以及大语言模型（LLMs）在各种评估任务中提供了新的见解。

    arXiv:2509.24789v5 Announce Type: replace  Abstract: The evaluation of time series forecasting models is hindered by a lack of high-quality benchmarks, leading to overestimated assessments of progress. Existing datasets suffer from issues ranging from small-scale, low-frequency, pre-training data contamination in unimodal designs to the temporal and description leakage prevalent in early multimodal designs. To address this, we formalize the core principles of high-fidelity benchmarking, focusing on data sourcing integrity, leak-free design, and structural clarity. We introduce Fidel-TS, a new large-scale benchmark built from these principles. Our experiments reveal the limitations of prior benchmarks and the potential discrepancies in model evaluation, providing new insights into multiple existing unimodal and multimodal forecasting models and LLMs across various evaluation tasks.
    
[^226]: 一种用于不动点迭代的遗憾最小化方法

    A regret minimization approach to fixed-point iterations

    [https://arxiv.org/abs/2509.21653](https://arxiv.org/abs/2509.21653)

    本文提出了一种将遗憾最小化算法转换为不动点迭代的通用方案，推广了经典的Krasnoselskii-Mann迭代，并基于AdaGrad算法获得了收敛更快的新型自适应不动点迭代方法。

    

    我们提出了一种转换方案，能够将遗憾最小化算法转化为不动点迭代，其收敛保证由遗憾界推导而来。所得到的迭代可以看作是对经典Krasnoselskii--Mann迭代的一个重大推广，因为通过转换在线梯度下降算法即可恢复出后者。这种方法为寻找非自映射算子的不动点提供了新的简单迭代方法。我们还重点研究了转换AdaGrad家族的遗憾最小化算法，从而获得了具有新型自适应保证的不动点迭代。在多个问题上的数值实验表明，基于AdaGrad的不动点迭代比Krasnoselskii--Mann迭代收敛更快。

    arXiv:2509.21653v2 Announce Type: replace-cross  Abstract: We propose a conversion scheme that turns regret minimizing algorithms into fixed point iterations, with convergence guarantees following from regret bounds. The resulting iterations can be seen as a grand extension of the classical Krasnoselskii--Mann iterations, as the latter are recovered by converting the Online Gradient Descent algorithm. This approach yields new simple iterations for finding fixed points of non-self operators. We also focus on converting algorithms from the AdaGrad family of regret minimizers, and thus obtain fixed point iterations with adaptive guarantees of a new kind. Numerical experiments on various problems demonstrate faster convergence of AdaGrad-based fixed point iterations over Krasnoselskii--Mann iterations.
    
[^227]: 潜在异质性下面向算法公平性的鲁棒混合模型

    Robust Mixture Models for Algorithmic Fairness Under Latent Heterogeneity

    [https://arxiv.org/abs/2509.17411](https://arxiv.org/abs/2509.17411)

    提出了ROME框架，将潜变量混合建模与分布鲁棒优化（DRO）相结合，在无需预先指定分组的情况下自动学习潜在的、交叉性的子群体结构，并同时优化最差群体的预测性能，从而提升算法公平性。

    

    为平均性能而优化的机器学习模型可能在弱势子群体上表现不佳。现有方法通常依赖于预先指定的分组，然而与公平性相关的子群体结构可能是潜在的、交叉性的，并由连续与离散属性之间复杂的交互作用所驱动。我们提出了ROME（鲁棒混合集成，RObust Mixture Ensemble），这是一个在学习潜在群体结构的同时优化最差群体预测性能的框架。ROME通过两种互补的方法将潜变量建模与分布鲁棒优化（DRO）联系起来：针对线性模型的带鲁棒聚合的期望最大化（EM）方法，以及针对非线性设置的神经混合专家方法。在仿真实验和三个真实世界回归数据集上，ROME在保持具有竞争力的整体性能的同时，提升了最差群体的表现。

    arXiv:2509.17411v2 Announce Type: replace-cross  Abstract: Machine learning models optimized for average performance can perform poorly on vulnerable subpopulations. Existing approaches often rely on groups specified in advance, yet fairness-relevant subgroup structure may be latent, intersectional, and driven by complex interactions among continuous and discrete attributes. We introduce \textbf{ROME} (\textbf{\underline{RO}}bust \textbf{\underline{M}}ixture \textbf{\underline{E}}nsemble), a framework that learns latent group structure while optimizing worst-group predictive performance. ROME connects latent-variable modeling with distributionally robust optimization (DRO) through two complementary approaches: an Expectation-Maximization formulation with robust aggregation for linear models and a neural Mixture-of-Experts formulation for nonlinear settings. Across simulations and three real-world regression datasets, ROME improves worst-group performance while maintaining competitive o
    
[^228]: 超越次优性进行泛化：离线强化学习通过随机解学习有效调度

    Generalizing Beyond Suboptimality: Offline Reinforcement Learning Learns Effective Scheduling through Random Solutions

    [https://arxiv.org/abs/2509.10303](https://arxiv.org/abs/2509.10303)

    提出了离线强化学习算法CDQAC，通过结合分位数评论家与延迟策略更新，直接从静态的次优数据集中学习有效的车间调度策略，其性能达到或超越生成数据的启发式方法以及现有的离线和在线RL基线，同时具有极高的样本效率。

    

    在线强化学习（RL）方法通过与模拟环境直接交互学习调度策略，在作业车间调度（JSP）和柔性作业车间调度（FJSP）问题上展现出强大性能。然而，这些方法通常需要大量的训练交互，限制了其样本效率和实际适用性。受此挑战的启发，我们提出了保守离散分位演员-评论家算法（CDQAC），这是一种离线RL算法，能够直接从静态的、次优的数据集中学习有效的调度策略。CDQAC将基于分位数的评论家与延迟策略更新相结合，以估计机器-工序对的回报分布。在JSP和FJSP基准测试上的大量实验表明，CDQAC能够达到或超越生成数据的启发式方法，优于近期面向JSP和FJSP的离线与在线RL基线，并且具有极高的样本效率，仅需……（原文摘要在此处截断）

    arXiv:2509.10303v3 Announce Type: replace-cross  Abstract: Online reinforcement learning (RL) approaches have demonstrated strong performance on Job Shop Scheduling (JSP) and Flexible JSP (FJSP) problems by learning scheduling policies through direct interaction with simulated environments. However, these methods often require extensive training interactions, limiting their sample efficiency and practical applicability. Motivated by this challenge, we introduce Conservative Discrete Quantile Actor-Critic (CDQAC), an offline RL algorithm that learns effective scheduling policies directly from static, suboptimal datasets. CDQAC couples a quantile-based critic with delayed policy updates to estimate the return distribution of machine-operation pairs. Extensive experiments on JSP and FJSP benchmarks demonstrate that CDQAC matches or outperforms the data-generating heuristics, outperforms recent offline and online RL baselines for JSP and FJSP, and is highly sample efficient, requiring only
    
[^229]: CASE：用于类别敏感解释的对比激活方法

    CASE: Contrastive Activation for Class-Sensitive Explanations

    [https://arxiv.org/abs/2506.07327](https://arxiv.org/abs/2506.07327)

    该论文提出类别敏感性诊断测试，揭示了许多主流显著性方法无论类别标签如何都产生几乎相同解释的结构性缺陷，并据此提出对比解释方法CASE，可分离出对预测类别具有独特判别性的特征。

    

    显著性方法被广泛用于可视化哪些输入特征被认为与模型的预测相关。然而，它们在视觉上的合理性可能掩盖了关键的局限性。在这项工作中，我们提出了一种类别敏感性诊断测试：即评估方法在相同输入上区分相互竞争的类别标签的能力。通过大量实验，我们表明许多广泛使用的显著性方法无论类别标签如何，都会产生几乎相同的解释，这使它们的可靠性受到质疑。我们发现这种类别不敏感的行为在不同架构和数据集上都普遍存在，表明这种失效模式是结构性的，而非特定于某个模型。基于这些发现，我们提出了CASE，这是一种对比解释方法，能够分离出对预测类别具有独特判别性的特征。我们使用所提出的诊断测试和基于扰动的保真度测试对CASE进行评估，结果表明它能够产生更符合类别区分性的解释。

    arXiv:2506.07327v4 Announce Type: replace-cross  Abstract: Saliency methods are widely used to visualize which input features are deemed relevant to a model's prediction. However, their visual plausibility can obscure critical limitations. In this work, we propose a diagnostic test for class sensitivity: a method's ability to distinguish between competing class labels on the same input. Through extensive experiments, we show that many widely used saliency methods produce nearly identical explanations regardless of the class label, calling into question their reliability. We find that class-insensitive behavior persists across architectures and datasets, suggesting the failure mode is structural rather than model-specific. Motivated by these findings, we introduce CASE, a contrastive explanation method that isolates features uniquely discriminative for the predicted class. We evaluate CASE using the proposed diagnostic and a perturbation-based fidelity test, and show that it produces fa
    
[^230]: 通过激活子空间理解加法的上下文学习

    Understanding In-context Learning of Addition via Activation Subspaces

    [https://arxiv.org/abs/2505.05145](https://arxiv.org/abs/2505.05145)

    该研究提出一种新方法，将语言模型加法少样本学习的机制定位到仅三个注意力头，并揭示其通过六维激活子空间实现。

    

    为了执行少样本学习，语言模型需要从少量输入-标签对中提取信号，将其聚合为学到的预测规则，并将该规则应用于新的输入。这一过程在现代Transformer模型的前向传播中是如何实现的？为探究这一问题，我们研究了一族结构化的少样本学习任务，其真实预测规则是给输入加上一个整数 $k$。我们提出了一种新颖的方法，能够将模型的少样本学习能力定位到仅仅少数几个注意力头上。该方法及相关发现可推广到涵盖算术与语义任务的另外四个任务族。随后，我们通过降维以及对注意力头输出空间的分解，对各个注意力头进行了深入分析。例如，在Llama-3-8B-Instruct中，我们将这些任务的底层机制简化为仅三个注意力头及其六维子空间，其中四个维度用于追踪所加的数字。

    arXiv:2505.05145v4 Announce Type: replace-cross  Abstract: To perform few-shot learning, language models extract signals from a few input-label pairs, aggregate them into a learned prediction rule, and apply this rule to new inputs. How is this implemented in the forward pass of modern transformer models? To explore this question, we study a structured family of few-shot learning tasks for which the true prediction rule is to add an integer $k$ to the input. We introduce a novel method that localizes the model's few-shot learning ability to only a few attention heads. This method and the findings generalize to four additional task families spanning arithmetic and semantic tasks. We then perform an in-depth analysis of individual heads via dimensionality reduction and decomposition of the heads' output spaces. For example, in Llama-3-8B-Instruct, we reduce the mechanism underlying these tasks to just three attention heads with six-dimensional subspaces, in which four dimensions track th
    
[^231]: 面向鲁棒机器人运动技能学习的轨迹熵强化学习

    Trajectory Entropy Reinforcement Learning for Robust Robot Motor Skill Learning

    [https://arxiv.org/abs/2505.04193](https://arxiv.org/abs/2505.04193)

    该论文提出轨迹熵强化学习（TERL），通过最小化整个动作轨迹的熵向策略引入简洁性归纳偏置，从而学习到对环境轻微扰动更鲁棒的机器人运动技能。

    

    简洁性是设计数据驱动控制器的一个关键归纳偏置，尤其是在鲁棒性至关重要的情况下。尽管深度强化学习在复杂控制任务中取得了令人瞩目的成果，但它容易捕捉到观测与动作之间复杂而虚假的相关性，导致在环境受到轻微扰动时发生失败。为了解决这一问题，本工作在强化学习中引入了一种新颖的面向简单策略的归纳偏置。这种简洁性归纳偏置通过最小化整个动作轨迹的熵来引入，其对应于智能体在观测状态轨迹之后描述动作轨迹中信息所需的比特数。我们的强化学习智能体——轨迹熵强化学习——被优化为在最大化奖励的同时最小化轨迹熵。我们证明了轨迹熵可以被有效地估计。

    arXiv:2505.04193v2 Announce Type: replace  Abstract: Simplicity is a critical inductive bias for designing data-driven controllers, especially when robustness is important. Despite the impressive results of deep reinforcement learning in complex control tasks, it is prone to capturing intricate and spurious correlations between observations and actions, leading to failure under slight perturbations to the environment. To tackle this problem, in this work we introduce a novel inductive bias towards simple policies in reinforcement learning. The simplicity inductive bias is introduced by minimizing the entropy of entire action trajectories, corresponding to the number of bits required to describe information in action trajectories after the agent observes state trajectories. Our reinforcement learning agent, Trajectory Entropy Reinforcement Learning, is optimized to minimize the trajectory entropy while maximizing rewards. We show that the trajectory entropy can be effectively estimated 
    
[^232]: OverThink：针对推理大语言模型的减速攻击

    OverThink: Slowdown Attacks on Reasoning LLMs

    [https://arxiv.org/abs/2502.02542](https://arxiv.org/abs/2502.02542)

    Overthink攻击通过向外部上下文注入单独看似无害的诱饵推理问题，迫使推理大语言模型生成多达数十倍的推理标记，从而显著增加计算成本，同时仍能产生正确答案并轻易绕过安全过滤器。

    

    推理语言模型（RLM）会生成成本高昂的推理标记（token），这些标记通常对用户隐藏，帮助模型在许多任务中表现出色。我们的Overthink攻击针对依赖外部上下文的基于RLM的应用（如聊天机器人或编程代理），迫使这些模型生成大幅增加的推理标记，同时仍能产生上下文正确的答案。攻击者通过向可用内容中注入经过优化的诱饵推理问题来实施攻击，以引出大量标记。我们精心设计了诱饵挑战（使用马尔可夫决策过程、语言翻译或图形理解），这些挑战单独看起来无害，但插入上下文后会产生对抗性影响，使其能够轻易绕过安全过滤器。我们在FreshQA、SQuAD和MuSR数据集上对专有和开源推理模型评估了Overthink，分别观察到13倍、46倍和12倍的推理标记增长。

    arXiv:2502.02542v5 Announce Type: replace  Abstract: A reasoning language model (RLM) generates costly reasoning tokens, often hidden from the users, that help it excel at many tasks. Our Overthink attack targets RLM-based applications (such as chatbots or coding agents) that rely on external context by forcing these models to generate substantially more reasoning tokens while still producing contextually correct answers. An adversary conducts the attack by injecting decoy reasoning problems into available content, optimized to elicit a large number of tokens. We craft decoy challenges (using Markov decision processes, language translation, or graphic comprehension) that appear benign individually yet have an adversarial impact when inserted in the context, allowing them to easily evade safety filters. We evaluate Overthink on proprietary and open-source reasoning models across the FreshQA, SQuAD, and MuSR datasets, where we observe 13x, 46x, and 12x increases, respectively. We also ex
    
[^233]: dSTAR：容忍掉队者且具备拜占庭韧性的分布式SGD

    dSTAR: Straggler Tolerant and Byzantine Resilient Distributed SGD

    [https://arxiv.org/abs/2412.07151](https://arxiv.org/abs/2412.07151)

    dSTAR是一种轻量高效的分布式SGD方法，通过收集最先响应的前k个工作节点的梯度并利用集成中位数过滤偏差，同时缓解掉队者效应并抵御拜占庭攻击，且具有理论上的拜占庭韧性和线性收敛速率保证。

    

    分布式模型训练需要适应掉队者效应和拜占庭攻击等挑战。在使用多个计算节点协调训练过程时，确保在网络和系统故障情况下及时可靠地进行梯度聚合至关重要。为了解决这些问题，我们提出了dSTAR，这是一种轻量级且高效的分布式随机梯度下降（SGD）方法，可增强鲁棒性和收敛性。dSTAR通过收集最先响应的前k个工作节点的更新，并使用集成中位数计算出的偏差对其进行过滤，从而选择性地聚合梯度。这种方法不仅减轻了掉队者的影响，还增强了模型对拜占庭攻击者的防御能力。我们从理论上证明dSTAR具有(α, f)-拜占庭韧性，并实现了线性收敛速率。在各种场景下的实证评估表明……

    arXiv:2412.07151v1 Announce Type: cross  Abstract: Distributed model training needs to be adapted to challenges such as the straggler effect and Byzantine attacks. When coordinating the training process with multiple computing nodes, ensuring timely and reliable gradient aggregation amidst network and system malfunctions is essential. To tackle these issues, we propose \textit{dSTAR}, a lightweight and efficient approach for distributed stochastic gradient descent (SGD) that enhances robustness and convergence. \textit{dSTAR} selectively aggregates gradients by collecting updates from the first \(k\) workers to respond, filtering them based on deviations calculated using an ensemble median. This method not only mitigates the impact of stragglers but also fortifies the model against Byzantine adversaries. We theoretically establish that \textit{dSTAR} is (\(\alpha, f\))-Byzantine resilient and achieves a linear convergence rate. Empirical evaluations across various scenarios demonstrate
    
[^234]: 连续脉冲图神经网络

    Continuous Spiking Graph Neural Networks

    [https://arxiv.org/abs/2404.01897](https://arxiv.org/abs/2404.01897)

    COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。

    

    连续图神经网络（CGNNs）因引入连续动力学而引起了极大关注，能够推广现有的离散图神经网络（GNNs）。它们通常受扩散类方法启发，引入了一种新颖的传播方案，并使用常微分方程（ODE）进行分析。然而，CGNNs的实现需要大量计算能力，这使得它们难以部署在电池供电设备上。受最近脉冲神经网络（SNNs）的启发，SNNs模拟生物推理过程并提供一种节能的神经架构，我们将SNNs与CGNNs结合到一个统一框架中，命名为连续脉冲图神经网络（COS-GNN）。我们在每个时间步骤使用SNNs进行图节点表示，这些表示进一步与时间一起集成到ODE过程中，以增强信息保存和缓解...

    arXiv:2404.01897v1 Announce Type: cross  Abstract: Continuous graph neural networks (CGNNs) have garnered significant attention due to their ability to generalize existing discrete graph neural networks (GNNs) by introducing continuous dynamics. They typically draw inspiration from diffusion-based methods to introduce a novel propagation scheme, which is analyzed using ordinary differential equations (ODE). However, the implementation of CGNNs requires significant computational power, making them challenging to deploy on battery-powered devices. Inspired by recent spiking neural networks (SNNs), which emulate a biological inference process and provide an energy-efficient neural architecture, we incorporate the SNNs with CGNNs in a unified framework, named Continuous Spiking Graph Neural Networks (COS-GNN). We employ SNNs for graph node representation at each time step, which are further integrated into the ODE process along with time. To enhance information preservation and mitigate in
    
[^235]: 基于阻尼高斯-牛顿优化的多目标超参数搜索

    Multi-Objective Hyperparameter Search via Damped Gauss--Newton Optimization

    [https://arxiv.org/abs/2401.03580](https://arxiv.org/abs/2401.03580)

    本文提出一种基于阻尼高斯-牛顿优化的多目标超参数搜索方法，利用有限差分雅可比矩阵和Tikhonov正则化实现有向的联合参数更新，在欠定情况下以远少于网格搜索的试验次数达到同等的最佳验证精度。

    

    本文从数值优化的角度研究超参数优化（HPO），提出了一种多目标的阻尼高斯-牛顿搜索方法。与将模型评估视为独立试验的做法不同，该方法通过有限差分估计雅可比矩阵，以捕捉多个验证指标对超参数扰动的局部敏感性。随后，利用Tikhonov正则化的高斯-牛顿系统产生有向的联合更新，从而解决了超参数数量超过性能目标数量时的欠定问题。作者通过调整四个XGBoost超参数，在三个公开分类数据集上对该方法进行了评估，并与穷举网格搜索、随机搜索以及树结构Parzen估计器（TPE）优化进行了比较。在一个受控的乳腺癌数据集划分上，所提出的方法达到了320个配置的网格搜索所获得的最佳验证精度，同时以略……（摘要原文在此处截断）

    arXiv:2401.03580v2 Announce Type: replace-cross  Abstract: We study hyperparameter optimization (HPO) from a numerical-optimization perspective and propose a multi-objective, damped Gauss--Newton search method. Rather than treating model evaluations as independent trials, the method estimates a finite-difference Jacobian that captures the local sensitivity of multiple validation metrics to hyperparameter perturbations. A Tikhonov-regularized Gauss--Newton system then produces a directed joint update, addressing the underdetermined setting in which the number of hyperparameters exceeds the number of performance objectives. We evaluate the method on three public classification datasets by tuning four XGBoost hyperparameters and compare it with exhaustive grid search, random search, and tree-structured Parzen estimator (TPE) optimization. On a controlled Breast Cancer split, the proposed method matches the best validation accuracy of a 320-configuration grid search while obtaining slightl
    
[^236]: 受外部影响下的强化学习：理论保证、算法与样本复杂度

    Reinforcement Learning under External Influence: Guarantees, Algorithms, and Sample Complexity

    [https://arxiv.org/abs/2305.16056](https://arxiv.org/abs/2305.16056)

    本文研究了连续状态-动作空间中受外部过程非马尔可夫扰动影响的强化学习问题，提出了基于有限外生事件历史的策略迭代算法，并给出了可处理性条件、策略改进保证及样本复杂度分析。

    

    本文研究了在外部事件影响下的强化学习问题。为此，我们考虑具有连续状态和动作空间的马尔可夫决策过程，其转移动态受到一个外部过程以非马尔可夫方式的扰动。首先，基于外生过程所引入扰动的性质，我们建立了使该问题变得可处理（tractable）的条件，即仅需考虑有限的历史事件即可求解该问题。我们提出并从理论上分析了一种策略迭代算法来解决这一问题，该算法学习的策略依赖于环境的当前状态以及先前外生事件的有限历史。由于该算法并不保证收敛，我们为策略改进提供了保证，该保证适用于由可处理策略和价值函数近似误差所决定的状态空间区域。

    arXiv:2305.16056v5 Announce Type: replace-cross  Abstract: In this paper, we study the problem of reinforcement learning under the influence of external events. For this, we consider Markov decision processes with continuous state and action spaces whose transition dynamics are perturbed by an external process in a non-Markovian manner. First, we establish the conditions under which the problem becomes tractable, allowing it to be addressed by considering only a finite history of events, based on the properties of the perturbations introduced by the exogenous process. We propose and theoretically analyze a policy iteration algorithm to tackle this problem that learns policies contingent on the current state of the environment and a finite history of prior exogenous events. Since this algorithm is not guaranteed to converge, we provide a guarantee for policy improvement in regions of the state space determined by the approximation error induced by considering tractable policies and valu
    
[^237]: 深度强化学习的展开总相关性

    Rollout Total Correlation for Deep Reinforcement Learning

    [https://arxiv.org/abs/2209.05333](https://arxiv.org/abs/2209.05333)

    本文提出通过最大化智能体轨迹中所有学习表征与动作之间的展开总相关性来学习任务相关表征，并采用基于生成式与判别式模型的两个互补下界以及分块小批量技术来优化该目标。

    

    学习任务相关的表征对强化学习至关重要。近期的方法旨在通过提高观测转移中的时间一致性来学习这类表征。然而，这些方法仅考虑单个转移，可能无法实现长期一致性。相反，我们认为捕获状态中与轨迹内其他状态和动作（甚至是更遥远的未来）相关的方面，能够进一步帮助提取任务相关信息。因此，本文研究了如何通过最大化展开总相关性（rollout total correlation）来学习表征，即智能体所产生的轨迹中所有学习到的表征与动作之间的相关性。为了提升展开总相关性，我们提出结合基于生成模型和判别模型的两个互补下界，并辅以一种简单有效的分块小批量（chunk-wise mini-batching）技术。

    arXiv:2209.05333v2 Announce Type: replace  Abstract: Learning task-relevant representations is crucial for reinforcement learning. Recent approaches aim to learn such representations by improving the temporal consistency in the observed transitions. However, they only consider individual transitions and can fail to achieve long-term consistency. Instead, we argue that capturing aspects of the state that correlate with other states and actions of the trajectory---even more distant in the future---could further help in extracting task-relevant information. Hence, in this paper we investigate how to learn representations by maximizing the rollout total correlation, the correlation among all learned representations and actions within the trajectories produced by the agent. For improving rollout total correlation, we propose to combine two complementary lower bounds based on a generative and a discriminative model, combined with a simple and effective technique of chunk-wise mini-batching. 
    

