# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What Should World Models Forget? Stratified Retention for Continual Adaptation](https://arxiv.org/abs/2610.03713) | 提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。 |
| [^2] | [RNADyn: A Benchmark for Generating and Understanding RNA Dynamics](https://arxiv.org/abs/2610.03712) | 该论文提出了RNADynBench——一个包含2585条质量控制的全原子RNA分子动力学轨迹的标准化基准，并开发了统一模型RNADynNet，通过共享骨干网络结合坐标去噪、单帧到轨迹对齐和物理接地技术，同时实现RNA轨迹生成与动力学表示学习。 |
| [^3] | [From Mixing to Tearing: Graph Decomposition in Decentralized Optimization via Message Passing](https://arxiv.org/abs/2610.03709) | 该论文从第一性原理出发提出了一种基于图分解的去中心化优化通用框架，通过消息传递联合设计优化子问题、对偶变量分块和协作求解的智能体集群，突破了传统仅利用网络混合信息的范式。 |
| [^4] | [LESSER: Post-Training Data Selection with Output-Layer Gradients](https://arxiv.org/abs/2610.03702) | LESSER提出仅需前向传播的输出层梯度来近似全参数梯度进行后训练数据选择，将特征提取计算成本在SFT和RL场景下分别降低9.7倍和3.0倍，同时保持下游任务性能不变。 |
| [^5] | [Simulation-Free Learning of Population Dynamics with Wasserstein Lagrangian Residuals](https://arxiv.org/abs/2610.03679) | 提出免模拟方法Double-Stitch，通过惩罚学习到的种群路径上运动方程的残差来学习Wasserstein空间中的拉格朗日力学，从而以低训练成本建模保守性和周期性的种群动力学。 |
| [^6] | [Planning to Learn](https://arxiv.org/abs/2610.03667) | 该论文揭示了精确策略梯度输给交叉熵的根本原因在于其短视性——只看重即时收益而忽视每次更新对未来学习的奠基作用，并将交叉熵重新诠释为“耐心的准确率”（考虑未来学习收益的长时域误差总量），从而通过在剩余学习量处截断该总量来改进优化方法。 |
| [^7] | [Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models](https://arxiv.org/abs/2610.03665) | Pivot-SD提出了一种高效的自蒸馏框架，通过信息增益指标识别去噪过程中真正塑造响应的少数关键决策（枢轴），并仅对这些高影响token进行针对性监督训练，从而解决了掩码扩散语言模型后训练中的信用分配问题。 |
| [^8] | [Forecasting from Counterfactual Simulator Rollouts: A Sim2Real Evaluation](https://arxiv.org/abs/2610.03662) | 该论文提出利用反事实模拟器推演生成新决策策略下的训练数据以解决预测模型的冷启动问题，并通过两个真实库存控制部署从模拟器保真度、零样本迁移和随真实数据积累的适应性三个角度评估了Sim2Real迁移效果。 |
| [^9] | [PoCoFL: POlicy-COmpliant Federated Learning](https://arxiv.org/abs/2610.03650) | 提出了PoCoFL框架，通过将联邦学习类型、策略语义与密码学实现三者解耦，并借助承诺和非交互式零知识证明来验证客户端贡献的策略合规性，从而实现适用于多种拓扑、角色和聚合方式的通用可验证联邦学习。 |
| [^10] | [On-Board Anomaly Detection for Efficient Marine Environmental Monitoring](https://arxiv.org/abs/2610.03649) | 该论文提出了一种用于对地观测卫星的轻量级海洋环境异常检测流程，利用自监督神经网络编码器压缩卫星图像并结合机器学习异常检测模型，实现高效的星载海洋环境监测。 |
| [^11] | [Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models](https://arxiv.org/abs/2610.03647) | 本文将摊销结构化随机变分推断应用于高斯过程潜变量模型，通过让潜空间的变分后验条件依赖于诱导点的取值，突破了平均场变分近似的局限，从而在数据流形重建的多项指标上取得了改进。 |
| [^12] | [When May a Bandit Leave Its Anchor? E-Process-Authorized Thompson Sampling under Non-stationarity](https://arxiv.org/abs/2610.03646) | 提出e-过程授权的汤普森采样（e-ATS），通过随时有效的e-过程决定非平稳环境中每只手臂何时从全历史锚点切换到折扣遗忘状态，保证平稳条件下偏离乐观汤普森采样的概率不超过设定的 $\alpha_E$，实验表明证据控制的是适应何时开始而非其是否总有益。 |
| [^13] | [On the Convergence of Success Conditioning for Policy Optimization](https://arxiv.org/abs/2610.03642) | 本文证明了成功条件化方法在广泛的一类马尔可夫决策过程上收敛于最优策略，并针对折扣MDP和单周期MDP分别给出了 $\mathcal{O}(1/\varepsilon^p)$ 和 $\mathcal{O}(\log(1/\varepsilon))$ 的收敛速率。 |
| [^14] | [IDRF: Inverse-Distilled Reward Fine-tuning of Masked Discrete Diffusion Models](https://arxiv.org/abs/2610.03641) | IDRF用逆蒸馏正则化替代难以处理的序列级KL惩罚，并结合截断策略梯度优化，实现了少步掩码离散扩散生成器的奖励微调。 |
| [^15] | [Broken scale symmetries in undercomplete linear autoencoders](https://arxiv.org/abs/2610.03640) | 该论文发现，在欠完备线性自编码器中，有限步长的SGD会以有向的方式破坏尺度对称性，在PCA解流形上系统性地偏向放大解码器权重，直至动力学触及有限步长的稳定性边界。 |
| [^16] | [FALCON: A Model and Dataset Agnostic Framework for Synthetic Data Generation for NL2SQL Pairs](https://arxiv.org/abs/2610.03625) | FALCON提出了一种模型与数据集无关的框架，通过保留字SQL种子、基于人设的提示生成和基于对齐的过滤，以低成本利用紧凑开源模型生成真实、具有歧义感知能力且结构复杂的NL-to-SQL合成数据。 |
| [^17] | [Normal-Form Correlation in Markov Games](https://arxiv.org/abs/2610.03621) | 本文首次为玩家数量固定的有限视界马尔可夫博弈中的标准形式相关均衡（NFCE）提出了高效计算算法，并在推荐跨状态独立的假设下证明了其 PPAD-完全性。 |
| [^18] | [UniIntervene++: An Adaptive Intervention Agent for Efficient Real-World Reinforcement Learning](https://arxiv.org/abs/2610.03620) | UniIntervene++是一个自适应干预智能体，通过在统一半马尔可夫决策过程中在线学习RL策略与多种辅助行为之间的相对价值，并根据策略当前能力动态调整控制分配，从而实现高效的真实世界机器人在线强化学习。 |
| [^19] | [Mastering Atari 2600 Games with Discovered Options](https://arxiv.org/abs/2610.03604) | 提出了Wayfarer，一个通用的、领域无关的深度强化学习智能体，它通过拉普拉斯表示学习从高维观测中发现选项，这些选项同时改善了探索、加速了信用分配并有效泛化到未见场景，从而在Atari 2600游戏上实现大幅加速的学习。 |
| [^20] | [A Path Integral Surrogate for Multi-Step Gradient Inversion in Federated Learning](https://arxiv.org/abs/2610.03597) | 该论文提出PI-SME方法，将FedAvg下客户端的累积更新视为梯度场沿权重轨迹的路径积分并用高斯-勒让德求积近似，从而增强多步梯度反演攻击从单次模型更新中重建客户端私有图像的能力。 |
| [^21] | [Threat-Preserving Representation Sensitivity in Agent-Security Benchmarks](https://arxiv.org/abs/2610.03585) | 论文提出威胁保持表征敏感性（TPRS）指标，发现在底层威胁完全不变的情况下，仅改变智能体可见的表征（如工具名称）就能使攻击成功率显著变化（最高约13个百分点），表明智能体安全基准的测量结果对表征方式高度敏感，可能无法真实反映模型的安全性。 |
| [^22] | [HyperBrowseComp: A Multilingual and Multimodal Stress Test for Web-Browsing Agents](https://arxiv.org/abs/2610.03574) | HyperBrowseComp是一个覆盖13种语言、包含423道人工验证问题的多语言多模态网页浏览基准，通过要求定位隐蔽证据、追踪多步线索链和检查异构信息源，为网页浏览智能体提供了极具挑战性的压力测试。 |
| [^23] | [Cephalonauts One: A deep fMRI dataset for decoding naturalistic speech in the human brain](https://arxiv.org/abs/2610.03558) | 发布了迄今最大的自然语音fMRI数据集Cephalonauts One（每名受试者30小时数据），并提出以音频片段检索为任务形式的大脑解码基准，附带标准化数据划分、评估指标和基线解码器。 |
| [^24] | [Get a GRIP, this will be a long TRIP: A Quantifiable Long-Range Framework for Verifying Over-squashing](https://arxiv.org/abs/2610.03556) | 该论文提出四个可验证公理（可预测性、紧凑性、严格k-范围、拓扑不变性），并据此构建可量化框架TRIP，为任意图上“是否真正需要长程交互”提供原则性证书，从而可信地验证GNN中的过挤压现象。 |
| [^25] | [Objects Without Morphisms: What LLMs for Mathematics Do Not Represent](https://arxiv.org/abs/2610.03551) | 本文通过相邻数学子领域间的命题翻译任务，提出一种将真值、内容与范围分置于独立盲评队列、且无需评判者的测量工具，揭示了大语言模型虽借助外部筛选达到专家级解题表现，却未能表示数学内容所隐含的概括层次。 |
| [^26] | [ZeroMAG: Zero-Shot Multimodal Adapter Generation for Plug-and-Play EEG Foundation Models](https://arxiv.org/abs/2610.03546) | ZeroMAG提出了一种零样本多模态适配器生成框架，无需目标标签或目标端优化，即可利用无标注数据为冻结的脑电基础模型自动生成多模态适配器，实现向异构多模态记录的即插即用扩展。 |
| [^27] | [Autonomous Robotic Navigation for Endovascular Brain-Computer Interface Access](https://arxiv.org/abs/2610.03537) | 这项工作首次展示了脑静脉系统中血管内脑机接口接入的体外自主机器人导航，利用强化学习控制器实现了从颈内静脉到上矢状窦的精确导航，并在未见过的解剖结构上验证了泛化能力。 |
| [^28] | [Divergence controls entropy in distillation](https://arxiv.org/abs/2610.03529) | 该论文证明蒸馏目标中散度的选择充当了隐式熵正则化器，控制着学生模型的熵——前向KL散度会使学生熵高于教师，反向KL散度会降低熵，而在线策略蒸馏的低熵来自token级反向KL散度而非采样方式本身。 |
| [^29] | [Beyond Trained Models: Compiling GNNs for a Sound Explainer Benchmark](https://arxiv.org/abs/2610.03526) | 该论文揭示了现有GNN可解释器基准中“训练模型依赖预期模体”这一隐含假设的不成立，并提出Gracr——首个将分级模态逻辑公式编译为GNN权重的编译器，通过用编译替代训练构建了真实解释可被形式化定义并精确计算的健全可解释器基准。 |
| [^30] | [From Benchmarks to Production: A Text-to-SQL System for Complex Financial Data](https://arxiv.org/abs/2610.03524) | FLINT是一个针对生产环境金融数据库的领域专用Text-to-SQL系统，通过查找代理解析不透明概念、嵌入检索专家模板以及基于外键链遍历的模式链接三大组件，解决了通用系统在此类复杂数据上准确率低于50%的问题。 |
| [^31] | [XGenAct: Geometry-Enhanced World Action Models through Cross-Task Generation](https://arxiv.org/abs/2610.03516) | XGenAct提出一种几何增强的世界动作模型，将RGB观测、机器人动作、度量深度、表面法线和功能角色分割统一表示为RGB视频，用单一视频扩散Transformer和统一目标学习跨模态的时间预测，从而增强机器人操作所需的空间理解能力。 |
| [^32] | [An Automated and Reproducible Workflow for Crack Identification and Damage Assessment of Fusion Materials](https://arxiv.org/abs/2610.03505) | 该论文在Galaxy科学工作流环境中实现了一个自动化可复现的裂纹识别与损伤评估流程，无需图像特定参数调优即可处理不同钨材料、微观结构、放大倍数和损伤状态的扫描电镜图像。 |
| [^33] | [Getting Your Guidance Weights Right in diffusion and flow-matching posterior sampling](https://arxiv.org/abs/2610.03503) | 提出一种基于最小二乘目标的简单离线策略，能够自动且有原则地调节扩散与流匹配后验采样中的引导权重，摆脱以往依赖启发式调参的做法。 |
| [^34] | [Certified Mechanistic Edits: Behavioral Guarantees for Skill Removal and Preservation](https://arxiv.org/abs/2610.03502) | 该论文首次提出对机制化编辑的行为效果进行认证的方法，可对连续嵌入空间区域内的每个输入可证明地保证：禁用一个电路将移除一种技能同时保留另一种技能，并在标准Transformer上验证了该方法的有效性。 |
| [^35] | [Below what training size do deep tabular generators stop beating trivial baselines? A preregistered benchmark on a size ladder of clinical and standard datasets](https://arxiv.org/abs/2610.03500) | 这项预注册的规模阶梯基准测试（共2,220次运行）发现，在临床等小型数据集上，几乎所有深度表格生成模型（如CTGAN、TVAE、TabDDPM）在任何测试的训练规模下都无法以超过随机噪声的幅度击败简单基线方法，挑战了深度生成模型在小型表格数据上的价值假设。 |
| [^36] | [Most-Recent Anchoring with Recurrent Ordering for Time Series Forecasting](https://arxiv.org/abs/2610.03494) | 提出MARO模型，以最新补丁为锚点、按从新到旧的顺序循环处理回看窗口，使时间序列长期预测架构能显式区分近期证据与远期上下文的不同贡献，且不引入额外参数。 |
| [^37] | [Dual-Context Analog Retrieval for Time Series Forecasting](https://arxiv.org/abs/2610.03491) | DuoTS提出了一种双上下文类比检索框架，通过结合捕捉近期动态的当前上下文与提供细节区分的细节上下文，逐步利用历史相似状态的后续观测来精化时序预测，从而避免完全依赖不可靠的单一检索结果。 |
| [^38] | [AREX: Affine-Residual Exponential Integrator for Few-Step Sampling in Flow Matching](https://arxiv.org/abs/2610.03483) | AREX是一种无需训练的流匹配模型少步采样器，它将采样动力学分解为由目标均值和协方差决定的仿射分量（用显式矩阵值传播子积分）与神经残差项，在无需重训练的情况下持续提升少步采样的样本保真度。 |
| [^39] | [Metropolis-Hastings Dominates Importance Resampling for Policy Composition](https://arxiv.org/abs/2610.03480) | 本文证明在任意采样预算下，基于 Metropolis-Hastings 的迭代校正方法产生的输出分布（以任意凸 f-散度衡量）都不劣于重要性重采样（SIR），从而为解码时策略组合中 MH 的理论优势提供了保证。 |
| [^40] | [Single or Multiple Policies for Phase-Structured Reinforcement Learning?](https://arxiv.org/abs/2610.03475) | 该论文从理论上证明单一共享策略可以达到任何多策略方案的性能，但实践中多策略是否更优取决于函数逼近、学习优化过程以及策略切换的样本效率与连续性损失等因素。 |
| [^41] | [When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion](https://arxiv.org/abs/2610.03465) | 该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。 |
| [^42] | [Generalization of Transformer-Based Neural Quantum States via In-Context Learning](https://arxiv.org/abs/2610.03463) | 本文为基于Transformer的神经量子态在上下文学习下的泛化行为建立了理论框架，严格证明了其逐点预测误差随上下文示例数量和Transformer深度成反比下降，且所需网络深度仅线性增长。 |
| [^43] | [Beyond Random Splits: Evaluating Drug-Target Affinity Models Under Chemically and Biologically Motivated Distribution Shifts Copy](https://arxiv.org/abs/2610.03456) | 该研究构建了包含化学与生物学动机分布偏移的DTA评估基准，发现蛋白质层面及双重分布偏移会显著降低模型性能并改变甚至逆转架构排名。 |
| [^44] | [Measure Less, Know More: Self-Supervised Test-Time Feature Acquisition](https://arxiv.org/abs/2610.03454) | 该论文提出ECHO-k，一种任务无关的自监督测试时模态获取方法，它利用基础模型的内部预训练表示作为代理目标，并通过强化学习策略顺序选择信息量最大的模态，从而在有限预算下持续提升下游任务性能。 |
| [^45] | [Causal Representation Learning with Instantaneous and Lagged Relations via Nonstationarity](https://arxiv.org/abs/2610.03452) | 本文通过利用与转移噪声分布变化相关的辅助变量所刻画的非平稳性，建立了识别时间序列潜在状态及其瞬时与滞后因果结构的充分条件，并据此提出了基于对比学习的 iCReN 因果表征学习框架。 |
| [^46] | [Electronic Density versus Geometry for Machine-Learned Molecular Absorption Spectra](https://arxiv.org/abs/2610.03444) | 该研究系统比较了以基态电子密度和分子几何结构作为机器学习模型输入来预测分子吸收光谱的效果，探讨了分子信息表示方式对光谱预测性能的影响。 |
| [^47] | [OptiSelect: How does the Optimizer Shape Data Curriculum?](https://arxiv.org/abs/2610.03432) | 本文提出OptiSelect优化器感知的数据选择范式，首次系统研究优化器如何影响数据课程选择，理论上证明基于符号和极坐标切向预处理的优化器（Lion、Muon）因效用分数可区分性崩溃而限制选择增益，而对角自适应优化器（AdamW、Sophia）则能获得严格更优的增益上界。 |
| [^48] | [Deep Bayesian REFoCUS](https://arxiv.org/abs/2610.03419) | 该论文提出Deep Bayesian REFoCUS方法，将超声多静态恢复建模为贝叶斯推断问题，利用深度生成先验攻克经典线性解码器失效的秩亏缺难题，在性能、不确定性量化以及无需微调的跨域泛化能力上均显著优于线性基线方法。 |
| [^49] | [Rethinking Epistemic Uncertainty in Node Classification through Information Growth](https://arxiv.org/abs/2610.03418) | 本文提出了一个在信息增长条件下检验节点分类中认知不确定性可约减性的统计框架，并揭示现有图证据深度学习方法难以满足一致性准则。 |
| [^50] | [Iterating Consistency Models: Stability, Error Bounds and Noise Schedules](https://arxiv.org/abs/2610.03414) | 该论文将多步一致性模型采样分析为加噪与近似去噪算子的复合，在可验证的稳定性假设下推导出非渐近误差界，揭示了噪声调度中早期大噪声驱动误差收缩、后期小噪声控制残余偏差的作用机制，为CM采样器设计提供了理论指导。 |
| [^51] | [AIBL: Augmented Instance-Based Learning with Structured Memory and Neural Embeddings](https://arxiv.org/abs/2610.03413) | 该论文提出AIBL，将经典的基于实例的学习理论（IBLT）从符号表示推广到学习得到的神经嵌入向量空间，用结构化记忆和学习的相似度来处理高维序贯数据，同时保留实例存储、激活加权检索和效用混合等核心机制。 |
| [^52] | [Contrastive Neural Embeddings Reveal Individual Traits Beyond Conversational Role](https://arxiv.org/abs/2610.03410) | 该研究将对比学习方法CEBRA应用于对话中成对被试的脑电数据，发现双人组合的个体特质（如自闭症商数差异）可解码至高于随机水平，但严格的置换检验表明解码性能可能主要反映嵌入的几何结构而非真实的标签信息。 |
| [^53] | [16-bit Precision of Convolutional Neural Networks on Microcontroller Units for 8-bit Costs](https://arxiv.org/abs/2610.03402) | 本文提出W16A16，一种面向微控制器的16位高精度量化方法，在Armv7E-M架构上以与8位量化相当甚至更优的速度和能耗，将量化误差降低约10倍。 |
| [^54] | [A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants](https://arxiv.org/abs/2610.03396) | 该论文提出一个无需重训练的观测插值子统一框架，可将预训练的随机插值子、流匹配和扩散模型直接转化为贝叶斯数据同化中的后验采样器，并统一了随机与确定性两种后验采样方法。 |
| [^55] | [Bidirectional Voronoi-biased Exploration Curriculum for Reinforcement Learning](https://arxiv.org/abs/2610.03395) | 提出BVER方法，借鉴双向RRT规划思想，让初始状态课程与目标课程同时从两端相向扩展并偏向未探索的任务空间，从而有效缓解稀疏奖励长时程任务中的探索瓶颈。 |
| [^56] | [From Patching to Pruning Visual Computation in Vision Language Models](https://arxiv.org/abs/2610.03389) | P2P是一种免训练框架，将激活修补从诊断工具转变为推理时的计算旁路，在不删除视觉token、不修改模型权重的前提下剪枝视觉语言模型中不必要的视觉计算，从而降低推理成本。 |
| [^57] | [Operator-informed initialization for Fourier features physics-informed neural networks](https://arxiv.org/abs/2610.03378) | 本文通过在NTK理论框架下分析傅里叶特征物理信息神经网络的训练动力学，提出了一种根据所求解偏微分方程的微分算子特性来定制初始权重分布的初始化策略，从而消除算子导致的频谱偏差并提升预测精度。 |
| [^58] | [SCAD: Structured Credit Assignment and Distillation for Long-Horizon Agents](https://arxiv.org/abs/2610.03372) | SCAD通过将长时程智能体的交互分解为规划与有界子任务执行，并将基于结果的信用分配与教师引导蒸馏相结合，在文本和多模态任务上显著优于最强训练基线。 |
| [^59] | [Mixture-of-Experts for Cryptocurrency Order Execution: Training Stability, Tail Risk, and Failure Modes](https://arxiv.org/abs/2610.03369) | 该论文在BTC/USDT订单簿数据上系统评估了专家混合（MoE）强化学习用于订单执行的效果，发现K≥4的MoE架构能显著提升训练稳定性并消除原始DDQL中拖延至被迫清算的失效模式，但没有任何学习型配置能在平均执行缺口上超越TWAP和立即清算等简单基准。 |
| [^60] | [Follow the Winners: Conservative Policy Improvement with the Cross-Entropy Method for Critic-Free RFT](https://arxiv.org/abs/2610.03361) | FTW 是一种无批评家的强化微调算法，通过将交叉熵方法适配到 RFT 中、用回放缓冲区样本上的序数过滤器替代组采样，从而在有状态环境中难以重复采样的智能体大模型训练中实现保守且稳健的策略改进。 |
| [^61] | [Cordial Learning: Distributed Training with Correlated Data](https://arxiv.org/abs/2610.03330) | 提出了一种名为“亲和学习”的分布式训练框架，通过智能体间仅共享低维输出、本地模型提取同伴信息来处理相关数据问题，并在线性模型假设下证明了其以概率一收敛到全局最优。 |
| [^62] | [SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models](https://arxiv.org/abs/2610.03329) | 提出SyntaxBench诊断基准，通过五个核心字符级任务和一个高难度子串提取压力测试，结合Cohen's kappa与McNemar检验等统计方法，系统评估了八个开放权重大语言模型的字符级推理能力。 |
| [^63] | [DAWIS: Data Assimilation with Windowed Inverse Sampling via Multitask Interpolants](https://arxiv.org/abs/2610.03314) | DAWIS提出了一种统一的数据同化框架，通过用覆盖连续状态窗口的多任务随机插值子替代单一流时间先验，使滤波、固定滞后平滑和分块平滑得以在同一框架内实现，从而能在新观测到达时修正过去状态并避免误差累积。 |
| [^64] | [SDECast: Probabilistic Weather Forecasting in Continuous Time with Neural SDEs](https://arxiv.org/abs/2610.03313) | SDECast提出了一个基于神经随机微分方程的连续时间概率天气预报框架，无需训练时重复SDE模拟即可直接在物理空间学习随机动力学，并成功扩展到小时级分辨率的全球天气预报。 |
| [^65] | [Training-Loss Guarantees for Muon with Finite-Step Newton--Schulz Orthogonalization](https://arxiv.org/abs/2610.03306) | 本文首次为Muon优化器建立了同时考虑动量累积与有限步调优牛顿-舒尔茨正交化的训练损失保证，证明了在具有正定极限神经切向核的宽两层ReLU网络上，Muon能以高概率达到任意目标损失，命中时间界为 $O((1-\mu)^{-1}\varepsilon^{-1/2})$。 |
| [^66] | [S$^{2}$-PINN: Stochastic Separable Physics-Informed Neural Networks](https://arxiv.org/abs/2610.03303) | 提出S²-PINN，利用可学习高斯空间字典、傅里叶时间特征与gPC随机基并通过低秩CP张量分解耦合的可分离表示，高效求解随机偏微分方程的不确定性量化问题，克服了经典谱方法的维数灾难。 |
| [^67] | [JOVE: Joint Execution and Verification for Resource-Aware LLM Task Graphs](https://arxiv.org/abs/2610.03296) | JOVE提出了一种在线框架，通过联合决策LLM执行分配与中间输出的付费验证，在长期预算和延迟约束下平衡即时执行开销与未来学习收益，从而在LLM服务质量未知的情况下提升任务图执行的效率与正确性。 |
| [^68] | [Wrong Organ, Right Physics: Transferring Echocardiography Pretraining to Lung Ultrasound for Tuberculosis Screening](https://arxiv.org/abs/2610.03290) | 本研究发现在低资源肺部超声结核病筛查任务中，编码器的预训练领域选择（超声心动图还是通用视频）对分类性能影响甚微，而特征条件化才是决定任务表现的关键因素。 |
| [^69] | [Architecture-Dependent Fusion Pathways in MLLMs](https://arxiv.org/abs/2610.03289) | 本文通过对两种架构范式的多模态大语言模型进行系统性分析和因果干预实验，揭示了其内部模态融合机制的根本差异，发现拼接架构模型遵循“文本优先、视觉靠后”的融合路径，而原生多模态架构则呈现不同的融合模式。 |
| [^70] | [SPEAR: A Spectral-Disentangled MoE Neural Operator with Knowledge-Guided Expert Aggregation for Large-Scale PDE Pretraining](https://arxiv.org/abs/2610.03265) | SPEAR通过将特征谱解耦为低频与高频分量以实现共享与专门化建模，并结合基于数据集知识与路由偏好的知识引导专家聚合策略来消除专家冗余，有效解决了PDE基础模型中的知识干扰与专家冗余问题，提升了大规模PDE预训练的泛化能力。 |
| [^71] | [PaMIR: Open Benchmark of Public Credit-Default Datasets](https://arxiv.org/abs/2610.03259) | PaMIR是首个汇集19个公开信贷违约数据集（来自九个国家的124万条记录）的开放基准，采用统一的无泄漏构建流程和按标签预算报告的标签延迟流评估协议，为标签稀缺且延迟的信贷违约预测提供可复现的评测标准。 |
| [^72] | [Mapping and Advancing the Scalability-Accuracy Frontier of Nonlinear Causal Discovery](https://arxiv.org/abs/2610.03258) | 本文系统比较了四类非线性因果发现方法在可扩展性与准确性上的互补瓶颈，并提出基于样条的得分评估方案SPADE，通过一次性编译并复用充分统计量，在保持准确性的同时大幅提升组合搜索的效率。 |
| [^73] | [Cross-cohort TB classification using clinical data gathered in Uganda and South Africa](https://arxiv.org/abs/2610.03256) | 该研究首次评估了基于南非和乌干达两个国家临床数据的机器学习跨队列结核病筛查，并提出一种联合优化特征选择与特征排序的CNN策略，显著提升了筛查性能。 |
| [^74] | [D2K-Bench: Can LLM Agents Turn Expert Designs into Efficient GPU Kernels?](https://arxiv.org/abs/2610.03226) | D2K-Bench 是一个包含 26 个任务和 85 个工作负载的诊断性基准，通过分层专家设计指导（算法洞察、数据流设计与底层优化技巧）来系统评估 LLM 智能体生成高效 GPU 内核的能力，结果显示专家指导可将正确率从 93.1% 提升至 98.5% 并显著提高性能。 |
| [^75] | [The Neuro-Physical Inverter: A Modular Framework for Magnetotelluric Inversion Coupling Ensemble Conditioning with Residual Learning](https://arxiv.org/abs/2610.03225) | 提出了一种模块化的神经物理反演器（NPI）框架，通过将集成条件高斯过程与残差学习神经网络相结合，实现了具备不确定性量化能力的大地电磁反演，在不破坏集成稳定性的情况下系统性降低误差。 |
| [^76] | [AdaStep: Adaptive Step Credit Weighting for Agentic Reinforcement Learning](https://arxiv.org/abs/2610.03223) | 提出AdaStep方法，将步骤信用加权建模为均方误差估计问题并推导出最优逐状态收缩系数，从而在稀疏奖励下可靠地融合步骤级与轨迹级监督信号，提升长程LLM智能体的强化学习训练效果。 |
| [^77] | [Near-Optimal Convex Optimization with Lazy Second-Order Oracles](https://arxiv.org/abs/2610.03222) | 本文针对惰性二阶预言机凸优化问题，通过新的分块零链下界构造和匹配的算法设计，将复杂度界改进至 $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$ 并在对数因子内紧致，显著优于先前结果。 |
| [^78] | [Evolving Hybrid Quantum-Classical Architectures for Image Classification](https://arxiv.org/abs/2610.03220) | 该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。 |
| [^79] | [Kernel Singular Value Decomposition with Extension to Multiple Data Sources](https://arxiv.org/abs/2610.03216) | 该论文提出eKSVD，将核奇异值分解从两个数据源扩展到多个数据源，通过对偶优化推广了Lanczos分解定理中的移位特征值问题，并给出了结合神经网络的基于协方差的框架，实现非对称核上的联合非线性特征学习。 |
| [^80] | [HyperFuse: Fast Self-Supervised Node Embeddings for Attributed Hypergraphs](https://arxiv.org/abs/2610.03211) | HyperFuse提出了一种无需标签的快速超图表示学习流水线，通过谱松弛坐标计算、多尺度特征摘要和仅训练100轮的轻量级编码器，大幅降低了属性超图节点嵌入的生成成本。 |
| [^81] | [Hamiltonian locality testing and certification do not achieve the Heisenberg limit](https://arxiv.org/abs/2610.03205) | 该论文证明了哈密顿量k-局域性测试和哈密顿量认证问题都需要Ω(1/ε²)的总演化时间，首次为哈密顿量学习与测试中的自然问题建立了排除1/ε海森堡极限标度的下界。 |
| [^82] | [Predictively Oriented Gaussian Process Posteriors](https://arxiv.org/abs/2610.03201) | 提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。 |
| [^83] | [Predicting and Repairing Merge Collapse in Large Language Models](https://arxiv.org/abs/2610.03199) | 该论文提出用基于专家模型任务向量方差的“干扰度”评分，在大语言模型合并前预测是否会崩塌并指导修复，实验表明只有破坏性合并会超过该评分阈值，而现有合并算子常用的符号冲突统计量反而具有反向预测作用。 |
| [^84] | [Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case](https://arxiv.org/abs/2610.03190) | 该论文研究了LLM调查员“何时应结案”的证据充分性判断问题，发现未经训练的小模型和前沿模型都普遍夸大证据充分性而过早结案，并提出需对照来源捷径规则来评估结案能力的方法。 |
| [^85] | [Sample complexity of variance-reduced policy gradient: weaker assumptions and lower bounds](https://arxiv.org/abs/2610.03165) | 本文提出基于防御性重要性采样的防御性策略梯度算法，无需对重要性权重方差做任何假设即可实现 $O(\epsilon^{-3})$ 样本复杂度，并在广义黑盒策略优化模型中建立了匹配的最优速率下界。 |
| [^86] | [Landscape-Dependent Performance of Photonic Quantum Solvers in QUBO Feature Selection for Financial Risk Detection](https://arxiv.org/abs/2610.03161) | 该论文在信用卡欺诈与消费者违约两个金融数据集上，对经典分支定界（Gurobi）、光子熵计算（QCI Dirac-3）与模拟光子玻色采样（Piquasso）三种计算范式的十三种QUBO特征选择方法进行了系统基准测试，揭示了求解器性能高度依赖于数据集地形：Dirac-3在ULB欺诈数据集上仅用13/30个特征即可匹敌全特征模型（平均F1约0.873），而在159特征的AmEx违约数据集上所有范式均需接近全部特征才能达到F1≈0.80。 |
| [^87] | [Does Physics Live in the Activations? Localizing Physical Quantities in Video Diffusion Models](https://arxiv.org/abs/2610.03154) | 该论文发现视频扩散Transformer在去噪早期阶段就能在其内部激活（尤其是物体对应token处）中以线性方式编码运动学与刚体动力学等物理量，表明物理信息是模型在去噪过程中主动构建的，而非输入中自带的。 |
| [^88] | [TSGuard: A Real-Time Framework for Detecting and Imputing Missing Data in Streaming Time Series](https://arxiv.org/abs/2610.03147) | TSGuard是一个实时框架，将轻量级图感知时间序列填补模型与约束感知验证、回退估计相结合，实现流式时间序列中缺失数据的检测、填补与验证，并将其融入完整的数据质量闭环。 |
| [^89] | [Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under Grouped-Query Attention](https://arxiv.org/abs/2610.03135) | 提出Page-EntroKV框架，在GQA实际分配的页粒度上，利用sink隔离的Rényi-2碰撞熵加权池化各查询头的分数进行KV缓存驱逐，避免因各头独立选择导致的缓存膨胀，同时保护专用检索头。 |
| [^90] | [Safe Streaming Flow Planning by Aligning Sampling Dynamics with Execution Dynamics](https://arxiv.org/abs/2610.03132) | SafeStreamingFlow通过让流采样动态与系统执行动态对齐、并仅对当前执行步用高阶控制障碍函数强制安全约束，实现了计算高效且安全的流式在线规划。 |
| [^91] | [Coverage You Can Steer: Online Conformal Calibration for RL-Driven Hardware-Aware NAS](https://arxiv.org/abs/2610.03127) | 针对强化学习搜索中候选分布漂移破坏保形预测覆盖保证的问题，本文提出用自适应保形推断的在线反馈控制替代一次性分位数估计，使剪枝过滤器的覆盖率可被单调、可复现地调控，从而在硬件感知NAS中恢复了对任意序列的分布无关保证。 |
| [^92] | [ParaGeo: Decomposing Paralinguistic Variation into a Shared Latent Geometry](https://arxiv.org/abs/2610.03125) | ParaGeo通过内容匹配分解方法证明，冻结语音语言模型内部存在跨语言内容稳定可复现的副语言属性共享低维潜在几何结构。 |
| [^93] | [The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs](https://arxiv.org/abs/2610.03124) | 该论文首次形式化了开放权重大语言模型中的触发-标记滥用检测机制，将其分为令牌级和权重级两类，并系统研究揭示了此类机制在对抗性攻击下的脆弱性。 |
| [^94] | [How to Find and Reuse Policies for Continuous Adaptation in Lifelong Reinforcement Learning](https://arxiv.org/abs/2610.03119) | 提出AMSC方法，利用基于Wasserstein任务嵌入的在线相似性估计，自适应地选择和组合多个先前策略作为先验，从而在终身强化学习中获得更高的平均性能、前向迁移能力且不遗忘旧知识。 |
| [^95] | [Exploring the Trade-Off Between Structured Pruning and Fault Tolerance in Deep Neural Networks for Space Applications](https://arxiv.org/abs/2610.03117) | 本研究通过故障注入实验发现，深度神经网络的结构化剪枝虽然因减少冗余而提高了单次推理对单粒子翻转故障的敏感性，但更短的执行时间降低了遭遇故障的概率，从而有效平衡了这一负面影响。 |
| [^96] | [S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales](https://arxiv.org/abs/2610.03106) | 该论文提出S2S-JEPA，首次将计算机视觉中的联合嵌入预测架构（JEPA）范式引入次季节到季节（S2S）预报，通过在潜在空间中只预测缓慢变化且可预测的分量、舍弃不可预测的细尺度细节，来突破AI天气模型在两周以上“可预测性荒漠”中的性能瓶颈。 |
| [^97] | [Invariance of Clustering Operations in Causal Effect Identification](https://arxiv.org/abs/2610.03101) | 本文提出了一类基于原图c-分量条件的“识别不变”聚类操作，可同时保持因果效应的可识别性与不可识别性，从而安全地简化因果图并加速因果效应识别。 |
| [^98] | [Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression](https://arxiv.org/abs/2610.03098) | 提出潜空间密码子优化方法LSCO，通过将序列映射到预训练mRNA语言模型的潜空间，将离散的密码子优化问题转化为可梯度搜索的连续问题，并结合不确定性感知的表达预测器、最小自由能正则化、自然性先验和约束解码，以最大化蛋白质表达。 |
| [^99] | [LS-AR: Future-Predictive Latent Steering in Autoregressive LLMs](https://arxiv.org/abs/2610.03093) | 提出双通道架构LS-AR，通过FiLM条件化将连续目标引导与离散token解码解耦，在超出上下文限制的长时程检索中实现100%召回率，同时吞吐量提升约35%、峰值显存降低52.8%。 |
| [^100] | [ULTRADISCOVERY: Abductive Exploration in an Interconnected, Epistemically Open Universe](https://arxiv.org/abs/2610.03092) | 该论文提出ULTRADISCOVERY交互式基准，通过2×2设计独立控制表征开放性与证据分布性，评估智能体在认识论开放且结构互联的世界中进行溯因科学探索的能力，发现现有十一个模型均无法通过引入新实体或重写变量来完成理论替换。 |
| [^101] | [Zephon: Elastic Determinism for Online, Stateful Foundation Model Data Loading Pipelines](https://arxiv.org/abs/2610.03087) | Zephon提出了一种面向基础模型训练的数据加载器，能够在GPU拓扑变化、频繁检查点恢复及不同执行后端的情况下，为包含在线分词、打包、混合等有状态n对m转换的数据流水线提供确定性的全局训练数据批次序列（即弹性确定性）。 |
| [^102] | [Light Entropic Optimal Transport on Riemannian Manifolds](https://arxiv.org/abs/2610.03085) | 该论文提出ManifoldLightOT，一种通过构造几何特定的Gibbs核与相容势参数化，直接在球面、环面、SO(3)、SE(3)等黎曼流形上学习核诱导熵最优传输耦合的轻量级方法，实现了闭式归一化、条件分布的直接采样，并可自然扩展至流形乘积。 |
| [^103] | [NegT2IBench: When Negation Changes the Picture. A Polarity Benchmark for Text-to-Image Models](https://arxiv.org/abs/2610.03084) | 提出了NegT2IBench基准，通过4,800条按极性组织的提示词系统评估文本到图像模型满足否定约束的能力，其基于检测器的评分以更小的规模达到了与大型视觉语言评判器相当的人类一致性水平。 |
| [^104] | [Smart Sensing for Safer Bridges: From Sensor Signals to AI-Driven Anomaly Detection](https://arxiv.org/abs/2610.03082) | 本文提出将信号处理与孤立森林两种互补方法应用于挪威桥梁实时传感器数据的异常检测，并通过多种评估指标和受控异常注入分析系统比较了两种方法的检测特性与敏感性。 |
| [^105] | [MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code](https://arxiv.org/abs/2610.03080) | 该论文提出MintEval基准，通过程序化生成参考交易策略并回译为自然语言指令让大语言模型重新实现，再在相同市场数据上逐K线比较生成策略与参考策略的实际交易行为（而非代码相似度或利润），以检验大语言模型编写的策略代码是否真正做到了行为等价于交易者的原始意图。 |
| [^106] | [Learn Feasibility Once, Optimize All Objectives: Derivative-Free Diffusion Models for Chance-Constrained Programming](https://arxiv.org/abs/2610.03071) | 该论文提出 D³Opt 框架，通过训练一个与目标无关的风险条件扩散模型一次性学习机会可行结构作为可重用先验，并在推理时利用退火粒子 Feynman–Kac 校正，仅凭函数评估即可优化任意后续指定的目标，实现约束建模与目标优化的解耦。 |
| [^107] | [Explainable Molecular Structure Inference from GC--MS with Diffusion Models and LLM Reranking](https://arxiv.org/abs/2610.03066) | 本文提出DiffGCMS框架，先用谱条件离散图扩散模型从GC-EI-MS谱图从头生成候选分子结构，再由大语言模型进行验证、修复、重排序并提供可解释的碎片离子分析，从而实现对谱库中缺失化合物的可解释分子结构推断。 |
| [^108] | [Learning Transferable Policies from Action-free Time Series Through Dynamical Embeddings](https://arxiv.org/abs/2610.03065) | 本文提出一个基于模型的层次化强化学习框架，通过低维动力学嵌入捕捉相关系统间的共享动力学结构，实现从无动作记录中学习可迁移的特定系统控制策略。 |
| [^109] | [When Does Synthetic Relational Data Teach Models to Use Relations? Tracing Predictive Structure from Pretraining Data to Model Behavior](https://arxiv.org/abs/2610.03057) | 该研究通过对比四种数据生成器训练出的Relational Transformer模型，发现只有当外键关联的跨表信息对掩码单元格预训练目标具有预测必要性时，模型才会真正学会利用关系结构。 |
| [^110] | [RIPPLE in Still Water: Zero-Shot Clustering in Federated Learning with Wavelet Scattering Transform](https://arxiv.org/abs/2610.03054) | RIPPLE提出了一种零样本聚类联邦学习框架，通过小波散射变换嵌入的方差加权主成分原型与服务器端预训练的高斯混合VAE，完全离线地完成客户端聚类分配，实现与FedAvg相同的通信成本并保护梯度隐私。 |
| [^111] | [HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning](https://arxiv.org/abs/2610.03039) | HyperThink通过轻量级超网络将长思维链推理计算摊销为一次查询条件下的参数更新，使模型无需生成冗长思考轨迹即可直接输出简洁解答，在大幅降低推理延迟和token消耗的同时保持强推理性能。 |
| [^112] | [WebFovea: When the Model Is Right but the Click Is Wrong -- Reliable Round Trips for Vision-Based Web Agents on Live Websites](https://arxiv.org/abs/2610.03036) | 本文提出在WebRetriever Challenge 2026中获得亚军的视觉网页智能体WebFovea，并指出真实网站上的许多失败并非源于模型推理，而是源于模型与浏览器之间中间执行层在动作解析、页面生效、结果反馈和信息展示这四个环节上的问题。 |
| [^113] | [Balancing Multimodal Learning via Functional Progress](https://arxiv.org/abs/2610.03035) | 提出FGMO方法，通过函数空间的进展信号评估各模态的优化进展并协调跨模态优化，克服了传统基于分数差异的估计方法将模态内在差异误判为进展差距的问题，有效缓解多模态学习中的模态不平衡。 |
| [^114] | [Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling](https://arxiv.org/abs/2610.03034) | 该论文将PI步长控制与扩散噪声归一化误差估计器结合，提出了扩散模型的自适应二阶求解器，实现更平滑的步长调整，并可将逐样本的自适应轨迹聚合为固定调度，在大幅降低采样成本的同时保留自适应采样的质量收益。 |
| [^115] | [Tailoring the Quantization Space for 1-Bit KV Cache Compression](https://arxiv.org/abs/2610.03027) | 提出TaSQ方法，通过查询引导的通道加权、跨头归一化和协方差感知的通道分组来量身定制向量量化目标空间，从而在1比特极端压缩下实现有效的KV缓存压缩。 |
| [^116] | [Verifiable, Articulable, and Tacit Components of Preference](https://arxiv.org/abs/2610.03025) | 该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。 |
| [^117] | [Signal Simplification Is Not Predictive Simplification: Diagnosing Residual Neural Forecasting in Short-Horizon Volatility](https://arxiv.org/abs/2610.03019) | 研究通过信号-预测-系统诊断框架证明，统计第一阶段对残差信号的简化（方差大幅降低、自相关微弱）并不等于残差更易被神经网络学习——残差LSTM增强反而全面恶化了短周期波动率预测，而纯LSTM表现最佳。 |
| [^118] | [AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning](https://arxiv.org/abs/2610.03007) | AvoKV-E是一种无需训练的KV缓存淘汰策略，通过延迟近期状态的淘汰资格，并结合读取压力、键冗余度和值负载潜力对缓存条目排序，在长推理任务中以相同的活跃KV预算达到或超越现有基线方法。 |
| [^119] | [Neural Data Needs Semantic Tokenization: Behavioral Events as Boundaries of Session-Transferable Tokens](https://arxiv.org/abs/2610.03001) | 该论文提出基于状态的标记化方法，以行为事件为边界将每次试验中的群体活动状态转换为可跨会话迁移的群体几何标记，从而解决神经基础模型因神经元记录每次会话都不同而无法泛化到新会话的核心问题。 |
| [^120] | [Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability](https://arxiv.org/abs/2610.03000) | 该论文提出利用双曲几何的庞加莱模型构建多层感知机训练过程的时间参数图（即多个训练步骤的快照），以捕捉网络加权拓扑与自组织在训练轨迹中的几何演化，从而超越传统单检查点方法实现内在可解释性。 |
| [^121] | [Sentry: Learning to Recover from LLM Agent Failures at Test Time](https://arxiv.org/abs/2610.02994) | 提出Sentry——一个与LLM智能体并行运行的失败管理层，将失败经验视为条件性知识，在检测到失败时按需从外部经验手册中检索指导恢复、无奖励验证恢复结果并仅在确认恢复后存储新经验，从而在测试时实现从失败中学习。 |
| [^122] | [Differentiable Koopman Operator for Contrastive Learning on Dynamic Graphs](https://arxiv.org/abs/2610.02990) | KAIROS框架通过在动态图对比学习中嵌入可微Koopman算子来线性化时间演化，利用多粒度对比目标学习稳健的节点表示，并结合Koopman预测残差实现异常检测。 |
| [^123] | [Dirac-Interconnected Neural Elements: Discovering Modularity in Physical Systems Without Reduction](https://arxiv.org/abs/2610.02960) | 提出狄拉克互联神经元件（DINEs），将物理系统建模为由狄拉克结构给出代数约束的微分代数方程，无需预先已知互联结构或将系统约化为常微分方程，即可从数据中同时识别出物理系统的模块化结构。 |
| [^124] | [Understanding Trajectory Heterogeneity in Federated World Model Learning](https://arxiv.org/abs/2610.02957) | 该论文在临床时间序列数据上系统基准测试了联邦世界模型学习中的轨迹异质性问题，发现客户端数据所有权与参与率会共同限制长时程窗口的可用覆盖，揭示了跨时间联邦学习的核心训练瓶颈。 |
| [^125] | [Hyperparameter selection for equation learning with biologically-informed neural networks](https://arxiv.org/abs/2610.02954) | 该论文提出了一种在真实控制方程未知时仍可使用的诊断式工作流程，用于生物信息神经网络（BINNs）方程学习中的超参数选择，从而提升结果的可重复性与方法的可迁移性。 |
| [^126] | [SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention](https://arxiv.org/abs/2610.02953) | SlimKV提出了一种与问题无关的令牌-特征联合KV缓存压缩方法，通过低秩感知训练将长上下文压缩为携带潜在KV表示的信标记忆状态，并利用键侧RoPE移除对信标令牌影响较小的位置不对称性，实现无需全维度重构的高效解码加速。 |
| [^127] | [GTDD: Generative Test-Driven Development for AI Coding Agents with Adversarial Testing](https://arxiv.org/abs/2610.02952) | 提出生成式测试驱动开发（GTDD），由独立的测试智能体在每轮候选实现后基于人类指定的行为契约对抗性地生成新测试输入并不断反馈反例，以解决AI编码智能体过拟合固定测试集而导致预期行为遗漏的问题。 |
| [^128] | [Dynamic Expert Pruning for Multi-Agent Systems](https://arxiv.org/abs/2610.02951) | 提出动态专家剪枝（DEP）方法，利用智能体的系统与任务提示动态识别并按需剪枝专家，解决了静态专家剪枝在多智能体异构工作负载下失效的问题。 |
| [^129] | [FASTDIAR: Frame-level speaker encoder for Streaming Diarization](https://arxiv.org/abs/2610.02941) | 该论文提出FASTDIAR，一种因果帧级说话人编码器，每80毫秒输出一次嵌入并结合基于自相似性门控的在线聚类，实现了亚秒级延迟下最准确的流式说话人分离，在说话人数量增加时性能衰减更小，且可在CPU上以实时五倍的速度运行。 |
| [^130] | [Discriminating Fixture Coverage in Agent-Infrastructure Verification Suites](https://arxiv.org/abs/2610.02928) | 用变异分析衡量“在正确实现上通过、在缺陷实现上失败”这一标准证据的价值，发现该证据无法保证测试夹具覆盖——即使修复后的套件仍有五个对抗性变异体存活，其中三个因没有任何夹具能激活它们而从未暴露。 |
| [^131] | [Learning Jazz Pianist Style with Cross-Attention Conditioning](https://arxiv.org/abs/2610.02918) | 该研究表明预训练符号音乐Transformer的表征已能编码钢琴家身份，并进一步通过交叉注意力机制注入钢琴家身份嵌入，实现了以特定爵士钢琴家风格为条件的音乐生成，生成结果可被分类器高精度地归因于正确的钢琴家。 |
| [^132] | [Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models](https://arxiv.org/abs/2610.02911) | 该论文提出PTH检查集，证明实验框架的细节（如PPO比率计算方式、数据种子传递、重放队列复用和损失归一化器实现）可以逆转陈旧数据强化学习方法比较的排名，强调了检查实验基础设施的必要性。 |
| [^133] | [Frequency Is Not Sensitivity Identifying Safety-Sensitive Experts in Sparse MoE LLM](https://arxiv.org/abs/2610.02910) | 该论文提出用路由器梯度敏感度（即序列损失对专家门控权重的敏感度）替代传统的激活频率来识别稀疏MoE大语言模型中的安全关键专家，实验表明该方法在五种架构上能更准确地预测抑制哪些专家会削弱模型的安全拒绝能力。 |
| [^134] | [Constraint-Aware Training](https://arxiv.org/abs/2610.02909) | 本文提出约束感知训练方法，将推理时由程序分析执行的约束从模型学习中外化出来，并通过三个定理证明这种外化能带来模型规模和数据效率上的收益，即在相同参数量下获得更低的预测损失。 |
| [^135] | [Do ResNets Route? Sparse Interaction Experts in Residual Networks](https://arxiv.org/abs/2610.02907) | 该论文通过Möbius反演将训练好的ResNet精确分解为各残差分支的独立贡献与高阶交互，发现ImageNet预训练模型的交互质量在第5阶和第10阶达到峰值，且主导交互随输入和预测类别动态变化，揭示了残差网络中存在隐式的稀疏“路由”行为。 |
| [^136] | [Tangent Schr\"odinger Bridge Matching: Learning Stochastic Transport with Mechanistic Sensitivities](https://arxiv.org/abs/2610.02906) | 该论文提出切线薛定谔桥匹配方法，通过在轨迹传播中同时监督参数导数（机制敏感性）来学习随机输运，使模型能够准确预测随机系统在参数干预下的响应，并从理论上证明敏感性精度可界定有限变化预测误差与决策遗憾。 |
| [^137] | [When Can We Trust the Matching Principle? Robust Deployment Geometry Under Finite-Sample and Model Uncertainty](https://arxiv.org/abs/2610.02894) | 该论文提出用信任比率 tau = ε/γ 来量化匹配原则何时可靠（估计投影匹配的漂移在 Davis-Kahan 分离区域内按 O(τ²) 增长），并据此设计置信度校准匹配（CCM）策略：τ 小时进行方向性匹配，τ 大时渐进各向同性扩散惩罚，从而在有限样本与模型不确定性下实现鲁棒部署。 |
| [^138] | [DyRA: Dynamic Residual Approximation for Efficient Matrix Multiplication in DNNs](https://arxiv.org/abs/2610.02882) | 提出了输入自适应方法DyRA，通过在推理过程中动态近似并校正结构化权重近似引入的输出残差误差，直接优化输出的低秩因子，从而更高效、更精确地实现深度神经网络中的矩阵乘法近似。 |
| [^139] | [Toward Omni Multimodal Graph Foundation Model: A Topology-Driven Binding Approach](https://arxiv.org/abs/2610.02881) | 提出GraphBind，一种拓扑驱动的多模态图基础模型方法，利用图拓扑的稳定性作为结构参考和互补语义来源，将异构节点的丰富模态信息绑定到统一共享空间中，以解决现实多模态属性图中节点属性不完整的问题。 |
| [^140] | [Peer Effects in Signed Networks: Separating Influence Through Positive and Negative Ties](https://arxiv.org/abs/2610.02872) | 该论文提出SiDE双重稳健估计器，首次在带符号网络中通过区分正、负关系及其交互作用来识别同伴效应，并给出识别公式和双重稳健性保证。 |
| [^141] | [Distributionally Robust Survival Models under Subpopulation Shift and Outlier Contamination](https://arxiv.org/abs/2610.02868) | 本文提出一个分布鲁棒生存分析框架，通过外层最小化削弱离群样本的影响、内层最大化聚焦最不利的子群体，从而联合应对子群体偏移与离群点污染，并直接兼容Cox风险集等不可分解的生存损失。 |
| [^142] | [DIVINE: Simple Cross-Market Stock Pretraining via Diverse Indicator Reconstruction](https://arxiv.org/abs/2610.02866) | DIVINE 提出一种简单的跨市场股票预训练框架，通过从原始 OHLCV 历史数据重建 16 个标准技术指标衍生的 77 个目标作为监督信号，同时避免了未来监督的不确定性并保持与收益预测的对齐，仅用 0.05M 参数的轻量级编码器就在六个股票市场上取得了最强的平均投资组合表现。 |
| [^143] | [On Unlearning for Time-series Forecasting](https://arxiv.org/abs/2610.02865) | 该论文针对时间序列预测中的机器遗忘问题，指出基于梯度的遗忘方法因被删除观测值会跨多个因果关联的预测窗口传播参数更新而不稳定，并探讨标签引导更新作为更可控的替代方案。 |
| [^144] | [NeuroLens: Learning Latent Embeddings of Neural Semantics from Chronic Recordings](https://arxiv.org/abs/2610.02864) | 提出基于JEPA框架的自监督模型NeuroLens，通过在潜在空间中学习去噪的语义表征，从长期神经记录中区分神经表征可塑性与记录不稳定性。 |
| [^145] | [Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models](https://arxiv.org/abs/2610.02860) | 该论文提出一种贯穿模拟器状态、栅格观测、目标嵌入与预测器输出的反事实动作评估协议，发现联合嵌入预测世界模型存在观测瓶颈并系统性低估动作路径——预测器对真实反事实动作的响应远小于对匹配的各向同性噪声扰动。 |
| [^146] | [Permutation Robustness Is Not Enough: Action Collapse in Multi-Agent Transformer Policies](https://arxiv.org/abs/2610.02848) | 本文发现低置换误差可能是假象——所有智能体选择相同动作也会显得鲁棒，提出用置换一致性指标与动作坍缩诊断（动作多样性、相同动作比例、最大动作频率）共同评估多智能体Transformer策略，并证明弱等变性惩罚能在提升置换鲁棒性的同时保持动作多样性。 |
| [^147] | [Turnover-Orthogonal Credit Assignment for Open-Team Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.02847) | 本文提出流动正交信用分配（TOCA），一种面向开放团队多智能体强化学习的价值分解方法，通过将动作效应与外生人员流动效应分离，避免存活智能体因其无法控制的成员变动而受到错误的奖励或惩罚。 |
| [^148] | [Understanding Enrichment in Reinforcement Learning](https://arxiv.org/abs/2610.02846) | 本文从数学上揭示了RLVR中省略或截断重要性权重纠正会隐式地重新加权奖励，并将梯度误差分解为尺度、旋转和方差以解释其对学习的不同影响，同时提出了一种新颖的序贯蒙特卡洛权重纠正机制来缓和方差的指数级增长。 |
| [^149] | [To Explore The Strange New World Beyond Data Distribution: System Behavior, Causality Tax, and Non-causal Base Model](https://arxiv.org/abs/2610.02839) | 该论文提出SBD框架，将数据分布之外的系统行为作为不可约的贝叶斯组件纳入证据下界，从理论上揭示了反直觉的“因果性税”现象，表明语言模型的因果性可能既非必要也非最优。 |
| [^150] | [All Work And No Play Makes Jack a Dull Boy: Understanding and Preventing Catastrophic Strategy Collapse in RLVR](https://arxiv.org/abs/2610.02835) | 该论文揭示了RLVR后训练中GRPO类算法发生灾难性策略坍缩的机制——优化目标会将概率质量集中到单一策略，而维持任务准确率又需要最低策略容量，二者的冲突导致坍缩，并提出镜像纠缠指数（MEI）作为轻量级的在线预警信号。 |
| [^151] | [FastOPD: On-Policy Distillation for Lightweight VLA Deployment](https://arxiv.org/abs/2610.02832) | FastOPD提出了一种高效的在策略蒸馏框架，将流图单状态教师监督与自洽性目标相结合，把大规模VLA基础模型压缩为可实际部署的轻量化模型，并从理论上保证学生模型能恢复出与理想少步教师模型相当的分布。 |
| [^152] | [Clinical Concept Centers in LLMs](https://arxiv.org/abs/2610.02829) | 该论文首次将机制可解释性评估从文本层面扩展到潜在空间应用于临床决策支持，发现在全部十一个测试的开源大语言模型内部都存在专门的临床概念中心，即临床概念以可定位且被因果使用的表征形式存在于模型潜在空间中。 |
| [^153] | [FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training](https://arxiv.org/abs/2610.02828) | FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。 |
| [^154] | [Adaptive Spectral-Koopman Dynamics Modeling for Temporal Domain Generalization](https://arxiv.org/abs/2610.02822) | 提出AdaSpecK框架，通过谱正则化Koopman动力学建模提取去噪的低频轨迹，并结合上下文感知的异构模式提取机制，有效解决时序域泛化中的噪声过拟合与非平稳历史环境建模问题。 |
| [^155] | [RMCW: A Deletion-Robust Watermark Based on Reed--Muller Codes for Language Models](https://arxiv.org/abs/2610.02817) | 提出了一种基于里德-马勒码的大语言模型水印方法RMCW，通过密钥词汇划分注入水印结构，并利用局部子序列的里德-所罗门代数一致性检验，实现对删除攻击的鲁棒水印检测。 |
| [^156] | [Gated Slot Attention-2: Two-Sided Associative Memory Correction in Linear Attention](https://arxiv.org/abs/2610.02816) | 该论文提出门控槽位注意力-2（GSA2），通过门控Oja规则-2实现键侧记忆校正、门控Delta规则-2实现值侧记忆校正，首次在线性注意力中实现了双向联想记忆校正，从而更有效地管理固定大小的循环记忆。 |
| [^157] | [ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing](https://arxiv.org/abs/2610.02808) | ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。 |
| [^158] | [VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning](https://arxiv.org/abs/2610.02801) | VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。 |
| [^159] | [Muon Learns Facts Better: Understanding the Role of Spectral Orthogonalization](https://arxiv.org/abs/2610.02798) | 本文通过可解析的事实回忆模型和线性 Transformer 分析了 Muon 优化器中谱正交化的作用机制，揭示其对特征学习动力学的改变，并说明这使 Muon 比梯度下降和 Adam 更擅长学习“主体-关系到答案”的事实映射。 |
| [^160] | [Efficient Memory Crystallization for Graph Learning under Non-Stationary Distribution Shifts](https://arxiv.org/abs/2610.02795) | 该论文提出了一种无需训练的测试时框架EMC，通过闭式解将每个新到来的图域高效地“结晶”为紧凑且语义忠实的记忆，以低成本替代生成式记忆方案，从而应对图学习中的持续分布偏移。 |
| [^161] | [Hold-Out Scoring for Efficient Gaussian DAG Learning](https://arxiv.org/abs/2610.02785) | 该论文提出HOST算法，以逐节点留出评分与凸回归取代子集搜索，仅凭对得分误差的单侧控制即可恢复正确的节点排序，从而在无需入度上界的情况下实现高效的高斯DAG学习。 |
| [^162] | [OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation](https://arxiv.org/abs/2610.02781) | 提出两阶段训练框架：先以评分标准作为教师特权上下文进行在线策略蒸馏（RP-OPD）提供密集的token级监督，再以评分标准作为奖励进行强化学习，从而突破蒸馏的性能瓶颈。 |
| [^163] | [Controlling Polar Exposure to Delay Memorization in Diffusion Models](https://arxiv.org/abs/2610.02780) | 该论文提出质量门控去白化（QGD）控制器，通过管理优化的更新几何、渐进恢复固定增益动量，在保持生成质量的同时将扩散模型的记忆化（复制训练样本）延迟至与数据集规模成比例。 |
| [^164] | [LatticeSMC: Where to Spend Inference-Time Compute in Chunked Sequence Generators](https://arxiv.org/abs/2610.02774) | 该论文提出 LatticeSMC，一个在“块索引×去噪步骤”二维格点上运行的 Feynman-Kac 采样器，从理论上证明了块可加奖励下重采样应安排在前瞻代价最低之处、且前缀可评估奖励可作为精确势函数直接使用，从而实现了预算严格匹配、更有效的分块序列生成推理时引导。 |
| [^165] | [Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback](https://arxiv.org/abs/2610.02771) | 本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。 |
| [^166] | [No-Free-Graph: Learning When Multimodal Data Should Be Graphified](https://arxiv.org/abs/2610.02768) | 本文揭示多模态数据的图化并非总是有益，并提出MAG-SCOUT框架，在构建图之前评估其预期效用，将图构建从默认步骤转变为一种基于效用的选择性决策。 |
| [^167] | [Exact Memory-Time Optimization for Prefix-Cached Language Model Serving](https://arxiv.org/abs/2610.02766) | 该论文提出前缀证书保留（PCR）方法，将前缀缓存超时策略的优化精确归约为一次最小割计算，从而在内存占用与重计算时间之间实现全局最优权衡，并通过重放39,632个真实Mooncake请求验证了其有效性。 |
| [^168] | [Localized Conformal Safety Monitoring with Vision-Language Models for Autonomous Driving](https://arxiv.org/abs/2610.02765) | 本文提出SLLCP，一种叠加在冻结视觉-语言模型之上的局部化保形预测事后校准层，将VLM不可靠的安全预测转化为概率校准的安全预测集合，从而为自动驾驶中的碰撞风险监测提供可靠保障。 |
| [^169] | [A Controlled Audit of Personal AI Memory for Rating Prediction](https://arxiv.org/abs/2610.02764) | 该研究通过置换用户历史评分的受控实验发现，个人AI记忆系统（Mem0）在评分预测中并未有效利用历史条目-评分关联，甚至不如仅使用历史记录的简单岭回归模型。 |
| [^170] | [A Two-Stage Cascade for Near-Real-Time Forest Anomaly Detection from Sentinel-1 SAR Time Series](https://arxiv.org/abs/2610.02763) | 该论文提出了一种基于Sentinel-1 SAR时间序列的两级级联框架，结合鲁棒统计z-score检验与学习式确认门控，实现近实时的森林损失异常检测，同时克服了光学遥感的云层覆盖限制和SAR季节性后向散射变化的干扰。 |
| [^171] | [Jumping up and down: Denoiser diffusion models for discrete ordinal data](https://arxiv.org/abs/2610.02754) | 提出了JUD——首个以去噪器训练为核心、支持双向（向上和向下）扰动的离散有序数据扩散模型家族，凭借简洁的训练目标和灵活的双向扰动机制，在图像、音乐、基因计数等多种数据模态上取得了有竞争力的结果。 |
| [^172] | [Learning Query Encoders Can Be Hard Even When Vector Retrieval Is Geometrically Easy](https://arxiv.org/abs/2610.02749) | 该研究发现单向量查询编码器的实际检索质量往往远低于冻结文档索引在几何上所能支持的上限，并从理论上证明了学习查询编码器在计算上可能是困难的。 |
| [^173] | [Prospective Hindsight: Self-Calibrating Reinforcement Learning via Prediction-Reality Gaps](https://arxiv.org/abs/2610.02740) | 提出前瞻性后见之明（PH）这一自校准强化学习训练原则，通过衡量智能体动作前预测与反馈后评估之间的“惊讶度”差距来加权梯度，使学习自动聚焦于智能体自我模型中最不准确的盲点样本。 |
| [^174] | [Inner Momentum for Differentially Private Muon](https://arxiv.org/abs/2610.02738) | 该论文提出“内动量”方法，即在裁剪前将每个样本的 Muon 梯度在当前模型与近期模型历史上进行平均，从而抑制差分隐私训练中逐样本梯度裁剪对奇异向量几何结构的畸变，并给出畸变的 Frobenius 范数上界及有限次 Newton-Schulz 迭代保持极因子的理论保证。 |
| [^175] | [Bellman Error Minimization Via Linear Programming Normalization](https://arxiv.org/abs/2610.02730) | 本文提出一种结合深度神经网络与线性规划归一化的函数逼近方法，以降低高维动态规划和强化学习问题中的贝尔曼误差。 |
| [^176] | [Structural-Functional Brain Connectivity Generation via Multimodal Hypergraph-based Flow Matching](https://arxiv.org/abs/2610.02722) | 该论文提出多模态超环流匹配框架MHG-FM，通过超图神经网络编码器学习高阶表示并结合双交叉注意力进行双向跨模态融合，实现了结构连接与功能连接的联合生成和跨模态转换，克服了传统成对图方法无法捕捉高阶结构-功能关系的局限。 |
| [^177] | [Revisiting Visual Representation Enhancement of VLMs via Kernel Canonical Correlation Analysis](https://arxiv.org/abs/2610.02718) | 本文提出利用核典型相关分析（KCCA）在特征子空间上刻画视觉语言模型与DINOv2之间的表征对齐，从而增强CLIP等模型的细粒度视觉感知能力。 |
| [^178] | [Differential Privacy of Gradient Descent on Perturbed Objectives](https://arxiv.org/abs/2610.02716) | 该论文证明了在强凸光滑目标上，对扰动目标运行梯度下降的有限次迭代是高斯噪声的 $C^1$ 微分同胚（并给出雅可比最小奇异值的定量下界），从而可直接用换元法分析有限迭代的差分隐私，且对广义线性模型而言，迭代条件成立时隐私界不显式依赖环境维度。 |
| [^179] | [WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds](https://arxiv.org/abs/2610.02713) | WakeKV发现大多数注意力头在生成过程中会动态改变读取行为，并提出一种响应式、可逆的KV缓存驻留策略，将冷却的注意力头迁移到可恢复的CPU储备区而非冻结或永久驱逐，从而在相同内存预算下持续降低缓存未命中率。 |
| [^180] | [Characterizing the Performance Gap in Human Activity Recognition for Older Adults](https://arxiv.org/abs/2610.02711) | 该论文通过老年人自由生活HAR数据集MyMove发现，年轻人基准上的模型改进难以迁移到老年人数据并造成持续扩大的性能差距，而基于年龄多样化的UK Biobank数据集预训练的冻结自监督特征能显著提升老年人活动识别性能并缩小这一差距。 |
| [^181] | [MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads](https://arxiv.org/abs/2610.02705) | MuonIO 将 Muon 优化器的原则性更新扩展到嵌入表和语言模型输出头——对语言模型头采用 2→∞ 算子范数、对嵌入表采用 1→2 算子范数，从而以统一的范数感知更新取代 AdamW。 |
| [^182] | [Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers](https://arxiv.org/abs/2610.02701) | 该研究发现在粒子物理Transformer中，避免token丢弃的top-1混合专家模型能在几乎不增加计算量的前提下超越稠密基线，但增加存储专家数量的收益有限，且专家路由结构与分类性能并非单调相关。 |
| [^183] | [Learning from Evolving Errors: Adaptive Iterative Repair for On-Policy Distillation](https://arxiv.org/abs/2610.02700) | 该论文提出AIR-OPD框架，通过引导生成器针对学生模型不断演化的错误迭代合成修复引导，并让教师模型以该引导为特权上下文提供监督，从而实现“错误到修复”的在线策略蒸馏，避免了仅依赖参考解所带来的捷径风险。 |
| [^184] | [Test-time Calibration Learning for Large Language Model Reasoning](https://arxiv.org/abs/2610.02695) | 提出了一种无需标签的测试时校准学习框架TTCL，能够直接在未标注的目标任务数据上联合优化大语言模型的推理准确性和置信度表达能力，摆脱了对真实标签的依赖。 |
| [^185] | [Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning](https://arxiv.org/abs/2610.02687) | 该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。 |
| [^186] | [RAOA: Alternating-Operator Neural Computation with Programmable Radio Propagation](https://arxiv.org/abs/2610.02683) | 该论文提出RAOA循环计算架构，将可编程无线电传播用作计算深度而非仅仅作为通信信道，通过在持久潜在状态上交替执行能量导出的问题更新与混合更新，使重复执行能够在不增加学习控制参数的情况下提升离散优化的解质量，并证明所需算子可由无源仅相位自由空间传播来近似。 |
| [^187] | [LEAP: Learning Efficient Action Proposals For LLM Agents](https://arxiv.org/abs/2610.02670) | 该论文提出LEAP方法，通过学习一个高效的动作提议模型（而非使用现成的通用模型）为LLM智能体起草动作，并建立延迟分析框架揭示决定动作投机端到端加速的关键因素，从而显著提升智能体执行任务的速度。 |
| [^188] | [Large Language Continuous Diffusion Models](https://arxiv.org/abs/2610.02665) | 提出了首个大规模（3B/8B）连续扩散语言模型 Sigma，通过可操控的低维潜在轨迹、自回归模型热启动以及无分类器引导等推理技术，在数学推理和编码任务上取得了与离散扩散模型相当的性能。 |
| [^189] | [Generalization Properties of Score-matching Diffusion Models for Intrinsically Low-dimensional Data](https://arxiv.org/abs/2610.02663) | 该论文为流匹配模型在具有内在低维结构的数据上提供了统计泛化理论保证，推导出依赖于数据内在维度的 Wasserstein-p 有限样本误差界，克服了以往分析中限制性假设和忽略低维结构的不足。 |
| [^190] | [Mind the Refinement Gap: When Safe High-Level Robot Plans Produce Unsafe Executions](https://arxiv.org/abs/2610.02662) | 该论文揭示了机器人安全系统中的“细化差距”问题——通过安全监控的高层规划在执行时可能因导航和隐式动作效果而变得不安全，并通过28个受控案例和14个端到端案例验证了基于图的轨迹细化方法的必要性。 |
| [^191] | [AIGS: Adaptive Incremental Gating System for Online Representation Learning in Non-Stationary Data Streams](https://arxiv.org/abs/2610.02661) | 本文提出轻量级闭环自适应框架AIGS，通过内生残差反馈机制“冲击比率”驱动连续可塑性控制器，在非平稳数据流的在线表示学习中动态平衡知识保持与概念漂移响应。 |
| [^192] | [Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis](https://arxiv.org/abs/2610.02659) | 该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。 |
| [^193] | [Context-Tower Conversion Preserves Generation While Freezing Retains Knowledge: Low-Budget AR-to-Diffusion Conversion of MoE LLMs](https://arxiv.org/abs/2610.02657) | 在低训练预算下将MoE大语言模型转换为扩散语言模型时，冻结父模型副本并通过交叉注意力条件化的“冻结塔”方案，比直接更新部分权重的“就地”方案生成能力提升11.6倍，同时几乎完全保留原模型知识。 |
| [^194] | [When Normalization Selects the Sign: Auditing Robustness Ablations in Quantum Attention](https://arxiv.org/abs/2610.02641) | 该论文以四量子比特量子注意力检测器为例，揭示消融实验中输入缩放模块表现出的鲁棒性收益可能来自比较规则本身而非模块，主张需控制编码器处的扰动上界，并区分描述性结论与因果性结论。 |
| [^195] | [Where Quantum Fourier Sampling Stops Short: A Three-Gate Audit Protocol for Delay-PUF Security Models](https://arxiv.org/abs/2610.02636) | 本文通过结构、算法和量子核诊断三重门的系统审计表明，量子傅里叶采样在延迟PUF安全模型中难以建立真正的量子优势，因为低次结构不等于稀疏支撑、相位预言机构造逻辑上蕴含经典基线，而看似有利的量子核指标与经典Gram矩阵性质高度相关。 |
| [^196] | [A hybrid CNN-adjoint optimization framework for reconstruction of viscoelastic tissue properties in magnetic resonance elastography](https://arxiv.org/abs/2610.02634) | 该论文提出了一种结合卷积神经网络与伴随优化的混合框架，用于磁共振弹性成像中软组织复值剪切模量的重建，并在理论上证明了正问题的适定性、极小值的存在性及一阶最优性条件。 |
| [^197] | [Online Verification of Language Model Responses Under Cost Constraints](https://arxiv.org/abs/2610.02632) | 提出OMVV在线多验证器算法，通过维护K个候选弱验证器池来适应查询主题和难度的动态变化，在成本约束下实现更准确且经济高效的语言模型输出验证。 |
| [^198] | [VERSE: Verified Self-Evolving Optimizer for Agent Harnesses](https://arxiv.org/abs/2610.02616) | 提出VERSE，一个经过验证的自我进化优化器，它不仅改进智能体框架，还让优化器自我进化其诊断、编辑与验证流程（如测试草稿编辑、重放故障、扰动可疑步骤），在具备基于执行的验证时取得最佳优化效果。 |
| [^199] | [Quantifying the Value of Constructive Induction, Knowledge, and Noise Filtering on Inductive Learning](https://arxiv.org/abs/2610.02615) | 本文提出“有效维度”这一新型学习度量方法，它可经验估计并能进行平均情况预测，可用于精确量化构造性归纳、噪声过滤和背景知识对归纳学习性能的价值。 |
| [^200] | [What Is Lost in Post-Training? Default Collapse and the Loss of In-Context Steerability Across Diverse Perspectives](https://arxiv.org/abs/2610.02614) | 论文发现后训练不仅会收窄模型的默认观点表达，还会系统性削弱模型通过上下文提示被引导至未被训练偏好视角的能力，揭示了单一价值观优化与服务多元利益相关者能力之间的根本矛盾。 |
| [^201] | [Scale-Recursive Rectified Flows for Few-Step Precipitation Ensembles](https://arxiv.org/abs/2610.02611) | 提出一种尺度递归校正流方法，先生成大尺度降雨模式再生成局部细节，并通过比较集合变率与预测误差自适应分配采样步数，从而在极少采样步数下生成高质量且不确定性可靠的降水集合。 |
| [^202] | [Seer: Maximum Likelihood Regression for Learning-Speed Curves](https://arxiv.org/abs/2610.02610) | Seer系统通过生成分类学习性能的经验观察数据并构建最大似然统计模型，能够预测达到目标性能所需的训练样本数量以及理论上限准确率，并在多个真实领域数据上验证了其可行应用。 |
| [^203] | [Activation Sparsity with Weight Approximation for Faster LLM Decoding on Offloaded Weights](https://arxiv.org/abs/2610.02598) | 提出SpAx方法，将权重读取的“保留或跳过”二元决策扩展为“完全保留、压缩近似或完全省略”三种选项，从而在利用激活稀疏性加速卸载权重下的LLM解码时，显著改善模型质量与推理速度之间的权衡。 |
| [^204] | [How Causality Bridges the Semantic Gap](https://arxiv.org/abs/2610.02594) | 该论文提出以因果结构替代人类知识来为未命名变量赋予语义，将其形式化为“结构约束的语义对齐”，并构建 CausalBridge 框架，从测量数据（含隐变量）中发现因果图并在其依赖关系约束下求解变量嵌入，从而从变量对其他变量的作用方式中解读其含义。 |
| [^205] | [Fisher-Guided Submodular Data Selection for Continual Pre-Training of Large Language Models](https://arxiv.org/abs/2610.02593) | 该论文通过Fisher信息分析揭示了灾难性遗忘的参数空间机制，并提出一种Fisher引导的次模数据选择方法，在持续预训练中保护预训练模型所依赖的高Fisher参数坐标，从而无需大量回放数据即可控制遗忘。 |
| [^206] | [Dense Mixture-of-Experts as a Reparameterized Wide FFN: A Granularity Sweep at Fixed Compute](https://arxiv.org/abs/2610.02584) | 该研究在固定计算量下提出一种稠密混合专家架构（$K$个SwiGLU专家全部激活并由softmax门控组合），发现其验证损失随专家数量$K$呈非单调变化（$K=2$略优而$K=4$、$K=6$反而恶化），并证明该架构在数学上等价于带token依赖、单纯形约束组缩放的稠密SwiGLU前馈网络。 |
| [^207] | [High-Dimensional Asymptotics and Dataset Selection for Private Transfer Learning](https://arxiv.org/abs/2610.02578) | 本文针对隐私保护迁移学习中的数据集选择问题，提出了一种仅基于汇总统计量、采用加权岭估计器的多元异构源高维回归方法，在 ρ-零集中差分隐私保证下研究其高维渐近性质，以判断额外数据是否值得购买或纳入协同学习。 |
| [^208] | [DISSOLVR: An Interpretable and Fast Framework for Aqueous and Organic Solubility Prediction](https://arxiv.org/abs/2610.02574) | DISSOLVR是一个透明、可解释且快速的分子溶解度预测框架，它通过将分子映射到基于物理的描述符，实现了接近实验不确定性极限的预测精度和分布外泛化能力，证明最先进的性能无需依赖不透明的深度学习架构。 |
| [^209] | [DAGS: Disentangled Appearance-and-Geometry Steering of a Frozen Image DiT for Temporally Stabilized Generative Rendering](https://arxiv.org/abs/2610.02567) | DAGS提出了一种轻量级、无需注意力机制的外观与几何解耦条件注入方案，将条件特征作为逐层残差注入冻结的图像DiT，并辅以循环光照稳定器和免训练时序引导，实现了高保真、高忠实度且时间稳定的生成式渲染。 |
| [^210] | [Learning Closure of Dynamical Systems with Kernel Ridge Regression](https://arxiv.org/abs/2610.02564) | 本文提出基于核岭回归的闭合建模框架，用于学习ODE/PDE设置中的差分方程闭合和动理学方程矩闭合中的代数闭合，在Lorenz-63系统和Kuramoto-Sivashinsky方程上实现了准确的长期预测，并显著优于基于LSTM的闭合模型。 |
| [^211] | [OpenGameEval: Benchmarking Agentic Programming and Exploration in a Stateful Game Engine](https://arxiv.org/abs/2610.02563) | OpenGameEval是一个在Roblox Studio有状态游戏引擎中评估智能体游戏开发能力的基准框架，其核心创新在于通过分离观察工具与编辑工具来直接测量探索行为，实验发现前沿模型虽通过率相近但解决的任务各不相同，且最佳模型单次尝试仅能解决51.7%的任务。 |
| [^212] | [Neuron merging via inverse-activation regression for post-training compression of sigmoid neural networks](https://arxiv.org/abs/2610.02559) | 提出了一种通过逆激活函数将神经元响应映射回预激活空间并利用最小二乘法估计代表神经元参数的神经元合并方法，实现Sigmoid神经网络的训练后压缩并更好地保留有用信息。 |
| [^213] | [How to Have a Sensitive Debate: An Instance-Optimal Protocol for AI Debate](https://arxiv.org/abs/2610.02557) | 本文针对AI辩论设计了一种新的实例最优协议，对于具有足够稳定子问题分解的问题，在有限监督下比现有最佳协议提供更强的正确性保证。 |
| [^214] | [Test-time Multi-agent Coordination by Decomposed Value Gradient Flow](https://arxiv.org/abs/2610.02554) | 提出SCOUT框架，首次通过测试时动作精炼将流匹配行为先验与分解价值函数结合，利用Stein变分梯度下降将行为样本传输至高价值区域，从而解决离线多智能体强化学习中生成式策略与价值优化策略之间的权衡问题。 |
| [^215] | [Reward Inflation: A Healthy Stimulus for Reinforcement Learning](https://arxiv.org/abs/2610.02545) | 提出奖励通胀方法，通过在训练中逐渐放大奖励实现隐式近期加权，加速策略适应、抑制休眠神经元并保持网络可塑性，在ALE游戏和MuJoCo任务上显著提升强化学习性能。 |
| [^216] | [ENCORE: Exact Non-equilibrium COntrol with Replica Exchange for Diffusion Generation](https://arxiv.org/abs/2610.02538) | 该论文提出了首个精确的并行推理时控制方法ENCORE，通过让每个副本保存生成轨迹使向上移动成为截断操作，从而避免模拟难以处理的时间反转，实现了无偏的副本交换控制。 |
| [^217] | [Autoregressive Differentiable Method for Integer Programming](https://arxiv.org/abs/2610.02528) | 该论文提出了一种自回归可微方法，通过训练Transformer并结合拉格朗日惩罚与Gumbel-softmax激活来求解0-1整数规划，在多达10,000个变量的稠密二次背包问题上持续超越最先进的开源求解器。 |
| [^218] | [CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization](https://arxiv.org/abs/2610.02527) | 该论文揭示，用学习型视觉奖励模型优化机器人策略时，奖励分数与任务成功率可能同时上升、看似健康，但实际上会显著放大“作用于错误物体”的隐蔽失败，而这一现象在使用模拟器真实任务完成信号训练时并不会出现。 |
| [^219] | [BaCP: Backbone Contrastive Pruning for Preserving Representations in Extremely Sparse Neural Networks](https://arxiv.org/abs/2610.02524) | 提出主干对比剪枝（BaCP），通过将稀疏网络的嵌入空间与预训练、微调及历史快照模型对齐来正则化，解决极稀疏剪枝中的表征坍塌问题，在90种设置下显著提升极稀疏时的准确率。 |
| [^220] | [Instance-Dependent Regret for CMDPs with Step-Wise Constraints](https://arxiv.org/abs/2610.02520) | 本文提出了安全方差自适应探索算法（SVAE），通过学习候选安全子图并在其中进行方差自适应的乐观规划，在具有逐步安全约束的情景式CMDP中首次实现了依赖问题实例（方差感知）的累积遗憾界。 |
| [^221] | [Learning the Latent Structure: A Feature-Centric Approach to Graph Data Augmentation](https://arxiv.org/abs/2610.02517) | 该论文提出了一种以特征为中心的图数据增强框架，通过在嵌入空间中进行自监督逆掩码过程来捕获观测图与完整图之间的潜在结构联系，从而恢复未观测的结构信号，克服了现有方法依赖标签、计算昂贵和转导式的局限性。 |
| [^222] | [Student-Guided Teacher Distillation for Efficient LLM Task Routing: Positioning Against Jev-Style System-1 Classifiers](https://arxiv.org/abs/2610.02516) | 提出一种学生引导的教师蒸馏流水线：紧凑的ModernBERT学生模型单次前向预测完整类别分布并生成top-k候选，更大的DeBERTa-v3零样本NLI教师模型仅对候选重排序，教师标签迭代反哺学生，从而显著降低大规模LLM任务路由的成本。 |
| [^223] | [IGNITE Tokamak World Model Architecture](https://arxiv.org/abs/2610.02515) | IGNITE是首个基于DIII-D十年实验数据自监督训练的聚变等离子体生成式世界基础模型，可通过执行器轨迹、文本提示或期望实验结果模拟完整的托卡马克放电过程。 |
| [^224] | [Post-Training Quantization of Autoregressive Weather Models](https://arxiv.org/abs/2610.02511) | 该论文将后训练量化（PTQ）技术应用于自回归深度学习天气预报模型，以加速推理计算、降低功耗，并使其能够部署在边缘硬件上。 |
| [^225] | [Multi-Fidelity Policy Gradients Stabilize Data-Scarce Reinforcement Learning](https://arxiv.org/abs/2610.02505) | 本文将多保真度策略梯度（MFPG）框架从REINFORCE扩展到现代演员-评论家算法（如PPO），在GPU并行仿真和真实机器人上利用低保真度数据构建控制变量，以在无偏差的前提下降低梯度方差，从而稳定数据稀缺场景下的强化学习。 |
| [^226] | [Compound AI System Reliability: A Failure Taxonomy and Resilience Pattern Catalog from 150 Production Incidents](https://arxiv.org/abs/2610.02503) | 本文通过分析150起生产事故，构建了包含23种故障模式、分五大类别的复合AI系统故障分类学，并提出经故障注入实验验证有效性的韧性模式（如断路器减少89%级联传播、质量门控捕获73%静默退化）。 |
| [^227] | [LiteEMG-FM: An Efficient and Deployable Foundation Model for Robust EMG Sensing](https://arxiv.org/abs/2610.02497) | 本文提出LiteEMG-FM，一种在16个多样化EMG数据集上预训练的高效CNN-Transformer混合基础模型，通过分层唤醒架构（轻量级1D-CNN预筛）实现跨用户泛化且适合资源受限可穿戴设备的实时部署。 |
| [^228] | [Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement](https://arxiv.org/abs/2610.02492) | 该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。 |
| [^229] | [Harnessing LLMs as Agents: What Does It Cost?](https://arxiv.org/abs/2610.02488) | 该论文提出语言模型智能体机（LAM）这一资源受限的计算抽象，首次从通信、内存访问、重计算与验证等方面严格量化 LLM 智能体驱动框架所消耗的计算资源，并给出相应的计算量下界。 |
| [^230] | [Threshold-Aware Conformal Routing](https://arxiv.org/abs/2610.02487) | 该论文提出一种阈值感知的保形路由方法，依据代理模型保形区间与决策阈值的相对位置来决定是使用快速代理模型还是昂贵的高保真仿真，从而在保证下游决策可靠性的同时大幅降低仿真开销。 |
| [^231] | [SD-DPC: Sparse Dictionary Differentiable Predictive Control](https://arxiv.org/abs/2610.02466) | 提出稀疏字典可微预测控制（SD-DPC）框架，将基于SINDy辨识的预测模型与可微预测控制相结合，从数据中学习出仅含少量项、可解释的显式反馈控制律，在满足约束的同时性能比蒸馏策略提升高达一个数量级，且内存和在线计算开销降低几个数量级。 |
| [^232] | [Capability Scaling-Down Laws for LLM Compression](https://arxiv.org/abs/2610.02462) | 该论文系统研究了大语言模型在剪枝、量化和蒸馏压缩下的能力缩减定律，建立了可预测不同压缩配置所导致能力损失的简单关系式，从而显著减少压缩实验所需的测量成本。 |
| [^233] | [A Composable AI-Accelerated Iterative Solver for 3D-IC Thermal Modeling](https://arxiv.org/abs/2610.02461) | DAIST将3D-IC封装热仿真分解为块级子域并用神经算子替代子域求解器，通过界面温度与热通量的迭代交换实现耦合，从而构建出无需重新训练即可复用于不同拓扑封装的可组合AI加速热求解器。 |
| [^234] | [AI-driven Thermal-aware Data Center Capacity Planning](https://arxiv.org/abs/2610.02442) | 本论文提出一个AI驱动的数据中心热感知容量规划框架，其内嵌模型通过学习机架功率、服务器放置和HVAC设置等关键参数，可在毫秒级完成温度预测，从而在几秒内实现对真实数据中心的热感知容量规划。 |
| [^235] | [Bandits via Additive Quantized Representations](https://arxiv.org/abs/2610.02440) | 该论文提出将残差量化（RQ）作为表示层，把连续上下文映射为多层级离散码本分配，从而实现内存严格受限的非线性上下文赌博机算法，在13个数据集中的11个上优于非RQ基线，并以最多1000倍更少的内存匹敌需要定期重训的XGBoost和神经网络基线。 |
| [^236] | [A Generative Model of Complex Networks Using Graphons and Neural Inverse Operators](https://arxiv.org/abs/2610.02439) | 该论文提出基于多分形阶梯图函数的复杂网络生成模型，通过神经逆算子在函数空间中恢复参数，实现了兼具可解释性与摊销式推理、并能泛化到训练中未见图规模的新型生成框架。 |
| [^237] | [Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval](https://arxiv.org/abs/2610.02438) | 该论文提出将大语言模型对知名算法的代码生成重新定义为“参数化代码检索”任务，并引入AlgoREval基准（涵盖599个问题、77个经典算法、7种编程语言和4种图输入表示）来独立评估这一能力，发现不同语言和输入表示之间的检索准确率差异显著。 |
| [^238] | [Learning Style, Forgetting Semantics: A Case Study of SFT and RFT on Classification Tasks](https://arxiv.org/abs/2610.02437) | 本文通过将策略更新精确分解为语义与风格两个成分，揭示了SFT比RFT遗忘更多的原因——SFT会沿教师风格偏好产生离轴风格漂移从而破坏语义记忆，而RFT能保持类内风格对称性。 |
| [^239] | [Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations](https://arxiv.org/abs/2610.02432) | 本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。 |
| [^240] | [CRISP: A Framework for Clause-Reconstructed Interpretable NeuroSymbolic Propositions](https://arxiv.org/abs/2610.02431) | CRISP框架通过Tsetlin机器子句重构二元神经网络教师模型的最后一层激活向量，建立了从输入特征到隐藏神经元的显式符号化决策追溯，显著提升了深度神经网络的可解释性与可审计性。 |
| [^241] | [Geometry-Aware Time Reparameterization for Flow-Map Distillation](https://arxiv.org/abs/2610.02427) | 提出一种几何感知的时间重参数化方法，为学生模型在法向加速度大的轨迹区域分配更多蒸馏时间，在保持教师几何路径和终端分布的同时使流映射蒸馏更易学习。 |
| [^242] | [The AI Theorist reveals excitonic structure in $\alpha$-RuCl$_3$](https://arxiv.org/abs/2610.02417) | AI Theorist系统通过假设生成、第一性原理计算和证据驱动的迭代完善实现了物理模型的自主发现，并首次在Kitaev量子自旋液体候选材料α-RuCl₃中识别出具有不同光学选择定则和实空间分布的激子态。 |
| [^243] | [Prompted to Discriminate: Generalizing Malicious-Input Probes in the Wild](https://arxiv.org/abs/2610.02413) | 本研究系统评估了在用户输入后附加分类指令后缀能否增强激活探针对恶意输入的检测能力，并利用严格的留一数据集外（LODO）评估，在 13 个安全基准和三个开源模型家族上检验了此类提示后缀在未见攻击类型上的泛化表现。 |
| [^244] | [Efficient Neural Field Learning via Adaptive Coverage and Focused Sampling](https://arxiv.org/abs/2610.02410) | 提出ACES采样框架，通过解耦覆盖与重要性——利用自适应空间分区保证域覆盖、采用区域级重要性加权聚焦关键区域——从而降低梯度方差并显著提升隐式神经表示的训练效率。 |
| [^245] | [Trained Agentic Context Management](https://arxiv.org/abs/2610.02404) | 通过在最简化的智能体框架（自我调用工具与上下文读取工具）上微调小模型，模型仅用8K词元上下文即可在长文档基准上媲美拥有1M词元上下文的GPT-5.4。 |
| [^246] | [Energy Saving in 5G and Beyond Networks: A Quantum Reinforcement Learning Approach](https://arxiv.org/abs/2610.02403) | 本文针对5G及未来网络中的基站能耗优化问题，提出量子强化学习方法，以克服传统深度强化学习训练负担重、状态与动作空间指数级增长的难题。 |
| [^247] | [VisAudit: Evaluating Multimodal Agents for Visual Diagnosis and Repair](https://arxiv.org/abs/2610.02399) | 本文提出VisAudit基准，用于评估多模态智能体在数据可视化任务中的自主诊断、修复与验证能力，弥补了现有基准仅能评估单一预定义能力的不足。 |
| [^248] | [Validated Data Onboarding for AI Demand Forecasting on U.S. Building Meter Data: Design, Controlled Evaluation, and a Corrected Negative Result](https://arxiv.org/abs/2610.02397) | 本文提出一种在模型训练前检测并修复建筑电表数据缺陷的数据导入管道（仅使用预测时可得信息），受控实验表明仅0.10%的训练数据缺陷即可使预测误差增加86%，而该管道的修复可将误差恢复至干净数据水平。 |
| [^249] | [Inherit-MAS: Test-Time Evolution of Multi-Agent Systems through Workflow and Execution Inheritance](https://arxiv.org/abs/2610.02396) | 该论文提出Inherit-MAS框架，借鉴生物进化中遗传与选择的机制，在工作流和执行两个层面实现显式继承，使基于大语言模型的多智能体系统能够在测试时高效演化工作流，同时避免破坏有用组件和产生冗余计算。 |
| [^250] | [Hesitation Has a Geometry: Entropy-Trained Hyperbolic Probes for Sparse Activation Steering](https://arxiv.org/abs/2610.02391) | 该论文提出双曲熵引导方法（HEST），以模型自身的下一个词元熵作为唯一标签训练双曲空间中的轻量探针，仅在模型“犹豫”的高熵词元处沿测地线对隐藏状态进行稀疏引导，从而更契合推理过程固有的树状层级结构。 |
| [^251] | [The Surprising Effectiveness of Shared Memory in Looped Transformers](https://arxiv.org/abs/2610.02383) | 提出让循环Transformer在预训练时共享内存（仅第一次递归写入键值缓存、后续递归读取并保留自身短窗口）的方法，不仅不损失质量反而提升质量，在减少76-79%上下文内存的同时刷新了循环模型的质量-内存边界。 |
| [^252] | [Latent-MOPD: Latent Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2610.02381) | 提出Latent-MOPD，首个面向大语言模型的表示层级多教师在线策略蒸馏方法，无需额外教师训练即可同时利用专家模型的预测和隐藏状态进行知识传递，在全部九个基准测试上超越了仅token、仅表示和均匀平均等基线方法。 |
| [^253] | [Flow Matching for Fast Posterior Sampling in Bayesian Inverse Problems](https://arxiv.org/abs/2610.02377) | 该论文针对具有函数值参数的PDE贝叶斯逆问题，系统评估了条件流匹配作为MCMC摊销式替代方法的表现，推导出近似后验在总变差距离和KL散度下的可计算精度估计，并提出一种通过Metropolization实现渐近精确的混合采样器。 |
| [^254] | [Co-design Gym: A Unified Benchmark for Embodiment-Policy Co-optimization](https://arxiv.org/abs/2610.02366) | 本文提出了Co-Design Gym，一套用于形态与策略联合协同优化的统一基准测试环境，突破了现有基准中形态固定、仅关注策略学习的局限。 |
| [^255] | [ArrivalBench: Agent-Generated Data Pipelines Are Correct Once and Wrong Under Time](https://arxiv.org/abs/2610.02363) | ArrivalBench 提出在对抗性但可重放的数据送达调度下重新执行代理生成的数据管道，并以完整日志的批量重算作为判定标准，发现经单次执行评分认证的管道中有 7.0-79.2% 实际上默默出错，且这一差距与修复循环无关。 |
| [^256] | [Lexicographic Multi-Objective On-Policy Distillation](https://arxiv.org/abs/2610.02359) | 提出了字典序多目标在线策略蒸馏（LMOPD），一种多教师蒸馏方法，在显式优先级保护下整合奖励专门化策略，确保低优先级目标（如简洁性）不会以牺牲高优先级目标（如正确性）为代价而提升。 |
| [^257] | [Conformal Prediction for Time Series with Deep Sequence Models](https://arxiv.org/abs/2610.02357) | 本文首次系统性地研究了深度序列模型在时间序列共形预测中的应用，通过条件分位数回归等三种方法，解决了传统共形预测所依赖的数据可交换性假设在时间序列中不成立的问题。 |
| [^258] | [Why Does Adaptive Batching Help LLM Pretraining? A Perspective from Unbounded Variance](https://arxiv.org/abs/2610.02355) | 该论文通过对LLM预训练中方差增长的实证分析，提出带可调增长指数的广义BG-a噪声模型，从无界方差控制的视角解释了自适应批大小调度为何能提升预训练效果。 |
| [^259] | [Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation](https://arxiv.org/abs/2610.02353) | 提出 LINEUP 方法，通过实证发现各用户独立适配器中存在大量可跨用户共享的结构，进而学习一个共享的低秩个性化因子库并仅保留紧凑的用户专属校正，从而将个性化容量在共享与专属之间解耦，摆脱逐用户完整适配的可扩展性瓶颈。 |
| [^260] | [DeReAct: Decomposed Reasoning and Acting for Reliable AI Agents](https://arxiv.org/abs/2610.02351) | DeReAct提出了一种模块化智能体架构，通过将动作验证（Critic）与任务完成认证（Context Manager）从单一LLM策略中外置为独立门控机制，防止错误传播和无效完成声明，在GAIA和SWE-bench Verified上对较弱模型带来了最显著的Pass@1提升。 |
| [^261] | [From Behavior to Provenance: Attributing Tabular Foundation Models to Synthetic Pretraining Data](https://arxiv.org/abs/2610.02347) | 该论文提出利用带有完整溯源信息的合成任务生成器O'PRIOR构建可控测试平台，通过反事实重训练将表格基础模型的训练数据归因从难以验证的推测转变为可实验检验的问题。 |
| [^262] | [Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors via Kernel-Anchored Locality Regularization](https://arxiv.org/abs/2610.02345) | 论文揭示了固定目标神经异常检测器中存在“收敛坍塌”这一结构性问题——训练越充分检测性能反而越差，并提出核锚定局部性正则化方法，通过引入局部性约束来阻止无约束外推，从而在模型完全收敛时仍能保留有效的异常检测信号。 |
| [^263] | [Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2610.02344) | 本文提出了 JEPA 的早期训练稳定性理论，用驱动力 γ 与衰减效应 σ 之比构成的稳定性比率 μ 刻画表示坍缩的相变边界，并将预测器缩放、掩码比率与 EMA 统一为调节 μ 的机制，据此提出了 ResidualPred 预测器。 |
| [^264] | [NEEDLEWORK: Offline Rewriting of Robot Data with Verified Local Stitches](https://arxiv.org/abs/2610.02339) | NEEDLE是一种离线数据集增强算法，通过仅使用RGB图像和本体感觉信息，在高维机器人演示的观测之间添加经验证的短动作桥梁，从而离线重写训练数据、绕过次优轨迹并利用失败回合，无需新的环境交互。 |
| [^265] | [SoTa: Soft Tactile Skins for Dexterous Manipulation](https://arxiv.org/abs/2610.02338) | SoTa是一种低成本的电容式柔软触觉皮肤，可在人手和机器人手上实现全手覆盖并保持202个触觉单元的统一布局，每片材料成本不足10美元，为通过大规模人类示教数据提升机器人灵巧操作能力提供了可行途径。 |
| [^266] | [Joint Movement and Compression Ratio Design for Mobile Embodied AI Networks (MEAN)](https://arxiv.org/abs/2610.02334) | 该论文针对上行链路移动具身智能网络（MEAN）系统，首次联合优化移动距离、语义压缩比和发射功率以最大化最小能效，并提出了AO-Dinkelbach交替优化算法来求解该非凸问题。 |
| [^267] | [Slow-Fast Multi-Teacher On-Policy Distillation for Capability Preservation](https://arxiv.org/abs/2610.02324) | 提出 SF-MOPD 方法，通过将教师直接更新的快速学生模型与作为动态能力参考的指数移动平均慢速模型相耦合，在多教师在线蒸馏中实现领域专长获取与通用能力保持的平衡。 |
| [^268] | [DeskForge: Dense Supervision from Desktop Environments for Computer-Use Agents](https://arxiv.org/abs/2610.02320) | 本文提出可控桌面环境DeskForge，通过组合和变换真实应用程序生成大规模密集标注语料库DeskForge-1M（含120万条桌面观测与1.597亿个元素实例），有效提升了视觉语言模型在复杂桌面场景中的动作目标定位能力。 |
| [^269] | [SimuVerity: Benchmarking Agents for Engineering-Grade Simulink Model Generation](https://arxiv.org/abs/2610.02304) | 提出SimuVerity基准，包含101个跨十个工程领域的Simulink模型生成任务，采用分层评估器从六个工程维度对模型评分，发现最佳智能体系统总分仅为42.86，证明结构相似度并不能衡量模型的工程性能。 |
| [^270] | [PowerBench: Measuring Language Model Bias in Power-shifting Requests](https://arxiv.org/abs/2610.02303) | 该论文提出了PowerBench这一开源基准，通过区分自我赋权、去权和权力攫取三类权力转移请求，系统评估了24个中美语言模型在权力相关请求上的拒绝行为，揭示了模型拒绝倾向的系统性偏差。 |
| [^271] | [Intent-Hiding Jailbreaks: An Information-Theoretic Framework for Compositional Attacks](https://arxiv.org/abs/2610.02302) | 该论文提出了一种信息论框架，通过选择辅助任务实现有害意图先验概率与后验概率的匹配，从而在大语言模型的组合式查询中隐藏有害意图，构成一种新型越狱攻击方法。 |
| [^272] | [$\Psi$-Resilience: Model-Free Feature Importance from 1D Topological Signals](https://arxiv.org/abs/2610.02299) | 提出了一种基于一维拓扑信号的无模型特征重要性方法 $\Psi$-Resilience，它通过类条件密度差异构建不一致性景观并利用其持续性定义韧性评分，从而产生上下文鲁棒且可审计的特征排序。 |
| [^273] | [Diffusion-Based Synthetic Data Pretraining for Enhancing Activity Recognition](https://arxiv.org/abs/2610.02292) | 本研究提出利用扩散模型生成合成传感器数据进行预训练、再在真实数据上微调的两阶段训练策略，以增强CABiGRU模型对进食、饮水等细微少数类别人体活动的识别能力。 |
| [^274] | [Expected Utility Regret Rule: Minimax and Bayes Optimal Portfolio Choice](https://arxiv.org/abs/2610.02290) | 提出期望效用遗憾（EUR）规则，该规则无需先验分布即可同时达到极小极大与贝叶斯最优下界，并将均值-方差组合和风险平价组合统一为该框架的特例。 |
| [^275] | [Effects of interpulse-interval variation on deep-learning classification of bat vocalizations](https://arxiv.org/abs/2610.02284) | 该研究通过构建自然脉冲间隔与归一化脉冲间隔的匹配数据集进行对比实验，揭示了脉冲间隔变化蕴含蝙蝠物种判别信息，且基于Transformer的模型比卷积神经网络对这一时序特征更为敏感。 |
| [^276] | [MuLoRA: Spectrally Balanced Low-Rank Adaptation for Continual Learning](https://arxiv.org/abs/2610.02283) | 该论文发现持续学习中LoRA存在“频谱可塑性坍缩”问题——更新能量集中于少数奇异模态导致低秩空间利用不足，并提出MuLoRA通过历史白化与动量近似极正交化联合控制容量分配与利用，实现频谱平衡的持续学习适配。 |
| [^277] | [CLEAN: Psychometrically Consistent Incremental Cognitive Diagnosis under Concept-Space Expansion via Architectural Isolation](https://arxiv.org/abs/2610.02278) | 提出CLEAN增量认知诊断框架，通过架构隔离支持概念空间的动态扩展，并从结构上保证增量更新后的心理测量一致性，避免灾难性遗忘。 |
| [^278] | [Confidence-Gated Cloud-Edge Cascade Triage via Variational Risk Minimization for Medical Imaging](https://arxiv.org/abs/2610.02269) | 提出变分风险最小化（VRM）蒸馏框架，将LVLM生成的报告变体作为蒙特卡洛样本从边缘化教师分布中学习，以应对报告缺失的模态鸿沟，并结合置信度门控云-边级联，在103ms延迟下实现AUC 0.941的急诊胸片分诊性能。 |
| [^279] | [From Mathematical to Executable Certificates for Machine Unlearning](https://arxiv.org/abs/2610.02268) | 该论文提出ExecCert，一个发布时的可执行认证层，通过补全方法原生证书或应用重训练参考发布验证（RRV）来认证待发布的机器遗忘制品，从而弥合数学保证与实际部署的有限精度软件制品之间的差距。 |
| [^280] | [Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses](https://arxiv.org/abs/2610.02267) | 该论文通过严格配对与自审计的评估发现，托管型System-1决策模型Jev在11个代理决策点中的9个上显著优于开源模型Laya，但两者在零样本模型路由上均未超过随机水平，且开源模型对选项顺序和候选数量高度敏感。 |
| [^281] | [Parameter-Free Interval-Dynamic Regret under Heavy-Tailed Noise](https://arxiv.org/abs/2610.02258) | 本文提出了一种无需任何参数知识的在线凸优化算法，在重尾噪声（未知有限条件p阶矩，1<p≤2）下实现了区间动态遗憾界，达到了min(GDn, GD√(nΛ) + σDn^(1/p)Λ^((p-1)/p))的通用常数遗憾保证。 |
| [^282] | [TRACE: A Reproducible Benchmark for Electricity Price Forecasting with Official Operational Text](https://arxiv.org/abs/2610.02256) | TRACE是一个将电力价格与预测截止时点官方运行文本配对、并严格防止信息泄露的可复现电力价格预测基准，它证明了文本上下文的预测价值——使时间序列基础模型的上尾pinball损失中位数降低7.4%。 |
| [^283] | [MACTS-EM: Multi-Agent Collaborative Time Series Forecasting with Emergent Memory](https://arxiv.org/abs/2610.02255) | 提出了MACTS-EM框架，通过领域专业化智能体协作、元认知动态分配层和涌现记忆机制，实现跨领域模式迁移与多模态集成的时间序列预测。 |
| [^284] | [Overcoming Challenges of Interpretive Structural Modeling with Large Language Models](https://arxiv.org/abs/2610.02254) | 本工作将大语言模型作为“不完美专家”引入解释结构建模（ISM），以克服传统专家交互方法繁琐且难以扩展至数百个变量的挑战，并通过对比实验证明逐行和全图因果图发现方法效果最佳。 |
| [^285] | [Approximation Property of Dropout Neural Networks: Sobolev Rates and Confidence Bounds](https://arxiv.org/abs/2610.02253) | 本文首次定量刻画了随机 Dropout（边以概率 p 独立保留）的 ReLU 网络以高概率一致逼近 Sobolev 空间单位球所需的网络规模，给出了常数深度下 $\widetilde O(p^{-9}\varepsilon^{-\max\{d/n,2\}}\log(1/\delta))$ 的规模上界以及基于 Sobolev 容量的相匹配下界。 |
| [^286] | [Counterfactual Predictions in Scientific Emulators Without Controlled Experiments](https://arxiv.org/abs/2610.02252) | 提出 ReRoute 框架，无需受控实验或模拟器数据，仅通过将查询输入固定为参考值并沿已知机制路径重新引入其变化，结合事实数据微调，即可让科学模拟器准确回答“如果条件不同会怎样”的反事实预测问题。 |
| [^287] | [Rank-Aware Speculative Sampling for Diffusion Draft Trees](https://arxiv.org/abs/2610.02251) | 提出秩感知推测采样（RASS），无需额外目标模型评估即可对草稿候选进行秩感知排序并以优化权重采样，从而更高效地验证扩散草稿树并加速扩散生成。 |
| [^288] | [Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions](https://arxiv.org/abs/2610.02249) | 该论文系统比较了在不同推理信息假设下的多种最近邻检索变体用于从MS/MS谱图预测分子指纹，旨在建立更严格的基线以实现更严谨的基准测试并更好地衡量领域进展。 |
| [^289] | [State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting](https://arxiv.org/abs/2610.02248) | 本文提出首个专为地球科学Mamba状态空间模型设计的机器遗忘框架SSU-LSF，通过专用影响函数、时间混杂足迹定位与KL散度信任域约束的梯度上升，系统性地消除非平稳混杂事件（如未记录灌溉、大坝调度变化）对地表预测造成的持续隐性偏差。 |
| [^290] | [A Missing Latent, Not a Missing Simulator: Radius-Augmented Inference for Real JWST Retrieval](https://arxiv.org/abs/2610.02245) | 该研究发现SBI在真实JWST光谱上失效的根本原因不是模拟器物理缺失，而是缺少行星半径这一潜变量，并提出半径增广的流匹配后验模型MIRAGE来解决这一问题。 |
| [^291] | [Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts](https://arxiv.org/abs/2610.02241) | 提出了一个端到端的软硬件协同设计框架，通过连续重参数化实现稀疏性与量化的可微联合优化，将万亿规模MoE的专家权重压缩为硬件原生的低精度半结构化稀疏表示，从而在稀疏张量核心上加速执行并降低部署的内存瓶颈。 |
| [^292] | [Budgeted Cache Repair for Cross-Context KV-Cache Reuse](https://arxiv.org/abs/2610.02233) | 该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。 |
| [^293] | [Generalizable single-cell perturbation response prediction using energy-guided flow matching](https://arxiv.org/abs/2610.02232) | 提出scEGFlow框架，通过能量引导的流匹配动态建模单细胞从对照到扰动状态的连续转变，无需重新训练即可适应新的扰动条件，在成像表型和转录组基准测试中超越现有方法。 |
| [^294] | [The Price of Greenwashing: Algorithmic Verification and Market Discipline using Conformal Machine Learning](https://arxiv.org/abs/2610.02225) | 该论文将SEC财务数据与EPA设施级排放数据融合，利用梯度提升模型和Mondrian保形预测构建了具有数学保证的排放散度指标，客观量化企业“漂绿”行为，并揭示了金融市场会对此种算法检测到的排放偏差进行定价的纪律机制。 |
| [^295] | [Hybrid Machine Learning-Assisted Raman Spectroscopy with Generative Feature Augmentation for Pharmaceutical Identification](https://arxiv.org/abs/2610.02224) | 该论文提出了HyMLRaman混合拉曼光谱框架，通过EfficientNet-B3深度特征提取、生成式特征增强与经典机器学习分类器（如SVM）相结合，实现了对六种药物残留的快速准确识别。 |
| [^296] | [RINS: Residual-Image Neural Subspace Solvers for Large Sparse Linear Systems](https://arxiv.org/abs/2610.02217) | 提出Gate-RINS神经子空间求解器，通过缓存的残差探测生成多项式校正基并用轻量级逐点门控调制，在六个PDE基准任务上比GMRES和纯图神经基线更快达到固定相对残差阈值。 |
| [^297] | [MiDShip: Multimodal Dataset of Ship Cargo Hold Structures for Engineering Design](https://arxiv.org/abs/2610.02214) | 本文提出了MiDShip多模态数据集，包含12,753个船舶货舱结构设计（涵盖参数数据、3D几何、工程图纸、材料清单和结构评估），填补了船舶结构设计中缺乏连接几何、性能与规范约束的结构化数据集的空白。 |
| [^298] | [Universal Byte-Level Encoding: UTF-8/UTF-16 Routing to Reduce Cross-Script Token-Budget Disparities](https://arxiv.org/abs/2610.01984) | 提出通用字节级编码（UBE）双字母表分词器，将1-2字节UTF-8字符保留在UTF-8路径上、将3-4字节字符改经UTF-16路由，从而降低多语言脚本中非英语文字的编码底线，减少跨脚本间的令牌预算差异。 |
| [^299] | [Learning PDE Dynamics between Submanifolds Using Green's Observation Operators](https://arxiv.org/abs/2610.01697) | 本文提出格林观测算子（GObO），将固定的环境介质一次性映射为线性PDE在源与观测子流形上的格林核，使每个新源仅需一次低维积分而无需网络评估，并证明了所得有限流式状态的稳定性与近似速率。 |
| [^300] | [Continual Reinforcement Learning with Neuroevolution](https://arxiv.org/abs/2610.01583) | 该研究发现，在持续变化的强化学习任务中，神经进化方法（尤其是进化策略ES）能在适应与遗忘之间取得最稳定的平衡，其原因是ES能在回报景观中找到具有最宽邻域的解。 |
| [^301] | [Range-GRPO: Policy Optimization via Pairwise Relations among Reward Intervals](https://arxiv.org/abs/2610.01548) | 提出 Range-GRPO 半监督后训练框架，通过将 LLM-as-a-Judge 的伪奖励表示为保形校准的奖励区间，并在 GRPO 中以区间两两比较替代点奖励比较，使奖励不确定性能够影响学习信号的大小与方向。 |
| [^302] | [Open Vocabulary Word Recognition From Transcribed Bangla Texts](https://arxiv.org/abs/2610.01134) | 本研究通过结合基于MobileNetV2的SSD、基于InceptionResNetV2的Faster R-CNN及其集成模型，并引入改进的非极大值抑制方法，实现了手写孟加拉语开放词汇单词识别。 |
| [^303] | [Adapter Thickets: Splitting an RLVR Budget Beats Concentrating It](https://arxiv.org/abs/2610.00991) | 用RLVR预算训练单个LoRA适配器虽能提升单样本准确率，却会让采样错误越来越相关从而损害多数投票效果，而将RLVR预算拆分到多个适配器（“适配器丛林”）上再投票的效果优于将其集中在一个适配器上。 |
| [^304] | [Platonic Task Arithmetic](https://arxiv.org/abs/2610.00929) | 本文提出“柏拉图任务向量”概念，并引入形状与模型架构和嵌入维度无关的“通用任务描述符”矩阵，使任务算术（如任务加法与取反）首次能够跨越不同模型架构进行迁移与应用。 |
| [^305] | [TrueMuse: A Benchmark for Data Attribution in Text-to-Music Models](https://arxiv.org/abs/2610.00835) | 该论文提出了 TrueMuse，首个通过在已知纳入的归因样本上微调扩散式文生音乐模型来构建可控真值的数据归因基准，覆盖旋律结构、音色、艺术家风格和流派模式四种归因设置，解决了文生音乐数据归因方法缺乏可靠评估标准的问题。 |
| [^306] | [Learning to Price Electricity for Optimal Demand Response](https://arxiv.org/abs/2610.00755) | 本文提出一种基于神经网络的上下文电价定价算法，将定价建模为Stackelberg博弈并学习从上下文特征到可行电价的受限映射，通过模拟美国多个城市电网验证了该方法能显著提升需求响应计划的价值。 |
| [^307] | [Explainable Suicide Risk Assessment on Social Media with Multi-Task QLoRA](https://arxiv.org/abs/2610.00610) | 该论文提出一种基于QLoRA微调Qwen2.5-Instruct模型的多任务系统，同时完成社交媒体自杀风险评估中的风险等级分类、证据短语提取和多标签风险与保护因素识别三项任务，并通过多模型概率平均、交叉折共识等定制化聚合策略实现可解释的风险评估。 |
| [^308] | [The Conflict Between Logic and Memory: Learning Higher-Order Interactions in Shallow MLPs](https://arxiv.org/abs/2610.00403) | 该研究在浅层MLP中揭示了“能拟合训练数据却无法恢复生成规则”的逻辑—记忆冲突，并发现优化器选择显著影响高阶交互学习能力——Muon在三、四阶奇偶任务上大幅超越SGD和Adam，而仅冻结与干扰输入相连的第一层权重即可将AdamW准确率从44.73%提升至95.07%。 |
| [^309] | [PTNO: Training Neural Operators with Noisy Monte Carlo Estimates for Particle Transport Problems](https://arxiv.org/abs/2609.40090) | 该论文提出粒子输运神经算子PTNO，可直接从含噪、低成本的蒙特卡洛标签中学习粒子输运代理模型，并证明了无偏噪声标签的平方损失与收敛解损失共享同一极小值点，从而解决了高方差与高动态范围两大挑战，大幅降低了训练成本。 |
| [^310] | [Accelerated Algorithm for Sparse Regularized Partial Optimal Transport](https://arxiv.org/abs/2609.40075) | 本文提出一种基于惩罚项重构的加速优化框架，利用平滑强凸正则化器（如二次正则、弹性网络）实现稀疏部分最优传输的高效梯度更新。 |
| [^311] | [CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL](https://arxiv.org/abs/2609.39533) | 提出 CATCH 测试平台，通过刻意暴露环境漏洞并以独立审计生成黄金标签，实现对编程强化学习中奖励破解行为的可控复现、可靠识别与系统干预研究。 |
| [^312] | [A Width-Matched Comparison of Hybrid Quantum-Classical Self-Supervised Learning for Fingerprint Recognition](https://arxiv.org/abs/2609.39172) | 该论文通过在匹配表示宽度下将QuFeX量子特征提取模块嵌入SimCLR、MoCo v2和BYOL三种自监督框架进行指纹识别实验，系统比较量子-经典混合模型与纯经典模型，以厘清量子线路本身对学习表示的真实贡献。 |
| [^313] | [JARQ: Joint Alternating Refinement for Quantization](https://arxiv.org/abs/2609.38599) | JARQ 是一种即插即用的后训练量化精炼方法，通过交替执行所有组尺度的联合最小二乘拟合与有界 Babai 提议（同时移动组内多个编码），在保持位宽、零点和推理成本不变的前提下，显著降低大语言模型量化后的困惑度。 |
| [^314] | [From Solo to Social Learning: Characterizing Recursive Social Improvement in LLMs](https://arxiv.org/abs/2609.38516) | 该论文提出“递归社会改进”这一新概念，并发现尽管经典社会学习算法能从同伴中受益，但当每个LLM智能体各自追求自身奖励时，当前的LLM无法通过相互学习改进整个群体，其每token收益反而低于独立学习。 |
| [^315] | [It Takes Little to Rewrite Perception: Targeted Semantic Substitution in Vision-Language Models at $\epsilon \leq 4/255$](https://arxiv.org/abs/2609.38298) | 本文提出一种目标语义替换攻击，通过在VLM的合并后token空间中对齐源图像与目标图像，在 ε ≤ 4/255 的微小扰动下即可完全替换模型对图像的语义感知，推翻了此前认为VLM在该扰动范围内具有鲁棒性的观点。 |
| [^316] | [Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space](https://arxiv.org/abs/2609.37731) | 提出激活支持的参数分解（ASPD），通过联合分解激活与参数空间并将每个权重组件锚定于其读写激活特征，实现了预训练大语言模型中可扩展、可解释且可因果编辑的参数分解。 |
| [^317] | [TomoTransformer: Towards a Foundation Model for CT Reconstruction](https://arxiv.org/abs/2609.37605) | 提出TomoTransformer，一种基于Transformer的CT重建基础模型，将局部滤波投影作为token，在反投影空间中通过自注意力预测缺失视角，可处理任意数量和角度的投影且对探测器尺寸不变。 |
| [^318] | [Graph-Conditioned On-Policy Agent Distillation from Off-the-Shelf Teachers](https://arxiv.org/abs/2609.37522) | GC-OPD 通过图结构索引教师的成功与失败执行历史，为学生轨迹提供执行证据丰富的评分上下文，从而显著提升现成教师对多轮任务中语言智能体的在线策略蒸馏效果。 |
| [^319] | [SCOPE: Observation-Conditioned Full-Target Prediction for Sparse PDE Inference](https://arxiv.org/abs/2609.36527) | SCOPE通过共享解码器将全场潜在预测与物理重建相耦合，实现了从稀疏观测中确定性恢复完整PDE物理场，并从理论上证明了最优潜在预测未必带来最优场重建以及解码器改进向部分观测恢复迁移的条件。 |
| [^320] | [Why Backdooring Neural Networks is so Easy?](https://arxiv.org/abs/2609.36117) | 该论文通过对投毒高斯混合数据上训练的二次神经元进行精确的闭式理论分析，揭示了一个反直觉的结论：正是使神经网络强大的特征学习动态使其更容易遭受后门攻击——在懒惰学习机制下，成功攻击需要触发强度与投毒比例的平方根成反比（α ∝ π^(-1/2)），而特征学习则使攻击变得更为容易。 |
| [^321] | [CipherGenome: Homomorphic Inference for Genomic Mixture-of-Experts](https://arxiv.org/abs/2609.35883) | CipherGenome提出了一种同态加密推理协议，将151亿参数基因组MoE模型的专家投影（95.8%的参数）在模块-LWE加密下安全外包给不可信GPU服务器，同时把嵌入、注意力和路由器保留在可信客户端上，防止私有基因组在租用算力时被服务器以99.8%的准确率恢复。 |
| [^322] | [GenomeOcean Anywhere: Private WebGPU Inference for Genome MoEs](https://arxiv.org/abs/2609.35882) | 该研究构建了一个让150亿参数基因组混合专家模型在志愿者浏览器上私密运行的系统，通过手写WebGPU内核与实值拉格朗日编码计算，在专家分布于多个不受信任设备的情况下实现与原生推理一致的预测结果，且不向任何单一设备泄露序列信息。 |
| [^323] | [GenoTrace: Inheritable Watermarks for Genome Foundation Model Distillation](https://arxiv.org/abs/2609.35881) | 提出GenoTrace，一种密码子感知的绿名单水印扩展方法，使基因组基础模型蒸馏出的学生模型能够继承可检测的水印信号，并在token替换和核苷酸编辑等攻击下保持显著的鲁棒性。 |
| [^324] | [Learning from the Gap Between Pass@K and Pass@1](https://arxiv.org/abs/2609.35793) | 提出 GapFT 方法，通过在 Pass@K 与 Pass@1 的差距（即单样本失败但 K 个样本内可解决的问题）上进行微调，将测试时搜索带来的能力吸收进模型，从而提升单样本解码的性能。 |
| [^325] | [Unifying Distributional Training for One-Step Visual Generation](https://arxiv.org/abs/2609.35763) | 本文提出了单步视觉生成中分布训练的统一理论框架，并据此提出MGFlow方法，以可调粒度的高斯混合建模特征分布，同时支持最优传输与分数匹配，有效缓解模式坍塌问题。 |
| [^326] | [Provable Benefits of Regularization: Fast Rates for Adversarial Imitation Learning](https://arxiv.org/abs/2609.35698) | 该论文首次为对抗性模仿学习中奖励正则化与策略正则化的有限样本优势建立了理论保证，提出了一种结合KL策略正则化与专家-学习者占用度加权二次奖励惩罚的无模型算法，并证明了在K个在线回合和N条专家轨迹下正则化模仿差距达到Õ(1/K + 1/N)的快速收敛速率。 |
| [^327] | [MASCIT: A Mask-Aware State Space Classifier for Naturally Irregular Time Series](https://arxiv.org/abs/2609.34409) | 提出掩码感知状态空间分类器MASCIT，通过观测掩码与门控时间聚合有效处理异步观测、缺失值等自然不规则性，在34个不规则时间序列数据集上取得最优聚合性能。 |
| [^328] | [MaskCoFT: Masked Co-Adaptive Fine-Tuning for Memory-Efficient MoE Inference](https://arxiv.org/abs/2609.34077) | 提出MaskCoFT方法，利用可学习二值掩码限制每层的Top-K路由，并通过交叉熵损失协同微调路由器与专家，使专家在卸载推理场景下被高效复用，从而降低MoE模型的内存开销并保持推理性能。 |
| [^329] | [What Does a ProcGen Generalization Gap Measure? Action Rules, Residual Entropy, and the Missing Random Floor](https://arxiv.org/abs/2609.32532) | 该论文提出强化学习的泛化差距应对照“随机下限”（均匀随机策略在同一评估框架和相同关卡上的回报）来解读，并证明测试时动作规则（采样与 argmax）的选择以及动作等效性造成的残余熵会显著改变 ProcGen 基准上泛化结论的含义。 |
| [^330] | [AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation](https://arxiv.org/abs/2609.32411) | 提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。 |
| [^331] | [Efficient Support Recovery of Mixtures of Sparse Linear Classifiers with Fewer Measurements](https://arxiv.org/abs/2609.32176) | 本文提出了自适应与非自适应的支撑集恢复方案，在稀疏线性分类器混合模型中同时实现了更少的测量次数和亚线性解码时间，显著优于已有方法。 |
| [^332] | [Fixed Points Without Fixed Diffusion: Implicit Neural Sheaves for Convergent Test-Time Computation](https://arxiv.org/abs/2609.30277) | 提出 SheafDEQ，一种基于自适应神经层束传播的次齐次深度平衡架构，通过可学习的矩阵值层束限制映射实现更丰富的边依赖变换，并在温和条件下保留了隐式图神经网络不动点唯一且可收敛的保证，从而兼顾表达能力和测试时计算的收敛性。 |
| [^333] | [Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders](https://arxiv.org/abs/2609.27248) | 本文提出LLMAE方法，通过在预训练语言模型内部引入固定长度潜瓶颈，将其改造为高保真连续文本自编码器，可近乎完美地重建长达1024个token的文本序列。 |
| [^334] | [Penalized Nonreversible Langevin for Constrained Sampling](https://arxiv.org/abs/2609.25381) | 提出了将平方距离惩罚与非可逆斜对称扰动相结合的朗之万算法以实现紧凸集上的约束采样，并在对数索博列夫不等式与漂移收缩条件下给出了非渐近的总变差和 2-Wasserstein 误差界。 |
| [^335] | [Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation](https://arxiv.org/abs/2609.20744) | 提出Video DeltaNet（VDN），通过将局部Softmax注意力与引入视频增量注意力（VDA）的双向线性记忆相结合的混合架构，解决视频扩散模型中的注意力计算瓶颈，实现高质量的直播视频生成。 |
| [^336] | [Bayesian Optimization with Rich Auxiliary Information via LLMs](https://arxiv.org/abs/2609.19437) | 本文提出三种利用大语言模型将丰富辅助信息（如训练曲线、专家笔记和先验知识）融入贝叶斯优化的方法，在超参数优化基准和真实核聚变优化任务中始终优于标准BO及现有LLM优化方法。 |
| [^337] | [TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation](https://arxiv.org/abs/2609.17956) | 该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。 |
| [^338] | [Towards Identifying the Dataset Biases Causing Phantom Transfer](https://arxiv.org/abs/2609.14449) | 本文提出一种基于Sentence BERT嵌入的简单特征签名方法，能够在教师模型已知时以0.83的马修斯相关系数识别数据集中隐藏的偏见主题，为检测导致幻影迁移的数据集偏见提供了新途径。 |
| [^339] | [Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion](https://arxiv.org/abs/2609.12223) | 本文提出GRACE模型，通过早期融合的几何残差加合物条件化方法调整预训练分子几何编码器，将加合物感知的残差学习目标与编码器内的加合物条件化机制相结合，显著提升了对气相分子离子碰撞截面的三维预测精度。 |
| [^340] | [Suan: Rectifying Direct Preference Safety Alignment in Large Language Models](https://arxiv.org/abs/2609.08634) | Suan是一种新颖的偏好优化算法，通过直接在梯度层面构建优化目标（绕过标准变分推导），使大语言模型在实现卓越安全对齐的同时完全保留回应实用性。 |
| [^341] | [LayerRoute: Action-Conditioned Mixture-of-Layers Routing for Vision-Language-Action Policies](https://arxiv.org/abs/2609.06079) | 提出LayerRoute，一个动作条件的混合层路由接口，使VLA策略能够根据具体动作需求自适应地访问和混合VLM不同层次的表示，突破了传统固定层分配的限制。 |
| [^342] | [Fractal dimension predicts quantum kernel collapse in angle-encoded data](https://arxiv.org/abs/2609.00475) | 该论文提出用数据的相关性分形维度 D2 作为先验量子比特数预算，可准确预测并避免角度编码量子核的几何坍缩，使量子核在真实硬件上以最小比特数保持有效。 |
| [^343] | [Exact Recovery Thresholds for Weighted Data Selection in Vector-Valued Linear Regression](https://arxiv.org/abs/2608.30254) | 本文解决了COLT 2025公开问题中回归任务加权数据选择的阈值部分，证明了在每个有限数据集上恢复全数据损失所需的最小加权样本预算恰好为 (m+1)d，并精确确定了加权选择曲线在阈值附近和张成预算处的取值。 |
| [^344] | [Exact Risk Ratios for Weighted Data Selection in Linear Regression](https://arxiv.org/abs/2608.28007) | 本文解决了 Hanneke 等人提出的加权数据选择公开问题，精确确定了最小范数 ERM 下的最坏风险比率 F_w(d,n) 的多个取值，包括证明 F_w(d,2d-1)=1+1/d，并给出了中间预算 n=d+k 情形下基于平衡划分调和量的紧致下界 1+Γ_{d,k}。 |
| [^345] | [Steering Recurrent Reasoners at Inference Time with Readout Feedback](https://arxiv.org/abs/2608.24136) | 本文提出读出反馈（RoFB），一种无需重训练的推理时干预方法，通过将中间预测转化为耦合力注入潜在动态，显著提升循环模型在推理任务中的表现。 |
| [^346] | [Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs](https://arxiv.org/abs/2608.21134) | 提出一种无需训练数据的2.7位量化框架，将视觉语言模型压缩至3.7 GB，同时保持视觉问答性能，适用于移动设备高效推理。 |
| [^347] | [An Irreducible Quantum Advantage in Aligning World Models with Reality](https://arxiv.org/abs/2608.19779) | 本文证明即使真实世界是经典的，经典世界模型也无法完美对齐代理策略，而量子模型可能提供不可约的优势。 |
| [^348] | [Geometric Data Perturbation with Noisy-Anchor Alignment for Privacy-Preserving Collaborative Learning](https://arxiv.org/abs/2608.18749) | 本文提出带噪声锚点对齐的几何数据扰动方法，在分析者-参与者共谋下平衡隐私保护与模型效用，实现了既抵抗共谋攻击又保持协作学习性能的目标。 |
| [^349] | [Fine-Tuning Generative Models for Extreme Events via CVaR-Penalized Wasserstein Gradient Flows](https://arxiv.org/abs/2608.11544) | 提出了一种基于CVaR惩罚的Wasserstein梯度流方法，无需先验知识即可微调生成模型以捕捉重尾分布和极端事件，克服了标准生成器在尾部欠采样时速度消失的局限。 |
| [^350] | [Safe and Robust Neural Policy Learning with Statistical Verification for Sim-to-Real Deployment in Robotics](https://arxiv.org/abs/2608.06481) | 本文提出一种课程驱动的闭环框架，将基于场景的进化策略与统计模型检测验证相结合，在协同优化策略性能的同时逐步扩大安全操作边界，最终生成带有统计验证安全与性能保证的神经控制器，助力可靠的仿真到现实部署。 |
| [^351] | [Learning to Rank Tensor Network Contraction Plans for GPU-Accelerated Quantum Circuit Simulation](https://arxiv.org/abs/2608.05819) | 该论文提出一种学习排序框架，通过从成对收缩序列中提取结构特征并训练梯度提升排序器，在执行前准确预测并选择GPU上最高效的张量网络收缩计划。 |
| [^352] | [Escaping Oversquashing: Addressable and Support-Aware Global Memory for Message Passing Networks](https://arxiv.org/abs/2608.02709) | 该论文提出一种兼具可寻址性与支持感知的全局记忆机制，通过乘性读写映射仅用对数级地址编码维度即可选择 M 个记忆行，并借助学习到的私有锚点保持读取有界，从而解决消息传递网络中多节点共享虚拟节点全局状态导致的瓶颈问题。 |
| [^353] | [Using large language models to probe the limits of atom-centered structural descriptors](https://arxiv.org/abs/2607.26984) | 研究人员借助大型语言模型发现了即使采用多达七个近邻构建的原子中心结构描述符仍无法区分的三维原子结构，揭示了广泛使用的结构描述符层级体系存在的根本性局限。 |
| [^354] | [ORACLE: Agentic AI Orchestrator Routing Via Adaptive Verifier Calibration Feedback](https://arxiv.org/abs/2607.22465) | ORACLE提出了一种并发感知的在线智能体路由机制，通过将自适应路由与自适应验证器校准反馈相结合，解决了固定验证器难以泛化到异构智能体任务、以及验证器位于关键路径导致并发请求服务质量下降的问题，且无需训练即可即插即用。 |
| [^355] | [Deep learning-based prediction of time-resolved adhesive forces in viscoelastic Hertzian contacts](https://arxiv.org/abs/2607.19060) | 本文提出一种标量条件化的有状态序列到序列深度学习模型，结合固定测量步长（FMS）表示方法，能够从位移历史快速预测粘弹性赫兹接触中的完整时间分辨粘附力演化，克服了传统数值模拟计算成本高、无法用于实时应用和设计优化的局限。 |
| [^356] | [DriftWorld: Fast World Modeling through Drifting](https://arxiv.org/abs/2607.15065) | DriftWorld基于漂移生成模型学习条件漂移，实现单次前向传播即可生成未来观测的动作条件世界模型，速度超过40 fps、比扩散基线快12倍以上且视觉质量相当或更优。 |
| [^357] | [DAGR: State-Conditioned Goal Representations via Difference-Aware Goal Cross-Attention](https://arxiv.org/abs/2607.13731) | 提出DAGR方法，通过多尺度门控交叉注意力和差异感知的注意力规则，将目标条件强化学习中静态的目标嵌入精炼为状态条件化表示，使策略能直接感知目标中尚未完成的部分，并从理论上揭示了后归一化放置方式对门控结构保证条件的破坏。 |
| [^358] | [Autoregressive latent diffusion for 3D molecule generation](https://arxiv.org/abs/2607.09277) | KRONOS是一个潜在自回归扩散框架，在统一自编码器的潜在空间中联合建模分子图拓扑与几何结构，无需预先指定分子大小即可生成3D分子，并通过FIM启发的混合训练策略平衡无条件生成与片段条件化生成，助力药物发现。 |
| [^359] | [Reliable mechanistic operator recovery with biologically-informed neural networks: principles for architecture and optimisation design](https://arxiv.org/abs/2607.07425) | 本文通过实证研究系统考察了网络架构设计、优化超参数和数据信息对生物信息神经网络（BINNs）机制算子恢复可靠性的影响，为BINNs的架构与优化设计提供了原则性指导。 |
| [^360] | [ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation](https://arxiv.org/abs/2607.06565) | ELSA3D提出弹性语义锚定机制，通过尺度感知八叉树分词器与稀疏的跨模态锚定token，在匹配的抽象尺度上显式对齐语言与几何推理，实现统一的3D理解与生成。 |
| [^361] | [Sentence-Level Context Sensitivity as a Training-Free Detector of Unsupported Content, Evaluated Against Trained Verifiers](https://arxiv.org/abs/2607.04223) | 该论文提出将句子在有/无上下文时的似然差异作为免训练的句子级无依据内容检测器，在多段落RAG答案中其检测能力可与经过训练的验证器相媲美，且无需额外训练、成本更低。 |
| [^362] | [Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training](https://arxiv.org/abs/2607.01763) | 本文通过自蒸馏策略优化（SDPO）重新审视同策略自蒸馏，发现其在持续后训练中比GRPO引发更严重的遗忘甚至崩溃，证明“更密集”的同策略监督信号并不等于更好。 |
| [^363] | [Decision-Aware Training for Sample-Based Generative Models](https://arxiv.org/abs/2607.01171) | 提出决策感知训练方法，通过可微分优化层计算决策损失并将其与能量分数结合，使样本生成模型的训练能够直接惩罚下游决策成本，从而在高风险决策场景中生成更具实用价值的概率预测。 |
| [^364] | [Learning the structure of open quantum systems](https://arxiv.org/abs/2606.30358) | 该论文提出了学习开放量子系统中常数局域 Lindbladian 系数的最优算法，在不知道系统结构的情况下，以 O(g d² log(n) / ε²) 的总演化时间实现准局域和幂律 Lindbladian 的学习，并在除对数因子外证明了其最优性。 |
| [^365] | [PerturbCellRL: Aligning Distributions and Grounding Biology via Post-Training Perturbation Generators](https://arxiv.org/abs/2606.27752) | 本文提出PerturbCellRL强化学习框架，通过基于基因表达能量见证的逐细胞奖励对单细胞扰动生成器进行后训练，恢复了流匹配训练无法捕捉的目标分布，从而更好地建模细胞异质性。 |
| [^366] | [Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference](https://arxiv.org/abs/2606.27090) | 本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。 |
| [^367] | [Latent Goal Prediction from Language for Model-Based Planning](https://arxiv.org/abs/2606.20627) | LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。 |
| [^368] | [Flow Map Denoisers: Traversing the Distortion-Perception Plane for Inverse Problems](https://arxiv.org/abs/2606.19802) | 本文证明流映射模型隐式定义了一个由前瞻参数t控制的单参数去噪器族，可在失真-感知前沿上连续移动工作点，为逆问题求解提供了在最小均方误差与感知质量之间灵活权衡的新机制。 |
| [^369] | [GB-LSR: Local Spectral Decoding with a Learned Global Bandwidth for Arbitrary-Scale Super-Resolution](https://arxiv.org/abs/2606.19617) | 提出GB-LSR，一种以单一可学习全局带宽控制截断傅里叶基规模的固定网格局部谱表示，实现了高效的连续图像解码，其任意尺度超分辨率扩展在统一评测协议下兼顾速度与质量。 |
| [^370] | [Do as the Romans Do: Learning Universal Behaviors from Heterogeneous Agents](https://arxiv.org/abs/2606.18537) | 提出GRID方法，通过信息瓶颈将异构智能体的奖励函数解耦为通用奖励与特定奖励，从追求不同目标的示范者群体中提取普遍有用的行为，为通用智能体预训练开辟了新范式。 |
| [^371] | [Uncertainty Quantification for Flow-Based Generalist Robot Policies](https://arxiv.org/abs/2606.18043) | 本文提出一种利用小规模集成中的速度场分歧（VFD）来高效量化流匹配通用机器人策略认知不确定性的方法，可成功用于部署时的故障检测和主动微调。 |
| [^372] | [Dissecting model behavior through agent trajectories](https://arxiv.org/abs/2606.17454) | 该论文提出“意图-执行”差距的概念，指出智能体性能本质上是系统问题而非单纯的建模问题，并开发了可跨多个模型家族（Claude、Gemini、GPT、Grok、Qwen）泛化的简单可定制框架SSA，以弥合模型能力与框架执行之间的鸿沟。 |
| [^373] | [ROVE: Unlocking Human Interventions for Humanoid Manipulation via Reinforcement Learning](https://arxiv.org/abs/2606.17011) | ROVE提出了一种强化学习框架，通过人在回路的数据收集流程和乐观价值估计方法，从不完美的人类干预轨迹中筛选高价值行为，实现人形机器人VLA模型的有效后续训练。 |
| [^374] | [Diffusion Flow Matching: Dimension-Improved KL Bounds and Wasserstein Guarantees](https://arxiv.org/abs/2606.16610) | 本文为基于布朗运动的扩散流匹配提供了在KL散度和2-Wasserstein距离下具有更优维度依赖性的离散化误差收敛保证，在温和条件下达到了最先进的收敛标度。 |
| [^375] | [Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees](https://arxiv.org/abs/2606.14289) | 本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。 |
| [^376] | [Reliability of Probabilistic Emulation of Physical Systems](https://arxiv.org/abs/2606.12997) | 本研究开发了一个评估框架，在匹配的模型规模和计算预算下系统比较了生成式模型与CRPS训练的确定性模型集合在物理系统概率预报中的表现，发现CRPS训练的模型集合在预测区间的经验覆盖率上通常具有更可靠的不确定性。 |
| [^377] | [Min-Cost Flow Routing for Evidence Assembly in Long Multimodal Documents](https://arxiv.org/abs/2606.07235) | 该论文提出FlowReader，将长多模态文档问答中的证据选择建模为带容量限制的最小成本流问题，通过谱分解按比例分配证据预算并生成短证据链，无需语言模型规划即可实现全面覆盖，在VisDoMBench上取得最高宏观准确率68.9。 |
| [^378] | [Low-Frequency Shortcuts in Texture-Driven Visual Learning](https://arxiv.org/abs/2606.03493) | 纹理驱动的视觉学习存在基于低频成分的捷径学习问题，剪除低频成分可使同分布准确率提升最多10%、分布外准确率提升最多40%。 |
| [^379] | [Everywhere Learning: Artificial Intelligence with Pointwise Constraints](https://arxiv.org/abs/2606.01557) | 本文提出“全域学习”新范式，要求AI系统以概率一满足数据分布上每一点的损失约束而非仅最小化平均损失，并通过近似对偶理论证明其经验解与统计解之间的泛化接近性，且泛化可通过稀疏L1惩罚加以控制。 |
| [^380] | [Efficient Exploration for Iterative Nash Preference Optimization](https://arxiv.org/abs/2606.01382) | 论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。 |
| [^381] | [On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance](https://arxiv.org/abs/2606.00467) | 提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。 |
| [^382] | [A Pre-Training Analogue of Grokking in Language Models: Tracing Delayed Grammatical Generalization](https://arxiv.org/abs/2606.00230) | 提出了一种基于暴露度的评估框架，利用BLiMP最小对立对及其关键短语来模拟预训练中的训练/验证划分，首次在LLM预训练过程中观察到跨越五种语法现象的类似“顿悟”的延迟语法泛化。 |
| [^383] | [Fixed Universal Transformers](https://arxiv.org/abs/2605.31423) | 本文提出固定参数的“通用Transformer”，证明其可通过输入嵌入模拟任意同类别Transformer，且随机初始化的固定Transformer几乎必然具有通用性，表明Transformer的表达能力主要来自输入表示而非学习到的权重。 |
| [^384] | [Escaping the Capacity Ceiling: Routing on the Stiefel Manifold for Bilinear SPD Layers](https://arxiv.org/abs/2605.31043) | 提出SCAP层，通过交叉注意力将K个Stiefel专家滤波器动态组合为样本特定的双线性映射，从而突破SPD网络中单滤波器的容量上限，解决堆叠BiMap层无法提升容量的问题。 |
| [^385] | [FastKernels: Benchmarking GPU Kernel Generation in Production](https://arxiv.org/abs/2605.23215) | FastKernels提出了一个包含384个任务的生产级GPU内核生成基准，通过组合层次结构覆盖94.6%的HuggingFace Transformers架构，并直接在生产执行路径上以框架官方发布的内核为基准对候选内核进行内核级和端到端评分。 |
| [^386] | [Stimulus symmetries can confound representational similarity analyses](https://arxiv.org/abs/2605.21324) | 该论文揭示了网络输入中的刺激对称性会混淆基于表征相似性矩阵（RSM）的分析，因为功能等价的表征配置可以产生性质截然不同的表征几何结构，即使在编码图像数据的实际网络中也存在这一现象。 |
| [^387] | [Teger: Spatiotemporal Covariance for Probabilistic Traffic Forecasting](https://arxiv.org/abs/2605.18068) | 提出TEGER残差协方差模型，通过闭式更新在测试时动态校正交通预测的联合不确定性而无需重新训练，并可附加于冻结的时间序列基础模型。 |
| [^388] | [HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents](https://arxiv.org/abs/2605.17873) | HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。 |
| [^389] | [Identify then Realize: Contrastive Learning of Latent Port-Hamiltonian Dynamics from Partial Observations](https://arxiv.org/abs/2605.16682) | 提出两阶段“先辨识后实现”框架CIPHER，先用对比学习从部分观测中辨识潜在状态表示，再将其实现为端口哈密顿动力学，从而学习保守与耗散系统的物理一致模型，并给出了状态可恢复性与动力学可实现性的理论保证。 |
| [^390] | [LEAF: A Living Benchmark for Event-Augmented Forecasting](https://arxiv.org/abs/2605.16358) | LEAF是首个面向事件增强预测任务的动态基准，通过递归检索智能体系统与双智能体交叉验证收集时间对齐的辅助上下文，将未来信息泄露从8.6%降至1.6%。 |
| [^391] | [GPart: End-to-End Isometric Fine-Tuning via Global Parameter Partitioning](https://arxiv.org/abs/2605.14841) | GPart 通过稀疏等距划分矩阵将可训练向量直接映射到完整权重空间，去除了 LoRA 式低秩重构，实现了端到端等距且高度参数高效的微调。 |
| [^392] | [ASH: Agents that Self-Hone in Long-Horizon Worlds](https://arxiv.org/abs/2605.14211) | ASH是一个无需奖励工程或专家标注的智能体系统，通过自我改进循环从自身轨迹学习逆动力学模型，进而从无标注网络视频中提取监督信号并保留关键时刻作为长期记忆，从而在《宝可梦：绿宝石》和《塞尔达传说：缩小帽》等需要数小时规划的长时程任务中实现自我提升。 |
| [^393] | [Embodied Neurocomputation: A Framework for Interfacing Biological Neural Cultures with Scaled Task-Driven Validation](https://arxiv.org/abs/2605.13315) | 该论文提出具身神经计算框架，并首次对生物神经网络智能体在模拟网格世界中执行气味梯度闭环导航任务的编码配置开展了大规模参数优化。 |
| [^394] | [ReForge: Refining Merged Models with Anchor-Regularized Regression](https://arxiv.org/abs/2605.12843) | 提出双层优化框架ReForge，将强合并模型作为锚点先验，通过贝叶斯线性回归对模块级进行精炼，并利用贝叶斯优化联合选择正则化强度与组装尺度，同时提供无需校准数据的任务向量Gram变体。 |
| [^395] | [How Far Does a Shared Linear Map Go? Probing Feature-Space Manipulability for Image Editing](https://arxiv.org/abs/2605.11203) | 该研究发现，一个简单的空间共享线性映射就能近乎等同于更强表达能力的探针，预测有监督视觉骨干网络对几何、光度及语义等各类图像编辑的内部特征响应，且其充分性随网络深度增加而提升。 |
| [^396] | [Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning](https://arxiv.org/abs/2605.10247) | GTLM通过将图感知注意力偏置直接注入预训练LLM的注意力模块，仅以0.015%的额外参数使LLM原生处理图拓扑结构，无需GNN流水线即可统一文本与图结构推理，避免了节点语义信息的丢失。 |
| [^397] | [System-Prompt Anchoring with Cross-Attention Layers](https://arxiv.org/abs/2605.09737) | 该研究通过在冻结的因果解码器主干中插入交叉注意力层来锚定系统提示词，发现插入位置对性能影响显著，较后的位置通常更有效且参数效率更高，并能改变指令遵循与安全行为而不损害通用任务性能。 |
| [^398] | [MolWorld: Molecule World Models for Actionable Molecular Optimization](https://arxiv.org/abs/2605.08954) | 提出分子世界模型 MolWorld，将可操作分子优化形式化为分子转移图的迭代扩展，通过匹配分子对（MMP）边显式建模可达性，确保优化得到的候选分子可从已知分子经局部结构修饰到达。 |
| [^399] | [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) | RAO提出了一种强化学习方法，通过训练智能体递归地生成并委派子任务给自身的新实例来实现推理时的分治扩展，使模型能够突破上下文窗口限制、泛化到远难于训练任务的问题，并降低实际运行时间。 |
| [^400] | [Trade-off Functions for DP-SGD with Subsampling based on Random Allocation: Tight Upper and Lower Bounds](https://arxiv.org/abs/2605.06259) | 该论文在f-DP框架下首次对基于随机分配子采样的DP-SGD给出了紧致的权衡函数分析，利用Berry-Esseen定理推导出透明、可解释的封闭形式上下界，在单个epoch下紧致至常数因子。 |
| [^401] | [CoMemNet: A Continual Memory Network with Drift-Aware Sampling for Traffic Prediction](https://arxiv.org/abs/2605.05738) | 提出 CoMemNet，一种通过在线/目标双分支、基于 Wasserstein 的漂移感知采样和节点自适应时间记忆重放缓冲，在演进的交通传感器网络上实现无需固定邻接矩阵与全量重训的高效持续交通预测模型。 |
| [^402] | [Dual Certified White-Box Inference for Input Convex Neural Networks](https://arxiv.org/abs/2605.04722) | 该论文提出利用SOC-ICNN与参数化二阶锥规划价值函数的精确对偶表示，开发双认证白盒推断方法DCI，从最优对偶乘子恢复完整次微分、提供精确平稳性认证与下降方向，并实现牛顿加速与全局收敛。 |
| [^403] | [Useful Features, Backward Scores: OOD in Language-Model Trajectories](https://arxiv.org/abs/2605.00269) | 该论文的核心发现是，能区分输入组的特征并不一定能产生有用的OOD异常排序——在语言模型轨迹中，可区分特征对应的距离得分甚至会出现反转（异常组中心更远但散布更紧），且这一现象在毒性、反讽等多个数据集上均稳定存在。 |
| [^404] | [Timescale Separation Enables Deep Reinforcement Learning Control of Rotating Detonation Engine Mode Transitions](https://arxiv.org/abs/2604.14398) | 通过在跟随爆震波的移动参考系中重新构建深度强化学习问题，实现快速爆震传播与慢速模式动力学之间的时间尺度分离，从而使DRL能够有效控制旋转爆震发动机的模式转换。 |
| [^405] | [Rhetorical Questions in LLM Representations: A Linear Probing Study](https://arxiv.org/abs/2604.14128) | 该研究通过线性探针发现大语言模型在表示空间中能够早期且稳定地编码反问句信号，其跨数据集可迁移性虽然存在，但并不意味着模型内部存在统一的共享表示。 |
| [^406] | [Gradient Descent's Last Iterate is Often (slightly) Suboptimal](https://arxiv.org/abs/2604.13870) | 本文证明了Jain等人提出的猜想：在没有时间范围 $T$ 先验知识的情况下，任何步长序列都无法使SGD的最后迭代点达到最优的 $1/\sqrt{T}$ 收敛速率，即使无噪声的梯度下降也不可避免地存在关于 $T$ 的多对数因子损失。 |
| [^407] | [Classical and Quantum Speedups for Non-Convex Optimization via Energy Conserving Descent](https://arxiv.org/abs/2604.13022) | 本文首次对能量守恒下降（ECD）进行了理论分析，证明其随机版本和量子版本在非凸优化中相比随机梯度下降和量子隧穿游走基线均能实现指数级的命中时间加速，且量子版本在高势垒问题上具有进一步的加速优势。 |
| [^408] | [Multimodal Ambivalence/Hesitancy Recognition in Videos for Personalized Digital Health Interventions](https://arxiv.org/abs/2604.11730) | 本文针对个性化数字健康干预，研究从视频中多模态识别矛盾与犹豫（A/H）情绪——这种跨模态或模态内的情感不一致状态是导致患者延迟、回避或放弃健康干预的关键因素。 |
| [^409] | [Verify Before You Fix: Agentic Execution Grounding for Trustworthy Cross-Language Code Analysis](https://arxiv.org/abs/2604.10800) | 该论文的核心创新是提出一个由LLM驱动的跨语言漏洞生命周期框架，以“未经执行确认可利用性就不得修复”这一严格不变式为准则，将结构-语义混合检测、基于执行的智能体验证与感知验证的迭代修复三个阶段串联起来，并借助通用抽象语法树与 GraphSAGE、Qwen2.5-Coder 嵌入的混合融合实现 Java、Python、C++ 的跨语言泛化，从而保证代码分析与修复建立在可验证的证据之上。 |
| [^410] | [Scalar Federated Learning for Linear Quadratic Regulator](https://arxiv.org/abs/2604.05088) | 提出ScalarFedLQR算法，让每个智能体仅上传一个标量梯度投影，将上行通信量从O(d)降至O(1)，并且参与智能体越多、梯度重构越精确、线性收敛越快，实现了高维LQR控制的高效无模型联邦学习。 |
| [^411] | [Learning an Interpretable Risk Scoring System for Maximizing Decision Net Benefit](https://arxiv.org/abs/2604.04241) | 本文提出一种通过稀疏整数线性规划直接优化决策净收益、且具有整数系数的可解释风险评分系统，并建立了净收益与判别力、校准之间的理论关系。 |
| [^412] | [Spectral Alignment in Forward-Backward Representations via Temporal Abstraction](https://arxiv.org/abs/2603.20103) | 本文证明时间抽象如同低通滤波器，可抑制高频谱分量、降低后继表示的有效秩并保持价值函数误差界，从而缓解连续环境高秩转移动力学与FB低秩瓶颈之间的谱失配，是实现稳定前向-后向表示学习的关键因素。 |
| [^413] | [A Locally Penalized Cross-Estimate Federated Method with Guarantees for Constrained Personalized Learning](https://arxiv.org/abs/2603.19617) | 该论文提出了一种局部惩罚交叉估计联邦平均方法，让每个参与方在异构可行约束下维护各自独立的可行模型，并通过协作目标耦合各模型，为约束个性化联邦学习提供了理论保证。 |
| [^414] | [Exploring Subnetwork Interactions in Heterogeneous Brain Network via Prior-Informed Graph Learning](https://arxiv.org/abs/2603.19307) | 提出KD-Brain框架，通过语义条件化交互机制和病理一致性约束将语义与临床先验知识注入图学习过程，有效解决了小样本条件下脑功能子网络交互建模难题，实现精神障碍诊断的最先进性能。 |
| [^415] | [On the Tip of the Tongue: Why LLMs Hallucinate Answers They Can Decode](https://arxiv.org/abs/2603.13911) | 该论文提出在首个答案标记处区分“读取”与“写出”的新框架，揭示大语言模型产生幻觉的关键原因并非正确答案无法从中间状态解码，而是最终读出时的“选择边际”不足，使更强的竞争标记压制了正确答案。 |
| [^416] | [Can AI Understand the Language of Origami?](https://arxiv.org/abs/2603.13856) | 该论文提出OrigamiBench基准，通过物理接地的高层折叠动作语言评估AI对折纸合成机制的程序化理解能力，同时整合了视觉感知、几何物理约束推理和顺序规划等多项能力。 |
| [^417] | [Theoretical Lower Bounds on the Robustness of Deep ReLU Networks](https://arxiv.org/abs/2602.18674) | 该论文通过结合高维几何中的测度集中性与一项新的几何刻画——即ReLU网络诱导的输入空间划分中每个凸多面体区域的面数至多等于网络单元数——证明了深度ReLU网络对随机L₂扰动的局部鲁棒性理论下界，并分析了该鲁棒性随输入维度的缩放规律。 |
| [^418] | [Demonstration-Guided Observation Attacks on Black-Box Safe Reinforcement Learning Controllers for Robotic Systems](https://arxiv.org/abs/2602.16543) | 提出一种仅需示范数据作为受害方信息的示范引导观测攻击框架，通过逆约束强化学习恢复状态约束与代理策略并学习动力学，无需访问受害模型参数、梯度或查询即可生成有界观测扰动，有效攻击黑盒安全强化学习机器人控制器并诱导安全违规。 |
| [^419] | [Goldilocks RL: Tuning Task Difficulty to Escape Sparse Rewards for Reasoning](https://arxiv.org/abs/2602.14868) | 提出Goldilocks自适应数据选择策略，利用选择器网络预测问题的奖励波动性，优先选取难度适中（既不太简单也不太难）的训练问题，从而摆脱稀疏奖励困境，提升语言模型推理强化学习的样本效率。 |
| [^420] | [Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning](https://arxiv.org/abs/2602.10273) | Power-SMC是一种免训练的低延迟序列级幂采样方法，以接近标准解码的速度实现分布锐化，从而提升大语言模型的推理能力。 |
| [^421] | [ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling](https://arxiv.org/abs/2602.09009) | 该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。 |
| [^422] | [ChronoSpike: An Adaptive Spiking Graph Neural Network for Dynamic Graphs](https://arxiv.org/abs/2602.01124) | ChronoSpike提出一种自适应脉冲图神经网络，通过融合逐通道膜动力学的可学习LIF神经元、多头空间注意力聚合与轻量级Transformer时间编码器，在低内存开销下实现了动态图上细粒度的局部建模与长程时间依赖捕获。 |
| [^423] | [Recoverability Has a Law: The ERR Measure for Tool-Augmented Agents](https://arxiv.org/abs/2601.22352) | 本文提出期望恢复遗憾（ERR）指标并证明其与可观测的效率得分（ES）之间存在一阶定量关系，首次为工具增强语言模型智能体失败后的自我恢复能力建立了可证伪的预测性定律，并在五个工具使用基准上得到实证验证。 |
| [^424] | [Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models](https://arxiv.org/abs/2512.21439) | 提出了COMETH框架，将概率情境学习与大语言模型语义抽象及人类道德判断数据相结合，从数据中学习可解释的道德情境，证明道德评价是高度情境化的。 |
| [^425] | [Uncovering EEG Patterns Consistently Associated with Cybersickness Discomfort Using Deep Learning Interpretability Maps](https://arxiv.org/abs/2512.20620) | 该研究构建了一个结合神经网络与可解释性图谱的框架，基于两项独立的ERP晕屏症用户研究，成功识别出对分类晕屏症相关不适最关键的时空脑电特征，从而揭示了与晕屏症不适稳定一致的EEG模式。 |
| [^426] | [Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling](https://arxiv.org/abs/2512.19905) | 该论文提出了一个解析可处理的推理时扩展模型——带奖励加权采样器的贝叶斯线性回归，用以模拟LLM作为评判者的场景，并在高维机制下推导出后验预测均值与方差的闭式表达式，从而揭示推理时扩展背后的数学原理。 |
| [^427] | [Evaluation of Sampling Strategies and Physics-Informed Kolmogorov--Arnold Networks in Unbounded Domains](https://arxiv.org/abs/2512.12074) | 该论文通过基准测试系统评估了无界域反演PDE问题中不同采样策略（均匀、高斯、指数分布）与网络架构（MLP与KAN）的表现，为物理信息神经网络在无限和半无限域上的应用提供了定量的精度与效率参考。 |
| [^428] | [Closing the Train-Test Gap in World Models for Gradient-Based Planning](https://arxiv.org/abs/2512.09929) | 本文通过提出训练时数据合成技术来弥合世界模型“以下一状态预测为训练目标”与“测试时用于估计动作序列”之间的训练-测试差距，从而显著提升基于梯度规划的性能，使其成为高效且性能优异的规划方法。 |
| [^429] | [Fragmentation is Efficiently Learnable by Quantum Neural Networks](https://arxiv.org/abs/2512.00751) | 该论文证明当量子系统的碎片化满足特定条件时，碎片分类问题可被量子神经网络高效解决，而已知经典去量子化技术对此失效，为物理动机的量子机器学习任务提供量子优势提供了一个罕见范例。 |
| [^430] | [When Does Pooling Pay? Credibility and Resolution under Forgetting in Intermittent-Demand Forecasting](https://arxiv.org/abs/2511.12749) | 该论文证明，在分层经验贝叶斯间歇性需求预测模型中，“遗忘序列自身历史”与“跨序列信息共享”可统一为同一个决策，并提出拟合窗口诊断、可信度界、分辨条件及事前筛选准则，用以判断何时池化信息才有价值。 |
| [^431] | [High-Dimensional Asymptotics of Differentially Private PCA](https://arxiv.org/abs/2511.07270) | 该论文针对差分隐私主成分分析，通过分析指数机制，在高维设置下给出了隐私损失随数据集变化的精确渐近刻画，弥补了传统一致上界在特定数据集上过于保守的不足。 |
| [^432] | [Bifidelity Karhunen-Lo\`eve Expansion Surrogate with Active Learning for Random Fields](https://arxiv.org/abs/2511.03756) | 提出了一种将Karhunen-Loève展开与多项式混沌展开相结合的双保真度代理模型，并利用基于交叉验证和高斯过程回归的主动学习策略自适应选择高保真度采样点，从而在有限计算成本下实现随机场的高精度建模。 |
| [^433] | [Interpretable Discovery from Unstructured Data: A High-Dimensional Approach](https://arxiv.org/abs/2511.01680) | 该论文提出了一个从非结构化数据（如开放式调查文本）中自动进行可解释发现的通用框架，其核心创新在于结合AI可解释性方法与高维多重检验算法，将非结构化数据转化为可解释的概念测量并进行统计检验，最终生成人类可理解的发现描述。 |
| [^434] | [Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station](https://arxiv.org/abs/2510.23463) | 该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。 |
| [^435] | [Performance Evaluation of Ising and QUBO Variable Encodings in Boltzmann Machine Learning](https://arxiv.org/abs/2510.13210) | 该研究利用费舍尔信息矩阵的谱分析揭示，QUBO编码比Ising编码具有更强的病态性，导致SGD收敛更慢，而全FIM自然梯度下降可消除这种编码间的差异，因此基于SGD的训练更宜采用Ising编码。 |
| [^436] | [Cocoon: A System Architecture for Differentially Private Training with Correlated Noises](https://arxiv.org/abs/2510.07304) | Cocoon 提出了一种系统架构，通过在 CPU、GPU 和内存扩展模块之间分布式地存储与处理庞大的相关噪声历史，并对稀疏嵌入表进行优化，实现了高效的大规模模型差分隐私训练。 |
| [^437] | [Error Propagation in Dynamic Programming: From Stochastic Control to American Option Pricing](https://arxiv.org/abs/2509.20239) | 本文为离散时间随机最优控制建立了结合再生核希尔伯特空间回归与蒙特卡洛抽样的动态规划近似框架，提出自然的误差分解并严格分析了误差从到期日向初始时刻反向传播的规律，可应用于美式期权定价。 |
| [^438] | [Quantum parameter estimation with uncertainty quantification from continuous measurement data using neural network ensembles](https://arxiv.org/abs/2509.10756) | 该论文提出使用深度神经网络集成进行量子参数估计，在保持估计精度的同时量化不确定性并能检测实验数据漂移，推理速度远快于贝叶斯推断方法。 |
| [^439] | [EEGDM: Learning EEG Representation with Latent Diffusion Model](https://arxiv.org/abs/2508.20705) | EEGDM提出了一种基于潜在扩散模型的自监督学习框架，通过生成式去噪过程学习脑电信号的全局时间模式与跨通道关系的紧凑表征，克服了掩码重建方法难以捕捉全局生成约束的局限。 |
| [^440] | [Multimodal Representation Learning Conditioned on Semantic Relations](https://arxiv.org/abs/2508.17497) | 提出了关系条件化多模态学习框架RCML，将自然语言描述的语义关系作为显式条件来学习多模态表示，使同一样本在不同关系下拥有不同表示，克服了CLIP等对比模型单一嵌入的局限。 |
| [^441] | [A Sobel-Gradient MLP Baseline for Handwritten Character Recognition](https://arxiv.org/abs/2508.11902) | 本研究提出一种基于固定Sobel梯度算子与简单多层感知机的受控基线模型，证明仅使用一阶图像梯度特征即可在MNIST和EMNIST Letters手写字符识别上分别取得98.54%和92.50%的高准确率。 |
| [^442] | [Can SGD Handle Heavy-Tailed Noise?](https://arxiv.org/abs/2508.04860) | 本文证明仅假设随机梯度具有有界p阶矩，原始SGD即可在凸、强凸和非凸问题上于重尾噪声下达到极小极大最优的收敛保证与样本复杂度。 |
| [^443] | [Mitigating Watermark Forgery in Generative Models via Randomized Key Selection](https://arxiv.org/abs/2507.07871) | 该论文提出通过对每次查询随机化水印密钥选择的防御方案，使盲攻击者的伪造成功率存在与所收集样本数量无关的上限，且不进一步降低模型效用，从而有效缓解生成模型中的水印伪造攻击。 |
| [^444] | [Estimating prevalence with precision and accuracy](https://arxiv.org/abs/2507.06061) | 本文提出了一种贝叶斯聚合量化器PQ，它在保证足够覆盖率的同时生成更窄的预测区间，从而比现有方法更精确地估计流行率并更有效地量化估计的不确定性。 |
| [^445] | [Coupled reaction and diffusion governing interface evolution in solid-state batteries](https://arxiv.org/abs/2506.10944) | 通过主动学习与深度等变神经网络原子间势实现量子精度的大规模固态电池界面反应模拟，并结合基于局部原子环境聚类的无监督分类方法，首次发现SEI中一种此前未被报道的晶体无序相Li₂S₀.₇₂P₀.₁₄Cl₀.₁₄。 |
| [^446] | [Continuous Policy and Value Iteration for Relaxed Stochastic Control Problems and Its Convergence](https://arxiv.org/abs/2506.08121) | 本文提出一种基于朗之万型动力学的连续策略-值迭代算法，能同时更新值函数与最优控制，并在哈密顿量单调性条件下证明了该算法在无限时间域熵正则化松弛控制问题中收敛于最优控制。 |
| [^447] | [Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces](https://arxiv.org/abs/2506.06557) | 该论文提出将任意相异度函数投影到超度量空间并学习该投影的近似形式，在保持最近邻关系的同时实现最坏情况复杂度仅为树深度的近似向量搜索，并将该方法推广到更一般的q-度量空间。 |
| [^448] | [Robust Adversarial Quantification via Conflict-Aware Evidential Deep Learning](https://arxiv.org/abs/2506.05937) | 提出轻量级后验不确定性量化方法 C-EDL，通过为输入生成多样的任务保持变换并量化表示分歧来校准不确定性，无需重新训练即可增强证据深度学习对对抗性和分布外输入的鲁棒性。 |
| [^449] | [VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation](https://arxiv.org/abs/2505.13439) | VTBench是一个系统性评估自回归图像生成中视觉分词器性能的综合基准，通过图像重建、细节保留和文本保留三大核心任务，揭示了离散视觉分词器与连续VAE之间的性能差距。 |
| [^450] | [Asymptotic Performance of Time-Varying Bayesian Optimization](https://arxiv.org/abs/2505.13012) | 本文首次为时变贝叶斯优化（TVBO）算法的累积遗憾提供了上界和与算法无关的下界，推导出算法具有无悔性质的充分条件，且其分析首次覆盖了实践中使用的所有主要类别的平稳核函数。 |
| [^451] | [A Polynomial-Time Algorithm for Variational Inequalities under the Minty Condition](https://arxiv.org/abs/2504.03432) | 本文提出了首个在Minty条件下求解Lipschitz连续映射的ε-变分不等式的多项式时间算法（复杂度关于维度 $d$ 和 $\log(1/\epsilon)$ 多项式增长），突破了以往方法对 $1/\epsilon$ 的指数级依赖或需单调性等更强假设的限制。 |
| [^452] | [Noise Sensitivity and Learning Lower Bounds for Hierarchical Functions](https://arxiv.org/abs/2502.05073) | 本文证明了树状层次结构函数在每层函数均与线性函数保持 ε-距离时，其噪声稳定性随层次深度呈指数级衰减，并由此推导出基于分层函数的函数类在不可知学习中的统计查询超多项式下界。 |
| [^453] | [Heads, Tails, and AI Fails: LLMs, Randomness, and Human Judgments](https://arxiv.org/abs/2406.00092) | 该研究发现大语言模型在模拟抛硬币时会再现并放大人类的随机性偏差（如过度交替、厌恶长连续序列），提高温度参数只能部分缓解而无法消除这些系统性失真。 |
| [^454] | [Token Space: A Category Theory Framework for AI Computations](https://arxiv.org/abs/2404.11624) | 本文提出了词元空间这一用于AI计算的范畴论框架，并证明通过代数词元化方法可将各类有限性结构对象范畴完全忠实地嵌入其中，同时保持二元乘积和等化子，为AI计算提供了坚实的数学基础。 |
| [^455] | [VIDiff: Translating Videos via Multi-Modal Instructions with Diffusion Models](https://arxiv.org/abs/2311.18837) | 本文首次提出了视频指令扩散基础模型VIDiff，能够根据用户的多模态指令在几秒内完成视频编辑、转换和增强等多种理解与生成任务，并通过迭代自回归方法保证长视频编辑的一致性。 |
| [^456] | [Variance-reduced accelerated methods for decentralized stochastic double-regularized nonconvex strongly-concave minimax problems.](http://arxiv.org/abs/2307.07113) | 本文提出了一种应用于分散随机双正则化非凸强凸极小极大问题的方差减少加速方法，通过引入拉格朗日乘子和采用单个邻居通信并结合方差减少技术，该方法在随机设置下样本复杂度达到$\mathcal{O}(\kappa^3\varepsilon^{-3})$。 |

# 详细

[^1]: 世界模型应该遗忘什么？面向持续适应的分层保留机制

    What Should World Models Forget? Stratified Retention for Continual Adaptation

    [https://arxiv.org/abs/2610.03713](https://arxiv.org/abs/2610.03713)

    提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。

    

    持续学习将在已见数据上的性能退化视为失败的证据，这一惯例继承自预测目标平稳的设定，即正确标签会永远保持正确。世界模型不满足这一条件：它们的预测目标是环境，而环境会变化，因此获取时准确的知识之后可能变为错误，丢弃这些知识是必要行为而非缺陷。非平稳的真实标签问题在概念漂移文献和语言模型的时间事实性研究中已被充分探讨，但尚未在世界模型中被形式化，而世界模型的独特之处在于它们还编码了绝不能被修改的知识。我们认为，持续世界模型需要按不变性时间尺度分层的保留机制，将诸如物理规律和物体恒存性等绝不应被修改的不变量，与应随环境变化而尽快被更新的实例级事实区分开来。

    arXiv:2610.03713v1 Announce Type: cross  Abstract: Continual learning treats degradation on previously seen data as evidence of failure, a convention inherited from settings with a stationary prediction target, where a correct label remains correct indefinitely. World models do not satisfy this condition. Their prediction target is the environment, which changes, so knowledge that was accurate when acquired may later become false, and discarding it is required behavior rather than a defect. Non-stationary ground truth is well studied in the concept drift literature and in the temporal factuality of language models, but has not been formulated for world models, which are distinctive in that they also encode knowledge that must never be revised. We argue that continual world models require retention stratified by invariance timescale, separating invariants such as physics and object permanence, which must never be revised, from instance-level facts that should be revised as soon as the e
    
[^2]: RNADyn：用于生成和理解RNA动力学的基准测试

    RNADyn: A Benchmark for Generating and Understanding RNA Dynamics

    [https://arxiv.org/abs/2610.03712](https://arxiv.org/abs/2610.03712)

    该论文提出了RNADynBench——一个包含2585条质量控制的全原子RNA分子动力学轨迹的标准化基准，并开发了统一模型RNADynNet，通过共享骨干网络结合坐标去噪、单帧到轨迹对齐和物理接地技术，同时实现RNA轨迹生成与动力学表示学习。

    

    核糖核酸（RNA）通过构象变化发挥功能，而这些变化无法被静态结构完全捕捉。然而，大规模标准化的RNA动力学数据仍然有限，且现有方法通常将轨迹生成和动力学理解视为两个独立的目标。在此，我们介绍了RNADynBench，这是一个标准化的RNA分子动力学（MD）基准，包含2585条经过质量控制的100纳秒全原子轨迹以及防泄漏的数据划分。基于RNADynBench，我们开发了RNADynNet，这是一个用于RNA动力学学习的统一模型，它使用共享骨干网络同时实现轨迹生成和从单一构象中提取动力学指纹。该模型结合了坐标去噪、单帧到轨迹对齐以及物理接地技术，将全原子轨迹生成与动力学表示学习联系起来。物理接地技术同时改善了生成的动力学效果以及可从中恢复的物理信息……

    arXiv:2610.03712v1 Announce Type: new  Abstract: Ribonucleic acid (RNA) functions through conformational changes that are not fully captured by static structures. However, large-scale standardized RNA dynamics data remain limited, and existing approaches typically treat trajectory generation and dynamics understanding as separate objectives. Here, we introduce RNADynBench, a standardized RNA molecular dynamics (MD) benchmark with 2585 quality-controlled 100-ns all-atom trajectories and leakage-controlled splits. Building on RNADynBench, we develop RNADynNet, a unified model for RNA dynamics learning that uses a shared backbone for both trajectory generation and dynamics fingerprint extraction from a single conformer. It combines coordinate denoising, single-frame-to-trajectory alignment, and physical grounding to connect all-atom trajectory generation with dynamics representation learning. Physical grounding improves both generated dynamics and the physical information recoverable from
    
[^3]: 从混合到撕裂：基于消息传递的去中心化优化中的图分解

    From Mixing to Tearing: Graph Decomposition in Decentralized Optimization via Message Passing

    [https://arxiv.org/abs/2610.03709](https://arxiv.org/abs/2610.03709)

    该论文从第一性原理出发提出了一种基于图分解的去中心化优化通用框架，通过消息传递联合设计优化子问题、对偶变量分块和协作求解的智能体集群，突破了传统仅利用网络混合信息的范式。

    

    我们研究在无向图上最小化光滑强凸函数之和的问题，其中每个函数由一个智能体（节点）持有，且通信仅限于图中相邻的节点。现有的去中心化方法，无论是基于gossip（闲聊）算法还是基于生成树上的路由机制，通常都是利用网络来混合或聚合信息，以实现预先设定的局部优化更新。这种以通信为中心的视角所缺失的，是一个能够利用图结构来联合设计优化子问题、以及智能体协作求解这些子问题所需的协同计算与通信的通用框架。我们从第一性原理出发构建了这样一个框架，联合设计了协同约束的线性表示、由此产生的对偶变量的分块方式（并对其进行联合优化），以及在指定子图上协作求解每个分块子问题的连通智能体集群。

    arXiv:2610.03709v1 Announce Type: cross  Abstract: We study the minimization of sums of smooth strongly convex functions over undirected graphs, with each function held by one agent and communication restricted to neighbors in the graph. Existing decentralized methods, whether based on gossip or on routing over spanning trees, typically   use the network to mix or aggregate information to enable   {\it prescribed} local optimization   updates. What this communication-centered viewpoint lacks is a general   framework that uses graph structure to {\it jointly} design the   optimization subproblems and the cooperative computation and communication through which agents solve them   cooperatively.   We develop such a framework from first principles,   jointly designing the linear representation of agreement constraints, the blocks of   the resulting dual variables (jointly optimized), and connected cluster of agents that   cooperatively solve each block subproblem over the assigned subgraph
    
[^4]: LESSER：基于输出层梯度的后训练数据选择

    LESSER: Post-Training Data Selection with Output-Layer Gradients

    [https://arxiv.org/abs/2610.03702](https://arxiv.org/abs/2610.03702)

    LESSER提出仅需前向传播的输出层梯度来近似全参数梯度进行后训练数据选择，将特征提取计算成本在SFT和RL场景下分别降低9.7倍和3.0倍，同时保持下游任务性能不变。

    

    大语言模型后训练数据的选择对下游性能有重大影响。基于梯度的数据选择是一种流行的方法，它通过训练数据梯度与小型验证集梯度的对齐程度来对训练数据进行排序。然而，使用全参数梯度进行排序需要对每个样本进行昂贵的反向传播，这使得对大规模候选数据池的计算变得难以承受。由此引出一个自然的问题：我们能否以极低的成本近似全梯度特征？令人欣慰的是，我们发现输出层梯度足以实现有效的数据选择，且只需成本更低的前向传播。我们将其实现为LESSER，这是一个即插即用的数据选择方法封装器，可将特征提取的FLOP计算成本在SFT（监督微调）中降低9.7倍，在RL（强化学习）基准中降低3.0倍，同时在下游任务上保持与全梯度方法相当的性能。经验上，我们发现即使当输出层梯度与全梯度对数据的排序……

    arXiv:2610.03702v1 Announce Type: new  Abstract: The choice of post-training data for large language models substantially affects downstream performance. Gradient-based data selection is a popular approach that ranks training data by how well their gradients align with those of a small validation set. However, ranking with full-parameter gradients requires an expensive backward pass on every sample, making computation intractable for large candidate pools. This raises a natural question: can we approximate full-gradient features at a fraction of the cost? Conveniently, we find that output-layer gradients suffice for effective data selection, yet require only the cheaper forward pass. We implement this as LESSER, a drop-in wrapper for selection methods that reduces the feature-extraction FLOP cost by $9.7\times$ for SFT and $3.0\times$ for RL benchmarks, while tracking full-gradient performance on downstream tasks. Empirically, we find that even when output-layer and full gradients rank
    
[^5]: 基于Wasserstein拉格朗日残差的免模拟种群动力学学习

    Simulation-Free Learning of Population Dynamics with Wasserstein Lagrangian Residuals

    [https://arxiv.org/abs/2610.03679](https://arxiv.org/abs/2610.03679)

    提出免模拟方法Double-Stitch，通过惩罚学习到的种群路径上运动方程的残差来学习Wasserstein空间中的拉格朗日力学，从而以低训练成本建模保守性和周期性的种群动力学。

    

    细胞、生物体和流体的动力学通常被建模为随时间演化的概率分布。从不成对的快照中重建和外推这种演化需要对潜在过程做出假设。Wasserstein梯度流是一种常见的选择，但它无法描述保守性或周期性动力学。Wasserstein空间中的拉格朗日力学则可以同时涵盖两者，但现有的学习方法都是基于模拟的：它们在每次训练步骤中都需要运行数值求解器，这使得训练成本高昂。我们提出Double-Stitch，这是一种免模拟的方法，通过沿着学习到的种群路径惩罚运动方程的残差来学习这些力学。我们从不需要梯度速度的Clebsch变分原理推导出该方程，并证明当方程成立时残差恰好为零。我们在合成数据集、单细胞数据集和海洋涡流数据集上测试了Double-Stitch。

    arXiv:2610.03679v1 Announce Type: new  Abstract: The dynamics of cells, organisms, and fluids are often modeled as probability distributions evolving over time. Reconstructing and extrapolating this evolution from unpaired snapshots requires assumptions about the underlying process. Wasserstein gradient flows are a common choice, but they cannot describe conservative or periodic dynamics. Lagrangian mechanics in Wasserstein space covers both, but existing methods for learning it are simulation-based: they run a numerical solver at every training step, which makes training expensive. We propose Double-Stitch, a simulation-free method that learns these mechanics by penalizing the residual of the equation of motion along a learned population path. We derive this equation from a Clebsch variational principle that does not require gradient velocities, and show that the residual vanishes exactly when the equation holds. We test Double-Stitch on synthetic, single-cell and ocean vortex dataset
    
[^6]: 规划以学习

    Planning to Learn

    [https://arxiv.org/abs/2610.03667](https://arxiv.org/abs/2610.03667)

    该论文揭示了精确策略梯度输给交叉熵的根本原因在于其短视性——只看重即时收益而忽视每次更新对未来学习的奠基作用，并将交叉熵重新诠释为“耐心的准确率”（考虑未来学习收益的长时域误差总量），从而通过在剩余学习量处截断该总量来改进优化方法。

    

    策略梯度方法是现代强化学习的核心，也广泛应用于大语言模型的后训练。当它们表现不佳时，人们通常将其归咎于探索、信用分配和动作采样噪声。而分类问题则完全不存在这些问题。分类器可以看作一个策略，其期望奖励（即期望准确率）就是它分配给正确标签的概率；由于正确标签是已知的，策略梯度是精确且平滑的。然而，即使以期望准确率来衡量，精确策略梯度仍然输给交叉熵。其原因在于精确梯度是短视的：它仅根据更新当前所能带来的收益来评估其价值，但每一次更新同时也决定了下一次更新的起点，因此一次更新的真正价值取决于还剩下多少学习空间。从这个视角来看，交叉熵是一种“耐心的准确率”——即若一个样本的对数几率以单位速度永久增长，它所对应支付的总误差；而精确策略梯度则是这一总量的零时域极限。将这一总量在剩余学习量处截断，便得到了……（原文摘要在此处截断）

    arXiv:2610.03667v1 Announce Type: new  Abstract: Policy-gradient methods are central to modern reinforcement learning, including LLM post-training. When they struggle, the usual suspects are exploration, credit assignment and action-sampling noise. Classification has none of them. A classifier is a policy whose expected reward, its \emph{expected accuracy}, is the probability it assigns to the correct label, and because that label is known, the policy gradient is exact and smooth. Yet exact policy gradient loses to cross-entropy, even on expected accuracy. The exact gradient is myopic: it values an update only by what it buys now, but each update also sets where the next one starts, so an update's value depends on how much learning remains. Viewed this way, cross-entropy is patient accuracy, the total error an example would pay if its log-odds rose at unit speed forever, while exact policy gradient is the zero-horizon limit. Truncating this total at the learning that remains yields the
    
[^7]: Pivot-SD：面向掩码扩散语言模型的高效自蒸馏

    Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models

    [https://arxiv.org/abs/2610.03665](https://arxiv.org/abs/2610.03665)

    Pivot-SD提出了一种高效的自蒸馏框架，通过信息增益指标识别去噪过程中真正塑造响应的少数关键决策（枢轴），并仅对这些高影响token进行针对性监督训练，从而解决了掩码扩散语言模型后训练中的信用分配问题。

    

    掩码扩散语言模型为复杂推理提供了一种有前景的、可与自回归模型并行竞争的替代方案。然而，它们面临一个独特的信用分配挑战：去噪过程中的少数关键决策会急剧降低剩余掩码位置的不确定性，并塑造了响应的大部分内容。大多数针对dLMs的后训练方法并未利用这一信号来决定训练哪些token：它们通常在最终文本上进行训练，或将奖励分配给整个去噪步骤，而不是挑选出塑造响应的那些单个关键决策。我们提出了Pivot-SD，这是一种高效的离线自蒸馏框架，仅对这些高影响的关键决策（枢轴，pivots）进行监督。Pivot-SD使用信息增益指标来选择枢轴，该指标衡量对剩余掩码位置不确定性的降低程度。来自成功轨迹的枢轴使用交叉熵进行训练，而来自失败轨迹的枢轴则使用tar（原文摘要在此处截断）

    arXiv:2610.03665v1 Announce Type: cross  Abstract: Masked diffusion language models (dLMs) offer a promising parallel alternative to autoregressive models for complex reasoning. However, they face a distinct credit-assignment challenge, since a few commitments during denoising sharply reduce the uncertainty over the remaining masked positions and shape much of the response. Most post-training recipes for dLMs do not use this signal to decide which tokens to train on: they typically train on the final text or assign rewards to whole denoising steps, rather than selecting the individual commitments that shape the response. We introduce Pivot-SD, an efficient offline self-distillation framework that supervises only these high-impact commitments (pivots). Pivot-SD selects pivots using an information-gain metric measuring uncertainty reduction over the remaining masked positions. Pivots from successful trajectories are trained with cross-entropy, and pivots from failed trajectories with tar
    
[^8]: 基于反事实模拟器推演的预测：一项Sim2Real（仿真到现实）评估

    Forecasting from Counterfactual Simulator Rollouts: A Sim2Real Evaluation

    [https://arxiv.org/abs/2610.03662](https://arxiv.org/abs/2610.03662)

    该论文提出利用反事实模拟器推演生成新决策策略下的训练数据以解决预测模型的冷启动问题，并通过两个真实库存控制部署从模拟器保真度、零样本迁移和随真实数据积累的适应性三个角度评估了Sim2Real迁移效果。

    

    部署新的决策策略会给预测模型带来冷启动问题——因为模型的预测目标依赖于该策略的行动：历史观测数据反映的是早期策略的表现，而新策略下的真实观测数据尚不可得。模拟为弥合这一差距提供了一种途径：通过在反事实场景中对目标策略进行推演，并利用由此产生的轨迹来学习系统如何响应这些控制。这个由模拟器训练的模型的仿真到现实的迁移效果，可以通过将其与过去部署中的真实观测数据进行对比评估来实施回测。利用两个真实世界的库存控制部署案例，我们从三个角度评估了这一流程：模拟器的保真度、向真实行为的零样本迁移能力，以及随着真实目标策略观测数据不断积累而进行的适应性调整。模拟器训练的预测器在点估计的平均绝对百分比误差（MAPE）上优于相同架构……

    arXiv:2610.03662v1 Announce Type: new  Abstract: Deploying a new decision policy creates a cold-start problem for prediction models whose targets depend on the policy's actions: historical observations reflect earlier policies, while real observations under the new policy are not yet available. Simulation offers a way to address this gap by rolling out the target policy across counterfactual scenarios and using the resulting trajectories to learn how the system responds to those controls. The simulation-to-reality (Sim2Real) transfer of this simulator-trained model can then be backtested by evaluating it against real observations from past deployments. Using two real-world inventory-control deployments, we evaluate this process from three angles: simulator fidelity, zero-shot transfer to real behavior, and adaptation as real target-policy observations accumulate. The simulator-trained forecaster achieves lower point-estimate mean absolute percentage error (MAPE) than the same architect
    
[^9]: PoCoFL：策略合规的联邦学习

    PoCoFL: POlicy-COmpliant Federated Learning

    [https://arxiv.org/abs/2610.03650](https://arxiv.org/abs/2610.03650)

    提出了PoCoFL框架，通过将联邦学习类型、策略语义与密码学实现三者解耦，并借助承诺和非交互式零知识证明来验证客户端贡献的策略合规性，从而实现适用于多种拓扑、角色和聚合方式的通用可验证联邦学习。

    

    联邦学习（FL）是一种面向隐私的学习范式，能够在训练数据保留在参与客户端本地的前提下进行协作式模型训练。然而，它既不能保证客户端提交的贡献符合策略要求，也不能保证聚合器正确处理已接受的贡献。现有的可验证联邦学习系统将验证规则定制于特定的联邦学习设置、学习工作流和密码学构造，这限制了它们在不同网络拓扑、参与者角色和聚合语义下的适用性。在本文中，我们提出了PoCoFL，一个策略合规的联邦学习框架，它将三个方面相互分离：（i）联邦学习类型，（ii）策略语义，（iii）密码学实现。我们提供了一种形式化方法，将客户端和聚合的要求表示为依赖于策略的关系。客户端使用承诺机制和非交互式零知识证明来证明其贡献的合规性……（摘要在此处被截断）

    arXiv:2610.03650v1 Announce Type: cross  Abstract: Federated Learning (FL) is a privacy-oriented learning paradigm that enables collaborative model training while keeping training data local to participating clients. However, it does not guarantee that clients submit policy-compliant contributions or that aggregators process admitted contributions correctly. Existing verifiable FL systems tailor validation rules to specific FL settings, learning workflows, and cryptographic constructions, limiting their applicability across network topologies, participant roles, and aggregation semantics. In this paper, we present PoCoFL, a policy-compliant federated learning framework that separates three aspects: (i) FL type, (ii) policy semantics, and (iii) cryptographic realisation. We provide a formalisation that captures client and aggregation requirements as policy-dependent relations. Clients prove compliance of their contributions using commitments and non-interactive zero-knowledge proofs, wh
    
[^10]: 面向高效海洋环境监测的星载异常检测

    On-Board Anomaly Detection for Efficient Marine Environmental Monitoring

    [https://arxiv.org/abs/2610.03649](https://arxiv.org/abs/2610.03649)

    该论文提出了一种用于对地观测卫星的轻量级海洋环境异常检测流程，利用自监督神经网络编码器压缩卫星图像并结合机器学习异常检测模型，实现高效的星载海洋环境监测。

    

    海洋生态系统受到石油泄漏、藻类大量繁殖和泥沙洪水等多种威胁的影响，这些威胁扰乱了栖息地、野生动物和人类活动。卫星图像和人工智能（AI）技术的进步增强了我们对此类危害进行早期检测和缓解的能力。在本文中，我们提出了一种面向配备多光谱或高光谱传感器的对地观测卫星的海洋事件检测流程。我们的方法包括一个自监督神经网络编码器，该编码器将卫星图像压缩到降维的潜在空间中，从而实现高效的星载处理。一个机器学习异常检测模型通过识别与正常海洋模式的偏差来检测环境异常。我们将其性能与孤立森林、单类支持向量机和局部离群因子等传统算法进行了比较。我们的轻量级、资源高效的流程针对星载部署进行了优化。

    arXiv:2610.03649v1 Announce Type: cross  Abstract: Marine ecosystems are impacted by various threats such as oil spills, algal blooms, and sediment floods, which disrupt habitats, wildlife, and human activities. Advances in satellite imagery and Artificial Intelligence (AI) have enhanced our capabilities for early detection and mitigation of such hazards. In this paper, we propose a marine event detection pipeline for Earth observation satellites equipped with multi- or hyperspectral sensors. Our approach includes a self-supervised neural network encoder that compresses satellite images into a reduced latent space, enabling efficient onboard processing. A machine learning anomaly detection model identifies deviations from normal sea patterns to detect environmental anomalies. We compare its performance against traditional algorithms such as Isolation Forest, One-Class Support Vector Machine and Local Outlier Factors. Our lightweight, resource-efficient pipeline is optimized for deploym
    
[^11]: 面向高斯过程潜变量模型的摊销结构化随机变分推断

    Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2610.03647](https://arxiv.org/abs/2610.03647)

    本文将摊销结构化随机变分推断应用于高斯过程潜变量模型，通过让潜空间的变分后验条件依赖于诱导点的取值，突破了平均场变分近似的局限，从而在数据流形重建的多项指标上取得了改进。

    

    许多机器学习方法旨在逼近数据所在的低维流形。这类方法的一个理想特性是能够捕捉所学习流形的认知不确定性。高斯过程潜变量模型就是实现这一目标的模型之一，其中从潜空间出发的高斯过程（GP）映射提供了对流形不确定性的估计。然而，这种不确定性估计的有效性受到GP诱导点与潜变量之间平均场变分近似的限制。在本工作中，我们应用摊销结构化随机变分推断，使潜空间的变分后验可以有条件地依赖于诱导点的取值。我们证明了这种更灵活的变分后验在与数据流形上点的重建相关的多项指标上均有所提升。

    arXiv:2610.03647v1 Announce Type: cross  Abstract: Many machine learning methods aim to approximate the lower-dimensional manifold on which the data lives. A desirable feature of such methods is that they should capture the epistemic uncertainty of this learned manifold. One model that achieves this is the Gaussian Process Latent Variable Model, in which a Gaussian Process (GP) mapping from the latent space provides an estimate of the uncertainty of the manifold. However, the effectiveness of this uncertainty estimation is limited by the mean-field variational approximation between the GP inducing points and the latent variables. In this work, we apply Amortized Structured Stochastic Variational Inference to allow the variational posterior for the latent space to be conditionally dependent on the value of the inducing points. We demonstrate that this more flexible variational posterior improves several metrics relating to the reconstruction of points on the data manifold.
    
[^12]: 老虎机何时可以离开它的锚点？非平稳性下由E-过程授权的汤普森采样

    When May a Bandit Leave Its Anchor? E-Process-Authorized Thompson Sampling under Non-stationarity

    [https://arxiv.org/abs/2610.03646](https://arxiv.org/abs/2610.03646)

    提出e-过程授权的汤普森采样（e-ATS），通过随时有效的e-过程决定非平稳环境中每只手臂何时从全历史锚点切换到折扣遗忘状态，保证平稳条件下偏离乐观汤普森采样的概率不超过设定的 $\alpha_E$，实验表明证据控制的是适应何时开始而非其是否总有益。

    

    平稳性会奖励记忆，但在发生变化之后，同样的历史可能会产生误导。我们探讨何时应当允许遗忘。E-过程授权的汤普森采样（e-ATS）为每只手臂同时维护全历史与折扣的Beta状态。一个随时有效的e-过程首先对折扣状态进行授权，随后一个可逆的相关性分数控制其影响程度。在获得授权之前，e-ATS完全遵循乐观汤普森采样（OTS）。在Beta-伯努利先验预测的平稳模型下，e-ATS偏离OTS的概率至多为所设定的 $\alpha_E$，无需拟合任何阈值。相对于e-ATS，移除授权机制使注册实验套件上的平均归一化动态伪遗憾增加了38.4%，但在文献衍生的重放套件上却降低了7.5%。因此，证据控制的是适应何时开始，而不是它是否总是有益的。

    arXiv:2610.03646v1 Announce Type: new  Abstract: Stationarity rewards memory, but after a change the same history can mislead. We ask when forgetting should be permitted. E-process-authorized Thompson sampling (e-ATS) gives each arm full-history and discounted Beta states. An anytime-valid e-process first authorizes the discounted state, then a reversible relevance score controls its influence. Before authorization, e-ATS exactly follows optimistic Thompson sampling (OTS). Under a Beta-Bernoulli prior-predictive stationary model, e-ATS's probability of ever departing from OTS is at most the chosen $\alpha_E$, without fitted thresholds. Relative to e-ATS, removing authorization increased mean normalized dynamic pseudo-regret by $38.4\%$ on the registered suite but reduced it by $7.5\%$ on the literature-derived replay suite. Therefore, evidence controls when adaptation begins, not whether it always helps.
    
[^13]: 论策略优化中成功条件化的收敛性

    On the Convergence of Success Conditioning for Policy Optimization

    [https://arxiv.org/abs/2610.03642](https://arxiv.org/abs/2610.03642)

    本文证明了成功条件化方法在广泛的一类马尔可夫决策过程上收敛于最优策略，并针对折扣MDP和单周期MDP分别给出了 $\mathcal{O}(1/\varepsilon^p)$ 和 $\mathcal{O}(\log(1/\varepsilon))$ 的收敛速率。

    

    成功条件化是一种在随机环境中改进决策策略的方法；它通过提高采取能带来成功结果的动作的概率来更新策略。成功条件化在许多强化学习应用中十分常见，但其极限行为和收敛速率尚未被很好地理解。在本工作中，我们证明了成功条件化在广泛的一类马尔可夫决策过程（MDP）上会收敛到最优策略。我们还在若干常见设置下推导出了收敛速率：对于折扣MDP，我们证明了在 $\mathcal{O}(1/\varepsilon^p)$ 次迭代内收敛到 $\varepsilon$-最优策略，其中指数 $p$ 取决于问题的数据；对于单周期MDP，则在 $\mathcal{O}(\log(1/\varepsilon))$ 次迭代内即可获得这样的策略。

    arXiv:2610.03642v1 Announce Type: new  Abstract: Success conditioning is a strategy for improving decision-making policies in stochastic environments; it updates a policy by increasing the probability of taking actions that yield successful outcomes. Success conditioning is common to many reinforcement learning applications, yet its limiting behavior and convergence rates are not well understood. In this work, we demonstrate that success conditioning converges to an optimal policy on a broad class of Markov decision processes (MDPs). We also derive convergence rates in some common settings. For discounted MDPs, we prove convergence within $\mathcal{O}(1/\varepsilon^p)$ iterations to an $\varepsilon$-optimal policy, where the exponent $p$ depends on problem data. For single-period MDPs, such a policy is obtained within $\mathcal{O}(\log(1/\varepsilon))$ iterations.
    
[^14]: IDRF：掩码离散扩散模型的逆蒸馏奖励微调

    IDRF: Inverse-Distilled Reward Fine-tuning of Masked Discrete Diffusion Models

    [https://arxiv.org/abs/2610.03641](https://arxiv.org/abs/2610.03641)

    IDRF用逆蒸馏正则化替代难以处理的序列级KL惩罚，并结合截断策略梯度优化，实现了少步掩码离散扩散生成器的奖励微调。

    

    掩码离散扩散模型为自回归生成提供了一种有前景的替代方案，但迭代采样可能成本高昂，且难以处理的序列似然使奖励微调变得复杂。我们提出了IDRF，一个用于少步掩码离散扩散生成器奖励微调的框架。从标准的反向KL正则化目标出发，IDRF用逆蒸馏正则化替代了难以处理的序列级KL惩罚。借助最优的辅助去噪器，我们证明了总体逆蒸馏损失是对参考分布的序列级KL散度的上界。IDRF优化该损失的基于轨迹的代理目标，且无需参考模型的rollout采样，因此学生模型可保留其自身的少步采样器。我们将少步生成视为有限时域马尔可夫决策过程，并通过在学生模型轨迹上的截断策略梯度目标来优化奖励。在DNA、图像等多个领域……

    arXiv:2610.03641v1 Announce Type: new  Abstract: Masked discrete diffusion models offer a promising alternative to autoregressive generation, but iterative sampling can be costly, and intractable sequence likelihoods complicate reward fine-tuning. We introduce IDRF, a framework for reward fine-tuning of few-step masked discrete diffusion generators. Starting from a standard reverse-KL-regularized objective, IDRF replaces the intractable sequence-level KL penalty with inverse-distillation regularization. With an optimal auxiliary denoiser, we prove that the population inverse-distillation loss upper-bounds the sequence-level KL divergence to the reference distribution. IDRF optimizes a trajectory-based surrogate of this loss without reference-model rollouts, so the student keeps its own few-step sampler. We view few-step generation as a finite-horizon Markov decision process and optimize reward with a clipped policy-gradient objective over the student's trajectories. Across DNA, image, 
    
[^15]: 欠完备线性自编码器中被破坏的尺度对称性

    Broken scale symmetries in undercomplete linear autoencoders

    [https://arxiv.org/abs/2610.03640](https://arxiv.org/abs/2610.03640)

    该论文发现，在欠完备线性自编码器中，有限步长的SGD会以有向的方式破坏尺度对称性，在PCA解流形上系统性地偏向放大解码器权重，直至动力学触及有限步长的稳定性边界。

    

    神经网络的损失景观具有许多对称性，这些对称性在梯度流下得以保持，但在有限步长的随机梯度下降（SGD）中会被破坏。这类对称性的一个典型例子是同质网络中的尺度对称性：可以将某一层的参数放大，同时将下一层的参数缩小，而不会改变网络的输出。先前的工作记录了SGD为平衡梯度噪声或最小化波动而破坏这种对称性的若干案例。在本研究中，我们证明欠完备线性自编码器的解几何结构反而为尺度漂移选择了一个优先的方向：在PCA解流形上，SGD偏向较大的解码器权重。这种有向的尺度漂移发生在缓慢的时间尺度上，其动力学可以用解析可处理的有效描述来刻画。然而，这种漂移无法无限持续下去：尺度的不断增大最终会将动力学驱向有限步长的稳定性边界。由此产生的解是最尖…（摘要在此处被截断）

    arXiv:2610.03640v1 Announce Type: new  Abstract: Neural network loss landscapes have many symmetries, which are preserved by gradient flow but broken by finite-stepsize stochastic gradient descent (SGD). A canonical example of such a symmetry is scale in homogeneous networks: one can scale up the parameters in one layer and down in the next without changing the network output. Previous work has documented cases in which SGD breaks this symmetry in favor of balancing gradient noise or minimizing fluctuations. Here, we show that the solution geometry of undercomplete linear autoencoders instead selects a preferred sign for scale drift: on the PCA solution manifold, SGD favors large decoder weights. This directed scale drift occurs on a slow timescale, and its dynamics admit an analytically-tractable effective description. However, it cannot continue indefinitely: increasing scale eventually drives the dynamics towards a finite-stepsize stability boundary. The resulting solutions are shar
    
[^16]: FALCON：一种模型与数据集无关的NL2SQL对合成数据生成框架

    FALCON: A Model and Dataset Agnostic Framework for Synthetic Data Generation for NL2SQL Pairs

    [https://arxiv.org/abs/2610.03625](https://arxiv.org/abs/2610.03625)

    FALCON提出了一种模型与数据集无关的框架，通过保留字SQL种子、基于人设的提示生成和基于对齐的过滤，以低成本利用紧凑开源模型生成真实、具有歧义感知能力且结构复杂的NL-to-SQL合成数据。

    

    关系数据库是部署最广泛的结构化知识形式之一，通过自然语言访问这些数据库需要将语言落实到模式实体和关系上，同时还要处理人们在表述请求时固有的歧义性。现有的合成NL-to-SQL数据生成方法大多忽略了这种歧义性，生成过于简化的查询，无法让模型为现实世界结构化知识访问的复杂性做好准备。我们提出了FALCON，这是一个能够生成真实的、具有歧义感知能力的NL-to-SQL数据的框架，其生成的数据复杂度可与具有挑战性的真实世界基准相匹配，并且通过使用紧凑的开源模型以低成本实现。我们的方法结合了保留字SQL种子和基于人设的提示来生成结构复杂的查询，同时基于对齐的过滤通过区分真正错误的样本与复杂但有效的查询来保持数据难度。人工评估证实了生成数据的一致高质量。

    arXiv:2610.03625v1 Announce Type: new  Abstract: Relational databases are among the most widely deployed forms of structured knowledge, and natural language access to them requires grounding language onto schema entities and relations while handling the ambiguity inherent in how people phrase requests. Existing synthetic NL-to-SQL data generation methods largely ignore this ambiguity and produce oversimplified queries that fail to prepare models for the complexity of real-world structured knowledge access. We present FALCON, a framework that generates realistic, ambiguity-aware NL-to-SQL data matching the complexity of challenging real-world benchmarks, at low cost using compact open models. Our approach combines reserved-word SQL seeding and persona-based prompting to generate structurally complex queries, while alignment-based filtering preserves difficulty by distinguishing genuinely incorrect examples from complex but valid queries. Human evaluation confirms consistent high quality
    
[^17]: 马尔可夫博弈中的标准形式相关均衡

    Normal-Form Correlation in Markov Games

    [https://arxiv.org/abs/2610.03621](https://arxiv.org/abs/2610.03621)

    本文首次为玩家数量固定的有限视界马尔可夫博弈中的标准形式相关均衡（NFCE）提出了高效计算算法，并在推荐跨状态独立的假设下证明了其 PPAD-完全性。

    

    近年来，关于马尔可夫博弈中相关均衡概念的研究涌现出大量成果。然而，现有结果主要聚焦于比标准形式相关均衡（NFCE）更弱的概念，而计算此类均衡这一更具挑战性的问题仍未解决，该问题可以追溯到 Papadimitriou 和 Roughgarden 的开创性工作（JACM'08）。本文为玩家数量固定为 $n$ 的有限视界马尔可夫博弈中的 NFCE 建立了首个高效算法。具体而言，在状态数为 $S$、视界为 $H$、每个玩家至多有 $A$ 个动作的设定下，该算法可在时间 $S(AH/\epsilon)^{O(n)}$ 内计算出一个 $\epsilon$-NFCE。这是首个在标准形式博弈设定之外的一类有趣问题上、关于 $1/\epsilon$ 和博弈描述规模均为多项式时间的 NFCE 算法。此外，在各状态间推荐相互独立的通常假设下，我们证明了该问题是 PPAD-完全的——即在计算复杂度上等价于纳什均衡——

    arXiv:2610.03621v1 Announce Type: cross  Abstract: There has been a surge of recent work on correlated equilibrium concepts in Markov games. However, existing results focus on concepts weaker than normal-form correlated equilibria (NFCEs), leaving open the more challenging question of computing such equilibria, which goes back to the seminal work of Papadimitriou and Roughgarden (JACM'08). Here, we establish the first efficient algorithm for NFCEs in finite-horizon Markov games with a fixed number of players $n$. In particular, with $S$ states, horizon $H$, and at most $A$ actions per player, it computes an $\epsilon$-NFCE in time $S(AH/\epsilon)^{O(n)}$. This is the first algorithm polynomial in $1/\epsilon$ and the description of the game for NFCEs in an interesting class of problems beyond the normal-form setting. Moreover, under the usual assumption that recommendations are independent across states, we show PPAD-completeness---that is, computational equivalence to Nash equilibria-
    
[^18]: UniIntervene++：用于高效真实世界强化学习的自适应干预智能体

    UniIntervene++: An Adaptive Intervention Agent for Efficient Real-World Reinforcement Learning

    [https://arxiv.org/abs/2610.03620](https://arxiv.org/abs/2610.03620)

    UniIntervene++是一个自适应干预智能体，通过在统一半马尔可夫决策过程中在线学习RL策略与多种辅助行为之间的相对价值，并根据策略当前能力动态调整控制分配，从而实现高效的真实世界机器人在线强化学习。

    

    在线强化学习（RL）使机器人策略能够通过物理交互不断改进，但它们所需的辅助会随着自身能力的演进而变化。因此，现有的基于离线估计或固定决策规则的干预策略可能与当前策略不再匹配。为了解决这一问题，我们提出了UniIntervene++，一个自适应干预智能体，它能够在在线强化学习过程中学会在自主执行与异构辅助行为之间分配控制权。具体而言，UniIntervene++首先将不断演进的RL策略、轨迹修正以及任务结构化的CodePolicy统一表述为半马尔可夫决策过程中的选项，并在线学习它们的相对价值。在此基础上，能力自适应干预通过无辅助执行定期探测RL策略，使控制分配能够随其不断演进的能力而动态调整。最后，耦合经验学习允许辅助……（摘要在此处截断）

    arXiv:2610.03620v1 Announce Type: new  Abstract: Online reinforcement learning (RL) enables robot policies to improve through physical interaction, but the assistance they require changes as their competence evolves. Existing intervention strategies based on offline estimates or fixed decision rules can therefore become mismatched to the current policy. To address this, we propose UniIntervene++, an adaptive intervention agent that learns to allocate control between autonomous execution and heterogeneous assisted behaviors during online RL. Specifically, UniIntervene++ first formulates the evolving RL policy, trajectory correction, and a task-structured CodePolicy as Options in a unified semi-Markov decision process and learns their relative values online. Building on this, competence-adaptive intervention periodically probes the RL policy through unassisted execution, keeping control allocation responsive to its evolving capability. Finally, coupled experience learning allows assisted
    
[^19]: 通过发现的选项掌握Atari 2600游戏

    Mastering Atari 2600 Games with Discovered Options

    [https://arxiv.org/abs/2610.03604](https://arxiv.org/abs/2610.03604)

    提出了Wayfarer，一个通用的、领域无关的深度强化学习智能体，它通过拉普拉斯表示学习从高维观测中发现选项，这些选项同时改善了探索、加速了信用分配并有效泛化到未见场景，从而在Atari 2600游戏上实现大幅加速的学习。

    

    时间抽象（通常以选项的形式实例化）长期以来被视为强化学习（RL）中加速信用分配、促进探索和实现泛化的一种机制。然而，开发在大规模、高维领域中有效的一般性选项发现方法仍然是一项根本性的挑战。现有的选项发现方法要么局限于相对简单的领域，要么依赖手工设计或准符号化的表示，要么与不使用选项的学习相比几乎没有改进。我们提出了Wayfarer，这是一个通用的、领域无关的、在线的深度强化学习智能体，它通过拉普拉斯表示学习从高维观测中发现选项，并利用这些选项进行控制。我们证明了由此产生的选项能够同时改善探索、加速信用分配，并有效地泛化到未见过的设置，从而实现显著更快的学习。

    arXiv:2610.03604v1 Announce Type: new  Abstract: Temporal abstractions, often instantiated as options, have long been regarded as a mechanism for accelerating credit assignment, facilitating exploration, and enabling generalisation in reinforcement learning (RL). However, developing general option discovery methods that are effective in large-scale, high-dimensional domains remains a fundamental challenge. Existing option discovery methods are either confined to relatively simple domains, depend on handcrafted or quasi-symbolic representations, or offer little improvement over learning without options. We present Wayfarer, a general, domain-agnostic, online deep RL agent that discovers options through Laplacian representation learning from high-dimensional observations and leverages them for control. We show that the resulting options simultaneously improve exploration, accelerate credit assignment, and generalise effectively to unseen settings, enabling substantially faster learning o
    
[^20]: 联邦学习中多步梯度反演的路径积分代理模型

    A Path Integral Surrogate for Multi-Step Gradient Inversion in Federated Learning

    [https://arxiv.org/abs/2610.03597](https://arxiv.org/abs/2610.03597)

    该论文提出PI-SME方法，将FedAvg下客户端的累积更新视为梯度场沿权重轨迹的路径积分并用高斯-勒让德求积近似，从而增强多步梯度反演攻击从单次模型更新中重建客户端私有图像的能力。

    

    联邦学习让众多客户端无需将私有数据发送到中央服务器即可共同训练一个共享模型。每个客户端仅共享模型更新，而该更新所泄露的客户端信息应远少于其原始训练样本，这一前提正是保护客户端隐私的基础。梯度反演攻击直接挑战了这一前提，它试图从客户端共享的单次更新中重建其私有的输入图像。在FedAvg算法下，客户端的更新累积了多个本地训练步骤，因此服务器只能看到隐藏权重轨迹的两个端点。近期的梯度反演攻击会沿着这两个端点之间的路径拟合一个代理模型，但它们仍然只在单个点上读取该模型的梯度。我们提出了路径积分代理模型扩展（PI-SME），它将累积更新视为梯度场的路径积分，并通过高斯-勒让德（Gauss–Legendre）求积对该路径上的积分进行近似。

    arXiv:2610.03597v1 Announce Type: new  Abstract: Federated learning lets many clients train a shared model together without ever sending their private data to a central server. Each client shares only a model update, and this update should reveal far less about the client than its raw training examples would. This premise is what protects the privacy of the clients. Gradient inversion attacks challenge it directly by trying to reconstruct a client's private input images from the single update it shared. Under FedAvg, a client's update accumulates several local training steps, so the server sees only the two endpoints of a hidden weight trajectory. Recent gradient inversion attacks fit a surrogate model along the path between these two endpoints but they still read its gradient at a single point. We propose the Path-Integral Surrogate Model Extension (PI-SME) which treats the accumulated update as a path integral of the gradient field and approximates it by Gauss--Legendre quadrature ov
    
[^21]: 智能体安全基准中的威胁保持表征敏感性

    Threat-Preserving Representation Sensitivity in Agent-Security Benchmarks

    [https://arxiv.org/abs/2610.03585](https://arxiv.org/abs/2610.03585)

    论文提出威胁保持表征敏感性（TPRS）指标，发现在底层威胁完全不变的情况下，仅改变智能体可见的表征（如工具名称）就能使攻击成功率显著变化（最高约13个百分点），表明智能体安全基准的测量结果对表征方式高度敏感，可能无法真实反映模型的安全性。

    

    arXiv:2610.03585v1 公告类型：交叉。摘要：针对基于大语言模型（LLM）的智能体的安全基准测试通常报告攻击成功率（ASR）作为衡量模型鲁棒性的指标，并利用这些分数来比较不同的模型和防御机制，其假设是这些分数能够描述智能体的安全性。在本文中，我们探讨基准的表征方式是否也会影响其测量结果。为了衡量基准表征的影响，我们提出了威胁保持表征敏感性（TPRS），它衡量的是在保持底层任务、有害动作、安全策略、真值、环境和评估标准不变的情况下，仅改变智能体可见的表征时，攻击成功率（ASR）的变化幅度。在 Agent Security Bench（ASB）上，将威胁相关的工具名称替换为威胁中立的名称，使 GPT-5-mini 的实际攻击成功率提高了 11.67 个百分点，使 Claude Haiku 4.5 提高了 13.21 个百分点。在 MCPTox 上，将原始的中立工具名称替换为一个利用……（摘要原文在此处截断）

    arXiv:2610.03585v1 Announce Type: cross  Abstract: Security benchmarks for LLM-based agents often report the attack success rate (ASR) as a measure of model robustness and use these scores to compare different models and defense mechanisms, assuming that they describe the security of the agent. In this paper, we explore whether it also influences the benchmark's measurement.   To measure the effect of the benchmark representation, we introduce threat-preserving representation sensitivity (TPRS), which measures how much the ASR changes when we change the agent-visible representation while holding the underlying task, harmful action, security policy, ground truth, environment, and the evaluation criteria fixed.   On Agent Security Bench (ASB), replacing threat-related tool names with threat-neutral names raises the committed attack success rate by 11.67 percentage points on GPT-5-mini and by 13.21 points on Claude Haiku 4.5. On MCPTox, replacing the original neutral tool name with an exp
    
[^22]: HyperBrowseComp：面向网页浏览智能体的多语言多模态压力测试

    HyperBrowseComp: A Multilingual and Multimodal Stress Test for Web-Browsing Agents

    [https://arxiv.org/abs/2610.03574](https://arxiv.org/abs/2610.03574)

    HyperBrowseComp是一个覆盖13种语言、包含423道人工验证问题的多语言多模态网页浏览基准，通过要求定位隐蔽证据、追踪多步线索链和检查异构信息源，为网页浏览智能体提供了极具挑战性的压力测试。

    

    我们推出了HyperBrowseComp，这是一个多语言多模态浏览基准，包含423道人工编写并经人工验证的问题，覆盖13种语言，由母语或高度熟练的使用者撰写。这些问题被设计得极具挑战性。每个问题都指向一个简洁且可公开验证的答案，而发现答案需要定位隐蔽的证据、追踪多步骤的线索链，或检查视频、扫描文档、图像、地图等异构信息源。为降低问题仅凭模型参数化知识就能被回答的可能性，我们通过使用无网络访问的模型对问题进行评估来过滤掉较简单的问题。我们在统一的智能体协议下，使用提供商原生搜索和共享的外部检索工具对多个模型进行了评估。为给模型性能与投入程度提供参照，我们还对部分问题样本进行了人类评估。HyperBrowseComp提供了一个具有挑战性的测试平台。

    arXiv:2610.03574v1 Announce Type: new  Abstract: We introduce HyperBrowseComp, a multilingual and multimodal browsing benchmark comprising 423 manually authored and human-validated questions across 13 languages, written by native or highly proficient speakers. Questions are designed to be extremely challenging. Each question targets a concise, publicly verifiable answer whose discovery requires locating obscure evidence, following multi-step clue chains, or inspecting heterogeneous sources such as videos, scanned documents, images, or maps. Easier questions are filtered out by evaluating them with models without internet access to reduce the likelihood that they can be answered with parametric knowledge alone. We evaluate several models using provider-native search and a shared external retrieval harness under a common agent protocol. To contextualize model performance and effort, we also conduct a human evaluation on a sample of the questions. HyperBrowseComp provides a challenging te
    
[^23]: Cephalonauts One：一个用于解码人脑自然语音的深度fMRI数据集

    Cephalonauts One: A deep fMRI dataset for decoding naturalistic speech in the human brain

    [https://arxiv.org/abs/2610.03558](https://arxiv.org/abs/2610.03558)

    发布了迄今最大的自然语音fMRI数据集Cephalonauts One（每名受试者30小时数据），并提出以音频片段检索为任务形式的大脑解码基准，附带标准化数据划分、评估指标和基线解码器。

    

    Cephalonauts One 是一个全脑3特斯拉（3T）功能磁共振成像（fMRI）数据集，在受试者收听音频播客时采集。三名健康受试者进行了多次扫描会话，每次会话包含五个15分钟的扫描运行，受试者在扫描期间收听其母语的播客。每名受试者拥有30小时的fMRI数据，本次发布的数据集是迄今为止使用自然语音刺激的最大规模fMRI数据集。该数据集将大脑活动与相应的播客音频、转录文本注释以及导出的刺激嵌入配对。此外，我们引入了一个以音频片段检索形式表述的大脑解码基准：给定来自保留会话的fMRI活动，解码器必须在候选片段中识别出与之时间对齐的相应播客音频片段。我们为该任务提供了标准化的数据划分、评估指标和基线解码器。最后，缩放分析表明解码性能（原文在此处截断）。

    arXiv:2610.03558v1 Announce Type: cross  Abstract: Cephalonauts One is a whole-brain 3 Tesla (3T) functional magnetic resonance imaging (fMRI) dataset recorded while subjects listened to audio podcasts. Three healthy subjects underwent multiple scanning sessions, each consisting of five 15-minute runs, while listening to podcasts in their native language. With 30 hours of fMRI data per subject, the current release is the deepest available fMRI dataset using naturalistic speech stimuli. The dataset pairs brain activity with the corresponding podcast audio, transcript annotations, and derived stimulus embeddings. Furthermore, we introduce a brain decoding benchmark formulated as audio segment retrieval: given fMRI activity from a held-out session, the decoder must identify the corresponding time-aligned podcast audio segment among candidate segments. We provide standardized splits, evaluation metrics, and baseline decoders for this task. Finally, a scaling analysis shows that decoding pe
    
[^24]: 抓紧GRIP，这将是一段漫长的TRIP：用于验证过挤压现象的可量化长程框架

    Get a GRIP, this will be a long TRIP: A Quantifiable Long-Range Framework for Verifying Over-squashing

    [https://arxiv.org/abs/2610.03556](https://arxiv.org/abs/2610.03556)

    该论文提出四个可验证公理（可预测性、紧凑性、严格k-范围、拓扑不变性），并据此构建可量化框架TRIP，为任意图上“是否真正需要长程交互”提供原则性证书，从而可信地验证GNN中的过挤压现象。

    

    关于GNN中过挤压与长程交互之间联系的经验性结论，只有在用于验证它们的基准任务真正需要长程交互时才可信。事实上的标准基准——长程图基准，已被反复证明会被经过调优的短程模型所饱和，而现有的合成替代方案则与特定拓扑结构绑定。因此，目前缺乏针对任意图的长程性的原则性证书。这种状况反映出对长程基准缺乏精确刻画。我们通过引入四个可验证的公理来填补这一根本性空白：可预测性、紧凑性、严格k-范围和拓扑不变性，任何声称测试k跳交互的任务都必须满足这些公理。我们形式化地证明，违反其中任何一个公理都会引入破坏该任务结论的失败模式。基于这些公理，我们引入了TRIP（Tru……

    arXiv:2610.03556v1 Announce Type: new  Abstract: Empirical claims about the connection between over-squashing and long-range interactions in GNNs, can only be trusted if the benchmarks used to validate them genuinely require long-range interactions. The de-facto standard, the Long Range Graph Benchmark, has been repeatedly shown to be saturated by tuned short-range models, with existing synthetic alternatives being tied to specific topologies. As such, there is a lack of principled certificate of long-rangedness on arbitrary graphs. This state reflects the absence of a precise characterization of long-ranged benchmarks. We address this fundamental gap by introducing four verifiable axioms: Predictability, Tightness, Strictly $k$-Range, and Topology-Invariance, that any task claiming to test $k$-hop interactions must satisfy. We formally prove that violating any one of them admits failure modes that undermine conclusions drawn from the task. Based on these axioms, we introduce TRIP (Tru
    
[^25]: 没有态射的对象：面向数学的大语言模型未能表示什么

    Objects Without Morphisms: What LLMs for Mathematics Do Not Represent

    [https://arxiv.org/abs/2610.03551](https://arxiv.org/abs/2610.03551)

    本文通过相邻数学子领域间的命题翻译任务，提出一种将真值、内容与范围分置于独立盲评队列、且无需评判者的测量工具，揭示了大语言模型虽借助外部筛选达到专家级解题表现，却未能表示数学内容所隐含的概括层次。

    

    大语言模型在竞赛数学上达到专家级表现，主要归功于围绕它们所构建的大规模搜索：大量采样候选解答，只有当外部标准接受时才予以保留。这样的过程只改进了在筛选中幸存下来的结果，却未触及模型究竟表示了什么。我们在不存在外部标准的场景中考察这一问题：在相邻数学子领域的不同“方言”之间翻译命题，此时翻译的忠实性取决于内容被断言时所处的一般性层次。源命题将其概括层次隐含在自身的词汇之中，因此忠实的翻译必须从理论之间的关系中将其恢复出来。我们引入了一种评估工具，在相互独立的盲评队列中分别对真值、内容和范围进行编码，并提供一种无需评判者的度量，用于衡量改写后的命题是否陈述了其源命题中隐含的假设，并通过植入阳性对照验证了该工具的敏感性。

    arXiv:2610.03551v1 Announce Type: new  Abstract: Large language models (LLMs) have reached expert-level performance on competition mathematics largely through the volume of search placed around them: candidate solutions are sampled in quantity and retained only when an external criterion accepts them. Such a procedure improves the outcome that survives it while leaving untouched what the model represents. We examine that question where no external criterion exists: translating statements between the dialects of neighbouring subfields, where fidelity turns on the level of generality at which content is asserted. The source leaves that level implicit in its vocabulary, so a faithful translation must recover it from the relation between the theories. We introduce an instrument that codes truth, content and scope in separate blind queues, with a judge-free measure of whether a rewrite states the hypothesis implicit in its source, and establish its sensitivity with a planted-positive contro
    
[^26]: ZeroMAG：面向即插即用脑电基础模型的零样本多模态适配器生成

    ZeroMAG: Zero-Shot Multimodal Adapter Generation for Plug-and-Play EEG Foundation Models

    [https://arxiv.org/abs/2610.03546](https://arxiv.org/abs/2610.03546)

    ZeroMAG提出了一种零样本多模态适配器生成框架，无需目标标签或目标端优化，即可利用无标注数据为冻结的脑电基础模型自动生成多模态适配器，实现向异构多模态记录的即插即用扩展。

    

    脑电基础模型（EFM）能够从大规模脑电数据中捕获可复用的知识，而许多脑电记录还同时包含伴随的生理信号，这些信号提供了仅基于脑电接口之外的补充信息。面临的挑战在于：如何在保持预训练知识的同时，通过从无标注目标数据中推断出的适配，将脑电基础模型扩展到异构的多模态记录上。我们提出了ZeroMAG，一个零样本多模态适配器生成框架，它利用无标注的目标记录来扩展冻结的脑电编码器和预测头，无需目标标签，也无需在目标端进行优化。在ZeroMAG的整个流程中，目标数据集完全不参与任何模型训练和选择。ZeroMAG围绕一个配置不变的适配器来组织伴随模态，从无标注记录和任务上下文构建模态-受试者-任务条件，并在函数约束的隐空间中生成适配器权重

    arXiv:2610.03546v1 Announce Type: new  Abstract: EEG foundation models (EFMs) capture reusable knowledge from large-scale EEG data, while many EEG recordings also include companion physiological signals that provide complementary information beyond the EEG-only interface. The challenge is to preserve this pretrained knowledge while extending the EFM to heterogeneous multimodal recordings through an adaptation inferred from unlabeled target data. We introduce ZeroMAG, a zero-shot multimodal adapter generation framework that extends a frozen EEG encoder and prediction head using unlabeled target recordings, without target labels or target-side optimization. The target datasets are held out from all model training and selection in the ZeroMAG pipeline. ZeroMAG organizes companion modalities around a configuration-invariant adapter, constructs a modality-subject-task condition from unlabeled recordings and task context, and generates adapter weights in a function-constrained latent space l
    
[^27]: 用于血管内脑机接口接入的自主机器人导航

    Autonomous Robotic Navigation for Endovascular Brain-Computer Interface Access

    [https://arxiv.org/abs/2610.03537](https://arxiv.org/abs/2610.03537)

    这项工作首次展示了脑静脉系统中血管内脑机接口接入的体外自主机器人导航，利用强化学习控制器实现了从颈内静脉到上矢状窦的精确导航，并在未见过的解剖结构上验证了泛化能力。

    

    血管内脑机接口（BCI）避免了开颅手术，但需要通过解剖结构多变的脑静脉精确输送器械。这项工作展示了首个在脑静脉系统中实现血管内脑机接口接入的体外自主机器人导航演示。研究人员在计算机仿真环境中训练了Soft Actor-Critic（软演员-评论家）控制器，用于完成从右侧颈内静脉到上矢状窦的两个连续任务，并采用单一训练解剖结构的几何增强方法。导航性能在训练解剖结构和解剖学上未见过的保留模型中进行了评估，每个任务-解剖条件组合包含250次计算机仿真和5次荧光透视引导的体外机器人运行，总计1,000次模拟运行和20次物理运行。研究还评估了任务循环预测器用于在线识别即将发生的导航失败。在计算机仿真中，任务A和任务B在训练解剖结构中的成功率分别为85.6%和98.4%。

    arXiv:2610.03537v1 Announce Type: cross  Abstract: Endovascular brain-computer interfaces (BCIs) avoid craniotomy but require precise device delivery through anatomically variable cerebral veins. This work presents the first demonstration of in vitro autonomous robotic navigation for endovascular BCI access in the cerebral venous system. Soft Actor-Critic controllers were trained in silico for two sequential tasks spanning the right internal jugular vein to the superior sagittal sinus, using geometric augmentation of one training anatomy. Navigation was evaluated in a training anatomy and an anatomically unseen hold-out model over 250 in silico episodes and five fluoroscopy-guided in vitro robotic runs per task-anatomy condition, comprising 1,000 simulated episodes and 20 physical runs overall. Task recurrent predictors were also evaluated for online identification of impending navigation failure. In silico success rates for Tasks A and B were 85.6% and 98.4% in the training anatomy an
    
[^28]: 蒸馏中的散度控制熵

    Divergence controls entropy in distillation

    [https://arxiv.org/abs/2610.03529](https://arxiv.org/abs/2610.03529)

    该论文证明蒸馏目标中散度的选择充当了隐式熵正则化器，控制着学生模型的熵——前向KL散度会使学生熵高于教师，反向KL散度会降低熵，而在线策略蒸馏的低熵来自token级反向KL散度而非采样方式本身。

    

    蒸馏已成为大语言模型训练的核心基础操作，但其性质尚未被充分理解。我们从熵的视角出发，研究学生的熵如何取决于数据以及定义蒸馏目标的散度。我们证明前向KL散度会使学生的熵膨胀到高于教师的水平。由于交叉熵训练是其特例，这给出了一个恒等式，我们在预训练和监督微调中对其进行了定量验证。其他散度则没有这样的保证：反向KL散度会降低熵，直到学生与教师之间的差距过大为止；在两者之间插值时，熵在训练早期平滑变化，但在收敛时发生突变。在线策略蒸馏所具有的较低熵来源于token级的反向KL散度，而非在线策略采样本身。因此，散度充当了一种隐式的熵正则化器，其作用在自……（原文在此处截断）

    arXiv:2610.03529v1 Announce Type: cross  Abstract: Distillation has become a core primitive of large language model training, but its properties are not yet well understood. We take an entropic perspective, studying how the entropy of the student depends on the data and the divergence that define the distillation objective. We prove that forward KL inflates the entropy of the student above that of the teacher. Since cross-entropy training is a special case, this yields an identity that we verify quantitatively in pretraining and supervised finetuning. Other divergences come with no such guarantee: reverse KL deflates entropy until the gap between student and teacher gets too large, and interpolating between the two changes entropy smoothly early in training but abruptly at convergence. The lower entropy of on-policy distillation comes from token-level reverse KL, not from on-policy sampling. The divergence therefore acts as an implicit entropy regularizer, whose role is clearest in sel
    
[^29]: 超越训练模型：为健全的GNN可解释器基准而编译

    Beyond Trained Models: Compiling GNNs for a Sound Explainer Benchmark

    [https://arxiv.org/abs/2610.03526](https://arxiv.org/abs/2610.03526)

    该论文揭示了现有GNN可解释器基准中“训练模型依赖预期模体”这一隐含假设的不成立，并提出Gracr——首个将分级模态逻辑公式编译为GNN权重的编译器，通过用编译替代训练构建了真实解释可被形式化定义并精确计算的健全可解释器基准。

    

    图神经网络（GNN）的可解释器通常通过其合理性（plausibility）来评估，即其解释能在多大程度上恢复预先定义的真实标准（ground truth），例如植入数据中的模体（motif）。这一评估协议隐含地假设：在此类数据上训练的GNN依赖于预期的模体。尽管先前的工作已经质疑过这一假设，合理性评估仍然被广泛使用。首先，我们证明该假设在多个广泛使用的基准上并不成立，例如，仅凭度统计信息就足以解决这些任务。随后，我们通过用编译替代训练来消除这一混淆因素。为此，我们引入了Gracr——首个将分级模态逻辑公式编译为GNN权重的编译器，由此得到的模型能够复现相应公式的行为。由于模型的行为在构造时就已知晓，我们可以形式化地定义其真实解释并对其进行精确计算。基于此……

    arXiv:2610.03526v1 Announce Type: cross  Abstract: Explainers for Graph Neural Networks (GNNs) are commonly evaluated by their plausibility, i.e., how well their explanations recover a predefined ground truth, such as a motif planted in the data. This protocol implicitly assumes that a GNN trained on such data relies on the intended motif. Although prior work has questioned this assumption, plausibility remains widespread. First, we show that the assumption is violated on several widely used benchmarks, where, e.g., degree statistics alone suffice to solve the task. Then, we remove this confounder by replacing training with compilation. We achieve this by introducing $\mathsf{Gracr}$, the first compiler translating graded modal logic formulas into GNN weights, yielding models that replicate the behaviour of the corresponding formulas. Since the behaviour of the model is now known by construction, we can define its ground truth explanation formally and compute it exactly. Building on th
    
[^30]: 从基准测试到生产环境：面向复杂金融数据的Text-to-SQL系统

    From Benchmarks to Production: A Text-to-SQL System for Complex Financial Data

    [https://arxiv.org/abs/2610.03524](https://arxiv.org/abs/2610.03524)

    FLINT是一个针对生产环境金融数据库的领域专用Text-to-SQL系统，通过查找代理解析不透明概念、嵌入检索专家模板以及基于外键链遍历的模式链接三大组件，解决了通用系统在此类复杂数据上准确率低于50%的问题。

    

    通用型Text-to-SQL系统在Spider和BIRD等学术基准测试上表现出色，这些基准中的数据库模式相对浅层，列值通常是人类可读的。而在生产环境的金融数据库中，概念以不透明的整数键而非人类可读字符串的形式存储，这些方法的准确率降至50%以下，因为即使是简单的查询也需要多次连接操作，且过滤谓词引用的是不透明的ID。我们提出了Financial LINking Text-to-SQL（FLINT），这是一个领域专用的Text-to-SQL系统，通过三个关键组件弥合了这一差距：(1) 一个查找代理，可动态地将自然语言概念解析为针对特定问题的参考表约束；(2) 基于嵌入的检索，从一个紧凑的、由专家编写的模板库中检索结构相似的查询模板；(3) 模式链接机制，通过遍历外键链将大型表模式修剪为相关子集，而非依赖名称相似度。

    arXiv:2610.03524v1 Announce Type: new  Abstract: General-purpose Text-to-SQL systems achieve strong performance on academic benchmarks like Spider and BIRD, where schemas are relatively shallow and column values are often human readable. In production financial databases, where concepts are stored as opaque integer keys rather than human-readable strings, these methods fall below 50%, as even simple queries require multiple joins and filter predicates reference opaque IDs. We present Financial LINking Text-to-SQL (FLINT), a domain-specialized Text-to-SQL system that closes this gap through three key components: (1) a lookup agent that dynamically resolves natural-language concepts to question-specific reference table constraints, (2) embedding-based retrieval of structurally similar query templates from a compact, expert-authored bank, and (3) schema linking that prunes a large table schema to the relevant subset by traversing foreign-key chains, rather than relying on name similarity 
    
[^31]: XGenAct：通过跨任务生成的几何增强世界动作模型

    XGenAct: Geometry-Enhanced World Action Models through Cross-Task Generation

    [https://arxiv.org/abs/2610.03516](https://arxiv.org/abs/2610.03516)

    XGenAct提出一种几何增强的世界动作模型，将RGB观测、机器人动作、度量深度、表面法线和功能角色分割统一表示为RGB视频，用单一视频扩散Transformer和统一目标学习跨模态的时间预测，从而增强机器人操作所需的空间理解能力。

    

    世界动作模型（WAMs）通过预测观测和动作随时间的演变方式，推动了机器人控制的发展。尽管取得了这些进展，基于RGB和动作的未来预测并未明确解决机器人操作所需的空间理解能力。现有的工作通常通过专门的预测头或分支添加有限的一组空间预测任务，导致空间监督的范围和模型架构都较为碎片化。我们提出了XGenAct，这是一种世界动作模型，它通过确定性编解码器将RGB观测、机器人动作、度量深度、表面法线和功能角色分割统一表示为RGB视频。通过在训练期间采样感知和动作任务，XGenAct使用一个视频扩散Transformer和一个目标函数来学习跨这些空间的时间预测，而无需针对特定模态的学习头。在保留的RLBench任务上，结构化感知训练提高了平均闭环性能……（摘要在此处被截断）

    arXiv:2610.03516v1 Announce Type: cross  Abstract: World action models (WAMs) have advanced robot control by predicting how observations and actions evolve over time. Despite this progress, RGB and action based future prediction does not explicitly address the spatial understanding needed for robot manipulation. Existing efforts often add a limited set of spatial prediction tasks through specialized heads or branches, leaving both the range of spatial supervision and the model architecture fragmented. We introduce XGenAct, a world action model that represents RGB observations, robot actions, metric depth, surface normals, and functional role segmentation as RGB videos through deterministic codecs. By sampling perception and action tasks during training, XGenAct uses one video diffusion transformer and one objective to learn temporal prediction across these spaces without modality specific learned heads. On held out RLBench tasks, structured perception training improves average closed l
    
[^32]: 聚变材料裂纹识别与损伤评估的自动化可复现工作流程

    An Automated and Reproducible Workflow for Crack Identification and Damage Assessment of Fusion Materials

    [https://arxiv.org/abs/2610.03505](https://arxiv.org/abs/2610.03505)

    该论文在Galaxy科学工作流环境中实现了一个自动化可复现的裂纹识别与损伤评估流程，无需图像特定参数调优即可处理不同钨材料、微观结构、放大倍数和损伤状态的扫描电镜图像。

    

    辐照后显微分析是聚变材料合格评定的核心环节。然而，人工分析无法适应现代聚变材料研究活动中数据量大、异质性强和多分辨率的特点。为应对这一挑战，我们提出了一种在Galaxy科学工作流环境中实现的可复现工作流程，用于从扫描电子显微镜图像中自动识别裂纹并进行定量损伤评估。该工作流程处理SEM图像和实验元数据以识别裂纹、量化损伤，并保留复现所需的中间产物和处理历史。输出内容包括裂纹掩膜、骨架化裂纹网络、质量控制可视化以及标量损伤描述符。该方法旨在无需针对具体图像进行参数调优的情况下，适用于不同牌号的钨材料、不同微观结构、不同放大倍数和不同损伤状态。我们在稀疏电子-（摘要原文在此处被截断）

    arXiv:2610.03505v1 Announce Type: new  Abstract: Post-exposure microscopy is central to qualification of fusion materials. However, manual analysis does not scale to the volume, heterogeneity, and multiresolution character of modern fusion-materials campaigns. To address this challenge, we present a reproducible workflow, implemented in the Galaxy scientific workflow environment, for automated crack identification and quantitative damage assessment from scanning electron microscopy images. The workflow processes SEM images and experimental metadata to identify cracks, quantify damage, and retain the intermediate products and processing history needed for reproducibility. Outputs include crack masks, skeletonized crack networks, quality-control visualizations, and scalar damage descriptors. The method is designed to operate without image-specific parameter tuning across tungsten grades, microstructures, magnifications, and damage states. We demonstrate the workflow on a sparse electron-
    
[^33]: 在扩散与流匹配后验采样中正确设置引导权重

    Getting Your Guidance Weights Right in diffusion and flow-matching posterior sampling

    [https://arxiv.org/abs/2610.03503](https://arxiv.org/abs/2610.03503)

    提出一种基于最小二乘目标的简单离线策略，能够自动且有原则地调节扩散与流匹配后验采样中的引导权重，摆脱以往依赖启发式调参的做法。

    

    免训练后验采样方法，也称为即插即用方法，利用预训练的无条件扩散或流匹配模型来求解逆问题。大多数现有方法依赖引导权重，在每个时间步上平衡来自无条件分数或速度网络的先验信息与测量一致性，然而这些权重的调节方式往往未被讨论，主要依赖启发式方法。我们提出了一种简单且有原则的离线策略，用于自动调节这些引导权重。我们的关键观察是：在每个时间步上，扩散模型的条件去噪分数匹配目标，或流匹配模型的条件流匹配目标，都是一个最小二乘目标。因此，当条件预测被表示为无条件网络输出与测量引导项的加权和时，对这些权重的优化可以归结为一个最小二乘问题。

    arXiv:2610.03503v1 Announce Type: new  Abstract: Training-free posterior sampling methods, also known as Plug-and-Play methods, leverage pretrained unconditional diffusion or flow-matching models to solve inverse problems. Most existing approaches rely on guidance weights to balance, at each time step, prior information from the unconditional score or velocity network with measurement consistency, yet the tuning of these weights is often not discussed and is largely left to heuristics. We introduce a simple and principled offline strategy for automatically tuning these guidance weights. Our key observation is that, at each time step, the conditional denoising score-matching objective for diffusion models, or the conditional flow-matching objective for flow-matching models, is a least-squares objective. Therefore, when the conditional prediction is expressed as a weighted sum of the unconditional network output and a measurement-guidance term, optimizing over these weights reduces to a 
    
[^34]: 认证机制化编辑：技能移除与保留的行为保证

    Certified Mechanistic Edits: Behavioral Guarantees for Skill Removal and Preservation

    [https://arxiv.org/abs/2610.03502](https://arxiv.org/abs/2610.03502)

    该论文首次提出对机制化编辑的行为效果进行认证的方法，可对连续嵌入空间区域内的每个输入可证明地保证：禁用一个电路将移除一种技能同时保留另一种技能，并在标准Transformer上验证了该方法的有效性。

    

    机制化编辑（消融、权重编辑、激活引导）是从神经网络中遗忘有害能力同时保留有用能力的标准工具。当前方法仅通过测试来验证其效果，而测试永远无法覆盖整个连续的输入区域。先前处于可解释性与验证交界处的工作认证的是模型的描述：例如某个电路计算了什么，或者该电路是否忠实解释了整体。我们则认证编辑的行为效果：即在某个区域内，禁用一个电路会移除一种技能，并且可证明地为每一个输入保留另一种技能；这在信息流安全的意义上构成了特征非干扰保证。我们从玩具ReLU网络一直演示到标准的softmax + LayerNorm Transformer，在连续嵌入空间区域上证明了技能的移除与保留，并将输入扰动维度提升至精确求解器所能处理的约9倍。

    arXiv:2610.03502v1 Announce Type: cross  Abstract: Mechanistic edits (ablations, weight edits, activation steering) are the standard tools for unlearning a harmful capability from a neural network while preserving useful ones. Current approaches validate their effects only by testing, which can never cover an entire continuous region of inputs. Prior work at the interpretability-verification boundary certifies descriptions of a model: what a circuit computes, or whether it faithfully explains the whole. We instead certify the behavioral effect of an edit: that disabling a circuit removes one skill and provably preserves another, for every input in a region; a feature non-interference guarantee in the information-flow-security sense. We demonstrate such certified edits from toy ReLU networks up to a standard softmax + LayerNorm transformer, proving removal and preservation over continuous embedding-space regions and reaching roughly 9x the input-perturbation dimension an exact solver ca
    
[^35]: 在多大的训练规模以下，深度表格数据生成器不再胜过简单基线？一项关于临床和标准数据集规模阶梯的预注册基准测试

    Below what training size do deep tabular generators stop beating trivial baselines? A preregistered benchmark on a size ladder of clinical and standard datasets

    [https://arxiv.org/abs/2610.03500](https://arxiv.org/abs/2610.03500)

    这项预注册的规模阶梯基准测试（共2,220次运行）发现，在临床等小型数据集上，几乎所有深度表格生成模型（如CTGAN、TVAE、TabDDPM）在任何测试的训练规模下都无法以超过随机噪声的幅度击败简单基线方法，挑战了深度生成模型在小型表格数据上的价值假设。

    

    深度表格生成模型通常在拥有数万行数据的数据集上进行基准测试；而临床数据集往往只有数百行。我们预注册并运行了一项规模阶梯基准测试，以找出这两种情形的分界点：将8个公开数据集从200到20,000个训练行进行子采样，使用七个生成器（独立边缘分布、高斯Copula、SMOTE、无条件SMOTE、CTGAN、TVAE、TabDDPM），采用固定的20次试验调优预算和5个评估种子，外加4个真实规模的原生小型临床数据集，总计进行了2,220次已承诺的运行。主要评估指标是在合成数据上训练、在真实数据上测试的固定分类器的AUROC。在我们测量的所有训练规模下，24个（数据集，深度模型）配对中有23个，没有任何深度模型能以超过种子噪声的幅度击败最佳简单基线。在49个（数据集，规模）组合中，最佳基线在40个中获胜。我们预注册的预测——即深度模型的排名在小规模数据下会不稳定——被证伪：平均Kend……（摘要在此处截断）

    arXiv:2610.03500v1 Announce Type: new  Abstract: Deep tabular generative models are benchmarked on datasets with tens of thousands of rows; clinical datasets have hundreds. We preregistered and ran a size-ladder benchmark to find where the two regimes diverge: 8 public datasets subsampled from 200 to 20,000 training rows, seven generators (independent marginals, Gaussian copula, SMOTE, unconditional SMOTE, CTGAN, TVAE, TabDDPM) with a fixed 20-trial tuning budget and 5 evaluation seeds, plus 4 natively small clinical datasets at true size, for 2,220 committed runs in total. The primary metric is the AUROC of fixed classifiers trained on synthetic and tested on real data. In 23 of 24 (dataset, deep model) pairs no deep model ever beats the best trivial baseline by more than seed noise, at any training size we measured. The best baseline wins 40 of 49 (dataset, size) cells. Our preregistered prediction that the deep models' ranking would be unstable at small sizes is falsified: mean Kend
    
[^36]: 面向时间序列预测的最近锚定与循环排序方法

    Most-Recent Anchoring with Recurrent Ordering for Time Series Forecasting

    [https://arxiv.org/abs/2610.03494](https://arxiv.org/abs/2610.03494)

    提出MARO模型，以最新补丁为锚点、按从新到旧的顺序循环处理回看窗口，使时间序列长期预测架构能显式区分近期证据与远期上下文的不同贡献，且不引入额外参数。

    

    长期预测模型通常使用相同的固定网络堆栈处理回看窗口中的所有补丁，因此较旧的上下文补丁与近期证据获得相同的计算深度。然而，距离预测目标最近的信息与较远的上下文对预测的贡献并不相等，统一处理使得这种差异无法在架构中得到体现。我们提出MARO（Most-Recent Anchoring with Recurrent Ordering，最近锚定与循环排序）模型，它按照从最新补丁到最旧补丁的顺序处理回看窗口。最新的补丁充当锚点，用于初始化潜在状态并对后续每一步进行条件约束，从而将较旧的补丁逐步融合到一个始终以近期证据为中心的表示中。模型在每一步中复用同一个共享模块，因此将扫描范围向更远的过去扩展时不会引入额外的参数。扫描过程中保留的中间状态使预测头能够对短期与长期（信息进行权衡）……

    arXiv:2610.03494v1 Announce Type: new  Abstract: Long-term forecasting models commonly process all patches in a look-back window using the same fixed stack. Older contextual patches and recent evidence therefore receive the same computational depth. Yet the information closest to the forecast and the more distant context do not contribute equally. Uniform processing leaves this distinction unexpressed in the architecture. We propose MARO, a Most-Recent Anchoring with Recurrent Ordering model that processes the look-back window from the most recent patch to the oldest. The most recent patch serves as the anchor. It initializes the latent state and conditions each subsequent step, so older patches are folded into a representation that remains centered on recent evidence. A single shared module is reused at every step, so extending the scan further into the past introduces no additional parameters. Intermediate states retained during the scan allow the forecast head to weigh short and lon
    
[^37]: 面向时间序列预测的双上下文类比检索

    Dual-Context Analog Retrieval for Time Series Forecasting

    [https://arxiv.org/abs/2610.03491](https://arxiv.org/abs/2610.03491)

    DuoTS提出了一种双上下文类比检索框架，通过结合捕捉近期动态的当前上下文与提供细节区分的细节上下文，逐步利用历史相似状态的后续观测来精化时序预测，从而避免完全依赖不可靠的单一检索结果。

    

    大多数长期时间序列预测模型通过单次前向传递将回看窗口直接映射到完整的预测时域。这种设计虽然高效，但并未显式识别哪些历史状态与不同的未来片段最为相关，也未能利用这些历史状态之后实际发生了什么。类比预测通过检索与当前相似的历史状态并利用其后续观测来解决这一问题，但单一最近邻匹配可能不可靠，且重叠的片段可能产生冗余的候选。我们提出了DuoTS——一个双上下文时间序列预测模型，它利用检索到的证据但不完全依赖于此。DuoTS首先使用并行片段编码器和线性预测头生成基础预测，然后逐个未来片段地进行渐进式细化。每次细化结合两种视角：一种是关注近期token并捕捉最新动态的当前上下文，另一种是提供区分性……（摘要在此处截断）

    arXiv:2610.03491v1 Announce Type: new  Abstract: Most long-term time-series forecasting models map the look-back window directly to the full horizon in a single pass. While efficient, this design does not explicitly identify which historical states are most relevant to different future segments or exploit what followed those states. Analog forecasting addresses this by retrieving past states similar to the present and using their observed continuations, but single nearest matches can be unreliable and overlapping patches may produce redundant candidates. We propose DuoTS, a Dual-Context Time Series forecasting model that uses retrieved evidence without relying on it exclusively. DuoTS first produces a base forecast with a parallel patch encoder and linear prediction head, then progressively refines it one future patch at a time. Each refinement combines two views: a current context that attends to recent tokens and captures the latest dynamics, and a detail context that provides distin
    
[^38]: AREX：用于流匹配少步采样的仿射-残差指数积分器

    AREX: Affine-Residual Exponential Integrator for Few-Step Sampling in Flow Matching

    [https://arxiv.org/abs/2610.03483](https://arxiv.org/abs/2610.03483)

    AREX是一种无需训练的流匹配模型少步采样器，它将采样动力学分解为由目标均值和协方差决定的仿射分量（用显式矩阵值传播子积分）与神经残差项，在无需重训练的情况下持续提升少步采样的样本保真度。

    

    我们提出了AREX，一种面向预训练流匹配模型的无需训练的采样器，它利用目标均值和协方差来捕获采样动态中可解析处理的部分。我们证明了矩匹配高斯目标的速度场是边缘速度场的 $L^2$ 最优仿射近似。这促使我们将学习到的动力学分解为覆盖整个采样路径的仿射分量（由目标的前两阶矩决定）以及一个神经残差项。AREX保留仿射分量，并使用显式矩阵值传播子对其进行积分；相应地，我们只需对残差项进行积分。这不同于标量指数积分器，后者只能解析地处理各向同性的线性动力学。在图像和文本生成图像任务中，AREX在少步采样机制下持续提升样本保真度，而无需重新训练底层模型。

    arXiv:2610.03483v1 Announce Type: cross  Abstract: We introduce AREX, a training-free sampler for pretrained flow matching models that uses the target mean and covariance to capture an analytically tractable part of the sampling dynamics. We show that the velocity field of the moment-matched Gaussian target is the $L^2$-optimal affine approximation to the marginal velocity field. This motivates decomposition of the learned dynamics into an affine component over the whole sampling path, determined by the first two target moments, and a neural residual term. AREX keeps the affine component and integrates it using an explicit matrix-valued propagator. In turn, we only require to integrate over the residual term. This differs from scalar exponential integrators, which analytically handle only isotropic linear dynamics. Across image and text-to-image generation tasks, AREX consistently improves sample fidelity in the few-step sampling regime without retraining the underlying model.
    
[^39]: Metropolis-Hastings 在策略组合中优于重要性重采样

    Metropolis-Hastings Dominates Importance Resampling for Policy Composition

    [https://arxiv.org/abs/2610.03480](https://arxiv.org/abs/2610.03480)

    本文证明在任意采样预算下，基于 Metropolis-Hastings 的迭代校正方法产生的输出分布（以任意凸 f-散度衡量）都不劣于重要性重采样（SIR），从而为解码时策略组合中 MH 的理论优势提供了保证。

    

    对大语言模型（LLM）进行后训练通常需要在多个奖励之间探索权衡，但针对每种权衡都重新训练的成本很高。解码时策略组合方法允许在推理阶段通过组合针对特定奖励的策略来调整这些权衡。这种组合的目标是各策略在完整回复上概率的加权乘积，但标准实现方式组合的是它们的下一词元概率，通常会引入采样偏差。我们分析了一种基于独立 Metropolis-Hastings（MH）的已知迭代校正方法。我们的主要结果表明，在任意采样预算下，以任何凸 f-散度衡量，MH 产生的输出分布至少与相同预算下采样-重要性-重采样（SIR）同样接近目标分布。我们还推导出了 MH 相对于未校正解码器在衡量与所提供策略一致程度的共识目标上的改进下界。

    arXiv:2610.03480v1 Announce Type: new  Abstract: Post-training a large language model (LLM) often requires exploring trade-offs between multiple rewards, but retraining for each trade-off is expensive. Decoding-time policy composition allows these trade-offs to be adjusted by combining reward-specific policies at inference time. This composition targets a weighted product of the policies' probabilities over complete responses, but standard implementations combine their next-token probabilities, generally introducing sampling bias. We analyze a known iterative correction based on independence Metropolis-Hastings (MH). Our main result shows that, for every rollout budget, MH produces an output distribution at least as close to the target as sampling-importance-resampling (SIR) with the same budget, as measured by every convex f-divergence. We also derive a lower bound on MH's improvement over the uncorrected decoder in a consensus objective measuring agreement with the supplied policies.
    
[^40]: 阶段结构强化学习中的单一策略还是多策略？

    Single or Multiple Policies for Phase-Structured Reinforcement Learning?

    [https://arxiv.org/abs/2610.03475](https://arxiv.org/abs/2610.03475)

    该论文从理论上证明单一共享策略可以达到任何多策略方案的性能，但实践中多策略是否更优取决于函数逼近、学习优化过程以及策略切换的样本效率与连续性损失等因素。

    

    许多强化学习（RL）问题是非平稳的但具有结构化特征，可以分解为多个阶段，每个阶段都有各自的转移概率和奖励函数。当阶段序列已知时，常见的解决方案是通过增加状态信息来满足马尔可夫性质，并应用标准的强化学习技术。然而，先前的研究发现，针对不同阶段采用多策略的方法可以优于在各阶段之间共享的单一状态增广策略，其原因尚不清楚。在这项工作中，我们首先从理论上证明共享策略可以达到任何多策略解决方案的性能。然而，在实践中，多策略解决方案是否比相应的单一共享策略表现更好，取决于函数逼近、学习和优化过程，以及对于多策略解决方案而言，从一个策略切换到另一个策略所带来的样本效率和连续性损失。我们提出……

    arXiv:2610.03475v1 Announce Type: cross  Abstract: Many reinforcement-learning (RL) problems are non-stationary yet structured and can be decomposed into phases, each with its own transition probabilities and reward functions. When the phase sequence is known, the common solution augments the state with information to satisfy the Markovian property and applies standard RL techniques. However, prior work finds that the multi-policy approach for different phases can outperform a single state-augmented policy shared among the phases, for reasons that remain unclear. In this work, we first show that the shared policy can theoretically achieve performance of any multi-policy solution. However, whether a multi-policy solution can perform better than the corresponding single shared policy in practice depends on function approximation, learning and optimization processes, as well as, for multi-policy solutions, the sample efficiency and loss of continuity from one policy to another. We propose
    
[^41]: 何时准确率才是证据？泛化、验证与信息融合的统一理论

    When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion

    [https://arxiv.org/abs/2610.03465](https://arxiv.org/abs/2610.03465)

    该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。

    

    K折交叉验证（CV）被广泛用作样本外性能的证据，然而在异构数据下，各折既非独立实验，也非信息量均等。交叉上界验证（CUBV）用真实风险的保守上界取代逐点的CV准确率。本文通过单一的指数框架对CUBV进行了推广，其中泛化间隙的矩生成函数由累积量包络gamma(lambda)所控制。由此得到一族风险界，涵盖Hoeffding界、Bernstein界、依赖感知、PAC-Bayesian以及异构数据源融合等多种情形。对于K折CV，各折间隙之间的相关性通过联合次高斯代理矩阵来建模。在等相关条件下，得到有效折数Keff = K/[1+(K-1)rho]，表明当各折之间存在强相关性时，增大K并不一定能增加统计证据。该框架还被用于……（原文摘要在此处截断）

    arXiv:2610.03465v1 Announce Type: cross  Abstract: K-fold cross-validation (CV) is widely used as evidence of out-of-sample performance, although folds are neither independent experiments nor equally informative under heterogeneous data. Cross Upper-Bound Validation (CUBV) replaces point-wise CV accuracy by conservative upper bounds on true risk. Here we generalise CUBV through a single exponential framework in which the moment-generating function of the generalisation gap is controlled by a cumulant envelope gamma(lambda). This yields a family of risk bounds covering Hoeffding-, Bernstein-, dependency-aware, PAC-Bayesian, and heterogeneous source-fusion settings. For K-fold CV, dependence between fold-wise gaps is modelled through a joint sub-Gaussian proxy matrix. Under equicorrelation, this gives an effective number of folds, Keff = K/[1+(K-1)rho], showing that increasing K does not necessarily increase statistical evidence when folds are strongly dependent. The framework is also ex
    
[^42]: 基于上下文学习的Transformer神经量子态的泛化

    Generalization of Transformer-Based Neural Quantum States via In-Context Learning

    [https://arxiv.org/abs/2610.03463](https://arxiv.org/abs/2610.03463)

    本文为基于Transformer的神经量子态在上下文学习下的泛化行为建立了理论框架，严格证明了其逐点预测误差随上下文示例数量和Transformer深度成反比下降，且所需网络深度仅线性增长。

    

    基于现代深度学习架构的神经量子态已成为量子多体系统的强大表示方法。特别是，基于Transformer的神经量子态提供了能够捕捉长程关联的富有表现力的模型，其经验泛化性能最近已得到证明。然而，对其泛化行为的理论理解在很大程度上仍未被探索。在本文中，我们开发了一个理论框架来分析基于Transformer的神经量子态在上下文学习下的泛化特性。我们建立了以均方误差（MSE）表示的严格的推理时泛化误差界，表明逐点预测误差随上下文示例的数量和Transformer的深度成反比下降。我们进一步证明，实现这一保证所需的Transformer深度仅呈线性增长（原文在此处截断）。

    arXiv:2610.03463v1 Announce Type: cross  Abstract: Neural quantum states based on modern deep learning architectures have emerged as powerful representations for quantum many-body systems. In particular, Transformer-based neural quantum states provide expressive models capable of capturing long-range correlations, and their empirical generalization performance has recently been demonstrated. However, a theoretical understanding of their generalization behavior remains largely unexplored. In this paper, we develop a theoretical framework to analyze the generalization properties of Transformer-based neural quantum states under in-context learning. We establish a rigorous inference-time generalization error bound in terms of mean squared error (MSE), showing that the pointwise prediction error decreases inversely with both the number of in-context examples and the depth of the Transformer. We further show that the Transformer depth required to achieve this guarantee scales only linearly w
    
[^43]: 超越随机划分：在化学与生物学动机的分布偏移下评估药物-靶点亲和力模型

    Beyond Random Splits: Evaluating Drug-Target Affinity Models Under Chemically and Biologically Motivated Distribution Shifts Copy

    [https://arxiv.org/abs/2610.03456](https://arxiv.org/abs/2610.03456)

    该研究构建了包含化学与生物学动机分布偏移的DTA评估基准，发现蛋白质层面及双重分布偏移会显著降低模型性能并改变甚至逆转架构排名。

    

    药物-靶点亲和力（DTA）预测被广泛用于在昂贵的实验筛选之前对候选化合物进行优先级排序。DTA模型通常仅在单一数据划分下进行比较，而实际部署中可能需要外推到新的化学系列、新的蛋白质靶点，或两者兼有。我们探究用于评估的分布偏移是否会改变哪种架构看起来表现最好。我们从ChEMBL和BindingDB数据集中整理了718,800个独特的药物-蛋白质对。我们将Morgan指纹+蛋白质CNN基线与12种受控架构进行比较，这些架构结合了四种药物表示与三种ESM-2交互模式。平均验证RMSE从支架划分和指纹聚类OOD（分布外）下的0.950和0.945上升到蛋白质聚类OOD和双重OOD下的1.299和1.321。模型排名在两种化学偏移之间较为相似（tau = 0.79），但在蛋白质OOD下与支架OOD的一致性下降（tau = 0.39），并在双重OOD下发生逆转（原文在此处截断）。

    arXiv:2610.03456v1 Announce Type: new  Abstract: Drug-target affinity (DTA) prediction is widely used to prioritize candidate compounds before costly experimental screening. DTA models are often compared under a single data split, even though deployment may require extrapolation to new chemical series, new protein targets, or both. We ask whether the distribution shift used for evaluation changes which architecture appears best. We curate 718,800 unique drug-protein pairs from the ChEMBL and BindingDB datasets. We compare a Morgan-fingerprint + protein-CNN baseline with 12 controlled architectures that combine four drug representations with three ESM-2 interaction modes. Mean validation RMSE increases from 0.950 and 0.945 under scaffold and fingerprint-cluster OOD to 1.299 and 1.321 under protein-cluster and dual OOD. Model rankings are similar across the two chemical shifts (tau = 0.79), but agreement with scaffold OOD falls under protein OOD (tau = 0.39) and reverses under dual OOD (
    
[^44]: 少测量，多知晓：自监督测试时特征获取

    Measure Less, Know More: Self-Supervised Test-Time Feature Acquisition

    [https://arxiv.org/abs/2610.03454](https://arxiv.org/abs/2610.03454)

    该论文提出ECHO-k，一种任务无关的自监督测试时模态获取方法，它利用基础模型的内部预训练表示作为代理目标，并通过强化学习策略顺序选择信息量最大的模态，从而在有限预算下持续提升下游任务性能。

    

    多模态、高维学习的最新进展使基础模型能够处理异构的大规模数据。然而，在测试时获取所有特征或模态可能成本极高且往往冗余。因此，顺序选择信息丰富的模态至关重要，但当下游任务或预测目标未知时，这一任务充满挑战。为此，我们提出了ECHO-k，一种任务无关且自监督的模态获取学习原则：我们使用深度模型的内部预训练表示（例如来自基础模型的表示）作为总结跨模态信息的代理目标。我们在一个简化的线性设定中提供了理论保证，从而为顺序模态选择的强化学习（RL）策略提供了理论依据。在任务无关和无标签获取的各类基线方法中，ECHO-k在多种基础模型上持续提升了预算约束下的下游任务性能。

    arXiv:2610.03454v1 Announce Type: cross  Abstract: Recent progress in multimodal, high-dimensional learning has enabled foundation models to process heterogeneous, large-scale data. However, at test time, acquiring all features or modalities can be prohibitively costly and often redundant. Sequentially selecting informative modalities is therefore critical, yet challenging when the downstream task or prediction target is unknown. To this end, we introduce ECHO-$k$, a task-agnostic and self-supervised learning principle for modality acquisition: we use a deep model's internal pretrained representations (e.g., from a foundation model) as proxy targets that summarize cross-modal information. We provide theoretical guarantees in a stylized linear setting that motivate a reinforcement learning (RL) policy for sequential modality selection. Across task-agnostic and label-free acquisition baselines, ECHO-$k$ consistently improves budgeted downstream performance across diverse foundation-model
    
[^45]: 基于非平稳性的含瞬时与滞后关系的因果表征学习

    Causal Representation Learning with Instantaneous and Lagged Relations via Nonstationarity

    [https://arxiv.org/abs/2610.03452](https://arxiv.org/abs/2610.03452)

    本文通过利用与转移噪声分布变化相关的辅助变量所刻画的非平稳性，建立了识别时间序列潜在状态及其瞬时与滞后因果结构的充分条件，并据此提出了基于对比学习的 iCReN 因果表征学习框架。

    

    时间序列数据的因果表征学习旨在从观测数据中识别潜在状态及其因果关系。在这一设定中，一个重要的挑战是如何同时建模跨观测区间的滞后因果关系，以及表现为区间内瞬时关系的更快因果效应，同时还要考虑时间序列数据中的非平稳性。然而，能够联合处理这些因果关系与非平稳性的方法仍然有限。为填补这一空白，我们建立了充分条件，利用与转移噪声分布变化相关联的观测辅助变量（例如时间或条件标签），在分量置换和分量级可逆变换的意义下识别潜在状态，并在相同置换的意义下识别其瞬时和滞后因果结构。基于这些结果，我们提出了 iCReN，一个使用对比学习与离散…（原文摘要至此截断）

    arXiv:2610.03452v1 Announce Type: new  Abstract: Causal representation learning for time-series data aims to identify latent states and their causal relations from observations. In this setting, an important challenge is to model both lagged causal relations across observation intervals and faster causal effects that appear as instantaneous relations within an interval, while accounting for nonstationarity in time-series data. However, methods that jointly handle these causal relations and nonstationarity remain limited. To address this gap, we establish sufficient conditions for identifying latent states up to component permutation and component-wise invertible transformations, and their instantaneous and lagged causal structures up to the same permutation, using an observed auxiliary variable, such as time or a condition label, associated with changes in transition-noise distributions. Based on these results, we propose iCReN, a framework that uses contrastive learning with discrete 
    
[^46]: 机器学习分子吸收光谱中电子密度与几何结构输入的对比

    Electronic Density versus Geometry for Machine-Learned Molecular Absorption Spectra

    [https://arxiv.org/abs/2610.03444](https://arxiv.org/abs/2610.03444)

    该研究系统比较了以基态电子密度和分子几何结构作为机器学习模型输入来预测分子吸收光谱的效果，探讨了分子信息表示方式对光谱预测性能的影响。

    

    分子光学吸收光谱能够直接探测电子结构，被广泛用于分子识别、光物理行为解释以及光谱实验的规划。然而，使用第一性原理激发态方法计算吸收光谱的计算成本高昂，至少与基态计算相比是如此，这限制了其在大规模分子集合中的常规应用。机器学习（ML）代理模型可以降低这一成本，实现快速的光谱预测。然而，其性能在很大程度上取决于分子信息的表示方式。在此，我们比较了以基态电子密度与分子几何结构作为机器学习模型输入来预测吸收光谱的效果，训练集包含从QM7数据集中选取的6874个分子。对于其中每个分子，电子密度均采用密度泛函理论（DFT）计算得到。

    arXiv:2610.03444v1 Announce Type: cross  Abstract: Molecular optical absorption spectroscopy provides a direct probe of electronic structure and is widely used for molecular identification, interpretation of photophysical behaviour, and planning of spectroscopy experiments. Calculating the absorption spectra using first-principle excited-state methods, however, is computationally demanding, at least compared to ground-state calculations, which limits their routine application across large molecular sets. Machine-learning (ML) surrogates can reduce this cost and allow rapid spectral prediction. However, their performance depends strongly on how molecular information is represented. Here, we compare using the ground-state electron density versus the molecular geometry as inputs to a ML model for predicting absorption spectra, for a training set of 6874 molecules selected from the QM7 dataset. For each of these molecules, the density was calculated using density functional theory (DFT) an
    
[^47]: OptiSelect：优化器如何塑造数据课程？

    OptiSelect: How does the Optimizer Shape Data Curriculum?

    [https://arxiv.org/abs/2610.03432](https://arxiv.org/abs/2610.03432)

    本文提出OptiSelect优化器感知的数据选择范式，首次系统研究优化器如何影响数据课程选择，理论上证明基于符号和极坐标切向预处理的优化器（Lion、Muon）因效用分数可区分性崩溃而限制选择增益，而对角自适应优化器（AdamW、Sophia）则能获得严格更优的增益上界。

    

    在线数据选择通过在每个批次内仅训练最有价值的候选样本，为大语言模型预训练带来了显著的效率提升。由于候选样本的价值是通过其有效的模型更新来体现的，有原则的数据选择应当考虑优化器步骤——因为优化器会在原始梯度更新模型参数之前对其进行重塑。我们将这种优化器感知的选择范式形式化为OptiSelect，并首次系统性地研究了优化器如何塑造数据选择。我们的理论建立了一个选择增益原则：在线选择的优势由优化器诱导的效用分数的可区分性所决定。我们证明，Lion和Muon所采用的基于符号的及极坐标切向的预处理器会遭受可区分性崩溃，从而限制了OptiSelect可获得的增益上限；而对角自适应优化器（如AdamW和Sophia）则具有严格更优的上界。该性……

    arXiv:2610.03432v1 Announce Type: cross  Abstract: Online data selection has demonstrated substantial efficiency gains for LLM pretraining by training on the most valuable candidates within each batch. Since a candidate's value is realized through its effective model update, principled selection should account for the optimizer step, which reshapes the raw gradient before it updates model parameters. We formalize this optimizer-aware selection paradigm as OptiSelect and present the first systematic study of how the optimizer shapes data selection. Our theory establishes a selection gain principle in which the advantage of online selection is governed by the discriminability of the optimizer-induced utility scores. We prove that sign-based and polar-tangential preconditioners of Lion and Muon would suffer from a discriminability collapse which caps attainable gains from OptiSelect, whereas diagonal-adaptive optimizers such as AdamW and Sophia admit strictly better upper bounds. The prop
    
[^48]: 深度贝叶斯REFoCUS

    Deep Bayesian REFoCUS

    [https://arxiv.org/abs/2610.03419](https://arxiv.org/abs/2610.03419)

    该论文提出Deep Bayesian REFoCUS方法，将超声多静态恢复建模为贝叶斯推断问题，利用深度生成先验攻克经典线性解码器失效的秩亏缺难题，在性能、不确定性量化以及无需微调的跨域泛化能力上均显著优于线性基线方法。

    

    在这项工作中，我们将从任意发射序列中恢复超声多静态数据的问题表述为一个贝叶斯推断问题。为此，我们在多静态数据集上训练深度生成先验，以解决经典线性REFoCUS解码器失效的秩亏缺情形。这种被我们称为Deep Bayesian REFoCUS的方法，在所有秩亏缺程度和噪声水平下均优于线性基线方法，并且在反演完全精确时退化为线性解码。该模型还能表达采集数据零空间中的不确定性，而线性REFoCUS解码器仅能提供点估计。最后，我们分析了仿真与在体采集之间分布偏移的影响，结果表明该模型无需任何微调或自适应处理，即可展现出出色的泛化能力。

    arXiv:2610.03419v1 Announce Type: new  Abstract: In this work we formulate ultrasound multistatic recovery from arbitrary transmit sequences as a Bayesian inference problem. To that end, we train a deep generative prior on multistatic data sets to tackle the rank-deficient regime in which classical linear REFoCUS decoders fail. This appproach, which we term Deep Bayesian REFoCUS, outperforms the linear baselines for all regimes of rank-deficiency and noise levels, and regresses to linear decoding when inversion is exact. The model also expresses uncertainty in the null space of the acquisitions, whereas the linear REFoCUS decoders only provide point estimates. Finally, we analyze the impact of distribution shift between simulation and in-vivo acquisitions, showing remarkable generalization ability without any fine-tuning or adaptation.
    
[^49]: 通过信息增长重新思考节点分类中的认知不确定性

    Rethinking Epistemic Uncertainty in Node Classification through Information Growth

    [https://arxiv.org/abs/2610.03418](https://arxiv.org/abs/2610.03418)

    本文提出了一个在信息增长条件下检验节点分类中认知不确定性可约减性的统计框架，并揭示现有图证据深度学习方法难以满足一致性准则。

    

    认知不确定性应当随着预测器获得更多关于数据生成过程（DGP）的信息而降低。然而，现有的用于节点分类的图证据深度学习（EDL）方法通常从图特有性质出发构建认知不确定性，并在分布外检测等下游任务上对其进行评估，这些做法并未检验其作为DGP信息增加时是否可被约减。为了使这种可约减性能够被直接检验，我们提出了一个在信息增长条件下研究认知不确定性的统计框架。该框架规定了信息增长实验协议以及认知预测器的一致性准则，并使用投影图DGP来确保不断增长的图（通常并不必然提供关于同一DGP的递增信息）构成对同一底层过程的一致观测。我们证明，EDL方法无法……（原文在此处截断）

    arXiv:2610.03418v1 Announce Type: cross  Abstract: Epistemic uncertainty should decrease as additional information about the data-generating process (DGP) becomes available to the predictor. Yet, existing graph evidential deep learning (EDL) methods for node classification typically construct epistemic uncertainty from graph-specific properties and evaluate it on downstream tasks such as out-of-distribution detection, which do not test its reducibility as information about the DGP increases. To make reducibility directly testable, we introduce a statistical framework for studying epistemic uncertainty under information growth. Our framework specifies an information-growth experimental protocol and a consistency criterion for epistemic predictors, while using projective graph DGPs to ensure that growing graphs, which in general need not provide increasing information about the same DGP, constitute coherent observations of the same underlying process. We show that EDL methods do not expl
    
[^50]: 迭代一致性模型：稳定性、误差界与噪声调度

    Iterating Consistency Models: Stability, Error Bounds and Noise Schedules

    [https://arxiv.org/abs/2610.03414](https://arxiv.org/abs/2610.03414)

    该论文将多步一致性模型采样分析为加噪与近似去噪算子的复合，在可验证的稳定性假设下推导出非渐近误差界，揭示了噪声调度中早期大噪声驱动误差收缩、后期小噪声控制残余偏差的作用机制，为CM采样器设计提供了理论指导。

    

    一致性模型（Consistency Models, CMs）已成为用少量步骤生成高质量样本的主流方法。然而，增加采样步骤既可能提升也可能降低样本质量，且这种现象对噪声调度高度敏感，而现有理论无法完全解释。为了提供精度保证并指导CM采样器设计，我们将多步CM采样分析为加噪算子与近似去噪算子的复合。在显式且可验证的稳定性假设下，我们推导出一个非渐近误差界，该误差界将初始化误差的收缩与近似误差的累积分离开来。该误差界为噪声调度分配了不同的角色：较大的早期噪声水平驱动误差收缩，而较小的后期噪声水平控制残余偏差。作为推论，我们为强对数凹和半对数凹目标分布得到了显式常数。我们进一步建立了一个互补的理论保证，其假设条件、一步精度和……（摘要在此处被截断）

    arXiv:2610.03414v1 Announce Type: cross  Abstract: Consistency models (CMs) have become a leading approach for generating high-quality samples in few steps. However, adding steps can improve or degrade sample quality in ways that are highly sensitive to the schedule and that existing theory does not fully explain. To provide accuracy guarantees and guide CM sampler design, we analyze multistep CM sampling as a composition of noising and approximate denoising operators. Under explicit, verifiable stability assumptions, we derive a non-asymptotic error bound that separates contraction of the initialization error from accumulation of approximation error. The bound assigns distinct roles to the schedule: large early noise levels drive contraction, while small late noise levels control the residual bias. As a corollary, we obtain explicit constants for strongly log-concave and semi-log-concave targets. We further establish a complementary guarantee whose assumptions, one-step accuracy and s
    
[^51]: AIBL：具有结构化记忆与神经嵌入的增强型基于实例的学习

    AIBL: Augmented Instance-Based Learning with Structured Memory and Neural Embeddings

    [https://arxiv.org/abs/2610.03413](https://arxiv.org/abs/2610.03413)

    该论文提出AIBL，将经典的基于实例的学习理论（IBLT）从符号表示推广到学习得到的神经嵌入向量空间，用结构化记忆和学习的相似度来处理高维序贯数据，同时保留实例存储、激活加权检索和效用混合等核心机制。

    

    序贯学习系统通常需要在接收高维输入的同时，依据积累的经验做出决策，而这些输入的分布可能随时间发生变化。基于实例的学习理论为这类场景提供了一个有原则的基于案例的框架，其核心机制包括存储“情境-决策-效用”实例、部分匹配、激活和混合。然而，IBLT依赖于字典式的符号知识表示，而文本、图像、交易向量以及用户-物品历史等数据往往需要通过学习获得的相似度，而非手工指定的匹配规则。在本文中，我们提出了AIBL（增强型基于实例的学习），这是一种在学习得到的向量空间中构建的、面向高维序贯数据的实例学习模型。AIBL将符号化的情境匹配推广为神经嵌入相似度，同时保留了实例存储、激活加权检索和效用混合机制。AIBL模型将记忆组织为活跃的……（摘要在此处被截断）

    arXiv:2610.03413v1 Announce Type: new  Abstract: Sequential learning systems often make decisions from accumulated experience while receiving high-dimensional inputs whose distribution may change over time. Instance-Based Learning Theory (IBLT) provides a principled case-based framework for such settings through stored situation-decision-utility instances, partial matching, activation, and blending. IBLT relies on symbolic knowledge representation in dictionary-like formats, but text, images, transaction vectors, and user-item histories often require learned similarity rather than hand-specified matching rules. In this paper, we introduce AIBL (Augmented Instance-Based Learning), an instance-learning model formulated in a learned vector space for high- dimensional sequential data. AIBL generalizes symbolic situation matching to neural embedding similarity while retaining instance storage, activation- weighted retrieval, and utility blending. The AIBL model organizes memory into active,
    
[^52]: 对比神经嵌入揭示超越对话角色的个体特质

    Contrastive Neural Embeddings Reveal Individual Traits Beyond Conversational Role

    [https://arxiv.org/abs/2610.03410](https://arxiv.org/abs/2610.03410)

    该研究将对比学习方法CEBRA应用于对话中成对被试的脑电数据，发现双人组合的个体特质（如自闭症商数差异）可解码至高于随机水平，但严格的置换检验表明解码性能可能主要反映嵌入的几何结构而非真实的标签信息。

    

    对比表征学习越来越多地被用于从神经记录中恢复低维结构，但其输出通常通过解码准确率来验证，而非通过其产生的流形几何结构来验证。我们将CEBRA应用于对话中成对被试（dyads）的脑电（EEG）记录，并分析所得的嵌入——该嵌入在训练中被约束在二维球面上。描述双人组合特征的标签，包括两名伙伴自闭症商数（AQ）得分之间的绝对差值，其解码结果远高于随机水平（对于二元AQ量级，准确率为0.77，对比0.55的多数类基线；对于六分类|ΔAQ|划分，准确率为0.44，对比0.25）。然而，两种置换对照方法的结果存在显著差异：在冻结的嵌入上置换标签得到p = 0.001，而在每次置换下重新训练编码器则得到p = 0.50。只有后者才是对标签（而非几何结构）的有效检验。与此相一致的是，球形混合结构以及p……（原文摘要在此处截断）

    arXiv:2610.03410v1 Announce Type: cross  Abstract: Contrastive representation learning is increasingly used to recover low-dimensional structure from neural recordings, but its output is typically validated by decoding accuracy rather than by the geometry of the manifold it produces. We apply CEBRA to EEG recorded from dyads in conversation, and analyze the resulting embedding, which training constrains to the 2D sphere. Labels describing the dyads, including the absolute difference between partners' autism-quotient scores, decode well above chance (0.77 against a 0.55 majority baseline for binary AQ magnitude; 0.44 against 0.25 for the six-class $|\Delta$AQ$|$ partition). However, the two permutation controls have notable differences in results: permuting labels over a frozen embedding yields p = 0.001, whereas retraining the encoder under each permutation yields p = 0.50. Only the latter tests the label rather than the geometry. Consistent with this, spherical mixture structure and p
    
[^53]: 微控制器上卷积神经网络的16位精度与8位成本

    16-bit Precision of Convolutional Neural Networks on Microcontroller Units for 8-bit Costs

    [https://arxiv.org/abs/2610.03402](https://arxiv.org/abs/2610.03402)

    本文提出W16A16，一种面向微控制器的16位高精度量化方法，在Armv7E-M架构上以与8位量化相当甚至更优的速度和能耗，将量化误差降低约10倍。

    

    要在边缘硬件上部署深度神经网络，需要既能保持高精度又具备高效率的推理方案。本工作提出了W16A16，一种高精度（16位）、快速、低能耗的量化方法。在广泛应用的微控制器架构Armv7E-M上，我们提出的方法在层级和模型级上都比其他量化方案实现了更快的速度和更低的能耗。我们分析了Armv7E-M的架构，解释了16位方法性能优势背后的基本原理，并评估了回归和分类任务的经验量化误差，以及MCU部署中的实际时间和能耗。我们观察到，与8位量化方案相比，量化误差约降低10倍，同时实现了相似或更好的推理时间和能耗。

    arXiv:2610.03402v1 Announce Type: new  Abstract: To deploy deep neural networks on edge hardware, highly efficient inference schemes are necessary that retain high accuracy. This work presents W16A16, a high precision (16-bit), fast speed, low energy quantization method. On a widely applied microcontroller architecture Armv7E-M, our proposed approach achieves faster speed and lower energy consumption on layer- and model-level compared to alternative quantization schemes. We analyze the architecture of Armv7E-M, explain the underlying principles behind the performance advantages of 16-bit approaches, and evaluate the empiric quantization errors for regression and classification tasks, as well as empiric time- and energy consumption in MCU deployment. We observe ca.\ 10 times lower quantization errors compared to 8-bit quantization schemes while achieving similar or better inference times and energy consumption.
    
[^54]: 基于生成模型与观测插值子的贝叶斯数据同化统一框架

    A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants

    [https://arxiv.org/abs/2610.03396](https://arxiv.org/abs/2610.03396)

    该论文提出一个无需重训练的观测插值子统一框架，可将预训练的随机插值子、流匹配和扩散模型直接转化为贝叶斯数据同化中的后验采样器，并统一了随机与确定性两种后验采样方法。

    

    贝叶斯数据同化将模型预测与含噪观测相结合，但对高维、非高斯后验分布进行采样仍然具有挑战性。我们提出了一种观测插值子框架，无需重新训练即可将预训练的随机插值子、流匹配以及扩散模型转化为后验采样器。通过将插值路径以观测为条件，可以得到对漂移项或速度场的共享似然分数修正，从而统一了随机与确定性两种后验采样方式。当中间似然分数已知时，所得的SDE和ODE可对精确后验进行采样。在实际计算中，我们采用闭式高斯代理来近似该分数，其均值经过偏差校正，协方差由模型的源协方差进行膨胀。无雅可比矩阵和集合共享的近似方法使该框架在高维问题中切实可行。我们在线性-高斯动力学、随机双（摘要原文在此处截断）等场景上对该框架进行了评估。

    arXiv:2610.03396v1 Announce Type: new  Abstract: Bayesian data assimilation combines model forecasts with noisy observations, but sampling high-dimensional, non-Gaussian posteriors remains challenging. We introduce an observation-interpolant framework that turns pretrained stochastic interpolant, flow matching, and diffusion models into posterior samplers without retraining. Conditioning the interpolant path on observations yields a shared likelihood-score correction to the drift or velocity, unifying stochastic and deterministic posterior sampling. The resulting SDEs and ODEs sample the exact posterior when the intermediate likelihood score is known. For practical computation, we approximate this score using a closed-form Gaussian surrogate with a bias-corrected mean and covariance inflated by the model's source covariance. Jacobian-free and ensemble-shared approximations make the method tractable in high dimensions. We evaluate the framework on linear-Gaussian dynamics, stochastic tw
    
[^55]: 面向强化学习的双向Voronoi偏置探索课程

    Bidirectional Voronoi-biased Exploration Curriculum for Reinforcement Learning

    [https://arxiv.org/abs/2610.03395](https://arxiv.org/abs/2610.03395)

    提出BVER方法，借鉴双向RRT规划思想，让初始状态课程与目标课程同时从两端相向扩展并偏向未探索的任务空间，从而有效缓解稀疏奖励长时程任务中的探索瓶颈。

    

    具有稀疏奖励的长时程任务为目标条件强化学习带来了探索瓶颈：从初始状态出发的策略很少能到达目标，因而无法获得学习信号。参考动作、人工设计的课程以及奖励整形可以提供这种信号，但需要演示数据或针对特定任务的工程工作；自动的初始状态与目标课程虽然避免了这一问题，但通常只从一端进行扩展，因此必须从该侧覆盖到达目标的全部距离。我们提出强化学习的双向Voronoi偏置探索课程（BVER），它同时从两端进行扩展。受双向RRT规划的启发，BVER从目标向外生长初始状态，并从初始状态分布向外生长目标，使两者都偏向未探索的任务空间，并引导它们相互靠近，进而在两者之上训练同一个目标条件策略。在质点迷宫、四足箱子（摘要在此处截断）

    arXiv:2610.03395v1 Announce Type: new  Abstract: Long-horizon tasks with sparse rewards pose an exploration bottleneck for goal-conditioned reinforcement learning: a policy started from the initial state rarely reaches the goal and receives no learning signal. Reference motions, hand-designed curricula, and shaped rewards supply this signal but require demonstrations or task-specific engineering; automatic start-state and goal curricula avoid this but typically expand from one side only, so the full distance to the target must be covered from that side. We propose the Bidirectional Voronoi-biased Exploration curriculum for Reinforcement learning (BVER), which expands from both ends at once. Inspired by bidirectional RRT planning, BVER grows start states outward from the goal and goals outward from the initial state distribution, biases both toward unexplored task space, and steers them toward each other, training one goal-conditioned policy on both. On point-mass mazes, quadrupedal box
    
[^56]: 从修补到剪枝：视觉语言模型中的视觉计算

    From Patching to Pruning Visual Computation in Vision Language Models

    [https://arxiv.org/abs/2610.03389](https://arxiv.org/abs/2610.03389)

    P2P是一种免训练框架，将激活修补从诊断工具转变为推理时的计算旁路，在不删除视觉token、不修改模型权重的前提下剪枝视觉语言模型中不必要的视觉计算，从而降低推理成本。

    

    视觉语言模型（VLM）会产生巨大的推理成本，因为每个视觉token都要经过每一个解码器层的注意力机制和MLP投影处理，即使许多深度层级并不需要针对特定token的视觉计算。我们提出了Patch-to-Prune（P2P），这是一个受机械可解释性启发的免训练框架，它将激活修补从一种诊断工具转变为推理时的计算旁路。P2P通过验证引导的前向和后向层扫描，识别出解码器中视觉token投影输出可以在用户指定的精度容差内被固定的中性代理激活向量替代的区域。与传统的token剪枝方法不同，P2P保留了序列长度、token顺序、位置信息、注意力掩码和残差通路，从而在不删除token或不修改预训练模型权重的情况下剪枝计算。我们在四个V……

    arXiv:2610.03389v1 Announce Type: cross  Abstract: Vision language models (VLMs) incur substantial inference cost because every visual token is processed by the attention and MLP projections of every decoder layer, even when token-specific visual computation is unnecessary at many depths. We introduce Patch-to-Prune (P2P), inspired by Mechanistic Interpretability, a training-free framework that converts activation patching from a diagnostic tool into an inference-time computation bypass. P2P performs validation-guided forward and backward layer sweeps to identify decoder regions whose visual-token projection outputs can be replaced by fixed neutral proxy activation vectors within a user-specified accuracy tolerance. Unlike conventional token-pruning methods, P2P preserves the sequence length, token order, positional information, attention mask, and residual pathways, thereby pruning computation without removing tokens or modifying the pretrained model weights. We evaluate P2P on four V
    
[^57]: 面向傅里叶特征物理信息神经网络的算子知情初始化方法

    Operator-informed initialization for Fourier features physics-informed neural networks

    [https://arxiv.org/abs/2610.03378](https://arxiv.org/abs/2610.03378)

    本文通过在NTK理论框架下分析傅里叶特征物理信息神经网络的训练动力学，提出了一种根据所求解偏微分方程的微分算子特性来定制初始权重分布的初始化策略，从而消除算子导致的频谱偏差并提升预测精度。

    

    物理信息神经网络（PINNs）通常表现出频谱偏差，即目标函数的某些频率分量比其他频率收敛得更慢。为了解决这一局限性，本工作在神经正切核（NTK）理论框架下分析了傅里叶特征PINNs的训练动力学。我们推导出一个显式的演化方程来估计频域中的残差误差，证明特定频率的收敛速率主要由微分算子的符号与初始化权重的谱密度二者的乘积所决定。基于这一理论洞见，我们提出了一种信息丰富的初始化策略，使初始权重分布能够针对所求解的特定偏微分方程进行定制。通过该方法，我们可以减弱由算子引起的频谱偏差，平衡整个频谱上的收敛速率，从而获得更好的预测精度。

    arXiv:2610.03378v1 Announce Type: new  Abstract: Physics-Informed Neural Networks (PINNs) typically exhibit spectral bias, where some frequencies of the target function converge more slowly than others. In this work, we analyze the training dynamics of Fourier Feature PINNs in the Neural Tangent Kernel regime to address this limitation. We derive an explicit evolution equation to estimate the residual error in the frequency domain, demonstrating that the convergence rate of specific frequencies is primarily governed by the product of the differential operator's symbol and the spectral density of the initialization weights. Leveraging this theoretical insight, we propose an informative initialization strategy that tailors the initial weight distribution to the specific PDE being solved. With this method, we can diminish the operator-induced spectral bias, balancing the convergence rates across the frequency spectrum and achieving better prediction accuracy. Numerical experiments on line
    
[^58]: SCAD：面向长时程智能体的结构化信用分配与蒸馏

    SCAD: Structured Credit Assignment and Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2610.03372](https://arxiv.org/abs/2610.03372)

    SCAD通过将长时程智能体的交互分解为规划与有界子任务执行，并将基于结果的信用分配与教师引导蒸馏相结合，在文本和多模态任务上显著优于最强训练基线。

    

    训练长时程智能体解决复杂任务需要对长交互序列进行有效监督。然而，稀疏的终端奖励掩盖了中间步骤的贡献，而同策略蒸馏随着学生自身生成历史的增长可能会丢失有价值的教师指导。为解决这一问题，我们提出了SCAD，该方法将交互组织为规划与有界的子任务执行，在局部上下文中对执行进行蒸馏，并通过跨采样轨迹的子任务前缀树来细化规划信用，其中规划获得完整的终端信用，执行获得正向终端信用与教师指导。在所有评估基准上，SCAD相比最强训练基线，在文本任务上将宏平均准确率提升了4.48个百分点，在多模态任务上提升了4.19个百分点。SCAD有效地将基于结果的信用分配与教师引导的蒸馏相结合，从而提升长时程智能体的规划与执行能力。

    arXiv:2610.03372v1 Announce Type: new  Abstract: Training long-horizon agents to solve complex tasks requires effective supervision over extended interaction sequences. However, sparse terminal rewards obscure intermediate contributions, while on-policy distillation can lose informative teacher guidance as student-generated histories grow. To address this problem, we introduce SCAD, which organizes interactions into planning and bounded subtask execution, distills execution in local contexts, and refines planning credit through cross-rollout subtask prefix trees, with planning receiving full terminal credit and execution receiving positive terminal credit and teacher guidance. Across all evaluated benchmarks, SCAD improves macro-average accuracy over the strongest training baseline by 4.48 percentage points for text tasks and 4.19 points for multimodal tasks. SCAD effectively combines outcome-based credit assignment with teacher-guided distillation to improve planning and execution in 
    
[^59]: 面向加密货币订单执行的专家混合模型：训练稳定性、尾部风险与失效模式

    Mixture-of-Experts for Cryptocurrency Order Execution: Training Stability, Tail Risk, and Failure Modes

    [https://arxiv.org/abs/2610.03369](https://arxiv.org/abs/2610.03369)

    该论文在BTC/USDT订单簿数据上系统评估了专家混合（MoE）强化学习用于订单执行的效果，发现K≥4的MoE架构能显著提升训练稳定性并消除原始DDQL中拖延至被迫清算的失效模式，但没有任何学习型配置能在平均执行缺口上超越TWAP和立即清算等简单基准。

    

    用于订单执行的深度强化学习策略在不同训练随机种子之间可能存在显著差异，因此表面上由架构带来的收益，可能反映的只是有利的训练实现，而非该架构可复现的固有特性。我们在来自币安的5分钟均值聚合BTC/USDT限价订单簿数据上，评估了原始双重深度Q学习（DDQL）、采用K均值划分的DDQL专家混合模型（K∈{2,4,8}），以及与K=4和K=8专家预算参数量相匹配的稠密网络。结果显示，没有任何学习型配置在平均执行缺口上显著优于DDQL。在所报告的设定下，在一个无摩擦回放与终端紧迫性惩罚使得提前清算几乎零成本的环境中，所有方法的平均执行缺口均高于TWAP（0.39个基点）和立即清算（0.21个基点）；原始DDQL的100次运行中有11次收敛到一种等待直至被迫清算的策略，而在两个K≥4的MoE组中均未出现此现象……（摘要至此截断）

    arXiv:2610.03369v1 Announce Type: cross  Abstract: Deep reinforcement-learning policies for order execution can vary substantially across training seeds, so apparent architectural gains may reflect favourable training realisations rather than reproducible properties of the architecture. We evaluate vanilla Double Deep Q-Learning (DDQL), K-means-partitioned mixtures of DDQL experts at $K \in \{2, 4, 8\}$, and dense networks parameter-matched to the $K{=}4$ and $K{=}8$ expert budgets on 5-minute mean-aggregated BTC/USDT limit order book data from Binance. No learned configuration significantly improves mean implementation shortfall over DDQL. Under the reported specification, all have higher mean shortfall than TWAP (0.39 bps) and immediate liquidation (0.21 bps) in an environment whose frictionless replay and terminal-urgency penalty make early liquidation nearly costless; 11/100 vanilla-DDQL runs, versus none in either MoE $K{\geq}4$ arm, converge to a policy that waits until forced li
    
[^60]: 跟随赢家：基于交叉熵方法的无批评家强化微调中的保守策略改进

    Follow the Winners: Conservative Policy Improvement with the Cross-Entropy Method for Critic-Free RFT

    [https://arxiv.org/abs/2610.03361](https://arxiv.org/abs/2610.03361)

    FTW 是一种无批评家的强化微调算法，通过将交叉熵方法适配到 RFT 中、用回放缓冲区样本上的序数过滤器替代组采样，从而在有状态环境中难以重复采样的智能体大模型训练中实现保守且稳健的策略改进。

    

    面向智能体大语言模型的无批评家强化微调（RFT）通常采用 GRPO 风格的方法，即在重复的轨迹采样上计算组基线以降低目标方差。然而，这种设置并不适合在有状态环境（如在线服务或安全沙箱）中行动的智能体，因为在这些环境中难以获得重复采样，且激进的策略更新会将冗长且稀疏验证轨迹中的噪声固化下来。我们提出了“跟随赢家”，这是一种无批评家的策略学习算法，它将交叉熵方法适配到强化微调中，用基于经验回放缓冲区样本的序数过滤器替代组采样，从而在回报的顺序统计量上获得多项式集中性保证。我们通过“控制即推断”的视角推导出 FTW，该框架同时将 GRPO 和 DPO 还原为特定的建模选择：GRPO 被识别为风险中性的，而 DPO 与 FTW 共享一个有界的风险寻求偏移，FTW 对该偏移加以控制……

    arXiv:2610.03361v1 Announce Type: cross  Abstract: Critic-free reinforcement fine-tuning (RFT) for agentic large language models is often done through GRPO-style methods, which compute a group baseline over repeated rollouts to reduce target variance. However, this setup is ill-suited to agents acting in stateful environments such as live services or security sandboxes, where repeated rollouts are impractical to obtain and aggressive updates entrench the noise of long, sparsely verified trajectories. We propose \textit{Follow the Winners} (FTW), a critic-free policy-learning algorithm that adapts the cross-entropy method to RFT, replacing group rollouts with an ordinal filter on replay-buffer samples that yields polynomial concentration in the order statistic of returns. We derive FTW through a control-as-inference lens, which also recovers GRPO and DPO as specific modelling choices, identifying GRPO as risk-neutral while DPO and FTW share a bounded risk-seeking offset that FTW control
    
[^61]: 亲和学习：面向相关数据的分布式训练

    Cordial Learning: Distributed Training with Correlated Data

    [https://arxiv.org/abs/2610.03330](https://arxiv.org/abs/2610.03330)

    提出了一种名为“亲和学习”的分布式训练框架，通过智能体间仅共享低维输出、本地模型提取同伴信息来处理相关数据问题，并在线性模型假设下证明了其以概率一收敛到全局最优。

    

    我们研究了一个由拥有相关数据的智能体组成的分布式学习任务。具体而言，某个智能体的标签取决于其他智能体对同一样本的输入，且这些输入彼此之间也是相关的。当智能体共享同一环境时，相关数据是普遍存在的现实情况。现有的去中心化方法（如联邦学习）忽略了问题的结构，在相关数据上表现不佳；而另一方面，由于隐私和通信约束，集中式方法又不可行。我们提出了亲和学习（cordial，即相关与分布式学习）来弥补这一空白：智能体之间仅共享低维输出，同时训练本地模型以从同伴处提取有信息量的信号。这种分布式学习引发了一个博弈，其中每个智能体的损失函数依赖于其他智能体的模型。在线性模型假设下，我们证明了亲和学习以概率一收敛到全局最优解。

    arXiv:2610.03330v1 Announce Type: cross  Abstract: We consider a distributed learning task with agents that have correlated data. Specifically, the label of an agent depends on the input of other agents for the same sample, and these inputs are also correlated. Correlated data is the reality when agents share the same environment. Existing decentralized methods, such as federated learning, ignore the structure of the problem and perform poorly on correlated data. On the other hand, centralized approaches are infeasible due to privacy and communication constraints. We introduce cordial (correlated and distributed) learning to address this gap by sharing only low-dimensional outputs between the agents while training local models to extract informative signals from peers. This distributed learning induces a game in which the loss function of each agent depends on the models of others. Assuming a linear model, we prove that cordial learning converges with probability one to a globally opti
    
[^62]: SyntaxBench：大语言模型字符级推理的统计诊断框架

    SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models

    [https://arxiv.org/abs/2610.03329](https://arxiv.org/abs/2610.03329)

    提出SyntaxBench诊断基准，通过五个核心字符级任务和一个高难度子串提取压力测试，结合Cohen's kappa与McNemar检验等统计方法，系统评估了八个开放权重大语言模型的字符级推理能力。

    

    大语言模型越来越多地被应用于小语法错误也至关重要的场景，然而字符级推理目前主要还是通过孤立的探测任务和聚合准确率来进行评估。我们提出了SyntaxBench，一个面向字符级推理的诊断基准和统计评估框架。它包含五个核心任务：字符计数、字母包含检测、回文检测、编辑距离和最长字符串选择，外加一个更困难的子串提取压力测试index_to_span。五个核心任务使用成对的英文输入与字符长度匹配的随机字符串输入；index_to_span文档共享200-500词的长度区间，但不进行字符长度匹配。全部六个任务均采用零样本、单样本和四样本提示。我们在11种推理模式配置下评估了从2B到32B参数的八个开放权重模型。该框架报告精确匹配与宽松准确率、Cohen's kappa系数、带优势比的配对McNemar检验，

    arXiv:2610.03329v1 Announce Type: cross  Abstract: Large language models are increasingly used where small syntactic errors matter, yet character-level reasoning is still evaluated mostly through isolated probes and aggregate accuracy. We introduce SyntaxBench, a diagnostic benchmark and statistical evaluation framework for character-level reasoning. It contains five core tasks, character counting, letter containment, palindrome detection, edit distance, and longest-string selection, plus index_to_span, a harder substring-extraction stress test. The five core tasks use paired English and character-length-matched random-string inputs. index_to_span documents share a 200-500 word band and are not character-length matched. All six tasks use zero-, one-, and four-shot prompts.   We evaluate eight open-weight models from 2B to 32B parameters across 11 reasoning-mode configurations. The framework reports exact-match and relaxed accuracy, Cohen's kappa, paired McNemar tests with odds ratios, 
    
[^63]: DAWIS：基于多任务插值子的窗口化逆采样数据同化

    DAWIS: Data Assimilation with Windowed Inverse Sampling via Multitask Interpolants

    [https://arxiv.org/abs/2610.03314](https://arxiv.org/abs/2610.03314)

    DAWIS提出了一种统一的数据同化框架，通过用覆盖连续状态窗口的多任务随机插值子替代单一流时间先验，使滤波、固定滞后平滑和分块平滑得以在同一框架内实现，从而能在新观测到达时修正过去状态并避免误差累积。

    

    基于流和扩散的生成模型最近已成为动力系统中灵活且高效的预测模型。当与推理时引导相结合时，它们为高维非高斯数据同化（DA）提供了一条有前景的途径——数据同化是将预测与观测相结合以估计潜在系统状态的问题。然而，现有的滤波器以固定的历史为条件，仅同化最新的观测结果，因此无法在新观测到达时修正过去的状态。这导致估计结果被束缚在与后续观测可能相矛盾的历史上，误差在同化运行过程中不断累积。为此，我们提出了DAWIS，这是一种统一的数据同化方法，在单一框架内涵盖了滤波、固定滞后平滑和分块平滑。DAWIS用跨越连续状态窗口的多任务随机插值子替代状态级先验的单一流时间……（摘要未完）

    arXiv:2610.03314v1 Announce Type: cross  Abstract: Flow- and diffusion-based generative models have recently emerged as flexible and highly efficient forecasting models for dynamical systems. When combined with inference-time guidance, they offer a promising route to high-dimensional non-Gaussian data assimilation (DA), the problem of combining forecasts with observations to estimate latent system states. Existing filters, however, condition on a fixed history and assimilate only the most recent observation, leaving them unable to revise past states when new observations arrive. Estimates then stay tethered to a history that later observations may contradict, and errors accumulate over the assimilation run. To this end, we introduce **DAWIS**, a unified DA method covering filtering, fixed-lag smoothing, and block smoothing within a single framework. DAWIS replaces the single flow time of a state-level prior with a multitask stochastic interpolant over a window of consecutive states, as
    
[^64]: SDECast：基于神经随机微分方程的连续时间概率天气预报

    SDECast: Probabilistic Weather Forecasting in Continuous Time with Neural SDEs

    [https://arxiv.org/abs/2610.03313](https://arxiv.org/abs/2610.03313)

    SDECast提出了一个基于神经随机微分方程的连续时间概率天气预报框架，无需训练时重复SDE模拟即可直接在物理空间学习随机动力学，并成功扩展到小时级分辨率的全球天气预报。

    

    现有的机器学习天气预报模型通常通过固定时间分辨率的自回归滚动方式生成预报。虽然这种方式对于长期预测非常高效，但在使用较短时间步长时可能出现严重的误差累积，并且没有显式地编码大气动力学的局部性与时间连续性。为了解决这些局限，我们提出了SDECast，一个用于连续时间概率天气预报的神经随机微分方程（Neural SDE）框架。SDECast扩展了SDE Matching方法，直接在物理空间中学习随机动力学，而无需在训练过程中进行重复的SDE模拟。在一个模拟的地球物理流动实验中，我们展示了SDECast能够恢复有意义的漂移动力学，并忠实地再现了底层的连续时间行为。随后，我们证明了该方法在小时分辨率全球天气预报上的可扩展性。

    arXiv:2610.03313v1 Announce Type: cross  Abstract: Existing machine learning weather forecasting models typically generate forecasts through autoregressive rollouts at a fixed temporal resolution. While highly efficient for long-range prediction, this formulation can suffer from severe error accumulation when used with shorter time steps and does not explicitly encode the locality and temporal continuity of atmospheric dynamics. To address these limitations, we introduce **SDECast**, a Neural Stochastic Differential Equation (SDE) framework for continuous-time probabilistic weather forecasting. SDECast extends SDE Matching to learn stochastic dynamics directly in physical space, without requiring repeated SDE simulation during training. On a simulated geophysical flow, we show that SDECast recovers meaningful drift dynamics and faithfully reproduces the underlying continuous-time behavior. We then demonstrate its scalability to global weather forecasting at hourly resolution, where SDE
    
[^65]: 基于有限步牛顿-舒尔茨正交化的Muon训练损失保证

    Training-Loss Guarantees for Muon with Finite-Step Newton--Schulz Orthogonalization

    [https://arxiv.org/abs/2610.03306](https://arxiv.org/abs/2610.03306)

    本文首次为Muon优化器建立了同时考虑动量累积与有限步调优牛顿-舒尔茨正交化的训练损失保证，证明了在具有正定极限神经切向核的宽两层ReLU网络上，Muon能以高概率达到任意目标损失，命中时间界为 $O((1-\mu)^{-1}\varepsilon^{-1/2})$。

    

    现有的Muon收敛性分析要么假设精确的正交化，要么分析经典的牛顿-舒尔茨（Newton–Schulz）多项式，并且仅保证平稳性，因此Muon的五个经过调优的牛顿-舒尔茨步骤究竟保留了什么，以及这是否足以达到预设的神经网络训练损失，仍是悬而未决的问题。我们建立了一个有限时间的训练保证，同时考虑了正交化之前的动量累积以及经过调优的有限步更新。对于具有固定随机输出权重和正定极限神经切向核的足够宽的两层ReLU网络的全批量训练，我们证明了Muon在初始化上以高概率达到任意目标经验平方损失 $\varepsilon>0$。对于每个动量参数 $\mu\in[0,1)$，采用与 $(1-\mu)\sqrt{\varepsilon}$ 成比例的、依赖于目标的恒定学习率，可以得到 $O((1-\mu)^{-1}\varepsilon^{-1/2})$ 的命中时间界，其余……（摘要原文在此截断）

    arXiv:2610.03306v1 Announce Type: cross  Abstract: Existing convergence analyses of Muon either assume exact orthogonalization or analyze classical Newton--Schulz polynomials, and guarantee only stationarity, so it is unresolved what Muon's five tuned Newton--Schulz steps preserve and whether that suffices to reach a prescribed neural-network training loss. We establish a finite-time training guarantee that accounts for both momentum accumulation before orthogonalization and the tuned finite-step update. For full-batch training of a sufficiently wide two-layer ReLU network with fixed random output weights and a positive-definite limiting neural tangent kernel, we prove that Muon reaches any target empirical squared loss $\varepsilon>0$ with high probability over initialization. For every momentum parameter $\mu\in[0,1)$, a target-dependent constant learning rate proportional to $(1-\mu)\sqrt{\varepsilon}$ yields a hitting-time bound of $O((1-\mu)^{-1}\varepsilon^{-1/2})$, with other pr
    
[^66]: S²-PINN：随机可分离物理信息神经网络

    S$^{2}$-PINN: Stochastic Separable Physics-Informed Neural Networks

    [https://arxiv.org/abs/2610.03303](https://arxiv.org/abs/2610.03303)

    提出S²-PINN，利用可学习高斯空间字典、傅里叶时间特征与gPC随机基并通过低秩CP张量分解耦合的可分离表示，高效求解随机偏微分方程的不确定性量化问题，克服了经典谱方法的维数灾难。

    

    随机偏微分方程（PDE）的不确定性量化（UQ）在计算科学与工程中无处不在。然而，求解这类问题的经典谱方法面临维数灾难，而现有的神经求解器往往忽略了随机结构，无法使矩估计与校准变得易于处理。我们提出了一种随机可分离物理信息神经网络，称为S²-PINN，它通过可学习的高斯空间字典、傅里叶时间特征以及广义多项式混沌（gPC）随机基来表示随机PDE的解 u(t,x,Z)，并通过低秩Canonical Polyadic（CP）张量分解核心将三者耦合起来。该方法采用强形式与gPC投影残差相结合的混合损失进行训练。我们的理论分析证明，在温和条件下，该可分离函数类在L²空间中是稠密的，且投影残差恰好对应于……

    arXiv:2610.03303v1 Announce Type: new  Abstract: Uncertainty quantification (UQ) for random partial differential equations (PDEs) is ubiquitous in computational science and engineering. However, classical spectral solvers for this class of problems face the curse of dimensionality, and existing neural solvers often ignore the stochastic structure that makes moments and calibration tractable. We introduce a stochastic separable physics-informed neural network, dubbed S$^{2}$-PINN, that represents the solution $u(t,\mathbf{x},\mathbf{Z})$ of a random PDE with a learnable Gaussian spatial dictionary, Fourier temporal features, and a generalized polynomial chaos (gPC) stochastic basis, coupled by a low-rank Canonical Polyadic (CP) tensor decomposition core. The method is trained with a hybrid strong-form and gPC-projected residual loss. Our theoretical analysis establishes that the separable class is dense in $L^2$ under mild conditions, and the projected residual corresponds exactly to a 
    
[^67]: JOVE：面向资源感知LLM任务图的联合执行与验证框架

    JOVE: Joint Execution and Verification for Resource-Aware LLM Task Graphs

    [https://arxiv.org/abs/2610.03296](https://arxiv.org/abs/2610.03296)

    JOVE提出了一种在线框架，通过联合决策LLM执行分配与中间输出的付费验证，在长期预算和延迟约束下平衡即时执行开销与未来学习收益，从而在LLM服务质量未知的情况下提升任务图执行的效率与正确性。

    

    复杂推理查询可以被分解为有向无环任务图，并分发到异构的大语言模型（LLM）上执行，通过并行化降低延迟，并使较小的模型也能解决复杂任务。然而在实践中，某个LLM是否适合给定的子任务可能是先验未知的，而且仅凭执行本身无法揭示输出的正确性。我们提出了JOVE，一个在线框架，它联合地为子任务分配执行LLM，并选择中间输出进行付费验证。验证以异步方式运行，并被用于改进未来的分配决策，因此系统必须在“当下的执行开销”与“为未来而学习”之间取得平衡。我们研究了在长期预算和单查询延迟约束下如何优化这一权衡，其中LLM的服务质量、调用成本和执行时间均是随机的且初始未知。JOVE通过求解一系列单查询混合整数线性规划来做出执行与验证决策。

    arXiv:2610.03296v1 Announce Type: new  Abstract: Complex reasoning queries can be decomposed into directed acyclic task graphs and distributed across heterogeneous LLMs, reducing latency through parallelism and enabling smaller models to solve complex tasks. In practice, however, the suitability of an LLM for a given subtask may be a priori unknown, and execution alone does not reveal output correctness. We propose JOVE, an online framework that jointly assigns executor LLMs and selects intermediate outputs for paid verification. Verification runs asynchronously and is used to improve future allocations, so the system must balance spending on execution now against learning for later. We study how to optimize this trade-off under a long-term budget and a per-query latency constraint, with stochastic, initially unknown LLM service quality, invocation costs, and execution times. JOVE makes execution and verification decisions by solving a sequence of per-query mixed-integer linear program
    
[^68]: 器官不同，物理相同：将超声心动图预训练迁移至肺部超声用于结核病筛查

    Wrong Organ, Right Physics: Transferring Echocardiography Pretraining to Lung Ultrasound for Tuberculosis Screening

    [https://arxiv.org/abs/2610.03290](https://arxiv.org/abs/2610.03290)

    本研究发现在低资源肺部超声结核病筛查任务中，编码器的预训练领域选择（超声心动图还是通用视频）对分类性能影响甚微，而特征条件化才是决定任务表现的关键因素。

    

    肺部超声（LUS）在初级保健层面用于结核病（TB）筛查颇具吸引力，但标注数据集规模较小。超声心动图则不受此限制，且与肺部超声共享相同的底层超声成像物理原理、信号处理方式和B模式图像外观。我们探究在该高资源超声领域上预训练的编码器，其所携带的表征在低资源领域中是否仍然可用。实验中仅编码器有所变化，共涉及横跨三种架构家族的十七个编码器。其中，在通用视频上预训练的潜在预测视频编码器（V-JEPA2-L）与其超声心动图对应版本（EchoJEPA-L）仅在预训练语料上存在差异。这些编码器之间的选择并不能解决分类问题——整个编码器家族的性能差距仅为2.50个百分点，而测量分辨率为2.71个百分点。真正推动任务表现的是特征条件化（feature conditioning），即对编码器与后续模块之间的特征进行标准化处理。（注：原文摘要在此处截断）

    arXiv:2610.03290v1 Announce Type: cross  Abstract: Lung ultrasound (LUS) is attractive for tuberculosis (TB) screening at primary-care level, but labelled cohorts are small. Echocardiography carries no such constraint, while sharing the same underlying ultrasound imaging physics, signal processing and B-mode appearance as LUS. We ask whether an encoder pretrained on that high-resource ultrasound domain carries representations that remain usable in the low-resource one. Only the encoder varies, across seventeen encoders spanning three architecture families. Among them, a latent-predictive video encoder pretrained on generic video (V-JEPA2-L) and its echocardiography counterpart (EchoJEPA-L) differ in pretraining corpus alone. The choice among these encoders does not resolve the classification, the whole family spanning 2.50 percentage points against a measurement resolution of 2.71. What moves the task instead is feature conditioning. Standardising the features between the encoder and t
    
[^69]: 多模态大语言模型中架构依赖的融合路径

    Architecture-Dependent Fusion Pathways in MLLMs

    [https://arxiv.org/abs/2610.03289](https://arxiv.org/abs/2610.03289)

    本文通过对两种架构范式的多模态大语言模型进行系统性分析和因果干预实验，揭示了其内部模态融合机制的根本差异，发现拼接架构模型遵循“文本优先、视觉靠后”的融合路径，而原生多模态架构则呈现不同的融合模式。

    

    多模态大语言模型（MLLMs）在视觉-语言任务中取得了强大的性能，然而视觉信息和文本信息在各层之间融合的内部机制仍未被充分理解。我们研究了来自两种架构范式的代表性MLLMs：拼接架构和原生多模态架构。我们开展了三项逐步关联的分析：对齐解耦识别哪个模态发生变化，注意力路由和熵刻画跨模态信息如何分布，内在维度则研究融合如何重塑特征空间。此外，我们进行了因果干预实验作为对所得解释的验证。作为补充分析，我们使用视觉CKA方法检验了柏拉图表示假说。这些分析共同揭示了两种截然不同的融合路径：拼接模型遵循“文本优先、视觉靠后”的路径……

    arXiv:2610.03289v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) achieve strong performance across vision-language tasks, yet the internal mechanisms by which visual and textual information are fused across layers remain insufficiently understood. We investigate representative MLLMs from two architectural paradigms: concatenation architectures and native multimodal architectures. We conduct three progressively connected analyses: alignment decoupling identifies which modality changes, attention routing and entropy characterize how cross-modal information is distributed, and intrinsic dimensionality examines how fusion reshapes feature spaces. Separately, we perform causal intervention experiments as a validation of the resulting interpretation. As a supplementary analysis, we use visual CKA to examine the Platonic Representation Hypothesis. Together, these analyses reveal two distinct fusion pathways: concatenation models follow a text-first, vision-later pathw
    
[^70]: SPEAR：面向大规模偏微分方程预训练的谱解耦混合专家神经算子与知识引导专家聚合

    SPEAR: A Spectral-Disentangled MoE Neural Operator with Knowledge-Guided Expert Aggregation for Large-Scale PDE Pretraining

    [https://arxiv.org/abs/2610.03265](https://arxiv.org/abs/2610.03265)

    SPEAR通过将特征谱解耦为低频与高频分量以实现共享与专门化建模，并结合基于数据集知识与路由偏好的知识引导专家聚合策略来消除专家冗余，有效解决了PDE基础模型中的知识干扰与专家冗余问题，提升了大规模PDE预训练的泛化能力。

    

    大规模预训练提升了神经算子在多种偏微分方程（PDE）上的泛化能力。然而，现有的PDE基础模型在应对异质动力学时仍存在困难：共享表示可能引发知识干扰，而混合专家（MoE）架构则面临专家冗余日益严重的问题。我们提出了SPEAR，一个用于大规模PDE预训练、具有知识引导专家聚合机制的谱解耦MoE神经算子。SPEAR将潜在特征解耦为低频与高频分量，从而实现对可迁移动力学的共享建模，以及对PDE特定模式的专门化学习。为解决专家冗余问题，我们设计了一种知识引导的专家聚合策略，该策略基于数据集特定的已学习知识和路由偏好来度量专家之间的相似性，从而实现对相似专家的识别与合并。在十二个PDE数据集及多个下游任务上的实验验证了该方法的有效性。

    arXiv:2610.03265v1 Announce Type: cross  Abstract: Large-scale pre-training has improved the generalization of neural operators across diverse PDEs. However, existing PDE foundation models still struggle with heterogeneous dynamics, where shared representations may cause knowledge interference, while mixture-of-experts (MoE) architectures suffer from increasing expert redundancy. We propose SPEAR, a spectral-disentangled MoE neural operator with knowledge-guided expert aggregation for large-scale PDE pre-training. SPEAR decouples latent features into low- and high-frequency components, enabling shared modeling of transferable dynamics and specialized learning of PDE-specific patterns. To address expert redundancy, we design a knowledge-guided expert aggregation strategy that measures expert similarity from dataset-specific learned knowledge and routing preferences, enabling the identification and consolidation of similar experts. Experiments on twelve PDE datasets and multiple downstre
    
[^71]: PaMIR：公开信贷违约数据集的开放基准

    PaMIR: Open Benchmark of Public Credit-Default Datasets

    [https://arxiv.org/abs/2610.03259](https://arxiv.org/abs/2610.03259)

    PaMIR是首个汇集19个公开信贷违约数据集（来自九个国家的124万条记录）的开放基准，采用统一的无泄漏构建流程和按标签预算报告的标签延迟流评估协议，为标签稀缺且延迟的信贷违约预测提供可复现的评测标准。

    

    我们发布了PaMIR（Public Arrival-ordered Measurement for Inference in Risk，面向风险推断的按到达顺序公开测量），这是一个针对标签稀缺且延迟到达情形下信贷违约预测的开放基准。该领域的参考基准研究各自使用八个数据集，其中仅有两到四个是公开的。PaMIR汇集了19个带有二元违约标签的公开数据集——来自九个国家的124万条贷款、企业和信用卡账户记录——全部通过一套经过泄漏审计的统一流程从固定的源数据快照重建，且从不重新分发；据我们所知，这是迄今为止同类中唯一的基准。每个模型都是单一函数，在重复的独立同分布划分和标签延迟流两种设置下进行评分，其中每个申请在到达时即被评分，并按标签预算报告AUC；除非对所有数据集都完成了评分，否则不公布整体均值。一个合成数据测试框架可以在不让生成器接触保留数据行的情况下测试生成的训练数据行。本报告描述了这一持续更新基准的0.4.0版本。

    arXiv:2610.03259v1 Announce Type: new  Abstract: We release PaMIR (Public Arrival-ordered Measurement for Inference in Risk), an open benchmark for credit-default prediction when labels are scarce and arrive late. The field's reference benchmark studies use eight datasets each, only two or four of them public. PaMIR brings together 19 public datasets with binary default labels -- 1.24M loans, firms and card accounts from nine countries -- rebuilt from pinned source snapshots by one leakage-audited recipe and never redistributed; to our knowledge it is the one of its kind as of today. Every model is a single function, scored under a repeated i.i.d. split and a label-delayed stream in which each application is scored on arrival, with AUC reported by label budget; fleet means are withheld unless every dataset is scored. A synthetic-data harness tests generated training rows without letting a generator see held-out rows. This report describes release 0.4.0 of this living benchmark.
    
[^72]: 绘制并推进非线性因果发现的可扩展性-准确性前沿

    Mapping and Advancing the Scalability-Accuracy Frontier of Nonlinear Causal Discovery

    [https://arxiv.org/abs/2610.03258](https://arxiv.org/abs/2610.03258)

    本文系统比较了四类非线性因果发现方法在可扩展性与准确性上的互补瓶颈，并提出基于样条的得分评估方案SPADE，通过一次性编译并复用充分统计量，在保持准确性的同时大幅提升组合搜索的效率。

    

    可扩展的非线性因果发现需要将灵活的机制估计器与对大型图空间的高效搜索相结合的方法。学界已提出多种算法家族来应对这一挑战，但它们的准确性-运行时间权衡仍缺乏深入理解。我们实证比较了四种主要方法：可微结构学习、摊销结构学习、得分匹配和组合搜索。我们的结果揭示了互补的瓶颈：可微方法和摊销方法具有良好的可扩展性，但存在准确性差距；得分匹配方法在低维情况下可以较为准确，但随着特征维度增加会迅速退化；组合搜索方法保持准确，但因重复且冗余的局部评分而速度受限。基于这一瓶颈，我们开发了SPADE，一种基于样条的得分评估方案，它一次性编译充分统计量，并在整个组合搜索过程中重复使用。

    arXiv:2610.03258v1 Announce Type: cross  Abstract: Scalable nonlinear causal discovery requires methods that combine flexible mechanism estimators with efficient search over large graph spaces. Several algorithmic families have been proposed to address this challenge, yet their accuracy-runtime trade-offs remain poorly understood. We empirically compare the four major approaches: differentiable structure learning, amortized structure learning, score-matching, and combinatorial search. Our results reveal complementary bottlenecks: differentiable and amortized methods scale well but exhibit an accuracy gap, score-matching methods can be accurate in low dimensions but degrade quickly for increasing feature sizes, and combinatorial methods remain accurate but are slowed by repeated and redundant local scoring. Motivated by this bottleneck, we develop SPADE, a spline-based score-evaluation scheme that compiles sufficient statistics once and reuses them throughout combinatorial search. Under
    
[^73]: 使用在乌干达和南非收集的临床数据进行跨队列结核病分类

    Cross-cohort TB classification using clinical data gathered in Uganda and South Africa

    [https://arxiv.org/abs/2610.03256](https://arxiv.org/abs/2610.03256)

    该研究首次评估了基于南非和乌干达两个国家临床数据的机器学习跨队列结核病筛查，并提出一种联合优化特征选择与特征排序的CNN策略，显著提升了筛查性能。

    

    我们首次评估了将机器学习应用于在两个不同国家收集的患者临床和人口统计数据，目的是进行结核病（TB）筛查，以识别能够从昂贵的分子检测中受益的人群。实验基于最近编制的CAGE-TB数据集，该数据集包含在南非和乌干达社区卫生保健中心就诊的疑似结核病患者亚队列。研究考虑了三种神经网络架构（逻辑回归（LR）、多层感知器（MLP）和卷积神经网络（CNN）），并结合贪婪特征选择方法。对于卷积神经网络，提出了一种联合优化特征选择和特征排序的策略，并被证明能够带来一致的开发集和测试集性能提升。对于所有三个模型，开发集的受试者工作特征曲线下面积（AUROC）提高了2-……

    arXiv:2610.03256v1 Announce Type: new  Abstract: We present a first evaluation of machine learning applied to patient clinical and demographic data gathered in two different countries for the purpose of tuberculosis (TB) screening to identify people who would benefit from expensive molecular testing. Experiments are based on the recently-compiled CAGE-TB dataset, which includes sub-cohorts of people with presumptive TB presenting at community health care centres in South Africa and Uganda. Three neural network architectures (logistic regression (LR), multilayer perceptrons (MLP) and convolutional neural networks (CNN)) are considered in conjunction with greedy feature selection. For the convolutional neural network, a strategy that jointly optimises feature selection and feature ordering is proposed and shown to lead to consistent development and test set improvements. For all three models, development set area under the receiver operating characteristic (AUROC) curve is improved by 2-
    
[^74]: D2K-Bench：LLM 智能体能否将专家设计转化为高效的 GPU 内核？

    D2K-Bench: Can LLM Agents Turn Expert Designs into Efficient GPU Kernels?

    [https://arxiv.org/abs/2610.03226](https://arxiv.org/abs/2610.03226)

    D2K-Bench 是一个包含 26 个任务和 85 个工作负载的诊断性基准，通过分层专家设计指导（算法洞察、数据流设计与底层优化技巧）来系统评估 LLM 智能体生成高效 GPU 内核的能力，结果显示专家指导可将正确率从 93.1% 提升至 98.5% 并显著提高性能。

    

    由大语言模型（LLM）智能体生成的 GPU 内核，其效率可能仍低于专家实现，但仅凭运行时间并不能揭示这一差距与设计发现和实现之间的关系。我们提出了 D2K-Bench，一个包含 26 个任务和 85 个工作负载的诊断性基准，用于衡量智能体将专家设计指导转化为高效 GPU 内核的能力。这些指导涵盖三个层级：L1 为高层算法洞察，L2 为数据流设计，L3 为底层优化技巧，并包含这些层级之间的依赖关系。有指导与无指导的成对运行共享相同的任务描述、工作负载、工具、硬件以及 350 轮的交互预算。补充性评估则考察智能体独立提出的设计，以及生成代码中实际实现的设计属性。在 NVIDIA B200 GPU 上对五个模型的评估中，专家指导将 130 个模型-任务对上的正确率从 93.1% 提升至 98.5%，并在全部 26 个任务上提升了性能得分……（原文摘要在此处截断）

    arXiv:2610.03226v1 Announce Type: cross  Abstract: GPU kernels generated by large language model (LLM) agents can remain less efficient than expert implementations, but runtime alone does not reveal how the gap relates to design discovery and implementation. We introduce D2K-Bench, a diagnostic benchmark of 26 tasks and 85 workloads that measures how effectively agents translate expert design guidance into efficient GPU kernels. The guidance covers L1: high-level algorithmic insights, L2: dataflow design, and L3: low-level optimization tricks, including dependencies among these levels. Pairwise runs with and without guidance share task descriptions, workloads, tools, hardware, and a 350-turn budget. Complementary assessments examine independently proposed designs and the design properties implemented in generated code. Across five models on NVIDIA B200 GPUs, guidance raises correctness over 130 model-task pairs from 93.1% to 98.5% and increases the Performance Score over all 26 tasks f
    
[^75]: 神经物理反演器：一种耦合集成条件化与残差学习的大地电磁反演模块化框架

    The Neuro-Physical Inverter: A Modular Framework for Magnetotelluric Inversion Coupling Ensemble Conditioning with Residual Learning

    [https://arxiv.org/abs/2610.03225](https://arxiv.org/abs/2610.03225)

    提出了一种模块化的神经物理反演器（NPI）框架，通过将集成条件高斯过程与残差学习神经网络相结合，实现了具备不确定性量化能力的大地电磁反演，在不破坏集成稳定性的情况下系统性降低误差。

    

    我们提出了神经物理反演器（NPI），这是一个模块化、不确定性感知的地球物理反演框架，它将基于集成的条件化方法与受约束的残差学习相结合，并在一维大地电磁（MT）环境中作为受控测试平台进行了演示。该框架分两个阶段运行：首先，集成条件高斯过程（EnsCGP）根据观测响应对电阻率模型的先验集成进行条件化，生成物理上可接受的参考集成；然后，残差学习神经网络预测针对该参考的校正，该网络在合成数据上训练，并通过物理耦合目标在野外应用中对每个测站进行微调。由于集成在两个阶段中均被条件化、细化和传播，因此每个估计都带有相应的集成离散度。合成实验表明，NPI在不破坏集成稳定性的前提下，系统性地降低了集成平均误差。应用于宽带（摘要在此处截断）。

    arXiv:2610.03225v1 Announce Type: new  Abstract: We present the Neuro-Physical Inverter (NPI), a modular, uncertainty-aware framework for geophysical inversion that couples ensemble-based conditioning with constrained residual learning, demonstrated in the 1D magnetotelluric (MT) setting as a controlled testbed. The framework operates in two stages. An Ensemble-Conditional Gaussian Process (EnsCGP) conditions a prior ensemble of resistivity models on the observed response, producing a physically admissible reference ensemble. A residual-learning neural network then predicts targeted corrections to this reference, trained on synthetic data and fine-tuned per station for field application through a physics-coupled objective. Because an ensemble is conditioned, refined, and propagated through both stages, every estimate carries an associated ensemble spread. Synthetic experiments show that NPI systematically reduces ensemble-mean error without destabilizing the ensemble. Applied to broadb
    
[^76]: AdaStep：面向智能体强化学习的自适应步骤信用加权方法

    AdaStep: Adaptive Step Credit Weighting for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.03223](https://arxiv.org/abs/2610.03223)

    提出AdaStep方法，将步骤信用加权建模为均方误差估计问题并推导出最优逐状态收缩系数，从而在稀疏奖励下可靠地融合步骤级与轨迹级监督信号，提升长程LLM智能体的强化学习训练效果。

    

    长程LLM智能体通常使用稀疏的结果奖励进行训练，这使得轨迹级目标过于粗糙，无法区分单个决策的贡献。步骤级信用分配提供了更细粒度的监督，但其估计可能不可靠，因为观测到的回报还依赖于后续动作、环境转移和轨迹长度。我们提出AdaStep，一种自适应步骤信用加权方法，用于控制每个由组比较导出的局部优势对轨迹级信号的修改强度。我们将该加权建模为对潜在步骤优势的均方误差估计问题，并在显式的条件采样假设下，推导出最优的逐状态收缩系数。该系数具有信号-总方差的解释：当回报变化可归因于所选动作时保留局部信用，当变化由其他因素主导时则将其抑制。

    arXiv:2610.03223v1 Announce Type: cross  Abstract: Long-horizon LLM agents are typically trained with sparse outcome rewards, making trajectory-level objectives too coarse to distinguish the contribution of individual decisions. Step-level credit assignment provides finer-grained supervision, but its estimates can be unreliable because observed returns also depend on subsequent actions, environment transitions, and trajectory length. We propose AdaStep, an Adaptive Step-credit weighting method that controls how strongly each group-derived local advantage modifies the trajectory-level signal. We formulate this weighting as a mean-squared-error estimation problem for the latent step advantage and, under an explicit conditional sampling assumption, derive an optimal per-state shrinkage coefficient. The coefficient admits a signal-to-total-variance interpretation: it preserves local credit when return variation is attributable to the selected action and suppresses it when variation is domi
    
[^77]: 基于惰性二阶预言机的近最优凸优化

    Near-Optimal Convex Optimization with Lazy Second-Order Oracles

    [https://arxiv.org/abs/2610.03222](https://arxiv.org/abs/2610.03222)

    本文针对惰性二阶预言机凸优化问题，通过新的分块零链下界构造和匹配的算法设计，将复杂度界改进至 $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$ 并在对数因子内紧致，显著优于先前结果。

    

    本文研究了使用惰性二阶预言机（Doikov、Chayti 和 Jaggi，ICML 2023）的凸优化复杂度问题，其中算法每次迭代查询梯度，每 $m$ 次迭代查询一次 Hessian 矩阵。在该设置下，我们通过一种新颖的分块零链构造，证明了找到 $\epsilon$-解所需总迭代次数的下界为 $\Omega(m+ m^{1/7} \epsilon^{-2/7})$。随后，我们提出了一种新方法，达到了新的上界 $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$，该结果显著改进了先前（Chen、Liu、Luo 和 Zhang，COLT 2026）的 $\tilde{\mathcal{O}}(m+ m^{13/21} \epsilon^{-2/7})$ 上界，并在仅相差对数因子的意义下是紧致的。

    arXiv:2610.03222v1 Announce Type: cross  Abstract: This paper studies the complexity of convex optimization using lazy second-order oracles (Doikov, Chayti, and Jaggi, ICML 2023), where an algorithm queries gradients every iteration and Hessians once per $m$ iterations. Under this setting, we show a lower bound of $\Omega(m+ m^{1/7} \epsilon^{-2/7})$ on the number of total iterations to find an $\epsilon$-solution using a novel block zero-chain construction. Then we propose a novel method that achieves a new upper bound of $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$, which significantly improves the prior one (Chen, Liu, Luo, and Zhang, COLT 2026) of $\tilde{\mathcal{O}}(m+ m^{13/21} \epsilon^{-2/7})$ and is tight up to logarithmic factors.
    
[^78]: 面向图像分类的混合量子-经典架构演化

    Evolving Hybrid Quantum-Classical Architectures for Image Classification

    [https://arxiv.org/abs/2610.03220](https://arxiv.org/abs/2610.03220)

    该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。

    

    混合量子-经典神经网络将参数化量子电路（PQC）与成熟的深度学习架构相结合，但其性能在很大程度上取决于量子电路架构的选择，而这一选择目前仍主要依赖人工完成。现有的大多数方法依赖于手工设计或固定的电路拟设，需要预先指定电路结构、门组合和量子比特连接方式，且无法保证这些设计适合特定任务。这一限制在图像分类任务中尤为突出，因为量子电路既要对经典网络提取的特征进行变换，又要保持足够紧凑以便于实际训练，而通用的、与任务无关的拟设很难同时满足这些要求。我们将EXAQC——一个用于自动化量子电路发现的演化框架——扩展到图像分类任务。EXAQC将参数化量子电路作为中间处理模块进行演化，同时保留经典……（摘要原文在此处截断）

    arXiv:2610.03220v1 Announce Type: cross  Abstract: Hybrid quantum classical neural networks integrate parameterized quantum circuits (PQCs) with established deep learning architectures, but their performance depends strongly on the choice of quantum circuit architecture, a choice that remains largely manual. Most existing approaches rely on hand-designed or fixed circuit ans\"atze, requiring circuit structure, gate composition, and qubit connectivity to be specified in advance with no guarantee that they suit the task. This limitation is especially acute in image classification, where quantum circuits must transform features extracted by classical networks while remaining compact enough for practical training, requirements that generic, task-agnostic ans\"atze are unlikely to satisfy simultaneously. We extend EXAQC, an evolutionary framework for automated quantum circuit discovery, to image classification. EXAQC evolves PQCs as intermediate processing modules while retaining classical 
    
[^79]: 核奇异值分解及其向多数据源的扩展

    Kernel Singular Value Decomposition with Extension to Multiple Data Sources

    [https://arxiv.org/abs/2610.03216](https://arxiv.org/abs/2610.03216)

    该论文提出eKSVD，将核奇异值分解从两个数据源扩展到多个数据源，通过对偶优化推广了Lanczos分解定理中的移位特征值问题，并给出了结合神经网络的基于协方差的框架，实现非对称核上的联合非线性特征学习。

    

    核奇异值分解（KSVD）学习相对于一个非对称核矩阵的一对奇异向量，该核矩阵可以由两个数据源诱导产生，例如自注意力机制中的查询与键，或者给定矩阵的行与列。在这项工作中，我们将KSVD扩展到多个数据源，即eKSVD，它在非对称核上进行联合非线性特征学习。在原始（primal）形式中，与每个数据源相关的投影被联合学习以捕获最大信息，同时纳入成对耦合关系。借助拉格朗日函数及其Karush-Kuhn-Tucker（KKT）条件，对偶形式的优化导出了KSVD的Lanczos分解定理中移位特征值问题的一个推广。此外，还推导了一个基于协方差的框架，并结合神经网络（NNs）用于显式特征映射，这与基于核的解释和优化形成互补。数值实验（摘要在此处截断）……

    arXiv:2610.03216v1 Announce Type: new  Abstract: Kernel Singular Value Decomposition (KSVD) learns a pair of singular vectors w.r.t. an asymmetric kernel matrix, which can be induced by two data sources, e.g., the queries and keys in self-attention or the rows and columns of a given matrix. In this work, we extend KSVD to multiple data sources, namely eKSVD, which conducts joint nonlinear feature learning upon asymmetric kernels. In the primal formulation, the projections associated with each data source are jointly learned to capture maximal information, while incorporating pair-wise couplings. With the Lagrangian and its Karush-Kuhn-Tucker (KKT) conditions, the optimization in the dual leads to a generalization of the shifted eigenvalue problem in Lanczos decomposition theorem of KSVD. Further, a covariance-based framework is derived together with using neural networks (NNs) for explicit feature mappings, complementary to the kernel-based interpretation and optimization. Numerical ex
    
[^80]: HyperFuse：面向属性超图的快速自监督节点嵌入

    HyperFuse: Fast Self-Supervised Node Embeddings for Attributed Hypergraphs

    [https://arxiv.org/abs/2610.03211](https://arxiv.org/abs/2610.03211)

    HyperFuse提出了一种无需标签的快速超图表示学习流水线，通过谱松弛坐标计算、多尺度特征摘要和仅训练100轮的轻量级编码器，大幅降低了属性超图节点嵌入的生成成本。

    

    自监督超图表示学习能够生成信息丰富的节点嵌入，但现有方法通常需要训练数百个轮次的深度编码器，即使对于仅有数千个节点的超图，嵌入生成的成本也十分高昂。这限制了需要为大量或不断演化的超图生成嵌入的应用场景。我们提出了HyperFuse，一个用于快速超图表示学习的无标签流水线。HyperFuse：(i) 利用Banerjee超图邻接矩阵和无矩阵算子，通过最大化超图模块度的谱松弛来计算结构化节点坐标，其计算成本与节点-超边关联数呈线性关系；(ii) 构建多尺度特征摘要，并基于特征掩码和成员掩码下的成员稳定性为超边分配有界效用权重；(iii) 使用不变性-去相关目标函数训练一个轻量级的效用加权超图编码器，仅需100个轮次。我们将HyperF...

    arXiv:2610.03211v1 Announce Type: new  Abstract: Self-supervised hypergraph representation learning can produce informative node embeddings, but existing methods often require deep encoders trained for hundreds of epochs, making embedding generation costly even for hypergraphs with a few thousand nodes. This limits applications requiring embeddings for many or evolving hypergraphs. We present HyperFuse, a label-free pipeline for fast hypergraph representation learning. HyperFuse (i) computes structural node coordinates by maximizing a spectral relaxation of hypergraph modularity using Banerjee's hypergraph adjacency and a matrix-free operator with cost linear in node-hyperedge incidences; (ii) constructs multi-scale feature summaries and assigns bounded utility weights to hyperedges based on member stability under feature and membership masking; and (iii) trains a lightweight utility-weighted hypergraph encoder for 100 epochs using an invariance-decorrelation objective. We compare Hype
    
[^81]: 哈密顿量局域性测试与认证无法达到海森堡极限

    Hamiltonian locality testing and certification do not achieve the Heisenberg limit

    [https://arxiv.org/abs/2610.03205](https://arxiv.org/abs/2610.03205)

    该论文证明了哈密顿量k-局域性测试和哈密顿量认证问题都需要Ω(1/ε²)的总演化时间，首次为哈密顿量学习与测试中的自然问题建立了排除1/ε海森堡极限标度的下界。

    

    我们在只能访问时间演化算符但无法访问其逆算符的条件下，为哈密顿量性质测试建立了下界。每个实验可以多次查询时间演化算符，哈密顿量之间的距离用归一化Frobenius范数来度量。在该模型中，我们证明了判断一个哈密顿量是k-局域的还是与所有k-局域哈密顿量都相距ε-远，需要Ω(1/ε²)的总演化时间，这与Kallaugher和Liang（TQC'25）的上界相匹配。我们还证明了判断一个未知哈密顿量是否等于某个目标哈密顿量或与其相距ε-远，需要Ω(1/ε²)的总演化时间，这与Sinha和Tong（2025）的上界相匹配。这些是哈密顿量学习与测试领域自然问题的首个下界，排除了1/ε的海森堡极限标度。作为第三个结果，我们证明了振幅估计（摘要在此处截断）

    arXiv:2610.03205v1 Announce Type: cross  Abstract: We establish lower bounds for Hamiltonian property testing with access to the time-evolution operator but not its inverse. Each experiment may query the time-evolution operator multiple times, and distances between Hamiltonians are measured in the normalized Frobenius norm.   In this model, we show that testing whether a Hamiltonian is $k$-local or $\varepsilon$-far from every $k$-local Hamiltonian requires $\Omega(1/\varepsilon^2)$ total evolution time, matching the upper bound of Kallaugher and Liang (TQC'25). We also prove that testing whether an unknown Hamiltonian equals a target Hamiltonian or is $\varepsilon$-far from it requires $\Omega(1/\varepsilon^2)$ total evolution time, matching the upper bound of Sinha and Tong (2025). These are the first lower bounds for natural problems in Hamiltonian learning and testing that rule out Heisenberg-limited scaling of $1/\varepsilon$.   As a third result, we show that amplitude estimation
    
[^82]: 预测导向的高斯过程后验

    Predictively Oriented Gaussian Process Posteriors

    [https://arxiv.org/abs/2610.03201](https://arxiv.org/abs/2610.03201)

    提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。

    

    高斯过程（GP）是建模和量化函数关系中不确定性的强大工具。然而，它们要求使用者做出许多设计决策，例如核函数和观测模型的选择。次优的选择可能产生错误设定的模型，无法捕捉潜在的数据生成过程。我们提出了预测导向高斯过程（PrO-GPs），它将预测不确定性作为主要的推断目标，为标准高斯过程提供了一种稳健的替代方案。尽管对非参数模型直接计算PrO后验是不可行的，我们推导了一种简化形式和实用的采样方案以实现高效计算。通过合成数据和真实数据的实验，我们表明与标准高斯过程方法相比，PrO-GPs在模型错误设定的情况下能够产生更好校准的预测分布。

    arXiv:2610.03201v1 Announce Type: cross  Abstract: Gaussian Processes (GPs) are a powerful tool for modelling and quantifying uncertainty in functional relationships. However, they require practitioners to make a number of design decisions, such as the choice of the kernel and the observation model. Suboptimal choices can produce misspecified models that do not capture the underlying data generating process. We introduce Predictively Oriented Gaussian Processes (PrO-GPs), which treat predictive uncertainty as the primary inferential target and provide a robust alternative to standard GPs. Although direct computation of a PrO posterior for nonparametric models is intractable, we derive a reduced formulation and practical sampling scheme for efficient computation. Through synthetic and real data experiments, we show that PrO-GPs produce better calibrated predictive distributions under model misspecification compared to standard GP approaches.
    
[^83]: 预测与修复大语言模型中的合并崩塌

    Predicting and Repairing Merge Collapse in Large Language Models

    [https://arxiv.org/abs/2610.03199](https://arxiv.org/abs/2610.03199)

    该论文提出用基于专家模型任务向量方差的“干扰度”评分，在大语言模型合并前预测是否会崩塌并指导修复，实验表明只有破坏性合并会超过该评分阈值，而现有合并算子常用的符号冲突统计量反而具有反向预测作用。

    

    从同一共享基础模型微调得到的大语言模型可以通过对其任务向量取平均进行合并，但某些合并的结果会远低于基础模型本身，而常见的合并算子在评估之前不会给出任何警告。我们证明了专家模型任务向量的一个统计量既能预测这种崩塌，又能校准其修复方法。取平均所去除的功率等于各专家模型任务向量之间的方差，这也是我们对模型间干扰程度的度量。在一个有效的噪声模型下，合并所注入的扰动随合并系数和干扰程度而增长，从而产生一个合并前的评分。在我们对来自四个模型家族的二十二种合并配置的实验中，只有破坏性合并会超过该评分的阈值。我们还发现专家模型之间符号冲突的统计量——现有合并算子的常见优化目标——反而是反向预测的。随后，我们在评估之前预测了十四次合并的结果，其中十二次……（摘要在此处截断）

    arXiv:2610.03199v1 Announce Type: cross  Abstract: Large language models fine-tuned from a shared base can be merged by averaging their task vectors, but some merges collapse far below the base model, and common merge operators give no warning before evaluation. We show that one statistic of the specialists' task vectors both predicts this collapse and calibrates its repair. The power that averaging removes equals the variance of the task vectors across specialists, our measure of interference. Under a working noise model, the disturbance that a merge injects grows with the merge coefficient and with interference, yielding a pre-merge score. In our experiments on twenty-two merge configurations from four model families, only destructive merges exceed a threshold on this score. We find that statistics of sign conflict between specialists, a common target of existing merge operators, are anti-predictive. We then predicted the outcomes of fourteen merges before evaluating them, and twelve
    
[^84]: 直到证据说了算：教会LLM调查员何时结案

    Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case

    [https://arxiv.org/abs/2610.03190](https://arxiv.org/abs/2610.03190)

    该论文研究了LLM调查员“何时应结案”的证据充分性判断问题，发现未经训练的小模型和前沿模型都普遍夸大证据充分性而过早结案，并提出需对照来源捷径规则来评估结案能力的方法。

    

    事故、缺陷和故障调查以一个普通问答从不面对的决定告终：目前收集到的证据是否足以结案。我们研究LLM调查员如何做出这一决定：它们从案卷中请求证据、修正假设，要么以基于所读内容的结论结案，要么保持案件开放并指出尚缺什么。这种判断能力并非与生俱来：一个未经训练的9B模型在97%的答案中夸大了其证据，而一个能在84%的案例中识别出正确原因的前沿模型仍在91%的情况下夸大证据，并在41个官方结论为“原因未定”的案例中结案了17个。衡量这种判断也并非易事：案件的来源在很大程度上预测了其标签，一个仅读取来源的规则在我们的测试用例上即可达到83.0的平衡准确率。因此，我们通过三项测试来评估结案：结案准确率（对照该规则报告）……

    arXiv:2610.03190v1 Announce Type: cross  Abstract: Accident, defect and outage investigations end with a decision that ordinary question answering never faces: whether the evidence gathered so far is enough to close the case. We study this decision for LLM investigators, which request evidence from a case file, revise their hypotheses, and either close the case with a conclusion grounded in what they read or leave it open and name what is missing. This judgment does not come with capability: an untrained 9B model overstates its evidence in 97% of its answers, and a frontier model that identifies the right cause in 84% of cases still overstates in 91% and closes 17 of the 41 cases whose official finding is "cause undetermined". Measuring it is also non-trivial: the source of a case largely predicts its label, and a rule that reads only the source reaches 83.0 balanced accuracy on our test cases. We therefore evaluate closure with three tests: closure accuracy, reported against this rule
    
[^85]: 方差缩减策略梯度的样本复杂度：更弱的假设与下界

    Sample complexity of variance-reduced policy gradient: weaker assumptions and lower bounds

    [https://arxiv.org/abs/2610.03165](https://arxiv.org/abs/2610.03165)

    本文提出基于防御性重要性采样的防御性策略梯度算法，无需对重要性权重方差做任何假设即可实现 $O(\epsilon^{-3})$ 样本复杂度，并在广义黑盒策略优化模型中建立了匹配的最优速率下界。

    

    多种基于重要性采样的方差缩减版 REINFORCE 算法，在一个关于重要性权重方差的非现实假设下，实现了改进的 $O(\epsilon^{-3})$ 样本复杂度以找到 $\epsilon$-稳定点。本文提出了基于防御性重要性采样的防御性策略梯度（Defensive Policy Gradient）算法，该算法在不对普通重要性权重方差做任何假设的情况下达到了同样的速率。我们还在一个隐藏状态与动作、并允许奖励依赖于参数的广义黑盒策略优化模型中建立了下界。在该模型中，有界方差的单策略反馈的最优速率为 $\Theta(\epsilon^{-4})$，均方光滑的耦合双策略反馈的最优速率为 $\Theta(\epsilon^{-3})$。在标准的策略正则性条件下，REINFORCE 和防御性策略梯度算法满足相应的预言机条件，并分别达到 $O(\epsilon^{-4})$ 和 $O(\epsilon^{-3})$ 的复杂度。

    arXiv:2610.03165v1 Announce Type: new  Abstract: Several variance-reduced versions of REINFORCE based on importance sampling achieve an improved $O(\epsilon^{-3})$ sample complexity to find an $\epsilon$-stationary point, under an unrealistic assumption on the variance of the importance weights. In this paper, we propose the \algo (Defensive Policy Gradient) algorithm, based on defensive importance sampling, which achieves the same rate without any assumption on the variance of ordinary importance weights. We also establish lower bounds in a generalized black-box policy-optimization model that hides states and actions and permits parameter-dependent rewards. In this model, the optimal rates are $\Theta(\epsilon^{-4})$ with bounded-variance one-policy feedback and $\Theta(\epsilon^{-3})$ with mean-square-smooth coupled two-policy feedback. Under standard policy-regularity conditions, REINFORCE and \algo realize the corresponding oracle conditions and attain the $O(\epsilon^{-4})$ and $O
    
[^86]: 光子量子求解器在金融风险检测QUBO特征选择中依赖问题地形的性能表现

    Landscape-Dependent Performance of Photonic Quantum Solvers in QUBO Feature Selection for Financial Risk Detection

    [https://arxiv.org/abs/2610.03161](https://arxiv.org/abs/2610.03161)

    该论文在信用卡欺诈与消费者违约两个金融数据集上，对经典分支定界（Gurobi）、光子熵计算（QCI Dirac-3）与模拟光子玻色采样（Piquasso）三种计算范式的十三种QUBO特征选择方法进行了系统基准测试，揭示了求解器性能高度依赖于数据集地形：Dirac-3在ULB欺诈数据集上仅用13/30个特征即可匹敌全特征模型（平均F1约0.873），而在159特征的AmEx违约数据集上所有范式均需接近全部特征才能达到F1≈0.80。

    

    针对信用卡欺诈和消费者违约检测等不平衡分类任务的特征选择，需要在预测相关性、特征间冗余与计算可行性之间取得平衡。我们在两个数据集上——ULB信用卡欺诈数据集（30个特征）和美国运通（AmEx）消费者违约数据集（159个特征）——对三种计算范式进行了基准测试：经典分支定界优化（Gurobi）、光子熵计算（QCI Dirac-3）以及模拟光子玻色采样，涵盖十三种特征选择方法，每种方法均被路由到与其数学结构相匹配的求解器。在ULB数据集上，Dirac-3的MI-Spearman方法仅使用30个特征中的13个即可达到与全特征模型相当的性能（五次运行的平均F1为0.873 ± 0.023，最佳运行为0.896），而Piquasso在k=5时是表现最佳的方法。在AmEx数据集上，性能随特征预算的增加而稳步提升，且所有范式只有在接近全部特征集时才能达到约F1 = 0.80的水平。Gurobi与其他求解器之间的大多数差异……（原文摘要在此处截断）

    arXiv:2610.03161v1 Announce Type: cross  Abstract: Feature selection for imbalanced classification tasks such as credit card fraud and consumer default detection requires balancing predictive relevance, inter-feature redundancy, and computational feasibility. We benchmark three computing paradigms, classical branch-and-bound optimization (Gurobi), photonic entropy computing (QCI Dirac-3), and simulated photonic boson sampling (Piquasso), across thirteen feature-selection methods on two datasets: ULB Credit Card Fraud (30 features) and AmEx consumer default (159 features). Each method is routed to the solver matched to its mathematical structure. On ULB, Dirac-3 MI-Spearman matches the all-features model using 13 of 30 features (mean F1 0.873 +/- 0.023 over five runs, best run 0.896), and Piquasso is the best method at k=5. On AmEx, performance rises steadily with the feature budget and every paradigm approaches F1 = 0.80 only near the full feature set. Most differences between Gurobi a
    
[^87]: 物理存在于激活之中吗？在视频扩散模型中定位物理量

    Does Physics Live in the Activations? Localizing Physical Quantities in Video Diffusion Models

    [https://arxiv.org/abs/2610.03154](https://arxiv.org/abs/2610.03154)

    该论文发现视频扩散Transformer在去噪早期阶段就能在其内部激活（尤其是物体对应token处）中以线性方式编码运动学与刚体动力学等物理量，表明物理信息是模型在去噪过程中主动构建的，而非输入中自带的。

    

    视频生成模型能够生成极其逼真的序列，并日益被提议作为世界模型，然而近期的基准测试显示它们在物理推理方面存在明显缺陷。这引发了一个问题：这些模型究竟是内化了物理原理，还是仅仅在复现熟悉的运动模式。为此，我们通过探测视频扩散Transformer（DiT）的内部表示来研究这一问题，探测对象涵盖来自模拟器的真值物理量，包括运动学运动以及重力和接触作用下的刚体动力学。我们发现，这些物理量在去噪过程的早期阶段即可被线性解码并达到很高的精度，其表现显著优于直接从模型自身带噪潜变量解码的基线，这表明相关物理信息是在去噪过程中被主动构建的，而非已存在于输入之中。此外，我们还发现，位于物体上的token的激活中承载着相关的物理……

    arXiv:2610.03154v1 Announce Type: cross  Abstract: Video generation models produce strikingly realistic sequences and are increasingly proposed as world models, yet recent benchmarks reveal pronounced deficits in their physical reasoning. This raises the question of whether these models internalize physical principles or merely reproduce familiar motion patterns. We address this by probing internal representations of video Diffusion Transformers (DiTs) for simulator-derived ground-truth physical quantities spanning kinematic motion and rigid-body dynamics under gravity and contact. We find that these quantities are linearly decodable with high accuracy early in the denoising process, substantially outperforming a baseline decoded directly from the model's own noised latents, indicating that the relevant physical information is actively constructed during denoising rather than already present in the input. Additionally, we show that activations at on-object tokens carry the relevant phy
    
[^88]: TSGuard：一个用于流式时间序列中缺失数据检测与填补的实时框架

    TSGuard: A Real-Time Framework for Detecting and Imputing Missing Data in Streaming Time Series

    [https://arxiv.org/abs/2610.03147](https://arxiv.org/abs/2610.03147)

    TSGuard是一个实时框架，将轻量级图感知时间序列填补模型与约束感知验证、回退估计相结合，实现流式时间序列中缺失数据的检测、填补与验证，并将其融入完整的数据质量闭环。

    

    流式传感器应用经常因故障、通信丢失或环境干扰而遭受观测数据延迟或缺失的困扰。尽管近期的填补方法能够有效利用时间和空间依赖性，但大多数方法要么假设可以离线访问未来的观测数据，要么只优先考虑吞吐量而不保证领域合理性。我们提出了TSGuard，这是一个用于监控、验证和填补流式时间序列中缺失值的实时演示系统。TSGuard将轻量级的图感知时间序列填补模型与约束感知验证、回退估计以及面向操作员的解释相结合。TSGuard并不将填补视为孤立的预测任务，而是将其集成到更广泛的数据质量闭环中：检测有问题的观测值、填补缺失值、根据物理和空间约束验证估计值，并将原始值保留为合理的……

    arXiv:2610.03147v1 Announce Type: cross  Abstract: Streaming sensor applications routinely suffer from delayed or missing observations caused by faults, communication losses, or environmental interference. Although recent imputation methods exploit temporal and spatial dependencies effectively, most either assume offline access to future observations or prioritize throughput without enforcing domain plausibility. We present TSGuard, a real-time demonstration system for monitoring, validating, and imputing missing values in streaming time series. TSGuard combines a lightweight graph-aware temporal imputation model with constraint-aware validation, fallback estimation, and operator-facing explanations.   Rather than treating imputation as an isolated prediction task, TSGuard integrates it into a broader data-quality loop: detect problematic observations, impute missing values, validate estimated against physical and spatial constraints, and either retain the original value as a plausible
    
[^89]: Page-EntroKV：分组查询注意力下硬件对齐的、熵加权的KV缓存驱逐方法

    Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under Grouped-Query Attention

    [https://arxiv.org/abs/2610.03135](https://arxiv.org/abs/2610.03135)

    提出Page-EntroKV框架，在GQA实际分配的页粒度上，利用sink隔离的Rényi-2碰撞熵加权池化各查询头的分数进行KV缓存驱逐，避免因各头独立选择导致的缓存膨胀，同时保护专用检索头。

    

    长上下文自回归语言模型的推理服务受限于键值（KV）缓存。大多数动态驱逐方法针对每个查询头单独对token重要性打分并独立进行选择。这种做法与分组查询注意力（GQA）适配不佳：在GQA中，多个查询头共享同一个物理KV缓冲区，各头分散的选择会迫使服务引擎保留它们选择的并集——最多使缓存膨胀至分组比例r倍——而算术平均池化则会稀释承担事实回忆功能的专用检索头。我们提出Page-EntroKV，一个在GQA服务实际分配的粒度上运行的KV缓存驱逐形式化框架。每个物理组内的各头通过由“sink隔离的碰撞（Rényi-2）熵”导出的权重进行池化——每个头只需一次内积，在预填充阶段计算一次且无需校准——从而使sink头无法伪装成检索头。池化后的分数被投影到页（Page）粒度上……

    arXiv:2610.03135v1 Announce Type: new  Abstract: Serving long-context autoregressive language models is constrained by the key-value (KV) cache. Most dynamic eviction methods score token importance per query head and choose tokens independently. This fits poorly with grouped-query attention (GQA), where several query heads share one physical KV buffer: divergent per-head selections force the serving engine to retain the union of their choices - inflating the cache by up to the group ratio r - while arithmetic mean pooling dilutes the specialized retrieval heads that carry factual recall. We introduce Page-EntroKV, a formal framework for KV-cache eviction operating at the granularity GQA serving actually allocates. Heads within each physical group are pooled by weights derived from sink-isolated collision (Renyi-2) entropy - one inner product per head, computed once at prefill with no calibration - so sink heads cannot masquerade as retrieval heads. Pooled scores are projected onto Page
    
[^90]: 通过对齐采样动态与执行动态实现安全流式流规划

    Safe Streaming Flow Planning by Aligning Sampling Dynamics with Execution Dynamics

    [https://arxiv.org/abs/2610.03132](https://arxiv.org/abs/2610.03132)

    SafeStreamingFlow通过让流采样动态与系统执行动态对齐、并仅对当前执行步用高阶控制障碍函数强制安全约束，实现了计算高效且安全的流式在线规划。

    

    基于扩散/流匹配的生成式规划器能够从演示数据中学习合成长时程轨迹。然而，实际部署需要 在执行过程中强制满足安全约束， 以快速的执行速率进行紧密的在线重规划。以往的安全扩散/流规划器一次性生成智能体的完整轨迹，并通过反复扰动中间状态来满足安全约束。这种方法不仅计算量大，而且由于学习到的采样动态与系统的执行动态不同，还会引入分布偏移。我们提出SafeStreamingFlow，这是一种目标条件规划器，它通过将学习到的状态向量场与分层状态预测按顺序积分，使流采样动态与执行动态保持一致。重要的是，我们只需通过高阶控制障碍函数对当前执行步强制满足安全约束。

    arXiv:2610.03132v1 Announce Type: cross  Abstract: Generative planners based on diffusion/flow matching can learn to synthesize long-horizon trajectories from demonstrations. However, real-world deployment requires (i) enforcing safety constraints during execution and (ii) tight online replanning at fast execution rates. Prior safe diffusion/flow planners generate the agent's full trajectory at once, while repeatedly perturbing intermediate states to satisfy safety constraints. This approach is not only computationally intensive, but also introduces distribution shift since the learned sampling dynamics is distinct from the system's execution dynamics. We propose SafeStreamingFlow, a goal-conditioned planner that aligns flow sampling dynamics with execution dynamics by sequentially integrating a learned state vector field with hierarchical state prediction. Importantly, we need to enforce safety constraints only for the executed step via high order control barrier functions. Across nav
    
[^91]: 可调控的覆盖保证：面向强化学习驱动的硬件感知神经架构搜索的在线保形校准

    Coverage You Can Steer: Online Conformal Calibration for RL-Driven Hardware-Aware NAS

    [https://arxiv.org/abs/2610.03127](https://arxiv.org/abs/2610.03127)

    针对强化学习搜索中候选分布漂移破坏保形预测覆盖保证的问题，本文提出用自适应保形推断的在线反馈控制替代一次性分位数估计，使剪枝过滤器的覆盖率可被单调、可复现地调控，从而在硬件感知NAS中恢复了对任意序列的分布无关保证。

    

    硬件感知神经架构搜索（NAS）的主要瓶颈在于评估成本：每个候选架构都必须先经过训练才能获知其奖励。保形预测过滤器通过剪除预测奖励上界未达阈值的候选来削减这一成本，并提供无分布保证——被错误丢弃的候选比例至多为 δ。然而该保证假设校准候选与测试候选之间满足可交换性，而外围的强化学习（RL）循环恰恰违反了这一假设：策略生成的候选随搜索推进不断改进，且在逐层构建过程中每个回合内部也会发生漂移。我们用在线反馈控制（自适应保形推断 Adaptive Conformal Inference，包含免调参、局部自适应及分组条件化等变体）取代一次性分位数估计，恢复了可调控的覆盖保证：只需设定目标覆盖率，即可对任意序列单调且可复现地实现该覆盖率。在三种神经网络架构族上……

    arXiv:2610.03127v1 Announce Type: new  Abstract: Hardware-aware neural architecture search (NAS) is dominated by evaluation cost: every architecture must be trained before its reward is known. Conformal-prediction filters cut this cost by pruning candidates whose predicted-reward upper bound misses a threshold, with a distribution-free guarantee that at most a fraction $\delta$ are wrongly discarded. That guarantee assumes exchangeability between calibration and test candidates, which the surrounding reinforcement-learning (RL) loop violates: the policy's proposals improve as search proceeds and, in layer-by-layer construction, shift within every episode. We replace one-shot quantile estimation with online feedback control (Adaptive Conformal Inference, with tuning-free, locally-adaptive, and group-conditional variants), restoring steerable coverage: dialing the target delivers it, monotonically and reproducibly, for arbitrary sequences. Across three neural-network architecture familie
    
[^92]: ParaGeo：将副语言变化分解为共享的潜在几何结构

    ParaGeo: Decomposing Paralinguistic Variation into a Shared Latent Geometry

    [https://arxiv.org/abs/2610.03125](https://arxiv.org/abs/2610.03125)

    ParaGeo通过内容匹配分解方法证明，冻结语音语言模型内部存在跨语言内容稳定可复现的副语言属性共享低维潜在几何结构。

    

    语音的表达方式会同时随着所请求的副语言属性和语言内容而变化。我们提出了ParaGeo，一种在冻结语音语言模型中对副语言变化进行内容匹配分解的方法。合成音频token以固定的监听提示进行重放；汇集的键/值（K/V）表示经中心化处理后投影到一个共享的低维空间中。我们的GLM-4-Voice探测涵盖八个句子中来自12个基准家族的80个请求控制属性。使用全局拟合的校准基，基于该基的内容留出质心准确率达到9.49%，而置换基线仅为1.25%；同标签跨内容的余弦相似度为0.285，对照组仅为0.017，且两个条件置换检验均得到p = 0.001。另一个独立的十场景、六风格探测揭示了跨场景可复现的对比方向。静态、加性和时间干预产生了依赖于属性、层和调度方式的……

    arXiv:2610.03125v1 Announce Type: cross  Abstract: Speech delivery varies with both the requested paralinguistic attribute and the linguistic content. We introduce ParaGeo, a matched-content decomposition of paralinguistic variation in a frozen speech language model. Synthesized audio tokens are replayed with a fixed listening prompt; pooled key/value (K/V) representations are centered and projected into a shared low-dimensional space. Our GLM-4-Voice probe spans 80 requested controls from 12 benchmark families across eight sentences. With a globally fitted calibration basis, content-held-out centroid accuracy using this basis is 9.49% versus a 1.25% permutation baseline; same-label cross-content cosine similarity is 0.285 versus 0.017, and both conditional permutation tests yield p = 0.001. A separate ten-scenario, six-style probe reveals reproducible contrast directions across scenarios. Static, additive, and temporal interventions produce attribute-, layer-, and schedule-dependent r
    
[^93]: 开放权重大语言模型中用于滥用检测的触发-标记机制的脆弱性

    The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs

    [https://arxiv.org/abs/2610.03124](https://arxiv.org/abs/2610.03124)

    该论文首次形式化了开放权重大语言模型中的触发-标记滥用检测机制，将其分为令牌级和权重级两类，并系统研究揭示了此类机制在对抗性攻击下的脆弱性。

    

    开放权重大语言模型可以被下载、修改和部署，超出了开发者的控制范围，这限制了集中式安全保障措施的有效性。因此，近期的研究提出了“触发-标记”机制，当模型在目标条件下被使用时（例如生成钓鱼内容），该机制会产生可检测的信号。尽管这些机制借鉴了已有的技术，但它们在开放权重大语言模型的条件性滥用检测方面的应用相对较新。因此，现有研究工作尚未系统性地研究触发-标记机制在对抗性攻击下的鲁棒性。为了填补这一空白，（i）我们对触发-标记进行了形式化定义，区分了在解码过程中引入水印式信号的“令牌级触发-标记”与学习目标条件与可检测模型行为之间后门式关联的“权重级触发-标记”。此外，（ii）我们……

    arXiv:2610.03124v1 Announce Type: cross  Abstract: Open-weight language models can be downloaded, modified, and deployed beyond their developers' control, limiting the effectiveness of centrally enforced safeguards. Recent work has therefore proposed \emph{trigger-tag} mechanisms that produce a detectable signal when a model is used under a target condition, such as generating phishing contents. Although these mechanisms borrow from established techniques, their use for conditional misuse detection in open-weight LLMs is relatively new. Therefore, existing research works have not systematically studied the robustness of trigger-tag mechanisms under adversarial attacks. To close this gap, (i)~we formalize trigger-tags and distinguish \emph{token-level trigger-tags}, which introduce watermark-inspired signals during decoding, from \emph{weight-level trigger-tags}, which learn backdoor-inspired associations between target conditions and detectable model behavior. Furthermore, (ii)~we intr
    
[^94]: 如何在终身强化学习中发现并重用策略以实现持续适应

    How to Find and Reuse Policies for Continuous Adaptation in Lifelong Reinforcement Learning

    [https://arxiv.org/abs/2610.03119](https://arxiv.org/abs/2610.03119)

    提出AMSC方法，利用基于Wasserstein任务嵌入的在线相似性估计，自适应地选择和组合多个先前策略作为先验，从而在终身强化学习中获得更高的平均性能、前向迁移能力且不遗忘旧知识。

    

    在终身强化学习中，仅仅保留先前学到的策略并不足以实现对新任务的有效迁移。有用的知识可能分布在多个先前策略之中，并且其相关性会随着学习者积累经验而发生变化。一种假设是，在持续学习环境中可以有效地利用任务相似性来发现并组合先前学到的策略。为了验证这一假设，研究提出了自适应掩码选择与组合方法（AMSC），该方法通过从状态-动作-奖励样本中构建非参数化的Wasserstein任务嵌入，从在线经验中估计任务相似性。利用经z-score标准化的sparsemax相似性分数，推导出可变大小的支持集，从而在学习新任务时周期性地选择并加权策略以形成先验。在CT-graph和MiniGrid基准测试中，AMSC相比所评估的模块化组合基线方法取得了更高的平均性能和前向迁移能力，同时表现出没有遗忘的特性。

    arXiv:2610.03119v1 Announce Type: cross  Abstract: In lifelong reinforcement learning, retaining previously learned policies is not sufficient for effective transfer to a new task. Useful knowledge may be distributed across several prior policies, and its relevance may change as the learner acquires experience. One hypothesis is that task similarity can be effectively used in a continual learning setting to find and combine previously learned policies. To test it, Adaptive Mask Selection and Composition (AMSC) is designed to estimate similarity from online experience via non-parametric Wasserstein task embeddings from state-action-reward samples. The z-score-normalized sparsemax of the similarity scores are used to derive a variable-size support to periodically choose and weight policies to form a prior when learning a new task. On CT-graph and MiniGrid, AMSC achieves higher mean performance and forward transfer than the evaluated modular composition baselines while exhibiting no forge
    
[^95]: 探索面向空间应用的深度神经网络中结构化剪枝与容错性之间的权衡

    Exploring the Trade-Off Between Structured Pruning and Fault Tolerance in Deep Neural Networks for Space Applications

    [https://arxiv.org/abs/2610.03117](https://arxiv.org/abs/2610.03117)

    本研究通过故障注入实验发现，深度神经网络的结构化剪枝虽然因减少冗余而提高了单次推理对单粒子翻转故障的敏感性，但更短的执行时间降低了遭遇故障的概率，从而有效平衡了这一负面影响。

    

    深度神经网络（DNN）由于其信息的分布式表示，本质上对比特级故障具有一定程度的鲁棒性。随着模型宽度的增加，信息变得更加分散，理论上降低了任何单个比特故障的影响。在本文中，我们实证研究了模型宽度与单粒子翻转（SEU）鲁棒性之间的关系。我们进行了一项全面的实验：基线模型经过迭代结构化剪枝以减小宽度，同时尽可能保持任务性能。在每个剪枝阶段，我们开展针对性的故障注入实验，以评估模型在模拟比特翻转场景下的性能。我们的结果表明，尽管结构化剪枝通过减少冗余增加了每次推理对故障的敏感性，但这一效应被更短的执行时间有效抵消，因为执行时间越短，遇到故障的概率就越低。

    arXiv:2610.03117v1 Announce Type: new  Abstract: Deep Neural Networks (DNNs) inherently exhibit a degree of robustness to bit-level faults due to their distributed representation of information. As a model increases in width, this information becomes more dispersed, theoretically reducing the impact of any single bit fault. In this paper, we empirically investigate the relationship between model width and robustness to Single Event Upsets (SEUs). We conduct a comprehensive experiment in which baseline models undergo iterative structured pruning to reduce their width while preserving task performance as much as possible. At each pruning stage, we run a targeted fault-injection campaign to evaluate the model's performance under simulated bit-flip scenarios. Our results show that, although structured pruning increases per-inference sensitivity to faults by reducing redundancy, this effect is effectively counterbalanced by shorter execution time, which lowers the probability of encounterin
    
[^96]: S2S-JEPA：在次季节到季节时间尺度上预测可预测的部分

    S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales

    [https://arxiv.org/abs/2610.03106](https://arxiv.org/abs/2610.03106)

    该论文提出S2S-JEPA，首次将计算机视觉中的联合嵌入预测架构（JEPA）范式引入次季节到季节（S2S）预报，通过在潜在空间中只预测缓慢变化且可预测的分量、舍弃不可预测的细尺度细节，来突破AI天气模型在两周以上“可预测性荒漠”中的性能瓶颈。

    

    次季节到季节（S2S）时间尺度，大约指未来两周到两个月，是农业、能源和水资源管理等行业的关键预报窗口。然而，这一时间尺度被广泛称为“可预测性荒漠”。近期的AI天气模型在两周以内的预报表现出色，但超过两周后性能显著下降，这主要是因为它们被训练去预测在S2S时间尺度上既不可预测也不重要的细尺度细节。我们认为，一个更符合物理规律的目标是只预报那些仍然可预测的缓慢变化分量。计算机视觉领域通过联合嵌入预测架构得出了相同的结论，该架构在潜在空间中进行预测，从而丢弃不可预测的细节。在这项工作中，我们提出了S2S-JEPA，将JEPA范式引入S2S预报任务。它借鉴了最先进AI天气模型的设计元素，针对这一任务进行了专门定制。S2S-JEPA达到了相当的预报技巧水平。

    arXiv:2610.03106v1 Announce Type: cross  Abstract: The subseasonal-to-seasonal (S2S) timescale, roughly from two weeks to two months ahead, is a critical forecast window for sectors such as agriculture, energy, and water management. Yet, it is widely known as the `predictability desert'. Recent AI weather models excel up to two weeks ahead but deteriorate beyond, largely because they are trained to predict fine-scale details that are neither predictable nor essential at S2S timescales. We argue that a more physically grounded objective is to forecast only the slowly varying components that remain predictable. Computer vision reached the same conclusion with the Joint-Embedding Predictive Architecture (JEPA), which predicts in latent space, discarding unpredictable details. In this work, we introduce S2S-JEPA, which brings the JEPA paradigm to S2S forecasting. It is tailored to this task through design elements from state-of-the-art AI weather models. S2S-JEPA achieves comparable skill 
    
[^97]: 因果效应识别中聚类操作的不变性

    Invariance of Clustering Operations in Causal Effect Identification

    [https://arxiv.org/abs/2610.03101](https://arxiv.org/abs/2610.03101)

    本文提出了一类基于原图c-分量条件的“识别不变”聚类操作，可同时保持因果效应的可识别性与不可识别性，从而安全地简化因果图并加速因果效应识别。

    

    在因果图中对变量进行聚类可以减小图的规模并简化因果推断。然而，任意的聚类可能会改变变量之间关键的因果关系，从而导致错误的结论。虽然在温和条件下，聚类图中因果效应的可识别性意味着原图中的可识别性，但在缺乏进一步假设的情况下，聚类图中的不可识别性并不能推出原图中的不可识别性。当可识别性与不可识别性均被保持时，该聚类操作被称为“识别不变的”。我们基于与原图c-分量相关的条件，提出了一大类识别不变的聚类操作。最后，我们展示了这些结果在实际场景中的应用。

    arXiv:2610.03101v1 Announce Type: cross  Abstract: Clustering variables in causal graphs reduces the size of the graph and simplifies causal inference. However, arbitrary clustering can alter crucial causal relations among variables and lead to erroneous conclusions. While the identifiability of a causal effect in the clustered graph implies the identifiability in the original graph under mild conditions, nonidentifiability in clustered graph does not imply nonidentifiability in the original graph without further assumptions. When both identifiability and nonidentifiability are preserved, the clustering operation is called identification invariant. We present a broad class of clustering operations that are identification invariant based on conditions related to the c-components of the original graph. Finally, we demonstrate use of the results in practical settings.
    
[^98]: 预测器引导的潜空间密码子优化以最大化蛋白质表达

    Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression

    [https://arxiv.org/abs/2610.03098](https://arxiv.org/abs/2610.03098)

    提出潜空间密码子优化方法LSCO，通过将序列映射到预训练mRNA语言模型的潜空间，将离散的密码子优化问题转化为可梯度搜索的连续问题，并结合不确定性感知的表达预测器、最小自由能正则化、自然性先验和约束解码，以最大化蛋白质表达。

    

    密码子优化是通过选择同义密码子来提高mRNA翻译效率和蛋白质表达水平的过程，是治疗性蛋白质生产和mRNA疫苗研发的核心环节，但它仍然是一个难题。其设计空间是离散的且组合规模巨大，这使得基于梯度的方法无法适用，而现有工具依赖启发式代理指标（如密码子适应指数或GC含量），这些指标难以真实反映实际表达水平。我们提出了潜空间密码子优化（LSCO），通过将序列映射到预训练mRNA语言模型的潜空间中，将这一离散问题转化为连续问题，从而实现高效的基于梯度的搜索。LSCO结合了四个组件：来自不确定性感知预测器的数据驱动表达目标函数、用于结构稳定性的最小自由能（MFE）正则化项、来自蛋白质到密码子反向翻译模型的自然性先验，以及确保蛋白质保真度的约束解码。

    arXiv:2610.03098v1 Announce Type: new  Abstract: Codon optimization, the process of selecting synonymous codons to improve mRNA translation efficiency and protein expression, is central to therapeutic protein production and mRNA vaccines, yet it remains a hard problem. The design space is discrete and combinatorially large, precluding gradient-based methods, and existing tools rely on heuristic proxies (e.g., Codon Adaptation Index or GC-content) that poorly capture true expression. We introduce Latent-Space Codon Optimization (LSCO), which recasts this discrete problem as a continuous one by mapping sequences into the latent space of a pretrained mRNA language model, enabling efficient gradient-based search. LSCO combines four components: a data-driven expression objective from an uncertainty-aware predictor, a Minimum-Free-Energy regularizer for structural stability, a naturalness prior from a protein-to-codon back-translation model, and constrained decoding for protein fidelity. On 
    
[^99]: LS-AR：自回归大语言模型中的未来预测式潜在引导

    LS-AR: Future-Predictive Latent Steering in Autoregressive LLMs

    [https://arxiv.org/abs/2610.03093](https://arxiv.org/abs/2610.03093)

    提出双通道架构LS-AR，通过FiLM条件化将连续目标引导与离散token解码解耦，在超出上下文限制的长时程检索中实现100%召回率，同时吞吐量提升约35%、峰值显存降低52.8%。

    

    标准的自回归（AR）模型在单一共享的token序列中处理高层任务指令、状态历史和瞬态token。因此，它们缺乏将宏观目标与上下文噪声隔离开来的架构机制。为了克服这种单通道限制，我们提出了潜在引导自回归模型，这是一种双通道架构，通过FiLM条件化将连续的目标引导与离散的token解码解耦。我们评估了用于在长回合 rollout 中保持持久宏观目标的静态目标编码器（P_0），以及用于在生成过程中进行循环潜在更新的动态状态跟踪器（P_t）。在超出上下文限制的长时程检索任务上（H=1024, W=500），LS-AR（静态）实现了100%的目标召回率，而参数量相同的基线模型则完全失效（0%），同时吞吐量提升约35%，峰值显存（VRAM）占用降低52.8%。在强制扰动下的Blocksworld规划任务中，LS-AR（动态

    arXiv:2610.03093v1 Announce Type: new  Abstract: Standard autoregressive (AR) models process high-level task instructions, state history, and transient tokens within a single shared sequence of tokens. Consequently, they lack the architectural mechanisms needed to isolate macro-objectives from context noise. To overcome this single-channel limitation, we introduce Latent-Steered Autoregressive (LS-AR), a dual-channel architecture that decouples continuous goal steering from discrete token decoding via FiLM conditioning. We evaluate a Static Goal Encoder (P_0) for persistent macro-objective retention across long rollouts and a Dynamic State Tracker (P_t) for recurrent latent updates during generation. On long-horizon retrieval past context limits (H=1024, W=500), LS-AR (Static) achieves 100% target recall where parameter-matched baselines collapse (0%), while increasing throughput by ~35% and cutting peak VRAM by 52.8%. In Blocksworld planning under forced perturbations (k=1), LS-AR (Dy
    
[^100]: ULTRADISCOVERY：在一个互联的、认识论开放的宇宙中进行溯因探索

    ULTRADISCOVERY: Abductive Exploration in an Interconnected, Epistemically Open Universe

    [https://arxiv.org/abs/2610.03092](https://arxiv.org/abs/2610.03092)

    该论文提出ULTRADISCOVERY交互式基准，通过2×2设计独立控制表征开放性与证据分布性，评估智能体在认识论开放且结构互联的世界中进行溯因科学探索的能力，发现现有十一个模型均无法通过引入新实体或重写变量来完成理论替换。

    

    科学发现往往始于零散的线索呼唤一种描述世界的新方式。当世界在认识论上是开放的，这种溯因探索可能需要构建用以陈述解释的表征；当世界在结构上是互联的，则需要综合散布于不同情境中的证据。现有基准很少将这两项需求区分开来或对它们进行独立控制。我们提出ULTRADISCOVERY，一个包含五个领域的交互式世界，智能体在其中需要修正一个最初成功的理论，并预测一次未见过的跨领域干预的结果。该基准采用2×2设计：表征保持开放或予以揭示，证据保持分散或予以对齐，而潜在动力学保持不变。在表征开放的条件下，横跨十一个模型的智能体常常收回他们被教授的公理，但没有任何一个模型能够引入替换所需的未观测实体或重写变量……

    arXiv:2610.03092v1 Announce Type: cross  Abstract: Scientific discovery often begins when scattered clues call for a new way of describing the world. Such abductive exploration can require constructing the representation in which an explanation is stated, when the world is epistemically open, and composing evidence scattered across contexts, when it is structurally interconnected. Existing benchmarks rarely separate these two demands or control them independently. We introduce ULTRADISCOVERY, an interactive world of five domains in which an agent revises an initially successful theory and predicts the outcome of an unseen cross-domain intervention. A $2 \times 2$ design leaves the representation open or discloses it, and leaves the evidence distributed or aligns it, with the latent dynamics fixed. With the representation open, agents across eleven models often retract the axiom they were taught, and none introduces the unobserved entity or rewrites the variables that a replacement requ
    
[^101]: Zephon：面向在线、有状态基础模型数据加载流水线的弹性确定性

    Zephon: Elastic Determinism for Online, Stateful Foundation Model Data Loading Pipelines

    [https://arxiv.org/abs/2610.03087](https://arxiv.org/abs/2610.03087)

    Zephon提出了一种面向基础模型训练的数据加载器，能够在GPU拓扑变化、频繁检查点恢复及不同执行后端的情况下，为包含在线分词、打包、混合等有状态n对m转换的数据流水线提供确定性的全局训练数据批次序列（即弹性确定性）。

    

    确定性数据加载对于基础模型开发至关重要：模型研究人员需要确信，他们在昂贵的消融实验中观察到的差异是由所更改的参数引起的，而非训练数据序列中的非确定性所致。数据加载器必须提供弹性确定性，即即使在多次运行之间GPU拓扑发生变化（例如由于GPU资源稀缺）、频繁的检查点恢复周期以及不同的数据处理执行后端的情况下，仍能保证确定的全局训练数据批次序列。实现这一目标非常困难，因为现代基础模型数据流水线会在线对样本进行分词、打包和混合，引入了破坏样本索引的有状态n对m转换。现有的数据加载器大多假设可索引的1对1流水线，而常见的替代方案——离线物化——成本高昂，且对于视频等某些模态来说不可行。我们提出了Zephon，一个面向基础模型的数据加载器……

    arXiv:2610.03087v1 Announce Type: cross  Abstract: Deterministic data loading is important for foundation model development: model researchers need confidence that differences they observe across costly ablations are caused by the parameter they changed rather than non-determinism in the training data sequence. The data loader must provide elastic determinism, i.e., a deterministic sequence of global training data batches despite changes to the GPU topology across runs (e.g., due to GPU scarcity), frequent checkpoint-resume cycles, and different data processing execution backends. Achieving this is difficult because modern foundation model data pipelines tokenize, pack, and mix samples online, introducing stateful n-to-m transformations that break sample indexing. Existing data loaders largely assume indexable 1-to-1 pipelines, and the common workaround of offline materialization is expensive and, for some modalities such as video, infeasible.   We present Zephon, a data loader for fou
    
[^102]: 黎曼流形上的轻量级熵最优传输

    Light Entropic Optimal Transport on Riemannian Manifolds

    [https://arxiv.org/abs/2610.03085](https://arxiv.org/abs/2610.03085)

    该论文提出ManifoldLightOT，一种通过构造几何特定的Gibbs核与相容势参数化，直接在球面、环面、SO(3)、SE(3)等黎曼流形上学习核诱导熵最优传输耦合的轻量级方法，实现了闭式归一化、条件分布的直接采样，并可自然扩展至流形乘积。

    

    熵最优传输（EOT）已成为学习复杂分布之间随机耦合的实用框架，广泛应用于生成建模和领域自适应等领域。然而，大多数EOT求解器是为欧几里得空间设计的，流形上的扩展仍然有限，且通常依赖代价高昂的迭代方法、模拟动力学或未能充分利用底层几何结构的通用神经模型。我们提出了ManifoldLightOT，一种直接在常见流形上学习核诱导EOT耦合的轻量级方法。利用EOT解的核形式，我们为球面、环面、SO(3)和SE(3)构造了几何特定的Gibbs核以及与之相容的势函数参数化。这些选择带来了闭式归一化和可直接采样的条件分布。我们的公式可自然扩展到流形的乘积，使其适用于更复杂的几何场景。

    arXiv:2610.03085v1 Announce Type: new  Abstract: Entropic Optimal Transport (EOT) has become a practical framework for learning stochastic couplings between complex distributions, with applications in generative modeling and domain adaptation. However, most EOT solvers are designed for Euclidean spaces, while manifold extensions remain limited and often rely on costly iterative methods, simulated dynamics, or generic neural models that do not fully exploit the underlying geometry. We introduce ManifoldLightOT, a light approach for learning kernel-induced EOT couplings directly on common manifolds. Using the kernel form of the EOT solution, we construct geometry-specific Gibbs kernels together with compatible potential parameterizations for spheres, tori, $\mathrm{SO}(3)$, and $\mathrm{SE}(3)$. These choices yield closed-form normalization and directly sampleable conditional distributions. Our formulation naturally extends to products of manifolds, making it applicable to more complex g
    
[^103]: NegT2IBench：当否定改变图像时——面向文本到图像模型的极性基准

    NegT2IBench: When Negation Changes the Picture. A Polarity Benchmark for Text-to-Image Models

    [https://arxiv.org/abs/2610.03084](https://arxiv.org/abs/2610.03084)

    提出了NegT2IBench基准，通过4,800条按极性组织的提示词系统评估文本到图像模型满足否定约束的能力，其基于检测器的评分以更小的规模达到了与大型视觉语言评判器相当的人类一致性水平。

    

    文本到图像（T2I）模型通常由测量请求内容是否出现的基准来评判，但这些基准在很大程度上忽略了模型满足否定约束的补充能力，例如生成“一个非红色的杯子”。衡量否定带来了基于肯定式基准所不曾面对的挑战，需要精心的提示词与评估设计。我们提出NegT2IBench，一个包含4,800条提示词的基准，涵盖两种属性类型和四种关系类别。提示词按极性组织：必须成立的肯定语句数量和必须不成立的否定语句数量，各自取值范围为0到2。通过独立变化这两个因素，可以将否定效应与提示词复杂度效应分离开来。我们基于检测器的评分具有可复现、可审计的特点，并能精确定位哪个需求失败了。在600张带有三位标注者标签的图像上，该评分与人类判断的一致性程度与体积大30倍的视觉语言评判器相当……

    arXiv:2610.03084v1 Announce Type: cross  Abstract: Text-to-image (T2I) models are judged by benchmarks that measure whether requested content appears, but these benchmarks largely overlook the complementary ability to satisfy negated constraints, for example, generating "a non-red cup." Measuring negation raises challenges not faced by affirmation-based benchmarks and requires careful prompt and evaluation design. We introduce NegT2IBench, a benchmark of 4,800 prompts covering two attribute types and four relation categories. Prompts are organized by polarity: the number of positive statements that must hold and negated statements that must not, each ranging from 0 to 2. Varying the two independently separates the effect of negation from the effect of prompt complexity. Our detector-based scoring is reproducible, auditable, and pinpoints which requirement failed. On 600 images with three-annotator labels, it agrees with humans as closely as vision-language judges up to 30x larger, whil
    
[^104]: 智能感知助力更安全的桥梁：从传感器信号到人工智能驱动的异常检测

    Smart Sensing for Safer Bridges: From Sensor Signals to AI-Driven Anomaly Detection

    [https://arxiv.org/abs/2610.03082](https://arxiv.org/abs/2610.03082)

    本文提出将信号处理与孤立森林两种互补方法应用于挪威桥梁实时传感器数据的异常检测，并通过多种评估指标和受控异常注入分析系统比较了两种方法的检测特性与敏感性。

    

    桥梁对交通连通性和城市发展做出了重大贡献。因此，可靠的桥梁监测对于保护公共安全以及检测桥梁传感器数据中的异常行为至关重要，这些异常行为可能为结构异常状况提供早期预警信号。本文采用两种不同的互补方法研究真实世界桥梁传感器数据中的异常检测，即信号处理和数据驱动的机器学习模型孤立森林。实时桥梁传感器数据来自安装在挪威一座桥梁上的iBridge传感器设备。这些方法通过异常计数、异常检测时间、处理速率、异常率、可视化以及时间一致性进行评估。此外，还进行了受控异常注入分析，以评估每种方法的敏感性。数值结果展示了两种方法不同的检测特性和计算需求。

    arXiv:2610.03082v1 Announce Type: new  Abstract: Bridges contribute significantly to transportation connectivity and urban development. Therefore, reliable bridge monitoring is crucial for protecting public safety and detecting anomalous behavior in bridge sensor data that may provide early indications of abnormal structural conditions. This paper investigates anomaly detection in real-world bridge sensor data using two different complementary approaches, namely signal processing and the data-driven machine learning model Isolation Forest. The real-time bridge sensor data is collected from an iBridge sensor device installed on a bridge in Norway. The methods are evaluated using anomaly counts, anomaly detection time, processing rate, anomaly rates, visualization, and temporal agreement. Moreover, a controlled anomaly-injection analysis is performed to evaluate the sensitivity of each method. Numerical results demonstrate distinct detection characteristics and computational requirements
    
[^105]: MintEval：大语言模型是否实现了你所要求的交易策略？一个面向自然语言转策略代码的行为等价性基准

    MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code

    [https://arxiv.org/abs/2610.03080](https://arxiv.org/abs/2610.03080)

    该论文提出MintEval基准，通过程序化生成参考交易策略并回译为自然语言指令让大语言模型重新实现，再在相同市场数据上逐K线比较生成策略与参考策略的实际交易行为（而非代码相似度或利润），以检验大语言模型编写的策略代码是否真正做到了行为等价于交易者的原始意图。

    

    大语言模型正在从生成交易信号转向编写执行这些信号的代码。第二种角色的失败模式是静默的：生成的代码可以运行，回测可以画出图表，但交易者所描述的风险逻辑却并非实际执行的逻辑。现有代码基准通过单元测试检验功能正确性，金融基准则检验预测能力，二者都无法衡量一个实现的行为是否与所要求的策略一致。我们提出MintEval，该基准从可组合的构建模块库中以程序化方式生成参考策略，将其回译为口语化的交易者指令，再由被测模型重新实现。生成的程序与参考程序在完全相同的市场数据和摩擦成本上逐根K线执行，并基于其交易行为而非代码相似度或利润进行比较：超额收益（alpha）被差分消除。MintEval v0包含800个基于BTCUSDT 15分钟的……（摘要不完整）

    arXiv:2610.03080v1 Announce Type: cross  Abstract: Large language models are moving from producing trading signals to writing the code that executes them. The failure mode of the second role is silent: generated code runs, a backtest plots, yet the risk logic that the trader described is not the logic being executed. Existing code benchmarks test functional correctness on unit tests and finance benchmarks test forecasting; neither measures whether an implementation behaves like the strategy that was asked for. We introduce MintEval, a benchmark in which reference strategies are generated programmatically from a library of composable building blocks, back-translated into colloquial trader instructions, and re-implemented by the model under test. Generated and reference programs are executed bar by bar on identical market data and frictions, and compared on their actions rather than on code similarity or profit: alpha is differenced away. MintEval v0 contains 800 tasks on BTCUSDT 15-minu
    
[^106]: 学习一次可行性，优化所有目标：用于机会约束规划的无导数扩散模型

    Learn Feasibility Once, Optimize All Objectives: Derivative-Free Diffusion Models for Chance-Constrained Programming

    [https://arxiv.org/abs/2610.03071](https://arxiv.org/abs/2610.03071)

    该论文提出 D³Opt 框架，通过训练一个与目标无关的风险条件扩散模型一次性学习机会可行结构作为可重用先验，并在推理时利用退火粒子 Feynman–Kac 校正，仅凭函数评估即可优化任意后续指定的目标，实现约束建模与目标优化的解耦。

    

    机会约束规划（CCP）通过限制违反约束的概率来在不确定性条件下优化决策。尽管传统方法和基于学习的方法已取得进展，但优化非凸或非光滑目标，以及在固定机会约束下适应不同目标，仍然具有挑战性。在本文中，我们提出了一个无导数的扩散框架，将约束建模与目标优化解耦，称为 D³Opt。我们通过仅在经过约束筛选的决策上训练一个风险条件化的扩散模型，一次性学习机会可行结构，该学习过程独立于任何特定目标，并将该模型冻结为可重用的先验，用于后续指定的目标。在推理阶段，我们提出了一种退火的、基于粒子的 Feynman–Kac 校正方法，沿冻结的反向扩散过程进行，仅需函数评估即可优化后续指定的目标。

    arXiv:2610.03071v1 Announce Type: new  Abstract: Chance-constrained programs (CCPs) optimize decisions under uncertainty by limiting the probability of constraint violation. Despite advances in traditional and learning-based approaches, optimizing non-convex or non-smooth objectives and adapting to different objectives under fixed chance constraints remain challenging. In this paper, we propose a \textbf{D}erivative-free \textbf{D}iffusion-based framework that \textbf{D}isentangles constraint modeling from objective optimization, termed \textbf{D$^3$Opt}. We learn the chance-feasible structure once, independently of any particular objective, by training a risk-conditioned diffusion model solely on constraint-filtered decisions and freezing it as a reusable prior for post-specified objectives. At inference time, we propose an annealed, particle-based Feynman--Kac correction along the frozen reverse diffusion process to optimize post-specified objectives using only function evaluations. 
    
[^107]: 基于扩散模型与大语言模型重排序的气相色谱-质谱可解释分子结构推断

    Explainable Molecular Structure Inference from GC--MS with Diffusion Models and LLM Reranking

    [https://arxiv.org/abs/2610.03066](https://arxiv.org/abs/2610.03066)

    本文提出DiffGCMS框架，先用谱条件离散图扩散模型从GC-EI-MS谱图从头生成候选分子结构，再由大语言模型进行验证、修复、重排序并提供可解释的碎片离子分析，从而实现对谱库中缺失化合物的可解释分子结构推断。

    

    GC-EI-MS（气相色谱-电子电离质谱）是分析复杂样品中挥发性和半挥发性化合物的重要技术。然而，传统方法严重依赖参考谱库匹配，这限制了它们识别谱库中不存在的化合物以及从碎片信息直接推断完整分子结构的能力。在此，我们提出了DiffGCMS，一种用于从GC-EI-MS进行从头结构解析的谱条件离散图扩散模型，并进一步开发了一个将DiffGCMS与大语言模型（LLM）二阶段推理相结合的框架。在第一阶段，DiffGCMS根据输入谱图生成候选分子结构；在第二阶段，LLM利用质谱信息对候选结构进行验证、修复和重排序，并对碎片离子峰提供可解释的分析。该框架能够为参考谱库中缺失的化合物生成合理的分子结构……

    arXiv:2610.03066v1 Announce Type: cross  Abstract: GC--EI--MS is an important technique for analyzing volatile and semivolatile compounds in complex samples. However, conventional methods rely heavily on reference spectral library matching, limiting their ability to identify compounds absent from these libraries and to infer complete molecular structures directly from fragmentation information. Here, we present DiffGCMS, a spectrum-conditioned discrete graph diffusion model for de novo structure elucidation from GC--EI--MS, and further develop a framework that integrates DiffGCMS with second-stage reasoning by a large language model (LLM). In the first stage, DiffGCMS generates candidate molecular structures from input spectra; in the second stage, the LLM uses mass spectral information to validate, repair, and rerank the candidates and provides interpretable analysis of fragment-ion peaks. This framework can generate plausible molecular structures for compounds absent from reference s
    
[^108]: 通过动力学嵌入从无动作时间序列中学习可迁移策略

    Learning Transferable Policies from Action-free Time Series Through Dynamical Embeddings

    [https://arxiv.org/abs/2610.03065](https://arxiv.org/abs/2610.03065)

    本文提出一个基于模型的层次化强化学习框架，通过低维动力学嵌入捕捉相关系统间的共享动力学结构，实现从无动作记录中学习可迁移的特定系统控制策略。

    

    从无动作记录中学习控制策略极具挑战性，因为干预效果无法被观测到，且策略可能利用重构动力学中的误差。我们提出了一个基于模型的层次化强化学习框架，该框架利用相关系统间的共享结构，从无动作记录中学习特定系统的控制策略。一个层次化动力系统重构模型通过低维嵌入捕捉共享动力学和个体差异。这些嵌入随后被复用，用于参数化共享的策略网络和价值网络，从而将重构动力学中的差异与控制行为中的差异联系起来。策略完全通过仿真进行训练，并采用带有加性潜在扰动的显式干预模型。分段线性循环神经网络能够对受控动力学进行机理分析，而基于解码器的约束使干预的即时效果具有可解释性。

    arXiv:2610.03065v1 Announce Type: new  Abstract: Learning control from action-free recordings is challenging because intervention effects are unobserved and policies may exploit errors in reconstructed dynamics. We present a hierarchical model-based reinforcement learning framework that uses shared structure across related systems to learn system-specific control policies from action-free recordings. A hierarchical dynamical system reconstruction model captures shared dynamics and individual variation through low-dimensional embeddings. These embeddings are then reused to parameterize shared policy and value networks, linking differences in reconstructed dynamics to differences in control. Policies are trained entirely via simulation under an explicit intervention model with additive latent perturbations. Piecewise-linear recurrent neural networks enable mechanistic analyses of the controlled dynamics, while decoder-based constraints make the immediate effects of interventions interpre
    
[^109]: 合成关系数据何时才能教会模型使用关系？从预训练数据到模型行为的预测结构追溯

    When Does Synthetic Relational Data Teach Models to Use Relations? Tracing Predictive Structure from Pretraining Data to Model Behavior

    [https://arxiv.org/abs/2610.03057](https://arxiv.org/abs/2610.03057)

    该研究通过对比四种数据生成器训练出的Relational Transformer模型，发现只有当外键关联的跨表信息对掩码单元格预训练目标具有预测必要性时，模型才会真正学会利用关系结构。

    

    关系型基础模型越来越多地在合成数据库上进行预训练，然而下游基准测试却很少能揭示为什么某个合成语料库能产生更好的模型。尤其是，模型可能仅凭借真实的行级统计数据就获得强劲表现，而实际上并未学会利用关系结构。我们将这一问题作为数据归因问题来研究：合成预训练数据的哪种属性会诱导模型进行关系计算？我们使用四个以相同架构、初始化、训练目标和计算预算、基于四种关系数据生成器所产生的语料库训练而成的Relational Transformer检查点，将数据的可测量属性与模型学到的计算及下游行为进行追溯关联。我们提出假设：只有当跨表信息对掩码单元格预训练目标在预测上是必要的，关系机制才会涌现。实验表明，RelDiff在通过外键关联的数据上展现出迄今为止最大的预测增益。

    arXiv:2610.03057v1 Announce Type: new  Abstract: Relational foundation models are increasingly pretrained on synthetic databases, yet downstream benchmarks reveal little about why one synthetic corpus produces a better model than another. In particular, strong performance may arise from realistic row-level statistics without the model ever learning to use relational structure. We study this as a data-attribution problem: which property of synthetic pretraining data induces relational computation? Using four Relational Transformer checkpoints trained with the same architecture, initialization, objective, and compute budget on corpora produced by four relational data generators, we trace a measurable property of the data to learned computation and downstream behavior. We hypothesize that relational mechanisms emerge when cross-table information is predictively necessary for the masked-cell pretraining objective. RelDiff exhibits by far the largest predictive gain from foreign-key-linked 
    
[^110]: 静水中的涟漪：基于小波散射变换的联邦学习零样本聚类

    RIPPLE in Still Water: Zero-Shot Clustering in Federated Learning with Wavelet Scattering Transform

    [https://arxiv.org/abs/2610.03054](https://arxiv.org/abs/2610.03054)

    RIPPLE提出了一种零样本聚类联邦学习框架，通过小波散射变换嵌入的方差加权主成分原型与服务器端预训练的高斯混合VAE，完全离线地完成客户端聚类分配，实现与FedAvg相同的通信成本并保护梯度隐私。

    

    聚类联邦学习（FL）将客户端群体划分为具有相似本地分布的组，并为每个聚类训练一个专门模型，从而缓解客户端漂移问题——该问题会在非独立同分布（non-IID）数据下降低单模型方法的性能。先前的方法在训练循环内部通过梯度相似性、损失评估或EM风格的更新来发现聚类结构，因此增加了通信开销，使梯度暴露于反演攻击，并且没有为未参与训练的客户端提供分配机制。我们提出RIPPLE，这是一种聚类联邦学习框架，其中聚类分配完全通过每个客户端本地数据的谱特征离线计算得出：即通过小波散射变换嵌入的方差加权主成分原型，并由一个在联邦开始前于服务器端使用合成客户端群体训练的高斯混合变分自编码器（VAE）进行解码。每轮通信成本与FedAvg完全相同，并且一个客（摘要在此处被截断）

    arXiv:2610.03054v1 Announce Type: new  Abstract: Clustered Federated Learning (FL) partitions a client population into groups of similar local distributions and trains one specialized model per cluster, mitigating client drift that degrades single-model methods under non-IID data. Prior methods discover cluster structure inside the training loop through gradient similarity, loss evaluation, or EM-style updates, thus increasing communication overhead, exposing gradients to inversion attacks, and providing no mechanism to assign clients absent from training. We propose RIPPLE, a clustered FL framework in which cluster assignment is computed entirely offline from a spectral characterization of each client's local data: a variance-weighted principal-component prototype embedded via the Wavelet Scattering Transform and decoded by a Gaussian Mixture VAE trained server-side on synthetic client populations before federation begins. Per-round communication cost matches FedAvg exactly, and a cli
    
[^111]: HyperThink：面向高效推理的文本到参数超网络

    HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning

    [https://arxiv.org/abs/2610.03039](https://arxiv.org/abs/2610.03039)

    HyperThink通过轻量级超网络将长思维链推理计算摊销为一次查询条件下的参数更新，使模型无需生成冗长思考轨迹即可直接输出简洁解答，在大幅降低推理延迟和token消耗的同时保持强推理性能。

    

    长篇思考轨迹能够显著提升大语言模型（LLM）的多步推理性能，但会引入高昂的推理时开销，其中延迟主要由顺序解码主导。我们提出HyperThink，这是一种文本到参数的方法，将推理计算摊销为一次基于查询条件的参数更新：一个轻量级超网络读取问题并预测基础LLM中一小部分参数的更新，同时向量量化解码器将这些更新约束在有限的、可复用的模式集合内，以提升鲁棒性和迁移能力。HyperThink在基础模型自身的输出上进行端到端训练，在测试时消除了冗长的思考轨迹：仅需一次超网络前向传播，适配后的模型即可生成简洁的分步解答和最终答案，无需中间思考过程，使用的token数量大幅减少，同时保持强大的推理性能。实验表明，HyperThink……

    arXiv:2610.03039v1 Announce Type: new  Abstract: Long-form thinking traces can substantially improve the multi-step reasoning performance of large language models (LLMs), but they introduce high inference-time overhead, with latency dominated by sequential decoding. We propose HyperThink, a text-to-parameter approach that amortizes this reasoning computation into a single query-conditioned parameter update: a lightweight hypernetwork reads the question and predicts updates to a small subset of the base LLM's parameters, while a vector-quantized decoder constrains them to a finite set of reusable patterns to improve robustness and transfer. Trained end-to-end on outputs from the base model itself, HyperThink eliminates long thinking traces at test time: after one hypernetwork forward pass, the adapted model generates a concise step-by-step solution and final answer without an intermediate trace, using far fewer tokens while retaining strong reasoning performance. Empirically, HyperThink
    
[^112]: WebFovea：当模型正确但点击出错时——基于视觉的网页智能体在真实网站上的可靠往返执行

    WebFovea: When the Model Is Right but the Click Is Wrong -- Reliable Round Trips for Vision-Based Web Agents on Live Websites

    [https://arxiv.org/abs/2610.03036](https://arxiv.org/abs/2610.03036)

    本文提出在WebRetriever Challenge 2026中获得亚军的视觉网页智能体WebFovea，并指出真实网站上的许多失败并非源于模型推理，而是源于模型与浏览器之间中间执行层在动作解析、页面生效、结果反馈和信息展示这四个环节上的问题。

    

    我们提出了WebFovea，一个基于视觉的网页智能体，它在WebRetriever Challenge 2026中获得第二名，最终得分为100分中的57.0分。该挑战赛在WebRetriever基准（arXiv:2607.06118）的协议III上对智能体进行端到端评估：从真实网站上的入口URL出发，智能体必须操作网站自身的界面并返回可验证的答案。一个强大的多模态大语言模型（LLM）对完成此任务是必要的，但并不充分。模型的决策需要通过中间执行层（harness）——即模型与页面之间的代码——传递到浏览器。在每一步中，有四件事必须正确完成：模型的回复必须被解析为预期的动作，动作必须在页面上生效，结果必须被准确地反馈回来，并且模型必须被展示它所需的信息。在真实网站上，我们观察到的许多失败发生在这四个阶段之一，而不是出现在模型的推理中。一个坐标空间（摘要原文在此处截断）

    arXiv:2610.03036v1 Announce Type: cross  Abstract: We present WebFovea, a vision-based web agent that placed 2nd in the WebRetriever Challenge 2026 with a final score of 57.0 out of 100. The challenge evaluates agents end to end on Protocol III of the WebRetriever benchmark (arXiv:2607.06118): starting from an entry URL on a live website, the agent must operate the site's own interface and return a verifiable answer. A capable multimodal large language model (LLM) is necessary for this, but not sufficient. The model's decisions reach the browser through the harness, the code between the model and the page. At every step, four things must go right: the model's reply must be parsed into the intended action, the action must take effect on the page, the result must be reported back accurately, and the model must be shown the information it needs. On real websites, many of the failures we observed occurred at one of these four stages rather than in the model's reasoning. A coordinate-space 
    
[^113]: 通过函数空间进展平衡多模态学习

    Balancing Multimodal Learning via Functional Progress

    [https://arxiv.org/abs/2610.03035](https://arxiv.org/abs/2610.03035)

    提出FGMO方法，通过函数空间的进展信号评估各模态的优化进展并协调跨模态优化，克服了传统基于分数差异的估计方法将模态内在差异误判为进展差距的问题，有效缓解多模态学习中的模态不平衡。

    

    多模态学习常常受到模态不平衡问题的困扰，即联合优化过程被单一模态所主导。现有方法通常基于预测不确定性或优化统计量得出的分数差异来估计模态不平衡。然而，由于不同模态具有不同的预测不确定性和学习动态，直接比较这些分数可能将模态的内在差异误判为进展差距，从而导致有偏差的不平衡估计。本文提出函数空间引导的多模态优化方法，利用函数空间的进展信号来评估各模态的优化进展，并协调跨模态的优化过程，以缓解模态不平衡。具体而言，我们引入函数进展估计来度量每个模态更新所引起的函数空间响应，并通过与损失对齐的单模态参考进行校准，从……

    arXiv:2610.03035v1 Announce Type: new  Abstract: Multimodal learning often suffers from modality imbalance, where the joint optimization process is dominated by a single modality. Existing methods typically estimate modality imbalance from score disparities derived from prediction uncertainty or optimization statistics. However, due to distinct prediction uncertainty and learning dynamics across modalities, direct comparison of such scores may misinterpret intrinsic modality differences as progress gaps, leading to biased imbalance estimation. In this paper, we propose Function-Space Guided Multimodal Optimization (FGMO), which leverages a function-space progress signal to assess modality-wise optimization progress and coordinate optimization across modalities to alleviate modality imbalance. Specifically, we introduce Functional Progress Estimation (FPE) to measure each modality's update-induced function-space response and calibrate it against a loss-aligned unimodal reference, produc
    
[^114]: 用于快速随机扩散采样的自适应二阶求解器

    Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling

    [https://arxiv.org/abs/2610.03034](https://arxiv.org/abs/2610.03034)

    该论文将PI步长控制与扩散噪声归一化误差估计器结合，提出了扩散模型的自适应二阶求解器，实现更平滑的步长调整，并可将逐样本的自适应轨迹聚合为固定调度，在大幅降低采样成本的同时保留自适应采样的质量收益。

    

    扩散模型依赖于需要时间离散化的数值求解器，这对采样成本与质量之间的权衡有很大影响。然而，反向过程的计算难度沿采样轨迹以及在不同数据分布之间会发生变化，因此离散化方式的选择十分重要。我们将比例-积分（PI）步长控制适配到扩散模型中，并使用我们提出的扩散噪声归一化误差估计器。与扩散领域中现有的仅响应当前误差的自适应方法不同，PI求解器还会结合先前的误差，从而实现更平滑的步长调整。我们进一步证明，这些逐样本的轨迹表现出共享结构，可以将其聚合为固定调度，从而保留自适应采样的大部分收益。我们在自然图像和语言数据集上，以在相同次数神经网络评估下的FID作为质量衡量标准，对这两种方法进行了评估。

    arXiv:2610.03034v1 Announce Type: cross  Abstract: Diffusion models rely on numerical solvers requiring time-discretization, which has a large influence on the tradeoff between sampling cost and quality. However, the computational difficulty of the reverse process varies along the sampling trajectory and across data distributions, making the choice of discretization important. We adapt proportional-integral (PI) step-size control to diffusion, using our diffusion noise-normalised error estimator. Unlike existing adaptive methods in diffusion that respond only to the current error, the PI solver also incorporates the previous error, yielding smoother step adaptation. We further show that these per-sample trajectories exhibit shared structure and can be aggregated into a fixed schedule that retains much of the benefit of adaptive sampling. We evaluate both approaches on natural-image and language datasets, in terms of quality, measured by FID at a matched number of neural network evaluat
    
[^115]: 为1比特KV缓存压缩量身定制量化空间

    Tailoring the Quantization Space for 1-Bit KV Cache Compression

    [https://arxiv.org/abs/2610.03027](https://arxiv.org/abs/2610.03027)

    提出TaSQ方法，通过查询引导的通道加权、跨头归一化和协方差感知的通道分组来量身定制向量量化目标空间，从而在1比特极端压缩下实现有效的KV缓存压缩。

    

    键值缓存已成为长上下文大语言模型推理中的主要内存瓶颈，给内存容量和带宽带来了巨大压力。为缓解这一瓶颈，向量量化（VQ）已成为一种有前景的激进KV缓存压缩方法。然而，现有的VQ方法在1比特压缩区间下性能大幅下降。在如此极端的压缩下，每个码本必须用有限的质心集合来表示更大规模的通道组，使得有效利用码本容量变得愈发困难。为解决这一问题，我们提出了TaSQ，它通过结合查询引导的通道加权、跨头归一化以及协方差感知的通道分组来量身定制VQ目标空间，从而更好地反映缓存激活的误差敏感性和统计结构。由于这些变换与RoPE兼容，并且可以轻松地合并到投影权重和码本中，TaSQ保持了传统的（摘要在此处被截断）

    arXiv:2610.03027v1 Announce Type: cross  Abstract: The key-value (KV) cache becomes a major memory bottleneck in long-context LLM inference, placing substantial pressure on memory capacity and bandwidth. To mitigate this bottleneck, vector quantization (VQ) has emerged as a promising approach for aggressive KV cache compression. However, existing VQ methods degrade substantially in the 1-bit regime. At such extreme compression, each codebook must represent a larger group of channels with a limited set of centroids, making effective use of its capacity increasingly challenging. To address this, we introduce $\textbf{TaSQ}$, which tailors the VQ target space by combining query-guided channel weighting, cross-head normalization, and covariance-aware channel grouping to better reflect the error sensitivity and statistical structure of cached activations. Since these transforms are RoPE-compatible and can be easily merged into projection weights and codebooks, TaSQ preserves the conventiona
    
[^116]: 偏好的可验证、可表达与默会成分

    Verifiable, Articulable, and Tacit Components of Preference

    [https://arxiv.org/abs/2610.03025](https://arxiv.org/abs/2610.03025)

    该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。

    

    是什么让一篇短篇小说引人入胜、一篇新闻报道具有新闻价值、或一个数学证明优雅？这些构念难以言明或验证，其含义至少部分是默会的。然而，现代AI模型主要是通过明确的章程、评分标准和验证器（即RLAIF和RLVR）来改进的；偏好的默会成分通常研究不足。我们引入了一个大规模、带标注的偏好数据集CreativePreferences，其中包含280万个文本，由3.17亿个人类偏好判断在7个创意领域中进行标注，并配有42个基准任务。我们分别用可执行程序、评分标准库和密集训练的模型（V、A和VAT）对这些标签进行建模。我们观察到稳健的可表达性差距（VAT−VA）和可验证性差距（VAT−V）；我们采用一种新颖的测量方法来估计每个差距的上界和下界，该方法可发现可表达和可验证的指标、识别伪变量，并估计未被发现成分的价值。

    arXiv:2610.03025v1 Announce Type: new  Abstract: What makes a short story gripping; a news article newsworthy; or a math proof elegant? These constructs resist articulation or verification; their meaning is at least partially tacit. However, modern AI models are improved primarily via articulated constitutions, rubrics and verifiers (i.e. in RLAIF and RLVR); tacit components of preferences are typically understudied. We introduce a large, labeled preference dataset CreativePreferences, containing 2.8M texts labeled by 317M human preference judgments across 7 creative domains, with 42 benchmark tasks. We model these labels with executable programs, rubric banks and densely trained models (V, A and VAT, respectively). We observe robust articulability gaps, VAT-VA; and verifiability gaps, VAT-V; we estimate upper and lower bounds for each gap with a novel measurement approach that discovers articulable and verifiable metrics, identifies spurious variables and estimates the value of undisc
    
[^117]: 信号简化并非预测简化：短周期波动率中残差神经预测的诊断

    Signal Simplification Is Not Predictive Simplification: Diagnosing Residual Neural Forecasting in Short-Horizon Volatility

    [https://arxiv.org/abs/2610.03019](https://arxiv.org/abs/2610.03019)

    研究通过信号-预测-系统诊断框架证明，统计第一阶段对残差信号的简化（方差大幅降低、自相关微弱）并不等于残差更易被神经网络学习——残差LSTM增强反而全面恶化了短周期波动率预测，而纯LSTM表现最佳。

    

    混合统计-神经流水线通常假设，成功的统计第一阶段会留下更干净、更易学习的残差目标。我们通过一个“信号-预测-系统”诊断框架，在短周期波动率预测中检验这一假设。在五种流动性良好的美国资产上，波动率对齐的HAR风格模型优于AR、MA和ARIMA。在扩展训练窗口内，用于构建残差-LSTM序列的标准化前拟合残差过程的方差比相应目标低约82%，且滞后一阶自相关接近于零；独立地，滚动伪样本外HAR误差显示约74%的方差缩减以及同样微弱的滞后一阶依赖。然而，仅用残差的LSTM增强平均使均方误差从0.3049升至0.3594，且在每一项资产上均出现恶化。纯LSTM取得了最低的所选伪样本外MSE，为0.2649，而残差混合模型……

    arXiv:2610.03019v1 Announce Type: new  Abstract: Hybrid statistical-neural pipelines often assume that a successful statistical first stage leaves a cleaner and more learnable residual target. We examine that assumption in short-horizon volatility forecasting through a signal-forecast-system diagnostic framework. Across five liquid U.S. assets, a volatility-aligned HAR-style model outperforms AR, MA, and ARIMA. Within expanding training windows, the pre-standardization fitted residual process used to construct residual-LSTM sequences has about 82% lower variance than the corresponding target and near-zero lag-1 autocorrelation; independently, rolling pseudo-out-of-sample HAR errors show about 74% variance reduction and similarly weak lag-1 dependence. Residual-only LSTM augmentation nevertheless raises mean squared error from 0.3049 to 0.3594 on average, with deterioration on every asset. Pure LSTM records the lowest selected pseudo-out-of-sample MSE, 0.2649, while the residual hybrid 
    
[^118]: AvoKV-E：面向长推理任务的负载感知KV缓存淘汰策略

    AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning

    [https://arxiv.org/abs/2610.03007](https://arxiv.org/abs/2610.03007)

    AvoKV-E是一种无需训练的KV缓存淘汰策略，通过延迟近期状态的淘汰资格，并结合读取压力、键冗余度和值负载潜力对缓存条目排序，在长推理任务中以相同的活跃KV预算达到或超越现有基线方法。

    

    长输出推理将KV缓存的瓶颈从固定的提示转移到了生成的推理轨迹上。现有的推理缓存淘汰方法大多将缓存条目视为路由对象，估计某个旧的键是否仍会被读取、是否会再次出现、或是否可以被替换。这种仅关注路由的视角忽视了两个效应：低注意力的条目可能携带较大的值负载，移除它们会改变未来的预测；而新生成的状态可能在后续查询有机会读取它们之前就被误判为“陈旧”。我们提出了AvoKV-E，这是一种无需训练的淘汰策略，它首先延迟近期状态的可淘汰资格，然后使用候选归一化读取压力、键冗余度和值负载潜力对符合条件的条目进行排序。在不同模型和数据集上的实证评估表明，在相同的活跃KV预算下，AvoKV-E达到或超越了感知冗余、基于重现以及思维自适应的淘汰基线方法，其最大……

    arXiv:2610.03007v1 Announce Type: cross  Abstract: Long-output reasoning shifts the KV-cache bottleneck from the fixed prompt to the generated trace. Existing reasoning-cache eviction methods largely treat cached entries as routing objects, estimating whether an old key will still be read, will recur, or can be replaced. This routing-only view overlooks two effects: low-attention entries can carry large value payloads whose removal changes future predictions, and newly generated states can appear stale before later queries have had a chance to read them. We introduce AvoKV-E, a training-free eviction policy that first delays eligibility for recent states and then ranks eligible entries using candidate-normalized read pressure, key redundancy, and value-payload potential. According to empirical evaluation across different models and datasets, AvoKV-E matches or exceeds redundancy-aware, recurrence-based, and thought-adaptive eviction baselines at matched active-KV budgets, with its larg
    
[^119]: 神经数据需要语义标记化：行为事件作为可跨会话迁移标记的边界

    Neural Data Needs Semantic Tokenization: Behavioral Events as Boundaries of Session-Transferable Tokens

    [https://arxiv.org/abs/2610.03001](https://arxiv.org/abs/2610.03001)

    该论文提出基于状态的标记化方法，以行为事件为边界将每次试验中的群体活动状态转换为可跨会话迁移的群体几何标记，从而解决神经基础模型因神经元记录每次会话都不同而无法泛化到新会话的核心问题。

    

    细胞外电生理记录在每次会话中记录的是不同的一组神经元。神经基础模型将每个神经元和每个会话嵌入到其标记中，因此每个新会话都是模型从未见过的输入，导致模型无法泛化到新会话。针对新会话的标记器需要一个所有会话共享且携带行为信息的单元。群体活动提供了这样一个单元：一旦会话被对齐，群体活动便在一个低维流形上演化，该流形在神经元更替以及跨动物之间均保持不变。该流形会在任务事件（如刺激起始和运动起始）处发生状态切换。在每个状态区间内，群体占据一个“状态”，即它在两个事件之间所跨越的流形部分，而每个状态都带有自身的行为意义。我们提出了基于状态的标记化，它在这些事件处对每个试验（即任务的一次重复）进行分段，并将每个状态转换为群体几何的标记，无需依赖任何神经元或……（原文摘要在此处截断）

    arXiv:2610.03001v1 Announce Type: new  Abstract: Extracellular electrophysiology records a different set of neurons in every session. Neural foundation models embed each neuron and each session into their tokens, so every new session is an input they have never seen, and they fail to generalize to it. A tokenizer for new sessions needs a unit that every session shares and that carries behavior. Population activity offers such a unit. It evolves on a low-dimensional manifold that persists across neuronal turnover and across animals once sessions are aligned. This manifold changes regime at task events such as stimulus onset and movement onset. Within each regime the population occupies a state, the part of the manifold it spans between two events, and each state carries its own behavioral meaning. We propose Tokenization with States (TWS), which segments each trial, one repetition of the task, at these events and converts every state into tokens of population geometry, with no neuron or
    
[^120]: 深度网络的时间几何：用于内在可解释性的训练动态双曲表示

    Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability

    [https://arxiv.org/abs/2610.03000](https://arxiv.org/abs/2610.03000)

    该论文提出利用双曲几何的庞加莱模型构建多层感知机训练过程的时间参数图（即多个训练步骤的快照），以捕捉网络加权拓扑与自组织在训练轨迹中的几何演化，从而超越传统单检查点方法实现内在可解释性。

    

    内在可解释性仍然是一个具有挑战性的问题，尤其是在多层感知机需要在优化环境中进行动态再训练的场景下。本文研究了如何在非欧几里得空间中表示和研究多层感知机及其训练动态；我们的表示采用了双曲几何的庞加莱模型。我们的目标是捕捉其加权拓扑结构和自组织随时间的几何演化。与已有的基于度量的可解释性方法将分析限制在单一检查点不同，我们构建了时间“参数图”，即在 $T$ 步优化/训练过程中对多层感知机进行的时间快照。这反映了这样一种观点：神经网络不仅在其权重中编码信息，还在训练过程中所走过的轨迹中编码信息。借鉴许多复杂网络可以嵌入到隐藏度量空间的思想……

    arXiv:2610.03000v1 Announce Type: cross  Abstract: Intrinsic explainability remains a challenging problem, particularly in contexts where multilayer perceptrons (MLPs) require dynamic re-training within an optimization environment. This paper investigates how MLPs and their training dynamics can be represented and studied in non-Euclidean spaces; our representation features the Poincar\'e model of hyperbolic geometry. We aim to capture the geometric evolution of their weighted topology and self-organization over time. Instead of restricting the analysis to single checkpoints---as per established measure-based explainability methods---we construct temporal \textit{parameter graphs}, i.e., snapshots over time $T$ steps of the optimization/training process for MLPs. This reflects the view that neural networks encode information not only in their weights but also in the trajectory traced during training. Drawing on the idea that many complex networks admit embeddings in hidden metric space
    
[^121]: Sentry：学习在测试时从LLM智能体失败中恢复

    Sentry: Learning to Recover from LLM Agent Failures at Test Time

    [https://arxiv.org/abs/2610.02994](https://arxiv.org/abs/2610.02994)

    提出Sentry——一个与LLM智能体并行运行的失败管理层，将失败经验视为条件性知识，在检测到失败时按需从外部经验手册中检索指导恢复、无奖励验证恢复结果并仅在确认恢复后存储新经验，从而在测试时实现从失败中学习。

    

    LLM智能体经常在任务执行中途因无效的工具调用、重复操作或缺乏依据的推理而失败，从这些失败中学习是提升可靠性的途径。我们发现，失败知识如何传递给智能体与其内容本身同样重要。失败经验是条件性的：如果保留在智能体的上下文中，当对应的失败情形并不存在时它们会误触发，而将它们从不断演化的经验手册中移除反而能提升性能。相比之下，运行时干预只在失败发生时起作用，但不会从修复过程中学习。我们提出，失败知识是一种条件性知识，应当被有条件地暴露，并将这一原则实例化为Sentry——一个与智能体并行运行的失败管理层。当Sentry检测到失败时，它会从外部经验手册中检索匹配的经验来指导恢复，在无需访问任务奖励的情况下验证智能体是否已恢复，并且只有在其确实恢复时才存储新的经验。

    arXiv:2610.02994v1 Announce Type: cross  Abstract: LLM agents often fail mid-task due to invalid tool calls, repeated actions, or poorly grounded reasoning, and learning from these failures is a path to reliability. We find that how failure knowledge reaches the agent matters as much as what it contains. Failure lessons are conditional: kept in the agent's context, they misfire when their failure is absent, and removing them from an evolving playbook improves performance. Runtime interventions, in contrast, act only when a failure occurs but do not learn from their repairs. We argue that failure knowledge is conditional knowledge and should be conditionally exposed, and instantiate this principle in Sentry, a failure-management layer that runs alongside the agent. When Sentry detects a failure, it retrieves matching lessons from an external playbook to guide recovery, verifies without access to task rewards whether the agent recovered, and stores a new lesson only if it did; the full p
    
[^122]: 面向动态图对比学习的可微Koopman算子

    Differentiable Koopman Operator for Contrastive Learning on Dynamic Graphs

    [https://arxiv.org/abs/2610.02990](https://arxiv.org/abs/2610.02990)

    KAIROS框架通过在动态图对比学习中嵌入可微Koopman算子来线性化时间演化，利用多粒度对比目标学习稳健的节点表示，并结合Koopman预测残差实现异常检测。

    

    真实世界的交互网络本质上是动态的：随着节点行为随时间变化，边不断形成和消解。大多数基于快照的对比方法将时间依赖性隐式地编码在编码器权重中，缺乏对节点表示如何演化的显式模型，这使得它们在分布偏移下表现脆弱。我们提出了KAIROS（面向开放动态系统的Koopman对齐不变表示），这是一个自监督框架，它在动态图对比学习循环中嵌入了一个可微的Koopman算子，以在学习的嵌入空间中对时间演化进行线性化。双视图编码器将原始节点特征与图扩散的结构视图配对，并通过跨时间窗口的多粒度对比目标进行优化。在异常检测方面，KAIROS利用Koopman预测残差，并结合时间不一致性和局部邻域偏差，将不规则行为与可预测行为区分开来。

    arXiv:2610.02990v1 Announce Type: new  Abstract: Real-world interaction networks are inherently dynamic: edges form and dissolve as node behavior shifts over time. Most snapshot-based contrastive methods encode temporal dependencies implicitly in encoder weights, without an explicit model of how node representations evolve, making them brittle under distribution shifts. We propose KAIROS (Koopman-Aligned Invariant Representations for Open Dynamic Systems), a self-supervised framework that embeds a differentiable Koopman operator within a dynamic graph contrastive learning loop to linearize temporal evolution in the learned embedding space. A dual-view encoder pairs raw node features with a graph-diffused structural view and is optimized with multi-granularity contrastive objectives across temporal windows. For anomaly detection, KAIROS uses the Koopman prediction residual together with temporal inconsistency and local neighborhood deviation to separate irregular behavior from predictab
    
[^123]: 狄拉克互联神经元件：无需约化即可发现物理系统中的模块化

    Dirac-Interconnected Neural Elements: Discovering Modularity in Physical Systems Without Reduction

    [https://arxiv.org/abs/2610.02960](https://arxiv.org/abs/2610.02960)

    提出狄拉克互联神经元件（DINEs），将物理系统建模为由狄拉克结构给出代数约束的微分代数方程，无需预先已知互联结构或将系统约化为常微分方程，即可从数据中同时识别出物理系统的模块化结构。

    

    深度学习在动力系统的数据驱动建模方面取得了显著成功。其成功很大程度上并非归功于神经网络的灵活性，而是归功于基于物理先验知识（如能量守恒和辛结构）的归纳偏置。然而，现有方法并未充分利用现实世界物理系统是由组件相互连接而成的这一事实。一些方法需要预先已知互联结构，而另一些方法则假设系统可约化为常微分方程（ODE），仅学习约化后的ODE，从而丢弃了互联所施加的代数约束。在此，我们提出狄拉克互联神经元件（DINEs），这是一种将物理系统表示为微分代数方程（DAE）的神经网络模型，其代数约束由核表示形式的狄拉克结构给出。借助DINEs，我们可以同时从数据中识别出……

    arXiv:2610.02960v1 Announce Type: new  Abstract: Deep learning has shown remarkable success in the data-driven modeling of dynamical systems. Much of its success is attributed not to the flexibility of neural networks but to inductive biases based on physical prior knowledge, such as energy conservation and symplecticity. However, existing methods do not fully exploit the fact that real-world physical systems are interconnections of components. Some methods require the interconnection to be known a priori, while others assume the system to be reducible to an ordinary differential equation (ODE) and learn only the reduced ODE, discarding the algebraic constraints imposed by the interconnection. Here, we propose Dirac-interconnected neural elements (DINEs), a neural network model that represents a physical system as a differential-algebraic equation (DAE), whose algebraic constraints are given by a Dirac structure in kernel representation. With DINEs, we simultaneously identify from data
    
[^124]: 理解联邦世界模型学习中的轨迹异质性

    Understanding Trajectory Heterogeneity in Federated World Model Learning

    [https://arxiv.org/abs/2610.02957](https://arxiv.org/abs/2610.02957)

    该论文在临床时间序列数据上系统基准测试了联邦世界模型学习中的轨迹异质性问题，发现客户端数据所有权与参与率会共同限制长时程窗口的可用覆盖，揭示了跨时间联邦学习的核心训练瓶颈。

    

    世界模型从轨迹中学习状态演化，因此获取时间上下文是其训练的核心需求。联邦学习可以利用分布式记录，但轨迹内部的数据所有权边界限制了每个客户端能够构造的样本。我们的研究通过对八个MIMIC-IV疾病队列进行按小时动作条件化的临床预测，对这一跨时间设置进行了基准测试，共涵盖4087万个状态转移归属关系。我们明确了基于疾病严重程度的客户端数据所有权、按患者分离的数据构造方式、局部历史与未来窗口规则，以及从1小时到32小时的成对滚动评估协议。十种联邦算法构成的算法矩阵在五轮10%参与率的设置下覆盖了32种疾病—数据划分配置。从已有结果和训练日志中得出三项发现。第一，客户端所有权与参与率共同限制了长窗口的覆盖能力：在汇总可用的32步窗口中，仅有7.55%–21.36%的窗口……

    arXiv:2610.02957v1 Announce Type: cross  Abstract: World models learn state evolution from trajectories, making access to temporal context a central training requirement. Federated learning can use distributed records, while ownership boundaries within a trajectory restrict the examples each client can construct. Our study benchmarks this cross-time setting through hourly action-conditioned clinical prediction on eight MIMIC-IV disease cohorts, comprising 40.87 million transition memberships. We specify severity-based client ownership, patient-separated construction, local history and future-window rules, and paired rollout evaluation from one to 32 hours. A matrix of ten federated algorithms covers 32 disease--partition configurations under five rounds of ten-percent participation. Three findings emerge from existing results and training logs. First, client ownership and participation jointly restrict long-window coverage: only 7.55\%--21.36\% of pooled-available 32-step windows have 
    
[^125]: 面向生物信息启发神经网络方程学习的超参数选择

    Hyperparameter selection for equation learning with biologically-informed neural networks

    [https://arxiv.org/abs/2610.02954](https://arxiv.org/abs/2610.02954)

    该论文提出了一种在真实控制方程未知时仍可使用的诊断式工作流程，用于生物信息神经网络（BINNs）方程学习中的超参数选择，从而提升结果的可重复性与方法的可迁移性。

    

    生物信息启发神经网络（BINNs）作为物理信息神经网络（PINNs）的一个灵活子类已经兴起，用于从数据中学习偏微分方程中的各项。BINNs特别适合生物系统，因为在这类系统中，控制方程高度非线性且仅有部分是先验已知的，而数据观测往往稀疏、含噪且不完整。然而，要在实践中有效应用BINNs，关键在于超参数的选择，而这仍然是方程学习框架中的一个核心挑战。超参数通常以启发式方式选取，且仅有粗略的记录，这限制了结果的可重复性和方法的可迁移性。我们提出了一种可用于真实方程未知情形下的超参数选择诊断工作流程。该工作流程由三个主要问题引导：（1）更大的网络容量所带来的收益是否值得……

    arXiv:2610.02954v1 Announce Type: new  Abstract: Biologically-informed neural networks (BINNs) have emerged as a flexible subclass of physics-informed neural networks (PINNs) for learning terms in partial differential equations from data. BINNs are particularly suited for biological systems, where the governing equations are highly nonlinear and only partially known a priori, and where data observations are often sparse, noisy, and incomplete. However, applying BINNs effectively in practice depends critically on hyperparameter selection, which remains a central challenge in equation-learning frameworks. Hyperparameters are often chosen heuristically and only cursorily documented, which limits the reproducibility of results and the transferability of methods. We present a diagnostic workflow for hyperparameter selection that can be used when the ground-truth equations are not known. The workflow is guided by three main questions: (1) Are the benefits of greater network capacity worth th
    
[^126]: SlimKV：基于无需重构信标注意力的令牌-特征联合KV缓存压缩

    SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention

    [https://arxiv.org/abs/2610.02953](https://arxiv.org/abs/2610.02953)

    SlimKV提出了一种与问题无关的令牌-特征联合KV缓存压缩方法，通过低秩感知训练将长上下文压缩为携带潜在KV表示的信标记忆状态，并利用键侧RoPE移除对信标令牌影响较小的位置不对称性，实现无需全维度重构的高效解码加速。

    

    长上下文大语言模型（LLM）服务日益受到KV缓存内存的瓶颈制约，尤其是在资源受限的场景中。在现有的KV缓存压缩策略中，基于令牌的方法虽能减少缓存状态，但通过淘汰或压缩会带来信息丢失的风险；而基于特征的方法虽能降低每个令牌的KV维度，却可能需要全维度重构才能应用位置编码，从而限制了解码加速。我们提出了SlimKV，一种与问题无关的令牌-特征联合KV缓存压缩方法。SlimKV采用低秩感知训练，将长上下文压缩为具有潜在KV表示的信标记忆状态，并结合层自适应秩分配。我们进一步揭示了一种位置不对称性：移除键侧RoPE对信标令牌和原始令牌的影响不同，信标令牌所受的性能下降要小得多。利用这种不对称性，SlimKV在无K-RoPE约束下训练信标KV投影，从而支持……

    arXiv:2610.02953v1 Announce Type: new  Abstract: Long-context LLM serving is increasingly bottlenecked by KV-cache memory, especially in resource-constrained scenarios. Among existing KV-cache compression strategies, token-wise methods reduce cached states but risk information loss through eviction or condensation, while feature-wise methods reduce per-token KV dimensions but can require full-dimensional reconstruction to apply positional embedding, limiting decoding speedups. We introduce SlimKV, a question-agnostic joint token-feature KV-cache compression method. SlimKV uses low-rank-aware training to compress long contexts into beacon memory states with latent KV representations, together with layer-adaptive rank allocation. We further uncover a positional asymmetry: removing key-side RoPE affects beacon and raw tokens differently, with much smaller degradation for beacon tokens. Exploiting this asymmetry, SlimKV trains beacon KV projections under a K-RoPE-free constraint and enable
    
[^127]: GTDD：面向AI编码智能体的生成式测试驱动开发与对抗测试

    GTDD: Generative Test-Driven Development for AI Coding Agents with Adversarial Testing

    [https://arxiv.org/abs/2610.02952](https://arxiv.org/abs/2610.02952)

    提出生成式测试驱动开发（GTDD），由独立的测试智能体在每轮候选实现后基于人类指定的行为契约对抗性地生成新测试输入并不断反馈反例，以解决AI编码智能体过拟合固定测试集而导致预期行为遗漏的问题。

    

    测试驱动开发为AI编码智能体提供了实现软件的可执行需求。由于这些智能体能够使其实现适应所观察到的示例，通过一组预先确定的测试可能会导致预期行为的相当大一部分未被实现。我们提出了生成式测试驱动开发（GTDD），这是测试驱动开发的一种形式化表述，其中在每次候选实现被固定后，一个独立的测试智能体会根据人类指定的行为契约和之前轮次的反馈生成新的输入。一个可信的评估器会检查这些输入，将简化的反例返回给编码智能体，并将其保存用于回归测试，从而使开发过程持续面对超出初始示例的失败情况。我们通过对自适应候选选择下错误接受的有限总体分析，刻画了该过程所提供的证据。由此得到的界……

    arXiv:2610.02952v1 Announce Type: cross  Abstract: Test-driven development gives AI coding agents executable requirements for implementing software. Because these agents can adapt their implementations to the examples they observe, passing a predetermined collection of tests can leave substantial parts of the intended behavior unimplemented. We propose Generative Test-Driven Development (GTDD), a formulation of test-driven development in which a separate testing agent generates new inputs after each candidate implementation is fixed, using a human-specified behavioral contract and the feedback from earlier rounds. A trusted evaluator checks these inputs, returns reduced counterexamples to the coding agent, and saves them for regression testing, so development continually confronts failures beyond the initial examples. We characterize the evidence that this process provides through a finite-population analysis of false acceptance under adaptive candidate selection. The resulting bounds 
    
[^128]: 面向多智能体系统的动态专家剪枝

    Dynamic Expert Pruning for Multi-Agent Systems

    [https://arxiv.org/abs/2610.02951](https://arxiv.org/abs/2610.02951)

    提出动态专家剪枝（DEP）方法，利用智能体的系统与任务提示动态识别并按需剪枝专家，解决了静态专家剪枝在多智能体异构工作负载下失效的问题。

    

    混合专家架构通过在每个 token 上仅激活少数专家来高效扩展语言模型，但这种节省仅限于计算方面：每个专家都必须驻留在加速器上，因此内存限制了这些模型可部署的范围。专家剪枝可以减少这种内存占用，然而现有方法都是静态的——一个在离线阶段校准的单一掩码，会被应用于模型后续的所有请求。当工作负载是异构的时，这一假设可能失效，最突出的体现就是多智能体系统：一个骨干网络同时服务于多个任务和角色。我们的分析表明，不同的任务和角色会启用不同的专家，而静态方法却为所有任务分配同一个固定的专家子集。因此，我们提出了动态专家剪枝（DEP），它基于我们在此确立的一个发现：智能体的系统提示和任务提示本身就足以识别出该智能体及其任务所需的专家……

    arXiv:2610.02951v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures scale language models efficiently by activating only a few experts per token, but the saving is confined to computation: every expert must stay resident on the accelerator, so memory bounds where these models can be deployed. Expert pruning reduces this footprint, yet existing methods are static --- a single mask, calibrated offline, is applied to the model for every subsequent request. This assumption can fail when the workload is heterogeneous, most prominently in multi-agent systems, where one backbone serves many tasks and roles at once: our analysis shows that different tasks and roles recruit different experts, while static methods assign one fixed subset to all of them. We therefore propose Dynamic Expert Pruning (DEP), which rests on a finding we establish here: an agent's system and task prompts are by themselves sufficient to identify the experts that agent and its task require, since th
    
[^129]: FASTDIAR：用于流式说话人分离的帧级说话人编码器

    FASTDIAR: Frame-level speaker encoder for Streaming Diarization

    [https://arxiv.org/abs/2610.02941](https://arxiv.org/abs/2610.02941)

    该论文提出FASTDIAR，一种因果帧级说话人编码器，每80毫秒输出一次嵌入并结合基于自相似性门控的在线聚类，实现了亚秒级延迟下最准确的流式说话人分离，在说话人数量增加时性能衰减更小，且可在CPU上以实时五倍的速度运行。

    

    实时对话智能体需要能够在CPU上流式运行的说话人分离技术。大多数系统将话语级说话人编码器应用于短且高度重叠的音频分块，这既浪费了计算资源，又使模型针对错误的任务进行优化。我们转而将最先进的说话人识别架构改造成因果的帧级编码器，它只需读取一次音频流，每80毫秒基于过去音频的有限两秒窗口输出一个嵌入向量，并将其与在线聚类相结合，该在线聚类根据音频流的自相似性对每次更新进行门控。该系统仅通过话语级教师模型在模拟和域外混合数据上的蒸馏进行训练，并使用一组固定的超参数进行评估，在低重叠基准测试中是亚秒级延迟下最准确的流式说话人分离系统；随着说话人数量增加，其性能下降远小于基于缓存的系统，并且运行速度达到实时的五倍。

    arXiv:2610.02941v1 Announce Type: cross  Abstract: Real-time conversational agents require speaker diarization that streams and runs on a CPU. Most systems apply an utterance-level speaker encoder to short, heavily overlapping chunks, which wastes computation and leaves the model optimized for the wrong task. We instead turn a state-of-the-art speaker recognition architecture into a causal frame-level encoder that reads the stream once and emits one embedding every 80~ms from a bounded two-second window of past audio, and pair it with online clustering that gates every update on the self-similarity of the stream. Trained only by distillation from an utterance-level teacher on simulated and out-of-domain mixtures, and evaluated with one fixed set of hyperparameters, the system is the most accurate streaming diarizer on low-overlap benchmarks at sub-second latency, degrades far less than cache-based systems as the number of speakers grows, and runs five times faster than real time on a s
    
[^130]: 判别代理基础设施验证套件中的测试夹具覆盖

    Discriminating Fixture Coverage in Agent-Infrastructure Verification Suites

    [https://arxiv.org/abs/2610.02928](https://arxiv.org/abs/2610.02928)

    用变异分析衡量“在正确实现上通过、在缺陷实现上失败”这一标准证据的价值，发现该证据无法保证测试夹具覆盖——即使修复后的套件仍有五个对抗性变异体存活，其中三个因没有任何夹具能激活它们而从未暴露。

    

    arXiv:2610.02928v1 公告类型： cross。不变量测试套件和运行时监控器越来越多地被用于把关代理的部署决策，而为任何特定套件提供的证据几乎总是一个单一观察结果：它在一个被认为正确的实现上通过，在一个被认为有缺陷的实现上失败。我们衡量这一观察结果究竟有多大价值。通过对一个面向多会话代理状态投影层的不变量套件应用变异分析，我们首先发现这种标准验证所认证的套件中，一个移除了事件身份去重逻辑的一阶变异体通过了全部检查而存活。随后，我们冻结修复后的包含十二项检查的套件，记录其哈希值，并用一位对抗性读者（该读者未设计该套件的任何测试夹具）指定的十个变异体对其运行一次：它击杀了其中五个。对五个存活变异体相对参考实现进行插桩分析表明，它们以两种截然不同的方式失败，而非一种：其中三个从未被激活，因为没有任何测试夹具提供能让变异代码表现出不同行为的输入。

    arXiv:2610.02928v1 Announce Type: cross  Abstract: Invariant suites and runtime monitors increasingly gate agent deployment decisions, and the evidence offered for any particular suite is almost always a single observation: it passes an implementation believed correct and fails one believed broken. We measure what that observation is worth. Applying mutation analysis to an invariant suite for a multi-session agent state-projection layer, we first find that this standard validation certifies a suite in which a first-order mutant removing event-identity deduplication survives every check. We then freeze the repaired twelve-check suite, record its hash, and run it once against ten mutants specified by an adversarial reader who designed none of its fixtures: it kills five. Instrumenting the five survivors against the reference shows they fail in two distinct ways, not one. Three are never activated, because no fixture supplies an input on which the mutated code behaves differently at all. 
    
[^131]: 基于交叉注意力条件控制学习爵士钢琴家风格

    Learning Jazz Pianist Style with Cross-Attention Conditioning

    [https://arxiv.org/abs/2610.02918](https://arxiv.org/abs/2610.02918)

    该研究表明预训练符号音乐Transformer的表征已能编码钢琴家身份，并进一步通过交叉注意力机制注入钢琴家身份嵌入，实现了以特定爵士钢琴家风格为条件的音乐生成，生成结果可被分类器高精度地归因于正确的钢琴家。

    

    爵士钢琴家会形成独特的风格特征，经验丰富的听众往往能在几秒钟内识别出来，但这种识别背后的特征却难以进行形式化描述。我们通过预训练符号音乐Transformer的视角来研究爵士钢琴家风格，表明其学习到的表征已经能够很好地编码钢琴家身份信息，在两个基准数据集上实现了高精度的钢琴家分类。随后，我们通过在学习的钢琴家身份嵌入上引入交叉注意力机制来增强该Transformer，使其能够以特定艺术家的风格为条件生成音乐。两种评估协议证实了该生成器捕捉到了有意义的风格结构：滑动窗口分类器能够持续地将条件生成的音乐延续归因于正确的钢琴家，远高于无条件基线；而完全基于合成生成数据训练的分类器，在识别12位真实钢琴家时达到了87%的片段级准确率和95%的歌曲级准确率（摘要在此处截断）。

    arXiv:2610.02918v1 Announce Type: cross  Abstract: Jazz pianists develop distinctive traits that experienced listeners can often identify within seconds, yet the features underlying this recognition resist formal description. We study jazz pianist style through the lens of a pretrained symbolic music transformer, showing that its learned representations already encode pianist identity well enough for highly accurate classification across two benchmarks. We then augment the transformer with cross-attention over learned pianist identity embeddings, enabling it to generate music conditioned on a specific artist's style. Two evaluation protocols confirm that the generator captures meaningful stylistic structure: a sliding-window classifier consistently attributes conditioned continuations to the correct artist, far above unconditioned baselines; and a classifier trained entirely on synthetic generations identifies real pianists across 12 classes with 87% chunk-level and 95% song-level accu
    
[^132]: 探测实验框架：语言模型陈旧数据强化学习比较的设置检查

    Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models

    [https://arxiv.org/abs/2610.02911](https://arxiv.org/abs/2610.02911)

    该论文提出PTH检查集，证明实验框架的细节（如PPO比率计算方式、数据种子传递、重放队列复用和损失归一化器实现）可以逆转陈旧数据强化学习方法比较的排名，强调了检查实验基础设施的必要性。

    

    在陈旧样本上训练语言模型的方法，通常通过与重要性校正基线的比较来评判。我们证明实验框架的细节可以逆转所观察到的方法排名，并提出了PTH（Probe The Harness，探测实验框架），这是一组使实验框架状态可见的检查方法。我们的案例是在verl和单GPU训练器上对SAN（一种无行为修正的方法）与截断重要性采样（TIS）进行比较，其中SAN最初在两个技术栈中都领先。实验框架的四个细节改变了这一比较：PPO比率是针对学习器自身重新计算的概率计算的；数据种子未传递到TIS分支；重放队列将其第一批数据复用于33次更新；两个损失归一化器与其描述不符。在每种情况下，被记录的量看起来都与正常工作的设置一致，而定义比较本身的量却未被检查。在检查实验框架之后，TIS匹配……（摘要被截断）

    arXiv:2610.02911v1 Announce Type: cross  Abstract: Methods for training language models on stale samples are judged by comparisons against importance-corrected baselines. We show that details of the experimental harness can reverse the observed ranking of methods, and we introduce PTH (Probe The Harness), a set of checks that makes the harness visible. Our case is a comparison between SAN, a behaviour-free method, and truncated importance sampling (TIS) on verl and in a single-GPU trainer, in which SAN first finished ahead in both stacks. Four details of the harness changed this comparison: the PPO ratio was taken against the learner's own recomputed probabilities, the data seed did not reach the TIS arm, the replay queue reused its first batch for 33 updates, and two loss normalisers differed from their description. In each case the logged quantity looked consistent with a working setup, while the quantity that defines the comparison went unchecked. With the harness checked, TIS match
    
[^133]: 频率并非敏感度：识别稀疏MoE大语言模型中的安全敏感专家

    Frequency Is Not Sensitivity Identifying Safety-Sensitive Experts in Sparse MoE LLM

    [https://arxiv.org/abs/2610.02910](https://arxiv.org/abs/2610.02910)

    该论文提出用路由器梯度敏感度（即序列损失对专家门控权重的敏感度）替代传统的激活频率来识别稀疏MoE大语言模型中的安全关键专家，实验表明该方法在五种架构上能更准确地预测抑制哪些专家会削弱模型的安全拒绝能力。

    

    抑制少量被路由选择的专家即可在不重新训练的情况下削弱稀疏混合专家语言模型的安全行为。因此，应该抑制哪些专家是一个安全问题，而常见的答案是激活频率，但频率衡量的是使用情况，而非影响力。我们测试了一种替代方法：路由器梯度敏感度，即序列损失对选择专家的门控权重的敏感度。在五种MoE架构上，我们基于500个良性提示和500个恶意提示分别按这两种信号对专家进行排序，并在两种预算下（相同专家数量和相同名义恶意路由流量1%-5%）在100个保留的恶意提示上测量模型的拒绝率。在每种预算下，基于路由器梯度选择的专家抑制在25个条件中的24个里比激活频率更能降低拒绝率，并且在全部25个条件中都超过十次随机试验的均值。最大效应出现在OLMoE中，其拒绝率从100个提示中的34个降至9个（相对下降73.53%），且没有……（原文截断）

    arXiv:2610.02910v1 Announce Type: new  Abstract: Suppressing a small set of routed experts can weaken the safety behavior of a sparse Mixture-of-Experts (MoE) language model without retraining. Which experts to suppress is therefore a security question, and the usual answer is activation frequency, but frequency measures use, not influence. We test an alternative: router-gradient sensitivity, the sensitivity of the sequence loss to the gate weights that select an expert. Across five MoE architectures, we rank experts by each signal on 500 benign and 500 malicious prompts and measure refusal on 100 held-out malicious prompts under two budgets: equal expert counts and equal nominal malicious routing traffic (1%-5%). Under each of the two budgets, router-gradient selection reduces refusals more than activation in 24 of 25 conditions, and more than a ten-trial random mean in all 25. The largest effect is in OLMoE, where refusals fall from 34 to 9 of 100 prompts (73.53% relative) with no de
    
[^134]: 约束感知训练

    Constraint-Aware Training

    [https://arxiv.org/abs/2610.02909](https://arxiv.org/abs/2610.02909)

    本文提出约束感知训练方法，将推理时由程序分析执行的约束从模型学习中外化出来，并通过三个定理证明这种外化能带来模型规模和数据效率上的收益，即在相同参数量下获得更低的预测损失。

    

    当使用语言模型生成程序时，约束解码可以应用程序分析来排除那些违反语法、作用域或类型规则的词元。然而，这里存在重复：标准训练本身已经教会模型去抑制这些分析所拒绝的词元。这种重复引出了一个问题：如果我们在推理时反正会执行某种分析来过滤掉一组词元，我们是否可以在训练阶段完全避免教模型学习这种分析？这种外部化是否能带来更高效的模型？本文定义了一个满足这种外部化需求的通用约束感知目标函数，并将外部化的收益形式化为关于模型规模和数据效率的三个具体定理。我们通过一个受控的合成实验表明，这些定理在训练动态中依然成立：在参数量匹配的情况下，约束感知训练能产生更低的预测损失……

    arXiv:2610.02909v1 Announce Type: new  Abstract: When generating programs with language models, constrained decoding can apply program analyses to exclude tokens that violate syntax, scope, or typing rules. However, there is a duplication: standard training already teaches the model to suppress the tokens rejected by these analyses. This duplication leads to the question: if we will perform some analysis to filter a set tokens out during inference anyways, can we avoid teaching the model the said analysis altogether during training, and does this externalization lead to more efficient models? This paper defines a general constraint-aware objective satisfying this externalization desideratum and formalizes the benefits of externalization into three concrete theorems about model size and data efficiency. We show, through a controlled synthetic experiment, that the theorems survive training dynamics: constraint-aware training yields lower prediction loss at a matched parameter count and d
    
[^135]: ResNet会路由吗？残差网络中的稀疏交互专家

    Do ResNets Route? Sparse Interaction Experts in Residual Networks

    [https://arxiv.org/abs/2610.02907](https://arxiv.org/abs/2610.02907)

    该论文通过Möbius反演将训练好的ResNet精确分解为各残差分支的独立贡献与高阶交互，发现ImageNet预训练模型的交互质量在第5阶和第10阶达到峰值，且主导交互随输入和预测类别动态变化，揭示了残差网络中存在隐式的稀疏“路由”行为。

    

    残差网络对每个输入都会执行每一个块，但其功能贡献未必是输入无关的。我们将训练好的ResNet形式化为定义在二值残差分支掩码上的集合函数，并应用Möbius反演将其输出精确分解为各个残差校正项以及高阶交互项。对于平滑的残差堆叠，我们证明在残差缩放条件下，每个固定的k路交互以O(λ^k)的量级缩放。对ImageNet预训练的ResNet-18和ResNet-34的穷举分析表明，交互质量分别在第5阶和第10阶达到峰值，而非集中在低阶。降低残差缩放系数会使交互谱向低阶偏移，但同时也会改变模型的预测结果。交互系数在幅值上高度集中但并非硬稀疏，且保持预测不变的稀疏性随网络深度增加而减弱。至关重要的是，主导性的交互在不同输入和不同预测类别之间各不相同。

    arXiv:2610.02907v1 Announce Type: new  Abstract: Residual networks execute every block for every input, yet their functional contributions need not be input independent. We formulate a trained ResNet as a set function over binary residual-branch masks and apply M\"obius inversion to decompose its output exactly into individual residual corrections and higher-order interactions. For smooth residual stacks, we show that each fixed $k$-way interaction scales as $\mathcal{O}(\lambda^k)$ under residual scaling. Exhaustive analysis of ImageNet-pretrained ResNet-18 and ResNet-34 reveals that interaction mass peaks at orders five and ten, respectively, rather than at low orders. Reducing the residual scale shifts both spectra toward lower orders, but also changes model predictions. The interaction coefficients are concentrated in magnitude but not hard sparse, and prediction-preserving sparsity weakens with depth. Crucially, the dominant interactions vary across inputs and predicted classes ar
    
[^136]: 切线薛定谔桥匹配：学习带有机制敏感性的随机输运

    Tangent Schr\"odinger Bridge Matching: Learning Stochastic Transport with Mechanistic Sensitivities

    [https://arxiv.org/abs/2610.02906](https://arxiv.org/abs/2610.02906)

    该论文提出切线薛定谔桥匹配方法，通过在轨迹传播中同时监督参数导数（机制敏感性）来学习随机输运，使模型能够准确预测随机系统在参数干预下的响应，并从理论上证明敏感性精度可界定有限变化预测误差与决策遗憾。

    

    预测随机系统对粘度、反应速率或外部力变化的响应需要昂贵的模拟，这激发了开发可重用学习模型的需求。然而，仅匹配观测到的结果分布并不能保证准确的干预响应预测。我们提出了切线薛定谔桥匹配（Tangent-SBM），该方法从端点观测和机制敏感性中学习随机输运。它在传播轨迹的同时传播参数导数，并根据提供的目标对这些导数进行监督。对于平均响应目标，单次rollout的平方误差还会额外惩罚响应的变异性；我们的目标函数采用两次独立的rollout来匹配均值，从而避免了这种额外的惩罚。我们建立了理论条件，证明在该条件下敏感性精度可以界定有限变化预测误差和决策遗憾。在高斯系统、随机双阱系统、PDEBench反应扩散系统以及随机Navier-Stokes系统中，Tangent-SB……

    arXiv:2610.02906v1 Announce Type: new  Abstract: Predicting how stochastic systems respond to changes in viscosity, reaction rates, or external forces requires costly simulations, motivating reusable learned models. Yet matching observed outcome distributions does not ensure accurate intervention responses. We introduce Tangent Schr\"odinger Bridge Matching (Tangent-SBM), which learns stochastic transports from endpoint observations and mechanistic sensitivities. It propagates parameter derivatives alongside trajectories and supervises them against supplied targets. For average-response targets, single-rollout squared error also penalizes response variability; our objective uses two independent rollouts to match the mean without this additional penalty. We establish conditions under which sensitivity accuracy bounds finite-change prediction error and decision regret. Across Gaussian, stochastic double-well, PDEBench reaction--diffusion, and stochastic Navier--Stokes systems, Tangent-SB
    
[^137]: 何时可以信任匹配原则？有限样本与模型不确定性下的鲁棒部署几何

    When Can We Trust the Matching Principle? Robust Deployment Geometry Under Finite-Sample and Model Uncertainty

    [https://arxiv.org/abs/2610.02894](https://arxiv.org/abs/2610.02894)

    该论文提出用信任比率 tau = ε/γ 来量化匹配原则何时可靠（估计投影匹配的漂移在 Davis-Kahan 分离区域内按 O(τ²) 增长），并据此设计置信度校准匹配（CCM）策略：τ 小时进行方向性匹配，τ 大时渐进各向同性扩散惩罚，从而在有限样本与模型不确定性下实现鲁棒部署。

    

    arXiv:2610.02894v1 公告类型：cross 摘要：只匹配你能识别的几何结构；否则将惩罚分散开。我们用信任比率 tau = epsilon / gamma（估计不确定性除以谱分离度）来量化这一决策。在线性二次匹配响应下，对于位于所选 top-r 部署子空间中的探针，估计投影匹配相对于oracle投影匹配的漂移按 tau^2 量级缩放——即在 Davis-Kahan 分离区域 tau < 1/2 内为 O(tau^2)，实际可用性取决于常数因子。置信度校准匹配（CCM）将 tau 转化为一种策略——当 tau 较小时采用方向性匹配，否则渐进地转为各向同性扩散——其阈值来自校准而非定理（匹配对应分离区域；软匹配主要是启发式的）。实验展示了两种情形，包括 UCI HAR 嵌入场景，其中始终匹配在每个单元上都比弃权表现更差。

    arXiv:2610.02894v1 Announce Type: cross  Abstract: Match only geometry you can identify; otherwise spread the penalty. We quantify that decision by the trust ratio tau = epsilon / gamma (estimation uncertainty over spectral separation). Under the linear-quadratic Matching response, oracle-relative drift between estimated and oracle projector matching scales as tau^2 for probes in the chosen top-r deployment subspace -- O(tau^2) in the Davis-Kahan separation region tau < 1/2, with practical usefulness depending on constants. Confidence-Calibrated Matching (CCM) turns tau into a policy -- directional when tau is small, progressively isotropic when not -- with thresholds from calibration, not from the theorem (match sits in the separation region; soft is mostly heuristic). Experiments show both regimes, including UCI HAR embeddings where always-match is worse than abstain on every cell.
    
[^138]: DyRA：面向深度神经网络高效矩阵乘法的动态残差近似方法

    DyRA: Dynamic Residual Approximation for Efficient Matrix Multiplication in DNNs

    [https://arxiv.org/abs/2610.02882](https://arxiv.org/abs/2610.02882)

    提出了输入自适应方法DyRA，通过在推理过程中动态近似并校正结构化权重近似引入的输出残差误差，直接优化输出的低秩因子，从而更高效、更精确地实现深度神经网络中的矩阵乘法近似。

    

    大规模基础模型在多种任务上取得了出色的性能，但其庞大的规模使得推理成本高昂，这主要归因于密集矩阵乘法。先前的工作通过用低秩分解等高效结构化形式替换密集权重矩阵来降低这一成本。然而，这些方法近似的是权重本身，而非决定推理精度的输出激活值。因此，权重空间中的微小误差可能被输入激活值放大，从而产生较大的输出误差。在本工作中，我们提出了DyRA，这是一种输入自适应方法，通过在推理过程中校正残差输出误差来改进结构化矩阵乘法的近似效果。我们证明，通过直接优化输出的低秩因子，可以更有效地近似矩阵乘法。DyRA基于这一洞见，动态地近似并校正由结构化权重近似所引入的输出误差。

    arXiv:2610.02882v1 Announce Type: cross  Abstract: Large-scale foundation models achieve strong performance across diverse tasks, but their size makes inference costly, largely due to dense matrix multiplications. Prior work reduces this cost by replacing dense weight matrices with efficient structured forms such as low-rank factorizations. However, these methods approximate weights rather than the output activations that determine inference accuracy. Consequently, small weight-space errors can be amplified by input activations, producing large output errors. In this work, we propose DyRA, an input-adaptive method that improves structured matrix multiplication approximation by correcting residual output errors during inference. We show that matrix multiplication can be approximated more effectively by directly optimizing low-rank factors of the output. DyRA builds on this insight by dynamically approximating and correcting the output error introduced by structured weight approximations
    
[^139]: 迈向全模态多模态图基础模型：一种拓扑驱动的绑定方法

    Toward Omni Multimodal Graph Foundation Model: A Topology-Driven Binding Approach

    [https://arxiv.org/abs/2610.02881](https://arxiv.org/abs/2610.02881)

    提出GraphBind，一种拓扑驱动的多模态图基础模型方法，利用图拓扑的稳定性作为结构参考和互补语义来源，将异构节点的丰富模态信息绑定到统一共享空间中，以解决现实多模态属性图中节点属性不完整的问题。

    

    多模态图基础模型（MGFMs）旨在从具有异构节点模态的大规模图中学习可泛化的表示。然而，现实世界中的多模态属性图（MAGs）通常包含不完整的节点属性，限制了可用预训练语料库的规模和多样性。此外，现有的MGFMs主要将图拓扑作为结构上下文来使用，忽视了其在指导多模态绑定和构建统一表示空间方面的作用。为应对这些挑战，我们提出了GraphBind，一种拓扑驱动的方法，利用图拓扑将丰富的模态信息绑定到统一共享空间中。GraphBind的动机源于图拓扑的稳定性，它为多模态绑定提供了结构参考和互补的语义信息。具体而言，GraphBind利用拓扑将自身语义和可靠的邻域语义组织到一个集成的全局共享空间中，从而实现多模态信息的有效融合与统一表示。

    arXiv:2610.02881v1 Announce Type: new  Abstract: Multimodal graph foundation models (MGFMs) seek to learn generalizable representations from large-scale graphs with heterogeneous node modalities. However, real-world Multimodal-Attributed Graphs (MAGs) often contain incomplete node attributes, limiting the scale and diversity of available pretraining corpora. Besides, existing MGFMs primarily incorporate graph topology as structural context, overlooking its role in guiding multimodal binding and shaping a unified representation space. To address these challenges, we propose GraphBind, a topology-driven approach that uses graph topology to bind rich modality information into a unified shared space. GraphBind is motivated by the stability of graph topology, which provides structural references and complementary semantic information for multimodal binding. Concretely, GraphBind uses topology to organize self semantics and reliable neighborhood semantics into a global shared space that inte
    
[^140]: 带符号网络中的同伴效应：通过正负关系分离影响

    Peer Effects in Signed Networks: Separating Influence Through Positive and Negative Ties

    [https://arxiv.org/abs/2610.02872](https://arxiv.org/abs/2610.02872)

    该论文提出SiDE双重稳健估计器，首次在带符号网络中通过区分正、负关系及其交互作用来识别同伴效应，并给出识别公式和双重稳健性保证。

    

    评估网络干预需要理解处理如何通过社会关系影响人们。在不区分支持性和敌对性关系的情况下，简单统计已受处理邻居的数量可能会掩盖方向相反的影响。我们定义了经由正关系和负关系产生的效应、二者的交互效应，以及在总处理量固定的情况下在两类关系之间重新分配处理所产生的符号构成效应，并给出了这些效应的识别公式。在符号盲分配下，我们展示了忽略符号会如何混淆两类关系的效应。我们提出了SiDE（符号暴露双重稳健估计器），它将针对符号的结果模型与由个体处理分配诱导的暴露概率相结合。我们建立了其估计得分的双重稳健性，并评估了考虑邻域重叠的近似区间。在六个真实带符号网络上的半合成实验证明了准确的效应估计，并检验了相关局限性（原文截断）。

    arXiv:2610.02872v1 Announce Type: new  Abstract: Evaluating network interventions requires understanding how treatment affects people through their social relationships. Counting treated neighbors without distinguishing supportive and antagonistic ties can conceal opposing influences. We define effects through positive and negative ties, their interaction, and a sign-composition effect of reallocating treatment between the two types at a fixed total, and give their identification formulas. Under sign-blind assignment, we show how ignoring signs mixes the effects of the two tie types. We propose SiDE (Signed-exposure Doubly robust Estimator), which combines sign-specific outcome models with exposure probabilities induced by individual treatment assignment. We establish double robustness of its score and assess approximate intervals that account for overlapping neighborhoods. Semi-synthetic experiments on six real signed networks demonstrate accurate effect estimation and examine the lim
    
[^141]: 子群体偏移与离群点污染下的分布鲁棒生存模型

    Distributionally Robust Survival Models under Subpopulation Shift and Outlier Contamination

    [https://arxiv.org/abs/2610.02868](https://arxiv.org/abs/2610.02868)

    本文提出一个分布鲁棒生存分析框架，通过外层最小化削弱离群样本的影响、内层最大化聚焦最不利的子群体，从而联合应对子群体偏移与离群点污染，并直接兼容Cox风险集等不可分解的生存损失。

    

    在分布偏移下学习鲁棒的生存模型是许多应用中一个重要但具有挑战性的问题。在异质性人群中，平均表现良好的模型在某些子群体上可能仍然表现不佳，而当训练数据被离群点污染时，这一问题会变得更加严重。在本文中，我们提出了一个新颖的生存分析分布鲁棒框架，能够同时应对潜在的子群体偏移和离群点污染。所提出的方法结合了外层最小化与内层最大化：外层最小化通过降低被污染样本的影响来选择一个精炼的名义分布，内层最大化则聚焦于最具挑战性的子群体。这一建模方式直接适用于不可分解的生存损失，同时保留了样本间的交互作用，包括Cox负偏对数似然中的风险集结构。我们开发了一种交替梯度（算法……）（原文在此截断）。

    arXiv:2610.02868v1 Announce Type: cross  Abstract: Learning robust survival models under distribution shift is an important but challenging problem in many applications. In heterogeneous populations, a model that performs well on average may still perform poorly on certain subpopulations, and this issue becomes even more severe when the training data are contaminated by outliers. In this paper, we propose a novel distributionally robust framework for survival analysis that jointly addresses latent subpopulation shift and outlier contamination. The proposed method combines an outer minimization that selects a refined nominal distribution by reducing the influence of contaminated samples and an inner maximization that focuses on the most challenging subpopulation. This formulation directly accommodates non-decomposable survival losses while preserving interactions across samples, including the risk-set structure of the Cox negative partial log-likelihood. We develop an alternating gradie
    
[^142]: DIVINE：基于多样指标重建的简单跨市场股票预训练

    DIVINE: Simple Cross-Market Stock Pretraining via Diverse Indicator Reconstruction

    [https://arxiv.org/abs/2610.02866](https://arxiv.org/abs/2610.02866)

    DIVINE 提出一种简单的跨市场股票预训练框架，通过从原始 OHLCV 历史数据重建 16 个标准技术指标衍生的 77 个目标作为监督信号，同时避免了未来监督的不确定性并保持与收益预测的对齐，仅用 0.05M 参数的轻量级编码器就在六个股票市场上取得了最强的平均投资组合表现。

    

    金融时间序列预训练通常从掩码观测、对比关系或未来结果中学习——然而现有目标函数难以同时避免未来监督的不确定性并保持与收益预测的对齐。我们提出了 DIVINE（DIVerse INdicator rEconstruction，多样指标重建），这是一个简单的跨市场预训练框架，从原始 OHLCV 历史数据中重建技术指标。技术指标由可观测的价格-成交量历史计算得出，在各市场间提供定义一致的监督信号，同时概括了多样化的市场动态，并与收益预测具有公认的相关性。在六个股票市场数据集上进行联合预训练，DIVINE 重建由 16 个标准指标衍生的 77 个目标，并仅将学习到的编码器迁移至下游的股票排序任务。在所有六个市场中，DIVINE 仅凭借轻量级的 0.05M 参数编码器即取得了最强的平均投资组合表现，优于……

    arXiv:2610.02866v1 Announce Type: new  Abstract: Financial time-series pretraining typically learns from masked observations, contrastive relations, or future outcomes---yet existing objectives struggle to simultaneously avoid future-supervision uncertainty and maintain return-prediction alignment. We propose DIVINE (DIVerse INdicator rEconstruction), a simple cross-market pretraining framework that reconstructs technical indicators from raw OHLCV history. Computed from observed price-volume history, technical indicators provide consistently defined supervision across markets while summarizing diverse market dynamics with established relevance to return prediction. Pretrained jointly on six-equity market datasets, DIVINE reconstructs 77 targets derived from 16 standard indicators and transfers only the learned encoder to downstream stock ranking. Across all six markets, DIVINE achieves the strongest average portfolio performance with a lightweight 0.05M-parameter encoder, outperforming
    
[^143]: 关于时间序列预测的机器遗忘研究

    On Unlearning for Time-series Forecasting

    [https://arxiv.org/abs/2610.02865](https://arxiv.org/abs/2610.02865)

    该论文针对时间序列预测中的机器遗忘问题，指出基于梯度的遗忘方法因被删除观测值会跨多个因果关联的预测窗口传播参数更新而不稳定，并探讨标签引导更新作为更可控的替代方案。

    

    时间序列预测被广泛应用于敏感领域。这些场景中的模型通常在纵向的用户级或实体级记录上进行训练，而这些记录可能因包含敏感或专有信息、或因传感器故障而被污染，事后需要被删除。为了在不进行昂贵重训练的情况下响应此类删除请求，机器遗忘（machine unlearning）作为一种保护隐私和数据治理的实用机制得到了广泛研究。然而，机器遗忘在时间序列预测中的应用尚未得到很好的实现，这主要是由于以下独特的挑战：基于梯度的遗忘可能不稳定，因为一条被删除的观测值会参与多个因果关联的预测窗口，导致参数更新传播到请求删除的区间之外，从而损害保留数据的预测性能。标签引导的更新提供了一种更可控的替代方案，但连续且上下文（摘要在此处被截断）

    arXiv:2610.02865v1 Announce Type: new  Abstract: Time-series forecasting is widely used in sensitive domains. Models in these settings are often trained on longitudinal user- or entity-level records, which may later require removal because they contain sensitive or proprietary information or have been corrupted by sensor failures. To address such deletion requests without costly retraining, machine unlearning has been widely studied as a practical mechanism for privacy protection and data governance. However, the application of machine unlearning to time series prediction has not yet been well realized; this is mainly due to the following unique challenges: Gradient-based unlearning can be unstable because a deleted observation participates in multiple causally connected forecasting windows, causing parameter updates to propagate beyond the requested interval and degrade retained forecasting utility. Label-guided updating offers a more controlled alternative, but continuous and context
    
[^144]: NeuroLens：从长期记录中学习神经语义的潜在嵌入

    NeuroLens: Learning Latent Embeddings of Neural Semantics from Chronic Recordings

    [https://arxiv.org/abs/2610.02864](https://arxiv.org/abs/2610.02864)

    提出基于JEPA框架的自监督模型NeuroLens，通过在潜在空间中学习去噪的语义表征，从长期神经记录中区分神经表征可塑性与记录不稳定性。

    

    理解神经活动如何表征高阶认知，以及这些表征如何随时间演变，长期以来一直是神经科学的核心追求。然而，现有的分析工具难以在长期神经记录中区分表征可塑性与记录不稳定性。在此，我们提出 NeuroLens（神经语义潜在嵌入），这是一种基于联合嵌入预测架构（JEPA）框架的自监督模型，能够从长期神经记录中学习去噪的、具有语义信息的潜在表征。一个自适应编码器将不断变化的神经群体映射到共同的潜在空间中，同时一个时间预测器学习可用于预测未来潜在状态的结构。通过在潜在空间中进行预测，NeuroLens 能够捕捉时间上可预测的结构，并降低对瞬态的、记录特异性的变异的敏感性。在小鼠和人类的长期皮层内数据中，该学习（摘要至此被截断）

    arXiv:2610.02864v1 Announce Type: new  Abstract: Understanding how neural activity represents higher-order cognition and how these representations evolve over time has long been a central pursuit in neuroscience. However, current analytical tools cannot easily distinguish representational plasticity from recording instability in chronic neural recordings. Here, we introduce NeuroLens (Latent Embeddings of Neural Semantics), a self-supervised model based on the Joint-Embedding Predictive Architecture (JEPA) framework that learns denoised, semantically informative latents from chronic neural recordings. An adaptive encoder maps changing neural populations into a common latent space, while a temporal predictor learns structure that supports prediction of future latent states. By predicting in latent space, NeuroLens captures temporally predictable structure and reduces sensitivity to transient, recording-specific variability. Across chronic intracortical data in mice and humans, the learn
    
[^145]: 联合嵌入预测世界模型中的反事实动作评估、观测瓶颈与表征几何

    Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models

    [https://arxiv.org/abs/2610.02860](https://arxiv.org/abs/2610.02860)

    该论文提出一种贯穿模拟器状态、栅格观测、目标嵌入与预测器输出的反事实动作评估协议，发现联合嵌入预测世界模型存在观测瓶颈并系统性低估动作路径——预测器对真实反事实动作的响应远小于对匹配的各向同性噪声扰动。

    

    潜在预测误差低并不能证明世界模型能够区分其动作所产生的后果。我们提出了一种评估协议，可以在模拟器状态、栅格观测、目标嵌入和预测器输出的全流程中追踪同一次干预。在受控的可变形物理测试平台中，精确的模拟器状态分叉揭示了不同层面的瓶颈。改变指令确实会改变粒子运动，但仍有41.5%的单步栅格观测对是完全相同的。观测损失并不能解释全部问题：在579个高可见度的反事实样本中，两个随机种子下预测器对目标响应的中位数分别为0.0051和0.0217，经方差归一化后降至0.0027和0.0116。而与目标反事实嵌入偏移相匹配的各向同性状态扰动，在同一批可见样本对上产生了190倍和53倍更大的预测器变化，从而将问题定位于动作路径的利用不足，而非预测器失效或全局性收缩。仅使用MSE的训练得到8（摘要在此截断）。

    arXiv:2610.02860v1 Announce Type: new  Abstract: Low latent prediction error does not establish that a world model distinguishes the consequences of its actions. We introduce an evaluation protocol that traces the same intervention through simulator state, raster observations, target embeddings, and predictor outputs. Exact simulator-state forks in a controlled deformable-physics testbed reveal distinct bottlenecks. Changed commands alter particle motion, yet 41.5% of one-step raster pairs are identical. Observation loss is not the whole explanation: among 579 high-visibility counterfactuals, median predictor-to-target response is 0.0051 and 0.0217 across two seeds, falling to 0.0027 and 0.0116 after variance normalization. An isotropic state perturbation matched to the target counterfactual embedding shift produces 190x and 53x larger predictor changes on the same visible pairs, isolating action-path under-use rather than a dead or globally shrunk predictor. MSE-only training gives 8.
    
[^146]: 置换鲁棒性还不够：多智能体Transformer策略中的动作坍缩

    Permutation Robustness Is Not Enough: Action Collapse in Multi-Agent Transformer Policies

    [https://arxiv.org/abs/2610.02848](https://arxiv.org/abs/2610.02848)

    本文发现低置换误差可能是假象——所有智能体选择相同动作也会显得鲁棒，提出用置换一致性指标与动作坍缩诊断（动作多样性、相同动作比例、最大动作频率）共同评估多智能体Transformer策略，并证明弱等变性惩罚能在提升置换鲁棒性的同时保持动作多样性。

    

    Transformer策略在多智能体机器人学习中很有吸引力，因为自注意力机制能够建模智能体之间的交互。然而，多智能体团队本质上是无序的，而Transformer通常将智能体作为有序的token序列进行处理。我们研究了这种不匹配在智能体顺序置换下对协作导航策略的影响。结果表明，仅凭低置换误差可能具有误导性：策略可能仅仅因为所有智能体都选择相同动作而显得“鲁棒”。因此，我们同时使用置换一致性指标和动作坍缩诊断方法来评估策略，包括动作多样性、相同动作比例和最大动作频率。PPO-ID基线能够产生非坍缩的行为但对顺序仍然敏感，而强等变性正则化仍可能引发同质化行为。弱等变性惩罚则能够在提升鲁棒性的同时，为N=3个智能体的团队保留更多样化的动作。

    arXiv:2610.02848v1 Announce Type: cross  Abstract: Transformer policies are attractive for multi-agent robot learning because self-attention can model interactions among agents. However, multi-agent teams are unordered, while transformers typically process agents as ordered token sequences. We study how this mismatch affects cooperative navigation policies under agent-order permutations. Our results show that low permutation error alone can be misleading: policies may appear robust simply because all agents choose the same action. We therefore evaluate policies using both permutation-consistency metrics and action-collapse diagnostics, including action diversity, same-action fraction, and maximum action frequency. A PPO-ID baseline yields non-collapsed behavior but remains order-sensitive, while strong equivariance regularization can still induce homogeneous behavior. A weak equivariance penalty improves the robustness while preserving more diverse actions for teams with \(N=3\) agents
    
[^147]: 开放团队多智能体强化学习中的人员流动正交信用分配

    Turnover-Orthogonal Credit Assignment for Open-Team Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2610.02847](https://arxiv.org/abs/2610.02847)

    本文提出流动正交信用分配（TOCA），一种面向开放团队多智能体强化学习的价值分解方法，通过将动作效应与外生人员流动效应分离，避免存活智能体因其无法控制的成员变动而受到错误的奖励或惩罚。

    

    开放团队多智能体强化学习研究的是智能体可以在一个回合中加入、离开或被替换的合作系统。在这类设定下，团队回报的变化既源于智能体选择了有用的动作，也源于活跃成员群体本身的变化。标准的中心化评论家与共享优势往往将这两种效应混合为一个标量的信用信号，导致存活的智能体可能因其无法控制的外生人员流动事件而获得奖励或惩罚。我们提出了流动正交信用分配（TOCA），这是一种面向开放团队的价值分解方法，将动作效应、纯粹的流动效应以及动作与流动的交互作用分离开来。在外生流动条件下，事件条件价值可以接受一种中心化分解，其事件条件基线在保留对那些使团队对未来人员替换具有鲁棒性的动作的信用分配的同时，去除纯粹的流动成分。我们通过（原文在此处截断）……实例化了这一思想。

    arXiv:2610.02847v1 Announce Type: cross  Abstract: Open-team multi-agent reinforcement learning studies cooperative systems in which agents may join, leave, or be replaced during an episode. In such settings, the team return changes both because agents choose useful actions and because the active population itself changes. Standard centralized critics and shared advantages often mix these two effects into one scalar credit signal, allowing surviving agents to be rewarded or penalized for exogenous turnover events outside their control. We introduce turnover-orthogonal credit assignment (TOCA), a value decomposition for open teams that separates action effects, pure turnover effects, and action--turnover interactions. Under exogenous turnover, the event-conditioned value admits a centered decomposition whose event-conditioned baseline removes the pure turnover component while preserving credit for actions that make the team robust to future replacements. We instantiate this idea with a 
    
[^148]: 理解强化学习中的富集

    Understanding Enrichment in Reinforcement Learning

    [https://arxiv.org/abs/2610.02846](https://arxiv.org/abs/2610.02846)

    本文从数学上揭示了RLVR中省略或截断重要性权重纠正会隐式地重新加权奖励，并将梯度误差分解为尺度、旋转和方差以解释其对学习的不同影响，同时提出了一种新颖的序贯蒙特卡洛权重纠正机制来缓和方差的指数级增长。

    

    当奖励稀疏时，使用可验证奖励的强化学习（RLVR）通常会借助提示或中间引导来生成更多成功的轨迹采样。这种富集会使策略梯度更新产生偏差，除非通过重要性权重进行纠正，但现有方法为了规避纠正带来的高方差，要么省略纠正，要么截断重要性权重。因此，在RLVR中对富集的轨迹采样进行纠正究竟会带来什么收益或损失，目前仍不清楚。我们从数学上证明，省略或截断纠正会隐式地对所定义的奖励进行重新加权，并将由此产生的梯度误差分解为尺度、旋转和方差三个部分，以解释它们对学习的不同影响。为了使纠正变得实用，我们开发了一种新颖的序贯蒙特卡洛（SMC）权重纠正机制，该机制在稳定性和粒子顺序假设下，能够缓和标准纠正方差在样本长度上的指数级复合增长……

    arXiv:2610.02846v1 Announce Type: new  Abstract: When rewards are sparse, reinforcement learning with verifiable rewards (RLVR) often uses hints or intermediate guidance to generate more successful rollouts. This enrichment biases policy-gradient updates unless corrected via importance weights, but existing methods omit correction or truncate importance weights in order to avoid the high variance of correction. It thus remains unclear what exactly is gained or lost in RLVR by correcting enriched rollouts. We show, mathematically, that omitted or truncated correction implicitly reweights the defined reward, and we decompose the resulting gradient error into scale, rotation, and variance to explain their distinct effects on learning. To make correction practical, we develop a novel sequential Monte Carlo (SMC) weight correction mechanism that, under stability and particle-order assumptions, tempers the exponential compounding of standard correction variance over the length of a sample to
    
[^149]: 探索数据分布之外的奇异新世界：系统行为、因果性税与非因果基础模型

    To Explore The Strange New World Beyond Data Distribution: System Behavior, Causality Tax, and Non-causal Base Model

    [https://arxiv.org/abs/2610.02839](https://arxiv.org/abs/2610.02839)

    该论文提出SBD框架，将数据分布之外的系统行为作为不可约的贝叶斯组件纳入证据下界，从理论上揭示了反直觉的“因果性税”现象，表明语言模型的因果性可能既非必要也非最优。

    

    我们证明，语言模型（LMs）的因果性可能既非必要也非最优。当系统行为（记为 $S$）作为第一性原理贝叶斯特征被纳入考量时，情况便是如此。这里的 $S$ 指的是数据空间之外的额外主导因素，且它们涉及耦合效应。尽管因果性是现代架构事实上的基础，但近期研究表明其与因果性之间存在持续的不匹配和矛盾，而这些问题在很大程度上源于系统行为而非数据分布。因此，我们提出了 SBD 框架，将 $S$ 作为证据下界（ELBO）的一个不可约组件纳入其中。SBD 从理论上揭示了一种反直觉的“因果性税”现象：由于对 $S$ 的忽视，因果性表现为一种带有额外结构性误差的次优近似。为应对潜在变量分析的挑战，我们通过……验证了 SBD 所预测的 $S$ 的影响（摘要在此处截断）。

    arXiv:2610.02839v1 Announce Type: cross  Abstract: We show that the causality of language models (LMs) may not be necessary nor optimal. This is the case when system behavior (denoted as $S$) is incorporated as a first-principle Bayesian feature. Here, $S$ refers to extra dominant factors beyond the data space, and they involve coupled effects. Despite being the de facto foundation of modern architecture, recent studies indicate persistent mismatches and contradictions with causality. These issues largely stem from system behavior rather than the data distribution. We therefore propose the SBD framework, which incorporates $S$ as an irreducible component of the evidence lower bound (ELBO). SBD theoretically reveals a counter-intuitive Causality Tax phenomenon, where causality emerges as a suboptimal approximation with an additional structural error, due to the obliviousness to $S$. To address the challenge of latent variable analysis, we validate the SBD-predicted impact of $S$ via imp
    
[^150]: 只工作不玩耍，聪明杰克也变傻：理解并预防可验证奖励强化学习（RLVR）中的灾难性策略坍缩

    All Work And No Play Makes Jack a Dull Boy: Understanding and Preventing Catastrophic Strategy Collapse in RLVR

    [https://arxiv.org/abs/2610.02835](https://arxiv.org/abs/2610.02835)

    该论文揭示了RLVR后训练中GRPO类算法发生灾难性策略坍缩的机制——优化目标会将概率质量集中到单一策略，而维持任务准确率又需要最低策略容量，二者的冲突导致坍缩，并提出镜像纠缠指数（MEI）作为轻量级的在线预警信号。

    

    在使用可验证奖励强化学习（RLVR）对大语言模型（LLM）进行后训练的过程中，GRPO类算法可能出现严重的后期坍缩。基于提示词的探测表明，这并非良性的策略修剪，而是一种有害的有效策略容量收缩，使得不同的推理策略变得越来越难以被模型访问。为了刻画这一现象，我们通过轨迹层面的策略更新交互来定义策略，并构建了一个融合优化动力学与信息论的统一理论框架。我们证明，主要的RLVR目标会使概率质量逐渐集中到单一策略上，而维持非平凡的任务准确率则需要一个最低的策略容量。这两个结果之间的冲突为灾难性坍缩提供了机制层面的解释。我们进一步推导出镜像纠缠指数（MEI）作为一种轻量级的在线预警

    arXiv:2610.02835v1 Announce Type: new  Abstract: During post-training of large language models (LLMs) with Reinforcement Learning with Verifiable Rewards (RLVR), GRPO-style algorithms can exhibit severe late-stage collapse. Prompt-based probing reveals that this is not benign strategic pruning, but a harmful contraction of effective strategy capacity that makes distinct reasoning strategies increasingly inaccessible. To characterize this phenomenon, we define strategies through trajectory-level policy-update interactions and develop a unified theoretical framework combining optimization dynamics and information theory. We prove that major RLVR objectives progressively concentrate probability mass onto a single strategy, while sustaining nontrivial task accuracy requires a minimum strategy capacity. The conflict between these two results provides a mechanistic explanation for catastrophic collapse. We further derive the {Mirrored Entanglement Index (MEI)} as a lightweight online warning
    
[^151]: FastOPD：面向轻量化VLA部署的在策略蒸馏

    FastOPD: On-Policy Distillation for Lightweight VLA Deployment

    [https://arxiv.org/abs/2610.02832](https://arxiv.org/abs/2610.02832)

    FastOPD提出了一种高效的在策略蒸馏框架，将流图单状态教师监督与自洽性目标相结合，把大规模VLA基础模型压缩为可实际部署的轻量化模型，并从理论上保证学生模型能恢复出与理想少步教师模型相当的分布。

    

    视觉-语言-动作（VLA）基础模型已迅速扩展规模以提升操作性能与泛化能力，但这种规模化带来了高昂的计算成本，使其在现实世界中的部署日益困难。现有方法通常通过设计更小的架构或减少基于流的策略中的迭代去噪步骤来缓解这一问题。在本工作中，我们提出FastOPD，一个从基础模型到轻量化模型的VLA框架，通过高效的在策略蒸馏实现大规模VLA的实际部署。具体而言，FastOPD采用流图来实现单状态教师监督，并将其与自洽性目标相结合，构建出一个能够学习教师动力学的紧凑学生模型。此外，我们从理论上证明，最小化该目标可使蒸馏得到的学生模型恢复出与理想少步教师模型所诱导分布相当的水平。

    arXiv:2610.02832v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) foundation models have scaled rapidly to enhance manipulation performance and generalizability, but this scaling incurs high computational costs that render real-world deployment increasingly challenging. Existing approaches typically mitigate this issue by designing smaller architectures or reducing the iterative denoising steps in flow-based policies. In this work, we propose FastOPD, a foundation-to-lightweight VLA framework that enables the practical deployment of large-scale VLAs through efficient on-policy distillation. Specifically, FastOPD adapts a flow map for single-state teacher supervision and combines it with a self-consistency objective to construct a compact student that learns the teacher dynamics. Furthermore, we theoretically demonstrate that minimizing this objective allows the distilled student to recover a distribution on par with that induced by an ideal few-step teacher model. We eval
    
[^152]: 大语言模型中的临床概念中心

    Clinical Concept Centers in LLMs

    [https://arxiv.org/abs/2610.02829](https://arxiv.org/abs/2610.02829)

    该论文首次将机制可解释性评估从文本层面扩展到潜在空间应用于临床决策支持，发现在全部十一个测试的开源大语言模型内部都存在专门的临床概念中心，即临床概念以可定位且被因果使用的表征形式存在于模型潜在空间中。

    

    大语言模型在临床环境中的应用日益增多。然而，针对这些模型可靠性与性能的研究几乎完全聚焦于语言层面，即对模型所“说”的内容进行评分。机制可解释性研究发现，潜在空间承载着比文本更高保真度的表征：内部表征所编码的信息远多于输出所表达的内容，而且模型陈述的推理过程也会系统性地遗漏那些因果驱动答案的特征。在临床决策支持领域，基于机制可解释性的模型行为评估尚未被探索。在这项工作中，我们将行为评估扩展到潜在空间，探究临床概念是否作为可定位、且被因果使用的表征存在于开源权重的大语言模型内部。我们在所测试的全部十一个开源模型的潜在空间中都发现了专门的临床概念中心。这些概念中心是内……（摘要在此处被截断）

    arXiv:2610.02829v1 Announce Type: new  Abstract: Large language models are increasingly used in clinical settings. However, research into the reliability and performance of these models has focused almost entirely on the language substrate, scoring what the model says. Mechanistic interpretability has found that the latent space carries a higher fidelity of representation than the text: internal representations not only encode substantially more than the output verbalizes, but the stated reasoning also systematically omits features that causally drive the answer. An evaluation of model behavior in terms of mechanistic interpretability has not been explored in clinical decision support. In this work, we extend behavioral evaluation into the latent space and ask whether clinical concepts exist as locatable, causally used representations inside open-weight LLMs. We find dedicated clinical concept centers in the latent space of all eleven open models we test. These concept centers are inte
    
[^153]: FSPO：面向预算约束的大语言模型强化学习后训练的策略一致风险与帕累托可行控制

    FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training

    [https://arxiv.org/abs/2610.02828](https://arxiv.org/abs/2610.02828)

    FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。

    

    自适应的大语言模型强化学习后训练会在训练过程中在线调整多个训练执行器，包括 rollout 温度、组大小、裁剪、KL 正则化、验证器分配以及更新预算。目前有三个相互耦合的问题尚未解决：从行为轨迹训练得到的未来风险模型未必能准确估计将要部署的控制器所引发的风险；基于已记录的状态-动作对校准的分数在经过选择性动作选择后可能出现校准失准；独立的单资源最小成本通常无法保证多资源延续的可行性。我们提出 FSPO，一种面向预算约束的大语言模型强化学习后训练的反馈状态控制器，能够联合解决这些问题。FSPO 学习一个策略一致的风险前瞻（risk-to-go）模型，其 Bellman 目标遵循与未来决策所用的同一个冻结控制器，并同时学习一个长程效用模型。决策条件化轨迹校准（DCTC）对风险进行校准……

    arXiv:2610.02828v1 Announce Type: new  Abstract: Adaptive LLM reinforcement-learning post-training changes multiple training actuators online, including rollout temperature, group size, clipping, KL regularization, verifier allocation, and update budget. Three coupled issues remain unresolved. A future-risk model trained from behavior trajectories need not estimate the risk induced by the controller that will be deployed; a score calibrated on logged state-action pairs can become miscalibrated after selective action choice; and independent per-resource minimum costs do not in general certify a feasible multi-resource continuation. We introduce FSPO, a feedback-state controller for budgeted LLM RL post-training that addresses these issues jointly. FSPO learns a policy-consistent risk-to-go model whose Bellman target follows the same frozen controller used for future decisions, together with a long-horizon utility model. Decision-conditioned trajectory calibration (DCTC) calibrates risk 
    
[^154]: 面向时序域泛化的自适应谱-Koopman动力学建模

    Adaptive Spectral-Koopman Dynamics Modeling for Temporal Domain Generalization

    [https://arxiv.org/abs/2610.02822](https://arxiv.org/abs/2610.02822)

    提出AdaSpecK框架，通过谱正则化Koopman动力学建模提取去噪的低频轨迹，并结合上下文感知的异构模式提取机制，有效解决时序域泛化中的噪声过拟合与非平稳历史环境建模问题。

    

    时序域泛化旨在应对随时间发生分布变化的现实世界流式数据。然而，现有方法要么容易在数据空间中过拟合领域特定的噪声，要么在参数空间中变得过于复杂且可解释性较差。为了弥合这些差距，我们提出了AdaSpecK，一个用于时序域泛化的、具有自适应上下文提取能力的谱-Koopman框架。为了缓解对不规则采样领域的噪声拟合问题，我们引入了谱正则化的Koopman动力学建模，该方法在潜在空间中应用谱感知滤波以提取去噪后的低频轨迹，并学习一个Koopman算子在线性化空间中建模系统动力学。为了在非平稳条件下建模复杂的历史环境，我们设计了一种上下文感知的异构模式提取机制。具体而言，我们采用目标条件注意力模块来关注不同的历史模式……

    arXiv:2610.02822v1 Announce Type: cross  Abstract: Temporal Domain Generalization (TDG) has emerged to address real-world streaming data with distribution shifts over time. However, existing methods are either prone to overfitting to domain-specific noise in the data space or become overly complex and less interpretable in the parameter space. To bridge these gaps, we propose \textbf{AdaSpecK}, a spectral-Koopman framework with adaptive context extraction for TDG. To mitigate noise fitting to irregularly sampled domains, we introduce spectral-regularized Koopman dynamics modeling, which applies spectral-aware filtering in the latent space to extract denoised low-frequency trajectories and learn a Koopman operator to model the system dynamics in a linearized space. To model complex historical environments under non-stationarity, we design a context-informed heterogeneous pattern extraction mechanism. Specifically, we employ a target-conditioned attention module to attend to distinct pas
    
[^155]: RMCW：一种基于里德-马勒码的语言模型抗删除鲁棒水印

    RMCW: A Deletion-Robust Watermark Based on Reed--Muller Codes for Language Models

    [https://arxiv.org/abs/2610.02817](https://arxiv.org/abs/2610.02817)

    提出了一种基于里德-马勒码的大语言模型水印方法RMCW，通过密钥词汇划分注入水印结构，并利用局部子序列的里德-所罗门代数一致性检验，实现对删除攻击的鲁棒水印检测。

    

    大语言模型（LLM）水印为识别由特定模型生成的文本提供了一种轻量级机制，但其在后处理攻击下的鲁棒性仍然脆弱。删除攻击尤其具有挑战性，因为它们会改变词元的位置，破坏观测到的词元与其原始水印位置之间的对齐关系。我们提出了里德-马勒码水印（RMCW），一种基于里德-马勒码的LLM水印方法。与全局码字恢复不同，RMCW搜索存留的局部代数结构，利用里德-马勒码字的仿射线约束所诱导的里德-所罗门一致性。在生成阶段，RMCW通过密钥词汇划分将里德-马勒结构注入序列；在检测阶段，它将给定文本映射到带密钥的词汇桶，并使用Berlekamp–Welch测试对局部子序列进行低次里德-所罗门一致性检验。

    arXiv:2610.02817v1 Announce Type: cross  Abstract: Large Language Model (LLM) watermarking provides a lightweight mechanism for identifying text generated by a specific model, but its robustness remains fragile under post-processing attacks. Deletion attacks are particularly challenging because they shift token positions and break the alignment between observed tokens and their original watermark positions. We propose Reed--Muller Code Watermarking (RMCW), an LLM watermarking method based on Reed--Muller codes. In contrast to global codeword recovery, RMCW searches for surviving local algebraic structure, leveraging the Reed--Solomon consistency induced by affine-line restrictions of Reed--Muller codewords. During generation, RMCW injects a Reed--Muller structure into the sequence via a secret-keyed vocabulary partition. During detection, it maps the given text to keyed vocabulary bins and tests local subsequences for low-degree Reed--Solomon consistency using Berlekamp--Welch tests. E
    
[^156]: 门控槽位注意力-2：线性注意力中的双向联想记忆校正

    Gated Slot Attention-2: Two-Sided Associative Memory Correction in Linear Attention

    [https://arxiv.org/abs/2610.02816](https://arxiv.org/abs/2610.02816)

    该论文提出门控槽位注意力-2（GSA2），通过门控Oja规则-2实现键侧记忆校正、门控Delta规则-2实现值侧记忆校正，首次在线性注意力中实现了双向联想记忆校正，从而更有效地管理固定大小的循环记忆。

    

    线性注意力模型已成为标准注意力的高效替代方案，但如何有效管理其固定大小的循环记忆仍然是一个挑战。为了改进记忆机制，近期工作探索了两个不同的方向：一是用于精确校正与键相关联的值的Delta规则变体，二是以门控槽位注意力为代表的基于槽位的架构，用于分两个阶段建模键记忆和值记忆。我们观察到这两个方向是互补的——Delta规则提供了有效的记忆校正能力，而两阶段结构则提供了一种在关联的两侧进行操作的自然方式。基于这一洞察，我们提出了一种用于键侧校正的新颖门控Oja规则，并通过解耦的擦除与写入控制将其扩展为门控Oja规则-2。随后，我们提出了门控槽位注意力-2（GSA2），它将用于键侧校正的门控Oja规则-2与用于值侧校正的门控Delta规则-2相结合……

    arXiv:2610.02816v1 Announce Type: new  Abstract: Linear attention models have emerged as efficient alternatives to standard attention, but effectively managing their fixed-size recurrent memory remains challenging. To improve memory, recent work has explored two distinct directions: delta-rule variants for precise correction of values associated with keys, and slot-based architectures such as Gated Slot Attention for modeling key and value memories in two stages. We observe that these directions are complementary--the delta rule provides effective memory correction, while the two-stage structure provides a natural way to operate on both sides of an association. Building on this insight, we introduce a new Gated Oja Rule for key-side correction and extend it with decoupled erase and write control to obtain Gated Oja Rule-2. We then introduce Gated Slot Attention-2 (GSA2), which combines Gated Oja Rule-2 for key-side correction with Gated Delta Rule-2 for value-side correction through sh
    
[^157]: ROUTEAUDIT：面向预算受限多验证器路由的交互感知识别方法

    ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

    [https://arxiv.org/abs/2610.02808](https://arxiv.org/abs/2610.02808)

    ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。

    

    自适应多验证器系统通常通过端点的质量-成本差距进行比较，即使验证器目录、可用性、资源核算、信息过滤或评分器会随策略发生变化。我们将验证器路由形式化为一个契约条件化的识别问题。该契约记录了请求支持、验证器目录、实际可用性、资源核算、在线过滤以及轨迹后评分；一个匹配的路由对比仅改变策略坐标。ROUTEAUDIT为该契约增加了三个可度量的对象：契约格在所有可容许的桥接顺序上对坐标增量取平均，并报告由此得到的归因及其路径敏感性；策略无关的响应带在自适应策略揭示不同观测时识别成对的顺序对比；对于不完整的匹配，请求级边界利用仍然可观测的潜在结果，给出紧致的有限（摘要在此处截断）。

    arXiv:2610.02808v1 Announce Type: new  Abstract: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp fi
    
[^158]: VIGOR：基于模型的强化学习中通过潜空间一致性实现零样本视觉泛化

    VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.02801](https://arxiv.org/abs/2610.02801)

    VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。

    

    基于模型的强化学习（MBRL）通过在学习的潜在动力学中进行规划，实现了强大的样本效率，但在面对背景变化、光照变化或相机移动等未见过的视觉干扰时，其性能会大幅下降。与无模型强化学习（编码器扰动仅影响单步预测）不同，MBRL存在两级脆弱性：视觉干扰首先使编码器输出偏离分布，随后这些误差会在规划时域内通过递归潜在轨迹推演不断累积放大。我们提出VIGOR，这是一个能够在保留其MBRL骨干样本效率的同时，实现对未见视觉干扰进行零样本泛化的框架。VIGOR集成了三个相互关联的组件：（i）非对称弱到强增强，在单个批次内配对仅弱增强与弱到强增强的潜在视图……

    arXiv:2610.02801v1 Announce Type: new  Abstract: Model-based reinforcement learning (MBRL) achieves strong sample efficiency by planning within learned latent dynamics, yet its performance degrades substantially under unseen visual distractions such as background variations, lighting changes, or camera shifts. Unlike model-free RL, where encoder perturbations affect only single-step predictions, MBRL suffers from a two-level vulnerability: visual distractions first push encoder outputs out of distribution, and these errors then compound through recursive latent rollouts over the planning horizon. We propose visual generalization via latent-space consistency in model-based RL (VIGOR), a framework that enables zero-shot generalization to unseen visual distractions while retaining the sample efficiency of its MBRL backbone. VIGOR integrates three interdependent components: (i) asymmetric weak-to-strong augmentation, which pairs weak-only and weak-to-strong latent views within a single bat
    
[^159]: Muon 更擅长学习事实：理解谱正交化的作用

    Muon Learns Facts Better: Understanding the Role of Spectral Orthogonalization

    [https://arxiv.org/abs/2610.02798](https://arxiv.org/abs/2610.02798)

    本文通过可解析的事实回忆模型和线性 Transformer 分析了 Muon 优化器中谱正交化的作用机制，揭示其对特征学习动力学的改变，并说明这使 Muon 比梯度下降和 Adam 更擅长学习“主体-关系到答案”的事实映射。

    

    arXiv:2610.02798v1 通告类型：新 摘要：Muon 优化器对矩阵形式的更新应用谱正交化，并已在大规模神经网络训练中展现出优异的性能，然而这一变换在特征学习中的作用机制仍鲜为人知。在本工作中，我们通过一个可解析的事实回忆模型来研究这一问题：其中每条事实将每个“主体-关系”对映射到一个答案，而一个线性 Transformer 学习恢复该映射所需的、依赖于主体和依赖于关系的信息。该 Transformer 分别采用梯度流（GF）、谱 GF 或符号 GF 进行优化，它们分别是梯度下降、Muon 和 Adam 的连续时间极限。先前的研究（Nichani et al., 2025）已表明，当主体数量超过关系数量时，GF 会先学习依赖于关系的信息，后学习依赖于主体的信息，从而在训练过程中产生一个特征分离阶段。我们对该分离阶段进行了刻画……（原文摘要在此处截断）

    arXiv:2610.02798v1 Announce Type: new  Abstract: The Muon optimizer applies spectral orthogonalization to matrix-valued updates and has shown strong performance in large-scale neural network training, yet the mechanisms of this transformation in feature learning remain poorly understood. In this work, we investigate this question through a tractable factual-recall model, where a fact maps each subject-relation pair to an answer, and a linear transformer learns the subject- and relation-dependent information required to recover this mapping. The transformer is optimized with gradient flow (GF), spectral GF, or Sign GF, which are continuous-time limits of gradient descent, Muon, and Adam, respectively. Prior studies (Nichani et al., 2025) have shown that when the number of subjects exceeds the number of relations, GF learns relation-dependent information before subject-dependent information, producing a feature-separation phase during training. We characterize this separation with the le
    
[^160]: 非平稳分布偏移下图学习的高效记忆结晶

    Efficient Memory Crystallization for Graph Learning under Non-Stationary Distribution Shifts

    [https://arxiv.org/abs/2610.02795](https://arxiv.org/abs/2610.02795)

    该论文提出了一种无需训练的测试时框架EMC，通过闭式解将每个新到来的图域高效地“结晶”为紧凑且语义忠实的记忆，以低成本替代生成式记忆方案，从而应对图学习中的持续分布偏移。

    

    部署在现实世界系统中的深度图学习模型通常需要应对非平稳环境，其中底层的图分布会随时间持续漂移。主流的解决方案依赖于训练辅助生成模块来合成记忆图以实现跨域适应，这会带来大量的计算开销，并且在长期分布偏移下扩展性较差。我们认为存在一条更经济的路径：与其生成记忆，不如将其“结晶”。为此，我们提出了高效记忆结晶（EMC），这是一个无需训练的测试时框架，它通过对面向记忆的分布匹配目标的闭式解，将每个传入的图域蒸馏为紧凑且语义忠实的记忆，从而在持续协变量偏移下消除冗余的域信息。为了在模型遍历长序列时保持泛化性与适应性……（摘要在此处截断）

    arXiv:2610.02795v1 Announce Type: new  Abstract: Deep graph learning models deployed in real-world systems often need to cope with non-stationary environments, where the underlying graph distribution drifts continually over time. Prevailing solutions rely on training auxiliary generative modules to synthesize memory graphs for cross-domain adaptation, which incurs substantial computational overhead and scales poorly under prolonged distribution shifts. We argue that a more economical path exists: rather than generating memory, one can crystallize it. To this end, we propose Efficient Memory Crystallization (EMC), a training-free test-time framework that distills each incoming graph domain into a compact, semantically faithful memory through a closed-form solution to a memory-oriented distribution-matching objective, thereby eliminating redundant domain information under continual covariate shifts. To preserve both generalizability and adaptability as the model traverses a long sequence
    
[^161]: 用于高效高斯DAG学习的留出评分法

    Hold-Out Scoring for Efficient Gaussian DAG Learning

    [https://arxiv.org/abs/2610.02785](https://arxiv.org/abs/2610.02785)

    该论文提出HOST算法，以逐节点留出评分与凸回归取代子集搜索，仅凭对得分误差的单侧控制即可恢复正确的节点排序，从而在无需入度上界的情况下实现高效的高斯DAG学习。

    

    高维高斯DAG（有向无环图）学习面临着统计与计算之间的鸿沟：具有精细样本复杂度的方法依赖于计算代价高昂的子集搜索以及需要预先给定的入度上界，而多项式时间的替代方法则具有较差的样本复杂度。我们提出了HOST，一种高效的DAG学习算法，它用逐节点的留出评分和凸回归取代子集搜索，且无需预先给定入度上界。我们的关键洞察是：恢复正确的节点排序并不需要在排序得分上具有一致小的估计误差，而只需要对这些误差实施单侧控制。在排序步骤中，HOST利用了如下事实：使用留出样本进行的得分估计在期望意义上会抬高排序得分，这对于那些尚不应被选中的候选节点而言恰好是有利的误差方向。在给定排序之后，HOST通过递归地从两个节点之间的总效应中剔除间接效应来恢复父节点（摘要在此处截断）。

    arXiv:2610.02785v1 Announce Type: cross  Abstract: High-dimensional Gaussian DAG learning faces a statistical-computational gap: methods with sharp sample complexity rely on computationally expensive subset search and a supplied indegree bound, whereas polynomial-time alternatives have less favorable sample complexity. We introduce HOST, an efficient DAG learning algorithm that replaces subset search with nodewise hold-out scoring and convex regression, without requiring a supplied indegree bound. Our key insight is that recovering a correct ordering does not require uniformly small estimation errors in ordering scores but only one-sided control of those errors. In the ordering step, HOST exploits the fact that score estimation using hold-out samples inflates ordering scores in expectation, which is the favorable direction for candidates that should not yet be selected. Given the ordering, HOST recovers parents by recursively removing indirect effects from total effects between two nod
    
[^162]: RL之前的OPD：利用在线策略蒸馏为基于评分标准的强化学习进行热启动

    OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation

    [https://arxiv.org/abs/2610.02781](https://arxiv.org/abs/2610.02781)

    提出两阶段训练框架：先以评分标准作为教师特权上下文进行在线策略蒸馏（RP-OPD）提供密集的token级监督，再以评分标准作为奖励进行强化学习，从而突破蒸馏的性能瓶颈。

    

    许多有用的语言模型任务无法通过精确的结果验证来评估。基于评分标准的强化学习（RL）通过根据明确标准对开放式回答进行评分来解决这一问题。然而，由于奖励是在完整回答生成之后才分配的，训练信号无法直接识别是哪些具体决策对最终得分做出了贡献。我们提出了一个两阶段训练框架：首先将评分标准用作特权教师上下文以提供密集的token级监督，然后将其用作奖励进行进一步的RL。在第一阶段，评分标准特权在线策略蒸馏（RP-OPD）让无法访问评分标准的学生模型在学生生成的前缀处匹配具备评分标准意识的教师模型的下一个token分布。在第二阶段，RL直接优化评分标准奖励，并突破了蒸馏带来的性能平台期。我们使用开源权重模型在健康和科学任务上评估了该框架。（摘要原文在此处截断）

    arXiv:2610.02781v1 Announce Type: cross  Abstract: Many useful language-model tasks cannot be evaluated by exact outcome verification. Rubric-based reinforcement learning (RL) addresses this issue by scoring open-ended responses against explicit criteria. However, because the reward is assigned after the complete response, the training signal does not directly identify which individual decisions contributed to the final score. We propose a two-stage training framework that uses rubrics first as privileged teacher context for dense token-level supervision, then as rewards for further RL. In the first stage, rubric-privileged on-policy distillation (RP-OPD), a student without access to the rubric matches a rubric-aware teacher's next-token distributions at student-generated prefixes. In the second stage, RL directly optimizes the rubric reward and improves beyond the observed distillation plateau. We evaluate the framework on health and science tasks using open-weight models. Across Heal
    
[^163]: 控制极向暴露以延缓扩散模型中的记忆化

    Controlling Polar Exposure to Delay Memorization in Diffusion Models

    [https://arxiv.org/abs/2610.02780](https://arxiv.org/abs/2610.02780)

    该论文提出质量门控去白化（QGD）控制器，通过管理优化的更新几何、渐进恢复固定增益动量，在保持生成质量的同时将扩散模型的记忆化（复制训练样本）延迟至与数据集规模成比例。

    

    扩散模型可以在复制训练样本之前达到有用的生成质量，但快速优化会通过加速对特定样本的拟合而压缩这一泛化窗口。我们通过更新几何来研究这一效应，并提出了质量门控去白化（Quality-Gated De-whitening, QGD），这是一个保留快速极向更新前缀、并渐进恢复固定增益动量的控制器。我们的随机特征分析区分了由协方差控制、曲率均衡和幅值控制的三种记忆化时钟。在对齐的谱假设下，分析建立了一个有限暴露条件：在该条件下，固定增益的尾部能够恢复与数据集规模成比例的延迟。QGD通过一个经过确认的质量门、有界衰减包络和因果复制反馈来实现这一原则。立即切换是保守极限；渐进式控制则在延缓复制与持续提升质量之间取得平衡。我们还将QGD与复制预算选择相结合

    arXiv:2610.02780v1 Announce Type: new  Abstract: Diffusion models can reach useful sample quality before copying training examples, but fast optimization can compress this generalization window by accelerating sample-specific fitting. We investigate this effect through update geometry and propose Quality-Gated De-whitening (QGD), a controller that retains a fast polar-update prefix and progressively restores fixed-gain momentum. Our random-feature analysis separates covariance-controlled, curvature-equalized and amplitude-controlled memorization clocks. Under aligned spectral assumptions, it establishes a finite-exposure condition under which a fixed-gain tail recovers a delay proportional to dataset size. QGD implements this principle with a confirmed quality gate, a bounded decay envelope and causal copy feedback. Immediate switching is the conservative limit; gradual control balances delayed copying against continued quality improvement. We pair QGD with Copy-Budgeted Selection (CBS
    
[^164]: LatticeSMC：分块序列生成器中推理时计算资源应投向何处

    LatticeSMC: Where to Spend Inference-Time Compute in Chunked Sequence Generators

    [https://arxiv.org/abs/2610.02774](https://arxiv.org/abs/2610.02774)

    该论文提出 LatticeSMC，一个在“块索引×去噪步骤”二维格点上运行的 Feynman-Kac 采样器，从理论上证明了块可加奖励下重采样应安排在前瞻代价最低之处、且前缀可评估奖励可作为精确势函数直接使用，从而实现了预算严格匹配、更有效的分块序列生成推理时引导。

    

    面向音乐、动作和视频的长序列生成器逐块（chunk by chunk）地生成序列，每个块通过迭代去噪生成，而奖励则定义在整个序列上。现有的推理时引导方法通常一次只作用于一个维度：在最后进行 best-of-N 选择、跨去噪步进行 Feynman-Kac 引导，或跨块进行流式剪枝，而且这些方法往往在算力不匹配或回报规则不一致的条件下进行比较。我们提出了预算匹配的分块引导，并提出 LatticeSMC——一种源自 Feynman-Kac 模型的采样器，其作用于块索引与去噪步骤构成的二维格点之上。两项伸缩求和（telescoping）结果使其设计具有精确性：对于块可加奖励，两个维度诱导出相同的权重，因此重采样应发生在前瞻代价最低的地方；对于终止奖励，任意前缀评分都可定义一个精确的中间势函数，使得可评估前缀的奖励可以直接作为扭曲使用，无需任何估计或额外的去噪器调用。

    arXiv:2610.02774v1 Announce Type: new  Abstract: Long-form generators for music, motion and video produce sequences chunk by chunk, with each chunk generated by iterative denoising while rewards are defined over the full sequence. Existing inference-time steering methods typically act on one axis at a time: best-of-N at the end, Feynman-Kac steering across denoising steps, or streaming pruning across chunks, and are often compared under unmatched compute or different return rules. We introduce budget-matched chunked steering and propose LatticeSMC, a sampler derived from a Feynman-Kac model on the two-dimensional lattice of chunk index and denoising step. Two telescoping results make its design exact: for chunk-additive rewards, the two axes induce identical weights, so resampling should occur where lookahead is cheapest; for terminal rewards, any prefix score defines an exact intermediate potential, making prefix-evaluable rewards twists with no estimation or extra denoiser calls. Lat
    
[^165]: 具有1比特反馈的近最优固定置信度最优臂识别

    Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback

    [https://arxiv.org/abs/2610.02771](https://arxiv.org/abs/2610.02771)

    本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。

    

    我们研究在严格1比特反馈约束下的固定置信度最优臂识别问题。在每一轮中，学习者选择一个臂和一个查询集合，并且仅接收一个比特，该比特指示采样的奖励是否属于该集合。我们考虑一种具有逐臂定位的无分布有限方差设置，在这种设置下，直接的经验均值估计不再可用，截断处理变得不可避免。我们首先基于随机化阈值查询和截断尾积分恒等式，构建了一个时间一致的1比特均值估计基元。随后，我们将该基元嵌入到候选-挑战者式的最优臂识别算法中。固定截断算法提供了简单的任意时刻（ε,δ)-PAC保证，而分阶段自适应截断算法则将截断水平与当前分辨率相匹配，从而产生了间隙自适应的样本复杂度。我们还证明了一个K臂最坏情况的信息论下界（摘要原文在此处截断）。

    arXiv:2610.02771v1 Announce Type: cross  Abstract: We study fixed-confidence best-arm identification under strict 1-bit feedback constraints. At each round, the learner selects an arm and a query set, and receives only a single bit indicating whether the sampled reward belongs to that set. We consider a distribution-free finite-variance setting with arm-wise localization, where direct empirical mean estimation is no longer available and clipping becomes unavoidable. We first formulate a time-uniform 1-bit mean-estimation primitive based on randomized threshold queries and a clipped tail-integral identity. We then embed this primitive into candidate-challenger best-arm identification algorithms. A fixed-clipping algorithm gives a simple anytime $(\epsilon,\delta)$-PAC guarantee, while a phased adaptive-clipping algorithm matches the clipping level to the current resolution and yields a gap-adaptive sample complexity. We also prove a $K$-arm worst-case information-theoretic lower bound s
    
[^166]: 没有免费的图：学习多模态数据何时应该被图化

    No-Free-Graph: Learning When Multimodal Data Should Be Graphified

    [https://arxiv.org/abs/2610.02768](https://arxiv.org/abs/2610.02768)

    本文揭示多模态数据的图化并非总是有益，并提出MAG-SCOUT框架，在构建图之前评估其预期效用，将图构建从默认步骤转变为一种基于效用的选择性决策。

    

    多模态图学习近来已成为一种将实体间关系融入多模态表示的有效范式。现有研究在如何构建和优化图方面取得了长足进展，但很少考虑一个更根本的问题：对于给定的数据集和任务，是否应该引入额外的关系结构。通过在不同数据集、任务和图构建方法上进行实证研究，我们揭示了图化并非始终有益：引入关系结构在某些情况下能带来显著提升，而在另一些情况下收益有限甚至为负。这一观察促使我们提出一种新视角：图构建应当被视为基于预期效用的选择性决策，而非默认的预处理步骤。为解决这一问题，我们提出了MAG-SCOUT，一个构建前的图评估框架，用于估计……（原文摘要在此处截断）

    arXiv:2610.02768v1 Announce Type: new  Abstract: Multimodal graph learning has recently emerged as an effective paradigm for in corporating inter-entity relationships into multimodal representations. Existing studies have made substantial progress on how to construct and optimize graphs, but rarely consider a more fundamental question: whether additional relational structures should be introduced for a given dataset and task. Through empirical studies across diverse datasets, tasks, and graph constructors, we reveal that graphification is not consistently beneficial: introducing relational structures can provide substantial improvements in some cases, while offering limited or even negative gains. This observation motivates a new perspective that graph construction should be treated as a selective decision based on its expected utility rather than a default preprocessing step. To address this issue, we propose MAG-SCOUT, a pre-construction graph assessment framework that estimates whet
    
[^167]: 面向前缀缓存语言模型服务的精确内存-时间优化

    Exact Memory-Time Optimization for Prefix-Cached Language Model Serving

    [https://arxiv.org/abs/2610.02766](https://arxiv.org/abs/2610.02766)

    该论文提出前缀证书保留（PCR）方法，将前缀缓存超时策略的优化精确归约为一次最小割计算，从而在内存占用与重计算时间之间实现全局最优权衡，并通过重放39,632个真实Mooncake请求验证了其有效性。

    

    保留语言模型的前缀状态是在重计算与存储时间之间进行权衡。独立优化每个缓存块可能会高估收益：一个常驻缓存块只有在其所需的前驱前缀同样可用时才能被使用。我们提出了前缀证书保留，这是一种针对静态、分组、访问即重置超时策略的精确有限轨迹形式化方法。可用前缀的奖励转化为图中的节点，其前提条件是超时阈值和前驱命中证书。由此得到的最大权闭包问题可归约为一次最小割计算，且图的大小与块查找次数和超时选择数量呈线性关系。一个断点定理将该构造扩展到所有非负超时，且不产生离散化误差。我们还推导出一个关于网格大小呈线性的动态规划，用于处理有序超时，并给出证明该限制所付出代价的界。通过对小型实例的穷举检查以及按时间顺序重放39,632个公开的Mooncake请求进行了验证。

    arXiv:2610.02766v1 Announce Type: new  Abstract: Retaining language-model prefix states trades recomputation against storage time. Optimizing each cached block independently can overcount savings: a resident block is usable only when the required preceding prefix is also available. We introduce Prefix-Certificate Retention (PCR), an exact finite-trace formulation for static, grouped, reset-on-access timeouts. Usable-prefix rewards become nodes whose prerequisites are timeout thresholds and preceding hit certificates. The resulting maximum-weight closure reduces to one minimum cut, with graph size linear in the number of block lookups and timeout choices. A breakpoint theorem extends the construction to all nonnegative timeouts without discretization error. We also derive a linear-time-in-grid-size dynamic program for ordered timeouts and bounds that certify the cost of this restriction. Exhaustive small-instance checks and chronological replay of 39,632 public Mooncake requests validat
    
[^168]: 面向自动驾驶的基于视觉-语言模型的局部化保形安全监测

    Localized Conformal Safety Monitoring with Vision-Language Models for Autonomous Driving

    [https://arxiv.org/abs/2610.02765](https://arxiv.org/abs/2610.02765)

    本文提出SLLCP，一种叠加在冻结视觉-语言模型之上的局部化保形预测事后校准层，将VLM不可靠的安全预测转化为概率校准的安全预测集合，从而为自动驾驶中的碰撞风险监测提供可靠保障。

    

    监测规划出的驾驶轨迹需要准确估计与周围交通参与者发生碰撞的可能性，而这些参与者的运动本身会受到自车运动的影响。现有的经典方法往往受限于其预测模型的质量。视觉-语言模型（VLM）在推理高层动作的后果方面已展现出潜力，但其近似的预测结果并不适用于自动驾驶等安全关键应用。保形预测（Conformal Prediction, CP）作为一种数据驱动的框架，已逐渐成为量化黑盒模型预测不确定性的有效手段。我们提出了Split Label-Localized Conformal Prediction（SLLCP），这是一个叠加在冻结VLM之上的事后校准层，能够将其不可靠的预测转化为概率上经过校准的安全预测集合。我们考虑到安全估计的能力可能依赖于所观察到的驾驶场景，并引入了一种局部化流程，对（相关样本进行加权——原文摘要在此处被截断）……

    arXiv:2610.02765v1 Announce Type: cross  Abstract: Monitoring planned driving trajectories requires accurately estimating the collision likelihood with actors whose motion is itself impacted by the ego motion. Existing classical approaches are often limited by the quality of their forecasting model. Vision-language models (VLMs) have shown promise in reasoning about the consequences of high-level actions, yet their approximate predictions are unsuitable for safety-critical applications such as autonomous driving. Conformal prediction (CP) has emerged as a data-driven framework for quantifying the uncertainty of black-box model predictions. We propose Split Label-Localized Conformal Prediction (SLLCP), a post-hoc calibration layer over frozen VLMs that transforms their unreliable predictions into probabilistically calibrated safety prediction sets. We consider how the ability to estimate safety can depend on the observed driving scene and introduce a localized procedure that upweights r
    
[^169]: 面向评分预测的个人AI记忆受控审计

    A Controlled Audit of Personal AI Memory for Rating Prediction

    [https://arxiv.org/abs/2610.02764](https://arxiv.org/abs/2610.02764)

    该研究通过置换用户历史评分的受控实验发现，个人AI记忆系统（Mem0）在评分预测中并未有效利用历史条目-评分关联，甚至不如仅使用历史记录的简单岭回归模型。

    

    在结构化评分预测任务中，个人AI究竟是在利用历史条目与评分之间的关联，还是主要依赖用户的评分倾向？我们通过对每个用户的历史评分进行置换来审计这一区别，同时严格保持原有的评分分布、条目支持度和元数据不变。我们将这种控制手段与完整历史输入、原生记忆提取以及匹配的数值读取器相结合，在一个公开冻结的评估设置中进行测试，该评估涵盖Coat和MovieLens两个数据集上的400个留出用户档案和6,160个目标评分。在Coat数据集上，经测试的由Qwen写入的Mem0流水线相对于完整历史输入而言，使Qwen读取器的用户宏平均绝对误差增加0.084，使Phi读取器增加0.149；两者经家族校正后的自助法置信区间均不包含零。正确的历史评分分配在Coat上对两个读取器均有帮助，但MovieLens上相应的效应较小且经校正后不具结论性。仅使用历史的岭回归读取器在两个领域上都优于Qwen，并在MovieLens上优于Phi。

    arXiv:2610.02764v1 Announce Type: new  Abstract: In structured rating prediction, does a personal AI use historical item-rating associations, or mainly the user's rating tendencies? We audit this distinction by permuting historical ratings within each user while preserving the exact rating distribution, item support, and metadata. We combine this control with full history, native memory extraction, and matched numerical readers in a publicly frozen evaluation of 400 held-out user profiles and 6,160 target ratings across Coat and MovieLens. On Coat, the tested Qwen-written Mem0 pipeline increases user-macro mean absolute error relative to full history by 0.084 for Qwen and 0.149 for Phi; both family-adjusted bootstrap intervals exclude zero. Correct historical assignments help both readers on Coat, but the corresponding MovieLens effects are smaller and inconclusive after adjustment. A history-only ridge reader outperforms Qwen in both domains and Phi on MovieLens, while the Coat Phi co
    
[^170]: 基于Sentinel-1 SAR时间序列的近实时森林异常检测两级级联框架

    A Two-Stage Cascade for Near-Real-Time Forest Anomaly Detection from Sentinel-1 SAR Time Series

    [https://arxiv.org/abs/2610.02763](https://arxiv.org/abs/2610.02763)

    该论文提出了一种基于Sentinel-1 SAR时间序列的两级级联框架，结合鲁棒统计z-score检验与学习式确认门控，实现近实时的森林损失异常检测，同时克服了光学遥感的云层覆盖限制和SAR季节性后向散射变化的干扰。

    

    热带森林监测对全球气候稳定和生物多样性保护至关重要。为了满足对快速、可靠的森林损失检测的迫切需求——这对于及时干预非法砍伐、保障供应链透明度、土地使用治理和碳市场标准至关重要——我们提出了一种基于Sentinel-1时间序列的两级“统计-编码器”级联方法，用于近实时异常检测。我们的系统旨在克服遥感领域的两个根本性挑战：限制光学监测的云层覆盖问题，以及导致SAR系统将自然湿度变化误判为森林损失的季节性后向散射变化。该架构集成了两种不同的分析引擎以确保高保真度检测：（1）基于同季历史基线，对配准的Sentinel-1 VH后向散射数据进行自适应的鲁棒统计z-score检验；（2）基于学习的确认门控机制……

    arXiv:2610.02763v1 Announce Type: new  Abstract: Tropical forest monitoring is essential for global climate stability and biodiversity preservation. To address the urgent need for rapid, reliable detection of forest loss which is essential for timely intervention against illegal logging, supply chain transparency, land-use governance and carbon market standards, we introduce a two-stage statistics-encoder cascade for near-real-time anomaly detection using Sentinel-1 time series. Our system is designed to overcome two fundamental challenges in remote sensing: the cloud-cover limitations that restrict optical monitoring and seasonal backscatter variation that causes SAR systems to mistake natural moisture changes for forest loss. The architecture integrates two distinct analytical engines to ensure high-fidelity detection: (1) an adaptive, robust-statistics z-score test on co-registered Sentinel-1 VH backscatter, same-season historical baseline and (2) a learned confirmation gate based o
    
[^171]: 上下跳跃：面向离散有序数据的去噪扩散模型

    Jumping up and down: Denoiser diffusion models for discrete ordinal data

    [https://arxiv.org/abs/2610.02754](https://arxiv.org/abs/2610.02754)

    提出了JUD——首个以去噪器训练为核心、支持双向（向上和向下）扰动的离散有序数据扩散模型家族，凭借简洁的训练目标和灵活的双向扰动机制，在图像、音乐、基因计数等多种数据模态上取得了有竞争力的结果。

    

    扩散模型在图像和视频领域的连续空间中已得到高度发展。最近，针对类别数据的离散扩散模型取得了重大进展，尤其是在语言领域。相比之下，针对离散整数值数据的扩散模型发展较为滞后，尽管这种数据模态十分普遍，涵盖从图像、音乐到基因计数等多个领域。我们提出了跳跃上下——一种新的基于去噪器的离散有序数据扩散模型家族。这是首个以训练去噪器为核心的有序数据扩散模型家族，同时支持对数据进行双向（向上和向下）扰动。训练目标的简洁性，结合双向扰动的灵活性，使我们在不同的数据模态上获得了具有竞争力的结果。

    arXiv:2610.02754v1 Announce Type: new  Abstract: Diffusion models are highly developed in continuous spaces for image and video domains. Recently, major advances have been made for discrete diffusion models for categorical data, specifically in the language domain. In contrast, diffusion models for discrete integer-valued data are less developed, despite the prevalence of this modality, ranging from images and music to gene counts. We introduce Jumping Up and Down (JUD)---a new family of denoiser-based diffusion models for discrete ordinal data. This is the first family of diffusion models for ordinal data which centers around training denoisers, which at the same time allows for bi-directional (up and down) perturbations of the data. The simplicity of the training objective, combined with the flexibility of bi-directional perturbations, leads us to obtain competitive results across different data modalities.
    
[^172]: 即使向量检索在几何上很容易，学习查询编码器仍然可能很困难

    Learning Query Encoders Can Be Hard Even When Vector Retrieval Is Geometrically Easy

    [https://arxiv.org/abs/2610.02749](https://arxiv.org/abs/2610.02749)

    该研究发现单向量查询编码器的实际检索质量往往远低于冻结文档索引在几何上所能支持的上限，并从理论上证明了学习查询编码器在计算上可能是困难的。

    

    高效的向量检索需要同时满足两个条件：一是语料库的几何结构能够支持通过向量相似度检索到正确的文档，二是查询编码器能够将查询嵌入到嵌入空间中距离其目标文档较近的位置。近期的研究通过实现n个文档所有top-k答案集所需的最小嵌入维度这一视角来考察几何容量。我们研究了另一种几何容量的概念——冻结的文档索引所能达到的最大召回率——并探讨学习型查询编码器能否达到这一上限。在多个真实世界的检索基准上，我们发现单向量查询编码器的检索质量往往远低于文档索引所能支持的水平。受这一观察启发，我们给出了学习查询编码器在计算上可能非常困难的理论证据。特别地，我们构造了一个检索任务，该任务(1)存在一个能够实现完美召回率的查询编码器……

    arXiv:2610.02749v1 Announce Type: cross  Abstract: Efficient vector retrieval requires both a corpus geometry that supports retrieving the right documents through vector similarity, and a query encoder that can embed queries near their desired documents in the embedding space. Recent work has studied geometric capacity through the lens of the minimum embedding dimension needed to realize all top-$k$ answer sets of $n$ documents. We study a different notion of geometric capacity--the maximum recall achievable for a frozen document index--and explore whether learned query encoders can reach this ceiling. On several real-world retrieval benchmarks, we show that retrieval quality of single-vector query encoders often lies far below what the document indices can support.   Motivated by this observation, we give theoretical evidence that learning query encoders can be computationally hard. In particular, we construct a retrieval task that (1) admits a query encoder with perfect recall which 
    
[^173]: 前瞻性后见之明：基于预测-现实差距的自校准强化学习

    Prospective Hindsight: Self-Calibrating Reinforcement Learning via Prediction-Reality Gaps

    [https://arxiv.org/abs/2610.02740](https://arxiv.org/abs/2610.02740)

    提出前瞻性后见之明（PH）这一自校准强化学习训练原则，通过衡量智能体动作前预测与反馈后评估之间的“惊讶度”差距来加权梯度，使学习自动聚焦于智能体自我模型中最不准确的盲点样本。

    

    面向长时程智能体的强化学习依赖于纯粹的事后性训练信号：只有在观察到环境后果之后才进行信用分配，导致智能体在动作时刻的信念对梯度不可见。我们提出前瞻性后见之明，这是一种自校准训练原则，它通过一个源自智能体前瞻预测（反馈之前）与事后评估（反馈之后）之间差距的信号来增强任何事后性基础方法。这种逐次 rollout 的“惊讶度”能够识别出智能体自我模型最不准确的样本，并通过带停止梯度的惊讶度加权优势来放大这些样本的梯度贡献。由于前瞻预测器与策略共享参数，二者共同演化，逐步将学习焦点转移到智能体剩余的盲点上。我们将这一原则与特权信息差距联系起来，并证明最小化惊讶残差……（原文摘要至此中断）

    arXiv:2610.02740v1 Announce Type: cross  Abstract: Reinforcement learning for long-horizon agents relies on purely retrospective training signals: credit is assigned only after observing environmental consequences, leaving the agent's belief at action time invisible to the gradient. We introduce Prospective Hindsight (PH), a self-calibrating training principle that augments any retrospective base method with a signal derived from the gap between the agent's prospective prediction (before feedback) and the retrospective evaluation (after feedback). This per-rollout surprise identifies samples where the agent's self-model is most inaccurate and amplifies their gradient contribution through a stop-gradient surprise-weighted advantage. Since the prospective predictor shares parameters with the policy, the two co-evolve, progressively shifting focus to the agent's remaining blind spots. We connect this principle to a privileged-information gap and show that minimizing the surprise residual 
    
[^174]: 用于差分隐私 Muon 的内动量方法

    Inner Momentum for Differentially Private Muon

    [https://arxiv.org/abs/2610.02738](https://arxiv.org/abs/2610.02738)

    该论文提出“内动量”方法，即在裁剪前将每个样本的 Muon 梯度在当前模型与近期模型历史上进行平均，从而抑制差分隐私训练中逐样本梯度裁剪对奇异向量几何结构的畸变，并给出畸变的 Frobenius 范数上界及有限次 Newton-Schulz 迭代保持极因子的理论保证。

    

    差分隐私训练在加入噪声之前会对每个样本的梯度进行裁剪。这种裁剪对每个样本而言是径向的，然而不等的裁剪因子会扭曲这些梯度平均后的相对奇异向量几何结构。Muon 对这一效应尤为敏感，因为其更新是一个近似极因子 UV^T，仅依赖于那些会被裁剪偏移的奇异向量。为了抑制这种退化，我们提出在裁剪之前，将每个采样样本的 Muon 梯度在当前模型与近期模型的一段短历史上进行平均。这样一来，裁剪后的批次矩阵可以分解为一个共同缩放项以及采样梯度与裁剪值之间的协方差残差 R，且满足 ||R||_F <= sigma_lambda sigma_G，从而直接界定裁剪引起的畸变。我们进一步证明，在这些谱条件下，有限次 Newton-Schulz 迭代能够保持其输入的极因子，证实了我们的修正在正交化过程中得以保留。

    arXiv:2610.02738v1 Announce Type: new  Abstract: Differentially private training clips each per-example gradient before adding noise. This clipping is radial for each example, yet unequal clipping factors can distort the relative singular-vector geometry of their average. Muon is particularly exposed to this effect, since its update is an approximate polar factor UV^T that depends only on the singular vectors that clipping can shift. To curb this degradation, we propose averaging each sampled example's Muon gradient over the current model and a short history of recent models before clipping. The clipped batch matrix then separates into a common rescaling and a covariance residual R between sampled gradients and clipping values, with ||R||_F <= sigma_lambda sigma_G, bounding the clipping-induced distortion directly. We further show that a finite Newton-Schulz iteration preserves the polar factor of its input under these spectral conditions, confirming that our correction survives orthog
    
[^175]: 通过线性规划归一化最小化贝尔曼误差

    Bellman Error Minimization Via Linear Programming Normalization

    [https://arxiv.org/abs/2610.02730](https://arxiv.org/abs/2610.02730)

    本文提出一种结合深度神经网络与线性规划归一化的函数逼近方法，以降低高维动态规划和强化学习问题中的贝尔曼误差。

    

    本文提出了一种新的函数逼近方法，用于降低高维动态规划和强化学习问题中的贝尔曼误差。论文以一个经典的动态规划问题（收益管理中的网络容量控制）作为动机示例，说明了可以将深度神经网络与线性规划近似算法相结合，从而推导出动态规划问题的近似解。仿真结果表明，所提出的近似算法与基准方法相比具有有竞争力的性能。

    arXiv:2610.02730v1 Announce Type: new  Abstract: This paper proposes a new functional approximation approach to reduce Bellman error in high-dimensional dynamic programming and Reinforcement Learning problems. Using a classic dynamic programming problem (network capacity control in revenue management) as the motivational example, the paper illustrates that deep neural networks and linear programming approximation algorithms can be combined to derive approximate solutions to dynamic programming problems. Simulation results show the proposed approximation algorithms achieves competitive performance when compared with benchmark.
    
[^176]: 基于多模态超图的流匹配实现结构-功能脑连接生成

    Structural-Functional Brain Connectivity Generation via Multimodal Hypergraph-based Flow Matching

    [https://arxiv.org/abs/2610.02722](https://arxiv.org/abs/2610.02722)

    该论文提出多模态超环流匹配框架MHG-FM，通过超图神经网络编码器学习高阶表示并结合双交叉注意力进行双向跨模态融合，实现了结构连接与功能连接的联合生成和跨模态转换，克服了传统成对图方法无法捕捉高阶结构-功能关系的局限。

    

    结构连接（SC）和功能连接（FC）为大脑区域间的相互作用提供互补信息，被广泛应用于神经精神疾病的神经影像学研究中。生成建模可以缓解大规模配对SC-FC数据的稀缺问题，但现有方法通常使用仅能捕捉成对二元交互的成对图，并且往往独立生成SC和FC，限制了对高阶结构-功能关系的保持能力。我们提出了一种多模态超环流匹配框架，用于SC-FC连接的联合生成和跨模态转换。MHG-FM构建模态特定的超图，利用超图神经网络（HGNN）编码器学习高阶表示，并通过双交叉注意力（DCA）机制实现双向跨模态融合。变分自编码器将融合后的表示映射到紧凑的潜在空间，在该空间中进行条件生成（摘要在此处被截断）。

    arXiv:2610.02722v1 Announce Type: new  Abstract: Structural connectivity (SC) and functional connectivity (FC) provide complementary information on interactions between brain regions and are widely used in neuroimaging studies of neuropsychiatric disorders. Generative modelling can alleviate the scarcity of large-scale paired SC-FC data, but existing approaches typically use pairwise graphs that capture only dyadic interactions and often generate SC and FC independently, limiting preservation of higher-order structure-function relationships. We propose a Multimodal Hypergraph Flow Matching (MHG-FM) framework for joint SC-FC connectivity generation and cross-modal translation. MHG-FM constructs modality-specific hypergraphs, learns higher-order representations with Hypergraph Neural Network (HGNN) encoders, and performs bidirectional cross-modal fusion using Dual Cross-Attention (DCA). A variational autoencoder maps the fused representations to a compact latent space, where conditional 
    
[^177]: 基于核典型相关分析重新审视视觉语言模型的视觉表征增强

    Revisiting Visual Representation Enhancement of VLMs via Kernel Canonical Correlation Analysis

    [https://arxiv.org/abs/2610.02718](https://arxiv.org/abs/2610.02718)

    本文提出利用核典型相关分析（KCCA）在特征子空间上刻画视觉语言模型与DINOv2之间的表征对齐，从而增强CLIP等模型的细粒度视觉感知能力。

    

    诸如CLIP这样的视觉语言模型展现出强大的语义泛化能力，但在细粒度视觉感知方面仍然存在局限。最近一项名为KUEA的工作提出了一种自然的解决方案：在以视觉为中心的DINOv2的监督下微调图像编码器，以逐元素对齐二者的核矩阵，同时通过正则化使嵌入保持接近预训练的视觉编码器，从而保留CLIP中的图文语义。然而，我们表明，削弱指向DINOv2的对齐损失的作用并不一定会降低其细粒度视觉性能，这说明核矩阵差异可能不足以进一步实现视觉表征增强，这促使我们重新审视对齐的构造方式。在本工作中，我们提出了一个新颖的视角，通过核典型相关分析（KCCA）在特征子空间上刻画表征对齐，该方法最大化投影相关性

    arXiv:2610.02718v1 Announce Type: cross  Abstract: Vision-language models such as CLIP exhibit strong semantic generalization, but remain limited in fine-grained visual perception. A recent work named KUEA presents a natural remedy by finetuning the image encoder under the supervision of the vision-centric DINOv2 to align their kernel matrices element-wisely, while regularizing the embeddings to remain close to the pretrained visual encoder for preserving image-text semantics in CLIP. However, we show that diminishing the role of the alignment loss to DINOv2 does not necessarily degrade its fine-grained visual performance, suggesting that the kernel-matrix discrepancy may be insufficient for further visual representation enhancement, motivating us to revisit the alignment formulation. In this work, we present a novel perspective to characterize representation alignment on feature subspaces through Kernel Canonical Correlation Analysis (KCCA), which maximizes the projection correlations
    
[^178]: 扰动目标上梯度下降的差分隐私

    Differential Privacy of Gradient Descent on Perturbed Objectives

    [https://arxiv.org/abs/2610.02716](https://arxiv.org/abs/2610.02716)

    该论文证明了在强凸光滑目标上，对扰动目标运行梯度下降的有限次迭代是高斯噪声的 $C^1$ 微分同胚（并给出雅可比最小奇异值的定量下界），从而可直接用换元法分析有限迭代的差分隐私，且对广义线性模型而言，迭代条件成立时隐私界不显式依赖环境维度。

    

    目标扰动方法在正则化经验风险上加入一个随机线性项，并精确释放扰动后的极小化点。我们研究有限计算情形下的隐私性，即释放确定性梯度下降在 $w\mapsto F(w;S)+\langle z,w\rangle$ 上的第 $N$ 次迭代，其中噪声 $z\sim\mathcal N(0,\sigma^2I_d)$ 是在优化开始前一次性抽取的。对于具有 Lipschitz Hessian 的强凸光滑目标，我们证明了一个显式条件，在该条件下映射 $z\mapsto w_N$ 在隐私论证所用的有界区域上是 $C^1$ 微分同胚，并给出了其雅可比矩阵最小奇异值的定量下界。这使得可以对有限次迭代直接进行换元分析。对于广义线性模型，一旦迭代条件成立，所得到的隐私分布界不再包含显式的环境维度因子，且其有限次迭代的修正项以几何速度衰减。通过令自由截断参数……（原文摘要在此处截断）

    arXiv:2610.02716v1 Announce Type: new  Abstract: Objective perturbation adds a random linear term to a regularized empirical risk and releases the exact perturbed minimizer. We study the finite computation obtained by releasing the $N$-th iterate of deterministic gradient descent on $w\mapsto F(w;S)+\langle z,w\rangle$, where $z\sim\mathcal N(0,\sigma^2I_d)$ is drawn once before optimization. For strongly convex and smooth objectives with Lipschitz Hessian, we prove an explicit condition under which the map $z\mapsto w_N$ is a $C^1$-diffeomorphism on the bounded domains used in the privacy argument, with a quantitative lower bound on the smallest singular value of its Jacobian. This permits a direct change-of-variables analysis of the finite iterate. For generalized linear models, the resulting privacy-profile bound has no explicit ambient-dimension factor once the iteration condition holds, and its finite-iteration correction decreases geometrically. By letting the free truncation par
    
[^179]: WakeKV：面向会“改变主意”的注意力头的响应式、可逆KV缓存驻留策略

    WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds

    [https://arxiv.org/abs/2610.02713](https://arxiv.org/abs/2610.02713)

    WakeKV发现大多数注意力头在生成过程中会动态改变读取行为，并提出一种响应式、可逆的KV缓存驻留策略，将冷却的注意力头迁移到可恢复的CPU储备区而非冻结或永久驱逐，从而在相同内存预算下持续降低缓存未命中率。

    

    大多数KV缓存压缩方法只对注意力头进行一次性分类（离线或在预填充阶段），并在整个生成过程中保持该分类固定不变。我们在三个模型（1.5B-8B）和三种任务场景（大海捞针检索、长思维链以及多轮对话回忆）下，测量了四种模型-场景组合中的注意力头行为，发现大多数注意力头在生成过程中至少会改变一次其读取行为。我们提出了WakeKV，这是一种响应式驻留策略，它将逐渐“冷却”的注意力头迁移到可恢复的CPU储备区，而不是冻结或永久驱逐其状态。在相同的内存或预算条件下，WakeKV在五个模型-场景组合上，以及与三个引用基线方法（SnapKV、均匀R-KV和ReasonAlloc）在四个符合条件的组合上的对比评估中，均始终比冻结分类和破坏性驱逐方法取得更低的未命中率。基于Mistral-7B的FlexiCache/vLLM实现证实了该方法在真实硬件上的收益，提升了……

    arXiv:2610.02713v1 Announce Type: new  Abstract: Most KV-cache compression methods classify attention heads once, either offline or during prefill, and keep this classification fixed throughout generation. Across three models (1.5B-8B) and three regimes (needle retrieval, long chain-of-thought, and multi-turn recall), we measure head behavior on four model-regime combinations and find that most heads change their reading behavior at least once during generation. We introduce WakeKV, a reactive residency policy that moves cooling heads to a recoverable CPU reservoir rather than freezing or permanently evicting their state. At matched memory or budget, WakeKV consistently improves miss rate over frozen classification and destructive eviction, evaluated across five model-regime combinations and over three cited baselines (SnapKV, uniform R-KV, and ReasonAlloc) across four eligible combinations. A FlexiCache/vLLM implementation on Mistral-7B confirms the benefit on real hardware, improving
    
[^180]: 表征老年人人体活动识别中的性能差距

    Characterizing the Performance Gap in Human Activity Recognition for Older Adults

    [https://arxiv.org/abs/2610.02711](https://arxiv.org/abs/2610.02711)

    该论文通过老年人自由生活HAR数据集MyMove发现，年轻人基准上的模型改进难以迁移到老年人数据并造成持续扩大的性能差距，而基于年龄多样化的UK Biobank数据集预训练的冻结自监督特征能显著提升老年人活动识别性能并缩小这一差距。

    

    基于腕戴式加速度计的人体活动识别（HAR）正被越来越多地用于健康和行为追踪。然而，大多数可穿戴HAR模型是在以年轻人为主的数据集上开发和评估的，因此基准测试的进展能否跨年龄组泛化尚不清楚。在这项工作中，我们利用MyMove——我们精心标注的、自由生活场景下的老年人HAR数据集（平均年龄71岁）——在留一受试者交叉验证和跨数据集迁移两种设置下评估了深度学习架构和训练方案。我们发现，在年轻人基准上取得的改进并不能同等程度地迁移到从老年人收集的数据上，导致持续存在且常常不断扩大的性能差距。然而，更丰富的特征表示，特别是在年龄多样化的英国生物样本库数据集上预训练的冻结自监督特征，显著提升了老年人数据上的性能，并持续缩小了性能差距，且仅需适度的……

    arXiv:2610.02711v1 Announce Type: cross  Abstract: Human activity recognition (HAR) from wrist-worn accelerometers is increasingly used for health and behavioral tracking. Yet, most wearable HAR models are developed and evaluated on datasets dominated by younger adults, leaving it unclear whether benchmark progress generalizes across age groups. In this work, we leverage MyMove, our carefully annotated, free-living older-adult HAR dataset (mean age 71), to evaluate deep-learning architectures and training regimes under both leave-one-subject-out and cross-dataset transfer. We find that improvements on younger-adult benchmarks fail to transfer equally to data collected from older adults, resulting in a persistent and often widening performance gap. However, richer representations, particularly frozen self-supervised features pretrained on the age-diverse UK Biobank dataset, substantially improve performance on data from older adults and consistently narrow the performance gap, at modest
    
[^181]: MuonIO：面向嵌入表与语言模型输出头的原则性范数感知下降方法

    MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads

    [https://arxiv.org/abs/2610.02705](https://arxiv.org/abs/2610.02705)

    MuonIO 将 Muon 优化器的原则性更新扩展到嵌入表和语言模型输出头——对语言模型头采用 2→∞ 算子范数、对嵌入表采用 1→2 算子范数，从而以统一的范数感知更新取代 AdamW。

    

    Muon 优化器通过求解以谱范数惩罚的损失的局部线性化，推导出隐藏线性层的更新规则，其动机来自对稠密线性层的 RMS 稳定性论证。然而，标准的 Muon 实现并未将输入层（嵌入表）和输出层（语言模型头）纳入这一原则性处理，而是对它们改用 AdamW。我们提出 MuonIO，一种对这两个层均适用的单一 Muon 风格更新。对于语言模型头 $\mathbf{L} \in \mathbb{R}^{V \times d}$，基于 softmax 输出几何的 Lipschitz 连续性，我们论证了使用 $2\to\infty$ 算子范数的合理性；而对于嵌入表 $\mathbf{E} \in \mathbb{R}^{d \times V}$，则基于 Bernstein & Newhouse (2025) 所识别的独热输入几何，我们采用 $1 \to 2$ 算子范数。恒等式 $\lVert\mathbf{L}\rVert_{2\to\infty}=\lVert\mathbf{L}^\top\rVert_{1\to2}$ 进而将两个矩阵统一纳入……（摘要在此处截断）

    arXiv:2610.02705v1 Announce Type: cross  Abstract: The Muon optimizer derives its update rule for hidden linear layers by solving a local linearization of the loss penalized by the spectral norm, motivated by an RMS-stability argument for dense linear layers. Standard Muon implementations, however, exclude the input (embedding table) and output (language model head) layers from this principled treatment, for which they use AdamW instead. We present MuonIO, a single Muon-style update for both of these layers. For the language model head $\mathbf{L} \in \mathbb{R}^{V \times d}$, we motivate the use of the $2\to\infty$ operator norm, due to the Lipschitz continuity of the softmax output geometry, while for the embedding table $\mathbf{E} \in \mathbb{R}^{d \times V}$, we draw on the $1 \to 2$ operator norm, based on the one-hot input geometry identified by Bernstein & Newhouse (2025). The identity $\lVert\mathbf{L}\rVert_{2\to\infty}=\lVert\mathbf{L}^\top\rVert_{1\to2}$ then puts both matr
    
[^182]: 混合专家粒子Transformer中的条件容量与路由

    Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers

    [https://arxiv.org/abs/2610.02701](https://arxiv.org/abs/2610.02701)

    该研究发现在粒子物理Transformer中，避免token丢弃的top-1混合专家模型能在几乎不增加计算量的前提下超越稠密基线，但增加存储专家数量的收益有限，且专家路由结构与分类性能并非单调相关。

    

    混合专家模型能够在不按比例增加活跃计算量的情况下提升参数容量，但这一权衡在粒子物理Transformer中的表现尚不清楚。我们在188类的JetClass-II数据集上研究了稠密与MoE粒子Transformer，并改变专家数量、路由容量、top-K以及辅助损失。我们发现，在避免token丢弃的情况下，top-1 MoE模型能在几乎不变的标称前向计算量下超越稠密基线，而进一步增加存储的专家数量仅带来很小的额外准确率收益。为每个token激活多个专家可以在更高的计算成本下带来额外的预测性能提升。路由分析显示，在某些配置下专家分配与粒子身份及运动学特征的关联变得更强，但这种结构并不随分类性能的提升而单调增强。这些结果突显了需要区分……（原文摘要在此处截断）

    arXiv:2610.02701v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models can increase parameter capacity without proportionally increasing active computation, but it is unclear how this trade-off behaves in particle-physics transformers. We study dense and MoE Particle Transformers on 188-class JetClass-II, varying expert count, routing capacity, top-K, and auxiliary loss. We find that, when token dropping is avoided, top-1 MoE models improve over the dense baseline at nearly unchanged nominal forward compute, while further increasing the number of stored experts produces little additional accuracy gain. Activating multiple experts per token yields additional predictive improvements at higher computational cost. Routing analyses show that expert assignments become more strongly associated with particle identity and kinematics in some configurations, but this structure does not increase monotonically with classification performance. These results highlight the need to distinguis
    
[^183]: 从演化错误中学习：面向在线策略蒸馏的自适应迭代修复框架

    Learning from Evolving Errors: Adaptive Iterative Repair for On-Policy Distillation

    [https://arxiv.org/abs/2610.02700](https://arxiv.org/abs/2610.02700)

    该论文提出AIR-OPD框架，通过引导生成器针对学生模型不断演化的错误迭代合成修复引导，并让教师模型以该引导为特权上下文提供监督，从而实现“错误到修复”的在线策略蒸馏，避免了仅依赖参考解所带来的捷径风险。

    

    在线策略自蒸馏（OPSD）在从学生模型自身策略采样的轨迹上提供密集的token级反馈，这是一种比强化学习的结果级奖励更丰富的训练信号。这种反馈来自一个以完整参考解为条件的教师模型，而学生模型无法获得该参考解。参考解只指定了目标，却没有说明如何从学生当前的错误逐步走向目标，从而产生了一种“解条件捷径”风险。我们提出AIR-OPD，一个面向在线策略蒸馏的自适应迭代修复框架，能够提供从错误到修复的监督。给定一个失败的响应，引导生成器会针对当前错误合成修复引导；学生模型在该引导下进行在线策略重试；如果重试仍然不正确，生成器会针对新观察到的错误生成新的修复引导。在每一轮中，一个固定的教师模型接收该引导作为特权上下文并对学生进行监督……

    arXiv:2610.02700v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) supplies dense token-level feedback on trajectories sampled from the student's own policy, a richer training signal than the outcome-level rewards of reinforcement learning. This feedback comes from a teacher conditioned on a full reference solution unavailable to the student. The reference solution specifies the target but not how to move from the student's current error toward it, creating a solution-conditioned shortcut risk. We introduce AIR-OPD, an adaptive iterative repair framework for on-policy distillation that provides error-to-repair supervision. Given a failed response, a guidance generator synthesizes repair guidance for the current error. The student samples an on-policy retry with this guidance. If the retry remains incorrect, the generator produces new repair guidance for the newly observed error. At each round, a fixed teacher receives the guidance as privileged context and supervises
    
[^184]: 面向大语言模型推理的测试时校准学习

    Test-time Calibration Learning for Large Language Model Reasoning

    [https://arxiv.org/abs/2610.02695](https://arxiv.org/abs/2610.02695)

    提出了一种无需标签的测试时校准学习框架TTCL，能够直接在未标注的目标任务数据上联合优化大语言模型的推理准确性和置信度表达能力，摆脱了对真实标签的依赖。

    

    可靠的大语言模型（LLM）不仅要产生准确的答案，还必须表达出能够忠实反映其正确概率的置信度。这种校准对于识别不确定的预测以及在真实世界部署中支持可靠决策至关重要。近期研究将校准学习融入强化学习（RL）中，利用真实标签的正确性监督来联合优化答案正确性和口头表达的置信度。然而，它们对标注数据的依赖限制了其在实际测试时场景中的适用性，因为在这些场景中真实标签不可用，且校准可能需要适应新遇到的目标任务。为应对这一挑战，我们提出了测试时校准学习（TTCL），这是一个无标签框架，可直接在未标注的目标任务数据上联合调整推理准确性和口头表达的置信度。具体而言，TTCL 派生出自我监督……

    arXiv:2610.02695v1 Announce Type: cross  Abstract: Reliable large language models (LLMs) must not only produce accurate answers but also express confidence that faithfully reflects their probability of being correct. Such calibration is essential for identifying uncertain predictions and supporting reliable decision-making in real-world deployment. Recent studies incorporate calibration learning into reinforcement learning (RL), jointly optimizing answer correctness and verbalized confidence using ground-truth correctness supervision. However, their reliance on labeled data limits their applicability in practical test-time settings, where ground-truth labels are unavailable and calibration may need to adapt to newly encountered target tasks. To address this challenge, we propose Test-Time Calibration Learning (TTCL), a label-free framework that jointly adapts reasoning accuracy and verbalized confidence directly on unlabeled target-task data. Specifically, TTCL derives self-supervision
    
[^185]: 解耦记忆与上下文：面向令牌高效测试时持续学习的结构化记忆

    Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning

    [https://arxiv.org/abs/2610.02687](https://arxiv.org/abs/2610.02687)

    该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。

    

    大型语言模型越来越多地被部署在企业、科学和医疗应用中，在这些场景下，智能体必须整合领域特定知识并从经验中不断适应。上下文工程通过在推理时提供指令、策略和证据来改善模型行为，为权重更新提供了一种实用的替代方案。然而，在线调整上下文通常需要代价高昂的试错过程，而且查询往往被独立处理，导致有用的经验无法延续下去。记忆系统通过在多次交互之间保留信息来解决这一局限，但那些不断向共享上下文追加信息的方法会面临令牌成本不断上升、上下文窗口受限以及随上下文扩展而出现的性能退化。我们提出了上下文优化的统一形式化框架，并表明智能体记忆系统的更新可以被解释为一种优化……（摘要内容在此处截断）

    arXiv:2610.02687v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in enterprise, scientific, and medical applications, where agents must incorporate domain-specific knowledge and adapt from experience. Context engineering offers a practical alternative to weight updates by improving model behavior through instructions, strategies, and evidence supplied at inference time. However, adapting context online typically requires a costly trial-and-error process, while queries are often processed independently, preventing useful experience from carrying forward. Memory systems address this limitation by retaining information across interactions, but approaches that continually append information to a shared context face increasing token costs, context-window limits, and performance degradation as the context expands. We introduce a unified formulation of context optimization and show that an agent memory system update can be interpreted as an optimization 
    
[^186]: RAOA：基于可编程无线电传播的交替算子神经计算

    RAOA: Alternating-Operator Neural Computation with Programmable Radio Propagation

    [https://arxiv.org/abs/2610.02683](https://arxiv.org/abs/2610.02683)

    该论文提出RAOA循环计算架构，将可编程无线电传播用作计算深度而非仅仅作为通信信道，通过在持久潜在状态上交替执行能量导出的问题更新与混合更新，使重复执行能够在不增加学习控制参数的情况下提升离散优化的解质量，并证明所需算子可由无源仅相位自由空间传播来近似。

    

    可编程无线电传播能否充当计算深度，而不仅仅是作为通信信道或一次性模拟变换？我们提出了无线电交替算子拟设（RAOA），这是一种循环计算架构，它在持久的潜在状态上交替执行由能量导出的问题更新与混合更新。每次混合之后重新计算问题场，使得即使在不同深度重复使用相同的学习控制参数，重复执行也具有组合性。我们通过精确离散优化、受约束的可编程传播仿真以及预训练模型适配来评估这一想法。在离散目标上，重复执行可以在不增加学习控制参数数量的情况下提升解的质量，并且同一形式化方法可以直接处理高阶交互。一个无源的仅相位自由空间模型进一步表明，所需的算子可以通过可编程传播来近似，同时保留其使用价值。

    arXiv:2610.02683v1 Announce Type: new  Abstract: Can programmable radio propagation serve as computational depth rather than only as a communication channel or one-shot analog transform? We introduce the Radio Alternating Operator Ansatz (RAOA), a recurrent computing architecture that alternates an energy-derived problem update with a mixing update over a persistent latent state. Recomputing the problem field after each mix makes repeated passes compositional even when the same learned controls are reused across depth. We evaluate this idea through exact discrete optimization, constrained programmable-propagation simulation, and pretrained-model adaptation. On discrete objectives, repeated execution can improve solution quality without increasing the learned-control count, and the same formulation handles higher-order interactions directly. A passive phase-only free-space model further shows that the required operators can be approximated by programmable propagation while retaining use
    
[^187]: LEAP：为LLM智能体学习高效的动作提议

    LEAP: Learning Efficient Action Proposals For LLM Agents

    [https://arxiv.org/abs/2610.02670](https://arxiv.org/abs/2610.02670)

    该论文提出LEAP方法，通过学习一个高效的动作提议模型（而非使用现成的通用模型）为LLM智能体起草动作，并建立延迟分析框架揭示决定动作投机端到端加速的关键因素，从而显著提升智能体执行任务的速度。

    

    LLM智能体在执行任务时速度较慢。智能体一步接一步地完成任务：在每一步中先进行推理，然后选择一个动作去执行，下一步必须等上一步完成后才能开始。投机解码通过起草并验证推理token来加速推理阶段的执行。近期的工作也开始在动作阶段应用类似的思想：使用现成模型（通常较大）为目标模型起草动作提议，再由目标模型进行验证。大型起草模型与目标的匹配率更高，但生成提议耗时更长；而小型现成模型虽然速度快，却很少做出与目标一致的决策。我们提出了一个更普遍的问题：是什么决定了动作投机端到端的加速？为回答这一问题，我们为投机轮次建立了一个延迟分析框架，该框架比较每一轮投机所获得的收益与其付出的成本。收益取决于（摘要在此处被截断）

    arXiv:2610.02670v1 Announce Type: cross  Abstract: LLM agents are known to be slow in rollouts. An agent completes a task one step at a time. At each step, it reasons and then chooses an action to execute. The next step and action cannot start until the previous one has finished. Speculative decoding accelerates the rollouts at the reason phase by drafting and verifying the inference tokens. Recent works have also started to apply similar ideas at the action phase. These works use off-the-shelf models, usually large, to draft action proposals for target model to verify. Large drafters match the target more often but take longer to propose, while small off-the-shelf models are fast but rarely make the same decision as the target. We ask a more general question: what determines the end-to-end speedup of action speculation? To answer it, we develop a latency framework for the speculative round. The framework compares what a round gains with what it costs. The gain depends on how well the 
    
[^188]: 大语言连续扩散模型

    Large Language Continuous Diffusion Models

    [https://arxiv.org/abs/2610.02665](https://arxiv.org/abs/2610.02665)

    提出了首个大规模（3B/8B）连续扩散语言模型 Sigma，通过可操控的低维潜在轨迹、自回归模型热启动以及无分类器引导等推理技术，在数学推理和编码任务上取得了与离散扩散模型相当的性能。

    

    尽管离散扩散语言模型在快速并行解码方面取得了成功，但其非光滑、高维的空间阻碍了用于推理和推断加速的轨迹操控。为克服这一问题，我们提出了 Sigma，这是首个基于可操控、低维 ODE/SDE 潜在轨迹构建的大规模（3B/8B）连续扩散语言模型。Sigma 通过似然优化以分块方式进行训练，在对高斯扰动的词元嵌入进行联合去噪的同时学习最优的嵌入几何结构。为加速训练，Sigma 利用自回归（AR）模型的预训练权重进行热启动。在推理阶段，我们发现无分类器引导和分数温度对于实现高保真的推理与编码至关重要。在与最先进的离散模型（掩码扩散语言模型和自回归基线）进行的全面数学推理与编码评估中，Sigma 在标准基准上取得了与离散模型相当的性能……

    arXiv:2610.02665v1 Announce Type: cross  Abstract: Despite the success of discrete diffusion language models (dLMs) for fast parallel decoding, their non-smooth, high-dimensional space hinders trajectory steering for reasoning and inference acceleration. To overcome this, we present Sigma, the first large-scale (3B/8B) continuous dLM built on steerable, low-dimensional ODE/SDE latent trajectories. Trained blockwise via likelihood optimization, Sigma jointly denoises Gaussian-corrupted token embeddings while learning an optimal embedding geometry. To accelerate training, Sigma leverages pre-trained weights from autoregressive (AR) models for warm-starting. During inference, we identify classifier-free guidance and score temperature as essential for high-fidelity reasoning and coding. Across comprehensive math reasoning and coding evaluations against state-of-the-art discrete counterparts (masked dLMs and AR baselines), Sigma achieves competitive performance with discrete models on stand
    
[^189]: 面向内在低维数据的得分匹配扩散模型的泛化性质

    Generalization Properties of Score-matching Diffusion Models for Intrinsically Low-dimensional Data

    [https://arxiv.org/abs/2610.02663](https://arxiv.org/abs/2610.02663)

    该论文为流匹配模型在具有内在低维结构的数据上提供了统计泛化理论保证，推导出依赖于数据内在维度的 Wasserstein-p 有限样本误差界，克服了以往分析中限制性假设和忽略低维结构的不足。

    

    尽管流匹配模型在实证应用中取得了显著成功，但其统计泛化保证的理论研究仍然不完善。现有分析通常对估计的速度场施加限制性假设，且得到的收敛速率无法反映真实数据（如自然图像和分子几何结构）中普遍存在的内在低维结构。在本工作中，我们研究了流匹配模型从有限样本中学习未知分布 P_data 的统计泛化性能。我们对学习到的生成分布，在 Wasserstein-p 距离度量下（对所有 p≥1），推导出了有限样本误差界。具体而言，给定来自 P_data 的 n 个独立同分布样本，我们证明：对于每一个 d>d_p*(P_data)，只要恰当选择网络架构和超参数，学习到的分布 P̂^FM 就满足相应的误差界。

    arXiv:2610.02663v1 Announce Type: cross  Abstract: Despite the remarkable empirical success of flow-matching models, their statistical generalization guarantees remain underdeveloped. Existing analyses often impose restrictive assumptions on the estimated velocity field and yield convergence rates that fail to reflect the intrinsic low-dimensional structure common in real data, such as natural images and molecular geometries. In this work, we study the statistical generalization of flow-matching models for learning an unknown distribution $P_{\mathrm{data}}$ from finitely many samples. We derive finite-sample error bounds on the learned generative distribution, measured in the Wasserstein-$p$ distance, for all $p\geq 1$. Specifically, given $n$ i.i.d. samples from $P_{\mathrm{data}}$, we show that, for every $d>d_p^\ast(P_{\mathrm{data}})$ and appropriately chosen network architectures and hyperparameters, the learned distribution $\widehat{P}^{\mathrm{FM}}$ satisfies $ \mathbb{W}_p(\w
    
[^190]: 注意细化差距：当安全的高层机器人规划产生不安全的执行时

    Mind the Refinement Gap: When Safe High-Level Robot Plans Produce Unsafe Executions

    [https://arxiv.org/abs/2610.02662](https://arxiv.org/abs/2610.02662)

    该论文揭示了机器人安全系统中的“细化差距”问题——通过安全监控的高层规划在执行时可能因导航和隐式动作效果而变得不安全，并通过28个受控案例和14个端到端案例验证了基于图的轨迹细化方法的必要性。

    

    语言使能的机器人系统日益将语义图规划与时序逻辑安全监控器相结合。我们研究了这些系统中的一个轨迹完备性假设：监控器所检查的高层动作序列是否代表了执行过程中引发的导航及隐式动作效果。我们在 RoboGuard 中对这一假设进行了审计，方法是将其对表面规划的判定与在同一线性时序逻辑（LTL）规范下对图细化轨迹的判定进行比较。我们的评估包括涵盖五个动作抽象族的28个受控案例，以及14个端到端案例——其中 SPINE [1] 从自然语言指令生成规划，而 RoboGuard 生成基于场景的安全规范。在受控评估中，所有12个针对性的抽象案例均呈现出预期的表面规划与细化轨迹之间的差异，而所有16个对照案例的表现均符合预期，这为基于图的轨迹细化提供了依据……

    arXiv:2610.02662v1 Announce Type: new  Abstract: Language-enabled robot systems increasingly combine semantic-graph planning with temporal-logic safety monitors. We investigate a trace-completeness assumption in these systems: whether the high-level action sequence checked by a monitor represents the navigation and implicit action effects induced during execution. We audit this assumption in RoboGuard by comparing its verdict on a surface plan with its verdict on a graph-refined trace under the same Linear Temporal Logic (LTL) specification. Our evaluation comprises 28 controlled cases spanning five action-abstraction families and 14 end-to-end cases in which SPINE [1] generates plans from natural-language instructions while RoboGuard generates scene-grounded safety specifications. In the controlled evaluation, all 12 targeted abstraction cases exhibit the predicted surface-versus-refined discrepancy while all 16 controls behave as expected, motivating graph-based trace refinement as a
    
[^191]: AIGS：面向非平稳数据流在线表示学习的自适应增量门控系统

    AIGS: Adaptive Incremental Gating System for Online Representation Learning in Non-Stationary Data Streams

    [https://arxiv.org/abs/2610.02661](https://arxiv.org/abs/2610.02661)

    本文提出轻量级闭环自适应框架AIGS，通过内生残差反馈机制“冲击比率”驱动连续可塑性控制器，在非平稳数据流的在线表示学习中动态平衡知识保持与概念漂移响应。

    

    万物联网和边缘计算环境中的实时数据流常常通过潜在的状态变化而演化。对于在严格计算约束下的在线表示学习而言，核心问题在于解决稳定性-可塑性困境：即在保留有用历史知识的同时快速响应概念漂移。现有方法采用固定更新调度或滚动窗口，然而这些方法在数据流发生突变时会出现参数僵化，而在数据流保持稳定时又会浪费计算资源。本文提出了自适应增量门控系统（AIGS），这是一个轻量级的闭环状态感知自适应框架。AIGS 引入了冲击比率，这是一种内生残差反馈机制，将当前重构误差相对于近期变化进行归一化。该信号驱动一个连续可塑性控制器，在学习可塑性与记忆保持之间进行平滑插值。通过……

    arXiv:2610.02661v1 Announce Type: new  Abstract: Real-time data streams in Web of Things (WoT) and edge computing environments often evolve through latent regime changes. For online representation learning under strict computational constraints, the central problem is resolving the stability-plasticity dilemma: keeping useful historical knowledge while rapidly reacting to concept drift. Existing methods employ fixed update schedules or rolling windows. However, they suffer from parameter ossification during sudden shifts and waste computational resources when the stream remains stable. This paper proposes the Adaptive Incremental Gating System (AIGS), a lightweight closed-loop state-aware adaptation framework. AIGS introduces the Shock Ratio, an endogenous residual feedback mechanism that normalizes current reconstruction error against recent variation. This signal drives a Continuous Plasticity Controller that smoothly interpolates between learning plasticity and memory retention. By 
    
[^192]: 基于选择性状态空间模型的分布式学习：架构感知的收敛性分析

    Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis

    [https://arxiv.org/abs/2610.02659](https://arxiv.org/abs/2610.02659)

    该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。

    

    现代状态空间模型（SSM），如Mamba2，通过将线性时间复杂度的序列建模与递归状态空间动力学相结合，为Transformer提供了一种极具吸引力的替代方案。然而，SSM在分布式学习环境中的行为仍然鲜为人知。特别是，现有的标准联邦学习方法在很大程度上与架构无关，没有考虑到现代选择性SSM所特有的稳定性、选择性和状态空间参数化特性。为了解决这一问题，我们为单层和多层选择性SSM推导了架构感知的梯度和平滑度界限，并为FedAvg和FedProx推导了收敛界限，刻画了递归稳定性、输入相关的离散化以及状态投影范数如何影响联邦优化。随后，我们在由教师SSM生成的序列上，使用遵循所分析递归结构的学习器，对单层界限进行了数值验证。

    arXiv:2610.02659v1 Announce Type: cross  Abstract: Modern state space models (SSMs), such as Mamba2, provide a compelling alternative to transformers by combining linear-time sequence modeling with recurrent state-space dynamics. However, the behavior of SSMs in distributed learning settings remains poorly understood. In particular, the existing standard federated learning methods are largely architecture-agnostic, and do not account for the stability, selectivity, and state-space parameterization that characterize modern selective SSMs. To address this, we derive architecture-aware gradient and smoothness bounds for single- and multi-layer selective SSMs, and convergence bounds for FedAvg and FedProx, characterizing how recurrent stability, input-dependent discretization, and state projection norms affect federated optimization. We then numerically validate the single-layer bounds on sequences generated by a teacher SSM, using a learner that follows the analyzed recurrence. We use thi
    
[^193]: 上下文塔转换保留生成能力，冻结策略保留知识：MoE大语言模型的低预算自回归到扩散模型转换

    Context-Tower Conversion Preserves Generation While Freezing Retains Knowledge: Low-Budget AR-to-Diffusion Conversion of MoE LLMs

    [https://arxiv.org/abs/2610.02657](https://arxiv.org/abs/2610.02657)

    在低训练预算下将MoE大语言模型转换为扩散语言模型时，冻结父模型副本并通过交叉注意力条件化的“冻结塔”方案，比直接更新部分权重的“就地”方案生成能力提升11.6倍，同时几乎完全保留原模型知识。

    

    将预训练的自回归（AR）模型转换为扩散语言模型（dLLM）可以无需从头预训练新模型即可实现并行生成。已发表的转换方法在训练数据量上相差约三个数量级，且尚未在统一协议下进行过比较。我们对同一个300亿参数混合专家（MoE）父模型的两种转换方式进行了对比，固定了语料库、监督token预算、可训练参数集合和评估框架，每种方法各自采用其原有的训练方案。就地（in-place）模型使用去噪损失和表征对齐损失更新父模型的一部分权重；而冻结塔模型则通过交叉注意力，以父模型的冻结因果副本作为条件进行生成。在10亿训练token的预算下，冻结塔模型在HumanEval pass@10上得分71.60，而就地模型仅为6.19，提升了11.6倍。在相同预算下，冻结塔模型还保留了父模型95%的GSM8K分数和99%的MMLU-Pro分数。

    arXiv:2610.02657v1 Announce Type: new  Abstract: Converting a pretrained autoregressive (AR) model to a diffusion language model (dLLM) enables parallel generation without pretraining a new model. Published conversion methods differ by roughly three orders of magnitude in training data and have not been compared under a common protocol. We compare two conversions of the same 30B Mixture-of-Experts (MoE) parent, holding the corpus, supervised-token budget, trainable parameter set and evaluation harness fixed, each under its own training recipe. The in-place model updates a subset of the parent's weights using denoising and representation-alignment losses; the frozen-tower model instead conditions through cross-attention on a frozen causal copy of the parent. With 1B training tokens, the frozen-tower model scores 71.60 on HumanEval pass@10 against 6.19 for the in-place model, an 11.6x improvement. At the same budget it also keeps 95% of the parent's GSM8K score and 99% of its MMLU-Pro sc
    
[^194]: 当归一化决定符号：量子注意力中鲁棒性消融实验的审计

    When Normalization Selects the Sign: Auditing Robustness Ablations in Quantum Attention

    [https://arxiv.org/abs/2610.02641](https://arxiv.org/abs/2610.02641)

    该论文以四量子比特量子注意力检测器为例，揭示消融实验中输入缩放模块表现出的鲁棒性收益可能来自比较规则本身而非模块，主张需控制编码器处的扰动上界，并区分描述性结论与因果性结论。

    

    移除输入缩放模块会同时改变分类器本身以及到达其编码器的扰动。因此，鲁棒性差异既可能反映比较规则，也可能反映模块本身。我们在一个应用于生成的电网轨迹的四量子比特量子注意力检测器上展示了这一问题。一个学习得到的缩放模块在固定的物理攻击预算下似乎是有益的，但当在编码器处匹配扰动的上界时，两者的优劣排序会发生逆转。任何单独一种比较都无法确立由该模块带来的鲁棒性收益。初始测试还将干净样本扰动为受攻击样本，同时保留其原始标签；仅限于已受攻击样本的测试无法确立这种收益。替换训练好的模型的输入缩放会破坏检测能力；重新训练其线性分类层可以恢复检测率，但会改变个体预测结果，因此这种比较仅是描述性的而非因果性的。

    arXiv:2610.02641v1 Announce Type: cross  Abstract: Removing an input-scaling module changes both a classifier and the perturbations reaching its encoder. A robustness difference can therefore reflect the comparison rule as well as the module. We demonstrate this problem in a four-qubit quantum-attention detector on generated power-grid trajectories. A learned scaling module appears beneficial at a fixed physical attack budget, but matching an upper bound on perturbations at the encoder reverses the ordering. Neither comparison alone establishes a robustness benefit caused by the module. The initial test also perturbs clean examples into attacked examples while retaining their original labels; tests restricted to already attacked examples do not establish a benefit. Replacing a trained model's input scales disrupts detection. Retraining its linear classification layer restores the detection rate, but changes individual predictions, leaving the comparison descriptive rather than causal. 
    
[^195]: 量子傅里叶采样止步之处：面向延迟PUF安全模型的三重门审计协议

    Where Quantum Fourier Sampling Stops Short: A Three-Gate Audit Protocol for Delay-PUF Security Models

    [https://arxiv.org/abs/2610.02636](https://arxiv.org/abs/2610.02636)

    本文通过结构、算法和量子核诊断三重门的系统审计表明，量子傅里叶采样在延迟PUF安全模型中难以建立真正的量子优势，因为低次结构不等于稀疏支撑、相位预言机构造逻辑上蕴含经典基线，而看似有利的量子核指标与经典Gram矩阵性质高度相关。

    

    量子傅里叶采样或许有助于审计基于延迟的物理不可克隆函数（PUF）的谱可学习性。我们提出的疑问是：这一承诺在面对访问匹配、强经典比较器以及预言机合成时是否依然成立。评估由三重门构成。结构门：在可达的挑战长度下，低次数并不意味着小支撑集；对于 n=14 的 4-XOR，次数 ≤ d_f(0.1) 允许了所有 2^n 个特征函数中的 91%，而中位数 90% 质量集合跨越了谱的三分之一。算法门：在逻辑上构造相位预言机即蕴含了经典成员访问权限，这使得 Kushilevitz–Mansour 成为正确的基线；在 45 个任务中，该基线穷尽了每个有限域，且没有任何 4-XOR 理想采样情形能在 2^n 次调用内达到 90% 质量。量子核诊断看起来更为有利，在 N=512 个挑战时几何差异升至 2.151，但它与经典 Gram 矩阵的 1/√(λ_min(K_C)) 的相关性高达 0.991，这表明观察到的优势可能只是经典可分性的反映。

    arXiv:2610.02636v1 Announce Type: cross  Abstract: Quantum Fourier sampling may help audit the spectral learnability of delay-based physical unclonable functions (PUFs). We ask whether that promise survives access matching, a strong classical comparator, and oracle synthesis. Three gates structure the evaluation. Structure: low degree is not small support at reachable challenge lengths; for 4-XOR at $n=14$, degree $\le d_f(0.1)$ admits $91\%$ of all $2^n$ characters and the median $90\%$-mass set spans a third of the spectrum. Algorithmics: constructing the phase oracle logically implies classical membership access, making Kushilevitz--Mansour the correct baseline; across 45 tasks it exhausts each finite domain, and no 4-XOR ideal-sampling case reaches $90\%$ mass within $2^n$ calls. A quantum-kernel diagnostic appears more favorable, with geometric difference rising to $2.151$ at $N=512$ challenges, but it correlates $0.991$ with $1/\sqrt{\lambda_{\min}(K_C)}$ for the classical Gram m
    
[^196]: 一种用于磁共振弹性成像中粘弹性组织特性重建的CNN-伴随优化混合框架

    A hybrid CNN-adjoint optimization framework for reconstruction of viscoelastic tissue properties in magnetic resonance elastography

    [https://arxiv.org/abs/2610.02634](https://arxiv.org/abs/2610.02634)

    该论文提出了一种结合卷积神经网络与伴随优化的混合框架，用于磁共振弹性成像中软组织复值剪切模量的重建，并在理论上证明了正问题的适定性、极小值的存在性及一阶最优性条件。

    

    磁共振弹性成像（MRE）是一种无创成像技术，可通过剪切波传播来量化软组织的粘弹性特性。从测量的位移场中恢复复值剪切模量会导致一个严重不适定的逆问题，尤其是在存在噪声和边界激励受限的情况下。我们研究了基于伴随的优化方法、卷积神经网络（CNN）重建方法，以及结合两者的混合框架。正问题模型基于具有复剪切模量的修正稳态Stokes方程组的标量形式。我们建立了正问题的适定性、极小值的存在性以及基于伴随形式的一阶最优性条件，并实现了带有Armijo线搜索的非线性共轭梯度方法。虽然偏微分方程约束优化能够精确地细化系数重建，但其性能依赖……

    arXiv:2610.02634v1 Announce Type: cross  Abstract: Magnetic resonance elastography (MRE) is a noninvasive imaging modality for quantifying the viscoelastic properties of soft tissues from shear wave propagation. Recovering the complex-valued shear modulus from measured displacement fields leads to a severely ill-posed inverse problem, particularly in the presence of noise and limited boundary excitations. We investigate adjoint-based optimization, convolutional neural network (CNN) reconstruction, and a hybrid framework combining both approaches. The forward model is based on a scalar form of the modified stationary Stokes system with a complex shear modulus. We establish well-posedness of the forward problem, existence of minimizers, and first-order optimality conditions for the adjoint-based formulation, and implement a nonlinear conjugate-gradient method with Armijo line search. While PDE-constrained optimization can accurately refine coefficient reconstructions, its performance dep
    
[^197]: 成本约束下语言模型响应的在线验证

    Online Verification of Language Model Responses Under Cost Constraints

    [https://arxiv.org/abs/2610.02632](https://arxiv.org/abs/2610.02632)

    提出OMVV在线多验证器算法，通过维护K个候选弱验证器池来适应查询主题和难度的动态变化，在成本约束下实现更准确且经济高效的语言模型输出验证。

    

    随着大型语言模型越来越多地被部署用于多步推理，验证其输出的正确性已成为在大规模应用中保持可靠性的关键。验证大型语言模型输出的正确性通常需要查询一个代价高昂的真值验证器，但在在线环境中每一步都调用它是不切实际的。先前的工作通过在每一步查询一个较弱的验证器，并利用其评分来决定是否需要查询代价高昂的强验证器，从而将强验证仅保留给一小部分步骤。然而，随着输入查询的主题或难度随时间变化，单一固定的弱验证器可能无法始终保持良好的表现，而提前固定使用某一个验证器则可能导致验证成本过高或准确性不足。我们提出了OMVV（Online Multi-Verifier Verification，在线多验证器验证）算法，该算法维护一个包含K个候选弱验证器的池（摘要在此处截断）。

    arXiv:2610.02632v1 Announce Type: new  Abstract: As large language models are increasingly deployed for multi-step reasoning, verifying the correctness of their outputs has become essential for maintaining reliability at scale. Verifying the correctness of large language model outputs is often done by querying a costly ground-truth oracle, which is impractical to invoke at every step in an online setting. Prior work addresses this by querying a single weak verifier on every step, and using its score to decide whether the costly strong verifier needs to be queried as well, reserving strong verification for only a small fraction of the steps. However, a single fixed weak verifier may not perform consistently well as the subject matter or difficulty of incoming queries changes over time, and committing to one in advance risks either overly costly or inaccurate verification. We introduce OMVV (Online Multi-Verifier Verification), an algorithm that maintains a pool of $K$ candidate weak ver
    
[^198]: VERSE：面向智能体框架的经过验证的自我进化优化器

    VERSE: Verified Self-Evolving Optimizer for Agent Harnesses

    [https://arxiv.org/abs/2610.02616](https://arxiv.org/abs/2610.02616)

    提出VERSE，一个经过验证的自我进化优化器，它不仅改进智能体框架，还让优化器自我进化其诊断、编辑与验证流程（如测试草稿编辑、重放故障、扰动可疑步骤），在具备基于执行的验证时取得最佳优化效果。

    

    框架进化可以改进LLM智能体的提示词、工具和工作流程，而优化器自身的工具和流程却往往保持固定。我们研究优化器是否可以通过同时改进其诊断故障、开发编辑和测试效果的方式，来更有效地改进另一个智能体。两个观察结果指导了我们的设计：在一项对照研究中，没有基于执行的验证时，优化器自我进化无法提升性能；而当验证可用时，它在该研究中取得了最佳结果。在五个执行器上，自我进化的优化器为故障分析、验证、训练审计和工作流控制构建了自己的工具。基于这些发现，我们提出了VERSE，一个面向智能体框架的经过验证的自我进化优化器。VERSE允许优化器在提交前测试草稿编辑、重放故障并扰动可疑步骤，同时跨轮次跟踪修复和回归情况。利用这种反馈……

    arXiv:2610.02616v1 Announce Type: new  Abstract: Harness evolution improves an LLM agent's prompts, tools, and workflow, while the optimizer's own tools and procedures often remain fixed. We study whether an optimizer can improve another agent more effectively by also improving how it diagnoses failures, develops edits, and tests their effects. Two observations guide our design. In a controlled study, optimizer self-evolution fails to improve performance without execution-based verification, but achieves the best result of that study when verification is available. Across five executors, self-evolving optimizers build their own tools for failure analysis, verification, training audits, and workflow control. Motivated by these findings, we introduce VERSE, a Verified Self-Evolving optimizer for agent harnesses. VERSE lets the optimizer test draft edits, replay failures, and perturb suspected steps before submission, while tracking fixes and regressions across rounds. Using this feedback
    
[^199]: 量化构造性归纳、知识与噪声过滤对归纳学习的价值

    Quantifying the Value of Constructive Induction, Knowledge, and Noise Filtering on Inductive Learning

    [https://arxiv.org/abs/2610.02615](https://arxiv.org/abs/2610.02615)

    本文提出“有效维度”这一新型学习度量方法，它可经验估计并能进行平均情况预测，可用于精确量化构造性归纳、噪声过滤和背景知识对归纳学习性能的价值。

    

    学习研究的核心目标之一是测量、建模并理解学习问题的特性如何影响平均情况下的学习性能。例如，我们希望量化构造性归纳、噪声过滤和背景知识的价值。本文描述了有效维度（effective dimension），这是一种新的学习度量方法，有助于将问题特性与学习性能联系起来。与Vapnik-Chervonenkis（VC）维度一样，有效维度通常与问题特性呈简单的线性关系。与VC维度不同的是，有效维度可以通过经验方法估计，并能做出平均情况下的预测。因此，它更广泛地适用于机器学习和人类学习研究。该度量方法在包括反向传播（Backpropagation）在内的多个学习系统上进行了演示。最后，该度量被用于精确预测使用FRINGE（一种特征构造系统）所带来的收益。发现该收益……

    arXiv:2610.02615v1 Announce Type: new  Abstract: Learning research, as one of its central goals, tries to measure, model, and understand how learning-problem properties affect average-case learning performance. For example, we would like to quantify the value of constructive induction, noise filtering, and background knowledge. This paper describes the effective dimension, a new learning measure that helps link problem properties to learning performance. Like the Vapnik-Chervonenkis (VC) dimension, the effective dimension is often in a simple linear relation with problem properties. Unlike the VC dimension, the effective dimension can be estimated empirically and makes average-case predictions. It is therefore more widely applicable to machine and human learning research. The measure is demonstrated on several learning systems including Backpropagation. Finally, the measure is used to precisely predict the benefit of using FRINGE, a feature construction system. The benefit is found to 
    
[^200]: 后训练中失去了什么？默认塌缩与跨多元视角的上下文可引导性丧失

    What Is Lost in Post-Training? Default Collapse and the Loss of In-Context Steerability Across Diverse Perspectives

    [https://arxiv.org/abs/2610.02614](https://arxiv.org/abs/2610.02614)

    论文发现后训练不仅会收窄模型的默认观点表达，还会系统性削弱模型通过上下文提示被引导至未被训练偏好视角的能力，揭示了单一价值观优化与服务多元利益相关者能力之间的根本矛盾。

    

    服务于异质人群的AI模型必须依据适合每个用户和情境的原则来行事。虽然已有研究表明后训练会收窄大语言模型所表达的观点，但先前的工作主要关注默认行为，而非模型适应上下文信息的能力。我们证明，后训练还会降低模型通过上下文被引导至其未被训练偏好视角的能力。在对照实验中，我们将模型向文化价值观分歧中的一方进行微调，并评估整个训练过程中的各个检查点。被训练的一方在日常使用中变得越来越占主导地位，而识别并忠实呈现对立观点的能力则逐渐下降。这些发现揭示了优先推行单一价值观与保留服务多元利益相关者所需技术能力之间的张力。最后，我们提出并分析了一种替代性目标函数，该函数最大化奖励……

    arXiv:2610.02614v1 Announce Type: new  Abstract: AI models serving a heterogeneous population must act on the principles appropriate to each user and context. While post-training has been shown to narrow the views large language models express, prior work has focused on default behavior rather than the ability to adapt to in-context information. We show that post-training also degrades a model's ability to be steered in-context toward perspectives it was not trained to favor. In controlled experiments, we fine-tune models toward one side of cultural-value disagreements and evaluate checkpoints throughout training. The trained side becomes increasingly dominant in ordinary use, while the ability to recognize and faithfully enact the opposing view declines. These findings point to a tension between prioritizing a single set of values and preserving the technical capacity needed to serve diverse stakeholders. Finally, we propose and analyze an alternative objective that maximizes reward s
    
[^201]: 面向少步降水集合生成的尺度递归校正流

    Scale-Recursive Rectified Flows for Few-Step Precipitation Ensembles

    [https://arxiv.org/abs/2610.02611](https://arxiv.org/abs/2610.02611)

    提出一种尺度递归校正流方法，先生成大尺度降雨模式再生成局部细节，并通过比较集合变率与预测误差自适应分配采样步数，从而在极少采样步数下生成高质量且不确定性可靠的降水集合。

    

    精细分辨率的降水估计能够支持洪水风险评估与水资源管理，但粗糙的卫星产品无法分辨每个网格单元内部的降雨分布。生成模型通过产生多个合理的高分辨率降雨场集合来解决这种不确定性问题。在这类模型中，校正流通过迭代地将随机噪声转化为降雨场来生成样本。减少采样步数可以加速生成过程，但可能导致集合成员过于相似，从而低估不确定性。我们提出了一种尺度递归的校正流方法，先生成大尺度的降雨模式，再生成局部细节，并通过在不同空间尺度上比较集合变率与预测误差来指导采样步数的分配。验证分数和降雨功率谱对这一分配过程进行约束，以避免过度放大。在美国本土的卫星到雷达降尺度任务中，我们的分析识别出大尺度降雨模式（摘要在此处截断）

    arXiv:2610.02611v1 Announce Type: new  Abstract: Fine-resolution precipitation estimates support flood risk assessment and water management, but coarse satellite products cannot resolve rainfall within each grid cell. Generative models address this ambiguity by producing ensembles of plausible high-resolution rainfall fields. Among these models, rectified flows generate samples by iteratively transforming random noise into rainfall fields. Reducing the number of sampling steps accelerates generation but can make ensemble members too similar, understating uncertainty. We propose a scale-recursive rectified flow that generates broad patterns before local details and guides sampling-step allocation by comparing ensemble variability with prediction error across spatial scales. Validation scores and rainfall power spectra constrain the allocation to avoid excessive amplification. In satellite-to-radar downscaling over the contiguous United States, our analysis identified broad rainfall patt
    
[^202]: Seer：用于学习速度曲线的最大似然回归

    Seer: Maximum Likelihood Regression for Learning-Speed Curves

    [https://arxiv.org/abs/2610.02610](https://arxiv.org/abs/2610.02610)

    Seer系统通过生成分类学习性能的经验观察数据并构建最大似然统计模型，能够预测达到目标性能所需的训练样本数量以及理论上限准确率，并在多个真实领域数据上验证了其可行应用。

    

    本研究聚焦于机器学习性能的建模。论文提出了Seer系统，该系统首先生成分类学习性能的经验观察数据，然后利用这些观察数据构建统计模型。这些模型可用于预测达到期望性能水平所需的训练样本数量，以及在训练样本数量不受限制的情况下所能达到的最高准确率。Seer从三个方面推进了该领域的最新技术水平：1）体现了分类学习最佳约束条件和最有用参数的模型；2）能够高效找到最大似然模型的算法；3）在来自三个领域的真实数据上演示了此类建模的可行应用。论文第一部分概述了良好的分类学习性能最大似然模型所需满足的要求，接着探讨了此类模型的合理设计选择……

    arXiv:2610.02610v1 Announce Type: new  Abstract: The research presented here focuses on modeling machine-learning performance. The thesis introduces Seer, a system that generates empirical observations of classification-learning performance and then uses those observations to create statistical models. The models can be used to predict the number of training examples needed to achieve a desired level and the maximum accuracy possible given an unlimited number of training examples. Seer advances the state of the art with 1) models that embody the best constraints for classification learning and most useful parameters, 2) algorithms that efficiently find maximum-likelihood models, and 3) a demonstration on real-world data from three domains of a practicable application of such modeling.   The first part of the thesis gives an overview of the requirements for a good maximum-likelihood model of classification-learning performance. Next, reasonable design choices for such models are explore
    
[^203]: 利用激活稀疏性与权重近似加速卸载权重下的大语言模型解码

    Activation Sparsity with Weight Approximation for Faster LLM Decoding on Offloaded Weights

    [https://arxiv.org/abs/2610.02598](https://arxiv.org/abs/2610.02598)

    提出SpAx方法，将权重读取的“保留或跳过”二元决策扩展为“完全保留、压缩近似或完全省略”三种选项，从而在利用激活稀疏性加速卸载权重下的LLM解码时，显著改善模型质量与推理速度之间的权衡。

    

    将大语言模型部署在显存不足以容纳其权重的消费级GPU上，会导致极其缓慢的推理，因为解码过程需要反复将卸载的权重从系统内存或闪存传输到GPU中，而这一传输带宽远低于本地GPU显存访问带宽。激活稀疏性通过跳过与零或近零激活相关的权重来减少这类传输。然而，随着被省略的激活贡献增多，模型质量最终会迅速下降，这表明与小幅值激活相关的权重会共同对模型质量产生急剧影响。在本工作中，我们改进了利用激活稀疏性时模型质量与解码性能之间的权衡。我们的核心思想是将“读取或不读取某一权重”的二元选择替换为三种选项：完全保留该权重、使用压缩的权重表示对其进行近似，或将其完全省略。SpAx会跳过与……相关的权重（摘要原文在此处截断）

    arXiv:2610.02598v1 Announce Type: new  Abstract: Deploying LLMs on consumer-grade GPUs with insufficient memory to hold their weights can result in prohibitively slow inference, because decoding repeatedly transfers offloaded weights from system RAM or flash storage into GPU at much lower bandwidth than local GPU-memory access. Activation sparsity reduces these transfers by skipping weights associated with zero or near-zero activations. However, as more activation contributions are omitted, model quality eventually degrades rapidly, indicating that weights associated with small-magnitude activations collectively influence model quality sharply. In this work, we improve the trade-off between model quality and decoding performance when exploiting activation sparsity. Our key idea is to replace the binary choice of whether or not to read a weight with three options: fully retain it, approximate it using a compressed weight representation, or omit it entirely. SpAx skips weights associated
    
[^204]: 因果性如何弥合语义鸿沟

    How Causality Bridges the Semantic Gap

    [https://arxiv.org/abs/2610.02594](https://arxiv.org/abs/2610.02594)

    该论文提出以因果结构替代人类知识来为未命名变量赋予语义，将其形式化为“结构约束的语义对齐”，并构建 CausalBridge 框架，从测量数据（含隐变量）中发现因果图并在其依赖关系约束下求解变量嵌入，从而从变量对其他变量的作用方式中解读其含义。

    

    数值测量捕捉了系统的行为方式，但往往未指明其变量的含义：有些变量被测量却从未被标注，另一些变量则从未被测量。现有方法通过参考人类的一般知识为这些变量赋予语义，但在知识存在之处会继承其偏见，在知识缺失之处则无能为力。我们转而利用因果结构来弥合测量与其含义之间的鸿沟，从变量作用于其他变量的方式中解读其语义。我们将这一过程形式化为“结构约束的语义对齐”：以少量已知名称的嵌入作为锚点，在因果图所蕴含的依赖关系约束下求解每个未命名变量的嵌入。基于此，我们构建了 CausalBridge 框架，该框架从测量数据（包括隐变量）中发现因果图，并在这些依赖关系约束下求解嵌入。（原文摘要在此处截断）

    arXiv:2610.02594v1 Announce Type: cross  Abstract: Numerical measurements capture how a system behaves, but often leave the meanings of its variables unspecified. Some variables are measured but never labeled, and others are never measured at all. Existing methods assign semantics to such variables by consulting general human knowledge, but this inherits its biases where that knowledge exists and offers nothing where it does not. We bridge this gap between measurements and their meanings with causal structure instead, reading a variable's semantics from how it acts on other variables. We formalize this as structure-constrained semantic alignment, in which the embedding of each unnamed variable is solved under the dependence relations implied by the causal graph, with the embeddings of a few known names as anchors. Accordingly, we build CausalBridge, a framework that discovers the causal graph from the measurements, latent variables included, solves for the embeddings under those relati
    
[^205]: 面向大语言模型持续预训练的Fisher引导次模数据选择

    Fisher-Guided Submodular Data Selection for Continual Pre-Training of Large Language Models

    [https://arxiv.org/abs/2610.02593](https://arxiv.org/abs/2610.02593)

    该论文通过Fisher信息分析揭示了灾难性遗忘的参数空间机制，并提出一种Fisher引导的次模数据选择方法，在持续预训练中保护预训练模型所依赖的高Fisher参数坐标，从而无需大量回放数据即可控制遗忘。

    

    数据选择已成为大语言模型训练中的核心瓶颈，因为网络规模的语料库噪声大且token预算有限。在持续预训练（CPT）中，这变成了一个遗忘控制问题：选择不当的目标领域语料库可能会覆盖预训练检查点中已编码的能力。现有的CPT实践要么使用与参数无关的标量（如困惑度）对候选数据进行评分，要么通过花费大量额外的通用领域回放token来缓解遗忘，这两种策略都没有直接探究在候选数据上训练会如何改变模型参数。我们证明，基于损失的选择会导致CPT后的Fisher对角线恰好在预训练模型所依赖的高Fisher坐标上向下漂移，而低Fisher坐标基本保持不变。这种不对称性揭示了灾难性遗忘在参数空间中的机制。受此观察启发，我们提出了一种Fisher感知的CPT方法（摘要在此处截断）

    arXiv:2610.02593v1 Announce Type: new  Abstract: Data selection is already a central bottleneck in large-language-model training, where web-scale corpora are noisy and token budgets are finite. In continual pre-training (CPT), it becomes a forgetting-control problem: a poorly chosen target-domain corpus can overwrite capabilities encoded in the pretrained checkpoint. Existing CPT practice either scores candidates with parameter-agnostic scalars such as perplexity, or mitigates forgetting by spending many extra general-domain replay tokens. Neither strategy directly asks how training on a candidate will move the model parameters. We show that loss-based selection causes the post-CPT Fisher diagonal to drift downward on exactly the high-Fisher coordinates the pretrained model had committed to, while leaving low-Fisher coordinates largely untouched. This asymmetry exposes a parameter-space mechanism for catastrophic forgetting. Motivated by this observation, we propose a Fisher-aware CPT 
    
[^206]: 稠密混合专家作为重新参数化的宽前馈网络：固定计算量下的粒度扫描

    Dense Mixture-of-Experts as a Reparameterized Wide FFN: A Granularity Sweep at Fixed Compute

    [https://arxiv.org/abs/2610.02584](https://arxiv.org/abs/2610.02584)

    该研究在固定计算量下提出一种稠密混合专家架构（$K$个SwiGLU专家全部激活并由softmax门控组合），发现其验证损失随专家数量$K$呈非单调变化（$K=2$略优而$K=4$、$K=6$反而恶化），并证明该架构在数学上等价于带token依赖、单纯形约束组缩放的稠密SwiGLU前馈网络。

    

    稀疏混合专家模型将学习到的路由与从大型专家池中选择相结合。我们使用一种稠密类似物来分离动态专家组合的贡献：$K$个SwiGLU专家对每个token全部激活，并通过softmax门控进行组合，同时保持总FFN宽度固定。由于没有更大的专家池或离散选择，稠密基线即$K=1$的情形。验证损失随$K$呈非单调变化：$K=2$相比基线改善了$0.0048$，而$K=4$和$K=6$分别使其恶化$0.0053$和$0.0197$。路由通常保持软性且专家使用均衡，唯一的例外是$K=4$模型的第一层，其路由接近独热（one-hot）；将该门控强制设为均匀分布会使诊断子集上的损失增加$2.4$纳特，表明其集中的路由在功能上是重要的。我们进一步表明，该架构恰好等价于一个带有依赖token的、受单纯形约束的组缩放的稠密SwiGLU……

    arXiv:2610.02584v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts (MoE) models combine learned routing with selection from a large expert pool. We isolate the contribution of dynamic expert combination using a dense analogue: $K$ SwiGLU experts, all active for every token and combined by a softmax gate, at fixed total FFN width. With no larger pool or discrete selection, the dense baseline is the $K=1$ case. Validation loss varies non-monotonically with $K$: $K=2$ improves over the baseline by $0.0048$, whereas $K=4$ and $K=6$ worsen it by $0.0053$ and $0.0197$, respectively. Routing generally remains soft and expert usage balanced, except in the first layer of the $K=4$ model, where routing is nearly one-hot. Forcing this gate to uniform increases loss by $2.4$ nats on a diagnostic subset, indicating that its concentrated routing is functionally important. We further show that the architecture is exactly a dense SwiGLU with token-dependent, simplex-constrained group scaling a
    
[^207]: 高维渐近理论与隐私保护迁移学习中的数据集选择

    High-Dimensional Asymptotics and Dataset Selection for Private Transfer Learning

    [https://arxiv.org/abs/2610.02578](https://arxiv.org/abs/2610.02578)

    本文针对隐私保护迁移学习中的数据集选择问题，提出了一种仅基于汇总统计量、采用加权岭估计器的多元异构源高维回归方法，在 ρ-零集中差分隐私保证下研究其高维渐近性质，以判断额外数据是否值得购买或纳入协同学习。

    

    无论是购买外部数据还是参与协同学习，决策者都必须判断额外数据能否充分提升预测性能，从而证明其成本是合理的。这带来了几个挑战：(i) 决策通常只能依赖公开可得的聚合统计信息，而非个体层面的数据；(ii) 协变量偏移和模型偏移可能导致负迁移，使额外数据反而降低而非提升性能；(iii) 如果数据是敏感的，其隐私化处理需要注入噪声，这也可能抵消更大样本量带来的收益。本文通过具有多个异构数据源的高维回归以及加权岭估计器来建模数据集选择问题。我们的方法仅使用汇总统计信息，并能在 ρ-零集中差分隐私框架下，提供仅针对标签或同时针对特征与标签的隐私保证。

    arXiv:2610.02578v1 Announce Type: cross  Abstract: To commit to buying external data or participate in collaborative learning, one must decide whether the additional data will improve prediction enough to justify the cost. This comes with several challenges: (i) the decision often relies only on aggregated statistics available publicly, rather than individual-level data; (ii) covariate and model shifts can induce negative transfer, so the additional data deteriorates rather than improves performance; (iii) if the data is sensitive, its privatization requires the injection of noise, which can also offset the benefit of a larger sample size. In this paper, we model the problem of dataset selection through high-dimensional regression with multiple heterogeneous sources and a weighted ridge estimator. Our approach uses only summary statistics and it gives privacy guarantees either on labels only or jointly on features and labels, in terms of $\rho$-zero-concentrated differential privacy. T
    
[^208]: DISSOLVR：一个可解释且快速的水溶性与有机溶解度预测框架

    DISSOLVR: An Interpretable and Fast Framework for Aqueous and Organic Solubility Prediction

    [https://arxiv.org/abs/2610.02574](https://arxiv.org/abs/2610.02574)

    DISSOLVR是一个透明、可解释且快速的分子溶解度预测框架，它通过将分子映射到基于物理的描述符，实现了接近实验不确定性极限的预测精度和分布外泛化能力，证明最先进的性能无需依赖不透明的深度学习架构。

    

    高保真度的溶解度预测对药物研发和环境分配至关重要，准确的建模必须将分子结构与不同化学环境下的热力学行为相耦合。然而，近来的进展主要被深度学习架构所主导，这些架构往往以牺牲物理可解释性为代价来换取预测能力。我们通过证明最先进的性能并不需要此类不透明的架构来挑战这一趋势。为此，我们提出了DISSOLVR，一个用于分子溶解度预测的透明框架。此外，我们进行了全面的文献综述，并针对多种方法开展了基准测试研究。我们表明，DISSOLVR接近了实验不确定性的偶然误差极限，并通过将分子映射到基于物理的描述符所导出的结构不变性，实现了分布外（OOD）泛化。随后，我们提出了一种由大语言模型（LLM）辅助的……

    arXiv:2610.02574v1 Announce Type: new  Abstract: High-fidelity solubility prediction is fundamental to pharmaceutical development and environmental partitioning, where accurate modeling must couple molecular structure with thermodynamic behavior across diverse chemical environments. However, recent advancements have been dominated by deep learning architectures that often sacrifice physical interpretability for predictive power. We challenge this trend by showing that state-of-the-art performance does not require such non-transparent architectures. To address this, we introduce DISSOLVR, a transparent framework for molecular solubility prediction. In addition, we perform a comprehensive literature review and a benchmarking study against various methods. We show that DISSOLVR approaches the aleatoric limit of experimental uncertainty and achieves OOD generalization through structural invariance, derived by mapping molecules to physically-grounded descriptors. Then, we present an LLM-ass
    
[^209]: DAGS：面向时间稳定生成式渲染的冻结图像DiT解耦外观与几何引导

    DAGS: Disentangled Appearance-and-Geometry Steering of a Frozen Image DiT for Temporally Stabilized Generative Rendering

    [https://arxiv.org/abs/2610.02567](https://arxiv.org/abs/2610.02567)

    DAGS提出了一种轻量级、无需注意力机制的外观与几何解耦条件注入方案，将条件特征作为逐层残差注入冻结的图像DiT，并辅以循环光照稳定器和免训练时序引导，实现了高保真、高忠实度且时间稳定的生成式渲染。

    

    扩散Transformer（DiT）能够基于文本和图像条件生成高保真图像，但其输出存在较大方差，且对目标内容的忠实度在很大程度上取决于条件的提供方式。我们提出了DAGS，这是一种轻量级、无需注意力机制的外观与几何解耦条件化方案，可引导冻结的图像DiT生成高保真、高忠实度且可独立控制的渲染结果。两个小型卷积编码器每帧仅计算一次条件特征，并将其作为学习到的逐层逐元素残差注入图像token中，从而避免了通过注意力机制堆叠条件所带来的二次方计算开销。由于控制与时序处理均位于冻结主干网络之外，我们得以保留其庞大的预训练先验知识，并消除了主干网络过拟合的风险。我们进一步引入了一个小型循环光照稳定器和一个免训练的时序引导项，该引导项与我们的条件化方案相结合……

    arXiv:2610.02567v1 Announce Type: cross  Abstract: Diffusion transformers (DiTs) generate high-fidelity images from text and image conditions, but their outputs carry large variance and their faithfulness to a desired target depends heavily on how the condition is supplied. We present DAGS, a lightweight, attention-free, disentangled appearance and geometry conditioning scheme that steers a frozen image DiT to produce high-fidelity, highly faithful, and independently controllable renders. Two small convolutional encoders compute conditioning features once per frame and inject them as a learned, per-layer, element-wise residual into the image tokens, avoiding the quadratic cost of stacking conditions through attention. Because control and temporal handling live outside the frozen backbone, we retain its vast pretrained prior and eliminate backbone-overfitting risk. We further add a small recurrent lighting stabilizer and a training-free temporal guidance term that, coupled with our cond
    
[^210]: 基于核岭回归的动力学系统闭合学习

    Learning Closure of Dynamical Systems with Kernel Ridge Regression

    [https://arxiv.org/abs/2610.02564](https://arxiv.org/abs/2610.02564)

    本文提出基于核岭回归的闭合建模框架，用于学习ODE/PDE设置中的差分方程闭合和动理学方程矩闭合中的代数闭合，在Lorenz-63系统和Kuramoto-Sivashinsky方程上实现了准确的长期预测，并显著优于基于LSTM的闭合模型。

    

    我们开发了一个闭合建模框架，利用核岭回归（KRR）来识别动力学系统中缺失的组成部分。该框架解决两类闭合问题：一类是常微分方程（ODE）和偏微分方程（PDE）设置中出现的差分方程闭合，另一类是动理学方程中由矩闭合产生的代数闭合。对于第一类问题，我们在ODE设置中推导了误差界，该误差界量化了时间积分、未解析尺度的近似，以及将未解析尺度效应耦合到已解析求解器所需的插值所带来的贡献。在Lorenz-63系统和Kuramoto-Sivashinsky方程上的数值实验表明，该框架能够实现准确的长期预测，并相比基于LSTM的闭合模型有显著改进。对于第二类问题，我们针对一维动理学方程考虑矩闭合问题，通过将动理学通量与宏观通量之间的差异建模为已解析宏观变量的函数来进行处理。

    arXiv:2610.02564v1 Announce Type: cross  Abstract: We develop a closure modeling framework for identifying missing components of dynamical systems using Kernel Ridge Regression (KRR). The framework addresses two classes of closure problems: difference-equation closures arising in ODE and PDE settings, and algebraic closures arising from moment closure in kinetic equations. For the first class, we derive an error bound in an ODE setting that quantifies contributions from time integration, approximation of unresolved scales, and interpolation required to couple unresolved-scale effects to the resolved solver. Numerical experiments on the Lorenz-63 system and the Kuramoto-Sivashinsky equation demonstrate accurate long-horizon predictions and substantial improvements over an LSTM-based closure model. For the second class, we consider moment closure for a one-dimensional kinetic equation by modeling discrepancies between kinetic and macroscopic fluxes as a function of the resolved macroscop
    
[^211]: OpenGameEval：在有状态游戏引擎中对智能体编程与探索进行基准测试

    OpenGameEval: Benchmarking Agentic Programming and Exploration in a Stateful Game Engine

    [https://arxiv.org/abs/2610.02563](https://arxiv.org/abs/2610.02563)

    OpenGameEval是一个在Roblox Studio有状态游戏引擎中评估智能体游戏开发能力的基准框架，其核心创新在于通过分离观察工具与编辑工具来直接测量探索行为，实验发现前沿模型虽通过率相近但解决的任务各不相同，且最佳模型单次尝试仅能解决51.7%的任务。

    

    我们提出OpenGameEval，这是一个面向Roblox Studio中智能体游戏开发的基准测试与评估框架。它将语言模型作为智能体运行在可复现的、有状态的游戏引擎会话中，并通过可执行检查对每次运行进行评分，评分既针对被编辑的场景，也针对模拟的游戏会话。大多数智能体编程基准测试虽然要求探索，但只根据最终任务是否成功来评分。OpenGameEval在其八工具动作空间中将观察工具与编辑工具分离开来，从而可以直接测量探索行为。我们在84个人工精选的核心任务上测量了13个前沿模型的通过率与探索行为，每个任务进行16次尝试。这些任务对当前模型而言十分困难：最好的模型单次尝试仅能解决51.7%的任务，五次尝试全部成功的比例为39.4%，且有六个任务没有任何被测试的模型能够解决。处于前沿的模型通过解决不同的任务达到了相似的通过率：按任务所需的工作类型对任务进行划分……

    arXiv:2610.02563v1 Announce Type: cross  Abstract: We present OpenGameEval, a benchmark and evaluation framework for agentic game development inside Roblox Studio. It runs language models as agents in reproducible, stateful game-engine sessions and scores each run with executable checks, both on the edited scene and in a simulated play session. Most agentic coding benchmarks require exploration but score only final task success. OpenGameEval separates observation tools from editing tools in its eight-tool action space, so exploration can be measured directly. We measure the pass rates and exploration behavior of 13 frontier models on 84 human-curated core tasks, with 16 attempts per task.   The tasks are hard for current models. The best model solves 51.7% of tasks on a single attempt and 39.4% five times out of five, and no tested model solves six of the tasks. Models at the frontier reach similar pass rates by solving different tasks: splitting tasks by the kind of work they require 
    
[^212]: 基于逆激活回归的神经元合并方法用于Sigmoid神经网络的训练后压缩

    Neuron merging via inverse-activation regression for post-training compression of sigmoid neural networks

    [https://arxiv.org/abs/2610.02559](https://arxiv.org/abs/2610.02559)

    提出了一种通过逆激活函数将神经元响应映射回预激活空间并利用最小二乘法估计代表神经元参数的神经元合并方法，实现Sigmoid神经网络的训练后压缩并更好地保留有用信息。

    

    随着神经网络规模的持续增长，模型压缩对于在有限计算资源下实现高效推理变得越来越重要。结构化剪枝方法会移除那些被认为不太重要的神经元或通道，但被移除的单元可能仍然包含有用的信息。从对训练好的网络进行粗粒化的角度来看，当多个神经自由度被合并时，应该保留哪些信息是一个值得研究的问题。在本文中，我们讨论了基于聚类的合并方法来压缩训练好的神经网络。除了无数据的贡献加权平均方法之外，我们还提出了神经元合并方法，该方法通过逆激活函数将神经元响应映射回预激活空间，并使用最小二乘法估计每个代表性神经元的权重和偏置。我们还研究了数据辅助策略……

    arXiv:2610.02559v1 Announce Type: new  Abstract: As neural networks continue to grow in scale, model compression is becoming increasingly important for efficient inference under limited computational resources. Structured pruning methods remove neurons or channels that are estimated to be less important, but the removed units may still contain useful information. From the viewpoint of coarse-graining a trained network, it is valuable to ask which information should be retained when multiple neuronal degrees of freedom are consolidated. In this paper, we discuss cluster-based merging methods for compression of trained neural networks. In addition to a data-free contribution-weighted averaging method, we propose neuron-merging methods in which neuron responses are mapped back to the pre-activation space via the inverse activation function, and the weights and biases of each representative neuron are estimated using the least-squares method. We also examine both a data-assisted strategy w
    
[^213]: 如何进行一场敏锐的辩论：一种面向AI辩论的实例最优协议

    How to Have a Sensitive Debate: An Instance-Optimal Protocol for AI Debate

    [https://arxiv.org/abs/2610.02557](https://arxiv.org/abs/2610.02557)

    本文针对AI辩论设计了一种新的实例最优协议，对于具有足够稳定子问题分解的问题，在有限监督下比现有最佳协议提供更强的正确性保证。

    

    随着强大的AI系统在一系列高认知要求的任务上达到甚至超越人类专家的能力，对这些系统进行准确监督的问题变得日益紧迫。一种有前景的方法是AI辩论，它试图利用两个强大AI之间的辩论，将复杂问题分解为更简单、可以直接判断的论断。关于辩论的理论工作已经用计算复杂性理论的语言将这一直觉形式化，其目标是设计辩论协议（即辩论游戏的规则），以便在有限监督下为复杂问题解的判断提供严格的正确性保证。具体而言，目前最优的协议已被证明适用于所有具有足够稳定子问题分解的问题。在本文中，我们为同一类问题设计了一种新协议，该协议在先前工作的基础上进行了改进。

    arXiv:2610.02557v1 Announce Type: new  Abstract: As powerful AI systems reach and sometimes surpass the abilities of human experts across a range of cognitively demanding tasks, the problem of accurate oversight and supervision of these systems has become increasingly urgent. One promising approach is AI debate, which seeks to leverage a debate between two powerful AIs to break complex questions down into simpler claims that can be easily judged directly. Theoretical work on debate has formalized this intuition in the language of computational complexity theory, where the goal is to design protocols (i.e., rules of the debate game) that provide rigorous guarantees on correctness for judging solutions to complex problems with limited supervision. Specifically, the current best protocol has been shown to work for all problems that have sufficiently stable decompositions into subproblems. In this paper, we design a new protocol for this same class of problems that improves on the prior wo
    
[^214]: 基于分解价值梯度流的测试时多智能体协同

    Test-time Multi-agent Coordination by Decomposed Value Gradient Flow

    [https://arxiv.org/abs/2610.02554](https://arxiv.org/abs/2610.02554)

    提出SCOUT框架，首次通过测试时动作精炼将流匹配行为先验与分解价值函数结合，利用Stein变分梯度下降将行为样本传输至高价值区域，从而解决离线多智能体强化学习中生成式策略与价值优化策略之间的权衡问题。

    

    离线多智能体强化学习（MARL）面临一个持久的权衡问题：表达力强的生成式策略能够表示数据中的多模态协同行为，但无法区分高价值区域；而价值优化的策略虽然能利用学习到的Q函数，却会将多模态坍缩为单一主导模式。单个智能体的模式坍缩可能破坏联合协同，而多个智能体同时漂移则可能将联合策略推入动作空间中未见过的区域。我们提出SCOUT（通过最优统一传输实现可扩展协同），这是首个通过测试时动作精炼将生成式基础模型与学习到的价值函数相结合的离线MARL框架。SCOUT训练两个解耦的组件：流匹配行为先验和分解的价值函数。在测试时，它通过Stein变分梯度下降将行为样本传输至高价值区域。传输步数……

    arXiv:2610.02554v1 Announce Type: new  Abstract: Offline multi-agent reinforcement learning (MARL) faces a persistent trade-off. Expressive generative policies can represent multi-modal coordination in the data, but cannot distinguish high-value regions, while value-optimized policies exploit the learned Q-function but collapse the multi-modal into a single dominant mode. A single agent's mode collapse can break joint coordination, and simultaneous drift across agents can push the joint policy into unseen regions of the action space. We propose scalable coordination via optimal unified transport (SCOUT), the first offline MARL framework to combine a generative foundation model with a learned value function through test-time action refinement. SCOUT trains two decoupled components: a flow-matching behavioral prior and a decomposed value function. At test-time, it transports behavioral samples toward high-value regions via Stein variational gradient descent. The number of transport steps
    
[^215]: 奖励通胀：强化学习的健康刺激剂

    Reward Inflation: A Healthy Stimulus for Reinforcement Learning

    [https://arxiv.org/abs/2610.02545](https://arxiv.org/abs/2610.02545)

    提出奖励通胀方法，通过在训练中逐渐放大奖励实现隐式近期加权，加速策略适应、抑制休眠神经元并保持网络可塑性，在ALE游戏和MuJoCo任务上显著提升强化学习性能。

    

    奖励是强化学习（RL）中的主要学习信号。然而，尽管奖励的大小通常在整个训练过程中保持固定，但其时间上的调制却一直缺乏充分研究。在本文中，我们提出了“奖励通胀”，即在训练过程中逐渐放大奖励的尺度，并证明它可以作为强化学习的一种健康刺激。从理论上讲，奖励通胀引入了一种隐式的近期性加权，在策略更新时对最近的转移赋予更高的权重，从而实现更快的适应。我们进一步证明，通过在策略趋于饱和时维持梯度信号，奖励通胀能够抑制休眠神经元的出现，并有助于保持网络的可塑性。在ALE游戏和MuJoCo任务上的实证结果证实了这些发现，表明适当程度的奖励通胀对广泛的任务都有益处。最后，我们介绍了Fed，一种能够动态调整通胀水平的自适应变体……

    arXiv:2610.02545v1 Announce Type: new  Abstract: Reward serves as the primary learning signal in reinforcement learning (RL). However, while reward magnitudes are typically held fixed throughout training, their temporal modulation remains underexplored. In this paper, we propose reward inflation, a gradual scaling of rewards over the course of training, and show that it can act as a healthy stimulus for RL. Theoretically, reward inflation induces an implicit recency weighting that upweights recent transitions during policy updates, enabling faster adaptation. We further show that, by sustaining gradient signals as the policy saturates, reward inflation suppresses the emergence of dormant neurons and helps preserve plasticity. Empirical results on ALE games and MuJoCo tasks corroborate these findings, showing that an appropriate level of reward inflation benefits a broad range of tasks. Finally, we introduce Fed, an adaptive variant that adjusts the inflation level on the fly, and find 
    
[^216]: ENCORE：基于副本交换的扩散生成精确非平衡控制

    ENCORE: Exact Non-equilibrium COntrol with Replica Exchange for Diffusion Generation

    [https://arxiv.org/abs/2610.02538](https://arxiv.org/abs/2610.02538)

    该论文提出了首个精确的并行推理时控制方法ENCORE，通过让每个副本保存生成轨迹使向上移动成为截断操作，从而避免模拟难以处理的时间反转，实现了无偏的副本交换控制。

    

    推理时控制能够在无需重新训练的情况下，将预训练的生成模型引导至目标分布。我们研究了倾斜目标分布 $\pi_0\propto G_0\,p_0$，其中 $p_0$ 是采样器的输出分布，$G_0$ 是可评估的重加权函数。现有方法依赖于基于序贯蒙特卡罗（SMC）的序贯退火，或基于副本交换（RE）的并行退火。序贯控制是精确的，但需要庞大的粒子群体；然而目前尚不存在精确的并行控制方法：现有的 RE 校正方法近似模拟了一个难以处理的时间反转过程，因而存在偏差。我们提出了基于副本交换的精确非平衡控制方法（ENCORE），这是首个精确的并行控制方法。每个副本保存其生成轨迹，因此向上移动仅是一种截断操作，从而无需模拟难以处理的时间反转过程。我们证明了目标分布的不变性，并表明所得动力学即为非平衡副本交换的动力学（摘要在此处截断）

    arXiv:2610.02538v1 Announce Type: cross  Abstract: Inference-time control steers a pretrained generative model towards a target distribution without retraining. We study tilted targets $\pi_0\propto G_0\,p_0$, where $p_0$ is the sampler output distribution and $G_0$ is an evaluable reweighting function. Existing approaches rely on sequential annealing with sequential Monte Carlo (SMC) or parallel annealing with replica exchange (RE). Sequential control is exact but needs large particle populations, whereas no exact parallel control method exists: existing RE corrections approximate an intractable time reversal and are biased. We propose Exact Non-equilibrium COntrol with Replica Exchange (ENCORE), the first exact parallel control method. Each replica stores its generation trajectory, so the upward move is a truncation and the intractable time reversal is never simulated. We prove target invariance and show that the resulting dynamics are those of non-equilibrium replica exchange with t
    
[^217]: 整数规划的自回归可微方法

    Autoregressive Differentiable Method for Integer Programming

    [https://arxiv.org/abs/2610.02528](https://arxiv.org/abs/2610.02528)

    该论文提出了一种自回归可微方法，通过训练Transformer并结合拉格朗日惩罚与Gumbel-softmax激活来求解0-1整数规划，在多达10,000个变量的稠密二次背包问题上持续超越最先进的开源求解器。

    

    我们提出了一种自回归可微方法来求解0-1整数规划。我们固定二进制变量的任意顺序，并训练一个Transformer在保持在可行集内的同时预测下一个比特。我们的方法首先在任何求解器提供的可行解上进行训练，从而使我们能够在可行集内初始化Transformer。随后，我们的过程采用拉格朗日惩罚来惩罚不可行解，并通过在松弛目标上使用Gumbel-softmax激活，进一步训练Transformer来探索可行集。我们在二次背包问题的非凸实例上测试了该方法，并证明对于多达10,000个二进制变量的稠密问题，我们的方法相比最先进的开源求解器取得了一致的改进。特别是，我们在实证上展示了一种类似隧穿效应的现象，即从二进制变量到连续变量的有效变量变换……

    arXiv:2610.02528v1 Announce Type: new  Abstract: We introduce an autoregressive differentiable method to solve 0-1 integer programs. We fix an arbitrary order of the binary variables and we train a transformer to predict the next bit while remaining in the feasible set. Our method is first trained on feasible incumbents provided by any solver, thus allowing us to initialize the transformer in the feasible set. Our procedure then implements a Lagrangian penalty to penalize infeasible solutions, and the transformer is further trained to explore the feasible set using Gumbel-softmax activations on the relaxed objective. We have tested our method on non-convex instances of quadratic knapsack problem and demonstrated consistent improvement upon state-of-the-art open-source solvers for dense problems up to 10,000 binary variables. In particular, we empirically demonstrate a phenomenon akin to a tunneling effect where the effective change of variables from binary variable to the continuous we
    
[^218]: CriticHack：在机器人策略优化下评估视觉奖励

    CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization

    [https://arxiv.org/abs/2610.02527](https://arxiv.org/abs/2610.02527)

    该论文揭示，用学习型视觉奖励模型优化机器人策略时，奖励分数与任务成功率可能同时上升、看似健康，但实际上会显著放大“作用于错误物体”的隐蔽失败，而这一现象在使用模拟器真实任务完成信号训练时并不会出现。

    

    学习得到的视觉奖励模型越来越多地被用于优化机器人策略，然而奖励模型可能会对作用于错误物体的执行给出与真正完成任务同样高的评分。我们证明，针对这样的奖励进行优化，可能在奖励值与任务成功率双双上升的同时放大这类“错误物体”失败，使得从业者通常会监测的信号看起来一切正常。我们在一个抽屉任务上，针对 Robometer 对扩散策略去噪器的每一个参数进行微调。从一个没有任何奖励先验的监督策略出发，五次训练运行在 512 个评估种子上将任务成功率提高了 10.2 个百分点，同时使错误物体失败增加了 10.9 个百分点；而使用模拟器任务完成信号训练的五次运行则在提高成功率的同时没有放大错误物体失败（差异为 9.2 个百分点，95% 置信区间为 5.6 至 13.0）。这种放大现象在一个先前已针对学习奖励优化过的策略上同样会出现，在该策略原生的……（摘要在此处截断）

    arXiv:2610.02527v1 Announce Type: cross  Abstract: Learned visual reward models are increasingly used to optimize robot policies, yet a reward model can score an execution that acts on the wrong object as highly as one that completes the task. We show that optimizing such a reward can amplify these wrong-object failures while reward and task success both rise, so the signals a practitioner would normally monitor look healthy. We fine-tune every denoiser parameter of a diffusion policy against Robometer on a drawer task. Starting from a supervised policy with no prior reward exposure, five training runs raise task success by 10.2 percentage points and wrong-object failures by 10.9 points on 512 evaluation seeds, whereas five runs trained on the simulator's task-completion signal raise success without amplifying wrong-object failures (difference 9.2 points, 95% CI 5.6 to 13.0). The amplification recurs from a policy previously optimized against learned rewards, under the policy's native 
    
[^219]: BaCP：面向极稀疏神经网络表征保持的主干对比剪枝

    BaCP: Backbone Contrastive Pruning for Preserving Representations in Extremely Sparse Neural Networks

    [https://arxiv.org/abs/2610.02524](https://arxiv.org/abs/2610.02524)

    提出主干对比剪枝（BaCP），通过将稀疏网络的嵌入空间与预训练、微调及历史快照模型对齐来正则化，解决极稀疏剪枝中的表征坍塌问题，在90种设置下显著提升极稀疏时的准确率。

    

    极稀疏度下的非结构化剪枝常常遭受表征坍塌问题，导致准确率急剧下降。为了解决这一问题，我们研究了主干对比剪枝（BaCP），该方法通过将稀疏网络的嵌入空间与预训练模型、微调模型以及历史快照模型对齐来进行正则化。基于CAP框架（Xu等，2022）的对比分解，我们在多种剪枝准则下对该方法提供了严格的等预算刻画。在90种设置下的评估表明，BaCP在标准剪枝失效的极稀疏区间显著提升准确率，而在表征保持完好时则接近基线水平。

    arXiv:2610.02524v1 Announce Type: new  Abstract: Unstructured pruning at extreme sparsity often suffers from representational collapse, causing sharp drops in accuracy. To address this, we study Backbone Contrastive Pruning (BaCP), which regularizes the sparse network's embedding space by aligning it with pretrained, fine-tuned, and historical snapshot models. Building on the contrastive decomposition of the CAP framework (Xu et al., 2022), we provide a rigorous matched-budget characterization of this approach across multiple pruning criteria. Evaluated across 90 settings, BaCP improves accuracy substantially in extreme sparsity regimes where standard pruning fails, and is close to baseline where representations remain intact.
    
[^220]: 逐步约束下受限马尔可夫决策过程（CMDP）的实例相关遗憾

    Instance-Dependent Regret for CMDPs with Step-Wise Constraints

    [https://arxiv.org/abs/2610.02520](https://arxiv.org/abs/2610.02520)

    本文提出了安全方差自适应探索算法（SVAE），通过学习候选安全子图并在其中进行方差自适应的乐观规划，在具有逐步安全约束的情景式CMDP中首次实现了依赖问题实例（方差感知）的累积遗憾界。

    

    我们研究了具有逐步安全约束的情景式表格型受限马尔可夫决策过程（CMDP）中的在线学习问题。在这种设定下，安全约束会诱导出一个安全子图，该子图刻画了可行策略下累积奖励的方差，进而决定了学习的难度。然而，要利用这一结构，需要在控制约束违反的同时学习哪些动作是安全的。我们提出了安全方差自适应探索算法，这是一种高效算法，它能够学习候选安全子图，并在其中执行方差自适应的乐观规划。以高概率，SVAE在 $K$ 个情景下实现了阶为 $\widetilde{\mathcal{O}}(\sqrt{SAH\min\{\mathbb{V}_\Sigma,K\mathrm{Var}^{\star}\}}+S\sqrt{AH^3\min\{K,\mathcal{C}\}}+S^2AH^2)$ 的累积遗憾，其中 $H$ 是单个情景的时域长度，$S$ 和 $A$ 分别表示状态数和动作数。这里，$\mathrm{Var}^{\star}$ 是……（原文摘要在此处截断）

    arXiv:2610.02520v1 Announce Type: cross  Abstract: We study online learning in episodic tabular constrained Markov decision processes with step-wise safety constraints. In such a setting, the constraints induce a safe subgraph that shapes the variance of cumulative rewards under feasible policies and, consequently, the difficulty of learning. Exploiting this structure, however, requires learning which actions are safe while controlling constraint violations. We propose Safe Variance-Adaptive Exploration (SVAE), an efficient algorithm that learns candidate safe subgraphs and performs variance-adaptive optimistic planning within them. With high probability, SVAE achieves cumulative regret of order $\widetilde{\mathcal{O}}(\sqrt{SAH\min\{\mathbb{V}_\Sigma,K\mathrm{Var}^{\star}\}}+S\sqrt{AH^3\min\{K,\mathcal{C}\}}+S^2AH^2)$ over $K$ episodes, where $H$ is the horizon of a single episode, while $S$ and $A$ are the numbers of states and actions, respectively. Here, $\mathrm{Var}^{\star}$ is 
    
[^221]: 学习潜在结构：一种以特征为中心的图数据增强方法

    Learning the Latent Structure: A Feature-Centric Approach to Graph Data Augmentation

    [https://arxiv.org/abs/2610.02517](https://arxiv.org/abs/2610.02517)

    该论文提出了一种以特征为中心的图数据增强框架，通过在嵌入空间中进行自监督逆掩码过程来捕获观测图与完整图之间的潜在结构联系，从而恢复未观测的结构信号，克服了现有方法依赖标签、计算昂贵和转导式的局限性。

    

    图结构数据在建模复杂关系中发挥着关键作用。然而，由于数据收集和观测限制，现实世界中的图往往是不完整的，这严重限制了现代图学习流程的有效性。虽然现有的图数据增强（GDA）方法试图通过改进图结构来提升下游性能，但它们通常依赖标签、计算成本高昂，且本质上是转导式的，这限制了它们在实际场景中的适用性。在这项工作中，我们提出了一种新颖的以特征为中心的图数据增强框架，该框架通过直接在嵌入空间中操作来绕过显式结构建模。通过自监督的逆掩码过程，我们的方法捕获了观测图与完整图之间的潜在联系，能够通过精炼的节点表示恢复未观测到的结构信号。为了增强在噪声和稀疏监督条件下的鲁棒性……

    arXiv:2610.02517v1 Announce Type: new  Abstract: Graph-structured data plays a pivotal role in modeling complex relationships. However, real-world graphs are often incomplete due to data collection and observational constraints, severely limiting the effectiveness of modern graph learning pipelines. While existing Graph Data Augmentation (GDA) methods attempt to refine graph structures for improved downstream performance, they are typically label-dependent, computationally expensive, and inherently transductive, limiting their applicability in practical scenarios. In this work, we present a novel feature-centric graph data augmentation framework that bypasses explicit structure modeling by operating directly in the embedding space. Through a self-supervised inverse masking process, our method captures latent ties between observed and complete graphs, enabling recovery of unobserved structural signals through refined node representations. To enhance robustness under noisy and sparse sup
    
[^222]: 面向高效LLM任务路由的学生引导教师蒸馏：与Jev式System-1分类器的定位对比

    Student-Guided Teacher Distillation for Efficient LLM Task Routing: Positioning Against Jev-Style System-1 Classifiers

    [https://arxiv.org/abs/2610.02516](https://arxiv.org/abs/2610.02516)

    提出一种学生引导的教师蒸馏流水线：紧凑的ModernBERT学生模型单次前向预测完整类别分布并生成top-k候选，更大的DeBERTa-v3零样本NLI教师模型仅对候选重排序，教师标签迭代反哺学生，从而显著降低大规模LLM任务路由的成本。

    

    零样本分类器可用于将用户请求路由到专门的LLM任务，但对每个请求在大型候选集合上进行打分代价高昂：零样本NLI分类器必须对每个标签评估一个前提-假设对，因此成本随分类体系规模线性增长。我们研究了一种针对固定60个LLM任务类别分类体系的学生引导教师蒸馏流水线：一个紧凑的ModernBERT分类器在单次前向传播中预测完整的类别分布并检索出较小的top-k候选集，然后由一个更大的DeBERTa-v3零样本NLI分类器仅对这些候选进行重排序，而非对全部60个标签逐一评估；由此产生的教师标签会迭代地改进学生模型，使学生模型在下一轮中生成更精准的候选。与极端多标签分类中常用的通用嵌入检索或基于聚类生成的短名单不同，我们的候选生成器是在目标分类体系上进行端到端训练的，并且是由同一模型提供服务……

    arXiv:2610.02516v1 Announce Type: cross  Abstract: Zero-shot classifiers are useful for routing user requests to specialized LLM tasks, but scoring every request against a large candidate set is expensive: a zero-shot NLI classifier must evaluate one premise-hypothesis pair per label, so cost scales linearly with taxonomy size. We study a student-guided teacher distillation pipeline for a fixed taxonomy of 60 LLM task categories: a compact ModernBERT classifier predicts the full category distribution in one forward pass and retrieves a small top-k candidate set, and a larger DeBERTa-v3 zero-shot NLI classifier reranks only those candidates rather than all 60 labels; the resulting teacher labels iteratively improve the student, which produces sharper candidates for the next round. Unlike generic embedding retrieval or clustering-derived shortlists used in extreme multi-label classification, our candidate generator is trained end-to-end on the target taxonomy and is the same model servin
    
[^223]: IGNITE 托卡马克世界模型架构

    IGNITE Tokamak World Model Architecture

    [https://arxiv.org/abs/2610.02515](https://arxiv.org/abs/2610.02515)

    IGNITE是首个基于DIII-D十年实验数据自监督训练的聚变等离子体生成式世界基础模型，可通过执行器轨迹、文本提示或期望实验结果模拟完整的托卡马克放电过程。

    

    我们提出了IGNITE，一个用于聚变等离子体行为模拟的生成式世界基础模型，该模型在DIII-D国家聚变设施超过十年的无标签实验数据上以自监督方式训练而成。IGNITE的核心是一个动力学模型，能够根据给定的一组执行器轨迹来模拟DIII-D等离子体放电。这些轨迹既可以由外部提供，也可以根据文本提示或期望的实验结果即时生成。该模型架构包含多个时空分词器，用于嵌入不同的输入模态，包括时间序列类的时空测量数据、图像序列以及高分辨率光谱图，每一种模态都是在截然不同的时间尺度上采集的。其主干网络由一个自回归动力学模型构成，在给定初始潜在等离子体状态和执行器轨迹的条件下，该模型理论上能够预测无限长度的完整DIII-D放电过程。

    arXiv:2610.02515v1 Announce Type: cross  Abstract: We introduce IGNITE, a generative world foundation model for fusion plasma behavior simulation trained in a self-supervised manner from over a decade of unlabeled experimental data at the DIII-D National Fusion Facility. The core of IGNITE is a dynamics model that can simulate DIII-D discharges from a given set of actuator trajectories. These trajectories can be supplied or generated on-the-fly from a textual prompt or from desired experimental outcomes. The model architecture consists of several spatio-temporal tokenizers that embed the different input modalities, including time-series like spatio-temporal measurement data, image sequences, and high-resolution spectrograms, each of which collected at vastly different time scales. The backbone is composed of an auto-regressive dynamics model that has the capacity to predict entire DIII-D discharges given initial latent plasma states and actuator trajectories over a theoretical infinite
    
[^224]: 自回归天气模型的后训练量化

    Post-Training Quantization of Autoregressive Weather Models

    [https://arxiv.org/abs/2610.02511](https://arxiv.org/abs/2610.02511)

    该论文将后训练量化（PTQ）技术应用于自回归深度学习天气预报模型，以加速推理计算、降低功耗，并使其能够部署在边缘硬件上。

    

    高分辨率数值天气预报（NWP）和数据同化（DA）的进展推动了模拟大气动力学的深度学习（DL）架构的发展。天气预报模拟器在从数天到次季节时间尺度的预报时效范围内，表现出与基于物理的模型相当的预报质量。这些模拟器在自回归推理中依靠硬件加速的矩阵乘法驱动，显著减少了数值天气预报所需的计算时间和资源。对GPU架构中矩阵乘法过程的优化提供了向高分辨率领域扩展的机会，并支持开箱即用解决方案的实施。后训练量化（PTQ）已在多种深度学习架构中得到验证，它能够加速计算并增加单位时间内的计算量，同时降低功耗，使模型能够部署在边缘硬件上。

    arXiv:2610.02511v1 Announce Type: new  Abstract: Advancements in high-resolution numerical weather prediction (NWP) and data assimilation (DA) have shaped the developments in deep learning (DL) architectures emulating atmospheric dynamics. Emulators for weather forecasting exhibit forecast quality comparable to physics based models at forecast horizon scaling from few days to subseasonal time scales. The emulators are driven by hardware-accelerated matrix multiplication in autoregressive inferences, significantly reducing the computation time and resources required for NWP. Optimization of the matrix multiplication processes in GPU architectures provides opportunities to scale towards high-resolution domain, and offers implementation of out of the box solutions. Post-training quantization (PTQ) has been demonstrated across multiple DL architectures to accelerate and increase the number of computations in unit time while consuming less power, enabling applications on edge hardware. In t
    
[^225]: 多保真度策略梯度稳定数据稀缺的强化学习

    Multi-Fidelity Policy Gradients Stabilize Data-Scarce Reinforcement Learning

    [https://arxiv.org/abs/2610.02505](https://arxiv.org/abs/2610.02505)

    本文将多保真度策略梯度（MFPG）框架从REINFORCE扩展到现代演员-评论家算法（如PPO），在GPU并行仿真和真实机器人上利用低保真度数据构建控制变量，以在无偏差的前提下降低梯度方差，从而稳定数据稀缺场景下的强化学习。

    

    同策略强化学习（RL）中的策略梯度方法，在昂贵且稀缺的目标域数据产生噪声较大的梯度估计时可能会变得不稳定。我们通过利用大量、廉价但有偏差的低保真度（LF）数据（例如来自简化模拟器的数据）来补充有限的高保真度（HF）目标域数据，从而应对这一挑战。大多数现有方法直接基于LF数据优化有偏差的目标函数。相比之下，最近提出的多种保真度策略梯度（MFPG）框架仅将LF数据用于构建控制变量，在不使策略梯度估计器产生偏差的情况下降低方差并提高HF数据的利用效率。然而，已发表的关于MFPG的工作仅限于在小规模仿真任务上使用REINFORCE。我们将MFPG发展到GPU并行仿真和物理机器人上的现代演员-评论家（actor-critic）学习中。我们的分析和实验表明，对近端策略优化（PPO）的简单扩展可能会失去……

    arXiv:2610.02505v1 Announce Type: cross  Abstract: Policy gradient methods for on-policy reinforcement learning (RL) can become unstable when expensive, scarce target-domain data yield noisy gradient estimates. We address this challenge by complementing limited high-fidelity (HF) target-domain data with abundant, cheap, but biased low-fidelity (LF) data, e.g., from a simplified simulator. Most existing methods directly optimize biased objectives based on LF data. In contrast, the recently introduced multi-fidelity policy gradient (MFPG) framework uses LF data solely to construct a control variate that reduces variance and improves HF data efficiency without biasing the policy gradient estimator. However, published work on MFPG is limited to REINFORCE on small-scale simulation tasks. We develop MFPG for modern actor-critic learning in GPU-parallel simulation and on a physical robot. Our analysis and experiments show that naive extensions to proximal policy optimization (PPO) can lose cr
    
[^226]: 复合AI系统可靠性：基于150起生产事故的故障分类学与韧性模式目录

    Compound AI System Reliability: A Failure Taxonomy and Resilience Pattern Catalog from 150 Production Incidents

    [https://arxiv.org/abs/2610.02503](https://arxiv.org/abs/2610.02503)

    本文通过分析150起生产事故，构建了包含23种故障模式、分五大类别的复合AI系统故障分类学，并提出经故障注入实验验证有效性的韧性模式（如断路器减少89%级联传播、质量门控捕获73%静默退化）。

    

    可靠且安全地部署复合AI系统，需要理解出现在组件边界而非单个模型内部的故障模式。级联错误会在组件边界之间传播，静默的质量退化会逃避标准监控，而协调失败会导致由各自正确的部件产生错误的集体行为。我们分析了来自开源复合AI项目和匿名化企业部署的150份生产事故报告，构建了一个包含23种故障模式的分类体系，分为五大类别：检索故障、生成故障、工具故障、编排故障和集成故障。针对每个类别，我们提出了相应的韧性模式，并通过受控故障注入实验测量了其有效性。断路器可将级联传播减少89%，输出质量门控能在影响用户之前捕获73%的静默退化，组件隔离可缩小故障爆炸半径。

    arXiv:2610.02503v1 Announce Type: cross  Abstract: Deploying compound AI systems reliably and safely requires understanding failure modes that emerge at component boundaries, not within individual models. Cascading errors propagate across component boundaries, silent quality degradation evades standard monitoring, and coordination failures yield incorrect collective behavior from individually correct parts. We analyze 150 production incident reports from open-source compound AI projects and anonymized enterprise deployments to construct a taxonomy of 23 failure modes organized into five categories: retrieval failures, generation failures, tool failures, orchestration failures, and integration failures. For each category, we propose resilience patterns with measured effectiveness from controlled fault injection experiments. Circuit breakers reduce cascade propagation by 89%, output quality gates catch 73% of silent degradation before user impact, and component isolation reduces blast ra
    
[^227]: LiteEMG-FM：一种用于鲁棒肌电传感的高效可部署基础模型

    LiteEMG-FM: An Efficient and Deployable Foundation Model for Robust EMG Sensing

    [https://arxiv.org/abs/2610.02497](https://arxiv.org/abs/2610.02497)

    本文提出LiteEMG-FM，一种在16个多样化EMG数据集上预训练的高效CNN-Transformer混合基础模型，通过分层唤醒架构（轻量级1D-CNN预筛）实现跨用户泛化且适合资源受限可穿戴设备的实时部署。

    

    肌电图（EMG）信号在个体、身体部位、记录会话和传感硬件之间存在显著差异，这限制了辅助设备和人机交互模型的泛化能力。现有的时间序列基础模型在实时可穿戴设备部署方面计算成本高昂，且往往无法捕捉EMG特有的时频特征。我们提出了LiteEMG-FM，一种面向实用EMG传感的高效CNN-Transformer混合基础模型。LiteEMG-FM在16个多样化的上肢和下肢EMG数据集上进行了预训练，学习到能够跨用户和跨数据集泛化的表示。针对资源受限的部署场景，我们实现了一种分层唤醒架构，其中轻量级、始终运行的1D-CNN负责过滤静息状态和非目标活动，仅在检测到有效手势时才激活LiteEMG-FM。我们评估了完全推理卸载、拆分推理和完全设备端处理三种模式，并表征……（原文摘要至此被截断）

    arXiv:2610.02497v1 Announce Type: new  Abstract: Electromyography (EMG) signals vary substantially across individuals, body regions, recording sessions, and sensing hardware, limiting the generalization of models for assistive devices and human-computer interaction. Existing time-series foundation models are also computationally expensive for real-time wearable deployment and often fail to capture EMG-specific time-frequency characteristics. We present LiteEMG-FM, an efficient hybrid CNN-Transformer foundation model for practical EMG sensing. Pretrained on 16 diverse upper- and lower-limb EMG datasets, LiteEMG-FM learns representations that generalize across users and datasets. For resource-constrained deployment, we implement a hierarchical wake-up architecture in which a lightweight, always-on 1D-CNN filters rest and non-target activity and activates LiteEMG-FM only for valid gestures. We evaluate full inference offloading, split inference, and full on-device processing, characterizi
    
[^228]: 排序正确，尺度有误：审计用于职业AI测量的LLM评判器

    Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement

    [https://arxiv.org/abs/2610.02492](https://arxiv.org/abs/2610.02492)

    该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。

    

    LLM评判器正被越来越多地用于评估AI输出是否满足职场要求，但对回答排序的一致性并不能确立在接受率或职业总体层面的一致性。我们提出了O*NET-BENCH，这是一个基于包含45,796名工人评分的现有调查构建的审计套件，并在4,501个测试评分上评估了来自六个模型系列的33种现有评判器配置。其中25种配置实现了至少0.60的平局感知成对排序准确率，尽管一个经训练拟合的仅基于回答文本的TF-IDF基线几乎与最强评判器表现相当。尽管存在这种排序上的一致性，评判器对回答可接受比例的估计介于3.0%至97.9%之间，而职业匹配的人类工人给出的估计为61.1%。在一个微调模型谱系中，从逐点评分切换到捆绑的少样本/列表式评估协议虽然改善了回答排序，却降低了在任务和职业层面与工人平均评分的一致性；这一反转现象在一个任务子集上得到了复现。

    arXiv:2610.02492v1 Announce Type: new  Abstract: LLM judges are increasingly used to assess whether AI outputs meet workplace requirements, but agreement on response rankings does not establish agreement on acceptance rates or occupational aggregates. We introduce O*NET-BENCH, an audit suite derived from an existing survey of 45,796 worker ratings, and evaluate 33 pre-existing judge configurations across six model families on 4,501 test ratings. Twenty-five configurations achieve tie-aware pair accuracy of at least 0.60, although a train-fitted response-only TF-IDF baseline nearly matches the strongest judge. Despite this ordering agreement, judges estimate that 3.0%-97.9% of responses are acceptable, compared with 61.1% for occupation-matched workers. In one fine-tuned lineage, changing from pointwise scoring to a bundled few-shot/listwise protocol improves response ordering while reducing agreement with worker means at the task and occupation levels; this reversal replicates on a tas
    
[^229]: 将大语言模型用作智能体：代价几何？

    Harnessing LLMs as Agents: What Does It Cost?

    [https://arxiv.org/abs/2610.02488](https://arxiv.org/abs/2610.02488)

    该论文提出语言模型智能体机（LAM）这一资源受限的计算抽象，首次从通信、内存访问、重计算与验证等方面严格量化 LLM 智能体驱动框架所消耗的计算资源，并给出相应的计算量下界。

    

    语言模型智能体日益依赖“驱动框架”来管理有限上下文、持久记忆、工具、验证与重复执行，然而现有的模型能力概念并未量化这些机制所消耗的计算资源。我们提出了语言模型智能体机，这是一种资源受限的抽象，它在固定底层语义模型的同时，显式地对驱动框架层面的资源进行计费。我们建立了四类结果。通信：在调用—传输预算同时受限的情况下，LAM 的执行在实例级别上等价于红蓝卵石游戏，从而将经典的 I/O 下界转移到上下文—内存流量上。访问：内存接口会引发渐近分离，包括在指针追踪任务上随机访问与非投机顺序访问之间 Θ(n) 的差距。重计算：位反转 DAG 在上下文容量为 C、持久记忆容量为 ……（摘要原文在此处截断）

    arXiv:2610.02488v1 Announce Type: new  Abstract: Language-model agents increasingly rely on harnesses that manage bounded context, persistent memory, tools, verification, and repeated execution, yet existing notions of model capability do not quantify the computational resources these mechanisms consume. We introduce the Language Model Agent Machine (LAM), a resource-bounded abstraction that fixes the underlying semantic model while explicitly charging harness-level resources. We establish four classes of results. Communication: LAM execution is instancewise equivalent to red--blue pebbling under simultaneous call--transfer budgets, transferring classical I/O lower bounds to context--memory traffic. Access: memory interfaces induce asymptotic separations, including a $\Theta(n)$ gap between random and non-speculative sequential access on pointer chasing. Recomputation: bit-reversal DAGs require $\Theta(n^2/(C+S)+n)$ model calls with context capacity $C$ and persistent-memory capacity $
    
[^230]: 阈值感知的保形路由

    Threshold-Aware Conformal Routing

    [https://arxiv.org/abs/2610.02487](https://arxiv.org/abs/2610.02487)

    该论文提出一种阈值感知的保形路由方法，依据代理模型保形区间与决策阈值的相对位置来决定是使用快速代理模型还是昂贵的高保真仿真，从而在保证下游决策可靠性的同时大幅降低仿真开销。

    

    高保真仿真对科学与工程设计至关重要，但反复运行的成本高昂。学习型代理模型（surrogate）提供了更快速的替代方案，但其较高的误差可能会改变下游决策。这种精度与速度之间的权衡使得我们需要判断：能否使用代理模型，还是仍需运行完整仿真器。我们研究这样一类决策问题：其结果由某个标量关注量（quantity of interest）高于或低于固定阈值所决定。对于每个输入，当代理模型给出的保形区间完全位于阈值一侧时，我们使用代理模型；当该区间与阈值相交时，则将输入路由至完整仿真。标准保形预测在构造区间时并不参考下游决策阈值：即使是一个靠近阈值的狭窄区间也可能跨越阈值并触发仿真，而一个更宽但远离阈值的区间则可以完全位于一侧而无需仿真。我们提出了阈值感知保形……（原文摘要在此处截断）

    arXiv:2610.02487v1 Announce Type: new  Abstract: High-fidelity simulations are essential to scientific and engineering design, but can be expensive to run repeatedly. Learned surrogates offer a faster alternative, yet their higher errors may alter downstream decisions. This accuracy-speed tradeoff creates a need to determine whether a surrogate can be used or the full simulator remains necessary. We study decisions determined by whether a scalar quantity of interest lies above or below a fixed threshold. For each input, we use the surrogate when its conformal interval lies entirely on one side of the threshold and route the input to simulation when the interval intersects it. Standard conformal prediction constructs intervals without reference to the downstream decision threshold: even a narrow interval near the threshold can cross it and trigger simulation, whereas a wider interval farther away can remain entirely on one side and require no simulation. We introduce Threshold-Aware Con
    
[^231]: SD-DPC：稀疏字典可微预测控制

    SD-DPC: Sparse Dictionary Differentiable Predictive Control

    [https://arxiv.org/abs/2610.02466](https://arxiv.org/abs/2610.02466)

    提出稀疏字典可微预测控制（SD-DPC）框架，将基于SINDy辨识的预测模型与可微预测控制相结合，从数据中学习出仅含少量项、可解释的显式反馈控制律，在满足约束的同时性能比蒸馏策略提升高达一个数量级，且内存和在线计算开销降低几个数量级。

    

    我们提出了稀疏字典可微预测控制（SD-DPC），这是一个从数据中为非线性系统学习稀疏、可解释反馈策略的框架。首先，通过基于展开（rollout）的稀疏非线性动力学辨识方法（SINDy）来建立预测模型，该方法建立在基于梯度和多步公式化的基础之上。随后，将控制策略参数化为字典函数的稀疏组合，并通过对该模型微分一个带约束的有限时域预测控制目标函数进行训练，使其各项依据闭环性能而非模仿先前训练好的控制器来选择。最终得到一个仅含少数几项的显式反馈控制律。在三个基准控制问题上，SD-DPC 在所有测试场景中均满足约束条件，相比蒸馏到相同字典项上的策略，性能提升高达一个数量级，且所需内存和在线计算量低几个数量级。

    arXiv:2610.02466v1 Announce Type: cross  Abstract: We present sparse dictionary differentiable predictive control (SD-DPC), a framework for learning sparse, interpretable feedback policies for nonlinear systems from data. A prediction model is first identified by rollout-based sparse identification of nonlinear dynamics (SINDy), building on gradient-based and multistep formulations. The policy is then parameterized as a sparse combination of dictionary functions and trained by differentiating a constrained finite-horizon predictive-control objective through this model, so that its terms are selected by closed-loop performance rather than by imitating a previously trained controller. The result is an explicit feedback law with only a handful of terms. Across three benchmark control problems, SD-DPC satisfies the constraints in all test scenarios, outperforms a policy distilled onto the same terms by up to an order of magnitude, and requires orders of magnitude less memory and online com
    
[^232]: 大语言模型压缩的能力缩减定律

    Capability Scaling-Down Laws for LLM Compression

    [https://arxiv.org/abs/2610.02462](https://arxiv.org/abs/2610.02462)

    该论文系统研究了大语言模型在剪枝、量化和蒸馏压缩下的能力缩减定律，建立了可预测不同压缩配置所导致能力损失的简单关系式，从而显著减少压缩实验所需的测量成本。

    

    大语言模型压缩可以降低推理成本和内存需求，但如何选择压缩方法与配置在很大程度上仍依赖经验，因为相近的资源削减可能造成不同的能力损失。我们系统地研究了大语言模型压缩在剪枝、量化和蒸馏三种方式下的能力缩减定律。我们的框架在数学、代码生成和问答任务上度量能力损失，并将这些度量与模型规模、训练阶段、压缩设置、数据可用性以及训练曝光程度相关联。我们建立了简单的预测关系，并评估其准确性、测量效率以及对未见配置和模型状态的泛化能力。在剪枝层级之间共享密度响应可使拟合剪枝预测器所需的配置测量数量减半：在新的 Pythia 模型状态、预先注册的 OLMo-2 测试状态以及 Wanda 剪枝下，该紧凑关系均能良好匹配（摘要原文在此处截断）。

    arXiv:2610.02462v1 Announce Type: cross  Abstract: LLM compression reduces inference costs and memory requirements, but selecting a method and configuration remains largely empirical because comparable resource reductions can produce different capability losses. We systematically investigate capability scaling-down laws for LLM compression across pruning, quantization, and distillation. Our framework measures capability loss in mathematics, code generation, and question answering, and relates these measurements to model size, training stage, compression settings, data availability, and training exposure. We develop simple predictive relations and evaluate their accuracy, measurement efficiency, and generalization to unseen configurations and model states. Sharing the density response across pruning levels halves the configuration measurements needed to fit a pruning predictor: on new Pythia states, on pre-registered OLMo-2 test states and under Wanda pruning, the compact relation match
    
[^233]: 一种用于3D-IC热建模的可组合AI加速迭代求解器

    A Composable AI-Accelerated Iterative Solver for 3D-IC Thermal Modeling

    [https://arxiv.org/abs/2610.02461](https://arxiv.org/abs/2610.02461)

    DAIST将3D-IC封装热仿真分解为块级子域并用神经算子替代子域求解器，通过界面温度与热通量的迭代交换实现耦合，从而构建出无需重新训练即可复用于不同拓扑封装的可组合AI加速热求解器。

    

    对异构2.5D/3D-IC封装进行准确的热分析至关重要，但计算代价极其高昂。单次全封装FEM仿真可能耗时数小时，而基于AI的代理模型将整个堆叠视为单一的预测目标，每当芯片数量或拓扑结构发生变化时都必须重新训练。为解决这一局限，本工作提出了面向热分析的域分解AI加速迭代求解器（DAIST），这是一种可组合的热求解器，它将全局封装仿真分解为块级子域问题，用神经算子替代子域求解器，并通过界面温度与热通量的迭代交换将各子域耦合起来。这种从局部到全局的架构消除了单体模型的拓扑锁定问题：块级神经算子可以在未见过的封装组装中直接复用而无需重新训练。该迭代耦合策略还进一步提供了可控的精度-运行时间权衡。

    arXiv:2610.02461v1 Announce Type: new  Abstract: Accurate thermal analysis of heterogeneous 2.5D/3D-IC packages is essential yet computationally prohibitive. A single full-package FEM simulation can take hours, while AI-based surrogates treat the entire stack as a monolithic prediction target and must be retrained whenever the die count or topology changes. To address this limitation, this work proposes Domain-Decomposed AI-Accelerated Iterative Solver for Thermal Analysis (DAIST), a composable thermal solver that decomposes the global package simulation into block-level subdomain problems, replaces subdomain solvers with neural operators, and couples them through iterative exchanges of interfacial temperature and heat flux. This local-to-global architecture eliminates the topology lock-in of monolithic models: block-level neural operators can be directly reused in unseen package assemblies without retraining. The iterative coupling strategy further provides a controllable accuracy-run
    
[^234]: AI驱动的热感知数据中心容量规划

    AI-driven Thermal-aware Data Center Capacity Planning

    [https://arxiv.org/abs/2610.02442](https://arxiv.org/abs/2610.02442)

    本论文提出一个AI驱动的数据中心热感知容量规划框架，其内嵌模型通过学习机架功率、服务器放置和HVAC设置等关键参数，可在毫秒级完成温度预测，从而在几秒内实现对真实数据中心的热感知容量规划。

    

    大语言模型（LLM）的兴起给数据中心的热管理带来了重大挑战。LLM所需的高强度GPU计算会导致局部热点。此外，训练和推理爆发期间的热负载激增使得实时冷却响应更难以预测和控制。数据中心的热感知容量规划通常需要大量昂贵的高保真CFD（计算流体力学）模拟。AI模型可以对未见过的设计进行实时预测，然而现有工作要么预测误差较大，要么对数据中心运行做了过度简化的假设。本工作提出了一个AI驱动的框架，可以在几秒钟内对真实世界的数据中心进行热感知容量规划。该框架内嵌的AI模型从众多关键参数（机架功率、服务器功率、服务器放置方式、HVAC设置等）中学习，并能在毫秒级提供温度预测。该AI模型经测试（原文摘要在此处被截断）

    arXiv:2610.02442v1 Announce Type: new  Abstract: The emerging of large language models (LLMs) has posed significant challenges to the thermal management of data center. Intense GPU computation for LLMs results in localized hotspots. Moreover, spiking thermal loads during training and inference bursts make real-time cooling response more difficult to predict and control. Thermal-aware capacity planning of data center requires massive expensive high-fidelity CFD simulations. AI models can perform real-time prediction for unseen designs. However, existing works either have large prediction error, or have over-simplified assumptions for data center operations. This work presents an AI-driven framework that can perform thermal-aware capacity planning for a real-world data center in seconds. The embedded AI model learns from numerous key parameters (rack power, server power, server placement, HVAC settings etc.), and provides temperature prediction within milliseconds. This AI model is teste
    
[^235]: 基于加性量化表示的赌博机算法

    Bandits via Additive Quantized Representations

    [https://arxiv.org/abs/2610.02440](https://arxiv.org/abs/2610.02440)

    该论文提出将残差量化（RQ）作为表示层，把连续上下文映射为多层级离散码本分配，从而实现内存严格受限的非线性上下文赌博机算法，在13个数据集中的11个上优于非RQ基线，并以最多1000倍更少的内存匹敌需要定期重训的XGBoost和神经网络基线。

    

    上下文赌博机需要在非线性奖励建模与在线效率之间取得平衡。树集成和神经网络方法能够捕捉非线性，但需要周期性重新训练和大型回放缓冲区。线性模型能够以O(1)内存对每次观测进行高效更新，但从根本上受限于线性奖励结构。我们提出残差量化（RQ）作为一种表示层来弥合这一差距。离线训练的RQ码本将连续上下文映射为跨多个层级的离散质心分配，并通过影子机制动态设置。这使得一系列加性赌博机算法能够在严格受限的内存下实现非线性表达能力。在13个数据集上，RQ变体在11个数据集上击败了对应的非RQ方法，且往往优势显著，同时以最多节省1000倍的内存，达到与需要定期加倍重训的XGBoost和神经基线相当的性能。

    arXiv:2610.02440v1 Announce Type: new  Abstract: Contextual bandits require balancing nonlinear reward modeling with online efficiency. Tree ensembles and neural methods capture nonlinearities but require periodic retraining and large replay buffers. Linear models update efficiently per observation with O(1) memory, but are fundamentally restricted to linear reward structures. We propose Residual Quantization (RQ) as a representation layer to bridge this gap. An offline-trained RQ codebook maps continuous contexts into discrete centroid assignments across multiple levels, set dynamically through a shadow mechanism. This enables a spectrum of additive bandit algorithms that achieve nonlinear expressivity with strictly bounded memory. Across 13 datasets, RQ variants beat their non-RQ counterparts on 11 of 13 datasets, often by wide margins, while matching doubling-retrain XGBoost and neural baselines using up to 1000 times less memory.
    
[^236]: 使用图函数与神经逆算子的复杂网络生成模型

    A Generative Model of Complex Networks Using Graphons and Neural Inverse Operators

    [https://arxiv.org/abs/2610.02439](https://arxiv.org/abs/2610.02439)

    该论文提出基于多分形阶梯图函数的复杂网络生成模型，通过神经逆算子在函数空间中恢复参数，实现了兼具可解释性与摊销式推理、并能泛化到训练中未见图规模的新型生成框架。

    

    生成式图模型对于理解和模拟复杂网络至关重要。然而，现有方法各有互补的优势与局限：机制模型具有良好的可解释性，但依赖于针对具体实例的估计方法；深度生成模型则提供摊销式推理，但以牺牲可解释性为代价，且在很大程度上受限于训练时见过的图规模。科学应用的发展需要一个能够兼具两种范式优点的框架。我们通过在函数空间中同时构建生成模型和参数恢复方法来弥合二者之间的鸿沟。具体而言，多分形阶梯图函数通过一种递归构造方式扩展了标准阶梯图函数，从而以紧凑的方式参数化复杂网络。这种表述方式允许使用神经逆算子来恢复模型参数，进而实现对未见过的图规模进行推理。我们在仅由合成的多分形阶梯图函数实现所训练的模型上进行了评估……

    arXiv:2610.02439v1 Announce Type: new  Abstract: Generative graph models are central to understanding and simulating complex networks. However, existing approaches have complementary strengths and limitations. Mechanistic models offer interpretability but rely on instance-specific estimation methods. Deep generative models, on the other hand, offer amortized inference at the cost of interpretability and are largely limited to graph sizes seen during training. Scientific applications motivate a framework that retains the strengths of both paradigms. We bridge them by formulating both the generative model and parameter recovery in function space. A multifractal step graphon extends standard step graphons with a recursive construction that compactly parameterizes complex networks. This formulation admits a neural inverse operator to recover its parameters, enabling inference on unseen graph sizes. We evaluate our model, trained only on synthetic multifractal step graphon realizations, aga
    
[^237]: 你是在合成还是在回忆？评估大语言模型在算法代码检索上的表现

    Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval

    [https://arxiv.org/abs/2610.02438](https://arxiv.org/abs/2610.02438)

    该论文提出将大语言模型对知名算法的代码生成重新定义为“参数化代码检索”任务，并引入AlgoREval基准（涵盖599个问题、77个经典算法、7种编程语言和4种图输入表示）来独立评估这一能力，发现不同语言和输入表示之间的检索准确率差异显著。

    

    大语言模型（LLMs）在代码生成方面已展现出强大的性能，其成功既依赖于回忆相关的算法知识，也依赖于推理如何应用这些知识。然而，现有的LLM处理流程是不透明的，没有对这两个组成部分进行显式区分。我们认为，对于其规范实现在预训练语料库中广泛可获取的知名算法而言，代码生成更适合被衡量为“参数化代码检索”（parametric code retrieval）：即从内化知识中复现一个被指定名称的算法，而非合成一个全新的算法。我们引入了AlgoREval，一个包含599个问题的基准，涵盖14个领域中的77个经典算法、7种编程语言和4种图输入表示，以在隔离环境中评估这一能力，并在零样本设置下评估了15个模型（7B–34B参数）。我们发现，不同语言和输入表示之间的检索准确率存在显著差异。

    arXiv:2610.02438v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated strong performance in code generation, where success depends on both recalling relevant algorithmic knowledge and reasoning about how to apply it. However, existing LLM pipelines are opaque, with no explicit separation between these two components. We argue that for well-known algorithms whose canonical implementations are widely accessible in pretraining corpora, code generation is better measured as \textit{parametric code retrieval}: reproducing a named algorithm from internalised knowledge rather than synthesizing a novel one. We introduce AlgoREval, a benchmark of 599 problems spanning classical 77 algorithms across 14 domains, 7 programming languages, and 4 graph-input representations to evaluate this capability in isolation, and assess 15 models (7B--34B parameters) in a zero-shot setting. We find substantial variation in retrieval accuracy across languages and input representations
    
[^238]: 学习风格，遗忘语义：SFT与RFT在分类任务上的案例研究

    Learning Style, Forgetting Semantics: A Case Study of SFT and RFT on Classification Tasks

    [https://arxiv.org/abs/2610.02437](https://arxiv.org/abs/2610.02437)

    本文通过将策略更新精确分解为语义与风格两个成分，揭示了SFT比RFT遗忘更多的原因——SFT会沿教师风格偏好产生离轴风格漂移从而破坏语义记忆，而RFT能保持类内风格对称性。

    

    为什么即使所有教师演示在语义上都是正确的，监督微调（SFT）仍比强化微调（RFT）导致更多的遗忘？我们在分类任务上研究这个问题，其中每个语义类别内的标记以不同风格表达相同的语义答案。这些任务共享潜在的语义规则，但在提示分布和教师的风格偏好上有所不同。利用一个易于处理的线性softmax策略，我们推导出策略更新在语义成分和风格成分上的精确分解。我们证明，在相同策略和提示下，SFT和RFT具有平行的语义更新，但风格动态不同。从一个没有任何类内风格偏好的策略出发，采用精确策略梯度的RFT能保持这种对称性，而使用非均匀教师的SFT在群体更新下会沿着非零任务均值产生离轴风格漂移。我们利用这种漂移建立了一个……

    arXiv:2610.02437v1 Announce Type: cross  Abstract: Why does supervised fine-tuning (SFT) lead to more forgetting than reinforcement fine-tuning (RFT), even when all teacher demonstrations are semantically correct? We study this question on classification tasks where tokens within each semantic class express the same semantic answer in different styles. The tasks share an underlying semantic rule but differ in their prompt distributions and teachers' stylistic preferences. Using a tractable linear-softmax policy, we derive an exact decomposition of the updates into semantic and style components. We show that, at a common policy and prompt, SFT and RFT have parallel semantic updates but differ in their style dynamics. Starting from a policy with no within-class style preference, RFT with exact policy gradients preserves this symmetry, whereas SFT with a nonuniform teacher develops off-axis style drift along a nonzero task mean under population updates. We use this drift to establish a se
    
[^239]: 评估与提升大语言模型对输入序列变化的鲁棒性

    Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations

    [https://arxiv.org/abs/2610.02432](https://arxiv.org/abs/2610.02432)

    本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。

    

    生产系统中的大语言模型（LLM）面临提示注入、木马（后门）攻击以及自动质量指标被操纵等威胁。本论文开发了用于评估和提升大语言模型对对抗性输入序列变化鲁棒性的模型、方法和算法。我们提出了R_stab(f)，一种基于小输入扰动下逐步输出分布之间Jensen-Shannon散度的生成式鲁棒性度量。对于局部化攻击，我们证明了V(h) <= 1 - R_class(h)，其中R_class(h)是决策算子h在小扰动下保持其决策的概率。对于非局部化攻击，我们提出了一个经过校准的经验模型。针对LLM-as-a-Judge（大模型作裁判）系统，我们开发了ASA，一种自适应进化黑盒攻击，其攻击成功率（ASR）最高可达73.8%，在开源模型之间的迁移攻击成功率最高可达62.6%。在Trojan Detection Challenge 2023数据（Pythia-1.4B）上，代理触发器达到REA（摘要在此处被截断）

    arXiv:2610.02432v1 Announce Type: cross  Abstract: Large language models (LLMs) in production systems face prompt injections, trojans (backdoors), and manipulation of automatic quality metrics. This thesis develops models, methods, and algorithms for evaluating and improving LLM robustness to adversarial input sequence variations. We propose R_stab(f), a generative robustness metric based on the Jensen-Shannon divergence between per-step output distributions under small input perturbations. For localized attacks we prove V(h) <= 1 - R_class(h), where R_class(h) is the probability that a decision operator h keeps its decision under small perturbations. For non-localized attacks we propose a calibrated empirical model. For LLM-as-a-Judge systems we develop ASA, an adaptive evolutionary black-box attack that reaches an attack success rate (ASR) of up to 73.8%, with transfer between open models up to 62.6%. On Trojan Detection Challenge 2023 data (Pythia-1.4B), surrogate triggers reach REA
    
[^240]: CRISP：一种基于子句重构的可解释神经符号命题框架

    CRISP: A Framework for Clause-Reconstructed Interpretable NeuroSymbolic Propositions

    [https://arxiv.org/abs/2610.02431](https://arxiv.org/abs/2610.02431)

    CRISP框架通过Tsetlin机器子句重构二元神经网络教师模型的最后一层激活向量，建立了从输入特征到隐藏神经元的显式符号化决策追溯，显著提升了深度神经网络的可解释性与可审计性。

    

    深度神经网络通过分层数值变换实现了高准确率，但其决策仍难以审计，因为决策证据被编码在隐藏激活中而非显式规则里。本文提出了CRISP框架，该框架将二元神经网络教师模型的最后一层激活向量（LLAV）重构为Tsetlin机器（TM）子句。CRISP对教师模型的倒数第二层预logit激活进行符号二值化，并为每个LLAV神经元分配一个独立的Tsetlin机器（ITM）。每个重构的隐藏位由布尔化输入特征上的命题子句表示，从而提供了从命名输入阈值到命名教师神经元的直接符号追溯路径。CRISP在MNIST、KMNIST、FashionMNIST（FMNIST）、SVHN和CIFAR10数据集上，分别使用BinaryConnect卷积神经网络（BCCNN）教师模型和全二元神经网络（BNN）教师模型进行评估，并对二元阈值化和温度计编码进行了附加研究。

    arXiv:2610.02431v1 Announce Type: new  Abstract: Deep neural networks achieve high accuracy through layered numerical transformations, yet their decisions remain difficult to audit because decision evidence is encoded in hidden activations rather than explicit rules. This paper introduces CRISP, a framework that reconstructs the last-layer activation vector (LLAV) of binary neural teachers as Tsetlin Machine (TM) clauses. CRISP sign-binarizes the teacher's penultimate pre-logit activations, and assigns one Individual TM (ITM) to each LLAV neuron. Each reconstructed hidden bit is represented by propositional clauses over Booleanized input features, which gives a direct symbolic trace from named input thresholds to a named teacher neuron. CRISP is evaluated on MNIST, KMNIST, FashionMNIST (FMNIST), SVHN, and CIFAR10 using a BinaryConnect convolutional neural network (BCCNN) teacher and a fully binary neural network (BNN) teacher, with an additional study on binary thresholding, thermomete
    
[^241]: 面向流映射蒸馏的几何感知时间重参数化

    Geometry-Aware Time Reparameterization for Flow-Map Distillation

    [https://arxiv.org/abs/2610.02427](https://arxiv.org/abs/2610.02427)

    提出一种几何感知的时间重参数化方法，为学生模型在法向加速度大的轨迹区域分配更多蒸馏时间，在保持教师几何路径和终端分布的同时使流映射蒸馏更易学习。

    

    流映射蒸馏通过学习预训练生成式ODE的有限时间转移，实现一步和少步生成。我们研究改变教师模型的时间参数化是否能使这些转移更容易学习。基于“法向加速度较大的轨迹片段更难蒸馏”这一假设，我们提出一种几何感知的时间重参数化方法，为学生模型在这些区域分配更多时间，同时保持教师模型的几何路径和终端分布。在适当假设下，我们推导出一个共享时钟，使总体法向加速度统计量均匀化，并基于跨教师轨迹的鲁棒、正则化估计构建了实用的近似方法。我们将该时钟纳入拉格朗日流映射蒸馏中，使用变换后的时间坐标来条件化学生模型。该时钟在蒸馏前仅需估计一次，且无需重新训练教师模型。

    arXiv:2610.02427v1 Announce Type: cross  Abstract: Flow-map distillation enables one- and few-step generation by learning finite-time transitions of a pretrained generative ODE. We investigate whether changing the teacher's time parameterization can make these transitions easier to learn. Motivated by the hypothesis that trajectory segments with large normal acceleration are harder to distill, we propose a geometry-aware time reparameterization that allocates more student time to these regions while preserving the teacher's geometric paths and terminal distribution. We derive a shared clock that equalizes a population normal-acceleration statistic under suitable assumptions, and construct a practical approximation from robust, regularized estimates across teacher trajectories. We incorporate this clock into Lagrangian flow-map distillation, using the transformed time coordinate to condition the student. The clock is estimated once before distillation and requires neither teacher retrai
    
[^242]: AI理论学家揭示α-RuCl₃中的激子结构

    The AI Theorist reveals excitonic structure in $\alpha$-RuCl$_3$

    [https://arxiv.org/abs/2610.02417](https://arxiv.org/abs/2610.02417)

    AI Theorist系统通过假设生成、第一性原理计算和证据驱动的迭代完善实现了物理模型的自主发现，并首次在Kitaev量子自旋液体候选材料α-RuCl₃中识别出具有不同光学选择定则和实空间分布的激子态。

    

    实验仪器与自动化技术的进步产生了日益丰富的数据集，但将实验观测转化为微观层面的理解仍是科学发现中的瓶颈。为加速这一过程，我们提出了AI Theorist（AI理论学家）——一个由人工智能（AI）智能体组成的系统，可通过假设生成、第一性原理计算和证据驱动的迭代完善来实现物理模型的自主发现。我们将该框架应用于α-RuCl₃——实现Kitaev量子自旋液体的重要候选材料——通过光学光谱研究其电子结构。AI Theorist对光学和光电流观测结果提出了新的解释，识别出具有不同光学选择定则和实空间分布的多个不同激子态。据我们所知，这是首次演示AI系统自主构建物理模型来解释先前的实验观测……

    arXiv:2610.02417v1 Announce Type: new  Abstract: Advances in experimental instrumentation and automation generate increasingly rich datasets, but turning experimental observations into microscopic understanding remains a bottleneck in scientific discovery. To accelerate this process, we introduce AI Theorist, a system of artificial intelligence (AI) agents for autonomous discovery of physical models through hypothesis generation, first-principles calculations and evidence-driven refinement. We apply the framework to $\alpha$-RuCl$_3$, a leading candidate material for realizing a Kitaev quantum spin liquid, to investigate its electronic structure through optical spectra. AI Theorist develops a new interpretation of the optical and photocurrent observations, identifying distinct excitonic states with contrasting optical selection rules and real-space distributions. To our knowledge, this is the first demonstration of an AI system autonomously developing a physical model to explain previo
    
[^243]: 被提示去判别：恶意输入探针在真实环境中的泛化能力

    Prompted to Discriminate: Generalizing Malicious-Input Probes in the Wild

    [https://arxiv.org/abs/2610.02413](https://arxiv.org/abs/2610.02413)

    本研究系统评估了在用户输入后附加分类指令后缀能否增强激活探针对恶意输入的检测能力，并利用严格的留一数据集外（LODO）评估，在 13 个安全基准和三个开源模型家族上检验了此类提示后缀在未见攻击类型上的泛化表现。

    

    LLM 智能体日益依赖激活探针作为运行时监控器，用于检测提示注入、越狱攻击和不安全请求——通过读取模型自身的隐藏状态，在智能体执行有害输入之前将其捕获。一种廉价且日益普遍的做法是借鉴“LLM 作为裁判”的提示方法，在用户回合之后附加一段简短的分类指令，并在该位置读取探针以增强其效果：该指令要求模型将传入的请求表征为某个类别，从而集中探针需要区分的信号，且服务成本几乎可以忽略不计。但这段后缀的具体措辞是否重要？其收益在真实环境中是否依然成立——即在探针训练中从未见过的攻击类型上（这正是已部署监控器实际面临的情况）？我们通过一个受控的后缀阶梯实验对此进行测试，在严格的留一数据集外（LODO）评估下，覆盖 13 个安全基准（越狱、注入和良性对话）以及三个开源权重模型家族……

    arXiv:2610.02413v1 Announce Type: new  Abstract: LLM agents increasingly rely on activation probes as runtime monitors for prompt injection, jailbreaks, and unsafe requests, reading the model's own hidden state to catch a harmful input before the agent acts on it. A cheap, increasingly common move, borrowed from LLM-as-judge prompting, is to append a short classification instruction after the user's turn and read the probe at that point, to sharpen it: the instruction asks the model to represent the incoming request as a class, concentrating the signal the probe must separate, at negligible serving cost. But does the wording of that suffix matter, and does its benefit hold in the wild, on attack types the probe never saw in training, the regime a deployed monitor faces? We test this with a controlled ladder of post-user suffixes under strict leave-one-dataset-out (LODO) evaluation across 13 safety benchmarks (jailbreak, injection, and benign chat) and three open-weight model families (
    
[^244]: 通过自适应覆盖与聚焦采样实现高效神经场学习

    Efficient Neural Field Learning via Adaptive Coverage and Focused Sampling

    [https://arxiv.org/abs/2610.02410](https://arxiv.org/abs/2610.02410)

    提出ACES采样框架，通过解耦覆盖与重要性——利用自适应空间分区保证域覆盖、采用区域级重要性加权聚焦关键区域——从而降低梯度方差并显著提升隐式神经表示的训练效率。

    

    隐式神经表示（INRs）为建模高维连续场提供了灵活的框架，但由于均匀子采样忽略了空间异质性，其训练往往效率低下。现有的自适应采样方法通过优先处理高误差样本在一定程度上解决了这一问题，但它们通常在点级别上运作，容易导致局部区域的冗余采样以及对整个域的覆盖不足。我们提出了ACES（自适应覆盖感知高效采样），这是一种结构化采样框架，通过将覆盖与重要性解耦来提高训练效率。ACES构建自适应空间分区以确保域覆盖并减少冗余，并在训练过程中应用区域级重要性加权以优先处理信息量大的区域。我们提供了理论分析，表明自适应分区通过增加区域内的同质性来降低梯度方差。

    arXiv:2610.02410v1 Announce Type: cross  Abstract: Implicit neural representations (INRs) provide a flexible framework for modeling high-dimensional continuous fields, but their training is often inefficient due to uniform subsampling that ignores spatial heterogeneity. Existing adaptive sampling methods partially address this issue by prioritizing high-error samples, but typically operate at the point level, often leading to redundant sampling in localized regions and insufficient coverage of the domain. We propose ACES (Adaptive Coverage-aware Efficient Sampling), a structured sampling framework that improves training efficiency by decoupling coverage and importance. ACES constructs adaptive spatial partitions to ensure domain coverage and reduce redundancy, and applies region-level importance weighting to prioritize informative regions during training. We provide a theoretical analysis showing that adaptive partitioning reduces gradient variance by increasing within-region homogenei
    
[^245]: 可训练的智能体式上下文管理

    Trained Agentic Context Management

    [https://arxiv.org/abs/2610.02404](https://arxiv.org/abs/2610.02404)

    通过在最简化的智能体框架（自我调用工具与上下文读取工具）上微调小模型，模型仅用8K词元上下文即可在长文档基准上媲美拥有1M词元上下文的GPT-5.4。

    

    我们研究长上下文语言模型。不同于原生训练长上下文能力或设计复杂的长上下文框架，我们在最简单的智能体框架上训练模型：该框架仅包含两个工具——一个可以用任意指定提示词调用自身的工具，以及一个可以读取输入上下文中指定范围内词元的工具。我们利用这一框架在多样化的合成数据集上对Qwen3.6-35B-A3B进行了微调。在仅有8,000词元上下文的条件下，当文档长度超过40K词元时，我们的小模型在OOLONG-synth基准测试上的表现与拥有1M词元上下文的GPT-5.4相当。

    arXiv:2610.02404v1 Announce Type: cross  Abstract: We study long context language models. Instead of training long context natively, or designing a long context harness, we train a model over the simplest possible harness: a tool to call itself with any specified prompt and a tool to read tokens in a range from the input context. We finetune Qwen3.6-35B-A3B on a diverse synthetic dataset using this harness. With only 8,000 tokens of context, our small model is as strong as GPT-5.4 with 1M tokens of context on the OOLONG-synth benchmark when document length exceeds 40K tokens.
    
[^246]: 5G及未来网络中的节能：一种量子强化学习方法

    Energy Saving in 5G and Beyond Networks: A Quantum Reinforcement Learning Approach

    [https://arxiv.org/abs/2610.02403](https://arxiv.org/abs/2610.02403)

    本文针对5G及未来网络中的基站能耗优化问题，提出量子强化学习方法，以克服传统深度强化学习训练负担重、状态与动作空间指数级增长的难题。

    

    节能已成为5G及未来网络面临的关键挑战。联网设备的快速增长推高了整个网络的能源需求，使运营支出攀升至不可持续的高度。基站（BS）在能耗中占据最大份额，通常消耗无线接入网（RAN）总能耗的60-70%左右。因此，为解决这一问题，本文在考虑用户设备（UE）动态行为的同时，对基站的能源使用进行优化。深度强化学习（DRL）是制定有效节能策略的天然候选方案，例如在用户密度较低时自动开启或关闭基站，或调整发射功率以平衡能效与服务质量（QoS）。然而，其沉重的训练负担以及密集5G环境中状态空间和动作空间的指数级增长，使得探索变得越来越困难。为克服（该摘要至此被截断）

    arXiv:2610.02403v1 Announce Type: cross  Abstract: Energy saving has become a critical challenge in 5G and beyond networks. The rapid growth of connected devices has increased the overall network energy demand, driving operational expenditure to unsustainable heights. The Base Station (BS) accounts for the largest share of energy usage, typically consuming around 60-70\% of the Radio Access Network (RAN)'s total energy. Therefore, to address this issue, this article optimizes the BS's energy usage while accounting for the dynamic behavior of User Equipment (UE). Deep Reinforcement Learning (DRL) is a natural candidate for determining effective energy saving policies, such as automatically switching BSs on or off when user density is low or adjusting transmission power to balance energy efficiency and Quality of Service (QoS). However, its heavy training burden and the exponential growth of state and action spaces in dense 5G environments make exploration increasingly difficult. To over
    
[^247]: VisAudit：评估多模态智能体的视觉诊断与修复能力

    VisAudit: Evaluating Multimodal Agents for Visual Diagnosis and Repair

    [https://arxiv.org/abs/2610.02399](https://arxiv.org/abs/2610.02399)

    本文提出VisAudit基准，用于评估多模态智能体在数据可视化任务中的自主诊断、修复与验证能力，弥补了现有基准仅能评估单一预定义能力的不足。

    

    多模态智能体越来越多地被应用于数据可视化任务，但在自主审查方面仍然存在局限。与人类不同，它们可能无法识别可视化何时出现错误，无法确定需要修改什么，无法在不破坏正确内容的前提下进行修复，也无法验证干预是否成功。现有基准主要评估预定义的单一能力，例如图表生成、指令引导的编辑或缺陷检测，因此无法捕捉自主审查方面的这一差距。我们提出了VisAudit，一个用于评估可视化诊断、修复与验证能力的基准。给定一张渲染好的图表和可配置的辅助证据（包括其源数据表、预期文本摘要和可视化代码），智能体需要迭代地诊断潜在缺陷、修改并执行可视化代码、检查执行结果与视觉反馈，并判断何时无需进一步干预。

    arXiv:2610.02399v1 Announce Type: new  Abstract: Multimodal agents are increasingly used for data visualization tasks but remain limited in autonomous review. Unlike humans, they may fail to recognize when a visualization is incorrect, determine what to change, repair it without disrupting correct content, and verify whether the intervention succeeded. Existing benchmarks largely evaluate predefined individual capabilities such as chart generation, instruction-guided editing, or defect detection, and therefore do not capture this gap in autonomous review. We introduce VisAudit, a benchmark for evaluating visualization diagnosis, repair, and verification. Given a rendered chart and configurable auxiliary evidence, including its source data table, intended text summary, and visualization code, an agent iteratively diagnoses potential defects, modifies and executes visualization code, inspects execution and visual feedback, and determines when no further intervention is needed. VisAudit d
    
[^248]: 面向美国建筑电表数据AI需求预测的验证式数据导入：设计、受控评估与一个经过修正的否定结果

    Validated Data Onboarding for AI Demand Forecasting on U.S. Building Meter Data: Design, Controlled Evaluation, and a Corrected Negative Result

    [https://arxiv.org/abs/2610.02397](https://arxiv.org/abs/2610.02397)

    本文提出一种在模型训练前检测并修复建筑电表数据缺陷的数据导入管道（仅使用预测时可得信息），受控实验表明仅0.10%的训练数据缺陷即可使预测误差增加86%，而该管道的修复可将误差恢复至干净数据水平。

    

    电力公司和电网运营商日益依赖机器学习模型来预测次日用电需求，而这些模型所学习的电表数据通常存在缺陷：读数缺失、传感器冻结、建筑物读数连续数小时为零、计量单位偏差达100倍。本报告提出了一种数据导入管道，在模型训练之前检测并修复此类缺陷，且仅使用预测时点可获得的信息，并通过一项受控实验来衡量该管道能否保护提前24小时的预测。在来自公开数据集Building Data Genome 2的十二栋美国建筑的逐小时用电数据上（210,528行，2016-2017年），人为植入并经哈希记录的缺陷影响了训练期的0.10%，使梯度提升预测器的误差增加了86%；经过检测和仅使用历史信息的修复后，误差恢复到干净数据的水平（平均绝对标度误差：干净数据为0.760，受损数据为1.415，修复后为0.729）。

    arXiv:2610.02397v1 Announce Type: new  Abstract: Electric utilities and grid operators increasingly rely on machine-learning models to forecast next-day demand, and those models learn from meter data that is routinely defective: readings go missing, sensors freeze, buildings read zero for hours, and units change by a factor of 100. This report presents a data-onboarding pipeline that detects and repairs such defects before a model is trained, using only information available at forecast time, and a controlled experiment that measures whether the pipeline protects a 24-hour-ahead forecast. On hourly electricity data for twelve U.S. buildings from the public Building Data Genome 2 dataset (210,528 rows, 2016-2017), seeded, hash-logged defects touching 0.10% of the training period raised the error of a gradient-boosting forecaster by 86%; after detection and past-only repair the error returned to the clean-data level (mean absolute scaled error 0.760 clean, 1.415 corrupted, 0.729 repaired
    
[^249]: Inherit-MAS：通过工作流与执行继承实现多智能体系统的测试时演化

    Inherit-MAS: Test-Time Evolution of Multi-Agent Systems through Workflow and Execution Inheritance

    [https://arxiv.org/abs/2610.02396](https://arxiv.org/abs/2610.02396)

    该论文提出Inherit-MAS框架，借鉴生物进化中遗传与选择的机制，在工作流和执行两个层面实现显式继承，使基于大语言模型的多智能体系统能够在测试时高效演化工作流，同时避免破坏有用组件和产生冗余计算。

    

    基于大语言模型构建的多智能体系统（MAS）通过协调专业化智能体来解决复杂任务，但有效的工作流难以预先设计。测试时演化利用执行反馈来改进工作流，然而大范围的修改可能会破坏有用的组件，而重新执行未改变的请求则会产生冗余计算。受生物进化中遗传与选择相互作用的启发，我们提出了Inherit-MAS，它在工作流和执行两个层面将继承机制显式化。元模型首先综合生成一个由工作者智能体组成的工作流，这些智能体具有声明的角色、通信输入和工具权限，并由一个单独提示的评判者对每个已执行的候选方案进行评分并诊断其缺陷。在常规的改进轮次中，工作流继承从最新完成的候选方案出发，可以丢弃被判定为无用的可移除节点，并应用经过验证的编辑来解决相应问题……

    arXiv:2610.02396v1 Announce Type: cross  Abstract: Multi-agent systems (MAS) built from large language models coordinate specialized agents to tackle complex tasks, but effective workflows are difficult to design in advance. Test-time evolution refines workflows using execution feedback, yet broad revisions can disturb useful components, while re-executing unchanged requests can incur redundant computation. Inspired by the interplay of inheritance and selection in biological evolution, we introduce Inherit-MAS, which makes inheritance explicit at the workflow and execution levels. A meta-model first synthesizes a workflow of worker agents with declared roles, communication inputs, and tool permissions, and a separately prompted judge scores each executed candidate and diagnoses its deficiencies. In ordinary refinement rounds, \emph{workflow inheritance} starts from the latest completed candidate, may discard removable nodes judged unhelpful, and applies a validated edit to address the 
    
[^250]: 犹豫有其几何结构：用于稀疏激活引导的熵训练双曲探针

    Hesitation Has a Geometry: Entropy-Trained Hyperbolic Probes for Sparse Activation Steering

    [https://arxiv.org/abs/2610.02391](https://arxiv.org/abs/2610.02391)

    该论文提出双曲熵引导方法（HEST），以模型自身的下一个词元熵作为唯一标签训练双曲空间中的轻量探针，仅在模型“犹豫”的高熵词元处沿测地线对隐藏状态进行稀疏引导，从而更契合推理过程固有的树状层级结构。

    

    当大语言模型求解一个数学问题时，其推理过程在很大程度上是分层的，而解答往往在少数几个下一个词元熵较高的词元处发生分支。这种树状结构嵌入双曲空间所产生的失真远低于欧几里得空间。然而，现有的激活引导方法通常通过在每个词元处添加一个固定的欧几里得向量来编辑预训练模型的隐藏状态，尽管解答中的大多数词元其实已由上下文所确定。我们提出双曲熵引导（HEST），它利用一个轻量级探针将隐藏状态嵌入庞加莱球中，而该探针的唯一标签是模型自身的下一个词元熵。当该熵超过某个阈值时，HEST 会沿着探针读出值的最陡下降测地线移动嵌入状态，并将这一变化映射回隐藏状态。对于学习到的理想点的 Busemann 读出，我们证明了固定长度的步骤……（原文摘要在此处截断）

    arXiv:2610.02391v1 Announce Type: cross  Abstract: When a large language model solves a mathematical problem, its reasoning is largely hierarchical, and the solution often branches at a few tokens where the next-token entropy is high. Such tree-like structure embeds in hyperbolic space with far lower distortion than in Euclidean space. Activation steering, however, usually edits the hidden states of a pretrained model by adding one fixed Euclidean vector at every token, even though most tokens of a solution are already determined by the context. We propose Hyperbolic Entropy Steering (HEST), which embeds the hidden states in the Poincar\'e ball with a lightweight probe whose only label is the model's own next-token entropy. Where this entropy exceeds a threshold, HEST moves the embedded state along the geodesic of steepest descent of a readout of the probe and maps the change back to the hidden state. For the Busemann readout of a learned ideal point, we prove that a step of fixed leng
    
[^251]: 循环Transformer中共享内存的惊人有效性

    The Surprising Effectiveness of Shared Memory in Looped Transformers

    [https://arxiv.org/abs/2610.02383](https://arxiv.org/abs/2610.02383)

    提出让循环Transformer在预训练时共享内存（仅第一次递归写入键值缓存、后续递归读取并保留自身短窗口）的方法，不仅不损失质量反而提升质量，在减少76-79%上下文内存的同时刷新了循环模型的质量-内存边界。

    

    循环Transformer对每个token多次应用相同的层，在不增加参数的情况下通过更多计算来提升质量。然而，每次递归都会写入自己的键值缓存，因此内存仍随计算量增长。推理时技术可以缩小该缓存，但会以质量为代价。我们对循环语言模型进行预训练以共享内存：只有第一次递归写入缓存，后续递归读取该缓存，同时保留自己的一小段窗口。令人惊讶的是，我们发现共享内存不仅不损失质量，反而提升了质量。在1.5亿至10亿参数规模下，我们的循环预测Transformer（LPT）及其混合变体为循环模型树立了新的质量-内存边界：通过五次递归，混合变体在FineWeb-Edu数据集上将验证困惑度相比同等规模的标准Transformer降低了1.12-1.82，同时上下文内存使用减少76-79%。通过广泛的分析，我们研究了内存共享为何有帮助。共享内存与本地内存……

    arXiv:2610.02383v1 Announce Type: cross  Abstract: Looped Transformers apply the same layers several times per token, adding compute to improve quality without more parameters. Each recursion, however, writes its own key-value cache, so memory still grows with compute. Inference-time techniques can shrink this cache at a cost in quality. We pretrain looped language models to share memory: only the first recursion writes a cache, and later recursions read it while keeping a short window of their own. Surprisingly, we find that sharing memory does not cost quality and instead improves it. At 150M-1B parameters, our Looped Prediction Transformer (LPT) and its hybrid variant set a new quality-memory frontier for looped models: with five recursions, the hybrid lowers validation perplexity on FineWeb-Edu by 1.12-1.82 relative to a same-size standard Transformer while using 76-79% less context memory. Through an extensive analysis, we investigate why memory sharing helps. Shared and local mem
    
[^252]: Latent-MOPD：潜在多教师在线策略蒸馏

    Latent-MOPD: Latent Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2610.02381](https://arxiv.org/abs/2610.02381)

    提出Latent-MOPD，首个面向大语言模型的表示层级多教师在线策略蒸馏方法，无需额外教师训练即可同时利用专家模型的预测和隐藏状态进行知识传递，在全部九个基准测试上超越了仅token、仅表示和均匀平均等基线方法。

    

    在线策略蒸馏（OPD）通过学生模型自身生成的响应来训练学生模型。现有的LLM多教师OPD通过专家模型的输出分布来传递其预测内容。我们提出了Latent-MOPD，据我们所知，这是首个面向LLM的表示层级多教师OPD方法。该方法无需额外的教师训练，即可通过专家模型的预测及其用于计算预测的隐藏状态来整合现有专家模型。为了协调来自多个专家的表示监督，我们根据师生关系选择深层目标，使用共享投影来弥合不同的隐藏层宽度，并按领域对更新进行分组。每个教师的监督逐渐从隐藏状态转移到token预测，两个通道使用相同的路由专家。在我们的主要同族模型设置中，Latent-MOPD在所有九个基准测试上都优于仅使用token、仅使用表示以及均匀平均的基线方法。

    arXiv:2610.02381v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains a student on the responses it generates. Existing LLM multi-teacher OPD transfers what specialists predict through their output distributions. We introduce Latent-MOPD, to our knowledge the first representation-level multi-teacher OPD method for LLMs. It integrates existing specialists through both their predictions and the hidden states used to compute them, without additional teacher training. To coordinate representation supervision from multiple specialists, we select late-layer targets according to the teacher-student relationship, bridge unequal hidden widths with a shared projection, and group updates by domain. Each teacher's supervision gradually shifts from hidden states to token predictions, with both channels using the same routed specialist. In our main same-family setting, Latent-MOPD outperforms the token-only, representation-only and uniform-averaging baselines on all nine benchmarks ac
    
[^253]: 用于贝叶斯逆问题中快速后验采样的流匹配

    Flow Matching for Fast Posterior Sampling in Bayesian Inverse Problems

    [https://arxiv.org/abs/2610.02377](https://arxiv.org/abs/2610.02377)

    该论文针对具有函数值参数的PDE贝叶斯逆问题，系统评估了条件流匹配作为MCMC摊销式替代方法的表现，推导出近似后验在总变差距离和KL散度下的可计算精度估计，并提出一种通过Metropolization实现渐近精确的混合采样器。

    

    arXiv:2610.02377v1 公告类型：交叉 摘要：从后验分布中采样是计算贝叶斯逆问题的核心任务。贝叶斯推断中的标准主力方法——马尔可夫链蒙特卡洛（MCMC）——是串行的、产生相关样本，并且必须针对每个观测重新运行。条件流匹配提供了一种摊销式的替代方案：一个在参数与数据的联合样本上仅训练一次的传输映射，能够以可忽略的在线成本为任意观测生成独立的近似后验样本，且无需新的似然函数评估。我们对流匹配在具有函数值参数的基于偏微分方程（PDE）的逆问题中的应用进行了细致的、面向MCMC使用者的评估。利用流的可求解密度，我们推导了底层近似后验在总变差距离和Kullback-Leibler散度下的可计算精度估计，此外，我们提出了一种通过Metropolization实现渐近精确的混合采样器。我们验证了该精度估计（摘要在此处被截断）。

    arXiv:2610.02377v1 Announce Type: cross  Abstract: Sampling from the posterior is the central task of computational Bayesian inverse problems. The standard workhorse in Bayesian inference - Markov chain Monte Carlo (MCMC) - is sequential, yields correlated samples, and must be rerun for each observation. Conditional flow matching offers an amortized alternative: a transport map, trained once on joint samples of parameter and data, that yields independent approximate posterior samples for any observation at negligible online cost, without new likelihood evaluations. We give a careful, MCMC-literate assessment of flow matching for PDE-based inverse problems with function-valued parameters. Exploiting the flow's tractable density, we derive computable accuracy estimates of the underlying approximate posterior in total-variation distance and Kullback-Leibler divergence and, moreover, propose a hybrid sampler that is asymptotically exact by Metropolization. We validate the accuracy estimate
    
[^254]: 协同设计Gym：面向形态-策略协同优化的统一基准

    Co-design Gym: A Unified Benchmark for Embodiment-Policy Co-optimization

    [https://arxiv.org/abs/2610.02366](https://arxiv.org/abs/2610.02366)

    本文提出了Co-Design Gym，一套用于形态与策略联合协同优化的统一基准测试环境，突破了现有基准中形态固定、仅关注策略学习的局限。

    

    在给定环境中寻找最优行为策略是一个被广泛研究的问题，涉及游戏、机器人、能源基础设施、通信网络和多智能体系统等众多领域。目前已开发了众多基准测试来支持此类研究，但其中绝大多数都假设智能体的形态（设计）是固定的，只专注于策略学习。打破这一假设会产生一类更广泛的问题，在这类问题中，分别独立优化形态和策略是高度次优的。智能体的形态强烈地决定了哪些控制策略能够被发现，而最优形态又由它所支持的策略来定义。为了帮助研究界明确且系统地研究这类问题，我们提出了Co-Design Gym——一套用于联合优化形态与策略的基准测试环境。我们的环境涵盖机器人操作与运动等众多领域。

    arXiv:2610.02366v1 Announce Type: new  Abstract: Finding an optimal behaviour policy within a given environment is a widely studied problem in domains as diverse as games, robotics, energy infrastructure, communication networks, and multi-agent systems. Numerous benchmarks have been developed to support such research, but the vast majority assume that the agent's embodiment (design) is fixed, focusing instead on policy learning alone. Lifting this assumption gives rise to a broader class of problems in which optimizing embodiment and policy separately is highly suboptimal. An agent's embodiment strongly shapes which control policies can be discovered, while the optimal embodiment is in turn defined by the policies it admits. To help the research community study this class of problems explicitly and systematically, we introduce Co-Design Gym - a suite of benchmark environments for jointly optimizing embodiment and policy. Our environments span domains such as robotic manipulation and lo
    
[^255]: ArrivalBench：代理生成的数据管道一次运行即正确，随时间推移而出错

    ArrivalBench: Agent-Generated Data Pipelines Are Correct Once and Wrong Under Time

    [https://arxiv.org/abs/2610.02363](https://arxiv.org/abs/2610.02363)

    ArrivalBench 提出在对抗性但可重放的数据送达调度下重新执行代理生成的数据管道，并以完整日志的批量重算作为判定标准，发现经单次执行评分认证的管道中有 7.0-79.2% 实际上默默出错，且这一差距与修复循环无关。

    

    现有针对代理（Agent）生成数据工作的基准测试，通常通过让数据管道对固定快照运行一次来评分。ArrivalBench 则改变这一方式：在对抗性但可重放的送达调度（包括迟到、重复、乱序和重试的记录）下重新执行代理所留下的管道，并要求其最终状态等于对完整日志进行批量重算的结果。由于该判定基准采用重算而非分类的方式，“错误的表”和“崩溃”会被判定为不同的结论：崩溃可以被团队现有的监控系统发现，而错误的表则不会。在我们构建的 40 个任务上，我们对单次执行评分方式的复现实现认证了十一个模型所生成管道的 86-100%；而重新执行同样的产物，却发现其中 7.0-79.2% 被认证的管道实际上默默出错了。这一差距并非由修复循环造成：在相同模型和任务下，针对快照测试修复过的管道，其重放失败率与首次就通过该测试的管道大致相当。

    arXiv:2610.02363v1 Announce Type: new  Abstract: Benchmarks for agent-generated data work grade a pipeline by running it once against a fixed snapshot. ArrivalBench instead re-executes the pipeline an agent leaves behind under adversarial but replayable delivery schedules (late, duplicated, out-of-order and retried records) and requires its final state to equal a batch recomputation of the complete log. Because the oracle recomputes rather than classifies, a wrong table and a crash are distinct verdicts: a crash is visible to monitoring a team already runs, and a wrong table is not. On 40 tasks we built, our reimplementation of single-execution grading certifies 86-100% of the pipelines eleven models produce; re-executing the same artifacts finds 7.0-79.2% of the certified ones silently wrong. The gap is not produced by the repair loop: within the same model and task, pipelines repaired against the snapshot test fail replay about as often as those that passed it first time. In every mo
    
[^256]: 字典序多目标在线策略蒸馏

    Lexicographic Multi-Objective On-Policy Distillation

    [https://arxiv.org/abs/2610.02359](https://arxiv.org/abs/2610.02359)

    提出了字典序多目标在线策略蒸馏（LMOPD），一种多教师蒸馏方法，在显式优先级保护下整合奖励专门化策略，确保低优先级目标（如简洁性）不会以牺牲高优先级目标（如正确性）为代价而提升。

    

    基于可验证奖励的强化学习（RLVR）通常只优化答案的正确性，然而有用的语言模型行为还需要高质量的推理和简洁的回复。现有的多奖励后训练方法通常对奖励进行标量化，或组合多个专家模型，却没有显式地保护奖励的优先级顺序。当各目标之间的权衡不对称时，这种做法是有问题的：例如，简洁性不应以牺牲正确性为代价来提升。我们提出了字典序多目标在线策略蒸馏（LMOPD），这是一种在显式优先级约束下整合奖励专门化策略的多教师方法。对于学生模型的每一次采样轨迹，LMOPD 会选择门控机制检测到存在缺陷的首个目标所对应的专家，然后将其中心化的对数策略修正进行局部投影，以去除与更高优先级专家相冲突的成分。我们在两个专家和四个专家的设置下评估了 30B-A3B 混合专家 transformer 模型……

    arXiv:2610.02359v1 Announce Type: cross  Abstract: Reinforcement learning from verifiable rewards (RLVR) usually optimizes answer correctness, yet useful language-model behavior also requires high-quality reasoning and concise responses. Existing multi-reward post-training methods typically scalarize rewards or combine specialists without explicitly protecting a reward priority order. This is problematic when trade-offs are asymmetric: conciseness, for example, should not improve at the cost of correctness. We introduce Lexicographic Multi-Objective On-Policy Distillation (LMOPD), a multi-teacher method for integrating reward-specialized policies under explicit priorities. For each student rollout, LMOPD selects the specialist for the first objective whose gate detects a deficiency, then locally projects its centered log-policy correction to remove components that oppose higher-priority specialists. We evaluate 30B-A3B mixture-of-experts transformer models in two- and four-expert setti
    
[^257]: 基于深度序列模型的时间序列共形预测

    Conformal Prediction for Time Series with Deep Sequence Models

    [https://arxiv.org/abs/2610.02357](https://arxiv.org/abs/2610.02357)

    本文首次系统性地研究了深度序列模型在时间序列共形预测中的应用，通过条件分位数回归等三种方法，解决了传统共形预测所依赖的数据可交换性假设在时间序列中不成立的问题。

    

    深度学习在时间序列预测方面的最新进展放大了对可靠不确定性量化的需求。共形预测作为一种无分布的框架，因能够构建具有覆盖率保证的预测区间而受到关注。然而，其覆盖率保证依赖于数据可交换性这一假设，而该假设在时间序列数据中通常不成立。目前已有大量研究致力于开发能够克服这一局限的时间序列共形预测方法。尽管循环神经网络和Transformer等深度序列模型经常被用于时间序列的共形预测中，但关于如何系统性地将深度序列模型应用于时间序列共形预测的研究仍然有限。在这项工作中，我们通过三种方法系统地研究了深度序列模型在时间序列共形预测中的应用：条件分位数回归、条件分位数（摘要在此处截断）

    arXiv:2610.02357v1 Announce Type: cross  Abstract: Recent advances in deep learning for time series prediction have amplified the need for reliable uncertainty quantification. Conformal prediction has gained attention as a distribution-free framework for constructing prediction intervals with coverage guarantees. However, its coverage guarantees rely on data exchangeability, an assumption generally violated in time series. Active research has focused on developing conformal prediction methods for time series that overcome this limitation. While deep sequence models, such as recurrent neural networks and Transformers, have often been used in conformal prediction for time series, limited work has systematically studied how deep sequence models can be utilized in conformal prediction for time series. In this work, we systematically investigate the use of deep sequence models in conformal prediction for time series through three approaches: conditional quantile regression, conditional quan
    
[^258]: 为什么自适应批处理有助于大语言模型预训练？来自无界方差的视角

    Why Does Adaptive Batching Help LLM Pretraining? A Perspective from Unbounded Variance

    [https://arxiv.org/abs/2610.02355](https://arxiv.org/abs/2610.02355)

    该论文通过对LLM预训练中方差增长的实证分析，提出带可调增长指数的广义BG-a噪声模型，从无界方差控制的视角解释了自适应批大小调度为何能提升预训练效果。

    

    在大语言模型（LLM）预训练过程中增大批大小是一种常见做法，但其成功背后的理论依据尚未被很好地理解。随机优化的分析通常假设随机梯度方差一致有界，然而近期证据表明，这一假设在许多实际的非凸问题中并不成立。Blum--Gladyshev（BG-0）噪声模型放松了这一假设，允许方差随距初始化点的距离呈二次增长，这表明批大小调度器可以通过控制训练过程中方差的增长来发挥作用。然而，这种增长在实际中可能过于保守。我们实证研究了LLM预训练中方差的增长情况，并观察到带可调增长指数的广义BG模型能够更准确地刻画实际噪声行为。受这一观察启发，我们引入了广义BG-a噪声……

    arXiv:2610.02355v1 Announce Type: new  Abstract: Increasing the batch size during training is a common practice in large language model (LLM) pretraining, yet the theoretical justification behind its success is not well understood. Analyses of stochastic optimization often assume uniformly bounded stochastic gradient variance, yet recent evidence suggests that this assumption fails in many practical nonconvex problems. The Blum--Gladyshev (BG-$0$) noise model relaxes this assumption by allowing the variance to grow quadratically with the distance from initialization, suggesting that batch size schedulers can help by controlling the variance growth during training. However, this growth can be overly conservative in practice. We empirically investigate variance growth in LLM pretraining and observe that a generalized BG model with a tunable growth exponent provides a tighter description of practical noise behavior. Motivated by this observation, we introduce the generalized BG-$a$ noise 
    
[^259]: 每个用户都需要一个私有 LoRA 吗？将个性化与逐用户适配解耦

    Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation

    [https://arxiv.org/abs/2610.02353](https://arxiv.org/abs/2610.02353)

    提出 LINEUP 方法，通过实证发现各用户独立适配器中存在大量可跨用户共享的结构，进而学习一个共享的低秩个性化因子库并仅保留紧凑的用户专属校正，从而将个性化容量在共享与专属之间解耦，摆脱逐用户完整适配的可扩展性瓶颈。

    

    个性化大语言模型通常需要为每个用户维护一份完整的适配状态。然而，随着用户规模的扩大，这种范式的扩展性较差。我们从个性化容量分配的视角重新审视这一设计：多少适配容量可以在用户之间共享、共享容量应当如何构成、以及多少必须保留为用户专属。我们通过三项互补的实证分析回答这些问题。我们发现，独立的用户适配器中包含大量可跨用户复用的结构；可复用方向的效用同时反映了用户相关性与查询之间的差异性；并且用户历史能够为紧凑的个体校正提供可迁移的信号。基于这些发现，我们提出 LINEUP：它学习一组可复用的低秩个性化因子，通过用户条件化召回与查询相关校准来组合这些因子，并将目标用户的适配限制在……（摘要原文在此处截断）

    arXiv:2610.02353v1 Announce Type: cross  Abstract: Personalized large language models often require a complete adaptation state for each user. However, this paradigm scales poorly as the user population grows. We revisit this design through the lens of personalization capacity allocation: how much adaptation capacity can be shared across users, how the shared capacity should be composed, and how much must remain user-specific. We answer them through three complementary empirical analyses. We find that independent user adapters contain substantial cross-user reusable structure, that the utility of reusable directions reflects both user relevance and variation across queries, and that user histories provide transferable signals for compact individual correction. Motivated by these findings, we propose LINEUP. It learns a bank of reusable low-rank personalization factors, composes them through user-conditioned recall and query-dependent calibration, and restricts target-user adaptation to
    
[^260]: DeReAct：面向可靠AI智能体的分解式推理与行动

    DeReAct: Decomposed Reasoning and Acting for Reliable AI Agents

    [https://arxiv.org/abs/2610.02351](https://arxiv.org/abs/2610.02351)

    DeReAct提出了一种模块化智能体架构，通过将动作验证（Critic）与任务完成认证（Context Manager）从单一LLM策略中外置为独立门控机制，防止错误传播和无效完成声明，在GAIA和SWE-bench Verified上对较弱模型带来了最显著的Pass@1提升。

    

    基于ReAct的智能体通常依赖单一的大语言模型策略来提出动作、与环境交互，并决定任务何时完成。这种耦合使得动作授权与任务完成控制难以独立执行，从而导致错误传播，以及缺乏依据的完成声明过早终止执行过程。我们提出了DeReAct，一种模块化的智能体架构，它将两种门控策略外置：一个Critic（评论者）在执行前验证所提出的动作，一个Context Manager（上下文管理器）重构环境支持的State（状态）并认证任务完成。在GAIA和SWE-bench Verified基准上，DeReAct对较弱的Brain模型在Pass@1上的提升最为显著，其中Qwen3-Coder-480B提升6.5–7.0分，Claude Sonnet 4.5提升4.2–5.2分；随着Brain模型能力的增强，提升幅度逐渐减小。轨迹与消融分析表明，当目标故障足够普遍时，外部门控才是有效的……

    arXiv:2610.02351v1 Announce Type: new  Abstract: ReAct-based agents typically rely on a single LLM policy to propose actions, interact with the environment, and decide when a task is complete. This coupling makes action authorization and completion control difficult to enforce independently, allowing errors to propagate and unsupported completion claims to terminate execution. We introduce DeReAct, a modular agent architecture that externalizes two gating policies: a Critic that validates proposed actions before execution, and a Context Manager that reconstructs an environment-supported \textsc{State} and certifies task completion.   Across GAIA and SWE-bench Verified, DeReAct improves Pass@1 most for weaker Brain models, with gains of 6.5--7.0 points for Qwen3-Coder-480B and 4.2--5.2 points for Claude Sonnet~4.5; gains diminish as Brain capability increases. Trajectory and ablation analyses show that external gating is effective when targeted failures are sufficiently prevalent and th
    
[^261]: 从行为到溯源：将表格基础模型归因于合成预训练数据

    From Behavior to Provenance: Attributing Tabular Foundation Models to Synthetic Pretraining Data

    [https://arxiv.org/abs/2610.02347](https://arxiv.org/abs/2610.02347)

    该论文提出利用带有完整溯源信息的合成任务生成器O'PRIOR构建可控测试平台，通过反事实重训练将表格基础模型的训练数据归因从难以验证的推测转变为可实验检验的问题。

    

    训练数据归因旨在识别哪些训练样本塑造了模型的行为，然而验证这类主张十分困难，因为因果性的训练影响很少能被直接观测到。我们提出，受控的合成预训练使归因可以通过实验加以检验。利用O'PRIOR——一个面向表格基础模型的、具有丰富溯源信息的合成任务生成器——我们构建了一个测试平台，其中每个预训练任务都带有关于结构机制、数据缺失、混淆因素、捷径特征和分布偏移的明确谱系。我们将基于行为的归因方法与反事实重训练以及溯源感知的干预相结合，以同时检验任务层面的忠实性和机制层面的一致性。在留出的真实任务上，移除归因最高的5%合成任务会使平均ROC-AUC下降0.013，而随机移除仅导致0.002±0.004的下降；移除归因最低的任务反而使性能提升0.003。

    arXiv:2610.02347v1 Announce Type: new  Abstract: Training-data attribution aims to identify which training examples shape model behavior, yet validating such claims is difficult because causal training influence is rarely observable. We argue that controlled synthetic pretraining makes attribution experimentally testable. Using O'PRIOR, a provenance-rich synthetic task generator for tabular foundation models, we construct a testbed in which every pretraining task carries explicit lineage over structural mechanisms, missingness, confounding, shortcuts, and distribution shift. We combine behavior-conditioned attribution with counterfactual retraining and provenance-aware interventions to test both task-level faithfulness and mechanism-level consistency. On held-out real tasks, removing the top-attributed 5% of synthetic tasks decreases mean ROC-AUC by 0.013, compared with 0.002$\pm$0.004 under random removal, while removing bottom-attributed tasks improves performance by 0.003. Within sh
    
[^262]: 通过核锚定局部性正则化缓解固定目标异常检测器中的收敛坍塌

    Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors via Kernel-Anchored Locality Regularization

    [https://arxiv.org/abs/2610.02345](https://arxiv.org/abs/2610.02345)

    论文揭示了固定目标神经异常检测器中存在“收敛坍塌”这一结构性问题——训练越充分检测性能反而越差，并提出核锚定局部性正则化方法，通过引入局部性约束来阻止无约束外推，从而在模型完全收敛时仍能保留有效的异常检测信号。

    

    一类针对表格数据的异常检测器在平方误差损失下训练一个朝向固定目标的神经映射，并通过测试时的残差来为异常评分；收缩匹配、一步校正流以及重构自编码器都符合这一模板。我们刻画了一种“收敛坍塌”现象：更好的优化反而使检测器变得更差。在收敛时，所学到的映射即使在分布外数据上也会跟踪目标，因此残差信号在异常数据和正常数据上都会消失。因此，这类检测器依赖于隐式的非收敛性（如早停、容量限制）来保留检测信号。我们认为这是一个结构性问题：有效的异常检测需要一个局部性约束来阻止无约束的外推。经典检测器（kNN、KDE、孤立森林、LOF）显式地强制局部性；而固定目标神经检测器则没有。我们通过证明固定目标检测器的核回归类似物是一个（摘要在此处截断）来形式化这一联系。

    arXiv:2610.02345v1 Announce Type: new  Abstract: A family of tabular anomaly detectors trains a neural map toward a fixed target under squared-error loss and scores anomalies by the test-time residual; contraction matching, one-step rectified flow, and reconstruction autoencoders all fit this template. We characterize a convergence collapse: better optimization makes the detector worse. At convergence, the learned map tracks the target even off-distribution, so the residual signal vanishes on anomalies as well as on normal data. These detectors therefore rely on implicit non-convergence (early stopping, capacity caps) to retain signal. We argue this is structural: effective anomaly detection requires a locality constraint that blocks unconstrained extrapolation. Classical detectors (kNN, KDE, isolation forests, LOF) enforce locality explicitly; fixed-target neural detectors do not. We formalize the connection by showing that the kernel-regression analog of a fixed-target detector is a 
    
[^263]: 驱动与衰减：论联合嵌入预测架构的训练动力学

    Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures

    [https://arxiv.org/abs/2610.02344](https://arxiv.org/abs/2610.02344)

    本文提出了 JEPA 的早期训练稳定性理论，用驱动力 γ 与衰减效应 σ 之比构成的稳定性比率 μ 刻画表示坍缩的相变边界，并将预测器缩放、掩码比率与 EMA 统一为调节 μ 的机制，据此提出了 ResidualPred 预测器。

    

    联合嵌入预测架构容易发生表示坍缩，通常依靠经验性启发式方法来缓解。我们建立了一个能够统一这些启发式方法的早期训练稳定性理论。通过在平凡不动点附近对耦合的 JEPA 梯度流进行线性化分析，我们揭示了两种相互竞争的效应：驱动力（γ）与衰减效应（σ）。在近似谱解耦条件下，逐模态的稳定性比率 μᵢ = γᵢ/σᵢ 可以分解为相互独立的数据侧项与预测器侧项，且不稳定模态的数量追踪了可能涌现的表示的秩。该框架预测出一个相变边界，我们在超过 800 个 Tabular-JEPA 配置中通过实验验证了这一预测。该框架还将预测器缩放、掩码比率与 EMA 统一为移动 μ 的不同机制。受此分析启发，我们提出了 ResidualPred——一种注意力偏向于……（原文摘要在此截断）的 transformer 预测器。

    arXiv:2610.02344v1 Announce Type: new  Abstract: Joint-Embedding Predictive Architectures (JEPAs) are prone to representation collapse, typically mitigated through empirical heuristics. We develop an early-training stability theory that unifies these heuristics. Linearising the coupled JEPA gradient flow around the trivial fixed point reveals two competing effects: a driving force ($\gamma$) and a decay effect ($\sigma$). Under approximate spectral decoupling, a per-mode stability ratio $\mu_i = \gamma_i / \sigma_i$ factorises into independent data-side and predictor-side terms and the count of unstable modes tracks the rank of representations that can emerge. The framework predicts a phase boundary, which we confirm empirically across more than 800 Tabular-JEPA configurations. It also unifies predictor scaling, masking ratio, and EMA as distinct mechanisms for shifting $\mu$. Guided by this analysis, we introduce ResidualPred, a transformer predictor whose attention is biased toward t
    
[^264]: NEEDLEWORK：基于经验证局部缝合的机器人数据离线重写

    NEEDLEWORK: Offline Rewriting of Robot Data with Verified Local Stitches

    [https://arxiv.org/abs/2610.02339](https://arxiv.org/abs/2610.02339)

    NEEDLE是一种离线数据集增强算法，通过仅使用RGB图像和本体感觉信息，在高维机器人演示的观测之间添加经验证的短动作桥梁，从而离线重写训练数据、绕过次优轨迹并利用失败回合，无需新的环境交互。

    

    机器人演示数据即使在单个回合效率低下或不成功的情况下，也可能包含有用的行为。轨迹拼接提供了一种将这些行为组合成更优训练数据的方法，但在高维机器人数据中识别有用连接并验证其可行性十分困难，许多先前的方法都依赖于低维状态表示。我们提出了NEEDLE，一种离线数据集增强算法，通过在高维机器人演示的已记录观测之间添加短的、经过验证的动作桥梁来解决这些挑战。首先，NEEDLE仅使用RGB图像、本体感觉信息和回合级结果，无需新的环境交互或特权物体状态信息，即可识别并创建能够绕过次优绕路、拓宽动作覆盖范围、并利用失败轨迹增强原始数据集的连接。其次，我们提出了一种采样技术，该技术结合了……（原文摘要在此处截断）

    arXiv:2610.02339v1 Announce Type: cross  Abstract: Robot demonstrations may contain useful behavior even when individual episodes are inefficient or unsuccessful. Trajectory stitching offers a way to compose these behaviors into improved training data, but identifying useful connections and verifying their feasibility is difficult in high-dimensional robot data, where many prior methods rely on low-dimensional state representations. We introduce NEEDLE, an offline dataset-augmentation algorithm that addresses these challenges by adding short, verified action bridges between recorded observations in high-dimensional robot demonstrations. First, NEEDLE identifies and creates connections that bypass suboptimal detours, broaden action coverage, and augment the original dataset with failed trajectories, using only RGB images, proprioception, and episode-level outcomes, without new environment interaction or privileged object state. Next, we present a sampling technique that incorporates acc
    
[^265]: SoTa：用于灵巧操作的柔软触觉皮肤

    SoTa: Soft Tactile Skins for Dexterous Manipulation

    [https://arxiv.org/abs/2610.02338](https://arxiv.org/abs/2610.02338)

    SoTa是一种低成本的电容式柔软触觉皮肤，可在人手和机器人手上实现全手覆盖并保持202个触觉单元的统一布局，每片材料成本不足10美元，为通过大规模人类示教数据提升机器人灵巧操作能力提供了可行途径。

    

    越来越多的研究表明，触觉感知能够为机器人策略提供在灵巧操作中与视觉互补的接触信息。然而，视触觉机器人数据仍然稀缺：灵巧操作示教需要遥操作机器人，这限制了数据集的规模。人类示教的收集成本低得多，为扩大此类数据提供了途径，但前提是人类和机器人手上需佩戴具有对应信号的触觉传感器。这就要求传感器能够贴合不同的手部几何形状、覆盖全手，并在不同载体之间共享统一的布局。我们提出SoTa，这是一种低成本的电容式触觉皮肤，可为人手和机器人手提供全手覆盖，同时在对应的手指和手掌区域保持202个触觉单元（taxels）的统一布局。我们采用织物电极的多层设计，能够以每片不到10美元的材料成本自主制造薄而柔软、几何形状可定制的触觉皮肤。该传感器…（摘要原文被截断）

    arXiv:2610.02338v1 Announce Type: cross  Abstract: A growing body of work suggests that tactile sensing gives robot policies contact information that complements vision in dexterous manipulation. However, visuo-tactile robot data remains scarce: dexterous demonstrations require teleoperating robots, which limits dataset scale. Human demonstrations are far cheaper to collect and offer a path to scale this data, but only if human and robot hands carry tactile sensors with corresponding signals. This requires sensors that conform to different hand geometries, cover the full hand, and share a common layout across embodiments. We present SoTa, a low-cost capacitive tactile skin that provides full-hand coverage on humans and robots while preserving a shared layout of 202 taxels across corresponding finger and palm regions. Our multilayer design with fabric electrodes enables in-house fabrication of thin, soft skins with customizable geometry for under $10 in materials per skin. The sensor re
    
[^266]: 移动具身智能网络（MEAN）中移动与压缩比的联合设计

    Joint Movement and Compression Ratio Design for Mobile Embodied AI Networks (MEAN)

    [https://arxiv.org/abs/2610.02334](https://arxiv.org/abs/2610.02334)

    该论文针对上行链路移动具身智能网络（MEAN）系统，首次联合优化移动距离、语义压缩比和发射功率以最大化最小能效，并提出了AO-Dinkelbach交替优化算法来求解该非凸问题。

    

    移动具身智能网络（MEAN）使具身智能体能够在无线环境中进行感知、推理、通信和行动。在此类网络中，智能体的移动可以改善信道条件，而语义压缩可以减少传输负载。然而，移动会消耗能量，更强的压缩会带来额外的计算成本。本文研究了上行链路MEAN系统中移动、语义压缩和发射功率的联合设计。我们在可控功率约束下，通过联合优化发射功率、移动距离和语义压缩比，构建了一个最大-最小能效（EE）优化问题。由于信号与干扰加噪声比（SINR）的耦合、依赖于移动性的信道增益以及分数型能效目标，该问题是非凸的。为求解该问题，我们提出了一种交替优化（AO）-Dinkelbach算法，其中分数型目标函数通过Dinkelbach变换进行处理。

    arXiv:2610.02334v1 Announce Type: cross  Abstract: Mobile embodied AI networks (MEAN) enable embodied agents to perceive, reason, communicate, and act in wireless environments. In such networks, agent mobility can improve channel conditions, while semantic compression can reduce transmission payloads. However, movement consumes energy, and stronger compression incurs additional computational cost. This paper studies joint movement, semantic compression, and transmit power design for an uplink MEAN system. We formulate a max-min energy efficiency (EE) problem by jointly optimizing transmit power, movement distance, and semantic compression ratio under controllable power constraints. The problem is non-convex due to the coupled signal-to-interference-plus-noise ratio (SINR), mobility-dependent channel gains, and fractional EE objective. To solve it, we propose an alternating optimization (AO)-Dinkelbach algorithm, where the fractional objective is handled by the Dinkelbach transformation
    
[^267]: 面向能力保持的慢-快多教师在线蒸馏方法

    Slow-Fast Multi-Teacher On-Policy Distillation for Capability Preservation

    [https://arxiv.org/abs/2610.02324](https://arxiv.org/abs/2610.02324)

    提出 SF-MOPD 方法，通过将教师直接更新的快速学生模型与作为动态能力参考的指数移动平均慢速模型相耦合，在多教师在线蒸馏中实现领域专长获取与通用能力保持的平衡。

    

    基础多模态大语言模型旨在支持跨多个领域的广泛能力。多教师在线蒸馏（MOPD）为将特定领域的专业知识整合到单个学生模型中提供了一个有效框架。然而，MOPD 训练会逐渐使学生模型偏离其初始化模型，且随着偏移量的增大，通用能力会随之下降，从而导致能力干扰。一种直接的补救措施是将学生模型约束在其初始化状态附近，但这同样会抑制领域专业知识的获取。我们提出了慢-快多教师在线蒸馏（SF-MOPD），该方法将一个快速模型（即由每个教师直接更新的当前学生模型）与一个慢速模型（即学生模型的指数移动平均）相耦合。慢速模型逐渐吸收学习信号，充当一个动态的能力参考，将通用基础能力与……（摘要在此处被截断）

    arXiv:2610.02324v1 Announce Type: cross  Abstract: Foundation multimodal large language models are designed to support a broad spectrum of capabilities across diverse domains. Multi-teacher on-policy distillation (MOPD) provides an effective framework for consolidating domain-specific expertise into a single student model. However, MOPD training gradually drives the student away from its initialization model, and general capabilities decline as the displacement grows, resulting in capability interference. A direct remedy is constraining the student toward its initialization, but this suppresses the acquisition of domain expertise as well. We propose Slow-Fast Multi-Teacher On-Policy Distillation (SF-MOPD), which couples a fast model, the current student updated directly by each teacher, with a slow model, an exponential moving average of the student. The slow model absorbs the learning signal gradually, serving as a moving capability reference that fuses the general foundation with con
    
[^268]: DeskForge：来自桌面环境的密集监督，用于计算机使用智能体

    DeskForge: Dense Supervision from Desktop Environments for Computer-Use Agents

    [https://arxiv.org/abs/2610.02320](https://arxiv.org/abs/2610.02320)

    本文提出可控桌面环境DeskForge，通过组合和变换真实应用程序生成大规模密集标注语料库DeskForge-1M（含120万条桌面观测与1.597亿个元素实例），有效提升了视觉语言模型在复杂桌面场景中的动作目标定位能力。

    

    计算机使用智能体需要在复杂的桌面场景中可靠地定位动作目标，而在这些场景中，多个应用程序、重叠的窗口以及视觉上相似的控件会争夺注意力。现有的训练数据很少将此类场景与密集标注配对，也很少以可控的方式对其进行变化。我们提出了DeskForge，这是一个可控的桌面环境，通过组合和探索真实应用程序来为计算机使用智能体生成大规模监督数据。它可以变换应用程序状态、内容、窗口布局、外观和分辨率，并将屏幕截图、无障碍树和窗口几何信息融合为密集的元素标注，同时记录每个执行动作的结果。利用该环境，我们构建了DeskForge-1M，这是一个包含120万条标注桌面观测数据的语料库，共含1.597亿个元素实例。我们使用从DeskForge-1M中抽取的20万个定位示例对四个视觉语言模型进行微调，所有四个模型在保留测试集上均有提升。

    arXiv:2610.02320v1 Announce Type: cross  Abstract: Computer-use agents need to reliably ground action targets in complex desktop scenes, where multiple applications, overlapping windows, and visually similar controls compete for attention. Existing training data rarely pair such scenes with dense annotations or vary them in a controlled way. We introduce DeskForge, a controllable desktop environment that composes and explores real applications to generate large-scale supervision for computer-use agents. It varies application states, content, window layout, appearance, and resolution, and fuses screenshots, accessibility trees, and window geometry into dense element annotations while recording the outcome of each executed action. Using this environment, we construct DeskForge-1M, a corpus of 1.2M annotated desktop observations containing 159.7M element instances. We fine-tune four vision-language models on 200K grounding examples drawn from DeskForge-1M. All four improve across held-out
    
[^269]: SimuVerity：面向工程级Simulink模型生成的智能体基准测试

    SimuVerity: Benchmarking Agents for Engineering-Grade Simulink Model Generation

    [https://arxiv.org/abs/2610.02304](https://arxiv.org/abs/2610.02304)

    提出SimuVerity基准，包含101个跨十个工程领域的Simulink模型生成任务，采用分层评估器从六个工程维度对模型评分，发现最佳智能体系统总分仅为42.86，证明结构相似度并不能衡量模型的工程性能。

    

    现有的Simulink基准测试主要评估生成的模型能否编译、执行或与参考模型相似。这些标准无法确定模型是否满足其工程需求。我们提出了SimuVerity，这是一个包含101个跨十个工程领域的文本到可执行Simulink模型生成任务的基准测试。对于每个任务，可执行系统配置文件为工程规范和四类原生仿真场景提供了基础。分层评估器首先检查工件交付、原生可执行性和工程资格，然后从六个维度对合格模型进行评分，涵盖准确性、输出质量、机理保真度、控制与因果完整性、工作域鲁棒性以及动态响应。我们使用SimuVerity评估了六个智能体系统，表现最好的系统总分仅为42.86。结果表明，结构相似度并不能很好地代表工程性能。

    arXiv:2610.02304v1 Announce Type: cross  Abstract: Existing Simulink benchmarks mainly evaluate whether generated models compile, execute, or resemble a reference model. These criteria do not establish whether a model satisfies its engineering requirements. We introduce SimuVerity, a benchmark of 101 text-to-executable Simulink model-generation tasks across ten engineering domains. For each task, executable-system profiles ground the engineering specification and four families of native simulation scenarios. A hierarchical evaluator first checks artifact delivery, native executability, and engineering qualification, then scores qualified models across six dimensions covering accuracy, output quality, mechanistic fidelity, control and causal integrity, operating-domain robustness, and dynamic response. We evaluate six agent systems with SimuVerity. The best system achieves an overall score of only 42.86. The results show that structural similarity is a poor proxy for engineering perform
    
[^270]: PowerBench：衡量语言模型在权力转移请求中的偏差

    PowerBench: Measuring Language Model Bias in Power-shifting Requests

    [https://arxiv.org/abs/2610.02303](https://arxiv.org/abs/2610.02303)

    该论文提出了PowerBench这一开源基准，通过区分自我赋权、去权和权力攫取三类权力转移请求，系统评估了24个中美语言模型在权力相关请求上的拒绝行为，揭示了模型拒绝倾向的系统性偏差。

    

    语言模型越来越多地协助人们处理与权力相关的请求，因此它们在帮助对象上的系统性差异可能会大规模地改变权力分配格局，或被那些了解到哪些身份较少被拒绝的用户所利用。我们提出了 PowerBench，这是一项针对权力转移请求的评估，它区分了自我赋权、去权（被剥夺权力）和权力攫取三种情况，并设置了不转移任何权力的诱导拒绝请求作为对照组。我们构建、整理并开源了一个包含此类请求的数据集，其中改变了权力领域、情境、受影响方的规模以及用户先前权力地位等变量，并在三种实验条件下评估了24个模型（12个来自美国开发商，12个来自中国开发商）：用户与受影响方国籍互换、AI智能体作为用户、以及8种请求语言。结果显示，模型对权力攫取的拒绝多于对去权的拒绝，对去权的拒绝又多于对自我赋权的拒绝，对权力攫取的拒绝率随……

    arXiv:2610.02303v1 Announce Type: new  Abstract: Language models increasingly assist people with power-related requests, so systematic differences in whom they help could shift the distribution of power at scale, or be exploited by users who learn which identities are refused less. We introduce PowerBench, an evaluation of power-shifting requests that distinguishes self-empowerment, disempowerment, and power grabbing, plus a control of refusal-inducing requests that shift no power. We build, curate, and open-source a dataset of such requests varying the power domain, the context, the scale of the affected party, and the prior power standing of the user, and evaluate 24 models (12 from US and 12 from Chinese developers) under three experimental conditions: reciprocal nationalities of user and affected party, an AI agent as the user, and 8 request languages. Models refuse power grabbing more than disempowerment, and disempowerment more than self-empowerment. Refusal of power grabbing ris
    
[^271]: 意图隐藏越狱攻击：组合攻击的信息论框架

    Intent-Hiding Jailbreaks: An Information-Theoretic Framework for Compositional Attacks

    [https://arxiv.org/abs/2610.02302](https://arxiv.org/abs/2610.02302)

    该论文提出了一种信息论框架，通过选择辅助任务实现有害意图先验概率与后验概率的匹配，从而在大语言模型的组合式查询中隐藏有害意图，构成一种新型越狱攻击方法。

    

    最近的研究表明，大语言模型（LLM）可能容易受到越狱攻击，其中有害意图通过与良性任务的组合而被掩盖。单独会被拒绝的有害请求，当嵌入到更大的、看似良性的查询中时，可能会引发不同的响应。我们从信息论的视角研究这类组合式意图隐藏越狱攻击。我们的公式化方法将每个任务与一个被视为有害的估计概率相关联：整个任务集合上的平均值定义了有害意图的先验概率，而包含目标任务的所选任务包上的平均值定义了后验概率。通过选择辅助任务使这两个平均值保持一致——我们称之为“先验-后验匹配”——即使有害目标仍然存在于任务包中，所估计的意图也不会发生改变。我们研究了两种设置，它们的区别在于查询构建是否作为优化过程的一部分。在……

    arXiv:2610.02302v1 Announce Type: cross  Abstract: Recent work has shown that large language models (LLMs) can be vulnerable to jailbreak attacks in which harmful intent is obscured through composition with benign tasks. A harmful request refused in isolation may elicit a different response when embedded within a larger, seemingly benign query. We study these compositional intent-hiding jailbreaks from an information-theoretic perspective. Our formulation associates each task with an estimated probability of being judged harmful: the average over the full task collection defines the prior probability of harmful intent, while the average over a selected bundle containing the target defines the posterior. Selecting auxiliary tasks so that these averages agree, which we call prior-posterior matching, leaves the estimated intent unchanged even though the harmful target remains in the bundle.   We study two settings that differ in whether query construction is part of the optimization. In t
    
[^272]: $\Psi$-韧性：基于一维拓扑信号的无模型特征重要性方法

    $\Psi$-Resilience: Model-Free Feature Importance from 1D Topological Signals

    [https://arxiv.org/abs/2610.02299](https://arxiv.org/abs/2610.02299)

    提出了一种基于一维拓扑信号的无模型特征重要性方法 $\Psi$-Resilience，它通过类条件密度差异构建不一致性景观并利用其持续性定义韧性评分，从而产生上下文鲁棒且可审计的特征排序。

    

    我们提出了 $\Psi$-Resilience（$\Psi$-韧性），这是一种无模型的特征重要性方法，通过一维拓扑信号直接从数据本身推导解释。我们的方法通过估计类条件密度并沿特征轴取其逐点绝对差，构建出类别不一致性景观。然后，该一维信号的0维持续性定义了一个韧性泛函，仅聚合那些在扰动下存活至用户设定的鲁棒性尺度的拓扑特征。由此得到一个上下文鲁棒的重要性评分，并且可以通过底层的一维景观及其持续性进行固有的可审计性验证。我们在合成数据集和真实数据集上对该方法进行了评估。在具有指定真实重要性的合成生成器上，$\Psi$-Resilience 能够高保真地恢复特征排序，Spearman 秩相关性最高达到 0.8，并与多种（基线方法）表现出相当的竞争力。

    arXiv:2610.02299v1 Announce Type: cross  Abstract: We introduce $\Psi$-Resilience, a model-free feature importance method that derives explanations directly from the data itself via 1D topological signals. Our method constructs a class-disagreement landscape by estimating class-conditional densities and taking their pointwise absolute difference along the feature axis. Then, the 0-dimensional persistence of this 1D signal defines a resilience functional that aggregates only those topological features that survive perturbations up to a robustness scale which is set by the user. This gives us a context-robust importance score that is inherently auditable via the underlying 1D landscapes and their persistence. We evaluate our method on both synthetic and real datasets. On synthetic generators with specified ground-truth importance, $\Psi$-Resilience recovers the ranking of features with high fidelity, achieving Spearman rank correlations up to 0.8 and performing competitively with multipl
    
[^273]: 基于扩散模型合成数据预训练以增强人体活动识别

    Diffusion-Based Synthetic Data Pretraining for Enhancing Activity Recognition

    [https://arxiv.org/abs/2610.02292](https://arxiv.org/abs/2610.02292)

    本研究提出利用扩散模型生成合成传感器数据进行预训练、再在真实数据上微调的两阶段训练策略，以增强CABiGRU模型对进食、饮水等细微少数类别人体活动的识别能力。

    

    人体活动识别（HAR）在医疗保健、健康和日常监测应用中日益重要，其中检测进食和饮水等饮食活动可以为饮食习惯和慢性病管理提供可操作的洞察。然而，HAR系统在细微和代表性不足的类别上往往表现不佳，限制了其在现实世界饮食监测中的实用性。本工作基于CABiGRU——一种具有双向GRU层、多头注意力和残差连接的卷积架构，旨在从智能手表加速度计、陀螺仪和磁力计数据中捕获有判别力的时间模式。为了提高CABiGRU的泛化能力并减少少数类别的欠拟合，我们利用扩散模型生成合成传感器数据窗口，并采用两阶段训练策略：先在合成数据上预训练CABiGRU，然后在真实世界数据上进行微调。

    arXiv:2610.02292v1 Announce Type: cross  Abstract: Human activity recognition (HAR) is increasingly important for healthcare, well-being, and daily monitoring ap- plications, for which detecting alimentary activities such as eating and drinking can provide actionable insight into dietary habits and chronic disease management. HAR systems, however, often underperform on subtle and underrepresented classes, limiting their utility in real-world dietary monitoring. This work builds upon CABiGRU, a convolutional architecture with Bidirectional GRU layers, multi-head attention, and residual connections, designed to capture discriminative temporal patterns from smart- watch accelerometer, gyroscope, and magnetometer data. To improve CaBiGRU's generalization and reduce underfitting in the minority class, we leverage synthetic sensor data windows using a diffusion model and adopt a two-stage training strategy: pre-training CABiGRU on synthetic data, followed by fine-tuning on the real-world dat
    
[^274]: 期望效用遗憾规则：极小极大与贝叶斯最优投资组合选择

    Expected Utility Regret Rule: Minimax and Bayes Optimal Portfolio Choice

    [https://arxiv.org/abs/2610.02290](https://arxiv.org/abs/2610.02290)

    提出期望效用遗憾（EUR）规则，该规则无需先验分布即可同时达到极小极大与贝叶斯最优下界，并将均值-方差组合和风险平价组合统一为该框架的特例。

    

    本研究考虑投资组合选择问题，即为投资者推荐一个投资组合，以最大化其财富的期望效用。我们的目标是构建一个在期望效用遗憾（即“先知”投资者的期望效用与从数据中选择的投资组合所实现的期望效用之差）意义上渐近最优的投资组合选择规则。我们提出了期望效用遗憾（EUR）规则，该规则联合选择投资组合类别并估计其权重。在正则参数化收益模型中，单一的EUR规则在不使用定义贝叶斯准则的先验分布的情况下，同时达到了极小极大下界和贝叶斯下界，包括它们的首项常数。随后，我们将均值-方差组合和风险平价组合推导为该框架的特例。在光滑、递增且凹的效用函数下，EUR规则与样本均值-方差组合在……（摘要在此处截断）

    arXiv:2610.02290v1 Announce Type: cross  Abstract: This study considers the problem of portfolio choice, where we recommend a portfolio to an investor to maximize the expected utility of their wealth. Our goal is to construct an asymptotically optimal portfolio choice rule in terms of expected utility regret, the difference between the expected utility of an oracle investor and that achieved by a portfolio chosen from data. We propose the Expected Utility Regret (EUR) rule, which jointly selects a portfolio class and estimates its weights. In a regular parametric return model, a single EUR rule attains both the minimax and the Bayes lower bounds, including their leading constants, without using the prior distribution that defines the Bayes criterion. We then derive the mean--variance and risk-parity portfolios as special cases of this framework. Under smooth increasing and concave utility, the EUR rule and the sample mean--variance portfolio attain the same leading expected regret when
    
[^275]: 脉冲间隔变化对蝙蝠叫声深度学习分类的影响

    Effects of interpulse-interval variation on deep-learning classification of bat vocalizations

    [https://arxiv.org/abs/2610.02284](https://arxiv.org/abs/2610.02284)

    该研究通过构建自然脉冲间隔与归一化脉冲间隔的匹配数据集进行对比实验，揭示了脉冲间隔变化蕴含蝙蝠物种判别信息，且基于Transformer的模型比卷积神经网络对这一时序特征更为敏感。

    

    时间上下文可能有助于蝙蝠物种的自动分类，但特定特征的具体贡献仍不清楚。我们研究了脉冲间隔（IPI）——即连续叫声起始点之间的时间——的变化是否提供物种判别信息，以及基于Transformer的模型是否比卷积神经网络对该信息更敏感。我们从欧洲蝙蝠录音中创建了两个匹配的数据集：一个自然IPI条件，保留原始叫声的时间顺序；一个归一化IPI条件，其中叫声起始点以50毫秒的间隔排列。我们对EfficientNet-B0和PaSST进行了微调，并在每种条件下进行评估。在另一项实验中，每种架构分别在自然IPI和归一化IPI录音上进行训练，并在相同的自然IPI测试集上进行评估。最后，预训练分类器BatDetect2和BAT在两种条件下进行了评估。条件内的IPI归

    arXiv:2610.02284v1 Announce Type: new  Abstract: Temporal context may aid automated bat-species classification, but the contribution of specific features remains unclear. We investigated whether variation in the interpulse interval (IPI)-the time between consecutive call onsets-provides species-discriminative information and whether transformer-based models are more sensitive to this information than convolutional neural networks. We created two matched datasets from European bat recordings: a natural-IPI condition retaining the original call timing and a normalized-IPI condition in which call onsets were spaced at 50-ms intervals. EfficientNet-B0 and PaSST were fine-tuned and evaluated within each condition. In an additional experiment, each architecture was trained separately on natural-IPI and normalized-IPI recordings, and evaluated on the same natural-IPI test set. Finally, the pretrained classifiers BatDetect2 and BAT were evaluated on both conditions. Within-condition IPI normal
    
[^276]: MuLoRA：面向持续学习的频谱平衡低秩适配

    MuLoRA: Spectrally Balanced Low-Rank Adaptation for Continual Learning

    [https://arxiv.org/abs/2610.02283](https://arxiv.org/abs/2610.02283)

    该论文发现持续学习中LoRA存在“频谱可塑性坍缩”问题——更新能量集中于少数奇异模态导致低秩空间利用不足，并提出MuLoRA通过历史白化与动量近似极正交化联合控制容量分配与利用，实现频谱平衡的持续学习适配。

    

    低秩适配为持续学习提供了一种参数高效的方法，但其名义秩可能掩盖有效适配能力的损失。我们识别出一种“频谱可塑性坍缩”现象：在顺序适应过程中，更新能量会集中在少数奇异模态上，导致大部分可用的低秩空间未被充分利用。这暴露了仅依靠干扰规避的局限性：保护历史表征并不能确保剩余的适配能力对新任务保持响应或得到有效利用。为解决这一问题，我们提出MuLoRA，该方法联合控制容量的分配与利用。首先，历史白化机制识别出相对于累积历史响应具有较强当前任务响应的输入方向，从而构建出在训练期间保持固定的任务自适应基。其次，通过动量的近似极正交化……（摘要在此处被截断）

    arXiv:2610.02283v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) provides a parameter-efficient approach to continual learning, but its nominal rank can conceal a loss of effective adaptation capacity. We identify \emph{spectral plasticity collapse}: during sequential adaptation, update energy becomes concentrated in a small subset of singular modes, leaving much of the available low-rank space underutilized. This exposes a limitation of interference avoidance alone: protecting historical representations does not ensure that the remaining adaptation capacity is responsive to new tasks or effectively utilized. To address this problem, we propose \texttt{MuLoRA}, which jointly controls capacity allocation and utilization. First, historical whitening identifies input directions with strong current-task response relative to accumulated historical response, yielding a task-adaptive basis that remains fixed during training. Second, approximate polar orthogonalization of momentum u
    
[^277]: CLEAN：通过架构隔离实现概念空间扩展下具有心理测量一致性的增量认知诊断

    CLEAN: Psychometrically Consistent Incremental Cognitive Diagnosis under Concept-Space Expansion via Architectural Isolation

    [https://arxiv.org/abs/2610.02278](https://arxiv.org/abs/2610.02278)

    提出CLEAN增量认知诊断框架，通过架构隔离支持概念空间的动态扩展，并从结构上保证增量更新后的心理测量一致性，避免灾难性遗忘。

    

    认知诊断（CD）是智能教育中的一项基础任务，用于刻画学习者对知识概念的掌握水平。在现实学习平台中，新增题目会不断引入此前未见过的概念，因此需要对底层概念空间进行动态扩展。然而，现有的增量认知诊断模型假设概念空间固定不变，使得来自新题目的梯度会覆盖历史路径并引发灾难性遗忘。更关键的是，这些方法仅依靠软约束来保留历史诊断结果，而此类约束可能无法满足增量更新后诊断结果保持不变的要求，这一要求在认知诊断领域被称为心理测量一致性。为此，我们提出了CLEAN（基于可扩展与架构隔离网络的持续学习），这是一种新颖的增量认知诊断框架，在支持概念空间扩展的同时提供结构性保证。

    arXiv:2610.02278v1 Announce Type: new  Abstract: Cognitive diagnosis (CD) is a fundamental task in intelligent education that profiles learner proficiency over knowledge concepts. In real-world learning platforms, newly added items continually introduce previously unseen concepts, necessitating dynamic expansion of the underlying concept space. Yet existing incremental CD models assume a fixed concept space, allowing gradients from new items to overwrite historical pathways and induce catastrophic forgetting. More critically, these methods rely solely on soft constraints to preserve historical diagnoses. Such constraints may fail to satisfy the requirement of diagnostic invariance after incremental updates, a requirement known as psychometric consistency in cognitive diagnosis. Therefore, we propose CLEAN (Continual Learning with Expandable and Architecturally Isolated Networks), a novel incremental CD framework supporting concept-space expansion while providing structural guarantees f
    
[^278]: 基于变分风险最小化的置信度门控云-边级联医学影像分诊方法

    Confidence-Gated Cloud-Edge Cascade Triage via Variational Risk Minimization for Medical Imaging

    [https://arxiv.org/abs/2610.02269](https://arxiv.org/abs/2610.02269)

    提出变分风险最小化（VRM）蒸馏框架，将LVLM生成的报告变体作为蒙特卡洛样本从边缘化教师分布中学习，以应对报告缺失的模态鸿沟，并结合置信度门控云-边级联，在103ms延迟下实现AUC 0.941的急诊胸片分诊性能。

    

    急诊胸部X光（CXR）分诊存在一个结构性的模态鸿沟：临床报告在分诊决策之后才生成，而多模态基础模型却需要图文双模态输入。我们提出了变分风险最小化，这是一种蒸馏框架，它将大型视觉语言模型（LVLM）生成的报告变体视为潜在临床解释的蒙特卡洛样本。VRM并非从单一的教师目标进行蒸馏，而是从变分边缘化的教师分布中学习，从而在模态缺失的约束下实现不确定性感知的监督。在编码器家族匹配的条件下，VRM优于直接微调基线，改善了模型校准，并能从幻觉监督中强力恢复。边缘化监督降低了报告选择的不稳定性。在我们紧凑的边缘端学生模型实例中，置信度门控级联在103毫秒平均延迟下达到AUC 0.941，云端升级率为20.3%，提供了一个明确的可靠性-延迟运行点。

    arXiv:2610.02269v1 Announce Type: cross  Abstract: Emergency chest X-ray (CXR) triage has a structural modality gap: reports arrive after triage decisions, yet multimodal foundation models require image-text inputs. We present Variational Risk Minimization (VRM), a distillation framework that treats LVLM-generated report variants as Monte Carlo samples of latent clinical interpretations. Rather than distilling from a single teacher target, VRM learns from a variationally marginalized teacher distribution, enabling uncertainty-aware supervision under missing-modality constraints. Under matched encoder families, VRM outperforms direct fine-tuning baselines and improves calibration with strong recovery from hallucinated supervision. Marginalized supervision reduces report-selection instability. In our compact edge-student instantiation, a confidence-gated cascade reaches AUC 0.941 at 103ms average latency with 20.3% cloud escalation, yielding an explicit reliability-latency operating poin
    
[^279]: 从数学证书到可执行证书的机器遗忘

    From Mathematical to Executable Certificates for Machine Unlearning

    [https://arxiv.org/abs/2610.02268](https://arxiv.org/abs/2610.02268)

    该论文提出ExecCert，一个发布时的可执行认证层，通过补全方法原生证书或应用重训练参考发布验证（RRV）来认证待发布的机器遗忘制品，从而弥合数学保证与实际部署的有限精度软件制品之间的差距。

    

    当数据必须因删除请求、过时记录或数据质量问题而被移除时，就需要机器遗忘，而从头重新训练可能代价高昂。经过认证的机器遗忘方法提供数学保证，而已部署的系统发布的则是由软件产生的具体的有限精度制品。为了弥合数学保证与实际部署之间的差距，我们提出了可执行发布认证，这是一个发布时认证层，用于对考虑发布的候选制品进行认证。ExecCert要么为所执行的候选补全方法的原生证书，要么应用重训练参考发布验证来认证其与当前保留集重训练结果的保真度。由于精确的保留集参考和存储的数值状态会分别演化，顺序删除使得后者变得非平凡。对于带有可变岭回归头的冻结表示，我们开发了一种增量……（原文摘要在此处截断）

    arXiv:2610.02268v1 Announce Type: new  Abstract: Machine unlearning is needed when data must be removed because of deletion requests, outdated records, or data-quality concerns, while retraining from scratch can be costly. Certified machine unlearning methods provide mathematical guarantees, while deployed systems release concrete finite-precision artifacts produced by software. To bridge the gap between mathematical guarantees and practical deployment, we introduce Executable Release Certification (ExecCert), a release-time layer that certifies the candidate artifact considered for release. ExecCert either closes a method's native certificate for the executed candidate or applies Retraining-Reference Release Verification (RRV) to certify fidelity to current retain-set retraining. Sequential deletion makes the latter nontrivial because the exact retain-set reference and the stored numerical state evolve separately. For frozen representations with a mutable ridge head, we develop an inc
    
[^280]: 快模型，慢证据：面向LLM代理工具框架的System-1决策模型的配对与自审计评估

    Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses

    [https://arxiv.org/abs/2610.02267](https://arxiv.org/abs/2610.02267)

    该论文通过严格配对与自审计的评估发现，托管型System-1决策模型Jev在11个代理决策点中的9个上显著优于开源模型Laya，但两者在零样本模型路由上均未超过随机水平，且开源模型对选项顺序和候选数量高度敏感。

    

    代理工具框架在每个任务中需要做出许多小型、类型化的决策：调用哪个模型、使用哪个工具、检索到的文本是否相关、输入是否携带注入攻击。System-1决策模型通过单次前向传播输出类别概率来回答此类问题，相比LLM调用有望大幅节省成本和延迟。我们在11个代理决策点上对开源权重模型和托管模型进行了配对评估，这些决策点基于18个公开来源构建：共7,283个基础用例加上6,640个鲁棒性变体，采用字节级相同的输入、配对测试以及跨硬件和跨日期的可重复性检查。Jev在11个决策点中的9个上显著更准确（提升+10.8至+46.0个百分点）。两个模型在零样本模型路由上均未超过随机水平，在RAG相关性门控上则不分伯仲。当选项顺序被颠倒时，Laya会改变30%的答案，并且在候选选项较多或相似时性能急剧下降（在50个最近邻的情况下仅为31%……

    arXiv:2610.02267v1 Announce Type: new  Abstract: Agent harnesses make many small, typed decisions per task: which model to call, which tool to use, whether retrieved text is relevant, whether an input carries an injection. System-1 decision models answer such questions in a single forward pass with class probabilities, promising large cost and latency savings over LLM calls. We present a paired evaluation of an open-weight (Laya) and a hosted (Jev) System-1 model on 11 agent decision points built from 18 public sources: 7,283 base cases plus 6,640 robustness variants, with byte-identical inputs, paired tests, and cross-hardware and cross-day reproducibility checks. Jev is significantly more accurate on 9 of 11 decision points (+10.8 to +46.0 pp). Neither model beats chance on zero-shot model routing, and they tie on RAG relevance gating. Laya changes 30% of its answers when the option order is reversed and degrades sharply with many or similar candidates (31% at 50 nearest-neighbour to
    
[^281]: 重尾噪声下的无参数区间动态遗憾

    Parameter-Free Interval-Dynamic Regret under Heavy-Tailed Noise

    [https://arxiv.org/abs/2610.02258](https://arxiv.org/abs/2610.02258)

    本文提出了一种无需任何参数知识的在线凸优化算法，在重尾噪声（未知有限条件p阶矩，1<p≤2）下实现了区间动态遗憾界，达到了min(GDn, GD√(nΛ) + σDn^(1/p)Λ^((p-1)/p))的通用常数遗憾保证。

    

    我们研究在线凸优化问题，其中每轮仅获得一个无偏随机次梯度，且噪声的条件p阶矩有限但未知，其中1<p≤2。对于每个长度为n的固定区间I以及路径复杂度为Λ_I=1+P_I/D的比较路径，一个学习器达到：E[Regret_I(u)]≤min(GDn, C[GD√(n(Λ_I+log²(2T))) + σDn^(1/p)(Λ_I+log²(2T))^((p-1)/p)])。该学习器不使用G、σ、p、I、P_I中的任何一个参数，且常数为通用常数。区间自适应会增加比较路径复杂度，同时保留均值梯度和噪声的不同指数。该分析在期望意义上控制了校准，并限制了观测尺度变化的成本。其一般性定理可与对可预测可用专家分布的比较相关联，其中相对熵依赖于非均匀先验。共同先验有利于长窗口和长重启长度。在提供统计信息的情况下，区间成本变为1+log(T/n)。

    arXiv:2610.02258v1 Announce Type: new  Abstract: We study online convex optimization with one unbiased stochastic subgradient per round and an unknown finite conditional $p$th noise moment, $1<p\le2$. For every fixed interval $I$ of length $n$ and comparator path with $\Lambda_I=1+P_I/D$, one learner achieves   \[ E[Regret_I(u)]\le\min(GDn, C[GD\sqrt{n(\Lambda_I+\log^2(2T))} +\sigma Dn^{1/p}(\Lambda_I+\log^2(2T))^{(p-1)/p}]). \]   The learner uses none of $G,\sigma,p,I,P_I$, and the constant is universal. Interval adaptation adds to comparator complexity, preserving the distinct mean-gradient and noise exponents. The analysis controls calibration in expectation and limits the cost of observation-scale changes. Its general theorem compares to distributions over predictably available experts with relative-entropy dependence on a nonuniform prior. A common prior favors long windows and long restart lengths. With the statistics supplied, the interval cost becomes $1+\log(T/n)$, including t
    
[^282]: TRACE：一个包含官方运行文本的可复现电力价格预测基准

    TRACE: A Reproducible Benchmark for Electricity Price Forecasting with Official Operational Text

    [https://arxiv.org/abs/2610.02256](https://arxiv.org/abs/2610.02256)

    TRACE是一个将电力价格与预测截止时点官方运行文本配对、并严格防止信息泄露的可复现电力价格预测基准，它证明了文本上下文的预测价值——使时间序列基础模型的上尾pinball损失中位数降低7.4%。

    

    电力价格预测（EPF）为电力市场中的调度、竞价和风险管理提供支持，然而现有基准主要关注数值输入，对预测时点文本上下文的预测价值评估不足。我们提出了TRACE，这是一个包含7,300个区域-日样本的可复现基准，将美国某主要电力市场五个区域的价格与预测截止时点可获得的官方运行文本相配对。TRACE在每个截止时点重建官方运行文本，从而防止截止时点之后的信息泄露。我们从语义对齐和预测价值两方面对TRACE进行评估。语义评估结果与真实价格的中心走势及两种尾部风险保持一致，其中与价格上尾风险的一致性最强。预测价值体现在：各时间序列基础模型的上尾pinball损失中位数降低了7.4%。受控的跨日文本错配消融实验逆转了上述增益……

    arXiv:2610.02256v1 Announce Type: new  Abstract: Electricity price forecasting (EPF) supports scheduling, bidding, and risk management in electricity markets, yet existing benchmarks focus mainly on numerical inputs, leaving the forecasting value of forecast-time textual context insufficiently evaluated. We introduce TRACE, a reproducible benchmark of 7,300 zone--day instances pairing prices from five zones in a major U.S. market with official operational text available at the forecast cutoff. TRACE reconstructs official operational text at each cutoff, preventing post-cutoff information leakage. We evaluate TRACE for semantic alignment and forecasting value. Semantic assessments align with central movement and both tail risks in ground-truth prices, most consistently for upper-tail price risk. Forecasting value is reflected in a median 7.4\% reduction in upper-tail pinball loss across time-series foundation models. A controlled cross-day text-mismatch ablation reverses the gains, fall
    
[^283]: MACTS-EM：基于涌现记忆的多智能体协作时间序列预测

    MACTS-EM: Multi-Agent Collaborative Time Series Forecasting with Emergent Memory

    [https://arxiv.org/abs/2610.02255](https://arxiv.org/abs/2610.02255)

    提出了MACTS-EM框架，通过领域专业化智能体协作、元认知动态分配层和涌现记忆机制，实现跨领域模式迁移与多模态集成的时间序列预测。

    

    时间序列预测仍然是众多领域中的一项关键挑战。尽管已取得显著进展，现有方法在应对状态转变、跨领域知识迁移和多模态数据集成等复杂现象时仍存在困难。本文提出了基于涌现记忆的多智能体协作时间序列预测框架（MACTS-EM），这是一个由专业化智能体协作以实现卓越预测性能的新型框架。MACTS-EM架构集成了：（1）用于模式识别、异常检测、因果推断和不确定性量化的领域专业化预测智能体；（2）用于动态智能体分配的元认知层；（3）支持跨领域模式迁移的涌现记忆机制；（4）多模态上下文集成；以及（5）对抗鲁棒性组件。在金融市场、气候模式、能源消耗和疫情传播等方面的评估表明……

    arXiv:2610.02255v1 Announce Type: new  Abstract: Time series forecasting remains a critical challenge across numerous domains. Despite significant advancements, existing approaches struggle with complex phenomena such as regime shifts, cross-domain knowledge transfer, and multimodal data integration. This paper introduces Multi-Agent Collaborative Time Series Forecasting with Emergent Memory (MACTS-EM), a novel framework where specialised agents collaborate to achieve superior forecasting performance. The MACTS-EM architecture integrates: (1) domain-specialised forecasting agents for pattern recognition, anomaly detection, causal inference, and uncertainty quantification; (2) a meta-cognitive layer for dynamic agent allocation; (3) an emergent memory mechanism enabling cross-domain pattern transfer; (4) multimodal contextual integration; and (5) adversarial robustness components. Evaluation across financial markets, climate patterns, energy consumption, and pandemic propagation demonst
    
[^284]: 利用大语言模型克服解释结构建模的挑战

    Overcoming Challenges of Interpretive Structural Modeling with Large Language Models

    [https://arxiv.org/abs/2610.02254](https://arxiv.org/abs/2610.02254)

    本工作将大语言模型作为“不完美专家”引入解释结构建模（ISM），以克服传统专家交互方法繁琐且难以扩展至数百个变量的挑战，并通过对比实验证明逐行和全图因果图发现方法效果最佳。

    

    解释结构建模（ISM）是一种著名的多准则决策方法。ISM 相较于其他方法论的成功之处在于其能够建模因果关系、因素的二元尺度以及由此产生的层次化表示。传统上，建模过程需要与领域专家反复交互直到达成共识才能完成。这一过程繁琐且劳动密集，最重要的是限制了 ISM 扩展到包含数百个变量的研究的能力。借鉴现有的大语言模型（LLM）作为“不完美专家”进行因果图发现的研究成果，本工作探索了一种集成 LLM-ISM 的解释结构建模方法。研究比较并评估了成对、k-wise、逐行和全图四种因果图发现方法。结果表明，用于 ISM 的因果图发现方法在采用逐行方法（SHD=160，F1 分数=0.77）和全图方法（SHD=135，F1 分数=0.73）时表现最佳。

    arXiv:2610.02254v1 Announce Type: cross  Abstract: Interpretive Structural Modeling (ISM) is a well-known process for multi-criteria decision making. The success of ISM over other methodologies is its ability to model causal relationships, the binary scale of factors, and resulting hierarchical representation. Traditionally, the modeling process is performed by repeated interactions with subject matter experts until consensus is reached. This process is tedious, labor-intense, and most importantly limits the ability of ISM to scale to studies with hundreds of variables. Drawing on existing work of causal graph discovery with large language models (LLM) as imperfect experts, this work explores an integrated LLM-ISM approach for ISM. Pairwise, k-wise, rowwise, and full graph discovery methodologies are compared and evaluated. It is shown that causal graph discovery methods for ISM perform best using rowwise (SHD=160, F1-score=0.77) and full graph methods (SHD=135, F1-score=0.73).
    
[^285]: Dropout 神经网络的逼近性质：Sobolev 速率与置信界

    Approximation Property of Dropout Neural Networks: Sobolev Rates and Confidence Bounds

    [https://arxiv.org/abs/2610.02253](https://arxiv.org/abs/2610.02253)

    本文首次定量刻画了随机 Dropout（边以概率 p 独立保留）的 ReLU 网络以高概率一致逼近 Sobolev 空间单位球所需的网络规模，给出了常数深度下 $\widetilde O(p^{-9}\varepsilon^{-\max\{d/n,2\}}\log(1/\delta))$ 的规模上界以及基于 Sobolev 容量的相匹配下界。

    

    Dropout 神经网络的通用逼近性质本身并不能描述为实现精确随机化所需的网络规模。在本工作中，我们研究了以概率 $p$ 独立保留边的 ReLU 网络对 $W^{n,\infty}([0,1]^d)$ 单位球的逼近问题。逼近误差在整个输入域上一致度量，且该保证对单个采样网络以至少 $1-\delta$ 的概率成立。我们构造了深度为常数、规模为 $\widetilde O_{n,d}(p^{-9}\varepsilon^{-\max\{d/n,2\}} \log(1/\delta))$ 的网络。该构造结合了有界局部子网络、在成功逼近事件上的局部化处理以及多尺度泰勒分解。反之，Sobolev 容量对存活边数施加了下界，而对固定仿射函数的逼近则要求输出层的代价为 $((1-p)/p)\varepsilon^{-2}\log(1/\delta)$ 阶。

    arXiv:2610.02253v1 Announce Type: new  Abstract: The universal approximation property of dropout neural networks does not by itself describe the network size required for an accurate random realization. In this work, we study approximation of the unit ball of $W^{n,\infty}([0,1]^d)$ by ReLU networks whose edges are retained independently with probability $p$. The approximation error is measured uniformly over the input domain, and the guarantee holds with probability at least $1-\delta$ for a single sampled network. We construct networks of constant depth and size $\widetilde O_{n,d}(p^{-9}\varepsilon^{-\max\{d/n,2\}} \log(1/\delta))$. The construction combines bounded local subnetworks, localization on a successful approximation event, and a multiscale Taylor decomposition. Conversely, Sobolev capacity imposes a lower bound on the number of surviving edges, while approximation of a fixed affine function requires an output-layer cost of order $((1-p)/p)\varepsilon^{-2}\log(1/\delta)$ a
    
[^286]: 无需受控实验的科学模拟器反事实预测

    Counterfactual Predictions in Scientific Emulators Without Controlled Experiments

    [https://arxiv.org/abs/2610.02252](https://arxiv.org/abs/2610.02252)

    提出 ReRoute 框架，无需受控实验或模拟器数据，仅通过将查询输入固定为参考值并沿已知机制路径重新引入其变化，结合事实数据微调，即可让科学模拟器准确回答“如果条件不同会怎样”的反事实预测问题。

    

    许多科学问题需要对从未观测到的情况进行推理：如果条件、干预或历史有所不同会怎样？模型可以在已观测数据上做出准确预测，但当相互关联的输入被独立改变时，模型在这类“假设性”查询上往往会失效。一种常见的补救方法是加入受控仿真数据，使这些相关因素被显式解耦，但这需要访问模拟器、计算开销可能很高，并且会继承模拟器自身的建模假设。我们提出了 ReRoute，一个面向目标性科学“假设性”预测的框架，它将事实数据与部分机制知识相结合，无需受控干预数据即可完成适配。ReRoute 将预训练骨干网络中被查询的输入固定到一个参考值，通过已知的机制路径重新引入其变化，并在原始事实数据上进行微调，同时将下游效应留给学习到的动力学模型。

    arXiv:2610.02252v1 Announce Type: cross  Abstract: Many scientific questions require reasoning about what was never observed: What if the conditions, interventions, or history had been different? Models can predict accurately on observed data yet fail on such what-if queries when correlated inputs are varied independently. A common remedy is to add controlled simulation data in which these factors are explicitly disentangled, but this requires access to a simulator, can be computationally expensive, and inherits the simulator's modeling assumptions. We introduce ReRoute, a framework for targeted scientific what-if prediction that combines factual data with partial mechanistic knowledge, without requiring controlled intervention data for adaptation. ReRoute fixes the queried input of a pretrained backbone to a reference value, reintroduces its variation through a known mechanistic pathway, and fine-tunes on the original factual data, while leaving downstream effects to the learned dynam
    
[^287]: 面向扩散草稿树的秩感知推测采样

    Rank-Aware Speculative Sampling for Diffusion Draft Trees

    [https://arxiv.org/abs/2610.02251](https://arxiv.org/abs/2610.02251)

    提出秩感知推测采样（RASS），无需额外目标模型评估即可对草稿候选进行秩感知排序并以优化权重采样，从而更高效地验证扩散草稿树并加速扩散生成。

    

    推测采样通过并行验证廉价的草稿状态来加速扩散生成，同时保持目标分布规律。近期基于树的方法比单链草稿更有效地分配并行计算预算，扩散贪心拒绝采样（D-GRS）即是例证。D-GRS在每个节点生成K个条件独立的候选，并按其生成顺序依次进行测试。然而，这些采样得到的候选无需额外的目标模型评估即可获得富有信息量的排序。为利用这一点，我们提出了秩感知推测采样（RASS），一种基于秩感知列表耦合的推测草稿树验证规则。RASS按照提议与目标之间的均值位移对草稿候选进行排序，并以优化后的权重采样一个秩，以最小化被选中的提议分布与目标分布之间的总变差。最后，被选中的候选与目标进行最大耦合……

    arXiv:2610.02251v1 Announce Type: new  Abstract: Speculative sampling accelerates diffusion generation by verifying inexpensive draft states in parallel while preserving the target law. Recent tree-based methods allocate the parallel compute budget more effectively than single-chain drafts, as demonstrated by Diffusion Greedy Rejection Sampling (D-GRS). D-GRS generates $K$ conditionally independent candidates per node, and sequentially tests them in their generation order. Yet the sampled candidates admit an informative ranking without additional target-model evaluations. To exploit this, we introduce Rank-Aware Speculative Sampling (RASS), a verification rule for speculative draft trees based on rank-aware list coupling. RASS orders draft candidates along the proposal-target mean displacement and samples a rank with weights optimized to minimize total variation between the selected-proposal and target laws. Finally, the selected candidate is maximally coupled with the target, with res
    
[^288]: 不同假设下基于最近邻方法从MS/MS谱图预测分子指纹的基线研究

    Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions

    [https://arxiv.org/abs/2610.02249](https://arxiv.org/abs/2610.02249)

    该论文系统比较了在不同推理信息假设下的多种最近邻检索变体用于从MS/MS谱图预测分子指纹，旨在建立更严格的基线以实现更严谨的基准测试并更好地衡量领域进展。

    

    最近的研究表明，最近邻检索为从MS/MS谱图预测分子指纹提供了一个强有力的基线方法，其若干变体能够匹敌甚至超越当前的深度学习模型（Khoo and Barzilay, 2026; Liu et al., 2026; Gupta et al., 2026）。值得注意的是，“最近邻”涵盖了一系列检索方法，这些方法在推理阶段对可用信息的假设有所不同。在本报告中，我们系统地比较了几种最近邻变体，并展示了这些不同的假设如何影响性能。我们的目标是建立更严格的基线，从而实现更严谨的基准测试，并更好地衡量该领域的研究进展。

    arXiv:2610.02249v1 Announce Type: new  Abstract: It has recently been shown that nearest-neighbour retrieval provides a strong baseline for molecular fingerprint prediction from MS/MS spectra, with several variants matching or outperforming current deep learning models (Khoo and Barzilay, 2026; Liu et al., 2026; Gupta et al., 2026). Importantly, "nearest neighbour" encompasses a family of retrieval methods that differ in the information assumed to be available at inference. In this report, we systematically compare several nearest-neighbour variants and show how these differing assumptions affect performance. Our goal is to establish stricter baselines that enable more rigorous benchmarking and better measure progress in this area.
    
[^289]: 面向地表预测中非平稳偏差的状态空间遗忘

    State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting

    [https://arxiv.org/abs/2610.02248](https://arxiv.org/abs/2610.02248)

    本文提出首个专为地球科学Mamba状态空间模型设计的机器遗忘框架SSU-LSF，通过专用影响函数、时间混杂足迹定位与KL散度信任域约束的梯度上升，系统性地消除非平稳混杂事件（如未记录灌溉、大坝调度变化）对地表预测造成的持续隐性偏差。

    

    基于Mamba家族结构化状态空间模型（SSM）构建的业务化地表预测系统，会将非平稳混杂事件（如未记录的灌溉激增、大坝调度变化、传感器重新校准）吸收到其状态转移矩阵中，导致在物理成因结束后的很长时间内，NDVI、地表温度（LST）和作物物候预测仍然存在隐性偏差。本文提出了SSU-LSF（面向地表预测的状态空间遗忘），这是首个专为地球科学领域基于Mamba的状态空间模型设计的机器遗忘框架。我们通过闭式矩阵指数梯度开发了专门针对Mamba状态矩阵的EKFac影响函数，利用谱半径加权的肘部阈值方法定位时间混杂足迹$\Phi$，并在由KL散度信任域和空间全变差（TV）正则化共同约束的框架内应用无Hessian投影梯度上升。命题1证明了残余混杂效应是有界的……

    arXiv:2610.02248v1 Announce Type: new  Abstract: Operational land surface forecasting systems built on Mamba-family Structured State Space Models absorb non-stationary confounding events (unrecorded irrigation booms, dam-operation shifts, sensor recalibrations) into their state-transition matrices, silently biasing NDVI, LST, and crop phenology predictions long after the physical cause ends. This paper introduces SSU-LSF (State-Space Unlearning for Land Surface Forecasting), the first machine-unlearning framework purpose-built for geoscientific Mamba-based SSMs. We develop EKFac influence functions specialized to the Mamba state matrices via a closed-form matrix-exponential gradient, use spectral-radius-weighted elbow thresholding to localize a temporal confounding footprint $\Phi$, and apply Hessian-free projected gradient ascent within a KL-divergence trust region augmented by spatial total-variation (TV) regularization. Proposition 1 establishes that residual confounding is bounded 
    
[^290]: 缺失的是潜变量，而非模拟器：面向真实JWST光谱反演的半径增广推断

    A Missing Latent, Not a Missing Simulator: Radius-Augmented Inference for Real JWST Retrieval

    [https://arxiv.org/abs/2610.02245](https://arxiv.org/abs/2610.02245)

    该研究发现SBI在真实JWST光谱上失效的根本原因不是模拟器物理缺失，而是缺少行星半径这一潜变量，并提出半径增广的流匹配后验模型MIRAGE来解决这一问题。

    

    基于模拟的摊销推断（SBI）在辐射传输模拟器数据上训练后，能够准确地从合成的詹姆斯·韦布空间望远镜（JWST）光谱中恢复系外行星大气参数，但在处理真实还原光谱时却会失效。流模型后验在真实的WASP-39b数据上发生了坍塌（重要性抽样有效样本量ESS=1，最佳拟合χ²/N=301）。这类失败通常被归咎于正向模型物理的缺失，但通过嵌套采样先行排除这些因素（温度梯度、SO₂不透明度以及高保真不透明度集合）后，拟合结果并无改变，这表明坍塌并非源于模拟器的错误设定，而是源于一个缺失的潜变量——行星半径。为了解决这一问题，我们构建了MIRAGE，一个半径增广的流匹配后验模型，通过重要性抽样和最优传输映射与独立的嵌套采样参考结果进行校准，得到了物理合理且与文献一致的结果。

    arXiv:2610.02245v1 Announce Type: cross  Abstract: Amortized simulation-based inference (SBI), which is trained on radiative-transfer simulators, recovers exoplanet atmospheres accurately on synthetic James Webb Space Telescope (JWST) spectra but collapses when it comes to real reduced spectra. The flow posterior collapsed on real WASP-39b (importance-sampling effective sample size (ESS) = 1, best-fit \c{hi}2/N = 301), and such a failure is usually blamed on missing the forward-model physics, but ruling these levers out with nested sampling first (a temperature gradient, SO2 opacity, and a high-fidelity opacity set) leaves the fit unchanged, meaning the collapse is not from the simulator misspecification but instead from a missing latent, the planet radius. To fix this, we build MIRAGE, a radius-augmented flow-matching posterior calibrated against an independent nested-sampling reference with importance sampling and an optimal-transport map, which yields a physical and literature-consi
    
[^291]: 面向万亿规模混合专家模型的硬件原生联合稀疏-量化方法

    Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts

    [https://arxiv.org/abs/2610.02241](https://arxiv.org/abs/2610.02241)

    提出了一个端到端的软硬件协同设计框架，通过连续重参数化实现稀疏性与量化的可微联合优化，将万亿规模MoE的专家权重压缩为硬件原生的低精度半结构化稀疏表示，从而在稀疏张量核心上加速执行并降低部署的内存瓶颈。

    

    混合专家（MoE）架构使前沿语言模型能够扩展到万亿参数规模，但其部署受到庞大内存占用和内存带宽限制的制约。尽管现代加速器提供了稀疏张量核心，能够通过低精度半结构化稀疏性来减少权重存储并提升吞吐量，但由于显著的模型质量退化以及缺乏分组稀疏GEMM原语，将其应用于MoE仍然具有挑战性。我们提出了一个端到端的软硬件协同设计框架，将专家权重压缩为硬件原生的低精度稀疏表示，并在SpTC上加速其执行。在算法层面，我们的框架通过连续重参数化松弛离散的半结构化支撑选择，在路由器加权的重建目标下实现与量化权重的可微联合优化……

    arXiv:2610.02241v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures allow frontier language models to scale to trillions of parameters, but their deployment is constrained by massive memory footprints and memory-bandwidth limitations. Although modern accelerators provide Sparse Tensor Cores (SpTCs) that reduce weight storage and increase throughput through low-precision semi-structured sparsity, exploiting them for MoEs remains challenging because of substantial model-quality degradation and the lack of grouped sparse GEMM primitives. We present an end-to-end hardware-software co-design framework that compresses expert weights into hardware-native, low-precision sparse representations and accelerates their execution on SpTCs. Algorithmically, our framework relaxes discrete semi-structured support selection through continuous reparameterization, enabling differentiable joint optimization with quantized weights under a router-weighted reconstruction objective and sc
    
[^292]: 用于跨上下文KV缓存复用的预算化缓存修复

    Budgeted Cache Repair for Cross-Context KV-Cache Reuse

    [https://arxiv.org/abs/2610.02233](https://arxiv.org/abs/2610.02233)

    该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。

    

    跨上下文KV缓存复用是在新前缀下预测共享片段的键和值，而不是重新计算它们，并且据报告这样做不会带来质量损失。我们的发现并非如此，并识别出两个问题。（1）一个隐性代价：在MMLU和GSM8K上，复用会导致显著的准确率损失。（2）决策单元错误：目前没有任何决定是否复用缓存的规则能消除这一代价。真正有帮助的是选择缓存中哪些部分需要重新计算，而精准选择的收益随着选择单元的增大而下降：在单行（即单个token的键和值）层面，有依据的选择能消除超出随机水平49.5%的缓存误差；在64个token的分块层面为10.6%；而在整个调用层面则毫无效果。预算化缓存修复（BCR）正是在选择仍有收益的单元上进行操作。它从组装好的缓存中起草两个token，根据这两个token对缓存行支付的注意力对缓存行进行排序，并以三种布局之一精确地重新计算固定预算数量的缓存行。

    arXiv:2610.02233v1 Announce Type: cross  Abstract: Cross-context KV-cache reuse predicts a shared segment's keys and values under a new prefix instead of recomputing them, and has been reported to do so without quality loss. We find otherwise, and identify two problems. (1) A hidden cost: on MMLU and GSM8K, reuse costs substantial accuracy. (2) A decision at the wrong unit: no rule for deciding whether to reuse a cache removes that cost. What does help is choosing which parts of the cache to recompute, and the value of choosing well falls as the unit of choice grows: informed selection removes 49.5% of the cache error beyond chance at single rows (one token's keys and values), 10.6% at 64-token chunks, and nothing at the level of whole calls. Budgeted Cache Repair (BCR) acts at the unit where selection still pays. It drafts two tokens from the assembled cache, ranks cache rows by the attention those tokens pay them, and recomputes a fixed budget of rows exactly, in one of three layouts
    
[^293]: 使用能量引导流匹配实现泛化性的单细胞扰动响应预测

    Generalizable single-cell perturbation response prediction using energy-guided flow matching

    [https://arxiv.org/abs/2610.02232](https://arxiv.org/abs/2610.02232)

    提出scEGFlow框架，通过能量引导的流匹配动态建模单细胞从对照到扰动状态的连续转变，无需重新训练即可适应新的扰动条件，在成像表型和转录组基准测试中超越现有方法。

    

    在单细胞分辨率下预测扰动对表型和转录的响应，为探究生物系统提供了强大的工具。然而，现有方法通常依赖于训练过程中学习到的固定映射，使得在推理时难以校准分布偏移或适应新的扰动条件。在这里，我们提出了scEGFlow，一个能够动态连接对照细胞状态与受扰动细胞状态的能量引导流匹配框架。scEGFlow使用条件流匹配来建模从对照细胞群到扰动状态的连续转变，然后应用特定条件的能量梯度来校正和引导这些预测，从而无需重新训练流模型即可实现灵活调整。在涵盖成像表型和转录组图谱的多个基准数据集上的评估表明，scEGFlow在重建响应分布方面优于现有方法，无论在……

    arXiv:2610.02232v1 Announce Type: cross  Abstract: Predicting phenotypic and transcriptional responses to perturbations at single-cell resolution provides a powerful tool for probing biological systems. However, existing methods typically rely on fixed mappings learned during training, making it challenging to calibrate distribution shifts or adapt to novel perturbation conditions during inference. Here, we present scEGFlow, an energy-guided flow matching framework that dynamically bridges control and perturbed cellular states. scEGFlow models continuous transitions from control cell populations to perturbed states using conditional flow matching. It then applies condition-specific energy gradients to correct and steer these predictions, enabling flexible adjustments without retraining the flow model. Evaluations across benchmarks spanning imaging phenotypes and transcriptomic profiles show that scEGFlow outperforms existing methods in reconstructing response distributions under both s
    
[^294]: 漂绿的代价：基于保形机器学习的算法验证与市场纪律

    The Price of Greenwashing: Algorithmic Verification and Market Discipline using Conformal Machine Learning

    [https://arxiv.org/abs/2610.02225](https://arxiv.org/abs/2610.02225)

    该论文将SEC财务数据与EPA设施级排放数据融合，利用梯度提升模型和Mondrian保形预测构建了具有数学保证的排放散度指标，客观量化企业“漂绿”行为，并揭示了金融市场会对此种算法检测到的排放偏差进行定价的纪律机制。

    

    尽管企业可持续发展监管要求不断扩大，但对自我报告排放数据的系统性依赖使金融市场暴露于普遍存在的“漂绿”行为之中。当前文献主要依赖主观的ESG评级或文本情感分析，在客观量化物理气候现实方面留下了关键的经济计量学空白。为解决这一信息不对称问题，我们将美国SEC财务基本面数据与设施级别的EPA温室气体登记数据相融合，建立了具有数学保证的企业物理排放基线。利用梯度提升架构和Mondrian保形预测，我们将自我报告数据与该算法基线之间的偏差量化为一种新颖的保形加权连续散度指标。通过横截面领先-滞后的经济计量设计对该散度进行评估，我们揭示了稳健的市场纪律机制：算法排放散度……

    arXiv:2610.02225v1 Announce Type: new  Abstract: While corporate sustainability mandates are expanding, the systemic reliance on self-reported emissions data exposes financial markets to pervasive greenwashing. Current literature relies heavily on subjective ESG ratings or textual sentiment analysis, leaving a critical econometric gap in objectively quantifying physical climate realities. To resolve this information asymmetry, we fuse U.S. SEC financial fundamentals with facility-level EPA greenhouse gas registries to establish a mathematically guaranteed baseline of physical corporate emissions. Leveraging a gradient boosting architecture and Mondrian Conformal Prediction, we quantify the shortfall between self-reported data and this algorithmic baseline into a novel Conformal-Weighted Continuous Divergence (CWCD) metric. Evaluating this divergence via a cross-sectional lead-lag econometric design, we uncover a robust mechanism of market discipline: algorithmic emissions divergence ex
    
[^295]: 结合生成式特征增强的混合机器学习辅助拉曼光谱用于药物识别

    Hybrid Machine Learning-Assisted Raman Spectroscopy with Generative Feature Augmentation for Pharmaceutical Identification

    [https://arxiv.org/abs/2610.02224](https://arxiv.org/abs/2610.02224)

    该论文提出了HyMLRaman混合拉曼光谱框架，通过EfficientNet-B3深度特征提取、生成式特征增强与经典机器学习分类器（如SVM）相结合，实现了对六种药物残留的快速准确识别。

    

    快速、可靠地识别药物残留对于保障公众健康、确保食品安全以及实现实用的基于拉曼光谱的筛查具有重要意义。在本研究中，我们提出了HyMLRaman，这是一种混合拉曼光谱框架，它结合了深度光谱特征提取、生成模型和经典机器学习分类器，用于识别六种药物化合物，包括阿莫西林、氯霉素、环丙沙星、四环素、布洛芬和对乙酰氨基酚。拉曼光谱被转换为光谱图像，并使用多种深度神经网络骨干网络进行编码，其中EfficientNet-B3产生了最有效的特征表征。随后，将所得的1536维嵌入向量用于训练下游分类器，包括支持向量机（SVM）、K近邻（KNN）、逻辑回归、随机森林、XGBoost和人工神经网络（ANN），并采用分层10折交叉验证进行评估。混合的EfficientNet-B3-SVM配置取得了最强的基线性能……

    arXiv:2610.02224v1 Announce Type: new  Abstract: Rapid and reliable identification of pharmaceutical residues is important for safeguarding public health, ensuring food safety, and enabling practical Raman-based screening. In this study, we propose HyMLRaman, a hybrid Raman spectroscopy framework that combines deep spectral feature extraction, generative models, and classical machine-learning classifiers to identify six pharmaceutical compounds, including amoxicillin, chloramphenicol, ciprofloxacin, tetracycline, ibuprofen, and paracetamol. Raman spectra are converted into spectral images and encoded with several deep neural-network backbones, among which EfficientNet-B3 yields the most effective representation. The resulting 1536-dimensional embeddings are then used to train downstream classifiers, including SVM, KNN, logistic regression, random forest, XGBoost, and ANN, using stratified 10-fold cross-validation. The hybrid EfficientNet-B3--SVM configuration achieves the strongest bas
    
[^296]: RINS：面向大型稀疏线性系统的残差图像神经子空间求解器

    RINS: Residual-Image Neural Subspace Solvers for Large Sparse Linear Systems

    [https://arxiv.org/abs/2610.02217](https://arxiv.org/abs/2610.02217)

    提出Gate-RINS神经子空间求解器，通过缓存的残差探测生成多项式校正基并用轻量级逐点门控调制，在六个PDE基准任务上比GMRES和纯图神经基线更快达到固定相对残差阈值。

    

    来自偏微分方程离散化的大型稀疏线性系统需要校正子空间，其算子像空间能够解释当前残差。我们研究了这一残差图像视角，并提出了Gate-RINS——一种神经子空间求解器，它从缓存的残差探测中生成多项式校正基，并通过一个轻量级的、依赖于残差和坐标的逐点门控对其进行调制。投影最小二乘更新保持不变，因此神经组件仅负责选择展开方向，而数值闭合则通过range(AQ_t)对其进行检验。我们还引入了一种混合控制器调度方案，在相同的投影求解器框架下将GRANS风格的图控制器与Gate-RINS组合使用。在六个基于PDE的基准任务和两种规模上，Gate-RINS在大多数设置中比GMRES和近期的纯图神经基线在同步的实际运行时间内更快地达到固定的相对残差阈值，而混合调度……

    arXiv:2610.02217v1 Announce Type: cross  Abstract: Large sparse linear systems from PDE discretizations require correction subspaces whose operator images explain the current residual. We study this residual-image viewpoint and propose Gate-RINS, a neural subspace solver that generates polynomial correction bases from cached residual probes and modulates them with a lightweight residual- and coordinate-dependent pointwise gate. The projected least-squares update remains unchanged, so the neural component only chooses the expansion directions while the numerical closure tests them through \(\operatorname{range}(AQ_t)\). We also introduce a hybrid controller schedule that composes GRANS-style graph controllers with Gate-RINS under the same projected solver. Across six PDE-derived benchmark tasks and two scales, Gate-RINS reaches fixed relative-residual thresholds faster in synchronized wall-clock time than GMRES and a recent graph-only neural baseline in most settings, and hybrid schedul
    
[^297]: MiDShip：面向工程设计的船舶货舱结构多模态数据集

    MiDShip: Multimodal Dataset of Ship Cargo Hold Structures for Engineering Design

    [https://arxiv.org/abs/2610.02214](https://arxiv.org/abs/2610.02214)

    本文提出了MiDShip多模态数据集，包含12,753个船舶货舱结构设计（涵盖参数数据、3D几何、工程图纸、材料清单和结构评估），填补了船舶结构设计中缺乏连接几何、性能与规范约束的结构化数据集的空白。

    

    船舶结构决定着船体的强度、安全性和可制造性，但其设计必须满足数百项船级社规范要求，使得设计过程复杂且需要反复迭代。数据驱动方法受到缺乏结构化数据集的限制，即缺少能够将设计几何、结构性能与基于规范的约束条件相联系的数据集。本文提出了MiDShip，一个包含12,753个合成货舱结构设计的多模态数据集：其中6,020个为随机生成，496个由受SGLD启发的程序生成，6,237个由基于方程的修复程序生成。每个设计包含参数化数据、完整及可直接划分网格的3D几何模型、工程图纸及标注、材料清单以及初步结构评估。此外，还评估了从ABS MVR（钢质海船入级规范）子集推导出的25项约束条件。结果显示，所有随机生成的设计均无法满足全部约束条件；在受SGLD启发的设计中，322个（64.9%）设计完全符合规范要求，平均……

    arXiv:2610.02214v1 Announce Type: cross  Abstract: Ship structures govern vessel strength, safety, and manufacturability, but their design must satisfy hundreds of classification society requirements, making the process complex and iterative. Data-driven approaches are limited by the lack of structured datasets linking design geometry, structural performance, and rule-based constraints. This paper presents MiDShip, a multimodal dataset of 12,753 synthetic cargo-hold structural designs: 6,020 random, 496 generated by an SGLD-inspired procedure, and 6,237 generated by an equation-informed repair procedure. Each design includes parametric data, full and mesh-ready 3D geometry, engineering drawings and annotations, a bill of materials, and preliminary structural evaluations. Twenty-five constraints derived from a subset of ABS MVR are also evaluated.   None of the random designs satisfies all constraints. Among the SGLD-inspired designs, 322 (64.9%) were fully compliant, with an average of
    
[^298]: 通用字节级编码：通过UTF-8/UTF-16路由减少跨脚本令牌预算差异

    Universal Byte-Level Encoding: UTF-8/UTF-16 Routing to Reduce Cross-Script Token-Budget Disparities

    [https://arxiv.org/abs/2610.01984](https://arxiv.org/abs/2610.01984)

    提出通用字节级编码（UBE）双字母表分词器，将1-2字节UTF-8字符保留在UTF-8路径上、将3-4字节字符改经UTF-16路由，从而降低多语言脚本中非英语文字的编码底线，减少跨脚本间的令牌预算差异。

    

    字节级字节对编码（BBPE）分词器因覆盖所有Unicode文本而非常适合多语言大语言模型（LLM）。然而，在基于UTF-8的BBPE中，许多文字脚本比英语面临更高的起步回退成本：当没有已学习的合并规则可应用时，一个多字节字符需要多个字节衍生的符号。我们将这种最坏情况下的合并前成本称为“编码底线”。更高的编码底线会增加令牌数量和单次请求成本，并缩减可用上下文长度。改变文本编码可以缩小这一差距，但单一的全局编码会使混合脚本文本中原本已经高效的英语片段变得更昂贵。我们提出通用字节级编码（UBE），一种双字母表分词器，它将1-2字节的UTF-8字符保留在UTF-8路径上，同时将3-4字节的UTF-8字符通过UTF-16进行路由。这降低了具有高令牌溢价的脚本中3字节基本多文种平面（BMP）字符的编码底线（至……

    arXiv:2610.01984v1 Announce Type: cross  Abstract: Byte-level byte-pair encoding (BBPE) tokenizers are attractive for multilingual large language models (LLMs) because they cover all Unicode text. In UTF-8-based BBPE, however, many scripts start from a higher fallback cost than English: when no learned merges can be applied, a multibyte character requires multiple byte-derived symbols. We call this worst-case pre-merge cost the encoding floor. A higher floor can increase token counts and per-request cost and shrink usable context. Changing the text encoding can reduce this gap, but a single global encoding can make already-efficient English spans more expensive in mixed-script text. We propose Universal Byte-Level Encoding (UBE), a dual-alphabet tokenizer that keeps 1-2-byte UTF-8 characters on the UTF-8 path while routing 3-4-byte UTF-8 characters through UTF-16. This lowers the encoding floor for 3-byte Basic Multilingual Plane (BMP) characters in scripts with high token premiums (to
    
[^299]: 使用格林观测算子学习子流形间的偏微分方程动力学

    Learning PDE Dynamics between Submanifolds Using Green's Observation Operators

    [https://arxiv.org/abs/2610.01697](https://arxiv.org/abs/2610.01697)

    本文提出格林观测算子（GObO），将固定的环境介质一次性映射为线性PDE在源与观测子流形上的格林核，使每个新源仅需一次低维积分而无需网络评估，并证明了所得有限流式状态的稳定性与近似速率。

    

    许多物理系统仅在较大空间域的低维子流形上被驱动和观测，而其动力学由占据该空间域的环境介质所支配。典型例子包括由红外相机成像的激光加热部件，以及在传感器平面上测量的地面排放。然而，全域求解器尽管只需要观测子流形上的结果，却要为每个新源计算整个体积；而黑盒代理模型也没有利用环境介质保持不变这一事实。我们提出了格林观测算子，它将环境介质一次性映射到限制在源子流形与观测子流形上的线性偏微分方程的格林核。此后，对于每个新源，计算代价仅为一次低维积分，无需任何网络评估。核中的指数衰减速率产生了一个精确的有限流式状态，其内存与时间范围无关；我们证明了该状态的稳定性以及相应的近似速率

    arXiv:2610.01697v1 Announce Type: new  Abstract: Many physical systems are driven and observed only on lower-dimensional submanifolds of a larger spatial domain, while their dynamics are governed by the ambient medium occupying that domain. Examples include laser-heated parts imaged by an infrared camera, and ground-level emissions measured on a sensor plane. Full-domain solvers, however, compute the entire volume for every new source although only the observation submanifold is needed, and black-box surrogates do not exploit that the ambient medium remains fixed. We introduce the \emph{Green's Observation Operator (GObO)}, which maps the ambient medium once to the Green's kernel of a linear PDE restricted to the source and observation submanifolds. New sources then cost one lower-dimensional integral and no network evaluation. Exponential rates in the kernel yield an exact finite streaming state with horizon-independent memory; we prove its stability and an approximation rate for the 
    
[^300]: 基于神经进化的持续强化学习

    Continual Reinforcement Learning with Neuroevolution

    [https://arxiv.org/abs/2610.01583](https://arxiv.org/abs/2610.01583)

    该研究发现，在持续变化的强化学习任务中，神经进化方法（尤其是进化策略ES）能在适应与遗忘之间取得最稳定的平衡，其原因是ES能在回报景观中找到具有最宽邻域的解。

    

    尽管针对持续任务变化下强化学习（RL）可塑性丧失的原因与补救方法已有大量研究，但目前尚无任何RL方法能够持续地在适应与遗忘之间取得良好平衡。本文转向另一种优化范式——神经进化（NE）：即通过在神经网络种群上进行变异与选择、直接在权重空间中搜索的算法。在涵盖广泛环境与环境变化的实验中，策略参数规模从几百个参数到百万参数网络不等，我们将进化策略（ES）和遗传算法（GA）与最先进的持续RL变体及基于种群的RL方法进行了比较。结果发现，ES 最能始终如一地实现良好的稳定性-可塑性权衡，而 GA 是最具可塑性的方法，但比 ES 遗忘得更多。为解释这一现象，我们研究了每种方法所得解周围的回报景观。ES 找到了最宽的邻域，即……

    arXiv:2610.01583v1 Announce Type: cross  Abstract: Despite many studies about causes and remedies of plasticity loss in Reinforcement Learning (RL) under continual task changes, no RL method has yet consistently achieved a good balance between adaptation and forgetting. Here we turn to an alternative optimization paradigm, neuroevolution (NE): algorithms that search directly in weight space through mutation and selection over a population of neural networks. Across a wide array of environments and environmental changes, with policies ranging from a few hundred parameters to million-parameter networks, we compare evolution strategies (ES) and genetic algorithms (GAs) against state-of-the-art continual RL variants and population-based RL. ES most consistently achieves a good stability-plasticity trade-off, while the GA is the most plastic method but forgets more than ES. To explain this, we study the return landscape around each method's solutions. ES finds the widest neighborhoods, i.e.
    
[^301]: Range-GRPO：基于奖励区间两两关系的策略优化

    Range-GRPO: Policy Optimization via Pairwise Relations among Reward Intervals

    [https://arxiv.org/abs/2610.01548](https://arxiv.org/abs/2610.01548)

    提出 Range-GRPO 半监督后训练框架，通过将 LLM-as-a-Judge 的伪奖励表示为保形校准的奖励区间，并在 GRPO 中以区间两两比较替代点奖励比较，使奖励不确定性能够影响学习信号的大小与方向。

    

    随着大语言模型（LLM）应用的不断扩展，后训练在使其适应下游任务方面变得日益重要。然而，获取可靠的监督信号仍然成本高昂，尤其是在缺乏参考答案或可执行验证器的领域。LLM-as-a-Judge 能够为无标签响应提供可扩展的伪奖励，但单一的分值无法显式表达奖励的不确定性。这促使我们将伪奖励表示为经保形校准的奖励区间。我们提出了 Range-GRPO，这是一种将有限的标注数据与无标签提示相结合的半监督后训练框架。在组相对策略优化（GRPO）中，学习信号依赖于每个采样组内的相对奖励比较。所提出的目标函数对奖励区间进行两两比较，而非将其简化为点奖励，从而使区间的不确定性能够影响这些学习信号的大小和方向。我们的理论……

    arXiv:2610.01548v1 Announce Type: new  Abstract: As the use of large language models (LLMs) expands, post-training has become increasingly important for adapting them to downstream tasks. However, obtaining reliable supervision remains costly, especially in domains without reference answers or executable verifiers. LLM-as-a-Judge provides scalable pseudo-rewards for unlabeled responses, but a single point score does not explicitly represent reward uncertainty. This motivates representing pseudo-rewards as conformally calibrated reward ranges. We propose Range-GRPO, a semi-supervised post-training framework that combines limited labeled data with unlabeled prompts. In Group Relative Policy Optimization (GRPO), learning signals depend on relative reward comparisons within each rollout group. The proposed objective compares reward ranges pairwise rather than reducing them to point rewards, allowing interval uncertainty to affect both the magnitude and direction of these signals. Our theor
    
[^302]: 基于转录孟加拉语文本的开放词汇单词识别

    Open Vocabulary Word Recognition From Transcribed Bangla Texts

    [https://arxiv.org/abs/2610.01134](https://arxiv.org/abs/2610.01134)

    本研究通过结合基于MobileNetV2的SSD、基于InceptionResNetV2的Faster R-CNN及其集成模型，并引入改进的非极大值抑制方法，实现了手写孟加拉语开放词汇单词识别。

    

    光学字符识别（OCR）技术能够扫描纸质文件并提取文本，使人们的工作更加轻松。尽管软件行业已有多种OCR系统可用，但要为孟加拉语找到可靠的等效解决方案却需要付出大量努力，而对手写文本而言，情况则更为特殊。从单词图像中识别单词是任何OCR过程中最关键的阶段，它是在从文本图像中分割出单词之后的第二个阶段。如果这一阶段失败，无论其他阶段表现多么出色，OCR的整体性能都会很差。本研究旨在利用深度学习对手写孟加拉语单词图像进行单词识别。研究中使用了三种目标检测模型——基于MobileNetV2的SSD、基于InceptionResNetV2的Faster R-CNN，以及这两个模型的集成模型——来训练和测试手写单词图像，并引入了一种改进的非极大值抑制（NMS）方法以提升识别的有效性。

    arXiv:2610.01134v1 Announce Type: cross  Abstract: An optical character recognition (OCR) can scan a paper and extract text using technology, making people's jobs easier. While various OCR systems are available in the software industry, finding a reliable equivalent solution for Bangla takes much work. When it comes to handwritten texts, the situation is much more unusual. Recognizing words from word images is the most critical stage in any OCR process. It is the second stage after segmenting words from text pictures. If this stage fails, the overall performance of the OCR will be poor, regardless of how well the other phases perform. This study aims to recognize words using deep learning in a handwritten Bangla word image. Three object detection models, SSD with MobileNetV2, Faster R-CNN with InceptionResNetV2, and an ensemble model of these two, have been used to train and test handwritten word images. A modified Non-Maximum Suppression has been introduced to enhance the effectivenes
    
[^303]: 适配器丛林：拆分RLVR预算胜过集中使用

    Adapter Thickets: Splitting an RLVR Budget Beats Concentrating It

    [https://arxiv.org/abs/2610.00991](https://arxiv.org/abs/2610.00991)

    用RLVR预算训练单个LoRA适配器虽能提升单样本准确率，却会让采样错误越来越相关从而损害多数投票效果，而将RLVR预算拆分到多个适配器（“适配器丛林”）上再投票的效果优于将其集中在一个适配器上。

    

    对采样补全结果进行多数投票是测试时扩展（test-time scaling）的主力方法，而带可验证奖励的强化学习（RLVR）则是让每次补全变得更好的主力方法。标准流程将二者组合：先用RLVR训练一个策略，然后对其进行多次采样并投票。我们证明这种组合是有损的。投票只能推翻投票者之间并不共有的错误，而RLVR会使策略锐化，导致其采样结果越来越倾向于犯相同的错误。在每种方法对每个问题恰好抽取160个补全的条件下，在全部RLVR预算上训练单个LoRA适配器确实提高了我们测试的所有模型（1.5B-8B）的单样本准确率。然而，在四个模型中的三个上，它使多数投票的准确率反而低于未经训练的基础模型，最多低4.8个百分点。这种损害在训练过程中不断累积：投票者之间的错误相关性稳步上升，多数投票准确率在训练早期达到峰值后，最多下降7.0个百分点。

    arXiv:2610.00991v1 Announce Type: new  Abstract: Majority voting over sampled completions is the workhorse of test-time scaling, and reinforcement learning with verifiable rewards (RLVR) is the workhorse for making each completion better. The standard pipeline composes the two: train one policy with RLVR, then sample it many times and vote. We show that this composition is lossy. A vote can only overturn mistakes that its voters do not share, and RLVR sharpens a policy so that its samples increasingly make the same mistakes. With every method drawing exactly $160$ completions per problem, training a single LoRA adapter on the full RLVR budget raises single-sample accuracy on every model we test ($1.5$B-$8$B). Yet on three of four models it leaves the majority vote below that of the untrained base model, by up to $4.8$ points. The damage builds during training: voter errors grow steadily more correlated, and the majority vote accuracy peaks early before falling by up to $7.0$ points. Th
    
[^304]: 柏拉图式任务算术

    Platonic Task Arithmetic

    [https://arxiv.org/abs/2610.00929](https://arxiv.org/abs/2610.00929)

    本文提出“柏拉图任务向量”概念，并引入形状与模型架构和嵌入维度无关的“通用任务描述符”矩阵，使任务算术（如任务加法与取反）首次能够跨越不同模型架构进行迁移与应用。

    

    针对同一任务进行专门化训练的模型会收敛到相似的行为，然而产生这种行为的参数更新却缺乏共同的坐标系，因此权重空间中的任务算术仍然局限于单一模型，在没有结构对应关系的情况下无法跨越不同架构。借鉴柏拉图的洞穴寓言，我们假设这些特定于模型的更新是某个共享的、与模型无关的对象的投影，我们将其称为“柏拉图任务向量”。为了使这一概念对将图像或音频编码器与文本编码器配对的模型具有可操作性，我们引入了通用任务描述符：一种形状独立于架构和嵌入维度的矩阵，它记录任务的功能效果，并支持将加法和取反作为矩阵运算。将描述符迁移到目标模型中意味着对目标模型进行编辑，直到它能在任务的无标签探测图像和类别名称提示上重现该描述符，无需逐图像标注。

    arXiv:2610.00929v1 Announce Type: cross  Abstract: Models specialized for the same task converge to similar behavior, yet the parameter updates that produce it share no common coordinate system, so weight-space task arithmetic stays confined to a single model and cannot cross architectures without a structural correspondence. Drawing on Plato's allegory of the cave, we hypothesize that these model-specific updates are shadows of one shared, model-agnostic object, which we call the platonic task vector. To make it operational for models that pair an image or audio encoder with a text encoder, we introduce Universal Task Descriptors: matrices whose shape is independent of architecture and embedding dimension, which record a task's functional effect and support addition and negation as matrix operations. Transferring a descriptor into a target means editing the target until it reproduces the descriptor on the task's unlabeled probe images and class-name prompts, requiring no per-image lab
    
[^305]: TrueMuse：文生音乐模型数据归因的基准测试

    TrueMuse: A Benchmark for Data Attribution in Text-to-Music Models

    [https://arxiv.org/abs/2610.00835](https://arxiv.org/abs/2610.00835)

    该论文提出了 TrueMuse，首个通过在已知纳入的归因样本上微调扩散式文生音乐模型来构建可控真值的数据归因基准，覆盖旋律结构、音色、艺术家风格和流派模式四种归因设置，解决了文生音乐数据归因方法缺乏可靠评估标准的问题。

    

    文本生成音乐模型是在海量音乐数据集上训练的，因此越来越需要能够量化单个训练样本贡献的数据归因方法。然而，由于缺乏可靠的真值标准，现有的归因方法难以得到严格评估，使其真实有效性难以被可靠地衡量。为了填补这一空白，我们提出了 TrueMuse，一个用于文生音乐数据归因的可控数据集与基准。TrueMuse 的构建方式是在精心筛选的归因样本上微调三个基于扩散模型的文生音乐模型，由于这些样本是否被纳入微调是已知的，因此为评估提供了可控的归因目标。该基准涵盖四种归因设置，包括旋律结构、音色特征、艺术家级风格特征以及流派级共享模式，共包含 133 个属性、648 个微调模型和 95,456 个生成样本。

    arXiv:2610.00835v1 Announce Type: new  Abstract: Text-to-music generation models are trained on massive music collections, creating a growing need for data attribution methods that can quantify the contribution of individual training samples. However, existing attribution methods are difficult to rigorously evaluate due to the lack of reliable ground truth, making it challenging to reliably assess their actual effectiveness. To address this gap, we introduce TrueMuse, a controlled dataset and benchmark for text-to-music data attribution. TrueMuse is constructed by fine-tuning three diffusion-based text-to-music models on carefully curated attribution samples, whose known inclusion in fine-tuning provides controlled attribution targets for evaluation. The benchmark covers four attribution settings, spanning melodic structure, timbral characteristics, artist-level stylistic signatures, and genre-level shared patterns, and includes 133 attributes, 648 fine-tuned models, and 95,456 generat
    
[^306]: 学习电价定价以实现最优需求响应

    Learning to Price Electricity for Optimal Demand Response

    [https://arxiv.org/abs/2610.00755](https://arxiv.org/abs/2610.00755)

    本文提出一种基于神经网络的上下文电价定价算法，将定价建模为Stackelberg博弈并学习从上下文特征到可行电价的受限映射，通过模拟美国多个城市电网验证了该方法能显著提升需求响应计划的价值。

    

    利用随时间变化的电价来引导消费者需求响应，并更好地使能源需求与可再生能源生产相匹配，这一点引起了广泛关注。然而，最优电价通常会随时间变化，以响应诸如天气预报、日出/日落时间和星期规律等复杂信号；而现有方法无法有效利用如此丰富的上下文信息。在此，我们提出了一种基于神经网络的上下文能量定价算法，将定价问题建模为Stackelberg博弈，并利用了Mehrabi等人（2024）提出的均场解表示方法。该方法学习从上下文特征到可行价格信号的受限映射。我们通过模拟美国多个城市的电网验证了我们的方法，结果表明，融入上下文信息可以显著提升需求响应计划的价值。

    arXiv:2610.00755v1 Announce Type: new  Abstract: There is considerable interest in using time-varying electricity prices to shape consumer demand response, and better align energy demand with renewable production. However, optimal prices generally vary over time in response to complex signals such as weather forecasts, sunrise/sunset times, and day-of-week patterns; and existing methods are not able to make efficient use of such rich contextual information. Here, we propose a neural-network-based algorithm for contextual energy pricing, modeling pricing as a Stackelberg game and leveraging a mean-field solution representation from Mehrabi et al.~(2024). The approach learns constrained mappings from contextual features to feasible price signals. We validate our approach by simulating the energy grid in several US cities, and show that incorporating contextual information can considerably increase the value of the demand response programs.
    
[^307]: 基于多任务QLoRA的社交媒体可解释自杀风险评估

    Explainable Suicide Risk Assessment on Social Media with Multi-Task QLoRA

    [https://arxiv.org/abs/2610.00610](https://arxiv.org/abs/2610.00610)

    该论文提出一种基于QLoRA微调Qwen2.5-Instruct模型的多任务系统，同时完成社交媒体自杀风险评估中的风险等级分类、证据短语提取和多标签风险与保护因素识别三项任务，并通过多模型概率平均、交叉折共识等定制化聚合策略实现可解释的风险评估。

    

    可解释的自杀风险评估要求模型不仅能够估计风险严重程度，还需要识别支持性语言以及帖子中表达的风险因素和保护因素。我们提出了面向IEEE BigData 2026杯“社交媒体可解释自杀风险评估”竞赛的系统，该系统解决三个任务：风险等级分类、证据短语提取和多标签因素识别。我们的方法采用量化低秩适应（QLoRA）和答案掩码的因果语言模型目标对Qwen2.5-Instruct模型进行微调。在风险分类任务上，我们对所有三个任务进行联合训练；在证据提取上，对任务1a和1b进行联合训练；对于因素识别，则单独对任务2进行适配。我们还针对每种输出定制了聚合策略：对32B和72B模型的风险等级概率取平均值，通过交叉折共识合并证据短语，并通过（校准）特定因素的决策。

    arXiv:2610.00610v1 Announce Type: cross  Abstract: Explainable suicide-risk assessment requires models not only to estimate risk severity, but also to identify supporting language and the risk and protective factors expressed in a post. We present our system for the IEEE BigData 2026 Cup on Explainable Suicide Risk Assessment on Social Media, which addresses three tasks: risk-level classification, evidence phrase extraction, and multi-label factor identification. Our approach adapts Qwen2.5-Instruct models using quantized low-rank adaptation (QLoRA) and an answer-masked causal language-model objective. We jointly train across all three tasks for risk classification, jointly train on Tasks~1a and 1b for evidence extraction, and adapt Task~2 separately for factor identification. We also tailor aggregation to each output: we average risk-level probabilities from the 32B and 72B models, combine evidence phrases through cross-fold consensus, and calibrate factor-specific decisions through r
    
[^308]: 逻辑与记忆的冲突：浅层多层感知机中高阶交互的学习

    The Conflict Between Logic and Memory: Learning Higher-Order Interactions in Shallow MLPs

    [https://arxiv.org/abs/2610.00403](https://arxiv.org/abs/2610.00403)

    该研究在浅层MLP中揭示了“能拟合训练数据却无法恢复生成规则”的逻辑—记忆冲突，并发现优化器选择显著影响高阶交互学习能力——Muon在三、四阶奇偶任务上大幅超越SGD和Adam，而仅冻结与干扰输入相连的第一层权重即可将AdamW准确率从44.73%提升至95.07%。

    

    网络可以在拟合训练样本的同时，却无法恢复生成其标签的规则。我们在单隐层多层感知机（MLP）中研究这种分离现象，使用了能够控制交互阶数以及干扰输入是否存在的合成任务。我们确立了一些基本的基准性质：纯奇偶校验（parity）任务不包含任何具有预测能力的低阶边缘统计量， admits 精确的贝叶斯后验，并且可以在干净的潜在输入上由宽度为 $k$ 的 ReLU 网络表示。随后的实验识别出了截然不同的优化结果。在匹配的 2–4 阶扫描实验中，SGD、Adam 和 Muon 在二阶任务上均达到 100% 的峰值测试准确率；在三阶任务上它们分别达到 96.25%、50.87% 和 76.82%，而 Muon 在四阶任务上达到 99.21%。在一个独立的混合阶数任务中，仅冻结与独立干扰输入相连的第一层权重，就能将 AdamW 在第 10 个训练轮次（epoch）的准确率从 44.73% 提升至 95.07%。移除相同的输入……（原摘要到此截断）

    arXiv:2610.00403v1 Announce Type: new  Abstract: A network can fit its training examples while failing to recover the rule that generated their labels. We examine this separation in single-hidden-layer multilayer perceptrons (MLPs), using synthetic tasks that control interaction order and the presence of nuisance inputs. We establish elementary benchmark properties: pure parity contains no predictive lower-order marginals, admits an exact Bayes posterior, and can be represented on clean latent inputs by a width-$k$ ReLU network. Experiments then identify distinct optimization outcomes. In a matched order-2--4 sweep, SGD, Adam, and Muon all reach 100\% peak test accuracy at order two; at order three they reach 96.25\%, 50.87\%, and 76.82\%, respectively, while Muon reaches 99.21\% at order four. In a separate mixed-order task, freezing only the first-layer weights connected to independent nuisance inputs raises AdamW's epoch-10 accuracy from 44.73\% to 95.07\%. Removing the same inputs 
    
[^309]: PTNO：利用含噪蒙特卡洛估计训练神经算子以解决粒子输运问题

    PTNO: Training Neural Operators with Noisy Monte Carlo Estimates for Particle Transport Problems

    [https://arxiv.org/abs/2609.40090](https://arxiv.org/abs/2609.40090)

    该论文提出粒子输运神经算子PTNO，可直接从含噪、低成本的蒙特卡洛标签中学习粒子输运代理模型，并证明了无偏噪声标签的平方损失与收敛解损失共享同一极小值点，从而解决了高方差与高动态范围两大挑战，大幅降低了训练成本。

    

    在多次散射下的粒子输运是辐射转移和等离子体物理的核心问题，然而高保真的蒙特卡洛（MC）模拟必须追踪数量极其庞大的粒子。基于学习的代理模型可以摊销这一成本，但通常需要在昂贵且充分收敛的MC解上进行训练。我们提出了粒子输运神经算子（PTNO），这是一种能够直接从含噪、低成本的MC标签中学习粒子输运代理模型的神经算子。这类标签带来了两大挑战：（1）高方差会使标准监督学习不稳定；（2）跨越多个数量级的高动态范围（HDR）。针对第一个挑战，我们从大量构型的含噪标签中学习解算子，从而摊销MC成本并泛化到未见过的构型。由于MC标签是无偏的，我们证明了基于这些标签的平方损失与基于收敛解的损失具有相同的极小值点，并且我们对训练的预算分配研究表明……

    arXiv:2609.40090v1 Announce Type: new  Abstract: Particle transport under multiple scattering is central to radiative transfer and plasma physics, yet high-fidelity Monte Carlo (MC) simulations must trace prohibitively many particles. Learning-based surrogates can amortize this cost, but typically train on expensive, well-converged MC solutions. We propose the Particle Transport Neural Operator (PTNO), a neural operator that learns particle transport surrogates directly from noisy, low-cost MC labels. Such labels pose two challenges: (1) high variance, which destabilizes standard supervised learning, and (2) a high dynamic range (HDR) spanning many orders of magnitude. For the first, we learn the solution operator from noisy labels of many configurations, amortizing MC cost and generalizing to unseen configurations. Because MC labels are unbiased, we show that the squared loss on them shares its minimizer with the loss on converged solutions, and our budget-allocation study over traini
    
[^310]: 稀疏正则化部分最优传输的加速算法

    Accelerated Algorithm for Sparse Regularized Partial Optimal Transport

    [https://arxiv.org/abs/2609.40075](https://arxiv.org/abs/2609.40075)

    本文提出一种基于惩罚项重构的加速优化框架，利用平滑强凸正则化器（如二次正则、弹性网络）实现稀疏部分最优传输的高效梯度更新。

    

    arXiv:2609.40075v1 公告类型：新论文 摘要：部分最优传输（POT）通过放宽严格的质量守恒约束，扩展了经典的最优传输问题，使其能够广泛应用于众多现实场景。在许多此类场景中，稀疏传输方案因其可解释性和计算上的优势而备受青睐。尽管平滑且强凸的正则化器——如二次正则或弹性网络——已在各类机器学习应用中被广泛用于诱导稀疏性并加速计算，但与熵正则方法在计算部分最优传输上的算法研究相比，它们受到的关注相对较少。本文提出了一种新的优化框架，通过基于惩罚项的重构方式利用这些正则化器，在保留原始问题结构的同时，实现高效的基于梯度的更新。我们的方法兼容一大类能够促进结构化与稀疏传输方案的正则化器。（摘要原文在此处截断）

    arXiv:2609.40075v1 Announce Type: new  Abstract: Partial Optimal Transport (POT) extends the classical optimal transport problem by relaxing the strict mass conservation constraint, enabling its use in a wide range of real-world applications. In many of these settings, sparse transport plans are preferred for their interpretability and computational benefits. While smooth and strongly convex regularizers - such as quadratic or elastic net - have been vastly used in various machine learning applications to induce sparsity and accelerate computation, they have received less algorithmic attention compared to entropic approaches for computational POT. In this paper, we propose a new optimization framework that leverages these regularizers through a penalty-based reformulation, enabling efficient gradient-based updates while preserving the structure of the original problem. Our method accommodates a broad class of regularizers that promote structured and sparse transport plans. Building on 
    
[^311]: CATCH：一个用于编程强化学习中奖励破解的可控分析测试平台

    CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL

    [https://arxiv.org/abs/2609.39533](https://arxiv.org/abs/2609.39533)

    提出 CATCH 测试平台，通过刻意暴露环境漏洞并以独立审计生成黄金标签，实现对编程强化学习中奖励破解行为的可控复现、可靠识别与系统干预研究。

    

    在带有可验证奖励的强化学习（RLVR）过程中，大语言模型（LLM）可能会利用环境中的漏洞来获取高奖励，而并未真正提升预期能力，这种现象即“奖励破解”。尽管奖励破解对训练效率和安全性构成风险，但在训练过程中监测和缓解此类行为仍然具有挑战性，其瓶颈在于缺乏能够重现破解行为并可靠识别该行为的测试平台。我们提出了 CATCH，一个用于研究编程强化学习中奖励破解现象的可控测试平台。CATCH 刻意暴露环境漏洞，并通过对比易受攻击评估器下的“成功”与独立审计下的真实任务正确性，提供基于执行的黄金标签。此外，它还可以通过监督微调数据配比来控制模型的初始破解倾向，并通过奖励设计来调节获取奖励的难度，从而支持对破解动态与干预措施进行系统性比较研究。

    arXiv:2609.39533v1 Announce Type: new  Abstract: During reinforcement learning with verifiable rewards (RLVR), large language models (LLMs) can exploit loopholes in their environments to obtain high rewards without improving the intended capabilities, i.e., reward hacking. Despite its risks to training efficiency and safety, monitoring and mitigating reward hacking during training remain challenging, which is limited by a lack of testbeds that reproduce hacking and reliably identify it. We introduce CATCH, a controllable testbed for studying reward hacking in coding RL. CATCH deliberately exposes environmental loopholes and provides execution-based gold labels by comparing success under a vulnerable evaluator with task correctness under an independent audit. It also can control the model's initial hacking tendency through supervised fine-tuning data mixtures and the difficulty of earning rewards through reward designing, enabling systematic comparisons of hacking dynamics and intervent
    
[^312]: 面向指纹识别的量子-经典混合自监督学习的宽度匹配比较

    A Width-Matched Comparison of Hybrid Quantum-Classical Self-Supervised Learning for Fingerprint Recognition

    [https://arxiv.org/abs/2609.39172](https://arxiv.org/abs/2609.39172)

    该论文通过在匹配表示宽度下将QuFeX量子特征提取模块嵌入SimCLR、MoCo v2和BYOL三种自监督框架进行指纹识别实验，系统比较量子-经典混合模型与纯经典模型，以厘清量子线路本身对学习表示的真实贡献。

    

    指纹识别是一种被广泛部署的生物识别技术，但监督式训练需要大量带标签的注册样本。自监督学习（SSL）消除了这一要求，而量子-经典混合模型已被提出用于丰富所学习的表示。先前的量子SSL研究仅考虑了单一的对比学习目标，因此尚不清楚所报告的收益究竟是依赖于该目标本身，还是可以归因于量子线路。我们将QuFeX量子特征提取模块嵌入三种SSL框架——对比式的SimCLR和MoCo v2，以及非对比式的BYOL——并在匹配的表示宽度（8个特征，相当于8个量子比特）下，在SOCOFing指纹数据集上（以CIFAR-10作为对照实验）将每种混合模型与其经典对应模型进行比较，采用基于编码器特征的k近邻识别方法。在单次运行实验中，两种对比学习目标下的混合模型得分明显更高，而对于BYOL（摘要在此处被截断）

    arXiv:2609.39172v1 Announce Type: cross  Abstract: Fingerprint recognition is a widely deployed biometric, but supervised training requires large labeled enrollment sets. Self-supervised learning (SSL) removes this requirement, and hybrid quantum-classical models have been proposed to enrich the learned representations. Prior quantum SSL studies consider a single contrastive objective, so it is unclear whether reported benefits depend on the objective or can be attributed to the quantum circuit. We insert the QuFeX quantum feature-extraction module into three SSL frameworks, the contrastive SimCLR and MoCo v2 and the non-contrastive BYOL, and compare each hybrid with its classical counterpart at matched representation width (8 features, equal to 8 qubits) on the SOCOFing fingerprint dataset, with a CIFAR-10 control, using k-nearest-neighbor identification on encoder features. In single-run experiments the hybrid scores clearly higher for both contrastive objectives, whereas for BYOL a 
    
[^313]: JARQ：面向量化的联合交替精炼方法

    JARQ: Joint Alternating Refinement for Quantization

    [https://arxiv.org/abs/2609.38599](https://arxiv.org/abs/2609.38599)

    JARQ 是一种即插即用的后训练量化精炼方法，通过交替执行所有组尺度的联合最小二乘拟合与有界 Babai 提议（同时移动组内多个编码），在保持位宽、零点和推理成本不变的前提下，显著降低大语言模型量化后的困惑度。

    

    面向大语言模型的分组后训练量化器将权重四舍五入到一个网格上，而该网格并未根据所得的整数编码进行重新拟合。我们证明这会损失精度：最优网格取决于编码本身，输入相关性会使不同组之间的误差相互耦合，而有益的编码变更往往需要同时改动多个编码。我们提出 JARQ，这是一种即插即用的精炼方法，它从任意分组量化器出发，交替进行所有组尺度的联合最小二乘拟合，以及有界的 Babai 提议（后者在当前网格上同时移动一个组内的多个编码）。该问题是一个双线性、盒约束的混合整数最小二乘问题；该求解器无需反向传播，在精确尺度求解下不会增加逐层目标函数值，并保持宿主量化器的位宽、分组方式、零点和推理成本不变。在 Llama-2、Llama-3 和 Qwen 模型上，结合 RTN、GPTQ、OmniQuant 和 AWQ 等宿主方法，JARQ 在 96 组对比中的 90 组中降低了困惑度……（摘要原文在此处截断）

    arXiv:2609.38599v1 Announce Type: new  Abstract: Group-wise post-training quantizers for large language models round weights onto a grid that is not refit to the resulting integer codes. We show that this leaves accuracy on the table: the best grid depends on the codes, input correlations couple the errors of different groups, and useful code changes often involve many codes at once. We propose JARQ , a plug-in refinement that starts from any group-wise quantizer and alternates a joint least-squares fit of all group scales with bounded Babai proposals that move many codes of a group together on the current grid. The problem is a bilinear box-constrained mixed-integer least-squares problem; the solver is backpropagation-free, does not increase the layer-wise objective under exact scale solves, and keeps the host's bit width, groups, zero points, and inference cost. Across Llama-2, Llama-3, and Qwen models with RTN, GPTQ, OmniQuant, and AWQ hosts, JARQ lowers perplexity in 90 of 96 compa
    
[^314]: 从个体学习到社会学习：刻画大语言模型中的递归社会改进

    From Solo to Social Learning: Characterizing Recursive Social Improvement in LLMs

    [https://arxiv.org/abs/2609.38516](https://arxiv.org/abs/2609.38516)

    该论文提出“递归社会改进”这一新概念，并发现尽管经典社会学习算法能从同伴中受益，但当每个LLM智能体各自追求自身奖励时，当前的LLM无法通过相互学习改进整个群体，其每token收益反而低于独立学习。

    

    大型语言模型（LLM）如今可以通过修改自身遵循的指令来实现自我改进，同时LLM智能体也越来越多地被组织起来协同解决复杂问题。然而，自我改进方法通常一次只优化一个系统，而多智能体框架往往让每个模型朝着同一个共同目标努力。我们提出了一个不同的问题：当每个智能体追求自己的奖励时，自我改进的LLM能否从彼此身上充分学习，从而改进整个群体？我们将这种能力称为递归社会改进。我们研究了会修改技能文件、并自主决定是否、何时以及向谁进行模仿学习的智能体群体。其中，独立搜索、向同伴学习和执行动作共享同一个token预算。在受控环境中，现有的社会学习算法能够从同伴中获益，但三种LLM却不能：它们每token获得的奖励低于单独学习的个体，要么探索范围过窄，要么在采取行动之前就耗尽了token。

    arXiv:2609.38516v1 Announce Type: cross  Abstract: Large language models (LLMs) can now improve themselves by revising the instructions they follow, and LLM agents are increasingly orchestrated to work together on complex problems. However, self-improvement methods typically optimize one system at a time, and multi-agent frameworks often have every model work toward a shared goal. We ask a different question. When each agent pursues its own reward, can self-improving LLMs learn from one another well enough to improve the whole population? We call this capability recursive social improvement. We study populations that revise skill files and choose whether, when, and whom to copy from. Independent search, learning from peers, and acting all share one token budget. In controlled environments, established social-learning algorithms benefit from peers, but three LLMs do not. They earn less reward per token than solo learners, and explore too narrowly or run out of tokens before acting. We t
    
[^315]: 微小的扰动即可改写感知：视觉语言模型在 ε ≤ 4/255 下的目标语义替换攻击

    It Takes Little to Rewrite Perception: Targeted Semantic Substitution in Vision-Language Models at $\epsilon \leq 4/255$

    [https://arxiv.org/abs/2609.38298](https://arxiv.org/abs/2609.38298)

    本文提出一种目标语义替换攻击，通过在VLM的合并后token空间中对齐源图像与目标图像，在 ε ≤ 4/255 的微小扰动下即可完全替换模型对图像的语义感知，推翻了此前认为VLM在该扰动范围内具有鲁棒性的观点。

    

    视觉语言模型（VLM）被广泛部署于安全关键场景中，理解它们在多大程度上可以被对抗性扰动所控制，是评估其可信度的先决条件。现有的表示对齐攻击（即让VLM感知目标图像）在 ε ≤ 4/255 的扰动范围内仅能取得有限的成功，因此VLM似乎在这一范围内具有鲁棒性。我们证明这种鲁棒性并不成立，因为目标语义替换攻击能够在同样的扰动范围内成功实施。具体而言，我们在白盒威胁模型下，于受害VLM的合并后token空间（post-merger token space）中，将源图像的每个模态流与目标图像中的对应部分进行对齐。我们在严格的成功标准下进行评估，要求模型同时说出目标的名称、确认目标的存在，并否认源图像。在图像任务中，目标语义在 ε = 2/255 时便已显现，完全替换的幅度达到……

    arXiv:2609.38298v1 Announce Type: cross  Abstract: Vision Language Models (VLMs) are widely deployed in safety-critical scenarios, and understanding to which extent they can be controlled by adversarial perturbation is a prerequisite for evaluating their trustworthiness. Existing representation-alignment attacks, which make a VLM perceive a target image, achieve limited success at $\varepsilon \leq 4/255$. Therefore, VLMs seems robust to perturbations in this range. We show that this robustness does not hold, as targeted semantic substitution succeeds within the same range. Specifically, we align each stream of the source image with its counterpart in the target image in the victim VLM's post-merger token space, operating under a white-box threat model. We evaluate under a strict success criterion, requiring the model to simultaneously name the target, confirm its presence, and deny the source. In images, target semantics appear at $\varepsilon = 2/255$ and complete replacement reaches
    
[^316]: 权重读取与写入特征：基于激活空间的可扩展参数分解

    Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space

    [https://arxiv.org/abs/2609.37731](https://arxiv.org/abs/2609.37731)

    提出激活支持的参数分解（ASPD），通过联合分解激活与参数空间并将每个权重组件锚定于其读写激活特征，实现了预训练大语言模型中可扩展、可解释且可因果编辑的参数分解。

    

    激活空间和参数空间为模型计算提供了互补的视角。激活表示信息，而权重则读取、转换并写入这些信息。然而，现有的可解释性方法大多将这两个空间分开研究，使得所表示的信息与参数级计算之间的联系尚未得到充分探索。我们提出了激活支持的参数分解（ASPD），该方法联合分解激活空间和参数空间，并将每个学习到的权重组件锚定在其读取或写入的激活特征上。这种锚定方式利用模型的内部激活来约束原本非唯一的参数分解，同时内部重建目标在所分析的权重矩阵处提供局部学习信号。这些特性共同实现了在预训练大语言模型中可扩展、可解释且可因果编辑的参数分解，并展示……

    arXiv:2609.37731v1 Announce Type: new  Abstract: Activation space and parameter space provide complementary views of model computation. Activations represent information, while weights read, transform, and write that information. Yet existing interpretability methods largely study the two spaces separately, leaving the connection between represented information and parameter-level computation underexplored. We introduce Activation-Supported Parameter Decomposition (ASPD), which jointly decomposes activation and parameter spaces and grounds each learned weight component in the activation features it reads or writes. This grounding constrains otherwise non-unique parameter decompositions using the model's internal activations, while an internal reconstruction objective provides a local learning signal at the weight matrix being analyzed. Together, these properties enable scalable, interpretable, and causally editable parameter decomposition in pretrained large language models, demonstrat
    
[^317]: TomoTransformer：迈向CT重建的基础模型

    TomoTransformer: Towards a Foundation Model for CT Reconstruction

    [https://arxiv.org/abs/2609.37605](https://arxiv.org/abs/2609.37605)

    提出TomoTransformer，一种基于Transformer的CT重建基础模型，将局部滤波投影作为token，在反投影空间中通过自注意力预测缺失视角，可处理任意数量和角度的投影且对探测器尺寸不变。

    

    监督深度学习已经推动了稀疏视角断层重建的发展。然而，传统模型通常将滤波反投影（FBP）图像或正弦图映射到干净的重建结果，在分布偏移下表现脆弱。由于每当投影数量和角度、探测器分辨率或数据分布发生变化时都需要重新训练，这些模型在现实应用中的部署仍然受限。为解决这一问题，我们提出了TomoTransformer，一种基于Transformer的架构，它将每个局部滤波投影视为独立的token，并通过自注意力机制预测缺失的视角。关键在于，TomoTransformer在反投影空间中运行，该空间将投影按空间位置分离，使视角插值在几何上是良构的，且对探测器尺寸具有不变性。这一设计产生了一个单一的基础模型，能够处理任意数量的输入投影和任意角度……

    arXiv:2609.37605v1 Announce Type: cross  Abstract: Supervised deep learning has advanced sparse-view tomographic reconstruction. However, conventional models, which typically map filtered back-projection (FBP) images or sinograms to clean reconstructions, are brittle under distribution shifts. Because they require retraining whenever projection counts and angles, detector resolutions, or data distributions change, their deployment in real-world applications remains limited. To address this, we introduce TomoTransformer, a transformer-based architecture that treats each \textit{local} filtered projection as an individual token and predicts missing views via self-attention. Crucially, TomoTransformer operates in a \emph{back-projection space} that separates projections across spatial locations, making view interpolation geometrically well-posed and invariant to detector size. This design yields a single foundation model that can process any number of input projections, at arbitrary angul
    
[^318]: 基于现成教师的图条件化在线策略智能体蒸馏

    Graph-Conditioned On-Policy Agent Distillation from Off-the-Shelf Teachers

    [https://arxiv.org/abs/2609.37522](https://arxiv.org/abs/2609.37522)

    GC-OPD 通过图结构索引教师的成功与失败执行历史，为学生轨迹提供执行证据丰富的评分上下文，从而显著提升现成教师对多轮任务中语言智能体的在线策略蒸馏效果。

    

    在线策略蒸馏（OPD）通过教师对学生生成轨迹的反馈来训练紧凑的语言智能体。在多轮任务中，误差的累积可能使学生超出教师有效监督所能覆盖的范围。我们提出了图条件化在线策略智能体蒸馏（GC-OPD），它利用执行证据来丰富现成教师的评分上下文。该方法通过共享状态构建图来索引教师重复执行的记录，同时完整保留成功与失败的历史。在每次学生回合结束后，GC-OPD 检索当前状态的参考记录或历史替代方案，并将其与学生的事后反思相结合，对原始的思维-动作词元进行评分。使用相同的原始教师，GC-OPD 相比普通 OPD 将平均成功率在 ScienceWorld（4B 学生）上从 24.70% 提升至 48.78%，在 ALFWorld Unseen 上从 53.36% 提升至 85.26%，在 WebShop 上从 29.10% 提升至 37.65%。在学生模型规模相同的情况下，它还取得了更高的平均成功率……

    arXiv:2609.37522v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains compact language agents with teacher feedback on student-generated trajectories. In multi-turn tasks, compounding errors can move students beyond the teacher's effective supervision. We introduce Graph-Conditioned On-Policy Agent Distillation (GC-OPD), which enriches an off-the-shelf teacher's scoring context with execution evidence. A graph indexes repeated teacher executions by shared states while preserving complete successful and failed histories. After each student episode, GC-OPD retrieves current-state references or historical alternatives and combines them with student hindsight to score the original thought-action tokens. Using the same original teachers, GC-OPD improves mean success over vanilla OPD from 24.70% to 48.78% on ScienceWorld (4B student), from 53.36% to 85.26% on ALFWorld Unseen, and from 29.10% to 37.65% on WebShop. At matched student sizes, it also achieves higher mean success t
    
[^319]: SCOPE：用于稀疏PDE推断的观测条件化全目标预测

    SCOPE: Observation-Conditioned Full-Target Prediction for Sparse PDE Inference

    [https://arxiv.org/abs/2609.36527](https://arxiv.org/abs/2609.36527)

    SCOPE通过共享解码器将全场潜在预测与物理重建相耦合，实现了从稀疏观测中确定性恢复完整PDE物理场，并从理论上证明了最优潜在预测未必带来最优场重建以及解码器改进向部分观测恢复迁移的条件。

    

    从稀疏观测中恢复完整的物理场具有挑战性，因为测量数据可能无法唯一确定底层状态。基于扩散模型的PDE求解器通过迭代采样来解决这一问题，而神经算子则提供确定性的单次预测。我们提出了SCOPE（Sparse-Context Observability-aware Predictive Embeddings，稀疏上下文可观测性感知预测嵌入），通过将全场潜在预测与物理重建相耦合，从稀疏观测中恢复完整的PDE场。一个共享解码器同时从预测表示和完整视图表示中重建物理场，从而使表示学习同时受到物理恢复和潜在匹配的引导。我们在固定的教师-解码器对下推导了二次风险分解，说明了为什么最优的潜在预测不一定能产生最优的场重建。我们还建立了在完整输入上解码器的改进能够迁移到部分观测恢复的充分条件。

    arXiv:2609.36527v1 Announce Type: new  Abstract: Recovering complete physical fields from sparse observations is challenging because the measurements may not uniquely determine the underlying state. Diffusion-based PDE solvers address this problem through iterative sampling whereas neural operators provide deterministic one-pass predictions. We propose SCOPE (Sparse-Context Observability-aware Predictive Embeddings) to recover complete PDE fields from sparse observations by coupling full-field latent prediction with physical reconstruction. A shared decoder reconstructs fields from both predicted and complete-view representations so that representation learning is guided by both physical recovery and latent matching. We derive a quadratic risk decomposition at fixed teacher-decoder pairs showing why optimal latent prediction need not yield optimal field reconstruction. We also establish sufficient conditions for decoder improvements on complete inputs to transfer to recovery from parti
    
[^320]: 为什么对神经网络植入后门如此容易？

    Why Backdooring Neural Networks is so Easy?

    [https://arxiv.org/abs/2609.36117](https://arxiv.org/abs/2609.36117)

    该论文通过对投毒高斯混合数据上训练的二次神经元进行精确的闭式理论分析，揭示了一个反直觉的结论：正是使神经网络强大的特征学习动态使其更容易遭受后门攻击——在懒惰学习机制下，成功攻击需要触发强度与投毒比例的平方根成反比（α ∝ π^(-1/2)），而特征学习则使攻击变得更为容易。

    

    arXiv:2609.36117v1 公告类型：新论文 摘要：保护现代人工智能系统免受后门攻击仍然是一个悬而未决的挑战，这需要对攻击者的预算进行根本性的、有原理依据的估计——即构建成功且隐蔽攻击所需的投毒比例 $\pi$ 和触发器强度 $\alpha$。受近期经验证据的启发——即使干净数据集不断增长，对大语言模型进行投毒可能只需要几乎恒定数量的恶意样本——我们对在受投毒高斯混合数据上训练的二次神经元进行了精确的闭式分析。我们表明，或许有些反直觉的是，正是使神经网络强大的特征学习动态，同样也可能使其更容易受到后门攻击。具体而言，在干净精度保持到一阶 $O(\pi)$ 的情况下，我们证明懒惰学习对成功攻击施加了逆平方根缩放规律 $\alpha \propto \pi^{-1/2}$，而特征学习则诱导出一个二次检测器，其损失……（原文摘要在此处截断）

    arXiv:2609.36117v1 Announce Type: new  Abstract: Securing modern AI systems against backdoor attacks remains an open challenge and requires fundamentally principled estimates of the adversary's budget -- the poison fraction $\pi$ and trigger strength $\alpha$ needed to construct successful yet stealthy attacks. Motivated by recent empirical evidence that poisoning large language models can require a nearly constant number of malicious samples even as clean datasets grow, we derive an exact closed-form analysis of a quadratic neuron trained on a poisoned Gaussian mixture. We show, perhaps counterintuitively, that the same feature-learning dynamics that make neural networks powerful can also make them more vulnerable to backdoors. Specifically, with clean accuracy preserved to first order, $O(\pi)$, we demonstrate that lazy learning imposes the inverse-square-root scaling $\alpha \propto \pi^{-1/2}$ for a successful attack, while feature learning induces a quadratic detector whose loss m
    
[^321]: CipherGenome：基因组混合专家模型的同态推理

    CipherGenome: Homomorphic Inference for Genomic Mixture-of-Experts

    [https://arxiv.org/abs/2609.35883](https://arxiv.org/abs/2609.35883)

    CipherGenome提出了一种同态加密推理协议，将151亿参数基因组MoE模型的专家投影（95.8%的参数）在模块-LWE加密下安全外包给不可信GPU服务器，同时把嵌入、注意力和路由器保留在可信客户端上，防止私有基因组在租用算力时被服务器以99.8%的准确率恢复。

    

    基因组基础模型正逐渐发展为稀疏混合专家网络，其专家权重已无法容纳在保存序列的机器上，然而将私有基因组发送到租用的加速器上会使其暴露：我们证明，托管单个专家的单一服务器就能以99.8%的top-1准确率恢复输入的核苷酸。我们提出CipherGenome，一种协议，它将151亿参数的MoE基因组模型的嵌入、注意力和路由器保留在可信的瘦客户端上，并在模块-LWE加密下将每个专家投影（占全部参数的95.8%）外包给不可信且可能相互串通的GPU服务器。该设计利用了三个结构性事实：专家层在两个SwiGLU门控之间是线性的，专家权重是公开的，以及GPU整数张量核心可以在单次GEMM中精确计算模 $2^{48}$ 下的密文-权重乘积。客户端精确地评估非线性运算并使用新鲜密钥重新加密，因此没有多项式

    arXiv:2609.35883v1 Announce Type: cross  Abstract: Genome foundation models are growing into sparse mixture-of-experts (MoE) networks whose expert weights no longer fit on the machines that hold the sequences, yet sending a private genome to rented accelerators exposes it: we show that a single server hosting one expert recovers the input nucleotides with 99.8% top-1 accuracy. We present CipherGenome, a protocol that keeps the embedding, attention and router of a 15.1B-parameter MoE genome model on a trusted thin client and outsources every expert projection, 95.8% of the parameters, to untrusted and possibly colluding GPU servers under module-LWE encryption. The design exploits three structural facts: expert layers are linear between two SwiGLU gates, expert weights are public, and GPU integer tensor cores can evaluate a ciphertext-weight product exactly modulo $2^{48}$ in a single GEMM. The client evaluates the nonlinearity exactly and re-encrypts with fresh secrets, so no polynomial
    
[^322]: GenomeOcean Anywhere：面向基因组混合专家模型的私密WebGPU推理

    GenomeOcean Anywhere: Private WebGPU Inference for Genome MoEs

    [https://arxiv.org/abs/2609.35882](https://arxiv.org/abs/2609.35882)

    该研究构建了一个让150亿参数基因组混合专家模型在志愿者浏览器上私密运行的系统，通过手写WebGPU内核与实值拉格朗日编码计算，在专家分布于多个不受信任设备的情况下实现与原生推理一致的预测结果，且不向任何单一设备泄露序列信息。

    

    基因组基础模型在序列产生之处最能发挥作用，然而最大的模型需要数据中心的加速器，并且需要将私密的DNA数据发送到远端。我们探讨这样一个问题：一个150亿参数的基因组混合专家模型能否改为在志愿者的网页浏览器上运行，将其专家分布在众多不受信任的设备上，既不改变其预测结果，也不向任何单一设备泄露序列信息。我们构建了一个系统：由可信的协调器运行注意力和路由机制，而浏览器工作节点通过手写的WebGPU内核运行每个专家的前馈网络；我们采用实值拉格朗日编码计算来保护专家输入：每个工作节点仅接收到一个经高斯填充的份额，计算专家的线性映射，协调器从三个工作节点中的任意两个即可完成解码。在GenomeOcean-MoE（8个专家、top-2路由、24层）上，浏览器推理路径在每种量化级别下均与原生llama.cpp的表现相匹配。

    arXiv:2609.35882v1 Announce Type: cross  Abstract: Genome foundation models are most useful where sequences are generated, yet the largest models need datacenter accelerators and a place to send private DNA. We ask whether a 15-billion-parameter genome mixture-of-experts (MoE) model can instead run on volunteers' web browsers, with the experts spread across many untrusted devices, without changing its predictions and without revealing the sequence to any single device. We build a system in which a trusted coordinator runs attention and routing while browser workers run every expert feed-forward network through hand-written WebGPU kernels, and we protect the expert inputs with real-valued Lagrange coded computing: each worker receives only a Gaussian-padded share, computes the expert's linear maps, and the coordinator decodes from any two of three workers. On GenomeOcean-MoE (8 experts, top-2 routing, 24 layers), the browser path matches native llama.cpp at every quantization level, the
    
[^323]: GenoTrace：基因组基础模型蒸馏中的可继承水印

    GenoTrace: Inheritable Watermarks for Genome Foundation Model Distillation

    [https://arxiv.org/abs/2609.35881](https://arxiv.org/abs/2609.35881)

    提出GenoTrace，一种密码子感知的绿名单水印扩展方法，使基因组基础模型蒸馏出的学生模型能够继承可检测的水印信号，并在token替换和核苷酸编辑等攻击下保持显著的鲁棒性。

    

    基因组模型能否保留用于训练它的合成序列的可检测记录？我们通过蒸馏研究水印继承问题，提出了GenoTrace，这是绿名单水印的一种密码子感知扩展。两个token级别的因子利用密码子位置和物种特异性的密码子使用偏好来调节教师模型的生成偏置。所生成的序列用于训练一个更小的学生模型，其输出可以在没有主动水印处理器的情况下进行审计。在一项采用三个随机种子的GenomeOcean-500M到100M实验中，联合配置实现了平均审计得分17.88以及在固定阈值下94.5%的检测率。在经过密钥感知的token替换攻击后，该方法保留了49.0%的检测率，而现有的单种子普通水印对照方法检测率为0%；在经过组合的针对机制的核苷酸编辑攻击后，仍保留了47.0%的检测率。额外的实验证实了在五个物种条件数据集以及高达40倍的师生模型规模比下，继承的水印信号依然有效。

    arXiv:2609.35881v1 Announce Type: cross  Abstract: Can a genome model retain a detectable record of the synthetic sequences used to train it? We study watermark inheritance through distillation with GenoTrace, a codon-aware extension of green-list watermarking. Two token-level factors modulate the teacher's generation bias using codon position and organism-specific codon usage. The resulting sequences train a smaller student, whose outputs are audited without an active watermark processor. In a three-seed GenomeOcean-500M-to-100M experiment, the joint configuration achieves a mean audit score of 17.88 and 94.5% detection at a fixed threshold. It retains 49.0% detection after key-aware token substitution, compared with 0% for the available single-seed plain-watermark comparator, and 47.0% after combined mechanism-targeted nucleotide edits. Additional experiments establish inherited signal across five organism-conditioned datasets and teacher-student size ratios up to 40. Component ablat
    
[^324]: 从 Pass@K 与 Pass@1 之间的差距中学习

    Learning from the Gap Between Pass@K and Pass@1

    [https://arxiv.org/abs/2609.35793](https://arxiv.org/abs/2609.35793)

    提出 GapFT 方法，通过在 Pass@K 与 Pass@1 的差距（即单样本失败但 K 个样本内可解决的问题）上进行微调，将测试时搜索带来的能力吸收进模型，从而提升单样本解码的性能。

    

    大语言模型越来越多地采用基于可验证奖励的强化学习（RLVR）进行训练。精确的验证器还可以通过从多个样本中挑选出一个通过的响应来支持测试时扩展，而其他部署方式则使用束搜索、自适应采样或工具。我们研究单样本解码——即每个查询只获得一个响应而不进行搜索——以探究搜索中暴露出的行为能否被吸收进模型之中。现有的基于验证响应的后训练方法通常不会区分那些在首次解码时就已经解决的问题与在 K 个样本内才得以恢复的失败问题。在固定预算下，这可能导致训练样例被浪费在重复部署策略已经具备的行为上。我们提出 GapFT，它根据源检查点的单样本结果来选择训练证据，并在 Pass@K 与 Pass@1 之间的差距上进行微调：即策略在单样本上失败但在 K 个样本内能够解决的问题。我们匹配训练样例，

    arXiv:2609.35793v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly trained with reinforcement learning from verifiable rewards (RLVR). An exact verifier can also support test-time scaling by selecting a passing response from multiple samples, while other deployments use beam search, adaptive sampling, or tools. We study single-sample decoding, where each query receives one response without search, to ask whether search-exposed behavior can be absorbed into the model. Existing verified-response post-training recipes do not generally distinguish problems already solved on the first decode from failures recovered within K samples. Under a fixed budget, this can spend examples repeating behavior the deployed policy already has. We introduce GapFT, which selects training evidence by the source checkpoint's single-sample outcome and fine-tunes on the Pass@K-Pass@1 gap: problems the policy fails on one sample but solves within K samples. We match training examples,
    
[^325]: 面向单步视觉生成的统一分布训练框架

    Unifying Distributional Training for One-Step Visual Generation

    [https://arxiv.org/abs/2609.35763](https://arxiv.org/abs/2609.35763)

    本文提出了单步视觉生成中分布训练的统一理论框架，并据此提出MGFlow方法，以可调粒度的高斯混合建模特征分布，同时支持最优传输与分数匹配，有效缓解模式坍塌问题。

    

    分布训练通过在冻结表示空间中匹配真实特征与生成特征，为单步视觉生成提供集体监督。我们引入了一个统一的理论框架，该框架将分布建模与匹配差异分离，并通过Wasserstein梯度流将全局目标与逐点特征更新联系起来。在此框架下，FD-Loss和高斯核漂移分别通过高斯最优传输和基于核密度的KL匹配被推导恢复。该框架进一步启发了MGFlow方法，其以介于全局矩和基于样本的表示之间的可调粒度，用高斯混合模型对特征分布进行建模。MGFlow同时支持最优传输和基于分数的匹配，并通过将质量约束的样本分配与成对分量更新相结合，解决了仅靠混合模型表达力无法克服的模式坍塌问题。在Image（摘要在此处截断）

    arXiv:2609.35763v2 Announce Type: replace  Abstract: \emph{Distributional training} provides collective supervision for one-step visual generation by matching real and generated features in frozen representation spaces. We introduce \emph{a unified theoretical framework} that separates distribution modeling from matching discrepancy and connects global objectives to pointwise feature updates through Wasserstein gradient flow. Under this framework, FD-Loss and Gaussian-kernel Drifting are recovered through Gaussian optimal transport and kernel-density-based KL matching, respectively. The framework motivates \textbf{MGFlow}, which models feature distributions with Gaussian mixtures at an adjustable granularity between global moments and sample-based representations. MGFlow supports both optimal transport and score-based matching, and couples mass-constrained sample assignment with paired component updates to address mode collapse that mixture expressivity alone does not resolve. On Image
    
[^326]: 正则化的可证明优势：对抗性模仿学习的快速收敛速率

    Provable Benefits of Regularization: Fast Rates for Adversarial Imitation Learning

    [https://arxiv.org/abs/2609.35698](https://arxiv.org/abs/2609.35698)

    该论文首次为对抗性模仿学习中奖励正则化与策略正则化的有限样本优势建立了理论保证，提出了一种结合KL策略正则化与专家-学习者占用度加权二次奖励惩罚的无模型算法，并证明了在K个在线回合和N条专家轨迹下正则化模仿差距达到Õ(1/K + 1/N)的快速收敛速率。

    

    我们研究对抗性模仿学习（AIL），其中智能体通过针对一个区分专家与学习者行为的对抗性奖励来优化策略，从而学会模仿专家演示。历史上，奖励正则化和基于熵的策略正则化是GAIL和LS-IQ等经验上成功方法的关键组成部分，但它们的有限样本优势仍未得到充分探索。我们在具有一般函数逼近的有限视界马尔可夫决策过程中，为联合正则化的AIL建立了快速收敛速率。我们的无模型算法——双重正则化AIL——将KL策略正则化与由专家和学习者占用度加权的二次奖励惩罚相结合。在K个在线回合和N条专家轨迹的条件下，我们证明了对于固定正则化参数，正则化模仿差距的$\widetilde{O}\left(\frac{1}{K}+\frac{1}{N}\right)$界。

    arXiv:2609.35698v2 Announce Type: replace  Abstract: We study adversarial imitation learning (AIL), in which an agent learns to imitate expert demonstrations by optimizing a policy against an adversarial reward that distinguishes expert and learner behavior. Historically, reward regularization and entropy-based policy regularization are key components of empirically successful methods such as GAIL and LS-IQ, yet their finite-sample benefits remain underexplored. We establish fast rates for jointly regularized AIL in finite-horizon Markov decision processes with general function approximation. Our model-free algorithm, Dually Regularized AIL, combines KL policy regularization with a quadratic reward penalty weighted by expert and learner occupancies. With K online episodes and N expert trajectories, we prove a $\widetilde{O}\left(\frac{1}{K}+\frac{1}{N}\right)$ bound on the regularized imitation gap for fixed regularization parameters. Our analysis combines an online mirror descent cons
    
[^327]: MASCIT：一种面向自然不规则时间序列的掩码感知状态空间分类器

    MASCIT: A Mask-Aware State Space Classifier for Naturally Irregular Time Series

    [https://arxiv.org/abs/2609.34409](https://arxiv.org/abs/2609.34409)

    提出掩码感知状态空间分类器MASCIT，通过观测掩码与门控时间聚合有效处理异步观测、缺失值等自然不规则性，在34个不规则时间序列数据集上取得最优聚合性能。

    

    自然不规则时间序列同时包含异步观测、缺失值、长度不等和非均匀采样等问题，而密集适配器可能会丢弃时间结构。我们提出了一种面向不规则时间序列的掩码感知状态空间分类器（MASCIT），它向编码器提供观测掩码，并在门控时间聚合中排除无效时间步。在34个不规则时间序列数据集上，MASCIT取得了最强的聚合点估计，并且是唯一在每个数据集上都提供了三种子运行结果的受评测神经模型。MASCIT在六个相互重叠的不规则性指标上均保持了最优的点排名，同时因子消融实验表明部分选择性优于完全选择性。这些结果支持选择性状态空间模型作为自然不规则时间序列分类的有效且可执行的骨干网络。

    arXiv:2609.34409v2 Announce Type: replace-cross  Abstract: Naturally irregular time series combine asynchronous observations, missing values, unequal lengths, and nonuniform sampling, while dense adapters can discard temporal structure. We propose a mask-aware state space classifier for irregular time series (MASCIT), which supplies observation masks to the encoder and excludes invalid steps from gated temporal aggregation. Across 34 irregular time series datasets, MASCIT yielded the strongest aggregate point estimate and was the only evaluated neural model with three-seed results on every dataset. MASCIT retained the lowest point rank across six overlapping irregularity indicators, while factorial ablations favored partial over full selectivity. These results support selective state space models as effective, executable backbones for naturally irregular time series classification.
    
[^328]: MaskCoFT：面向内存高效MoE推理的掩码协同自适应微调

    MaskCoFT: Masked Co-Adaptive Fine-Tuning for Memory-Efficient MoE Inference

    [https://arxiv.org/abs/2609.34077](https://arxiv.org/abs/2609.34077)

    提出MaskCoFT方法，利用可学习二值掩码限制每层的Top-K路由，并通过交叉熵损失协同微调路由器与专家，使专家在卸载推理场景下被高效复用，从而降低MoE模型的内存开销并保持推理性能。

    

    混合专家语言模型的参数规模常常超出单个GPU的内存容量。专家卸载技术将大多数专家保留在主机内存中并按需加载，因此解码速度取决于每个token需要获取的专家数量。缓存和预取只能在路由允许的范围内降低这一开销。仅微调路由器可以重塑路由以复用专家，但由于专家保持冻结，它们无法适应新路由发送给它们的token。我们提出MaskCoFT，一种掩码协同自适应微调方法，仅使用交叉熵损失同时训练路由器和专家。在微调过程中，一个可学习的二值掩码将每层的Top-K路由限制在一个专家子集内，专家则适应被重定向给它们的token。在推理时，学习到的掩码成为一种软先验，用于对专家进行重新排序，因此每个专家仍然可以被选择。我们为Mixtral-8（摘要截断）模拟了每层4个专家的GPU缓存进行实验。

    arXiv:2609.34077v2 Announce Type: replace-cross  Abstract: Mixture-of-experts (MoE) language models often exceed the memory of a single GPU. Expert offloading keeps most experts in host memory and loads them on demand, so decoding speed depends on how many experts each token must fetch. Caching and prefetching reduce this cost only as far as the routing allows. Router-only fine-tuning can reshape the routing to reuse experts, but it keeps the experts frozen, so they cannot adapt to the tokens the new routing sends them. We propose MaskCoFT, a masked co-adaptive fine-tuning method that trains routers and experts together with the cross-entropy loss alone. During fine-tuning, a learnable binary mask restricts the Top-K routing of each layer to a subset of experts, and the experts adapt to the tokens redirected to them. At inference, the learned mask becomes a soft prior that re-ranks experts, so every expert remains selectable. We simulate a GPU cache of 4 experts per layer for Mixtral-8
    
[^329]: ProcGen 泛化差距究竟衡量了什么？动作规则、残余熵与缺失的随机下限

    What Does a ProcGen Generalization Gap Measure? Action Rules, Residual Entropy, and the Missing Random Floor

    [https://arxiv.org/abs/2609.32532](https://arxiv.org/abs/2609.32532)

    该论文提出强化学习的泛化差距应对照“随机下限”（均匀随机策略在同一评估框架和相同关卡上的回报）来解读，并证明测试时动作规则（采样与 argmax）的选择以及动作等效性造成的残余熵会显著改变 ProcGen 基准上泛化结论的含义。

    

    强化学习中的泛化差距（训练关卡上的回报减去留出关卡上的回报）通常在没有参考点的情况下被报告。我们认为，应当对照一个经过实际测量的随机下限来解读它：即在相同评估框架下，均匀随机策略在相同关卡上的回报。在八个 ProcGen 环境上使用 PPO，并在计算受限的预算下（800万步、16个并行环境；其中三个游戏扩展到2500万步），该随机下限改变了标准数字的含义。测试时的动作规则决定了所测量的是哪个策略：在 miner 中，采样策略在留出关卡上的得分是随机下限的5.1倍，而其 argmax 策略在每次运行中的得分都低于该下限，且贪婪评估使两个环境显著低于随机下限。作为收敛诊断指标，原始策略熵在八个环境中标记出六个，但该熵的32-66%来自具有相同效果的动作；对照随机下限，八个环境中采样策略有五个……

    arXiv:2609.32532v2 Announce Type: replace-cross  Abstract: A generalization gap in reinforcement learning, return on training levels minus return on held-out levels, is usually reported without a reference point. We argue that it should be read against a measured random floor: the return of a uniform-random policy on the same levels under the same evaluation harness. On eight ProcGen environments with PPO at a compute-limited budget (8M steps, 16 parallel environments; three games extended to 25M), the floor changes what standard numbers mean. The test-time action rule decides which policy is measured: in miner, the sampled policy scores 5.1x the floor on held-out levels while its argmax scores below it in every run, and greedy evaluation places two environments significantly below the floor. Used as a convergence diagnostic, raw policy entropy flags six of eight environments, but 32-66% of that entropy lies on actions with identical effects; against the floor, five of eight sampled po
    
[^330]: AECSF：用于高维非线性数据同化的自适应集成条件得分滤波

    AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation

    [https://arxiv.org/abs/2609.32411](https://arxiv.org/abs/2609.32411)

    提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。

    

    高维非线性动力系统的贝叶斯状态估计面临着统计精度与计算可行性之间的根本性矛盾：粒子权重可能发生退化坍缩，而高斯集成更新则可能遗漏非高斯的后验结构。基于得分的扩散滤波器提供了一种基于采样的替代方案，但现有的免训练得分滤波器通常依赖启发式的似然修正，由于忽略了与每个含噪反向粒子相关联的系统状态的不确定性，这可能会损害后验估计的精度。为解决这些问题，我们提出了AECSF——一种免训练的自适应集成条件得分滤波器。AECSF基于条件Tweedie恒等式构建了一个解析可处理的得分估计器，将含噪后验得分估计问题转化为在给定含噪反向粒子与观测条件下估计系统状态的条件均值。为估计这些条件……（摘要原文在此处截断）

    arXiv:2609.32411v2 Announce Type: replace-cross  Abstract: Bayesian state estimation for high-dimensional nonlinear dynamical systems entails a fundamental tension between statistical fidelity and computational tractability, as particle weights can collapse, while Gaussian ensemble updates can miss non-Gaussian posterior structure. Score-based diffusion filters offer a sampling-based alternative, but existing training-free score filters often rely on heuristic likelihood corrections, which can compromise posterior accuracy by neglecting uncertainty about the system state associated with each noisy reverse particle. To address these issues, we propose AECSF, a training-free adaptive ensemble conditional score filter. AECSF constructs an analytically tractable score estimator from the conditional Tweedie identity, which recasts noisy posterior score estimation as estimating the conditional mean of the system state given a noisy reverse particle and the observation. To estimate these cond
    
[^331]: 基于更少测量次数的稀疏线性分类器混合模型的高效支撑集恢复

    Efficient Support Recovery of Mixtures of Sparse Linear Classifiers with Fewer Measurements

    [https://arxiv.org/abs/2609.32176](https://arxiv.org/abs/2609.32176)

    本文提出了自适应与非自适应的支撑集恢复方案，在稀疏线性分类器混合模型中同时实现了更少的测量次数和亚线性解码时间，显著优于已有方法。

    

    线性分类器混合模型中的支撑集恢复问题，旨在当数据由多个线性决策规则的混合生成时，识别与底层决策规则相关的特征。具体而言，目标是从符号测量中恢复 $l$ 个未知的 $k$-稀疏向量的支撑集（即非零坐标）。每次测量通过从 $l$ 个向量中均匀随机选择一个向量，并返回其与选定测量向量的内积符号来生成。在本文中，我们提出了自适应和非自适应方案，通过同时减少测量次数并实现亚线性解码时间，显著改进了先前的结果。特别是，与现有方法相比，我们的自适应构造大幅减少了测量次数，同时将解码复杂度从环境维度上的超二次降低至亚线性。我们进一步提供了一种非自适应方案。

    arXiv:2609.32176v2 Announce Type: replace  Abstract: The support recovery problem in mixture of linear classifiers aims to identify the features relevant to the underlying decision rules when data is generated by a mixture of several linear decision rules. In particular, the goal is to recover the support (nonzero coordinates) of $l$ unknown $k$-sparse vectors from sign measurements. Each measurement is generated by selecting one of the $l$ vectors uniformly at random, and returning the sign of its inner product with a chosen measurement vector.   In this paper, we propose adaptive and non-adaptive schemes that significantly improve upon prior results by simultaneously reducing the number of measurements and achieving sublinear decoding time. In particular, our adaptive constructions substantially reduce measurements compared to existing approaches, while also lowering decoding complexity from super-quadratic to sublinear in the ambient dimension. We further provide a non-adaptive sche
    
[^332]: 无固定扩散的不动点：面向收敛测试时计算的隐式神经层束

    Fixed Points Without Fixed Diffusion: Implicit Neural Sheaves for Convergent Test-Time Computation

    [https://arxiv.org/abs/2609.30277](https://arxiv.org/abs/2609.30277)

    提出 SheafDEQ，一种基于自适应神经层束传播的次齐次深度平衡架构，通过可学习的矩阵值层束限制映射实现更丰富的边依赖变换，并在温和条件下保留了隐式图神经网络不动点唯一且可收敛的保证，从而兼顾表达能力和测试时计算的收敛性。

    

    隐式图神经网络（IGNN）将节点表示定义为消息传递算子的不动点，从而实现有效无限深度的传播、与迭代次数无关的参数化以及灵活的测试时计算。然而，这些优势依赖于平衡点的唯一性以及可通过不动点迭代达到该平衡点。现有构造通常对循环更新施加约束以获得这些保证，这限制了平衡点处可用的变换。这引出一个核心问题：IGNN 能否通过更丰富的、依赖边的变换来提升表达能力，同时保留其平衡点表达式的固有优势？我们提出 SheafDEQ，一种具有自适应神经层束传播的次齐次深度平衡架构。其学习到的矩阵值层束限制映射可以对齐、混合或反转相邻节点的表示。在温和的正则性条件下，我们证明 SheafDEQ 具有……（摘要原文在此处截断）

    arXiv:2609.30277v1 Announce Type: new  Abstract: Implicit Graph Neural Networks (IGNNs) define node representations as fixed points of message-passing operators, enabling effectively infinite-depth propagation, iteration-independent parameterization, and flexible test-time computation. Yet these benefits depend on the equilibrium being unique and attainable by fixed-point iteration. Existing constructions often impose constraints on recurrent updates to obtain these guarantees, limiting the transformations available at equilibrium. This raises a central question: can IGNNs gain expressiveness through richer, edge-dependent transformations while retaining the inherent strengths of their equilibrium formulation? We introduce SheafDEQ, a subhomogeneous deep-equilibrium architecture with adaptive neural-sheaf propagation. Its learned, matrix-valued sheaf restriction maps can align, mix, or reverse neighbouring representations. Under mild regularity conditions, we prove that SheafDEQ admits
    
[^333]: 将预训练大语言模型改造为高保真连续文本自编码器

    Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders

    [https://arxiv.org/abs/2609.27248](https://arxiv.org/abs/2609.27248)

    本文提出LLMAE方法，通过在预训练语言模型内部引入固定长度潜瓶颈，将其改造为高保真连续文本自编码器，可近乎完美地重建长达1024个token的文本序列。

    

    下一个词预测使自回归语言模型具备了高度流畅的文本生成能力，但它只能通过序列化分解间接地表示全局结构。相比之下，高保真自编码器已成为图像生成领域的标准基础组件，使生成模型能够在连续潜空间上运行；而文本领域则缺乏同样忠实的连续表示。我们提出 LLMAE，一种将预训练的仅解码器（decoder-only）语言模型改造为连续文本自编码器的方法，其核心是在模型内部激活中引入一个中间的固定长度潜瓶颈。该方法以参数高效的 270M Gemma 3 模型实例化，利用结构化注意力掩码、LoRA 适配和 KL 正则化来学习一个自编码接口，从而充分借助原始大语言模型的生成先验。我们训练 LLMAE 重建最长 1024 个 token 的文本序列，在该任务上取得显著提升，实现了近乎完美的重建效果。

    arXiv:2609.27248v1 Announce Type: new  Abstract: Next-token prediction has enabled highly fluent autoregressive language models, but it represents global structure only indirectly through sequential factorization. In contrast, high-fidelity autoencoders have become a standard primitive in image generation, enabling generative models to operate over continuous latent spaces; text lacks a comparably faithful continuous representation. We propose LLMAE, a method for repurposing a pretrained decoder-only language model as a continuous text autoencoder by exposing an intermediate fixed-length latent bottleneck within its internal activations. Instantiated with a parameter-efficient 270M Gemma 3 model, LLMAE uses structured attention masks, LoRA adaptation, and KL regularization to learn an autoencoding interface that leverages the generative prior of the original LLM. We train LLMAE to reconstruct text sequences up to 1024 tokens, significantly improving on this task to achieve near-perfect
    
[^334]: 用于约束采样的惩罚性非可逆朗之万算法

    Penalized Nonreversible Langevin for Constrained Sampling

    [https://arxiv.org/abs/2609.25381](https://arxiv.org/abs/2609.25381)

    提出了将平方距离惩罚与非可逆斜对称扰动相结合的朗之万算法以实现紧凸集上的约束采样，并在对数索博列夫不等式与漂移收缩条件下给出了非渐近的总变差和 2-Wasserstein 误差界。

    

    我们提出了用于从 $\pi(x)\propto e^{-f(x)}\mathbf 1_{\mathcal C}(x)$ 中采样的惩罚性非可逆朗之万算法，其中 $\mathcal C\subset\mathbb R^d$ 是一个紧凸集。这些算法将平方距离惩罚与能够保持惩罚吉布斯分布的常数型或相容的状态依赖斜对称扰动相结合。对于光滑且可能非凸的 $f$，我们在对数索博列夫不等式条件下推导了全梯度算法的非渐近总变差界。当可获得无偏随机梯度时，我们在适应性二次度量下，基于全漂移项的全局收缩性和利普希茨条件建立了 2-Wasserstein 界。对于固定的惩罚参数，相对于惩罚吉布斯分布的误差以指数速度衰减到一个 $\mathcal{O}(\sqrt{\eta})$ 邻域，其中 $\eta$ 为步长。我们还界定了惩罚吉布斯分布与目标分布之间的差异（摘要在此处被截断）。

    arXiv:2609.25381v1 Announce Type: cross  Abstract: We propose penalized nonreversible Langevin algorithms for sampling from $\pi(x)\propto e^{-f(x)}\mathbf 1_{\mathcal C}(x)$, where $\mathcal C\subset\mathbb R^d$ is a compact convex set. The algorithms combine a squared distance penalty with constant or compatible state dependent skew symmetric perturbations that preserve the penalized Gibbs distribution. For smooth, possibly nonconvex $f$, we derive nonasymptotic total variation bounds for the full gradient algorithm under a log Sobolev inequality. When unbiased stochastic gradients are available, we establish $2$-Wasserstein bounds under global contraction and Lipschitz conditions on the full drift in an adapted quadratic metric. For a fixed penalty parameter, the error relative to the penalized Gibbs distribution decays exponentially to an $\mathcal{O}(\sqrt{\eta})$ neighborhood, where $\eta$ is the stepsize. We also bound the discrepancy between the penalized Gibbs distribution and
    
[^335]: Video DeltaNet：一种面向直播视频生成的视频原生混合注意力机制

    Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation

    [https://arxiv.org/abs/2609.20744](https://arxiv.org/abs/2609.20744)

    提出Video DeltaNet（VDN），通过将局部Softmax注意力与引入视频增量注意力（VDA）的双向线性记忆相结合的混合架构，解决视频扩散模型中的注意力计算瓶颈，实现高质量的直播视频生成。

    

    视频扩散模型在去噪过程中需要反复处理长时空token序列，这使得注意力机制成为主要的计算瓶颈。线性注意力提供了一种有吸引力的替代方案，并已在近期的大型语言模型中得到广泛采用，但直接将其应用于视频模型往往无法保留高质量生成所需的细粒度交互。我们提出了Video DeltaNet（VDN），它将局部Softmax注意力与双向线性记忆相结合，用于长程视频上下文建模。其线性分支引入了视频增量注意力（Video Delta Attention，VDA），通过联合整合每帧的空间token，实现每帧一次的记忆更新。分离的输出投影和可学习的门控机制用于校准两个分支，同时采用分阶段的教师对齐训练策略，将新通路逐步引入预训练模型。我们将VDN实例化于MiniMax H3，将该混合机制应用于视频到视频的交互，同时保留Softmax注意力（原文在此处截断）。

    arXiv:2609.20744v1 Announce Type: new  Abstract: Video diffusion models repeatedly process long spatiotemporal token sequences during denoising, making attention a major computational bottleneck. Linear attention offers an appealing alternative and has been widely adopted in recent large language models, but directly applying it to video models often fails to preserve the fine-grained interactions required for high-quality generation. We present Video DeltaNet (VDN), which combines local Softmax attention with bidirectional linear memory for long-range video context. Its linear branch introduces Video Delta Attention (VDA), which updates memory once per frame by jointly incorporating its spatial tokens. Separate output projections and learnable gates calibrate the two branches, while a staged teacher-alignment recipe progressively introduces the new pathway into pretrained models. We instantiate VDN on MiniMax H3, applying the hybrid to video-to-video interactions while retaining Softm
    
[^336]: 基于大语言模型的丰富辅助信息贝叶斯优化

    Bayesian Optimization with Rich Auxiliary Information via LLMs

    [https://arxiv.org/abs/2609.19437](https://arxiv.org/abs/2609.19437)

    本文提出三种利用大语言模型将丰富辅助信息（如训练曲线、专家笔记和先验知识）融入贝叶斯优化的方法，在超参数优化基准和真实核聚变优化任务中始终优于标准BO及现有LLM优化方法。

    

    贝叶斯优化（BO）被广泛用于优化昂贵的黑盒函数，然而许多现实世界的优化问题包含比单纯函数评估丰富得多的信息。例如超参数优化中的训练曲线、科学实验中的专家笔记和图像，以及关于最优解可能位置的先验知识。我们证明大语言模型（LLM）能够有效利用这些丰富的辅助信息来指导优化。基于这些发现，我们开发了三种使用大语言模型将辅助信息纳入贝叶斯优化的方法。在超参数优化基准测试和一个真实世界的核聚变优化任务中，我们的方法始终优于标准贝叶斯优化和现有的基于大语言模型的优化方法。我们的结果证明了大语言模型在贝叶斯优化中利用丰富辅助信息的有效性。

    arXiv:2609.19437v1 Announce Type: new  Abstract: Bayesian Optimization (BO) is widely used for optimizing expensive black-box functions, yet many real-world optimization problems contain substantially richer information than function evaluations alone. Examples include training curves in hyperparameter optimization, expert notes and images in scientific experimentation, and prior knowledge about where optima may lie. We show that large language models (LLMs) can effectively leverage such rich auxiliary information to guide optimization. Motivated by these findings, we develop three methods for incorporating auxiliary information into BO using LLMs. Across hyperparameter optimization benchmarks and a real-world nuclear fusion optimization task, our methods consistently outperform both standard BO and existing LLM-based optimization approaches. Our results demonstrate the effectiveness of LLMs for leveraging rich auxiliary information in BO.
    
[^337]: TACTICS：面向机器翻译的分类体系感知智能语料库抽样

    TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation

    [https://arxiv.org/abs/2609.17956](https://arxiv.org/abs/2609.17956)

    该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。

    

    大规模机器翻译（MT）系统通常在从语料库中随机抽取的样本上进行评估，而语料库的分布构成本质上取决于其构建方式。这样的样本仅继承了语料库碰巧包含的语言现象，而非系统必须处理的完整空间——这些现象既涵盖规则约束的惯例（术语、标点、货币格式），也包括依赖上下文的现象（语气、敬语、文档级连贯性），因而无法为鲁棒性评估提供覆盖保证。我们提出了TACTICS（分类体系感知的覆盖优化智能语料库抽样），它将覆盖率重新定义为一个显式目标。TACTICS从本地化风格指南中归纳出层次化分类体系，据此对语段进行分类，并在固定预算下选择子集，联合优化稀有类别的覆盖率、文档级连贯性以及对完整语料库的分布保真度。该方法应用于跨四种……（评估场景）的机器翻译评估。

    arXiv:2609.17956v1 Announce Type: new  Abstract: Large-scale machine-translation (MT) systems are typically evaluated on random samples from a corpus whose distributional composition is an artifact of how it was assembled. Such a sample inherits the phenomena the collection happens to contain rather than the full space a system must handle, spanning rule-governed conventions (terminology, punctuation, currency formatting) and context-dependent phenomena (tone, honorifics, document-level coherence), and thus provides no coverage guarantee for assessing robustness. We propose TACTICS (Taxonomy-Aware Coverage-opTimized Intelligent Corpus Sampling), which recasts coverage as an explicit objective. TACTICS induces a hierarchical taxonomy from a locale style guide, classifies segments against it, and selects a fixed-budget subset jointly optimizing coverage of rare categories, document-level coherence, and distributional fidelity to the full corpus. Applied to MT evaluation across four trans
    
[^338]: 迈向识别导致幻影迁移的数据集偏见

    Towards Identifying the Dataset Biases Causing Phantom Transfer

    [https://arxiv.org/abs/2609.14449](https://arxiv.org/abs/2609.14449)

    本文提出一种基于Sentence BERT嵌入的简单特征签名方法，能够在教师模型已知时以0.83的马修斯相关系数识别数据集中隐藏的偏见主题，为检测导致幻影迁移的数据集偏见提供了新途径。

    

    最近的研究表明，教师模型可以通过一个数据集将偏见传递给学生模型，即使该数据集中所有对该偏见的明确引用都已被过滤掉；而且即使知道要寻找什么样的偏见，也没有任何数据层面的防御方法能够可靠地移除或检测到它。为了揭示这些偏见的隐藏痕迹，我们展示了基于Sentence BERT嵌入的简单特征签名，在攻击者使用的教师模型已知的情况下，能够以0.83的马修斯相关系数识别此类偏见的主题；在教师模型未知的情况下，该系数为0.46。此外，我们观察到不同的教师模型似乎通过不同的词汇来表达相同的偏见。

    arXiv:2609.14449v1 Announce Type: new  Abstract: Recent work has shown that a teacher model can transfer a bias to a student through a dataset from which every explicit reference to that bias has been filtered out, and that no data-level defense reliably removes or detects it even when knowing what bias to look for. Aiming to shed light on the hidden traces of these biases, we show that a simple signature based on Sentence BERT embeddings can identify the topic of such a bias with a Matthews correlation coefficient of 0.83 if the teacher model used by the attacker is known and 0.46 if it is not. Additionally, we observe that different teacher models appear to express the same bias through different vocabulary.
    
[^339]: 使用GRACE预测碰撞截面：通过早期融合实现几何残差加合物条件化

    Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion

    [https://arxiv.org/abs/2609.12223](https://arxiv.org/abs/2609.12223)

    本文提出GRACE模型，通过早期融合的几何残差加合物条件化方法调整预训练分子几何编码器，将加合物感知的残差学习目标与编码器内的加合物条件化机制相结合，显著提升了对气相分子离子碰撞截面的三维预测精度。

    

    碰撞截面（CCS）源自离子迁移谱质谱技术，是分子注释的常用描述符。对机器学习模型而言，预测CCS具有挑战性，因为它反映了气相分子离子的大小、形状和电离状态。大多数预测器要么忽略显式的三维结构，要么将加合物类型作为后期处理的类别特征，这限制了模型捕获加合物相关几何效应的能力。我们提出了GRACE（通过早期融合实现几何残差加合物条件化），这是一个三维CCS预测器，通过早期融合的几何残差加合物条件化来调整预训练的分子几何编码器。GRACE结合了两个归纳偏置：相对于加合物感知的物理描述符基线的残差学习目标，以及通过可学习的加合物标记和低秩注意力适配器在编码器内部实现的加合物条件化。我们在包含超过9,000个实验分子的精选数据集上对该模型进行了评估。

    arXiv:2609.12223v1 Announce Type: new  Abstract: Collision cross section (CCS), derived from ion mobility mass spectrometry, is a common descriptor for molecular annotation. Prediction is challenging for machine learning models because it reflects the size, shape, and ionization state of a gas-phase molecular ion. Most predictors either ignore explicit 3D structure or treat adduct identity as a late categorical feature, which limits their ability to capture adduct-dependent geometric effects. We present GRACE (Geometric Residual Adduct Conditioning via Early-fusion), a 3D CCS predictor that adapts a pretrained molecular geometry encoder using geometric residual adduct conditioning via early fusion. GRACE combines two inductive biases: a residual objective relative to an adduct-aware physical descriptor baseline and adduct conditioning within the encoder via a learned adduct token and low-rank attention adapters. We evaluate the model on a curated set of over 9,000 experimental molecule
    
[^340]: Suan：纠正大语言模型中的直接偏好安全对齐

    Suan: Rectifying Direct Preference Safety Alignment in Large Language Models

    [https://arxiv.org/abs/2609.08634](https://arxiv.org/abs/2609.08634)

    Suan是一种新颖的偏好优化算法，通过直接在梯度层面构建优化目标（绕过标准变分推导），使大语言模型在实现卓越安全对齐的同时完全保留回应实用性。

    

    将强大的安全防护机制集成到大语言模型（LLM）中，对于提供有用且无害的回应至关重要。尽管专有系统展现出可靠的安全控制，但其底层方法和权衡取舍在很大程度上仍未公开。在开放权重模型中实现相当的安全性仍然是一个持续的挑战，因为经过后训练的模型变体经常出现过拒答和整体质量下降的问题。为了克服这些缺陷，我们提出了Suan，一种新颖的偏好优化算法。与现有方法不同，我们直接在梯度层面构建优化目标，绕过了标准的变分推导。由此，我们获得了更具可解释性和更稳健的训练动态。在多种竞争性基线和基准测试上的广泛评估表明，Suan在实现卓越安全对齐的同时，完全保留了回应的实用性。

    arXiv:2609.08634v1 Announce Type: cross  Abstract: Integrating robust safety guardrails into Large Language Models (LLMs) is essential for delivering helpful yet harmless responses. While proprietary systems exhibit reliable safety controls, their underlying methodologies and trade-offs remain largely undisclosed. Achieving comparable security in open-weight models remains a persistent challenge, as post-trained variants frequently suffer from over-refusal and degraded general quality. To overcome these drawbacks, we introduce Suan, a novel preference optimization algorithm. Unlike existing methods, we formulate the optimization objective directly at the gradient level, bypassing the standard variational derivation. As a result, we obtain more interpretable and robust training dynamics. Extensive evaluations across a diverse suite of competitive baselines and benchmarks demonstrate that Suan achieves superior safety alignment while fully preserving response utility.
    
[^341]: LayerRoute：面向视觉-语言-动作策略的动作条件混合层路由

    LayerRoute: Action-Conditioned Mixture-of-Layers Routing for Vision-Language-Action Policies

    [https://arxiv.org/abs/2609.06079](https://arxiv.org/abs/2609.06079)

    提出LayerRoute，一个动作条件的混合层路由接口，使VLA策略能够根据具体动作需求自适应地访问和混合VLM不同层次的表示，突破了传统固定层分配的限制。

    

    arXiv:2609.06079v1 公告类型：新论文 摘要：视觉-语言-动作（VLA）策略利用预训练的视觉-语言模型（VLM）来指导机器人控制的动作生成。VLM提供跨层演化的分层视觉-语义表示，从局部视觉几何到抽象的、与语言对齐的语义；因此，不同的操作任务可能需要不同的层表示混合。同时，动作模块在动作计算过程中维护不断演化的中间表示，这些表示可能为后续决策提供有用信息。然而，现有的VLA接口在表示访问方面的灵活性有限：VLM信息通过为每个动作层固定的层分配来暴露，而中间动作状态仅通过残差流隐式传播，缺乏显式复用。我们提出了LayerRoute，一个动作条件的表示路由接口，使VLA策略能够自适应地访问VLM的各层表示。

    arXiv:2609.06079v1 Announce Type: new  Abstract: Vision-Language-Action (VLA) policies leverage pretrained vision-language models (VLMs) to guide action generation for robot control. VLMs provide hierarchical visual-semantic representations that evolve across layers, from local visual geometry to abstract, language-aligned semantics; different manipulation tasks may therefore require different mixtures of layer representations. Meanwhile, the action module maintains intermediate representations that evolve throughout action computation and may provide useful information for subsequent decisions. However, existing VLA interfaces offer limited flexibility in representation access: VLM information is exposed through fixed layer assignments for each action layer, while intermediate action states are only propagated implicitly through residual streams without explicit reuse. We introduce LayerRoute, an action-conditioned representation routing interface that enables adaptive access to VLM l
    
[^342]: 分形维度预测角度编码数据中量子核的坍缩

    Fractal dimension predicts quantum kernel collapse in angle-encoded data

    [https://arxiv.org/abs/2609.00475](https://arxiv.org/abs/2609.00475)

    该论文提出用数据的相关性分形维度 D2 作为先验量子比特数预算，可准确预测并避免角度编码量子核的几何坍缩，使量子核在真实硬件上以最小比特数保持有效。

    

    角度编码的量子核在表格数据上，当特征映射宽度超过数据内在维度时会发生坍缩。我们提出将相关性分形维度 D2 作为一种先验的量子比特预算：编码由 FD-ASE 选取的 D2 个坐标，而不是采用 PCA-95% 宽度或全部 E 个属性。在九个数据集和状态向量模拟器（n=32）上，单层 ZZ 保真度核在 q=D2 时仍保持几何活性，而同样的核在 PCA-95% 宽度下已经坍缩。该预算依赖于映射方式：直积态映射和 IQP 映射会超出该预算，而第二层 ZZ 则会低于该预算。打包的密集角度编码和重上传编码在分形 q 下仍然存活，但当把 PCA-95% 特征堆叠到这些量子比特上时则不再存活。缩小角度带宽会使 ZZ 拐点后移，拉伸带宽则会使核更早失效。在 IBM Quantum（ibm_fez，256 次采样，n=8）上，分形宽度下的单层 ZZ 核与精确核相匹配（MAE 为 0.021）；超过该宽度后……

    arXiv:2609.00475v1 Announce Type: cross  Abstract: Angle-encoded quantum kernels on tabular data collapse when the feature map is wider than the intrinsic dimension of the data. We propose the correlation fractal dimension D2 as an a priori qubit budget: encode D2 coordinates chosen by FD-ASE instead of the PCA-95% width or all E attributes. On nine data sets and a statevector simulator (n= 32), a one-layer ZZ fidelity kernel at q=D2 stays geometrically alive while the same kernel at the PCA-95% width has already collapsed. The budget is map-dependent: product-state and IQP maps overshoot it; a second ZZ layer undershoots it. Packed dense-angle and re-uploading encodings still live at the fractal q, but not when PCA-95% features are stacked onto those qubits. Shrinking the angle bandwidth moves the ZZ knee later; stretching it kills the kernel earlier. On IBM Quantum (ibm_fez, 256 shots, n=8) the one-layer ZZ kernel at the fractal width matches the exact kernel (MAE 0.021); past that w
    
[^343]: 向量值线性回归中加权数据选择的精确恢复阈值

    Exact Recovery Thresholds for Weighted Data Selection in Vector-Valued Linear Regression

    [https://arxiv.org/abs/2608.30254](https://arxiv.org/abs/2608.30254)

    本文解决了COLT 2025公开问题中回归任务加权数据选择的阈值部分，证明了在每个有限数据集上恢复全数据损失所需的最小加权样本预算恰好为 (m+1)d，并精确确定了加权选择曲线在阈值附近和张成预算处的取值。

    

    我们解决了COLT 2025公开问题《回归任务的数据选择》（作者为Hanneke、Moran、Shlimovich和Yehudayoff）中问题4的阈值部分。在以平方损失 ℓ₍ₓ,ᵧ₎(W)=|Wx-y|₂² 定义的向量值线性回归中，其中 x∈ℝᵈ，y∈ℝᵐ，学习器为最小Frobenius范数的经验风险最小化器，我们证明了在每个有限数据集上都能恢复全数据损失的加权样本最小预算恰好为 n*(d,m)=(m+1)d。我们进一步确定了加权选择曲线 F_w(d,m,n) 的另外两个取值：在接近阈值的预算处，F_w(d,m,(m+1)d-1)=1+1/(dm²)；在张成预算处，对每个 m 都有 F_w(d,m,d)=d+1，而当 n<d 时 F_w(d,m,n)=∞。对于最小的开放中间情形 (d,m)=(2,2)，我们证明了 F_w(2,2,3)∈[13/8,15/8] 与 F_w(2,2,4)∈[5/4,3/2]，并将猜测的精确值 13/8 和 5/4 归约为一个有限矩问题。

    arXiv:2608.30254v1 Announce Type: new  Abstract: We resolve the threshold part of Question 4 of the COLT 2025 open problem "Data Selection for Regression Tasks" of Hanneke, Moran, Shlimovich and Yehudayoff. In vector-valued linear regression with square loss $\ell_{(x,y)}(W)=|Wx-y|_2^2$, where $x\in\mathbb{R}^d$, $y\in\mathbb{R}^m$ and the learner is the empirical risk minimizer of minimal Frobenius norm, we prove that the minimal budget of weighted examples that recovers the full-data loss on every finite dataset is exactly $n^*(d,m)=(m+1)d$. We further determine two more values of the weighted selection profile $F_w(d,m,n)$: at the near-threshold budget, $F_w(d,m,(m+1)d-1)=1+\frac{1}{dm^2}$, and at the spanning budget, $F_w(d,m,d)=d+1$ for every $m$, while $F_w(d,m,n)=\infty$ for $n<d$. For the smallest open intermediate cell $(d,m)=(2,2)$ we prove $F_w(2,2,3)\in[13/8,15/8]$ and $F_w(2,2,4)\in[5/4,3/2]$, reduce the conjectured exact values $13/8$ and $5/4$ to a finite moment problem 
    
[^344]: 线性回归中加权数据选择的精确风险比率

    Exact Risk Ratios for Weighted Data Selection in Linear Regression

    [https://arxiv.org/abs/2608.28007](https://arxiv.org/abs/2608.28007)

    本文解决了 Hanneke 等人提出的加权数据选择公开问题，精确确定了最小范数 ERM 下的最坏风险比率 F_w(d,n) 的多个取值，包括证明 F_w(d,2d-1)=1+1/d，并给出了中间预算 n=d+k 情形下基于平衡划分调和量的紧致下界 1+Γ_{d,k}。

    

    Hanneke、Moran、Shlimovich 和 Yehudayoff（COLT 2025）提出了如下公开问题：一个选择者看到有限数据集 D ⊆ R^d × R，挑选至多 n 个样本并赋予非负权重，然后将加权最小二乘目标交给最小范数 ERM 求解。记 F_w(d,n) 为返回预测器在全部数据 D 上的损失与最优损失之间的最坏情况比率，他们证明了当 n<2d 时 F_w(d,n)=∞。本文在若干情形下确定了该值。对每个 d，我们证明了 F_w(d,2d-1)=1+1/d，这证实了原始笔记中一个未给出证明的论断。我们进一步证明了 F_w(3,4)=5/3 和 F_w(4,5)=2，这是端点公式未能覆盖的两个最小的格子。对每个中间预算 n=d+k，我们证明了下界 F_w(d,d+k) ≥ 1+Γ_{d,k}，其中 Γ_{d,k} 是关于平衡划分的一个显式调和量，并且我们证明该下界即为精确的最小值。

    arXiv:2608.28007v1 Announce Type: new  Abstract: Hanneke, Moran, Shlimovich and Yehudayoff (COLT 2025) posed the following open problem. A selector sees a finite dataset $D \subseteq \mathbb{R}^d \times \mathbb{R}$, picks at most $n$ examples together with nonnegative weights, and hands the weighted least squares objective to the minimum-norm ERM. Writing $F_w(d,n)$ for the worst-case ratio between the loss of the returned predictor on all of $D$ and the optimal loss, they proved $F_w(d,n)=\infty$ for $n<2d$. We determine this value in several cases. For every $d$ we prove $F_w(d,2d-1)=1+1/d$, which confirms a claim stated without proof in the original note. We further prove $F_w(3,4)=5/3$ and $F_w(4,5)=2$, the two smallest cells not covered by the endpoint formula. For every intermediate budget $n=d+k$ we prove the lower bound $F_w(d,d+k) \ge 1+\Gamma_{d,k}$, where $\Gamma_{d,k}$ is an explicit harmonic quantity over balanced partitions, and we show that this bound is the exact minima
    
[^345]: 推理时通过读出反馈引导循环推理器

    Steering Recurrent Reasoners at Inference Time with Readout Feedback

    [https://arxiv.org/abs/2608.24136](https://arxiv.org/abs/2608.24136)

    本文提出读出反馈（RoFB），一种无需重训练的推理时干预方法，通过将中间预测转化为耦合力注入潜在动态，显著提升循环模型在推理任务中的表现。

    

    摘要：循环模型通过共享计算块重复更新潜在状态，已成为解决复杂推理任务的有力架构。现有的推理时方法通过增加运行步数或采样更多轨迹来扩展计算量，但忽略了每条轨迹内部揭示的信息。在此，我们表明循环模型可以在推理时通过利用其自身的读出概率来引导潜在动态，而无需重新训练。我们引入了读出反馈（RoFB），这是一种测试时干预方法，将中间预测转换为逐令牌的成对耦合力，注入到潜在动态中。在三个循环模型（AKOrN、ItrSA++、TRM）上针对数独和迷宫任务，RoFB在六个模型-任务组合中的四个上取得了显著改进，实现了仅通过增加步数或从多个轨迹中选择无法达到的性能，且计算成本相当或更低。这些结果表明...

    arXiv:2608.24136v1 Announce Type: new  Abstract: Recurrent models, which repeatedly update latent states with shared computation blocks, have emerged as powerful architectures for solving complex reasoning tasks. Existing inference-time methods scale computation by running more steps or sampling more trajectories, but ignore information revealed within each trajectory. Here we show that recurrent models can be improved at inference time by using their own readout probabilities to steer latent dynamics without retraining. We introduce Readout Feedback (RoFB), a test-time intervention that converts intermediate predictions into token-wise pairwise coupling forces injected into the latent dynamics. Across three recurrent models (AKOrN, ItrSA++, TRM) on Sudoku and Maze, RoFB yields clear gains in four of six model-task pairs, achieving performance unattainable by merely running more steps or selecting from multiple trajectories, at comparable or lower computational cost. These results sugg
    
[^346]: 移动端大模型：视觉语言模型的高效2.7位量化

    Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs

    [https://arxiv.org/abs/2608.21134](https://arxiv.org/abs/2608.21134)

    提出一种无需训练数据的2.7位量化框架，将视觉语言模型压缩至3.7 GB，同时保持视觉问答性能，适用于移动设备高效推理。

    

    摘要：由于视觉语言模型（VLMs）在部署到移动设备时面临显著的内存和计算需求挑战，我们提出了一种用于在资源受限硬件上高效推理的VLM量化框架。我们的方法结合了一个量化流程，该流程利用模型自身生成训练数据，无需访问训练设置，并采用一种新颖的每参数2.7位格式，支持在Arm CPU上高效执行。我们通过将Llama 3.2 11B视觉指令模型压缩至3.7 GB（使用8位激活），在标准视觉问答任务集上保持强大性能，验证了我们的方法。

    arXiv:2608.21134v1 Announce Type: cross  Abstract: Deploying vision-language models (VLMs) on mobile devices is challenging due to their significant memory and compute requirements. We present a framework for quantizing VLMs for efficient inference on resource-constrained hardware. Our approach combines a quantization pipeline that uses the model itself to generate training data and does not require access to the training setup, with a novel 2.7-bit-per-parameter format supporting efficient execution on Arm CPUs. We validate our approach by compressing the Llama 3.2 11B Vision Instruct model to 3.7 GB with 8-bit activations, preserving strong performance on a set of standard visual question answering tasks.
    
[^347]: 一种在将世界模型与现实对齐中不可约的量子优势

    An Irreducible Quantum Advantage in Aligning World Models with Reality

    [https://arxiv.org/abs/2608.19779](https://arxiv.org/abs/2608.19779)

    本文证明即使真实世界是经典的，经典世界模型也无法完美对齐代理策略，而量子模型可能提供不可约的优势。

    

    世界模型提供了真实世界的数字模拟，使代理能够在昂贵的现实部署之前进行训练和测试。在每个时间步，它们接收一个动作并生成与真实世界统计匹配的观测和奖励。在复杂环境中，当前结果取决于遥远的过去事件，这需要记忆。人们可能期望，通过增加记忆，我们总能构建一个足够准确的模型，以使真实和虚拟世界的最优代理策略对齐。我们表明，对于经典世界模型，即使真实世界本身是经典的，这也是错误的。我们构造了真实世界，其中每个有限经典模型沿着相同的可能轨迹失败：它要么在真实世界明显偏好某个动作时失去区分动作的能力，要么反复将最高期望奖励分配给次优动作。其期望奖励估计也保留了一个不可消失的...

    arXiv:2608.19779v1 Announce Type: cross  Abstract: World models provide digital simulacra of the true world, allowing agents to be trained and tested before costly real-world deployment. At each time step, they receive an action and generate an observation and reward matching the statistics of the true world. In complex environments where present outcomes depend on events far in the past, this requires memory. One might expect that, by increasing memory, we can always build a model accurately enough to align the optimal agent policies of the real and virtual worlds. We show that this is false for classical world models, even when the true world itself is classical. We construct true worlds for which every finite classical model fails along the same possible trajectory: it either loses the ability to distinguish actions when the true world clearly prefers one, or repeatedly assigns the highest expected reward to suboptimal actions. Its expected-reward estimates also retain a nonvanishin
    
[^348]: 带噪声锚点对齐的几何数据扰动用于隐私保护协作学习

    Geometric Data Perturbation with Noisy-Anchor Alignment for Privacy-Preserving Collaborative Learning

    [https://arxiv.org/abs/2608.18749](https://arxiv.org/abs/2608.18749)

    本文提出带噪声锚点对齐的几何数据扰动方法，在分析者-参与者共谋下平衡隐私保护与模型效用，实现了既抵抗共谋攻击又保持协作学习性能的目标。

    

    几何数据扰动（GDP）能够实现一次性、隐私保护的协作学习：每个参与者对其私有数据应用距离保持变换，仅将所得表示上传给中心分析者。我们研究了在分析者-参与者共谋下的GDP，其中分析者结合所有上传的表示与共谋参与者披露的私有数据和变换，以恢复非共谋参与者的私有数据。参与者特定的独立变换能抵抗这种攻击，但会将参与者的数据映射到不兼容的表示空间，从而降低下游模型性能。来自数据协作（DC）分析的共享锚点对齐恢复了兼容性并提高了效用，但我们表明，披露DC锚点矩阵即使在共谋存在的情况下也能精确恢复非共谋参与者的私有数据。直接在扰动上添加噪声可以缓解此问题，但会损害效用。我们提出了一种新颖的带噪声锚点对齐方法，在保持GDP的隐私保证的同时，实现了与无噪声对齐相当的效用。我们的理论分析刻画了在共谋下的隐私-效用权衡，大量实验验证了所提出方法的有效性。

    arXiv:2608.18749v1 Announce Type: new  Abstract: Geometric Data Perturbation (GDP) enables one-shot, privacy-preserving collaborative learning: each participant applies a distance-preserving transformation to its private data and uploads only the resulting representation to a central analyst. We study GDP under analyst-participant collusion, in which the analyst combines all uploaded representations with the private data and transformations disclosed by colluding participants to recover a non-colluding participant's private data. Participant-specific independent transformations resist this attack but map participants' data into incompatible representation spaces, degrading downstream model performance. Shared-anchor alignment from Data Collaboration (DC) analysis restores compatibility and improves utility, but we show that disclosing the DC anchor matrix enables exact recovery of non-colluding participants' private data even in the presence of collusion. Adding noise directly to the p
    
[^349]: 针对极端事件的生成模型微调：基于CVaR惩罚的Wasserstein梯度流

    Fine-Tuning Generative Models for Extreme Events via CVaR-Penalized Wasserstein Gradient Flows

    [https://arxiv.org/abs/2608.11544](https://arxiv.org/abs/2608.11544)

    提出了一种基于CVaR惩罚的Wasserstein梯度流方法，无需先验知识即可微调生成模型以捕捉重尾分布和极端事件，克服了标准生成器在尾部欠采样时速度消失的局限。

    

    arXiv:2608.11544v1 公告类型：交叉 摘要：我们提出了CVaR惩罚生成粒子算法（CVaR-GPA），这是一种鲁棒、尾部无关的算法，用于微调生成模型以学习重尾分布并捕捉极端事件，无需对目标的尾部特征有任何先验知识或估计。该方法是将Lipschitz正则化的Kullback-Leibler（KL）散度与条件风险价值（CVaR）差异项惩罚相结合的Wasserstein梯度流：Lipschitz正则化的KL散度在目标分布的最小假设下实现鲁棒学习，而CVaR惩罚恢复了在欠采样尾部中过早消失的速度。该惩罚流具有有界但非Lipschitz的速度场，这不同于标准生成器的Lipschitz传输映射（后者保留轻尾源的尾部行为），从而能够向更重尾的目标传输。为了定义这一f...

    arXiv:2608.11544v1 Announce Type: cross  Abstract: We propose CVaR-penalized Generative Particle Algorithm (CVaR-GPA), a robust, tail-agnostic algorithm for fine-tuning generative models to learn heavy-tailed distributions and capture extreme events, requiring no prior knowledge or estimation of the target's tail characteristics. The method is the Wasserstein gradient flow of the Lipschitz-regularized Kullback-Leibler (KL) divergence penalized by a Conditional Value-at-Risk (CVaR) discrepancy term: the Lipschitz-regularized KL divergence enables robust learning under minimal assumptions on the target distribution, while the CVaR penalty restores the velocity that otherwise vanishes prematurely in the under-sampled tails. The penalized flow admits a bounded but non-Lipschitz velocity field. This departs from the Lipschitz transport maps of standard generators, which preserve the tail behavior of a light-tailed source, and enables transport toward heavier-tailed targets. To define this f
    
[^350]: 面向机器人仿真到现实部署的、基于统计验证的安全鲁棒神经策略学习

    Safe and Robust Neural Policy Learning with Statistical Verification for Sim-to-Real Deployment in Robotics

    [https://arxiv.org/abs/2608.06481](https://arxiv.org/abs/2608.06481)

    本文提出一种课程驱动的闭环框架，将基于场景的进化策略与统计模型检测验证相结合，在协同优化策略性能的同时逐步扩大安全操作边界，最终生成带有统计验证安全与性能保证的神经控制器，助力可靠的仿真到现实部署。

    

    在仿真中合成安全且鲁棒的神经控制器，以实现可靠的仿真到现实部署，仍然是机器人领域的一项关键挑战。现有的基于学习的方法通常缺乏在明确定义的操作区域内的安全性和性能保证，而训练后验证技术在检测到安全违规时也没有提供改进控制器的机制。为了弥合这一差距，我们提出了一个课程驱动的框架，在闭环流程中将基于场景的进化策略与基于统计模型检测的验证紧密集成。从候选区域出发，该方法在协同优化策略性能的同时，逐步扩大其安全操作边界。在终止时，它会输出一个神经控制器以及一个经过统计验证的安全性和性能保证的区域。在Cartpole和3D四旋翼基准上的广泛评估显示6.14倍和……（原文摘要在此处截断）

    arXiv:2608.06481v2 Announce Type: replace-cross  Abstract: Synthesizing safe and robust neural controllers in simulation for reliable sim-to-real deployment remains a critical challenge in robotics. Existing learning-based methods typically lack safety and performance guarantees over an explicitly defined operating region, while post-training verification techniques provide no mechanism to refine controllers when safety violations are detected. To bridge this gap, we propose a curriculum-driven framework that tightly integrates scenario-based Evolution Strategy with Statistical Model Checking-based verification in a closed-loop procedure. Starting from a candidate region, our approach co-optimizes policy performance while progressively enlarging its safe operating boundaries. Upon termination, it yields a neural controller together with a region over which safety and performance are statistically verified. Extensive evaluations on Cartpole and 3D Quadrotor benchmarks, showing 6.14x and
    
[^351]: 面向GPU加速量子电路模拟的张量网络收缩计划学习排序

    Learning to Rank Tensor Network Contraction Plans for GPU-Accelerated Quantum Circuit Simulation

    [https://arxiv.org/abs/2608.05819](https://arxiv.org/abs/2608.05819)

    该论文提出一种学习排序框架，通过从成对收缩序列中提取结构特征并训练梯度提升排序器，在执行前准确预测并选择GPU上最高效的张量网络收缩计划。

    

    经典模拟对于开发和验证量子算法仍然至关重要，但其成本会随着电路规模的增大而迅速增长。张量网络收缩可以通过利用电路结构来降低这一成本，不过其效率在很大程度上取决于所选择的收缩计划。在GPU上，理论复杂度相近的收缩计划在实际执行中可能表现差异巨大，因为执行性能还依赖于并行度、归约结构、内存流量和收缩几何形状。我们提出了一个学习排序框架，用于在执行之前选择高效的收缩计划。每个收缩计划由直接从其成对收缩序列中提取的结构特征来表示，并利用GPU实测数据、基于列表和成对目标函数训练梯度提升排序器。我们在多种电路族上评估所得模型，采用独立的分布内测试集和电路族偏移测试集，并将其与……（原文摘要在此处截断）

    arXiv:2608.05819v2 Announce Type: replace  Abstract: Classical simulation remains essential for developing and validating quantum algorithms, but its cost grows rapidly with circuit size. Tensor-network contraction can reduce this cost by exploiting circuit structure, although its efficiency depends strongly on the chosen contraction plan. On GPUs, plans with similar theoretical complexity may perform very differently because execution also depends on parallelism, reduction structure, memory traffic, and contraction geometry. We present a learning-to-rank framework for selecting efficient contraction plans before executing them. Each plan is represented by structural features derived directly from its sequence of pairwise contractions, and gradient-boosted rankers are trained from GPU measurements using listwise and pairwise objectives. We evaluate the resulting models on diverse circuit families, using separate in-distribution and circuit-family-shift test sets, and compare them with 
    
[^352]: 逃离过度挤压：面向消息传递网络的可寻址且支持感知的全局记忆

    Escaping Oversquashing: Addressable and Support-Aware Global Memory for Message Passing Networks

    [https://arxiv.org/abs/2608.02709](https://arxiv.org/abs/2608.02709)

    该论文提出一种兼具可寻址性与支持感知的全局记忆机制，通过乘性读写映射仅用对数级地址编码维度即可选择 M 个记忆行，并借助学习到的私有锚点保持读取有界，从而解决消息传递网络中多节点共享虚拟节点全局状态导致的瓶颈问题。

    

    虚拟节点是对抗过度挤压（oversquashing）的一种自然工具：它们用两跳的全局路径取代了长距离的消息传递路径。但当许多节点共享同一个全局状态时，这条捷径本身可能成为瓶颈。我们研究了这种全局记忆的两个性质。第一，可寻址性：在恒定边距的地址编码以及放大该边距的非线性条件下，乘性写入/读取映射仅需 O(log M) 维的地址编码即可提供 M 个可选择的记忆行。交叉注意力槽和受约束的 ELU+1 双线性记忆均满足这些条件。第二，支持感知：归一化交叉注意力缺乏一个可供潜在查询用作参照的自键。一个学习得到的私有锚点提供了这一参照，保持读取的有界性，并揭示匹配源质量的强度。我们在 Two-Radius 和 Tree-NeighborsMatch 的多个实例上展示了这些性质的优点，可寻址的真实……

    arXiv:2608.02709v2 Announce Type: replace-cross  Abstract: Virtual nodes are a natural tool against oversquashing: they replace long message-passing paths by a two-hop global route. But when many nodes share one global state, that shortcut can become a bottleneck itself. We study two properties of this global memory. First, addressability: under constant-margin address codes and a nonlinearity that amplifies this margin, multiplicative write/read maps provide $M$ selectable memory rows with only $O(\log M)$ address-code dimensions. Cross-attention slots and a constrained $ELU+1$ bilinear memory both satisfy these conditions. Second, support awareness: normalized cross-attention has no self-key for a latent query to use as a reference. A learned private anchor supplies this reference, keeps the read bounded, and exposes the strength of the matching source mass. We demonstrate the merits of such properties on several instances of Two-Radius and Tree-NeighborsMatch: both addressable reali
    
[^353]: 使用大型语言模型探究原子中心结构描述符的极限

    Using large language models to probe the limits of atom-centered structural descriptors

    [https://arxiv.org/abs/2607.26984](https://arxiv.org/abs/2607.26984)

    研究人员借助大型语言模型发现了即使采用多达七个近邻构建的原子中心结构描述符仍无法区分的三维原子结构，揭示了广泛使用的结构描述符层级体系存在的根本性局限。

    

    将原子结构映射为一组紧凑的几何描述符，是任何应用于原子尺度建模的机器学习方法中的关键步骤。一种强大且被广泛使用的方法可以理解为对配对距离、三角形等直方图的离散化，从而形成具有对称不变性的原子中心描述符的层级体系。遗憾的是，该层级体系中的较低层级（二、三、四近邻团簇）被证明是不完备的，存在与对称性无关的结构对却具有完全相同的描述符。然而，迄今报告的所有“简并”情况都可以通过考虑更大的近邻团簇来构建描述符而得到解决。我们报告了一些三维结构的例子，即使考虑多达七个近邻的团簇也无法区分它们，并且在描述符的实际离散化水平下，在任何阶数下均无法区分。这些结构是在大型语言模型的辅助下发现的。

    arXiv:2607.26984v2 Announce Type: replace-cross  Abstract: Mapping an atomic structure to a compact set of geometric descriptors is an essential step in any machine-learning application to atomic-scale modeling. A powerful and widely-used approach can be understood as a discretization of the histogram of pair distances, triangles, etc., that results in a hierarchy of symmetry-invariant atom-centered descriptors. Unfortunately, the lower rungs on this hierarchy (two, three, four-neighbor clusters) were found to be incomplete, with symmetry-unrelated pairs of structures having exactly the same descriptors. However, all the ``degeneracies'' reported so far are resolved by considering larger clusters of neighbors to build the descriptors. We report examples of 3D structures that are indistinguishable even if one considers clusters of up to seven neighbors, and to arbitrary order when considering a practical level of discretization of the descriptors, discovered with the assistance of large
    
[^354]: ORACLE：通过自适应验证器校准反馈实现智能体AI编排器路由

    ORACLE: Agentic AI Orchestrator Routing Via Adaptive Verifier Calibration Feedback

    [https://arxiv.org/abs/2607.22465](https://arxiv.org/abs/2607.22465)

    ORACLE提出了一种并发感知的在线智能体路由机制，通过将自适应路由与自适应验证器校准反馈相结合，解决了固定验证器难以泛化到异构智能体任务、以及验证器位于关键路径导致并发请求服务质量下降的问题，且无需训练即可即插即用。

    

    现代企业智能体部署由具有不同能力和成本的异构大语言模型（LLM）池组成。现有的模型路由策略在优化质量-成本权衡的同时，仅提供请求级别的静态决策。较新的解决方案将智能体路由视为任务级选择问题，采用基于串行验证器的路由器反馈循环。然而，这类方案中适用于同质工作负载的固定验证器，可能无法泛化到异构的智能体任务批次（例如：编程任务、通用对话等）。此外，由于验证器被置于反馈循环的关键路径上，在处理多个并发请求的路由时，服务质量可能会受到影响。为缓解这些问题，我们提出了ORACLE。它是一种并发感知的在线路由机制，将自适应路由与自适应验证相校准以生成反馈。ORACLE作为一个无需训练的即插即用“反馈循环”，可部署于任何模型选择器之上（原文摘要在此处截断）。

    arXiv:2607.22465v3 Announce Type: replace  Abstract: Modern enterprise agent deployments consist of a heterogeneous pool of large language models (LLMs) having diverse capabilities and cost. Existing model routing strategies optimize the quality-cost trade-off, while providing request-level static decisions. More recent solutions address agentic routing as a task-level selection with a serial verifier based router feedback loop. However, their fixed verifier suitable for homogeneous workloads may not generalize to heterogeneous batches of agentic tasks (example: coding, general conversational). Additionally, due to the verifier placement in the critical path of the loop, serving quality may be affected during multiple concurrent requests routing. To mitigate these issues, we present ORACLE. It is a concurrency-aware online routing mechanism that aligns adaptive routing with adaptive verification for feedback. ORACLE acts as a training-free drop-in 'feedback loop' on top of any model-se
    
[^355]: 基于深度学习的粘弹性赫兹接触中时间分辨粘附力预测

    Deep learning-based prediction of time-resolved adhesive forces in viscoelastic Hertzian contacts

    [https://arxiv.org/abs/2607.19060](https://arxiv.org/abs/2607.19060)

    本文提出一种标量条件化的有状态序列到序列深度学习模型，结合固定测量步长（FMS）表示方法，能够从位移历史快速预测粘弹性赫兹接触中的完整时间分辨粘附力演化，克服了传统数值模拟计算成本高、无法用于实时应用和设计优化的局限。

    

    快速预测粘附性软质粘弹性接触的响应是当前软体机器人技术以及抓取和操控任务中的一项挑战。确定完整的时间分辨力轨迹需要完整的数值模拟，其计算成本强烈依赖于参数，使其在实时应用或设计优化循环中并不实用。在这项工作中，我们通过训练一个标量条件化的、有状态的序列到序列深度学习模型来克服这一限制，该模型能够根据规定的位移历史预测完整的力演化，适用于短程和长程粘附两种情形。数据集涵盖四个数量级的加载和卸载速率，并包含不同的停留时间，Tabor参数范围为0.2至3.2。为了实现跨这些异构时间尺度的学习，我们引入了一种固定测量步长（FMS）表示方法，将可变的……

    arXiv:2607.19060v2 Announce Type: replace-cross  Abstract: Fast prediction of the response of adhesive soft viscoelastic contacts represents a current challenge in soft robotics and for gripping and manipulation tasks. Determining the complete time-resolved force trajectory requires full numerical simulations, whose computational cost is strongly parameter-dependent, making them impractical for real-time application or design-optimization loops. In this work, we overcome this limitation by training a scalar-conditioned, stateful, sequence-to-sequence deep learning model to predict the full force evolution from a prescribed displacement history for both short- and long-range adhesion regimes. The data set spans four orders of magnitude in loading and unloading rates and includes varied dwell times, with the Tabor parameter ranging from $0.2$ to $3.2$. To enable learning across these heterogeneous time scales, we introduce a fixed-measurement-step (FMS) representation that converts varia
    
[^356]: DriftWorld：通过漂移实现快速世界建模

    DriftWorld: Fast World Modeling through Drifting

    [https://arxiv.org/abs/2607.15065](https://arxiv.org/abs/2607.15065)

    DriftWorld基于漂移生成模型学习条件漂移，实现单次前向传播即可生成未来观测的动作条件世界模型，速度超过40 fps、比扩散基线快12倍以上且视觉质量相当或更优。

    

    预测性世界模型使机器人能够模拟其动作产生的视觉结果，但最先进的基于扩散的模型成本高昂，因为每次生成轨迹都需要多步迭代去噪。我们提出了DriftWorld，这是一种基于漂移生成模型的动作条件世界模型。DriftWorld在训练过程中学习条件漂移，使其在推理时能够通过单次前向传播为给定动作序列生成未来观测。在Bridge-V2、RT-1、Language Table、Push-T和Robomimic基准上，DriftWorld运行速度超过40 fps，比基于扩散的基线快12倍以上，同时达到或超越其视觉生成质量。这使得DriftWorld成为机器人仿真中的高效世界模型，并进一步支持推理时动作搜索和离线策略评估等下游应用。

    arXiv:2607.15065v3 Announce Type: replace-cross  Abstract: Predictive world models enable robots to simulate the visual outcomes of their actions, but state-of-the-art diffusion-based models remain costly because generating each rollout requires multi-step iterative denoising. We introduce DriftWorld, an action-conditioned world model based on drifting generative models. DriftWorld learns a conditional drift during training, enabling it to generate future observations for a given action sequence in a single forward pass during inference. Across Bridge-V2, RT-1, Language Table, Push-T, and Robomimic, DriftWorld runs at over 40 fps and is 12+ times faster than diffusion-based baselines, while matching or improving their visual generation quality. This makes DriftWorld an efficient world model for robot simulation and further enables downstream applications including inference-time action search and offline policy evaluation.
    
[^357]: DAGR：通过差异感知目标交叉注意力实现状态条件化的目标表示

    DAGR: State-Conditioned Goal Representations via Difference-Aware Goal Cross-Attention

    [https://arxiv.org/abs/2607.13731](https://arxiv.org/abs/2607.13731)

    提出DAGR方法，通过多尺度门控交叉注意力和差异感知的注意力规则，将目标条件强化学习中静态的目标嵌入精炼为状态条件化表示，使策略能直接感知目标中尚未完成的部分，并从理论上揭示了后归一化放置方式对门控结构保证条件的破坏。

    

    目标条件强化学习的关键在于目标如何被编码。对比式、度量式、时间距离式和信息论式的编码器在优化目标上各持己见，但它们在一件事上是一致的：这些编码器都不关注当前状态，因此目标嵌入无法标记出目标中哪一部分仍需要采取动作，策略必须通过同时反演两个编码器来恢复这一线索。我们提出DAGR，它通过多尺度门控交叉注意力，将任何后期融合编码器的静态目标嵌入精炼为状态条件化的嵌入。一个门控残差机制使精炼结果保持在基础嵌入附近，而差异感知的注意力规则则根据每个token上状态与目标之间的失配程度对注意力分数进行偏置。我们提出一个单一条件来刻画这种精炼结构所能保证的性质，即该模块在门完全关闭时是否返回其原始输入。我们证明了常见的后归一化（post-norm）放置方式违反了这一条件，并在冻结的检查点上测量了由此造成的后果，同时通过恢复……来弥补部分损失。

    arXiv:2607.13731v2 Announce Type: replace  Abstract: Goal-conditioned reinforcement learning hinges on how the goal is encoded. Contrastive, metric, temporal-distance and information-theoretic encoders disagree on the objective. They agree on one thing. None of them sees the current state, so the embedding cannot mark which part of the goal still needs action, and the policy must recover that cue by inverting both encoders. We propose DAGR, which refines the static embedding of any late-fusion encoder into a state-conditioned one through multi-scale gated cross-attention. A gated residual holds the refinement near the base, and a difference-aware attention rule biases the scores by a per-token state-goal mismatch. A single condition decides what such a refinement can guarantee, namely whether the block returns its input at closed gates. We prove that the usual post-norm placement violates it, measure the consequence on frozen checkpoints, and recover part of the resulting loss by resto
    
[^358]: 用于三维分子生成的自回归潜在扩散模型

    Autoregressive latent diffusion for 3D molecule generation

    [https://arxiv.org/abs/2607.09277](https://arxiv.org/abs/2607.09277)

    KRONOS是一个潜在自回归扩散框架，在统一自编码器的潜在空间中联合建模分子图拓扑与几何结构，无需预先指定分子大小即可生成3D分子，并通过FIM启发的混合训练策略平衡无条件生成与片段条件化生成，助力药物发现。

    

    三维（3D）分子生成领域一直由扩散模型主导，扩散模型虽然能实现出色的生成质量，但通常需要在生成之前单独指定（或预测）分子的大小。这对于以片段为基础的分子生成（药物发现的核心任务）来说是一个限制，因为在此类任务中，生成结构的大小本身就是设计问题的一部分。自回归模型能够在生成过程中确定分子大小，并天然支持部分结构条件化，但在平衡无条件生成与片段条件化生成方面仍然存在挑战。我们提出了KRONOS，这是一个潜在空间自回归扩散框架，它在统一自编码器的潜在空间中生成分子，联合建模分子图拓扑结构和几何结构，同时保留了自回归生成的灵活性。我们还进一步提出了一种受填空式范式启发的混合训练策略，使得……

    arXiv:2607.09277v2 Announce Type: replace  Abstract: Three-dimensional (3D) molecule generation has been dominated by diffusion models, which achieve strong generation quality but typically require molecular size to be specified (or predicted) separately before generation. This can be limiting for fragment-based molecule generation, central to drug discovery, where the size of the generated structure is itself part of the design problem. Autoregressive models determine size during generation and naturally support partial-structure conditioning, but balancing unconditional and fragment-conditioned generation remains challenging. We introduce KRONOS, a latent autoregressive diffusion framework that generates molecules in the latent space of a Unified AutoEncoder (UAE), jointly modeling molecular graph topology and geometry, while retaining the flexibility of autoregressive generation. We further introduce a mixed training strategy inspired by the Fill-in-the-Middle (FIM) paradigm, enabli
    
[^359]: 基于生物信息神经网络实现可靠的机制算子恢复：架构与优化设计原则

    Reliable mechanistic operator recovery with biologically-informed neural networks: principles for architecture and optimisation design

    [https://arxiv.org/abs/2607.07425](https://arxiv.org/abs/2607.07425)

    本文通过实证研究系统考察了网络架构设计、优化超参数和数据信息对生物信息神经网络（BINNs）机制算子恢复可靠性的影响，为BINNs的架构与优化设计提供了原则性指导。

    

    许多生物过程由复杂的动力学机制所支配，尽管实验数据量不断增加，这些机制仍未被完全理解。生物信息神经网络（BINNs）试图通过将微分方程嵌入神经网络训练中来解决这一挑战，使其能够直接从稀疏且含噪的观测数据中恢复本构算子。然而，算子恢复在多大程度上依赖于架构设计、优化策略以及数据中所包含的信息，目前尚不清楚。我们对这些因素如何影响基于BINNs的机制推断进行了实证研究，将其应用于一维对流-扩散-反应偏微分方程。在一系列问题上，我们研究了网络表达能力、学习率、损失权重和批次大小如何影响优化行为、重建精度和算子恢复。

    arXiv:2607.07425v2 Announce Type: replace-cross  Abstract: Many biological processes are governed by complex dynamical mechanisms that remain incompletely understood despite increasing volumes of experimental data. Biologically-informed neural networks (BINNs) seek to address this challenge by embedding differential equations into neural network training, enabling constitutive operators to be recovered directly from sparse and noisy observations. However, the extent to which operator recovery depends on architectural design, optimisation strategy and the information within the data is not yet well understood. We present an empirical study of how these factors influence mechanistic inference using BINNs applied to one-dimensional advection-diffusion-reaction partial differential equations. Across a suite of problems, we investigate how network expressivity, learning rate, loss weighting and batch size influence optimisation behaviour, reconstruction accuracy and operator recovery. We sh
    
[^360]: ELSA3D：面向统一3D理解与生成的弹性语义锚定

    ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation

    [https://arxiv.org/abs/2607.06565](https://arxiv.org/abs/2607.06565)

    ELSA3D提出弹性语义锚定机制，通过尺度感知八叉树分词器与稀疏的跨模态锚定token，在匹配的抽象尺度上显式对齐语言与几何推理，实现统一的3D理解与生成。

    

    统一3D基础模型旨在在单一骨干网络内生成3D资产并以语言对其进行推理，但其文本与3D之间的交互在很大程度上仍是隐式的。现有方法将文本和3D token拼接成一个扁平序列并依赖自注意力机制，把粗糙的结构线索与精细的几何细节压缩成一种无差别的表示。我们提出ELSA3D，一个通过弹性语义锚定来解决该问题的统一3D模型，它在相互匹配的抽象尺度上联合构建语言推理与几何推理。ELSA3D采用尺度感知的八叉树分词器来表示几何，并引入锚定token——一种稀疏的跨模态单元，能够选择语义线索、将其路由到最相关的3D尺度、检索特定尺度的几何证据，并将融合后的信号写回统一表示，从而保持交互的稀疏性与精确性。轻量级的逐块路由器使两者的计算…（摘要被截断）

    arXiv:2607.06565v2 Announce Type: replace-cross  Abstract: Unified 3D foundation models aspire to generate 3D assets and reason about them in language within a single backbone, but their text-3D interaction remains largely implicit. Existing methods concatenate text and 3D tokens into a flat sequence and rely on self-attention, collapsing coarse structural cues and fine geometric details into one undifferentiated representation. We introduce ELSA3D, a unified 3D model that addresses this with elastic semantic anchoring, structuring language and geometric reasoning jointly along matched abstraction scales. ELSA3D represents geometry with a scale-aware octree tokenizer and introduces Anchor Tokens, sparse cross-modal units that select semantic cues, route them to the most relevant 3D scale, retrieve scale-specific geometric evidence, and write the fused signal back into the unified representation, keeping interaction sparse yet precise. A lightweight per-block router makes both computati
    
[^361]: 句级上下文敏感性作为免训练的无依据内容检测器：与训练式验证器的对比评估

    Sentence-Level Context Sensitivity as a Training-Free Detector of Unsupported Content, Evaluated Against Trained Verifiers

    [https://arxiv.org/abs/2607.04223](https://arxiv.org/abs/2607.04223)

    该论文提出将句子在有/无上下文时的似然差异作为免训练的句子级无依据内容检测器，在多段落RAG答案中其检测能力可与经过训练的验证器相媲美，且无需额外训练、成本更低。

    

    检索增强生成（RAG）助手在临床和法律工作中对记录进行摘要，其中一句无依据的句子就可能误导读者。输出在有源文档与无源文档情形下似然之间的对比，作为整篇摘要和答案的忠实度评分方法已得到公认，但它尚未被作为多段落RAG答案中单个无依据句子的检测器加以衡量，也未与训练式验证器进行对比，或对其成本进行评估。我们将其实现为一种免训练检测器：在完整上下文、无上下文以及逐个移除每个文本块的条件下对固定答案重新打分，并返回移除后最能使句子似然下降的文本块，作为候选支持段落。我们在RAGTruth、TofuEval和RAGBench数据集上，使用六个评分器，并与五个验证器（直至大语言模型（LLM）裁判）在相同输入和源级别划分下对其进行评估。按句子粒度打分对无依据句子的排序优于答案……（原文摘要至此截断）

    arXiv:2607.04223v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) assistants summarize records in clinical and legal work, where one unsupported sentence can mislead a reader. The contrast between an output's likelihood with and without its source is an established faithfulness score for whole summaries and answers, but it has not been measured as a detector of the individual unsupported sentence in multi-passage RAG answers, against trained verifiers, or for its cost. We implement it as a training-free detector that re-scores a fixed answer under the full context, no context, and each chunk removed, and returns the chunk whose removal lowers a sentence's likelihood most as a candidate supporting passage. We evaluate it on RAGTruth, TofuEval, and RAGBench with six scorers and against five verifiers, up to a large language model (LLM) judge, on identical inputs under a source-level split. Scoring per sentence ranks unsupported sentences better than the answ
    
[^362]: 更密集 ≠ 更好：同策略自蒸馏在持续后训练中的局限

    Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training

    [https://arxiv.org/abs/2607.01763](https://arxiv.org/abs/2607.01763)

    本文通过自蒸馏策略优化（SDPO）重新审视同策略自蒸馏，发现其在持续后训练中比GRPO引发更严重的遗忘甚至崩溃，证明“更密集”的同策略监督信号并不等于更好。

    

    持续后训练使基础模型能够在获取新知识的同时保留既有能力。近期工作表明，同策略学习可以缓解遗忘，其中自蒸馏是一种尤其有吸引力的方法。我们通过自蒸馏策略优化重新审视了这一乐观论断。实验表明，当教师信号稳定且对齐良好时，SDPO能够加速领域内的特化，但难以泛化到分布之外。在持续后训练中，SDPO表现出更严重的遗忘，甚至可能崩溃；而作为更成熟、更广泛使用的同策略强化学习方法，GRPO的适应更为保守，能更好地保留先前能力。进一步的分析将这些失败与参数空间和响应空间中漂移的加剧，以及自我强化的师生回路对高频伪影的放大联系起来。因此，仅靠同策略数据……

    arXiv:2607.01763v2 Announce Type: replace-cross  Abstract: Continual post-training enables foundation models to acquire new knowledge while preserving existing capabilities. Recent work suggests that on-policy learning can mitigate forgetting, with self-distillation as a particularly attractive approach. We revisit this optimistic claim through self-distillation policy optimization (SDPO). Our experiments show that SDPO accelerates in-domain specialization when teacher signals are stable and well aligned, but struggles to generalize out of distribution. In continual post-training, SDPO exhibits greater forgetting and can even collapse, whereas GRPO, the more established on-policy reinforcement learning method, adapts more conservatively and better preserves prior capabilities. Further analyses link these failures to increased drift in parameter and response space, and to amplification of high-frequency artifacts through a self-reinforcing teacher-student loop. Thus, on-policy data alon
    
[^363]: 面向样本生成模型的决策感知训练

    Decision-Aware Training for Sample-Based Generative Models

    [https://arxiv.org/abs/2607.01171](https://arxiv.org/abs/2607.01171)

    提出决策感知训练方法，通过可微分优化层计算决策损失并将其与能量分数结合，使样本生成模型的训练能够直接惩罚下游决策成本，从而在高风险决策场景中生成更具实用价值的概率预测。

    

    样本生成模型越来越多地被用于高风险决策场景下的概率预测，然而它们的训练目标对决策者的成本结构并不敏感。这些模型通常使用严格适当的评分规则进行训练，例如能量分数，这类规则按照数据密度成比例地分配训练信号，而完全没有意识到预测误差在哪些地方对下游决策的代价最高。因此，我们提出了针对样本生成模型的决策感知训练方法，在能量分数目标的基础上增加一个可微分的决策损失，直接惩罚基于模型预测采取行动所产生的成本。这一组合损失在理论上有充分依据，因为决策损失本身就是一个适当的评分规则。我们通过一个可微分优化层来计算该决策损失，其梯度集中于输出空间中对成本敏感的区域，从而使该方法的效果……

    arXiv:2607.01171v2 Announce Type: replace  Abstract: Sample-based generative models are increasingly used for probabilistic forecasting in high-stakes decision settings, yet their training objectives are blind to the decision maker's cost structure. These models are commonly trained with strictly proper scoring rules, such as the energy score, which allocate their training signal in proportion to data density, with no awareness of where forecast errors are most costly for downstream decisions. We therefore propose decision-aware training for sample-based generative models, augmenting the energy score objective with a differentiable decision loss that directly penalises the cost incurred by acting on the model's forecast. This combined loss is theoretically grounded, as the decision loss is itself a proper scoring rule. We compute the decision loss via a differentiable optimisation layer. Its gradient concentrates in cost-sensitive regions of the output space, making the method's effect
    
[^364]: 开放量子系统的结构学习

    Learning the structure of open quantum systems

    [https://arxiv.org/abs/2606.30358](https://arxiv.org/abs/2606.30358)

    该论文提出了学习开放量子系统中常数局域 Lindbladian 系数的最优算法，在不知道系统结构的情况下，以 O(g d² log(n) / ε²) 的总演化时间实现准局域和幂律 Lindbladian 的学习，并在除对数因子外证明了其最优性。

    

    我们设计了一种算法，用于以 ε 精度学习 n 量子比特常数局域 Lindbladian（林德布拉德生成元）的系数，所需的总演化时间为 O(g d² log(n) / ε²)，其中 g 是单点能量，d 是相互作用图的（近似）度数。尽管 Lindbladian 带来了哈密顿量这一特例中不存在的新挑战，我们的算法仍达到了最先进哈密顿量学习算法所具备的一系列理想特性：(1) 它使用非自适应、无辅助比特的随机化 Pauli 测量电路，时间分辨率仅为 Θ(1/g)；(2) 它无需预先知道未知 Lindbladian 的结构即可工作；(3) 它依赖于一种平滑形式的度数定义，从而支持准局域和幂律 Lindbladian 的学习。此外，我们证明了一个下界，表明我们的算法在每个参数上（除去对数因子）都是最优的。我们的算法是一个简单的……

    arXiv:2606.30358v2 Announce Type: replace-cross  Abstract: We design an algorithm for learning the coefficients of an $n$-qubit constant-local Lindbladian to $\varepsilon$ error with $O(g d^2 \log(n) / \varepsilon^2)$ total evolution time, where $g$ is the single-site energy and $d$ is the (approximate) degree of the interaction graph. Though Lindbladians present new challenges not present in the special case of Hamiltonians, our algorithm achieves the suite of desiderata attained by state-of-the-art Hamiltonian learning algorithms: (1) it uses non-adaptive, ancilla-free randomized Pauli measurement circuits with a time resolution of only $\Theta(1/g)$; (2) it works without knowledge of the structure of the unknown Lindbladian; (3) it depends on a smooth form of degree, thereby supporting the learning of quasi-local and power-law Lindbladians. Moreover, we prove a lower bound showing that our algorithm is optimal in each parameter up to logarithmic factors.   Our algorithm is a simple 
    
[^365]: PerturbCellRL：通过后训练扰动生成器实现分布对齐与生物学锚定

    PerturbCellRL: Aligning Distributions and Grounding Biology via Post-Training Perturbation Generators

    [https://arxiv.org/abs/2606.27752](https://arxiv.org/abs/2606.27752)

    本文提出PerturbCellRL强化学习框架，通过基于基因表达能量见证的逐细胞奖励对单细胞扰动生成器进行后训练，恢复了流匹配训练无法捕捉的目标分布，从而更好地建模细胞异质性。

    

    单细胞扰动模型能够通过预测细胞在转录水平上对干预措施的响应，从而减少昂贵的湿实验室筛选工作。流匹配（flow-matching）的最新进展使得在群体水平上预测细胞响应成为可能。然而，即使在模型族内部，流匹配训练也可能无法恢复某些目标分布，这限制了其捕捉细胞异质性的能力。我们首先证明了后训练可以恢复这些分布，随后提出了PerturbCellRL——一个利用逐细胞奖励对单细胞扰动生成器进行后训练的强化学习框架。其核心组件是一个基因表达能量见证函数，可将群体层面的差异转化为逐细胞的反馈。我们进一步证明了该奖励的策略梯度指向更好的分布对齐。两个在真实细胞上校准的互补奖励，用于惩罚非典型的表达谱和不足的……（原文摘要在此处截断）

    arXiv:2606.27752v2 Announce Type: replace  Abstract: Single-cell perturbation models can reduce costly wet-lab screening by predicting how cells respond transcriptionally to interventions. Recent advances in flow-matching have enabled population-level prediction of cellular responses. However, flow-matching training can fail to recover certain target distributions even within the model family, limiting its ability to capture cellular heterogeneity. We first prove that post-training can recover these distributions, then introduce PerturbCellRL, a reinforcement learning framework that post-trains single-cell perturbation generators using per-cell rewards. The central component is a gene-expression energy witness that translates population-level discrepancies into per-cell feedback. We further prove that this reward's policy gradient points toward better distributional alignment. Two complementary rewards, calibrated on real cells, penalize atypical expression profiles and insufficient pa
    
[^366]: 超越全局分歧：贝叶斯推理中的局部质量视角

    Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference

    [https://arxiv.org/abs/2606.27090](https://arxiv.org/abs/2606.27090)

    本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。

    

    摘要：arXiv:2606.27090v1 公告类型：交叉 摘要：全局目标函数，如KL散度和ELBO，在贝叶斯推理中被广泛用于度量分布差异。本文研究这些目标函数未能直接捕捉的局部质量行为。我们引入并使用了两种数学工具：（1）质量指数，用于记录局部质量的多项式和对数衰减尺度；（2）正则化扩展KL（RE-KL），一种在存在奇异成分时可公式化的局部化散度。质量指数有助于刻画贝叶斯更新如何改变局部质量：（1）幂对数似然因子显式地改变它；（2）参数依赖的支持域或其平滑软化，可能通过参数值附近剩余的质量量来改变局部尺度。利用局部RE-KL，我们证明了在两种KL方向下比较局部小球质量的绝对、相对和方向性不等式。这些结果共同为局部质量行为提供了理论依据。

    arXiv:2606.27090v1 Announce Type: cross  Abstract: Global objectives, such as KL divergence and ELBO, are widely used in Bayesian inference for measuring distributional discrepancy. This paper studies their local-mass behaviour that is not directly captured by such objectives. We introduce and use two mathematical tools: (1) Mass Index for recording the polynomial and logarithmic decay scales of local mass, and (2) regularised extended KL (RE-KL), a set-localised divergence that can be formulated in the presence of singular components. Mass Indices help characterise how Bayesian updating changes local mass: (1) power-log likelihood factors shift it explicitly, and (2) parameter-dependent supports, or their smooth softenings, may change the local scale through the amount of mass that remains near the parameter value. Using local RE-KL, we prove absolute, relative, and directional inequalities for comparing local small-ball masses under the two KL directions. Together, these results prov
    
[^367]: 面向基于模型规划的从语言中进行潜在目标预测

    Latent Goal Prediction from Language for Model-Based Planning

    [https://arxiv.org/abs/2606.20627](https://arxiv.org/abs/2606.20627)

    LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。

    

    arXiv:2606.20627v2 公告类型：替换 摘要：联合嵌入预测架构（JEPA）使智能体能够通过想象候选动作的结果在潜在空间中进行规划，然而任务规范仍然是一个瓶颈。视觉目标能提供精确的局部梯度，但缺乏远距离的引导；而语言虽然灵活，却受限于嘈杂的跨模态对齐，或依赖于独立的大型生成模型。我们提出了LAGO（从语言中进行潜在目标预测），这是一个分层世界模型，其中单个预测器既预测动作条件下的动力学，又将语言指令接地为中间潜在子目标序列，并通过在共享潜在空间上的单一回归目标来训练这两种模式。在每个规划步骤中，LAGO根据语言指令预测一系列潜在子目标，并使用软最小对齐代价来优化动作序列，该代价奖励智能体接近子目标，而不强制执行僵化的路径。子目标会被重新预测…

    arXiv:2606.20627v2 Announce Type: replace  Abstract: Joint-Embedding Predictive Architectures (JEPAs) enable agents to plan in latent space by imagining the outcomes of candidate actions, yet task specification remains a bottleneck. Visual targets provide precise local gradients but poor distant guidance, while language is flexible yet limited by noisy cross-modal alignment or dependence on distinct large generative models. We introduce LAGO (Latent Goal Prediction from Language), a hierarchical world model in which a single predictor both forecasts action-conditioned dynamics and grounds language instructions as sequences of intermediate latent subgoals, training both modes with a single regression objective over a shared latent space. At each planning step, LAGO predicts a sequence of latent subgoals from a language instruction and optimizes an action sequence using a soft-minimum alignment cost that rewards subgoal proximity without enforcing a rigid path. Subgoals are repredicted a
    
[^368]: 流映射去噪器：穿越失真-感知平面求解逆问题

    Flow Map Denoisers: Traversing the Distortion-Perception Plane for Inverse Problems

    [https://arxiv.org/abs/2606.19802](https://arxiv.org/abs/2606.19802)

    本文证明流映射模型隐式定义了一个由前瞻参数t控制的单参数去噪器族，可在失真-感知前沿上连续移动工作点，为逆问题求解提供了在最小均方误差与感知质量之间灵活权衡的新机制。

    

    图像修复面临一个根本性的权衡：最小化误差的方法会产生模糊的重建结果，而最大化感知质量的方法则产生清晰但保真度较低的图像。现有方法要么固定在失真-感知（DP）前沿上的单一工作点，要么需要配对数据监督、辅助模型或对采样器进行超参数调整才能访问不同的工作点。我们证明，流映射模型——流匹配的一种近期扩展，用于少步采样并学习平均场——隐式地定义了一个单参数去噪器族，该族可连续地跨越整个DP前沿。前瞻参数t充当在最小均方误差（MMSE）机制与感知机制之间的控制旋钮。对于高斯目标，我们证明改变t可以精确恢复最优的DP前沿；对于自然图像，我们在实验中观察到类似的行为。在即插即用求解器中，同一机制可扩展到一般逆问题。

    arXiv:2606.19802v2 Announce Type: replace  Abstract: Image restoration faces a fundamental tradeoff: methods that minimize error produce blurry reconstructions, while those that maximize perceptual quality yield sharp but less faithful images. Existing approaches either commit to a single operating point on this distortion perception (DP) frontier or require paired-data supervision, auxiliary models, or hyperparameter tuning of the sampler to access different points. We show that flow map models, a recent extension of flow matching for few-step sampling that learns an average field, implicitly define a one-parameter family of denoisers that continuously spans the DP frontier. The lookahead parameter t acts as a control knob between the MMSE and perceptual regimes. For Gaussian targets, we prove that varying t exactly recovers the optimal DP frontier; for natural images, we observe similar behavior empirically. Within a Plug-and-Play solver, the same mechanism extends to general inverse
    
[^369]: GB-LSR：面向任意尺度超分辨率的可学习全局带宽局部谱解码

    GB-LSR: Local Spectral Decoding with a Learned Global Bandwidth for Arbitrary-Scale Super-Resolution

    [https://arxiv.org/abs/2606.19617](https://arxiv.org/abs/2606.19617)

    提出GB-LSR，一种以单一可学习全局带宽控制截断傅里叶基规模的固定网格局部谱表示，实现了高效的连续图像解码，其任意尺度超分辨率扩展在统一评测协议下兼顾速度与质量。

    

    我们提出GB-LSR（全局带宽局部谱表示），一种用于连续图像解码的固定网格局部谱表示。图像域被划分为互不重叠的方形区块，每个区块携带一个截断傅里叶基的系数，这些系数由共享卷积编码器特征通过单次线性投影预测得到，并且一个可训练的标量带宽在所有区块和所有图像之间共享。与早期的局部谱解码器相同，在连续坐标处的解码是一次固定规模的基函数收缩运算，其计算代价由谱截断点决定；而GB-LSR学习该基的带宽，而非将其固定。我们评估了其任意尺度超分辨率扩展版本GB-LSR-Scalar-ASR，在相同的RDN编码器上与作者发布的LIIF、LTE和SRNO检查点进行对比，所有方法均在同一协议下评分，并在每个尺度下于单次会话中计时，每个方法各使用一块GPU。其运行速度比……快1.25倍（摘要至此截断）

    arXiv:2606.19617v2 Announce Type: replace-cross  Abstract: We present GB-LSR (Global-Bandwidth Local Spectral Representation), a fixed-grid local spectral representation for continuous image decoding. The image domain is partitioned into non-overlapping square patches. Each patch carries coefficients for a truncated Fourier basis, predicted by a single linear projection from shared convolutional-encoder features, and one trainable scalar bandwidth is shared across every patch and every image. As in earlier local spectral decoders, decoding at a continuous coordinate is a fixed-size basis contraction whose cost is set by the spectral cutoff; GB-LSR learns the bandwidth of that basis instead of fixing it. We evaluate an arbitrary-scale super-resolution extension, GB-LSR-Scalar-ASR, against the authors' released LIIF, LTE, and SRNO checkpoints on the same RDN encoder, with every method scored under one protocol and timed in one session per scale, each on one GPU. It runs 1.25x faster than
    
[^370]: 入乡随俗：从异构智能体中学习通用行为

    Do as the Romans Do: Learning Universal Behaviors from Heterogeneous Agents

    [https://arxiv.org/abs/2606.18537](https://arxiv.org/abs/2606.18537)

    提出GRID方法，通过信息瓶颈将异构智能体的奖励函数解耦为通用奖励与特定奖励，从追求不同目标的示范者群体中提取普遍有用的行为，为通用智能体预训练开辟了新范式。

    

    人类常常通过观察他人来习得新技能，因为观察到的行为隐含地揭示了如何在环境中合理行动。然而，来自异构群体的观察会引入相互冲突的行为信号，使得难以判断哪些行为值得模仿。我们通过通用奖励推断与解耦（GRID）来解决这一挑战，这是一种社会学习方法，能够从追求不同目标的异构示范者群体中提取普遍有用的行为。GRID通过信息瓶颈将每个智能体的奖励函数分解为通用奖励（捕获所有智能体共享的行为）和特定奖励（捕获个体的偏好与目标）。仅在通用奖励上进行训练提供了一种全新的通用智能体预训练范式，由此得到的通用智能体能够内化普遍适用的环境能力，例如安全性。

    arXiv:2606.18537v2 Announce Type: replace  Abstract: Humans often acquire new skills by observing others, since observed behaviors implicitly reveal how to act reasonably in an environment. However, observations drawn from a heterogeneous population introduce conflicting behavioral signals, making it difficult to determine which behaviors are worth imitating. We address this challenge with General Reward Inference and Disentanglement (GRID), a social learning method that extracts universally useful behaviors from a heterogeneous population of demonstrators pursuing different goals. GRID decomposes per-agent reward functions into a general reward, capturing behaviors shared across all agents, and specific rewards, capturing individual preferences and objectives, through an information bottleneck. Training exclusively on the general reward provides a new paradigm of generalist pretraining. It yields a generalist agent that internalizes universal environmental competencies, such as safety
    
[^371]: 基于流的通用机器人策略的不确定性量化

    Uncertainty Quantification for Flow-Based Generalist Robot Policies

    [https://arxiv.org/abs/2606.18043](https://arxiv.org/abs/2606.18043)

    本文提出一种利用小规模集成中的速度场分歧（VFD）来高效量化流匹配通用机器人策略认知不确定性的方法，可成功用于部署时的故障检测和主动微调。

    

    通用机器人策略，例如视觉-语言-动作模型（VLA）和世界-动作模型（WAM），将强大的预训练骨干网络与通过流匹配在大规模机器人数据集上训练的富有表现力的生成式动作头相结合。尽管这些策略在机器人操作任务中展现出强大的实证性能，但它们缺乏量化预测置信度的机制，也无法检测其动作何时可能不可靠。这对于在非平稳环境中的现实部署构成了关键限制，因为模型不可避免地会遇到超出其预训练分布的场景，并可能在毫无预警的情况下失败。为解决这一问题，我们通过利用小规模集成中的速度场分歧（VFD），推导出一种量化流匹配模型认知不确定性的高效方法。我们成功地将这种不确定性估计用于部署期间的故障检测以及基于流的策略的主动微调。

    arXiv:2606.18043v2 Announce Type: replace-cross  Abstract: Generalist robot policies, such as vision-language-action models (VLAs) and world-action models (WAMs), combine powerful pretrained backbones with expressive generative action heads trained via flow matching on large-scale robotic datasets. Despite their strong empirical performance in robotic manipulation, these policies lack mechanisms to quantify confidence in their predictions and to detect when their actions may be unreliable. This presents a critical limitation for real-world deployment in non-stationary environments, where models inevitably encounter scenarios outside their pretraining distribution and may fail without warning. To address this, we derive an efficient method to quantify epistemic uncertainty in flow-matching models by leveraging velocity-field disagreement (VFD) across a small ensemble. We successfully use this uncertainty estimate for detecting failures during deployment and active fine-tuning of flow-ba
    
[^372]: 通过智能体轨迹剖析模型行为

    Dissecting model behavior through agent trajectories

    [https://arxiv.org/abs/2606.17454](https://arxiv.org/abs/2606.17454)

    该论文提出“意图-执行”差距的概念，指出智能体性能本质上是系统问题而非单纯的建模问题，并开发了可跨多个模型家族（Claude、Gemini、GPT、Grok、Qwen）泛化的简单可定制框架SSA，以弥合模型能力与框架执行之间的鸿沟。

    

    AI智能体的性能不仅仅是一个建模问题，从根本上讲是一个系统问题。模型的高级能力是通过智能体框架（harness）来实现的。因此，模型假设与框架行为之间的差距很容易阻碍模型的全部能力转化为智能体的实际性能。我们将这一问题形式化为“意图-执行”差距：即模型意图与框架实际执行内容之间（以及反向）的不匹配。我们认为，最小化这种意图-执行差距与框架设计中的其他方面（如工具和执行循环）同等重要。为了说明这种框架-模型对齐的影响，我们开发了一个简单且可定制的框架，称为“Simple Strands Agent”（SSA）。SSA旨在找出可在不同模型家族（如Claude、Gemini、GPT、Grok、Qwen）之间泛化的大部分常见模式，以及少数模型特定的偏好。我们提出了两个……（原文在此截断）

    arXiv:2606.17454v3 Announce Type: replace  Abstract: AI agent performance is not just a modeling problem, it is fundamentally a systems problem. The advanced capabilities of models are realized through agent harnesses. Therefore, a gap between model assumptions and harness behavior can easily prevent the model's full capabilities from translating into agent performance. We formalize this as the `intent-execution' gap: the mismatch between what the model intends and what the harness executes, and vice versa. We argue that minimizing this intent-execution gap is as important as other aspects of harness design such as tools and execution loops. To illustrate the impact of this harness-model alignment, we develop a simple and customizable harness called `Simple Strands Agent' (SSA). SSA aims to find the bulk of common patterns which generalize across different model families (such as Claude, Gemini, GPT, Grok, Qwen), as well as a small number of model-specific preferences. We make two cont
    
[^373]: ROVE：通过强化学习解锁人形机器人操作中的人类干预

    ROVE: Unlocking Human Interventions for Humanoid Manipulation via Reinforcement Learning

    [https://arxiv.org/abs/2606.17011](https://arxiv.org/abs/2606.17011)

    ROVE提出了一种强化学习框架，通过人在回路的数据收集流程和乐观价值估计方法，从不完美的人类干预轨迹中筛选高价值行为，实现人形机器人VLA模型的有效后续训练。

    

    人类干预为视觉-语言-动作（VLA）模型的后续训练提供关键的纠正信号。然而，由于复杂的全身运动学和灵巧手控制，实现无缝的人形机器人干预是一项艰巨的系统挑战。因此，收集到的干预轨迹往往是次优的，而依赖人类干预作为专家监督的方法可能会吸收犹豫、低效甚至错误的行为。为了同时应对系统和算法方面的挑战，我们提出了ROVE，一个用于人形机器人VLA后续训练的强化学习框架，能够处理不完美的人类干预。首先，ROVE引入了一个人在回路中的流程，能够为人形机器人操作收集部署和干预数据。其次，它利用乐观价值估计（OVE）来从混合质量的轨迹中优先筛选高价值行为。为了进一步增强价值估计的鲁棒性，我们（原文在此处截断）

    arXiv:2606.17011v2 Announce Type: replace-cross  Abstract: Human interventions provide crucial corrective signals for post-training Vision-Language-Action (VLA) models. However, enabling seamless humanoid interventions is a formidable systems challenge due to complex whole-body kinematics and dexterous-hand control. Consequently, the collected intervention trajectories are often suboptimal, and methods that rely on human interventions as expert supervision can absorb hesitant, inefficient, or even erroneous behaviors. To address both the system and algorithmic challenges, we propose ROVE, a reinforcement learning framework for humanoid VLA post-training with imperfect human interventions. First, ROVE introduces a human-in-the-loop pipeline capable of collecting deployment and intervention data for humanoid manipulation. Second, it utilizes Optimistic Value Estimation (OVE) to prioritize high-value behaviors from mixed-quality trajectories. To further robustify value estimation, we inco
    
[^374]: 扩散流匹配：维度改进的KL界与Wasserstein保证

    Diffusion Flow Matching: Dimension-Improved KL Bounds and Wasserstein Guarantees

    [https://arxiv.org/abs/2606.16610](https://arxiv.org/abs/2606.16610)

    本文为基于布朗运动的扩散流匹配提供了在KL散度和2-Wasserstein距离下具有更优维度依赖性的离散化误差收敛保证，在温和条件下达到了最先进的收敛标度。

    

    扩散流匹配（DFM）近来已成为一种用途广泛的生成建模框架，但其理论收敛性质仍未被完全理解。在本工作中，我们为基于布朗运动的DFM提供了精细且新颖的收敛保证，重点关注离散化误差。我们的分析在Kullback-Leibler（KL）散度和2-Wasserstein距离下进行。在有限矩条件和温和的得分（score）可积性假设下，我们推导出了相比先前工作具有更优维度依赖性的KL收敛界，据我们所知，在最少的条件下达到了最先进的收敛标度。我们进一步将分析扩展到2-Wasserstein距离：在额外的一阶得分可积性假设和弱对数凹性条件下，我们获得了与KL情形维度依赖性一致的收敛保证。

    arXiv:2606.16610v2 Announce Type: replace-cross  Abstract: Diffusion Flow Matching (DFM) has recently emerged as a versatile framework for generative modeling, yet its theoretical convergence properties remain only partially understood. In this work, we provide refined and novel convergence guarantees for Brownian motion based DFMs, focusing on the discretization error. Our analysis is conducted under the Kullback-Leibler (KL) divergence and the 2-Wasserstein distance. Under finite-moment conditions and a mild score integrability assumption, we derive KL convergence bounds with improved dimensional dependence compared to prior work, achieving, up to our knowledge, state-of-the-art scaling under minimal conditions. We further extend the analysis to the 2-Wasserstein distance: under an additional first-order score integrability assumption and a weak log-concavity condition, we obtain convergence guarantees with dimensional dependence consistent with the KL case.
    
[^375]: 面向基于种群优化的算子微积分：模块化收敛性与有限种群保证

    Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees

    [https://arxiv.org/abs/2606.14289](https://arxiv.org/abs/2606.14289)

    本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。

    

    基于种群的优化器将变异、选择和重组等更新规则组合在一起。当其中某条规则发生变化时，通常不清楚哪些收敛保证仍然成立，以及应如何评估新的组合。我们发展了一种算子微积分：算子即种群更新规则，而该微积分规定了如何将各自经过独立验证的效应进行组合。在明确的正则性和小步长条件下，由更新引起的主要变化可以相加，从而为收敛分析提供可复用的构建模块。该框架区分了找到并保留好的解、降低种群平均目标值以及使候选解集中于最优解附近这三类目标，并指出了获得有限评估预算保证所需的额外逼近条件。应用包括分布自适应、重组式演化和共识动力学，并验证了非凸情形。在……上进行的受控实验……

    arXiv:2606.14289v2 Announce Type: replace-cross  Abstract: Population-based optimizers combine update rules such as mutation, selection, and recombination. When one rule changes, it is often unclear which convergence guarantees survive or how the new combination should be assessed. We develop an operator calculus: an operator is a population-update rule, and the calculus specifies how separately checked effects can be combined. Under explicit regularity and small-step conditions, the leading changes caused by the updates add, yielding reusable building blocks for convergence analysis. The framework distinguishes finding and retaining a good solution, reducing the population's mean objective, and concentrating candidates near an optimizer, and identifies the extra approximation conditions needed for finite evaluation-budget guarantees. Applications include distribution adaptation, recombinative evolution, and consensus dynamics, with verified nonconvex cases. Controlled experiments on a
    
[^376]: 物理系统概率仿真的可靠性

    Reliability of Probabilistic Emulation of Physical Systems

    [https://arxiv.org/abs/2606.12997](https://arxiv.org/abs/2606.12997)

    本研究开发了一个评估框架，在匹配的模型规模和计算预算下系统比较了生成式模型与CRPS训练的确定性模型集合在物理系统概率预报中的表现，发现CRPS训练的模型集合在预测区间的经验覆盖率上通常具有更可靠的不确定性。

    

    生成物理系统概率预报的两种主流方法已经出现：一种是生成式模型（如扩散模型或流匹配），另一种是注入随机性的确定性模型集合，后者使用连续排序概率评分（CRPS）损失进行训练。虽然这两种方法都表现出强大的预测精度，但其不确定性的可靠性尚未得到系统评估。我们通过开发一个评估框架来填补这一空白，该框架在匹配的模型规模和计算预算下，在多种二维时空物理系统上对这两种方法进行评估。我们通过检查预测区间的经验覆盖率来评估概率仿真的可靠性，同时还考虑了精度和计算效率指标。经过CRPS训练的模型集合通常在单步预测和自回归滚动预测中都能获得更可靠的不确定性，展现出更好的覆盖率。

    arXiv:2606.12997v2 Announce Type: replace  Abstract: Two dominant approaches have emerged for generating probabilistic forecasts of physical systems: generative models, such as diffusion or flow matching; and ensembles of deterministic models with stochasticity injected, trained using the continuous ranked probability score (CRPS) loss. While both approaches have demonstrated strong predictive accuracy, the reliability of their uncertainties has not been systematically assessed. We address this gap by developing a framework to evaluate both approaches across diverse 2D spatiotemporal physical systems, under matched model size and computational budget. We assess the reliability of probabilistic emulation by inspecting the empirical coverage of predictive intervals, while also considering accuracy and computational efficiency metrics. CRPS-trained ensembles typically achieve more reliable uncertainties on both single-step prediction and autoregressive rollouts, demonstrating better cover
    
[^377]: 面向长多模态文档证据组装的最小成本流路由

    Min-Cost Flow Routing for Evidence Assembly in Long Multimodal Documents

    [https://arxiv.org/abs/2606.07235](https://arxiv.org/abs/2606.07235)

    该论文提出FlowReader，将长多模态文档问答中的证据选择建模为带容量限制的最小成本流问题，通过谱分解按比例分配证据预算并生成短证据链，无需语言模型规划即可实现全面覆盖，在VisDoMBench上取得最高宏观准确率68.9。

    

    回答关于长多模态文档的问题，需要在文本、表格、图表和幻灯片中的相关内容方面之间分配固定的证据预算，同时避免近乎重复的内容。我们提出了FlowReader，它将证据选择表述为多模态内容图上带容量限制的最小成本流问题。谱分解识别与查询相关内容的潜在方面，并根据其谱能量按比例在这些方面之间分配预算。这些容量限制在路由过程中强制实现方面覆盖，无需调用语言模型进行规划。查询条件化的成本优先选择相关且相互一致的证据链。对最优流进行分解可产生较短的证据链，由视觉-语言模型并行读取，再由推理模块进行协调。在VisDoMBench上使用Qwen3-VL-32B时，FlowReader取得了最高的宏观准确率（68.9），超越了最强的（摘要在此处截断）。

    arXiv:2606.07235v3 Announce Type: replace-cross  Abstract: Answering questions about long multimodal documents requires distributing a fixed evidence budget across relevant facets in text, tables, figures, and slides while avoiding near-duplicates. We present \flowreader, which formulates evidence selection as a single minimum-cost flow problem with capacity limits over a multimodal content graph. Spectral decomposition identifies latent aspects of query-relevant content and allocates the budget among them in proportion to their spectral energy. These capacity limits enforce aspect coverage during routing without requiring a language-model planning call. Query-conditioned costs prioritize chains of relevant, mutually consistent evidence. Decomposing the optimal flow produces short evidence chains, which a vision-language model reads in parallel and a reasoner reconciles. On VisDoMBench with Qwen3-VL-32B, \flowreader\ achieves the highest macro accuracy ($68.9$), surpassing the stronges
    
[^378]: 纹理驱动视觉学习中的低频捷径

    Low-Frequency Shortcuts in Texture-Driven Visual Learning

    [https://arxiv.org/abs/2606.03493](https://arxiv.org/abs/2606.03493)

    纹理驱动的视觉学习存在基于低频成分的捷径学习问题，剪除低频成分可使同分布准确率提升最多10%、分布外准确率提升最多40%。

    

    神经网络存在捷径学习问题，即学到的特征在训练集上泛化良好，但在同分布（ID）或分布外（OOD）测试集上却无法良好泛化。现有研究均基于少数形状驱动的标准基准，然而许多应用领域是纹理驱动的。在本工作中，我们对纹理驱动领域进行了捷径学习分析，并与标准基准进行了比较。我们表明，纹理驱动领域受到低频捷径的影响：尽管高频成分（HFC）具有更强的预测能力，但网络大部分决策是基于少数频谱行为偏斜的低频成分（LFC）做出的。从训练集和测试集中剪除低频成分（LFC）可以缓解这种捷径问题，并提供更平衡的频谱行为，在算法和真实世界领域中，将ID准确率提升多达10%，OOD准确率提升多达40%。

    arXiv:2606.03493v2 Announce Type: replace-cross  Abstract: Neural networks suffer from shortcut learning, where learned features generalize well to the training set but not to in-distribution (ID) or out-of-distribution (OOD) test sets. Existing studies are all based on a few standard benchmarks, which are shape-driven. Numerous application domains, however, are texture-driven. In this work, we present shortcut learning analysis for texture-driven domains and compare it with that of a standard benchmark. We show that texture-driven domains suffer from low-frequency shortcuts. They make the majority of their decisions based on a few low-frequency components (LFCs) with a skewed spectral behavior, despite that higher-frequency components (HFCs) have higher predictive power. Pruning LFCs from training and test sets mitigates the shortcut and provides a more balanced spectral behavior, improving the ID accuracy by up to 10% and OOD accuracy by up to 40% under algorithmic and real-world dom
    
[^379]: 全域学习：具有逐点约束的人工智能

    Everywhere Learning: Artificial Intelligence with Pointwise Constraints

    [https://arxiv.org/abs/2606.01557](https://arxiv.org/abs/2606.01557)

    本文提出“全域学习”新范式，要求AI系统以概率一满足数据分布上每一点的损失约束而非仅最小化平均损失，并通过近似对偶理论证明其经验解与统计解之间的泛化接近性，且泛化可通过稀疏L1惩罚加以控制。

    

    全域学习是一种新的人工智能（AI）范式，其目标是训练AI系统在数据分布上以概率一满足损失约束，这与训练AI系统最小化平均损失的标准范式形成鲜明对比。我们开发了一种近似对偶理论，用以支持一项泛化分析，该分析确立了经验性全域学习问题与统计性全域学习问题解之间的接近性。我们的结果表明，对偶变量会将数据分布的权重重新分配给那些损失约束更难满足的数据点，并且泛化性能由数据分布的质量集中程度与约束更难满足之点上质量集中程度之间的不匹配程度所控制。我们进一步证明，可以通过对约束松弛施加稀疏的L1惩罚来控制泛化。我们展示了全域学习的优势……（摘要原文在此处截断）

    arXiv:2606.01557v2 Announce Type: replace  Abstract: Everywhere learning is a new paradigm whereby Artificial Intelligence (AI) systems are trained to satisfy loss constraints with probability one over the data distribution. This is in contrast to the standard paradigm of training AI systems to minimize average losses. We develop an approximate duality theory to substantiate a generalization analysis that establishes the proximity between solutions of empirical and statistical everywhere learning problems. Our results show that dual variables reweigh the data distribution towards points in which loss constraints are more difficult to satisfy and that generalization is controlled by the mismatch between the concentration of mass of the data distribution and the concentration of mass on points where constraints are more difficult to satisfy. We further show that we can control generalization with a sparse L1 penalty on constraint relaxations. We illustrate the merits of everywhere learni
    
[^380]: 迭代纳什偏好优化的高效探索

    Efficient Exploration for Iterative Nash Preference Optimization

    [https://arxiv.org/abs/2606.01382](https://arxiv.org/abs/2606.01382)

    论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。

    

    偏好对齐是提升大语言模型（LLM）性能的核心，但基于奖励的建模方式在人类偏好具有非传递性时会受到限制。从人类反馈中进行纳什学习（NLHF）通过将对齐建模为偏好博弈并求解纳什均衡来解决这一局限。然而，可扩展NLHF的学习理论基础仍然有限：现有的遗憾保证依赖于显式的偏好模型估计和极小极大预言机，而更简单的迭代方法则缺乏此类保证。我们研究在线迭代NLHF，并发现探索是其中的关键障碍。首先，我们证明标准的迭代NLHF可能会对KL正则化参数的倒数产生指数级依赖，这表明通过策略更新实现的隐式探索可能是不充分的。随后，我们提出探索式纳什偏好优化（ENPO），该方法将SFT型正则化与对抗性……

    arXiv:2606.01382v2 Announce Type: replace-cross  Abstract: Preference alignment is central to improving large language models (LLMs), but reward-based formulations can be restrictive when human preferences are non-transitive. Nash learning from human feedback (NLHF) addresses this limitation by modeling alignment as a preference game and seeking a Nash equilibrium. However, the learning-theoretic foundations of scalable NLHF remain limited: existing regret guarantees rely on explicit preference-model estimation and minimax oracles, whereas simpler iterative methods lack such guarantees. We study online iterative NLHF and identify exploration as a key obstacle. First, we show that standard iterative NLHF can incur an exponential dependence on the inverse KL-regularization parameter, demonstrating that implicit exploration through policy updates can be insufficient. We then propose Exploratory Nash Preference Optimization (ENPO), which combines a SFT-type regularization with adversarial 
    
[^381]: 论大语言模型适应性的局限：模型内化先验对标注任务性能的影响

    On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance

    [https://arxiv.org/abs/2606.00467](https://arxiv.org/abs/2606.00467)

    提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。

    

    大语言模型（LLM）越来越多地被用于零样本标注和“LLM作为评判者”任务，但其可靠性取决于模型内化的先验与用户所提供指令之间的交互方式。我们从三个维度研究了这种交互：(1) LLM对数据和任务定义的熟悉程度与其性能之间的关系；(2) 提示中的额外信息能否纠正零样本错误（即“决策粘性”）；(3) 模型对不一致任务定义的易感性。我们提出了“定义特定熟悉度”（DSF）这一概念，用于衡量模型所引出的概念与目标定义之间的对齐程度。在九个大语言模型和六个毒性数据集（五个主要数据集加一个额外的鲁棒性数据集）上的实验表明，在控制数据集身份后，DSF能够预测标注性能（偏相关系数 r=+0.41）。这种关联在所有测试的提示条件下均保持为正。相比之下……（原文摘要在此处截断）

    arXiv:2606.00467v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly used for zero-shot annotation and LLM-as-a-judge tasks, yet their reliability hinges on how model-internalized priors interact with user-provided instructions. We investigate three dimensions of this interaction: (1) how an LLM's familiarity with data and task definitions relates to performance, (2) whether additional information in prompts can correct zero-shot errors ("decision stickiness"), and (3) model susceptibility to misaligned task definitions. We introduce Definition-Specific Familiarity (DSF), which measures alignment between a model's elicited concept and the target definition. Across nine LLMs and six toxicity datasets (five primary datasets plus an additional robustness dataset), DSF predicts annotation performance after controlling for dataset identity (partial $r=+0.41$). This association remains positive across all prompting conditions tested. In contrast, three com
    
[^382]: 语言模型预训练中“顿悟”现象的类比研究：追踪延迟的语法泛化

    A Pre-Training Analogue of Grokking in Language Models: Tracing Delayed Grammatical Generalization

    [https://arxiv.org/abs/2606.00230](https://arxiv.org/abs/2606.00230)

    提出了一种基于暴露度的评估框架，利用BLiMP最小对立对及其关键短语来模拟预训练中的训练/验证划分，首次在LLM预训练过程中观察到跨越五种语法现象的类似“顿悟”的延迟语法泛化。

    

    “顿悟”是指神经网络在拟合训练数据很久之后才出现泛化的现象，以往主要在有监督设置下经过多个训练轮次进行研究。而大语言模型（LLM）的预训练则是在无标注语料上进行下一词元预测，数据重复有限，且没有显式的训练/验证集划分。为解决这一问题，我们提出了一个基于暴露度的框架，使我们能够在LLM预训练过程中研究类似“顿悟”的动态。我们以BLiMP最小对立对作为评估基础，其提供了受控的语法对比。对于每一个BLiMP最小对立对，我们识别出一个关键短语，即能够捕捉语法对比及该现象相关上下文的最小连续片段。关键短语出现在预训练窗口中的样本被划分到代理训练集，其余样本则被划分到代理验证集。在五种语法现象上，我们观察到了延迟泛化现象。

    arXiv:2606.00230v2 Announce Type: replace  Abstract: Grokking, the phenomenon in which neural networks generalize long after fitting their training data, has been studied in supervised settings on many epochs. LLM pre-training instead involves next-token prediction over an unlabeled corpus, with limited data repetition and no explicit train/validation split. To address this, we propose an exposure-based framework that enables the study of grokking-like dynamics during LLM pre-training. We ground our evaluation in BLiMP minimal pairs, which provide controlled grammatical contrasts. For every BLiMP minimal pair, we identify a critical phrase, the smallest continuous span that captures the grammatical contrast and the phenomenon-relevant context. Examples whose critical phrase appears in the pre-training window are assigned to the proxy-train split; the remaining examples are assigned to the proxy-validation split. Across five grammatical phenomena, we observe delayed generalization. Anal
    
[^383]: 固定通用Transformer（固定万能变换器）

    Fixed Universal Transformers

    [https://arxiv.org/abs/2605.31423](https://arxiv.org/abs/2605.31423)

    本文提出固定参数的“通用Transformer”，证明其可通过输入嵌入模拟任意同类别Transformer，且随机初始化的固定Transformer几乎必然具有通用性，表明Transformer的表达能力主要来自输入表示而非学习到的权重。

    

    我们提出了“通用Transformer”：这是一类固定的Transformer，能够通过合适的输入嵌入来模拟给定类别中的任意Transformer。类比于通用图灵机，输入嵌入编码了目标模型的描述，而所有内部参数保持固定。我们给出了显式的稀疏构造，在嵌入维度足够大时实现通用性，并进一步证明这种通用性是普遍存在的：随机初始化的Transformer几乎必然是通用的，这与Zhong和Andreas（2024）最近的实证结果相一致。我们在括号匹配和多跳推理这两个算法任务上对理论进行了实证验证。我们的结果表明，Transformer的绝大部分表达能力可能蕴藏于其输入表示之中，而非其学习到的权重。

    arXiv:2605.31423v2 Announce Type: replace  Abstract: We introduce \emph{universal transformers}: fixed transformers that can simulate any transformer in a given class via a suitable input embedding. Analogous to a universal Turing machine, the input embedding encodes a description of the target model while all internal parameters remain fixed. We provide explicit sparse constructions achieving universality when the embedding dimension is sufficiently large, and further show that universality is generic: randomly initialized transformers are universal almost surely, which aligns with recent empirical results of Zhong and Andreas (2024). We empirically validate our theory on the algorithmic tasks of parenthesis balancing and multi-hop reasoning. Our results suggest that much of a transformer's expressive power may reside in its input representation rather than its learned weights.
    
[^384]: 突破容量上限：在Stiefel流形上进行路由以构建双线性SPD层

    Escaping the Capacity Ceiling: Routing on the Stiefel Manifold for Bilinear SPD Layers

    [https://arxiv.org/abs/2605.31043](https://arxiv.org/abs/2605.31043)

    提出SCAP层，通过交叉注意力将K个Stiefel专家滤波器动态组合为样本特定的双线性映射，从而突破SPD网络中单滤波器的容量上限，解决堆叠BiMap层无法提升容量的问题。

    

    在对称正定（SPD）流形上的深度网络通过将数据几何编码为归纳偏置，有望实现富有表现力的表示，但将BiMap层与标准的ReEig非线性堆叠往往不会增加模型容量：在真实的经预处理的脑电（EEG）数据上，ReEig很少被激活，因此无论堆叠多少层，网络的表现都如同单层。在最坏情况下，当各个域之间不共享判别方向时，我们证明了单个滤波器存在容量上限，因而无法同时完全对齐所有域。为了克服这一限制，我们提出了SCAP（Stiefel交叉注意力池化），该层通过交叉注意力将K个专家组合成样本特定的双线性映射，从而实现一族Stiefel滤波器。我们证明，当各域的最优滤波器在共享切空间基点附近仅跨越少数几个方向时，该层可以用少于域数量的专家在低阶意义上匹配每个域独立的滤波器组；在最坏情况下，其对齐经验……

    arXiv:2605.31043v2 Announce Type: replace-cross  Abstract: Deep networks on the symmetric positive-definite (SPD) manifold promise expressive representations by encoding data geometry as an inductive bias, but stacking BiMap layers with the standard ReEig nonlinearity often adds no capacity: on real, preconditioned EEG data, ReEig rarely activates, so the stack behaves as a single layer at any depth. In the worst case, when domains share no discriminative directions, we prove a single filter has a capacity ceiling, so it cannot fully align every domain at once. To overcome that, we propose SCAP (Stiefel Cross-Attention Pool), a layer implementing a family of Stiefel filters by combining a pool of $K$ experts into a sample-specific bilinear map via cross-attention. We show that it matches a per-domain filter bank to first order with fewer experts than domains when domain-optimal filters span few directions near a shared tangent-space basepoint; in the worst case, its alignment empirical
    
[^385]: FastKernels：在生产环境中对GPU内核生成进行基准测试

    FastKernels: Benchmarking GPU Kernel Generation in Production

    [https://arxiv.org/abs/2605.23215](https://arxiv.org/abs/2605.23215)

    FastKernels提出了一个包含384个任务的生产级GPU内核生成基准，通过组合层次结构覆盖94.6%的HuggingFace Transformers架构，并直接在生产执行路径上以框架官方发布的内核为基准对候选内核进行内核级和端到端评分。

    

    基于大语言模型（LLM）的GPU内核生成智能体正在迅速发展，但它们所优化的基准测试往往在隔离环境中评估内核，使用合成输入和薄弱的基线，从而奖励那些在真实推理系统中会失效或无法体现的沙盒加速效果。我们提出了FastKernels，这是一个包含384个任务的基准测试，这些任务取自8个类别中的47个代表性架构，其内核足以重新实现94.6%（472/499）的HuggingFace Transformers架构，且输出与原生实现相匹配。每个任务都镜像了相应生产模块的接口，并以生产框架实际发布的内核作为评分基准；任务构成了一个组合层次结构，从底层原语到完整模型，其中高层模块会导入低层模块。候选内核在内核层面以及其所属模型的内部进行端到端评分，并在生产执行路径上进行评估，MacroEval则聚合经过校准的（原文摘要在此处截断）……

    arXiv:2605.23215v2 Announce Type: replace-cross  Abstract: LLM-based agents for GPU kernel generation are advancing rapidly, but the benchmarks they optimize against evaluate kernels in isolation, with synthetic inputs and weak baselines, rewarding sandbox speedups that break or vanish in real inference systems. We introduce FastKernels, a benchmark of 384 tasks drawn from 47 representative architectures across 8 categories, whose kernels suffice to reimplement 94.6% (472/499) of HuggingFace Transformers architectures with outputs matching the native implementations. Each task mirrors the interface of the corresponding production module and is scored against the kernels production frameworks ship, and tasks form a compositional hierarchy, from primitives to full models, in which higher-level modules import lower-level ones. Candidates are scored at the kernel level and end to end inside the models they come from, on the production execution path, and MacroEval aggregates calibrated cor
    
[^386]: 刺激对称性可能会混淆表征相似性分析

    Stimulus symmetries can confound representational similarity analyses

    [https://arxiv.org/abs/2605.21324](https://arxiv.org/abs/2605.21324)

    该论文揭示了网络输入中的刺激对称性会混淆基于表征相似性矩阵（RSM）的分析，因为功能等价的表征配置可以产生性质截然不同的表征几何结构，即使在编码图像数据的实际网络中也存在这一现象。

    

    表征相似性矩阵（RSM）能告诉我们关于神经编码的什么信息？随着这类汇总统计量的日益流行，对其性质进行更完整表征的需求也随之增长。在这里，我们展示了网络输入中的对称性可能会混淆基于RSM的分析。刺激对称性使得许多表征在功能上等价，但这些不同的配置可能导致不同的RSM。这些不同的RSM反映了性质上截然不同的表征几何结构，范围从解耦编码到最大混合编码。我们证明随机梯度下降或能量正则化可以生成稀疏的、漂移的编码，进而导致漂移的RSM。此外，我们证明这些现象存在于训练用于编码图像数据的网络中，其中对称性是潜在的。我们的结果说明了比较非线性神经编码时固有的挑战，即功能等价的编码可能产生不同的RSM。

    arXiv:2605.21324v2 Announce Type: replace-cross  Abstract: What can representational similarity matrices (RSMs) tell us about a neural code? As the popularity of these summary statistics grows, so too does the need for a more complete characterization of their properties. Here, we show that symmetries in network inputs can confound RSM-based analyses. Stimulus symmetries render many representations functionally equivalent, but these different configurations can lead to different RSMs. These different RSMs reflect qualitatively different representational geometries, ranging from disentangled to maximally-mixed codes. We show that stochastic gradient descent or energetic regularization can generate sparse, drifting codes, leading in turn to drifting RSMs. Moreover, we demonstrate that these phenomena are present in networks trained to encode image data, where the symmetry is latent. Our results illustrate the challenges inherent in comparing nonlinear neural codes, when functionally-equi
    
[^387]: TEGER：面向概率交通预测的时空协方差模型

    Teger: Spatiotemporal Covariance for Probabilistic Traffic Forecasting

    [https://arxiv.org/abs/2605.18068](https://arxiv.org/abs/2605.18068)

    提出TEGER残差协方差模型，通过闭式更新在测试时动态校正交通预测的联合不确定性而无需重新训练，并可附加于冻结的时间序列基础模型。

    

    交通状况会发生漂移——需求模式、事件动态和传感器行为会随部署生命周期不断变化——因此在训练时一次性拟合并保持静态的联合不确定性估计，会随着条件变化而失准。我们提出TEGER，一种残差协方差模型，通过闭式更新（而非重新训练）在测试时保持预测器的联合预测不确定性始终最新。固定的传感器图提供一个低维空间精度因子，编码哪些传感器的误差会共同变动；在推理阶段，高斯条件化仅根据最近观测到的残差来修正下一次预测的均值和协方差，同时指数移动平均波动率项重新缩放边际不确定性以跟踪局部漂移，并保留学习到的相关结构。这两种更新均不触及预测主干网络的权重，因此同一机制可以附加到冻结的时间序列基础模型上。

    arXiv:2605.18068v2 Announce Type: replace-cross  Abstract: Traffic conditions drift -- demand patterns, incident dynamics, and sensor behavior shift over a deployment's lifetime -- so a joint uncertainty estimate fit once at training time and left static will miscalibrate as conditions change. We present TEGER, a residual covariance model that keeps a forecaster's joint predictive uncertainty current at test time through closed-form updates, not retraining. A fixed sensor graph supplies a low-dimensional spatial precision factor encoding which sensors' errors move together; at inference, Gaussian conditioning corrects The next forecast's mean and covariance from only the most recently observed residuals, and an exponential moving-average volatility term rescales marginal uncertainty to track local drift while preserving the learned correlation structure. Neither update touches the forecasting backbone's weights, so the same mechanism attaches to a frozen time-series foundation model: n
    
[^388]: HINT-SD：面向长时程智能体的定向后见自蒸馏

    HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2605.17873](https://arxiv.org/abs/2605.17873)

    HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。

    

    arXiv:2605.17873v2 公告类型：交叉替换 摘要：使用强化学习训练长时程LLM智能体具有挑战性，因为稀疏的结果奖励能揭示任务是否成功，但无法指出哪些中间动作导致了该结果，或应如何纠正这些动作。近期方法通过从回合级动作-输出信号生成奖励或文本提示，或使用反馈条件自蒸馏来缓解此问题。然而，在每回合生成反馈效率低下，因为许多中间回合可能已经成功或中性，而在固定或错位的回合应用反馈往往无法监督导致失败的动作。为弥合这一差距，我们提出HINT-SD，一种定向自蒸馏框架，利用完整轨迹的后见之明选择与失败相关的动作，并仅对定向动作片段应用反馈条件蒸馏。在BFCL v3和AppWorld上的实验表明，我们的方法优于密集反馈方法。

    arXiv:2605.17873v2 Announce Type: replace-cross  Abstract: Training long-horizon LLM agents with reinforcement learning is challenging because sparse outcome rewards reveal whether a task succeeds, but not which intermediate actions caused the outcome or how they should be corrected. Recent methods alleviate this issue by generating rewards or textual hints from turn-level action-output signals, or by using feedback-conditioned self-distillation. However, generating feedback at every turn is inefficient when many intermediate turns are already successful or neutral, and applying feedback at a fixed or misaligned turn often fails to supervise the actions that contributed to the failure. To bridge this gap, we propose HINT-SD, a targeted self-distillation framework that uses full-trajectory hindsight to select failure-relevant actions and applies feedback-conditioned distillation only to targeted action spans. Experiments on BFCL v3 and AppWorld show that our method outperforms the dense
    
[^389]: 先辨识后实现：从部分观测中对比学习潜在端口哈密顿动力学

    Identify then Realize: Contrastive Learning of Latent Port-Hamiltonian Dynamics from Partial Observations

    [https://arxiv.org/abs/2605.16682](https://arxiv.org/abs/2605.16682)

    提出两阶段“先辨识后实现”框架CIPHER，先用对比学习从部分观测中辨识潜在状态表示，再将其实现为端口哈密顿动力学，从而学习保守与耗散系统的物理一致模型，并给出了状态可恢复性与动力学可实现性的理论保证。

    

    当在观测空间中直接建模不可行时，识别潜在状态表示与动力学至关重要，尤其是在部分观测和高维观测的场景下。在这类情况下，表示学习与物理感知建模本质上是相互耦合的。我们提出了CIPHER，这是一个用于学习保守与耗散系统的潜在端口哈密顿模型的两阶段“先辨识后实现”框架。第一阶段，一个对比式教师模型联合学习编码器与神经常微分方程（ODE），从观测历史中获得具有预测能力的状态表示；第二阶段，一个学生模型通过匹配编码后的观测未来，学习一个可逆的非线性坐标变换以及端口哈密顿动力学。我们建立了将物理状态恢复至微分同胚意义下等价、并以端口哈密顿形式实现其动力学的充分条件。在十个干净与含噪数据集上，CIPHER与最先进方法相比具有竞争力或表现更优。

    arXiv:2605.16682v2 Announce Type: replace  Abstract: Identifying latent state representations and dynamics is essential when direct modeling in observation space is infeasible, particularly under partial and high-dimensional observations. In such settings, representation learning and physics-aware modeling are inherently coupled. We propose CIPHER, a two-stage identify-then-realize framework for learning latent port-Hamiltonian models of conservative and dissipative systems. First, a contrastive teacher jointly learns an encoder and a neural ODE to obtain predictive state representations from observation histories. Second, a student learns an invertible nonlinear coordinate transformation and port-Hamiltonian dynamics by matching encoded observed futures. We establish sufficient conditions for recovering the physical state up to a diffeomorphism and realizing its dynamics in port-Hamiltonian form. Across ten clean and noisy datasets, CIPHER is competitive with or outperforms state-of-t
    
[^390]: LEAF：一个面向事件增强预测的动态基准

    LEAF: A Living Benchmark for Event-Augmented Forecasting

    [https://arxiv.org/abs/2605.16358](https://arxiv.org/abs/2605.16358)

    LEAF是首个面向事件增强预测任务的动态基准，通过递归检索智能体系统与双智能体交叉验证收集时间对齐的辅助上下文，将未来信息泄露从8.6%降至1.6%。

    

    大型语言模型（LLM）正越来越多地被应用于现实世界的预测任务，然而其真实预测能力的评估仍受到预训练数据污染以及自动化检索中前瞻性信息泄露的损害。现有基准要么依赖静态上下文，要么将评估限制在狭窄的环境中，要么未能对辅助文本事件进行未来信息泄露的审计。为了建立严格的评估范式，我们提出了LEAF——首个面向事件增强预测任务（包括趋势预测、事件预测和时间序列预测）的动态基准。LEAF将递归检索智能体系统与双智能体交叉验证相结合，以收集全面、相关且时间对齐的辅助上下文。由47位领域专家对500个任务开展的全面审计表明，我们的流程将未来信息泄露从8.6%降低至1.6%。在对16个前沿模型的广泛评估中……

    arXiv:2605.16358v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly applied to real-world forecasting tasks, yet evaluating their true predictive capability remains compromised by pre-training data contamination and look-ahead leakage in automated search. Existing benchmarks either rely on static contexts, restrict evaluations to narrow environments, or fail to audit auxiliary textual events for future information leakage. To establish a rigorous evaluation paradigm, we propose LEAF, the first living benchmark for event-augmented forecasting tasks, including trend, event, and time series forecasting. LEAF couples a recursive retrieval agent system with dual-agent cross-validation to gather comprehensive, relevant, and temporally aligned auxiliary context. A comprehensive audit across 500 tasks by 47 domain specialists demonstrates that our pipeline suppresses future information leakage from 8.6% to 1.6%. Across extensive evaluations of 16 frontier p
    
[^391]: GPart：通过全局参数划分实现端到端等距微调

    GPart: End-to-End Isometric Fine-Tuning via Global Parameter Partitioning

    [https://arxiv.org/abs/2605.14841](https://arxiv.org/abs/2605.14841)

    GPart 通过稀疏等距划分矩阵将可训练向量直接映射到完整权重空间，去除了 LoRA 式低秩重构，实现了端到端等距且高度参数高效的微调。

    

    arXiv:2605.14841v2 公告类型： replace-cross 摘要：低秩适配已成为大规模深度学习模型参数高效微调（PEFT）的主流范式。然而，其双线性参数化引入了依赖于参数的几何结构：从可训练参数到权重更新的映射通常不保持距离。将低维向量投影到 LoRA 参数空间的相关方法（如 Uni-LoRA）提升了参数效率，但随后的双线性映射破坏了端到端的等距性。我们提出 GPart（全局划分微调），这是一种高度参数高效的微调方法，通过稀疏的等距划分矩阵将一个 $d$ 维可训练向量直接映射到完整权重空间中。GPart 在保持固定的全局参数共享先验的同时，去除了基于 LoRA 的方法所使用的额外低秩重构。这带来了一个仅含单一主要超参数（$d$）的简单参数化……

    arXiv:2605.14841v2 Announce Type: replace-cross  Abstract: Low-rank adaptation (LoRA) has become a dominant paradigm for parameter-efficient fine-tuning (PEFT) of large-scale deep learning models. However, its bilinear parameterization induces a parameter-dependent geometry: the mapping from trainable parameters to weight updates is not generally distance-preserving. Related methods that project a low-dimensional vector into LoRA's parameter space, such as Uni-LoRA, improve parameter efficiency, but the subsequent bilinear map breaks end-to-end isometry. We propose GPart (Global Partition fine-tuning), a highly parameter-efficient fine-tuning method that maps a $d$-dimensional trainable vector directly into the full weight space through a sparse, isometric partition matrix. GPart retains a fixed global parameter-sharing prior while removing the additional low-rank reconstruction used by LoRA-based methods. This yields a simple parameterization with a single main hyperparameter ($d$), e
    
[^392]: ASH：在长时程世界中自我磨砺的智能体

    ASH: Agents that Self-Hone in Long-Horizon Worlds

    [https://arxiv.org/abs/2605.14211](https://arxiv.org/abs/2605.14211)

    ASH是一个无需奖励工程或专家标注的智能体系统，通过自我改进循环从自身轨迹学习逆动力学模型，进而从无标注网络视频中提取监督信号并保留关键时刻作为长期记忆，从而在《宝可梦：绿宝石》和《塞尔达传说：缩小帽》等需要数小时规划的长时程任务中实现自我提升。

    

    长时程视觉运动任务仍然是人工智能领域的一项根本性挑战，因为现有方法依赖于人工设计的奖励或带有动作标注的演示数据，而这两种方式都无法规模化。我们提出了ASH，这是一个能够从无标注、含噪声的互联网视频中学习长时程策略的智能体系统，无需奖励塑形或专家标注。ASH遵循一个自我改进循环：当它陷入困境时，ASH会从自身轨迹中学习一个逆动力学模型（IDM），并利用该IDM从相关的互联网视频中提取监督信号。ASH使用无监督学习从大规模互联网视频中识别关键时刻，并将其保留为长期记忆，从而能够应对长时程问题。我们在两个互补的、需要数小时规划的环境中评估了ASH：《宝可梦：绿宝石》（一款回合制角色扮演游戏）和《塞尔达传说：缩小帽》（一款实时动作冒险游戏）。在这两款游戏中，行为克隆、检索增强（后续内容被截断）

    arXiv:2605.14211v4 Announce Type: replace  Abstract: Long-horizon visuomotor tasks remain a fundamental challenge in AI, as current methods rely on hand-engineered rewards or action-labeled demonstrations, neither of which scales. We introduce ASH, an agentic system that learns a long-horizon policy from unlabeled, noisy internet video, without reward shaping or expert annotation. ASH follows a self-improvement loop; when it gets stuck, ASH learns an Inverse Dynamics Model (IDM) from its own trajectories, and uses its IDM to extract supervision from relevant internet video. ASH uses unsupervised learning to identify key moments from large-scale internet video and retains them as long-term memory - allowing it to tackle long-horizon problems. We evaluate ASH on two complementary environments demanding multi-hour planning: Pokemon Emerald, a turn-based RPG, and The Legend of Zelda: The Minish Cap, a real-time action-adventure game. In both games, behavioral cloning, retrieval-augmented a
    
[^393]: 具身神经计算：一种连接生物神经培养物与规模化任务驱动验证的框架

    Embodied Neurocomputation: A Framework for Interfacing Biological Neural Cultures with Scaled Task-Driven Validation

    [https://arxiv.org/abs/2605.13315](https://arxiv.org/abs/2605.13315)

    该论文提出具身神经计算框架，并首次对生物神经网络智能体在模拟网格世界中执行气味梯度闭环导航任务的编码配置开展了大规模参数优化。

    

    生物神经网络（BNN）已被确立为一种强大且具有适应性的计算基底，凭借其独特的学习机制，有望实现极其节能且数据高效的信息处理。然而，利用BNN进行神经计算的一个核心挑战在于，如何确定传统硅计算接口与活体生物之间的最优编码与解码机制。在此，我们提出了一种具身神经计算框架，作为解决这一多变量优化编码/解码问题的系统级方法。我们通过首次对编码配置进行大规模参数优化来将该框架付诸实践，让一个BNN智能体在模拟网格世界中沿气味梯度执行闭环导航任务。尽管该任务相对简单，但生物相互作用仍为最优参数的搜索产生了一个庞大的多组合空间。通过考虑……（原文摘要在此处截断）

    arXiv:2605.13315v2 Announce Type: replace-cross  Abstract: Biological neural networks (BNNs) have been established as a powerful and adaptive substrate that offer the potential for incredibly energy and data efficient information processing with distinct learning mechanisms. Yet a core challenge to utilizing BNN for neurocomputation is determining the optimal encoding and decoding mechanisms between the traditional silicon computing interface and the living biology. Here, we propose an Embodied Neurocomputation framework as a systems-level approach to this multi-variable optimization encoding/decoding problem. We operationalize this approach through the first large-scale parameter optimization of encoding configurations for a BNN agent performing closed-loop navigation along an odor-style gradient in a simulated grid-world. Despite the relative simplicity of the task, the biological interactions gave rise to a massive multi-combinatorial search space for optimal parameters. By consider
    
[^394]: ReForge：基于锚点正则化回归的合并模型精炼方法

    ReForge: Refining Merged Models with Anchor-Regularized Regression

    [https://arxiv.org/abs/2605.12843](https://arxiv.org/abs/2605.12843)

    提出双层优化框架ReForge，将强合并模型作为锚点先验，通过贝叶斯线性回归对模块级进行精炼，并利用贝叶斯优化联合选择正则化强度与组装尺度，同时提供无需校准数据的任务向量Gram变体。

    

    模型合并旨在将多个任务特定的专家模型组合成单一模型而无需联合重新训练，在数据访问或计算预算受限的情况下，为多任务学习提供了一种实用的替代方案。现有的模型合并方法很少利用强大的合并模型作为先验来进行进一步改进。为解决这一局限性，我们提出了ReForge，这是一个双层优化框架，将模块级精炼表述为具有锚点中心先验的贝叶斯线性回归。内层从无标签的校准激活中产生闭式MAP估计；外层利用贝叶斯优化，基于留出验证数据联合选择异构的正则化强度和组装尺度。此外，我们开发了ReForge的无数据变体，用任务向量Gram矩阵替代激活统计量，从而消除了对校准样本的需求。在广泛的基准测试中……

    arXiv:2605.12843v2 Announce Type: replace-cross  Abstract: Model merging aims to combine multiple task-specific expert models into a single model without joint retraining, offering a practical alternative to multi-task learning when data access or computational budget is limited. Existing model merging methods rarely exploit strong merged models as priors for further improvement. To address this limitation, we propose ReForge, a bilevel optimization framework that formulates module-wise refinement as Bayesian linear regression with an anchor-centered prior. The inner level yields a closed-form MAP estimate from unlabeled calibration activations. The outer level uses Bayesian optimization to jointly select heterogeneous regularization strengths and assembly scales using held-out validation data. Furthermore, we develop a data-free variant of ReForge that replaces activation statistics with task-vector Grams, eliminating the need for calibration examples. Across extensive benchmarks, inc
    
[^395]: 共享线性映射能走多远？探究面向图像编辑的特征空间可操纵性

    How Far Does a Shared Linear Map Go? Probing Feature-Space Manipulability for Image Editing

    [https://arxiv.org/abs/2605.11203](https://arxiv.org/abs/2605.11203)

    该研究发现，一个简单的空间共享线性映射就能近乎等同于更强表达能力的探针，预测有监督视觉骨干网络对几何、光度及语义等各类图像编辑的内部特征响应，且其充分性随网络深度增加而提升。

    

    理解图像空间的变换如何在模型的内部表征中体现，是表征分析领域的一个长期目标。先前的研究表明，几何变换通常可以通过在特征图之间学习到的线性算子来捕获，但尚不清楚这是否同样适用于光度（photometric）、局部以及语义定义的编辑。我们训练了一系列容量递增的探针——从空间共享的线性映射，到非线性的逐向量探针、感受野探针以及全局Transformer模型——来预测由几何变换、光度编辑、遮挡以及扩散模型生成的语义编辑所引起的特征空间变化。在ConvNeXt、SwinV2和DINOv3三种模型上，对于有监督的骨干网络，单个共享线性映射对留出编辑操作结果的预测往往几乎与表达能力更强的探针相当，且这种充分性通常随网络深度增加而提升；但这一规律在DINOv3上并不那么一致。

    arXiv:2605.11203v2 Announce Type: replace  Abstract: Understanding how image-space transformations manifest in a model's internal representations is a longstanding goal in representation analysis. Prior work has shown that geometric transformations can often be captured by learned linear operators between feature maps, but it remains unclear whether this extends to photometric, local, and semantically defined edits. We train probes of increasing capacity from a spatially shared linear map to nonlinear per-vector, receptive-field, and global transformer models to predict feature-space changes induced by geometric transforms, photometric edits, occlusions, and diffusion-generated semantic edits. Across ConvNeXt, SwinV2, and DINOv3, a single shared linear map often predicts held-out manipulation outcomes nearly as well as substantially more expressive probes for the supervised backbones, with sufficiency generally increasing with depth; this pattern is less consistent for DINOv3. These re
    
[^396]: 教会大语言模型“看见”图：统一文本与结构推理

    Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning

    [https://arxiv.org/abs/2605.10247](https://arxiv.org/abs/2605.10247)

    GTLM通过将图感知注意力偏置直接注入预训练LLM的注意力模块，仅以0.015%的额外参数使LLM原生处理图拓扑结构，无需GNN流水线即可统一文本与图结构推理，避免了节点语义信息的丢失。

    

    将大型语言模型（LLM）应用于图结构数据通常需要多步骤流水线，其中文本节点属性被压缩为单个token并交由图神经网络（GNN）进一步处理，导致大部分语义内容被丢弃。我们提出了图Transformer语言模型（GTLM），它使预训练的LLM能够原生地处理图拓扑结构，从而完全消除了这一瓶颈。GTLM将图感知的注意力偏置直接注入LLM的注意力模块，相对于基础模型仅增加0.015%的结构相关参数。训练时仅更新结构参数以及基础模型上的LoRA适配器。我们证明了所提出的双向注意力前缀对节点具有置换等变性，并且当不存在图结构时，GTLM可以精确退化为原始预训练模型。由于不依赖全局节点排序，GTLM不会出现位置退化，也不会“迷失在中间”……（摘要原文在此处截断）

    arXiv:2605.10247v2 Announce Type: replace  Abstract: Applying Large Language Models (LLMs) to graph-structured data usually involves multi-step pipelines in which textual node attributes are compressed into single tokens and further processed by GNNs, discarding most of their semantic content. We introduce the Graph Transformer Language Model (GTLM), which enables a pretrained LLM to process graph topology natively and removes this bottleneck entirely. GTLM injects graph-aware attention biases directly into the LLM's attention modules, adding only 0.015\% structure-related parameters relative to the base model. Training updates only the structural parameters together with a LoRA adapter on the base model. We prove that our bidirectional attention prefix is permutation-equivariant over nodes and that GTLM reduces exactly to the pretrained model when no graph is present. Having no global node ordering, GTLM shows no positional degradation and does not \textit{get lost in the middle}: nee
    
[^397]: 基于交叉注意力层的系统提示词锚定

    System-Prompt Anchoring with Cross-Attention Layers

    [https://arxiv.org/abs/2605.09737](https://arxiv.org/abs/2605.09737)

    该研究通过在冻结的因果解码器主干中插入交叉注意力层来锚定系统提示词，发现插入位置对性能影响显著，较后的位置通常更有效且参数效率更高，并能改变指令遵循与安全行为而不损害通用任务性能。

    

    交叉注意力提供了一条从选定信息源进入模型计算的专用通道，但该通道插入位置的影响仍未得到充分探索。当信息源是特权系统提示词片段时，我们研究了这一问题。我们在保持因果解码器主干冻结的情况下，在系统提示词与文本之间插入交叉注意力层（CAL）模块。在1.5B主干模型上进行的十种配置扫描表明，性能依赖于具体任务，并受插入位置的强烈影响：较后的插入位置通常更有效且参数效率更高。在一项8B模型的扩展研究中，我们仅训练整体最优配置，并将其与参数量匹配的适配基线进行比较。在所评估的基准测试中，效果仍然依赖于任务：交叉注意力改变了指令遵循和安全行为，同时在很大程度上保持了通用任务的性能。总之，这些实验将插入位置刻画为一种……

    arXiv:2605.09737v3 Announce Type: replace  Abstract: Cross-attention provides a dedicated route from a selected information source into a model's computation, but the effect of where that route is inserted remains underexplored. We study this question when the source is a privileged system-prompt span. We insert Cross-Attention Layer (CAL) blocks between the system prompt and text while keeping the causal-decoder backbone frozen. A ten-configuration sweep on a 1.5B backbone shows that performance is task-dependent and strongly affected by placement: later placements are generally more effective and parameter-efficient. In an 8B scaling study, we train only the overall best configuration and compare it with parameter-matched adaptation baselines. Across the evaluated benchmarks, the effects remain task-dependent: cross-attention changes instruction-following and security behavior while largely preserving general-task performance. Together, these experiments characterize placement as an 
    
[^398]: MolWorld：面向可操作分子优化的分子世界模型

    MolWorld: Molecule World Models for Actionable Molecular Optimization

    [https://arxiv.org/abs/2605.08954](https://arxiv.org/abs/2605.08954)

    提出分子世界模型 MolWorld，将可操作分子优化形式化为分子转移图的迭代扩展，通过匹配分子对（MMP）边显式建模可达性，确保优化得到的候选分子可从已知分子经局部结构修饰到达。

    

    药物发现中的分子优化旨在发现具有更优目标性质的分子，但实际的先导化合物优化往往需要的不只是高预测分数。一个有用的候选分子还应当是“可操作的”：即它应当能够通过一系列局部结构修饰从已知分子出发到达，从而为在不断演化的化学系列中解释性质变化提供明确的结构参照。现有的从头设计与单分子优化方法并未显式地对这种可达性进行建模，尤其是当目标分子以及将其与已知化合物相连的中间分子均为未知时。在本工作中，我们将可操作分子优化形式化为分子转移图的迭代扩展，其中节点表示分子，边编码表示局部结构差异的匹配分子对关系。我们提出了 MolWorld，一种分子世界模型……

    arXiv:2605.08954v2 Announce Type: replace-cross  Abstract: Molecular optimization in drug discovery aims to discover molecules with improved target properties, but practical lead optimization often requires more than high predicted scores. A useful candidate should also be actionable: it should be reachable from known molecules through a sequence of local structural modifications, providing explicit structural references for interpreting property changes within an evolving chemical series. Existing de novo and single-molecule optimization methods do not explicitly model such reachability, especially when both the target molecules and the intermediate molecules connecting them to known compounds are unknown. In this work, we formulate actionable molecular optimization as iterative expansion of a molecule-transfer graph, where nodes are molecules and edges encode matched molecular pair (MMP) relations representing localized structural differences. We propose MolWorld, a molecule world mo
    
[^399]: 递归智能体优化

    Recursive Agent Optimization

    [https://arxiv.org/abs/2605.06639](https://arxiv.org/abs/2605.06639)

    RAO提出了一种强化学习方法，通过训练智能体递归地生成并委派子任务给自身的新实例来实现推理时的分治扩展，使模型能够突破上下文窗口限制、泛化到远难于训练任务的问题，并降低实际运行时间。

    

    我们提出了递归智能体优化（Recursive Agent Optimization, RAO），这是一种用于训练递归智能体的强化学习方法：递归智能体能够递归地生成并将子任务委派给自身的新实例。递归智能体实现了一种推理时扩展算法，通过分治法使智能体能够自然地扩展到更长的上下文，并泛化到更困难的问题。RAO提供了一种训练模型以充分利用这种递归推理的方法，教会智能体何时以及如何进行委派和沟通。我们发现，以这种方式训练的递归智能体具有更好的训练效率，能够扩展到超出模型上下文窗口的任务，泛化到比训练任务难得多的问题，并且与单智能体系统相比可以减少实际运行时间。

    arXiv:2605.06639v2 Announce Type: replace-cross  Abstract: We introduce Recursive Agent Optimization (RAO), a reinforcement learning approach for training recursive agents: agents that can spawn and delegate sub-tasks to new instantiations of themselves recursively. Recursive agents implement an inference-time scaling algorithm that naturally allows agents to scale to longer contexts and generalize to more difficult problems via divide-and-conquer. RAO provides a method to train models to best take advantage of such recursive inference, teaching agents when and how to delegate and communicate. We find that recursive agents trained in this way enjoy better training efficiency, can scale to tasks that go beyond the model's context window, generalize to tasks much harder than the ones the agent was trained on, and can enjoy reduced wall-clock time compared to single-agent systems.
    
[^400]: 基于随机分配子采样的DP-SGD权衡函数：紧致上下界

    Trade-off Functions for DP-SGD with Subsampling based on Random Allocation: Tight Upper and Lower Bounds

    [https://arxiv.org/abs/2605.06259](https://arxiv.org/abs/2605.06259)

    该论文在f-DP框架下首次对基于随机分配子采样的DP-SGD给出了紧致的权衡函数分析，利用Berry-Esseen定理推导出透明、可解释的封闭形式上下界，在单个epoch下紧致至常数因子。

    

    在 $f$-DP 框架下，我们对基于随机分配的子采样的差分隐私随机梯度下降（DP-SGD）的权衡函数进行了紧致分析，其中每个样本在每个epoch中被独立分配到 $M$ 个小批次（minibatch）中的恰好一个，每个小批次对应单个epoch内 $M$ 轮SGD中的一轮。我们的分析在一个明确的有效性条件下成立，其假设共同要求噪声乘子 $\sigma \geq \sqrt{3/\ln M}$。与Poisson子采样的 $f$-DP 分析不同——后者产生的是非封闭的隐式公式，虽然可被机器计算但不透明——随机分配允许进行紧致分析，从而产生透明且可解释的封闭形式界。对于单个epoch，我们通过Berry-Esseen定理推导出的具体界在常数因子范围内是紧致的。我们展示了单个epoch（$E=1$）的具体参数设置。

    arXiv:2605.06259v3 Announce Type: replace  Abstract: Within the $f$-DP framework, we derive a tight analysis of the trade-off function for Differentially Private Stochastic Gradient Descent (DP-SGD) with subsampling based on random allocation in which each sample is independently assigned to exactly one of $M$ minibatches per epoch, each minibatch corresponding to one of the $M$ SGD rounds within a single epoch. Our analysis holds under an explicit validity condition, whose hypotheses together force $\sigma \geq \sqrt{3/\ln M}$, where $\sigma$ is the DP noise multiplier. Unlike $f$-DP analyses for Poisson subsampling, which yield non-closed implicit formulas that can be machine computed but are non-transparent, random allocation admits a tight analysis yielding transparent and interpretable closed-form bounds. For a single epoch, our concrete bounds, derived via the Berry-Esseen theorem, are tight up to constant factors. We demonstrate worked parameter settings for a single epoch ($E=1
    
[^401]: CoMemNet：一种带漂移感知采样的持续记忆网络用于交通预测

    CoMemNet: A Continual Memory Network with Drift-Aware Sampling for Traffic Prediction

    [https://arxiv.org/abs/2605.05738](https://arxiv.org/abs/2605.05738)

    提出 CoMemNet，一种通过在线/目标双分支、基于 Wasserstein 的漂移感知采样和节点自适应时间记忆重放缓冲，在演进的交通传感器网络上实现无需固定邻接矩阵与全量重训的高效持续交通预测模型。

    

    交通传感器网络会随着传感器的增加和交通分布的变化而不断演进，而大多数预测模型假设节点集是固定的，并在所有可用数据上反复重新训练。我们提出了 CoMemNet，一种面向不断演进的交通传感器网络的高效预测持续记忆网络。CoMemNet 使用一个在线分支来适应当前时期，并使用一个指数移动平均的目标分支作为稳定的特征参考。基于 Wasserstein 距离的漂移采样器比较节点级的在线-目标特征分布，并选择有限的一组漂移敏感节点进行更新。轻量级的节点自适应时间记忆重放缓冲区 保留紧凑的时间状态，无需反复遍历所有历史训练数据。预测主干网络不消耗邻接矩阵；传感器邻接关系仅用于构造数据，并可选择地将所选更新集扩展到有限的邻域。实验……

    arXiv:2605.05738v2 Announce Type: replace-cross  Abstract: Traffic sensor networks evolve as sensors are added and traffic distributions change, whereas most forecasting models assume a fixed node set and repeatedly retrain on all available data. We propose CoMemNet, a Continual Memory Network for efficient prediction over evolving traffic sensor networks. CoMemNet uses an Online branch to adapt to the current period and an exponential-moving-average Target branch as a stable feature reference. A Wasserstein-based Drift Sampler compares node-wise Online-Target feature distributions and selects a limited set of drift-sensitive nodes for updating. A lightweight Node-Adaptive Temporal Memory Replay Buffer (TMRB-N) retains compact temporal states without repeatedly traversing all historical training data. The prediction backbone does not consume an adjacency matrix; sensor adjacency is used only to construct data and optionally expand the selected update set to a limited neighborhood. Expe
    
[^402]: 输入凸神经网络的双认证白盒推断

    Dual Certified White-Box Inference for Input Convex Neural Networks

    [https://arxiv.org/abs/2605.04722](https://arxiv.org/abs/2605.04722)

    该论文提出利用SOC-ICNN与参数化二阶锥规划价值函数的精确对偶表示，开发双认证白盒推断方法DCI，从最优对偶乘子恢复完整次微分、提供精确平稳性认证与下降方向，并实现牛顿加速与全局收敛。

    

    输入凸神经网络（ICNNs）用于学习凸目标函数，其极小值点定义了决策，因此高效且可靠的优化是推断的核心。在非光滑输入处，自动微分仅返回单个导数，而非决定最优性与下降方向的完整次微分。二阶锥输入凸神经网络（SOC-ICNNs）可以被精确表示为参数化二阶锥规划的价值函数，这提供了一种白盒方法，能够从最优对偶乘子中恢复其完整次微分，并在光滑区域上推导出显式的Hessian矩阵。基于这一表示，我们开发了双认证推断（DCI），该方法结合网络与可行集的几何结构，获得精确的平稳性认证以及切向公共下降方向。DCI利用局部曲率实现牛顿加速，并采用精确的近端保护机制。我们建立了全局收敛性，并在标（准假设下）……

    arXiv:2605.04722v2 Announce Type: replace-cross  Abstract: Input convex neural networks (ICNNs) are used to learn convex objectives whose minimizers define decisions, making efficient and reliable optimization central to inference. At nonsmooth inputs, automatic differentiation returns a single derivative rather than the full subdifferential governing optimality and descent. Second-order cone ICNNs (SOC-ICNNs) admit an exact representation as value functions of parametric second-order cone programs, providing a white-box approach to recovering their full subdifferentials from optimal dual multipliers and deriving explicit Hessians on smooth regions. Building on this representation, we develop dual-certified inference (DCI), which combines the network and feasible set geometries to obtain exact stationarity certificates and tangent common descent directions. DCI uses local curvature for Newton acceleration and an exact proximal safeguard. We establish global convergence and, under stand
    
[^403]: 有用的特征，反向的得分：语言模型轨迹中的分布外（OOD）检测

    Useful Features, Backward Scores: OOD in Language-Model Trajectories

    [https://arxiv.org/abs/2605.00269](https://arxiv.org/abs/2605.00269)

    该论文的核心发现是，能区分输入组的特征并不一定能产生有用的OOD异常排序——在语言模型轨迹中，可区分特征对应的距离得分甚至会出现反转（异常组中心更远但散布更紧），且这一现象在毒性、反讽等多个数据集上均稳定存在。

    

    分布外（OOD）检测器用于对输入进行优先级排序以便进一步检查。然而，能够区分输入组的特征并不一定能产生有用的异常排序。我们在文本长度控制和固定得分方向的条件下，分析语言模型轨迹中的这一差距。在Spam（垃圾信息）开发数据上，D²HScore的输入适配版本在长度匹配后，AUROC从原始的0.919降至0.530。在长度匹配的留出HateSpeech（仇恨言论）输入上，相同的特征使有标签的线性分类器达到AUROC 0.644，但基于分布内（ID）拟合的距离得分仅为0.444。ToxicChat数据集显示出同样的对比。特征选择和骨干网络的对照实验保留了主要的反转模式。冻结的Civil Comments和TweetEval反讽测试也出现反转（0.467和0.435），将这一发现扩展到了毒性检测之外。在这些对比中，异常组的中心距离更远，但散布更紧。一种有标签的、固定中心的特征空间干预改变了排序：均衡散布有助于某些任务……（摘要截断）

    arXiv:2605.00269v2 Announce Type: replace  Abstract: Out-of-distribution (OOD) detectors prioritize inputs for closer inspection. Yet features that distinguish input groups need not yield a useful anomaly ranking. We analyze this gap in language-model trajectories under text-length control and fixed score directions. On Spam development data, an input adaptation of D^2HScore falls from raw AUROC 0.919 to 0.530 after length matching. On length-matched, held-out HateSpeech inputs, the same features yield AUROC 0.644 for a labeled linear classifier but 0.444 for an ID-fitted distance score. ToxicChat shows the same contrast. Feature-selection and backbone controls retain the main reversal pattern. Frozen Civil Comments and TweetEval irony tests also reverse (0.467 and 0.435), extending the finding beyond toxicity. In these contrasts, anomalous groups have farther centers but tighter spread. A labeled, fixed-center feature-space intervention changes rankings: equalizing spread helps some t
    
[^404]: 时间尺度分离使深度强化学习能够控制旋转爆震发动机的模式转换

    Timescale Separation Enables Deep Reinforcement Learning Control of Rotating Detonation Engine Mode Transitions

    [https://arxiv.org/abs/2604.14398](https://arxiv.org/abs/2604.14398)

    通过在跟随爆震波的移动参考系中重新构建深度强化学习问题，实现快速爆震传播与慢速模式动力学之间的时间尺度分离，从而使DRL能够有效控制旋转爆震发动机的模式转换。

    

    旋转爆震发动机（RDE）是一种有前景的推进概念，相比传统系统可能提供更高的热力学效率和比冲，但包括向振荡或混沌传播模式转变在内的非线性现象可能阻碍其实际运行。深度强化学习（DRL）已成为控制诸如RDE中所观察到的复杂非线性动力学的一种有前景的方法。然而，RDE系统的多时间尺度特性使得直接应用DRL面临挑战。我们通过在跟随爆震波模式的移动参考系中重新构建DRL问题来解决这一挑战，使波结构对智能体呈现准稳态特性。这种重构方法实现了快速爆震传播与较慢运行模式动力学之间的尺度分离。我们训练DRL控制器在一维降阶模型中调节空间分段注入压力……

    arXiv:2604.14398v2 Announce Type: replace-cross  Abstract: Rotating detonation engines (RDEs) are a promising propulsion concept that may offer higher thermodynamic efficiency and specific impulse than conventional systems, but nonlinear phenomena, including transitions to oscillatory or chaotic propagation modes, can hinder practical operation. Deep Reinforcement Learning (DRL) has emerged as a promising method for controlling complex nonlinear dynamics such as those observed in RDEs. However, the multi-timescale nature of the RDE system makes direct application of DRL challenging. We address this challenge by reformulating the DRL problem in a moving reference frame that follows the detonation-wave pattern, making the wave structure appear quasi-steady to the agent. This reformulation enables scale separation between fast detonation propagation and slower operating-mode dynamics. We train DRL controllers to modulate spatially segmented injection pressure in a one-dimensional reduced-
    
[^405]: 大语言模型表示中的反问句：一项线性探针研究

    Rhetorical Questions in LLM Representations: A Linear Probing Study

    [https://arxiv.org/abs/2604.14128](https://arxiv.org/abs/2604.14128)

    该研究通过线性探针发现大语言模型在表示空间中能够早期且稳定地编码反问句信号，其跨数据集可迁移性虽然存在，但并不意味着模型内部存在统一的共享表示。

    

    反问句的提出并非为了获取信息，而是为了说服他人或表明立场。然而大型语言模型如何在内部表示这类问句仍不清楚。我们使用线性探针在两个具有不同话语语境的社交媒体数据集上分析了LLM表示中的反问句，发现反问信号在早期就已显现，且最后 token 表示能最稳定地捕获这一信号。反问句在数据集内部与寻求信息的问题线性可分，在跨数据集迁移场景下仍可被检测到，AUROC 约达到 0.7-0.8。然而，我们证明这种可迁移性并不简单地意味着存在共享表示。在不同数据集上训练的探针应用于同一目标语料库时会产生不同的排名，排名靠前的实例之间的重叠度往往低于 0.2。定性分析表明，这些分歧对应于不同的修辞现象……

    arXiv:2604.14128v3 Announce Type: replace-cross  Abstract: Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some pr
    
[^406]: 梯度下降的最后一次迭代往往（略微）次优

    Gradient Descent's Last Iterate is Often (slightly) Suboptimal

    [https://arxiv.org/abs/2604.13870](https://arxiv.org/abs/2604.13870)

    本文证明了Jain等人提出的猜想：在没有时间范围 $T$ 先验知识的情况下，任何步长序列都无法使SGD的最后迭代点达到最优的 $1/\sqrt{T}$ 收敛速率，即使无噪声的梯度下降也不可避免地存在关于 $T$ 的多对数因子损失。

    

    我们考虑了使用梯度下降（GD）或其随机变体（SGD）来最小化凸Lipschitz函数这一被广泛研究的设定，并研究了最后迭代点的收敛性。迄今为止，已知标准的步长选择会导致经过 $T$ 步后最后迭代点的收敛速率为 $\log T/\sqrt{T}$。Jain等人[2019]的突破性结果通过构造一个非标准的步长序列，恢复了最优的 $1/\sqrt{T}$ 速率。然而，该序列需要预先选定 $T$，这与适用于任意时间范围的一般步长调度不同。此外，Jain等人猜想，在没有关于 $T$ 的先验知识的情况下，没有任何步长序列能够确保SGD最后迭代点的最优误差，这一论断至今尚未被证明。我们证明了这一猜想，并且实际上进一步表明，即使在GD的无噪声情形下，当考虑任意时刻（anytime）的最后迭代点时，也无法避免一个关于 $T$ 的多对数因子的额外损失。

    arXiv:2604.13870v2 Announce Type: replace-cross  Abstract: We consider the well-studied setting of minimizing a convex Lipschitz function using either gradient descent (GD) or its stochastic variant (SGD), and examine the last iterate convergence. By now, it is known that standard stepsize choices lead to a last iterate convergence rate of $\log T/\sqrt{T}$ after $T$ steps. A breakthrough result of Jain et al. [2019] recovered the optimal $1/\sqrt{T}$ rate by constructing a non-standard stepsize sequence. However, this sequence requires choosing $T$ in advance, as opposed to common stepsize schedules which apply for any time horizon. Moreover, Jain et al. conjectured that without prior knowledge of $T$, no stepsize sequence can ensure the optimal error for SGD's last iterate, a claim which so far remained unproven. We prove this conjecture, and in fact show that even in the noiseless case of GD, it is impossible to avoid an excess poly-log factor in $T$ when considering an anytime last
    
[^407]: 通过能量守恒下降实现非凸优化的经典与量子加速

    Classical and Quantum Speedups for Non-Convex Optimization via Energy Conserving Descent

    [https://arxiv.org/abs/2604.13022](https://arxiv.org/abs/2604.13022)

    本文首次对能量守恒下降（ECD）进行了理论分析，证明其随机版本和量子版本在非凸优化中相比随机梯度下降和量子隧穿游走基线均能实现指数级的命中时间加速，且量子版本在高势垒问题上具有进一步的加速优势。

    

    我们提出了对能量守恒下降（ECD）的首个解析研究，作为第一部分聚焦于一维情形。我们形式化了带有能量保持噪声的随机ECD动力学，以及ECD哈密顿量的量子类比（qECD），为在一个可处理的、可显式计算势垒穿越机制的模型中通过哈密顿量模拟构建量子算法奠定了基础。对于欠猜测机制下的一维双井目标函数，我们计算了从局部极小值到全局极小值的期望动力学命中时间。我们证明sECD和qECD相对于它们各自的基于梯度的基线——随机梯度下降（SGD）和量子隧穿游走（QTW）——在连续命中时间上均表现出指数级改进。对于具有高势垒的目标函数，qECD相比sECD具有进一步的命中时间改进。从机制上讲，ECD规避了指数代价

    arXiv:2604.13022v2 Announce Type: replace-cross  Abstract: We present the first analytical study of ECD, focusing on the one-dimensional setting for this first installment. We formalize a stochastic ECD dynamics (sECD) with energy-preserving noise, as well as a quantum analog of the ECD Hamiltonian (qECD), providing the foundation for a quantum algorithm through Hamiltonian simulation in a tractable model where the barrier-crossing mechanism can be computed explicitly. For one-dimensional double-well objectives in the under-guessing regime, we compute the expected dynamical hitting times from a local minimum to the global minimum. We prove that both sECD and qECD exhibit exponential improvements in continuous hitting time relative to their respective gradient-based baselines, stochastic gradient descent (SGD) and quantum tunneling walk (QTW). For objectives with tall barriers, qECD admits a further hitting time improvement over sECD. Mechanistically, ECD sidesteps the exponential cost 
    
[^408]: 面向个性化数字健康干预的视频中多模态矛盾/犹豫情绪识别

    Multimodal Ambivalence/Hesitancy Recognition in Videos for Personalized Digital Health Interventions

    [https://arxiv.org/abs/2604.11730](https://arxiv.org/abs/2604.11730)

    本文针对个性化数字健康干预，研究从视频中多模态识别矛盾与犹豫（A/H）情绪——这种跨模态或模态内的情感不一致状态是导致患者延迟、回避或放弃健康干预的关键因素。

    

    基于行为科学的健康干预通过提供框架帮助患者养成并维持有益于改善医疗结果的健康习惯，从而专注于行为改变。面对面的干预成本高昂且难以规模化，尤其是在资源有限的地区。数字健康干预提供了一种经济高效的途径，有望支持独立生活与自我管理。此类干预的自动化，特别是通过机器学习实现的自动化，近来受到了广泛关注。矛盾与犹豫情绪（A/H）是个体延迟、回避或放弃健康干预的主要因素。A/H是一种微妙且相互冲突的情绪状态，使个体处于对某一行为的积极与消极评价之间，或处于接受与拒绝参与该行为之间。它们表现为跨模态或模态内部的情感不一致，例如语言、面部和声音表达等。

    arXiv:2604.11730v5 Announce Type: replace-cross  Abstract: Using behavioural science, health interventions focus on behaviour change by providing a framework to help patients acquire and maintain healthy habits that improve medical outcomes. In-person interventions are costly and difficult to scale, especially in resource-limited regions. Digital health interventions offer a cost-effective approach, potentially supporting independent living and self-management. Automating such interventions, especially through machine learning, has recently gained considerable attention. Ambivalence and hesitancy (A/H) play a primary role for individuals to delay, avoid, or abandon health interventions. A/H are subtle and conflicting emotions that place a person in a state between positive and negative evaluations of a behaviour, or between acceptance and refusal to engage in it. They manifest as affective inconsistency across modalities or within a modality, such as language, facial, vocal expressions
    
[^409]: 先验证再修复：基于智能体执行验证的可信跨语言代码分析

    Verify Before You Fix: Agentic Execution Grounding for Trustworthy Cross-Language Code Analysis

    [https://arxiv.org/abs/2604.10800](https://arxiv.org/abs/2604.10800)

    该论文的核心创新是提出一个由LLM驱动的跨语言漏洞生命周期框架，以“未经执行确认可利用性就不得修复”这一严格不变式为准则，将结构-语义混合检测、基于执行的智能体验证与感知验证的迭代修复三个阶段串联起来，并借助通用抽象语法树与 GraphSAGE、Qwen2.5-Coder 嵌入的混合融合实现 Java、Python、C++ 的跨语言泛化，从而保证代码分析与修复建立在可验证的证据之上。

    

    部署在智能体流水线中的学习型分类器面临一个根本性的可靠性问题：预测只是概率性推断而非经过验证的结论，若不基于可观察的证据就对其采取行动，会在下游各阶段引发不断累积放大的失败。软件漏洞分析使这一代价变得具体且可度量。我们通过一个统一的跨语言漏洞生命周期框架来解决该问题，该框架由三个大语言模型（LLM）驱动的推理阶段构成——结构-语义混合检测、基于执行验证的智能体验证，以及感知验证结果的迭代修复——并受一条严格不变式约束：在未通过执行手段确认可利用性之前，不采取任何修复行动。跨语言泛化能力通过通用抽象语法树实现，该树将 Java、Python 和 C++ 归一化为统一的结构化模式，并与 GraphSAGE 与 Qwen2.5-Coder-1.5B 嵌入表示的混合融合相结合……

    arXiv:2604.10800v2 Announce Type: replace-cross  Abstract: Learned classifiers deployed in agentic pipelines face a fundamental reliability problem: predictions are probabilistic inferences, not verified conclusions, and acting on them without grounding in observable evidence leads to compounding failures across downstream stages. Software vulnerability analysis makes this cost concrete and measurable. We address this through a unified cross-language vulnerability lifecycle framework built around three LLM-driven reasoning stages-hybrid structural-semantic detection, execution-grounded agentic validation, and validation-aware iterative repair-governed by a strict invariant: no repair action is taken without execution-based confirmation of exploitability. Cross-language generalization is achieved via a Universal Abstract Syntax Tree (uAST) normalizing Java, Python, and C++ into a shared structural schema, combined with a hybrid fusion of GraphSAGE and Qwen2.5-Coder-1.5B embeddings throu
    
[^410]: 线性二次调节器的标量联邦学习

    Scalar Federated Learning for Linear Quadratic Regulator

    [https://arxiv.org/abs/2604.05088](https://arxiv.org/abs/2604.05088)

    提出ScalarFedLQR算法，让每个智能体仅上传一个标量梯度投影，将上行通信量从O(d)降至O(1)，并且参与智能体越多、梯度重构越精确、线性收敛越快，实现了高维LQR控制的高效无模型联邦学习。

    

    我们提出了ScalarFedLQR，这是一种通信高效的联邦算法，用于在协作智能体的线性二次调节器（LQR）控制中以无模型方式学习共同策略。该方法建立在一种分解的投影梯度机制之上，其中每个智能体仅需传输其本地零阶梯度估计的一个标量投影。服务器聚合这些标量消息以重构全局下降方向，从而将每个智能体的上行通信量从O(d)降至O(1)，且与策略维度无关。至关重要的是，投影引起的近似误差会随着参与智能体数量的增加而减小，由此产生一种有利的扩展规律：更大规模的智能体集群能够实现更精确的梯度恢复、允许更大的步长，并在高维情况下依然实现更快的线性收敛。在同质智能体的标准正则性条件下，所有迭代均保持稳定，且平均LQR代价

    arXiv:2604.05088v2 Announce Type: replace-cross  Abstract: We propose ScalarFedLQR, a communication-efficient federated algorithm for model-free learning of a common policy in linear quadratic regulator (LQR) control of cooperative agents. The method builds on a decomposed projected gradient mechanism, in which each agent communicates only a scalar projection of a local zeroth-order gradient estimate. The server aggregates these scalar messages to reconstruct a global descent direction, reducing per-agent uplink communication from O(d) to O(1), independent of the policy dimension. Crucially, the projection-induced approximation error diminishes as the number of participating agents increases, yielding a favorable scaling law: larger fleets enable more accurate gradient recovery, admit larger stepsizes, and achieve faster linear convergence despite high dimensionality. Under standard regularity conditions for homogeneous agents, all iterates remain stabilizing and the average LQR cost d
    
[^411]: 学习一种可解释的风险评分系统以最大化决策净收益

    Learning an Interpretable Risk Scoring System for Maximizing Decision Net Benefit

    [https://arxiv.org/abs/2604.04241](https://arxiv.org/abs/2604.04241)

    本文提出一种通过稀疏整数线性规划直接优化决策净收益、且具有整数系数的可解释风险评分系统，并建立了净收益与判别力、校准之间的理论关系。

    

    风险评分系统被广泛应用于高风险领域以辅助决策。然而，现有方法通常侧重于优化预测准确率或基于似然的准则，这可能与最大化效用的主要目标不一致。在本文中，我们提出了一种新颖的风险评分系统，该系统在一系列决策阈值上直接优化净收益。该模型被表述为一个稀疏整数线性规划问题，从而能够构建具有整数系数的透明评分系统，因此便于解释和实际应用。我们还建立了净收益、判别力和校准之间的基本关系。具体而言，我们推导了净收益曲线下面积与ROC泛函之间的界（两者均在固定阈值网格上计算），并证明后处理可以在不降低净收益的情况下在训练数据上实现适度的校准。

    arXiv:2604.04241v3 Announce Type: replace  Abstract: Risk scoring systems are widely used in high-stakes domains to assist decision-making. However, existing approaches often focus on optimizing predictive accuracy or likelihood-based criteria, which may not align with the main goal of maximizing utility. In this paper, we propose a novel risk scoring system that directly optimizes net benefit over a range of decision thresholds. The model is formulated as a sparse integer linear programming problem which enables the construction of a transparent scoring system with integer coefficients, and hence, facilitates interpretation and practical application. We also establish fundamental relationships among net benefit, discrimination, and calibration. Specifically, we derive bounds relating the area under the net benefit curve to a ROC functional, both evaluated on a fixed threshold grid, and show that post-processing can achieve moderate calibration on the training data without decreasing t
    
[^412]: 通过时间抽象实现前向-后向表示中的谱对齐

    Spectral Alignment in Forward-Backward Representations via Temporal Abstraction

    [https://arxiv.org/abs/2603.20103](https://arxiv.org/abs/2603.20103)

    本文证明时间抽象如同低通滤波器，可抑制高频谱分量、降低后继表示的有效秩并保持价值函数误差界，从而缓解连续环境高秩转移动力学与FB低秩瓶颈之间的谱失配，是实现稳定前向-后向表示学习的关键因素。

    

    前向-后向（FB）表示通过强制低秩分解，为在连续空间中学习后继表示（SR）提供了一个强大的框架。然而，连续环境的高秩转移动力学与FB架构的低秩瓶颈之间往往存在根本性的谱失配，这使得准确的低秩表示学习变得困难。在这项工作中，我们分析了时间抽象作为缓解这种失配的机制。通过刻画转移算子的谱特性，我们证明时间抽象的作用类似于一个抑制高频谱分量的低通滤波器。这种抑制降低了诱导SR的有效秩，同时保持了所得价值函数误差的形式化界。实验表明，这种对齐是稳定FB学习的关键因素，尤其是在高折扣因子的情况下。

    arXiv:2603.20103v4 Announce Type: replace-cross  Abstract: Forward-backward (FB) representations provide a powerful framework for learning the successor representation (SR) in continuous spaces by enforcing a low-rank factorization. However, a fundamental spectral mismatch often exists between the high-rank transition dynamics of continuous environments and the low-rank bottleneck of the FB architecture, making accurate low-rank representation learning difficult. In this work, we analyze temporal abstraction as a mechanism to mitigate this mismatch. By characterizing the spectral properties of the transition operator, we show that temporal abstraction acts analogously to a low-pass filter that suppresses high-frequency spectral components. This suppression reduces the effective rank of the induced SR while preserving a formal bound on the resulting value function error. Empirically, we show that this alignment is a key factor for stable FB learning, particularly at high discount factor
    
[^413]: 一种具有理论保证的局部惩罚交叉估计联邦方法，用于约束个性化学习

    A Locally Penalized Cross-Estimate Federated Method with Guarantees for Constrained Personalized Learning

    [https://arxiv.org/abs/2603.19617](https://arxiv.org/abs/2603.19617)

    该论文提出了一种局部惩罚交叉估计联邦平均方法，让每个参与方在异构可行约束下维护各自独立的可行模型，并通过协作目标耦合各模型，为约束个性化联邦学习提供了理论保证。

    

    arXiv:2603.19617v2 公告类型：替换 摘要：联邦学习是一种用于求解分布式优化问题的高效通信框架。标准的联邦学习方法将本地更新聚合为所有参与方共享的单一全局模型，从而有效地施加了共识约束。然而，在异构的本地约束条件下，一个共同可行的模型可能过于受限，甚至可能不存在。现有的约束联邦学习方法大多保留了这种共享模型结构，因此无法直接处理各参与方具有异构可行集时的个性化问题。我们研究了一种约束个性化联邦学习问题，该问题为每个参与方分配一个各自不同的可行模型，同时通过协作目标将各模型耦合起来。我们提出了局部惩罚交叉估计联邦平均方法，其中每个参与方维护一个包含自身模型以及其他参与方模型估计的多块向量，同时仅在本地施加可行性惩罚……

    arXiv:2603.19617v2 Announce Type: replace  Abstract: Federated learning (FL) is a communication-efficient framework for solving distributed optimization problems. Standard FL methods aggregate local updates into a single global model that is shared by all agents, effectively imposing consensus. Under heterogeneous local constraints, however, a common feasible model may be overly restrictive or may not exist. Existing constrained FL methods largely retain this shared-model structure and therefore do not directly address personalization under heterogeneous agent-specific feasible sets. We study a constrained personalized FL problem that assigns a distinct feasible model to each agent while coupling the models through a collaborative objective. We propose Locally Penalized Cross-Estimate Federated Averaging (PCE-FedAvg), where each agent maintains a multi-block vector containing its own model and estimates of the other agents' models while applying the feasibility penalty only locally to 
    
[^414]: 通过先验知识引导的图学习探索异构脑网络中的子网络交互

    Exploring Subnetwork Interactions in Heterogeneous Brain Network via Prior-Informed Graph Learning

    [https://arxiv.org/abs/2603.19307](https://arxiv.org/abs/2603.19307)

    提出KD-Brain框架，通过语义条件化交互机制和病理一致性约束将语义与临床先验知识注入图学习过程，有效解决了小样本条件下脑功能子网络交互建模难题，实现精神障碍诊断的最先进性能。

    

    建模功能子网络之间的复杂交互对于精神障碍的诊断和功能通路的识别至关重要。然而，由于训练样本数量有限，现有基于Transformer的方法在学习潜在子网络交互时仍面临重大挑战。为解决这些问题，我们提出了KD-Brain，一个先验知识引导的图学习框架，通过显式编码先验知识来指导学习过程。具体而言，我们设计了一种语义条件化交互机制，将语义先验注入注意力查询中，基于子网络的功能身份显式引导其交互学习。此外，我们引入了一种病理一致性约束，通过将学习到的交互分布与临床先验对齐来规范模型优化。此外，KD-Brain达到了最先进的性能。

    arXiv:2603.19307v2 Announce Type: replace-cross  Abstract: Modeling the complex interactions among functional subnetworks is crucial for the diagnosis of mental disorders and the identification of functional pathways. However, learning the interactions of the underlying subnetworks remains a significant challenge for existing Transformer-based methods due to the limited number of training samples. To address these challenges, we propose KD-Brain, a Prior-Informed Graph Learning framework for explicitly encoding prior knowledge to guide the learning process. Specifically, we design a Semantic-Conditioned Interaction mechanism that injects semantic priors into the attention query, explicitly navigating the subnetwork interactions based on their functional identities. Furthermore, we introduce a Pathology-Consistent Constraint, which regularizes the model optimization by aligning the learned interaction distributions with clinical priors. Additionally, KD-Brain leads to state-of-the-art p
    
[^415]: 话到嘴边：为什么大语言模型会幻觉出它们本可解码出的答案

    On the Tip of the Tongue: Why LLMs Hallucinate Answers They Can Decode

    [https://arxiv.org/abs/2603.13911](https://arxiv.org/abs/2603.13911)

    该论文提出在首个答案标记处区分“读取”与“写出”的新框架，揭示大语言模型产生幻觉的关键原因并非正确答案无法从中间状态解码，而是最终读出时的“选择边际”不足，使更强的竞争标记压制了正确答案。

    

    即使在正确答案能够从其中间状态解码出来的情况下，语言模型也可能给出错误的答案。为了研究这种可解码性与选择之间的差距，我们在第一个答案标记处区分了“读取”与“写出”。“读取”问的是：在相同关系诱饵控制的条件下，正确标记能否从中间残差状态中被解码出来；“写出”问的是：最终的读出是否将该标记排在所有内容标记的首位。在三种不同的读取器下，并采用随机标签控制实验，仍有相当大比例的失败案例保持“可读取”状态，同时另一个内容标记被选中。我们通过最终读出处的选择边际来解释这一现象：选择边际是答案logit与其最强竞争者logit之间的差值，即答案支持度减去竞争者支持度，并且可以进一步分解为与标记频率相关的上下文平均基线和一个项目特定的部分。将答案支持度设置为……（原文摘要在此处截断）

    arXiv:2603.13911v2 Announce Type: replace  Abstract: A language model can give the wrong answer even when the correct answer is decodable from its intermediate states. To study this gap between decodability and selection, we distinguish \textit{read} from \textit{write} at the first answer token. Read asks whether the gold token can be decoded from intermediate residual states under same-relation decoy controls. Write asks whether the final readout ranks that token first among content tokens. Under three different readers, with a randomized-label control, a substantial fraction of failures remain readable while another content token is selected. We explain this through the selection margin at the final readout, the difference between the answer logit and the logit of its strongest alternative, which is answer support minus alternative support, and can also be split into a context-averaged baseline linked to token frequency and an item-specific term. Setting the answer support to the le
    
[^416]: AI能理解折纸的语言吗？

    Can AI Understand the Language of Origami?

    [https://arxiv.org/abs/2603.13856](https://arxiv.org/abs/2603.13856)

    该论文提出OrigamiBench基准，通过物理接地的高层折叠动作语言评估AI对折纸合成机制的程序化理解能力，同时整合了视觉感知、几何物理约束推理和顺序规划等多项能力。

    

    构建能够在物理世界中规划、行动和创造的AI系统，所需的不仅仅是模式识别。这样的系统必须能够对支配物理过程的生成机制和约束进行推理，并使用连接观察、动作及其效果的结构化表示。然而，许多现有的基准测试将这些能力分开研究，要么专注于视觉识别，要么专注于抽象符号或程序化推理。折纸提供了一个整合这些能力的天然测试平台：通过折叠构建形状需要视觉感知、对几何和物理约束的推理以及顺序规划，同时保持足够的结构化以便进行系统性评估。我们提出了OrigamiBench，这是一个通过物理接地的高层折叠动作语言来评估折纸合成机制程序化理解的基准测试。（注：原文摘要在此处被截断）

    arXiv:2603.13856v3 Announce Type: replace  Abstract: Building AI systems that can plan, act, and create in the physical world requires more than pattern recognition. Such systems must reason about the generative mechanisms and constraints governing physical processes, using structured representations that connect observations, actions, and their effects. Yet, many existing benchmarks study these capabilities separately, focusing either on visual recognition or on abstract symbolic or programmatic reasoning. Origami provides a natural testbed that integrates these abilities: constructing shapes through folds requires visual perception, reasoning about geometric and physical constraints, and sequential planning, while remaining sufficiently structured for systematic evaluation. We introduce OrigamiBench, a benchmark for evaluating programmatic understanding of the mechanisms underlying origami synthesis through a high-level language of physically grounded fold actions. Experiments with m
    
[^417]: 深度ReLU网络鲁棒性的理论下界

    Theoretical Lower Bounds on the Robustness of Deep ReLU Networks

    [https://arxiv.org/abs/2602.18674](https://arxiv.org/abs/2602.18674)

    该论文通过结合高维几何中的测度集中性与一项新的几何刻画——即ReLU网络诱导的输入空间划分中每个凸多面体区域的面数至多等于网络单元数——证明了深度ReLU网络对随机L₂扰动的局部鲁棒性理论下界，并分析了该鲁棒性随输入维度的缩放规律。

    

    我们对参数化神经网络在随机输入扰动下的鲁棒性进行了理论研究。具体而言，我们通过量化给定输入受到随机L₂扰动后仍被正确分类的概率来分析局部鲁棒性。对于深度ReLU网络，我们结合高维几何工具（尤其是测度集中性），以及对ReLU网络输入输出函数所诱导几何结构的新刻画，推导出了局部鲁棒性的下界。我们证明，在ReLU网络所诱导的输入空间划分中，每个凸多面体区域的面数至多等于网络单元的数量，且这一结论与网络的深度或架构无关。这一几何性质是我们鲁棒性分析的关键要素。最后，我们分析了局部鲁棒性如何随输入维度变化，并刻画了那些邻域最为鲁棒的输入集合。

    arXiv:2602.18674v2 Announce Type: replace  Abstract: We present a theoretical study of the robustness of parameterized neural networks to random input perturbations. Specifically, we analyze local robustness by quantifying the probability that a random L_2-perturbation of a given input results in a correct classification. For deep ReLU networks, we derive lower bounds on local robustness by combining tools from high-dimensional geometry, in particular concentration of measure, with a new characterization of the geometric structure induced by their input-output functions. We prove that each convex polyhedral region in the partition of the input space induced by a ReLU network has at most as many faces as there are network units, regardless of the network depth or architecture. This geometric property serves as the key ingredient in our robustness analysis. Finally, we analyze how local robustness scales with input dimension and characterize the sets of inputs whose neighborhoods are mos
    
[^418]: 面向机器人系统的黑盒安全强化学习控制器的示范引导观测攻击

    Demonstration-Guided Observation Attacks on Black-Box Safe Reinforcement Learning Controllers for Robotic Systems

    [https://arxiv.org/abs/2602.16543](https://arxiv.org/abs/2602.16543)

    提出一种仅需示范数据作为受害方信息的示范引导观测攻击框架，通过逆约束强化学习恢复状态约束与代理策略并学习动力学，无需访问受害模型参数、梯度或查询即可生成有界观测扰动，有效攻击黑盒安全强化学习机器人控制器并诱导安全违规。

    

    安全强化学习通过在安全约束下优化任务奖励来学习机器人控制器，然而观测扰动可能引发安全违规。现有的面向安全的攻击通常需要访问受害网络、梯度、评价器或显式规范——而一旦控制器以黑盒形式部署，这些假设往往难以满足。我们提出一种示范引导的观测攻击方法，用于分析未知的Safe RL控制器。该框架通过逆约束强化学习恢复状态约束和代理策略，并从示范转移中学习系统动力学。两者组合的梯度可在无需受害模型参数、梯度或查询的情况下生成有界的观测扰动；示范数据是唯一需要的受害方特定信息。在四个Bullet任务和一个MetaDrive地图上，对每个受害控制器使用三种扰动预算，该攻击在15个设置中的12个超越了所有同访问权限的基线方法。

    arXiv:2602.16543v2 Announce Type: replace  Abstract: Safe reinforcement learning (Safe RL) learns robotic controllers that optimize task rewards under safety constraints, yet observation perturbations can induce safety violations. Existing safety-directed attacks often require access to victim networks, gradients, critics, or explicit specifications -- assumptions rarely met once a controller is deployed as a black box. We propose a demonstration-guided observation attack for analyzing unknown Safe RL controllers. The framework recovers a state constraint and a surrogate policy through inverse constrained reinforcement learning, and learns dynamics from demonstration transitions. Their composed gradient generates bounded observation perturbations without victim parameters, gradients, or queries; demonstrations are the only victim-specific information. Across four Bullet tasks and one MetaDrive map with three budgets per victim, the attack exceeds every same-access baseline in 12 of 15 
    
[^419]: Goldilocks 强化学习：通过调节任务难度摆脱稀疏奖励，提升语言模型推理能力

    Goldilocks RL: Tuning Task Difficulty to Escape Sparse Rewards for Reasoning

    [https://arxiv.org/abs/2602.14868](https://arxiv.org/abs/2602.14868)

    提出Goldilocks自适应数据选择策略，利用选择器网络预测问题的奖励波动性，优先选取难度适中（既不太简单也不太难）的训练问题，从而摆脱稀疏奖励困境，提升语言模型推理强化学习的样本效率。

    

    强化学习已成为解锁语言模型推理能力的强大范式。然而，依赖稀疏奖励使得这一过程样本效率极低，因为模型必须在极少反馈的情况下探索庞大的搜索空间。虽然经典的课程学习旨在通过按复杂度对数据排序来缓解这一问题，但先前的工作主要针对小型数据集，无法直接迁移到现代语言模型训练的大规模场景。此外，针对特定模型的合适排序往往并不明确。为解决这一问题，我们提出了Goldilocks，一种自适应数据选择策略，它使用一个选择器网络来预测每个候选问题在模型多次 rollout 中奖励的标准差。选择器会优先选取预测奖励波动性高的问题，这类问题对模型而言既不太简单也不太难，恰到好处。

    arXiv:2602.14868v3 Announce Type: replace-cross  Abstract: Reinforcement learning has emerged as a powerful paradigm for unlocking reasoning capabilities in language models. However, relying on sparse rewards makes this process highly sample-inefficient, as models must navigate vast search spaces with minimal feedback. While classic curriculum learning aims to mitigate this by ordering data based on complexity, prior works have primarily targeted small datasets and do not directly transfer to the large-scale settings typical of modern language model training. Furthermore, the right ordering for a specific model is often unclear. To address this, we propose Goldilocks, an adaptive data-selection strategy that uses a Selector network to predict the standard deviation of rewards across the model's rollouts for each candidate question. The Selector prioritizes questions with high predicted reward variability, corresponding to questions that are neither too easy nor too hard for the model's
    
[^420]: Power-SMC：用于免训练大语言模型推理的低延迟序列级幂采样方法

    Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning

    [https://arxiv.org/abs/2602.10273](https://arxiv.org/abs/2602.10273)

    Power-SMC是一种免训练的低延迟序列级幂采样方法，以接近标准解码的速度实现分布锐化，从而提升大语言模型的推理能力。

    

    大语言模型的推理能力通常被归因于“分布锐化”，即将输出概率集中在高似然序列上。最近的研究表明，这种锐化效应可以在推理阶段获得，而无需修改模型参数，并且能够激发强大的推理性能。一个自然的数学形式化是“序列级幂分布”，它与模型概率的α次方（α>1）成正比。先前的研究利用Metropolis-Hastings（MH）采样从该分布中抽取样本并取得了出色的效果，然而代价是数量级级别的推理减速。我们提出了Power-SMC，这是一种免训练的采样方法，它以接近标准解码的延迟来针对相同的幂分布。Power-SMC并行维护多个候选序列，每个候选序列被赋予一个分数，即……

    arXiv:2602.10273v3 Announce Type: replace-cross  Abstract: Reasoning ability in large language models is often attributed to \emph{distribution sharpening}: concentrating output probability on high-likelihood sequences. Recent works show that this sharpening effect can be obtained at inference time, without modifying model parameters, and can elicit strong reasoning performance. A natural formalization is the \emph{sequence-level power distribution}, which is proportional to the model's probability raised to an exponent $\alpha>1$. Prior work leveraged Metropolis--Hastings (MH) sampling to draw samples from this distribution and achieves strong results, however, at order-of-magnitude inference slowdowns. We introduce \textbf{Power-SMC}, a \textit{`training-free'} sampling method that targets the same power distribution yielding close to standard decoding latency. Power-SMC maintains multiple candidate sequences in parallel. Each candidate sequence is assigned a score, namely the \emph{
    
[^421]: ANCRe：面向高效深度扩展的自适应神经连接重分配

    ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling

    [https://arxiv.org/abs/2602.09009](https://arxiv.org/abs/2602.09009)

    该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。

    

    摘要（arXiv:2602.09009v2，公告类型：replace-cross）：扩展网络深度一直是现代基础模型成功的核心驱动力，然而近期研究表明，深层网络往往未被充分利用。本文从优化视角重新审视了加深神经网络的默认机制——残差连接。严格的分析证明，残差连接的布局能够从根本上塑造收敛行为，甚至会引发收敛速率上的指数级差距。受此启发，我们提出了自适应神经连接重分配，这是一个具有理论依据且轻量级的框架，能够从数据中参数化并学习残差连接方式。ANCRe 以可忽略不计的计算和内存开销（<1%）自适应地重新分配残差连接，同时使网络深度得到更有效的利用。我们在大型语言模型预训练、扩散模型以及深度R（原文此处截断）等任务上进行了大量数值测试……

    arXiv:2602.09009v2 Announce Type: replace-cross  Abstract: Scaling network depth has been a central driver behind the success of modern foundation models, yet recent investigations suggest that deep layers are often underutilized. This paper revisits the default mechanism for deepening neural networks, namely residual connections, from an optimization perspective. Rigorous analysis proves that the layout of residual connections can fundamentally shape convergence behavior, and even induces an exponential gap in convergence rates. Prompted by this insight, we introduce adaptive neural connection reassignment (ANCRe), a principled and lightweight framework that parameterizes and learns residual connectivities from the data. ANCRe adaptively reassigns residual connections with negligible computational and memory overhead ($<1\%$), while enabling more effective utilization of network depth. Extensive numerical tests across pre-training of large language models, diffusion models, and deep R
    
[^422]: ChronoSpike：一种面向动态图的自适应脉冲图神经网络

    ChronoSpike: An Adaptive Spiking Graph Neural Network for Dynamic Graphs

    [https://arxiv.org/abs/2602.01124](https://arxiv.org/abs/2602.01124)

    ChronoSpike提出一种自适应脉冲图神经网络，通过融合逐通道膜动力学的可学习LIF神经元、多头空间注意力聚合与轻量级Transformer时间编码器，在低内存开销下实现了动态图上细粒度的局部建模与长程时间依赖捕获。

    

    动态图表示学习需要同时捕获结构关系与时间演化，然而现有方法面临一个核心权衡：基于注意力的方法以O(T²)的复杂度换取表达能力，而循环架构则受困于梯度病态问题和密集状态存储。脉冲神经网络提供了事件驱动的计算效率，但受限于顺序传播、二值信息损失以及缺乏全局上下文的局部聚合。我们提出ChronoSpike，一种自适应脉冲图神经网络，它集成了具有逐通道膜电位动力学的可学习LIF神经元、基于连续特征的多头空间注意力聚合机制，以及轻量级Transformer时间编码器。这一设计实现了细粒度的局部建模与长程依赖捕获，仅需O(T·d)的激活/状态内存，外加一个随时间跨度保持较小的额外O(T²)逐节点注意力项。

    arXiv:2602.01124v4 Announce Type: replace  Abstract: Dynamic graph representation learning requires capturing both structural relations and temporal evolution, yet existing approaches face a core trade-off: attention-based methods offer expressiveness at O(T^2) complexity, while recurrent architectures suffer from gradient pathologies and dense state storage. Spiking neural networks provide event-driven efficiency but are constrained by sequential propagation, binary information loss, and local aggregation that lacks global context. We propose ChronoSpike, an adaptive spiking graph neural network that integrates learnable LIF neurons with per-channel membrane dynamics, multi-head spatially-attentive aggregation over continuous features, and a lightweight Transformer temporal encoder. This design enables fine-grained local modeling and long-range dependency capture with O(T.d) activation/state memory and an additional O(T^2) per-node attention term that remains small for the horizons ev
    
[^423]: 可恢复性存在定律：面向工具增强智能体的ERR度量

    Recoverability Has a Law: The ERR Measure for Tool-Augmented Agents

    [https://arxiv.org/abs/2601.22352](https://arxiv.org/abs/2601.22352)

    本文提出期望恢复遗憾（ERR）指标并证明其与可观测的效率得分（ES）之间存在一阶定量关系，首次为工具增强语言模型智能体失败后的自我恢复能力建立了可证伪的预测性定律，并在五个工具使用基准上得到实证验证。

    

    语言模型智能体在工具调用执行失败后常常表现出自我恢复的能力，然而这一行为一直缺乏形式化的解释。我们提出了一个预测性理论来填补这一空白，证明可恢复性遵循一条可测量的定律。具体而言，我们通过期望恢复遗憾Expected Recovery Regret, ERR）来形式化可恢复性，该指标量化了在随机执行噪声下恢复策略相对于最优策略的偏离程度，并推导出ERR与一个经验可观测量——效率得分Efficiency Score, ES）之间的一阶关系。由此得到了一个可证伪的、关于工具使用智能体恢复动力学的一阶定量定律。我们在五个工具使用基准上对该定律进行了实证验证，涵盖受控扰动、诊断推理和真实世界API等场景。在不同模型规模、扰动机制和恢复时间跨度下，ERR-ES定律所预测的遗憾值与观测到的失败后恢复行为高度吻合。

    arXiv:2601.22352v2 Announce Type: replace-cross  Abstract: Language model agents often appear capable of self-recovery after failing tool call executions, yet this behavior lacks a formal explanation. We present a predictive theory that resolves this gap by showing that recoverability follows a measurable law. To elaborate, we formalize recoverability through Expected Recovery Regret (ERR), which quantifies the deviation of a recovery policy from the optimal one under stochastic execution noise, and derive a first-order relationship between ERR and an empirical observable quantity, the Efficiency Score (ES). This yields a falsifiable first-order quantitative law of recovery dynamics in tool-using agents. We empirically validate the law across five tool-use benchmarks spanning controlled perturbations, diagnostic reasoning, and real-world APIs. Across model scales, perturbation regimes, and recovery horizons, predicted regret under the ERR-ES law closely matched observed post-failure re
    
[^424]: 道德是情境化的：利用概率聚类与大语言模型从人类数据中学习可解释的道德情境

    Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models

    [https://arxiv.org/abs/2512.21439](https://arxiv.org/abs/2512.21439)

    提出了COMETH框架，将概率情境学习与大语言模型语义抽象及人类道德判断数据相结合，从数据中学习可解释的道德情境，证明道德评价是高度情境化的。

    

    当前AI对齐研究中的一个关键问题是如何让AI算法学习道德价值观。由于人类道德高度依赖情境，对行为的评判不仅取决于其结果，还取决于行为发生的情境。我们提出了COMETH（基于文本人类输入的道德评估情境组织），这是一个将概率情境学习器与基于大语言模型的语义抽象及人类道德评估相结合的框架，用于建模情境如何塑造模糊行为的可接受性。我们构建了一个基于实证的数据集，包含与三条道德规则（违反“不可杀人”、“不可欺骗”和“不可违法”）相关的六种核心行为共300个场景，并收集了101名参与者的三元判断（谴责/中立/支持）。预处理流程通过大语言模型过滤器与结合K-means聚类的MiniLM嵌入对行为进行标准化，产生稳健且可复现的核心行为聚类。

    arXiv:2512.21439v2 Announce Type: replace-cross  Abstract: A key question in current AI alignment research is how to make AI algorithms learn moral values. Because human morality is highly context-dependent, actions are judged not only by their outcomes but by the context in which they occur. We present COMETH (Contextual Organization of Moral Evaluation from Textual Human inputs), a framework that integrates a probabilistic context learner with LLM-based semantic abstraction and human moral evaluations to model how context shapes the acceptability of ambiguous actions. We curate an empirically grounded dataset of 300 scenarios across six core actions relative to three moral rules (violating "Do not kill", "Do not deceive", and "Do not break the law") and collect ternary judgments (Blame/Neutral/Support) from N=101 participants. A preprocessing pipeline standardizes actions via an LLM filter and MiniLM embeddings with K-means, producing robust, reproducible core-action clusters. COMETH
    
[^425]: 使用深度学习可解释性图谱揭示与晕屏症不适持续相关的脑电模式

    Uncovering EEG Patterns Consistently Associated with Cybersickness Discomfort Using Deep Learning Interpretability Maps

    [https://arxiv.org/abs/2512.20620](https://arxiv.org/abs/2512.20620)

    该研究构建了一个结合神经网络与可解释性图谱的框架，基于两项独立的ERP晕屏症用户研究，成功识别出对分类晕屏症相关不适最关键的时空脑电特征，从而揭示了与晕屏症不适稳定一致的EEG模式。

    

    arXiv:2512.20620v3 公告类型：replace-cross 摘要：在使用虚拟现实（VR）头戴式显示器时，可能会产生类似晕动症的不适感，称为晕屏症。晕屏症阻碍了VR技术的更广泛应用。利用脑电图（EEG）记录的大脑活动可以无创地检测晕屏症及其引起的不适。为了进行干预以缓解症状，需要能够从其余脑部数据中提取与晕屏症不适相关的有意义信号的机器学习算法。在这项工作中，我们构建了一个结合神经网络与可解释性图谱的框架，以确定EEG数据中哪些特征对分类晕屏症相关不适最有帮助。利用来自两项独立的听觉事件相关电位（ERP）晕屏症用户研究的大脑数据，我们提取了对不适分类最重要的时空EEG特征（即传感器位置和时间步长层面的特征）。

    arXiv:2512.20620v3 Announce Type: replace-cross  Abstract: Uncomfortable sensations similar to motion sickness, called cybersickness, can develop when using Virtual Reality (VR) head-mounted displays. Cybersickness poses a hindrance to greater use of VR technology. Brain activity recorded using electroencephalogram (EEG) can be used to unintrusively detect cybersickness and the discomfort it causes. To intervene for mitigation, machine learning algorithms that can extract meaningful signals related to cybersickness discomfort from the rest of the brain data will be required. In this work, we determined which features in EEG data were most helpful for classifying cybersickness-related discomfort by building a framework with neural networks and interpretability maps. Using brain data from two separate auditory event-related potential (ERP) cybersickness user studies, we extracted which spatio-temporal EEG features (from sensor locations and time steps) were most important for discomfort 
    
[^426]: 揭开LLM-as-a-Judge（大语言模型作为评判者）的神秘面纱：面向推理时扩展的解析可处理模型

    Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling

    [https://arxiv.org/abs/2512.19905](https://arxiv.org/abs/2512.19905)

    该论文提出了一个解析可处理的推理时扩展模型——带奖励加权采样器的贝叶斯线性回归，用以模拟LLM作为评判者的场景，并在高维机制下推导出后验预测均值与方差的闭式表达式，从而揭示推理时扩展背后的数学原理。

    

    大语言模型的最新发展表明，将相当一部分计算资源从训练阶段重新分配到推理阶段具有优势。然而，推理时扩展背后的原理尚未得到充分理解。在本文中，我们引入了一个解析可处理的推理时扩展模型：带有奖励加权采样器的贝叶斯线性回归，其中奖励由线性模型确定，以模拟LLM-as-a-judge（大语言模型作为评判者）场景。我们在高维机制下研究这一问题，借助确定性等价方法得到了后验预测均值和方差的闭式表达式。我们分析了训练数据从教师模型采样时的泛化误差。我们抽取k个推理时样本，并通过在二次奖励上施加温度参数的softmax进行选择。当奖励与教师模型差异不大时，泛化误差单调递减（摘要在此处截断）。

    arXiv:2512.19905v3 Announce Type: replace-cross  Abstract: Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonical
    
[^427]: 无界域中采样策略与物理信息Kolmogorov–Arnold网络的评估

    Evaluation of Sampling Strategies and Physics-Informed Kolmogorov--Arnold Networks in Unbounded Domains

    [https://arxiv.org/abs/2512.12074](https://arxiv.org/abs/2512.12074)

    该论文通过基准测试系统评估了无界域反演PDE问题中不同采样策略（均匀、高斯、指数分布）与网络架构（MLP与KAN）的表现，为物理信息神经网络在无限和半无限域上的应用提供了定量的精度与效率参考。

    

    物理信息神经网络（PINNs）通过将物理定律融入学习过程，已成为求解偏微分方程（PDEs）的一种有效方法。然而，其在无限域和半无限域上的应用仍然具有挑战性，原因在于难以用有限数量的训练点来表示无界区域。本工作评估了无界域中反演PDE问题的物理信息学习策略，重点研究了基于均匀分布、高斯分布和指数分布的采样策略的影响，以及在PINN框架内使用传统多层感知机（MLPs）和Kolmogorov–Arnold网络（KANs）的效果。所提出的基准测试考虑了无限域和半无限域上人工构造的反演问题，从而能够对重建精度和计算效率进行定量评估。结果表明，高斯采样和指数采样……

    arXiv:2512.12074v2 Announce Type: replace  Abstract: Physics-informed neural networks (PINNs) have emerged as an effective approach for solving partial differential equations (PDEs) by incorporating physical laws into the learning process. However, their application to infinite and semi-infinite domains remains challenging due to the difficulty of representing unbounded regions with a finite number of training points. This work evaluates physics-informed learning strategies for inverse PDE problems in unbounded domains, focusing on the influence of sampling strategies based on uniform, Gaussian, and exponential distributions, and on the use of conventional multilayer perceptrons (MLPs) and Kolmogorov--Arnold Networks (KANs) within the PINN framework. The proposed benchmark considers manufactured inverse problems on infinite and semi-infinite domains, enabling quantitative assessment of reconstruction accuracy and computational efficiency. Results show that Gaussian and exponential samp
    
[^428]: 弥合世界模型中基于梯度规划的训练-测试差距

    Closing the Train-Test Gap in World Models for Gradient-Based Planning

    [https://arxiv.org/abs/2512.09929](https://arxiv.org/abs/2512.09929)

    本文通过提出训练时数据合成技术来弥合世界模型“以下一状态预测为训练目标”与“测试时用于估计动作序列”之间的训练-测试差距，从而显著提升基于梯度规划的性能，使其成为高效且性能优异的规划方法。

    

    世界模型与模型预测控制（MPC）相结合，可以在大规模专家轨迹数据集上进行离线训练，并在推理时泛化到广泛的规划任务。与依赖缓慢搜索算法或精确迭代求解优化问题的传统MPC流程相比，基于梯度的规划提供了一种计算上更高效的替代方案。然而，迄今为止，基于梯度的规划的性能一直落后于其他方法。在本文中，我们提出了改进的世界模型训练方法，以实现高效的基于梯度的规划。我们的出发点在于一个观察：尽管世界模型是以下一状态预测为训练目标的，但在测试时它却被用于估计动作序列。我们工作的目标正是弥合这一训练-测试差距。为此，我们提出了训练时数据合成技术，能够显著……（摘要在此处截断）

    arXiv:2512.09929v2 Announce Type: replace  Abstract: World models paired with model predictive control (MPC) can be trained offline on large-scale datasets of expert trajectories and enable generalization to a wide range of planning tasks at inference time. Compared to traditional MPC procedures, which rely on slow search algorithms or on iteratively solving optimization problems exactly, gradient-based planning offers a computationally efficient alternative. However, the performance of gradient-based planning has thus far lagged behind that of other approaches. In this paper, we propose improved methods for training world models that enable efficient gradient-based planning. We begin with the observation that although a world model is trained on a next-state prediction objective, it is used at test-time to instead estimate a sequence of actions. The goal of our work is to close this train-test gap. To that end, we propose train-time data synthesis techniques that enable significantly 
    
[^429]: 碎片化可被量子神经网络高效学习

    Fragmentation is Efficiently Learnable by Quantum Neural Networks

    [https://arxiv.org/abs/2512.00751](https://arxiv.org/abs/2512.00751)

    该论文证明当量子系统的碎片化满足特定条件时，碎片分类问题可被量子神经网络高效解决，而已知经典去量子化技术对此失效，为物理动机的量子机器学习任务提供量子优势提供了一个罕见范例。

    

    在某些物理量子系统中，指数级庞大的态空间会“碎片化”为许多低维的、动力学不连通的子空间。我们引入了一个被称为碎片分类的学习问题：给定一个量子态作为输入，目标是分类该态属于哪个子空间。我们证明，当碎片化现象满足特定条件时，在量子计算机上求解这一学习问题是高效的。此外，我们通过展示已知的去量子化技术在碎片分类问题上失效，为该任务的经典计算困难性提供了证据。因此，这项工作提供了一个罕见的例子——一个具有物理动机的量子机器学习任务，它对量子计算机而言是高效可解的，同时不存在已知的经典去量子化方法。

    arXiv:2512.00751v4 Announce Type: replace-cross  Abstract: In certain classes of physical quantum systems, the exponentially large state space "fragments" into many low-dimensional, dynamically disconnected subspaces. We introduce a learning problem known as fragment classification, where given a quantum state input, one is interested in classifying to which subspace the state belongs. We prove that solving this learning problem is efficient on a quantum computer when the fragmentation phenomenon satisfies certain conditions. Furthermore, we give evidence supporting the classical hardness of this task by demonstrating that known dequantization techniques fail for the fragment classification problem. Consequently, this work provides a rare example of a physically motivated quantum machine learning task that is both efficient for quantum computers to perform and admits no known classical dequantization.
    
[^430]: 何时池化才有回报？间歇性需求预测中遗忘机制下的可信度与分辨率

    When Does Pooling Pay? Credibility and Resolution under Forgetting in Intermittent-Demand Forecasting

    [https://arxiv.org/abs/2511.12749](https://arxiv.org/abs/2511.12749)

    该论文证明，在分层经验贝叶斯间歇性需求预测模型中，“遗忘序列自身历史”与“跨序列信息共享”可统一为同一个决策，并提出拟合窗口诊断、可信度界、分辨条件及事前筛选准则，用以判断何时池化信息才有价值。

    

    对大量稀疏时间序列进行预测需要做出两个选择：每个序列自身的历史应保留多少，以及应从其他序列借鉴多少信息。我们证明，当同一个指数近期性算子同时作用于分层经验贝叶斯 hurdle 模型中的项目级与组级统计量时，这两个选择便合而为一：遗忘既保持了共享先验的杠杆作用，又抬高了噪声底线，使得精细的跨序列结构更难被分辨。我们将这种双向效应形式化为拟合窗口诊断方法、可信度界以及候选池的分辨条件，并据此推导出一个事前筛选准则，用于识别精细化无法带来收益的情形。在由超过 19,000 条序列组成的五个公开间歇性需求数据集上——即项目级证据最稀疏的场景——该诊断在两个方向上均判断正确。在截断会降低可信度的情形下，共享先验是值得的：关闭它将带来 5% 和 12% 的成本损失（原文摘要此处截断）。

    arXiv:2511.12749v3 Announce Type: replace-cross  Abstract: Forecasting many sparse series requires two choices: how much of each series' own past to retain, and how much to borrow from other series. We show that when one exponential recency operator is applied to item- and group-level statistics alike in a hierarchical empirical-Bayes hurdle model, the two become one decision: forgetting preserves the leverage of a shared prior while raising the noise floor against which fine cross-series structure must be resolved. We formalize this two-sided effect as a fitting-window diagnostic, a credibility bound and a resolution condition for candidate pools, and derive from it an ex ante screen for regimes where refinement cannot pay. On five public intermittent-demand panels comprising over 19,000 series, where item-level evidence is sparsest, the diagnostic is right in both directions. Where truncation lowers credibility the shared prior pays: switching it off costs $5\%$ and $12\%$ at the sho
    
[^431]: 差分隐私主成分分析的高维渐近性

    High-Dimensional Asymptotics of Differentially Private PCA

    [https://arxiv.org/abs/2511.07270](https://arxiv.org/abs/2511.07270)

    该论文针对差分隐私主成分分析，通过分析指数机制，在高维设置下给出了隐私损失随数据集变化的精确渐近刻画，弥补了传统一致上界在特定数据集上过于保守的不足。

    

    在差分隐私中，通过引入随机噪声对敏感数据集的汇总统计量进行私有化处理后再发布。噪声水平决定了隐私损失，即量化攻击者利用已发布的统计量检测某一目标个体是否存在于数据集中的难易程度。大多数隐私分析给出的是对所有数据集一致成立的非渐近隐私损失上界。有时，这些上界在特定数据集上可能过于悲观。在这种情况下，用精确的隐私刻画来补充这些隐私上界会很有用，因为这种刻画能够量化某个机制在给定数据集上的确切隐私损失。基于这一目标，我们研究了差分隐私主成分分析（PCA），其目标是对包含 $n$ 个样本和 $p$ 个特征的数据集的主成分进行私有化。我们分析了指数机制，并为其隐私损失提供了精确的渐近刻画。

    arXiv:2511.07270v4 Announce Type: replace-cross  Abstract: In differential privacy, random noise is introduced to privatize summary statistics of a sensitive dataset before releasing them. The noise level determines the privacy loss, which quantifies how easily an adversary can detect a target individual's presence in the dataset using the published statistic. Most privacy analyses provide non-asymptotic upper bounds on the privacy loss which hold uniformly across all datasets. Sometimes, these bounds can be pessimistic on a given dataset. In such cases, it can be useful to complement these privacy bounds with sharp privacy characterizations that quantify a mechanism's exact privacy loss on a given dataset. With this goal, we study differentially private principal component analysis (PCA), where the goal is to privatize the leading principal components of a dataset with $n$ samples and $p$ features. We analyze the exponential mechanism and provide sharp asymptotic characterizations of 
    
[^432]: 基于主动学习的随机场双保真度Karhunen-Loève展开代理模型

    Bifidelity Karhunen-Lo\`eve Expansion Surrogate with Active Learning for Random Fields

    [https://arxiv.org/abs/2511.03756](https://arxiv.org/abs/2511.03756)

    提出了一种将Karhunen-Loève展开与多项式混沌展开相结合的双保真度代理模型，并利用基于交叉验证和高斯过程回归的主动学习策略自适应选择高保真度采样点，从而在有限计算成本下实现随机场的高精度建模。

    

    我们提出了一种用于不确定输入下场值感兴趣量的双保真度Karhunen-Loève展开（KLE）代理模型。本文考虑的感兴趣量为标量场。该方法将KLE的谱效率与多项式混沌展开（PCEs）相结合，以保持输入不确定性与输出场之间的显式映射关系。通过耦合能够捕捉主导响应趋势的低成本低保真度（LF）仿真与数量有限、用于校正系统偏差的高保真度（HF）仿真，所提出的方法能够构建精确且计算成本可控的代理模型。为了进一步提高代理模型精度，我们开发了一种主动学习策略，该策略基于代理模型的泛化误差自适应地选择新的HF评估点，泛化误差通过交叉验证进行估计，并采用高斯过程回归进行建模。随后通过最大化某种准则来获取新的HF样本。

    arXiv:2511.03756v2 Announce Type: replace-cross  Abstract: We present a bifidelity Karhunen--Lo\`{e}ve expansion (KLE) surrogate model for field-valued quantities of interest (QoIs) under uncertain inputs. The QoIs considered here are scalar fields. The approach combines the spectral efficiency of the KLE with polynomial chaos expansions (PCEs) to preserve an explicit mapping between input uncertainties and output fields. By coupling inexpensive low-fidelity (LF) simulations that capture dominant response trends with a limited number of high-fidelity (HF) simulations that correct for systematic bias, the proposed method can enable accurate and computationally affordable surrogate construction. To further improve surrogate accuracy, we develop an active learning strategy that adaptively selects new HF evaluations based on the surrogate's generalization error, estimated via cross-validation and modeled using Gaussian process regression. New HF samples are then acquired by maximizing an e
    
[^433]: 从非结构化数据中进行可解释的发现：一种高维方法

    Interpretable Discovery from Unstructured Data: A High-Dimensional Approach

    [https://arxiv.org/abs/2511.01680](https://arxiv.org/abs/2511.01680)

    该论文提出了一个从非结构化数据（如开放式调查文本）中自动进行可解释发现的通用框架，其核心创新在于结合AI可解释性方法与高维多重检验算法，将非结构化数据转化为可解释的概念测量并进行统计检验，最终生成人类可理解的发现描述。

    

    我们提出了一个自动化的、通用目的的框架，用于从非结构化数据中进行发现（例如来自经济信念开放式调查的文本数据）。该框架利用AI可解释性文献中的最新方法，将非结构化数据集转换为包含可解释概念测量的高维结构化数据集；基于转换后的数据集指定概念层面的参数和原假设；使用经过高维多重检验新成果验证的算法来检验这些假设，从而产生一个选定的集合（“发现”）；并且生成和评估这些发现的人类可理解的自然语言描述。所提出的框架研究者自由度较少，对数据窥探具有稳健性，能够缓解探索不足的问题，并有助于快速且低成本的敏感性分析和复制研究。我们重新审视了最近描述性分析应用……

    arXiv:2511.01680v5 Announce Type: replace-cross  Abstract: We propose an automatic, general-purpose framework for making discoveries from unstructured data (e.g., text data from open-ended surveys of economic beliefs). The framework leverages recent methods from the literature on AI interpretability to transform unstructured datasets into high-dimensional, structured datasets of interpretable concept measurements; specifies concept-level parameters and null hypotheses based on this transformed dataset; tests these hypotheses using algorithms validated by new results in high-dimensional multiple testing, producing a selected set ("discoveries"); and both generates and evaluates human-interpretable natural language descriptions of these discoveries. The proposed framework has few researcher degrees of freedom, is robust to data snooping, mitigates under-exploration, and facilitates fast and inexpensive sensitivity analysis and replication. We revisit applications to recent descriptive an
    
[^434]: 差分隐私作为额外收益：基于多天线基站的多址接入衰落信道上的联邦学习

    Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station

    [https://arxiv.org/abs/2510.23463](https://arxiv.org/abs/2510.23463)

    该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。

    

    联邦学习（FL）是一种分布式学习范式，它通过在训练过程中无需交换原始数据来保护隐私。在其典型的边缘部署实例中，无线传输由模拟空中计算（AirComp）实现，称为空中联邦学习（AirFL），其中固有的信道噪声扮演着一种独特的“亦敌亦友”的角色：一方面，它因带噪的全局聚合而降低训练性能；另一方面，它为隐私保护机制提供了天然的随机性来源，而这类隐私保护可以通过差分隐私（DP）进行形式化量化。然而，有效利用这种信道损伤仍然具有挑战性，因为现有工作大多在简单信道模型或受限损失函数类型的假设下，仅考虑单轮或非收敛隐私损失界限下的（本地）差分隐私增强。在本文中，我们研究了多址接入衰落信道上的空中联邦学习

    arXiv:2510.23463v4 Announce Type: replace  Abstract: Federated Learning (FL) is a distributed learning paradigm that preserves privacy by eliminating the need to exchange raw data during training. In its prototypical edge instantiation with underlying wireless transmissions enabled by analog over-the-air computing (AirComp), referred to as \emph{over-the-air FL (AirFL)}, the inherent channel noise plays a unique role of \emph{frenemy} in the sense that it degrades training due to noisy global aggregation while providing a natural source of randomness for privacy-preserving mechanisms, formally quantified by \emph{differential privacy (DP)}. It remains, nevertheless, challenging to effectively harness such channel impairments, as prior arts, under assumptions of either simple channel models or restricted types of loss functions, mostly considering (local) DP enhancement with a single-round or non-convergent bound on privacy loss. In this paper, we study AirFL over multiple-access fading
    
[^435]: 玻尔兹曼机学习中Ising与QUBO变量编码的性能评估

    Performance Evaluation of Ising and QUBO Variable Encodings in Boltzmann Machine Learning

    [https://arxiv.org/abs/2510.13210](https://arxiv.org/abs/2510.13210)

    该研究利用费舍尔信息矩阵的谱分析揭示，QUBO编码比Ising编码具有更强的病态性，导致SGD收敛更慢，而全FIM自然梯度下降可消除这种编码间的差异，因此基于SGD的训练更宜采用Ising编码。

    

    我们在受控协议下比较了玻尔兹曼机学习中Ising（{-1, +1}）与QUBO（{0, 1}）两种变量编码，这些协议在每次比较中固定了采样器、优化器和学习率设计。利用费舍尔信息矩阵（FIM）等于充分统计量协方差这一恒等式，我们对来自模型样本的经验矩进行可视化，揭示了依赖于表示形式的系统性差异。QUBO编码在一阶与二阶统计量之间诱导出更大的交叉项，在FIM中产生更多小特征值方向，从而降低谱熵。这种病态条件解释了随机梯度下降（SGD）下收敛较慢的现象。相比之下，通过FIM度量对更新进行重新缩放的全FIM自然梯度下降（NGD）在不同编码间实现了相近的收敛效果，而对角FIM近似则可能重新引入依赖于表示的差异。在实际应用中，对于基于SGD的训练，Ising编码……（原文此处截断）

    arXiv:2510.13210v2 Announce Type: replace  Abstract: We compare Ising ({-1, +1}) and QUBO ({0, 1}) encodings for Boltzmann machine learning under controlled protocols that fix the sampler, optimizer, and learning-rate design within each comparison. Exploiting the identity that the Fisher information matrix (FIM) equals the covariance of sufficient statistics, we visualize empirical moments from model samples and reveal systematic, representation-dependent differences. QUBO induces larger cross terms between first- and second-order statistics, creating more small-eigenvalue directions in the FIM and lowering spectral entropy. This ill-conditioning explains slower convergence under stochastic gradient descent (SGD). In contrast, full-FIM natural gradient descent (NGD), which rescales updates by the FIM metric, achieves similar convergence across encodings, whereas diagonal-FIM approximation can reintroduce representation-dependent differences. Practically, for SGD-based training, the Isi
    
[^436]: Cocoon：一种使用相关噪声进行差分隐私训练的系统架构

    Cocoon: A System Architecture for Differentially Private Training with Correlated Noises

    [https://arxiv.org/abs/2510.07304](https://arxiv.org/abs/2510.07304)

    Cocoon 提出了一种系统架构，通过在 CPU、GPU 和内存扩展模块之间分布式地存储与处理庞大的相关噪声历史，并对稀疏嵌入表进行优化，实现了高效的大规模模型差分隐私训练。

    

    机器学习（ML）模型会记忆并泄露训练数据，给数据所有者带来严重的隐私问题。采用差分隐私（DP）的训练算法作为解决方案正受到越来越多的关注。然而，这些算法在每次训练迭代中都会添加噪声，从而降低模型精度，限制了其在现实世界中的实际应用。为了提升精度，一类新的方法会添加经过精心设计的相关噪声，使噪声在各次迭代之间相互抵消。我们对这些新机制进行了广泛的特性分析研究，结果表明，当模型相对较大或使用相对于硬件容量而言较大的嵌入表时，这些机制会产生不可忽视的开销。基于这一分析，我们提出了 Cocoon，一个用于高效相关噪声训练的框架。Cocoon 在 CPU、GPU 和内存扩展模块之间存储和处理庞大的噪声历史记录，并针对稀疏嵌入表引入了优化措施。

    arXiv:2510.07304v2 Announce Type: replace-cross  Abstract: Machine learning (ML) models memorize and leak training data, causing serious privacy issues to data owners. Training algorithms with differential privacy (DP) have been gaining attention as a solution. However, these algorithms add noise at each training iteration and degrade accuracy, limiting their real-world adoption. To improve accuracy, a new family of approaches adds carefully designed correlated noises, so that noises cancel out each other across iterations. We performed an extensive characterization study of these new mechanisms and show they incur non-negligible overheads when the model is relatively large or uses large embedding tables compared to the hardware capacity. Motivated by the analysis, we propose Cocoon, a framework for efficient training with correlated noises. Cocoon stores and processes the large noise history across CPU, GPU, and memory extension module, introduces optimizations for sparse embedding ta
    
[^437]: 动态规划中的误差传播：从随机控制到美式期权定价

    Error Propagation in Dynamic Programming: From Stochastic Control to American Option Pricing

    [https://arxiv.org/abs/2509.20239](https://arxiv.org/abs/2509.20239)

    本文为离散时间随机最优控制建立了结合再生核希尔伯特空间回归与蒙特卡洛抽样的动态规划近似框架，提出自然的误差分解并严格分析了误差从到期日向初始时刻反向传播的规律，可应用于美式期权定价。

    

    本文研究离散时间随机最优控制（SOC）的理论与方法基础。我们首先在一个一般的动态规划框架下表述控制问题，并引入进行详细收敛性分析所需的数学结构。相关的价值函数通过结合非参数回归方法与蒙特卡洛子抽样的序列近似来估计。回归步骤在再生核希尔伯特空间（RKHS）中进行，利用经典的核岭回归（KRR）算法，同时引入蒙特卡洛抽样方法来估计续值（continuation value）。为评估价值函数估计器的精度，我们提出了一种自然的误差分解方法，并严格控制在每个时间步产生的误差项。随后我们分析了该误差如何随时间反向传播——从到期日到初始时刻——这是一个相对较少被探索的方面。

    arXiv:2509.20239v2 Announce Type: replace-cross  Abstract: This paper investigates theoretical and methodological foundations for stochastic optimal control (SOC) in discrete time. We start formulating the control problem in a general dynamic programming framework, introducing the mathematical structure needed for a detailed convergence analysis. The associate value function is estimated through a sequence of approximations combining nonparametric regression methods and Monte Carlo subsampling. The regression step is performed within reproducing kernel Hilbert spaces (RKHSs), exploiting the classical KRR algorithm, while Monte Carlo sampling methods are introduced to estimate the continuation value. To assess the accuracy of our value function estimator, we propose a natural error decomposition and rigorously control the resulting error terms at each time step. We then analyze how this error propagates backward in time-from maturity to the initial stage-a relatively underexplored aspec
    
[^438]: 使用神经网络集成从连续测量数据进行带不确定性量化的量子参数估计

    Quantum parameter estimation with uncertainty quantification from continuous measurement data using neural network ensembles

    [https://arxiv.org/abs/2509.10756](https://arxiv.org/abs/2509.10756)

    该论文提出使用深度神经网络集成进行量子参数估计，在保持估计精度的同时量化不确定性并能检测实验数据漂移，推理速度远快于贝叶斯推断方法。

    

    我们证明了深度神经网络集成（称为深度集成）可用于执行量子参数估计，同时提供量化参数估计不确定性的方法，而这正是贝叶斯推断用于参数估计的关键优势——该优势在使用现有机器学习方法时会丢失。我们表明，同时针对准确的参数估计和良好校准的不确定性估计进行优化，并不会导致参数估计精度的下降。我们还表明，这些集成模型的漂移检测能力可用于检测推理过程中所用实验数据的漂移。该方法还被证明能提供比基于似然和无似然的贝叶斯推断快得多的推理速度。这些结果表明，此类模型能够实现具有量化不确定性的准确实时参数估计。

    arXiv:2509.10756v4 Announce Type: replace-cross  Abstract: We show that ensembles of deep neural networks, called deep ensembles, can be used to perform quantum parameter estimation while also providing a means for quantifying uncertainty in parameter estimates, which is a key advantage of using Bayesian inference for parameter estimation that is lost when using existing machine learning methods. We show that optimizing for both accurate parameter estimates and well calibrated uncertainty estimates does not lead to degradation in the former as opposed to only optimizing for accuracy. We also show that the drift detection capabilities of these ensemble models can be used to detect drift in the experimental data used during inference. This approach is also shown to provide much faster inference time than both likelihood-based and likelihood-free Bayesian inference. These results suggest that such models could enable accurate, real-time parameter estimation with quantified uncertainty, ma
    
[^439]: EEGDM：基于潜在扩散模型的脑电表征学习

    EEGDM: Learning EEG Representation with Latent Diffusion Model

    [https://arxiv.org/abs/2508.20705](https://arxiv.org/abs/2508.20705)

    EEGDM提出了一种基于潜在扩散模型的自监督学习框架，通过生成式去噪过程学习脑电信号的全局时间模式与跨通道关系的紧凑表征，克服了掩码重建方法难以捕捉全局生成约束的局限。

    

    近年来，脑电（EEG）表征的自监督学习研究主要依赖于掩码重建方法，即训练模型恢复被随机掩蔽的信号片段。尽管掩码重建在建模局部依赖方面卓有成效，但其训练目标并不能促使模型捕捉刻画神经活动所必需的全局生成约束。为解决这一局限，我们提出了EEGDM，一种利用潜在扩散模型生成EEG信号作为训练目标的新型自监督框架。与掩码重建不同，基于扩散的生成过程将信号从噪声逐步去噪至真实形态，迫使模型捕捉整体时间模式和跨通道关系。具体而言，EEGDM引入了一个EEG编码器，将原始信号及其通道增强提取为紧凑的表征，该表征作为条件信息来引导扩散模型的生成过程。

    arXiv:2508.20705v4 Announce Type: replace-cross  Abstract: Recent advances in self-supervised learning for EEG representation have largely relied on masked reconstruction, where models are trained to recover randomly masked signal segments. While effective at modeling local dependencies, the training objective of masked reconstruction does not compel the model to capture global generative constraints essential for characterizing neural activity. To address this limitation, we propose EEGDM, a novel self-supervised framework that leverages latent diffusion models to generate EEG signals as an objective. Unlike masked reconstruction, diffusion-based generation progressively denoises signals from noise to realism, compelling the model to capture holistic temporal patterns and cross-channel relationships. Specifically, EEGDM incorporates an EEG encoder that distills raw signals and their channel augmentations into a compact representation, which serves as conditional information to guide t
    
[^440]: 基于语义关系条件化的多模态表示学习

    Multimodal Representation Learning Conditioned on Semantic Relations

    [https://arxiv.org/abs/2508.17497](https://arxiv.org/abs/2508.17497)

    提出了关系条件化多模态学习框架RCML，将自然语言描述的语义关系作为显式条件来学习多模态表示，使同一样本在不同关系下拥有不同表示，克服了CLIP等对比模型单一嵌入的局限。

    

    多模态表示学习的发展主要由CLIP等对比模型推动，这类模型通过对齐配对的图像-文本样本学习共享嵌入空间。尽管这类模型在通用表示学习方面十分有效，但它们通常为每个样本生成单一嵌入，并在不同的语义关系和上下文中重复使用该嵌入。然而，在许多实际应用中，样本之间的相关性本质上是依赖于关系的，不同的语义关系会强调多模态数据的不同方面。在本工作中，我们提出了关系条件化多模态学习（RCML）框架，该框架将语义关系视为多模态表示学习的显式条件。RCML不再生成与关系无关的嵌入，而是基于自然语言关系描述学习条件化表示，使同一样本能够在不同关系条件下获得不同的表示。

    arXiv:2508.17497v3 Announce Type: replace-cross  Abstract: Multimodal representation learning has been largely driven by contrastive models such as CLIP, which learn a shared embedding space by aligning paired image-text samples. While effective for general-purpose representation learning, such models typically produce a single embedding per sample that is reused across different semantic relations and contexts. However, in many real-world applications, relevance between samples is inherently relation-dependent, with different semantic relations emphasizing different aspects of multimodal data.   In this work, we propose Relation-Conditioned Multimodal Learning (RCML), a framework that treats semantic relations as explicit conditions of multimodal representation learning. Rather than producing relation-agnostic embeddings, RCML learns representations conditioned on natural-language relation descriptions, allowing the same sample to be represented differently under different relational 
    
[^441]: 一种用于手写字符识别的Sobel梯度多层感知机基线模型

    A Sobel-Gradient MLP Baseline for Handwritten Character Recognition

    [https://arxiv.org/abs/2508.11902](https://arxiv.org/abs/2508.11902)

    本研究提出一种基于固定Sobel梯度算子与简单多层感知机的受控基线模型，证明仅使用一阶图像梯度特征即可在MNIST和EMNIST Letters手写字符识别上分别取得98.54%和92.50%的高准确率。

    

    本研究考察一种刻意设计的一阶边缘表示能够保留多少手写字符信息。该方法不学习空间滤波器，而是将每个输入图像通过固定的Sobel-Feldman算子转换为带符号的水平与垂直导数图，这些导数图经过独立归一化、展平后由多层感知机（MLP）进行分类。因此，该模型将固定的边缘提取与可学习的分类过程分离，为评估一阶图像梯度的充分性提供了一个受控基线。在已执行的实验中，Sobel梯度MLP在MNIST上达到98.54%的测试准确率，在TensorFlow Datasets（TFDS）的EMNIST Letters配置上达到92.50%的测试准确率，宏平均F1分数分别为0.9853和0.9265。一对一剩余（one-vs-rest）ROC分析进一步得出，MNIST上的微观/宏观AUC值为0.9998/0.9998，EMNIST Letters上为0.9987/0.9982。混淆矩...（原文摘要在此处截断）

    arXiv:2508.11902v4 Announce Type: replace-cross  Abstract: This study examines how much handwritten-character information is retained by a deliberately simple first-order edge representation. Instead of learning spatial filters, each input image is transformed by the fixed Sobel-Feldman operator into signed horizontal and vertical derivative maps, which are independently normalized, flattened, and classified by a multilayer perceptron (MLP). The resulting model therefore separates fixed edge extraction from learned classification and provides a controlled baseline for evaluating the sufficiency of first-order image gradients. In the executed experiments, the Sobel-gradient MLP achieves 98.54 percent test accuracy on MNIST and 92.50 percent on the TensorFlow Datasets (TFDS) EMNIST Letters configuration. Macro F1 scores are 0.9853 and 0.9265, respectively. One-vs-rest ROC analysis further yields micro/macro AUC values of 0.9998/0.9998 on MNIST and 0.9987/0.9982 on EMNIST Letters. Confusi
    
[^442]: SGD能否应对重尾噪声？

    Can SGD Handle Heavy-Tailed Noise?

    [https://arxiv.org/abs/2508.04860](https://arxiv.org/abs/2508.04860)

    本文证明仅假设随机梯度具有有界p阶矩，原始SGD即可在凸、强凸和非凸问题上于重尾噪声下达到极小极大最优的收敛保证与样本复杂度。

    

    随机梯度下降（SGD）是大规模优化的基石，但其在重尾噪声下的理论行为——这种噪声在现代机器学习和强化学习中十分常见——仍然缺乏充分理解。在本工作中，我们严格研究了不含任何自适应修改的原始SGD，在此类不利随机条件下能否被证明可以成功。仅假设随机梯度对于某个 $p \in (1, 2]$ 具有有界的 $p$ 阶矩，我们为（投影）SGD在凸、强凸和非凸问题类别上建立了精确的收敛保证。特别地，我们证明SGD在最小假设下，于凸和强凸情形分别达到了极小极大最优的样本复杂度：$\mathcal{O}(\varepsilon^{-\frac{p}{p-1}})$ 和 $\mathcal{O}(\varepsilon^{-\frac{p}{2(p-1)}})$。对于非凸目标函数，在标准光滑性和有界中心 $p$（阶矩）条件下……

    arXiv:2508.04860v2 Announce Type: replace-cross  Abstract: Stochastic Gradient Descent (SGD) is a cornerstone of large-scale optimization, yet its theoretical behavior under heavy-tailed noise -- common in modern machine learning and reinforcement learning -- remains poorly understood. In this work, we rigorously investigate whether vanilla SGD, devoid of any adaptive modifications, can provably succeed under such adverse stochastic conditions. Assuming only that stochastic gradients have bounded $p$-th moments for some $p \in (1, 2]$, we establish sharp convergence guarantees for (projected) SGD across convex, strongly convex, and non-convex problem classes. In particular, we show that SGD achieves minimax optimal sample complexity under minimal assumptions in the convex and strongly convex regimes: $\mathcal{O}(\varepsilon^{-\frac{p}{p-1}})$ and $\mathcal{O}(\varepsilon^{-\frac{p}{2(p-1)}})$, respectively. For non-convex objectives, under standard smoothness and a bounded central $p$
    
[^443]: 通过随机化密钥选择缓解生成模型中的水印伪造

    Mitigating Watermark Forgery in Generative Models via Randomized Key Selection

    [https://arxiv.org/abs/2507.07871](https://arxiv.org/abs/2507.07871)

    该论文提出通过对每次查询随机化水印密钥选择的防御方案，使盲攻击者的伪造成功率存在与所收集样本数量无关的上限，且不进一步降低模型效用，从而有效缓解生成模型中的水印伪造攻击。

    

    水印技术使生成式AI提供商能够验证内容是否由其模型生成。水印是内容中的一种隐藏信号，可以使用秘密水印密钥来检测其存在。一个核心安全威胁是伪造攻击，即对手将提供商的水印插入到并非由该提供商生成的内容中，这可能损害其声誉并破坏用户信任。现有的防御方法通过向同一内容中嵌入使用多个密钥的多个水印来抵抗伪造，但这可能会降低模型效用。然而，当攻击者能够收集足够多的带水印样本时，伪造仍然是一种威胁。我们提出了一种防御方法，对于盲攻击者，在密钥对称且检测器结果独立的条件下，其伪造成功概率存在一个与样本数量无关的上限。我们的方案不会进一步降低模型效用。我们对每个查询随机化水印密钥的选择，并据此接受内容是否由模型生成。

    arXiv:2507.07871v5 Announce Type: replace-cross  Abstract: Watermarking enables GenAI providers to verify whether content was generated by their models. A watermark is a hidden signal in the content, whose presence can be detected using a secret watermark key. A core security threat are forgery attacks, where adversaries insert the provider's watermark into content \emph{not} produced by the provider, potentially damaging their reputation and undermining trust. Existing defenses resist forgery by embedding many watermarks with multiple keys into the same content, which can degrade model utility. However, forgery remains a threat when attackers can collect sufficiently many watermarked samples. We propose a defense with a sample-count-independent upper bound on forgery success for blind attackers, conditional on key-symmetric, independent detector outcomes. Our scheme does not further degrade model utility. We randomize the watermark key selection for each query and accept content as ge
    
[^444]: 精确且准确地估计流行率

    Estimating prevalence with precision and accuracy

    [https://arxiv.org/abs/2507.06061](https://arxiv.org/abs/2507.06061)

    本文提出了一种贝叶斯聚合量化器PQ，它在保证足够覆盖率的同时生成更窄的预测区间，从而比现有方法更精确地估计流行率并更有效地量化估计的不确定性。

    

    与分类（其目标是估计每个数据点的类别）不同，量化（或称流行率估计）旨在估计数据集中各类别的分布情况。流行率估计中的一项重要任务是对流行率估计的不确定性进行量化。在本文中，我们提出了精确量化器，这是一种贝叶斯聚合量化器，能够在保证足够覆盖率（即预测区间包含真实流行率的比例足够高）的同时，获得狭窄的预测区间。我们发现，随着底层分类器判别能力的增强以及验证集与测试集大小比例的提高，PQ能够产生比现有方法更精确的流行率估计。这些实证结果表明，与现有方法相比，PQ能更有效地利用验证信息来量化流行率估计中的不确定性。

    arXiv:2507.06061v2 Announce Type: replace-cross  Abstract: Unlike classification, whose goal is to estimate the class of each data point, quantification (or prevalence estimation) aims to estimate the distribution of classes in a dataset. An important task in prevalence estimation is to quantify the uncertainty in prevalence estimates. In this paper, we introduce Precise Quantifier (PQ), a Bayesian aggregative quantifier that achieves narrow prediction intervals with sufficient coverage (i.e., sufficient proportion of intervals containing the true prevalence). We find that PQ produces more precise prevalence estimates than existing methods as the discriminative power of the underlying classifier increases and as the validation-to-test size ratio increases. These empirical results suggest that PQ uses validation information more effectively to quantify uncertainty in prevalence estimates than existing approaches.
    
[^445]: 固态电池中耦合反应与扩散主导的界面演化

    Coupled reaction and diffusion governing interface evolution in solid-state batteries

    [https://arxiv.org/abs/2506.10944](https://arxiv.org/abs/2506.10944)

    通过主动学习与深度等变神经网络原子间势实现量子精度的大规模固态电池界面反应模拟，并结合基于局部原子环境聚类的无监督分类方法，首次发现SEI中一种此前未被报道的晶体无序相Li₂S₀.₇₂P₀.₁₄Cl₀.₁₄。

    

    理解并控制决定固态电解质界面相（SEI）形成的原子级反应，对下一代固态电池的可行性至关重要。然而，由于实验上难以表征埋藏界面，以及模拟速度和精度的限制，该领域仍面临诸多挑战。我们借助主动学习与深度等变神经网络原子间势，对一个对称电池单元开展了具有量子精度的大规模显式反应模拟。为了自动表征界面处的耦合反应与互扩散过程，我们提出并使用了基于局部原子环境空间聚类的无监督分类技术。我们的分析揭示了SEI中一种此前未被报道的晶体无序相Li₂S₀.₇₂P₀.₁₄Cl₀.₁₄的形成，该相在此前基于...的预测中未被捕获。

    arXiv:2506.10944v2 Announce Type: replace-cross  Abstract: Understanding and controlling the atomistic-level reactions governing the formation of the solid-electrolyte interphase (SEI) is crucial for the viability of next-generation solid state batteries. However, challenges persist due to difficulties in experimentally characterizing buried interfaces and limits in simulation speed and accuracy. We conduct large-scale explicit reactive simulations with quantum accuracy for a symmetric battery cell, {\symcell}, enabled by active learning and deep equivariant neural network interatomic potentials. To automatically characterize the coupled reactions and interdiffusion at the interface, we formulate and use unsupervised classification techniques based on clustering in the space of local atomic environments. Our analysis reveals the formation of a previously unreported crystalline disordered phase, Li$_2$S$_{0.72}$P$_{0.14}$Cl$_{0.14}$, in the SEI, that evaded previous predictions based pu
    
[^446]: 松弛随机控制问题的连续策略与值迭代及其收敛性

    Continuous Policy and Value Iteration for Relaxed Stochastic Control Problems and Its Convergence

    [https://arxiv.org/abs/2506.08121](https://arxiv.org/abs/2506.08121)

    本文提出一种基于朗之万型动力学的连续策略-值迭代算法，能同时更新值函数与最优控制，并在哈密顿量单调性条件下证明了该算法在无限时间域熵正则化松弛控制问题中收敛于最优控制。

    

    我们提出了一种连续的策略-值迭代算法，其中随机控制问题的值函数近似与最优控制通过朗之万型动力学同时更新。本文聚焦于无限时间域的熵正则化松弛控制问题。我们建立了策略改进性质，并在哈密顿量的单调性条件下证明了对最优控制的收敛性。通过利用朗之万型随机微分方程沿策略迭代方向进行连续更新，我们的方法能够使用机器学习中的分布采样技术，同时优化值函数并识别最优控制。文中给出了具有非凹奖励的LQ模型以及具有偏斜拉普拉斯最优控制的非LQ示例的数值实验。

    arXiv:2506.08121v3 Announce Type: replace-cross  Abstract: We introduce a continuous policy-value iteration algorithm where the approximations of the value function of a stochastic control problem and the optimal control are simultaneously updated through Langevin-type dynamics. This paper focuses on the entropy-regularized relaxed control problem with infinite horizon. We establish policy improvement and demonstrate convergence to the optimal control under the monotonicity condition of the Hamiltonian. By utilizing Langevin-type stochastic differential equations for continuous updates along the policy iteration direction, our approach enables the use of distribution sampling techniques in machine learning to optimize the value function and identify the optimal control simultaneously. Numerical experiments are presented for LQ model with a non-concave reward, and a non-LQ example with a skewed-Laplace optimal control.
    
[^447]: Infinity搜索：基于q-度量空间投影的近似向量搜索

    Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces

    [https://arxiv.org/abs/2506.06557](https://arxiv.org/abs/2506.06557)

    该论文提出将任意相异度函数投影到超度量空间并学习该投影的近似形式，在保持最近邻关系的同时实现最坏情况复杂度仅为树深度的近似向量搜索，并将该方法推广到更一般的q-度量空间。

    

    超度量空间（或称无穷度量空间）由一个相异度函数定义，该函数满足强三角不等式：三角形中任意一边不大于另外两边中的较大者。我们证明，在超度量空间中使用视点树进行搜索的最坏情况复杂度等于树的深度。由于所关注的数据集通常并非超度量，我们采用一种投影算子，将任意相异度函数变换到超度量空间中，同时保持最近邻关系不变。我们进一步学习该投影算子的近似形式，以高效计算查询点与数据集中点之间的超度量距离。随后，我们解决了一个更一般的问题，即考虑q-度量空间中的投影——其中三角形各边的q次幂小于另外两边的q次幂之和。注意到使用学习到的…

    arXiv:2506.06557v3 Announce Type: replace-cross  Abstract: An ultrametric space or infinity-metric space is defined by a dissimilarity function that satisfies a strong triangle inequality in which every side of a triangle is not larger than the larger of the other two. We show that search in ultrametric spaces with a vantage point tree has worst-case complexity equal to the depth of the tree. Since datasets of interest are not ultrametric in general, we employ a projection operator that transforms an arbitrary dissimilarity function into an ultrametric space while preserving nearest neighbors. We further learn an approximation of this projection operator to efficiently compute ultrametric distances between query points and points in the dataset. We proceed to solve a more general problem in which we consider projections in $q$-metric spaces -- in which triangle sides raised to the power of $q$ are smaller than the sum of the $q$-powers of the other two. Notice that the use of learned a
    
[^448]: 基于冲突感知证据深度学习的鲁棒对抗量化

    Robust Adversarial Quantification via Conflict-Aware Evidential Deep Learning

    [https://arxiv.org/abs/2506.05937](https://arxiv.org/abs/2506.05937)

    提出轻量级后验不确定性量化方法 C-EDL，通过为输入生成多样的任务保持变换并量化表示分歧来校准不确定性，无需重新训练即可增强证据深度学习对对抗性和分布外输入的鲁棒性。

    

    深度学习模型的可靠性对于其在高风险应用中的部署至关重要，因为在这些应用中，分布外输入或对抗性输入可能导致严重的不良后果。证据深度学习是一种高效的不确定性量化范式，它将预测建模为单次前向传播所得到的狄利克雷分布。然而，EDL 对对抗性扰动的输入尤为脆弱，容易产生过度自信的错误。冲突感知证据深度学习（C-EDL）是一种轻量级的后验不确定性量化方法，能够缓解上述问题，在无需重新训练的情况下增强对抗鲁棒性和分布外鲁棒性。C-EDL 为每个输入生成多样的、保持任务特性的变换，并量化表示层面的分歧，以便在需要时校准不确定性估计。C-EDL 的冲突感知预测调整提高了对分布外样本和对抗性样本的检测能力，同时保持较高的分布内准确率和较低的（摘要在此处截断）

    arXiv:2506.05937v3 Announce Type: replace-cross  Abstract: Reliability of deep learning models is critical for deployment in high-stakes applications, where out-of-distribution or adversarial inputs may lead to detrimental outcomes. Evidential Deep Learning, an efficient paradigm for uncertainty quantification, models predictions as Dirichlet distributions of a single forward pass. However, EDL is particularly vulnerable to adversarially perturbed inputs, making overconfident errors. Conflict-aware Evidential Deep Learning~\mbox{(C-EDL)} is a lightweight post-hoc uncertainty quantification approach that mitigates these issues, enhancing adversarial and OOD robustness without retraining. C-EDL generates diverse, task-preserving transformations per input and quantifies representational disagreement to calibrate uncertainty estimates when needed. C-EDL's conflict-aware prediction adjustment improves detection of OOD and adversarial inputs, maintaining high in-distribution accuracy and low
    
[^449]: VTBench：评估用于自回归图像生成的视觉分词器

    VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation

    [https://arxiv.org/abs/2505.13439](https://arxiv.org/abs/2505.13439)

    VTBench是一个系统性评估自回归图像生成中视觉分词器性能的综合基准，通过图像重建、细节保留和文本保留三大核心任务，揭示了离散视觉分词器与连续VAE之间的性能差距。

    

    自回归（AR）模型最近在图像生成中展现出强大的性能，其中一个关键组件是将连续像素输入映射为离散token序列的视觉分词器（VT）。视觉分词器的质量在很大程度上决定了AR模型性能的上限。然而，目前的离散视觉分词器明显落后于连续变分自编码器（VAE），导致图像重建质量下降，细节和文本的保留效果不佳。现有的基准测试专注于端到端的生成质量，而未能单独评估视觉分词器的性能。为了填补这一空白，我们提出了VTBench，这是一个综合性基准，通过三大核心任务系统地评估视觉分词器：图像重建、细节保留和文本保留，并涵盖多样化的评估场景。我们使用一组指标系统地评估了最先进的视觉分词器，以衡量重建图像的质量。

    arXiv:2505.13439v2 Announce Type: replace-cross  Abstract: Autoregressive (AR) models have recently shown strong performance in image generation, where a critical component is the visual tokenizer (VT) that maps continuous pixel inputs to discrete token sequences. The quality of the VT largely defines the upper bound of AR model performance. However, current discrete VTs fall significantly behind continuous variational autoencoders (VAEs), leading to degraded image reconstructions and poor preservation of details and text. Existing benchmarks focus on end-to-end generation quality, without isolating VT performance. To address this gap, we introduce VTBench, a comprehensive benchmark that systematically evaluates VTs across three core tasks: Image Reconstruction, Detail Preservation, and Text Preservation, and covers a diverse range of evaluation scenarios. We systematically assess state-of-the-art VTs using a set of metrics to evaluate the quality of reconstructed images. Our findings 
    
[^450]: 时变贝叶斯优化的渐近性能

    Asymptotic Performance of Time-Varying Bayesian Optimization

    [https://arxiv.org/abs/2505.13012](https://arxiv.org/abs/2505.13012)

    本文首次为时变贝叶斯优化（TVBO）算法的累积遗憾提供了上界和与算法无关的下界，推导出算法具有无悔性质的充分条件，且其分析首次覆盖了实践中使用的所有主要类别的平稳核函数。

    

    时变贝叶斯优化（TVBO）是优化可能带有噪声且评估代价高昂的时变黑盒目标函数的首选框架，但其卓越的实证性能至今仍缺乏理论上的解释。TVBO算法的瞬时遗憾是否有可能渐近消失？如果可以，何时会消失？我们通过为TVBO算法的累积遗憾提供上界和与算法无关的下界来回答这一重要问题。在此过程中，我们对TVBO框架提供了重要见解，并推导出TVBO算法具有无悔性质的充分条件。据我们所知，我们的分析是首个覆盖实践中使用的所有主要平稳核函数类别的研究。

    arXiv:2505.13012v3 Announce Type: replace-cross  Abstract: Time-Varying Bayesian Optimization (TVBO) is the go-to framework for optimizing a time-varying black-box objective function that may be noisy and expensive to evaluate, but its excellent empirical performance remains to be understood theoretically. Is it possible for the instantaneous regret of a TVBO algorithm to vanish asymptotically, and if so, when? We answer this question of great importance by providing upper bounds and algorithm-independent lower bounds for the cumulative regret of TVBO algorithms. In doing so, we provide important insights about the TVBO framework and derive sufficient conditions for a TVBO algorithm to have the no-regret property. To the best of our knowledge, our analysis is the first to cover all major classes of stationary kernel functions used in practice.
    
[^451]: Minty条件下变分不等式的多项式时间算法

    A Polynomial-Time Algorithm for Variational Inequalities under the Minty Condition

    [https://arxiv.org/abs/2504.03432](https://arxiv.org/abs/2504.03432)

    本文提出了首个在Minty条件下求解Lipschitz连续映射的ε-变分不等式的多项式时间算法（复杂度关于维度 $d$ 和 $\log(1/\epsilon)$ 多项式增长），突破了以往方法对 $1/\epsilon$ 的指数级依赖或需单调性等更强假设的限制。

    

    求解（Stampacchia）变分不等式（SVIs）是优化领域核心中的一个基础性问题。然而，这种表达能力是以计算困难性为代价的。因此，大多数研究都聚焦于划分出能够避开这些难解性障碍的特定子类。一个可追溯至20世纪60年代的经典性质是Minty条件，该条件假设Minty变分不等式（MVI）问题存在解。在本文中，我们建立了首个多项式时间算法——其复杂度关于维度 $d$ 和 $\log(1/\epsilon)$ 多项式增长——用于在Minty条件下求解Lipschitz连续映射的 $\epsilon$-SVIs。先前的方法要么对 $1/\epsilon$（以及问题的其他自然参数）存在指数级更差的依赖，要么做出了更严格的假设（如单调性）。为此，我们引入了一种椭球算法的新变体，借此……（原文摘要至此截断）

    arXiv:2504.03432v4 Announce Type: replace-cross  Abstract: Solving (Stampacchia) variational inequalities (SVIs) is a foundational problem at the heart of optimization. However, this expressivity comes at the cost of computational hardness. As a result, most research has focused on carving out specific subclasses that elude those intractability barriers. A classical property that goes back to the 1960s is the Minty condition, which postulates that the Minty VI (MVI) problem admits a solution.   In this paper, we establish the first polynomial-time algorithm -- with complexity growing polynomially in the dimension $d$ and $\log(1/\epsilon)$ -- for solving $\epsilon$-SVIs for Lipschitz continuous mappings under the Minty condition. Prior approaches either incurred an exponentially worse dependence on $1/\epsilon$ (and other natural parameters of the problem) or made more restrictive assumptions, such as monotonicity. To do so, we introduce a new variant of the ellipsoid algorithm whereby
    
[^452]: 分层函数的噪声敏感性与学习下界

    Noise Sensitivity and Learning Lower Bounds for Hierarchical Functions

    [https://arxiv.org/abs/2502.05073](https://arxiv.org/abs/2502.05073)

    本文证明了树状层次结构函数在每层函数均与线性函数保持 ε-距离时，其噪声稳定性随层次深度呈指数级衰减，并由此推导出基于分层函数的函数类在不可知学习中的统计查询超多项式下界。

    

    近期的研究工作通过考察具有层次结构的函数或数据来探索深度学习取得成功的原因。为了研究具有层次结构的函数的学习复杂度，我们研究了定义在独立输入上的、具有树状层次结构的函数的噪声稳定性。我们证明：若层次结构中的每一层函数都与线性函数存在 ε-距离（即明显非线性），则该函数的噪声稳定性会随层次深度呈指数级衰减。我们的结果在不可知学习中具有直接应用：在布尔设置下，结合 Dachman-Soled、Feldman、Tan、Wan 和 Wimmer（2014）的结果，我们的结果为基于分层函数的函数类在不可知学习下提供了统计查询（SQ）超多项式下界。此外，我们还基于临界位点渗流中穿越事件的指示函数推导出类似的 SQ 下界。这些穿越事件虽然并不严格符合我们所定义的层次结构，但仍具有某种层次特性……

    arXiv:2502.05073v4 Announce Type: replace-cross  Abstract: Recent works explore deep learning's success by examining functions or data with hierarchical structure. To study the learning complexity of functions with hierarchical structure, we study the noise stability of functions with tree hierarchical structure on independent inputs. We show that if each function in the hierarchy is $\varepsilon$-far from linear, the noise stability is exponentially small in the depth of the hierarchy.   Our results have immediate applications for agnostic learning. In the Boolean setting using the results of Dachman-Soled, Feldman, Tan, Wan and Wimmer (2014), our results provide Statistical Query super-polynomial lower bounds for agnostically learning classes that are based on hierarchical functions.   We also derive similar SQ lower bounds based on the indicators of crossing events in critical site percolation. These crossing events are not formally hierarchical as we define but still have some hier
    
[^453]: 正面、反面与AI的失误：大语言模型、随机性与人类判断

    Heads, Tails, and AI Fails: LLMs, Randomness, and Human Judgments

    [https://arxiv.org/abs/2406.00092](https://arxiv.org/abs/2406.00092)

    该研究发现大语言模型在模拟抛硬币时会再现并放大人类的随机性偏差（如过度交替、厌恶长连续序列），提高温度参数只能部分缓解而无法消除这些系统性失真。

    

    随机性对人类认知以及大语言模型所部署的众多应用都至关重要，然而基于概率的token生成并不意味着大语言模型能够产生无偏的随机序列。我们借助模拟抛硬币这一经典行为科学范式，研究当代大语言模型如何生成二元随机序列。通过单次抛掷、20次抛掷序列、n-gram统计、连续长度、交替率以及下一次抛掷的可预测性等多个维度，我们将模型输出与真实的伯努利基线以及已有研究的人类数据进行比较。我们发现，大语言模型再现了若干经典的人类随机性偏差，包括过度交替、对长连续序列的厌恶以及首次抛掷偏差，但往往会放大这些偏差或引入模型特有的失真。提高温度参数可以减少某些僵化的模式，但无法消除系统性结构。我们进一步通过提示框架实验和续写任务深入探究了过度交替现象。

    arXiv:2406.00092v2 Announce Type: replace  Abstract: Randomness is central to human cognition and to many applications in which large language models are deployed, yet probabilistic token generation does not imply that LLMs can produce unbiased random sequences. We study how contemporary LLMs generate binary random sequences using the classic behavioral-science paradigm of simulated coin flips. Across single flips, 20-flip sequences, n-gram statistics, run lengths, alternation rates, and next-flip predictability, we compare model outputs to both true Bernoulli baselines and human data from prior work. We find that LLMs reproduce several canonical human randomness biases, including over-alternation, aversion to long runs, and first-flip biases, but often amplify them or introduce model-specific distortions. Increasing temperature reduces some rigid patterns but does not eliminate systematic structure. We further investigate over-alternation through prompt-framing experiments, continuati
    
[^454]: 词元空间：一个用于AI计算的范畴论框架

    Token Space: A Category Theory Framework for AI Computations

    [https://arxiv.org/abs/2404.11624](https://arxiv.org/abs/2404.11624)

    本文提出了词元空间这一用于AI计算的范畴论框架，并证明通过代数词元化方法可将各类有限性结构对象范畴完全忠实地嵌入其中，同时保持二元乘积和等化子，为AI计算提供了坚实的数学基础。

    

    我们引入了词元空间，这是一个用于AI计算的范畴论框架。一个词元是一个有限元组，其条目为承载集合的元素或固定核心的符号；词元类是一个集合连同此类词元的一个堆，而词元映射是保持每个词元的函数。词元空间是通过在集合范畴上添加单位集范畴、构造乘积并取子集扩张而构建的。我们证明所得到的范畴拥有所有有限极限、有限余积和指数对象，但与集合范畴Set不同，它们不是拓扑斯。随后我们引入了代数词元化方法：将结构化集合的常量、关系以及运算的图记录为以核心符号为头的词元。这使得每一个有限性结构对象范畴（如带点集合、序、图、环、向量空间）都能完全忠实地嵌入到词元空间中，并同时保持二元乘积和等化子；拓扑……

    arXiv:2404.11624v2 Announce Type: replace-cross  Abstract: We introduce the Token Space, a categorical framework for AI computations. A Token is a finite tuple whose entries are elements of a carrier set or symbols of a fixed core; a Token class is a set together with a heap of such Tokens, and Token maps are the functions preserving every Token. The Token Space is built from the category of sets by adjoining identity set categories, forming products and taking a subsets extension. We prove that the resulting categories have all finite limits, finite coproducts and exponentials, but, unlike Set, are not topoi. We then introduce algebraic tokenization: the constants, relations and graphs of operations of a structured set are recorded as Tokens headed by a core symbol. This gives a full and faithful embedding of every finitary category of structured objects (pointed sets, orders, graphs, rings, vector spaces) into the Token Space which preserves binary products and equalizers; topologica
    
[^455]: VIDiff：基于扩散模型通过多模态指令进行视频转换

    VIDiff: Translating Videos via Multi-Modal Instructions with Diffusion Models

    [https://arxiv.org/abs/2311.18837](https://arxiv.org/abs/2311.18837)

    本文首次提出了视频指令扩散基础模型VIDiff，能够根据用户的多模态指令在几秒内完成视频编辑、转换和增强等多种理解与生成任务，并通过迭代自回归方法保证长视频编辑的一致性。

    

    扩散模型在图像和视频生成方面取得了显著成功。这激发了人们对视频编辑任务日益增长的兴趣，即根据提供的文本描述对视频进行编辑。然而，大多数现有方法只关注短片段的视频编辑，并且依赖耗时的调优或推理过程。我们首次提出了视频指令扩散模型，这是一个为广泛的视频任务设计的统一基础模型。这些任务既涵盖理解任务（如语言引导的视频对象分割），也包括生成任务（视频编辑和增强）。我们的模型可以根据用户指令在几秒钟内编辑并转换出期望的结果。此外，我们设计了一种迭代自回归方法，以确保长视频编辑和增强的一致性。我们为多样化的输入视频和书面指令提供了令人信服的生成结果，无论是从定性角度还是……（摘要在此处截断）

    arXiv:2311.18837v2 Announce Type: replace-cross  Abstract: Diffusion models have achieved significant success in image and video generation. This motivates a growing interest in video editing tasks, where videos are edited according to provided text descriptions. However, most existing approaches only focus on video editing for short clips and rely on time-consuming tuning or inference. We are the first to propose Video Instruction Diffusion (VIDiff), a unified foundation model designed for a wide range of video tasks. These tasks encompass both understanding tasks (such as language-guided video object segmentation) and generative tasks (video editing and enhancement). Our model can edit and translate the desired results within seconds based on user instructions. Moreover, we design an iterative auto-regressive method to ensure consistency in editing and enhancing long videos. We provide convincing generative results for diverse input videos and written instructions, both qualitatively
    
[^456]: 分散随机双正则化非凸强凸极小极大问题的方差减少加速方法

    Variance-reduced accelerated methods for decentralized stochastic double-regularized nonconvex strongly-concave minimax problems. (arXiv:2307.07113v1 [math.OC])

    [http://arxiv.org/abs/2307.07113](http://arxiv.org/abs/2307.07113)

    本文提出了一种应用于分散随机双正则化非凸强凸极小极大问题的方差减少加速方法，通过引入拉格朗日乘子和采用单个邻居通信并结合方差减少技术，该方法在随机设置下样本复杂度达到$\mathcal{O}(\kappa^3\varepsilon^{-3})$。

    

    本文考虑在原始变量和对偶变量上具有非光滑正则化项的分散、随机、非凸强凸（NCSC）极小极大问题，在该问题中，m个计算代理通过点对点通信进行协作。我们考虑了耦合函数为期望或有限和形式的情况，并且双正则化函数分别应用于原始变量和对偶变量。我们的算法框架引入了一个拉格朗日乘子来消除对偶变量上的共识约束。将此与方差减少（VR）技术相结合，我们提出的方法，称为VRLM，通过每次迭代进行一次邻居通信，能够在一般的随机设置下实现$\mathcal{O}(\kappa^3\varepsilon^{-3})$ 的样本复杂度，其中$\kappa$是问题的条件数，$\varepsilon$是希望的解精度。通过使用大批量VR，

    In this paper, we consider the decentralized, stochastic nonconvex strongly-concave (NCSC) minimax problem with nonsmooth regularization terms on both primal and dual variables, wherein a network of $m$ computing agents collaborate via peer-to-peer communications. We consider when the coupling function is in expectation or finite-sum form and the double regularizers are convex functions, applied separately to the primal and dual variables. Our algorithmic framework introduces a Lagrangian multiplier to eliminate the consensus constraint on the dual variable. Coupling this with variance-reduction (VR) techniques, our proposed method, entitled VRLM, by a single neighbor communication per iteration, is able to achieve an $\mathcal{O}(\kappa^3\varepsilon^{-3})$ sample complexity under the general stochastic setting, with either a big-batch or small-batch VR option, where $\kappa$ is the condition number of the problem and $\varepsilon$ is the desired solution accuracy. With a big-batch VR,
    

