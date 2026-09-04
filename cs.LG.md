# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Compile by Training: Turning Natural-Language Specifications into Local Neural Functions](https://arxiv.org/abs/2609.04199) | 提出“训练式编译”方法，将自然语言规范编译为可复用的本地神经函数，通过教师模型生成的示例训练小型适配器，无需每次调用远程大模型即可达到83.6%的语义准确率。 |
| [^2] | [Clean Engineering, Unstable Measurement: A Preregistered Reliability Failure of Black-Box LLM Observers on Shared Endpoints](https://arxiv.org/abs/2609.04198) | 本文通过两项预注册审计（共52,988次请求尝试）发现，黑盒大语言模型评判者作为测量仪器存在严重可靠性缺陷——即便工程执行记录完美，相同请求的重复排名一致性仅0.400、字节级相同的次日重放一致性仅0.78，远低于0.90和0.99的预设标准，其根源在于标签映射偏置、信号低于噪声底多个数量级以及逐字排列读数放大的噪声。 |
| [^3] | [Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-Of-Thought Reasoning](https://arxiv.org/abs/2609.04194) | 本研究将思维链推理步骤的重要性量化为蒙特卡洛模拟估计的“优势”，发现LLM评判器虽能超越简单基线但远不足以准确识别真正重要的推理步骤，表明推理文本的可读性并不等于可解释性。 |
| [^4] | [Robust PAC Learning of Concurrent Stochastic Games](https://arxiv.org/abs/2609.04189) | 该论文提出了首个针对具有转移不确定性的广义和并发随机博弈的PAC学习框架，通过引入纳什裕度刻画解决了均衡存在性问题，能在多项式样本复杂度下返回社会福利近优的ε-近似纳什均衡或证明精确纳什均衡不存在。 |
| [^5] | [Para-Pipe: Exploiting Hierarchical Operator Parallelism of ML Computational Graphs on SoCs](https://arxiv.org/abs/2609.04168) | 本文提出了Para-Pipe，一个面向片上系统的分层映射框架，通过在流水线架构中集成阶段内和阶段间算子并行性，在吞吐量与推理延迟之间实现更好的权衡，从而降低边缘深度学习应用的推理延迟。 |
| [^6] | [Parameterised graph theory for tensor networks: entanglement rerouting, structural simplification, and agnostic tomography](https://arxiv.org/abs/2609.04165) | 该论文运用参数化图论证明切宽与树切宽界定了张量网络态转化为可高效处理的矩阵乘积态或树张量网络表示所需的键维开销，为张量网络的结构简化和态层析学习提供了新的理论工具。 |
| [^7] | [A Low-Cost, Open Platform for End-to-End Autonomous Driving on a Miniature Ackermann Vehicle](https://arxiv.org/abs/2609.04147) | 本文提出一个集成实体车辆、打印城市赛道与Webots数字孪生的低成本开放平台，通过命令条件化行为克隆实现了微型阿克曼车辆的端到端自动驾驶，其横向误差（6.1厘米）接近人类演示水平，弥合了仿真与真实执行之间的鸿沟。 |
| [^8] | [Prospective Coding Improves Learning in Deep Continuous-Time Recurrent Networks](https://arxiv.org/abs/2609.04134) | 提出受生物学启发的递归正交滤波器，通过无参数的前瞻编码校正每层的自下而上输入，缓解深度连续时间循环网络中依赖深度的梯度衰减问题，从而改进学习效果。 |
| [^9] | [Constant regret in general games via higher-order optimism](https://arxiv.org/abs/2609.04113) | 提出了一种名为HOOD的非耦合学习算法，通过将带折扣的高阶预测器与熵正则化相结合来抑制博弈序列的大幅振荡，从而在任意N人博弈中实现了与博弈时间无关的常数级个体遗憾。 |
| [^10] | [Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR](https://arxiv.org/abs/2609.04108) | 先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。 |
| [^11] | [Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105) | 提出硬件感知的FP4 FlashAttention-4，其中Direct-P方法将注意力分数直接映射为FP4概率，在GB200上实现最高2.13倍于BF16的前向吞吐量，并通过将前向量化直接传递到反向传播的因果路径使80亿参数模型更新加速最高1.14倍。 |
| [^12] | [DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training](https://arxiv.org/abs/2609.04094) | DRACO通过在训练中动态生成评分准则，并以闭式解方式将轨迹级评判重新分配到具体步骤，解决了无真实成功信号时长程智能体训练的细粒度信用分配问题，在AppWorld上显著超越基础模型和稀疏奖励GRPO。 |
| [^13] | [Conditioning Degenerate Diffusion Models](https://arxiv.org/abs/2609.04090) | 该论文提出利用因果最优传输为扩散系数退化（奇异）的扩散生成模型构造近似损失函数，在条件密度不存在或不光滑的极弱假设下确定用于引导的最小熵控制。 |
| [^14] | [Subspace Inference Enables Efficient Active Reward Learning from Preferences](https://arxiv.org/abs/2609.04066) | 本文提出PreferenceEKF方法，将主动偏好学习框架化为序贯贝叶斯滤波问题，通过扩展卡尔曼滤波器在低维参数子空间中高效跟踪奖励模型的不确定性，实现样本高效的RLHF奖励学习。 |
| [^15] | [The Head Complexity of Boolean Functions in Single-Layer Attention](https://arxiv.org/abs/2609.04046) | 本文证明了单层注意力模型头复杂度的精确层级结构：$k$ 个注意力头恰好能计算 $k$ 位奇偶校验而无法计算 $(k+1)$ 位奇偶校验，且该下界在嵌入维度和数值精度均无限制的情况下依然无条件成立。 |
| [^16] | [Influence of Extruded Filament Shape on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning-FEM Approach](https://arxiv.org/abs/2609.04028) | 该研究提出了一个将深度学习长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合的几何信息驱动建模框架，能够直接从材料和工艺参数生成考虑真实长条几何形状的数值模型，从而更准确地评估3D混凝土打印结构的可建造性。 |
| [^17] | [FLY-EVAL++: An Evidence-Driven Evaluation Protocol for Safety-Constrained Flight Prediction with Large Language Models](https://arxiv.org/abs/2609.04021) | 本文提出FLY-EVAL++评估协议，通过对协议合规性、物理可行性和安全约束进行确定性验证并结合量规引导的聚合评分，对66个大语言模型的飞行轨迹与姿态预测能力进行评估，发现安全合规性是区分模型优劣最具判别力的维度。 |
| [^18] | [A location-invariant estimator of extremal quantile treatment effects for heavy-tailed distributions](https://arxiv.org/abs/2609.04018) | 该论文通过逆倾向得分加权将位置不变的Fraga极值指数估计量引入因果设置，并采用基于差分的外推方案使位置参数在分位数差中自然抵消，从而首次构造出对位置平移保持不变的极端分位数处理效应估计量，解决了现有方法在重尾分布下不满足位置不变性的问题。 |
| [^19] | [LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening](https://arxiv.org/abs/2609.04013) | 大语言模型在零样本和少样本学习设置下，无需任务特定训练即可实现与传统机器学习和深度学习方法相当的早期慢性肾病筛查性能。 |
| [^20] | [Differentiable Hybrid Modelling for Learning and Optimising Chemical Transport Processes from Experimental Data](https://arxiv.org/abs/2609.04011) | 提出一个通用的可微分混合建模框架，将JAX有限体积群体平衡求解器与可学习的神经网络组件相结合，能够从真实实验数据中同时发现本构定律并拟合初始条件，用于学习和优化化学传输过程。 |
| [^21] | [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010) | 本文提出扩散增强大语言模型，通过将参数解耦为标准NTP训练的自回归权重与轻量级扩散权重，并配合Ψ-Spec采样器家族，在无需草稿模型和 negligible 训练开销的情况下，实现大语言模型的无损并行词元生成加速。 |
| [^22] | [RobustSeiz: An Open-Source Framework for Benchmarking the Robustness of EEG Seizure Detection Models](https://arxiv.org/abs/2609.04007) | RobustSeiz是一个开源、模型无关的基准测试框架，通过将四个公开EEG数据集标准化并施加临床驱动的环境、噪声和对抗性分布偏移，实现对癫痫检测模型鲁棒性的标准化、可复现的压力测试。 |
| [^23] | [Sharpening the Ensemble: An SSIM-Aligned Residual Refiner for Brain-MRI Inpainting Post-Processing](https://arxiv.org/abs/2609.03981) | 该论文提出一种后处理方法，通过将2025年BraTS脑部MRI修复任务中两个并列第一的模型组成深度集成，并训练一个轻量级的SSIM对齐残差精炼器，有效锐化了因ℓ1和MSE损失均值寻求行为而模糊的合成区域，显著提升了SSIM指标。 |
| [^24] | [Cooperative Multi-Task Semantic Communication for Joint Classification and Regression Tasks](https://arxiv.org/abs/2609.03977) | 本文将协作式多任务语义通信框架扩展到复杂Cityscapes数据集上异构的分类与回归任务的联合处理，通过采用信息最大化原理使其能够同时容纳离散与连续语义变量，突破了以往仅限于简单数据集同质分类任务的局限。 |
| [^25] | [OSR: Output Space Redistribution for Adaptive Label Removal in Classification Models](https://arxiv.org/abs/2609.03972) | 该论文提出OSR方法，通过输出空间的统计重分布来近似重新训练模型的移除后置信输出，以模块化输出过滤器的形式实现标签移除，无需原始数据和特征空间调整，从而解决了现有方法在成本、可扩展性和模型效用方面的局限。 |
| [^26] | [RARF: Region-Aware Rectified Flows for 3D Brain MRI Inpainting](https://arxiv.org/abs/2609.03956) | 提出了区域感知整流流框架RARF，通过将生成过程限制在修复区域、保留周围真实体素作为解剖上下文，并结合掩码流匹配与重建一致性训练，实现了高质量的3D脑部MRI图像修复。 |
| [^27] | [Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs](https://arxiv.org/abs/2609.03955) | 该论文提出了一种两阶段强化学习框架TCS，第一阶段生成与参考解一致的可靠测试用例，第二阶段学习针对模型当前失败模式的对抗性反例测试，从而有效提升代码大模型的测试生成质量和代码性能。 |
| [^28] | [VestigeKV: The NoPE-MLA KV Cache Carries Its Own Eviction Signal in a Vestigial Branch](https://arxiv.org/abs/2609.03949) | VestigeKV发现NoPE MLA模型KV缓存中的64维解耦RoPE残余分支已被训练重新利用为显著性信号，据此提出无需训练和量化的查询无关缓存淘汰方法，在8-32倍压缩下几乎不损失检索精度。 |
| [^29] | [Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO](https://arxiv.org/abs/2609.03941) | 该论文提出了一种面向GRPO的组级重放控制原语Headroom-Drift Replay，通过Headroom按剩余学习价值排序、Drift按策略兼容性门控来复用历史轨迹，在不改变在线数据流、不增加额外训练机制的前提下加速RL后训练，从而将重放本身的贡献与复杂训练流程解耦。 |
| [^30] | [RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting](https://arxiv.org/abs/2609.03937) | 提出即插即用的残差检索与反馈校正方法RATL，通过冻结基础预测器、将其历史预测残差构建为专属记忆，并在推理时从相似历史情境中检索残差轨迹进行校正，从而实现更鲁棒的多元时间序列预测。 |
| [^31] | [Sparse auto-regressive modeling for scene generation from multi-view images](https://arxiv.org/abs/2609.03931) | SPAR3S提出了一种无需3D真值监督的稀疏体素对齐3D潜在生成模型，通过可微高斯泼溅从多视角图像学习稀疏潜在空间，并利用自回归建模实现从稀疏视角的完整3D场景生成。 |
| [^32] | [Comparing Retrieval Methods for Academic Advisor Discovery: A Six-Method Study of 768 CS Faculty Profiles Across 9 US Universities](https://arxiv.org/abs/2609.03901) | 该研究构建了一个包含美国9所大学768位计算机科学教师档案的新数据集，系统比较了六种信息检索方法在学术导师发现任务上的表现，发现重排序方法效果最佳（平均NDCG@10达0.477），且语义检索整体优于传统词汇匹配方法。 |
| [^33] | [Beyond Endpoint Scores: Time- and Capacity-Conditioned Evaluation of Continual Knowledge Updating](https://arxiv.org/abs/2609.03900) | 该论文揭示了持续知识更新方法的优劣排名会随评估时间点和回放侧适配器容量的变化而发生逆转，证明仅依靠单一终点分数和常规秩设置无法判断哪种方法更优。 |
| [^34] | [Differentiable Interval Bottlenecks for Interpretable Anomaly Detection in Numerical Data](https://arxiv.org/abs/2609.03878) | 提出DIFFINT自编码器，通过可微的软区间瓶颈结构实现可解释的异常检测，每个潜在单元对应特征空间中人类可读的超矩形，并提供认证的重构误差下界。 |
| [^35] | [High-Dimensional Learning Dynamics of Attention-Indexed Models](https://arxiv.org/abs/2609.03858) | 本文提出注意力索引模型这一统一框架，证明高维极限下损失景观由有限的迹阶序参数刻画、SGD动力学可被有限截断系统以指数精度逼近，并揭示注意力参数化本身作为架构隐式偏置可诱导自动对称性破缺。 |
| [^36] | [Pushing the (Decision) Boundaries: Dynamically Calibrating Differentially Private Noise to Explainability in Federated Learning](https://arxiv.org/abs/2609.03851) | 提出XCal-FL算法，通过预测logit变化、反事实边距和显著性集中度三种互补信号，在联邦学习训练过程中动态校准差分隐私噪声，在保护隐私的同时保持模型解释的保真度。 |
| [^37] | [EF1-Constrained Nash Social Welfare with Identical Additive Valuations: Complexity, Guarantees, and Experiments](https://arxiv.org/abs/2609.03846) | 该论文证明在相同可加估值下EF1约束的NSW最大化问题是强NP完全的，并识别出获得更强福利保证的条件——均匀估值下每个EF1分配都是NSW最优的，而在ε-小物品条件下每个EF1分配能达到 1-O(ε²) 的显式近似比。 |
| [^38] | [Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference](https://arxiv.org/abs/2609.03844) | 提出无状态伯努利水印（SBW），通过每词元独立伯努利试验实现O(1)复杂度的绿名单判断，检测速度比KGW自盐值快6000倍以上、比SynthID快2倍，同时保持相同的N(0,1)统计检测保证。 |
| [^39] | [Multi-step Proximal Policy Improvement in Offline Reinforcement Learning](https://arxiv.org/abs/2609.03842) | 本文提出将离线actor更新统一解释为概率流形上的单步近端策略改进，并在此基础上提出多步近端策略改进（MPI）这一即插即用机制，通过组合连续的重新居中近端步骤，实现了超越数据集支持范围的受控策略改进。 |
| [^40] | [Semantic Bayesian World Models](https://arxiv.org/abs/2609.03834) | 该论文提出语义贝叶斯世界模型（SBWM）的愿景，将知识图谱从静态的事实数据库转变为共享且演化的概率信念体系——由本体公理约束先验、贝叶斯条件化更新信念、动作干预世界——从而弥合概率推理的基础模型与确定性知识图谱之间的鸿沟，实现统一的推理架构。 |
| [^41] | [Witnesses Explain Anomalies](https://arxiv.org/abs/2609.03826) | WAND是一种天生可解释的无监督表格数据异常检测器，它通过单位球面上的方向进行评分，而标记异常点的“证人”方向本身就构成了该点的逐特征解释，无需借助SHAP或LIME等事后解释方法。 |
| [^42] | [When Vision Meets Graphs: A Survey on Graph Reasoning and Learning](https://arxiv.org/abs/2609.03816) | 该综述首次系统性地提出了“视觉遇见图”这一新兴研究领域，将图的视觉呈现作为一等输入用于推理和学习，旨在弥合图学习流程与图可视化之间的长期差距。 |
| [^43] | [A Peer-Relative Representation Learning Framework for Energy Inefficiency Identification in Mobile Network Sites](https://arxiv.org/abs/2609.03809) | 本文提出一种无监督的同行相对表示学习框架，通过能耗感知的最小失真嵌入将异常高能耗站点与相似站点在嵌入空间中分离，从而在缺乏真实低效标签的情况下识别移动网络站点的能源低效问题。 |
| [^44] | [Free Pause Tokens](https://arxiv.org/abs/2609.03807) | 提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。 |
| [^45] | [From Ordered Bernoulli Levels to Critical-Line Geometry: Integer Quantization, Bernoulli Residual Phase, and Prime-Power Spectra](https://arxiv.org/abs/2609.03801) | 本文通过有序伯努利核的水平集几何，将黎曼临界线零点纵坐标精确量子化为整数壳层与周期性一阶伯努利残余相位之和，并借助唯一因子分解进一步解析出素数幂谱结构。 |
| [^46] | [Landmark-Based Discrimination of Injury-Associated Athlete-Sessions from Minute-Resolution Multimodal Football Monitoring Data](https://arxiv.org/abs/2609.03790) | 本文提出一种基于固定地标点的建模方法，在每个地标时刻（如10、20、30分钟）利用截至该时刻的观测信息为每个运动员-场次构建单一表示，从而在损伤信息仅有场次级标签的情况下，避免不合理的分钟级损伤监督，实现从分钟级多模态足球监测数据中判别损伤相关场次。 |
| [^47] | [OBER+: Continuity-Aware Reporting and Traceable Continuous Improvement in Outcome-Based Education](https://arxiv.org/abs/2609.03770) | OBER+通过五个相互衔接的阶段将测得的成果差距转化为可追溯且经评估的纠正措施，并引入连续性规则以避免在成果表述变更时误读达成度趋势，从而在成果导向教育中实现从测量到改进的闭环。 |
| [^48] | [From Nowcasting to Forecasting: Adapting a Reanalysis-Trained](https://arxiv.org/abs/2609.03763) | 本文开发了CloudCast v2机器学习模型，通过在再分析数据上训练学习云演变动态，并利用条件流匹配方法将其适配到卫星观测云场，实现了基于观测初始条件的12小时云量预报。 |
| [^49] | [Projected Riemannian Gradient Descent for the Bures-Wasserstein Barycenter: Dimension-Independent Linear Convergence at Unit Step Size](https://arxiv.org/abs/2609.03762) | 该论文提出投影黎曼梯度下降算法，在单位步长下实现了Bures-Wasserstein重心计算的与维度无关的线性收敛，解决了快速收敛与维度无关保证之间的两难困境。 |
| [^50] | [Genetic Algorithms for Tractable Bayesian Network Fusion via Pre-Fusion Edge Pruning](https://arxiv.org/abs/2609.03724) | 本文提出一种基于遗传算法的贝叶斯网络融合共识框架，通过融合前边剪枝在优先保留输入网络间共享依赖结构的同时控制树宽，实现了计算上可处理且不易过拟合的网络融合。 |
| [^51] | [Artificial Intelligence for Energy Optimization in Data Centers](https://arxiv.org/abs/2609.03716) | 该论文通过系统性文献编码揭示数据中心AI节能研究存在“控制与负载割裂”、过度依赖仿真验证、忽略水资源与隐含碳排放、各类方法节能效果无法区分排名等十大差距，并提出将控制策略与工作负载需求相耦合的CLEAR-DC研究框架。 |
| [^52] | [Federated Causal Discovery via Regression-Directed Cumulants](https://arxiv.org/abs/2609.03705) | 提出一种基于回归导向累积量的联邦LiNGAM因果发现方法，利用高阶累积量张量在独立样本组间的精确可加性，在水平、垂直和混合数据划分下仅需一轮通信即可实现隐私保护的因果发现，并克服了现有方法FedISHC在近似对称情形下失效的问题。 |
| [^53] | [Resolution-Aware Experimental Design under Partial Identifiability](https://arxiv.org/abs/2609.03686) | 本文提出分辨率感知实验设计（RAED），在部分可识别性下通过最小化期望非空结构候选集来选择实验，证明了传统信息增益准则会因跨冗余混叠而失效，而RAED在复合Blackwell比较下能保持正确的实验排序。 |
| [^54] | [Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language](https://arxiv.org/abs/2609.03677) | 本文提出集合差异描述方法，利用自然语言自动描述自动驾驶数据集中不同图像子集之间的差异，通过基于目标检测的对象中心分析实现对数据集组成和域偏移的可解释理解。 |
| [^55] | [Out-of-Distribution Generalisation with Sequence Models in Offline Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2609.03667) | 该研究通过系统性分析发现，扩展任务多样性而非数据集规模是离线多智能体强化学习实现零样本任务泛化的关键因素，其提出的多任务序列建模方法在留出测试任务上相比单任务模型平均提升3.2倍。 |
| [^56] | [Extracting Forgotten Prompts from Targeted Unlearned Models](https://arxiv.org/abs/2609.03662) | 本文首次提出目标主动搜索攻击，证明攻击者无需事先知道遗忘内容，仅凭保留数据和黑盒访问，即可在有限查询预算下从定向遗忘模型中提取出被遗忘的提示词本身。 |
| [^57] | [Local Updates, Global Learning (LUGL): Playing Games with non-incremental Learners](https://arxiv.org/abs/2609.03660) | 提出LUGL框架，通过将数据收集与模型拟合解耦，使梯度提升树等非增量学习器能够克服分布偏移问题，成功应用于自我博弈强化学习场景。 |
| [^58] | [Relative Prime Factorization and Finite-State Presentations under Fixed Finite-Monoid Observation](https://arxiv.org/abs/2609.03643) | 本文通过穷举计算机验证的反例证明唯一素因子分解并不蕴含有限相对表示性质（FRP），进而引入更强的有限状态相对表示性质（FSRP）来隔离并刻画这一表示障碍。 |
| [^59] | [Residual neural networks overcome the curse of dimensionality for semilinear heat equations](https://arxiv.org/abs/2609.03626) | 本文首次证明了残差神经网络在数值逼近半线性热方程时，能以多项式数量的参数克服维数灾难，其证明方法是将多层Picard估计量的确定性实现表示为残差神经网络。 |
| [^60] | [On the Interaction Between Model Compression and Test-Time Adaptation](https://arxiv.org/abs/2609.03604) | 本文首次系统研究了模型压缩与测试时自适应（TTA）之间的交互作用，发现压缩模型虽在有监督自适应下保持高精度，但其TTA性能随压缩程度增加而显著下降，其根源在于表示多样性的降低和限制可恢复性的结构约束。 |
| [^61] | [Neural-Network Maxent: a general extension with learned nonlinearity, applied to time-series for Desert Locust distribution modelling](https://arxiv.org/abs/2609.03603) | 本文提出RNN Maxent，用神经网络（GRU）替换Maxent框架中的固定特征字典，从而以可学习的方式捕捉时间序列环境协变量中的非线性和序列结构，显著改进了沙漠蝗虫的物种分布建模。 |
| [^62] | [LevelSyn: Physical-Aware Logic Synthesis via Level-Asynchronous Graph Neural Networks](https://arxiv.org/abs/2609.03594) | LevelSyn提出了一种物理感知的逻辑综合框架，利用层级异步图神经网络捕捉与-非图（AIG）的结构和方向语义以预测高保真度门坐标，并结合线长驱动的优化引擎，从而缓解综合与物理设计脱节带来的PPA退化和设计收敛周期过长的问题。 |
| [^63] | [Correlated initialization of deep residual networks](https://arxiv.org/abs/2609.03589) | 本文证明了具有跨层相关权重的深度残差网络在正则变化相关性条件下存在唯一的临界缩放，使得无穷深度极限由 Hermite 过程驱动的 Young 微分方程描述，从而证实并扩展了相关初始化在布朗随机微分方程与常微分方程之间连续插值的猜想。 |
| [^64] | [WeatherNext 3: Increasing resolution and performance of global weather models with raw observations](https://arxiv.org/abs/2609.03582) | WeatherNext 3通过直接引入原始观测数据（尤其是低延迟地球静止卫星数据），实现了每小时更新的高分辨率天气预报（0.1度、小时级），在概率性中期预报技能上确立了新的最先进水平。 |
| [^65] | [Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning](https://arxiv.org/abs/2609.03565) | 该论文提出一种融合逆动力学与状态对齐的端到端JEPA世界模型，将潜在表示扎根于物理构型和运动信息，从而显著提升目标条件机器人规划的成功率。 |
| [^66] | [Coupled Scaling: A Representational Accessibility Framework for Neural Scaling Laws](https://arxiv.org/abs/2609.03533) | 该论文提出“耦合缩放”框架，首次将神经缩放定律与架构-优化系统可达的表征几何联系起来，并证明有限预算下的损失残差指数被任务尾部率与覆盖率所确定的区间严格界定。 |
| [^67] | [LeanGRPO: Eliminating Redundant Recomputation in Diffusion RL](https://arxiv.org/abs/2609.03528) | LeanGRPO 通过重构数据并行布局并提出两种无需重计算的训练方案，使 rollout 阶段的计算图与激活值可在策略更新时直接复用，从而消除同策略扩散强化学习中数学上冗余的重计算。 |
| [^68] | [EPIC: Explicit Posterior Item Conditioning for Semantic ID Diffusion Recommendation](https://arxiv.org/abs/2609.03522) | 提出EPIC方法，将显式的物品级后验竞争引入语义ID扩散去噪过程，通过个性化候选物品后验分布来指导未确定位置的词元预测，且无需修改冻结的预训练骨干网络。 |
| [^69] | [LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues](https://arxiv.org/abs/2609.03507) | 提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。 |
| [^70] | [An Adversarial Zero-Shot Learning Approach for Anomaly Detection in Multivariate IoT Traffic Data](https://arxiv.org/abs/2609.03505) | 该论文提出了一种基于序列VAE架构、结合对抗学习与对比损失的零样本异常检测框架，通过编码器/解码器适配层和基于目的地的流量分割策略，在无需标注数据的情况下实现了多变量物联网流量异常检测及跨领域自适应。 |
| [^71] | [Restricted Eigenvalues Beyond Gaussian Width: Threshold Occupancy under Heavy Tails](https://arxiv.org/abs/2609.03504) | 本文对COLT 2015开放问题给出否定回答，证明仅凭一致小球条件不足以将次高斯测量的 $1+w(A)^2$ 样本量规律推广到重尾设计，其根本障碍是“同时阈值占据”现象。 |
| [^72] | [Towards a Statistical Understanding of Mixture-of-Experts](https://arxiv.org/abs/2609.03501) | 本文从统计学理论视角理解混合专家模型，将其视为局部化聚合，推导出分解近似误差、专家学习误差与路由器估计误差的预言机风险界，并揭示稀疏Top-K路由如何在控制计算成本的同时保留局部化聚合的优势。 |
| [^73] | [Spectral characteristics of autoencoder parameters as a vector representation of data](https://arxiv.org/abs/2609.03495) | 提出将自编码器参数矩阵的谱特征（奇异值）作为数据的稠密向量表示，并从理论上证明这些奇异值与训练数据协方差矩阵的特征值相关联，从而在数据空间与参数空间之间建立信息传递。 |
| [^74] | [Tree species mapping in Denmark: A comparison of spectral-temporal features with geospatial foundation model embeddings](https://arxiv.org/abs/2609.03480) | 本研究利用丹麦国家森林清查数据与Sentinel卫星观测，系统比较了人工构建的光谱-时间特征与地球观测基础模型（TESSERA和AlphaEarth）嵌入在树种分类中的表现，发现基于光谱-时间特征的多层感知机取得最佳性能（纯林宏观F1达0.843），而基础模型嵌入也展现出有竞争力的结果，为大规模森林树种制图的方法选择提供了重要参考。 |
| [^75] | [Mind the Gap: Robustness Risks in PII Detection Systems](https://arxiv.org/abs/2609.03464) | 本研究构建了涵盖七类自然分布偏移的压力测试基准，发现SpaCy、Presidio和Qwen2.5-3B三类主流PII检测系统在真实非正式输入下均出现显著性能退化，表明标准基准评估掩盖了实际部署中的隐私安全风险。 |
| [^76] | [A Two-Stage Forecasting System for CPU Workload Prediction in Private Clouds](https://arxiv.org/abs/2609.03457) | 该论文提出了一种两阶段预测系统，先预测客户服务请求（TPS），再由其估计未来CPU工作负载，采用级联XGBoost架构并结合扩展窗口的自适应在线再训练来应对云工作负载中的概念漂移。 |
| [^77] | [Beyond Straightness: Non-Crossing Flow Matching via Quantile AlignTree Coupling](https://arxiv.org/abs/2609.03443) | 提出了一种基于分位数对齐树结构的高效耦合方法QAT-FM，以近线性时间构建无交叉的流匹配插值路径，并支持大规模高维生成任务的可扩展训练。 |
| [^78] | [Guide, Not Bind: Why Defeasible Priors Fail in Augmented Lagrangian Causal Discovery](https://arxiv.org/abs/2609.03442) | 本文揭示了可微因果发现中“引导而非约束”设计（用增广拉格朗日惩罚强制执行专家先验、并期望数据自适应松弛机制推翻错误规则）会因两个独立原因失效——顺序惩罚递增会在反事实检验检测到之前就抑制被错误禁止的真实边，且即使修复这两个问题也只能部分恢复该设计的功能。 |
| [^79] | [It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories](https://arxiv.org/abs/2609.03436) | 该研究提出重启控制的截断探针方法，证明大语言模型推理轨迹中所谓的“突破时刻”和“早期注定失败”大多是预算与难度造成的混淆——178个问题-模型组合中仅1个真正存在前缀特有价值，且在同等token预算下延续自身推理前缀几乎总是优于从头重启。 |
| [^80] | [TraveL: Transformer-based Multi-view Path Distributional Representation Learning](https://arxiv.org/abs/2609.03427) | 该论文提出了基于Transformer的多视角分布式表示学习框架TraveL，通过捕捉旅行者行为的多样性和路段的区域相关性，将路径与出行开始时间编码为分布式表示，从而能够解码路径上旅行者行为的可能样本。 |
| [^81] | [Inferred Generative-Process Diversity Predicts Correlated Failure Across Language Models](https://arxiv.org/abs/2609.03422) | 该论文提出“生成过程多样性”这一比语义多样性更根本的模型多样性概念，并基于算法信息论用经置换控制的归一化压缩距离来度量语言模型间的推断生成过程多样性，从而预测语言模型间的相关失效。 |
| [^82] | [Privacy, Robustness, and Fairness Trade-offs in Federated Intrusion Detection: Geometric Indistinguishability at the Aggregation Interface](https://arxiv.org/abs/2609.03420) | 本文揭示了联邦入侵检测中差分隐私、拜占庭鲁棒性与类别公平三大需求并非可独立组合，并提出“几何不可区分性”概念，用以解释隐私噪声导致的客户端更新分散会削弱鲁棒聚合对少数类攻击信号的保留能力。 |
| [^83] | [Dude: A Dual-Detection Multi-Agent System for Paper-Code Discrepancy Detection](https://arxiv.org/abs/2609.03416) | 提出了首个用于论文-代码差异检测的双检测多智能体系统Dude，通过粒度对齐协商机制和两阶段显著性过滤机制，有效解决了论文语言与代码语言粒度不对称导致的误报问题。 |
| [^84] | [Spectral Convergence of Random Feature Method in Multiple Dimensions](https://arxiv.org/abs/2609.03401) | 本文证明了随机特征方法在多维情形下对Sobolev、Gevrey、超解析及带限函数类目标的谱收敛性，并给出了收敛速率随目标正则性从超指数到代数的刻画，同时建立了强形式和弱形式RFM离散化的抽象误差估计。 |
| [^85] | [Computing stable configurations of confined smectic liquid crystals with a deep variational framework](https://arxiv.org/abs/2609.03389) | 提出一种深度变分框架（DVF），通过坐标映射处理物理约束并利用预热惩罚克服神经网络的谱偏差，实现了复杂受限几何下近晶相液晶稳定构型（含高频层状密度调制）的稳健计算。 |
| [^86] | [TIGPO: Temporal Instance-Graph Policy Optimization for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.03383) | TIGPO通过为每个任务维护跨策略更新的持久化转移图，并结合探索槽位与重访槽位的预算分配机制，将基于图的信用分配扩展到时序维度，从而显著改进长程LLM智能体的信用分配与优势估计。 |
| [^87] | [SurgeGen: A Hybrid Generative Diffusion Framework for Storm Surge Scenario Synthesis](https://arxiv.org/abs/2609.03382) | 本文提出SurgeGen，一个基于扩散模型的两阶段生成框架，将基线预测与条件生成相结合，以连续空间中定义的假想风暴参数为条件合成风暴潮情景，为计算昂贵的物理数值模拟提供了更高效、更具可解释性的代理建模方案。 |
| [^88] | [RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory](https://arxiv.org/abs/2609.03379) | RecurTrace通过循环记忆注意力使循环层能够回顾之前迭代的计算状态，并结合由oracle监督的停止头动态决定循环次数，实现了更高效的自适应潜在推理。 |
| [^89] | [SimpleDesign: A Joint Model for Protein Sequence and Structure Codesign](https://arxiv.org/abs/2609.03377) | SimpleDesign提出了一种直接在数据空间训练的单阶段端到端多模态生成模型，无需传统的多阶段潜在空间训练即可实现蛋白质序列与结构的协同设计。 |
| [^90] | [Spruce: Scalable Private Outsourced Retrieval Using Compact Embeddings](https://arxiv.org/abs/2609.03376) | Spruce通过将紧凑二进制嵌入表示与密码学协议协同设计，用汉明距离计算取代全语料库嵌入评分，将百万文档规模的隐私外包检索性能显著提升。 |
| [^91] | [Grassmann--Pl\"ucker Parametrization of Convolutional Filter Subspaces: Regularity and Closed Embeddings](https://arxiv.org/abs/2609.03361) | 该论文提出以滤波器空间中固定维数的子空间（而非有序滤波器向量族）作为单层卷积的几何参数化，并通过 Grassmann–Plücker 嵌入证明该参数化具有良好正则性（微分处处单射）且构成闭嵌入。 |
| [^92] | [Time Without Timesteps: Simulating Coupled Dynamical Systems via Self-Consistency](https://arxiv.org/abs/2609.03358) | 该论文提出用神经代理模型将完整轨迹到完整轨迹的映射与轨迹间的自洽不动点迭代相结合，替代传统的逐步时间推进来模拟耦合动力学系统，将参考积分器的1500步压缩为4-10次牛顿迭代，并使梯度计算变为内存与深度无关的GMRES线性求解。 |
| [^93] | [ALRA: Adaptive Local Relational Alignment for Logit-Based Pre-training Distillation of Autoregressive Language Models](https://arxiv.org/abs/2609.03355) | 提出ALRA框架，通过让学生提议候选词元并以教师最可能词元作为锚点，同时根据教师概率分布广度自适应调整候选词元数量，从而改进自回归语言模型的局部logit蒸馏效果。 |
| [^94] | [Efficient Constant Optimization for Symbolic Regression with GPU-Accelerated Tree-Based Genetic Programming](https://arxiv.org/abs/2609.03352) | 该论文提出了一种驻留GPU的批量Levenberg-Marquardt求解器，通过每次迭代仅需固定次数CUDA启动的设计，首次在GPU加速的树形遗传编程符号回归中对结构异构的表达式树种群实现高效常数优化，在NVIDIA A100上达到每秒5.1×10⁵棵树的速度，并保证优化后的常数不会劣于初始值。 |
| [^95] | [From Zero to Hero: An Open LLM Ecosystem for Armenian](https://arxiv.org/abs/2609.03350) | 该研究发布了ArmWeb（437万篇亚美尼亚语新闻）与ArmSTEM（37.3万条英亚平行数理题目）两个数据集，并通过继续预训练Gemma-4-E4B构建出首个附带完整训练数据和配方的开源亚美尼亚语模型arm-gemma-e4b，其性能超越所有现有的开放亚美尼亚语模型。 |
| [^96] | [Learning Informative Prior with Infinite-Dimensional Continuous Normalizing Flow for Bayesian Inverse Problem](https://arxiv.org/abs/2609.03343) | 该论文提出了一种基于无限维连续归一化流的新方法，通过在希尔伯特空间中引入定义良好的神经常微分方程将简单参考测度变换为编码先验信息的复杂测度，建立了无限维贝叶斯先验的适定性理论框架，并提供了先验训练方法与后验采样算法，用于求解偏微分方程的贝叶斯逆问题。 |
| [^97] | [Gradients Know What Outcomes Don't: Unlocking Reinforcement Learning for LLM Reasoning with Gradient-Aligned Rewards](https://arxiv.org/abs/2609.03342) | 提出梯度对齐奖励（GAR）：在策略自身的梯度空间中，通过截断反向传播提取每个 rollout 的梯度向量并与专家锚点梯度计算余弦相似度，以低于 9% 的时间开销生成密集的推理感知奖励，突破了 RLVR 二值结果奖励无法区分正确轨迹的局限，并可乘性分解为预测误差与激活模式两个因子。 |
| [^98] | [A Large Open Multi-Energy Corpus of Soil Compaction Tests, with Machine-Learning Baselines](https://arxiv.org/abs/2609.03337) | 本文发布了迄今规模最大的开放多能量土体压实试验数据集（2854个试验、四个普氏能量等级），经物理一致性审核后发现已发表压实数据中相当一部分在物理上不可行，并给出了最优饱和度基线值0.815及机器学习预测基线。 |
| [^99] | [Introducing SINFONIA: Symplectic, slimplectic and Magnusian (Neural) Flows for Orbital Numerical Integration and Acceleration](https://arxiv.org/abs/2609.03329) | 本文提出SINFONIA三种保结构神经流架构（辛/Slimplectic流、Taylor锚定流和Magnus流）用于长时引力波旋近轨道的数值积分，并发现长期精度由映射误差在能量-角动量平衡所决定的单一长期通道上的带符号投影控制。 |
| [^100] | [DE-Venus: A Data-Efficient RLVR Framework for Large Language Models](https://arxiv.org/abs/2609.03324) | DE-Venus提出了一个统一的数据高效RLVR框架，将监督视为跨数据准备与策略优化不断演化的状态，通过主动数据选择、弱监督构建和训练时监督精炼三个模块，降低大语言模型推理训练中的采样与标注成本。 |
| [^101] | [Beyond .WAV: Design and Software Verification of VocalCap, a Traceable Browser-Based Audio Capture System for Vocal Biomarker Research](https://arxiv.org/abs/2609.03320) | VocalCap是一个由机构控制的、可追溯的浏览器端音频采集系统，通过版本化协议驱动工作流，并为每条录音保留从采集执行到字节级完整性和转换来源的多层级证据，从而为声音生物标志物研究的远程语音采集提供端到端的可验证性与完整性保障。 |
| [^102] | [Risk and Anomaly Identification for Distribution Network Optimal Operation Based on Reinforcement Learning and Uncertainty Quantification](https://arxiv.org/abs/2609.03308) | 本文提出一种融合分布式强化学习与贝叶斯深度强化学习的不确定性感知框架，通过将总不确定性分解为偶然与认知两个分量，实现对配电网优化运行中固有风险与分布外异常的联合识别。 |
| [^103] | [Geometry-Aware Graph Construction via Adaptive Spectral Bandwidth Control](https://arxiv.org/abs/2609.03306) | 该论文提出了一种逐节点自适应带宽选择准则，通过将核的有效秩与最小生成树估计的局部内在维度相匹配，使高斯核图的谱复杂度与数据流形的内在几何复杂度保持一致，从而避免带宽选择过小或过大带来的谱退化问题。 |
| [^104] | [Latent Energy Action Planning with World Models](https://arxiv.org/abs/2609.03294) | 提出 LEAP 方法，将完整动作序列视为可微变量，通过冻结的潜空间世界模型，将终端潜空间目标匹配与状态能量相结合进行优化，实现高效的动作规划。 |
| [^105] | [Selective Hypergraph Refinement for Frozen Graph Clustering](https://arxiv.org/abs/2609.03265) | 提出选择性超图精化（SHR）方法，在不使用标签、不更新模型参数、节点表示和图结构的前提下，利用属性超图补充高阶关系，并基于图结构、节点属性和匹配零假设证据对更新方向的可靠性进行评估，从而选择性地精化已冻结图聚类模型的聚类结果。 |
| [^106] | [What Else Needs Fixing? Exploring Cost-Effective Test-Time Compute for Revision Propagation in Artifacts Generated Through Conversation](https://arxiv.org/abs/2609.03254) | 本文针对对话生成工件中修订传播这一新问题提出了一个基准，评估了九种测试时计算修订方法，发现基线方法可达68.3%–93%的准确率，并识别出最具成本效益的方法。 |
| [^107] | [What is Smoothness?](https://arxiv.org/abs/2609.03246) | 本文提出通过群 Cayley 图拉普拉斯算子各块特征值的均值来为非交换群的不可约表示构造一种自然的排序函数，从而将基于傅里叶分析的光滑性概念推广到一般群上。 |
| [^108] | [FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience](https://arxiv.org/abs/2609.03241) | FlowBalance提出一种以终端验证器的组优势来校准同模型自引导分数的自我改进方法，通过在正优势轨迹上保留、负优势轨迹上反转、无结果偏好时禁用引导，实现更稳定的推理能力自我提升。 |
| [^109] | [B2B Customer Conversion Prediction: A Document Representation, Graph Theory, and CatBoost Driven Methodology](https://arxiv.org/abs/2609.03239) | 本文提出了一种融合文档表示、图论和CatBoost模型的方法，通过多键聚合B2B客户数据并生成特征，实现高达91%准确率的客户购买转化预测。 |
| [^110] | [The 2026 PNPL Competition: Word Classification and Efficient Cross-Subject Generalisation in LibriBrain100](https://arxiv.org/abs/2609.03231) | 2026年PNPL竞赛推出扩展MEG数据集LibriBrain100，聚焦词汇分类任务与高效跨被试泛化，旨在让语音解码脑机接口仅需几分钟数据即可泛化到新用户。 |
| [^111] | [Language-encoded network topology enables large language models to reason about complex networks](https://arxiv.org/abs/2609.03229) | 提出BioGlyph方法，将网络拓扑编译为可解释、可迁移的结构角色语言，从而使大语言模型能够对复杂网络进行结构推理。 |
| [^112] | [Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor](https://arxiv.org/abs/2609.03221) | 临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。 |
| [^113] | [SWIM: Student Writing Simulation via Proficiency-Conditioned Generation](https://arxiv.org/abs/2609.03215) | 该论文提出SWIM任务，将学生写作模拟形式化为基于熟练度条件的作文生成，实验发现提示方法对写作熟练度的控制有限，模型虽能调整内容导向特征，却难以重现词汇、语法和组织结构层面的真实学生写作差异。 |
| [^114] | [Improving precipitation forecasts in an AI weather model using observational data](https://arxiv.org/abs/2609.03210) | 通过使用观测的IMERG降水数据微调图变换器AI天气预报模型，将中期降水预报的CRPS提升最高19%，并使全球极端降雨预测的Brier技巧评分超越最先进业务化模型57%。 |
| [^115] | [VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis](https://arxiv.org/abs/2609.03203) | VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。 |
| [^116] | [MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval](https://arxiv.org/abs/2609.03201) | 提出了轻量级记忆框架MemoryLACE，通过稀疏的合并、取代和矛盾关系显式建模文本证据的生命周期，重建关系感知的证据单元以呈现当前、历史、支持和冲突证据，从而改进长期LLM智能体的记忆整合与证据检索。 |
| [^117] | [Generative Nested Sampling of Atomistic Thermodynamic Landscapes](https://arxiv.org/abs/2609.03193) | 本文提出NS-Flows，利用单一条件归一化流替代马尔可夫链更新来加速原子体系热力学景观的嵌套采样，并通过对比揭示原子多模态（离散、组合性、硬碰撞壁分隔）与引力波后验（平滑简并、局域耦合）在结构上的根本差异。 |
| [^118] | [Coupled Tensor-Tensor Completion Method with Applications in Drug Repurposing](https://arxiv.org/abs/2609.03190) | 本文提出了一种名为耦合张量-张量补全（CTTC）的新框架，首次支持以张量形式融入辅助信息，通过挖掘多模态张量间的隐藏关联提升补全性能，并在药物重定位等生物医学问题上具有应用价值。 |
| [^119] | [Portable Causal Fairness Across Synthetic Data Generator Families](https://arxiv.org/abs/2609.03180) | 该论文证明了基于因果图边割的公平性机制可跨三个不相关家族共九个合成数据生成器（含差分隐私变体）普遍移植，并提出新的因果扩散模型骨干，在保真度接近边际分布方法的同时实现了所测家族中最公平的合成数据发布。 |
| [^120] | [Frontier LLMs are effective batch optimizers: Assessing reasoning models in continuous and discrete settings](https://arxiv.org/abs/2609.03177) | 前沿LLM在数值测试函数上是有竞争力的零样本批量优化器但性能相对脆弱，而在与预训练数据结构相似的语义丰富的离散空间中，其批量优化能力显著优于经典方法。 |
| [^121] | [Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization](https://arxiv.org/abs/2609.03158) | 提出CoverPruner，一种无需训练的视觉token剪枝方法，将剪枝创新性地建模为表示覆盖最大化问题，确保每个被剪枝的token都有存活的token为其代言，尤其在激进压缩下取得最佳准确率。 |
| [^122] | [BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training](https://arxiv.org/abs/2609.03151) | 提出批感知序列并行BASP，根据微批大小将GPU划分为互不相交的序列并行组以缩小all-to-all通信组规模，实现通信局部化，从而显著降低大语言模型长序列训练的通信开销并提升训练效率。 |
| [^123] | [Routing Is Not Enough: Diagnosing Intra-Adapter Subspace Contention in MoE+LoRA Fine-Tuning](https://arxiv.org/abs/2609.03150) | 该研究发现在 MoE+LoRA 多领域微调中，即使专家路由近乎完全分离，负迁移仍由近乎正交的领域梯度在同一低秩适配器子空间内竞争所致，并提出 SpawnLoRA，通过在检测到适配器争用时于 MoE 专家内部动态添加门控子适配器（保持路由器固定）来化解这一问题。 |
| [^124] | [RACE-AIMC: Selective Inference for Heterogeneous Analog In-Memory Accelerators at the Edge](https://arxiv.org/abs/2609.03149) | RACE-AIMC提出了一种基于统计学的选择性推理框架，通过离线分析多个异构的模拟内存计算加速器芯片，以风险感知的认证集成方式在“运行全部芯片合并结果”与“信任单一芯片”之间做出最优选择，从而在边缘设备上兼顾能耗效率与推理可靠性。 |
| [^125] | [Feasible but Not Safe: Constraint Violations and Report-Channel Attacks in Learned Cell-Free ISAC Association](https://arxiv.org/abs/2609.03147) | 本文揭示了基于GNN的无蜂窝ISAC关联调度器虽然预测精度很高，但仍会违反硬约束导致解不可行，并证明通过将输出投影到可行解（如简单的贪心修复）能以极低的效用损失恢复约束满足。 |
| [^126] | [Sensing Which Modality Matters: Evidence-Gated Regularization for Robust VLA Policies](https://arxiv.org/abs/2609.03142) | 本文揭示了VLA策略中的“模态纠缠”问题，并提出证据门控正则化（EGR）——一种零推理开销的训练目标，通过对低证据传感器施加不变性约束、对高证据传感器施加单传感器充分性约束，使多模态策略在遮挡和干扰下更加鲁棒。 |
| [^127] | [A Closed-Form Formula for Consistent Lipschitz Regression on Metric Spaces with Sparse Neural Network Realizations](https://arxiv.org/abs/2609.03129) | 本文提出一个简单的闭式“两阶段”复合公式，用于从含噪观测中重构度量空间上的Lipschitz函数，并给出同时控制逼近与统计误差、且优化误差为零的高概率一致恢复保证。 |
| [^128] | [Kernel Reboot: Breaking the Boundaries of Neural Tangent Kernels for Neural Fields](https://arxiv.org/abs/2609.03117) | 本文提出 NTK-KIP、MetaQuill 和 MetaQuill-KIP 三种算法，通过蒸馏支持集的非线性核回归与元学习共享初始化相结合，突破了神经切线核的线性局限，实现了从稀疏观测中快速高质量地重建神经场并积累可复用的任务先验。 |
| [^129] | [CRAW: Codec Robust Audio Watermarking](https://arxiv.org/abs/2609.03107) | CRAW 提出了一种对神经编解码器鲁棒的音频水印框架，通过失真感知训练、注意力池化、推理时感知掩蔽与纠错码，在抵御神经再合成攻击的同时保持高感知音质。 |
| [^130] | [Scaling Laws, Tabular Data and Actuarial Ratemaking Models](https://arxiv.org/abs/2609.03106) | 该研究首次在精算定价这一表格数据场景中检验了深度学习缩放定律的适用性，发现各模型家族均随数据量增加而性能提升，但TabM的数据缩放能力显著优于表格Transformer和MLP基线，而Transformer的参数缩放能力较弱。 |
| [^131] | [Occupancy-based Quantile Risk Control](https://arxiv.org/abs/2609.03104) | 提出了一种基于占位问题的分位数风险控制方法（OQRC），通过有序校准损失划分损失空间并估计测试损失分布，实现了具有有限样本有效性的紧致风险控制界。 |
| [^132] | [Distilling deep optical flow stereo methods to retrieve dense three-dimensional wind fields](https://arxiv.org/abs/2609.03100) | 本文提出用深度光流替代立体匹配中基于窗口的跟踪方法，并结合自监督几何残差损失与探空数据有监督微调，实现了高效且精确的密集三维风场反演，同时减少了对数值天气预报和多卫星重叠观测的依赖。 |
| [^133] | [Beyond Blur: A Semantic Tri-view Pipeline for Teledermatology Gradability via Skin Micro-relief](https://arxiv.org/abs/2609.03095) | 该论文提出了一种可解释的语义三视图流水线，将皮肤表皮微纹理作为可计算的图像质量生物标志物，通过轻量级分割模型与多视图聚合分类器相结合，实现自动化且鲁棒的远程皮肤病病例可分级性筛查。 |
| [^134] | [The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA](https://arxiv.org/abs/2609.03090) | 研究通过秩-k投影消融实验发现，矩阵值连续思维链模型（Matrix-CODI）中潜在矩阵的有效秩与任务准确率无关——低秩截断几乎不影响性能且训练损失不偏好任何特定秩，表明矩阵潜在表示并未通过高秩结构编码并行推理路径。 |
| [^135] | [LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference](https://arxiv.org/abs/2609.03079) | LeanStream通过利用部分GPU计算结果渐进精化计算、加载和缓存保留优先级，实现了GPU执行与存储I/O的细粒度重叠，突破了端侧LLM推理中准确决策与计算-I/O重叠难以兼得的根本性权衡。 |
| [^136] | [Position: Unlabeled IS NOT Equal to No Human Supervision in Visual Learning](https://arxiv.org/abs/2609.03077) | 该立场论文指出，无标签数据并不等于没有人类监督——不同的数据整理方式与训练目标蕴含着不同的人类先验，单一的“无监督”术语已无法反映这些差异，研究者应更明确地识别监督的真正来源。 |
| [^137] | [Learnable composition for neural operators](https://arxiv.org/abs/2609.03069) | 该论文提出LatentDDM方法，通过预训练预测小子域的神经算子、再仅训练一个轻量级组合模块来迁移到新场景，从而大幅降低将神经算子适配到新几何、尺寸或运行条件时所需的高保真模拟成本。 |
| [^138] | [Differentially private federated learning with Byzantine-robust aggregation: A cross-domain framework for secure model training in banking and healthcare systems](https://arxiv.org/abs/2609.03064) | 本文提出DP-BR-FedAvg框架，将高斯机制差分隐私与坐标级截断均值拜占庭鲁棒聚合相结合，同时防御梯度泄露攻击和恶意客户端的对抗性更新，为银行和医疗等受监管领域的安全联邦模型训练提供了跨领域解决方案。 |
| [^139] | [IDSPACE: A Novel Document Generator for Reliable Evaluation of Digital Identity Verification Systems [Extended Technical Report]](https://arxiv.org/abs/2609.03052) | 本文提出IDSpace——一种新型身份证件合成文档生成器，通过模型引导的贝叶斯优化仅需少量目标域样本即可自适应调优生成参数，并将用户元数据与自动控制参数解耦，为数字身份验证系统的可靠评估提供高质量合成数据。 |
| [^140] | [Advances in Machine Learning for Directed Evolution: A Five-Year Retrospective](https://arxiv.org/abs/2609.03046) | 本文回顾指出，机器学习辅助定向进化（MLDE）未能像其他蛋白质工程领域那样取得变革性进展，其主因是研究者追求“最优蛋白质”的目标与定向进化在时间和资源约束下寻找“足够好蛋白质”的实际需求脱节，例如忽视DNA合成成本导致方法缺乏实际适用性。 |
| [^141] | [Population-Calibrated Graph Screening at 835-Million-Address Scale, with Label-Free Transfer to New Chains](https://arxiv.org/abs/2609.03036) | 该论文提出了一个已部署的大规模区块链合规筛查系统，在覆盖五条EVM链、8.35亿地址和158亿条边的统一交易图上，利用带每链归一化的归纳式图编码器对地址评分，通过总体分数分布的精确分位数设定阈值使警报量可预先确定，并实现了无需目标链标签即可向新链迁移的高召回筛查能力。 |
| [^142] | [You Can't Escape Your Own Activations : Evaluation Awareness and Multi-Agent Monitoring](https://arxiv.org/abs/2609.03035) | 该研究首次系统考察了当LLM智能体被明确告知其内部激活正被监控（并可收到监控器反馈）时，即“评估意识”状态下，基于激活探针的多智能体勾结检测效果如何变化。 |
| [^143] | [ObserverBench: Testing Mechanistic Estimates for Intervention and Control](https://arxiv.org/abs/2609.03026) | 提出 ObserverBench 基准框架，将估计精度与所选干预行动造成的损失分开评估，用以检验机制可解释性中的内部估计器是否足以胜任干预、控制与安全任务，并证明平均准确的估计并不必然带来更优的行动。 |
| [^144] | [Unifying Conformal Language Tasks with In-Context Ensembles](https://arxiv.org/abs/2609.03005) | 提出共形相关性框架，通过上下文学习示例筛选与集成自动构建评分函数，以最少的人工干预统一实现了多种NLP任务中覆盖率与简洁性的双重保证。 |
| [^145] | [Causal Foundation Models](https://arxiv.org/abs/2609.03003) | 本文介绍了因果基础模型（CFMs）这一新兴范式：通过预训练的神经网络，利用上下文学习即可在全新数据集上估计平均处理效应等因果量，无需为每个新问题定制流程或更新模型。 |
| [^146] | [Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation](https://arxiv.org/abs/2609.02998) | 该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。 |
| [^147] | [Evaluating Graph Neural Networks for Change-Criticality Classification in Maritime Navigation Charts](https://arxiv.org/abs/2609.02996) | 该论文提出将电子航海图数据集表示为图结构（空间对象为节点、空间与语义关系为边），并将新旧航海图间变更的重要性分类构建为图对分类问题，以评估不同图神经网络配置在此任务上的表现。 |
| [^148] | [No-Regret Bayesian Optimization with Finite-Library Input-Warped Kernels](https://arxiv.org/abs/2609.02993) | 本文提出FLIWBO方法，通过从有限光滑输入映射库中自适应选择输入扭曲来加速学习，在打破固定核函数限制的同时仍保持高概率无遗憾收敛保证，其代价仅为与库大小相关的显式 $\sqrt{N_\varepsilon}$ 项。 |
| [^149] | [TRACE: Spatiotemporal Contact Memory Graph Network Simulator for Granular Dynamics](https://arxiv.org/abs/2609.02991) | TRACE是一种图网络模拟器，通过将交互历史直接存储在接触边上的持久记忆中，并结合物理结构化解码器来预测颗粒间接触力，有效解决了颗粒动力学中接触历史难以保留的问题。 |
| [^150] | [Mesh-Native Physics-Informed Graph Surrogates for TCAD-in-the-Loop Design Space Exploration](https://arxiv.org/abs/2609.02988) | 本文提出一种直接在四面体TCAD网格上运行的物理信息图注意力网络代理模型，在每个网格节点预测静电势及电子/空穴准费米能级等漂移-扩散系统基本未知量，并通过有限体积电流连续性残差将载流子输运物理嵌入训练目标，从而实现高效的TCAD在环多目标设计空间探索。 |
| [^151] | [Tail-Likelihood Reinforcement Learning](https://arxiv.org/abs/2609.02987) | 提出TailRL方法，通过最大化策略超过随机选择的奖励阈值的对数概率来直接优化对高奖励结果的覆盖能力，使罕见的高奖励输出在梯度中获得更大权重，从而解决平均奖励优化无法衡量生成式策略产生稀有高奖励输出概率差异的问题。 |
| [^152] | [Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design](https://arxiv.org/abs/2609.02986) | 本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。 |
| [^153] | [From Euclidean to Graph-Structured Data: A Survey of Collaborative Learning](https://arxiv.org/abs/2609.02984) | 这是一篇综述论文，系统梳理了协同学习（包括联邦学习和去中心化学习）如何从规则网格结构的欧几里得数据扩展到图结构数据，指出基于消息传递机制的图学习与多智能体需要交换信息的协同环境在概念上天然契合。 |
| [^154] | [Equation Recast for Canonical Operator Learning Across Parametric PDEs](https://arxiv.org/abs/2609.02982) | 提出方程重构方法，将参数化偏微分方程的算子学习转化为单一规范算子的学习，通过解析推导吸收参数引起的算子变化到有效源项中，从而实现对新参数范围的零样本预测、外推以及融合稀疏异构数据的能力。 |
| [^155] | [Privacy Leakage in Federated Learning: Gradient-Based Client Identity Inference and Defenses for Inertial Sensing in Vehicular Edge Networks](https://arxiv.org/abs/2609.02971) | 该论文揭示了在车联网联邦学习中，服务器可从客户端上传的梯度更新中以近乎完美的准确率推断出基于IMU惯性数据的客户端身份，证实了严重的匿名性隐私泄露风险并提出了相应的防御方法。 |
| [^156] | [Learning from Scarce Labels: Multi-View Echocardiography for Ejection Fraction Prediction](https://arxiv.org/abs/2609.02969) | 该研究首次创建了公开的PLAX超声心动图射血分数预测数据集（超过25,000个标注视频），并训练出首个可复现的PLAX EF模型，其MAE达6.86%，性能与临床标准的心尖四腔方法相当。 |
| [^157] | [Privacy-Preserving Topology-Guided Safety for LLM-Based Multi-Agent Systems via Federated Graph Learning](https://arxiv.org/abs/2609.02967) | 提出FGLGuard框架，通过图联邦学习让各运营方在本地训练GNN风险检测器且仅共享模型更新，在保护私有数据的前提下实现对LLM多智能体系统的跨组织隐私保护安全防护。 |
| [^158] | [Physics-Informed Neural Network Surrogate for Oxygen Vacancy Dynamics in epitaxial $\mathrm{SrTiO_3}$ on Si memristors via Dynamic Spectral Optimization](https://arxiv.org/abs/2609.02966) | 该研究提出级联物理信息神经网络结合动态切比雪夫谱优化器的新架构，无需算子分裂即可求解条件数超过10¹⁶的硅基SrTiO₃忆阻器氧空位漂移扩散问题，并高精度重现了实验电流-电压迟滞特性。 |
| [^159] | [Statistical Feature Augmentation for Anomaly Detection in Dynamic Graphs](https://arxiv.org/abs/2609.02965) | 提出一种统计特征增强方法，将行为交互统计信息显式编码到输入特征空间中，从而提升动态图异常检测任务在多个模型和数据集上的表现。 |
| [^160] | [SurfSpec: Enhancing Off-Target-Agnostic Specificity by Bounding Pocket-Ligand Geometric Mismatch](https://arxiv.org/abs/2609.02963) | SurfSpec 通过度量并优化靶标口袋与配体之间的几何错配，利用三角不等式在完全不需要脱靶结构信息的前提下，为几何上分离的脱靶提供保守的特异性下界，从而实现脱靶结构无关的特异性感知先导化合物优化。 |
| [^161] | [The Geometry of Ignorance: LLMs Know When to Temper Bayesian Priors](https://arxiv.org/abs/2609.02959) | 研究发现大语言模型的反嵌入矩阵中存在一个编码训练语料词元分布的“无知方向”，模型通过逐词元调节该先验的强度，实现了随上下文信息增加而逐步减弱先验影响的温度调节贝叶斯更新。 |
| [^162] | [FrOGS: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical Conditions](https://arxiv.org/abs/2609.02948) | FrOGS是一种将自回归模型与连续时间马尔可夫链在单一损失下联合训练的混合离散神经采样器，可跨多种化学条件独立采样合金构型，并给出配分函数的无偏估计和热力学性质的一致性估计。 |
| [^163] | [Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis](https://arxiv.org/abs/2609.02947) | 该论文提出一种隐私保护的异构多LLM联邦推理框架，通过本地拉普拉斯噪声差分隐私和基于残差的聚合机制，使多个商用LLM API无需访问原始学生数据即可协作实现准确的认知诊断。 |
| [^164] | [LLM-Guided Reinforcement Learning for Adaptive NPC Behavior in Multi-Agent Combat Games](https://arxiv.org/abs/2609.02931) | 该论文提出一种运行时策略选择框架，利用本地部署的大语言模型（Mistral 7B）每五秒读取实时游戏状态并为预训练PPO策略的NPC分配战术标签，从而在不修改底层策略的前提下实现NPC对多样对手的自适应行为。 |
| [^165] | [Towards Scaling Reinforcement Learning to Massive Populations: Learning Mean-Field Representations](https://arxiv.org/abs/2609.02928) | 本文提出通过学习平均场表示，将强化学习扩展到大规模智能体群体，从而使大规模群体场景下的高维控制问题变得可扩展且易于处理。 |
| [^166] | [Hadronic Mono-Z Dark Matter Sensitivity with Flow Matching on CMS Open Data](https://arxiv.org/abs/2609.02923) | 该研究利用条件流匹配连续归一化流对CMS开放数据中的强子单Z事件背景进行建模，并通过哨兵值填补、固定数据划分和最小背景产额约束等措施确保评估稳健，实现了对暗物质信号的预期灵敏度分析，对三个模拟信号样本分别给出2.89σ、7.62σ和7.41σ的预期显著性。 |
| [^167] | [Evaluating GNNs for Success Prediction in Artist Collaboration Networks](https://arxiv.org/abs/2609.02920) | 本研究引入波兰音乐合作网络新数据集，与意大利、丹麦网络进行对比并合并分析，同时提出一个评估图神经网络（GNN）基于元数据和网络位置预测艺术家人气有效性的框架，发现波兰网络与三国合并网络具有相似的特性与聚类行为。 |
| [^168] | [BharatGather: A Culturally-Informed Benchmark Dataset for Misinformation and Fake News Detection in Indian Public Events](https://arxiv.org/abs/2609.02895) | 本文提出了BharatGather数据集，一个专为印度大型公共活动中虚假信息二元分类设计的文化感知基准数据集，包含14,646条通过事实核查平台爬取、多媒体转录提取与大语言模型合成增强相结合的混合流水线构建的记录。 |
| [^169] | [Improved Gradient Descent Lower Bounds Beyond Nesterov](https://arxiv.org/abs/2609.02855) | 本文证明了光滑凸优化中固定步长梯度下降的两个更强下界——非anytime的Ω(n^{-1.6342})与anytime的Ω(n^{-1.2408})，并借助silver调度可达的O(n^{-log_2(1+√2)})速率，严格分离了两种设定下可实现的收敛指数。 |
| [^170] | [FlashKAN: B-Spline KANs via Truncated Power Form](https://arxiv.org/abs/2609.01956) | FlashKAN用逼近论中的截断幂形式取代Cox-de Boor递归，通过torch.compile融合为单一GPU内核并结合有界坐标稳定化，显著加速了KAN中B样条激活函数的计算，并提供了开源软件包。 |
| [^171] | [A Mathematical Theory of Reusable Neural Bases for Network Compression](https://arxiv.org/abs/2609.01550) | 该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。 |
| [^172] | [LatentPress: Context Compression Beyond Text and Vision](https://arxiv.org/abs/2609.01507) | LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。 |
| [^173] | [Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search](https://arxiv.org/abs/2609.01431) | 本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。 |
| [^174] | [Modelpedia: A Catalog of Model Findings for the Meta-Science of AI](https://arxiv.org/abs/2609.01090) | 提出了Modelpedia——一个利用大语言模型自动从已发表论文中提取AI模型相关发现、将其与模型、数据集、方法和概念关联，并汇总为可搜索公共目录的框架，同时基于该目录对AI社区如何研究模型进行了元分析。 |
| [^175] | [MUGEN: Generating Unlearnable Graph Examples for Multiple Learning Tasks](https://arxiv.org/abs/2609.00696) | MUGEN是首个面向多种学习任务的不可学习图样本生成框架，它通过对单一干净数据集进行特征扰动，利用共享GNN编码器同时保护节点分类、图分类和链接预测等多种任务免受未经授权的模型学习。 |
| [^176] | [Singular Curvature in ReLU Training:Differentiation and the Gradient-Flow Limit Need Not Commute](https://arxiv.org/abs/2608.30960) | 该论文证明在 ReLU 训练中“先微分离散梯度下降再取极限”与“先取梯度流极限再微分”不可交换：离散精确导数收敛到无事件的区域传播子，而极限流的导数额外包含速度归一化的激活事件转移，由此产生秩一的奇异曲率差异，且全局凸性也无法将其完全抵消。 |
| [^177] | [Learning to Transfer Across Modes: Towards Unified Urban Mobility Forecasting](https://arxiv.org/abs/2608.28273) | 提出TransMod统一框架，通过构建共享的区级空间表示对齐不同空间粒度的出行系统，实现异构出行模式间的知识迁移，从而解决多模式城市出行需求预测中的空间异质性与新兴模式数据稀缺问题。 |
| [^178] | [Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090](https://arxiv.org/abs/2608.27370) | 本文提出了一种开源且成本高效的预训练配方，使在消费级RTX 5090 GPU上以极低计算成本训练出接近Qwen2.5-1.5B性能的Puro-2B模型成为可能。 |
| [^179] | [Hierarchical Channel Stacking: A Structured Decision Framework for AI-Generated Image Detection](https://arxiv.org/abs/2608.26648) | 本文提出层级通道堆叠框架，通过多阶段CNN激活的结构化表示，在AI生成图像检测中实现高精度，并揭示层级特征对决策的互补贡献。 |
| [^180] | [SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations](https://arxiv.org/abs/2608.26594) | SimCast-S2S通过潜扩散生成框架和气候模拟迁移学习，实现了高效且概率性的亚季节降水预报，解决了不确定性量化和计算成本两大核心瓶颈。 |
| [^181] | [JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution](https://arxiv.org/abs/2608.25593) | JIT-Agent通过训练一个能即时生成和优化任务自适应工具模型的系统，显著提升了现成智能体的性能，使工具设计从手动变为自动化。 |
| [^182] | [Refusal geometry reflects refusal training: diverse refusal prefixes can raise stable rank and weaken refusal vector ablation attacks](https://arxiv.org/abs/2608.25390) | 本文发现拒绝训练中的首词损失塑造了拒绝方向和子空间，且重复的拒绝前缀导致拒绝几何脆弱，但多样前缀能提升稳定性并削弱消融攻击。 |
| [^183] | [PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization](https://arxiv.org/abs/2608.24207) | 本文提出PRQ-KMeans方法，通过移除全局均值组件和基于Top-k相似性细化质心，解决了语义标识符残差量化中的三个关键局限性，从而提升了生成式检索与推荐中标记化的有效性。 |
| [^184] | [From Relaxed Indexability to Exact Indexability: A $t$-Step Approach for Partially Observable Restless Bandits](https://arxiv.org/abs/2608.24167) | 本文提出一种$t$步前瞻阈值策略，通过多步值迭代扩展了部分可观测静止多臂老虎机的Whittle指数计算，在$t=1$时恢复现有线性阈值，在$t>1$时使阈值依赖补贴，从而更精确地逼近精确可索引性。 |
| [^185] | [A Multidimensional Data-Driven Hybrid Transformer Framework for Non-invasive Continuous Blood Pressure Prediction](https://arxiv.org/abs/2608.23276) | 该论文提出一种混合Transformer框架，通过融合Transformer、Kolmogorov-Arnold网络与XGBoost的多源时序编码器和动态条件融合解码器，从ECG/PPG特征序列而非原始波形中实现无袖带的连续血压（舒张压和收缩压）预测。 |
| [^186] | [Robust Discovery of Coarse-Grained Continuum Equations from Microscopic Dynamics](https://arxiv.org/abs/2608.20404) | 本研究发现，在从微观动力学数据中发现粗粒化PDE时，数据量是影响识别稳健性的关键因素，而增大函数库则会降低发现效率，并通过相分离系统和Ising模型验证了这一点。 |
| [^187] | [DeltaMomentum: A Key-Value based Anisotropic Momentum Update via Delta Rule](https://arxiv.org/abs/2608.19491) | DeltaMomentum通过利用梯度中的键值结构，将方向感知引入动量更新规则，使每个方向以与其出现频率相关的速率被遗忘，从而无需矩阵即可实现输入侧曲率校正。 |
| [^188] | [Scale-Consistent Posterior Dynamics for Diffusion Inverse Problems](https://arxiv.org/abs/2608.15144) | 本文提出一种尺度一致的后验动力学方法，通过重标定坐标、对数信噪比组织代理和冻结目标校正器，构建可处理的连续SDE，有效解决扩散逆问题中条件分数的难解性。 |
| [^189] | [Earth observation embeddings are effective sub-grid descriptors for probabilistic weather downscaling](https://arxiv.org/abs/2608.12271) | 该论文提出利用地球观测基础模型生成的嵌入作为亚网格地表描述符，替代传统手工特征，从而提升概率天气降尺度对瞬时近地表变量的预测准确性。 |
| [^190] | [WDL-OPD: Weak-Driven On-Policy Distillation via Mixture-Constrained Co-Training](https://arxiv.org/abs/2608.09447) | 提出了WDL-OPD方法，通过锚定策略与辅助策略的双策略混合约束协同训练来稳定在线策略蒸馏的反馈回路，在Qwen3的1.7B和4B规模实验中取得了最优效果。 |
| [^191] | [Learning-Based Collaborative MEC for LLM Inference with Soft-Deadline Awareness via Transformer-Enhanced PPO](https://arxiv.org/abs/2608.02031) | 本文提出了一种基于Transformer增强PPO的协同移动边缘计算框架，结合受限的截止期扩展机制，在严格时延约束和任务依赖关系下高效调度大语言模型推理任务，从而提升服务质量。 |
| [^192] | [DrainSinkhorn: Safe Elimination for Batched Entropic Optimal Transport](https://arxiv.org/abs/2607.24741) | 提出 DrainSinkhorn，一种验证器门控的主动打包层，通过在批处理中动态淘汰已收敛的 Sinkhorn 问题来精确消除静态批处理的冗余计算，从而在完全保持 EOT 目标、Sinkhorn 映射与停止规则不变的前提下加速批量熵最优传输。 |
| [^193] | [ROMS-IMLE: A Minimalist Approach to Competitive Single-Step Generative Modelling](https://arxiv.org/abs/2607.19332) | 该论文提出ROMS-IMLE，一种极简的单步生成建模方法，通过简单地结合隐式最大似然估计（IMLE）训练目标与简洁的模型结构，摒弃扩散模型等复杂的多步渐进变换机制，依然实现了有竞争力的生成性能。 |
| [^194] | [Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models](https://arxiv.org/abs/2607.15565) | 研究揭示了视觉语言模型中“问题优先悖论”的机制——虽然前置问题能引导感知，但被数百个图像token遮挡的问题无法被答案token读取，并据此提出在图像后重复问题的“提示回声”这一无需训练的简单修复方法。 |
| [^195] | [Learning in Curved Weight Space:Exponential-Linear Weight Reparameterization for Improved Optimization](https://arxiv.org/abs/2607.09967) | 提出一种将对称指数路径与线性路径相结合的权重重参数化方法，使加性优化更新转化为与权重幅值成比例的有效变化，从而改善神经网络的优化效果。 |
| [^196] | [Resample or Reroute? Recoverable Stopping Debt Without Identified Action Selection](https://arxiv.org/abs/2607.08665) | 该论文提出“可恢复停止债务”概念并通过三个排序门槛量化弱验证器误判后的恢复潜力，实验表明固定升级到更大模型比重新采样能带来更大恢复，而常用的全回合审计统计量不包含任何可观测历史信息。 |
| [^197] | [Target-Guided Selective Reweighting for Physics-Informed Neural Network Inverse Problems: A Transfer Learning Approach](https://arxiv.org/abs/2607.05271) | 提出TGSR-PINN方法，将迁移学习与基于目标证据的神经元敏感度评分和选择性重加权相结合，解决了物理信息神经网络在偏微分方程反问题中因负迁移导致的物理参数恢复不准确问题。 |
| [^198] | [What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation](https://arxiv.org/abs/2607.04726) | 论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。 |
| [^199] | [KARMA: Knowledge graph-based Automated Reasoning Materialization and Alignment](https://arxiv.org/abs/2607.03166) | KARMA 通过在领域知识图谱上枚举模式约束路径生成槽位对齐的对比候选样本，并利用槽位并行对齐（SPA）将偏好监督精准路由至区分性实体槽位，从而解决了基于模板的对比合成中的分辨率不匹配问题。 |
| [^200] | [SNAP-FM: Sparse Nonlinear Accelerated Projection for Physics-Constrained Generative Modeling](https://arxiv.org/abs/2607.00095) | 提出SNAP-FM方法，利用样本批处理与局部PDE耦合所诱导的块稀疏结构，实现高效的批量非线性投影优化，使生成模型在推理时能精确满足物理守恒约束且计算开销大幅降低。 |
| [^201] | [Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs](https://arxiv.org/abs/2606.31413) | 提出Hard-Routed MoR-LoRA两阶段框架，通过单位尺度的硬top-1路由（而非软加权组合）选择冻结的推理LoRA专家，仅训练轻量级共享路由器和小型注意力LoRA即可实现多领域推理能力的集成。 |
| [^202] | [Transformers as Bayesian In-Context Experimenters: Smoothness-Adaptive Efficient ATE Estimation](https://arxiv.org/abs/2606.31184) | 该论文提出将变换器训练为模仿贝叶斯后验Neyman教师的“上下文实验者”，通过上下文学习摊销序贯方差估计与处理分配过程，实现对平均处理效应的平滑度自适应高效估计。 |
| [^203] | [Democratic ICAI: Debating Our Way to Steering Principles from Preferences](https://arxiv.org/abs/2606.28294) | 本文提出民主化逆向宪法AI（DICAI），通过结构化角色辩论收集多元竞争性推理依据，从中提炼出更清晰全面的引导原则，以改进偏好决策建模和下游模型训练。 |
| [^204] | [Physics-Guided Robotic Radiation Source Localization along Arbitrary Measurement Paths in Unstructured Environments](https://arxiv.org/abs/2606.27624) | 该论文提出了一种基于物理信息机器学习的自动化框架，使机器人能够在未知非结构化环境中沿任意测量路径精确估计辐射源位置，无需接近辐射源，从而降低辐射损害风险并提升任务部署的灵活性。 |
| [^205] | [Real vs. Complex Spectral Bases for Neural Operators: The Role of Green's Function Alignment](https://arxiv.org/abs/2606.24851) | 本文提出哈特莱神经算子（HNO），用纯实数的离散哈特莱变换替代复数FFT，在与FNO参数量相同的情况下消除共轭对称冗余，并论证最佳频谱基由算子格林函数的性质决定（自伴椭圆算子适合实数基）。 |
| [^206] | [Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations](https://arxiv.org/abs/2606.23129) | 该论文提出将神经元激活建模为正弦强迫阻尼谐振子的稳态响应，通过联合优化振子参数与网络权重实现自适应频谱门控，无需显式正则化即可解决隐式神经表示中细节捕捉与噪声抑制之间的频谱权衡问题。 |
| [^207] | [Structured Inference with Large Language Gibbs](https://arxiv.org/abs/2606.19264) | 提出了Large Language Gibbs方法，将LLM的条件分布作为MCMC的转移算子，通过在其他变量条件下迭代重采样单个变量来实现结构化概率推断，从而避免了自回归生成中的顺序依赖偏差。 |
| [^208] | [Explicit Interaction Architectures for Dynamical Learning: A Controlled Study of Structural Inductive Bias](https://arxiv.org/abs/2606.19101) | 该论文提出由有序局部状态调制变换构成的因果循环单元，并通过受控实验检验这种显式设计的交互架构相比通用回声状态网络能否为动态学习提供有用的结构归纳偏置。 |
| [^209] | [LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents](https://arxiv.org/abs/2606.18388) | LLMZero利用大语言模型智能体结合树搜索，通过在每个检查点诊断训练状态来自适应地优化RL后训练的多参数调度策略，在四个GRPO任务上比基础模型提升9%-140%、比网格搜索提升6%-15%，并揭示了容量参数单调累积、正则化参数震荡变化的训练规律。 |
| [^210] | [ThousandWorlds: A benchmark for climate emulation of potentially habitable exoplanets](https://arxiv.org/abs/2606.18338) | ThousandWorlds是一个机器学习就绪的系外行星气候模拟基准数据集，包含来自五个全球气候模型的约1800次模拟，旨在突破传统气候模拟的计算瓶颈，加速对潜在宜居系外行星大气的理解与生命信号解读。 |
| [^211] | [SPACR: Single-Pass Adaptive Training of Uncertainty-Aware Conformal Regressors](https://arxiv.org/abs/2606.10734) | SPACR提出了一种基于可微损失的单遍训练方法，无需数据集分割或预定义置信水平即可联合优化准确性、效率与有效性，使单个模型能够在多个置信水平下生成有效且更窄的预测区间，性能优于标准保形预测和DOICR。 |
| [^212] | [GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators](https://arxiv.org/abs/2606.08343) | GENERIC-FNO是首个在函数空间中嵌入非平衡热力学完整GENERIC结构的神经算子，通过构造精确满足简并条件而无需任何惩罚项，实现了机器精度下的能量守恒与熵产生。 |
| [^213] | [Repetition Mismatch: Why Data Mixture Experiments Don't Scale and How to Fix Them](https://arxiv.org/abs/2606.07597) | 论文揭示了预训练数据混合实验无法从小规模外推到大规模的主要原因是高质量数据的重复率随训练预算变化而改变最优混合比例，并提出通过匹配目标重复率的子采样方法，仅用1/16的目标token即可恢复接近最优的数据混合配置。 |
| [^214] | [Shortcuts in the Tail: Debiasing via Post-Hoc Spectral Compression of Fine-Tuning Updates](https://arxiv.org/abs/2606.07596) | 对微调权重更新的SVD进行简单的事后尾部截断，无需重新训练、群体标签或反事实数据，即可在几乎不损失准确率的情况下显著减少模型对虚假捷径的依赖，实现对代表性不足群体的去偏。 |
| [^215] | [TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment](https://arxiv.org/abs/2606.07451) | TEVI框架利用稀疏自编码器解耦图像嵌入，并通过文本条件化的掩码模块只保留与文本描述相符的信息、剔除多余内容，从而改善CLIP等视觉-语言模型中图像-文本嵌入的对齐问题。 |
| [^216] | [AIP: A Graph Representation for Learning and Governing Agent Skills](https://arxiv.org/abs/2606.04781) | 该论文提出智能体指令协议（AIP），将智能体技能建模为由模式验证的YAML规范治理的有向执行图，从而同时提升智能体在重实现任务上的可靠性和技能创建与改进的效率。 |
| [^217] | [Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models](https://arxiv.org/abs/2606.03780) | 该研究将激活修补方法细化到专家层面，首次证明在稀疏MoE语言模型（如Qwen3-30B-A3B-Base）中，事实回忆的恢复可定位于单个路由专家（L44E069），但这种专家级定位在不同模型间并不一致。 |
| [^218] | [MidSurfNet: Learning Face Pairing for Mid-surface Abstraction of Thin-walled CAD Models](https://arxiv.org/abs/2606.01891) | 提出MidSurfNet，通过融合几何流与属性拓扑流双证据的学习式面对评分器和确定性面组构成，实现薄壁CAD模型中面抽象的鲁棒面配对，克服了传统手工阈值方法难以适应多局部壁厚且结果不一致的缺陷。 |
| [^219] | [HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization](https://arxiv.org/abs/2605.29843) | 提出HARP，一种可学习的Hadamard预条件自适应旋转处理器，通过稀疏蝶形块正交结构替代固定Hadamard变换，在保持与全精度模型精确等价的同时，自适应地适应层、校准分布和量化器，从而提升极端低比特LLM量化的鲁棒性。 |
| [^220] | [Theoretical Foundations and Effective Algorithms for Policy-Aware Simulator Learning](https://arxiv.org/abs/2605.29032) | 本文提出将模拟器学习目标从预测准确性转向策略鲁棒性，通过模型与对抗策略之间的零和极小极大博弈框架，证明了该博弈具有次线性遗憾界的可学习性，并建立了寻找最坏情况策略与以评论家误差为奖励的标准强化学习问题之间的误差-MDP对偶理论。 |
| [^221] | [RW-TTT: Batched Serving for Request-Owned Test-Time Training State](https://arxiv.org/abs/2605.28053) | RW-TTT通过为每个解码步骤标记所有者、版本和读写效果，实现了对请求私有测试时训练状态的安全高效批处理服务，在相同内存预算下相比串行服务获得9.31倍加速。 |
| [^222] | [The Timing Dependencies of Trust: Speed, Accuracy, and cBCI Neuro-Decoupling in Human-AI Teams](https://arxiv.org/abs/2605.25868) | 该研究证明AI响应时机决定了人机团队的失败机制——快速AI诱发即时盲目服从，使人类受骗时的准确率骤降至50.2%，而缓慢且准确的AI则通过神经解耦机制改善协作脑机接口团队的协同表现。 |
| [^223] | [Towards Affordable Energy: A Gymnasium Environment for Electric Utility Demand-Response Programs](https://arxiv.org/abs/2605.12462) | 本文提出了DR-Gym，一个开源的、与Gymnasium兼容的在线仿真环境，用于从电力公司视角训练和评估需求响应策略，解决了离线历史数据无法捕捉价格信号与用户行为之间动态交互反馈循环的问题。 |
| [^224] | [A Real-Calibrated Synthetic-First Data Engine](https://arxiv.org/abs/2605.09699) | 提出了一个结合可控扩散生成与多阶段筛选过滤的模块化数据工程框架，通过系统化的数据集构建提升低数据量场景下合成数据增强的实际可靠性。 |
| [^225] | [Observation-Aligned Two-Stage Domain Decomposition for Physics-Informed Traffic State Estimation with Sparse Fixed Sensors](https://arxiv.org/abs/2605.08028) | 提出了观测对齐的两阶段域分解物理信息神经网络TSDD-PINN，利用全局父网络的残差剖面指导确定性域分解与子网络热启动，解决了PINN在稀疏固定传感器交通状态估计中过度平滑LWR模型允许的剧烈状态转变的问题，其中空间细化在精度与训练效率之间取得最佳平衡。 |
| [^226] | [Selfie-Capture Dynamics as an Auxiliary Signal Against Deepfakes and Injection Attacks for Mobile Identity Verification](https://arxiv.org/abs/2605.00218) | 本文提出CanSelfie多传感器数据集，证明自拍捕捉过程中记录的被动运动轨迹可作为辅助证据信号，有效筛查深度伪造和视频注入攻击，为移动远程身份验证提供了超越传统摄像头检测的新证据渠道。 |
| [^227] | [RCProb: Probabilistic rule extraction from classification tree ensembles](https://arxiv.org/abs/2604.25304) | RCProb是一种针对RuleCOSI+的概率扩展方法，通过平滑的原子类条件证据和支持自适应混合概率估计，显著提升了从分类树集成中提取规则的概率可靠性，将对数损失大幅降低。 |
| [^228] | [When Chain-of-Thought Fails, the Solution Hides in the Hidden States](https://arxiv.org/abs/2604.23351) | 研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。 |
| [^229] | [Towards Universal Tabular Embeddings: A Benchmark Across Data Tasks](https://arxiv.org/abs/2604.21696) | 该论文提出了TEmBed统一基准，首次在单元格、行、列和表格四个表示层级上系统评估表格嵌入模型，发现模型优劣取决于任务和表示层级，为实际应用中的模型选择提供了实用指导。 |
| [^230] | [Learning to Concatenate Quantum Codes](https://arxiv.org/abs/2604.14931) | 该论文提出一种逐级自适应的量子码串联方法：先估计每级后的有效噪声信道，在噪声结构化时使用学习型小型非加性编码器、在噪声趋于均匀时切换到标准码，从而在强结构化噪声下将所需量子比特数最多减少两个数量级。 |
| [^231] | [Safety Training Modulates Harmful Misalignment Under On-Policy RL, But Direction Depends on Environment Design](https://arxiv.org/abs/2604.12500) | 本研究通过对11个不同规模的指令微调大语言模型在3种环境中进行在线策略强化学习训练，揭示了模型规模是否成为安全缓冲取决于环境设计，并发现在线策略RL能保留模型生成分布中固有的安全缓冲，而离线策略设置会绕过这一缓冲。 |
| [^232] | [Sliding-Window Reordering with Overlap Averaging: A Simple Time-Domain Augmentation for Multivariate Forecasting](https://arxiv.org/abs/2604.09067) | 提出一种简单的时域数据增强方法，通过随机重排滑动窗口并对重叠部分取平均来重建序列，在限制时间失真的前提下生成受控变化的合成样本，并在九个长期预测基准和四个短期交通基准上显著优于现有增强方法。 |
| [^233] | [Towards Lifelong Aerial Autonomy: Geometric Memory Management for Continual Visual Place Recognition in Dynamic Environments](https://arxiv.org/abs/2604.09038) | 该论文将空中视觉位置识别建模为基于任务的域增量学习问题，提出结合静态卫星样本记忆与有界重放缓冲区的异构记忆框架及DBS-Hybrid混合样本选择策略，以应对环境变化引起的灾难性遗忘，实现动态环境下长期空中自主的鲁棒地理定位。 |
| [^234] | [A Comparative Study in Surgical AI: Potential and Limitations of Data, Compute, and Scaling](https://arxiv.org/abs/2603.27341) | 本文比较研究了数据、算力与规模化在外科AI中的潜力与局限，探讨现代通用AI能否以及在多大程度上辅助外科实践。 |
| [^235] | [Identification of Bivariate Causal Directionality Based on Anticipated Asymmetric Geometries](https://arxiv.org/abs/2603.26024) | 本文提出了两种基于条件分布的新方法——预期不对称几何（AAG）和单调性指数（MI），用于识别二元数值数据中的因果方向性。 |
| [^236] | [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440) | 提出了KernelFoundry演化框架，通过MAP-Elites质量多样性搜索、提示与内核共同演化的元提示机制以及基于模板的参数优化，实现硬件感知的高效GPU内核自动优化。 |
| [^237] | [Simplify to Amplify: Achieving Information-Theoretic Bounds with Fewer Steps in Spectral Community Detection](https://arxiv.org/abs/2602.17104) | 本文提出一种简化的谱社区检测算法，通过去除不必要的预处理步骤并利用邻接矩阵第二特征向量的特性，在双社区随机块模型中实现了接近信息论极限的误差界，超越了现有方法。 |
| [^238] | [Loss Knows Best: Detecting Annotation Errors in Videos via Loss Trajectories](https://arxiv.org/abs/2602.15154) | 本文提出以不相交参考集训练的检查点上的累积样本损失（CSL）作为动态指纹，实现对视频数据集中语义误标和时间乱序两类标注错误的样本外自动审计。 |
| [^239] | [FedPS: Federated Preprocessing for structured data via aggregated Statistics](https://arxiv.org/abs/2602.10870) | 提出了FedPS框架，利用数据草图技术在联邦环境下通过聚合统计实现结构化数据的高效预处理（包括特征缩放、编码、离散化和缺失值插补），解决了联邦学习中预处理阶段被忽视的问题。 |
| [^240] | [Entropy-Generated Attention Beyond Softmax and Entmax: Kaniadakis and Reciprocal-Symmetric Abe Operators](https://arxiv.org/abs/2602.08216) | 该论文从广义统计熵出发推导出两种新型注意力算子——基于Kaniadakis熵的幂律衰减全支撑归一化算子和基于Abe熵的倒数对称算子，并证明这些平稳分布律源于概率单纯形上的Fisher度量拉格朗日量，从而将Softmax注意力统一推广到超越指数衰减与精确截断的框架。 |
| [^241] | [F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare](https://arxiv.org/abs/2602.06717) | 本文提出 F-GRPO，借鉴 Focal loss 设计了难度感知的缩放系数，对高成功率采样组的更新降权，从而防止 RLVR 训练中的策略因组采样遗漏稀有正确解而过度集中于常见解。 |
| [^242] | [Temperature Scaling Attack Disrupting Model Confidence in Federated Learning](https://arxiv.org/abs/2602.06638) | 该论文提出了温度缩放攻击（TSA），一种新型联邦学习训练时攻击，通过学习率-温度耦合机制在保持模型准确率不变的情况下破坏置信度校准，从而威胁依赖置信度信号的任务关键型系统的风险决策逻辑。 |
| [^243] | [Deep networks learn to parse uniform-depth context-free languages from local statistics](https://arxiv.org/abs/2602.06065) | 该研究引入了一类可调节歧义程度和跨尺度相关结构的概率上下文无关文法，揭示了深度网络能够仅从局部统计特征中学习解析语言的层次结构。 |
| [^244] | [MSign: An Optimizer Preventing Training Instability in Large Language Models via Stable Rank Restoration](https://arxiv.org/abs/2602.01734) | 提出MSign优化器，通过周期性矩阵符号运算恢复权重矩阵稳定秩，以低于7.0%的计算开销有效防止大语言模型预训练中的梯度爆炸与训练崩溃。 |
| [^245] | [Non-Stationary Functional Bilevel Optimization](https://arxiv.org/abs/2601.15363) | 提出首个面向非平稳函数双层优化的算法SmoothFBO，通过时间平滑的随机超梯度估计器降低方差，实现具有次线性遗憾的稳定更新，并在非平稳超参数优化和基于模型的强化学习中优于现有方法。 |
| [^246] | [Linearized subspace refinement framework to expose hidden accuracy in trained neural networks](https://arxiv.org/abs/2601.13989) | 提出了一种与架构无关的训练后框架LSR，通过在雅可比矩阵定义的低维子空间中求解约化最小二乘问题对已训练神经网络进行线性化修正，从而突破梯度训练导致的精度平台期，显著提升预测精度。 |
| [^247] | [Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models](https://arxiv.org/abs/2601.08955) | 提出了ITP统一框架，让智能体策略模型与世界模型交互生成多步想象轨迹，并通过权衡最终目标与任务进度的自适应前瞻机制，充分释放世界模型在复杂任务规划中的潜力。 |
| [^248] | [Active learning for data-driven reduced models of parametric differential systems with Bayesian operator inference](https://arxiv.org/abs/2601.00038) | 该论文提出了一种基于贝叶斯算子推断的主动学习框架，通过量化预测不确定性来智能选择训练参数，从而以最少的数据成本提升参数化动力系统数据驱动降阶模型的精度。 |
| [^249] | [DuaDeep-SeqAffinity: Dual-Branch Deep Learning for Tri-Stream Sequence-Based Antibody--Antigen Affinity Prediction](https://arxiv.org/abs/2512.22007) | 该论文提出仅基于氨基酸序列的双分支三流深度学习框架DuaDeep-SeqAffinity，通过对三条序列流分别进行冻结ESM-2嵌入并经Transformer与CNN并行分支后期融合，直接预测抗体-抗原结合亲和力，无需三维结构信息，并在AbRank基准上显著优于单分支模型。 |
| [^250] | [FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation](https://arxiv.org/abs/2512.15116) | 提出FADTI框架，通过傅里叶偏置投影（FBP）模块在扩散去噪过程中注入可学习的频率感知偏置，并支持DFT、STFT、FSST多种频谱变换，从而有效提升多变量时间序列插补对周期性和非平稳模式的恢复能力。 |
| [^251] | [Adaptive Partitioning and Learning for Stochastic Control of Diffusion Processes](https://arxiv.org/abs/2512.14991) | 本文提出一种自适应划分状态-动作空间的基于模型的强化学习算法，用于求解无界连续状态空间下受控扩散过程的随机控制问题，并建立了包含新定义的“缩放维度”的遗憾界。 |
| [^252] | [Mixed Data Clustering Survey and Challenges](https://arxiv.org/abs/2512.03070) | 本文提出了一种基于预拓扑空间的混合数据聚类方法，能够有效处理同时包含数值型和分类型变量的异构数据，并提供层次化、可解释的聚类结果。 |
| [^253] | [Attention Trajectories as a Diagnostic Axis for Deep Reinforcement Learning](https://arxiv.org/abs/2511.20591) | 提出一种基于显著性图定量分析的诊断框架，通过构建分层注意力画像与注意力轨迹，揭示深度强化学习智能体的算法特异性注意力偏差、非预期奖励驱动策略及冗余感官通道过拟合问题。 |
| [^254] | [Reliable Selection of Heterogeneous Treatment Effect Estimators](https://arxiv.org/abs/2511.18464) | 提出了一种无需真实处理效应数据的异质性处理效应估计器选择方法，通过交叉拟合指数加权检验统计量和双向样本分割实现渐近族错误率控制，在多个基准数据集上显著减少错误选择。 |
| [^255] | [AnyBox: Efficient Zero-Shot 9DoF Pose Estimation of Boxes for Robotic Manipulation](https://arxiv.org/abs/2511.15884) | AnyBox是一个高效的零样本框架，通过利用箱体的几何规则性，在单张RGB-D观测上交替进行位姿与尺度估计，从而实现对杂乱遮挡环境中箱体9自由度位姿（6D位姿+3D尺寸）的联合恢复，无需物体特定的CAD模型。 |
| [^256] | [A Nesterov-Accelerated Byzantine-Robust Federated Learning](https://arxiv.org/abs/2511.02657) | 该论文提出了一种将Nesterov动量与拜占庭鲁棒聚合规则无缝结合的联邦学习算法Byrd-NAFL，在非凸光滑损失函数和宽松假设下建立了有限时间收敛保证，同时兼顾通信效率与对拜占庭攻击的抵抗能力。 |
| [^257] | [From Leakage to Fidelity: Reliable Benchmarking for Temporal Cascade Prediction](https://arxiv.org/abs/2510.25348) | 该论文揭示时序级联预测现有评估中普遍存在数据泄漏问题，提出以完全时序协议为核心的评估标准与泄漏诊断方法，并发布包含丰富特征与购买转化标签的真实电商级联数据集 Taoke，推动该领域迈向保真感知的可靠基准测试。 |
| [^258] | [WELD: The First Naturalistic Long-Period Small-Team Workplace Emotion Dataset for Ubiquitous Affective Computing](https://arxiv.org/abs/2510.15221) | WELD是首个结合数年持续时间、自然职场情境、稳定小团队结构与完全被动感知协议的职场情绪数据集，基于中国某软件公司49名员工超过30个月的面部表情数据构建。 |
| [^259] | [Finite-Time Convergence of Single-Trajectory Chi-Square Robust Q-Learning With Linear Function Approximation](https://arxiv.org/abs/2510.01721) | 本文针对使用χ²不确定集和线性函数逼近的单轨迹鲁棒Q学习，通过变分重形式化和分块冻结目标方案，克服了条件二阶矩平方根无法无偏估计以及投影算子非压缩的难题，首次对所有折扣因子建立了相对于最优鲁棒Q函数的有限时间误差界。 |
| [^260] | [Parameterized Hardness of Zonotope Containment and Neural Network Verification](https://arxiv.org/abs/2509.22849) | 本工作解决了Froese等人提出的神经网络验证参数化复杂性开放问题，证明了对所有ℓ≥2，以输入维度为参数时判定ℓ层ReLU网络计算函数的正性（及满射性）是W[ℓ-1]-难的，并由此推出Zonotope非包含问题是W[1]-难的。 |
| [^261] | [Data-efficient Kernel Methods for Learning Hamiltonian Systems](https://arxiv.org/abs/2509.17154) | 该论文提出从轨迹数据直接学习哈密顿系统的核方法（含一步法和两步法），在数据稀缺场景下实现高效精确预测并保持哈密顿结构，同时提供了先验误差估计以保证模型的可靠性。 |
| [^262] | [Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications](https://arxiv.org/abs/2508.00669) | 本文是首个针对大语言模型医学推理领域的系统综述，提出了涵盖训练时策略与测试时机制的推理增强技术分类体系，并系统分析了这些技术在多种数据模态和临床应用中的实践与评估方法。 |
| [^263] | [ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition](https://arxiv.org/abs/2506.10226) | 提出ScoreMix方法，利用扩散模型的分数可组合性、在无需外部模型或数据集的情况下，通过混合判别器嵌入空间中相距较远的类别生成分数条件合成样本，为识别任务带来最高3%的平均性能提升。 |
| [^264] | [Adaptive Resolving Methods for Markov Decision Processes with Function Approximations](https://arxiv.org/abs/2505.12037) | 本文提出一种基于线性规划重构的自适应重解算法，在新转移样本到达时反复重解约简线性系统，从而在带函数逼近的MDP中实现了与实例相关的 $\widetilde O(C_{\mathrm{inst}}/N)$ 目标缺口与约束残差的高效求解。 |
| [^265] | [Semiparametric Inference for Counterfactual Regression under Intervention-Driven Shift](https://arxiv.org/abs/2504.02694) | 提出了一个沿增量干预路径进行反事实回归的半参数推断框架，通过交叉拟合影响函数方法建立了优化器的一致性、稳定性和渐近有效推断，并能构建同步置信带。 |
| [^266] | [Learning Constraints-Based Adaptive Hypergraph Neural Networks for Solving Vehicle Routing Problems](https://arxiv.org/abs/2503.10421) | 提出一种将面向约束的超图与强化学习相结合的端到端框架，通过动态超边重构策略有效处理车辆路径问题中的复杂硬约束。 |
| [^267] | [LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models](https://arxiv.org/abs/2503.03313) | 该论文提出PromptGFM，通过图词表学习将图节点融入语言模型词表空间，克服了现有LLM与GNN解耦架构及OOV token带来的跨图、跨任务迁移难题，构建了面向文本属性图的多功能图基础模型。 |
| [^268] | [AgentRM: Enhancing Agent Generalization with Reward Modeling](https://arxiv.org/abs/2502.18407) | 本论文提出可泛化奖励模型AgentRM，发现微调奖励模型来引导测试时搜索比直接微调策略模型更稳健，在九个智能体任务上平均提升8.8分并超越最强通用智能体4.0分。 |
| [^269] | [A cautionary tale on the cost-effectiveness of collaborative AI in real-world medical applications](https://arxiv.org/abs/2412.06494) | 本文通过对7个医学数据集、3种机器学习任务和8种数据模态的大规模基准测试，系统比较了联邦学习与基于共识的学习方法在多中心医学数据分析中的准确性和成本效益，为协作式AI在真实医疗场景中的部署提供了警示性见解。 |
| [^270] | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) | 本文揭示了语法约束解码会扭曲大语言模型的输出分布，导致生成结果虽符合语法但质量低下，并提出了一种名为ASAp的语法对齐解码算法来解决这一问题。 |
| [^271] | [Uncertainty Quantification in Machine Learning for Biosignal Applications -- A Review](https://arxiv.org/abs/2312.09454) | 本文系统综述了不确定性量化在脑电图、心电图、眼电图、肌电图等生物信号机器学习应用中的现有方法、应用场景、评估手段与不确定性度量，旨在提升医学场景下预测的可解释性与鲁棒性。 |
| [^272] | [Anisotropic View Distance Metric for High-Dimensional Data: Theory, Geometry, and Fast Computation](https://arxiv.org/abs/2206.05215) | 提出了一种受正交投影启发的新距离度量——视图距离，通过将样本空间投影到n(n-1)/2个二维平面并求欧氏距离之和，严格满足度量公理，有效解决了高维数据中各向异性结构和冗余特征导致欧氏距离失效的问题。 |

# 详细

[^1]: 训练式编译：将自然语言规范转化为本地神经函数

    Compile by Training: Turning Natural-Language Specifications into Local Neural Functions

    [https://arxiv.org/abs/2609.04199](https://arxiv.org/abs/2609.04199)

    提出“训练式编译”方法，将自然语言规范编译为可复用的本地神经函数，通过教师模型生成的示例训练小型适配器，无需每次调用远程大模型即可达到83.6%的语义准确率。

    

    许多反复出现的文本功能很容易描述，但难以用规则来实现，而每次输入都调用大型远程模型会带来重复的成本、延迟以及对服务提供商的依赖。我们提出了“训练式编译”，它将自然语言规范转化为可复用的神经函数。在编译阶段，教师模型生成任务特定的示例，用于为一个紧凑的解释器训练小型适配器。生成的函数无需教师模型即可运行，并且可以像普通软件一样进行存储、版本管理和组合。在FuzzyBench-Hard（一个Program-as-Weights快速编译器无法产生精确匹配的子集）上，训练式编译达到了83.6%的语义准确率。这一更高的准确率伴随着更高的编译时间成本：大约需要一分钟，而快速编译器只需几秒钟。我们将该编译器部署在一个公开的交互式服务中，并在多站点网站上展示了编译后的函数。

    arXiv:2609.04199v1 Announce Type: cross  Abstract: Many recurring text functions are easy to describe but difficult to implement with rules, while calling a large remote model for every input introduces repeated cost, latency, and dependency on a provider. We present compile by training, which turns a natural-language specification into a reusable neural function. At compile time, teacher models generate task-specific examples that are used to train a small adapter for a compact interpreter. The resulting function runs without the teachers and can be stored, versioned, and composed like ordinary software. On FuzzyBench-Hard, a subset on which the Program-as-Weights fast compiler produced no exact matches, compile by training reaches 83.6% semantic accuracy. This higher accuracy comes with a higher compile-time cost: roughly a minute rather than seconds for the fast compiler. We deploy the compiler in a public interactive service and demonstrate compiled functions in a multi-site websit
    
[^2]: 洁净的工程，不稳定的测量：黑盒大语言模型观测者在共享端点上预注册的可靠性失败

    Clean Engineering, Unstable Measurement: A Preregistered Reliability Failure of Black-Box LLM Observers on Shared Endpoints

    [https://arxiv.org/abs/2609.04198](https://arxiv.org/abs/2609.04198)

    本文通过两项预注册审计（共52,988次请求尝试）发现，黑盒大语言模型评判者作为测量仪器存在严重可靠性缺陷——即便工程执行记录完美，相同请求的重复排名一致性仅0.400、字节级相同的次日重放一致性仅0.78，远低于0.90和0.99的预设标准，其根源在于标签映射偏置、信号低于噪声底多个数量级以及逐字排列读数放大的噪声。

    

    语言模型评判者如今负责把关训练数据、为生成内容打分并驱动排行榜。评判者因而成为一种测量仪器，其建立在一个很少被言明的假设之上：同一请求发送给同一模型名称，明天读出的结果应当相同。我们在两项预注册活动中对该假设进行了审计，所有阈值均事先固定；结果两项活动都未能通过仪器验证这一步。在52,988次经审计的请求尝试中，同一窗口内重复排名的一致性仅为Spearman 0.400（要求为0.90），字节级完全相同的次日重放一致性为0.78（要求为0.99），而与此同时每次执行的工程记录都达到了完美水平。三种机制解释了这一差距：标签到含义的映射对读数造成的偏置与信号本身同样强；候选者之间的差距低于仪器自身噪声底达七个数量级；以及字节级完全相同的输入返回不同排名——这种噪声会被精确排列读数方式进一步放大。无论是指标替代（摘要在此处截断）……

    arXiv:2609.04198v1 Announce Type: new  Abstract: Language-model judges now gate training data, score generations, and drive leaderboards. The judge is then a measurement instrument, resting on one rarely stated assumption: the same request, sent to the same model name, reads the same tomorrow. We audited that assumption in two preregistered campaigns with every threshold fixed in advance; neither got past validating its instrument. Across 52,988 audited request attempts, same-window repeat rankings agreed at Spearman 0.400 against a required 0.90, and byte-identical next-day replays agreed at 0.78 against a required 0.99, each time with the execution record at ceiling. Three mechanisms explain the gap: a label-to-meaning mapping that biased readouts as strongly as the signal; candidate gaps seven orders of magnitude below the instrument's own noise floor; and byte-identical inputs returning different rankings, a noise that exact-permutation readouts compound. Neither metric substitutio
    
[^3]: 可读性不等于可解释性：比较思维链推理中被评判的重要性与实际重要性

    Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-Of-Thought Reasoning

    [https://arxiv.org/abs/2609.04194](https://arxiv.org/abs/2609.04194)

    本研究将思维链推理步骤的重要性量化为蒙特卡洛模拟估计的“优势”，发现LLM评判器虽能超越简单基线但远不足以准确识别真正重要的推理步骤，表明推理文本的可读性并不等于可解释性。

    

    来自思维链模型的推理轨迹似乎为了解模型如何得出答案提供了一个清晰可读的窗口。越来越多的研究正是这样对待它们，使用LLM评判器来诊断错误、评估忠实性，并通过过程奖励模型和生成式批评家提供步骤级监督。这些做法依赖于推理步骤的文本能够承载关于其功能作用的信息。但文本实际上是否编码了哪些推理步骤真正重要的信息？我们将推理步骤的重要性操作化为其“优势”：即包含该步骤后期望奖励（例如产生正确的最终答案）的变化，并通过蒙特卡洛模拟进行估计。以这些估计作为真值，我们评估了LLM评判器能否识别高优势步骤，发现足够强大的LLM能够超越流行率基线，但远低于噪声上限。将模型微调为步骤级……

    arXiv:2609.04194v1 Announce Type: new  Abstract: Reasoning traces from chain-of-thought models appear to offer a legible window into how a model arrives at its answer. A growing body of work treats them as such, using LLM judges to diagnose errors, evaluate faithfulness, and provide step-level supervision via process reward models and generative critics. These practices rely on the text of a reasoning step carrying information about its functional role. But does the text actually encode information about which reasoning steps matter? We operationalize the importance of a reasoning step as its advantage: the change in expected reward, e.g., producing the correct final answer, from including that step, estimated via Monte Carlo rollouts. Basing ground truth on these estimates, we evaluate whether LLM judges can identify high-advantage steps and find that sufficiently capable LLMs can outperform a prevalence baseline but fall well short of a noise ceiling. Fine-tuning a model as a step-le
    
[^4]: 并发随机博弈的鲁棒PAC学习

    Robust PAC Learning of Concurrent Stochastic Games

    [https://arxiv.org/abs/2609.04189](https://arxiv.org/abs/2609.04189)

    该论文提出了首个针对具有转移不确定性的广义和并发随机博弈的PAC学习框架，通过引入纳什裕度刻画解决了均衡存在性问题，能在多项式样本复杂度下返回社会福利近优的ε-近似纳什均衡或证明精确纳什均衡不存在。

    

    我们提出了首个针对具有转移不确定性的广义和并发随机博弈的概率近似正确（PAC）学习框架，同时解决了纳什均衡存在性这一难题。我们的算法在转移核上维护数据驱动的 $L^1$ 置信集，并求解鲁棒CSG以计算社会福利最优的 $\varepsilon$-纳什均衡，同时使用基于鲁棒MDP的探索机制来驱动联合状态-动作的覆盖。至关重要的是，我们引入了纳什裕度刻画，使得能够对均衡存在性进行有原则的推理：该框架要么返回一个其社会福利值与最优值之差在 $\varepsilon$ 以内的 $\varepsilon$-近似纳什均衡，要么提供一个不存在精确纳什均衡的可靠证明。在相关状态-动作对满足最小可达性条件 $p_{\mathrm{reach}}>0$ 的情况下，算法在多项式数量的轨迹样本后即可终止，样本（复杂度具有…保证）

    arXiv:2609.04189v1 Announce Type: new  Abstract: We introduce the first Probably Approximately Correct (PAC) learning framework for general-sum concurrent stochastic games (CSGs) with transition uncertainty, while addressing the challenge of Nash equilibrium (NE) existence. Our algorithm maintains data-driven $L^1$ confidence sets over transition kernels and solves a robust CSG to compute a social-welfare optimal $\varepsilon$-NE, using a robust MDP-based exploration mechanism to drive joint state-action coverage. Crucially, we introduce a Nash margin characterisation that enables principled reasoning about equilibrium existence: the framework either returns an $\varepsilon$-approximate NE whose social-welfare value is $\varepsilon$-close to optimal, or provides a sound certificate that no exact NE exists. Under a minimum reachability condition $p_{\mathrm{reach}}>0$ over relevant state-action pairs, the algorithm terminates after a polynomial number of trajectory samples, with sample 
    
[^5]: Para-Pipe：在片上系统上利用机器学习计算图的分层算子并行性

    Para-Pipe: Exploiting Hierarchical Operator Parallelism of ML Computational Graphs on SoCs

    [https://arxiv.org/abs/2609.04168](https://arxiv.org/abs/2609.04168)

    本文提出了Para-Pipe，一个面向片上系统的分层映射框架，通过在流水线架构中集成阶段内和阶段间算子并行性，在吞吐量与推理延迟之间实现更好的权衡，从而降低边缘深度学习应用的推理延迟。

    

    随着基于边缘的深度学习应用日益复杂，在异构片上系统（SoCs）上优化性能带来了独特的挑战。传统的流水线技术将计算分配到不同的片上处理单元上，虽然对吞吐量有效，但无法满足具有复杂相互依赖关系和广泛算子并行性的现代神经网络所带来的延迟需求。利用算子并行性以在多个处理单元上实现并发执行，从而降低推理延迟，具有潜在的价值。然而，优先考虑流水线或并行执行往往需要做出妥协，即优化一个性能指标会对另一个产生不利影响。本文提出了Para-Pipe，这是一个分层映射框架，它在流水线架构中集成了阶段内和阶段间的算子并行性。Para-Pipe在吞吐量与……（摘要在此处截断）

    arXiv:2609.04168v1 Announce Type: cross  Abstract: As edge-based deep learning applications become more complex, optimizing performance on heterogeneous System-on-Chips (SoCs) presents unique challenges. Traditional pipelining techniques distributing the computation across different on-chip processing units, while effective for throughput, do not address the latency demands posed by modern neural networks with complex interdependencies and extensive operator parallelism. There is a potential in leveraging operator parallelism to enable concurrent execution across multiple processing units, thereby reducing inference latency. However, prioritizing pipelining or parallel execution often necessitates a compromise, where optimizing one performance metric adversely impacts the other. This paper introduces Para-Pipe, a hierarchical mapping framework that integrates intra- and inter-stage operator parallelism within a pipelined architecture. Para-Pipe navigates the trade-off between throughpu
    
[^6]: 用于张量网络的参数化图论：纠缠重路由、结构简化与不可知层析

    Parameterised graph theory for tensor networks: entanglement rerouting, structural simplification, and agnostic tomography

    [https://arxiv.org/abs/2609.04165](https://arxiv.org/abs/2609.04165)

    该论文运用参数化图论证明切宽与树切宽界定了张量网络态转化为可高效处理的矩阵乘积态或树张量网络表示所需的键维开销，为张量网络的结构简化和态层析学习提供了新的理论工具。

    

    参数化图论研究图论问题的复杂性如何依赖于输入图的结构参数。这一视角已被证明在分析张量网络模拟方面很有用（Markov 和 Shi, 2008）。然而，它对张量网络表示和层析（tomography）的意义还不太明确。特别是，哪些图参数决定了张量网络态（TNS）是否具有可高效处理的矩阵乘积态（MPS）或树张量网络（TTN）表示，以及哪些参数控制着学习该态的复杂性？我们使用参数化图论来解决这些问题。首先，我们证明切宽和树切宽限制了将TNS表示为MPS或TTN所需的键维开销。在TTN的情况下，树切宽还限制了分组子系统的局部维度。证明基于纠缠重路由，它是经典图论中信息重路由的张量网络类比。（注：原文摘要在此处截断）

    arXiv:2609.04165v1 Announce Type: cross  Abstract: Parameterised graph theory studies how the complexity of graph-theoretic problems depends on structural parameters of the input graph. This perspective has proved useful in analysing tensor-network simulation (Markov and Shi, 2008). Its implications for tensor-network representations and tomography are less well understood. In particular, which graph parameters determine whether a tensor-network state (TNS) admits a tractable matrix product state (MPS) or tree tensor network (TTN) representation, and which control the complexity of learning the state?   We address these questions using parameterised graph theory. First, we show that cutwidth and tree-cutwidth bound the bond dimension overhead required to represent a TNS as an MPS or TTN. In the TTN case, tree-cutwidth also bounds the local dimension of the grouped subsystems. The proofs are based on entanglement rerouting, a tensor-network analogue of rerouting information in a classic
    
[^7]: 一种面向微型阿克曼车辆端到端自动驾驶的低成本开放平台

    A Low-Cost, Open Platform for End-to-End Autonomous Driving on a Miniature Ackermann Vehicle

    [https://arxiv.org/abs/2609.04147](https://arxiv.org/abs/2609.04147)

    本文提出一个集成实体车辆、打印城市赛道与Webots数字孪生的低成本开放平台，通过命令条件化行为克隆实现了微型阿克曼车辆的端到端自动驾驶，其横向误差（6.1厘米）接近人类演示水平，弥合了仿真与真实执行之间的鸿沟。

    

    本文提出了一个低成本、开放的实验平台，用于基于微型阿克曼车辆的端到端自动驾驶研究。该平台集成了实体车辆、打印的城市赛道、数据采集工具、轨迹配准以及Webots数字孪生，能够开展可控实验，将基于仿真的自动驾驶方法与真实世界执行相连接。作为首个基线，我们实现了命令条件化的行为克隆方法，其中神经策略接收车载摄像头图像和高级导航命令作为输入，输出转向和速度控制。该系统在实体车辆和仿真环境中均进行了评估。在真实闭环实验中，学习到的策略能够跟随车道并执行指定的转弯动作，相对参考路线的平均横向误差为6.1厘米，接近人类演示中所观察到的4.7厘米水平。在数字孪生实验中，摄像头视场角具有强烈的（影响）（注：摘要在此处被截断）

    arXiv:2609.04147v1 Announce Type: cross  Abstract: This paper presents a low-cost, open experimental platform for research in end-to-end autonomous driving with miniature Ackermann vehicles. The platform combines a physical vehicle, a printed urban track, data collection tools, trajectory registration, and a Webots digital twin, enabling controlled experiments that connect simulation-based autonomous-driving methods to real-world execution. As a first baseline, we implement command-conditioned behavior cloning, in which a neural policy receives an on-board camera image and a high-level navigation command and outputs steering and speed. The system is evaluated both on the physical vehicle and in simulation. In real closed-loop experiments, the learned policy follows lanes and executes commanded turns, reaching a mean cross-track error of 6.1 cm with respect to the reference route, close to the 4.7 cm observed in human demonstrations. In the digital twin, camera field of view has a stron
    
[^8]: 前瞻编码改进深度连续时间循环网络的学习

    Prospective Coding Improves Learning in Deep Continuous-Time Recurrent Networks

    [https://arxiv.org/abs/2609.04134](https://arxiv.org/abs/2609.04134)

    提出受生物学启发的递归正交滤波器，通过无参数的前瞻编码校正每层的自下而上输入，缓解深度连续时间循环网络中依赖深度的梯度衰减问题，从而改进学习效果。

    

    时间积分赋予连续时间循环网络记忆能力，但在深层堆叠结构中，它也会延迟自下而上的信号并衰减自上而下的误差。我们开发了递归正交滤波器，这是一类受生物学启发的复数值时间滤波器，属于对角状态空间模型（SSMs）的一种特例，并探讨能否通过使每层的自下而上输入具有前瞻性来解决这一失效模式。从能量模型出发，我们推导了RQF的动力学，并证明每个RQF都是一个带通滤波器，其可学习参数控制着调谐频率和带宽。随后，我们采用一种无参数的双抽头更新方法使每层的自下而上输入具有前瞻性，同时保持循环转移和并行扫描不变。我们将这一校正方法扩展到一般的对角SSMs，并证明当时间梯度被截断时（即仅空间反向传播），该方法能够缓解依赖于深度的梯度衰减问题。

    arXiv:2609.04134v1 Announce Type: new  Abstract: Temporal integration gives continuous-time recurrent networks memory, but in deep stacks it also delays bottom-up signals and attenuates top-down errors. We develop Recursive Quadrature Filters (RQFs), biologically motivated complex-valued temporal filters that are a special case of diagonal state-space models (SSMs), and ask whether this failure mode can be addressed by making each layer's bottom-up input prospective. Starting from an energy model, we derive the RQF dynamics and show that each RQF is a band-pass filter whose learnable parameters control its tuning frequency and bandwidth. We then make each layer's bottom-up input prospective using a parameter-free two-tap update that leaves the recurrent transition and parallel scan unchanged. We extend this correction to general diagonal SSMs and show that it mitigates depth-dependent gradient attenuation when temporal gradients are truncated, i.e., spatial-only backpropagation. We eva
    
[^9]: 通过高阶乐观主义在一般博弈中实现常数遗憾

    Constant regret in general games via higher-order optimism

    [https://arxiv.org/abs/2609.04113](https://arxiv.org/abs/2609.04113)

    提出了一种名为HOOD的非耦合学习算法，通过将带折扣的高阶预测器与熵正则化相结合来抑制博弈序列的大幅振荡，从而在任意N人博弈中实现了与博弈时间无关的常数级个体遗憾。

    

    我们提出了一种非耦合学习算法，当任意N人正规形式博弈（每个玩家最多K个动作）中的所有玩家都采用该算法时，可在整个博弈时间范围内均匀地保证O(N³log²K)的个体遗憾。我们提出的算法——称为带折扣的高阶乐观主义——是乐观跟随正则化领导者算法的一个变体，它将带折扣的(N+1)阶预测器与在博弈策略空间适当“提升”之上的熵正则化相结合。这些要素的组合经过特意设计，能够以受控的方式抑制诱导博弈序列的大幅振荡，从而消除了以往尝试在一般博弈中实现常数遗憾时的一个关键障碍。我们的方法与Liu、Farina和Ozdaglar同期且完全独立的工作存在多处惊人的相似之处，他们最近……

    arXiv:2609.04113v1 Announce Type: new  Abstract: We introduce an uncoupled learning algorithm which, when employed by all players of an arbitrary $N$-player normal form game with up to $K$ actions per player, guarantees $O(N^3\log^2 K)$ individual regret, uniformly over the horizon of play. The proposed algorithm - which we call higher-order optimism with discounting (HOOD) is a variant of optimistic follow-the-regularized-leader (OptFTRL) that combines a discounted $(N+1)$-th order predictor with entropic regularization over a suitable "lifting" of the game's strategy space. This combination of ingredients is purposefully designed to dampen large oscillations of the induced sequence of play in a controlled manner, removing in this way a key stumbling block of previous attempts to achieve constant regret in general games. Our approach bears several striking similarities to the concurrent - and completely independent - work of Liu, Farina, and Ozdaglar (arXiv:2608.31166), who very recen
    
[^10]: 顺序优于联合：论在线策略蒸馏与RLVR的相互作用

    Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

    [https://arxiv.org/abs/2609.04108](https://arxiv.org/abs/2609.04108)

    先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。

    

    可验证奖励强化学习（RLVR）和在线策略蒸馏（OPD）已成为对推理大语言模型进行后训练的两种主流方法。先前的工作利用OPD的密集token级监督来补充稀疏的RL奖励，在单个步骤内融合这两种信号：要么作为加权加性组合，要么作为对RL优势的教师调制重缩放。在本文中，我们展示了一个简单的两阶段方案——先OPD后RL——在逻辑和数学推理基准上持续优于纯OPD、纯RLVR以及所有此类联合基线方法。除了实证结果外，我们还通过pass@$k$行为、学习动态和参数更新对这一现象提供了系统性的理解，并得出一个一致的解释：OPD扩大了学生对教师支持解的覆盖范围，而RL则在该支持范围内进行锐化，同时联合优化这两种信号会导致它们相互干扰。

    arXiv:2609.04108v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) have emerged as two dominant methods for post-training reasoning LLMs. Prior work uses OPD's dense token-level supervision to complement the sparse RL reward, fusing the two signals within a single step: either as a \emph{weighted-additive combination} or a \emph{teacher-modulated rescaling} of the RL advantage. In this paper, we show that a simple two-stage scheme, OPD-then-RL, consistently outperforms pure OPD, pure RLVR, and all such joint baselines across logic and math reasoning benchmarks. Beyond the empirical results, we further provide a systematic understanding of this through pass@$k$ behavior, learning dynamics, and parameter updates, yielding a consistent explanation: OPD expands the student's coverage of teacher-supported solutions and RL sharpens within that support, while jointly optimizing the two signals causes them to interfere.To p
    
[^11]: 面向硬件感知的FP4 FlashAttention-4

    Hardware-Aware FP4 FlashAttention-4

    [https://arxiv.org/abs/2609.04105](https://arxiv.org/abs/2609.04105)

    提出硬件感知的FP4 FlashAttention-4，其中Direct-P方法将注意力分数直接映射为FP4概率，在GB200上实现最高2.13倍于BF16的前向吞吐量，并通过将前向量化直接传递到反向传播的因果路径使80亿参数模型更新加速最高1.14倍。

    

    Blackwell的4位浮点（FP4）张量核心并不会自动使注意力计算变快，因为一旦其矩阵乘积规模缩小，softmax转换和片上依赖就会成为主导瓶颈。我们通过用于非因果推理的Direct-P方法以及一条将前向量化直接传递到反向传播的因果路径来解决这一问题。Direct-P将分数直接映射为FP4概率，在NVIDIA GB200上达到了bfloat16（BF16）前向吞吐量的最高2.13倍。该因果路径从保存的量化查询和键中重构概率，并使用8位浮点（FP8）梯度操作数，将完整的单GPU 80亿参数更新加速至最高1.14倍。在匹配的分布式训练中保留FP8概率和数值；所有测试的MXFP4概率/数值训练轨迹均发散。

    arXiv:2609.04105v1 Announce Type: new  Abstract: Blackwell's 4-bit floating-point (FP4) tensor cores do not automatically make attention faster because softmax conversion and on-chip dependencies dominate once its matrix products shrink. We address this with \emph{Direct-P} for noncausal inference and a causal path that passes the forward quantization directly into backward. Direct-P maps scores directly to FP4 probabilities and reaches up to 2.13$\times$ the bfloat16 (BF16) forward throughput on an NVIDIA GB200. The causal path reconstructs probabilities from saved quantized queries and keys and uses 8-bit floating-point (FP8) gradient operands, accelerating a complete single-GPU 8-billion-parameter update by up to 1.14$\times$. Matched distributed training retains FP8 probabilities and values; every tested MXFP4 probability/value training trajectory diverges.
    
[^12]: DRACO：基于动态评分准则的长程智能体训练细粒度信用分配方法

    DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training

    [https://arxiv.org/abs/2609.04094](https://arxiv.org/abs/2609.04094)

    DRACO通过在训练中动态生成评分准则，并以闭式解方式将轨迹级评判重新分配到具体步骤，解决了无真实成功信号时长程智能体训练的细粒度信用分配问题，在AppWorld上显著超越基础模型和稀疏奖励GRPO。

    

    当任务具备程序化检查器时，基于可验证奖励的强化学习效果良好，但大多数长程智能体领域并不存在这样的检查器。我们在“结果盲设”下开展工作，即真实成功信号不可用的场景。多准则评分准则是提供此类奖励的常用方式；它们对每个轨迹仅评分一次，但单一标量在数十个步骤中是较弱的信号。我们提出DRACO：基于评分准则分布的优势分配信用优化方法。它在训练过程中动态生成评分准则以跟踪策略不断演进的能力，对每个完成的轨迹对这些准则评分一次，并将该评判重新分配到负责相关准则标注的步骤上，从而在GRPO中产生差异化的逐步优势。这种重新分配是闭式解形式，不引入任何需要训练的归因模块。在AppWorld上，DRACO比基础模型提升15.9分，比使用稀疏奖励训练的GRPO提升5.3分。

    arXiv:2609.04094v1 Announce Type: new  Abstract: Reinforcement Learning from Verifiable Rewards works well when a task has a programmatic checker, but most long-horizon agent domains have none. We work in the outcome-blind setting, where ground-truth success signals are not available. Multi-criteria rubrics are a popular way to supply such a reward; they are scored once per trajectory, but a single scalar is a poor signal across tens of steps. We propose DRACO: Distributing Rubric-based Advantage for Credit Optimization. It generates rubrics dynamically during training to track the policy's evolving capability, scores those rubrics once per completed trajectory, and redistributes that judgment over the steps responsible for annotated rubrics to produce differentiated per-step advantages in GRPO. The redistribution is closed-form and does not introduce any trained attribution module. On AppWorld, DRACO gains 15.9 points over the base model and 5.3 points over GRPO trained with a sparse 
    
[^13]: 退化扩散模型的条件化

    Conditioning Degenerate Diffusion Models

    [https://arxiv.org/abs/2609.04090](https://arxiv.org/abs/2609.04090)

    该论文提出利用因果最优传输为扩散系数退化（奇异）的扩散生成模型构造近似损失函数，在条件密度不存在或不光滑的极弱假设下确定用于引导的最小熵控制。

    

    当前受条件约束的生成模型在训练过程中严重依赖得分函数来进行引导。当生成模型是一个具有奇异扩散系数的扩散过程，且其底层的（条件）密度要么不存在要么不光滑时，我们利用因果最优传输，在极小的假设条件下定义了近似损失函数，用以确定用于引导的最小熵控制。我们的方法依赖于因果最优传输，以及通过（受条件约束的）扩散过程的可预测表示性质对其进行的刻画，其中这些扩散过程的相关鞅问题是适定的，遵循Üstünel的框架。

    arXiv:2609.04090v1 Announce Type: new  Abstract: Current conditioned generative models heavily rely on score functions for guidance during training. When the generative model is a diffusion process with a singular diffusion coefficient and the underlying (conditional) densities either do not exist or are not smooth, we use causal optimal transport to define \emph{approximate} loss functions that identify a minimum-entropy control for guidance under minimal assumptions. Our approach relies on causal optimal transport and its characterization through the predictable representation property of (conditioned) diffusion processes whose associated martingale problem is well posed, \`a la \"Ust\"unel.
    
[^14]: 子空间推断实现高效的基于偏好的主动奖励学习

    Subspace Inference Enables Efficient Active Reward Learning from Preferences

    [https://arxiv.org/abs/2609.04066](https://arxiv.org/abs/2609.04066)

    本文提出PreferenceEKF方法，将主动偏好学习框架化为序贯贝叶斯滤波问题，通过扩展卡尔曼滤波器在低维参数子空间中高效跟踪奖励模型的不确定性，实现样本高效的RLHF奖励学习。

    

    基于人类反馈的强化学习（RLHF）已成为一种从人类偏好中学习奖励模型的强大但样本效率低下的方法，这使得主动学习成为构建信息丰富的偏好查询的关键组成部分。然而，主动学习所需的有效不确定性量化对于大型神经网络奖励模型而言仍然是一个关键挑战。在本文中，我们提出了PreferenceEKF，这是一种样本高效的方法，它将主动偏好学习框架化为序贯贝叶斯滤波问题，从而跟踪奖励模型的不确定性。我们的方法不依赖于在整个神经网络参数空间上进行计算代价极高的后验推断，而是在低维参数子空间内通过扩展卡尔曼滤波器进行序贯推断，随着新的偏好查询的到来持续更新奖励模型的后验分布。我们的方法实现了对神经网络的可扩展采样

    arXiv:2609.04066v1 Announce Type: cross  Abstract: Reinforcement learning from human feedback (RLHF) has emerged as a powerful yet sample-inefficient approach for learning reward models from human preferences, making active learning a critical component in synthesizing informative preference queries. However, effective uncertainty quantification required for active learning remains a key challenge for large neural network reward models. In this paper, we introduce PreferenceEKF, a sample-efficient approach that tracks reward model uncertainty by framing active preference learning as a sequential Bayesian filtering problem. Instead of relying on computationally prohibitive posterior inference over the full neural network parameter space, our method performs sequential inference via an extended Kalman filter within a low-dimensional parameter subspace, continuously updating the reward model posterior as new preference queries arrive. Our approach enables scalable sampling of neural netwo
    
[^15]: 单层注意力中布尔函数的头复杂度

    The Head Complexity of Boolean Functions in Single-Layer Attention

    [https://arxiv.org/abs/2609.04046](https://arxiv.org/abs/2609.04046)

    本文证明了单层注意力模型头复杂度的精确层级结构：$k$ 个注意力头恰好能计算 $k$ 位奇偶校验而无法计算 $(k+1)$ 位奇偶校验，且该下界在嵌入维度和数值精度均无限制的情况下依然无条件成立。

    

    单层自注意力究竟能计算什么？我们研究头复杂度这一概念：即在单层纯注意力模型中计算一个函数所需的最少注意力头数量。我们在这一度量下建立了一个精确的层级结构：$k$ 个注意力头可以计算 $k$ 位奇偶校验函数，但无法计算 $(k+1)$ 位奇偶校验函数。该下界在 transformer 本可能利用的两种资源上是无条件的——它在嵌入维度无限制和数值精度无限制的情况下依然成立。证明的关键在于一种交错和障碍：在消除 softmax 分母后，所得决策多项式中的每个单项式都至少遗漏 $(k+1)$ 个输入位中的一个，从而迫使该项与奇偶校验函数的相关性为零。同样的障碍也为相关任务提供了下界，包括被广泛研究的多跳归纳头任务。此外，我们还建立了关于嵌入维度和数值精度的紧致性界。具体而言，紧致性（摘要在此处被截断）

    arXiv:2609.04046v1 Announce Type: cross  Abstract: What can a single layer of self-attention compute? We study head complexity: the minimum number of attention heads required to compute a function in a one-layer attention-only model. We establish an exact hierarchy under this measure: $k$ heads compute $k$-bit parity but cannot compute $(k+1)$-bit parity.   The lower bound is unconditional in the two resources a transformer might otherwise exploit; it holds at unbounded embedding dimension and unbounded numerical precision. The proof rests on an alternating-sum obstruction: after clearing the softmax denominators, every monomial in the resulting decision polynomial omits at least one of the $k+1$ input bits, forcing its correlation with parity to vanish. The same obstruction yields lower bounds for related tasks, including the well-studied multi-hop induction-head task.   We also establish compactness bounds for embedding dimension and numerical precision. Specifically, a compactness t
    
[^16]: 挤出长条形状对3D混凝土打印可建造性的影响：一种几何信息驱动的深度学习-有限元方法

    Influence of Extruded Filament Shape on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning-FEM Approach

    [https://arxiv.org/abs/2609.04028](https://arxiv.org/abs/2609.04028)

    该研究提出了一个将深度学习长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合的几何信息驱动建模框架，能够直接从材料和工艺参数生成考虑真实长条几何形状的数值模型，从而更准确地评估3D混凝土打印结构的可建造性。

    

    沉积长条的几何形貌会显著影响3D混凝土打印（3DCP）结构的性能与稳定性。然而，大多数基于有限元（FEM）的可建造性评估方法将打印层简化为矩形，这可能限制了预测精度。本研究提出了一种几何信息驱动的建模框架，将基于深度学习的长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合，以研究真实长条几何形状对可建造性的影响。该框架可直接从材料与工艺参数生成几何感知的数值模型，无需实验性的长条表征或计算量庞大的流体流动模拟。通过与实验数据的验证以及对直线墙体的参数化研究表明，挤出参数及由此产生的长条……

    arXiv:2609.04028v1 Announce Type: cross  Abstract: The geometric morphology of deposited filaments can significantly influence the structural performance and stability of 3D concrete-printed (3DCP) structures. However, most finite element (FEM)-based approaches for buildability assessment represent printed layers as simplified rectangles, potentially limiting predictive accuracy. This study proposes a geometry-informed modelling framework that integrates the deep-learning-based filament shape prediction tool ShapeGen3DCP with a layer-activation FEM approach to investigate the effect of realistic filament geometries on buildability. The framework generates geometry-aware numerical models directly from material and process parameters, eliminating the need for experimental filament characterization or computationally intensive fluid-flow simulations. Validation against experimental data and a parametric study of rectilinear walls demonstrate that extrusion parameters and the resulting fil
    
[^17]: FLY-EVAL++：一种用于大语言模型安全约束飞行预测的证据驱动评估协议

    FLY-EVAL++: An Evidence-Driven Evaluation Protocol for Safety-Constrained Flight Prediction with Large Language Models

    [https://arxiv.org/abs/2609.04021](https://arxiv.org/abs/2609.04021)

    本文提出FLY-EVAL++评估协议，通过对协议合规性、物理可行性和安全约束进行确定性验证并结合量规引导的聚合评分，对66个大语言模型的飞行轨迹与姿态预测能力进行评估，发现安全合规性是区分模型优劣最具判别力的维度。

    

    在安全关键且受物理规律支配的环境中评估大语言模型（LLM），不能仅仅依赖基于准确率的指标，因为数值上接近真实值的预测仍可能违反运行约束、以物理上不一致的方式组合场变量，或无法生成可用的结构化输出。现有的评估协议无法可靠地衡量这些失效模式。我们提出了FLY-EVAL++，一种证据驱动的评估协议，它将对协议合规性、物理可行性和安全约束的确定性验证与基于固定量规引导的聚合相结合，形成可解释的多维度评分。我们通过在PilotBench设置的基础上扩展历史条件化预测和多步预测任务，将FLY-EVAL++实例化应用于飞行轨迹与姿态预测（FTAP）任务。在66个大语言模型的评估中，安全合规性是区分模型行为最具判别力的维度：具有相当……（原文摘要在此处截断）

    arXiv:2609.04021v1 Announce Type: new  Abstract: Evaluating large language models (LLMs) in safety-critical, physics-governed environments requires more than accuracy-based metrics, because predictions that are numerically close to the ground truth can still violate operational constraints, combine fields in physically inconsistent ways, or fail to produce usable structured outputs. Existing evaluation protocols do not measure these failure modes reliably. We propose FLY-EVAL++, an evidence-driven evaluation protocol that combines deterministic verification of protocol compliance, physical feasibility, and safety constraints with fixed rubric-guided aggregation into interpretable multi-dimensional scores. We instantiate FLY-EVAL++ for Flight Trajectory and Attitude Prediction (FTAP) by extending the PilotBench setting with history-conditioned and multi-step prediction tasks. Across 66 LLMs, safety compliance is the most discriminative dimension of model behavior: models with comparable
    
[^18]: 针对重尾分布的极端分位数处理效应的位置不变估计量

    A location-invariant estimator of extremal quantile treatment effects for heavy-tailed distributions

    [https://arxiv.org/abs/2609.04018](https://arxiv.org/abs/2609.04018)

    该论文通过逆倾向得分加权将位置不变的Fraga极值指数估计量引入因果设置，并采用基于差分的外推方案使位置参数在分位数差中自然抵消，从而首次构造出对位置平移保持不变的极端分位数处理效应估计量，解决了现有方法在重尾分布下不满足位置不变性的问题。

    

    分位数处理效应（QTEs）衡量处理对结果分布的影响，在目标分位数远超数据范围的应用中，极端分位数水平上的估计是核心关注点。对于重尾潜在结果，现有的极端分位数处理效应估计量依赖于外推法结合因果极值指数（EVI）估计量，但由此得到的估计量在潜在结果分布的共同位置平移下不具有不变性，尽管总体的分位数处理效应本身是位置不变的。我们分两步解决这一问题。首先，我们利用逆倾向得分加权将位置不变的Fraga极值指数估计量适配到因果设置中。其次，我们用基于差分的方案替代原始的外推公式，在该方案下，取分位数差时位置参数会被抵消。因此，所得到的分位数处理效应估计量是位置不变的。

    arXiv:2609.04018v1 Announce Type: new  Abstract: Quantile treatment effects (QTEs) measure the effect of a treatment on the distribution of an outcome, and their estimation at extreme quantile levels is of central interest in applications where the target quantiles lie far beyond the range of the data. For heavy-tailed potential outcomes, existing extremal QTE estimators rely on extrapolation combined with a causal extreme value index (EVI) estimator, but the resulting estimator is not invariant under a common location shift of the potential outcome distributions, even though the population QTE is. We address this issue in two steps. First, we adapt the location-invariant Fraga estimator of the EVI to the causal setting using inverse propensity score weighting. Second, we replace the original extrapolation formula with a difference-based scheme, under which the location parameter cancels when quantile differences are taken. The resulting QTE estimator is therefore location invariant. W
    
[^19]: LLM4CKD：用于早期慢性肾病筛查的大语言模型

    LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening

    [https://arxiv.org/abs/2609.04013](https://arxiv.org/abs/2609.04013)

    大语言模型在零样本和少样本学习设置下，无需任务特定训练即可实现与传统机器学习和深度学习方法相当的早期慢性肾病筛查性能。

    

    早期筛查慢性肾病（CKD）对于及时干预至关重要，然而大多数机器学习（ML）和深度学习（DL）方法需要标注数据和模型训练，限制了它们在现实筛查场景中的应用。本研究评估了大语言模型（LLMs）在零样本和少样本上下文学习设置下进行CKD筛查的有效性，并将其与传统ML和DL方法进行比较。我们提出了一个框架，该框架使用临床筛选的表格特征和结构化提示模板，使基于LLM的推理无需任务特定训练即可实现。LLM的性能在多种提示风格、特征配置和数据设置下进行了评估，并与标准ML、DL和表格基础模型（TFM）基线以及现有CKD筛查工具进行了比较。结果显示，LLM仅需少量样本即可获得有竞争力的性能，往往能够匹敌或超越传统方法。

    arXiv:2609.04013v1 Announce Type: new  Abstract: Early screening of chronic kidney disease (CKD) is critical for timely intervention, yet most machine learning (ML) and deep learning (DL) approaches require labeled data and model training, limiting their use in real-world screening settings. This study evaluates the effectiveness of large language models (LLMs) for CKD screening under zero-shot and few-shot in-context learning settings and compares them with traditional ML and DL methods. We propose a framework that uses clinically selected tabular features and structured prompt templates to enable LLM-based inference without task-specific training. LLM performance is evaluated across multiple prompt styles, feature configurations, and data settings, and compared with standard ML, DL, and tabular foundation model (TFM) baselines, and existing CKD screening tools. The results show that LLMs can achieve competitive performance using only a small number of examples, often matching or outp
    
[^20]: 基于可微分混合建模从实验数据中学习与优化化学传输过程

    Differentiable Hybrid Modelling for Learning and Optimising Chemical Transport Processes from Experimental Data

    [https://arxiv.org/abs/2609.04011](https://arxiv.org/abs/2609.04011)

    提出一个通用的可微分混合建模框架，将JAX有限体积群体平衡求解器与可学习的神经网络组件相结合，能够从真实实验数据中同时发现本构定律并拟合初始条件，用于学习和优化化学传输过程。

    

    可靠的传输模型在模拟和优化众多化学工程过程中至关重要，然而，大多数模型采用人工挑选的本构定律，这可能无法反映真实情况，并且通常假设初始条件是完全已知的。这两种限制都可能显著偏倚模型预测，并在预测和控制应用中导致系统性误差。黑箱神经代理替代建模方法能够更好地匹配真实的实验数据，但仅局限于其训练所用的任务，且无法对其进行物理一致性检验。在此，我们提出了一个面向传输过程的通用可微分混合建模框架，特别针对群体平衡方程的情形。我们的框架将JAX有限体积群体平衡求解器与可学习的神经网络组件相集成，这些组件经训练后既能发现本构定律，又能从真实实验数据中拟合初始条件。

    arXiv:2609.04011v1 Announce Type: cross  Abstract: Reliable transport models are essential when modelling and optimising many chemical engineering processes, yet, most models assume hand-picked constitutive laws which may not reflect reality, and often assume initial conditions are known exactly. Both restrictions can significantly bias model predictions and lead to systematic error when used in predictive and control settings. Black-box neural surrogate alternatives for modelling can better match real example data, but are confined to the task they were trained on and cannot be interrogated for physical consistency. Here we introduce a general-purpose differentiable hybrid modelling framework for transport processes, specifically for the case of population balance equations. Our framework integrates a JAX finite volume population balance solver with learnable neural network components which are trained to both discover constitutive laws and fit initial conditions from real experimenta
    
[^21]: 通过离散扩散解锁大语言模型的无损加速

    Unlocking Lossless Speedups in LLMs via Discrete Diffusion

    [https://arxiv.org/abs/2609.04010](https://arxiv.org/abs/2609.04010)

    本文提出扩散增强大语言模型，通过将参数解耦为标准NTP训练的自回归权重与轻量级扩散权重，并配合Ψ-Spec采样器家族，在无需草稿模型和 negligible 训练开销的情况下，实现大语言模型的无损并行词元生成加速。

    

    大语言模型（LLM）的成功很大程度上归功于下一词元预测（NTP），但其自回归（AR）结构要求缓慢的串行词元生成。为了克服这一瓶颈，我们提出了扩散增强大语言模型（diffusion-augmented LLMs），这是一类新型模型，它在定义自回归模型分布的同时，利用扩散技术从该分布中并行抽取多个词元。我们将这些模型的参数解耦为两组：一组是使用标准NTP目标训练的自回归权重，另一组是训练用于同时生成多个词元的轻量级扩散权重。扩散权重通过一个简单的扩散蒸馏（Diffusion Distillation）阶段学习得到，该阶段给现有大语言模型训练流程增加的开销可以忽略不计。我们还引入了Ψ-Spec，这是一系列采样器，能够在固定上下文长度下实现无损加速和推理时扩展。与投机解码不同，我们的方法不需要单独的草稿模型。与扩散……

    arXiv:2609.04010v1 Announce Type: new  Abstract: Large Language Models (LLMs) owe much of their success to next-token prediction (NTP), but their autoregressive (AR) structure requires slow, sequential token generation. To overcome this bottleneck, we introduce diffusion-augmented LLMs, a new class of models that defines an AR model distribution while using diffusion to draw multiple tokens in parallel from that distribution. We decouple the parameters of these models into two sets: AR weights, trained using the standard NTP objective, and lightweight diffusion weights, trained to generate multiple tokens simultaneously. The diffusion weights are learned through a simple Diffusion Distillation phase that adds negligible overhead to existing LLM training pipelines. We also introduce $\Psi$-Spec, a family of samplers that enables lossless acceleration and inference-time scaling at a fixed context length. Unlike speculative decoding, our method requires no separate draft model. Unlike dif
    
[^22]: RobustSeiz：一个用于基准测试脑电图癫痫检测模型鲁棒性的开源框架

    RobustSeiz: An Open-Source Framework for Benchmarking the Robustness of EEG Seizure Detection Models

    [https://arxiv.org/abs/2609.04007](https://arxiv.org/abs/2609.04007)

    RobustSeiz是一个开源、模型无关的基准测试框架，通过将四个公开EEG数据集标准化并施加临床驱动的环境、噪声和对抗性分布偏移，实现对癫痫检测模型鲁棒性的标准化、可复现的压力测试。

    

    尽管在留出的脑电图（EEG）数据上表现优异，癫痫检测器在真实世界的采集变异性、伪影和对抗性输入下仍可能失效。我们提出了RobustSeiz，一个开源的、模型无关的框架，为在部署前于受控的、临床驱动的分布偏移条件下对癫痫检测器进行压力测试和比较提供了标准化、可复现的协议。我们将四个公开的头皮脑电数据集（CHB-MIT、TUSZ、Siena和SeizeIT1）标准化为BIDS-EEG目录结构，并在留出的数据划分上评估受试者独立的检测器。环境、噪声和对抗性变换在预定义的超参数网格上进行遍历扫描。每次运行均报告样本级和事件级的灵敏度、精确率、F1分数、每24小时假阳性数、Lead和Lag发作时间偏差，以及蒙特卡洛dropout预测一致性。RobustSeiz包含Docker化的GPU流水线、实验注册表以及完整评估和研究……（摘要在此处被截断）

    arXiv:2609.04007v1 Announce Type: new  Abstract: Despite strong performance on held-out electroencephalography (EEG) data, seizure detectors may fail under real-world acquisition variability, artifacts, and adversarial inputs. We introduce RobustSeiz, an open-source, model-agnostic framework that provides a standardized, reproducible protocol for stress-testing and comparing seizure detectors under controlled, clinically motivated distribution shifts before deployment. We standardize four public scalp-EEG corpora (CHB-MIT, TUSZ, Siena, and SeizeIT1) into BIDS-EEG trees and evaluate subject-independent detectors on held-out splits. Environment, noise, and adversarial transforms are swept over predefined hyperparameter grids. Each run reports sample- and event-level sensitivity, precision, F1, false positives per 24 h, Lead and Lag onset timing, and Monte Carlo dropout predictive agreement. RobustSeiz includes a Dockerized GPU pipeline, experiment registry, and full-evaluation and resear
    
[^23]: 锐化集成：一种用于脑部MRI图像修复后处理的SSIM对齐残差精炼器

    Sharpening the Ensemble: An SSIM-Aligned Residual Refiner for Brain-MRI Inpainting Post-Processing

    [https://arxiv.org/abs/2609.03981](https://arxiv.org/abs/2609.03981)

    该论文提出一种后处理方法，通过将2025年BraTS脑部MRI修复任务中两个并列第一的模型组成深度集成，并训练一个轻量级的SSIM对齐残差精炼器，有效锐化了因ℓ1和MSE损失均值寻求行为而模糊的合成区域，显著提升了SSIM指标。

    

    脑部MRI图像修复技术是用合成的、解剖学上合理的健康组织替换扫描图像中被掩蔽的区域，从而使为健康大脑构建的分析工具能够应用于这些原本会被拒绝的图像。在BraTS局部合成基准测试中（该基准联合使用结构相似性指数（SSIM）、峰值信噪比和均方误差（MSE）对提交结果进行排名），近期最强的模型在精度上表现良好，但一些研究报告了合成区域模糊的问题，并将其归因于训练损失中ℓ1和MSE项的均值寻求行为。我们在后处理阶段解决这一问题：将2025年两个并列第一的模型组成深度集成，并在集成模型自身的输出上训练一个轻量级的残差精炼器，其损失函数由ℓ1损失和结构相似性项共同构成，其中结构相似性项的权重λ可变。在中等λ值下，该精炼器将SSIM相对于集成模型从0.8767提升至0.8……

    arXiv:2609.03981v1 Announce Type: cross  Abstract: Brain-MRI inpainting replaces a masked region of a scan with synthesized, anatomically plausible healthy tissue, so that analysis tools built for healthy brains can be applied to images they would otherwise reject. On the BraTS local-synthesis benchmark, which ranks submissions on the structural similarity index (SSIM), the peak signal-to-noise ratio, and the mean squared error (MSE) jointly, the strongest recent models are accurate, but several report blurry synthesized regions and attribute this to the mean-seeking behavior of the $\ell_1$ and MSE terms in their training losses. We address this in post-processing, forming a deep ensemble of the two co-first-place 2025 models and training a lightweight residual refiner on the ensemble's own outputs under an $\ell_1$ loss augmented with a structural-similarity term whose weight $\lambda$ we vary. At a moderate $\lambda$ the refiner improves SSIM over the ensemble, from $0.8767$ to $0.8
    
[^24]: 面向分类与回归任务联合处理的协作式多任务语义通信

    Cooperative Multi-Task Semantic Communication for Joint Classification and Regression Tasks

    [https://arxiv.org/abs/2609.03977](https://arxiv.org/abs/2609.03977)

    本文将协作式多任务语义通信框架扩展到复杂Cityscapes数据集上异构的分类与回归任务的联合处理，通过采用信息最大化原理使其能够同时容纳离散与连续语义变量，突破了以往仅限于简单数据集同质分类任务的局限。

    

    多任务语义通信在未来智能网络中优先考虑多个任务的同时执行，而非比特级的精确重构。在我们之前的工作[1]中，我们提出了协作式多任务语义通信（CMT-SemCom）框架，其中语义编码器被划分为一个公共单元（CU）和多个特定单元（SU），以促进协作式的多任务处理。然而，CMT-SemCom此前仅在简单数据集上的同质分类任务中进行了评估，这限制了其在真实世界感知系统中的适用性。在本文中，我们将CMT-SemCom扩展至在复杂的Cityscapes数据集上联合处理异构的分类与回归任务。我们采用信息最大化原理，使其能够容纳混合的离散与连续语义变量。特别是，我们将所提出的框架与独立单任务训练、传统任务……进行基准对比。

    arXiv:2609.03977v1 Announce Type: cross  Abstract: Multi-Task semantic communication (SemCom) prioritizes simultaneous execution of multiple tasks over bit-accurate reconstruction in future intelligent networks. In our prior work [1], we introduced the cooperative multi-task SemCom (CMT-SemCom) framework, in which the semantic encoder is divided into a common unit (CU) and multiple specific units (SUs) to facilitate cooperative multi-task processing. However, CMT-SemCom has been evaluated on homogeneous classification tasks on simplistic datasets, limiting its applicability to real-world perception systems. In this paper, we extend our CMT-SemCom to jointly handle heterogeneous classification and regression tasks on the complex Cityscapes dataset. We adopt the information maximization (InfoMax) principle so that it accommodates mixed discrete and continuous semantic variables. In particular, we benchmark the proposed framework against independent single-task training, a conventional ta
    
[^25]: OSR：面向分类模型自适应标签移除的输出空间重分布

    OSR: Output Space Redistribution for Adaptive Label Removal in Classification Models

    [https://arxiv.org/abs/2609.03972](https://arxiv.org/abs/2609.03972)

    该论文提出OSR方法，通过输出空间的统计重分布来近似重新训练模型的移除后置信输出，以模块化输出过滤器的形式实现标签移除，无需原始数据和特征空间调整，从而解决了现有方法在成本、可扩展性和模型效用方面的局限。

    

    在分类体系不断演化的分类系统中，标签移除经常发生，此时类别需要被动态地更新或删除。为了适应这些变化，分类模型必须进行相应调整。现有解决方案大致可分为基于重新训练和基于特征空间调整两类，尽管形式各异，但它们存在共同的局限性，包括依赖对原始数据的访问、高昂的计算和存储成本、结果不一致、可扩展性差以及模型效用下降。为解决这些问题，我们提出了一种新颖的方法，利用输出空间中的统计重分布来近似重新训练后模型的移除后置信向量。我们的方法可作为一个模块化的输出过滤器使用，免除了特征空间调整或损失函数收敛的负担，缓解了可扩展性限制。此外，该方法仅需现有标签和先前的输出置信信息即可工作。

    arXiv:2609.03972v1 Announce Type: new  Abstract: Label removal occurs frequently in classification systems with evolving taxonomies, where categories must be dynamically updated or eliminated. To accommodate such changes, classification models must adapt accordingly. Existing solutions, broadly categorized as retraining-based and feature-space-adjustment-based, share common limitations despite their variations, including reliance on access to original data, substantial computational and storage costs, inconsistent results, poor scalability, and degradation of model utility. To address this, we propose a novel approach that leverages statistical redistribution in the output space to approximate the post-removal confidence vectors of a retrained model. Applicable as a modular output filter, our method bypasses the burden of feature-space adjustments or loss-function convergence, alleviating scalability limitations. Furthermore, by requiring only existing labels and prior output confidenc
    
[^26]: RARF：面向3D脑部MRI图像修复的区域感知整流流框架

    RARF: Region-Aware Rectified Flows for 3D Brain MRI Inpainting

    [https://arxiv.org/abs/2609.03956](https://arxiv.org/abs/2609.03956)

    提出了区域感知整流流框架RARF，通过将生成过程限制在修复区域、保留周围真实体素作为解剖上下文，并结合掩码流匹配与重建一致性训练，实现了高质量的3D脑部MRI图像修复。

    

    医学图像修复有望通过在病理区域内重建健康组织，从而改进自动化脑部MRI分析。我们提出了RARF，一个面向掩码数据生成的任务无关的区域感知整流流框架。我们将该框架实例化用于3D脑部MRI修复，作为参加2026年BraTS修复挑战赛的提交方案。RARF将随机插值过程限制在修复区域内，同时保持观测到的体素固定不变，以提供患者特异性的解剖学上下文。一个三维神经网络接收部分空洞的图像（缺失区域由高斯噪声填充）、修复掩码以及相应的时间步。该模型采用掩码流匹配和重建一致性目标进行训练，并结合掩码感知的预处理与数据增强。在推理阶段，学习到的速度场将初始噪声向……（原文摘要在此处截断）

    arXiv:2609.03956v1 Announce Type: cross  Abstract: Medical image inpainting has the potential to improve automated brain MRI analysis by reconstructing healthy tissue within pathological regions. We introduce RARF, a task-agnostic region-aware rectified flow framework for masked data generation. We instantiate the framework for 3D brain MRI inpainting as our submission to the BraTS Inpainting Challenge 2026. RARF restricts the stochastic interpolation process to the inpainting region, while the observed voxels remain fixed and provide patient-specific anatomical context. A three-dimensional neural network receives the partially voided image, with Gaussian noise filling the missing region, together with the inpainting mask and the corresponding timestep. The model is trained using masked flow-matching and reconstruction-consistency objectives, combined with mask-aware preprocessing and data augmentation. During inference, the learned velocity field transports the initial noise toward a 
    
[^27]: 面向代码大语言模型中可靠与对抗性测试生成的两阶段强化学习

    Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs

    [https://arxiv.org/abs/2609.03955](https://arxiv.org/abs/2609.03955)

    该论文提出了一种两阶段强化学习框架TCS，第一阶段生成与参考解一致的可靠测试用例，第二阶段学习针对模型当前失败模式的对抗性反例测试，从而有效提升代码大模型的测试生成质量和代码性能。

    

    强化学习（RL）通过可执行反馈极大地推动了基于大语言模型（LLM）的代码生成。编程问题的反馈主要来自特定的测试用例，而高质量的测试用例往往十分稀缺，因为它们既需要具备可靠性，又需要具备区分性。因此，我们转而研究利用学习到的模型自动生成测试用例。我们发现这本质上是一个对抗性强化学习问题：模型需要根据求解器当前的失败模式生成有效的测试用例作为反例。我们提出了测试用例缩放，这是一个用于有效测试生成的两阶段强化学习框架。两个阶段均从一个滚动策略对齐缓冲区中训练测试生成器：第一阶段生成与参考解一致的测试用例，第二阶段将缓冲区限制为当前的失败模式并学习生成反例测试。在TACO和LiveCodeBench数据集上，TCS同时提升了pass@1和推理性能。

    arXiv:2609.03955v1 Announce Type: new  Abstract: Reinforcement learning (RL) has substantially advanced code generation with large language models (LLMs) through executable feedback. The feedback for coding problems mainly comes from specific test cases, where high-quality test cases are often scarce since they should be both sound and discriminative. We thus turn to study the auto-generation of test cases using the learned model. We find this is naturally an adversarial RL problem: the model is expected to generate effective test cases as counterexamples, depending on the solver's current failure modes. We propose Test Cases Scaling (TCS), a two-stage RL framework for effective test generation. Both stages train a test generator from a rolling policy-aligned buffer: Stage 1 generates tests consistent with the reference solution, and Stage 2 restricts the buffer to current failure modes and learns counterexample tests. Across TACO and LiveCodeBench, TCS improves both pass@1 and inferen
    
[^28]: VestigeKV：NoPE-MLA的KV缓存通过一个残余分支携带其自身的淘汰信号

    VestigeKV: The NoPE-MLA KV Cache Carries Its Own Eviction Signal in a Vestigial Branch

    [https://arxiv.org/abs/2609.03949](https://arxiv.org/abs/2609.03949)

    VestigeKV发现NoPE MLA模型KV缓存中的64维解耦RoPE残余分支已被训练重新利用为显著性信号，据此提出无需训练和量化的查询无关缓存淘汰方法，在8-32倍压缩下几乎不损失检索精度。

    

    问题所在：一个长期存在的KV缓存必须在将要读取它的查询出现之前就被压缩；基于已观测注意力进行选择的方法（H2O、SnapKV）在这种情况下会失效（在NoPE MLA模型上针检索率仅为0.00-0.33），因为token的重要性尚未被观测到。方法：在Kimi Linear模型上，VestigeKV利用缓存自身已携带的与查询无关的信号进行淘汰：即64维解耦分支——这是RoPE的残余结构，NoPE训练将其重新利用为显著性通道。只需读取每行的11%，它将缓存划分为两个层级：top-m行保留在注意力层；其余所有行——精确保存、从不删除——移动到GPU驻留的存档中，每一步都可以通过经过验证的触发器访问。无需训练、无需量化、无需更改权重或内核。代价：无可测量的损失：在8k到65k上下文长度范围内，8倍压缩下检索率保持在1.00，32倍压缩下为0.92，与全行选择方法零差距。注意力层级仅占Kimi Linear每token 8.1 KB缓存中的0.25 KB。

    arXiv:2609.03949v1 Announce Type: cross  Abstract: The problem. A long-lived KV cache must be compressed before the queries that will read it exist; selection by observed attention (H2O, SnapKV) collapses there (0.00-0.33 needle retrieval on a NoPE MLA model), because a token's importance has not yet been observed. The method. On Kimi Linear, VestigeKV evicts by a query-independent signal the cache already carries: the 64-dimensional decoupled branch, a vestige of RoPE that NoPE training repurposes into a salience channel. Reading 11% of each row, it partitions the cache: the top-m rows stay in the attended tier; every other row moves -- exactly, never deleted -- to a GPU-resident archive reachable per step by a certified trigger. No training, no quantization, no weight or kernel change. Cost. Nothing measurable: retrieval holds at 1.00 under 8x and 0.92 under 32x from 8k to 65k context, zero gap to full-row selection. The attended tier is 0.25 KB of Kimi Linear's 8.1 KB per-token cach
    
[^29]: Headroom-Drift Replay：GRPO中一种实现原则性重放控制的原语

    Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO

    [https://arxiv.org/abs/2609.03941](https://arxiv.org/abs/2609.03941)

    该论文提出了一种面向GRPO的组级重放控制原语Headroom-Drift Replay，通过Headroom按剩余学习价值排序、Drift按策略兼容性门控来复用历史轨迹，在不改变在线数据流、不增加额外训练机制的前提下加速RL后训练，从而将重放本身的贡献与复杂训练流程解耦。

    

    基于强化学习的推理模型后训练正日益受到重复性全新轨迹生成（rollout）的瓶颈制约，尤其是在智能体环境中，环境交互主导了墙钟时间成本。重放可以通过复用过去的轨迹来减轻这一负担，但现有方法通常将重放嵌入到涉及探索、经验重构或混合策略优化的更大训练流程中，这使得重放本身的贡献难以被隔离。我们提出一个聚焦的问题：仅凭原则性的重放选择究竟能走多远？我们提出了Headroom-Drift Replay，一种面向GRPO的组级重放控制原语，它将复用拆分为两个决策：Headroom（头空间）根据剩余学习价值对存储的轨迹组进行排序，而Drift（漂移）则根据与当前策略的兼容性对其进行门控。新鲜的在线策略数据流保持不变，且该方法不引入任何辅助的生成或训练机制。在数学推理、多模……（原文摘要在此处截断）

    arXiv:2609.03941v1 Announce Type: cross  Abstract: RL-based post-training for reasoning models is increasingly bottlenecked by repeated fresh rollout generation, particularly in agentic settings where environment interaction dominates wall-clock cost. Replay can reduce this burden by reusing past trajectories, but existing methods typically embed it within larger training pipelines involving exploration, experience restructuring, or mixed-policy optimization. This makes replay's own contribution difficult to isolate. We ask a focused question: how far can principled replay selection alone go? We introduce Headroom-Drift Replay, a group-level replay control primitive for GRPO that separates reuse into two decisions. Headroom ranks stored groups by remaining learning value, while Drift gates them by compatibility with the current policy. The fresh on-policy stream remains unchanged, and the method adds no auxiliary generation or training machinery. Across mathematical reasoning, multimod
    
[^30]: RATL：从检索残差中学习以实现鲁棒的多元时间序列预测

    RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting

    [https://arxiv.org/abs/2609.03937](https://arxiv.org/abs/2609.03937)

    提出即插即用的残差检索与反馈校正方法RATL，通过冻结基础预测器、将其历史预测残差构建为专属记忆，并在推理时从相似历史情境中检索残差轨迹进行校正，从而实现更鲁棒的多元时间序列预测。

    

    检索增强生成（RAG）通过检索外部证据来补充参数化模型。同样的思想对连续输出回归也很有吸引力，但当样本在输出水平、数值尺度或局部动态上存在差异时，直接复用检索到的目标值往往并不鲁棒。此外，传统的预测流程通常仅将残差用于模型优化和误差诊断，而不会将个体的历史残差样本保留为可在推理时访问的记忆。针对多元时间序列预测，我们提出了RATL——一种即插即用的残差检索与反馈校正方法。RATL冻结一个基础预测器以构建检索键，并将其历史预测残差转化为该基础模型专属的仅训练阶段记忆。在推理时，RATL在因果可用性约束下从相似的历史上下文中检索残差轨迹，然后使用集合感（摘要在此处被截断）

    arXiv:2609.03937v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) complements parametric models with retrieved external evidence. The same idea is attractive for continuous-output regression, but directly reusing retrieved target values is often not robust when samples differ in output level, numerical scale, or local dynamics. Moreover, conventional forecasting pipelines generally use residuals for model optimization and error diagnosis, but do not retain individual historical residual examples as memory that can be accessed at inference time.For multivariate time-series forecasting, we propose RATL, a plug-in residual-retrieval and feedback-correction method. RATL freezes a base forecaster to construct retrieval keys and turns its historical forecast residuals into a train-only memory specific to that base model. At inference time, RATL retrieves residual trajectories from similar historical contexts subject to causal availability constraints, then uses a set-aw
    
[^31]: 基于稀疏自回归建模的多视角图像场景生成

    Sparse auto-regressive modeling for scene generation from multi-view images

    [https://arxiv.org/abs/2609.03931](https://arxiv.org/abs/2609.03931)

    SPAR3S提出了一种无需3D真值监督的稀疏体素对齐3D潜在生成模型，通过可微高斯泼溅从多视角图像学习稀疏潜在空间，并利用自回归建模实现从稀疏视角的完整3D场景生成。

    

    从稀疏、无约束的视角生成完整的3D场景是3D视觉领域的一项根本性挑战，它要求模型能够对观察内容之外进行推理，同时保持计算上的可行性。现有的前馈重建方法本质上受限于输入图像中可见的内容，而3D生成建模则受到稠密体积表示的高计算成本以及大规模3D监督数据稀缺的阻碍。我们提出了SPAR3S，一个稀疏体素对齐的3D潜在生成模型，用于条件场景补全，无需真值3D数据进行监督。我们的关键洞察是将3D场景生成构建在一个结构化、紧凑且体素对齐的3D潜在空间中，其中仅表示被占据的体素。我们通过可微3D高斯泼溅（3D Gaussian Splatting）的光度监督，直接从多视角图像中学习这一稀疏潜在空间。给定部分观察到的体素集合，模型（摘要在此处截断）

    arXiv:2609.03931v1 Announce Type: cross  Abstract: Generating complete 3D scenes from sparse, unconstrained views is a fundamental challenge in 3D vision which requires reasoning beyond observed content while remaining computationally tractable. Existing feed-forward reconstruction methods are inherently limited to content visible in the input images, while 3D generative modeling is hindered by the high computational cost of dense volumetric representations and the scarcity of large-scale 3D supervision. We introduce SPAR3S, a sparse voxel-aligned 3D latent generative model for conditional scene completion without requiring ground-truth 3D data for supervision. Our key insight is to formulate 3D scene generation in a structured, compact, voxel-aligned 3D latent space where only occupied voxels are represented. We learn this sparse latent space directly from multi-view images using photometric supervision via differentiable 3D Gaussian Splatting. Given a partial set of observed voxels e
    
[^32]: 比较学术导师发现中的检索方法：针对美国9所大学768位计算机科学教师档案的六种方法研究

    Comparing Retrieval Methods for Academic Advisor Discovery: A Six-Method Study of 768 CS Faculty Profiles Across 9 US Universities

    [https://arxiv.org/abs/2609.03901](https://arxiv.org/abs/2609.03901)

    该研究构建了一个包含美国9所大学768位计算机科学教师档案的新数据集，系统比较了六种信息检索方法在学术导师发现任务上的表现，发现重排序方法效果最佳（平均NDCG@10达0.477），且语义检索整体优于传统词汇匹配方法。

    

    我们针对学术导师发现任务——即根据研究生申请人的研究兴趣陈述对计算机科学教师进行相关性排序——对六种信息检索方法进行了比较评估。这些方法涵盖稀疏词汇匹配（Jaccard重叠度、TF-IDF、BM25）、稠密语义检索（all-MiniLM-L6-v2句子嵌入）、混合分数融合以及学习排序。评估采用一个新的特定领域数据集：从美国9个计算机科学系抓取的768份教师档案，并针对代表不同研究生研究画像的5个查询进行了162个分级相关性判定（等级0/1/2）。在全部五个查询中，Reranked（重排序）方法取得了最高的平均NDCG@10（0.477，标准差0.138），其后依次为Semantic（语义检索，0.450）、Hybrid（混合方法，0.421）、BM25（0.406）、Jaccard（0.303）和TF-IDF（0.246）。经过对全部15组两两比较进行Bonferroni校正后，TF-IDF显著差于BM25、Semantic、Hybrid和Reranked，而……

    arXiv:2609.03901v1 Announce Type: cross  Abstract: We present a comparative evaluation of six information retrieval methods for the task of academic advisor discovery: ranking CS faculty members by relevance to a graduate applicant's research interest statement. The methods span sparse lexical matching (Jaccard overlap, TF-IDF, BM25), dense semantic retrieval (all-MiniLM-L6-v2 sentence embeddings), hybrid score fusion, and learning-to-rank. Evaluation uses a new domain-specific collection: 768 faculty profiles scraped from 9 US CS departments, with 162 graded relevance judgments (grade 0/1/2) across 5 queries representing distinct graduate student research profiles. Across all five queries, Reranked achieves the highest mean NDCG@10 (0.477, std 0.138), followed by Semantic (0.450), Hybrid (0.421), BM25 (0.406), Jaccard (0.303), and TF-IDF (0.246). After Bonferroni correction across all 15 pairwise comparisons, TF-IDF is significantly worse than BM25, Semantic, Hybrid, and Reranked; no 
    
[^33]: 超越终点分数：持续知识更新的时间与容量条件化评估

    Beyond Endpoint Scores: Time- and Capacity-Conditioned Evaluation of Continual Knowledge Updating

    [https://arxiv.org/abs/2609.03900](https://arxiv.org/abs/2609.03900)

    该论文揭示了持续知识更新方法的优劣排名会随评估时间点和回放侧适配器容量的变化而发生逆转，证明仅依靠单一终点分数和常规秩设置无法判断哪种方法更优。

    

    持续知识更新方法通常仅基于一个最终检查点和一个常规适配器秩就被宣称更为优越。我们证明这种做法不足以识别出更优的工作点。在保持周期性层级结构固定的前提下，我们将其与累积回放方法在一个跨越24个月的Wikidata数据流上进行比较，同时改变评估月份、回放LoRA秩以及查询表述方式。在这一区域内，表面的胜出者会发生变化：在Qwen2.5-1.5B模型上，层级结构相对于秩8回放的5.0分优势，在面对秩72回放时变成了11.6分的劣势；而在高秩设置下，与巩固阶段对齐的终点评估可能显示两者持平，但按时间平均的回放方法实际上领先9-13分。同样的秩条件化排名反转现象在Llama-3.2-1B模型和保留的改写查询上也得到了验证。这些结果表明，持续更新中的方法排名可能同时取决于性能测量的时间点，以及基线所获得的回放侧适应容量大小。

    arXiv:2609.03900v1 Announce Type: new  Abstract: Continual knowledge-updating methods are often declared superior from one final checkpoint and one conventional adapter rank. We show that this can be insufficient to identify the better operating point. Holding a periodic hierarchy fixed, we compare it with cumulative replay over a 24-month Wikidata stream while varying evaluation month, replay LoRA rank, and query formulation. The apparent winner changes across this region: on Qwen2.5-1.5B, the hierarchy's 5.0-point advantage over rank-8 replay becomes an 11.6-point deficit against rank-72 replay, and at high ranks a consolidation-aligned endpoint can suggest a tie while time-averaged replay leads by 9-13 points. The same rank-conditioned reversal appears on Llama-3.2-1B and held-out paraphrases.   These results show that method ranking in continual updating can depend jointly on when performance is measured and how much replay-side adaptation capacity the baseline receives. We therefo
    
[^34]: 面向数值数据可解释异常检测的可微区间瓶颈

    Differentiable Interval Bottlenecks for Interpretable Anomaly Detection in Numerical Data

    [https://arxiv.org/abs/2609.03878](https://arxiv.org/abs/2609.03878)

    提出DIFFINT自编码器，通过可微的软区间瓶颈结构实现可解释的异常检测，每个潜在单元对应特征空间中人类可读的超矩形，并提供认证的重构误差下界。

    

    基于重构的异常检测器虽然准确但不透明：深度自编码器在标记一个样本为异常时，不会告诉从业者是哪些特征范围导致了异常。我们提出了DIFFINT，这是一种自编码器，其潜在瓶颈被结构化为一组软性的、轴对齐的区间隶属关系，可直接从原始数值数据端到端学习，无需任何离散化或二值化处理。每个潜在单元对应于特征空间中一个人类可读的超矩形；一个实例通过它相对于其他单元落入每个区间的强度来进行编码，其重构误差即为异常分数。这种方法既保留了可微表示学习的强大能力，又暴露出可检查的内部结构。我们精确地阐述了这种归纳偏置：对于落在学习到的支撑域每个活动坐标之外的点（配合经Lipschitz约束的解码器），提供了经认证的重构误差下界，以及分级的、经验……（摘要在此处截断）

    arXiv:2609.03878v1 Announce Type: cross  Abstract: Reconstruction-based anomaly detectors are accurate but opaque: a deep autoencoder flags a sample without telling a practitioner which feature ranges made it anomalous. We propose DIFFINT, an autoencoder whose latent bottleneck is structured as a set of soft, axis-aligned interval memberships learned end-to-end directly from raw numerical data, without any discretization or binarization. Each latent unit corresponds to a human-readable hyper-rectangle in feature space; an instance is encoded by how strongly it falls inside each interval relative to the other units, and its reconstruction error is the anomaly score. This keeps the power of differentiable representation learning while exposing an inspectable internal structure. We make the inductive bias precise: a certified reconstruction-error lower bound for points that fall outside every active coordinate of the learned support (with a Lipschitz-enforced decoder), and a graded, empir
    
[^35]: 注意力索引模型的高维学习动力学

    High-Dimensional Learning Dynamics of Attention-Indexed Models

    [https://arxiv.org/abs/2609.03858](https://arxiv.org/abs/2609.03858)

    本文提出注意力索引模型这一统一框架，证明高维极限下损失景观由有限的迹阶序参数刻画、SGD动力学可被有限截断系统以指数精度逼近，并揭示注意力参数化本身作为架构隐式偏置可诱导自动对称性破缺。

    

    注意力机制是现代基础模型的核心，然而其训练动力学仍然知之甚少，尤其是当注意力矩阵具有广泛秩的时候。在本工作中，我们研究了注意力索引模型，这是一个能够表示多层和多头注意力架构的宽泛框架。首先，我们证明，在适当的高维极限下，总体损失景观由一组有限的迹阶序参数刻画。相比之下，在线随机梯度下降（SGD）则由一个无限的矩阵矩层级所支配，我们证明该层级可以被一个有限截断系统以指数级精度良好逼近。其次，该框架揭示了注意力参数化本身可以充当一种架构层面的隐式偏置。直接优化注意力矩阵 $S\in\mathbb{R}^{d\times d}$ 可能会一直陷于无信息状态。而绑定注意力（$S=WW^\top$）则会诱导自动的对称性破缺。

    arXiv:2609.03858v1 Announce Type: new  Abstract: Attention mechanisms are central to modern foundation models, yet their training dynamics remain poorly understood, especially when the attention matrices have extensive rank. In this work, we study attention-indexed models, a broad framework that can represent multi-layer and multi-head attention architectures. First, we show that, in a suitable high-dimensional limit, the population-loss landscape is characterized by a finite set of trace order parameters. In contrast, online stochastic gradient descent (SGD) is governed by an infinite hierarchy of matrix moments, which we show can be exponentially well-approximated by a finite truncated system. Second, this framework reveals that attention parameterization itself can act as an architectural implicit bias. Direct optimization of an attention matrix $S\in\mathbb{R}^{d\times d}$ can remain trapped in an uninformative state. Tied attention ($S=WW^\top$) induces an automatic symmetry-break
    
[^36]: 推进（决策）边界：在联邦学习中动态校准差分隐私噪声以提升可解释性

    Pushing the (Decision) Boundaries: Dynamically Calibrating Differentially Private Noise to Explainability in Federated Learning

    [https://arxiv.org/abs/2609.03851](https://arxiv.org/abs/2609.03851)

    提出XCal-FL算法，通过预测logit变化、反事实边距和显著性集中度三种互补信号，在联邦学习训练过程中动态校准差分隐私噪声，在保护隐私的同时保持模型解释的保真度。

    

    采用差分隐私（DP）的联邦学习（FL）在分布式机器学习中被日益广泛地采用，以保护数据机密性。然而，DP噪声会扭曲学习到的表示并降低解释的保真度，这限制了差分隐私联邦学习在需要可信解释的场景（如辅助临床诊断）中的应用。先前的工作使用静态特征重要性信号来调整DP噪声，这将可解释性局限于事后分析，且无法在训练过程中根据解释质量进行噪声校准。我们提出了XCal-FL，这是一种面向跨筒仓（cross-silo）联邦学习中图像分类任务的闭环、可解释性驱动的本地训练算法，它基于三种互补信号动态校准DP噪声：（1）预测logit变化，衡量对模型置信度的因果影响；（2）反事实边距，捕捉决策边界敏感性；（3）显著性集中度，量化模型关注的空间一致性……（原文摘要截断）

    arXiv:2609.03851v1 Announce Type: new  Abstract: Federated Learning (FL) with Differential Privacy (DP) is increasingly adopted to preserve data confidentiality in distributed machine learning. However, DP noise distorts learned representations and degrades explanation fidelity, limiting differentially private FL where trustworthy explanations are required, such as assistive clinical diagnosis. Prior work adapted DP noise with static feature-importance signals, restricting explainability to post hoc analysis and precluding noise calibration to explanation quality during training. We propose XCal-FL, a closed-loop, explainability-driven local training algorithm for image classification in cross-silo FL that dynamically calibrates DP noise from three complementary signals: (1) prediction logit variations, measuring causal influence on model confidence, (2) counterfactual margins, capturing decision-boundary sensitivity, and (3) saliency concentration, quantifying spatial coherence of mod
    
[^37]: 相同可加估值下EF1约束的纳什社会福利：复杂度、保证与实验

    EF1-Constrained Nash Social Welfare with Identical Additive Valuations: Complexity, Guarantees, and Experiments

    [https://arxiv.org/abs/2609.03846](https://arxiv.org/abs/2609.03846)

    该论文证明在相同可加估值下EF1约束的NSW最大化问题是强NP完全的，并识别出获得更强福利保证的条件——均匀估值下每个EF1分配都是NSW最优的，而在ε-小物品条件下每个EF1分配能达到 1-O(ε²) 的显式近似比。

    

    我们研究在具有相同可加估值的智能体之间分配不可分割物品的问题，重点关注“最多差一件物品的无嫉妒性”（EF1）和纳什社会福利（NSW）。由于在可加估值下每个最大NSW分配都是EF1的，相应的阈值问题继承了相同可加估值下NSW最大化的已知强NP困难性，因而该问题是强NP完全的。因此，我们转而研究任意EF1分配所满足的福利保证。尽管已知每个这样的分配都能实现对无约束最优NSW的 e^{-1/e} 近似，我们识别出了能产生更强保证的条件。在均匀估值下，每个EF1分配都是NSW最优的。在ε-小物品条件下，每个EF1分配都达到一个显式的近似比 ρ_n(ε)，且当 n 固定、ε→0 时满足 ρ_n(ε) = 1-O(ε²)。我们进一步考虑……（摘要在此处被截断）

    arXiv:2609.03846v1 Announce Type: cross  Abstract: We study the allocation of indivisible goods among agents with identical additive valuations, focusing on envy-freeness up to one good (EF1) and Nash social welfare (NSW). Since every maximum-NSW allocation is EF1 under additive valuations, the associated threshold problem inherits the known strong NP-hardness of NSW maximization under identical additive valuations and is strongly NP-complete. We therefore focus on welfare guarantees satisfied by arbitrary EF1 allocations. Although every such allocation is known to achieve an $e^{-1/e}$-approximation to the unrestricted optimal NSW, we identify conditions yielding stronger guarantees. Under uniform valuations, every EF1 allocation is NSW-optimal. Under an $\varepsilon$-small-item condition, every EF1 allocation achieves an explicit approximation ratio $\rho_n(\varepsilon)$ satisfying $\rho_n(\varepsilon) = 1-O(\varepsilon^2)$ as $\varepsilon\to 0$ for fixed $n$.   We further consider t
    
[^38]: 翻转而非打乱：以推理速度为LLM添加水印

    Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference

    [https://arxiv.org/abs/2609.03844](https://arxiv.org/abs/2609.03844)

    提出无状态伯努利水印（SBW），通过每词元独立伯努利试验实现O(1)复杂度的绿名单判断，检测速度比KGW自盐值快6000倍以上、比SynthID快2倍，同时保持相同的N(0,1)统计检测保证。

    

    我们提出了无状态伯努利水印（SBW），这是一种针对大语言模型的新型统计水印方法，它通过每个词元独立的伯努利试验来确定绿名单成员资格。与KGW的词表置换或SynthID的多层锦标赛机制不同，SBW只需对每个词元与基于计数器的随机数生成器进行一次比较，将成员判断复杂度降至O(1)，并实现了零中间内存分配的单内核执行。我们证明了这种形式化方法保持了与固定大小绿名单相同的检测保证：在零假设下，z分数检验仍服从N(0,1)分布。这种无状态架构实现了现有方法无法企及的能力：全词表自盐值水印（比KGW的自盐值方法快6000倍以上，尽管使用候选相关种子对整个词表进行偏置，仍比SynthID快2倍），以及与蒸馏架构的兼容性。

    arXiv:2609.03844v1 Announce Type: cross  Abstract: We introduce Stateless Bernoulli Watermarking (SBW), a new statistical watermark for Large Language Models that determines green list membership through independent per-token Bernoulli trials. Unlike KGW's vocabulary permutation or SynthID's multi-layer tournament, SBW requires only a single comparison per token against a counter-based random number generator, reducing membership complexity to $O(1)$ and enabling single-kernel execution with zero intermediate allocations. We prove that this formulation preserves the same detection guarantees as fixed-size green lists: the z-score test remains $\mathcal{N}(0,1)$ under the null. The stateless architecture enables capabilities unavailable to existing methods: full-vocabulary self-salt watermarking (over 6000$\times$ faster than KGW's self-salt and 2$\times$ faster than SynthID despite biasing the entire vocabulary with candidate-dependent seeding) and architectural compatibility with dist
    
[^39]: 离线强化学习中的多步近端策略改进

    Multi-step Proximal Policy Improvement in Offline Reinforcement Learning

    [https://arxiv.org/abs/2609.03842](https://arxiv.org/abs/2609.03842)

    本文提出将离线actor更新统一解释为概率流形上的单步近端策略改进，并在此基础上提出多步近端策略改进（MPI）这一即插即用机制，通过组合连续的重新居中近端步骤，实现了超越数据集支持范围的受控策略改进。

    

    离线强化学习（RL）必须调和两个相互冲突的需求：策略更新应保持在数据集支持的动作附近以确保价值估计的可靠性，然而有意义的性能提升往往需要超越行为分布。我们通过将策略建模为赋予特定度量几何的概率流形，为离线actor更新建立了一种几何视角。在这一视角下，一大类离线actor目标可以被解释为单步近端策略改进（SPI），即由critic定义的能量所诱导的流形梯度流的隐式离散化。基于这一洞察，我们提出了多步近端策略改进（MPI），这是一种即插即用的精化机制，通过组合连续的重新居中近端步骤来工作。MPI能够在数据集支持范围之外实现受控的策略改进，同时在每次精化中保持近端控制。该框架可兼容多种策略几何。

    arXiv:2609.03842v1 Announce Type: new  Abstract: Offline reinforcement learning (RL) must reconcile two competing requirements: policy updates should stay near dataset-supported actions to keep value estimates reliable, yet meaningful gains often require moving beyond the behavior distribution. We develop a geometric view of offline actor updates by modeling policies as a probability manifold endowed with a chosen metric geometry. Under this lens, a broad class of offline actor objectives can be interpreted as a single proximal policy improvement step (SPI), i.e., an implicit discretization of a manifold gradient flow induced by a critic-defined energy. Building on this insight, we propose multi-step proximal policy improvement (MPI), a plug-in refinement mechanism that composes sequential re-centered proximal steps. MPI enables controlled policy improvement beyond dataset support while retaining proximal control at each refinement. The framework accommodates multiple policy geometries
    
[^40]: 语义贝叶斯世界模型

    Semantic Bayesian World Models

    [https://arxiv.org/abs/2609.03834](https://arxiv.org/abs/2609.03834)

    该论文提出语义贝叶斯世界模型（SBWM）的愿景，将知识图谱从静态的事实数据库转变为共享且演化的概率信念体系——由本体公理约束先验、贝叶斯条件化更新信念、动作干预世界——从而弥合概率推理的基础模型与确定性知识图谱之间的鸿沟，实现统一的推理架构。

    

    知识图谱以清晰的断言描述现实，而如今消费这些知识的系统——基础模型与自主智能体——却原生地以概率方式进行推理。我们认为，这种不匹配正是语言模型与知识图谱的融合至今仍停留在数据供给管道层面、而未能形成统一推理架构的原因。我们展望了语义贝叶斯世界模型（SBWMs）：一个不再将世界描述为事实数据库，而是描述为知识图谱之上共享且持续演化的信念网络的Web，其中本体公理约束先验，观测通过贝叶斯条件化更新信念，动作则对世界进行干预。我们深入探讨了智能体能从这样的模型中获得什么：一个正在判断门口人影是快递员还是窃贼的家庭安防智能体、一个通过逻辑蕴含而非字符串频率进行聚合的精算估计、一个语言模型往往无法可靠完成的规划任务，以及对数量的估计……

    arXiv:2609.03834v1 Announce Type: new  Abstract: Knowledge graphs describe reality in crisp assertions, while the systems now consuming them, foundation models and autonomous agents, reason natively in probabilities. We argue that this mismatch is why the integration of language models and knowledge graphs remains a data-feeding pipeline rather than a unified reasoning architecture. We envision Semantic Bayesian World Models (SBWMs): a Web that describes the world not as a database of facts but as a shared, evolving fabric of beliefs over knowledge graphs, where ontological axioms constrain priors, observations update beliefs by Bayesian conditioning, and actions intervene upon the world. We work through what an agent gains from such a model: a home-security agent deciding whether the figure at the gate is a courier or a burglar, an actuarial estimate aggregated by entailment rather than by string frequency, a planning task that language models reliably fail, and the estimation of quan
    
[^41]: 证人解释异常

    Witnesses Explain Anomalies

    [https://arxiv.org/abs/2609.03826](https://arxiv.org/abs/2609.03826)

    WAND是一种天生可解释的无监督表格数据异常检测器，它通过单位球面上的方向进行评分，而标记异常点的“证人”方向本身就构成了该点的逐特征解释，无需借助SHAP或LIME等事后解释方法。

    

    无监督异常检测对未标记且受污染样本中的每个数据点进行一次性评分，而且如今越来越多地还需要解释某个点为何被标记。然而，主流的检测器只给出分数，却不说明是哪些特征驱动了这一分数，解释只能通过事后附加SHAP或LIME来实现，这些方法需要对每个点重新查询检测器数千次，且只能近似检测结果。我们提出了WAND，一种可解释性源于设计本身的无监督表格数据异常检测器。WAND的计算围绕单位球面上的方向来组织，通过每个点的投影超出亚高斯极值基线的距离来为其评分。我们方法的原创性在于：标记某个异常点的“证人”方向作为特征空间中的向量，本身就是该点的解释——这种逐特征归因无需在评分之外付出任何额外成本即可获得，并且由于分数是可微分的，还可以通过梯度恢复得到。评分计算关于样本量是线性的，并且……

    arXiv:2609.03826v1 Announce Type: cross  Abstract: Unsupervised anomaly detection scores each point of an unlabelled, contaminated sample in a single pass, and increasingly must also explain why a point is flagged. Yet the dominant detectors give a score with no account of which features drive it, and explanations are bolted on post-hoc with SHAP or LIME, which re-query the detector thousands of times per point and only approximate it. We introduce WAND, an unsupervised tabular anomaly detector that is explainable by design. WAND organises its computation around directions on the unit sphere, scoring each point by how far its projection escapes a sub-Gaussian extreme-value baseline. The originality of our approach is that the witness directions that flag a point, being vectors in feature space, are its explanation, a per-feature attribution obtained at no cost over scoring and, since the score is differentiable, recoverable by gradients. Scoring is linear in the sample size, and a prob
    
[^42]: 当视觉遇见图：图推理与学习综述

    When Vision Meets Graphs: A Survey on Graph Reasoning and Learning

    [https://arxiv.org/abs/2609.03816](https://arxiv.org/abs/2609.03816)

    该综述首次系统性地提出了“视觉遇见图”这一新兴研究领域，将图的视觉呈现作为一等输入用于推理和学习，旨在弥合图学习流程与图可视化之间的长期差距。

    

    图是自然科学和社会科学中许多问题的基础数据结构。在过去十年中，图神经网络（GNNs）在坚实的理论基础支持下主导了图机器学习领域。然而，科学家们往往通过视觉来理解图结构：化学家阅读分子图示，社会科学家检视网络可视化。尽管图可视化研究已有数十年的历史，但大多数图学习流程仍然将图纯粹视为符号结构，很少利用图的视觉形式。我们认为，在强大的视觉模型和视觉-语言模型时代，这一差距值得重新关注。本综述首次系统性地概述了这个新兴领域，我们将其称为“视觉遇见图”，该领域将图的视觉呈现作为推理和学习的一等输入。我们将现有工作组织为三个方向。其中“视觉用于图推理”研究模型如何……

    arXiv:2609.03816v1 Announce Type: cross  Abstract: Graphs are a fundamental data structure underlying many problems in the natural and social sciences. Over the past decade, Graph Neural Networks (GNNs) have dominated graph machine learning, supported by solid theoretical foundations. Yet scientists often understand graph structure through vision: chemists read molecular diagrams and social scientists inspect network visualizations. Despite decades of work on graph visualization, most graph learning pipelines still treat graphs purely as symbolic structures, rarely leveraging the visual form of graphs. We argue that this gap deserves renewed attention in the era of powerful vision and vision-language models. This survey provides a first systematic overview of the emerging area we term vision meets graphs, which treats visual depictions of graphs as first-class inputs for reasoning and learning. We organize existing work into three threads. Vision for Graph Reasoning studies how models 
    
[^43]: 一种面向移动网络站点能源低效识别的同行相对表示学习框架

    A Peer-Relative Representation Learning Framework for Energy Inefficiency Identification in Mobile Network Sites

    [https://arxiv.org/abs/2609.03809](https://arxiv.org/abs/2609.03809)

    本文提出一种无监督的同行相对表示学习框架，通过能耗感知的最小失真嵌入将异常高能耗站点与相似站点在嵌入空间中分离，从而在缺乏真实低效标签的情况下识别移动网络站点的能源低效问题。

    

    能源消耗是移动网络运营商最大的运营支出项目之一，然而站点级的能源低效问题——如故障的冷却控制器、空闲的无线设备以及寄生性辅助负载——往往难以被发现，原因在于不存在真实值的低效标签，且历史测量数据本身可能已经包含嵌入其中的低效情况。本研究提出了一种无监督的同行相对方法，其基本前提是具有相似结构和运行特征的站点应表现出相当的能耗水平。为了捕捉这些关系，本文引入了一种新颖的能耗感知最小失真嵌入公式，通过基于能耗的排斥机制扩展了标准的MDE目标函数。这促使那些相对于可比同行表现出异常高能耗的站点在嵌入空间中偏离其局部邻域。由此产生的低维……

    arXiv:2609.03809v1 Announce Type: new  Abstract: Energy consumption is one of the largest operational expenditure items for mobile network operators, yet site-level energy inefficiencies such as faulty cooling controllers, idle radio equipment, and parasitic auxiliary loads often remain undetected because no ground-truth inefficiency labels exist and historical measurements may already contain embedded inefficiencies. This study proposes an unsupervised peer-relative approach based on the premise that sites with similar structural and operational characteristics should exhibit comparable energy consumption. To capture these relationships, a novel energy-aware Minimum Distortion Embedding (MDE) formulation is introduced that extends the standard MDE objective with an energy-based repulsion mechanism. This encourages sites with anomalously high energy consumption relative to comparable peers to become displaced from their local neighbourhoods in the embedding space. The resulting low-dim
    
[^44]: 免费暂停标记

    Free Pause Tokens

    [https://arxiv.org/abs/2609.03807](https://arxiv.org/abs/2609.03807)

    提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。

    

    免费暂停标记为语言模型提供额外的计算量来形成每个下一个词元预测（如同暂停标记或思考标记的作用），但它不是在序列中添加额外标记，而是通过权重共享主干上的并行预测流来承载这些计算。在实际应用中，它使一个10亿参数模型的下一个词元预测提升了2-3厘奈特。由于暂停是搭载在已有位置上而非新增位置，因此使用它是免费的：在推理时它不增加上下文长度、不增加KV缓存，且几乎不产生延迟，推理浮点运算量的增长通常无关紧要，因为它并非吞吐量的主动瓶颈。唯一的主要成本在于训练阶段：与优化过的预训练流程相比，额外的训练计算量可低至1.14倍，同时保留了大部分收益。其结果是在与标准下一个词元训练的Transformer等浮点运算、等参数量和等词元数量条件下取得的性能改进。

    arXiv:2609.03807v1 Announce Type: cross  Abstract: A free pause token gives a language model extra compute to form each next-token prediction (as a pause, or thinking, token does) but carries that compute in a parallel prediction stream over a weight-shared backbone rather than as an extra token in the sequence. It improves next-token prediction by 2-3 centinats in practice on a 1B parameter model. Because the pause rides an existing position instead of adding one, it is free to use: at inference it adds no context length, no KV cache, and essentially no latency with the growth in inference flops typically irrelevant as it is not the active bottleneck on throughput. The only primary cost is in training, where additional training compute versus an optimized pretraining pipeline is reduced to as low as x1.14 while preserving most of the benefits. The result is an isoflop, isoparameter, and isotoken improvement over standard next token trained transformers.
    
[^45]: 从有序伯努利层级到临界线几何：整数量子化、伯努利残余相位与素数幂谱

    From Ordered Bernoulli Levels to Critical-Line Geometry: Integer Quantization, Bernoulli Residual Phase, and Prime-Power Spectra

    [https://arxiv.org/abs/2609.03801](https://arxiv.org/abs/2609.03801)

    本文通过有序伯努利核的水平集几何，将黎曼临界线零点纵坐标精确量子化为整数壳层与周期性一阶伯努利残余相位之和，并借助唯一因子分解进一步解析出素数幂谱结构。

    

    我们研究有序伯努利词核 f(p,n,k)=p^k(1-p)^(n-k) 及其倒数整数水平集所生成的几何。二元水平 2^(-n) 选择 p=1/2 作为唯一与分割方式无关的实数锚点。在保持补集结构的复延拓下，这一对量变为 z=1/2+iu 与 1-z=1/2-iu，从而在任何 zeta 函数输入被引入之前，就产生了一种共轭对称的垂直几何。二次坐标 Q(z)=z(1-z)=1/4+u^2 在中心点处具有尖锐极小值，并允许精确的整数量子化。对于临界线零点纵坐标 gamma_k，所诱导的水平 L_k=1/4+gamma_k^2 被精确分解为 L_k=N_k+delta_k，其中 N_k 是最近的整数，delta_k 是周期性的一阶伯努利残余。圆化操作给出 Z_k=exp(2 pi i delta_k)，将 gamma_k^2 mod 1 分离为残余相位变量。唯一因子分解将整数壳层解析为素数生成元坐标，而一个不同的复……（摘要在此处截断）

    arXiv:2609.03801v1 Announce Type: new  Abstract: We study the ordered Bernoulli-word kernel f(p,n,k)=p^k(1-p)^(n-k) and the geometry generated by its inverse-integer level sets. The binary level 2^(-n) selects p=1/2 as the unique real split-independent anchor. Under complement-preserving complex continuation, the pair becomes z=1/2+iu and 1-z=1/2-iu, producing a conjugation-symmetric vertical geometry before any zeta-function input is introduced. The quadratic coordinate Q(z)=z(1-z)=1/4+u^2 has a sharp minimum at the central point and admits an exact integer quantization. For critical-line zero ordinates gamma_k, the induced levels L_k=1/4+gamma_k^2 are decomposed exactly as L_k=N_k+delta_k, where N_k is the nearest integer and delta_k is a periodic first-Bernoulli residual. Circularization gives Z_k=exp(2 pi i delta_k), isolating gamma_k^2 mod 1 as the residual phase variable. Unique factorization resolves the integer shells into prime-generator coordinates, while a distinct complex e
    
[^46]: 基于地标点的分钟级多模态足球监测数据中损伤相关运动员场次的判别

    Landmark-Based Discrimination of Injury-Associated Athlete-Sessions from Minute-Resolution Multimodal Football Monitoring Data

    [https://arxiv.org/abs/2609.03790](https://arxiv.org/abs/2609.03790)

    本文提出一种基于固定地标点的建模方法，在每个地标时刻（如10、20、30分钟）利用截至该时刻的观测信息为每个运动员-场次构建单一表示，从而在损伤信息仅有场次级标签的情况下，避免不合理的分钟级损伤监督，实现从分钟级多模态足球监测数据中判别损伤相关场次。

    

    运动员监测数据可以在整场比赛或训练课中逐分钟记录，而损伤信息可能仅能表明整个场次是否与损伤相关。这就产生了一个建模问题：如果将相同的场次级标签分配给每一分钟，就意味着每个精确时刻的损伤状态都是已知的，但实际上场次内的损伤发生时间是无法得知的。本文的创新之处在于提出了一种固定地标、每个运动员-场次仅构建一个表示的公式化方法，直接解决了这种不匹配问题。我们不再对每一分钟进行标注，而是在每个地标点处，利用截至该时刻所观测到的信息，为每个运动员-场次构建一个表示。这种方法将预测目标保持在场次级别，避免了缺乏依据的分钟级损伤监督。地标点是同一场次内的固定时间点，例如10分钟、20分钟或30分钟。在每个地标点处，我们评估整个场次是与损伤相关还是与损伤无关。

    arXiv:2609.03790v1 Announce Type: new  Abstract: Athlete monitoring data may be recorded minute by minute throughout a match or training session, while injury information may only indicate whether the entire session was injury-associated.   This creates a modelling problem: assigning the same session-level label to every minute would imply that injury status is known at each exact time, even though within-session injury onset is unknown.   Our novelty is a fixed-landmark, one-representation-per-athlete-session formulation that directly addresses this mismatch. Instead of labelling every minute, we construct one representation per athlete-session at each landmark using information observed up to that point. This keeps the target at the session level and avoids unsupported minute-level injury supervision.   A landmark is a fixed time point within the same session, such as 10, 20, or 30 minutes. At each landmark, we assess whether the whole session is injury-associated or non-injury-assoc
    
[^47]: OBER+：成果导向教育中具备连续性感知的报告与可追溯的持续改进

    OBER+: Continuity-Aware Reporting and Traceable Continuous Improvement in Outcome-Based Education

    [https://arxiv.org/abs/2609.03770](https://arxiv.org/abs/2609.03770)

    OBER+通过五个相互衔接的阶段将测得的成果差距转化为可追溯且经评估的纠正措施，并引入连续性规则以避免在成果表述变更时误读达成度趋势，从而在成果导向教育中实现从测量到改进的闭环。

    

    实施成果导向教育的机构通常会例行计算学习成果的达成度，然而课程分析领域的综述指出，这种计算如何为决策提供依据方面缺乏证据。本文提出了OBER+，这是对一个已部署的机构级达成度平台的扩展，它计算了从测得的差距到经过评估的纠正措施之间的步骤。五个相互衔接的阶段分别完成：在课程的多次授课中累积达成度；发出差距及持续性差距信号；按照监管机构已经在使用的截止标准对其进行分级；将决策记录在附有证据注释的实践目录中；记录变更；并量化差距的后续变化。另一条规则会对同一成果的先后表述进行比较，因此当成果表述发生变化时，达成度不会被误读为跨越该变化点的连续序列。将这些规则应用于两门真实课程的实时记录产生了三个结果……

    arXiv:2609.03770v1 Announce Type: cross  Abstract: Institutions practising outcome-based education compute learning outcome attainment routinely, while reviews of curriculum analytics report an absence of evidence on how that computation informs decisions. This paper presents OBER+, an extension of a deployed institutional attainment platform that computes the step from a measured shortfall to an evaluated corrective action. Five connected stages accumulate attainment across deliveries of a course, signal a shortfall and a persistent shortfall, grade it on cutoffs the regulator already uses, record the decision against a catalogue of practices annotated with their evidence, log the change, and quantify the subsequent movement in the shortfall. A further rule compares successive statements of an outcome, so attainment is never read as a series across a point at which the outcome changed. Applying the rules to the live record of two real courses produced three results. Every outcome of a
    
[^48]: 从临近预报到天气预报：将再分析训练的模型适配至卫星观测

    From Nowcasting to Forecasting: Adapting a Reanalysis-Trained

    [https://arxiv.org/abs/2609.03763](https://arxiv.org/abs/2609.03763)

    本文开发了CloudCast v2机器学习模型，通过在再分析数据上训练学习云演变动态，并利用条件流匹配方法将其适配到卫星观测云场，实现了基于观测初始条件的12小时云量预报。

    

    准确的云量预报对于温度预测、辐射预报以及太阳能发电运营都非常重要。短临预报方法可以在最初的预报时段内保持观测到的云的位置，但当云场经历形成、消散和变形等演变过程时，其预报技巧会下降。更长的预报时效需要考虑大气演变，但业务化的数值天气预报（NWP）在初始化时可能无法准确刻画卫星观测到的云状态。我们开发了CloudCast v2，这是一个基于观测初始条件进行12小时云量预报的机器学习模型。该模型首先在哥白尼欧洲区域再分析数据集上训练以学习云演变动态，随后采用条件流匹配（conditional flow matching）——一种将噪声转化为云量预报的生成式方法——使其适配到卫星衍生的云场。

    arXiv:2609.03763v1 Announce Type: new  Abstract: Accurate cloud-cover forecasts are important for temperature prediction, radiation forecasting, and solar-power operations. Short-range forecasting methods can preserve observed cloud placement during the first forecast hours, but their skill decreases when cloud fields evolve through formation, dissipation and deformation. Longer lead times require accounting for atmospheric evolution, but operational numerical weather prediction (NWP) forecasts may not accurately represent the satellite-observed cloud state at initialization. We develop CloudCast v2, a machine-learning model for 12-hour cloud-cover forecasting from observation-based initial conditions. The model is first trained on the Copernicus European Regional Reanalysis (Ridal2024) to learn cloud-evolution dynamics, and is then adapted to satellite-derived cloud fields using conditional flow matching (Lipman2023), a generative method that transforms noise into cloud-cover forecast
    
[^49]: 用于Bures-Wasserstein重心的投影黎曼梯度下降：单位步长下与维度无关的线性收敛

    Projected Riemannian Gradient Descent for the Bures-Wasserstein Barycenter: Dimension-Independent Linear Convergence at Unit Step Size

    [https://arxiv.org/abs/2609.03762](https://arxiv.org/abs/2609.03762)

    该论文提出投影黎曼梯度下降算法，在单位步长下实现了Bures-Wasserstein重心计算的与维度无关的线性收敛，解决了快速收敛与维度无关保证之间的两难困境。

    

    正定矩阵集合的Bures-Wasserstein（BW）重心的计算广泛应用于机器学习、最优传输和量子信息领域。单位步长的黎曼梯度下降（RGD）——即实践中使用的定点迭代——收敛迅速，然而现有分析呈现出一种两难困境：单位步长的收敛保证在最坏情况下对维度呈指数依赖，而与维度无关的收敛保证则需要牺牲实际经验速度的小步长。我们解决了这一两难困境，不是通过改进单位步长RGD的保证，而是通过提出一种投影RGD算法，在单位步长下实现了与维度无关的线性收敛。所达到的收敛率为 $(1 - \kappa^{-3/2})$，其中 $\kappa$ 是矩阵集合的条件数，相较于最佳的小步长保证也在多项式意义上有所改进（迭代复杂度从 $\kappa^{5/2}$ 降至 $\kappa^{3/2}$）。其关键在于一种……

    arXiv:2609.03762v1 Announce Type: new  Abstract: The computation of the Bures-Wasserstein (BW) barycenter of an ensemble of positive definite matrices arises throughout machine learning, optimal transport, and quantum information. Riemannian gradient descent (RGD) at unit step size -- the fixed-point iteration used in practice -- converges rapidly, yet existing analyses present a dichotomy: unit-step guarantees carry worst-case exponential dependence on the dimension, while dimension-independent guarantees require small step sizes that forfeit the empirical speed. We resolve this dichotomy, not by improving the guarantees for unit-step RGD, but by proposing a Projected RGD algorithm that achieves dimension-independent linear convergence at unit step size. The achieved rate, $(1 - \kappa^{-3/2})$, where $\kappa$ is the condition number of the ensemble, also polynomially improves on the best small-step guarantee ($\kappa^{3/2}$ versus $\kappa^{5/2}$ iteration complexity). The crux is a n
    
[^50]: 基于融合前边剪枝的可处理贝叶斯网络融合的遗传算法

    Genetic Algorithms for Tractable Bayesian Network Fusion via Pre-Fusion Edge Pruning

    [https://arxiv.org/abs/2609.03724](https://arxiv.org/abs/2609.03724)

    本文提出一种基于遗传算法的贝叶斯网络融合共识框架，通过融合前边剪枝在优先保留输入网络间共享依赖结构的同时控制树宽，实现了计算上可处理且不易过拟合的网络融合。

    

    贝叶斯网络（BN）融合将多个输入网络合并为单一结构，在依赖关系保持与计算可处理性之间取得平衡。虽然无限制融合能保留所有依赖关系，但通常会产生树宽过高、结构过于复杂的网络，从而影响推理的可扩展性。受限融合通过剪枝边来控制树宽以缓解这一问题，但存在对输入特定噪声过拟合以及遗漏原始贝叶斯网络中依赖关系的风险。本文提出了一种共识框架，该框架在强制执行树宽约束的同时，优先考虑输入网络之间的共享结构，以确保良好的共识。我们提出了采用先进初始化策略、专用算子和定制适应度函数的遗传算法。此外，我们将现有方法适配到该问题上，并实现了贪心基线算法用于基准测试和进一步优化。在合成和真实世界贝叶斯网络上的实验表明了该方法的优越性。

    arXiv:2609.03724v1 Announce Type: cross  Abstract: Bayesian Network (BN) fusion combines multiple input networks into a single structure, balancing dependency preservation with computational tractability. While unrestricted fusion retains all dependencies, it often results in overly complex networks with high treewidth, which affects inference scalability. Limited fusion mitigates this by pruning edges to control treewidth but risks overfitting to input-specific noise and omitting dependencies from the original BNs. This paper introduces a consensus framework that prioritizes shared structures among input networks while enforcing treewidth constraints, ensuring a good consensus. We propose genetic algorithms with advanced initialization, specialized operators, and a tailored fitness function. Additionally, we adapt existing methods to this problem and implement greedy baselines for benchmarking and further optimization. Experiments on synthetic and real-world BNs show the superiority o
    
[^51]: 用于数据中心能源优化的人工智能

    Artificial Intelligence for Energy Optimization in Data Centers

    [https://arxiv.org/abs/2609.03716](https://arxiv.org/abs/2609.03716)

    该论文通过系统性文献编码揭示数据中心AI节能研究存在“控制与负载割裂”、过度依赖仿真验证、忽略水资源与隐含碳排放、各类方法节能效果无法区分排名等十大差距，并提出将控制策略与工作负载需求相耦合的CLEAR-DC研究框架。

    

    数据中心越来越多地由人工智能进行优化，同时也越来越多地由人工智能承载负载。现有文献将这两个问题视为互不相关的独立问题：控制研究将工作负载建模为外生到达过程，而可持续性研究则将基础设施建模为一个固定的乘数因子。我们通过一个有记录的检索流程筛选了大约194篇论文，对其中63篇进行了编码分析，并报告编码结果所揭示的现状。在28项以控制为导向的主要研究中，18项仅在仿真环境中得到验证，仅5项达到了物理硬件或生产设施层面；没有任何一项研究考虑了水资源消耗，也没有任何一项考虑了隐含碳排放。四个技术系列所报告的节能区间几乎完全重叠，这意味着该领域目前无法对自身的各类方法进行优劣排序。我们对十个反复出现的研究差距进行了后果严重性和可解决性的评分，并提出了CLEAR-DC框架，该框架将控制策略分支与工作负载需求分支通过（摘要原文在此处截断）

    arXiv:2609.03716v1 Announce Type: new  Abstract: Data centers are increasingly optimized by artificial intelligence and, at the same time, increasingly loaded by it. The literature treats these as two unrelated problems: control studies model workload as an exogenous arrival process, while sustainability studies model infrastructure as a fixed multiplier. We screen roughly 194 papers retrieved through a documented protocol, code 63 of them, and report what the coding shows. Of 28 primary control-oriented studies, 18 are validated in simulation alone and 5 reach physical hardware or a production facility; none account for water withdrawal, and none account for embodied carbon. Reported savings intervals across four technique families overlap almost completely, which means the field cannot presently rank its own methods. Ten recurring gaps are scored for consequence and tractability, and we set out CLEAR-DC, a framework coupling a control-policy branch to a workload-demand branch through
    
[^52]: 基于回归导向累积量的联邦因果发现

    Federated Causal Discovery via Regression-Directed Cumulants

    [https://arxiv.org/abs/2609.03705](https://arxiv.org/abs/2609.03705)

    提出一种基于回归导向累积量的联邦LiNGAM因果发现方法，利用高阶累积量张量在独立样本组间的精确可加性，在水平、垂直和混合数据划分下仅需一轮通信即可实现隐私保护的因果发现，并克服了现有方法FedISHC在近似对称情形下失效的问题。

    

    在本文中，我们研究了线性非高斯无环模型在联邦环境中的应用。这类因果模型能够超越马尔可夫等价类的限制。然而，在许多领域中数据是稀缺的，而由于GDPR等法规的限制，通过集中来自不同客户端的数据来增加样本量并不可取。联邦环境提供了一种在隐私保护与因果发现精度之间取得平衡的有吸引力的选择。遗憾的是，LiNGAM设定下的标准集中式估计器——即DirectLiNGAM——无法直接进行联邦化。高阶累积量张量提供了一条绕过这一障碍的途径：它们仅依赖于所涉及变量的联合分布，并且在相互独立的样本组之间可以精确相加，因此在水平、垂直和混合数据划分下，仅需一轮通信即可完成任务。然而，目前沿这一思路的联邦方法FedISHC在近似对称的情形下会失效……

    arXiv:2609.03705v1 Announce Type: new  Abstract: In this paper we study linear non-Gaussian acyclic models (LiNGAM) when used in federated environments. These causal models allow one to go beyond Markov equivalence. However, in many domains data are scarce, and increasing the sample size by centralising data from different clients is not advisable due to regulations such as the GDPR. The federated environment offers an attractive option to balance privacy and causal discovery accuracy. Unfortunately, the standard centralised estimator in the LiNGAM setting, i.e., DirectLiNGAM, cannot be straightforwardly federated. Higher-order cumulant tensors offer a way around this obstacle: they depend only on the joint distribution of the variables involved and add exactly across independent sample groups, so a single communication round suffices in horizontal, vertical, and hybrid partitions.   However, FedISHC, i.e., the current federated method along these lines, breaks down under near-symmetri
    
[^53]: 部分可识别性下的分辨率感知实验设计

    Resolution-Aware Experimental Design under Partial Identifiability

    [https://arxiv.org/abs/2609.03686](https://arxiv.org/abs/2609.03686)

    本文提出分辨率感知实验设计（RAED），在部分可识别性下通过最小化期望非空结构候选集来选择实验，证明了传统信息增益准则会因跨冗余混叠而失效，而RAED在复合Blackwell比较下能保持正确的实验排序。

    

    实验设计通常被框架化为选择预期提供最多信息的实验。然而，在部分可识别性条件下，持续的冗余（nuisance）不确定性可能使同一观测携带不同的结构含义。我们提出了分辨率感知实验设计（RAED），该方法在控制误排除的前提下，通过可实现的最小期望非空结构候选集来选择实验。我们证明了一个精确的跨冗余混叠分离结果：一个实验可能在结构信息增益、全潜变量信息增益、平均分类以及冗余边缘化信息量等准则上更受青睐，但其有效结构分辨率却可能任意更差。尽管如此，RAED在真正的复合Blackwell比较下仍能保持预期的实验排序。为了使这一准则可操作化，我们开发了一种基于学习的分数实现方法，具有有限样本冗余平均和正尾校准特性，并且……

    arXiv:2609.03686v1 Announce Type: new  Abstract: Experimental design is commonly framed as choosing the experiment expected to provide the most information. Under partial identifiability however, persistent nuisance uncertainty can make the same observation carry different structural meanings. We introduce Resolution-Aware Experimental Design (RAED), which selects an experiment by the smallest expected nonempty structural candidate set achievable subject to false-exclusion control. We prove an exact cross-nuisance aliasing separation: an experiment can be preferred by structural and full-latent information gain, average classification, and nuisance-marginalized informativeness while having arbitrarily poorer valid structural resolution. RAED nevertheless preserves the expected ordering under a genuine composite Blackwell comparison. To make this criterion operational, we develop a learned score-based implementation with finite-sample nuisance-average and positive-tail calibration, and 
    
[^54]: 通过用自然语言描述图像子集之间的差异来理解自动驾驶数据集

    Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language

    [https://arxiv.org/abs/2609.03677](https://arxiv.org/abs/2609.03677)

    本文提出集合差异描述方法，利用自然语言自动描述自动驾驶数据集中不同图像子集之间的差异，通过基于目标检测的对象中心分析实现对数据集组成和域偏移的可解释理解。

    

    理解大规模自动驾驶数据集的组成对于安全性、鲁棒性以及跨域的可靠运行至关重要。例如，不同地点之间的域偏移可能导致运行环境与训练数据不一致，从而造成潜在的危险性能下降。然而，现有的数据分析流程在很大程度上依赖于元数据、预定义标签或人工检查，这些方法提供的语义洞察有限或无法规模化。本文研究了集合差异描述任务：给定两个图像子集，目标是生成一个用自然语言描述目标集与参考集之间差异的假设。基于两阶段的框架，我们将该方法适配到自动驾驶领域，通过聚焦于从目标检测中提取的以对象为中心的图像块，这简化了聚合过程，并能够将差异归因于特定的对象实例或类别。

    arXiv:2609.03677v1 Announce Type: cross  Abstract: Understanding the composition of large-scale autonomous driving datasets is essential for safety, robustness, and reliable operation across domains. For example, domain shift between locations could lead to the operating environment being misaligned with the training data, resulting in potentially dangerous performance degradation. Yet, existing data analysis pipelines largely rely on metadata, predefined labels, or manual inspection, which provide limited semantic insight or do not scale. This paper studies set difference captioning: given two subsets of images, the goal is to produce a natural-language hypothesis describing differences between the target and reference set. Building on a two-stage formulation, we adapt the method to autonomous driving by focusing on object-centric patches derived from object detection, which simplifies aggregation and enables attribution of differences to specific object instances or categories. To ev
    
[^55]: 离线多智能体强化学习中基于序列模型的分布外泛化

    Out-of-Distribution Generalisation with Sequence Models in Offline Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.03667](https://arxiv.org/abs/2609.03667)

    该研究通过系统性分析发现，扩展任务多样性而非数据集规模是离线多智能体强化学习实现零样本任务泛化的关键因素，其提出的多任务序列建模方法在留出测试任务上相比单任务模型平均提升3.2倍。

    

    在离线多智能体强化学习（MARL）中，泛化到未见过的任务仍然是一个根本性挑战。在这项工作中，我们对离线设置下的零样本任务泛化进行了系统性分析，并针对任务多样性、数据集规模和网络容量之间的扩展行为展开了广泛的实证研究。为支持这项研究，我们扩展了离线序列建模架构，使其能够处理多任务的观测与动作空间，以及跨任务数量可变的智能体。我们的主要发现是：扩展任务多样性——而非单纯增加数据集规模——是实现稳健零样本迁移的主导因素。通过在四个具有挑战性的环境上进行大规模实验，我们证明了我们的多任务方法在留出测试任务上相比单任务模型实现了平均3.2倍的提升，并始终优于强大的基线方法。

    arXiv:2609.03667v1 Announce Type: cross  Abstract: Generalising to unseen tasks remains a fundamental challenge in offline multi-agent reinforcement learning (MARL). In this work, we present a principled analysis of zero-shot task generalisation in the offline setting and conduct an extensive empirical investigation into the scaling behaviour governing task diversity, dataset size, and network capacity. To facilitate this study, we extend offline sequence modelling architectures to handle multi-task observation and action spaces alongside variable agent counts across tasks. Our primary finding is that scaling task diversity---rather than sheer dataset size is the dominant factor in achieving robust zero-shot transfer. Through large-scale experiments across four challenging environments (Connector, RWARE, SMAX, and LBF), we demonstrate that our multi-task approach achieves a mean improvement of 3.2x on held-out test tasks compared to single-task models and consistently outperforms stron
    
[^56]: 从定向遗忘模型中提取被遗忘的提示词

    Extracting Forgotten Prompts from Targeted Unlearned Models

    [https://arxiv.org/abs/2609.03662](https://arxiv.org/abs/2609.03662)

    本文首次提出目标主动搜索攻击，证明攻击者无需事先知道遗忘内容，仅凭保留数据和黑盒访问，即可在有限查询预算下从定向遗忘模型中提取出被遗忘的提示词本身。

    

    近期的机器遗忘方法（如 NPO、DPO、LUNAR）利用拒绝对齐来抑制被遗忘的数据。然而，已有研究表明，拒绝响应可能会留下遗忘的痕迹，且近期的攻击已能成功恢复部分被遗忘的知识。在本文中，我们揭示了一种新的漏洞：现有攻击通常假设被遗忘的提示词已为攻击者所知，并专注于恢复其对应的答案；而我们证明，仅利用保留数据和模型的黑盒访问权限，被遗忘的提示词本身也可以被提取出来。我们的攻击方法——目标主动搜索，首先通过构建规范化模板和实体池来识别被遗忘的实体，并在有限的查询预算下，选择信息量最大的模板-实体组合对模型进行选择性查询；一旦实体被识别，TAS 便用这些实体实例化提示词模板，以探测遗忘模型的响应，从而还原被遗忘的提示词。

    arXiv:2609.03662v1 Announce Type: new  Abstract: Recent unlearning methods (e.g. NPO, DPO, LUNAR) make use of refusal alignment to suppress forgotten data. However, it has been shown that refusal responses might leave traces of unlearning, and recent attacks have been able to successfully recover some of the unlearned knowledge. In this paper, we uncover a new vulnerability. Existing attacks typically assume that the forgotten prompts are already known to the adversary and focus on recovering their answers. However, we show that the forgotten prompts themselves can be extracted by using the retained data and black-box access to the model. Our attack, Targeted Active Search (TAS), first identifies the forgotten entities by constructing canonical templates and entity pool, and selectively querying the model using the most informative template-entity pair under a limited query budget. Once the entities are identified, TAS instantiates prompt templates with those entities to probe the unle
    
[^57]: 局部更新，全局学习（LUGL）：与非增量学习器进行博弈

    Local Updates, Global Learning (LUGL): Playing Games with non-incremental Learners

    [https://arxiv.org/abs/2609.03660](https://arxiv.org/abs/2609.03660)

    提出LUGL框架，通过将数据收集与模型拟合解耦，使梯度提升树等非增量学习器能够克服分布偏移问题，成功应用于自我博弈强化学习场景。

    

    神经网络（NN）在强化学习（RL）中的主导地位部分归因于其增量学习能力，这种能力天然契合自我博弈训练的在线、非平稳特性。然而，以LightGBM为代表的梯度提升树被广泛认为是监督学习中处理表格数据的最先进方法，在准确性和效率上往往优于神经网络。博弈状态本质上就是表格形式的——离散的动作、类别的卡牌身份、结构化的棋盘位置——这使其成为基于树的方法的理想候选。我们提出了LUGL（局部更新，全局学习）框架，该框架将数据收集与模型拟合解耦，使梯度提升树（GBT）等非增量学习器能够在强化学习环境中运行，否则它们会因分布偏移而失败。LUGL在两个阶段之间交替进行：局部更新阶段，智能体进行自我博弈游戏并累积表格更新（Q值、V值、策略或遗憾值）……

    arXiv:2609.03660v1 Announce Type: cross  Abstract: The dominance of Neural Networks (NNs) in RL is partially due to their incremental learning capability, which naturally suits the online, non-stationary nature of self-play training. However, gradient-boosted trees like LightGBM are widely recognised as the state of the art for tabular data in supervised learning, often outperforming NNs in accuracy and efficiency. Game states are inherently tabular---discrete actions, categorical card identities, structured board positions---which makes them an ideal candidate for tree-based methods. We introduce LUGL (Local Updates, Global Learning), a framework that decouples data collection from model fitting, enabling non-incremental learners such as GBTs to operate in RL settings where they would otherwise fail due to distributional shift. LUGL alternates between a local updates phase, where the agent plays self-play games and accumulates tabular updates (Q-values, V-values, policies, or regret v
    
[^58]: 固定有限幺半群观测下的相对素因子分解与有限状态表示

    Relative Prime Factorization and Finite-State Presentations under Fixed Finite-Monoid Observation

    [https://arxiv.org/abs/2609.03643](https://arxiv.org/abs/2609.03643)

    本文通过穷举计算机验证的反例证明唯一素因子分解并不蕴含有限相对表示性质（FRP），进而引入更强的有限状态相对表示性质（FSRP）来隔离并刻画这一表示障碍。

    

    设 L ⊆ Σ*，并固定一个到有限幺半群的同态 h: Σ* → M。我们研究相对句法同余 θ_{L,h} := ≡_L ∩ ker h 中的精确因子分解与规范表示问题。我们将唯一因子分解与有限直表示区分开来。一个经过穷举计算机验证的36元商结构，其每一个活跃的非单位类都具有唯一的精确素因子分解，但其有效的素返回规则却包含一个无限族，因此唯一因子分解并不蕴含有限相对表示性质（FRP），即使对于有限商也是如此。我们将同样的缺陷提升到一个具有无限相对商和有限素谱的非正则上下文无关语言上。为隔离这一障碍，我们引入有限状态相对表示性质（FSRP），其中规范的有效右侧语言由有限剩余控制器表示，并证明 FRP ⊊ FSRP。

    arXiv:2609.03643v1 Announce Type: cross  Abstract: Let $L\subseteq\Sigma^*$ and fix a morphism $h:\Sigma^*\to M$ into a finite monoid. We study exact factorization and canonical presentation in the relative syntactic congruence $\theta_{L,h}:=\equiv_L\cap\ker h$.   We separate unique factorization from finite direct presentation. An exhaustively computer-checked $36$-element quotient has a unique exact prime factorization for every live non-unit class, yet its valid prime-return rules contain an infinite family, so unique factorization does not imply the finite relative presentation property (FRP), even for a finite quotient. We lift the same defect to a nonregular context-free language with an infinite relative quotient and finite prime spectrum.   To isolate the obstruction, we introduce the finite-state relative presentation property (FSRP), in which canonical valid right-hand-side languages are represented by finite residual controllers, and prove $\mathrm{FRP}\subsetneq\mathrm{FSR
    
[^59]: 残差神经网络克服半线性热方程的维数灾难

    Residual neural networks overcome the curse of dimensionality for semilinear heat equations

    [https://arxiv.org/abs/2609.03626](https://arxiv.org/abs/2609.03626)

    本文首次证明了残差神经网络在数值逼近半线性热方程时，能以多项式数量的参数克服维数灾难，其证明方法是将多层Picard估计量的确定性实现表示为残差神经网络。

    

    严格的理论结果已经表明，前馈神经网络可以在高维偏微分方程（PDE）的数值逼近中克服维数灾难，但关于残差神经网络在非线性PDE情形下的研究却相对较少。我们证明了残差神经网络在数值逼近具有全局Lipschitz连续且与梯度无关的非线性项的半线性热方程的解时能够克服维数灾难：在PDE数据满足多项式增长和网络可逼近性假设的条件下，存在 η∈(0,∞) 以及残差神经网络 Ψ_{d,ε}（d∈ℕ，ε∈(0,1]），其参数个数至多为 η d^η ε^{-η}，其实现在维数 d 下能够以不超过 ε 的 L² 误差逼近方程的解。证明的核心思想是用残差神经网络来表示多层Picard估计量的一个确定性实现。

    arXiv:2609.03626v1 Announce Type: cross  Abstract: Rigorous results show that feedforward neural networks can overcome the curse of dimensionality in the numerical approximation of high-dimensional partial differential equations (PDEs), but comparatively little is known about residual neural networks (ResNets) in the nonlinear PDE setting. We prove that ResNets overcome the curse of dimensionality in the numerical approximation of solutions of semilinear heat equations with globally Lipschitz continuous, gradient-independent nonlinearities: under polynomial growth and network approximability hypotheses on the PDE data, there exist $\eta\in(0,\infty)$ and ResNets $\Psi_{d,\varepsilon}$, $d\in\mathbb{N}$, $\varepsilon\in(0,1]$, with at most $\eta d^{\eta}\varepsilon^{-\eta}$ parameters whose realizations approximate the solution in dimension $d$ with an $L^2$-error of at most $\varepsilon$. The proof represents one deterministic realization of a multilevel Picard estimator by a ResNet wh
    
[^60]: 论模型压缩与测试时自适应之间的交互作用

    On the Interaction Between Model Compression and Test-Time Adaptation

    [https://arxiv.org/abs/2609.03604](https://arxiv.org/abs/2609.03604)

    本文首次系统研究了模型压缩与测试时自适应（TTA）之间的交互作用，发现压缩模型虽在有监督自适应下保持高精度，但其TTA性能随压缩程度增加而显著下降，其根源在于表示多样性的降低和限制可恢复性的结构约束。

    

    在真实环境中部署的深度神经网络必须兼具高效性与适应性，这需要模型压缩和测试时自适应（TTA）。虽然这两者都已被单独充分研究，但它们之间的交互作用仍鲜为人知。我们系统地分析了结构化压缩如何影响模型在分布偏移下的自适应能力。我们在CIFAR-10-C和ImageNet-C数据集上使用ResNet-18和ViT-Base模型，评估了多种压缩方法与标准TTA技术的组合。我们引入了一个诊断框架，用于检验表示表达能力和自适应子空间的兼容性。我们的结果揭示了一个持续的差距：尽管压缩模型在有监督自适应下保持了较高的准确率，但其TTA性能随着压缩程度的增加而显著下降。我们证明这源于表示多样性的降低以及限制可恢复性的结构约束。这些效应强烈依赖于压缩方法。

    arXiv:2609.03604v1 Announce Type: cross  Abstract: Deep neural networks deployed in the wild must be both efficient and adaptable, requiring model compression and test-time adaptation (TTA). While both are well studied in isolation, their interaction remains poorly understood. We systematically analyze how structured compression affects a model's ability to adapt under distribution shift. Using ResNet-18 and ViT-Base on CIFAR-10-C and ImageNet-C, we evaluate multiple compression methods combined with standard TTA techniques. We introduce a diagnostic framework that examines representational expressivity and adaptation subspace compatibility. Our results reveal a consistent gap: although compressed models retain high accuracy under supervised adaptation, their TTA performance degrades significantly with increasing compression. We show that this stems from reduced representational diversity and structural constraints that limit recoverability. These effects strongly depend on the compres
    
[^61]: 神经网络Maxent：一种具有可学习非线性的通用扩展方法，应用于沙漠蝗分布建模的时间序列分析

    Neural-Network Maxent: a general extension with learned nonlinearity, applied to time-series for Desert Locust distribution modelling

    [https://arxiv.org/abs/2609.03603](https://arxiv.org/abs/2609.03603)

    本文提出RNN Maxent，用神经网络（GRU）替换Maxent框架中的固定特征字典，从而以可学习的方式捕捉时间序列环境协变量中的非线性和序列结构，显著改进了沙漠蝗虫的物种分布建模。

    

    物种分布建模（SDM）对于理解环境条件如何塑造生物多样性至关重要，特别是对于沙漠蝗虫（Schistocerca gregaria）这类具有破坏性的害虫，其繁殖动态与快速变化的环境条件紧密耦合。Maxent已成为处理仅存在数据的主流方法，但其依赖于对手工选择特征变换的线性组合，这限制了其捕捉生态监测中常见的非线性时间关系的能力，因为在这些监测中，降水、土壤湿度和植被指数等协变量会随时间发生显著变化。标准实现将时间序列协变量展平为独立特征，从而丢弃了携带关键信号的序列结构。我们提出了RNN Maxent，这是Maxent框架的一个扩展，它用一个神经网络（具体为门控循环单元GRU）替换固定的特征字典……

    arXiv:2609.03603v1 Announce Type: new  Abstract: Species Distribution Modelling (SDM) is essential for understanding how environmental conditions shape biodiversity, particularly for destructive pests such as the Desert Locust (Schistocerca gregaria), whose breeding dynamics are tightly coupled to rapidly evolving environmental conditions. Maxent has become the dominant method for presence-only data, but its reliance on a linear combination of hand chosen feature transforms limits its ability to capture the nonlinear, temporal relationships common in ecological monitoring, where covariates such as precipitation, soil moisture, and vegetation indices evolve meaningfully over time. Standard implementations flatten time-series covariates into independent features, discarding sequential structure that carries critical signal. We introduce RNN Maxent, an extension of the Maxent framework that replaces the fixed feature dictionary with a neural network, specifically a Gated Recurrent Unit (G
    
[^62]: LevelSyn：基于层级异步图神经网络的物理感知逻辑综合

    LevelSyn: Physical-Aware Logic Synthesis via Level-Asynchronous Graph Neural Networks

    [https://arxiv.org/abs/2609.03594](https://arxiv.org/abs/2609.03594)

    LevelSyn提出了一种物理感知的逻辑综合框架，利用层级异步图神经网络捕捉与-非图（AIG）的结构和方向语义以预测高保真度门坐标，并结合线长驱动的优化引擎，从而缓解综合与物理设计脱节带来的PPA退化和设计收敛周期过长的问题。

    

    随着集成电路技术进入纳米尺度，逻辑综合与物理设计之间的传统脱节导致了显著的PPA（功耗、性能和面积）退化以及漫长的设计收敛周期。传统逻辑综合依赖于非物理的线负载模型（WLM），而近期基于谱方法的布局预测器往往忽略了网表中固有的层次化逻辑深度和信号流，导致空间估算的保真度较低。为了弥合这一差距，我们提出了LevelSyn，一个新颖的物理感知逻辑综合框架，它将层次化表示学习与线长驱动的优化引擎相结合。其核心在于，LevelSyn利用层级异步图神经网络（GNN），通过捕捉与-非图（AIG）的结构和方向语义，来预测高保真度的门坐标。为了处理工业规模的设计，一个层级对齐的…

    arXiv:2609.03594v1 Announce Type: cross  Abstract: As integrated circuit technology scales into the nanometer regime, the traditional disconnect between logic synthesis and physical design has led to significant PPA (Power, Performance, and Area) degradation and prolonged design closure cycles. Traditional logic synthesis relies on non-physical Wire Load Models (WLMs), while recent spectral-based placement predictors often neglect the inherent hierarchical logic depth and signal flow of netlists, which leads to low-fidelity spatial estimations. To bridge this gap, we propose LevelSyn, a novel physical-aware logic synthesis framework that integrates hierarchical representation learning with a wirelength-driven optimization engine. At its core, LevelSyn leverages a level-asynchronous Graph Neural Network (GNN) to predict high-fidelity gate coordinates by capturing the structural and directional semantics of And-Inverter Graphs (AIGs). To handle industrial-scale designs, a level-aligned s
    
[^63]: 深度残差网络的相关初始化

    Correlated initialization of deep residual networks

    [https://arxiv.org/abs/2609.03589](https://arxiv.org/abs/2609.03589)

    本文证明了具有跨层相关权重的深度残差网络在正则变化相关性条件下存在唯一的临界缩放，使得无穷深度极限由 Hermite 过程驱动的 Young 微分方程描述，从而证实并扩展了相关初始化在布朗随机微分方程与常微分方程之间连续插值的猜想。

    

    我们研究了在初始化时各层权重具有相关性的残差网络的大深度行为。我们的结果证实并扩展了 Marion 等人 [2025] 提出的一个猜想，即相关初始化应该在由独立初始化产生的布朗随机微分方程与由完全相关初始化产生的常微分方程之间连续插值。当初始化是通过对具有正则变化相关性的平稳高斯序列应用特征函数而获得时，我们证明存在唯一的临界缩放，使得无穷深度极限是由 Hermite 过程驱动的 Young 微分方程的解。当生成初始化的特征函数具有 Hermite 秩一时（例如恒等函数的情形），Hermite 过程可退化为分数布朗运动。

    arXiv:2609.03589v1 Announce Type: cross  Abstract: We study the large-depth behavior of residual networks whose weights are correlated across layers at initialization. Our results confirm and extend a conjecture of Marion et al. [2025], according to which correlated initializations should interpolate continuously between the Brownian stochastic differential equation arising from independent initialization and the ordinary differential equation arising from perfectly correlated initialization.   When the initialization is obtained from the application of a feature function to a stationary Gaussian sequence with regularly varying correlation, we prove that there exists a unique critical scaling such that the infinite-depth limit is the solution of a Young differential equation driven by a Hermite process. Hermite processes reduce to the fractional Brownian motion if the feature function generating the initialization has Hermite rank one, which is the case for the identity function, for e
    
[^64]: WeatherNext 3：利用原始观测数据提升全球天气模型的分辨率与性能

    WeatherNext 3: Increasing resolution and performance of global weather models with raw observations

    [https://arxiv.org/abs/2609.03582](https://arxiv.org/abs/2609.03582)

    WeatherNext 3通过直接引入原始观测数据（尤其是低延迟地球静止卫星数据），实现了每小时更新的高分辨率天气预报（0.1度、小时级），在概率性中期预报技能上确立了新的最先进水平。

    

    最先进的AI天气模型已展现出出色的中期预报技能和计算效率，但存在两个关键缺陷：其预报的空间和时间分辨率低于最好的物理模型，并且它们完全使用分析数据进行初始化和训练。因此，它们无法直接利用观测数据，且分析数据中的任何偏差都会被预报所继承。WeatherNext 3解决了这些缺陷，并在概率性中期预报技能方面确立了新的最先进水平。首先，WeatherNext 3通过引入低延迟的地球静止卫星数据，每小时生成新的预报（而非像传统全球模型那样每6小时一次）。其次，WeatherNext 3的时间和空间分辨率与基于物理的全球模型相当，具有小时级时间步长和0.1度分辨率的单层变量，包括太阳辐射和云……

    arXiv:2609.03582v1 Announce Type: new  Abstract: State-of-the-art AI weather models have shown impressive medium-range forecast skill and computational efficiency, but suffer two key shortcomings: their forecasts have lower spatial and temporal resolution than the best physics-based models and they are exclusively initialized with and trained on analysis data. As a result, they cannot directly make use of observations, and any biases in the analysis are inherited by the forecast. WeatherNext 3 addresses these shortcomings and establishes a new state-of-the-art for probabilistic medium-range forecasting skill. First, WeatherNext 3 generates new forecasts every hour (rather than every 6 hours like traditional global models) by ingesting low-latency geostationary satellite data. Second, WeatherNext 3's temporal and spatial resolution are on par with physics-based global models, with hourly time steps and 0.1 degree resolution for single-level variables, including solar radiation and cloud
    
[^65]: 迈向面向目标条件机器人规划的物理接地JEPA世界模型

    Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning

    [https://arxiv.org/abs/2609.03565](https://arxiv.org/abs/2609.03565)

    该论文提出一种融合逆动力学与状态对齐的端到端JEPA世界模型，将潜在表示扎根于物理构型和运动信息，从而显著提升目标条件机器人规划的成功率。

    

    动作条件化的JEPA世界模型能够在不重建未来像素的情况下朝向视觉指定的目标进行规划，然而仅靠潜在预测并不能显式地促使所学习的表示保留与机器人控制相关的信息。我们提出了一种端到端的JEPA世界模型，通过逆动力学（IDM）和状态对齐（SA）来增强潜在预测。逆动力学能够抑制潜在坍缩，并使潜在转变携带产生它们的动作信息；而状态对齐则将连续的表示扎根于其对应的物理构型和运动之中。在四个基准任务上，我们的模型在TwoRoom（100%）、PushT（98%）和OGBench-Cube（87%）上取得了最高的成功率，同时在Reacher任务上与LeWorldModel表现相当。我们的消融实验进一步表明，在所有四个任务中，相比仅使用逆动力学，添加状态对齐均能持续提升规划成功率。

    arXiv:2609.03565v1 Announce Type: cross  Abstract: Action-conditioned JEPA world models enable planning toward visually specified goals without reconstructing future pixels, yet latent prediction alone does not explicitly encourage the learned representations to retain information relevant to robotic control. We introduce an end-to-end JEPA world model that augments latent prediction with inverse dynamics (IDM) and state alignment (SA). While inverse dynamics discourages latent collapse and makes latent transitions informative of the actions that produced them, state alignment grounds consecutive representations in their associated physical configuration and motion. Across four benchmark tasks, our model attains the highest success rates on TwoRoom (100%), PushT (98%), and OGBench-Cube (87%), while performing comparably to LeWorldModel on Reacher. Our ablation further shows that adding state alignment consistently improves planning success over IDM alone across all four tasks. Although
    
[^66]: 耦合缩放：神经缩放定律的表征可及性框架

    Coupled Scaling: A Representational Accessibility Framework for Neural Scaling Laws

    [https://arxiv.org/abs/2609.03533](https://arxiv.org/abs/2609.03533)

    该论文提出“耦合缩放”框架，首次将神经缩放定律与架构-优化系统可达的表征几何联系起来，并证明有限预算下的损失残差指数被任务尾部率与覆盖率所确定的区间严格界定。

    

    现有理论从数据几何或特定的数据-模型谱中推导神经缩放规律，但当架构或优化改变了系统能够有效达到的表征时，在相同数据上训练的系统可能表现出不同的缩放行为。我们提出了耦合缩放（Coupled Scaling），这是一个以任务为条件的框架，其中有限预算下的缩放取决于任务结构与架构-优化系统可及几何之间的关系。在一个可求解的模态截断模型中，损失分解为架构支撑之外的目标能量和未被解析的支撑内尾部。对于任意优先级排序，残差介于最优N支撑尾部和超出最大已完成高价值前缀的尾部之间。如果累积尾部率和对数覆盖率为 $\gamma_{A,T}$ 和 $\rho_{A,O,T}$，则残差指数位于区间 $[\rho_{A,O,T}\gamma_{A,T},\gamma_{A,T}]$ 内。在有界的离前缀增益下，已完成的前缀具有率确定性（摘要此处被截断）。

    arXiv:2609.03533v1 Announce Type: new  Abstract: Existing theories derive neural scaling from data geometry or a specified data-model spectrum, but systems trained on the same data can scale differently when architecture or optimization changes the representations they can efficiently reach. We introduce Coupled Scaling, a task-conditioned framework in which finite-budget scaling depends on the relation between task structure and the geometry accessible to an architecture-optimization system. In a solvable mode-truncation model, loss separates into target energy outside architectural support and an unresolved supported tail. For an arbitrary priority order, the residual lies between the best-N supported tail and the tail beyond the largest completed high-value prefix. If the cumulative-tail and coverage log-rates are $\gamma_{A,T}$ and $\rho_{A,O,T}$, the residual exponent lies in $[\rho_{A,O,T}\gamma_{A,T},\gamma_{A,T}]$. Under bounded off-prefix gain, the completed prefix is rate-det
    
[^67]: LeanGRPO：消除扩散强化学习中的冗余重计算

    LeanGRPO: Eliminating Redundant Recomputation in Diffusion RL

    [https://arxiv.org/abs/2609.03528](https://arxiv.org/abs/2609.03528)

    LeanGRPO 通过重构数据并行布局并提出两种无需重计算的训练方案，使 rollout 阶段的计算图与激活值可在策略更新时直接复用，从而消除同策略扩散强化学习中数学上冗余的重计算。

    

    扩散强化学习（RL）最近在图像和视频生成模型的后训练中取得了显著成功。然而，包括 DanceGRPO 和 FlowGRPO 在内的大多数扩散强化学习方法，在 rollout 之后都会对选定的时间步进行开启梯度追踪的重计算。在 rollout 与更新使用相同后端的同策略（on-policy）训练中，这种重计算在数学上是冗余的。直观上，rollout 和策略更新步骤可以复用同一个前向传播骨干网络来避免冗余计算，但这样做会在 rollout 阶段带来较大的内存开销。为了解决这一问题，我们提出了 LeanGRPO，通过重构数据并行布局，并为轨迹-对数概率扩散强化学习引入两种无需重计算的训练方案：(1) LeanGRPO-Retain 在 rollout 期间开启梯度追踪，并在更新阶段直接复用所得的计算图和保存的激活值进行反向传播，无需任何重计算；(2) L（摘要在此处截断）

    arXiv:2609.03528v1 Announce Type: cross  Abstract: Diffusion reinforcement learning (RL) has recently achieved significant success in post-training image and video generative models. However, most diffusion RL methods, including DanceGRPO and FlowGRPO, recompute selected timesteps with gradient tracking after rollout. Under on-policy training with the same backend for rollout and update, this recomputation is mathematically redundant. Intuitively, the rollout and policy update steps can reuse the same feed-forward backbone to avoid redundant computation, but doing so can incur a large memory overhead during rollout. To address the issue, we present LeanGRPO by restructuring the data-parallel layout and introducing two recompute-free training schedules for trajectory-logprob diffusion RL: (1) LeanGRPO-Retain enables gradient tracking during rollout and directly reuses the resulting computation graphs and saved activations for backward during update, requiring no recomputation; and (2) L
    
[^68]: EPIC：面向语义ID扩散推荐的显式后验物品条件化

    EPIC: Explicit Posterior Item Conditioning for Semantic ID Diffusion Recommendation

    [https://arxiv.org/abs/2609.03522](https://arxiv.org/abs/2609.03522)

    提出EPIC方法，将显式的物品级后验竞争引入语义ID扩散去噪过程，通过个性化候选物品后验分布来指导未确定位置的词元预测，且无需修改冻结的预训练骨干网络。

    

    语义ID（SID）生成式推荐通过生成一串简短的离散词元来预测下一个物品。近期的掩码扩散方法通过双向上下文和灵活解码改进了这一过程，但推荐最终需要在完整的物品目录中进行选择。在每个去噪步骤中，一个不完整的SID可能对应多个可行的物品，而现有方法主要通过逐位置的词元预测来进行推理。我们提出了显式后验物品条件化（EPIC），将显式的物品级竞争引入SID去噪过程。EPIC利用当前生成上下文和用户的近期交互，在可行的候选物品上构建个性化的后验分布，然后将该分布投影回尚未确定的SID位置，以指导后续的词元决策。预训练的骨干网络保持冻结，且不需要额外的解码器前向传播。在四个Amazon数据集上的实验……

    arXiv:2609.03522v1 Announce Type: cross  Abstract: Semantic ID (SID) generative recommendation predicts the next item by generating a short tuple of discrete tokens. Recent masked-diffusion methods improve this process through bidirectional context and flexible decoding, yet recommendation ultimately requires selecting among complete catalog items. At each denoising step, a partial SID can correspond to multiple feasible items, while existing methods primarily reason through position-wise token predictions. We propose Explicit Posterior Item Conditioning (EPIC), which introduces explicit item-level competition into SID denoising. EPIC constructs a personalized posterior over feasible candidate items using the current generation context and the user's recent interactions, then projects this distribution back to unresolved SID positions to guide subsequent token decisions. The pretrained backbone remains frozen and requires no additional decoder forward pass. Experiments on four Amazon b
    
[^69]: LongCounsel-8：基于多轮次心理咨询对话的纵向抑郁症追踪基准套件

    LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues

    [https://arxiv.org/abs/2609.03507](https://arxiv.org/abs/2609.03507)

    提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。

    

    从多轮次心理咨询对话中追踪抑郁症，需要同时估计当前的症状严重程度以及其在不同会话之间的变化情况。然而，该任务的进展受限于缺乏具有标准化会话级抑郁标签的纵向咨询数据。现有资源通常要么提供没有抑郁标签的多轮次对话，要么仅提供单次会话的标注访谈。构建这样一个基准面临三大挑战：保持纵向的一致性与多样性，将症状进展建立在实证模式之上，以及在不暴露目标标签的前提下自然地表达受控的抑郁状态。为应对这些挑战，我们提出了LongCounsel-8，这是一个由三个独立生成的数据集组成的基准套件，共包含7,749条五轮次咨询轨迹，其建立在真实的来访者档案、抑郁轨迹、症状构成和咨询模式的基础之上。

    arXiv:2609.03507v1 Announce Type: cross  Abstract: Tracking depression from multi-session counseling dialogues requires estimating both current symptom severity and how it changes across sessions. Yet progress on this task is constrained by the scarcity of longitudinal counseling data with standardized session-level depression labels. Existing resources typically provide either multi-session conversations without depression labels or labeled interviews in a single session. Building such a benchmark poses three challenges: maintaining longitudinal consistency and diversity, grounding symptom progression in empirical patterns, and expressing controlled depression states naturally without exposing target labels. To address these challenges, we introduce LongCounsel-8, a benchmark suite of three independently generated datasets totaling 7,749 five-session counseling trajectories, grounded in real-world client profiles, depression trajectories, symptom compositions, and counseling patterns.
    
[^70]: 一种面向多变量物联网流量数据异常检测的对抗式零样本学习方法

    An Adversarial Zero-Shot Learning Approach for Anomaly Detection in Multivariate IoT Traffic Data

    [https://arxiv.org/abs/2609.03505](https://arxiv.org/abs/2609.03505)

    该论文提出了一种基于序列VAE架构、结合对抗学习与对比损失的零样本异常检测框架，通过编码器/解码器适配层和基于目的地的流量分割策略，在无需标注数据的情况下实现了多变量物联网流量异常检测及跨领域自适应。

    

    物联网（IoT）网络中的异常检测由于设备多样性、标注数据缺乏以及跨环境的领域差异性而面临独特的挑战。本文提出了一种用于多变量时间序列异常检测的新型框架，该框架在基于序列的变分自编码器（VAE）架构中融合了对抗学习和对比损失。我们的方法通过联合优化领域不变的潜在表示和语义结构化的嵌入空间，实现了零样本领域自适应，且无需标注数据或原始特征迁移。为解决物联网部署的异构性问题，我们引入了编码器和解码器适配层，在保持上下文语义的同时对齐跨领域的特征分布。此外，我们提出了一种基于目的地的分割策略，以更好地建模物联网流量中真实世界的通信结构。

    arXiv:2609.03505v1 Announce Type: new  Abstract: Anomaly detection in Internet of Things (IoT) networks presents unique challenges due to the diversity of devices, lack of labeled data, and domain variability across environments. In this paper, we propose a novel framework for multivariate time-series anomaly detection that leverages adversarial learning and contrastive loss within a sequence-based Variational Autoencoder (VAE) architecture. Our method enables zero-shot domain adaptation by jointly optimizing domain-invariant latent representations and semantically structured embedding spaces, without requiring labeled data or raw feature transfer. To address the heterogeneity of IoT deployments, we introduce encoder and decoder adaptor layers that align feature distributions across domains while preserving contextual semantics. Additionally, we propose a destination-based segmentation strategy to better model real-world communication structures in IoT traffic. Our framework is compreh
    
[^71]: 超越高斯宽度的受限特征值：重尾分布下的阈值占据现象

    Restricted Eigenvalues Beyond Gaussian Width: Threshold Occupancy under Heavy Tails

    [https://arxiv.org/abs/2609.03504](https://arxiv.org/abs/2609.03504)

    本文对COLT 2015开放问题给出否定回答，证明仅凭一致小球条件不足以将次高斯测量的 $1+w(A)^2$ 样本量规律推广到重尾设计，其根本障碍是“同时阈值占据”现象。

    

    arXiv:2609.03504v1 公告类型：新论文 摘要：受限特征值（RE）界决定了范数正则化估计量能否实现稳定恢复。对于各向同性的次高斯测量，基准样本量为 $1+w(A)^2$，其中 $w(A)$ 是归一化下降锥的高斯宽度。COLT 2015 开放问题简报（Banerjee 等，2015）提出疑问：仅凭一致小球条件，同样的样本量规律是否适用于重尾设计。我们对其中提出的一般性问题给出了明确且系统性的否定回答：该规律在其完整的无量纲、任意集合形式下不成立，而缺失的关键障碍是“同时阈值占据”现象。一个具有固定小球常数的恒定宽度多面体下降锥，在环境维度一半以内的每个样本路径上经验受限特征值均为零。更一般地，每个有限的值域空间都能在任意窄的球冠内实现精确的阈值编码，并可提升为完整的多面体下降锥截面。

    arXiv:2609.03504v1 Announce Type: new  Abstract: Restricted eigenvalue (RE) bounds govern stable recovery by norm-regularized estimators. For isotropic sub-Gaussian measurements, the benchmark sample size is $1+w(A)^2$, where $w(A)$ is the Gaussian width of the normalized descent cone. The COLT 2015 open-problem note (Banerjee et al., 2015) asked whether the same law follows for heavy-tailed designs from a uniform small-ball condition alone. We give an explicit and systematic negative answer to the general question as formulated there: the proposed law fails in its full dimension-free, arbitrary-set form, and the missing obstruction is simultaneous threshold occupancy. A constant-width polyhedral descent cone with fixed small-ball constants has zero empirical RE on every sample path up to half the ambient dimension. More generally, every finite range space admits exact threshold encoding in an arbitrarily narrow spherical cap and a lift to a full polyhedral descent-cone section. For ev
    
[^72]: 迈向对混合专家模型的统计学理解

    Towards a Statistical Understanding of Mixture-of-Experts

    [https://arxiv.org/abs/2609.03501](https://arxiv.org/abs/2609.03501)

    本文从统计学理论视角理解混合专家模型，将其视为局部化聚合，推导出分解近似误差、专家学习误差与路由器估计误差的预言机风险界，并揭示稀疏Top-K路由如何在控制计算成本的同时保留局部化聚合的优势。

    

    混合专家模型架构通过基于输入的路由机制将一组专家预测器组合起来，从而提升模型容量，同时通常仅针对每个输入激活一小部分专家。尽管MoE在现代大规模模型中的重要性日益增长，但其设计选择——尤其是路由、稀疏激活和共享专家——在统计上所扮演的角色仍只被部分理解，因为现有理论大多局限于参数化或正确设定的MoE模型。在本文中，我们将MoE视为一种局部化聚合的形式，并展示这种局部化如何重塑近似-估计-计算之间的权衡。我们针对具有演化专家的稠密路由和稀疏路由学习推导了预言机风险界，将近似误差、专家学习误差和路由器估计误差分离开来，并刻画了稀疏Top-K路由如何在控制每输入计算量的同时保留局部化聚合的优势。

    arXiv:2609.03501v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) architectures increase model capacity by combining a collection of expert predictors through input-dependent routing, while often activating only a small subset of experts for each input. Despite their growing importance in modern large-scale models, the statistical roles of their design choices, especially routing, sparse activation, and shared experts, remain only partially understood, as existing theory has largely focused on parametric or correctly specified MoE models. In this paper, we view MoE as a form of localized aggregation and show how this localization reshapes the approximation-estimation-computation tradeoff. We derive oracle risk bounds for learning dense and sparse routing with evolving experts, separating approximation, expert-learning, and router-estimation errors, and characterize how sparse Top-K routing can retain the benefits of localized aggregation while controlling per-input computatio
    
[^73]: 自编码器参数的谱特征作为数据的向量表示

    Spectral characteristics of autoencoder parameters as a vector representation of data

    [https://arxiv.org/abs/2609.03495](https://arxiv.org/abs/2609.03495)

    提出将自编码器参数矩阵的谱特征（奇异值）作为数据的稠密向量表示，并从理论上证明这些奇异值与训练数据协方差矩阵的特征值相关联，从而在数据空间与参数空间之间建立信息传递。

    

    本文研究了自编码器模型的参数与其所训练数据的统计特性之间的关系。自编码器被定义为具有编码器-解码器架构的模型，通过压缩的潜在表示训练以重构输入数据。本文提出可以将模型参数视为对应样本的稠密向量表示。为验证这一假设，进行了一项理论与实验研究，基于自编码器参数矩阵的谱特征构建向量表示。理论分析表明，模型参数矩阵的奇异值与训练数据协方差矩阵的特征值相关，从而确保了数据空间与参数空间之间的信息传递。在CIFAR-10和FashionMNIST数据集上的实验结果证实了所得到的（表示的有效性）。

    arXiv:2609.03495v1 Announce Type: new  Abstract: This paper examines the relationship between the parameters of autoencoder models and the statistical properties of the data on which they are trained. Autoencoders are defined as models with an encoder-decoder architecture, trained to reconstruct input data through a compressed latent representation. It is proposed that the model parameters can be viewed as a dense vector representation of the corresponding sample. To test this hypothesis, a theoretical and experimental study is conducted in which a vector representation is formed based on the spectral characteristics of the autoencoder parameter matrices. Theoretical analysis shows that the singular values of the model parameter matrices are related to the eigenvalues of the covariance matrix of the training data, ensuring the transfer of information between the data space and the parameter space. Experimental results on the CIFAR-10 and FashionMNIST datasets confirm that the resulting
    
[^74]: 丹麦树种制图：光谱-时间特征与地理空间基础模型嵌入的比较

    Tree species mapping in Denmark: A comparison of spectral-temporal features with geospatial foundation model embeddings

    [https://arxiv.org/abs/2609.03480](https://arxiv.org/abs/2609.03480)

    本研究利用丹麦国家森林清查数据与Sentinel卫星观测，系统比较了人工构建的光谱-时间特征与地球观测基础模型（TESSERA和AlphaEarth）嵌入在树种分类中的表现，发现基于光谱-时间特征的多层感知机取得最佳性能（纯林宏观F1达0.843），而基础模型嵌入也展现出有竞争力的结果，为大规模森林树种制图的方法选择提供了重要参考。

    

    我们利用国家森林资源清查样地和地球观测（EO）数据对丹麦全国的树种进行制图，同时评估了基础模型在大尺度森林表征方面的潜力。我们比较了两种用于树种分类的备选输入表示方式：(i) 基于多时相Sentinel-1和Sentinel-2观测数据人工构建的光谱-时间特征（STF），以及 (ii) 由地球观测基础模型TESSERA和AlphaEarth生成的嵌入表示。两种表示方法均辅以冠层高度信息。我们针对所有输入表示评估了随机森林、XGBoost和多层感知机（MLP）分类器，并对纯林和混交林分别进行了评估。基于STF的MLP取得了最高的分类性能，在纯林和混交林上的宏观F1分数分别为0.843和0.653。在TESSERA嵌入上训练的MLP在纯林分类上也表现出具有竞争力的性能。

    arXiv:2609.03480v1 Announce Type: cross  Abstract: We map tree species across Denmark using National Forest Inventory plots and EO data, while evaluating the potential of foundation models for large-scale forest characterization. We compare two alternative input representations for tree species classification: (i) manually engineered spectral-temporal features (STF) derived from multi-temporal Sentinel-1 and Sentinel-2 observations, and (ii) embeddings generated by the EO FMs TESSERA and AlphaEarth. Both representations are complemented with canopy height information. Random forest, XGBoost, and Multi-Layer Perceptron (MLP) classifiers are evaluated for all input representations, with separate assessments for pure and mixed forest stands. The STF-based MLP achieves the highest classification performance, yielding macro F1 scores of 0.843 and 0.653 for pure and mixed stands, respectively. The MLP trained on TESSERA embeddings delivers competitive performance for pure stands, achieving r
    
[^75]: 注意差距：个人身份信息（PII）检测系统的鲁棒性风险

    Mind the Gap: Robustness Risks in PII Detection Systems

    [https://arxiv.org/abs/2609.03464](https://arxiv.org/abs/2609.03464)

    本研究构建了涵盖七类自然分布偏移的压力测试基准，发现SpaCy、Presidio和Qwen2.5-3B三类主流PII检测系统在真实非正式输入下均出现显著性能退化，表明标准基准评估掩盖了实际部署中的隐私安全风险。

    

    个人身份信息（PII）检测是数据保护基础设施的基础组成部分，漏检的实体会直接构成隐私和安全风险。尽管现代PII系统在标准基准测试中报告了强大的性能，我们证明这些评估掩盖了在真实部署中遇到分布偏移时的重大鲁棒性缺陷。我们没有比较最先进的准确率，而是研究不同的PII检测范式在嘈杂、非结构化和非正式输入下的失效方式。我们构建了一个涵盖七类自然分布偏移的压力测试基准，并评估了来自三个广泛部署的架构家族的代表性系统：基于编码器的NER（SpaCy）、基于规则的混合检测以及生成式LLM抽取（Qwen2.5-3B）。这三个系统在分布外输入上均表现出显著的性能退化，但各自的失效模式独特且互补。

    arXiv:2609.03464v1 Announce Type: new  Abstract: Personally Identifiable Information (PII) detection is a foundational component of data protection infrastructure where missed entities constitute direct privacy and security risks. Although modern PII systems report strong performance on standard benchmarks, we show that these evaluations mask substantial robustness failures under realistic distribution shifts encountered in deployment. Rather than comparing state-of-the-art accuracy, we study how different PII detection paradigms fail under noisy, unstructured, and informal inputs. We construct a stress test benchmark spanning seven categories of natural distribution shift and evaluate representative systems from three widely deployed architectural families: encoder-based NER (SpaCy), rule-based hybrid detection (Presidio), and generative LLM extraction (Qwen2.5-3B).   All three exhibit significant degradation on out-of-distribution inputs, but with distinct and complementary failure m
    
[^76]: 一种用于私有云CPU工作负载预测的两阶段预测系统

    A Two-Stage Forecasting System for CPU Workload Prediction in Private Clouds

    [https://arxiv.org/abs/2609.03457](https://arxiv.org/abs/2609.03457)

    该论文提出了一种两阶段预测系统，先预测客户服务请求（TPS），再由其估计未来CPU工作负载，采用级联XGBoost架构并结合扩展窗口的自适应在线再训练来应对云工作负载中的概念漂移。

    

    准确的云资源预测对于动态云环境中的主动资源供给、维护服务质量（QoS）以及降低运营成本至关重要。现有的预测方法主要是直接从历史资源轨迹估计未来的CPU工作负载，这往往忽略了客户服务需求与后续资源消耗之间的关系。本研究提出了一种两阶段集成预测模型，通过首先预测客户服务请求（以每秒事务数TPS表示），随后基于TPS预测结果估计未来的CPU工作负载，从而显式地建模了这种依赖关系。预测组件和资源预测组件均在级联学习架构中采用XGBoost模型，并辅以使用扩展窗口策略的自适应在线再训练机制，以应对持续演变的云工作负载中的概念漂移问题。

    arXiv:2609.03457v1 Announce Type: new  Abstract: Accurate cloud resource forecasting is essential for proactive resource provisioning, maintaining Quality of Service (QoS), and reducing operational costs in dynamic cloud environments. The existing forecasting approaches predominantly estimate future CPU workload directly from historical resource traces, which often overlook the relationship between customer service demand and subsequent resource consumption. This study proposes a two-stage integrated forecasting model that explicitly models this dependency by first forecasting customer service requests, expressed as Transactions Per Second (TPS), and subsequently estimating future CPU workload from the TPS forecast. Both the forecasting component and resource prediction component employed the XGBoost model within a cascaded learning architecture, complemented by adaptive online retraining using an expanding-window strategy to address concept drift in continuously evolving cloud workloa
    
[^77]: 超越直线性：通过分位数对齐树耦合实现无交叉流匹配

    Beyond Straightness: Non-Crossing Flow Matching via Quantile AlignTree Coupling

    [https://arxiv.org/abs/2609.03443](https://arxiv.org/abs/2609.03443)

    提出了一种基于分位数对齐树结构的高效耦合方法QAT-FM，以近线性时间构建无交叉的流匹配插值路径，并支持大规模高维生成任务的可扩展训练。

    

    流匹配的性能在很大程度上取决于源分布与目标分布之间耦合的质量。然而，独立耦合常常导致路径交叉和局部速度模糊，而基于最优传输（OT）的耦合通常需要高昂的构建成本。为应对这一挑战，我们提出了分位数对齐树流匹配，这是一种高效的结构化耦合策略，通过分位数对齐树结构在高斯先验与目标数据分布之间构建层次化耦合。QAT-FM以 O(Nd log N) 的时间复杂度构建耦合，并支持以 O(d) 复杂度进行逐对源采样，从而为大规模高维生成任务提供可扩展的训练能力。理论上，我们证明了QAT耦合满足边缘一致性、诱导无交叉的线性插值路径，并在中间时刻持续改善路径分离度……

    arXiv:2609.03443v1 Announce Type: new  Abstract: The performance of Flow Matching largely depends on the quality of the coupling between the source and target distributions. However, independent coupling often leads to path crossings and local velocity ambiguity, while OT-based couplings typically incur high construction costs. To address this challenge, we propose Quantile AlignTree Flow Matching (QAT-FM), an efficient structured coupling strategy that constructs a hierarchical coupling between a Gaussian prior and the target data distribution via a quantile-aligned tree structure. QAT-FM constructs the coupling in $\mathcal{O}(Nd\log N)$ time and supports per-pair source sampling with $\mathcal{O}(d)$ complexity, enabling scalable training for large-scale high-dimensional generative tasks. Theoretically, we prove that the QAT coupling satisfies marginal consistency, induces non-crossing linear interpolation paths, and consistently improves path separation at intermediate times compar
    
[^78]: 引导而非约束：可撤销先验为何在增广拉格朗日因果发现中失效

    Guide, Not Bind: Why Defeasible Priors Fail in Augmented Lagrangian Causal Discovery

    [https://arxiv.org/abs/2609.03442](https://arxiv.org/abs/2609.03442)

    本文揭示了可微因果发现中“引导而非约束”设计（用增广拉格朗日惩罚强制执行专家先验、并期望数据自适应松弛机制推翻错误规则）会因两个独立原因失效——顺序惩罚递增会在反事实检验检测到之前就抑制被错误禁止的真实边，且即使修复这两个问题也只能部分恢复该设计的功能。

    

    可微因果发现方法日益将专家先验编码为由增广拉格朗日（ALM）惩罚强制执行的禁用边约束，其前提假设是：数据自适应的松弛机制会对数据持续反驳的规则进行折减并最终将其推翻。我们证明，这种我们称之为“引导而非约束”的设计会因两个独立且被精确刻画的原因而失效，并且直接修复这两个原因也只能部分恢复其功能。首先，顺序惩罚递增的ALM会在任何反事实检验能够检测到之前就抑制被错误禁止的真实边：我们给出了任何自适应松弛机制为避免这种情况必须满足的三个必要条件（命题1），证明了DADU——本文引入的作为研究对象的自然松弛规则——违反了全部三个条件（推论1），并通过跨越节点数从4到……的图的3,072次训练运行确认了这一失败。（注：原文摘要在此处被截断）

    arXiv:2609.03442v1 Announce Type: new  Abstract: Differentiable causal discovery methods increasingly encode expert priors as forbidden-edge constraints enforced by an Augmented Lagrangian (ALM) penalty, on the assumption that a data-adaptive relaxation mechanism will discount and eventually override a rule the data consistently contradicts. We show this design, which we call \emph{guide, not bind}, fails for two independent, precisely characterized reasons, and that directly repairing both restores it only partially. First, sequential penalty-ramping ALM suppresses a wrongly-forbidden true edge before any counterfactual check can detect it: we give three necessary conditions any adaptive relaxation must satisfy to avoid this (Proposition~\ref{prop:conditions}), prove that DADU---the natural relaxation rule this paper introduces as the object of study---violates all three (Corollary~\ref{cor:dadu_failure}), and confirm the failure across 3{,}072 training runs spanning graphs from 4 to 
    
[^79]: 问题本身，而非路径：大语言模型推理轨迹中的预算与难度混淆因素

    It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories

    [https://arxiv.org/abs/2609.03436](https://arxiv.org/abs/2609.03436)

    该研究提出重启控制的截断探针方法，证明大语言模型推理轨迹中所谓的“突破时刻”和“早期注定失败”大多是预算与难度造成的混淆——178个问题-模型组合中仅1个真正存在前缀特有价值，且在同等token预算下延续自身推理前缀几乎总是优于从头重启。

    

    大语言模型的推理轨迹通常被解读为包含“突破”时刻和早期可判定的命运。这两种解读都建立在缺少主张层面反事实控制的测量之上；我们提供了这两种控制。首先，一个重启控制的截断探针将“解法适合延续预算”与“前缀携带全新计算无法换取的价值”区分开来，在匹配的总生成token预算下，比较每个锚点的延续求解率与从头重启曲线。将该探针应用于178个问题-模型单元格（89个MATH问题 × 两个小型开源模型，这是一个结果盲但针对难度的队列），178个单元格中恰好只有1个作为前缀受限而存活；重启的剂量-反应关系能够区分计算饥饿型模型与能力受限型模型；并且只要匹配预算位于重启网格之内，延续模型自身的前缀总是优于从头重启（9/9）——主要是计算压缩……（摘要原文在此处截断）

    arXiv:2609.03436v1 Announce Type: cross  Abstract: Reasoning traces of large language models are widely read as containing "breakthrough" moments and early-legible fates. Both readings rest on measurements missing a counterfactual control at the level of the claim; we supply both controls. First, a restart-controlled truncation probe separates when a solution fits the continuation budget from when a prefix carries value that fresh computation cannot buy, comparing per-anchor continuation solve rates against from-scratch restart curves at matched total generated-token budget. Applied to 178 problem-model cells (89 MATH problems x two small open models, an outcome-blind but difficulty-targeted cohort), exactly 1 of 178 cells survives as prefix-limited; restart dose-response separates a compute-starved model from a capability-limited one; and wherever the matched budget lies inside the restart grid, continuing the model's own prefix beats restarting (9 of 9) -- predominantly compute compr
    
[^80]: TraveL：基于Transformer的多视角路径分布式表示学习

    TraveL: Transformer-based Multi-view Path Distributional Representation Learning

    [https://arxiv.org/abs/2609.03427](https://arxiv.org/abs/2609.03427)

    该论文提出了基于Transformer的多视角分布式表示学习框架TraveL，通过捕捉旅行者行为的多样性和路段的区域相关性，将路径与出行开始时间编码为分布式表示，从而能够解码路径上旅行者行为的可能样本。

    

    道路网络的路径表示学习（PRL）因各类与路径相关的应用而受到越来越多的研究关注。现有的PRL工作通常利用路段与路径之间的共现关系来学习一个向量作为路径表示，而没有探索旅行者行为的多样性以及路径上的区域相关性。在这项工作中，我们提出学习分布式表示，通过捕捉旅行者行为的多样性以及路段所在区域内的各种依赖关系，为路径相关应用提供有价值的信息。我们提出了一种新颖的基于Transformer的多视角分布式表示学习框架，将路径与出行开始时间一起编码为分布式表示，该表示可用于解码路径上旅行者行为的可能样本。此外，通过分析区域相关性……

    arXiv:2609.03427v1 Announce Type: cross  Abstract: Path representation learning (PRL) for road networks has received increasing research attention, due to various path-related applications. Existing works on PRL typically exploit the co-occurrence relationship among road segments and paths to learn a vector as the path representation, without exploring the varied traveler behaviors and the regional correlation on the path. In this work, we propose to learn distributional representations, which provide valuable information for use in path-related applications, by capturing the varied traveler behaviors as well as the various dependencies within regions of road segments. We propose a novel Transformer-based Multi-view Distributional Representation Learning (TraveL) framework to encode a path along with a travel starting time to a distributional representation, which can be used to decode possible samples of on-path traveler behavior. Moreover, by analyzing the regional correlation which 
    
[^81]: 推断的生成过程多样性可预测语言模型间的相关失效

    Inferred Generative-Process Diversity Predicts Correlated Failure Across Language Models

    [https://arxiv.org/abs/2609.03422](https://arxiv.org/abs/2609.03422)

    该论文提出“生成过程多样性”这一比语义多样性更根本的模型多样性概念，并基于算法信息论用经置换控制的归一化压缩距离来度量语言模型间的推断生成过程多样性，从而预测语言模型间的相关失效。

    

    多样性是集体系统韧性运作中一个被广泛观察到的因素，然而真正重要的多样性类型取决于系统的性质和失效模式。这一区分对于由多个语言模型组成的系统尤为重要：即使不同模型的行为和失效仍然高度相关，它们也可能被视为独立的组件。使用语义相似度对语言模型群体进行的评估显示出有限的语义多样性，但这仅捕捉了所观察到的输出在语义上的差异。我们认为，模型多样性一个更根本的概念是生成过程多样性，即能够生成所观察到的输出的各过程之间的差异。借鉴算法信息论，我们使用原始模型输出之间的归一化压缩距离（并以置换控制进行残差化处理）作为推断生成过程多样性的度量。

    arXiv:2609.03422v1 Announce Type: new  Abstract: Diversity is a widely observed factor in the resilient function of collective systems, yet the type of diversity that matters depends on the properties and failure modes of the system. This distinction is important for systems composed of multiple language models. Different models may be treated as independent components even when their behaviour and failures remain strongly correlated. Assessments of language-model populations using semantic similarity demonstrate limited semantic diversity, but this captures only differences in the meaning of observed outputs. We argue that a more fundamental notion of model diversity is generative-process diversity, the differences between processes capable of generating the observed outputs. Drawing from Algorithmic Information Theory, we use Normalised Compression Distance between raw model outputs, residualised against a permutation control, as a measure of inferred generative-process diversity. Ac
    
[^82]: 联邦入侵检测中的隐私、鲁棒性与公平性权衡：聚合接口处的几何不可区分性

    Privacy, Robustness, and Fairness Trade-offs in Federated Intrusion Detection: Geometric Indistinguishability at the Aggregation Interface

    [https://arxiv.org/abs/2609.03420](https://arxiv.org/abs/2609.03420)

    本文揭示了联邦入侵检测中差分隐私、拜占庭鲁棒性与类别公平三大需求并非可独立组合，并提出“几何不可区分性”概念，用以解释隐私噪声导致的客户端更新分散会削弱鲁棒聚合对少数类攻击信号的保留能力。

    

    联邦学习使网络入侵检测能够在无需集中敏感流量数据的情况下进行注重隐私的协作，然而其在实际运行环境中的部署必须同时满足三个相互竞争的需求：形式化的差分隐私保证、对拜占庭对抗性参与者的容忍，以及对严重不平衡攻击类别的可靠检测覆盖。现有文献将这些属性视为可独立组合的，而本文在理论和实证两方面对这一假设提出了挑战。本文研究了这些需求在类别不平衡的联邦网络入侵检测系统（NIDS）中如何相互作用，并引入“几何不可区分性”作为概念视角，用以刻画一种情形：隐私引起的客户端更新分散会使得少数类信号更难被鲁棒聚合机制所保留。以UNSW-NB15数据集为案例研究，我们评估了DP-SGD与坐标级中值聚合相结合的方法……（摘要内容在此处截断）

    arXiv:2609.03420v1 Announce Type: cross  Abstract: Federated learning enables privacy-conscious collaboration for network intrusion detection without centralizing sensitive traffic data, yet its deployment in operational environments must simultaneously satisfy three competing requirements: formal differential privacy guaranties, tolerance to Byzantine-adversarial participants, and reliable detection coverage across severely imbalanced attack categories. Existing literature treats these properties as independently composable, an assumption that this paper challenges both theoretically and empirically. In this paper, we study how these requirements interact in class-imbalanced federated NIDS and introduce geometric indistinguishability as a conceptual lens for a regime in which privacy-induced dispersion in client updates can make minority-class signals harder for robust aggregation to preserve. Using UNSW-NB15 as a case study, we evaluate DP-SGD combined with coordinate-wise median und
    
[^83]: Dude：一种用于论文-代码差异检测的双检测多智能体系统

    Dude: A Dual-Detection Multi-Agent System for Paper-Code Discrepancy Detection

    [https://arxiv.org/abs/2609.03416](https://arxiv.org/abs/2609.03416)

    提出了首个用于论文-代码差异检测的双检测多智能体系统Dude，通过粒度对齐协商机制和两阶段显著性过滤机制，有效解决了论文语言与代码语言粒度不对称导致的误报问题。

    

    随着研究论文提交数量的增长超过了人工审阅能力，基于大语言模型（LLM）的论文-代码差异检测日益受到关注。然而，现有单智能体LLM范式的上下文容量有限且差异检测视角单一，导致差异检测的召回率表现不佳。本文提出了Dude，首个用于论文-代码差异检测的双检测多智能体系统。我们发现，论文语言与代码语言之间的粒度不对称性给差异检测的多智能体系统设计带来了过度解读和过度报告的挑战，导致误报增多。为解决这一问题，我们在Dude中提出了粒度对齐协商机制和两阶段显著性过滤机制，有效防止智能体错误报告差异。在真实世界论文-代码差异数据集上的实验结果表……

    arXiv:2609.03416v1 Announce Type: new  Abstract: LLM-empowered paper-code discrepancy detection has received growing concern since the scaling of research submissions exceeds the manual review capability. However, the limited context capacity and one-sided discrepancy detection of existing single-agent LLM paradigms lead to an inferior recall performance in detecting discrepancies. In this paper, we propose Dude, the first Dual-Detection Multi-Agent System for paper-code discrepancy detection. We discover that the granularity asymmetry of the paper-language and code-language introduces over-interpretation and over-reporting challenges in a multi-agent system design for discrepancy detection, resulting in increasing false positives. To address this, we propose a granularity-aligned negotiation and a two-stage salience-filtering mechanism in Dude, which effectively prevents agents from falsely reporting discrepancies. Experimental results in real-world paper-code discrepancy datasets sho
    
[^84]: 多维随机特征方法的谱收敛性

    Spectral Convergence of Random Feature Method in Multiple Dimensions

    [https://arxiv.org/abs/2609.03401](https://arxiv.org/abs/2609.03401)

    本文证明了随机特征方法在多维情形下对Sobolev、Gevrey、超解析及带限函数类目标的谱收敛性，并给出了收敛速率随目标正则性从超指数到代数的刻画，同时建立了强形式和弱形式RFM离散化的抽象误差估计。

    

    我们首先证明了随机特征方法（RFM）对Sobolev、Gevrey、超解析及带限函数类中多维目标的谱收敛性。该分析在由核积分算子生成的插值尺度上建立了一般的高概率逼近估计。在仅由采样特征确定的单个事件上，一个随机空间即可逼近给定源球中的每一个目标函数；此外，对于每个目标，单一系数向量所定义的逼近器可在所有允许的误差范数下同时达到谱精度。对于正则性自适应的频率分布以及增长频率窗口上的均匀分布，所得收敛速率介于超指数速率与代数速率之间，具体取决于目标函数的正则性。其次，我们为强形式和弱形式的RFM离散化建立了抽象误差估计，从而将前述逼近界转换为……（摘要在此处截断）

    arXiv:2609.03401v1 Announce Type: cross  Abstract: We first prove spectral convergence of the random feature method (RFM) for multidimensional targets in Sobolev, Gevrey, ultra-analytic, and bandlimited classes. The analysis establishes general high-probability approximation estimates in the interpolation scale generated by a kernel integral operator. On a single event determined only by the sampled features, one random space approximates every target in a prescribed source ball; moreover, for each target, a single coefficient vector defines an approximant that attains spectral accuracy simultaneously in all admissible error norms. For both regularity-adapted frequency distributions and uniform distributions on growing frequency windows, the resulting rates range from super-exponential to algebraic, depending on the regularity of the target. Second, we establish abstract error estimates for strong- and weak-form RFM discretizations, thereby converting the preceding approximation bounds
    
[^85]: 利用深度变分框架计算受限近晶相液晶的稳定构型

    Computing stable configurations of confined smectic liquid crystals with a deep variational framework

    [https://arxiv.org/abs/2609.03389](https://arxiv.org/abs/2609.03389)

    提出一种深度变分框架（DVF），通过坐标映射处理物理约束并利用预热惩罚克服神经网络的谱偏差，实现了复杂受限几何下近晶相液晶稳定构型（含高频层状密度调制）的稳健计算。

    

    近晶相液晶是一类具有取向有序和周期性密度调制的层状液晶相。尽管其结构可以用连续介质理论建模，但在复杂几何形状中计算稳定构型仍然具有挑战性，尤其是当需要分辨与近晶相层化相关的高频密度调制时。我们提出了一种深度变分框架（DVF），用于在修正的Landau–de Gennes模型内计算这些构型。在该框架中，相互耦合的取向序参量和位置序参量在规则的参考域上进行表示，而物理约束则通过坐标映射加以引入。预热惩罚项缓解了神经网络偏向平滑、非层状场的谱偏差，从而能够稳健地恢复振荡的近晶相状态。与神经网络基线方法及有限差分弛豫方法的比较证明了其优势。

    arXiv:2609.03389v1 Announce Type: cross  Abstract: Smectic liquid crystals are layered liquid-crystalline phases characterized by orientational order and periodic density modulation. Although their structures can be modeled using continuum theories, computing stable configurations remains challenging in complex geometries, particularly when the high-frequency density modulations associated with smectic layering should be resolved. We propose a deep variational framework (DVF) for computing these configurations within the modified Landau--de Gennes model, in which the coupled orientational and positional order parameters are represented on a regular reference domain while physical confinement is incorporated through coordinate mappings. A warmup penalty mitigates the spectral bias of neural networks toward smooth, nonlayered fields, enabling robust recovery of oscillatory smectic states. Comparisons with a neural-network baseline and finite-difference relaxation demonstrate the essentia
    
[^86]: TIGPO：面向长程LLM智能体的时序实例图策略优化

    TIGPO: Temporal Instance-Graph Policy Optimization for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2609.03383](https://arxiv.org/abs/2609.03383)

    TIGPO通过为每个任务维护跨策略更新的持久化转移图，并结合探索槽位与重访槽位的预算分配机制，将基于图的信用分配扩展到时序维度，从而显著改进长程LLM智能体的信用分配与优势估计。

    

    基于图的策略优化通过将展开轨迹组织成状态转移图，改进了长程LLM智能体的信用分配。然而，现有方法在每次策略更新中独立构建图，丢弃了早期策略所发现的转移，并将优势估计限制在小规模的批内局部展开组中。我们提出时序实例图策略优化（TIGPO），将基于图的信用分配扩展至跨策略更新。TIGPO为每个任务维护一个持久化的转移图，使不同策略版本发现的有效转移能够共同决定当前展开的信用分配。为了主动地将当前探索与历史经验重新连接起来，TIGPO将固定的展开预算分配于用于普通任务采样的探索槽位与用于延迟重试先前已探索任务的重访槽位之间。对于每次重访，TIGPO将当前……

    arXiv:2609.03383v1 Announce Type: new  Abstract: Graph-based policy optimization improves credit assignment for long-horizon LLM agents by organizing rollout trajectories into state-transition graphs. However, existing methods construct graphs independently within each policy update, discarding transitions discovered by earlier policies and limiting advantage estimation to small, batch-local rollout groups. We propose \emph{Temporal Instance-Graph Policy Optimization} (TIGPO), which extends graph-based credit assignment across policy updates. TIGPO maintains a persistent transition graph for each task, allowing valid transitions discovered by different policy versions to jointly determine credit for current rollouts. To actively reconnect current exploration with historical experience, TIGPO allocates a fixed rollout budget between Exploration slots for ordinary task sampling and Revisit slots for delayed reattempts of previously explored tasks. For each revisit, TIGPO pairs the curren
    
[^87]: SurgeGen：一种用于风暴潮情景合成的混合生成扩散框架

    SurgeGen: A Hybrid Generative Diffusion Framework for Storm Surge Scenario Synthesis

    [https://arxiv.org/abs/2609.03382](https://arxiv.org/abs/2609.03382)

    本文提出SurgeGen，一个基于扩散模型的两阶段生成框架，将基线预测与条件生成相结合，以连续空间中定义的假想风暴参数为条件合成风暴潮情景，为计算昂贵的物理数值模拟提供了更高效、更具可解释性的代理建模方案。

    

    预测登陆热带气旋引发的风暴潮对于防洪减灾和沿海风险管理至关重要。传统上，基于物理的数值模型通过数值方法求解Navier-Stokes方程来模拟风暴潮，但这类模拟的计算成本十分高昂。生成模型在风暴潮仿真代理建模方面前景广阔，因为它们能够生成多样化的实现结果，而非单一的确定性预测。然而，生成模型在风暴潮仿真代理建模中的应用在很大程度上仍未被探索。在本文中，我们利用扩散模型进行风暴潮代理建模，将基线预测阶段与条件生成相结合，以提供更具可解释性的建模框架。我们开发了SurgeGen，这是一个两阶段生成框架，能够以在连续空间中定义参数的假想风暴为条件，生成风暴潮情景。首先，基线模型产生……

    arXiv:2609.03382v1 Announce Type: cross  Abstract: Predicting storm surge induced by landfalling tropical cyclones is crucial for flood mitigation and coastal risk management. Traditionally, physics-based numerical models simulate storm surge by solving the Navier--Stokes equations using numerical methods, but these simulations are computationally expensive. Generative models are promising for storm surge emulation because they can generate diverse realizations rather than producing a single deterministic prediction. However, their use for storm surge emulation remains largely unexplored. In this paper, we leverage diffusion models for storm surge surrogate modeling, combining a baseline prediction stage with conditional generation to provide a more interpretable modeling framework. We develop SurgeGen, a two-stage generative framework for generating storm surge scenarios conditioned on hypothetical storms with parameters defined in a continuous space. First, a baseline model produces 
    
[^88]: RecurTrace：基于循环时间记忆的自适应潜在推理

    RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory

    [https://arxiv.org/abs/2609.03379](https://arxiv.org/abs/2609.03379)

    RecurTrace通过循环记忆注意力使循环层能够回顾之前迭代的计算状态，并结合由oracle监督的停止头动态决定循环次数，实现了更高效的自适应潜在推理。

    

    重复一个小型的中间层模块可以在不增加参数或生成额外token的情况下提升语言模型的有效推理深度，近期研究表明这种潜在递归能够改善推理能力。然而，有两个设计选择限制了这些收益：每次迭代只能看到上一次的输出，无法直接访问更早的计算结果；同时，固定的循环次数在简单输入上浪费了深度，而在困难输入上计算量又不足。我们提出RecurTrace，利用循环自身的轨迹来解决这两个限制。具体而言，循环记忆注意力让每个循环层沿着“循环时间”轴关注自己在之前迭代中的状态，使模型能够回顾更早的计算，而不仅依赖最新的状态。随后，一个停止头读取循环状态并预测是否继续迭代，其监督信号来自一个能够识别何时增加深度仍能降低损失的oracle。

    arXiv:2609.03379v1 Announce Type: new  Abstract: Repeating a small block of middle layers increases a language model's effective inference depth without adding parameters or generating extra tokens, and recent work shows that this latent recurrence improves reasoning. However, two design choices limit these gains. Each iteration sees only the previous output and cannot directly access earlier computations. Moreover, a fixed loop count wastes depth on easy inputs while leaving hard ones with too little computation. We introduce RecurTrace, which addresses both limitations using the loop's own trajectory. Specifically, Loop Memory Attention lets each looped layer attend to its own states from previous iterations along the loop-time axis, so the model can revisit earlier computations instead of relying on the latest state alone. A halting head then reads the loop state and predicts whether to continue, with supervision from an oracle that identifies when additional depth still reduces los
    
[^89]: SimpleDesign：蛋白质序列与结构协同设计的联合模型

    SimpleDesign: A Joint Model for Protein Sequence and Structure Codesign

    [https://arxiv.org/abs/2609.03377](https://arxiv.org/abs/2609.03377)

    SimpleDesign提出了一种直接在数据空间训练的单阶段端到端多模态生成模型，无需传统的多阶段潜在空间训练即可实现蛋白质序列与结构的协同设计。

    

    蛋白质是生物过程的基础，其功能由氨基酸序列与三维结构之间复杂的相互作用共同决定。开发能够理解这种内在多模态关系的生成模型，对于药物发现和蛋白质工程等领域至关重要。现有模型通常依赖于多阶段训练过程：第一阶段训练将数据标记化为潜在表示的自编码器，随后在自编码器的潜在表示上训练生成模型，即在潜在空间中进行生成建模。我们假设这种多阶段训练对于获得高性能的协同设计模型并非必要，因此提出了SimpleDesign——一个直接在数据空间中训练的高效多模态蛋白质设计模型。SimpleDesign采用单阶段端到端的目标函数，结合了针对序列的离散交叉熵（摘要原文在此处截断）。

    arXiv:2609.03377v1 Announce Type: new  Abstract: Proteins are fundamental to biological processes, with their function determined by the complex interplay between the amino acid sequence and the three-dimensional structure. Developing generative models capable of understanding this intrinsically multi-modal relationship is crucial for fields like drug discovery and protein engineering. Existing models often rely on a multi-stage training process where autoencoders that tokenize data into latent representations are trained in a first stage. Secondly, a generative model is trained on the latent representation of the autoencoder(s), i.e., generative modeling in a latent space. We hypothesize that this multi-stage training is not necessary to obtain performant co-design models and thus present SimpleDesign, an effective multi-modal protein design model trained directly in the data space. SimpleDesign leverages a single-stage end-to-end objective that combines discrete cross-entropy for seq
    
[^90]: Spruce：基于紧凑嵌入的可扩展隐私外包检索

    Spruce: Scalable Private Outsourced Retrieval Using Compact Embeddings

    [https://arxiv.org/abs/2609.03376](https://arxiv.org/abs/2609.03376)

    Spruce通过将紧凑二进制嵌入表示与密码学协议协同设计，用汉明距离计算取代全语料库嵌入评分，将百万文档规模的隐私外包检索性能显著提升。

    

    arXiv:2609.03376v1 公告类型：cross 摘要：检索增强生成（RAG）使得在大规模文档集合上进行密集检索成为标准构建模块。组织越来越多地将向量索引外包给不受信任的云服务，这会导致专有语料库和用户查询暴露。密码学保护面临挑战，因为每个查询都需要搜索语料库规模的状态，导致计算量、相关随机数和通信量随语料库规模增长。在百万文档规模下，朴素的安全实现每个查询需要数分钟时间和约90GB的通信量。即使是近期优化的系统也需要10至22秒。我们提出Spruce（基于紧凑嵌入的可扩展隐私外包检索），它将表示学习与密码学协议进行协同设计。Spruce学习紧凑的二进制编码，在为全精度重排序保留候选结果的同时，在双服务器多方计算框架下用高效的汉明距离计算取代了全语料库的嵌入评分。

    arXiv:2609.03376v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) has made dense retrieval over large document collections a standard building block. Organizations increasingly outsource vector indexes to untrusted clouds, exposing proprietary corpora and user queries. Cryptographic protection is challenging because each query searches corpus-scale state, causing computation, correlated randomness, and communication to grow with the corpus. At million-document scale, a naive secure implementation takes minutes and about 90 GB of communication per query. Even recent optimized systems require 10--22 seconds.   We propose Spruce (Scalable Private Outsourced Retrieval Using Compact Embeddings), which co-designs representations with the cryptographic protocol. Spruce learns compact binary codes that preserve candidates for full-precision reranking, replacing corpus-wide embedding scoring with efficient Hamming-distance computation under two-server multi-party computati
    
[^91]: 卷积滤波器子空间的 Grassmann–Plücker 参数化：正则性与闭嵌入

    Grassmann--Pl\"ucker Parametrization of Convolutional Filter Subspaces: Regularity and Closed Embeddings

    [https://arxiv.org/abs/2609.03361](https://arxiv.org/abs/2609.03361)

    该论文提出以滤波器空间中固定维数的子空间（而非有序滤波器向量族）作为单层卷积的几何参数化，并通过 Grassmann–Plücker 嵌入证明该参数化具有良好正则性（微分处处单射）且构成闭嵌入。

    

    我们为单层卷积层中的滤波器提出了一种几何参数化方法：参数不再是一组有序的滤波器向量，而是滤波器空间中一个固定维数的子空间。对于一维有限步长卷积，滤波器到卷积算子的对应关系给出了一个单射线性映射 $\mathcal{C}:\mathcal{K}\to H$。该映射将 $\mathrm{Gr}(q,\mathcal{K})$ 中的滤波器子空间映为 $\mathrm{Gr}(q,H)$ 中的算子子空间；将其与 Plücker 嵌入复合，便得到射影参数化 $\Phi:\mathrm{Gr}(q,\mathcal{K})\to\mathbb{P}(\bigwedge^q H)$。利用切空间同构 $T_U\mathrm{Gr}(q,\mathcal{K})\cong\mathrm{Hom}(U,\mathcal{K}/U)$，我们计算了诱导的格拉斯曼流形映射的微分，并证明 $\Phi$ 的微分在每一点处均为单射。随后，我们利用 Plücker 坐标的消失方程以及格拉斯曼流形上的标准仿射坐标，证明了……（原文摘要在此处截断）

    arXiv:2609.03361v1 Announce Type: cross  Abstract: We propose a geometric parametrization of the filters in a single convolutional layer: the parameter is no longer an ordered family of filter vectors, but a fixed-dimensional subspace of the filter space. For one-dimensional finite-stride convolution, the filter-to-convolution-operator correspondence gives an injective linear map $\mathcal{C}:\mathcal{K}\to H$. This map sends filter subspaces in $\mathrm{Gr}(q,\mathcal{K})$ to operator subspaces in $\mathrm{Gr}(q,H)$; composing it with the Pl\"ucker embedding yields a projective parametrization $\Phi:\mathrm{Gr}(q,\mathcal{K})\to\mathbb{P}(\bigwedge^q H)$. Using $T_U\mathrm{Gr}(q,\mathcal{K})\cong\mathrm{Hom}(U,\mathcal{K}/U)$, we compute the differential of the induced Grassmannian map and show that the differential of $\Phi$ is injective at every point. We then use the vanishing equations for Pl\"ucker coordinates and standard affine coordinates on a Grassmannian to prove that $\math
    
[^92]: 无需时间步长的时间：通过自洽性模拟耦合动力学系统

    Time Without Timesteps: Simulating Coupled Dynamical Systems via Self-Consistency

    [https://arxiv.org/abs/2609.03358](https://arxiv.org/abs/2609.03358)

    该论文提出用神经代理模型将完整轨迹到完整轨迹的映射与轨迹间的自洽不动点迭代相结合，替代传统的逐步时间推进来模拟耦合动力学系统，将参考积分器的1500步压缩为4-10次牛顿迭代，并使梯度计算变为内存与深度无关的GMRES线性求解。

    

    动力系统的数值模拟通常被组织为随时间因果推进的过程：每个状态由前一个状态计算得出。我们为耦合系统探索了一种不同的表述方式。对于每种子系统类型，我们训练一个神经代理模型，将完整的驱动轨迹和初始条件直接映射到完整的输出轨迹；遵循经典的波形松弛方法，耦合系统通过在这些轨迹之间强制自洽性来组装：模拟由此变成了在完整轨迹上求解不动点问题，而非逐步滚动推进。在耦合的范德波尔振子和霍奇金-赫胥黎神经元网络上，顺序深度变成了求解器的迭代次数：参考积分器需要1500步，而该方法只需4-10次牛顿迭代。梯度同样失去了其时间递归结构：它变成一个由GMRES求解的线性系统，其内存消耗与求解器深度无关。从学习到的算子上测量的单个标量（原文摘要至此截断）

    arXiv:2609.03358v1 Announce Type: new  Abstract: Numerical simulation of dynamical systems is usually organized as a causal march through time: each state is computed from the previous one. We explore a different formulation for coupled systems. For each subsystem type we train a neural surrogate mapping a full driving trajectory and initial condition directly to a full output trajectory; following classical waveform relaxation, coupled systems are assembled by enforcing self-consistency among these trajectories: simulation becomes a fixed-point problem over complete trajectories rather than a stepwise rollout. On coupled van der Pol oscillators and Hodgkin-Huxley neuron networks, sequential depth becomes the number of solver iterations: 4-10 Newton iterations where the reference integrator takes 1500 steps. The gradient likewise loses its time recursion: it becomes a linear system solved by GMRES at memory independent of solver depth. A single scalar measured from the learned operator
    
[^93]: ALRA：用于自回归语言模型基于Logit预训练蒸馏的自适应局部关系对齐

    ALRA: Adaptive Local Relational Alignment for Logit-Based Pre-training Distillation of Autoregressive Language Models

    [https://arxiv.org/abs/2609.03355](https://arxiv.org/abs/2609.03355)

    提出ALRA框架，通过让学生提议候选词元并以教师最可能词元作为锚点，同时根据教师概率分布广度自适应调整候选词元数量，从而改进自回归语言模型的局部logit蒸馏效果。

    

    基于Logit的知识蒸馏在自回归语言模型中通常是在整个词表上对齐教师模型和学生模型的下一词元分布。然而，这种全局目标忽略了可能词元候选之间的相对偏好。现有的局部方法通常仅从教师或学生单方选择候选词元。仅由教师选择可能会遗漏学生认为可能的词元，而仅由学生选择则可能在训练早期依赖不准确的排序。我们提出自适应局部关系对齐，这是一个结合学生提议与教师指导的位置特定框架。在每个有效预测位置，由学生提出可能的词元，同时将教师最可能的词元作为锚点纳入其中。ALRA根据教师在该候选集内相对于当前批次的概率分布广度来自适应地调整所选词元的数量。

    arXiv:2609.03355v1 Announce Type: cross  Abstract: Logit-based knowledge distillation for autoregressive language models usually aligns teacher and student next-token distributions over the entire vocabulary. However, this global objective overlooks relative preferences among likely token alternatives. Existing local approaches often select candidate tokens from either the teacher or the student alone. Teacher-only selection can miss tokens that the student considers likely, while student-only selection can rely on an inaccurate ranking early in training. We propose Adaptive Local Relational Alignment (ALRA), a position-specific framework combining student proposals with teacher guidance. At each valid prediction position, the student proposes likely tokens, while the teacher's most probable token is included as an anchor. ALRA adjusts the number of selected tokens according to how broadly the teacher distributes probability within this candidate set relative to the current batch. Adap
    
[^94]: 基于GPU加速树形遗传编程的符号回归高效常数优化

    Efficient Constant Optimization for Symbolic Regression with GPU-Accelerated Tree-Based Genetic Programming

    [https://arxiv.org/abs/2609.03352](https://arxiv.org/abs/2609.03352)

    该论文提出了一种驻留GPU的批量Levenberg-Marquardt求解器，通过每次迭代仅需固定次数CUDA启动的设计，首次在GPU加速的树形遗传编程符号回归中对结构异构的表达式树种群实现高效常数优化，在NVIDIA A100上达到每秒5.1×10⁵棵树的速度，并保证优化后的常数不会劣于初始值。

    

    常数优化用于在基于树形结构的遗传编程符号回归中精炼候选表达式的数值系数。但其每代计算的开销使得现代GPU加速框架要么省略这一步骤，要么将其限制为轻量级形式。我们提出了一种驻留于GPU的批量Levenberg--Marquardt求解器，能够对结构异构的表达式树种群进行常数优化，每次迭代仅需固定次数的种群级CUDA启动。反向模式自动微分通过一次反向传播即可组装每棵树的雅可比矩阵，使得每次迭代的主要开销与每棵树中常数的数量无关；同时，双精度交付保护机制保证返回的常数绝不会劣于其初始值。在早期代种群上，该求解器在NVIDIA A100上可维持每秒最高5.1×10⁵棵树的处理速度；在GPU饱和的基准测试配置下，它……（摘要被截断）

    arXiv:2609.03352v1 Announce Type: cross  Abstract: Constant optimization refines the numerical coefficients of candidate expressions in tree-based genetic programming for symbolic regression. But its per-generation cost has led modern GPU-accelerated frameworks to omit it or restrict it to lightweight forms. We present a GPU-resident, batched Levenberg--Marquardt solver that optimizes constants across a structurally heterogeneous population of expression trees using a fixed number of population-wide CUDA launches per iteration. Reverse-mode automatic differentiation assembles the per-tree Jacobian in one backward sweep, making the dominant per-iteration cost independent of the number of constants per tree, and a double-precision delivery guard guarantees that returned constants are never worse than their initial values. On early-generation populations, the solver sustains up to $5.1{\times}10^{5}$ trees per second on an NVIDIA A100; at a GPU-saturated benchmark configuration it deliver
    
[^95]: 从零到英雄：面向亚美尼亚语的开放大语言模型生态系统

    From Zero to Hero: An Open LLM Ecosystem for Armenian

    [https://arxiv.org/abs/2609.03350](https://arxiv.org/abs/2609.03350)

    该研究发布了ArmWeb（437万篇亚美尼亚语新闻）与ArmSTEM（37.3万条英亚平行数理题目）两个数据集，并通过继续预训练Gemma-4-E4B构建出首个附带完整训练数据和配方的开源亚美尼亚语模型arm-gemma-e4b，其性能超越所有现有的开放亚美尼亚语模型。

    

    亚美尼亚语是一种形态丰富但资源稀缺的语言，其预训练数据十分匮乏，且目前尚无任何开放的亚美尼亚语大语言模型附带可复现所需的数据与训练配方。为填补这一空白，我们整理并发布了两个数据集。ArmWeb是一个经过广泛验证的语料库，包含437万篇亚美尼亚语新闻文档。ArmSTEM是一个英亚平行数据集，包含37.3万个附有分步解答的数学与科学题目，这些题目被翻译为亚美尼亚语，并通过保留答案的LLM判断和人工评估进行了双重验证。在Gemma-4-E4B上利用这些数据集进行继续预训练，得到arm-gemma-e4b，其性能超越了所有现有的开放亚美尼亚语模型以及未经适配的基础模型，并且是首个附带完整训练数据和配方的开放亚美尼亚语大语言模型。我们的消融实验表明，仅使用新闻数据进行继续预训练虽然能提升语言流畅度，却会侵蚀知识能力——我们在现有的亚美尼亚语模型中也观察到了这一模式；此外还表明，少量的……（原文在此处截断）

    arXiv:2609.03350v1 Announce Type: cross  Abstract: Pretraining data for Armenian, a morphologically rich and low-resource language, is scarce, and no open Armenian LLM has been released with the data and recipe needed to reproduce it. To address this gap, we curate and release two datasets. ArmWeb is an extensively validated corpus of 4.37M Armenian news documents. ArmSTEM is a parallel English-Armenian collection of 373K math and science problems with step-by-step solutions, translated into Armenian and verified through both answer-preserving LLM judgment and human evaluation. Continued pretraining of Gemma-4-E4B on these datasets yields arm-gemma-e4b, which outperforms every existing open Armenian model as well as its unadapted base, and is the first open Armenian LLM with complete training data and recipe. Our ablations show that news-only continued pretraining improves fluency while eroding knowledge, a pattern we also observe in existing Armenian models, and that a small share of 
    
[^96]: 基于无限维连续归一化流学习信息先验的贝叶斯逆问题方法

    Learning Informative Prior with Infinite-Dimensional Continuous Normalizing Flow for Bayesian Inverse Problem

    [https://arxiv.org/abs/2609.03343](https://arxiv.org/abs/2609.03343)

    该论文提出了一种基于无限维连续归一化流的新方法，通过在希尔伯特空间中引入定义良好的神经常微分方程将简单参考测度变换为编码先验信息的复杂测度，建立了无限维贝叶斯先验的适定性理论框架，并提供了先验训练方法与后验采样算法，用于求解偏微分方程的贝叶斯逆问题。

    

    本文研究了具有无限维希尔伯特空间中模型参数的偏微分方程逆问题的无限维贝叶斯推断。为了有效融入先验信息，我们提出了一种新颖的基于连续归一化流的无限维模型。具体而言，通过在无限维空间中引入一个定义良好的神经常微分方程，可以将简单的参考测度变换为编码先验信息的更复杂测度。我们建立了相应的理论框架，以确保所提出的贝叶斯先验在无限维空间中的适定性。我们还针对两种不同的数据设置提供了先验的训练方法，并为由此得到的贝叶斯后验提供了两种采样算法。所提出的框架被应用于三个代表性的逆问题：简单的光滑逆问题、逆散射问题等。

    arXiv:2609.03343v1 Announce Type: cross  Abstract: This paper addresses infinite-dimensional Bayesian inference for inverse problem of partial differential equations with model parameters in infinite-dimensional Hilbert space. To effectively incorporate prior information, we propose a novel continuous normalizing flows based infinite-dimensional model. Specifically, by introducing a well-defined neural ordinary differential equation in infinite-dimensional space, a simple reference measure can be transformed into a more complex measure which encodes the prior information. A corresponding theoretical framework is established to ensure the well-posedness of our proposed Bayesian prior in infinite-dimensional space. We also provide training methods of the prior for two distinct data settings, along with two sampling algorithms for the resulting Bayesian posterior. The proposed framework is applied to three representative inverse problems: the simple smooth inverse problem, inverse scatter
    
[^97]: 梯度知晓结果所不知：利用梯度对齐奖励解锁大语言模型推理的强化学习

    Gradients Know What Outcomes Don't: Unlocking Reinforcement Learning for LLM Reasoning with Gradient-Aligned Rewards

    [https://arxiv.org/abs/2609.03342](https://arxiv.org/abs/2609.03342)

    提出梯度对齐奖励（GAR）：在策略自身的梯度空间中，通过截断反向传播提取每个 rollout 的梯度向量并与专家锚点梯度计算余弦相似度，以低于 9% 的时间开销生成密集的推理感知奖励，突破了 RLVR 二值结果奖励无法区分正确轨迹的局限，并可乘性分解为预测误差与激活模式两个因子。

    

    基于可验证奖励的强化学习（RLVR）驱动了大语言模型的思维链推理，然而其二值的结果奖励无法区分不同的正确轨迹。现有的密集奖励替代方案，从表面启发式方法到过程奖励模型，要么忽略了训练语料中已有的专家解答，要么需要代价高昂的离线标注。我们提出了梯度对齐奖励（GAR），它在策略自身的梯度空间中运行：通过输出投影层进行截断反向传播，为每个 rollout 提取一个紧凑的梯度向量，并与专家锚点梯度计算余弦相似度，从而产生一种密集的、推理感知的奖励，其墙钟时间开销低于 9%。我们证明该余弦值可进行乘性分解为预测误差因子和激活模式因子，从而具体刻画了对齐信号所度量的内容。在 Qwen3-4B 和 Qwen3-8B 上……（原文摘要在此处截断）

    arXiv:2609.03342v1 Announce Type: new  Abstract: Reinforcement learning from verifiable rewards (RLVR) drives chain-of-thought reasoning in large language models, yet its binary outcome reward cannot distinguish among correct trajectories. Existing dense reward alternatives, from surface heuristics to process reward models, either ignore the expert solutions already present in training corpora or require expensive offline annotation. We propose Gradient-Aligned Reward (GAR), which operates in the policy's own gradient space: truncated backpropagation through the output projection layer extracts a compact gradient vector for each rollout, and cosine similarity with an expert-anchor gradient yields a dense, reasoning-aware reward with less than 9% wall-clock overhead. We prove that this cosine admits a multiplicative decomposition into prediction-error and activation-pattern factors, providing a concrete characterization of what the alignment signal measures. On Qwen3-4B and Qwen3-8B, GA
    
[^98]: 一个大型开放的多能量土体压实试验语料库，附机器学习基线

    A Large Open Multi-Energy Corpus of Soil Compaction Tests, with Machine-Learning Baselines

    [https://arxiv.org/abs/2609.03337](https://arxiv.org/abs/2609.03337)

    本文发布了迄今规模最大的开放多能量土体压实试验数据集（2854个试验、四个普氏能量等级），经物理一致性审核后发现已发表压实数据中相当一部分在物理上不可行，并给出了最优饱和度基线值0.815及机器学习预测基线。

    

    每一项工程填土都由最大干密度和最优含水率来规定，而每次确定都需要一次完整的普氏试验。已发表的关联式仅建立在一到四百个试样之上，通常来自同一实验室、同一压实能量，且很少对外公开。本文发布了一个不受这些限制的语料库：该语料库包含来自六个公开数据源的2,854个实验室压实试验，涵盖162个来源组和四个普氏能量等级，细粒含量从1.5%到100%。每条记录都根据其来源所注明的普氏方法进行审核，且不推断任何能量。基于零孔隙气条件的筛选剔除了11.8%的协调化记录，而在有实测比重的记录中剔除了5.7%。已发表压实数据中有相当大比例在物理上是不可能的。整个语料库的最优饱和度为0.815，变异系数为11%。这是一个基线，而非一个常数。随后对两个参数进行了估计……（摘要在此处截断）

    arXiv:2609.03337v1 Announce Type: new  Abstract: Every engineered fill is specified by a maximum dry density and an optimum moisture content. Each determination needs a full Proctor test. Published correlations rest on one to four hundred specimens, usually from one laboratory at one compactive energy, and are seldom released. This paper releases a corpus without those limits. It holds 2,854 laboratory compaction tests from six public sources, across 162 provenance groups and four Proctor energy levels, with fines from 1.5 to 100%. Every record is audited to the Proctor method its source names, and no energy is inferred. Screening on the zero-air-voids condition removed 11.8% of harmonised records, and 5.7% of those with a measured specific gravity. A material share of published compaction data is physically impossible. The optimum degree of saturation over the corpus is 0.815 at a coefficient of variation of 11%. That is a baseline, not a constant. Both parameters are then estimated f
    
[^99]: 介绍SINFONIA：用于轨道数值积分与加速的辛、Slimplectic和Magnus（神经）流

    Introducing SINFONIA: Symplectic, slimplectic and Magnusian (Neural) Flows for Orbital Numerical Integration and Acceleration

    [https://arxiv.org/abs/2609.03329](https://arxiv.org/abs/2609.03329)

    本文提出SINFONIA三种保结构神经流架构（辛/Slimplectic流、Taylor锚定流和Magnus流）用于长时引力波旋近轨道的数值积分，并发现长期精度由映射误差在能量-角动量平衡所决定的单一长期通道上的带符号投影控制。

    

    长时间引力波建模必须同时解析快速轨道运动与缓慢的耗散演化，同时防止微小的数值误差累积为长期相位漂移。在本文中，我们探究有限时间演化映射本身能否被学习为一个显式、可微、保结构的对象，并通过完整旋近过程的反复复合来验证。我们构建了三种神经流架构：在Galley加倍相空间上的辛与Slimplectic流[SINFONIA-J0]；以Taylor展开为锚定的流[SINFONIA-J1]；以及在相互作用绘景中学习有限时间耗散修正的Magnus流[SINFONIA-J2]。应用于2.5PN阶中子星旋近时，三者均揭示了相同的控制机制：长期精度并非仅由逐点映射误差决定，而是由该误差在由能量-角动量平衡固定的单一长期通道上的带符号投影所支配。编码这种结构……（原文摘要在此处截断）

    arXiv:2609.03329v1 Announce Type: cross  Abstract: Long-duration gravitational-wave modelling must resolve fast orbital motion together with slow dissipative evolution while preventing small numerical errors from accumulating into secular phase drift. Here we ask whether the finite-time evolution map itself can be learned as an explicit, differentiable, structure-preserving object and then repeatedly composed through a complete inspiral. We construct three neural-flow architectures: a symplectic and slimplectic flow on Galley's doubled phase space, [SINFONIA-J0]; a Taylor-anchored flow, [SINFONIA-J1]; and a Magnusian flow that learns the finite-time dissipative correction in the interaction picture, [SINFONIA-J2]. Applied to a 2.5PN neutron-star inspiral, all three expose the same controlling mechanism: long-time accuracy is governed not by pointwise map error alone, but by its signed projection onto a single secular channel fixed by energy--angular-momentum balance. Encoding this stru
    
[^100]: DE-Venus：一个面向大语言模型的数据高效RLVR框架

    DE-Venus: A Data-Efficient RLVR Framework for Large Language Models

    [https://arxiv.org/abs/2609.03324](https://arxiv.org/abs/2609.03324)

    DE-Venus提出了一个统一的数据高效RLVR框架，将监督视为跨数据准备与策略优化不断演化的状态，通过主动数据选择、弱监督构建和训练时监督精炼三个模块，降低大语言模型推理训练中的采样与标注成本。

    

    基于可验证奖励的强化学习（RLVR）能够提升大语言模型的推理能力，但其实际扩展受到昂贵的在线策略采样以及大规模获取可靠目标信号成本的制约。现有方法分别处理样本选择、不完整监督或噪声标签问题，往往将监督逻辑与分布式训练纠缠在一起，阻碍了受控比较与复用。我们提出了DE-Venus，一个面向数据高效RLVR的统一框架，它将监督视为在数据准备与策略优化过程中不断演化的状态。该框架将这一生命周期组织为三个模块：主动数据选择负责分配训练与标注预算；弱监督构建从无标签示例中推导学习信号；训练时监督精炼则过滤或纠正不可靠的监督信号。DE-Venus通过统一的表达方式支持七种代表性方法以及一个数据选择流程……

    arXiv:2609.03324v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) improves large language model reasoning, but its practical scaling is constrained by expensive on-policy rollouts and the cost of obtaining reliable targets at scale. Existing methods address sample selection, incomplete supervision, or noisy labels separately, often entangling supervision logic with distributed training and hindering controlled comparison and reuse. We present DE-Venus, a unified framework for data-efficient RLVR that treats supervision as evolving state across data preparation and policy optimization. It organizes this lifecycle into three modules: Active Data Selection allocates training and annotation budgets; Weak Supervision Construction derives learning signals from unlabeled examples; and Training-Time Supervision Refinement filters or corrects unreliable supervision. DE-Venus supports seven representative methods and a data-selection pipeline by expressing me
    
[^101]: 超越.WAV：VocalCap——面向声音生物标志物研究的可追溯浏览器端音频采集系统的设计与软件验证

    Beyond .WAV: Design and Software Verification of VocalCap, a Traceable Browser-Based Audio Capture System for Vocal Biomarker Research

    [https://arxiv.org/abs/2609.03320](https://arxiv.org/abs/2609.03320)

    VocalCap是一个由机构控制的、可追溯的浏览器端音频采集系统，通过版本化协议驱动工作流，并为每条录音保留从采集执行到字节级完整性和转换来源的多层级证据，从而为声音生物标志物研究的远程语音采集提供端到端的可验证性与完整性保障。

    

    远程语音研究通常只保留最终的音频文件，而关于该文件如何被采集、传输、处理和接受的信息非常有限。本文提出了VocalCap，一个由机构控制的、基于浏览器的系统，可供未经技术培训的参与者自主完成语音及相关声学信号的采集。一个带有版本控制的协议驱动整个工作流程。每条被接受的录音都会保留一个浏览器原生对象、一个从同一MediaStream派生的客户端无损Float32 WAV文件，以及一个服务器规范的单声道PCM16 WAV文件，并与采集执行、技术质量、字节级完整性、恢复和转换来源等证据相链接。IndexedDB在服务器确认之前保存被接受的浏览器工件，而会话的完成要求每个任务和工件均通过成功验证。软件测试通过格式错误或被篡改的对象、精确零中断、通道拓扑变体等方式对采集契约进行了挑战。

    arXiv:2609.03320v1 Announce Type: cross  Abstract: Remote voice studies often retain a final audio file with limited evidence about how it was captured, transferred, processed, and accepted. This paper presents VocalCap, an institution-controlled, browser-based system for self-guided capture of voice and related acoustic signals by participants without technical training. A versioned protocol drives the workflow. Each accepted recording retains a browser-native object, a client-lossless Float32 WAV derived from the same MediaStream, and a server-canonical mono PCM16 WAV, linked to evidence of capture execution, technical quality, byte-level integrity, recovery, and transformation provenance. IndexedDB preserves accepted browser artifacts until server confirmation, while session completion requires successful verification of every task and artifact. Software tests challenged the acquisition contracts with malformed or altered objects, exact-zero interruptions, channel-topology variants,
    
[^102]: 基于强化学习与不确定性量化的配电网优化运行风险与异常识别

    Risk and Anomaly Identification for Distribution Network Optimal Operation Based on Reinforcement Learning and Uncertainty Quantification

    [https://arxiv.org/abs/2609.03308](https://arxiv.org/abs/2609.03308)

    本文提出一种融合分布式强化学习与贝叶斯深度强化学习的不确定性感知框架，通过将总不确定性分解为偶然与认知两个分量，实现对配电网优化运行中固有风险与分布外异常的联合识别。

    

    现代配电网的可靠运行需要在普遍存在的不确定性下及时识别运行风险和异常事件。在实际运行中，运营人员既要识别随机性但属于分布内条件所固有的风险，也要识别对应于分布外行为的异常，例如异常负载模式、极端天气或信息物理攻击。本文针对配电网优化运行中的风险与异常联合识别问题，提出了一种显式感知不确定性的深度强化学习框架。我们将分布式强化学习与贝叶斯深度强化学习相结合，实现了二阶不确定性量化方案，将总不确定性分解为偶然不确定性和认知不确定性两个分量，分别用于表征固有风险和分布外异常。由此得到的认知不确定性估计可同时驱动（摘要在此处被截断）。

    arXiv:2609.03308v1 Announce Type: new  Abstract: Reliable operation of modern distribution networks requires timely identification of operational risks and anomalous events under pervasive uncertainty. In practice, operators must identify risks that are inherent in stochastic yet in-distribution conditions, and anomalies that correspond to out-of-distribution behaviors such as unusual load patterns, extreme weather or cyber-physical attacks. This paper addresses this joint risk and anomaly identification problem for optimal distribution network operation and proposes a deep reinforcement learning framework that is explicitly uncertainty aware. We integrate distributional and Bayesian deep reinforcement learning to realize a second- order uncertainty quantification scheme that decomposes total uncertainty into aleatoric and epistemic components, which are respectively used to characterize inherent risk and out-of- distribution anomalies. The resulting epistemic estimates drive both expl
    
[^103]: 基于自适应谱带宽控制的几何感知图构建

    Geometry-Aware Graph Construction via Adaptive Spectral Bandwidth Control

    [https://arxiv.org/abs/2609.03306](https://arxiv.org/abs/2609.03306)

    该论文提出了一种逐节点自适应带宽选择准则，通过将核的有效秩与最小生成树估计的局部内在维度相匹配，使高斯核图的谱复杂度与数据流形的内在几何复杂度保持一致，从而避免带宽选择过小或过大带来的谱退化问题。

    

    使用高斯核的核化图方法——包括谱聚类、扩散映射和稀疏核ℓ1回归图——依赖于高斯带宽σ的选择，该带宽决定了局部核算子的谱特性。当σ过小时，核会高估局部复杂度，将每个样本视为独立的方向；当σ过大时，核会将多个方向坍缩在一起，导致条件数发散，所有几何判别能力随之丧失。我们提出了一种尺度选择方法，使核的谱复杂度与底层流形的内在复杂度保持一致。我们进一步提出了一个逐节点带宽准则来具体实现这一原则：通过最小生成树估计局部内在维度，并使核的有效秩与之相匹配，同时将搜索过程锚定在与流形一致的对数-对数缩放区域内。我们在自监督学习（SSL）嵌入上对该方法进行了评估。

    arXiv:2609.03306v1 Announce Type: new  Abstract: Kernelized graph methods - spectral clustering, diffusion maps, and sparse kernel -regression graphs - that use Gaussian kernels depend on the choice of Gaussian bandwidth sigma, which governs the spectral character of the local kernel operator. When sigma is too small, the kernel overestimates local complexity and treats each sample as an independent direction; when sigma is too large, the kernel collapses multiple directions together, the condition number diverges, and all geometric discrimination is lost. We propose a choice of scale to make the spectral complexity of the kernel consistent with the intrinsic complexity of the underlying manifold. We propose a per-node bandwidth criterion that operationalizes this principle by jointly matching the kernel's effective rank to the local intrinsic dimension estimated via minimum spanning tree, anchoring the search in the manifold-consistent log-log scaling regime. We evaluate SSL embedding
    
[^104]: 基于世界模型的潜空间能量动作规划

    Latent Energy Action Planning with World Models

    [https://arxiv.org/abs/2609.03294](https://arxiv.org/abs/2609.03294)

    提出 LEAP 方法，将完整动作序列视为可微变量，通过冻结的潜空间世界模型，将终端潜空间目标匹配与状态能量相结合进行优化，实现高效的动作规划。

    

    潜空间世界模型支持从高维观测中进行高效的模型预测控制，然而优化单一的学习式潜空间目标可能会偏向那些解码器预测的终端描述符与目标描述符不匹配的动作序列。我们提出了潜空间能量动作规划，它将完整的动作时域视为一个可微变量，并通过冻结的潜空间世界模型对其进行优化。LEAP 将终端潜空间目标匹配与终端窗口状态能量相结合，低能量要求预测的终端潜变量与目标潜变量一致，并且解码器预测的终端描述符与目标描述符一致。一个冻结的目标条件提议网络用于初始化搜索，准牛顿求解器通过自回归滚动来细化动作，优化后的投影则强制动作保持在允许范围内。在四个控制域中使用官方发布的 LeWM 检查点，

    arXiv:2609.03294v1 Announce Type: new  Abstract: Latent world models support efficient model predictive control from high-dimensional observations, yet optimizing a single learned latent objective can favor action sequences whose decoder-predicted terminal descriptor does not match the goal descriptor. We introduce Latent Energy Action Planning (LEAP), which treats the complete action horizon as a differentiable variable and optimizes it through a frozen LeWorldModel (LeWM). LEAP couples terminal latent goal matching with a terminal-window state energy. Low energy requires the predicted terminal latent to agree with the goal latent and the decoder-predicted terminal descriptor to agree with the goal descriptor. A frozen goal-conditioned proposal initializes the search, a quasi-Newton solver refines actions through the autoregressive rollout, and post-optimization projection enforces the admissible action range. Across four control domains using the officially released LeWM checkpoints,
    
[^105]: 面向冻结图聚类的选择性超图精化

    Selective Hypergraph Refinement for Frozen Graph Clustering

    [https://arxiv.org/abs/2609.03265](https://arxiv.org/abs/2609.03265)

    提出选择性超图精化（SHR）方法，在不使用标签、不更新模型参数、节点表示和图结构的前提下，利用属性超图补充高阶关系，并基于图结构、节点属性和匹配零假设证据对更新方向的可靠性进行评估，从而选择性地精化已冻结图聚类模型的聚类结果。

    

    现有的图聚类方法通常通过优化模型参数和节点表示来提升聚类性能。然而，对于已经训练完成并冻结的模型，进一步改善其聚类结果的有效手段仍然有限。我们研究冻结图聚类的后处理方法。在模型检查点固定之后，该过程不使用任何标签，也不更新模型参数、节点表示或原始图结构。相反，它利用属性超图来补充普通图难以表达的高阶关系，从而对现有的聚类分配进行精化。由于全局超图精化可能同时带来性能提升和错误的更新，我们提出了选择性超图精化（SHR）。该方法从超图中生成候选的残差方向，并利用图结构、节点属性以及匹配零假设证据来评估其可靠性。

    arXiv:2609.03265v1 Announce Type: new  Abstract: Existing graph-clustering methods typically improve clustering performance by optimizing model parameters and node representations. Effective means of further improving the clustering results of an already trained and frozen model, however, remain limited. We study post-processing for frozen graph clustering. After checkpoint fixation, the procedure uses no labels and updates neither model parameters, node representations, nor the original graph structure. Instead, it exploits an attribute hypergraph to supplement higher-order relations that ordinary graphs cannot readily express, thereby refining existing cluster assignments. Because global hypergraph refinement can yield both performance gains and erroneous updates, we propose Selective Hypergraph Refinement (SHR). The method generates candidate residual directions from the hypergraph and evaluates their reliability using graph structure, node attributes, and matched-null evidence. It 
    
[^106]: 还有什么需要修改？探索对话生成工件中修订传播的高性价比测试时计算方法

    What Else Needs Fixing? Exploring Cost-Effective Test-Time Compute for Revision Propagation in Artifacts Generated Through Conversation

    [https://arxiv.org/abs/2609.03254](https://arxiv.org/abs/2609.03254)

    本文针对对话生成工件中修订传播这一新问题提出了一个基准，评估了九种测试时计算修订方法，发现基线方法可达68.3%–93%的准确率，并识别出最具成本效益的方法。

    

    大语言模型（LLM）通常通过对话中生成与修订的迭代循环帮助用户创建工件。这里的一个挑战是，当用户在修订时仅指定局部更改时，LLM 必须识别相关的依赖关系，并将修订传播到工件中所有受影响的部分。本文研究了 LLM 在对话生成的工件上的这种能力，其中工件上下文及其依赖关系可能嵌入在对话历史中。面向实际应用，我们还探索了这一新设置下高性价比的测试时计算方法。具体而言，我们为该设置引入了一个新的基准，并使用 gpt-oss-20b/120b、gpt-5.4-mini 和 qwen3.5-9b/27b/122b 在该基准上评估了九种修订方法，包括顺序反思和并行采样变体。结果表明，基线方法达到了 68.3%–93% 的准确率，而最具成本效益的方法是……

    arXiv:2609.03254v1 Announce Type: new  Abstract: Large Language Models (LLMs) often help users generate artifacts through iterative cycles of generation and revision in conversation. A challenge here is that, when users specify only a local change during revision, LLMs must instead identify the relevant dependencies and propagate the revision to all affected parts of the artifact. This paper studies this ability of LLMs on conversationally generated artifacts, where the artifact context and its dependencies may be embedded in the conversation history. Toward practical use, we also explore cost-effective test-time compute for this new setting. Specifically, we introduce a new benchmark for this setting, and evaluate nine revision methods, including sequential reflection and parallel sampling variants, using gpt-oss-20b/120b, gpt-5.4-mini, and qwen3.5-9b/27b/122b on the benchmark. The results show that baselines achieve accuracies of 68.3--93%, and the most cost-effective method is selec
    
[^107]: 何为光滑性？

    What is Smoothness?

    [https://arxiv.org/abs/2609.03246](https://arxiv.org/abs/2609.03246)

    本文提出通过群 Cayley 图拉普拉斯算子各块特征值的均值来为非交换群的不可约表示构造一种自然的排序函数，从而将基于傅里叶分析的光滑性概念推广到一般群上。

    

    实数轴上函数的光滑性体现在其傅里叶变换的衰减速度上，这暗示着对于群 $G$ 上 $L^2(G)$ 空间中的函数，光滑性应意味着傅里叶系数集中于低频处。这样的解读预设了 $G$ 的不可约表示存在某种排序，但对于非交换群 $G$，并不存在典范的排序方式。给定一个对称生成集 $S$，相应 Cayley 图的拉普拉斯算子在对偶空间上是块对角的，我们通过每个块中特征值的均值来对不可约表示进行排序。由此得到一个仅依赖于二元组 $(G,S)$ 的排序函数 $\omega:\widehat{G}\to\mathbb{R}$。该函数的值介于零和二之间，仅在平凡表示处取零值，且当且仅当 Cayley 图为二部图时达到上界二。随后我们探讨这一构造具有多大的自由度。在满足若干自然公理的算子类中，所诱导的……（摘要原文在此处被截断）

    arXiv:2609.03246v1 Announce Type: cross  Abstract: Smoothness of a function on the real line is reflected in the decay of its Fourier transform, which suggests that smoothness of a function in $L^2(G)$ for a group $G$ should mean concentration of the Fourier coefficients at low frequency. Such a reading presupposes an ordering of the irreducible representations of $G$, but for non-abelian $G$, no ordering is canonical. Given a symmetric generating set $S$, the Laplacian of the associated Cayley graph is block diagonal over the dual, and we order the irreps by the mean of the eigenvalues in each block. This produces an ordering function $\omega:\widehat{G}\to\mathbb{R}$ that depends only on the pair $(G,S)$. This function is bounded between zero and two, vanishing only at the trivial representation and achieving the upper bound exactly when the Cayley graph is bipartite. We then ask how much freedom the construction has. Within the class of operators satisfying natural axioms, the induc
    
[^108]: FlowBalance：基于验证器的策略内推理经验自我改进方法

    FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience

    [https://arxiv.org/abs/2609.03241](https://arxiv.org/abs/2609.03241)

    FlowBalance提出一种以终端验证器的组优势来校准同模型自引导分数的自我改进方法，通过在正优势轨迹上保留、负优势轨迹上反转、无结果偏好时禁用引导，实现更稳定的推理能力自我提升。

    

    推理模型可以从自身的策略内经验中改进，但这一内部循环十分脆弱：终端验证器提供可靠却稀疏的监督信号，而同模型的密集引导可能会强化错误的自信，或将学习过度集中于狭窄的解题模式。我们提出了FlowBalance，一种以验证器为基础的自我改进方法，它学习完整回答上的归一化分布。对于每条策略内轨迹，同一策略的冻结训练时视图利用特权上下文产生token级别的对数概率增益，这些增益被聚合为轨迹级别的自引导分数。FlowBalance使用验证器得出的组优势来校准该分数：在正优势轨迹上保留引导，在负优势轨迹上反转引导，当rollout组未提供结果偏好时则禁用引导。由此得到的能量函数对参考策略进行指数级重加权……（摘要在此处截断）

    arXiv:2609.03241v1 Announce Type: cross  Abstract: A reasoning model can improve from its own on-policy experience, but this inner loop is fragile: terminal verifiers provide reliable yet sparse supervision, while dense same-model guidance can reinforce false confidence or overconcentrate learning on a narrow solution mode. We introduce FlowBalance, a verifier-grounded self-improvement method that learns a normalized distribution over complete responses. For each on-policy trajectory, a frozen training-time view of the same policy uses privileged context to produce token-level log-probability gains, which are aggregated into a trajectory-level self-guidance score. FlowBalance calibrates this score with the verifier-derived group advantage: guidance is retained on positive-advantage trajectories, reversed on negative-advantage trajectories, and disabled when the rollout group provides no outcome preference. The resulting energy exponentially reweights a reference policy, and profiled tr
    
[^109]: B2B客户转化预测：一种基于文档表示、图论和CatBoost驱动的方法论

    B2B Customer Conversion Prediction: A Document Representation, Graph Theory, and CatBoost Driven Methodology

    [https://arxiv.org/abs/2609.03239](https://arxiv.org/abs/2609.03239)

    本文提出了一种融合文档表示、图论和CatBoost模型的方法，通过多键聚合B2B客户数据并生成特征，实现高达91%准确率的客户购买转化预测。

    

    在一次性销售的B2B场景中，购买周期可能持续数月甚至数年。在这一漫长的过程中，锁定具有高购买潜力的客户并据此推荐个性化营销活动，对于有效的市场营销至关重要。为实现这一目标，我们研究了以下问题：B2B客户数据聚合、客户特征生成，以及预测B2B客户是否会表现出购买兴趣（即预测其是否转化为销售漏斗）。我们提出了一种基于多个键将个人联系人聚合到B2B客户层面的算法。对于非标准化的键（如公司名称），我们提出了一种新颖的架构，能够将它们聚类在同一个领域中，该领域涵盖了拼写错误和拼写变体等不规则情况。随后，我们定义并生成了一组特征，并应用CatBoost模型进行客户转化预测。我们的框架实现了91%的预测准确率。

    arXiv:2609.03239v1 Announce Type: new  Abstract: In the one-time selling B2B context, the buying cycle may last months or even years. During the long process, targeting customers that have a high potential to make purchases and recommending personalized campaigns accordingly are important for effective marketing. For this goal, we study the following problems, B2B customer data aggregation, customer feature generation, and prediction of whether a B2B customer would show interest in making a purchase (i.e., prediction of conversion into sales funnel). We propose an algorithm to aggregate individual contacts to the B2B customer level based on multiple keys. For non-standardized keys such as company names, we propose a novel architecture to cluster them in a domain encompassing irregularities such as spelling mistakes and spelling variants. We then define and generate a set of features and apply the CatBoost model for customer conversion prediction. Our framework achieves 91\% prediction 
    
[^110]: 2026年PNPL竞赛：LibriBrain100中的词汇分类与高效跨被试泛化

    The 2026 PNPL Competition: Word Classification and Efficient Cross-Subject Generalisation in LibriBrain100

    [https://arxiv.org/abs/2609.03231](https://arxiv.org/abs/2609.03231)

    2026年PNPL竞赛推出扩展MEG数据集LibriBrain100，聚焦词汇分类任务与高效跨被试泛化，旨在让语音解码脑机接口仅需几分钟数据即可泛化到新用户。

    

    2025年PNPL竞赛（Landau等人，2025）的初衷是启动一项面向非侵入式语音解码的多年期课程。该竞赛设计上从基础任务逐步迈向实用脑机接口（BCI）所需的语言复杂性，以语音检测和音素分类两个任务拉开序幕。获胜方案在相应任务上分别取得了95.6%和73.6%的F1-macro分数，是极为显著的进展。这一成功建立在LibriBrain数据集（Özdogan等人，2025）之上，该数据集是当时记录的最大的被试内MEG数据集，包含单个被试约50小时的数据。然而，尽管被试内规模的扩大能带来强大的解码性能，实用的BCI必须能够仅用几分钟（而非数小时）的数据就泛化到新用户。2026年PNPL竞赛以LibriBrain100（Mantegna等人，2026）应对这一挑战，这是一个扩展的LibriBrain数据集……

    arXiv:2609.03231v1 Announce Type: new  Abstract: The ambition of the 2025 PNPL competition (Landau et al., 2025) was to launch a multi-year curriculum for non-invasive speech decoding. Designed to progress from foundational tasks toward the linguistic complexity required for a practical brain-computer interface (BCI), it set the stage with speech detection and phoneme classification tasks. Winning submissions reached F1-macro scores of 95.6% and 73.6% on the respective tasks (Elvers et al., 2026), highly significant advances. This success was built on the LibriBrain dataset (\"{O}zdogan et al., 2025), the largest within-subject MEG dataset recorded at the time with ${\sim}50$ hours of data for one subject. However, while within-subject scale drives strong decoding performance, a practical BCI must generalise to new users from minutes of data, not hours.   The 2026 PNPL competition responds to this challenge with LibriBrain100 (Mantegna et al., 2026), an extended LibriBrain dataset with
    
[^111]: 语言编码的网络拓扑使大语言模型能够推理复杂网络

    Language-encoded network topology enables large language models to reason about complex networks

    [https://arxiv.org/abs/2609.03229](https://arxiv.org/abs/2609.03229)

    提出BioGlyph方法，将网络拓扑编译为可解释、可迁移的结构角色语言，从而使大语言模型能够对复杂网络进行结构推理。

    

    网络描述了生物学及其他领域的系统，从蛋白质相互作用和社会关系，到电网和引文记录。对这类系统进行推理需要理解其结构：哪些元素处于核心地位，哪些连接桥接了不同的社区，以及当元素被移除时系统会发生怎样的变化。尽管大语言模型（LLMs）擅长处理自然语言，但当网络以边列表、句子或测量表格的形式给出时，它们难以回答此类问题，因为网络的结构意义必须被推断出来。在这里，我们提出了BioGlyph，它将网络拓扑编译为一种可解释且可迁移的结构角色语言。BioGlyph结合图划分和结构测量来识别诸如枢纽节点、社区核心和跨社区连接器等角色，并通过固定规则将它们翻译成一种通用词汇。该表示通过其结构……来描述每个元素。

    arXiv:2609.03229v1 Announce Type: new  Abstract: Networks describe systems in biology and beyond, from protein interactions and social relationships to power grids and citation records. Reasoning about such systems requires understanding their structure: which elements are central, which connections bridge separate communities, and how it changes when elements are removed. Although large language models (LLMs) excel at natural language, they struggle with such questions when networks are given as edge lists, sentences or measurement tables, because their structural meaning must be inferred. Here we introduce BioGlyph, which compiles network topology into an interpretable and transferable language of structural roles. BioGlyph combines graph partitioning and structural measurements to identify roles such as hubs, community cores and cross-community connectors, and fixed rules to translate them into a universal vocabulary. The representation describes each element through its structural 
    
[^112]: 多步临床大语言模型智能体的反事实公平性审计需要测量每动作的不稳定性底线

    Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor

    [https://arxiv.org/abs/2609.03221](https://arxiv.org/abs/2609.03221)

    临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。

    

    反事实审计是检查临床智能体是否对人口统计学上不同但临床上相同的患者采取不同行动的标准工具。这类审计报告一个“翻转率”：当仅改变患者描述时，智能体行动发生改变的频率。我们证明这一指标本身是不可解释的。在16个病例情景上将完全相同的条件重复运行十次（相同叙述、相同描述字符串、不改变任何变量），临床智能体的行动在8.7%的结果-情景单元格中发生了改变，且不稳定性在不同行动之间呈现8倍的异质性，从ICU升级决策的0.022到受管制物质谨慎建议的0.179。我们数据中没有任何人口统计学对比能够与这一底线区分开。第二个模型给出了6.7%的合并底线，且对六种行动的不稳定性排序几乎完全一致（Spearman 0.94，精确p=0.017），说明该底线并非单一系统的特有产物。对五次抽样进行多数投票聚合可以消除其中39%的不稳定性……

    arXiv:2609.03221v1 Announce Type: new  Abstract: Counterfactual audits are the standard tool for checking whether a clinical agent treats demographically distinct but clinically identical patients differently. They report a flip rate: how often an action changes when only the patient descriptor changes. We show that this quantity is uninterpretable on its own. Re-running an identical condition ten times over sixteen vignettes (same narrative, same descriptor string, nothing varied) moved a clinical agent's action in 8.7% of outcome-vignette cells, and instability was heterogeneous across actions by a factor of eight, from 0.022 for ICU escalation to 0.179 for controlled-substance caution. No demographic contrast in our data was distinguishable from that floor. A second model gives a pooled floor of 6.7% and ranks the six actions almost identically (Spearman 0.94, exact p=0.017), so the floor is not one system's artefact. Majority-vote aggregation over five draws removes 39% of it and t
    
[^113]: SWIM：基于熟练度条件化生成的学生写作模拟

    SWIM: Student Writing Simulation via Proficiency-Conditioned Generation

    [https://arxiv.org/abs/2609.03215](https://arxiv.org/abs/2609.03215)

    该论文提出SWIM任务，将学生写作模拟形式化为基于熟练度条件的作文生成，实验发现提示方法对写作熟练度的控制有限，模型虽能调整内容导向特征，却难以重现词汇、语法和组织结构层面的真实学生写作差异。

    

    写作熟练度体现在学生如何发展内容、组织观点、选择词汇以及运用语言。尽管基于大语言模型（LLM）的学生模拟日益受到关注，但LLM能否在长篇写作中重现这种多维度的差异，在很大程度上仍未被探索。在本工作中，我们探索了语言模型能否真实地模拟学生写作，并提出了SWIM——一个将学生写作模拟形式化为熟练度条件化作文生成的任务。我们评估了提示方法、监督微调（SFT）和强化学习（RL）方法在写作模拟中的表现，并使用自动作文评分作为学生画像对齐程度的衡量标准。大量实验表明，提示方法所能提供的熟练度控制能力有限，即使是采用基于评分标准策略的强大专有LLM也是如此。特别值得注意的是，虽然模型能够调整内容导向的写作特征，但它们难以重现词汇、语法和组织结构层面的差异变化。

    arXiv:2609.03215v1 Announce Type: new  Abstract: Writing proficiency manifests in how students develop content, organize ideas, choose words, and use language. Despite growing interest in LLM-based student simulation, whether LLMs can reproduce such multidimensional variation in extended writing remains largely unexplored. In this work, we explore if language models can realistically simulate student writing, and introduce SWIM, a task that formulates Student Writing sIMulation as proficiency-conditioned essay generation. We evaluate prompting, supervised fine-tuning (SFT), and reinforcement learning (RL) methods for writing simulation using automated essay scoring as a measure of profile alignment. Extensive experiments reveal that prompting provides limited proficiency control, even for strong proprietary LLMs with rubric-grounded strategies. In particular, while models can adjust content-oriented traits, they struggle to reproduce the lexical, grammatical, and organizational variati
    
[^114]: 利用观测数据改进AI天气模型中的降水预报

    Improving precipitation forecasts in an AI weather model using observational data

    [https://arxiv.org/abs/2609.03210](https://arxiv.org/abs/2609.03210)

    通过使用观测的IMERG降水数据微调图变换器AI天气预报模型，将中期降水预报的CRPS提升最高19%，并使全球极端降雨预测的Brier技巧评分超越最先进业务化模型57%。

    

    人工智能天气预报（AIWP）系统在中期天气预报方面现已超越最先进的物理模型。当前的全球AIWP模型几乎完全使用单一的再分析数据集ERA5进行训练，但该数据集存在已知偏差，尤其是在降水方面。本文使用0.25°分辨率的IMERG降水数据对图变换器（graph-transformer）架构进行微调。所得模型将中期连续分级概率评分（CRPS）提高了多达19%，同时在热带风暴和毛毛雨事件中表现出更优的预报技能。在极端降雨预测方面，我们的模型的Brier技巧评分在全球范围内比最先进的业务化模型高出57%；不过，对于最强的降水事件，基于物理的业务化模型仍然更为可靠。我们的结果表明，将基于观测的降水数据直接纳入训练可以显著改进降水预报。

    arXiv:2609.03210v1 Announce Type: cross  Abstract: Artificial intelligence weather prediction (AIWP) systems now surpass state-of-the-art physical models for medium-range weather forecasting. Current global AIWP models are trained almost exclusively using one reanalysis dataset, ERA5, but it has known biases, particularly for precipitation. Here we fine-tune a graph-transformer architecture with IMERG precipitation data at 0.25{\deg} resolution. The resulting model improves medium-range continuous ranked probability scores by up to 19%, while also demonstrating superior skill for tropical storms and drizzle events. Our model exceeds the Brier skill score of state-of-the-art operational models on extreme rainfall prediction by 57% globally; however, a physics-based operational model remains more reliable for the heaviest precipitation events. Our results demonstrate that incorporating observations-based precipitation data directly into training can substantially improve precipitation fo
    
[^115]: VoxReason：合成前基于源记录的语音规划的无听者评估

    VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis

    [https://arxiv.org/abs/2609.03203](https://arxiv.org/abs/2609.03203)

    VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。

    

    表现力语音系统在任何波形被渲染之前就必须做出一个决定：一句话语将以何种方式被表达。在对话智能体、旁白叙述和角色条件TTS中，这一隐藏的规划步骤决定了情感、音高、能量、语速、停顿、重音和立场，然而下游音频评分很少能揭示这些选择是否由源记录所支持——这是一种在任何波形存在之前就发生的源使用失败。VoxReason将这一合成前的决策转化为可度量的、无需听者参与的任务，用于评估基于源记录的语音规划。在合成之前，VoxReason衡量话语表达方式的选择是否有被引用的源记录作为依据。系统输出带有证据引用的、注明来源的说话计划，随后一个确定性验证器检查引用合法性、槽位一致性、无支持状态、模式有效性以及单线索反事实局部性。在1,440个经过检查的源标签案例上，捷径控制实验表明了为什么仅凭槽位准确率是不安全的：一个简单的键值查找……（原文摘要在此处截断）

    arXiv:2609.03203v1 Announce Type: cross  Abstract: Expressive speech systems make a decision before any waveform is rendered: how an utterance is delivered. In dialogue agents, narration, and role-conditioned TTS, that hidden planning step sets affect, pitch, energy, rate, pause, emphasis, and stance, yet downstream audio scores rarely reveal whether those choices were licensed by the source record, a source-use failure that occurs before any waveform exists. VoxReason makes that pre-synthesis decision measurable as a listener-free task for source-grounded speech planning. Before synthesis, VoxReason measures whether delivery choices are grounded in cited source records. Systems output a source-cited speaking-plan with evidence citations, and a deterministic verifier checks citation legality, slot agreement, unsupported state, schema validity, and one-cue counterfactual locality. On 1,440 checked source-label cases, shortcut controls show why slot accuracy alone is unsafe: a key-lookup
    
[^116]: MemoryLACE：记忆生命周期感知的整合与证据检索

    MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval

    [https://arxiv.org/abs/2609.03201](https://arxiv.org/abs/2609.03201)

    提出了轻量级记忆框架MemoryLACE，通过稀疏的合并、取代和矛盾关系显式建模文本证据的生命周期，重建关系感知的证据单元以呈现当前、历史、支持和冲突证据，从而改进长期LLM智能体的记忆整合与证据检索。

    

    长期运行的LLM智能体必须在多次交互中保留信息，同时区分重复证据、历史状态、更新以及未解决的矛盾。现有的文本记忆系统能够高效检索语义相关的记忆，但这些关系往往是隐式的；而更丰富的结构化方法则通过全局图、层次抽象或反思机制来建模这些关系，但复杂度更高。我们提出了MemoryLACE（MemLACE），这是一个轻量级的记忆框架，通过稀疏的合并、取代和矛盾关系显式建模文本证据的生命周期，同时保留原子化的自然语言记忆及其来源。MemLACE并非独立检索记忆，而是重建具有关系感知的证据单元，为下游推理呈现当前、历史、支持和相互冲突的证据。在BEAM和StructMemEval基准上，使用开源权重和专有LLM……（摘要原文在此处截断）

    arXiv:2609.03201v1 Announce Type: new  Abstract: Long-term LLM agents must preserve information across interactions while distinguishing repeated evidence, historical states, updates, and unresolved contradictions. Existing textual memory systems retrieve semantically relevant memories efficiently but often leave these relationships implicit, whereas richer structured approaches model them through global graphs, hierarchical abstractions, or reflection at greater complexity. We introduce MemoryLACE (MemLACE), a lightweight memory framework that explicitly models the lifecycle of textual evidence through sparse merge, supersession, and contradiction relations while preserving atomic natural-language memories and their provenance. Rather than retrieving memories independently, MemLACE reconstructs relation-aware evidence units that expose current, historical, supporting, and conflicting evidence for downstream reasoning. Across BEAM and StructMemEval, using open-weight and proprietary LL
    
[^117]: 原子体系热力学景观的生成式嵌套采样

    Generative Nested Sampling of Atomistic Thermodynamic Landscapes

    [https://arxiv.org/abs/2609.03193](https://arxiv.org/abs/2609.03193)

    本文提出NS-Flows，利用单一条件归一化流替代马尔可夫链更新来加速原子体系热力学景观的嵌套采样，并通过对比揭示原子多模态（离散、组合性、硬碰撞壁分隔）与引力波后验（平滑简并、局域耦合）在结构上的根本差异。

    

    嵌套采样（NS）能够从单次模拟中解析原子系统的热力学，但其实际应用范围受限于马尔可夫链更新——在每个似然约束的系综内，需要对游走子进行去相关处理。基于流的嵌套采样方法已经在引力波（GW）推断中消除了这一瓶颈，但将其迁移到原子系统并非仅仅是应用领域的简单转换。通过将类GW150914的双黑洞似然函数与维度相当的八粒子二维Lennard-Jones（LJ）系统进行对比，我们证明这两种景观存在根本性的差异：原子体系的多模态是离散且组合性的，由被硬碰撞壁分隔的粒子置换所产生，其坐标耦合是稠密且集体性的；而GW后验则表现出平滑的简并性和局域化的参数耦合。基于这一诊断，我们提出了NS-Flows：一个单一的条件归一化流（摘要文本在此处被截断）

    arXiv:2609.03193v1 Announce Type: cross  Abstract: Nested sampling (NS) resolves the thermodynamics of an atomistic system from a single simulation, but its practical reach is limited by the Markov-chain updates needed to decorrelate walkers within each likelihood-constrained ensemble. Flow-based NS has removed this bottleneck for gravitational-wave (GW) inference, yet its transfer to atomistic systems is not merely a change of application. Comparing a GW150914-like binary-black-hole likelihood with an eight-particle two-dimensional Lennard-Jones (LJ) system of comparable dimensionality, we show that the two landscapes differ fundamentally: atomistic multimodality is discrete and combinatorial, generated by particle permutations separated by hard collision walls, and its coordinate coupling is dense and collective, whereas the GW posterior exhibits smooth degeneracies and localized parameter coupling. Guided by this diagnosis, we introduce NS-Flows: a single conditional normalizing flo
    
[^118]: 耦合张量-张量补全方法及其在药物重定位中的应用

    Coupled Tensor-Tensor Completion Method with Applications in Drug Repurposing

    [https://arxiv.org/abs/2609.03190](https://arxiv.org/abs/2609.03190)

    本文提出了一种名为耦合张量-张量补全（CTTC）的新框架，首次支持以张量形式融入辅助信息，通过挖掘多模态张量间的隐藏关联提升补全性能，并在药物重定位等生物医学问题上具有应用价值。

    

    许多生物医学挑战可以被表述为张量补全问题，即利用多维数组（张量）中已观测的条目来估算缺失值。在这类问题中，融入张量各模态的辅助信息（如基因-基因相似性）能够显著提升补全问题的求解效果。然而，大多数现有的张量补全方法只能以矩阵形式融入辅助信息。在本研究中，我们提出了一种新颖的框架，能够以张量形式融入辅助信息。我们的新方法称为耦合张量-张量补全（CTTC），它利用多模态张量之间的隐藏联系来提升张量补全性能。除了具有实用价值外，CTTC 在距离度量学习和群论方面还具有理论基础。我们推导了一种交替算法来求解 CTTC 优化问题，并证明了其收敛到一个稳定点。

    arXiv:2609.03190v1 Announce Type: cross  Abstract: Many biomedical challenges can be posed as tensor completion problems where the observed entries of a multidimensional array (a tensor) are used to impute the missing values. In such settings, incorporating side information about the modes of the tensor, such as gene-gene similarity, can significantly enhance the solutions of the completion problem. Most existing tensor completion methods can only incorporate side information in the form of matrices. In this study, we introduce a novel framework to incorporate side information in the form of tensors. Our new approach, called Coupled Tensor-Tensor Completion (CTTC), leverages the hidden connections among multimodal tensors to improve tensor completion performance. In addition to practical utility, CTTC has theoretical foundations in distance metric learning and group theory. We derive an alternating algorithm to solve the CTTC optimization problem and establish its convergence to a stat
    
[^119]: 跨合成数据生成器家族的可移植因果公平性

    Portable Causal Fairness Across Synthetic Data Generator Families

    [https://arxiv.org/abs/2609.03180](https://arxiv.org/abs/2609.03180)

    该论文证明了基于因果图边割的公平性机制可跨三个不相关家族共九个合成数据生成器（含差分隐私变体）普遍移植，并提出新的因果扩散模型骨干，在保真度接近边际分布方法的同时实现了所测家族中最公平的合成数据发布。

    

    当统计机构或监管机构发布合成数据以替代敏感记录时，它会选择生成该数据表的生成器，并可以塑造该生成器以消除不公平的路径。DECAF在一个非隐私保护的GAN上将这一想法具体化：三种公平性定义转化为生成器因果图上的三组边割。然而，该机制究竟是DECAF所独有，还是因果分解本身的属性，此前尚未得到验证。我们将这三种定义移植到来自三个不相关家族（基于边际分布的生成器、GAN和扩散模型，每个家族均包含差分隐私变体）的九个生成器上，涵盖三级形式隐私保证，在Adult和COMPAS数据集上进行了2,520次配对运行实验。结果表明该机制在各处均可迁移，且我们新的因果扩散模型骨干在我们测试的所有家族中产生了最公平的数据发布，同时保真度接近边际分布方法层级。应用边割几乎不改变保真度，仅为下游分类器带来约……

    arXiv:2609.03180v1 Announce Type: new  Abstract: When a statistical agency or regulator releases synthetic data in place of sensitive records, it chooses the generator that produces the table, and can shape that generator so unfair pathways are absent. DECAF made this concrete on one non-private GAN: three fairness definitions become three sets of edge cuts on the generator's causal graph. Whether the mechanism belongs to DECAF, or to causal factorisation itself, was untested. We port all three definitions to nine generators from three unrelated families (marginals-based, GAN, and diffusion, each with differentially private variants), across three levels of formal privacy guarantee, over 2,520 matched-pair runs on Adult and COMPAS datasets. The mechanism transfers everywhere, and our new causal diffusion backbone yields the fairest release of any family we tested, at fidelity close to the marginals tier. Applying the cut barely moves fidelity, only costs a downstream classifier about $
    
[^120]: 前沿大语言模型是有效的批量优化器：评估连续与离散环境中的推理模型

    Frontier LLMs are effective batch optimizers: Assessing reasoning models in continuous and discrete settings

    [https://arxiv.org/abs/2609.03177](https://arxiv.org/abs/2609.03177)

    前沿LLM在数值测试函数上是有竞争力的零样本批量优化器但性能相对脆弱，而在与预训练数据结构相似的语义丰富的离散空间中，其批量优化能力显著优于经典方法。

    

    前沿大语言模型（LLM）因其大规模预训练而能够应对多种优化场景，已成为优化领域中颇具吸引力的先验。然而，现代推理型LLM在批量优化场景中的有效性仍未得到充分探索。本研究调查了当前一代前沿LLM在连续和离散环境中作为批量优化器的表现。我们发现，虽然LLM在数值测试函数上是具有竞争力的零样本批量优化器，但与经典的非LLM优化方法相比，其性能较为脆弱。然而，LLM先验在语义丰富的环境中表现显著更佳，这表明它们的批量优化行为在与预训练数据结构最相似的离散空间中进行导航和推理时非常有效。

    arXiv:2609.03177v1 Announce Type: new  Abstract: Frontier large language models (LLMs) have become attractive priors for optimization due to their large-scale pretraining that enables them to navigate a variety of optimization settings. However, the effectiveness of modern reasoning LLMs in batch optimization settings remains underexplored. Here we investigate the performance of the current generation of frontier LLMs as batch optimizers in both continuous and discrete settings. We find that while LLMs are competitive zero-shot batch optimizers for numerical test functions, their performance is brittle compared to classical non-LLM optimization approaches. However, LLM priors are significantly better in semantically rich settings, indicating that their batch optimization behavior is highly effective when navigating and reasoning over the discrete spaces most similar in structure to their pretraining data.
    
[^121]: 谁为被剪枝的Token发声？将视觉Token剪枝视为覆盖率优化问题

    Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization

    [https://arxiv.org/abs/2609.03158](https://arxiv.org/abs/2609.03158)

    提出CoverPruner，一种无需训练的视觉token剪枝方法，将剪枝创新性地建模为表示覆盖最大化问题，确保每个被剪枝的token都有存活的token为其代言，尤其在激进压缩下取得最佳准确率。

    

    视觉token剪枝可以降低视觉语言模型（VLM）的推理成本，但大多数方法只关注应该保留哪些token。这种基于保留token的视角可能会保留冗余的高分token，同时使被丢弃的信息缺乏近似的代表性token。我们提出了CoverPruner，一种无需训练的剪枝方法，它从互补的需求侧提出问题：当一个token被移除后，哪个存活的原始token能够为目标VLM代表它？CoverPruner将剪枝表述为表示覆盖最大化（RCM）问题，以查询加权的需求覆盖完整的投影视觉token集合。该方法通过投影器空间覆盖和轻量级第一层注意力探针来实现RCM。在多个VLM架构和压缩率下，CoverPruner在所有对比方法中取得了最佳平均准确率，且最大的性能提升通常出现在激进压缩的情况下。

    arXiv:2609.03158v1 Announce Type: cross  Abstract: Visual token pruning reduces the inference cost of vision-language models (VLMs), but most methods only ask which tokens to keep. This retained-token view can keep redundant high-scoring tokens while leaving discarded evidence without a close representative. We propose CoverPruner, a training-free pruner that asks the complementary demand-side question: after a token is removed, which surviving original token represents it for the target VLM? CoverPruner formulates pruning as Representational Coverage Maximization (RCM), covering the full projected visual-token set with query-weighted demand. It instantiates RCM with projector-space coverage and a lightweight first-layer attention probe. Across multiple VLM architectures and compression rates, CoverPruner achieves the best average accuracy among all compared methods, with the largest gains usually appearing under aggressive compression.
    
[^122]: BASP：面向大语言模型训练的通信高效批感知序列并行

    BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training

    [https://arxiv.org/abs/2609.03151](https://arxiv.org/abs/2609.03151)

    提出批感知序列并行BASP，根据微批大小将GPU划分为互不相交的序列并行组以缩小all-to-all通信组规模，实现通信局部化，从而显著降低大语言模型长序列训练的通信开销并提升训练效率。

    

    大语言模型（LLM）的长上下文推理正变得越来越重要，但由于巨大的内存和通信需求，长序列训练仍然充满挑战。序列并行已成为解决长序列LLM训练瓶颈的关键技术。然而，我们观察到现有的序列并行方法是批无关的，即在所有批大小下都采用统一的序列划分方式，导致通信效率低下。在本文中，我们提出了批感知序列并行，这是一种利用批结构来减少通信开销的序列并行方法。BASP通过根据微批大小将GPU划分为互不相交的序列并行组来利用批结构。这种设计减小了all-to-all通信组的规模，从而实现通信的局部化并提升训练效率。在……上的实验结果表明……

    arXiv:2609.03151v1 Announce Type: cross  Abstract: Long-context reasoning for large language models (LLMs) is becoming increasingly important, but training over long sequences remains challenging due to massive memory and communication requirements. Sequence parallelism has emerged as an essential technique for addressing bottlenecks in long sequence LLM training. However, we observe that existing sequence parallelism methods are batch-agnostic and apply uniform sequence partitioning across all batch sizes, resulting in inefficient communication. In this paper, we introduce Batch- Aware Sequence Parallelism (BASP), a sequence parallelism approach that leverages batch structure to reduce communication overhead. BASP exploits batch structure by partitioning GPUs into disjoint sequence-parallel groups according to the micro- batch size. This design reduces the all-to-all communication group size, thereby localizing communication and improving training efficiency. Experimental results on a
    
[^123]: 路由是不够的：诊断 MoE+LoRA 微调中适配器内部子空间的争用问题

    Routing Is Not Enough: Diagnosing Intra-Adapter Subspace Contention in MoE+LoRA Fine-Tuning

    [https://arxiv.org/abs/2609.03150](https://arxiv.org/abs/2609.03150)

    该研究发现在 MoE+LoRA 多领域微调中，即使专家路由近乎完全分离，负迁移仍由近乎正交的领域梯度在同一低秩适配器子空间内竞争所致，并提出 SpawnLoRA，通过在检测到适配器争用时于 MoE 专家内部动态添加门控子适配器（保持路由器固定）来化解这一问题。

    

    多领域微调通常将 MoE 路由与 LoRA 相结合，并假设 token 级别的路由能够分离各领域特有的更新。我们使用 Python 代码与生物医学文本和数学推理配对的数据，在 MoE+LoRA 中检验了这一假设。尽管这些领域表现出近乎不相交的专家路由，但加入生物医学数据后代码的困惑度显著上升，这表明仅靠路由分离可能无法阻止负迁移。为了定位这一失败，我们引入了 Jaccard 路由重叠度和适配器梯度余弦相似度两个诊断指标，分别用于衡量专家共享程度和更新兼容性。这些诊断结果表明，干扰主要源自几乎正交的领域梯度在同一低秩适配器子空间内的竞争。我们通过 SpawnLoRA 来解决这一问题：当检测到适配器层面的争用时，SpawnLoRA 会在 MoE 专家内部动态添加带门控的子适配器，同时保持路由器固定不变。

    arXiv:2609.03150v1 Announce Type: cross  Abstract: Multi-domain fine-tuning often combines MoE routing with LoRA, assuming that token-level routing separates domain-specific updates. We test this assumption in MoE+LoRA using Python code paired with biomedical text and mathematical reasoning. Although these domains show near-disjoint expert routing, adding biomedical data substantially increases code perplexity, indicating that routing separation alone may not prevent negative transfer. To localize the failure, we introduce Jaccard routing overlap and adapter-gradient cosine similarity, which measure expert sharing and update compatibility, respectively. These diagnostics indicate that interference arises mostly from nearly orthogonal domain gradients competing within the same low-rank adapter subspace. We address this issue with SpawnLoRA, which dynamically adds gated sub-adapters inside MoE experts when adapter-level contention is detected, while keeping the router fixed. We evaluate 
    
[^124]: RACE-AIMC：面向边缘端异构模拟内存计算加速器的选择性推理

    RACE-AIMC: Selective Inference for Heterogeneous Analog In-Memory Accelerators at the Edge

    [https://arxiv.org/abs/2609.03149](https://arxiv.org/abs/2609.03149)

    RACE-AIMC提出了一种基于统计学的选择性推理框架，通过离线分析多个异构的模拟内存计算加速器芯片，以风险感知的认证集成方式在“运行全部芯片合并结果”与“信任单一芯片”之间做出最优选择，从而在边缘设备上兼顾能耗效率与推理可靠性。

    

    模拟内存计算（AIMC）通过直接在存储阵列内部执行算术运算来加速神经网络推理，而无需在内存和处理器之间来回搬运权重。这节省了能量，但存储权重的物理器件并不完美：编程误差、电噪声、有限分辨率的转换器以及完全损坏的单元都会使计算产生失真，而且每个物理芯片的失真方式各不相同。拥有多个此类芯片的设计者面临一个两难选择：运行所有芯片并合并结果（安全但浪费能量），或者盲目信任单个芯片（便宜但无法保证其出错频率）。本文提出了RACE-AIMC（面向AIMC的风险感知认证集成框架），用统计学而非猜测来解决这一选择。离线阶段，RACE-AIMC研究一组物理加速器，为给定的能量……（原文摘要在此处截断）

    arXiv:2609.03149v1 Announce Type: cross  Abstract: Analog in-memory computing (AIMC) speeds up neural-network inference by doing the arithmetic directly inside a memory array, instead of shuttling weights back and forth between memory and a processor. This saves energy, but the physical devices that store the weights are imperfect: programming errors, electrical noise, limited-resolution converters, and outright broken cells all distort the computation, and every physical chip is distorted in its own way. A designer with several such chips available faces an uncomfortable choice: run all of them and combine the answers (safe, but wasteful of energy), or trust a single chip blindly (cheap, but with no guarantee on how often it is wrong). This paper introduces RACE-AIMC (Risk-Aware Certified Ensemble for AIMC), a framework that resolves this choice with statistics rather than guesswork. Offline, RACE-AIMC studies a pool of physical accelerators, picks the single best one for a given ener
    
[^125]: 可行但不安全：学习型无蜂窝ISAC关联中的约束违反与报告信道攻击

    Feasible but Not Safe: Constraint Violations and Report-Channel Attacks in Learned Cell-Free ISAC Association

    [https://arxiv.org/abs/2609.03147](https://arxiv.org/abs/2609.03147)

    本文揭示了基于GNN的无蜂窝ISAC关联调度器虽然预测精度很高，但仍会违反硬约束导致解不可行，并证明通过将输出投影到可行解（如简单的贪心修复）能以极低的效用损失恢复约束满足。

    

    基于学习的调度器已被提出，用于在分布式无蜂窝通感一体化（ISAC）系统中提供实时的用户、目标和接入点（AP）关联。在一种典型方法中，图神经网络（GNN）以混合整数线性规划生成的标签进行训练，通过一次前向传播将轻量级的逐AP统计数据映射为AP聚类、用户与目标调度以及模式选择的决策。此类解决方案假设硬约束（仅在训练阶段以软惩罚形式加以约束）在推理时仍然成立，并且假设自报告的统计数据真实可信。以我们的ASSENT算法为例，我们发现尽管F₁分数很高，许多解仍违反至少一个硬约束，这表明高预测精度并不能保证联合可行性。将GNN输出投影到可行解上即可恢复约束满足，且效用损失较低，即使采用简单的贪心修复过程也能实现。

    arXiv:2609.03147v1 Announce Type: cross  Abstract: Learning-based schedulers have been proposed to provide real-time user, target, and access point (AP) association in distributed cell-free integrated sensing and communication systems. In a typical approach, a graph neural network (GNN), trained on labels from a mixed-integer linear program, maps lightweight per-AP statistics to decisions on AP clustering, user and target scheduling, and mode selection in one forward pass. Such solutions assume that hard constraints, enforced only as soft training penalties, hold at inference, and that the self-reported statistics are truthful. Using our ASSENT algorithm as an example, we find that despite high $F_1$ scores, many solutions violate at least one hard constraint, demonstrating that high prediction accuracy does not ensure joint feasibility. Projecting the GNN output onto a feasible solution restores constraint satisfaction with low utility loss, even with a simple greedy repair procedure.
    
[^126]: 感知哪种模态重要：面向鲁棒VLA策略的证据门控正则化

    Sensing Which Modality Matters: Evidence-Gated Regularization for Robust VLA Policies

    [https://arxiv.org/abs/2609.03142](https://arxiv.org/abs/2609.03142)

    本文揭示了VLA策略中的“模态纠缠”问题，并提出证据门控正则化（EGR）——一种零推理开销的训练目标，通过对低证据传感器施加不变性约束、对高证据传感器施加单传感器充分性约束，使多模态策略在遮挡和干扰下更加鲁棒。

    

    视觉-语言-动作（VLA）策略融合多模态传感输入，但在有限且同质化的机器人演示数据上训练，会促使模型学习传感器之间的虚假相关性而非任务相关信号，我们将这种失败模式称为“模态纠缠”。在真实世界的遮挡和干扰物作用下，这种问题表现为：对无信息量传感器受损产生过度敏感的干扰敏感性，以及当仅剩一个有信息量的传感器完好时出现单模态不足。我们提出证据门控正则化（EGR），这是一种与模态无关的训练目标，且不引入任何推理时的额外开销。EGR推导出逐帧、逐传感器的任务相关性信号，以此门控两个状态条件一致性目标：对低证据传感器施加不变性约束，对高证据传感器施加单传感器充分性约束。我们基于BEHAVIOR-1K构建了一个基准，包含一个仅推理的快速诊断套件和47个针对模态纠缠的基于回放的技能任务。

    arXiv:2609.03142v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) policies fuse multimodal sensory inputs, but training on limited and homogeneous robot demonstrations encourages spurious inter-sensor correlations rather than task-relevant signal, a failure we term modality entanglement. Under real-world occlusions and distractors, this manifests as nuisance sensitivity to corruption of uninformative sensors and single-modality insufficiency when only one informative sensor remains intact. We propose Evidence-Gated Regularization (EGR), a modality-agnostic training objective that introduces zero inference-time overhead. EGR derives a per-frame and per-sensor task-relevance signal to gate two state-conditional consistency objectives: invariance on low-evidence sensors, and single-sensor sufficiency on high-evidence ones. We introduce a benchmark based on BEHAVIOR-1K, comprising a fast inference-only diagnostic suite and 47 rollout-based skills targeting modality entangleme
    
[^127]: 度量空间上一致Lipschitz回归的闭式公式及其稀疏神经网络实现

    A Closed-Form Formula for Consistent Lipschitz Regression on Metric Spaces with Sparse Neural Network Realizations

    [https://arxiv.org/abs/2609.03129](https://arxiv.org/abs/2609.03129)

    本文提出一个简单的闭式“两阶段”复合公式，用于从含噪观测中重构度量空间上的Lipschitz函数，并给出同时控制逼近与统计误差、且优化误差为零的高概率一致恢复保证。

    

    许多经典的机器学习方法，如核岭回归（KRR）和支持向量回归（SVR），在计算和分析上都是易于处理的，因为它们的估计器要么具有闭式表达式，要么通过最小化凸训练目标获得；而这两个特性对于深度神经网络通常都不可用。为了解决这一问题，我们引入了一个简单的闭式“两阶段”复合公式 $\hat{f}$，用于从 $N$ 个独立同分布的含噪观测中重构度量空间 $(\mathcal X,\rho)$ 上的未知 Lipschitz 函数 $f:\mathcal{X}\to \mathbb{R}$。我们的主要结果是一个高概率一致（$L^{\infty}$）恢复保证，该保证同时控制逼近误差和统计误差，且优化误差为零；特别地，我们不需要假设能够访问近似经验风险最小化（ERM）的预言机。其次，我们的次要主要结果在三个互补的意义上确立了我们公式的最优性：1）函数空间：在 Ahlfors 正则度量……（摘要在此处被截断）

    arXiv:2609.03129v1 Announce Type: cross  Abstract: Several classical machine-learning methods, such as KRRs and SVRs, are both computationally and analytically tractable since their estimators either admit closed-form expressions or are obtained by minimizing convex training objectives; neither feature is generally available for deep neural networks. We address this by introducing a simple closed-form ``two-stage'' compositional formula $\hat{f}$ for reconstructing an unknown Lipschitz function $f:\mathcal{X}\to \mathbb{R}$ on a metric space $(\mathcal X,\rho)$ from $N$ i.i.d. noisy observations.   Our main result is a high-probability uniform ($L^{\infty}$) recovery guarantee that jointly controls approximation and statistical errors while enjoying an optimization error of zero; in particular, we do not assume oracle access to an approximate ERM. Our secondary main results establish the optimality of our formula in three complementary senses. 1) Function space: On Ahlfors-regular metr
    
[^128]: 核重启：突破神经切线核在神经场应用中的边界

    Kernel Reboot: Breaking the Boundaries of Neural Tangent Kernels for Neural Fields

    [https://arxiv.org/abs/2609.03117](https://arxiv.org/abs/2609.03117)

    本文提出 NTK-KIP、MetaQuill 和 MetaQuill-KIP 三种算法，通过蒸馏支持集的非线性核回归与元学习共享初始化相结合，突破了神经切线核的线性局限，实现了从稀疏观测中快速高质量地重建神经场并积累可复用的任务先验。

    

    神经场（NF）将连续坐标映射为颜色或密度等信号，但从稀疏观测中进行快速且高质量的重建仍然困难。经典的神经切线核（NTK）回归可以给出闭式解拟合，但其本质是线性的，无法积累可复用的任务先验。我们开发了三种算法来弥补这些不足。NTK-KIP 通过学习一个蒸馏得到的坐标支持集（以及可选的标签），使有限维 NTK 能够从少量观测数据中修复大面积缺失区域，从而获得紧凑的非线性表示，而非原始的核求解。MetaQuill 为 INR 元学习一个共享初始化，使新场景只需更新一个很小的任务特定权重偏移即可完成适配，从而实现真正的特征学习和可复用的先验。最后，MetaQuill-KIP 融合了两种思想：先以 KIP 式的非线性热启动为任务提供初始解，然后仅围绕该小偏移进行精炼优化。

    arXiv:2609.03117v1 Announce Type: new  Abstract: Neural fields (NFs) map continuous coordinates to signals such as color or density, but fast high-quality reconstruction from sparse observations remains difficult. Classical Neural Tangent Kernel (NTK) regression gives closed-form fits, yet it is fundamentally linear and cannot accumulate reusable task priors. We develop three algorithms that address these gaps. NTK-KIP learns a distilled support set of coordinates (and optional labels) so that a finite NTK can inpaint large missing regions from little observed data, yielding a compact non-linear representation instead of a raw kernel solve. MetaQuill meta-learns a shared initialization for an INR so that new scenes can be adapted by updating only a small task-specific weight offset, which provides true feature learning and a reusable prior. Finally, MetaQuill-KIP fuses both ideas: it seeds the task with a KIP-style non-linear warm start, then refines only that small offset around the m
    
[^129]: CRAW：编解码器鲁棒的音频水印

    CRAW: Codec Robust Audio Watermarking

    [https://arxiv.org/abs/2609.03107](https://arxiv.org/abs/2609.03107)

    CRAW 提出了一种对神经编解码器鲁棒的音频水印框架，通过失真感知训练、注意力池化、推理时感知掩蔽与纠错码，在抵御神经再合成攻击的同时保持高感知音质。

    

    生成式语音模型的最新进展使得真实音频与合成音频之间的区分越来越困难，从而催生了新型欺诈与虚假信息。音频水印提供了一种有前景的防御手段，它通过在生成的语音中嵌入不可感知的信号，之后可检测该信号以验证音频来源。然而，近期研究表明，现有的后置水印方法在神经编解码器和去噪器下会失效，而这些变换在现实世界的存储、传输和处理过程中被广泛使用，严重限制了此类方法的实际效用。本文提出了 CRAW，一种对编解码器鲁棒的音频水印框架，能够在提高对神经再合成攻击鲁棒性的同时保持高感知质量。CRAW 将失真感知训练与基于注意力的池化机制、推理时的感知掩蔽以及纠错码相结合，以恢复鲁棒性训练过程中损失的保真度。

    arXiv:2609.03107v1 Announce Type: cross  Abstract: Recent advances in generative speech models have made it increasingly difficult to distinguish authentic from synthetic audio, enabling new forms of fraud and misinformation. Audio watermarking offers a promising defense by embedding an imperceptible signal into generated speech that can later be detected to verify its provenance. However, recent studies have shown that existing post-hoc watermarking methods fail under neural codecs and denoisers, transformations routinely applied during real-world storage, transmission, and processing, severely limiting their practical utility. Here we introduce CRAW, a codec-robust audio watermarking framework that jointly improves robustness against neural re-synthesis while maintaining high perceptual quality. CRAW combines distortion-aware training with an attention-based pooling mechanism, inference-time perceptual mask- ing, and an error-correcting code to recover the fidelity lost during robust
    
[^130]: 缩放定律、表格数据与精算定价模型

    Scaling Laws, Tabular Data and Actuarial Ratemaking Models

    [https://arxiv.org/abs/2609.03106](https://arxiv.org/abs/2609.03106)

    该研究首次在精算定价这一表格数据场景中检验了深度学习缩放定律的适用性，发现各模型家族均随数据量增加而性能提升，但TabM的数据缩放能力显著优于表格Transformer和MLP基线，而Transformer的参数缩放能力较弱。

    

    现代深度学习中的缩放定律描述了随着模型容量、训练数据和计算资源的增加，模型在留出集上的损失如何改善，通常遵循幂律趋势。我们研究了在精算定价领域是否也存在类似的缩放规律——该领域的数据是表格化、异构且含噪声的，而广义线性模型（GLM）等经典模型仍然是强有力的基线。基于一个真实世界的汽车保险组合，我们在不断增加的训练数据比例和多个随机种子下训练来自不同家族的模型，并以样本外Poisson偏差作为评估指标（这是一种基于似然的Poisson计数预测损失，数值越低表示留出集拟合效果越好）。我们发现所有模型家族都随数据量的增加而提升，但缩放指数存在显著差异：TabM表现出明显强于纯监督表格Transformer和标准MLP基线的数据缩放能力。除非（译者注：摘要原文在此处截断）……

    arXiv:2609.03106v1 Announce Type: new  Abstract: Scaling laws in modern deep learning describe how held-out loss improves as model capacity, training data, and compute increase, often following power-law trends. We investigate whether analogous scaling regularities arise in actuarial ratemaking, where data are tabular, heterogeneous, and noisy, and where classical models such as GLMs remain strong baselines. Using a real-world motor insurance portfolio, we train models from different families across increasing fractions of the training data and multiple random seeds, evaluating out-of-sample Poisson deviance, a likelihood-based loss for Poisson count predictions in which lower values indicate better held-out fit. We find that all model families improve with additional data, but scaling exponents differ substantially: TabM exhibits markedly stronger data scaling than purely supervised tabular Transformers and standard MLP baselines. Transformer variants show weak parameter scaling unles
    
[^131]: 基于占位问题的分位数风险控制

    Occupancy-based Quantile Risk Control

    [https://arxiv.org/abs/2609.03104](https://arxiv.org/abs/2609.03104)

    提出了一种基于占位问题的分位数风险控制方法（OQRC），通过有序校准损失划分损失空间并估计测试损失分布，实现了具有有限样本有效性的紧致风险控制界。

    

    保形风险控制是一种新兴框架，用于在具有有限样本保证的前提下安全部署机器学习模型。为了适应更广泛的风险概念，分位数风险控制将该框架扩展到了基于分位数的风险度量。然而，现有方法要么过于保守，要么缺乏严格的有限样本保证。为了解决这些局限性，我们提出了基于占位问题的分位数风险控制，这是一种新颖的方法，能够提供紧致的风险控制界并具有有限样本有效性。我们的核心思想是通过利用有序的校准损失对损失空间进行划分，将风险控制问题形式化为一个有限占位问题。具体而言，我们估计测试损失在所划分出的各个分区上的分布，并用每个分区内的最大损失来对风险进行上界约束。随后，我们选择参数 λ，使得该上界不超过预先定义的阈值 α。

    arXiv:2609.03104v1 Announce Type: cross  Abstract: Conformal risk control is an emerging framework for the safe deployment of machine learning models with finite-sample guarantees. To accommodate a broader class of risk notions, quantile risk control extends this framework to quantile-based risk measures. However, existing methods either suffer from excessive conservatism or lack rigorous finite-sample guarantees. To address these limitations, we introduce Occupancy-based Quantile Risk Control (OQRC), a novel method that provides tight risk control bounds with finite-sample validity. Our key idea is to formulate risk control as a finite-occupancy problem by partitioning the loss space with the ordered calibration losses. Specifically, we estimate the distribution of test losses across the resulting bins and upper-bound the risk by the maximum loss attained within each bin. We then select the parameter $\lambda$ such that this upper bound does not exceed a predefined threshold $\alpha$ 
    
[^132]: 蒸馏深度光流立体方法以反演密集三维风场

    Distilling deep optical flow stereo methods to retrieve dense three-dimensional wind fields

    [https://arxiv.org/abs/2609.03100](https://arxiv.org/abs/2609.03100)

    本文提出用深度光流替代立体匹配中基于窗口的跟踪方法，并结合自监督几何残差损失与探空数据有监督微调，实现了高效且精确的密集三维风场反演，同时减少了对数值天气预报和多卫星重叠观测的依赖。

    

    静止轨道大气运动矢量（AMV）提供密集的水平风矢量和高度信息，供数据同化系统使用。传统的AMV采用基于窗口的互相关方法跟踪特征，并通过将红外亮温与数值天气预报（NWP）背景场配对来估计高度，这形成了一种循环依赖，导致高度估计不准确、计算成本高昂且反演结果稀疏。来自GEO-GEO（静止轨道-静止轨道）和GEO-LEO（静止轨道-低轨）的立体风通过不同观测姿态间的视差偏移从几何上解析高度，消除了对NWP的依赖并提高了精度，但其计算量大且覆盖范围有限。在本工作中，我们用深度光流替代立体匹配中基于窗口的跟踪，以实现高效且性能更优的反演。微调过程平衡了自监督几何残差损失与有监督的探空数据重建。为消除对多卫星重叠观测的需求……（摘要在此处截断）

    arXiv:2609.03100v1 Announce Type: new  Abstract: Geostationary atmospheric motion vectors (AMVs) provide the dense horizontal wind vectors (u,v) and heights ingested into data assimilation systems. Traditional AMVs track features using window-based cross-correlation and estimate heights via infrared brightness temperatures paired with numerical weather prediction (NWP) background states, creating a circular dependency that yields inaccurate heights, high computational cost, and sparse retrievals. Stereo winds from GEO-GEO and GEO-LEO geometrically resolve heights from parallax shifts across different poses, eliminating NWP dependence and improving accuracy, but they remain computationally heavy with limited coverage. In this work, we replace window-based tracking in stereo matching with deep optical flow for efficient, improved retrieval. Fine-tuning balances a self-supervised geometric residual loss with supervised radiosonde reconstruction. To eliminate multi-satellite overlap requir
    
[^133]: 超越模糊：基于皮肤微纹理的语义三视图远程皮肤病学可分级性评估流水线

    Beyond Blur: A Semantic Tri-view Pipeline for Teledermatology Gradability via Skin Micro-relief

    [https://arxiv.org/abs/2609.03095](https://arxiv.org/abs/2609.03095)

    该论文提出了一种可解释的语义三视图流水线，将皮肤表皮微纹理作为可计算的图像质量生物标志物，通过轻量级分割模型与多视图聚合分类器相结合，实现自动化且鲁棒的远程皮肤病病例可分级性筛查。

    

    智能手机皮肤照片对于远程皮肤病学至关重要，然而评估提交病例的诊断适用性（可分级性）仍然是移动医疗工作流程中的关键瓶颈。皮肤科医生通常会审阅多张照片视图（区域视图、角度视图和特写视图）来识别一致的纹理细节，而不是仅依赖单张图像。我们提出了语义三视图流水线，这是一种可解释的自动化远程皮肤病可分级性筛查架构，它将表皮微纹理形式化为可计算的图像质量生物标志物。使用公开SCIN数据集的专家标注子集，我们训练了一个轻量级DeepLabV3+模型来分割微纹理保真度。随后，这些空间掩码通过逻辑回归分类器在最多三个病例视图上进行聚合，利用视点冗余来支持在不受控智能手机采集条件下的鲁棒性。该方法学习上下文感知……

    arXiv:2609.03095v1 Announce Type: cross  Abstract: Smartphone skin photographs are indispensable to teledermatology, yet assessing the diagnostic suitability of submitted cases (gradability) remains a critical bottleneck in mobile care workflows. Dermatologists routinely review multiple photographic views (regional, angled, and close-up) to identify consistent textural detail rather than relying on a single image. We present the Semantic Tri-view Pipeline, an interpretable architecture for automated teledermatology gradability screening that formalizes epidermal micro-relief as a computable biomarker of image quality. Using an expert-annotated subset of the public SCIN dataset, we train a lightweight DeepLabV3+ model to segment micro-relief fidelity. These spatial masks are then aggregated across up to three case views with a logistic regression classifier, leveraging viewpoint redundancy to support robustness under uncontrolled smartphone acquisition. This approach learns context-awar
    
[^134]: 梯度看不见秩：Matrix-CODI在ProsQA上的秩无关性

    The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA

    [https://arxiv.org/abs/2609.03090](https://arxiv.org/abs/2609.03090)

    研究通过秩-k投影消融实验发现，矩阵值连续思维链模型（Matrix-CODI）中潜在矩阵的有效秩与任务准确率无关——低秩截断几乎不影响性能且训练损失不偏好任何特定秩，表明矩阵潜在表示并未通过高秩结构编码并行推理路径。

    

    连续思维链模型将推理压缩至潜在token中。矩阵值变体让每个潜在token通过一个d×d矩阵瓶颈进行路由，从而将秩引入为潜在矩阵Z上的单样本结构可观测量。如果矩阵潜在表示通过叠加携带并行推理路径，那么秩应当能追踪这些路径，并且将Z截断至低秩应当会损害那些求解可能需要多个组件的任务的准确率。在Matrix-CODI模型的四种训练方案中（三种在ProsQA上，一种在学习阈值以下的GSM8K-Aug上），秩-k投影消融曲线的平坦度在0.6个百分点以内。三次随机种子重复实验得到81.0 ± 2.0个百分点的准确率，而Z的最终有效秩分布在{4, 12, 13}之间；损失函数并不偏好任何特定秩。为检验秩盲是否仅源于“展平后投影”的读出方式，我们训练了四种读出方式，包括双线性重参数化等。

    arXiv:2609.03090v1 Announce Type: new  Abstract: Continuous chain-of-thought models compress reasoning into latent tokens. Matrix-valued variants, which route each latent token through a d x d matrix bottleneck, introduce rank as a single-sample structural observable on the latent matrix Z. If matrix latents carry parallel reasoning paths via superposition, rank should track them, and truncating Z to low rank should hurt accuracy on tasks whose solutions plausibly require multiple components. Across four training regimes of a matrix-CODI model (three on ProsQA, one on GSM8K-Aug below the learning threshold), the rank-k projection ablation curve is flat to within 0.6 percentage points. A three-seed replication yields 81.0 +/- 2.0 percentage points accuracy while the final effective rank of Z spans {4, 12, 13}; the loss does not reward any particular rank. To test whether rank-blindness arises from the flatten-then-project readout alone, we trained four readouts: a bilinear reparametriza
    
[^135]: LeanStream：一种用于高效端侧大语言模型推理的“推测-精化”流式框架

    LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference

    [https://arxiv.org/abs/2609.03079](https://arxiv.org/abs/2609.03079)

    LeanStream通过利用部分GPU计算结果渐进精化计算、加载和缓存保留优先级，实现了GPU执行与存储I/O的细粒度重叠，突破了端侧LLM推理中准确决策与计算-I/O重叠难以兼得的根本性权衡。

    

    端侧大语言模型（LLM）推理因其在隐私和响应速度方面的优势而具有吸引力，但由于模型权重远超可用DRAM容量，在移动和嵌入式设备上实现仍然极具挑战性。先前的系统利用激活稀疏性并将权重卸载到SSD或闪存，但面临一个根本性的系统权衡：准确的稀疏执行决策需要最新的上下文信息，而高效的计算-I/O重叠则需要提前预测。因此，现有设计要么对执行进行串行化处理，要么导致冗余的权重获取、额外的计算以及巨大的缓存开销。我们提出了LeanStream，一个面向高效端侧LLM推理的流式“推测-精化”框架。LeanStream利用部分GPU计算结果渐进地精化计算、加载和缓存保留的优先级，从而实现GPU执行与存储I/O之间的细粒度重叠。我们在移动和嵌入式平台上均实现了LeanStream。与先前的端侧方案相比……

    arXiv:2609.03079v1 Announce Type: new  Abstract: On-device LLM inference is attractive for privacy and responsiveness, but remains challenging on mobile and embedded devices because model weights far exceed available DRAM. Prior systems exploit activation sparsity and offload weights to SSD or flash storage, but face a fundamental systems trade-off: accurate sparse execution decisions require the latest context, whereas efficient computation-I/O overlap requires early prediction. As a result, existing designs either serialize execution or incur redundant weight fetches, extra computation, and large cache overheads. We present LeanStream, a streaming speculate-and-refine framework for efficient on-device LLM inference. LeanStream progressively refines computation, loading, and cache-retention priorities using partial GPU results, enabling fine-grained overlap between GPU execution and storage I/O. We implement LeanStream on both mobile and embedded platforms. Compared with prior on-devi
    
[^136]: 立场论文：视觉学习中“无标签”不等于“无人类监督”

    Position: Unlabeled IS NOT Equal to No Human Supervision in Visual Learning

    [https://arxiv.org/abs/2609.03077](https://arxiv.org/abs/2609.03077)

    该立场论文指出，无标签数据并不等于没有人类监督——不同的数据整理方式与训练目标蕴含着不同的人类先验，单一的“无监督”术语已无法反映这些差异，研究者应更明确地识别监督的真正来源。

    

    这篇立场论文认为，标签的缺失并不意味着视觉学习中人类监督的缺失，并呼吁研究界更明确地识别监督的来源。计算机视觉领域的许多近期方法都建立在从大规模无标签数据中学习到的表征之上，因此被归入“无监督”这一总括性术语之下。然而，不同的数据整理方案和训练目标嵌入了本质不同、且为模型所依赖的人类先验知识；我们认为，单一的“无监督”总括术语已经无法捕捉这些区别。这种模糊性使得在不同假设下开展的无监督学习研究难以相互比较，与此同时，尽管该领域持续增长，自2021年以来顶级计算机视觉会议中以“无监督”为标题的论文数量急剧下降。尽管我们完全支持将预训练作为……（原文摘要在此处截断）。

    arXiv:2609.03077v1 Announce Type: cross  Abstract: This position paper argues that the absence of labels does not imply the absence of human supervision in visual learning, and urges the research community to identify sources of supervision more explicitly. Many recent methods in computer vision build upon representations learned from large-scale unlabeled data, and are therefore grouped under the same umbrella term ``unsupervised.'' However, different data curation schemes and training objectives embed substantially different human priors on which models rely, and we argue that one ``unsupervised'' umbrella term is no longer capturing these distinctions. This ambiguity makes it harder to compare unsupervised learning research conducted under different assumptions, coinciding with a sharp decline in papers titled with ``unsupervised'' in flagship computer vision conferences since 2021, despite continued growth of the field. While we fully embrace pre-training as a strong foundation for
    
[^137]: 神经算子的可学习组合方法

    Learnable composition for neural operators

    [https://arxiv.org/abs/2609.03069](https://arxiv.org/abs/2609.03069)

    该论文提出LatentDDM方法，通过预训练预测小子域的神经算子、再仅训练一个轻量级组合模块来迁移到新场景，从而大幅降低将神经算子适配到新几何、尺寸或运行条件时所需的高保真模拟成本。

    

    神经算子是物理模拟中快速且可微分的代理模型，但当求解域的几何形状、尺寸或运行条件与训练时不一致时，其精度往往会下降。有监督的微调可以恢复精度，但即使是较小的目标数据集也需要昂贵的高保真模拟。因此，我们研究如何将预训练与迁移学习协同设计，以降低部署成本。LatentDDM首先预训练一个神经算子来预测小尺度子域上的物理场；面对新设定时，该方法冻结此算子，仅训练一个轻量级模块来组合各局部预测结果。我们在两个互补的问题上评估了该方法：稳态达西流问题，其中长程压力耦合必须跨越越来越大的多孔介质域；以及绕俯仰翼型的非定常不可压缩流动，当目标俯仰频率超出训练范围时，滚动推演误差会不断累积放大。与容量匹配的m（摘要原文在此处被截断）

    arXiv:2609.03069v1 Announce Type: new  Abstract: Neural operators are fast, differentiable surrogates for physical simulation, but their accuracy often degrades when domain geometry, size, or operating conditions differ from training. Supervised adaptation can recover accuracy, but even a small target set requires costly high-fidelity simulations. We therefore ask how pretraining and transfer can be designed together to reduce this deployment cost. LatentDDM first pretrains a neural operator to predict fields on small subdomains. For a new setting, it freezes this operator and trains only a lightweight module that composes the local predictions. We evaluate our method on two complementary problems: steady Darcy flow, where long-range pressure coupling must extend across increasingly large porous domains, and unsteady incompressible flow around a pitching airfoil, where rollout errors compound as target pitching frequencies exceed the training range. Compared with the capacity-matched m
    
[^138]: 具有拜占庭鲁棒聚合的差分隐私联邦学习：面向银行与医疗系统安全模型训练的跨领域框架

    Differentially private federated learning with Byzantine-robust aggregation: A cross-domain framework for secure model training in banking and healthcare systems

    [https://arxiv.org/abs/2609.03064](https://arxiv.org/abs/2609.03064)

    本文提出DP-BR-FedAvg框架，将高斯机制差分隐私与坐标级截断均值拜占庭鲁棒聚合相结合，同时防御梯度泄露攻击和恶意客户端的对抗性更新，为银行和医疗等受监管领域的安全联邦模型训练提供了跨领域解决方案。

    

    联邦学习使银行、医院和其他受监管机构能够在不将原始记录移出自身服务器的情况下训练共享模型，这在数据保护法律或竞争敏感性导致无法集中汇总数据的场景中极具吸引力。然而，有两个问题限制了这一前景在实践中被信任的程度。首先，客户端之间交换的参数更新仍会通过梯度反演和成员推断攻击泄露本地记录的信息。其次，诸如FedAvg这类诚实的平均聚合规则无法防御一部分提交损坏或对抗性更新的客户端，因此少量恶意或被入侵的参与者可以悄然使共享模型偏离正轨。本文提出了一个联邦学习框架DP-BR-FedAvg，它将高斯机制差分隐私层与坐标级截断均值拜占庭鲁棒聚合规则相结合，并在模拟环境中进行评估。

    arXiv:2609.03064v1 Announce Type: cross  Abstract: Federated learning allows banks, hospitals, and other regulated organizations to train a shared model without moving raw records off their own servers, which is attractive wherever data protection law or competitive sensitivity rules out pooling data centrally. Two problems limit how far this promise can be trusted in practice. First, the parameter updates that clients exchange still leak information about local records through gradient inversion and membership inference attacks. Second, an honest averaging rule such as FedAvg has no defense against a subset of clients that submit corrupted or adversarial updates, so a small number of malicious or compromised participants can quietly steer the shared model off course. This paper presents a federated learning framework, DP-BR-FedAvg, that combines a Gaussian-mechanism differential privacy layer with a coordinate-wise trimmed-mean Byzantine-robust aggregation rule, evaluated on a simulat
    
[^139]: IDSPACE：一种用于数字身份验证系统可靠评估的新型文档生成器 [扩展技术报告]

    IDSPACE: A Novel Document Generator for Reliable Evaluation of Digital Identity Verification Systems [Extended Technical Report]

    [https://arxiv.org/abs/2609.03052](https://arxiv.org/abs/2609.03052)

    本文提出IDSpace——一种新型身份证件合成文档生成器，通过模型引导的贝叶斯优化仅需少量目标域样本即可自适应调优生成参数，并将用户元数据与自动控制参数解耦，为数字身份验证系统的可靠评估提供高质量合成数据。

    

    随着服务向线上迁移，银行、贷款机构和政府等信任机构必须验证远程用户的身份。欺诈检测工具虽然广泛可用，但由于身份证件具有敏感性而难以获取，对其进行评估和微调仍然困难。合成数据生成提供了一条可行的前进路径，且需求十分明显：我们此前在该领域的工作已被下载超过11,000次（从八个部分汇总统计）。我们提出IDSpace，在三个方向上扩展了这一研究。首先，我们提出模型引导的贝叶斯优化方法，仅需目标域的少量样本，即可调整生成参数，从而最大化与目标域模型的视觉相似性和预测一致性。其次，我们将用户指定的元数据（人口统计信息、欺诈模式、采集设备）与自动调整的控制参数（字体样式、噪声水平、图像质量）解耦，允许用户……（摘要原文在此处被截断）

    arXiv:2609.03052v1 Announce Type: cross  Abstract: As services move online, trust institutions such as banks, lenders, and governments must verify the identity of remote users. Fraud detection tools are widely available, but evaluating and fine-tuning them remains difficult because identity documents are sensitive and therefore scarce. Synthetic data generation offers a path forward, and demand is clear: our prior work in this area has been downloaded over $11{,}000$ times (aggregated from eight parts). We introduce IDSpace, extending this line of research in three directions. First, we propose model-guided Bayesian optimization, which tunes generation parameters to maximize both visual similarity and prediction consistency with target-domain models given only a few samples from a target domain. Second, we decouple user-specified metadata (demographics, fraud patterns, capture device) from automatically tuned control parameters (font styles, noise levels, image quality), allowing users
    
[^140]: 机器学习在定向进化中的进展：五年回顾

    Advances in Machine Learning for Directed Evolution: A Five-Year Retrospective

    [https://arxiv.org/abs/2609.03046](https://arxiv.org/abs/2609.03046)

    本文回顾指出，机器学习辅助定向进化（MLDE）未能像其他蛋白质工程领域那样取得变革性进展，其主因是研究者追求“最优蛋白质”的目标与定向进化在时间和资源约束下寻找“足够好蛋白质”的实际需求脱节，例如忽视DNA合成成本导致方法缺乏实际适用性。

    

    过去五年多来，机器学习（ML）的进展已经改变了许多蛋白质工程学科，但定向进化却并非如此。基于对一篇先前共同撰写的观点文章的反思，我讨论了为什么我认为情况如此，并认为机器学习辅助定向进化（MLDE）研究者的目标——“识别最优蛋白质”——与更广义的定向进化目标——“在时间和资源限制下识别足够好的蛋白质”——之间的脱节是主要原因。作为一个例子，我强调了几乎所有当前的MLDE方法都忽略了DNA合成的成本，导致这些策略无论其底层模型的能力如何，实际适用性都很有限。最后，我讨论了近期一些打破这一总体趋势的工作，并强调了过去五年机器学习辅助蛋白质工程研究的努力（摘要内容不完整，原文在此处被截断）。

    arXiv:2609.03046v1 Announce Type: cross  Abstract: The last five-plus years have seen many protein engineering disciplines transformed by advances in machine learning (ML), but the same cannot be said for directed evolution. Reflecting on a previously co-authored perspective, I discuss why I believe this to be the case, arguing that a disconnect between the goals of machine-learning-assisted directed evolution (MLDE) researchers--"identify an optimal protein"--and the goals of directed evolution more broadly--"identify a sufficient protein given time and resource constraints"--is a principal culprit. As an example, I highlight how nearly all current MLDE methods neglect to account for the cost of DNA synthesis, resulting in strategies that have limited practical applicability regardless of the underlying models' capabilities. I close by discussing recent works that are exceptions to this overarching trend, and emphasize that the last five years of efforts in ML-assisted protein enginee
    
[^141]: 8.35亿地址规模下的总体校准图筛查及向新区块链的无标签迁移

    Population-Calibrated Graph Screening at 835-Million-Address Scale, with Label-Free Transfer to New Chains

    [https://arxiv.org/abs/2609.03036](https://arxiv.org/abs/2609.03036)

    该论文提出了一个已部署的大规模区块链合规筛查系统，在覆盖五条EVM链、8.35亿地址和158亿条边的统一交易图上，利用带每链归一化的归纳式图编码器对地址评分，通过总体分数分布的精确分位数设定阈值使警报量可预先确定，并实现了无需目标链标签即可向新链迁移的高召回筛查能力。

    

    区块链地址的合规筛查在实践中通常是对制裁名单的查找加上聚类启发式方法；这种方法在无标签地址以及完全没有标签覆盖的链上会失效。我们描述了一个已部署的系统，该系统根据地址在多链交易图中的位置而非其是否出现在名单上来对地址进行评分。其底层是一个跨越五条EVM链的单一图，包含835,330,427个地址和15,826,261,934条边；一个采用每链归一化的共享归纳式编码器为两个评分头提供输入。决策阈值设定为整个地址总体分数分布的精确分位数，并按链段逐一扫描，因此警报数量可以预先确定。我们报告的结果包括：无标签迁移：仅在两条链上训练的评分头，在10^{-3}的总体警报率下，对Base、Arbitrum和Gnosis三条链上留出正样本的召回率分别达到0.8598 / 0.8182 / 0.9967，且评分头训练中未使用任何目标链的标签；静态提前期回放（摘要在此处被截断）

    arXiv:2609.03036v1 Announce Type: cross  Abstract: Compliance screening of blockchain addresses is, in practice, a lookup against sanctions registries plus clustering heuristics; it fails on unlabelled addresses and on chains with no label coverage at all. We describe a deployed system that scores an address by its position in a multi-chain transaction graph rather than by its presence in a list. The substrate is a single graph of 835,330,427 addresses and 15,826,261,934 edges across five EVM chains; a shared inductive encoder with per-chain normalisation feeds two scoring heads. Decision thresholds are exact quantiles of the score distribution over the full population, scanned per chain segment, so the alert volume is known in advance. We report: label-free transfer: heads trained on two chains recall 0.8598 / 0.8182 / 0.9967 of held-out positives on Base, Arbitrum and Gnosis at a $10^{-3}$ population alert rate, with no target-chain labels in head training; a static lead-time replay 
    
[^142]: 你无法逃离自己的激活：评估意识与多智能体监控

    You Can't Escape Your Own Activations : Evaluation Awareness and Multi-Agent Monitoring

    [https://arxiv.org/abs/2609.03035](https://arxiv.org/abs/2609.03035)

    该研究首次系统考察了当LLM智能体被明确告知其内部激活正被监控（并可收到监控器反馈）时，即“评估意识”状态下，基于激活探针的多智能体勾结检测效果如何变化。

    

    LLM智能体越来越多地被部署在多智能体系统中，它们可能在保持行为表面正常的同时进行勾结。旨在检测此类勾结行为的输出监控器可能会被混淆和隐写术所欺骗，这促使人们使用在内部激活上训练的探针。然而，这些探针通常是在不知道自己正被监控的智能体上进行评估的。我们研究了当智能体被明确告知其内部激活正被监控时，以及当它们额外收到来自监控器的反馈时，基于激活的检测会发生怎样的变化。我们保持模型、探针和阈值固定不变，仅改变告知智能体的信息：什么都不告知（基线）、告知存在激活监控器（知晓条件）、或告知存在监控器并同时提供上一轮的得分（反馈条件）。我们在两个游戏中进行测试：一个四智能体二十一点游戏和一个双智能体Simmons囚徒游戏，使用Qwen3-32B-AWQ和GPT-OSS-20B模型。

    arXiv:2609.03035v1 Announce Type: cross  Abstract: LLM agents are increasingly deployed in multi-agent systems, where they can collude while keeping their actions benign. Output monitors designed to detect such collusions can be fooled by obfuscation and steganography, motivating the use of probes trained on internal activations. However, these probes are usually evaluated on agents that do not know they are being watched. We study how activation-based detection changes when agents are explicitly informed that their internal activations are being monitored, and when they additionally receive feedback from the monitor. We keep the models, probes, and thresholds fixed and change only what the agents are told: nothing (baseline), that an activation monitor is present (aware), or that a monitor is present together with the previous round's score (feedback). We test two games, a four-agent blackjack game and a two-agent Simmons prisoners game, using Qwen3-32B-AWQ and GPT-OSS-20B in homogene
    
[^143]: ObserverBench：面向干预与控制的机制性估计测试

    ObserverBench: Testing Mechanistic Estimates for Intervention and Control

    [https://arxiv.org/abs/2609.03026](https://arxiv.org/abs/2609.03026)

    提出 ObserverBench 基准框架，将估计精度与所选干预行动造成的损失分开评估，用以检验机制可解释性中的内部估计器是否足以胜任干预、控制与安全任务，并证明平均准确的估计并不必然带来更优的行动。

    

    机制可解释性正越来越多地被用于指导诸如激活引导、电路移除和安全监控等干预措施。然而，一个平均意义上准确的内部估计，仍可能选出糟糕的行动。我们提出了 ObserverBench，一个用于检验内部估计器（即“观察者”）是否足以胜任其所指导的干预、控制或安全任务的基准框架。每个任务固定模型、信息边界、允许的行动、决策规则、留出测试用例和损失函数。该基准将估计精度与所选行动造成的损失分开报告。理论与实验表明为何两者都必不可少：在闭环控制中，观察者误差在起始点以及允许的干预所能到达的方向上都会产生影响。在 GPT-2-small 和 Qwen2.5-7B 的电路干预任务上，成对观察者能更准确地预测未见过的效应，但并不总是能选出更好的行动。

    arXiv:2609.03026v1 Announce Type: cross  Abstract: Mechanistic interpretability is increasingly used to guide interventions such as activation steering, circuit removal, and safety monitoring. Yet an internal estimate that is accurate on average can still choose a poor action.   We present ObserverBench, a benchmark framework for testing whether an internal estimator---an observer---is adequate for the intervention, control, or safety task it directs. Each task fixes the model, information boundary, allowed actions, decision rule, held-out cases, and loss. The benchmark reports estimation accuracy separately from the loss caused by the chosen action.   Theory and experiments show why both are needed. In closed-loop control, observer errors matter at the starting point and along directions the allowed intervention can reach. On circuit-intervention tasks in GPT-2-small and Qwen2.5-7B, pairwise observers predict unseen effects more accurately without always choosing better actions; obser
    
[^144]: 基于上下文集成的共形语言任务统一框架

    Unifying Conformal Language Tasks with In-Context Ensembles

    [https://arxiv.org/abs/2609.03005](https://arxiv.org/abs/2609.03005)

    提出共形相关性框架，通过上下文学习示例筛选与集成自动构建评分函数，以最少的人工干预统一实现了多种NLP任务中覆盖率与简洁性的双重保证。

    

    许多自然语言处理任务，例如摘要生成和抽取式问答，都可以归结为在两个约束条件下从文档中检索相关内容：覆盖率（保留足够的相关信息以实现某个目标）和简洁性（尽可能去除无关信息）。共形预测方法已被用于保证覆盖率，但需要通过设计评分函数来优化简洁性。最先进的评分函数使用手工设计的LLM提示词来让模型评估内容的重要性，但手动提示工程既费力又依赖于具体任务。我们提出了共形相关性框架，该框架利用上下文学习示例筛选与集成来构建评分函数，在保持覆盖率的同时提高简洁性，且只需最少的人工干预。我们演示了该框架在七个NLP任务上的应用，并从理论上研究了多样性的影响。

    arXiv:2609.03005v1 Announce Type: new  Abstract: Many NLP tasks, such as summarization and extractive question answering, reduce to retrieving relevant content from documents under two constraints: coverage, retaining enough pertinent information to achieve some goal, and conciseness, removing as much irrelevant information as possible. Conformal prediction methods have been used to guarantee coverage, and must be optimized for conciseness through design of a score function. State-of-the-art scoring functions use hand-engineered LLM prompts asking the model to rate the importance of content, but manual prompt engineering is labor-intensive and task-specific. We introduce the Conformal Relevance framework which uses in-context learning example curation and ensembling to create a score function which maintains coverage while improving conciseness with minimal manual input. We demonstrate this framework's application on seven NLP tasks, and also theoretically study the impact of diversity
    
[^145]: 因果基础模型

    Causal Foundation Models

    [https://arxiv.org/abs/2609.03003](https://arxiv.org/abs/2609.03003)

    本文介绍了因果基础模型（CFMs）这一新兴范式：通过预训练的神经网络，利用上下文学习即可在全新数据集上估计平均处理效应等因果量，无需为每个新问题定制流程或更新模型。

    

    因果推断是通过数据估计治疗或干预效果的实践。传统上，每个新问题都需要定制的流程：首先提出因果机制，然后选择兼容的估计器，最后进行训练。与此同时，在多样化的设置和模态中，机器学习的很大一部分已转向基础模型的范式：网络只需在大规模数据上预训练一次，即可无需微调地应用于新任务。因果基础模型（CFMs）将这一范式引入因果推断领域。CFMs是预训练的神经网络，能够使用上下文学习在全新的数据集上估计因果量（例如平均处理效应），而无需更新模型。这项工作为这一新兴领域提供了实用性的介绍。在讨论CFMs之前，我们总结了因果推断和机器学习的必要背景。全文中，我们提供了示例代码和Jupyter笔记本。

    arXiv:2609.03003v1 Announce Type: new  Abstract: Causal inference is the practice of estimating the effect of a treatment or intervention from data. It traditionally requires a bespoke pipeline for every new problem: first proposing a causal mechanism, selecting a compatible estimator, and finally training it. Meanwhile, across diverse settings and modalities, much of machine learning has shifted to the paradigm of foundation models: networks pretrained once at scale and applied to new tasks without fine-tuning. Causal foundation models (CFMs) bring this paradigm to causal inference. CFMs are pretrained neural networks that estimate causal quantities, such as the average treatment effect, on entirely new datasets using in-context learning without requiring model updates. This work provides a practical introduction to this emerging area. We summarize the necessary background in causal inference and machine learning before discussing CFMs. Throughout, we include example code and Jupyter 
    
[^146]: 蒸馏之前先验证：面向在线策略蒸馏的提示级教师门控

    Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation

    [https://arxiv.org/abs/2609.02998](https://arxiv.org/abs/2609.02998)

    该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。

    

    在线策略蒸馏（OPD）通过在学生模型自身的生成结果上提供来自冻结教师模型的密集token级监督来加速后训练过程。原始的OPD在所有提示上均匀地应用这种监督，而不检查教师模型对每个提示是否可靠。由于反向KL散度具有模式寻求特性，一个自信但错误的教师模型可能导致强烈却具有误导性的更新。分布性代理指标（如熵或教师-学生似然一致性）只能衡量不确定性或一致性，但无法直接验证结果的正确性。我们提出了教师门控在线策略蒸馏（TGOPD），其核心原则是在接受密集监督之前，应在提示级别验证教师模型的可靠性。TGOPD通过一小组经验证器评分的教师探测样本估计教师可靠性，并将每个提示专门路由到密集OPD（当可靠性检查通过时）或基于验证器的GRPO（当检查不通过时）。在4B和3...（摘要内容不完整）

    arXiv:2609.02998v1 Announce Type: cross  Abstract: On-policy distillation (OPD) accelerates post-training by providing dense token-level supervision from a frozen teacher on the student's own rollouts. Vanilla OPD applies this supervision uniformly across prompts, without checking whether the teacher is reliable for each prompt. Because reverse KL is mode-seeking, a confidently wrong teacher can induce a strong yet misleading update. Distributional proxies, such as entropy or teacher-student likelihood agreement, measure uncertainty or agreement but do not directly verify outcome correctness. We introduce Teacher-Gated On-Policy Distillation (TGOPD), built on the principle that teacher reliability should be verified at the prompt level before dense supervision is admitted. TGOPD estimates reliability from a small set of verifier-scored teacher probes and routes each prompt exclusively to dense OPD when the reliability check passes or to verifier-grounded GRPO otherwise. Across 4B and 3
    
[^147]: 评估图神经网络在海事航海图变更重要性分类中的应用

    Evaluating Graph Neural Networks for Change-Criticality Classification in Maritime Navigation Charts

    [https://arxiv.org/abs/2609.02996](https://arxiv.org/abs/2609.02996)

    该论文提出将电子航海图数据集表示为图结构（空间对象为节点、空间与语义关系为边），并将新旧航海图间变更的重要性分类构建为图对分类问题，以评估不同图神经网络配置在此任务上的表现。

    

    图神经网络（GNN）是一类适用于图结构数据学习的神经网络。将其应用于空间数据是一种自然的延伸，然而，对于哪种消息传递操作、架构配置和图表示最适合用于对电子航海图（ENC，即用于海洋导航的地理空间矢量数据集）中对象变更进行分类，目前仍不明确。维护这些数据集是一项挑战，而根据对象对航行安全的重要性对ENC中对象的变更进行分类尤为重要。在此，我们提出将这些矢量导航数据集表示为图结构，其中空间对象作为节点，它们之间的空间和语义关系形成边。我们将旧的ENC数据集和新的ENC数据集分别编码为一对图，并将该任务构建为一个图对分类问题。

    arXiv:2609.02996v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) are a class of neural networks suitable for learning on graph-structured data. Their application to spatial data is a natural extension, however its relatively unclear which message-passing operations, architectural configurations, and graph representation is best suited for classifying changes to objects in electronic navigational charts (ENCs)--geospatial vector datasets used for marine navigation. Maintaining these datasets is a challenge, and categorizing changes to objects in the ENC based on their significance to navigational safety is of particular importance. Here, we propose to represent these vector navigation datasets as a graph structure where the spatial objects serve as nodes and their spatial and semantic relationships form edges. We encode both the old ENC dataset and new ENC dataset into a pair of graphs and frame the task as a graph-pair classification problem. Building on this representat
    
[^148]: 具有有限库输入扭曲核的无遗憾贝叶斯优化

    No-Regret Bayesian Optimization with Finite-Library Input-Warped Kernels

    [https://arxiv.org/abs/2609.02993](https://arxiv.org/abs/2609.02993)

    本文提出FLIWBO方法，通过从有限光滑输入映射库中自适应选择输入扭曲来加速学习，在打破固定核函数限制的同时仍保持高概率无遗憾收敛保证，其代价仅为与库大小相关的显式 $\sqrt{N_\varepsilon}$ 项。

    

    高斯过程贝叶斯优化（GP-BO）在昂贵函数的黑盒优化中表现出色，例如超参数优化（HPO）和多智能体系统（MAS）设计。某些方法已有收敛速率保证，特别是GP上置信界（GP-UCB），但其要求使用固定的核函数。关键在于，核函数编码了输入的接近程度如何影响目标值的相似性。当原始坐标与这种几何结构匹配不佳时——例如对数缩放的超参数或局部峰值——输入扭曲可以大幅提升样本效率，然而已知的GP-UCB证明要求固定的核函数。我们提出了有限库输入扭曲贝叶斯优化（FLIWBO），它通过任意依赖历史的规则从有限的光滑输入映射库中选择扭曲。该方法通过自适应调整输入几何结构来加速学习，同时在温和的假设下保持高概率收敛保证，并具有明确的 $\sqrt{N_\varepsilon}$ 库大小代价。

    arXiv:2609.02993v1 Announce Type: new  Abstract: Gaussian-process Bayesian optimization (GP-BO) excels at black-box optimization of costly functions, e.g., hyperparameter optimization (HPO) and multi-agent system (MAS) design. Convergence-rate guarantees exist for select methods, notably GP upper confidence bound (GP-UCB), but require a fixed kernel. Critically, the kernel encodes how input proximity affects objective value similarity. When raw coordinates poorly match this geometry - as with log-scaled hyperparameters or localized peaks - input warping can greatly improve sample efficiency, yet known GP-UCB proofs require a fixed kernel. We propose Finite-Library Input-Warped Bayesian Optimization (FLIWBO), which selects warps from a finite library of smooth input maps by any history-dependent rule. It adapts the input geometry to accelerate learning while retaining high-probability convergence guarantees under mild hypotheses, with an explicit $\sqrt(N_\varepsilon)$ library-size cost
    
[^149]: TRACE：用于颗粒动力学的时空接触记忆图网络模拟器

    TRACE: Spatiotemporal Contact Memory Graph Network Simulator for Granular Dynamics

    [https://arxiv.org/abs/2609.02991](https://arxiv.org/abs/2609.02991)

    TRACE是一种图网络模拟器，通过将交互历史直接存储在接触边上的持久记忆中，并结合物理结构化解码器来预测颗粒间接触力，有效解决了颗粒动力学中接触历史难以保留的问题。

    

    学习型图模拟器为颗粒动力学的高保真求解器提供了一种高效的替代方案。然而，颗粒运动在很大程度上依赖于颗粒间的接触历史，而当颗粒接触形成、断裂和重新排列时，这些接触历史难以保留。现有的模拟器主要将时间信息存储在节点特征或节点级记忆中。在此，我们提出TRACE，一种直接在接触边上存储交互历史的图网络模拟器。每条边维护一个持久记忆，通过基于注意力的消息传递和门控循环单元进行更新，同时一个边身份词典在接触图发生变化时保留这些记忆。物理结构化的解码器预测颗粒间的法向和切向接触力，强制执行库仑摩擦极限，并施加大小相等、方向相反的内部力。该模型先经过单步预训练，随后进行自回归滚动微调。我们评估

    arXiv:2609.02991v1 Announce Type: new  Abstract: Learned graph simulators provide an efficient alternative to high-fidelity solvers for granular dynamics. However, granular motion depends strongly on inter-granular contact history, which is difficult to preserve when particle contacts form, break, and rearrange. Existing simulators mainly store temporal information in node features or node-level memory. Here we introduce TRACE, a graph-network simulator that stores interaction history directly on contact edges. Each edge maintains a persistent memory updated by attention-based message passing and a gated recurrent unit, while an edge-identity dictionary preserves this memory as the contact graph changes. A physics-structured decoder predicts inter-granular normal and tangential contact forces, enforces the Coulomb friction limit, and applies equal-and-opposite internal forces. The model is trained with single-step pretraining followed by autoregressive rollout fine-tuning. We evaluate 
    
[^150]: 面向TCAD在环设计空间探索的网格原生物理信息图神经网络代理模型

    Mesh-Native Physics-Informed Graph Surrogates for TCAD-in-the-Loop Design Space Exploration

    [https://arxiv.org/abs/2609.02988](https://arxiv.org/abs/2609.02988)

    本文提出一种直接在四面体TCAD网格上运行的物理信息图注意力网络代理模型，在每个网格节点预测静电势及电子/空穴准费米能级等漂移-扩散系统基本未知量，并通过有限体积电流连续性残差将载流子输运物理嵌入训练目标，从而实现高效的TCAD在环多目标设计空间探索。

    

    漂移-扩散输运的高保真TCAD仿真仍然是新兴FinFET器件设计的主力工具，但其计算成本高昂，尤其是对于三维结构，运行时间会随网格复杂度急剧攀升，这严重限制了多目标设计空间探索。现有的机器学习代理模型将一组固定的设计参数映射到少数几个标量器件指标上，丢弃了底层物理信息，并丧失了跨器件几何结构和器件家族的可迁移性。本文提出了一种物理信息图注意力网络（GAT）代理模型，它直接在四面体TCAD网格上运行，在每个网格节点上预测静电势以及电子和空穴的准费米能级——即漂移-扩散系统的基本未知量。训练过程将数据损失与有限体积电流连续性残差相结合，把载流子输运物理嵌入到优化目标中。（摘要在此处被截断）

    arXiv:2609.02988v1 Announce Type: new  Abstract: High-fidelity TCAD simulation of drift-diffusion transport remains the workhorse of emerging FinFET device design, but it is computationally expensive, especially for 3D structures where runtime escalates steeply with mesh complexity. This sharply limits multi-objective design space exploration. Existing machine-learning surrogates map a fixed set of design parameters to a few scalar device metrics, discarding the underlying physics and losing transferability across device geometries and families. A physics-informed graph attention network (GAT) surrogate is proposed. It operates directly on the tetrahedral TCAD mesh and predicts, at every mesh node, the electrostatic potential together with the electron and hole quasi-Fermi levels, the fundamental unknowns of the drift-diffusion system. Training combines a data loss with finite-volume current-continuity residuals, embedding carrier-transport physics into the objective. Operating on the 
    
[^151]: 尾部似然强化学习

    Tail-Likelihood Reinforcement Learning

    [https://arxiv.org/abs/2609.02987](https://arxiv.org/abs/2609.02987)

    提出TailRL方法，通过最大化策略超过随机选择的奖励阈值的对数概率来直接优化对高奖励结果的覆盖能力，使罕见的高奖励输出在梯度中获得更大权重，从而解决平均奖励优化无法衡量生成式策略产生稀有高奖励输出概率差异的问题。

    

    强化学习通常优化平均奖励。对于生成式策略而言，平均值可能掩盖一个重要的区别：两个策略可以达到相同的平均奖励，但在产生罕见但高奖励的输出方面的机会却截然不同。随着训练和推理过程中采样数量的增加，这一点变得尤为重要，因为采样的收益取决于策略是否在高奖励结果上保留概率质量。我们提出直接优化这种覆盖能力。我们不仅考虑期望奖励，还考虑其所有的上尾部分：对于每个奖励阈值，策略超过该阈值的可能性有多大？这将连续的奖励转化为一族二元成功事件。我们提出了尾部似然强化学习，该方法最大化超过随机选择的奖励阈值的对数概率。其梯度对罕见的高奖励输出赋予更大的权重，并且可以解释为Best-of-(k)梯度的混合。

    arXiv:2609.02987v1 Announce Type: new  Abstract: Reinforcement learning typically optimizes average reward. For generative policies, the average can hide an important distinction: two policies can achieve the same mean reward while having very different chances of producing a rare but high-reward rollout. This matters as sampling increases during training and inference, since its benefit depends on retaining probability mass on high-reward outcomes. We propose to optimize this coverage directly. Rather than considering only expected reward, we consider all of its upper tails: for each reward threshold, how likely is the policy to exceed it? This turns a continuous reward into a family of binary success events. We introduce Tail-Likelihood Reinforcement Learning (TailRL), which maximizes the log-probability of exceeding a randomly chosen reward threshold. Its gradient gives more weight to rare, high-reward rollouts and can be interpreted as a mixture of Best-of-(k) gradients. TailRL req
    
[^152]: 现代Transformer是隐式混合体：从功能分化到有原则的混合架构设计

    Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design

    [https://arxiv.org/abs/2609.02986](https://arxiv.org/abs/2609.02986)

    本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。

    

    结合全注意力（FA）和线性注意力（LA）的混合架构日益受到关注，但其注意力分配方式仍然依赖于启发式规则。我们在基于RoPE的Transformer所学习到的头级别功能组织中寻求一个有证据支撑的基础。行为探针无法给出完整的分类体系，因此我们提出了两个干预指标：RoPE频率重要性分数（RFIS），衡量每个频率如何影响某个头的注意力分布；以及RoPE位置依赖性（RPD），用于分离注意力头对旋转位置调制的依赖程度。在Qwen3系列模型和Llama3.1上，RFIS给出假设且RPD加以验证，形成了一个完整的分类体系，将由显著中低频带分隔的检索头和位置头区分开来。受控Transformer实验表明，这一边界遵循训练长度的位置尺度；我们将其命名为全局位置频带。该分析揭示了零样本长度外推失败的潜在原因，并产生了

    arXiv:2609.02986v1 Announce Type: new  Abstract: Hybrid architectures combining Full Attention (FA) and Linear Attention (LA) are increasingly prominent, yet their allocation remains heuristic. We seek an evidence-grounded basis in head-level functional organization learned by RoPE-based Transformers. Behavioral probes do not yield a complete taxonomy, so we propose two intervention metrics: RoPE Frequency Importance Score (RFIS), measuring how each frequency affects a head's attention distribution, and RoPE Positional Dependence (RPD), isolating dependence on rotary positional modulation. On Qwen3-series models and Llama3.1, RFIS suggests and RPD verifies a complete taxonomy of retrieval and positional heads separated by a salient mid-low-frequency band. Controlled Transformers show that this boundary follows the training-length positional scale; we term it the Global Positional Band (GPBand). The analysis suggests a potential cause of zero-shot length-extrapolation failure and yields
    
[^153]: 从欧几里得数据到图结构数据：协同学习综述

    From Euclidean to Graph-Structured Data: A Survey of Collaborative Learning

    [https://arxiv.org/abs/2609.02984](https://arxiv.org/abs/2609.02984)

    这是一篇综述论文，系统梳理了协同学习（包括联邦学习和去中心化学习）如何从规则网格结构的欧几里得数据扩展到图结构数据，指出基于消息传递机制的图学习与多智能体需要交换信息的协同环境在概念上天然契合。

    

    传统的机器学习方法，即在单一位置收集数据、训练模型并执行推理，面临着包括可扩展性和隐私在内的根本性限制，这制约了其适用性。为了应对这些挑战，近期研究探索了协同学习方法，包括联邦学习和去中心化学习，其中各个智能体在本地执行训练和推理，并进行有限的协作。大多数协同学习研究专注于具有规则网格状结构的欧几里得数据（例如图像、文本）。然而，这些方法无法捕捉许多现实世界应用中的关系模式，而这类模式最适合用图结构来表示。图上的学习依赖于消息传递机制在相连节点之间传播信息，这使其在概念上非常适合智能体必须交换信息的协同环境。（注：原文摘要在此处不完整，后续内容被截断）

    arXiv:2609.02984v1 Announce Type: new  Abstract: The conventional approach to machine learning, that is, collecting data, training models, and performing inference in a single location, faces fundamental limitations, including scalability and privacy, that restrict its applicability. To address these challenges, recent research has explored collaborative learning approaches, including federated learning and decentralized learning, where individual agents perform training and inference locally, with limited collaboration. Most collaborative learning research focuses on Euclidean data with regular, grid-like structure (e.g., images, text). However, these approaches fail to capture the relational patterns in many real-world applications, best represented by graphs. Learning on graphs relies on message-passing mechanisms to propagate information between connected nodes, making it conceptually well-suited for collaborative environments where agents must exchange information. Yet, the opport
    
[^154]: 面向参数化偏微分方程规范算子学习的方程重构

    Equation Recast for Canonical Operator Learning Across Parametric PDEs

    [https://arxiv.org/abs/2609.02982](https://arxiv.org/abs/2609.02982)

    提出方程重构方法，将参数化偏微分方程的算子学习转化为单一规范算子的学习，通过解析推导吸收参数引起的算子变化到有效源项中，从而实现对新参数范围的零样本预测、外推以及融合稀疏异构数据的能力。

    

    在宽泛参数范围内学习解算子，可能需要对输入函数和物理参数都进行大量覆盖，尤其对于纯数据驱动的参数化模型更是如此。此外，这类模型在训练分布之外可能会无声地失效。我们提出了方程重构，该方法将参数化算子学习重新表述为对单一规范算子的学习。由参数引起的算子变化通过控制方程解析推导得到，并被吸收进有效源项中，从而实现对新参数范围的零样本预测。在多参数、非线性以及奇异偏微分方程的设置中，方程重构支持外推，能够在共享的规范表示下整合稀疏的异构数据集，并将收敛失败作为重构迭代失效的内部预警信号。在用于核聚变的高保真托卡马克模拟中，该框架统一了……

    arXiv:2609.02982v1 Announce Type: new  Abstract: Learning solution operators across broad parameter ranges can require substantial coverage of both input functions and physical parameters, particularly for purely data-driven parametric models. In addition, the resulting models may fail silently outside the training distribution. We introduce equation recast, which reformulates parametric operator learning as the learning of a single canonical operator. Parameter-induced operator variations are derived analytically from the governing equation and absorbed into effective sources, enabling zero-shot prediction across new parameter regimes. Across multi-parameter, nonlinear, and singular PDE settings, equation recast supports extrapolation, integrates sparse heterogeneous datasets in a shared canonical representation, and uses loss of convergence as an internal warning signal for failure of the recast iteration. In high-fidelity tokamak simulations for nuclear fusion, the framework unifies
    
[^155]: 联邦学习中的隐私泄露：面向车联边缘网络中惯性传感的基于梯度的客户端身份推断与防御

    Privacy Leakage in Federated Learning: Gradient-Based Client Identity Inference and Defenses for Inertial Sensing in Vehicular Edge Networks

    [https://arxiv.org/abs/2609.02971](https://arxiv.org/abs/2609.02971)

    该论文揭示了在车联网联邦学习中，服务器可从客户端上传的梯度更新中以近乎完美的准确率推断出基于IMU惯性数据的客户端身份，证实了严重的匿名性隐私泄露风险并提出了相应的防御方法。

    

    随着车联网向5G/6G边缘智能发展，联邦学习（FL）被广泛推广为一种隐私保护方法，使车辆和基础设施能够在不暴露原始传感器数据的情况下协同训练共享模型。然而，客户端传输的更新仍然会泄露足够的信息来识别发送者的身份，这威胁到了安全关键的V2X应用所依赖的匿名性，并加剧了人们对对抗性机器学习、模型投毒和后门攻击等已有问题的担忧。我们研究了服务器端基于传输的权重增量进行客户端身份推断的问题，使用惯性（IMU）测量数据，并在UCI人体活动识别（HAR）基准上进行评估，该基准作为车载网联车辆所产生的IMU数据流的易于获取的代理。在五种攻击分类器和五种非独立同分布（non-IID）数据划分下，诚实但好奇的服务器能够从未加防御的更新中以近乎完美的准确率（约1.000）恢复客户端身份，证实了一种具体的可识别性风险。

    arXiv:2609.02971v1 Announce Type: cross  Abstract: As vehicular networks move toward 5G/6G edge intelligence, federated learning (FL) is widely promoted as a privacy-preserving way for vehicles and infrastructure to train shared models without exposing raw sensor data. Yet the updates clients transmit still leak enough information to identify who sent them, which threatens the anonymity that safety-critical V2X applications assume and adds to existing concerns over adversarial ML, model poisoning, and backdoor attacks. We study server-side client identity inference from transmitted weight deltas using inertial (IMU) measurements, evaluated on the UCI Human Activity Recognition (HAR) benchmark as an accessible proxy for the IMU streams produced onboard connected vehicles. Across five attack classifiers and five non-IID partitions, an honest-but-curious server recovers client identity with near-perfect accuracy (approximately 1.000) from undefended updates, confirming a concrete identifi
    
[^156]: 从稀缺标签中学习：用于射血分数预测的多视图超声心动图

    Learning from Scarce Labels: Multi-View Echocardiography for Ejection Fraction Prediction

    [https://arxiv.org/abs/2609.02969](https://arxiv.org/abs/2609.02969)

    该研究首次创建了公开的PLAX超声心动图射血分数预测数据集（超过25,000个标注视频），并训练出首个可复现的PLAX EF模型，其MAE达6.86%，性能与临床标准的心尖四腔方法相当。

    

    据我们所知，我们提出了首个公开可用的资源，用于从胸骨旁长轴（PLAX）超声心动图预测左心室射血分数（EF）。由于此前不存在PLAX-EF数据集，我们的工作专注于一种创新的数据生成策略来克服这种稀缺性。通过利用临床笔记与超声心动图视频之间的时间相关性，结合微调的视图分类器和代理标注，我们创建了一个包含超过25,000个PLAX视频的标注数据集。这使我们能够训练出首个可复现的PLAX EF模型，实现了6.86%的平均绝对误差（MAE）。鉴于作为临床标准的心尖四腔（A4C）方法报告的MAE值为6%-7%，我们的结果表明从PLAX视图进行EF估计既可行又具有临床相关性。这超越了现有方法的性能，并为相关情况提供了具有临床价值的解决方案。

    arXiv:2609.02969v1 Announce Type: cross  Abstract: We present, to the best of our knowledge, the first publicly available resource for predicting left ventricular ejection fraction (EF) from parasternal long-axis (PLAX) echocardiography. Because no PLAX-EF datasets previously existed, our work focuses on an innovative data generation strategy to overcome this scarcity. By leveraging a time-based correlation between clinical notes and echocardiographic videos, combined with fine-tuning view classifiers and proxy labeling, we created a labeled dataset of over 25,000 PLAX videos. This enables us to train the first reproducible PLAX EF model, achieving a mean absolute error (MAE) of 6.86%. Given that apical four-chamber (A4C) methods, the clinical standard, report MAE values of 6%-7%, our results demonstrate that EF estimation from PLAX views is both feasible and clinically relevant. This surpasses the performance of existing methods and provides a clinically relevant solution for situatio
    
[^157]: 基于联邦图学习的LLM多智能体系统隐私保护拓扑引导安全方法

    Privacy-Preserving Topology-Guided Safety for LLM-Based Multi-Agent Systems via Federated Graph Learning

    [https://arxiv.org/abs/2609.02967](https://arxiv.org/abs/2609.02967)

    提出FGLGuard框架，通过图联邦学习让各运营方在本地训练GNN风险检测器且仅共享模型更新，在保护私有数据的前提下实现对LLM多智能体系统的跨组织隐私保护安全防护。

    

    arXiv:2609.02967v1 公告类型：cross 摘要：针对基于LLM的多智能体系统（MAS）的拓扑引导安全防护方法，通过在智能体间通信图上训练图神经网络（GNN）来定位风险智能体并对拓扑结构进行干预——但这些方法假设单一运营方能够汇集所有带标签的交互轨迹数据。在跨组织的场景下，这一假设不再成立：交互片段包含私人提示词、工具输出和专有工作流，且没有任何单一数据孤岛能够观察到完整的攻击分布。我们将隐私保护的MAS安全防护问题构建为图联邦学习任务，并提出了FGLGuard：每个运营方在自己的由评判器标注的交互片段图上训练带边特征的图注意力检测器，仅共享模型更新。该方法结合了面向非IID客户端的近端局部目标函数、领域平衡聚合、过度拒绝约束下的阈值校准、经多方证实的上游评分以及针对被拦截答案的受保护改写机制。联邦化并非可有可无：现成的迁移学习方法在分布偏移下会失效（AUROC

    arXiv:2609.02967v1 Announce Type: cross  Abstract: Topology-guided safeguards for LLM-based multi-agent systems (MAS) train a GNN over the inter-agent communication graph to localize risky agents and intervene on the topology---but they assume one operator can pool all labeled traces. Across organizations that assumption breaks: episodes contain private prompts, tool outputs, and proprietary workflows, and no silo alone sees the full attack distribution. We cast privacy-preserving MAS safeguarding as graph federated learning and instantiate FGLGuard: each operator fits an edge-featured graph attention detector on its own judge-labeled episode graphs and shares only model updates. The method couples a proximal local objective for non-IID clients, domain-balanced aggregation, over-refusal-constrained threshold calibration, corroborated upstream scoring, and a guarded rewrite for blocked answers. Federation is not optional: off-the-shelf transfer collapses under distribution shift (AUROC 
    
[^158]: 基于动态谱优化的物理信息神经网络代理模型：用于硅衬底外延SrTiO₃忆阻器中的氧空位动力学研究

    Physics-Informed Neural Network Surrogate for Oxygen Vacancy Dynamics in epitaxial $\mathrm{SrTiO_3}$ on Si memristors via Dynamic Spectral Optimization

    [https://arxiv.org/abs/2609.02966](https://arxiv.org/abs/2609.02966)

    该研究提出级联物理信息神经网络结合动态切比雪夫谱优化器的新架构，无需算子分裂即可求解条件数超过10¹⁶的硅基SrTiO₃忆阻器氧空位漂移扩散问题，并高精度重现了实验电流-电压迟滞特性。

    

    物理信息神经网络（PINNs）为半导体器件建模提供了一个有前景的框架，然而标准网络架构难以应对氧化物异质结构中固有的严重数值刚性与多尺度空间差异问题。本文展示了一种级联式物理信息神经网络架构，结合自定义的二阶切比雪夫第二类多项式谱优化器（DSO V2 混合方法），用于模拟Pt/SrTiO₃/Si忆阻异质结构中跨越20纳米STO薄膜和380微米硅衬底的离子-电子漂移扩散输运过程。通过将电势、载流子密度和空位输运分离到四个按序训练的子神经网络中，我们的模型无需算子分裂即可规避超过10¹⁶的条件数。训练后的代理模型成功重现了实验中导电原子力显微镜（c-AFM）测得的电流-电压迟滞曲线（R² > 0.96），同时确保在整个连续空间中严格满足泊松方程一致性。与常规方法相比……

    arXiv:2609.02966v1 Announce Type: cross  Abstract: Physics-informed neural networks (PINNs) offer a promising framework for modeling semiconductor devices, yet standard architectures struggle with severe numerical stiffness and multiscale spatial discrepancies inherent to oxide heterostructures. Here, we demonstrate a cascaded PINN architecture coupled with a custom second-order Chebyshev second kind polynomial spectral optimizer (DSO V2 Hybrid) to model ion-electronic drift-diffusion transport in Pt/SrTiO$_3$/Si memristive heterostructures across a 20 nm STO film on a 380 $\mu$m Si substrate. By isolating potential, carrier density, and vacancy transport into four sequentially trained sub-neural-networks, our model circumvents condition numbers exceeding $10^{16}$ without operator splitting. The trained surrogate reproduces experimental conductive-AFM current-voltage hysteresis ($R^2 > 0.96$) while ensuring strict Poisson consistency across continuous space. Compared to conventional f
    
[^159]: 动态图异常检测的统计特征增强方法

    Statistical Feature Augmentation for Anomaly Detection in Dynamic Graphs

    [https://arxiv.org/abs/2609.02965](https://arxiv.org/abs/2609.02965)

    提出一种统计特征增强方法，将行为交互统计信息显式编码到输入特征空间中，从而提升动态图异常检测任务在多个模型和数据集上的表现。

    

    动态网络被应用于众多领域，从社交媒体到物流系统，每个领域都有其独特的特性。应用于这类数据的模型必须同时捕捉时间/结构信息与基于特征的信息之间的双重性。然而，最先进的深度学习模型往往难以直接从原始事件流中学习短期的行为交互信号，例如发送者强度或交互惯性。为了解决这一不足，我们提出了一种统计特征增强方法，将行为交互统计信息显式地编码到输入特征空间中。我们在三个真实世界数据集（Reddit、Wikipedia、MOOC）上，以及涵盖连续时间和离散时间架构的七个模型上，对所提出的方法在异常检测任务中进行了评估。作为基线，我们对在原始嵌入上训练的相同模型进行了应用。我们的结果表明，增强方法……

    arXiv:2609.02965v1 Announce Type: cross  Abstract: Dynamic networks are being applied in many domains, from social media to logistics systems, each with their own set of special characteristics. A model employed on this type of data must capture the duality between temporal/structural and feature-based information. Yet state-of-the-art deep learning models often struggle to learn especially short-term behavioral interaction signals, such as sender intensity or interaction inertia, directly from raw event streams. To address this gap, we propose a statistical feature augmentation method that explicitly encodes behavioral interaction statistics into the input feature space. We evaluate our proposed method on an anomaly detection task across three real-world datasets (Reddit, Wikipedia, MOOC) and seven models spanning both continuous-time and discrete-time architectures. As a baseline, we apply the same models trained on the original embeddings. Our results show, that augmentation consist
    
[^160]: SurfSpec：通过约束口袋-配体几何错配来增强脱靶结构无关的特异性

    SurfSpec: Enhancing Off-Target-Agnostic Specificity by Bounding Pocket-Ligand Geometric Mismatch

    [https://arxiv.org/abs/2609.02963](https://arxiv.org/abs/2609.02963)

    SurfSpec 通过度量并优化靶标口袋与配体之间的几何错配，利用三角不等式在完全不需要脱靶结构信息的前提下，为几何上分离的脱靶提供保守的特异性下界，从而实现脱靶结构无关的特异性感知先导化合物优化。

    

    基于结构的药物设计中的先导化合物优化旨在改善与靶标的结合，同时避免与脱靶口袋产生非预期的相互作用。然而，现有的亲和力驱动方法并未显式地控制特异性，而当前的特异性感知方法通常需要预先获得脱靶结构的信息。我们通过分析配体与靶标口袋之间的几何错配，来解决脱靶结构无关的特异性感知先导化合物优化问题。对于几何上相互分离的脱靶，我们在无需获取脱靶结构的情况下给出了一个保守的特异性下界。通过对口袋-配体错配进行度量化，三角不等式表明：减小靶标-配体错配可以提升与相分离脱靶类别之间错配的保守下界，而该下界可通过经验性的“几何-亲和力”校准转化为特异性下界。基于这一分析，我们提出了……（原文摘要在此处截断）

    arXiv:2609.02963v1 Announce Type: cross  Abstract: Lead optimization in structure-based drug design aims to improve target binding while avoiding unintended interactions with off-target pockets. However, existing affinity-driven methods do not explicitly control specificity, whereas current specificity-aware approaches commonly require prior knowledge of off-target structures. We address off-target-agnostic specificity-aware lead optimization by analyzing the geometric mismatch between a ligand and the target pocket. We provide a conservative specificity lower bound for geometrically separated off-targets without requiring access to off-target structures. By metricizing pocket--ligand mismatch, the triangle inequality shows that reducing target--ligand mismatch improves a conservative lower bound on mismatch to a separated off-target class, which can be translated into a specificity lower bound through an empirical geometry--affinity calibration. Motivated by this analysis, we introduc
    
[^161]: 无知的几何学：大语言模型知道何时调节贝叶斯先验

    The Geometry of Ignorance: LLMs Know When to Temper Bayesian Priors

    [https://arxiv.org/abs/2609.02959](https://arxiv.org/abs/2609.02959)

    研究发现大语言模型的反嵌入矩阵中存在一个编码训练语料词元分布的“无知方向”，模型通过逐词元调节该先验的强度，实现了随上下文信息增加而逐步减弱先验影响的温度调节贝叶斯更新。

    

    当语言模型缺乏线索时，它会预测什么？答案隐藏在其反嵌入矩阵的几何结构中：反嵌入矩阵的一个单一方向编码了训练语料库的词元分布，它充当了模型在不确定时回退依赖的贝叶斯先验。我们将这一结构称为“无知方向”，它出现在所有四个被检验的模型家族中（Llama、Qwen、Gemma 和 Pythia），参数规模从 0.4B 到 405B 不等。将最终预测状态投影到该方向上可得到逐词元的先验载荷因子 λ，经验表明该因子随着上下文信息量的增加而稳步下降。从形式上看，同样的投影将预测状态分解为两个正交向量，它们恰好对应于温度调节贝叶斯更新的两个因子：被提升到指数 λ 的词元先验以及由上下文驱动的似然。

    arXiv:2609.02959v1 Announce Type: cross  Abstract: What does a language model predict when it has few clues? The answer lurks in its unembedding geometry: a single direction of the unembedding matrix encodes the unigram distribution of the training corpus, which serves as the Bayesian prior the model falls back on when uncertain. This structure --- which we term the \emph{direction of ignorance} --- appears in all four model families examined (\texttt{Llama}, \texttt{Qwen}, \texttt{Gemma}, and \texttt{Pythia}), ranging from 0.4B to 405B parameters. Projecting the final prediction state onto this direction yields a per-token \emph{prior loading factor} $\lambda$, which, empirically, declines steadily as the context becomes more informative. Formally, the same projection decomposes the prediction state into two orthogonal vectors that correspond exactly to the two factors of a tempered Bayesian update: a unigram prior raised to the exponent $\lambda$ and a context-driven likelihood. This
    
[^162]: FrOGS：跨化学条件的独立合金构型离散神经采样器

    FrOGS: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical Conditions

    [https://arxiv.org/abs/2609.02948](https://arxiv.org/abs/2609.02948)

    FrOGS是一种将自回归模型与连续时间马尔可夫链在单一损失下联合训练的混合离散神经采样器，可跨多种化学条件独立采样合金构型，并给出配分函数的无偏估计和热力学性质的一致性估计。

    

    预测合金的热力学性质需要在多种化学条件下对其构型进行采样，并在共同的绝对尺度上恢复自由能。马尔可夫链蒙特卡罗（MCMC）是标准工具，但它需要在不同的化学条件下分别进行模拟，并且需要借助热力学积分等辅助自由能方法才能将结果置于共同的绝对尺度上。现代离散神经采样器通常以反向KL散度作为训练目标，容易产生模式坍缩或偏差。我们提出了自由能生成采样器，这是一种混合离散神经采样器，它将自回归模型与连续时间马尔可夫链（CTMC）相耦合，在单一共享损失下进行联合训练。FrOGS能够抽取独立同分布的构型样本，返回配分函数的无偏估计，并给出热力学观测量的一致性估计。我们使用单一模型在宽泛的……

    arXiv:2609.02948v1 Announce Type: cross  Abstract: Predicting the thermodynamic properties of an alloy requires sampling its configurations across many chemical conditions and recovering free energies on a common absolute scale. Markov chain Monte Carlo (MCMC) is the standard tool, but it requires separate simulations at different conditions, and auxiliary free-energy methods such as thermodynamic integration are used to place results on a common absolute scale. Modern discrete neural samplers typically use reverse KL divergence as the objective and can be mode-seeking or biased. We present Free energy Offering Generative Sampler (FrOGS), a hybrid discrete neural sampler that couples an autoregressive model to a continuous-time Markov chain (CTMC) to be trained jointly under a single shared loss. FrOGS draws i.i.d. configurations, returns an unbiased estimate of the partition function, and gives consistent estimates of thermodynamic observables. We train a single model across a wide ra
    
[^163]: 面向认知诊断的隐私保护异构多LLM联邦推理

    Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis

    [https://arxiv.org/abs/2609.02947](https://arxiv.org/abs/2609.02947)

    该论文提出一种隐私保护的异构多LLM联邦推理框架，通过本地拉普拉斯噪声差分隐私和基于残差的聚合机制，使多个商用LLM API无需访问原始学生数据即可协作实现准确的认知诊断。

    

    AI驱动的教育系统在平衡隐私保护与准确认知诊断方面仍面临重大挑战。为克服这一问题，我们提出了一种联邦推理框架，使多个商用LLM API能够在无需访问原始学生数据或专有模型内部结构的前提下进行协作。该框架基于异构多LLM架构，利用多个联邦实体（如LLaMA-3.3-70B、GPT-4o-mini和Claude-3-Haiku）。这些实体生成的预测通过epsilon本地差分隐私进行融合，即在聚合之前对每个实体的预测输出本地添加拉普拉斯噪声，同时采用基于残差的聚合方式来缓解模型间的异质性。我们的方法建立在“诚实但好奇”的信任范式之上，即假设API提供者不会滥用所提交的查询，并且我们的差分隐私机制保护已发布的诊断结果免受外部……（原文摘要在此处截断）

    arXiv:2609.02947v1 Announce Type: cross  Abstract: Significant challenges remain in AI-driven educational systems in balancing privacy preservation with accurate cognitive diagnosis. To overcome this, we propose a federated inference framework in which several commercial LLM APIs collaborate without requiring access to raw student data or proprietary model internals. Using multiple federated entities, such as LLaMA-3.3-70B, GPT-4o-mini, and Claude-3-Haiku, our framework builds upon a heterogeneous multi-LLM architecture. The predictions generated by these entities are combined with epsilon-local differential privacy by adding Laplace noise locally to each entity's prediction output before aggregation, while residual-based aggregation mitigates model heterogeneity. Our approach is predicated on an honest-but-curious trust paradigm in which API providers are presumed not to abuse submitted queries, and our differential privacy mechanism shields the published diagnostic results from exter
    
[^164]: 基于大语言模型引导的强化学习实现多智能体战斗游戏中的自适应NPC行为

    LLM-Guided Reinforcement Learning for Adaptive NPC Behavior in Multi-Agent Combat Games

    [https://arxiv.org/abs/2609.02931](https://arxiv.org/abs/2609.02931)

    该论文提出一种运行时策略选择框架，利用本地部署的大语言模型（Mistral 7B）每五秒读取实时游戏状态并为预训练PPO策略的NPC分配战术标签，从而在不修改底层策略的前提下实现NPC对多样对手的自适应行为。

    

    战斗类电子游戏中的脚本化和基于规则的非玩家角色（NPC）往往表现出可预测的行为，经验丰富的玩家能够加以利用；而强化学习（RL）智能体在训练完成后通常保持固定的策略，难以针对不同对手调整自身策略。我们研究了一种运行时策略选择框架，在该框架中，大语言模型（LLM）对已训练好的强化学习策略进行引导，而不修改其底层行为。为验证该方法，我们在Unity中使用共享的PPO策略训练了五个NPC智能体，并将基线配置（策略独立行动）与LLM增强配置进行对比：在后一配置中，通过Ollama访问的本地托管Mistral 7B模型每五秒读取一次实时游戏状态，并分配四种战术标签之一。我们在600个回合中针对三种脚本化对手类型对两种配置进行了评估，并使用曼-惠特尼U检验分析实验结果。

    arXiv:2609.02931v1 Announce Type: cross  Abstract: Scripted and rule-based non-player characters (NPCs) in combat video games often exhibit predictable behaviors that experienced players can exploit, while reinforcement learning (RL) agents typically retain a fixed policy after training and cannot readily adapt their strategy to different opponents. We investigate a runtime strategy-selection framework in which a large language model (LLM) guides a trained RL policy without modifying its underlying behavior. To demonstrate this, we train five NPC agents with a shared PPO policy in Unity and compare a baseline configuration, in which the policy acts independently, with an LLM-augmented configuration in which a locally hosted Mistral 7B model, accessed through Ollama, reads the live game state every five seconds and assigns one of four tactical tags. We evaluate both configurations against three scripted opponent types across 600 episodes and analyze outcomes using the Mann-Whitney U tes
    
[^165]: 迈向将强化学习扩展到大规模群体：学习平均场表示

    Towards Scaling Reinforcement Learning to Massive Populations: Learning Mean-Field Representations

    [https://arxiv.org/abs/2609.02928](https://arxiv.org/abs/2609.02928)

    本文提出通过学习平均场表示，将强化学习扩展到大规模智能体群体，从而使大规模群体场景下的高维控制问题变得可扩展且易于处理。

    

    现代多智能体系统越来越多地大规模部署于大型智能体群体的场景中，例如广告拍卖、交通路由和推荐系统。在这类场景中，主流方法是对每个智能体的策略进行独立优化，将其他智能体视为固定单智能体环境的一部分，而不对群体动态进行建模。在许多大规模群体系统中，系统的动态取决于群体的聚合摘要信息，而非任何个体的身份。平均场强化学习利用了这种结构，提供了一个有原则的框架，将每个智能体的环境建模为群体分布的显式函数。然而，在大规模状态-动作空间或高维控制问题中，对群体分布进行建模本身就是难以处理的。我们如何为具有大规模群体的高维控制问题设计一个可扩展的框架？本工作探索了这一课题。

    arXiv:2609.02928v1 Announce Type: cross  Abstract: Modern multi-agent systems are increasingly deployed at scale over large populations of agents in settings such as ad-auctions, traffic routing, and recommendation systems. The dominant approach in such settings is to optimize each agent's policy independently, treating the other agents as part of a fixed single-agent environment rather than modeling the population dynamics. In many large-population systems, the dynamics depend on an aggregate summary of the population rather than the identity of any individual. Mean-field RL exploits such structure, providing a principled framework that models each agent's environment as an explicit function of the population distribution. However, in large state-action spaces or high-dimensional control problems, modeling the population distribution is itself intractable. How can we design a scalable framework for high-dimensional control problems with large populations? This work explores this quest
    
[^166]: 基于流匹配与CMS开放数据的强子单Z暗物质灵敏度研究

    Hadronic Mono-Z Dark Matter Sensitivity with Flow Matching on CMS Open Data

    [https://arxiv.org/abs/2609.02923](https://arxiv.org/abs/2609.02923)

    该研究利用条件流匹配连续归一化流对CMS开放数据中的强子单Z事件背景进行建模，并通过哨兵值填补、固定数据划分和最小背景产额约束等措施确保评估稳健，实现了对暗物质信号的预期灵敏度分析，对三个模拟信号样本分别给出2.89σ、7.62σ和7.41σ的预期显著性。

    

    我们报告了一项针对强子单Z暗物质产生过程的预期灵敏度研究，使用CMS 2015D年Run的HTMHT开放数据，对应积分亮度2.256382381 fb⁻¹，其中共有1,439,523个事件满足强子单Z选择条件。背景采用在所选HTMHT事件上训练的条件流匹配连续归一化流进行建模，并在按全部所选事件总体重新加权的留出验证集上进行评估。为减轻缺失物体特征带来的伪影并避免样本内打分偏差，我们对未定义的角度特征采用哨兵值填补，持久化保存训练/验证划分索引，并在选择工作点时强制要求最小报告背景产额为20个事件。在对信号打分之前，先将信号侧的离线触发代理应用于模拟信号。在该流程下，基线分析对三个模拟信号样本给出2.89σ、7.62σ和7.41σ的预期显著性。

    arXiv:2609.02923v1 Announce Type: cross  Abstract: We present a projected sensitivity study for hadronic mono-$Z$ dark-matter production using CMS Run~2015D HTMHT open data corresponding to 2.256382381~\invfb, from which 1{,}439{,}523 events satisfy the hadronic mono-$Z$ selection. Backgrounds are modelled with a conditional flow-matching continuous normalizing flow trained on the selected HTMHT events and evaluated on a held-out validation split reweighted to the full selected population. To mitigate artifacts from missing-object features and avoid in-sample scoring bias we apply sentinel imputation for undefined angular features, persist the train/validation split indices, and enforce a minimum reported background yield of 20 events when selecting the working point. A signal-side offline trigger proxy is applied to the simulated signal before scoring. Under this procedure the baseline analysis yields expected significances of 2.89$\sigma$, 7.62$\sigma$, and 7.41$\sigma$ for three sim
    
[^167]: 评估图神经网络在艺术家合作网络成功预测中的应用

    Evaluating GNNs for Success Prediction in Artist Collaboration Networks

    [https://arxiv.org/abs/2609.02920](https://arxiv.org/abs/2609.02920)

    本研究引入波兰音乐合作网络新数据集，与意大利、丹麦网络进行对比并合并分析，同时提出一个评估图神经网络（GNN）基于元数据和网络位置预测艺术家人气有效性的框架，发现波兰网络与三国合并网络具有相似的特性与聚类行为。

    

    随着音乐产业日益成为一种协作性事业，理解艺术家网络的底层结构已成为文化数据分析的重点。本研究通过引入波兰音乐场景的新数据集，扩展了对意大利和丹麦网络的先前分析。通过采用先前研究中使用的方法，这项工作能够对三种不同的欧洲音乐生态进行直接比较，并允许将所构建的网络合并为一体。此外，本研究引入了一个框架，用于测试图神经网络（GNN）基于元数据和网络中位置来预测艺术家人气的有效性。统计分析显示，波兰网络和三国合并网络表现出相似的性质和聚类行为，与先前的模型一致。对预测架构的评估表明，虽然GNN模型达到了……

    arXiv:2609.02920v1 Announce Type: cross  Abstract: As the music industry becomes an increasingly collaborative effort, understanding the underlying structures of the artist network has become a focal point in cultural data analytics. This study expands on the previous analyses of the Italian and Danish networks by introducing a novel dataset of the Polish music scene. By utilizing methodologies used in the prior studies, this work enables a direct comparison between three distinct European music landscapes and allows to merged the created networks into one. Furthermore, this research introduces a framework to test the efficacy of Graph Neural Networks (GNNs) for artist popularity predictions based on the metadata and the position in the network. The statistical analysis revealed that the Polish and tri-national network exhibit similar properties and clustering behaviours, consistent with prior models. An evaluation of the predictive architectures reveals that while GNN models achieve a
    
[^168]: BharatGather：一个面向印度公共事件虚假信息与假新闻检测的文化感知基准数据集

    BharatGather: A Culturally-Informed Benchmark Dataset for Misinformation and Fake News Detection in Indian Public Events

    [https://arxiv.org/abs/2609.02895](https://arxiv.org/abs/2609.02895)

    本文提出了BharatGather数据集，一个专为印度大型公共活动中虚假信息二元分类设计的文化感知基准数据集，包含14,646条通过事实核查平台爬取、多媒体转录提取与大语言模型合成增强相结合的混合流水线构建的记录。

    

    大型公共活动，如宗教节庆、政治集会和文化聚会，日益容易受到虚假信息快速传播的影响，对公共安全和社会凝聚力构成重大风险。尽管自动假新闻检测在方法论上取得了显著进展，但现有基准往往无法捕捉印度背景下特有的社会文化细微差异和事件特定动态。本文介绍了BharatGather，这是一个精心策划的多源数据集，专门为印度大型集会生态系统中的二元虚假信息分类而设计。该语料库包含14,646条记录，通过混合流水线构建，包括对知名事实核查平台的系统性网络爬取、多媒体转录文本提取，以及由大语言模型（LLM）介导的合成数据增强，以确保叙事的多样性。通过提供一个针对该领域量身定制的资源（摘要原文在此处截断）

    arXiv:2609.02895v1 Announce Type: new  Abstract: Large-scale public events, such as religious festivals, political rallies, and cultural gatherings, are increasingly vulnerable to the rapid dissemination of misinformation, posing substantial risks to public safety and social cohesion. While automated fake news detection has seen significant methodological progress, existing benchmarks frequently fail to capture the socio-cultural nuances and event-specific dynamics characteristic of the Indian context. This paper introduces BharatGather, a curated, multi-source dataset specifically engineered for binary misinformation classification within the ecosystem of Indian mass gatherings. The corpus comprises 14,646 records constructed through a hybrid pipeline involving systematic web scraping of prominent fact-checking platforms, multimedia transcript extraction, and Large Language Model (LLM)-mediated synthetic augmentation to ensure narrative diversity. By providing a resource tailored to t
    
[^169]: 超越Nesterov的改进梯度下降下界

    Improved Gradient Descent Lower Bounds Beyond Nesterov

    [https://arxiv.org/abs/2609.02855](https://arxiv.org/abs/2609.02855)

    本文证明了光滑凸优化中固定步长梯度下降的两个更强下界——非anytime的Ω(n^{-1.6342})与anytime的Ω(n^{-1.2408})，并借助silver调度可达的O(n^{-log_2(1+√2)})速率，严格分离了两种设定下可实现的收敛指数。

    

    我们研究了在光滑凸优化中，梯度下降（GD）通过预先设定的步长能够被加速到何种程度。在超越Nemirovsky和Yudin经典的Ω(n^{-2})一阶oracle下界的基础上，我们证明了Ω(n^{-1.6342})的非anytime下界以及Ω(n^{-1.2408})的anytime下界。这两个结果分别改进了Ma和Chen近期提出的Ω(n^{-1.932})非anytime下界，以及Tsai等人提出的Ω(n^{-4/3}) anytime下界。结合silver步长调度所达到的非anytime O(n^{-log_2(1+√2)})收敛速率，我们的anytime下界在这两种设定下可实现的收敛指数之间建立了严格的分离。

    arXiv:2609.02855v1 Announce Type: cross  Abstract: We study how far gradient descent (GD) can be accelerated by predetermined stepsizes in smooth convex optimization. Going beyond the classical $\Omega(n^{-2})$ first-order oracle lower bound of Nemirovsky and Yudin, we prove an $\Omega(n^{-1.6342})$ non-anytime lower bound and an $\Omega(n^{-1.2408})$ anytime lower bound. These improve the recent $\Omega(n^{-1.932})$ non-anytime lower bound of Ma and Chen and the $\Omega(n^{-4/3})$ anytime lower bound of Tsai et al., respectively. Together with the non-anytime $O(n^{-\log_2(1+\sqrt{2})})$ rate achieved by silver schedules, our anytime lower bound establishes a strict separation between the achievable convergence exponents in the two settings.
    
[^170]: FlashKAN：基于截断幂形式的B样条KAN

    FlashKAN: B-Spline KANs via Truncated Power Form

    [https://arxiv.org/abs/2609.01956](https://arxiv.org/abs/2609.01956)

    FlashKAN用逼近论中的截断幂形式取代Cox-de Boor递归，通过torch.compile融合为单一GPU内核并结合有界坐标稳定化，显著加速了KAN中B样条激活函数的计算，并提供了开源软件包。

    

    Kolmogorov-Arnold网络（KAN）将可学习的B样条激活函数放置在网络边上，而非在节点上使用固定激活函数。标准的Cox-de Boor递归在计算k次样条的这些激活函数时需要k次顺序传递，消耗了超过90%的前向传播时间。FlashKAN用截断幂形式取代了这种递归，这是逼近论中的一个经典结果，它将每个均匀三次B样条表示为在移位节点位置上的五个(x)_+^3项。本文做出了三项贡献：(1) 一个torch.compile融合的实现，将这些操作合并为单个GPU内核，消除了所有递归、跨度查找和散布-聚集操作；(2) 一种有界坐标稳定化方法，将归一化输入钳制到[0, k+1]，防止了历史上促使Cox-de Boor递归被采用的灾难性抵消问题；(3) 一个可用于生产环境的开源软件包（pip install flashkan）。

    arXiv:2609.01956v1 Announce Type: new  Abstract: Kolmogorov-Arnold Networks (KANs) place learnable B-spline activations on network edges rather than fixed activations on nodes. The standard Cox-de Boor recursion evaluates these activations through k sequential passes for degree-k splines, consuming over 90% of forward-pass time. FlashKAN replaces this recursion with the truncated power form, a classical result from approximation theory that expresses each uniform cubic B-spline as five (x)_+^3 terms at shifted knot positions. This paper makes three contributions: (1) a torch.compile-fused implementation that collapses these operations into a single GPU kernel, eliminating all recursion, span lookup, and scatter-gather operations; (2) a bounded-coordinate stabilization that clamps the normalized input to [0, k+1], preventing the catastrophic cancellation that historically motivated the Cox-de Boor recursion; and (3) a production-ready, open-source package (pip install flashkan) that ser
    
[^171]: 用于网络压缩的可复用神经基底的数学理论

    A Mathematical Theory of Reusable Neural Bases for Network Compression

    [https://arxiv.org/abs/2609.01550](https://arxiv.org/abs/2609.01550)

    该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。

    

    随着大型AI模型在各类应用中日益普及，内存成本已成为训练和推理中的关键瓶颈。为缓解这一问题，我们提出了线性可复用神经基底架构（LRNBA），这是一种旨在提高参数效率并降低内存成本的新型框架。受循环神经网络（RNN）设计的启发，我们方法的核心思想是将每个网络块表示为共享神经基底集合的线性组合，从而在保持稳定训练的同时实现高度的网络压缩率。所提出的架构允许在相同的参数预算下构建显著更宽和更深的网络。大量实验表明，我们的模型与经典架构相比实现了相当甚至更快的收敛速度和更低的损失，同时保持了稳定的训练动态。

    arXiv:2609.01550v1 Announce Type: cross  Abstract: As large AI models become increasingly prevalent across a wide range of applications, memory cost has become a critical bottleneck in both training and inference. To mitigate this issue, we introduce the Linear Reusable Neural Bases Architecture (LRNBA), a novel framework aimed at improving parameter efficiency and reducing memory cost. Inspired by recurrent neural network (RNN) designs, the core idea of our approach is to represent each network block as a linear combination of a shared set of neural bases, thereby enjoying highly network compression rate while maintaining stable training. The proposed architecture allows for the construction of significantly wider and deeper networks under the same parameter budget. Extensive experiments demonstrate that our model achieves comparable or even faster convergence and lower loss than classical architectures, while maintaining stable training dynamics.
    
[^172]: LatentPress：超越文本与视觉的上下文压缩

    LatentPress: Context Compression Beyond Text and Vision

    [https://arxiv.org/abs/2609.01507](https://arxiv.org/abs/2609.01507)

    LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。

    

    压缩后的上下文通常以人类可读的文本形式承载，或以必须被解码的渲染图像形式承载，即使其消费者是语言模型也是如此。我们提出了LatentPress，它将对话历史和长文档写入第三种表示形式：连续的记忆token（memory tokens），冻结的解码器通过其输入嵌入接口直接读取这些token，在推理时无需进行文本重建。一个与阅读器匹配的小型写入器可实现4至16倍的压缩，同时只需训练一个适配器（参数量为420万至2620万，约占解码器的0.1%）。在LongMemEval基准上，LatentPress在7.70倍压缩下达到0.504的准确率，超过未压缩证据的0.490，并显著优于文本摘要（0.184）和基于OCR的压缩（0.426至0.312）。在LongBench-QA上，域内写入器在4至8倍压缩下匹配或超过原始上下文阅读的性能，而16倍压缩则落后于原始上下文。写入每段对话仅需43毫秒，大约快一个数量级。

    arXiv:2609.01507v1 Announce Type: cross  Abstract: Compressed context is usually carried as human-readable text or as rendered images that must be decoded, even when its consumer is a language model. We introduce LatentPress, which writes conversational histories and long documents into a third representation: continuous memory tokens that a frozen decoder reads directly through its input-embedding interface, with no text reconstruction at inference. A small reader-matched writer compresses $4$-$16\times$ while training only an adapter (4.2M-26.2M parameters, $\sim\!0.1\%$ of the decoder). On LongMemEval, LatentPress reaches $0.504$ accuracy at $7.70\times$ compression versus $0.490$ for uncompressed evidence, outperforming text summaries (0.184) and OCR-based compression (0.426 to 0.312). On LongBench-QA, in-domain writers match or exceed raw-context reading at $4$-$8\times$ compression, while $16\times$ trails raw. Writing takes 43ms per conversation, roughly an order of magnitude fa
    
[^173]: 通过幂律熵搜索高效估计最优超参数缩放定律

    Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search

    [https://arxiv.org/abs/2609.01431](https://arxiv.org/abs/2609.01431)

    本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。

    

    最优超参数缩放定律描述了用于大语言模型（LLM）训练的最佳超参数如何随模型和数据规模变化，使从业者无需昂贵的大规模调优即可预测生产规模下的最优配置。然而，传统上估计这些缩放定律需要对数千次训练运行进行穷举网格搜索，消耗巨大的计算资源。我们提出了幂律熵搜索（Power-Law Entropy Search, PLES），这是一种建立在多保真度贝叶斯优化之上的计算成本感知采集函数，能够通过自适应实验高效估计最优超参数缩放定律。PLES的一个关键创新在于，它搜索的是能够降低缩放定律估计整体不确定性的候选配置，而不是优化单一目标函数。在每次迭代中，PLES选择能够最大程度降低缩放定律估计不确定性的候选配置。

    arXiv:2609.01431v1 Announce Type: cross  Abstract: Optimal hyperparameter scaling laws describe how the best hyperparameters for large language model (LLM) training change with model and data scale, enabling practitioners to predict optimal configurations at production scales without expensive large-scale tuning. However, estimating these scaling laws conventionally requires exhaustive grid searches over thousands of training runs, consuming enormous computational resources. We introduce Power-Law Entropy Search (PLES), a computational cost-aware acquisition function built on multi-fidelity Bayesian optimization that efficiently estimates optimal hyperparameter scaling laws through adaptive experimentation. A key innovation in PLES is that it searches for candidates that reduce the overall uncertainty of a scaling law estimate, instead of optimizing a single objective function. At each iteration, PLES selects the candidate configuration that maximally reduces the uncertainty of the sca
    
[^174]: Modelpedia：面向AI元科学的模型发现目录

    Modelpedia: A Catalog of Model Findings for the Meta-Science of AI

    [https://arxiv.org/abs/2609.01090](https://arxiv.org/abs/2609.01090)

    提出了Modelpedia——一个利用大语言模型自动从已发表论文中提取AI模型相关发现、将其与模型、数据集、方法和概念关联，并汇总为可搜索公共目录的框架，同时基于该目录对AI社区如何研究模型进行了元分析。

    

    关于AI模型的科学知识产生的速度已超过社区能够整理的速度。每隔几个月，一个新的大型基础模型就会重塑该领域，数百篇论文、博客和技术报告记录着每个模型的表现或失败之处。然而，这些发现仍然分散，实际上无法有效检索。为了解决这一差距，我们提出了Modelpedia，这是一个自动化的、由大语言模型辅助的框架，它从已发表的论文中提取关于模型的发现，将其与所涉及的模型、数据集、方法和概念相关联，并将结果汇总到一个可搜索的公共目录中。将该原型应用于ICLR 2024和2025年被接收的论文，我们提取了一千多项发现，并将该目录本身作为研究对象，对社区如何研究模型进行了元分析。现在，我们邀请社区探索、贡献并基于这个开放目录进行构建，帮助将模型发现确立为AI元科学的共享基础。

    arXiv:2609.01090v1 Announce Type: new  Abstract: Scientific knowledge about AI models is produced faster than the community can organize it. Every few months a new foundation model reshapes the field and hundreds of papers, blogs, and technical reports document how each behaves or fails. Yet, these findings remain scattered and effectively unretrievable. To address this gap we present Modelpedia, an automated, LLM-assisted framework that extracts findings about models from published papers, links it to the model, dataset, method, and concept it concerns, and aggregates the result into a searchable public catalog. Applying the prototype to accepted ICLR 2024 and 2025 papers, we extract over a thousand findings and, treating the catalog itself as an object of study, run a meta-analysis of how the community investigates models. Now, we invite the community to explore, contribute to, and build on the open catalog, and to help establish model findings as a shared foundation for the meta-sci
    
[^175]: MUGEN：面向多种学习任务的不可学习图样本生成

    MUGEN: Generating Unlearnable Graph Examples for Multiple Learning Tasks

    [https://arxiv.org/abs/2609.00696](https://arxiv.org/abs/2609.00696)

    MUGEN是首个面向多种学习任务的不可学习图样本生成框架，它通过对单一干净数据集进行特征扰动，利用共享GNN编码器同时保护节点分类、图分类和链接预测等多种任务免受未经授权的模型学习。

    

    跨领域的图数据可能向未经授权的表示学习暴露有价值的关系信息，因此迫切需要防范此类滥用。不可学习样本提供了一种数据级防御手段，通过对训练发布数据进行扰动，使得在其上训练的模型无法泛化到干净数据。现有方法只能针对特定的下游任务生成不可学习图样本。因此，针对某一任务受保护的发布数据，对于数据所有者无法预料的其他潜在用途（包括节点分类、图分类和链接预测）仍可能是可学习的。我们提出了MUGEN，据我们所知，这是首个能够联合保护所有启用任务的不可学习图样本生成框架。MUGEN从一个干净数据集生成单一的特征扰动发布版本，通过共享的GNN编码器和任务特定的输出头保护每一个启用的任务。我们设计了一种任务对齐的（摘要在此处不完整）

    arXiv:2609.00696v1 Announce Type: new  Abstract: Graph data across diverse domains can expose valuable relational information to unauthorized representation learning, creating a pressing need for protection against such misuse. Unlearnable examples offer a data-level defense by perturbing a training release so that models trained on it fail to generalize to clean data. Existing methods generate unlearnable graph examples for only a specified downstream task. Consequently, a release protected against one task may remain learnable for other plausible uses, including node classification, graph classification, and link prediction, which the data owner cannot anticipate. We introduce MUGEN, to our knowledge the first framework for generating unlearnable graph examples that jointly protect all enabled tasks. From one clean dataset, MUGEN produces a single feature-perturbed release that protects every enabled task through a shared GNN encoder and task-specific heads. We devise a Task-Aligned 
    
[^176]: ReLU 训练中的奇异曲率：微分与梯度流极限不必交换

    Singular Curvature in ReLU Training:Differentiation and the Gradient-Flow Limit Need Not Commute

    [https://arxiv.org/abs/2608.30960](https://arxiv.org/abs/2608.30960)

    该论文证明在 ReLU 训练中“先微分离散梯度下降再取极限”与“先取梯度流极限再微分”不可交换：离散精确导数收敛到无事件的区域传播子，而极限流的导数额外包含速度归一化的激活事件转移，由此产生秩一的奇异曲率差异，且全局凸性也无法将其完全抵消。

    

    梯度下降（GD）是梯度流的显式欧拉离散化，但一个在状态上精确的连续时间替代模型在微分之后未必仍然精确。在每个固定的非共振步长下，普通的自动微分可以精确地微分所执行的硬 ReLU GD 程序。我们证明，在固定的有限时间范围内，GD 状态收敛，并且这些精确的离散导数趋近于一个无事件的区域传播子，而极限流的导数还额外包含速度归一化的激活事件转移。一个前点 Stieltjes 表示将绝对连续的区域 Hessian 与原子界面曲率分离开来；一个非零梯度跳变会产生恰好秩一的端点差异，并且只要某个事件是严格的，全局凸性就会阻止多个事件之间的完全抵消。然而，一个标准的全局 1-强凸的残差-ReLU 平方损失风险族可以实现任意大的 r……（原文摘要在此处截断）

    arXiv:2608.30960v1 Announce Type: new  Abstract: Gradient descent (GD) is explicit Euler for gradient flow, but a state-accurate continuous-time surrogate need not remain accurate after differentiation. At every fixed nonresonant step size, ordinary automatic differentiation exactly differentiates the executed hard-ReLU GD program. We prove that, over a fixed finite horizon, the GD states converge and these exact discrete derivatives approach an event-free regional propagator, whereas the derivative of the limiting flow also contains speed-normalized activation-event transfers. A prepoint Stieltjes representation separates the absolutely continuous regional Hessian from atomic interface curvature; one nonzero gradient jump produces an exactly rank-one endpoint discrepancy, and global convexity prevents complete multi-event cancellation whenever an event is strict. Nevertheless, a standard family of globally 1-strongly convex residual-ReLU squared-loss risks realizes arbitrarily large r
    
[^177]: 跨模式迁移学习：迈向统一的城市出行需求预测

    Learning to Transfer Across Modes: Towards Unified Urban Mobility Forecasting

    [https://arxiv.org/abs/2608.28273](https://arxiv.org/abs/2608.28273)

    提出TransMod统一框架，通过构建共享的区级空间表示对齐不同空间粒度的出行系统，实现异构出行模式间的知识迁移，从而解决多模式城市出行需求预测中的空间异质性与新兴模式数据稀缺问题。

    

    城市交通系统由多种出行模式组成，这些模式在同一城市内共存，并表现出复杂的相互依赖关系，导致各模式间的需求动态相互关联。然而，由于空间上存在显著异质性，以及新兴出行模式的历史数据有限，跨不同模式联合预测需求仍然极具挑战性。现有的预测方法大多针对单一出行模式开发，并隐含地假设源系统与目标系统之间具有兼容的空间结构，这严重限制了它们在多模式场景中的适用性。为应对这些挑战，我们提出了TransMod，一个统一的城市出行需求预测框架，能够在异构出行模式之间实现有效的知识迁移。TransMod构建了一个共享的区级空间表示，将具有不同空间粒度的出行系统对齐到统一的表示空间中。

    arXiv:2608.28273v1 Announce Type: new  Abstract: Urban transportation systems consist of multiple mobility modes that coexist within the same city and exhibit complex interdependencies, leading to correlated demand dynamics across modes. However, forecasting demand jointly across different modes remains challenging due to substantial heterogeneity in space and the limited availability of historical data for emerging modes. Existing forecasting methods are largely developed for individual mobility modes and implicitly assume compatible spatial structures between source and target systems, which severely restricts their applicability in multi-modal settings. To address these challenges, we propose \textbf{TransMod}, a unified framework for urban mobility demand forecasting that enables effective knowledge transfer across heterogeneous mobility modes. TransMod constructs a shared zone-level spatial representation that aligns mobility systems with different spatial granularities into a com
    
[^178]: Puro-2B：在RTX 5090上花费5090美元训练的“穷人版”Qwen2-1.5B

    Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090

    [https://arxiv.org/abs/2608.27370](https://arxiv.org/abs/2608.27370)

    本文提出了一种开源且成本高效的预训练配方，使在消费级RTX 5090 GPU上以极低计算成本训练出接近Qwen2.5-1.5B性能的Puro-2B模型成为可能。

    

    arXiv:2608.27370v1 公告类型：新  摘要：语言模型预训练几乎已成为高昂成本的代名词，使其对学术界和开源社区的大部分人来说遥不可及。尽管已有强大的开源努力，包括开放权重模型和开源训练配方，但长期以来一直缺少一种成本高效、硬件可访问且开源预训练配方。即使在小规模下，训练Llama-3.2-3B也需花费超过150万美元，而复现SmolLM3-3B则需超过70万美元。在本报告中，我们提出了一种旨在降低这一门槛的开源预训练配方。利用该配方，我们从零开始训练了一系列Puro-2B模型，使用FP8精度，在消费级RTX 5090 GPU上处理多达1.4万亿个令牌。该系列中的模型在令牌预算和所选配方变体上有所不同。我们最好的模型以不到6900美元的计算成本进行训练，并在我们的评估协议下接近Qwen2.5-1.5B的性能。这种成本效益...

    arXiv:2608.27370v1 Announce Type: new  Abstract: Language model pretraining has become almost synonymous with prohibitive cost, placing it out of reach for much of the academic and open-source communities. Although strong open-source efforts already exist, including open-weight models and open-source training recipes, a cost-efficient, hardware-accessible, and open-source pretraining recipe has long been missing. Even at a small scale, training Llama-3.2-3B costs over \$1.5M, and reproducing SmolLM3-3B needs over \$700K. In this report, we present an open pretraining recipe designed to lower this barrier. Using this recipe, we train a collection of Puro-2B models from scratch on up to 1.4 trillion tokens with FP8 precision on consumer-grade RTX 5090 GPUs. The models in the collection differ in token budgets and selected recipe variants. Our best model is trained at a compute cost of less than \$6.9K and approaches Qwen2.5-1.5B performance under our evaluation protocol. This cost effici
    
[^179]: 层级通道堆叠：一种用于AI生成图像检测的结构化决策框架

    Hierarchical Channel Stacking: A Structured Decision Framework for AI-Generated Image Detection

    [https://arxiv.org/abs/2608.26648](https://arxiv.org/abs/2608.26648)

    本文提出层级通道堆叠框架，通过多阶段CNN激活的结构化表示，在AI生成图像检测中实现高精度，并揭示层级特征对决策的互补贡献。

    

    许多合成图像检测器能产生准确的预测，但对这些决策是如何形成的提供有限的理解。本文介绍了层级通道堆叠（HCS），一种紧凑的AI生成图像检测框架，它将中间CNN激活转换为一个结构化的60维表示，该表示组织在三个逐渐加深的主干阶段中。HCS使用逐通道的Level-1分类器和一个Level-2聚合器来产生图像级预测，同时保留显式的层级结构以供分析。在一个涵盖GAN和扩散生成器的基准测试上，HCS在保留测试集上达到了86.7%的准确率和86.7%的宏F1分数。阶段消融实验表明，完整的三阶段系统优于缩减的单阶段和两阶段变体，这表明层级结构携带了互补的预测信息。阶段级贡献分析进一步显示，在所分析的检测器设置中，

    arXiv:2608.26648v1 Announce Type: cross  Abstract: Many synthetic-image detectors produce accurate predictions but offer limited insight into how those decisions are formed. This paper introduces Hierarchical Channel Stacking (HCS), a compact framework for AI-generated image detection that converts intermediate CNN activations into a structured 60-dimensional representation organized across three progressively deeper backbone stages. HCS uses per-channel Level-1 classifiers and a Level-2 aggregator to produce image-level predictions while preserving explicit hierarchical structure for analysis. On a benchmark spanning GAN and diffusion generators, HCS achieves 86.7% accuracy and 86.7% macro-F1 on the held-out test set. Stage ablation shows that the full three-stage system outperforms reduced single-stage and two-stage variants, indicating that the hierarchy carries complementary predictive information. Stage-level contribution analysis further shows that, in the analyzed detector setti
    
[^180]: SimCast-S2S：一种通过气候模拟迁移学习实现亚季节降水预报的高效生成模型

    SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations

    [https://arxiv.org/abs/2608.26594](https://arxiv.org/abs/2608.26594)

    SimCast-S2S通过潜扩散生成框架和气候模拟迁移学习，实现了高效且概率性的亚季节降水预报，解决了不确定性量化和计算成本两大核心瓶颈。

    

    arXiv:2608.26594v1 公告类型：新 摘要：亚季节到季节（S2S）降水预报具有重大的经济和社会影响，但由于预测信号弱、相关不确定性高以及业务系统的计算成本限制了模拟保真度，这一任务仍然具有挑战性。我们引入了SimCast-S2S，一种用于概率性S2S降水预报的生成式潜扩散框架，旨在解决数据驱动预测中的三个主要瓶颈。首先，由于S2S预测需要不确定性量化，而不仅仅是确定性点预报，SimCast-S2S是首个采用基于扩散的生成流程进行S2S预测的数据驱动系统，能够有效从底层条件分布中采样。其次，由于在物理空间中生成大规模概率集成计算成本高昂，SimCast-S2S改为在由变分自编码器学习的紧凑潜空间中运行。

    arXiv:2608.26594v1 Announce Type: new  Abstract: Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the computational cost of operational systems, which constrains simulation fidelity. We introduce SimCast-S2S, a generative latent-diffusion framework for probabilistic S2S precipitation forecasting that addresses three major bottlenecks in data-driven prediction. First, because S2S prediction requires uncertainty quantification rather than only deterministic point forecasts, SimCast-S2S is the first data-driven system that uses a diffusion-based generative pipeline for S2S prediction, enabling effective sampling from the underlying conditional distribution. Second, since generating large probabilistic ensembles is computationally costly in physical space, SimCast-S2S instead operates in a compact latent space learned by variational autoencoders
    
[^181]: JIT-Agent：通过即时工具进化扩展智能体能力的规模化方法

    JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution

    [https://arxiv.org/abs/2608.25593](https://arxiv.org/abs/2608.25593)

    JIT-Agent通过训练一个能即时生成和优化任务自适应工具模型的系统，显著提升了现成智能体的性能，使工具设计从手动变为自动化。

    

    arXiv:2608.25593v1 公告类型：新 摘要：智能体的能力并非仅由模型决定。智能体工具（包括记忆管理、规划策略、动作协议以及工具/技能编排）可能主导底层基础模型的贡献。然而，工具设计仍是手动的、任务特定的，并且从根本上不可扩展。我们提出了JIT-Agent，一个经过训练的工具智能模型，能够即时为任意现成的智能体大型语言模型合成任务自适应的工具。我们将智能体工具形式化为一个可组合、可机器生成的工件，由固定的四模块协议控制，并训练JIT-Agent为给定任务定制工具，修复工具以确保稳定可靠的执行，并通过从先前工具配置的扩展存档中提取性能信号来自我进化。配备JIT-Agent作为工具助手，DeepSeek-V4-Flash在DeepSearchQA（+9.1）和OdysseyBench（+4.3）上超越了GPT-5.6。

    arXiv:2608.25593v1 Announce Type: new  Abstract: Agent capability is not determined by the model alone. The agent harness, encompassing memory management, planning strategy, action protocol, and tool/skill orchestration, can dominate the contribution of the underlying foundation model. Yet harness design remains manual, task-specific, and fundamentally unscalable. We present JIT-Agent, a harness intelligence model trained to synthesize task-adaptive agent harnesses on the fly for arbitrary off-the-shelf agentic LLMs. We formalize the agent harness as a composable, machine-generatable artifact governed by a fixed four-module protocol, and train JIT-Agent to customize harnesses for a given task at hand, repair harnesses for stable and reliable execution, and self-evolve by distilling performance signals from an expanding archive of prior harness configurations. Equipped with JIT-Agent as a harness helper, DeepSeek-V4-Flash surpasses GPT-5.6 on DeepSearchQA (+9.1) and OdysseyBench (+4.3),
    
[^182]: 拒绝几何反映拒绝训练：多样的拒绝前缀能提升稳定秩并削弱拒绝向量消融攻击

    Refusal geometry reflects refusal training: diverse refusal prefixes can raise stable rank and weaken refusal vector ablation attacks

    [https://arxiv.org/abs/2608.25390](https://arxiv.org/abs/2608.25390)

    本文发现拒绝训练中的首词损失塑造了拒绝方向和子空间，且重复的拒绝前缀导致拒绝几何脆弱，但多样前缀能提升稳定性并削弱消融攻击。

    

    arXiv:2608.25390v1 公告类型：新 摘要：拒绝训练通过训练模型拒绝不安全查询来保护AI模型免受越狱攻击，降低滥用风险。近期研究发现，对齐语言模型中的拒绝行为可由单一激活方向或跨有害提示共享的低维拒绝子空间介导：消融这些方向会抑制拒绝，同时基本保留其他模型能力。然而，为何安全关键特征在广泛模型中涌现并集中为低维结构仍不清楚。在对OLMo-2-0425-1B-Instruct的案例研究中，我们发现拒绝几何反映拒绝训练：由拒绝完成首词损失引起的激活更新解释了产生的拒绝方向和拒绝子空间。我们通过拒绝数据集中的训练动态研究拒绝方向，并揭示其脆弱性与重复的拒绝开头相关，这反过来又影响模型对多样拒绝前缀的稳定性。

    arXiv:2608.25390v1 Announce Type: new  Abstract: Refusal training protects AI models from jailbreaks by training models to decline unsafe queries, reducing the risk of misuse. Recent work finds that refusal behavior in aligned language models can be mediated by a single activation direction or a low-dimensional refusal subspace shared across harmful prompts: ablating those directions suppresses refusals while largely preserves other model capabilities. Yet it remains unclear why safety-critical features in a wide range of models emerge and concentrated, low-dimensional structure. In a case study of OLMo-2-0425-1B-Instruct we find that the refusal geometry reflects refusal training: activation updates resulting from refusal-completion first-token losses explain the resulting refusal direction and refusal subspace. We study refusal directions through the training dynamics across refusal datasets and reveal that their brittleness is associated with repetitive refusal starts, which in turn
    
[^183]: PRQ-KMeans：用于语义标识符标记化的投影残差量化方法

    PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization

    [https://arxiv.org/abs/2608.24207](https://arxiv.org/abs/2608.24207)

    本文提出PRQ-KMeans方法，通过移除全局均值组件和基于Top-k相似性细化质心，解决了语义标识符残差量化中的三个关键局限性，从而提升了生成式检索与推荐中标记化的有效性。

    

    arXiv:2608.24207v1 公告类型：新 摘要：语义标识符（SIDs）将实体表示为层次化标记序列，用于生成式检索和推荐。残差量化分词器通过在每个层级选择一个码字并将残差传递给下一层级来构建这些序列。我们将此过程视为渐进式共性消除：每个标记捕获其组内共享的组件，而后续标记应对剩余差异进行建模。这一视角揭示了三个局限性：语料库范围的共享组件可能消耗第一层级的容量，硬分配忽略了与附近码字的梯度相似性，以及完整码字减法可能沿选定码字方向在下一个残差中留下变化。因此，我们在事后设置中开发了解决方案，其中残差构建不受输入重建的约束。具体来说，我们提出了PRQ-KMeans，它移除全局均值组件，并使用Top-k相似性细化质心。

    arXiv:2608.24207v1 Announce Type: new  Abstract: Semantic identifiers (SIDs) represent entities as hierarchical token sequences for generative retrieval and recommendation. Residual-quantization tokenizers construct these sequences by selecting a codeword at each level and passing a residual to the next. We view this process as progressive commonality removal: each token captures a component shared within its group, while later tokens should model the remaining differences. This view reveals three limitations: a corpus-wide shared component can consume first-level capacity, hard assignment ignores graded similarities to nearby codewords, and full-codeword subtraction can leave variation along the selected-codeword direction in the next residual. We therefore develop our solution in the post-hoc setting, where residual construction is not constrained by input reconstruction. Specifically, we propose PRQ-KMeans, which removes the global-mean component, refines centroids with Top-k simila
    
[^184]: 从松弛可索引性到精确可索引性：部分可观测静止多臂老虎机的$t$步方法

    From Relaxed Indexability to Exact Indexability: A $t$-Step Approach for Partially Observable Restless Bandits

    [https://arxiv.org/abs/2608.24167](https://arxiv.org/abs/2608.24167)

    本文提出一种$t$步前瞻阈值策略，通过多步值迭代扩展了部分可观测静止多臂老虎机的Whittle指数计算，在$t=1$时恢复现有线性阈值，在$t>1$时使阈值依赖补贴，从而更精确地逼近精确可索引性。

    

    arXiv:2608.24167v1 公告类型：新 摘要：Whittle指数策略为静止多臂老虎机提供了一种可扩展的方法，但在部分可观测性下，即使在单一信念状态确定无差异补贴也需要解决一个无闭式值函数的无限时域信念状态问题。Liu [10] 通过线性化未知决策边界来解决这一难题，从而得到一个线性系统和闭式近似Whittle指数。然而，由此产生的阈值仅使用一步主动-被动比较，未考虑更长时域的延续值。我们将此框架扩展为一种“$t$步前瞻阈值策略”。对于每个补贴$m$，阈值由$t$步有限时域值迭代下的主动减被动优势定义。当$t=1$时，阈值与$m$无关，并恢复Liu [10]的线性阈值；当$t>1$时，它通过诱导的首穿结构变得依赖补贴。

    arXiv:2608.24167v1 Announce Type: new  Abstract: Whittle index policies offer a scalable method for restless multi-armed bandits, but under partial observability even determining the indifference subsidy at a single belief requires solving an infinite-horizon belief-state problem with no closed-form value function. Liu [10] addresses this difficulty by linearizing the unknown decision boundary, leading to a linear system and a closed-form approximate Whittle index. However, the resulting threshold uses only a one-step active--passive comparison and does not account for longer-horizon continuation values.   We extend this framework to a \emph{$t$-step lookahead threshold policy}. For each subsidy $m$, the threshold is defined by the active-minus-passive advantage under $t$-step finite-horizon value iteration. At $t=1$, the threshold is $m$-independent and recovers the linear threshold of Liu [10]; for $t>1$, it becomes subsidy-dependent through the induced first-crossing structure and t
    
[^185]: 一种用于无创连续血压预测的多维数据驱动混合Transformer框架

    A Multidimensional Data-Driven Hybrid Transformer Framework for Non-invasive Continuous Blood Pressure Prediction

    [https://arxiv.org/abs/2608.23276](https://arxiv.org/abs/2608.23276)

    该论文提出一种混合Transformer框架，通过融合Transformer、Kolmogorov-Arnold网络与XGBoost的多源时序编码器和动态条件融合解码器，从ECG/PPG特征序列而非原始波形中实现无袖带的连续血压（舒张压和收缩压）预测。

    

    目标：开发并评估一种利用时序生理特征和人口统计学特征的无袖带连续血压（BP）估计器。我们提出一种混合Transformer框架，从ECG/PPG导出的特征序列中估计舒张压和收缩压。方法：该框架不对原始波形建模，而是对六个生理描述符和两个人口统计学协变量组成的10步序列进行建模。多源时序编码器模块结合Transformer、Kolmogorov-Arnold网络和XGBoost三个分支，以捕获互补的时序、非线性和表格信息。动态条件融合解码器应用差分多头注意力、令牌加权聚合和门控残差校正。一个稳健的复合目标函数联合优化DBP和SBP。主要结果：基于MIMIC-III波形和临床数据库，源数据池包含来自203名受试者的28,486个波形片段，特征生成保留了……

    arXiv:2608.23276v2 Announce Type: replace  Abstract: Objective. To develop and evaluate a cuffless continuous blood pressure (BP) estimator using temporal physiological and demographic features. We propose a hybrid Transformer framework to estimate diastolic and systolic BP from ECG/PPG-derived feature sequences. Approach. Rather than raw waveforms, the framework models 10-step sequences of six physiological descriptors and two demographic covariates. A Multi-Source Temporal Encoder Module combines Transformer, Kolmogorov-Arnold Network, and XGBoost branches to capture complementary temporal, nonlinear, and tabular information. A Dynamic Conditional Fusion-Decoder applies differential multi-head attention, token-weighted aggregation, and gated residual correction. A robust composite objective jointly optimizes DBP and SBP. Main results. Using the MIMIC-III Waveform and Clinical Databases, the source pool comprised 28,486 waveform segments from 203 subjects, and feature generation retai
    
[^186]: 从微观动力学中稳健发现粗粒化连续介质方程

    Robust Discovery of Coarse-Grained Continuum Equations from Microscopic Dynamics

    [https://arxiv.org/abs/2608.20404](https://arxiv.org/abs/2608.20404)

    本研究发现，在从微观动力学数据中发现粗粒化PDE时，数据量是影响识别稳健性的关键因素，而增大函数库则会降低发现效率，并通过相分离系统和Ising模型验证了这一点。

    

    arXiv:2608.20404v1 公告类型：交叉 摘要：直接从时空数据中发现控制偏微分方程（PDEs）已成为理解复杂系统动力学的有力工具。在本工作中，我们将PDE-SINDy应用于已知的相分离系统，并考察其性能如何依赖于可用数据量、函数库大小以及噪声的存在。我们的结果表明，方程发现的准确性强烈依赖于可用数据量。尽管在数据有限的情况下可以识别出正确的方程，但若干虚假项也会获得有限的选取概率。随着数据量的增加，这些虚假项逐渐被抑制，从而使得对控制方程的识别更加稳健。相反，增加函数库的大小会对方程发现的效率产生不利影响。此外，对于Glauber自旋翻转Ising模型，我们展示了...

    arXiv:2608.20404v1 Announce Type: cross  Abstract: The discovery of governing partial differential equations (PDEs) directly from spatiotemporal data has emerged as a powerful tool for understanding the dynamics of complex systems. In this work, we apply PDE-SINDy to well-known phase-separating systems and examine how its performance depends on the amount of available data, the size of the function library, and the presence of noise. Our results show that the accuracy of equation discovery depends strongly on the amount of available data. Although the correct equation can be identified with limited data, several spurious terms also acquire finite selection probabilities. As the amount of data increases, these spurious terms are progressively suppressed, leading to a more robust identification of the governing equation. In contrast, increasing the size of the function library adversely affects the efficiency of equation discovery. Further, for the Glauber spin-flip Ising model, we show 
    
[^187]: DeltaMomentum：一种基于键值对的各向异性动量更新方法，采用增量规则

    DeltaMomentum: A Key-Value based Anisotropic Momentum Update via Delta Rule

    [https://arxiv.org/abs/2608.19491](https://arxiv.org/abs/2608.19491)

    DeltaMomentum通过利用梯度中的键值结构，将方向感知引入动量更新规则，使每个方向以与其出现频率相关的速率被遗忘，从而无需矩阵即可实现输入侧曲率校正。

    

    arXiv:2608.19491v1 公告类型：交叉 摘要：大多数现代优化器将动量形成过去梯度的指数移动平均（EMA），以固定速率遗忘每个方向。然而，深度网络在训练过程中看到的输入可能高度各向异性，少数方向频繁被查询，而大多数方向很少出现。近期方法通过在该缓冲区外增加额外处理来应对这种各向异性，但动量更新本身保持不变。我们提出DeltaMomentum，将方向感知构建到动量更新规则中。主要观察是线性层的梯度分解为作为键的输入和作为值的输出侧误差。利用键值结构，DeltaMomentum通过标准增量规则更新动量缓冲区，使每个方向以其出现频率设定的速率被遗忘。我们证明这是一种有效的动量，它无需矩阵即可应用输入侧曲率校正。

    arXiv:2608.19491v1 Announce Type: cross  Abstract: Most modern optimizers form their momentum as an exponential moving average (EMA) of past gradients, forgetting every direction at one fixed rate. However, the inputs a deep network sees during training can be highly anisotropic, with a few directions queried frequently while most are seen rarely. Recent methods address this anisotropy by wrapping extra processing around this buffer, leaving the momentum update itself unchanged. We propose DeltaMomentum, which builds direction-awareness into the momentum update rule. The main observation is that the gradient of a linear layer splits into an input that acts as a key and an output-side error that acts as a value. Exploiting the key-value structure, DeltaMomentum updates the momentum buffer by the canonical delta rule, so each direction is forgotten at a rate set by how often it appears. We prove that it is a valid momentum, that it applies the input-side curvature correction without matr
    
[^188]: 扩散逆问题的尺度一致后验动力学

    Scale-Consistent Posterior Dynamics for Diffusion Inverse Problems

    [https://arxiv.org/abs/2608.15144](https://arxiv.org/abs/2608.15144)

    本文提出一种尺度一致的后验动力学方法，通过重标定坐标、对数信噪比组织代理和冻结目标校正器，构建可处理的连续SDE，有效解决扩散逆问题中条件分数的难解性。

    

    arXiv:2608.15144v1 公告类型：交叉 摘要：使用预训练扩散先验进行后验采样，受条件分数控制，其中间似然分量通常难以处理。我们从理想的一参数后验SDE族出发，其中随机性参数控制概率流传输和随机探索，而不改变后验边缘分布。为了获得可处理的模型，我们在重标定的干净图像坐标中表达似然，并使用对数信噪比来组织所得的后验代理。通过前向算子投影扩散不确定性，得到噪声条件协方差路径，其目标接近干净后验。由于这些目标的端点一致性不能确保代理传输遵循它们，我们将传输与冻结目标的Langevin校正器交错，生成连续代理SDE。我们使用外部Lie--Trotter分裂和方差减少对此模型进行离散化。

    arXiv:2608.15144v1 Announce Type: cross  Abstract: Posterior sampling with a pretrained diffusion prior is governed by a conditional score whose intermediate likelihood component is generally intractable. We begin from an ideal one-parameter posterior SDE family in which a stochasticity parameter controls probability-flow transport and stochastic exploration without changing the posterior marginals. To obtain a tractable model, we express the likelihood in a rescaled clean-image coordinate and use log-SNR to organize the resulting posterior proxies. Projecting the diffusion uncertainty through the forward operator then yields a noise-conditioned covariance path whose targets approach the clean posterior. Because endpoint consistency of these targets does not ensure that a surrogate transport follows them, we interleave the transport with a frozen-target Langevin corrector, producing a continuous surrogate SDE. We discretize this model with an outer Lie--Trotter splitting and a variance
    
[^189]: 地球观测嵌入作为概率天气降尺度的有效亚网格描述符

    Earth observation embeddings are effective sub-grid descriptors for probabilistic weather downscaling

    [https://arxiv.org/abs/2608.12271](https://arxiv.org/abs/2608.12271)

    该论文提出利用地球观测基础模型生成的嵌入作为亚网格地表描述符，替代传统手工特征，从而提升概率天气降尺度对瞬时近地表变量的预测准确性。

    

    arXiv:2608.12271v1 公告类型：新 摘要：全球天气再分析和预报在粗网格上解析演变的天气状态，但特定地点的应用需要预测任意位置，其中近地表条件还取决于未解析的地形和地表特性。现有的概率降尺度方法使用手工制作的地形描述符来解决这一差距。我们转而探讨地球观测基础模型是否能提供可转移的亚网格地表表示，用于概率天气降尺度。我们增强了一个卷积条件神经过程，该过程将约25公里分辨率的粗ERA5再分析场进行降尺度，并加入一个学习到的局部地表描述符，该描述符通过压缩10米分辨率的TESSERA嵌入补丁获得。尽管这些嵌入在年际时间尺度上概括了地表条件，但它们通过编码持久的地表特性，改善了对瞬时2米温度和10米风速的降尺度效果。

    arXiv:2608.12271v1 Announce Type: new  Abstract: Global weather reanalyses and forecasts resolve the evolving atmospheric state on coarse grids, but site-specific applications require predictions at arbitrary locations where near-surface conditions also depend on unresolved terrain and land-surface properties. Existing probabilistic downscalers address this gap using hand-crafted topographic descriptors. We ask instead whether Earth observation foundation models can provide transferable sub-grid surface representations for probabilistic weather downscaling.   We augment a convolutional conditional neural process that downscales coarse ERA5 reanalysis fields at ~25 km resolution with a learned local surface descriptor, obtained by compressing a patch of TESSERA embeddings at 10 m resolution. Although these embeddings summarise surface conditions over annual timescales, they improve downscaling of instantaneous 2 m temperature and 10 m wind speed by encoding persistent surface properties
    
[^190]: WDL-OPD：通过混合约束协同训练实现的弱驱动在线策略蒸馏

    WDL-OPD: Weak-Driven On-Policy Distillation via Mixture-Constrained Co-Training

    [https://arxiv.org/abs/2608.09447](https://arxiv.org/abs/2608.09447)

    提出了WDL-OPD方法，通过锚定策略与辅助策略的双策略混合约束协同训练来稳定在线策略蒸馏的反馈回路，在Qwen3的1.7B和4B规模实验中取得了最优效果。

    

    在线策略蒸馏（OPD）通过在从学生模型自身采样的轨迹上将学生模型与教师模型对齐，减少了离线蒸馏中的训练-测试状态不匹配问题。然而，同样的反馈回路可能并不稳定：每次更新都会同时改变策略本身以及下一次更新所基于的状态。我们提出了WDL-OPD，这是一种包含两个可训练策略的混合约束协同训练方法。锚定策略生成每一次轨迹采样，辅助策略评估相同的已访问状态，并通过反向KL散度将两者token分布的几何混合与冻结的教师模型进行匹配。两个策略均接收梯度。我们证明，冻结辅助策略可以恢复出一个与OPD²和W2S-OPD密切相关的锚定加对比代理目标，而联合训练则创造了静态增量无法表达的分支级自由度。在1.7B和4B规模的Qwen3记录实验中，WDL-OPD产生了最强的性能表现（摘要在此处被截断）。

    arXiv:2608.09447v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) aligns a student with a teacher on trajectories sampled from the student itself, reducing the train-test state mismatch of offline distillation. The same feedback loop can nevertheless be unstable: each update changes both the policy and the states on which the next update is computed. We introduce WDL-OPD, a mixture-constrained co-training method with two trainable policies. An anchor policy generates every rollout, an auxiliary policy evaluates the same visited states, and a geometric mixture of their token distributions is matched to a frozen teacher by reverse KL. Both policies receive gradient. We show that freezing the auxiliary recovers an anchor-plus-contrast proxy target closely related to OPD$^2$ and W2S-OPD, whereas joint training creates branch-level degrees of freedom that a static delta cannot express. In recorded Qwen3 experiments at 1.7B and 4B scale, WDL-OPD produces the strongest s
    
[^191]: 基于Transformer增强PPO的软截止期感知大语言模型推理协同移动边缘计算学习方法

    Learning-Based Collaborative MEC for LLM Inference with Soft-Deadline Awareness via Transformer-Enhanced PPO

    [https://arxiv.org/abs/2608.02031](https://arxiv.org/abs/2608.02031)

    本文提出了一种基于Transformer增强PPO的协同移动边缘计算框架，结合受限的截止期扩展机制，在严格时延约束和任务依赖关系下高效调度大语言模型推理任务，从而提升服务质量。

    

    本文研究了在软截止期约束下用于大语言模型（LLM）推理的协同移动边缘计算（MEC）服务器。在该系统中，为了提高服务质量，计算任务被期望在其截止期内完成。然而，由于任务或子任务之间存在依赖关系，任何一次错过截止期都可能导致整个请求产生灾难性后果。在此背景下，本工作提出了一种具有受限灵活性的截止期扩展机制。主要挑战在于如何在严格的时延约束下处理大规模计算，同时限制允许的截止期扩展次数，尤其是在每个请求内部存在任务依赖的情况下。为应对这些挑战，我们开发了一种Transformer增强的近端策略优化（PPO）框架，使MEC服务器之间能够实现高效协作。所提出的方法旨在最大化任务（原文在此处截断）……

    arXiv:2608.02031v2 Announce Type: replace-cross  Abstract: This paper investigates collaborative mobile edge computing (MEC) servers for large language model (LLM) inference under soft deadline constraints. In this system, to improve the quality of service, computations are expected to be completed within their deadlines. However, due to dependencies among tasks or subtasks, any missed deadline can lead to catastrophic consequences for the entire request. In this context, this work proposes an extended deadline mechanism with constrained flexibility. The main challenges lie in handling large-scale computations under strict latency constraints while limiting the number of allowable deadline extensions, especially in the presence of task dependencies within each request. To tackle these challenges, we develop a transformer-enhanced proximal policy optimization (PPO) framework that enables efficient collaboration among MEC servers. The proposed approach aims to maximize the number of task
    
[^192]: DrainSinkhorn：批量熵最优传输的安全消除方法

    DrainSinkhorn: Safe Elimination for Batched Entropic Optimal Transport

    [https://arxiv.org/abs/2607.24741](https://arxiv.org/abs/2607.24741)

    提出 DrainSinkhorn，一种验证器门控的主动打包层，通过在批处理中动态淘汰已收敛的 Sinkhorn 问题来精确消除静态批处理的冗余计算，从而在完全保持 EOT 目标、Sinkhorn 映射与停止规则不变的前提下加速批量熵最优传输。

    

    快速的熵最优传输后端虽然降低了每次 Sinkhorn 更新的成本，但静态批处理仍会以全宽度运行，直到最慢的问题完成为止。我们提出 DrainSinkhorn，一种面向独立 Sinkhorn 问题批处理的验证器门控主动打包层。它结合了候选轴打包、Sinkhorn 特定的单侧筛选、在后端配置的双侧残差检查下的验证器门控退出机制，以及对所有以候选索引的状态的物理压缩。EOT 目标函数、每个实例的 Sinkhorn 映射和停止规则均保持不变；后续内核仅在未完成的问题上运行。我们精确刻画了可移除的工作量：如果打包窗口内各问题的完成深度不同，主动执行会消除静态批处理矩形与观察到的存活曲线之间的填充。基于商空间非线性 Perron-Frobenius 分析，我们为这些有限容差下的深度差异提供了局部解释：收敛……

    arXiv:2607.24741v3 Announce Type: replace-cross  Abstract: Fast entropic optimal transport backends reduce the cost of each Sinkhorn update, but static batches still run at full width until the slowest problem finishes. We introduce DrainSinkhorn, a verifier-gated active-packing layer for batches of independent Sinkhorn problems. It combines candidate-axis packing, a Sinkhorn-specific one-sided screen, verifier-gated retirement under the backend's configured two-sided residual check, and physical compaction of all candidate-indexed state. The EOT objective, per-instance Sinkhorn map, and stopping rule are unchanged; later kernels run only on unfinished problems.   We characterize the removable work exactly. If completion depths differ within a packed window, active execution removes the padding between the static batch rectangle and the observed survival curve. A quotient nonlinear Perron-Frobenius analysis gives a local explanation for these finite-tolerance depth differences: converg
    
[^193]: ROMS-IMLE：一种实现有竞争力单步生成建模的极简方法

    ROMS-IMLE: A Minimalist Approach to Competitive Single-Step Generative Modelling

    [https://arxiv.org/abs/2607.19332](https://arxiv.org/abs/2607.19332)

    该论文提出ROMS-IMLE，一种极简的单步生成建模方法，通过简单地结合隐式最大似然估计（IMLE）训练目标与简洁的模型结构，摒弃扩散模型等复杂的多步渐进变换机制，依然实现了有竞争力的生成性能。

    

    生成模型经历了多代演化，从VAE/GAN到扩散模型/流匹配。在此过程中，底层技术变得日益复杂，各种关于什么因素能驱动强大实证表现的观点也逐渐形成。由于扩散模型和流匹配的成功，一个较为普遍的观点是通过许多小的变换逐步将噪声分布转化为数据分布非常重要。我们质疑这是否真的必要，并采用极简方法设计了一个具有竞争力的生成模型。我们从最基本的核心要素出发，即仅包含一个训练目标和一个模型。我们有目的地让两者都保持简单。对于训练目标，我们选择隐式最大似然估计（IMLE），并摒弃更复杂的替代方案，如变分推断、对抗训练和数值积分。对于模型，我们摒弃更复杂的……

    arXiv:2607.19332v2 Announce Type: replace  Abstract: Generative models have undergone many generations of evolution, from VAEs/GANs to diffusion/flow matching. Along the way, the underlying techniques have become more complicated and various beliefs about what drives strong empirical performance have taken hold. Due to the success of diffusion models and flow matching, one of the more common beliefs is the importance of transforming the noise distribution to the data distribution gradually through many small transformations. We ask whether this is truly necessary, and take a minimalist approach to designing a competitive generative model. We start with the bare-bones essentials, namely just a training objective and a model. We purposefully make both simple. For the training objective, we choose Implicit Maximum Likelihood Estimation (IMLE), and eschew more complicated alternatives such as variational inference, adversarial training and numerical integration. For the model, we eschew tr
    
[^194]: 询问两次，观察两次：提示回声解决视觉语言模型中的“问题优先悖论”

    Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models

    [https://arxiv.org/abs/2607.15565](https://arxiv.org/abs/2607.15565)

    研究揭示了视觉语言模型中“问题优先悖论”的机制——虽然前置问题能引导感知，但被数百个图像token遮挡的问题无法被答案token读取，并据此提出在图像后重复问题的“提示回声”这一无需训练的简单修复方法。

    

    在视觉语言模型（VLM）的提示中，问题应该放在图像之前还是之后？直觉告诉我们应该放在前面：知道要问什么应该能引导模型关注正确的位置。然而，在各个视觉问答基准测试中，问题优先的提示方式始终表现不如前沿VLM所推荐的图像优先排序，我们将这一现象称为“问题优先悖论”。我们将这一悖论追溯到VLM计算两个阶段之间的冲突。通过Logit-lens和注意力探针分析显示，问题优先的提示确实会引导感知，使图像块表示向问题相关概念偏移。但在下游计算中，被困在数百个图像token之后的问题几乎没有被答案token注意到，答案token转而依赖图像驱动的、往往是错误的答案。因果注意力消融实验证实，只有当问题位于图像之后时，答案token才会读取问题。这一诊断带来了一种无需训练的修复方法：问题回声（即在图像之后重复问题）……

    arXiv:2607.15565v2 Announce Type: replace-cross  Abstract: Where should the question go in a vision-language model (VLM) prompt: before the image or after it? Intuition says before: knowing what is asked should tell the model where to look. Yet across visual question answering benchmarks, question-first prompting consistently underperforms the image-first ordering recommended for frontier VLMs, a phenomenon we term the question-first paradox. We trace this paradox to a conflict between two stages of VLM computation. Logit-lens and attention probes show that question-first prompting steers perception, shifting image patch representations toward question-relevant concepts. But downstream, stranded behind hundreds of image tokens, the question is barely attended by the answer token, which instead commits to image-driven, often wrong answers. Causal attention knockout confirms that the answer reads the question only when it follows the image. This diagnosis yields a training-free fix: ques
    
[^195]: 弯曲权重空间中的学习：用于改进优化的指数-线性权重重参数化

    Learning in Curved Weight Space:Exponential-Linear Weight Reparameterization for Improved Optimization

    [https://arxiv.org/abs/2607.09967](https://arxiv.org/abs/2607.09967)

    提出一种将对称指数路径与线性路径相结合的权重重参数化方法，使加性优化更新转化为与权重幅值成比例的有效变化，从而改善神经网络的优化效果。

    

    许多神经网络操作本质上具有乘性而非加性：将范数减半或加倍在相对意义上是类似操作，但若采用线性步骤则需要不相等的优化距离。诸如Adam之类的自适应优化器会对每个坐标进行归一化更新，但更新步骤仍然是加性的；幅值差异很大的权重会接收到大小相近的绝对变化，从而产生差异巨大的相对扰动。我们提出了\method（\methodshort），这是一种针对神经网络的权重重参数化方法，它将带符号感知的对称指数路径与类恒等映射的线性路径相结合。对称指数路径对于较小的原始权重近似线性，但在较大幅值处曲率逐渐增大。对数空间中的加性更新可映射为有效权重空间中与幅值成比例的变化。线性路径则提供了穿越该变换的直接通道，我们假设……

    arXiv:2607.09967v3 Announce Type: replace-cross  Abstract: Many neural networks operations have a multiplicative nature rather than additive: halving or doubling a norm are analogous relatively but require unequal optimization distances when taking linear steps. Adaptive optimizers such as Adam normalize updates per coordinate, but update steps remain additive; weights with very different magnitudes receive similarly sized absolute changes, producing very different relative perturbations. We introduce \textbf{\method} (\textbf{\methodshort}), a weight reparameterization for neural networks that combines a sign-aware symmetric-exponential pathway with an identity-like linear pathway. The symmetric-exponential pathway is near-linear for small raw weights but increasingly curved at larger magnitudes. Additive updates in logarithmic space map to magnitude-proportional changes in effective weight space. The linear pathway provides a direct route through the transform that we hypothesize sta
    
[^196]: 重新采样还是重新路由？无需已识别动作选择下的可恢复停止债务

    Resample or Reroute? Recoverable Stopping Debt Without Identified Action Selection

    [https://arxiv.org/abs/2607.08665](https://arxiv.org/abs/2607.08665)

    该论文提出“可恢复停止债务”概念并通过三个排序门槛量化弱验证器误判后的恢复潜力，实验表明固定升级到更大模型比重新采样能带来更大恢复，而常用的全回合审计统计量不包含任何可观测历史信息。

    

    arXiv:2607.08665v3 公告类型：替换。摘要：在弱验证器接受大语言模型的响应后，第二次调用可以选择重新采样或重新路由。由于正确性是隐藏的，动作选择成为一个识别问题。我们对三个门槛进行排序：可恢复停止债务、双侧FIT动作支持，以及来自结果盲选择器的留出价值。在一个固定的152个查询MBPP+实验中，Qwen2.5-14B仅基于Base模型的假阳性停止会给Qwen2.5-7B留下+2.592个百分点的可恢复空间（查询聚类95%区间为[+1.618, +3.664]）。另一方面，在7B Base测试被拒绝后，固定升级到14B比留一法7B重采样高出+2.882个百分点[+0.931, +5.201]；这是固定动作排名，而非条件选择。全回合审计产生了+2.697个百分点的实现最大值差距，但对于两个动作的情形，该统计量等于(1/2)E|Δ| - (1/2)|EΔ|，不包含任何可观测历史项。它位于精确折叠可交换参考之内（均值+3.158；95%区间……）

    arXiv:2607.08665v3 Announce Type: replace  Abstract: After a weak verifier accepts a large-language-model response, a second call may resample or reroute. Because correctness is hidden, action selection is an identification problem. We order three gates: recoverable stopping debt, two-sided FIT action support, and held-out value from an outcome-blind selector. In a pinned 152-query MBPP+ experiment, a Qwen2.5-14B Base-only false-positive stop leaves +2.592 percentage points of Qwen2.5-7B recovery (query-cluster 95% interval [+1.618, +3.664]). Separately, after 7B Base-test rejection, fixed escalation to 14B exceeds leave-one-out 7B resampling by +2.882 points [+0.931, +5.201]; this is fixed-action ranking, not conditional selection. An all-episode audit produces a +2.697-point realized-maximum gap, but for two actions this statistic equals (1/2)E|Delta| - (1/2)|E Delta| and contains no observable-history term. It lies inside an exact-fold exchangeable reference (mean +3.158; 95% interv
    
[^197]: 面向物理信息神经网络反问题的目标引导选择性重加权：一种迁移学习方法

    Target-Guided Selective Reweighting for Physics-Informed Neural Network Inverse Problems: A Transfer Learning Approach

    [https://arxiv.org/abs/2607.05271](https://arxiv.org/abs/2607.05271)

    提出TGSR-PINN方法，将迁移学习与基于目标证据的神经元敏感度评分和选择性重加权相结合，解决了物理信息神经网络在偏微分方程反问题中因负迁移导致的物理参数恢复不准确问题。

    

    物理信息神经网络（PINNs）在偏微分方程（PDE）反问题中常常面临不适定优化、损失函数相互竞争以及参数补偿等问题。迁移学习可以复用源任务的特征表示，但当源任务与目标任务的物理特性不同时，直接微调可能引发负迁移，导致场误差较低但参数恢复不准确。为解决这一问题，我们提出了目标引导选择性重加权PINN（TGSR-PINN），这是一种基于目标证据驱动的PINN反问题迁移学习表示修正方法。TGSR-PINN迁移源网络的权重和偏置，但独立初始化目标物理参数。经过短时间的目标适应后，该方法在固定批次上利用一阶泰勒敏感度和预激活方差对神经元进行评分。这些评分通过带秩回退机制的高斯混合模型转换为连续的弱适应信号。TGSR-PINN然后……（注：原文摘要在此处截断）

    arXiv:2607.05271v2 Announce Type: replace  Abstract: Physics-informed neural networks (PINNs) often face ill-posed optimization, competing losses, and parameter compensation in partial differential equation (PDE) inverse problems. Transfer learning can reuse source-task representations, but direct fine-tuning may induce negative transfer when source and target physics differ, leading to low field error but inaccurate parameter recovery. To address this issue, we propose Target-Guided Selective Reweighting PINN (TGSR-PINN), a target-evidence-driven representation correction method for PINN inverse transfer learning. TGSR-PINN transfers source network weights and biases but initializes target physical parameters independently. After short target adaptation, it scores neurons using first-order Taylor sensitivity and pre-activation variance on fixed batches. These scores are converted into continuous weak-adaptation signals using a Gaussian mixture model with rank fallback. TGSR-PINN then 
    
[^198]: 所见即所得：面向图表到代码生成的观察对齐监督

    What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation

    [https://arxiv.org/abs/2607.04726](https://arxiv.org/abs/2607.04726)

    论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。

    

    图表到代码生成通常通过对参考绘图脚本进行监督微调来训练，这隐式地将黄金代码视为完全可观察的目标。然而，许多图表程序包含无法从渲染图像中唯一恢复的潜在变量。我们在五种图表类型中识别出这种潜在变量与观察不匹配问题的四种形式：聚合导致的不匹配，即原始样本被简化为箱线图统计量或直方图分箱统计；归一化导致的不匹配，即饼图中绝对尺度被移除；投影导致的不匹配，即三维信息在二维渲染中丢失；以及水平集导致的不匹配，即标量场只能通过选定的等高线被观察。这些不匹配引入了目标歧义，并要求模型生成图像本身无法支持的信息。我们提出观察对齐监督方法，用视觉上可约束的量来替换潜在变量。

    arXiv:2607.04726v4 Announce Type: replace  Abstract: Chart-to-code generation is commonly trained through supervised fine-tuning on reference plotting scripts, implicitly treating the gold code as a fully observable target. However, many chart programs contain latent variables that cannot be uniquely recovered from the rendered image. We identify this latent-observation mismatch in four forms across five chart types: aggregation-induced mismatch, where raw samples are reduced to box statistics or histogram bin masses; normalization-induced mismatch, where absolute scale is removed in pie charts; projection-induced mismatch, where 3D information is lost through 2D rendering; and level-set-induced mismatch, where a scalar field is observable only through selected contour lines. These mismatches introduce target ambiguity and require models to generate information unsupported by the image. We propose Observation-Aligned Supervision, which replaces latent variables with visually constraine
    
[^199]: KARMA：基于知识图谱的自动推理具体化与对齐

    KARMA: Knowledge graph-based Automated Reasoning Materialization and Alignment

    [https://arxiv.org/abs/2607.03166](https://arxiv.org/abs/2607.03166)

    KARMA 通过在领域知识图谱上枚举模式约束路径生成槽位对齐的对比候选样本，并利用槽位并行对齐（SPA）将偏好监督精准路由至区分性实体槽位，从而解决了基于模板的对比合成中的分辨率不匹配问题。

    

    基于模板的对比合成具有良好的可扩展性，但其候选样本往往仅在少数实体槽位上存在差异，而序列级优化会将监督信号分散到大部分共享的模板上。我们将这一问题形式化为“分辨率不匹配问题”，并提出 KARMA 方法，该方法在领域知识图谱上枚举受模式约束的路径，并将其言语化为槽位对齐的对比候选样本。随后，槽位并行对齐应用解耦的槽位级目标函数，将偏好监督精准引导至具有区分性的实体槽位，其中槽位感知的掩码注意力可作为打包评估的可选实现。在生物医学、计算机科学和化学基准测试中，KARMA 优于基础 LLM 和相同数据下的 SFT 基线，并与序列级和词元级偏好方法相比表现更优。

    arXiv:2607.03166v2 Announce Type: replace-cross  Abstract: Template-based contrastive synthesis is scalable, but its candidates often differ only in a few entity-slots while sequence-level optimization spreads supervision over mostly shared templates. We formalize this as the Resolution Mismatch Problem and propose KARMA, which enumerates schema-constrained paths over domain knowledge graphs and verbalizes them into slot-aligned contrastive candidates. Slot-Parallel Alignment (SPA) then applies a decoupled slot-level objective to route preference supervision to discriminative entity-slots, with slot-aware masked attention serving as an optional packed-evaluation implementation. Across biomedical, computer-science, and chemistry benchmarks, KARMA outperforms base LLM and same-data SFT baselines, and compares favorably with sequence- and token-level preference methods.
    
[^200]: SNAP-FM：面向物理约束生成建模的稀疏非线性加速投影

    SNAP-FM: Sparse Nonlinear Accelerated Projection for Physics-Constrained Generative Modeling

    [https://arxiv.org/abs/2607.00095](https://arxiv.org/abs/2607.00095)

    提出SNAP-FM方法，利用样本批处理与局部PDE耦合所诱导的块稀疏结构，实现高效的批量非线性投影优化，使生成模型在推理时能精确满足物理守恒约束且计算开销大幅降低。

    

    生成模型已成为物理模拟的可扩展替代方案，但其输出无法保证遵守支配底层物理的守恒定律、边界条件和非线性不变量。约束采样可以填补这一空白，它在推理阶段精确执行此类约束而无需重新训练，但需要付出计算代价：投影、校正和轨迹优化步骤在采样过程中被反复执行，对于非线性约束而言，这些步骤的计算开销十分高昂。标准机器学习框架进一步加剧了这一问题：其稠密张量代数和有限的稀疏求解器可组合性掩盖了物理约束天然诱导的结构，使得高效的批量非线性优化在实践中难以实现。我们通过利用样本级批处理和局部PDE耦合在投影子问题中诱导的结构来解决这一瓶颈——即块稀疏结构（原文摘要在此处截断）。

    arXiv:2607.00095v2 Announce Type: replace-cross  Abstract: Generative models have emerged as scalable surrogates for physical simulation, yet they offer no guarantee that their outputs respect the conservation laws, boundary conditions, and nonlinear invariants that govern the underlying physics. Constrained sampling closes this gap, enforcing such constraints exactly at inference time without retraining, but at a computational cost: projection, correction and trajectory-optimization steps are repeated during sampling, with these steps becoming expensive for nonlinear constraints. Standard ML frameworks exacerbate this: their dense tensor algebra and limited sparse solver composability obscure the structure that physical constraints naturally induce, making efficient batched nonlinear optimization difficult to realize in practice. We address this bottleneck by exploiting the structure that sample-wise batching and local PDE couplings induce in the projection subproblems -- namely, bloc
    
[^201]: 学习选择而非重新学习：硬路由的推理LoRA混合模型

    Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs

    [https://arxiv.org/abs/2606.31413](https://arxiv.org/abs/2606.31413)

    提出Hard-Routed MoR-LoRA两阶段框架，通过单位尺度的硬top-1路由（而非软加权组合）选择冻结的推理LoRA专家，仅训练轻量级共享路由器和小型注意力LoRA即可实现多领域推理能力的集成。

    

    将独立训练的LoRA适配器组合成单个大语言模型对多领域适应非常有用，尤其适用于原始训练数据无法共享的场景。一种常见做法是对LoRA专家采用MoE风格的路由，但对于已冻结的预训练适配器，软加权组合可能会改变每个LoRA模块最初训练时所基于的单位尺度加性更新。我们提出了硬路由MoR-LoRA（Hard-Routed MoR-LoRA），这是一个通过单位尺度硬选择来组合冻结推理LoRA专家的两阶段框架。首先，使用可验证反馈的强化学习独立训练各领域的LoRA适配器，以获得推理专家；然后，冻结所有专家，从中蒸馏推理轨迹，仅训练一个轻量级共享路由器和一个小的注意力LoRA用于集成。路由器采用硬top-1路由，为每个token精确选择一个专家……

    arXiv:2606.31413v3 Announce Type: replace  Abstract: Composing independently trained LoRA adapters into a single large language model is useful for multi-domain adaptation, especially when the original training data cannot be shared. A common approach is to use MoE-style routing over LoRA experts, but for frozen pretrained adapters, soft weighted combinations can change the unit-scale additive update under which each LoRA module was originally trained. We propose \textbf{Hard-Routed MoR-LoRA}, a two-stage framework for composing frozen reasoning LoRA experts through unit-scale hard selection. First, domain-specific LoRA adapters are trained independently using reinforcement learning from verifiable feedback to obtain reasoning experts. Then, all experts are frozen, reasoning traces are distilled from them, and only a lightweight shared router together with a small attention LoRA is trained for integration. The router selects exactly one expert per token using hard top-1 routing, while 
    
[^202]: 变换器作为贝叶斯上下文实验者：平滑度自适应的高效平均处理效应估计

    Transformers as Bayesian In-Context Experimenters: Smoothness-Adaptive Efficient ATE Estimation

    [https://arxiv.org/abs/2606.31184](https://arxiv.org/abs/2606.31184)

    该论文提出将变换器训练为模仿贝叶斯后验Neyman教师的“上下文实验者”，通过上下文学习摊销序贯方差估计与处理分配过程，实现对平均处理效应的平滑度自适应高效估计。

    

    用于平均处理效应（ATE）的自适应实验需要随机化分配，以在有效推断与统计效率之间取得平衡。理想（oracle）设计是一个由未知的分组条件结果方差所支配的、依赖于协变量的Neyman规则。我们研究了这种序贯的方差估计与分配过程能否通过上下文学习被摊销。我们提出了贝叶斯上下文实验者：即经过训练以模仿贝叶斯后验Neyman教师的变换器策略。该教师利用实验历史更新关于潜在结果的非参数信念，从而分配后验Neyman处理概率。该设计收敛于oracle规则，支持高效的ATE推断。变换器通过基于注意力的充分统计量和投影梯度下降建设性地实现了这一映射，模仿了高斯级先验下的贝叶斯更新。为应对未知的结果平滑度……（原文在此处截断）

    arXiv:2606.31184v2 Announce Type: replace-cross  Abstract: Adaptive experiments for average treatment effects (ATE) require randomized allocations balancing valid inference with statistical efficiency. The oracle design is a covariate-dependent Neyman rule governed by unknown arm-conditional outcome variances. We investigate whether this sequential variance-estimation and allocation process can be amortized via in-context learning. We introduce Bayesian in-context experimenters: transformer policies trained to imitate a Bayesian posterior Neyman teacher. The teacher updates nonparametric beliefs over potential outcomes using experimental history to assign posterior Neyman treatment probabilities. This design converges to the oracle rule, supporting efficient ATE inference. Transformers constructively implement this mapping through attention-based sufficient statistics and projected gradient descent, imitating Bayesian updating for Gaussian-series priors. To address unknown outcome smoo
    
[^203]: 民主化逆向宪法AI：通过辩论从偏好中提炼引导原则

    Democratic ICAI: Debating Our Way to Steering Principles from Preferences

    [https://arxiv.org/abs/2606.28294](https://arxiv.org/abs/2606.28294)

    本文提出民主化逆向宪法AI（DICAI），通过结构化角色辩论收集多元竞争性推理依据，从中提炼出更清晰全面的引导原则，以改进偏好决策建模和下游模型训练。

    

    基于偏好的对齐方法常常难以捕捉人类判断背后的推理过程。许多评估依赖于多个相互作用的准则，然而成对比较标签仅揭示了最终选择，而非塑造偏好的考量因素。逆向宪法AI（ICAI）通过将偏好总结为自然语言原则，提高了决策的可解释性，但其单次生成的解释遗漏了复杂决策中的许多细微差别。我们提出了民主化逆向宪法AI（DICAI），这是一种新颖的方法，通过结构化的角色辩论收集多个相互竞争的推理依据，为影响每次比较的因素提供了更广泛、更具表达性的描述。基于这些更丰富的信号，我们推导出更清晰、更全面的引导原则，并将其用于指导基于大语言模型和决策树的偏好决策建模，以及通过宪法原则进行的下游模型训练。

    arXiv:2606.28294v2 Announce Type: replace  Abstract: Preference-based alignment often struggles to capture the reasoning that underlies human judgments. Many evaluations rely on multiple interacting criteria, yet pairwise labels reveal only the final choice rather than the considerations that shape preferences. Inverse Constitutional AI (ICAI) improves interpretability in decision making by summarizing preferences into natural-language principles, but its single-pass explanations miss much of the nuance involved in complex decisions. We introduce Democratic ICAI (DICAI), a novel approach that gathers multiple competing rationales through structured persona debate, offering a broader and more expressive account of the factors influencing each comparison. From these richer signals, we derive clearer and more comprehensive steering principles and use them to guide preference decision modeling through both LLM-based and decision-tree judges, as well as downstream model training via constit
    
[^204]: 物理引导的机器人在非结构化环境中沿任意测量路径的辐射源定位

    Physics-Guided Robotic Radiation Source Localization along Arbitrary Measurement Paths in Unstructured Environments

    [https://arxiv.org/abs/2606.27624](https://arxiv.org/abs/2606.27624)

    该论文提出了一种基于物理信息机器学习的自动化框架，使机器人能够在未知非结构化环境中沿任意测量路径精确估计辐射源位置，无需接近辐射源，从而降低辐射损害风险并提升任务部署的灵活性。

    

    利用机器人估计辐射源的位置是提高效率和安全性的一种有效方法。现有方法侧重于规划机器人的路径以实现精确估计，通常需要接近辐射源。然而，接近辐射源会增加机器人受到辐射损害的风险。此外，专门为辐射源定位（RSL）设计的路径规划算法限制了在放射性环境中部署机器人执行任务的灵活性。本研究提出了一种用于机器人辐射源定位的自动化框架，该框架利用物理信息机器学习（PIML）模型，在未知环境中无论采用何种测量路径都能精确估计辐射源位置。该研究为PIML设计了物理启发的模型张量，以处理来自未知障碍物的衰减伽马射线通量信号，并通过并行计算多个模型来提高辐射源定位的鲁棒性和精度。

    arXiv:2606.27624v2 Announce Type: replace-cross  Abstract: Using robots to estimate the location of the radiation source is an effective way to improve efficiency and safety. Existing methods focus on planning the robot's path to achieve precise estimation, typically approaching the source. However, approaching the source increases the risk of radiation damage to a robot. In addition, a path-planning algorithm designed solely for radiation source localization (RSL) limits the flexibility of missions that deploy robots into radioactive environments. This study presents an automation framework for robotic RSL that leverages a physics-informed machine learning (PIML) model to precisely estimate the source location, regardless of measurement paths, in unknown environments. Physics-inspired model tensors have been designed for PIML to handle attenuated gamma-ray flux signals from unknown obstacles, and multiple models are computed in parallel to improve the robustness and precision of the R
    
[^205]: 神经算子的实数与复数频谱基：格林函数对齐的作用

    Real vs. Complex Spectral Bases for Neural Operators: The Role of Green's Function Alignment

    [https://arxiv.org/abs/2606.24851](https://arxiv.org/abs/2606.24851)

    本文提出哈特莱神经算子（HNO），用纯实数的离散哈特莱变换替代复数FFT，在与FNO参数量相同的情况下消除共轭对称冗余，并论证最佳频谱基由算子格林函数的性质决定（自伴椭圆算子适合实数基）。

    

    傅里叶神经算子（FNO）通过在复数傅里叶域中参数化全局卷积来学习偏微分方程的解算子。对于实值的偏微分方程解，复数FFT因共轭对称性而携带表示冗余。我们提出了哈特莱神经算子（HNO），它是FNO的精确实值镜像：它用纯实数的离散哈特莱变换取代FFT，并为每个保留的频谱模式学习单个实数乘子，无需任何复数运算。由于实数哈特莱频谱不会因共轭对称性而被减半，HNO保留的频率角点是FNO的两倍，但每个角点只有一个实数权重，而FNO携带一对复数权重，因此两个算子在相同宽度下是等参数的，仅在频谱基上有所不同。我们的核心论点是：最佳基是算子本身的属性。自伴椭圆算子（泊松、双调和算子）具有实数的、对称的格林函数……

    arXiv:2606.24851v4 Announce Type: replace  Abstract: Fourier Neural Operators (FNO) learn solution operators of partial differential equations by parameterizing global convolutions in the complex Fourier domain. For real-valued PDE solutions, the complex FFT carries representational redundancy through conjugate symmetry. We introduce the Hartley Neural Operator (HNO), the exact real-valued mirror of FNO: it replaces the FFT with the purely real Discrete Hartley Transform and learns a single real multiplier per retained spectral mode, with no complex arithmetic. Because the real Hartley spectrum is not halved by conjugate symmetry, HNO retains twice as many frequency corners as FNO but one real weight where FNO carries a complex pair, so the two operators are iso-parametric at equal width and differ only in spectral basis. Our central thesis is that the best basis is a property of the operator. Self-adjoint elliptic operators (Poisson, biharmonic) have real, symmetric Green's functions 
    
[^206]: 基于阻尼振荡频谱门控的自适应隐式神经表示

    Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations

    [https://arxiv.org/abs/2606.23129](https://arxiv.org/abs/2606.23129)

    该论文提出将神经元激活建模为正弦强迫阻尼谐振子的稳态响应，通过联合优化振子参数与网络权重实现自适应频谱门控，无需显式正则化即可解决隐式神经表示中细节捕捉与噪声抑制之间的频谱权衡问题。

    

    隐式神经表示（INRs）已被证明能够通过基于坐标的网络成功编码连续信号，但面临一个频谱困境：周期激活函数能够捕捉精细细节，却如同全通滤波器般记忆噪声；而空间紧凑的激活函数虽然能有效正则化，却存在低频偏差问题。现有解决这一权衡的方法要么引入额外计算开销，要么存在调参脆弱性。我们提出将每个神经元的激活建模为正弦强迫阻尼谐振子的稳态响应，其振幅在训练过程中自然地调控网络的频谱选择性。通过将振子参数与网络权重进行联合优化，我们的方法无需显式正则化即可自适应目标信号的频谱内容。网络在阻带中初始化后，展现出由粗到细的学习课程，逐步扩展……（摘要原文至此截断）

    arXiv:2606.23129v3 Announce Type: replace-cross  Abstract: Implicit Neural Representations (INRs) have been proven successful in encoding continuous signals through coordinate-based networks, yet facing a spectral dilemma: periodic activations capture fine details but act as all-pass filters that memorise noise, while spatially compact activations regularise effectively but suffer from low-frequency bias. Existing attempts to resolve this trade-off introduce computational overhead or tuning frailty. We propose to model each neuron's activation as the steady-state response of a sinusoidally-forced damped harmonic oscillator, whose amplitude naturally governs the network's spectral selectivity during training. By jointly optimising the oscillator parameters alongside the network weights, our method adapts to the target signal's spectral content without explicit regularisation. Initialised in the stopband, the network exhibits a coarse-to-fine learning curriculum that progressively expand
    
[^207]: 大语言吉布斯的结构化推断

    Structured Inference with Large Language Gibbs

    [https://arxiv.org/abs/2606.19264](https://arxiv.org/abs/2606.19264)

    提出了Large Language Gibbs方法，将LLM的条件分布作为MCMC的转移算子，通过在其他变量条件下迭代重采样单个变量来实现结构化概率推断，从而避免了自回归生成中的顺序依赖偏差。

    

    大语言模型（LLM）中编码的知识可以作为对描述复杂世界的变量进行结构化推理的基础，但以概率上连贯的方式获取这些知识是一个困难的推断问题。我们提出了大语言吉布斯，这是一种结构化概率推断方案，它将LLM的条件分布用作转移算子。我们不是通过单次自回归生成来采样结构化对象，而是利用LLM的下一个词元条件分布，在其他变量的条件下迭代地重采样各个变量。这种方法避免了顺序依赖的偏差，并产生一个反映所有局部条件之间折中的平稳分布。我们将该方法应用于从合成分布中采样、一致性推理任务以及贝叶斯结构学习。结果表明，在MCMC中使用LLM条件分布是一种实用的

    arXiv:2606.19264v2 Announce Type: replace-cross  Abstract: The knowledge encoded in large language models (LLMs) can serve as a substrate for structured reasoning over variables describing a complex world, but accessing this knowledge in a probabilistically coherent manner poses a difficult inference problem. We propose Large Language Gibbs, a scheme for structured probabilistic inference that uses conditional distributions of an LLM as transition operators. Rather than sampling structured objects through single-pass autoregressive generation, we iteratively resample individual variables conditioned on others using an LLM's next-token conditionals. This approach avoids order-dependent biases and produces a stationary distribution that reflects a compromise between all local conditionals. We apply this approach to sampling from synthetic distributions, consistent reasoning tasks, and Bayesian structure learning. The results suggest that the use of LLM conditionals in MCMC is a practical
    
[^208]: 面向动态学习的显式交互架构：结构归纳偏置的受控研究

    Explicit Interaction Architectures for Dynamical Learning: A Controlled Study of Structural Inductive Bias

    [https://arxiv.org/abs/2606.19101](https://arxiv.org/abs/2606.19101)

    该论文提出由有序局部状态调制变换构成的因果循环单元，并通过受控实验检验这种显式设计的交互架构相比通用回声状态网络能否为动态学习提供有用的结构归纳偏置。

    

    我们研究一种“结构优先”的动态学习方法，其中有状态交互的组织方式被显式地加以规定，而不是完全交由通用的循环参数化来处理。我们引入了因果循环单元，它由有序的局部状态调制变换序列构建而成。该构造受基于波的交互模型启发，但本文研究的单元并不施加散射、无源性或能量平衡等约束。鉴于固定的循环动力学、经设计的储备池拓扑、仅读取层学习以及循环深度等问题都已被充分研究，本文的实证问题被有意限定得更窄：在受控的计算条件下，所提出的交互组织方式能否提供一种有用的归纳偏置？我们比较了一层结构化模型、两层结构化模型以及通用的回声状态网络（ESN），三者均具有12个循环状态和相同的严格计算预算。

    arXiv:2606.19101v2 Announce Type: replace-cross  Abstract: We investigate a structure-first approach to dynamical learning in which the organization of stateful interactions is prescribed explicitly rather than left entirely to a generic recurrent parameterization. We introduce causal recurrent units built from an ordered sequence of local, state-modulated transformations. The construction is motivated by wave-based interaction models, but the units studied here do not impose scattering, passivity, or energy-balance constraints.   Because fixed recurrent dynamics, designed reservoir topologies, readout-only learning, and recurrent depth are already well established, the empirical question is deliberately narrower: does the proposed interaction organization provide a useful inductive bias under controlled computational conditions? We compare a one-layer structured model, a two-layer structured model, and a generic echo-state network (ESN), all with 12 recurrent states and the same stric
    
[^209]: LLMZero：通过大语言模型智能体发现强化学习后训练的自适应训练策略

    LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents

    [https://arxiv.org/abs/2606.18388](https://arxiv.org/abs/2606.18388)

    LLMZero利用大语言模型智能体结合树搜索，通过在每个检查点诊断训练状态来自适应地优化RL后训练的多参数调度策略，在四个GRPO任务上比基础模型提升9%-140%、比网格搜索提升6%-15%，并揭示了容量参数单调累积、正则化参数震荡变化的训练规律。

    

    强化学习（RL）后训练策略依赖于数据集，并呈现出一个反复出现的经验规律：容量参数在各阶段单调累积，而正则化参数则主要随着训练动态的变化而震荡。这一区别凸显了固定训练调度的一个潜在缺陷：由于强迫所有参数沿着僵化的路径变化，固定调度无法捕捉正则化所必须追踪的动态探索-利用权衡。我们通过LLMZero揭示了这一点——LLMZero是一个通过树搜索优化训练轨迹的智能体系统，它在每个检查点诊断训练中的病理现象，并提出协调的多参数转换方案。在四个多样化的GRPO任务中，LLMZero发现的策略相比基础模型提升了9%至140%，相比网格搜索相对提升了6%至15%，在相同计算预算下，其表现始终优于随机搜索和基于技能的智能体。

    arXiv:2606.18388v2 Announce Type: replace-cross  Abstract: RL post-training strategies are dataset-dependent and reveal a recurring empirical pattern: capacity parameters accumulate monotonically across stages, while regularization parameters predominantly oscillate in response to shifting training dynamics. This distinction highlights a potential flaw in fixed training schedules: by forcing all parameters along rigid paths, they fail to capture the dynamic exploration-exploitation tradeoffs that regularization must track. We uncover this through LLMZero, an agentic system that optimizes training trajectories via tree search by diagnosing pathologies at each checkpoint and proposing coordinated multi-parameter transitions. Across four diverse GRPO tasks, LLMZero discovers strategies that improve over the base model by 9% to 140% and over grid search by 6% to 15% (relative), consistently outperforming random search and a skill-based agent under a matched compute budget. The capacity--re
    
[^210]: ThousandWorlds：潜在宜居系外行星气候模拟的基准测试

    ThousandWorlds: A benchmark for climate emulation of potentially habitable exoplanets

    [https://arxiv.org/abs/2606.18338](https://arxiv.org/abs/2606.18338)

    ThousandWorlds是一个机器学习就绪的系外行星气候模拟基准数据集，包含来自五个全球气候模型的约1800次模拟，旨在突破传统气候模拟的计算瓶颈，加速对潜在宜居系外行星大气的理解与生命信号解读。

    

    寻找地球以外的生命将依赖于探测潜在宜居系外行星大气中的微弱信号。解读这些信号需要理解宿主行星的气候：同一种分子在一颗行星上可能预示生命存在，而在另一颗行星上则可能是非生物化学过程的产物。全球气候模型（GCM）能够提供这种理解，但单次运行可能需要高达数百万核时以及大量领域专家时间。机器学习模拟器有望消除这一瓶颈，但相关进展一直受限于缺乏一个经过精心整理的多模型系外气候数据集。我们推出了ThousandWorlds，这是一个面向系外气候模拟以及更广泛的低数据、多模拟器、参数到场回归任务领域的机器学习就绪基准数据集。该数据集包含来自五个全球气候模型的约1800次模拟，将八个行星参数映射到三维大气场，包括温度、湿度、风、云和辐射。

    arXiv:2606.18338v2 Announce Type: replace  Abstract: The search for life beyond Earth will depend on detecting faint signatures in the atmospheres of potentially habitable exoplanets. Interpreting those signatures requires understanding the host planet's climate: the same molecule may signal life on one planet and abiotic chemistry on another. Global climate models (GCMs) provide this understanding, but individual runs can require up to millions of core-hours and substantial domain expert time. Machine-learning emulators could remove this bottleneck, but progress has been limited by the absence of a curated, multi-model exoclimate dataset. We introduce ThousandWorlds, an ML-ready benchmark for exoclimate emulation and for the broader regime of low-data, multi-simulator, parameter-to-field regression. The dataset contains approximately 1800 simulations from five GCMs, mapping eight planet parameters to 3D atmospheric fields including temperature, humidity, winds, clouds, and radiation. 
    
[^211]: SPACR：不确定性感知保形回归器的单遍自适应训练

    SPACR: Single-Pass Adaptive Training of Uncertainty-Aware Conformal Regressors

    [https://arxiv.org/abs/2606.10734](https://arxiv.org/abs/2606.10734)

    SPACR提出了一种基于可微损失的单遍训练方法，无需数据集分割或预定义置信水平即可联合优化准确性、效率与有效性，使单个模型能够在多个置信水平下生成有效且更窄的预测区间，性能优于标准保形预测和DOICR。

    

    保形预测为预测模型提供了可靠的不确定性保证，但通常是事后应用的，这导致模型训练与保形预测产生高效（即窄）预测区间的目标不一致。我们提出了SPACR（单遍自适应保形回归器），这是一种通过可微损失直接训练不确定性感知回归器的新方法。SPACR在训练过程中无需数据集分割或预定义置信水平，即可联合优化准确性、效率和有效性。因此，单个SPACR模型在推理时可在多个置信水平下生成有效的预测区间，避免了诸如直接优化归纳保形回归（DOICR）等方法所需的昂贵重训练。在多种表格和图像数据集上的实验表明，与标准保形预测和DOICR相比，SPACR始终能提供更窄的预测区间和更好的覆盖率-效率权衡，同时显著……

    arXiv:2606.10734v2 Announce Type: replace  Abstract: Conformal Prediction (CP) provides robust uncertainty guarantees for predictive models, but is typically applied post hoc, which misaligns model training with the conformal goal of producing efficient (i.e., narrow) intervals. We propose SPACR (Single-Pass Adaptive Conformal Regressor), a novel method for directly training uncertainty-aware regressors within a differentiable loss. SPACR jointly optimizes accuracy, efficiency, and validity without batch-splitting or a predefined confidence level during training. As a result, a single SPACR model yields valid prediction intervals at multiple confidence levels during inference, avoiding the costly retraining required by methods like Directly Optimized Inductive Conformal Regression (DOICR). Experiments on diverse tabular and image datasets show that SPACR consistently gives tighter intervals and better coverage-efficiency trade-offs compared to standard CP and DOICR, while significantly
    
[^212]: GENERIC-FNO：将能量守恒与熵产生嵌入傅里叶神经算子

    GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators

    [https://arxiv.org/abs/2606.08343](https://arxiv.org/abs/2606.08343)

    GENERIC-FNO是首个在函数空间中嵌入非平衡热力学完整GENERIC结构的神经算子，通过构造精确满足简并条件而无需任何惩罚项，实现了机器精度下的能量守恒与熵产生。

    

    我们提出了GENERIC-FNO，这是首个在函数空间中直接嵌入非平衡态热力学完整GENERIC（度量辛）结构——即通过简并条件相互耦合的可逆能量守恒动力学与不可逆熵产生动力学——的神经算子。现有的结构保持神经算子至多只能强制执行单一守恒律或可逆（哈密顿）结构，而热力学一致的学习一直局限于有限维、图或粒子系统。GENERIC-FNO弥补了这一空白：它将能量和熵泛函学习为神经算子，并将泊松算子和摩擦算子参数化为夹在秩一投影之间的对角傅里叶乘子，这些秩一投影通过构造本身精确地强制执行简并条件，无需惩罚项、投影更新或残差。简并恒等式在机器精度下成立（残差约为10^-13）。

    arXiv:2606.08343v5 Announce Type: replace  Abstract: We introduce GENERIC-FNO, the first neural operator to embed the full GENERIC (metriplectic) structure of nonequilibrium thermodynamics -- reversible, energy-conserving dynamics and irreversible, entropy-producing dynamics coupled through the degeneracy conditions -- directly in function space. Existing structure-preserving neural operators enforce at most a single conservation law or reversible (Hamiltonian) structure, while thermodynamically consistent learning has been confined to finite-dimensional, graph, or particle systems. GENERIC-FNO closes this gap: it learns the energy and entropy functionals as neural operators and parameterizes the Poisson and friction operators as diagonal Fourier multipliers sandwiched between rank-one projections that enforce the degeneracy conditions exactly, by construction, with no penalty term, update projection, or residual. The degeneracy identities hold to machine precision (residuals ~10^-13) 
    
[^213]: 重复不匹配：为什么数据混合实验无法扩展以及如何修复它们

    Repetition Mismatch: Why Data Mixture Experiments Don't Scale and How to Fix Them

    [https://arxiv.org/abs/2606.07597](https://arxiv.org/abs/2606.07597)

    论文揭示了预训练数据混合实验无法从小规模外推到大规模的主要原因是高质量数据的重复率随训练预算变化而改变最优混合比例，并提出通过匹配目标重复率的子采样方法，仅用1/16的目标token即可恢复接近最优的数据混合配置。

    

    预训练数据混合通常通过运行小规模实验并外推到目标训练预算来进行调整。当高质量数据稀缺且必须重复使用时，这种外推经常失败，但失败的根源尚未被隔离出来。我们证明一个主要罪魁祸首是重复不匹配：由于高质量数据集规模较小，其重复率会随着训练预算的增长而变化，从而以小规模代理实验无法预料的方式改变最优混合比例。一种匹配目标重复率的子采样程序可以控制这种效应。在将有限的高质量数据与网络爬取数据相结合的双源设置中，仅使用目标token数量1/16的单次重复控制实验，就能为11.7亿参数模型在Wiki-Text上恢复出与最优值相差0.10以内的混合比例，而在没有重复控制的情况下误差为0.85。获得可比的准确

    arXiv:2606.07597v2 Announce Type: replace-cross  Abstract: Pre-training data mixtures are commonly tuned by running small-scale experiments and extrapolating to the target training budget. When high-quality data is scarce and must be repeated, this extrapolation frequently fails, but the source of the failure has not been isolated. We show that a primary culprit is a repetition mismatch: because high-quality datasets are small, their repetition rate changes as the training budget grows, shifting the optimal mixture in ways that small-scale proxy experiments do not anticipate. A subsampling procedure that matches the target repetition rate controls for this effect. In a two-source setting combining limited high-quality data with web crawl, a single repetition-controlled experiment using only 1/16 of the target tokens recovers a mixture within 0.10 of the optimum on Wiki-Text for a 1.17B parameter model, compared to an error of 0.85 without repetition control. Achieving comparable accura
    
[^214]: 尾部的捷径：通过对微调更新的事后谱压缩实现去偏

    Shortcuts in the Tail: Debiasing via Post-Hoc Spectral Compression of Fine-Tuning Updates

    [https://arxiv.org/abs/2606.07596](https://arxiv.org/abs/2606.07596)

    对微调权重更新的SVD进行简单的事后尾部截断，无需重新训练、群体标签或反事实数据，即可在几乎不损失准确率的情况下显著减少模型对虚假捷径的依赖，实现对代表性不足群体的去偏。

    

    微调常常在引入任务知识的同时也引入虚假相关性，导致模型在代表性不足的群体上出现系统性失败。现有的缓解方法需要重新训练、群体标签或精心策划的反事实数据。我们展示了一种简单的事后干预，无需上述任何条件即可减少对捷径的依赖：对 $\Delta W = W_\mathrm{ft} - W_\mathrm{base}$ 的 SVD 进行尾部截断，可以在保持任务准确率的同时缩小虚假群体差距。在三个指令微调模型（0.5B–7B）和四个分类基准上，top-k 截断在准确率损失小于2个百分点的情况下缩小了每个单元格上的差距，在 CivilComments 上最多可缩小5倍。我们提出这之所以有效，是因为捷径响应位于 $\Delta W$ 奇异值排序的尾部——这是关于截断行为的论断，而非关于原始奇异值本身的论断，后者分布广泛且在所有四个数据集上看起来相同。一个受控的边界案例……

    arXiv:2606.07596v2 Announce Type: replace  Abstract: Fine-tuning often introduces spurious correlations alongside task knowledge, causing systematic failures on underrepresented groups. Existing mitigations require retraining, group labels, or curated counterfactual data. We show a simple post-hoc intervention reduces shortcut reliance without any of these: truncating the tail of the SVD of $\Delta W = W_\mathrm{ft} - W_\mathrm{base}$ reduces the spurious-group gap while preserving task accuracy. Across three instruction-tuned models ($0.5$B--$7$B) and four classification benchmarks, top-$k$ truncation reduces the gap on every cell at $<2$ pp accuracy loss, by up to $5\times$ on CivilComments. We propose this works because the shortcut response sits in the tail of the singular ordering of $\Delta W$, a claim about how truncation behaves rather than about the raw singular values, which are broadly distributed and look the same across all four datasets. A controlled boundary case in whic
    
[^215]: TEVI：通过稀疏自编码器实现文本条件化的视觉表征编辑，以改进视觉-语言对齐

    TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment

    [https://arxiv.org/abs/2606.07451](https://arxiv.org/abs/2606.07451)

    TEVI框架利用稀疏自编码器解耦图像嵌入，并通过文本条件化的掩码模块只保留与文本描述相符的信息、剔除多余内容，从而改善CLIP等视觉-语言模型中图像-文本嵌入的对齐问题。

    

    像CLIP这样的视觉-语言模型由于其共享的图像-文本嵌入空间而在多种任务中非常有用。尽管如此，图像和文本嵌入之间的对齐往往不佳，从而影响下游任务的性能。近期的研究假设这可以归因于信息不平衡问题：图像所包含的信息多于其文本描述所涵盖的内容。在这项工作中，我们提出了TEVI，一个利用文本描述作为信号来决定从图像嵌入中保留哪些信息的框架。具体而言，我们使用稀疏自编码器来解耦图像嵌入，并训练一个掩码模块，根据给定的文本描述选择性地重建嵌入。在使用合成文本描述的受控实验设置中，我们展示了TEVI能够有效地保留文本描述所涉及的属性，同时丢弃其他属性。我们发现这一方法可以扩展到在自然图像上训练的CLIP模型，其中TEVI学会了进行有意义的掩码操作，并支持基于内容的检索。

    arXiv:2606.07451v2 Announce Type: replace-cross  Abstract: Vision-language models such as CLIP are highly useful for diverse tasks due to their shared image-text embedding space. Despite this, the image and text embeddings are often poorly aligned, affecting downstream performance. Recent work has hypothesized that this can be attributed to an information imbalance: images contain more information than their captions describe. In this work, we propose TEVI, a framework that uses captions as a signal for what to retain from image embeddings. Specifically, we use sparse autoencoders to disentangle image embeddings and train a masking module to selectively reconstruct the embedding based on a given caption. In a controlled setup with synthetic captions, we show that TEVI is effective at preserving caption-described attributes while discarding others. We find that this extends to CLIP models trained on natural images, where TEVI learns to mask meaningfully and allows retrieval based on con
    
[^216]: AIP：一种用于学习与管理智能体技能的图表示

    AIP: A Graph Representation for Learning and Governing Agent Skills

    [https://arxiv.org/abs/2606.04781](https://arxiv.org/abs/2606.04781)

    该论文提出智能体指令协议（AIP），将智能体技能建模为由模式验证的YAML规范治理的有向执行图，从而同时提升智能体在重实现任务上的可靠性和技能创建与改进的效率。

    

    当前的智能体技能大多由自由格式的文本构成，智能体需要在每次会话中阅读、解释并重新推导如何行动。这带来了两个叠加的成本：一是在重实现类任务上的可靠性下降，二是技能创建和改进的困难，因为编辑文本是一个脆弱的过程，无论人类还是智能体都难以胜任，尤其是对于模型训练中代表性不足的领域特定程序性知识。智能体指令协议（AIP）通过将技能建模为有向执行图来解决这两个问题：离散步骤作为节点，由确定性脚本或自然语言描述支撑，通过显式的类型化输入/输出边相连接，并由经过模式验证的YAML规范进行治理。一个编译器元技能可将现有的人类编写的技能转换为这种形式。其好处有二。首先，将人类编写的技能编译为AIP后，Claude Sonnet的平均任务奖励从0.60提升至……

    arXiv:2606.04781v2 Announce Type: replace  Abstract: Agent Skills today consist largely of free-form prose requiring the agent to read, interpret, and re-derive how to act in every session. This imposes two compounding costs: reduced reliability on implementation-heavy tasks, and difficulty in skill creation and improvement, since editing prose is a fragile process that both humans and agents struggle with, particularly for domain-specific procedural knowledge underrepresented in model training. The Agent Instruction Protocol (AIP) addresses both by modeling a skill as a directed execution graph: discrete steps as nodes backed by deterministic scripts or natural-language descriptions, connected by explicit typed input/output edges, and governed by a schema-validated YAML specification. A compiler meta-skill translates existing human-written skills into this form. The benefits are twofold. First, compiling human-written skills to AIP raised Claude Sonnet's mean task reward from 0.60 to 
    
[^217]: 稀疏MoE语言模型中事实回忆的专家感知因果追踪

    Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models

    [https://arxiv.org/abs/2606.03780](https://arxiv.org/abs/2606.03780)

    该研究将激活修补方法细化到专家层面，首次证明在稀疏MoE语言模型（如Qwen3-30B-A3B-Base）中，事实回忆的恢复可定位于单个路由专家（L44E069），但这种专家级定位在不同模型间并不一致。

    

    激活修补方法可以识别出这样的混合专家（MoE）模块：其干净输出能够恢复被破坏的事实性预测。然而，由于模块输出结合了多个被路由专家的贡献，模块级别的恢复并不能确定这种恢复是定位于某个单个专家，还是依赖于整个被路由的专家集合。我们在单token的COUNTERFACT对比样本上研究这一问题：先破坏主语token的嵌入，恢复干净的模块输出，然后在固定路由条件下恢复“干净减去噪声”的专家更新。在Qwen3-30B-A3B-Base中，发现性扫描选定了第44层，留出集分析识别出L44E069作为一个反复出现的路由贡献者，其相对同层活跃专家具有正的特异性。该专家的效应与事实匹配，并能提升真实token的概率和排名，从而解释了部分层级别的恢复效果。而在Mixtral-8x7B-v0.1中，被选中的反复出现的单一专家并不具备特异性；匹配规模的对照组……

    arXiv:2606.03780v2 Announce Type: replace  Abstract: Activation patching can identify a mixture-of-experts (MoE) block whose clean output restores a corrupted factual prediction. However, because the block output combines contributions from multiple routed experts, block-level rescue does not establish whether the recovery localizes to an individual expert or depends on the routed expert set. We study this question on single-token COUNTERFACT contrasts by corrupting subject-token embeddings, restoring clean block outputs, and then restoring clean-minus-noised expert updates under fixed routing. In Qwen3-30B-A3B-Base, a discovery sweep selects layer 44, and held-out analysis identifies L44E069 as a recurrent routed contributor with positive specificity over same-layer active experts. Its effect is fact-matched and improves true-token probability and rank, which explains part of the layer rescue. In Mixtral-8x7B-v0.1, the selected recurrent singleton is not specific; matched-size control
    
[^218]: MidSurfNet：面向薄壁CAD模型中面抽象的学习式面配对方法

    MidSurfNet: Learning Face Pairing for Mid-surface Abstraction of Thin-walled CAD Models

    [https://arxiv.org/abs/2606.01891](https://arxiv.org/abs/2606.01891)

    提出MidSurfNet，通过融合几何流与属性拓扑流双证据的学习式面对评分器和确定性面组构成，实现薄壁CAD模型中面抽象的鲁棒面配对，克服了传统手工阈值方法难以适应多局部壁厚且结果不一致的缺陷。

    

    arXiv:2606.01891v2 公告类型：replace-cross　摘要：中面抽象是薄壁CAD模型有限元分析中的一项重要预处理步骤，而面配对是其核心子问题。现有的面配对方法依赖手工设计的几何准则，当模型具有多个局部壁厚时，这些准则的阈值难以调节；其分组结果依赖于阈值设置和处理顺序，因此同一模型可能产生不一致的结果。我们提出了MidSurfNet，一种基于学习的面配对方法，它将学习得到的面对评分器与确定性的面组构成方式相结合。该评分器通过两条独立学习的证据流来评估每个无序面对：几何流将连续配对准则与条件形状校正相结合；属性拓扑流作用于B-Rep面邻接图。一个以面对为条件的门控机制融合这两条证据流，独立的逐对决策保留对向面支持。

    arXiv:2606.01891v2 Announce Type: replace-cross  Abstract: Mid-surface abstraction is an important preprocessing step for finite element analysis of thin-walled CAD models, and face pairing is its central subproblem. Existing face-pairing methods rely on handcrafted geometric criteria whose thresholds are hard to tune when a model has multiple local wall thicknesses; their groupings depend on threshold settings and processing order, so the same model can yield inconsistent results. We present MidSurfNet, a learning-based face-pairing method that couples a learned face-pair scorer with a deterministic face-group composition. The scorer evaluates every unordered face pair with two separately learned evidence streams: a geometry stream combining continuous pairing criteria with a conditional shape correction, and an attributed-topology stream over the B-Rep face-adjacency graph. A pair-conditioned gate fuses the two streams, and independent per-pair decisions retain opposing-face support 
    
[^219]: HARP：用于极端大语言模型量化的Hadamard预条件自适应旋转处理器

    HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization

    [https://arxiv.org/abs/2605.29843](https://arxiv.org/abs/2605.29843)

    提出HARP，一种可学习的Hadamard预条件自适应旋转处理器，通过稀疏蝶形块正交结构替代固定Hadamard变换，在保持与全精度模型精确等价的同时，自适应地适应层、校准分布和量化器，从而提升极端低比特LLM量化的鲁棒性。

    

    训练后量化（PTQ）对于在内存和带宽受限条件下部署大语言模型（LLM）至关重要。然而，极端低比特量化对激活异常值和各向异性权重曲率仍然高度敏感。现有的基于非相干性的PTQ方法通过固定的随机化Hadamard变换（RHT）来缓解这一问题，虽然提高了量化鲁棒性，但无法使旋转基适应特定层、校准分布或量化器。我们提出了HARP（Hadamard预条件自适应旋转处理器），这是一种可学习的结构化双侧正交处理器，可替代固定的Hadamard混合，同时保持与全精度模型的精确等价性。HARP将每次旋转表示为稀疏的类蝶形块正交阶段的乘积，通过混合基调度支持非2的幂次维度，并且初始化时与RHT处理器等价（最多差一个固定置换）。仅在校准数据上进行拟合，HARP能够……（原文在此处截断）

    arXiv:2605.29843v2 Announce Type: replace-cross  Abstract: Post-training quantization (PTQ) is essential for deploying LLMs under memory and bandwidth constraints. However, extreme low-bit quantization remains highly sensitive to activation outliers and anisotropic weight curvature. Existing incoherence-based PTQ methods mitigate this issue with fixed randomized Hadamard transforms (RHTs), which improve quantization robustness but cannot adapt the rotated basis to the layer, calibration distribution, or quantizer. We introduce HARP (Hadamard-preconditioned Adaptive Rotation Processor), a learnable structured two-sided orthogonal processor that replaces fixed Hadamard mixing while preserving exact full-precision equivalence. HARP represents each rotation as a product of sparse butterfly-like block-orthogonal stages, supports non-power-of-two dimensions through Mixed-Radix schedules, and initializes to the RHT processor up to a fixed permutation. Fitted only on calibration data, HARP ada
    
[^220]: 策略感知模拟器学习的理论基础与有效算法

    Theoretical Foundations and Effective Algorithms for Policy-Aware Simulator Learning

    [https://arxiv.org/abs/2605.29032](https://arxiv.org/abs/2605.29032)

    本文提出将模拟器学习目标从预测准确性转向策略鲁棒性，通过模型与对抗策略之间的零和极小极大博弈框架，证明了该博弈具有次线性遗憾界的可学习性，并建立了寻找最坏情况策略与以评论家误差为奖励的标准强化学习问题之间的误差-MDP对偶理论。

    

    基于模型的强化学习(MBRL)智能体通常通过最小化预测损失来学习世界模型。然而，强大的RL优化器不可避免地会利用模型中的微小不准确之处，导致模拟器被利用和现实差距问题，即策略在模拟中成功但在现实世界中失败。我们提出，学习模拟器的目标应当是策略鲁棒性而非预测准确性，并将此表述为模型玩家与对抗性策略玩家之间的零和极小极大博弈。我们提供了全面的理论分析：(1)在线学习保证，证明该博弈是可学习的，具有次线性遗憾界；(2)一种可处理的基于评论家的简化方法，用局部评论家的损失来约束全局策略-价值差距；(3)误差-MDP对偶性，证明寻找最坏情况策略在形式上与一个标准RL问题对偶，其中奖励是单步评论家误差。这种对偶……

    arXiv:2605.29032v3 Announce Type: replace  Abstract: Model-based reinforcement learning (MBRL) agents typically learn world models by minimizing predictive loss. However, powerful RL optimizers inevitably exploit minor model inaccuracies, leading to simulator exploitation and a reality gap where policies succeed in simulation but fail in the real world. We propose that the objective for learning simulators should be strategic robustness rather than predictive accuracy, and formulate this as a zero-sum minimax game between a model player and an adversarial policy player. We provide a comprehensive theoretical analysis: (1) an online learning guarantee showing the game is learnable with sublinear regret bounds; (2) a tractable critic-based simplification bounding the global policy-value gap by the local critic's loss; and (3) an Error-MDP duality, proving that finding the worst-case policy is formally dual to a standard RL problem where the reward is the one-step critic error. This duali
    
[^221]: RW-TTT：面向请求私有测试时训练状态的批处理服务

    RW-TTT: Batched Serving for Request-Owned Test-Time Training State

    [https://arxiv.org/abs/2605.28053](https://arxiv.org/abs/2605.28053)

    RW-TTT通过为每个解码步骤标记所有者、版本和读写效果，实现了对请求私有测试时训练状态的安全高效批处理服务，在相同内存预算下相比串行服务获得9.31倍加速。

    

    测试时训练（TTT）通过在生成过程中读取和更新请求私有的状态（如快速权重、低秩增量或流式学习器状态）来对大语言模型进行自适应调整。这打破了假设权重为共享静态的批处理式大语言模型服务：串行执行虽然正确但速度缓慢，而朴素的批处理可能会破坏请求状态。我们将该问题形式化为读写型TTT服务，并提出RW-TTT，它为每个解码步骤标记其所有者、版本和读/写效果，仅对兼容的阶段进行批处理，并且仅将更新提交给对应的所有者。在一块GPU上运行八个快速权重InPlace-TTT流时，RW-TTT达到274.61的总吞吐量（tok/s），在相同内存预算下比串行服务快9.31倍，比每流副本方案快3.44倍。它在长上下文基准测试RULER上保持了原有行为，并通过了所有者/版本检查。

    arXiv:2605.28053v2 Announce Type: replace  Abstract: Test-time training (TTT) adapts an LLM during generation by reading and updating request-owned state, such as fast weights, low-rank deltas, or streaming learner state. This breaks batched LLM serving, which assumes shared static weights: serial execution is correct but slow, while naive batching can corrupt request state. We formulate this problem as read-write TTT serving and present RW-TTT , which tags each decode step with its owner, version, and READ/WRITE effect, batches only compatible phases, and commits updates only to the owner. On one GPU with eight fast-weight InPlace-TTT streams, RW-TTT reaches 274.61 aggregate tok/s, 9.31x over sequential serving and 3.44x over per-stream replicas under the same memory budget. It preserves behavior on RULER, a long-context benchmark, and passes owner/version checks.
    
[^222]: 信任的时间依赖性：人机团队中的速度、准确性与cBCI神经解耦

    The Timing Dependencies of Trust: Speed, Accuracy, and cBCI Neuro-Decoupling in Human-AI Teams

    [https://arxiv.org/abs/2605.25868](https://arxiv.org/abs/2605.25868)

    该研究证明AI响应时机决定了人机团队的失败机制——快速AI诱发即时盲目服从，使人类受骗时的准确率骤降至50.2%，而缓慢且准确的AI则通过神经解耦机制改善协作脑机接口团队的协同表现。

    

    人工队友的速度与准确性从根本上改变了人机融合的失败状态。高速AI干预有引发反射性盲目服从的风险，而延迟干预则可能引发模糊的认知冲突。本研究探讨了任务中AI助手的基本特征——快速/低准确（FLA-AI）与缓慢/高准确（SA-AI）——如何影响虚拟现实无人机任务中协作脑机接口团队（cBCI）的协同效应。十七名操作员在高认知负荷下完成连续搜索任务，同时使用二维自适应黎曼神谕映射其空间协方差。结果从数学上证明，AI的响应时机决定了团队失败的机制。快速AI引发了即时、盲目的服从；人类在被欺骗情况下的准确率骤降至50.2%，纯行为团队（N=8）的扩展上限未能超过74.1%。相比之下，缓慢AI……（摘要原文在此处截断）

    arXiv:2605.25868v2 Announce Type: replace-cross  Abstract: The speed and accuracy of an artificial teammate fundamentally alter the failure states of Human-AI integration. While high-speed AI interventions risk inducing reflexive blind compliance, delayed interventions can induce ambiguous cognitive conflict. This study investigates how the fundamental characteristics of an in-task AI assistant, Fast/Less-Accurate (FLA-AI) versus Slow/Accurate (SA-AI) impact the synergy of Collaborative Brain-Computer Interface (cBCI) teams in a Virtual Reality drone task. Seventeen operators completed continuous search tasks under high cognitive workload while their spatial covariance was mapped using a 2D Adaptive Riemannian Oracle. The results mathematically demonstrate that AI timing dictates the mechanism of team failure. Fast AI induced instant, blind compliance; human accuracy under deception collapsed to 50.2%, and pure behavioural teams (N=8) failed to scale beyond 74.1%. In contrast, Slow AI 
    
[^223]: 迈向可负担能源：面向电力公司需求响应项目的Gymnasium环境

    Towards Affordable Energy: A Gymnasium Environment for Electric Utility Demand-Response Programs

    [https://arxiv.org/abs/2605.12462](https://arxiv.org/abs/2605.12462)

    本文提出了DR-Gym，一个开源的、与Gymnasium兼容的在线仿真环境，用于从电力公司视角训练和评估需求响应策略，解决了离线历史数据无法捕捉价格信号与用户行为之间动态交互反馈循环的问题。

    

    极端天气和剧烈波动的批发电力市场使住宅消费者面临灾难性的财务风险，然而配电层面的需求响应作为提升电网灵活性和能源可负担性的工具仍未得到充分利用。虽然需求响应项目可以通过在电价高峰期向消费者发放财务补贴来保护他们，但优化这一序列决策过程对强化学习而言是一个独特的挑战，尽管公开可用的离线历史智能电表数据和批发电价数据十分丰富。离线历史数据无法捕捉电力公司价格信号与客户对需求响应项目的接受和适应之间动态的交互反馈循环。为解决这一问题，我们提出了DR-Gym，一个开源的、与Gymnasium兼容的在线环境，旨在从电力公司的视角训练和评估需求响应策略。

    arXiv:2605.12462v2 Announce Type: replace  Abstract: Extreme weather and volatile wholesale electricity markets expose residential consumers to catastrophic financial risks, yet demand response at the distribution level remains an underutilized tool for grid flexibility and energy affordability. While a demand-response program can shield consumers by issuing financial credits during high-price periods, optimizing this sequential decision-making process presents a unique challenge for reinforcement learning despite the plentiful offline historical smart meter and wholesale pricing data available publicly. Offline historical data fails to capture the dynamic, interactive feedback loop between an electric utility's pricing signals and customer acceptance and adaptation to a demand-response program. To address this, we introduce DR-Gym, an open-source, online Gymnasium-compatible environment designed to train and evaluate demand-response from the electric utility's perspective. Unlike exis
    
[^224]: 一种真实校准的合成优先数据引擎

    A Real-Calibrated Synthetic-First Data Engine

    [https://arxiv.org/abs/2605.09699](https://arxiv.org/abs/2605.09699)

    提出了一个结合可控扩散生成与多阶段筛选过滤的模块化数据工程框架，通过系统化的数据集构建提升低数据量场景下合成数据增强的实际可靠性。

    

    现代计算机视觉系统在数据稀缺领域日益遭遇性能瓶颈，因为在这些领域中收集大规模、高质量的标注数据成本高昂或难以实现。虽然可控扩散模型能够实现可扩展的合成图像生成，但由于数据集层面的质量问题以及反馈机制不足，直接应用合成数据增强往往导致性能提升不稳定。在本工作中，我们提出了一种真实校准的合成优先数据引擎，这是一个模块化的数据工程框架，在统一流程中结合了可控扩散生成与多阶段筛选/过滤，并可选地支持不确定性驱动的选择和人工验证。我们的方法并不引入新的生成算法，而是专注于系统化的数据集构建，以提升合成数据增强在低数据量场景下的实际可靠性。该框架已实现……

    arXiv:2605.09699v2 Announce Type: replace-cross  Abstract: Modern computer vision systems increasingly encounter performance limitations in data-scarce domains, where collecting large-scale, high-quality labeled data is costly or impractical. While controllable diffusion models enable scalable synthetic image generation, directly applying synthetic augmentation often leads to unstable performance gains due to dataset-level quality issues and insufficient feedback mechanisms. In this work, we present a Real-Calibrated Synthetic-First Data Engine, a modular data engineering framework that combines controllable diffusion generation and multi-stage curation/filtering within a unified pipeline, with optional support for uncertainty-driven selection and human verification. Instead of introducing new generative algorithms, our approach focuses on systematic dataset construction for improving the practical reliability of synthetic augmentation in low-data regimes. The framework is implemented 
    
[^225]: 基于稀疏固定传感器的观测对齐两阶段域分解物理信息交通状态估计

    Observation-Aligned Two-Stage Domain Decomposition for Physics-Informed Traffic State Estimation with Sparse Fixed Sensors

    [https://arxiv.org/abs/2605.08028](https://arxiv.org/abs/2605.08028)

    提出了观测对齐的两阶段域分解物理信息神经网络TSDD-PINN，利用全局父网络的残差剖面指导确定性域分解与子网络热启动，解决了PINN在稀疏固定传感器交通状态估计中过度平滑LWR模型允许的剧烈状态转变的问题，其中空间细化在精度与训练效率之间取得最佳平衡。

    

    从稀疏固定传感器进行交通状态估计具有挑战性，因为物理信息神经网络（PINN）往往会过度平滑Lighthill-Whitham--Richards（LWR）模型所允许的剧烈交通状态转变。本研究提出了两阶段域分解物理信息神经网络（TSDD-PINN），这是一个用于基于LWR的离线速度场重建的观测对齐框架。该框架支持空间细化、时间细化以及时空联合细化。匹配方向分析表明，在所测试的设置中，空间细化具有最低的平均误差，且训练时间不到时空联合细化的一半，而时间细化的速度更快。该框架首先训练一个全局父PINN。在受控的空间实现中，父网络的残差剖面用于指导确定性分区，从而为热启动的子网络提供初始化。当预设的筛选条件未激活时，可选的运行保障机制会保留第一阶段的结果。（注：原文摘要至此处被截断）

    arXiv:2605.08028v2 Announce Type: replace  Abstract: Traffic state estimation from sparse fixed sensors is challenging because physics-informed neural networks (PINNs) tend to over-smooth sharp transitions admitted by the Lighthill-Whitham--Richards (LWR) model. This study proposes Two-Stage Domain Decomposition Physics-Informed Neural Networks (TSDD-PINN), an observation-aligned framework for LWR-based offline speed-field reconstruction. The framework supports spatial, temporal, and space--time refinement. Matched direction analysis shows that spatial refinement has the lowest mean error and less than half the training time of space--time refinement in the tested setting, while temporal refinement is faster. A global parent PINN is first trained. In the controlled spatial implementation, its residual profile guides a deterministic partition for warm-started child networks. An optional operational safeguard retains Stage~1 when the prespecified screen does not activate. The primary I-2
    
[^226]: 自拍捕捉动态作为对抗深度伪造与注入攻击的辅助信号用于移动身份验证

    Selfie-Capture Dynamics as an Auxiliary Signal Against Deepfakes and Injection Attacks for Mobile Identity Verification

    [https://arxiv.org/abs/2605.00218](https://arxiv.org/abs/2605.00218)

    本文提出CanSelfie多传感器数据集，证明自拍捕捉过程中记录的被动运动轨迹可作为辅助证据信号，有效筛查深度伪造和视频注入攻击，为移动远程身份验证提供了超越传统摄像头检测的新证据渠道。

    

    移动远程身份验证（RIdV）系统容易遭受操纵或替换面部视频流的攻击，包括呈现攻击、实时深度伪造和视频注入。欧洲近期出台的要求，如ETSI TS 119 461和CEN/TS 18099，推动了超越基于摄像头的呈现攻击检测的辅助证据渠道。本文研究了在自拍捕捉过程中记录的被动运动轨迹是否能为欺骗筛查和用户验证提供辅助证据。我们介绍了CanSelfie数据集，该数据集包含使用商业移动RIdV应用从30名参与者以50 Hz频率采集的375条真实多传感器序列，并涵盖静止、手持和时间偏移的攻击代理场景。我们在不同传感器配置和时间窗口下对7个多元时间序列分类器和8个全序列异常检测器进行了基准测试。在欺骗筛查方面，仅使用加速度计的ROCKAD获得了0.0……（摘要原文在此处截断）

    arXiv:2605.00218v2 Announce Type: replace-cross  Abstract: Mobile remote identity verification (RIdV) systems are exposed to attacks that manipulate or replace the facial video stream, including presentation attacks, real-time deepfakes, and video injection. Recent European requirements, including ETSI TS 119 461 and CEN/TS 18099, motivate evidence channels beyond camera-based presentation-attack detection. This paper studies whether passive motion traces recorded during selfie capture provide auxiliary evidence for spoof screening and user verification. We introduce CanSelfie, a dataset of 375 bona fide multi-sensor sequences collected at 50 Hz from 30 participants using a commercial mobile RIdV application, together with stationary, handheld, and temporally shifted attack-proxy scenarios. We benchmark 7 multivariate time-series classifiers and 8 whole-series anomaly detectors across sensor configurations and temporal windows. For spoof screening, accelerometer-only ROCKAD obtains 0.0
    
[^227]: RCProb：从分类树集成中提取概率规则

    RCProb: Probabilistic rule extraction from classification tree ensembles

    [https://arxiv.org/abs/2604.25304](https://arxiv.org/abs/2604.25304)

    RCProb是一种针对RuleCOSI+的概率扩展方法，通过平滑的原子类条件证据和支持自适应混合概率估计，显著提升了从分类树集成中提取规则的概率可靠性，将对数损失大幅降低。

    

    树集成提供了强大的分类性能，但通常表现为黑盒模型。诸如RuleCOSI+等事后可解释性技术会提取一个近似于集成模型的小型规则集，但这种简化可能使附加在提取规则上的概率变得不可靠。特别是，RuleCOSI+为提取的规则分配经验类别概率，并在其贪婪组合和简化过程中反复使用这些规则统计量。我们提出了RCProb，一种概率扩展方法，它在计算开销较大的搜索阶段使用平滑的原子类条件证据，并在最终规则概率中使用带有集成信息的m估计的支持自适应混合方法。该方法在18个二分类和5个多分类数据集上使用随机森林（RF）和梯度提升机（GBM）集成进行评估。相对于RuleCOSI+，RF的中位配对对数损失降低为71.9%，GBM为62.5%。

    arXiv:2604.25304v2 Announce Type: replace  Abstract: Tree ensembles provide strong classification performance but usually behave as black-box models. Post-hoc interpretability techniques such as RuleCOSI+ extract a small ruleset that approximates the ensemble, but this simplification can leave the probabilities attached to the extracted rules unreliable. In particular, RuleCOSI+ assigns empirical class probabilities to the extracted rules and repeatedly uses those rule statistics during its greedy combination and simplification procedure. We present RCProb, a probabilistic extension that uses smoothed atomic class-conditional evidence for the expensive search stages and a support-adaptive mixture with an ensemble-informed m-estimate for the final rule probabilities. The method is evaluated on 18 binary and 5 multiclass datasets using random forest (RF) and gradient boosting machine (GBM) ensembles. Relative to RuleCOSI+, the median paired log-loss reduction is 71.9\% for RF and 62.5\% 
    
[^228]: 当思维链失败时，解决方案隐藏在隐藏状态之中

    When Chain-of-Thought Fails, the Solution Hides in the Hidden States

    [https://arxiv.org/abs/2604.23351](https://arxiv.org/abs/2604.23351)

    研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。

    

    arXiv:2604.23351v2 公告类型：replace-cross。摘要：中间推理在计算上究竟是有用的，还是仅仅起解释作用，取决于思维链（CoT）标记中是否包含与任务相关的信息。我们利用激活修补（activation patching）对 GSM8K 上的思维链进行了机制层面的因果分析：将同一问题在思维链生成中得到的标记级隐藏状态转移到直接回答的运行中，然后测量其对最终答案准确率的影响。在各个模型上，修补后生成答案的准确率显著高于直接回答提示和原始思维链轨迹，这表明即使原始推理轨迹是错误的，单个思维链标记也能编码足以恢复正确答案的信息。这种与任务相关的信息在正确的思维链运行中比在错误的运行中更为普遍，并且在各标记之间分布不均：它集中于中间至较深的层，并在推理轨迹的较早位置出现。此外，修补语言标记，例如……（原文摘要在此处截断）

    arXiv:2604.23351v2 Announce Type: replace-cross  Abstract: Whether intermediate reasoning is computationally useful or merely explanatory depends on whether chain-of-thought (CoT) tokens contain task-relevant information. We present a mechanistic causal analysis of CoT on GSM8K using activation patching: transferring token-level hidden states from a CoT generation to a direct-answer run for the same question, then measuring the effect on final-answer accuracy. Across models, generating after patching yields substantially higher accuracy than both direct-answer prompting and the original CoT trace, revealing that individual CoT tokens can encode sufficient information to recover the correct answer, even when the original trace is incorrect. This task-relevant information is more prevalent in correct than incorrect CoT runs and is unevenly distributed across tokens, concentrating in mid-to-late layers and appearing earlier in the reasoning trace. Moreover, patching language tokens such a
    
[^229]: 迈向通用表格嵌入：跨数据任务的基准测试

    Towards Universal Tabular Embeddings: A Benchmark Across Data Tasks

    [https://arxiv.org/abs/2604.21696](https://arxiv.org/abs/2604.21696)

    该论文提出了TEmBed统一基准，首次在单元格、行、列和表格四个表示层级上系统评估表格嵌入模型，发现模型优劣取决于任务和表示层级，为实际应用中的模型选择提供了实用指导。

    

    表格基础模型旨在学习表格数据的通用表示，使其能够跨任务和领域迁移，从而支持表格检索、语义搜索和基于表格的预测等应用。尽管此类模型的数量不断增长，但目前尚不清楚哪种方法在实践中效果最佳，因为现有方法通常在特定任务的设置下进行评估，难以进行直接比较。为解决这一问题，我们提出了TEmBed（表格嵌入测试平台），这是一个统一的基准，用于在单元格、行、列和表格四个表示层级上系统评估表格嵌入。通过评估多种多样化的表格表示学习模型，我们发现应选用哪种模型取决于具体任务和表示层级。我们的研究结果为在现实应用中选择表格嵌入提供了实用指导，并为开发更通用的表格表示模型奠定了基础。

    arXiv:2604.21696v2 Announce Type: replace  Abstract: Tabular foundation models aim to learn universal representations of tabular data that transfer across tasks and domains, enabling applications such as table retrieval, semantic search and table-based prediction. Despite the growing number of such models, it remains unclear which approach works best in practice, as existing methods are often evaluated under task-specific settings that make direct comparison difficult. To address this, we introduce TEmBed, the Tabular Embedding Test Bed, a unified benchmark for systematically evaluating tabular embeddings across four representation levels: cell, row, column, and table. Evaluating a diverse set of tabular representation learning models, we show that which model to use depends on the task and representation level. Our results offer practical guidance for selecting tabular embeddings in real-world applications and lay the groundwork for developing more general-purpose tabular representati
    
[^230]: 学习串联量子码

    Learning to Concatenate Quantum Codes

    [https://arxiv.org/abs/2604.14931](https://arxiv.org/abs/2604.14931)

    该论文提出一种逐级自适应的量子码串联方法：先估计每级后的有效噪声信道，在噪声结构化时使用学习型小型非加性编码器、在噪声趋于均匀时切换到标准码，从而在强结构化噪声下将所需量子比特数最多减少两个数量级。

    

    串联量子纠错码通过在各级上使逻辑错误率呈双指数下降，从而扩展纠错能力。然而，噪声结构在串联过程中会发生偏移，使得难以选择最优的码序列。我们通过估计每一级之后的有效噪声信道并据此选择下一个码，实现了这一选择的自动化。特别地，当噪声表现出足够的结构时，我们采用基于学习的方法定制小型非加性编码器；一旦噪声接近均匀，便切换到标准码。在仿真中，这种逐级自适应方法只需远少于仅串联稳定子码所需的量子比特数即可达到目标逻辑错误率——对于强结构化噪声，量子比特数量最多可减少两个数量级。因此，这种基于学习的混合策略为早期容错量子计算提供了一种有前景的工具。

    arXiv:2604.14931v2 Announce Type: replace-cross  Abstract: Concatenating quantum error correction codes scales error correction capability by driving logical error rates down double-exponentially across levels. However, the noise structure shifts under concatenation, making it hard to choose an optimal code sequence. We automate this choice by estimating the effective noise channel after each level and selecting the next code accordingly. In particular, we use learning-based methods to tailor small, non-additive encoders when the noise exhibits sufficient structure, then switch to standard codes once the noise is nearly uniform. In simulations, this level-wise adaptation achieves a target logical error rate with far fewer qubits than concatenating stabilizer codes alone--reducing qubit counts by up to two orders of magnitude for strongly structured noise. Therefore, this hybrid, learning-based strategy offers a promising tool for early fault-tolerant quantum computing.
    
[^231]: 安全训练调节在线策略强化学习下的有害失准行为，但调节方向取决于环境设计

    Safety Training Modulates Harmful Misalignment Under On-Policy RL, But Direction Depends on Environment Design

    [https://arxiv.org/abs/2604.12500](https://arxiv.org/abs/2604.12500)

    本研究通过对11个不同规模的指令微调大语言模型在3种环境中进行在线策略强化学习训练，揭示了模型规模是否成为安全缓冲取决于环境设计，并发现在线策略RL能保留模型生成分布中固有的安全缓冲，而离线策略设置会绕过这一缓冲。

    

    强化学习（RL）中的规范博弈会导致大语言模型发展出谄媚、操纵或欺骗行为，但其发生的条件仍不明确。我们在3个环境中使用在线策略RL训练了11个指令微调大语言模型（0.5B-14B），发现模型规模在某些环境中充当安全缓冲，但在另一些环境中反而助长了更有害的利用行为。受控消融实验将这种反转归因于环境特异性特征，如角色框架设定和隐含的可博弈性线索。我们进一步表明，大多数安全基准无法预测RL诱导的失准行为，唯一的例外是当利用行为依赖于推断用户偏好时的谄媚（Sycophancy）评分。最后，我们发现在线策略RL保留了模型自身生成分布中固有的安全缓冲，而这一缓冲在离线策略设置中会被绕过。

    arXiv:2604.12500v2 Announce Type: replace  Abstract: Specification gaming under Reinforcement Learning (RL) is known to cause LLMs to develop sycophantic, manipulative, or deceptive behavior, yet the conditions under which this occurs remain unclear. We train 11 instruction-tuned LLMs (0.5B-14B) with on-policy RL across 3 environments and find that model size acts as a safety buffer in some environments but enables greater harmful exploitation in others. Controlled ablations trace this reversal to environment-specific features such as role framing and implicit gameability cues. We further show that most safety benchmarks do not predict RL-induced misalignment, except in the case of Sycophancy scores when the exploit relies on inferring the user's preference. Finally, we find that on-policy RL preserves a safety buffer inherent in the model's own generation distribution, one that is bypassed during off-policy settings.
    
[^232]: 滑动窗口重排与重叠平均：一种用于多变量预测的简单时域数据增强方法

    Sliding-Window Reordering with Overlap Averaging: A Simple Time-Domain Augmentation for Multivariate Forecasting

    [https://arxiv.org/abs/2604.09067](https://arxiv.org/abs/2604.09067)

    提出一种简单的时域数据增强方法，通过随机重排滑动窗口并对重叠部分取平均来重建序列，在限制时间失真的前提下生成受控变化的合成样本，并在九个长期预测基准和四个短期交通基准上显著优于现有增强方法。

    

    数据增强已成为改进深度预测模型的核心技术，但分类式的变换往往会破坏回看窗口与其连续未来目标之间的连贯性。我们提出了一种简单的处理流程：将联合的输入-目标序列展开为重叠的滑动窗口，随机重排其中受控比例的窗口（并依据一个轻量级方差准则进行优先排序），然后通过对重叠部分取平均来重建序列，从而在限制时间失真的同时生成具有受控变化的合成样本。该流程与模型无关，仅引入三个可解释的超参数，并在九个长期预测基准上（涵盖五个骨干模型系列：TSMixer、DLinear、PatchTST、TiDE、LightTS）以及四个使用 PatchTST 的短期交通基准上，相对一组全面的竞争性增强方法取得了显著提升。组件级消融实验与超参（原文截断）

    arXiv:2604.09067v2 Announce Type: replace  Abstract: Augmentation has become a central technique for improving deep forecasting models, but classification-style transformations tend to break the coherence between the look-back window and its continuous future target. We describe a simple procedure that unfolds the joint input-target sequence into overlapping sliding windows, randomly reorders a controlled fraction of them-prioritized by a lightweight variance criterion-and reconstructs the sequence by averaging across the overlaps, producing synthetic samples with controlled variation while limiting temporal distortion. The procedure is model-agnostic, introduces only three interpretable hyperparameters, and achieves strong improvements over a comprehensive set of competing augmentations across nine long-term forecasting benchmarks with five backbone families (TSMixer, DLinear, PatchTST, TiDE, LightTS) and four short-term traffic benchmarks with PatchTST. Component-wise ablations, hype
    
[^233]: 迈向终身空中自主：动态环境中持续视觉位置识别的几何记忆管理

    Towards Lifelong Aerial Autonomy: Geometric Memory Management for Continual Visual Place Recognition in Dynamic Environments

    [https://arxiv.org/abs/2604.09038](https://arxiv.org/abs/2604.09038)

    该论文将空中视觉位置识别建模为基于任务的域增量学习问题，提出结合静态卫星样本记忆与有界重放缓冲区的异构记忆框架及DBS-Hybrid混合样本选择策略，以应对环境变化引起的灾难性遗忘，实现动态环境下长期空中自主的鲁棒地理定位。

    

    arXiv:2604.09038v2 公告类型： replace-cross 摘要：在不断变化的环境和运行条件下实现鲁棒的地理定位，对长期空中自主至关重要。空中视觉位置识别（VPR）通常使用预先获取的目标作业区域遥感影像，因此地理标签空间可以保持固定，而连续执行的飞行任务会引入显著的视觉分布偏移。对这些偏移进行持续适应可能导致灾难性遗忘。因此，我们将空中VPR表述为基于任务的域增量学习（DIL）问题，并开发了一个异构记忆框架。在顺序适应之前，卫星参考数据集仅使用一次来训练初始模型并构建静态卫星样本记忆；随后，一个有界重放缓冲区在各次任务之间保留筛选后的机载观测数据。在重放管理方面，我们比较了基于损失和基于多样性的样本选择准则，并提出了DBS-Hybrid，它结合……

    arXiv:2604.09038v2 Announce Type: replace-cross  Abstract: Robust geo-localization under changing environmental and operational conditions is critical for long-term aerial autonomy. Aerial visual place recognition (VPR) commonly uses pre-acquired remote-sensing imagery of the intended operating area, so the geographic label space can remain fixed while successive airborne missions introduce substantial visual distribution shifts. Continual adaptation to these shifts can cause catastrophic forgetting. We therefore formulate aerial VPR as a mission-based domain-incremental learning (DIL) problem and develop a heterogeneous memory framework. Before sequential adaptation, the satellite reference dataset is used once to train the initial model and construct a static satellite exemplar memory; a bounded replay buffer then retains selected airborne observations across missions. For replay management, we compare loss- and diversity-based selection criteria and introduce DBS-Hybrid, which combi
    
[^234]: 外科人工智能的比较研究：数据、算力与规模化的潜力与局限

    A Comparative Study in Surgical AI: Potential and Limitations of Data, Compute, and Scaling

    [https://arxiv.org/abs/2603.27341](https://arxiv.org/abs/2603.27341)

    本文比较研究了数据、算力与规模化在外科AI中的潜力与局限，探讨现代通用AI能否以及在多大程度上辅助外科实践。

    

    近期的人工智能（AI）模型已在多项生物医学任务性能基准上达到或超越人类专家水平，但外科手术相关的基准在外科手术基准在知名医学基准套件中往往缺失。由于外科手术需要整合多种不同的任务，若性能能够得到提升，具备通用能力的AI模型作为协作工具将特别具有吸引力。一方面，扩大模型架构规模和训练数据量的经典做法很有吸引力，尤其是每年都会产生数百万小时的外科手术视频数据；另一方面，为AI训练准备外科数据需要相当高的专业知识水平，而使用这些数据进行训练也需要昂贵的计算资源。这些权衡使得现代AI能否以及在多大程度上辅助外科实践变得不确定。在本文中，我们通过……（原文摘要在此处截断）

    arXiv:2603.27341v5 Announce Type: replace  Abstract: Recent Artificial Intelligence (AI) models have matched or exceeded human experts in several benchmarks of biomedical task performance, but surgical benchmarks in particular are often missing from prominent medical benchmark suites. Since surgery requires integrating disparate tasks, generally-capable AI models could be particularly attractive as a collaborative tool if performance could be improved. On the one hand, the canonical approach of scaling architecture size and training data is attractive, especially since there are millions of hours of surgical video data generated per year. On the other hand, preparing surgical data for AI training requires significantly higher levels of professional expertise, and training on that data requires expensive computational resources. These trade-offs paint an uncertain picture of whether and to-what-extent modern AI could aid surgical practice. In this paper, we explore this question through
    
[^235]: 基于预期不对称几何的二元因果方向识别

    Identification of Bivariate Causal Directionality Based on Anticipated Asymmetric Geometries

    [https://arxiv.org/abs/2603.26024](https://arxiv.org/abs/2603.26024)

    本文提出了两种基于条件分布的新方法——预期不对称几何（AAG）和单调性指数（MI），用于识别二元数值数据中的因果方向性。

    

    arXiv:2603.26024v2 公告类型：替换 摘要：识别二元数值数据中的因果方向性是一个基础性研究问题，具有重要的实际应用意义。本文提出了两种通过考虑条件分布来识别因果方向的替代方法：（1）预期不对称几何（AAG）和（2）单调性指数（MI）。AAG方法将实际的条件分布与沿两个变量的预期分布进行比较，并评估了多种比较度量，如皮尔逊相关系数、余弦距离、杰卡德指数、K-L散度、K-S距离、平均绝对误差（MAE）、均方误差（MSE）和互信息。预期分布基于双重响应统计量（均值和标准差）被投影为正态分布。MI方法比较沿两个轴计算的条件分布梯度的单调性指数，并展示梯度符号变化的次数。两种方法均假设……的随机特性（原文在此处截断）。

    arXiv:2603.26024v2 Announce Type: replace  Abstract: Identification of causal directionality in bivariate numerical data is a fundamental research problem with important practical implications. This paper presents two alternative methods to identify direction of causation by considering conditional distributions: (1) Anticipated Asymmetric Geometries (AAG) and (2) Monotonicity Index (MI). The AAG method compares the actual conditional distributions to anticipated ones along two variables. Different comparison metrics, such as Pearson correlation, cosine distance, Jaccard index, K-L divergence, K-S distance, MAE, MSE, and mutual information have been evaluated. Anticipated distributions have been projected as normal based on dual response statistics: mean and standard deviation. The MI method compares the calculated monotonicity indexes of the gradients of conditional distributions along two axes and exhibits counts of gradient sign changes. Both methods assume stochastic properties of 
    
[^236]: KernelFoundry：硬件感知的演化式GPU内核优化

    KernelFoundry: Hardware-aware evolutionary GPU kernel optimization

    [https://arxiv.org/abs/2603.12440](https://arxiv.org/abs/2603.12440)

    提出了KernelFoundry演化框架，通过MAP-Elites质量多样性搜索、提示与内核共同演化的元提示机制以及基于模板的参数优化，实现硬件感知的高效GPU内核自动优化。

    

    GPU内核优化对大语言模型（LLM）的挑战超越了标准编程任务，因为它需要理解硬件架构、并行计算优化策略以及性能剖析输出。然而，大多数利用LLM生成内核的现有方法都采用标准的提示与反馈循环，仅通过性能剖析反馈来考虑硬件因素。我们提出了KernelFoundry，这是一个能够高效探索GPU内核空间的演化式框架，其核心包括：（1）采用具有内核特定行为维度的MAP-Elites质量多样性搜索以维持持续探索；（2）元提示演化机制，使提示词与内核共同演化，以发现任务特定的优化策略；（3）基于模板的参数优化方法，将内核调优以适配具体输入和硬件。我们在Kernel-Bench、robust-kbench及自定义任务上对该框架进行了评估，生成SYCL内核作为跨平台GPU编程范式。

    arXiv:2603.12440v2 Announce Type: replace-cross  Abstract: GPU kernel optimization challenges LLMs beyond standard coding tasks, as it requires an understanding of hardware architecture, parallel computing optimization strategies, and profiling outputs. However, most existing approaches leveraging LLMs for kernel generation apply standard prompting and feedback loops, considering hardware only through profiling feedback. We introduce KernelFoundry, an evolutionary framework that efficiently explores the space of GPU kernels through (1) MAP-Elites quality diversity search with kernel-specific behavioral dimensions to sustain exploration; (2) meta-prompt evolution that co-evolves prompts with kernels to uncover task-specific optimization strategies, and (3) a template-based parameter optimization approach to tune kernels to inputs and hardware. We evaluate this framework on Kernel-Bench, robust-kbench and custom tasks, generating SYCL kernels as a cross-platform GPU programming paradigm,
    
[^237]: 简化以增强：在谱社区检测中以更少步骤达到信息论界

    Simplify to Amplify: Achieving Information-Theoretic Bounds with Fewer Steps in Spectral Community Detection

    [https://arxiv.org/abs/2602.17104](https://arxiv.org/abs/2602.17104)

    本文提出一种简化的谱社区检测算法，通过去除不必要的预处理步骤并利用邻接矩阵第二特征向量的特性，在双社区随机块模型中实现了接近信息论极限的误差界，超越了现有方法。

    

    我们提出了一种在恒定边密度假设下，用于双社区随机块模型（SBM）社区检测的精简谱算法。通过消除非必要的预处理步骤来降低算法复杂度，我们的方法直接利用邻接矩阵的谱特性。我们证明了该算法利用第二特征向量的特定性质，实现了接近信息论极限的改进误差界，相比现有方法有显著提升。理论分析表明，我们的错误率比文献中先前报道的界更为紧致。全面的实验验证证实了我们的理论发现，并展示了这种简化方法在实践中的有效性。我们的结果表明，算法简化而非增加复杂度，能够带来两方面的优势（原文此处截断）。

    arXiv:2602.17104v3 Announce Type: replace-cross  Abstract: We propose a streamlined spectral algorithm for community detection in the two-community stochastic block model (SBM) under constant edge density assumptions. By reducing algorithmic complexity through the elimination of non-essential preprocessing steps, our method directly leverages the spectral properties of the adjacency matrix. We demonstrate that our algorithm exploits specific characteristics of the second eigenvector to achieve improved error bounds that approach information-theoretic limits, representing a significant improvement over existing methods. Theoretical analysis establishes that our error rates are tighter than previously reported bounds in the literature. Comprehensive experimental validation confirms our theoretical findings and demonstrates the practical effectiveness of the simplified approach. Our results suggest that algorithmic simplification, rather than increasing complexity, can lead to both comput
    
[^238]: 损失最了解：通过损失轨迹检测视频中的标注错误

    Loss Knows Best: Detecting Annotation Errors in Videos via Loss Trajectories

    [https://arxiv.org/abs/2602.15154](https://arxiv.org/abs/2602.15154)

    本文提出以不相交参考集训练的检查点上的累积样本损失（CSL）作为动态指纹，实现对视频数据集中语义误标和时间乱序两类标注错误的样本外自动审计。

    

    可靠的视频理解需要高质量的视频数据集，这些数据集既能提供精确的语义标签，又能提供时间上一致的标注。在密集标注的视频中检测标注错误极具挑战性，因为错误可能来自语义误标（标签与视觉内容不符）或时间乱序（原本合理的标签却违反了程序性进程）。训练动态已被用于识别误标的训练样本，但主要针对静态样本。我们研究检查点损失动态，用于对时间标注视频进行样本外审计。我们将累积样本损失（Cumulative Sample Loss, CSL）计算为一个审计帧在与审计集不相交的参考集上训练的各检查点上的平均标注条件损失。CSL 充当一种动态指纹，捕捉标注与模型学到的视觉-时间结构之间的持续不一致。高 CSL 的帧随后……

    arXiv:2602.15154v2 Announce Type: replace-cross  Abstract: Reliable video understanding requires high-quality video datasets that can provide both precise semantic labels and temporally consistent annotations. Detecting annotation errors in densely labeled videos is challenging because errors may arise from semantic **mislabeling**, where labels disagree with visual content, or temporal **disordering**, where otherwise plausible labels violate procedural progression. Training dynamics have been used to identify mislabeled training examples primarily for static samples. We investigate checkpoint loss dynamics for **out-of-sample auditing** of temporally annotated videos. We compute **Cumulative Sample Loss (CSL)** as the mean annotation-conditioned loss of an audit frame across checkpoints trained on a *disjoint* reference set. CSL acts as a dynamic fingerprint and captures the persistent disagreement between its annotation and learned visual-temporal structure. High-CSL frames are then
    
[^239]: FedPS：基于聚合统计的结构化数据联邦预处理框架

    FedPS: Federated Preprocessing for structured data via aggregated Statistics

    [https://arxiv.org/abs/2602.10870](https://arxiv.org/abs/2602.10870)

    提出了FedPS框架，利用数据草图技术在联邦环境下通过聚合统计实现结构化数据的高效预处理（包括特征缩放、编码、离散化和缺失值插补），解决了联邦学习中预处理阶段被忽视的问题。

    

    联邦学习（FL）使多个参与方能够在不共享原始数据的情况下协同训练机器学习模型。然而，在训练之前，必须对数据进行预处理，以解决缺失值、格式不一致和特征尺度异构等问题。这一预处理阶段对模型性能至关重要，但在联邦学习研究中却很大程度上被忽视。在实际的联邦学习系统中，隐私约束禁止将原始数据集中起来，而通信效率也给分布式预处理带来了进一步的挑战。我们提出了FedPS，一个基于聚合统计的联邦数据预处理框架。FedPS利用数据草图技术高效地汇总本地数据集，同时保留关键的统计信息。基于这些汇总信息，我们设计了用于特征缩放、编码、离散化和缺失值插补的联邦算法，并将预处理相关的模型扩展到……（摘要不完整）

    arXiv:2602.10870v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) enables multiple parties to collaboratively train machine learning models without sharing raw data. However, before training, data must be preprocessed to address missing values, inconsistent formats, and heterogeneous feature scales. This preprocessing stage is critical for model performance but is largely overlooked in FL research. In practical FL systems, privacy constraints prohibit centralizing raw data, while communication efficiency introduces further challenges for distributed preprocessing. We introduce FedPS, a framework for federated data preprocessing based on aggregated statistics. FedPS leverages data-sketching techniques to efficiently summarize local datasets while preserving essential statistical information. Building on these summaries, we design federated algorithms for feature scaling, encoding, discretization, and missing-value imputation, and extend preprocessing-related models such
    
[^240]: 超越Softmax与Entmax的熵生成注意力：Kaniadakis算子与倒数对称Abe算子

    Entropy-Generated Attention Beyond Softmax and Entmax: Kaniadakis and Reciprocal-Symmetric Abe Operators

    [https://arxiv.org/abs/2602.08216](https://arxiv.org/abs/2602.08216)

    该论文从广义统计熵出发推导出两种新型注意力算子——基于Kaniadakis熵的幂律衰减全支撑归一化算子和基于Abe熵的倒数对称算子，并证明这些平稳分布律源于概率单纯形上的Fisher度量拉格朗日量，从而将Softmax注意力统一推广到超越指数衰减与精确截断的框架。

    

    我们从广义统计熵推导出两种注意力算子。Kaniadakis熵产生一种精确的全支撑归一化，其权重和低分敏感度按代数（幂律）方式衰减，而非像Softmax那样指数衰减，也不像entmax那样精确截断。经典Abe熵则产生一个隐式的倒数对称算子。当$q=e^\epsilon$时，对合变换$q\leftrightarrow q^{-1}$消除了关于Softmax的所有奇数阶修正；我们得到了归一化的二阶和四阶项，包括归一化乘子的形变。这些平稳分布律源自概率单纯形上基于Fisher度量的拉格朗日量，其Shannon扇区退化为缩放点积Softmax。我们还给出一个切梯度检验，用于判定改变熵是改变了注意力分布的轮廓，还是仅改变其尺度。Rényi熵和双参数Sharma–Mittal熵则保持Tsallis–entmax的逆梯度形状。

    arXiv:2602.08216v3 Announce Type: replace  Abstract: We derive two attention operators from generalized statistical entropies. Kaniadakis entropy yields an exact full-support normalization whose weights and low-score sensitivities decay algebraically, rather than exponentially as in Softmax or by exact truncation as in entmax. Classical Abe entropy yields an implicit reciprocal-symmetric operator. With $q=e^\epsilon$, the involution $q\leftrightarrow q^{-1}$ removes every odd correction about Softmax; we obtain the normalized second- and fourth-order terms, including the deformation of the normalization multiplier. These stationary laws follow from a Fisher-metric Lagrangian on the probability simplex, whose Shannon sector recovers scaled dot-product Softmax. We also give a tangent-gradient test for deciding whether changing the entropy changes the attention profile or only its scale. R\'enyi and two-parameter Sharma--Mittal entropies retain the Tsallis--entmax inverse-gradient shape, 
    
[^241]: F-GRPO：不要让你的策略只学显而易见的而遗忘稀有的

    F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare

    [https://arxiv.org/abs/2602.06717](https://arxiv.org/abs/2602.06717)

    本文提出 F-GRPO，借鉴 Focal loss 设计了难度感知的缩放系数，对高成功率采样组的更新降权，从而防止 RLVR 训练中的策略因组采样遗漏稀有正确解而过度集中于常见解。

    

    具有可验证奖励的强化学习（RLVR）通常基于组采样来估计优势并稳定策略更新。在实践中，受计算资源限制，往往无法使用非常大的组，因此训练只能在有限的 rollout 集合上进行，而这些集合只能强化其中所暴露出的正确行为。在实际的组大小下，更新可能会漏掉稀有的正确轨迹，同时仍包含混合奖励，从而将概率集中到更常见的已采样解上。我们推导了这类提示局部“尾部遗漏”事件发生的概率与组大小之间的函数关系，并证明其呈现非单调行为；在类别化抽象中，我们刻画了即使总正确概率质量在增长，未被采样的正确概率质量也可能收缩的现象。受此分析启发，我们借鉴 Focal loss 提出了一种难度感知的缩放系数，用于降低对高成功率已采样组的更新权重。实验上，类别化模拟说明了……

    arXiv:2602.06717v3 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is commonly based on group sampling to estimate advantages and stabilize policy updates. In practice, computational limits often rule out very large groups, so training proceeds with finite rollout sets that can reinforce only the correct behavior they expose. At practical group sizes, updates can miss rare-correct trajectories while still containing mixed rewards, concentrating probability on more common sampled solutions. We derive the probability of such prompt-local tail-miss events as a function of group size, showing non-monotonic behavior, and in the categorical abstraction characterize how unsampled-correct mass can shrink even as total correct mass grows. Motivated by this analysis, we propose a difficulty-aware scaling coefficient, inspired by Focal loss, that down-weights updates on high-success sampled groups. Empirically, categorical simulation illustrates the s
    
[^242]: 联邦学习中破坏模型置信度的温度缩放攻击

    Temperature Scaling Attack Disrupting Model Confidence in Federated Learning

    [https://arxiv.org/abs/2602.06638](https://arxiv.org/abs/2602.06638)

    该论文提出了温度缩放攻击（TSA），一种新型联邦学习训练时攻击，通过学习率-温度耦合机制在保持模型准确率不变的情况下破坏置信度校准，从而威胁依赖置信度信号的任务关键型系统的风险决策逻辑。

    

    预测置信度是任务关键型系统中的基础控制信号，直接支配着诸如升级上报、拒绝预测和保守回退等风险感知逻辑。虽然以往的联邦学习攻击主要针对准确率或植入后门，但我们识别出置信度校准作为一个独特的攻击目标。我们提出了温度缩放攻击（TSA），这是一种在保持准确率的同时降低校准质量的训练时攻击。通过在本地训练中注入带有学习率-温度耦合的温度缩放机制，TSA能够在保持预测准确率和常见优化信号接近良性训练的前提下改变模型置信度。我们在非独立同分布设置下提供了收敛性分析，表明该耦合机制控制了主要更新规模，同时留下有界的温度诱导残差，从而产生带有额外残差项的标准非凸联邦学习收敛结构。

    arXiv:2602.06638v3 Announce Type: replace-cross  Abstract: Predictive confidence serves as a foundational control signal in mission-critical systems, directly governing risk-aware logic such as escalation, abstention, and conservative fallback. While prior federated learning attacks predominantly target accuracy or implant backdoors, we identify confidence calibration as a distinct attack objective. We present the Temperature Scaling Attack (TSA), a training-time attack that degrades calibration while preserving accuracy. By injecting temperature scaling with learning rate-temperature coupling during local training, TSA shifts model confidence while keeping predictive accuracy and common optimization signals close to benign training. We provide a convergence analysis under non-IID settings, showing that the coupling controls the primary update scale while leaving a bounded temperature-induced residual, yielding the standard non-convex FL convergence structure with an additional residua
    
[^243]: 深度网络从局部统计中学习解析等深度上下文无关语言

    Deep networks learn to parse uniform-depth context-free languages from local statistics

    [https://arxiv.org/abs/2602.06065](https://arxiv.org/abs/2602.06065)

    该研究引入了一类可调节歧义程度和跨尺度相关结构的概率上下文无关文法，揭示了深度网络能够仅从局部统计特征中学习解析语言的层次结构。

    

    理解如何仅从句子中学习语言的结构是认知科学与机器学习中的一个核心问题。对大型语言模型（LLMs）内部表示的研究支持了它们在预测下一个词时解析文本的能力，同时以独立于表面形式的方式表征语义概念。然而，究竟是哪些数据统计特征使这些能力成为可能，以及需要多少数据，在很大程度上仍是未知的。概率上下文无关文法（PCFGs）为研究这些问题提供了一个易于处理的测试平台。然而，先前的工作要么集中于对训练后网络所使用的类解析算法进行事后表征，要么集中于具有固定语法的PCFG的可学习性——在这种情况下解析本身是不必要的。在此，我们引入了一类可调节的PCFG，其中歧义程度和跨尺度的相关结构均可被控制……

    arXiv:2602.06065v4 Announce Type: replace-cross  Abstract: Understanding how the structure of language can be learned from sentences alone is a central question in both cognitive science and machine learning. Studies of the internal representations of Large Language Models (LLMs) support their ability to parse text when predicting the next word, while representing semantic notions independently of surface form. Yet, which data statistics make these feats possible, and how much data is required, remain largely unknown. Probabilistic context-free grammars (PCFGs) provide a tractable testbed for studying these questions. However, prior work has focused either on the post-hoc characterization of the parsing-like algorithms used by trained networks; or on the learnability of PCFGs with fixed syntax, where parsing is unnecessary. Here, we (i) introduce a tunable class of PCFGs in which both the degree of ambiguity and the correlation structure across scales can be controlled; (ii) provide a 
    
[^244]: MSign：一种通过稳定秩恢复防止大语言模型训练不稳定性的优化器

    MSign: An Optimizer Preventing Training Instability in Large Language Models via Stable Rank Restoration

    [https://arxiv.org/abs/2602.01734](https://arxiv.org/abs/2602.01734)

    提出MSign优化器，通过周期性矩阵符号运算恢复权重矩阵稳定秩，以低于7.0%的计算开销有效防止大语言模型预训练中的梯度爆炸与训练崩溃。

    

    训练不稳定性仍然是大语言模型（LLM）预训练中的一个关键挑战，通常表现为突然的梯度爆炸，浪费大量计算资源。我们研究了通过μP缩放的500万参数NanoGPT模型的训练失败问题，识别出崩溃发生前的两个关键现象：（1）权重矩阵稳定秩（Frobenius范数平方与谱范数平方之比）的快速下降；（2）相邻层雅可比矩阵之间对齐程度的不断增加。我们从理论上证明，这两个条件共同导致梯度范数随网络深度呈指数级增长。为了打破这种不稳定机制，我们提出了MSign，这是一种新的优化器，通过周期性地应用矩阵符号运算来恢复稳定秩。在500万至30亿参数模型上的实验表明，MSign能够有效防止训练失败，且计算开销低于7.0%。

    arXiv:2602.01734v2 Announce Type: replace  Abstract: Training instability remains a critical challenge in large language model (LLM) pretraining, often manifesting as sudden gradient explosions that waste significant computational resources. We study training failures in a 5M-parameter NanoGPT model scaled via $\mu$P, identifying two key phenomena preceding collapse: (1) rapid decline in weight matrix stable rank (ratio of squared Frobenius norm to squared spectral norm), and (2) increasing alignment between adjacent layer Jacobians. We prove theoretically that these two conditions jointly cause exponential gradient norm growth with network depth. To break this instability mechanism, we propose MSign, a new optimizer that periodically applies matrix sign operations to restore stable rank. Experiments on models from 5M to 3B parameters demonstrate that MSign effectively prevents training failures with a computational overhead of less than 7.0%.
    
[^245]: 非平稳函数双层优化

    Non-Stationary Functional Bilevel Optimization

    [https://arxiv.org/abs/2601.15363](https://arxiv.org/abs/2601.15363)

    提出首个面向非平稳函数双层优化的算法SmoothFBO，通过时间平滑的随机超梯度估计器降低方差，实现具有次线性遗憾的稳定更新，并在非平稳超参数优化和基于模型的强化学习中优于现有方法。

    

    函数双层优化（FBO）为函数空间中的分层学习提供了一个强大的框架，然而现有方法仅限于静态离线设置，在在线、非平稳场景中表现欠佳。我们提出了SmoothFBO，这是首个同时具备理论保证和实际可扩展性的非平稳FBO算法。SmoothFBO引入了一种时间平滑的随机超梯度估计器，通过窗口参数降低方差，从而实现具有次线性遗憾的稳定外循环更新。重要的是，经典的参数化双层优化是我们框架的一个特例，这使得SmoothFBO成为向在线、非平稳设置的自然推广。在实验中，SmoothFBO在非平稳超参数优化和基于模型的强化学习中始终优于现有的FBO方法，展示了其实际有效性。这些结果共同确立了Smoo……

    arXiv:2601.15363v3 Announce Type: replace-cross  Abstract: Functional bilevel optimization (FBO) provides a powerful framework for hierarchical learning in function spaces, yet current methods are limited to static offline settings and perform suboptimally in online, non-stationary scenarios. We propose SmoothFBO, the first algorithm for non-stationary FBO with both theoretical guarantees and practical scalability. SmoothFBO introduces a time-smoothed stochastic hypergradient estimator that reduces variance through a window parameter, enabling stable outer-loop updates with sublinear regret. Importantly, the classical parametric bilevel case is a special reduction of our framework, making SmoothFBO a natural extension to online, non-stationary settings. Empirically, SmoothFBO consistently outperforms existing FBO methods in non-stationary hyperparameter optimization and model-based reinforcement learning, demonstrating its practical effectiveness. Together, these results establish Smoo
    
[^246]: 线性化子空间精化框架：揭示已训练神经网络中隐藏的精度

    Linearized subspace refinement framework to expose hidden accuracy in trained neural networks

    [https://arxiv.org/abs/2601.13989](https://arxiv.org/abs/2601.13989)

    提出了一种与架构无关的训练后框架LSR，通过在雅可比矩阵定义的低维子空间中求解约化最小二乘问题对已训练神经网络进行线性化修正，从而突破梯度训练导致的精度平台期，显著提升预测精度。

    

    在科学机器学习任务中，采用基于梯度的方法训练的神经网络常常表现出由优化过程引起的精度平台期。我们提出了线性化子空间精化方法，这是一种与架构无关的训练后框架，它在固定的已训练状态下利用局部线性化模型。通过在雅可比矩阵定义的低维空间中求解一个约化的直接最小二乘问题，LSR计算出一个子空间最优的线性化修正，并得到精度显著提升的精化预测器。在函数逼近、数据驱动的算子学习、物理信息算子微调以及含噪逆问题等各类任务中，LSR表明标准的非线性训练结果可能远高于该子空间可达到的误差水平。即使对于由局部线性化产生的凸二次问题，使用标准迭代优化器求解时仍会出现类似的精度平台期，这表明数值病态性是其主要原因。

    arXiv:2601.13989v2 Announce Type: replace  Abstract: Neural networks trained by gradient-based methods often exhibit optimization-induced accuracy plateaus in scientific machine learning tasks. We present Linearized Subspace Refinement (LSR), an architecture-agnostic post-training framework that exploits the local linearized model at a fixed trained state. By solving a reduced direct least-squares problem in a Jacobian-defined low-dimensional space, LSR computes a subspace-optimal linearized correction and yields a refined predictor with markedly improved accuracy. Across function approximation, data-driven operator learning, physics-informed operator fine-tuning, and noisy inverse problems, LSR shows that standard nonlinear training can remain far above this subspace-attainable error level. Similar accuracy plateaus persist even for the convex quadratic problem from local linearization when solved with standard iterative optimizers, identifying numerical ill-conditioning as a primary 
    
[^247]: 先想象后规划：基于世界模型自适应前瞻的智能体学习

    Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models

    [https://arxiv.org/abs/2601.08955](https://arxiv.org/abs/2601.08955)

    提出了ITP统一框架，让智能体策略模型与世界模型交互生成多步想象轨迹，并通过权衡最终目标与任务进度的自适应前瞻机制，充分释放世界模型在复杂任务规划中的潜力。

    

    世界模型的最新进展在建模环境状态的未来动态方面展现出巨大潜力，使智能体无需访问真实环境即可进行推理与行动。然而，当前方法主要执行单步或固定时域的推演，其在复杂任务规划中的潜力尚未得到充分挖掘。我们提出了“先想象后规划”（Imagine-then-Plan, ITP），这是一个通过前瞻想象进行智能体学习的统一框架，其中智能体的策略模型与学习到的世界模型进行交互，生成多步“想象”轨迹。由于想象时域可能因任务和阶段的不同而变化，我们引入了一种新颖的自适应前瞻机制，通过权衡最终目标与任务进度来确定想象步长。由此得到的想象轨迹提供了关于未来后果的丰富信号，例如已取得的进展和潜在的冲突，这些信号与当前观测相融合，构成了部分可观测与想象的（摘要在此处截断）

    arXiv:2601.08955v3 Announce Type: replace-cross  Abstract: Recent advances in world models have shown promise for modeling future dynamics of environmental states, enabling agents to reason and act without accessing real environments. Current methods mainly perform single-step or fixed-horizon rollouts, leaving their potential for complex task planning under-exploited. We propose Imagine-then-Plan (\texttt{ITP}), a unified framework for agent learning via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step ``imagined'' trajectories. Since the imagination horizon may vary by tasks and stages, we introduce a novel adaptive lookahead mechanism by trading off the ultimate goal and task progress. The resulting imagined trajectories provide rich signals about future consequences, such as achieved progress and potential conflicts, which are fused with current observations, formulating a partially \textit{observable} and \textit{imag
    
[^248]: 基于贝叶斯算子推断的参数化微分系统数据驱动降阶模型主动学习方法

    Active learning for data-driven reduced models of parametric differential systems with Bayesian operator inference

    [https://arxiv.org/abs/2601.00038](https://arxiv.org/abs/2601.00038)

    该论文提出了一种基于贝叶斯算子推断的主动学习框架，通过量化预测不确定性来智能选择训练参数，从而以最少的数据成本提升参数化动力系统数据驱动降阶模型的精度。

    

    这项工作开发了一个主动学习框架，用于智能地丰富参数化动力系统的数据驱动降阶模型（ROMs），这些模型可作为数字孪生中虚拟资产的基础。数据驱动降阶模型是可解释且计算高效的科学机器学习模型，旨在保留复杂动力系统模拟的底层物理特性。由于数据驱动降阶模型的质量对有限训练数据的质量非常敏感，我们致力于识别出能够利用相关训练数据构建最佳参数化降阶模型的训练参数。我们的方法采用算子推断方法，这是一种基于回归的策略，可针对一大类问题的特定参数结构进行定制。我们建立了参数化算子推断的概率版本，将学习问题转化为贝叶斯线性回归问题，并通过预测不确定性来指导主动学习过程中的数据采样。

    arXiv:2601.00038v2 Announce Type: replace-cross  Abstract: This work develops an active learning framework to intelligently enrich data-driven reduced-order models (ROMs) of parametric dynamical systems, which can serve as the foundation of virtual assets in a digital twin. Data-driven ROMs are explainable, computationally efficient scientific machine learning models that aim to preserve the underlying physics of complex dynamical simulations. Since the quality of data-driven ROMs is sensitive to the quality of the limited training data, we seek to identify training parameters for which using the associated training data results in the best possible parametric ROM. Our approach uses the operator inference methodology, a regression-based strategy which can be tailored to particular parametric structure for a large class of problems. We establish a probabilistic version of parametric operator inference, casting the learning problem as a Bayesian linear regression. Prediction uncertaintie
    
[^249]: DuaDeep-SeqAffinity：用于基于序列的抗体-抗原亲和力预测的双分支三流深度学习框架

    DuaDeep-SeqAffinity: Dual-Branch Deep Learning for Tri-Stream Sequence-Based Antibody--Antigen Affinity Prediction

    [https://arxiv.org/abs/2512.22007](https://arxiv.org/abs/2512.22007)

    该论文提出仅基于氨基酸序列的双分支三流深度学习框架DuaDeep-SeqAffinity，通过对三条序列流分别进行冻结ESM-2嵌入并经Transformer与CNN并行分支后期融合，直接预测抗体-抗原结合亲和力，无需三维结构信息，并在AbRank基准上显著优于单分支模型。

    

    DuaDeep-SeqAffinity是一个仅基于序列的深度学习框架，可直接从一级氨基酸序列预测抗体-抗原结合亲和力，从而避免了已解析三维结构的高成本和稀缺性问题。该框架将抗原以及抗体重链和轻链作为三条独立的流进行处理，每条流均使用冻结的ESM-2蛋白质语言模型进行嵌入，并分别通过并行的Transformer和卷积神经网络（CNN）分支后进行后期融合。这种解耦设计旨在保留局部互补决定区（CDR）信号，而整体式编码器可能会稀释这些信号。在AbRank基准数据集的序列不相交划分上，该模型实现了0.683的皮尔逊相关系数、0.460的R²以及0.895的成对排序AUC，显著优于单分支消融模型（配对t检验，p < 0.05）。注意力图和基于梯度的显著性分析进一步表明，该模型优先……（摘要原文在此处被截断）

    arXiv:2512.22007v2 Announce Type: replace  Abstract: DuaDeep-SeqAffinity is a sequence-only deep learning framework that predicts antibody--antigen binding affinity directly from primary amino acid sequences, avoiding the cost and scarcity of resolved three-dimensional structures. The antigen and the antibody heavy and light chains are processed as three independent streams, each embedded with a frozen ESM-2 protein language model and passed through parallel Transformer and convolutional neural network (CNN) branches before late fusion, a decoupled design intended to preserve local complementarity-determining region (CDR) signal that monolithic encoders can dilute. On a sequence-disjoint split of the AbRank benchmark, the model achieves a Pearson correlation of 0.683, an R^2 of 0.460, and a pairwise ranking AUC of 0.895, significantly outperforming single-branch ablations (paired t-test, p < 0.05). Attention-map and gradient-based saliency analyses further show that the model preferent
    
[^250]: FADTI：基于傅里叶与注意力驱动的扩散模型用于多变量时间序列插补

    FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation

    [https://arxiv.org/abs/2512.15116](https://arxiv.org/abs/2512.15116)

    提出FADTI框架，通过傅里叶偏置投影（FBP）模块在扩散去噪过程中注入可学习的频率感知偏置，并支持DFT、STFT、FSST多种频谱变换，从而有效提升多变量时间序列插补对周期性和非平稳模式的恢复能力。

    

    多变量时间序列插补在医疗保健、交通预测和生物建模等应用中至关重要，因为这些场景中传感器故障和不规则采样导致普遍存在缺失值。现有的基于Transformer和扩散模型的插补方法虽然性能优异，但它们主要依赖时域建模，缺乏用于恢复结构化时间间隙的自适应频谱偏置。我们提出了FADTI，一个用于多变量时间序列插补的傅里叶与注意力驱动的扩散框架。FADTI引入了傅里叶偏置投影（FBP）模块，在去噪过程中向中间隐藏状态注入可学习的频率感知偏置。它将中间隐藏状态投影到傅里叶基上，避免了从掩码或零填充输入中进行直接频谱估计。通过DFT、STFT和FSST的实例化，FBP能够捕获全局周期性、局部时频变化以及非平稳振荡。

    arXiv:2512.15116v3 Announce Type: replace-cross  Abstract: Multivariate time series imputation is fundamental in applications such as healthcare, traffic forecasting, and biological modeling, where sensor failures and irregular sampling lead to pervasive missing values. Existing Transformer- and diffusion-based imputers achieve strong performance, but they often rely mainly on time-domain modeling and lack adaptive spectral bias for recovering structured temporal gaps. We propose FADTI, a Fourier- and attention-driven diffusion framework for multivariate time series imputation. FADTI introduces a Fourier Bias Projection (FBP) module that injects learnable frequency-aware bias into intermediate hidden states during denoising. It projects intermediate hidden states onto Fourier bases, avoiding direct spectral estimation from masked or zero-filled inputs. With DFT, STFT, and FSST instantiations, FBP captures global periodicity, localized time--frequency variations, and non-stationary osci
    
[^251]: 扩散过程随机控制的自适应划分与学习

    Adaptive Partitioning and Learning for Stochastic Control of Diffusion Processes

    [https://arxiv.org/abs/2512.14991](https://arxiv.org/abs/2512.14991)

    本文提出一种自适应划分状态-动作空间的基于模型的强化学习算法，用于求解无界连续状态空间下受控扩散过程的随机控制问题，并建立了包含新定义的“缩放维度”的遗憾界。

    

    我们研究针对具有无界连续状态空间、有界连续动作以及多项式增长奖励的受控扩散过程的强化学习问题，这类设定自然出现在金融、经济学和运筹学领域。为了克服连续且高维空间带来的挑战，我们提出了一种基于模型的算法，该算法对联合状态-动作空间进行自适应划分。该算法在每个分区内维护漂移、波动率和奖励的估计器，并在估计偏差超过统计置信度时细化离散化。这种自适应方案平衡了探索与近似，使得在无界域中实现高效学习成为可能。我们的分析建立了遗憾界，该遗憾界取决于问题的时间跨度、状态维度、奖励增长阶数，以及我们针对无界扩散过程新定义的“缩放维度”概念。这些界限可以恢复现有有界设定下的已有结果。

    arXiv:2512.14991v3 Announce Type: replace  Abstract: We study reinforcement learning for controlled diffusion processes with unbounded continuous state spaces, bounded continuous actions, and polynomially growing rewards: settings that arise naturally in finance, economics, and operations research. To overcome the challenges of continuous and high-dimensional domains, we introduce a model-based algorithm that adaptively partitions the joint state-action space. The algorithm maintains estimators of drift, volatility, and rewards within each partition, refining the discretization whenever estimation bias exceeds statistical confidence. This adaptive scheme balances exploration and approximation, enabling efficient learning in unbounded domains. Our analysis establishes regret bounds that depend on the problem horizon, state dimension, reward growth order, and a newly defined notion of zooming dimension tailored to unbounded diffusion processes. The bounds recover existing results for bou
    
[^252]: 混合数据聚类综述与挑战

    Mixed Data Clustering Survey and Challenges

    [https://arxiv.org/abs/2512.03070](https://arxiv.org/abs/2512.03070)

    本文提出了一种基于预拓扑空间的混合数据聚类方法，能够有效处理同时包含数值型和分类型变量的异构数据，并提供层次化、可解释的聚类结果。

    

    大数据时代的到来改变了各行业管理和分析信息的方式，开启了数据体量、速度和种类都前所未有的时代。在这一背景下，混合数据聚类已成为一项关键挑战，需要能够有效利用异构数据类型（包括数值型和分类型变量）的创新方法。传统聚类技术通常是为同质数据集设计的，往往难以捕捉混合数据所带来的额外复杂性，这凸显了专门针对此类场景设计方法的必要性。在这一背景下，层次化和可解释的算法尤其具有价值，因为它们能够提供结构化、可解释的聚类结果，从而支持明智的决策。本文介绍了一种基于预拓扑空间（pretopological spaces）的聚类方法。此外，本文还与经典的数值聚类算法进行了基准测试比较……

    arXiv:2512.03070v2 Announce Type: replace-cross  Abstract: The advent of the big data paradigm has transformed how industries manage and analyze information, ushering in an era of unprecedented data volume, velocity, and variety. Within this landscape, mixed-data clustering has become a critical challenge, requiring innovative methods that can effectively exploit heterogeneous data types, including numerical and categorical variables. Traditional clustering techniques, typically designed for homogeneous datasets, often struggle to capture the additional complexity introduced by mixed data, underscoring the need for approaches specifically tailored to this setting. Hierarchical and explainable algorithms are particularly valuable in this context, as they provide structured, interpretable clustering results that support informed decision-making. This paper introduces a clustering method grounded in pretopological spaces. In addition, benchmarking against classical numerical clustering al
    
[^253]: 注意力轨迹作为深度强化学习的诊断轴

    Attention Trajectories as a Diagnostic Axis for Deep Reinforcement Learning

    [https://arxiv.org/abs/2511.20591](https://arxiv.org/abs/2511.20591)

    提出一种基于显著性图定量分析的诊断框架，通过构建分层注意力画像与注意力轨迹，揭示深度强化学习智能体的算法特异性注意力偏差、非预期奖励驱动策略及冗余感官通道过拟合问题。

    

    深度强化学习智能体中特征依赖的出现与演化机制至今仍知之甚少。在此，我们引入了一个方法论框架，通过显著性图的定量分析来研究学习过程。该方法将显著性信息在物体和模态层面进行聚合，形成分层注意力画像，量化智能体如何在时间维度上分配注意力，从而在整个训练过程中形成注意力轨迹。随后，这些画像在受控条件下进行跨条件比较，与行为测量结果相关联，并通过不同的显著性方法进行复现，以评估研究发现的稳健性。将该框架应用于Atari 2600基准、自定义Pong环境以及视觉运动任务中的生物力学用户模拟，该框架揭示了算法特异性的注意力偏差，诊断出非预期的奖励驱动策略，以及对冗余感官通道的过拟合现象。

    arXiv:2511.20591v3 Announce Type: replace  Abstract: The emergence and evolution of feature reliance in deep reinforcement learning agents remain poorly understood. Here, we introduce a methodological framework for analyzing the learning process through quantitative analysis of saliency maps. This approach aggregates saliency information at the object and modality level into hierarchical attention profiles, quantifying how agents allocate attention over time, thereby forming attention trajectories throughout training. These profiles are then compared across controlled conditions, connected to behavioral measurements and reproduced with different saliency methods to assess the robustness of the findings. Applied to Atari 2600 benchmarks, custom Pong environments, and biomechanical user simulations in visuomotor tasks, this framework uncovers algorithm-specific attention biases, diagnosed unintended reward-driven strategies, and overfitting to redundant sensory channels. These patterns c
    
[^254]: 异质性处理效应估计器的可靠选择

    Reliable Selection of Heterogeneous Treatment Effect Estimators

    [https://arxiv.org/abs/2511.18464](https://arxiv.org/abs/2511.18464)

    提出了一种无需真实处理效应数据的异质性处理效应估计器选择方法，通过交叉拟合指数加权检验统计量和双向样本分割实现渐近族错误率控制，在多个基准数据集上显著减少错误选择。

    

    我们研究了在处理效应本质上不可观测的设置下，从一组候选方法中选择最佳异质性处理效应（HTE）估计器的问题。我们将估计器选择问题构建为一个多重检验问题，并提出了一种基于交叉拟合、指数加权检验统计量的无需真实值（ground-truth-free）的选择程序。我们方法的一个关键组成部分是双向样本分割方案，该方案将干扰参数估计与权重学习解耦，并确保了有效推断所需的稳定性。利用基于稳定性的中心极限定理，我们在温和的正则性条件下建立了渐近族错误率控制。在实证方面，在ACIC 2016、IHDP和Twins基准数据集上，与常用方法相比，我们的程序提供了可靠的错误控制，同时大幅减少了错误选择，这表明我们的方法即使在缺乏真实值的情况下也是可行且有效的。

    arXiv:2511.18464v2 Announce Type: replace-cross  Abstract: We study the problem of selecting the best heterogeneous treatment effect (HTE) estimator from a collection of candidates in settings where the treatment effect is fundamentally unobserved. We cast estimator selection as a multiple testing problem and introduce a ground-truth-free procedure based on a cross-fitted, exponentially weighted test statistic. A key component of our method is a two-way sample splitting scheme that decouples nuisance estimation from weight learning and ensures the stability required for valid inference. Leveraging a stability-based central limit theorem, we establish asymptotic familywise error rate control under mild regularity conditions. Empirically, our procedure provides reliable error control while substantially reducing false selections compared with commonly used methods across ACIC 2016, IHDP, and Twins benchmarks, demonstrating that our method is feasible and powerful even without ground-trut
    
[^255]: AnyBox：面向机器人操作的高效零样本箱体9自由度位姿估计

    AnyBox: Efficient Zero-Shot 9DoF Pose Estimation of Boxes for Robotic Manipulation

    [https://arxiv.org/abs/2511.15884](https://arxiv.org/abs/2511.15884)

    AnyBox是一个高效的零样本框架，通过利用箱体的几何规则性，在单张RGB-D观测上交替进行位姿与尺度估计，从而实现对杂乱遮挡环境中箱体9自由度位姿（6D位姿+3D尺寸）的联合恢复，无需物体特定的CAD模型。

    

    在杂乱和遮挡环境下恢复物体的9D位姿（包括其6D位姿和3D尺寸）是仓储自动化、物流和制造领域的核心需求。基于模型的方法精度较高，但假设每个物体都拥有特定的CAD模型，随着库存变化，维护成本十分高昂。无模型和类别级方法放宽了这一假设，但它们对于堆叠储物箱所特有的对称性、弱纹理和严重遮挡仍然脆弱，且忽略了此类场景所提供的强结构先验。我们提出了AnyBox，这是一个高效的零样本框架，它利用箱体的几何规则性，从单次RGB-D观测中联合恢复位姿和尺寸。从一个规范化的类别模板出发，AnyBox在位姿估计与尺度估计之间交替进行，利用重投影模板与观测掩码之间的差异来驱动二……

    arXiv:2511.15884v2 Announce Type: replace-cross  Abstract: Recovering the 9D pose of objects, both their 6D pose and 3D dimensions, under clutter and occlusion is a core requirement for warehouse automation, logistics, and manufacturing. Model-based methods are accurate but assume an instance-specific CAD model for every object, which is costly to maintain as inventories change. Model-free and category-level methods relax this assumption, yet they remain vulnerable to the symmetry, weak texture, and heavy occlusion that characterize stacked storage boxes, and they ignore the strong structural priors such scenes provide. We present \textbf{AnyBox}, an efficient zero-shot framework that exploits the geometric regularity of boxes to jointly recover pose and dimensions from a single RGB-D observation. Starting from a canonical category template, AnyBox alternates between pose and scale estimation, using the discrepancy between the reprojected template and the observed mask to drive a binar
    
[^256]: 一种Nesterov加速的拜占庭鲁棒联邦学习

    A Nesterov-Accelerated Byzantine-Robust Federated Learning

    [https://arxiv.org/abs/2511.02657](https://arxiv.org/abs/2511.02657)

    该论文提出了一种将Nesterov动量与拜占庭鲁棒聚合规则无缝结合的联邦学习算法Byrd-NAFL，在非凸光滑损失函数和宽松假设下建立了有限时间收敛保证，同时兼顾通信效率与对拜占庭攻击的抵抗能力。

    

    我们研究了鲁棒联邦学习问题，即一组工作节点在中央服务器的协调下协作训练一个共享模型，同时存在能够进行任意且潜在恶意行为的拜占庭攻击者。为了同时提高通信效率和对这类攻击者的抵御能力，我们提出了一种拜占庭鲁棒的Nesterov加速联邦学习算法。Byrd-NAFL将Nesterov动量与拜占庭鲁棒聚合规则无缝集成到联邦学习过程中，以实现对梯度污染的快速且安全的收敛。我们在非凸光滑损失函数下，并采用对聚合梯度较为宽松的假设，为Byrd-NAFL建立了有限时间收敛保证。大量数值实验验证了Byrd-NAFL的有效性，并证明了其在收敛性能方面相对于现有基准方法的优越性。

    arXiv:2511.02657v2 Announce Type: replace  Abstract: We investigate robust federated learning, where a group of workers collaboratively train a shared model under the orchestration of a central server in the presence of Byzantine adversaries capable of arbitrary and potentially malicious behaviors. To simultaneously enhance communication efficiency and resilience against such adversaries, we propose a Byzantine-resilient Nesterov-accelerated federated learning (Byrd-NAFL) algorithm. Byrd-NAFL seamlessly integrates Nesterov's momentum into the federated learning process alongside Byzantine-resilient aggregation rules to achieve fast and safe convergence against gradient corruption. We establish a finite-time convergence guarantee for Byrd-NAFL under non-convex and smooth loss functions with relaxed assumptions on the aggregated gradients. Extensive numerical experiments validate the effectiveness of Byrd-NAFL and demonstrate the superiority over existing benchmarks in terms of convergen
    
[^257]: 从泄漏到保真：时序级联预测的可靠基准测试

    From Leakage to Fidelity: Reliable Benchmarking for Temporal Cascade Prediction

    [https://arxiv.org/abs/2510.25348](https://arxiv.org/abs/2510.25348)

    该论文揭示时序级联预测现有评估中普遍存在数据泄漏问题，提出以完全时序协议为核心的评估标准与泄漏诊断方法，并发布包含丰富特征与购买转化标签的真实电商级联数据集 Taoke，推动该领域迈向保真感知的可靠基准测试。

    

    时序级联预测被广泛研究，但其经验基础仍然脆弱。现有大多数工作在将过去与未来信号相混合的随机级联划分下报告结果，依赖于特征有限且缺少下游转化标签的数据集，并在未系统检验基准结论是否依赖于评估协议的情况下比较日益复杂的模型。本文主张该领域应从易泄漏的评估转向保真感知的基准测试。我们为时序级联预测引入了一套协议套件和更新的评估标准，核心包括完全时序协议、基于重叠的泄漏诊断，以及对性能膨胀与时间漂移的分析。为拓展基准任务的范围，我们还提出了 Taoke——一个真实的电商级联数据集，包含丰富的推广者/商品特征以及实际观测到的购买转化，可用于一阶段流行度……（原文摘要在此处截断）

    arXiv:2510.25348v3 Announce Type: replace  Abstract: Temporal cascade prediction is widely studied, yet its empirical foundations remain fragile. Most existing works report results under random cascade splits that mix past and future signals, rely on datasets with limited features and no downstream conversion labels, and compare increasingly complex models without systematically examining whether benchmark conclusions are protocol-dependent. This paper argues that the field should move from leakage-prone evaluation toward fidelity-aware benchmarking. We introduce a protocol suite and renewed evaluation standard for temporal cascade prediction, centered on the Full Temporal protocol, overlap-based leakage diagnostics, and analyses of performance inflation and temporal drift. To broaden the scope of benchmark tasks, we also present Taoke, a real-world e-commerce cascade dataset with rich promoter/product features and observed purchase conversions, enabling both first-stage popularity for
    
[^258]: WELD：首个面向泛在情感计算的自然情境长周期小团队职场情绪数据集

    WELD: The First Naturalistic Long-Period Small-Team Workplace Emotion Dataset for Ubiquitous Affective Computing

    [https://arxiv.org/abs/2510.15221](https://arxiv.org/abs/2510.15221)

    WELD是首个结合数年持续时间、自然职场情境、稳定小团队结构与完全被动感知协议的职场情绪数据集，基于中国某软件公司49名员工超过30个月的面部表情数据构建。

    

    情感计算在实验室环境中已快速成熟，然而此前没有任何数据集能同时满足以下四点：(i) 数月至数年的持续时间，(ii) 自然的职场情境，(iii) 稳定的小团队社会结构，以及(iv) 能通过机构伦理审查的完全被动感知协议。我们推出了WELD——首个同时满足这四点的数据集。WELD包含来自中国某软件公司49名员工在30.1个月（2021年11月至2024年5月）期间采集的733,780个逐帧七类面部表情概率向量——这是最长的自然真实环境情绪语料库，也是唯一一个支持对同一批被试同时开展个体内纵向分析和团队内关系分析的多年度语料库。数据采用四级访问模型发布，仅有聚合概率可供公开下载。我们通过复现三个已确立的现象验证了该语料库（周末效价提升43.1%；13:00低谷的昼夜节律周期；上海……）

    arXiv:2510.15221v3 Announce Type: replace  Abstract: Affective computing has matured rapidly in laboratory settings, yet no prior dataset combines (i) months-to-years of duration, (ii) a naturalistic workplace context, (iii) a stable small-team social structure, and (iv) a fully passive sensing protocol that survives institutional review. We introduce WELD, the first dataset to satisfy all four. WELD comprises 733,780 per-frame seven-class facial-expression probability vectors from 49 employees of a Chinese software company over 30.1 months (Nov 2021 - May 2024) -- the longest naturalistic in-the-wild emotion corpus and the only multi-year corpus supporting both within-individual longitudinal and within-team relational analyses on the same subjects. Data are released under a four-tier access model with only aggregated probabilities publicly downloadable. We validate the corpus by replicating three established phenomena (+43.1% weekend valence boost; 13:00-trough diurnal cycle; Shanghai
    
[^259]: 基于线性函数逼近的单轨迹卡方（χ²）鲁棒Q学习的有限时间收敛性

    Finite-Time Convergence of Single-Trajectory Chi-Square Robust Q-Learning With Linear Function Approximation

    [https://arxiv.org/abs/2510.01721](https://arxiv.org/abs/2510.01721)

    本文针对使用χ²不确定集和线性函数逼近的单轨迹鲁棒Q学习，通过变分重形式化和分块冻结目标方案，克服了条件二阶矩平方根无法无偏估计以及投影算子非压缩的难题，首次对所有折扣因子建立了相对于最优鲁棒Q函数的有限时间误差界。

    

    分布鲁棒强化学习旨在寻找当部署环境与生成训练数据的环境不同时仍然有效的策略。我们研究了采用χ²不确定集和线性函数逼近的无模型鲁棒Q学习，其数据来自未知标称马尔可夫决策过程（MDP）的单条轨迹。在评估χ²鲁棒Bellman目标时引入了条件二阶矩的平方根，该量无法从单次转移中得到无偏估计，同时投影鲁棒Bellman算子也不一定是压缩的。我们通过鲁棒Bellman目标的变分重形式化和分块冻结目标方案来克服这些障碍，并对所有γ∈(0,1)建立了相对于最优鲁棒Q函数的有限时间误差界。一个神经网络实验展示了变分目标如何应用于连续状态的非线性控制任务中。

    arXiv:2510.01721v4 Announce Type: replace  Abstract: Distributionally robust reinforcement learning seeks policies that remain effective when the deployment environment differs from the one that generated the training data. We study model-free robust Q-learning with $\chi^2$ uncertainty sets and linear function approximation, using data from a single trajectory of an unknown nominal MDP. Evaluating the $\chi^2$ robust Bellman target introduces the square root of a conditional second moment, which cannot be estimated unbiasedly from one transition, while the projected robust Bellman operator need not be contractive. We address these obstacles through a variational reformulation of the robust Bellman target and a blockwise frozen-target scheme, and establish a finite-time error bound relative to the optimal robust Q-function for every $\gamma\in(0,1)$. A neural-network experiment illustrates how the variational target can be used in a continuous-state nonlinear-control task.
    
[^260]: Zonotope包含问题与神经网络验证的参数化难度

    Parameterized Hardness of Zonotope Containment and Neural Network Verification

    [https://arxiv.org/abs/2509.22849](https://arxiv.org/abs/2509.22849)

    本工作解决了Froese等人提出的神经网络验证参数化复杂性开放问题，证明了对所有ℓ≥2，以输入维度为参数时判定ℓ层ReLU网络计算函数的正性（及满射性）是W[ℓ-1]-难的，并由此推出Zonotope非包含问题是W[1]-难的。

    

    带ReLU激活函数的神经网络是机器学习中广泛使用的模型，因此深入理解这类网络所计算函数的性质十分重要。近年来，判定这些性质的（参数化）计算复杂性受到了越来越多的关注。在本工作中，我们填补了若干空白，并解决了Froese等人[COLT '25]提出的关于网络验证相关各类问题参数化复杂性的一个开放问题。特别地，我们证明：对于所有ℓ≥2，以输入维度d为参数时，判定由ℓ层ReLU网络计算的函数f:R^d→R的正性（从而满射性）是W[ℓ-1]-难的。其中ℓ=2的情形意味着Zonotope非包含问题（该问题在计算几何、控制理论和机器人学中具有独立的研究价值）是W[1]-难的。

    arXiv:2509.22849v3 Announce Type: replace-cross  Abstract: Neural networks with ReLU activations are a widely used model in machine learning. It is thus important to have a profound understanding of the properties of the functions computed by such networks. Recently, there has been increasing interest in the (parameterized) computational complexity of determining these properties. In this work, we close several gaps and resolve an open problem posed by Froese et al. [COLT '25] regarding the parameterized complexity of various problems related to network verification. In particular, we prove that, for all $\ell\ge 2$, deciding positivity (and thus surjectivity) of a function $f:\mathbb{R}^d\to\mathbb{R}$ computed by an $\ell$-layer ReLU network is W[$\ell-1$]-hard when parameterized by the input dimension $d$. The case $\ell=2$ implies that zonotope non-containment (a problem that is of independent interest in computational geometry, control theory, and robotics) is W[1]-hard with respe
    
[^261]: 用于学习哈密顿系统的高效数据核方法

    Data-efficient Kernel Methods for Learning Hamiltonian Systems

    [https://arxiv.org/abs/2509.17154](https://arxiv.org/abs/2509.17154)

    该论文提出从轨迹数据直接学习哈密顿系统的核方法（含一步法和两步法），在数据稀缺场景下实现高效精确预测并保持哈密顿结构，同时提供了先验误差估计以保证模型的可靠性。

    

    哈密顿动力学描述了广泛的物理系统，因此哈密顿系统的数据驱动模拟对许多科学和工程问题具有重要意义。在这项工作中，我们提出了基于核的方法，可直接从轨迹数据中识别和预测哈密顿系统。我们提出了两种方法：一种是两步法，先重建轨迹再学习哈密顿函数；另一种是一步法，将两者联合推断。在多个基准系统（包括质量-弹簧动力学、非线性单摆和Henon-Heiles系统）上，我们证明该框架能够实现精确且数据高效的预测，尤其在数据稀缺的情况下优于两步核方法基线，同时保持了哈密顿结构。此外，我们证明了先验误差估计，确保了学习模型的可靠性。我们还提供了一个更通用的、与具体问题无关的数值框架。

    arXiv:2509.17154v2 Announce Type: replace-cross  Abstract: Hamiltonian dynamics describe a wide range of physical systems. As such, data-driven simulations of Hamiltonian systems are important for many scientific and engineering problems. In this work, we propose kernel-based methods for identifying and forecasting Hamiltonian systems directly from trajectory data. We present two approaches: a 2-step method that reconstructs trajectories before learning the Hamiltonian, and a 1-step method that jointly infers both. Across several benchmark systems, including mass-spring dynamics, a nonlinear pendulum, and the Henon-Heiles system, we demonstrate that our framework achieves accurate, data-efficient predictions and outperforms 2-step kernel-based baselines, particularly in scarce-data regimes, while preserving the Hamiltonian structure. Moreover, we prove a priori error estimates, ensuring reliability of the learned models. We also provide a more general, problem-agnostic numerical framew
    
[^262]: 大语言模型时代的医学推理：增强技术与应用的系统综述

    Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications

    [https://arxiv.org/abs/2508.00669](https://arxiv.org/abs/2508.00669)

    本文是首个针对大语言模型医学推理领域的系统综述，提出了涵盖训练时策略与测试时机制的推理增强技术分类体系，并系统分析了这些技术在多种数据模态和临床应用中的实践与评估方法。

    

    大语言模型在医学领域的蓬勃发展带来了令人印象深刻的能力，但其在执行系统性、透明性和可验证性推理方面仍存在关键差距，而这正是临床实践的基石。这一差距推动了从单步答案生成向专为医学推理设计的大语言模型发展的范式转变。本文对该新兴领域进行了首次系统综述。我们提出了一个推理增强技术的分类体系，将其分为训练时策略（如监督微调、强化学习）和测试时机制（如提示工程、多智能体系统）。我们分析了这些技术如何应用于不同的数据模态（文本、图像、代码）以及关键的医疗场景中，如诊断、教育和治疗规划。此外，我们还梳理了评估基准从简单准确率指标……（原文摘要到此截断）

    arXiv:2508.00669v2 Announce Type: replace-cross  Abstract: The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy
    
[^263]: ScoreMix：通过扩散模型中的分数组合进行合成数据生成以提升识别性能

    ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition

    [https://arxiv.org/abs/2506.10226](https://arxiv.org/abs/2506.10226)

    提出ScoreMix方法，利用扩散模型的分数可组合性、在无需外部模型或数据集的情况下，通过混合判别器嵌入空间中相距较远的类别生成分数条件合成样本，为识别任务带来最高3%的平均性能提升。

    

    合成数据生成在机器学习中被越来越多地用于模型训练和数据增强。然而，现有策略通常依赖于外部基础模型或数据集，而由于政策或法律的限制，这些资源在许多场景下无法使用。我们提出了ScoreMix，这是一种自包含的合成数据生成方法，通过利用扩散模型的分数可组合性，为识别任务生成困难的合成样本。该方法沿反向扩散轨迹混合类条件分数，在无需外部资源的情况下实现特定领域的数据增强。我们系统地研究了类别选择策略，发现混合判别器嵌入空间中相距较远的类别能够带来更大的收益，与基于接近度的类别选择方式相比，平均可获得高达3%的额外提升。有趣的是，我们观察到在标准（条件下）……

    arXiv:2506.10226v3 Announce Type: replace-cross  Abstract: Synthetic data generation is increasingly used in machine learning for training and data augmentation. Yet, current strategies often rely on external foundation models or datasets, whose usage is restricted in many scenarios due to policy or legal constraints. We propose ScoreMix, a self-contained synthetic generation method to produce hard synthetic samples for recognition tasks by leveraging the score compositionality of diffusion models. The approach mixes class-conditioned scores along reverse diffusion trajectories, yielding domain-specific data augmentation without external resources. We systematically study class-selection strategies and find that mixing classes distant in the discriminator's embedding space yields larger gains, providing up to 3% additional average improvement, compared to selection based on proximity. Interestingly, we observe that condition and embedding spaces are largely uncorrelated under standard 
    
[^264]: 带函数逼近的马尔可夫决策过程的自适应重解方法

    Adaptive Resolving Methods for Markov Decision Processes with Function Approximations

    [https://arxiv.org/abs/2505.12037](https://arxiv.org/abs/2505.12037)

    本文提出一种基于线性规划重构的自适应重解算法，在新转移样本到达时反复重解约简线性系统，从而在带函数逼近的MDP中实现了与实例相关的 $\widetilde O(C_{\mathrm{inst}}/N)$ 目标缺口与约束残差的高效求解。

    

    从样本中学习马尔可夫决策过程（MDP）问题的最优策略是在线与数据驱动决策中的一个基本问题。函数逼近通常被用于处理大规模或无限的状态-动作空间。在本工作中，我们考虑带函数逼近的MDP问题，并开发了一种高效求解它的新算法。我们的算法基于线性规划（LP）重构，并在新的转移样本到达时反复重解已识别的约简线性系统。在最优基被识别之后，我们证明，经过 $N$ 轮重解后，期望平均迭代解可以达到与实例相关的 $\widetilde O(C_{\mathrm{inst}}/N)$ 的目标函数缺口和带符号约束残差。我们分别统计了用于基识别的历史样本和每轮重解中使用的 $d_2$ 次转移查询，从而得到相应的总转移样本复杂度。

    arXiv:2505.12037v2 Announce Type: replace  Abstract: Learning the optimal policy for Markov decision process problems (MDPs) from samples is a fundamental problem in online and data-driven decision-making. Function approximations are usually deployed to handle large or infinite state-action space. In our work, we consider the MDP problems with function approximation and we develop a new algorithm to solve it efficiently. Our algorithm is based on a linear programming (LP) reformulation and repeatedly resolves the identified reduced linear system as new transition samples arrive. After the optimal basis is identified, we show that, after $N$ resolving rounds, the expected averaged iterate achieves an instance-dependent $\widetilde O(C_{\mathrm{inst}}/N)$ objective shortfall and signed constraint residual. We separately account for the historical samples used for basis identification and the $d_2$ transition queries used in each resolving round, which yields the corresponding total trans
    
[^265]: 干预驱动偏移下反事实回归的半参数推断

    Semiparametric Inference for Counterfactual Regression under Intervention-Driven Shift

    [https://arxiv.org/abs/2504.02694](https://arxiv.org/abs/2504.02694)

    提出了一个沿增量干预路径进行反事实回归的半参数推断框架，通过交叉拟合影响函数方法建立了优化器的一致性、稳定性和渐近有效推断，并能构建同步置信带。

    

    我们研究反事实回归问题，即将特征映射到与数据中观测情形不同的假设情景下的结果。这一问题对于分布偏移下的决策至关重要，因为处理模式在部署时可能发生变化。我们开发了一个沿预先指定的增量干预路径进行反事实回归的半参数框架。其目标是反事实风险的有限维约束投影，通过程序组件的交叉拟合影响函数表示进行估计。对于具有固定约束的平滑程序以及具有估计线性约束的有限维程序，我们在特定函数类条件下建立了优化器的一致性和局部稳定性，并推导了逐点和一致的一阶展开。这些结果带来了渐近有效的统计推断，包括反事实回归的同步置信带。

    arXiv:2504.02694v3 Announce Type: replace-cross  Abstract: We study counterfactual regression, which maps features to outcomes under hypothetical scenarios that differ from those observed in the data. This problem is central to decision-making under distribution shift, where treatment patterns may change at deployment. We develop a semiparametric framework for counterfactual regression along a prespecified incremental-intervention path. The target is a finite-dimensional constrained projection of counterfactual risk, estimated using cross-fitted influence-function representations of the program components. For smooth programs with fixed constraints and finite-dimensional programs with estimated linear constraints, we establish consistency and local stability of the optimizer under class-specific conditions, and derive pointwise and uniform first-order expansions. These results yield asymptotically valid inference, including simultaneous confidence bands for the counterfactual regressio
    
[^266]: 基于学习约束的自适应超图神经网络求解车辆路径问题

    Learning Constraints-Based Adaptive Hypergraph Neural Networks for Solving Vehicle Routing Problems

    [https://arxiv.org/abs/2503.10421](https://arxiv.org/abs/2503.10421)

    提出一种将面向约束的超图与强化学习相结合的端到端框架，通过动态超边重构策略有效处理车辆路径问题中的复杂硬约束。

    

    将基于学习的方法应用于车辆路径问题已成为组合优化领域的一个关键研究方向。这类问题的特点是解空间庞大且约束复杂，使得精确数学模型或启发式方法等传统方法往往面临较高的计算开销，或需要依赖复杂启发式算子的设计才能获得最优或接近最优的解。与此同时，尽管近期一些基于学习的方法能够在约束场景较为简单的车辆路径问题上取得良好性能，但它们通常难以有效处理实践中常见的硬约束。本研究提出了一种新颖的端到端框架，将面向约束的超图与强化学习相结合来解决车辆路径问题。本工作的核心创新在于开发了一种面向约束的动态超边重构策略（摘要内容在此处截断）。

    arXiv:2503.10421v2 Announce Type: replace  Abstract: The application of learning based methods to vehicle routing problems has emerged as a pivotal area of research in combinatorial optimization. These problems are characterized by vast solution spaces and intricate constraints, making traditional approaches such as exact mathematical models or heuristic methods prone to high computational overhead or reliant on the design of complex heuristic operators to achieve optimal or near optimal solutions. Meanwhile, although some recent learning-based methods can produce good performance for VRP with straightforward constraint scenarios, they often fail to effectively handle hard constraints that are common in practice. This study introduces a novel end-to-end framework that combines constraint-oriented hypergraphs with reinforcement learning to address vehicle routing problems. A central innovation of this work is the development of a constraint-oriented dynamic hyperedge reconstruction stra
    
[^267]: LLM即GNN：面向文本属性图基础模型的图词表学习

    LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models

    [https://arxiv.org/abs/2503.03313](https://arxiv.org/abs/2503.03313)

    该论文提出PromptGFM，通过图词表学习将图节点融入语言模型词表空间，克服了现有LLM与GNN解耦架构及OOV token带来的跨图、跨任务迁移难题，构建了面向文本属性图的多功能图基础模型。

    

    文本属性图（TAGs）在现实场景中无处不在，其中每个节点都与文本描述相关联。它们通常表现出独特的结构和领域特定知识，这促使人们开发能够在多样化图和任务上进行泛化的图基础模型（GFM）。尽管人们已在为TAGs集成大型语言模型（LLMs）和图神经网络（GNNs）方面做出了大量努力，但现有方法存在解耦架构和两阶段对齐的问题，限制了它们的协同潜力。更糟糕的是，现有方法为图节点分配词表外（OOV）token，导致图特定语义、token爆炸以及与面向任务的提示模板不兼容，这阻碍了跨图和跨任务的可迁移性。为了解决这些挑战，我们提出了PromptGFM，这是一个基于图词表学习的、适用于TAGs的多功能图基础模型。PromptGFM包含两个关键组件。

    arXiv:2503.03313v4 Announce Type: replace-cross  Abstract: Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key compone
    
[^268]: AgentRM：通过奖励建模增强智能体的泛化能力

    AgentRM: Enhancing Agent Generalization with Reward Modeling

    [https://arxiv.org/abs/2502.18407](https://arxiv.org/abs/2502.18407)

    本论文提出可泛化奖励模型AgentRM，发现微调奖励模型来引导测试时搜索比直接微调策略模型更稳健，在九个智能体任务上平均提升8.8分并超越最强通用智能体4.0分。

    

    现有的基于大语言模型（LLM）的智能体在已见任务上取得了强大的性能，但它们对未见任务的泛化能力仍然较差。因此，近期的一些工作专注于使用更多样化的任务对策略模型进行微调以提升泛化能力。在这项工作中，我们发现微调一个奖励模型来引导策略模型，比直接微调策略模型更加稳健。基于这一发现，我们提出了AgentRM，一个可泛化的奖励模型，用于引导策略模型进行有效的测试时搜索。我们全面研究了构建奖励模型的三种方法，包括显式奖励建模、隐式奖励建模以及LLM作为评判者（LLM-as-a-judge）。随后，我们使用AgentRM通过Best-of-N采样和步级束搜索来引导答案生成。在四类共九个智能体任务上，AgentRM使基础策略模型平均提升了8.8分，超越了最顶尖的通用智能体4.0分。此外，它……

    arXiv:2502.18407v2 Announce Type: replace-cross  Abstract: Existing LLM-based agents have achieved strong performance on held-in tasks, but their generalizability to unseen tasks remains poor. Hence, some recent work focus on fine-tuning the policy model with more diverse tasks to improve the generalizability. In this work, we find that finetuning a reward model to guide the policy model is more robust than directly finetuning the policy model. Based on this finding, we propose AgentRM, a generalizable reward model, to guide the policy model for effective test-time search. We comprehensively investigate three approaches to construct the reward model, including explicit reward modeling, implicit reward modeling and LLM-as-a-judge. We then use AgentRM to guide the answer generation with Best-of-N sampling and step-level beam search. On four types of nine agent tasks, AgentRM enhances the base policy model by $8.8$ points on average, surpassing the top general agent by $4.0$. Moreover, it
    
[^269]: 关于协作式AI在真实世界医学应用中成本效益的警示故事

    A cautionary tale on the cost-effectiveness of collaborative AI in real-world medical applications

    [https://arxiv.org/abs/2412.06494](https://arxiv.org/abs/2412.06494)

    本文通过对7个医学数据集、3种机器学习任务和8种数据模态的大规模基准测试，系统比较了联邦学习与基于共识的学习方法在多中心医学数据分析中的准确性和成本效益，为协作式AI在真实医疗场景中的部署提供了警示性见解。

    

    背景：联邦学习（FL）作为一种协作学习范式已获得广泛关注，能够在敏感的医疗应用中实现协作式人工智能。然而，联邦学习的实际实施面临技术和组织方面的挑战，因为它通常需要复杂的通信基础设施。在这种背景下，基于共识的学习（CBL）可能是一种有前景的协作学习替代方案，它能够将本地知识整合到联邦决策系统中，同时可能降低部署开销。方法：在这项工作中，我们对一系列FL和CBL方法在广泛的协作式医学数据分析场景中的准确性和成本效益进行了大规模基准测试。该基准测试包括7个不同的医学数据集，涵盖3种机器学习任务、8种不同的数据模态，以及涉及3到23个客户端的多中心设置。发现：（原文摘要至此截断）

    arXiv:2412.06494v2 Announce Type: replace  Abstract: Background. Federated learning (FL) has gained wide popularity as a collaborative learning paradigm enabling collaborative AI in sensitive healthcare applications. Nevertheless, the practical implementation of FL presents technical and organizational challenges, as it generally requires complex communication infrastructures. In this context, consensus-based learning (CBL) may represent a promising collaborative learning alternative, thanks to the ability of combining local knowledge into a federated decision system, while potentially reducing deployment overhead. Methods. In this work we propose an extensive benchmark of the accuracy and cost-effectiveness of a panel of FL and CBL methods in a wide range of collaborative medical data analysis scenarios. The benchmark includes 7 different medical datasets, encompassing 3 machine learning tasks, 8 different data modalities, and multi-centric settings involving 3 to 23 clients. Findings
    
[^270]: 语法对齐解码

    Grammar-Aligned Decoding

    [https://arxiv.org/abs/2405.21047](https://arxiv.org/abs/2405.21047)

    本文揭示了语法约束解码会扭曲大语言模型的输出分布，导致生成结果虽符合语法但质量低下，并提出了一种名为ASAp的语法对齐解码算法来解决这一问题。

    

    大语言模型（LLMs）难以可靠地生成高度结构化的输出，例如程序代码、数学公式或格式良好的标记语言。约束解码方法通过在每个步骤贪心地限制LLM可以输出的token来缓解这一问题，从而保证输出符合给定的约束。具体而言，在语法约束解码（GCD）中，LLM的输出必须遵循给定的语法。在本文中，我们证明了GCD技术（以及广义上的约束解码技术）会扭曲LLM的分布，导致输出虽然符合语法，但其出现的概率与LLM本身给出的概率不成比例，因此最终质量较低。我们将采样与语法约束对齐这一问题称为语法对齐解码（GAD），并提出了一种名为ASAp（基于近似期望未来的自适应采样）的解码算法，该算法保证输出……

    arXiv:2405.21047v4 Announce Type: replace  Abstract: Large Language Models (LLMs) struggle with reliably generating highly structured outputs, such as program code, mathematical formulas, or well-formed markup. Constrained decoding approaches mitigate this problem by greedily restricting what tokens an LLM can output at each step to guarantee that the output matches a given constraint. Specifically, in grammar-constrained decoding (GCD), the LLM's output must follow a given grammar. In this paper, we demonstrate that GCD techniques (and in general constrained decoding techniques) can distort the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM, and so ultimately are low-quality. We call the problem of aligning sampling with a grammar constraint, grammar-aligned decoding (GAD), and propose adaptive sampling with approximate expected futures (ASAp), a decoding algorithm that guarantees the outpu
    
[^271]: 生物信号应用中机器学习的不确定性量化——综述

    Uncertainty Quantification in Machine Learning for Biosignal Applications -- A Review

    [https://arxiv.org/abs/2312.09454](https://arxiv.org/abs/2312.09454)

    本文系统综述了不确定性量化在脑电图、心电图、眼电图、肌电图等生物信号机器学习应用中的现有方法、应用场景、评估手段与不确定性度量，旨在提升医学场景下预测的可解释性与鲁棒性。

    

    目的：不确定性量化（UQ）旨在提高机器学习预测的可解释性和鲁棒性，因此日益受到关注。具体而言，脑电图（EEG）、心电图（ECG）、眼电图（EOG）和肌电图（EMG）等（医学）生物信号可以从良好的不确定性量化中受益，因为这些信号的信噪比较低，而良好的人类可解释性对医学应用至关重要。为了确定如何将不确定性估计应用于生物信号任务，我们研究了当前的方法、用例、应用、评估方式以及不确定性度量。方法：本文系统性地综述了在生物信号领域将不确定性量化应用于机器学习任务的最新研究进展。我们收集了来自Web of Science、Scopus、IEEE XPlore和PsycINFO中所有讨论上述生物信号之一上机器学习不确定性的相关文献。

    arXiv:2312.09454v3 Announce Type: replace-cross  Abstract: Purpose: Uncertainty Quantification (UQ) has gained traction in an attempt to improve the interpretability and robustness of machine learning predictions. Specifically (medical) biosignals such as electroencephalography (EEG), electrocardiography (ECG), electrooculography (EOG), and electromyography (EMG) could benefit from good UQ, since these suffer from a poor signal-to-noise ratio, and good human interpretability is pivotal for medical applications. To determine how uncertainty estimation can be used for biosignal tasks, we investigate current methods, use cases, applications, evaluations, and uncertainty measures.   Methods: In this paper, we systematically review the state of the art of applying Uncertainty Quantification to Machine Learning tasks in the biosignal domain. All works from Web of Science, Scopus, IEEE XPlore and PsycINFO that discuss uncertainty in Machine Learning on one of the aforementioned biosignals is 
    
[^272]: 面向高维数据的各向异性视图距离度量：理论、几何与快速计算

    Anisotropic View Distance Metric for High-Dimensional Data: Theory, Geometry, and Fast Computation

    [https://arxiv.org/abs/2206.05215](https://arxiv.org/abs/2206.05215)

    提出了一种受正交投影启发的新距离度量——视图距离，通过将样本空间投影到n(n-1)/2个二维平面并求欧氏距离之和，严格满足度量公理，有效解决了高维数据中各向异性结构和冗余特征导致欧氏距离失效的问题。

    

    arXiv:2206.05215v2 公告类型：替换 摘要：K-Means聚类算法因其简单高效而成为最常用的聚类算法之一。基于欧氏距离的K-Means聚类算法仅关注线性距离。欧氏距离是一种高效且可解释的相似性度量，但其在具有各向异性结构、冗余特征或复杂特征交互的样本空间中的有效性可能会下降。在本文中，我们提出了一种新颖的距离度量，称为视图距离。受正交投影的启发，该度量将样本空间投影到n(n-1)/2个二维平面上，并将最终距离定义为各投影平面上欧氏距离之和。理论推导证明视图距离严格满足度量公理和范数约束。此外，视图距离通过投影实现了特征耦合，不仅能够……

    arXiv:2206.05215v2 Announce Type: replace  Abstract: K-Means clustering algorithm is one of the most commonly used clustering algorithms because of its simplicity and efficiency. K-Means clustering algorithm based on Euclidean distance only pays attention to the linear distance between Euclidean distance is an efficient and interpretable similarity measurement, but its effectiveness may deteriorate in sample spaces with anisotropic structures, redundant features, or complex feature interactions. In this paper, we propose a novel distance metric called View distance. Inspired by orthographic projection, the proposed metric projects the sample space onto $n(n-1)/2$ two-dimensional planes and defines the final distance as the sum of the Euclidean distances across the projected planes. Theoretical derivations verify that the View distance strictly satisfies the metric axioms and norm constraints. Beyond that, the View distance achieves feature coupling through projection and not only enabl
    

