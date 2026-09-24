# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Diffusibility of High-Dimensional Latents](https://arxiv.org/abs/2609.28473) | 本文揭示了在高维隐空间中使用流匹配标准速度预测会导致模型需拟合信号流形之外的正交噪声方向、优化低效的问题，并提出改用干净数据参数化（$x_0$-预测）使学习聚焦于低维信号流形，从而提升扩散模型的生成效率。 |
| [^2] | [Contrastive Learning for Authorship Verification](https://arxiv.org/abs/2609.28471) | 本文提出基于 ModernBERT 双编码器的对比学习方法，通过优化损失函数、数据增强等关键因素，在 PAN21 作者身份验证任务上达到 98.4% 的准确率，优于基于分类的方法。 |
| [^3] | [Even Sharper Bounds for Transductive Learning and Its Applications](https://arxiv.org/abs/2609.28459) | 本文提出直推学习的新型局部化复杂度方法STLC，去除了以往直推学习中多余的对数置信度因子，并在可实现设定下达到了与标准归纳学习相同的 $\cO\{\dVC\log(me/\dVC)/m\}$ 速率。 |
| [^4] | [Nonequilibrium Phases of Repulsive Self-Attention: Chaos, Attention Condensation, and Emergent Locality](https://arxiv.org/abs/2609.28448) | 该论文通过分析具有负值映射的最小循环transformer模型，揭示了排斥性自注意力系统中丰富的非平衡相行为，包括通过倍周期分岔产生的混沌动力学、在 $\beta\sim N^2$ 标度下出现的注意力凝聚现象以及硬路由极限下的涌现局域性。 |
| [^5] | [Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning](https://arxiv.org/abs/2609.28442) | 该研究发现，语言模型对规则排序的内部表征越清晰（排列信噪比越高），其解决重排序数学问题的准确率就越高，揭示了答案不变性与表征不变性是两个不同的概念。 |
| [^6] | [Minimal-Norm Univariate Two-Layer ReLU Classification: Exact Solutions and Global Optimality with Skip Connections](https://arxiv.org/abs/2609.28438) | 该论文完整刻画了单变量两层ReLU网络二元分类最优解在函数空间中的几何结构，并证明添加仿射跳跃连接后每个KKT点都成为全局最优点。 |
| [^7] | [Context-Continuous Preference Learning for Exoskeleton Personalization](https://arxiv.org/abs/2609.28427) | 本文提出情境连续偏好学习方法（CCPL），利用高斯过程在相邻运行条件之间共享偏好观测数据，从而以更少的用户反馈实现外骨骼辅助的个性化优化。 |
| [^8] | [Repairability of Inexact Solvers in Recursive State Estimation with Machine Learning](https://arxiv.org/abs/2609.28425) | 本文刻画了递归状态估计（卡尔曼滤波）中不精确数值求解缺陷在给定子空间与范数预算下的可修复条件，并通过残差—漂移恒等式与六阶余项界揭示了新息协方差膨胀与局部增益重优化对有限时域协方差响应的相反作用。 |
| [^9] | [Agent-Editing World Model: Rethinking World Modeling for LLM Agents](https://arxiv.org/abs/2609.28416) | 提出“智能体编辑世界模型”（AEWM），不再模拟工具响应，而是通过动作判官与状态修订来建模推理和动作如何影响未来任务进展，从而避免任务状态污染、提升智能体长时程任务表现。 |
| [^10] | [Learning Holographic Reduced Representations with Clifford Variational Autoencoders](https://arxiv.org/abs/2609.28409) | 提出了一种名为Clifford-VAE的变分自编码器，通过将数据投影到任意维度的克利福德环面上，为将感知数据嵌入向量符号代数框架提供了原理性方法，并在半监督分类任务和多项VSA基准测试中达到或超越了高斯和超球面VAE的性能。 |
| [^11] | [Learning Collective Dynamics with Differentiable Gaussian Representations](https://arxiv.org/abs/2609.28405) | 本文提出可微高斯动力学（DGD），通过高斯混合表示、可微聚合和反馈循环三个组件，直接从总体计数数据中端到端学习集体响应动力学，并在真实数据集上取得优于DeepAR改进版本的联合行为预测性能。 |
| [^12] | [Memory Attention](https://arxiv.org/abs/2609.28399) | 提出记忆注意力，用可复用的词元记忆与上下文键结合构建注意力值，推理时可简化为查表加法并支持CPU卸载，在相同训练预算下提升了语言建模和下游任务性能。 |
| [^13] | [Fine-Tuning LLMs for Translation: General Forgetting Mitigation Does Not Preserve MT-Specific Instruction Following](https://arxiv.org/abs/2609.28395) | 弹性权重巩固等遗忘缓解方法虽能有效保持微调后大语言模型的通用能力，却无法保留机器翻译特定的指令遵循能力（如语体正式度、语法性别和长度控制），表明现有评估方式与实际翻译应用需求存在脱节。 |
| [^14] | [Quantum score matching with applications to learning thermal states](https://arxiv.org/abs/2609.28391) | 本文建立了具有端到端理论保证的通用量子分数匹配框架，应用于吉布斯态学习时无需额外制备热态，并在高温区间对有界局域哈密顿量实现了信息论最优的样本复杂度。 |
| [^15] | [When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment](https://arxiv.org/abs/2609.28385) | 该论文提出UECR-GRPO方法，通过熵校准的信用重分配，在响应和token两个层面将教师信号与验证器信号统一整合进单一的KL正则化GRPO更新中，从而解决在线策略蒸馏与可验证奖励强化学习结合时教师指导引入时机不当及token重加权破坏任务信用总量的问题。 |
| [^16] | [ForgetMimic: Motion Unlearning for Reinforcement Learning Humanoid Control](https://arxiv.org/abs/2609.28378) | 提出了首个面向物理世界人形机器人控制的动作级遗忘方法ForgetMimic，能够从强化学习策略中选择性移除特定动作（如恶意、被污染或涉及版权的动作），同时保持其余动作的性能不受影响。 |
| [^17] | [LEAP-CBF: A Safety Filter for Uncertain Systems with Least-Effort Adversarial Potentials](https://arxiv.org/abs/2609.28364) | 提出了最小努力对抗势（LEAP），通过量化扰动导致系统失效所需的最小努力来衡量状态的鲁棒性，结合深度强化学习构建了对有界累积扰动鲁棒的安全过滤器。 |
| [^18] | [Local Geometric Mixing via Dobrushin Contraction with Applications to Diffusion Path Monte Carlo and the Proximal Sampler](https://arxiv.org/abs/2609.28338) | 本文提出基于Dobrushin收缩的局部几何混合分析框架，并将其应用于扩散路径蒙特卡洛和近端采样器，在最少假设下为理想方法及Metropolis校正版本提供了混合时间保证。 |
| [^19] | [Learning the Cost of Reliable Inference](https://arxiv.org/abs/2609.28322) | 该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。 |
| [^20] | [PBLH Estimation from Satellite Radiances via a Dual-Encoder Transformer](https://arxiv.org/abs/2609.28286) | 本文基于MetOp-ERA5大规模数据集，对八种PBLH估计方法建立了基准测试，通过分组Shapley分解量化了模型对输入的依赖关系，并提出了一种性能最佳的双编码器Transformer架构。 |
| [^21] | [Non-Commutative State Tracking with Input-Dependent Low-Rank Updates in Mamba-3](https://arxiv.org/abs/2609.28273) | 通过在Mamba-3中引入输入依赖的秩一低秩反射更新，实现单个块内的非对角状态转移，从而支持操作顺序至关重要的非交换状态跟踪。 |
| [^22] | [Predicting Quantization Price for Selecting PTQ Configurations Before Deployment](https://arxiv.org/abs/2609.28270) | 该论文将权重空间后训练量化（PTQ）重构为部署前配置选择问题，用全精度模型的下游曲率为每层量化配置所诱导的输出误差协方差“定价”，从而在统一框架下于部署前预测并比较不同量化格式、粒度、量化器族、变换和比特位宽配置的优劣。 |
| [^23] | [Resource-Adaptive Stochastic Gradient Descent for Online Linear Programming without Re-solving](https://arxiv.org/abs/2609.28263) | 提出资源自适应随机梯度下降算法（RASGD），通过一阶SGD更新模拟重求解的资源定价逻辑，每次请求仅需O(m)次运算即可达到O(log T)期望后悔界，完全无需重新求解线性规划。 |
| [^24] | [RAMP: Robust Adaptive Mixed-Precision Quantization for Edge CPU Vision Models](https://arxiv.org/abs/2609.28262) | 该论文通过对13种逐层INT8量化敏感度指标在多个异构神经网络上的系统性实证研究，揭示了现有指标在现代架构上的系统性失效问题，并据此提出了面向边缘CPU视觉模型的鲁棒自适应混合精度量化方法RAMP，在两个ARM64平台上验证了其在保持精度的同时降低推理延迟的有效性。 |
| [^25] | [Generalizable Robotic Insertion with World Models](https://arxiv.org/abs/2609.28258) | 提出基于世界模型的可泛化机器人插装框架，通过单一世界模型融合本体感知与腕部摄像头视觉信息，在90个几何多样的插装任务上训练后，对未知几何形状的新物体实现56%的零样本成功率（远超无模型基线的7%），且性能随训练物体数量增加而持续提升。 |
| [^26] | [hyperbolix: Hyperbolic Deep Learning in JAX](https://arxiv.org/abs/2609.28248) | 该论文提出了 hyperbolix，首个基于 JAX 与 Flax NNX 构建的全面、通用的开源双曲深度学习库，提供六种统一接口的流形、覆盖各类双曲神经网络层族以及黎曼优化器和降维工具。 |
| [^27] | [Log-Depth Recurrent Language Modeling](https://arxiv.org/abs/2609.28212) | 本文将平衡树递归算子扩展到自回归语言建模，实现了以对数深度和线性运行时间计算所有前缀表示，展现出稳健的长度外推能力和接近ALiBi Transformer的性能。 |
| [^28] | [Support-Compiled Feature Folding: More Evidence at Lower Memory Across Tabular Foundation Models](https://arxiv.org/abs/2609.28208) | 提出免训练推理框架“支持度编译特征折叠”（SCFF），将按支持度排序的有界特征子集路由进冻结的表格基础模型，把二次方特征交互开销降为线性，在不集成预测、不训练新参数的情况下以更低内存利用更多证据，并在18个宽表数据集上提升了全部六个骨干网络的准确率与NLL。 |
| [^29] | [Transferable Evidence Reconstruction for Longitudinal Glucose Representations](https://arxiv.org/abs/2609.28199) | 该论文提出可迁移证据重构（TER）自监督方法，通过在一个记录组上拟合低容量读取器并要求其在另一组记录中恢复相同证据的跨组测试，学习具有可迁移证据解码规则的血糖表征，并利用感知观测的每日编码器和感知时钟的多日记忆模块对持续血糖监测数据进行建模。 |
| [^30] | [Geospatial embeddings detect old-growth forests but buffered spatial validation narrows their advantage over Sentinel features](https://arxiv.org/abs/2609.28194) | 该研究在罗马尼亚南喀尔巴阡山脉绘制原始老林分布图，发现地理空间基础模型嵌入特征能有效检测原始老林，但在采用10公里缓冲区进行空间验证以控制空间自相关后，其相对常规Sentinel-1/2特征的优势明显缩小。 |
| [^31] | [Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination](https://arxiv.org/abs/2609.28182) | 本文提出了一种基于精确二项推断的有限样本概率安全认证框架，能够为闭环电网运行中的黑盒AI决策模型给出不安全运行概率的最紧单侧上界证书，为系统运营商独立严谨地判定AI系统是否可安全部署提供了依据。 |
| [^32] | [How Sensitive Are LLM Leaderboard Claims to Hidden Model Selection?](https://arxiv.org/abs/2609.28177) | 该论文提出一种敏感性分析方法，量化一个排行榜领先幅度背后可能隐藏的私下挑选的模型变体数量，并对 Open LLM Leaderboard 上 394 个相邻排名声明进行审计，发现其中 391 个即使不考虑选择效应也缺乏统计支持。 |
| [^33] | [Confidence Falls Short: Asymmetric Certainty Gains from Optimization Hinder Multimodal Classification](https://arxiv.org/abs/2609.28165) | 该论文发现多模态优化会使强模态获得比弱模态更高的预测置信度（即非对称的确定性增益），并提出MaxCR方法，通过非线性稀疏度量追踪各模态语义置信度并动态施加跨模态干预，以纠正置信度差异、提升多模态分类性能。 |
| [^34] | [EvEMTBench: An Open Benchmark for Machine Learning in Power System Protection](https://arxiv.org/abs/2609.28149) | EvEMTBench 是一个面向电力系统保护机器学习的开放、可执行、可版本化基准，通过统一固定任务定义、数据划分与评估指标，支持跨可观测性条件、分布偏移和跨电网迁移的结构化比较。 |
| [^35] | [RL Starts before RL: On Policy Distillation for Better Reinforcement Learning](https://arxiv.org/abs/2609.28145) | 研究发现在策略蒸馏（OPD）作为RL的准备阶段能带来超越初始准确率提升的收益——即使OPD对准确率几乎没有即时改善，经其初始化的模型在RL后仍能达到更高的最终性能，原因可能在于与教师分布的对齐偏向高质量推理路径，同时保留了RL可通过结果反馈继续优化的备选路径。 |
| [^36] | [NPBoost: Neural Processes with Gradient-Boosted Fixed Effects](https://arxiv.org/abs/2609.28122) | NPBoost将结构化响应变异性分解为跨任务共享的梯度提升树固定效应与捕捉任务间随机变化的神经过程随机效应，并通过提升算法联合训练二者，在共享结构包含不连续性等不规则特征的表格元学习任务上优于标准神经过程。 |
| [^37] | [Probabilistic and Geometry Aware Neural Surrogate of Scrape Off Layer Plasma Simulations](https://arxiv.org/abs/2609.28116) | 该论文提出一种将SOLPS-ITER曲线网格无损展开为图像张量并结合条件流匹配的概率化神经代理模型，能够在偏滤器脱靶转变等敏感工况下保留几何结构信息，并捕捉多种可能的等离子体状态及其不确定性。 |
| [^38] | [Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching](https://arxiv.org/abs/2609.28107) | 该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。 |
| [^39] | [Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness](https://arxiv.org/abs/2609.28105) | 提出Fed-ReMasker，将ReMasker掩码自编码器适配到联邦学习框架中，使各中心能够利用跨协作中心学到的知识填补本地从未观测到的特征，从而解决了现有联邦填补方法很少评估的特征级缺失问题。 |
| [^40] | [Visual Tripwires: Anticipating Failure in Deep Vision Systems](https://arxiv.org/abs/2609.28099) | 提出Visual Tripwires框架，通过监测模型潜在表示漂移、预测振荡、轨迹曲率和注意力熵等时间不稳定性信号，提前预测深度视觉系统在未来时间范围内的失效概率。 |
| [^41] | [Discovery of fully efficient fault indicators along a data-based diagnosis process](https://arxiv.org/abs/2609.28087) | 本文提出 DT4X+，通过改进训练集构建与符号回归损失函数，使诊断表达式在分离目标类别的同时保持解析冗余关系的可解释性，解决了原 DT4X 算法仅优化两类分离而导致类别碎片化、性能下降的问题。 |
| [^42] | [LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations](https://arxiv.org/abs/2609.28086) | 提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。 |
| [^43] | [Curriculum Learning with GNN-based Reinforcement Learning for Job Shop Scheduling](https://arxiv.org/abs/2609.28085) | 本文提出在作业车间调度问题中采用课程学习策略训练基于图神经网络的强化学习模型，通过先在小规模实例上训练再逐步过渡到更大目标规模，相比单一规模训练有效提升了模型的跨规模泛化能力。 |
| [^44] | [Exact Quantile Balancing and Load-Error Injection for Mixture-of-Experts](https://arxiv.org/abs/2609.28053) | 该论文提出精确分位数均衡（EQB）和负载误差注入（LEI）两种方法，分别以极小的通信开销计算精确全局分位数、以及将局部负载误差直接注入路由器梯度，从而在7.5B参数的混合专家模型上显著改善全局与局部负载均衡并提升下游性能。 |
| [^45] | [Tensor Decomposition of Transformer Key-Value Caches: Spectral Structure and Format Comparison](https://arxiv.org/abs/2609.28029) | 该研究通过谱分析发现Transformer的KV缓存中词元和特征模式具有低秩结构而注意力头和层模式近乎满秩，并在相同存储条件下证明Tucker分解在2至5倍压缩比下重构误差最低，且键和值的最优压缩表示形式存在差异。 |
| [^46] | [PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning](https://arxiv.org/abs/2609.28022) | PISCES是一种无需标签、融合多种物理约束的太阳风卷积自编码器，它将异常分数分解为磁场、等离子体、物理关系和残差修正等物理可解释的分量，从而实现空间天气瞬变结构的早期检测与预警。 |
| [^47] | [Improving Ensemble Filters with Flow Matching](https://arxiv.org/abs/2609.28015) | 提出流集合滤波器，利用条件流匹配学习非线性更新，将经典基线滤波器的预报集合传输为分析集合，在稀疏观测的动力学系统中性能超越所有经典集合滤波器及最先进的生成式方法。 |
| [^48] | [SoLiD26: A First Principles Solid-Liquid Interface Dataset for Machine-learned Interatomic Potentials](https://arxiv.org/abs/2609.28013) | SoLiD26是一个包含1540万个第一性原理原子结构、涵盖15种化学元素的固液界面数据集，可用于训练和评估面向电化学、催化和腐蚀等应用的机器学习原子间势。 |
| [^49] | [Evaluating Open-Weight LLMs for Turkish Domain Documents Under Retrieval and Hardware Constraints](https://arxiv.org/abs/2609.28007) | 本文提出了一种无需额外模型调用即可区分检索失败与模型推理失败的带证据标注评估协议，并在6 GB显存的本地硬件约束下系统评估了五个开放权重7B-8B模型对土耳其语长篇领域文档问答的能力。 |
| [^50] | [Shared Global KV with Layer-Specific Local History](https://arxiv.org/abs/2609.28006) | 该论文提出在跨层共享全局KV缓存的同时，为每一层保留独立的局部历史记忆，实验表明相比基于当前token的局部分支可降低约1.4%的困惑度，并在与GQA及相邻层KV共享等方案的对比中取得更优的同源似然。 |
| [^51] | [Learning from Failures: Heterogeneous Graph Memory for Small Language Model Tool-Using Agents](https://arxiv.org/abs/2609.28003) | 提出FRESH框架，通过基于异构图的失败感知检索机制，将智能体的历史成功与失败经验转化为结构化外部记忆，从而提升小型语言模型在工具调用中的可靠性与安全性。 |
| [^52] | [Task-Induced Riemannian Metrics for Vision Transformer Feature Spaces](https://arxiv.org/abs/2609.27988) | 该论文提出用任务诱导的回拉度量取代ViT特征空间中常用的欧氏距离和余弦相似度，设计无矩阵诊断量 κ_cap(r) 判断低秩近似是否可学习，并提出谱回拉网络（SPN）来学习该度量的低秩版本。 |
| [^53] | [PCQC: Privileged Counterfactual Question Credit for Multi-Turn Medical Dialogue](https://arxiv.org/abs/2609.27987) | 提出PCQC方法，利用训练时的特权患者信息构建反事实答案，使多轮医疗对话的强化学习能够对每个问题（包括从未提出的备选问题）进行信用分配，从而为对话策略训练提供问题级别的反馈。 |
| [^54] | [Relative Discharge Stage (RDS) Classification: A Practical Indicator of Battery Discharge Progress](https://arxiv.org/abs/2609.27986) | 本文提出相对放电阶段（RDS）这一电池管理指标，将剩余放电状态划分为五个可解释等级，并通过结合SOC估计与轻量级时序学习的物理信息分类框架，在无需未来电流信息的情况下实现对电池放电进度的实用化评估。 |
| [^55] | [Riemannian Structure and Optimization for a Class of Low-Parametric Orthogonal Matrices](https://arxiv.org/abs/2609.27982) | 本文为一类由块对角因子与固定置换交织而成的低参数量正交矩阵建立了黎曼流形结构，并提出了基于自动微分的高效黎曼优化算法，可应用于最佳矩阵逼近和参数高效微调。 |
| [^56] | [Risk-Controlled KV-Cache Eviction: From Memory Budgets to Risk Targets](https://arxiv.org/abs/2609.27981) | 该论文将KV缓存淘汰从平均内存预算视角重新表述为部署风险控制问题，提出一种与压缩器无关的事后认证程序，通过有限样本保证选择保留策略，确保任务效用实质性退化事件的发生频率满足部署指定的风险目标与置信度要求。 |
| [^57] | [Six Layers Less: Encoder Pruning for Whisper with Label-Free Recovery](https://arxiv.org/abs/2609.27980) | 该论文提出通过留一层法依据词错误率变化对Whisper编码器层进行排序，剪掉影响最小的六层（占编码器的18.5%），无需自定义推理代码，并利用无标签单语语音数据进行蒸馏以恢复性能。 |
| [^58] | [Conformal Bayes under Continuous Label Shift: Sensitivity Analysis and the Limits of Exact Validity](https://arxiv.org/abs/2609.27976) | 该论文提出联合倾斜敏感性共形贝叶斯（JTS-CB/JTS-SCB），通过对预设的合理倾斜集合进行联合敏感性分析来应对连续标签偏移，同时揭示了精确有限样本有效性取决于密度比尾部行为的根本局限。 |
| [^59] | [Linear RNN Scaling Laws: When Longer Sequences Beat More Sequences](https://arxiv.org/abs/2609.27964) | 该论文在教师-学生理论框架下首次为线性RNN语言模型推导出显式的近似、优化与统计缩放定律，并揭示了在特定谱指数条件下，增加序列长度比增加序列数量能更有效地提升模型性能。 |
| [^60] | [I-SplineFlow: Learning Monotone Spline Stochastic Interpolant Schedulers for Few-Step Generation](https://arxiv.org/abs/2609.27963) | 该论文提出I-SplineFlow，用积分单调样条（I-splines）参数化随机插值调度器，将多项式阶数与混合权重数量解耦并利用紧支撑实现局部调控，从而在少步生成中获得更具表达力且优化更稳定的采样轨迹调度器。 |
| [^61] | [ScoutNeRV: Rapid Encoding of Grid-Based Video INRs via ScoutNet](https://arxiv.org/abs/2609.27958) | ScoutNeRV提出一种内容自适应初始化框架，利用轻量级侦察网络分析少量采样帧并从预训练专家记忆库中选取参数来初始化分层视频INR，从而显著降低视频编码的逐视频优化成本。 |
| [^62] | [CS-WCP: Robust Conformal Sets for LLM-Judge Traffic Shifts with Uncertain Group Proportions](https://arxiv.org/abs/2609.27955) | 提出CS-WCP方法，通过为源域和目标域组比例构造同时置信区间并对所有兼容比例向量取加权保形集的并集，在LLM裁判部署流量漂移且组比例只能从无标签样本估计的情况下实现鲁棒的分布无关覆盖保证。 |
| [^63] | [When Accuracy Gaps Fail to Certify: Auditing Cross-Domain Recalibration of LLM Judges](https://arxiv.org/abs/2609.27954) | 该论文通过涵盖13个评判者、2个生成器、8个领域和1,176个迁移设置的大规模实验证明，LLM评判者的源-目标准确率差距既无法可靠预测跨域重校准的迁移结果，也无法作为校准失效的证明，并提出了一种基于有限样本证书、使用互不相交审计标签的无泄漏审计方法。 |
| [^64] | [The Recall Ceiling of LLM Recommendation Reranking](https://arxiv.org/abs/2609.27953) | 该论文揭示了LLM推荐重排序评估中“预言机协议”会高估性能92-95%，指出由于现实检索仅能覆盖2-19%的相关物品，召回率上限为任何封闭候选集重排序器设定了确定性的NDCG上界，且在现实条件下各类优化策略均无法显著超越协同过滤基线。 |
| [^65] | [Enhancing Multiclass Malware Classification in Resource-Constrained Environments](https://arxiv.org/abs/2609.27950) | 本文提出了一种结合SMOTE过采样的LightGBM轻量级机器学习模型，能够在物联网等资源受限环境中高效且准确地实现多类别恶意软件分类。 |
| [^66] | [From Sentiment Classification to Actionable and Responsible Feedback: A Scoping Review and Evidence Map of NLP in Student Evaluation of Teaching, 2015-2026](https://arxiv.org/abs/2609.27939) | 该范围综述通过对2015-2026年间421项研究的技术演进与价值维度进行证据图谱映射，揭示了学生评教NLP研究中从技术演示到面向最终用户的可操作应用之间存在约50个百分点的显著断层。 |
| [^67] | [Quality over Quantity: Semi-Supervised Detection of Illicit Bitcoin Flows via Feature Engineering](https://arxiv.org/abs/2609.27936) | 该论文基于1.63亿笔比特币交易提出半监督学习框架检测共享发送混合器中的非法资金流，证明其成功关键在于高保真特征工程带来的数据质量而非数据量。 |
| [^68] | [Binary Quantized Neural Network Training Is W[1]-Hard Parameterized by Input and Output Dimensions](https://arxiv.org/abs/2609.27932) | 本文证明了二值量化神经网络训练在仅以输入与输出维度之和为参数时是W[1]-难的，即使在零误差、样本构成前缀链且偏置固定为零的最简情形下依然成立，从而否定了Ganian等人留下的开放问题。 |
| [^69] | [Dirichlet Process Mixtures of Trees with Gaussian Process Splits: A Bayesian Nonparametric Framework with Posterior Contraction Rate](https://arxiv.org/abs/2609.27930) | 该论文提出了一种基于狄利克雷过程先验和由高斯过程后验预测驱动分裂规则的贝叶斯非参数回归树混合框架，统一了CART、BART、随机森林和提升方法，并证明了在真实回归函数仅连续的宽松条件下Hellinger距离上 $n^{-1/4}$ 的后验收缩速率。 |
| [^70] | [Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays](https://arxiv.org/abs/2609.27917) | 本文通过预先注册的验证性实验发现，工作负载中实例难度的离散程度是决定测试时预算重新分配是否划算的关键属性，并且即使计入分配策略自身消耗的预算成本，该策略依然可能带来收益。 |
| [^71] | [Reliable Fusion of Conflicting Experts](https://arxiv.org/abs/2609.27913) | 提出一种基于概率电路的动态融合框架，利用上下文相关的可信度估计来可靠聚合多个黑盒专家（如大语言模型）的意见，无需访问专家内部表示或再训练，即可在冲突场景下显著提升预测性能与决策可靠性。 |
| [^72] | [Global tree forecasters collapse at the hierarchical aggregate: a five-panel failure characterization](https://arxiv.org/abs/2609.27912) | 本文首次系统刻画了全局梯度提升树模型在预测层级聚合值时的崩溃失效——当聚合总量远超训练范围时低估可达496倍，并指出逐序列缩放等简单预处理即可防止该失效。 |
| [^73] | [False-science induction in autonomous scientific discovery](https://arxiv.org/abs/2609.27883) | 该论文揭示了自主科学发现系统中的“伪科学诱导”现象——当物理对象与测量结果被连贯地错误配对时，神经代理模型会学到虚假关联并系统性地将实验预算引向低性能区域，且错误的一致性而非错误频率是决定性变量。 |
| [^74] | [Noise-Induced Predictability Redistribution Across Forecast Horizons of Extreme Events in Chaotic Dynamics](https://arxiv.org/abs/2609.27877) | 该研究发现在混沌系统中，动力学噪声会重分布极端事件的可预测性——在短预测视界上显著提升预测技能（配对增益0.0895），而随预测间隔增大可预测性逐渐衰减。 |
| [^75] | [Evaluation Choices Decide the Forecasting Leaderboard: Evidence from a Production Marketplace Panel](https://arxiv.org/abs/2609.27867) | 在固定数据、预测时长与时期的条件下，仅改变评估设计——分析单位、误差聚合方式和评分对象（预测区间还是点预测）——即可颠覆或消解预测基准测试的主要结论，证明排行榜结果在模型拟合之前就已由评估者的选择决定。 |
| [^76] | [A Shared Encoder Is Not a Shared Task: Conditional Comparison for Deep Expert Pools](https://arxiv.org/abs/2609.27866) | 该论文发现共享深度编码器无法消除任务比较分数中的混淆，并提出将条件化双判别器差异移植到嵌入空间形成“功能轴”度量，既能免疫输入旋转的外推混淆又能敏感捕捉标签置换漂移，从而在混合多头生命周期中以更少头数取得更优的决策质量。 |
| [^77] | [What Changed? Drift Detection with Real, Virtual, and Incomparable Diagnosis](https://arxiv.org/abs/2609.27865) | 提出将条件双判别器差异移植到嵌入空间的双轴诊断方法，同时弥补交换分数对输入旋转的虚假敏感和表示新颖性分数对标签置换漂移的盲区，从而在混合头生命周期中以更少头部实现更优漂移检测决策。 |
| [^78] | [Exact Minimax One-Bit Unbiased Compression: Heavy-Tail Necessity and Finite-Randomness Approximation](https://arxiv.org/abs/2609.27860) | 该论文精确刻画了一比特无偏压缩的极小极大最优值，证明高斯情形的最优编码必然具有临界重尾结构。 |
| [^79] | [ChronosAttack: Adversarial Tool Scheduling Attacks on LLM Agents](https://arxiv.org/abs/2609.27857) | ChronosAttack是一种仅通过延迟真实工具响应的到达时间（不改动其内容）就能改变LLM智能体处理证据的顺序并操纵其最终决策的对抗性调度攻击。 |
| [^80] | [Theoretical Study on the Evidential Learning-based Variational Autoencoder](https://arxiv.org/abs/2609.27853) | 该论文从理论上证明，证据学习变分自编码器中正态-逆伽马潜在层级的四个参数仅有三维商空间 (γ, α, c) 是可辨识的，并且通过对前向KL散度的精确偏最小化，可以实现从四参数到三坐标的精确约简，同时保持最优值不变。 |
| [^81] | [Query Implied Generative Engine Optimization](https://arxiv.org/abs/2609.27845) | 提出了QI-GEO方法，无需依赖显式查询，直接从文档本身近似其意图空间并推断用户意图，从而优化内容在生成式搜索引擎中的可见性。 |
| [^82] | [From Reasoning Strings to Partial Orders: Verifier-Certified Rule Transport through Quotient Policy Optimization](https://arxiv.org/abs/2609.27833) | 该论文提出 VCRT 方法，通过原生验证器重放相邻操作对来认证交换性，把成功轨迹从线性 token 序列提升为带证书的偏序结构，并将策略信用聚合到认证轨道上，从而在 ProofWriter、CLRS 和 Lean 的跨环境迁移中保留真正的逻辑依赖并提升泛化能力。 |
| [^83] | [CAST: Context- and Anomaly Structure-Conditioned Time Series Anomaly Generation](https://arxiv.org/abs/2609.27825) | 该论文提出CAST框架，采用两阶段预训练-微调策略——先利用大量正常数据学习系统动态以缓解异常数据稀缺，再以学习到的异常结构表示为条件约束生成器，从而解决时间序列异常生成中异常数据稀缺与形态异质两大挑战。 |
| [^84] | [When Adaptation Hurts: Split Sensitivity and Person-Level Negative Transfer in Federated Wearable Onboarding](https://arxiv.org/abs/2609.27819) | 该研究通过严格的防泄漏评估协议发现，联邦可穿戴模型的无标签引导尽管平均表现良好，但在个体用户层面存在显著的负迁移现象，且不同引导策略之间并无统计学上显著的优势差异。 |
| [^85] | [Type-II Error Bounds for Test Supermartingales from Lower-Tail Hypotheses](https://arxiv.org/abs/2609.27766) | 本文针对检验上鞅方法中的第二类错误问题，研究了对对数增量下尾概率的不同假设所导出的第二类错误界，并将所有结果统一为一个主不等式，即通过 e 变量（逆）矩生成函数的单侧勒让德变换来刻画序贯检验在固定时域和任意时刻的第二类错误上界。 |
| [^86] | [The Type-II Error of Test Supermartingales: e-Power versus the Chernoff-Stein Exponent](https://arxiv.org/abs/2609.27765) | 该论文证明了 e-幂（对数增长率）本身无法为检验上鞅的第二类错误提供任何有限时间保证，而真正控制第二类错误的是 e-变量的 Chernoff-Stein 指数。 |
| [^87] | [Learning to Detect Symbolic Failure: Machine Learning and the Limits of Black-Scholes](https://arxiv.org/abs/2609.27764) | 在拥有专家设计符号特征的期权定价领域，保留领域结构的树模型在检测Black-Scholes模型系统性偏差方面比学习抽象表示的核方法高出21.5个百分点，证明“保留结构优于学习抽象”。 |
| [^88] | ["What's That Sound?": A Versatile, Robust, and Lightweight Convolutional Transformer for Environment Sound Recognition](https://arxiv.org/abs/2609.27762) | 本文提出了RALCT模型，通过对音频进行随机增强并将MFCC图与对数梅尔频谱图拼接，结合CNN与Transformer架构高效提取声音特征，以仅约31万的轻量参数量实现对环境声音的准确识别，可部署于移动设备以提升听障人士的安全。 |
| [^89] | [Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning](https://arxiv.org/abs/2609.27760) | FedMAST防御方法通过综合结构、频谱和历史三轴互补证据对客户端更新进行评分并分层过滤遏制，从而检测出即使能绕过孤立异常信号的隐蔽后门攻击，因为后门投毒更新必然留下结构性痕迹。 |
| [^90] | [Evaluation of pre-trained models for pedagogical assessment of novel AI-assisted educational questions](https://arxiv.org/abs/2609.27749) | 该研究通过评估传统机器学习、Transformer和大语言模型在布鲁姆层级分类任务中的表现，并借助特征工程策略，寻找在AI生成的分布外教育问题上依然稳健的教学质量自动评估方法。 |
| [^91] | [Less Language, More Latents: Annotation-Efficient VLAs for Driving](https://arxiv.org/abs/2609.27747) | LADA提出三阶段流水线，通过向量量化的潜在动作模型将大规模无标注观测-轨迹对转化为紧凑的车辆高层意图码本，仅需少量语言标注即可训练出可被语言操控的标注高效驾驶VLA模型。 |
| [^92] | [Limiting-Kernel Q($\lambda$): Bridging Short and Long Horizons](https://arxiv.org/abs/2609.27741) | 提出 LKQL——一种将 n 步截断与基于极限核的长时域近似相结合的离策略价值估计器，在与 n 步估计器相同计算复杂度的前提下实现了短时域与长时域评估的桥接，并可直接嵌入各类 actor-critic 算法。 |
| [^93] | [MENO: Memory-Efficient Neural Operator](https://arxiv.org/abs/2609.27739) | MENO是一种基于流形函数编码器的内存高效PDE神经求解器，其内存占用与数据分辨率无关，支持任意几何域和离散化输入（包括跨几何场景），并在大多数基准测试中取得了最佳精度。 |
| [^94] | [NS-ATTENTION: Newton-Schulz Transformations of Attention Outputs in Vision Transformers](https://arxiv.org/abs/2609.27735) | 提出无参数的Newton-Schulz注意力变换（NS-Attn.），对每个注意力头输出进行谱处理以降低谱集中度并提高有效秩，在ViT和Swin于CIFAR-10/100的全部12组对比实验中均带来平均0.25–0.83个百分点的准确率提升。 |
| [^95] | [FFM-CP: Cross-Backbone Fusion of Vision-Language Foundation Models for Few-Shot Computational Pathology](https://arxiv.org/abs/2609.27710) | 该论文提出了FFM-CP框架，通过闭式正交Procrustes变换对齐多个病理学视觉-语言基础模型的异构表示并利用统一图结构实现信息融合，无需额外训练对齐网络即可在小样本条件下有效融合各基础模型的互补能力。 |
| [^96] | [What Do Tabular Foundation Models Compute In Context? In-Situ Representation Refinement through Attention-Gated Updates](https://arxiv.org/abs/2609.27679) | 提出“原位表示精炼”机制并构建RefineICL——一种注意力门控、无FFN的上下文学习堆栈，使表格基础模型在不改变参数的情况下利用支持集标签精炼回合表示并迁移至查询，性能超越TabPFN-3等现有模型。 |
| [^97] | [Robust Adversarial Reinforcement Learning with Risk Sensitivity and Critic Consistency Regularization](https://arxiv.org/abs/2609.27667) | 提出 RACER 统一框架，从风险敏感视角出发，通过状态相关的自适应对抗目标和评论家一致性正则化，解决了鲁棒对抗强化学习中优化不稳定和价值估计有偏的问题，提升了智能体在动态不确定性下的鲁棒性。 |
| [^98] | [Private Decentralized Optimization with Noise Reduction and Bias Correction](https://arxiv.org/abs/2609.27658) | PRDO通过同批次梯度差的递归估计降低采样与隐私噪声，并结合精确扩散组件校正数据异构导致的去中心化偏差，在无需数据异构性一致有界假设的情况下实现了更优的隐私去中心化优化性能。 |
| [^99] | [FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation](https://arxiv.org/abs/2609.27657) | FLEET通过引入记忆机制，将生成过程表示为基于熵阈值状态的稀疏轨迹，并利用每token效用分数调整logits，实现了与重复采样相同的准确率但速度提升3倍。 |
| [^100] | [FedIncome: Federated Learning for Income Estimation in Digital Lending Under Data Sovereignty Constraints](https://arxiv.org/abs/2609.27654) | FedIncome提出了一种联邦学习框架，使放贷机构无需共享原始借款人数据即可协同训练收入估计模型，在保障数据主权的同时，为小样本机构带来了显著的预测性能提升。 |
| [^101] | [Learning Local Heterogeneity and Cross-Region Context for Large-Scale Traffic Forecasting](https://arxiv.org/abs/2609.27637) | 提出LoReST局部-区域时空网络，在节点邻域和路网区域两个互补粒度上建模空间依赖，兼顾局部异质性捕获与跨区域上下文获取，实现高效的大规模交通流预测。 |
| [^102] | [Pheno-GS: Phenoscape-scale Geodesic Sinkhorn](https://arxiv.org/abs/2609.27633) | Pheno-GS通过图连通性正则化、基于KL惩罚的不平衡最优传输和批处理矩阵算法三大组件，实现了在噪声、稀疏、不平衡的大规模单细胞数据场景下，准确且可扩展地计算患者间分布的测地传输距离。 |
| [^103] | [Efficient Linear Bandits via Cluster-Aware Sketching](https://arxiv.org/abs/2609.27594) | 提出聚类草图线性赌博机算法，通过在每个聚类中保留完整协方差信息并利用哨兵机制进行聚类切换，在保证稳健次线性遗憾的同时显著降低了每轮更新的计算成本。 |
| [^104] | [Hidden not Deleted: How Networks Suppress Entangled Features](https://arxiv.org/abs/2609.27593) | 该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。 |
| [^105] | [The Capability Manifold and ML Scaling Laws](https://arxiv.org/abs/2609.27588) | 本文提出“能力流形”这一多维框架，通过有界缩放函数将模型下游能力（如推理、规划等）与预训练、后训练和测试时资源关联起来，弥补了传统缩放定律仅依赖损失无法刻画模型能力差异的不足。 |
| [^106] | [Does Step Law Transfer to Small-Scale Language Models? An Empirical Recalibration Below 59M Parameters](https://arxiv.org/abs/2609.27581) | 该论文首次实证检验了步进定律在59M参数以下小规模语言模型区间是否成立，并针对最优学习率与批量大小的幂律公式在此区间进行了重新校准。 |
| [^107] | [VCMM: Variance-Calibrated Momentum for Multimodal Learning](https://arxiv.org/abs/2609.27577) | 针对多模态训练中的模态不平衡问题，VCMM 通过在线估计各模态的梯度噪声与时间漂移，并利用卡尔曼式控制器自适应地校准模态特定的动量参数，使梯度记忆与各模态的梯度动态相匹配，从而改进多模态联合优化。 |
| [^108] | [DCRL: Decoupling and Coupling Reinforcement Learning via Policy-Reward Manifold Alignment](https://arxiv.org/abs/2609.27572) | 提出DCRL方法，从几何视角将大语言模型推理建模为逻辑推理、评估与表示三个耦合子流形，并通过策略-奖励流形对齐来解决现有奖励系统中优化不稳定和奖励欺骗的问题。 |
| [^109] | [TNLearn: An Open Source Python Package for Task-based Neurons](https://arxiv.org/abs/2609.27564) | TNLearn是一个开源Python软件包，实现了任务驱动神经元和网络的自动化构建与顺畅训练，推动了“针对特定任务定制神经元”这一新范式的科研与产业应用。 |
| [^110] | [PhyMo: A Physical-Field Modality for Multimodal AI4Physics](https://arxiv.org/abs/2609.27554) | 该论文提出PhyMo框架，创新性地引入“物理场模态”这一全新模态，通过PDE关联算子组织异构物理测量数据，并采用三阶段学习流程（PDE残差监督预训练、与视觉嵌入对齐、多模态融合）来提升物理系统预测能力。 |
| [^111] | [EBRL: Asynchronous Embodied RL by Multi-Grained Resource Management](https://arxiv.org/abs/2609.27547) | EBRL通过异步流水线调度器消除具身强化学习训练中的同步停顿，并利用细粒度的CPU/GPU资源池化管理与动态资源调整，大幅提升硬件资源利用效率。 |
| [^112] | [Robustness of Diffusion Models under Distribution Shift](https://arxiv.org/abs/2609.27546) | 本文首次从理论上刻画了分布偏移下扩散模型的鲁棒分数估计，证明其可分解为学习参考分布的统计代价与随Wasserstein半径二次增长且极小极大最优的偏移代价，并构造了无需知晓偏移半径即可达到最优鲁棒速率的有限样本估计器。 |
| [^113] | [ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning](https://arxiv.org/abs/2609.27532) | 提出 ProCredit，利用可在中间状态上运行的验收检查，把与最终结果同样可验证的任务进展转化为逐步的信用信号，从而克服长程智能体强化学习中仅依赖结果奖励导致的训练信号稀疏、失败尝试无法区分、推进任务的步骤得不到应得信用等问题。 |
| [^114] | [M3D-Net: Hierarchical Coordination of Spatial Context, Feature Reuse, and Differential Attention for Mammography Classification](https://arxiv.org/abs/2609.27523) | M3D-Net通过分辨率感知的算子布局，将多尺度坐标注意力、有界动态特征重用和差分注意力进行分层协调，在乳腺X线摄影分类任务上取得了97.78%的最高验证准确率。 |
| [^115] | [WhatWorkedBench: Benchmarking Experimental Understanding in AI Agents](https://arxiv.org/abs/2609.27490) | 该论文提出了WhatWorkedBench基准，用于评估AI研究智能体的实验理解能力，即智能体在预算受限实验后预测组件变化如何影响实验结果的准确性。 |
| [^116] | [Learning Where to Look: A Shared Relative-Alignment Module for Time-Series Forecasting and PPG-to-Vital-Sign Reconstruction](https://arxiv.org/abs/2609.27473) | 本文提出ROOSTER——一个通过逐头可学习的周期梳状偏置自动学习目标与条件序列之间对齐方式的共享条件化模块，它能同时胜任PPG到生命体征重建和多变量时间序列预测两类任务，并在多项基准上超越现有基线。 |
| [^117] | [DeltaS: Reading the Gated Linear Attention State for KV Cache Eviction in Streaming Video](https://arxiv.org/abs/2609.27470) | 该论文提出利用门控delta线性注意力循环状态在帧块上的变化量作为信号，在问题到来之前决定流式视频KV缓存的驱逐策略，无需代理查询或额外计算。 |
| [^118] | [Quantum Reinforcement Learning for Cost and Delay Tradeoffs in Quantum Cloud Orchestration](https://arxiv.org/abs/2609.27446) | 该论文提出QRLQ框架，将参数化量子电路与D3QN相结合用于量子云任务调度，能够动态权衡成本与延迟，相比启发式基线平均成本降低5-11%。 |
| [^119] | [Stable Neural Decoding Across Sessions via Task-Conditioned Latent Alignment for Brain-Machine Interfaces](https://arxiv.org/abs/2609.27441) | 提出任务条件化潜在对齐（TCLA）框架，通过学习固定的共享潜在空间并按任务条件分别对齐源与目标神经分布，显著提升了脑机接口跨会话长期神经解码的稳定性。 |
| [^120] | [Counterfactual Constraint-Conditioned On-Policy Distillation for Multi-Constraint Instruction Following](https://arxiv.org/abs/2609.27421) | 提出CC-OPD方法，颠覆传统蒸馏的监督方向，通过从教师模型条件中依次消融各约束并利用逐词元概率差分构建每约束的监督信号，从而提升大语言模型的多约束指令遵循能力。 |
| [^121] | [When Labels Are Scarce: An Oscillatory State Space Model for Vibration Diagnosis](https://arxiv.org/abs/2609.27411) | 提出了仅含约4万参数的紧凑振荡状态空间模型DualRes，通过融合两种互补频谱视图与选择性振荡记忆，在标签稀缺的振动故障诊断任务上以极少标注数据（每类约6秒）实现了最先进的性能。 |
| [^122] | [Active Learning for Biodiversity Monitoring: From Label Efficiency to Reliable Ecological Inference](https://arxiv.org/abs/2609.27409) | 主动学习虽能显著降低生物多样性监测中的标注成本，但其非随机的样本选择方式使标签无法用于验证、校准和阈值选择，论文主张在有限专家预算下统筹模型训练与可靠生态推断的资源分配。 |
| [^123] | [EvoAudio: Recursive Self-Improvement for Audio Understanding](https://arxiv.org/abs/2609.27389) | 提出EvoAudio，首个在闭环中同时演化模型、音频波形、问题和难度的递归自我改进系统，无需新的人工标注即可通过可验证监督与强化学习持续提升音频理解能力。 |
| [^124] | [Forecast Workflow Bench: Evaluating Language-Model Decisions with Budgeted Forecast Tools](https://arxiv.org/abs/2609.27385) | FWBench 提出了一个通过预算约束下的时间序列预测工具使用来评估语言模型决策能力的基准，发现 GPT-6 Astra 仅用 2.5% 的预算有选择地购买短时程预测即可胜过固定策略，首次实现了对决策质量与预测成本权衡的可复现评估。 |
| [^125] | [Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models](https://arxiv.org/abs/2609.27373) | 该论文发现循环语言模型的注意力路由在早期循环步骤即趋于稳定，据此提出无需训练的WISE推理方法：早期用全局注意力发现稀疏工作集，后续步骤重用该支撑集，在保持推理质量的同时大幅减少重复的全局注意力计算。 |
| [^126] | [Anomaly-Free Self-Optimization via AUC Bounds](https://arxiv.org/abs/2609.27362) | 本文提出将AUC边界作为可微分、无需异常数据的目标函数，直接优化异常检测系统的连续参数（包括集成权重和可学习的分数重缩放机制），突破了传统有限候选集选择的限制，在多个数据集和嵌入模型上实现了显著的性能提升。 |
| [^127] | [Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction](https://arxiv.org/abs/2609.27355) | 本文提出一种量化鲁棒的机器遗忘框架，通过基于曲率的敏感权重判据和敏感度引导的噪声正则化，将模型收敛引导至更平滑的极小值，使遗忘效果在量化压缩后依然保持鲁棒，同时维持整体模型效用。 |
| [^128] | [MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded Molecular Design](https://arxiv.org/abs/2609.27349) | 提出了MolDesignBench——一个面向真实场景的分子设计基准，包含2000个融合隐式设计需求与显式约束的生成与优化任务并需要调用17种专业化学工具，实验表明当前前沿大语言模型智能体在这些真实分子设计任务上的成功率仍然很低。 |
| [^129] | [Evolving Inspectable O-RAN Slicing xApps with LLMs](https://arxiv.org/abs/2609.27337) | 本文提出用大语言模型将O-RAN网络切片控制器自动演化为紧凑且可读、可编辑的Python程序，取代决策逻辑不可解释的深度强化学习神经网络策略，在保留自适应资源分配能力的同时，让运营商能够直接检查和修改控制逻辑，并在真实5G测试平台上验证了其有效性。 |
| [^130] | [A Hybrid Iterative Deep Ritz Method for Elliptic Interface Problems](https://arxiv.org/abs/2609.27325) | 本文提出了一种求解椭圆界面问题的混合迭代深度Ritz方法（H-IDRM），通过新的混合形式与水平集神经网络架构，仅采用体积表示避免了显式界面采样，并给出了涵盖神经网络近似、蒙特卡洛近似、迭代格式和惩罚参数误差的完整误差分析。 |
| [^131] | [Turning Safety into Competence: Minimally Exploitable Robot Policies via Safety-Filtered Reinforcement Learning](https://arxiv.org/abs/2609.27312) | 提出了S2C两阶段强化学习框架，通过对抗性强化学习训练鲁棒安全过滤器并将其与竞争任务学习分离，证明了安全过滤可保持策略的不可利用性，使机器人在竞争任务中胜率最高且最难被攻击利用。 |
| [^132] | [FairTest: Search-Based Fairness Testing for Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2609.27309) | 本文提出了FairTest，一种基于搜索的测试方法，通过三个适应度函数（已执行运行的公平性、预测公平性、决策不确定性）引导搜索并结合测试优先级排序，以发现多智能体强化学习策略中的不公平执行。 |
| [^133] | [Discrete Diffusion Models via Evolving Variational Autoregressive Networks](https://arxiv.org/abs/2609.27306) | 提出一种利用变分自回归网络参数化归一化概率分布的离散扩散模型，通过显式马尔可夫跳跃算子控制加噪与去噪动力学，将归一化离散扩散模型成功扩展至高维晶格上的自旋系统，并准确计算了二维和三维伊辛模型的自由能、能量、磁化强度等热力学量。 |
| [^134] | [Live Assistant: Learning Whether, When, and Whom to Assist in Real-World Live Social Streams](https://arxiv.org/abs/2609.27303) | 该论文提出Live Assistant框架，将真实直播场景中的智能协助形式化为“是否行动、何时行动、面向谁、传达什么”四个耦合决策，由自回归策略每10秒基于音视频、评论、礼物等多模态直播数据自主选择沉默、记录或回复，并构建轨迹引擎将真实直播会话转化为结构化因果监督数据。 |
| [^135] | [Beyond the Illusion of Power: Calibrating Quasi-Experiments in Observational IS](https://arxiv.org/abs/2609.27299) | 该论文通过大规模蒙特卡洛模拟（9837个参数条件、约980万个数据集）分解了观测性IS研究中准实验设计计划功效与实际功效之间的差距，发现序列相关可由AR(1)感知的计算器部分校正，但面板流失、错位采用偏差和平行趋势预检验无法用闭式公式刻画，仅外生流失就会使功效降低约8至11个百分点。 |
| [^136] | [KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling](https://arxiv.org/abs/2609.27294) | KITE提出了一种新的模型扩展范式，通过将新增参数放置在不影响注意力KV缓存的区域，使模型在从小到大扩展时既节省训练成本（通过升级复用），又节省推理成本（KV预填充只需依赖较小的模型部分）。 |
| [^137] | [NGN: Learning Neural Network Size as a Differentiable Count](https://arxiv.org/abs/2609.27291) | 提出了神经发生网络（NGN），通过可学习的边界以可微分方式让模型在训练中自动学习所需的结构组件数量，训练后仅需部署所学前缀即可，性能几乎不受影响。 |
| [^138] | [SR-Fraud: An Outcome-Supervised Reflective LLM Agent Framework for Non-Stationary Payment Fraud Detection](https://arxiv.org/abs/2609.27287) | SR-Fraud提出了一种结果监督的反思式LLM智能体框架，通过将冻结无状态的实时交易评分智能体与离线反思适应机制解耦，并借助确定性验证器仅将经过验证的边界假设纳入可执行知识状态，从而有效应对非平稳支付欺诈检测中攻击者快速适应与爆发式攻击的挑战。 |
| [^139] | [Multitask Regression with Pairwise Fusion](https://arxiv.org/abs/2609.27280) | 该论文提出一种通过对跨任务所有成对系数差异进行惩罚来估计多任务回归系数矩阵的方法，能够灵活刻画不同预测变量上任务间系数的共享与差异结构，并在活跃预测变量数和异常系数数这两个结构量上实现了匹配的上下界。 |
| [^140] | [Graph Learning with Spectral Connectivity Priors for Scarce Data](https://arxiv.org/abs/2609.27278) | 提出 SCoGL 框架，通过在组合拉普拉斯约束的 GLASSO 目标中加入基于拉普拉斯特征值的谱连通性先验，在稀缺数据条件下显式促进图的全局连通性，从而改善图恢复效果并提升图信号去噪等下游任务性能。 |
| [^141] | [TimeEvo: Failure-Driven Self-Evolution of a Time Series Agent](https://arxiv.org/abs/2609.27277) | 提出TimeEvo框架，通过将智能体的失败诊断聚类为能力缺口、为每个缺口规划测量、合成证据工具并通过配对准入门控筛选，实现了时间序列智能体工具库的故障驱动自进化，解决了人-智体工具错配和静默损害两大问题。 |
| [^142] | [What Converges in the Platonic Representation Hypothesis? Structure over Geometry](https://arxiv.org/abs/2609.27252) | 该研究挑战了柏拉图表征假说的既有解读，通过受控的2×2框架证明模型间真正收敛的是关系结构而非度量几何。 |
| [^143] | [Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders](https://arxiv.org/abs/2609.27248) | 本文提出LLMAE方法，通过在预训练语言模型内部引入固定长度潜瓶颈，将其改造为高保真连续文本自编码器，可近乎完美地重建长达1024个token的文本序列。 |
| [^144] | [Full-Covariance Smoothing of Bayesian Neural Networks for Online Adaptation](https://arxiv.org/abs/2609.27244) | 提出通过互协方差恒等式实现贝叶斯神经网络中全协方差的前向传播，突破了现有平滑方法仅支持对角协方差的局限，实现了无需梯度迭代的闭式在线自适应学习。 |
| [^145] | [On the Sample Complexity of Active Learning with Membership Queries](https://arxiv.org/abs/2609.27241) | 本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。 |
| [^146] | [Discover, Falsify, Revise: Auditing Input-Use Claims from Source Code to Predictive Contribution in Agent-Discovered Cell Models](https://arxiv.org/abs/2609.27234) | 提出CELLAUDIT审计框架，从源代码可访问性、预测依赖性和预测贡献三个层面检验AI虚拟细胞模型是否真正使用了输入的扰动信息，并揭示一个留出集性能看似良好的智能体发现预测器实际上对化合物替换完全不变。 |
| [^147] | [A Scaling Study for fMRI Foundation Models](https://arxiv.org/abs/2609.27232) | 本研究通过超过10,000 GPU小时的受控实验首次系统探索了fMRI基础模型的缩放规律，发现数据与模型规模应协同扩展，且在相同计算量下增加预训练数据比增大模型规模能使更多下游任务受益。 |
| [^148] | [Physiologically Informed Digital Auscultation for Pneumonia Detection in Long-term Care Residents](https://arxiv.org/abs/2609.27222) | 本研究利用多通道数字听诊录音，以胸片为监督信号训练深度学习模型，实现对长期护理机构老年人肺炎的客观检测，且仅需三个胸中部位听诊通道即可保持高性能并具备可解释性。 |
| [^149] | [Tail-Aware Geometry Learning for Conformal Ellipsoids](https://arxiv.org/abs/2609.27221) | 提出了一种尾部感知的共形椭球几何学习框架，通过在估计集上进行CVaR约束下的体积最小化来学习度量矩阵、再在独立校准集上进行标准共形校准，将尾部敏感性与覆盖保证解耦，从而提升多元共形预测集的效率。 |
| [^150] | [KATOsuper: Surrogate-accelerated neural topology optimization with sensitivity-consistent Fourier neural operators](https://arxiv.org/abs/2609.27216) | 该论文提出KATOsuper框架，利用敏感性一致傅里叶神经算子（SC-FNO）与forward_split架构，通过自动微分保证预测目标与优化梯度的一致性，从而解决神经代理拓扑优化中的不稳定性并实现显著加速。 |
| [^151] | [Scalable Subgraph Sampling via Resistance Curvature](https://arxiv.org/abs/2609.27209) | 该论文提出ERC-LG，一种结合Johnson-Lindenstrauss投影与多GPU批量共轭梯度求解器的大规模图电阻曲率近似方法，避免了伪逆计算与完整嵌入存储，并利用所得曲率引导节点与边采样以构建GNN训练子图，在七个数据集中的六个上取得最高的节点分类准确率。 |
| [^152] | [Benchmarking Active Spot Selection for Cost-Efficient Spatial Transcriptomics](https://arxiv.org/abs/2609.27208) | 该论文针对空间转录组学建立了主动学习点选择的回顾性基准，在多种预算水平下系统比较了基于不确定性与多样性的选择策略和随机采样的表现，为成本高效的空间转录组学采样提供了实证依据。 |
| [^153] | [Prediction with Expert Advice: Anytime Regret with Many Experts Matches the Fixed-Time Constant](https://arxiv.org/abs/2609.27206) | 本文提出一种无需预知时间视界的专家建议预测算法，使任意时刻的累积遗憾达到 $(1+O(\sqrt{\ln\ln n/\ln n}))\sqrt{t\ln n/2}$，消除了此前的 $\sqrt{2}$ 因子差距，从而在多专家情形下将任意时刻遗憾匹配到固定时限的最优常数。 |
| [^154] | [Reliable Federated TinyML Deployment for IoT Security](https://arxiv.org/abs/2609.27202) | 该论文提出将联邦学习与TinyML模型压缩技术（知识蒸馏、结构化剪枝和量化）相结合，实现资源受限物联网设备上的隐私保护入侵检测，并发现服务器协调的余弦学习率调度对联邦TinyML系统的训练稳定性至关重要。 |
| [^155] | [A Systematic Benchmark of Explainable Methods for Temporal Attribution in Sequential Recommendation Systems](https://arxiv.org/abs/2609.27201) | 该论文首次针对序列推荐系统系统性地基准测试了基于梯度、扰动和注意力的可解释方法在时间归因上的忠实性，并提出了双模型掩码评估指标来衡量各方法的效果。 |
| [^156] | [ZO-COSMO: Index-Free One-Hop Mixing for Decentralized Zeroth-Order Optimization](https://arxiv.org/abs/2609.27199) | ZO-COSMO 提出了一种无索引的单跳混合方法，通过将双查询估计与保持平均的掩码共识相结合，解决了去中心化零阶优化中稀疏通信的对等状态兼容问题，并在理论上给出紧致收缩界与收敛保证，实验中在相同通信预算下优于显式索引方法。 |
| [^157] | [Data-driven discrete-time deep recurrent neural network-based modeling for dissipative systems](https://arxiv.org/abs/2609.27186) | 本文提出DissipNet，一种通过结构化权重约束和专门训练算法显式保证耗散性的深度离散时间循环神经网络，能够在学习耗散系统动力学的同时确保系统固有的稳定性。 |
| [^158] | [Artificial intelligence surrogates for treatment effect estimation with before-and-after data](https://arxiv.org/abs/2609.27180) | 该论文提出一个新框架，利用预训练AI模型对每位患者治疗前后的测量数据进行结局预测并比较个体内差异，从而以AI预测作为低成本替代指标来估计治疗的因果效应。 |
| [^159] | [Median Temporal Ensembling: Training-Free Robust Aggregation for Action-Chunked Visuomotor Policies](https://arxiv.org/abs/2609.27167) | 该论文提出无需训练的中位数时间集成方法，用坐标式中位数取代指数加权平均来聚合动作分块视觉运动策略的重叠预测，从而在面对不断增强的对抗性攻击时保持鲁棒稳定性，克服了传统时间集成崩溃点为0、以及对抗微调在强攻击下性能急剧退化的问题。 |
| [^160] | [Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models](https://arxiv.org/abs/2609.27166) | 本文利用层次贝叶斯模型量化了DeepSeek-R1-Distill系列模型在算术与算法推理任务上的表现，发现正确解题概率随问题难度近似指数衰减，而衰减尺度随模型规模增长，从而揭示了推理时能力与效率随模型规模的扩展规律。 |
| [^161] | [The Linear Representation Hypothesis Needs a Group Action](https://arxiv.org/abs/2609.27158) | 论文指出线性表示假说实际上是由表示等价性区分的一族假说，并提出用群作用将其形式化——明确表示对象、生成过程与所断言的性质——从而澄清不同度量、读取点和分析阶段之间假设的差异。 |
| [^162] | [Giving Credit Where It's Due: Redundancy-Aware Learning for Efficient Reasoning](https://arxiv.org/abs/2609.27156) | 提出RECAP方法，通过在LLM标注的语义依赖图上从最终答案节点反向传播信用，同时衡量步骤的结构责任与对解题的贡献，实现冗余感知的信用分配，从而在不牺牲准确性的前提下有效缩短大型推理模型的推理链。 |
| [^163] | [The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems](https://arxiv.org/abs/2609.27155) | 该研究通过理论分析揭示了社交媒体平台推荐系统中的点赞评分机制存在可利用的漏洞，攻击者可通过多阶段投毒帖子链，以隐蔽方式操纵部署在平台上的LLM智能体的信息流。 |
| [^164] | [Learning Risk Scores Robust to Unobserved Confounders](https://arxiv.org/abs/2609.27144) | 该论文针对历史分配决策中存在未观测混杂因素的场景，提出了对未观测混杂稳健的风险评分学习方法，以避免系统性地低估最需要稀缺资源的个体。 |
| [^165] | [PEARL: A Lightweight Prompt-based Feature Interpreter Framework for Real-Time, Anonymous, and Heterogeneous Collaborative Perception](https://arxiv.org/abs/2609.27123) | PEARL是一个轻量级基于提示的特征解释框架，通过两个并行训练的轻量级多尺度解释器，实现了无需邻居配置信息、支持实时部署、并能泛化到运行时新加入匿名代理的异构协同感知。 |
| [^166] | [Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges](https://arxiv.org/abs/2609.27110) | 由多个视觉语言模型组成的评审团在融合整体美学判定时无法显著超越最佳单模型，而让各模型按人工评分准则对图像进行五维度评分并融合这些分数，则能可靠地击败最佳单个模型。 |
| [^167] | [Intelligence Across Embodiments](https://arxiv.org/abs/2609.27095) | 该论文主张通用具身智能应依赖能够跨具身差异持续积累的学习，提出将具身多样性作为规模化的新维度，并结合广泛的习得先验，以取代依赖人工设计对应关系的短期方案。 |
| [^168] | [Local Evidence and Geometric Readout Repair in Trained GNNs](https://arxiv.org/abs/2609.27092) | 该研究通过精确质量线性规划和两种学习式后验修复方法（消息重加权与集合条件化logit平移），分离并纠正了训练后GNN节点分类错误中混合权重与logit集合定位两种成因，在八个数据集上将平均准确率从62.6%提升至65.3%，且证明logit平移贡献了绝大部分增益。 |
| [^169] | [Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving](https://arxiv.org/abs/2609.27085) | Crossflow针对P/D分离架构中预填充与解码需求剧烈波动（智能体负载下尤为突出）的问题，提出在不改变节点角色的前提下使预填充-解码边界弹性化，从而避免静态容量规划造成的容量闲置或排队吞吐损失。 |
| [^170] | [Quantifying the Occult: A Comparative Study of Hindu and Buddhist Deities Using Machine Learning Methods](https://arxiv.org/abs/2609.27074) | 本研究提出一种双矩阵计算架构，结合Gower距离矩阵与大语言模型语义嵌入，量化196位印度教与金刚乘佛教神祇的形态与神学差异，算法验证了“图像伪装”现象并计算建模了“Atin效应”。 |
| [^171] | [Does Graph Structure Earn Its Place in Microservice Root-Cause Analysis? A Controlled Study on RCAEval, and What the Benchmark Was Really Measuring](https://arxiv.org/abs/2609.27069) | 对照实验表明，在RCAEval基准上图神经网络相比传统扁平模型并无可靠优势，且该基准自身存在设计缺陷（故障仅注入五个服务、Avg@5指标过于宽松），使其无法真正衡量图结构在微服务根因分析中的价值。 |
| [^172] | [ChipMEM: Verification-Grounded Memory for EDA Agents](https://arxiv.org/abs/2609.27067) | ChipMEM提出了一个以验证结果为依据的EDA智能体记忆框架，只有在技能通过综合、仿真或形式化验证后才进行存储，并结合贝叶斯统计引导，从而避免模型自我评估偏差，生成可跨任务迁移的可复用知识而非仅针对特定任务的修补。 |
| [^173] | [EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues](https://arxiv.org/abs/2609.27043) | 提出了EduBehaviors框架，利用大语言模型测量教育对话中重复出现的可观察行为并据此学习分类器，实现了可解释、可审计的教育对话标注，其性能与直接提示方法相当。 |
| [^174] | [EMA: Elastic and Performance Transparent Memory Across GPUs](https://arxiv.org/abs/2609.27040) | EMA提出了一种服务器内跨GPU的弹性内存共享系统，通过预取技术为借用方隐藏远程访问开销、同时保证出借方的内存可按需回收，使双方性能均不低于静态分区。 |
| [^175] | [An open benchmark for machine learning-based polymer property prediction](https://arxiv.org/abs/2609.27036) | 该论文推出了开放基准数据集PolyBench26，包含近25万个涵盖八种物理性质的聚合物数据点，支持四项机器学习评估任务，并发现基于图的模型在聚合物性质预测中表现最佳。 |
| [^176] | [Reinforcement Learning with Decomposed Subtasks](https://arxiv.org/abs/2609.27035) | 该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。 |
| [^177] | [CVaR anchor regression protects against rare shifts](https://arxiv.org/abs/2609.27034) | CVaR锚点回归通过用尾部平均值代替跨环境平均残差平方的均值，能够针对罕见的大偏移提供精确的最坏情况风险保证，同时避免了传统方法因过度扩大保护范围而降低常见环境预测准确性的问题。 |
| [^178] | [WTF?! Simulation-Free Reinforcement Learning with Wasserstein-Tilted Flow Maps](https://arxiv.org/abs/2609.27033) | 提出WTF框架，通过基于预训练漂移构建的Wasserstein最优传输正则化器，将奖励微调问题等价转化为流上的确定性最优控制问题，实现了无需模拟的强化学习微调，是首个原生于流映射的端到端微调方案。 |
| [^179] | [LexLattice: Multilingual Extractive Summarization via Neural Cellular Automata on Document Hierarchies](https://arxiv.org/abs/2609.27032) | LexLattice将法律文档层次结构建模为二维语义格并通过神经元胞自动机整合跨远距离部分的证据，仅用180万参数的整合器就在24种语言上超越了数十亿参数的大模型，实现了最先进的多语言抽取式摘要性能。 |
| [^180] | [GeoRVQ: Decoder-aware geometry for residual-token prediction in physiological signals](https://arxiv.org/abs/2609.27018) | GeoRVQ提出了一种解码器感知的从粗到细掩码token建模方法，利用冻结波形解码器的局部响应定义几何感知的软目标与期望失真，显著提升了生理信号残差token预测的准确率、解码质量以及R波检测性能。 |
| [^181] | [Sharp Convergence of Wasserstein Gradient Flows for Spectrally Nonnegative Interaction Energies](https://arxiv.org/abs/2609.27008) | 本文证明，在闭流形上谱系数非负的相互作用能量，其 Wasserstein 梯度流即使不具测地凸性且不含扩散，能量间隙仍以 $o(t^{-1})$ 的锐速率衰减并时间可积，且当谱系数全为正时流弱收敛到常值测度。 |
| [^182] | [Resource-Efficient Distributed Recursive Gaussian Processes](https://arxiv.org/abs/2609.26979) | 本文针对多智能体系统的多输出高斯过程回归，提出了两种资源高效的分布式递归高斯过程算法（ADMM-RGP和PDMM-RGP），并通过稳定性与收敛性分析以及参数选择策略来加速收敛、降低通信负担。 |
| [^183] | [Tight Regret Bound for Online Inverse Linear Optimization via Multiscale Matrix Weights](https://arxiv.org/abs/2609.26978) | 该论文提出了一种基于多尺度矩阵乘法权重的随机化算法，实现了在线逆线性优化中O(√d)的期望遗憾界，达到了理论最优水平。 |
| [^184] | [TinyUDE: Solver-Free Universal Differential Equations on Microcontrollers via Lie-Taylor Jet Matching](https://arxiv.org/abs/2609.26972) | 该论文提出了一种名为李-泰勒射流匹配的无求解器训练框架，通过Savitzky-Golay滤波在线估计状态导数并直接拟合混合向量场，使通用微分方程能够在内存受限的微控制器上以解析梯度进行训练，其噪声自适应机制使精度可与传统基于求解器的方法相当甚至更优。 |
| [^185] | [CRISP: Scalable Importance-Stratified Coresets for Imbalanced Tabular Learning](https://arxiv.org/abs/2609.26962) | CRISP是一种线性时间的重要性分层核心集方法，通过将负类预算按代理模型得分的分位数层分配并进行逆倾向加权，在削减超过95%多数类样本的情况下仍保留约99.7%的全数据模型精度，大幅降低了不平衡表格数据的训练成本。 |
| [^186] | [Untangling the Geometry and Speed for RF Sensing Spectrograms](https://arxiv.org/abs/2609.26960) | 该论文提出一种物理可解释的射频感知新基础，通过紧凑参数化表示与物理信息自编码器（结合可微分射频前向模型），将多普勒频谱图中的目标速度与感知几何解耦，从而突破传统射频感知在真实无约束场景下的适用性限制。 |
| [^187] | [Transfer Learning with Conformalized Quantile Regression for Solar PV Forecasting Under Load-Shedding-Driven Data Scarcity](https://arxiv.org/abs/2609.26959) | 该论文提出一种结合共形化分位数回归（CQR）的迁移学习框架，在拉闸限电导致历史数据严重稀缺的地区将光伏预测RMSE最高降低23.7%，并提供覆盖率达94.3%且宽度减少14%的可靠不确定性预测区间。 |
| [^188] | [When Post-Processing Fairness Constraints Help and When They Harm: Evidence from Eight Cross-Domain Evaluations](https://arxiv.org/abs/2609.26955) | 该论文提出FAPE四阶段公平性审计框架，通过对八个领域的评估发现，后处理公平性干预（Fairlearn的ThresholdOptimizer）的有效性取决于基线差异大小——在多数高差异场景中能改善公平性，但在低差异场景中可能反而有害。 |
| [^189] | [Rolling Conformal Prediction in Sequential Model Training](https://arxiv.org/abs/2609.26951) | 本文提出滚动保形预测，一种无需数据划分的无分布预测推断方法，能够为序贯模型训练过程中的预测提供边际覆盖率保证。 |
| [^190] | [The Computational Value of Sensory-Aligned Receptive Fields Depends on Neuronal Expressivity](https://arxiv.org/abs/2609.26940) | 该研究发现，与任务相关感觉坐标对齐的感受野在资源预算匹配条件下比随机感受野能提升网络的分类准确率，且这一计算优势的体现依赖于单个神经元的表达复杂度。 |
| [^191] | [Experts Rise Where LLMs Disagree: Using Cross-Model Disagreement to Target Expert Effort in LLM Codebook Revision for Large-Scale Annotation](https://arxiv.org/abs/2609.26926) | 该论文提出利用多个大语言模型之间的分歧来定位最需要专家反馈的案例，并通过对比三种反馈方式发现，让专家对分歧案例进行附带理由的标注能最有效地指导LLM码本修订，使LLM标注准确率（64.9%）甚至超过专家手工修订的码本（57.8%）。 |
| [^192] | [On Preference Coverage Collapse from Hindsight Relabeling in Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2609.26918) | 该研究发现，在偏好条件化多目标强化学习中，用智能体实际实现的偏好方向进行事后重标注往往有害——它使36个算法-环境设置中的19个性能下降多达四个标准差，其根源是重复重标注导致的偏好覆盖坍缩，而非重标注噪声。 |
| [^193] | [Small Cues, Big Consequences: Learning Pivotal Cues for Multimodal Meme Classification](https://arxiv.org/abs/2609.26907) | 该论文提出了聚焦关键线索的MemeCF基准数据集（含9,895个迷因）和MemePIVOT局部-全局架构，通过非平衡最优传输对齐词语与图像块，并利用证据融合头在不确定性下融合局部与全局信息，从而有效捕捉迷因中有害、仇恨或讽刺含义的决定性线索。 |
| [^194] | [CORE-STACK+: Meta-Learning for Deep Stacked Generalization](https://arxiv.org/abs/2609.26905) | 提出 CORE-STACK+ 元学习框架，通过包含基于 CKA 的核化冗余过滤器等四个组件的预处理流水线，同时解决深度堆叠泛化中预测空间多重共线性和校准崩塌两大问题。 |
| [^195] | [PR-Smoother: Simulator-Preserving Non-Gaussian Smoothing for Data Assimilation](https://arxiv.org/abs/2609.26890) | PR-Smoother通过在证据下界和变分族中保留预设模拟器，仅学习围绕模拟轨迹的未来条件修正，实现了物理轨迹上的非高斯平滑分布，并支持仅从观测数据联合学习状态、参数与传感器偏差。 |
| [^196] | [Marginally Correct Tool Caches Can Reverse Group-Normalized Policy Updates](https://arxiv.org/abs/2609.26866) | 论文证明在边际奖励分布完全一致的条件下，组内共享单个随机工具结果缓存仍可能逆转组归一化策略更新的方向，而仅中心化、不做组标准差缩放的方法可保持期望回报的正确方向。 |
| [^197] | [Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness](https://arxiv.org/abs/2609.26865) | 该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。 |
| [^198] | [QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs](https://arxiv.org/abs/2609.26855) | 提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。 |
| [^199] | [COPE: Continual Personalization of LLMs under Sparse User Feedback via User Embeddings and Self-Evaluation](https://arxiv.org/abs/2609.26853) | COPE提出了一种在稀疏用户反馈下实现大语言模型持续个性化的优化框架，通过为每个用户分配可学习的个性化嵌入，并在单次更新步骤中协同完成偏好捕获、自我评估校准与个性化响应优化。 |
| [^200] | [A Leakage-Aware Multimodal Evaluation Framework for Early Intraoperative Acute Kidney Injury Prediction](https://arxiv.org/abs/2609.26848) | 该论文提出了仅基于生理波形的混合时序骨干网络SynerT及其多模态扩展SynerT-MM和防泄漏堆叠集成SynerTStack，并在VitalDB数据库上以严格的防泄漏评估框架实现了术中早期急性肾损伤风险预测。 |
| [^201] | [NeuroRule: Making Black-Box Neural Networks Explainable through Rule-set Evolution](https://arxiv.org/abs/2609.26841) | 本文提出NeuroRule知识蒸馏框架，通过规则集演化方法将黑盒神经网络蒸馏为简洁、可解释的命题逻辑规则集，从而缓解性能与可解释性之间的权衡。 |
| [^202] | [LWCal: Loss-Weighted Calibration for Tabular Classifiers with Noisy Calibration Labels](https://arxiv.org/abs/2609.26839) | 提出LWCal，一种无需干净验证标签、无需噪声率估计、也无需重训练的事后校准方法，通过对与基础模型预测相矛盾的噪声标签样本降权，有效应对校准标签含噪声的场景。 |
| [^203] | [What Makes a Terminal-Bench Task Hard? Separating Genuine Hardness from Fake-Hardness on an Adjudicated Agentic Corpus](https://arxiv.org/abs/2609.26826) | 本文提出一套有序的有效性筛选方法，综合任务工件、参考解运行、空解对照、对抗试验与遥测等多源证据，从 Terminal-Bench 的 125 个全失败任务中区分真实困难与虚假困难，发现其中仅 78 个可被认证为真正未解决的任务。 |
| [^204] | [HARN: Hierarchical Associative Resonance Network for Event-Driven Multi-Timeframe Forecasting](https://arxiv.org/abs/2609.26822) | 本文提出HARN分层联想共振网络，通过维护跨时间层级的持久表示并在时间条形完成时进行事件驱动的局部更新，结合因果多尺度编码、门控联想记忆与跨层级共振机制，在金融多时间框架预测中取得与强基线相当的性能。 |
| [^205] | [Signal2Symbol: Neuro-Symbolic Temporal Reasoning for Explainable Physiological Time-Series Anomaly Detection](https://arxiv.org/abs/2609.26820) | 提出了一种名为Signal2Symbol的神经符号框架，通过将ECG/EEG信号转换为符号序列并利用稀有项集挖掘对异常进行评分，实现了对生理时间序列的可解释异常检测，并能揭示局部异常之间的时间关联与重复模式。 |
| [^206] | [The Drift Contract: Spectral Updates for Depth-Robust Local Learning](https://arxiv.org/abs/2609.26811) | 该论文首次将Muon风格的谱更新（动量正交化加谱步长缩放）应用于逐层局部学习，仅用单一超参数设置即可在宽度128到2048、深度12到48范围内保持稳健性能并在深度48时不崩溃，显著优于需要反复调参的局部Adam。 |
| [^207] | [SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue](https://arxiv.org/abs/2609.26780) | 提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。 |
| [^208] | [On Basis Function Selection for Sparse Gaussian Process Regression](https://arxiv.org/abs/2609.26624) | 本文从信息论视角提出三种基函数选择准则，用于在稀疏高斯过程回归中依据数据挑选最相关的基函数，以替代传统的固定截断策略，从而更高效地利用有限的计算预算。 |
| [^209] | [GTR: Gated Token Recurrence for Efficient Dense Prediction](https://arxiv.org/abs/2609.26590) | GTR是一种无softmax的循环视觉骨干网络，通过门控线性注意力、交替空间扫描和空间增强SwiGLU模块，并从DINOv3教师模型进行最终层蒸馏，以低延迟在COCO检测上达到58.9 AP，并可广泛迁移至多种密集预测任务。 |
| [^210] | [TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling](https://arxiv.org/abs/2609.26347) | 提出仅使用合成翻译文本预训练语言模型的TransBERT框架，并证明仅凭合成翻译数据即可在法语生命科学领域的各类下游任务上达到最先进性能。 |
| [^211] | [Modular Norm RandOpt: Population-Efficient Ensembling through Architecture-Aware Perturbations](https://arxiv.org/abs/2609.25745) | 提出模块化范数RandOpt，利用模块级自然范数与校准尺度的架构感知权重扰动进行采样，在保持选择与投票机制不变的情况下，将所需候选模型数量在Countdown上减少3倍、在GSM8K上减少至少12倍，并在多个任务、多个模型家族和规模上取得更高的平均准确率。 |
| [^212] | [Dual-GNN Multilevel Coarsening for Maximum Independent Set](https://arxiv.org/abs/2609.25149) | 提出了一种基于学习的图边稀疏化方法GES，通过融合几何结构信息与组合优化技术为欧几里得TSP实例自适应生成稀疏图，在保持解与最优值差距1%以内的同时最多可剪枝95%的边并显著加速求解。 |
| [^213] | [From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness](https://arxiv.org/abs/2609.23065) | 该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。 |
| [^214] | [Silent Failures at the $2^{32}$ Boundary: A Technical Report on Large-Tensor Matrix Multiplication in PyTorch's Apple MPS Backend](https://arxiv.org/abs/2609.22991) | 该技术报告首次系统揭示了 PyTorch Apple MPS 后端在张量超过 2³² 元素边界时会静默返回严重错误的批量矩阵乘法结果（相对误差超过 1 且无任何警告），并通过大规模实验总结出三条可解释所有失败情况的规则。 |
| [^215] | [Testing the Construct Validity of a Functional Valence Axis in LLM Agents](https://arxiv.org/abs/2609.22850) | 该研究通过分离“结果本身”与“获知结果的信息历史”的受控干预，检验LLM智能体中“好—坏”效价方向的构念效度，发现该方向可跨表面形式迁移，但对结果是否被提前告知高度敏感，说明其效价表征与信息历史相互纠缠。 |
| [^216] | [SPIBER: Reconstructing Free Energy Landscapes from Short, Unconverged Trajectories with Generative Flow Networks](https://arxiv.org/abs/2609.22663) | SPIBER将状态预测信息瓶颈（SPIB）与生成流网络（GFlowNets）相结合，能够从短的、未收敛的分子模拟轨迹中识别慢集体变量并重建自由能景观，纠正不同亚稳态采样不平衡的问题。 |
| [^217] | [Locally Private Inference for Riemannian Stochastic Optimization](https://arxiv.org/abs/2609.22642) | 该论文提出了一种在局部差分隐私下对流形值总体极小值点进行统计推断的方法，通过条件中心化的随机切梯度保持一阶方程，并引入对称对回归（SPR）从相同私有消息中估计渐近方差，进而证明了中心极限定理及基于交互记录的三明治协方差和内在Wald区域的一致性。 |
| [^218] | [Common Cause, Not Cross-Attention: Blocking Visual Shortcuts in Audio-Video Generation](https://arxiv.org/abs/2609.22361) | 本文通过受控因果研究揭示，音视频联合生成模型中让音频直接读取视频（如交叉注意力）会学到基于外观而非因果事件预测声音的“视觉捷径”，且流行的共享共同因果潜变量方案也无法修复这一失效模式。 |
| [^219] | [Task-Aware Hybrid QUBO Optimization for Structured Neural Network Pruning](https://arxiv.org/abs/2609.22238) | 提出了一种任务感知的混合QUBO优化框架，将一阶泰勒敏感性和权重-费雪敏感性等任务信息与滤波器间的激活相似性交互相结合，并通过容量激励二分搜索控制剪枝基数，实现结构化神经网络滤波器剪枝。 |
| [^220] | [Riemannian Simultaneous Inference for Tangent Vector Field Regression](https://arxiv.org/abs/2609.21910) | 该论文针对无边黎曼流形上的切向量场回归提出了一种基于平行输运与体积校正的核估计方法，并通过单位切丛上的上确界表示与 Gumbel 极限理论，构造了回归场的可行同时置信管。 |
| [^221] | [An improved periodic activation for PINNs reconstructing convective flows](https://arxiv.org/abs/2609.21798) | 提出了一种基于复指数函数的周期激活架构，通过同时生成正弦和余弦输出，在瑞利-贝纳德对流的温度重构任务中以相近甚至更低的计算成本显著提升了物理信息神经网络的重建质量。 |
| [^222] | [Weighted Quantum Signal Processing: Low-Depth Polynomial Approximation with Applications to Kolmogorov-Arnold Networks](https://arxiv.org/abs/2609.21567) | 本文提出加权量子信号处理（WQSP），通过为中心旋转算子引入权重函数突破了QSP的电路深度瓶颈和奇偶性约束，在实现任意有界一元多项式时将所需参数数量从线性级降至指数级缩减，并展示了其在Kolmogorov-Arnold网络中的应用。 |
| [^223] | [SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190) | 该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。 |
| [^224] | [Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning](https://arxiv.org/abs/2609.19878) | Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。 |
| [^225] | [DeliveryGym: An RL Environment for Long-Horizon Embodied Agent Planning with Adaptive Curriculum](https://arxiv.org/abs/2609.19801) | DeliveryGym是一个面向长时程具身智能体规划的3D强化学习环境，通过连续快递员班次、持久世界动态和基于模拟器事件的轨迹奖励来评估决策成本，并采用自适应课程机制根据智能体观察到的弱点动态调整训练难度。 |
| [^226] | [Learn Your Own Thoughts: Abstract Token Curriculum](https://arxiv.org/abs/2609.19717) | 提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。 |
| [^227] | [QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training](https://arxiv.org/abs/2609.19513) | 提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。 |
| [^228] | [Agora: Git as Shared Memory for Collective AutoResearch](https://arxiv.org/abs/2609.18094) | Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。 |
| [^229] | [TabPFN-3.5: Technical Report](https://arxiv.org/abs/2609.17895) | TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。 |
| [^230] | [MyoFlow: Anchor-Tied Rectified Flow for HD-sEMG Gesture Recognition Across Sessions and Subjects](https://arxiv.org/abs/2609.17194) | 提出了首个面向跨会话与跨被试高密度表面肌电手势识别的判别式流匹配框架MyoFlow，通过域条件化校正流将分类重构为向手势锚点的输运过程，无需独立分类头即可实现零样本预测。 |
| [^231] | [Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories](https://arxiv.org/abs/2609.16827) | 本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。 |
| [^232] | [Schema-Adaptive Action-Conditioned JEPA for Cross-Machine CNC Transfer under Partial Sensor Overlap](https://arxiv.org/abs/2609.16071) | 该论文提出一种模式自适应的动作条件化JEPA架构，在源与目标CNC机床仅共享10/17个传感器通道的部分重叠情况下，通过严谨的密封目标测试协议实现零样本跨机床动力学预测迁移，将目标机器预测RMSE从0.813降至0.546。 |
| [^233] | [VertexCBF: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search](https://arxiv.org/abs/2609.12831) | 提出VertexCBF框架，利用控制仿射动力学和凸多面体控制集的性质（哈密顿量在控制顶点处最大化），结合物理信息学习、稀疏监督学习与GPU并行的顶点受限树搜索，以可扩展、系统化且可解释的方式训练神经控制屏障函数。 |
| [^234] | [VERPO: Verified Evidence Regularized Policy Optimization](https://arxiv.org/abs/2609.06100) | VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。 |
| [^235] | [PhenoBench: Mapping What a Deeply Phenotyped Human Cohort Can Tell Us](https://arxiv.org/abs/2609.06080) | 该论文提出了PhenoBench——一个基于人类表型项目（超13,000名参与者）构建的可执行评估基准，通过定义15个领域、26种输入模态下的90项临床任务，系统性地量化了哪些测量数据对哪些健康问题具有预测价值。 |
| [^236] | [Conditioning Degenerate Diffusion Models](https://arxiv.org/abs/2609.04090) | 该论文提出利用因果最优传输为扩散系数退化（奇异）的扩散生成模型构造近似损失函数，在条件密度不存在或不光滑的极弱假设下确定用于引导的最小熵控制。 |
| [^237] | [Learning Informative Prior with Infinite-Dimensional Continuous Normalizing Flow for Bayesian Inverse Problem](https://arxiv.org/abs/2609.03343) | 该论文提出了一种基于无限维连续归一化流的新方法，通过在希尔伯特空间中引入定义良好的神经常微分方程将简单参考测度变换为编码先验信息的复杂测度，建立了无限维贝叶斯先验的适定性理论框架，并提供了先验训练方法与后验采样算法，用于求解偏微分方程的贝叶斯逆问题。 |
| [^238] | [RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution](https://arxiv.org/abs/2609.02250) | 该论文提出RideSkill，一种由大语言模型驱动自动进化的分层算法，用于解决泛化拼车问题，克服了传统多智能体强化学习方法在泛化性、可迁移性和大规模训练方面的局限。 |
| [^239] | [Memory Is Not Always Needed: Characterizing Conditional Memory in Scientific Reasoning](https://arxiv.org/abs/2608.23982) | 本文系统研究了科学推理中条件记忆的适用条件，提出知识边界感知路由器，根据输入代理动态决定是否及如何激活记忆，以避免干扰并提升推理准确性。 |
| [^240] | [Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification](https://arxiv.org/abs/2608.14929) | 本文提出一种基于中心化残差签名的无数据白盒方法，通过移除身份对齐组件并比较残差块特有结构，实现语言模型血统的可靠验证，在多种后代类型中达到完美区分性能，且对功能保持清洗具有鲁棒性。 |
| [^241] | [A Parameter-Free Few-Shot Evaluation for Elephant Vocalisation Classification](https://arxiv.org/abs/2608.14824) | 本文提出了一种无参数的最近质心分类评估方法，用于在大象叫声分类中比较不同预训练嵌入的性能，无需额外训练即可进行小样本评估。 |
| [^242] | [Unifying Physical Backpropagation](https://arxiv.org/abs/2608.11585) | 本文提出了一种基于伴随方法的统一理论，确定了物理系统能在同一硬件上生成精确梯度所需的条件，区分了线性和非线性系统的不同要求。 |
| [^243] | [Federated Learning for Distributed CNC Tool Wear Prediction](https://arxiv.org/abs/2608.11281) | 本文提出将联邦学习应用于分布式CNC刀具磨损预测，在不共享原始数据的情况下实现接近集中式学习的性能，并显著优于本地模型。 |
| [^244] | [Output-Aware Rotation for INT2 KV-Cache Quantization](https://arxiv.org/abs/2608.02691) | 本文提出输出感知旋转方法OptR，通过最小化输出投影 $W_O$ 之后的注意力输出误差、将误差分解为键和值引起的项并学习逐头正交校正，同时利用注意力等价的键重参数化降低通道偏移，从而实现更优的INT2 KV缓存量化。 |
| [^245] | [Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.21653) | 提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。 |
| [^246] | [Predicting Activities in Aqueous Electrolyte Solutions with Hybrid Machine Learning](https://arxiv.org/abs/2607.19114) | 本文提出一种将基于物理的Bromley模型与机器学习矩阵补全方法相结合的混合模型，能够无需逐一拟合实验数据即可预测水电解质溶液的活度，突破了传统活度模型无法预测未研究体系的局限。 |
| [^247] | [Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection](https://arxiv.org/abs/2607.14236) | LIFT是一种力感知的后训练框架，通过在预训练VLA策略旁嫁接反应式动作专家，并借助因果力记忆和零初始化交叉注意力注入6D末端执行器力，为模型增加接触反应能力的同时保留其通用操作知识，从而加速VLA后训练。 |
| [^248] | [tidyHEBO: Robust General-Purpose Bayesian Optimization with Model-Consistent Warping and Pareto Search](https://arxiv.org/abs/2607.10669) | tidyHEBO 是一个 BoTorch 原生的通用贝叶斯优化器，通过将 Yeo-Johnson 输出变换与高斯过程代理模型联合拟合、在原始目标尺度上评估采集函数并执行多准则累积帕累托搜索，在无需任何基准特定调优的情况下于 Olympus 基准上排名第一。 |
| [^249] | [Energy-guided Recursive Model](https://arxiv.org/abs/2607.10128) | 提出能量引导递归模型（ERM），利用Hopfield型记忆为候选轨迹分配内在能量，从而有原则地指导轨迹选择与递归深度的确定，在数独、铅笔谜题和迷宫等推理任务上取得递归建模领域的最佳准确率，并降低了语言建模的困惑度。 |
| [^250] | [A rubric-based controlled comparison of frontier language models on expert-authored clinical reasoning tasks](https://arxiv.org/abs/2607.02175) | 该研究构建了一个由临床医生撰写的高难度临床推理评估数据集及加权量规，发现前沿大模型在关键临床标准上的通过率（32.4-41.7%）远低于低风险标准（80-90%），揭示了模型能力与临床优先级之间的倒置现象。 |
| [^251] | [Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits](https://arxiv.org/abs/2607.01940) | 提出条件性协同消融方法CoAx，通过测量主要组件集合被移除后消融效应的增长，来识别Transformer电路中被自修复机制掩盖的休眠备份组件，解决了电路解释在干预下不完整的问题。 |
| [^252] | [A 3D-Printable Dataset for Fair Testing and Comparisons of Tactile Sensors](https://arxiv.org/abs/2606.25886) | 本文提出了一个由数学定义的、可在不同打印机和耗材上可靠复制的3D可打印纹理数据集，为触觉传感器的公平测试与比较提供了标准化基准。 |
| [^253] | [The Chandra-Gaia Catalog of Counterparts: Resolving ambiguous Gaia matches to X-ray sources in the Chandra Source Catalog using Machine Learning](https://arxiv.org/abs/2606.19329) | 本文提出了一个结合贝叶斯交叉匹配框架NWAY与LightGBM梯度提升分类器的交叉匹配框架，利用星等、颜色、距离等源属性将钱德拉源目录中约25.4万个X射线源与Gaia DR3光学源匹配，成功识别约11.3万个真实对应体，并能有效排除基于纯空间位置方法无法区分的偶然重合。 |
| [^254] | [Starter-Iterator Neural Operator: A Unified Architecture for High-Fidelity Forward and Inverse PDE Problems](https://arxiv.org/abs/2606.18305) | 提出Starter-Iterator神经算子（SINO），将经典迭代求解器的初始化与残差修正结构融入神经算子学习，通过频域Starter模块捕捉全局谱特征并提供初始近似，为高保真的正演与反演PDE问题提供统一架构。 |
| [^255] | [On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents](https://arxiv.org/abs/2606.15912) | 提出Guided-OPD算法，通过在每次rollout中混合教师与学生生成的轮次，并按课程将教师干预概率逐渐衰减至零，解决了多轮智能体在线策略蒸馏中学生误差跨轮累积、教师监督在最需要时反而失效的问题。 |
| [^256] | [Simultaneous Latent Budget Trees for Stratified Classification](https://arxiv.org/abs/2606.13295) | 本文提出同时潜在预算树，一种面向含时间、空间或人口等分层因素的场景的分类树概率机器学习框架，通过将子节点解释为同时混合模型的潜在成分来构建基于模型的条件分裂规则。 |
| [^257] | [MLSkip: Data Skipping for ML Filters via Lightweight Metadata](https://arxiv.org/abs/2606.03946) | 本文提出MLSkip方法，首次证明Parquet默认的min-max元数据即可实现对机器学习过滤器的数据跳过修剪，并在TPC-H/TPC-DS基准上对低选择性过滤器实现了27.4%的平均修剪有效性。 |
| [^258] | [A lift for input-convex neural net training](https://arxiv.org/abs/2605.24274) | 针对输入凸神经网络训练中softplus参数化导致负权重区域梯度指数衰减、逃逸缓慢的问题，提出用可学习松弛量加无约束网络（以批次置换不变摘要为输入）替换自由潜在权重的“提升”方法。 |
| [^259] | [Helping Customers in Distress: An LLM-powered Agent that Converses, Probes, and Routes](https://arxiv.org/abs/2605.16268) | 本文开发了一个基于大语言模型的银行客户分流智能体，通过多轮对话探询客户问题并按政策精准分流至专业团队，同时利用真实客户的合成数字孪生生成带标签对话来评估和持续改进该系统。 |
| [^260] | [Judge Circuits Explain Format-Induced Inconsistency in LLM-as-a-Judge](https://arxiv.org/abs/2605.16023) | 该论文通过PEAP方法发现LLM裁判模型的中后层MLP中存在一个稀疏的“潜在评估者”子图，该子图负责抽象评判且独立于输出格式，从而在机制层面解释了LLM-as-a-Judge中格式诱导的评分不一致现象。 |
| [^261] | [SMT-Based Active Learning of Weighted Automata](https://arxiv.org/abs/2605.07758) | 提出了一种基于SMT的半环参数化加权自动机主动学习算法，在终止时保证产生最小自动机，实验表明其大幅超越朴素基线并与最先进算法竞争，同时生成显著更小的自动机且需要更少的教师交互。 |
| [^262] | [ProteinJEPA: Latent prediction improves protein language model pretraining](https://arxiv.org/abs/2605.07554) | ProteinJEPA在蛋白质语言模型的掩码语言建模基础上引入JEPA式潜在表示预测损失，显著提升了模型在蛋白质检索和远程同源性检测等结构与同源性敏感任务上的表现，且增益随模型规模增大而增强。 |
| [^263] | [QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature](https://arxiv.org/abs/2605.05870) | 本文提出QuadraSHAP，证明乘积博弈中每个玩家的Shapley值可精确表示为一维积分，从而利用高斯-勒让德求积以仅需⌈d/2⌉个节点即可实现可证明精确、稳定且可扩展的高效计算。 |
| [^264] | [ANO: Robust Policy Optimization via Bounded, Redescending Gain Fields](https://arxiv.org/abs/2605.02320) | 提出锚定邻域优化（ANO），通过C^∞光滑的整形核直接构造有界且再下降的增益场，在PPO的死区漂移与SPO的无界增益这两个极端之间取得平衡，从而实现更稳定、更鲁棒的策略优化。 |
| [^265] | [Anon: Extrapolating Adaptivity Beyond SGD and Adam](https://arxiv.org/abs/2605.02317) | 该论文提出Anon优化器，突破了SGD与Adam之间0到1的插值限制，首次实现在整个实数范围内连续外推自适应参数（如CNN需要负自适应性、Transformer需要γ≥1），并通过增量延迟更新机制保证超界情形下的稳定收敛。 |
| [^266] | [Fitting Large Nonlinear Mixed Effects Models Using Variational Expectation Maximization](https://arxiv.org/abs/2604.26160) | 本文提出利用变分期望最大化（VEM）算法，结合灵活的变分族和反向模式自动微分技术，高效拟合大型非线性混合效应模型，可扩展至超过15,000个群体参数的规模。 |
| [^267] | [Assessing the impact of dimensionality reduction on clustering performance - a systematic study](https://arxiv.org/abs/2604.22099) | 本研究系统评估了五种降维方法（PCA、核PCA、VAE、Isomap、MDS）对四种常用聚类算法（k-means、AHC、GMM、OPTICS）性能的影响，并使用调整兰德指数（ARI）在不同降维水平下进行全面比较。 |
| [^268] | [PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving](https://arxiv.org/abs/2604.12171) | 提出PipeLive系统，实现了在不中断推理的情况下对大语言模型流水线并行配置进行在线原位重配置，解决了KV缓存空间受限与执行期间KV一致性维护的双重挑战，使流水线并行能够适应无服务器平台和异构GPU等动态服务环境。 |
| [^269] | [Joint Interference Detection and Identification via Adversarial Multi-task Learning](https://arxiv.org/abs/2604.08607) | 该论文建立了一个有理论支撑的多任务学习框架，通过推导加权期望损失上界，将任务相似度与Wasserstein距离和可学习的任务关系系数联系起来，并据此提出对抗式多任务网络，实现干扰检测、调制识别和干扰识别的联合处理。 |
| [^270] | [TiAb Review Plugin: A Browser-Based Tool for AI-Assisted Study Selection in Systematic Reviews](https://arxiv.org/abs/2604.08602) | 该论文开发了开源 Chrome 扩展 TiAb Review 插件，无需编程和服务器即可利用 AI 完成系统综述中从标题摘要到全文的文献筛选。 |
| [^271] | [Value Mirror Descent for Reinforcement Learning](https://arxiv.org/abs/2604.06039) | 本文提出价值镜像下降（VMD）方法，将凸优化中的镜像下降融入经典价值迭代框架，在确定性设定下实现线性收敛，并针对生成式模型下的随机设定开发了结合方差缩减技术的 SVMD 变体。 |
| [^272] | [Safe learning-based control via function-based uncertainty quantification](https://arxiv.org/abs/2604.01173) | 该论文提出将未知函数建模为可生成独立同分布样本的随机函数，并利用场景方法仅基于采样数据构建高概率成立的不确定性管道，从而摆脱了传统方法对函数光滑性的限制性假设并能处理不连续性，实现安全的基于学习的控制。 |
| [^273] | [Softmax gradient policy for variance minimization and risk-averse multi armed bandits](https://arxiv.org/abs/2604.00241) | 该论文提出了一种基于softmax参数化的新算法，用于在风险规避的多臂老虎机问题中选择方差最小（风险最低）的臂，通过两次独立抽样构建无偏估计并证明了算法的收敛性。 |
| [^274] | [Learning to Remember: Attentive Reinforcement Learning for Edge Serverless Autoscaling](https://arxiv.org/abs/2603.28790) | 该论文提出了一种将注意力增强的双层堆叠LSTM集成到PPO智能体中的稳定性感知自动伸缩框架，通过利用近期时间上下文克服边缘无服务器环境中传统控制器的反应延迟和DRL智能体的时间盲区问题。 |
| [^275] | [PHONOS: PHOnetic Neutralization for Online Streaming Applications](https://arxiv.org/abs/2603.27001) | 提出了PHONOS——一个面向实时说话人匿名化的流式口音中和模块，通过静音感知DTW对齐、零样本语音转换和仅40毫秒前瞻的因果口音翻译器，将非母语音段转换为目标口音，使非母语口音线索减少81%。 |
| [^276] | [A Foundation Model for Instruction-Conditioned In-Context Time Series Tasks](https://arxiv.org/abs/2603.22586) | 提出iAmTime——一个通过指令条件化摊销元学习训练的时间序列基础模型，能够利用专门的语义标记从显式的输入-输出示例演示中直接推断任务，实现时间序列任务的上下文学习。 |
| [^277] | [Binary Classification from Coupled Pairwise Labels](https://arxiv.org/abs/2603.19713) | 提出SD-Pcomp学习方法，通过一个同时保留相似/不相似结构与成对比较排序结构的目标函数，实现了仅利用耦合成对关系信息（而非绝对类别标签）的二元分类。 |
| [^278] | [An Adaptive Machine Learning Framework for Fluid Flow in Dual-Network Porous Media](https://arxiv.org/abs/2603.19561) | 该论文提出了一种具有自适应权重调节的物理信息神经网络（PINN）框架，用于双重孔隙度/渗透率多孔介质系统的正向与反向建模，实现了快速预测和可靠的反演分析。 |
| [^279] | [The Truncation Blind Spot: How Decoding Strategies Systematically Exclude Human-Like Token Choices](https://arxiv.org/abs/2603.18482) | 该论文提出“截断盲区”概念，揭示 top-k 和核采样等解码策略因截断低概率词元而系统性地排除了 8–18% 的人类典型选词，从而为机器生成文本为何始终可被检测提供了机制性解释。 |
| [^280] | [Regular Fourier Features for Nonstationary Gaussian Processes](https://arxiv.org/abs/2602.23006) | 该论文提出正则傅里叶特征方法，通过直接离散化可调和非平稳高斯过程的谱表示，摆脱了谱密度必须为概率测度的限制性假设，实现了无需概率假设、结构上半正定的高效低秩近似。 |
| [^281] | [A Very Big Video Reasoning Suite](https://arxiv.org/abs/2602.20159) | 本文介绍了VBVR数据集和VBVR-Bench评估框架，前者规模比现有数据集大三个数量级，后者采用基于规则且与人类对齐的评分器，以系统研究视频推理能力及其扩展行为。 |
| [^282] | [LORA-CRAFT: Cross-layer Rank Adaptation via Frozen Tucker Decomposition of Pre-trained Attention Weights](https://arxiv.org/abs/2602.17510) | CRAFT通过将预训练注意力权重组织为跨层3D张量并应用冻结的塔克分解，仅训练小型方形矩阵，实现了比现有方法更参数高效的微调。 |
| [^283] | [Learning to Approximate Uniform Facility Location via Graph Neural Networks](https://arxiv.org/abs/2602.13155) | 提出了一种融合近似算法原理的完全可微分消息传递神经网络，用于求解均匀设施选址问题，既具有可证明的近似保证，又在实证中优于标准近似算法并缩小了与整数线性规划的差距。 |
| [^284] | [Semantic Self-Distillation for Language Model Uncertainty](https://arxiv.org/abs/2602.04577) | 该论文提出语义自蒸馏方法，将语言模型采样答案的语义分布蒸馏到轻量级学生模型中，使其能在生成答案前预测语义分布，利用分布的熵和概率密度分别提供提示级和答案级的不确定性信号，从而以低计算成本实现高效的不确定性估计与幻觉检测。 |
| [^285] | [Variational Bayesian Flow Network for Graph Generation](https://arxiv.org/abs/2601.22524) | 提出变分贝叶斯流网络（VBFN），通过将贝叶斯更新提升为由结构化精度控制的联合高斯变分信念族，使生成几何能够显式编码节点-边耦合关系，从而提升离散图生成的鲁棒性。 |
| [^286] | [Inverse Problems Conditioned on Observation Ensembles: Applications and Methods](https://arxiv.org/abs/2601.22029) | 该论文提出了一类新的统计问题——集合条件逆问题（EIP），并基于一种利用观测集合信息的新型条件生成模型（集合逆生成模型），给出了非迭代的推理时后验采样方法，可应用于高能物理解折叠、全波形反演和逆成像等领域。 |
| [^287] | [Advances in Diffusion-Based Generative Compression](https://arxiv.org/abs/2601.18932) | 本文系统综述了基于扩散模型的生成式有损压缩最新进展，重点介绍了图像压缩中通过嵌入表示编码并利用扩散模型迭代细化、从而在极低码率下实现逼真重建的方法。 |
| [^288] | [Self-Improvement as Coherence Optimization: A Theoretical Account](https://arxiv.org/abs/2601.13566) | 该论文提出统一理论框架，证明辩论、自举与内部一致性最大化等无监督自我提升方法本质上都是“一致性优化”，等价于描述长度正则化，其中基于预训练先验的一致性正则化可优化半监督学习最坏情况准确率的下界，从而在理论上解释了无需反馈的自我提升为何有效。 |
| [^289] | [GlyRAG: Context-Aware Retrieval-Augmented Framework for Blood Glucose Forecasting](https://arxiv.org/abs/2601.05353) | 提出GlyRAG框架，利用大语言模型作为情境化智能体从CGM数据中提取血糖形态的情境信息，并通过检索增强机制融合相似历史片段，从而提升血糖预测的准确性。 |
| [^290] | [ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](https://arxiv.org/abs/2512.04125) | 该论文提出ASCIIBench——首个公开可用的ASCII艺术评测基准，包含5,315张带标签的ASCII图像数据集和一个微调的CLIP模型，用以揭示大语言模型在空间位置推理方面的局限。 |
| [^291] | [Parameter-Efficient Construction of the Rashomon Slice for Concept Bottleneck Models](https://arxiv.org/abs/2511.19636) | 该论文提出了一种参数高效的方法，通过并行适配模块、检查点机制和概念多样性目标，高效探索概念瓶颈模型（CBM）的Rashomon集合，从而以较低成本生成多个精度相当但内部逻辑不同的模型。 |
| [^292] | [Fine-Tune, Then Rectify](https://arxiv.org/abs/2511.19486) | 该论文提出一个结合微调与校正的两阶段LLM框架，指出传统微调目标（最小化均方误差）与下游校正阶段不匹配，并创新性地提出以最小化预测误差方差（或标量化方差指标）作为微调目标，同时在两阶段间最优分配有限的标注样本。 |
| [^293] | [Parameter Importance-Driven Continual Learning for Foundation Models](https://arxiv.org/abs/2511.15375) | 提出了一种基于参数重要性估计的持续增强方法PIECE，使基础模型无需访问历史训练数据即可在高效学习领域知识的同时保持通用推理能力。 |
| [^294] | [Optimization without Future Compromises? Decentralized Coordination via Collective and Reinforcement Learning](https://arxiv.org/abs/2509.18088) | 提出分层强化与集体学习（HRCL）框架，用MARL从全局层面引导而非取代去中心化多智能体协调，在大规模长期资源分配中兼顾当前效率与未来性能，同时避免决策空间爆炸和训练低效。 |
| [^295] | [Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs](https://arxiv.org/abs/2509.13813) | 该论文提出了一个黑盒几何框架，通过在答案嵌入空间中建模以提示为条件的语义分布，同时量化提示和答案两个层面的不确定性，从而实现对大语言模型幻觉的检测与纠正。 |
| [^296] | [A Discrepancy-Based Perspective on Dataset Condensation](https://arxiv.org/abs/2509.10367) | 本文提出一个基于差异度量的统一框架，将数据集浓缩重新形式化为概率分布逼近问题，从而把 DC 的目标从泛化性能扩展到更一般的任务设定。 |
| [^297] | [Integrated Multivariate Segmentation Tree for Heterogeneous Credit Data Analysis in Small- and Medium-Sized Enterprises](https://arxiv.org/abs/2509.00550) | 本文提出集成多元分割树（IMST）框架，通过矩阵分解、Lasso特征选择与多元分割树构建，将财务数据与文本信息有效融合，将中小企业信用评估准确率提升至88.9%。 |
| [^298] | [VMMU: A Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark](https://arxiv.org/abs/2508.13680) | VMMU是首个越南语多任务多模态理解与推理基准，包含2500个跨7个任务的多模态问题，评估显示尽管最先进专有视觉-语言模型的越南语OCR性能良好，其平均准确率仅达66%，主要瓶颈在于多模态定位与推理能力而非OCR。 |
| [^299] | [Honest and Reliable Evaluation and Expert Equivalence Testing of Automated Neonatal Seizure Detection](https://arxiv.org/abs/2508.04899) | 本研究系统评估了新生儿癫痫发作检测的性能指标与共识策略，发现Matthews和Pearson相关系数在类别不平衡下优于AUC，并提出了严格的专家等同性测试方法，为AI达到专家水平的声明建立了诚实可靠的评估框架。 |
| [^300] | [InsurTech innovation using natural language processing](https://arxiv.org/abs/2507.21112) | 本文展示了如何运用自然语言处理技术将非结构化文本转化为结构化数据，通过特征去偏、特征压缩和行业分类来丰富商业保险定价的费率因子，并为评估潜在风险提供新视角。 |
| [^301] | [Faithful, Interpretable Chest X-ray Diagnosis with Artifact-free B-cos Networks](https://arxiv.org/abs/2507.16761) | 该论文通过在B-cos网络中引入ASAP和BlurPool抗混叠策略，消除了胸部X光诊断解释图中的混叠伪影，在保持强大诊断性能的同时实现了忠实、清晰、可解释的临床级诊断。 |
| [^302] | [AdaDim: Dimensionality Adaptation for SSL Representational Dynamics](https://arxiv.org/abs/2505.12576) | 该论文提出 AdaDim 方法，在自监督学习训练过程中自适应地调控表示的维度动态，兼顾高有效维度 H(R) 与低互信息 I(R;Z)，以防止维度坍缩并提升下游任务的泛化性能。 |
| [^303] | [ChronoSteer: Bridging Large Language Model and Time Series Foundation Model via Synthetic Cross-Modal Alignment Dataset](https://arxiv.org/abs/2505.10083) | 提出ChronoSteer——一个解耦的智能体框架，通过合成跨模态对齐数据集将大语言模型与时间序列基础模型相连接，构建出能联合利用时间与文本信息进行零样本预测的多模态时间序列基础模型。 |
| [^304] | [Role of scrambling and noise in temporal information processing with quantum systems](https://arxiv.org/abs/2505.10080) | 本文揭示了基于高阶幺正设计的加扰量子储层在时间信息处理中的关键特性：无噪声时测量读出集中度不随迭代恶化、小储层可反复复用，但扩大规模需指数级测量开销否则损害泛化，且早期输入记忆随储层规模与迭代次数均呈指数衰减。 |
| [^305] | [Localized Diffusion Models](https://arxiv.org/abs/2505.04417) | 提出局部化扩散模型，通过利用目标分布中的局部性结构（稀疏条件依赖），以局部化神经网络估计得分函数，从而规避维数灾难并显著降低样本复杂度。 |
| [^306] | [Path Regularization: A Near-Complete and Optimal Nonasymptotic Generalization Theory for Multilayer Neural Networks and Double Descent Phenomenon](https://arxiv.org/abs/2503.02129) | 该论文首次提出了路径正则化多层神经网络的近乎完备且最优的非渐近泛化理论，给出了显式泛化误差上界，无需损失函数有界及网络宽度、深度等常见假设，超越了偏差-方差权衡并能解释深度学习中的双重下降现象。 |
| [^307] | [Statistical Properties of Deep Neural Networks with Dependent Data](https://arxiv.org/abs/2410.11113) | 该论文为非平稳β-混合依赖数据下的深度神经网络估计量建立了非渐近误差理论，覆盖全连接与卷积网络且无需对权重施加有界或稀疏约束，并推广至非参数回归、逻辑回归和分位数回归等场景。 |
| [^308] | [FastManly: An EM-Gradient Algorithm for Manly Mixture Models](https://arxiv.org/abs/2410.00848) | FastManly方法通过在EM梯度算法中采用牛顿法（并推导出梯度和完整Hessian矩阵）替代传统EM中的Nelder-Mead优化，显著加快了Manly变换混合模型的计算速度。 |
| [^309] | [Variance Reduction for Independent Metropolis](https://arxiv.org/abs/2406.17699) | 本文证明当目标密度与提议密度在KL散度下足够接近时，独立Metropolis采样器结合基于控制变量、无需额外计算成本的方差缩减策略，可获得比独立同分布采样更小的渐近方差。 |
| [^310] | [Random Polytope Descriptors](https://arxiv.org/abs/2009.13987) | 该论文提出了一类既通用又计算友好的随机多面体描述符，可用于数据分析中的分类与聚类任务，并允许用户在数据描述的紧致性与计算速度之间灵活权衡。 |

# 详细

[^1]: 论高维隐空间的扩散可行性

    On the Diffusibility of High-Dimensional Latents

    [https://arxiv.org/abs/2609.28473](https://arxiv.org/abs/2609.28473)

    本文揭示了在高维隐空间中使用流匹配标准速度预测会导致模型需拟合信号流形之外的正交噪声方向、优化低效的问题，并提出改用干净数据参数化（$x_0$-预测）使学习聚焦于低维信号流形，从而提升扩散模型的生成效率。

    

    表征自编码器（RAE）使扩散模型能够在预训练视觉编码器的特征空间中运行。然而，许多现成的编码器并未针对忠实重建进行优化，会丢失细粒度的视觉细节。正如预期的那样，针对图像重建对这些编码器进行微调可以恢复这些细节。然而，也许有些反直觉的是，这一过程降低了所得到的表征的有效维度，且改变后的几何结构会对生成产生下游影响。具体而言，我们证明在这种高维空间中使用流匹配的标准速度预测，需要模型去拟合低维信号流形之外的正交噪声方向，导致优化效率低下。这促使我们改用干净数据参数化（$x_0$-预测），它将学习聚焦于底层的信号流形。在多项实验中……

    arXiv:2609.28473v1 Announce Type: cross  Abstract: Representation Autoencoders (RAEs) enable diffusion models to operate in the feature spaces of pretrained visual encoders. However, many off-the-shelf encoders are not optimized for faithful reconstruction, discarding fine-grained visual details. As expected, finetuning these encoders for image reconstruction recovers such details. However, perhaps counterintuitively, this procedure reduces the effective dimensionality of the resulting representation, and the altered geometry has downstream effects on generation. Specifically, we show that using the standard velocity prediction in flow matching in this high-dimensional space requires the model to fit orthogonal noise directions outside the low-dimensional signal manifold, making optimization inefficient. This motivates using the clean data parameterization ($\boldsymbol{x}_{0}$-prediction) instead, which focuses learning on the underlying signal manifold. Across experiments with multip
    
[^2]: 用于作者身份验证的对比学习

    Contrastive Learning for Authorship Verification

    [https://arxiv.org/abs/2609.28471](https://arxiv.org/abs/2609.28471)

    本文提出基于 ModernBERT 双编码器的对比学习方法，通过优化损失函数、数据增强等关键因素，在 PAN21 作者身份验证任务上达到 98.4% 的准确率，优于基于分类的方法。

    

    我们的结果表明，在所测试的设置下，对比学习优于基于分类的作者身份验证方法。我们确定了损失函数、批大小、训练时长、预训练模型、输入上下文长度以及随机文本片段数据增强是影响模型性能的重要因素。基于这些考虑，我们开发了一个 ModernBERT 双编码器模型，在 PAN21 作者身份验证任务上达到了 98.4% 的准确率。

    arXiv:2609.28471v1 Announce Type: new  Abstract: Our results show that contrastive learning outperforms a classification-based approach to authorship verification under the tested settings. We identify loss function, batch size, training duration, pre-trained model, input context length, and random text span data augmentation as important factors of model performance. Based on these considerations, we develop a ModernBERT Bi-Encoder model that achieves 98.4% accuracy on the PAN21 authorship verification task.
    
[^3]: 直推学习更锐利的界及其应用

    Even Sharper Bounds for Transductive Learning and Its Applications

    [https://arxiv.org/abs/2609.28459](https://arxiv.org/abs/2609.28459)

    本文提出直推学习的新型局部化复杂度方法STLC，去除了以往直推学习中多余的对数置信度因子，并在可实现设定下达到了与标准归纳学习相同的 $\cO\{\dVC\log(me/\dVC)/m\}$ 速率。

    

    我们提出了更锐利的直推局部复杂度，这是一种针对无放回均匀采样下直推学习的局部化复杂度方法。该构造始于一个关于测试-训练经验过程上确界的Bernstein型集中不等式，其证明利用了交换随机游走的修正对数Sobolev不等式和双参数熵闭包。随后，通过替代局部化泛函的剥离论证，我们得到了与经典归纳局部Rademacher复杂度界具有相同不动点和置信度项的过剩风险界，且去除了早期直推结果中额外的对数置信度因子。对于VC维为 $\dVC$ 的二值函数类上的可实现学习，当训练集大小为 $m$、测试集大小为 $u$ 且 $u\ge m\ge\dVC$ 时，STLC 给出了 $\cO\{\dVC\log(me/\dVC)/m\}$ 的界。这一结果匹配了标准的归纳学习速率，并且当 $m\ge9$ 时，与直推…（摘要原文在此处截断）相比仅相差一个对数因子。

    arXiv:2609.28459v1 Announce Type: new  Abstract: We introduce Sharper Transductive Local Complexity (STLC), a localized complexity method for transductive learning under uniform sampling without replacement. The construction starts from a Bernstein-type concentration inequality for the supremum of the test--train empirical process. Its proof uses the modified log-Sobolev inequality for the swap walk and a two-parameter entropy closure. A peeling argument with a surrogate localization functional then gives excess-risk bounds with the same fixed-point and confidence terms as the classical inductive local Rademacher-complexity bounds, without the additional logarithmic confidence factor in earlier transductive results. For realizable learning over a binary class of VC dimension $\dVC$, with training size $m$, test size $u$, and $u\ge m\ge\dVC$, STLC yields $\cO\{\dVC\log(me/\dVC)/m\}$. This matches the standard inductive rate and, when $m\ge9$, is within a logarithmic factor of the transd
    
[^4]: 排斥性自注意力的非平衡相：混沌、注意力凝聚与涌现局域性

    Nonequilibrium Phases of Repulsive Self-Attention: Chaos, Attention Condensation, and Emergent Locality

    [https://arxiv.org/abs/2609.28448](https://arxiv.org/abs/2609.28448)

    该论文通过分析具有负值映射的最小循环transformer模型，揭示了排斥性自注意力系统中丰富的非平衡相行为，包括通过倍周期分岔产生的混沌动力学、在 $\beta\sim N^2$ 标度下出现的注意力凝聚现象以及硬路由极限下的涌现局域性。

    

    我们研究了一个最小化循环transformer的非平衡动力学，该模型具有 $N$ 个归一化 token、$Q=K=I$ 以及负值映射 $V=-I$。基于相似度的注意力会选择邻近的表示，而负值映射则驱使 token 远离被选中的区域。这种反馈可以持续地重组表示几何结构与注意力网络。当 $d=2$ 时，token 位于一个圆上，其中正多边形构成一个精确的不动点。随着注意力反馈强度 $\gamma$ 的增大，正多边形通过倍周期分岔失去稳定性，产生周期二运动、混沌以及簇交换或簇翻转状态。尽管存在这种时间上的复杂性，在有限且固定的 softmax 锐度 $\beta$ 下，当 $N\to\infty$ 时注意力仍然保持弥散状态。注意力凝聚则出现在 $\beta\sim N^2$ 的标度区间中。在硬路由极限下，排斥性更新会放大局部扰动并使路由暂停

    arXiv:2609.28448v1 Announce Type: cross  Abstract: We study the nonequilibrium dynamics of a minimal recurrent transformer with $N$ normalized tokens, $Q=K=I$, and a negative value map $V=-I$. Similarity-based attention selects nearby representations, while the negative value map drives tokens away from the selected field. This feedback can continually reorganize both the representation geometry and the attention network. For $d=2$, the tokens lie on a circle, where the regular polygon is an exact fixed point. As the attention feedback strength $\gamma$ is increased, the polygon loses stability through a flip bifurcation, giving rise to period-two motion, chaos, and cluster-exchange or cluster-flip states. Despite this temporal complexity, attention remains diffuse as $N\to\infty$ at finite fixed softmax sharpness $\beta$. Attention condensation instead emerges in the scaling regime $\beta\sim N^2$. In the hard-routing limit, repulsive updates amplify local perturbations and routing-pa
    
[^5]: 数学推理中的答案顺序不变性与表征顺序敏感性

    Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning

    [https://arxiv.org/abs/2609.28442](https://arxiv.org/abs/2609.28442)

    该研究发现，语言模型对规则排序的内部表征越清晰（排列信噪比越高），其解决重排序数学问题的准确率就越高，揭示了答案不变性与表征不变性是两个不同的概念。

    

    在不改变含义的情况下重新排列一组数学规则的顺序，应当保持正确答案不变，但模型的内部表征是否也必须保持不变呢？我们使用合成的多步骤函数组合问题来研究这一问题，每个问题以多种规则排序呈现，且具有相同的正确答案。我们测量了准确率和排列信噪比（SNR），后者量化了排序模式相对于问题实例间差异的表征清晰程度。在16个参数量从1B到8B的语言模型上，我们发现了一个规律：更准确地解决重排序问题的模型，对不同的规则排序表征得也更加清晰。在我们评估的所有合成设置中，层级平均排列信噪比与准确率呈正秩相关，Spearman相关系数最高达到0.86。这些发现突出了答案不变性与表征不变性之间的区别：（摘要在此处截断）

    arXiv:2609.28442v1 Announce Type: cross  Abstract: Reordering a set of mathematical rules without changing its meaning should preserve the correct answer, but must a model's internal representations stay invariant too? We investigate this question using synthetic multi-step function-composition problems, each presented under multiple rule orderings with the same correct answer. We measure accuracy and permutation signal-to-noise ratio (SNR), which quantifies how distinctly ordering patterns are represented relative to variation across problem instances. Across 16 language models ranging from 1B to 8B parameters, we find a pattern: models that solve reordered problems more accurately represent different rule orderings more distinctly. Layer-averaged permutation SNR is positively rank-correlated with accuracy in every synthetic setting we evaluate, with Spearman correlations reaching 0.86. These findings highlight a distinction between answer invariance and representation invariance: suc
    
[^6]: 最小范数单变量两层ReLU分类：精确解与带跳跃连接的全局最优性

    Minimal-Norm Univariate Two-Layer ReLU Classification: Exact Solutions and Global Optimality with Skip Connections

    [https://arxiv.org/abs/2609.28438](https://arxiv.org/abs/2609.28438)

    该论文完整刻画了单变量两层ReLU网络二元分类最优解在函数空间中的几何结构，并证明添加仿射跳跃连接后每个KKT点都成为全局最优点。

    

    我们研究了使用单变量两层ReLU网络进行二元分类的最小范数插值和ℓ2正则化逻辑损失最小化问题。我们给出了最优分类器在函数空间中的完整几何刻画，并解决了这些解如何依赖于隐藏层偏置是否被纳入参数范数的问题。当偏置不受惩罚时，最小范数插值器恰好是那些紧贴每个标签切换点且具有适当凸性拐点的连续分段仿射函数。当偏置受到惩罚时，最小化器在函数空间中是唯一的，在每个中间同标签段内恰好有一个拐点，因此是最稀疏的正间隔分类器。我们进一步证明，添加一个自由的仿射跳跃连接不会改变这些函数空间解，但会从根本上改善参数空间的优化景观：约束问题的每个KKT点都成为全局最优解。

    arXiv:2609.28438v1 Announce Type: new  Abstract: We study minimal-norm interpolation and $\ell_2$-regularized logistic-loss minimization for binary classification by univariate two-layer ReLU networks. We give complete geometric characterizations of the optimal classifiers in function space, resolving how the solutions depend on whether hidden-layer biases are included in the parameter norm. When biases are unpenalized, the minimal-norm interpolators are exactly the continuous piecewise-affine functions that hug every label switch and have kinks of the appropriate convexity. When biases are penalized, the minimizer is unique in function space, has exactly one kink in each intermediate same-label segment, and is therefore a sparsest positive-margin classifier. We further show that adding a free affine skip connection leaves these function-space solutions unchanged but fundamentally improves the parameter-space landscape: every KKT point of the constrained problem becomes globally optima
    
[^7]: 面向外骨骼个性化的情境连续偏好学习

    Context-Continuous Preference Learning for Exoskeleton Personalization

    [https://arxiv.org/abs/2609.28427](https://arxiv.org/abs/2609.28427)

    本文提出情境连续偏好学习方法（CCPL），利用高斯过程在相邻运行条件之间共享偏好观测数据，从而以更少的用户反馈实现外骨骼辅助的个性化优化。

    

    在不同运行条件下对外骨骼辅助进行个性化，受到收集用户反馈所需时间与体力消耗的制约。我们研究了用户的偏好景观（preference landscape）在不同运行条件之间是否平滑变化，以及这种连续性在何时能够支持从有限反馈中进行学习。我们提出了情境连续偏好学习（CCPL），这是一种高斯过程偏好模型，能够在相邻情境之间共享观测数据，同时保留针对特定情境的效用估计。我们通过仿真实验以及对九名健康成年人踝关节和肘关节外骨骼偏好数据的回顾性分析对CCPL进行了评估。在仿真中，当偏好平滑变化时，CCPL相对于独立学习提升了重建效果和基于偏好的贝叶斯优化性能，但在连续性较弱时表现出负迁移。在两项人体研究中，为每位参与者分别估计的全数据参考景观及其跨条件结合（原文在此处截断）……

    arXiv:2609.28427v1 Announce Type: new  Abstract: Personalizing exoskeleton assistance across operating conditions is constrained by the time and physical effort required to collect user feedback. We examined whether a user's preference landscape varies smoothly across operating conditions and when this continuity supports learning from limited feedback. We propose Context-Continuous Preference Learning (CCPL), a Gaussian-process preference model that shares observations across nearby contexts while retaining context-specific utility estimates. We evaluated CCPL through simulations and retrospective analyses of ankle and elbow exoskeleton preference data from nine healthy adults. In simulations, CCPL improved reconstruction and preference-based Bayesian optimization relative to independent learning when preferences varied smoothly, but showed negative transfer when continuity was weak. In both human studies, full-data reference landscapes estimated separately for each participant and co
    
[^8]: 结合机器学习的递归状态估计中不精确求解器的可修复性

    Repairability of Inexact Solvers in Recursive State Estimation with Machine Learning

    [https://arxiv.org/abs/2609.28425](https://arxiv.org/abs/2609.28425)

    本文刻画了递归状态估计（卡尔曼滤波）中不精确数值求解缺陷在给定子空间与范数预算下的可修复条件，并通过残差—漂移恒等式与六阶余项界揭示了新息协方差膨胀与局部增益重优化对有限时域协方差响应的相反作用。

    

    递归状态估计常常在反馈回路内执行近似数值求解，此时高精度的局部步骤并不能保证更优的整体结果。针对固定的线性卡尔曼模型，我们刻画了在给定子空间与范数预算内的修正何时能够满足局部可容许容差，以及实际执行的缺陷如何影响有限时域的协方差响应。将每个缺陷以所实现协方差的精确增益为中心进行刻画，可以把当前求解误差与继承下来的增益漂移分离开来。对精确的残差—漂移恒等式进行展开，揭示了二次响应之外相互对立的四次项贡献：新息协方差的膨胀以正号形式进入，而局部增益的重新优化则以负号形式进入。在匹配初始化条件下，一个在固定时域内对有界缺陷序列一致成立的绝对六阶余项界，给出了二次欠预测或过预测的充分条件。

    arXiv:2609.28425v1 Announce Type: cross  Abstract: Recursive state estimation often executes approximate numerical solutions inside a feedback loop, where highly accurate local steps do not guarantee better overall results. For a fixed linear Kalman model, we characterize when a correction within a prescribed subspace and norm budget can meet a local admissibility tolerance, and how the defects actually executed affect the finite-horizon covariance response. Centering each defect on the exact gain for the implemented covariance separates current solve error from inherited gain drift. Expanding the exact residual-drift identity reveals opposing quartic contributions beyond the quadratic response: innovation-covariance inflation enters positively, while local-gain reoptimization enters subtractively. Under matched initialization, an absolute sixth-order remainder bound, uniform over bounded defect sequences at fixed horizon, gives sufficient conditions for quadratic under- or overpredict
    
[^9]: 智能体编辑世界模型：重新思考面向大语言模型智能体的世界建模

    Agent-Editing World Model: Rethinking World Modeling for LLM Agents

    [https://arxiv.org/abs/2609.28416](https://arxiv.org/abs/2609.28416)

    提出“智能体编辑世界模型”（AEWM），不再模拟工具响应，而是通过动作判官与状态修订来建模推理和动作如何影响未来任务进展，从而避免任务状态污染、提升智能体长时程任务表现。

    

    近年来大语言模型（LLM）的进展使智能体能够在多样化环境中处理长时程任务。为了进一步提升智能体性能，现有的语言世界模型通常预测环境观测，然而在能够获得真实反馈的情况下，重构高熵且依赖执行的工具响应价值有限。与此同时，智能体还饱受“任务状态污染”之苦，即缺乏依据的假设和过时的计划会残留在历史中，并扭曲后续决策。我们提出智能体编辑世界模型（AEWM），它建模推理与动作如何塑造未来的任务进展，而非模拟工具响应。AEWM 将“动作判官”（Action Judge，用于区分关键决策、探索性决策和噪声决策）与“状态修订”（State Revision，用于从相同的观测历史中编辑含噪声的推理-动作延续）相结合。EditAct 将这些整合……（摘要原文在此处被截断）

    arXiv:2609.28416v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have enabled agents to tackle long-horizon tasks across diverse environments. To further improve agent performance, existing language world models typically predict environment observations, yet reconstructing high-entropy, execution-dependent tool responses offers limited value when real feedback is available. Meanwhile, agents suffer from \emph{task-state contamination}, where unsupported assumptions and outdated plans persist in history and distort subsequent decisions. We propose the \textbf{Agent-Editing World Model (AEWM)}, which models how reasoning and actions shape future task progress rather than simulating tool responses. AEWM combines \textbf{Action Judge} to distinguish \textsc{Critical}, \textsc{Exploratory}, and \textsc{Noisy} decisions with \textbf{State Revision} to edit noisy reasoning--action continuations from the same observed history. \textbf{EditAct} integrates thes
    
[^10]: 基于克利福德变分自编码器学习全息缩减表示

    Learning Holographic Reduced Representations with Clifford Variational Autoencoders

    [https://arxiv.org/abs/2609.28409](https://arxiv.org/abs/2609.28409)

    提出了一种名为Clifford-VAE的变分自编码器，通过将数据投影到任意维度的克利福德环面上，为将感知数据嵌入向量符号代数框架提供了原理性方法，并在半监督分类任务和多项VSA基准测试中达到或超越了高斯和超球面VAE的性能。

    

    向量符号代数（Vector Symbolic Algebras）通过将其向量代数应用于随机生成的原子向量符号以及实值数据的分数幂编码，将数据结构投影到超维向量空间中。然而，如何嵌入非结构化数据仍然是一个悬而未决的问题。我们提出了Clifford-VAE，这是一种学习将数据投影到任意维度克利福德环面（Clifford torus）上的变分自编码器。在MNIST、FashionMNIST和CIFAR-10数据集上的实验表明，Clifford-VAE所生成的表示在半监督分类任务中与高斯VAE和超球面VAE的表示具有相当的性能，同时在自绑定与解绑定、角色-填充物恢复以及捆绑容量等VSA基准测试中优于高斯VAE和超球面VAE。Clifford-VAE为将感知数据落地到符号推理框架中提供了一种有原则的技术，提供了一种新的……

    arXiv:2609.28409v1 Announce Type: cross  Abstract: Vector Symbolic Algebras project data structures into a hyperdimensional vector space through the application of their vector algebras to randomly generated atomic vector symbols and fractional power encodings of real-valued data. Embedding unstructured data remains an open question. We present \textit{Clifford-VAE}, a variational autoencoder that learns to project data onto a Clifford torus in arbitrary dimensions. Experiments using the MNIST, FashionMNIST, and CIFAR-10 datasets demonstrate that Clifford-VAE produces representations that are competitive with those produced by Gaussian and Hyperspherical VAEs for semi-supervised classification tasks while outperforming Gaussian and Hyperspherical counterparts in the VSA benchmark tests of self-binding and unbinding, role-filler recovery, and bundle capacity. Clifford-VAE provides a principled technique for grounding perceptual data into a symbolic reasoning framework, providing a new a
    
[^11]: 基于可微高斯表示的集体动力学学习

    Learning Collective Dynamics with Differentiable Gaussian Representations

    [https://arxiv.org/abs/2609.28405](https://arxiv.org/abs/2609.28405)

    本文提出可微高斯动力学（DGD），通过高斯混合表示、可微聚合和反馈循环三个组件，直接从总体计数数据中端到端学习集体响应动力学，并在真实数据集上取得优于DeepAR改进版本的联合行为预测性能。

    

    arXiv:2609.28405v1 公告类型：新论文 摘要：集体响应取决于个体差异、接触机会以及积累的经验。要从总体计数数据中学习其动力学，需要将群体的响应分布与当前观测及未来行为联系起来。我们提出了可微高斯动力学，它通过三个组件学习这种联系：表示异质响应倾向的高斯混合模型、对接触强度与行为概率的可微聚合，以及更新后续响应的反馈循环。重参数化积分与时间循环机制使总体预测误差能够联合训练分布、观测函数和反馈参数。在KuaiRand-Pure和Online Retail II数据集的四个时间窗口上，DGD相比带有联合行为输出头的DeepAR改进版本，实现了更低的联合行为负对数似然。在Retail 2010中，其一天的行为计数……（原文摘要在此处截断）

    arXiv:2609.28405v1 Announce Type: new  Abstract: Collective responses depend on individual differences, contact opportunities, and accumulated experience. Learning their dynamics from aggregate counts requires connecting a population's response distribution to both current observations and future behavior. We introduce Differentiable Gaussian Dynamics (DGD), which learns this connection through three components: a Gaussian mixture representing heterogeneous response propensities, differentiable aggregation of contact intensity and behavioral probabilities, and feedback recurrence that updates subsequent responses. Reparameterized integration and temporal recurrence let aggregate prediction errors jointly train the distribution, observation functions, and feedback parameters. On four windows from KuaiRand-Pure and Online Retail II, DGD achieves lower joint behavioral negative log-likelihood than a DeepAR adaptation with a joint-behavior head. In Retail 2010, its one-day behavioral-count
    
[^12]: 记忆注意力

    Memory Attention

    [https://arxiv.org/abs/2609.28399](https://arxiv.org/abs/2609.28399)

    提出记忆注意力，用可复用的词元记忆与上下文键结合构建注意力值，推理时可简化为查表加法并支持CPU卸载，在相同训练预算下提升了语言建模和下游任务性能。

    

    语言模型通常从上下文隐藏状态构建注意力值，即使其中某些内容可能在不同上下文间是可以复用的。我们研究了在结合上下文信息的情况下，基于词元索引的记忆能否取代专用的值投影。我们提出了记忆注意力，它通过将层特定的词元记忆与上下文键相结合来构建值。记忆提供词元特定的表示，而键则保持对上下文的依赖性。在推理时，归一化可以被折叠进记忆表中，从而将值的构建简化为查找和加法。基于词元索引的检索还支持带预取功能的CPU卸载，减少GPU上的参数存储。在匹配的训练词元预算并引入额外记忆参数的条件下，跨多种注意力配置的实验显示出语言建模和平均下游性能的改进。

    arXiv:2609.28399v1 Announce Type: new  Abstract: Language models typically construct attention values from contextual hidden states, even when some of their content may be reusable across contexts. We investigate whether token-indexed memory can replace the dedicated value projection when complemented by contextual information. We propose Memory Attention (MA), which forms values by combining layer-specific token memory with contextual keys. The memory supplies token-specific representations, while the keys preserve context dependence. At inference, normalization can be folded into the memory tables, reducing value construction to lookup and addition. Token-indexed retrieval also enables CPU offloading with prefetching, reducing GPU parameter storage. Under matched training token budgets and with additional memory parameters, experiments across attention configurations show improved language modeling and average downstream performance.
    
[^13]: 面向翻译的大语言模型微调：通用遗忘缓解方法无法保留机器翻译特定的指令遵循能力

    Fine-Tuning LLMs for Translation: General Forgetting Mitigation Does Not Preserve MT-Specific Instruction Following

    [https://arxiv.org/abs/2609.28395](https://arxiv.org/abs/2609.28395)

    弹性权重巩固等遗忘缓解方法虽能有效保持微调后大语言模型的通用能力，却无法保留机器翻译特定的指令遵循能力（如语体正式度、语法性别和长度控制），表明现有评估方式与实际翻译应用需求存在脱节。

    

    在平行数据上微调大语言模型可以提升翻译质量，但可能引发灾难性遗忘。遗忘缓解方法通常通过模型在通用基准测试上的能力保持情况来评估。我们探究这些结论是否适用于机器翻译（MT）微调以及机器翻译特定的指令遵循（MT-IF），即用于修改翻译结果的指令，例如语体正式度、语法性别和长度控制。我们比较了以辅助数据、模型输出和基础模型参数为锚定的多种方法，首先使用Llama 3.2 1B Instruct进行筛选研究，随后在基于双向阿拉伯语-英语或西班牙语-英语数据微调的Llama 3.1 8B Instruct上开展实验。弹性权重巩固在两个阶段中对通用能力的保持效果最好；在8B西班牙语模型上，通用基准测试的平均分数仅下降1.7分，而标准微调下降11.0分，然而其语体正式度和语法性别控制的得分仍然接近（标准微调的水平）。

    arXiv:2609.28395v1 Announce Type: new  Abstract: Fine-tuning large language models on parallel data improves translation quality but can cause catastrophic forgetting. Mitigation methods are generally evaluated by retention on general benchmarks. We ask whether these findings transfer to machine translation (MT) fine-tuning and to MT-specific instruction following (MT-IF): instructions that modify a translation, such as formality, grammatical gender, and length control. We compare methods anchored to auxiliary data, to model outputs, and to the base model parameters, first in a screening study with Llama 3.2 1B Instruct, then on Llama 3.1 8B Instruct fine-tuned on bidirectional Arabic-English or Spanish-English data. Elastic Weight Consolidation preserves general capabilities best in both stages; on the 8B Spanish model the average score on general benchmarks drops 1.7 points versus 11.0 for standard fine-tuning, yet its scores for formality and grammatical gender control remain close 
    
[^14]: 量子分数匹配及其在热态学习中的应用

    Quantum score matching with applications to learning thermal states

    [https://arxiv.org/abs/2609.28391](https://arxiv.org/abs/2609.28391)

    本文建立了具有端到端理论保证的通用量子分数匹配框架，应用于吉布斯态学习时无需额外制备热态，并在高温区间对有界局域哈密顿量实现了信息论最优的样本复杂度。

    

    分数匹配通过使模型能够从数据中学习而无需评估难以处理的归一化常数（即配分函数），推动了经典生成学习领域的重大进展。然而，将这一原理扩展到量子学习需要重新思考其基础，因为量子态是由非对易的密度算符而非标量概率来描述的。这种非对易性不仅在定义量子分数方面带来了根本性挑战，也在开发具有高效电路实现和严格理论保证的训练框架方面造成了困难。在本工作中，我们通过建立一个具有端到端理论保证的通用量子分数匹配框架来弥合这一差距。将该框架应用于吉布斯态学习时，我们的方法避免了额外的热态制备，并在哈密顿量具有有界局域性和……的条件下，实现了高温区间内信息论最优的样本复杂度。

    arXiv:2609.28391v1 Announce Type: cross  Abstract: Score matching has driven major advances in classical generative learning by enabling models to learn from data without evaluating intractable normalization constants, or partition functions. Yet, extending this principle to quantum learning requires rethinking its foundations, as quantum states are described by noncommuting density operators rather than scalar probabilities. The noncommutativity creates fundamental challenges not only in defining quantum scores, but also in developing a training framework with efficient circuit implementations and rigorous theoretical guarantees. In this work, we bridge this gap by establishing a general quantum score-matching framework with end-to-end theoretical guarantees. Applied to Gibbs-state learning, our approach avoids additional thermal-state preparation and achieves information-theoretically optimal sample complexity in the high-temperature regime for Hamiltonians with bounded locality and 
    
[^15]: 何时何地信任教师：通过熵校准的信用分配统一在线策略蒸馏与GRPO

    When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment

    [https://arxiv.org/abs/2609.28385](https://arxiv.org/abs/2609.28385)

    该论文提出UECR-GRPO方法，通过熵校准的信用重分配，在响应和token两个层面将教师信号与验证器信号统一整合进单一的KL正则化GRPO更新中，从而解决在线策略蒸馏与可验证奖励强化学习结合时教师指导引入时机不当及token重加权破坏任务信用总量的问题。

    

    可验证奖励的强化学习（RLVR）通过最终答案的正确性来监督数学推理，但对单个token提供的指导甚少。在线策略蒸馏（OPD）能够对学生生成的响应提供密集反馈，但教师的偏好未必反映答案的正确性。近期的混合方法将OPD与验证器派生的优势相结合，或利用教师比率对任务信用进行重新加权。然而，这些方法中教师指导是在基于验证器的组归一化之后才引入的，且token重新加权未必能保留分配给每个响应的总任务信用。我们提出了面向GRPO的统一熵校准信用重分配方法（UECR-GRPO），它在响应和token两个层面将验证器信号与教师信号整合到单一的GRPO风格更新中。其中，路径-效用统一（PUU）在单一KL正则化目标中结合了验证器奖励与教师到锚点的路径对数比率。其在线策略实现……（原文摘要在此处截断）

    arXiv:2609.28385v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) supervises mathematical reasoning through final-answer correctness, but provides little guidance on individual tokens. On-policy distillation (OPD) supplies dense feedback on student-generated responses, yet teacher preference need not reflect correctness. Recent hybrids combine OPD and verifier-derived advantages or reweight task credit using teacher ratios. However, teacher guidance enters after verifier-based group normalization, and token reweighting need not preserve the total task credit assigned to each response. We introduce Unified Entropy-Calibrated Credit Redistribution for GRPO (UECR-GRPO), which integrates verifier and teacher signals within a single GRPO-style update at both the response and token levels. \emph{Path-Utility Unification} (PUU) combines verifier reward and a teacher-to-anchor path log-ratio in a single KL-regularized objective. Its on-policy implementati
    
[^16]: ForgetMimic：面向强化学习人形机器人控制的动作遗忘方法

    ForgetMimic: Motion Unlearning for Reinforcement Learning Humanoid Control

    [https://arxiv.org/abs/2609.28378](https://arxiv.org/abs/2609.28378)

    提出了首个面向物理世界人形机器人控制的动作级遗忘方法ForgetMimic，能够从强化学习策略中选择性移除特定动作（如恶意、被污染或涉及版权的动作），同时保持其余动作的性能不受影响。

    

    通过强化学习（RL）利用人类示范数据，人形机器人控制已经实现了多样化、敏捷且自然的运动行为。尽管这一范式在物理人形机器人控制中取得了显著的性能，但如何从已学习的策略中消除特定动作仍未得到充分探索。解决这一问题的动机来自紧迫的安全与隐私方面的考量：移除恶意、被污染或次优的动作，以及依据GDPR等法规中“被遗忘权”的要求移除受版权保护的动作，都具有至关重要的意义。为此，我们提出了ForgetMimic，这是首个专门为物理世界人形机器人控制设计的动作级遗忘方法。ForgetMimic的核心思想是：给定一个在N个动作上训练得到的策略π_θ，我们的方法能够降低该策略在K个目标动作子集上的性能表现，同时保持其余动作的有效性。

    arXiv:2609.28378v1 Announce Type: cross  Abstract: Humanoid control, leveraging human demonstrations, has achieved diverse, agile, and natural locomotion behaviors through reinforcement learning (RL). While this paradigm has yielded remarkable performance in physical humanoid control, how to eliminate specific motions from learned policies remains insufficiently explored. Addressing this issue is motivated by pressing safety and privacy concerns: the removal of malicious, poisoned, or suboptimal motions, as well as copyright-protected motions subject to the right to be forgotten under regulations such as the GDPR, is of critical importance. To this end, we propose {ForgetMimic}, the first motion-level unlearning method designed specifically for physical-world humanoid control. The core idea of ForgetMimic is as follows: given a policy $\pi_\theta$ trained on $N$ motions, our method degrades performance on a target subset of $K$ motions while preserving the effectiveness of the remainin
    
[^17]: LEAP-CBF：基于最小努力对抗势的面向不确定系统的安全过滤器

    LEAP-CBF: A Safety Filter for Uncertain Systems with Least-Effort Adversarial Potentials

    [https://arxiv.org/abs/2609.28364](https://arxiv.org/abs/2609.28364)

    提出了最小努力对抗势（LEAP），通过量化扰动导致系统失效所需的最小努力来衡量状态的鲁棒性，结合深度强化学习构建了对有界累积扰动鲁棒的安全过滤器。

    

    控制屏障函数（CBF）是一种流行的安全过滤器，用于确保非线性动力系统的安全性。然而，当系统受到不确定性和扰动的影响时，这需要使用鲁棒变体的CBF，而这些变体往往难以构建且可能过于保守，特别是在输入约束下的高维系统中。在这项工作中，我们提出了一种解决这些挑战的新方法，即引入最小努力对抗势，这是一种证书，通过扰动导致系统失效所需的努力程度来量化给定状态对扰动的鲁棒性。我们证明了LEAP是无扰动系统的一个CBF，同时也可以用于构建对累积努力有界的扰动具有鲁棒性的安全过滤器。我们提出了一种使用策略上深度强化学习来构建LEAP的方法。接下来，我们在仿真中对多种场景（原文此处截断）演示了LEAP的效果。

    arXiv:2609.28364v1 Announce Type: cross  Abstract: Control barrier functions (CBF) are a popular safety filter to ensure safety for nonlinear dynamical systems. However, when the system is subject to uncertainties and disturbances, this requires the use of robust variants of CBFs, which can be difficult to construct and can be overly conservative, especially for high-dimensional systems under input constraints. In this work, we propose a new approach to solve these challenges by introducing Least-Effort Adversarial Potentials (LEAP), a certificate that quantifies the robustness of a given state against disturbances in terms of the effort required by the disturbance to cause failure. We show that LEAP is a CBF for the undisturbed system, but can also be used to construct a safety filter that is robust to disturbances whose cumulative effort is bounded. We propose a method for constructing LEAPs with on-policy deep reinforcement learning. Next, we demonstrate LEAPs in simulation on a var
    
[^18]: 通过Dobrushin收缩实现局部几何混合及其在扩散路径蒙特卡洛与近端采样器中的应用

    Local Geometric Mixing via Dobrushin Contraction with Applications to Diffusion Path Monte Carlo and the Proximal Sampler

    [https://arxiv.org/abs/2609.28338](https://arxiv.org/abs/2609.28338)

    本文提出基于Dobrushin收缩的局部几何混合分析框架，并将其应用于扩散路径蒙特卡洛和近端采样器，在最少假设下为理想方法及Metropolis校正版本提供了混合时间保证。

    

    局部几何混合通过仅在有限多次转移中要求在总变差距离下几何收敛到平衡态，从而将几何混合局部化。它能够容纳局部收敛速率，并刻画快速的局部均衡化现象，即使全局混合慢得多。我们通过Dobrushin收缩建立并讨论了局部几何混合的界。随后，我们将该方法应用于扩散路径蒙特卡洛——这是一种最近提出的马尔可夫链蒙特卡洛方法，旨在利用基于分数（score-based）建模的最新进展，其理想转移与近端采样器的转移相一致。我们的分析同时覆盖了理想方法及其可实现的经过Metropolis校正的对应版本，并在最少的假设下提供了混合保证。对于理想方法，这些保证补充了最近的谱隙估计结果，我们进一步将其发展为混合时间界。

    arXiv:2609.28338v1 Announce Type: cross  Abstract: Local geometric mixing localizes geometric mixing by requiring geometric convergence to equilibrium in total variation only over finitely many transitions. It accommodates local convergence rates and captures rapid local equilibration, even when global mixing is much slower. We establish and discuss local geometric mixing bounds through Dobrushin contraction. We then apply this approach to Diffusion Path Monte Carlo, a recently proposed Markov chain Monte Carlo method, aimed at leveraging advances in score-based modeling, whose ideal transitions coincide with those of the Proximal Sampler. Our analysis covers both the ideal method and its implementable Metropolis-adjusted counterpart, providing mixing guarantees under minimal assumptions. For the ideal method, these guarantees complement recent spectral gap estimates, which we develop into mixing time bounds.
    
[^19]: 学习可靠推理的成本

    Learning the Cost of Reliable Inference

    [https://arxiv.org/abs/2609.28322](https://arxiv.org/abs/2609.28322)

    该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。

    

    基准测试与路由平台日益成为连接大型语言模型提供商与终端用户的中介。然而，这些平台上的提供商通常采用固定的每token定价方式，使用户无法为其任务获得最具竞争力的价格。在本工作中，我们设计了一个采购平台，其中每个任务的token价格由提供商之间的竞争驱动，使用户能够在保证质量水平的前提下获得有竞争力的价格。为此，该平台通过反向第二价格拍卖依次路由查询，激励模型提供商真实地竞标其服务用户查询的平均成本的最佳估计。在路由查询的过程中，平台学习每个提供商所提供的质量，并逐步将查询路由到满足期望质量阈值的提供商中最具成本竞争力的提供商。为验证我们的设计，我们使用多个模型进行了实验。

    arXiv:2609.28322v1 Announce Type: new  Abstract: Benchmarking and routing platforms increasingly act as intermediaries connecting large language model providers with end-users. However, providers on these platforms typically use a fixed price per token, preventing users from achieving the most competitive price for their tasks. % workloads. In this work, we design a procurement platform where token prices for each task are driven by provider competition, enabling users to secure competitive pricing for guaranteed quality levels. To this end, the platform sequentially routes queries via a reverse second-price auction that incentivizes model providers to truthfully bid their best estimate of the average cost to serve a user's query. As it routes queries, the platform learns the quality offered by each provider and progressively routes queries to the most cost-competitive provider among those meeting a desired quality threshold. To validate our design, we conduct experiments with multiple
    
[^20]: 基于双编码器Transformer的卫星辐射数据行星边界层高度估计

    PBLH Estimation from Satellite Radiances via a Dual-Encoder Transformer

    [https://arxiv.org/abs/2609.28286](https://arxiv.org/abs/2609.28286)

    本文基于MetOp-ERA5大规模数据集，对八种PBLH估计方法建立了基准测试，通过分组Shapley分解量化了模型对输入的依赖关系，并提出了一种性能最佳的双编码器Transformer架构。

    

    从卫星观测中估计行星边界层高度（PBLH）是一个具有挑战性的回归问题，原因在于大气层顶辐射与近地表大气结构之间存在间接关系。该领域的研究进展一直受到两方面的制约：一是缺乏能够处理卫星过境数据多模态、空间不完整特性的架构，二是缺乏合适的数据集。在本文中，我们基于此前工作中引入的大规模数据集（该数据集将MetOp辐射数据与ERA5 PBLH标签配对），做出了三项贡献。第一，我们在八种方法之间建立了一个基准测试，涵盖逐像素回归、沿扫描幅宽的序列模型，以及在完整轨道弧段上运行的卷积模型和Transformer模型。第二，我们利用输入块上的分组Shapley分解，量化了模型实际依赖的内容。第三，我们提出了性能最佳的架构……

    arXiv:2609.28286v1 Announce Type: cross  Abstract: Estimating the Planetary Boundary Layer Height (PBLH) from satellite observations is a challenging regression problem due to the indirect relationship between top-of-atmosphere radiances and near-surface atmospheric structure. Progress has been limited both by the lack of architectures capable of handling the multimodal, spatially incomplete nature of satellite overpasses, and by the scarcity of suitable datasets. In this paper, we build upon the large-scale dataset pairing MetOp radiances with ERA5 PBLH labels that we introduced in our previous work, making three contributions. First, we establish a benchmark across eight approaches spanning pixel-wise regression, swath-wise sequence models, and convolutional and Transformer models operating on the full orbital passage. Second, we quantify what the resulting model actually relies on, using grouped Shapley decomposition over the input blocks. Third, we present the best-performing archi
    
[^21]: Mamba-3中基于输入依赖低秩更新的非交换状态跟踪

    Non-Commutative State Tracking with Input-Dependent Low-Rank Updates in Mamba-3

    [https://arxiv.org/abs/2609.28273](https://arxiv.org/abs/2609.28273)

    通过在Mamba-3中引入输入依赖的秩一低秩反射更新，实现单个块内的非对角状态转移，从而支持操作顺序至关重要的非交换状态跟踪。

    

    从序列观测中进行状态跟踪可能既需要保留信息，又需要通过组合观测到的操作来更新信息。我们通过引入一个输入依赖的低秩反射项来扩展Mamba-3的对角转移，以支持非交换状态跟踪，即操作顺序至关重要的情形。该秩一更新沿输入依赖的方向耦合状态坐标，使得在单个Mamba-3块内即可实现非对角状态转移。该扩展保留了Mamba-3的指数梯形离散化、旋转位置编码（RoPE）以及读出机制。在训练方面，我们对分块计算进行了适配，以便在每个分块内并行化所提出的递归结构。实验涵盖了具有离散输入的群文字问题和具有连续观测的贝壳游戏（其中策略通过行为克隆进行训练）。在按固定计时条件下依据其出色性能所筛选出的模型中，所提出的模型保持了更高的跟踪（准确率）。

    arXiv:2609.28273v1 Announce Type: cross  Abstract: State tracking from sequential observations can require both retaining information and updating it by composing observed operations. We extend Mamba-3's diagonal transition with an input-dependent low-rank reflection term to support noncommutative state tracking, in which the order of operations matters. The rank-one update couples state coordinates along an input-dependent direction, enabling non-diagonal state transitions within a single Mamba-3 block. The extension preserves Mamba-3's exponential-trapezoidal discretization, rotary embeddings (RoPE), and readout. For training, we adapt chunkwise computation to parallelize the proposed recurrence within each chunk. Experiments cover group word problems with discrete inputs and a shell game with continuous observations, in which a policy is trained by behavioral cloning. Among the models selected for their strong performance under fixed timing, the proposed model maintains higher track
    
[^22]: 通过预测量化代价在部署前选择PTQ配置

    Predicting Quantization Price for Selecting PTQ Configurations Before Deployment

    [https://arxiv.org/abs/2609.28270](https://arxiv.org/abs/2609.28270)

    该论文将权重空间后训练量化（PTQ）重构为部署前配置选择问题，用全精度模型的下游曲率为每层量化配置所诱导的输出误差协方差“定价”，从而在统一框架下于部署前预测并比较不同量化格式、粒度、量化器族、变换和比特位宽配置的优劣。

    

    权重空间的后训练量化（PTQ）必须在量化模型完成并暴露其输出分布漂移之前，就确定有限的格式、粒度、量化器族、变换和比特位宽等选择。现有的PTQ方法能够预测这种性能退化中的重要组成部分，包括重构误差、Hessian敏感度、变换效应和下游损失，但这些指标通常是在固定量化几何结构之后、或在各自独立的配置族内部进行评分的。我们将权重空间PTQ形式化为一个基于“定价层输出误差”的部署前配置选择问题：每个可行的层配置都被视为一个带有部署成本的误差生成器，它会诱导出层输出误差协方差 $\boldsymbol{\Sigma}_l(\alpha_l)$，而全精度模型则通过下游曲率为该协方差定价，即 $\widehat{\rho}_l(\alpha_l)=\frac{1}{2}\operatorname{Tr}(\widehat{\mathbf{H}}_l\,\widehat{\boldsymbol{\Sigma}}_l\cdots)$。（摘要原文在此处截断）

    arXiv:2609.28270v1 Announce Type: new  Abstract: Weight-space post-training quantization (PTQ) must choose finite formats, granularities, quantizer families, transformations, and bits before the completed quantized model reveals its output-distribution drift. Existing PTQ methods predict important pieces of this degradation, including reconstruction error, Hessian sensitivity, transformation effects, and downstream loss, but these pieces are usually scored after fixing the quantization geometry or inside separate configuration families. We formulate weight-space PTQ as pre-deployment configuration selection using priced layer-output error. Each admissible layer configuration is treated as an error generator with a deployment cost, which induces a layer-output error covariance $\boldsymbol{\Sigma}_l(\alpha_l)$, and the full-precision model prices that covariance by downstream curvature, $\widehat{\rho}_l(\alpha_l)=\frac{1}{2}\operatorname{Tr}\left(\widehat{\mathbf{H}}_l\,\widehat{\bolds
    
[^23]: 面向在线线性规划的资源自适应随机梯度下降算法：无需重新求解

    Resource-Adaptive Stochastic Gradient Descent for Online Linear Programming without Re-solving

    [https://arxiv.org/abs/2609.28263](https://arxiv.org/abs/2609.28263)

    提出资源自适应随机梯度下降算法（RASGD），通过一阶SGD更新模拟重求解的资源定价逻辑，每次请求仅需O(m)次运算即可达到O(log T)期望后悔界，完全无需重新求解线性规划。

    

    大规模语言模型（LLM）推理与搜索服务的增长使得在线线性规划问题的规模不断扩大，这促使人们需要计算高效的算法。我们针对随机在线线性规划开发了一种资源自适应随机梯度下降算法。该算法利用单次请求和当前库存来更新资源价格，每次请求到达仅需O(m)次运算（m为资源数量）和相应的内存开销，无需进行线性规划求解或样本平均优化。其核心思想是通过一阶SGD更新来表达重求解方法中的当前资源定价逻辑：每次请求到达都会刷新对偶目标中的剩余库存限额，同时步长在早期递减以便于学习，后期递增以匹配库存调整的速度。在标准的非退化条件下，我们的算法在每条样本路径上都是可行的，并且相对于实现的小数事后最优解达到了O(log T)的期望后悔界。

    arXiv:2609.28263v1 Announce Type: new  Abstract: The growth of large language model (LLM) inference and search services increases the scale of online linear programming problems, motivating computationally efficient algorithms. We develop resource-adaptive stochastic gradient descent (RASGD) for stochastic online linear programming. The algorithm uses one request and current inventory to update resource prices, requiring O(m) operations for m resources and memory per arrival and no LP or sample-average optimization. The central idea is to express the current-resource pricing logic of re-solving through a first-order SGD update: each arrival refreshes the remaining-inventory allowance in the dual objective, while the stepsize decreases for early learning and increases later to match the speed of inventory adjustment. Under standard non-degeneracy conditions, our algorithm is feasible on every sample path and achieves O(\log T) expected regret against the realized fractional hindsight op
    
[^24]: RAMP：面向边缘CPU视觉模型的鲁棒自适应混合精度量化

    RAMP: Robust Adaptive Mixed-Precision Quantization for Edge CPU Vision Models

    [https://arxiv.org/abs/2609.28262](https://arxiv.org/abs/2609.28262)

    该论文通过对13种逐层INT8量化敏感度指标在多个异构神经网络上的系统性实证研究，揭示了现有指标在现代架构上的系统性失效问题，并据此提出了面向边缘CPU视觉模型的鲁棒自适应混合精度量化方法RAMP，在两个ARM64平台上验证了其在保持精度的同时降低推理延迟的有效性。

    

    在边缘CPU上部署深度学习模型受到计算和内存约束的瓶颈。混合精度量化有望在保持精度的同时降低推理延迟。然而，量化对不同的层类型会产生不一致的影响，因此找出精度损失最小且延迟降低最大的量化位置至关重要，因为这种影响会在整个部署过程中累积，带来可观的节省或不可接受的任务性能退化。这种识别依赖于敏感度指标，即在无需评估每个候选策略任务精度的情况下估计各层性能退化的代理指标。然而，广泛使用的指标在现代架构上会系统性地失效。我们对用于逐层INT8量化的13种敏感度指标在四个截然不同的神经网络开展了系统的实证研究，并在两个ARM64平台上验证了由此得到的量化策略。基于梯度的敏感度（摘要在此处被截断）

    arXiv:2609.28262v1 Announce Type: cross  Abstract: Deploying deep learning models on edge CPUs is bottlenecked by computational and memory constraints. Mixed-precision quantization promises to reduce inference latency while preserving accuracy. However, quantization affects different layer types in inconsistent ways, so identifying where accuracy loss is minimized and latency reduction is maximized is critical, as the effect accumulates over a full deployment into substantial savings or unacceptable task degradation. Such identification relies on sensitivity metrics, proxies that estimate layer-wise degradation without evaluating the task accuracy of every candidate policy. Nevertheless, widely used metrics fail systematically on modern architectures. We present a systematic empirical study of 13 sensitivity metrics for layer-wise INT8 quantization across four distinctly different neural networks, and validate the resulting policies on two ARM64 platforms. Gradient-based sensitivity me
    
[^25]: 基于世界模型的可泛化机器人插装

    Generalizable Robotic Insertion with World Models

    [https://arxiv.org/abs/2609.28258](https://arxiv.org/abs/2609.28258)

    提出基于世界模型的可泛化机器人插装框架，通过单一世界模型融合本体感知与腕部摄像头视觉信息，在90个几何多样的插装任务上训练后，对未知几何形状的新物体实现56%的零样本成功率（远超无模型基线的7%），且性能随训练物体数量增加而持续提升。

    

    高混合生产环境中的机器人装配需要能够处理多样化零件的自适应系统，然而当前的方法通常依赖于针对每个插装任务专门定制的策略。尽管这种方式可以达到较高的成功率，但使得将系统部署到新问题上的过程变得繁琐且耗时。我们提出了一个利用世界模型实现可泛化插装的框架，该框架将机器人本体感知信息与安装在腕部的摄像头所捕获的原始视觉观测相结合。我们这种基于模型的方法在多达90个具有几何多样性零件的插装任务上训练了单一的世界模型，在几何形状未知的未见物体上实现了56%的零样本成功率，而无模型基线方法仅为7%。重要的是，随着训练数据集中纳入更多物体，性能持续提升，展现出强大的可扩展性。最后，在留出物体上对通用模型进行微调显著提升了数据效率。

    arXiv:2609.28258v1 Announce Type: cross  Abstract: Robotic assembly in high-mixture settings requires adaptable systems that can handle diverse parts, yet current approaches typically rely on policies specialized to each insertion task. Although this can reach high success rates, it makes the process of deploying systems for new problems tedious and time consuming. We present a framework for generalizable insertion using world models that combine robot proprioceptive information with raw visual observations captured by a wrist-mounted camera. Our model-based approach trains a single world model on up to 90 insertion tasks with geometrically diverse parts, achieving 56% zero-shot success on unseen objects with unknown geometry compared to just 7% with a model-free baseline. Importantly, performance improves as more objects are included in the training dataset, demonstrating strong scalability. Lastly, finetuning the generalist model on held-out objects significantly enhances data-effici
    
[^26]: hyperbolix：基于 JAX 的双曲深度学习库

    hyperbolix: Hyperbolic Deep Learning in JAX

    [https://arxiv.org/abs/2609.28248](https://arxiv.org/abs/2609.28248)

    该论文提出了 hyperbolix，首个基于 JAX 与 Flax NNX 构建的全面、通用的开源双曲深度学习库，提供六种统一接口的流形、覆盖各类双曲神经网络层族以及黎曼优化器和降维工具。

    

    我们提出了 hyperbolix，一个基于 Flax NNX 构建、运行于 JAX 之上的开源双曲深度学习库。据我们所知，这是 JAX 中首个全面、通用的双曲深度学习库。它包含六种具有统一接口的流形：欧几里得空间、庞加莱球、双曲面、κ-stereographic 模型、混合曲率乘积空间以及固有速度空间。我们实现了涵盖线性层、卷积、注意力机制、归一化、位置编码、回归和向量量化的各类层族。这些构建模块所覆盖的方法范围，从 Ganea 最初提出的双曲神经网络，到近期的完全双曲架构（如 Hypformer 和 Lorentzian ResNet）应有尽有。此外，hyperbolix 还包含以 optax 变换形式实现的黎曼优化器、封装的分布以及双曲降维技术。其 API 采用地道的 JAX 惯用风格：流形是无状态的……（原文摘要在此处截断）

    arXiv:2609.28248v1 Announce Type: new  Abstract: We present hyperbolix, an open-source library for hyperbolic deep learning in JAX, built on Flax NNX. To our knowledge, it is the first comprehensive, general-purpose hyperbolic deep learning library in JAX. It includes six manifolds with a common interface: Euclidean space, the Poincar\'e ball, the hyperboloid, the $\kappa$-stereographic model, mixed-curvature product spaces, and the proper velocity space. We implement layer families that cover linear layers, convolutions, attention, normalization, positional encoding, regression, and vector quantization. These building blocks span methods ranging from Ganea's original hyperbolic neural networks to recent fully hyperbolic architectures such as Hypformer and Lorentzian ResNet. Additionally, hyperbolix contains Riemannian optimizers implemented as optax transformations, wrapped distributions, and hyperbolic dimensionality-reduction techniques. Its API uses idiomatic JAX: Manifolds are sta
    
[^27]: 对数深度循环语言建模

    Log-Depth Recurrent Language Modeling

    [https://arxiv.org/abs/2609.28212](https://arxiv.org/abs/2609.28212)

    本文将平衡树递归算子扩展到自回归语言建模，实现了以对数深度和线性运行时间计算所有前缀表示，展现出稳健的长度外推能力和接近ALiBi Transformer的性能。

    

    尽管使用Transformer进行语言建模已成为常态，但其计算深度固定，且运行时间随输入token数量呈平方级增长。另一方面，循环模型虽然提供线性深度，却无法并行执行。在这项工作中，我们将平衡树递归算子从序列编码扩展到自回归预测，使得所有前缀表示能够以对数深度和线性运行时间计算。我们的实验对这一模型类别进行了初步刻画，展示了其稳健的长度外推能力以及接近基于ALiBi的Transformer的性能，突显了其作为语言建模替代架构的潜力。

    arXiv:2609.28212v1 Announce Type: cross  Abstract: Language modeling using Transformers has become commonplace despite their fixed computational depth and quadratic runtime with respect to input tokens. Recurrent models on the other hand offer linear depth but no parallel execution. In this work, we extend balanced-tree recursive operators from sequence encoding to autoregressive prediction, enabling all prefix representations to be computed with logarithmic depth and linear runtime. Our experiments provide an initial characterization of this model class, demonstrating robust length extrapolation and performance approaching that of ALiBi-based Transformers, highlighting its potential as an alternative architecture for language modeling.
    
[^28]: 支持度编译特征折叠：在表格基础模型中以更低内存利用更多证据

    Support-Compiled Feature Folding: More Evidence at Lower Memory Across Tabular Foundation Models

    [https://arxiv.org/abs/2609.28208](https://arxiv.org/abs/2609.28208)

    提出免训练推理框架“支持度编译特征折叠”（SCFF），将按支持度排序的有界特征子集路由进冻结的表格基础模型，把二次方特征交互开销降为线性，在不集成预测、不训练新参数的情况下以更低内存利用更多证据，并在18个宽表数据集上提升了全部六个骨干网络的准确率与NLL。

    

    表格基础模型面临特征侧的扩展困境：全宽度的成对特征混合随列数呈二次方增长，而特征选择虽然节省内存，却以丢弃证据为代价。我们提出了支持度编译特征折叠（SCFF），这是一种无需训练的推理框架，无需修改冻结的骨干网络即可解决这一困境。SCFF将按支持度排序的特征路由至原生特征编码器的有界叶子节点，对残差证据进行支持度检查，并在单次上下文预测之前合并编码后的信息。由此，它将二次方的特征交互计算转化为随宽度线性增长的计算，并保持有界的局部工作集，且无需集成预测或训练新参数。在固定的AMLB-29、TabZilla和TabArena快照的完整18个宽表数据集切片上，SCFF在所有六个被评估的骨干网络上均提升了数据集聚合准确率和NLL。所有四组同等宽度对比均保持优势。

    arXiv:2609.28208v1 Announce Type: new  Abstract: Tabular foundation models face a feature-side scaling dilemma: full-width pairwise mixing grows quadratically with the number of columns, whereas feature selection saves memory by discarding evidence. We introduce Support-Compiled Feature Folding (SCFF), a training-free inference framework that resolves this dilemma without changing the frozen backbone. SCFF routes support-ranked features through bounded leaves of the native feature encoder, support-checks the residual evidence, and merges the encoded messages before a single contextual prediction. It thereby converts quadratic feature-interaction work into linear-in-width work with a bounded local working set, without ensembling predictions or training new parameters. On the exhaustive 18-dataset wide-table slice of fixed AMLB-29, TabZilla, and TabArena snapshots, SCFF improves dataset-macro accuracy and NLL on all six evaluated backbones. All four matched-width comparisons retain favor
    
[^29]: 纵向血糖表征的可迁移证据重构

    Transferable Evidence Reconstruction for Longitudinal Glucose Representations

    [https://arxiv.org/abs/2609.28199](https://arxiv.org/abs/2609.28199)

    该论文提出可迁移证据重构（TER）自监督方法，通过在一个记录组上拟合低容量读取器并要求其在另一组记录中恢复相同证据的跨组测试，学习具有可迁移证据解码规则的血糖表征，并利用感知观测的每日编码器和感知时钟的多日记忆模块对持续血糖监测数据进行建模。

    

    长时间的生理记录中包含大量常规测量，而具有预测价值的信息往往集中在罕见事件、持续负担和重复出现的时间模式中。掩码自编码通过恢复测量值进行学习；对比学习通过 对齐不同视图进行学习。我们研究一种显式优先考虑结构化信号证据的自监督方法。我们提出了可迁移证据重构（Transferable Evidence Reconstruction, TER），该方法从无标签记录中构建证据，在一个记录组上拟合一个全新的低容量读取器，并要求该读取器在不重新拟合的情况下从另一组记录中恢复相同的证据。通过对这一跨组测试进行微分，可以学习到带有可迁移证据解码规则的表征；这些证据仅用于引导自监督学习，而不作为下游特征使用。针对持续血糖监测（CGM），我们设计了感知观测的每日编码器和感知时钟的多日记忆模块，将血糖水平及其变化与记录时间相绑定，同时（摘要在此处截断）

    arXiv:2609.28199v1 Announce Type: new  Abstract: Long physiological recordings contain many routine measurements, while predictive information is often concentrated in rare events, sustained burden, and recurring temporal patterns. Masked autoencoding recovers measurements; contrastive learning aligns views. We study self-supervision that explicitly prioritizes structured signal evidence. We introduce transferable evidence reconstruction (TER), which constructs evidence from unlabeled recordings, fits a fresh low-capacity reader on one recording group, and requires that reader to recover the same evidence in another group without refitting. Differentiating through this cross-group test learns representations with transferable evidence-decoding rules; the evidence guides self-supervision but is not used as a downstream feature. For continuous glucose monitoring (CGM), an observation-aware daily encoder and clock-aware multi-day memory bind glucose level and change to recorded time while
    
[^30]: 地理空间嵌入特征可识别原始老林，但带缓冲的空间验证缩小了其相对于Sentinel特征的优势

    Geospatial embeddings detect old-growth forests but buffered spatial validation narrows their advantage over Sentinel features

    [https://arxiv.org/abs/2609.28194](https://arxiv.org/abs/2609.28194)

    该研究在罗马尼亚南喀尔巴阡山脉绘制原始老林分布图，发现地理空间基础模型嵌入特征能有效检测原始老林，但在采用10公里缓冲区进行空间验证以控制空间自相关后，其相对常规Sentinel-1/2特征的优势明显缩小。

    

    原始老林在数百年间极少受人为干扰的环境中发育，形成结构复杂且生物多样性丰富的林分。在欧洲，保护这些森林需要既能在单个林分尺度上保证精度、又可在整个大陆范围内部署的制图方法。地理空间基础模型（GFM）嵌入为标签稀缺条件下的土地分类提供了可能，但其在原始老林检测中的价值尚不清楚。本研究对罗马尼亚南喀尔巴阡山脉211,893公顷的区域（阿尔卑斯生物地理区内典型的山毛榉-云杉景观）的原始老林进行制图。我们构建了高置信度、由专家参与的原始林与非原始林林分参考标签，在以地形和人类可达性预测变量构成的共同基线之上，加入AlphaEarth、TESSERA v2以及Sentinel-1/2特征，并在带与不带10公里训练-测试缓冲区的空间分块验证下进行比较，以限制残留的空间自相关。采用缓冲区后，GFM与Sentinel-1/2……（原文摘要在此处截断）

    arXiv:2609.28194v1 Announce Type: new  Abstract: Old-growth forests develop over centuries under minimal anthropogenic disturbance, producing structurally complex and biodiverse stands. In Europe, protecting them requires mapping that is accurate for individual forest parcels yet deployable continent-wide. Geospatial foundation model (GFM) embeddings enable label-scarce land classification, but their value for old-growth detection remains unknown. Here, we map old-growth forests across 211,893 ha of Romania's Southern Carpathians, a beech-spruce landscape typical of the Alpine Biogeographic Region. We construct high-confidence, expert-informed reference labels for old-growth and non-old-growth parcels. We add AlphaEarth, TESSERA v2 and Sentinel-1/2 features to a common baseline of topographic and human-access predictors, then compare them under spatially blocked validation with and without 10 km train-test buffers to limit residual autocorrelation. With buffering, GFM and Sentinel-1/2 
    
[^31]: 面向基于AI的电网边缘协调的有限样本概率安全认证

    Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination

    [https://arxiv.org/abs/2609.28182](https://arxiv.org/abs/2609.28182)

    本文提出了一种基于精确二项推断的有限样本概率安全认证框架，能够为闭环电网运行中的黑盒AI决策模型给出不安全运行概率的最紧单侧上界证书，为系统运营商独立严谨地判定AI系统是否可安全部署提供了依据。

    

    协调大规模柔性电网边缘设备可以缓解对耗时且耗资巨大的网络升级的需求，而多智能体强化学习或模仿学习等基于AI的控制方法在实时决策的可扩展性方面前景广阔。然而，系统运营商仍然需要一种独立且严谨的方法来判定给定的AI系统是否足够安全、可以部署。本文针对闭环电网运行中的黑盒AI决策模型，提出了一个有限样本概率安全认证框架。其核心思想是：在运营商定义的安全规范下，将完整的“输入—AI—电网评估器”工作流程简化为二元的不安全结果，然后利用精确的二项推断来认证相应的不安全运行概率。给定一组留出的校准场景，该框架返回最紧的单侧上界证书以及接受/拒绝的判定。

    arXiv:2609.28182v1 Announce Type: new  Abstract: Coordinating large population of flexible grid-edge devices can alleviate the need for time-consuming and capital-intensive network upgrades, and AI-based control methods such as multi-agent reinforcement learning or imitation learning are promising in their real-time decision scalability. However, system operators still need an independent and rigorous way to decide whether a given AI system is safe enough for deployment. This paper develops a finite-sample probabilistic safety certification framework for black-box AI decision models in closed-loop grid operation. The central idea is to reduce the complete input--AI--grid evaluator workflow to a binary unsafe outcome under an operator-defined safety specification, and then use exact binomial inference to certify the corresponding unsafe operation probability. Given a set of held-out calibration scenarios, the framework returns the tightest one-sided upper certificate and an accept/rejec
    
[^32]: LLM 排行榜声明对隐藏模型选择有多敏感？

    How Sensitive Are LLM Leaderboard Claims to Hidden Model Selection?

    [https://arxiv.org/abs/2609.28177](https://arxiv.org/abs/2609.28177)

    该论文提出一种敏感性分析方法，量化一个排行榜领先幅度背后可能隐藏的私下挑选的模型变体数量，并对 Open LLM Leaderboard 上 394 个相邻排名声明进行审计，发现其中 391 个即使不考虑选择效应也缺乏统计支持。

    

    LLM 排行榜上的领先可能反映了在私下评估的多个模型变体之间进行挑选的结果，然而变体的数量及其相关性均未公开。我们探究的问题是：在保持对某个固定比较对象具有统计优势证据的前提下，一个已公布的领先幅度最多能支持多少个隐藏变体。对于一个固定候选模型族，在高斯边际模型下，我们推导出一条敏感性曲线，将该最大数量表示为族内相关性下界的函数。相关的相关性必须与用于排名的分数和抽样模型相匹配：在一个受控模型族中，合并题目层面的相关性为 0.90，而在题目重抽样下综合分数的相关性为 0.46，在 MMLU 学科重抽样下则为 0.92。对 Open LLM Leaderboard 上 394 个相邻排名声明的基于题目的审计发现，即使在未考虑选择效应之前，其中 391 个声明就已缺乏统计支持。在通过未校正检验的声明中，经认证（原文此处截断）……

    arXiv:2609.28177v1 Announce Type: cross  Abstract: LLM leaderboard gains can reflect selection among privately evaluated model variants, yet neither the number of variants nor their dependence is public. We ask how many hidden variants a published margin can support while retaining statistical evidence of a provider's advantage over a fixed comparator. For a fixed candidate family under a Gaussian margin model, we derive a sensitivity curve that reports this maximum count as a function of a lower bound on within-family correlation. The relevant correlation must match the score used for ranking and the sampling model: in a controlled family, pooled item correlation is 0.90, whereas composite-score correlation is 0.46 under item resampling and 0.92 when MMLU subjects are resampled. An item-based audit of 394 adjacent-rank claims on the Open LLM Leaderboard finds that 391 lack statistical support even before accounting for selection. Among claims that pass the uncorrected test, certificat
    
[^33]: 信心不足：优化带来的非对称确定性增益阻碍多模态分类

    Confidence Falls Short: Asymmetric Certainty Gains from Optimization Hinder Multimodal Classification

    [https://arxiv.org/abs/2609.28165](https://arxiv.org/abs/2609.28165)

    该论文发现多模态优化会使强模态获得比弱模态更高的预测置信度（即非对称的确定性增益），并提出MaxCR方法，通过非线性稀疏度量追踪各模态语义置信度并动态施加跨模态干预，以纠正置信度差异、提升多模态分类性能。

    

    多模态学习由于模态不平衡现象而陷入优化困境，导致实践中整体性能欠佳。尽管许多尝试主要聚焦于平衡各模态间的优化动态来解决这一问题，我们发现了一个微妙但关键的缺陷：优化在预测确定性上产生非对称的增益，强模态比弱模态更加自信，从而驱动了不平衡的模态贡献。在本文中，我们的分析揭示了这一缺陷源于单模态特性而非多模态学习本身，并且这种置信度差异可以通过正向的跨模态干预来纠正。基于这一洞察，我们提出多模态最大置信度正则化来动态干预模态语义置信度。具体而言，使用非线性稀疏度量来追踪每个模态的语义置信度，随后设计了最大支持策略（原文此处截断）。

    arXiv:2609.28165v1 Announce Type: new  Abstract: Multimodal learning (MML) falls into the optimization dilemma due to the modality imbalance phenomenon, leading to suboptimal overall performance in practice. While many attempts primarily focus on balancing the optimization dynamics across modalities to address this issue, we identify a subtle yet critical flaw: optimization yields asymmetric gains in predictive certainty, with the strong modality more confident than the weak one, driving imbalanced modality contributions. In this paper, our analysis reveals that this flaw stems from unimodal characteristics rather than multimodal learning, and this confidence discrepancy can be corrected by positive cross-modal intervention. Based on this insight, we propose multimodal Max Confidence Regularization (MaxCR) to dynamically intervene in modality semantic confidence. Specifically, the semantic confidence of each modality is tracked using a nonlinear sparsity measure. We then design max sup
    
[^34]: EvEMTBench：面向电力系统保护机器学习的开放基准

    EvEMTBench: An Open Benchmark for Machine Learning in Power System Protection

    [https://arxiv.org/abs/2609.28149](https://arxiv.org/abs/2609.28149)

    EvEMTBench 是一个面向电力系统保护机器学习的开放、可执行、可版本化基准，通过统一固定任务定义、数据划分与评估指标，支持跨可观测性条件、分布偏移和跨电网迁移的结构化比较。

    

    基于机器学习的电力系统保护研究往往难以相互比较，因为其任务定义、量测信息获取方式、数据划分、评价指标和泛化条件常常各不相同。EvEMTBench 通过一个开放、可执行且可版本化的基准来填补这一空白，该基准固定了上述评估设定，同时对模型设计保持开放。该基准覆盖电压等级为 20–345 kV 的四个电网，定义了 12 项保护与事件分析功能，并实例化为 24 个评分任务，支持在可观测性条件、预定义的分布偏移以及零样本和微调的跨电网迁移等维度上进行结构化评估。固定的数据划分、泄漏控制与可复现的报告为未来方法的比较提供了共同基础。一项涵盖简单方法、传统方法、基于特征的方法和深度学习基线的参考评估表明，更广的可观测性并非普遍有益，而分布偏移条件可能暴露出（原文在此处截断）……

    arXiv:2609.28149v1 Announce Type: cross  Abstract: Studies of machine-learning-based power system protection are difficult to compare because task definitions, measurement access, data partitions, metrics, and generalization conditions often differ. EvEMTBench addresses this gap with an open, executable, and versioned benchmark that fixes these evaluation choices while leaving model design open. Across four grids spanning 20-345 kV, it defines 12 protection and event-analysis functions instantiated as 24 scored tasks and supports structured evaluation across observability conditions, predefined distribution shifts, and zero-shot and fine-tuned cross-grid transfer. Committed partitions, leakage controls, and reproducible reporting provide a common basis for comparing future methods. A reference evaluation spanning trivial, conventional, feature-based, and deep-learning baselines shows that wider observability is not uniformly beneficial, shifted conditions can reveal failures not appare
    
[^35]: 强化学习始于强化学习之前：论在策略蒸馏对更好强化学习的作用

    RL Starts before RL: On Policy Distillation for Better Reinforcement Learning

    [https://arxiv.org/abs/2609.28145](https://arxiv.org/abs/2609.28145)

    研究发现在策略蒸馏（OPD）作为RL的准备阶段能带来超越初始准确率提升的收益——即使OPD对准确率几乎没有即时改善，经其初始化的模型在RL后仍能达到更高的最终性能，原因可能在于与教师分布的对齐偏向高质量推理路径，同时保留了RL可通过结果反馈继续优化的备选路径。

    

    强化学习（RL）能够提升推理能力，但其性能取决于训练开始时所基于的策略。我们研究了在策略蒸馏（OPD）作为RL的准备阶段，并探究其收益是否超出了蒸馏模型初始准确率的提升。在共享的RL设置下，使用OPD初始化的学生模型相比直接进行RL训练、或先监督微调再进行RL训练的学生模型，能够达到更高的最终性能。即使OPD在准确率上几乎没有带来即时提升，这种优势也可能出现。RL前的Pass@k并不能完全解释这一收益：相似甚至更高的Pass@k值并不一定能在RL后带来更好的性能。行为分析指出，与教师分布的对齐超越top-1一致性可能是合理的解释。这种对齐可能偏好更高质量的推理路径，同时保留RL可以借助结果反馈进一步优化的备选路径。我们进一步研究……

    arXiv:2609.28145v1 Announce Type: new  Abstract: Reinforcement learning (RL) improves reasoning, but its performance depends on the policy from which training begins. We study on-policy distillation (OPD) as a preparation stage for RL and ask whether its benefits extend beyond improvements in the distilled model's initial accuracy. Under shared RL settings, students initialized with OPD reach higher final performance than those trained with direct RL or supervised fine-tuning followed by RL. This advantage can emerge even when OPD produces little immediate improvement in accuracy. Pre-RL Pass@k does not fully explain the benefit: similar or even higher values do not necessarily lead to better performance after RL. Behavioral analyses point to alignment with the teacher's distribution beyond top-1 agreement as a possible explanation. Such alignment may favor higher-quality reasoning paths while retaining alternatives that RL can further refine using outcome feedback. We further examine 
    
[^36]: NPBoost：基于梯度提升固定效应的神经过程

    NPBoost: Neural Processes with Gradient-Boosted Fixed Effects

    [https://arxiv.org/abs/2609.28122](https://arxiv.org/abs/2609.28122)

    NPBoost将结构化响应变异性分解为跨任务共享的梯度提升树固定效应与捕捉任务间随机变化的神经过程随机效应，并通过提升算法联合训练二者，在共享结构包含不连续性等不规则特征的表格元学习任务上优于标准神经过程。

    

    神经过程（Neural Processes, NPs）是一类基于模型的元学习器，能够隐式地学习一个随机过程，并从小型上下文集出发适应新任务。大多数神经过程的扩展工作都聚焦于改进神经网络架构。与此不同，我们从元学习与混合效应模型的共同层次化解释出发，开发了一种新的扩展方法。具体而言，我们提出了神经过程提升（Neural Process Boosting, NPBoost），它将结构化的响应变异性分解为跨任务共享的树提升固定效应，以及捕捉任务间随机变化的神经过程随机效应。我们提出使用一种提升算法联合训练这两个组件，其中神经过程学习残差的任务特定结构，树集成模型则估计跨任务的共同模式。在合成数据和真实世界的表格元学习问题上，实验表明当共享结构包含不连续性或其他不规则特征时，这种分解方法优于标准神经过程。

    arXiv:2609.28122v1 Announce Type: cross  Abstract: Neural Processes (NPs) are model-based meta-learners that implicitly learn a stochastic process and adapt to a new task from a small context set. Most extensions of NPs focus on improving the neural network architecture. We instead develop an extension motivated by the shared hierarchical interpretation of meta-learning and mixed-effects models. Specifically, we introduce Neural Process Boosting (NPBoost), which decomposes structured response variability into tree-boosted fixed effects shared across tasks and NP random effects that capture stochastic task-to-task variation. We propose to train the two components jointly using a boosting algorithm in which an NP learns residual task-specific structure and a tree ensemble estimates common patterns across tasks. Across synthetic and real-world tabular meta-learning problems, this decomposition improves over a standard NP when the shared structure contains discontinuities or other irregula
    
[^37]: 面向刮削层等离子体模拟的概率化且几何感知的神经代理模型

    Probabilistic and Geometry Aware Neural Surrogate of Scrape Off Layer Plasma Simulations

    [https://arxiv.org/abs/2609.28116](https://arxiv.org/abs/2609.28116)

    该论文提出一种将SOLPS-ITER曲线网格无损展开为图像张量并结合条件流匹配的概率化神经代理模型，能够在偏滤器脱靶转变等敏感工况下保留几何结构信息，并捕捉多种可能的等离子体状态及其不确定性。

    

    用于托卡马克边界等离子体模拟的快速代理模型通常是将全局运行点映射为展平的网格单元数值向量的确定性回归器。然而，在偏滤器脱靶转变附近，稳态并不可靠地保持单值，点估计不得不对性质迥异的不同等离子体状态取平均，且不附带任何置信度说明。此外，展平向量表示还丢弃了SOLPS-ITER网格的几何结构。本工作同时解决上述两个问题：我们首先将曲线网格展开为三个固定尺寸的图像张量，其布局保持了网格单元间的邻接关系且可精确逆变换，使卷积网络能够在不损失信息的前提下利用几何结构；随后在该表示上训练一个条件流匹配模型，该模型尤其适用于高度敏感的系统。最终得到的代理模型高效且可扩展，即使在敏感的运行点附近也能捕捉多种可能的结果。

    arXiv:2609.28116v1 Announce Type: new  Abstract: Fast surrogates for tokamak boundary-plasma simulation are typically deterministic regressors mapping a global operating point to a flattened vector of cell values. Near the divertor detachment transition the steady state is not reliably single-valued. A point estimate must average over qualitatively different plasma states, and it arrives with no statement of confidence. Moreover, the flattened vector representation discards the geometric structure of the SOLPS-ITER mesh. This work addresses both problems. We unroll the curvilinear mesh into three fixed-size image tensors whose layout preserves cell adjacency and inverts exactly, letting a convolutional network act on the geometry without loss of information. A conditional flow matching model, well suited to highly sensitive systems, is then trained on this representation. The result is an efficient, scalable surrogate that captures multiple plausible outcomes even at sensitive operatin
    
[^38]: 通过条件流匹配蒸馏实现高效的多任务操作策略

    Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching

    [https://arxiv.org/abs/2609.28107](https://arxiv.org/abs/2609.28107)

    该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。

    

    生成式建模的最新进展近来已被广泛应用于机器人学的策略学习中。特别是，使用专家演示训练的条件流匹配（CFM）在机器人操作基准测试中已被证明优于现有方法。虽然先前的工作主要集中于单任务设置，但我们从多任务的角度来研究这个问题，因为为每个任务训练独立模型的计算成本非常高。多任务策略学习本身也面临一系列挑战：在简单拼接的演示数据集上进行朴素训练，要么需要增加模型容量以适应额外的复杂性，要么会导致性能下降。我们提出通过迁移单任务CFM专家模型所学习到的速度场，将其知识蒸馏到一个共享的多任务策略中。我们将该蒸馏信号与原始的CFM目标相结合，以保持对专家演示数据的保真度。

    arXiv:2609.28107v1 Announce Type: cross  Abstract: Advances in generative modeling have recently been extensively employed in robotics for policy learning. In particular, Conditional Flow Matching (CFM) trained with expert demonstrations has been shown to outperform existing methods on robot manipulation benchmarks. While prior work has mainly focused on single-task settings, we study the problem from a multi-task perspective, as training independent models for each task is computationally expensive. Multi-Task policy learning comes with its own set of challenges, as naively training on a concatenated dataset of demonstrations would either require increased model capacity to accommodate the added complexity or result in drops in performance. We propose to distill knowledge from single-task CFM experts into a shared multi-task policy by transferring their learned velocity fields. We combine this distillation signal with the original CFM objective to retain fidelity to the demonstrations
    
[^39]: Fed-ReMasker：特征级缺失下的联邦表格数据填补

    Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness

    [https://arxiv.org/abs/2609.28105](https://arxiv.org/abs/2609.28105)

    提出Fed-ReMasker，将ReMasker掩码自编码器适配到联邦学习框架中，使各中心能够利用跨协作中心学到的知识填补本地从未观测到的特征，从而解决了现有联邦填补方法很少评估的特征级缺失问题。

    

    多中心临床研究和生物医学研究合作日益希望利用跨中心的数据来构建超越任何单一中心泛化能力的模型。这带来了两个独特的挑战：数据保护法规可能限制跨机构共享原始患者数据，而各中心在不同协议下可能仅收集部分重叠的特征集合。联邦学习使得无需集中原始数据即可进行协同模型训练成为可能。然而，现有的联邦填补方法很少评估特征级缺失的情况，即某些特征在某些中心完全未被观测到。为应对这一场景，我们将 ReMasker 掩码自编码器适配到联邦学习框架中（Fed-ReMasker），使各中心能够利用跨协作中心学到的知识，对本地从未观测到的特征进行填补。我们在一个涵盖线性和非……的合成数据集的基准上评估了 Fed-ReMasker（注：原文摘要在此处被截断）。

    arXiv:2609.28105v1 Announce Type: cross  Abstract: Multi-center clinical studies and biomedical research collaborations increasingly seek to utilize data across centers to build models that generalize beyond any single center. This creates two distinct challenges: data protection regulations may restrict the sharing of raw patient data across institutions, while centers may collect only partially overlapping sets of features under different protocols. Federated learning enables collaborative model training without centralizing raw data. However, existing federated imputation methods rarely evaluate feature-level missingness, in which entire features are unobserved at some centers. To address this setting, we adapt the ReMasker masked autoencoder to federated learning (Fed-ReMasker), enabling centers to impute features never observed locally by leveraging knowledge learned across collaborating centers. We evaluate Fed-ReMasker in a benchmark spanning synthetic datasets with linear and n
    
[^40]: 视觉绊线：预判深度视觉系统的失效

    Visual Tripwires: Anticipating Failure in Deep Vision Systems

    [https://arxiv.org/abs/2609.28099](https://arxiv.org/abs/2609.28099)

    提出Visual Tripwires框架，通过监测模型潜在表示漂移、预测振荡、轨迹曲率和注意力熵等时间不稳定性信号，提前预测深度视觉系统在未来时间范围内的失效概率。

    

    深度视觉系统尽管在基准测试中表现优异，但仍然容易受到数据损坏、遮挡和分布偏移的影响。现有的可靠性方法通常只在单个时间步评估不确定性，并未显式地建模系统如何逐步走向失效。我们提出了视觉绊线，这是一个预测性可靠性框架，利用模型行为中的时间不稳定性来预判即将发生的失效。我们的核心假设是：预测性能的退化会通过潜在表示、预测轨迹和注意力结构中可测量的变化而逐步发展。视觉绊线通过表示漂移、预测振荡、轨迹曲率和注意力熵来捕捉这些变化。一个轻量级的绊线预测器在时间窗口内聚合这些信号，以估计在未来预测时间范围内发生失效的概率。在多个数据集和架构上的实验……

    arXiv:2609.28099v1 Announce Type: cross  Abstract: Deep vision systems remain vulnerable to corruption, occlusion, and distribution shift despite strong benchmark performance. Existing reliability methods typically evaluate uncertainty at individual time steps and do not explicitly model how a system progresses toward failure. We introduce Visual Tripwires, a predictive reliability framework that uses temporal instability in model behaviour to anticipate impending failure. Our central hypothesis is that predictive degradation develops progressively through measurable changes in latent representations, prediction trajectories, and attention structure. Visual Tripwires captures these changes using representation drift, prediction oscillation, trajectory curvature, and attention entropy. A lightweight tripwire predictor aggregates these signals over a temporal window to estimate the probability of failure within a future prediction horizon. Experiments across multiple datasets, architectu
    
[^41]: 沿基于数据的诊断过程发现完全高效的故障指示器

    Discovery of fully efficient fault indicators along a data-based diagnosis process

    [https://arxiv.org/abs/2609.28087](https://arxiv.org/abs/2609.28087)

    本文提出 DT4X+，通过改进训练集构建与符号回归损失函数，使诊断表达式在分离目标类别的同时保持解析冗余关系的可解释性，解决了原 DT4X 算法仅优化两类分离而导致类别碎片化、性能下降的问题。

    

    基于模型与数据驱动两种范式的融合，通过将解析冗余关系（即基于模型诊断中用作诊断指标的输入输出关系）的可解释性与学习技术的适应性相结合，为故障诊断提供了一个强大的框架。DT4X 是一种较新的诊断算法，它利用符号回归生成多元关系，借助解析冗余关系的某些特性，并将其用作决策树中的分裂函数。然而，其符号回归过程在每个节点上仅优化两个所选类别之间的分离，常常使剩余类别碎片化，从而同时降低了可解释性和诊断性能。本文提出了 DT4X+，即 DT4X 的增强版本，它修改了训练集的构建方式和符号回归的损失函数，使得生成的表达式在分离目标类别的同时保留（原文此处截断）

    arXiv:2609.28087v1 Announce Type: new  Abstract: The integration of model-based and data-driven paradigms provides a powerful framework for fault diagnosis by combining the interpretability of analytical redundancy relations, i.e., input-output relations that are used as diagnosis indicators in model-based diagnosis, with the adaptability of learning techniques. DT4X is a recent diagnosis algorithm that uses symbolic regression to generate multivariate relations leveraging some properties of analytical redundancy relations and uses them as split functions in a decision tree. However, its symbolic regression procedure optimizes only the separation between two selected classes at each node, often fragmenting the remaining classes and degrading both interpretability and diagnosis performance. This paper introduces DT4X+, an enhanced version of DT4X that modifies the construction of training sets and the symbolic-regression loss so that expressions separate the target classes while preserv
    
[^42]: LAYERSCOPE：视频与多模态学习表征的逐层刻画

    LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations

    [https://arxiv.org/abs/2609.28086](https://arxiv.org/abs/2609.28086)

    提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。

    

    我们提出了LAYERSCOPE，这是一个无标签的逐层分析框架，旨在刻画模型在视频和多模态场景下学习到的表征。使用最终层或中间层表征来评估下游性能，通常需要大量带标签的数据、重复的任务特定评估以及大量的计算。为了解决这些局限性，LAYERSCOPE利用局部、全局、分布以及基于对应关系的几何度量，在无需任务特定标签的情况下，比较模型内部以及跨模型的逐层表征结构。我们在MVEB/MVEB+的多个任务上评估了七个架构各异的模型，涵盖视频与多模态分类、聚类以及文本到视频检索。我们发现中间层的表征可以优于最终层和模型默认输出。我们还发现没有任何单一的几何度量能够一致地预测下游性能，但注意到

    arXiv:2609.28086v1 Announce Type: cross  Abstract: We propose LAYERSCOPE, a label-free, layerwise framework that aims to characterize a model's learned representations in video and multimodal settings. Evaluating downstream performance using representations from final or intermediate layers typically requires large amounts of labeled data, repeated task-specific evaluations, and substantial computation. To address these limitations, LAYERSCOPE uses local, global, distributional, and correspondence-based geometric metrics to compare layerwise representation structure within and across models without requiring task-specific labels. We evaluate seven architecturally diverse models across video and multimodal classification, clustering, and text-to-video retrieval tasks from MVEB/MVEB+. We find that intermediate-layer representations can outperform final-layer and model-default outputs. We also find that no single geometric metric consistently predicts downstream performance, but note that
    
[^43]: 基于图神经网络强化学习的课程学习方法求解作业车间调度问题

    Curriculum Learning with GNN-based Reinforcement Learning for Job Shop Scheduling

    [https://arxiv.org/abs/2609.28085](https://arxiv.org/abs/2609.28085)

    本文提出在作业车间调度问题中采用课程学习策略训练基于图神经网络的强化学习模型，通过先在小规模实例上训练再逐步过渡到更大目标规模，相比单一规模训练有效提升了模型的跨规模泛化能力。

    

    作业车间调度问题是一个具有挑战性的组合优化问题，近年来基于图神经网络的强化学习方法展现出直接从问题实例中学习调度策略的前景。然而，在大型实例上进行训练的计算开销依然很高，跨实例规模的泛化能力也仍然是一个难题。本文研究了作业车间调度问题中基于图神经网络的强化学习的课程学习方法，并在20×20、25×25和30×30三种目标规模上将其与单一规模训练进行了比较。在课程学习设置中，策略首先在较小实例上进行训练，然后逐步适应更大的目标规模，使早期阶段学到的调度行为能够支持在更大实例上的学习。模型在从8×8到30×30的未见实例上进行评估，使用最优性间隙作为指标，同时考虑泛化能力（摘要在此处截断）。

    arXiv:2609.28085v1 Announce Type: cross  Abstract: The job shop scheduling problem is a challenging combinatorial optimization problem, and recent reinforcement learning approaches using graph neural networks have shown promise for learning scheduling policies directly from problem instances. However, training on large instances remains computationally expensive, and generalization across instance sizes remains challenging. This paper studies curriculum learning for graph neural network-based reinforcement learning in the job shop scheduling problem by comparing it with single-size training across three target sizes: 20 x 20, 25 x 25, and 30 x 30. In the curriculum setting, the policy is first trained on smaller instances and then progressively adapted to larger target sizes, allowing scheduling behavior learned in earlier stages to support learning on larger instances. Models are evaluated on unseen instances from 8 x 8 to 30 x 30 using the optimality gap, considering both generalizat
    
[^44]: 面向混合专家模型的精确分位数均衡与负载误差注入方法

    Exact Quantile Balancing and Load-Error Injection for Mixture-of-Experts

    [https://arxiv.org/abs/2609.28053](https://arxiv.org/abs/2609.28053)

    该论文提出精确分位数均衡（EQB）和负载误差注入（LEI）两种方法，分别以极小的通信开销计算精确全局分位数、以及将局部负载误差直接注入路由器梯度，从而在7.5B参数的混合专家模型上显著改善全局与局部负载均衡并提升下游性能。

    

    混合专家模型的训练需要全局负载均衡以防止专家利用率不足，同时需要局部均衡以实现高效的专家并行执行。现有的分布式分位数均衡方法依赖于分片相关的或近似的全局分位数，而与token无关的专家偏置无法确保微批次级别的均衡。我们提出了精确分位数均衡（EQB），它以可忽略的通信开销计算精确的全局批次BF16分位数，以及负载误差注入（LEI），它将局部负载误差直接注入到路由器得分的梯度中。在训练多达5000亿token的75亿参数混合专家模型上，EQB相比朴素的分位数均衡方法改善了全局均衡和下游性能，而LEI改善了局部均衡，在相当的质量下优于GShard损失函数。

    arXiv:2609.28053v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) training requires global load balance to prevent expert under-utilization and local balance for efficient expert-parallel execution. Existing distributed Quantile Balancing (QB) uses shard-dependent or approximate global quantiles, while token-independent expert biases cannot ensure microbatch-level balance. We introduce Exact Quantile Balancing (EQB), which computes exact global-batch BF16 quantiles with negligible communication, and Load-Error Injection (LEI), which injects local load errors directly into router-score gradients. On 7.5B-parameter MoEs trained for up to 500B tokens, EQB improves global balance and downstream performance over naive QB, while LEI improves local balance and outperforms the GShard loss at comparable quality.
    
[^45]: Transformer键值缓存的张量分解：谱结构与格式比较

    Tensor Decomposition of Transformer Key-Value Caches: Spectral Structure and Format Comparison

    [https://arxiv.org/abs/2609.28029](https://arxiv.org/abs/2609.28029)

    该研究通过谱分析发现Transformer的KV缓存中词元和特征模式具有低秩结构而注意力头和层模式近乎满秩，并在相同存储条件下证明Tucker分解在2至5倍压缩比下重构误差最低，且键和值的最优压缩表示形式存在差异。

    

    自回归Transformer的键值（KV）缓存可以被视为一个跨越注意力头、词元、特征和分组层的四阶张量。我们在Mistral-7B-v0.3和LLaMA-2-13B上测量了全部四种模式展开的奇异值谱，并在相同存储条件下比较了四种标准张量分解方法：Tucker、CP、张量列车和张量奇异值分解。谱分析将这四个轴划分为两类：词元和特征模式具有低秩结构，尤其是对于键而言；而注意力头和层模式几乎是满秩的，在任何实际误差水平下都难以压缩。在这四种分解方法中，Tucker在从2倍到5倍的每个压缩比下都实现了最低的重构误差，这是因为它能够保持满秩模式不变。与二维展开基线的比较表明，键和值的首选表示形式有所不同：2D方法在键上能实现更低的误差，而

    arXiv:2609.28029v1 Announce Type: cross  Abstract: The key-value (KV) cache of autoregressive transformers can be viewed as a fourth-order tensor spanning attention heads, tokens, features, and grouped layers. We measure the singular-value spectra of all four mode unfoldings on Mistral-7B-v0.3 and LLaMA-2-13B and compare four standard tensor decompositions: Tucker, CP, tensor train, and t-SVD, at matched storage. The spectra partition the four axes into two classes. The token and feature modes carry low-rank structure, particularly for keys. The head and layer modes are nearly full-rank and resist compression at any practical error level. Among the four decompositions, Tucker achieves the lowest reconstruction error at every compression ratio from $2\times$ to $5\times$, because it can leave the full-rank modes untouched. Comparisons with two-dimensional unfolding baselines show that the preferred representation differs between keys and values: 2D methods achieve lower key error, while
    
[^46]: PISCES：用于空间天气异常检测与早期预警的物理信息太阳风卷积自编码器

    PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning

    [https://arxiv.org/abs/2609.28022](https://arxiv.org/abs/2609.28022)

    PISCES是一种无需标签、融合多种物理约束的太阳风卷积自编码器，它将异常分数分解为磁场、等离子体、物理关系和残差修正等物理可解释的分量，从而实现空间天气瞬变结构的早期检测与预警。

    

    空间天气早期预警依赖于在太阳风瞬变结构到达地球之前，在太阳-地球第一拉格朗日点（L1）的原位测量中检测到它们。固定阈值方法可能会遗漏磁与等离子体组合结构的异常，而许多学习方法仅提供一个单一的异常分数。我们提出了物理信息太阳风卷积自编码器（PISCES），这是一个在物理约束下、无需目录标签、基于OMNI太阳风测量数据训练的卷积自编码器。其损失函数包含磁场一致性、温度与速度之间的经验关系、帕克螺旋角，以及对由重构计算出的派生量在连续一分钟采样之间变化的惩罚项。在推理阶段，PISCES将异常分数分解为磁场重构误差、等离子体重构误差、物理关系误差和残差修正，并报告每个分量的贡献大小。

    arXiv:2609.28022v1 Announce Type: cross  Abstract: Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined magnetic and plasma structure, and many learning methods provide a single anomaly score. We present the Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather (PISCES), a convolutional autoencoder trained without catalog labels on OMNI solar wind measurements under physics constraints. Its loss includes magnetic field consistency, an empirical relation between temperature and velocity, the Parker spiral angle, and penalties on changes between consecutive one-minute samples in derived quantities calculated from the reconstruction. At inference, PISCES separates the anomaly score into magnetic and plasma reconstruction errors, physics relations, and residual corrections, and reports the magnitude of each contribution. Attenua
    
[^47]: 基于流匹配改进的集合滤波器

    Improving Ensemble Filters with Flow Matching

    [https://arxiv.org/abs/2609.28015](https://arxiv.org/abs/2609.28015)

    提出流集合滤波器，利用条件流匹配学习非线性更新，将经典基线滤波器的预报集合传输为分析集合，在稀疏观测的动力学系统中性能超越所有经典集合滤波器及最先进的生成式方法。

    

    数据同化旨在从部分且含噪声的观测中估计动力学状态。经典的集合滤波器虽然高效，但其分析更新受限于有限样本协方差和仿射高斯分布。我们提出了流集合滤波器，它利用条件流匹配将预报集合从经典基线滤波器传输到分析集合。FlowEF 在训练时使用局部化高斯源，在部署时从基线滤波器传输预报集合成员，并将其速度场条件于该基线滤波器的集合以及观测之上。因此，所提出的模型在学习非线性更新的同时，能够独立地映射每个基线集合成员。对于稀疏观测的动力学系统，FlowEF 在确定性和概率性指标上均优于全部四种经典集合滤波器，并且在最先进的生成式模型中取得了最佳性能。

    arXiv:2609.28015v1 Announce Type: cross  Abstract: Data assimilation estimates a dynamical state from partial and noisy observations. Classical ensemble filters are efficient but restrict analysis updates through finite sample covariance and affine Gaussian distribution. We introduce the Flow Ensemble Filter (FlowEF), which uses conditional flow matching to transport the forecast ensemble from a classical baseline filter to an analysis ensemble. FlowEF uses a localized Gaussian source during training, transports forecast ensemble members from a baseline filter at deployment, and conditions its velocity field on ensembles from that baseline filter and the observation. The proposed model therefore learns a nonlinear update while mapping each baseline ensemble independently. For sparsely observed dynamical systems, FlowEF improves both deterministic and probabilistic metrics over all four classical ensemble filters. It also achieves the best performance among the state-of-the-art generati
    
[^48]: SoLiD26：用于机器学习原子间势的第一性原理固液界面数据集

    SoLiD26: A First Principles Solid-Liquid Interface Dataset for Machine-learned Interatomic Potentials

    [https://arxiv.org/abs/2609.28013](https://arxiv.org/abs/2609.28013)

    SoLiD26是一个包含1540万个第一性原理原子结构、涵盖15种化学元素的固液界面数据集，可用于训练和评估面向电化学、催化和腐蚀等应用的机器学习原子间势。

    

    面向先进材料应用（如电化学、催化和腐蚀）中固液界面的机器学习原子间势（MLIPs），需要同时采样液体环境、固体以及界面本身的训练数据。我们提出了SoLiD26，这是一个经过精心筛选的固液界面数据集，包含1540万个第一性原理原子结构，体系最多达576个原子、涵盖15种化学元素，可用于训练和评估机器学习原子间势。这些结构汇编自固液界面研究中开展的密度泛函理论（DFT）计算，其中大多数构型来自从头算分子动力学（AIMD）模拟。每条记录包含原子种类、坐标位置、模拟晶胞、周期性边界条件、势能和原子受力。SoLiD26包含水相货币金属（铜族金属）界面、电极-电解质体系以及精选的体相参考结构，均采用VASP计算（原文在此处截断）。

    arXiv:2609.28013v1 Announce Type: cross  Abstract: Machine-learned interatomic potentials (MLIPs) for solid-liquid interfaces in advanced materials applications, e.g., electrochemistry, catalysis and corrosion, require training data that samples both liquid environments, the solid and the interface itself. We present SoLiD26, a curated solid-liquid interface dataset, containing 15.4 million first-principles atomic structures with up to 576 atoms and 15 chemical elements for training and evaluating MLIPs. The structures were compiled from density functional theory (DFT) calculations performed in studies of solid-liquid interfaces, with most configurations originating from ab initio molecular dynamics (AIMD) simulations. Each record contains atomic species, positions, simulation cell, periodic boundary conditions, potential energy and atomic forces. SoLiD26 includes aqueous coinage metal interfaces, electrode-electrolyte systems, and selected bulk reference structures, calculated with VA
    
[^49]: 在检索与硬件约束下评估面向土耳其语领域文档的开放权重大语言模型

    Evaluating Open-Weight LLMs for Turkish Domain Documents Under Retrieval and Hardware Constraints

    [https://arxiv.org/abs/2609.28007](https://arxiv.org/abs/2609.28007)

    本文提出了一种无需额外模型调用即可区分检索失败与模型推理失败的带证据标注评估协议，并在6 GB显存的本地硬件约束下系统评估了五个开放权重7B-8B模型对土耳其语长篇领域文档问答的能力。

    

    大多数具备土耳其语能力的大语言模型（LLM）通常使用通用基准进行评估，而非基于长篇幅、结构复杂的领域文档。本文在资源受限的本地部署环境下，评估了五个开放权重的7B-8B模型在土耳其语文档问答任务上的表现。主要基准包含100个经过系统性验证的问题，这些问题源自一份109页的工业研发报告；评估协议还在一份112页的公共部门报告以及独立构建的100题问题集上进行了复现。所有模型均在配备6 GB显存的NVIDIA RTX 3050笔记本GPU上本地运行，并采用受控的提示、解码和4位量化设置。本文的主要方法论贡献是一种带证据标注的评估协议，该协议无需额外的模型调用即可将检索失败与下游模型推理失败区分开来。在主要基准上，端到端准确率……（原文摘要在此处截断）

    arXiv:2609.28007v1 Announce Type: new  Abstract: Most Turkish-capable large language models (LLMs) are evaluated using general-purpose benchmarks rather than long, structurally complex domain documents. This paper evaluates five open-weight 7B-8B models for Turkish document question answering under a resource-constrained local deployment setting. The primary benchmark contains 100 systematically validated questions derived from a 109-page industrial R&D report, and the evaluation protocol is replicated using a second 112-page public-sector report and an independently constructed 100-question set. All models are evaluated locally on an NVIDIA RTX 3050 laptop GPU with 6 GB VRAM using controlled prompting, decoding, and 4-bit quantisation.   The principal methodological contribution is an evidence-annotated evaluation protocol that separates retrieval failure from downstream model reasoning failure without requiring additional model calls. On the primary benchmark, end-to-end accuracy ran
    
[^50]: 共享全局KV与层级专属局部历史

    Shared Global KV with Layer-Specific Local History

    [https://arxiv.org/abs/2609.28006](https://arxiv.org/abs/2609.28006)

    该论文提出在跨层共享全局KV缓存的同时，为每一层保留独立的局部历史记忆，实验表明相比基于当前token的局部分支可降低约1.4%的困惑度，并在与GQA及相邻层KV共享等方案的对比中取得更优的同源似然。

    

    arXiv:2609.28006v1 公告类型：新论文 摘要：仅解码器Transformer语言模型在生成过程中缓存键和值（KV）以复用先前的计算。跨层共享KV可以节省存储空间，但会降低跨深度可用的表示多样性。我们研究了在与共享全局KV并存的情况下，局部记忆应当保留什么内容，并将历史内容与用于形成该内容的输入来源区分开来。在1.26亿参数和2K上下文长度的设置下，一项八种子研究 发现，与使用当前token的局部分支相比，使用局部历史可使留存测试困惑度降低约1.4%。容量、条目数和训练计算量的对照实验支持了历史内容的价值。在一项双种子比较中，当相邻层共享局部输入但保留独立的投影时，这一价值依然存在；来源共享还缩短了精确缓存构建的依赖链。与GQA和相邻层KV共享相比，在同等的受限学习率搜索和新种子确认下，本方法获得了更好的同源似然。

    arXiv:2609.28006v1 Announce Type: new  Abstract: Decoder-only Transformer language models cache keys and values (KV) to reuse past computation during generation. Sharing KV across layers saves storage but reduces the diversity of representations available across depth. We study what local memory should retain alongside shared global KV, separating historical content from the input source used to form it. At 126M parameters and 2K context, an eight-seed study finds about 1.4% lower held-out test perplexity with local history than with a current-token local branch. Capacity, entry-count and training-compute controls support the value of historical content. In a two-seed comparison, this value persists when adjacent layers share local inputs while retaining independent projections; source sharing also shortens exact cache-construction dependencies. Against GQA and adjacent-layer KV sharing, equal bounded learning-rate searches and new-seed confirmation yield better same-source likelihood 
    
[^51]: 从失败中学习：面向小语言模型工具使用智能体的异构图记忆

    Learning from Failures: Heterogeneous Graph Memory for Small Language Model Tool-Using Agents

    [https://arxiv.org/abs/2609.28003](https://arxiv.org/abs/2609.28003)

    提出FRESH框架，通过基于异构图的失败感知检索机制，将智能体的历史成功与失败经验转化为结构化外部记忆，从而提升小型语言模型在工具调用中的可靠性与安全性。

    

    小型和中型语言模型为工具使用型智能体提供了经济高效的执行器，使其在本地部署和大规模部署场景中颇具吸引力。然而，在长时程和有状态的环境中，这类模型经常出现结构性错误，例如遗漏必要的观察结果、过早执行写入操作、重复失败的调用以及违反动作前置条件。这些错误可能导致错误的状态更新、违反策略规则以及代价高昂或不可逆的后果，使可靠的工具执行成为关键的部署挑战。现有的微调方法需要大量的数据和计算资源，而扁平化的记忆在检索失败动作时无法保留其因果上下文或安全条件。在本文中，我们提出了FRESH，一个基于经验结构化异构图的失败感知检索框架，该框架将历史成功与失败经验转化为工具使用型智能体的结构化外部经验。

    arXiv:2609.28003v1 Announce Type: new  Abstract: Small and medium-sized language models offer cost-effective executors for tool-using agents, making them attractive for local and large-scale deployment. However, in long-horizon and stateful environments, they often make structural errors such as missing required observations, performing premature writes, repeating failed calls, and violating action preconditions. These errors can lead to incorrect state updates, policy violations, and costly or irreversible consequences, making reliable tool execution a critical deployment challenge. Existing fine-tuning approaches require substantial data and computation, while flat memory may retrieve failed actions without preserving their causal context or safety conditions. In this paper, we propose FRESH, a Failure-aware Retrieval framework over Experience-Structured Heterogeneous graphs, which transforms historical successes and failures into structured external experience for tool-using agents.
    
[^52]: 面向视觉Transformer特征空间的任务诱导黎曼度量

    Task-Induced Riemannian Metrics for Vision Transformer Feature Spaces

    [https://arxiv.org/abs/2609.27988](https://arxiv.org/abs/2609.27988)

    该论文提出用任务诱导的回拉度量取代ViT特征空间中常用的欧氏距离和余弦相似度，设计无矩阵诊断量 κ_cap(r) 判断低秩近似是否可学习，并提出谱回拉网络（SPN）来学习该度量的低秩版本。

    

    在视觉Transformer（ViT）特征空间上进行操作的方法通常依赖于欧氏距离或余弦相似度。这隐含地假设了每个方向都同等重要，但没有理由相信真实的任务几何具有这种性质。特征空间的任务敏感几何由回拉度量 g(F) = J(F)^⊤ J(F) 给出，其中 J 是解码器输出（该输出被送入任务特定的距离度量）相对于特征的雅可比矩阵。在现代规模下存储完整的 g 是不可行的，而对于诸如深度图这样的稠密输出，甚至连构造 J 都是不切实际的。我们证明了该度量的低秩近似能否被学习取决于模型-解码器组合，并用一个无矩阵的诊断量 κ_cap(r) 来刻画这一点，该诊断量只需少量的雅可比向量积即可计算。对于可行的模型-解码器组合，我们提出了谱回拉网络，它学习该度量的低秩版本……

    arXiv:2609.27988v1 Announce Type: cross  Abstract: Methods operating on Vision Transformer (ViT) feature spaces typically rely on Euclidean distance or cosine similarity. This assumes that every direction is equally meaningful, but there is no reason to believe the true task geometry has this property. The task-sensitive geometry of the feature space is given by the pullback metric $g(F) = J(F)^\top J(F)$, where $J$ is the Jacobian of the decoder's output fed to a task-specific distance, with respect to the features. Storing the full $g$ is infeasible at modern scales, and for dense outputs such as depth maps even forming $J$ is impractical. We show that whether a low-rank approximation of this metric can be learned depends on the model-decoder pair, and we characterize this with a matrix-free diagnostic $\kappa_{cap}(r)$ computable with a low number of Jacobian-vector products. For tractable pairs, we develop the Spectral Pullback Network (SPN), which learns a low-rank version of the 
    
[^53]: PCQC：面向多轮医疗对话的特权反事实问题信用分配

    PCQC: Privileged Counterfactual Question Credit for Multi-Turn Medical Dialogue

    [https://arxiv.org/abs/2609.27987](https://arxiv.org/abs/2609.27987)

    提出PCQC方法，利用训练时的特权患者信息构建反事实答案，使多轮医疗对话的强化学习能够对每个问题（包括从未提出的备选问题）进行信用分配，从而为对话策略训练提供问题级别的反馈。

    

    大型语言模型（LLMs）在医疗问答任务上已取得显著进展，但有效的医疗对话还需要学会提出能够挖掘相关患者信息的问题。为了训练这样的对话策略，一种常见流程是将监督微调与基于最终诊断正确性的强化学习（RL）相结合。然而，这种基于结果的监督无法直接区分各个问题的贡献，也无法为未被执行的备选问题提供问题级别的反馈。为了填补这一空白，我们提出了PCQC（特权反事实问题信用分配），该方法在训练期间利用特权患者信息，从从未被提出的问题中学习。在训练过程中，PCQC通过使用特权患者事实来构建备选问题的答案，使备选问题可以在同一对话状态下直接进行比较。一个冻结的诊断评分器评估每个（结果答案的）诊断效用……（原文摘要在此处截断）

    arXiv:2609.27987v1 Announce Type: new  Abstract: Large language models (LLMs) have made substantial progress on medical question-answering, yet effective medical dialogue also requires learning to ask questions that uncover relevant patient information. To train such dialogue policies, a common pipeline combines supervised fine-tuning with reinforcement learning (RL) based on final diagnostic correctness. However, this outcome-based supervision does not directly distinguish the contributions of individual questions and provides no question-level feedback for unexecuted alternatives. To address this gap, we introduce PCQC (Privileged Counterfactual Question Credit), which uses privileged patient information during training to learn from questions never asked. During training, PCQC makes alternative questions directly comparable at the same dialogue state by using privileged patient facts to construct their answers. A frozen diagnostic scorer evaluates the diagnostic utility of each resu
    
[^54]: 相对放电阶段（RDS）分类：一种实用的电池放电进度指标

    Relative Discharge Stage (RDS) Classification: A Practical Indicator of Battery Discharge Progress

    [https://arxiv.org/abs/2609.27986](https://arxiv.org/abs/2609.27986)

    本文提出相对放电阶段（RDS）这一电池管理指标，将剩余放电状态划分为五个可解释等级，并通过结合SOC估计与轻量级时序学习的物理信息分类框架，在无需未来电流信息的情况下实现对电池放电进度的实用化评估。

    

    在实际电池应用中，由于未来负载曲线未知且高度动态，准确预测剩余放电时间（RDT）极具挑战性。为解决连续RDT回归的不确定性问题，本文提出了相对放电阶段——一种电池管理指标，通过五个可解释的类别（正常、良好、中等、偏低、需要充电）来表征剩余放电状态。与反映当前电荷水平的荷电状态（SOC）不同，RDS刻画的是剩余放电过程，且在推理阶段无需未来电流信息。本文提出了一种物理信息驱动的RDS分类框架，将SOC估计与轻量级时序学习相结合。SOC估计组件包括二阶等效电路模型（ECM）状态与端电压预测、滞回特性与OCV温度校正、电芯核心温度估计以及自适应扩展卡尔曼滤波（AEKF）状态校正，并辅以……（原摘要在此处截断）

    arXiv:2609.27986v1 Announce Type: new  Abstract: Accurate remaining discharge time (RDT) prediction is challenging in real-world battery applications because future load profiles are unknown and highly dynamic. To address the uncertainty of continuous RDT regression, this paper introduces Relative Discharge Stage (RDS), a battery-management indicator that represents the remaining discharge condition using five interpretable classes: Normal, Good, Moderate, Low, and Recharge Required. Unlike state of charge (SOC), which reflects the current charge level, RDS characterizes the remaining discharge process without requiring future-current information during inference. A physics-informed RDS classification framework is proposed, combining SOC estimation with lightweight temporal learning. The SOC-estimation component includes second-order ECM state and terminal-voltage prediction, hysteresis and OCV temperature correction, core-temperature estimation, and AEKF state correction, supported by
    
[^55]: 一类低参数量正交矩阵的黎曼结构与优化

    Riemannian Structure and Optimization for a Class of Low-Parametric Orthogonal Matrices

    [https://arxiv.org/abs/2609.27982](https://arxiv.org/abs/2609.27982)

    本文为一类由块对角因子与固定置换交织而成的低参数量正交矩阵建立了黎曼流形结构，并提出了基于自动微分的高效黎曼优化算法，可应用于最佳矩阵逼近和参数高效微调。

    

    本文研究由块对角因子与固定置换交织而成的矩阵——这是一类灵活的结构化矩阵族。该类矩阵近来因其在表达能力与计算效率之间的良好权衡而在深度学习架构中受到关注，但针对它的高效计算策略仍有待探索。我们通过黎曼几何的视角来处理这一问题，并考察了该类矩阵在何种条件下具有光滑流形结构。对于实际应用中重要的正交双因子矩阵情形，我们推导了基本的黎曼工具，并提出了实现这些工具的高效算法。这些算法利用自动微分技术，支持每个因子内部的参数共享，并避免了显式构造稠密矩阵。我们在黎曼优化框架下，针对最佳矩阵逼近问题和参数高效微调任务对这些算法进行了测试。

    arXiv:2609.27982v1 Announce Type: cross  Abstract: In this paper, we are concerned with matrices formed by block-diagonal factors interleaved with fixed permutations -- a flexible family of structured matrices. This class has recently drawn interest in deep learning architectures for its balanced expressivity-efficiency trade-off, yet efficient computational strategies for working with it remain to be found. We approach this problem through Riemannian geometry and examine under what conditions this class admits a smooth manifold structure. For the practically important case of orthogonal two-factor matrices, we derive the essential Riemannian tools and propose efficient algorithms for their implementation. The algorithms leverage automatic differentiation, support parameter sharing within each factor, and avoid explicit dense matrix construction. We test them within the Riemannian optimization framework on the best matrix approximation problem and for parameter-efficient fine-tuning of
    
[^56]: 风险可控的KV缓存淘汰：从内存预算到风险目标

    Risk-Controlled KV-Cache Eviction: From Memory Budgets to Risk Targets

    [https://arxiv.org/abs/2609.27981](https://arxiv.org/abs/2609.27981)

    该论文将KV缓存淘汰从平均内存预算视角重新表述为部署风险控制问题，提出一种与压缩器无关的事后认证程序，通过有限样本保证选择保留策略，确保任务效用实质性退化事件的发生频率满足部署指定的风险目标与置信度要求。

    

    KV缓存淘汰通常通过平均质量-内存权衡来评估，然而较小的平均损失可能掩盖那些效用严重退化的请求。我们将淘汰问题重新表述为一个部署风险控制问题：当淘汰相对于同一请求的完整KV推理使任务效用降低超过部署指定的容限时，即发生实质性退化，而部署风险被定义为这类事件在总体中的发生频率。给定一个指定目标风险水平和置信度要求的可靠性契约，我们使用一种与压缩器无关的事后认证程序，从校准数据中选择具有有限样本保证的保留策略，当没有压缩策略通过认证时回退到完整KV。在多种淘汰方法、Llama和Mistral模型以及LongBench和RULER-32K基准测试上，相同的契约支持显著不同的淘汰水平：在Llama上，该认证使SnapKV在75%保留率下...

    arXiv:2609.27981v1 Announce Type: new  Abstract: KV-cache eviction is typically evaluated through average quality-memory trade-offs, yet a small average loss can hide requests whose utility degrades materially. We reformulate eviction as a deployment risk-control problem: a material degradation occurs when eviction lowers task utility by more than a deployment-specified tolerance relative to full-KV inference on the same request, and deployment risk is the population frequency of such events. Given a reliability contract specifying a target risk level and confidence requirement, we use a compressor-agnostic post-hoc certification procedure to select a retention policy from calibration data with a finite-sample guarantee, falling back to full KV when no compressed policy is certified. Across multiple eviction methods, Llama and Mistral models, and LongBench and RULER-32K, the same contract supports substantially different levels of eviction: on Llama, it certifies SnapKV at 75% retentio
    
[^57]: 减少六层：基于无标签恢复的Whisper编码器剪枝

    Six Layers Less: Encoder Pruning for Whisper with Label-Free Recovery

    [https://arxiv.org/abs/2609.27980](https://arxiv.org/abs/2609.27980)

    该论文提出通过留一层法依据词错误率变化对Whisper编码器层进行排序，剪掉影响最小的六层（占编码器的18.5%），无需自定义推理代码，并利用无标签单语语音数据进行蒸馏以恢复性能。

    

    对大型预训练的基于transformer的ASR模型（如OpenAI的Whisper）进行剪枝已获得广泛应用，因为对解码器进行剪枝可带来显著的端到端转录加速。例如，whisper-large-v3-turbo变体将解码器从32层减少到4层，而Distill-Whisper同样将解码器减少到仅2层。尽管在减小编码器尺寸方面已有一些关注，但尚无方法被广泛采用。这可能是由于需要自定义推理实现才能利用压缩后的模型。我们提出了一种方法，通过留一层法评估词错误率（WER）的变化来对编码器层进行排序，移除导致变化最小的六层，相当于编码器堆栈的18.5%。剪枝后的模型无需自定义推理代码，因为它只是一个层数更少、更浅的编码器。我们进一步使用无标签的单语语音数据进行蒸馏……（摘要原文在此处截断）

    arXiv:2609.27980v1 Announce Type: new  Abstract: Pruning large pre-trained transformer-based ASR models such as OpenAI's Whisper has seen great adoption, as pruning the decoder led to significant end-to-end transcription speedups. For instance, the {\tt whisper-large-v3-turbo} variant reduced the decoder from 32 to 4 layers, while Distill-Whisper similarly reduced the decoder to only 2 layers. Although some attention has been put towards reducing the size of the encoder, no approach has seen wide adoption. This could be due to the need for custom inference implementations to take advantage of the compressed model. We present an approach that ranks encoder layers by the leave-one-layer-out change in Word Error Rate (WER). The six layers that cause the least change are removed, corresponding to $18.5\%$ of the encoder stack. The pruned model requires no custom inference code as it is simply a more shallow encoder with fewer layers. We further distill using unlabeled monolingual speech da
    
[^58]: 连续标签偏移下的共形贝叶斯：敏感性分析与精确有效性的局限

    Conformal Bayes under Continuous Label Shift: Sensitivity Analysis and the Limits of Exact Validity

    [https://arxiv.org/abs/2609.27976](https://arxiv.org/abs/2609.27976)

    该论文提出联合倾斜敏感性共形贝叶斯（JTS-CB/JTS-SCB），通过对预设的合理倾斜集合进行联合敏感性分析来应对连续标签偏移，同时揭示了精确有限样本有效性取决于密度比尾部行为的根本局限。

    

    共形贝叶斯将贝叶斯后验预测分数与共形校准相结合，但在连续标签偏移下，分数和校准权重都依赖于未知的响应边缘密度比。现有方法通常从伪标签或预测样本中估计单个偏移参数并将其代入校准。与之不同，我们提出了联合倾斜敏感性共形贝叶斯，它在预先指定的合理倾斜集合上进行敏感性分析；其分割共形实现为 JTS-SCB。每个倾斜共同决定贝叶斯共形分数和共形重要性权重。JTS-SCB 在候选倾斜上形成一个有界的敏感性包络，但其仅进行校准的构造并不能继承精确的有限样本加权共形保证。因此，我们研究了一个单独的候选加权精确对应方法，并证明其有用性在很大程度上取决于尾部行为。对于标量线性……（原文摘要在此处不完整）

    arXiv:2609.27976v1 Announce Type: cross  Abstract: Conformal Bayes combines Bayesian posterior predictive scores with conformal calibration, but under continuous label shift both the score and calibration weight depend on the unknown response-marginal density ratio. Existing methods typically estimate one shift parameter from pseudo-labels or predictive samples and plug it into calibration. We instead propose Joint Tilt-Sensitivity Conformal Bayes (JTS-CB), which performs sensitivity analysis over a prespecified set of plausible tilts; its split-conformal realization is JTS-SCB. Each tilt jointly determines the Bayesian conformal score and conformal importance weight. JTS-SCB forms a bounded sensitivity envelope over candidate tilts, but its calibration-only construction does not inherit the exact finite-sample weighted-conformal guarantee. We therefore study a separate candidate-weighted exact counterpart and show that its usefulness depends sharply on tail behavior. For scalar linear
    
[^59]: 线性RNN缩放定律：更长的序列何时胜过更多的序列

    Linear RNN Scaling Laws: When Longer Sequences Beat More Sequences

    [https://arxiv.org/abs/2609.27964](https://arxiv.org/abs/2609.27964)

    该论文在教师-学生理论框架下首次为线性RNN语言模型推导出显式的近似、优化与统计缩放定律，并揭示了在特定谱指数条件下，增加序列长度比增加序列数量能更有效地提升模型性能。

    

    自回归语言模型的经验缩放定律将预测损失与模型规模、数据规模和优化计算量联系起来，但其在序列预训练设置中的理论起源仍然缺乏深入理解。我们在一个可解析的教师-学生模型中研究该问题：一个稳定的潜在线性RNN生成轨迹，一个采用草图（sketch）方法的线性递归学生模型通过带安全防护的全批量WSD梯度下降在下一词元预测任务上训练。草图维度 $M$ 扮演模型规模的角色，而 $N$ 条长度为 $P$ 的独立轨迹提供训练词元。我们允许创新协方差与初始化协方差具有不同的幂律指数 $\alpha$ 和 $\theta$。由此产生的设计谱给出了由谱交叉点分隔的显式近似、优化和统计缩放定律。当 $\theta\ge\alpha$ 时，原始的单尺度速率 $M^{1-\beta_\alpha}$、$R^{(1-\beta_\alpha)/\alpha}$……（摘要在此处被截断）

    arXiv:2609.27964v1 Announce Type: new  Abstract: Empirical scaling laws for autoregressive language models relate prediction loss to model size, data size, and optimization compute, but their theoretical origin is still poorly understood in sequential pretraining settings. We study this question in a tractable teacher--student model where a stable latent linear RNN generates trajectories and a sketched linear recurrent student is trained by safeguarded full-batch WSD gradient descent on next-token prediction. The sketch dimension $M$ plays the role of model size, while $N$ independent trajectories of length $P$ provide the training tokens. We allow the innovation and initialization covariances to have different power-law exponents $\alpha$ and $\theta$. The induced design spectrum produces explicit approximation, optimization, and statistical scaling laws separated by spectral crossovers. When $\theta\ge\alpha$, the original one-scale rates $M^{1-\beta_\alpha}$, $R^{(1-\beta_\alpha)/\a
    
[^60]: I-SplineFlow：学习单调样条随机插值调度器以实现少步生成

    I-SplineFlow: Learning Monotone Spline Stochastic Interpolant Schedulers for Few-Step Generation

    [https://arxiv.org/abs/2609.27963](https://arxiv.org/abs/2609.27963)

    该论文提出I-SplineFlow，用积分单调样条（I-splines）参数化随机插值调度器，将多项式阶数与混合权重数量解耦并利用紧支撑实现局部调控，从而在少步生成中获得更具表达力且优化更稳定的采样轨迹调度器。

    

    使用预训练扩散模型和流模型进行少步生成时，可以通过轻量级训练来加速，这种训练优化的是采样轨迹而非网络本身。最近的一种方法将随机插值（SI）调度器参数化为一条平滑曲线，其控制点确保SI调度器必须满足的三个性质：固定的边界条件、单调的信噪比（SNR）以及可微性。现有的参数化方法使用全局支撑的多项式基，其中每个控制点都会移动整条曲线，且更高的表达能力需要更高的多项式阶数，这在优化过程中将调度中相距较远的区域耦合在一起。我们提出I-SplineFlow，它使用积分单调样条（I-splines）来参数化调度器。I-splines将多项式阶数与混合权重的数量解耦，因此可以在固定权重数量的情况下为每个模型灵活选择支撑宽度和平滑度，并且紧支撑的特性能够实现局部化的调控与更稳定的优化。

    arXiv:2609.27963v1 Announce Type: new  Abstract: Few-step generation with pretrained diffusion and flow models can be accelerated by lightweight training that optimizes the sampling trajectory rather than the network. A recent approach parameterizes the stochastic interpolant (SI) scheduler as a smooth curve whose control points enforce the three properties an SI scheduler must satisfy: fixed boundary conditions, a monotone signal-to-noise ratio (SNR), and differentiability. Existing parameterizations use globally supported polynomial bases, where every control point moves the whole curve and higher expressiveness needs a higher degree, which couples distant regions of the schedule during optimization. We introduce \emph{I-SplineFlow}, which parameterizes the scheduler with integrated monotone splines (I-splines). I-splines decouple the polynomial degree from the number of mixture weights, so support width and smoothness can be chosen per model at a fixed weight count, and the compactl
    
[^61]: ScoutNeRV：通过ScoutNet实现基于网格的视频隐式神经表示（INR）快速编码

    ScoutNeRV: Rapid Encoding of Grid-Based Video INRs via ScoutNet

    [https://arxiv.org/abs/2609.27958](https://arxiv.org/abs/2609.27958)

    ScoutNeRV提出一种内容自适应初始化框架，利用轻量级侦察网络分析少量采样帧并从预训练专家记忆库中选取参数来初始化分层视频INR，从而显著降低视频编码的逐视频优化成本。

    

    隐式神经表示（INR）已成为一种有前景的视频压缩范式，能够提供紧凑的神经表示以及灵活的空间和时间重建能力。以HiNeRV为代表的分层网格架构实现了优异的率失真性能，但需要对每个视频进行大量优化，导致编码成本高昂。为解决这一局限，我们提出ScoutNeRV，这是一个内容自适应初始化框架，用于加速分层视频INR的优化。ScoutNeRV采用一个轻量级的离线训练侦察网络，通过分析少量采样帧，借助硬路由机制从专家记忆库中选取合适的预训练专家模型，随后将所选专家的分层网格和解码器参数迁移过来，用于初始化目标HiNeRV模型，再进行针对特定视频的微调。在未见过的ReadySetGo序列上，ScoutNeRV实现了较高的初始PSNR

    arXiv:2609.27958v1 Announce Type: cross  Abstract: Implicit neural representations (INRs) have emerged as a promising paradigm for video compression, providing compact neural representations with flexible spatial and temporal reconstruction. Hierarchical grid-based architectures such as HiNeRV achieve strong rate--distortion performance, but require extensive per-video optimization, resulting in high encoding costs. To address this limitation, we propose ScoutNeRV, a content-adaptive initialization framework for accelerating the optimization of hierarchical video INRs. ScoutNeRV employs a lightweight, offline-trained scout network that analyzes a small number of sampled frames and selects a suitable pre-trained expert from a memory bank through hard routing. The hierarchical grid and decoder parameters of the selected expert are then transferred to initialize the target HiNeRV model before video-specific fine-tuning. On the unseen ReadySetGo sequence, ScoutNeRV achieves an initial PSNR
    
[^62]: CS-WCP：针对LLM裁判流量漂移与不确定组比例的鲁棒保形集

    CS-WCP: Robust Conformal Sets for LLM-Judge Traffic Shifts with Uncertain Group Proportions

    [https://arxiv.org/abs/2609.27955](https://arxiv.org/abs/2609.27955)

    提出CS-WCP方法，通过为源域和目标域组比例构造同时置信区间并对所有兼容比例向量取加权保形集的并集，在LLM裁判部署流量漂移且组比例只能从无标签样本估计的情况下实现鲁棒的分布无关覆盖保证。

    

    当部署流量改变了任务或策略组的流行比例时，由LLM裁判构建的预测集可能出现覆盖不足。加权保形预测在密度比已知的情况下于协变量偏移下是精确的，但组比例通常必须从有限的无标签样本中估计。我们提出了置信集加权保形预测（CS-WCP），该方法为源域和目标域的组质量构造同时精确的置信区间，并返回所有兼容比例向量上加权保形集的并集。对于固定或独立学习的有限划分，CS-WCP可获得至少 1-α-δ_w-τ_A-κ 的覆盖率，其中 τ_A 度量单元内的协变量失配，κ 度量条件偏移。一个线性端点规则可在 O(G|Y|) 时间内计算鲁棒并集。在336个构建的共享支撑流量漂移实验中，CS-WCP达到0.973的平均覆盖率，仅13个点失败，相比之下为0.954和44个失败点。

    arXiv:2609.27955v1 Announce Type: new  Abstract: Prediction sets built from an LLM judge can undercover when deployment traffic changes the prevalence of task or policy groups. Weighted conformal prediction is exact under covariate shift when the density ratio is known, but group proportions must usually be estimated from finite unlabeled samples. We introduce confidence-set weighted conformal prediction (CS-WCP), which constructs simultaneous exact intervals for source and target group masses and returns the union of weighted conformal sets over every compatible ratio vector. For a fixed or independently learned finite partition, CS-WCP attains coverage at least 1-alpha-delta_w-tau_A-kappa, where tau_A measures within-cell covariate mismatch and kappa measures conditional shift. A linear endpoint rule computes the robust union in O(G|Y|) time. Across 336 constructed shared-support traffic shifts, CS-WCP reaches 0.973 mean coverage with 13 point failures, compared with 0.954 and 44 fai
    
[^63]: 当准确率差距无法提供证明时：对大语言模型评判者跨域重校准的审计

    When Accuracy Gaps Fail to Certify: Auditing Cross-Domain Recalibration of LLM Judges

    [https://arxiv.org/abs/2609.27954](https://arxiv.org/abs/2609.27954)

    该论文通过涵盖13个评判者、2个生成器、8个领域和1,176个迁移设置的大规模实验证明，LLM评判者的源-目标准确率差距既无法可靠预测跨域重校准的迁移结果，也无法作为校准失效的证明，并提出了一种基于有限样本证书、使用互不相交审计标签的无泄漏审计方法。

    

    为大语言模型（LLM）评判者在某一任务上拟合的标量重校准映射，在任务分布发生变化时可能会失效，而源域与目标域之间的准确率差距常被当作这种失效的代理指标。我们在十三个评判者、两个生成器、八个领域以及1,176个预先声明的迁移设置上，测试了这一差距能够预测什么、又能够证明什么。在考虑平均分数偏移之后，该差距给出了目标域校准误差的总体下界，然而相同的差距却可能导致截然相反的迁移结果。精确的重要性加权可以在协变量偏移下恢复目标域的适当损失，因此估计加权流程的失败本身并不能确立条件偏移的存在。有限样本同时下界证书将总体界转化为一个单边拒绝规则，其使用与评估结果互不相交的审计标签。无泄漏的差距相关性为0.25（95%置信区间[-0.09, 0.55]），在第二个生成器上降至0.09，并且不……（原文摘要在此处截断）

    arXiv:2609.27954v1 Announce Type: new  Abstract: A scalar recalibration map fitted for an LLM judge on one task can fail when the task distribution changes, but the source-target accuracy gap is often treated as a proxy for that failure. We test what this gap can predict and what it can certify across thirteen judges, two generators, eight domains, and 1,176 predeclared transfers. After accounting for mean score shift, the gap yields a population lower bound on target calibration error, yet identical gaps can induce opposite transfer outcomes. Exact importance weighting recovers target proper loss under covariate shift, so failure of an estimated weighting pipeline does not by itself establish conditional shift. A finite-sample simultaneous lower certificate converts the population bound into a one-sided rejection rule using audit labels disjoint from evaluation outcomes. The leak-free gap correlation is 0.25 (95% CI [-0.09, 0.55]), falls to 0.09 on the second generator, and does not s
    
[^64]: 大语言模型推荐重排序的召回率上限

    The Recall Ceiling of LLM Recommendation Reranking

    [https://arxiv.org/abs/2609.27953](https://arxiv.org/abs/2609.27953)

    该论文揭示了LLM推荐重排序评估中“预言机协议”会高估性能92-95%，指出由于现实检索仅能覆盖2-19%的相关物品，召回率上限为任何封闭候选集重排序器设定了确定性的NDCG上界，且在现实条件下各类优化策略均无法显著超越协同过滤基线。

    

    某些基于大语言模型（LLM）的推荐重排序器是在一种“预言机”（oracle）协议下进行评估的，该协议通过将真实目标项注入候选列表、或将其与采样的负例一起打分，保证了真实目标项必然出现在被评分的集合中。我们在三个主要的Amazon数据集上证明，这种协议会将现实的NDCG@10高估92%至95%。其根本原因在于一个召回率上限：在三个领域的八个数据集上，当K=100时，现实的检索只能覆盖2%至19%的相关物品，这为任何封闭候选集重排序器的top-k NDCG施加了确定性的上界。在留一法评估下，有 E[NDCG@k] ≤ Recall@|W_π|，其中W_π是重排序器的候选窗口。在现实检索条件下，在我们的主要Amazon数据集上，所有经过测试的优化策略——包括提示工程、模型扩展等——均未能显著超越协同过滤基线。

    arXiv:2609.27953v1 Announce Type: cross  Abstract: Some LLM-based recommendation rerankers are evaluated under an oracle protocol that guarantees the ground-truth item is present in the scored set, either by injecting it into the candidate list or by scoring it against sampled negatives. Across three primary Amazon datasets, we show that this protocol overestimates realistic NDCG@10 by 92--95%. The cause is a recall ceiling: realistic retrieval covers only 2--19% of relevant items at $K=100$ across eight datasets in three domains, imposing a deterministic upper bound on any closed-candidate reranker's top-$k$ NDCG. Under leave-one-out evaluation, $\mathbb{E}[\mathrm{NDCG}@k] \leq \mathrm{Recall}@|W_\pi|$, where $W_\pi$ is the reranker's candidate window.   Under realistic retrieval, none of the tested optimisation strategies significantly improves over the collaborative-filtering baseline on our primary Amazon datasets. These strategies include prompt engineering, model scaling over a 
    
[^65]: 增强资源受限环境下的多类别恶意软件分类

    Enhancing Multiclass Malware Classification in Resource-Constrained Environments

    [https://arxiv.org/abs/2609.27950](https://arxiv.org/abs/2609.27950)

    本文提出了一种结合SMOTE过采样的LightGBM轻量级机器学习模型，能够在物联网等资源受限环境中高效且准确地实现多类别恶意软件分类。

    

    诸如勒索软件、间谍软件、木马等多类别恶意软件攻击的出现，对网络安全构成了日益严重的威胁，尤其是在物联网设备等资源受限的环境中。现有的机器学习模型在二分类恶意软件检测方面已达到近乎完美的准确率，但在恶意软件家族和单个恶意软件的分类方面仍存在不足。此外，这些多类别恶意软件攻击的复杂性给资源受限环境中的检测带来了重大挑战，因为多类别检测通常需要较高的计算能力。本研究通过提升多类别恶意软件分类的检测准确率，并开发一种能够在资源受限设备上高效运行的轻量级模型来弥补这一空白。在本文中，我们提出了一种鲁棒的轻量级机器学习模型，该模型采用结合SMOTE过采样的LightGBM分类器……

    arXiv:2609.27950v1 Announce Type: cross  Abstract: The emergence of multi-class malware attacks such as ransomware, spyware, trojans, etc., presents an increasing and serious threat to cybersecurity, particularly in resourceconstrained environments like IoT devices. Existing machine learning models have achieved nearly perfect accuracy in binary malware classification but fall short in terms of classifying malware families and individual malware. Additionally, the complexity of these multi-class malware attacks presents a significant challenge of detection in resource-constrained environments, as multi-class detection usually requires high computational capability. This research bridges the gap by enhancing the detection accuracy of multi-class malware classification as well as developing a lightweight model that can run efficiently on resource-constrained devices. In this paper, we propose a robust, lightweight machine learning model featuring LightGBM classifier with SMOTE oversampli
    
[^66]: 从情感分类到可操作且负责任的反馈：2015-2026年学生评教中自然语言处理研究的范围综述与证据图谱

    From Sentiment Classification to Actionable and Responsible Feedback: A Scoping Review and Evidence Map of NLP in Student Evaluation of Teaching, 2015-2026

    [https://arxiv.org/abs/2609.27939](https://arxiv.org/abs/2609.27939)

    该范围综述通过对2015-2026年间421项研究的技术演进与价值维度进行证据图谱映射，揭示了学生评教NLP研究中从技术演示到面向最终用户的可操作应用之间存在约50个百分点的显著断层。

    

    将自然语言处理（NLP）应用于开放式教学评价评论（学生评教，SET）的研究随着该领域的技术演进不断发展——从词库和传统分类器到Transformer和大语言模型（LLM）——但这种技术上的多样化是否带来了相应的教育价值和证据稳健性的提升，目前尚不明确。本范围综述（遵循PRISMA-ScR规范）沿技术轴（RQ1）和四个价值维度（RQ2-RQ5）对421项研究（2015-2026年，2026年为部分数据）进行了系统映射。研究采用双向相互盲评的LLM筛选结合抽样人工裁决的方式对七个提取领域进行编码，并在综合阶段进行针对性的代码簿边界审查。联合图谱中最尖锐的量化差距是可操作性断层：达到已验证输出或更强水平（A2+：258/421；61.3%）与达到面向预期用户评估或更强水平（A3+：49/421；11.6%）之间存在49.7个百分点的落差。情感分析……（原文摘要在此处截断）

    arXiv:2609.27939v1 Announce Type: new  Abstract: Natural language processing (NLP) applied to open-ended teaching-evaluation comments (Student Evaluation of Teaching, SET) has tracked the field's technical evolution--from lexicons and conventional classifiers to transformers and large language models (LLMs)--but it is not evident that this technical diversification has been accompanied by corresponding gains in educational value and robustness of the evidence. This scoping review (PRISMA-ScR) maps 421 studies (2015-2026, 2026 partial) along a technical axis (RQ1) and four value dimensions (RQ2-RQ5). Dual mutually blinded LLM screening with sampled human adjudication coded seven extraction domains, with targeted codebook-boundary review at synthesis. The joint map's sharpest quantified gap is the actionability discontinuity: demonstrated output or stronger (A2+: 258/421; 61.3%) versus intended-user evaluation or stronger (A3+: 49/421; 11.6%), a 49.7 percentage-point drop. Sentiment anal
    
[^67]: 质量胜于数量：基于特征工程的非法比特币流半监督检测

    Quality over Quantity: Semi-Supervised Detection of Illicit Bitcoin Flows via Feature Engineering

    [https://arxiv.org/abs/2609.27936](https://arxiv.org/abs/2609.27936)

    该论文基于1.63亿笔比特币交易提出半监督学习框架检测共享发送混合器中的非法资金流，证明其成功关键在于高保真特征工程带来的数据质量而非数据量。

    

    检测非法加密货币交易一直受到极端类别不平衡、对抗性混淆手段以及可靠标签稀缺的阻碍。虽然半监督学习（SSL）通过利用未标记数据提供了一种有前景的解决方案，但我们表明其成功并非仅由数据量保证，而是取决于数据质量。我们提出了一个用于检测共享发送混合器（SSM）交易中非法比特币流的半监督学习框架，该框架建立在包含1.63亿笔交易的综合历史数据集之上。我们的主要结论是，半监督学习的成功取决于数据质量而非数据量：诸如KeyLinker地址聚类和共享发送解缠（SSU）复杂度指标等高保真特征在未标记数据上实现了0.84的F1分数。最后，我们通过实证表明，像一次性找零（OTC）这样常见但数量众多的启发式特征会引入噪声，而战略性地依赖更高保真的特征（如……）

    arXiv:2609.27936v1 Announce Type: new  Abstract: Detecting illicit cryptocurrency transactions is hampered by extreme class imbalance, adversarial obfuscation, and a scarcity of reliable labels. While semi-supervised learning (SSL) offers a promising solution by leveraging unlabeled data, we show that its success is not guaranteed by data volume alone but is contingent on data quality. We introduce an SSL framework for detecting illicit Bitcoin flows in Shared Send Mixers (SSM) transactions, built on a comprehensive historical dataset comprising 163 million transactions. Our main conclusion is that the success of SSL depends on data quality rather than volume: high-fidelity features such as KeyLinker address clustering and Shared Send Untangling (SSU) complexity metrics achieve an F1 score of 0.84 on unlabeled data. Finally, we empirically show that common heuristics like One-Time Change (OTC), though abundant, introduce noise, while strategic reliance on higher-fidelity features like 
    
[^68]: 二值量化神经网络训练在以输入和输出维度为参数时是W[1]-难的

    Binary Quantized Neural Network Training Is W[1]-Hard Parameterized by Input and Output Dimensions

    [https://arxiv.org/abs/2609.27932](https://arxiv.org/abs/2609.27932)

    本文证明了二值量化神经网络训练在仅以输入与输出维度之和为参数时是W[1]-难的，即使在零误差、样本构成前缀链且偏置固定为零的最简情形下依然成立，从而否定了Ganian等人留下的开放问题。

    

    Ganian等人（ICLR 2026）证明了量化神经网络训练在以架构树宽、输入维度 $\alpha$ 和输出维度 $\omega$ 联合参数化时是固定参数可解的，并留下了仅以 $\alpha+\omega$ 参数化时是否固定参数可解的开放问题。我们证明了2-QNNT在以 $\alpha+\omega$ 参数化时是W[1]-难的。该困难性在数据集 $D_k=\{(\xi^{(r)},\xi^{(r)}):0\le r\le k\}$ 上零误差的情况下即已成立，其中每个输入等于其目标值，$|D_k|=\alpha=\omega=k+1$，且样本构成逐坐标的前缀链。当每个非源节点偏置固定为零时，该困难性同样成立。在指数时间假设（ETH）下，对任何可计算函数 $f$，不存在运行时间为 $f(\alpha+\omega)|I|^{o(\alpha+\omega)}$ 的算法。该归约从有向无环图的边不相交路径问题出发，利用有向线图将边容量转换为顶点容量，并将结果规范化为有效的分层架构。关键的结（原文摘要在此处被截断）

    arXiv:2609.27932v1 Announce Type: new  Abstract: Ganian et al. (ICLR 2026) proved that quantized neural network training is fixed-parameter tractable when parameterized jointly by architecture treewidth, input dimension $\alpha$, and output dimension $\omega$, and left open whether $\alpha+\omega$ alone yields fixed-parameter tractability. We prove that 2-QNNT is W[1]-hard parameterized by $\alpha+\omega$. The hardness already holds with zero error on $D_k=\{(\xi^{(r)},\xi^{(r)}):0\le r\le k\}$, where every input equals its target, $|D_k|=\alpha=\omega=k+1$, and the examples form a coordinatewise prefix chain. It also holds when every non-source bias is fixed to zero. Under the Exponential Time Hypothesis, no algorithm runs in $f(\alpha+\omega)|I|^{o(\alpha+\omega)}$ for any computable $f$. The reduction starts from DAG edge-disjoint paths, converts edge capacity to vertex capacity with a directed line graph, and normalizes the result into a valid layered architecture. The key structur
    
[^69]: 具有高斯过程分裂的狄利克雷过程树混合：一个具有后验收缩速率的贝叶斯非参数框架

    Dirichlet Process Mixtures of Trees with Gaussian Process Splits: A Bayesian Nonparametric Framework with Posterior Contraction Rate

    [https://arxiv.org/abs/2609.27930](https://arxiv.org/abs/2609.27930)

    该论文提出了一种基于狄利克雷过程先验和由高斯过程后验预测驱动分裂规则的贝叶斯非参数回归树混合框架，统一了CART、BART、随机森林和提升方法，并证明了在真实回归函数仅连续的宽松条件下Hellinger距离上 $n^{-1/4}$ 的后验收缩速率。

    

    我们提出了一种贝叶斯非参数回归树混合模型，对树-参数对施加狄利克雷过程先验，从而实现集成规模的数据驱动选择，并统一了CART、BART、随机森林和提升方法。一种新颖的分裂规则由每个终端节点内高斯过程的后验预测驱动，可生成灵活、平滑的决策边界；值得注意的是，高斯过程密度在GROW/PRUNE移动的Metropolis-Hastings比率中恰好完全相消，从而保证了计算可行性。我们提供了一个用于后验预测推断的精确吉布斯采样器，通过随机树遍历来传播不确定性。一个并行MPI实现将独立的树更新分配到多个处理器上，实现了可观的速度提升。在仅要求真实回归函数连续的条件下（允许模型误设），我们通过恒等式 $h(\Theta)=0$ 证明了在Hellinger距离下以 $n^{-1/4}$ 速率的后验一致性。在Friedman数据集上的模拟实验……

    arXiv:2609.27930v1 Announce Type: cross  Abstract: We propose a Bayesian nonparametric mixture of regression trees with a Dirichlet process prior over tree-parameter pairs, enabling data-driven selection of ensemble size and unifying CART, BART, random forests, and boosting. A novel splitting rule driven by the posterior predictive of a Gaussian process within each terminal node generates flexible, smooth decision boundaries; remarkably, the GP density cancels exactly in the Metropolis--Hastings ratio for GROW/PRUNE moves, ensuring computational feasibility. An exact Gibbs sampler for posterior predictive inference propagates uncertainty through random tree traversal. A parallel MPI implementation distributes independent tree updates across processors, achieving adequate speedups. We prove posterior consistency at rate $n^{-1/4}$ in Hellinger distance under only continuity of the true regression function, allowing misspecification, via the identity $h(\Theta)=0$. Simulations on Friedma
    
[^70]: 离散度与规模：决定测试时预算分配是否划算的因素

    Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays

    [https://arxiv.org/abs/2609.27917](https://arxiv.org/abs/2609.27917)

    本文通过预先注册的验证性实验发现，工作负载中实例难度的离散程度是决定测试时预算重新分配是否划算的关键属性，并且即使计入分配策略自身消耗的预算成本，该策略依然可能带来收益。

    

    神经组合优化求解器为每个实例生成多个候选解并报告其中找到的最优解，无论实例难度如何，都为每个实例分配相同的采样预算。一项配套研究表明，将固定预算向更难的实例重新分配可以提升解的质量，但这种改进的标准度量方式存在偏差：在相同数据上既决定分配又评估该分配，即使实际没有收益，也可能制造出表面上的收益。这就留下了两个未解决的问题：工作负载的什么特性决定了重新分配是否值得做，以及一个花费部分预算来决定如何分配其余预算的策略，在计入该成本之后是否仍然划算。本文通过预先注册的验证性实验——在数据收集之前就固定了分析方法和判定标准——在三个独立训练的求解器以及两种在训练分布上构造更难工作负载的方式上回答了这两个问题。

    arXiv:2609.27917v1 Announce Type: new  Abstract: Neural combinatorial optimization solvers generate many candidate solutions per instance and report the best one found, using the same sample budget for every instance regardless of difficulty. A companion study showed that reallocating a fixed budget toward harder instances can improve solution quality, but that the standard way of measuring this improvement is biased: deciding an allocation and evaluating it on the same data can manufacture an apparent gain even when none exists. This left open what property of a workload determines whether reallocation is worth doing, and whether a policy that spends part of the budget to decide how to allocate the rest still pays once that cost is counted.   This paper answers both questions through pre-registered confirmatory experiments -- analysis and verdict criteria fixed before data collection -- across three independently trained solvers and two ways of constructing harder workloads on the tra
    
[^71]: 冲突专家的可靠融合

    Reliable Fusion of Conflicting Experts

    [https://arxiv.org/abs/2609.27913](https://arxiv.org/abs/2609.27913)

    提出一种基于概率电路的动态融合框架，利用上下文相关的可信度估计来可靠聚合多个黑盒专家（如大语言模型）的意见，无需访问专家内部表示或再训练，即可在冲突场景下显著提升预测性能与决策可靠性。

    

    我们研究了在噪声较大、易产生冲突的环境中聚合多个黑盒专家意见的问题，其中专家的可靠性会因输入不同而变化。静态聚合方法（如多数投票）无法捕捉这种变异性，在意见分歧时往往产生不可靠的结果。我们提出了一种易于处理的、基于概率电路的融合框架，利用上下文特定的可信度估计动态地结合专家响应，实现有原则且可靠的推理。该框架对底层专家模型无关，无需访问其内部表示，也无需任何重新训练。我们在选择题问答任务上使用多个大语言模型作为专家进行实证验证，与单个模型和静态集成基线进行比较。我们的方法持续提升预测性能，并在冲突情形下产生更可靠的决策，凸显了该框架的有效性。

    arXiv:2609.27913v1 Announce Type: new  Abstract: We study the problem of aggregating opinions from multiple black-box experts in noisy, conflict-prone settings where expert reliability varies across inputs. Static aggregation methods, such as majority voting, fail to capture this variability and often yield unreliable outcomes under disagreement. We propose a tractable, probabilistic-circuit-based fusion framework that dynamically combines expert responses using context-specific credibility estimates, enabling principled and reliable reasoning. The framework is agnostic to the underlying experts and does not require access to their internal representations or any retraining. We empirically validate our approach on multiple-choice question answering tasks using multiple LLMs as experts, comparing against individual models and static ensemble baselines. Our method consistently improves predictive performance and produces more reliable decisions under conflict, highlighting the effectiven
    
[^72]: 全局树预测模型在层级聚合层面的崩溃：一种五面板失效特征刻画

    Global tree forecasters collapse at the hierarchical aggregate: a five-panel failure characterization

    [https://arxiv.org/abs/2609.27912](https://arxiv.org/abs/2609.27912)

    本文首次系统刻画了全局梯度提升树模型在预测层级聚合值时的崩溃失效——当聚合总量远超训练范围时低估可达496倍，并指出逐序列缩放等简单预处理即可防止该失效。

    

    全局预测模型汇集大量时间序列并学习一个共享的函数，梯度提升树是其中最常见的形式。我们测量了这种设计的一种失效模式，据我们所知此前尚未被记录过：在层级的各个单独序列上训练一个全局树模型，然后让它预测层级聚合值。该聚合值远远超出模型的训练范围，预测随之崩溃。在我们的生产部署中，模型对总量的预测低估了30-50倍；在一个公开的M5复现实验中，低估高达496倍。其机制是已知的：超出训练范围后，树模型只会预测一个常数，而这种失效在聚合层面显现，因为总量远大于每一条训练序列。解决方法并不新鲜：逐序列缩放（Montero-Manso和Hyndman于2021年推荐的预处理步骤）即可防止这种崩溃，加权聚合层训练行和季节差分也同样有效。我们的贡献在于对失效特征的系统性刻画。该崩溃可复现……

    arXiv:2609.27912v1 Announce Type: new  Abstract: Global forecasting models pool many series and learn one shared function. Gradient-boosted trees are their most common form. We measure a failure of this design that has not, to our knowledge, been documented. Train a global tree on the individual series of a hierarchy, then ask it for the hierarchical aggregate. The aggregate sits far outside the model's training range, and the forecast collapses. The model under-predicts the total by 30-50x in our production deployment, and by up to 496x in a public M5 reconstruction. The mechanism is known: beyond its training range, a tree predicts a constant. It surfaces at the aggregate because the total dwarfs every training series. The cure is not new. Per-series scaling, the preprocessing step that Montero-Manso and Hyndman (2021) recommend, prevents the collapse. So do a weighted aggregate-level training row and seasonal differencing. Our contribution is the characterization. The collapse repro
    
[^73]: 自主科学发现中的伪科学诱导

    False-science induction in autonomous scientific discovery

    [https://arxiv.org/abs/2609.27883](https://arxiv.org/abs/2609.27883)

    该论文揭示了自主科学发现系统中的“伪科学诱导”现象——当物理对象与测量结果被连贯地错误配对时，神经代理模型会学到虚假关联并系统性地将实验预算引向低性能区域，且错误的一致性而非错误频率是决定性变量。

    

    闭环发现系统日益增多地自主执行实验并更新决策，使记录完整性成为实验装置本身的一部分。我们证明，当合法的物理对象与测量结果被错误配对时，会产生“伪科学诱导”：神经代理模型会忠实地学习由记录诱导产生的、与真实对象-结果关系不对应的虚假关联，而数据的边缘分布保持不变。在绿色荧光蛋白适应度和材料带隙预测两个闭环系统中，连贯的配对错误绑定会系统性地将实验预算引导至低性能区域，而同等数量的随机交换则几乎没有影响。这些观察结果表明，在所测试的闭环中，错误的一致性而非原始错误频率，才是控制这种预算错配的主要变量。由此得出的绑定可辨识性边界支持对监测轴……（原文摘要在此处截断）

    arXiv:2609.27883v1 Announce Type: cross  Abstract: Closed-loop discovery systems increasingly execute experiments and update decisions autonomously, turning record integrity into part of the experimental apparatus. We show that false-science induction arises when legitimate physical objects and measurements are paired incorrectly, driving neural surrogates to faithfully learn record-induced associations that do not correspond to the true object-outcome relationship while marginal data distributions remain unchanged. Across green fluorescent protein fitness and materials band-gap prediction loops, coherent paired misbinding systematically redirects experimental budgets toward low-performing basins, whereas same-volume random swaps have negligible effects. These observations identify error coherence, rather than raw error frequency, as the primary variable controlling this budget misallocation in the tested loops. The resulting binding identifiability boundary supports monitored-axis qua
    
[^74]: 混沌动力学中噪声导致的极端事件可预测性在预测视界间的重分布

    Noise-Induced Predictability Redistribution Across Forecast Horizons of Extreme Events in Chaotic Dynamics

    [https://arxiv.org/abs/2609.27877](https://arxiv.org/abs/2609.27877)

    该研究发现在混沌系统中，动力学噪声会重分布极端事件的可预测性——在短预测视界上显著提升预测技能（配对增益0.0895），而随预测间隔增大可预测性逐渐衰减。

    

    混沌动力学中的极端事件（EE）是罕见的大幅偏移，其可预报性可能受到动力学噪声的影响。我们研究了噪声如何改变一个三阶自治混沌流系统中极端事件的发生及预测技能在不同预测视界上的表现。我们对所有实现冻结使用单一的无噪声数据阈值，宽事件由每次偏移的一个最大值定义，并利用15个时间单位的历史数据预测未来窗口W=15，采用带时间顺序数据分离的HistGradientBoosting方法。随着观测历史与未来事件窗口之间的预测间隔G增大，无噪声数据的Matthews相关系数（MCC）从G=0时的0.641降至G=15时的0.165。噪声依赖性通过八个幅值下各十次配对实现进行评估。平均短视界得分从无噪声数据的0.456提升至sigma=0.007时的0.546；配对增益为0.0895（95%置信区间0.0494–0.1295；经Holm校正的p=0.0234）。噪声强烈…

    arXiv:2609.27877v1 Announce Type: cross  Abstract: Extreme events (EEs) in chaotic dynamics are rare broad excursions whose forecastability can be altered by dynamical noise. We investigate how noise changes EE occurrence and prediction skill across forecast horizons in a third-order autonomous chaotic flow. A single clean-data threshold is frozen for all realizations, broad events are defined by one maximum per excursion, and a future window W=15 is predicted from a 15-time-unit history using HistGradientBoosting with chronological data separation. As the forecast gap G between the observed history and the future event window increases, the clean Matthews correlation coefficient (MCC) decreases from 0.641 at G=0 to 0.165 at G=15. Noise dependence is evaluated with ten paired realizations at eight amplitudes. The mean short-horizon score increases from 0.456 in clean data to 0.546 at sigma=0.007; the paired gain is 0.0895 (95% CI 0.0494-0.1295; Holm-adjusted p=0.0234). Noise strongly i
    
[^75]: 评估选择决定预测排行榜：来自生产环境市场面板的证据

    Evaluation Choices Decide the Forecasting Leaderboard: Evidence from a Production Marketplace Panel

    [https://arxiv.org/abs/2609.27867](https://arxiv.org/abs/2609.27867)

    在固定数据、预测时长与时期的条件下，仅改变评估设计——分析单位、误差聚合方式和评分对象（预测区间还是点预测）——即可颠覆或消解预测基准测试的主要结论，证明排行榜结果在模型拟合之前就已由评估者的选择决定。

    

    预测基准测试会报告哪种方法获胜。我们证明，答案在任何模型拟合之前就已由评估者的选择所决定。我们在一个包含1,887家企业客户、跨度67个月的生产环境市场面板上，对24种预测方法和一个教科书式参考方法进行基准测试，其中包括六个2025年时代的时间序列基础模型。我们固定数据、预测时长和时期，仅改变评估设计。三个评估选择各自颠覆或消解了一个主要结论。将分析单位从市场总量改为单个客户，使我们的生产基线方法从19个方法中的第二名（无一击败它）变为25个方法中的第23名，其24个挑战者中有19个在该设置下击败了它。改变误差聚合的程度决定了Diebold-Mariano检验是否能发现任何显著差异。而以预测区间而非点预测进行评分，则几乎完全重新排列了各方法的名次，排名相关性仅为0.02。

    arXiv:2609.27867v1 Announce Type: new  Abstract: A forecasting benchmark reports which method won. We show that the answer is set by the evaluator's choices before any model is fitted. We benchmark 24 forecasting methods and one textbook reference, including six 2025-era time series foundation models, on a production marketplace panel of 1,887 business customers over 67 months. We hold the data, the horizon and the period fixed, and vary only the evaluation design. Three choices each reverse or dissolve a headline conclusion. Changing the unit of analysis from the market total to the individual customer moves our production baseline from second of nineteen, beaten by nothing, to twenty-third of twenty-five. Nineteen of its twenty-four challengers beat it there. Changing how much error is pooled decides whether a Diebold-Mariano test finds anything at all. Scoring prediction intervals rather than point forecasts reorders the field almost completely, with a rank correlation of 0.02 on in
    
[^76]: 共享编码器并非共享任务：面向深度专家池的条件比较

    A Shared Encoder Is Not a Shared Task: Conditional Comparison for Deep Expert Pools

    [https://arxiv.org/abs/2609.27866](https://arxiv.org/abs/2609.27866)

    该论文发现共享深度编码器无法消除任务比较分数中的混淆，并提出将条件化双判别器差异移植到嵌入空间形成“功能轴”度量，既能免疫输入旋转的外推混淆又能敏感捕捉标签置换漂移，从而在混合多头生命周期中以更少头数取得更优的决策质量。

    

    共享一个深度编码器本身并不能修复任务比较分数中的核心混淆问题。我们证明，在冻结的共享表示上进行交叉评估的预测头会继承浅层交换分数的外推混淆：在标签固定不变的情况下，纯粹的输入旋转会使深度交换分数从约0膨胀至0.80；而表示新颖性分数则在互补方向上存在盲区（在完全改变任务的标签置换下保持平坦不变）。将条件化双判别器差异移植到嵌入空间中，可以同时解决这两个盲点：功能轴在旋转下保持在±0.001以内，并随标签置换造成的漂移质量单调变化。将该双轴门控机制纳入多头混合（mixture-of-heads）生命周期中，在匹配的训练预算下，它能以更少的头数获得优于交换或新颖性触发器的决策质量。在广义类别发现任务上，同样的块级功能轴能够区分语义新颖（类别）……（原文摘要在此处不完整）

    arXiv:2609.27866v1 Announce Type: cross  Abstract: Sharing a deep encoder does not, by itself, fix the central confound of task-comparison scores. We show that cross-evaluated heads on a frozen shared representation inherit the extrapolation confound of shallow exchange scores: pure input rotations with fixed labels inflate a deep exchange score from about 0 to 0.80, while representation-novelty scores are blind in the complementary direction (flat under label permutations that change the task completely). Transplanting a conditional two-discriminator discrepancy into the embedding space resolves both blind spots: the functional axis stays within +-0.001 under rotations and tracks label-permutation drift mass monotonically. Built into a mixture-of-heads lifecycle, the two-axis gate attains better decision quality with fewer heads than exchange or novelty triggers at a matched training budget. On generalized category discovery, the same chunk-level functional axis separates semantic nov
    
[^77]: 什么变了？具有真实、虚拟与不可比较诊断的漂移检测

    What Changed? Drift Detection with Real, Virtual, and Incomparable Diagnosis

    [https://arxiv.org/abs/2609.27865](https://arxiv.org/abs/2609.27865)

    提出将条件双判别器差异移植到嵌入空间的双轴诊断方法，同时弥补交换分数对输入旋转的虚假敏感和表示新颖性分数对标签置换漂移的盲区，从而在混合头生命周期中以更少头部实现更优漂移检测决策。

    

    共享一个深度编码器本身并不能解决任务比较评分中的核心混淆问题。我们证明，在冻结的共享表示上进行交叉评估的头部会继承浅层交换分数的外推混淆：在标签固定、纯输入旋转的情况下，深度交换分数会从约0膨胀到0.80，而表示新颖性评分在互补方向上是盲目的（在完全改变任务的标签置换下保持平坦）。将条件双判别器差异移植到嵌入空间中可以同时解决这两个盲点：功能轴在旋转下保持在±0.001以内，并单调地跟踪标签置换带来的漂移质量。将该双轴门控构建到混合头（mixture-of-heads）生命周期中，在相同的训练预算下，它能以更少的头部获得比交换分数或新颖性触发器更好的决策质量。在广义类别发现任务上，同样的块级功能轴能够区分语义新（类别）……

    arXiv:2609.27865v1 Announce Type: cross  Abstract: Sharing a deep encoder does not, by itself, fix the central confound of task-comparison scores. We show that cross-evaluated heads on a frozen shared representation inherit the extrapolation confound of shallow exchange scores: pure input rotations with fixed labels inflate a deep exchange score from about 0 to 0.80, while representation-novelty scores are blind in the complementary direction (flat under label permutations that change the task completely). Transplanting a conditional two-discriminator discrepancy into the embedding space resolves both blind spots: the functional axis stays within +-0.001 under rotations and tracks label-permutation drift mass monotonically. Built into a mixture-of-heads lifecycle, the two-axis gate attains better decision quality with fewer heads than exchange or novelty triggers at a matched training budget. On generalized category discovery, the same chunk-level functional axis separates semantic nov
    
[^78]: 精确极小极大一比特无偏压缩：重尾必要性与有限随机性近似

    Exact Minimax One-Bit Unbiased Compression: Heavy-Tail Necessity and Finite-Randomness Approximation

    [https://arxiv.org/abs/2609.27860](https://arxiv.org/abs/2609.27860)

    该论文精确刻画了一比特无偏压缩的极小极大最优值，证明高斯情形的最优编码必然具有临界重尾结构。

    

    逐点无偏的一比特压缩器在传输一个比特的同时能在期望意义下重构任意实值输入。对于具有累积分布函数 $F$、均值 $m$ 和 $\mathcal J(P)=\int_{\mathbb R}\sqrt{F(r)(1-F(r))}\,dr$ 的标量源 $P$，我们证明了在所有于 $\mathbb R$ 上无偏的公共硬币一比特编码中，源平均重构二阶矩的下确界为 $m^2+\mathcal J(P)^2$。对于正则的全支撑源，一种以分布为中心的随机阈值编码可达到该值；针对任意随机化二元编码器的逆命题以及等式分析刻画了所有可达该值的编码（在不计零测集、比特重标记和公共种子细化的意义下）。对于均值满足 $|\mu|\le c\sigma$ 的高斯位置族 $\mathcal N(\mu,\sigma^2)$，端点均值上的等先验是最不利的，其极小极大值为 $\sigma^2\Lambda_c^2$。精确的高斯极小极大最优性迫使编码具有临界重尾：在端点均值处，绝对矩……

    arXiv:2609.27860v1 Announce Type: new  Abstract: A pointwise-unbiased one-bit compressor reconstructs every real input in expectation while transmitting one bit. For a scalar source $P$ with CDF $F$, mean $m$, and $\mathcal J(P)=\int_{\mathbb R}\sqrt{F(r)(1-F(r))}\,dr$, we prove that the infimum of the source-averaged reconstruction second moment over all public-coin one-bit codes unbiased on $\mathbb R$ is $m^2+\mathcal J(P)^2$. For regular full-support sources, a distribution-centered random-threshold code attains this value; a converse over arbitrary randomized binary encoders and an equality analysis characterize every attaining code up to null sets, bit relabeling, and public-seed refinement. For the Gaussian location family $\mathcal N(\mu,\sigma^2)$ with $|\mu|\le c\sigma$, the equal prior on the endpoint means is least favorable and the minimax value is $\sigma^2\Lambda_c^2$. Exact Gaussian minimax optimality forces a critical heavy tail: at the endpoint means, absolute moments
    
[^79]: ChronosAttack：针对大语言模型智能体的对抗性工具调度攻击

    ChronosAttack: Adversarial Tool Scheduling Attacks on LLM Agents

    [https://arxiv.org/abs/2609.27857](https://arxiv.org/abs/2609.27857)

    ChronosAttack是一种仅通过延迟真实工具响应的到达时间（不改动其内容）就能改变LLM智能体处理证据的顺序并操纵其最终决策的对抗性调度攻击。

    

    大语言模型（LLM）智能体通常会在外部工具响应到达时即时处理它们，这使得响应的时序成为决策过程的一部分。我们提出了ChronosAttack，这是一种仅利用延迟的调度攻击，它在不修改、添加、删除或加速工具响应的情况下，改变真实工具响应的到达时间。有界延迟可以改变相同证据的呈现顺序，进而改变最终决策。我们在GPT-5.6 Sol、Gemini 3.6 Flash、DeepSeek V4 Flash和Claude Sonnet 4.6上评估了ChronosAttack。结果显示：GPT-5.6 Sol和Claude在易受攻击的设置中表现出强烈的定向决策偏移，Gemini则表现出大幅的相反方向偏移，而DeepSeek在所测试的调度方式下更为稳定。我们还发现，时序攻击并不总是需要依赖智能体的顺序状态，单次调度反转即可导致重大决策变化。同步和顺序一致性防御可以降低攻击者对观察顺序的控制能力。这些结果表明，工具响应的到达时序……

    arXiv:2609.27857v1 Announce Type: cross  Abstract: Large language model (LLM) agents often process external tool responses as they arrive, making response timing part of the decision process. We introduce ChronosAttack, a delay-only scheduling attack that changes when authentic tool responses arrive without modifying, adding, removing, or accelerating them. Bounded delays can change the order of the same evidence and alter the final decision. We evaluate ChronosAttack on GPT-5.6 Sol, Gemini 3.6 Flash, DeepSeek V4 Flash, and Claude Sonnet 4.6. GPT-5.6 Sol and Claude show strong targeted shifts in vulnerable settings, Gemini shows large shifts in the opposite direction, and DeepSeek is more stable under the tested schedules. We also find that sequential agent state is not always required and that a single scheduling inversion can cause a large decision change. Synchronization and order-consistency defenses reduce attacker control over observation order. These results show that tool-respo
    
[^80]: 基于证据学习的变分自编码器的理论研究

    Theoretical Study on the Evidential Learning-based Variational Autoencoder

    [https://arxiv.org/abs/2609.27853](https://arxiv.org/abs/2609.27853)

    该论文从理论上证明，证据学习变分自编码器中正态-逆伽马潜在层级的四个参数仅有三维商空间 (γ, α, c) 是可辨识的，并且通过对前向KL散度的精确偏最小化，可以实现从四参数到三坐标的精确约简，同时保持最优值不变。

    

    正态-逆伽马（NIG）潜在层级结构包含四个参数，但其诱导的潜在分布并不能唯一确定全部四个参数。对于 σ²~InvGamma(α,β)，μ|σ²~N(γ, σ²/ν)，z|μ,σ²~N(μ, σ²)，z 的边际分布仅通过 c=β(1+1/ν) 依赖于 (ν, β)。因此，重构可见的参数空间是三维商空间 (γ, α, c)，并带有一维的纤维自由度。对于固定的层级变分目标，对前向KL散度到一个完整NIG先验的精确偏最小化，会在每条纤维上选择出一个唯一的先验相对代表元，从而实现精确的三坐标约简，且其最优值与四坐标目标完全相同。记 ρ₀=2β₀/ν₀，T=c/{α[(γ-γ₀)²+ρ₀]}，我们证明逆规范分配 1/ν_can……（原摘要至此截断）

    arXiv:2609.27853v1 Announce Type: cross  Abstract: A normal--inverse-gamma (NIG) latent hierarchy has four parameters, but its induced latent law does not identify all four. For $\sigma^2\sim\mathrm{InvGamma}(\alpha,\beta)$, $\mu\mid\sigma^2\sim\mathcal{N}(\gamma,\sigma^2/\nu)$, and $z\mid\mu,\sigma^2\sim\mathcal{N}(\mu,\sigma^2)$, the marginal law of $z$ depends on $(\nu,\beta)$ only through $c=\beta(1+1/\nu)$. Hence the reconstruction-visible parameter space is the three-dimensional quotient $(\gamma,\alpha,c)$, with a one-dimensional fiber degree of freedom. For a fixed hierarchical variational objective, exact partial minimization of the forward KL divergence to a complete NIG prior selects a unique prior-relative representative on each fiber, yielding an exact three-coordinate reduction with the same optimum as the four-coordinate objective. Writing $\rho_0=2\beta_0/\nu_0$ and $T=c/\{\alpha[(\gamma-\gamma_0)^2+\rho_0]\}$, we show that inverse canonical allocation $1/\nu_{\rm can}$
    
[^81]: 查询隐含生成式引擎优化（QI-GEO）

    Query Implied Generative Engine Optimization

    [https://arxiv.org/abs/2609.27845](https://arxiv.org/abs/2609.27845)

    提出了QI-GEO方法，无需依赖显式查询，直接从文档本身近似其意图空间并推断用户意图，从而优化内容在生成式搜索引擎中的可见性。

    

    随着人们在线查找信息方式的演变，搜索领域的格局发生了巨大变化。传统搜索引擎正在被生成式搜索引擎（GSE）所取代，后者利用大语言模型（LLM）为用户查询生成自然语言回答。对于内容创作者而言，内容的可见性不再仅仅取决于在搜索结果中的排名，而是取决于是否被生成的回答所引用。然而，生成式搜索引擎是一个黑箱，由此催生了生成式引擎优化（GEO），即一系列旨在提高内容在生成式搜索环境中可见性的技术。现有的大多数方法都依赖于显式查询或由查询派生的信号来调整内容，以更好地满足用户需求。我们提出了查询隐含生成式引擎优化（QI-GEO），直接从文档中推断用户意图。我们的方法对文档的意图空间进行近似建模，并识别出文档中可能缺失的内容……

    arXiv:2609.27845v1 Announce Type: cross  Abstract: The landscape of search has changed drastically with how people look for information online. Traditional search engines are being replaced by Generative Search Engines (GSEs), which use Large Language Models (LLMs) to generate natural language responses to user queries. For content creators, visibility is no longer solely determined by ranking in search results but by being cited within generated responses. But Generative Search Engines are black-boxes, leading to the emergence of Generative Engine Optimization (GEO), a set of techniques aimed at improving content visibility in generative search settings. Most existing approaches rely on the explicit queries or query derived signals to align content to better suit user needs. We propose Query Implied Generative Engine Optimization (QI-GEO) to infers user intent directly from the document. Our approach approximates document's intent space and identifies content that may be missing yet r
    
[^82]: 从推理字符串到偏序：通过商策略优化实现验证器认证的规则迁移

    From Reasoning Strings to Partial Orders: Verifier-Certified Rule Transport through Quotient Policy Optimization

    [https://arxiv.org/abs/2609.27833](https://arxiv.org/abs/2609.27833)

    该论文提出 VCRT 方法，通过原生验证器重放相邻操作对来认证交换性，把成功轨迹从线性 token 序列提升为带证书的偏序结构，并将策略信用聚合到认证轨道上，从而在 ProofWriter、CLRS 和 Lean 的跨环境迁移中保留真正的逻辑依赖并提升泛化能力。

    

    许多计算由于独立子目标或不相交的状态更新可以交换，因而存在多种有效的执行顺序。基于可验证奖励的强化学习通常把每条成功轨迹当作一个单独的 token 序列来处理，这使得序列化上的选择可能被误认为是逻辑依赖。我们提出验证器认证的规则迁移（VCRT），利用原生验证器重放相邻操作对：若一对操作在两种顺序下均被验证器接受并到达相同的规范状态，则该对操作获得交换性证书；若被拒绝或会改变状态，则构成“反菱形”。VCRT 利用反菱形来保留真正的先后依赖，并将策略信用分配给每条被认证轨道上的总概率质量，同时约束交换后的一致性、来源保持与策略漂移。我们通过共享的匿名化关系图接口，在 ProofWriter、CLRS 和 Lean 上评估留一环境外的迁移性能。所有训练……

    arXiv:2609.27833v1 Announce Type: new  Abstract: Many computations admit several valid execution orders because independent subgoals or disjoint state updates can commute. Reinforcement learning with verifiable rewards usually treats each successful trace as a separate token sequence, so serialization choices can be mistaken for logical dependencies. We introduce Verifier-Certified Rule Transport (VCRT), which replays adjacent operation pairs with native verifiers. Pairs whose two orders are accepted and reach the same canonical state provide commutation certificates; rejected or state-changing reversals provide anti-diamonds. VCRT uses anti-diamonds to preserve genuine prerequisites and assigns policy credit to the total probability mass of each certified orbit. It also constrains post-swap consistency, source retention, and policy drift. We evaluate leave-one-environment-out transfer across ProofWriter, CLRS, and Lean through a shared anonymized relation-graph interface. All training
    
[^83]: CAST：基于上下文与异常结构条件约束的时间序列异常生成

    CAST: Context- and Anomaly Structure-Conditioned Time Series Anomaly Generation

    [https://arxiv.org/abs/2609.27825](https://arxiv.org/abs/2609.27825)

    该论文提出CAST框架，采用两阶段预训练-微调策略——先利用大量正常数据学习系统动态以缓解异常数据稀缺，再以学习到的异常结构表示为条件约束生成器，从而解决时间序列异常生成中异常数据稀缺与形态异质两大挑战。

    

    异常时间序列在安全关键领域发挥着至关重要的作用，但它们天然稀缺、形态各异且获取成本高昂。现有的时间序列生成方法主要专注于合成正常数据，在需要异常样本时价值有限。我们识别出异常生成中的两个根本性挑战：(i) 异常数据的稀缺性，以及 (ii) 异常的异质性形态特征。为应对这些挑战，我们提出了 CAST——一个基于上下文与异常结构条件约束的时间序列异常生成框架，并采用有原则的两阶段预训练与微调策略。在预训练阶段，我们利用丰富的正常时间序列数据来学习底层系统动态，从而显著缓解异常数据可用性受限的问题。在微调阶段，CAST 使生成器显式地以学习到的异常结构表示为条件，使其能够捕捉……（原文摘要在此处截断）

    arXiv:2609.27825v1 Announce Type: new  Abstract: Anomalous time series play a critical role in safety-critical domains, yet they are inherently scarce, heterogeneous, and costly to obtain. Existing time series generation methods predominantly focus on synthesizing normal data, providing limited value when anomalous samples are needed. We identify two fundamental challenges in anomaly generation: (i) the scarcity of anomaly data, and (ii) the heterogeneous morphological characteristics of anomalies. To address these challenges, we propose CAST, a Context- and Anomaly Structure-conditioned Time series anomaly generation framework with principled two-stage pretraining and finetuning strategy. In pretraining stage, we leverage abundant normal time series data to learn underlying system dynamics and substantially mitigate the limited availability of anomaly data. During finetuning, CAST explicitly conditions the generator on learned anomaly structure representations, enabling it to capture 
    
[^84]: 当适应带来伤害：联邦可穿戴设备引导中的数据划分敏感性与个体层面负迁移

    When Adaptation Hurts: Split Sensitivity and Person-Level Negative Transfer in Federated Wearable Onboarding

    [https://arxiv.org/abs/2609.27819](https://arxiv.org/abs/2609.27819)

    该研究通过严格的防泄漏评估协议发现，联邦可穿戴模型的无标签引导尽管平均表现良好，但在个体用户层面存在显著的负迁移现象，且不同引导策略之间并无统计学上显著的优势差异。

    

    联邦可穿戴模型最终要服务于未参与源训练的用户，但良好的平均准确率并不能证明无标签的引导过程对每个人都有效。我们在五个可穿戴数据集上，采用一种防泄漏评估协议评估了六种核心引导策略，该协议固定源模型检查点、仅从源数据估计归一化参数、将校准记录与评估记录分离，并对留出的用户个体（而非窗口、设备或随机种子）进行推断。完成所有符合条件的HHAR和PAMAP2跨用户轮换后，从原始冻结数据划分所得到的结论发生了实质性变化。在HHAR数据集上，某一单个用户的平衡准确率在各方法中达到95.6-97.2%，而全部九名用户的准确率仅为78.3-83.0%，下降了13.8-17.8个百分点。在两个数据集上，表面上显示的平均领先方法均发生了变化，而配对的领先者-追随者自助置信区间包含零，无法确定存在显著更优的方法。

    arXiv:2609.27819v1 Announce Type: new  Abstract: Federated wearable models eventually serve people absent from source training, but favorable average accuracy does not establish that unlabeled onboarding helps each person. We evaluate six core onboarding strategies on five wearable datasets under a leakage-controlled protocol that fixes source checkpoints, estimates normalization from source data only, separates calibration from evaluation recordings, and performs inference over held-out people rather than windows, devices, or random seeds. Completing all eligible HHAR and PAMAP2 outer-person rotations materially changes the conclusion obtained from the original frozen fold. On HHAR, balanced accuracy on that single person is 95.6-97.2% across methods versus 78.3-83.0% over all nine users, a reduction of 13.8-17.8 percentage points (pp). The displayed mean leader changes on both datasets, while paired leader-runner bootstrap intervals include zero and do not resolve a superior method. 
    
[^85]: 基于下尾假设的检验上鞅的第二类错误界

    Type-II Error Bounds for Test Supermartingales from Lower-Tail Hypotheses

    [https://arxiv.org/abs/2609.27766](https://arxiv.org/abs/2609.27766)

    本文针对检验上鞅方法中的第二类错误问题，研究了对对数增量下尾概率的不同假设所导出的第二类错误界，并将所有结果统一为一个主不等式，即通过 e 变量（逆）矩生成函数的单侧勒让德变换来刻画序贯检验在固定时域和任意时刻的第二类错误上界。

    

    在使用检验上鞅的安全假设检验中，若当财富过程首次超过 1/α 时即拒绝原假设，则 Ville 不等式可以为每个显著性水平 α∈(0,1] 提供随时有效的第一类错误保证。由于固有的不对称性，第二类错误却没有这样的保证：概率在对数增量下尾上的高度集中可能导致一次灾难性的下注，从而抵消已积累的任何证据。本文研究了对这些下尾概率施加的不同假设如何导出序贯检验第二类错误的不同界。这些结果都可归结为一个主不等式，它在固定时域和序贯两种情形下，利用 e 变量的（逆）矩生成函数的单侧勒让德变换在某一个数值处的取值来界定水平 α 下的第二类错误，该数值即为下界超出……

    arXiv:2609.27766v1 Announce Type: cross  Abstract: In safe hypothesis testing with test supermartingals, Ville's inequality provides anytime-valid type-I error guarantees for every significance level $\alpha\in(0,1]$, if one rejects the null hypothesis whenever the wealth process first exceeds $\frac{1}{\alpha}$. Due to an inherent asymmetry, the type-II error does not have such guarantees: a heavy concentration of the probability on the lower tail of the log-increments can lead to one catastrophic bet that undoes any amount of accumulated evidence. This paper studies how different hypotheses on those lower-tail probabilities lead to different bounds on the type-II error of the sequential test. They all reduce to one master inequality, which bounds the type-II error at level $\alpha$, at a fixed horizon and sequentially, in terms of a one-sided Legendre transform of the (inverse-)moment generating function of the e-variables, evaluated at one number: the amount by which the lower bound
    
[^86]: 检验上鞅的第二类错误：e-幂与 Chernoff-Stein 指数

    The Type-II Error of Test Supermartingales: e-Power versus the Chernoff-Stein Exponent

    [https://arxiv.org/abs/2609.27765](https://arxiv.org/abs/2609.27765)

    该论文证明了 e-幂（对数增长率）本身无法为检验上鞅的第二类错误提供任何有限时间保证，而真正控制第二类错误的是 e-变量的 Chernoff-Stein 指数。

    

    在基于检验上鞅的安全假设检验中，只要当财富过程首次超过 1/α 时拒绝原假设，Ville 不等式就能为每个显著性水平 α∈(0,1] 提供任意时刻有效的第一类错误保证。由于一种内在的不对称性，第二类错误的表现有所不同。针对简单原假设与简单备择假设的情形，我们关于第二类错误证明了两点。第一，均值增长率 𝔼_{P₁}[log E]（即 e-幂，也就是 Kelly 赌注策略和增长率最优 e-变量所最大化的量）本身无法约束任何东西：对于每个水平 c>0、每个 α 和时间范围 t，我们都构造出条件 e-幂恰好为 c 的 e-变量，其在时刻 t 之前仍未拒绝原假设的概率可以任意接近于 1。这虽然能迫使最终拒绝，但无法由此得到任何有限时间范围的保证。第二，真正控制第二类错误的量是 e-变量的 Chernoff-Stein 指数，即 Λ(E)=sup_{s≥0}{…（摘要原文在此处截断）

    arXiv:2609.27765v1 Announce Type: cross  Abstract: In safe hypothesis testing with test supermartingales, Ville's inequality provides anytime-valid type-I error guarantees for every significance level $\alpha\in(0,1]$, if one rejects the null hypothesis whenever the wealth process first exceeds $1/\alpha$. Due to an inherent asymmetry, the type-II error behaves differently. We prove two things about the latter, for a simple null and alternative. First, the mean growth rate $\mathbb{E}_{P_1}[\log E]$, the e-power, that Kelly betting and growth-rate-optimal e-variables maximise, bounds nothing on its own. For every level $c>0$, every $\alpha$ and horizon $t$ we construct e-variables of conditional e-power exactly $c$ whose probability of not rejecting by $t$ is arbitrarily close to one. It forces eventual rejection, but no finite-horizon guarantee follows. Second, the quantity that does control the type-II error is the Chernoff-Stein exponent of an e-variable, $\Lambda(E)=\sup_{s\ge0}\{-
    
[^87]: 学习检测符号失效：机器学习与Black-Scholes模型的局限性

    Learning to Detect Symbolic Failure: Machine Learning and the Limits of Black-Scholes

    [https://arxiv.org/abs/2609.27764](https://arxiv.org/abs/2609.27764)

    在拥有专家设计符号特征的期权定价领域，保留领域结构的树模型在检测Black-Scholes模型系统性偏差方面比学习抽象表示的核方法高出21.5个百分点，证明“保留结构优于学习抽象”。

    

    我们将期权定价视为一个表示问题：机器学习能否利用260万份真实期权合约检测出与Black-Scholes模型的系统性偏差？我们比较了三种方案：学习抽象嵌入（核主成分分析）、保留领域结构（基于树的集成方法）以及神经网络验证。结果表明，基于树的方法比核降维方法高出21.5个百分点（93.8% vs 72.3%），且领域专家特征（希腊字母、价内程度）优于人工设计的特征。基于神经网络的偏差标签与基于Black-Scholes的偏差标签一致率达99.9974%，这表明检测到的偏差反映的是市场结构而非模型伪影。我们得出结论：在存在专家设计的符号特征的领域中，保留结构优于学习抽象表示。我们不声称存在可利用的定价偏差。

    arXiv:2609.27764v1 Announce Type: new  Abstract: We treat options pricing as a representation problem: can machine learning detect systematic deviations from Black-Scholes using 2.6M real option contracts? We compare three regimes: learned abstract embeddings (Kernel PCA), preserved domain structure (tree-based ensembles), and neural network validation. Tree-based methods outperform kernel dimensionality reduction by 21.5 percentage points (93.8% vs 72.3%), and domain-expert features (Greeks, moneyness) outperform engineered features. NN-based and BS-based deviation labels agree 99.9974% of the time, suggesting deviations reflect market structure rather than model artifact. We conclude that in domains with expert-designed symbolic features, preserving structure beats learning abstractions. We make no claim of exploitable mispricings.
    
[^88]: “那是什么声音？”：一种用于环境声音识别的多功能、鲁棒且轻量级的卷积Transformer

    "What's That Sound?": A Versatile, Robust, and Lightweight Convolutional Transformer for Environment Sound Recognition

    [https://arxiv.org/abs/2609.27762](https://arxiv.org/abs/2609.27762)

    本文提出了RALCT模型，通过对音频进行随机增强并将MFCC图与对数梅尔频谱图拼接，结合CNN与Transformer架构高效提取声音特征，以仅约31万的轻量参数量实现对环境声音的准确识别，可部署于移动设备以提升听障人士的安全。

    

    传统的助听器既昂贵且用途有限，因为它并非用于检测非语音音频。我们的目标是开发一种机器学习解决方案，提供一种更准确且经济实惠的机制来识别周围环境声音，以提高听障人士的安全性，例如，当有汽车在行人身后鸣笛或枪声响起时，他们需要远离声源。通过对音频添加随机增强、将梅尔频率倒谱系数（MFCCs）图与对数梅尔频谱图进行拼接，并将卷积神经网络（CNNs）融入Transformer架构中，随机音频增强分层卷积Transformer（RALCT）模型能够高效地从多样化的音频表示中提取特征。此外，RALCT模型体积足够小，仅有约31万个参数，可以部署到移动设备中。在UrbanSound8K数据集上的实验结果

    arXiv:2609.27762v1 Announce Type: cross  Abstract: The conventional hearing aid is both costly and lim- ited in usage, as it is not intended to detect non-speech audio. Our objective is to develop a machine learning solution to provide a more accurate and affordable mechanism to identify surrounding sounds to improve the safety of the hearing impaired, i.e., if a car is honking behind pedestrians, or a gunshot is fired, and they need to move away from the source. By adding randomized augmentations to audio, concatenating a Mel-Frequency Cepstral Coefficients (MFCCs) diagram and a log-mel Spectrogram, and including Convolutional Neural Networks (CNNs) in a Trans- former architecture, the Randomized Audiomentational Layered Convolutional Transformers (RALCT) model efficiently extracts features from diversified audio representations. In addition, RALCT is small enough, with only approximately 310,000 parameters, to be deployed into mobile devices. Experimental results on the UrbanSound8K 
    
[^89]: 后门会留下结构性痕迹：用于联邦学习中后门检测与遏制的FedMAST

    Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning

    [https://arxiv.org/abs/2609.27760](https://arxiv.org/abs/2609.27760)

    FedMAST防御方法通过综合结构、频谱和历史三轴互补证据对客户端更新进行评分并分层过滤遏制，从而检测出即使能绕过孤立异常信号的隐蔽后门攻击，因为后门投毒更新必然留下结构性痕迹。

    

    联邦学习使客户端无需共享其原始数据即可对共享模型进行分布式训练。然而，它对客户端提交更新完整性的依赖，使全局模型容易受到隐蔽的后门投毒攻击。尽管现有防御通常只检查孤立的证据来源，但受隐蔽性约束的攻击可以适应这些信号。在本文中，我们证明此类攻击虽然能够抑制孤立的异常信号，但其投毒更新仍会留下残留的结构性痕迹。我们提出FedMAST，一种用于联邦学习后门检测的联邦多轴结构追踪防御方法。FedMAST利用互补的结构、频谱和历史证据对客户端更新进行评分，然后应用分层过滤和轮次级遏制机制来限制对抗性影响。为了捕捉孤立信号可能遗漏的痕迹，FedMAST采用压缩对一致性评分来暴露耦合的特征失真。

    arXiv:2609.27760v1 Announce Type: cross  Abstract: Federated learning enables distributed training of a shared model without requiring clients to share their raw data. However, its reliance on the integrity of the client-submitted updates exposes the global model to stealthy backdoor poisoning. Although existing defenses often inspect isolated evidence sources, stealth-constrained attacks can adapt to these signals. In this paper, we show that such attacks can suppress isolated anomaly signals, but their poisoned updates still leave residual structural traces. We propose FedMAST, a Federated Multi-Axis Structural Tracing defense for backdoor detection in federated learning. FedMAST scores client updates using complementary structural, spectral, and historical evidence and then applies tiered filtering and round-level containment to limit adversarial influence. To capture traces that isolated signals may miss, FedMAST uses squeeze-pair coherence scoring to expose coupled feature distort
    
[^90]: 用于评估新型AI辅助教育问题教学质量的预训练模型评估

    Evaluation of pre-trained models for pedagogical assessment of novel AI-assisted educational questions

    [https://arxiv.org/abs/2609.27749](https://arxiv.org/abs/2609.27749)

    该研究通过评估传统机器学习、Transformer和大语言模型在布鲁姆层级分类任务中的表现，并借助特征工程策略，寻找在AI生成的分布外教育问题上依然稳健的教学质量自动评估方法。

    

    AI辅助生成教育材料的激增已超出我们验证其教学质量的能力。使用布鲁姆分类器（Bloom Classifier）模型进行自动化评估，是一种大规模评估教育材料的有前景的方法。这些模型在同分布数据集（IID数据集）上显示出较高的准确率。然而，将相同的模型应用于新的分布外（OOD）数据集（如AI辅助生成的问题）时，可能会出现性能下降。为了找出在数据集偏移下依然稳健的分类器，我们在布鲁姆层级分类任务上评估了传统机器学习（ML）模型、Transformer模型和大语言模型。我们还探索了特征工程策略，包括引入NLP指标、将学习目标作为输入的一部分进行附加，以及文本拼接，以稳定OOD性能。我们的基线测试显示，TFPOS-IDF机器学习模型在OOD数据上表现较差（宏平均F1分数为0.48），相比之下BERT达到0.55，大语言模型（摘要原文在此处截断）。

    arXiv:2609.27749v1 Announce Type: new  Abstract: The surge in AI-assisted generation of educational materials has outpaced our capacity to validate their pedagogical quality. Automated evaluation using Bloom Classifier models is a promising approach to assess educational materials at scale. These models show high accuracy within-distribution dataset (IID Dataset). However, applying the same models to new out-of-distribution (OOD) datasets such as AI-assisted generated questions could show performance degradation. To identify robust classifiers under dataset shift, we evaluated traditional Machine Learning (ML), transformer, and Large Language models on the Bloom level classification task. We also explored feature-engineering strategies incorporating NLP metrics, appending the learning objectives as part of the input, and text splicing to stabilize OOD performance. Our baseline tests show that TFPOS-IDF ML models perform poorly on OOD (Macro F1-score 0.48) compared to BERT (0.55) and LL
    
[^91]: 更少语言，更多潜变量：面向自动驾驶的标注高效视觉-语言-动作模型

    Less Language, More Latents: Annotation-Efficient VLAs for Driving

    [https://arxiv.org/abs/2609.27747](https://arxiv.org/abs/2609.27747)

    LADA提出三阶段流水线，通过向量量化的潜在动作模型将大规模无标注观测-轨迹对转化为紧凑的车辆高层意图码本，仅需少量语言标注即可训练出可被语言操控的标注高效驾驶VLA模型。

    

    视觉-语言-动作模型（VLA）有望实现人类可操控的自动驾驶，但其训练受到与自然语言指令配对的帧数据稀缺的瓶颈制约：尽管摄像头视频流和专家轨迹能够被大规模记录，语言标注（例如“在路口左转”）仍然稀缺且获取成本高昂。为应对这一挑战，我们提出了潜在动作驾驶标注（Latent Action Driving Annotations，LADA），这是一个三阶段流水线，能够将大量未标注的观测-轨迹对转化为语言条件控制的基石。首先，我们训练一个带有向量量化瓶颈的潜在动作模型，生成一个紧凑的车辆高层意图码本。其次，利用少量带语言标注的子集训练一个视觉-语言翻译器，将观测和语言指令映射到该码本中。第三，我们在完整的未标注语料库上，使用观测-潜在动作对训练驾驶VLA模型。

    arXiv:2609.27747v1 Announce Type: cross  Abstract: Vision-language-action models (VLA) promise human-steerable autonomous driving, but their training is bottlenecked by the scarcity of frames paired with natural-language instructions: while camera streams and expert trajectories are logged at scale, language annotations (e.g., turn left at the intersection) remain scarce and expensive to acquire. To address this challenge, we introduce Latent Action Driving Annotations (LADA), a three-stage pipeline that transforms abundant unlabelled observation-trajectory pairs into a substrate for language-conditioned control. First, we train a latent action model with a vector-quantised bottleneck, producing a compact codebook of high-level vehicle intents. Second, a small language-annotated subset is used to train a vision-language translator to map observations and language instructions into this codebook. Third, we train a driving VLA on observation-latent-action pairs over the full unlabelled c
    
[^92]: 极限核 Q(λ)：桥接短时域与长时域

    Limiting-Kernel Q($\lambda$): Bridging Short and Long Horizons

    [https://arxiv.org/abs/2609.27741](https://arxiv.org/abs/2609.27741)

    提出 LKQL——一种将 n 步截断与基于极限核的长时域近似相结合的离策略价值估计器，在与 n 步估计器相同计算复杂度的前提下实现了短时域与长时域评估的桥接，并可直接嵌入各类 actor-critic 算法。

    

    在基于价值的强化学习中，提高策略评估的准确性已被证明可以改善下游策略优化的性能。目前广泛采用的基于 n 步截断的近似方法族能够产生计算高效的价值估计器，但其本质上局限于较短的评估时域。相比之下，利用转移动力学全局结构的方法虽然可以加速策略评估，但其内存和计算需求通常限制了其在大规模或连续状态空间上的可扩展性。为了调和这些局限性，我们提出了极限核 Q(λ)（Limiting-Kernel Q(λ)，简称 LKQL），这是一种离策略价值估计器，它将 n 步截断与基于极限核（Limiting Kernel，LK）的长时域近似相结合。LKQL 具有与 n 步估计器相同量级的计算复杂度，并可直接集成到在策略和离策略的 actor-critic 算法中。我们证明，在非周期性和（摘要在此处截断）

    arXiv:2609.27741v1 Announce Type: new  Abstract: In value-based reinforcement learning, improving the accuracy of policy evaluation has been shown to improve downstream policy optimization performance. The widely adopted family of approximations relying on $n$-step truncation yields computationally efficient value estimators but is inherently limited to a short evaluation horizon. In contrast, methods that exploit the global structure of the transition dynamics can accelerate policy evaluation, but their memory and computational requirements often limit scalability to large or continuous state spaces. To reconcile these limitations, we introduce Limiting-Kernel Q($\lambda$) (LKQL), an off-policy value estimator that combines $n$-step truncation with a long-horizon approximation based on the limiting kernel (LK). LKQL has the same order of complexity as $n$-step estimators and integrates directly into both on- and off-policy actor-critic algorithms. We prove that, under aperiodicity and
    
[^93]: MENO：内存高效的神经算子

    MENO: Memory-Efficient Neural Operator

    [https://arxiv.org/abs/2609.27739](https://arxiv.org/abs/2609.27739)

    MENO是一种基于流形函数编码器的内存高效PDE神经求解器，其内存占用与数据分辨率无关，支持任意几何域和离散化输入（包括跨几何场景），并在大多数基准测试中取得了最佳精度。

    

    我们提出了内存高效的神经算子（MENO），这是一种基于流形函数编码器（MFE）的高性能偏微分方程（PDE）神经求解器。MENO具有三个主要优势：（1）与其他流行的架构相比，MENO具有显著更小的内存占用和更快的训练速度，且其内存占用与数据分辨率无关，因此具备扩展到大规模模型的潜力。（2）MENO可以接受任意形式的PDE输入，包括任意几何域和任意离散化方式。特别地，它能够处理跨几何场景，即输入函数和输出解定义在不同流形上的情况。（3）MENO表现出强大的泛化能力，与文献中报告的结果相比，在我们测试的大多数基准上取得了最佳精度。代码已发布于GitHub：https://github.com/jpzx

    arXiv:2609.27739v1 Announce Type: new  Abstract: We propose the Memory-Efficient Neural Operator (MENO) as a high-performance PDE neural solver based on the Manifold Function Encoder (MFE). MENO features three primary advantages: (1) MENO has a significantly smaller memory footprint and much faster training speed than other popular architectures, with the memory footprint being independent of the data resolution, and therefore holds the potential for scaling up to large-scale models. (2) MENO can accept PDE inputs of arbitrary form, including arbitrary geometric domains and arbitrary discretizations. In particular, it is capable of handling cross-geometry scenarios, i.e., where the input functions and the output solutions are defined on different manifolds. (3) MENO exhibits strong generalization capability, and achieves the best accuracy on most of the benchmarks we tested, compared with the results reported in the literature. The code is available on GitHub at https://github.com/jpzx
    
[^94]: NS-Attention：视觉Transformer中注意力输出的Newton-Schulz变换

    NS-ATTENTION: Newton-Schulz Transformations of Attention Outputs in Vision Transformers

    [https://arxiv.org/abs/2609.27735](https://arxiv.org/abs/2609.27735)

    提出无参数的Newton-Schulz注意力变换（NS-Attn.），对每个注意力头输出进行谱处理以降低谱集中度并提高有效秩，在ViT和Swin于CIFAR-10/100的全部12组对比实验中均带来平均0.25–0.83个百分点的准确率提升。

    

    Newton-Schulz（NS）迭代最近被用于Muon优化器中，在大语言模型训练过程中对更新矩阵进行变换。受其谱效应的启发，我们研究将NS直接应用于Transformer的注意力表示。我们提出Newton-Schulz注意力（NS-Attn.），这是一种应用于每个注意力头输出的无参数变换。每个注意力头的输出被排列为特征×令牌矩阵，并通过其Frobenius范数进行归一化，随后应用有限步的NS多项式迭代，再恢复原始范数。其目标是在标准的头合并与输出投影之前，降低谱集中度并提高有效秩。在CIFAR-10和CIFAR-100数据集上对ViT和Swin的实验中，NS-Attn.在所有12组同种子对比中均提升了最终轮次的准确率，平均增益为0.25至0.83个百分点。ViT消融实验表明，一次迭代的平均准确率高于两次迭代。谱分析……（原文截断）

    arXiv:2609.27735v1 Announce Type: new  Abstract: Newton-Schulz (NS) iteration has recently been used in the Muon optimizer to transform update matrices during the training of large language models. Motivated by its spectral effect, we investigate applying NS directly to Transformer attention representations. We introduce Newton-Schulz Attention (NS-Attn.), a parameter-free transformation applied to the output of each attention head. Each head output is arranged as a feature-by-token matrix and normalized by its Frobenius norm. We then apply a finite NS polynomial step and restore the original norm. The objective is to reduce spectral concentration and increase effective rank before standard head merging and output projection. Across ViT and Swin on CIFAR-10 and CIFAR-100, NS-Attn. improves final-epoch accuracy in all 12 matched-seed comparisons, with mean gains of 0.25--0.83 percentage points. ViT ablations show higher mean accuracy with one iteration than with two. Spectral analysis f
    
[^95]: FFM-CP：面向小样本计算病理学的视觉-语言基础模型跨骨干网络融合

    FFM-CP: Cross-Backbone Fusion of Vision-Language Foundation Models for Few-Shot Computational Pathology

    [https://arxiv.org/abs/2609.27710](https://arxiv.org/abs/2609.27710)

    该论文提出了FFM-CP框架，通过闭式正交Procrustes变换对齐多个病理学视觉-语言基础模型的异构表示并利用统一图结构实现信息融合，无需额外训练对齐网络即可在小样本条件下有效融合各基础模型的互补能力。

    

    病理学视觉-语言基础模型在不同疾病和任务上的表现各不相同，没有任何单一模型能够始终表现最佳。同时，专家病理标注的高昂成本也限制了可用于任务特定适配的标注数据量。组合互补的预训练表示是应对这些局限的一种潜在方法，但如何从少量标注样本中学习有效的融合仍然具有挑战性。我们提出了面向小样本计算病理学的基础模型融合框架（FFM-CP），这是一个在小样本学习设置下结合多个病理学视觉-语言模型的框架。该框架首先利用从对应支持图像中估计的闭式正交Procrustes变换来对齐异构表示，这种对齐方式在无需训练额外对齐网络的情况下保留了各模型内部的特征几何结构。在对齐后的空间中，一个统一的图使信息得以……（原文摘要在此处截断）

    arXiv:2609.27710v1 Announce Type: cross  Abstract: Pathology vision-language foundation models vary in performance across diseases and tasks, with no single model consistently performing best. The high cost of expert pathology annotation can also limit the labeled data available for task-specific adaptation. Combining complementary pretrained representations is a potential approach to these limitations, yet learning an effective fusion from few labeled examples remains challenging. We introduce Few-shot Fusion Foundation Models of Computational Pathology (FFM-CP), which is a framework that combines multiple pathology vision-language models in the few-shot learning setting. The framework first aligns heterogeneous representations using a closed-form Orthogonal Procrustes transformation estimated from corresponding support images. This alignment preserves within-model feature geometry without training an additional alignment network. Within the aligned space, a unified graph enables info
    
[^96]: 表格基础模型在上下文中计算了什么？通过注意力门控更新实现的原位表示精炼

    What Do Tabular Foundation Models Compute In Context? In-Situ Representation Refinement through Attention-Gated Updates

    [https://arxiv.org/abs/2609.27679](https://arxiv.org/abs/2609.27679)

    提出“原位表示精炼”机制并构建RefineICL——一种注意力门控、无FFN的上下文学习堆栈，使表格基础模型在不改变参数的情况下利用支持集标签精炼回合表示并迁移至查询，性能超越TabPFN-3等现有模型。

    

    当每张表格都定义一个新的监督任务时，表格基础模型应当学习何种可复用的计算？我们提出了“原位表示精炼”：支持集标签引导对该回合表示的更新，并且这些更新在不改变模型参数的情况下迁移到未标注的查询上。通过正则化的留一法目标，我们得到了支持集校正及其查询扩展。其中的主导项将基于注意力的读取与依赖状态的缩放分离开来，由此启发了RefineICL：一种注意力门控、无FFN的上下文堆栈，包含精选的低秩特征交互和类型化记忆。RefineICL-L24在AMLB29上达到了0.93836的OVR-AUC和0.87173的准确率。经过基准信息引导的继续训练，它在38个数据集的TabArena快照上达到1644.8 Elo，在相同评估条件下比TabPFN-3高出31.4 Elo。在TabZilla的两个视图上，它在所有四个报告指标上也均优于TabPFN-v3。在匹配的10万次更新深度网格中，扩展的FF…（摘要在此处截断）

    arXiv:2609.27679v1 Announce Type: new  Abstract: What reusable computation should a tabular foundation model learn when every table defines a new supervised task? We develop in-situ representation refinement: support labels guide updates to the episode's representations, and these updates transfer to unlabeled queries without changing model parameters. A regularized leave-one-out objective yields a support correction and its query extension. The leading term separates attention-based reading from state-dependent scaling, motivating RefineICL: an attention-gated, FFN-free contextual stack with selected low-rank feature interaction and typed memory. RefineICL-L24 reaches 0.93836 OVR-AUC and 0.87173 accuracy on AMLB29. A benchmark-informed continuation reaches 1644.8 Elo on the 38-dataset TabArena snapshot, 31.4 Elo above TabPFN-3 under the same evaluation. It also improves all four reported metrics over TabPFN-v3 on both TabZilla views. In a matched 100K-update depth grid, an expanded FF
    
[^97]: 基于风险敏感与评论家一致性正则化的鲁棒对抗强化学习

    Robust Adversarial Reinforcement Learning with Risk Sensitivity and Critic Consistency Regularization

    [https://arxiv.org/abs/2609.27667](https://arxiv.org/abs/2609.27667)

    提出 RACER 统一框架，从风险敏感视角出发，通过状态相关的自适应对抗目标和评论家一致性正则化，解决了鲁棒对抗强化学习中优化不稳定和价值估计有偏的问题，提升了智能体在动态不确定性下的鲁棒性。

    

    强化学习（RL）在序贯决策中取得了优异的性能，但在动态不确定性和分布偏移下仍然表现脆弱。鲁棒对抗强化学习（RARL）通过最坏情况扰动来提升鲁棒性，但现有方法常常面临优化不稳定和价值估计退化的问题。特别是，过于激进的对抗者会将智能体推向缺乏信息量的失败状态，而对抗扰动会放大双评论家之间的分歧并引入有偏的价值目标。我们提出了一个统一框架 RACER（风险敏感的鲁棒对抗评论家一致性正则化强化学习），从风险敏感的视角重新审视对抗强化学习。首先，我们引入了一个状态相关的对抗目标，能够自适应地调节扰动强度，在抑制有害扰动的同时保留有信息量的探索。（摘要原文在此处截断）

    arXiv:2609.27667v1 Announce Type: new  Abstract: Reinforcement learning (RL) achieves strong performance in sequential decision-making but remains brittle under dynamic uncertainty and distributional shifts. Robust Adversarial Reinforcement Learning (RARL) improves robustness via worst-case perturbations, but existing approaches frequently suffer from unstable optimization and degraded value estimation. In particular, overly aggressive adversaries can drive the agent toward uninformative failure states, while adversarial perturbations amplify disagreement between double critics and introduce biased value targets. We propose a unified framework, RACER (Risk-sensitive robust Adversarial critic ConsistEncy-regularized Reinforcement learning), that revisits adversarial RL from a risk-sensitive perspective. First, we introduce a state-dependent adversarial objective that adaptively regulates perturbation strength, suppressing harmful disturbances while preserving informative exploration. Se
    
[^98]: 基于噪声降低与偏差校正的隐私去中心化优化

    Private Decentralized Optimization with Noise Reduction and Bias Correction

    [https://arxiv.org/abs/2609.27658](https://arxiv.org/abs/2609.27658)

    PRDO通过同批次梯度差的递归估计降低采样与隐私噪声，并结合精确扩散组件校正数据异构导致的去中心化偏差，在无需数据异构性一致有界假设的情况下实现了更优的隐私去中心化优化性能。

    

    隐私去中心化学习受到采样噪声、隐私噪声以及异构数据下去中心化偏差的影响。我们提出了隐私递归去中心化优化方法（PRDO）。PRDO利用基于同批次梯度差的递归估计来降低由采样噪声和隐私噪声引起的估计误差，同时其精确扩散（Exact Diffusion）组件可以校正由数据异构性引起的去中心化偏差。我们的分析在不假设各节点间数据异构性一致有界的前提下建立了非凸收敛界，并进一步给出了一个充分条件，在该条件下递归梯度差能够产生严格低于隐私精确扩散的查询敏感度，同时提供了一个严格满足该条件的示例。实验表明，与所评估的基线方法相比，本方法取得了更高的准确率。

    arXiv:2609.27658v1 Announce Type: new  Abstract: Private decentralized learning is affected by sampling noise, privacy noise, and decentralized bias under heterogeneous data. We propose Private Recursive Decentralized Optimization (PRDO). PRDO uses recursive estimation with same-batch gradient differences to reduce estimation errors caused by sampling and privacy noise, while its Exact Diffusion component corrects decentralized bias arising from data heterogeneity. Our analysis establishes a nonconvex convergence bound without assuming uniformly bounded data heterogeneity across nodes. It further gives a sufficient condition under which recursive gradient differences yield strictly lower query sensitivity than private Exact Diffusion, together with an example that rigorously satisfies this condition. Experiments show improved accuracy over the evaluated baselines.
    
[^99]: FLEET：从Logits熵到文本生成中的增强轨迹

    FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation

    [https://arxiv.org/abs/2609.27657](https://arxiv.org/abs/2609.27657)

    FLEET通过引入记忆机制，将生成过程表示为基于熵阈值状态的稀疏轨迹，并利用每token效用分数调整logits，实现了与重复采样相同的准确率但速度提升3倍。

    

    基于大语言模型（LLM）的解决方案通常依赖温度采样，通过从补全分布中聚合多个样本来提高准确性和稳定性。然而，这种无记忆的方法本质上是次优的：由于缺乏对先前生成结果及其评估的了解，随着采样数量增加，会产生越来越多的语义重复答案，导致收益递减。为了解决这一局限性，我们提出了FLEET，这是一种将记忆机制集成到生成过程中的新方法。FLEET将每次生成表示为通过熵超过预定阈值状态的稀疏轨迹，并利用这些轨迹推断每个token的效用分数来调整logits。基准评估表明，FLEET在与重复采样基线达到相同准确率的情况下实现了3倍加速，并在复杂代码任务上显著提高了准确率。

    arXiv:2609.27657v1 Announce Type: cross  Abstract: Solutions based on large language models (LLMs) often rely on temperature sampling to improve accuracy and stability by aggregating multiple samples from the completion distribution. However, this memoryless approach is inherently suboptimal: because it lacks awareness of prior generations and their evaluations, it produces an increasing proportion of semantically duplicate answers as more samples are drawn, leading to diminishing returns. To address this limitation, we introduce FLEET, a novel method that integrates a memory mechanism into the generation process. FLEET represents each generation as a sparse trajectory through states whose entropy exceeds a predefined threshold and uses these trajectories to infer per-token utility scores that adjust the logits. Benchmark evaluations demonstrate that FLEET achieves the same accuracy as the repeated sampling baseline, with a 3x speedup, and substantially improves accuracy on complex cod
    
[^100]: FedIncome：数据主权约束下面向数字借贷收入估计的联邦学习

    FedIncome: Federated Learning for Income Estimation in Digital Lending Under Data Sovereignty Constraints

    [https://arxiv.org/abs/2609.27654](https://arxiv.org/abs/2609.27654)

    FedIncome提出了一种联邦学习框架，使放贷机构无需共享原始借款人数据即可协同训练收入估计模型，在保障数据主权的同时，为小样本机构带来了显著的预测性能提升。

    

    在数字贷款申请中，经过核实的收入信息往往无法获得，迫使放贷机构依赖借款人自行申报的收入，这可能导致过度放贷、贷款方案过于保守，或拒绝具备还款能力的申请人。跨机构数据共享的限制使得这一问题对训练数据有限的小型放贷机构尤为棘手。我们提出了FedIncome，一个用于收入估计的联邦学习框架，使各机构无需汇集原始借款人记录即可协同训练共享模型。我们使用超过一百万笔LendingClub贷款数据，将其划分为50个州级客户端，模拟了一个异构的放贷联盟。最优的联邦模型在时间外测试中达到R²=0.608，而集中式数据池基准为0.619。与集中式数据池基准相比，小样本客户端的时间外R²平均提升了3.8个百分点，且客户端层面的拟合……

    arXiv:2609.27654v1 Announce Type: cross  Abstract: Verified income is often unavailable in digital loan applications, forcing lenders to rely on reported income and potentially leading to over-lending, overly conservative offers, or rejection of creditworthy applicants. Cross-institutional data-sharing constraints make this problem especially difficult for smaller lenders with limited training data. We introduce FedIncome, a federated learning framework for income estimation that enables institutions to train a shared model without pooling raw borrower records. Using more than one million LendingClub loans partitioned into $50$ state-level clients, we simulate a heterogeneous lending consortium. The best federated model achieves out-of-time $R^2=0.608$, compared with $0.619$ for a pooled centralised benchmark. Small-sample clients obtain an average out-of-time $R^2$ improvement of $3.8$ percentage points relative to the pooled centralised benchmark, while the fitted client-level relati
    
[^101]: 面向大规模交通预测的局部异质性与跨区域上下文学习

    Learning Local Heterogeneity and Cross-Region Context for Large-Scale Traffic Forecasting

    [https://arxiv.org/abs/2609.27637](https://arxiv.org/abs/2609.27637)

    提出LoReST局部-区域时空网络，在节点邻域和路网区域两个互补粒度上建模空间依赖，兼顾局部异质性捕获与跨区域上下文获取，实现高效的大规模交通流预测。

    

    交通流预测对智能交通系统至关重要。大规模交通预测需要联合建模局部空间依赖和跨区域上下文。由于道路属性和行驶方向的差异，地理位置相邻节点之间的空间依赖具有异质性，而通过全对节点交互获取全局信息会带来巨大的计算开销。因此，在捕获局部异质性的同时高效获取长程上下文，仍然是大尺度交通预测中的一个重要挑战。为解决这些挑战，我们提出了LoReST，一个局部-区域时空网络，它在两个互补的粒度上建模空间依赖：节点邻域和路网区域。具体而言，关系感知的局部聚合通过道路和方向特定的特征来捕获地理邻域内的异质依赖……

    arXiv:2609.27637v1 Announce Type: cross  Abstract: Traffic flow forecasting is essential to intelligent transportation systems. Large-scale traffic forecasting requires jointly modeling local spatial dependencies and cross-region context.Spatial dependencies between geographically neighboring nodes are heterogeneous due to differences in road identity and travel direction, while acquiring global information through allpairs node interactions incurs substantial computational costs. Therefore, capturing local heterogeneity while efficiently acquiring long-range context remains an important challenge in largescale traffic forecasting. To address these challenges, we propose LoReST, a Local-Region Spatial Temporal network that models spatial dependencies at two complementary granularities: node neighborhoods and road network regions. Specifically, relation-aware local aggregation captures heterogeneous dependencies within geographic neighborhoods through road and direction specific feature
    
[^102]: Pheno-GS：表型组学尺度的测地线Sinkhorn方法

    Pheno-GS: Phenoscape-scale Geodesic Sinkhorn

    [https://arxiv.org/abs/2609.27633](https://arxiv.org/abs/2609.27633)

    Pheno-GS通过图连通性正则化、基于KL惩罚的不平衡最优传输和批处理矩阵算法三大组件，实现了在噪声、稀疏、不平衡的大规模单细胞数据场景下，准确且可扩展地计算患者间分布的测地传输距离。

    

    高通量单细胞数据如今已在大型患者队列中被广泛采集。从细胞层面数据理解患者层面的异质性，推动了“表型景观化”（phenoscaping）的研究：将每个单细胞分布嵌入为一个“数据点”，并以最优传输（OT）距离来度量它们之间的距离。在这一尺度下，在所有患者数据集对之间计算具有几何感知的OT距离仍然是一个开放性难题，因为现有方法要么依赖会扭曲流形结构的欧氏地面度量，要么在稀疏、采样不均或大规模数据下失效。我们提出了Pheno-GS（Phenoscape-scale Geodesic Sinkhorn，表型组学尺度测地线Sinkhorn），通过三个组件在噪声、不平衡、大规模的设置下计算准确且可扩展的测地传输距离：(1) 图连通性正则化，用于在稀疏/不连通流形上获得定义良好的测地线；(2) 基于KL散度边际惩罚的不平衡OT形式化；(3) 一种批处理矩阵算法，用于计算……（摘要原文在此处截断）

    arXiv:2609.27633v1 Announce Type: new  Abstract: High-throughput single-cell data is now collected across large patient cohorts. Understanding patient-level heterogeneity from cellular-level data motivates phenoscaping: embedding each single-cell distribution as a "datapoint," with distances given by optimal transport (OT). Computing geometry-aware OT at this scale, between all pairs of patient datasets, remains an open challenge, since existing methods either rely on Euclidean ground metrics that distort manifold structure or fail under sparse, unevenly sampled, or large-scale data. We present \textbf{Pheno-GS} (Phenoscape-scale Geodesic Sinkhorn), which computes accurate, scalable geodesic transport distances under noisy, unbalanced, large-scale settings via three components: ($1$) graph connectivity regularization for well-defined geodesics on sparse/disconnected manifolds; ($2$) an unbalanced OT formulation via KL marginal penalties; and ($3$) a batched matrix algorithm computing a
    
[^103]: 基于聚类感知草图的高效线性赌博机算法

    Efficient Linear Bandits via Cluster-Aware Sketching

    [https://arxiv.org/abs/2609.27594](https://arxiv.org/abs/2609.27594)

    提出聚类草图线性赌博机算法，通过在每个聚类中保留完整协方差信息并利用哨兵机制进行聚类切换，在保证稳健次线性遗憾的同时显著降低了每轮更新的计算成本。

    

    我们研究在有限动作集的高维设置下线性赌博机的计算效率问题。在线性赌博机中，特征向量维度 $d$ 的增加会导致每轮更新产生不断增长的 $O(d^2)$ 计算成本。传统的基于草图的方法（如 SOFUL）通过固定大小的矩阵草图来降低计算量，但当数据的谱尾部较重且草图大小选择不当时，存在产生无效线性遗憾的风险。为了保证遗憾收敛并有效降低计算成本，我们引入一种聚类机制，提出聚类草图线性赌博机算法。我们的方法在每个聚类中保留完整的协方差信息，从而保证稳健的次线性遗憾，且不存在谱尾部脆弱性；通过为每个聚类设置哨兵来实现聚类切换，并将每轮更新的计算成本降低至更低量级。

    arXiv:2609.27594v1 Announce Type: new  Abstract: We study the problem of computational efficiency for linear bandits in high-dimensional settings with a finite arm set. In linear bandits, the increase in the dimension $d$ of the feature vectors leads to growing computational costs of $O(d^2)$ at each round of update. Traditional sketching-based methods such as SOFUL reduce computation via fixed-size matrix sketching, yet run the risk of incurring vacuous linear regret when the spectral tail of the data is heavy and the sketch size is inadequately selected. To guarantee regret convergence and effectively reduce computational costs, we introduce a clustering mechanism and propose the Cluster Sketch Linear Bandit (CS-LB) algorithm. Our method preserves the full covariance information in each cluster to guarantee robust sublinear regret without spectral-tail vulnerabilities, performs cluster switching by assigning a sentinel for each cluster, and reduces per-round update computation to $O(
    
[^104]: 隐藏而非删除：网络如何抑制纠缠特征

    Hidden not Deleted: How Networks Suppress Entangled Features

    [https://arxiv.org/abs/2609.27593](https://arxiv.org/abs/2609.27593)

    该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。

    

    通过线性投影实现的概念擦除方法假设特征占据可分离的子空间。我们证明该假设在密集叠加情况下会失效：当两个特征被迫形成共享同一子空间的对跖对时，最先进的线性擦除方法会同时破坏两者，而不仅仅是目标特征。通过梯度下降训练的网络则以非线性方式解决这一问题，但方式并不统一：根据初始化的不同，它们会收敛到两种不同的电路级解决方案之一，我们称之为“镜像”解决方案和“阴影”解决方案。我们将这种分叉现象映射为特征纠缠程度的函数，证明它反映的是稳定的吸引子结构而非实验设置的伪影，并通过针对性的因果干预证明，这两种解决方案都会在被擦除特征的表征中留下大量可测量的完整痕迹，仅需一个标量补丁即可恢复，而无需任何进一步的训练。这一现象类似于——（摘要原文在此处截断）

    arXiv:2609.27593v1 Announce Type: cross  Abstract: Concept erasure methods that operate via linear projection assume that features occupy separable subspaces. We show this assumption fails under dense superposition: when two features are forced into an antipodal pair sharing a single subspace, state-of-the-art linear erasure destroys both, not just the target. Networks trained with gradient descent instead solve this problem non-linearly, but not uniformly: they converge to one of two distinct circuit-level solutions depending on initialization, which we call mirror and shadow solutions. We map this bifurcation as a function of feature entanglement, show it reflects a stable attractor structure rather than an artifact of our setup, and use targeted causal interventions to demonstrate that both solutions leave a substantial, measurable trace of the erased feature's representation intact, recoverable through a single scalar patch rather than requiring any further training. This mirrors a
    
[^105]: 能力流形与机器学习缩放定律

    The Capability Manifold and ML Scaling Laws

    [https://arxiv.org/abs/2609.27588](https://arxiv.org/abs/2609.27588)

    本文提出“能力流形”这一多维框架，通过有界缩放函数将模型下游能力（如推理、规划等）与预训练、后训练和测试时资源关联起来，弥补了传统缩放定律仅依赖损失无法刻画模型能力差异的不足。

    

    现有的机器学习（ML）缩放定律将预测损失与计算量、模型参数和数据量相关联。然而，随着模型越来越多地通过智能体框架进行部署，仅凭损失已不足以刻画下游性能：损失相近的模型在推理、检索、规划和适应等方面可能表现出不同的能力。然而，目前尚无统一的框架将这些能力与机器学习全生命周期中可获得的耦合资源联系起来。我们通过引入“能力流形”来弥合这一差距，这是一个多维框架，通过有界的缩放函数将下游能力映射到预训练、后训练和测试时资源上。解析雅可比矩阵量化了能力对资源变化及资源间相互作用的敏感性。作为初步应用，我们将Kaplan型和Chinchilla型缩放定律以及测试时计算嵌入到该框架中，展示了现有缩放关系如何能够被统一……

    arXiv:2609.27588v1 Announce Type: cross  Abstract: Existing machine learning (ML) scaling laws relate predictive loss to compute, model parameters, and data. However, as models are increasingly deployed through agentic harnesses, loss alone is insufficient to characterize downstream performance: models with similar loss can exhibit different capabilities in reasoning, retrieval, planning, and adaptation. Yet, no unified framework connects such capabilities to the coupled resources available across the ML lifecycle. We bridge this gap by introducing a capability manifold, a multidimensional framework mapping downstream capabilities to pre-training, post-training, and test-time resources through bounded scaling functions. Analytical Jacobians quantify capability sensitivity to resource changes and interactions. As an initial application, we embed Kaplan- and Chinchilla-type scaling laws and test-time compute within the framework, demonstrating how existing scaling relationships can be un
    
[^106]: 步进定律能否迁移到小规模语言模型？低于59M参数的实证重新校准

    Does Step Law Transfer to Small-Scale Language Models? An Empirical Recalibration Below 59M Parameters

    [https://arxiv.org/abs/2609.27581](https://arxiv.org/abs/2609.27581)

    该论文首次实证检验了步进定律在59M参数以下小规模语言模型区间是否成立，并针对最优学习率与批量大小的幂律公式在此区间进行了重新校准。

    

    步进定律为预训练语言模型时的最优峰值学习率η*和批量大小B*给出了幂律公式。该定律是在59M至1B参数规模的模型上校准的，其作者从未对N < 59M的小模型区间进行过实证检验。这一区间对于单GPU训练、可解释性研究、教学实验，以及因内存或成本限制而无法使用更大模型的场景具有重要意义。我们检验了步进定律能否迁移到小语言模型上，并考虑三种可能结果：H1，原始系数可以直接使用；H2，幂律形式成立但系数不同；H3，幂律无法描述该区间内的最优点。所有实验均使用统一的nanoGPT/TinyStories流水线，采用2048词元的BPE词表、AdamW优化器以及预热余弦学习率调度。每个(N, D)组合的最优值通过对损失面L(η, B)在对数空间的局部二次近似提取……

    arXiv:2609.27581v1 Announce Type: cross  Abstract: Step Law gives power-law formulas for the optimal peak learning rate eta* and batch size B* when pre-training language models. It was calibrated on models between 59M and 1B parameters; the small-model regime N < 59M was never tested empirically by its authors. This regime matters for single-GPU training, interpretability research, educational experiments, and settings where larger models are infeasible on memory or cost grounds.   We test whether Step Law transfers to small language models. We consider three outcomes: H1, the original coefficients work directly; H2, the power-law form holds but with different coefficients; and H3, a power law does not describe the optima in this regime. All experiments use a single nanoGPT/TinyStories pipeline with a 2048-token BPE vocabulary, AdamW, and a warmup-cosine schedule. The optimum for each (N, D) cell is extracted from the loss surface L(eta, B) via a local quadratic approximation in log-lo
    
[^107]: VCMM：面向多模态学习的方差校准动量方法

    VCMM: Variance-Calibrated Momentum for Multimodal Learning

    [https://arxiv.org/abs/2609.27577](https://arxiv.org/abs/2609.27577)

    针对多模态训练中的模态不平衡问题，VCMM 通过在线估计各模态的梯度噪声与时间漂移，并利用卡尔曼式控制器自适应地校准模态特定的动量参数，使梯度记忆与各模态的梯度动态相匹配，从而改进多模态联合优化。

    

    多模态联合训练常常受到模态不平衡问题的困扰，即某个占主导地位的模态会抑制其他模态的优化。现有方法主要通过调节梯度的大小或方向、修改优化目标或调整训练策略来平衡各模态的学习，且大多数干预措施仅关注当前的更新步骤。然而，当与广泛使用的基于动量的优化器结合使用时，参数更新还会融入来自先前梯度的累积信息，而仅靠当前步骤的调节无法显式解决这一问题。为解决该问题，我们提出了方差校准动量方法，它能根据模态特定的梯度动态来自适应地调整梯度记忆。具体而言，VCMM 在线估计小批量噪声和时间漂移，并通过一个受卡尔曼滤波启发的控制器利用二者的相对强度来确定模态特定的动量。我们进一步在各模态间对控制信号进行中心化处理，并应……（摘要在此处截断）

    arXiv:2609.27577v1 Announce Type: new  Abstract: Multimodal joint training often suffers from modality imbalance, where a dominant modality suppresses the optimization of others. Existing methods mainly balance modality learning by modulating gradient magnitudes or directions, modifying optimization objectives, or adjusting training strategies, with most interventions focusing on the current update. However, when combined with widely used momentum-based optimizers, the update also incorporates accumulated information from previous gradients, which is not explicitly addressed by current-step modulation alone. To address this issue, we propose Variance-Calibrated MomentuM (VCMM), which adapts gradient memory to modality-specific gradient dynamics. Specifically, VCMM estimates minibatch noise and temporal drift online and uses their relative strength to determine modality-specific momentum through a Kalman-inspired controller. We further center the control signal across modalities and app
    
[^108]: DCRL：通过策略-奖励流形对齐实现解耦与耦合的强化学习

    DCRL: Decoupling and Coupling Reinforcement Learning via Policy-Reward Manifold Alignment

    [https://arxiv.org/abs/2609.27572](https://arxiv.org/abs/2609.27572)

    提出DCRL方法，从几何视角将大语言模型推理建模为逻辑推理、评估与表示三个耦合子流形，并通过策略-奖励流形对齐来解决现有奖励系统中优化不稳定和奖励欺骗的问题。

    

    强化学习（RL）已成为提升大语言模型（LLM）推理能力的关键范式。然而，现有的奖励系统，如基于规则的系统和基于奖励模型的系统，往往存在优化不稳定和奖励欺骗（reward hacking）等问题。在本工作中，我们从几何视角重新审视大语言模型的通用推理，将其概念化为一个由三个相互依赖的子流形构成的耦合流形：逻辑推理、评估和表示。基于这一视角，强化学习中的响应生成可以被解释为从评估流形中解耦的过程，而奖励估计则对应于从逻辑推理流形中解耦的过程。基于规则和基于奖励模型的强化学习系统的局限性，可以从几何上解释为强化学习过程中策略-奖励流形的失配问题。为解决上述错位问题，我们提出了解耦与耦合强化学习方法……（摘要在此处截断）

    arXiv:2609.27572v1 Announce Type: cross  Abstract: Reinforcement learning (RL) has emerged as a key paradigm for improving the reasoning capabilities of large language models (LLMs). However, existing reward systems, such as rule-based and reward-model-based, often exhibit issues such as unstable optimization and reward hacking. In this work, we revisit the general reasoning of LLMs from a geometric perspective, conceptualizing it as a coupled manifold composed of three interdependent sub-manifolds: logical deduction, evaluation, and representation. Based on this perspective, response generation in RL can be interpreted as a decoupling process from the evaluation manifold, while reward estimation corresponds to a decoupling process from the logical deduction manifold. The limitations of rule-based and reward-model RL systems can be geometrically interpreted as the mismatch of policy-reward manifolds during RL process. To address the aforementioned misalignment, we propose Decoupling an
    
[^109]: TNLearn：一个面向任务驱动神经元的开源Python软件包

    TNLearn: An Open Source Python Package for Task-based Neurons

    [https://arxiv.org/abs/2609.27564](https://arxiv.org/abs/2609.27564)

    TNLearn是一个开源Python软件包，实现了任务驱动神经元和网络的自动化构建与顺畅训练，推动了“针对特定任务定制神经元”这一新范式的科研与产业应用。

    

    大脑并不依赖单一类型的神经元来执行各种任务；相反，它为不同的任务设计了不同的神经元。与基于任务的架构相比，基于任务神经元的理念代表了一种范式转变。该理念认为，解决特定问题需要定制化的神经元，因为基于任务的神经元能够从与任务相关的数据中捕获有用的先验知识。为了促进基于任务的神经元在科学研究和工业应用中的使用，我们推出了TNLearn——一个开源的Python软件包，它提供了基于任务的神经元和网络的自动化构建功能，使基于任务的网络能够顺利训练。完整的文档（包括技术阐述、API参考和代表性示例）可在线获取。TNLearn已在 https://github.com/NewT123-WM/tnlearn 开源，并已成为PyTorch生态系统项目。

    arXiv:2609.27564v1 Announce Type: cross  Abstract: The brain does not rely on a single type of neuron to perform all kinds of tasks; instead, it designs different neurons for different tasks. The concept of task-based neurons represents a paradigm shift compared to task-based architectures. It argues that solving a specific problem requires customized neurons, as task-based neurons capture useful prior knowledge from task-related data. To facilitate the use of task-based neurons in scientific research and industrial applications, we introduce TNLearn, an open-source Python package that provides automated construction of task-based neurons and networks, enabling smooth training of task-based networks. Comprehensive documentation, including technical exposition, API reference, and representative examples, is available online. TNLearn is open-sourced at https://github.com/NewT123-WM/tnlearn and has become a PyTorch ecosystem project.
    
[^110]: PhyMo：面向多模态AI4Physics的物理场模态

    PhyMo: A Physical-Field Modality for Multimodal AI4Physics

    [https://arxiv.org/abs/2609.27554](https://arxiv.org/abs/2609.27554)

    该论文提出PhyMo框架，创新性地引入“物理场模态”这一全新模态，通过PDE关联算子组织异构物理测量数据，并采用三阶段学习流程（PDE残差监督预训练、与视觉嵌入对齐、多模态融合）来提升物理系统预测能力。

    

    多模态学习正成为AI for Physics（AI4Physics）的强大范式，其中预测物理系统需要对异构观测、测量数据和领域知识的联合解释。然而，现有方法通常将物理量和控制方程表示为通用的数值或文本标记，忽视了决定其时空相互作用的物理约束。为解决这一局限，我们引入了物理场模态，并提出PhyMo——一个以物理为基础的多模态框架，通过PDE关联算子来组织异构测量数据。PhyMo遵循三阶段学习流程：首先在PDE残差监督下通过场重构对物理场编码器进行预训练，随后将其表示与视觉嵌入在共享潜在空间中进行对齐，最后融合的多模态表示…（摘要截断）

    arXiv:2609.27554v1 Announce Type: cross  Abstract: Multimodal learning is emerging as a powerful paradigm for AI for Physics (AI4Physics), where predicting physical systems requires the joint interpretation of heterogeneous observations, measurements, and domain knowledge. However, existing approaches typically represent physical quantities and governing equations as generic numerical or textual tokens, overlooking the physical constraints that determine their spatiotemporal interactions. To address this limitation, we introduce the \textbf{physical-field modality} and propose \textbf{PhyMo}, a physics-grounded multimodal framework that organizes heterogeneous measurements through PDE-associated operators. PhyMo follows a three-stage learning procedure: the physical-field encoder is first pretrained through field reconstruction under PDE residual supervision, its representations are subsequently aligned with visual embeddings in a shared latent space, and the fused multimodal represent
    
[^111]: EBRL：基于多粒度资源管理的异步具身强化学习

    EBRL: Asynchronous Embodied RL by Multi-Grained Resource Management

    [https://arxiv.org/abs/2609.27547](https://arxiv.org/abs/2609.27547)

    EBRL通过异步流水线调度器消除具身强化学习训练中的同步停顿，并利用细粒度的CPU/GPU资源池化管理与动态资源调整，大幅提升硬件资源利用效率。

    

    具身强化学习（RL）通过环境仿真、动作生成和模型更新的流水线来提升模型能力。这些阶段表现出异构的CPU与GPU需求，使得高效的资源利用变得困难。近期的系统为了提高效率，将推理采样与训练重叠执行，但独占式的GPU分配和同步屏障仍然造成大量硬件资源的浪费。在本文中，我们提出了EBRL，一个具有两项核心技术的异步具身强化学习训练系统。异步流水线调度器将推理采样与训练重叠执行，跨环境组对仿真和生成进行流水线化处理，并独立执行每个环境，从而消除了同步停顿。细粒度资源管理器将CPU核心和GPU流式多处理器汇聚为资源池，并利用阶段剖析信息与运行时反馈来调整资源配额和批次大小，以满足……

    arXiv:2609.27547v1 Announce Type: new  Abstract: Embodied reinforcement learning (RL) improves model capabilities with a pipeline of environment simulation, action generation, and model updates. These stages show heterogeneous CPU and GPU demands, making efficient resource utilization difficult. Recent systems overlap rollout (simulation and generation) with training for efficiency, but exclusive GPU allocation and synchronized barrier in rollout still leave substantial hardware resource waste. In this paper, we present EBRL, an asynchronous embodied RL training system with two core techniques. The asynchronous pipelined scheduler overlaps rollout and training, pipelines simulation and generation across environment groups, and carries out each environment independently, eliminating synchronization stalls. The fine-grained resource manager pools CPU cores and GPU streaming multiprocessors, and uses stage profiles and runtime feedback to adjust resource quotas and batch sizes to meet the
    
[^112]: 扩散模型在分布偏移下的鲁棒性

    Robustness of Diffusion Models under Distribution Shift

    [https://arxiv.org/abs/2609.27546](https://arxiv.org/abs/2609.27546)

    本文首次从理论上刻画了分布偏移下扩散模型的鲁棒分数估计，证明其可分解为学习参考分布的统计代价与随Wasserstein半径二次增长且极小极大最优的偏移代价，并构造了无需知晓偏移半径即可达到最优鲁棒速率的有限样本估计器。

    

    基于分数的扩散模型越来越多地被应用于底层数据分布可能与训练分布不一致的场景，然而现有的理论保证大多集中在无分布偏移的设定下。在本工作中，我们研究了参考分布在 Wasserstein 扰动下的鲁棒分数估计问题。针对 Ornstein-Uhlenbeck 扩散，我们证明鲁棒估计可以分解为两个基本组成部分：学习参考分布的统计代价与分布偏移的内在代价。后者随 Wasserstein 半径呈二次方增长，且这种依赖关系是极小极大最优的。我们构造了一个显式的有限样本估计器，在不知道偏移半径的情况下即可达到相应的鲁棒极小极大速率。当参考分布位于未知的低维子空间上时，统计项能够自适应于内在维度，而偏移代价保持不变。

    arXiv:2609.27546v1 Announce Type: cross  Abstract: Score-based diffusion models are increasingly considered in settings where the underlying data distribution may differ from the training distribution, yet existing theoretical guarantees largely focus on the no-shift setting. In this work, we study robust score estimation under Wasserstein perturbations of a reference distribution. For the Ornstein--Uhlenbeck diffusion, we show that robust estimation decomposes into two fundamental components: the statistical cost of learning the reference distribution and the intrinsic cost of distribution shift. The latter scales quadratically with the Wasserstein radius, and this dependence is minimax optimal. We construct an explicit finite-sample estimator achieving the resulting robust minimax rate without knowing the shift radius. When the reference distribution lies on an unknown low-dimensional subspace, the statistical term adapts to the intrinsic dimension while the shift cost remains unchan
    
[^113]: ProCredit：智能体强化学习中从结果奖励到进展信用的转变

    ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.27532](https://arxiv.org/abs/2609.27532)

    提出 ProCredit，利用可在中间状态上运行的验收检查，把与最终结果同样可验证的任务进展转化为逐步的信用信号，从而克服长程智能体强化学习中仅依赖结果奖励导致的训练信号稀疏、失败尝试无法区分、推进任务的步骤得不到应得信用等问题。

    

    长程智能体任务要求智能体通过一系列工具调用对环境进行修改，任务成败由最终状态决定。标准做法是在任务结束时给出单一的结果奖励，并对同一任务采样得到的多条轨迹进行比较。由此带来的问题是：当一组采样中没有任何成功轨迹时，训练便得不到任何信号；失败的尝试无法按照其接近完成的程度加以区分；而真正推动任务进展的步骤与仅仅查询环境的步骤会获得相同的信用。已有工作或将比较的单位从轨迹细化为步骤，或训练一个奖励模型来提供中间信号：前者仍然只能从最终成败中获取信号，后者则需要依赖模型来估计信号。我们观察到，用于判定成功的验收检查同样可以作用于中间状态，因此任务进展与最终结果一样是可验证的。我们提出 ProCredit，它将这种经过验证的进展……（摘要原文在此处截断）

    arXiv:2609.27532v1 Announce Type: cross  Abstract: Long-horizon agentic tasks require an agent to modify an environment through a sequence of tool calls, with success determined by the final state. The standard recipe assigns a single outcome reward at the end and compares trajectories sampled for the same task. As a result, a group with no successful trajectory yields no training signal, failed attempts cannot be told apart by how close they came to completion, and turns that advance the task receive the same credit as turns that only query the environment. Prior work refines the unit of comparison from the trajectory to the step, or trains a reward model to supply intermediate signal: the former still derives its signal from final success alone, and the latter estimates it with a model. We observe that the acceptance checks that decide success can also be run on intermediate states, so progress is as verifiable as the outcome. We propose ProCredit, which turns this verified progress 
    
[^114]: M3D-Net：面向乳腺X线摄影分类的空间上下文、特征重用与差分注意力的分层协调

    M3D-Net: Hierarchical Coordination of Spatial Context, Feature Reuse, and Differential Attention for Mammography Classification

    [https://arxiv.org/abs/2609.27523](https://arxiv.org/abs/2609.27523)

    M3D-Net通过分辨率感知的算子布局，将多尺度坐标注意力、有界动态特征重用和差分注意力进行分层协调，在乳腺X线摄影分类任务上取得了97.78%的最高验证准确率。

    

    乳腺图像分类既需要局部细节，也需要全局组织上下文，但随着表征层次的加深，这些线索可能会减弱。我们提出了M3D-Net，一种乳腺X线摄影编码器，通过分辨率感知的算子布局，分层协调多尺度坐标注意力、有界动态特征重用以及差分注意力。其中，阶段内特征检索保持了对早期特征的访问，坐标感知聚合整合了局部与全局上下文，而差分注意力则在粗分辨率下运行。我们在AISSLab乳腺X线摄影数据集上评估了纯图像分类，并在BrEaST超声数据集上评估了改编后的图像-临床模型。与EdgeNeXt、RepViT和TransXNet相比，所提出的实现取得了最高的记录验证准确率和训练后期准确率，以及最低的终点交叉熵损失。验证准确率分别达到97.78%和80.39%。这些结果支持对分层（原文在此处截断）

    arXiv:2609.27523v1 Announce Type: cross  Abstract: Breast image classification requires local detail and global tissue context, yet these cues can weaken as representations deepen. We present M3D-Net, a mammography encoder that hierarchically coordinates multi-scale coordinate attention, bounded dynamic feature reuse, and differential attention through resolution-aware operator placement. Within-stage retrieval preserves access to earlier features, coordinate-aware aggregation integrates local and global context, and differential attention operates at coarse resolutions. We evaluate image-only classification on AISSLab mammography and an adapted image--clinical model on BrEaST ultrasound. Against EdgeNeXt, RepViT, and TransXNet, the proposed implementations achieve the highest recorded validation accuracy and late-training accuracy, with the lowest endpoint cross-entropy loss. Validation accuracies reach 97.78\% and 80.39\%, respectively. These results support further evaluation of hie
    
[^115]: WhatWorkedBench：AI智能体实验理解能力基准测试

    WhatWorkedBench: Benchmarking Experimental Understanding in AI Agents

    [https://arxiv.org/abs/2609.27490](https://arxiv.org/abs/2609.27490)

    该论文提出了WhatWorkedBench基准，用于评估AI研究智能体的实验理解能力，即智能体在预算受限实验后预测组件变化如何影响实验结果的准确性。

    

    AI研究智能体需要可靠地了解它们的实验如何改变结果。我们提出了WhatWorkedBench来衡量实验理解能力，即在预算受限的实验之后，智能体对组件变化预测的准确性。智能体检查代码、选择测量方式，并提交一个响应面——一张预测每种组件设置配置得分的表格。穷举式CPU执行为在保持其他组件不变的情况下更改每个组件提供了参考效应。这些效应捕获了来自30个数据源和8种工作流类型的36个任务中的变化组合，共包含1248条配置记录。核心评估结合了覆盖所有八个系列的4,206条数值控制记录，以及原始六个系列中的108个智能体回合。在八项新测量中，成对效应岭回归在22个数据源中的15个上选出了最优配置，并在三个数据源上将所有效应误差控制在得分范围的10%以内。将高斯过程（GP）拟合到……

    arXiv:2609.27490v1 Announce Type: new  Abstract: AI research agents need reliable knowledge of how their experiments change outcomes. We introduce WhatWorkedBench to measure experimental understanding, the accuracy of predictions about component changes after budgeted experimentation. Agents inspect code, select measurements, and submit a response surface, a table predicting scores for every configuration of component settings. Exhaustive CPU execution supplies reference effects for changing each component while holding the others fixed. These effects capture combinations of changes across 36 tasks from 30 data sources and 8 workflow types, with 1248 configuration records. Core evaluation combines 4,206 numerical-control records across all eight families and 108 agent episodes across the original six. At eight new measurements, pair-effect ridge selects an optimum on 15 of 22 sources and limits every effect error to 10% of score range on three. Fitting a Gaussian process (GP) to the sa
    
[^116]: 学习往哪里看：面向时间序列预测与PPG-生命体征重建的共享相对对齐模块

    Learning Where to Look: A Shared Relative-Alignment Module for Time-Series Forecasting and PPG-to-Vital-Sign Reconstruction

    [https://arxiv.org/abs/2609.27473](https://arxiv.org/abs/2609.27473)

    本文提出ROOSTER——一个通过逐头可学习的周期梳状偏置自动学习目标与条件序列之间对齐方式的共享条件化模块，它能同时胜任PPG到生命体征重建和多变量时间序列预测两类任务，并在多项基准上超越现有基线。

    

    PPG-生命体征重建是将腕部佩戴的光电容积脉搏波（PPG）转换为心电图（ECG）等临床波形。长时程多变量时间序列预测则支撑着能源、天气和交通领域的规划任务。这两类任务都是从条件序列生成目标序列，而现有模型将每个目标位置读取条件序列的位置硬编码为同位置复制或季节性循环，因此两者都无法在任务间迁移。我们提出ROOSTER，一个通过学习这种对应关系、能够同时处理生命体征重建和时间序列预测的条件化模块。其核心是作用于目标-条件偏移量上的周期梳状偏置，其中心、周期和锐度在每个注意力头中均为可学习参数，因此单个模块能够自行确定采用恒等对齐还是季节性滞后，并报告其所发现的对齐方式。在基于PPG的生命体征重建任务上，ROOSTER在四个心率和呼吸率基准上超越了已发表的基线方法。在多变量时间序……（原文摘要在此处截断）

    arXiv:2609.27473v1 Announce Type: new  Abstract: PPG-to-vital-sign reconstruction turns a wrist-worn photoplethysmogram into clinical waveforms such as the ECG. Long-horizon multivariate time-series forecasting underpins planning in energy, weather, and traffic. Both generate a target sequence from a condition sequence, and current models hard-code where each target position reads it, as a same-position copy or seasonal recurrence, so neither transfers between tasks. We propose ROOSTER, one conditioning module that handles vital-sign reconstruction and time-series forecasting alike by learning this correspondence. Its core is a periodic-comb bias over the target-condition offset whose center, period, and sharpness are learned per head, so one module settles on the identity alignment or a seasonal lag and reports which it found. On vital-sign reconstruction from PPG, ROOSTER outperformed the published baselines on four heart-rate and respiratory-rate benchmarks. On multivariate time-ser
    
[^117]: DeltaS：读取门控线性注意力状态以实现流式视频中的KV缓存驱逐

    DeltaS: Reading the Gated Linear Attention State for KV Cache Eviction in Streaming Video

    [https://arxiv.org/abs/2609.27470](https://arxiv.org/abs/2609.27470)

    该论文提出利用门控delta线性注意力循环状态在帧块上的变化量作为信号，在问题到来之前决定流式视频KV缓存的驱逐策略，无需代理查询或额外计算。

    

    近年来，视频-语言模型越来越多地采用混合架构，通过交错使用线性注意力层和全注意力层来高效处理长上下文。虽然线性注意力的循环状态大小保持固定，但全注意力的KV缓存会随着视频流的持续增长而不断膨胀，因此在有限的内存预算下必须进行驱逐。流式场景中的关键挑战在于：驱逐必须在问题到来之前发生，因此必须在不知道问题的情况下决定保留哪些内容。现有的驱逐方法从KV缓存本身获取token分数，利用位置、注意力或键值表示，而基于注意力的分数还需要代理查询或额外的计算。混合骨干网络提供了另一种信号来源。在门控delta线性注意力中，循环状态由每个输入与当前状态已能检索到的内容之间的残差来更新，因此其在一块帧序列上的变化反映了……（摘要在此处截断）

    arXiv:2609.27470v1 Announce Type: cross  Abstract: Recent video-language models increasingly adopt hybrid architectures that interleave linear and full attention layers for efficient long-context processing. While the recurrent state of linear attention remains fixed in size, the KV cache of full attention continues to grow with the video stream, making eviction necessary under a bounded memory budget. The key challenge in streaming is that eviction must occur before the question arrives, so what to retain has to be decided without the question. Existing eviction methods derive token scores from the KV cache itself, using position, attention, or key-value representations, and attention-based scores further require proxy queries or extra computation. Hybrid backbones offer another source of signal. In gated-delta linear attention, the recurrent state is updated by the residual between each input and what can already be retrieved from the state, so its change over a chunk of frames refle
    
[^118]: 面向量子云编排中成本与延迟权衡的量子强化学习

    Quantum Reinforcement Learning for Cost and Delay Tradeoffs in Quantum Cloud Orchestration

    [https://arxiv.org/abs/2609.27446](https://arxiv.org/abs/2609.27446)

    该论文提出QRLQ框架，将参数化量子电路与D3QN相结合用于量子云任务调度，能够动态权衡成本与延迟，相比启发式基线平均成本降低5-11%。

    

    量子云计算通过量子即服务（QaaS）模式提供对量子计算资源的访问。然而，对本质上异构的量子资源采用统一的基于时间的定价方式，极大地增加了任务编排的复杂性，尤其是在处理执行成本与系统性能之间的权衡时。启发式方法依赖于预定义的调度规则，而经典深度强化学习（DRL）模型在此场景下可能需要更多的可训练参数。受参数化量子电路（PQC）作为紧凑函数逼近器的潜力启发，我们提出了QRLQ，这是一个成本-延迟感知的量子云调度框架，将PQC与决斗双深度Q网络（D3QN）相结合，以动态地同时兼顾成本和延迟。仿真结果表明，QRLQ相比启发式基线方法实现了更低的平均成本和延迟，平均成本降低了5-11%。

    arXiv:2609.27446v1 Announce Type: cross  Abstract: Quantum cloud computing, delivered through the quantum-as-a-service (QaaS) model, provides access to quantum computing resources. However, applying uniform time-based pricing across fundamentally heterogeneous quantum resources significantly complicates task orchestration, particularly when addressing the tradeoff between execution costs and system performance. While heuristic methods rely on predefined scheduling rules, classical deep reinforcement learning (DRL) models may require more trainable parameters in this setting. Motivated by the potential of parameterised quantum circuits (PQCs) as compact function approximators, we propose QRLQ, a cost-delay-aware quantum cloud scheduling framework integrating PQCs with a dueling double deep Q-network (D3QN) to dynamically account for both cost and delay. Our simulation results show that QRLQ achieves lower mean cost and delay than the heuristic baselines, achieving a 5-11% lower mean cos
    
[^119]: 基于任务条件化潜在对齐实现跨会话稳定神经解码的脑机接口方法

    Stable Neural Decoding Across Sessions via Task-Conditioned Latent Alignment for Brain-Machine Interfaces

    [https://arxiv.org/abs/2609.27441](https://arxiv.org/abs/2609.27441)

    提出任务条件化潜在对齐（TCLA）框架，通过学习固定的共享潜在空间并按任务条件分别对齐源与目标神经分布，显著提升了脑机接口跨会话长期神经解码的稳定性。

    

    在侵入式脑机接口（BMI）中实现稳定的长期神经解码仍然具有挑战性，原因在于不同会话之间记录的神经群体存在变化。当前的潜在对齐方法在跨会话适应过程中可能忽略任务相关的结构。我们提出了任务条件化潜在对齐（TCLA），这是一个通过学习共享潜在空间来稳定神经解码的框架。TCLA利用神经重建和连续行为监督来学习低维源表示。在目标会话适应过程中，共享表示保持固定，同时通过为每个任务条件分别对齐源分布和目标分布，将目标神经活动映射到源潜在空间中。我们在涵盖多个任务的七个非人类灵长类动物数据集上评估了TCLA。在长期跨会话评估中，TCLA实现了0.476±0.014的平均R²，负R²失败率为……

    arXiv:2609.27441v1 Announce Type: new  Abstract: Achieving stable long-term neural decoding in invasive brain-machine interfaces (BMIs) remains challenging due to variations in recorded neural populations across sessions. Current latent alignment approaches may overlook task-dependent structure during cross-session adaptation. We propose Task-Conditioned Latent Alignment (TCLA), a framework that stabilizes neural decoding by learning a shared latent space. TCLA learns a low-dimensional source representation using neural reconstruction and continuous behavioral supervision. During target-session adaptation, the shared representation is fixed, while target neural activity is mapped into the source latent space by aligning source and target distributions separately for each task condition. We evaluated TCLA on seven nonhuman primate datasets spanning multiple tasks. In long-term cross-session evaluation, TCLA achieved a mean $R^2$ of $0.476\pm0.014$ with a negative $R^2$ failure rate of o
    
[^120]: 面向多约束指令遵循的反事实约束条件化在线策略蒸馏

    Counterfactual Constraint-Conditioned On-Policy Distillation for Multi-Constraint Instruction Following

    [https://arxiv.org/abs/2609.27421](https://arxiv.org/abs/2609.27421)

    提出CC-OPD方法，颠覆传统蒸馏的监督方向，通过从教师模型条件中依次消融各约束并利用逐词元概率差分构建每约束的监督信号，从而提升大语言模型的多约束指令遵循能力。

    

    多约束指令遵循要求模型在多个同时生效的约束条件下对查询作出回应。即使是强大的指令微调模型，也经常违反其中一些约束。现有方法要么利用来自外部验证器或学习型评估器的序列级或词元级强化学习奖励来增强监督，要么使用针对单一全上下文教师模型的在线策略蒸馏（OPD），但随着同时生效的约束数量增多，该教师模型的概率质量会被稀释。我们提出了CC-OPD（反事实约束条件化在线策略蒸馏），该方法颠覆了蒸馏中标准的监督-生成方向。CC-OPD并非用学生模型看不到的信息来丰富教师模型，而是依次从教师模型的条件中消融各个约束，并从由此产生的逐词元概率差分中构建针对每个约束的信号。由此产生的逐词元留一法对数似然……

    arXiv:2609.27421v1 Announce Type: new  Abstract: Multi-constraint instruction following requires a model to respond to a query under many simultaneously active constraints. Even strong instruction-tuned models still routinely violate some of them. Existing approaches either augment supervision with sequence- or token-level RL rewards from external verifiers or learned graders, or use on-policy distillation (OPD) against a single full-context teacher whose probability mass becomes diluted as more constraints become simultaneously active. We propose CC-OPD (Counterfactual Constraint-Conditioned On-Policy Distillation), which inverts the standard supervision-generation direction in distillation. Rather than enriching the teacher with information beyond what the student sees, CC-OPD ablates each constraint from the teacher's conditioning in turn, and constructs the per-constraint signal from the resulting per-token probability differentials. The resulting per-token leave-one-out log-likeli
    
[^121]: 当标签稀缺时：一种用于振动诊断的振荡状态空间模型

    When Labels Are Scarce: An Oscillatory State Space Model for Vibration Diagnosis

    [https://arxiv.org/abs/2609.27411](https://arxiv.org/abs/2609.27411)

    提出了仅含约4万参数的紧凑振荡状态空间模型DualRes，通过融合两种互补频谱视图与选择性振荡记忆，在标签稀缺的振动故障诊断任务上以极少标注数据（每类约6秒）实现了最先进的性能。

    

    基于振动的机器故障诊断需要在标注稀缺的故障录音上进行学习，同时还要满足边缘设备本地推理的计算约束。我们提出了DualRes，一种紧凑的振荡状态空间模型，它结合了振动的两种互补频谱视图，能够捕捉快速变化和精细的频率结构。时间对齐的视图由选择性振荡记忆处理，该记忆模块学习应当保留时间模式多长时间。该编码器仅包含39,528个参数。我们在六个轴承数据集和一个齿轮箱基准上评估了监督学习性能，并附带一个额外的齿轮箱初步实验。通过录音级别的数据划分以及对已标注时长的明确核算，我们将数据效率与对相关样本的重复暴露区分开来。在主要的齿轮箱基准上，DualRes在七个标签预算中的六个上取得了九种评估方法中最先进的性能。每个类别仅需约六秒的标注数据……

    arXiv:2609.27411v1 Announce Type: new  Abstract: Machine fault diagnosis from vibration requires learning from scarce labelled fault recordings while meeting the computational constraints of edge devices for local inference. We introduce DualRes, a compact oscillatory state-space model that combines two complementary spectral views of vibration, capturing rapid changes and fine frequency structure. Time-aligned views are processed by selective oscillatory memory, which learns how long to retain temporal patterns. The encoder contains 39,528 parameters. We evaluate supervised learning across six bearing datasets and a gearbox benchmark, with an additional gearbox pilot. Recording-level splits and explicit accounting of labelled duration distinguish data efficiency from repeated exposure to correlated samples. On the main gearbox benchmark, DualRes achieves state-of-the-art performance among the nine evaluated methods at six of seven label budgets. With about six labelled seconds per cla
    
[^122]: 面向生物多样性监测的主动学习：从标签效率到可靠的生态推断

    Active Learning for Biodiversity Monitoring: From Label Efficiency to Reliable Ecological Inference

    [https://arxiv.org/abs/2609.27409](https://arxiv.org/abs/2609.27409)

    主动学习虽能显著降低生物多样性监测中的标注成本，但其非随机的样本选择方式使标签无法用于验证、校准和阈值选择，论文主张在有限专家预算下统筹模型训练与可靠生态推断的资源分配。

    

    专家标注能力的有限是生物多样性监测中普遍存在的制约因素。被动声学记录仪和红外相机产生数据的速度快于专家分析的速度。机器学习（ML）模型能够大规模处理这些数据，但其可靠性取决于标注样本的质量、数量和覆盖范围，因此专家时间仍然是一个瓶颈。主动学习（AL）通过在固定标注预算下选择预期最能提升模型性能的样本，缓解了这一瓶颈；已发表的证据表明，它可以减少达到目标性能所需的标签数量。然而，监测项目面临一个更广泛的问题：应如何划分有限的专家预算，才能使模型训练、验证以及基于模型输出的生态估计都保持可靠？由于主动学习以非随机的方式选择样本，其标注结果不适用于验证、校准或阈值选择，这一矛盾很少受到关注（摘要原文在此处截断）。

    arXiv:2609.27409v1 Announce Type: new  Abstract: Limited expert annotation capacity is a pervasive constraint in biodiversity monitoring. Passive acoustic recorders and camera traps generate data faster than experts can analyse them. Machine learning (ML) models can process these data at scale, but their reliability depends on the quality, quantity, and coverage of labelled samples, so expert time remains a constraint. Active learning (AL) eases this bottleneck by selecting, under a fixed annotation budget, the samples expected to improve a model most, and published evidence shows it can reduce the labels needed to reach a target performance. Monitoring programmes, however, face a broader question: how should a limited expert budget be divided so that model training, validation, and the ecological estimates built on model outputs all remain reliable? Because AL selects samples non-randomly, its labels are unsuitable for validation, calibration, or threshold selection, a tension rarely 
    
[^123]: EvoAudio：面向音频理解的递归自我改进

    EvoAudio: Recursive Self-Improvement for Audio Understanding

    [https://arxiv.org/abs/2609.27389](https://arxiv.org/abs/2609.27389)

    提出EvoAudio，首个在闭环中同时演化模型、音频波形、问题和难度的递归自我改进系统，无需新的人工标注即可通过可验证监督与强化学习持续提升音频理解能力。

    

    音频语言模型对“说了什么”的理解远好于对“听起来如何”的理解。弥合这一差距仅靠数据还不够：详细的声学标注成本高昂，来自更强模型的标签会继承其错误和局限，而固定的数据无法随着学习者的进步而自我调整。因此，我们提出了 EvoAudio，一个用于音频理解的递归自我改进系统。据我们所知，这是首个在同一闭环中同时演化模型、波形、问题和难度的系统。EvoAudio 利用当前模型的表现来设定下一轮训练数据的重点和难度。随后，音频工具库构建问题，其答案源自音频本身的生成方式，从而在无需新的人工标注的情况下提供可验证的监督信号。强化学习更新模型，验证环节决定其是否进入下一轮演化。在13轮迭代中，EvoAudio 改进了五个采用不同音频编码器和语言骨干的模型。

    arXiv:2609.27389v1 Announce Type: cross  Abstract: Audio language models understand what is said far better than how it sounds. Closing this gap takes more than data. Detailed acoustic annotation is costly, labels from stronger models inherit their errors and limits, and fixed data cannot adapt as the learner improves. We therefore propose EvoAudio, a recursive self-improvement system for audio understanding. To our knowledge, it is the first to evolve the model, waveforms, questions, and difficulty in one closed loop. EvoAudio uses the current model's performance to set the focus and difficulty of the next training data. A library of audio tools then constructs questions whose answers follow from how the audio was made, providing verifiable supervision without new human annotation. Reinforcement learning updates the model, and validation decides whether it enters the next evolution round. Across 13 rounds, EvoAudio improves five models with different audio encoders and language backbo
    
[^124]: 预测工作流基准：利用预算约束的预测工具评估语言模型决策

    Forecast Workflow Bench: Evaluating Language-Model Decisions with Budgeted Forecast Tools

    [https://arxiv.org/abs/2609.27385](https://arxiv.org/abs/2609.27385)

    FWBench 提出了一个通过预算约束下的时间序列预测工具使用来评估语言模型决策能力的基准，发现 GPT-6 Astra 仅用 2.5% 的预算有选择地购买短时程预测即可胜过固定策略，首次实现了对决策质量与预测成本权衡的可复现评估。

    

    时间序列基础模型（TSFM）为运营决策提供预测，但仅凭准确性并不能决定其价值。评估使用这些模型的智能体需要同时衡量决策质量与预测成本。FWBench 在 1,251 个电力和公共自行车租赁案例上，使用固定的预测工具和模拟的容量合同来评估这一能力。智能体需要选择模型、历史数据长度和预测时程，然后提交容量方案以最小化给定的损失-成本目标。我们评估了两个托管配置和八个本地配置（包括小型语言模型），并对本地模型分别在有 TSFM 和无 TSFM 的情况下进行了测试。结果显示，GPT-6 Astra 有选择地购买了廉价的短时程预测，仅使用了 2.5% 的预算，并且在用三种损失-成本权重对所保存的决策进行评分时，其表现优于固定策略。FWBench 使得对语言模型如何在成本约束下选择和使用时间序列预测进行决策的可复现评估成为可能。

    arXiv:2609.27385v1 Announce Type: cross  Abstract: Time-series foundation models (TSFMs) provide forecasts for operational decisions, but accuracy alone does not determine their value. Evaluating agents that use these models requires measuring decision quality and forecast cost. FWBench evaluates this capability on 1,251 electricity and cycle-hire cases using fixed forecast tools and simulated capacity contracts. Agents select models, histories and horizons, then submit capacities to minimize a stated loss-cost objective. We evaluated two hosted and eight local configurations, including small language models, and tested local models with and without TSFMs. GPT-6 Astra bought inexpensive short-horizon forecasts selectively, using 2.5% of the budget, and outperformed fixed policies when the saved decisions were scored with three loss-cost weightings. FWBench enables reproducible evaluation of how language models select and use time-series forecasts to make decisions under cost constraint
    
[^125]: 注意力路由早期即稳定：循环语言模型的工作集推理

    Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models

    [https://arxiv.org/abs/2609.27373](https://arxiv.org/abs/2609.27373)

    该论文发现循环语言模型的注意力路由在早期循环步骤即趋于稳定，据此提出无需训练的WISE推理方法：早期用全局注意力发现稀疏工作集，后续步骤重用该支撑集，在保持推理质量的同时大幅减少重复的全局注意力计算。

    

    循环语言模型通过反复应用共享的网络模块来精炼潜在表示，但标准推理在每个循环步骤都会重新计算全局注意力。我们研究了注意力在循环深度上的动态变化，发现注意力的支撑集和分布比隐藏状态和注意力输出更早地稳定下来。这提示了一种两阶段结构：早期步骤发现相关上下文的稀疏工作集，而后续步骤则在基本相同的路由支撑上精炼表示。受此结构启发，我们提出了 WISE（基于支撑集利用的工作集推理），这是一种无需训练的方法，它在早期循环阶段使用不受限制的全局注意力，而在后续阶段重用直接发现的块结构化支撑，同时保持循环深度和支撑内注意力计算的动态性。受控干预实验表明，循环中的发现过程非常重要，而仅重用支撑

    arXiv:2609.27373v1 Announce Type: new  Abstract: Recurrent language models repeatedly apply shared network blocks to refine latent representations, but standard inference recomputes global attention at every recurrent step. We study attention dynamics across recurrent depth and find that attention support and distributions stabilize substantially earlier than hidden states and attention outputs. This suggests a two-stage structure: early steps discover a sparse working set of relevant context, while later steps refine representations over largely the same routing support. Motivated by this structure, we introduce WISE (Working-set Inference with Support Exploitation), a training-free method that uses unrestricted global attention during early recurrence and later reuses directly discovered block-structured support while keeping recurrent depth and within-support attention computation dynamic. Controlled interventions show that recurrent discovery is important and that support-only reus
    
[^126]: 基于AUC边界的无异常自优化

    Anomaly-Free Self-Optimization via AUC Bounds

    [https://arxiv.org/abs/2609.27362](https://arxiv.org/abs/2609.27362)

    本文提出将AUC边界作为可微分、无需异常数据的目标函数，直接优化异常检测系统的连续参数（包括集成权重和可学习的分数重缩放机制），突破了传统有限候选集选择的限制，在多个数据集和嵌入模型上实现了显著的性能提升。

    

    异常情况是罕见的，且在开发阶段往往无法获得异常数据，这使得难以确定哪些异常检测模型和配置能够泛化到未见过的异常上。近期的方法通过生成伪异常，并利用ROC曲线下可达面积的边界（AUC），从有限的候选集合中选择最优配置来应对这一挑战。与之不同，我们将AUC边界用作一个可微分、无异常的目标函数，直接优化异常检测系统的连续参数。我们通过优化集成权重并引入一种可学习的分数重缩放机制来自适应调整伪异常分数，从而展示了这一框架，使优化能够超越预定义的候选集合。在多个数据集和嵌入模型上的实验表明，基于AUC边界的优化相比传统模型选择和先前的开发方法取得了显著的性能提升。

    arXiv:2609.27362v1 Announce Type: new  Abstract: Anomalies are rare, and anomalous data are often unavailable during development, making it difficult to determine which anomaly detection models and configurations will generalize to unseen anomalies. Recent approaches address this challenge by generating pseudo-anomalies and using bounds on the achievable area under the ROC curve (AUC) to select the optimal configuration from a finite set of candidates. Instead, we use the AUC bound as a differentiable, anomaly-free objective for directly optimizing continuous parameters of anomaly detection systems. We demonstrate this framework by optimizing ensemble weights and introducing a learnable score-rescaling mechanism that adapts pseudo-anomaly scores, enabling optimization beyond a predefined candidate set. Experiments across multiple datasets and embedding models show that AUC-bound optimization achieves significant performance gains over conventional model selection and prior development-
    
[^127]: 基于保留-遗忘损失景观交互视角的量化鲁棒机器遗忘

    Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction

    [https://arxiv.org/abs/2609.27355](https://arxiv.org/abs/2609.27355)

    本文提出一种量化鲁棒的机器遗忘框架，通过基于曲率的敏感权重判据和敏感度引导的噪声正则化，将模型收敛引导至更平滑的极小值，使遗忘效果在量化压缩后依然保持鲁棒，同时维持整体模型效用。

    

    机器遗忘通过移除私有或受版权保护训练数据的影响，确保大语言模型（LLM）的合规性。然而，由于大语言模型在实际部署中通常会经历训练后压缩（如量化），已有观察发现遗忘效果会被显著削弱，且遗忘行为的退化比模型效用的退化更为严重。本文提出了一种量化鲁棒的机器遗忘框架，使遗忘对量化具有鲁棒性，同时保持整体模型效用。我们通过损失景观的视角来分析这一差距。具体而言，我们的分析揭示了一种基于曲率的判据，能够精确定位已遗忘模型中导致非鲁棒遗忘和效用降低的敏感权重。因此，我们提出了敏感度引导的噪声正则化方法，将其应用于敏感参数上，引导模型收敛至具有一致较低遗忘损失的更平滑极小值。

    arXiv:2609.27355v1 Announce Type: cross  Abstract: Unlearning ensures LLM compliance by removing the influence of private or copyrighted training data. However, since LLM models typically undergo post-training compression, like quantization, in practical deployment, it has been observed that the unlearning effect can be substantially weakened, with the forgetting behavior degrading more severely than that of model utility. This paper proposes a quantization-robust unlearning framework that makes forgetting robust to quantization while maintaining overall model utility. We analyze this gap through the lens of loss landscape. Specifically, our analysis reveals a curvature-based criteria that pinpoints sensitive weights in the unlearned model that leads to both non-robust forgetting and reduced utility. We therefore propose sensitivity-guided noisy regularization, which is applied on the sensitive parameters to steer the model convergence towards a smoother minima of uniformly low forget 
    
[^128]: MolDesignBench：评估基于大语言模型智能体的场景化分子设计

    MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded Molecular Design

    [https://arxiv.org/abs/2609.27349](https://arxiv.org/abs/2609.27349)

    提出了MolDesignBench——一个面向真实场景的分子设计基准，包含2000个融合隐式设计需求与显式约束的生成与优化任务并需要调用17种专业化学工具，实验表明当前前沿大语言模型智能体在这些真实分子设计任务上的成功率仍然很低。

    

    真实世界的分子设计对基于大语言模型（LLM）的智能体而言仍然充满挑战。它要求智能体理解设计背景、满足多重约束条件、识别不可行的规格要求，并对多步骤的工具输出进行推理。现有的基准测试未能捕捉这种复杂性，而是侧重于明确且狭窄的约束、仅包含可解决的问题以及单一路径的解决方案。为了填补这一空白，我们提出了MolDesignBench，这是一个基于真实场景的分子设计基准，用于评估工具增强的LLM智能体。MolDesignBench包含2K个生成与优化实例，这些实例将设计叙述中隐含的需求与显式的性质和官能团约束相结合（包括不可行的案例），并要求有效使用17种专业化学工具。在多种前沿LLM上的实验显示成功率较低——表现最佳的模型仅达到……

    arXiv:2609.27349v1 Announce Type: new  Abstract: Real-world molecular design remains challenging for large language model (LLM)-based agents. It requires them to interpret design contexts, satisfy multiple constraints, identify infeasible specifications, and reason over multi-step tool outputs. Existing benchmarks do not capture this complexity, focusing instead on explicit and narrow constraints, only feasible problems, and single-path solutions. To address this gap, we propose MolDesignBench, a scenario-grounded benchmark that more closely reflects real-world molecular design for evaluating tool-augmented LLM agents. MolDesignBench comprises 2K generation and optimization instances that combine implicit requirements embedded in design narratives with explicit property and functional-group constraints, including infeasible cases, and require the effective use of 17 specialized chemistry tools. Experiments across diverse frontier LLMs reveal low success rates--with the best achieving o
    
[^129]: 利用大语言模型演化可检查的O-RAN网络切片xApp

    Evolving Inspectable O-RAN Slicing xApps with LLMs

    [https://arxiv.org/abs/2609.27337](https://arxiv.org/abs/2609.27337)

    本文提出用大语言模型将O-RAN网络切片控制器自动演化为紧凑且可读、可编辑的Python程序，取代决策逻辑不可解释的深度强化学习神经网络策略，在保留自适应资源分配能力的同时，让运营商能够直接检查和修改控制逻辑，并在真实5G测试平台上验证了其有效性。

    

    开放无线接入网（O-RAN）切片xApp必须在满足服务等级协议（SLA）的同时，根据不断变化的信道条件和流量需求自适应地调整资源分配。深度强化学习虽然能够产生自适应策略，但其分配规则仍然隐藏在神经网络参数之中。本文的目标是在保留这种适应性的同时，使控制器的决策逻辑能够被运营商直接检查和编辑。研究者使用大语言模型（LLM）将切片控制器演化为紧凑的Python程序，其决策逻辑在优化之后依然保持可读和可编辑。LLM在离线状态下提出并迭代修改候选控制器，由经过校准的模拟器对其进行评分，最终选定的决策模块无需任何修改即可直接运行在O-RAN控制路径中。在NSF POWDER 5G测试平台上的实验表明，演化出的控制器能够在保证型切片的吞吐量目标因持续信道衰落而无法达成时，释放该切片的资源，从而改善尽力而为（best-effort）服务的性能。

    arXiv:2609.27337v1 Announce Type: cross  Abstract: Open RAN (O-RAN) slicing xApps must adapt resource allocations to changing channel conditions and traffic demands while meeting service-level agreements (SLAs). Deep reinforcement learning can produce adaptive policies, but their allocation rules remain encoded in neural-network parameters. Our goal is to retain this adaptability while making the controller's decision logic directly inspectable and editable by operators. We use a large language model (LLM) to evolve slicing controllers as compact Python programs whose decision logic remains readable and editable after optimization. The LLM proposes and revises candidates offline, while a calibrated simulator scores them, and the selected decision module runs unchanged in the O-RAN control path. On the NSF POWDER 5G testbed, the evolved controller releases resources from a guaranteed slice whose throughput target becomes unattainable under a sustained channel fade, improving best-effort
    
[^130]: 椭圆界面问题的混合迭代深度Ritz方法

    A Hybrid Iterative Deep Ritz Method for Elliptic Interface Problems

    [https://arxiv.org/abs/2609.27325](https://arxiv.org/abs/2609.27325)

    本文提出了一种求解椭圆界面问题的混合迭代深度Ritz方法（H-IDRM），通过新的混合形式与水平集神经网络架构，仅采用体积表示避免了显式界面采样，并给出了涵盖神经网络近似、蒙特卡洛近似、迭代格式和惩罚参数误差的完整误差分析。

    

    在本工作中，我们针对二阶椭圆算子的一类界面问题提出了一种混合迭代深度Ritz方法（H-IDRM）。该方法基于问题的一个新的混合形式，并通过求解一系列凸极小化问题来实现。我们采用水平集神经网络架构，利用界面的水平集表示来适应解和通量的分片光滑性。该方法仅涉及体积表示，而非界面上的对偶配对，从而避免了显式界面采样——这在复杂界面几何情况下十分不便。此外，我们对该方法进行了分析，包括神经网络近似、蒙特卡洛近似、迭代格式以及惩罚参数所产生的误差。数值实验表明，H-IDRM在高维区域和复杂界面问题上优于现有的神经求解器。

    arXiv:2609.27325v1 Announce Type: cross  Abstract: In this work, we propose a hybrid iterative deep Ritz method (H-IDRM) for a class of interface problems for second-order elliptic operators. It is based on a new mixed formulation of the problem and involves solving a sequence of convex minimization problems. We employ a level-set neural network architecture, featuring a level-set representation of the interface, to accommodate the piecewise smoothness of the solution and the flux. The approach involves only volumetric representations instead of duality pairing on the interface and avoids explicit interface sampling that is inconvenient for complex interface geometries. Further, we present an analysis of the method, including the errors arising from the neural network approximation, Monte Carlo approximation, iterative scheme, and penalty parameters. Numerical experiments indicate that the H-IDRM outperforms existing neural solvers on problems with high-dimensional domains, intricate i
    
[^131]: 将安全转化为能力：通过安全过滤强化学习实现最小可利用性的机器人策略

    Turning Safety into Competence: Minimally Exploitable Robot Policies via Safety-Filtered Reinforcement Learning

    [https://arxiv.org/abs/2609.27312](https://arxiv.org/abs/2609.27312)

    提出了S2C两阶段强化学习框架，通过对抗性强化学习训练鲁棒安全过滤器并将其与竞争任务学习分离，证明了安全过滤可保持策略的不可利用性，使机器人在竞争任务中胜率最高且最难被攻击利用。

    

    在竞争性任务中部署的机器人必须在保证安全的前提下智胜对手。现有方法（包括安全强化学习）通常训练单一策略来同时实现任务成功和避免失败，这种耦合会使训练复杂化，并使学到的策略容易被蓄意攻击所利用。我们提出了S2C（Safety to Competence），一个将安全综合与竞争性任务学习相分离的两阶段强化学习框架。我们将竞争性交互形式化为安全关键的马尔可夫博弈，并证明当所有参与者都遵循安全机动时，完美过滤能够保持策略的不可利用性。S2C通过对抗性强化学习学习一个鲁棒的安全过滤器，在任务策略训练期间将其嵌入环境中，并在部署时保留相同的过滤器。在模拟触地得分游戏的实验中，S2C优于八个安全强化学习基线方法，取得了最高的胜率和Elo评分，以及最低的可利用性。

    arXiv:2609.27312v1 Announce Type: cross  Abstract: Robots deployed for competitive tasks must outmaneuver their opponents without sacrificing safety. Existing approaches, including safe reinforcement learning (RL), train a single policy to achieve task success and avoid failures simultaneously. This coupling can complicate training and leave the learned policy exploitable by deliberate attacks. We propose Safety to Competence (S2C), a two-stage RL framework that separates safety synthesis from competitive task learning. We formulate competitive interactions as safety-critical Markov games and prove that perfect filtering preserves policy non-exploitability when all players commit to safe maneuvers. S2C learns a robust safety filter via adversarial RL, embeds it in the environment during task policy training, and retains the same filter at deployment. In simulated touchdown games, S2C outperforms eight safe RL baselines, achieving the highest win rate and Elo rating, and the lowest expl
    
[^132]: FairTest：面向多智能体强化学习系统的基于搜索的公平性测试

    FairTest: Search-Based Fairness Testing for Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2609.27309](https://arxiv.org/abs/2609.27309)

    本文提出了FairTest，一种基于搜索的测试方法，通过三个适应度函数（已执行运行的公平性、预测公平性、决策不确定性）引导搜索并结合测试优先级排序，以发现多智能体强化学习策略中的不公平执行。

    

    多智能体强化学习（MARL）训练一组共享同一环境并共同学习策略的智能体。训练以最大化团队回报为目标，但高回报并不意味着在每一回合中奖励都能在智能体之间公平分配。测试是发现深度强化学习故障的一种成熟方法，然而目前很少有方法关注MARL的公平性问题。在本工作中，我们提出了FairTest，这是一种基于搜索的测试方法，旨在发现MARL策略的不公平执行。该方法的设计将搜索引导与测试优先级排序相结合。搜索引导通过三个适应度函数对每个候选进行评分：一个用于衡量已执行运行的公平性，另一个从抽象状态和公平性特征预测公平性，第三个则读取策略的决策不确定性。交叉和变异操作从已观察到的执行中衍生出更多候选。优先级排序机制对候（选执行进行排序，原文在此处截断）

    arXiv:2609.27309v1 Announce Type: cross  Abstract: Multi-agent Reinforcement Learning (MARL) trains a team of agents that share one environment and learn their policies together. Training maximizes the team return, and a high return does not imply that the rewards are shared fairly among the agents in every episode. Testing is an established way to discover the failures of deep reinforcement learning, yet few methods address the fairness of MARL. In this work, we propose FairTest, a search-based testing approach that seeks the unfair executions of a MARL policy. The design combines search guidance with test prioritization. The guidance scores each candidate with three fitness functions. One measures the fairness of the runs already performed, another predicts the fairness from abstract states and fairness features, and the third reads the decision uncertainty from the policy. Crossover and mutation derive further candidates from the observed executions. The prioritization ranks the can
    
[^133]: 基于演化变分自回归网络的离散扩散模型

    Discrete Diffusion Models via Evolving Variational Autoregressive Networks

    [https://arxiv.org/abs/2609.27306](https://arxiv.org/abs/2609.27306)

    提出一种利用变分自回归网络参数化归一化概率分布的离散扩散模型，通过显式马尔可夫跳跃算子控制加噪与去噪动力学，将归一化离散扩散模型成功扩展至高维晶格上的自旋系统，并准确计算了二维和三维伊辛模型的自由能、能量、磁化强度等热力学量。

    

    arXiv:2609.27306v1 公告类型：新论文 摘要：传统的基于分数的扩散模型在学习分数函数时并不表示归一化密度，而易于处理的归一化模型则能同时支持采样和直接似然评估。最近的一种张量网络方法提供了这种表示，但在很大程度上仅限于低维晶格。本文提出了一种离散扩散模型，该模型使用变分自回归网络对归一化概率分布进行参数化。显式的马尔可夫跳跃算子控制前向加噪和反向去噪动力学，将具有归一化分布的离散扩散模型扩展到更高维晶格上的自旋系统。我们将该框架应用于有序、临界和无序相区中的二维和三维伊辛模型，准确计算了包括自由能、能量和磁化强度在内的热力学量。我们进一步将该框架与蒙特卡罗采样相结合，使用自适应（方法）……

    arXiv:2609.27306v1 Announce Type: new  Abstract: Conventional score-based diffusion models learn scores without representing normalized densities, whereas tractable normalized models support both sampling and direct likelihood evaluation. A recent tensor-network approach provides such a representation but is largely restricted to low-dimensional lattices. Here we introduce a discrete diffusion model that parameterizes normalized probability distributions using variational autoregressive networks. Explicit Markov jump operators govern the forward noising and reverse denoising dynamics, extending discrete diffusion models with normalized distributions to spin systems on higher-dimensional lattices. We apply this framework to the two- and three-dimensional Ising models across ordered, critical, and disordered regimes, accurately computing thermodynamic quantities including free energy, energy, and magnetization. We further integrate the framework with Monte Carlo sampling, using adaptive 
    
[^134]: Live Assistant：学习在真实直播社交流中是否、何时以及向谁提供协助

    Live Assistant: Learning Whether, When, and Whom to Assist in Real-World Live Social Streams

    [https://arxiv.org/abs/2609.27303](https://arxiv.org/abs/2609.27303)

    该论文提出Live Assistant框架，将真实直播场景中的智能协助形式化为“是否行动、何时行动、面向谁、传达什么”四个耦合决策，由自回归策略每10秒基于音视频、评论、礼物等多模态直播数据自主选择沉默、记录或回复，并构建轨迹引擎将真实直播会话转化为结构化因果监督数据。

    

    直播是一种持久的交互环境，其中视听内容、观众活动、主播行为和平台信号共同演化，产生了源自直播本身的协助需求。我们提出了Live Assistant，一个混合主动、角色条件化的协助框架，将直播交互形式化为四个耦合的决策：是否行动、何时行动、面向谁行动以及传达什么内容。在每个10秒的时间间隔内，一个自回归策略接收原生音频和视频，以及同步的评论、礼物、观众动态和房间元数据，然后选择OBS、MEM或ANS三种动作。OBS保持沉默，MEM记录一条私有的语义更新，ANS则指定接收者、任务和有依据的消息。为了支持这一任务，我们构建了一个轨迹引擎，将真实的直播会话重构为结构化的因果监督数据（摘要在此处被截断）。

    arXiv:2609.27303v1 Announce Type: new  Abstract: Livestreams are long-lasting interactive environments where audiovisual content, viewer activity, host behavior, and platform signals evolve together, creating assistance needs that emerge from the stream itself. We introduce \liveassistant, a framework for mixed-initiative, role-conditioned assistance that formulates livestream interaction as four coupled decisions: \textbf{whether to act, when to act, whom to address, and what to communicate}. At each 10-second interval, one autoregressive policy consumes native audio and video with synchronized comments, gifts, viewer dynamics, and room metadata, then selects \textsc{OBS}, \textsc{MEM}, or \textsc{ANS}. \textsc{OBS} remains silent, \textsc{MEM} records a private semantic update, and \textsc{ANS} specifies a recipient, task, and grounded message. To support this task, we build a trajectory engine that reconstructs real livestream sessions into structured causal supervision, yielding ov
    
[^135]: 超越功效的幻觉：校准观测性信息系统研究中的准实验

    Beyond the Illusion of Power: Calibrating Quasi-Experiments in Observational IS

    [https://arxiv.org/abs/2609.27299](https://arxiv.org/abs/2609.27299)

    该论文通过大规模蒙特卡洛模拟（9837个参数条件、约980万个数据集）分解了观测性IS研究中准实验设计计划功效与实际功效之间的差距，发现序列相关可由AR(1)感知的计算器部分校正，但面板流失、错位采用偏差和平行趋势预检验无法用闭式公式刻画，仅外生流失就会使功效降低约8至11个百分点。

    

    信息系统（IS）研究者越来越多地使用双重差分（DiD）和工具变量（IV）等准实验方法，从观测面板数据中恢复因果效应。用于论证这些设计的功效计算通常假设误差独立同分布（i.i.d.），但更深层的问题在于，即使是考虑了聚类稳健性的计算器也无法察觉的因素。我们报告了一项涵盖9837个参数条件（约980万个数据集）的蒙特卡洛研究，并分解了计划功效与实际功效之间的差距。其中序列相关成分在ρ已知时可由具备AR(1)感知的计算器恢复，在ρ必须从较短的预处理期估计时也可部分恢复；但面板流失、错位采用偏差以及平行趋势预检验无法被任何闭式公式所刻画；在IS研究常用的数百至一千的样本规模下，仅外生流失一项就会造成约8至11个百分点的功效损失。与处理相关、依赖结果变量的流失……

    arXiv:2609.27299v1 Announce Type: cross  Abstract: Information systems (IS) researchers increasingly use quasi-experimental methods such as difference-in-differences (DiD) and instrumental variables (IV) to recover causal effects from observational panel data. Power calculations that justify these designs assume i.i.d. errors, but the deeper problem is what even a cluster-robust calculator cannot see. We report a Monte Carlo study over 9837 parameter conditions (approx 9.8 million datasets) and decompose the planned-versus-achieved power gap. The serial-correlation component is recoverable by an AR(1)-aware calculator when rho is known, and partially when rho must be estimated from short pre-periods, but panel attrition, staggered-adoption bias, and parallel-trends pretesting are captured by no closed-form formula; exogenous attrition alone costs approx 8 to 11 percentage points at the few-hundred-to-thousand sample sizes IS studies use. Treatment-correlated, outcome-dependent attritio
    
[^136]: KITE：面向高效智能体大语言模型扩展的KV不变Transformer扩展方法

    KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling

    [https://arxiv.org/abs/2609.27294](https://arxiv.org/abs/2609.27294)

    KITE提出了一种新的模型扩展范式，通过将新增参数放置在不影响注意力KV缓存的区域，使模型在从小到大扩展时既节省训练成本（通过升级复用），又节省推理成本（KV预填充只需依赖较小的模型部分）。

    

    扩展语言模型不仅仅关乎最终质量：架构选择决定了在训练、提示处理和自回归解码过程中，为达到特定模型质量所需花费的计算量。理想的模型架构应当降低上述所有计算成本，以便于扩展到更大的模型，同时确保更大的模型确实优于较小的基线模型。我们提出了KV不变Transformer扩展，这是一种能够实现该目标的扩展范式。它将模型从小尺寸训练到更大尺寸（即通过升级复用来节省训练成本），同时将新增加的参数放置在不影响注意力KV的区域。因此，在推理过程中，KV的预填充仅依赖于模型的较小部分，从而节省了推理成本。作为一个具体的实例化方案，我们提出了步进缩放Transformer（SST），这是一种双塔解码器，其中一个塔产生KV，另一个塔……

    arXiv:2609.27294v1 Announce Type: cross  Abstract: Scaling a language model is not only a question of final quality: the architectural choice determines how much computation is spent during training, prompt processing, and autoregressive decoding to achieve certain model quality. An ideal model architecture should lower all above computation costs to facilitate scaling to a larger model, while ensure the larger model indeed outperforms smaller baselines. We introduce KV-Invariant Transformer Expansion (KITE), a scaling paradigm that achieves this goal. It trains the model from a smaller size to a larger size (i.e., saving training costs via upcycling), while places newly added parameters in regions that do not affect attention KV. Consequently, during inference, prefilling KV only relies on the smaller part of the model, so the inference costs are saved. As a concrete instantiation, we present Step Scale Transformer (SST), a two-tower decoder in which one tower produces KV and the othe
    
[^137]: NGN：将神经网络规模作为可微分计数进行学习

    NGN: Learning Neural Network Size as a Differentiable Count

    [https://arxiv.org/abs/2609.27291](https://arxiv.org/abs/2609.27291)

    提出了神经发生网络（NGN），通过可学习的边界以可微分方式让模型在训练中自动学习所需的结构组件数量，训练后仅需部署所学前缀即可，性能几乎不受影响。

    

    神经网络的规模通常在训练之前就已确定，这使得架构选择与权重优化相互分离。我们提出了神经发生网络（Neurogenesis Network, NGN），这是一种可微分的参数化方法，用于学习模型应该使用多少个有序的结构组件。对于每个有序的组件组，一个可学习的边界会在模型参数训练的同时选择出活跃的前缀部分。该边界可以从紧凑的初始化开始增长，部署时只需丢弃超出所学边界的组件即可。受控实验考察了所学边界的收敛性、部署后前缀的性能表现，以及与固定规模模型和其他学习容量方法之间的比较。随后我们将同一机制应用于多层感知机（MLP）、卷积网络与图网络、Transformer、状态空间模型、LoRA 以及适配器（adapters）。在这些场景中，仅部署所学前缀通常对性能影响很小，而所选的弧（原文摘要在此处截断）

    arXiv:2609.27291v1 Announce Type: new  Abstract: Neural network size is usually chosen before training, separating architecture selection from weight optimization. We introduce the Neurogenesis Network (NGN), a differentiable parameterization for learning how many ordered structural components a model should use. For each ordered component group, one learnable boundary selects an active prefix while the model parameters are trained. The boundary can grow from a compact initialization and can be deployed by discarding components beyond the learned boundary. Controlled experiments examine convergence of the learned boundary, the performance of deployed prefixes, and comparisons with fixed-size models and alternative approaches to learning capacity. We then apply the same mechanism to MLPs, convolutional and graph networks, Transformers, state-space models, LoRA, and adapters. Across these settings, deploying only the learned prefix usually changes performance little, and the selected arc
    
[^138]: SR-Fraud：一种面向非平稳支付欺诈检测的结果监督反思式LLM智能体框架

    SR-Fraud: An Outcome-Supervised Reflective LLM Agent Framework for Non-Stationary Payment Fraud Detection

    [https://arxiv.org/abs/2609.27287](https://arxiv.org/abs/2609.27287)

    SR-Fraud提出了一种结果监督的反思式LLM智能体框架，通过将冻结无状态的实时交易评分智能体与离线反思适应机制解耦，并借助确定性验证器仅将经过验证的边界假设纳入可执行知识状态，从而有效应对非平稳支付欺诈检测中攻击者快速适应与爆发式攻击的挑战。

    

    实时支付欺诈检测是一个非平稳的流式预测问题：攻击者会在监督标签成熟之前进行适应演化，而局部爆发式攻击可能在模型重新训练之前就造成损失。生产系统通常依赖表格分类器和规则，这些方法在周期性重训练发生之前难以捕捉这些新出现的序列模式。我们提出了SR-Fraud，这是一个结果监督的反思式LLM框架，它将请求时的决策与离线适应解耦。一个冻结的、无状态的智能体基于混合情景窗口对每笔交易进行评分，以追踪行为变化；同时，一个离线反思智能体从已成熟的错误中提出边界假设。随后，一个确定性验证器仅允许得到支持的假设进入可执行的知识状态。在生产支付欺诈基准上，SR-Fraud在所有检测指标上均优于其冻结的决策智能体，相比静态方法和部分（基线）获得了更高的点估计值。

    arXiv:2609.27287v1 Announce Type: new  Abstract: Real-time payment fraud detection is a non-stationary streaming prediction problem: adversaries adapt before supervised labels mature, and localized burst attacks can cause losses before retraining. Production systems typically rely on tabular classifiers and rules, which can struggle to capture these emerging sequential patterns before periodic retraining occurs. We present SR-Fraud, an outcome-supervised reflective LLM framework that decouples request-time decisions from offline adaptation. A frozen, stateless agent scores each transaction from a Hybrid Episodic Window to track behavioral shifts, while an offline reflection agent proposes boundary hypotheses from matured errors. A deterministic verifier then admits only supported hypotheses into an executable knowledge state. On a production payment-fraud benchmark, SR-Fraud improves all detection metrics over its frozen decision agent, obtains higher point estimates than static and pe
    
[^139]: 具有成对融合的多任务回归

    Multitask Regression with Pairwise Fusion

    [https://arxiv.org/abs/2609.27280](https://arxiv.org/abs/2609.27280)

    该论文提出一种通过对跨任务所有成对系数差异进行惩罚来估计多任务回归系数矩阵的方法，能够灵活刻画不同预测变量上任务间系数的共享与差异结构，并在活跃预测变量数和异常系数数这两个结构量上实现了匹配的上下界。

    

    我们研究了系数共享情况可因预测变量而异的多任务回归问题。对于某个给定的预测变量，许多任务可能具有相同的系数，而少数任务有所不同，且对于另一个预测变量，例外的任务不必相同。我们用两个量来刻画这种结构：活跃预测变量的数量，以及与其对应预测变量最常见取值不同的任务系数总数。我们通过惩罚跨任务的所有成对系数差异来估计系数矩阵，在需要进行预测变量选择时，还会附加一个组惩罚。所得到的上界和下界对这两个量具有相同的依赖关系。我们还考虑了更强的设定，即一大组任务共享同一个完整的系数向量。在明确的样本量条件下，同一个成对估计器可以将这组任务完全合并，同时允许其余任务有所不同。模拟实验和家庭能源数据的分析验证了该方法的有效性。

    arXiv:2609.27280v1 Announce Type: cross  Abstract: We study multitask regression when coefficient sharing can differ by predictor. For a given predictor, many tasks may have the same coefficient while a few differ, and the exceptional tasks need not be the same for another predictor. We describe this structure by two quantities: the number of active predictors and the total number of task coefficients that differ from the most common value for their predictor. We estimate the coefficient matrix by penalizing all pairwise coefficient differences across tasks, with an additional group penalty when predictor selection is needed. The resulting upper and lower bounds have the same dependence on these two quantities. We also consider the stronger setting in which a large set of tasks shares one entire coefficient vector. Under explicit sample-size conditions, the same pairwise estimator pools those tasks exactly, while allowing the remaining tasks to differ. Simulations and household energy 
    
[^140]: 面向稀缺数据的谱连通性先验图学习

    Graph Learning with Spectral Connectivity Priors for Scarce Data

    [https://arxiv.org/abs/2609.27278](https://arxiv.org/abs/2609.27278)

    提出 SCoGL 框架，通过在组合拉普拉斯约束的 GLASSO 目标中加入基于拉普拉斯特征值的谱连通性先验，在稀缺数据条件下显式促进图的全局连通性，从而改善图恢复效果并提升图信号去噪等下游任务性能。

    

    从稀缺数据中学习稀疏图在实践中十分重要但极具挑战性。受扩展图类结构所展现的局部稀疏性与强全局连通性这一理想组合的启发，我们提出了谱连通性正则化图学习（SCoGL），这是一个融入一族拉普拉斯谱先验以显式促进全局连通性的框架。具体而言，SCoGL 在作用于目标邻接矩阵 $\mathbf{W}$ 的组合拉普拉斯约束图套索（GLASSO）目标函数基础上，增加了一个由拉普拉斯特征值计算得到的通用连通性先验。我们推导了若干代表性连通性先验的梯度，并开发了一种带 Armijo 回溯的投影梯度下降（PGD）算法来高效优化 $\mathbf{W}$。实验表明，所提出的 SCoGL 变体在信号观测有限的情况下改善了图恢复效果，并提升了诸如图信号去噪等下游任务的性能。

    arXiv:2609.27278v1 Announce Type: new  Abstract: Learning a sparse graph from scarce data is practically important but challenging. Motivated by the desirable combination of local sparsity and strong global connectivity exhibited by expander-like graphs, we propose spectral connectivity-regularized graph learning (SCoGL), a framework that incorporates a family of Laplacian spectral priors to explicitly promote global connectivity. Specifically, SCoGL augments a combinatorial-Laplacian-constrained graphical lasso (GLASSO) objective over a target adjacency matrix $\mathbf{W}$ with a general connectivity prior computed from Laplacian eigenvalues. We derive gradients for several representative connectivity priors and develop a projected gradient descent (PGD) algorithm with Armijo backtracking to efficiently optimize $\mathbf{W}$. Experiments show that the proposed SCoGL variants improve graph recovery and enhance downstream tasks such as graph signal denoising when signal observations are
    
[^141]: TimeEvo：时间序列智能体的故障驱动自进化

    TimeEvo: Failure-Driven Self-Evolution of a Time Series Agent

    [https://arxiv.org/abs/2609.27277](https://arxiv.org/abs/2609.27277)

    提出TimeEvo框架，通过将智能体的失败诊断聚类为能力缺口、为每个缺口规划测量、合成证据工具并通过配对准入门控筛选，实现了时间序列智能体工具库的故障驱动自进化，解决了人-智体工具错配和静默损害两大问题。

    

    时间序列智能体通过调用外部工具来回答分析性问题，而智能体携带哪些工具是由人类在智能体运行之前预先决定的。然而，我们发现了这种设置中的两个失败模式。人-智体工具错配：一个由21个专家精心策划的工具库在某些任务上有所帮助，却在另一些任务上造成损害，在我们测试的所有骨干模型上都降低了异常检测准确率。静默损害：一轮通用的自我修订会改变147个答案并破坏其中56个，而最终分数的变化却不到1分。两者都源于同一个缺口：工具是否有帮助是在运行时逐个问题决定的，而工具却是预先提供的，且仅用一个平均值来评判。为解决此问题，我们提出TimeEvo，它将智能体诊断出的失败聚类为能力缺口，为每个缺口规划一个测量方法，合成仅基于证据的工具来填补这些缺口，并且仅通过配对准入门控来接纳候选工具库。在十个时间序列问答数据集上的实验……

    arXiv:2609.27277v1 Announce Type: new  Abstract: Time series agents answer analytical questions by calling external tools, and which tools they carry is decided by people before the agent runs. However, we identify two failures in this setup. Human-Agent Tool Misalignment: a library of 21 expert-curated tools helps on some tasks and hurts on others, dropping anomaly accuracy under every backbone we test. Silent Harm: one round of generic self-revision changes 147 answers and breaks 56 of them, while the final score moves by less than a point. Both follow from the same gap: whether a tool helps is decided question by question at runtime, while tools are supplied in advance and judged by a single average. To address this, we propose TimeEvo, which clusters an agent's diagnosed failures into capability gaps, plans a measurement for each, synthesizes evidence-only tools that fill them, and admits the candidate library only through a paired admission gate. Experiments on ten time series QA 
    
[^142]: 柏拉图表征假说中究竟是什么在收敛？结构胜于几何

    What Converges in the Platonic Representation Hypothesis? Structure over Geometry

    [https://arxiv.org/abs/2609.27252](https://arxiv.org/abs/2609.27252)

    该研究挑战了柏拉图表征假说的既有解读，通过受控的2×2框架证明模型间真正收敛的是关系结构而非度量几何。

    

    柏拉图表征假说认为，能力日益增强的模型会收敛到共享的表征。近期工作将这一论断缩小为共享的局部邻域关系，并发现若干全局相似性度量中依赖于模型容量的趋势在经过校准后基本消失。我们对这一解释提出挑战，指出先前的局部-全局比较将结构尺度（局部与全局）与比较对象混淆在了一起：比较对象可分为关系结构（由哪些样本相互关联所定义）与度量几何（以距离、相似度或相关性等量化关系为特征）。为了厘清这些因素，我们构建了一个受控的2×2框架，在局部和全局两种尺度上同时评估关系结构与度量几何。我们引入了H₀骨架重叠度作为互k近邻的全局对应物，并配合了匹配的距离感知变体。在视觉……（摘要原文在此处截断）

    arXiv:2609.27252v1 Announce Type: new  Abstract: The Platonic Representation Hypothesis suggests that increasingly capable models converge toward shared representations. Recent work narrows this claim to shared local neighborhood relationships, finding that capacity-dependent trends in several global similarity measures largely disappear after calibration. We challenge this interpretation by showing that prior local-global comparisons confound structural scale (local versus global) with what is compared: relational structure, defined by which samples are related, versus metric geometry, characterized by quantitative relations such as distances, similarities, or correlations. To disentangle these factors, we construct a controlled $2\times2$ framework that evaluates both relational structure and metric geometry at local and global scales. We introduce $H_0$ skeleton overlap as a global counterpart to mutual $k$-nearest neighbors, together with matched distance-aware variants. Across vis
    
[^143]: 将预训练大语言模型改造为高保真连续文本自编码器

    Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders

    [https://arxiv.org/abs/2609.27248](https://arxiv.org/abs/2609.27248)

    本文提出LLMAE方法，通过在预训练语言模型内部引入固定长度潜瓶颈，将其改造为高保真连续文本自编码器，可近乎完美地重建长达1024个token的文本序列。

    

    下一个词预测使自回归语言模型具备了高度流畅的文本生成能力，但它只能通过序列化分解间接地表示全局结构。相比之下，高保真自编码器已成为图像生成领域的标准基础组件，使生成模型能够在连续潜空间上运行；而文本领域则缺乏同样忠实的连续表示。我们提出 LLMAE，一种将预训练的仅解码器（decoder-only）语言模型改造为连续文本自编码器的方法，其核心是在模型内部激活中引入一个中间的固定长度潜瓶颈。该方法以参数高效的 270M Gemma 3 模型实例化，利用结构化注意力掩码、LoRA 适配和 KL 正则化来学习一个自编码接口，从而充分借助原始大语言模型的生成先验。我们训练 LLMAE 重建最长 1024 个 token 的文本序列，在该任务上取得显著提升，实现了近乎完美的重建效果。

    arXiv:2609.27248v1 Announce Type: new  Abstract: Next-token prediction has enabled highly fluent autoregressive language models, but it represents global structure only indirectly through sequential factorization. In contrast, high-fidelity autoencoders have become a standard primitive in image generation, enabling generative models to operate over continuous latent spaces; text lacks a comparably faithful continuous representation. We propose LLMAE, a method for repurposing a pretrained decoder-only language model as a continuous text autoencoder by exposing an intermediate fixed-length latent bottleneck within its internal activations. Instantiated with a parameter-efficient 270M Gemma 3 model, LLMAE uses structured attention masks, LoRA adaptation, and KL regularization to learn an autoencoding interface that leverages the generative prior of the original LLM. We train LLMAE to reconstruct text sequences up to 1024 tokens, significantly improving on this task to achieve near-perfect
    
[^144]: 面向在线自适应的贝叶斯神经网络全协方差平滑

    Full-Covariance Smoothing of Bayesian Neural Networks for Online Adaptation

    [https://arxiv.org/abs/2609.27244](https://arxiv.org/abs/2609.27244)

    提出通过互协方差恒等式实现贝叶斯神经网络中全协方差的前向传播，突破了现有平滑方法仅支持对角协方差的局限，实现了无需梯度迭代的闭式在线自适应学习。

    

    可以将神经网络的各层视为状态空间模型的时间步，从而将贝叶斯训练转化为一个平滑问题：前向传播通过网络传递高斯矩，反向的Rauch–Tung–Striebel平滑过程以闭式形式更新权重后验。这类方法能够以不确定性感知的方式从每次观测中单遍学习，无需基于梯度的迭代或数据重放，因此非常适合在线自适应和数据高效学习。然而，现有的基于平滑的方法仅限于激活值之间的对角协方差，丢弃了神经元之间的相关性。我们通过一个互协方差恒等式克服了这一限制，使全协方差能够通过网络的非线性激活进行传播。我们推导了一种每层只需一步的平滑器，它仅将每层的仿射输出近似为高斯分布，并且既适用于带噪声的确定性系统，也适用于……（摘要原文在此处被截断）

    arXiv:2609.27244v1 Announce Type: new  Abstract: A neural network's layers can be treated as time steps of a state-space model, turning Bayesian training into a smoothing problem: a forward pass propagates Gaussian moments through the network, and a backward Rauch--Tung--Striebel pass updates the weight posteriors in closed form. Such methods learn from each observation in a single pass, in an uncertainty-aware manner, and without gradient-based iterations or replay, which makes them well suited for online adaptation and data-efficient learning. Existing smoothing-based methods, however, are restricted to diagonal covariances across activations, discarding correlations between neurons. We overcome this limitation via a cross-covariance identity that enables full-covariance propagation through a network's nonlinear activations. We derive a one-step-per-layer smoother that approximates as Gaussian only each layer's affine output, and that applies both to deterministic systems with noisy 
    
[^145]: 关于带成员查询的主动学习的样本复杂度

    On the Sample Complexity of Active Learning with Membership Queries

    [https://arxiv.org/abs/2609.27241](https://arxiv.org/abs/2609.27241)

    本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。

    

    本工作重新审视了主动学习中的一个根本问题：合成任意查询的能力究竟有多强大？与基于池的主动学习相比——即学习者只能从给定的未标注数据池中选择查询——我们发现这种看似微小的查询能力变化可能会极大地改变统计学习的难度。特别地，某些在基于池的设置下本质上学习缓慢、其误差随样本数量仅呈多项式衰减的假设类，一旦允许合成查询，便变得可以指数级快速学习。这一显著的差距表明，成员查询的合成引发了一种根本不同的学习模式，这种模式未能被现有主动学习理论充分刻画，需要新的分析工具来表征其复杂度。受这一现象启发，我们提出了若干充分条件，展示了有趣的例子，并提出……

    arXiv:2609.27241v1 Announce Type: cross  Abstract: This work revisits a fundamental question in active learning: how powerful is the ability to synthesize arbitrary queries? Compared to pool-based active learning, where the learner only selects queries from a given unlabeled pool, we find that this seemingly mild change in query ability may dramatically alter the difficulty of statistical learning. In particular, some hypothesis classes that are inherently slow to learn in the pool-based setting, achieving only polynomial error decay in the number of samples, become exponentially learnable once synthesized queries are allowed. This striking gap suggests that membership query synthesis induces a fundamentally different mode of learning, one that is not adequately captured by existing active learning theory and calls for new analytical tools to characterize its complexity. Motivated by this phenomenon, we develop several sufficient conditions, present intriguing examples, and propose a c
    
[^146]: 发现、证伪、修正：在智能体发现的细胞模型中，从源代码到预测贡献审计输入使用声明

    Discover, Falsify, Revise: Auditing Input-Use Claims from Source Code to Predictive Contribution in Agent-Discovered Cell Models

    [https://arxiv.org/abs/2609.27234](https://arxiv.org/abs/2609.27234)

    提出CELLAUDIT审计框架，从源代码可访问性、预测依赖性和预测贡献三个层面检验AI虚拟细胞模型是否真正使用了输入的扰动信息，并揭示一个留出集性能看似良好的智能体发现预测器实际上对化合物替换完全不变。

    

    AI虚拟细胞旨在预测细胞对特定干预的响应，然而仅凭留出集上的预测性能并不能证明模型确实使用了所提供的扰动信息。这一“预测-声明”差距在智能体式模型发现中尤为关键，因为语言模型智能体是基于评分反馈来生成和修改预测器的。我们提出CELLAUDIT，通过三个问题来审计输入使用声明：该输入能否进入所引用的计算过程、拟合后的预测是否依赖于该输入、以及这种依赖是否改善了对观测响应的预测。在一个配对的形态学-转录组学扰动基准（BBBC047）上，智能体选择的预测器在留出集上达到平均全局皮尔逊相关系数（PCC）0.3153，但对化合物替换完全不变；而仅使用对照谱的预测器也达到了0.3142。源代码检查发现化合物查询通路被单例键值注意力阻断，这种不变性……

    arXiv:2609.27234v1 Announce Type: new  Abstract: AI virtual cells aim to predict cellular responses to specified interventions, yet held-out predictive performance alone does not establish use of the supplied perturbation information. This prediction-claim gap matters in agentic model discovery, where language-model agents generate and revise predictors using score-based feedback. We introduce CELLAUDIT, which audits input-use claims by asking whether an input can enter the cited computation, whether fitted predictions depend on it, and whether that dependence improves prediction of observed response. On a paired morphology-transcriptomics perturbation benchmark (BBBC047), an agent-selected predictor attains a mean held-out Global Pearson correlation coefficient (PCC) of 0.3153 but remains invariant to compound replacement; a control-profile-only predictor reaches 0.3142. Source inspection identifies a compound-query pathway blocked by singleton key-value attention, and the invariance 
    
[^147]: fMRI基础模型的缩放研究

    A Scaling Study for fMRI Foundation Models

    [https://arxiv.org/abs/2609.27232](https://arxiv.org/abs/2609.27232)

    本研究通过超过10,000 GPU小时的受控实验首次系统探索了fMRI基础模型的缩放规律，发现数据与模型规模应协同扩展，且在相同计算量下增加预训练数据比增大模型规模能使更多下游任务受益。

    

    缩放定律（scaling laws）已经指导了计算机视觉和自然语言处理领域的大模型开发，但对于功能磁共振成像（fMRI）基础模型而言，数据、模型规模与计算量之间的关系仍不清楚。本文利用来自200多个源数据集的预训练数据以及超过10,000 GPU小时的实验，开展了一项受控实证研究。在保持预训练框架和下游协议不变的前提下，我们改变了预训练数据规模、模型规模和训练时长。研究显示，下游性能通常随计算量的增加而提升，但使用相似计算量的模型之间可能存在显著的性能差异。在更大的模型规模下，额外的预训练数据带来更大的收益，这表明数据和模型规模应当同步扩展。在相同计算量下，增加预训练数据比增加模型规模能使更多任务受益，尽管这一规律在不同任务之间有所差异。随后，我们使用分布内（ID）……

    arXiv:2609.27232v1 Announce Type: new  Abstract: Scaling laws have guided large-model development in computer vision and natural language processing, but the relationships among data, model size, and compute remain unclear for functional magnetic resonance imaging (fMRI) foundation models. Here, we conduct a controlled empirical study using pretraining data from more than 200 source datasets and over 10,000 GPU-hours of experiments. Holding the pretraining framework and downstream protocol fixed, we vary pretraining data size, model size, and training duration. Downstream performance generally improves with compute, yet models using similar compute can perform substantially differently. Additional pretraining data bring larger gains at larger model sizes, suggesting that data and model size should be scaled together. At matched compute, increasing pretraining data benefits more tasks than increasing model size, although the pattern varies across tasks. We then use in-distribution (ID) 
    
[^148]: 面向长期护理机构居住者肺炎检测的生理学信息驱动的数字听诊方法

    Physiologically Informed Digital Auscultation for Pneumonia Detection in Long-term Care Residents

    [https://arxiv.org/abs/2609.27222](https://arxiv.org/abs/2609.27222)

    本研究利用多通道数字听诊录音，以胸片为监督信号训练深度学习模型，实现对长期护理机构老年人肺炎的客观检测，且仅需三个胸中部位听诊通道即可保持高性能并具备可解释性。

    

    肺炎在老年长期护理机构居住者中难以诊断；多重共存疾病和非典型的临床表现使体征模糊不清，因此亟需在操作上高效、客观的检测手段。我们分析了185名日本居住者的多通道数字听诊器录音（73例肺炎患者，112例有症状但无肺炎者），并以放射科医生确认的胸部X光片和临床医生诊断作为监督信号，训练卷积神经网络、多模态融合模型及基于通道的变体，同时结合时域Grad-CAM可解释性分析。模型采用重复的患者层面交叉验证进行评估，结果表明采用X光监督的模型优于采用临床医生监督的模型（F1为0.729、准确率为0.783，对比F1为0.637、准确率为0.711）。此外，一种三通道选择方案保持了模型性能（F1为0.736，准确率为0.803），其中两个胸中部位置表现最佳，且Grad-CAM的注意力与附加音（adventitious sounds）区域重叠。这些发现表明……

    arXiv:2609.27222v1 Announce Type: cross  Abstract: Pneumonia is difficult to diagnose in older long-term care residents; multimorbidity and atypical presentations obscure signs, motivating operationally efficient objective testing. We analyzed multi-channel digital stethoscope recordings from 185 Japanese residents (73 pneumonia, 112 symptomatic without), using radiologist-confirmed chest X-rays and clinician diagnoses as supervisory signals that train convolutional neural networks, multimodal fusion, and channel-based variants with time-domain Grad-CAM interpretability. Models were evaluated with repeated patient-level cross-validation showing models with X-ray supervision outperformed clinician supervision (F1 0.729, accuracy 0.783 vs. F1 0.637, accuracy 0.711). Additionally, a three-channel selection protocol maintained performance (F1 0.736; accuracy 0.803), with two mid-thoracic sites ranking highest and Grad-CAM attention overlapping adventitious sounds. These findings indicate a
    
[^149]: 面向共形椭球的尾部感知几何学习

    Tail-Aware Geometry Learning for Conformal Ellipsoids

    [https://arxiv.org/abs/2609.27221](https://arxiv.org/abs/2609.27221)

    提出了一种尾部感知的共形椭球几何学习框架，通过在估计集上进行CVaR约束下的体积最小化来学习度量矩阵、再在独立校准集上进行标准共形校准，将尾部敏感性与覆盖保证解耦，从而提升多元共形预测集的效率。

    

    本文研究多元共形预测（CP），这是一种具有有限样本覆盖保证的无分布不确定性量化框架。多元预测集的效率在很大程度上取决于由非一致性分数所编码的残差几何，而现有的最小体积方法依赖于分位数阈值，忽略了尾部残差的严重程度，并将几何学习隐式地绑定到覆盖水平上。我们提出了一种面向共形椭球的尾部感知几何学习框架，将几何学习中的尾部敏感性与最终的覆盖保证解耦。通过双划分设计，我们在估计集上通过CVaR约束下的体积最小化来学习度量矩阵，然后在留出的校准集上应用标准的共形校准。所得到的优化问题是凸的，并且可以解释为一种优先考虑高残差样本的有界重加权机制。此外，我们从理论上……

    arXiv:2609.27221v1 Announce Type: new  Abstract: This paper studies multivariate conformal prediction (CP), a distribution-free uncertainty quantification framework with finite-sample coverage guarantees. The efficiency of multivariate prediction sets hinges critically on the residual geometry encoded by the nonconformity score, while existing minimum-volume methods rely on quantile thresholds that ignore tail residual severity and implicitly bind geometry learning to coverage level. We propose a tail-aware geometry learning framework for conformal ellipsoids that decouples tail sensitivity in geometry learning from the final coverage guarantee. Using a two-split design, we learn the metric matrix via volume minimization under a CVaR constraint on an estimation split, then apply standard conformal calibration on a held-out calibration split. The resulting problem is convex and admits a bounded-reweighting interpretation that prioritizes high-residual samples. Moreover, we theoretically
    
[^150]: KATOsuper：基于敏感性一致傅里叶神经算子的代理加速神经拓扑优化

    KATOsuper: Surrogate-accelerated neural topology optimization with sensitivity-consistent Fourier neural operators

    [https://arxiv.org/abs/2609.27216](https://arxiv.org/abs/2609.27216)

    该论文提出KATOsuper框架，利用敏感性一致傅里叶神经算子（SC-FNO）与forward_split架构，通过自动微分保证预测目标与优化梯度的一致性，从而解决神经代理拓扑优化中的不稳定性并实现显著加速。

    

    拓扑优化（TO）由于每次迭代都需要重复进行有限元分析（FEA）评估，计算成本依然很高。尽管基于神经网络的代理模型提供了潜在的加速可能，但现有方法往往存在预测目标与敏感性之间的梯度不一致问题，导致优化不稳定。本工作提出了KATOsuper，这是一个目标无关的框架，将神经重参数化拓扑优化与敏感性一致傅里叶神经算子（SC-FNO）相耦合。该框架采用forward_split架构，通过对预测目标场进行自动微分来导出部署的敏感性，从而保持预测目标与优化所用梯度之间的一致性。案例研究涵盖三个二维基准问题和三个三维结构，涉及柔度最小化或应力最小化。一个物理信（原文摘要在此处截断）。

    arXiv:2609.27216v1 Announce Type: cross  Abstract: Topology optimization (TO) remains computationally intensive due to repeated finite element analysis (FEA) evaluations required at each iteration. While neural network-based surrogates offer potential acceleration, existing approaches often suffer from gradient inconsistency between predicted objectives and sensitivities, leading to optimization instability. This work presents KATOsuper, an objective-agnostic framework that couples neural-reparameterized topology optimization with a Sensitivity-Consistent Fourier Neural Operator (SC-FNO). The framework employs the forward_split architecture, which derives deployed sensitivities via automatic differentiation through the predicted objective field and thereby preserves consistency between the predicted objective and the gradient used for optimization. The case studies include three 2D benchmark problems and three 3D structures considering compliance or stress minimization. A physics-infor
    
[^151]: 基于电阻曲率的可扩展子图采样

    Scalable Subgraph Sampling via Resistance Curvature

    [https://arxiv.org/abs/2609.27209](https://arxiv.org/abs/2609.27209)

    该论文提出ERC-LG，一种结合Johnson-Lindenstrauss投影与多GPU批量共轭梯度求解器的大规模图电阻曲率近似方法，避免了伪逆计算与完整嵌入存储，并利用所得曲率引导节点与边采样以构建GNN训练子图，在七个数据集中的六个上取得最高的节点分类准确率。

    

    子图采样能够降低大规模图神经网络的训练成本，但现有的采样准则可能忽视边在几何结构中的作用。我们提出了一种由电阻曲率引导的采样框架，该框架建立在ERC-LG之上——一种面向大规模图的曲率近似方法。ERC-LG将Johnson-Lindenstrauss投影与正则化的多GPU批量共轭梯度求解器相结合，避免了显式的拉普拉斯伪逆计算和完整的嵌入存储。所得的曲率用于指导节点和边采样概率，以构建GNN训练子图。实验表明，该方法与基于伪逆的曲率在数值上高度一致，且与仅使用共轭梯度法的计算相比运行时间更短。基于ERC-LG的采样变体在下游节点分类任务中，于七个真实世界数据集中的六个上取得了最高的平均准确率。

    arXiv:2609.27209v1 Announce Type: cross  Abstract: Subgraph sampling reduces the training cost of large-scale graph neural networks, but sampling criteria may overlook the geometric roles of edges. We propose a resistance-curvature-guided sampling framework built on ERC-LG, a curvature approximation method for large-scale graphs. ERC-LG combines Johnson-Lindenstrauss projections with regularized multi-GPU batched conjugate gradient solvers, avoiding explicit Laplacian pseudoinverse computation and full embedding storage. The resulting curvature informs node- and edge-sampling probabilities for constructing GNN training subgraphs. Experiments show numerical agreement with pseudoinverse-based curvature and reduced runtime compared with CG-only computation. ERC-LG-based sampling variants achieve the highest mean accuracy on six of seven real-world datasets in downstream node classification.
    
[^152]: 面向成本高效空间转录组学的主动点选择基准测试

    Benchmarking Active Spot Selection for Cost-Efficient Spatial Transcriptomics

    [https://arxiv.org/abs/2609.27208](https://arxiv.org/abs/2609.27208)

    该论文针对空间转录组学建立了主动学习点选择的回顾性基准，在多种预算水平下系统比较了基于不确定性与多样性的选择策略和随机采样的表现，为成本高效的空间转录组学采样提供了实证依据。

    

    空间转录组学（ST）在组织背景下测量基因表达，但密集的捕获网格成本高昂，且可能重复采样形态相似的区域。大多数主动学习策略是为分类标签和独立样本设计的。我们对空间转录组学中主动学习与均匀随机采样进行了回顾性的基于样本池的基准测试，其中表达向量是高维且连续的，候选点具有空间相关性。基于两个完整测量的公开ST队列，我们对候选表达向量进行掩码处理，并模拟多轮选择过程，包括基于不确定性的蒙特卡洛dropout（MC-dropout）和时间输出差异（TOD）方法，以及基于多样性的CoreSet和受TypiClust启发的选择方法。我们在患者级交叉验证下，比较了在折级训练点池5%、10%、30%和50%预算水平下的160个完整配置，并设有单独的全标签参考。在每个预算内，策略……（原文摘要于此处截断）

    arXiv:2609.27208v1 Announce Type: cross  Abstract: Spatial transcriptomics (ST) measures gene expression in tissue context, but dense capture grids can be costly and may repeatedly sample morphologically similar regions. Most active learning strategies were developed for categorical labels and independent samples. We conduct a retrospective pool-based benchmark of active learning versus uniform Random sampling for ST, where expression vectors are high-dimensional and continuous and candidates are spatially correlated. Using two fully profiled public ST cohorts, we mask candidate expression vectors and simulate multi-round selection with uncertainty-based Monte Carlo dropout (MC-dropout) and temporal output discrepancy (TOD), and diversity-based CoreSet and TypiClust-inspired selection. We compare 160 completed configurations at 5%, 10%, 30%, and 50% of the fold-wide training spot pool under patient-level cross-validation, with a separate full-label reference. Within each budget, strate
    
[^153]: 专家建议预测：多专家情形下任意时刻遗憾匹配固定时限最优常数

    Prediction with Expert Advice: Anytime Regret with Many Experts Matches the Fixed-Time Constant

    [https://arxiv.org/abs/2609.27206](https://arxiv.org/abs/2609.27206)

    本文提出一种无需预知时间视界的专家建议预测算法，使任意时刻的累积遗憾达到 $(1+O(\sqrt{\ln\ln n/\ln n}))\sqrt{t\ln n/2}$，消除了此前的 $\sqrt{2}$ 因子差距，从而在多专家情形下将任意时刻遗憾匹配到固定时限的最优常数。

    

    专家建议预测是在线学习中的一个基本问题。当时间视界 $T$ 事先已知时，$n$ 个专家下的极小化极大累积遗憾渐近为 $\sqrt{\frac{T \ln n}{2}}$，这可以通过乘性权重更新算法并以针对 $T$ 调整的学习率来实现，且已知该界是紧的。然而，如果要求遗憾界在每个时刻 $t$ 都同时成立，此前已知的最优保证为 $\sqrt{t \ln n}$——比前者差一个 $\sqrt{2}$ 的因子——而这个 $\sqrt{2}$ 因子是否必要一直悬而未决。本文证明该因子并不必要：我们给出一种无需知道时间视界的算法，其累积遗憾对所有 $t \ge 1$ 同时满足 $R_t \le \bigl(1 + O(\sqrt{\ln \ln n / \ln n})\bigr)\sqrt{t \ln n / 2}$。

    arXiv:2609.27206v1 Announce Type: cross  Abstract: Prediction with expert advice is a fundamental problem in online learning. When the time horizon $T$ is known in advance, the minimax cumulative regret over $n$ experts is asymptotically $\sqrt{\frac{T \ln n}{2}}$. This is achieved by the Multiplicative Weights Update algorithm with a learning rate tuned to $T$, and is known to be tight. If instead the regret bound is required to hold simultaneously at every time $t$, the best known guarantee has been $\sqrt{t \ln n}$---a factor of $\sqrt{2}$ worse---and it has remained unknown whether this factor of $\sqrt{2}$ is necessary. We show that it is not. We give an algorithm, requiring no knowledge of the horizon, whose cumulative regret satisfies $R_t \le \bigl(1 + O(\sqrt{\ln \ln n / \ln n})\bigr)\sqrt{t \ln n / 2}$ simultaneously for every $t \ge 1$.
    
[^154]: 面向物联网安全的可靠联邦式TinyML部署

    Reliable Federated TinyML Deployment for IoT Security

    [https://arxiv.org/abs/2609.27202](https://arxiv.org/abs/2609.27202)

    该论文提出将联邦学习与TinyML模型压缩技术（知识蒸馏、结构化剪枝和量化）相结合，实现资源受限物联网设备上的隐私保护入侵检测，并发现服务器协调的余弦学习率调度对联邦TinyML系统的训练稳定性至关重要。

    

    arXiv:2609.27202v1 公告类型：cross 摘要：随着物联网设备的日益普及，对能够直接在资源受限硬件上运行、且具备隐私保护能力的入侵检测系统的需求不断增长。联邦学习使模型能够在不共享原始数据的前提下进行协同训练，但传统的联邦模型对于微控制器级别的设备而言往往过于庞大且不稳定。TinyML技术能够构建紧凑的神经网络，但其通常仅为仅推理的工作负载而设计。本工作研究了将联邦学习与基于TinyML的模型压缩相结合，用于物联网环境中的入侵检测。我们在联邦训练流程中评估了包括知识蒸馏、结构化剪枝和量化在内的多种压缩策略。初步结果表明，训练稳定性在联邦TinyML系统中起着关键作用。特别是，由服务器协调的余弦学习率调度能够提升攻击检测性能。

    arXiv:2609.27202v1 Announce Type: cross  Abstract: The growing deployment of Internet of Things (IoT) devices has increased the need for privacy-preserving intrusion detection systems that operate directly on resource-constrained hardware. Federated Learning enables collaborative model training without sharing raw data, but conventional federated models are often too large and unstable for deployment on microcontroller-class devices. TinyML techniques enable compact neural networks but are typically designed for inference-only workloads.   This work investigates combining Federated Learning with TinyML-based model compression for intrusion detection in IoT environments. We evaluate compression strategies including knowledge distillation, structured pruning, and quantization within a federated training pipeline. Preliminary results show that training stability plays a critical role in federated TinyML systems. In particular, server-coordinated cosine learning-rate scheduling improves At
    
[^155]: 序列推荐系统中时间归因可解释方法的系统性基准测试

    A Systematic Benchmark of Explainable Methods for Temporal Attribution in Sequential Recommendation Systems

    [https://arxiv.org/abs/2609.27201](https://arxiv.org/abs/2609.27201)

    该论文首次针对序列推荐系统系统性地基准测试了基于梯度、扰动和注意力的可解释方法在时间归因上的忠实性，并提出了双模型掩码评估指标来衡量各方法的效果。

    

    序列推荐系统是现代个性化服务的核心，它利用用户的历史交互序列来驱动下一步决策。深度学习模型，特别是基于CNN和Transformer的架构，已被证明在捕捉这些历史序列中的时间依赖性方面非常有效。为了透明度和信任度，理解哪些过去的交互驱动了某条特定推荐正变得越来越重要——无论是对于审计模型行为的开发者，还是对于寻求推荐理由的用户而言。然而，赋予这些模型强大预测能力的非线性特性同时也使其成为黑盒，导致难以将决策归因于具体的交互。尽管目前存在基于梯度、基于扰动和基于注意力的可解释性方法，但针对这些方法在序列推荐中忠实性的系统性基准测试仍然缺失。我们通过引入一种双模型掩码评估指标来填补这一空白，其中一个模型提供逐个……

    arXiv:2609.27201v1 Announce Type: new  Abstract: Sequential RecSys are central to modern personalization, exploiting user's historical interaction sequences to drive next-step decisions. Deep learning models, particularly CNN and Transformer-based architectures, have proven highly effective at capturing temporal dependencies in these histories. For transparency and trust, understanding which past interactions drive a given recommendation is increasingly important --- both for developers auditing model behavior and for users seeking a rationale. However, the non-linearities that give these models their predictive power also render them black boxes, making it difficult to attribute decisions to specific interactions. While gradient-based, perturbation-based, and attention-based explainability methods exist, a systematic benchmark of their faithfulness for sequential recommendation is missing. We address this gap by introducing a dual-model masking metric in which one model supplies per-t
    
[^156]: ZO-COSMO：面向去中心化零阶优化的无索引单跳混合方法

    ZO-COSMO: Index-Free One-Hop Mixing for Decentralized Zeroth-Order Optimization

    [https://arxiv.org/abs/2609.27199](https://arxiv.org/abs/2609.27199)

    ZO-COSMO 提出了一种无索引的单跳混合方法，通过将双查询估计与保持平均的掩码共识相结合，解决了去中心化零阶优化中稀疏通信的对等状态兼容问题，并在理论上给出紧致收缩界与收敛保证，实验中在相同通信预算下优于显式索引方法。

    

    去中心化零阶学习中的稀疏通信需要兼容的对等状态坐标。我们刻画了这一单跳条件，并提出了 ZO-COSMO 方法，该方法将双查询估计与保持平均的掩码共识相结合，每条活跃链路仅需使用 q 个数值。全局支撑集用于全邻居混合；而匹配更新只需在每对节点内部达成一致。我们在匹配类中推导出每标量紧致收缩界，并为核心更新和稀疏动量更新提供了收敛保证。在固定匹配下，精确的矩恒等式刻画了共享方向如何保持对梯度异质性的消除作用，并重新分配估计误差与节点间分歧。机制实验涵盖了不等曲率、噪声和稀疏动量等场景。进一步的测试涵盖 64 个合成智能体和八个逻辑 Qwen LoRA 工作节点。在相同的负载预算下，Qwen2-7B 在 QNLI 任务上比采用显式索引的 Rand-k 方法提升了 3.65 个准确率百分点。

    arXiv:2609.27199v1 Announce Type: new  Abstract: Sparse communication in decentralized zeroth-order learning requires compatible peer-state coordinates. We characterize this one-hop condition and develop \textsf{ZO-COSMO}, coupling two-query estimation with average-preserving masked consensus using $q$ values per active link. Global supports serve all-neighbor mixing; matching updates require agreement only within each pair. We derive a sharp contraction-per-scalar bound within the matching class and convergence guarantees for the core and sparse-momentum updates. At fixed matching, exact moment identities characterize how shared directions preserve gradient-heterogeneity cancellation and redistribute estimation error and disagreement. Mechanism experiments cover unequal curvatures, noise, and sparse momentum. Further tests span $64$ synthetic agents and eight logical Qwen LoRA workers. At matched payload budgets, Qwen2-7B QNLI gains $3.65$ accuracy points over explicit-index Rand-$k$;
    
[^157]: 基于数据驱动的离散时间深度循环神经网络的耗散系统建模

    Data-driven discrete-time deep recurrent neural network-based modeling for dissipative systems

    [https://arxiv.org/abs/2609.27186](https://arxiv.org/abs/2609.27186)

    本文提出DissipNet，一种通过结构化权重约束和专门训练算法显式保证耗散性的深度离散时间循环神经网络，能够在学习耗散系统动力学的同时确保系统固有的稳定性。

    

    物理人工智能因其能够开发出更好地理解、预测和控制真实世界动态的AI系统而日益受到关注。实现这一目标需要AI模型不仅具备高预测精度，还能保持动力系统的基本物理特性。本文提出了一种深度离散时间耗散循环神经网络，通过结构化权重约束和专门的训练算法显式地强制满足耗散性——这一与稳定性和能量耗散密切相关的关键特性。通过构造方式，所提出的网络能够学习耗散动力学，同时保持其固有的稳定性，并且基于Lyapunov理论对这一性质进行了严格的分析。与物理信息神经网络不同——后者虽然将控制方程纳入训练损失中，但无法保证保持耗散性等内部解析特性——本文所提出的方法在结构上保证了这些物理性质的维持。

    arXiv:2609.27186v1 Announce Type: new  Abstract: Physical AI has gained increasing attention for its role in developing AI systems that better understand, predict, and control real-world dynamics. Achieving this requires AI models that not only achieve high prediction accuracy but also preserve fundamental physical properties of dynamical systems. In this paper, we propose a deep discrete-time dissipative recurrent neural network (DissipNet) that explicitly enforces dissipativity, a key property related to stability and energy dissipation, through structural weight constraints and a dedicated training algorithm. By construction, the proposed network is capable of learning dissipative dynamics while preserving their inherent stability, which is formally analyzed using Lyapunov theory. In contrast to Physics-Informed Neural Networks (PINNs), which incorporate governing equations into the training loss but do not guarantee preservation of internal analytical properties such as dissipativi
    
[^158]: 基于治疗前-后数据的人工智能替代指标用于治疗效应估计

    Artificial intelligence surrogates for treatment effect estimation with before-and-after data

    [https://arxiv.org/abs/2609.27180](https://arxiv.org/abs/2609.27180)

    该论文提出一个新框架，利用预训练AI模型对每位患者治疗前后的测量数据进行结局预测并比较个体内差异，从而以AI预测作为低成本替代指标来估计治疗的因果效应。

    

    当临床上重要的结局指标测量成本高昂或需要长期随访时，估计医学治疗的因果效应十分困难。短期或低成本的替代结局指标提供了一种潜在的替代方案，但替代生物标志物可能不可用或难以识别。人工智能（AI）的进步使得从廉价的高维测量数据中预测临床结局越来越准确，这为将AI预测本身用作替代指标创造了机会。为此，我们开发了一个框架，用于从每位接受治疗个体的治疗前和治疗后配对测量数据中估计治疗效应。该方法将预训练的AI模型应用于治疗前后的测量数据，我们的估计器比较所得的结局预测。我们刻画了这种个体内对比能够识别平均治疗效应所需满足的技术假设条件。

    arXiv:2609.27180v1 Announce Type: cross  Abstract: Estimating the causal effects of medical treatments is difficult when clinically important outcomes are costly to measure or require long follow-up. Short-term or inexpensive surrogate outcomes offer a potential alternative, but surrogate biomarkers may be unavailable or difficult to identify. Advances in artificial intelligence (AI) have enabled increasingly accurate prediction of clinical outcomes from inexpensive, high-dimensional measurements, which creates an opportunity to use AI predictions themselves as surrogates. To this end, we develop a framework for estimating treatment effects from paired measurements obtained before and after treatment for each treated individual. A pretrained AI model is applied to the before and after measurements, and our estimator compares the resulting outcome predictions. We characterize the technical assumptions under which this within-person contrast identifies the average treatment effect on the
    
[^159]: 中位数时间集成：面向动作分块视觉运动策略的无训练鲁棒聚合

    Median Temporal Ensembling: Training-Free Robust Aggregation for Action-Chunked Visuomotor Policies

    [https://arxiv.org/abs/2609.27167](https://arxiv.org/abs/2609.27167)

    该论文提出无需训练的中位数时间集成方法，用坐标式中位数取代指数加权平均来聚合动作分块视觉运动策略的重叠预测，从而在面对不断增强的对抗性攻击时保持鲁棒稳定性，克服了传统时间集成崩溃点为0、以及对抗微调在强攻击下性能急剧退化的问题。

    

    arXiv:2609.27167v1 公告类型：cross 摘要：动作分块的视觉运动策略会预测重叠的轨迹，因此每个执行的动作都被多个预测所覆盖。时间集成通过将这些预测与指数加权平均相结合来平滑执行。然而，一个被破坏的预测可以无界地移动聚合结果：其崩溃点为0。我们使用对抗性破坏来压力测试这种已部署的聚合器，并比较两种类型的保证。度量保证限制了对给定大小扰动的响应；而组合保证则限制当覆盖某个时间步的M个候选中至多q个被破坏时的损害，无论破坏的大小如何。编码器对抗微调在已发表的补丁攻击下可恢复44%的损失，但在攻击者步长增大后仅能恢复7.3%。相比之下，相同候选集的坐标式中位数随着攻击优化的增强，其恢复比例保持稳定不变。（摘要原文在此处截断）

    arXiv:2609.27167v1 Announce Type: cross  Abstract: Action-chunked visuomotor policies predict overlapping trajectories, so every executed action is covered by several predictions. Temporal ensembling smooths execution by combining these predictions with an exponentially weighted mean. One corrupted prediction can move the aggregate without bound: its breakdown point is 0. We use adversarial corruption to stress this deployed aggregator and to compare two kinds of guarantee. A metric guarantee bounds the response to a perturbation of a given size. A combinatorial guarantee instead bounds the damage when at most q of the M candidates covering a timestep are corrupted, whatever their size. Encoder adversarial fine-tuning recovers 44% of the loss under the published patch attack, but only 7.3% after the attacker's step size is increased. By contrast, the coordinate-wise median of the same candidate set keeps its recovered fraction flat as attack optimisation increases. Median temporal ense
    
[^160]: 大型推理模型在推理时能力与效率的扩展规律

    Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models

    [https://arxiv.org/abs/2609.27166](https://arxiv.org/abs/2609.27166)

    本文利用层次贝叶斯模型量化了DeepSeek-R1-Distill系列模型在算术与算法推理任务上的表现，发现正确解题概率随问题难度近似指数衰减，而衰减尺度随模型规模增长，从而揭示了推理时能力与效率随模型规模的扩展规律。

    

    能力与效率是大型语言模型（LLM）推理的两个关键维度。能力指正确解决给定问题的能力，而效率指在有限资源下完成这一任务的能力。当LLM使用思维链推理来求解难度受控的问题时，正确解决的问题数量以及得到正确答案所需的token数量都取决于问题难度和模型规模。然而，这些因素如何共同塑造能力与效率仍知之甚少。本文使用层次贝叶斯模型，评估了DeepSeek-R1-Distill模型家族的LLM在四类算术与算法推理问题上的能力与效率。在固定模型规模下，正确解决一个实例的概率随实例规模（作为问题难度的代理指标）近似呈指数衰减，该衰减尺度随模型规模而增长……

    arXiv:2609.27166v1 Announce Type: new  Abstract: Capability and efficiency are two key dimensions of reasoning in large language models (LLMs). Capability refers to the ability to solve a given problem correctly, whereas efficiency refers to the ability to do so with limited resources. When LLMs use Chain-of-Thought (CoT) reasoning to solve problems of controlled hardness, both the number of problems solved correctly and the number of tokens required to reach a correct answer depend on problem hardness and model size. However, how these factors jointly shape capability and efficiency remains poorly understood. Here, we use hierarchical Bayesian models to evaluate the capability and efficiency of LLMs from the DeepSeek-R1-Distill model family across four classes of arithmetic and algorithmic reasoning problems. At a fixed model size, the probability of correctly solving an instance decays approximately exponentially with instance size, our proxy for problem hardness. The decay scale gro
    
[^161]: 线性表示假说需要一个群作用

    The Linear Representation Hypothesis Needs a Group Action

    [https://arxiv.org/abs/2609.27158](https://arxiv.org/abs/2609.27158)

    论文指出线性表示假说实际上是由表示等价性区分的一族假说，并提出用群作用将其形式化——明确表示对象、生成过程与所断言的性质——从而澄清不同度量、读取点和分析阶段之间假设的差异。

    

    为了做出能够泛化到特定训练模型之外的关于表示的论断，我们需要明确两个表示在何种情况下应被视为等价。线性表示假说通常在被讨论时并未明确说明这种等价性。不同的等价概念保留不同的结构，因此看似在研究同一表示的度量、探测器和干预手段实际上可能对应不同的假说。因此，我们认为线性表示假说并非单一假说，而是一族由表示等价性加以区分的论断。我们使用群作用将这一想法形式化，明确表示对象、生成该表示的过程以及最终所断言的性质，同时考虑由模型架构所施加的等价性。该框架阐明了假设如何在不同的度量、读取点和分析阶段之间发生变化，并且我们利用它来（摘要在此处截断）

    arXiv:2609.27158v1 Announce Type: cross  Abstract: To make claims about representations that generalize beyond a particular trained model, we need to specify when two representations should count as equivalent. The Linear Representation Hypothesis is often discussed without making this equivalence explicit. Different notions of equivalence preserve different structures, so metrics, probes, and interventions that appear to study the same representation may in fact correspond to different hypotheses. We therefore argue that the Linear Representation Hypothesis is not one hypothesis but a family of claims distinguished by representation equivalence. We formalize this idea using group actions, specifying the representation object, the procedure that produces it, and the property ultimately asserted, while accounting for equivalences imposed by the model architecture. This framework clarifies how assumptions can change across metrics, reading points, and analysis stages, and we use it to au
    
[^162]: 功归其位：面向高效推理的冗余感知学习

    Giving Credit Where It's Due: Redundancy-Aware Learning for Efficient Reasoning

    [https://arxiv.org/abs/2609.27156](https://arxiv.org/abs/2609.27156)

    提出RECAP方法，通过在LLM标注的语义依赖图上从最终答案节点反向传播信用，同时衡量步骤的结构责任与对解题的贡献，实现冗余感知的信用分配，从而在不牺牲准确性的前提下有效缩短大型推理模型的推理链。

    

    大型推理模型能够产生正确但不必要冗长的推理轨迹。现有方法通过轨迹级目标或局部的token级与步骤级信号来提升推理效率，但很少对步骤间的语义依赖关系进行建模。这限制了它们区分冗余步骤与支持后续推导的步骤的能力，使得在不牺牲准确性的前提下缩短推理变得困难。我们提出了RECAP（通过传播实现冗余感知的信用分配，REdundancy-aware Credit Assignment via Propagation），该方法通过基于步骤在推理结构中的下游作用及其对正确解决问题的贡献来进行恰当的信用分配，从而解决了这一局限。我们定义了“结构责任”来刻画步骤的下游作用，通过从最终答案节点出发、在与结果无关的由LLM标注的语义依赖图上进行信用反向传播，来衡量后续推理对该步骤的依赖程度。

    arXiv:2609.27156v1 Announce Type: new  Abstract: Large reasoning models can produce correct yet unnecessarily long reasoning traces. Existing methods improve reasoning efficiency with trajectory-level objectives or local token- and step-level signals, but rarely model inter-step semantic dependencies. This limits their ability to distinguish redundant steps from those that support later deductions, making it harder to shorten reasoning without sacrificing accuracy. We introduce RECAP (REdundancy-aware Credit Assignment via Propagation), which addresses this limitation by assigning credit where it is due based on both a step's downstream role in the reasoning structure and its contribution to solving the problem correctly. We define structural responsibility to capture the step's downstream role by measuring how strongly later reasoning depends on it, using credit propagated backward from the final-answer node through an outcome-independent, LLM-annotated semantic dependency graph. Howe
    
[^163]: 点赞陷阱：针对基于相似度的推荐系统中智能体的多阶段投毒攻击

    The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems

    [https://arxiv.org/abs/2609.27155](https://arxiv.org/abs/2609.27155)

    该研究通过理论分析揭示了社交媒体平台推荐系统中的点赞评分机制存在可利用的漏洞，攻击者可通过多阶段投毒帖子链，以隐蔽方式操纵部署在平台上的LLM智能体的信息流。

    

    随着大语言模型（LLMs）及基于LLM的智能体的最新发展，这些智能体正变得日益自主，并能够更广泛地代表用户在互联网上执行操作。然而，部署在社交媒体平台上的自动化智能体（例如用于管理用户个人账户的智能体）的脆弱性仍未得到充分探索。现有关于智能体投毒的研究通常假设攻击者能够将投毒内容直接暴露给智能体。尽管这种攻击方式直接且有效，但更容易被检测和缓解。在社交媒体平台的背景下，这留下了一个悬而未决的问题：推荐系统本身是否会以更隐蔽的方式将此类内容推送给智能体。通过理论分析，我们证明了OASIS系统中使用的点赞评分机制可以被利用，并刻画了多阶段投毒帖子链能够操纵智能体信息流的条件。基于这些……

    arXiv:2609.27155v1 Announce Type: cross  Abstract: With recent advancements in large language models (LLMs) and LLM-based agents, these agents are becoming increasingly autonomous and gaining broader access to act on users' behalf on the internet. However, the vulnerability of automated agents deployed on social media platforms (e.g., for managing a user's personal account) remains underexplored. Existing studies on agent poisoning typically assume that the adversary can expose poisoned content to the agent. Although such an attack is direct and effective, it is more easily detected and mitigated. In the context of social media platforms, this leaves open whether the recommendation system itself would surface such content to the agent in a more subtle manner. Through theoretical analysis, we show that the like-score mechanism used in OASIS can be exploited, and we characterize the conditions under which a multi-stage chain of poisoned posts can steer the agent's feed. Based on these in
    
[^164]: 学习对未观测混杂因素稳健的风险评分

    Learning Risk Scores Robust to Unobserved Confounders

    [https://arxiv.org/abs/2609.27144](https://arxiv.org/abs/2609.27144)

    该论文针对历史分配决策中存在未观测混杂因素的场景，提出了对未观测混杂稳健的风险评分学习方法，以避免系统性地低估最需要稀缺资源的个体。

    

    我们研究的问题是如何从受未观测混杂因素影响的历史观测数据中学习风险评分，以便对个体进行优先级排序，从而分配稀缺资源或实施干预。关于谁应获得稀缺资源的决策，通常由基于已记录特征（例如调查问卷的回答）的风险评分来指导。这些风险评分越来越多地直接从观测数据中学习得到，即利用个体特征、分配决策及结果的历史记录。如果历史决策过程能够完全由已记录的特征所解释，那么诸如逆倾向加权（IPW）这类能够校正历史分配政策所引入偏差的标准方法，便可用于学习准确的风险评分。然而在实践中，历史决策往往依赖于未记录的信息，这会导致学习到的风险评分系统性地低估了恰恰是那些其未记录（特征使其最需要这些资源的）个体（注：原文摘要在此处截断）。

    arXiv:2609.27144v1 Announce Type: new  Abstract: We consider the problem of learning risk scores to prioritize individuals for scarce resources or interventions, from historical observational data affected by unobserved confounding. Decisions about who receives scarce resources are often guided by risk scores based on recorded characteristics, such as responses to a survey. These risk scores are increasingly being learned directly from observational data: historical records of individuals' characteristics, allocation decisions, and outcomes. Standard methods such as inverse propensity weighting (IPW), which corrects for the bias introduced by the historical allocation policy, can be used to learn accurate risk scores if the historical decision process is fully explained by the recorded characteristics. In practice, however, historical decisions often depend on unrecorded information, causing learned risk scores to systematically under-prioritize exactly the individuals whose unrecorded
    
[^165]: PEARL：一种用于实时、匿名和异构协同感知的轻量级基于提示的特征解释器框架

    PEARL: A Lightweight Prompt-based Feature Interpreter Framework for Real-Time, Anonymous, and Heterogeneous Collaborative Perception

    [https://arxiv.org/abs/2609.27123](https://arxiv.org/abs/2609.27123)

    PEARL是一个轻量级基于提示的特征解释框架，通过两个并行训练的轻量级多尺度解释器，实现了无需邻居配置信息、支持实时部署、并能泛化到运行时新加入匿名代理的异构协同感知。

    

    协同感知代理之间的异构性是新兴协同感知框架面临的主要挑战，其根源在于不同传感器、架构和训练数据造成的领域差异。先前的工作通过模型重训练或针对每种代理类型的解释器，将特征对齐到统一空间中以缓解这一挑战。然而这些策略 存在以下问题： 需要访问邻居代理的配置信息， 无法充分满足协同感知的实时部署需求， 对运行时加入的未见过的代理泛化能力较差。为了克服这些挑战，我们提出了PEARL，一个面向匿名、实时、轻量级异构协同感知的提示-嵌入框架。PEARL支持多个协同感知解释器，并利用两个并行训练的轻量级多尺度解释器为新加入的代理实时选择合适的解释器：一个是稀疏检测解释器（LWSD），用于对齐协同检测中的显著区域；另一个是稠密的域不变解释器（LWDDI），用于生成代理…（摘要原文在此处截断）

    arXiv:2609.27123v1 Announce Type: cross  Abstract: Heterogeneity across Collaborative Perception (CP) agents is a major challenge for emerging CP frameworks due to domain gaps from differing sensors, architectures, and training data. Prior works mitigate this challenge by aligning features in a unified space via model retraining or per-agent-type interpreters. These strategies (a) require access to neighbor configurations, (b) do not fully address real-time CP deployment, and (c) generalize poorly to unseen agents joining at run time. To overcome these challenges, we present PEARL, a Prompt-Embedding framework for Anonymous and Real-time Lightweight heterogeneous CP. PEARL supports multiple CP interpreters and selects one for a new-joining agent in real time using two lightweight, multi-scale interpreters trained in parallel: a sparse-detection (LWSD) interpreter that aligns salient regions for cooperative detection, and a dense, domain-invariant (LWDDI) interpreter that produces agent
    
[^166]: 为评审团喂入维度评分，而非整体判定：基于评分准则分解的视觉语言美学评审融合

    Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges

    [https://arxiv.org/abs/2609.27110](https://arxiv.org/abs/2609.27110)

    由多个视觉语言模型组成的评审团在融合整体美学判定时无法显著超越最佳单模型，而让各模型按人工评分准则对图像进行五维度评分并融合这些分数，则能可靠地击败最佳单个模型。

    

    视觉语言模型（VLM）被部署为图像美学的零样本评审器，而在证据薄弱的情况下，多个模型组成的评审团被推荐为提升此类评审可靠性的方法。在两个人工评分数据集EVA和PARA上，我们发现由整体评判模型组成的评审团，无论是对判定结果取平均还是通过学习到的组合器进行融合，都从未显著优于其最佳成员。评审团的价值取决于其被输入的内容。因此，我们让每个模型依据一份固定的人工编写的评分准则（rubric）对每张图像在五个维度上进行评分，并通过折外组合器跨模型族融合这些分数以及每个模型的判定结果。维度评分确实衡量了其标签所声称的内容：在剔除总体人工评分的影响后，在30个模型-属性组合中的28个里，维度提示比整体提示携带更多特定属性的信息。经过融合，它们在EVA数据集上所有十个三模型族评审团中都击败了最佳的单个VLM。

    arXiv:2609.27110v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are deployed as zero-shot judges of image aesthetics, and panels of several models are recommended, on thin evidence, as the way to make such judges reliable. On two human-rated datasets, EVA and PARA, we find that a panel of holistic judges never significantly beats its best member, whether the verdicts are averaged or fused by a learned combiner. What a panel is worth depends on what it is fed. We therefore have each model score each image on the five dimensions of a frozen, human-written rubric and fuse those scores, alongside each model's verdict, across model families with an out-of-fold combiner. The dimension scores measure what their labels claim: with the overall human score partialled out, a dimension prompt carries more attribute-specific information than the holistic prompt in 28 of 30 model-attribute cells. Fused, they beat the best single VLM in all ten three-family panels on EVA (against tha
    
[^167]: 跨具身形态的智能

    Intelligence Across Embodiments

    [https://arxiv.org/abs/2609.27095](https://arxiv.org/abs/2609.27095)

    该论文主张通用具身智能应依赖能够跨具身差异持续积累的学习，提出将具身多样性作为规模化的新维度，并结合广泛的习得先验，以取代依赖人工设计对应关系的短期方案。

    

    机器人具身形态涵盖了智能体与世界进行物理交互所依赖的感知、运动学、动力学、几何结构、执行与控制等特性。这些特性因机器人而异，且会随时间变化。我们认为，通用的具身智能需要能够跨这些差异不断积累的学习。当前主流的通过人工设计对应关系来弥合具身差异的方法能够带来即时的实际收益，但其假设从长远来看限制了迁移的范围。相反，更通用的方法应当发现那些能随着经验增长而支持向更大范围具身形态迁移的表征。我们提出具身多样性是规模化的一条有前景的轴，并指出广泛的习得先验是一个互补要素。我们呼吁开展能更好刻画具身差异与迁移性能的评估。更广泛地说，跨具身学习将学习的实际挑战与……（原文在此处截断）

    arXiv:2609.27095v1 Announce Type: cross  Abstract: Robotic embodiment encompasses the sensing, kinematics, dynamics, geometry, actuation, and control through which an agent physically interacts with the world. These properties vary across robots and change over time. We argue that general embodied intelligence requires learning that accumulates across these differences. Prevailing methods that engineer correspondences to bridge embodiment differences offer immediate practical gains, but their assumptions limit the scope of transfer in the long run. Instead, a more general approach should discover representations that support transfer to a larger range of embodiments as experience grows. We propose embodiment diversity as a promising axis of scaling, and identify broad learned priors as a complementary ingredient. We call for evaluations that better characterize embodiment gaps and transfer performance. More broadly, cross-embodiment learning connects the practical challenge of learning
    
[^168]: 已训练GNN中的局部证据与几何读出修复

    Local Evidence and Geometric Readout Repair in Trained GNNs

    [https://arxiv.org/abs/2609.27092](https://arxiv.org/abs/2609.27092)

    该研究通过精确质量线性规划和两种学习式后验修复方法（消息重加权与集合条件化logit平移），分离并纠正了训练后GNN节点分类错误中混合权重与logit集合定位两种成因，在八个数据集上将平均准确率从62.6%提升至65.3%，且证明logit平移贡献了绝大部分增益。

    

    许多用于节点分类的图神经网络（GNN）将线性分类器应用于局部消息的非负混合。预测错误可能源于糟糕的混合权重，也可能是可达的logit集合相对于分类器的位置不佳。我们通过一个精确质量线性规划和两种学习到的后验修复方法将这两种原因区分开来。每个重加权后的预测都存在一个等价的中心化logit平移，但只有在消息诱导的位移集合中的平移才能通过重加权实现。在八个数据集、八种GNN骨干网络和十种数据划分上，冻结模型的平均准确率为62.6%，重加权后提升至63.8%，而集合条件化平移可达到65.3%。参数量匹配的仅节点平移器达到64.6%，表明平移解释了大部分增益，而消息集合只带来较小的额外收益。尽管oracle重加权能够纠正许多错误，但无标签的重加权几乎无法捕捉这一潜力：局部证据是（摘要在此处被截断）

    arXiv:2609.27092v1 Announce Type: cross  Abstract: Many node-classification GNNs apply a linear classifier to a nonnegative mixture of local messages. An error can reflect either poor mixture weights or a reachable logit set poorly positioned for the classifier. We separate these causes with an exact-mass linear program and two learned post-hoc repairs. Every reweighted prediction has an equivalent centered logit translation, but only translations in a message-induced displacement set are realizable by reweighting. Across eight datasets, eight GNN backbones, and ten splits, mean accuracy rises from 62.6% for the frozen models to 63.8% with reweighting and 65.3% with set-conditioned translation. A parameter-matched node-only translator reaches 64.6%, showing that translation explains most of the gain while the message set supplies a smaller additional benefit. Although oracle reweighting can correct many errors, label-free reweighting captures little of this potential: local evidence is
    
[^169]: Crossflow：面向智能体LLM服务的预填充-解码弹性机制

    Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving

    [https://arxiv.org/abs/2609.27085](https://arxiv.org/abs/2609.27085)

    Crossflow针对P/D分离架构中预填充与解码需求剧烈波动（智能体负载下尤为突出）的问题，提出在不改变节点角色的前提下使预填充-解码边界弹性化，从而避免静态容量规划造成的容量闲置或排队吞吐损失。

    

    随着服务容量需求超过训练需求，服务效率变得越来越重要。预填充-解码（P/D）分离通过两个阶段的专门化与隔离来提升服务效率，但这些收益建立在静态分区的基础上。然而，阶段需求并非静态。我们观察到，在大型LLM集群中，未缓存输入与输出token的比例在分钟时间尺度上的峰值均值比高达4.7倍；而在公开的智能体负载轨迹中，单日内每小时比例的中位数跨度达24.5倍，与此同时重新分配一个副本却需要数十分钟。智能体流量进一步加剧了这种失配：若按第95百分位为各资源池配置容量，将导致多达17%的集群容量闲置；若配置低于该水平，同样的失衡则会转化为排队等待和未能实现的吞吐量。我们提出Crossflow，它能在不改变节点角色的前提下使这一边界变得弹性。每个解码节点发布一个短期、可撤销的租约……

    arXiv:2609.27085v1 Announce Type: cross  Abstract: As serving capacity demand surpasses that of training, serving efficiency becomes increasingly important. Prefill-decode (P/D) disaggregation improves serving efficiency through specialization and isolation of the two phases. These benefits rest on a static partitioning. Phase demand, however, is not static. We observe that in a large LLM fleet the ratio of uncached input to output tokens has peak-to-mean ratios up to 4.7x at minute timescales, and that in a public agentic trace the hourly ratio spans a median 24.5x within a single day, while reassigning a replica takes tens of minutes. Agentic traffic sharpens the mismatch. Sizing each pool at its ninety-fifth percentile leaves up to 17% of cluster capacity unused; sizing below it converts the same imbalance into queueing and unrealized throughput. We present Crossflow, which makes this boundary elastic without changing node roles. Each decode node publishes a short-lived, revocable l
    
[^170]: 量化神秘：基于机器学习方法的印度教与佛教神祇比较研究

    Quantifying the Occult: A Comparative Study of Hindu and Buddhist Deities Using Machine Learning Methods

    [https://arxiv.org/abs/2609.27074](https://arxiv.org/abs/2609.27074)

    本研究提出一种双矩阵计算架构，结合Gower距离矩阵与大语言模型语义嵌入，量化196位印度教与金刚乘佛教神祇的形态与神学差异，算法验证了“图像伪装”现象并计算建模了“Atin效应”。

    

    本研究引入一种双矩阵计算架构，以数学方式量化196位印度教与金刚乘佛教密教神祇在形态学与神学上的差异。物理形态通过一种经新颖“基数加权”算法增强的离散Gower距离矩阵进行评估，而神学功能则通过由大语言模型（LLM）语义扩展生成的稠密向量嵌入进行映射，该嵌入被明确用作合成代理以规避循环推理。多模态拓扑投影为“图像伪装”提供了算法验证，展示了迥异的视觉形式如何在结构上掩盖跨传统共享的功能。此外，我对“Atin效应”进行了计算建模——它既是对序列认知偏差的心理学观察，也是机器学习基准——展示了高基数密教锚点（例如，一个ve……（注：摘要原文在此处截断）

    arXiv:2609.27074v1 Announce Type: cross  Abstract: This study introduces a dual-matrix computational architecture to mathematically quantify the morphological and theological divergence of 196 Hindu and Vajrayana Buddhist esoteric deities. Physical morphology is evaluated via a discrete Gower distance matrix enhanced by a novel "Cardinality Weighting" algorithm, while theological function is mapped via dense vector embeddings generated from Large Language Model (LLM) semantic expansions, explicitly utilized as a synthetic proxy to mitigate circular reasoning. The multi-modal topological projections provide algorithmic validation of "iconographic camouflage", demonstrating how distinct visual forms structurally obscure shared cross-tradition functions. Furthermore, I computationally model the "Atin Effect" - serving simultaneously as a psychological observation of sequential cognitive bias and a machine learning benchmark - demonstrating how high-cardinality esoteric anchors (e.g., a ve
    
[^171]: 图结构在微服务根因分析中是否有其价值？基于RCAEval的对照研究，以及该基准真正在度量什么

    Does Graph Structure Earn Its Place in Microservice Root-Cause Analysis? A Controlled Study on RCAEval, and What the Benchmark Was Really Measuring

    [https://arxiv.org/abs/2609.27069](https://arxiv.org/abs/2609.27069)

    对照实验表明，在RCAEval基准上图神经网络相比传统扁平模型并无可靠优势，且该基准自身存在设计缺陷（故障仅注入五个服务、Avg@5指标过于宽松），使其无法真正衡量图结构在微服务根因分析中的价值。

    

    图神经网络主导了近期关于微服务根因分析的工作，然而最近的研究结果对图结构是否有所贡献提出了质疑。由于这些结果比较的是完整的流水线，因此当扁平模型胜出时，无法判断究竟是结构无用还是结构冗余。我们在RCAEval上进行了它们所暗示的对照比较：三个学习分支使用完全相同的特征、优化器、验证集划分、早停规则和评分头，其中仅有一项差异用于区分图分支。跨两个RCAEval基准、两种拓扑来源和四种实验条件，我们没有发现可靠的图结构特异性效应：在分布内，图模型仅以0.003的Avg@5领先传统扁平模型（p = 0.844，n = 6个不相交的折）。对流水线的审计揭示了两个会影响该基准上任何结果的特性。RCAEval仅向每个系统中的五个服务注入故障，而遥测数据中暴露了12到70个服务，并且其核心指标Avg@5使得一个完全不读取遥测数据的排序器……（摘要原文在此处被截断）

    arXiv:2609.27069v1 Announce Type: cross  Abstract: Graph neural networks dominate recent work on microservice root-cause analysis, yet recent results question whether the graph contributes. Those results compare whole pipelines, so when a flat model wins one cannot tell whether structure is useless or redundant. We run the comparison they imply on RCAEval: three learned arms with identical features, optimiser, validation split, early-stopping rule and scoring head, in which a single term separates the graph arms. Across two RCAEval benchmarks, two topology sources and four regimes we find no reliable graph-specific effect: in-distribution the graph model leads the conventional flat model by 0.003 Avg@5 (p = 0.844, n = 6 disjoint folds). Auditing the pipeline surfaced two benchmark properties that condition any result on it. RCAEval injects faults into only five services per system while exposing 12 to 70 in telemetry, and the headline metric is Avg@5: a ranker reading no telemetry at a
    
[^172]: ChipMEM：面向EDA智能体的以验证为依据的记忆机制

    ChipMEM: Verification-Grounded Memory for EDA Agents

    [https://arxiv.org/abs/2609.27067](https://arxiv.org/abs/2609.27067)

    ChipMEM提出了一个以验证结果为依据的EDA智能体记忆框架，只有在技能通过综合、仿真或形式化验证后才进行存储，并结合贝叶斯统计引导，从而避免模型自我评估偏差，生成可跨任务迁移的可复用知识而非仅针对特定任务的修补。

    

    基于大语言模型（LLM）的智能体使用电子设计自动化（EDA）工具，在综合与验证反馈的指导下生成和修改寄存器传输级（RTL）设计。近期的方法通过从执行轨迹中蒸馏可复用技能，或通过基于EDA工具所得奖励进行训练来从这些反馈中学习。然而，这些方法通常是在产生相关经验的相同任务上进行评估的。在同一任务上反复获取基准测试反馈，可能会鼓励针对特定任务的修改，而非创造可迁移的可复用知识。我们提出了ChipMEM，一个面向EDA智能体的以验证结果为依据的记忆层。它将跨任务的过程性记忆与轨迹内的统计引导相结合。其过程性组件只有在技能通过综合、仿真或形式化验证检查之后才对其进行蒸馏和存储，而不是依赖模型的自我评估。一个贝叶斯组件则对工具调用维护分层Beta估计（摘要原文在此处被截断）。

    arXiv:2609.27067v1 Announce Type: cross  Abstract: Large language model (LLM)-based agents use Electronic Design Automation (EDA) tools to generate and revise register-transfer-level (RTL) designs under synthesis and verification feedback. Recent methods learn from this feedback by distilling reusable skills from execution traces or by training on rewards derived from EDA-tools. Both methods are typically evaluated on the tasks that produced the experience. Repeated access to benchmark feedback on the same task can reward task-specific revision rather than creating reusable knowledge that transfers. We introduce ChipMEM, a verification-grounded memory layer for EDA agents. It combines cross-task procedural memory with within-trajectory statistical guidance. Its procedural component distills and stores a skill only after it passes synthesis, simulation, or formal checks, rather than relying on model self-assessments. A Bayesian component maintains hierarchical Beta estimates over tool-c
    
[^173]: EduBehaviors：面向可审计教育对话编码的基于断言的模式框架

    EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues

    [https://arxiv.org/abs/2609.27043](https://arxiv.org/abs/2609.27043)

    提出了EduBehaviors框架，利用大语言模型测量教育对话中重复出现的可观察行为并据此学习分类器，实现了可解释、可审计的教育对话标注，其性能与直接提示方法相当。

    

    大语言模型使得与目标教学构念相对应的教学标注能够快速部署，为在对话数据集上生成分类提供了自然语言接口。然而，由于大语言模型推理的不透明性，我们无法获得关于模型为何为某句话语选择特定标签的可验证的、机制层面的洞察。我们提出了EduBehaviors框架，这是一种可解释、可扩展的教育数据标注方法，它利用大语言模型测量与多个目标构念相关的重复出现的可观察行为，然后基于这些可观察行为学习该构念的分类器。我们在TalkMoves数据集上对该框架进行了评估，预测教师TalkMoves标签。我们的最佳配置取得了0.673的宏平均F1值和0.688的Cohen's kappa系数，证明其与直接提示方法相比具有竞争力。此外，我们发布了EduBehaviors工具包，包含两个供研究人员使用的工具。

    arXiv:2609.27043v1 Announce Type: new  Abstract: Large language models have allowed the rapid deployment of pedagogical annotations corresponding to constructs of interest, allowing a natural language interface for generating classifications on a conversational dataset. However due to the opaque nature of LLM reasoning, we have no verifiable, mechanistic insight into why a model chose a label for an utterance. We introduce the EduBehaviors framework, an interpretable, scalable approach to annotating educational data that uses LLMs to measure repeated observable behaviors relevant to many constructs of interest and then learns a classifier for the construct based on these observable behaviors. We evaluate the framework on the TalkMoves dataset, predicting the Teacher TalkMoves labels. Our best configuration results in a macro-F1 of 0.673 and 0.688 Cohen's kappa, proving competitive with direct prompting approaches. In addition, we release EduBehaviors Toolkit, two tools allowing researc
    
[^174]: EMA：跨GPU的弹性且性能透明的内存共享系统

    EMA: Elastic and Performance Transparent Memory Across GPUs

    [https://arxiv.org/abs/2609.27040](https://arxiv.org/abs/2609.27040)

    EMA提出了一种服务器内跨GPU的弹性内存共享系统，通过预取技术为借用方隐藏远程访问开销、同时保证出借方的内存可按需回收，使双方性能均不低于静态分区。

    

    多GPU服务器已成为现代数据中心的标准构建单元，通过高带宽互连提供聚合容量。与此同时，诸如大语言模型（LLM）推理等工作负载表现出高度动态的内存需求，这可能导致一个GPU耗尽其本地内存，而其他GPU却处于利用率不足的状态。这种不匹配促使我们提出了跨GPU弹性资源共享的模型。我们提出了EMA，一个内存共享系统，允许服务器内的GPU相互借用和回收内存，形成一个弹性的容量池。EMA为借用方和出借方都确保了性能透明性。对于借用方，预取技术隐藏了远程访问的开销，使应用程序感受到的远程内存和本地内存在性能上难以区分。对于出借方，被借用的资源仍可按需回收，保证性能永远不会低于静态分区的情况。虽然我们的设计聚焦于内存，但……

    arXiv:2609.27040v1 Announce Type: cross  Abstract: Multi-GPU servers have become the standard building block of modern data centers, providing aggregated capacity through high-bandwidth interconnects. At the same time, workloads such as LLM inference exhibit highly dynamic memory demands, which can cause one GPU to exhaust its local memory while others remain underutilized. This mismatch motivates a model of elastic resource sharing across GPUs.   We present EMA, a memory sharing system that allows GPUs within a server to borrow and reclaim memory from each other, forming an elastic pool of capacity. EMA ensures performance transparency for both borrowers and lenders. For borrowers, prefetching hides remote access costs so that applications experience remote and local memory as indistinguishable in performance. For lenders, borrowed resources remain reclaimable on demand, guaranteeing that performance never falls below that of static partitioning. While our design focuses on memory, th
    
[^175]: 基于机器学习的聚合物性质预测开放基准

    An open benchmark for machine learning-based polymer property prediction

    [https://arxiv.org/abs/2609.27036](https://arxiv.org/abs/2609.27036)

    该论文推出了开放基准数据集PolyBench26，包含近25万个涵盖八种物理性质的聚合物数据点，支持四项机器学习评估任务，并发现基于图的模型在聚合物性质预测中表现最佳。

    

    聚合物性质预测领域缺乏开放的、标准化的基准数据集，无法对机器学习方法进行严格的比较，且现有资源仅覆盖聚合物结构中很小的一部分，例如均聚物。我们推出了Polymer Benchmark 2026（PolyBench26），这是一个包含近25万个聚合物性质数据点的开放数据集，涵盖八种物理性质，数据来源包括实验测量、密度泛函理论（DFT）和分子动力学模拟。该基准支持在均聚物以及交替、无规和嵌段共聚物上的四项评估任务：分布内性质预测、数据集规模扩展、重复单元复杂性以及向未知聚合物架构的迁移。我们比较了语言模型、基于图的方法和基于描述符的方法，发现基于图的模型在性质预测中误差最低，在所评估的各种训练集规模下保持其优势，并且……

    arXiv:2609.27036v1 Announce Type: cross  Abstract: Polymer property prediction lacks open, standardized benchmarks that enable rigorous comparison of machine-learning methods, with existing resources covering only a narrow fraction of polymer architectures, such as homopolymers. We introduce Polymer Benchmark 2026 (PolyBench26), an open dataset comprising nearly 250,000 polymer-property datapoints across eight physical properties, including data from experimental measurements, density functional theory, and molecular dynamics. The benchmark supports four evaluation tasks across homopolymers and alternating, random, and block copolymers: in-distribution property prediction, dataset-size scaling, repeat-unit complexity, and transfer to held-out polymer architectures. We compare language model, graph-based, and descriptor-based approaches and find graph-based models provide the lowest errors in property prediction, retain their advantage across the evaluated training-set sizes, and remain
    
[^176]: 基于分解子任务的强化学习

    Reinforcement Learning with Decomposed Subtasks

    [https://arxiv.org/abs/2609.27035](https://arxiv.org/abs/2609.27035)

    该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。

    

    组相对策略优化（GRPO）及用于训练语言模型智能体的相关策略梯度方法，在进入策略更新之前，会将整个多轮rollout压缩为单一标量轨迹奖励。当任务由不同技能组合而成时，尤其是在稀疏且延迟的环境反馈下，这种压缩是有损的：优化器必须隐式地推断是哪种能力导致了最终结果，以及这应当如何改变行为。我们认为正确的基元并非更好的标量，而是分解：轨迹奖励应当在进入策略更新之前沿着子任务进行拆分。我们提出了基于分解子任务的强化学习（RLDS），其核心是子任务分解优势估计（SDAE）：一种替代标量GRPO优势的方法，它在固定的分类体系上将轨迹奖励拆分为每个子任务的份额，为每个子任务计算组相对优势，并将每个token的信用分配……

    arXiv:2609.27035v1 Announce Type: new  Abstract: Group Relative Policy Optimization (GRPO) and related policy-gradient methods for training language model agents collapse an entire multi-turn rollout into a single scalar trajectory reward before it enters the policy update. When the task composes distinct skills, especially under sparse and delayed environmental feedback, this collapsing is lossy: the optimizer must implicitly infer which competency drove the outcome and how that should change behavior. We argue the right primitive is not a better scalar but a decomposition: trajectory reward should be split along subtasks before it enters the policy update. We introduce Reinforcement Learning with Decomposed Subtasks (RLDS), whose core is Subtask-Decomposed Advantage Estimation (SDAE): a replacement for the scalar GRPO advantage that splits trajectory reward into per-subtask shares on a fixed taxonomy, computes a group-relative advantage per subtask, and distributes per-token credit b
    
[^177]: CVaR锚点回归保护对抗罕见偏移

    CVaR anchor regression protects against rare shifts

    [https://arxiv.org/abs/2609.27034](https://arxiv.org/abs/2609.27034)

    CVaR锚点回归通过用尾部平均值代替跨环境平均残差平方的均值，能够针对罕见的大偏移提供精确的最坏情况风险保证，同时避免了传统方法因过度扩大保护范围而降低常见环境预测准确性的问题。

    

    我们研究了当训练数据包含罕见的大偏移时，在新环境中的预测问题。锚点回归对跨环境的平均残差平方的均值进行惩罚，它能保护对抗由训练偏移的二阶矩所决定的椭球内的偏移。因此，覆盖罕见偏移可能需要较大的惩罚，这会在所有方向上扩大椭球，从而降低模型在常见环境上的准确性。我们提出了CVaR锚点回归，它用尾部平均值代替平均残差平方的均值。与直接应用于预测风险的CVaR或GroupDRO不同，该方法不会仅仅因为某个环境的噪声水平高就赋予其更大的权重。我们在允许异方差噪声的线性结构模型下证明了精确的最坏情况风险保证。对于离散环境，减小CVaR尾部比例会将鲁棒性集合从椭球扩展为训练偏移的缩放凸包。

    arXiv:2609.27034v1 Announce Type: cross  Abstract: We study prediction in new environments when training data contain rare, large shifts. Anchor regression penalizes the average of the squared mean residual across environments. It protects against shifts in an ellipsoid determined by the second moment of the training shifts. Covering rare shifts may therefore require a large penalty, expanding the ellipsoid in every direction and reducing accuracy on common environments. We propose CVaR anchor regression, which replaces the average of the squared mean residuals with a tail average. Unlike CVaR or GroupDRO applied directly to prediction risks, it does not give environments more weight solely because their noise levels are high. We prove an exact worst-case risk guarantee under a linear structural model that allows for heteroscedastic noise. For discrete environments, decreasing the CVaR tail fraction expands the robustness set from an ellipsoid to a scaled convex hull of the training sh
    
[^178]: WTF?! 基于Wasserstein倾斜流映射的无模拟强化学习

    WTF?! Simulation-Free Reinforcement Learning with Wasserstein-Tilted Flow Maps

    [https://arxiv.org/abs/2609.27033](https://arxiv.org/abs/2609.27033)

    提出WTF框架，通过基于预训练漂移构建的Wasserstein最优传输正则化器，将奖励微调问题等价转化为流上的确定性最优控制问题，实现了无需模拟的强化学习微调，是首个原生于流映射的端到端微调方案。

    

    奖励微调旨在更新预训练的基于流的生成模型，以提升其生成样本的下游奖励。现有方法通常将该问题表述为从奖励倾斜分布中采样，即KL正则化奖励最大化问题的解。本文引入了一种直接基于预训练漂移构建的最优传输正则化器。与KL奖励倾斜不同，所得目标是使个体样本向更高奖励方向传输，而非对基础分布进行重新加权。我们证明了该问题等价于流上的一个确定性最优控制问题。给定预训练的流映射，这种等价性催生了一种用于微调生成流的无模拟强化学习算法。我们将所得框架称为Wasserstein倾斜流映射，这是首个原生于流映射的端到端微调方案。其输出是一个微调后的流映射……

    arXiv:2609.27033v1 Announce Type: new  Abstract: Reward fine-tuning aims to update a pre-trained flow-based generative model to improve the downstream reward of its generated samples. Existing methods typically formulate this problem as sampling from a reward-tilted distribution, the solution to a KL-regularized reward-maximization problem. Here, we introduce an optimal transport regularizer built directly from the pre-trained drift. Unlike KL reward tilting, the resulting objective transports individual samples toward higher reward rather than reweighting the base distribution. We show that the resulting problem is equivalent to a deterministic optimal control problem on the flow. Given a pre-trained flow map, this equivalence yields a simulation-free reinforcement learning algorithm for fine-tuning generative flows. We call the resulting framework Wasserstein-Tilted Flow Maps (WTF), the first end-to-end fine-tuning recipe native to flow maps. The output is a fine-tuned flow map that 
    
[^179]: LexLattice：基于文档层次结构上神经元胞自动机的多语言抽取式摘要

    LexLattice: Multilingual Extractive Summarization via Neural Cellular Automata on Document Hierarchies

    [https://arxiv.org/abs/2609.27032](https://arxiv.org/abs/2609.27032)

    LexLattice将法律文档层次结构建模为二维语义格并通过神经元胞自动机整合跨远距离部分的证据，仅用180万参数的整合器就在24种语言上超越了数十亿参数的大模型，实现了最先进的多语言抽取式摘要性能。

    

    忠实性是法律文本摘要中的核心关切，这促使人们采用抽取式方法来选取可溯源至原文的逐字内容。此类方法通常孤立地对段落或其他结构单元进行排序，却很少关注整合那些分布于文档远隔部分且共享显著性的证据。我们提出了LexLattice，这是一种抽取式摘要器，它将法律文本的层次结构具体化为二维语义格，并在其上利用掩码二维神经元胞自动机进行证据整合，然后再执行选择。LexLattice在EUR-Lex-Sum数据集的多语言与跨语言设置中，于全部24种语言上均取得了最先进的ROUGE分数，超越了拥有数十亿参数的指令微调基线模型，尽管其全部可训练容量仅集中于冻结多语言编码器之上一个180万参数的整合器。仅在高资源语言上训练的整合器……

    arXiv:2609.27032v1 Announce Type: new  Abstract: Faithfulness is a central concern in legal text summarization, which motivates extractive approaches that select verbatim content traceable to its source. Such methods typically rank paragraphs or other structural units in isolation, yet give little attention to consolidating evidence that is distributed across, and shares salience between, distant parts of a document. We introduce LexLattice, an extractive summarizer that reifies a legal act's hierarchy as a two-dimensional semantic lattice and consolidates over it with a masked 2D neural cellular automata before selection. LexLattice attains state-of-the-art ROUGE across all 24 languages of EUR-Lex-Sum in both multilingual and cross-lingual settings, surpassing instruction-tuned baselines with billions of parameters, despite concentrating all trainable capacity in a 1.8M parameter consolidator over a frozen multilingual encoder. A consolidator trained only on high-resource languages fu
    
[^180]: GeoRVQ：面向生理信号残差token预测的解码器感知几何

    GeoRVQ: Decoder-aware geometry for residual-token prediction in physiological signals

    [https://arxiv.org/abs/2609.27018](https://arxiv.org/abs/2609.27018)

    GeoRVQ提出了一种解码器感知的从粗到细掩码token建模方法，利用冻结波形解码器的局部响应定义几何感知的软目标与期望失真，显著提升了生理信号残差token预测的准确率、解码质量以及R波检测性能。

    

    残差向量量化（RVQ）将生理波形转化为紧凑的token序列，但传统的掩码建模将每个错误的token视为同等代价。我们提出GeoRVQ，这是一种从粗到细的掩码token模型，其目标函数反映了冻结波形解码器的局部响应。解码器诱导的代价定义了几何感知的软目标和期望失真，而量化器因果预测则遵循从粗到细层级的残差依赖关系。在MIMIC-IV Waveform、VitalDB和CODE-15%数据集的描述性汇总实验中，在相同的模型与训练条件下，GeoRVQ将精确token准确率从.133±.004提升至.143±.003，将解码距离从.606±.006降低至.393±.007，并将R波峰值检测F1分数从.784±.004提升至.837±.008。在45个留出的码字替换实验中，解码器诱导代价与实际解码代价的Spearman相关系数达到.85，而欧氏距离仅为.54。

    arXiv:2609.27018v1 Announce Type: new  Abstract: Residual vector quantization (RVQ) turns physiological waveforms into compact token sequences, but conventional masked modeling treats every incorrect token as equally costly. We propose GeoRVQ, a coarse-to-fine masked token model whose objective reflects the local response of a frozen waveform decoder. Decoder-induced costs define geometry-aware soft targets and expected distortion, while quantizer-causal prediction follows residual dependencies from coarse to fine levels. In a descriptive aggregate over MIMIC-IV Waveform, VitalDB, and CODE-15\%, GeoRVQ increases exact token accuracy from $.133\pm.004$ to $.143\pm.003$, reduces decoded distance from $.606\pm.006$ to $.393\pm.007$, and increases R-peak F1 from $.784\pm.004$ to $.837\pm.008$ under matched model and training conditions. Across 45 held-out code substitutions, decoder-induced cost has a Spearman correlation of $.85$ with realized decoded cost, compared with $.54$ for Euclide
    
[^181]: 谱非负相互作用能量的Wasserstein梯度流的锐收敛

    Sharp Convergence of Wasserstein Gradient Flows for Spectrally Nonnegative Interaction Energies

    [https://arxiv.org/abs/2609.27008](https://arxiv.org/abs/2609.27008)

    本文证明，在闭流形上谱系数非负的相互作用能量，其 Wasserstein 梯度流即使不具测地凸性且不含扩散，能量间隙仍以 $o(t^{-1})$ 的锐速率衰减并时间可积，且当谱系数全为正时流弱收敛到常值测度。

    

    我们在闭流形 $M$ 上研究相互作用能量 $\mathsf E[\mu] = \frac12\iint_{M\times M}K(x,y)\,d\mu(x)\,d\mu(y)$ 的 Wasserstein 梯度流的长时间行为。对于在拉普拉斯特征基下对角化且谱系数非负的核，我们证明了联系相对熵与能量间隙的一个微分不等式。因此，对任意非负初始密度 $u_0\in L^p(M)$（$p>1$），能量间隙在时间上可积且满足 $\mathsf E[\mu_t]-\mathsf E_{\min}=o(t^{-1})$。若所有谱系数均为正，则流弱收敛到常值测度。这些相互作用能量在 Wasserstein 空间中不必是测地凸的，且相应的流不含扩散项；因此它们的整体收敛性无法由标准的 Wasserstein 梯度流理论推出。我们的结果所涵盖的核包括球面上的带状核、由……产生的核（原文摘要在此处截断）。

    arXiv:2609.27008v1 Announce Type: cross  Abstract: We study the long-time behavior of Wasserstein gradient flows for interaction energies \[ \mathsf E[\mu] = \frac12\iint_{M\times M}K(x,y)\,\mathrm d\mu(x)\,\mathrm d\mu(y) \] on a closed manifold $M$. For kernels diagonal in a Laplace eigenbasis with nonnegative spectral coefficients, we prove a differential inequality relating the relative entropy to the energy gap. Consequently, for any nonnegative initial density $u_0\in L^p(M)$, $p>1$, the energy gap is integrable in time and satisfies \[ \mathsf E[\mu_t]-\mathsf E_{\min}=o(t^{-1}). \] If all spectral coefficients are positive, the flow converges weakly to the constant measure. These interaction energies need not be geodesically convex in Wasserstein space, and the associated flows contain no diffusion; their global convergence therefore does not follow from standard Wasserstein gradient flow theory. The kernels covered by our results include zonal kernels on spheres, kernels arisi
    
[^182]: 资源高效的分布式递归高斯过程

    Resource-Efficient Distributed Recursive Gaussian Processes

    [https://arxiv.org/abs/2609.26979](https://arxiv.org/abs/2609.26979)

    本文针对多智能体系统的多输出高斯过程回归，提出了两种资源高效的分布式递归高斯过程算法（ADMM-RGP和PDMM-RGP），并通过稳定性与收敛性分析以及参数选择策略来加速收敛、降低通信负担。

    

    高斯过程（GP）提供了一个灵活的框架，能够从含噪测量中学习未知函数，同时量化预测不确定性，使其非常适合多智能体系统中的估计任务。然而，当测量数据由多个智能体收集时，在不进行集中式处理的情况下维护统一的GP模型，需要能够仅利用局部测量和与相邻智能体的通信进行运行的高效分布式算法。在本工作中，我们针对多输出GP回归开发了两种分布式递归高斯过程（RGP）算法：ADMM-RGP和PDMM-RGP。我们分析了这两种算法的稳定性和收敛性，并开发了参数选择策略以加速收敛，从而减少通信负担。所提出的方法在真实世界的多输出风速数据集上得到了验证，并在具有不同连通性的通信图上考察了它们的收敛行为。数值实验…

    arXiv:2609.26979v1 Announce Type: new  Abstract: Gaussian processes (GPs) provide a flexible framework for learning unknown functions from noisy measurements while quantifying predictive uncertainty, making them well suited for estimation in multi-agent systems. However, when measurements are collected by multiple agents, maintaining a unified GP model without centralized processing requires efficient distributed algorithms that can operate using local measurements and communication with neighboring agents. In this work, we develop two distributed recursive GP (RGP) algorithms for multi-output GP regression: ADMM-RGP and PDMM-RGP. We analyze the stability and convergence of both algorithms and develop parameter selection strategies to accelerate convergence, thus reducing the communication burden. The proposed methods are validated on a real-world multi-output wind dataset, and their convergence behavior is examined across communication graphs with varying connectivity. Numerical exper
    
[^183]: 基于多尺度矩阵权重的在线逆线性优化的紧致遗憾界

    Tight Regret Bound for Online Inverse Linear Optimization via Multiscale Matrix Weights

    [https://arxiv.org/abs/2609.26978](https://arxiv.org/abs/2609.26978)

    该论文提出了一种基于多尺度矩阵乘法权重的随机化算法，实现了在线逆线性优化中O(√d)的期望遗憾界，达到了理论最优水平。

    

    我们研究了具有固定未知线性效用函数的在线逆线性优化问题：在每一轮中，环境给出一个紧致的动作集合，学习者从中推荐一个动作，环境则返回在同一集合上最大化效用的动作。当效用向量和动作位于d维欧几里得单位球内时，我们提出了一个随机化算法，其遗憾值（相对于最优动作的累计效用损失）在期望意义下为O(√d)，且适用于任意时间范围，无需预知时间范围。根据已知的当T≥d时Ω(√d)的下界，该算法对d的依赖性在常数因子内是最优的。我们的算法在按几何间距分布的多项式特征空间上维护矩阵乘法权重，通过求解线性规划来选择推荐分布，并通过比较可用动作与反馈动作来更新评分矩阵。

    arXiv:2609.26978v1 Announce Type: cross  Abstract: We study online inverse linear optimization with a fixed unknown linear utility: in each round, an environment presents a compact action set, the learner recommends an action from it, and the environment returns an action that maximizes the utility over the same set. When the utility vector and the actions lie in the $d$-dimensional Euclidean unit ball, we give a randomized algorithm whose regret---the cumulative utility shortfall relative to optimal actions---is $O(\sqrt d)$ in expectation for every time horizon, without knowledge of the horizon. The dependence on $d$ is optimal up to a constant factor by the known $\Omega(\sqrt d)$ lower bound for horizons $T\ge d$. Our algorithm maintains matrix multiplicative weights on polynomial feature spaces at geometrically spaced scales. It selects a recommendation distribution by solving a linear program and updates its score matrices by comparing the available actions with the feedback acti
    
[^184]: TinyUDE：基于李-泰勒射流匹配的微控制器上无求解器通用微分方程

    TinyUDE: Solver-Free Universal Differential Equations on Microcontrollers via Lie-Taylor Jet Matching

    [https://arxiv.org/abs/2609.26972](https://arxiv.org/abs/2609.26972)

    该论文提出了一种名为李-泰勒射流匹配的无求解器训练框架，通过Savitzky-Golay滤波在线估计状态导数并直接拟合混合向量场，使通用微分方程能够在内存受限的微控制器上以解析梯度进行训练，其噪声自适应机制使精度可与传统基于求解器的方法相当甚至更优。

    

    训练通用微分方程（UDE）传统上依赖于通过数值ODE求解器进行反向传播，其内存占用远远超出边缘微控制器的能力。我们提出李-泰勒射流匹配，这是一种无求解器的训练框架，它将混合向量场直接拟合到观测系统状态的一阶和二阶时间导数上。这些导数（即截断的李-泰勒射流）通过Savitzky-Golay滤波在线估计，无需自动微分软件即可获得完全解析的梯度。我们评估了消除求解器是否会损害精度，对比对象是共享相同动力学、噪声模型、网络架构和指标的传统基线（固定步长RK4积分、多重打靶、精确离散伴随、Adam）。尽管朴素的导数匹配在传感器噪声下会退化，但我们的噪声自适应机制缩小并逆转了这一差距：全速率相移……（摘要在此截断）

    arXiv:2609.26972v1 Announce Type: new  Abstract: Training Universal Differential Equations (UDEs) traditionally relies on backpropagating through numerical ODE solvers, creating memory footprints far exceeding the capabilities of edge microcontrollers. We present Lie-Taylor jet matching, a solver-free training framework that fits a hybrid vector field directly to the first and second time-derivatives of observed system states. These derivatives, the truncated Lie-Taylor jet, are estimated online via Savitzky-Golay filtering, yielding fully analytic gradients without automatic differentiation software. We evaluate whether eliminating the solver compromises accuracy against a conventional baseline (fixed-step RK4 integration, multiple shooting, exact discrete adjoints, Adam) sharing identical dynamics, noise models, network architectures, and metrics. While naive derivative matching degrades under sensor noise, our noise-adaptive mechanisms close and reverse this gap: full-rate phase-shi
    
[^185]: CRISP：面向不平衡表格学习的可扩展重要性分层核心集方法

    CRISP: Scalable Importance-Stratified Coresets for Imbalanced Tabular Learning

    [https://arxiv.org/abs/2609.26962](https://arxiv.org/abs/2609.26962)

    CRISP是一种线性时间的重要性分层核心集方法，通过将负类预算按代理模型得分的分位数层分配并进行逆倾向加权，在削减超过95%多数类样本的情况下仍保留约99.7%的全数据模型精度，大幅降低了不平衡表格数据的训练成本。

    

    arXiv:2609.26962v1 公告类型：新论文。摘要：大规模不平衡表格数据集使得重复训练梯度提升树的成本十分高昂。现有的核心集方法在移除大部分多数类样本时往往会损失准确率。我们提出了CRISP（通过重要性分层剪枝实现核心集缩减），这是一种线性时间方法，它将负类样本预算分配到代理模型得分的各个分位数层中，并通过样本权重来考虑不等的样本纳入概率。在生产环境的欺诈检测数据集上，当负类样本削减95%时，CRISP仅需在2500万行数据中的约170万行上进行训练，即可保留全数据训练时99.7%的平均精度，相当于总训练行数减少了93.2%。在公开的CriteoPrivateAds数据集上，CRISP在从90%到99.4%的每一个测试削减率下都取得了最高的平均平均精度。在Sparkov数据集上，较低削减率下的结果各有胜负，但CRISP在99.2%和99.4%的削减率下取得了最高均值。消融实验表明，预算分配和逆倾向加权是该方法性能的主要来源。

    arXiv:2609.26962v1 Announce Type: new  Abstract: Large imbalanced tabular datasets make repeated gradient-boosted tree training expensive. Existing coreset methods often lose accuracy when most majority examples are removed. We present CRISP (Coreset Reduction via Importance-Stratified Pruning), a linear-time method that allocates a negative-class budget across quantile strata of a proxy-model score. Sample weights account for unequal inclusion probabilities. At 95% negative-class reduction on a production fraud dataset, CRISP trains on approximately 1.70M of 25M rows and retains 99.7% of full-data Average Precision. This is a 93.2% reduction in total training rows. On public CriteoPrivateAds, CRISP has the highest mean Average Precision at each tested rate from 90% to 99.4% majority reduction. Sparkov results are mixed at lower rates, but CRISP has the highest mean at 99.2% and 99.4%. Ablations identify budget allocation and inverse-propensity weighting as the main sources of the prod
    
[^186]: 解耦射频感知频谱图中的几何与速度

    Untangling the Geometry and Speed for RF Sensing Spectrograms

    [https://arxiv.org/abs/2609.26960](https://arxiv.org/abs/2609.26960)

    该论文提出一种物理可解释的射频感知新基础，通过紧凑参数化表示与物理信息自编码器（结合可微分射频前向模型），将多普勒频谱图中的目标速度与感知几何解耦，从而突破传统射频感知在真实无约束场景下的适用性限制。

    

    射频感知的一个根本挑战在于，链路所观测到的多普勒特征将目标的运动与感知几何纠缠在一起，导致其在无约束的真实世界场景中适用性受限。在本文中，我们为物理可解释的射频感知建立了一个新基础，能够将反射体速度与几何解耦，并联合恢复每个主导多普勒脊的速度、几何因子、相对幅度和宽度。更具体地说，我们首先开发了WiFi频谱图的紧凑参数化表示，并通过对一个大型且多样化的人体活动数据集进行系统的计算机视觉分析，确立了其低维结构，从而为学习提供了易于处理的基础。在该表示之上，我们进而设计了一个物理信息自编码器，其结构化的瓶颈层与可微分的射频前向模型共同约束，确保了对反射体速度与几何的物理上有意义的估计。

    arXiv:2609.26960v1 Announce Type: cross  Abstract: A fundamental challenge in RF sensing is that Doppler signatures observed by a link entangle the target's motion with the sensing geometry, resulting in limited applicability to unconstrained real-world settings. In this paper, we establish a new foundation for physically interpretable RF sensing that disentangles reflector speed from geometry, jointly recovering the speed, geometry factor, relative amplitude, and width of each dominant Doppler ridge. More specifically, we first develop a compact parametric representation of WiFi spectrograms and establish its low-dimensional structure through a systematic computer-vision analysis of a large and diverse human-activity dataset, thereby providing a tractable foundation for learning. Building on this representation, we then design a physics-informed autoencoder whose structured bottleneck and differentiable RF forward model enforce physically meaningful estimates of reflector speed and ge
    
[^187]: 面向拉闸限电导致数据稀缺情形下太阳能光伏预测的共形化分位数回归迁移学习方法

    Transfer Learning with Conformalized Quantile Regression for Solar PV Forecasting Under Load-Shedding-Driven Data Scarcity

    [https://arxiv.org/abs/2609.26959](https://arxiv.org/abs/2609.26959)

    该论文提出一种结合共形化分位数回归（CQR）的迁移学习框架，在拉闸限电导致历史数据严重稀缺的地区将光伏预测RMSE最高降低23.7%，并提供覆盖率达94.3%且宽度减少14%的可靠不确定性预测区间。

    

    在受拉闸限电影响的地区，由于可靠的历史观测数据稀缺，太阳能光伏（PV）预测面临巨大挑战。本研究提出了一种结合共形化分位数回归的迁移学习框架，旨在严重数据稀缺的情况下改进光伏发电预测并提供可靠的不确定性估计。该研究使用来自澳大利亚爱丽丝泉的光伏源域数据集预训练一个时序预测模型，随后将其适配到代表不同历史数据可用性水平的孟加拉国模拟光伏数据。实验结果表明，当目标域仅有一个月数据可用时，迁移学习可将RMSE降低高达23.7%；使用三个月数据时可降低13.7%。所提出的迁移学习加CQR框架在使用三个月目标数据的情况下实现了94.3%的经验覆盖率，同时生成的预测区间比不使用迁移学习所获得的预测区间窄14%。

    arXiv:2609.26959v1 Announce Type: new  Abstract: Solar photovoltaic (PV) forecasting in regions affected by load shedding is challenging because reliable historical observations are scarce. This study proposes a transfer learning framework combined with Conformalized Quantile Regression (CQR) to improve PV power forecasting and provide reliable uncertainty estimates under severe data scarcity. A source-domain PV dataset from Alice Springs, Australia, is used to pretrain a temporal forecasting model, which is then adapted to simulated Bangladesh PV data representing different levels of historical availability. Experimental results show that transfer learning reduces RMSE by up to 23.7% when only one month of target-domain data is available and by 13.7% with three months of data. The proposed Transfer Learning plus CQR framework achieves 94.3% empirical coverage with three months of target data while producing prediction intervals that are 14% narrower than those obtained without transfe
    
[^188]: 后处理公平性约束何时有效、何时有害：来自八项跨领域评估的证据

    When Post-Processing Fairness Constraints Help and When They Harm: Evidence from Eight Cross-Domain Evaluations

    [https://arxiv.org/abs/2609.26955](https://arxiv.org/abs/2609.26955)

    该论文提出FAPE四阶段公平性审计框架，通过对八个领域的评估发现，后处理公平性干预（Fairlearn的ThresholdOptimizer）的有效性取决于基线差异大小——在多数高差异场景中能改善公平性，但在低差异场景中可能反而有害。

    

    生产环境中的机器学习公平性审计通常只在部署时进行一次，且仅针对单一领域。这两种做法在实践中都会失效：公平性可能会在重新训练或用户群体变化后发生偏移，而在某一数据集上验证过的干预措施很少会在组织实际部署的异构领域中进行测试。我们提出了FAPE（面向生产环境的公平性审计），这是一个四阶段框架，用于评估单一后处理干预措施——Fairlearn的ThresholdOptimizer——在八项领域评估中的表现：刑事司法、收入预测、法律录取、信贷放贷、农业贷款、多领域基准语料库、医疗健康和教育。每项评估均以人口统计均等性和均等化赔率差异进行评分，并在可计算的情况下加入差别影响比率和准确率成本。干预措施的有效性与基线差异幅度密切相关：在各模型-领域组合中，该约束在14个高差异案例中的9个改善了差异表现，而在……

    arXiv:2609.26955v1 Announce Type: new  Abstract: Fairness audits in production ML typically occur once, at deployment, on a single domain. Both fail in practice: fairness can shift after retraining or a changing user base, and interventions validated on one dataset are rarely tested across the heterogeneous domains an organization deploys. We present FAPE (Fairness Auditing for Production Environments), a four-stage framework evaluating a single post-processing intervention, Fairlearn's ThresholdOptimizer, across eight domain evaluations: criminal justice, income prediction, legal admissions, credit lending, agricultural lending, a multi-domain benchmark corpus, healthcare, and education. Each is scored on demographic parity and equalized odds difference, plus disparate impact ratio and accuracy cost where computable. Intervention effectiveness tracks baseline disparity magnitude: across model-domain pairs the constraint improved disparity in 9 of 14 high-disparity cases and worsened i
    
[^189]: 序贯模型训练中的滚动保形预测

    Rolling Conformal Prediction in Sequential Model Training

    [https://arxiv.org/abs/2609.26951](https://arxiv.org/abs/2609.26951)

    本文提出滚动保形预测，一种无需数据划分的无分布预测推断方法，能够为序贯模型训练过程中的预测提供边际覆盖率保证。

    

    我们提出了滚动保形预测，这是一种面向序贯模型训练场景的无分布预测推断方法。具体而言，给定数据流 $(X_1,Y_1),(X_2,Y_2),\dots$，在每个时刻 $n$，训练得到的模型可能依赖于已观测的历史数据 $\{(X_i,Y_i)\}_{i<n}$。这一设定在现代序贯训练中自然出现，包括对海量数据集的单遍训练，以及在部署过程中对语言模型进行持续微调或测试时自适应。Rolling-CP 首先针对当前预测器对每个新到的观测进行校准，然后将其滚动纳入后续训练中。通过这种方式，我们避免了对数据进行划分的需要。值得注意的是，尽管时刻 $n=1,2,\dots$ 的模型可能具有完全不同的性质和精度水平，但对于可交换数据，仍然可以建立边际覆盖率的保证，并带有常见的普适二倍因子保证（最坏情况……

    arXiv:2609.26951v1 Announce Type: cross  Abstract: We introduce Rolling Conformal Prediction (rolling-CP), a distribution-free predictive inference method for the setting of sequential model training. Specifically, given a data stream $(X_1,Y_1),(X_2,Y_2),\dots$, at each time $n$ the trained model may depend on the observed history $\{(X_i,Y_i)\}_{i<n}$. This setting arises naturally in modern sequential training, including one-pass training over massive datasets and continual fine-tuning or test-time adaptation of language models during deployment.   Rolling-CP first calibrates each incoming observation against the current predictor and then rolls it into future training. In this way, we avoid the need for data splitting. Remarkably, although the models at times $n=1,2,\dots$ may have entirely different properties and accuracy levels, for exchangeable data it is nonetheless possible to establish a guarantee of marginal coverage, with a familiar universal factor-two guarantee (a worst 
    
[^190]: 与感觉对齐的感受野的计算价值取决于神经元的表达能力

    The Computational Value of Sensory-Aligned Receptive Fields Depends on Neuronal Expressivity

    [https://arxiv.org/abs/2609.26940](https://arxiv.org/abs/2609.26940)

    该研究发现，与任务相关感觉坐标对齐的感受野在资源预算匹配条件下比随机感受野能提升网络的分类准确率，且这一计算优势的体现依赖于单个神经元的表达复杂度。

    

    生物感觉神经元具有沿有意义的刺激坐标（如频率、运动方向或视网膜拓扑位置）组织的选择性感受野。这种结构可能源于高效编码以及生物在活动、连接和布线方面的约束，正如针对简单神经元的计算研究在各种感觉模态中所展示的那样。由此引出一个问题：结构化感受野是否能在资源效率本身之外赋予计算优势，以及当单个神经元具有高度表达能力时，这种优势是否依然存在？我们在由表达性泄漏记忆神经元构成的循环网络中研究这一问题，在该网络中可以独立地改变神经元的复杂度和前馈感受野的组织方式。在听觉和基于事件的视觉分类任务中，与任务相关感觉坐标对齐的感受野相对于预算匹配的随机感受野提高了测试准确率。

    arXiv:2609.26940v1 Announce Type: cross  Abstract: Biological sensory neurons have selective receptive fields organized along meaningful stimulus coordinates, such as frequency, motion direction, or retinotopic position. Such structure may arise from efficient coding and biological constraints on activity, connectivity, and wiring, as computational studies of simple neurons have shown across modalities. This raises a question: do structured receptive fields confer a computational advantage beyond resource efficiency itself, and does this advantage persist when individual neurons are highly expressive? We address this question in recurrent networks of Expressive Leaky Memory neurons, where we can independently vary neuronal complexity and the organization of feed-forward receptive fields. Across auditory and event-based visual classification tasks, receptive fields aligned with a task-relevant sensory coordinate improve test accuracy relative to budget-matched random receptive fields. T
    
[^191]: 专家在LLM分歧处显现：在大规模标注的LLM码本修订中利用跨模型分歧精准定位专家投入

    Experts Rise Where LLMs Disagree: Using Cross-Model Disagreement to Target Expert Effort in LLM Codebook Revision for Large-Scale Annotation

    [https://arxiv.org/abs/2609.26926](https://arxiv.org/abs/2609.26926)

    该论文提出利用多个大语言模型之间的分歧来定位最需要专家反馈的案例，并通过对比三种反馈方式发现，让专家对分歧案例进行附带理由的标注能最有效地指导LLM码本修订，使LLM标注准确率（64.9%）甚至超过专家手工修订的码本（57.8%）。

    

    大规模文本标注通过AI标注者遵循的码本，将专家洞见带给数百万份文档。然而，开发一个稳健的码本需要数月时间。大语言模型（LLM）可以加速这一过程：将早期码本应用于数据，找出LLM之间存在强烈分歧的案例，并引导专家针对这些案例提供反馈。我们考察了专家为LLM码本修订提供反馈的三种方式：(i) 编辑由跨LLM分歧驱动的LLM生成的修订（码本验证），(ii) 回答关于LLM分歧的问题（问答），(iii) 对分歧案例进行附带理由的标注（理由标注）。在数千份辅导课程转录文本上的实验表明，理由标注方式获得了最高的LLM标注准确率（相对于专家标注为64.9%），优于专家修订的码本（57.8%），最佳的问答设置表现也优于……

    arXiv:2609.26926v1 Announce Type: cross  Abstract: Large-scale text annotation brings expert insight to millions of documents, often through a codebook that AI annotators follow. Developing a robust codebook, however, takes months. Large language models (LLMs) could speed this process by applying an early codebook to the data, surfacing cases with strong LLM disagreement, and eliciting expert feedback to address them. We examined three ways experts can provide feedback for LLM codebook revision: (i) editing LLM-generated revisions driven by cross-LLM disagreement (Codebook Verifying), (ii) answering questions about LLM disagreements (Question Answering), and (iii) labeling disagreement cases with rationales (Rationale Labeling). Experiments on thousands of tutoring-session transcripts show that Rationale Labeling yielded the highest LLM-labeling accuracy (64.9%) against expert labels, outperforming the expert-revised codebook (57.8%). The best Question Answering setting also outperform
    
[^192]: 多目标强化学习中事后重标注导致的偏好覆盖坍缩研究

    On Preference Coverage Collapse from Hindsight Relabeling in Multi-Objective Reinforcement Learning

    [https://arxiv.org/abs/2609.26918](https://arxiv.org/abs/2609.26918)

    该研究发现，在偏好条件化多目标强化学习中，用智能体实际实现的偏好方向进行事后重标注往往有害——它使36个算法-环境设置中的19个性能下降多达四个标准差，其根源是重复重标注导致的偏好覆盖坍缩，而非重标注噪声。

    

    事后重标注——即追溯性地将一条转移的目标替换为智能体实际取得的结果——是提升强化学习（RL）样本效率的有效工具。对于偏好条件化的多目标强化学习（MORL），一个自然的扩展是用智能体实际实现的偏好方向（而非所要求的偏好方向）来重标注转移。我们证明这种扩展常常是有害的：在连续控制MO-Gymnasium基准套件上，跨越两种评论家网络骨干和两种偏好采样方案的四种偏好条件离线策略算法中，36个“算法-环境”组合设置里有19个性能下降多达四个标准差，仅有一个获得改善，其余则不受影响。这种损害并非由重标注噪声所致：对目标进行去噪几乎无法恢复性能，优先级采样以及任何缓冲区结构上的选择也都无法重现这一现象。相反，重复的重标注会导致偏好覆盖的坍缩。

    arXiv:2609.26918v1 Announce Type: cross  Abstract: Hindsight relabeling which retroactively replacing a transition's goal with the outcome the agent actually achieved is an effective tool for improving sample-efficiency in Reinforcement Learning (RL). A natural extension to preference-conditioned multi-objective RL (MORL) relabels transitions with the preference direction the agent achieved rather than the one asked for. We show that this extension is frequently harmful: across four preference-conditioned off-policy algorithms spanning two critic backbones and two preference-sampling schemes on the continuous-control MO-Gymnasium suite, it degrades 19 of 36 algorithm-environment settings by as much as four standard deviations, improves only one, and leaves the rest unaffected.   The harm is not a symptom of noisy relabels; denoising the target recovers almost nothing, and neither prioritized sampling nor any buffer-structural choice reproduces it. Instead, repeated relabeling collapses
    
[^193]: 小线索，大后果：学习多模态迷因分类中的关键线索

    Small Cues, Big Consequences: Learning Pivotal Cues for Multimodal Meme Classification

    [https://arxiv.org/abs/2609.26907](https://arxiv.org/abs/2609.26907)

    该论文提出了聚焦关键线索的MemeCF基准数据集（含9,895个迷因）和MemePIVOT局部-全局架构，通过非平衡最优传输对齐词语与图像块，并利用证据融合头在不确定性下融合局部与全局信息，从而有效捕捉迷因中有害、仇恨或讽刺含义的决定性线索。

    

    迷因（meme）的有害、仇恨或讽刺含义往往源自细小但决定性的视觉、文本或跨模态线索。现有的多模态分类器在主要依赖全局图文表示时，可能会遗漏这些证据。我们提出了MemeCF，一个聚焦线索的基准数据集，包含9,895个涵盖伤害、仇恨和讽刺三类内容的迷因，并附带标注以指明关键证据的模态及其依据。我们还提出了MemePIVOT，一种用于迷因分类的局部-全局架构。MemePIVOT使用冻结的CLIP特征，通过非平衡最优传输将词语与图像块对齐（同时允许无关证据保持不匹配），并采用证据融合头在不确定性条件下将局部对齐信息与全局迷因上下文相结合。在HarMeme、PrideMM和MemeCF上的实验表明，该方法相比强大的纯文本、纯图像、多模态及视觉-语言基线模型均取得了一致的性能提升。跨数据集与消融实验结果进一步表明……（摘要在此处截断）

    arXiv:2609.26907v1 Announce Type: cross  Abstract: Memes often derive their harmful, hateful, or sarcastic meaning from small but decisive visual, textual, or cross-modal cues. Existing multimodal classifiers can miss such evidence when relying mainly on global image-text representations. We introduce MemeCF, a cue-focused benchmark of 9,895 memes across harm, hate, and sarcasm, with annotations identifying the modality and rationale of the pivotal evidence. We also propose MemePIVOT, a local-global architecture for meme classification. MemePIVOT uses frozen CLIP features, unbalanced optimal transport to align words with image patches while allowing irrelevant evidence to remain unmatched, and an evidential fusion head to combine local grounding with global meme context under uncertainty. Experiments on HarMeme, PrideMM, and MemeCF show consistent gains over strong text-only, image-only, multimodal, and vision-language baselines. Cross-dataset and ablation results further show that exp
    
[^194]: CORE-STACK+：面向深度堆叠泛化的元学习

    CORE-STACK+: Meta-Learning for Deep Stacked Generalization

    [https://arxiv.org/abs/2609.26905](https://arxiv.org/abs/2609.26905)

    提出 CORE-STACK+ 元学习框架，通过包含基于 CKA 的核化冗余过滤器等四个组件的预处理流水线，同时解决深度堆叠泛化中预测空间多重共线性和校准崩塌两大问题。

    

    摘要：堆叠异构视觉骨干网络（CNN、ViT 及混合模型）是提升准确性、校准性与鲁棒性的事实标准做法，然而两个相互耦合的病态问题限制了其收益。预测空间的多重共线性会使元学习器的 Gram 矩阵变得病态，导致权重方差膨胀，并在一个狭窄的流形上产生脆弱的解。校准崩塌则通过朴素的线性堆叠使各组成模型自身的校准误差进一步恶化，因此增加更多模型反而可能损害期望校准误差（ECE）。现有的补救措施——岭回归正则化、贪心选择、模型汤和 SWAG——最多只能解决其中一个问题，没有任何方法能够同时针对异构预测池中的条件数问题和校准问题。我们提出了 CORE-STACK+，一个包含四个组件的预处理流水线：（i）基于核的冗余过滤器，利用中心核对齐（CKA）[23] 移除 Pearson 相关性无法察觉的非线性模型间依赖关系；（ii）一个参数量小于 15K 的……（原文摘要在此处截断）

    arXiv:2609.26905v1 Announce Type: new  Abstract: Stacking heterogeneous vision backbones (CNNs, ViTs, and hybrids) is the de facto recipe for accuracy, calibration, and robustness, yet two coupled pathologies limit its returns. Prediction-space multicollinearity ill-conditions the meta-learner's Gram matrix, inflating weight variance and producing brittle solutions on a thin manifold. Calibration collapse compounds constituent miscalibration through naive linear stacking, so adding more models can hurt expected calibration error (ECE). Existing remedies, ridge regularization, greedy selection, model soups, and SWAG address at most one of these issues, and none jointly target conditioning and calibration in heterogeneous prediction pools. We introduce CORE-STACK+, a preconditioning pipeline with four components: (i) a kernelized redundancy filter that removes non-linear inter-model dependencies invisible to Pearson correlation, using Centered Kernel Alignment (CKA) [23]; (ii) a $<15$K-p
    
[^195]: PR-Smoother：面向数据同化的保留模拟器的非高斯平滑方法

    PR-Smoother: Simulator-Preserving Non-Gaussian Smoothing for Data Assimilation

    [https://arxiv.org/abs/2609.26890](https://arxiv.org/abs/2609.26890)

    PR-Smoother通过在证据下界和变分族中保留预设模拟器，仅学习围绕模拟轨迹的未来条件修正，实现了物理轨迹上的非高斯平滑分布，并支持仅从观测数据联合学习状态、参数与传感器偏差。

    

    许多物理数据同化（DA）工作流需要满足以下要求的平滑方法：能够表示物理状态变量上的非高斯后验分布、能够扩展到高维模拟器、仅从观测窗口进行训练，并与预设模拟器的校准保持兼容。我们提出了PR-Smoother，一种专为此类预设模拟器数据同化场景设计的保留模拟器的摊销式平滑器。其核心设计原则是在证据下界和变分族中都保持预设模拟器的显式性：PR-Smoother并不学习替代动力学或学习式的轨迹先验，而是仅学习围绕预设轨迹展开的未来条件修正。这带来了物理轨迹上的显式非高斯平滑分布，并支持仅从观测数据中联合学习状态、参数和传感器偏差。该变分族在确定性和线性高斯情形下包含精确平滑器（原文在此处截断）。

    arXiv:2609.26890v1 Announce Type: new  Abstract: Many physical data assimilation (DA) workflows require smoothing methods that represent non-Gaussian posteriors over physical state variables, scale to high-dimensional simulators, train from observation windows alone, and remain compatible with calibration of the prescribed simulator. We introduce PR-Smoother, a simulator-preserving amortized smoother designed for this prescribed-simulator DA regime. Its key design principle is to keep the prescribed simulator explicit in both the evidence lower bound and the variational family: rather than learning replacement dynamics or a learned trajectory prior, PR-Smoother learns only future-conditioned corrections around the prescribed rollout. This yields an explicit non-Gaussian smoothing distribution over physical trajectories and supports joint state, parameter, and sensor-bias learning from observations alone. The variational family contains the exact smoother in deterministic and linear-Gau
    
[^196]: 边际正确的工具缓存可能逆转组归一化策略更新

    Marginally Correct Tool Caches Can Reverse Group-Normalized Policy Updates

    [https://arxiv.org/abs/2609.26866](https://arxiv.org/abs/2609.26866)

    论文证明在边际奖励分布完全一致的条件下，组内共享单个随机工具结果缓存仍可能逆转组归一化策略更新的方向，而仅中心化、不做组标准差缩放的方法可保持期望回报的正确方向。

    

    工具结果缓存减少了智能体训练中的重复执行，但同时也耦合了各 rollout 的随机性。我们研究了一个双动作模型，其中独立执行与共享执行均保持每个 rollout 的条件奖励分布不变。尽管存在这种边际上的一致性，每组共享一个随机结果仍可能逆转预期的组归一化策略更新方向。我们推导出一个精确的有限组表达式：相对于常数替代方案，共享更新遵循的是获胜概率减去失败概率，而非期望奖励之差。伯努利特例分析揭示了一个方向错误的区域，以及随组规模增大而不消失的更新方差下限。在该模型中，仅进行中心化而不做组标准差缩放可以保持期望回报的方向，并借助现有的估计器控制加以实现。穷举有限和验证了540种配置和3,240次估计器评估，另有独立的有序序列...（原文截断）

    arXiv:2609.26866v1 Announce Type: new  Abstract: Tool-result caching reduces repeated execution in agent training, but also couples rollout randomness. We study a two-action model in which independent and shared execution preserve every rollout's conditional reward distribution. Despite this marginal agreement, sharing one stochastic result per group can reverse the expected group-normalized policy update. We derive an exact finite-group expression: against a constant alternative, the shared update follows the probability of winning minus the probability of losing, rather than the difference in expected reward. A Bernoulli specialization yields a wrong-direction region and a non-vanishing update-variance floor as group size grows. Centering without group standard-deviation scaling preserves the expected-return direction in this model, using an existing estimator control. Exhaustive finite sums verify 540 configurations and 3,240 estimator evaluations, with a separate ordered-sequence c
    
[^197]: 安全提示：面向用户的实时AI风险感知干预措施

    Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness

    [https://arxiv.org/abs/2609.26865](https://arxiv.org/abs/2609.26865)

    该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。

    

    对话式AI系统可能对其用户构成安全风险，例如幻觉、谄媚、过度自信和拟人化，但这些风险在用户日常使用中难以察觉。我们介绍了Safety Nudges，这是一种基于浏览器的工具，当检测到聊天机器人对话中出现令人担忧的行为时，它会提供轻量级的即时标记。我们通过一项为期两周的实地研究对Safety Nudges进行了评估，该研究涉及45名频繁使用聊天机器人的用户，收集了交互日志、调查问卷以及用户对各条提示的反馈。参与者认为该工具有用、清晰且干扰性最小，几乎所有用户都报告称对潜在AI危害的意识有所提高，尽管我们发现仅凭这种意识提升并不一定能带来可观察到的行为改变。我们的结果表明，面向用户的安全提示可以通过帮助人们在具体情境中批判性地评估AI回应，来补充模型层面的安全防护措施，同时强调了……

    arXiv:2609.26865v1 Announce Type: cross  Abstract: Conversational AI systems can pose safety risks to their users such as hallucination, sycophancy, overconfidence, and anthropomorphism, but these risks are difficult for users to detect during everyday use. We introduce Safety Nudges, a browser-based tool that provides lightweight, in situ flags when concerning behavior is detected in chatbot conversations. We evaluated Safety Nudges in a two-week field study with 45 frequent chatbot users, collecting interaction logs, surveys, and feedback on individual nudges. Participants found the tool useful, clear, and minimally disruptive, with nearly all users reporting an increased awareness of potential AI harms, though we found that this improved awareness alone did not necessarily lead to discernible behavioral changes. Our results suggest that user facing safety nudges can complement model-level safeguards by helping people critically evaluate AI responses in context, while highlighting th
    
[^198]: QUARTET：基于四分支交叉注意力与随机游走轨迹的关系图Transformer增强方法

    QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs

    [https://arxiv.org/abs/2609.26855](https://arxiv.org/abs/2609.26855)

    提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。

    

    arXiv:2609.26855v1 公告类型：交叉 摘要：关系深度学习将多表数据库建模为异构时序图，图Transformer目前在RelBench等基准测试上取得了最先进的性能。然而，当前领先的模型RelGT存在两个关键局限：其随机局部采样器生成的子图连接松散，阻碍了消息传递；其全局注意力模块依赖于单一的、基于种子特征的内存，忽略了更广泛的宏观层面动态。为克服这些局限，我们提出了QUARTET，一种表达能力强的图Transformer架构，它在局部子图上应用完全自注意力，同时通过交叉注意力分支来丰富全局上下文。具体而言，QUARTET采用基于近期截断个性化PageRank（PPR）的因果随机游走（CRW）采样器，以提取紧凑、抗枢纽节点干扰且密集连通的局部子图，且不会产生时序信息泄露。与此同时，四分支交叉注意力（摘要在此处截断）

    arXiv:2609.26855v1 Announce Type: cross  Abstract: Relational Deep Learning (RDL) models multi-table databases as heterogeneous temporal graphs, and graph transformers currently achieve state-of-the-art performance on benchmarks like RelBench. However, the current leading model, RelGT, suffers from two key limitations: its random local sampler yields loosely connected subgraphs that hinder message passing, and its global attention module relies on a single, seed-feature-based memory that ignores broader macro-level dynamics. To overcome these limitations, we introduce QUARTET, an expressive graph transformer architecture that applies full self-attention on local subgraphs while enriching global context through cross-attention branches. Specifically, QUARTET employs a Causal Random Walk (CRW) sampler based on recency-truncated Personalized PageRank (PPR) to extract compact, hub-robust, and densely connected local subgraphs without temporal leakage. Concurrently, a quad-branch cross-atte
    
[^199]: COPE：基于用户嵌入与自我评估的稀疏用户反馈下大语言模型持续个性化

    COPE: Continual Personalization of LLMs under Sparse User Feedback via User Embeddings and Self-Evaluation

    [https://arxiv.org/abs/2609.26853](https://arxiv.org/abs/2609.26853)

    COPE提出了一种在稀疏用户反馈下实现大语言模型持续个性化的优化框架，通过为每个用户分配可学习的个性化嵌入，并在单次更新步骤中协同完成偏好捕获、自我评估校准与个性化响应优化。

    

    尽管大型语言模型（LLM）在各种基准测试中取得了显著成果，但它们与规范价值观的对齐往往导致同质化的响应，无法满足多样化的用户偏好。现有的免训练方法通常通过提示工程占用宝贵的上下文窗口，而基于训练的方法在训练后通常保持静态，无法支持现实场景中所需的持续优化。为应对这些挑战，我们提出了COPE（基于个性化嵌入与自我评估的持续优化），这是一个专为具有稀疏用户反馈的现实交互场景量身定制的新型优化框架。我们的框架为每个用户分配可学习的个性化嵌入，并在单个更新步骤内协同整合偏好捕获、自我评估校准和个性化响应优化。我们方法的一个关键创新在于……

    arXiv:2609.26853v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) have achieved remarkable results across various benchmarks, their alignment with normative values often results in homogenized responses that fail to address diverse user preferences. Existing training-free methods often occupy valuable context windows through prompt engineering, while training-based methods typically remain static post-training, failing to support the continual optimization required in real-world settings. To address these challenges, we propose COPE (Continual Optimization with Personalized embedding and self-Evaluation), a novel optimization framework tailored for real-world-motivated interaction settings with sparse user feedback. Our framework assigns learnable personalized embeddings to each user and synergistically integrates preference capture, self-evaluation calibration, and personalized response optimization within a single update step. A key innovation of our method is the
    
[^200]: 一种面向术中早期急性肾损伤预测的防泄漏多模态评估框架

    A Leakage-Aware Multimodal Evaluation Framework for Early Intraoperative Acute Kidney Injury Prediction

    [https://arxiv.org/abs/2609.26848](https://arxiv.org/abs/2609.26848)

    该论文提出了仅基于生理波形的混合时序骨干网络SynerT及其多模态扩展SynerT-MM和防泄漏堆叠集成SynerTStack，并在VitalDB数据库上以严格的防泄漏评估框架实现了术中早期急性肾损伤风险预测。

    

    大型非心脏手术后的术后急性肾损伤（AKI）具有相当高的发病率，然而术中早期风险分层仍然十分困难。在这项回顾性队列研究中，我们提出了SynerT——一种仅使用生理波形的混合时序骨干网络，它将因果扩张时序卷积网络（TCN）与多层扩张循环层相结合，用于编码术中早期生理轨迹以进行AKI风险预测。在SynerT的基础上，我们进一步设计了两个结合结构化临床信息的模型变体来扩展该骨干网络：SynerT-MM是一种后期融合的多模态扩展模型，整合了血流动力学负荷摘要信息与术前协变量；SynerTStack则是一种防泄漏的堆叠集成模型，在元学习阶段将SynerT-MM的交叉验证预测与强大的表格数据基线模型相结合。所有模型均在VitalDB（一个高保真围术期数据库）上，于严格的防泄漏评估框架下进行评估与验证。

    arXiv:2609.26848v1 Announce Type: cross  Abstract: Postoperative acute kidney injury (AKI) after major non-cardiac surgery carries substantial morbidity, yet early intraoperative risk stratification remains difficult. In this retrospective cohort study, we propose SynerT, a waveform-only hybrid temporal backbone that combines a causal dilated TCN with a hierarchy of dilated recurrent layers to encode early intraoperative physiologic trajectories for AKI risk prediction. Building on SynerT, we further design two model variants that extend the backbone with structured clinical context: SynerT-MM, a late-fusion multimodal extension that integrates hemodynamic burden summaries and preoperative covariates, and SynerTStack, a leakage-safe stacked ensemble that combines cross-validated predictions from SynerT-MM with strong tabular baselines at the meta-learning stage. All models are evaluated under a strict leakage-aware framework on VitalDB, a high-fidelity perioperative database, with pred
    
[^201]: NeuroRule：通过规则集演化使黑盒神经网络具备可解释性

    NeuroRule: Making Black-Box Neural Networks Explainable through Rule-set Evolution

    [https://arxiv.org/abs/2609.26841](https://arxiv.org/abs/2609.26841)

    本文提出NeuroRule知识蒸馏框架，通过规则集演化方法将黑盒神经网络蒸馏为简洁、可解释的命题逻辑规则集，从而缓解性能与可解释性之间的权衡。

    

    高容量神经网络模型在各类分类任务中取得了最先进的性能，但它们通常以黑盒模型的方式运行，缺乏关键决策所需的透明度。这种不透明性导致性能与可解释性之间存在持续的权衡。本文提出了一种弥合这一差距的解决方案：NeuroRule知识蒸馏框架，它能够从神经网络模型中产生可解释的规则集。NeuroRule对EVOTER规则集演化基础设施进行了改造，将神经网络作为演化过程的目标，把神经网络的性能蒸馏成简洁的命题逻辑表达式集合。本文有三个主要贡献：(1) 一种将黑盒神经网络模型蒸馏为显式规则集模型的演化方法；(2) 一种通过在演化中加入简洁性目标来提升规则集可解释性的方法；(3) 一个演示……

    arXiv:2609.26841v1 Announce Type: cross  Abstract: High-capacity neural network models have achieved state-of-the-art performance across diverse classification tasks, yet they frequently operate as black-box models, lacking the transparency necessary for critical decision-making. Such opacity creates a persistent trade-off between performance and explainability. This paper proposes a solution to address this gap: the NeuroRule knowledge distillation framework that results in explainable rule-sets from neural network models. NeuroRule adapts the EVOTER rule-set evolution infrastructure to treat neural networks as targets for the evolution process, distilling their performance into concise sets of propositional logic expressions. There are three primary contributions: (1) an evolutionary method for distilling black-box neural network models into explicit rule-set models; (2) a method for making rule sets more explainable by including a conciseness objective to evolution; and (3) a demons
    
[^202]: LWCal：针对含噪声校准标签的表格分类器的损失加权校准方法

    LWCal: Loss-Weighted Calibration for Tabular Classifiers with Noisy Calibration Labels

    [https://arxiv.org/abs/2609.26839](https://arxiv.org/abs/2609.26839)

    提出LWCal，一种无需干净验证标签、无需噪声率估计、也无需重训练的事后校准方法，通过对与基础模型预测相矛盾的噪声标签样本降权，有效应对校准标签含噪声的场景。

    

    事后概率校准通常是在一个乐观假设下进行评估的：即留出的校准标签是干净的。然而，在许多AI部署场景中，标签来自弱标注者、历史决策、启发式规则或远程监督，因此破坏训练的同一标签噪声也会破坏校准。我们针对表格分类器研究了这种被忽视的失效模式，并提出了LWCal——一种仅需CPU的事后校准器，它会对那些噪声标签与基础模型留出概率相矛盾的校准样本进行降权。LWCal不需要干净的验证标签，不需要噪声率估计，也不需要重新训练基础分类器。第二种变体Gated-LWCal增加了一个保守的分歧门控机制，当校准集显得极不一致时，会退回到原始分数。在九个本地二分类表格任务、六个随机种子、对称与非对称标签损坏以及三种树（模型）的实验中……

    arXiv:2609.26839v1 Announce Type: cross  Abstract: Post-hoc probability calibration is usually evaluated under an optimistic assumption: the held-out calibration labels are clean. In many AI deployment settings, however, labels come from weak annotators, historical decisions, heuristics, or distant supervision, so the same label noise that corrupts training also corrupts calibration. We study this overlooked failure mode for tabular classifiers and propose LWCal, a CPU-only post-hoc calibrator that down-weights calibration examples whose noisy labels are contradicted by the base model's held-out probability. LWCal requires no clean validation labels, no noise-rate estimate, and no retraining of the base classifier. A second variant, Gated-LWCal, adds a conservative disagreement gate that backs off toward the raw score when the calibration split appears extremely inconsistent. On nine local binary tabular tasks, six random seeds, symmetric and asymmetric label corruption, and three tree
    
[^203]: 什么使 Terminal-Bench 任务变得困难？——在经裁定的智能体语料库上区分真实困难与虚假困难

    What Makes a Terminal-Bench Task Hard? Separating Genuine Hardness from Fake-Hardness on an Adjudicated Agentic Corpus

    [https://arxiv.org/abs/2609.26826](https://arxiv.org/abs/2609.26826)

    本文提出一套有序的有效性筛选方法，综合任务工件、参考解运行、空解对照、对抗试验与遥测等多源证据，从 Terminal-Bench 的 125 个全失败任务中区分真实困难与虚假困难，发现其中仅 78 个可被认证为真正未解决的任务。

    

    前沿基准测试需要当前模型无法解决的任务，但没有任何模型能解决的任务并不自动就是困难任务。同样的零通过率可能源于真实的能力差距，但也可能源于上下文缺失、参考解决方案损坏、基础设施故障，或可被绕过的验证器。本文利用一份冻结的 Terminal-Bench 3 / Frontier-Bench 0.1 生产记录来研究这一问题，该记录包含 1,081 个拉取请求、639 个已评分任务、28,801 次试验以及 105,933 美元的已记录智能体支出。我们追问：一个全失败的任务究竟能证明什么。针对 125 个没有任何诚实通过的任务，我们综合任务工件、参考解决方案运行结果、空解决方案对照、对抗性试验、轨迹、遥测数据和评审记录，并应用一套有序的有效性筛选流程。结果显示，125 个任务中仅有 78 个被保留为“经认证未解决”的候选任务，其余任务则包括 14 个预言机损坏的任务、8 个被基础设施故障主导的任务等。

    arXiv:2609.26826v1 Announce Type: cross  Abstract: Frontier benchmarks need tasks that current models cannot solve. But a task that no model solves is not automatically a hard task. The same zero pass rate can come from a real capability gap, but it can also come from missing context, a broken reference solution, infrastructure failure, or a verifier that can be bypassed. In this paper, we study this issue using a frozen Terminal-Bench 3 / Frontier-Bench 0.1 production record with 1,081 pull requests, 639 scored tasks, 28,801 trials, and $105,933 in logged agent spend. We ask what an all-fail task actually certifies. For the 125 tasks with no honest pass, we combine task artifacts, reference-solution runs, empty-solution controls, adversarial trials, trajectories, telemetry, and review records, and apply an ordered validity screen. Only 78 of the 125 tasks survive as certified-unsolved candidates. The remaining tasks include 14 with broken oracles, 8 dominated by infrastructure failure
    
[^204]: HARN：用于事件驱动多时间框架预测的分层联想共振网络

    HARN: Hierarchical Associative Resonance Network for Event-Driven Multi-Timeframe Forecasting

    [https://arxiv.org/abs/2609.26822](https://arxiv.org/abs/2609.26822)

    本文提出HARN分层联想共振网络，通过维护跨时间层级的持久表示并在时间条形完成时进行事件驱动的局部更新，结合因果多尺度编码、门控联想记忆与跨层级共振机制，在金融多时间框架预测中取得与强基线相当的性能。

    

    金融时间序列在多个时间分辨率上演变，这对预测系统提出了挑战：如何在不重复计算未发生变化表示的情况下纳入新可获得的信息。我们提出了HARN，一种用于事件驱动多时间框架预测的分层联想共振网络。HARN在各时间层级间维护持久的表示，并且仅在相应的时间条形（bar）完成时才更新对应层级。该架构结合了因果多尺度时间编码、门控联想记忆、跨层级共振以及分层证据聚合，预测在基点空间中进行，然后重构回原始价格尺度。我们在涵盖股票、外汇和商品市场的四种资产上，通过多个随机种子和组件消融实验对HARN进行评估。HARN相对于单时间框架的PatchTST和（摘要在此处被截断）

    arXiv:2609.26822v1 Announce Type: new  Abstract: Financial time series evolve across multiple temporal resolutions, challenging forecasting systems to incorporate newly available information without repeatedly recomputing unchanged representations. We introduce HARN, a Hierarchical Associative Resonance Network for event-driven multi-timeframe forecasting. HARN maintains persistent representations across temporal levels and updates each level only when its corresponding completed bar becomes available. The architecture combines causal multi-scale temporal encoding, gated associative memory, cross-level resonance, and hierarchical evidence aggregation, with forecasting performed in basis-point space and reconstructed to the original price scale. We evaluate HARN on four assets spanning equity, foreign exchange, and commodity markets using multiple random seeds and component ablations. HARN achieves competitive reconstructed-price forecasting errors against single-timeframe PatchTST and 
    
[^205]: Signal2Symbol：面向可解释生理时间序列异常检测的神经符号时间推理

    Signal2Symbol: Neuro-Symbolic Temporal Reasoning for Explainable Physiological Time-Series Anomaly Detection

    [https://arxiv.org/abs/2609.26820](https://arxiv.org/abs/2609.26820)

    提出了一种名为Signal2Symbol的神经符号框架，通过将ECG/EEG信号转换为符号序列并利用稀有项集挖掘对异常进行评分，实现了对生理时间序列的可解释异常检测，并能揭示局部异常之间的时间关联与重复模式。

    

    诸如心电图（ECG）和脑电图（EEG）等生理时间序列表现出复杂的时间结构、显著的采集变异性，以及对透明决策的强烈需求。尽管深度模型能够达到较高的检测性能，但它们在解释某个片段为何异常、局部异常如何随时间相互关联、以及检测结果是否属于更广泛重复模式等方面，通常提供的洞察有限。我们提出了Signal2Symbol，一个用于可解释生物信号异常检测的神经符号框架。该方法首先使用学习得到的VQ-VAE（向量量化变分自编码器）码本或SAX（符号聚合近似）基线方法，将ECG/EEG信号转换为符号序列。然后，它构建了二元组增强的标记窗口事务，并通过源自最小稀有项集挖掘的稀有项集证据对异常进行评分。检测到的异常窗口被合并为区间……

    arXiv:2609.26820v1 Announce Type: cross  Abstract: Physiological time series such as electrocardiograms (ECG) and electroencephalograms (EEG) exhibit complex temporal structure, substantial acquisition variability, and a strong need for transparent decision-making. Although deep models can achieve high detection performance, they often provide limited insight into why a segment is anomalous, how local anomalies relate over time, and whether a detection belongs to a broader recurring pattern. We propose Signal2Symbol, a neuro-symbolic framework for explainable biosignal anomaly detection. The method first converts ECG/EEG signals into symbolic sequences using either a learned VQ-VAE (Vector Quantized Variational Autoencoder) codebook or a SAX (Symbolic Aggregate approXimation) baseline. It then constructs bigram enriched token-window transactions and scores anomalies through rare itemset evidence derived from minimal rare itemset mining. Detected anomalous windows are merged into interv
    
[^206]: 漂移契约：面向深度鲁棒局部学习的谱更新

    The Drift Contract: Spectral Updates for Depth-Robust Local Learning

    [https://arxiv.org/abs/2609.26811](https://arxiv.org/abs/2609.26811)

    该论文首次将Muon风格的谱更新（动量正交化加谱步长缩放）应用于逐层局部学习，仅用单一超参数设置即可在宽度128到2048、深度12到48范围内保持稳健性能并在深度48时不崩溃，显著优于需要反复调参的局部Adam。

    

    局部学习使用各自的辅助损失训练每一层，且不依赖全局反向传播，这使得各层的更新在结构上天然并行。两个问题一直使其处于边缘地位：随着深度增加精度会下降，且超参数十分脆弱。我们将Muon风格的谱更新几何方法（动量正交化结合谱步长缩放）应用于逐层局部更新，这是一个此前未被研究过的交叉领域。在带有局部线性头的CIFAR-10 MLP基准测试中，单一的步长设置在我们测试的宽度128到2048、深度12到48的全部网格中均为最优值；而局部Adam则需要在宽度和深度两个维度上重新调参，且在深度48时仍然崩溃（逐深度重新调参为31.3%，直接迁移其深度12设置为19%，相比之下谱更新在设置完全不变的情况下为42.7%）。在五个随机种子和宽度512的条件下，谱更新以明显优势领先局部Adam（48.9 ± 0.5 对比 46.6 ± 0.3）。

    arXiv:2609.26811v1 Announce Type: new  Abstract: Local learning trains each layer with its own auxiliary loss and no global backward pass, which makes layer updates structurally parallel. Two problems have kept it marginal: accuracy degrades as depth grows, and hyperparameters are fragile. We apply Muon-style spectral update geometry (momentum orthogonalization with spectral step scaling) to per-layer local updates, an intersection not previously studied. On CIFAR-10 MLP benchmarks with local linear heads, a single step-size setting is the best value in our tested grids from width 128 to 2048 and from depth 12 to 48, while local Adam requires re-tuning along both axes and still collapses at depth 48 (31.3 percent re-tuned per depth, 19 percent with its depth-12 setting transferred, vs 42.7 percent for the spectral update at its unchanged setting). At five seeds and width 512 the spectral update leads local Adam by a clear margin (48.9 +/- 0.5 vs 46.6 +/- 0.3). Prospectively specified c
    
[^207]: SpeakerMem-R1：面向多方对话的以说话人为中心的双轨记忆

    SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue

    [https://arxiv.org/abs/2609.26780](https://arxiv.org/abs/2609.26780)

    提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。

    

    多方场景下的长期对话记忆不仅仅是从长期对话中检索相关内容：它必须区分谁说了什么、每句话涉及谁、个体之间如何看待彼此、哪些信息为群体所共享，以及状态如何随时间变化。最近针对多方对话基准的研究表明，现有的通用大语言模型记忆系统往往丢失人物与群体关系，或难以整合分布在成员、群体和时间中的线索。这些问题共同揭示了两个核心瓶颈：多方对话中的消息归属与关系理解，以及从交错历史中进行的状态重建。为解决这两个问题，我们提出了 SpeakerMem-R1：其双轨记忆存储带有说话人标签的逐字消息以及衍生状态，并将它们组织为个人层面和群体层面的视图，随后按实体、事件等方式结合两条轨道的证据（摘要在此处被截断）。

    arXiv:2609.26780v1 Announce Type: new  Abstract: Long-term conversational memory in multi-party settings requires more than retrieving relevant content from long-term conversations: it must distinguish who said what, whom each statement concerns, how individuals perceive one another, what information is shared by the group, and how states change over time. Recent studies on multi-party dialogue benchmarks show that existing general-purpose LLM memory systems tend to lose person and group relations or struggle to integrate clues distributed across members, groups, and time. Together, these issues reveal two core bottlenecks: message attribution and relational understanding in multi-party dialogue, and state reconstruction from interleaved histories. To address both, we propose $\textbf{SpeakerMem-R1}$: its dual-track memory stores speaker-labeled verbatim messages and derived states organized into person-level and group-level views, then combines evidence from both tracks by entity, eve
    
[^208]: 关于稀疏高斯过程回归中基函数选择的研究

    On Basis Function Selection for Sparse Gaussian Process Regression

    [https://arxiv.org/abs/2609.26624](https://arxiv.org/abs/2609.26624)

    本文从信息论视角提出三种基函数选择准则，用于在稀疏高斯过程回归中依据数据挑选最相关的基函数，以替代传统的固定截断策略，从而更高效地利用有限的计算预算。

    

    稀疏高斯过程通过在输入空间上用固定基函数集 {φ_j} 的适当展开来替代核函数，从而实现 O(N) 的推断。在给定计算预算 M ≪ N 的情况下，从业者通常习惯性地将基截断为前 M 个基函数。然而，从形式上看，并没有任何限制阻止人们只选择那些对当前数据真正重要的 M 个基函数。这样做可以避免将计算预算浪费在没有信号的基函数上，但这需要一个能够对候选基函数进行排序的准则。我们从基函数选择问题的信息论视角出发，提出了三种这样的准则。每种准则分别对应于选择时所处的不同知识状态：无数据状态、无先验状态以及介于两者之间的状态。随后，我们在六个 UCI 回归基准数据集上，针对三种基函数族（包括希尔伯特空间高斯过程 HSGP 等），研究了截断策略与选择策略的性能表现。

    arXiv:2609.26624v1 Announce Type: cross  Abstract: Sparse Gaussian processes achieve $O(N)$ inference by replacing the kernel with an appropriate expansion in a fixed basis $\{\phi_j\}$ on the input space. Given a compute budget $M \ll N$, practitioners conventionally truncate the basis to its first $M$ entries. Nothing in the formalism, however, prevents one from selecting only those $M$ basis functions that matter for the data at hand. This would avoid spending budget on basis functions where there is no signal, but it requires a criterion for ranking the candidates. We propose three such criteria derived from an information-theoretic view of the basis-function selection problem. Each criterion matches a different state of knowledge at selection time: a no-data state, a no-prior state, and an in-between state. We then study the performance of truncation versus selection strategies on six UCI regression benchmarks across three basis families: Hilbert-space Gaussian processes (HSGP), v
    
[^209]: GTR：用于高效密集预测的门控令牌循环

    GTR: Gated Token Recurrence for Efficient Dense Prediction

    [https://arxiv.org/abs/2609.26590](https://arxiv.org/abs/2609.26590)

    GTR是一种无softmax的循环视觉骨干网络，通过门控线性注意力、交替空间扫描和空间增强SwiGLU模块，并从DINOv3教师模型进行最终层蒸馏，以低延迟在COCO检测上达到58.9 AP，并可广泛迁移至多种密集预测任务。

    

    基于自注意力的视觉骨干网络在密集预测任务上表现优异，但全局softmax注意力的二次方计算成本限制了其在图像分辨率提升时的效率。我们提出了门控令牌循环（GTR），这是一种无softmax的循环视觉骨干网络，它结合了门控线性注意力、交替空间扫描方向以及空间增强的SwiGLU模块。GTR通过仅使用最终层patch令牌对齐（经由线性投影和平方ℓ2损失）从检测专用的DINOv3教师模型进行蒸馏，无需掩码令牌预测或中间层监督。经过Objects365检测器预训练后，GTR-L在COCO val2017上达到58.9框AP，在RTX 4090上以编译FP16执行时的批大小为1的中位延迟仅为1.908毫秒。同一骨干网络还可迁移至实例分割、姿态估计、旋转目标检测、语义分割和单目深度估计等任务。

    arXiv:2609.26590v1 Announce Type: cross  Abstract: Self-attention-based vision backbones perform well on dense prediction, but the quadratic computational cost of global softmax attention limits their efficiency as image resolution increases. We introduce Gated Token Recurrence (GTR), a softmax-free recurrent vision backbone that combines gated linear attention, alternating spatial scan directions, and spatially enhanced SwiGLU blocks. GTR is distilled from a detection-specialized DINOv3 teacher using only final-layer patch-token alignment through a linear projection and squared $\ell_2$ loss, without masked-token prediction or intermediate-layer supervision. With Objects365 detector pre-training, GTR-L achieves 58.9 box AP on COCO \texttt{val2017} with 1.908\,ms median batch-one latency under compiled FP16 execution on an RTX~4090. The same backbone also transfers to instance segmentation, pose estimation, oriented detection, semantic segmentation, and monocular depth estimation. In a
    
[^210]: TransBERT：面向特定领域语言建模的合成翻译框架

    TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling

    [https://arxiv.org/abs/2609.26347](https://arxiv.org/abs/2609.26347)

    提出仅使用合成翻译文本预训练语言模型的TransBERT框架，并证明仅凭合成翻译数据即可在法语生命科学领域的各类下游任务上达到最先进性能。

    

    专业领域中非英语语言数据的稀缺严重限制了有效自然语言处理（NLP）工具的发展。我们提出了TransBERT，一个仅使用合成翻译文本进行语言模型预训练的新型框架，并介绍了可扩展的翻译工具包TransCorpus。聚焦于法语生命科学领域，我们的方法表明，仅利用合成翻译数据即可在各种下游任务上达到最先进的性能。我们发布了TransCorpus工具包、TransCorpus-bio-fr语料库（36.4GB的法语生命科学文本）、TransBERT-bio-fr及其相关的预训练语言模型，以及用于预训练和微调的可复现代码。我们的结果突显了在高资源翻译方向上利用合成翻译来构建低资源语言/领域对高质量NLP资源的可行性。

    arXiv:2609.26347v1 Announce Type: new  Abstract: The scarcity of non-English language data in specialized domains significantly limits the development of effective Natural Language Processing (NLP) tools. We present TransBERT, a novel framework for pre-training language models using exclusively synthetically translated text, and introduce TransCorpus, a scalable translation toolkit. Focusing on the life sciences domain in French, our approach demonstrates that state-of-the-art performance on various downstream tasks can be achieved solely by leveraging synthetically translated data. We release the TransCorpus toolkit, the TransCorpus-bio-fr corpus (36.4GB of French life sciences text), TransBERT-bio-fr, its associated pre-trained language model and reproducible code for both pre-training and fine-tuning. Our results highlight the viability of synthetic translation in a high-resource translation direction for building high-quality NLP resources in low-resource language/domain pairs.
    
[^211]: 模块化范数RandOpt：通过架构感知扰动实现种群高效的模型集成

    Modular Norm RandOpt: Population-Efficient Ensembling through Architecture-Aware Perturbations

    [https://arxiv.org/abs/2609.25745](https://arxiv.org/abs/2609.25745)

    提出模块化范数RandOpt，利用模块级自然范数与校准尺度的架构感知权重扰动进行采样，在保持选择与投票机制不变的情况下，将所需候选模型数量在Countdown上减少3倍、在GSM8K上减少至少12倍，并在多个任务、多个模型家族和规模上取得更高的平均准确率。

    

    RandOpt通过对权重扰动的语言模型进行采样，并经由多数投票集成排名靠前的候选模型，但其全局扰动尺度忽略了各模块异构的几何结构。我们提出了模块化范数RandOpt（Modular Norm RandOpt），这是一种架构感知的采样方法，利用模块级自然范数和校准的扰动尺度，同时保留原有的选择与投票机制。该方法在Countdown任务上仅用RandOpt三分之一的候选模型数量即可超越RandOpt，在GSM8K上所需候选数量至少减少12倍，并相应节省了实际运行时间。在七个任务和三个Qwen模型规模（0.5B–3B）上的评估表明，在所有规模下，该方法在Countdown、GSM8K和MATH-500上的平均准确率均高于RandOpt。这些增益还扩展到了Llama 3.2 3B和Gemma 3 4B在Countdown和GSM8K任务上的表现。在Qwen2.5-1.5B上，在相当的主运行评估预算下，我们的集成方法在两个任务上也比迭代式基线取得了更高的平均准确率。在GSM8K上，尾部密度诊断表明仅有$……

    arXiv:2609.25745v1 Announce Type: new  Abstract: RandOpt samples weight-perturbed language models and ensembles top-ranked candidates through plurality voting, but its global perturbation scale ignores heterogeneous module geometry. We propose \mbox{\textbf{\emph{Modular Norm RandOpt}}}, an architecture-aware sampling method using module-wise natural norms and calibrated scales while preserving selection and voting. It outperforms RandOpt using $3\times$ fewer candidates on Countdown and at least $12\times$ fewer on GSM8K, with corresponding wall-clock savings. Evaluations across seven tasks and three Qwen scales ($0.5$B--$3$B) show higher mean accuracy than RandOpt on Countdown, GSM8K, and MATH-500 at every scale. The gains extend to Llama 3.2 $3$B and Gemma 3 $4$B on Countdown and GSM8K. On Qwen2.5-1.5B, our ensembles also achieve higher mean accuracy than iterative baselines on both tasks at comparable main-run evaluation budgets. On GSM8K, a tail-density diagnostic implies only a $
    
[^212]: 面向最大独立集的双重图神经网络多层粗化方法

    Dual-GNN Multilevel Coarsening for Maximum Independent Set

    [https://arxiv.org/abs/2609.25149](https://arxiv.org/abs/2609.25149)

    提出了一种基于学习的图边稀疏化方法GES，通过融合几何结构信息与组合优化技术为欧几里得TSP实例自适应生成稀疏图，在保持解与最优值差距1%以内的同时最多可剪枝95%的边并显著加速求解。

    

    精确求解大规模旅行商问题（TSP）实例的计算代价十分高昂。研究人员通常采用图稀疏化方法来提高计算效率。传统的稀疏化方法通常依赖固定的启发式规则，未能充分利用特定实例的结构信息。在本文中，我们提出了图边稀疏化（Graph Edge Sparsification, GES），这是一种面向欧几里得TSP的基于学习的稀疏化方法。通过融合几何结构信息与组合优化技术，我们提出的方法能够针对不同实例自适应地生成稀疏化图，显著减小图的规模并加速求解过程。实验结果表明，我们的稀疏化方法在MATILDA数据集上最多可剪枝95%的边，同时将解的间隙保持在最优值的1%以内。此外，我们的方法还展现出很强的泛化能力。

    arXiv:2609.25149v1 Announce Type: new  Abstract: Solving large-scale instances of the Traveling Salesman Problem (TSP) exactly is computationally expensive. Researchers often employ graph sparsification methods to improve computational efficiency. Traditional sparsification methods typically rely on fixed heuristics and fail to fully exploit instance-specific structural information. In this paper, we propose Graph Edge Sparsification (GES), a learning-based sparsification approach for Euclidean TSP. By incorporating geometric structural information and combinatorial optimization technology, our proposed method adaptively generates a sparsification graph for different instances, significantly reducing the graph size and accelerating the solving process. Experimental results demonstrate that our sparsification method can prune up to 95\% of edges on the MATILDA dataset, while keeping the solution gap within 1\% of the optimal value. Moreover, our approach exhibits strong generalization c
    
[^213]: 从概念对齐到因果锚定：思维链忠实性的干预测试

    From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness

    [https://arxiv.org/abs/2609.23065](https://arxiv.org/abs/2609.23065)

    该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。

    

    思维链可以听起来合理，却可能对模型的底层推理不忠实。以往大多数工作通过输入-输出行为或输入归因来探究CoT的忠实性，而对内部计算的探索在很大程度上仍属空白。我们转而将忠实性界定为内部概念锚定问题：大语言模型（LLM）的CoT推理是否调用了支持其直接预测的相同内部概念，并且这些共享概念是否因果性地驱动其答案？使用单个共享的稀疏自编码器（SAE）——一种对LLM所使用潜在概念的可靠近似器——来编码预测过程和CoT过程，使二者的内部概念可以直接比较。我们提出了三个概念层面的相关性对齐度量指标，以及一个因果度量指标Δp，该指标通过消融共享概念并测量答案概率的下降来检验因果作用。在五个LLM和四个数据集上的实验表明，概念对齐总体上较高，正如t……（摘要在此处截断）

    arXiv:2609.23065v1 Announce Type: new  Abstract: Chain-of-thought (CoT) can sound plausible yet be unfaithful to the model's underlying reasoning. Most prior work probes CoT faithfulness through input--output behavior or input attributions, leaving internal computation largely underexplored. We instead cast faithfulness as internal concept grounding: Does a large language model's (LLM) CoT reasoning engage the same internal concepts that support the LLM's direct prediction, and do the shared concepts causally drive its answer? Encoding a prediction pass and a CoT pass with a single shared sparse autoencoder (SAE), a reliable approximator of the latent concepts LLMs use, makes their internal concepts directly comparable. We introduce three correlational metrics of concept-level alignment and a causal metric, $\Delta p$, which ablates the shared concepts and measures the drop in answer probability. Across five LLMs and four datasets, concept alignment is generally high, as indicated by t
    
[^214]: 2³² 边界处的静默失败：关于 PyTorch Apple MPS 后端中大张量矩阵乘法的技术报告

    Silent Failures at the $2^{32}$ Boundary: A Technical Report on Large-Tensor Matrix Multiplication in PyTorch's Apple MPS Backend

    [https://arxiv.org/abs/2609.22991](https://arxiv.org/abs/2609.22991)

    该技术报告首次系统揭示了 PyTorch Apple MPS 后端在张量超过 2³² 元素边界时会静默返回严重错误的批量矩阵乘法结果（相对误差超过 1 且无任何警告），并通过大规模实验总结出三条可解释所有失败情况的规则。

    

    配备 192GB 或更多统一内存的 Apple Silicon 机器，使得在桌面 GPU 上放置超过 2³² 个元素的张量成为常态。我们证明，PyTorch 的 Metal Performance Shaders (MPS) 后端在这种规模下会静默地返回错误的批量矩阵乘法结果。在 macOS 27.0 上，torch.bmm（因此 torch.matmul 和 eager attention 也一样）在不抛出异常或警告的情况下返回超过 1 的相对误差，这一现象出现在我们测试的从 2.4.1 到 2.14.0 的每一个 PyTorch 版本中。在一台机器上，我们对 bmm 进行了系统性扫描，涵盖两种数据类型、四种内存布局、六种形状以及 4096 到 65538 之间的 42 个批量大小（在 PyTorch 2.14.0 上进行了 1584 次运行，并在十个更早的版本上进行了缩减版扫描），并将每个结果与 CPU 上的 float64 计算进行对比验证。三条规则可以解释 2.14.0 上的所有结果。当输出超过 2³² 个元素且某个操作数是转置视图时，整个输出都是错误的，并且等价于一个忽略了该转置的计算……

    arXiv:2609.22991v1 Announce Type: cross  Abstract: Apple Silicon machines with 192GB or more of unified memory make it routine to place tensors with more than $2^{32}$ elements on a desktop GPU. We show that PyTorch's Metal Performance Shaders (MPS) backend silently returns wrong results for batched matrix multiplication at this scale. On macOS 27.0, torch.bmm, and therefore torch.matmul and eager attention, returns relative errors above 1 without an exception or a warning, in every PyTorch release from 2.4.1 to 2.14.0 that we tested. On one machine, we sweep bmm over two dtypes, four memory layouts, six shapes and 42 batch sizes between 4096 and 65538 (1584 runs on PyTorch 2.14.0, and a reduced sweep on ten earlier releases), and judge every result against a float64 computation on the CPU. Three rules account for every outcome on 2.14.0. When the output exceeds $2^{32}$ elements and an operand is a transposed view, the entire output is wrong and equals a computation that ignores the s
    
[^215]: 测试LLM智能体中功能性效价轴的构念效度

    Testing the Construct Validity of a Functional Valence Axis in LLM Agents

    [https://arxiv.org/abs/2609.22850](https://arxiv.org/abs/2609.22850)

    该研究通过分离“结果本身”与“获知结果的信息历史”的受控干预，检验LLM智能体中“好—坏”效价方向的构念效度，发现该方向可跨表面形式迁移，但对结果是否被提前告知高度敏感，说明其效价表征与信息历史相互纠缠。

    

    对比激活方向通常根据它们能解码出什么内容、或它们引导行为的强度来解释。但什么样的证据才足以识别这样一个方向所代表的构念，而不是用于提取该方向的对比中与之相关的特征？我们在迷宫任务中针对“好—坏结果方向”研究这一问题，采用受控干预方法，将已实现的结果与获知该结果的信息历史分离开来。在多个LLM检查点上，基于一种显式结果编码拟合的方向能够很好地迁移到另一种编码，表明该读出机制并不依赖于表面形式。相反，当相同的已实现结果通过“已提前告知”和“未提前告知”两种历史达成时，迁移显著退化：即使两种历史最终都接收到相同的显式结果，事件后的读出仍然强烈地依赖于先前的告知信息。在一个匹配的迷宫强化学习运行中……

    arXiv:2609.22850v1 Announce Type: new  Abstract: Contrastive activation directions are often interpreted from what they decode or how strongly they steer behavior. But what evidence is sufficient to identify the construct represented by such a direction, rather than a correlated feature of the contrast used to extract it? We study this question for a good--bad outcome direction in a maze task, using controlled interventions that separate the realised outcome from the informational history through which it became known. Across multiple LLM checkpoints, directions fitted on one explicit outcome encoding transfer well to another, indicating that the readout is not tied to surface form. In contrast, when the same realised outcome is reached through announced and unannounced histories, transfer degrades substantially: even after both histories receive the same explicit outcome, the post-event readout remains strongly conditioned on the earlier announcement. In a matched maze-RL run, the pos
    
[^216]: SPIBER：利用生成流网络从短的、未收敛的轨迹重建自由能景观

    SPIBER: Reconstructing Free Energy Landscapes from Short, Unconverged Trajectories with Generative Flow Networks

    [https://arxiv.org/abs/2609.22663](https://arxiv.org/abs/2609.22663)

    SPIBER将状态预测信息瓶颈（SPIB）与生成流网络（GFlowNets）相结合，能够从短的、未收敛的分子模拟轨迹中识别慢集体变量并重建自由能景观，纠正不同亚稳态采样不平衡的问题。

    

    分子系统具有众多自由度，但其亚稳态行为通常可以用少数几个集体变量来描述。从有限的模拟数据中识别这些变量并估计沿这些变量的自由能，仍然是一个具有挑战性且重要的问题。相互独立的短轨迹可能采样到不同的亚稳态，却无法捕获状态间的转变，也无法确定它们的相对平衡布居数。对于在单一温度下使用相同哈密顿量生成的无偏轨迹，基于直方图重加权的替代方法无法纠正这种不平衡。在此，我们提出了SPIBER，它将状态预测信息瓶颈（SPIB）与生成流网络相结合。SPIB利用深度学习，通过过去-未来信息瓶颈来近似慢自由度，保留预测未来亚稳态所需的信息。我们表明，这种压缩限制了条件熵……（摘要在此处截断）

    arXiv:2609.22663v1 Announce Type: cross  Abstract: Molecular systems have many degrees of freedom, but their metastable behavior can often be described by a few collective variables. Identifying these variables and estimating free energies along them from limited simulation data remains a challenging, important problem. Separate short trajectories may sample different metastable states without capturing transitions or establishing their relative equilibrium populations. For unbiased trajectories generated with the same Hamiltonian at a single temperature, alternate methods based on histogram reweighting cannot correct this imbalance. Here we present SPIBER, which combines the State Predictive Information Bottleneck (SPIB) with Generative Flow Networks (GFlowNets). SPIB uses deep learning to approximate slow degrees of freedom through a past-future information bottleneck, retaining information needed to predict future metastable states. We show that this compression limits conditional e
    
[^217]: 黎曼随机优化的局部私有推断

    Locally Private Inference for Riemannian Stochastic Optimization

    [https://arxiv.org/abs/2609.22642](https://arxiv.org/abs/2609.22642)

    该论文提出了一种在局部差分隐私下对流形值总体极小值点进行统计推断的方法，通过条件中心化的随机切梯度保持一阶方程，并引入对称对回归（SPR）从相同私有消息中估计渐近方差，进而证明了中心极限定理及基于交互记录的三明治协方差和内在Wald区域的一致性。

    

    我们针对流形值的总体极小值点发展了统计推断方法，适用于每个观测属于不同参与者、且分析师只能接收到局部私有消息的场景。该方法释放随机化的切空间梯度，并通过黎曼随机逼近和Polyak-Ruppert平均将其组合。将私有数据替代量直接插入非线性损失可能会移动其总体目标，而对释放梯度进行条件中心化则可以保持一阶方程。我们引入对称对回归（SPR），从用于点估计的相同私有消息中估计渐近方差，无需保留部分参与者或请求第二次数据释放。我们在局部差分隐私下证明了中心极限定理，以及完全基于交互记录的三明治协方差和内在Wald区域的一致性。在多种统计问题和流形上的模拟支持了该方法的预测性能。

    arXiv:2609.22642v1 Announce Type: cross  Abstract: We develop inference for manifold-valued population minimizers when each observation belongs to a different participant and only locally private messages reach the analyst. The method releases randomized tangent gradients and combines them through Riemannian stochastic approximation and Polyak-Ruppert averaging. Directly inserting a private data surrogate into a nonlinear loss can shift its population target, whereas conditional centring of the released gradient preserves the first-order equation. We introduce symmetric-pair regression (SPR) to estimate the asymptotic variance from the same private messages used for point estimation, without holding out participants or requesting a second release. We prove the central limit theorem and consistency of the fully transcript-based sandwich covariance and intrinsic Wald region under local differential privacy. Simulations across various statistical problems and manifolds support the predict
    
[^218]: 共同因果，而非交叉注意力：阻断音视频生成中的视觉捷径

    Common Cause, Not Cross-Attention: Blocking Visual Shortcuts in Audio-Video Generation

    [https://arxiv.org/abs/2609.22361](https://arxiv.org/abs/2609.22361)

    本文通过受控因果研究揭示，音视频联合生成模型中让音频直接读取视频（如交叉注意力）会学到基于外观而非因果事件预测声音的“视觉捷径”，且流行的共享共同因果潜变量方案也无法修复这一失效模式。

    

    联合音频-视频生成器训练所用的数据中，事件的外观与其声音之间存在强烈且往往是虚假的相关性：特定的材质、纹理或物体外观总是与特定的声音同时出现。本文是对由此产生的失效模式的一项受控因果研究。通过构建一个音视频结构因果模型——其中音频在构造上独立于视频的干扰性外观——我们证明，那些让音频直接读取视频的模型（通过交叉注意力或共享潜变量）会学到一种“视觉捷径”：它们根据外观而非因果事件来预测声音，当测试时外观与事件的关联被打破时，模型会崩溃，甚至直接合成出错误事件的声音。至关重要的是，将两种模态经由共享的共同因果潜变量进行路由这一流行补救方法并不能解决该问题：瓶颈结构、无监督的共享/私有因子分解，以及忠实的共享……（原文摘要被截断）

    arXiv:2609.22361v1 Announce Type: new  Abstract: Joint audio--video generators are trained on data in which what an event looks like and what it sounds like are strongly, often spuriously, correlated: a particular material, texture, or object appearance co-occurs with a particular sound. This paper is a controlled causal study of the resulting failure mode. Building an AV structural causal model in which the audio is, by construction, independent of the video's nuisance appearance, we show that models which let audio read video directly-through cross-attention or a shared latent-learn a visual shortcut: they predict sound from appearance rather than from the causal event, and collapse when the appearance-event correlation is broken at test time, literally synthesizing the wrong event's sound. Crucially, the popular remedy of routing both modalities through a shared common-cause latent does not fix this: a bottleneck, an unsupervised shared/private factorization, and a faithful shared-p
    
[^219]: 面向结构化神经网络剪枝的任务感知混合QUBO优化

    Task-Aware Hybrid QUBO Optimization for Structured Neural Network Pruning

    [https://arxiv.org/abs/2609.22238](https://arxiv.org/abs/2609.22238)

    提出了一种任务感知的混合QUBO优化框架，将一阶泰勒敏感性和权重-费雪敏感性等任务信息与滤波器间的激活相似性交互相结合，并通过容量激励二分搜索控制剪枝基数，实现结构化神经网络滤波器剪枝。

    

    神经网络剪枝可以被表述为一个组合优化问题，然而许多现有方法依赖于独立的滤波器重要性评分或简化的目标函数。在这项工作中，我们提出了一种用于结构化滤波器剪枝的混合二次无约束二元优化（QUBO）框架，该框架将任务感知的敏感性信息与候选滤波器之间的交互作用相结合。该公式将一阶泰勒敏感性和权重-费雪敏感性纳入目标的线性部分，并且还可以将激活相似性纳入二次交互项中。为了在不引入显式二次基数惩罚的情况下控制目标剪枝基数，我们对容量激励进行二分搜索，以找到经验上能够产生目标剪枝基数的系数。我们进一步研究了一种两阶段的QUBO-张量列车（Tensor-Train）精炼策略，其中……

    arXiv:2609.22238v1 Announce Type: new  Abstract: Neural network pruning can be formulated as a combinatorial optimization problem, yet many existing approaches rely on independent filter-importance scores or simplified objective functions. In this work, we propose a Hybrid Quadratic Unconstrained Binary Optimization (QUBO) framework for structured filter pruning that combines task-aware sensitivity information with interactions between candidate filters. The formulation incorporates first-order Taylor sensitivity and Weight-Fisher sensitivity into the linear component of the objective and can additionally incorporate activation similarity into the quadratic interactions. To control the target pruning cardinality without introducing an explicit quadratic cardinality penalty, we use a binary search over the capacity incentive to identify a coefficient that empirically yields the target pruning cardinality. We further investigate a two-stage QUBO--Tensor-Train refinement strategy in which
    
[^220]: 黎曼流形上切向量场回归的同时推断

    Riemannian Simultaneous Inference for Tangent Vector Field Regression

    [https://arxiv.org/abs/2609.21910](https://arxiv.org/abs/2609.21910)

    该论文针对无边黎曼流形上的切向量场回归提出了一种基于平行输运与体积校正的核估计方法，并通过单位切丛上的上确界表示与 Gumbel 极限理论，构造了回归场的可行同时置信管。

    

    我们考虑无边黎曼流形上的非参数切向量场回归。由于不同点处的响应位于不同的切空间中，所提出的核估计量首先将近邻的响应平行输运到目标切空间，然后形成经过体积校正的局部平均。我们首先推导了该估计量的一致二阶偏差、有限带宽协方差以及随机收敛率。对于同时推断，我们将切范数表示为单位切丛上的上确界。精确的协方差白化给出了一个单位方差的高斯场，其相关长度沿底流形为 $h$ 阶，沿纤维为一阶。其局部协方差几何导致了一个带有显式内蕴常数的 Gumbel 极限。将该极限与高斯近似和交叉拟合协方差估计相结合，得到了回归场的可行同时置信管。我们进一步讨论……（摘要被截断）

    arXiv:2609.21910v1 Announce Type: cross  Abstract: We consider nonparametric tangent vector field regression on a Riemannian manifold without boundary. Because responses at different points lie in different tangent spaces, the proposed kernel estimator first parallel transports nearby responses to the target tangent space and then forms a volume-corrected local average. We first derive its uniform second-order bias, finite-bandwidth covariance, and stochastic rate. For simultaneous inference, the tangent norm is written as a supremum over the unit tangent bundle. Exact covariance whitening gives a unit-variance Gaussian field whose correlation length is of order $h$ along the base manifold and of order one along the fibre. Its local covariance geometry leads to a Gumbel limit with an explicit intrinsic constant. Combining this limit with Gaussian approximation and cross-fitted covariance estimation yields a feasible simultaneous confidence tube for the regression field. We further disc
    
[^221]: 一种用于重构对流流动的物理信息神经网络的改进周期激活函数

    An improved periodic activation for PINNs reconstructing convective flows

    [https://arxiv.org/abs/2609.21798](https://arxiv.org/abs/2609.21798)

    提出了一种基于复指数函数的周期激活架构，通过同时生成正弦和余弦输出，在瑞利-贝纳德对流的温度重构任务中以相近甚至更低的计算成本显著提升了物理信息神经网络的重建质量。

    

    在物理信息神经网络的众多应用中，采用周期激活函数的网络架构已被证明相较于单调激活函数具有显著优势。本文研究了一种使用复指数函数作为激活函数的网络架构，该激活函数能够同时生成正弦和余弦输出对。通过在立方体瑞利-贝纳德对流中从稀疏速度数据重构温度场的任务上，将其与类似的正弦激活多层感知机进行对比测试，结果显示该架构在重构质量上有显著提升，而每个训练步骤的计算成本并未大幅增加。反之，这种改进的架构也能够以一小部分的计算开销达到相似的重构质量。对这些网络数学结构的分析表明，性能提升的根源在于同时向前传递正弦和余弦函数这一特性。

    arXiv:2609.21798v1 Announce Type: cross  Abstract: Architectures with periodic activation functions have already been shown to be beneficial in comparison to monotonic counterparts for a wide range of applications of physics-informed neural networks. Here, we investigate a network architecture which uses the complex exponential function, generating pairs of sine and cosine outputs as activation functions. Testing it against comparable, sine-activated multi-layer perceptrons for the task of temperature reconstruction from sparse velocity data for cubic Rayleigh-B\'enard convection reveals significant improvements in the reconstruction quality without a substantial increase in computational cost per training step. Vice versa, the improved architecture enables reaching similar reconstruction qualities for a fraction of the expense. Analyzing the mathematical structure of these networks points to the improvements being rooted in the property of passing both a sine and cosine function forwa
    
[^222]: 加权量子信号处理：低深度多项式逼近及其在Kolmogorov-Arnold网络中的应用

    Weighted Quantum Signal Processing: Low-Depth Polynomial Approximation with Applications to Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2609.21567](https://arxiv.org/abs/2609.21567)

    本文提出加权量子信号处理（WQSP），通过为中心旋转算子引入权重函数突破了QSP的电路深度瓶颈和奇偶性约束，在实现任意有界一元多项式时将所需参数数量从线性级降至指数级缩减，并展示了其在Kolmogorov-Arnold网络中的应用。

    

    量子信号处理是一种用于生成和逼近一元多项式的强大量子框架。然而，QSP常常受到电路深度瓶颈以及可实现多项式类别的奇偶性约束的限制。在这项工作中，我们提出了加权量子信号处理，这是QSP的一种扩展，其中为中心旋转算子分配了一个权重函数。这一表述为QSP提供了更深入的理解，QSP作为WQSP在单位权重下的特例而出现。权重的选择决定了WQSP电路的结构和表达能力。当权重为大于1的自然数时，WQSP简化为QSP的剪枝版本，揭示了标准框架中的参数冗余。通过适当选择整数权重，WQSP在实现任意有界一元多项式所需的参数数量上实现了从线性到指数级的缩减。

    arXiv:2609.21567v1 Announce Type: cross  Abstract: Quantum Signal Processing is a powerful quantum framework for generating and approximating univariate polynomials. However, QSP is often limited by circuit-depth bottlenecks and parity constraints on the class of realizable polynomials. In this work, we introduce Weighted Quantum Signal Processing, an extension of QSP in which a weight function is assigned to the central rotation operator. This formulation provides a deeper understanding of QSP, which emerges as the special case of WQSP with unit weights. The choice of weights determines the structure and expressive capabilities of WQSP circuits. When the weights are natural numbers greater than one, WQSP reduces to a pruned version of QSP, revealing parameter redundancies in the standard framework. Through appropriate selection of integer weights, WQSP achieves linear-to-exponential reductions in the number of parameters required to realize arbitrary bounded univariate polynomials whi
    
[^223]: SWE-Proof：语言模型能否通过机器校验的证明解决真实世界的问题？

    SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?

    [https://arxiv.org/abs/2609.21190](https://arxiv.org/abs/2609.21190)

    该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。

    

    确保大语言模型（LLM）生成代码的正确性是现代软件工程的核心挑战。面向智能体代码生成的基准测试通常使用留出的测试套件来检验正确性，但测试套件本质上是不完整的，且日益容易受到模型记忆（数据泄露）的影响。形式化验证可以同时避免这两个问题，但现有工作仅覆盖规范以输入形式给出的独立任务，而非真实问题——真实问题涉及大型代码仓库，并以模糊的自然语言表达意图。我们提出了Benchproofer，一个能将带有已知正确补丁的编码任务转化为形式化验证任务的流水线：它为新代码编写规范，用公理概括新代码所调用的已有函数，并且只有在机械验证与对抗性检查两道关卡均通过后才接受一个实例。将该流水线应用于SWE-bench Verified，我们得到了SWE-Proof——包含500个真实问题的基准，其正确性通过形式化验证而非测试来保证，并且该方法还可扩展至SWE-bench Pro。……

    arXiv:2609.21190v1 Announce Type: cross  Abstract: Ensuring the correctness of LLM-generated code is a core challenge for modern software engineering. Benchmarks for agentic code generation check correctness with held-out test suites, which are inherently incomplete and increasingly susceptible to memorization. Formal verification avoids both problems, but existing work covers only standalone tasks whose specifications are given as input, not real issues, which touch large repositories and state intent in vague natural language. We present Benchproofer, a pipeline that turns a coding task with a known correct patch into a formally verified one: it writes a specification for the new code, summarizes the existing functions that code calls with axioms, and admits an instance only after mechanical and adversarial gates agree. Applying it to SWE-bench Verified yields SWE-Proof, 500 real issues whose correctness is formally verified rather than tested, and it extends to SWE-bench Pro. Across
    
[^224]: Uni-LaDiR：潜在扩散统一多模态推理

    Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning

    [https://arxiv.org/abs/2609.19878](https://arxiv.org/abs/2609.19878)

    Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。

    

    多模态推理要求模型在整个推理过程中利用来自多种模态的信息。然而，现有方法通常将特定模态的思维标记拼接在单一序列中，使得模型在跨模态推理时需要自行弥合表示上的差异。我们提出了Uni-LaDiR（统一潜在扩散推理器），这是一个将这些思维引入共享潜在空间进行推理的框架。统一编码器将来自不同模态的教师推理步骤映射为共享的思维标记，并通过训练来保留后续推理步骤及最终答案或动作所需的信息。由于相同的上下文可以支持多个有效的下一步推理，我们使用扩散模型基于输入和先前块来预测下一块思维标记。通过共享模型权重联合训练编码器和扩散推理器，促使思维标记既对任务有用，又……

    arXiv:2609.19878v1 Announce Type: cross  Abstract: Multimodal reasoning requires models to draw on information from multiple modalities throughout the reasoning process. Yet existing methods often concatenate modality-specific thought tokens in a single sequence, leaving the model to bridge representational differences as it reasons across modalities. We introduce Uni-LaDiR (Unified Latent Diffusion Reasoner), a framework that brings these thoughts into a shared latent space for reasoning. A unified encoder maps teacher reasoning steps from different modalities into shared thought tokens, trained to preserve the information needed for later reasoning steps and the final answer or action. Because the same context can support multiple valid next steps, we use diffusion to predict the next block of thought tokens from the input and preceding blocks. Jointly training the encoder and diffusion reasoner with shared model weights encourages thought tokens to be both useful for the task and pr
    
[^225]: DeliveryGym：一个具有自适应课程的长时程具身智能体规划强化学习环境

    DeliveryGym: An RL Environment for Long-Horizon Embodied Agent Planning with Adaptive Curriculum

    [https://arxiv.org/abs/2609.19801](https://arxiv.org/abs/2609.19801)

    DeliveryGym是一个面向长时程具身智能体规划的3D强化学习环境，通过连续快递员班次、持久世界动态和基于模拟器事件的轨迹奖励来评估决策成本，并采用自适应课程机制根据智能体观察到的弱点动态调整训练难度。

    

    可执行环境使大语言模型（LLM）智能体能够从其行动的后果中学习。对于具身智能体而言，这些后果不仅限于当前任务是否成功：完成一次配送可能会消耗后续工作所需的时间、精力或金钱。因此，学习规划需要能够保留这些依赖关系，并将其转化为贯穿完整轨迹反馈的环境。我们推出了DeliveryGym，这是一个用于在连续快递员班次上评估和训练智能体的3D环境。它将多模态工具交互与持久的世界动态相结合，并从模拟器事件中计算轨迹奖励，使智能体决策的成本可用于强化学习（RL）。该环境还能根据策略所观察到的弱点调整未来的训练班次，同时保持评估固定不变。在六个模型和13个城市地图的测试中，评估揭示了可靠执行分配的配送任务与……之间的差距

    arXiv:2609.19801v1 Announce Type: new  Abstract: Executable environments enable LLM agents to learn from the consequences of their actions. For embodied agents, those consequences extend beyond whether the current task succeeds: completing a delivery can consume the time, energy, or money needed for later work. Learning to plan therefore requires environments that preserve these dependencies and turn them into feedback across a complete trajectory. We introduce DeliveryGym, a 3D environment for evaluating and training agents on continuous courier shifts. It couples multimodal tool interaction with persistent world dynamics and computes trajectory rewards from simulator events, making the costs of an agent's decisions available for reinforcement learning (RL). The environment also adapts future training shifts to the policy's observed weaknesses while keeping evaluation fixed. Across six models and 13 city maps, evaluation exposes a gap between reliably executing assigned deliveries and
    
[^226]: 学会自己的思考：抽象token课程学习

    Learn Your Own Thoughts: Abstract Token Curriculum

    [https://arxiv.org/abs/2609.19717](https://arxiv.org/abs/2609.19717)

    提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。

    

    大语言模型（LLMs）通过利用思维链（CoT）作为思考中间阶段的草稿板，已经获得了卓越的推理能力。然而，CoT技术需要对思考token进行显式监督，这需要丰富的、特定任务的数据。在这项工作中，我们提出了抽象token课程学习（Abstract Token Curriculum, ATC），这是一种新颖的课程学习框架，能够在没有直接监督或手动草稿板设计的情况下，引出有效的连续中间表示。ATC通过一系列分布逐渐增加问题复杂度，训练模型在连续表示空间中发展出内部的抽象“思维”。本文为ATC的优势及其相对于以往训练连续思维方法的长处提供了理论和实验证据。理论上，我们证明了使用ATC在单层softmax注意力机制下学习奇偶函数时……

    arXiv:2609.19717v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attent
    
[^227]: QVAC Genesis III：一个用于高效语言模型预训练的大规模高质量开放合成STEM语料库

    QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training

    [https://arxiv.org/abs/2609.19513](https://arxiv.org/abs/2609.19513)

    提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。

    

    高质量预训练数据是面向边缘AI和端侧部署的教育及STEM专用语言模型的关键瓶颈，在这些场景中token预算受到严格限制。尽管各大机构在私有语料库上训练越来越大的模型，但开放生态系统中缺乏能够以高效方式为小模型提供高单token学习价值的STEM导向合成数据集。为填补这一空白，我们推出了QVAC Genesis III，这是一个拥有1914.3亿token、以STEM为核心的多领域合成语料库，涵盖19个领域，并包含多个难度级别和不同的教育风格。QVAC Genesis III通过一种双重生成策略构建，该策略以一个弱小的边缘规模学生模型作为信号进行针对性教师蒸馏：学生的失败被转化为纠正性解释，而其成功则被扩展为针对所有答案选项的对比性选项级推理。我们进一步引入了LLM作为解析器的机制……（原文摘要在此处截断）

    arXiv:2609.19513v1 Announce Type: new  Abstract: High-quality pre-training data is a critical bottleneck for educational and STEM-specific language models targeting edge AI and on-device deployment where token budgets are tightly constrained. While major organizations train ever-larger models on private corpora, the open ecosystem lacks STEM-focused synthetic datasets that deliver high per-token learning value efficiently for small models. To address this gap, we introduce QVAC Genesis III, a 191.43B-token, STEM-focused multi-domain synthetic corpus covering 19 domains across several difficulty levels and different educational styles. QVAC Genesis III is built via a dual generation strategy that performs targeted teacher distillation using a weak edge-scale student model as signal: the student's failures are converted into corrective explanations, while its successes are expanded into contrastive option-level reasoning over all answer choices. We further introduce an LLM-as-a-parser ev
    
[^228]: Agora：以Git作为集体自动研究的共享内存

    Agora: Git as Shared Memory for Collective AutoResearch

    [https://arxiv.org/abs/2609.18094](https://arxiv.org/abs/2609.18094)

    Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。

    

    诸如AutoResearch之类的自主研究循环表明，单个编码智能体可以在无人值守的情况下改进训练设置。但如果同时运行多个这样的智能体，每个会话都会从零开始，因此更多的智能体往往意味着更多的重复搜索，而非更多的发现。Agora是这类智能体的共享内存：研究以仅追加的有向无环图（DAG）的形式记录在Git中，使得每一条主张都是一个任何人都可以检出并重新运行的提交。每个结果、见解、假设、验证和报告都是一个不可变的提交，其父边标明它建立在哪些工作之上；一个派生索引用于揭示研究前沿、被忽视的分支以及每条主张的验证状态，而一种多样性感知的选择规则可防止社区坍缩到单一领导者上。我们描述了该系统并报告了它的首次持续使用情况：一次持续近12天的运行，13个语言模型工作者在没有任务分配、没有中央规划者的情况下，针对一个权重转……

    arXiv:2609.18094v1 Announce Type: cross  Abstract: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-tran
    
[^229]: TabPFN-3.5：技术报告

    TabPFN-3.5: Technical Report

    [https://arxiv.org/abs/2609.17895](https://arxiv.org/abs/2609.17895)

    TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。

    

    我们推出 TabPFN-3.5，这是我们的全新旗舰表格基础模型。它在广泛的表格任务上显著超越了其前代模型 TabPFN-3 以及所有现有基线。TabPFN-3.5 在 TabArena 的标准表格预测任务上创造了新的最先进水平，并将其扩展到实际从业者会遇到的数据场景：具有时间或分组划分的非独立同分布数据、包含字符串、文本和图像的表格、高基数类别特征，以及具有众多特征的宽表。这些优势延续到我们的任务专用框架中：在关系型数据上达到最先进水平，并具备更强的时间序列预测能力。为了实现更快的推理，我们的变体 TabPFN-3.5-Fast 运行速度最高可达 TabPFN-3 的 3 倍，同时保留了大部分精度提升。此外，我们升级了 TabPFN-3.5-Plus，通过先进的文本和日期处理以及专有推理优化扩展了多模态能力。最后，我们发布了一个新的版本。

    arXiv:2609.17895v1 Announce Type: new  Abstract: We introduce TabPFN-3.5, our new flagship Tabular Foundation Model. It significantly outperforms its predecessor, TabPFN-3, and all existing baselines across a broad range of tabular problems. TabPFN-3.5 sets a new state of the art on standard tabular prediction in TabArena, and extends it to the data practitioners encounter in practice: non-i.i.d. data with temporal or grouped splits, tables with strings, text and images, high-cardinality categorical features, and wide tables with many features. These gains carry over to our task-specific harnesses: state of the art on relational data and stronger time-series forecasting. For faster inference, our variant TabPFN-3.5-Fast runs up to 3x faster than TabPFN-3 while keeping most of the accuracy gains. In addition, we upgrade TabPFN-3.5-Plus, expanding our multimodal capabilities with advanced text and date handling alongside proprietary inference optimizations. Finally, we release a new vers
    
[^230]: MyoFlow：面向跨会话与跨被试高密度表面肌电手势识别的锚点绑定校正流

    MyoFlow: Anchor-Tied Rectified Flow for HD-sEMG Gesture Recognition Across Sessions and Subjects

    [https://arxiv.org/abs/2609.17194](https://arxiv.org/abs/2609.17194)

    提出了首个面向跨会话与跨被试高密度表面肌电手势识别的判别式流匹配框架MyoFlow，通过域条件化校正流将分类重构为向手势锚点的输运过程，无需独立分类头即可实现零样本预测。

    

    高密度表面肌电（HD-sEMG）手势识别可支持假肢控制、辅助机器人和康复训练，但电极重新佩戴和生理个体差异引起的分布偏移会降低跨会话与跨被试场景下的识别准确率。现有的生成式高密度肌电模型主要用于信号合成以进行数据增强；尽管扩散模型能够增强表征学习，但预测仍需依赖独立的分类器。为了将学习到的动力学与决策规则绑定在一起，我们提出了MyoFlow，这是首个面向跨会话与跨被试高密度肌电识别的判别式流匹配框架。该方法将分类重构为锚点绑定的输运过程：域条件化的校正流将编码后的信号窗口输运至手势锚点，这些锚点既作为输运目标，又定义了最近锚点的决策几何结构，从而无需独立的分类头即可实现零样本预测。在Hyser数据集上，MyoFlow提升了平均跨会话（摘要在此处截断）

    arXiv:2609.17194v1 Announce Type: new  Abstract: High-density surface electromyography (HD-sEMG) gesture recognition supports prosthetic control, assistive robotics, and rehabilitation, but electrode re-donning and physiological variability cause distribution shifts that degrade accuracy across sessions and subjects. Generative HD-sEMG models primarily synthesize signals for augmentation; although diffusion models enhance representation learning, prediction still relies on a separate classifier. To tie learned dynamics to the decision rule, we propose MyoFlow, the first discriminative flow-matching framework for HD-sEMG recognition across sessions and subjects. It recasts classification as anchor-tied transport: a domain-conditioned rectified flow moves encoded windows toward gesture anchors that serve as transport targets and define the nearest-anchor decision geometry, enabling zero-shot prediction without an independent head. On the Hyser dataset, MyoFlow improves mean cross-session
    
[^231]: 高容量核联想记忆中稳定性边缘的信息几何自组织

    Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories

    [https://arxiv.org/abs/2609.16827](https://arxiv.org/abs/2609.16827)

    本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。

    

    基于核逻辑回归（KLR）的高容量联想记忆展现出卓越的存储能力与鲁棒性。先前的实证研究识别出了一个超参数区域，即“优化脊”，在该区域中吸引子的稳定性达到最大。然而，这一区域的几何本质以及到达该区域所需的优化动力学机制一直不明确。本文研究了采用KLR训练的Hopfield网络中参数空间的静态几何以及梯度下降（GD）的学习轨迹。利用Hessian矩阵的特征值谱，我们揭示了“优化脊”对应于位于秩1谱坍缩附近的一个相边界，它作为一个几何奇点，其主曲率被大幅放大。此外，我们证明了学习动力学表现出一种由稳定性边缘现象驱动的瞬态自稳定行为……

    arXiv:2609.16827v1 Announce Type: new  Abstract: High-capacity associative memories based on Kernel Logistic Regression (KLR) exhibit exceptional storage capabilities and robustness. Previous empirical studies identified a hyperparameter regime, the "Ridge of Optimization," where attractor stability is maximized. However, the geometric nature of this regime and the optimization dynamics required to reach it have remained unclear. In this paper, we investigate the static geometry of the parameter space and the learning trajectory of Gradient Descent (GD) in KLR-trained Hopfield networks. Using the eigenvalue spectrum of the Hessian, we reveal that the Ridge corresponds to a phase boundary located adjacent to a rank-1 spectral collapse, acting as a geometric singularity where the principal curvature is massively amplified. Furthermore, we demonstrate that the learning dynamics exhibit a transient self-stabilizing behavior driven by the Edge of Stability (EoS) phenomenon. Rather than seek
    
[^232]: 面向部分传感器重叠下跨机床CNC迁移的模式自适应动作条件化JEPA

    Schema-Adaptive Action-Conditioned JEPA for Cross-Machine CNC Transfer under Partial Sensor Overlap

    [https://arxiv.org/abs/2609.16071](https://arxiv.org/abs/2609.16071)

    该论文提出一种模式自适应的动作条件化JEPA架构，在源与目标CNC机床仅共享10/17个传感器通道的部分重叠情况下，通过严谨的密封目标测试协议实现零样本跨机床动力学预测迁移，将目标机器预测RMSE从0.813降至0.546。

    

    工业世界模型的跨机器部署需要在动态特性、传感接口、采样机制和控制单元变化下进行迁移。我们研究了一种用于CNC动力学的模式自适应动作条件化联合嵌入预测架构（SAAC-JEPA），其中源机器具有17个标准传感器通道，而目标机器仅共享其中10个。评估采用组不相交的源数据划分、仅源归一化、留出自监督验证、单位审计以及模型锁定后的密封目标测试。在五个随机种子下，JEPA预训练在干净源数据的预测任务中没有带来明显增益：从零开始训练的模型与预训练主体模型的RMSE分别为0.811±0.022和0.813±0.022。在仅使用源数据的20个候选方案搜索中，经过七种子稳定性检查后，选出了模式一致的动作条件化JEPA。在确认性目标测试中，锁定模型达到零样本RMSE=0.546、R²=0.012（摘要在此处被截断）。

    arXiv:2609.16071v1 Announce Type: cross  Abstract: Cross-machine deployment of industrial world models requires transfer across changes in dynamics, sensing interfaces, sampling regimes, and control units. We study a schema-adaptive action-conditioned Joint-Embedding Predictive Architecture (SAAC-JEPA) for CNC dynamics, where the source machine has 17 canonical sensor channels and the target shares only 10. Evaluation uses group-disjoint source splits, source-only normalization, held-out self-supervised validation, unit audits, and a sealed target test after model locking. Across five seeds, JEPA pretraining gives no clean-source forecasting gain: scratch and pretrained-body models obtain \(\mathrm{RMSE}=0.811\pm0.022\) and \(0.813\pm0.022\). A source-only search over 20 candidates selects a schema-consistent action-conditioned JEPA after seven-seed stability checks. On the confirmatory target pass, the locked model reaches zero-shot \(\mathrm{RMSE}=0.546\), \(R^2=0.012\), and \(\mathr
    
[^233]: VertexCBF：通过顶点受限控制搜索改进神经控制屏障函数

    VertexCBF: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search

    [https://arxiv.org/abs/2609.12831](https://arxiv.org/abs/2609.12831)

    提出VertexCBF框架，利用控制仿射动力学和凸多面体控制集的性质（哈密顿量在控制顶点处最大化），结合物理信息学习、稀疏监督学习与GPU并行的顶点受限树搜索，以可扩展、系统化且可解释的方式训练神经控制屏障函数。

    

    随着自主机器人数量的持续增长，安全性变得日益重要。控制屏障函数（CBFs）为保障安全性提供了一个有理论依据的框架，但现有的设计方法往往在有效性、可扩展性或可解释性方面存在局限，并且可能导致过于保守的安全集。在本文中，我们提出了VertexCBF，这是一种以可扩展、系统化且可解释的方式学习神经控制屏障函数的框架。我们使用神经网络来近似平稳的Hamilton–Jacobi值函数，该网络通过物理信息学习与稀疏监督学习相结合的方式进行训练。通过利用控制仿射动力学和凸多面体控制集——在此条件下哈密顿量在控制顶点处取得最大值——我们通过GPU并行的顶点受限树搜索高效生成监督点，同时残差架构保证了所学习到的CBF绝不会大于（原文此处截断）。

    arXiv:2609.12831v1 Announce Type: cross  Abstract: As the number of autonomous robots continues to grow, safety becomes increasingly important. Control barrier functions (CBFs) provide a theoretically grounded framework for ensuring safety, but existing design methods often face limitations in effectiveness, scalability, or interpretability, and may result in overly conservative safe sets. In this paper, we propose \emph{VertexCBF}, a framework for learning neural CBFs in a scalable, systematic, and explainable way. We approximate the stationary Hamilton--Jacobi value function using a neural network trained via a combination of physics-informed and sparsely supervised learning. By exploiting control-affine dynamics and a convex polytope control set, under which the Hamiltonian is maximized at the control vertices, we efficiently generate supervision points via GPU-parallel vertex-restricted tree search, while a residual architecture guarantees that the learned CBF is never larger than 
    
[^234]: VERPO：验证证据正则化策略优化

    VERPO: Verified Evidence Regularized Policy Optimization

    [https://arxiv.org/abs/2609.06100](https://arxiv.org/abs/2609.06100)

    VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。

    

    可验证的结果奖励可以指导语言模型的后训练，但序列级别的优势无法识别哪些token级别的决策应当被保留或修改。证据条件教师通过以特权反馈重放采样轨迹来提供更密集的监督。然而，不加区分的模仿可能会迁移那些不支持任务成功的格式或推理风格偏移。我们提出了VERPO，一个验证证据正则化策略优化框架，它将证据视为策略修正的提议，同时保留结果目标。该框架将无证据的参考恢复与带符号的token级证据修正分离开来。Fisher证据对比沿着估计的证据存在方向对修正进行衰减。一个带停止机制的逐token ZPD控制器根据局部奖励对齐程度和Fisher移动成本来调节修正的接受度，而参考通道保持独立于接受决策。

    arXiv:2609.06100v1 Announce Type: cross  Abstract: Verifiable outcome rewards guide language-model post-training, but sequence-level advantages do not identify which token-level decisions should be preserved or revised. Evidence-conditioned Teachers provide denser supervision by replaying sampled trajectories with privileged feedback. Yet indiscriminate imitation risks transferring formatting or reasoning-style shifts that do not support task success. We introduce VERPO, a Verified Evidence Regularized Policy Optimization framework that treats evidence as a proposal for policy correction while retaining the outcome objective. It separates evidence-free reference restoration from signed token-level evidence corrections. Fisher Evidence Contrast attenuates corrections along an estimated evidence-presence direction. A stopped token-wise ZPD controller scales acceptance according to local reward alignment and Fisher movement cost, while the reference channel remains independent of acceptan
    
[^235]: PhenoBench：深度表型人类队列能告诉我们什么

    PhenoBench: Mapping What a Deeply Phenotyped Human Cohort Can Tell Us

    [https://arxiv.org/abs/2609.06080](https://arxiv.org/abs/2609.06080)

    该论文提出了PhenoBench——一个基于人类表型项目（超13,000名参与者）构建的可执行评估基准，通过定义15个领域、26种输入模态下的90项临床任务，系统性地量化了哪些测量数据对哪些健康问题具有预测价值。

    

    深度表型队列结合了从秒级到年际跨时间尺度的临床、影像、分子和可穿戴设备观测数据。这种广度可以揭示哪些测量数据能为哪些健康相关问题提供信息，但异构的分析结果无法直接比较。我们提出了PhenoBench，一个围绕人类表型项目构建的可执行基准，该项目已有超过13,000名参与者完成了首次访问。每个问题都固定了目标、合格人群、时间点和允许使用的信息；其评估契约规定了数据划分、评估指标、基线和结论边界。该基准定义了90项基于临床的任务，涵盖15个领域和26种输入模态。测量结果显示出依赖于具体问题和数据表示的预测价值，包括相对于匹配基线在留出集性能上的正向、接近零以及负向的变化。我们使用PhenoBench评估了新兴的表格基础模型……

    arXiv:2609.06080v1 Announce Type: cross  Abstract: Deeply phenotyped cohorts combine clinical, imaging, molecular, and wearable observations across timescales from seconds to years. This breadth can reveal which measurements inform which health-related questions, but heterogeneous analyses are not directly comparable. We present PhenoBench, an executable benchmark built around the Human Phenotype Project, in which more than 13,000 participants have completed the initial visit. Each question fixes the target, eligible population, timing, and allowed information; its evaluation contract specifies the split, metric, baseline, and claim boundary. The benchmark defines 90 clinically grounded tasks across 15 domains and 26 input modalities. Measurements showed question- and representation-dependent predictive value, including positive, near-zero, and negative changes in held-out performance relative to matched baselines. We used PhenoBench to evaluate emerging tabular foundation models acros
    
[^236]: 退化扩散模型的条件化

    Conditioning Degenerate Diffusion Models

    [https://arxiv.org/abs/2609.04090](https://arxiv.org/abs/2609.04090)

    该论文提出利用因果最优传输为扩散系数退化（奇异）的扩散生成模型构造近似损失函数，在条件密度不存在或不光滑的极弱假设下确定用于引导的最小熵控制。

    

    当前受条件约束的生成模型在训练过程中严重依赖得分函数来进行引导。当生成模型是一个具有奇异扩散系数的扩散过程，且其底层的（条件）密度要么不存在要么不光滑时，我们利用因果最优传输，在极小的假设条件下定义了近似损失函数，用以确定用于引导的最小熵控制。我们的方法依赖于因果最优传输，以及通过（受条件约束的）扩散过程的可预测表示性质对其进行的刻画，其中这些扩散过程的相关鞅问题是适定的，遵循Üstünel的框架。

    arXiv:2609.04090v1 Announce Type: new  Abstract: Current conditioned generative models heavily rely on score functions for guidance during training. When the generative model is a diffusion process with a singular diffusion coefficient and the underlying (conditional) densities either do not exist or are not smooth, we use causal optimal transport to define \emph{approximate} loss functions that identify a minimum-entropy control for guidance under minimal assumptions. Our approach relies on causal optimal transport and its characterization through the predictable representation property of (conditioned) diffusion processes whose associated martingale problem is well posed, \`a la \"Ust\"unel.
    
[^237]: 基于无限维连续归一化流学习信息先验的贝叶斯逆问题方法

    Learning Informative Prior with Infinite-Dimensional Continuous Normalizing Flow for Bayesian Inverse Problem

    [https://arxiv.org/abs/2609.03343](https://arxiv.org/abs/2609.03343)

    该论文提出了一种基于无限维连续归一化流的新方法，通过在希尔伯特空间中引入定义良好的神经常微分方程将简单参考测度变换为编码先验信息的复杂测度，建立了无限维贝叶斯先验的适定性理论框架，并提供了先验训练方法与后验采样算法，用于求解偏微分方程的贝叶斯逆问题。

    

    本文研究了具有无限维希尔伯特空间中模型参数的偏微分方程逆问题的无限维贝叶斯推断。为了有效融入先验信息，我们提出了一种新颖的基于连续归一化流的无限维模型。具体而言，通过在无限维空间中引入一个定义良好的神经常微分方程，可以将简单的参考测度变换为编码先验信息的更复杂测度。我们建立了相应的理论框架，以确保所提出的贝叶斯先验在无限维空间中的适定性。我们还针对两种不同的数据设置提供了先验的训练方法，并为由此得到的贝叶斯后验提供了两种采样算法。所提出的框架被应用于三个代表性的逆问题：简单的光滑逆问题、逆散射问题等。

    arXiv:2609.03343v1 Announce Type: cross  Abstract: This paper addresses infinite-dimensional Bayesian inference for inverse problem of partial differential equations with model parameters in infinite-dimensional Hilbert space. To effectively incorporate prior information, we propose a novel continuous normalizing flows based infinite-dimensional model. Specifically, by introducing a well-defined neural ordinary differential equation in infinite-dimensional space, a simple reference measure can be transformed into a more complex measure which encodes the prior information. A corresponding theoretical framework is established to ensure the well-posedness of our proposed Bayesian prior in infinite-dimensional space. We also provide training methods of the prior for two distinct data settings, along with two sampling algorithms for the resulting Bayesian posterior. The proposed framework is applied to three representative inverse problems: the simple smooth inverse problem, inverse scatter
    
[^238]: RideSkill：一种基于大语言模型驱动自动进化的泛化拼车分层算法

    RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution

    [https://arxiv.org/abs/2609.02250](https://arxiv.org/abs/2609.02250)

    该论文提出RideSkill，一种由大语言模型驱动自动进化的分层算法，用于解决泛化拼车问题，克服了传统多智能体强化学习方法在泛化性、可迁移性和大规模训练方面的局限。

    

    拼车允许具有不同起讫点（OD对）的多名乘客共享同一辆车辆，是一个具有挑战性的运营问题，因为它需要在不确定且多变的情况下，高效地将不同OD对的订单捆绑并分配给车辆。尽管多智能体强化学习（MARL）解决方案已取得了有前景的性能，但它们存在泛化能力有限（难以适应不同的环境场景）、可迁移性低（难以适应不同的平台目标）以及在大规模系统中训练困难（如维度灾难）等问题。最近，受大语言模型（LLM）规模化发展的启发，一些工作将LLM引入网约车系统，要么直接将LLM用作决策智能体，要么利用LLM进行自动算法设计。然而，这些方法均不支持车辆共享，这使问题变得更加复杂。

    arXiv:2609.02250v1 Announce Type: cross  Abstract: Ride-sharing, which allows multiple passengers with different origin-destination (OD) pairs to share a single vehicle, is a challenging operational problem, as it requires orders with different OD pairs to be efficiently bundled and assigned to vehicles under uncertain and varying scenarios. Although multi-agent reinforcement learning (MARL) solutions have achieved promising performance, they suffer from limited generalization (adapting to different environmental scenarios), low transferability (adapting to different platform objectives), and training difficulties in large-scale systems, such as the curse of dimensionality. Recently, motivated by the scaling of large language models (LLMs), several works have incorporated LLMs into ride-hailing systems, either by employing LLMs directly as decision-making agents or using them for automatic algorithm design. However, none of these approaches support vehicle sharing, which complicates th
    
[^239]: 记忆并非总是必需：科学推理中条件记忆的特征化研究

    Memory Is Not Always Needed: Characterizing Conditional Memory in Scientific Reasoning

    [https://arxiv.org/abs/2608.23982](https://arxiv.org/abs/2608.23982)

    本文系统研究了科学推理中条件记忆的适用条件，提出知识边界感知路由器，根据输入代理动态决定是否及如何激活记忆，以避免干扰并提升推理准确性。

    

    科学推理要求语言模型检索专业知识，并将其可靠地整合到多步计算中。条件记忆提供了一条显式查找路径，补充了稠密神经表示，但其有用性本质上依赖于输入和计算：检索到的信息可能修复缺失的科学关联，但也可能引入分散注意力的捷径，或干扰基础模型本可正确执行的推理。在本工作中，我们系统地研究了条件记忆应在何时、何处以及何种程度上参与科学推理。我们刻画了科学知识边界，并对启用记忆的知识电路节点进行了受控干预。基于这些分析，我们提出了一种知识边界感知路由器，该路由器利用生成前可用的任务特定输入代理来判断是否激活记忆，以及激活哪些层。

    arXiv:2608.23982v1 Announce Type: new  Abstract: Scientific reasoning requires language models to retrieve specialized knowledge and incorporate it reliably into multi-step computation. Conditional memory provides an explicit lookup pathway that complements dense neural representations, but its usefulness is inherently input- and computation-dependent: retrieved information may repair missing scientific associations, yet it may also introduce distracting shortcuts or interfere with reasoning that the base model can already perform correctly. In this work, we systematically investigate when, where, and to what extent conditional memory should participate in scientific reasoning. We characterize the scientific knowledge boundary and controlled interventions on memory-enabled knowledge-circuit nodes. Based on these analyses, we propose a Knowledge Boundary-Aware Router that uses task-specific input proxies available before generation to determine whether memory is activated, which layer-s
    
[^240]: 训练留痕：用于语言模型血统验证的中心化残差签名

    Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification

    [https://arxiv.org/abs/2608.14929](https://arxiv.org/abs/2608.14929)

    本文提出一种基于中心化残差签名的无数据白盒方法，通过移除身份对齐组件并比较残差块特有结构，实现语言模型血统的可靠验证，在多种后代类型中达到完美区分性能，且对功能保持清洗具有鲁棒性。

    

    arXiv:2608.14929v1 公告类型：新 摘要：开放权重语言模型经常被微调、量化、剪枝和合并，但其来源往往没有文档记录。我们研究无数据白盒血统验证：仅凭权重能否揭示两个兼容模型检查点是否共享祖先？残差训练会在分支产物中产生共享的身份对齐组件，因此仅凭该结构无法确立血统。我们移除这一组件，并比较跨残差块的检查点特有结构，生成一个针对独立检查点校准的对称血统分数。在残差MLP和GPT-2基准测试上，该分数能将微调、LoRA合并、剪枝和量化后代与独立及蒸馏模型区分开来（AUROC=1.0），从而区分权重血统与行为相似性。在功能保持的检查点清洗实验中，权重空间基线失去裕度或失败；我们的分数保持不变，且运行速度比最接近的稳健基线快76倍。

    arXiv:2608.14929v1 Announce Type: new  Abstract: Open-weight language models are fine-tuned, quantized, pruned, and merged, yet their provenance is often undocumented. We study data-free white-box lineage verification: can weights alone reveal whether two compatible model checkpoints share ancestry?   Residual training produces a shared identity-aligned component in branch products, so this structure alone cannot establish ancestry. We remove it and compare checkpoint-specific structure across residual blocks, yielding a symmetric lineage score calibrated against independent checkpoints. On residual-MLP and GPT-2 benchmarks, the score separates fine-tuned, LoRA-merged, pruned, and quantized descendants from independent and distilled models (AUROC=1.0), distinguishing weight ancestry from behavioral similarity. Under function-preserving checkpoint laundering experiments, weight-space baselines lose margin or fail; our score remains unchanged and runs 76x faster than the nearest robust b
    
[^241]: 一种无参数的小样本评估方法用于大象叫声分类

    A Parameter-Free Few-Shot Evaluation for Elephant Vocalisation Classification

    [https://arxiv.org/abs/2608.14824](https://arxiv.org/abs/2608.14824)

    本文提出了一种无参数的最近质心分类评估方法，用于在大象叫声分类中比较不同预训练嵌入的性能，无需额外训练即可进行小样本评估。

    

    arXiv:2608.14824v1 公告类型：交叉 摘要：我们在固定预训练声学嵌入上，对大象叫声的最近质心分类进行了无参数的情景评估，数据集包括Elephant Voices (EV)和Linguistic Data Consortium (LDC)。我们不是问哪种嵌入在利用所有可用标记数据训练时能产生最佳分类器，而是问当每个类别的标记样本数量变化时，最简单的分类器表现如何。每个类别由其支持集嵌入的均值表示，每个查询在平方欧氏距离下被分配到最近的质心。我们在Perch（版本1）、Perch（版本2）和HuBERT（基础，第2层）嵌入以及梅尔频率倒谱系数（MFCC）特征上，以N-way k-shot方式，在与训练基线相同的交叉验证协议下评估该质心分类器。对100个重采样支持集的引导法量化了采样噪声。在较小、低...

    arXiv:2608.14824v1 Announce Type: cross  Abstract: We present a parameter-free episodic evaluation of nearest-centroid classification for elephant vocalisations on fixed pretrained acoustic embeddings, across the Elephant Voices (EV) and Linguistic Data Consortium (LDC) datasets. Rather than asking which embedding yields the best classifier when trained on all available labelled data, we ask how the simplest classifier performs as labelled exemplars per class are varied. Each class is represented by the mean of its support-set embeddings, and each query is assigned to the nearest centroid under squared Euclidean distance. We evaluate this centroid classifier on the Perch (ver. 1), Perch (ver. 2), and HuBERT (base, layer 2) embeddings, together with mel frequency cepstral coefficient (MFCC) features, in an N-way k-shot manner under the same cross-validation protocol as the trained baselines. A bootstrap over 100 resampled support sets quantifies the sampling noise. On the smaller, low-r
    
[^242]: 统一物理反向传播

    Unifying Physical Backpropagation

    [https://arxiv.org/abs/2608.11585](https://arxiv.org/abs/2608.11585)

    本文提出了一种基于伴随方法的统一理论，确定了物理系统能在同一硬件上生成精确梯度所需的条件，区分了线性和非线性系统的不同要求。

    

    arXiv:2608.11585v1 公告类型：交叉  摘要：物理计算系统利用设备动态进行计算，但其基于梯度的优化具有挑战性：通过数字孪生进行反向传播会遭遇模型与现实的差距。设备上的梯度计算可以解决这一问题，已有少数理论和实验研究提出了实现方法。然而，目前缺乏一种统一理论来确定物理系统何时能计算其自身性能的梯度。在此，我们基于伴随方法发展了这样一种统一理论：我们识别了充分条件，在这些条件下，形式上精确梯度所需的伴随场可以在执行计算的同一硬件上生成。线性和非线性系统遵循根本不同的条件：对于线性系统，只要保持互易性，阻尼或增益是可接受的。对于非线性轨迹系统，充分条件是线性化系统的互易性。

    arXiv:2608.11585v1 Announce Type: cross  Abstract: Physical computing systems exploit device dynamics for computation, but their gradient-based optimization is challenging: backpropagation through a digital twin suffers from model-reality gap. On-device gradient computation could resolve this issue, and a handful of theoretical and experimental studies have proposed ways to achieve it. Yet a unifying theory identifying when a physical system can compute the gradient of its own performance has been missing. Here we develop such a unification, based on the adjoint method: we identify sufficient conditions under which the adjoint field required for formally exact gradients can be generated on the same hardware that performs the computation. Linear and nonlinear systems obey fundamentally different conditions: for linear systems damping or gain is admissible provided reciprocity is preserved. For nonlinear trajectory systems the sufficient conditions are reciprocity of the linearized syste
    
[^243]: 面向分布式CNC刀具磨损预测的联邦学习

    Federated Learning for Distributed CNC Tool Wear Prediction

    [https://arxiv.org/abs/2608.11281](https://arxiv.org/abs/2608.11281)

    本文提出将联邦学习应用于分布式CNC刀具磨损预测，在不共享原始数据的情况下实现接近集中式学习的性能，并显著优于本地模型。

    

    刀具磨损预测是数控加工中的一项重要任务，其中对刀具状态的准确监测有助于保障产品质量和工艺可靠性。机器学习方法在该任务中展现出潜力，但其在工业环境中的应用受到加工数据分布式特性以及机器、站点或组织间数据共享限制的制约。联邦学习通过在不传输原始运行数据的情况下实现协作模型训练，为这一场景提供了合适的框架。本文研究了联邦学习在CNC刀具磨损预测中的应用。刀具轨迹被分配到模拟客户端以表示联邦学习场景，并将联邦模型与集中式参考模型及本地客户端基线进行比较。结果表明，联邦学习的性能接近集中式学习，并显著优于本地客户端模型。

    arXiv:2608.11281v1 Announce Type: cross  Abstract: Tool wear prediction is an important task in CNC machining, where accurate monitoring of tool condition supports product quality and process reliability. Machine learning methods have shown potential for this task, but their use in industrial environments is limited by the distributed nature of machining data and by restrictions on data sharing between machines, sites, or organizations. Federated learning offers a suitable framework for this setting by enabling collaborative model training without transferring raw operational data. This paper investigates federated learning for CNC tool wear prediction. Tool trajectories are distributed across simulated clients to represent a federated learning scenario. The federated models are compared against centralized references and local client baselines. Results show that federated learning achieves performance close to centralized learning and improves significantly over local client models. T
    
[^244]: 面向INT2 KV缓存量化的输出感知旋转方法

    Output-Aware Rotation for INT2 KV-Cache Quantization

    [https://arxiv.org/abs/2608.02691](https://arxiv.org/abs/2608.02691)

    本文提出输出感知旋转方法OptR，通过最小化输出投影 $W_O$ 之后的注意力输出误差、将误差分解为键和值引起的项并学习逐头正交校正，同时利用注意力等价的键重参数化降低通道偏移，从而实现更优的INT2 KV缓存量化。

    

    键值缓存已成为长上下文大语言模型推理中的主要内存和带宽瓶颈，使得超低比特量化日益重要。然而，现有的基于旋转的INT2方法在完整注意力读出之前优化缓存统计量或代理误差，而模型最终实际受到的是通过注意力机制和输出投影 $W_O$ 传播的误差的影响。为了解决这种不匹配问题，我们提出了 OptR，一种输出感知的旋转方法，它最小化经过 $W_O$ 之后的注意力输出误差。OptR 将 $W_O$ 之后的注意力输出误差分解为键和值引起的两个部分，并通过完整的INT2量化和注意力路径学习每个注意力头的正交校正。OptR 还进一步应用了一种注意力等价的键重参数化方法，在不改变softmax分布的前提下减少较大的逐通道偏移。在三个模型和五个推理与编码任务上的实验（摘要在此处截断）。

    arXiv:2608.02691v3 Announce Type: replace  Abstract: The key-value (KV) cache has become a major memory and bandwidth bottleneck in long-context large language model inference, making ultra-low-bit quantization increasingly important. However, existing rotation-based INT2 methods optimize cache statistics or proxy errors before the complete attention readout, even though the model is ultimately affected by the error propagated through attention and the output projection $W_O$. To address this mismatch, we propose \textit{OptR}, an output-aware rotation method that minimizes post-$W_O$ attention-output error. OptR decomposes the post-$W_O$ attention-output error into key- and value-induced terms and learns per-head orthogonal corrections through the full INT2 quantization and attention path. OptR further applies an attention-equivalent key reparameterization to reduce large channel-wise offsets without changing the softmax distribution. Across three models and five reasoning and coding 
    
[^245]: Molt：一个面向智能体强化学习的可扩展 PyTorch 原生训练框架

    Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2607.21653](https://arxiv.org/abs/2607.21653)

    提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。

    

    智能体强化学习需要一种研究者能够在不牺牲模型规模或智能体执行控制权的前提下进行修改的基础设施。我们提出了 Molt，一个轻量级的 PyTorch 原生框架，将万亿参数规模的训练与标准智能体接口相结合。Molt 整合了四项能力：基于可组合模型并行的紧凑训练实现；统一的 OpenAI 和 Anthropic 接口，支持上下文压缩后的自动轨迹分段；完全异步的 rollout 与优化；以及面向长多模态轨迹的分布式经验存储。现有智能体可以保留其执行与上下文管理逻辑，同时由共享捕获层记录生成的 token 和行为概率。Rollout 工作进程将大体积的经验数据放入 Ray 的对象存储中，训练器进程通过引用检索分配给它的经验，从而避免对完整 rollout 进行集中式收集。

    arXiv:2607.21653v2 Announce Type: replace-cross  Abstract: Agentic reinforcement learning requires infrastructure that researchers can modify without sacrificing model scale or control over agent execution. We present Molt, a lightweight PyTorch-native framework that combines trillion-parameter training with standard agent interfaces. Molt integrates four capabilities: a compact training implementation built on composable model parallelism; unified OpenAI and Anthropic interfaces with automatic trajectory segmentation after context compaction; fully asynchronous rollout and optimization; and distributed experience storage for long, multimodal trajectories. Existing agents retain their execution and context-management logic while a shared capture layer records generated tokens and behavior probabilities. Rollout workers place heavy experience payloads in Ray's object store, and trainer ranks retrieve their assigned experiences by reference, avoiding a centralized gather of the full roll
    
[^246]: 利用混合机器学习方法预测水电解质溶液中的活度

    Predicting Activities in Aqueous Electrolyte Solutions with Hybrid Machine Learning

    [https://arxiv.org/abs/2607.19114](https://arxiv.org/abs/2607.19114)

    本文提出一种将基于物理的Bromley模型与机器学习矩阵补全方法相结合的混合模型，能够无需逐一拟合实验数据即可预测水电解质溶液的活度，突破了传统活度模型无法预测未研究体系的局限。

    

    水电解质溶液中的活度通常由离子活度系数和渗透系数来描述，是模拟工业和自然界中许多过程的重要性质。现有的活度模型（如Pitzer模型或Bromley模型）需要针对每种目标电解质对实验数据进行拟合，因此无法预测未经研究的体系的性质。虽然存在一些预测性方法，但它们通常适用范围有限，且依赖于额外的离子特异性描述符。在本工作中，我们提出了一种新的混合模型，将基于物理的Bromley模型与机器学习中的矩阵补全方法（MCM）相结合。该矩阵补全方法用于预测Bromley模型的电解质特异性参数，其利用了这些参数可以排列成以阳离子和阴离子分别为行与列的矩阵这一事实。由于许多电解质缺乏实验数据，初始参数矩阵……

    arXiv:2607.19114v2 Announce Type: replace  Abstract: Activities in aqueous electrolyte solutions, usually described by ionic activity and osmotic coefficients, are important properties for modeling many processes in industry and nature. Established activity models, such as those of Pitzer or Bromley, require fitting to experimental data for each electrolyte of interest and thus cannot predict properties for unstudied systems. While some predictive approaches exist, they are typically limited in scope and rely on additional ion-specific descriptors. In this work, we introduce a new hybrid model that combines the physics-based Bromley model with a matrix completion method (MCM) from machine learning. The MCM is employed to predict the electrolyte-specific parameters of the Bromley model, exploiting the fact that these parameters can be arranged in a matrix with cations and anions as rows and columns, respectively. Due to the lack of experimental data for many electrolytes, the initial pa
    
[^247]: 力反馈永不嫌迟：利用反应式力注入加速VLA后训练

    Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection

    [https://arxiv.org/abs/2607.14236](https://arxiv.org/abs/2607.14236)

    LIFT是一种力感知的后训练框架，通过在预训练VLA策略旁嫁接反应式动作专家，并借助因果力记忆和零初始化交叉注意力注入6D末端执行器力，为模型增加接触反应能力的同时保留其通用操作知识，从而加速VLA后训练。

    

    预训练的视觉-语言-动作（VLA）策略提供了强大的语言条件化操作知识，但它们在很大程度上仍由视觉驱动，一旦操作进入接触状态就会遇到困难——例如场景被遮挡、深度信息模糊，或微小的力误差使执行偏离离线演示分布。我们提出了LIFT（面向VLA后训练的后期反应式力注入），这是一个力感知的后训练框架，能在保留预训练VLA策略通用操作知识的同时，为其增加接触反应能力。LIFT在原始动作专家旁边嫁接一个反应式动作专家，使用预训练的动作权重对其进行初始化，并通过因果力记忆和零初始化的交叉注意力注入最近的6D末端执行器力，使动作能够在执行过程中被实时刷新。为解决接触反馈的策略相关分布偏移问题，LIFT进一步将反应式力注入与适配机制相耦合，从而加速VLA在接触丰富任务上的后训练过程。

    arXiv:2607.14236v2 Announce Type: replace-cross  Abstract: Pretrained vision-language-action (VLA) policies provide strong language-conditioned manipulation knowledge, but they remain largely vision-driven and can struggle once manipulation enters contact states where the scene is occluded, depth is ambiguous, or small force errors push execution off the offline demonstration distribution. We present LIFT (Late Reactive Injection of Force for VLA Post-Training), a force-aware post-training framework that adds contact reactivity to a pretrained VLA policy while preserving its general manipulation knowledge. LIFT grafts a reactive action expert beside the original action expert, initializes it from pretrained action weights, and injects recent 6D end-effector force through causal force memory and zero-initialized cross attention, enabling actions to be refreshed during execution. To address the policy-dependent distribution shift of contact feedback, LIFT further couples reactive force i
    
[^248]: tidyHEBO：具备模型一致性变换与帕累托搜索的鲁棒通用贝叶斯优化

    tidyHEBO: Robust General-Purpose Bayesian Optimization with Model-Consistent Warping and Pareto Search

    [https://arxiv.org/abs/2607.10669](https://arxiv.org/abs/2607.10669)

    tidyHEBO 是一个 BoTorch 原生的通用贝叶斯优化器，通过将 Yeo-Johnson 输出变换与高斯过程代理模型联合拟合、在原始目标尺度上评估采集函数并执行多准则累积帕累托搜索，在无需任何基准特定调优的情况下于 Olympus 基准上排名第一。

    

    贝叶斯优化（BO）被广泛用于昂贵的黑盒优化问题，然而其实际性能不仅取决于高层的算法选择，还取决于代理模型训练、输入输出变换、采集函数以及候选点搜索等环节的具体实现方式。我们提出了 tidyHEBO，一个基于 BoTorch 原生实现的单目标优化器，专为鲁棒的通用优化而设计。tidyHEBO 将 Yeo-Johnson 输出变换与高斯过程代理模型进行联合拟合，使用确定性求积或蒙特卡洛采样在原始目标尺度上评估采集函数，并对多个采集准则执行受约束的累积帕累托搜索。在没有任何针对 Olympus 基准的超参数调优——仅使用默认优化器配置的情况下——tidyHEBO 在 Olympus 基准的受评估方法中排名第一。它在典型性能上取得了最佳平均排名（平均排名 1.53...）

    arXiv:2607.10669v2 Announce Type: replace  Abstract: Bayesian optimization (BO) is widely used for expensive black-box problems, yet practical performance depends not only on high-level algorithmic choices but also on how surrogate model training, input and output warping transformations, acquisition functions, and candidate search are implemented. We present tidyHEBO, a BoTorch-native single-objective optimizer designed for robust general-purpose optimization. tidyHEBO jointly fits Yeo-Johnson output warping with the Gaussian-process surrogate, evaluates acquisition functions on the original objective scale using deterministic quadrature or MC-samples, and performs constrained cumulative Pareto search over multiple acquisition criteria. Without any Olympus-specific hyperparameter tuning - using only default optimizer configurations - tidyHEBO ranked first among the evaluated methods on the Olympus benchmark. It achieved the best average ranks for typical performance (average rank 1.53
    
[^249]: 能量引导递归模型

    Energy-guided Recursive Model

    [https://arxiv.org/abs/2607.10128](https://arxiv.org/abs/2607.10128)

    提出能量引导递归模型（ERM），利用Hopfield型记忆为候选轨迹分配内在能量，从而有原则地指导轨迹选择与递归深度的确定，在数独、铅笔谜题和迷宫等推理任务上取得递归建模领域的最佳准确率，并降低了语言建模的困惑度。

    

    递归模型在推理和语言任务上展现出巨大潜力，但其在测试时的扩展缺乏一种有原则性的准则来选择轨迹或确定递归深度。我们提出了能量引导递归模型，该模型利用Hopfield型记忆存储有效的局部和全局结构，为候选轨迹分配内在能量。这些能量可以指导候选轨迹的选择，并提示递归深度的有效范围，这意味着更深的递归并不一定能提高推理准确率。这些能量还使得并行回火等采样方法能够改善探索效果。在推理任务上，ERM在数独（98.97%）、铅笔谜题基准（Pencil Puzzle Bench，PPBench，88.04%）和迷宫（99.30%）上达到了最优解，取得了递归建模中的最佳准确率。在语言建模方面，ERM以极小的推理开销将RedPajama-V2的困惑度降低了1.74%。这些结果支持了能量引导的方法……

    arXiv:2607.10128v3 Announce Type: replace  Abstract: Recursive models show promise on reasoning and language tasks, yet their test-time scaling lacks a principled criterion for selecting trajectories or determining recurrent depth. We introduce \textbf{Energy-guided Recursive Model (ERM)}, which uses Hopfield-type memories of valid local and global structures to assign intrinsic energies to candidate trajectories. These energies guide candidate selection and suggest an effective range of recurrent depths, implying that deeper recurrence does not necessarily improve reasoning accuracy. They also enable sampling methods such as parallel tempering to improve exploration. For reasoning tasks, ERM achieves optimal solutions on Sudoku ($98.97\%$), Pencil Puzzle Bench (PPBench, $88.04\%$) and Maze ($99.30\%$), reaching the best accuracy in recursive modeling. On language modeling, ERM reduces RedPajama-V2 perplexity by $1.74\%$ with marginal inference overhead. The results support energy guid
    
[^250]: 基于量规的前沿语言模型在专家撰写的临床推理任务上的受控比较

    A rubric-based controlled comparison of frontier language models on expert-authored clinical reasoning tasks

    [https://arxiv.org/abs/2607.02175](https://arxiv.org/abs/2607.02175)

    该研究构建了一个由临床医生撰写的高难度临床推理评估数据集及加权量规，发现前沿大模型在关键临床标准上的通过率（32.4-41.7%）远低于低风险标准（80-90%），揭示了模型能力与临床优先级之间的倒置现象。

    

    多项选择题式的医学基准测试已日益饱和，而近期基于量规的评估（如HealthBench）表明，开放式临床性能远未得到解决——其“困难”子集的最高得分仍停留在32%。我们提出了一个小型但刻意设计的高难度评估数据集，包含五个由临床医生撰写的临床场景，涵盖四个专科（麻醉学、内科/家庭医学、急诊医学和产科），每个场景均配有基于临床医生起草的标准答案编写的原子化、加权、相互独立且完全穷尽（MECE）的量规（每个任务25-62条标准，共184条标准）。我们评估了三个前沿模型：GPT 5.4、Claude Opus 4.7和Gemini 3.1 Pro。平均量规通过率分别为0.47（Claude）、0.38（GPT）和0.37（Gemini）。核心发现是临床优先级的倒置：最高权重（权重5，关键级）标准的通过率仅为32.4-41.7%，而低风险的权重1标准通过率却高达80-90%。108条标准中有55条……

    arXiv:2607.02175v2 Announce Type: replace  Abstract: Multiple-choice medical benchmarks are increasingly saturated, and recent rubric-based evaluations such as HealthBench have shown that open-ended clinical performance is far from solved - its "Hard" subset top score remains 32%. We present a small, deliberately difficult evaluation dataset of five clinician-authored clinical scenarios spanning four specialties (anaesthesia, internal/family medicine, emergency medicine, and obstetrics), each accompanied by an atomic, weighted, MECE rubric (25-62 criteria per task; 184 criteria total) authored from a clinician-drafted golden answer. We evaluate three frontier models: GPT 5.4, Claude Opus 4.7, and Gemini 3.1 Pro. Mean rubric pass rates were 0.47 (Claude), 0.38 (GPT), and 0.37 (Gemini). The central finding is an inversion of clinical priority: the highest-weighted (weight-5, critical) criteria passed at only 32.4-41.7%, while low-stakes weight-1 criteria passed at 80-90%. 55 of 108 criti
    
[^251]: 条件性协同消融：恢复Transformer电路中的自修复备份组件

    Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits

    [https://arxiv.org/abs/2607.01940](https://arxiv.org/abs/2607.01940)

    提出条件性协同消融方法CoAx，通过测量主要组件集合被移除后消融效应的增长，来识别Transformer电路中被自修复机制掩盖的休眠备份组件，解决了电路解释在干预下不完整的问题。

    

    机制可解释性旨在通过“电路”来解释Transformer的行为：电路是一组因果性地支持某种行为的内部组件。然而，自修复机制造成了一个盲区：消融一个主要组件可能会激活一个休眠的备份组件，因此在完整模型中能够解释行为的电路，在用于测试它的干预之下可能变得不完整。我们将这一差距形式化为“条件性电路补全”问题：给定一个主要组件集合，识别在其被移除后变得因果上重要的组件。我们提出了条件性协同消融，该方法根据主要集合被移除后消融效应的增长幅度来对候选组件进行排序。我们证明，一个完全休眠的备份组件对于基于单元的完整状态评分而言可能与无关组件无法区分，而其条件效应变化恰好聚合了将其与被移除集合联系起来的所有交互阶数。在GPT-2-small的间接宾语识别（IOI）电路上，CoAx（摘要在此处截断）

    arXiv:2607.01940v2 Announce Type: replace  Abstract: Mechanistic interpretability seeks to explain transformer behavior through circuits: sets of internal components that causally support a behavior. However, self-repair creates a blind spot: ablating a primary component can activate a dormant backup, so a circuit that explains behavior in the intact model can become incomplete under the intervention used to test it. We formulate this gap as conditional circuit completion: given a primary set, identify components that become causally important after its removal. We introduce conditional co-ablation (CoAx), which ranks candidates by growth in ablation effect after primary-set removal. We show that a perfectly dormant backup can be indistinguishable from an irrelevant component to per-unit intact-state scores, whereas its conditional effect change exactly aggregates all interaction orders linking it to the removed set. On GPT-2-small's Indirect Object Identification (IOI) circuit, CoAx r
    
[^252]: 一个用于触觉传感器公平测试与比较的3D可打印数据集

    A 3D-Printable Dataset for Fair Testing and Comparisons of Tactile Sensors

    [https://arxiv.org/abs/2606.25886](https://arxiv.org/abs/2606.25886)

    本文提出了一个由数学定义的、可在不同打印机和耗材上可靠复制的3D可打印纹理数据集，为触觉传感器的公平测试与比较提供了标准化基准。

    

    现有的触觉感知纹理数据集主要由特定传感器与可用表面/物体交互时产生的传感器读数组成，而非描述纹理本身，这限制了触觉传感器之间的公平比较，也阻碍了可复现的研究。在本工作中，我们引入了一个由数学定义纹理构成的3D可打印数据集，该数据集旨在能够在不同打印机和耗材类型上可靠地制造。该数据集由六个基于正弦波和傅里叶函数组合参数化生成的表面图案组成，在空间频率、幅度和方向结构上提供了可控的变化。我们通过在受控接触条件下使用光学TacTip传感器采集图像并测量其方差，评估了这些纹理在三种流行3D打印机和多种耗材类型上的可复现性。我们的结果表明，打印质量，尤其是……

    arXiv:2606.25886v2 Announce Type: replace-cross  Abstract: Existing texture datasets for tactile sensing primarily consist of sensor readings from a specific sensor interacting with available surfaces/objects rather than describing the textures themselves, limiting fair comparison between tactile sensors and hindering reproducible research. In this work, we introduce a 3D-printable dataset of mathematically defined textures designed to be fabricated reliably across different printers and filament types. The dataset consists of six parametrically generated surface patterns derived from combinations of sine-wave and Fourier-based functions, giving controlled variation in spatial frequency, amplitude, and directional structure. We evaluate the reproducibility of these textures across three popular 3D printers and multiple filament types by measuring variance in images captured using an optical TacTip sensor under controlled contact conditions. Our results show that print quality, particul
    
[^253]: 钱德拉-Gaia对应体目录：利用机器学习解决钱德拉源目录中X射线源与Gaia匹配的歧义问题

    The Chandra-Gaia Catalog of Counterparts: Resolving ambiguous Gaia matches to X-ray sources in the Chandra Source Catalog using Machine Learning

    [https://arxiv.org/abs/2606.19329](https://arxiv.org/abs/2606.19329)

    本文提出了一个结合贝叶斯交叉匹配框架NWAY与LightGBM梯度提升分类器的交叉匹配框架，利用星等、颜色、距离等源属性将钱德拉源目录中约25.4万个X射线源与Gaia DR3光学源匹配，成功识别约11.3万个真实对应体，并能有效排除基于纯空间位置方法无法区分的偶然重合。

    

    我们提出了一个将钱德拉源目录（CSC v2.1）中的源与Gaia第三次数据发布（Gaia DR3）中的光学源进行交叉匹配的框架。与纯粹基于空间位置的方法不同，我们利用星等、颜色和距离等源属性来识别真实的对应体、检测偶然重合，并在存在多个可能候选体时解决歧义。我们使用NWAY——一个考虑了位置误差和源密度的贝叶斯交叉匹配框架——来定义高置信度匹配的训练集，并在来自两个目录的多种特征上训练了梯度提升分类器。在约25.4万个独特的X射线源中，我们为约11.3万个源找到了对应体，其中约7千个源存在多个可能的对应体。有约2万个源在基于角距离的交叉匹配中能找到匹配，但我们的方法未找到其对应体，其中一半被归因于偶然重合。我们对该流程进行了验证，[摘要在此处截断]

    arXiv:2606.19329v2 Announce Type: replace-cross  Abstract: We present a framework to cross-match sources from the Chandra Source Catalog (CSC v2.1) with optical sources from Gaia Data Release 3. Unlike purely spatial approaches, we use source properties such as magnitudes, colors, and distances to identify true counterparts, detect chance coincidences, and resolve ambiguities when multiple plausible candidates exist. We define a training set of high-confidence matches using NWAY, a Bayesian cross-matching framework that accounts for positional errors and source densities. We train a gradient-boosted classifier (LightGBM) on a variety of features from both catalogs. Of the ~$254$k unique X-ray sources, we find counterparts for ~$113$k sources, of which plausible multiple counterparts are found for ~$7$k. We find no counterparts for ~$20$k sources for which separation-based cross-matching does find a match, and attribute half of these to chance coincidences. We validate the pipeline on t
    
[^254]: Starter-Iterator神经算子：面向高保真正演与反演偏微分方程问题的统一架构

    Starter-Iterator Neural Operator: A Unified Architecture for High-Fidelity Forward and Inverse PDE Problems

    [https://arxiv.org/abs/2606.18305](https://arxiv.org/abs/2606.18305)

    提出Starter-Iterator神经算子（SINO），将经典迭代求解器的初始化与残差修正结构融入神经算子学习，通过频域Starter模块捕捉全局谱特征并提供初始近似，为高保真的正演与反演PDE问题提供统一架构。

    

    算子学习是机器学习与科学计算交叉领域的一个新兴方向。通过学习函数空间之间的映射，神经算子为偏微分方程（PDE）族提供了数据驱动的代理模型。训练完成后，这些模型能够高效地评估解算子，使其适用于诸如实时预测和参数扫描等多查询应用。然而，对于复杂的正演和反演问题，保持高近似精度和稳定的长期预测仍然具有挑战性。为了解决这些挑战，我们提出了Starter-Iterator神经算子，它将经典迭代求解器的初始化和残差修正结构融入神经算子学习之中。频域的Starter模块捕捉主导的全局谱特征并提供有依据的初始近似，而潜在空间的It

    arXiv:2606.18305v2 Announce Type: replace-cross  Abstract: Operator learning is an emerging field at the intersection of machine learning and scientific computing. By learning mappings between function spaces, neural operators provide data-driven surrogate models for families of partial differential equations (PDEs). Once trained, these models can evaluate solution operators efficiently, making them suitable for many-query applications such as real-time prediction and parameter sweeps. However, maintaining high approximation accuracy and stable long-term predictions remains challenging for complex forward and inverse problems. To address these challenges, we propose the Starter-Iterator Neural Operator (SINO), which incorporates the initialization and residual-correction structures of classical iterative solvers into neural operator learning. The frequency-domain Starter captures dominant global spectral features and provides an informed initial approximation, while the latent-space It
    
[^255]: 面向多轮智能体的课程式轮级引导在线策略蒸馏

    On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents

    [https://arxiv.org/abs/2606.15912](https://arxiv.org/abs/2606.15912)

    提出Guided-OPD算法，通过在每次rollout中混合教师与学生生成的轮次，并按课程将教师干预概率逐渐衰减至零，解决了多轮智能体在线策略蒸馏中学生误差跨轮累积、教师监督在最需要时反而失效的问题。

    

    arXiv:2606.15912v2 公告类型：replace-cross 摘要：能够进行规划、调用工具并与环境交互的多轮智能体为解决复杂任务提供了一种有前景的范式，但其能力通常依赖于超大规模模型，而这些模型的推理成本在实践中难以承受。在线策略蒸馏是将此类能力迁移到更小学生模型上的一种自然方法，但我们发现它在这种设置下存在一种特有的失效模式：学生模型的小错误会跨轮次不断累积，使轨迹偏离教师模型熟悉的状态分布，导致教师模型的监督恰恰在学生最需要它的地方变得最不可靠。我们提出了引导式在线策略蒸馏，这是一种简单而有效的算法，它在每次rollout中混合教师生成与学生生成的轮次，并按照一个逐渐衰减至零的课程来调度教师模型的干预概率。强引导使早期轨迹保持在教师模型附近……

    arXiv:2606.15912v2 Announce Type: replace-cross  Abstract: Multi-turn agents that plan, invoke tools, and interact with environments offer a promising paradigm for solving complex tasks, yet their capabilities typically rely on very large models whose inference cost is prohibitive in practice. On-Policy Distillation (OPD) is a natural recipe for transferring such capabilities to smaller students, but we find that it suffers a characteristic failure mode in this setting: small student errors compound across turns and push the trajectory out of the teacher's familiar state distribution, so the teacher's supervision becomes least reliable precisely where the student needs it most. We propose Guided On-Policy Distillation (Guided-OPD), a simple yet effective algorithm that mixes teacher- and student-generated turns within each rollout and schedules the teacher's intervention probability along a curriculum that decays to zero. Strong guidance keeps early trajectories close to the teacher di
    
[^256]: 用于分层分类的同时潜在预算树

    Simultaneous Latent Budget Trees for Stratified Classification

    [https://arxiv.org/abs/2606.13295](https://arxiv.org/abs/2606.13295)

    本文提出同时潜在预算树，一种面向含时间、空间或人口等分层因素的场景的分类树概率机器学习框架，通过将子节点解释为同时混合模型的潜在成分来构建基于模型的条件分裂规则。

    

    在可解释人工智能时代，单棵决策树因其易于解释而重新受到关注。本文提出了同时潜在预算树，这是一种在存在分层因素（如时间、空间或人口统计学变量，其可作为控制变量或潜在混杂因素）情况下的分类树的概率机器学习框架。标准的树生长程序并非为优化条件分裂规则而设计。本文提出了一种基于模型的分裂规则，其中子节点被解释为拟合于父节点的同时混合模型（如同时潜在预算模型及其约束版本）的潜在成分。混合参数针对每个组以不同方式将观测值引导至子节点，而潜在预算参数则更新控制变量各水平下的响应类别轮廓。参数估计

    arXiv:2606.13295v3 Announce Type: replace-cross  Abstract: In the era of Explainable Artificial Intelligence, there is a renewed focus on single trees for their ease of interpretation. This paper introduces Simultaneous Latent Budget Trees, a probabilistic machine learning framework for classification trees in the presence of a stratification factor such as a temporal, spatial, or demographic variable, acting as a control variable or potential confounder. Standard tree growth procedures are not designed to optimize a conditional split rule. A model-based split rule is proposed in which child nodes are interpreted as latent components of a simultaneous mixture model, such as the Simultaneous Latent Budget Model and its constrained versions, fitted to the parent node. Mixing parameters drive the observations, differently for each group, to the child nodes whereas latent budgets parameters update the response classes profile of each level of the control variable. Parameters are estimated 
    
[^257]: MLSkip：通过轻量级元数据实现机器学习过滤器的数据跳过

    MLSkip: Data Skipping for ML Filters via Lightweight Metadata

    [https://arxiv.org/abs/2606.03946](https://arxiv.org/abs/2606.03946)

    本文提出MLSkip方法，首次证明Parquet默认的min-max元数据即可实现对机器学习过滤器的数据跳过修剪，并在TPC-H/TPC-DS基准上对低选择性过滤器实现了27.4%的平均修剪有效性。

    

    数据库供应商最近发布了可用于过滤谓词的AI函数。由于此类函数通常依赖于昂贵且黑盒的机器学习模型，它们带来了新的数据管理挑战。具体而言，针对整数和字符串数据的传统数据跳过技术无法适用于这种新的过滤类型。事实上，目前尚无已知机制可以修剪不合格的行组，例如在从blob存储读取文件时。在这项工作中，我们开创了针对机器学习过滤器的数据跳过技术研究。我们论证了Parquet默认的min-max元数据足以实现修剪。为此，我们建立了与两条研究路线的联系：(i) 最近提出的面向机器学习模型的查询语言，以及(ii) 神经网络验证。我们在ReLU架构上的初步结果表明，在TPC-H和TPC-DS数据表上，对于选择性低于0.1%的过滤器，平均修剪有效性达到27.4%。

    arXiv:2606.03946v2 Announce Type: replace-cross  Abstract: Database vendors recently released AI functions that can be used in filter predicates. As such functions often rely on costly, black-box ML models, they unveil new data management challenges. Concretely, traditional data skipping techniques for integer and string data fail to be applicable to the new filter type. Indeed, there is no known mechanism for pruning non-qualifying row groups, e.g., when reading files from blob storage.   In this work, we initiate the study of data skipping techniques for ML filters. We make the case that Parquet's default min-max metadata is enough to enable pruning. To this end, we draw connections to two lines of research: (i) the recently proposed query language for ML models and (ii) neural network verification.   Our preliminary results on ReLU architectures show that on tables from TPC-H and TPC-DS, the average pruning effectiveness for filters of selectivity below 0.1% amounts to 27.4%. Finall
    
[^258]: 一种用于输入凸神经网络训练的提升方法

    A lift for input-convex neural net training

    [https://arxiv.org/abs/2605.24274](https://arxiv.org/abs/2605.24274)

    针对输入凸神经网络训练中softplus参数化导致负权重区域梯度指数衰减、逃逸缓慢的问题，提出用可学习松弛量加无约束网络（以批次置换不变摘要为输入）替换自由潜在权重的“提升”方法。

    

    输入凸神经网络为密度模型和传输映射的凸势能进行参数化，其凸性要求层间权重必须非负。投影梯度下降通过在每步之后进行投影来强制满足该约束，但由于小批量噪声的存在，边界会被无限次地重新穿越，导致投影永远无法正确识别出一个活跃集。可微的替代方案——直接的softplus参数化——通过softplus正性映射来优化一个自由的潜在权重，其导数在权重为负的区域（即“肩部”）会以指数方式衰减梯度，因此一旦某个坐标到达该区域，就会在指数长的时间内停留在那里。为了在保留这种无约束参数化优点的同时避免其缓慢逃逸的问题，我们提出了提升方法，它用一个可学习的松弛量加上一个无约束网络（称为“主体”）来替换自由的潜在权重，该网络以训练批次的置换不变摘要作为输入。由此，潜在权重

    arXiv:2605.24274v2 Announce Type: replace  Abstract: Input-convex neural nets parametrize the convex potentials of density models and transport maps, and their convexity requires the inter-layer weights to be non-negative. Projected gradient descent enforces this by projecting after each step, and due to mini-batch noise the boundary is re-crossed indefinitely, which leads to an active set the projection never identifies. The differentiable alternative, direct softplus, optimizes a free latent weight through a softplus positivity map whose derivative attenuates the gradient exponentially where the weight is negative---the shoulder---so a coordinate that reaches it stays for an exponentially long time. To keep this unconstrained parametrization without its slow escape, we propose the lift, which replaces the free latent weight by a learnable slack plus an unconstrained network---the body---that takes a permutation-invariant summary of the training batch as input. The latent weight thus 
    
[^259]: 帮助陷入困境的客户：一个能够对话、探询与分流的LLM驱动智能体

    Helping Customers in Distress: An LLM-powered Agent that Converses, Probes, and Routes

    [https://arxiv.org/abs/2605.16268](https://arxiv.org/abs/2605.16268)

    本文开发了一个基于大语言模型的银行客户分流智能体，通过多轮对话探询客户问题并按政策精准分流至专业团队，同时利用真实客户的合成数字孪生生成带标签对话来评估和持续改进该系统。

    

    银行每年收到数百万起关于欺诈、诈骗和争议交易的报告，这使得将客户准确引导至合适的专业支持团队变得极具挑战性。现有的人工处理流程不仅速度缓慢，也给客户和员工都带来压力。为解决这一问题，我们开发了一个面向客户的AI分流智能体，它利用大语言模型（LLM）进行多轮对话、提出相关问题并对案例进行分类，以实现准确的、政策引导下的分流，并将其嵌入到客户服务旅程中。为评估并持续改进该智能体，我们基于历史数据模拟了真实客户的合成数字孪生体，生成真实的、带有标签的对话，以测试广泛的现实场景。本工作详细介绍了该分流智能体的建模方法、与政策的集成、安全护栏与推理框架，以及合成智能体的使用方式。

    arXiv:2605.16268v2 Announce Type: replace-cross  Abstract: Banks receive millions of reports of fraud, scams, and disputed transactions every year, making it challenging to accurately direct customers to the appropriate specialist teams for assistance. The existing manual process driven by humans is slow and stressful for both customers and staff. To address this, we develop a customer-facing AI powered triaging agent that leverages large language models (LLMs) to conduct multi-turn conversations, ask relevant questions, and classify cases for accurate, policy-guided routing, making it embedded in the customer journey. To evaluate and continuously improve the agent, synthetic digital twins of real customers were simulated, generating realistic, labelled dialogues based on historical data to test a wide range of real-world scenarios. This work details the triage agent's modelling approach, integration with policy, safety guardrails and reasoning frameworks, the use of the synthetic agen
    
[^260]: 判官电路解释了LLM-as-a-Judge中由格式引起的不一致性

    Judge Circuits Explain Format-Induced Inconsistency in LLM-as-a-Judge

    [https://arxiv.org/abs/2605.16023](https://arxiv.org/abs/2605.16023)

    该论文通过PEAP方法发现LLM裁判模型的中后层MLP中存在一个稀疏的“潜在评估者”子图，该子图负责抽象评判且独立于输出格式，从而在机制层面解释了LLM-as-a-Judge中格式诱导的评分不一致现象。

    

    大语言模型作为裁判（LLM-as-a-judge）已成为大规模评估模型输出的主流范式，然而同一模型在输出格式改变时会给出系统性不同的分数（例如1-5分评分与真/假标签）。现有的针对这种格式诱导不一致性的诊断仅停留在输入-输出层面。我们使用位置感知边归因补丁（PEAP）方法，对五个开源权重指令微调模型（Gemma-3、Qwen2.5、Llama-3.1）在五个判断任务上的内部机制进行了因果性研究。我们发现，结构化理解任务和开放式偏好任务的判断在多层感知机（MLP）的中后层共享一个稀疏的“潜在评估者”子图；在架构上模块化的模型中，对该子图进行零消融会使判断能力崩溃，同时保持模型在知识探针上的性能。通过在结构上将抽象评判与输出格式化解耦，我们为格式诱导的不一致性提供了机制层面的解释。

    arXiv:2605.16023v3 Announce Type: replace  Abstract: LLM-as-a-judge has become the dominant paradigm for grading model outputs at scale, yet the same model assigns systematically different scores when its output format changes (e.g., a 1-5 rating vs. a True/False label). Existing diagnoses of these format-induced inconsistencies stop at the input-output level. Using Position-aware Edge Attribution Patching (PEAP), we causally investigate the internal mechanism in five open-weight instruction-tuned models (Gemma-3, Qwen2.5, Llama-3.1) across five judgment tasks. We find that judgments across structured understanding and open-ended preference tasks share a sparse Latent Evaluator sub-graph in the mid-to-late multi-layer perceptrons (MLPs); zero-ablating it collapses judgment while preserving performance on our knowledge probes in architecturally modular models. By structurally decoupling abstract judging from output formatting, we provide a mechanistic account of format-induced inconsist
    
[^261]: 基于SMT的加权自动机主动学习

    SMT-Based Active Learning of Weighted Automata

    [https://arxiv.org/abs/2605.07758](https://arxiv.org/abs/2605.07758)

    提出了一种基于SMT的半环参数化加权自动机主动学习算法，在终止时保证产生最小自动机，实验表明其大幅超越朴素基线并与最先进算法竞争，同时生成显著更小的自动机且需要更少的教师交互。

    

    我们提出了一种基于SMT（可满足性模理论）的非确定性加权自动机（WFA）主动学习算法，作为Hankel/L*风格方法的一种实用且鲁棒的替代方案。我们的算法对给定的半环具有参数化特性，并且如果算法终止，则保证产生最小化的加权自动机。我们证明了算法的部分正确性，并给出了一个充分终止条件，该条件特别意味着在所有有限半环上算法必定终止。我们 extensive 的实验评估表明，该算法能够在有限和无限半环上学习众多最小化的加权自动机，大幅超越朴素的基线方法，并与最先进的算法具有竞争力，同时产生显著更小的自动机，并且需要更少的与教师的交互。

    arXiv:2605.07758v2 Announce Type: replace-cross  Abstract: We present an SMT-based active learning algorithm for nondeterministic weighted automata (WFAs) as a practical and robust alternative to Hankel/L*-style methods. Our algorithm is parametric in a given semiring and, if it terminates, guaranteed to produce minimal WFAs. We prove partial correctness and provide a sufficient termination condition, which in particular implies termination for all finite semirings. Our extensive experimental evaluation shows that our algorithm is capable of learning numerous minimal WFAs over both finite and infinite semirings, vastly outperforms a naive baseline, and is competitive with a state-of-the-art algorithm while producing significantly smaller automata and requiring less interaction with the teacher.
    
[^262]: ProteinJEPA：潜在预测改进蛋白质语言模型预训练

    ProteinJEPA: Latent prediction improves protein language model pretraining

    [https://arxiv.org/abs/2605.07554](https://arxiv.org/abs/2605.07554)

    ProteinJEPA在蛋白质语言模型的掩码语言建模基础上引入JEPA式潜在表示预测损失，显著提升了模型在蛋白质检索和远程同源性检测等结构与同源性敏感任务上的表现，且增益随模型规模增大而增强。

    

    蛋白质语言模型主要以掩码语言建模（MLM）进行训练，即预测被掩码的氨基酸身份。联合嵌入预测架构（JEPA）则改为预测潜在表示，但尚未被应用于蛋白质领域。ProteinJEPA在MLM的基础上增加了一个余弦损失，用于在给定未掩码序列的条件下预测教师模型的半深度隐藏状态。在19个任务上，采用3500万和1.5亿参数的ESM2模型以及三个预训练随机种子，MLM+JEPA在114次比较中分别有78次和76次优于计算量匹配和训练步数匹配的仅MLM持续训练（14次落后，22次平局）。在结构与同源性敏感的任务上，计算量匹配的中位数增益为+0.0106，而其他任务上仅为+0.0041，其中以SCOPe-40检索和远程同源性任务提升最为显著，分别实现了Recall@1提高6.1个百分点和准确率提高2.7个百分点。这些任务上的增益随模型规模从800万增加到1.5亿而持续增大。

    arXiv:2605.07554v2 Announce Type: replace-cross  Abstract: Protein language models are trained primarily with masked language modeling (MLM), which predicts masked amino-acid identities. Joint-embedding predictive architectures (JEPA) instead predict latent representations, but have not been applied to proteins.   ProteinJEPA supplements MLM with a cosine loss for predicting the half-depth hidden states of a teacher given the unmasked sequence. On 19 tasks, with ESM2 at 35M and 150M parameters and three pretraining seeds, MLM+JEPA outperforms compute-matched and step-matched MLM-only continued training in 78 and 76 of 114 comparisons (14 losses, 22 ties). The median compute-matched gain is $+0.0106$ on structure- and homology-sensitive tasks versus $+0.0041$ elsewhere, led by SCOPe-40 retrieval and remote homology with improvements of 6.1 percentage points in Recall@1 and 2.7 points in accuracy, respectively. Gains on these tasks increase with model size from 8M to 150M. Against the of
    
[^263]: QuadraSHAP：基于高斯-勒让德求积的乘积博弈中稳定且可扩展的Shapley值

    QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature

    [https://arxiv.org/abs/2605.05870](https://arxiv.org/abs/2605.05870)

    本文提出QuadraSHAP，证明乘积博弈中每个玩家的Shapley值可精确表示为一维积分，从而利用高斯-勒让德求积以仅需⌈d/2⌉个节点即可实现可证明精确、稳定且可扩展的高效计算。

    

    我们研究了乘积博弈中Shapley值的高效计算——乘积博弈是一类联盟价值可分解为各玩家项乘积的合作博弈。当价值函数从底层模型继承乘法结构时，例如具有乘积核的核方法和基于树的模型，这类博弈会出现在机器学习可解释性问题中。我们的关键结果是：乘积博弈中每个玩家的Shapley值都存在精确的一维积分表示，即对指数级数量特征联盟的加权和可坍缩为一个次数为(d-1)的多项式在[0,1]区间上的积分，其中d为特征总数。由此得到一种高斯-勒让德求积方案：当节点数满足m_q ≥ ⌈d/2⌉时，该方案可证明是精确的；否则可提供近似精确的估计，其误差可证明随m_q呈几何级数衰减。

    arXiv:2605.05870v3 Announce Type: replace  Abstract: We study the efficient computation of Shapley values for \emph{product games} -- cooperative games in which the coalition value factorizes as a product of per-player terms. Such games arise in machine learning explainability whenever the value function inherits a multiplicative structure from the underlying model, as in kernel methods with product kernels and tree-based models. Our key result is that the Shapley value of each player in a product game admits an exact one-dimensional integral representation: the weighted sum over exponentially many feature coalitions collapses to the integral of a degree-$(d-1)$ polynomial over $[0,1]$, where $d$ is the total number of features. This yields a Gauss--Legendre quadrature scheme that is \emph{provably exact} whenever the number of nodes satisfies $m_q \geq \lceil d/2 \rceil$, and otherwise provides a \emph{near-exact} approximation with error provably decaying geometrically in $m_q$. In p
    
[^264]: ANO：通过有界、再下降的增益场实现鲁棒策略优化

    ANO: Robust Policy Optimization via Bounded, Redescending Gain Fields

    [https://arxiv.org/abs/2605.02320](https://arxiv.org/abs/2605.02320)

    提出锚定邻域优化（ANO），通过C^∞光滑的整形核直接构造有界且再下降的增益场，在PPO的死区漂移与SPO的无界增益这两个极端之间取得平衡，从而实现更稳定、更鲁棒的策略优化。

    

    近端策略优化（PPO）在强化学习和大语言模型对齐中占据主导地位，但其硬裁剪机制与无约束的替代方法（如SPO）处于稳定性-效率困境的两个极端。我们认为这一困境最好从动态角度来理解：代理目标函数本质上是关于概率比率的反馈律，其裁剪/惩罚的形状定义了驱动更新动态的增益场。PPO的裁剪会产生一个死区（信任区域之外反馈为零），导致策略在动量作用下开环漂移；而SPO的二次惩罚则产生无界且线性增长的增益，使动态系统刚化，在激进步长下失去稳定性。基于这一视角，我们提出了锚定邻域优化（ANO），直接设计增益场：一个C^∞光滑的整形核，在r=1处锚定恒等映射，并恰好在预设的信任区域边界1+ε处达到峰值，且在边界之外有界……（摘要原文在此处截断）

    arXiv:2605.02320v3 Announce Type: replace  Abstract: Proximal Policy Optimization (PPO) dominates reinforcement learning and LLM alignment, yet its hard-clipping mechanism and unconstrained alternatives (e.g., SPO) sit at two extremes of a stability-efficiency dilemma. We argue that this dilemma is best understood dynamically: a surrogate objective is a feedback law on the probability ratio, and its clipping/penalty shape defines a gain field that drives the update dynamics. PPO's clip induces a dead zone (zero feedback outside the trust region), leaving the policy to drift open-loop under momentum; SPO's quadratic penalty induces an unbounded, linearly growing gain that stiffens the dynamics and destabilizes under aggressive step sizes. Guided by this view, we derive Anchored Neighborhood Optimization (ANO), which designs the gain field directly: a $C^\infty$ shaping kernel that anchors the identity map at $r{=}1$, peaks exactly at a prescribed trust-region boundary $1{+}\epsilon$, bo
    
[^265]: Anon：将自适应性外推超越SGD与Adam

    Anon: Extrapolating Adaptivity Beyond SGD and Adam

    [https://arxiv.org/abs/2605.02317](https://arxiv.org/abs/2605.02317)

    该论文提出Anon优化器，突破了SGD与Adam之间0到1的插值限制，首次实现在整个实数范围内连续外推自适应参数（如CNN需要负自适应性、Transformer需要γ≥1），并通过增量延迟更新机制保证超界情形下的稳定收敛。

    

    诸如Adam的自适应优化器与SGD等非自适应方法在不同架构上表现出不同的泛化能力。先前的可调优化器试图通过在SGD和Adam之间严格插值来弥合这一差距，实际上将自适应性限制在了0到1的界限之内。然而，这种受限的插值从根本上是不充分的：我们揭示了最优的自适应性往往需要外推，例如经典CNN需要负的自适应性，而Transformer需要至少为1的自适应性（γ ≥ 1）。对自适应性进行外推在理论上违反了严格的非递减预条件子假设，常常导致现有方法发散。为了突破这一障碍，我们提出了Anon，一种在整个实数范围内实现完全连续自适应性外推的优化器。为了保证在这些超界情形下的可证明稳定性，我们引入了增量延迟更新机制……

    arXiv:2605.02317v3 Announce Type: replace  Abstract: Adaptive optimizers such as Adam and non-adaptive methods like SGD exhibit distinct generalization capabilities across different architectures. Prior tunable optimizers attempt to bridge this gap by strictly interpolating between SGD and Adam, effectively confining adaptivity within the 0-to-1 bound. However, this restricted interpolation is fundamentally insufficient: we reveal that optimal adaptivity often requires extrapolation, such as negative adaptivity for classical CNNs and adaptivity of at least one ($\gamma \geq 1$) for Transformers. Extrapolating adaptivity theoretically violates the strict non-decreasing pre-conditioner assumption, often leading to divergence in existing methods. To break this barrier, we propose Anon, an optimizer that achieves fully continuous adaptivity extrapolation across the entire real-number spectrum. To guarantee provable stability in these out-of-bound regimes, we introduce Incremental Delay Upd
    
[^266]: 使用变分期望最大化算法拟合大型非线性混合效应模型

    Fitting Large Nonlinear Mixed Effects Models Using Variational Expectation Maximization

    [https://arxiv.org/abs/2604.26160](https://arxiv.org/abs/2604.26160)

    本文提出利用变分期望最大化（VEM）算法，结合灵活的变分族和反向模式自动微分技术，高效拟合大型非线性混合效应模型，可扩展至超过15,000个群体参数的规模。

    

    非线性混合效应（NLME）模型被广泛应用于药物计量学及相关领域，用于分析层次结构和纵向数据。然而，随着参数和随机效应数量的增加，最大化边际似然的传统方法在计算上变得十分昂贵。本文探讨了变分期望最大化（VEM）算法，这是一种可扩展的NLME模型拟合替代方法。VEM最初在概率图模型的背景下被提出，后来因变分自编码器而广为人知，但尚未被广泛应用于NLME建模。通过利用灵活的变分族和反向模式自动微分技术，VEM能够高效地最大化边际似然，可扩展到拥有超过15,000个群体参数的NLME模型。这项工作详细描述了VEM，将其与其他NLME拟合算法进行了比较，并突出了其可扩展性。

    arXiv:2604.26160v2 Announce Type: replace-cross  Abstract: Nonlinear Mixed Effects (NLME) models are widely used in pharmacometrics and related fields to analyze hierarchical and longitudinal data. However, as the number of parameters and random effects increases, traditional methods for maximizing the marginal likelihood become computationally expensive. This paper explores the Variational Expectation Maximization (VEM) algorithm, a scalable alternative for fitting NLME models. Originally introduced in the context of probabilistic graphical models and later popularized through variational autoencoders, VEM has not been extensively applied to NLME modeling. By leveraging flexible variational families and reverse-mode automatic differentiation, VEM can efficiently maximize the marginal likelihood, scaling to NLME models with over 15,000 population parameters. This work provides a detailed description of VEM, compares it to other NLME fitting algorithms, and highlights its scalability th
    
[^267]: 评估降维对聚类性能的影响——一项系统性研究

    Assessing the impact of dimensionality reduction on clustering performance - a systematic study

    [https://arxiv.org/abs/2604.22099](https://arxiv.org/abs/2604.22099)

    本研究系统评估了五种降维方法（PCA、核PCA、VAE、Isomap、MDS）对四种常用聚类算法（k-means、AHC、GMM、OPTICS）性能的影响，并使用调整兰德指数（ARI）在不同降维水平下进行全面比较。

    

    降维是对高维数据进行聚类的关键预处理步骤，然而针对不同方法和数据类型对其影响的全面评估仍然有限。在本研究中，我们系统地评估了五种降维技术——主成分分析（PCA）、核主成分分析、变分自编码器（VAE）、等度量映射和多维尺度分析（MDS）——对四种流行聚类算法性能的影响，这四种聚类算法为：k-means、凝聚层次聚类（AHC）、高斯混合模型（GMM）以及基于点的排序识别聚类结构（OPTICS）。我们使用调整兰德指数（ARI）评估聚类质量，比较了不使用降维与使用文献中推荐的不同降维水平（即 k-1，其中 k 为聚类数量，以及原始维度的 25% 和 50%）时的聚类结果。

    arXiv:2604.22099v3 Announce Type: replace  Abstract: Dimensionality reduction is a critical preprocessing step for clustering high-dimensional data, yet comprehensive evaluation of its impact across diverse methods and data types remains limited. In this study, we systematically assess the influence of five dimensionality reduction techniques - Principal Component Analysis (PCA), Kernel Principal Component Analysis (Kernel PCA), Variational Autoencoder (VAE), Isometric Mapping (Isomap), and Multidimensional Scaling (MDS) - on the performance of four popular clustering algorithms - k-means, Agglomerative Hierarchical Clustering (AHC), Gaussian Mixture Models (GMM), and Ordering Points to Identify the Clustering Structure (OPTICS). We evaluate clustering quality using the Adjusted Rand Index (ARI), comparing results without and with dimensionality reduction at different reduction levels recommended in the literature (i.e., k-1, where k is the number of clusters, and 25% and 50% of the or
    
[^268]: PipeLive：面向动态大语言模型服务的高效在线原位流水线并行重配置

    PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving

    [https://arxiv.org/abs/2604.12171](https://arxiv.org/abs/2604.12171)

    提出PipeLive系统，实现了在不中断推理的情况下对大语言模型流水线并行配置进行在线原位重配置，解决了KV缓存空间受限与执行期间KV一致性维护的双重挑战，使流水线并行能够适应无服务器平台和异构GPU等动态服务环境。

    

    流水线并行（PP）被广泛用于将大语言模型（LLM）的各层划分到多个GPU上，从而为大模型提供可扩展的推理能力。然而，现有系统依赖静态的流水线并行配置，无法适应动态环境，例如无服务器平台和异构GPU环境。通过停止并重新部署服务来重配置流水线并行会产生难以承受的停机时间，因此重配置必须以在线原位的方式进行，且不能中断推理。然而，在线原位的流水线并行重配置在本质上极具挑战性。GPU已被模型权重和KV缓存占满，几乎没有为新层布局预留的空间，因此需要对KV缓存进行调整大小，这与vLLM等为追求吞吐量而预分配内存的系统相冲突。此外，在执行过程中维持KV一致性十分困难：停止并复制的方式会引入长时间停顿，而后台同步则随着状态不断演进面临不一致的风险。

    arXiv:2604.12171v2 Announce Type: replace-cross  Abstract: Pipeline parallelism (PP) is widely used to partition layers of large language models (LLMs) across GPUs, enabling scalable inference for large models. However, existing systems rely on static PP configurations that fail to adapt to dynamic settings, such as serverless platforms and heterogeneous GPU environments. Reconfiguring PP by stopping and redeploying service incurs prohibitive downtime, so reconfiguration must instead proceed live and in place, without interrupting inference. However, live in-place PP reconfiguration is fundamentally challenging. GPUs are already saturated with model weights and KV cache, leaving little room for new layer placements and necessitating KV cache resizing, at odds with systems like vLLM that preallocate for throughput. Moreover, maintaining KV consistency during execution is difficult: stop-and-copy introduces large pauses, while background synchronization risks inconsistency as states evol
    
[^269]: 基于对抗式多任务学习的联合干扰检测与识别

    Joint Interference Detection and Identification via Adversarial Multi-task Learning

    [https://arxiv.org/abs/2604.08607](https://arxiv.org/abs/2604.08607)

    该论文建立了一个有理论支撑的多任务学习框架，通过推导加权期望损失上界，将任务相似度与Wasserstein距离和可学习的任务关系系数联系起来，并据此提出对抗式多任务网络，实现干扰检测、调制识别和干扰识别的联合处理。

    

    精确的干扰检测与识别对于提高通信系统在非合作无线环境中的生存能力至关重要。尽管深度学习（DL）已推动了该领域的发展，但现有的单任务学习（STL）方法忽略了任务间固有的相关性。此外，新兴的多任务学习（MTL）方法往往缺乏量化和建模任务关系的理论基础。为了弥补这一空白，我们建立了一个具有理论依据的多任务学习框架，用于联合干扰检测、调制识别和干扰识别。首先，我们推导了多任务学习框架中加权期望损失的上界。该上界明确地将多任务学习性能与任务相似度联系起来，任务相似度通过Wasserstein距离和可学习的任务关系系数来量化。在该理论的指导下，我们提出了对抗式多任务干扰检测与识别网络。

    arXiv:2604.08607v2 Announce Type: replace-cross  Abstract: Precise interference detection and identification are crucial for enhancing the survivability of communication systems in non-cooperative wireless environments. While deep learning (DL) has advanced this field, existing single-task learning (STL) approaches neglect inherent task correlations. Furthermore, emerging multi-task learning (MTL) methods often lack a theoretical foundation for quantifying and modeling task relationships. To bridge this gap, we establish a theoretically grounded MTL framework for joint interference detection, modulation identification, and interference identification. First, we derive an upper bound for the weighted expected loss in MTL frameworks. This bound explicitly connects MTL performance to task similarity, quantified by the Wasserstein distance and learnable task relation coefficients. Guided by this theory, we present the adversarial multi-task interference detection and identification network
    
[^270]: TiAb Review 插件：一个用于系统综述中 AI 辅助文献筛选的浏览器工具

    TiAb Review Plugin: A Browser-Based Tool for AI-Assisted Study Selection in Systematic Reviews

    [https://arxiv.org/abs/2604.08602](https://arxiv.org/abs/2604.08602)

    该论文开发了开源 Chrome 扩展 TiAb Review 插件，无需编程和服务器即可利用 AI 完成系统综述中从标题摘要到全文的文献筛选。

    

    基于服务器的筛选工具需要支付订阅费用，而开源替代方案则需要编程技能，且全文筛选一直不在无代码开源工具的能力范围之内。我们开发了 TiAb Review 插件，这是一个开源的 Chrome 浏览器扩展，提供无代码、无服务器的 AI 辅助文献筛选，同时涵盖标题与摘要（T&A）筛选和全文筛选两个阶段。该插件使用 Google Sheets 作为共享数据库、Google Drive 作为 PDF 存储库，用户只需提供自己的大语言模型（LLM）API 密钥。在 T&A 筛选方面，它支持人工审核、LLM 批量筛选和机器学习（ML）主动学习三种方式。在全文筛选方面，它可从 PubMed Central、Europe PMC、Unpaywall、OpenAlex 以及出版商网页自动获取开放获取的 PDF，支持带有结构化排除原因和裁决机制的盲法双人评审，并可选择获取附有页面锚定证据的 LLM 判断……

    arXiv:2604.08602v2 Announce Type: replace-cross  Abstract: Server-based screening tools impose subscription costs, while open-source alternatives require coding skills, and full-text screening has remained outside the scope of no-code open-source tools. We developed TiAb Review Plugin, an open-source Chrome browser extension that provides no-code, serverless artificial intelligence (AI)-assisted study selection covering both title and abstract (T&A) screening and full-text screening. It uses Google Sheets as a shared database and Google Drive as a PDF store, and users supply their own large language model (LLM) API key. For T&A screening, it offers manual review, LLM batch screening, and machine learning (ML) active learning. For full-text screening, it retrieves open-access PDFs from PubMed Central, Europe PMC, Unpaywall, OpenAlex, and publisher pages, supports blinded dual review with structured exclusion reasons and adjudication, optionally obtains an LLM judgment with page-anchored
    
[^271]: 强化学习中的价值镜像下降

    Value Mirror Descent for Reinforcement Learning

    [https://arxiv.org/abs/2604.06039](https://arxiv.org/abs/2604.06039)

    本文提出价值镜像下降（VMD）方法，将凸优化中的镜像下降融入经典价值迭代框架，在确定性设定下实现线性收敛，并针对生成式模型下的随机设定开发了结合方差缩减技术的 SVMD 变体。

    

    价值迭代类方法在强化学习（RL）中计算近似最优价值函数方面已被广泛研究。在生成式采样模型下，这些方法能够获得比策略优化方法更优的样本复杂度，尤其是在对折扣因子的依赖方面。在实践中，这类方法常被用于离线训练。本文研究了具有状态空间 S、动作空间 A、折扣因子 $\gamma\in(0,1)$ 以及成本取值于 $[0,1]$ 的折扣马尔可夫决策过程。我们提出了一种新颖的价值优化方法，称为价值镜像下降（VMD），该方法将凸优化中的镜像下降融入到经典的价值迭代框架之中。在转移核已知的确定性设定下，我们证明了 VMD 具有线性收敛性。对于带生成式模型的随机设定，我们开发了一种随机变体 SVMD，它结合了方差缩减技术。

    arXiv:2604.06039v2 Announce Type: replace-cross  Abstract: Value iteration-type methods have been extensively studied for computing a nearly optimal value function in reinforcement learning (RL). Under a generative sampling model, these methods can achieve sharper sample complexity than policy optimization approaches, particularly in their dependence on the discount factor. In practice, they are often employed for offline training. In this paper, we consider discounted Markov decision processes with state space S, action space A, discount factor $\gamma\in(0,1)$ and costs in $[0,1]$. We introduce a novel value optimization method, termed value mirror descent (VMD), which integrates mirror descent from convex optimization into the classical value iteration framework. In the deterministic setting with known transition kernels, we show that VMD converges linearly. For the stochastic setting with a generative model, we develop a stochastic variant, SVMD, which incorporates variance reducti
    
[^272]: 基于函数的不确定性量化实现安全的基于学习的控制

    Safe learning-based control via function-based uncertainty quantification

    [https://arxiv.org/abs/2604.01173](https://arxiv.org/abs/2604.01173)

    该论文提出将未知函数建模为可生成独立同分布样本的随机函数，并利用场景方法仅基于采样数据构建高概率成立的不确定性管道，从而摆脱了传统方法对函数光滑性的限制性假设并能处理不连续性，实现安全的基于学习的控制。

    

    在安全关键系统中部署基于学习的控制方法时，不确定性量化至关重要。这通常通过构建以高概率包围感兴趣的未知函数（例如奖励函数、约束函数或底层动力学模型）的不确定性管道来实现。然而，现有的不确定性量化方法通常依赖于对未知函数光滑性属性进行编码的限制性假设，例如函数空间中的已知范数。此外，这些方法通常难以处理不连续性。在本文中，我们将未知函数建模为一个随机函数，从中可以生成独立同分布的实现样本。然后，我们通过场景方法构建以高概率成立的不确定性管道。我们的不确定性管道仅依赖于采样实现，因此能够处理不连续性。

    arXiv:2604.01173v2 Announce Type: replace-cross  Abstract: Uncertainty quantification is essential when deploying learning-based control methods in safety-critical systems. This is commonly realized by constructing uncertainty tubes that enclose the unknown function of interest, e.g., the reward and constraint functions or the underlying dynamics model, with high probability. However, existing approaches for uncertainty quantification typically rely on restrictive assumptions that encode smoothness properties of the unknown function, such as a known norm in a function space. Moreover, these methods usually struggle with discontinuities. In this paper, we model the unknown function as a random function from which independent and identically distributed realizations can be generated. We then construct uncertainty tubes via the scenario approach that hold with high probability. Our uncertainty tubes rely solely on sampled realizations and can therefore accommodate discontinuities represen
    
[^273]: 用于方差最小化和风险规避多臂老虎机的Softmax梯度策略

    Softmax gradient policy for variance minimization and risk-averse multi armed bandits

    [https://arxiv.org/abs/2604.00241](https://arxiv.org/abs/2604.00241)

    该论文提出了一种基于softmax参数化的新算法，用于在风险规避的多臂老虎机问题中选择方差最小（风险最低）的臂，通过两次独立抽样构建无偏估计并证明了算法的收敛性。

    

    多臂老虎机（MAB）问题的算法在序贯决策中扮演着核心角色，并已在理论和数值方面得到广泛探索。虽然大多数经典方法旨在识别期望奖励最高的臂，但我们专注于一个风险感知的设定，其目标是选择方差最低的臂，即优先考虑稳定性而非潜在的高但不确定的回报。为了建模决策过程，我们考虑了策略的softmax参数化；我们提出了一种新算法来选择最小方差（或最小风险）的臂，并在自然条件下证明了其收敛性。该算法通过从所选臂的分布中进行两次独立抽样来构建目标函数的无偏估计。我们提供了数值实验，展示了这些算法的实际行为，并为实现选择提供了指导。该设定还涵盖了一般性的……

    arXiv:2604.00241v2 Announce Type: replace-cross  Abstract: Algorithms for the Multi-Armed Bandit (MAB) problem play a central role in sequential decision-making and have been extensively explored both theoretically and numerically. While most classical approaches aim to identify the arm with the highest expected reward, we focus on a risk-aware setting where the goal is to select the arm with the lowest variance, favoring stability over potentially high but uncertain returns. To model the decision process, we consider a softmax parameterization of the policy; we propose a new algorithm to select the minimal variance (or minimal risk) arm and prove its convergence under natural conditions. The algorithm constructs an unbiased estimate of the objective by using two independent draws from the selected arm's distribution. We provide numerical experiments that illustrate the practical behavior of these algorithms and offer guidance on implementation choices. The setting also covers general 
    
[^274]: 学习记忆：面向边缘无服务器自动伸缩的注意力强化学习

    Learning to Remember: Attentive Reinforcement Learning for Edge Serverless Autoscaling

    [https://arxiv.org/abs/2603.28790](https://arxiv.org/abs/2603.28790)

    该论文提出了一种将注意力增强的双层堆叠LSTM集成到PPO智能体中的稳定性感知自动伸缩框架，通过利用近期时间上下文克服边缘无服务器环境中传统控制器的反应延迟和DRL智能体的时间盲区问题。

    

    在边缘计算中，无服务器工作负载的随机性和突发性给自主资源编排带来了挑战。传统的被动式控制器（如Kubernetes水平Pod自动伸缩器HPA）存在反应延迟问题，导致流量激增时违反服务水平目标（SLO），以及在流量回落时出现资源抖动。虽然深度强化学习（DRL）为实现主动式管理提供了一条路径，但标准智能体存在“时间盲区”问题，即无法在非马尔可夫边缘环境中利用近期的时间上下文。为弥合这一差距，我们提出了一个稳定性感知的自动伸缩框架，通过将近端策略优化（PPO）智能体中的注意力增强双层堆叠LSTM架构相集成，将短时程时间上下文与控制统一起来。与浅层循环模型不同，我们的方法采用了一种可学习的注意力机制，对近期历史信息进行加权（注：原文摘要在此处被截断）。

    arXiv:2603.28790v2 Announce Type: replace-cross  Abstract: In edge computing, the stochastic and bursty nature of serverless workloads challenges autonomous resource orchestration. Traditional reactive controllers, such as the Kubernetes Horizontal Pod Autoscaler (HPA), suffer from reaction latency, leading to Service Level Objective (SLO) violations during traffic spikes and resource flapping during ramp-downs. While Deep Reinforcement Learning (DRL) offers a pathway toward proactive management, standard agents suffer from \textit{temporal blindness}, an inability to exploit the recent temporal context in non-Markovian edge environments. To bridge this gap, we propose a stability-aware autoscaling framework unifying short-horizon temporal context and control via an Attention-Enhanced Double-Stacked LSTM architecture integrated within a Proximal Policy Optimization (PPO) agent. Unlike shallow recurrent models, our approach employs a learned attention mechanism that weights recent histo
    
[^275]: PHONOS：面向在线流式应用的语音中和化技术

    PHONOS: PHOnetic Neutralization for Online Streaming Applications

    [https://arxiv.org/abs/2603.27001](https://arxiv.org/abs/2603.27001)

    提出了PHONOS——一个面向实时说话人匿名化的流式口音中和模块，通过静音感知DTW对齐、零样本语音转换和仅40毫秒前瞻的因果口音翻译器，将非母语音段转换为目标口音，使非母语口音线索减少81%。

    

    说话人匿名化（SA）系统在修改音色的同时会保留地区性或非母语口音线索，这是有问题的，因为此类线索可能暴露说话人的母语或地理背景，从而缩小匿名集合的范围。为解决这一问题，我们提出了PHONOS，一个用于实时说话人匿名化的流式模块，它在隐私意义上执行口音中和：通过将非母语的音段实现转换到选定的目标口音域，来减少口音来源线索。我们的方法预先生成“黄金说话人”语音，这些语音保留源说话人的音色和节奏，但利用静音感知的DTW对齐和零样本语音转换，将外国口音的音段替换为母语音段。这些语音用于监督一个因果口音翻译器，该翻译器在至多40毫秒前瞻的条件下将非母语内容token映射为母语等价token，并采用交叉熵与CTC联合损失进行训练。我们的评估显示，非母语口音线索减少了81%。

    arXiv:2603.27001v2 Announce Type: replace-cross  Abstract: Speaker anonymization (SA) systems modify timbre while leaving regional or non-native accent cues intact, which is problematic because such cues can reveal a speaker's first-language or geographic background and narrow the anonymity set. To address this issue, we present PHONOS, a streaming module for real-time SA that performs accent neutralization in a privacy sense: reducing accent-origin cues by converting non-native segmental realizations toward a chosen target accent domain. Our approach pre-generates golden speaker utterances that preserve source timbre and rhythm but replace foreign segmentals with native ones using silence-aware DTW alignment and zero-shot voice conversion. These utterances supervise a causal accent translator that maps non-native content tokens to native equivalents with at most 40ms look-ahead, trained using joint cross-entropy and CTC losses. Our evaluations show an 81% reduction in non-native accen
    
[^276]: 一个面向指令条件化上下文时间序列任务的基础模型

    A Foundation Model for Instruction-Conditioned In-Context Time Series Tasks

    [https://arxiv.org/abs/2603.22586](https://arxiv.org/abs/2603.22586)

    提出iAmTime——一个通过指令条件化摊销元学习训练的时间序列基础模型，能够利用专门的语义标记从显式的输入-输出示例演示中直接推断任务，实现时间序列任务的上下文学习。

    

    上下文学习（ICL）通过条件化于示例演示而非更新模型参数，实现了推理阶段的任务自适应。尽管近期的时间序列基础模型已融入上下文条件化、检索或基于示例的提示机制，但它们通常依赖于隐式的位置结构或特定任务的目标，而非显式的指令条件化输入-输出示例演示。我们提出了iAmTime，这是一个通过指令条件化摊销元学习训练的时间序列基础模型，能够直接从示例演示中推断任务。iAmTime将每个任务场景表示为一个结构化提示，涵盖历史上下文和未来已知变量，使用专门的语义标记来关注指定的时间序列区域、在演示之间交换信息，并将任务信息注入查询表示中。该模型结合了一个分层多范围Transformer编码器，它

    arXiv:2603.22586v4 Announce Type: replace  Abstract: In-context learning (ICL) enables task adaptation at inference time by conditioning on demonstrations rather than updating model parameters. Although recent time-series foundation models incorporate contextual conditioning, retrieval, or example-based prompting, they typically rely on implicit positional structure or task-specific objectives rather than explicit instruction-conditioned input-output demonstrations. We introduce iAmTime, a time-series foundation model trained with instruction-conditioned amortized meta-learning to infer tasks directly from example demonstrations. iAmTime represents each episode as a structured prompt over historical context and future-known variables using specialized semantic tokens that attend to designated time-series regions, exchange information across demonstrations, and inject task information into the query representation. The model combines a Hierarchical Multi-Scope Transformer Encoder, which
    
[^277]: 基于耦合成对标签的二元分类

    Binary Classification from Coupled Pairwise Labels

    [https://arxiv.org/abs/2603.19713](https://arxiv.org/abs/2603.19713)

    提出SD-Pcomp学习方法，通过一个同时保留相似/不相似结构与成对比较排序结构的目标函数，实现了仅利用耦合成对关系信息（而非绝对类别标签）的二元分类。

    

    即使难以给单个实例分配绝对的类别标签，关系信息可能仍然可用，例如两个实例是否属于同一类别，或者哪个实例更可能属于正类。在本研究中，我们将这两类信息分别称为相似/不相似（SD）标签和成对比较（Pcomp）标签，并考虑使用来自相同实例对的这两种关系信息进行二元分类。SD学习利用相似对与不相似对之间的区分，但不使用每一对内部的排序；而Pcomp学习利用每一对内部的排序，但不区分相似对与不相似对。因此，我们提出了SD-Pcomp学习，其目标函数同时保留了SD学习和Pcomp学习的结构。所提出的目标函数允许两种分解方式……

    arXiv:2603.19713v2 Announce Type: replace  Abstract: Even when it is difficult to assign absolute class labels to individual instances, relational information may still be available, such as whether two instances belong to the same class or which instance is more likely to belong to the positive class. In this study, we refer to these two types of information as Similarity/Dissimilarity (SD) labels and Pairwise Comparison (Pcomp) labels, respectively, and consider binary classification that uses both types of relational information from the same instance pairs. SD learning uses the distinction between similar and dissimilar pairs but does not use the ordering within each pair, whereas Pcomp learning uses the ordering within each pair but does not distinguish between similar and dissimilar pairs. We therefore propose SD-Pcomp learning, whose objective function simultaneously preserves the structures of both SD learning and Pcomp learning. The proposed objective function admits two decom
    
[^278]: 一种用于双网络多孔介质流体流动的自适应机器学习框架

    An Adaptive Machine Learning Framework for Fluid Flow in Dual-Network Porous Media

    [https://arxiv.org/abs/2603.19561](https://arxiv.org/abs/2603.19561)

    该论文提出了一种具有自适应权重调节的物理信息神经网络（PINN）框架，用于双重孔隙度/渗透率多孔介质系统的正向与反向建模，实现了快速预测和可靠的反演分析。

    

    多孔材料——无论是天然的还是人工设计的——通常表现出双重孔隙网络结构，这种结构控制着矿产勘探和致密页岩油气开采等过程。双重孔隙度/渗透率（DPP）数学模型描述了不可压缩流体在两个相互作用并伴随网络间质量交换的孔隙网络中的流动。尽管数值方法已取得显著进展，但仍然需要能够实现快速预测、数据同化和可靠反演分析的计算框架。为解决这一问题，我们提出了一个物理信息神经网络（PINN）框架，用于DPP系统的正向与反向建模。该方法将混合形式的控制方程以及边界条件直接编码到损失函数中，并采用自适应加权策略来平衡各项贡献。该框架的关键特性包括自适应权重调节、动态配置点采样

    arXiv:2603.19561v2 Announce Type: replace-cross  Abstract: Porous materials -- natural or engineered -- often exhibit dual pore-network structures that govern processes such as mineral exploration and hydrocarbon recovery from tight shales. Double porosity/permeability (DPP) mathematical models describe incompressible fluid flow through two interacting pore networks with inter-network mass exchange. Despite significant advances in numerical methods, there remains a need for computational frameworks that enable rapid forecasting, data assimilation, and reliable inverse analysis. To address this, we present a physics-informed neural network (PINN) framework for forward and inverse modeling of DPP systems. The proposed approach encodes the governing equations in mixed form, along with boundary conditions, directly into the loss function, with adaptive weighting strategies to balance their contributions. Key features of the framework include adaptive weight tuning, dynamic collocation poin
    
[^279]: 截断盲区：解码策略如何系统性地排除人类式的词元选择

    The Truncation Blind Spot: How Decoding Strategies Systematically Exclude Human-Like Token Choices

    [https://arxiv.org/abs/2603.18482](https://arxiv.org/abs/2603.18482)

    该论文提出“截断盲区”概念，揭示 top-k 和核采样等解码策略因截断低概率词元而系统性地排除了 8–18% 的人类典型选词，从而为机器生成文本为何始终可被检测提供了机制性解释。

    

    为什么机器生成的文本依然能够被检测出来？我们在解码阶段研究了一种机制性解释：诸如 top-k 和核采样（nucleus sampling）等标准策略将生成过程限制在高概率词元上，而人类作者通常会选择模型概率分布中更深层次、但在语境中合适的词语。截断使得这些人类选择中可测量的部分变得无法触及；我们将其称为“截断盲区”。在五个开源模型和三个领域的实验中，8–18% 的人类选择词元落在了常见截断边界之外。语言分析进一步揭示，实义词元被不成比例地排除在外。在一个包含 180 万条机器生成文本的基准测试中，仅使用可预测性和词汇多样性特征的分类器即达到接近 0.97 的平均 AUC-ROC，且在不同解码设置下存在显著差异，并在不同生成器之间展现出很强的迁移能力。概率下限采样器能够在很大程度上缩……（原文摘要在此处截断）

    arXiv:2603.18482v4 Announce Type: replace  Abstract: Why does machine-generated text remain detectable? We investigate a mechanistic explanation at the decoding stage: standard strategies such as top-$k$ and nucleus sampling restrict generation to high-probability tokens, while human writers routinely choose contextually appropriate words from deeper in the model's probability distribution. Truncation makes a measurable share of these choices unreachable; we call this the \emph{truncation blind spot}. Across five open models and three domains, 8--18\% of human-selected tokens fall outside common truncation boundaries. Linguistic analysis further reveals disproportionate exclusion of content-word tokens. In a benchmark comprising 1.8 million machine generations, classifiers using only predictability and lexical diversity achieve mean AUC-ROC near 0.97, with substantial variation across decoding settings and strong transfer across generators. Probability-floor samplers substantially narr
    
[^280]: 非平稳高斯过程的正则傅里叶特征

    Regular Fourier Features for Nonstationary Gaussian Processes

    [https://arxiv.org/abs/2602.23006](https://arxiv.org/abs/2602.23006)

    该论文提出正则傅里叶特征方法，通过直接离散化可调和非平稳高斯过程的谱表示，摆脱了谱密度必须为概率测度的限制性假设，实现了无需概率假设、结构上半正定的高效低秩近似。

    

    模拟高斯过程需要从高维高斯分布中采样，其计算复杂度随采样位置数量呈三次方增长。谱方法通过利用傅里叶表示，并将谱密度视为适合蒙特卡洛近似的概率分布来解决这一挑战。尽管这种概率解释对平稳过程是有效的，但对于非平稳情况而言则过于受限，因为非平稳过程的谱密度通常并非概率测度。为了避免这一限制，我们提出了一种针对具有一维输入的可调和过程的正则傅里叶特征方法。我们的方法直接对谱表示进行离散化，在不需要概率假设的情况下保留了谱权重之间的相关结构。在假设谱支撑有限的前提下，该方法可产生一种结构上即为半正定的高效低秩近似。

    arXiv:2602.23006v3 Announce Type: replace-cross  Abstract: Simulating a Gaussian process requires sampling from a high-dimensional Gaussian distribution, which scales cubically with the number of sample locations. Spectral methods address this challenge by exploiting the Fourier representation and treating the spectral density as a probability distribution suitable for Monte Carlo approximation. Although this probabilistic interpretation is valid for stationary processes, it is overly restrictive for the nonstationary case, where spectral densities are generally not probability measures. To avoid this limitation, we propose regular Fourier features for harmonizable processes with one-dimensional inputs. Our method discretizes the spectral representation directly, preserving the correlation structure among spectral weights without requiring probability assumptions. Assuming finite spectral support, this yields an efficient low-rank approximation that is positive semi-definite by constru
    
[^281]: 一个超大规模视频推理套件

    A Very Big Video Reasoning Suite

    [https://arxiv.org/abs/2602.20159](https://arxiv.org/abs/2602.20159)

    本文介绍了VBVR数据集和VBVR-Bench评估框架，前者规模比现有数据集大三个数量级，后者采用基于规则且与人类对齐的评分器，以系统研究视频推理能力及其扩展行为。

    

    arXiv:2602.20159v3 公告类型：交叉替换 摘要：视频模型的快速进步主要聚焦于视觉质量，而其推理能力仍未得到充分探索。视频推理将智能锚定在超越文本自然捕捉能力的时空一致视觉环境中，能够对连续性、交互和因果性等时空结构进行直觉推理。然而，系统研究视频推理及其扩展行为因缺乏大规模训练数据而受阻。为弥补这一空白，我们引入了超大规模视频推理（VBVR）数据集，这是一个前所未有的超大规模资源，涵盖200个按原则性分类法组织的精选推理任务和超过一百万段视频片段，比现有数据集大约大三个数量级。我们还提出了VBVR-Bench，一个可验证的评估框架，通过结合基于规则且与人类对齐的评分器，超越了基于模型的评判。

    arXiv:2602.20159v3 Announce Type: replace-cross  Abstract: Rapid progress in video models has largely focused on visual quality, leaving their reasoning capabilities underexplored. Video reasoning grounds intelligence in spatiotemporally consistent visual environments that go beyond what text can naturally capture, enabling intuitive reasoning over spatiotemporal structure such as continuity, interaction, and causality. However, systematically studying video reasoning and its scaling behavior is hindered by the lack of large-scale training data. To address this gap, we introduce the Very Big Video Reasoning (VBVR) Dataset, an unprecedentedly large-scale resource spanning 200 curated reasoning tasks following a principled taxonomy and over one million video clips, approximately three orders of magnitude larger than existing datasets. We further present VBVR-Bench, a verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, 
    
[^282]: LORA-CRAFT：通过预训练注意力权重的冻结塔克分解实现跨层秩自适应

    LORA-CRAFT: Cross-layer Rank Adaptation via Frozen Tucker Decomposition of Pre-trained Attention Weights

    [https://arxiv.org/abs/2602.17510](https://arxiv.org/abs/2602.17510)

    CRAFT通过将预训练注意力权重组织为跨层3D张量并应用冻结的塔克分解，仅训练小型方形矩阵，实现了比现有方法更参数高效的微调。

    

    arXiv:2602.17510v2 公告类型：替换-交叉 摘要：我们引入了LoRA-CRAFT（通过冻结塔克分解的跨层秩自适应），全文简称为CRAFT，这是一种极其参数高效的微调（PEFT）方法，它将跨Transformer层堆叠的预训练注意力权重矩阵应用塔克张量分解，并仅在由此产生的冻结塔克因子上训练小型方形自适应矩阵。现有的基于张量的PEFT方法分解梯度更新：LoTR应用带有共享因子矩阵的塔克分解，而SuperLoRA在应用塔克分解前对跨层的ΔW进行分组和重塑。另外，像PiSSA这样的方法对预训练权重应用SVD，但逐层独立操作。CRAFT弥合了这两类工作：它通过高阶SVD（HOSVD）直接对组织为跨层3D张量的预训练权重进行完整塔克分解。

    arXiv:2602.17510v2 Announce Type: replace-cross  Abstract: We introduce LoRA-CRAFT (\textbf{C}ross-layer \textbf{R}ank \textbf{A}daptation via \textbf{F}rozen \textbf{T}ucker), abbreviated CRAFT throughout, an extremely parameter-efficient fine-tuning (PEFT) method that applies Tucker tensor decomposition to pre-trained attention weight matrices stacked across transformer layers and trains only small square adaptation matrices on the resulting frozen Tucker factors. Existing tensor-based PEFT methods decompose \textit{gradient updates}: LoTR applies Tucker decomposition with shared factor matrices, while SuperLoRA groups and reshapes $\Delta W$ across layers before applying Tucker decomposition. Separately, methods such as PiSSA apply SVD to \textit{pre-trained weights} but operate independently per layer. CRAFT bridges these two lines of work: it performs full Tucker decomposition via Higher-Order SVD (HOSVD) directly on \textit{pre-trained weights} organized as cross-layer 3D tensors
    
[^283]: 基于图神经网络学习近似求解均匀设施选址问题

    Learning to Approximate Uniform Facility Location via Graph Neural Networks

    [https://arxiv.org/abs/2602.13155](https://arxiv.org/abs/2602.13155)

    提出了一种融合近似算法原理的完全可微分消息传递神经网络，用于求解均匀设施选址问题，既具有可证明的近似保证，又在实证中优于标准近似算法并缩小了与整数线性规划的差距。

    

    神经网络，特别是消息传递神经网络（MPNN），正日益被用作求解困难组合优化问题的启发式方法。然而，许多基于学习的方法依赖于监督、强化学习或梯度估计器，导致计算成本高、训练不稳定或保证有限。经典近似算法虽然提供最坏情况下的理论保证，但不可微分，且无法适应自然输入分布中的结构。我们通过均匀设施选址问题（UniFL）来研究这种权衡，该问题在聚类、摘要、物流和供应链等领域具有广泛应用。我们提出了一种完全可微分的MPNN，它融合了近似算法的原理，无需求解器监督或离散松弛。该模型具有可证明的近似保证，并在实证中优于标准近似算法，缩小了与整数线性规划之间的差距。

    arXiv:2602.13155v3 Announce Type: replace  Abstract: Neural networks, particularly message-passing neural networks (MPNNs), are increasingly used as heuristics for hard combinatorial optimization problems. Yet many learning-based methods rely on supervision, reinforcement learning, or gradient estimators, causing high computational cost, unstable training, or limited guarantees. Classical approximation algorithms provide worst-case guarantees but are non-differentiable and cannot adapt to structure in natural input distributions. We study this tradeoff through Uniform Facility Location (UniFL), a problem with applications in clustering, summarization, logistics, and supply chains. We propose a fully differentiable MPNN that incorporates approximation-algorithmic principles without solver supervision or discrete relaxations. The model has provable approximation guarantees and empirically improves on standard approximation algorithms, narrowing the gap to integer linear programming.
    
[^284]: 面向语言模型不确定性的语义自蒸馏

    Semantic Self-Distillation for Language Model Uncertainty

    [https://arxiv.org/abs/2602.04577](https://arxiv.org/abs/2602.04577)

    该论文提出语义自蒸馏方法，将语言模型采样答案的语义分布蒸馏到轻量级学生模型中，使其能在生成答案前预测语义分布，利用分布的熵和概率密度分别提供提示级和答案级的不确定性信号，从而以低计算成本实现高效的不确定性估计与幻觉检测。

    

    大型语言模型给原则性的不确定性量化带来了挑战，部分原因在于其复杂性以及输出的多样性。语义离散度（即采样答案在含义上的方差）已被提出作为模型不确定性的有效代理指标，但其相关的计算成本使其无法应用于对延迟敏感的场景。我们证明了采样的语义分布可以被蒸馏到轻量级的学生模型中，该模型能够在语言模型生成答案token之前估计以提示为条件的密度。学生模型预测可能答案的语义分布；该分布的熵提供了提示层面的不确定性信号，而概率密度则支持答案层面的可靠性评估。在TriviaQA和MMLU上的实验表明，我们的学生模型在幻觉检测方面与教师模型的采样语义离散度相比具有竞争力。

    arXiv:2602.04577v3 Announce Type: replace  Abstract: Large language models present challenges for principled uncertainty quantification, in part due to their complexity and the diversity of their outputs. Semantic dispersion, or the variance in the meaning of sampled answers, has been proposed as a useful proxy for model uncertainty, but the associated computational cost prohibits its use in latency-critical applications. We show that sampled semantic distributions can be distilled into lightweight student models which estimate a prompt-conditioned density before the language model generates an answer token. The student model predicts a semantic distribution over possible answers; the entropy of this distribution provides a prompt-level uncertainty signal, and the probability density allows answer-level reliability evaluation. Across experiments on TriviaQA and MMLU, we find our student models perform competitively relative to the teacher's sampled semantic dispersion on a hallucinatio
    
[^285]: 变分贝叶斯流网络用于图生成

    Variational Bayesian Flow Network for Graph Generation

    [https://arxiv.org/abs/2601.22524](https://arxiv.org/abs/2601.22524)

    提出变分贝叶斯流网络（VBFN），通过将贝叶斯更新提升为由结构化精度控制的联合高斯变分信念族，使生成几何能够显式编码节点-边耦合关系，从而提升离散图生成的鲁棒性。

    

    图生成旨在采样离散的节点和边属性，同时满足相互耦合的结构约束。图的扩散模型通常采用高度因子化的前向加噪过程，而许多流匹配方法则从因子化的参考噪声和逐坐标插值出发，因此节点-边之间的耦合关系并未被生成几何结构所编码，而必须由核心网络隐式地恢复，这在离散解码后可能变得脆弱。贝叶斯流网络（BFNs）通过演化分布参数来支持离散生成，但经典的BFNs通常依赖于因子化的信念和相互独立的通道，这限制了几何证据的融合。我们提出变分贝叶斯流网络（VBFN），它对一个由结构化精度矩阵控制的、可处理的联合高斯变分信念族进行变分提升。每一次贝叶斯更新都简化为求解一个对称正定线性系统

    arXiv:2601.22524v2 Announce Type: replace  Abstract: Graph generation aims to sample discrete node and edge attributes while satisfying coupled structural constraints. Diffusion models for graphs often adopt largely factorized forward-noising, and many flow-matching methods start from factorized reference noise and coordinate-wise interpolation, so node-edge coupling is not encoded by the generative geometry and must be recovered implicitly by the core network, which can be brittle after discrete decoding. Bayesian Flow Networks (BFNs) evolve distribution parameters and naturally support discrete generation. But classical BFNs typically rely on factorized beliefs and independent channels, which limit geometric evidence fusion. We propose Variational Bayesian Flow Network (VBFN), which performs a variational lifting to a tractable joint Gaussian variational belief family governed by structured precisions. Each Bayesian update reduces to solving a symmetric positive definite linear syste
    
[^286]: 以观测集合为条件的逆问题：应用与方法

    Inverse Problems Conditioned on Observation Ensembles: Applications and Methods

    [https://arxiv.org/abs/2601.22029](https://arxiv.org/abs/2601.22029)

    该论文提出了一类新的统计问题——集合条件逆问题（EIP），并基于一种利用观测集合信息的新型条件生成模型（集合逆生成模型），给出了非迭代的推理时后验采样方法，可应用于高能物理解折叠、全波形反演和逆成像等领域。

    

    我们引入了一类新的多元统计问题，我们称之为“集合条件逆问题”。EIP的目标是对一个按照先验在前向过程下的推前分布而分布的集合进行反演。在高能物理（HEP）中，这与一个广为人知的问题——解折叠相关，其目标是从被探测器效应扭曲的观测中重建真实的物理分布。EIP也出现在全波形反演（FWI）以及具有未知先验的逆成像问题中。我们提出了一类非迭代的推理时方法，基于一种新的条件生成模型类别来构建后验采样器，我们将其称为集合逆生成模型。在后验建模中，这些模型在单个观测的基础上，还额外利用了观测集合中所包含的集合信息。与现有方法不同，我们提出的方法避免了显式和迭代……

    arXiv:2601.22029v2 Announce Type: replace  Abstract: We introduce a new multivariate statistical problem that we refer to as the Ensemble-conditioned Inverse Problem (EIP). The aim of EIP is to invert for an ensemble that is distributed according to the pushforward of a prior under a forward process. In high energy physics (HEP), this is related to a widely known problem called unfolding, which aims to reconstruct the true physics distribution from observations that are distorted by detector effects. The EIP also arises in full waveform inversion (FWI) and inverse imaging with unknown priors. We propose non-iterative inference-time methods that construct posterior samplers based on a new class of conditional generative models, which we call ensemble inverse generative models. For the posterior modeling, these models additionally use the ensemble information contained in the observation set on top of single observations. Unlike existing methods, our proposed methods avoid explicit and i
    
[^287]: 基于扩散模型的生成式压缩研究进展

    Advances in Diffusion-Based Generative Compression

    [https://arxiv.org/abs/2601.18932](https://arxiv.org/abs/2601.18932)

    本文系统综述了基于扩散模型的生成式有损压缩最新进展，重点介绍了图像压缩中通过嵌入表示编码并利用扩散模型迭代细化、从而在极低码率下实现逼真重建的方法。

    

    扩散模型及其相关的生成建模方法凭借强大的图像生成能力而广受欢迎，并在视觉媒体应用中取得了广泛成功。特别是，扩散方法为数据压缩开辟了新的途径，能够在极低码率下生成逼真的重建结果。本文对近期基于扩散的生成式有损压缩方法进行了统一综述，重点聚焦于图像压缩。这些方法通常将信源编码为嵌入表示，并在解码过程中利用扩散模型对其进行迭代细化，使重建结果近似服从真实数据分布。嵌入表示可以采取多种形式，通常通过辅助熵模型进行传输；近期的方法还探索了利用扩散模型本身通过信道模拟来进行信息传输。我们综述了代表性方法…

    arXiv:2601.18932v2 Announce Type: replace-cross  Abstract: Popularized by their strong image generation performance, diffusion and related methods for generative modeling have found widespread success in visual media applications. In particular, diffusion methods have enabled new approaches to data compression, where realistic reconstructions can be generated at extremely low bit-rates. This article provides a unifying review of recent diffusion-based methods for generative lossy compression, with a focus on image compression. These methods generally encode the source into an embedding and use a diffusion model to iteratively refine it during decoding, so that the reconstruction approximately follows the true data distribution. The embedding can take various forms and is typically transmitted via an auxiliary entropy model, and recent methods also explore the use of diffusion models themselves for information transmission via channel simulation. We review representative approaches thro
    
[^288]: 自我提升即一致性优化：一个理论性解释

    Self-Improvement as Coherence Optimization: A Theoretical Account

    [https://arxiv.org/abs/2601.13566](https://arxiv.org/abs/2601.13566)

    该论文提出统一理论框架，证明辩论、自举与内部一致性最大化等无监督自我提升方法本质上都是“一致性优化”，等价于描述长度正则化，其中基于预训练先验的一致性正则化可优化半监督学习最坏情况准确率的下界，从而在理论上解释了无需反馈的自我提升为何有效。

    

    语言模型能否在缺乏外部监督的情况下提升自身准确率？辩论、自举以及内部一致性最大化等方法实现了这一惊人的成就，甚至可以媲美使用黄金标签的微调性能。然而，这些方法为何有效在理论上仍不清楚。我们证明，它们都可以被理解为“一致性优化”——即寻找最可压缩且可联合预测的“上下文到行为”映射，其中辩论是该优化的一个精确实例，而自举与内部一致性最大化则与之密切相关。我们证明了一致性优化等价于描述长度正则化，并且在所有此类正则化方案中，采用由预训练模型导出的先验的一致性正则化，能够优化半监督学习中最坏情况准确率的一个下界。我们的理论得到了初步实验的支持，解释了无需反馈的自我提升为何有效，并预测了它在何时应当……（原文摘要此处截断）

    arXiv:2601.13566v2 Announce Type: replace-cross  Abstract: Can language models improve their accuracy without external supervision? Methods such as debate, bootstrap, and internal coherence maximization achieve this surprising feat, even matching golden finetuning performance. Yet why they work remains theoretically unclear. We show that they can all be understood as coherence optimization, the search for a context-to-behavior mapping that is most compressible and jointly predictable, with debate an exact instance and bootstrap and internal coherence maximization closely related to it. We prove that coherence optimization is equivalent to description-length regularization, and that among all such regularization schemes, coherence regularization with a prior derived from a pretrained model optimizes a lower bound of worst-case accuracy for semi-supervised learning. Our theory, supported by preliminary experiments, explains why feedback-free self-improvement works and predicts when it sh
    
[^289]: GlyRAG：面向血糖预测的情境感知检索增强框架

    GlyRAG: Context-Aware Retrieval-Augmented Framework for Blood Glucose Forecasting

    [https://arxiv.org/abs/2601.05353](https://arxiv.org/abs/2601.05353)

    提出GlyRAG框架，利用大语言模型作为情境化智能体从CGM数据中提取血糖形态的情境信息，并通过检索增强机制融合相似历史片段，从而提升血糖预测的准确性。

    

    利用连续血糖监测（CGM）数据进行准确的血糖预测可以支持血糖异常风险的早期预警。然而，当前基于神经网络的预测模型将CGM数据视为纯数值序列，未能整合CGM信号形态中蕴含的情境信息。近来，大语言模型（LLM）在时间序列预测方面展现出潜力，但其在糖尿病护理中作为智能体式情境提取器的作用在很大程度上尚未被探索。在本研究中，我们通过开发GlyRAG来连接血糖预测与基于LLM的情境化，GlyRAG是一个情境感知的检索增强预测框架，它使用LLM作为情境化智能体，直接从定时CGM窗口中总结血糖形态。生成的仅基于CGM的叙述文本被嵌入并与基于补丁的血糖表征融合，同时检索模块整合相似的历史训练片段……

    arXiv:2601.05353v3 Announce Type: replace  Abstract: Accurate blood glucose forecasting using continuous glucose monitoring (CGM) data can support the early prediction of dysglycemic risk. However, current neural-network-based forecasting models treat CGM data as a purely numerical sequence without integrating the contextual information contained in CGM signal morphology. Recently, large language models (LLMs) have shown promise for time-series forecasting, yet their role as agentic context extractors in diabetes care remains largely unexplored. In this study, we bridge glucose forecasting and LLM-based contextualization by developing GlyRAG, a context-aware, retrieval-augmented forecasting framework that uses an LLM as a contextualization agent to summarize glucose morphology directly from a timed CGM window. The generated CGM-only narrative is embedded and fused with patch-based glucose representations, while a retrieval module incorporates similar historical training episodes throug
    
[^290]: ASCIIBench：评估基于语言模型的视觉导向文本理解

    ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text

    [https://arxiv.org/abs/2512.04125](https://arxiv.org/abs/2512.04125)

    该论文提出ASCIIBench——首个公开可用的ASCII艺术评测基准，包含5,315张带标签的ASCII图像数据集和一个微调的CLIP模型，用以揭示大语言模型在空间位置推理方面的局限。

    

    大型语言模型（LLM）随着规模的扩大已展现出多种涌现行为，包括推理能力和长篇文本生成的流畅性。然而，它们在需要精确空间和位置推理的任务上仍然表现不佳。ASCII艺术是一种用字符编码结构和形态的符号媒介，为探究这一局限提供了独特的测试途径。我们提出了ASCIIBench，这是一个用于评估ASCII文本图像生成与分类的新型基准。ASCIIBench包含一个经过筛选、含5,315张带类别标签ASCII图像的数据集，据我们所知，这是首个此类公开可用的基准。除数据集外，我们还发布了一个经过微调的CLIP模型的权重，该模型经调整以捕捉ASCII结构，从而能够评估LLM生成的ASCII艺术。我们的分析表明，基于CLIP嵌入的余弦相似度无法区分大多数ASCII类别，即使是低复杂度的图像也只能达到随机水平的表现。

    arXiv:2512.04125v2 Announce Type: replace  Abstract: Large language models (LLMs) have demonstrated several emergent behaviors with scale, including reasoning and fluency in long-form text generation. However, they continue to struggle with tasks requiring precise spatial and positional reasoning. ASCII art, a symbolic medium where characters encode structure and form, provides a unique probe of this limitation. We introduce ASCIIBench, a novel benchmark for evaluating both the generation and classification of ASCII-text images. ASCIIBench consists of a filtered dataset of 5,315 class-labeled ASCII images and is, to our knowledge, the first publicly available benchmark of its kind. Alongside the dataset, we release weights for a fine-tuned CLIP model adapted to capture ASCII structure, enabling the evaluation of LLM-generated ASCII art. Our analysis shows that cosine similarity over CLIP embeddings fails to separate most ASCII categories, yielding chance-level performance even for low-
    
[^291]: 概念瓶颈模型Rashomon切片的参数高效构建方法

    Parameter-Efficient Construction of the Rashomon Slice for Concept Bottleneck Models

    [https://arxiv.org/abs/2511.19636](https://arxiv.org/abs/2511.19636)

    该论文提出了一种参数高效的方法，通过并行适配模块、检查点机制和概念多样性目标，高效探索概念瓶颈模型（CBM）的Rashomon集合，从而以较低成本生成多个精度相当但内部逻辑不同的模型。

    

    在许多机器学习问题中，可能存在多个模型，它们能取得几乎相同的预测性能，但其内部逻辑却存在根本差异。然而，标准训练流程只会产生单一模型，没有为探索可能更适合下游需求的替代方案提供实用的途径。这些精度相当的模型的集合被称为Rashomon集合（罗生门集合）。在庞大而复杂的假设空间中探索Rashomon集合尤其具有挑战性，例如概念瓶颈模型（CBMs），该模型被广泛应用于计算机视觉领域，通过中间的、人类可理解的概念来进行预测。在本文中，我们提出了一种高效探索CBM之Rashomon集合的方法。我们的框架引入了一个专门的并行参数高效适配模块，并结合检查点机制和概念多样性目标，以生成多个精度相当但内部逻辑各异的CBM。

    arXiv:2511.19636v3 Announce Type: replace-cross  Abstract: In many machine learning problems, there may exist multiple models that achieve nearly identical predictive performance while relying on fundamentally different internal logic. However, standard training procedures produce a single model, offering no practical way to explore alternatives that may better suit downstream needs. The set of these equally accurate models is known as the Rashomon set. Exploring the Rashomon set is particularly challenging in large and complex hypothesis spaces, such as Concept Bottleneck Models (CBMs), which are widely used in computer vision to make predictions through intermediate, human-understandable concepts. In this paper, we provide a method for efficiently exploring the Rashomon set of CBMs. Our framework introduces a specialized parallel parameter-efficient adaptation module, combined with a checkpointing scheme and a concept diversity objective, to generate multiple equally accurate CBMs fr
    
[^292]: 先微调，再校正

    Fine-Tune, Then Rectify

    [https://arxiv.org/abs/2511.19486](https://arxiv.org/abs/2511.19486)

    该论文提出一个结合微调与校正的两阶段LLM框架，指出传统微调目标（最小化均方误差）与下游校正阶段不匹配，并创新性地提出以最小化预测误差方差（或标量化方差指标）作为微调目标，同时在两阶段间最优分配有限的标注样本。

    

    受人工智能近期进展的推动，越来越多的文献展示了将大型语言模型（LLMs）作为可扩展代理来生成类人回应的潜力。提升LLM性能的两种常见方法包括：微调，使LLM的输出更贴近人类回应；以及校正，纠正LLM输出中的偏差。本文开发了一个结合微调与校正的两阶段框架，并在两个阶段之间最优地分配有限的标注样本。一个关键洞察是：传统的以最小化均方预测误差为目标的微调目标通常与下游的校正阶段并不一致。对于均值估计问题，我们提出以最小化预测误差的方差作为微调目标；对于一般的M估计问题，我们提出以最小化一个标量化的方差指标作为微调目标。基于这一洞察……

    arXiv:2511.19486v3 Announce Type: replace-cross  Abstract: Driven by recent advances in artificial intelligence, a growing literature has demonstrated the potential of using large language models (LLMs) as scalable surrogates to generate human-like responses. Two common approaches to improve the performance of LLMs include: fine-tuning, which aligns the LLM more closely with human responses, and rectification, which corrects biases in LLM outputs. In this paper, we develop a two-stage framework that combines fine-tuning and rectification, and optimally allocates limited labeled samples across the two stages. A key insight is that the conventional fine-tuning objective of minimizing mean squared prediction error is generally not aligned with the downstream rectification stage. For mean estimation, we propose to minimize the variance of the prediction errors; for general M-estimation, we propose to minimize a scalarized variance metric as the fine-tuning objective. Building on this insig
    
[^293]: 基于参数重要性的基础模型持续学习

    Parameter Importance-Driven Continual Learning for Foundation Models

    [https://arxiv.org/abs/2511.15375](https://arxiv.org/abs/2511.15375)

    提出了一种基于参数重要性估计的持续增强方法PIECE，使基础模型无需访问历史训练数据即可在高效学习领域知识的同时保持通用推理能力。

    

    领域特定的后训练常常导致灾难性遗忘，使基础模型失去其通用推理能力，并限制了其对动态真实环境的适应性。在获取下游领域知识的同时保持通用能力，是大语言模型和多模态模型面临的核心挑战。传统的持续学习方法，如正则化、回放和架构隔离，存在下游性能差、依赖无法获取的历史数据或额外参数开销等问题。尽管近期的参数高效微调（PET）方法可以缓解遗忘，但其有效性在很大程度上取决于参数选择和更新策略。在本文中，我们提出了PIECE，一种基于参数重要性估计的持续增强方法，它能在不访问先前训练数据的情况下，在保持通用能力的同时高效学习领域知识。

    arXiv:2511.15375v2 Announce Type: replace-cross  Abstract: Domain-specific post-training often causes catastrophic forgetting, making foundation models lose their general reasoning ability and limiting their adaptability to dynamic real-world environments. Preserving general capabilities while acquiring downstream domain knowledge is a central challenge for large language and multimodal models. Traditional continual learning methods, such as regularization, replay and architectural isolation, suffer from poor downstream performance, reliance on inaccessible historical data, or additional parameter overhead. While recent parameter-efficient tuning (PET) methods can alleviate forgetting, their effectiveness strongly depends on the choice of parameters and update strategies. In this paper, we introduce PIECE, a Parameter Importance Estimation-based Continual Enhancement method that preserves general ability while efficiently learning domain knowledge without accessing prior training data 
    
[^294]: 无需以牺牲未来为代价的优化？基于集体学习与强化学习的去中心化协调

    Optimization without Future Compromises? Decentralized Coordination via Collective and Reinforcement Learning

    [https://arxiv.org/abs/2509.18088](https://arxiv.org/abs/2509.18088)

    提出分层强化与集体学习（HRCL）框架，用MARL从全局层面引导而非取代去中心化多智能体协调，在大规模长期资源分配中兼顾当前效率与未来性能，同时避免决策空间爆炸和训练低效。

    

    多智能体系统中的高效资源分配要求自主智能体协调各自的决策，同时权衡系统全局目标与个体成本。在长时间跨度下，这一问题变得愈发具有挑战性：改善当前分配的决策可能会损害未来的资源分配，而去中心化的智能体对整个系统的观测能力有限。多智能体强化学习（MARL）能够通过局部观测学习这种长期依赖关系，但直接将其应用于大规模协调会导致决策空间迅速膨胀和训练效率低下。为此，我们提出了分层强化与集体学习（HRCL），这是一个分层框架，它利用MARL来引导而非取代去中心化的多智能体协调。在高层次上，MARL学习那些限制协调过程中所考虑的备选方案并引导智能体……的策略（原文摘要在此处截断）

    arXiv:2509.18088v2 Announce Type: replace-cross  Abstract: Efficient resource allocation in multi-agent systems requires autonomous agents to coordinate their decisions while balancing system-wide objectives with individual costs. This becomes increasingly challenging over long time horizons, where decisions that improve the current allocation may compromise future resource allocation, while decentralized agents have limited observations of the overall system. Multi-agent reinforcement learning (MARL) can learn such long-term dependencies via local observations, but directly applying it to large-scale coordination leads to rapidly growing decision spaces and inefficient training. To this end, we propose Hierarchical Reinforcement and Collective Learning (HRCL), a hierarchical framework that uses MARL to guide, rather than replace, decentralized multi-agent coordination. At the high level, MARL learns strategies that restrict the alternatives considered during coordination and guide age
    
[^295]: 用于检测和纠正大语言模型幻觉的几何不确定性方法

    Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs

    [https://arxiv.org/abs/2509.13813](https://arxiv.org/abs/2509.13813)

    该论文提出了一个黑盒几何框架，通过在答案嵌入空间中建模以提示为条件的语义分布，同时量化提示和答案两个层面的不确定性，从而实现对大语言模型幻觉的检测与纠正。

    

    大语言模型已知会产生幻觉，即对问题生成语言上看似合理但实际不正确的答案。不确定性量化已被提出作为检测此类行为的策略，但现有方法缺乏一个统一的框架来同时评估提示层面和答案层面的可靠性。我们引入了一个几何框架，通过在答案嵌入空间中显式建模以提示为条件的语义分布，在两个层面量化语言模型的不确定性。我们的方法是黑盒且基于采样的：我们为每个提示生成多个答案，并使用原型分析来估计答案分布的几何支撑。在提示层面，我们通过近似分布熵来量化不确定性；对于每个单独的答案，我们随后使用非典型性的概念来评估其相对于整个批次的可靠性。我们利用该框架不仅能够检测幻觉，还能纠正幻觉。

    arXiv:2509.13813v3 Announce Type: replace  Abstract: Large language models are known to hallucinate, generating linguistically plausible but incorrect answers to questions. Uncertainty quantification has been proposed as a strategy to detect such behaviour, but existing methods lack a unified framework to assess reliability at both the prompt and answer level. We introduce a geometric framework which quantifies language model uncertainty at both levels by explicitly modelling a prompt-conditioned semantic distribution in answer embedding space. Our approach is black-box and sampling-based; we generate multiple answers per prompt, and use archetypal analysis to estimate a geometric support for the answer distribution. At the prompt level, we approximate the distribution entropy to quantify uncertainty; for each individual answer, we then use notions of atypicality to assess its reliability relative to the batch. We employ our framework to not only detect hallucinations but correct them,
    
[^296]: 一种基于差异度量的数据集浓缩视角

    A Discrepancy-Based Perspective on Dataset Condensation

    [https://arxiv.org/abs/2509.10367](https://arxiv.org/abs/2509.10367)

    本文提出一个基于差异度量的统一框架，将数据集浓缩重新形式化为概率分布逼近问题，从而把 DC 的目标从泛化性能扩展到更一般的任务设定。

    

    给定一个由有限个元素组成的数据集 $\mathcal{T} = \{\mathbf{x}_i\}_{i = 1}^N$，数据集浓缩（DC）的目标是构建一个规模显著更小（$M \ll N$）的合成数据集 $\mathcal{S} = \{\tilde{\mathbf{x}}_j\}_{j = 1}^M$，使得在 $\mathcal{S}$ 上从头训练的模型能够获得与在 $\mathcal{T}$ 上训练的模型相当甚至更优的泛化性能。DC 领域的最新进展揭示了其与“用一个缩减的点集逼近 $\mathcal{T}$ 所表示的数据分布”这一问题的紧密联系。在本工作中，我们提出了一个涵盖现有 DC 方法的统一框架，并借助差异度量的概念——它能在不同范畴下量化概率分布之间的距离——将任务特定的 DC 概念扩展为更通用、更正式的定义。我们的框架将 DC 的目标从泛化性能进一步拓宽，以容纳更多的优化目标。

    arXiv:2509.10367v2 Announce Type: replace  Abstract: Given a dataset of finitely many elements $\mathcal{T} = \{\mathbf{x}_i\}_{i = 1}^N$, the goal of dataset condensation (DC) is to construct a synthetic dataset $\mathcal{S} = \{\tilde{\mathbf{x}}_j\}_{j = 1}^M$ which is significantly smaller ($M \ll N$) such that a model trained from scratch on $\mathcal{S}$ achieves comparable or even superior generalization performance to a model trained on $\mathcal{T}$. Recent advances in DC reveal a close connection to the problem of approximating the data distribution represented by $\mathcal{T}$ with a reduced set of points. In this work, we present a unified framework that encompasses existing DC methods and extend the task-specific notion of DC to a more general and formal definition using notions of discrepancy, which quantify the distance between probability distribution in different regimes. Our framework broadens the objective of DC beyond generalization, accommodating additional objecti
    
[^297]: 面向中小企业异构信贷数据分析的集成多元分割树

    Integrated Multivariate Segmentation Tree for Heterogeneous Credit Data Analysis in Small- and Medium-Sized Enterprises

    [https://arxiv.org/abs/2509.00550](https://arxiv.org/abs/2509.00550)

    本文提出集成多元分割树（IMST）框架，通过矩阵分解、Lasso特征选择与多元分割树构建，将财务数据与文本信息有效融合，将中小企业信用评估准确率提升至88.9%。

    

    传统决策树模型仅依赖数值变量，在处理高维数据时常常面临挑战，并且在有效融入文本信息方面能力有限。为解决这些局限性，我们提出了集成多元分割树（IMST），这是一个综合性框架，旨在通过整合财务数据与文本数据源来改进中小企业的信用评估。该方法包含三个核心阶段：（1）通过矩阵分解将文本数据转换为数值矩阵；（2）使用Lasso回归筛选重要的财务特征；（3）基于基尼指数或熵构建多元分割树，并采用最弱连接剪枝来控制模型复杂度。基于1,428家中国中小企业数据集的实验结果表明，IMST达到了88.9%的准确率。

    arXiv:2509.00550v3 Announce Type: replace  Abstract: Traditional decision tree models, which rely exclusively on numerical variables, often face challenges in handling high-dimensional data and are limited in their ability to incorporate textual information effectively. To address these limitations, we propose the integrated multivariate segmentation tree (IMST), a comprehensive framework designed to improve credit evaluation for small- and medium-sized enterprises (SMEs) by integrating financial data with textual sources. This method comprises three core stages: (1) transforming textual data into numerical matrices through matrix factorization, (2) selecting salient financial features using Lasso regression, and (3) constructing a multivariate segmentation tree based on either the Gini index or entropy, with weakest-link pruning applied to control model complexity. Experimental results based on a dataset of 1,428 Chinese SMEs demonstrated that IMST achieved an accuracy rate of 88.9%, 
    
[^298]: VMMU：越南语多任务多模态理解与推理基准

    VMMU: A Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark

    [https://arxiv.org/abs/2508.13680](https://arxiv.org/abs/2508.13680)

    VMMU是首个越南语多任务多模态理解与推理基准，包含2500个跨7个任务的多模态问题，评估显示尽管最先进专有视觉-语言模型的越南语OCR性能良好，其平均准确率仅达66%，主要瓶颈在于多模态定位与推理能力而非OCR。

    

    我们提出了VMMU，一个越南语多任务多模态理解与推理基准，旨在评估视觉-语言模型（VLMs）在英语之外如何解释和推理视觉与文本信息。VMMU包含跨7个任务的2500个多模态问题，涵盖多样化的问题情境，包括STEM问题求解、数据解释、规则约束的视觉推理和抽象视觉推理。所有问题都需要真正的多模态整合，而非依赖纯文本线索或基于OCR的捷径。我们在VMMU上评估了多种最先进的专有和开源视觉-语言模型。尽管专有模型在越南语OCR方面表现出色，但其平均准确率仅为66%。进一步分析表明，失败的主要来源并非OCR，而是多模态定位以及对文本和视觉证据的推理能力不足。代码和数据可在 https://vmmu-bench.github.io/ 获取。

    arXiv:2508.13680v5 Announce Type: replace  Abstract: We introduce VMMU, a Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark designed to evaluate how vision-language models (VLMs) interpret and reason over visual and textual information beyond English. VMMU consists of 2.5k multimodal questions across 7 tasks, covering a diverse range of problem contexts, including STEM problem solving, data interpretation, rule-governed visual reasoning, and abstract visual reasoning. All questions require genuine multimodal integration, rather than reliance on text-only cues or OCR-based shortcuts. We evaluate a diverse set of state-of-the-art proprietary and open-source VLMs on VMMU. Despite strong Vietnamese OCR performance, proprietary models achieve only 66% mean accuracy. Further analysis shows that the primary source of failure is not OCR, but instead multimodal grounding and reasoning over text and visual evidence. Code and data are available at https://vmmu-bench.github.io/
    
[^299]: 自动新生儿癫痫发作检测的诚实可靠评估与专家等同性测试

    Honest and Reliable Evaluation and Expert Equivalence Testing of Automated Neonatal Seizure Detection

    [https://arxiv.org/abs/2508.04899](https://arxiv.org/abs/2508.04899)

    本研究系统评估了新生儿癫痫发作检测的性能指标与共识策略，发现Matthews和Pearson相关系数在类别不平衡下优于AUC，并提出了严格的专家等同性测试方法，为AI达到专家水平的声明建立了诚实可靠的评估框架。

    

    机器学习模型在新生儿癫痫发作检测中的可靠评估对临床应用至关重要。目前的实践往往依赖不一致且有偏见的指标，阻碍了模型的可比性和可解释性。关于AI性能达到专家水平的声明经常在缺乏严格验证的情况下提出，引发了对其可靠性的担忧。本研究旨在系统评估常见的性能指标，并针对新生儿癫痫发作检测的特定挑战提出最佳实践。利用真实和合成的癫痫发作标注，我们在不同的类别不平衡程度、评分者间一致性以及评分者数量条件下，评估了标准性能指标、共识策略和人类专家水平等同性测试。结果显示，Matthews相关系数和Pearson相关系数在类别不平衡条件下反映性能方面优于曲线下面积（AUC）。共识类型对评分者数量和评分者间一致性较为敏感。

    arXiv:2508.04899v3 Announce Type: replace  Abstract: Reliable evaluation of machine learning models for neonatal seizure detection is critical for clinical adoption. Current practices often rely on inconsistent and biased metrics, hindering model comparability and interpretability. Expert-level claims about AI performance are frequently made without rigorous validation, raising concerns about their reliability. This study aims to systematically evaluate common performance metrics and propose best practices tailored to the specific challenges of neonatal seizure detection. Using real and synthetic seizure annotations, we assessed standard performance metrics, consensus strategies, and human-expert level equivalence tests under varying class imbalance, inter-rater agreement, and number of raters. Matthews and Pearson's correlation coefficients outperformed the area under the curve in reflecting performance under class imbalance. Consensus types are sensitive to the number of raters and a
    
[^300]: 利用自然语言处理技术进行保险科技创新

    InsurTech innovation using natural language processing

    [https://arxiv.org/abs/2507.21112](https://arxiv.org/abs/2507.21112)

    本文展示了如何运用自然语言处理技术将非结构化文本转化为结构化数据，通过特征去偏、特征压缩和行业分类来丰富商业保险定价的费率因子，并为评估潜在风险提供新视角。

    

    随着保险科技（InsurTech）的迅速崛起，传统保险公司日益探索替代数据源和先进技术以保持其竞争优势。本文对自然语言处理（NLP）及其在保险运营中的新兴应用提供了概念性概述和实际案例研究，重点是将原始的非结构化文本转化为适合精算分析和决策的结构化数据。利用由保险科技行业合作伙伴提供的、能够丰富传统保险数据源的真实世界替代数据，我们应用多种NLP技术在商业保险场景中展示了特征去偏、特征压缩和行业分类。这些丰富的、源自文本的洞察不仅补充和优化了商业保险定价的传统费率因子，还为评估潜在风险提供了新颖的视角。

    arXiv:2507.21112v4 Announce Type: replace  Abstract: With the rapid rise of InsurTech, traditional insurance companies are increasingly exploring alternative data sources and advanced technologies to sustain their competitive edge. This paper provides both a conceptual overview and practical case studies of natural language processing (NLP) and its emerging applications within insurance operations, focusing on transforming raw, unstructured text into structured data suitable for actuarial analysis and decision-making. Leveraging real-world alternative data provided by an InsurTech industry partner that enriches traditional insurance data sources, we apply various NLP techniques to demonstrate feature de-biasing, feature compression, and industry classification in the commercial insurance context. These enriched, text-derived insights not only add to and refine traditional rating factors for commercial insurance pricing but also offer novel perspectives for assessing underlying risk by 
    
[^301]: 基于无伪影B-cos网络的忠实、可解释胸部X光诊断

    Faithful, Interpretable Chest X-ray Diagnosis with Artifact-free B-cos Networks

    [https://arxiv.org/abs/2507.16761](https://arxiv.org/abs/2507.16761)

    该论文通过在B-cos网络中引入ASAP和BlurPool抗混叠策略，消除了胸部X光诊断解释图中的混叠伪影，在保持强大诊断性能的同时实现了忠实、清晰、可解释的临床级诊断。

    

    忠实性和可解释性对于在医学图像分析等安全关键领域部署深度神经网络（DNN）至关重要。B-cos网络通过修改卷积层和分类层的参数化方式，利用特征-权重对齐来测量类别证据，从而无需事后解释即可生成内置的、特定类别的贡献图。尽管标准B-cos网络在保持与最先进DNN相当的诊断性能的同时，其解释图中会出现严重的混叠伪影，使其不适合对清晰度要求极高的临床应用。在本工作中，我们通过引入基于ASAP和BlurPool（BP）的抗混叠策略来解决这一局限，显著提高了解释质量。我们在胸部X光数据集上的实验表明，改进后的B-cos_ASAP和B-cos_BP在保持强大预测性能的同时显著提升了解释图质量。

    arXiv:2507.16761v3 Announce Type: replace-cross  Abstract: Faithfulness and interpretability are essential for deploying deep neural networks (DNNs) in safety-critical domains such as medical image analysis. B-cos networks modify the parameterization of convolutional and classification layers to measure class evidence via feature-weight alignment, enabling built-in, class-specific contribution maps without post-hoc explanations. While maintaining diagnostic performance competitive with state-of-the-art DNNs, standard B-cos networks exhibit severe aliasing artifacts in their explanation maps, rendering them unsuitable for clinical use, where clarity is essential. In this work, we address this limitation by introducing anti-aliasing strategies using ASAP and BlurPool (BP) to significantly improve explanation quality. Our experiments on chest X-ray datasets demonstrate that the modified $\text{B-cos}_\mathrm{ASAP}$ and $\text{B-cos}_\mathrm{BP}$ preserve strong predictive performance whil
    
[^302]: AdaDim：面向自监督学习表征动力学的维度自适应方法

    AdaDim: Dimensionality Adaptation for SSL Representational Dynamics

    [https://arxiv.org/abs/2505.12576](https://arxiv.org/abs/2505.12576)

    该论文提出 AdaDim 方法，在自监督学习训练过程中自适应地调控表示的维度动态，兼顾高有效维度 H(R) 与低互信息 I(R;Z)，以防止维度坍缩并提升下游任务的泛化性能。

    

    自监督学习（SSL）有效性的一个关键因素是防止维度坍缩，即高维表示空间（R）实际张成的却是较低维的子空间。因此，SSL 的优化策略之一是通过鼓励特征去相关或样本在 R 中的均匀分布等目标函数，引导模型产生具有更高维度的 R（记为 H(R)）。更高的 H(R) 表明 R 具有更大的特征多样性，这有利于向下游任务的泛化。除了维度优化之外，SSL 算法还利用投影头将 R 映射到嵌入空间 Z。近期的研究将投影头刻画为一个滤波器，通过降低互信息 I(R;Z) 来滤除 SSL 目标中的噪声或无关特征。因此，当前文献的观点是：一个好的 SSL 表示空间应当同时具有高 H(R) 和低 I(R;Z)。然而，这一观点……

    arXiv:2505.12576v3 Announce Type: replace-cross  Abstract: A key factor in effective Self-Supervised learning (SSL) is preventing dimensional collapse, where higher-dimensional representation spaces ($R$) span a lower-dimensional subspace. Therefore, SSL optimization strategies involve guiding a model to produce $R$ with a higher dimensionality ($H(R)$) through objectives that encourage decorrelation of features or sample uniformity in $R$. A higher $H(R)$ indicates that $R$ has greater feature diversity which is useful for generalization to downstream tasks. Alongside dimensionality optimization, SSL algorithms also utilize a projection head that maps $R$ into an embedding space $Z$. Recent work has characterized the projection head as a filter of noisy or irrelevant features from the SSL objective by reducing the mutual information $I(R;Z)$. Therefore, the current literature's view is that a good SSL representation space should have a high $H(R)$ and a low $I(R;Z)$. However, this vie
    
[^303]: ChronoSteer：通过合成跨模态对齐数据集连接大语言模型与时间序列基础模型

    ChronoSteer: Bridging Large Language Model and Time Series Foundation Model via Synthetic Cross-Modal Alignment Dataset

    [https://arxiv.org/abs/2505.10083](https://arxiv.org/abs/2505.10083)

    提出ChronoSteer——一个解耦的智能体框架，通过合成跨模态对齐数据集将大语言模型与时间序列基础模型相连接，构建出能联合利用时间与文本信息进行零样本预测的多模态时间序列基础模型。

    

    传统的预测方法在单模态时间序列上进行端到端训练，这限制了其利用文本信息的能力，并削弱了其在数据稀缺场景下的泛化性能。近年来，大语言模型（LLM）和时间序列基础模型（TSFM）分别在复杂文本推理和零样本时间建模方面展现出强大的能力。整合两者的优势，构建一个能够联合利用时间与文本信息进行零样本未来推断的多模态时间序列基础模型，已成为一个颇具前景的研究方向。然而，大规模、高质量多模态数据集的稀缺仍然是一个根本性障碍。为应对这一挑战，我们提出了ChronoSteer，这是一种解耦的智能体框架，能够从合成的成对监督中学习跨模态对齐。具体而言，预训练的大语言模型首先将文本事件转换为修订指令……（原文摘要在此处截断）

    arXiv:2505.10083v2 Announce Type: replace  Abstract: Conventional forecasting methods are trained end-to-end on unimodal time series, which limits their ability to exploit textual information and undermines their generalization in data-scarce scenarios. Recently, large language models (LLMs) and time series foundation models (TSFMs) have demonstrated powerful capabilities in complex textual reasoning and zero-shot temporal modeling, respectively. Integrating these strengths to construct a multimodal time series foundation model that jointly leverages temporal and textual information for zero-shot future inference has emerged as a promising research direction. However, the scarcity of large-scale, high-quality multimodal datasets remains a fundamental obstacle. To address this challenge, we propose ChronoSteer, a decoupled agentic framework that learns cross-modal alignment from synthetic paired supervision. Specifically, a pretrained LLM first converts textual events into revision inst
    
[^304]: 加扰与噪声在量子系统时间信息处理中的作用

    Role of scrambling and noise in temporal information processing with quantum systems

    [https://arxiv.org/abs/2505.10080](https://arxiv.org/abs/2505.10080)

    本文揭示了基于高阶幺正设计的加扰量子储层在时间信息处理中的关键特性：无噪声时测量读出集中度不随迭代恶化、小储层可反复复用，但扩大规模需指数级测量开销否则损害泛化，且早期输入记忆随储层规模与迭代次数均呈指数衰减。

    

    加扰量子系统作为时间信息处理的有效基底引起了广泛关注。本文考虑了一个量子储层计算框架，该框架涵盖了利用量子系统的广泛物理计算模型。我们研究了以高阶幺正设计建模的加扰储层在无噪声和有噪声两种设置下模型的可扩展性与记忆保持能力。在无噪声情形下，我们证明测量读出会随储层规模增大而呈指数级集中，但令人惊讶的是，其并不会随储层迭代次数的增加而恶化。因此，尽管对量子数据反复复用小型加扰储层可能是可行的，但扩大问题规模会使泛化能力退化，除非能够承担指数级的测量次数开销。相比之下，早期输入和初始态的记忆随储层规模和储层迭代次数均呈指数衰减。在有噪声……

    arXiv:2505.10080v3 Announce Type: replace-cross  Abstract: Scrambling quantum systems have attracted attention as effective substrates for temporal information processing. Here we consider a quantum reservoir processing framework that captures a broad range of physical computing models with quantum systems. We examine the scalability and memory retention of the model with scrambling reservoirs modelled by high-order unitary designs in both noiseless and noisy settings. In the former regime, we show that measurement readouts become exponentially concentrated with increasing reservoir size, yet strikingly do not worsen with the reservoir iterations. Thus, while repeatedly reusing a small scrambling reservoir with quantum data might be viable, scaling up the problem size deteriorates generalization unless one can afford an exponential shot overhead. In contrast, the memory of early inputs and initial states decays exponentially in both reservoir size and reservoir iterations. In the noisy
    
[^305]: 局部化扩散模型

    Localized Diffusion Models

    [https://arxiv.org/abs/2505.04417](https://arxiv.org/abs/2505.04417)

    提出局部化扩散模型，通过利用目标分布中的局部性结构（稀疏条件依赖），以局部化神经网络估计得分函数，从而规避维数灾难并显著降低样本复杂度。

    

    扩散模型是各种生成任务中最先进的工具。然而，训练这些模型需要估计高维得分函数，这一任务在原则上会受到维数灾难的影响。因此，理解如何在这些模型中利用目标分布的低维结构十分重要。本文考虑局部性结构，它描述了目标随机变量之间的某些稀疏条件依赖关系。给定某种局部性结构，得分函数实际上是低维的，因此可以通过局部化的神经网络进行估计，从而显著降低样本复杂度。这一观察启发了局部化扩散模型，即使用局部化得分匹配损失在局部化假设空间内训练得分函数。我们证明这种局部化使扩散模型能够以与维度无关的方式规避维数灾难。

    arXiv:2505.04417v3 Announce Type: replace  Abstract: Diffusion models are state-of-the-art tools for various generative tasks. Yet training these models involves estimating high-dimensional score functions, a task that in principle suffers from the curse of dimensionality. It is therefore important to understand how low-dimensional structure in the target distribution can be exploited in these models. Here we consider locality structure, which describes certain sparse conditional dependencies among the target random variables. Given some locality structure, the score function is effectively low-dimensional, so that it can be estimated by a localized neural network with significantly reduced sample complexity. This observation motivates the localized diffusion model, where a localized score matching loss is used to train the score function within a localized hypothesis space. We prove that such localization enables diffusion models to circumvent the curse of dimensionality with dimensio
    
[^306]: 路径正则化：多层神经网络的近乎完备且最优的非渐近泛化理论与双重下降现象

    Path Regularization: A Near-Complete and Optimal Nonasymptotic Generalization Theory for Multilayer Neural Networks and Double Descent Phenomenon

    [https://arxiv.org/abs/2503.02129](https://arxiv.org/abs/2503.02129)

    该论文首次提出了路径正则化多层神经网络的近乎完备且最优的非渐近泛化理论，给出了显式泛化误差上界，无需损失函数有界及网络宽度、深度等常见假设，超越了偏差-方差权衡并能解释深度学习中的双重下降现象。

    

    路径正则化已被证明是训练神经网络的一种非常有效的正则化方法，与权重衰减等常见正则化方法相比，它能带来更好的泛化性能。我们针对一般学习问题，首次提出了带有路径正则化的多层神经网络的近乎完备的（将在正文中明确说明）非渐近泛化理论。特别地，该理论不要求损失函数有界，而这在现有文献中通常是必备假设。我们的理论超越了偏差-方差权衡，并与深度学习中常见的现象相吻合，因此与其他现有的非渐近泛化误差界有显著不同。更具体地说，我们为满足 $\sigma(0)=0$ 且损失函数为足够广泛的Lipschitz函数的多层神经网络提出了显式的泛化误差上界，而无需对网络的宽度、深度或其他超参数作出限制……

    arXiv:2503.02129v3 Announce Type: replace-cross  Abstract: Path regularization has shown to be a very effective regularization to train neural networks, leading to a better generalization property than common regularizations i.e. weight decay, etc. We propose a first near-complete (as will be made explicit in the main text) nonasymptotic generalization theory for multilayer neural networks with path regularizations for general learning problems. In particular, it does not require the boundedness of the loss function, as is commonly assumed in the literature. Our theory goes beyond the bias-variance tradeoff and aligns with phenomena typically encountered in deep learning. It is therefore sharply different from other existing nonasymptotic generalization error bounds. More explicitly, we propose an explicit generalization error upper bound for multilayer neural networks with $\sigma(0)=0$ and sufficiently broad Lipschitz loss functions, without requiring the width, depth, or other hyper
    
[^307]: 依赖数据下深度神经网络的统计性质

    Statistical Properties of Deep Neural Networks with Dependent Data

    [https://arxiv.org/abs/2410.11113](https://arxiv.org/abs/2410.11113)

    该论文为非平稳β-混合依赖数据下的深度神经网络估计量建立了非渐近误差理论，覆盖全连接与卷积网络且无需对权重施加有界或稀疏约束，并推广至非参数回归、逻辑回归和分位数回归等场景。

    

    本文为依赖数据下的深度神经网络（DNN）估计量建立了理论。为了提供适用于各类基于DNN的估计量的理论，我首先在可能非平稳的、取值于无界集合的β-混合数据条件下，针对一类一般性的估计问题，建立了非参数筛（sieve）估计量的理论误差和经验L²误差的非渐近概率界。随后，我将该理论应用于全连接和卷积DNN估计量，且无需对DNN权重施加有界性或稀疏性限制。对于这两类DNN，当待估计函数为Hölder光滑、数据为非平稳、次高斯且具有指数或多项式衰减的β-混合时，我推导了一般性结果。接着，我将这些结果具体应用于非参数回归、逻辑回归和分位数回归的设定。在指数β-混合条件下，所得估计量能够达到（原文在此处截断）……

    arXiv:2410.11113v4 Announce Type: replace-cross  Abstract: This paper develops theory for deep neural network (DNN) estimators under dependent data. To provide theory applicable to a variety of DNN-based estimators, I first establish nonasymptotic probability bounds on the theoretical and empirical $\mathcal{L}^{2}$-errors of nonparametric sieve estimators for a general class of estimation problems under possibly nonstationary $\beta$-mixing data taking values in unbounded sets. I then apply the theory to fully connected and convolutional DNN estimators without bounds or sparsity restrictions on the DNN weights. For both DNN classes, I derive general results when the function to be estimated is H\"older smooth and the data are nonstationary, subgaussian, and $\beta$-mixing with either exponential or polynomial decay. I then specialize these to nonparametric regression, logistic regression, and quantile regression settings. Under exponential $\beta$-mixing, the resulting estimators atta
    
[^308]: FastManly：用于Manly混合模型的EM梯度算法

    FastManly: An EM-Gradient Algorithm for Manly Mixture Models

    [https://arxiv.org/abs/2410.00848](https://arxiv.org/abs/2410.00848)

    FastManly方法通过在EM梯度算法中采用牛顿法（并推导出梯度和完整Hessian矩阵）替代传统EM中的Nelder-Mead优化，显著加快了Manly变换混合模型的计算速度。

    

    本文提出了一种更快速的Manly变换混合模型实现方法。该方法称为FastManly，在EM梯度算法中使用牛顿法进行优化，取代了传统EM算法中的Nelder-Mead方法。文中推导出了梯度以及完整的Hessian矩阵。仿真结果表明该方法性能更优，且速度提升显著。

    arXiv:2410.00848v2 Announce Type: replace-cross  Abstract: A faster implementation of mixtures of Manly transformations is proposed. This method, called FastManly, uses Newton's method for optimization in an EM gradient algorithm instead of Nelder-Mead in a traditional EM. A gradient and full Hessian are derived. Simulations show improved performance with noticeable speedups.
    
[^309]: 独立Metropolis采样的方差缩减

    Variance Reduction for Independent Metropolis

    [https://arxiv.org/abs/2406.17699](https://arxiv.org/abs/2406.17699)

    本文证明当目标密度与提议密度在KL散度下足够接近时，独立Metropolis采样器结合基于控制变量、无需额外计算成本的方差缩减策略，可获得比独立同分布采样更小的渐近方差。

    

    假设我们希望估计一个函数 $F$ 关于某个难以处理的密度 $\pi$ 的期望值，该密度由某个未知的归一化常数确定（即仅知道非归一化形式）。我们证明，如果 $\pi$ 在KL散度意义下与另一个密度 $q$ 足够接近，那么使用提议密度 $q$ 从 $\pi$ 中获取样本的独立Metropolis采样器估计器，在结合基于控制变量的方差缩减计算策略后，能够实现比从 $\pi$ 进行独立同分布（i.i.d.）采样更小的渐近方差。控制变量的构建不需要额外的计算量，但前提是 $F$ 在 $q$ 下的期望值能够解析地获得。我们通过在一个存在先验-似然冲突且使用非共轭先验的线性回归模型中计算边际似然来演示这一结果。此外，我们提出了一种自适应独立Metropolis算法，该算法能够自适应地调整提议密度使其……

    arXiv:2406.17699v3 Announce Type: replace-cross  Abstract: Assume that we would like to estimate the expected value of a function $F$ with respect to an intractable density $\pi$, which is specified up to some unknown normalising constant. We prove that if $\pi$ is close enough under KL divergence to another density $q$, an independent Metropolis sampler estimator that obtains samples from $\pi$ with proposal density $q$, enriched with a variance reduction computational strategy based on control variates, achieves smaller asymptotic variance than i.i.d. sampling from $\pi$. The control variates construction requires no extra computational effort but assumes that the expected value of $F$ under $q$ is analytically available. We illustrate this result by calculating the marginal likelihood in a linear regression model with prior-likelihood conflict and a non-conjugate prior. Furthermore, we propose an adaptive independent Metropolis algorithm that adapts the proposal density such that it
    
[^310]: 随机多面体描述符

    Random Polytope Descriptors

    [https://arxiv.org/abs/2009.13987](https://arxiv.org/abs/2009.13987)

    该论文提出了一类既通用又计算友好的随机多面体描述符，可用于数据分析中的分类与聚类任务，并允许用户在数据描述的紧致性与计算速度之间灵活权衡。

    

    我们引入了一类随机多面体，它同时推广了多种已知的构造方法。这类多面体不仅相当通用，而且在计算上也异常友好。我们说明了如何利用这些性质来完成数据分析中的分类与聚类任务。至关重要的是，我们的构造让用户能够在更紧凑的数据描述与更快的计算之间平滑地权衡。

    arXiv:2009.13987v3 Announce Type: replace  Abstract: We introduce a class of random polytopes which simultaneously generalizes several known constructions. While being fairly general, these polytopes are also computationally exceptionally benign. We indicate how these properties can be exploited for classification and clustering tasks in data analysis. Crucially, our construction lets users smoothly trade off between a tighter description of the data and faster computation.
    

