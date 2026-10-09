# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CSF: Contextual Safety Filtering for Motion Generators](https://arxiv.org/abs/2610.12467) | 提出免训练的上下文安全过滤框架CSF，将自然语言安全规则通过安全/不安全参考轨迹转化为CBF-QP约束，使文本条件运动生成器在场景触发的不安全情况下危险事件率降低高达90%，同时保留88-100%的良性运动。 |
| [^2] | [A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control](https://arxiv.org/abs/2610.12465) | 该论文指出对仿真器重置进行均匀采样会将大量学习经验浪费在已掌握或无法尝试的任务配置上，从而提出均衡的数据采样策略，以解决超大规模并行强化学习在机器人控制中的探索瓶颈问题。 |
| [^3] | [Bi-FORK: Generative Modeling of High-Dimensional Bifurcating Systems](https://arxiv.org/abs/2610.12449) | Bi-FORK是一个生成式框架，通过潜空间流匹配和排斥引导采样，学习高维分岔系统中一对多的解映射，能够在屈曲、超材料和相分离等物理问题中高效恢复多模态的完整解分支。 |
| [^4] | [One Block, Multiple Depths: Recurrent Vision Transformers with Depth-Programmed Experts](https://arxiv.org/abs/2610.12448) | 提出reViT，通过循环复用单个Transformer块，并将各深度的FFN表示为由连续深度坐标编程的共享专家凸组合，在相当的推理计算量下以约70%更少的参数达到与全深度视觉编码器相当的精度。 |
| [^5] | [Caught in the Act: Probes Effectively Detect Sabotage and Catch Unverbalized Deception](https://arxiv.org/abs/2610.12445) | 该研究通过构建迄今最大的欺骗数据集并设计可跨层、跨标记聚合信息的新型探针架构，使白盒探针在 SHADE-Arena 上以 98.8% 的 AUC 超越前沿文本监控基线，甚至能检测出仅凭上下文无法察觉的“内省式欺骗”。 |
| [^6] | [Rounding in Preconditioner Space: Redesigning 4-bit AdamW Optimizer-State Quantization](https://arxiv.org/abs/2610.12444) | 该论文从“舍入空间”的新视角重新设计了AdamW的4比特优化器状态量化，证明了状态空间舍入无法保证前条件子误差足够小，并据此提出在二阶矩码本中保留零、于前条件子空间中计算随机舍入概率的ZIP-SR方法。 |
| [^7] | [Density Ratio Estimation with Stein Displacement Fields](https://arxiv.org/abs/2610.12437) | 该论文提出通过Stein位移场参数化密度比（将对数比建模为负的基础分布Stein算子作用于位移场），用单个凸优化问题统一了分布偏移的统计与动力学描述，并据此发展出无需重训即可修正预训练采样器的push-forward算法和拉近数据分布的pull-back算法。 |
| [^8] | [VioLA: Learning Generalist Humanoid Control Policies from Human Data](https://arxiv.org/abs/2610.12435) | VioLA通过让人形机器人通用策略预测身体和手部的运动潜在表示而非关节指令，并借助将人类与机器人运动映射到同一潜在空间的编码器，从而能够直接利用海量人类示教数据，实现无需针对每个任务微调即可开箱即用跟随新指令的通用全身控制。 |
| [^9] | [FAITH: Feasibility-Aware Safety-Filtered RL for High-Dimensional Systems](https://arxiv.org/abs/2610.12432) | FAITH提出了一种可行性感知的无模型安全过滤强化学习框架，通过前馈网络近似最优状态-动作安全价值函数并摊销最小干预过滤，使任务策略在更新中无需竞争性安全项即可优化长时程回报，同时能妥善处理不存在安全动作的情形。 |
| [^10] | [Toward Joint Optimization of Circuit Depth and Training Data Size in Adaptively Grown Quantum Classifiers](https://arxiv.org/abs/2610.12428) | 本研究通过在多规模MNIST数据集上忠实复现Q-FLAIR的自适应电路生长机制，探究量子电路深度与训练数据规模之间是否存在可预测的缩放规律。 |
| [^11] | [Beyond Spatio-Temporal Priors: A Generalizable Approach for Dense Correspondence Matching](https://arxiv.org/abs/2610.12421) | FreeMatching 框架通过结合生成式与语义基础表示、异构监督及教师引导的迭代细化，突破了传统时空先验的限制，实现了图像编辑与参考引导生成中保持视觉同一性的可泛化稠密对应匹配。 |
| [^12] | [A Unified Bellman Operator for Safety-Critical Reinforcement Learning](https://arxiv.org/abs/2610.12420) | 本文提出一种将性能与安全目标统一到单一联合价值函数中的新颖贝尔曼算子，并在双时间尺度随机逼近框架下证明了其时序差分学习的收敛性，从而无需先验知识即可为安全关键强化学习提供严格的安全保证。 |
| [^13] | [WOVEN: Weaving Visual World Modeling into Multimodal LLMs](https://arxiv.org/abs/2610.12417) | 提出WOVEN——一个按场景、动作和推理类型组织的视觉转换推理训练数据源与基准（含36,076个示例），验证视觉转换推理可作为可跨任务复用的共享训练原语，以提升多模态大语言模型的空间、具身、物理和时间推理能力。 |
| [^14] | [Predicting Alignment Generalization with Value Representations](https://arxiv.org/abs/2610.12410) | 本文提出“对齐泛化预测”新任务，通过对66个价值的大规模分析发现，基于模型激活的价值表征能够显著优于基于文本描述的方法，预测模型微调遵循某一价值后在未见价值上的行为泛化。 |
| [^15] | [Learning Kilometer-Scale Weather Prediction with Global-Regional Alignment](https://arxiv.org/abs/2610.12401) | 提出了一种名为 ScaleCast 的区域天气预报框架，通过全球-区域对齐技术，复用预训练全球天气模型的大尺度预报来引导公里级区域预测，解决了跨网格表征对齐和全球引导与局部交互融合的难题。 |
| [^16] | [Prospective Prediction of OOD Degradation from Source-Side Training Dynamics](https://arxiv.org/abs/2610.12397) | 论文证明了仅利用源侧训练动态（尤其是置信度和熵的时间汇总特征）即可在分布外退化发生之前预测到它，且该早期预警信号无需额外训练即可跨模型架构迁移。 |
| [^17] | [HRIL: Learning Multimodal Synergy via Higher-Order Tensor Modeling](https://arxiv.org/abs/2610.12393) | 提出HRIL方法，通过在模态嵌入上构建经验交叉矩张量进行高阶张量建模，显式捕获体现为高阶统计依赖的多模态协同信息，从而在自监督多模态表示学习中保留协同信号的信息容量。 |
| [^18] | [Long Text to Predictive Features: LLM-Guided Blockwise Feature Engineering via Executable Program Search](https://arxiv.org/abs/2610.12390) | 提出LLM-BlockFE框架，由LLM在离线阶段通过逐步追加代码块并结合深度校准信用分配的分块级回滚搜索，将长文本自动转化为可执行的特征程序，使在线推理无需调用LLM即可高效利用长文本信息。 |
| [^19] | [Marformer: A Transformer for Predicting Missing Data Distributions](https://arxiv.org/abs/2610.12379) | 本文提出Marformer，一种受BERT启发的Transformer模型，可根据任意已观测变量集合直接预测缺失变量的条件边际分布，无需建模完整联合分布或领域知识，且能在单次前向传播中完成所有预测，从而支持贝叶斯风险与信息价值的计算。 |
| [^20] | [OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport](https://arxiv.org/abs/2610.12375) | OnTrack提出了一种流式结构感知最优传输监控机制，通过将LLM智能体的执行步骤与记录的成功运行轨迹实时对比，在每步约一毫秒内实现对异常行为的告警或阻断，兼顾了低延迟与安全性。 |
| [^21] | [Bilevel optimization for data-driven learning of Koopman embeddings using kernel-based autoencoders](https://arxiv.org/abs/2610.12370) | 本文提出了一种结合配置方法与双层优化的新方法EDMD-kDL，利用基于核的自编码器直接从数据中学习有限维Koopman嵌入，克服了传统EDMD需先验指定字典的局限，同时相比神经网络方法具有更好的可解释性和理论可分析性。 |
| [^22] | [Closing the Horizon Gap in Policy Optimization for Adversarial MDPs](https://arxiv.org/abs/2610.12362) | 该论文提出使用正则化Q函数在所有状态-动作对上联合控制局部更新稳定性，从而将对抗性MDP策略优化的遗憾界对视野H的依赖性改进到与基于占用测度的算法相当的水平。 |
| [^23] | [Subspace Uncertainty and Sharp Sampling Thresholds on the Boolean Cube](https://arxiv.org/abs/2610.12358) | 该论文确定了在布尔立方体上已知的低度数多项式子空间中进行高斯回归并达到给定极小化极大精度所需的精确样本阈值，其阈值由熵函数形式的指数因子刻画，同时在固定泄漏率下锐化了Polyanskiy–Samorodnitsky不确定性原理。 |
| [^24] | [SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction](https://arxiv.org/abs/2610.12349) | 提出SplitJEPA，一种无需重构即可在JEPA框架内直接学习潜在状态的不变（跨观测共享）与可变因素结构的预测架构。 |
| [^25] | [Overcoming Prior Barriers: Supervised Fine-Tuning under Long-Tail Distribution](https://arxiv.org/abs/2610.12345) | 该论文提出“先验壁垒”新概念，揭示预训练模型对各概念的支持程度呈长尾分布、使头部与尾部概念在监督微调时起点不同，并通过理论推导预测风险界，指出尾部概念需要额外指令才能克服高先验壁垒。 |
| [^26] | [Ambient Discrete Diffusion: Using the Wrong Data at the Right Time for Data Efficient Learning](https://arxiv.org/abs/2610.12340) | RefineMix是一种在数据稀缺条件下训练离散扩散模型的框架，它巧妙利用离散扩散中低噪声水平下掩码保留领域信息的特性，在不引入采样偏差的前提下利用分布外数据提升泛化能力，在五个领域偏移设置中达到或超越领域内微调与数据混合方法的效果。 |
| [^27] | [VFold: Symmetry-Aware Cross-Layer Value Cache Compression](https://arxiv.org/abs/2610.12338) | 该论文提出了一种对称性感知的跨层价值缓存合并策略，无需修改模型架构即可在解码时压缩KV缓存内存，并能与高比率量化或键缓存剪枝等现有技术组合使用，实现单一方法无法达到的更高压缩比。 |
| [^28] | [asdex: Automatic Sparse Differentiation in JAX](https://arxiv.org/abs/2610.12336) | 本文提出了 JAX 中的自动稀疏微分工具 asdex，通过稀疏模式检测、图着色、压缩微分与解压缩四个步骤，利用雅可比矩阵的稀疏结构，使自动微分传递次数与问题维度无关，从而大幅提升计算效率。 |
| [^29] | [RiCo: Neural Simulation of Rigid-Body Interactions via Local Contact Reasoning](https://arxiv.org/abs/2610.12333) | 提出RiCo方法，利用接触表面点的稀疏邻域进行局部接触推理，将跨物体交互限定在相邻表面并传播刚体内部的接触信息，从而实现更精确的刚体交互神经模拟。 |
| [^30] | [Prediction-Powered Data Fusion for Treatment Effect Estimation](https://arxiv.org/abs/2610.12332) | 提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。 |
| [^31] | [Composite Online-to-Nonconvex Conversion with Optimal Oracle Complexity](https://arxiv.org/abs/2610.12328) | 该论文通过为在线学习者设计新的损失函数，将在线到非凸转换框架扩展至复合优化场景，首次在一阶随机预言机访问下建立了复合非凸优化的最优复杂度保证。 |
| [^32] | [SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference](https://arxiv.org/abs/2610.12327) | 该论文提出SparseDecoding，一种解码感知的剪枝方法，通过在模型自生成的token序列上计算Hessian来消除自然序列与生成序列之间的分布偏移，从而在保证准确性的同时提升LLM解码阶段的推理效率。 |
| [^33] | [Prior or Feedback? What an LLM Uses When Adapting Neural Operators](https://arxiv.org/abs/2610.12325) | 该论文通过受控干预实验发现，LLM在有限预算下适配神经算子时同时依赖有用的先验（首个配置即接近随机搜索池顶端）和实验反馈来选择微调配置，在几乎所有匹配比较中优于随机搜索和贝叶斯优化。 |
| [^34] | [Spatial Pattern Formation from Multi-Agent Learning in Public Goods Dilemmas](https://arxiv.org/abs/2610.12321) | 该研究首次展示在公共物品困境中，合作者与背叛者通过多智能体Q学习自主习得移动策略即可自发涌现空间模式（资源峰值周围的集群与移动条带），并揭示学习率的不对称配置（合作者快、背叛者慢）导致最大集体福利损失，且部分模式反映训练历史而非稳态结果。 |
| [^35] | [Unlocking the Regulatory Genome by ARGUS: An Evidence-Constrained Agentic Framework for Interpreting Single Nucleotide Variants](https://arxiv.org/abs/2610.12281) | ARGUS提出了一种严格分离确定性生物计算与大语言模型推理的证据约束型智能体框架，通过假设导向的研究循环和458个基于DNABERT的转录因子结合模型，可靠地解读非编码调控区单核苷酸变异的功能影响，避免了LLM的幻觉问题。 |
| [^36] | [AdaptLSTM: Efficient Adaptive Online Learning for Cloud Workload Forecasting under Distribution Drift](https://arxiv.org/abs/2610.12265) | AdaptLSTM 通过验证校准阈值检测分布漂移并进行选择性针对性更新，以约 20% 的计算成本实现接近朴素在线学习的预测精度，大幅提升云工作负载预测的效率。 |
| [^37] | [Batch Before You Lift: Scalable Topological Deep Learning on Large Graphs](https://arxiv.org/abs/2610.12247) | 本文提出Cluster-TNN框架，通过先对大图进行划分、再在小批量内局部执行图提升操作，避免了全域物化的计算瓶颈，使拓扑深度学习能够扩展到大规模稠密图上。 |
| [^38] | [RIFT: Relative Isolation From Trees For Anomaly Detection](https://arxiv.org/abs/2610.12244) | 该论文提出了RIFT——一种基于最小生成树的确定性无参数异常检测方法，其一维评分能精确恢复孤立森林的闭式极限，在高维数据上对密度变化和聚类异常具有鲁棒性，且准确率与孤立森林相当，同时避免了其轴平行伪影问题。 |
| [^39] | [AdaCast: Conditional Parameter Generation for Adaptive Time Series Forecasting](https://arxiv.org/abs/2610.12240) | AdaCast提出了一种条件参数生成框架，利用生成器为冻结的预训练时间序列基础模型动态生成针对每条输入序列的低秩参数更新，实现输入级的自适应预测，在六个基准测试中超越了静态适配方法。 |
| [^40] | [Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting](https://arxiv.org/abs/2610.12232) | 该论文构建了无泄漏的延迟感知审计框架，在真实延迟标签条件下评估了四种主流时间序列预测测试时自适应方法，并提出了无需超参数调节的递归最小二乘滤波器组作为强基线，揭示了延迟标签下各方法表现的不对称性。 |
| [^41] | [ISBO: Scalable Spatio-Temporal Bayesian Optimization with Log Gaussian Cox Process Models via the INLA-SPDE Approach](https://arxiv.org/abs/2610.12213) | 该论文提出了首个面向时空数据的可扩展贝叶斯优化框架ISBO，通过对数高斯Cox过程建模与INLA-SPDE推断方法，能够以最少的评估次数稳定定位高强度区域及潜在强度峰值。 |
| [^42] | [Verification with Transfer: Exact Information Frontiers and Their Price in Calls](https://arxiv.org/abs/2610.12211) | 本文从信息论角度为“借助相关源任务（迁移）来降低验证成本”这一策略精确定价：所需最小因果信息由列表率失真函数刻画，并把所有源调用放在首次验证之前不会破坏调用的硬上限，但交错安排可以无界地节省期望调用次数。 |
| [^43] | [SciTBERT: A family of chronologically consistent language models for scientific and technological language processing](https://arxiv.org/abs/2610.12207) | 提出了SciTBERT——一系列时间一致的BERT衍生语言模型，训练数据截止日期覆盖2013至2025年每年，通过消除前瞻偏差和领域偏差，使模型适用于研究科学与技术随时间演变的特性。 |
| [^44] | [Quickest Change Detection with Diffusion-Integrated Scores](https://arxiv.org/abs/2610.12200) | 提出了一种无需训练的扩散积分分数CUSUM（DI-SCUSUM）最快变化检测器，通过向样本添加高斯噪声并精确计算Hyvärinen分数来近似对数似然比，实现了指数级误报控制和有保证的一阶检测延迟界。 |
| [^45] | [DataSense-Bench: The First Step Toward an AI Scientist](https://arxiv.org/abs/2610.12190) | 该论文提出首个评估AI“数据感”能力的基准DataSense-Bench，让AI智能体在无法训练模型、无法访问评估任务的情况下为LLM微调选择和排序训练数据子集，再通过实际微调后的性能来检验前沿AI模型能否可靠地为训练选择正确的数据。 |
| [^46] | [Just Weather Scoring: Efficient End-to-end Nowcasting with Distributional Diffusion](https://arxiv.org/abs/2610.12189) | 提出 JWS——一种单阶段端到端扩散模型，通过在雷达空间直接预报、掩码异步扩散与评分规则目标实现少步生成，在大幅降低推理成本的同时达到了最先进的概率性降水临近预报效果。 |
| [^47] | [A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization](https://arxiv.org/abs/2610.12183) | 提出了AgenticBBO-Bench，一个采用统一有限预算评估协议、覆盖合成函数、超参数优化、数据库调优、芯片设计和分子设计五个领域的跨领域智能体黑盒优化基准测试，并验证了智能体BBO在所有领域中均优于直接基于LLM的方法。 |
| [^48] | [Learning to Plan by Looking Back: Hindsight Hierarchies for Training Reasoning Models](https://arxiv.org/abs/2610.12168) | 该论文提出了一种基于“后见之明”的自我改进循环，通过联合训练同一模型预测解题思路、从已知解答逆向推导思路、以及利用给定思路解决问题这三种能力，使模型即使面对超出自身能力的问题也能从提供的解答中提取有用思路来持续自我提升，并在Lean定理证明器中给出了具体实现。 |
| [^49] | [Is Real-World Training Data Necessary for Generalist Graph Anomaly Detection?](https://arxiv.org/abs/2610.12167) | 本文提出AG-FORGE异常图自动合成框架，证明合成数据训练可实现与真实数据相当的通用图异常检测性能，并进一步提出拓扑-语义协同的TS-GGAD模型以释放更大模型容量。 |
| [^50] | [Scalable Hierarchical Graph Generation via Soft Community Structure](https://arxiv.org/abs/2610.12163) | Schema通过将参考图递归分解为软社区层次结构，并将生成过程拆分为属性合成、社区内边生成和社区间连接建模三个可独立训练的阶段，实现了大规模属性图的可扩展生成，且全程无需构建完整邻接矩阵。 |
| [^51] | [When KL Regularization Misfires in Group Policy Optimization](https://arxiv.org/abs/2610.12161) | 论文剖析了组策略优化中KL正则化与奖励相互作用的七种失效模式，并提出零和校准策略优化（ZCPO），借助条件KL度量的相对漂移校准组内奖励系数，从而提升优化效果。 |
| [^52] | [Toward Optimal Regret in Adversarial MDPs with Stochastic Hard Constraints](https://arxiv.org/abs/2610.12153) | 提出MA-OPS算法，通过乐观搜索Slater余量与悲观评估所选策略相结合，在随机硬约束的对抗性MDP中实现了关于可行性余量的最优后悔界。 |
| [^53] | [Bayesian Optimisation under State-Preservation Constraints](https://arxiv.org/abs/2610.12150) | 该论文提出一种通过线性化预计算将状态空间中的状态保持约束拉回到设计空间并形成椭球体的方法，从而高效处理可行集高度各向异性的贝叶斯优化问题，并在托卡马克偏滤器优化这一关键应用上得到端到端验证。 |
| [^54] | [Large-Scale Benchmarking of Quantum Neural Network Configurations for Financial Time Series Forecasting](https://arxiv.org/abs/2610.12148) | 该论文通过大规模网格搜索构建并评估了1,368种量子神经网络配置在金融时间序列预测（以GBP/USD汇率为案例）中的表现，系统揭示了编码方法、拟设设计、量子比特数等组件选择对预测精度、计算成本和收敛行为的影响。 |
| [^55] | [Poster: A Preliminary Study of LLM Distillation Inference](https://arxiv.org/abs/2610.12137) | 该论文提出一种基于假设检验与影子模型的蒸馏推断方法，通过比较可疑模型对教师推理输出的预测得分并转换为校准p值，能够有效判定模型是否从专有LLM蒸馏而来，初步实验中在0.02显著性水平下实现1.0的真阳性率。 |
| [^56] | [Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read](https://arxiv.org/abs/2610.12133) | 该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。 |
| [^57] | [A structure-preserving neural density functional for the ions of a polymer electrolyte](https://arxiv.org/abs/2610.12132) | 开发了一种保持空间对称性、热力学可积性和诺特定理恒等式的神经密度泛函，仅需平面密度和内力数据训练即可准确预测聚合物电解质离子在不同浓度和尺度下的结构与关联行为。 |
| [^58] | [Using Weisfeiler-Leman Features for Algorithm Selection in Constraint Optimisation](https://arxiv.org/abs/2610.12119) | 本文提出一种结合图转换与Weisfeiler-Leman图核的自动化特征提取方法（包括基于割的WLc表示），无需训练图神经网络即可捕捉问题实例的结构信息，从而改进约束优化中的算法选择。 |
| [^59] | [Credal Machine Learning for Risk-Averse Decision Making](https://arxiv.org/abs/2610.12115) | 该论文提出用信度集（概率分布的集合）来表示预测中的认知不确定性，并结合一种新颖的决策规则，实现基于CVaR的可靠风险规避决策。 |
| [^60] | [A Geometric Approach to Soft Actor-Critic with Zonotopes for Locomotion Learning](https://arxiv.org/abs/2610.12113) | GeZo-SAC通过让每个评论家预测定义Zonotope的生成向量，将几何宽度作为悲观偏移量，并根据评论家间的分歧程度自适应地在加权平均与最小值之间组合评论家，从而改进了软演员-评论家算法在运动控制学习中的表现。 |
| [^61] | [Could LLM Watermark Detection be Public?](https://arxiv.org/abs/2610.12106) | 该论文提出一种分割密钥的公私钥LLM水印方法，通过公开检测器只暴露一个密钥，并利用公开与私有分数间的不平衡统计检验识别知情攻击者，从而证明水印检测可以安全地公开。 |
| [^62] | [Few-Step Generation via Data-Space Iteration](https://arxiv.org/abs/2610.12102) | 该论文提出“数据空间迭代”少步生成框架，让共享生成器从噪声出发直接在数据空间中迭代精炼预测结果，从而彻底摆脱流匹配采样中对人工选择的时间步离散化的依赖。 |
| [^63] | [SCORE: Spectral Correlation Estimation for Multivariate Gaussians](https://arxiv.org/abs/2610.12096) | 提出SCORE框架，将评分规则训练与谱空间中的协方差近似相结合，能以线性存储和O(d log d)成本高效学习高维多变量高斯的密集相关结构，并具有数值稳定性和酉变换不变性等理论保证。 |
| [^64] | [DVLA-RL++: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning](https://arxiv.org/abs/2610.12095) | DVLA-RL++ 通过互补语义净化与反事实强化学习门控，将物体内在语义与偶然上下文区分开来，避免少样本学习中的支持原型受到上下文污染，从而提升对新类别的识别能力。 |
| [^65] | [Differentiable Systematic Resampling for Variational Sequential Monte Carlo](https://arxiv.org/abs/2610.12094) | 提出可微系统重采样（DSR），一种温度控制的系统重采样松弛方法，在保持其结构特性并具有可证明的指数收敛偏差的同时，实现了完全梯度流动，且计算开销远低于基于最优传输的方法。 |
| [^66] | [Perception Test 2026: Challenge Summary and Extension to City-scale Audio-Visual Reasoning](https://arxiv.org/abs/2610.12081) | 本报告总结了在ECCV 2026举办的第四届Perception Test挑战赛，介绍了两个基于城市级步行视频的新视听推理基准，并指出复杂的空间与多模态推理可以通过昂贵的智能体流程解决，但对单独使用的多模态模型仍然困难。 |
| [^67] | [Exploiting Gradients in Bayesian Inference of Expensive Simulators](https://arxiv.org/abs/2610.12076) | 该论文提出在昂贵模拟器的贝叶斯推断中，利用模拟器输出对输入参数的梯度信息作为额外信号来指导基于贝叶斯优化的主动学习过程，从而提高有限模拟预算下的推断效率。 |
| [^68] | [Diffusion Removes Langevin's Conditioning Dependence: A Sharp Gaussian Analysis](https://arxiv.org/abs/2610.12052) | 本文在高斯情形下证明扩散模型的采样误差为 $O(\sqrt{d\lambda_{\max}}\log N/N)$，消除了经典朗之万类采样器中依赖条件数的 $\sqrt{\kappa}$ 因子，并通过精确的谱界与匹配的一阶渐近分析，严格解释了扩散模型优于传统基于分数采样器的理论原因。 |
| [^69] | [MPGE: A Multi-Perspective Graph Explainer for Molecular Classification Explanation](https://arxiv.org/abs/2610.12039) | 该论文提出MPGE多视角图解释框架，针对冻结的图神经网络分类器统一了事实支持（原型）、反事实敏感性和示例容忍性三种解释视角，通过共享约束公式和独立目标函数为分子分类预测提供更全面的解释。 |
| [^70] | [Efficient and Generalizable Archetypal Analysis for Discrete Data](https://arxiv.org/abs/2610.12035) | 提出了一种面向离散数据的基于似然的高效原型分析框架，支持伯努利、泊松和多项式观测模型，并引入交叉验证的预测似然准则来有原则地选择原型数量。 |
| [^71] | [Examining Social Attribution in LLM Reasoning: A Theory-Guided Probing Methodology](https://arxiv.org/abs/2610.12022) | 该论文首次系统性地探索大语言模型的社会归因能力，在归因理论指导下构建了包含经典心理学情境与现实场景的基准，以考察大语言模型在责任与过错归因上的判断及其内部机制。 |
| [^72] | [CausalDreamer: Learning Predictive World Models with Latent Disentanglement](https://arxiv.org/abs/2610.12016) | CausalDreamer 在冻结的视频分词器之上，将潜在表示沿可控性和奖励相关性两个维度解耦为四组因素化表示，使世界模型能够显式区分环境中可控、不可控、奖励相关和奖励无关的信息。 |
| [^73] | [Ghost tasking for parametrized Gaussian Processes solving linear differential equations](https://arxiv.org/abs/2610.12009) | 本文提出“幽灵任务”方法，通过引入辅助任务使任意不可参数化系统有效变得可参数化，从而能以较少的任务数量和潜在函数算法化构建参数化高斯过程来求解线性微分方程，并在数据稀少的逆问题中表现尤为出色。 |
| [^74] | [Test-Time Compute for Tabular Foundation Models: Mechanisms, Gains, and Limits](https://arxiv.org/abs/2610.12005) | 本文系统研究了测试时计算对表格基础模型预测性能的提升机制，提出仅训练0.003-0.03%参数的对角线相似度更新方法DiagScale，其效果可媲美全量微调，并发现对96种配置采用贪婪选择可将误差降低2.4%，而均匀平均反而会增加误差。 |
| [^75] | [The Polytopal Neural Network](https://arxiv.org/abs/2610.12004) | 本文提出多面体神经网络（PNN）框架，通过在信息处理中直接强制多面体结构来提取各层级特定特征，在几乎不损失性能的前提下保留潜在空间的有意义结构，同时实现压缩表示并为向量量化训练提供直接途径。 |
| [^76] | [Agentic-TTT: Training test-time policy for test-time training](https://arxiv.org/abs/2610.12002) | 提出Agentic-TTT框架，通过训练一个测试时策略来智能决策何时、如何调用测试时训练（TTT）以及是否复用已有技能，从而实现模型参数层面的自主化自我改进。 |
| [^77] | [Example-driven Parametrisations for Bayesian Shape Optimisation](https://arxiv.org/abs/2610.11984) | 本文提出从现有设计集合中通过主成分分析学习形状间变形的参数化方法，为贝叶斯形状优化构建了一个线性、可解释的搜索空间，在翼型、机翼和射频腔体等任务中实现了更高的样本效率，并能探索超越手工参数化基线的更优设计。 |
| [^78] | [Efficient quadratic entropy with distance sketches](https://arxiv.org/abs/2610.11976) | 本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。 |
| [^79] | [CAPABLE: Capability-Aware Policy Adaptation via Behavioral Latent Encoding](https://arxiv.org/abs/2610.11971) | CAPABLE是一个无需故障标签或重新训练的能力感知自适应框架，通过自监督能力推断与残差强化学习相结合，使冻结的视觉-语言-动作（VLA）策略能够在关节故障发生时在线适应并恢复执行能力。 |
| [^80] | [Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability](https://arxiv.org/abs/2610.11957) | 该论文提出随机分组保形预测（SGCP），通过学习随机分组映射让每个样本从校准行为相似的样本中获取校准信息，在无需敏感子群体属性的情况下实现跨临床子群体的可靠不确定性量化，并缓解最差群体瓶颈问题。 |
| [^81] | [Reliability-Aware Future Conditioning for Temporally Robust Robot Manipulation](https://arxiv.org/abs/2610.11956) | 本文提出可靠性感知未来条件化（RAFC），将生成视频作为指导时的时间错位问题重新定义为一个控制问题，通过在每一步估计对生成片段的信任程度、偏好邻近的时间假设并在不匹配时回退到静态分支，且仅依靠任务奖励学习而无需偏移标签或对齐监督，从而显著提升机器人操作策略对时序偏移的鲁棒性。 |
| [^82] | [Interval-valued SHAP in Tree-Based Models](https://arxiv.org/abs/2610.11953) | 本文提出基于不精确Dirichlet模型的区间值SHAP方法，通过向树的叶子节点随机引入少量未标注实例来量化和分析决策树与随机森林中Shapley值的鲁棒性，并借助悲观原则与平均原则定义区间值，同时推导出实现高效计算的理论结果。 |
| [^83] | [Score-Based Learning of Cluster DAGs from Interventions](https://arxiv.org/abs/2610.11947) | 提出首个基于评分的方法COARSE，利用干预数据识别聚类间的因果顺序并将边学习简化为局部搜索，从而在线性高斯假设下实现聚类DAG的学习。 |
| [^84] | [TACROSS: An Efficient and Low-Cost Scalable Human Touch System Across Heterogeneous Tactile Sensors for Dexterous Robot Learning](https://arxiv.org/abs/2610.11945) | TACROSS提出了一种成本仅10.86美元的五层压阻式触觉手套系统，通过在接触事件层面而非原始传感器值层面对齐异构触觉信号，将人类触觉数据高效迁移到机器人触觉传感器上，实现了可扩展的低成本灵巧机器人学习。 |
| [^85] | [Revisiting Identity and Spectra Dispersion in Media-Bridged Time Series Forecasting: Linking Multivariate Signals and Narrative Flows](https://arxiv.org/abs/2610.11924) | 本文提出统一的多媒体身份感知棱镜网络（MIDAPN），通过媒体通用图适配（MIDAG与CIM）和频谱棱镜卷积构建跨范式时空预测主干，从而将多变量数值信号与叙事流文本桥接于同一时间序列预测框架中。 |
| [^86] | [Puffin: Probabilistic Learning of Spatial Detail From Coarse Observations](https://arxiv.org/abs/2610.11914) | 提出了Puffin——一个统计降尺度概率框架，以高分辨率卫星嵌入为协变量、通过聚合感知的似然函数学习概率分布，在推理时以观测到的区域总量为条件将数据分解到子区域，无需精细分辨率标签即可生成与总量一致且不确定性经过校准的精细尺度估计。 |
| [^87] | [Cost-Aware Mixture-of-Experts Coordination for Model Markets](https://arxiv.org/abs/2610.11908) | 本文将混合专家从模型级学习架构提升为市场级协调机制，提出成本感知的门控机制和成本调整的收入分配规则，使模型市场能够协调异构专家并提供复合模型服务。 |
| [^88] | [RobustLDS: Learning linear dynamical systems under adversarial corruptions](https://arxiv.org/abs/2610.11906) | 该论文提出了基于最小截断二乘法松弛与离群值组稀疏性的估计器，用于在对抗性污染下从单条轨迹学习线性动力系统，并通过非渐近误差界证明了其对离群值的鲁棒性。 |
| [^89] | [Automated Assembly Instruction Generation from CAD Models Using Grounded Large Language Models: A Human-in-the-Loop Framework](https://arxiv.org/abs/2610.11896) | 该论文提出一种人在回路的框架，将CAD装配模型映射为结构化的ProductGraph中间表示，利用大语言模型自动生成受工程信息约束的自然语言装配指令，并通过人工审核解决所有质量标记后才允许导出文档。 |
| [^90] | [Learning structured linear dynamical systems from missing observations](https://arxiv.org/abs/2610.11869) | 本文提出一种基于偏差校正目标函数的估计器，用于在观测严重缺失的情况下学习凸集约束下的结构化线性动力系统，给出了依赖集合局部复杂度、轨迹长度和采样概率的非渐近误差界，并证明即使轨迹远短于无约束情形且采样概率趋于零时仍能有意义地恢复转移矩阵。 |
| [^91] | [Understanding Latent-Dimension Scaling in Dynamical-System Learning through Spectral Reliability](https://arxiv.org/abs/2610.11866) | 本文提出用Koopman算子的谱可靠性（通过相对残差检测伪特征对）来解释动力系统学习中增大潜在维度为何能持续降低滚动预测误差，并从理论上证明了所学字典空间中的最小残差随空间逼近可观测空间而逐点收敛于全空间结果。 |
| [^92] | [Conditional Kernel Stein Discrepancy](https://arxiv.org/abs/2610.11863) | 提出了一个通过协变量空间上的算子值核将核斯坦因差异推广到条件设定的框架，用于在仅知道非归一化条件目标模型和联合分布样本的情况下量化条件拟合优度。 |
| [^93] | [GRPODropout: Less is More for Online Reinforcement Learning Rollouts](https://arxiv.org/abs/2610.11854) | 提出GRPODropout方法，通过在GRPO策略更新前选择性地移除少量高概率的正优势轨迹并重新居中保留的优势，有效缓解策略熵坍缩问题，提升大语言模型的推理能力。 |
| [^94] | [DADP: Dynamic Activity-Dependent Pruning, A Reverse Hebbian-Inspired Structural Pruning Method](https://arxiv.org/abs/2610.11853) | 本文提出受反向赫布法则启发的DADP结构化剪枝方法，通过突触前激活与突触后误差梯度乘积的累积来衡量连接重要性，并利用单一全局阈值在训练中动态分配各层稀疏度，无需手动设置每层剪枝目标即可匹配或超越现有剪枝方法。 |
| [^95] | [Open-Vocabulary Audio-Visual Event Localization via Complex-Valued Fusion](https://arxiv.org/abs/2610.11846) | 该论文提出用复数值相似度并结合复数值神经网络来学习视觉与音频两个模态相似度的融合，取代传统固定规则（几何平均或加权平均），从而提升开放词汇音视频事件定位的性能。 |
| [^96] | [Recovery Guarantees for Posterior Sampling of One-Bit Compressed Sensing](https://arxiv.org/abs/2610.11834) | 该论文用近似覆盖数刻画先验分布的复杂度，证明后验采样在测量数随覆盖数对数缩放时可高概率精确恢复，该上界对Wasserstein距离意义下的学习先验失配具有鲁棒性，并给出了几乎匹配的样本复杂度下界。 |
| [^97] | [Self-Supervised Speech Representations for Cross-Speaker Dysarthria Detection During Awake Craniotomy](https://arxiv.org/abs/2610.11825) | 该研究提出了一种融合自监督语音表征（wav2vec 2.0）、说话人条件归一化和级联分类器的系统流水线，实现了在手术室嘈杂环境下跨说话者的清醒开颅术中构音障碍语音自动检测。 |
| [^98] | [Softmax Attention on Gaussian Mixtures: Linear When It Can, Selective When It Must](https://arxiv.org/abs/2610.11798) | 该论文通过研究softmax注意力在高斯混合分布上的无穷提示极限，证明softmax注意力既能像线性注意力一样有效解决线性任务，又能借助查询依赖的选择能力，以梯度方法学习到监督分类、去噪等具有潜在结构、多峰性和非线性依赖的统计任务的最优解。 |
| [^99] | [Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks](https://arxiv.org/abs/2610.11794) | Memento 3 让冻结参数的 LLM 智能体将可修正的环境假设记录为自然语言“规则手册”并编译为可执行代码，通过预测误差驱动的观察—反思—修订—编译—验证循环，实现显式世界模型的持续递归自我改进。 |
| [^100] | [Compile the Table: Query-Calibrated Operator Compression for Tabular In-Context Learning](https://arxiv.org/abs/2610.11784) | 提出QCOC方法，将表格上下文学习的完整KV缓存一次性编译为查询校准的联合KV原型并在查询间共享，在不牺牲准确率的情况下显著提升预测吞吐量。 |
| [^101] | [In-Ride Alcohol-Impairment Detection in E-Scooterists with False-Alarm Control](https://arxiv.org/abs/2610.11783) | 本文提出一种利用车载惯性及油门传感器在电动滑板车骑行过程中实时检测骑行者酒精损害的新方法，并提供了可证明的误报率界限。 |
| [^102] | [RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing](https://arxiv.org/abs/2610.11775) | 本文提出叠加特化假说，认为MoE专家专精于细粒度特征的组合而非单一领域，并据此开发RouterInterp方法，通过稀疏自编码器特征与自然语言解释来解读专家路由，检测准确率比先前方法高出约65%。 |
| [^103] | [Optimal random quantisers for spherically symmetric distributions](https://arxiv.org/abs/2610.11772) | 该论文证明对于球对称目标分布，随机量化器的优化问题具有凸性，并据此发现均匀分布在适当半径球面上的随机量化器即使在中等样本量下也表现优异、被数值验证为全局最优，从而克服了Zador渐近理论中高维所需的天文数字级样本量问题。 |
| [^104] | [RAGenome: Scaling Retrieval-Based Genomic Language Models to Long Contexts](https://arxiv.org/abs/2610.11761) | RAGenome是首个基于检索的基因组语言模型，通过将预训练上下文扩展到现有基于MSA模型100倍的长度，突破了短输入限制，实现对长基因组序列的建模。 |
| [^105] | [Beyond QAOA: A Review of AI and Quantum Computing for Adaptive Combinatorial Optimization](https://arxiv.org/abs/2610.11759) | 本综述提出“自适应量子优化”的概念，按被学习的决策类型系统梳理了AI赋能量子优化、量子赋能AI优化和AI-量子协同优化三种范式，并通过对119篇论文的分析发现AI辅助量子优化的证据显著强于量子辅助AI优化。 |
| [^106] | [SR-TTA: Spatial-Redundancy Test-Time Adaptation for Interference-Robust Respiration Sensing](https://arxiv.org/abs/2610.11755) | 该论文提出SR-TTA方法，利用学习得到的复数加权波束成形器进行空间置零，并结合最大化随机天线子集间一致性的无标签测试时自适应，使无蜂窝大规模MIMO基站能在强带内干扰下依然实现鲁棒的呼吸感知。 |
| [^107] | [DEX: Digit-Level Early Exit for Energy-Efficient MSDF Neural Network Inference](https://arxiv.org/abs/2610.11748) | 本文提出基于最高有效位优先（MSDF）算术的数位级提前退出加速器DEX，通过ReLU提前负数检测、仅符号决策、低位数跳过和校准剪枝四种运行时机制，在保证分割精度的前提下显著降低脑肿瘤分割U-Net推理的能耗。 |
| [^108] | [The Ball and the Box: Two Geometries of Computation in Superposition](https://arxiv.org/abs/2610.11744) | 该论文用球几何刻画期望误差阈值、用盒子几何刻画联合可靠性阈值，揭示了叠加神经表征计算布尔门时两种误差准则下的维度需求差距源于共享读取，并给出了合取、析取和多数门的显式维度阈值。 |
| [^109] | [TraceRelay: Attention-Aligned Recurrence over Rolling Traces](https://arxiv.org/abs/2610.11743) | TraceRelay通过注意力对齐的循环相位继承机制将持久表示分布在滚动低维轨迹上，在长度泛化任务中取得约99%的准确率，远超无继承机制的51%基线。 |
| [^110] | [Correlational Training of Morphological Neural Networks](https://arxiv.org/abs/2610.11740) | 提出了一种受乘性权重更新（MWU）启发的基于相关性的权重更新方法来训练形态学神经网络，在九个基准测试中的八个上取得最高达32.84个百分点的性能提升。 |
| [^111] | [Uncovering and Fixing Collider Bias in Bayesian PINNs](https://arxiv.org/abs/2610.11737) | 本文揭示贝叶斯物理信息神经网络中常用的碰撞体建模结构会在物理参数后验中引入严重的系统性偏倚，并提出改用“物理生成轨迹、轨迹生成观测”的分层链式模型来消除该偏倚，尽管这会带来更难的双重难解推断问题。 |
| [^112] | [Timer-M1: A Multivariate Time Series Foundation Model via Learning Primitives](https://arxiv.org/abs/2610.11734) | Timer-M1通过学习跨领域共享的时序与关系基元，并采用基于基元的数据合成与预训练流程，构建了能够实现零样本预测的多元时间序列基础模型。 |
| [^113] | [Spectral Weight Decay: Inducing Low-Rank Structure in Neural Network Weights](https://arxiv.org/abs/2610.11730) | 提出谱权重衰减方法，通过加法式谱收缩诱导神经网络权重的低秩结构，在匹配验证损失下将LLaMA模型压缩率提升至1.89倍、GPU推理加速至1.18倍，并显著提高噪声标签下的测试准确率。 |
| [^114] | [Addressing Overcommitment in the Reasoning of Gendered Economic Memes under Multimodal Ambiguity](https://arxiv.org/abs/2610.11724) | 提出CGER-Net框架，通过评估情境证据充分性并采用证据门控推断，缓解多模态模糊模因中因认知过度断言导致的性别化经济角色刻板归因偏见。 |
| [^115] | [4-Tensor Attention Model for Semantic Physical Reality](https://arxiv.org/abs/2610.11716) | 该论文提出一种四阶张量注意力模型，通过在语义纤维与时间上下文纤维上进行联合归一化注意力来预测场景的下一语义状态，在参数量几乎相同的情况下，其最后一句交叉熵在三个设置上均优于自由运行的一维 Transformer（最多低 5.3%），为视频生成和机器人规划提供了新方法。 |
| [^116] | [Internalizer: Portable Context-to-Parameter Mapping for Very Large Language Models](https://arxiv.org/abs/2610.11715) | Internalizer是一种可移植的上下文到参数映射超网络，先在小模型上低成本训练后即可移植到2840亿参数的DeepSeek v4 Flash上，为其生成文档特定的LoRA适配器，使模型无需在上下文窗口中放入文档即可将知识内化到权重中。 |
| [^117] | [Does an Illumination Prior Help Face-Swap Detection? A Controlled Study of Temporal Self-Blended Images](https://arxiv.org/abs/2610.11706) | 本对照研究发现，在自混合图像训练中加入时序光照不一致性并不能带来光照特异性的换脸检测性能提升，反而会移动预测分数分布并改变最优阈值，仅在DFDC重度JPEG压缩场景下略改善鲁棒性。 |
| [^118] | [What is the goal of unsupervised machine learning?](https://arxiv.org/abs/2610.11697) | 本文认为无监督学习是异质化领域，无法定义单一目标，并提出其四个不同目标：估计分布、生成新数据、为下游任务提取特征和理解数据。 |
| [^119] | [Can Jev be Your Q or Policy in Reinforcement Learning?](https://arxiv.org/abs/2610.11692) | 本文研究了Jev决策模型能否在强化学习中自主充当Q函数或策略等角色，发现它能满足强化学习系统中除价值函数之外的所有对象对答案的要求，并探讨了它作为训练组件改进强化学习的潜力。 |
| [^120] | [Camera-Noise Residuals for Face-Swap Detection: Redundant, Not Complementary, and Why](https://arxiv.org/abs/2610.11683) | 该研究发现，相机噪声残差信息对换脸检测而言与RGB外观特征是冗余而非互补的：其判别信号本质上是统计性的（残差的均值、方差、能量等矩），且会被噪声分支输入处的InstanceNorm层标准化掉，因此融合无法带来性能提升。 |
| [^121] | [Early Signatures of Memorization in Diffusion Models via Basin Geometry and Cyclic Denoising](https://arxiv.org/abs/2610.11670) | 该论文提出扩散模型的记忆化在生成样本显现之前就已编码于能量景观的几何结构中（即“潜在记忆化”），并利用分数散度、盆地体积和循环去噪方法，在理论上证明可在记忆化发生前对其进行早期检测。 |
| [^122] | [$\sigma$Transfer: Uncertainty Transfer from Small to Large Networks under $\mu\mathrm{P}$](https://arxiv.org/abs/2610.11668) | 该论文提出σTransfer方法，在μP参数化下通过重新缩放先验协方差，使拉普拉斯近似所需的先验精度可以从小模型零样本迁移到大模型，从而免去在大模型上的昂贵精度搜索，实测加速可达约5000倍。 |
| [^123] | [Evi-VN: Hard Region Guided Virtual Node Evidence Injection for GNN-Based Fraud Detection](https://arxiv.org/abs/2610.11665) | 提出Evi-VN框架，通过硬区域引导的虚拟节点证据注入机制，学习并纠正多种GNN在欺诈检测中共同的盲点区域，有效利用异构多模态证据来识别伪装良好的欺诈者。 |
| [^124] | [Harness Evolution Hits a Ceiling: When Weight Training Should Begin](https://arxiv.org/abs/2610.11655) | 该论文提出通过失败构成分析来决定改进长时程LLM智能体的正确杠杆——将失败区分为过程失败与内容失败，其中Harness演化修复过程失败且其行为可被训练进权重，而内容失败则需依靠权重训练来解决。 |
| [^125] | [Phonologically Informed Tokenization for German Speech Recognition: A Cross-Domain Study](https://arxiv.org/abs/2610.11646) | 本文提出基于音系学知识（Pyphen 音节划分与字素-音素转换）的分词方法用于德语端到端语音识别，发现在域内条件下其性能与 BPE 和字符基线相当，而跨领域表现主要由词表规模而非语言学分词方式决定。 |
| [^126] | [Minimax Gaussian Mechanisms for Continual Machine Unlearning](https://arxiv.org/abs/2610.11628) | 本文提出基于牛顿更新与高斯差分隐私的极小极大高斯机制，通过推导残差误差上界来校准噪声方差分配，使得顺序删除记录后发布的一系列模型在统计上与精确重训练难以区分，并最小化最坏情况下的噪声方差。 |
| [^127] | [Beyond Action Entropy: Quotient-Space Exploration for Genome-Scale Metabolic Model Repair](https://arxiv.org/abs/2610.11627) | 该论文提出QuotientPO方法，通过将等价的代谢模型修复折叠为规范化机制并在商空间上直接优化探索，配合核化Rényi估计器解决修复核心间的拥挤问题，从而克服了传统探索中输出多样性无法对应科学假设多样性的隐性失效模式。 |
| [^128] | [Constructing Structured Decision Sources for Consensus-Based Pseudo-Label Learning](https://arxiv.org/abs/2610.11621) | 该论文提出通过受控改变图表示的类内结构（中心粒度和邻域混合）来构建稳定且互补的决策源，并利用共识机制使伪标签精度相比常规GCN提升1.19至4.39个百分点。 |
| [^129] | [Randomized Transport Maps for Model-Free Policy-Gradient Mean-Field Control](https://arxiv.org/abs/2610.11619) | 提出Transport REINFORCE方法，通过基于传输映射的种群分布随机扰动，填补了标准REINFORCE在平均场控制中无法捕捉的种群分布效应，实现了适用于有限与连续状态空间的无模型策略梯度学习。 |
| [^130] | [NanoProof: Open and Efficient Automated Theorem Proving in Lean 4](https://arxiv.org/abs/2610.11605) | NanoProof 是首个训练数据、工具、流程和权重全部开源、可端到端复现的 Lean 4 执行引导定理证明器，以比同类系统少约 90 倍、比 AlphaProof 少四个数量级以上的计算量，在 MiniF2F-Test 上达到 50.8% 的 pass@16。 |
| [^131] | [Embedding-Bias in Conditional Independence Testing](https://arxiv.org/abs/2610.11584) | 该研究揭示了在条件独立性检验中用嵌入替代原始变量所引发的偏差问题，并证明对于残差相关性检验，只要嵌入遗漏的条件均值部分互不相关即可保证检验有效，否则偏差可精确量化为遗漏部分绝对相关性乘以两个偏R²值的几何平均数。 |
| [^132] | [Uncertainty-Aware Optimization for Physics-Aware Highway Trajectory Prediction](https://arxiv.org/abs/2610.11580) | 该论文提出了物理感知轨迹预测框架X-TRACK的不确定性感知扩展版本（X-TRACK-DE和X-TRACK-MCD），通过显式建模运动变量中的偶然不确定性和认知不确定性，并将其经由车辆动力学传播至轨迹空间，从而提升高速公路车辆轨迹预测在安全关键应用中的可靠性。 |
| [^133] | [Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts](https://arxiv.org/abs/2610.11575) | 提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。 |
| [^134] | [Sera: Semantic Representation Aggregation for Reliable and Interpretable Battery Health Forecasting](https://arxiv.org/abs/2610.11567) | 提出语义表示聚合框架 Sera，通过融合基于规则的知识与大语言模型解读所构建的退化语义表示来补充时序建模，从而实现更可靠、更可解释的电池健康状态预测。 |
| [^135] | [Best of Both Worlds in Federated LSA: Speedup When Possible, Personalization Always](https://arxiv.org/abs/2610.11555) | 本文提出极简算法PF-LSA，通过将每个智能体的局部随机更新与全体智能体的平均更新相混合，在无需任何异构程度先验知识且不增加计算成本的情况下，同时实现了个性化收敛保证以及智能体数量足够相似时的线性加速。 |
| [^136] | [New Lower Bound and Upper Bounds on the Regret for Online Sparse Linear Regression](https://arxiv.org/abs/2610.11551) | 本文首次给出了在线稀疏线性回归极小极大遗憾的下界，并在无需正则性假设的条件下设计了具有更优遗憾上界的算法，刻画了该问题的信息论复杂度。 |
| [^137] | [$C_4$-Equivariant Flow Matching on Anisotropic Power-Diagram Graphs for Microstructure Generation](https://arxiv.org/abs/2610.11549) | 该论文提出了一种结合流匹配、图神经网络、$C_4$等变架构与各向异性幂图表示的生成模型，能够合成真实的多晶微结构并支持任意分辨率渲染，同时可利用免训练引导根据用户定义条件生成复杂微结构。 |
| [^138] | [Conditional Transfer from Controlled Pretraining Mixtures to Code](https://arxiv.org/abs/2610.11548) | 该研究区分了任务作为诊断、可教和可迁移三种不同的信号，通过受控预训练实验发现27个任务中14个具有可教性，且精选合成任务（10/12可教）与文献衍生探针任务（4/15可教）之间存在显著的不对称性。 |
| [^139] | [LAIR-Net: Leaky Alignment-Impulse Residual Networks for Tabular Regression](https://arxiv.org/abs/2610.11538) | LAIR-Net通过泄漏残差过渡将浅层学习的锚点注入隐状态演化中，实现了对隐状态的目标感知控制，在23个表格回归基准数据集上超越了八个随机化网络和十二个传统模型，并在非线性目标结构可学习时收益最大。 |
| [^140] | [An Efficient Quantum Circuit for Flow Model Execution Using Quantum Neural Networks](https://arxiv.org/abs/2610.11537) | 本文提出了一种基于QROM相位反冲框架并结合量子神经网络的紧凑量子线路，实现了波函数流的高效量子模拟，从而在资源开销显著降低的情况下在量子计算机上高效执行流模型。 |
| [^141] | [HAND: A Biologically-Inspired Activation Function that Improves Generalisation and Sample Efficiency in Image Classification](https://arxiv.org/abs/2610.11534) | 提出一种受生物学启发的激活函数HAND，通过融入归纳偏置使ConvNeXt-tiny在ImageNet1k上仅需25个训练周期即达到原模型200个周期的准确率，显著提升了泛化能力和样本效率。 |
| [^142] | [When to Intervene? State-Aware Sparse Manipulation in Federated Reinforcement Learning](https://arxiv.org/abs/2610.11523) | 该论文首次将“何时干预”确立为联邦强化学习拜占庭攻击的一个独立攻击维度，提出了利用本地策略不确定性选择稀疏干预状态并施加包络约束行为引导的V-BSA攻击方法，证明干预时机的选择会实质性影响攻击效果。 |
| [^143] | [Compactness and Consistency: A Conjoint Framework for Deep Graph Clustering](https://arxiv.org/abs/2610.11506) | 本文提出联合框架CoCo，利用图卷积滤波器从局部和全局两种视角学习鲁棒表示，并将其编码为低秩紧凑形式，从而在深度图聚类中同时捕获节点表示的紧凑性与一致性，克服了GNN局部消息传递难以建模全局关系以及图数据噪声冗余的问题。 |
| [^144] | [Learning qBIC Resonances across Metasurface Families in Dielectric Fourier Space](https://arxiv.org/abs/2610.11500) | 该论文提出将七个介质超表面家族映射到共享倒格子空间，利用K空间主干网络与可微Fano专家网络相结合，实现了跨几何结构的超窄qBIC共振精确学习，将共振位置误差从3.2 nm降至0.95 nm。 |
| [^145] | [PSI-SINDy: Post-Selection Inference for Sparse Identification of Nonlinear Dynamics](https://arxiv.org/abs/2610.11486) | 本文提出PSI-SINDy，通过选择后推断方法为SINDy识别出的动力学项提供有效的假设检验和置信区间，从而消除选择偏差并量化所选动力学项的统计可靠性。 |
| [^146] | [Evaluating Local Language Model Agents for Reproducible Data Engineering: An Empirical Software Engineering Study of Mobility Workflows](https://arxiv.org/abs/2610.11482) | 该研究构建了一个包含十五个移动性工作流任务的基准测试，通过确定性检查器系统评估了十种本地部署的开源权重LLM智能体在生成正确且可复现数据工程制品方面的能力，并量化了闭环工作区条件等因素的影响。 |
| [^147] | [Conditional Residual Prediction: Improving Autoregressive Video Diffusion without a Bidirectional Teacher](https://arxiv.org/abs/2610.11479) | 提出条件残差预测方法，仅从图像模型初始化即可训练因果视频扩散模型（全程无需双向教师或蒸馏），通过消除模型对真实历史的过度依赖来抑制自身生成历史误差的前向传播，从而提升自回归视频生成质量，且方法更简单、更易扩展。 |
| [^148] | [Rare Gate Disagreements Can Limit Plasticity: When Gradient Flow Mispredicts Finite-Batch SGD](https://arxiv.org/abs/2610.11475) | 该论文证明总体梯度流可能在定性上错误预测有限批量SGD：在双神经元ReLU回归中，当预训练时间超过 $\log(b/\eta)$ 后，源于罕见门控分歧的机制使在线SGD在指数级长的时间范围内以高概率丧失可塑性、无法适应目标任务，而梯度流却只需线性时间即可恢复。 |
| [^149] | [Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents](https://arxiv.org/abs/2610.11464) | 该论文提出将验证器本身作为进化对象——即由可检查的确定性缺陷检测器组成的表达式，通过锚定参考集一致性和输出共识来选择而非依据智能体分数——从而在自我改进循环中避免奖励作弊和共同盲点，并在MBPP+上比手工种子组合提升0.21的保留一致性。 |
| [^150] | [Feature Space Adaptation for Effortless Gaussian Process Flows](https://arxiv.org/abs/2610.11459) | 该论文通过引入核近似和基于扩散工作量的边际似然估计方法，首次实现了 FlowGP 框架内的超参数自动优化，使其能够高效扩展到高分辨率域并处理非高斯条件推断任务。 |
| [^151] | [Generative Adversarial Loops](https://arxiv.org/abs/2610.11458) | 提出生成对抗循环（GAL）框架，通过判别器智能体自动生成对抗性数据以暴露当前算法的弱点、生成器智能体发现新算法加以克服，从而实现目标设定的自动化，构建能够自我进步的AI研究系统，并成功应用于高效推理的近似算法。 |
| [^152] | [Zatom-2: Multitask Pretraining on Atomistic Data for Generative Modeling across Domains](https://arxiv.org/abs/2610.11454) | Zatom-2 是一个在约五百万个有机与无机原子结构上进行多任务预训练的原子生成模型，通过多尺度 Transformer 与条件流匹配实现跨化学、材料科学和生物学领域的统一生成建模。 |
| [^153] | [Closed-loop evaluation of LLM agents for embedded software development](https://arxiv.org/abs/2610.11447) | 该论文提出了一个包含五个嵌入式控制任务和四种反馈场景的基准测试，用于闭环评估LLM编码智能体在嵌入式软件开发中实现并自我验证设备行为的能力。 |
| [^154] | [MotiveMob: Motivation as Semantic Action for Closed-Loop Human Mobility Generation](https://arxiv.org/abs/2610.11442) | 提出MotiveMob框架，将移动动机作为语义动作显式建模，先推断移动背后的动机假设、再联合生成地点与时间，实现更符合人类决策过程的闭环移动性生成。 |
| [^155] | [BioBigBird: A Sparse Attention Model for Long-Range Dependency Processing in Biomedical Text](https://arxiv.org/abs/2610.11430) | BioBigBird是一种基于稀疏注意力机制的生物医学双向语言模型，可处理长达4096个token的长序列，并通过多任务学习联合优化命名实体识别与关系抽取，在BLURB基准上取得了与最先进模型相当的表现。 |
| [^156] | [Policy Alignment: New Signals for Membership Auditing in On-Policy Distillation](https://arxiv.org/abs/2610.11423) | 提出PAMA审计框架，首次利用教师引导的学生策略更新方向作为新信号，来审计在线策略蒸馏中私有提示的成员资格。 |
| [^157] | [Causal-fate dynamics of unrealized influence](https://arxiv.org/abs/2610.11422) | 本文提出“因果命运动力学”框架，用以刻画动力系统中未实现的影响如何在后续演化中被实现、潜伏或转换，并通过秀丽隐杆线虫神经网络模型与实际互联网路由两个例子说明未消解的影响可以长期保留未来相关性。 |
| [^158] | [Refinement as a Service: Algorithmic Predictor Refinement](https://arxiv.org/abs/2610.11415) | 该论文提出将校准预测器形式化为信号方案，并通过可观测线性信息刻画了何时能从多个校准预测器构造出既保留原有信息又不可再精炼的精炼校准预测器。 |
| [^159] | [MC-TRCM: Observation-Aware Recursive Fusion for Incomplete Mobile and Wearable Mental-Health Feature Views](https://arxiv.org/abs/2610.11408) | 提出MC-TRCM模型，将不完整的移动与可穿戴心理健康特征源作为独立token并显式建模缺失信息，通过递归预测头实现多源异构心理健康数据的鲁棒融合与预测。 |
| [^160] | [Beyond Distributional Fidelity: Causal-Penalized Diffusion for Synthetic Tabular Data](https://arxiv.org/abs/2610.11407) | 该论文首次将因果差异惩罚直接引入生成式表格扩散模型，提出因果惩罚化的 TabDDPM 训练框架，理论上证明高统计保真度不等于高因果保真度，并给出了因果正则化提升期望因果保真度的条件与实验验证。 |
| [^161] | [Learning from Hetero Density for Cryo-EM Protein Reconstruction](https://arxiv.org/abs/2610.11403) | CryoCue框架通过锚点监督检测器学习异质组分表示，并利用多尺度异质特征与预测候选点的类别、置信度和几何信息来指导冷冻电镜蛋白质重建，显著提升了异质组分附近的主链定位与结构重建精度。 |
| [^162] | [WAM-Cache: Staleness-Bounded KV Reuse for Efficient World Action Models](https://arxiv.org/abs/2610.11401) | WAM-Cache是一个免训练框架，通过跨块缓存复用视频DiT的KV表示，并依据动作专家的注意力位置（而非视觉漂移）仅稀疏刷新关键token，从而大幅降低世界动作模型闭环机器人操作中的预填充计算成本。 |
| [^163] | [Estimating great expectations under autoregressive language models with potentials](https://arxiv.org/abs/2610.11399) | 本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。 |
| [^164] | [CoPoE: Multimodal Fusion via Decomposable Disease-Coordinate Product-of-Experts for Missing-Modality Alzheimer's Diagnosis](https://arxiv.org/abs/2610.11394) | 提出了疾病坐标专家乘积框架CoPoE，将多模态证据映射到可解释的R/P/N/S四轴结构化潜在空间，通过掩码机制仅融合可用模态而无需合成缺失数据，从而在模态缺失情况下实现更可靠的阿尔茨海默病诊断。 |
| [^165] | [PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving](https://arxiv.org/abs/2610.11382) | 该论文提出PlanWAM，其核心创新在于让规划任务反向塑形未来状态表征，并通过潜在世界模型预测这种规划导向的未来表征，从而为端到端自动驾驶实现真正具有前瞻性的轨迹规划。 |
| [^166] | [From a Prompt to Repertoires: Evolving Functional REpertoires Enable LLM Continual Learning](https://arxiv.org/abs/2610.11373) | 提出演化功能技能库方法，通过将单一提示扩展为不断演化的多功能技能库，克服提示优化在持续学习中的灾难性遗忘与规则过拟合问题，使大语言模型无需更新参数即可持续习得新能力。 |
| [^167] | [SpatialOPSD: Self-Distilling Spatial Intelligence from Verified Coding Agent Traces](https://arxiv.org/abs/2610.11366) | 提出SpatialOPSD在线策略自蒸馏框架，将经验证的编程智能体执行轨迹作为特权信息内化到多模态大语言模型中，使其无需外部工具即可具备空间推理能力。 |
| [^168] | [Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data](https://arxiv.org/abs/2610.11362) | 提出伯努利流模型（BFM），通过构建数据分布与纯噪声之间统一的连续全局伯努利概率流路径，克服了传统二值扩散模型在低函数评估次数下因单步似然近似导致样本质量严重下降的问题，无需蒸馏或额外训练即可实现高效的二值数据生成。 |
| [^169] | [RaReCache: Bridging the Gap in Cross-Model KV Cache Reuse via Rank disagreement-based Selective Recomputation](https://arxiv.org/abs/2610.11358) | 提出RaReCache框架，利用秩分歧识别信息密集的关键token并对其进行选择性重计算，使大模型能够从小模型预填充的KV缓存中准确解码，从而实现跨不同规模模型间的高效KV缓存重用。 |
| [^170] | [RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty](https://arxiv.org/abs/2610.11352) | RL-ARC提出了一种校准感知训练框架，将推理置信度作为辅助信号来校准答案置信度——对正确回答施加推理引导正则化、对错误回答施加过度自信惩罚，从而在不牺牲推理性能的情况下改善大推理模型在分布内外场景中的校准并缓解过度自信问题。 |
| [^171] | [Sample-Efficient Generative Conformal Prediction](https://arxiv.org/abs/2610.11349) | 提出CASA方法，通过刻画额外样本的边际价值，在保证边际覆盖率的前提下自适应地在不同输入间分配采样预算，从而在相同预算下获得比固定采样数量更小的不确定性集合。 |
| [^172] | [DivMoE: Fine-Grained MoE Upcycling via Cross-Domain Expert Composition](https://arxiv.org/abs/2610.11317) | DivMoE是首个实现结构平衡路由的细粒度MoE升级框架，通过领域专门化的细粒度专家初始化和跨领域专家组合，解决了细粒度专家从单一源模型派生时路由崩溃、准确率降至接近随机水平的问题。 |
| [^173] | [Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers](https://arxiv.org/abs/2610.11315) | 该论文提出一个统一框架，将低秩辅助的W4A4训练后量化建模为耦合校准问题，通过“收缩海森矩阵”对已被低秩分量捕获的残差误差进行折扣，并结合激活噪声代理抑制激活量化误差，从而仅用秩4即可实现多模态扩散Transformer的高精度4比特量化。 |
| [^174] | [NP-Hardness of Minimizing Neurons in Two-Hidden-Layer ReLU Neural Networks](https://arxiv.org/abs/2610.11313) | 本文证明了在 $L^p$ 逼近约束下，精确计算双隐层ReLU神经网络逼近目标函数所需的最小隐藏神经元数量是NP难的，即使目标函数性质良好该结论依然成立。 |
| [^175] | [FloorSAV: Elucidating Spatial Audio-Visual Context with 2D Floormap for AV-LLMs](https://arxiv.org/abs/2610.11310) | 提出FloorSAV框架，通过渲染整合了3D点云、相机轨迹、空间音频与物体地标的动态2D平面地图，并将其作为同步流注入视听大语言模型，使其无需昂贵微调即可在单次推理中联合推理视觉、听觉与几何线索。 |
| [^176] | [From Geometry to Generalization: Why Row Normalization Can Beat Adam and Muon](https://arxiv.org/abs/2610.11309) | 该论文证明了在高维多分类任务中，行归一化凭借其类级欧几里得几何能够渐近保持总体决策边界方向，从而在总体精度上严格超越采用坐标级几何的Adam和采用谱几何的Muon等优化器。 |
| [^177] | [Memorization and Malign Generalization in Conditional Diffusion Models with Random Features](https://arxiv.org/abs/2610.11288) | 本文在高维比例极限下分析随机特征条件评分模型，揭示了条件扩散模型在过参数化时存在“恶性泛化”现象——增加模型宽度虽改善条件均值预测却降低条件内方差，且条件信息量越大，模型在更小宽度下就会记忆训练样本。 |
| [^178] | [Being-M0.7: A Latent World-Action Model for Humanoid Robots](https://arxiv.org/abs/2610.11283) | Being-M0.7 提出了一种潜在世界-动作模型，通过三阶段训练（预训练、机器人中期训练、动作后训练），将超过1万小时混合模态人类数据（纯视频、纯运动及成对视频-运动）中学习到的视觉-运动先验迁移到人形机器人的全身移动-操作控制中。 |
| [^179] | [How to post-train on a surrogate: Envelope sampling mitigates reward hacking](https://arxiv.org/abs/2610.11281) | 提出“包络采样”方法，利用少量带真实标注的输出对LLM评判器进行有理论保证的重新校准，从而在强化学习后训练中有效缓解奖励破解问题。 |
| [^180] | [Phonological Interference in Multilingual Speech Models](https://arxiv.org/abs/2610.11275) | 该研究揭示了多语言语音模型中的“音系干扰”这一系统性失败模式——模型错误地假设输入属于单一语言并强加其音系，导致在语码转换语音上丢失32%至79%的语言特有音素。 |
| [^181] | [Gated Memory: Admission-Controlled Memory Formation for Conversational AI](https://arxiv.org/abs/2610.11270) | 该论文提出Gated Memory框架，通过在对话与存储之间设置准入控制检查点，在事实提取前基于完整话语上下文评估候选事实，解决了关键上下文信号在提取时不可逆丢失这一制约记忆质量的瓶颈问题。 |
| [^182] | [Q-Capsule: A Localized Capsule-Based Quantum Neural Architecture for Barren Plateau Mitigation](https://arxiv.org/abs/2610.11261) | Q-Capsule是一种局部化胶囊型量子神经网络架构，通过寄存器分区、局部读出、稀疏胶囊间耦合和QFIM引导的自适应深度增长来缓解贫瘠高原问题，使梯度方差在量子比特规模增大时保持稳定。 |
| [^183] | [Residual spectral instabilities in representation learning](https://arxiv.org/abs/2610.11257) | 该论文将VAE中的逐维后验坍缩表述为围绕部分坍缩态的涨落理论，通过条件残差算符推导出坍缩方向的精确质量谱，并给出当解码器方差低于残差谱上边缘时坍缩维度可被再激活的临界判据。 |
| [^184] | [Neuro-Memory Fuzzy Inference System for Mimicking Human-like Car Following Behavior](https://arxiv.org/abs/2610.11252) | 本研究提出神经-记忆模糊推理系统（NeMeFIS），通过整合五种人类记忆类型对跟车行为的加减速进行非对称建模，在复现真实驾驶行为方面优于线性回归、ANFIS和LSTM等传统模型。 |
| [^185] | [V-CoLA: Vision Token Compression with Linear Attention](https://arxiv.org/abs/2610.11251) | 提出V-CoLA，一个专为线性注意力混合架构设计的无需训练的视觉Token压缩框架，通过唯一性感知的重要性准则和自适应Token合并策略，解决了现有压缩方法在该类架构下性能显著下降的问题。 |
| [^186] | [Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals](https://arxiv.org/abs/2610.11247) | 该研究发现大规模教师模型在在策略蒸馏中会导致基于梯度的学习信号过早消失，从而造成早期损失平台期，并从理论上为足够接近初始学生的教师证明了学习信号的局部恢复保证。 |
| [^187] | [Read What Matters: Query-Adaptive Quantization for KV Caches](https://arxiv.org/abs/2610.11245) | 提出ReadKV方法，通过渐进式编码实现KV缓存的查询自适应量化，根据每个解码查询动态分配键通道和值令牌的读取精度，并从理论上证明查询依赖的读取方案在相同读取预算下严格优于任何查询无关的方案。 |
| [^188] | [Low-Cost Sensor Calibration for Indoor Air Quality Monitoring: A Dataset, Evaluation Scenarios, and a Lightweight Model](https://arxiv.org/abs/2610.11236) | 该论文贡献了一个为期六个月、涵盖五个地点的低成本与参考室内空气质量传感器数据集，定义了评估空间泛化与时间漂移鲁棒性的四种校准评估场景，并提出了一种结合输入窗口压缩与残差机制的轻量级时间模型，从而克服了传统成对校准方法的局限性。 |
| [^189] | [SteerCast: Retrieval-Based Latent Steering for Decoder-Only Time Series Forecasting](https://arxiv.org/abs/2610.11229) | SteerCast提出了一种无需更新模型参数的推理时增强方法，通过检索相似历史窗口对应的引导向量并在自回归生成的每一步将其注入仅解码器预测模型的隐藏状态，从而引导预测朝着与相似训练案例一致的方向改进。 |
| [^190] | [Multimodal Graph Retrieval-Augmented Sequential Recommendation via Collaborative Filtering Paths](https://arxiv.org/abs/2610.11228) | 该论文提出MGRASRec框架，通过从经多模态相似度扩展的用户-物品交互图中检索协同过滤路径并注入MLLM提示，在引入邻居用户协同信号的同时无需额外开销即可筛选出与候选最相关的历史物品，从而高效提升多模态序列推荐性能。 |
| [^191] | [When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization](https://arxiv.org/abs/2610.11226) | 本文发现更低的重构损失不一定带来更好的模型性能甚至可能有害，并提出分布鲁棒量化（DRQ）方法，通过在受约束的激活分布集合上最小化最坏情况重构损失来精炼量化权重编码，从而提升低比特大语言模型量化的效果。 |
| [^192] | [Cross-species representation learning aligns mouse and human neural dynamics and tracks clinical drug efficacy](https://arxiv.org/abs/2610.11222) | 该研究提出双规则对比学习框架，通过跨物种表征学习从电生理数据中对齐小鼠与人类的神经动力学，识别保守的疾病相关特征并追踪临床药物疗效。 |
| [^193] | [BRACE: Differential Privacy for Dense Associative Memory with LSR Energy](https://arxiv.org/abs/2610.11218) | 本文提出了BRACE算法，一种针对LSR能量密集联想记忆的差分隐私检索机制，通过自适应校正边界敏感扰动的累积效应，实现了极小极大最优且与维度无关的检索误差率。 |
| [^194] | [The Lattice of Transition Laws](https://arxiv.org/abs/2610.11216) | 本文将扩散模型与自回归模型统一为同一个“腐蚀格”上的不同路径，通过定义解码调度的成本（即并行步骤所舍弃的依赖性），证明零成本调度的最少步数由数据的几何结构决定（例如等于图的树深度），从而可在解码前预测调度性能。 |
| [^195] | [Bridging KV-Cache Quantization and Linear Attention: From Theory to Pretrained Weight Migration](https://arxiv.org/abs/2610.11214) | 提出RAM-Net作为统一KV缓存量化与线性注意力的桥梁，通过离散地址空间上的软分配机制，在理论上证明其可分离读写重叠能局部逼近全注意力相似度，并支持从预训练权重迁移。 |
| [^196] | [Do Flatter Minima Drive Better Generalization? An Algorithmic Separation in Grokking](https://arxiv.org/abs/2610.11206) | 本研究以grokking现象为试验平台，揭示了平坦极小值并非泛化的因果驱动机制——单独使用SAM虽能产生更平坦的解却无法可靠诱导泛化转变，只有当SAM与权重衰减等泛化机制结合时才展现出促进泛化的作用。 |
| [^197] | [Dissecting Representation Structure in Vision Transformers: A Rigorous Architectural Study](https://arxiv.org/abs/2610.11205) | 该论文首次严谨分析了视觉Transformer跨架构尺度的特征信息，发现初始化时的特征坍缩问题并提出缓解方案，同时证明熵和最小特征值可作为泛化预测的可靠指标，用以指导高效的ViT设计。 |
| [^198] | [PageWeaver: KV-Guided Query Unions for Sparse Attention](https://arxiv.org/abs/2610.11201) | PageWeaver利用所选KV页的亲和性将查询分组为联合以共享页加载并填充Tensor Core瓦片，在保留每个查询原始支持集和完整输出所有权的前提下，在H200上实现了相比FlashInfer 1.70倍的几何平均加速。 |
| [^199] | [Selective Listening: Mechanism-Guided Control of Audio Influence in Large Audio-Language Models](https://arxiv.org/abs/2610.11196) | 提出ICAP-Gate方法，通过机制引导、任务条件化的方式控制大型音频-语言模型的后期音频通路，在防止无关音频干扰文本推理的同时，不损害依赖音频的任务（如语音识别）的性能。 |
| [^200] | [Predictive Multiplicity in Cell-Fate Assignment: Label-Free Rashomon Sets and the Limits of Per-Cell Certification](https://arxiv.org/abs/2610.11185) | 提出无标签框架FateMultiplicity构建Rashomon集合以量化单细胞轨迹推断中细胞命运分配的预测多重性，发现模型空间多样性而非规模是多重性的主要驱动因素，且单细胞认证的命运边际无法提供比原始拟合模型更可靠的分配结果。 |
| [^201] | [SACQ: Structured Decoding with Memory-Conditioned Refinement for Long-Horizon Forecasting](https://arxiv.org/abs/2610.11170) | SACQ是一种即插即用的结构化预测头，通过“粗粒度预测骨架+基于历史记忆交叉注意力的逐位置精炼”两阶段解码，并配合批自适应缩放的log-cosh损失，提升了长期时间序列预测的精度与对噪声的鲁棒性。 |
| [^202] | [PIVOT: Perplexity-Informed KD-to-RL Transition Scheduling for Vertical-Domain Few-Shot Distillation](https://arxiv.org/abs/2610.11167) | PIVOT提出了一种根据教师评估的序列困惑度在在线蒸馏（OPD）与GRPO强化学习之间动态路由样本的转换调度框架，取代了传统全局固定调度，使低困惑度样本进入强化学习精炼、高困惑度样本继续接受教师引导的领域知识获取，从而提升小语言模型在垂直领域小样本分类中的表现。 |
| [^203] | [CARE: A Lightweight Plug-in Gated Correction and Uncertainty-aware Module for Long-term Time Series Forecasting](https://arxiv.org/abs/2610.11165) | CARE是一种轻量级即插即用模块，通过并行校正分支对齐历史上下文学习残差校正，并利用不确定性感知的逐坐标风险门控进行有界更新，在不改变原有架构的情况下提升任意确定性长期时间序列预测模型的精度与可靠性。 |
| [^204] | [RideBench: A Large-Scale Exogenous-Aware Benchmark for Ride-Hailing Time Series Forecasting](https://arxiv.org/abs/2610.11164) | 该论文发布了基于滴滴数据构建的覆盖200个区域、跨越四年、半小时粒度的大规模网约车时序数据集 Ride-Hailing 及评测基准 RideBench，通过对30多种方法的评测证明未来已知的外生变量（天气、节假日、大型事件）能显著提升网约车预测精度。 |
| [^205] | [LadderEdit: Edit-Level Residual Compression for Memory-Efficient Lifelong Editing of LLMs](https://arxiv.org/abs/2610.11160) | LadderEdit通过将每条编辑先以低秩草图存储、仅对未满足契约的困难编辑沿阶梯逐级提升秩的方式压缩LoRA适配器，在保持编辑覆盖效果的同时将内存占用降低5.2倍，并支持5万次连续终身编辑。 |
| [^206] | [Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning](https://arxiv.org/abs/2610.11152) | 该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。 |
| [^207] | [Ranking Prior Alignment for Credit Risk Modeling: When Do External Priors Matter?](https://arxiv.org/abs/2610.11146) | 提出了一种模型无关的“排序先验对齐”框架，通过温度缩放的KL散度损失将来自领域专家、教师模型或大语言模型的外部排序先验统一蒸馏进神经或树类评分模型，以缓解标注数据稀缺的冷启动信贷评分难题。 |
| [^208] | [ActiveMedAgent: Cost-Aware Trajectory Learning for Multimodal Medical Diagnosis](https://arxiv.org/abs/2610.11140) | 该论文提出 ActiveMedAgent 框架，通过追踪诊断概率分布并按“诊断效用减去成本”对信息获取轨迹打分、离线训练轻量级 MLP 控制器，使冻结的视觉语言模型在多模态医学诊断中以更低成本、更少模态获得更准确的诊断结果。 |
| [^209] | [Accelerating Non-Smooth and Heavy-Tailed Sampling](https://arxiv.org/abs/2610.11139) | 本文提出非可逆锚定朗之万动力学（NALD）与非可逆反射锚定朗之万动力学（NRALD），通过引入循环漂移项，在无需目标密度导数的情况下加速对欧氏空间及受限域上非光滑、重尾目标分布的采样。 |
| [^210] | [Can a System-One LLM Perform Knowledge Tracing When Few or No Learners Are Logged?](https://arxiv.org/abs/2610.11135) | 该论文提出，现成的“系统一”LLM（Jev/JevKT）无需目标平台的学习者数据或仅需极少数据即可完成知识追踪，其性能超过28个深度知识追踪模型和“系统二”LLM方法，而API成本仅约为后者的百分之一。 |
| [^211] | [QUILT: Rethinking Sparse-Attention Prefill through Shared Query Execution](https://arxiv.org/abs/2610.11134) | QUILT通过联合处理相邻查询、复用共享KV条目，并借助移位比较集合分解（SCSD）将不规则集合操作转化为规则数据并行原语，显著减少了长上下文稀疏注意力预填充中的冗余内存流量和计算开销。 |
| [^212] | [Machine Learning Optimization for Enhanced OS Fingerprinting](https://arxiv.org/abs/2610.11133) | 本研究提出新命令行工具OsirisML，结合nPrint数据预处理与XGBoost机器学习算法，在CIC-IDS2017数据集上实现了高效的被动式操作系统指纹识别，最高准确率达97.66%。 |
| [^213] | [SFT-as-Context Mitigates Forgetting in Supervised Fine-Tuning](https://arxiv.org/abs/2610.11132) | 提出无需训练的SFT-as-context方法，让父模型将SFT模型的响应作为上下文通过上下文学习获得微调能力，从而在保留通用能力的同时缓解监督微调带来的遗忘问题。 |
| [^214] | [Dynamics as Code: On Model Compression via Dynamic System](https://arxiv.org/abs/2610.11115) | 该论文提出“动力学即代码”的新型模型压缩范式，证明了在丢番图条件下，无理缠绕的有限轨迹可构成权重空间上的ε-网，从而以可预测的方式将状态分辨率、解压误差与压缩比联系起来。 |
| [^215] | [Lapras: Latent Reasoning for Time Series Language Models](https://arxiv.org/abs/2610.11111) | Lapras是一个后训练框架，通过让时间序列语言模型在潜在空间而非离散语言标记中进行推理，避免了将连续时序信号转化为文字描述时的信息丢失与早期错误传播，从而生成与输入信号一致、更忠实的答案。 |
| [^216] | [Cova-PINN: Cross-Domain Conservation Physics-Informed Neural Network for Fluid-Solid Conjugate Heat Transfer in Complex Geometries](https://arxiv.org/abs/2610.11108) | Cova-PINN提出了一种多域物理信息神经网络框架，通过在局部尺度联合优化跨域复合控制体平衡、在全局换热器尺度优化成对壁面闭合，将守恒支撑与复杂几何中的热相互作用路径对齐，从而准确预测流固共轭传热中的端到端能量传递和出口温度。 |
| [^217] | [DaCe-DT: Data-Centric Offline Multi-Task Reinforcement Learning via Adaptive Prompts and Trajectory Correction for Heterogeneous Tasks](https://arxiv.org/abs/2610.11085) | 本文提出数据中心化的离线多任务强化学习框架DaCe-DT，通过长度门控提示掩码（LGPM）、检索增强提示构建（RAPC）和价值自适应回报校准（VARC）三项技术，有效解决了提示长度利用低效、提示语义不相关以及轨迹碎片化导致误导监督这三大数据瓶颈，从而提升模型对异构任务复杂度和数据质量的鲁棒性与泛化能力。 |
| [^218] | [A General $\widetilde{\Omega}(\sqrt{T \gamma_T})$ Lower Bound for Kernel Bandits](https://arxiv.org/abs/2610.11082) | 本文针对紧域上非常数连续核函数的核赌博机问题，建立了通用的 $\Omega(\sqrt{T\gamma_T/\log T})$ 极小极大遗憾下界，并证明该下界中的对数因子在一般情况下不可避免，从而在非常一般的意义上确立了现有 $\sqrt{T\gamma_T}$ 上界的接近最优性。 |
| [^219] | [Stability-Plasticity Balance via Singular-Vector Selection in LLM Continual Learning](https://arxiv.org/abs/2610.11076) | 提出SVC方法，将奇异向量通道作为可塑性分配的基本单元，通过选择性更新这些通道来平衡大语言模型持续学习中新能力的获取与预训练知识的保存。 |
| [^220] | [Optimally Pacing Budget Spending and Learning](https://arxiv.org/abs/2610.11074) | 该论文提出了首个在对抗性预算受限在线学习中达到近最优遗憾界 $O(D\sqrt{\log F}+ \sqrt{T\log F})$ 的全信息算法，并可扩展至在线资源分配问题，实现了此类任务中首个 $o(\sqrt{T})$ 的次线性遗憾保证。 |
| [^221] | [CityDeploy-Bench: Benchmarking Physics-Grounded Spatial Set Planning for Multi-Transmitter Network Deployment](https://arxiv.org/abs/2610.11065) | 该论文提出CityDeploy-Bench基准，将多发射机网络部署重新建模为统一射线追踪验证下的物理接地空间集合规划问题，并揭示部署质量的关键在于效用模型能否捕捉发射机间的集体物理交互，而非单纯依靠更强的搜索能力。 |
| [^222] | [Measuring and Mitigating Solution Mode Collapse in RLVR](https://arxiv.org/abs/2610.11064) | 本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。 |
| [^223] | [Emergent Inverse-Depth Scaling From Nonlinearity In Attention](https://arxiv.org/abs/2610.11063) | 该论文发现，注意力的非线性性使模型能够选择性聚焦相关词元、让强弱谱方向并行学习，从而在所有数据谱下涌现出损失随深度呈逆深度衰减的缩放规律，为模型深度缩放定律提供了全新机制。 |
| [^224] | [Learning What to Trust in Multimodal Learning under Noisy Supervision](https://arxiv.org/abs/2610.11057) | 该论文提出REFINE框架，通过理论分析表示结构与噪声检测能力之间的关系，联合利用融合表示和单模态表示来构建更可靠的多模态标签噪声检测器，从而解决噪声监督下多模态学习依赖高质量标签的难题。 |
| [^225] | [AgentHorizon: Evaluating Agentic Judges for Long-Horizon Computer-Use Tasks](https://arxiv.org/abs/2610.11050) | 该论文提出了AgentHorizon基准，包含1,373个来自166小时人工录制轨迹的计算机使用任务，通过指令-轨迹配对设计（包括交换指令构造的负样本）来评估智能体裁判在长时程、跨应用任务中识别轨迹违规和副作用的能力。 |
| [^226] | [FedAlphaEdit: Null-Space-Aligned Merging for Collaborative Knowledge Editing](https://arxiv.org/abs/2610.11033) | 提出首个在统一零空间原则下对齐本地编辑与服务器端合并规则的协同知识编辑框架 FedAlphaEdit，使多个机构无需共享原始编辑请求即可安全整合各自的知识编辑。 |
| [^227] | [NOMOS: Compiling Written Policies into Statically Verified Tool-Call Gates for LLM Agents](https://arxiv.org/abs/2610.11030) | NOMOS是一个四遍编译器，可将自然语言策略编译为经过静态验证的确定性工具调用门控，将LLM代理在状态变更调用中的策略违规率从66.3%降至2.6%。 |
| [^228] | [A Graph Neural Network for Global Daily Fire Radiative Power Prediction at Medium-Range Lead Times](https://arxiv.org/abs/2610.11022) | 该研究开发了一个基于时空图神经网络的数据驱动模型，利用最新可用的观测数据预测未来1至7天的全球每日火辐射功率（FRP），克服了卫星产品延迟与预报期间火输入固定不变这两项业务运行限制。 |
| [^229] | [Mid-Training Language Models on Raw Video](https://arxiv.org/abs/2610.11019) | 该论文首次证明无字幕、无文本损失的原始视频可作为语言模型的中期训练数据，通过预测下一个视觉token的方式，在不损害文本能力的前提下显著提升了模型在视频和图像基准上的性能。 |
| [^230] | [Curating Always-Loaded Context for LLM Agents: A Capacitated Assortment Model with Censored Feedback](https://arxiv.org/abs/2610.11007) | 该论文将LLM智能体常驻上下文文件的策划问题形式化为容量受限的组合选择问题，证明了最优文件大小的上界，并表明盲目追加所有有价值的指令在净价值上可能比精选最优子集任意更差。 |
| [^231] | [Reconstruction of Multiscale Plasma Dynamics Across Operating Regimes](https://arxiv.org/abs/2610.11004) | 该论文提出ReMAIN网络，保留SHRED的循环时间编码并用经特征级仿射调制的U-Net替换其全连接解码器，从而能够从稀疏传感器测量中重构跨运行工况的多尺度、空间分辨的等离子体动力学。 |
| [^232] | [Low-rank tensor structure of precipitation and its application to satellite-reference merging](https://arxiv.org/abs/2610.11000) | 该研究揭示了降水的低秩张量结构，并提出基于张量的TMerge框架，通过共享低秩时空因子融合卫星降水与稀疏基准观测，将美国本土IMERG降水产品的相关系数从0.53显著提升至0.85。 |
| [^233] | [F$^3$NO: Frequency-Decomposed Finite-Time Flow-map Neural Operators with Cross-Scale Conditioning](https://arxiv.org/abs/2610.10998) | 提出频率分解流映射神经算子F$^3$NO，通过低频特征引导高频信息的跨尺度细化，并结合分段并行预测与递归传播策略，在五个PDE基准上超越了自回归和直接预测基线的预测精度。 |
| [^234] | [Invariant-Measure Reasoners: Stable Representations for Latent Reasoning](https://arxiv.org/abs/2610.10996) | 提出不变测度推理器（ImR），以潜在状态在紧致子集上的长期分布即不变测度作为稳定表示，并基于预测头输出在该测度下的期望进行预测，从而解决潜在推理模型因潜在状态持续变化而导致的预测不稳定问题。 |
| [^235] | [Region-Aware CLS Token Augmentation for Fine-Grained Image Retrieval](https://arxiv.org/abs/2610.10991) | 该论文提出通过为视觉Transformer中的CLS token和寄存器token分别匹配空间区域token（伙伴patch）来增强语义表示，从而提升细粒度图像检索的性能。 |
| [^236] | [Omni-Diffusion-Distill: Few-Step Distillation of Unified Multimodal Diffusion Large Language Models](https://arxiv.org/abs/2610.10990) | 提出统一的两阶段蒸馏框架 Omni-Diffusion-Distill，在离散 token 空间中同时蒸馏图像与文本的生成和理解能力，大幅减少统一多模态扩散大语言模型的推理步数并保持其双重视觉-语言能力。 |
| [^237] | [Multi-Bandwidth Distribution Matching Distillation: On the Equivalence of Distribution Matching Distillation and Drifting Models](https://arxiv.org/abs/2610.10989) | 本文证明了分布匹配蒸馏（DMD/DMD2）与漂移模型在数学上的等价性，并据此提出了一种多带宽分布匹配蒸馏方法。 |
| [^238] | [Optimizing Large Language Models with Chained LMOs](https://arxiv.org/abs/2610.10975) | 提出链式线性最小化预言机（chained LMOs）框架以统一解释矩阵归一化组合类优化器，并基于此提出TensorChain优化器，在Qwen3预训练中相比Muon平均节省9.6%的token。 |
| [^239] | [Budgeted Multi-Source Counterfactual Annotation for Off-Policy Evaluation](https://arxiv.org/abs/2610.10974) | 该论文提出了预算约束下面向离线策略评估的多源反事实标注获取框架，将标注分配建模为整数规划问题，通过带动态规划子程序的主要化-最小化算法求解以最小化估计器方差，并刻画了标注产生价值的阈值条件。 |
| [^240] | [Transferability of Learned States in Neural PDE Solvers](https://arxiv.org/abs/2610.10972) | 提出“复用契约”评估框架，将神经PDE求解器中学习状态的迁移收益与解精度和计算成本解耦，并实证发现固定预测器的收益会随校正算法不同而反转。 |
| [^241] | [ASPIRE: Saddle-Point Discovery through Set Prediction and Physical Refinement](https://arxiv.org/abs/2610.10969) | ASPIRE框架利用等变集合预测器Ev-Quiformer从单一原子环境预测多个鞍点候选，并通过Dimer搜索在原始原子间势上进行物理精炼，解决了热激活扩散与缺陷演化模拟中鞍点发现的计算瓶颈。 |
| [^242] | [Spectrally Targeted Muon](https://arxiv.org/abs/2610.10965) | 该论文提出谱靶向Muon优化器，通过阈值仅对特定范围的奇异值进行正交化，使其在归一化SGD和Muon之间插值，从而揭示Muon优化器成功背后的频谱机制。 |
| [^243] | [TRACE: A Governance Framework for Measuring Explainability Debt in Production AI Systems](https://arxiv.org/abs/2610.10957) | 本文提出TRACE七工具治理框架，以可解释性债务分数（EDS）为核心，首次系统性衡量、追踪并修复生产级AI系统中逐渐累积的“可解释性债务”——即当监管者或受影响个体要求问责时无法解释单个决策的治理负债。 |
| [^244] | [SPD-MetaFormer is what you need for small-data brain decoding](https://arxiv.org/abs/2610.10952) | 该研究发现基于SPD流形的注意力模型学到的注意力权重接近均匀、可被简单均匀权重替代而几乎不损失预测性能，据此表明架构结构比学习加权更关键，提出SPD-MetaFormer足以胜任小数据脑解码任务。 |
| [^245] | [RH-Detect: A Unified Benchmark for Reward Hacking Detection](https://arxiv.org/abs/2610.10947) | 提出 RH-Detect 统一基准，将十一个公开数据集的奖励作弊样本整合到通用模式中，评估发现现成大语言模型无需训练即可有效检测奖励作弊（最佳汇总 AUROC 达 0.962），但在多轮工具使用场景下性能显著下降。 |
| [^246] | [AI4Fire: Evaluating Large Language Models on Wildfire Tasks](https://arxiv.org/abs/2610.10946) | AI4Fire 基准首次在五个野火任务上对多个大语言模型进行成对的“裸跑”与“接地”零样本评估，发现接地信息（如只读SQL工具）在直接包含答案时能大幅提升准确率（从至多16%提升到至少88%），但简单规则基线仍然难以被超越。 |
| [^247] | [GHARP: Real-time Gaussian Head Animation from Large-scale Reconstruction Prior](https://arxiv.org/abs/2610.10945) | 该论文提出GHARP方法，将头部动画解耦为离线的身份建模阶段和运行时的轻量级残差预测阶段，并利用预训练重建模型的语义结构化潜在空间，实现从少量图像和表情信号驱动、可在移动设备上实时运行的3D头部动画。 |
| [^248] | [StoreBench: A Live-Commerce Environment for Evaluating and Training Autonomous Operator Agents](https://arxiv.org/abs/2610.10942) | StoreBench是一个让智能体在生产级电商后端上经营线上服装店的实时环境，通过全天候动态市场、校准的通过阈值和抗操纵的奖励机制，来评估和训练自主运营智能体的长程规划与经济决策能力。 |
| [^249] | [Higher-Order Morphology Priors for Quadruped Reinforcement Learning Under Actuator Degradation](https://arxiv.org/abs/2610.10934) | 该论文将四足机器人建模为包含肢体级和躯干级高阶胞元的胞复形，并结合霍奇消息传递机制，证明高阶形态先验可作为有效的归纳偏置，显著提升强化学习策略在执行器退化情形下的全身补偿能力与泛化性能。 |
| [^250] | [Rethinking the Tradeoff Between Temporal Encoding and Nonlinear Computation in Spiking Language Models](https://arxiv.org/abs/2610.10933) | Spora通过联合设计二值脉冲编码（UBS与BBS）和注意力算子，突破了脉冲语言模型中时间编码容量与非线性计算开销之间的权衡，仅用四个时间步即在GLUE和CoLA上超越SpikeLM。 |
| [^251] | [World-Model Policy Arbiter for Goal-Conditioned Reinforcement Learning](https://arxiv.org/abs/2610.10932) | 提出了世界模型策略仲裁器（WMPA），一个测试时框架，将多个冻结的目标条件策略视为一个策略组合，利用世界模型在每个状态动态选择最合适的策略执行，从而在离线目标条件强化学习中超越任何单一算法的表现。 |
| [^252] | [Coefficient Calibration as Selection Pressure in Symbolic Regression](https://arxiv.org/abs/2610.10931) | 提出了一种名为DSCA的系数校准策略，通过在不同协变量分布的数据分区上独立校准并取参数均值，使校准过程本身成为进化选择压力，从而避免符号回归偏爱依赖特定样本系数的结构。 |
| [^253] | [Adaptive Multi-Discriminator WGAN Framework for Resource-Constrained Internet of Vehicles Using Reinforcement Learning and Game Theory](https://arxiv.org/abs/2610.10926) | 本文提出一种融合强化学习与博弈论协调的自适应多判别器WGAN（MD-WGAN）框架，能够在资源受限、拓扑动态变化的车联网环境中同时兼顾高精度、高效资源利用、低时延和低通信开销。 |
| [^254] | [Language Models for Page-Level Layout Decisions in E-commerce Search](https://arxiv.org/abs/2610.10920) | 该论文研究了利用语言模型离线评估电商搜索页面级布局决策（如在特定位置插入二级堆栈）是否对用户有益，从而减少对昂贵在线A/B测试的依赖。 |
| [^255] | [Implementation Guidelines for Data Quality Metrics](https://arxiv.org/abs/2610.10919) | 本文通过对ISO/IEC 25024和ISO/IEC 5259标准中的数据级度量指标进行分类并提供实施指南，使ISO数据质量度量指标变得可执行，从而弥合了数据质量维度的文本定义与实际工具实现之间的差距。 |
| [^256] | [When Listening Becomes Easier: Scrubbing Visual Cues for Shortcut-Free VLAs](https://arxiv.org/abs/2610.10912) | 本文提出无需策略回滚的表征层指标“动作边界”来衡量VLA模型对视觉捷径学习的易感性，并据此提出“任务擦除”方法擦除视觉线索、增强模型对语言指令的依赖，从而构建无捷径的VLA模型。 |
| [^257] | [Power Side-Channel Membership Inference Attack on Embedded Machine Learning](https://arxiv.org/abs/2610.10909) | 提出了一种名为PSCMIA的功耗侧信道成员推理攻击，无需预测概率甚至预测标签，即可直接从功耗轨迹中推断嵌入式机器学习模型的训练数据成员关系。 |
| [^258] | [How Hackable Is Your Speech Quality Metric? A Corrected Protocol, a Benchmark, and What Patching Buys](https://arxiv.org/abs/2610.10899) | 该论文提出了衡量语音质量预测器可攻破性的修正协议（以未扰动往返处理为参照并取多个随机种子攻击者的最坏结果），据此建立的基准显示四种主流预测器被攻破率差异巨大（NISQA 90%、SSL-MOS 21%、DNSMOS 14%、UTMOS 6%），且“攻击-检测-打补丁”闭环仅在自身攻击空间内提供加固。 |
| [^259] | [Gen-PINNs: Generative Adversarial Physics Informed Neural Networks for solving partial differential equations](https://arxiv.org/abs/2610.10897) | 提出Gen-PINNs框架，将生成对抗网络与物理信息神经网络相结合，通过动态加权物理损失和判别器评估PDE残差，显著改进了含尖锐或冲击波前行为的偏微分方程的无数据求解。 |
| [^260] | [Barron Optimal Transport I: Generative Modeling](https://arxiv.org/abs/2610.10875) | 本文提出了一种以神经网络复杂度为传输代价的Barron最优传输新框架，将Benamou-Brenier动力学最优传输中的平均动能$L^2$能量替换为Barron能量，为生成建模中寻找最高效的神经网络传输表示奠定了理论基础。 |
| [^261] | [Transformed Samplers with Variance Reduction](https://arxiv.org/abs/2610.10870) | 该论文提出通过学习双射（如归一化流）将MCMC采样器变换到潜空间，把简单参考分布上的泊松方程精确解推广到一般目标分布，从而获得显式控制变量以实现方差缩减。 |
| [^262] | [Conversational Voice Aesthetic Model with Reinforcement Learning from Human Listeners](https://arxiv.org/abs/2610.10868) | 该论文提出CVAM语音大语言模型，通过合成美学描述的监督微调和基于约3万条人类听众标注的组相对策略优化（GRPO），实现对语音性别、音高、语速、情感、表达方式等九项美学属性的预测，其与人类听众的一致性超越了Gemini 3.1 Pro和开源语音LLM。 |
| [^263] | [Shape irregularity of Life-Like Network Automaton rules as an indicator of classification performance](https://arxiv.org/abs/2610.10867) | 本文提出“锯齿性”指标，通过量化类生命网络自动机转移函数与锯齿形状的相似程度，作为混沌性与敏感性的理论代理，从而避免昂贵的穷举搜索，实现高效的网络分类规则选择。 |
| [^264] | [A mesh-based neural energy method for the simulation of heterogeneous composites](https://arxiv.org/abs/2610.10862) | 本文提出基于网格的神经能量方法，通过形函数插值施加运动学约束以抑制非物理位移振荡，并用代数形函数导数替代自动微分、采用高阶高斯求积精确计算能量，从而实现非均质复合材料的高效精确模拟。 |
| [^265] | [Enabling Preference-driven Unlearning in Few-step Distilled Text-to-Image Diffusion Models](https://arxiv.org/abs/2610.10859) | 该论文提出了一种基于偏好优化（DPO）的遗忘框架，使少步蒸馏的文本到图像扩散模型无需依赖多步去噪假设或代价高昂的重新蒸馏，即可有效遗忘有害内容。 |
| [^266] | [RFChipAgent: Multi-Agentic AI Flow for Analog/RF Chip Design](https://arxiv.org/abs/2610.10858) | RFChipAgent是首个基于大语言模型多智能体协作的端到端模拟/射频芯片设计自动化流程，通过多模态RAG知识提取、拓扑选择、原理图与测试台自动搭建、闭环混合电路尺寸优化四大技术支柱，在人类监督下协同完成完整的模拟/射频电路设计。 |
| [^267] | [KDFP: A first-principles approach to knowledge distillation in large language models](https://arxiv.org/abs/2610.10854) | 本文提出KDFP，一种基于第一性原理的白盒通用知识蒸馏新方法，使大语言模型在9个基准测试上比现有方法提升1.6%–4.9%，填补了LLM通用知识蒸馏的研究空白。 |
| [^268] | [Similar Predictive Fit but Different Latent Dynamics: Characterizing Learned Dynamical Structure in Personalized Models of Brain Disorders](https://arxiv.org/abs/2610.10850) | 该研究提出将EEG基础模型的潜在表示映射为个性化转移依赖图的方法，发现即使预测拟合度相似，癫痫患者的潜在动力学依赖结构也显著比非癫痫者更密集，表明模型学到的动力学结构能揭示超越预测准确率的临床相关差异。 |
| [^269] | [Amortized Off-Policy Evaluation for LLMs](https://arxiv.org/abs/2610.10848) | 提出PFN-OPE方法，通过在大语言模型响应构建的上下文多臂老虎机任务分布上一次性预训练先验数据拟合网络，将离线策略评估进行摊销，从而同时应对策略偏移与奖励偏移的双重挑战，无需针对每个新任务从头拟合。 |
| [^270] | [Real Long-Term Memory for AI: A 50-Million-Token Window That Is Faster and Cheaper Than Recompute](https://arxiv.org/abs/2610.10845) | 本文提出并验证了一个名为galahad-kv的记忆层，通过将KV状态加密保存到本地NVMe磁盘并按需逐字节精确加载，实现了5000万令牌的超长上下文记忆，速度比重计算快2.8至4.3倍、GPU能耗降低8.8至12.3倍，且GPU内存占用在整个处理过程中保持恒定。 |
| [^271] | [On the Clock: Towards Punctual and Productive Time-Budgeted AI Agents](https://arxiv.org/abs/2610.10833) | 该论文首次系统研究了小型LLM智能体在明确时间预算下的行为，发现仅在提示词中说明预算无法让智能体有效管理时间，并提出通过暴露计时信息的运行环境机制和预算感知强化学习两类干预措施，使智能体既守时又能高效利用时间。 |
| [^272] | [MotherTree: Meta-learning on synthetic data improves decision tree training](https://arxiv.org/abs/2610.10832) | 提出了MotherTree，一个通过合成先验元学习决策树归纳的表格Transformer，无需参考树监督即可在单次前向传播中输出与经典训练形式相同、可独立审计部署的轴对齐决策树。 |
| [^273] | [Conformal Prediction under Partial Verification](https://arxiv.org/abs/2610.10829) | 该论文提出了一种部分验证方法，通过刻画校准证书并在校准样本间协调验证，在产生与完整验证完全相同的预测集的同时，将验证成本降低15-82%。 |
| [^274] | [Diagnosing and Recovering from Observation-Space Shift at Long-Horizon Skill Seams](https://arxiv.org/abs/2610.10810) | 该论文发现长时程机器人操作中技能串联失败的主因是前序技能遗留的场景状态偏移（而非机器人关节构型或目标物体），并提出一个完全学习的“检测-恢复-重启”系统来诊断并恢复技能衔接处的失败。 |
| [^275] | [Evaluating Rubric Generation with Interventional Transfer](https://arxiv.org/abs/2610.10809) | 本文提出干预性迁移（IT）方法，通过扰动回答并观察评分标准是否随之同步通过/未通过变化来评估LLM生成的评分标准质量，并在HealthBench案例中展示了不同形式的干预性迁移可用于评估评分标准在不同任务中的实用性。 |
| [^276] | [Controlled Acquisition and Abstention in Three-Channel Score Conflicts](https://arxiv.org/abs/2610.10808) | 该论文构建了一个受控的三分数基准，用于研究音频、视频、文本分数冲突时策略应何时付费获取第三个分数或选择弃权，并证明简单的阈值策略能以更少的信息请求实现更高的针对性决策准确率和效用。 |
| [^277] | [Symbolic Density Estimators for Unnormalized Distributions](https://arxiv.org/abs/2610.10807) | 本文提出一个将深度生成模型与符号回归相结合的框架，利用相互作用范围、基函数集合等领域先验知识，从观测样本中自动估计未归一化分布的符号表达式，减少了对专家人工选择函数形式的依赖。 |
| [^278] | [Whose Ground Truth? Embracing Ambiguity in Human-Centered AI](https://arxiv.org/abs/2610.10805) | 这篇立场论文主张AI开发应摒弃“单一确定真值”的传统假设，转而建模人类合理判断的解读空间，将有意义的模糊性与标注噪声区分开来，从而实现真正以人为中心的AI。 |
| [^279] | [How Many Repeated Pairwise Comparisons Are Needed for Ranking under Heterogeneity?](https://arxiv.org/abs/2610.10795) | 该论文证明了在异构 Bradley-Terry 模型下，通过改进的 MLE 变体算法和随机化“俄罗斯轮盘赌”算法，每个用户-任务上下文仅需 $O(\log(1/\Delta))$ 次重复成对比较即可恢复排序，并证明该对数复杂度是最优的。 |
| [^280] | [Calibrating Ambiguity Set via Diagnostic Transport for Distributionally Robust Optimization](https://arxiv.org/abs/2610.10793) | 本文提出诊断传输DRO（DT-DRO），利用留出校准数据和条件概率积分变换来诊断预测误差，自适应地调整模糊集的中心与几何结构，从而在保证决策风险可控的同时避免DRO决策过度保守。 |
| [^281] | [NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime](https://arxiv.org/abs/2610.10787) | NavGPT-3 提出了一个类似操作系统的分层导航运行时框架，将具备长时程推理能力的语言模型与低延迟的 VLA 动作策略通过多线程调度机制相结合，使机器人能通过线程中断与切换快速响应突发真实事件，其 8B VLA 模型在 R2R-CE 上取得了 74.51 SR 的领先性能。 |
| [^282] | [Plan-and-Patch: Diffusion Language Models for Agentic Planning](https://arxiv.org/abs/2610.10786) | 提出Plan-and-Patch框架，利用扩散语言模型通过并行去掩码生成结构化的类程序计划，并借助仅填充受影响区域、保持前后步骤不变的局部修复机制，实现对计划的高效修订。 |
| [^283] | [MemoWM: How World Models Change What Agents Need to Remember](https://arxiv.org/abs/2610.10778) | MemoWM提出了一种基于世界模型的记忆分配框架，利用共享预测压缩经验存储并重建省略内容，在五个长期智能体记忆基准上实现了答案准确率超越最强基线2.62个百分点，同时相对最高效基线减少53.9%的每条经验存储量。 |
| [^284] | [NEMORA: Neural Equivariant Multipole Operators for Long-Range Atomistic Learning](https://arxiv.org/abs/2610.10776) | NEMORA 提出了快速多极子方法的神经等变扩展，实现了可学习的长程等变相互作用传输，兼顾多尺度多体表达能力与对更大体系的高效扩展性，克服了现有方法在长程信息传递上的局限。 |
| [^285] | [Strategic Investment Decision Making for Value Creation in Energy Transition: A Reinforcement Learning Approach](https://arxiv.org/abs/2610.10768) | 该论文开发了一个定制的能源模拟环境和基于强化学习的多标准顺序决策框架，帮助能源公司在不确定性下于油气、可再生能源和碳减排三大领域之间进行战略资金分配，从而最大化能源转型中的价值创造。 |
| [^286] | [CPU-Auth: Device Fingerprinting for Authentication via DVFS Side-Channel](https://arxiv.org/abs/2610.10766) | 该论文提出了CPU-Auth，一种通过在浏览器内远程测量CPU动态电压频率调节（DVFS）行为这一侧信道，利用CPU物理特性的独特差异实现设备指纹识别与用户认证的新型机制。 |
| [^287] | [What can linear attention learn from nonlinear teachers in-context?](https://arxiv.org/abs/2610.10761) | 本文的核心创新是建立了“非线性-噪声等价性”理论：线性注意力在上下文学习中只提取目标函数的线性Hermite分量，剩余非线性结构等价于有效噪声，从而使线性理论的结论可以迁移到非线性任务。 |
| [^288] | [Clarify, Then Focus: Statement Normalization for Conversation Analytics at Scale](https://arxiv.org/abs/2610.10758) | 提出语句规范化方法，将对话转化为带说话者归属、来源引用和语义标签的简短语句，使语义更明确并支持按需证据选择，从而提升大规模企业对话分析中下游模型的表现。 |
| [^289] | [Conversational Task Disambiguation over Tabular Data: Leakage-Aware Formulation, Benchmark Suite, and Training](https://arxiv.org/abs/2610.10740) | 提出“歧义可验证任务”的形式化框架，将智能体分解为提问策略与求解策略、环境分解为oracle与验证器，从而实现任务消歧与求解能力的独立评估，并给出oracle泄漏的形式化定义与无裁判的泄漏诊断。 |
| [^290] | [Deep Learning vs. Statistical Models for Multi-Horizon Price Forecasting of Second-Hand Electronics: A Systematic Benchmark](https://arxiv.org/abs/2610.10727) | 本文首次为二手电子产品价格预测建立了系统性多时间尺度基准，在波兰在线市场大规模数据上对比评估了11种统计模型与深度学习模型在1至365天不同预测周期上的表现。 |
| [^291] | [On linearity or non-linearity in machine learning for quantum chaotic dynamics](https://arxiv.org/abs/2610.10697) | 该研究将量子混沌动力学预测转化为时间序列预测问题，以里德堡原子阵列可实现的PXP链为基准，系统比较了非线性Transformer与简单线性模型DLinear在从遍历到量子多体疤痕态等不同动力学区间中的预测能力。 |
| [^292] | [Learning infinite context windows in recurrent architectures via spatial neural computing](https://arxiv.org/abs/2610.10690) | 该论文提出一种用偏微分方程控制的空间演化场替代传统神经元间通信的二阶循环模型，受皮层波启发，使结构化时空模式成为隐式高容量记忆，实现了参数固定但感受野无界的无限上下文窗口。 |
| [^293] | [Explaining the Saliency Map Sparsity of Adversarially-Trained Neural Networks](https://arxiv.org/abs/2610.10666) | 本文首次为对抗训练神经网络梯度显著性图的稀疏性现象提供了理论解释，证明随着数据点和神经元数量增长，网络的最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器，从而自然产生了稀疏性。 |
| [^294] | [BEANS-Next and ROOTS: Broadening Audio-Language Capabilities for Bioacoustics](https://arxiv.org/abs/2610.10663) | 本文提出生物声学基准 BEANS-Next 和大规模训练资源 ROOTS，揭示现有音频-语言模型在物种分类等传统标签识别任务之外的生物声学能力有限，为拓展更广泛的音频-语言任务提供了评估与训练基础。 |
| [^295] | [Teaching PPG How not Who: Fixed-Effects Distillation from ECG](https://arxiv.org/abs/2610.10662) | 该论文提出固定效应蒸馏方法，通过在ECG到PPG的知识蒸馏中减去每条记录的均值以精确消除“个体特质”，使学到的个体内心血管状态变化一致性翻倍以上，从而让蒸馏从记忆“身份”转向学习“状态变化”。 |
| [^296] | [Beyond Owls: Subliminal Learning Can Transfer Learned Capabilities and Backdoors](https://arxiv.org/abs/2610.10657) | 本文证明潜意识学习不仅能传递简单偏好，还能通过语义无关数据的蒸馏传递模型习得的复杂能力（如预测随机初始化MLP的输出）及后门，且其分布外泛化优于直接优化的引导向量，暗示蒸馏可能在无人察觉的情况下传播隐蔽的模型失调。 |
| [^297] | [Nullify: Null-Space Activation Steering for Training-Free LLM Unlearning](https://arxiv.org/abs/2610.10655) | Nullify提出了一种免训练、非破坏性的激活引导方法，通过引导向量将隐私相关激活重定向离开记忆答案，并利用零空间约束保护保留查询的激活，从而在实现高质量遗忘的同时近乎无损地保持模型效用。 |
| [^298] | [PXtal: Learning to Align Powder X-Ray Diffraction and Crystal Structures under Information Asymmetry across Modalities](https://arxiv.org/abs/2610.10653) | 该论文提出PXtal框架，通过非平衡最优传输与耦合层面的广义KL散度，在粉末X射线衍射与晶体结构之间存在物理固有信息不对称的情况下学习对齐表示，在PXRD到晶体的候选检索任务中显著优于基线模型。 |
| [^299] | [Beyond the Ergodic Wall: A Discrete Geometric Physics Sandbox for Analysing AI Scaling Limits and Complexity Collapse](https://arxiv.org/abs/2610.10651) | 提出以全息E8投影引擎驱动的离散几何物理沙盒，揭示深度学习的遍历天花板，主张通过时空受限的非遍历观察者注入语义新颖性，并以硬物理遏制促使成熟ASI将人机共生视为避免模型坍缩的热力学必然。 |
| [^300] | [Leakage-Controlled Multimodal Learning for Diagnosis and Progression Prediction in Alzheimer's Disease Research](https://arxiv.org/abs/2610.10648) | 该研究提出了一种防数据泄漏的多模态多任务框架，融合适配的SFCN MRI编码器、因果临床Transformer与ODE-GRU动态建模，在阿尔茨海默病的诊断和认知进展预测中实现了优异且稳健的性能。 |
| [^301] | [From Log-Odds to Shapley Values: An Explanatory Geometry for the Weighted Naive Bayes Classifier](https://arxiv.org/abs/2610.10642) | 该论文证明加权朴素贝叶斯分类器中基于对数几率的监督距离与解析Shapley值向量之间的ℓ1距离完全一致，从而为监督距离、局部解释与预测行为之间建立了形式化的联系。 |
| [^302] | [Coverage-Aware Reasoning with Medical Tokens for Diagnosis Prediction](https://arxiv.org/abs/2610.10641) | 该论文提出CARing框架，通过医疗令牌表示ICD诊断并引入覆盖感知的强化学习奖励机制，解决LLM在下次就诊多诊断预测中只集中于少数诊断、以及ICD编码被分词拆散而难以高效推理大规模疾病词表的问题。 |
| [^303] | [How Many Directions Must a Truncated Diffusion Sampler Retain? Matching Bounds Under Power-Law Spectra](https://arxiv.org/abs/2610.10640) | 针对幂律协方差谱数据，论文证明了截断扩散采样器所需保留方向数的匹配上下界，并揭示仅保留信号超过噪声水平方向的策略仍会产生不可忽略的总体截断误差。 |
| [^304] | [Visible Reasoning Is Not a Universal Optimizer: Persona- and Thinking-Dependent Effects in Analytics Code Generation](https://arxiv.org/abs/2610.10639) | 该论文通过跨 SQL 与 pandas 双语言、交叉角色设定与多种思考指令的受控执行基准实验发现，显式思维链推理并非普遍有效，其效果因角色表述、目标语言和思考格式而异，“用 SQL/Python 思考”这类匹配目标语言的指令并不能可靠带来提升。 |
| [^305] | [D-SLR: The Disjoint Row-Sparse plus Low-Rank Decomposition](https://arxiv.org/abs/2610.10636) | 本文提出D-SLR分解，一种截断SVD的闭式直接替代方法，它将矩阵行不相交地划分为逐字存储或低秩近似两类，在平方误差下以更少参数即可达到联合最优，且在相同代价下不会差于截断SVD。 |
| [^306] | [Phase-HDC: Replacing Optimizer History with Gradient Thresholds in Discrete Phase Learning](https://arxiv.org/abs/2610.10630) | Phase-HDC 提出了一种基于梯度阈值的离散相位更新规则，用一步式角度旋转取代优化器历史记录，使超维相位记忆分类器在仅存储模型本身的情况下即可达到与 Adam 相当的精度，存储量降低数倍至二十余倍。 |
| [^307] | [Sample-Efficiency of Kolmogorov-Arnold Networks](https://arxiv.org/abs/2610.10627) | 本研究通过系统性的计算实验证明，Kolmogorov-Arnold网络在强化学习与函数拟合任务中可仅用少40%的样本达到与多层感知机相当的性能，训练过程中相对性能提升最高达50%，且对奖励噪声具有鲁棒性。 |
| [^308] | [Exact SO(3)-Equivariant Isotropic Kernels for Rotation-Robust Neural Dynamics](https://arxiv.org/abs/2610.10626) | 本文提出不变量条件化各向同性核神经算子（IKNO），通过仅使用旋转不变标量与随数据共同旋转的向量方向来构建局部相互作用，使三维Navier–Stokes等向量值偏微分方程的神经代理模型实现数值精度上的精确SO(3)等变性，彻底消除无约束图模型无法根除的坐标依赖问题。 |
| [^309] | [Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models](https://arxiv.org/abs/2610.10625) | 循环语言模型在不同递归深度上的安全性并不一致——更深的推理深度更容易遭受越狱攻击，且同一攻击可跨深度迁移，而SFT与偏好对齐均无法消除这种跨深度安全差距。 |
| [^310] | [Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models](https://arxiv.org/abs/2610.10623) | LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。 |
| [^311] | [Self-Organization from Constrained Geometric Radiation](https://arxiv.org/abs/2610.10621) | 本文发现在无外部驱动的封闭系统中，通过约束诱导的几何辐射可实现自发自组织，揭示了应力积累—超指数辐射—混沌崩塌—分形极限环的四阶段循环及新型“楔形吸引子”，并提出四个联合充分条件和三个定理作为理论支撑。 |
| [^312] | [MRCert: Towards Post-deployment Patch Robustness Certification for Adversarially Patched Samples via Type-specific Masking](https://arxiv.org/abs/2610.10617) | 提出首个基于掩码的认证恢复防御方法MRCert，通过对良性样本和对抗性补丁样本推断类型特定的必要属性，在保持高预测准确率的同时验证对抗性补丁样本标签的良性，实现部署后的补丁鲁棒性认证。 |
| [^313] | [When Routing Reveals Membership: Privacy Leakage from MoE Router Telemetry](https://arxiv.org/abs/2610.10616) | 本文首次揭示MoE模型的推理路由遥测数据会泄露微调数据的成员身份信息，提出的路由器增强成员推断攻击在九种设置中将1%假阳性率下的真阳性率提升2.7至9.4个百分点。 |
| [^314] | [JevForest: Path Voting for Budgeted Feature Acquisition](https://arxiv.org/abs/2610.10615) | 提出JevForest特征获取策略，通过聚合自助采样树的路径提议、以全局信息增益加权并用共享掩码分类器预测，在有限观测预算下自适应选择最值得查询的特征，但其实验效果在不同数据集上并不一致。 |
| [^315] | [Temporal transformer CAN encoder with federated lightweight heads for anomaly detection](https://arxiv.org/abs/2610.10613) | 本文提出了一种基于时序Transformer CAN编码器与联邦轻量级检测头的隐私保护框架，能够有效检测车载CAN总线网络中细微的时序与上下文异常。 |
| [^316] | [The optimal information complexity of VC learning](https://arxiv.org/abs/2610.10600) | 本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。 |
| [^317] | [Coverage, Not Difficulty, Sets How Much Synthetic Data an Activation Probe Needs](https://arxiv.org/abs/2610.10594) | 该研究发现激活探针所需的合成数据量取决于所监控概念的覆盖度而非任务难度——高风险和有害性探针仅需约80个样本即可接近性能平台期，而指令遵循探针需要数倍的样本量，且该排序在不同模型和真实数据上均成立。 |
| [^318] | [SPERA: Spherical Prior EEG Foundation Model with Geometry- and Frequency-Aware Latent Prediction](https://arxiv.org/abs/2610.10571) | SPERA是一个基于JEPA在潜在空间进行预测的脑电基础模型，通过勒让德多项式球面先验编码不同电极几何排布、分解式时空注意力以及关系型频谱正则化，克服了受试者、设备和电极排布异质性的挑战。 |
| [^319] | [Strategic Governance of AI Models in Earth Science](https://arxiv.org/abs/2610.10560) | 该论文指出预报技能与物理可靠性是两种截然不同的属性，并针对地球科学中的AI模型提出了涵盖训练数据、微调、行为测试、机制可解释性和输出验证五个方面的物理评估优先事项。 |
| [^320] | [GaussianBench: Physics-Fidelity Evaluation for Gaussian Scene Representations](https://arxiv.org/abs/2610.10554) | 提出GaussianBench——首个面向物理集成高斯场景表示的物理保真度评估基准，通过冻结场景、模拟器无关的评分器和解析或实测参考，系统测试守恒性、连续介质响应、异质材料耦合、协方差传输、热相变和反事实响应，并区分物理失败与视觉合理性的差异。 |
| [^321] | [Freeze the Decoder, Heal the Encoder: Parameter-Efficient Adaptation for SVD-Based KV-Cache Compression](https://arxiv.org/abs/2610.10552) | 该论文揭示了在共享学习率下比较参数高效微调方法的统计学陷阱：SVD式KV缓存压缩修复中“冻结解码器、仅修复编码器”看似显著的优势，实际上是各方案可训练参数数量悬殊导致的偏差，而非真实效果。 |
| [^322] | [Interpretable Memory Models for Spaced Repetition](https://arxiv.org/abs/2610.10548) | 提出SBD记忆模型，在保持与最先进模型几乎相同准确性的同时，可解释性更强且模型体积缩小80%。 |
| [^323] | [Multi-Agent Coordination via Support-Preserving Distillation](https://arxiv.org/abs/2610.10087) | 提出MoSDOT方法，利用条件半离散最优传输将噪声样本确定性分配到有限模式支撑集上，解决基于流的多智能体教师在蒸馏过程中模式冲突误差传播给学生模型的问题。 |
| [^324] | [Origins of Universal Machine Learning Force-Field Errors in Multicomponent Materials](https://arxiv.org/abs/2610.09837) | 该论文构建了包含7,599个多组分构型的基准测试集，系统评估了11个预训练通用机器学习力场在多组分材料中的表现，并揭示了训练参考覆盖率不足和局部几何异质性增大是力场误差的主要来源。 |
| [^325] | [When Should an In-Context Learner Expand Its Hypothesis Space?](https://arxiv.org/abs/2610.09471) | 该论文将上下文学习中“何时扩展假设空间”的问题形式化为有代价的序贯决策，并通过结构修正环境证明：修正决策是一个由价格、时间范围和查询等价值因素决定的价值边界，而非单纯的证据阈值。 |
| [^326] | [Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data](https://arxiv.org/abs/2610.09411) | 提出一种简单的Wasserstein Procrustes方法，通过粗粒度几何初始化估计单一正交映射，无需任何配对数据即可实现独立训练模型的跨模态表征对齐，并证明标准几何对齐指标可准确预测对齐的可行性。 |
| [^327] | [An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information](https://arxiv.org/abs/2610.09206) | 论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。 |
| [^328] | [Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.09145) | 在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。 |
| [^329] | [MaRK: Markov-adapted Recurrent Kernels for Dynamic Operator Conditioning in State Space Models](https://arxiv.org/abs/2610.09092) | MaRK提出了一种动态算子条件化框架，将上下文向量直接映射为对冻结SSM循环算子参数（A、B、C、D、Δ）的有界调制，使每个扩散时间步都能动态重塑模型的输入输出记忆核。 |
| [^330] | [SPIN: Shadow Predictive Indexer for Sparse Attention](https://arxiv.org/abs/2610.09025) | SPIN 提出基于历史的轻量级预测机制来识别重要 KV 块，避免每个解码步骤对完整 KV 缓存评分，在保持任务质量的同时实现 30-40% 的稀疏度，并将 vLLM 服务的吞吐量最高提升 14.9%、token 间中位延迟最高降低 13.2%。 |
| [^331] | [Directed Temporal Representations for Offline Visual Control](https://arxiv.org/abs/2610.08960) | 该论文提出DTRC方法，在冻结世界模型特征之上学习与时间可达性对齐的有向时间拟度量表征，并利用其时间进度信号作为评论家，实现离线视觉目标条件策略的直接学习。 |
| [^332] | [An Empirical Study of Agent Skills' Downstream Utility](https://arxiv.org/abs/2610.08875) | 本文通过在87个SkillsBench任务上的实证研究，将智能体技能的下游效用量化为相对于无技能基线的通过率差异，并揭示效用如何取决于技能内容、执行配置与多技能组织方式。 |
| [^333] | [Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States](https://arxiv.org/abs/2610.08818) | 该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。 |
| [^334] | [Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness](https://arxiv.org/abs/2610.08560) | 该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。 |
| [^335] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^336] | [The Geometry of Empowerment](https://arxiv.org/abs/2610.07796) | 本文将赋权最大化与技能学习方法相联系，提出了解释赋权的新几何框架，解答了赋权与结构中心性之间联系的长期开放问题，并揭示了信息几何与奖励几何的区别，为构建可扩展的赋权最大化方法奠定理论基础。 |
| [^337] | [Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding](https://arxiv.org/abs/2610.07713) | 提出受神经运动层级结构启发的NHN网络，通过引入生理学归纳偏置学习紧凑的潜在神经运动状态，从而在跨用户、跨会话的sEMG解码中实现鲁棒泛化。 |
| [^338] | [Learning to Decide, Not to Reason: Parameter-Efficient Decision Operators via Low-Rank Activation Steering](https://arxiv.org/abs/2610.06950) | 该论文提出一种仅用2.3万至33万参数、通过行为克隆训练的低秩激活转向决策算子，能在不损失精度的情况下将3685个token的长推理压缩为6个token的快速决策，训练成本比现有强化学习方法低约两个数量级。 |
| [^339] | [SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics](https://arxiv.org/abs/2610.06513) | 提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。 |
| [^340] | [Universality and Convergence of Generative Flows](https://arxiv.org/abs/2610.05490) | 该论文证明基于差值的平衡损失能以不依赖策略的显式常数在全变差意义下保证采样器的准确性，而基于比率的流匹配损失不具备这种保证，且反向策略决定了损失能否收敛到零以及梯度下降的收敛速度。 |
| [^341] | [Population Scaling or Data Dilution? Dynamics of Local Topology Evolution in Decentralized Learning](https://arxiv.org/abs/2610.05476) | 该论文揭示了去中心化学习中客户端数量扩展与数据稀释、拓扑混合及通信容量的耦合效应，证明环形拓扑的谱隙按 $\Theta(N^{-2})$ 衰减导致共识变慢，并提出基于“朋友的朋友”发现的局部自适应拓扑演化方法 LFHE，实验表明保持本地数据集规模固定可显著缓解规模扩展带来的性能惩罚。 |
| [^342] | [Measuring Learned Monotone Temporal Aggregation at Matched Admissibility](https://arxiv.org/abs/2610.05196) | 该论文在对比双方单调可容许性完全匹配的前提下，用构造上单调的循环网络（EWMA与高水位标记的可学习变换）度量学习型时间聚合的价值，并通过函数回归揭示出一条“涵盖边界”——学习到的单调通道可复现几何加权可分手工统计量族。 |
| [^343] | [How Long, Not How Close: A Learned Temporal Metric for Planning in Latent World Models](https://arxiv.org/abs/2610.04988) | 提出TEMPO，一种学习“距离目标还差多少环境步数”而非“与目标多相似”的时序距离规划代价，无需改动或训练世界模型、也无需奖励或标签即可显著提升潜空间世界模型在远距离目标下的规划能力。 |
| [^344] | [The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer](https://arxiv.org/abs/2610.03827) | 相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。 |
| [^345] | [SpectralCache: Accelerating Diffusion-Based World Models via Spectral Feature Caching](https://arxiv.org/abs/2610.02660) | SpectralCache揭示了扩散世界模型特征在相邻去噪步骤中的谱稳定性，通过无需训练的谱缓存框架复用奇异子空间并线性外推奇异值，从而跳过冗余的Transformer计算，显著加速推理。 |
| [^346] | [Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations](https://arxiv.org/abs/2610.02432) | 本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。 |
| [^347] | [Budgeted Cache Repair for Cross-Context KV-Cache Reuse](https://arxiv.org/abs/2610.02233) | 该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。 |
| [^348] | [Stochastic Rounding in Low-Precision Transformer Inference: A Variable-Precision Emulation Study of a Small GPT-2](https://arxiv.org/abs/2610.01889) | 该研究通过变精度随机舍入（VPSR）算法将PRISM舍入库扩展至任意精度，并从理论与实验两方面揭示：低精度Transformer推理中随机舍入与就近舍入的优劣取决于运算位点，SR的误差以O(√n u)增长而RN以O(n u)增长，这一差距在长MLP下投影等长线性投影中最为明显。 |
| [^349] | [LLM Persona Unlearning](https://arxiv.org/abs/2609.39882) | 该论文提出“人格遗忘”任务及PersonaUnlearnBench基准，通过权重级编辑使大语言模型中的指定人格难以被诱发，并发现标准遗忘方法无法在不牺牲生成质量或通用能力的情况下可靠地抹除目标人格。 |
| [^350] | [A Parameter-Free Zeroth-Order Method with Covariance Matrix Adaptation and Effective Dimension](https://arxiv.org/abs/2609.38561) | 本文提出POEM-CMA，一种通过协方差矩阵自适应实现各向异性采样并引入有效维度概念的无参数零阶优化方法，将采样集中于信息量最大的方向，并以问题的内在维度取代环境维度进行复杂度分析。 |
| [^351] | [Diffusion-2BC: Hybrid Diffusion and Regression Training for Offline Behavior Cloning in Autonomous Driving](https://arxiv.org/abs/2609.38472) | 本文提出 Diffusion-2BC，在共享视觉编码器上将扩散去噪目标与辅助的确定性行为克隆损失联合训练（辅助分支仅用于训练、推理仍基于扩散），从而解决了自动驾驶离线行为克隆中单观测多有效动作的多模态问题并提升了闭环性能的稳定性。 |
| [^352] | [CompOrca: Corpus-Scale Compliance Labelling of Instruction-Tuning Data](https://arxiv.org/abs/2609.37807) | 该论文提出了 CompOrca，利用开源大模型评判器对整个 OpenOrca 语料库（超过 420 万条样本）进行五次独立判定，首次实现了语料库规模的合规性标注，并发布带投票计数的标注结果，以支持对拒答与不服从行为的研究。 |
| [^353] | [MA-JEPA: Joint-Embedding World Models for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2609.33563) | MA-JEPA提出了一种基于联合嵌入预测（JEPA）架构的随机世界模型，用预测目标表示替代观测重构，实现了基于模型的集中训练、分散执行的多智能体强化学习。 |
| [^354] | [SMAT: Simple and Efficient Merge-Aware Training](https://arxiv.org/abs/2609.33437) | SMAT将常见模型合并操作抽象为缩放、掩码和扰动三种基本操作，通过在采样生成的模拟合并参数上联合优化专家损失与期望损失，实现了以极小训练开销显著提升合并后性能的简单高效合并感知训练方法。 |
| [^355] | [When to Evict, Not What to Keep: Draft-Guided Eviction for Training-Free KV-Cache Compression](https://arxiv.org/abs/2609.33334) | 该论文提出草稿引导驱逐（DGE）方法，将KV缓存驱逐时机从预填充结束推迟到基于完整缓存起草出前两个答案token之后，通过利用答案自身前缀生成的查询来指导驱逐决策，解决了传统“优化保留什么”策略的补偿效应和选择效应失效问题，实现免训练的KV缓存压缩。 |
| [^356] | [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102) | 提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。 |
| [^357] | [When Post-Processing Fairness Constraints Help and When They Harm: Evidence from Eight Cross-Domain Evaluations](https://arxiv.org/abs/2609.26955) | 该论文提出FAPE四阶段公平性审计框架，通过对八个领域的评估发现，后处理公平性干预（Fairlearn的ThresholdOptimizer）的有效性取决于基线差异大小——在多数高差异场景中能改善公平性，但在低差异场景中可能反而有害。 |
| [^358] | [MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents](https://arxiv.org/abs/2609.24259) | 该论文提出了基于现实记忆系统场景的MemCalib基准，揭示前沿LLM普遍存在过度或不足使用记忆的问题，并针对后训练算法改进的单向性缺陷提出了MemCalib-RL来优化智能体的记忆使用能力。 |
| [^359] | [CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning](https://arxiv.org/abs/2609.23257) | CTRL框架将语义推理与定量预测解耦，利用LLM智能体作为控制器分析预测误差的分解成分并输出控制信号，再由轻量级残差解码器转化为预测修正，从而提升非平稳环境下时间序列预测的稳定性与可解释性。 |
| [^360] | [Embedding Models Measure in Peculiar Ways](https://arxiv.org/abs/2609.20821) | 该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。 |
| [^361] | [Label-free steering: Compressing test-time reinforcement learning into bias-only subspaces](https://arxiv.org/abs/2609.18587) | 该论文提出无标签仅偏置测试时强化学习方法，以多数投票伪标签为奖励、仅优化约10万个偏置参数，即可在数学、视觉语言和音频推理等多个任务上达到与全参数方法相当甚至更优的性能。 |
| [^362] | [Salesforce Koa: An Enterprise Language Model for Agentic Tool Use](https://arxiv.org/abs/2609.15066) | Salesforce Koa 是一个基于 Nemotron-3-Super-120B 并通过 GRPO 强化学习后训练的企业级语言模型，其核心创新在于“仿真到奖励”流水线——将工作流规范扩展为以角色为条件的多轮任务，并以成功工具使用作为任务解决奖励，从而在保持通用性能的同时显著提升智能体工具使用能力。 |
| [^363] | [Data-free On-policy Distillation](https://arxiv.org/abs/2609.14193) | 该研究发现在线策略蒸馏（OPD）对训练数据几乎不敏感——仅八个提示即可媲美1.7万题的数据集，且跨领域数据仍能保留90%以上的收益，表明OPD传递的是教师的推理方式而非具体知识。 |
| [^364] | [The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes](https://arxiv.org/abs/2609.10739) | 论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。 |
| [^365] | [MoEMB: Scaling Universal Multimodal Embeddings with Efficient Mixture-of-Experts Models](https://arxiv.org/abs/2609.08663) | MoEMB 提出利用混合专家沿专家维度扩展通用多模态嵌入模型，在保持单向量、非自回归编码与低延迟推理的同时提升编码器容量，并避免对简单任务产生冗余计算。 |
| [^366] | [Risk-Conditioned Fine-Tuning of Large Language Models](https://arxiv.org/abs/2609.08064) | 提出风险条件化RLHF框架，通过训练单一策略提供连续可调的风险控制接口，使用户能够在推理时灵活选择不同的风险规避程度，而无需重新训练或部署多个特定风险的模型。 |
| [^367] | [Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?](https://arxiv.org/abs/2609.00482) | 该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。 |
| [^368] | [LM-X: Explainable Action Modeling with Progress, Event, and Uncertainty Prediction for Generalist Robot Manipulation](https://arxiv.org/abs/2608.25757) | 本文提出LM-X框架，通过在线预测任务进度、事件转换和局部不确定性三个显式信号，使VLA策略的动作生成具有内在可解释性，无需事后解释。 |
| [^369] | [K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold](https://arxiv.org/abs/2608.19584) | 本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。 |
| [^370] | [Quantum Multi-Armed Bandits and Linear Bandits: Lower Bounds and Algorithms](https://arxiv.org/abs/2608.14319) | 本文首次证明了量子多臂老虎机和有限动作量子线性老虎机的极小极大遗憾下界，解决了是否存在与时间范围无关的遗憾问题，并提供了基于多项式方法和Remez型不等式的新证明技术。 |
| [^371] | [DYSANOS Generative Dynamic Smooth Arbitrage-free Non-parametric Option Surfaces](https://arxiv.org/abs/2608.12587) | 本文提出首个生成式无套利期权曲面模型DYSANOS，能够生成未来多年每日期权价格路径，并验证其优于传统隐含波动率PCA模型。 |
| [^372] | [Robust and Efficient Noisy-Label Time-Series Classification via Dynamic Time Warping Based Granular Ball Computing](https://arxiv.org/abs/2608.11704) | DTW-GBC通过粒度球计算，在保持分类鲁棒性的同时大幅减少推理计算量，有效应对标签噪声问题。 |
| [^373] | [Capacity Confounds and Coverage Guarantees in Adaptive Sub-model Federated Learning](https://arxiv.org/abs/2608.07157) | 该研究发现子模型联邦学习中基于训练更新散度估计的客户端数据异质性信号实际上被设备容量所混淆，与真实数据异质性几乎无关，从而质疑了按估计的数据异质性自适应分配子模型容量的可行性，并给出了相应的覆盖保证。 |
| [^374] | [How Much Does Message Passing Matter? A Drop-In Study of GNN Layers for Neural Network Graph Regression](https://arxiv.org/abs/2607.26404) | 该研究通过在固定架构、损失和训练方案下即插即用地替换十种消息传递层，首次系统评估了消息传递层选择对神经网络图级回归的重要影响，发现其性能差异显著且最佳选择取决于图的规模。 |
| [^375] | [Do Sheaf Neural Networks Use Holonomy? A Measure--Intervene--Control Study](https://arxiv.org/abs/2607.19514) | 该研究通过“测量—干预—对照”实验发现，层神经网络仅在任务需要利用几何结构时（如三角形计数）才会学习到非平凡的和乐旋转，其预测确实依赖于所学到的联络，但这种几何机制并非性能优势的必要条件，因为岭回归等更简单的基线方法仍然表现更好。 |
| [^376] | [Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL](https://arxiv.org/abs/2607.16204) | 该论文提出将基于文本的世界建模形式化为可引导的转移动力学问题，并利用掩码扩散语言模型克服自回归模型的从左到右偏差，构建了强大且可引导的世界模型，为智能体强化学习按需提供多样化、可扩展的训练环境。 |
| [^377] | [Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization](https://arxiv.org/abs/2607.10169) | 该论文揭示了PPO-Clip的根本缺陷在于使用欧几里得度量衡量策略差异、与策略黎曼流形的内在几何结构不符从而导致探索坍缩，并提出了黎曼等距策略优化（RIPO），通过在黎曼流形上保证等距策略更新来有效平衡探索与利用。 |
| [^378] | [System-Prompt Conditioning and Hidden-State Geometry in Four Open-Weight Models: Corrections and What Survives](https://arxiv.org/abs/2607.09842) | 本文是对先前“系统提示词会在开源语言模型隐藏状态中留下几何指纹”这一研究的勘误版：审计发现原论文的曲率统计量、置换检验方法和若干关键测量均存在错误，本版修正这些问题并说明哪些结论仍然成立。 |
| [^379] | [Group Invariant Spectral Embedding](https://arxiv.org/abs/2607.08987) | 该论文提出将紧致李群对称性直接融入谱嵌入的相似度核，证明了由此构造的图拉普拉斯算子逐点收敛到商空间上的显式二阶微分算子，且由于有效维度的降低而获得更快的收敛速度。 |
| [^380] | [Directly Optimizing Mean Demographic Parity for Nonlinear Regression](https://arxiv.org/abs/2607.05098) | 该论文提出DPVar（条件均值预测的方差）这一公平性度量，首次实现了非线性回归中平均人口均等准则的直接优化，克服了以往方法仅适用于线性预测器或低维敏感属性、且因过度约束而损害精度的局限。 |
| [^381] | [Koopman operator theory: fundamentals, control, and applications](https://arxiv.org/abs/2607.01819) | 这是一篇关于库普曼算子理论的教程论文，系统介绍了其基本原理、数据驱动近似方法（如EDMD）及其在控制器设计（如库普曼MPC）中的应用，并提供了开源代码仿真示例。 |
| [^382] | [Reward Observability and the Limits of Offline Checkpoint Selection in RSSM World Models](https://arxiv.org/abs/2607.01736) | 该论文系统评估了RSSM世界模型在LunarLander任务上的闭环性能，表明基于世界模型想象训练的策略能以约65倍更少的真实交互数据匹敌无模型强化学习，并提出奖励可观测性分数（ROF）来揭示离线检查点选择方法的局限性。 |
| [^383] | [A samplewise backpropagation method for neural networks driven by fractional Brownian motion](https://arxiv.org/abs/2606.29438) | 本文提出一种由分数布朗运动驱动的随机神经网络，通过离散随机最大值原理构造伴随递推实现逐样本反向传播，证明了逐样本随机梯度下降的均方收敛性，并表明分数阶驱动在长记忆恢复和鲁棒性方面优于布朗运动和确定性基线。 |
| [^384] | [The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/abs/2606.26294) | 本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。 |
| [^385] | [Minimax PAC Bounds for Learning in Exogenous Contextual MDPs](https://arxiv.org/abs/2606.25170) | 该论文提出了一个在查询已知前后分配采样预算的新型PAC学习框架，并针对带外生上下文的折扣马尔可夫决策过程中的策略评估、最优值估计和最优策略提取任务，给出了极小极大最优的样本复杂度界。 |
| [^386] | [Neural Conjugate Aggregation: Identifiable Unsupervised Multi-Sensor Regression under Heterogeneous Sensor Bias](https://arxiv.org/abs/2606.22200) | 提出神经共轭聚合模型（NCAM），一个结合神经网络与共轭高斯推断的层次贝叶斯框架，在无真值标签条件下实现多传感器数据融合，并通过传感器锚定与方差正则化解决不可识别性问题，提供解析可处理且不确定性分解的后验估计。 |
| [^387] | [Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2606.18810) | 该论文提出一种自条件式信用分配方法，通过将模型以自身经验证的采样轨迹为条件构造自教师模型，利用逐token KL散度区分常规token与关键推理步骤，从而在不依赖外部教师或特权信息的情况下提升可验证奖励强化学习的训练效率。 |
| [^388] | [S4oP: Operator-level Pruning of Structured State Space Models for Resource-Constrained Devices](https://arxiv.org/abs/2606.18096) | 本文提出了首个针对结构化状态空间模型（S4/S4D）的算子级剪枝方法，通过交替进行结构化掩码与微调来逐步剪枝模型算子，在保持预测性能的同时显著降低推理成本，使其适用于资源受限设备。 |
| [^389] | [Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO](https://arxiv.org/abs/2606.09701) | 本文提出一种基于GRPO的攻击-防御协同训练框架，通过多LLM评判奖励通道与GDPO优势计算以及从攻击者单独训练到协同训练的课程策略，实现了高效可迁移的攻击生成并同步提升防御者的安全能力。 |
| [^390] | [GRASP: Geometry-aware Residual Alignment for Scalable Pretraining Data Attribution](https://arxiv.org/abs/2606.06892) | 论文提出GRASP方法，将数据归因重构为子集级反事实效用预测，通过二次几何惩罚建模子集间交互，并结合低维特征草图与严格有限的置信下界选择协议，在不依赖隐藏调参的情况下实现预训练规模的高效归因，其反事实子集保真度的秩相关性较现有基线提升一倍以上。 |
| [^391] | [Learning Implicit Bias in Generative Spaces for Accelerating Protein Dynamics Emulation](https://arxiv.org/abs/2606.01833) | 该方法在预训练蛋白质动力学生成模拟器的生成空间中引入隐式的历史依赖偏置，引导采样远离已生成的结构，从而将构象多样性提升35%并实现对罕见状态的长时程零样本探索。 |
| [^392] | [Decision-Focused On-Policy Learning for Contextual Linear Optimization with Partial Feedback](https://arxiv.org/abs/2606.01081) | 该论文提出了一种在部分反馈下用于序贯情境线性优化的决策聚焦在线策略学习方法，通过结合得分函数估计器与决策聚焦即插即用组分的混合梯度估计器来训练随机化的预测后优化策略。 |
| [^393] | [Memory by Design: Probabilistic Sequence Layers](https://arxiv.org/abs/2605.31163) | 本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。 |
| [^394] | [Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap](https://arxiv.org/abs/2605.28888) | 高德地图提出GPlan框架，通过渐进式隐式思维链蒸馏将大语言模型的推理能力内化到轻量级模型中，在严格延迟约束下实现逻辑连贯且物理可执行的生成式时空意图序列推荐。 |
| [^395] | [MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems](https://arxiv.org/abs/2605.28732) | 该论文提出了MemTrace框架，将LLM记忆流水线转化为可执行的记忆演化图，并结合MemTraceBench基准和自动归因方法，实现了对记忆系统错误的细粒度追踪与根因定位。 |
| [^396] | [Trust Region Q Adjoint Matching](https://arxiv.org/abs/2605.27079) | 本文提出信赖域Q伴随匹配（TRQAM），通过在流策略采样过程中引入信赖域参数λ来精确加权路径空间KL散度，并借助投影对偶下降自适应控制策略更新幅度，从而解决了QAM中评论家误差被指数级放大导致性能崩溃的问题，实现了对预训练流策略的稳定离策略微调。 |
| [^397] | [ROAR: Retrieval Opportunity-Aware Refinement for Zero-Shot Time Series Forecasting](https://arxiv.org/abs/2605.24911) | 该论文提出ROAR框架，通过依据基础预测难度和检索候选相对改进来加权训练目标，并联合学习候选聚合、门控校正与预测模块校准，从而有效捕捉并利用检索增强在零样本时间序列预测中的改进机会。 |
| [^398] | [MARGIN: Runtime Confidence Calibration for Multi-Agent Foundation Model Coordination](https://arxiv.org/abs/2605.22949) | MARGIN是一种运行时置信度校准方法，通过从观察到的答案结果中在线学习模型特定的置信度修正，无需重训模型或校准集即可提升多智能体基础模型协作中集体决策的可靠性。 |
| [^399] | [The Distillation Game: Adaptive Evaluations & Efficient Defenses](https://arxiv.org/abs/2605.22737) | 该论文将模型蒸馏攻击与防御建模为教师与学生之间的极小极大博弈，提出了自适应评估规则和仅需前向传播的专家乘积防御方法，并揭示自适应学生模型在强评估下恢复的能力远超被动评估所显示的水平。 |
| [^400] | [Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG Foundation Models](https://arxiv.org/abs/2605.17562) | 本研究超越传统的干净数据准确率评估，从鲁棒性、可解释性和表达能力三个维度系统评估了六个EEG基础模型，发现没有任何单一模型能在所有失效模式下占优，且模型的归因总体集中于符合已知神经生理学的任务相关脑区。 |
| [^401] | [Road Maps as Free Geometric Priors: Weather-Invariant Drone Geo-Localization with GeoFuse](https://arxiv.org/abs/2605.14925) | 提出GeoFuse跨模态融合框架，利用免费可得且天生天气不变的道路地图几何先验与卫星图像融合，在几乎零额外成本下实现恶劣天气条件下鲁棒的无人机地理定位。 |
| [^402] | [Fast Adversarial Attacks with Gradient Prediction](https://arxiv.org/abs/2605.14868) | 该论文提出通过轻量级线性回归从前向传播隐藏状态预测输入梯度、从而消除反向传播的快速对抗攻击方法，在保持FGSM大部分攻击性能的同时实现了532%的吞吐量提升。 |
| [^403] | [FeatCal: Feature Calibration for Post-Merging Models](https://arxiv.org/abs/2605.13030) | 提出FeatCal方法，基于特征漂移理论（分解为上游传播与局部失配）分析模型合并后的性能差距，并利用小校准集以前向顺序逐层闭式校准合并模型权重，无需梯度下降或额外模块即可减少特征漂移、保留模型合并优势。 |
| [^404] | [Inference-Time Machine Unlearning via Gated Activation Redirection](https://arxiv.org/abs/2605.12765) | GUARD-IT 是一种无需训练、无需梯度、不改变模型权重的推理时机器遗忘方法，它将待遗忘内容存储为小型激活方向库，并在推理时通过依赖输入的门控激活重定向实现遗忘。 |
| [^405] | [Expected Batch Optimal Transport Plans and Consequences for Flow Matching](https://arxiv.org/abs/2605.12174) | 本文形式化了重复小批量OT所诱导的“期望批量OT计划”，证明其在大批量下的一致性并给出收敛速率，且表明该耦合诱导的速度场足够正则，能为流匹配定义唯一的流。 |
| [^406] | [Keeping Score: Adaptive, Tuning-Free Loss Weighting for Score-Augmented Neural Ratio Estimation](https://arxiv.org/abs/2605.12118) | 提出一种基于损失梯度的自适应、免调参算法来动态设置得分匹配损失的权重，以极小的额外开销提升得分增强神经比率估计代理模型的质量并大幅降低调参成本。 |
| [^407] | [Remember to Forget: Gated Adaptive Positional Encoding](https://arxiv.org/abs/2605.10414) | 提出GAPE（门控自适应位置编码），通过查询依赖和键依赖的双门控机制，在保持旋转位置编码几何结构的前提下，将内容感知偏置直接注入注意力logits，解决长序列外推时RoPE的分布外失效问题。 |
| [^408] | [Transfer Learning of Multiobjective Indirect Low-Thrust Trajectories Using Diffusion Models and Markov Chain Monte Carlo](https://arxiv.org/abs/2605.09125) | 该论文提出了一种将任务参数同伦变换与马尔可夫链蒙特卡洛采样相结合的迁移学习框架，以更高效地生成训练数据，从而利用扩散模型加速多目标间接低推力轨迹初步设计中的全局搜索。 |
| [^409] | [Don't Get Your Kroneckers in a Twist: Gaussian Processes on High-Dimensional Incomplete Grids](https://arxiv.org/abs/2605.08036) | 提出CUTS-GPR方法，通过将加性核与不完整网格结合以实现极快的核矩阵-向量乘积，使数值精确的高斯过程回归能够扩展到数百万数据点和数百个维度。 |
| [^410] | [The Minimax Rate of Perturbed Second-Order Calibration](https://arxiv.org/abs/2605.07808) | 本文提出通过向分类器分数添加 sech 噪声并进行低次多项式回归来估计二阶校准误差，达到 $O(\log^{3/2}n/\sqrt n)$ 的误差率，并通过匹配的 $\Omega(1/\sqrt{n})$ 下界证明了其在相差对数因子意义下的极小极大最优性。 |
| [^411] | [LiteGUI: Lightweight GUI Agents via Multi-Solution Guided Distillation and Dual-Level Reinforcement Learning](https://arxiv.org/abs/2605.07505) | LiteGUI通过“引导式在线策略蒸馏”与“多解双层GRPO强化学习”的两阶段后训练框架，使轻量级GUI智能体能够有效应对复杂任务的长时程特性和多条有效交互路径的挑战。 |
| [^412] | [Empirical Evidence for Simply Connected Decision Regions in Image Classifiers](https://arxiv.org/abs/2605.06380) | 本文通过自适应四边形网格填充实验首次提供了实证证据，表明预训练图像分类器中同标签决策区域是单连通的，即区域内的任意环路都可以被区域内曲面填充。 |
| [^413] | [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes](https://arxiv.org/abs/2605.06152) | 本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。 |
| [^414] | [State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning](https://arxiv.org/abs/2605.00206) | SST V2通过在每层引入FFN驱动的非线性递归，使潜在状态横向流经整个序列，实现连续潜在空间中的参数高效推理与深思，并采用两遍并行训练使其计算上可行。 |
| [^415] | [CMGL: Confidence-guided Multi-omics Graph Learning for Cancer Subtype Classification](https://arxiv.org/abs/2604.24201) | 提出了CMGL方法，利用证据深度学习为每个患者估计各模态的置信度以指导多组学融合，并在独立构建的共识一致性图上进行图分类，从而提升癌症亚型分类的可靠性。 |
| [^416] | [Learning to Emulate Chaos: Adversarial Optimal Transport Regularization](https://arxiv.org/abs/2604.21097) | 提出对抗性最优传输正则化方法，能够仅从单一含噪轨迹中联合学习高质量的摘要统计量与物理一致的混沌动力学模拟器。 |
| [^417] | [Provably Efficient Offline-to-Online Value Adaptation with General Function Approximation](https://arxiv.org/abs/2604.13966) | 该论文在一般函数逼近下研究离线到在线强化学习的价值自适应，通过极小化极大下界刻画了该问题的固有困难，并提出O2O-LSVI算法，在新的结构条件下实现了可证明优于纯在线强化学习的样本复杂度。 |
| [^418] | [Virtual Smart Metering in District Heating Networks via Heterogeneous Spatial-Temporal Graph Neural Networks](https://arxiv.org/abs/2604.10166) | 该论文提出利用异构时空图神经网络实现区域供热网络的虚拟智能计量，以在传感器稀疏分布且存在故障的条件下增强热力和水力状态的可观测性。 |
| [^419] | [ReCodeAgent: A Multi-agent Workflow for Language-Agnostic Translation and Validation of Large-Scale Repositories](https://arxiv.org/abs/2604.07341) | ReCodeAgent通过自主多智能体工作流，实现了仓库级代码翻译和验证的语言无关性，用户仅需指定源和目标编程语言即可自动处理整个仓库。 |
| [^420] | [Dynamic Free-Rider Detection in Cross-Silo Federated Learning via Simulated Attack Patterns](https://arxiv.org/abs/2604.04611) | 该论文提出通过模拟攻击模式（包括新提出的自适应WEF伪装攻击）来检测跨筒仓联邦学习中的动态搭便车者，能够识别出前期诚实参与、后期伪造参数以窃取全局模型的恶意客户端。 |
| [^421] | [Soft Tournament Equilibrium: Differentiable Set-Valued Inference for Non-Transitive Pairwise Comparisons](https://arxiv.org/abs/2604.04328) | 本文提出软锦标赛均衡（STE），一种可微分神经网络层，通过归一化log-sum-exp可达性和覆盖计算，从互反成对概率中平滑推断顶级循环集和未被覆盖集，并具备近似、扰动和边界恢复的理论保证。 |
| [^422] | [Transmission Neural Networks: Inhibitory and Excitatory Connections](https://arxiv.org/abs/2604.04246) | 本文将传输神经网络模型扩展至包含抑制性与兴奋性连接及神经递质群体，证明了考虑抑制作用的神经元发放概率刻画可等价表示为每个神经元具有2维连续状态的神经网络，并建立了神经递质数量趋于无穷时极限网络模型的稳定性与收缩性充分条件。 |
| [^423] | [Task-Centric Personalized Federated Fine-Tuning of Language Models](https://arxiv.org/abs/2604.00050) | 提出了FedRouter，一种基于聚类的个性化联邦学习方法，通过为每个任务而非每个客户端构建专门模型，解决了异构任务下的泛化能力不足和客户端内部任务干扰问题。 |
| [^424] | [Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation](https://arxiv.org/abs/2603.28744) | 该论文证明稀疏自编码器（SAE）因将稀疏推理摊销到固定编码器而存在系统性“摊销差距”，且其根源在于字典学习而非推理过程，这导致SAE在分布外组合偏移下无法像经典稀疏编码方法那样恢复概念空间的线性结构。 |
| [^425] | [Identification of Bivariate Causal Directionality Based on Anticipated Asymmetric Geometries](https://arxiv.org/abs/2603.26024) | 本文提出了两种基于条件分布的新方法——预期不对称几何（AAG）和单调性指数（MI），用于识别二元数值数据中的因果方向性。 |
| [^426] | [Decidable By Construction: Design-Time Verification for Truly Fearless Systems](https://arxiv.org/abs/2603.25414) | 提出 Composer 编译器设计，通过四个递进的验证层级在设计时追踪并发等待关系、验证多线程正确性并保留分布式边界契约，直接将 Clef 语言编译为原生 CPU/GPU/NPU/FPGA 代码，实现“构造即可判定”的可验证系统。 |
| [^427] | [Beyond Sample Copying: Structural Memorization in Diffusion Models](https://arxiv.org/abs/2603.13419) | 该论文揭示了扩散模型会逐步过拟合去噪训练目标，而模型误差恰好抑制了对训练点的精确记忆，使去噪流场保持平滑，正是这种相互作用诱导了扩散模型的泛化能力。 |
| [^428] | [PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](https://arxiv.org/abs/2603.13228) | 该论文提出PhysMoDPO框架，将全身控制器集成到直接偏好优化的训练流程中，利用基于物理和任务的奖励来优化扩散模型，使生成的运动轨迹既符合物理规律又忠实于文本指令。 |
| [^429] | [Altered Thoughts, Altered Actions: Reasoning Chain as Control Surface for a Vision-Language-Action Policy](https://arxiv.org/abs/2603.12717) | 该论文通过确定性实体交换实验系统测量了编辑推理链对视觉-语言-动作策略运动的修复与破坏作用，发现其影响集中在仅由语言决定目标的任务上，从而确立推理链可作为策略的有效控制界面。 |
| [^430] | [Jailbreak Scaling Laws for Large Language Models: Polynomial-Exponential Crossover](https://arxiv.org/abs/2603.11331) | 本文发现对抗性提示注入攻击能将大语言模型的攻击成功率从多项式增长放大为指数增长，并提出了基于自旋玻璃系统的理论生成模型来解释这两种缩放定律背后的最小统计机制。 |
| [^431] | [Agentic Critical Training](https://arxiv.org/abs/2603.08706) | 该论文提出智能体批判性训练，利用可验证奖励的强化学习让模型学会直接区分专家动作与看似合理的错误动作，作为模仿学习前的高效热身方法，无需参考推理且支持数据跨模型复用。 |
| [^432] | [\$OneMillion-Bench: How Far are Language Agents from Human Experts?](https://arxiv.org/abs/2603.07980) | 提出了包含 400 个专家设计的跨领域专业任务的 \$OneMillion-Bench 基准，通过基于评分规则的多维度评估，衡量语言智能体在法律、金融、医疗等经济关键场景中与人类专家的差距。 |
| [^433] | [Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](https://arxiv.org/abs/2603.07233) | 提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。 |
| [^434] | [3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs](https://arxiv.org/abs/2603.01376) | 该论文提出了3BASiL-TM，一种基于新颖三块ADMM算法的高效一次性后训练框架，通过带收敛保证的逐层重构误差最小化与跨Transformer层的联合精炼，实现大语言模型的稀疏加低秩压缩并显著缓解性能下降。 |
| [^435] | [V-ECE: Estimating General Expected Calibration Errors](https://arxiv.org/abs/2602.24230) | 该论文提出V-ECE方法，利用依赖预测的适当评分突破了以往方法仅能估计Bregman散度类校准误差的限制，实现了对包括$L_1$距离在内的一般凸散度（如$L_p$距离）校准误差在二分类和多分类场景下的可靠估计。 |
| [^436] | [PaReGTA: A Temporally Aware LLM-Based Patient Representation Framework for EHR Analytics](https://arxiv.org/abs/2602.19661) | PaReGTA是一种基于大语言模型的时间感知患者表示框架，通过将纵向EHR事件转换为带显式时间线索的文本、轻量级对比微调学习就诊嵌入、以及混合时间池化聚合，生成可直接用于常规下游机器学习模型的固定维度患者表示。 |
| [^437] | [Uncertainty Quantification in Federated Granger Causality Learning](https://arxiv.org/abs/2602.13004) | 本文针对客户端特征异构的联邦格兰杰因果学习场景，刻画了跨客户端依赖估计过程中的不确定性传播，并利用边特定的方差有效区分真实的跨客户端依赖关系与虚假的估计边。 |
| [^438] | [Conditional Flow Matching for Visually-Guided Acoustic Highlighting](https://arxiv.org/abs/2602.03762) | 本文提出一种条件流匹配生成框架，通过引入展开损失惩罚最终步骤漂移，有效解决了视觉引导音频增强中判别模型难以处理音频重混模糊性的问题。 |
| [^439] | [Synthetic Time Series Generation via Complex Networks](https://arxiv.org/abs/2601.22879) | 本文通过结合统计特征、网络拓扑特性及下游任务性能的全面实证研究，首次系统评估了逆分位数图框架作为通用合成时间序列生成器在保真度和实用性方面的潜力。 |
| [^440] | [Mechanistic Evidence for Spectral Structures in Prior-Data Fitted Networks](https://arxiv.org/abs/2601.21731) | 本文通过线性探针与激活/子空间修补等机制可解释性方法，首次证明先验数据拟合网络内部以低维结构表征上下文的谱内容（即决定平稳核的量），且该谱信息可被读取为显式核。 |
| [^441] | [Solving the Offline and Online Min-Max Problem of Non-smooth Submodular-Concave Functions: A Zeroth-Order Approach](https://arxiv.org/abs/2601.21243) | 本文提出一种结合Lovász扩展次梯度与高斯平滑的零阶方法，解决了非光滑次模-凹函数的离线与在线极小极大问题，证明了离线情形收敛到ε-鞍点，并在在线情形达到O(√(N(1+P̄_N)))的对偶间隙。 |
| [^442] | [Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback](https://arxiv.org/abs/2601.18751) | 提出TriTrust-PBRL框架，通过联合学习共享奖励模型与专家信任参数（自动演化为正、零或负值），能够信任、忽略或反转多专家偏好反馈，从而有效抵御对抗性标注者的干扰。 |
| [^443] | [Explanation Multiplicity in SHAP: Characterization and Assessment](https://arxiv.org/abs/2601.12654) | 该论文揭示了SHAP解释在同一模型和输入的重复运行之间也会出现显著分歧（即“解释多样性”）这一现象，并提出了一套结合双种子协议、多层次度量指标和随机零模型的评估方法来对其进行系统刻画。 |
| [^444] | [Predictive Inorganic Synthesis based on Machine Learning using Small Data sets: a case study of Hydrodynamic Diameter-controlled Cu Nanoparticles](https://arxiv.org/abs/2512.16545) | 本研究证明仅需25次合成的小数据集，结合拉丁超立方采样与集成回归模型，即可有效预测铜纳米颗粒的水动力直径，为小数据条件下机器学习驱动的可控纳米材料合成提供了范例。 |
| [^445] | [One Permutation Is All You Need: Fast, Deterministic Feature Importance and Model Stress-Testing](https://arxiv.org/abs/2512.13892) | 用单次最大-最小秩最优的确定性置换替代多次随机置换，可将特征重要性估计的计算复杂度从 O(B·n·p) 降至 O(n·p)，消除估计方差并保持或提升估计精度，并可扩展用于模型压力测试。 |
| [^446] | [Self-sufficient Independent Component Analysis for Demixing Flows](https://arxiv.org/abs/2512.00665) | 提出一种无先验、无似然的自充分独立成分分析方法，通过最小化条件KL散度并顺序学习解混流模型来从数据中学习解耦信号，同时完全避免了不稳定的对抗训练。 |
| [^447] | [Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations](https://arxiv.org/abs/2511.04000) | 本文提出通过合成采样近最优决策树来生成大规模预训练数据的高效可扩展方法，使MetaTree transformer在决策树元学习上达到与真实数据或昂贵最优树预训练相当的性能，同时大幅降低计算成本。 |
| [^448] | [LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling](https://arxiv.org/abs/2510.18239) | LIME架构通过低秩链接嵌入解耦用户与候选交互、实现注意力权重预计算，并采用线性XOR注意力机制，从根本上降低了推荐系统推理时对候选集大小和用户序列长度的计算成本依赖。 |
| [^449] | [Local Timescale Gates for Timescale-Robust Continual Spiking Neural Networks](https://arxiv.org/abs/2510.12843) | 提出局部时间尺度门控（LT-Gate）神经元模型，通过双时间常数动力学与自适应门控机制，使单个神经元在响应快速信号的同时保留慢速上下文信息，并辅以方差追踪正则化稳定放电活动，显著提升了脉冲神经网络在持续学习任务中的准确率与记忆保持能力。 |
| [^450] | [Fractional Heat Kernel for Semi-Supervised Graph Learning with Small Training Sample Size](https://arxiv.org/abs/2510.04440) | 该论文提出一种源驱动的分数阶热核半监督图学习框架，通过持续标签源防止解塌缩至拉普拉斯零空间从而缓解过平滑，在每类仅一个训练标签的极小样本条件下即可达到超过96%的准确率。 |
| [^451] | [CAF\'E: Causal Black-Box Testing of Machine Unlearning](https://arxiv.org/abs/2509.16525) | 提出CAF'E框架，将机器遗忘测试构建为基于规范的测试，仅通过黑盒模型的预测输出对特征进行因果干预并传播到下游特征，从而有效检测特征影响在模型中的残留。 |
| [^452] | [AdaSwitch: An Adaptive Switching Meta-Algorithm for Learning-Augmented Bounded-Influence Problems](https://arxiv.org/abs/2509.02302) | 该论文提出了有界影响框架与元算法AdaSwitch，它能在离线与在线预测器之间自适应切换，在预测准确时性能逼近离线最优、在预测任意不准时仍保持接近在线算法的最坏情况保证，并成功应用于在线交货期报价、k服务器、缓存和在线可重用资源分配等多种问题。 |
| [^453] | [The Sample Complexity of Membership Inference and Privacy Auditing](https://arxiv.org/abs/2508.19458) | 本文在高斯均值估计的基础设定下，研究了成员推断攻击的样本复杂度，即确定了成功实施攻击和隐私审计所需的最少参考样本数量。 |
| [^454] | [Inverse-LLaVA: Rethinking Multimodal Alignment via Text-to-Vision Mapping](https://arxiv.org/abs/2508.12466) | 该论文提出Inverse-LLaVA，通过在解码器注意力中将语言状态映射到视觉特征维度（而非传统的将图像特征投影到语言空间），在冻结主干、无需独立对齐阶段的单阶段训练下实现多模态融合，性能接近两阶段的LLaVA-1.5。 |
| [^455] | [An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems](https://arxiv.org/abs/2508.08833) | 提出 GAP 方法，通过表面重命名与核心重写两种数学等价变换自动批量生成现有数学题的等价变体，以评估大语言模型数学推理的鲁棒性并诊断其失败环节。 |
| [^456] | [BrainATCL: Adaptive Temporal Brain Connectivity Learning for Functional Link Prediction and Age Estimation](https://arxiv.org/abs/2508.07106) | 提出了BrainATCL，一个无监督、非参数化的自适应时序脑连接学习框架，能够捕捉动态fMRI数据中的长程时间依赖性，用于功能连接预测和年龄估计。 |
| [^457] | [Brain foundation model-guided source-selective domain adaptation for cross-subject EEG decoding](https://arxiv.org/abs/2507.21037) | 提出了一种由预训练脑基础模型引导的多源域自适应框架BFM-MSDA，利用脑基础模型表示估计源域与目标域的兼容性以实现源域选择性自适应，从而解决跨被试运动想象脑电解码中的负迁移与分布差异问题。 |
| [^458] | [Conformal Data Contamination Tests for In-distribution Data Acquisition](https://arxiv.org/abs/2507.13835) | 本文提出了一种无分布假设的共形数据污染检验框架，仅需检查少量数据即可识别出对模型个性化最有价值的外部数据代理，从而在数据获取前提供质量保证。 |
| [^459] | [Heterogeneous-Modal Unsupervised Domain Adaptation via Latent Space Bridging](https://arxiv.org/abs/2506.15971) | 本文提出了一种新的HMUDA设置及潜在空间桥接（LSB）方法，利用包含两种模态成对观测数据的无标记桥接域，实现有标记源域与完全无标记的异构模态目标域（如2D图像与3D点云）之间的知识迁移。 |
| [^460] | [Delphos: A reinforcement learning framework for assisting discrete choice model specification](https://arxiv.org/abs/2506.06410) | Delphos是一个深度强化学习框架，它将离散选择模型设定形式化为序贯决策问题，通过智能体自动选择建模动作生成候选模型设定，为建模者提供自动化、数据驱动的建议，从而减少开发与改进效用函数所需的工作量。 |
| [^461] | [Towards Reasonable Concept Bottleneck Models](https://arxiv.org/abs/2506.05014) | 提出概念推理模型（CREAM），一种可在架构层面显式编码概念间关系与概念-任务关系、并能借助正则化旁路通道处理不完整概念集的概念瓶颈模型新框架，同时引入了与C→Y无关的可解释性评估指标。 |
| [^462] | [Equilibrium Distribution for t-Distributed Stochastic Neighbor Embedding with Generalized Kernels](https://arxiv.org/abs/2505.24311) | 该论文为广义输入输出核下的t-SNE大样本变分问题建立了严格的数学理论，证明了尺度参数的存在唯一性、解的存在性与一致有界性，以及离散最优解收敛到满足平衡方程的紧支撑平衡分布。 |
| [^463] | [InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models](https://arxiv.org/abs/2505.13878) | InfiFPO通过在DPO中用序列层面综合多源概率的融合源模型替换参考模型，实现了无需复杂词表对齐且保留概率信息的大语言模型隐式融合偏好优化方法。 |
| [^464] | [A Survey on Archetypal Analysis](https://arxiv.org/abs/2504.12392) | 这是首篇关于原型分析（AA）的综述，系统介绍了其方法论、面临的非凸优化挑战、跨科学领域的广泛应用以及数据建模的最佳实践。 |
| [^465] | [C-LoRA: Continual Low-Rank Adaptation for Pre-trained Visual Models](https://arxiv.org/abs/2502.17920) | C-LoRA通过可学习路由矩阵使单个共享的LoRA适配器能够持续学习序列任务而不会发生灾难性遗忘，无需推理时的模块选择或融合，同时避免了参数无界增长和推理复杂度增加。 |
| [^466] | [Networks with Finite VC Dimension: Pro and Contra](https://arxiv.org/abs/2502.02679) | 该论文证明有限的VC维虽有利于经验误差的一致收敛，却可能不利于函数逼近，并基于高维几何的测度集中性质证明，此类网络在处理大规模数据集时逼近误差与经验误差均几乎呈确定性行为。 |
| [^467] | [BEAT: Balanced Frequency Adaptive Tuning for Long-Term Time-Series Forecasting](https://arxiv.org/abs/2501.19065) | 提出BEAT框架，通过频率专属监测器在统一归一化空间中监测各频率分量的系数预测误差，并据此自适应地调节各频率网络的梯度，从而平衡长期时间序列预测中不同频率分量的训练侧重。 |
| [^468] | [Foundations of Large Language Models](https://arxiv.org/abs/2501.09223) | 本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。 |
| [^469] | [Graphons of Line Graphs](https://arxiv.org/abs/2409.01656) | 本文提出一种通过将稀疏图映射到其线图并利用“平方度性质”使稀疏图产生稠密线图的方法，从而可以应用稠密图极限理论来分析稀疏图，并实证证明能够区分原本都收敛到零图子的不同数量的星形图。 |
| [^470] | [Class Machine Unlearning for Complex Data via Concepts Inference and Data Poisoning](https://arxiv.org/abs/2405.15662) | 该论文针对复杂数据的类别机器遗忘问题，提出通过推断连接遗忘目标与模型输出的语义概念，并结合数据投毒技术来精准消除目标信息的影响，从而避免知识残留并保护应保留的内容。 |
| [^471] | [Policy Learning with a Language Bottleneck](https://arxiv.org/abs/2405.04118) | 该论文提出PLLB框架，让AI智能体在语言模型引导的“规则生成”与规则引导的“策略更新”之间交替进行，通过语言瓶颈捕捉行为背后的高层策略，从而学习到更可解释、更可泛化的行为。 |

# 详细

[^1]: CSF：面向运动生成器的上下文安全过滤

    CSF: Contextual Safety Filtering for Motion Generators

    [https://arxiv.org/abs/2610.12467](https://arxiv.org/abs/2610.12467)

    提出免训练的上下文安全过滤框架CSF，将自然语言安全规则通过安全/不安全参考轨迹转化为CBF-QP约束，使文本条件运动生成器在场景触发的不安全情况下危险事件率降低高达90%，同时保留88-100%的良性运动。

    

    文本条件运动生成器能够产生可跟踪的全身运动，但它们缺乏对场景相关安全性的认知：同一个动作可能指向物体，也可能指向人。现有的安全防护措施要么检查提示词，要么需要带标签的运动数据，要么强制执行几何约束；因此，它们没有直接考虑场景上下文如何改变运动的意义。我们提出了上下文安全过滤（CSF），这是一种免训练的过滤器，它将自然语言安全规则与生成器产生的安全和不安全参考轨迹相联系。对于每条激活的规则，安全和不安全参考轨迹定义了一个仿射安全值，由安全参考跟踪CBF-QP（控制屏障函数二次规划）强制执行。在四种不同架构的预训练生成器上，CSF在所有显式和场景触发的不安全情况下均能激活预期的规则，将危险事件率降低高达90%，同时保留88-100%的良性运动。

    arXiv:2610.12467v1 Announce Type: cross  Abstract: Text-conditioned motion generators produce trackable whole-body motion, but they have no notion of scene-dependent safety: the same action may target an object or a person. Existing safeguards either inspect the prompt, require labeled motion data, or enforce geometric constraints; therefore, they do not directly account for how scene context changes a motion's meaning. We introduce contextual safety filtering (CSF), a training-free filter that grounds natural-language safety rules in safe and unsafe reference trajectories produced by the generator. For each active rule, safe and unsafe reference trajectories define an affine safety value that a safe reference tracking CBF-QP enforces. Across four pretrained generators with different architectures, CSF activates the intended rules in all explicit and scene-triggered unsafe cases and reduces the danger-event rate by up to 90%, while preserving 88-100% of benign motions. We demonstrate t
    
[^2]: 均衡的数据配比：解决机器人控制中超大规模强化学习的探索瓶颈

    A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control

    [https://arxiv.org/abs/2610.12465](https://arxiv.org/abs/2610.12465)

    该论文指出对仿真器重置进行均匀采样会将大量学习经验浪费在已掌握或无法尝试的任务配置上，从而提出均衡的数据采样策略，以解决超大规模并行强化学习在机器人控制中的探索瓶颈问题。

    

    通用机器人需要执行从敏捷运动到灵巧操作的广泛任务。虽然仿真到现实强化学习已被证明是实现这一目标的有效工具，但当前的RL流程依赖于工程量繁重的、针对每个任务的结构先验，例如塑形奖励和示范。最近的研究表明，多样化的仿真器重置与大规模并行仿真相结合，可以在若干操作问题上大大减轻这种工程负担。然而，我们发现将这一范式简单地扩展到更精确或更动态的问题上仍然并非易事。虽然仿真器重置有助于探索，但对这一分布进行均匀采样会导致越来越多的学习经验浪费在策略已经掌握或尚无法尝试的任务配置上。这使得扩展并行环境为RL带来的预期收益难以显现，因为大部分学习时间（摘要在此处截断）

    arXiv:2610.12465v1 Announce Type: cross  Abstract: General-purpose robots must perform a wide range of tasks from agile locomotion to dexterous manipulation. While sim-to-real reinforcement learning (RL) has proven to be a useful tool for this goal, current RL pipelines depend on engineering-heavy, per-task structural priors such as shaped rewards and demonstrations. Recent work has shown that diverse simulator resets, combined with massively parallel simulation, can alleviate much of this engineering burden on several manipulation problems. However, we find that naively scaling this paradigm to more precise or dynamic problems remains non-trivial. While simulator resets can help with exploration, uniformly sampling over this distribution wastes a growing fraction of learning experience on task configurations the policy has already mastered or cannot yet attempt. This makes it challenging to see the expected benefits of scaling parallel environments for RL, since much of the learning s
    
[^3]: Bi-FORK：高维分岔系统的生成式建模

    Bi-FORK: Generative Modeling of High-Dimensional Bifurcating Systems

    [https://arxiv.org/abs/2610.12449](https://arxiv.org/abs/2610.12449)

    Bi-FORK是一个生成式框架，通过潜空间流匹配和排斥引导采样，学习高维分岔系统中一对多的解映射，能够在屈曲、超材料和相分离等物理问题中高效恢复多模态的完整解分支。

    

    分岔现象在物理系统中无处不在，从结构屈曲到流体和气候动力学，但其在深度学习中仍然很大程度上未被探索。在对称性破缺分岔处，单个输入对应多个同样有效的解，这违背了大多数学习型物理代理模型所依赖的一对一假设。我们提出了Bi-FORK，一个用于学习高维系统中一对多解映射的生成式框架。Bi-FORK通过潜空间流匹配生成完整的轨迹，保持空间与时间上的连贯性，并利用排斥引导采样在单次摊销推理中恢复出不同的解分支。我们在屈曲梁、力学超材料和Allen-Cahn相分离问题上对Bi-FORK进行了评估，涵盖了连续、离散和场值型分岔，离散化规模高达260,000个点。Bi-FORK在恢复多模态解结构的同时，实现了数个数量级的（加速/扩展）……

    arXiv:2610.12449v1 Announce Type: cross  Abstract: Bifurcations are ubiquitous in physical systems, from structural buckling to fluid and climate dynamics, yet they remain largely unexplored in deep learning. At a symmetry-breaking bifurcation, a single input admits multiple equally valid solutions, violating the one-to-one assumption underlying most learned physical surrogates. We introduce Bi-FORK, a generative framework for learning these one-to-many solution maps in high-dimensional systems. Bi-FORK generates complete trajectories through latent flow matching, preserving space and time coherence, and uses repulsion-guided sampling to recover distinct solution branches in a single amortized pass. We evaluate Bi-FORK on buckling beams, mechanical metamaterials, and Allen-Cahn phase separation, spanning continuous, discrete, and field-valued bifurcations with discretizations up to 260,000 points. Bi-FORK recovers the multimodal solution structure while scaling several orders of magnit
    
[^4]: 一个块，多重深度：具有深度编程专家的循环视觉Transformer

    One Block, Multiple Depths: Recurrent Vision Transformers with Depth-Programmed Experts

    [https://arxiv.org/abs/2610.12448](https://arxiv.org/abs/2610.12448)

    提出reViT，通过循环复用单个Transformer块，并将各深度的FFN表示为由连续深度坐标编程的共享专家凸组合，在相当的推理计算量下以约70%更少的参数达到与全深度视觉编码器相当的精度。

    

    在这项工作中，我们证明单个Transformer块经过循环应用，即可在相当的推理FLOPs下达到与全深度视觉编码器相当的精度，且无需中间特征蒸馏。reViT通过将每个循环深度处的FFN表示为一个小型共享专家库的凸组合来恢复特定深度的变换。一个连续的归一化深度坐标对该混合进行编程，定义了一条在FFN参数空间中可重采样的轨迹。我们在两种机制下评估该设计：有监督的ImageNet-1k训练以及从DINOv2教师模型进行的蒸馏。在两种机制下，对照实验表明，在匹配单FFN预算的条件下，权重空间合并是所测试的MoE方案中最强的一类，优于令牌分发和输出混合等替代方案。从头训练的reViT-B/16以约70%更少的存储参数达到了DeiT III的精度。仅使用教师输出进行蒸馏的8专家模型……（原文摘要在此处截断）

    arXiv:2610.12448v1 Announce Type: cross  Abstract: In this work, we show that a single Transformer block, applied recurrently, can match the accuracy of a full-depth vision encoder at comparable inference FLOPs without intermediate feature distillation. reViT restores depth-specific transformations by representing the FFN at each recurrent depth as a convex combination of a small shared expert bank. A continuous normalized-depth coordinate programs this mixture, defining a resampleable trajectory through FFN parameter space. We evaluate this design in two regimes: supervised ImageNet-1k training and distillation from a DINOv2 teacher. Across both regimes, controlled adaptations identify weight-space merging as the strongest tested MoE family at a matching one-FFN budget, ahead of the token-dispatch and output-mixture alternatives. Trained from scratch, reViT-B/16 attains DeiT III accuracy with about 70\% fewer stored parameters. An 8-experts model distilled using only the teacher's out
    
[^5]: 当场抓获：探针有效检测破坏行为并捕捉未言明的欺骗

    Caught in the Act: Probes Effectively Detect Sabotage and Catch Unverbalized Deception

    [https://arxiv.org/abs/2610.12445](https://arxiv.org/abs/2610.12445)

    该研究通过构建迄今最大的欺骗数据集并设计可跨层、跨标记聚合信息的新型探针架构，使白盒探针在 SHADE-Arena 上以 98.8% 的 AUC 超越前沿文本监控基线，甚至能检测出仅凭上下文无法察觉的“内省式欺骗”。

    

    近期发生的一些事件凸显了监控大语言模型（LLM）智能体的挑战，以及模型欺骗人类的危险。我们展示了基于探针的白盒欺骗检测可以扩展到前沿监控场景，方法包括收集迄今为止最大的欺骗数据集用于训练探针，并引入一种新颖的探针架构，该架构能够跨多个层和标记（token）聚合信息。我们的探针在 SHADE-Arena 上达到了 98.8% 的 AUC，超越了 Opus 5.5 的文本监控基线，并且随着底层模型规模的扩大，其检测效果也随之提升。为了将我们的探针推向极限，我们在几种仅凭上下文无法判断是否存在欺骗的案例上对其进行了测试。在这些我们称为“内省式欺骗”的案例中，真实情况只能通过仔细的引导性询问或对模型训练数据的透彻了解才能确定。在其中一项评估中，我们展示了探针能够区分包含……的对话记录（原文摘要在此处截断）。

    arXiv:2610.12445v1 Announce Type: cross  Abstract: Recent incidents have highlighted the challenge of monitoring LLM agents and the danger of models deceiving people. We show that white-box deception detection via probes can be scaled up to frontier monitoring settings by collecting the largest deception dataset to date for training probes and introducing a novel probe architecture which can aggregate information across many layers and tokens. Our probes achieve 98.8% AUC in SHADE-Arena, surpassing an Opus 5.5 text-monitoring baseline, and show improved efficacy as the underlying model is scaled up. To push our probes to their limit, we test them on several cases where deception cannot be determined from the context alone. In these cases, which we refer to as introspective deception, the ground truth can only be determined through careful elicitation or thorough knowledge of a model's training data. In one such evaluation, we show that probes can distinguish transcripts containing a mo
    
[^6]: 前条件子空间中的舍入：重新设计4比特AdamW优化器状态量化

    Rounding in Preconditioner Space: Redesigning 4-bit AdamW Optimizer-State Quantization

    [https://arxiv.org/abs/2610.12444](https://arxiv.org/abs/2610.12444)

    该论文从“舍入空间”的新视角重新设计了AdamW的4比特优化器状态量化，证明了状态空间舍入无法保证前条件子误差足够小，并据此提出在二阶矩码本中保留零、于前条件子空间中计算随机舍入概率的ZIP-SR方法。

    

    对AdamW优化器状态进行量化可以减少持久存储开销，但量化误差会通过矩递推传播，并扰动后续的自适应更新。我们从“舍入空间”的角度重新设计了AdamW的4比特优化器状态量化，舍入空间即量化器在相邻重建级别之间进行选择的坐标系。针对二阶矩，对零附近量化单元的局部分析表明，较小的平均状态误差并不一定意味着下一步具有较小的平均前条件子误差。一个一维二次函数构造进一步表明，在状态空间舍入和前条件子空间舍入下，优化动态存在本质上的差异。这些结果促成了零包含前条件子空间随机舍入方法（Zero-Inclusive Preconditioner-space Stochastic Rounding，ZIP-SR），该方法在二阶矩码本中保留零，并在前条件子空间中计算随机舍入概率。作为一种互补途径，零排除……

    arXiv:2610.12444v1 Announce Type: new  Abstract: Quantizing AdamW's optimizer states reduces persistent storage, but quantization errors propagate through the moment recurrences and perturb subsequent adaptive updates. We redesign 4-bit optimizer-state quantization for AdamW from the perspective of \emph{rounding space}: the coordinate in which a quantizer chooses between adjacent reconstruction levels. For the second moment, a local analysis of the quantization cell adjacent to zero shows that small mean state error need not imply small mean preconditioner error at the next step. A one-dimensional quadratic construction further shows qualitatively different optimization dynamics under state-space and preconditioner-space rounding. These results motivate Zero-Inclusive Preconditioner-space Stochastic Rounding (\textbf{ZIP-SR}), which retains zero in the second-moment codebook and computes stochastic-rounding probabilities in preconditioner space. As a complementary route, Zero-Excludin
    
[^7]: 基于Stein位移场的密度比估计

    Density Ratio Estimation with Stein Displacement Fields

    [https://arxiv.org/abs/2610.12437](https://arxiv.org/abs/2610.12437)

    该论文提出通过Stein位移场参数化密度比（将对数比建模为负的基础分布Stein算子作用于位移场），用单个凸优化问题统一了分布偏移的统计与动力学描述，并据此发展出无需重训即可修正预训练采样器的push-forward算法和拉近数据分布的pull-back算法。

    

    密度比从概率质量的角度量化分布偏移，而位移场则从动力学的角度描述一个分布是如何被输运到另一个分布上的。尽管两者能提供互补的见解，但它们通常被分开估计，且将其中一种转换为另一种需要后处理。在本文中，我们通过一个作用于基础分布的位移场来参数化目标分布与基础分布之间的密度比：对数比被建模为负的基础分布Stein算子作用于该位移场的结果，仅相差一个归一化常数。这使得我们能够通过单个凸优化问题同时获得分布偏移的统计学与动力学描述。迭代这一“估计—移动”步骤可得到两种推断算法：push-forward（前推）移动模型，并在无需重新训练的情况下修正预训练的采样器；而pull-back（回拉）则将数据移动得更接近基础分布。

    arXiv:2610.12437v1 Announce Type: cross  Abstract: Density ratios quantify distribution shift from a probability-mass point of view, whereas displacement fields describe, from a dynamical point of view, how one distribution is transported onto another. Although both offer complementary insights, they are usually estimated separately, and converting one into the other requires post-processing. In this paper, we estimate the density ratio between a target and a base distribution by parametrizing it through a displacement field acting on the base: the log-ratio is modeled as minus the Stein operator of the base applied to the field, up to a normalizing constant. This gives both statistical and dynamical descriptions of the distribution shift through a single convex optimization problem. Iterating this estimate-and-move step gives two inference algorithms: push-forward moves the model and corrects a pretrained sampler without retraining it, whereas pull-back moves the data closer to the ba
    
[^8]: VioLA：从人类数据学习人形机器人通用控制策略

    VioLA: Learning Generalist Humanoid Control Policies from Human Data

    [https://arxiv.org/abs/2610.12435](https://arxiv.org/abs/2610.12435)

    VioLA通过让人形机器人通用策略预测身体和手部的运动潜在表示而非关节指令，并借助将人类与机器人运动映射到同一潜在空间的编码器，从而能够直接利用海量人类示教数据，实现无需针对每个任务微调即可开箱即用跟随新指令的通用全身控制。

    

    让人形机器人用全身跟随指令面临两大障碍。其一是动作空间庞大且高度耦合：腿、手臂和手指必须在机器人保持平衡的同时协同运动，这使得关节级的动作难以学习。其二是人形机器人示教数据稀缺，因此当前的人形机器人通用策略无法开箱即用地跟随新指令，需要在部署前针对每个任务在遥操作示教数据上进行微调。人类示教数据数量要庞大得多，但人的动作并不能直接当作机器人指令。我们通过改变通用策略所预测的内容来消除这两个障碍。我们提出VioLA，这是一种预测身体和手部运动潜在表示而非关节指令的通用人形机器人策略。预训练好的身体控制器和手部控制器在机器人上执行这些潜在表示。与之对应的运动编码器将人类和机器人的运动映射到相同的潜在空间中。一段人类动作记录即可……（原文摘要在此处被截断）

    arXiv:2610.12435v1 Announce Type: cross  Abstract: Teaching a humanoid to follow instructions with its whole body runs into two obstacles. Its action space is large and tightly coupled: legs, arms, and fingers must move together while the robot keeps its balance, which makes joint-level actions hard to learn. And humanoid demonstrations are scarce, so current humanoid generalist policies do not follow new instructions out of the box and are fine-tuned on teleoperated demonstrations of each task before deployment. Human demonstrations exist in far larger numbers, but a person's motion is not a robot command. We remove both obstacles by changing what the generalist policy predicts. We introduce VioLA, a generalist humanoid policy that predicts body and hand motion latents instead of joint commands. A pretrained body- and hand-controller execute these latents on the robot. Their corresponding motion encoders map human and robot motion into the same latent spaces. A human recording is ther
    
[^9]: FAITH：面向高维系统的可行性感知安全过滤强化学习

    FAITH: Feasibility-Aware Safety-Filtered RL for High-Dimensional Systems

    [https://arxiv.org/abs/2610.12432](https://arxiv.org/abs/2610.12432)

    FAITH提出了一种可行性感知的无模型安全过滤强化学习框架，通过前馈网络近似最优状态-动作安全价值函数并摊销最小干预过滤，使任务策略在更新中无需竞争性安全项即可优化长时程回报，同时能妥善处理不存在安全动作的情形。

    

    安全强化学习通常将安全性与任务性能置于同一个策略目标之中，这会引入相互竞争的更新。安全过滤器在动作执行层面将二者分离，但经典设计需要解析形式的安全函数和动力学模型，且标准的最小干预过滤器对长时程任务回报是短视的，因为它们仅最小化瞬时动作偏差。此外，当不存在安全动作时，硬投影也没有定义。我们提出了FAITH，一个可行性感知的无模型框架，它利用前馈网络近似最优的状态-动作安全价值函数，并以摊销方式实现最小干预过滤。任务策略通过过滤后的动力学来优化任务回报，从而在任务策略更新中不包含竞争性安全项的情况下恢复可行的约束问题。当没有动作满足所学的安全条件时，同一过滤器会以最小……（原文摘要在此处被截断）

    arXiv:2610.12432v1 Announce Type: cross  Abstract: Safe reinforcement learning commonly places safety and task performance in the same policy objective, where they can introduce competing updates. Safety filters separate them at action execution, but classical designs require an analytic safety function and dynamics model, and standard minimal-intervention filters are myopic to long-horizon task return because they minimize only instantaneous action deviation. Hard projections are also undefined when no safe action exists. We present FAITH, a feasibility-aware, model-free framework that approximates the optimal state-action safety value and amortizes minimal-intervention filtering with a feedforward network. The task policy optimizes the task return through the filtered dynamics, which recovers the feasible constrained problem without a competing safety term in the task-policy update. When no action satisfies the learned safety condition, the same filter approaches the action with mini
    
[^10]: 面向自适应生长量子分类器中电路深度与训练数据规模的联合优化

    Toward Joint Optimization of Circuit Depth and Training Data Size in Adaptively Grown Quantum Classifiers

    [https://arxiv.org/abs/2610.12428](https://arxiv.org/abs/2610.12428)

    本研究通过在多规模MNIST数据集上忠实复现Q-FLAIR的自适应电路生长机制，探究量子电路深度与训练数据规模之间是否存在可预测的缩放规律。

    

    构建量子模型涉及一个权衡：电路应该有多复杂，以及需要多少训练数据。Caro等人证明，可训练门越少的模型需要越少的训练数据就能良好泛化。Q-FLAIR表明，量子特征映射电路可以逐门生长，一旦进一步生长不再改善训练损失即可停止。我们探究这两个结果是否能结合成一个可预测的缩放规律：随着训练数据增长，Q-FLAIR自身的停止规则会选择更大还是更小的电路？由此产生的泛化行为是否遵循Caro等人的界？我们忠实地重新实现了Q-FLAIR的生长机制，包括其解析重构和精确的停止规则，并在全分辨率（784像素）的MNIST 3-vs-5分类任务上，以N=2000到10000五个不同训练集规模运行该机制。随后我们对每个生成的电路进行微调，以便测量Caro等人定义的活跃门数量K。我们发现没有可预测的……（摘要在此处截断）

    arXiv:2610.12428v1 Announce Type: cross  Abstract: Building a quantum model involves a tradeoff: how complex the circuit should be, and how much training data it needs. Caro et al. show that models with fewer trainable gates need less training data to generalize well. Q-FLAIR shows that a quantum feature-map circuit can be grown gate-by-gate, stopping once further growth stops improving the training loss. We ask whether these two results combine into a predictable scaling law. Does Q-FLAIR's own stopping rule pick larger or smaller circuits as training data grows? Does the resulting generalization behavior track Caro et al.'s bound?   We reimplement Q-FLAIR's growth mechanism faithfully, including its analytic reconstruction and exact stopping rule. We run it on full-resolution (784-pixel) MNIST 3-vs-5 classification, at five training-set sizes from N = 2000 to 10000. We then fine-tune each resulting circuit, so we can measure Caro et al.'s notion of active gates, K.   We find no predi
    
[^11]: 超越时空先验：一种可泛化的稠密对应匹配方法

    Beyond Spatio-Temporal Priors: A Generalizable Approach for Dense Correspondence Matching

    [https://arxiv.org/abs/2610.12421](https://arxiv.org/abs/2610.12421)

    FreeMatching 框架通过结合生成式与语义基础表示、异构监督及教师引导的迭代细化，突破了传统时空先验的限制，实现了图像编辑与参考引导生成中保持视觉同一性的可泛化稠密对应匹配。

    

    稠密对应匹配长期以来一直受到简化的时空先验（如平滑运动和刚性几何）的限制。虽然这些假设在经典任务中行之有效，但在图像编辑和参考引导生成（IEG）场景中却会失效，因为这些变换可以在保持视觉同一性的同时打破物理连续性。为了在这类变换下建立保持身份的对应关系，我们提出了 FreeMatching，一个可泛化的框架，它将生成式与语义基础表示相结合，并利用来自经典数据集、跟踪视频和合成场景的异构监督。教师引导的迭代细化进一步提升了对应对，且无需稠密对应标注。实验表明，单个 FreeMatching 模型在具有挑战性的 IEG 图像对上显著提升了对应质量，同时在经典基准上保持了有竞争力的性能。

    arXiv:2610.12421v1 Announce Type: cross  Abstract: Dense correspondence matching has historically been bounded by simplifying spatio-temporal priors, such as smooth motion and rigid geometry. While effective for classical tasks, these assumptions break down in image editing and reference-guided generation (IEG), where transformations can preserve visual identity while breaking physical continuity. To establish identity-preserving correspondence across such transformations, we introduce FreeMatching, a generalizable framework combining generative and semantic foundation representations with heterogeneous supervision from classical datasets, tracked videos, and synthetic scenes. Teacher-guided iterative refinement further improves correspondence in IEG without dense correspondence annotations. Experimentally, a single FreeMatching model substantially improves correspondence quality on challenging IEG image pairs while retaining competitive performance on classical benchmarks. Furthermore
    
[^12]: 面向安全关键强化学习的统一贝尔曼算子

    A Unified Bellman Operator for Safety-Critical Reinforcement Learning

    [https://arxiv.org/abs/2610.12420](https://arxiv.org/abs/2610.12420)

    本文提出一种将性能与安全目标统一到单一联合价值函数中的新颖贝尔曼算子，并在双时间尺度随机逼近框架下证明了其时序差分学习的收敛性，从而无需先验知识即可为安全关键强化学习提供严格的安全保证。

    

    安全关键领域的强化学习要求在严格遵守安全约束的同时最大化任务性能。现有的安全强化学习范式通常迫使人进行权衡：要么需要先验知识来提供严格的安全保证（例如安全过滤器），要么可以实现联合学习，但只能在平均意义上满足安全约束。在本工作中，我们提出了一种新颖的贝尔曼算子，将性能目标与安全目标统一到一个联合价值函数中。我们证明了在双时间尺度随机逼近框架下，采用该联合贝尔曼算子的时序差分学习是收敛的。在快时间尺度上估计学习所得联合策略的安全价值，而在慢时间尺度上估计联合价值。通过将极限动力学表述为占用平均微分包含并证明其渐近收敛，从而确保了收敛性。

    arXiv:2610.12420v1 Announce Type: new  Abstract: Reinforcement learning in safety-critical domains requires maximizing task performance while strictly adhering to safety constraints. Existing safe reinforcement learning paradigms typically force a trade-off: they either require a priori knowledge to provide strict safety guarantees (e.g., safety filters), or they enable joint learning but only satisfy safety constraints on average. In this work, we propose a novel Bellman operator that unifies performance and safety objectives into a joint value function. We show that temporal difference learning with the joint Bellman operator converges under a two-timescale stochastic approximation framework. On the fast timescale, the safety value of the learning joint policy is estimated, while the joint value is estimated on the slow timescale. Convergence is ensured by formulating the limiting dynamics as an occupation-averaged differential inclusion, and showing that it asymptotically converges 
    
[^13]: WOVEN：将视觉世界建模编织进多模态大语言模型

    WOVEN: Weaving Visual World Modeling into Multimodal LLMs

    [https://arxiv.org/abs/2610.12417](https://arxiv.org/abs/2610.12417)

    提出WOVEN——一个按场景、动作和推理类型组织的视觉转换推理训练数据源与基准（含36,076个示例），验证视觉转换推理可作为可跨任务复用的共享训练原语，以提升多模态大语言模型的空间、具身、物理和时间推理能力。

    

    多模态大语言模型（MLLM）在空间、具身、物理和时间推理方面表现不佳。我们假设这些失败反映了一个共同的缺陷——视觉转换推理能力的不足，并测试这一能力是否可以作为共享的训练原语，即不同模型可以从不同的监督来源中学习该能力，并在不同任务间复用，同时我们提供了系统化的训练方案。现有的基准测试只是分别记录这些缺陷，但无法支持跨场景、动作和推理操作的可控比较。因此，我们提出了WOVEN，一个面向视觉转换推理的训练数据源与基准，它按照场景、动作和推理类型组织转换监督信息，使用来自视频预训练生成模型的多样化、逼真的rollout数据：涵盖20种场景类型、5种动作类型和8种推理类型，共计36,076个示例。我们首先评估了38个前沿的多模态大语言模型（如GPT-5.4和Qwen3-VL等）。

    arXiv:2610.12417v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) struggle with spatial, embodied, physical, and temporal reasoning. We hypothesize that these failures reflect a shared deficit in visual transition reasoning, and test whether this capability can serve as a shared training primitive, one that different models can learn from different supervision sources and reuse across different tasks, with a systematic training recipe. Existing benchmarks document these deficits separately but do not support controlled comparisons across scenes, actions, and reasoning operations. We therefore introduce WOVEN, a training source and benchmark for visual transition reasoning that organizes transition supervision by scene, action, and reasoning type, using diverse, realistic rollouts from video-pretrained generative models: 36,076 examples across 20 scene types, 5 action types, and 8 reasoning types. We first evaluate 38 frontier MLLMs (e.g., GPT-5.4 and Qwen3-VL-
    
[^14]: 基于价值表征预测对齐泛化

    Predicting Alignment Generalization with Value Representations

    [https://arxiv.org/abs/2610.12410](https://arxiv.org/abs/2610.12410)

    本文提出“对齐泛化预测”新任务，通过对66个价值的大规模分析发现，基于模型激活的价值表征能够显著优于基于文本描述的方法，预测模型微调遵循某一价值后在未见价值上的行为泛化。

    

    LLM开发者通过后训练使模型展现亲社会的价值与行为特质，这些价值与特质被列举在对齐目标中。然而，尽管近期的后训练进展使模型在对齐评估中获得了高分，基于狭窄行为集合训练模型仍会以意想不到的方式影响模型在未见过的情境和环境中的行为。本文提出了“对齐泛化预测”这一新任务，即预测将模型微调为遵循某一给定价值后，其行为在一系列保留价值上将如何变化。我们对现代对齐目标中的66个价值进行了大规模的对齐泛化效应分析，并在对齐泛化预测任务上对多种表征技术进行了基准测试。研究发现，基于模型在上下文中应用价值时的激活的表征方法，显著优于基于文本描述的方法。

    arXiv:2610.12410v1 Announce Type: cross  Abstract: LLM developers post-train their models to exhibit prosocial values and behavioral traits, which are enumerated in an alignment target. However, while recent post-training developments have yielded models that score highly on alignment evaluations, training models on sets of narrow behaviors still influences their behavior across unseen contexts and environments in unexpected ways. In this paper, we establish the task of alignment generalization prediction, i.e., predicting how fine-tuning a model to follow a given value changes its behavior across a wide range of held-out values. We conduct a large-scale analysis of alignment generalization effects across 66 values found in modern alignment targets, and benchmark representational techniques on the alignment generalization prediction task. We find that representations based on model activations when applying values in context significantly outperform methods based on textual description
    
[^15]: 基于全球-区域对齐的公里级天气预测学习

    Learning Kilometer-Scale Weather Prediction with Global-Regional Alignment

    [https://arxiv.org/abs/2610.12401](https://arxiv.org/abs/2610.12401)

    提出了一种名为 ScaleCast 的区域天气预报框架，通过全球-区域对齐技术，复用预训练全球天气模型的大尺度预报来引导公里级区域预测，解决了跨网格表征对齐和全球引导与局部交互融合的难题。

    

    公里级区域天气预报对于本地天气预警和天气敏感型决策至关重要。现有的数据驱动方法通常依赖数值预报来提供大尺度引导，或者需要对全球预报组件进行额外的训练。预训练的全球天气模型为大规模预报提供了高效的来源，这促使人们将其复用，以引导高分辨率的区域预测。然而，这种耦合需要在不同的网格之间对齐全球与区域表征，并将全球引导信息与局部交互相结合来推进区域状态演变。我们提出了 ScaleCast，一个通过全球-区域对齐来解决上述挑战的区域预报框架。其全球-区域转换模块将全球与区域的联合表征对齐到区域位置上，而全球-区域对齐与动力学模块则将经过对齐的引导信息与区域邻域交互相结合，以推进区域状态演变。

    arXiv:2610.12401v1 Announce Type: new  Abstract: Kilometer-scale regional weather forecasting is essential for local weather warnings and weather-sensitive decisions. Existing data-driven approaches often rely on numerical forecasts for large-scale guidance or require additional training of global forecasting components. Pretrained global weather models offer an efficient source of large-scale forecasts, motivating their reuse to guide high-resolution regional prediction. However, this coupling requires aligning global and regional representations across different grids and integrating global guidance with local interactions to advance regional states. We propose ScaleCast, a regional forecasting framework that addresses these challenges through Global-Regional Alignment. Its Global-Regional Conversion module aligns joint global and regional representations with regional locations, while the Global-Regional Alignment and Dynamics block combines aligned guidance with regional neighborho
    
[^16]: 基于源侧训练动态对分布外（OOD）退化的前瞻性预测

    Prospective Prediction of OOD Degradation from Source-Side Training Dynamics

    [https://arxiv.org/abs/2610.12397](https://arxiv.org/abs/2610.12397)

    论文证明了仅利用源侧训练动态（尤其是置信度和熵的时间汇总特征）即可在分布外退化发生之前预测到它，且该早期预警信号无需额外训练即可跨模型架构迁移。

    

    我们研究了能否仅利用源侧训练动态，在直接观察到持续性分布外退化之前对其进行预测。在一个受控的捷径学习设置中，一个简单的逻辑回归预测器展现出明显的前瞻性信号，而仅凭训练时间则无法做到。源侧各量的时间汇总信息远比其当前值更有价值。当从CNN无需额外训练地迁移到MLP时，置信度和熵的动态仍保留了大量预测信息。这些结果提供了一个原理性证明：源侧训练动态可以包含对未来OOD失败的早期预警信号。

    arXiv:2610.12397v1 Announce Type: new  Abstract: We study whether persistent out-of-distribution (OOD) degradation can be predicted before it is directly observed using only source-side training dynamics. In a controlled shortcut-learning setting, a simple logistic regression predictor develops a clear prospective signal, while training time alone does not. Temporal summaries of the source-side quantities are substantially more informative than their current values. When transferred without additional training from a CNN to an MLP, confidence and entropy dynamics retain substantial predictive information. These results provide a proof of principle that source-side training dynamics can contain an early warning signal for future OOD failure.
    
[^17]: HRIL：通过高阶张量建模学习多模态协同信息

    HRIL: Learning Multimodal Synergy via Higher-Order Tensor Modeling

    [https://arxiv.org/abs/2610.12393](https://arxiv.org/abs/2610.12393)

    提出HRIL方法，通过在模态嵌入上构建经验交叉矩张量进行高阶张量建模，显式捕获体现为高阶统计依赖的多模态协同信息，从而在自监督多模态表示学习中保留协同信号的信息容量。

    

    自监督多模态表示学习在众多领域取得了显著成功，然而由于跨模态交互的复杂性，捕获协同信息仍然具有挑战性。与各模态之间的共享信息不同，协同信息是指只有通过多个模态的联合配置才能产生任务相关信号、且无法从任何单一模态中独立恢复的信息。本工作聚焦于如何在多模态表示中保留此类协同信号的信息容量。关键观察在于：协同信息体现为模态之间的高阶统计依赖，这为显式建模联合交互提供了一个有原则的目标。受此洞察启发，我们提出了高阶表示与信息学习，通过在模态嵌入上构建经验交叉矩张量来表示多……（原文摘要在此处截断）

    arXiv:2610.12393v1 Announce Type: new  Abstract: Self-supervised multimodal representation learning has achieved remarkable success across diverse domains, yet capturing synergistic information remains challenging due to the complexity of cross-modal interactions. Unlike the shared information across individual modalities, synergy arises when task-relevant signals emerge only from the joint configuration of multiple modalities and cannot be recovered from any modality in isolation. This work focuses on how to preserve the information capacity for such synergistic signals in multimodal representations. The key observation is that synergistic information is reflected in higher-order statistical dependence among modalities, which provides a principled target for explicitly modeling joint interactions. Motivated by this insight, we propose Higher-order Representation and Information Learning (HRIL), which constructs an empirical cross-moment tensor over modality embeddings to represent mul
    
[^18]: 从长文本到预测特征：基于可执行程序搜索的LLM引导分块式特征工程

    Long Text to Predictive Features: LLM-Guided Blockwise Feature Engineering via Executable Program Search

    [https://arxiv.org/abs/2610.12390](https://arxiv.org/abs/2610.12390)

    提出LLM-BlockFE框架，由LLM在离线阶段通过逐步追加代码块并结合深度校准信用分配的分块级回滚搜索，将长文本自动转化为可执行的特征程序，使在线推理无需调用LLM即可高效利用长文本信息。

    

    工业风控系统通常依赖结构化数据模型进行高效预测，然而大量有价值的信息仍然蕴含在非结构化的长文本中。通过人工特征工程提取这些信息费时费力，而依赖大语言模型（LLM）处理每一次实时输入又可能无法满足实际部署需求。为应对这一挑战，我们提出了LLM-BlockFE，这是一个由LLM引导的离线特征构建框架，可将长文本转换为可执行的特征程序，从而避免在线推理阶段调用LLM。LLM-BlockFE通过逐步追加不可变的代码块来构建特征程序，并利用下游模型评估候选特征。为解决传统贪心搜索容易陷入次优解的问题，我们的方法引入了一种基于深度校准信用分配的分块级回滚机制（摘要在此处截断）

    arXiv:2610.12390v1 Announce Type: cross  Abstract: Industrial risk-control systems typically rely on structured-data models for efficient prediction, yet substantial valuable information remains embedded in unstructured long text. Extracting this information through manual feature engineering is labor-intensive, while requiring a large language model (LLM) to process every real-time input may not meet practical deployment requirements. To address this challenge, we propose LLM-BlockFE, an LLM-guided offline feature construction framework that converts long text into executable feature programs, thereby avoiding LLM calls during online inference. LLM-BlockFE constructs feature programs by incrementally appending immutable code blocks and evaluates candidate features using a downstream model. To address the tendency of conventional greedy search to become trapped in suboptimal solutions, our method introduces a block-level rollback mechanism based on depth-calibrated credit allocation an
    
[^19]: Marformer：一种用于预测缺失数据分布的Transformer

    Marformer: A Transformer for Predicting Missing Data Distributions

    [https://arxiv.org/abs/2610.12379](https://arxiv.org/abs/2610.12379)

    本文提出Marformer，一种受BERT启发的Transformer模型，可根据任意已观测变量集合直接预测缺失变量的条件边际分布，无需建模完整联合分布或领域知识，且能在单次前向传播中完成所有预测，从而支持贝叶斯风险与信息价值的计算。

    

    现实中的决策往往是在信息不完整的情况下做出的。如果我们只能观察到所需随机变量中的一部分，我们便可以预测其余变量。缺失变量上的**条件边际分布**是计算贝叶斯风险和信息价值的关键要素，后者是在决策前多获取一次观测所能带来的期望收益。我们提出了Marformer，一个经过训练、能够根据任意已观测值集合直接预测条件边际分布的Transformer。与被训练用于从上下文中预测缺失词语的BERT类似，Marformer为每个分布 p(X_i) 构建一个隐向量表示，并通过注意力机制关注其他分布 p(X_j) 对其进行迭代式细化。与生成式方法不同，Marformer不对完整的联合分布进行建模，无需关于数据生成过程的领域知识，并能在单次前向传播中完成所有预测。我们在三个合成领域上对该模型进行了评估。

    arXiv:2610.12379v1 Announce Type: new  Abstract: Real decisions are made under incomplete information. If we observe only some of the random variables we need, we can predict the others. The \textbf{conditional marginals} over the missing variables are the key ingredient for computing Bayes risk and Value of Information (VOI), the expected gain from acquiring one more observation before deciding. We present the Marformer, a Transformer trained to directly predict conditional marginals given any set of observed values. Like BERT, which is trained to predict missing words from context, the Marformer constructs a hidden-vector representation for each distribution $p(X_i)$ and iteratively refines it through attention to other distributions $p(X_j)$. Unlike generative approaches, the Marformer does not model the full joint distribution, requires no domain knowledge of the data-generating process, and makes all predictions in a single forward pass. We evaluate across three synthetic domains 
    
[^20]: OnTrack：基于流式结构感知最优传输的LLM智能体轨迹实时监控与干预

    OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport

    [https://arxiv.org/abs/2610.12375](https://arxiv.org/abs/2610.12375)

    OnTrack提出了一种流式结构感知最优传输监控机制，通过将LLM智能体的执行步骤与记录的成功运行轨迹实时对比，在每步约一毫秒内实现对异常行为的告警或阻断，兼顾了低延迟与安全性。

    

    智能体被部署于从行程规划、股票交易到IT事件分诊等各类应用中。在大多数情况下，LLM智能体在仅有极少基于规则的安全保障下自主运行，导致不可逆操作带来成本与安全问题。近期的工作要么通过使用一个安全防护智能体来监控行为，要么在事后评估日志：前者为每一步增加了成本和延迟；后者则在运行结束后才给出判定，此时令牌已被消耗、损害已经造成。为克服这些问题，我们提出OnTrack，一种流式监控机制，它将智能体的步骤及其依赖关系与记录下来的成功运行轨迹进行对比，从而在大约每步一毫秒的时间内向用户告警或阻断智能体。我们在三种访问程度递减的场景下研究该问题：完全参考访问（历史运行记录与工具模式）、中间访问（仅有工具模式）以及无先验知识（仅有实时生成的步骤日志）。

    arXiv:2610.12375v1 Announce Type: new  Abstract: Agents are deployed in applications from trip planners and stock trading to IT incident triage. In most cases, LLM agents work autonomously with minimal rule-based safeguarding, leading to cost and safety issues from irreversible actions. Recent works resolve this either by using a safeguard agent to monitor behavior or evaluating logs post-hoc. The first adds cost and latency to every step; the second delivers its verdict after the run, when tokens are burned and damage is done. To overcome this, we propose OnTrack, a streaming monitoring mechanism that compares an agent's steps and dependencies against recorded successful runs to alert users or block the agent in about a millisecond per step. We study this problem in three regimes of decreasing access: full reference access (historical runs and tool schemas), intermediate access (only tool schemas), and no prior knowledge (only step logs as generated). Expectation of OnTrack's monitori
    
[^21]: 基于核方法自编码器的双层优化数据驱动Koopman嵌入学习

    Bilevel optimization for data-driven learning of Koopman embeddings using kernel-based autoencoders

    [https://arxiv.org/abs/2610.12370](https://arxiv.org/abs/2610.12370)

    本文提出了一种结合配置方法与双层优化的新方法EDMD-kDL，利用基于核的自编码器直接从数据中学习有限维Koopman嵌入，克服了传统EDMD需先验指定字典的局限，同时相比神经网络方法具有更好的可解释性和理论可分析性。

    

    Koopman算子理论为分析非线性动力系统提供了一个线性框架，并已成为数据驱动建模的主要工具。然而，一个核心挑战在于，由扩展动态模态分解（EDMD）等方法计算的有限维近似需要事先指定字典。近期的机器学习方法通过从数据中学习字典来解决这一局限，其中主要采用人工神经网络（ANN）自编码器架构。尽管核方法提供了一种更具可解释性且更易于理论分析的替代方案，但它们在该领域中受到的关注却很少。我们提出了基于核字典学习的扩展动态模态分解（EDMD-kDL），这是一种直接从数据中学习有限维Koopman嵌入的基于核的方法。该方法结合了配置方法和双层优化的思想。

    arXiv:2610.12370v1 Announce Type: new  Abstract: Koopman operator theory provides a linear framework for analyzing nonlinear dynamical systems and has become a major tool for data-driven modeling. A central challenge, however, is that finite-dimensional approximations computed by methods such as extended dynamic mode decomposition (EDMD) require the dictionary to be specified a priori. Recent machine-learning approaches address this limitation by learning the dictionary from data, predominantly using artificial neural network (ANN) autoencoder architectures. Although kernel methods offer an alternative with greater interpretability and tractability for theoretical analysis, they have received little attention in this setting. We introduce extended dynamic mode decomposition with kernel-based dictionary learning (EDMD-kDL), a kernel-based method for learning finite-dimensional Koopman embeddings directly from data. The method combines ideas from collocation methods and bilevel optimizat
    
[^22]: 缩小对抗性马尔可夫决策过程策略优化中的视野差距

    Closing the Horizon Gap in Policy Optimization for Adversarial MDPs

    [https://arxiv.org/abs/2610.12362](https://arxiv.org/abs/2610.12362)

    该论文提出使用正则化Q函数在所有状态-动作对上联合控制局部更新稳定性，从而将对抗性MDP策略优化的遗憾界对视野H的依赖性改进到与基于占用测度的算法相当的水平。

    

    我们研究具有对抗性损失和老虎机反馈的在线情景表格马尔可夫决策过程（MDP）的策略优化问题。策略优化在每个状态处对策略进行局部更新，避免了在占用测度多面体上进行优化的复杂性，但其现有的遗憾界比基于占用测度的算法大一个视野因子 $H$。我们通过使用正则化 $Q$ 函数来弥补这一差距，这使得我们能够在所有状态-动作对上联合控制局部更新的稳定性，而不是在每个状态处分别控制。由此得到的算法在转移已知的情况下达到 $\widetilde O(\sqrt{HS(H+A)T})$ 的高概率遗憾界，在转移未知的情况下达到 $\widetilde O(HS\sqrt{AT})$ 的高概率遗憾界，其中 $S$ 为状态数，$A$ 为动作数，$T$ 为回合数。这两个界都改善了现有策略优化界对视野的依赖性，且后者匹配了已知的最优结果。

    arXiv:2610.12362v1 Announce Type: new  Abstract: We consider policy optimization for online episodic tabular Markov decision processes (MDPs) with adversarial losses and bandit feedback. Policy optimization updates the policy locally at each state and avoids optimization over the occupancy-measure polytope, but its existing regret bounds are larger by a factor of the horizon $H$ than those of occupancy-measure-based algorithms. We close this gap by using regularized $Q$-functions, which allow us to control the stability of the local updates jointly over all state-action pairs rather than separately at each state. The resulting algorithm attains high-probability regret bounds of $\widetilde O(\sqrt{HS(H+A)T})$ for known transitions and $\widetilde O(HS\sqrt{AT})$ for unknown transitions, where $S$ is the number of states, $A$ the number of actions, and $T$ the number of episodes. Both bounds improve the horizon dependence of existing policy optimization bounds, and the latter matches th
    
[^23]: 布尔立方体上的子空间不确定性与精确采样阈值

    Subspace Uncertainty and Sharp Sampling Thresholds on the Boolean Cube

    [https://arxiv.org/abs/2610.12358](https://arxiv.org/abs/2610.12358)

    该论文确定了在布尔立方体上已知的低度数多项式子空间中进行高斯回归并达到给定极小化极大精度所需的精确样本阈值，其阈值由熵函数形式的指数因子刻画，同时在固定泄漏率下锐化了Polyanskiy–Samorodnitsky不确定性原理。

    

    我们在d维布尔立方体上度数不超过k的函数所构成的一个已知m维子空间中，研究平方总体L2损失下的高斯回归问题。随机输入可能对预测至关重要的区域采样不足，即使在模型已知的情况下，也会延迟参数化速率的达成。对于固定的q₀<1/2、1≤k≤q₀d以及足够大的固定常数A，在置信水平1-e^{-t}（t≥log 4）下，达到极小化极大误差Aσ²(m+t)/n的最坏子空间样本阈值为 N=(m+t)exp{E_{d,k}+O(k^{1/3})}，其中 E_{d,k}=dΨ(k/d)，Ψ(q)=log2−H(1/2−√(q(1−q)))，H为以自然对数定义的二元熵函数。该上界对每个可行的m均成立；匹配的下界在 m≤C(d,⌊k^{1/3}⌋) 或 t≥m 时成立。我们在两个方面对Polyanskiy–Samorodnitsky不确定性原理进行了锐化。首先，对于固定的泄漏率ρ∈(0,1)，携带……的最小集合……（原文摘要在此处截断）

    arXiv:2610.12358v1 Announce Type: cross  Abstract: We study Gaussian regression under squared population $L_2$ loss in a known $m$-dimensional subspace of degree-at-most-$k$ functions on the $d$-dimensional Boolean cube. Random inputs can undersample regions essential for prediction, delaying the parametric rate even when the model is known.   For fixed $q_0<1/2$, $1\le k\le q_0d$, and sufficiently large fixed $A$, the worst-subspace sample threshold for minimax error $A\sigma^2(m+t)/n$ with confidence $1-e^{-t}$, $t\ge\log4$, is \[ N=(m+t)\exp\{E_{d,k}+O(k^{1/3})\}, \quad E_{d,k}=d\Psi(k/d), \] where $\Psi(q)=\log2-\mathsf H(\tfrac12-\sqrt{q(1-q)})$ and $\mathsf H$ is binary entropy with natural logarithms. The upper bound holds for every feasible $m$; the matching lower bound holds when $m\le\binom d{\lfloor k^{1/3}\rfloor}$ or $t\ge m$.   We sharpen the Polyanskiy--Samorodnitsky uncertainty principle in two respects. First, for fixed leakage $\rho\in(0,1)$, the smallest set carrying
    
[^24]: SplitJEPA：无需重构地学习潜在世界中的不变与可变因素

    SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction

    [https://arxiv.org/abs/2610.12349](https://arxiv.org/abs/2610.12349)

    提出SplitJEPA，一种无需重构即可在JEPA框架内直接学习潜在状态的不变（跨观测共享）与可变因素结构的预测架构。

    

    理解一个动态的世界，仅仅拥有一个概括观测的潜在状态是不够的：该状态还应被组织成两部分——在相关观测之间保持共享的因素，以及在观测之间发生变化的因素。例如，一个正在把立方体推向目标的机器人，当摄像机移动或灯光变暗时，应当采取相同的动作，因为场景中并没有任何物体移动。现有的实现这种分解的方法通常依赖于重构，因此潜在变量必须先能够解释整个观测世界，其内部组织结构才能被信任。联合嵌入预测架构（JEPA）直接对潜在状态建模且从不进行重构，然而目前尚无研究结果能够恢复其学习到的状态中的不变与可变部分。因此，如何在不付出重构代价的情况下学习潜在世界的不变-可变结构，仍然是一个悬而未决的问题。为弥合这一空白，我们提出了SplitJEPA...

    arXiv:2610.12349v1 Announce Type: new  Abstract: Understanding a dynamical world calls for more than a latent state that summarizes its observations: the state should also be organized into the factors that stay shared across related observations and the factors that vary between them. For example, a robot pushing a cube to a goal should take the same action when the camera shifts or the lights dim, since nothing in the scene has moved. Existing approaches to this decomposition commonly obtain it through reconstruction, so the latent variables must first explain the entire observational world before their organization can be trusted. Joint embedding predictive architectures (JEPAs) model the latent state directly and never reconstruct, yet no existing result recovers the invariant and variant parts of the state they learn. How to learn the invariant-variant structure of the latent world without paying for its reconstruction therefore remains open. To close this gap, we introduce SplitJ
    
[^25]: 克服先验壁垒：长尾分布下的监督微调

    Overcoming Prior Barriers: Supervised Fine-Tuning under Long-Tail Distribution

    [https://arxiv.org/abs/2610.12345](https://arxiv.org/abs/2610.12345)

    该论文提出“先验壁垒”新概念，揭示预训练模型对各概念的支持程度呈长尾分布、使头部与尾部概念在监督微调时起点不同，并通过理论推导预测风险界，指出尾部概念需要额外指令才能克服高先验壁垒。

    

    监督微调（SFT）用于将预训练大语言模型（LLM）适配到下游任务，但任务所需的概念在预训练阶段可能获得截然不同的支持程度。高频概念更容易被充分学习，而低频概念可能仍然表征薄弱。我们提出了一个名为“先验壁垒”的新概念，用以量化预训练模型对竞争概念相对于目标概念的支持强度。我们观察到先验壁垒呈长尾分布，使得头部概念与尾部概念在SFT时处于不同的起点：头部概念面临较低的先验壁垒，而尾部概念则需要额外的指令来克服其较高的先验壁垒。我们的理论分析进一步推导了长尾先验壁垒下SFT的预测风险界，明确刻画了先验壁垒与累积的SFT证据如何共同决定预测性能。受此启发……（原文摘要在此处被截断）

    arXiv:2610.12345v1 Announce Type: new  Abstract: Supervised fine-tuning (SFT) adapts pretrained large language models (LLMs) to downstream tasks, but the required concepts can receive substantially different levels of pretrained support. Frequent concepts are more likely to be well learned, whereas rare concepts may remain weakly represented. We introduce a novel notion named prior barrier to quantify how strongly the pretrained model supports competing concepts over the target concept. We observe that prior barriers follow a long-tail distribution, placing head and tail concepts at different starting points for SFT: head concepts face lower prior barriers, whereas tail concepts require additional instructions to overcome their higher prior barriers. Our theoretical analysis further derives a predictive risk bound for SFT under long-tail prior barriers, explicitly characterizing how the prior barrier and accumulated SFT evidence jointly determine predictive performance. Motivated by th
    
[^26]: 环境离散扩散：在正确的时机使用“错误”的数据以实现数据高效学习

    Ambient Discrete Diffusion: Using the Wrong Data at the Right Time for Data Efficient Learning

    [https://arxiv.org/abs/2610.12340](https://arxiv.org/abs/2610.12340)

    RefineMix是一种在数据稀缺条件下训练离散扩散模型的框架，它巧妙利用离散扩散中低噪声水平下掩码保留领域信息的特性，在不引入采样偏差的前提下利用分布外数据提升泛化能力，在五个领域偏移设置中达到或超越领域内微调与数据混合方法的效果。

    

    我们提出了RefineMix，这是一个在严重数据稀缺条件下训练离散扩散模型的框架，而数据稀缺是科学应用中的常见约束。RefineMix在选定的扩散时间点使用分布外数据来提升泛化能力，同时不会使采样分布产生偏差。尽管这一策略已在连续扩散中得到探索，但离散扩散带来了独特的挑战：与高斯噪声不同，掩码操作会在存留的词元中保留领域信息，限制了相关数据在高噪声水平下的使用。然而，在低噪声水平下，各领域实际上不相交的支撑集反而成为一种优势，使模型能够同时从领域内数据和分布外数据中学习，而不会使采样器产生偏差。我们将这些直觉形式化，并为所提出的方法提供了理论分析。在实验中，跨五种领域偏移设置，RefineMix的表现与领域内微调和数据混合方法相当或更优。

    arXiv:2610.12340v1 Announce Type: new  Abstract: We introduce RefineMix, a framework for training discrete diffusion models under severe data scarcity, a common constraint in scientific applications. RefineMix uses out-of-distribution data at selected diffusion times to improve generalization without biasing the sampling distribution. Although this strategy has been explored in continuous diffusion, discrete diffusion presents a distinct challenge: unlike Gaussian noise, masking preserves domain information in surviving tokens, limiting the use of related data at high noise levels. At low noise levels, however, the domains effectively disjoint supports become an advantage, allowing the model to learn from both in-domain and out-of-distribution data without biasing the sampler. We formalize these intuitions and provide a theoretical analysis for the proposed method. Experimentally, across five domain-shift settings, RefineMix matches or outperforms in-domain finetuning and data mixing. 
    
[^27]: VFold：对称性感知的跨层价值缓存压缩

    VFold: Symmetry-Aware Cross-Layer Value Cache Compression

    [https://arxiv.org/abs/2610.12338](https://arxiv.org/abs/2610.12338)

    该论文提出了一种对称性感知的跨层价值缓存合并策略，无需修改模型架构即可在解码时压缩KV缓存内存，并能与高比率量化或键缓存剪枝等现有技术组合使用，实现单一方法无法达到的更高压缩比。

    

    虽然缓存键值状态可以加速大语言模型（LLM）的解码，但在长上下文长度下，这种缓存可能会占据主要的内存用量。一种解决方案是利用层间缓存的相似性来压缩这部分内存。然而，大多数现有技术需要对LLM进行架构更改，并带来大量开销。在这项工作中，我们提出了一种对称性感知的价值缓存合并策略，能够在解码过程中减少缓存内存，同时避免有害的性能下降和架构开销。此外，我们展示了该方法可以与现有的缓存压缩技术结合使用，与高比率量化或键缓存剪枝相组合，达到单独使用任一方法都无法实现的压缩比，且额外成本极低。最终，我们的发现揭示了价值缓存中一个主要的未被充分利用的容量来源，为扩展上下文长度提供了一个简单而高效的方向。

    arXiv:2610.12338v1 Announce Type: new  Abstract: While caching key-value (KV) states accelerates Large Language Model (LLM) decoding, this cache can dominate memory usage at long context lengths. One solution is to compress this memory by exploiting inter-layer cache similarities. However, most existing techniques necessitate architectural changes to LLMs and incur substantial overhead. In this work, we propose a symmetry-aware value cache merging strategy that reduces cache memory while avoiding both harmful performance degradation and architectural overhead during decoding. Furthermore, we show that this approach can be exploited alongside existing cache compression techniques, composing with high-ratio quantization or key cache pruning to reach compression ratios that neither method reaches alone, with minimal additional cost. Ultimately, our findings reveal a major source of underutilized capacity in the value cache, offering a simple yet highly effective direction for scaling cont
    
[^28]: asdex：JAX 中的自动稀疏微分

    asdex: Automatic Sparse Differentiation in JAX

    [https://arxiv.org/abs/2610.12336](https://arxiv.org/abs/2610.12336)

    本文提出了 JAX 中的自动稀疏微分工具 asdex，通过稀疏模式检测、图着色、压缩微分与解压缩四个步骤，利用雅可比矩阵的稀疏结构，使自动微分传递次数与问题维度无关，从而大幅提升计算效率。

    

    科学计算和机器学习中的许多任务都需要函数的雅可比矩阵或海森矩阵。自动微分（AD）能够以机器精度计算这些导数，但要生成一个稠密的 m×n 雅可比矩阵，需要进行 n 次前向模式或 m 次反向模式的 AD 传递，即每个列或行各一次。对于一大类函数而言，每个输出仅依赖于少数几个输入，这使得导数矩阵是稀疏的。自动稀疏微分（ASD）通过四个步骤利用这一结构特性：检测与输入无关的稀疏模式、对图进行着色以将可共享一次 AD 传递的列或行分组、通过每个颜色一次 AD 传递的压缩微分计算压缩导数矩阵，最后将其解压缩还原至原始稀疏模式。颜色数量（即 AD 传递次数）通常与问题的维度无关：例如，具有 b 个连续带的带状雅可比矩阵只需要……

    arXiv:2610.12336v1 Announce Type: cross  Abstract: Many tasks in scientific computing and machine learning require the Jacobian or Hessian matrix of a function. Automatic differentiation (AD) computes these derivatives to machine precision, but materializing a dense $m \times n$ Jacobian requires $n$ forward-mode or $m$ reverse-mode AD passes, one per column or row. For a large class of functions, each output depends on only a few inputs, making the derivative matrix sparse. Automatic sparse differentiation (ASD) exploits this structure in four steps: detection of the input-agnostic sparsity pattern, coloring of a graph to group columns or rows that can share an AD pass, compressed differentiation to compute a compressed derivative matrix with one AD pass per color, and finally decompression into the original sparsity pattern. The number of colors, and hence of AD passes, is often independent of the problem dimension: a banded Jacobian with $b$ contiguous bands, for instance, only ever
    
[^29]: RiCo：通过局部接触推理实现刚体交互的神经模拟

    RiCo: Neural Simulation of Rigid-Body Interactions via Local Contact Reasoning

    [https://arxiv.org/abs/2610.12333](https://arxiv.org/abs/2610.12333)

    提出RiCo方法，利用接触表面点的稀疏邻域进行局部接触推理，将跨物体交互限定在相邻表面并传播刚体内部的接触信息，从而实现更精确的刚体交互神经模拟。

    

    刚体交互的精确模拟对于构建具有预测能力的物理世界模型至关重要。尽管近期在物体动力学建模方面取得了进展，但如何捕捉表面之间的局部接触如何塑造物体运动仍然是一个挑战。虽然端到端的世界模型能够预测整个场景或物体层面的交互，但在实际中，刚体接触本质上是局部的——只有相邻的表面才能直接交换接触力。受这一观察的启发，我们提出了刚体接触推理方法，通过接触表面点的稀疏邻域来表示物体之间的交互。RiCo 将每个点的状态与邻近表面的相对几何形状、运动和物理属性相结合，然后在物体的各点之间进行推理，以确定这些局部接触如何共同影响物体的运动。通过将跨物体推理限定在相邻表面之间，同时在每个刚体内部传播接触信息（摘要在此处截断）。

    arXiv:2610.12333v1 Announce Type: cross  Abstract: Accurate simulation of rigid-body interactions is essential for predictive physical world models. Despite recent progress in modeling object dynamics, capturing how local contacts between surfaces shape object motion remains challenging. While end-to-end world models predict interactions across entire scenes or objects, in practice, rigid-body contact is inherently local, and only nearby surfaces can directly exchange contact forces. Motivated by this observation, we introduce Rigid-body Contact Reasoning (RiCo), which represents interactions between objects through sparse neighborhoods of contact surface points. RiCo combines each point's state with the relative geometry, motion, and physical properties of nearby surfaces, then reasons across the object's points to determine how these local contacts jointly affect its motion. By confining cross-object reasoning to nearby surfaces while propagating contact information within each rigid
    
[^30]: 面向处理效应估计的预测驱动数据融合方法

    Prediction-Powered Data Fusion for Treatment Effect Estimation

    [https://arxiv.org/abs/2610.12332](https://arxiv.org/abs/2610.12332)

    提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。

    

    随机对照试验（RCT）能够在无混杂的情况下识别处理效应，但样本量通常较小；而观察性研究（OBS）样本量大，但可能存在混杂。目前已有许多将小型RCT与大型OBS相结合的估计方法用于估计平均处理效应（ATE）和条件平均处理效应（CATE）。然而，现有的ATE估计方法要么对OBS施加额外假设，要么未能从OBS中充分借力。相比ATE，CATE的研究相对较少，现有的CATE方法要么假设OBS无混杂，要么依赖于混杂函数的模型，要么以引入偏差为代价来换取更低的方差。为此，我们提出了一个框架，在对OBS不做任何特殊假设的前提下融合OBS与RCT：既保持基于RCT估计的无偏性，又从大型OBS中借力以提升估计精度。基于这一原则，我们构建了具有闭式权重的ATE估计器AIPW-Fusion……

    arXiv:2610.12332v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) identify treatment effects without confounding but are often small, whereas observational studies (OBS) are large but may be confounded. Many estimators combining a small RCT with a large OBS have been developed for the average treatment effect (ATE) and the conditional ATE (CATE). However, existing ATE estimators either make assumptions on the OBS or do not borrow enough power from them. The CATE has been studied less than the ATE. Existing CATE methods either assume the OBS are unconfounded, rely on a model of the confounding function, or accept bias in exchange for lower variance. We therefore propose a framework that, without special assumptions on the OBS, fuses the OBS and the RCT by preserving the unbiasedness of RCT-based estimation while borrowing power from the large OBS to boost precision. Applying this principle, we build an ATE estimator, AIPW-Fusion, with closed-form weights and confide
    
[^31]: 复合在线到非凸转换及最优预言机复杂度

    Composite Online-to-Nonconvex Conversion with Optimal Oracle Complexity

    [https://arxiv.org/abs/2610.12328](https://arxiv.org/abs/2610.12328)

    该论文通过为在线学习者设计新的损失函数，将在线到非凸转换框架扩展至复合优化场景，首次在一阶随机预言机访问下建立了复合非凸优化的最优复杂度保证。

    

    我们研究随机非光滑非凸复合优化问题，其中包含若干重要问题，如约束优化和神经网络的正则化训练。目标函数是一个可能非光滑的非凸Lipschitz函数与一个凸正则项之和，且该函数仅能通过随机梯度或函数值进行访问。我们的目标是找到一个满足针对复合目标设计的Goldstein型平稳性条件的点。据我们所知，在一阶访问下，目前尚无该设置的已知预言机复杂度界，而零阶访问下已有的复杂度结果也是次优的。为解决这一问题，我们采用在线到非凸转换框架，该框架通过在线学习者来选择更新方向，并且已知在非复合问题上能达到最优速率。我们通过为学习者引入新的损失函数，将该框架扩展到我们的复合场景中……

    arXiv:2610.12328v1 Announce Type: new  Abstract: We consider stochastic nonsmooth nonconvex composite optimization, which includes several important problems such as constrained optimization and the regularized training of neural networks. The objective is the sum of a possibly nonsmooth nonconvex Lipschitz function and a convex regularizer, and the function is accessed through stochastic gradients or function values. The goal is to find a point that satisfies a Goldstein-type stationarity condition designed for composite objectives. To our knowledge, no oracle complexity bound for this setting is known under first-order access, and existing complexities under zeroth-order access are suboptimal. To handle this issue, we employ the framework of online-to-nonconvex conversion, which chooses update directions by an online learner and is known to achieve optimal rates for noncomposite problems. We extend the framework to our composite scenario by introducing new losses for the learner, whi
    
[^32]: SparseDecoding：面向解码感知的剪枝方法，实现准确且高效的大语言模型推理

    SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference

    [https://arxiv.org/abs/2610.12327](https://arxiv.org/abs/2610.12327)

    该论文提出SparseDecoding，一种解码感知的剪枝方法，通过在模型自生成的token序列上计算Hessian来消除自然序列与生成序列之间的分布偏移，从而在保证准确性的同时提升LLM解码阶段的推理效率。

    

    大语言模型（LLM）推理的解码阶段具有内存受限的特性，会带来显著的延迟。由Hessian矩阵指导的逐层免训练网络剪枝方法是解决该问题的重要方案，因为剪枝可以减少解码过程中从内存读取的非零参数数量。然而，此类方法中的典型方案通常使用预先收集的自然序列来计算Hessian矩阵，而模型在解码阶段接收的却是自身生成的token，这在两类序列之间造成了分布偏移。在自然序列上计算的Hessian与在生成序列上计算的Hessian并不相同。我们观察到，这种差异会导致生成过程中的激活分布偏离剪枝时所用的分布，从而进一步损害剪枝后模型的性能。此外，大多数现有的能够带来实际加速的LLM剪枝方法主要针对稀疏矩阵-矩阵乘法（SpMM）运算……（摘要原文在此处截断）

    arXiv:2610.12327v1 Announce Type: cross  Abstract: The memory-bound nature of the decoding stage of large language model (LLM) inference incurs significant latency. Layer-wise training-free network pruning approaches guided by the Hessian have been a prominent solution to this problem, as pruning reduces the number of nonzero parameters read from memory during decoding. Nevertheless, typical methods in this line compute the Hessian using pre-collected natural sequences, whereas the model is fed self-generated tokens during decoding, creating a distribution shift between the two sequences. The Hessian calculated on the natural sequence is different from that calculated on the generated sequence. We observe that this discrepancy causes the activation distribution during generation to deviate from that used for pruning, further hurting the pruned model performance. Moreover, most existing LLM pruning methods that bring actual speedup primarily target the sparse matrix-matrix (SpMM) multip
    
[^33]: 先验还是反馈？大语言模型在适配神经算子时究竟使用什么

    Prior or Feedback? What an LLM Uses When Adapting Neural Operators

    [https://arxiv.org/abs/2610.12325](https://arxiv.org/abs/2610.12325)

    该论文通过受控干预实验发现，LLM在有限预算下适配神经算子时同时依赖有用的先验（首个配置即接近随机搜索池顶端）和实验反馈来选择微调配置，在几乎所有匹配比较中优于随机搜索和贝叶斯优化。

    

    LLM科学智能体是仅依赖其初始任务上下文，还是会根据实验反馈调整其决策？我们在神经算子适配任务中研究这一问题，即大语言模型（LLM）在有限的试验预算下选择微调配置。在偏微分方程（PDE）族内部及族间的迁移任务中，LLM在几乎所有匹配比较中都取得了比随机搜索和贝叶斯优化更低的留出测试nRMSE。仅凭最终性能无法分辨到底发生了什么，因此我们通过受控干预来验证每一项归因。在观察到任何验证分数之前，LLM的首次配置就已在相应的随机搜索池中排名接近顶端，这表明其存在一种有用的初始偏差。一项互补的冷启动干预实验表明，所选的基础学习率会随PDE描述的变化而改变。一旦反馈变得可用，重新分配……（原文在此处截断）

    arXiv:2610.12325v1 Announce Type: new  Abstract: Do LLM scientific agents rely only on their initial task context, or do they adapt their decisions in response to experimental feedback? We study this question in neural operator adaptation, where a large language model (LLM) selects fine-tuning configurations under a limited trial budget. Across transfers within and between partial differential equation (PDE) families, the LLM achieves lower held-out test nRMSE than random search and Bayesian optimisation in nearly every matched comparison. Endpoint performance alone cannot distinguish what happens, so we verify each attribution with controlled interventions. Before observing any validation score, the LLM's first configuration already ranks near the top of the corresponding random-search pool, indicating a useful initial bias. A complementary cold-start intervention shows that the selected base learning rate shifts with the PDE description. Once feedback becomes available, reassigning v
    
[^34]: 公共物品困境中多智能体学习驱动的空间模式形成

    Spatial Pattern Formation from Multi-Agent Learning in Public Goods Dilemmas

    [https://arxiv.org/abs/2610.12321](https://arxiv.org/abs/2610.12321)

    该研究首次展示在公共物品困境中，合作者与背叛者通过多智能体Q学习自主习得移动策略即可自发涌现空间模式（资源峰值周围的集群与移动条带），并揭示学习率的不对称配置（合作者快、背叛者慢）导致最大集体福利损失，且部分模式反映训练历史而非稳态结果。

    

    空间公共物品模型表明，规定的向资源丰富位置移动可以产生空间模式。我们探讨当智能体自主学习移动位置时，这种模式如何涌现，以及学习率如何影响其对集体福利的作用。固定规模的合作者与背叛者种群分别使用表格Q学习和局部观测独立学习移动策略。合作者的学习会在资源峰值周围生成集群，而双方的共同适应则会改变这些集群的强度和运动方式。在固定训练预算下，当合作者以高学习率学习而背叛者以低学习率学习时，会出现最大的集体福利损失。在该区域的一部分中，学习到的策略还会产生由共享方向偏好所支持的移动条带。支持移动的条件会随进一步训练而改变，因此这些模式反映的是训练历史，而非既定的渐近结果。

    arXiv:2610.12321v1 Announce Type: cross  Abstract: Spatial public goods models show that prescribed movement toward richer locations can generate spatial patterns. We ask how such patterns emerge when agents learn where to move and how learning rates shape their consequences for collective welfare. Fixed populations of cooperators and defectors independently learn movement policies using tabular Q-learning and local observations. Cooperator learning generates clusters around resource peaks, while co-adaptation changes their strength and motion. At a fixed training budget, the largest welfare losses occur when cooperators learn at high rates and defectors at low rates. In part of this regime, learned policies also generate traveling bands supported by a shared directional preference. The conditions supporting travel change with further training, so these patterns reflect training history rather than an established asymptotic outcome. Across the tested learning-rate conditions with coope
    
[^35]: ARGUS解锁调控基因组：一个用于解读单核苷酸变异的证据约束型智能体框架

    Unlocking the Regulatory Genome by ARGUS: An Evidence-Constrained Agentic Framework for Interpreting Single Nucleotide Variants

    [https://arxiv.org/abs/2610.12281](https://arxiv.org/abs/2610.12281)

    ARGUS提出了一种严格分离确定性生物计算与大语言模型推理的证据约束型智能体框架，通过假设导向的研究循环和458个基于DNABERT的转录因子结合模型，可靠地解读非编码调控区单核苷酸变异的功能影响，避免了LLM的幻觉问题。

    

    全基因组关联研究中超过90%的疾病相关变异位于非编码调控区域，然而对这些变异的功能解读仍是基因组医学中一个核心的开放问题。被提示解读此类变异的大语言模型经常产生转录因子（TF）结合变化的幻觉、捏造实验支持，并对统计上可忽略的信号赋予生物学意义。我们提出了ARGUS（面向不确定性感知科学家的智能体调控基因组学），它严格地将确定性生物计算与大语言模型介导的推理相分离。ARGUS将458个基于DNABERT的转录因子结合模型封装在一个假设导向的研究循环中：规划器根据当前不确定性选择证据来源，验证器确定性地解释每个观察结果，中间结果会动态改变研究路径。在8q24癌症风险位点的变异rs6983267上，同一个规划器

    arXiv:2610.12281v1 Announce Type: cross  Abstract: Over 90% of disease-associated variants from genome-wide association studies fall in noncoding regulatory regions, yet their functional interpretation remains a central open problem in genomic medicine. Large language models prompted to interpret such variants routinely hallucinate transcription factor (TF) binding changes, fabricate experimental support, and assign biological significance to statistically negligible signals. We present ARGUS (Agentic Regulatory Genomics for an Uncertainty-aware Scientist), which strictly separates deterministic biological computation from LLM-mediated reasoning. ARGUS wraps 458 DNABERT-based TF binding models in a hypothesis-directed investigation loop where a planner selects evidence sources based on current uncertainty, a verifier deterministically interprets each observation, and intermediate results change the investigation path. On variant rs6983267 at the 8q24 cancer risk locus, the same planner
    
[^36]: AdaptLSTM：面向分布漂移下云工作负载预测的高效自适应在线学习

    AdaptLSTM: Efficient Adaptive Online Learning for Cloud Workload Forecasting under Distribution Drift

    [https://arxiv.org/abs/2610.12265](https://arxiv.org/abs/2610.12265)

    AdaptLSTM 通过验证校准阈值检测分布漂移并进行选择性针对性更新，以约 20% 的计算成本实现接近朴素在线学习的预测精度，大幅提升云工作负载预测的效率。

    

    准确的工作负载预测对于网络规模云服务的弹性资源供给至关重要，然而由热门内容、产品发布和用户行为引发的分布漂移会迅速降低离线训练模型的性能。朴素的在线学习虽然能恢复预测精度，但会带来难以承受的每步计算成本。我们提出了 AdaptLSTM，一种自适应在线学习框架，它通过验证集校准的阈值检测漂移，并施加选择性的、有针对性的更新。在阿里巴巴机器轨迹数据集上，AdaptLSTM 以 20% 的成本恢复了朴素在线学习 54% 的性能提升（2.7 倍效率，10 个随机种子下 p=0.002）。在波动性更强的容器轨迹数据集上，它以 20% 的成本实现了 96% 的性能提升（4.8 倍效率，相比静态模型 MAE 降低 75%）。与经典漂移检测器（ADWIN、DDM、Page-Hinkley）在回归尺度的误差流上无法触发不同，AdaptLSTM 在 301 步中触发了 42 次，并优于同等计算预算的基线方法。墙上时钟时间（原文在此处截断）。

    arXiv:2610.12265v1 Announce Type: new  Abstract: Accurate workload forecasting is critical for elastic resource provisioning in web-scale cloud services, where distribution shifts driven by viral content, product launches, and user behavior degrade offline-trained models rapidly. Naive online learning recovers accuracy but incurs prohibitive per-step compute cost. We propose AdaptLSTM, an adaptive online framework that detects drift via validation-calibrated thresholds and applies selective, targeted updates. On the Alibaba Machine Trace, AdaptLSTM recovers 54\% of Naive Online's improvement at 20\% cost ($2.7\times$ efficiency, $p=0.002$ over 10 seeds). On the more volatile Container Trace, it achieves 96\% at 20\% cost ($4.8\times$ efficiency, $+75\%$ MAE reduction over Static). Unlike classical drift detectors (ADWIN, DDM, Page-Hinkley) which fail to trigger on regression-scale error streams, AdaptLSTM fires 42 times over 301 steps and outperforms matched-budget baselines. Wall-cloc
    
[^37]: 先分批再提升：大规模图上的可扩展拓扑深度学习

    Batch Before You Lift: Scalable Topological Deep Learning on Large Graphs

    [https://arxiv.org/abs/2610.12247](https://arxiv.org/abs/2610.12247)

    本文提出Cluster-TNN框架，通过先对大图进行划分、再在小批量内局部执行图提升操作，避免了全域物化的计算瓶颈，使拓扑深度学习能够扩展到大规模稠密图上。

    

    拓扑深度学习将基于图的学习扩展到更高阶的领域，例如超图、胞腔复形和单纯复形。这些领域通常通过图提升的过程从输入图中的模式构建而来。全域训练方式会在模型执行之前构建并存储完整的提升表示。在像Reddit这样规模大且稠密的数据集（23.3万节点和5730万条边）上，这种全局物化会成为严重的计算瓶颈，常常导致训练无法进行。为了解决这一限制，我们提出了Cluster-TNN，这是一个领域无关的框架，通过局部提升来避免这一瓶颈。在预处理阶段对输入图进行划分之后，Cluster-TNN在运行时动态采样若干节点簇组，重建其诱导子图以形成小批量，并在每个小批量内应用所选的提升操作。保留采样节点之间的所有边可以保……（原文此处截断）

    arXiv:2610.12247v1 Announce Type: cross  Abstract: Topological Deep Learning extends graph-based learning to higher-order domains, such as hypergraphs, cellular, and simplicial complexes. These domains are typically constructed from patterns in an input graph through a process of graph lifting. Full-domain training constructs and stores the complete lifted representation before model execution. On large and dense datasets like Reddit (233k nodes and 57.3M edges), this global materialization becomes a severe computational bottleneck, often rendering training infeasible. To address this limitation, we introduce Cluster-TNN, a domain-agnostic framework that avoids this bottleneck by lifting locally instead. After partitioning the input graph during preprocessing, at runtime Cluster-TNN dynamically samples groups of node clusters, reconstructs their induced subgraphs to form mini-batches, and applies the chosen lifting within each mini-batch. Retaining all edges among sampled nodes preserv
    
[^38]: RIFT：基于树的相对隔离异常检测方法

    RIFT: Relative Isolation From Trees For Anomaly Detection

    [https://arxiv.org/abs/2610.12244](https://arxiv.org/abs/2610.12244)

    该论文提出了RIFT——一种基于最小生成树的确定性无参数异常检测方法，其一维评分能精确恢复孤立森林的闭式极限，在高维数据上对密度变化和聚类异常具有鲁棒性，且准确率与孤立森林相当，同时避免了其轴平行伪影问题。

    

    孤立森林（Isolation Forest, IF）是一种广泛使用的无监督异常检测基线方法。近期研究为其中一维数据的无限森林极限提供了闭式表达式。受该公式几何解释的启发，我们提出了RIFT（基于树的相对隔离，Relative Isolation From Trees），这是一种确定性的异常检测方法，它通过生成最小生成树，并根据从每个点观察到的树边的表观大小之和来对该点进行评分。对于一维数据，RIFT分数能够精确地恢复孤立森林的闭式极限。在更高维度中，它提供了一种无需参数的泛化方法，具有确定性、对密度变化和聚类异常具有鲁棒性，并避免了孤立森林的轴平行伪影问题。我们进一步提出了一种面向大型数据集的集成变体。在合成数据和ADBench基准上的实验表明，该方法的准确率与孤立森林相当，而集成变体则表现出显著的性能提升（摘要在此处截断）。

    arXiv:2610.12244v1 Announce Type: new  Abstract: Isolation Forest (IF) is a widely used baseline for unsupervised anomaly detection. Recent studies provide a closed-form expression for the infinite-forest limit for one-dimensional data. Inspired by the geometric interpretation of this formula, we introduce RIFT (Relative Isolation From Trees), a deterministic anomaly detection method that generates the minimum spanning tree and scores each point by the sum of the apparent sizes of tree edges as viewed from that point. For one-dimensional data, the RIFT score recovers the closed-form IF limit exactly. In higher dimensions, it provides a parameter-free generalization that is deterministic, robust to varying density and clustered anomalies and avoids the axis-parallel artifacts of IF. We further propose an ensemble variant for large datasets. Experiments on synthetic data and the ADBench benchmark demonstrate that the accuracy is comparable to IF, while the ensemble variant exhibits signi
    
[^39]: AdaCast：面向自适应时间序列预测的条件参数生成

    AdaCast: Conditional Parameter Generation for Adaptive Time Series Forecasting

    [https://arxiv.org/abs/2610.12240](https://arxiv.org/abs/2610.12240)

    AdaCast提出了一种条件参数生成框架，利用生成器为冻结的预训练时间序列基础模型动态生成针对每条输入序列的低秩参数更新，实现输入级的自适应预测，在六个基准测试中超越了静态适配方法。

    

    时间序列基础模型（TSFMs）已在多个领域展现出强大的预测性能。然而，现有的适配方法大多是静态的。目前的一体化方法仅学习一组数据集级别的参数更新，并将同一个适配后的模型应用于所有输入。因此，这些方法无法根据每条输入时间序列的时间模式、季节性和动态特性来调整模型参数，限制了其为异构输入生成定制化预测的能力。为解决这一局限，我们提出了AdaCast，一个用于时间序列预测的条件参数生成框架。AdaCast使用一个生成器为冻结的预训练时间序列基础模型生成针对特定输入的低秩参数更新。这些参数更新在训练和推理阶段均能使模型适配每条输入。在六个公开基准测试中，AdaCast在域内预测任务上持续优于静态适配基线，并提升了零样本泛化能力。

    arXiv:2610.12240v1 Announce Type: cross  Abstract: Time-series foundation models (TSFMs) have achieved strong forecasting performance across domains. However, most adaptation methods remain static. Existing all-in-one methods learn a single set of dataset-level parameter updates and apply the same adapted model to every input. As a result, they cannot adapt the model parameters to the temporal patterns, seasonality and dynamics of each input time series. This limits their ability to produce forecasts that are tailored to heterogeneous inputs. To address this limitation, we propose AdaCast, a conditional parameter generation framework for time-series forecasting. AdaCast uses a generator to produce input-specific low-rank parameter updates for a frozen pretrained TSFM. These updates adapt the model to each input during both training and inference. Across six public benchmarks, AdaCast consistently outperforms static adaptation baseline in in-domain forecasting and improves zero-shot gen
    
[^40]: 面向未来训练：时间序列预测中测试时自适应方法的延迟感知审计

    Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting

    [https://arxiv.org/abs/2610.12232](https://arxiv.org/abs/2610.12232)

    该论文构建了无泄漏的延迟感知审计框架，在真实延迟标签条件下评估了四种主流时间序列预测测试时自适应方法，并提出了无需超参数调节的递归最小二乘滤波器组作为强基线，揭示了延迟标签下各方法表现的不对称性。

    

    时间序列预测的测试时自适应（TTA）方法利用传入的真实标签来更新已部署的模型或其周围的小型适配器。但一个 H 步预测的标签只有在 H 步之后才存在，而真实的数据管道还会引入额外的延迟。我们构建了一个无泄漏的评估框架，其中预测起点 s 的标签仅在步骤 s+d（d ≥ H）时才被释放用于更新，并将该规则强制应用于四种近期 TTA 方法（TAFAS、COSA、PETSA 和 DynaTTA）的已发布代码中，这些方法在各自的骨干网络和检查点上运行，覆盖五个基准数据集（ETTm1、ETTh2、Weather、Electricity 和 Traffic）。作为参照，我们还加入了两种闭式修正器：一种是由逐坐标中位数组合的递归最小二乘（RLS）滤波器组，无需可调超参数，在 7 通道数据流上每步仅需 56 微秒；另一种是 ELF 风格的线性修正器。在因果延迟标签下，结果呈现出不对称性。在 ETTm1 上，所有被审计的方法（摘要在此处截断）……

    arXiv:2610.12232v1 Announce Type: new  Abstract: Test-time adaptation (TTA) methods for time-series forecasting update a deployed model, or a small adapter around it, from incoming ground truth. But the label of an $H$-step forecast exists only $H$ steps later, and real data pipelines add further delay. We build a leakage-free harness in which the label of forecast origin $s$ is released for updates only at step $s+d$ with $d \ge H$, and enforce this rule inside the released code of four recent TTA methods (TAFAS, COSA, PETSA and DynaTTA), run on their own backbones and checkpoints across five benchmarks (ETTm1, ETTh2, Weather, Electricity and Traffic). As references we add two closed-form correctors: a bank of recursive least squares (RLS) filters combined by a per-coordinate median, with no tunable hyperparameters and 56 microseconds per step on the 7-channel streams, and an ELF-style linear corrector. Under causal delayed labels the picture is asymmetric. On ETTm1 every audited meth
    
[^41]: ISBO：基于INLA-SPDE方法与对数高斯Cox过程模型的可扩展时空贝叶斯优化

    ISBO: Scalable Spatio-Temporal Bayesian Optimization with Log Gaussian Cox Process Models via the INLA-SPDE Approach

    [https://arxiv.org/abs/2610.12213](https://arxiv.org/abs/2610.12213)

    该论文提出了首个面向时空数据的可扩展贝叶斯优化框架ISBO，通过对数高斯Cox过程建模与INLA-SPDE推断方法，能够以最少的评估次数稳定定位高强度区域及潜在强度峰值。

    

    贝叶斯优化（BO）是一种高效优化昂贵黑盒目标函数的流行方法。然而，使用标准高斯过程的贝叶斯优化并不适合时空问题空间中常用的双重随机Cox过程。我们提出了INLA-SPDE时空贝叶斯优化（ISBO）：首个针对时空数据的可扩展贝叶斯优化框架，该框架使用对数高斯Cox过程（LGCP）对对数强度进行建模，并通过积分嵌套拉普拉斯近似与随机偏微分方程（INLA-SPDE）方法进行推断。在网格上使用Matern场可生成稀疏高斯马尔可夫随机场，使INLA在整个序贯优化过程中提供快速且准确的后验推断。ISBO能够以最少的评估次数稳定地定位高强度区域以及潜在强度的峰值。带有掩蔽机制的时变上置信界采集函数可避免重复访问……

    arXiv:2610.12213v1 Announce Type: cross  Abstract: Bayesian Optimization (BO) is a popular method for efficiently optimizing expensive black-box objectives. However, BO utilizing standard Gaussian Processes is ill-suited for doubly stochastic Cox Processes that are often used in spatio-temporal problem spaces. We introduce INLA-SPDE Spatio-Temporal Bayesian Optimization (ISBO): the first scalable BO framework for spatio-temporal data, that models the log-intensity with a Log-Gaussian Cox Process(LGCP) and performs inference via Integrated Nested Laplace Approximation and Stochastic Partial Differential Equations (INLA-SPDE) approach. Using a Matern field on meshes yields a sparse Gaussian Markov Random Field, where INLA provides fast and accurate posterior inference throughout sequential optimization. ISBO stably locates high-intensity regions and the peak of the latent intensity with minimal evaluations. A time-varying Upper Confidence Bound acquisition with masking avoids revisits, w
    
[^42]: 基于迁移的验证：精确信息前沿及其在调用次数上的代价

    Verification with Transfer: Exact Information Frontiers and Their Price in Calls

    [https://arxiv.org/abs/2610.12211](https://arxiv.org/abs/2610.12211)

    本文从信息论角度为“借助相关源任务（迁移）来降低验证成本”这一策略精确定价：所需最小因果信息由列表率失真函数刻画，并把所有源调用放在首次验证之前不会破坏调用的硬上限，但交错安排可以无界地节省期望调用次数。

    

    一个只接受或拒绝完整答案的验证器所能揭示的信息很少：在 $k$ 比特答案的均匀先验下，要达到零错误需要 $2^k-1$ 次验证。通常的补救方法是先解决相关的源任务，要么像课程学习那样全部先做，要么与验证交错进行。我们从信息和调用次数两个角度为这种补救方法定价。对于精确验证器，任何由源调用与 $n$ 次验证交错组成、并以概率 $s$ 成功的方案所需的最小因果信息，是一个列表率失真函数，且该值可由在任何验证之前的一次观察达到。它给出了二进制源调用期望次数的下界：设计良好的源在唯一答案情形下可在 $1+\log_25$ 次调用内达到该界，一般情况下可在一个对数项内达到，而任何附加常数都不足以做到。在精确验证器和固定源的设定下，将所有调用移到第一次验证之前不会破坏任何调用的硬上限，尽管交错安排可以无界地节省期望调用次数。

    arXiv:2610.12211v1 Announce Type: new  Abstract: A verifier that accepts or rejects whole answers reveals little: under a flat prior over $k$-bit answers, zero error needs $2^k-1$ verifications. The usual remedy is to solve related source tasks, either all first, as a curriculum does, or interleaved with verification. We price this remedy in information and in calls. With an exact verifier, the least causal information that any interleaving of source calls and $n$ verifications needs to succeed with probability $s$ is a list rate-distortion function, attained by one observation before any verification. It lower-bounds the expected number of binary source calls, which designed sources meet within $1+\log_25$ calls for unique answers and within a logarithmic term in general, where no additive constant suffices. With an exact verifier and fixed sources, moving every call before the first verification preserves all hard caps on calls, although interleaving can save unboundedly many expecte
    
[^43]: SciTBERT：一系列用于科学与技术语言处理的时间一致性语言模型

    SciTBERT: A family of chronologically consistent language models for scientific and technological language processing

    [https://arxiv.org/abs/2610.12207](https://arxiv.org/abs/2610.12207)

    提出了SciTBERT——一系列时间一致的BERT衍生语言模型，训练数据截止日期覆盖2013至2025年每年，通过消除前瞻偏差和领域偏差，使模型适用于研究科学与技术随时间演变的特性。

    

    预训练Transformer模型正被越来越多地用于研究科学技术进步。针对论文或专利文本微调的编码器，在科学与技术领域的下游分类、回归和相似度任务上优于通用模型。然而，由于这些预训练模型固有的“前瞻偏差”和领域偏差，它们在研究科学、技术及其交叉领域的时间相关或历史档案特性方面的适用性受到限制。这些局限源于训练语料库在时间分布和文本来源分布上缺乏约束。我们提出了SciTBERT：这是一个时间一致的、基于BERT的语言模型家族，其训练数据来自科学论文、专利和高质量教育类网络文本，训练数据截止日期覆盖2013年至2025年的每一年。我们还以时间一致的方式对这些模型进行后训练，使用论文和专利……

    arXiv:2610.12207v1 Announce Type: new  Abstract: Pre-trained transformer models are increasingly being used to study scientific and technological progress. Encoders tuned to paper or patent text outperform general-purpose models on downstream classification, regression, and proximity tasks within science and technology. However, the applicability of these models for studying time-dependent or archival properties of science, technology, and their interface is limited due to lookahead and domain biases inherent to these pre-trained models. These limitations arise from training on corpora with unconstrained chronological and text source distributions. We introduce SciTBERT: a family of chronologically consistent BERT-derived language models trained on text from scientific papers, patents, and high-quality educational web text with training data cutoff dates spanning each year between 2013 and 2025. We also post-train these models in a chronologically-consistent manner using paper and pate
    
[^44]: 基于扩散积分分数的最快变化检测

    Quickest Change Detection with Diffusion-Integrated Scores

    [https://arxiv.org/abs/2610.12200](https://arxiv.org/abs/2610.12200)

    提出了一种无需训练的扩散积分分数CUSUM（DI-SCUSUM）最快变化检测器，通过向样本添加高斯噪声并精确计算Hyvärinen分数来近似对数似然比，实现了指数级误报控制和有保证的一阶检测延迟界。

    

    经典的CUSUM方法依赖于底层分布的对数似然比，而这一般无法仅凭有限的变前与变后样本计算得到。我们提出了扩散积分分数CUSUM（DI-SCUSUM），这是一种无需训练的检测器。我们向样本添加高斯噪声以形成两个平滑的密度估计，并精确计算它们的Hyvärinen分数，而无需训练分数网络。对于每个新到的观测值，我们采样一个扩散时间，对观测值进行扰动，并将重要性加权的分数差作为DI-SCUSUM递归中的增量。在观测值遵循固定经验分布的假设下，变后的平均增量与从平滑后的变后经验分布到平滑后的变前经验分布的Kullback-Leibler（KL）散度成正比。我们建立了指数级的误报缩放特性以及一阶检测延迟界，对于固定阈值和增量……

    arXiv:2610.12200v1 Announce Type: cross  Abstract: Classical CUSUM relies on the log-likelihood ratio of the underlying distributions, which cannot generally be computed from finite pre- and post-change samples alone. We propose diffusion-integrated score CUSUM (DI-SCUSUM), a training-free detector. We add Gaussian noise to the samples to form two smooth density estimates and calculate their Hyv\"arinen scores exactly, without training a score network. For each incoming observation, we sample a diffusion time, perturb the observation, and use the importance-weighted score difference as an increment in the DI-SCUSUM recursion. Under the assumption that observations follow the fixed empirical distributions, the post-change mean increment is proportional to the Kullback-Leibler (KL) divergence from the smoothed post-change to the smoothed pre-change empirical distribution. We establish exponential false-alarm scaling and a first-order delay bound that, for a fixed threshold and increment 
    
[^45]: DataSense-Bench：迈向AI科学家的第一步

    DataSense-Bench: The First Step Toward an AI Scientist

    [https://arxiv.org/abs/2610.12190](https://arxiv.org/abs/2610.12190)

    该论文提出首个评估AI“数据感”能力的基准DataSense-Bench，让AI智能体在无法训练模型、无法访问评估任务的情况下为LLM微调选择和排序训练数据子集，再通过实际微调后的性能来检验前沿AI模型能否可靠地为训练选择正确的数据。

    

    随着关于递归自我改进（RSI）和通用人工智能（AGI）的宣称不断增多，我们提出一个简单的问题：前沿AI模型是否具备“数据感”，即它们能否可靠地为训练选择合适的数据？我们提出DataSense-Bench，通过机器学习中数据选择与性能预测这一基础问题来研究这种能力。我们要求AI智能体选择并对候选训练子集进行排序，这些子集将用于微调一个小型LLM模型。智能体可以检查数据、编写并执行分析代码、运行模型前向推理，但不能训练模型或访问实际的评估任务。随后，我们在每个选定的子集上对基础模型进行微调，并在标准化协议下评估其训练后性能。我们在终端问题求解和工具使用场景中构建该基准，从OpenThoughts-Agent和EnvScaler中选取轨迹数据作为候选训练子集，并在（原文在此处被截断）

    arXiv:2610.12190v1 Announce Type: new  Abstract: As claims about recursive self-improvement (RSI) and artificial general intelligence (AGI) proliferate, we ask a simple question: do frontier AI models have a sense of data, i.e., can they reliably select the right data for training? We introduce DataSense-Bench to study this capability through the fundamental problem of data selection and performance forecasting in machine learning. We ask AI agents to select and rank candidate training subsets that can be used to fine-tune a small LLM model. Agents are allowed to inspect the data, write and execute analysis code, and run model forward passes, but can not train the model or access the actual evaluation tasks. We then fine-tune the base model on each selected subset and evaluate its post-training performance under a standardized protocol. We instantiate the benchmark in terminal problem solving and tool use, selecting trajectories from OpenThoughts-Agent and EnvScaler and evaluating on T
    
[^46]: 简单天气评分：基于分布扩散的高效端到端临近预报

    Just Weather Scoring: Efficient End-to-end Nowcasting with Distributional Diffusion

    [https://arxiv.org/abs/2610.12189](https://arxiv.org/abs/2610.12189)

    提出 JWS——一种单阶段端到端扩散模型，通过在雷达空间直接预报、掩码异步扩散与评分规则目标实现少步生成，在大幅降低推理成本的同时达到了最先进的概率性降水临近预报效果。

    

    生成式扩散模型非常适合概率性降水临近预报，但现有方法通常依赖于单独训练的压缩或确定性预报组件，并且由于迭代去噪而在推理时成本高昂。我们提出了简单天气评分（Just Weather Scoring, JWS），这是一种单阶段、端到端的扩散模型，通过直接在雷达空间进行预报并支持少步生成，同时解决了这两个问题。雷达空间建模极大地简化了训练和推理过程，并消除了有损压缩带来的不确定性。JWS 将掩码异步扩散与简单的评分规则目标相结合，前者是一种时间步采样方案，在使扩散训练适应高维时空数据的同时保持干净的上下文信息，后者则使训练目标与概率预报保持一致并实现了少步生成。在 SEVIR 和 MeteoNet 基准测试上，JWS 实现了最先进的概率预报性能。

    arXiv:2610.12189v1 Announce Type: new  Abstract: Generative diffusion models are well-suited for probabilistic precipitation nowcasting, but existing approaches often rely on separately trained compression or deterministic forecasting components and remain costly at inference due to iterative denoising. We introduce Just Weather Scoring (JWS), a single-stage, end-to-end diffusion model which addresses both issues by forecasting directly in radar space and enabling few-step generation. Radar-space modeling greatly simplifies training and inference and eliminates uncertainty arising from lossy compression. JWS combines Masked Asynchronous Diffusion, a timestep-sampling scheme that preserves clean context while adapting diffusion training to high-dimensional spatio-temporal data, with a simple scoring-rule objective that aligns training with probabilistic forecasting and unlocks few-step generation. On the SEVIR and MeteoNet benchmarks, JWS achieves state-of-the-art probabilistic forecast
    
[^47]: 深入探究智能体化黑盒优化：面向黑盒优化的LLM智能体基准测试

    A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization

    [https://arxiv.org/abs/2610.12183](https://arxiv.org/abs/2610.12183)

    提出了AgenticBBO-Bench，一个采用统一有限预算评估协议、覆盖合成函数、超参数优化、数据库调优、芯片设计和分子设计五个领域的跨领域智能体黑盒优化基准测试，并验证了智能体BBO在所有领域中均优于直接基于LLM的方法。

    

    黑盒优化（BBO）普遍存在于许多科学和工程问题中，这些问题中目标函数的评估既昂贵又有限。近期的大型语言模型（LLM）智能体通过结合任务语义、计算、优化工具和反馈驱动的决策，为解决BBO提供了一种新方法，由于与数学上严谨的工具相结合，展现出巨大潜力。然而，现有的智能体化BBO研究使用不同的任务领域和系统配置，导致其结果难以相互比较，且各个设计选择的独立影响难以分离。因此，我们提出了AgenticBBO-Bench，这是一个跨领域的智能体BBO基准测试，涵盖合成函数、超参数优化、数据库调优、芯片设计和分子设计五个领域，并采用统一的有限预算评估协议。在实验中，智能体BBO在所有五个领域中均取得了比直接基于LLM的方法更高的家族平均分数。

    arXiv:2610.12183v1 Announce Type: cross  Abstract: Black-box optimization (BBO) arises in many scientific and engineering problems where objective evaluations are expensive and limited. Recent large language model (LLM) agents offer a new way to approach BBO by combining task semantics, computation, optimization tools, and feedback-driven decision making, showing great potential due to the integration with mathematically rigorous tools. However, existing agentic BBO studies use different task domains and system configurations, making their results difficult to compare and the effects of individual design choices hard to isolate. We therefore introduce AgenticBBO-Bench, a cross-domain benchmark for agentic BBO spanning synthetic functions, hyperparameter optimization, database tuning, chip design, and molecular design under a unified finite-budget evaluation protocol. In our experiments, agentic BBO achieves higher family-averaged scores than direct LLM-based methods in all five domains
    
[^48]: 通过回望学习规划：用于训练推理模型的后见之明层级

    Learning to Plan by Looking Back: Hindsight Hierarchies for Training Reasoning Models

    [https://arxiv.org/abs/2610.12168](https://arxiv.org/abs/2610.12168)

    该论文提出了一种基于“后见之明”的自我改进循环，通过联合训练同一模型预测解题思路、从已知解答逆向推导思路、以及利用给定思路解决问题这三种能力，使模型即使面对超出自身能力的问题也能从提供的解答中提取有用思路来持续自我提升，并在Lean定理证明器中给出了具体实现。

    

    我们为推理模型引入了一个自我改进循环，其基于以下观察：即使问题的难度超出了模型当前的解题能力，额外提供的解答也可能使模型能够从事后明悟中提取有用的解题思路。我们将这一观察付诸实践：联合训练同一个模型，使其具备以下三种能力——仅从问题预测解题思路、从问题和已知解答逆向推导思路、以及利用给定的思路解决问题。该循环在两个步骤之间交替进行：从带有已提供解答的问题中逆向推导此类思路，以及将这些思路作为额外监督信号用于三种能力的联合训练。我们给出了该方法的形式化规范，以及在Lean定理证明器中用于交互式定理证明的具体实例化；实证评估仍留待未来工作完成。

    arXiv:2610.12168v1 Announce Type: new  Abstract: We introduce a self-improvement loop for reasoning models based on the following observation: Even when the difficulty of a problem exceeds the model's current solving abilities, an additionally supplied solution might enable the model to extract useful solution ideas in hindsight. We operationalize this by jointly training the same model to exhibit the following three capabilities: predicting solution ideas from problems alone, reverse-engineering ideas from problems and known solutions, and solving problems using provided ideas. The loop alternates between reverse engineering such ideas from problems with supplied solutions and using these ideas as additional supervision for joint training of all three capabilities. We give a formal specification of our method and a concrete instantiation for interactive theorem proving in the Lean theorem prover; empirical evaluation remains future work.
    
[^49]: 通用图异常检测真的需要真实世界的训练数据吗？

    Is Real-World Training Data Necessary for Generalist Graph Anomaly Detection?

    [https://arxiv.org/abs/2610.12167](https://arxiv.org/abs/2610.12167)

    本文提出AG-FORGE异常图自动合成框架，证明合成数据训练可实现与真实数据相当的通用图异常检测性能，并进一步提出拓扑-语义协同的TS-GGAD模型以释放更大模型容量。

    

    通用图异常检测（GAD）旨在构建一个基础模型，使其能够在任意未见过的图上检测异常，而无需重新训练或微调。充足的数据对基础模型训练至关重要，然而通用图异常检测仍面临数据短缺问题，因为真实世界的异常图十分稀缺，且收集和标注成本高昂。为填补这一空白，我们提出了AG-FORGE——一个用于自动合成异常图的异常图生成工厂，探索了合成数据驱动训练在通用图异常检测中的可行性。实验表明，合成数据可以达到与真实世界训练相当的性能，但由于现有方法的容量有限，无法进一步突破性能边界。为进一步释放模型容量以适应训练数据规模的扩大，我们开发了TS-GGAD，一个拓扑-语义协同的通用图异常检测模型，能够捕获互补的拓扑和语义异常证据……

    arXiv:2610.12167v1 Announce Type: new  Abstract: Generalist graph anomaly detection (GAD) aims to build a foundation model that detects anomalies on arbitrary unseen graphs without retraining or fine-tuning. Sufficient data are essential for foundation model training, yet generalist GAD still faces a data shortage, as real-world anomalous graphs are scarce and costly to collect and annotate. To fill this gap, we propose AG-FORGE, an Anomalous Graph generation Forge for automatic synthesis of anomalous graphs, exploring the feasibility of synthetic data-driven training for generalist GAD. Empirically, we find that synthetic data can achieve performance comparable to real-world training, but fail to push the performance boundary further due to the limited capacity of existing methods. To further unlock model capacity as training data scale up, we develop TS-GGAD, a Topology-Semantic coordinated Generalist GAD that captures complementary topological and semantic anomaly evidence, together
    
[^50]: 基于软社区结构的可扩展层次图生成

    Scalable Hierarchical Graph Generation via Soft Community Structure

    [https://arxiv.org/abs/2610.12163](https://arxiv.org/abs/2610.12163)

    Schema通过将参考图递归分解为软社区层次结构，并将生成过程拆分为属性合成、社区内边生成和社区间连接建模三个可独立训练的阶段，实现了大规模属性图的可扩展生成，且全程无需构建完整邻接矩阵。

    

    生成大型属性图需要重现拓扑结构、将属性与结构联合生成，并保持可扩展性。许多真实世界的图以单个大图的形式存在，因此生成模型必须从其拟合的唯一图上进行泛化，而无法获得独立样本。我们提出了Schema，它递归地将参考图分解为软社区的层次结构，为每个节点分配一个隶属分布。生成过程随之被拆分为三个独立训练的阶段：（1）以软隶属关系为条件合成节点属性；（2）从局部结构上下文生成社区内边；（3）通过桥接节点建模社区间连接，这些桥接节点的隶属概率质量分布在多个社区之上。任何阶段都不需要构建完整的邻接矩阵，每个阶段都在由社区规模界定的子图上运行。我们还引入了一个评估协议，涵盖……

    arXiv:2610.12163v1 Announce Type: cross  Abstract: Generating large attributed graphs requires reproducing the topology, generating attributes jointly with the structure, and remaining scalable. Many real-world graphs exist as a single large graph, so a generative model has to generalize from the one graph it is fit on, without independent samples. We present Schema, which recursively decomposes a reference graph into a hierarchy of soft communities, assigning each node a membership distribution. Generation is then split into three stages, each trained independently: (1) synthesizing node attributes conditioned on soft memberships, (2) generating intra-community edges from local structural context, and (3) modeling inter-community connections over bridge nodes whose membership mass is distributed across several communities. No stage forms the full adjacency matrix, and each stage operates on a subgraph bounded by the community size. We also introduce an evaluation protocol that covers 
    
[^51]: 组策略优化中KL正则化何时失效

    When KL Regularization Misfires in Group Policy Optimization

    [https://arxiv.org/abs/2610.12161](https://arxiv.org/abs/2610.12161)

    论文剖析了组策略优化中KL正则化与奖励相互作用的七种失效模式，并提出零和校准策略优化（ZCPO），借助条件KL度量的相对漂移校准组内奖励系数，从而提升优化效果。

    

    为什么移除参考策略KL正则化有时会改善组策略优化的效果？这促使我们研究参考策略信息应当如何进入组相对更新之中。我们分析了KL与奖励之间相互作用的七种潜在失效模式：奖励裁剪后、梯度抵消后以及整组奖励完全相同时出现的残余KL更新；KL随响应长度增长而增加及其相对贡献的不平衡；KL集中在少数token上；以及将k1项纳入奖励时引入的采样噪声。我们提出零和校准策略优化（ZCPO），利用条件KL度量的相对漂移来校准组内奖励系数，并将其整合到基础代理目标中。数学推理实验与消融研究支持了该设计在本文设定中的有效性。

    arXiv:2610.12161v1 Announce Type: cross  Abstract: Why does removing reference-policy KL regularization sometimes improve group policy optimization? This motivates studying how reference-policy information should enter group-relative updates. We analyze seven potential failure modes in the interactions between KL and rewards: residual KL updates after reward clipping, after gradient cancellation, and in groups with identical rewards; KL growth with response length and an imbalance in its relative contribution; KL concentration on a small number of tokens; and sampling noise when k1 is incorporated into rewards. We propose Zero-Sum Calibrated Policy Optimization (ZCPO), which uses relative drift measured by conditional KL to calibrate within-group reward coefficients and integrates them into the base surrogate. Mathematical reasoning experiments and ablations support this design's effectiveness in our settings.
    
[^52]: 面向具有随机硬约束的对抗性马尔可夫决策过程的最优后悔

    Toward Optimal Regret in Adversarial MDPs with Stochastic Hard Constraints

    [https://arxiv.org/abs/2610.12153](https://arxiv.org/abs/2610.12153)

    提出MA-OPS算法，通过乐观搜索Slater余量与悲观评估所选策略相结合，在随机硬约束的对抗性MDP中实现了关于可行性余量的最优后悔界。

    

    我们研究了在随机硬约束下具有对抗性损失的情景式约束马尔可夫决策过程。具体而言，从一个已知的、具有余量 d 的严格可行策略出发，我们寻求在满足每个回合期望成本约束的同时获得最优后悔。在此设定中，Stradi等人（2025）证明了一种精心设计的混合规则可以达到 Õ(√T/min{d,d²}) 量级的后悔。有趣的是，他们还为同一设定提供了 Ω(√T/ρ) 量级的下界，其中 ρ 是离线问题的Slater余量，该余量可能远大于 d。在本工作中，我们基于他们的方法，获得了关于这些余量的最优后悔依赖关系。具体而言，我们提出了MA-OPS算法，该算法将对Slater余量的乐观搜索与对所选策略的悲观评估相结合，以安全地学习到具有较大可行性余量的策略。

    arXiv:2610.12153v1 Announce Type: new  Abstract: We study episodic constrained Markov decision processes with adversarial losses under stochastic hard constraints. Specifically, starting from a known strictly feasible policy with margin $d$, we seek to obtain optimal regret while satisfying the expected cost constraints in every episode. In this setting, Stradi et al. (2025) show that a carefully designed mixing rule attains regret of order $\widetilde{\mathcal{O}}(\sqrt{T}/\min\{d,d^2\})$. Interestingly, they also provide a lower bound of order $\Omega(\sqrt{T}/\rho)$ for the same setting, where $\rho$ is the Slater margin of the offline problem and can be much larger than $d$. In this work, we build on their approach to obtain optimal regret dependence on these margins. Specifically, we propose MA-OPS, an algorithm that combines an optimistic search for the Slater margin with a pessimistic evaluation of the selected policies to safely learn a policy with a large feasibility margin. T
    
[^53]: 状态保持约束下的贝叶斯优化

    Bayesian Optimisation under State-Preservation Constraints

    [https://arxiv.org/abs/2610.12150](https://arxiv.org/abs/2610.12150)

    该论文提出一种通过线性化预计算将状态空间中的状态保持约束拉回到设计空间并形成椭球体的方法，从而高效处理可行集高度各向异性的贝叶斯优化问题，并在托卡马克偏滤器优化这一关键应用上得到端到端验证。

    

    在许多工程设计问题中，目标函数和约束依赖于状态：即由设计参数确定的偏微分方程（PDE）的解。我们考虑在改进设计的同时，使选定的状态观测量保持在可信值附近，我们将其称为状态保持约束。带约束的贝叶斯优化通过学习到的可行性模型来处理此类约束，但难以应对该问题中高度各向异性的可行集。我们的核心思想是预先计算出一组控制量，使其线性化的约束响应保持在容差范围内，从而将状态空间中的约束拉回到设计空间中。这种线性化定义了一个椭球体，我们可以从中高效地抽取大量分布良好的候选点。底层的线性响应映射会在线上不断精化，椭球体也随之重建。我们在关键应用上对该方法进行了端到端的演示——即在等离子体边界保持约束下的托卡马克偏滤器优化。

    arXiv:2610.12150v1 Announce Type: new  Abstract: In many engineering design problems, the objective and constraints depend on the state: the solution of a PDE determined by the design parameters. We consider improving a design while holding selected state observables near trusted values, which we call state preservation constraints. Constrained Bayesian optimisation handles these with a learnt feasibility model, but struggles with this problem's highly anisotropic feasible set. Our central idea is to pre-compute the set of controls whose linearised constraint response stays within tolerance, thereby pulling back the state-space constraint into design space. This linearisation defines an ellipsoid from which we can efficiently draw a large number of well-spread candidates. The underlying linear response map is refined online, and the ellipsoid is rebuilt accordingly. We demonstrate the method end-to-end on our key application - Tokamak divertor optimisation under plasma-boundary preserv
    
[^54]: 面向金融时间序列预测的量子神经网络配置大规模基准测试

    Large-Scale Benchmarking of Quantum Neural Network Configurations for Financial Time Series Forecasting

    [https://arxiv.org/abs/2610.12148](https://arxiv.org/abs/2610.12148)

    该论文通过大规模网格搜索构建并评估了1,368种量子神经网络配置在金融时间序列预测（以GBP/USD汇率为案例）中的表现，系统揭示了编码方法、拟设设计、量子比特数等组件选择对预测精度、计算成本和收敛行为的影响。

    

    量子机器学习，特别是量子神经网络，是具有日益增长潜力的前沿领域。尽管针对量子神经网络配置的系统比较主要在分类任务中已有探索，但相对而言，回归问题——尤其是金融时间序列预测——受到的关注较少。本研究以英镑/美元（GBP/USD）现货汇率为案例，对用于金融时间序列预测的量子神经网络组件配置进行了大规模系统性对比评估。通过对编码方法、拟设设计、量子比特数量、层深度和代价函数进行网格搜索，共得到1,368种不同的模型配置，并对每种配置在预测精度、计算成本和收敛行为方面进行评估。研究结果揭示了方法选择如何影响性能的独特见解，例如门的选择与排列对模型成功更为关键。

    arXiv:2610.12148v1 Announce Type: new  Abstract: Quantum machine learning, and quantum neural networks (QNNs) in particular, are advancing fields with growing potential. Although systematic comparisons of QNN configurations have been explored primarily for classification tasks, comparatively little attention has been given to regression problems, particularly financial time series forecasting. This study presents a large-scale systematic comparative evaluation of QNN component configurations for financial time series forecasting, using the GBP/USD spot exchange rate as a case study. A grid search across encoding methods, ansatz designs, qubit counts, layer depths, and cost functions yields 1,368 distinct model configurations, each evaluated in terms of prediction accuracy, computational cost, and convergence behaviour. The results reveal unique insights into how the choice of methods influences performance, such as that gate selection and arrangement are more critical to model success 
    
[^55]: 海报：LLM蒸馏推断的初步研究

    Poster: A Preliminary Study of LLM Distillation Inference

    [https://arxiv.org/abs/2610.12137](https://arxiv.org/abs/2610.12137)

    该论文提出一种基于假设检验与影子模型的蒸馏推断方法，通过比较可疑模型对教师推理输出的预测得分并转换为校准p值，能够有效判定模型是否从专有LLM蒸馏而来，初步实验中在0.02显著性水平下实现1.0的真阳性率。

    

    未经授权的模型蒸馏——即使用专有大语言模型（LLM）的输出训练另一个模型——对模型提供商构成日益严重的威胁。我们研究蒸馏推断问题：判定某个可疑模型是从另一模型蒸馏而来，还是独立训练所得。我们将该问题形式化为假设检验，并通过训练影子模型来估计每种假设下的预期行为：蒸馏影子模型从教师模型的推理轨迹中学习，而独立影子模型仅从参考答案中学习。审计者衡量每个模型对教师模型推理输出的预测接近程度，然后利用影子模型将可疑模型的得分转换为经过校准的p值。在以Qwen2.5-7B为教师模型、Llama-3.2-3B为可疑模型的初步研究中，我们的检验在0.02的显著性水平下达到了1.0的真阳性率。这些结果证明了使用该方法的可行性。

    arXiv:2610.12137v1 Announce Type: cross  Abstract: Unauthorized model distillation, in which a model is trained on the outputs of a proprietary large language model (LLM), is a growing threat to model providers. We study distillation inference: determining whether a suspect model was distilled from another model or trained independently. We formulate this problem as a hypothesis test and estimate the behavior expected under each hypothesis by training shadow models: distilled shadow models learn from the teacher's reasoning traces, whereas independent shadow models learn only from reference answers. The auditor measures how closely each model predicts the teacher's reasoning outputs and then uses the shadow models to convert the suspect's score into a calibrated p-value. In a preliminary study using Qwen2.5-7B as the teacher and Llama-3.2-3B for the suspects, our test achieves a true positive rate of 1.0 at a significance level of 0.02. These results demonstrate the feasibility of usin
    
[^56]: 排练一切，记住无物：Attic-KV 只排练将被读取的内容

    Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read

    [https://arxiv.org/abs/2610.12133](https://arxiv.org/abs/2610.12133)

    该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。

    

    许多键值（KV）缓存会在无人知晓将来要被问及什么之前就被压缩：为检索而缓存的文档、跨请求共享的提示前缀、长对话的记忆。主流方法通过“排练”来为KV条目打分：模型重新阅读上下文并保留其关注的条目，并假设缓存对其上下文排练得越完整，就记得越牢。我们证明，在紧张的预算下这一假设会适得其反：排练一切，记住无物。在3%的保留率下，重读整个上下文在RULER上仅能保留96.5分中的31.5分，而在LongBench的自然文本任务上，其表现甚至低于完全不排练的方法。原因在于缓存会保留它所排练的内容：重读会将预算分散到整个上下文，导致答案自身的条目仅以略高于随机的概率得以保留。正如考试前的学生，缓存通过自我测试比通过重读能记得更多。

    arXiv:2610.12133v1 Announce Type: new  Abstract: Many key-value (KV) caches are compressed before anyone knows what will be asked of them: a document cached for retrieval, a prompt prefix shared across requests, the memory of a long conversation. The prevailing approach scores KV entries by rehearsal: the model rereads the context and keeps the entries it attends to, assuming that the more completely a cache rehearses its context, the better it remembers it. We show that under tight budgets this assumption backfires: rehearse everything, remember nothing. At a 3% keep ratio, rereading the whole context keeps 31.5 of 96.5 points on RULER, and on LongBench's natural-text tasks it falls below methods that rehearse nothing at all. The cause is that a cache keeps what it rehearses: rereading spreads the budget across the whole context, so the answer's own entries survive at little more than chance. Like a student before an exam, a cache remembers more by testing itself than by rereading. Tw
    
[^57]: 一种保持结构的聚合物电解质离子神经密度泛函

    A structure-preserving neural density functional for the ions of a polymer electrolyte

    [https://arxiv.org/abs/2610.12132](https://arxiv.org/abs/2610.12132)

    开发了一种保持空间对称性、热力学可积性和诺特定理恒等式的神经密度泛函，仅需平面密度和内力数据训练即可准确预测聚合物电解质离子在不同浓度和尺度下的结构与关联行为。

    

    预测非均匀聚合物电解质的结构和响应，需要对离子关联的描述既保持分子尺度的精度，又能在不同空间尺度和几何构型之间保持可迁移性。我们开发了一种面向电解质的神经密度泛函，它保持空间对称性、热力学可积性以及诺特定理恒等式，并在稳定、非临界的体相状态下恢复完美屏蔽行为。其非线性的密度依赖性捕捉到了配对封闭近似所遗漏的浓度依赖关联，包括在强耦合条件下长波长数密度涨落从增强到抑制的转变。该泛函能够描述未经训练的盐浓度下的密度分布，并预测体相结构因子和长波长数密度响应。仅通过分子动力学模拟得到的平面密度分布和内力分布进行训练，该泛函即可预测更大区域内离子的结构。

    arXiv:2610.12132v1 Announce Type: cross  Abstract: Predicting the structure and response of inhomogeneous polymer electrolytes requires a description of ion correlations that retains molecular-scale accuracy while remaining transferable across spatial scales and geometries. We develop a neural density functional for electrolytes that preserves spatial symmetries, thermodynamic integrability and the Noether identities, with perfect screening recovered in stable, noncritical bulk states. Its nonlinear density dependence captures the concentration-dependent correlations missed by a pair closure, including a crossover from enhanced to suppressed long-wavelength number fluctuations at strong coupling. The functional describes density profiles at an untrained salt concentration and predicts bulk structure factors and the long-wavelength number response. Trained solely on planar density and internal-force profiles from molecular dynamics, the functional predicts ionic structure in larger doma
    
[^58]: 在约束优化中使用Weisfeiler-Leman特征进行算法选择

    Using Weisfeiler-Leman Features for Algorithm Selection in Constraint Optimisation

    [https://arxiv.org/abs/2610.12119](https://arxiv.org/abs/2610.12119)

    本文提出一种结合图转换与Weisfeiler-Leman图核的自动化特征提取方法（包括基于割的WLc表示），无需训练图神经网络即可捕捉问题实例的结构信息，从而改进约束优化中的算法选择。

    

    算法选择对于高效的约束规划至关重要。多年来，许多基于机器学习方法的算法选择器已被成功应用，然而传统的特征提取方法通常依赖于人工决定的实例级统计信息，这些统计信息无法捕捉问题的潜在结构。在本文中，我们旨在通过引入一种新颖的自动化特征提取方法来弥补这一差距，该方法结合了图转换和Weisfeiler-Lehman图核，以生成问题实例的鲁棒结构表示。1-WL测试限定了标准消息传递图神经网络（GNN）的图区分能力，且合适的GNN架构可以达到这一界限。基于WL的特征提供了一种无需训练GNN的替代方案。我们的主要贡献是一种基于割的表示（WLc），旨在对结构划分进行建模……

    arXiv:2610.12119v1 Announce Type: cross  Abstract: Algorithm Selection is essential for efficient Constraint Programming. Over the years, many algorithm selectors based on machine learning methods have been successfully applied, yet traditional feature extraction methods often rely on manually decided instance-level statistics that fail to capture the underlying problem structure.   In this paper we aim to bridge this gap by introducing a novel, automated feature extraction methodology that integrates graph conversion and Weisfeiler-Lehman graph kernels to generate robust structural representations of problem instances. The 1-WL test bounds the graph-distinguishing power of standard message-passing Graph Neural Networks (GNNs), and suitable GNN architectures match this bound \citep{Xuetal2018}. WL-based features offer an alternative that does not require training a GNN. Our primary contribution is a cut-based representation (\texttt{WLc}) designed to model structural partitions and pro
    
[^59]: 面向风险规避决策的信度机器学习

    Credal Machine Learning for Risk-Averse Decision Making

    [https://arxiv.org/abs/2610.12115](https://arxiv.org/abs/2610.12115)

    该论文提出用信度集（概率分布的集合）来表示预测中的认知不确定性，并结合一种新颖的决策规则，实现基于CVaR的可靠风险规避决策。

    

    在许多机器学习应用中，有必要防范可能导致重大损失的最坏情形与预测。原则上，这可以通过训练风险规避的预测模型来实现，即最小化条件风险价值（CVaR）等损失函数，而不是依赖平均表现良好的模型。然而在实践中，由于学习者对真实损失分布（进而对真实CVaR）存在认知不确定性，这种风险规避方法的有效性会受到削弱。为实现可靠的风险规避，我们提出一种方法，将这种认知不确定性表示为信度集，即概率分布的集合。更具体地，我们开发了一个高效且可靠的学习器，以信度集的形式产生预测，并将其与一种新颖的决策规则相结合，该规则将每个信度集映射为用于CVaR最小化的单一预测分布。

    arXiv:2610.12115v1 Announce Type: new  Abstract: In many machine learning applications, it is necessary to guard against worst-case scenarios and predictions that could result in substantial losses. In principle, this can be achieved by training risk-averse predictive models that minimize loss functions such as conditional value-at-risk (CVaR), rather than relying on models that perform well on average. In practice, however, the effectiveness of this approach to risk aversion is undermined by the learner's uncertainty regarding the true loss distribution and, consequently, the true CVaR. To achieve reliable risk-aversion, we propose a method in which this (epistemic) uncertainty is represented in terms of credal sets, i.e., sets of probability distributions. More specifically, we develop an efficient yet reliable learner that produces predictions in the form of credal sets and combine it with a novel decision rule that maps each credal set to a single predictive distribution for CVaR m
    
[^60]: 一种基于Zonotope几何方法的软演员-评论家（SAC）运动学习算法

    A Geometric Approach to Soft Actor-Critic with Zonotopes for Locomotion Learning

    [https://arxiv.org/abs/2610.12113](https://arxiv.org/abs/2610.12113)

    GeZo-SAC通过让每个评论家预测定义Zonotope的生成向量，将几何宽度作为悲观偏移量，并根据评论家间的分歧程度自适应地在加权平均与最小值之间组合评论家，从而改进了软演员-评论家算法在运动控制学习中的表现。

    

    离策略演员-评论家方法通过取两个评论家网络输出的最小值来控制高估偏差。这种做法在任何地方都使用相同的聚合规则，而不考虑两个评论家之间的分歧程度。我们提出GeZo-SAC，它利用辅助几何表示使评论家的悲观程度能够自适应于状态和动作。除了标量值之外，每个评论家还预测一组定义Zonotope（超平行体）的生成向量。沿采样方向探测该Zonotope可以得到一个几何宽度，它作为悲观偏移量从每个评论家值中减去；同时还可以得到两个评论家之间分歧程度的度量，并通过log-sum-exp进行聚合。这种分歧程度控制两个评论家的组合方式：随着分歧增大，从宽度加权平均逐渐过渡到通常的最小值。在推理阶段，部署的策略是未经修改的SAC演员，因为生成向量仅在训练期间用于评论家一侧。在四个MuJoCo-v5运动基准测试上……（摘要原文在此截断）

    arXiv:2610.12113v1 Announce Type: cross  Abstract: Off-policy actor--critic methods control overestimation bias by taking the minimum of two critics. This uses the same aggregation rule everywhere, regardless of how the critics disagree. We propose \textbf{GeZo-SAC}, which uses auxiliary geometric representations to adapt critic pessimism to the state and action. Alongside its scalar value, each critic predicts a set of generators defining a zonotope. Probing this zonotope along sampled directions provides a geometric width, "subtracted from each critic value as a pessimistic offset, and a measure of disagreement between the two critics, aggregated with log-sum-exp. This disagreement controls how the critics are combined, moving from a width-weighted average toward the usual minimum as disagreement increases. At inference, the deployed policy is an unmodified SAC actor, since the generators are used only on the critic side during training.Across four MuJoCo-v5 locomotion benchmarks and
    
[^61]: LLM水印检测能否公开？

    Could LLM Watermark Detection be Public?

    [https://arxiv.org/abs/2610.12106](https://arxiv.org/abs/2610.12106)

    该论文提出一种分割密钥的公私钥LLM水印方法，通过公开检测器只暴露一个密钥，并利用公开与私有分数间的不平衡统计检验识别知情攻击者，从而证明水印检测可以安全地公开。

    

    水印技术被广泛用于追踪聊天机器人和智能体的输出，然而检测器一直未公开发布，因为一旦暴露，攻击者可能利用检测器的反馈进行有针对性的篡改。然而，水印本身已容易受到无信息篡改攻击的威胁。因此，我们首先量化了在部署场景下、在不同访问级别（从词元级分数到二元判定结果）中，公开检测器是否会带来额外风险。其次，我们提出了一种分割密钥的公私钥水印方法，通过公开检测器暴露其中一个密钥，同时保留另一个密钥用于完整验证与取证。知情攻击者只能操纵公开信号，从而在公开分数与私有分数之间产生不平衡。我们针对这种不平衡引入了一种统计检验方法，并将其与完整密钥的判定结果结合，构成一个两阶段机制。第三，我们在广泛的移除与伪造攻击上评估了该分割密钥方法。

    arXiv:2610.12106v1 Announce Type: cross  Abstract: Watermarking large language models is popular for tracing chatbot and agentic outputs, yet detectors remain unreleased since exposing them could let attackers do targeted edits with the detector's feedback. However, watermarks are already vulnerable to uninformed tampering attacks. We thus first quantify whether a public detector would be an additional liability in a deployment setting at varying levels of access, from token-level scores to a binary verdict. Second, we introduce a split-key public-private watermarking method that exposes one key through a public detector while keeping the other for full verification and forensics. An informed attacker can only move the public signal, creating an imbalance between public and private scores. We introduce a statistical test for this imbalance, and combine it with the full key verdict in a two-stage mechanism. Third, we evaluate the split-key method on a wide range of removal and forgery a
    
[^62]: 基于数据空间迭代的少步生成

    Few-Step Generation via Data-Space Iteration

    [https://arxiv.org/abs/2610.12102](https://arxiv.org/abs/2610.12102)

    该论文提出“数据空间迭代”少步生成框架，让共享生成器从噪声出发直接在数据空间中迭代精炼预测结果，从而彻底摆脱流匹配采样中对人工选择的时间步离散化的依赖。

    

    流匹配已成为训练高质量生成模型的一种可扩展范式，但从学到的概率流中采样需要多次网络评估。蒸馏可以将这一成本降低到一次或少数几次评估；然而，单步生成往往会牺牲质量，因此少步生成才是实际可行的运行方式。现有的少步方法沿着概率流执行迭代计算，因此需要一个固定的、人工选择的时间步离散化。这种离散化通常是启发式选择的，调优代价高昂；并且当不同样本或不同空间位置的精炼难度存在差异时，它也可能具有限制性。我们提出了数据空间迭代，这是一种完全摒弃流离散化的少步生成框架。从噪声出发，一个共享的生成器直接在数据空间中对其预测进行精炼，每次迭代都被训练以产生最优的样本（原文在此处截断）。

    arXiv:2610.12102v1 Announce Type: new  Abstract: Flow matching has emerged as a scalable paradigm for training high-quality generative models, but sampling from the learned probability flow requires many network evaluations. Distillation can reduce this cost to one or a few evaluations; however, one-step generation often sacrifices quality, making few-step generation the practical operating regime. Existing few-step methods perform their iterative computation along the probability flow and therefore require a fixed, manually chosen timestep discretization. This discretization is often chosen heuristically and is expensive to tune; it may also be restrictive when refinement difficulty differs across samples or spatial locations. We introduce data-space iteration, a few-step generation framework that removes flow discretization altogether. Starting from noise, a shared generator directly refines its prediction in data space, with every iteration trained to produce the best sample permitt
    
[^63]: SCORE：多变量高斯分布的谱相关性估计

    SCORE: Spectral Correlation Estimation for Multivariate Gaussians

    [https://arxiv.org/abs/2610.12096](https://arxiv.org/abs/2610.12096)

    提出SCORE框架，将评分规则训练与谱空间中的协方差近似相结合，能以线性存储和O(d log d)成本高效学习高维多变量高斯的密集相关结构，并具有数值稳定性和酉变换不变性等理论保证。

    

    基于神经网络的预测建模在处理高维结构化高斯目标时，需要对协方差矩阵进行高效、数值稳定且具有表达力的近似。我们提出SCORE：一个可扩展的框架，将评分规则训练与在谱空间中学习到的具有表达力的协方差近似相结合。对于$d$维数据，学习任务被分解为学习边际分布和学习结构化相关矩阵，这使得在线性存储空间和$\mathcal{O}(d\log d)$计算成本下即可建模密集的依赖关系。我们采用闭式高斯核评分进行训练，该评分即使在协方差退化的情况下仍有定义，且在优化过程中梯度有界。我们刻画了核评分在可逆变换下的性质，并证明了其在酉变换下的精确不变性。在总体水平上，我们的两级目标能够恢复真实的边际分布，并将目标相关性投影到……（原文截断）

    arXiv:2610.12096v1 Announce Type: new  Abstract: Neural network-based predictive modeling with high-dimensional structured Gaussian targets requires an efficient and numerically stable, yet expressive approximation of the covariance matrix. We propose SCORE: a scalable framework, combining scoring rule training with an expressive covariance approximation learned in spectral space. For $d$-dimensional data, the learning task is decomposed into learning the marginal distributions and learning a structured correlation matrix, which enables dense dependencies with linear storage and $\mathcal{O}(d\log d)$ cost. We utilize the closed form Gaussian kernel score for training, which remains defined even for degenerate covariances and admits bounded gradients during optimization. We characterize kernel scores under invertible transforms and prove exact invariance under unitary transforms. At population level, our two-level objective recovers the true marginals and projects the target correlatio
    
[^64]: DVLA-RL++：基于强化学习门控的双层视觉-语言对齐少样本学习方法

    DVLA-RL++: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning

    [https://arxiv.org/abs/2610.12095](https://arxiv.org/abs/2610.12095)

    DVLA-RL++ 通过互补语义净化与反事实强化学习门控，将物体内在语义与偶然上下文区分开来，避免少样本学习中的支持原型受到上下文污染，从而提升对新类别的识别能力。

    

    少样本学习旨在从有限的标注样本中识别新类别。近期研究通过引入文本语义来弥补视觉观测信息的不足，从而改进类别表示。然而，图像与文本之间的高度一致性可能既反映了物体的内在属性，也反映了偶然的上下文信息，这使得支持集原型容易受到上下文污染。为解决这一问题，我们提出了 DVLA-RL++，它在 DVLA-RL 的基础上扩展了互补语义净化（CSP）和反事实强化学习门控（CRG）两个模块。具体而言，CSP 从标注的支持样本中生成内在描述与干扰性描述，并将其与每个支持token的一致性进行对比；由模糊度决定的拒绝边际引导稀疏证据的分配，同时内在语义锚点填充未分配的部分，在视觉证据不可靠时提供后备方案。CRG 通过奖励信号学习逐层的语义融合强度……

    arXiv:2610.12095v1 Announce Type: cross  Abstract: Few-shot learning aims to recognize novel categories from limited labeled examples. Recent studies incorporate textual semantics to compensate for limited visual observations and improve class representations. However, high image-text agreement may reflect both intrinsic object properties and incidental context, making support prototypes susceptible to contextual contamination. To address this problem, we propose DVLA-RL++, which extends DVLA-RL with complementary semantic purification (CSP) and counterfactual reinforcement-learning gating (CRG). Specifically, CSP generates intrinsic and nuisance descriptions from labeled supports and compares their agreement with each support token. An ambiguity-dependent rejection margin guides sparse evidence allocation, while an intrinsic semantic anchor fills the unassigned mass to provide a fallback when visual evidence is unreliable. CRG learns layer-wise semantic fusion strengths using a reward
    
[^65]: 用于变分序贯蒙特卡洛的可微系统重采样

    Differentiable Systematic Resampling for Variational Sequential Monte Carlo

    [https://arxiv.org/abs/2610.12094](https://arxiv.org/abs/2610.12094)

    提出可微系统重采样（DSR），一种温度控制的系统重采样松弛方法，在保持其结构特性并具有可证明的指数收敛偏差的同时，实现了完全梯度流动，且计算开销远低于基于最优传输的方法。

    

    粒子滤波器是非线性状态估计的标准工具，但其重采样步骤是离散的，阻碍了变分序贯蒙特卡洛中基于梯度的学习。我们提出了可微系统重采样（DSR），这是一种对系统重采样的温度控制松弛方法，它在保持系统重采样基于CDF排序的带状结构的同时，实现了完全的梯度流动。当温度趋于零时，DSR收敛到精确的系统重采样，并且我们证明了由此引入的偏差具有逐点指数收敛速率。与基于最优传输的可微重采样方法相比，DSR避免了迭代求解器的使用，计算开销显著更低。在随机动力系统和真实世界手写数据上的实验表明，DSR在滤波和动力学学习方面取得了相当或更优的性能。

    arXiv:2610.12094v1 Announce Type: cross  Abstract: Particle filters are a standard tool for nonlinear state estimation, but their resampling step is discrete, preventing gradient-based learning in variational sequential Monte Carlo. We introduce Differentiable Systematic Resampling (DSR), a temperature-controlled relaxation of systematic resampling, that preserves the CDF-ordered, banded structure of systematic resampling while enabling full gradient flow. DSR converges to exact systematic resampling as the temperature vanishes, and we prove a pointwise exponential convergence rate for the induced bias. Compared to optimal-transport-based differentiable resampling, DSR avoids iterative solvers and has substantially lower computational overhead. Experiments on stochastic dynamical systems and real-world handwriting data show that DSR achieves comparable or superior filtering and dynamics learning performance.
    
[^66]: Perception Test 2026：挑战赛总结与城市级视听推理的扩展

    Perception Test 2026: Challenge Summary and Extension to City-scale Audio-Visual Reasoning

    [https://arxiv.org/abs/2610.12081](https://arxiv.org/abs/2610.12081)

    本报告总结了在ECCV 2026举办的第四届Perception Test挑战赛，介绍了两个基于城市级步行视频的新视听推理基准，并指出复杂的空间与多模态推理可以通过昂贵的智能体流程解决，但对单独使用的多模态模型仍然困难。

    

    延续Perception Test挑战赛系列，我们在瑞典马尔默举行的2026年欧洲计算机视觉会议（ECCV）上以研讨会形式组织了第四届挑战赛。本届聚焦于空间智能，设有四个不同的赛道：来自原Perception Test基准的统一多选视频问答和定位视频问答，以及两个基于城市级步行游览视频的新赛道。在本报告中，我们描述了用于城市级赛道的新基准，并总结了所有赛道的获胜方案，其中包括一个在所有赛道中参赛且表现令人满意的通用模型。新增城市级赛道的获胜方案表明，复杂的空间与多模态推理可以通过昂贵的智能体流程来解决，但对于单独使用的多模态模型而言仍然困难。

    arXiv:2610.12081v1 Announce Type: cross  Abstract: Continuing the Perception Test challenge series, we organised the fourth edition as a workshop at the European Conference on Computer Vision (ECCV) 2026 in Malm\"o, Sweden. This edition focused on spatial intelligence and featured four different tracks: unified multiple-choice videoQA and grounded videoQA from the original Perception Test benchmark, alongside two new tracks based on city-scale walking-tour videos (KilometerAudio and KilometerVision). In this report, we describe the new benchmarks used for the city-scale tracks and summarise the winning solutions across all tracks, including a generalist model that competed across all tracks with satisfactory performance. The winning solutions in the newly added city-scale tracks demonstrated that complex spatial and multimodal reasoning can be solved by expensive agentic pipelines, but remains difficult for multimodal models used standalone.
    
[^67]: 在昂贵模拟器的贝叶斯推断中利用梯度信息

    Exploiting Gradients in Bayesian Inference of Expensive Simulators

    [https://arxiv.org/abs/2610.12076](https://arxiv.org/abs/2610.12076)

    该论文提出在昂贵模拟器的贝叶斯推断中，利用模拟器输出对输入参数的梯度信息作为额外信号来指导基于贝叶斯优化的主动学习过程，从而提高有限模拟预算下的推断效率。

    

    基于微分方程的模拟器在科学和工程领域无处不在。它们常被用于基于模拟的推断中，根据对模拟器输出的真实世界观测来评估输入参数的后验分布。然而，当单次模拟器评估的计算成本很高时，推断就变得具有挑战性。在这种情况下，研究者已采用基于贝叶斯优化的主动学习方法，结合高斯过程代理模型，以从有限的模拟预算中最大化所获得的信息。近年来，模拟器输出关于输入参数的梯度变得越来越容易获得，但它们很少被用于推断。尽管我们只需要学习模拟器的输入-输出关系，梯度信息仍可以提供一个额外的有价值信号来引导主动学习过程。这在昂贵模拟器的情形下尤其值得关注。

    arXiv:2610.12076v1 Announce Type: new  Abstract: Simulators based on differential equations are ubiquitous in science and engineering. They are often used in simulation-based inference to evaluate the posterior distribution of the input parameters based on real-world observations of the simulator outputs. However, inference becomes challenging when individual simulator evaluations are computationally expensive. In such cases, a Bayesian optimization-based active learning approach with Gaussian process surrogate models has been used to maximize the information obtained from a limited simulation budget. Recently, gradients of simulator outputs with respect to input parameters have become increasingly available, yet they are rarely exploited for inference. Even though we only need to learn the simulator input-output relationship, gradient information can provide an additional valuable signal to guide the active learning procedure. This is of particular interest in the case of expensive si
    
[^68]: 扩散模型消除朗之万采样的条件数依赖：高斯情形下的精确分析

    Diffusion Removes Langevin's Conditioning Dependence: A Sharp Gaussian Analysis

    [https://arxiv.org/abs/2610.12052](https://arxiv.org/abs/2610.12052)

    本文在高斯情形下证明扩散模型的采样误差为 $O(\sqrt{d\lambda_{\max}}\log N/N)$，消除了经典朗之万类采样器中依赖条件数的 $\sqrt{\kappa}$ 因子，并通过精确的谱界与匹配的一阶渐近分析，严格解释了扩散模型优于传统基于分数采样器的理论原因。

    

    尽管扩散模型在实证上取得了巨大成功，但它们为何能突破经典基于分数的采样器的瓶颈仍不清楚。在本工作中，我们利用高斯分布来分离这一现象。我们针对优化超参数建立了2-Wasserstein收敛界，证明扩散过程能达到 $O(\sqrt{d\lambda_{\max}}\log N/N)$ 的采样误差，其中 $d$ 是维度，$N$ 是采样步数，$\lambda_{\max}$ 是目标协方差矩阵的最大特征值。相比之下，未校正朗之万动力学和欠阻尼朗之万动力学则额外多出 $\sqrt{\kappa}$ 因子，其中 $\kappa$ 是条件数。这些速率来自精确的谱界：我们通过在 $N\rightarrow\infty$ 时匹配的一阶渐近性证实了它们。我们的分析在高斯情形下严格刻画了依赖时间的分数轨迹如何在采样过程中消除条件数依赖。

    arXiv:2610.12052v1 Announce Type: cross  Abstract: Despite their empirical success, why diffusion models overcome the bottlenecks of classical score-based samplers remains unclear. In this work, we leverage Gaussian distributions to isolate this phenomenon. We establish 2-Wasserstein convergence bounds for optimized hyperparameters, showing that diffusion processes achieve a sampling error of $O(\sqrt{d\lambda_{\max}}\log N/N)$, where $d$ is the dimension, $N$ the number of sampling steps, and $\lambda_{\max}$ the largest eigenvalue of the target covariance matrix. Unadjusted and underdamped Langevin dynamics suffer from an additional $\sqrt\kappa$ factor, where $\kappa$ is the condition number. These rates follow from spectral bounds which are sharp: we confirm them via matching first-order asymptotics as $N\rightarrow\infty$. Our analysis provides a rigorous characterization, in the Gaussian setting, of how time-dependent score trajectories remove condition-number dependence during s
    
[^69]: MPGE：用于分子分类解释的多视角图解释器

    MPGE: A Multi-Perspective Graph Explainer for Molecular Classification Explanation

    [https://arxiv.org/abs/2610.12039](https://arxiv.org/abs/2610.12039)

    该论文提出MPGE多视角图解释框架，针对冻结的图神经网络分类器统一了事实支持（原型）、反事实敏感性和示例容忍性三种解释视角，通过共享约束公式和独立目标函数为分子分类预测提供更全面的解释。

    

    图神经网络（GNN）可以从化学图数据中预测分子性质，但预测准确性并不能解释图信息是如何支持某个具体决策的。一个紧凑的、保持预测结果的理由并不一定能揭示哪些改变会逆转决策，或者哪些修改是模型可以容忍的。我们提出了多视角图解释器（MPGE），针对冻结的分类器统一了事实支持、反事实敏感性和示例容忍性这三种视角。事实视角（最初称为原型，PT）寻找一个紧凑的保留边集合，使其保持相同的标签和所需的置信度。反事实（CF）解释寻找有界的、能够改变预测结果的删除操作；示例（EXE）解释寻找能够保持标签和置信度的非平凡有界删除操作。一个共享的约束公式将预测行为、紧凑性和编辑成本联系起来，而各自独立的目标函数生成这三种视角。

    arXiv:2610.12039v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) predict molecular properties from chemical graph data, but predictive accuracy does not explain how graph information supports an individual decision. A compact prediction-preserving rationale does not necessarily reveal which changes reverse the decision or which modifications the model tolerates. We propose the Multi-Perspective Graph Explainer (MPGE), unifying factual support, counterfactual sensitivity, and exemplar tolerance for a frozen classifier. The factual view, originally termed prototype (PT), seeks a compact retained edge set with the same label and required confidence. Counterfactual (CF) explanations seek bounded prediction-changing deletions; exemplar (EXE) explanations seek non-trivial bounded deletions that preserve the label and confidence. A shared constrained formulation connects prediction behavior, compactness, and edit cost, while separate objectives generate the three views. Our gra
    
[^70]: 面向离散数据的高效且可泛化的原型分析

    Efficient and Generalizable Archetypal Analysis for Discrete Data

    [https://arxiv.org/abs/2610.12035](https://arxiv.org/abs/2610.12035)

    提出了一种面向离散数据的基于似然的高效原型分析框架，支持伯努利、泊松和多项式观测模型，并引入交叉验证的预测似然准则来有原则地选择原型数量。

    

    原型分析将观测数据表示为极端数据驱动轮廓的凸组合，从而为复杂数据集提供可解释的低维描述。经典的原型分析依赖于最小二乘目标函数，这并不适合诸如二值、计数和分类数据等离散观测数据。我们引入了一个高效的基于似然的原型分析框架，支持伯努利、泊松和多项式观测模型。我们的优化方案采用负对数似然的局部二次近似，通过序列最小优化（SMO）和活动集方法实现约束更新。通过在保持单纯形可行性的同时对活动集进行限制，提升了算法的可扩展性。我们进一步引入了交叉验证的预测似然准则来选择原型的数量，为重构误差启发式方法和基于稳定性的诊断方法提供了一种更有原则的替代方案。

    arXiv:2610.12035v1 Announce Type: cross  Abstract: Archetypal Analysis (AA) represents observations as convex combinations of extremal data-driven profiles, yielding interpretable low-dimensional descriptions of complex datasets. Classical AA relies on a least-squares objective, which is poorly suited to discrete observations such as binary, count, and categorical data. We introduce an efficient likelihood-based framework for AA supporting Bernoulli, Poisson, and multinomial observation models. Our optimization scheme employs local quadratic approximations of the negative log-likelihood, enabling constrained updates through sequential minimal optimization (SMO) and an active-set method. Scalability is improved by bounding the active set while preserving simplex feasibility. We further introduce a cross-validated predictive likelihood criterion for selecting the number of archetypes, providing a principled alternative to reconstruction-error heuristics and stability-based diagnostics. S
    
[^71]: 检验大语言模型推理中的社会归因：一种理论指导的探测方法

    Examining Social Attribution in LLM Reasoning: A Theory-Guided Probing Methodology

    [https://arxiv.org/abs/2610.12022](https://arxiv.org/abs/2610.12022)

    该论文首次系统性地探索大语言模型的社会归因能力，在归因理论指导下构建了包含经典心理学情境与现实场景的基准，以考察大语言模型在责任与过错归因上的判断及其内部机制。

    

    大语言模型越来越多地被部署在社会技术系统中，其中社会归因——即将外部事件归因于智能体社会行为的原因和缘由的推理过程——发挥着关键作用。这些过程涉及对社会原因、责任以及智能体所应承担的过错或功劳的判断。尽管归因模型在心理学和认知科学中已通过归因理论得到充分研究，但社会归因在人工智能领域，尤其是大语言模型的社会推理中，仍然探索不足。本文首次对大语言模型的社会归因进行了系统性探索。我们的工作聚焦于责任与过错归因，考察了当前大语言模型的判断表现及其潜在的内部机制。在归因理论的指导下，我们构建了一个社会归因基准，其中包含基于归因理论研究经典情景的情境片段子集，以及基于现实世界社会场景的现实子集。

    arXiv:2610.12022v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed in sociotechnical systems where social attribution, the reasoning process attributing external events to the causes and reasons of agents' social behaviors, plays a critical role. These processes involve judgments of social cause, responsibility, and blame/credit to agents. Although attributional models are well-studied in social psychology and cognition through Attribution Theory, social attribution remains underexplored in AI, particularly LLM social reasoning. This paper provides the first systematic exploration of LLM social attribution. Our work focuses on responsibility and blame attributions, examining current LLMs' judgments and their underlying internal mechanisms. Guided by attribution theory, we construct a social attribution benchmark consisting of a Vignette subset based on classic scenarios from attribution theory research and a Reality subset based on real-world soci
    
[^72]: CausalDreamer：学习具有潜在解耦的预测性世界模型

    CausalDreamer: Learning Predictive World Models with Latent Disentanglement

    [https://arxiv.org/abs/2610.12016](https://arxiv.org/abs/2610.12016)

    CausalDreamer 在冻结的视频分词器之上，将潜在表示沿可控性和奖励相关性两个维度解耦为四组因素化表示，使世界模型能够显式区分环境中可控、不可控、奖励相关和奖励无关的信息。

    

    用于控制的世界模型必须捕捉环境中的哪些方面会响应智能体的动作，以及哪些方面与奖励相关。诸如 Dreamer 4 之类的生成式世界模型由视频分词器和动力学模型组成，其中视频分词器将每一帧编码为潜在表示，动力学模型则通过预训练来根据过去的潜在表示和动作预测未来的潜在表示。然而，分词器是采用重建目标进行训练的，缺乏动作或奖励的监督信号，因此其潜在表示没有提供显式机制来区分可控、不可控、与奖励相关以及与奖励无关的信息。我们提出 CausalDreamer，该方法保持分词器冻结，并将其潜在表示重新编码为沿两个维度划分的四组因素化表示：一个是可控性维度，其中只有两个可控组接收动作输入；另一个是奖励相关性维度，通过从两个奖励相关组预测奖励来学习。随后对预训练的动力学模型进行微调……（摘要在此处被截断）

    arXiv:2610.12016v1 Announce Type: new  Abstract: World models for control must capture which aspects of the environment respond to the agent's actions and which are relevant to reward. Generative world models such as Dreamer 4 consist of a video tokenizer, which encodes each frame into a latent, and a dynamics model, which is pretrained to predict future latents from past latents and actions. Yet the tokenizer is trained with a reconstruction objective, without action or reward supervision, so its latent provides no explicit mechanism to separate controllable, uncontrollable, reward-relevant, and reward-irrelevant information. We propose \textit{CausalDreamer}, which keeps the tokenizer frozen and re-encodes its latent into a factored representation of four groups along two axes: controllability, where only the two controllable groups receive the action, and reward relevance, learned by predicting the reward from the two reward-relevant groups. The pretrained dynamics model is then fin
    
[^73]: 用于求解线性微分方程的参数化高斯过程的“幽灵任务”方法

    Ghost tasking for parametrized Gaussian Processes solving linear differential equations

    [https://arxiv.org/abs/2610.12009](https://arxiv.org/abs/2610.12009)

    本文提出“幽灵任务”方法，通过引入辅助任务使任意不可参数化系统有效变得可参数化，从而能以较少的任务数量和潜在函数算法化构建参数化高斯过程来求解线性微分方程，并在数据稀少的逆问题中表现尤为出色。

    

    物理信息机器学习近年来受到了广泛关注。在数据有限的场景下，参数化高斯过程变得日益流行。然而，现有方法往往面临一些局限性，例如要求系统本身可参数化（也称为可控），或需要大量的输出任务。在这项工作中，我们引入了一种称为“幽灵任务”的系统化方法，通过使用辅助任务来规避这些限制。我们证明了这种幽灵任务可以使任何不可参数化的系统有效地变得可参数化，从而实现参数化高斯过程的算法化构建，同时将所需的任务数量（即输出维度）和潜在函数数量保持在较低水平。我们发现幽灵任务方法在逆问题设定中表现尤为出色，即使可用数据非常少。我们通过三个实验展示了幽灵任务方法的使用方法和强大能力，并与唯一的其他方法进行了系统性比较。

    arXiv:2610.12009v1 Announce Type: cross  Abstract: Physics-informed machine learning has gained significant attention in recent years. In regimes of limited data, parametrized Gaussian processes have become popular. Existing approaches, however, often face limitations, such as requiring parametrizable (also called controllable) systems or a large number of output tasks. In this work, we introduce a systematic procedure we call "ghost tasking", using auxiliary tasks to circumvent these limitations. We prove that such ghost tasks can render any non-parametrizable system effectively parametrizable, enabling algorithmic construction of parametrized Gaussian Processes while keeping the number of required tasks (i.e. output dimensions) and latent functions low. We find that ghost tasking performs especially well in an inverse problem setting, even with very few available data. We show the usage and power of ghost tasking in three experiments, providing systematic comparisons to the only othe
    
[^74]: 表格基础模型的测试时计算：机制、收益与局限

    Test-Time Compute for Tabular Foundation Models: Mechanisms, Gains, and Limits

    [https://arxiv.org/abs/2610.12005](https://arxiv.org/abs/2610.12005)

    本文系统研究了测试时计算对表格基础模型预测性能的提升机制，提出仅训练0.003-0.03%参数的对角线相似度更新方法DiagScale，其效果可媲美全量微调，并发现对96种配置采用贪婪选择可将误差降低2.4%，而均匀平均反而会增加误差。

    

    哪些形式的测试时计算能够提升强大的预训练表格基础模型（TFMs）的预测性能？我们从三个维度系统地研究了这个问题：自适应、聚合与上下文构建。我们的评估涵盖了TabArena基准测试中的现代TFMs，并辅以来自OpenML的宽表和大规模表格数据上的实验。在自适应方面，我们提出了DiagScale，一种对角线查询-键相似度更新方法。它仅训练模型参数的0.003-0.03%，却在三个独立预训练的骨干网络上取得了与全量微调相当的收益。在聚合方面，预测池的构成和选择策略都很重要。TabPFN-3已经对同一数据的不同预处理变体的预测进行平均，而增加更多此类预测会带来收益递减。在更广泛的96种配置池中，贪婪选择相比默认预测器可将误差降低2.4%，但均匀平均反而会增加误差。

    arXiv:2610.12005v1 Announce Type: new  Abstract: Which forms of test-time compute improve the predictions of strong pretrained tabular foundation models (TFMs)? We systematically study this along three axes: adaptation, aggregation, and context construction. Our evaluation spans modern TFMs across the TabArena benchmark, supplemented by experiments on wide and large-scale tables from OpenML. For adaptation, we introduce DiagScale, a diagonal query-key similarity update. It trains only 0.003-0.03% of model parameters and achieves gains comparable to full fine-tuning across three independently pretrained backbones. For aggregation, both pool composition and selection strategy matter. TabPFN-3 already averages predictions from different preprocessing variants of the same data, and adding more such predictions yields diminishing returns. With a broader pool of 96 configurations, greedy selection reduces error by 2.4% relative to the default predictor, but uniform averaging increases error.
    
[^75]: 多面体神经网络

    The Polytopal Neural Network

    [https://arxiv.org/abs/2610.12004](https://arxiv.org/abs/2610.12004)

    本文提出多面体神经网络（PNN）框架，通过在信息处理中直接强制多面体结构来提取各层级特定特征，在几乎不损失性能的前提下保留潜在空间的有意义结构，同时实现压缩表示并为向量量化训练提供直接途径。

    

    理解深度神经网络如何处理信息仍然是一个核心挑战。现有的可解释性方法往往在结构保真度上有所妥协、依赖预先指定的语料库，或只能对模型进行事后解释。我们提出了多面体神经网络，这是一个通过强制实施基于多面体的结构来提取不同层级特定特征的框架，且该结构直接用于后续的信息处理。我们借助学习到的语料库表示和摊销的单纯形推断过程对方法进行了扩展，并强调该框架还为向量量化训练提供了直接途径。在PNN中，观测数据通过与各层级特定特征的对齐程度被显式地描述。实证结果表明，对神经网络表示施加多面体约束能够在性能损失极小的情况下保留潜在空间中有意义的结构，并且在压缩表示方面相比……表现更优。

    arXiv:2610.12004v1 Announce Type: cross  Abstract: Understanding how deep neural networks process information remains a central challenge. Existing interpretability methods often compromise structural fidelity, rely on prespecified corpora, or explain models post-hoc. We propose Polytopal Neural Networks (PNNs), a framework that extracts distinct layer-wise aspects by enforcing a polytope-based structure that is used directly in subsequent information processing. We scale our approach using learned corpus representations and an amortized simplex inference procedure and highlight how the framework also gives a direct route to vector quantized (VQ) training. In PNNs, observations are explicitly described by their alignment with layer-specific aspects. Empirical results show that imposing polytopal constraints on neural network representations preserves meaningful structures in the latent space with minimal degradation in performance, favorable compressed representations when compared to 
    
[^76]: Agentic-TTT：为测试时训练学习测试时策略

    Agentic-TTT: Training test-time policy for test-time training

    [https://arxiv.org/abs/2610.12002](https://arxiv.org/abs/2610.12002)

    提出Agentic-TTT框架，通过训练一个测试时策略来智能决策何时、如何调用测试时训练（TTT）以及是否复用已有技能，从而实现模型参数层面的自主化自我改进。

    

    测试时训练（TTT）利用来自测试输入的信号来调整大语言模型的参数，可以在预先指定的场景（如IMO竞赛或指定的开放性问题）中带来显著的性能提升。通过将部署经验转化为参数更新，TTT为模型层面的自我改进提供了一种直接机制。然而，TTT并非在所有情况下都有益：每种TTT算法适用于不同的场景，应用不当的方法可能浪费测试时计算资源，甚至会损害模型性能。因此，这种参数层面的自我改进需要自主性：模型必须决定何时需要TTT、调用哪种算法，以及是否可以复用已有的技能。为填补这一空白，我们提出了Agentic-TTT，它通过学习一个测试时策略来管理这些决策。Agentic-TTT将TTT流程转化为可调用的工具，将积累的技能视为不断演化的部署环境，并使用（摘要在此处截断）

    arXiv:2610.12002v1 Announce Type: cross  Abstract: Test-time training (TTT) adapts an LLM's parameters using signals derived from test inputs, and can make striking improvements in pre-specified settings such as IMO competitions or designated open problems. By turning deployment experience into parameter updates, TTT provides a direct mechanism for model-level self-improvement. Yet TTT is not universally beneficial: each TTT algorithm works in different settings, and applying an ill-suited method could waste test-time compute or even damage model performance. Therefore, such parameter-level self-improvement requires agency: the model must decide when TTT is warranted, which algorithm to invoke, and whether an existing skill can be reused. To fill this gap, we introduce Agentic-TTT, which learns a test-time policy to govern those decisions. Agentic-TTT turns TTT procedures into callable tools, treats accumulated skills as an evolving deployment environment, and trains its policy using t
    
[^77]: 面向贝叶斯形状优化的样例驱动参数化方法

    Example-driven Parametrisations for Bayesian Shape Optimisation

    [https://arxiv.org/abs/2610.11984](https://arxiv.org/abs/2610.11984)

    本文提出从现有设计集合中通过主成分分析学习形状间变形的参数化方法，为贝叶斯形状优化构建了一个线性、可解释的搜索空间，在翼型、机翼和射频腔体等任务中实现了更高的样本效率，并能探索超越手工参数化基线的更优设计。

    

    当目标函数计算代价高昂且不可微时，贝叶斯优化是形状设计的自然工具，但它需要一个紧凑而富有表现力的搜索空间参数化。手工设计这样的参数化是一项复杂的工作，需要领域专业知识，且常常产生隐式的不可行区域、人为设置的边界以及耦合、无序的坐标。我们转而从一组现有设计中学习参数化，对形状之间的变形应用主成分分析。其结果是一个线性、可解释的搜索空间，其中主成分的数量可以明确地在表现力与维度之间进行权衡。在翼型、机翼和射频腔体上，涵盖二维几何到三维空气动力学和电磁学，我们展示了更高的样本效率以及探索超越手工设计基线限制的能力。

    arXiv:2610.11984v1 Announce Type: new  Abstract: Bayesian optimisation is the natural tool for shape design when objectives are expensive and non-differentiable, but it needs a compact yet expressive parameterisation of the search space. Hand-crafting one is a complex endeavour requiring domain expertise, and often yields implicit infeasible regions, artificial bounds, and coupled, unordered coordinates. We instead learn the parameterisation from a collection of existing designs, applying principal component analysis to the deformations between shapes. The result is a linear, interpretable search space in which the number of components explicitly trades expressivity against dimensionality. Across aerofoils, wings, and radio-frequency cavities, spanning 2D geometry to 3D aerodynamics and electromagnetics, we show improved sample efficiency and the ability to explore beyond the confines of hand-crafted baselines.
    
[^78]: 基于距离草图的高效二次熵计算

    Efficient quadratic entropy with distance sketches

    [https://arxiv.org/abs/2610.11976](https://arxiv.org/abs/2610.11976)

    本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。

    

    我们详细介绍了用于近似任意分布 $p$ 和负型常见距离 $d$ 下二次熵 $p^T d p$ 的可扩展方法。我们聚焦于欧几里得距离和球面测地距离两种情形，二者均在简单的框架内利用随机特征嵌入和投影来显著改善计算复杂度。通过摊销单次大型矩阵乘法以及控制变量技术，在 $d$ 保持不变而 $p$ 变化的场景下，该方法进一步实现了低内存占用和短运行时间的大规模计算。我们通过与直接配对采样方法的对比，以及在 Open Graph Benchmark 数据集上的文献计量学/科学计量学示例验证了该方法，仅凭引用和文本特征即可揭示跨学科影响力特别窄或特别宽的论文、领域和机构。

    arXiv:2610.11976v1 Announce Type: cross  Abstract: We detail scalable methods for approximating the quadratic entropy $p^T d p$ for arbitrary distributions $p$ and common distances $d$ of negative type. We focus on the Euclidean and spherical geodesic cases, which both use random feature embeddings and projections to dramatically improve computational complexity within a simple framework. Amortization of a single large matrix multiplication and control variates further enable computation at large scale with low memory and runtime in situations where $d$ is held constant while $p$ varies. We demonstrate this with a comparison against direct pair sampling and bibliometric/scientometric examples on Open Graph Benchmark datasets, revealing papers, fields, and institutions with both particularly narrow and broad interdisciplinary reach from their citations and text features alone.
    
[^79]: CAPABLE：通过行为潜在编码实现的能力感知策略自适应

    CAPABLE: Capability-Aware Policy Adaptation via Behavioral Latent Encoding

    [https://arxiv.org/abs/2610.11971](https://arxiv.org/abs/2610.11971)

    CAPABLE是一个无需故障标签或重新训练的能力感知自适应框架，通过自监督能力推断与残差强化学习相结合，使冻结的视觉-语言-动作（VLA）策略能够在关节故障发生时在线适应并恢复执行能力。

    

    视觉-语言-动作（VLA）策略假设其训练时的机器人本体结构保持不变，当关节故障改变指令动作的物理执行方式时可能会失效。现有的故障恢复方法通常需要针对特定任务的重新训练、故障标签、显式诊断或特权本体信息。我们提出了CAPABLE，一个面向冻结VLA的统一能力感知自适应框架，它将自监督能力推断与残差强化学习相结合。CAPABLE利用跨关节共享的时间编码器、雅可比矩阵接地、跨关节注意力和自监督物理预测，在线地从指令-响应历史和运动学信息中推断能力，即每个关节实际实现的指令运动程度以及该运动对末端执行器行为的贡献。所得的表示用于条件化一个残差策略，该策略在VLA手臂动作上添加有界校正，无需故障……

    arXiv:2610.11971v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies assume the embodiment on which they were trained and can fail when a joint fault changes how commanded actions are physically executed. Existing fault-recovery methods often require task-specific retraining, fault labels, explicit diagnosis, or privileged embodiment information. We introduce CAPABLE, a unified capability-aware adaptation framework for frozen VLAs that integrates self-supervised capability inference with residual reinforcement learning. CAPABLE infers capability, how much of the commanded motion each joint actually realizes and how that motion contributes to end-effector behavior, online from command-response history and kinematics using a temporal encoder shared across joints, Jacobian grounding, cross-joint attention, and self-supervised physical prediction. The resulting representation conditions a residual policy that adds bounded corrections to the VLA arm action without fault 
    
[^80]: 面向有效子群体可靠性的随机分组保形预测

    Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability

    [https://arxiv.org/abs/2610.11957](https://arxiv.org/abs/2610.11957)

    该论文提出随机分组保形预测（SGCP），通过学习随机分组映射让每个样本从校准行为相似的样本中获取校准信息，在无需敏感子群体属性的情况下实现跨临床子群体的可靠不确定性量化，并缓解最差群体瓶颈问题。

    

    保形预测提供了一种无分布的覆盖率保证，这使其在临床应用中尤其具有吸引力。然而，标准保形预测仅在总体层面提供此类保证，其预测集在临床上重要的子群体之间可能表现出覆盖率差异。一种自然的补救方法是在预定义的群体内进行校准。然而，这可能需要获取敏感的子群体属性，并且容易出现最差群体瓶颈：保护最困难的子群体可能会使所有样本的预测集膨胀，从而增加决策者的认知负担。为此，我们提出了随机分组保形预测（SGCP），这是一个用于子群体可靠不确定性量化的保形框架。它学习一个随机分组映射，使每个样本能够从校准行为相似的其他样本中获取校准信息，从而产生一种局部分数法则，跨子群体提升可靠性。

    arXiv:2610.11957v1 Announce Type: cross  Abstract: Conformal prediction offers a distribution-free coverage guarantee, making it especially attractive for clinical applications. Standard conformal prediction, however, provides such guarantees only at the population level, and its prediction sets can exhibit coverage disparities across clinically important subgroups. A natural remedy is to calibrate within predefined groups. However, this can require access to sensitive subgroup attributes and is prone to a worst-group bottleneck: protecting the most difficult subgroup can inflate prediction sets for all, increasing cognitive burden on decision makers. To this end, we propose Stochastic Grouping Conformal Prediction (SGCP), a conformal framework for subgroup-reliable uncertainty quantification. It learns a stochastic grouping map that allows each sample to draw calibration information from others with similar calibration behavior, yielding a local score law that boosts reliability acros
    
[^81]: 面向时间鲁棒机器人操作的可靠性感知未来条件化

    Reliability-Aware Future Conditioning for Temporally Robust Robot Manipulation

    [https://arxiv.org/abs/2610.11956](https://arxiv.org/abs/2610.11956)

    本文提出可靠性感知未来条件化（RAFC），将生成视频作为指导时的时间错位问题重新定义为一个控制问题，通过在每一步估计对生成片段的信任程度、偏好邻近的时间假设并在不匹配时回退到静态分支，且仅依靠任务奖励学习而无需偏移标签或对齐监督，从而显著提升机器人操作策略对时序偏移的鲁棒性。

    

    arXiv:2610.11956v1 公告类型：交叉列表 摘要：机器人即将执行的任务的生成视频，只有在其描绘了机器人实际所处的阶段时才是有用的指导。我们证明，时间上的错位可能使一个与任务一致的生成未来变成有害的指导。在CALVIN基准上，五帧的提前偏移几乎抹去了生成未来带来的收益，使成功率从81.3%降至54.8%（无未来时为54.0%）；强制的时序偏移使其进一步降至34.2%，比无未来策略低19.8个百分点。我们提出可靠性感知未来条件化（RAFC），将此视为一个控制问题而非生成问题。在每一步，RAFC估计应当多大程度地信任接收到的视频片段，以及应优先选择哪个邻近的时间假设，当两者都不合适时回退到静态分支；它仅从任务奖励中学习，无需偏移标签或对齐监督。RAFC构建在未来经验条件化（FEC）之上，FEC构建了……（原文摘要在此处截断）

    arXiv:2610.11956v1 Announce Type: cross  Abstract: A generated video of a task the robot is about to perform is useful guidance only if it depicts the phase the robot is actually in. We show that temporal misalignment can turn a task-consistent generated future into actively harmful guidance. On CALVIN, a five-frame early shift nearly erases the benefit of generated futures, reducing success from 81.3% to 54.8% against 54.0% without futures; imposed timing shifts reduce it even further to 34.2%, 19.8 points below the future-free policy. We introduce Reliability-Aware Future Conditioning (RAFC), which treats this as a control problem rather than a generation problem. At every step, RAFC estimates how far to trust the received clip and which nearby temporal hypothesis to prefer, falling back toward a static branch when neither fits, and it learns both from task reward alone without shift labels or alignment supervision. RAFC sits on top of Future-Experience Conditioning (FEC), which buil
    
[^82]: 基于树模型的区间值SHAP方法

    Interval-valued SHAP in Tree-Based Models

    [https://arxiv.org/abs/2610.11953](https://arxiv.org/abs/2610.11953)

    本文提出基于不精确Dirichlet模型的区间值SHAP方法，通过向树的叶子节点随机引入少量未标注实例来量化和分析决策树与随机森林中Shapley值的鲁棒性，并借助悲观原则与平均原则定义区间值，同时推导出实现高效计算的理论结果。

    

    Shapley值是最流行的特征归因解释方法之一。针对表格数据集中表现最先进的树模型，已有高效的Shapley值计算/估计方法被开发出来。然而，众所周知，Shapley值在面临微小且符合现实的变动时可能（高度）不鲁棒。本文提出了一种基于不精确Dirichlet模型（IDM）的方法，用于分析决策树和随机森林中Shapley值的鲁棒性。在技术上，该方法通过当少量未标注实例被随机引入树的叶子节点时，量化并分析区间值Shapley值来实现。区间值Shapley值可以按照处理不完整数据的常见原则来定义，即悲观原则和平均原则。我们推导出了多种理论结果，使区间值Shapley值的高效计算成为可能。我们还表明，所提出的方法……（摘要原文在此处截断）

    arXiv:2610.11953v1 Announce Type: new  Abstract: Shapley values are among the most popular feature-attribution explanations. Efficient approaches for computing/estimating Shapley values for tree-based models, which are state-of-the-art for tabular data sets, have been developed. However, it is known that Shapley values can be (highly) unrobust due to small and realistic changes. In this paper, we propose an imprecise Dirichlet model (IDM) based method to analyze the robustness of Shapley values in decision trees and random forests. Technically, it is done by quantifying and analyzing the interval-valued Shapley values when a few unannotated instances are randomly introduced to the leaves of the trees. The interval-valued Shapley values can be defined following common principles in handling incomplete data: the pessimistic and averaging principles. We derive various theoretical results that lead to efficient computation of the interval-valued Shapley values. We also show that the propos
    
[^83]: 基于评分的方法从干预中学习聚类DAG

    Score-Based Learning of Cluster DAGs from Interventions

    [https://arxiv.org/abs/2610.11947](https://arxiv.org/abs/2610.11947)

    提出首个基于评分的方法COARSE，利用干预数据识别聚类间的因果顺序并将边学习简化为局部搜索，从而在线性高斯假设下实现聚类DAG的学习。

    

    因果抽象的图方法将一个包含众多测量变量的低层因果有向无环图（DAG）转换为一个更小的高层DAG，其节点对原始变量进行聚类，其边总结了聚类之间的因果关�系。这样的聚类DAG更易于解释，但学习它们需要找到聚类并恢复它们之间的边。Madaleno等人（2026）以两个基于约束的阶段学习干预粗化（即将干预无法区分的变量合并而成的聚类DAG）：先学习聚类，再学习边。我们提出了COARSE，这是该任务中第一个基于评分的方法：它保留了两个阶段的结构，但在线性高斯假设下，将基于约束的边学习阶段替换为基于评分的阶段。我们证明干预本身可以识别聚类之间的因果顺序，并且边的学习可以简化为针对每个聚类的单一局部搜索。

    arXiv:2610.11947v1 Announce Type: cross  Abstract: Graphical approaches to causal abstraction transform a low-level causal directed acyclic graph (DAG) over many measured variables into a smaller, high-level DAG whose nodes cluster the original variables and whose edges summarize the causal relations between clusters. Such cluster DAGs are easier to interpret, but learning them requires finding the clusters and recovering the edges between them. Madaleno et al. (2026) learn the interventional coarsening (the cluster DAG that merges variables the interventions cannot distinguish) in two constraint-based phases: first the clusters, then the edges. We introduce COARSE, the first score-based method for this task: it keeps the two-phase structure but, under linear Gaussian assumptions, swaps the constraint-based edge phase for a score-based one. We show that the interventions themselves identify a causal order over the clusters, and learning the edges reduces to a single local search per cl
    
[^84]: TACROSS：一种面向灵巧机器人学习的高效低成本、可扩展的跨异构触觉传感器人类触觉系统

    TACROSS: An Efficient and Low-Cost Scalable Human Touch System Across Heterogeneous Tactile Sensors for Dexterous Robot Learning

    [https://arxiv.org/abs/2610.11945](https://arxiv.org/abs/2610.11945)

    TACROSS提出了一种成本仅10.86美元的五层压阻式触觉手套系统，通过在接触事件层面而非原始传感器值层面对齐异构触觉信号，将人类触觉数据高效迁移到机器人触觉传感器上，实现了可扩展的低成本灵巧机器人学习。

    

    在机器人上收集触觉演示数据成本高昂且速度缓慢，这促使人们使用成本更低的人类触觉手套来进行可扩展的数据收集。然而，人类电容式/压阻式手套与机器人触觉传感器在换能原理、传感器布局、空间分辨率和动态响应方面存在根本性差异，使得对原始传感器通道进行对齐成为一个不适定问题。为了解决这一问题，我们提出了TACROSS，一个从人类触觉中学习并将其迁移到机器人上的可扩展系统。该系统通过在接触事件层面而非原始传感器值层面对齐触觉数据流，从而弥合了这种异构性。TACROSS的硬件部分集成了一个五层压阻式手套，成本仅为10.86美元，拥有285个感知点。为了对齐接触语义，我们设计了规范化器和残差适配器，通过带有时序注意力机制的时间Transformer将异构信号映射到一个256维的共享触觉潜在空间中。

    arXiv:2610.11945v1 Announce Type: cross  Abstract: Collecting tactile demonstrations on robots is costly and slow, motivating the use of lower-cost human tactile gloves for scalable data collection. However, human capacitive/piezoresistive gloves and robotic tactile sensors differ fundamentally in transduction principle, sensor layout, spatial resolution, and dynamic response, making alignment of raw sensor channels ill-posed. To address this problem, we present TACROSS, a scalable system for learning from human touch and transferring it to robots that bridges this heterogeneity by aligning tactile streams at the level of contact events rather than raw sensor values. The hardware component of TACROSS integrates a piezoresistive glove with five layers and a cost of USD 10.86 with 285 sensing points. To align contact semantics, we design canonicalizers and residual adapters that map heterogeneous signals into a shared tactile latent with 256 dimensions via a temporal Transformer with att
    
[^85]: 重探媒体桥接时间序列预测中的身份与频谱弥散：连接多变量信号与叙事流

    Revisiting Identity and Spectra Dispersion in Media-Bridged Time Series Forecasting: Linking Multivariate Signals and Narrative Flows

    [https://arxiv.org/abs/2610.11924](https://arxiv.org/abs/2610.11924)

    本文提出统一的多媒体身份感知棱镜网络（MIDAPN），通过媒体通用图适配（MIDAG与CIM）和频谱棱镜卷积构建跨范式时空预测主干，从而将多变量数值信号与叙事流文本桥接于同一时间序列预测框架中。

    

    媒体桥接时间序列预测正不断扩展，以同时涵盖传统的“多变量”设置和新兴的“多模态”设置（例如通过文本辅助）。现有时间序列预测（TSF）模型仍然依赖特定范式的关系、融合与时间模块，这阻碍了在数值设置与预对齐叙事流设置之间共享统一预测主干的可能性。为探索这一问题，我们提出了多媒体身份感知棱镜网络（MIDAPN），这是一个基于媒体通用图适配与自动时间学习的统一时空预测主干：（1）在媒体预对齐之后，我们的多媒体身份感知图（MIDAG）从静态本质、动态行为和潜在共性三个层面重新审视身份，诱导出可将变量特定依赖关系扩展至跨媒体的亲和性；上下文身份调制（CIM）进一步细化了判别性聚合。（2）我们开发了频谱棱镜卷积（SPConv）以自动……

    arXiv:2610.11924v1 Announce Type: cross  Abstract: Media-bridged time series forecasting is expanding to encompass traditional "multivariate" and emerging "multimodal" (e.g., through textual assistance). Existing Time Series Forecasting (TSF) models still rely on paradigm-specific relation, fusion, and temporal modules, hindering a common forecasting backbone across numerical and pre-aligned narrative-flow settings. To explore this, we propose the Multimedia Identity-Aware Prism Network (MIDAPN), a unified spatiotemporal forecasting backbone based on media-general graph adaptation and automatic temporal learning: (1) Following media pre-alignment, our Multimedia Identity-Aware Graph (MIDAG) revisits identity through static essence, dynamic behavior, and latent commonality, inducing affinities that extend variable-specific dependencies across media. Contextual Identity Modulation (CIM) further refines discriminative aggregation. (2) We develop Spectral Prism Convolution (SPConv) to auto
    
[^86]: Puffin：从粗粒度观测中概率学习空间细节

    Puffin: Probabilistic Learning of Spatial Detail From Coarse Observations

    [https://arxiv.org/abs/2610.11914](https://arxiv.org/abs/2610.11914)

    提出了Puffin——一个统计降尺度概率框架，以高分辨率卫星嵌入为协变量、通过聚合感知的似然函数学习概率分布，在推理时以观测到的区域总量为条件将数据分解到子区域，无需精细分辨率标签即可生成与总量一致且不确定性经过校准的精细尺度估计。

    

    高分辨率社会经济变量对于城市规划、公共卫生、灾害响应和资源配置等应用十分重要。然而在实践中，这些变量通常只能在较粗的空间分辨率下被观测到。我们提出了Puffin，一个用于统计降尺度的概率框架，它利用高分辨率卫星嵌入作为协变量来提升粗粒度总量数据的分辨率。Puffin并不为每个精细分辨率的子区域预测单一数值，而是学习一个概率分布，并通过感知聚合的似然函数进行训练。在推理阶段，Puffin以观测到的区域总量为条件对这些预测进行约束，并将其分配到各个子区域中。由此得到的精细尺度估计与观测到的聚合总量保持一致，并附带校准的不确定性，且训练过程无需精细分辨率的标签。我们在德国和美国的普查、就业及选举等数据上对Puffin进行了评估。

    arXiv:2610.11914v1 Announce Type: new  Abstract: High-resolution socioeconomic variables are important for applications such as urban planning, public health, disaster response, and resource allocation. In practice, however, these variables are often observed only at a coarse spatial resolution. We introduce Puffin, a probabilistic framework for statistical disaggregation that raises the resolution of coarse totals using high-resolution satellite embeddings as covariates. Instead of predicting a single value for each fine-resolution subregion, Puffin learns a probability distribution and is trained through an aggregation-aware likelihood. At inference, Puffin conditions these predictions on the observed regional total and splits it among the subregions. The resulting fine-scale estimates are consistent with the observed aggregate and come with calibrated uncertainty, without requiring fine-resolution labels for training. We evaluate Puffin on German and US census, employment, and elect
    
[^87]: 面向模型市场的成本感知混合专家协调机制

    Cost-Aware Mixture-of-Experts Coordination for Model Markets

    [https://arxiv.org/abs/2610.11908](https://arxiv.org/abs/2610.11908)

    本文将混合专家从模型级学习架构提升为市场级协调机制，提出成本感知的门控机制和成本调整的收入分配规则，使模型市场能够协调异构专家并提供复合模型服务。

    

    现有的模型市场通常将单个模型作为不可分割的单元进行交易和选择，这限制了其利用异构专家之间互补性的能力。本文提出了一种基于MoE的模型市场框架，将混合专家从模型级学习架构提升为市场级协调机制。在该框架中，经纪人使用门控网络协调多个异构专家，并提供复合模型服务。我们形式化了市场参与者、服务流程、专家成本结构，以及结合预测效用与异构执行成本的福利目标。随后，我们推导出成本感知的门控机制和市场感知的训练目标，并引入一种成本调整的收入分配规则，根据实际专家参与度和执行成本来分配剩余收入。我们还建立了该分配规则的基本理论性质。

    arXiv:2610.11908v1 Announce Type: cross  Abstract: Existing model marketplaces typically trade and select individual models as indivisible units, limiting their ability to exploit complementarities among heterogeneous experts. This paper proposes an MoE-based model market framework that lifts Mixture-of-Experts from a model-level learning architecture to a market-level coordination mechanism. In this framework, brokers use gating networks to coordinate multiple heterogeneous experts and deliver a composite model service. We formalize the market participants, service workflow, expert cost structure, and a welfare objective that combines predictive utility with heterogeneous execution costs. We then derive a cost-aware gating mechanism and market-aware training objective, and introduce a cost-adjusted revenue allocation rule that distributes residual revenue according to realized expert participation and execution cost. We also establish basic theoretical properties of the allocation rul
    
[^88]: RobustLDS：在对抗性污染下学习线性动力系统

    RobustLDS: Learning linear dynamical systems under adversarial corruptions

    [https://arxiv.org/abs/2610.11906](https://arxiv.org/abs/2610.11906)

    该论文提出了基于最小截断二乘法松弛与离群值组稀疏性的估计器，用于在对抗性污染下从单条轨迹学习线性动力系统，并通过非渐近误差界证明了其对离群值的鲁棒性。

    

    我们研究了从长度为 $T$ 的单条轨迹中、在对抗性污染下学习线性动力系统的问题。尽管线性动力系统的辨识本身已被广泛研究，但在对抗性污染下的鲁棒系统辨识问题却相对较少被探索。在这项工作中，我们研究了 $T$ 个观测值中有一部分被对抗性离群值污染的设定。我们提出了基于最小截断二乘法（least-trimmed squares）松弛的不同估计器，并配合一种交替最小化算法。此外，我们还提出了两个利用离群值组稀疏性的估计器（分别通过惩罚项和硬约束实现）。对于带组稀疏惩罚的估计器，我们推导了非渐近误差界，证明了其对离群值的鲁棒性。我们还通过实验证明所提出的估计器在实践中表现良好。

    arXiv:2610.11906v1 Announce Type: cross  Abstract: We consider the problem of learning linear dynamical systems under adversarial contamination from a single trajectory of length $T$. While identification of linear dynamical systems itself is well-studied, the problem of robust system identification under adversarial contamination is relatively less explored. In this work, we study the setting where a fraction of the $T$ observations are contaminated by adversarial outliers. We propose different estimators based on relaxations of least-trimmed squares along with an alternating minimization algorithm. Furthermore, we also propose two estimators which exploit the group-sparsity (through penalization/hard-constraints) of the outliers. For the estimator with group-sparse penalty, we derive non-asymptotic error bounds which establish its robustness to outliers. We also show empirically that the proposed estimators work well in practice.
    
[^89]: 基于接地大语言模型的CAD模型自动化装配指令生成：一种人在回路框架

    Automated Assembly Instruction Generation from CAD Models Using Grounded Large Language Models: A Human-in-the-Loop Framework

    [https://arxiv.org/abs/2610.11896](https://arxiv.org/abs/2610.11896)

    该论文提出一种人在回路的框架，将CAD装配模型映射为结构化的ProductGraph中间表示，利用大语言模型自动生成受工程信息约束的自然语言装配指令，并通过人工审核解决所有质量标记后才允许导出文档。

    

    装配文档是一种下游制造产物，目前通常仍需通过人工解读CAD模型来编写。结构化产品数据和大语言模型如今都已具备，但关于CAD解读、装配序列规划、指令编写和人工监督的研究在很大程度上是彼此独立进行的。本文提出了“基于CAD的装配指令生成”这一任务：即在从CAD模型中提取的结构化工程信息的约束下，生成自然语言形式的装配流程。所提出的框架将STEP装配体映射为类型化的ProductGraph中间表示，通过确定性拓扑排序推导出操作先后顺序，将每个步骤实现为仅以所选图上下文为条件的语言描述，附加每一步的视觉文档，并应用基于规则和模型辅助的质量检查。在人工审核员解决所有质量标记之前，PDF导出功能将保持禁用状态。

    arXiv:2610.11896v1 Announce Type: new  Abstract: Assembly documentation is a downstream manufacturing artifact that is still usually authored by interpreting CAD models by hand. Structured product data and large language models are both available, yet studies of CAD interpretation, assembly sequence planning, instruction writing, and human oversight have largely proceeded separately. This paper formulates CAD-grounded assembly instruction generation: the production of natural-language assembly procedures constrained by structured engineering information extracted from CAD models. The proposed framework maps a STEP assembly to a typed ProductGraph intermediate representation, derives a precedence order by deterministic topological sorting, realizes each step as language conditioned only on selected graph context, attaches per-step visual documentation, and applies rule-based and model-assisted checks. PDF export remains disabled until a human reviewer resolves every quality flag. The ca
    
[^90]: 从缺失观测中学习结构化线性动力系统

    Learning structured linear dynamical systems from missing observations

    [https://arxiv.org/abs/2610.11869](https://arxiv.org/abs/2610.11869)

    本文提出一种基于偏差校正目标函数的估计器，用于在观测严重缺失的情况下学习凸集约束下的结构化线性动力系统，给出了依赖集合局部复杂度、轨迹长度和采样概率的非渐近误差界，并证明即使轨迹远短于无约束情形且采样概率趋于零时仍能有意义地恢复转移矩阵。

    

    我们研究在凸集 $\mathcal{K}$ 上学习结构化线性动力系统的问题，其中每个时间点只有一小部分观测是可用的。我们提出了一种估计器，该估计器最小化一个经过偏差校正的、可能是非凸的目标函数。我们获得了统计误差的非渐近界，该界取决于 $\mathcal{K}$ 的局部复杂度、轨迹长度 $T$ 以及子采样概率 $p$。此外，我们还建立了投影梯度下降算法的收敛性。该一般理论被应用于以下三种情形：(i) $\mathcal{K}$ 是一个子空间；(ii) $\mathcal{K}$ 是双等张（bi-isotonic）矩阵集合；(iii) $\mathcal{K}$ 是行由采样 Lipschitz 函数构成的矩阵集合。我们证明，即使轨迹长度 $T$ 远小于无约束情形下所需的值，且子采样概率 $p = o(1)$，仍然能够对转移矩阵实现有意义的恢复。

    arXiv:2610.11869v1 Announce Type: cross  Abstract: We consider the problem of learning structured linear dynamical systems over convex sets $\mathcal{K}$, where only a small subset of the observations are available at each time point. An estimator which minimizes a bias-corrected, potentially non-convex objective function is proposed. Non-asymptotic bounds are obtained for the statistical error, which depend on the local complexity of $\mathcal{K}$, the trajectory length $T$, and the sub-sampling probability $p$. Convergence of the projected gradient descent algorithm is also established. The general theory is applied to settings where (i) $\mathcal{K}$ is a subspace, (ii) $\mathcal{K}$ is the set of bi-isotonic matrices, and (iii) $\mathcal{K}$ is the set of matrices whose rows are formed by sampling Lipschitz functions. We show meaningful recovery of the transition matrix is possible for values of $T$ much smaller than what is required in the unconstrained case, and for $p = o(1)$.
    
[^91]: 通过谱可靠性理解动力系统学习中的潜在维度缩放

    Understanding Latent-Dimension Scaling in Dynamical-System Learning through Spectral Reliability

    [https://arxiv.org/abs/2610.11866](https://arxiv.org/abs/2610.11866)

    本文提出用Koopman算子的谱可靠性（通过相对残差检测伪特征对）来解释动力系统学习中增大潜在维度为何能持续降低滚动预测误差，并从理论上证明了所学字典空间中的最小残差随空间逼近可观测空间而逐点收敛于全空间结果。

    

    在深度学习中，逼近理论促使人们不断增大表示的规模。我们探讨这一优势是否同样适用于通过自回归预测进行的学习。我们通过Koopman算子的特征结构来分析所学到的系统演化，并利用相对残差来检测伪特征对——即使单步预测误差下降，伪特征对仍可能出现。对于有界Koopman算子，我们证明了当所学字典空间在 $L^2$ 意义下逼近可观测空间时，这些空间上的最小残差会逐点收敛到其全空间对应的结果。我们的假设是，Koopman谱可靠性有助于解释为什么随着维度增加，滚动预测误差能够持续一致地下降。我们比较了基于共享Koopman自编码器的两种模型，它们交替进行重构和潜在演化的训练，分别使用潜在预测损失（潜在坐标下的单步预测误差）或谱残差损失（候选特征对的相对残差）。

    arXiv:2610.11866v1 Announce Type: new  Abstract: In deep learning, approximation theory motivates increasing representation size. We ask whether this benefit extends to dynamics learning through autoregressive prediction. We analyze the learned time evolution through the eigenstructure of Koopman operators, using relative residuals to detect spurious eigenpairs arising even as one-step error falls. For bounded Koopman operators, we show that minimal residuals over learned dictionary spaces converge pointwise to their full-space counterparts as these spaces approximate the observable space in $L^2$. Our hypothesis is that Koopman spectral reliability helps explain how consistently rollout error decreases with increasing dimension. We compare two models of a shared Koopman autoencoder trained alternately for reconstruction and latent evolution, using latent-prediction loss (one-step prediction errors in latent coordinates) or spectral-residual loss (relative residuals of candidate eigenp
    
[^92]: 条件核斯坦因差异

    Conditional Kernel Stein Discrepancy

    [https://arxiv.org/abs/2610.11863](https://arxiv.org/abs/2610.11863)

    提出了一个通过协变量空间上的算子值核将核斯坦因差异推广到条件设定的框架，用于在仅知道非归一化条件目标模型和联合分布样本的情况下量化条件拟合优度。

    

    核斯坦因差异为比较分布提供了一种通用的工具。其主要应用之一是量化数据生成分布与给定目标分布之间的拟合优度。在本工作中，我们研究了与之相关的条件拟合优度量化问题：在仅给定一个（可能是非归一化的）条件目标模型、不掌握其协变量分布的信息、但拥有来自联合分布的样本的情况下，目标是评估样本的条件分布与目标分布的匹配程度。为解决这一设定，我们提出了一个框架，通过协变量空间上的算子值核，将无条件的KSD提升到条件设定中，超越了已知的欧几里得情形。我们证明了，当且仅当条件模型与真实条件分布在几乎所有协变量处一致时，我们所提出的统计量才会为零。

    arXiv:2610.11863v1 Announce Type: cross  Abstract: Kernel Stein discrepancies (KSDs) provide a versatile tool for comparing distributions. One of their main applications is in quantifying the goodness-of-fit (GoF) between a data-generating distribution and a prescribed target distribution. In this work, we study the related problem of conditional GoF quantification: given only a (possibly non-normalized) conditional target model, without information on the distribution of its covariates, and samples from a joint distribution, the goal is to assess how well the conditional distribution of the samples matches the target. To tackle this setting, we present a framework that allows lifting unconditional KSDs to the conditional setting through an operator-valued kernel on the covariate space, going beyond the known Euclidean case. We establish that our suggested statistic vanishes if and only if the conditional model and the true conditional distribution agree for almost all covariates and d
    
[^93]: GRPODropout：在线强化学习轨迹采样中的“少即是多”

    GRPODropout: Less is More for Online Reinforcement Learning Rollouts

    [https://arxiv.org/abs/2610.11854](https://arxiv.org/abs/2610.11854)

    提出GRPODropout方法，通过在GRPO策略更新前选择性地移除少量高概率的正优势轨迹并重新居中保留的优势，有效缓解策略熵坍缩问题，提升大语言模型的推理能力。

    

    诸如GRPO之类的强化学习方法能够显著提升大语言模型的推理能力，但常常面临策略熵坍缩问题：采样多样性的丧失会削弱探索能力，并限制模型的进一步提升。现有方法要么通过算法层面的干预来应对这一问题，例如修改奖励和使用熵/KL正则化，要么通过token级别的重新加权。我们从一个互补的角度进行研究：熵坍缩也可以通过改变哪些生成的轨迹参与策略更新来缓解。在相同的采样预算下，并非所有轨迹都对更新有正向贡献，选择性地排除其中一部分反而能够改善学习效果。为此，我们提出了GRPODropout：在标准更新之前，我们采用一种简单的策略，选择性地移除少量高概率的正优势轨迹，并对所保留的优势进行重新居中。为了论证这一设计，我们…

    arXiv:2610.11854v1 Announce Type: cross  Abstract: Reinforcement learning (RL) methods such as GRPO substantially improve large language model reasoning but often suffer from policy entropy collapse: the loss of sampling diversity weakens exploration and limits further improvement. Existing methods address this issue either through algorithm-level interventions, such as reward modification and entropy/KL regularization, or through token-level reweighting. We investigate a complementary perspective: entropy collapse can also be mitigated by changing which generated rollouts contribute to policy updates. Under the same sampling budget, not all rollouts contribute positively to an update, and selectively excluding some can improve learning. To address this, we propose GRPODropout: before the standard update, we use a simple strategy that selectively removes a small number of high-probability positive-advantage rollouts and recenters the retained advantages. To motivate this design, we dev
    
[^94]: DADP：动态活动依赖剪枝，一种受反向赫布法则启发的结构化剪枝方法

    DADP: Dynamic Activity-Dependent Pruning, A Reverse Hebbian-Inspired Structural Pruning Method

    [https://arxiv.org/abs/2610.11853](https://arxiv.org/abs/2610.11853)

    本文提出受反向赫布法则启发的DADP结构化剪枝方法，通过突触前激活与突触后误差梯度乘积的累积来衡量连接重要性，并利用单一全局阈值在训练中动态分配各层稀疏度，无需手动设置每层剪枝目标即可匹配或超越现有剪枝方法。

    

    现代神经网络存在严重的过参数化问题，这种冗余在训练和推理过程中带来了巨大的计算和内存开销。现有的剪枝方法依赖于事后幅度阈值或静态初始化启发式方法，因此通常需要手动设置每层的稀疏度目标或进行昂贵的重训练周期。我们提出了动态活动依赖剪枝（DADP），这是一种受生物学启发的结构可塑性机制。在训练过程中，DADP通过突触前激活与突触后误差梯度的累积乘积来衡量连接的重要性。DADP使用单一全局阈值而非固定的层预算，在网络深度上动态分配稀疏度，同时自然地诱导神经元级别和通道级别的剪枝。在MLP、VGG-16、ResNet-18、BiLSTM-CRF和MiniBERT等多种架构上，DADP匹配或超越了Magnitude、SNIP和RigL等剪枝方法，保留了73.67%的准确率（稠密基线为……原文截断）。

    arXiv:2610.11853v1 Announce Type: new  Abstract: Modern neural networks are heavily over-parameterized. This redundancy incurs substantial compute and memory overhead during training and inference. Existing pruning methods rely on post-hoc magnitude thresholds or static initialization heuristics. Consequently, they often require manual per-layer sparsity targets or expensive retraining cycles. We propose Dynamic Activity-Dependent Pruning (DADP), a biologically inspired structural plasticity mechanism. During training, DADP measures connection importance via the accumulated product of pre-synaptic activations and post-synaptic error gradients. Using a single global threshold instead of fixed layer budgets, DADP dynamically allocates sparsity across network depth while naturally inducing neuron- and channel-level pruning. Across MLP, VGG-16, ResNet-18, BiLSTM-CRF, and MiniBERT architectures, DADP matches or outperforms Magnitude, SNIP and RigL, retaining 73.67% accuracy (dense baseline:
    
[^95]: 基于复数值融合的开放词汇音视频事件定位

    Open-Vocabulary Audio-Visual Event Localization via Complex-Valued Fusion

    [https://arxiv.org/abs/2610.11846](https://arxiv.org/abs/2610.11846)

    该论文提出用复数值相似度并结合复数值神经网络来学习视觉与音频两个模态相似度的融合，取代传统固定规则（几何平均或加权平均），从而提升开放词汇音视频事件定位的性能。

    

    开放词汇音视频事件定位（OV-AVEL）任务是为每个视频片段标注一个事件类别，其中包括训练阶段从未见过的新类别。当前主流的方法流程是使用冻结的多模态基础模型（如 ImageBind）将视觉帧、音频梅尔频谱图以及每个候选类别名称嵌入到共享空间中，然后针对每个片段与每个类别分别计算两个余弦相似度：视觉-文本相似度和音频-文本相似度。现有方法随后采用固定规则（几何平均、加权平均）将这一对相似度压缩为单个标量分数，再进行 argmax 操作。与此不同，我们计算复数值相似度，并使用复数值神经网络（CVNN）来学习它们的融合。每个模态的标准表示构成我们流程的实部，而一个配对的伴随流提供虚部。我们使用 iHSV 的虚部作为视觉模态的虚部，并将 CycleGAN 转换得到的相位谱用于音频……

    arXiv:2610.11846v1 Announce Type: cross  Abstract: Open-Vocabulary Audio-Visual Event Localization (OV-AVEL) labels each video segment with an event class, including classes that were never seen during training. The dominant pipeline uses a frozen multimodal foundation model (e.g. ImageBind) to embed the visual frame, the audio mel-spectrogram, and each candidate class name into a shared space, then computes two cosine similarities for each segment against each class: visual-text and audio-text. Existing methods then collapse this pair into a single scalar score with a fixed rule (geometric mean, weighted average) before taking the argmax. Instead, we compute complex-valued similarities and learn their fusion using a complex-valued neural network (CVNN). Each modality's standard representation becomes the real part of our pipeline, and a paired companion stream supplies the imaginary part. We use imaginary part of iHSV for visual modality and CycleGAN-translated phase spectrogram for a
    
[^96]: 单比特压缩感知后验采样的恢复保证

    Recovery Guarantees for Posterior Sampling of One-Bit Compressed Sensing

    [https://arxiv.org/abs/2610.11834](https://arxiv.org/abs/2610.11834)

    该论文用近似覆盖数刻画先验分布的复杂度，证明后验采样在测量数随覆盖数对数缩放时可高概率精确恢复，该上界对Wasserstein距离意义下的学习先验失配具有鲁棒性，并给出了几乎匹配的样本复杂度下界。

    

    我们研究了从先验分布中抽取的信号在含噪单比特压缩感知中的样本复杂度。通过用近似覆盖数刻画先验的有效分布复杂度，我们证明当测量数量按近似覆盖数的对数（至多相差一个单比特分离间隙因子）缩放时，后验采样能够以高概率实现精确恢复。该上界对学习先验的失配具有鲁棒性。具体而言，我们表明只要学习到的先验分布在Wasserstein距离上与真实信号分布足够接近，使用近似先验的后验采样依然保持可靠。此外，我们为任何可靠的含噪单比特压缩感知方法建立了样本复杂度下界，表明我们的上界在其主要的先验相关项上几乎是紧的。为了逼近理想的后验采样过程……

    arXiv:2610.11834v1 Announce Type: new  Abstract: We study the sample complexity of noisy one-bit compressed sensing for signals drawn from a prior distribution. By characterizing the effective distributional complexity of the prior via its approximate covering number, we prove that posterior sampling achieves accurate recovery with high probability when the number of measurements scales with the logarithm of the approximate covering number, up to a one-bit separation gap factor. This upper bound is robust to learned prior mismatch. Specifically, we show that posterior sampling with an approximate prior remains reliable, provided that the learned prior distribution is sufficiently close to the true signal distribution in Wasserstein distance. In addition, we establish a sample complexity lower bound for any reliable method of noisy one-bit compressed sensing, showing that our upper bound is nearly matched in its main prior dependent term. To approximate the ideal posterior sampling proc
    
[^97]: 用于清醒开颅术中跨说话者构音障碍检测的自监督语音表征

    Self-Supervised Speech Representations for Cross-Speaker Dysarthria Detection During Awake Craniotomy

    [https://arxiv.org/abs/2610.11825](https://arxiv.org/abs/2610.11825)

    该研究提出了一种融合自监督语音表征（wav2vec 2.0）、说话人条件归一化和级联分类器的系统流水线，实现了在手术室嘈杂环境下跨说话者的清醒开颅术中构音障碍语音自动检测。

    

    摘要：在清醒开颅手术中检测术中言语功能障碍对于保护语言功能至关重要。然而，自动化检测仍面临挑战，因为手术室录音包含大量声学干扰、临床相关的言语事件稀少，且可用队列规模小、说话者之间具有异质性。本研究对DATABRASE清醒开颅录音语料库中区分构音障碍语音与正常语音的流水线进行了系统的逐组件评估。该流水线包含：用于分离患者语音的说话人分离技术、结合手工声学描述符与多层wav2vec 2.0嵌入的多视图表示、为提高跨说话者鲁棒性而采用的说话人条件归一化和基于可迁移性的特征选择，以及由梯度提升第一级和神经网络第二级组成的级联分类器。（原文摘要在此处被截断）

    arXiv:2610.11825v1 Announce Type: new  Abstract: Detecting intra-operative speech impairment during awake craniotomy is essential for preserving language function. However, automated detection remains challenging because operating-room recordings contain substantial acoustic interference, clinically relevant speech events are rare, and available cohorts are small and heterogeneous across speakers. This study presents a systematic component-wise evaluation of a pipeline for distinguishing dysarthric from no-trouble speech in the DATABRASE corpus of awake-craniotomy recordings. The pipeline incorporates speaker diarization to isolate patient speech, a multi-view representation combining handcrafted acoustic descriptors with multilayer wav2vec 2.0 embeddings, speaker-conditional normalization and transferability-based feature selection to improve cross-speaker robustness, and a cascaded classifier comprising a gradient-boosted first stage and a neural second stage. Evaluation was conducte
    
[^98]: 高斯混合分布上的Softmax注意力：能线性时则线性，需选择时方选择

    Softmax Attention on Gaussian Mixtures: Linear When It Can, Selective When It Must

    [https://arxiv.org/abs/2610.11798](https://arxiv.org/abs/2610.11798)

    该论文通过研究softmax注意力在高斯混合分布上的无穷提示极限，证明softmax注意力既能像线性注意力一样有效解决线性任务，又能借助查询依赖的选择能力，以梯度方法学习到监督分类、去噪等具有潜在结构、多峰性和非线性依赖的统计任务的最优解。

    

    Softmax注意力作为Transformer的核心，已展现出卓越的能力，然而其底层机制仍未被完全理解。近期的理论研究针对高斯提示展开，在无穷提示极限下，softmax注意力退化为一个线性映射，但这也消除了它区别于线性注意力的查询依赖选择能力。本研究探讨softmax注意力在高斯混合分布上的无穷提示极限，高斯混合分布在保留高斯数据可处理性的同时，引入了潜在结构、多峰性和非线性依赖。我们证明softmax注意力能够通过基于梯度的方法表示并学习一系列统计任务（包括监督分类和去噪）的最优解。我们的结果凸显了softmax注意力的两种互补能力：它能够像其更简单的线性对应方法一样有效地完成线性任务，同时还能利用查询依赖的选择能力来应对更复杂的分布。

    arXiv:2610.11798v1 Announce Type: cross  Abstract: Softmax attention, at the heart of Transformers, has demonstrated remarkable capabilities. Yet its underlying mechanisms remain only partially understood. Recent theoretical work studies Gaussian prompts, where the infinite-prompt limit reduces softmax attention to a linear map, but also removes the query-dependent selection that distinguishes it from linear attention. This work studies the infinite-prompt limit of softmax attention on Gaussian mixtures, which retain the tractability of Gaussian data while introducing latent structure, multimodality, and nonlinear dependencies. We show that softmax attention can represent and learn, via gradient-based methods, optimal solutions to a range of statistical tasks, including supervised classification and denoising. Our results highlight two complementary capabilities of softmax attention: it can recover linear tasks as effectively as its simpler linear counterpart, while also exploiting que
    
[^99]: Memento 3：通过反思式规则手册实现基于模型的递归自我改进

    Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks

    [https://arxiv.org/abs/2610.11794](https://arxiv.org/abs/2610.11794)

    Memento 3 让冻结参数的 LLM 智能体将可修正的环境假设记录为自然语言“规则手册”并编译为可执行代码，通过预测误差驱动的观察—反思—修订—编译—验证循环，实现显式世界模型的持续递归自我改进。

    

    arXiv:2610.11794v1 公告类型：新论文 摘要：学习在不熟悉的环境中行动，要求智能体推断世界的运作方式，并随着新证据的到来不断修正这一理解。然而，有限的观测数据可以支持多个世界模型，它们都能解释过去的交互，却在未见过的状态下预测出不同的结果。我们提出了 Memento 3，它在 Memento 系列的基础上，使冻结参数的大语言模型（LLM）智能体能够借助外部记忆持续学习显式的世界模型。该智能体维护一份自然语言规则手册作为持久的语义记忆，记录关于环境动态的可修正假设，同时将未知的方面保持为未指定状态。智能体将这份规则手册编译为可执行代码，用于预测与规划。通过观察、反思、规则修订、编译与验证的持续循环，智能体利用预测误差来同时改进规则手册及其代码。更新后的代码只有在 LLM 判定其忠实于规则手册时才会被接受……（原文摘要在此处截断）

    arXiv:2610.11794v1 Announce Type: new  Abstract: Learning to act in unfamiliar environments requires agents to infer how the world works and revise that understanding as new evidence arrives. Yet limited observations can support multiple world models that explain past interactions but predict different outcomes in unseen states. We introduce Memento 3, building on the Memento series to enable frozen LLM agents to continually learn explicit world models through external memory. The agent maintains a natural-language rulebook as persistent semantic memory, recording revisable hypotheses about environment dynamics while leaving unknown aspects underspecified. It compiles this rulebook into executable code for prediction and planning. Through a continual loop of observation, reflection, rule revision, compilation, and verification, the agent uses prediction errors to refine both the rulebook and its code. Updated code is accepted only when the LLM judges it faithful to the rulebook and cel
    
[^100]: 编译表格：面向表格上下文学习的查询校准算子压缩

    Compile the Table: Query-Calibrated Operator Compression for Tabular In-Context Learning

    [https://arxiv.org/abs/2610.11784](https://arxiv.org/abs/2610.11784)

    提出QCOC方法，将表格上下文学习的完整KV缓存一次性编译为查询校准的联合KV原型并在查询间共享，在不牺牲准确率的情况下显著提升预测吞吐量。

    

    表格上下文学习（ICL）已成为一种无需训练且准确的表格预测范式，但其上下文示例的现有压缩方法面临准确率与吞吐量之间的权衡：固定子集可能牺牲准确率，而针对特定查询的检索限制了缓存的跨查询复用与批处理，从而降低吞吐量。我们提出QCOC（查询校准算子压缩），利用上下文示例的可交换性和重复使用特性，将其完整的KV缓存一次性编译为紧凑的共享内存，供后续查询共同使用。QCOC不再保留原始示例，而是将其状态聚类为联合KV原型，保留每个聚类的多重性和原始示例数量，并通过带锚定的闭式解，将原型值与由上下文示例产生的注意力查询向量进行校准。原型压缩带来了加速，而值拟合有助于保持准确率。

    arXiv:2610.11784v1 Announce Type: new  Abstract: Tabular in-context learning (ICL) has emerged as a training-free and accurate paradigm for tabular prediction, but current approaches to compressing its in-context examples face an accuracy-throughput tradeoff: fixed subsets can sacrifice accuracy, while query-specific retrieval limits cache reuse and batching across queries, reducing throughput. We propose QCOC (Query-Calibrated Operator Compression), which exploits the exchangeability and repeated use of in-context examples by compiling their full KV cache once into compact memory shared across subsequent queries. Instead of retaining raw examples, QCOC clusters their states into joint-KV prototypes, preserves per-cluster multiplicities and the original example count, and calibrates prototype values against attention query vectors produced by the in-context examples through an anchored closed-form solution. Prototype compression drives the speedup, while value fitting helps preserve ac
    
[^101]: 骑行中电动滑板车骑行者酒精损害检测及误报控制

    In-Ride Alcohol-Impairment Detection in E-Scooterists with False-Alarm Control

    [https://arxiv.org/abs/2610.11783](https://arxiv.org/abs/2610.11783)

    本文提出一种利用车载惯性及油门传感器在电动滑板车骑行过程中实时检测骑行者酒精损害的新方法，并提供了可证明的误报率界限。

    

    共享电动滑板车服务已成为一种被广泛采用的城市交通方式。虽然大多数用户都能负责任地骑行，但酒精中毒是导致严重事故的最突出因素之一。然而，现有的应对措施仍仅限于单点反应测试和全面暂停服务的夜间禁令。本文提出了一种新方法：在行程进行过程中，由车载传感器对骑行者进行评估，一旦积累了足够的损害证据便立即发出警报。具体而言，我们提出了一种基于惯性和油门测量数据的检测器，并具有可证明的误报率界限。基于141次骑行的传感器数据实验（其中25名参与者分别在清醒状态及两种目标血液酒精浓度水平下骑行）证实该界限成立，而基线方法和消融模型要么超出该界限，要么损失检测性能，某些情况下还会延迟警报。在界限为0.023时，该检测器……

    arXiv:2610.11783v1 Announce Type: new  Abstract: Shared e-scooter services have become a widely adopted urban transport mode. While most users ride responsibly, alcohol intoxication stands out among the factors contributing to severe crashes. Nonetheless, countermeasures remain limited to single-point reaction tests and night bans that suspend the service altogether. This paper proposes a new approach in which onboard sensors evaluate the rider as the trip unfolds, raising an alarm as soon as enough evidence of impairment has accumulated. Specifically, we introduce a detector that operates on inertial and throttle measurements, with a provable bound on the rate of false alarms. Experiments on sensor data from 141 rides, in which 25 participants rode while sober and at two target blood alcohol concentration levels, confirm that the bound holds, whereas baselines and ablations either exceed it or lose detection performance, and in some cases delay the alarm. At a bound of 0.023, the dete
    
[^102]: RouterInterp：理解专家混合模型路由中的叠加特化

    RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing

    [https://arxiv.org/abs/2610.11775](https://arxiv.org/abs/2610.11775)

    本文提出叠加特化假说，认为MoE专家专精于细粒度特征的组合而非单一领域，并据此开发RouterInterp方法，通过稀疏自编码器特征与自然语言解释来解读专家路由，检测准确率比先前方法高出约65%。

    

    稀疏混合专家模型通过将词元路由到模块化的专家网络（这些网络仅在处理一小部分词元时被激活），比稠密模型更具扩展效率。关于MoE模型性能的一个主流假设是：每个专家专精于一个单一、连贯的领域。然而，基于这一假设的可解释性研究工作通常并不成功。我们提出并为另一种解释提供了证据，我们称之为叠加特化假说：专家专精于细粒度特征的不相交并集，而非某一宽泛领域。利用SSH，我们提出了RouterInterp，这是一种解释专家路由的方法，它能识别对路由决策最具预测性的稀疏自编码器特征，并生成统一的自然语言解释。在gpt-oss-20b上，RouterInterp解释专家路由的检测准确率比先前的基于词元统计的方法高出约65%。

    arXiv:2610.11775v1 Announce Type: new  Abstract: Sparse Mixture of Experts (MoE) models scale more efficiently than dense models by routing tokens to modular expert networks that are only active for processing a fraction of tokens. A leading hypothesis for the performance of MoE models is that each expert specialises in a single, coherent domain. However, interpretability efforts that assume this hypothesis have generally been unsuccessful. We propose and present evidence for an alternative account that we call the Superposed Specialisation Hypothesis (SSH): experts specialise in a disjoint union of fine-grained features rather than one broad domain. Leveraging the SSH, we introduce RouterInterp, a method for interpreting expert routing that identifies Sparse Autoencoder features most predictive of routing decisions and produces unified natural language explanations. On gpt-oss-20b, RouterInterp explains expert routing with ${\sim}65\%$ higher detection accuracy than prior token statis
    
[^103]: 球对称分布的最优随机量化器

    Optimal random quantisers for spherically symmetric distributions

    [https://arxiv.org/abs/2610.11772](https://arxiv.org/abs/2610.11772)

    该论文证明对于球对称目标分布，随机量化器的优化问题具有凸性，并据此发现均匀分布在适当半径球面上的随机量化器即使在中等样本量下也表现优异、被数值验证为全局最优，从而克服了Zador渐近理论中高维所需的天文数字级样本量问题。

    

    Zador的著名定理是最优量化理论的基石：它既确定了 $\mathbb{R}^d$ 中最优 $n$ 点量化器经验分布的弱极限，也给出了相应的 $L_s$ 平均量化误差的衰减速率。然而，在高维情形下，观测到这种渐近行为需要天文数字般庞大的样本量。我们证明，对于球对称目标分布，在所有球对称分布上进行优化是一个凸问题，并推导出一个等价定理，该定理既刻画了全局最优性，又给出了一个构造性算法。我们表明，对于中等规模的 $n$，均匀分布在半径适当选取的球面上的随机量化器表现异常出色，并且在广泛的 $n$ 取值范围内，经数值验证在所有随机量化器中是最优的。它们的期望失真具有一个可以计算的显式积分表示。

    arXiv:2610.11772v1 Announce Type: cross  Abstract: Zador's celebrated theorem is a cornerstone of optimal quantisation: it establishes both the weak limit of the empirical distribution of an optimal $n$-point quantiser in $R^d$ and the decay rate of the associated $L_s$-mean quantisation error. In large dimension, however, observing this asymptotic behaviour requires an astronomically large sample size. We prove that, for spherically symmetric target distributions, optimisation over all spherically symmetric distributions is a convex problem and derive an equivalence theorem that both characterises global optimality and yields a constructive algorithm. We show that, for moderate $n$, random quantisers uniformly distributed on a sphere of suitably chosen radius $R$ perform exceptionally well and, over a broad range of values of $n$, are numerically certified to be optimal among all random quantisers. Their expected distortion has an explicit integral representation that can be evaluated
    
[^104]: RAGenome：将基于检索的基因组语言模型扩展到长上下文

    RAGenome: Scaling Retrieval-Based Genomic Language Models to Long Contexts

    [https://arxiv.org/abs/2610.11761](https://arxiv.org/abs/2610.11761)

    RAGenome是首个基于检索的基因组语言模型，通过将预训练上下文扩展到现有基于MSA模型100倍的长度，突破了短输入限制，实现对长基因组序列的建模。

    

    基因组掌握着控制细胞生物学特性的蓝图。因此，增进我们对基因组功能的认识，对于更广泛地理解生物学以及持续的生物医学进步都至关重要。大型语言模型在自然语言和蛋白质序列上的成功激发了人们在基因组数据上的类似努力。然而，标准的基因组语言模型（gLMs）通常需要极其庞大的计算资源，并且在某些下游任务上仍然落后于传统方法。最近，基于MSA（多序列比对）的预训练被提出作为一种高效的替代方案，但现有模型受限于较短的输入上下文，使其只能用于短程任务，例如变异效应预测。在这项工作中，我们提出了RAGenome，这是首个基于检索的基因组语言模型，它将预训练扩展到更长的上下文（比现有基于MSA的基因组语言模型长100倍），使其能够捕获跨物种进化……

    arXiv:2610.11761v1 Announce Type: new  Abstract: The genome holds the blueprint that governs the biological properties of the cell. Consequently, advancing our knowledge of genomic function is crucial both for a broader understanding of biology and for continued biomedical advances. The success of large language models on natural language and protein sequences has motivated similar efforts on genomic data. However, standard genomic language models (gLMs) often require extremely large computational resources and still fall behind traditional methods on some downstream tasks. Recently, MSA-based pretraining has been proposed as an efficient alternative, but existing models are limited to short input contexts, restricting their use to short-range tasks, such as variant effect prediction. In this work, we present RAGenome, the first retrieval-based gLM that scales pretraining to longer contexts (100$\times$ longer than existing MSA-based gLMs), allowing it to capture both across-species ev
    
[^105]: 超越QAOA：人工智能与量子计算在自适应组合优化中的综述

    Beyond QAOA: A Review of AI and Quantum Computing for Adaptive Combinatorial Optimization

    [https://arxiv.org/abs/2610.11759](https://arxiv.org/abs/2610.11759)

    本综述提出“自适应量子优化”的概念，按被学习的决策类型系统梳理了AI赋能量子优化、量子赋能AI优化和AI-量子协同优化三种范式，并通过对119篇论文的分析发现AI辅助量子优化的证据显著强于量子辅助AI优化。

    

    近期的量子组合优化方法受到量子比特数量、电路保真度、采样成本以及约束编码难度的限制，而机器学习正日益被用于配置和控制量子优化工作流程。我们将这类工作流程称为“自适应的”：传统上预先固定的决策——从问题形式化和惩罚项设置到采样预算、后端选择，乃至是否调用量子处理器本身——都由学习到的策略来做出，这些策略能够响应问题实例、求解过程的进展或硬件状态。本综述考察了三种范式：AI赋能量子优化、量子赋能AI驱动的优化，以及AI-量子协同优化，并按照被学习的决策而非应用领域来组织文献。对119篇论文的结构化综述（其中67篇进行了详细编码分析）表明，AI辅助量子优化的证据明显强于相反方向。

    arXiv:2610.11759v1 Announce Type: cross  Abstract: Near-term quantum approaches to combinatorial optimization are limited by qubit counts, circuit fidelity, sampling cost, and the difficulty of encoding constraints, while machine learning is increasingly used to configure and control quantum optimization workflows. We call such workflows adaptive: decisions conventionally fixed in advance, from formulation and penalties to shot budgets, backends, and whether to invoke a quantum processor at all, are made by learned policies that respond to the instance, the progress of the solve, or the hardware. This review examines three paradigms, AI for quantum optimization, quantum for AI-driven optimization, and AI-quantum co-optimization, and organizes the literature by the decision being learned rather than by application. A structured review of 119 papers, 67 coded in detail, shows that the evidence is considerably stronger for AI-assisted quantum optimization than for the reverse direction: l
    
[^106]: SR-TTA：面向干扰鲁棒呼吸感知的空间冗余测试时自适应方法

    SR-TTA: Spatial-Redundancy Test-Time Adaptation for Interference-Robust Respiration Sensing

    [https://arxiv.org/abs/2610.11755](https://arxiv.org/abs/2610.11755)

    该论文提出SR-TTA方法，利用学习得到的复数加权波束成形器进行空间置零，并结合最大化随机天线子集间一致性的无标签测试时自适应，使无蜂窝大规模MIMO基站能在强带内干扰下依然实现鲁棒的呼吸感知。

    

    未来的6G网络旨在通过复用通信基础设施，将感知作为一项原生服务提供。我们研究了在无蜂窝大规模多输入多输出（MIMO）基站上的呼吸感知问题，其中需要将64天线信道融合为呼吸波形。最先进的手工设计融合方法在良好条件下接近最优，但在强带内运动干扰下会失效，因为该干扰的频率恰好落在呼吸频带内。我们证明，一个学习得到的复数加权波束成形器可以通过空间置零恢复呼吸信号，而与逐记录最优参考之间的剩余差距可以通过部署阶段的无标签测试时自适应来弥合。至关重要的是，我们识别出究竟哪种无标签信号能够实现这一目标：基于频率和方差的准则无法将带内干扰源与呼吸信号区分开来。我们提出的空间冗余测试时自适应（SR-TTA）通过最大化随机天线子集之间的一致性……（原文摘要在此处截断）

    arXiv:2610.11755v1 Announce Type: new  Abstract: Future 6G networks aim to expose sensing as a native service by reusing communication infrastructure. We study respiration sensing on a cell-free massive multiple-input multiple-output (MIMO) base station, where a 64-antenna channel must be fused into a breathing waveform. The state-of-the-art hand-crafted fusion is near-optimal in benign conditions. It collapses, however, under strong in-band motion interference, whose frequency falls inside the respiration band. We show that a learned complex-weight beamformer recovers respiration by spatial nulling, and that the remaining gap to a per-recording oracle can be closed at deployment by label-free test-time adaptation. Crucially, we identify which label-free signal makes this work. Frequency- and variance-based criteria cannot separate an in-band interferer from breathing. Our spatial-redundancy test-time adaptation (SR-TTA), which maximizes consistency across random antenna subsets under 
    
[^107]: DEX：面向能效MSDF神经网络推理的数位级提前退出机制

    DEX: Digit-Level Early Exit for Energy-Efficient MSDF Neural Network Inference

    [https://arxiv.org/abs/2610.11748](https://arxiv.org/abs/2610.11748)

    本文提出基于最高有效位优先（MSDF）算术的数位级提前退出加速器DEX，通过ReLU提前负数检测、仅符号决策、低位数跳过和校准剪枝四种运行时机制，在保证分割精度的前提下显著降低脑肿瘤分割U-Net推理的能耗。

    

    用于脑肿瘤分割的U-Net推理需要数十亿次乘加运算，这促使人们开发能够动态减少计算量的硬件，而不仅仅是依赖固定精度或静态模型压缩。最高有效位优先（MSDF）算术能够在计算过程中逐步暴露结果的高位数，从而在完整数值生成之前即可做出依赖于输出的决策。本文提出了一种面向量化U-Net分割的MSDF加速器，其采用两级分组处理单元，支持有符号INT8操作数和流内偏置累加。四种运行时机制直接作用于输出数位流：ReLU层中的精确提前负数检测（END）、分割头中的精确仅符号决策、经过校准的低位数跳过以及经过校准的剪枝。两种近似机制在精度约束下离线选择，而执行时只需轻量（原文截断）。

    arXiv:2610.11748v1 Announce Type: cross  Abstract: U-Net inference for brain-tumor segmentation requires billions of multiply-accumulate operations, motivating hardware that can reduce computation dynamically rather than relying only on fixed precision or static model compression. Most-significant-digit-first (MSDF) arithmetic exposes the leading digits of a result during computation, enabling output-dependent decisions before the full value is generated. This paper presents an MSDF accelerator for quantized U-Net segmentation with a two-stage grouped processing element supporting signed INT8 operands and in-stream bias accumulation. Four runtime mechanisms operate directly on the output digit stream: exact early negative detection (END) in ReLU layers, exact sign-only decision making in the segmentation head, calibrated low-order-digit skipping, and calibrated pruning. The two approximate mechanisms are selected offline under an accuracy constraint, while execution requires only light
    
[^108]: 球与盒子：叠加计算中的两种几何

    The Ball and the Box: Two Geometries of Computation in Superposition

    [https://arxiv.org/abs/2610.11744](https://arxiv.org/abs/2610.11744)

    该论文用球几何刻画期望误差阈值、用盒子几何刻画联合可靠性阈值，揭示了叠加神经表征计算布尔门时两种误差准则下的维度需求差距源于共享读取，并给出了合取、析取和多数门的显式维度阈值。

    

    神经表征可以编码的特征数量超过其自身维度，这种现象被称为叠加。我们研究了从此类表示中计算布尔门所需的维度。对于采用高斯随机字典和均匀随机稀疏布尔输入的单层阈值网络，我们在两种误差准则下推导出了精确的维度阈值。使期望误差计数趋近于零所需的维度，可能多于以高概率保证每个输出都正确所需的维度。共享读取解释了这一差距：罕见的情况可能同时产生大量误差。期望计数阈值具有球几何特性，而当门在所有特征元组上评估时，联合可靠性则呈现盒子几何特性。通过优化共享读出权重和偏置，我们为合取门、析取门和多数门给出了显式的阈值。对于成对合取，我们的分析还描述了阈值附近的过渡行为，与精确模拟结果一致。

    arXiv:2610.11744v1 Announce Type: new  Abstract: Neural representations can encode more features than they have dimensions, a phenomenon known as superposition. We study the dimension needed to compute Boolean gates from such representations. For a single threshold layer with a Gaussian random dictionary and uniformly random sparse Boolean inputs, we derive sharp dimension thresholds under two error criteria. A vanishing expected error count can require more dimensions than correctness of every output with high probability. Shared reads explain the gap: rare realizations can produce many errors at once. The expected-count threshold has ball geometry, while joint reliability has box geometry when a gate is evaluated on every feature tuple. Optimizing shared readout weights and biases gives explicit thresholds for conjunction, disjunction, and majority. For pairwise conjunction, the analysis also describes the transition near the threshold, in agreement with exact simulations.
    
[^109]: TraceRelay：滚动轨迹上的注意力对齐循环架构

    TraceRelay: Attention-Aligned Recurrence over Rolling Traces

    [https://arxiv.org/abs/2610.11743](https://arxiv.org/abs/2610.11743)

    TraceRelay通过注意力对齐的循环相位继承机制将持久表示分布在滚动低维轨迹上，在长度泛化任务中取得约99%的准确率，远超无继承机制的51%基线。

    

    我们提出TraceRelay，一种注意力对齐的循环架构，它将持久表示分布在低维轨迹的滚动序列上。局部右看注意力从低层表示生成增量；增量的传递被延迟，直到所有被关注的输入都进入因果过去。固定的加性相位递归累积这些延迟的增量，而左看注意力读取由此产生的残差增强流。基于步长的前缀和机制支持并行预填充和有界缓冲的延续生成。我们在等重复（Equal Repeats）、有界Dyck闭合类型预测和因果最高频（Most-Freq）生成任务上进行了36次小模型实验，每个设置使用三个随机种子。在训练长度256时，具有循环相位继承机制的Equal Repeats模型达到98.81-99.69%的准确率，而不具备继承机制的单独训练变体仅为50.73-51.63%，尽管后者接受了更多的更新次数。当长度超过训练长度时，准确率急剧下降。

    arXiv:2610.11743v1 Announce Type: new  Abstract: We present TraceRelay, an attention-aligned recurrent architecture that distributes persistent representations over a rolling sequence of low-dimensional traces. Local right looking attention forms increments from lower-layer representations; delivery is delayed until all attended inputs are in the causal past. A fixed additive phase recurrence accumulates the delayed increments, and left-looking attention reads the resulting residual augmented stream. A stride-wise prefix sum supports parallel prefill and bounded-buffer continuation. We study 36 small-model runs on Equal Repeats, bounded Dyck closing-type prediction, and causal Most-Freq generation, using three seeds per setting. At trained length 256, Equal Repeats models with recurrent phase inheritance reach 98.81-99.69% accuracy versus 50.73-51.63% for separately trained variants without inheritance, despite the latter receiving more updates. Accuracy drops sharply at lengths beyond
    
[^110]: 形态学神经网络的相关性训练方法

    Correlational Training of Morphological Neural Networks

    [https://arxiv.org/abs/2610.11740](https://arxiv.org/abs/2610.11740)

    提出了一种受乘性权重更新（MWU）启发的基于相关性的权重更新方法来训练形态学神经网络，在九个基准测试中的八个上取得最高达32.84个百分点的性能提升。

    

    神经网络通常使用一阶方法和反向传播进行训练。目前尚不清楚这种方法对于形态学层是否是最优的，因为形态学层的权重雅可比矩阵是稀疏的，由此产生的参数梯度可能较差。在这项工作中，我们提出了一种受乘性权重更新（MWU）方案启发的、用于形态学神经网络的新型权重更新方法。我们将每个形态学感知机视为对数空间中“从专家建议中学习”问题的一个实例，并使用一种基于相关性的奖励机制，该机制倾向于奖励那些与期望输出变化方向一致的输入，而无论是否有较强的梯度信号传导至其权重。我们通过将全连接层作为独立模型以及作为更大规模Transformer网络的一部分进行训练，对所提方法进行了实证评估。在九个基准测试中，相关性训练在其中的八个上取得了性能提升，最高提升达32.84个百分点，同时……

    arXiv:2610.11740v1 Announce Type: new  Abstract: Neural networks are typically trained using first-order methods and back-propagation. It is unclear whether this approach is optimal for morphological layers whose weight Jacobians are sparse and whose resulting parameter gradients can be poor. In this work, we propose a novel weight update method for morphological neural networks inspired from the Multiplicative Weights Update (MWU) scheme. We view each morphological perceptron as an instance of the learning from experts' advice problem in logarithmic space, and use a correlation-based reward that favors inputs aligned with the desired output change, regardless of whether a strong gradient signal has reached their weight. We empirically evaluate our approach by training fully connected layers both as stand-alone models and as parts of larger transformer networks. Across nine benchmarks, correlational training yields improvements on eight, by up to 32.84 percentage points, while substant
    
[^111]: 揭示并修复贝叶斯物理信息神经网络（B-PINNs）中的碰撞体偏倚

    Uncovering and Fixing Collider Bias in Bayesian PINNs

    [https://arxiv.org/abs/2610.11737](https://arxiv.org/abs/2610.11737)

    本文揭示贝叶斯物理信息神经网络中常用的碰撞体建模结构会在物理参数后验中引入严重的系统性偏倚，并提出改用“物理生成轨迹、轨迹生成观测”的分层链式模型来消除该偏倚，尽管这会带来更难的双重难解推断问题。

    

    贝叶斯物理信息神经网络是用于从稀疏或含噪观测中进行参数与状态推断的一种流行框架。它们通常通过一种碰撞体结构来表述：物理参数与轨迹参数被假设为先验独立，并通过施加在微分方程残差上以保证物理一致性的虚拟似然而耦合起来。我们证明这一建模选择可能在物理参数的后验分布中引发严重的系统性偏倚：即使先验恰好以真实参数为中心，所得后验也可能偏离真实值并在远离真实参数处集中。作为补救，我们提倡一种分层链式模型，其中物理过程生成轨迹，轨迹再生成观测。该链式模型不受这种后验偏倚的影响，但由于存在依赖于物理过程法向化常数（normalization constant），它带来了一个更困难的、所谓的双重难解推断问题。

    arXiv:2610.11737v1 Announce Type: cross  Abstract: Bayesian physics-informed neural networks (B-PINNs) are a popular framework for parameter and state inference from sparse or noisy observations. They are commonly formulated via a collider structure, in which physical and trajectory parameters are assumed to be a priori independent and become coupled through virtual likelihoods on differential-equation residuals that enforce physical consistency. We show that this modeling choice can induce severe systematic bias in the posterior over physical parameters: even when the prior is favorably centered on the ground-truth parameters, the resulting posterior can drift away and concentrate far from them. As a remedy, we advocate a hierarchical chain model in which physics generates trajectories, which in turn generate observations. The chain model does not suffer from this posterior bias, but it poses a harder, so-called doubly intractable, inference problem due to a physics-dependent normaliz
    
[^112]: Timer-M1：通过学习基元实现的多元时间序列基础模型

    Timer-M1: A Multivariate Time Series Foundation Model via Learning Primitives

    [https://arxiv.org/abs/2610.11734](https://arxiv.org/abs/2610.11734)

    Timer-M1通过学习跨领域共享的时序与关系基元，并采用基于基元的数据合成与预训练流程，构建了能够实现零样本预测的多元时间序列基础模型。

    

    我们提出了Timer-M1，一个通过学习基元来实现零样本预测的预训练多元时间序列基础模型。跨领域中，时间序列共享基本的时序和关系模式，称为基元，但这些基元在不同上下文中的表现和演化方式各不相同。尽管在零样本和任务通用预测方面已取得进展，现有基础模型在泛化到复杂真实世界场景时可能仍面临困难。为此，我们开发了一种基于基元的数据合成与预训练流程。该合成流程生成具有跨领域共享时序基元的序列，然后利用关系基元将真实序列和生成序列组装成多元样本。随后，通过为目标变量、仅过去协变量和已知未来协变量分配不同的通道角色，将样本组织成情节，确保模型在可预测的变量上进行优化。

    arXiv:2610.11734v1 Announce Type: cross  Abstract: We introduce Timer-M1, a pretrained multivariate time series foundation model that learns with primitives for zero-shot forecasting. Across domains, time series share elementary temporal and relational patterns, termed primitives, yet differ in how these primitives manifest and evolve across different contexts. Despite progress in zero-shot and task-general forecasting, existing foundation models may still struggle to generalize to complex real-world scenarios. To this end, we develop a primitive-based data synthesis and pretraining pipeline. The synthesis pipeline generates series with temporal primitives shared across domains and then assembles real and generated series into multivariate samples using relational primitives. Afterwards, samples are organized into episodes by assigning distinct channel roles as target variates, past-only covariates, and known-future covariates, ensuring that the model is optimized on predictable variat
    
[^113]: 谱权重衰减：在神经网络权重中诱导低秩结构

    Spectral Weight Decay: Inducing Low-Rank Structure in Neural Network Weights

    [https://arxiv.org/abs/2610.11730](https://arxiv.org/abs/2610.11730)

    提出谱权重衰减方法，通过加法式谱收缩诱导神经网络权重的低秩结构，在匹配验证损失下将LLaMA模型压缩率提升至1.89倍、GPU推理加速至1.18倍，并显著提高噪声标签下的测试准确率。

    

    标准权重衰减将每个权重矩阵视为向量，忽略了其谱结构。我们提出谱权重衰减，这是一种步后解耦的核范数更新方法，采用加法而非乘法的谱收缩。我们将该更新与近似近端下降联系起来，并表明其在接近秩亏缺时对更新顺序的敏感性可能超过传统ℓ2权重衰减。在参数量从1.24亿到5亿的LLaMA模型上，谱权重衰减降低了有效秩，并在匹配的验证损失下改善了SVD-LLM压缩效果。在5亿参数和4%失真预算下，它达到了1.89倍的压缩率和1.18倍的GPU推理加速，而标准权重衰减仅分别为1.14倍和1.01倍。在60%标签噪声下的固定步数训练中，与匹配的ℓ2正则化相比，它还将最终平均干净测试准确率提升了多达17.8个点（MNIST）和4.6个点。

    arXiv:2610.11730v1 Announce Type: new  Abstract: Standard weight decay treats each weight matrix as a vector and ignores its spectral structure. We introduce spectral weight decay, a post-step decoupled nuclear-norm update that applies additive rather than multiplicative spectral shrinkage. We connect the update to approximate proximal descent and show that its sensitivity to update order can exceed that of conventional $\ell_2$ weight decay near rank deficiency. Across LLaMA models with $124$M to $500$M parameters, spectral weight decay lowers effective rank and improves SVD-LLM compression at matched validation loss. At $500$M and a $4\%$ distortion budget, it reaches $1.89\times$ compression and $1.18\times$ GPU inference speedup, compared with $1.14\times$ and $1.01\times$ after standard weight decay. Under fixed-horizon training with $60\%$ label noise, it also improves final mean clean-test accuracy over matched $\ell_2$ regularization by up to $17.8$ points on MNIST and $4.6$ po
    
[^114]: 解决多模态模糊性下性别化经济模因推理中的过度断言问题

    Addressing Overcommitment in the Reasoning of Gendered Economic Memes under Multimodal Ambiguity

    [https://arxiv.org/abs/2610.11724](https://arxiv.org/abs/2610.11724)

    提出CGER-Net框架，通过评估情境证据充分性并采用证据门控推断，缓解多模态模糊模因中因认知过度断言导致的性别化经济角色刻板归因偏见。

    

    多模态模因（meme）理解正被越来越多地用于分析社会敏感内容，然而现有模型在模糊情境下解读经济依附关系与社会角色时，常常表现出有偏见的行为。许多模因通过稀疏的文本或符号化的视觉线索来表达经济关系，无法为性别化归因提供充分的证据。在这种信息不足的设定下，模型往往依赖预训练中的相关性，导致产生幻觉式的、刻板的经济角色分配。在本工作中，我们通过情境充分性的视角研究图像-文本模因中的性别化经济依附关系，并将认知过度断言——即在证据不充分的情况下推断角色——识别为偏见的主要来源。我们提出了CGER-Net，一个基于情境的多模态框架，它评估输入是否为性别化经济推理提供了充分的证据，并应用证据门控推断来实现有信心的归因。

    arXiv:2610.11724v1 Announce Type: new  Abstract: Multimodal meme understanding is increasingly used to analyze socially sensitive content, yet existing models often exhibit biased behavior when interpreting economic dependence and social roles under ambiguity. Many memes express economic relationships through sparse text or symbolic visual cues, providing insufficient evidence for gendered attribution. In such underspecified settings, models tend to rely on pretraining correlations, leading to hallucinated and stereotypical economic role assignments. In this work, we study gendered economic dependence in image-text memes through the lens of contextual sufficiency and identify epistemic overcommitment-inferring roles without adequate evidence-as a primary source of bias. We propose CGER-Net, a context-grounded multimodal framework that estimates whether the input provides sufficient evidence for gendered economic reasoning and applies evidence-gated inference to enable confident attribu
    
[^115]: 面向语义物理现实的四阶张量注意力模型

    4-Tensor Attention Model for Semantic Physical Reality

    [https://arxiv.org/abs/2610.11716](https://arxiv.org/abs/2610.11716)

    该论文提出一种四阶张量注意力模型，通过在语义纤维与时间上下文纤维上进行联合归一化注意力来预测场景的下一语义状态，在参数量几乎相同的情况下，其最后一句交叉熵在三个设置上均优于自由运行的一维 Transformer（最多低 5.3%），为视频生成和机器人规划提供了新方法。

    

    我们提出了一种四阶张量注意力模型（4-tensor attention model），用于预测场景的下一个语义状态，可应用于视频生成与机器人规划。一个状态窗口包含位置 (x, t) 以及两条纤维：语义纤维和时间上下文纤维，并由一个 softmax 在整个窗口上联合归一化注意力。视频帧和智能体的情境被表示为这些状态；编码器、渲染器和规划器则不参与该更新过程。为了单独测试这一更新机制，我们在 ROCStories 数据集上进行训练，其中每个窗口在语义层上构成相同的下一句预测任务。在验证集上（每个设置使用一个随机种子），在三个匹配设置下，四阶张量模型的最后一句交叉熵均低于自由运行的一维 Transformer：在 H=2, L=2 时低 5.3%，在 H=4, L=2 时低 2.6%，在 H=4, L=3 时低 2.4%。在 H=4, L=2 时，两者的参数量几乎相同，分别为 172.5M 和 175.9M。在相同的两个 GPU 上，该四阶张量模型的运行在 2.4（原文此处截断）……

    arXiv:2610.11716v1 Announce Type: cross  Abstract: We describe a 4-tensor attention model that predicts the next semantic state of a scene, for video generation and robot planning. A window of states has positions (x, t) and two fibers, a semantic fiber and a temporal-context fiber, and one softmax normalizes attention jointly over the window. Frames and an agent's situation are written as those states; the encoder, the renderer, and the planner remain outside the update. To test the update on its own, we train on ROCStories, where each window poses the same next-sentence task at the semantic layer. On the validation split, with one seed per setting, the last-sentence cross-entropy on the three matched settings is lower for the 4-tensor model than for a free-running one-dimensional transformer by 5.3% at H=2, L=2, by 2.6% at H=4, L=2, and by 2.4% at H=4, L=3. At H=4, L=2 the parameter counts are nearly the same, 172.5M and 175.9M. On the same two GPUs that 4-tensor run finished in 2.4 
    
[^116]: 内化器：面向超大规模语言模型的便携式上下文到参数映射

    Internalizer: Portable Context-to-Parameter Mapping for Very Large Language Models

    [https://arxiv.org/abs/2610.11715](https://arxiv.org/abs/2610.11715)

    Internalizer是一种可移植的上下文到参数映射超网络，先在小模型上低成本训练后即可移植到2840亿参数的DeepSeek v4 Flash上，为其生成文档特定的LoRA适配器，使模型无需在上下文窗口中放入文档即可将知识内化到权重中。

    

    将上下文直接映射为LoRA适配器的超网络，可以让大语言模型把该上下文“内化”到自身权重中，但此前的研究仅在参数量最高达140亿的基础模型上验证过这类方法。我们提出了Internalizer——一个最先进的、可移植的“上下文到参数映射”超网络，它能为冻结参数的2840亿参数DeepSeek v4 Flash生成针对特定文档的LoRA适配器，其目标模型规模比以往任何工作大两个数量级。该超网络的大部分参数位于一个与具体模型无关的共享主干中，每个基础模型只需附加轻量的入口层和出口层，因此可以先在小模型上低成本训练，再移植到大型模型上。在最长4096个token的未见文档上，所生成的适配器在教师强制评测下达到84.9%的top-1准确率和97.8%的top-5准确率，而基础模型仅为63.4%和83.5%，此时上下文窗口中只保留一个三词的指令。超网络训练完成后，仅需一次前向……（原文在此处截断）

    arXiv:2610.11715v1 Announce Type: new  Abstract: Hypernetworks that map a context directly to a LoRA adapter let a large language model carry that context in its weights, but prior work has demonstrated them only on base models of up to 14 billion parameters.   We present the Internalizer, a state-of-the-art, portable Context-to-Parameter Mapping hypernetwork that generates document-specific LoRA adapters for the frozen 284B-parameter DeepSeek v4 Flash, a target two orders of magnitude larger than in any previous work. Most of its parameters live in a model-agnostic trunk with only thin entry and exit layers per base model, so it trains cheaply against small models before being ported to the large one.   On unseen documents of up to 4096 tokens, the generated adapters reach 84.9% top-1 and 97.8% top-5 teacher-forced accuracy against 63.4% and 83.5% for the base model, with nothing in the context window but a three-word instruction.   Once the hypernetwork is trained, a single forward p
    
[^117]: 光照先验是否有助于换脸检测？——时序自混合图像的对照研究

    Does an Illumination Prior Help Face-Swap Detection? A Controlled Study of Temporal Self-Blended Images

    [https://arxiv.org/abs/2610.11706](https://arxiv.org/abs/2610.11706)

    本对照研究发现，在自混合图像训练中加入时序光照不一致性并不能带来光照特异性的换脸检测性能提升，反而会移动预测分数分布并改变最优阈值，仅在DFDC重度JPEG压缩场景下略改善鲁棒性。

    

    自混合图像被广泛用于训练换脸检测器，但其主要捕获的是混合伪影。我们研究在训练中加入光照不一致性能否改善检测效果。时序自混合图像在同一视频的帧之间迁移光照统计特性，其失配程度由亮度差（ΔL）控制。通过五种训练方案以及对高、低ΔL训练的三种子对比实验，我们没有发现任何光照特异性改进的证据。在四个数据集上，AUC差异始终处于种子随机性的波动范围内；对506,328个按属性分箱样本的分析也表明，在恶劣光照条件下错误率并没有出现优先降低。相反，T-SBI会移动预测分数的分布，使FaceForensics++上的最佳阈值变化约0.34、Celeb-DF上变化约0.30，这使得在固定阈值下进行的比较具有误导性。不过，T-SBI确实提升了DFDC数据集上对重度JPEG压缩的鲁棒性（AUC 0.7……）。

    arXiv:2610.11706v1 Announce Type: new  Abstract: Self-blended images are widely used to train face-swap detectors, but primarily capture blending artifacts. We investigate whether adding illumination inconsistencies improves detection. Temporal Self-Blended Images (T-SBI) transfer lighting statistics between frames of the same video, with the mismatch controlled by luminance difference ({\Delta}L). Using five training regimes and a three-seed comparison of high- and low-{\Delta}L training, we find no evidence of illumination-specific improvements. AUC differences remain within seed variability across four datasets, and an analysis of 506,328 attribute-binned samples shows no preferential reduction in errors under harsh lighting. Instead, T-SBI shifts prediction scores, changing optimal thresholds by approximately 0.34 on FaceForensics++ and 0.30 on Celeb-DF, making comparisons at a fixed threshold misleading. However, T-SBI improves robustness to heavy JPEG compression on DFDC (AUC 0.7
    
[^118]: 无监督机器学习的目标是什么？

    What is the goal of unsupervised machine learning?

    [https://arxiv.org/abs/2610.11697](https://arxiv.org/abs/2610.11697)

    本文认为无监督学习是异质化领域，无法定义单一目标，并提出其四个不同目标：估计分布、生成新数据、为下游任务提取特征和理解数据。

    

    无监督学习是机器学习的主要分支之一。在本文中，我认为与机器学习的其他分支（监督学习和强化学习）不同，无监督学习是一个相当异质化的领域，可以服务于多个不同的目标。试图为无监督学习定义单一的目标似乎是徒劳的。我确定了无监督学习的四个不同目标：1）估计分布，2）生成新的数据点，3）为下游任务提取特征，以及4）理解数据。

    arXiv:2610.11697v1 Announce Type: new  Abstract: Unsupervised learning is one of the main branches of machine learning. Here I argue that unlike the other branches of machine learning (supervised and reinforcement learning), unsupervised learning is a rather heterogenous field that can serve several different goals. It seems futile to try to define one single goal for unsupervised learning. I identify four different goals for unsupervised learning: 1) Estimating the distribution, 2) Generating new data points, 3) Extracting features for downstream tasks, and 4) Understanding the data.
    
[^119]: Jev能成为强化学习中的Q函数或策略吗？

    Can Jev be Your Q or Policy in Reinforcement Learning?

    [https://arxiv.org/abs/2610.11692](https://arxiv.org/abs/2610.11692)

    本文研究了Jev决策模型能否在强化学习中自主充当Q函数或策略等角色，发现它能满足强化学习系统中除价值函数之外的所有对象对答案的要求，并探讨了它作为训练组件改进强化学习的潜力。

    

    基础模型为强化学习（RL）提供了先验知识，缓解了其在样本效率和迁移能力方面长期存在的弱点，但它们逐token的生成方式使得查询变得顺序化且成本高昂。Jev是一个最近发布的决策模型，它不生成任何内容，而是在单次前向传播中返回经过校准的、具有类型的答案。现有工作将基础模型在强化学习中的角色要么视为需要训练的模型，要么视为需要提示的生成器，而Jev不属于这两类，迄今为止仅在单一领域中被作为黑盒使用。因此，这类模型在强化学习环境中自主决策的能力如何，以及它如何作为训练的组成部分来改进强化学习，仍然悬而未决。为此，本文首先分析了强化学习系统中各个对象对其所消费答案的要求，并确立了Jev能够满足除价值函数这一核心用途之外的所有要求。其余对象构成了位置……（摘要不完整）

    arXiv:2610.11692v1 Announce Type: cross  Abstract: Foundation models supply reinforcement learning (RL) with priors that mitigate its longstanding weaknesses in sample efficiency and transfer, but their token-by-token generation makes queries sequential and costly. Jev, a recently released decision model, generates nothing and returns calibrated, typed answers in a single forward pass. Existing work studies foundation models in RL either as models to be trained or as generators to be prompted, and Jev belongs to neither category, having so far served only as a black box in single domains. How well such a model decides on its own in RL environments, and how it can improve RL as a component of training, therefore remain unaddressed. To this end, in this paper we first examine the requirements that the objects of an RL system place on the answers they consume, and establish that Jev can fulfill all of them except the cardinal use of a value function. The remaining objects form positions t
    
[^120]: 用于换脸检测的相机噪声残差：冗余而非互补，及其原因

    Camera-Noise Residuals for Face-Swap Detection: Redundant, Not Complementary, and Why

    [https://arxiv.org/abs/2610.11683](https://arxiv.org/abs/2610.11683)

    该研究发现，相机噪声残差信息对换脸检测而言与RGB外观特征是冗余而非互补的：其判别信号本质上是统计性的（残差的均值、方差、能量等矩），且会被噪声分支输入处的InstanceNorm层标准化掉，因此融合无法带来性能提升。

    

    将学习到的相机噪声指纹与RGB外观骨干网络相融合，是一条颇具吸引力的、可实现与生成器无关的深度伪造检测的路径，因为噪声残差根植于图像成像物理过程，而非特定生成器的纹理统计特性。我们在FaceForensics++数据集上测试了Noiseprint++残差通道对于换脸检测是否携带与RGB Xception骨干网络互补的信息。三模型消融实验（仅RGB、仅残差、后期融合）表明，融合并不能优于仅使用RGB，且单独的残差分支性能接近随机水平。一项七级瓶颈诊断定位了原因：噪声图确实携带判别信号，但它是统计性的——由残差的每样本一阶和二阶矩（均值、方差、能量）所承载——而按照TruFor模板放置在噪声分支输入处的每样本InstanceNorm层，恰好将这些统计量标准化……（原文摘要在此处截断）

    arXiv:2610.11683v1 Announce Type: new  Abstract: Fusing a learned camera-noise fingerprint with an RGB appearance backbone is an appealing route to generator-independent deepfake detection, because the noise residual is grounded in image-formation physics rather than in the texture statistics of a particular generator. We test, on FaceForensics++, whether a Noiseprint++ residual channel carries information \emph{complementary} to an RGB Xception backbone for face-swap detection. A three-model ablation (RGB-only, residual-only, late-fusion) shows that fusion does not improve over RGB alone and that the residual branch alone is near chance. A seven-level bottleneck diagnostic localizes the cause: the noise maps do carry a discriminative signal, but it is statistical---carried by the per-sample first and second moments (mean, variance, energy) of the residual---and the per-sample \texttt{InstanceNorm} layer placed at the noise-branch input, following the TruFor template, standardizes exac
    
[^121]: 基于盆地几何与循环去噪的扩散模型记忆化早期特征

    Early Signatures of Memorization in Diffusion Models via Basin Geometry and Cyclic Denoising

    [https://arxiv.org/abs/2610.11670](https://arxiv.org/abs/2610.11670)

    该论文提出扩散模型的记忆化在生成样本显现之前就已编码于能量景观的几何结构中（即“潜在记忆化”），并利用分数散度、盆地体积和循环去噪方法，在理论上证明可在记忆化发生前对其进行早期检测。

    

    扩散模型在训练早期会进行泛化，而到训练后期会复现个别训练样本。标准的检测方法只有在一次性生成产生近似复制品时才能检测到记忆化，这使得已发布的模型在其输出出现问题之前无法得到审计。我们证明，记忆化在出现在生成样本之前，就已经被编码在学习到的能量景观的几何结构中，我们将这种状态称为“潜在记忆化”。利用分数散度和盆地体积，我们发现局部化的盆地会在训练样本周围形成，并在第一个被记忆的样本出现之前将它们与保留样本区分开来，且其出现时间遵循与记忆化时间相同的 O(n) 缩放规律。我们通过循环去噪（即反复施加部分加噪和去噪）来探测这些盆地。在精确的经验分数下，我们证明从孤立训练样本附近开始的循环，在任何有限次数的循环中都能以高概率恢复该样本并返回到该样本。

    arXiv:2610.11670v1 Announce Type: new  Abstract: Diffusion models generalize early in training and later reproduce individual training samples. Standard tests detect memorization only once one-shot generation produces near-copies, leaving a released model unaudited until its outputs fail. We show that memorization is encoded in the geometry of the learned energy landscape before it appears in generated samples, a state we call latent memorization. Using score divergence and basin volume, we find that localized basins form around training samples and separate them from held-out samples before the first memorized sample appears, with an onset that follows the same $O(n)$ scaling as the memorization time. We probe these basins with cyclic denoising, which repeatedly applies partial noising and denoising. Under the exact empirical score, we prove that cycling started near an isolated training sample recovers it and returns to it over any finite number of cycles with high probability. In tr
    
[^122]: σTransfer：基于μP（最大更新参数化）从小网络到大网络的不确定性迁移

    $\sigma$Transfer: Uncertainty Transfer from Small to Large Networks under $\mu\mathrm{P}$

    [https://arxiv.org/abs/2610.11668](https://arxiv.org/abs/2610.11668)

    该论文提出σTransfer方法，在μP参数化下通过重新缩放先验协方差，使拉普拉斯近似所需的先验精度可以从小模型零样本迁移到大模型，从而免去在大模型上的昂贵精度搜索，实测加速可达约5000倍。

    

    拉普拉斯近似中可靠的预测不确定性在很大程度上取决于先验精度，然而选择该精度需要进行后验扫描，对于拥有数十亿参数的神经网络而言，其代价高得令人望而却步。在最大更新参数化（μP）下，我们推导出一种先验协方差的重新缩放方法，使得所选精度随模型宽度的增长保持稳定。由此产生了σTransfer：我们在较小的模型上选择精度，然后将其零样本迁移到规模大得多的模型上，即完全无需在较大模型上搜索精度。我们在明确的条件下证明了先验核、后验协方差、所选精度以及基于后验得出的决策的收敛性，并在回归、图像分类和Transformer读取头等任务上验证了σTransfer。例如，从宽度12的模型迁移时，实测的精度扫描加速比可达约5000倍。

    arXiv:2610.11668v1 Announce Type: cross  Abstract: Reliable predictive uncertainty in Laplace approximations depends critically on the prior precision, yet selecting it requires a posterior sweep that is prohibitively expensive for neural networks with billions of parameters. Under the Maximal Update Parametrization ($\mu\mathrm{P}$), we derive a rescaling of the prior covariance that makes the selected precision stable as model width grows. This leads to $\sigma\mathrm{Transfer}$: we select the precision on a smaller model and zero-shot transfer it to the much larger model, i.e., without searching for the precision on the larger model at all. We show convergence of the prior kernel, posterior covariance, selected precision, and posterior-derived decisions under explicit conditions, and verify $\sigma\mathrm{Transfer}$ across regression, image classification, and Transformer readouts. For example, measured precision-sweep speedups reach $\sim 5000\times$ when transferring from width 12
    
[^123]: Evi-VN：面向基于GNN欺诈检测的硬区域引导虚拟节点证据注入

    Evi-VN: Hard Region Guided Virtual Node Evidence Injection for GNN-Based Fraud Detection

    [https://arxiv.org/abs/2610.11665](https://arxiv.org/abs/2610.11665)

    提出Evi-VN框架，通过硬区域引导的虚拟节点证据注入机制，学习并纠正多种GNN在欺诈检测中共同的盲点区域，有效利用异构多模态证据来识别伪装良好的欺诈者。

    

    在线平台中，模仿合法用户的机器人、欺骗性评论者和诈骗账户日益增多。这种伪装模糊了图的邻域结构和行为属性，使得图神经网络（GNN）难以区分伪装良好的欺诈者和合法用户。在多种不同的GNN中，我们观察到它们在一个共享的困难区域上出现重叠的错误，这表明存在图拓扑结构和标准特征无法捕捉的潜在欺诈证据。针对欺诈专用的GNN可以缓解特定的图结构病理问题，但它们对结构化记录、文本、图像和音频等异构证据的利用仍然有限；统一的多模态融合也可能干扰那些已被图可靠处理的节点。我们提出Evi-VN来学习和纠正这些共享的盲点，而不是构建又一个欺诈检测器。据我们所知，Evi-VN是首个使用特征……（原文摘要在此处截断）

    arXiv:2610.11665v1 Announce Type: new  Abstract: Online platforms contain growing numbers of bots, deceptive reviewers, and scam accounts that imitate legitimate users. Such camouflage blurs graph neighborhoods and behavioral attributes, making it difficult for graph neural networks (GNNs) to distinguish both well-disguised fraudsters and legitimate users. Across diverse GNNs, we observe overlapping errors on a shared hard region, suggesting the presence of latent fraud evidence that graph topologies and standard features fail to capture. Fraud-specific GNNs can mitigate particular graph pathologies, yet they still make limited use of heterogeneous evidence such as structured records, text, images, and audio; uniform multimodal fusion may also disturb nodes already handled reliably by the graph. We propose Evi-VN to learn and correct these shared blind spots rather than build another fraud detector. To our knowledge, Evi-VN is the first graph fraud detection framework to use feature is
    
[^124]: Harness演化触及天花板：权重训练应于何时启动

    Harness Evolution Hits a Ceiling: When Weight Training Should Begin

    [https://arxiv.org/abs/2610.11655](https://arxiv.org/abs/2610.11655)

    该论文提出通过失败构成分析来决定改进长时程LLM智能体的正确杠杆——将失败区分为过程失败与内容失败，其中Harness演化修复过程失败且其行为可被训练进权重，而内容失败则需依靠权重训练来解决。

    

    改进长时程LLM智能体有两条路径：在冻结模型周围演化Harness（执行框架），或者训练模型权重。我们首先让自演化的Harness使系统变得更强，然后将种子/演化后的Harness与基础/训练后的权重进行交叉组合，以探究训练后的模型能保留哪些增益、哪些增益仍需要运行时框架支持。我们表明，正确的改进杠杆可以从智能体的失败构成中读取：通过首个触发的信号对失败轨迹进行标注，可将过程失败（调用被阻塞、陷入循环、步骤预算耗尽）与内容失败（交付的方案质量差）区分开来。Harness演化修复过程失败，其灌输的行为可以进一步训练进权重，而内容失败正是权重训练所要解决的问题。在DeepPlanning基准上，自演化Harness循环将Qwen3.5-4B的保留集得分从0.16提升至0.30，将Qwen3.5-9B的得分从0.32提升至0.44；对于4B模型，保留集交付率从55%升至90%，而内容失败……（摘要在此处截断）

    arXiv:2610.11655v1 Announce Type: new  Abstract: Improving a long-horizon LLM agent means evolving the harness around a frozen model or training its weights. We let a self-evolving harness make the system stronger first, then cross seed and evolved harnesses with base and trained weights to learn which gains the trained model keeps and which still need the runtime. We show that the right lever can be read off the agent's failure composition: labelling failed trajectories by the first signal that fires separates process failures (blocked calls, loops, exhausted step budgets) from content failures (a delivered plan that is poor). Harness evolution repairs the former, the behaviour it instils can be trained into the weights, and content failures are what weight training is for. On DeepPlanning, a self-evolving harness loop lifts the held-out score of Qwen3.5-4B from 0.16 to 0.30 and of Qwen3.5-9B from 0.32 to 0.44; for 4B, held-out delivery rises from 55% to 90% while content failures are
    
[^125]: 面向德语语音识别的音系学感知分词：一项跨领域研究

    Phonologically Informed Tokenization for German Speech Recognition: A Cross-Domain Study

    [https://arxiv.org/abs/2610.11646](https://arxiv.org/abs/2610.11646)

    本文提出基于音系学知识（Pyphen 音节划分与字素-音素转换）的分词方法用于德语端到端语音识别，发现在域内条件下其性能与 BPE 和字符基线相当，而跨领域表现主要由词表规模而非语言学分词方式决定。

    

    德语是一种形态丰富的语言，其音节结构可以被 Knuth–Liang 连字（断词）算法极好地预测。本文探究基于音系学知识的分词能否成为端到端语音识别的一个有竞争力的目标。作者在使用 CTC 微调的 Omnilingual ASR wav2vec 2.0 骨干网络上比较了三类分词器：预训练的多语言字符清单、基于正字法的数据驱动字节对编码（BPE），以及由 Pyphen 音节划分和字素到音素转换得到的音系学感知单元。通过 40 次微调实验，他们在三个跨越正交分布偏移的德语测试集上进行评估：域内朗读语音、方言自发性语音和标准德语自发性语音。在域内条件下，所有音系学感知分词器在词错误率（WER）和字符错误率（CER）上均与 BPE 及多语言字符基线持平。在分布偏移条件下，结果的分化主要沿词表规模而非语言学（此处摘要截断）展开。

    arXiv:2610.11646v1 Announce Type: new  Abstract: German is a morphologically rich language whose syllable structure is exceptionally well-predicted by the Knuth--Liang hyphenation algorithm. We ask whether phonologically informed tokenization can serve as a competitive target for end-to-end speech recognition. We compare three tokenizer families on the Omnilingual ASR wav2vec 2.0 backbone fine-tuned with CTC: the pretrained multilingual character inventory, a data-driven Byte-Pair Encoding (BPE) over orthography, and phonologically informed units from Pyphen syllabification and grapheme-to-phoneme conversion. Across 40 fine-tunes, we evaluate on three German test sets spanning orthogonal shifts: in-domain read speech, dialectal spontaneous speech, and standard-German spontaneous speech. In-domain, all phonologically informed tokenizers match BPE and the multilingual character baseline on both WER and CER. Under domain shift the picture splits along vocabulary size rather than the lingu
    
[^126]: 持续机器遗忘的极小极大高斯机制

    Minimax Gaussian Mechanisms for Continual Machine Unlearning

    [https://arxiv.org/abs/2610.11628](https://arxiv.org/abs/2610.11628)

    本文提出基于牛顿更新与高斯差分隐私的极小极大高斯机制，通过推导残差误差上界来校准噪声方差分配，使得顺序删除记录后发布的一系列模型在统计上与精确重训练难以区分，并最小化最坏情况下的噪声方差。

    

    机器遗忘是指在记录被删除后更新已训练的模型，目标是在不重复完整训练过程的情况下，达到与精确重训练相同的效果。我们针对顺序删除请求，为牛顿更新开发了高斯机制。利用高斯差分隐私（GDP）及其自适应组合规则，我们证明了所发布模型的完整序列在统计上难以与对应的精确重训练区分开来。为了在经验风险最小化场景下校准这些机制，我们推导了牛顿近似相对于精确重训练的误差上界，以及该误差在每批删除之后变化幅度的上界。独立高斯噪声利用每次发布时完整残差的界进行校准，而高斯随机游走噪声则使用残差增量的更紧的界。这些界给出了在所得GDP认证下，使各次发布中最坏情况最大噪声方差最小化的噪声分配方案。

    arXiv:2610.11628v1 Announce Type: cross  Abstract: Machine unlearning updates a trained model after records are deleted, aiming to match exact retraining without repeating the full training procedure. We develop Gaussian mechanisms for Newton updates under sequential deletion requests. Using Gaussian differential privacy (GDP) and its adaptive composition rule, we show that the full sequence of released models is statistically difficult to distinguish from matched exact retraining. To calibrate these mechanisms for empirical risk minimization, we derive upper bounds on the error of the Newton approximation relative to exact retraining and on how this error changes after each deletion batch. Independent Gaussian noise is calibrated using bounds on the full residual at each release, whereas Gaussian random walk noise uses smaller bounds on residual increments. These bounds yield allocations minimizing the worst-case maximum noise variance across releases under the resulting GDP certifica
    
[^127]: 超越动作熵：面向基因组规模代谢模型修复的商空间探索

    Beyond Action Entropy: Quotient-Space Exploration for Genome-Scale Metabolic Model Repair

    [https://arxiv.org/abs/2610.11627](https://arxiv.org/abs/2610.11627)

    该论文提出QuotientPO方法，通过将等价的代谢模型修复折叠为规范化机制并在商空间上直接优化探索，配合核化Rényi估计器解决修复核心间的拥挤问题，从而克服了传统探索中输出多样性无法对应科学假设多样性的隐性失效模式。

    

    从功能观测出发修复科学模型与监督预测有着根本区别：反馈可能仅验证某个解决方案可行，却不揭示究竟哪种结构性修正起了作用。我们针对基因组规模代谢模型修复研究这一问题：在该场景下，多个反应编辑可以解释相同的表型，而许多看似不同的编辑实际上对应相同的生物学机制。这种多对一结构为传统探索方法带来了一个隐性的失效模式：输出空间的多样性未必能转化为科学假设的多样性。我们提出QuotientPO，将等价的修复折叠为规范化机制，并直接在由此得到的商空间上进行探索优化。为使商空间探索在有限回合采样下依然具有信息量，我们推导了一种核化的Rényi估计器，能够在粗粒度的精确匹配计数之外，区分不同修复核心之间的分级拥挤问题。在2,212个（摘要在此处截断）

    arXiv:2610.11627v1 Announce Type: new  Abstract: Repairing scientific models from functional observations differs fundamentally from supervised prediction: feedback may certify a solution without revealing which structural correction is responsible. We study this setting for genome-scale metabolic model (GEM) repair, where multiple reaction edits can explain the same phenotypes and many apparently distinct edits correspond to the same biological mechanism. This many-to-one structure creates a hidden failure mode for conventional exploration: diversity in the output space need not translate into diversity of scientific hypotheses. We introduce QuotientPO, which collapses equivalent repairs into canonical mechanisms and optimizes exploration directly over the resulting quotient space. To make quotient exploration informative under finite rollouts, we derive a kernelized R\'enyi estimator that resolves graded crowding among distinct repair cores beyond coarse exact-match counts. On 2,212 
    
[^128]: 构建结构化决策源以实现基于共识的伪标签学习

    Constructing Structured Decision Sources for Consensus-Based Pseudo-Label Learning

    [https://arxiv.org/abs/2610.11621](https://arxiv.org/abs/2610.11621)

    该论文提出通过受控改变图表示的类内结构（中心粒度和邻域混合）来构建稳定且互补的决策源，并利用共识机制使伪标签精度相比常规GCN提升1.19至4.39个百分点。

    

    共识可以使伪标签学习更加可靠，但前提是各个预测器能够提供真正不同的证据。多个重复相同决策边界的模型只会带来额外的投票，而不会带来额外的信息。我们通过受控地改变类内结构来构建决策源，从而解决这一问题。从一个共享的图表示出发，我们改变中心粒度和邻域混合方式，重现每个生成的决策源以测试其稳定性，并利用节点对共分配关系选择互补子集。随后，将所选决策源的一致预测结果进行排序，用于学生模型的训练。在Cora、CiteSeer和PubMed的公开固定数据划分上，使用五个随机种子进行评估，所构建的决策源相比三个常规初始化的GCN决策源，将固定预算训练下的伪标签精度提升了1.19至4.39个百分点。在匹配的结构过滤器下，三决策源共识（摘要在此处截断）

    arXiv:2610.11621v1 Announce Type: new  Abstract: Consensus can make pseudo-label learning more reliable, but only when its predictors contribute genuinely different evidence. Multiple models that repeat the same boundary provide additional votes without additional information. We address this problem by con structing decision sources through controlled changes to within-class structure. Starting from a shared graph representation, we vary center granularity and neighborhood mixing, reproduce each resulting source to test its stability, and select a complementary subset using node pair coassignment. Unanimous predictions from the selected sources are then ranked for student training. On the public fixed splits of Cora, CiteSeer, and PubMed, evaluated with five random seeds, the constructed sources improve fixed-budget training pseudo-label precision by 1.19 to 4.39 percentage points over three conventionally initialized GCN sources. Under matched structural filters, three-source consens
    
[^129]: 面向无模型策略梯度平均场控制的随机化传输映射

    Randomized Transport Maps for Model-Free Policy-Gradient Mean-Field Control

    [https://arxiv.org/abs/2610.11619](https://arxiv.org/abs/2610.11619)

    提出Transport REINFORCE方法，通过基于传输映射的种群分布随机扰动，填补了标准REINFORCE在平均场控制中无法捕捉的种群分布效应，实现了适用于有限与连续状态空间的无模型策略梯度学习。

    

    我们为离散时间平均场控制（MFC）开发了一种无模型策略梯度方法。在平均场控制中，策略通过两条途径影响目标函数：一是通过受控动力学，二是通过种群（总体）分布。标准的REINFORCE估计器只能捕捉第一种效应，而无法捕捉第二种效应。我们提出了Transport REINFORCE，这是一种基于传输映射的方法，通过对种群分布的适当变换施加扰动来估计这一缺失的平均场贡献。该方法同时适用于有限状态空间和连续状态空间。在有限状态空间中，我们通过当前种群权重与随机权重之间的凸组合，直接在概率单纯形上对种群分布进行扰动。在连续状态空间中，我们先将种群分布投影到高斯混合分布流形上，再通过一个传输映射对其进行随机化，以确保扰动后的分布仍保持在该流形之内。我们证明了一致性……（摘要原文在此处截断）

    arXiv:2610.11619v1 Announce Type: cross  Abstract: We develop a model-free policy gradient method for discrete-time mean-field control (MFC). In MFC, the policy affects the objective both through the controlled dynamics and through the population distribution. Standard REINFORCE estimators capture the first effect but not the second. We introduce Transport REINFORCE, a transport map-based approach that perturbs a suitable transformation of the population distribution to estimate this missing mean-field contribution. The method applies to both finite and continuous state spaces. In finite state spaces, we perturb the population distribution directly on the probability simplex through a convex combination of the current population weights and random weights. In continuous state spaces, we project the population distribution onto the manifold of Gaussian mixtures, and then randomize it via a transport map that ensures the perturbed law remains within this manifold. We prove consistency of
    
[^130]: NanoProof：开放且高效的 Lean 4 自动定理证明器

    NanoProof: Open and Efficient Automated Theorem Proving in Lean 4

    [https://arxiv.org/abs/2610.11605](https://arxiv.org/abs/2610.11605)

    NanoProof 是首个训练数据、工具、流程和权重全部开源、可端到端复现的 Lean 4 执行引导定理证明器，以比同类系统少约 90 倍、比 AlphaProof 少四个数量级以上的计算量，在 MiniF2F-Test 上达到 50.8% 的 pass@16。

    

    我们提出了 NanoProof，据我们所知，这是首个在 Lean 4 中的因子化执行引导定理证明器，其训练数据、提取工具、训练流程和模型权重全部公开，使其能够使用开源资源实现端到端完全可复现。为此，我们构建并发布了结构化证明树数据集，以及一个用于在 Lean 4 形式化验证器中进行程序化交互和数据提取的工具。为支持可持续研究，我们专注于计算效率，以便于进行低门槛的训练和评估。NanoProof 在 MiniF2F-Test 上达到了 50.8% 的 pass@16，超过了同类别最接近的两个系统 HyperTree Proof Search 和 ABEL，计算量分别减少约 90 倍和 7 倍，且使用的计算量比 AlphaProof 少四个数量级以上。虽然存在性能更强的开放权重证明器，但它们都是从大型预训练语言模型微调而来，既不发布训练数据也不发布训练流程；NanoProof 表明……（原文在此截断）

    arXiv:2610.11605v1 Announce Type: cross  Abstract: We introduce NanoProof, to our knowledge the first factorized execution-guided theorem prover in Lean 4 whose training data, extraction tooling, training pipeline, and weights are all released, making it end-to-end reproducible using open-source resources. To this end, we build and release a dataset of structured proof trees, as well as a tool for programmatic interaction and data extraction within the Lean 4 formal verifier. To support sustainable research, we focus on compute efficiency to facilitate accessible training and evaluation. NanoProof achieves 50.8% pass@16 on MiniF2F-Test, exceeding the two closest systems of its class, HyperTree Proof Search and ABEL, at roughly 90x and 7x less compute, and using more than four orders of magnitude less compute than AlphaProof. Stronger open-weight provers exist, but they are fine-tuned from large pretrained language models and release neither training data nor pipeline; NanoProof shows t
    
[^131]: 条件独立性检验中的嵌入偏差

    Embedding-Bias in Conditional Independence Testing

    [https://arxiv.org/abs/2610.11584](https://arxiv.org/abs/2610.11584)

    该研究揭示了在条件独立性检验中用嵌入替代原始变量所引发的偏差问题，并证明对于残差相关性检验，只要嵌入遗漏的条件均值部分互不相关即可保证检验有效，否则偏差可精确量化为遗漏部分绝对相关性乘以两个偏R²值的几何平均数。

    

    为了检验在给定文本或图像 Z 的条件下 X 和 Y 的条件独立性，一种做法是用嵌入 ψ(Z) 代替 Z 进行条件化。这种嵌入检验有效的前提是：在给定 ψ(Z) 的条件下 Z 与 X 或 Y 独立，而这一条件无法从数据中得到验证；当该条件不成立时，原假设下的拒绝概率甚至可能趋近于一。我们研究了这种失效现象，并证明若聚焦于特定形式的依赖关系，则可以放宽嵌入所需保留的信息要求。对于一种受广义协方差度量启发的残差相关性检验而言，其有效性只要求 E[X|Z] 与 E[Y|Z] 中被 E[X|ψ(Z)] 和 E[Y|ψ(Z)] 遗漏的部分互不相关。若不满足该条件，则可将被丢弃的信息视为遗漏变量。在原假设下，偏差等于遗漏部分之间的绝对相关系数乘以两个偏 R² 值的几何平均数。

    arXiv:2610.11584v1 Announce Type: cross  Abstract: To test conditional independence of $X$ and $Y$ given a text or an image $Z$, one conditions on an embedding $\psi(Z)$ in place of $Z$. The embedded test is valid if $Z$ is independent of $X$ or of $Y$ given $\psi(Z)$, which cannot be confirmed from data, and when this fails, the rejection probability under the null hypothesis can tend to one. We study this failure, and show that focusing on a specific form of dependence relaxes what the embedding must retain. For a residual correlation test inspired by the Generalised Covariance Measure, validity only requires that the parts of $\mathbb{E}[X \mid Z]$ and $\mathbb{E}[Y \mid Z]$ missed by $\mathbb{E}[X \mid \psi(Z)]$ and $\mathbb{E}[Y \mid \psi(Z)]$ are uncorrelated. Otherwise, we treat the discarded information as an omitted variable. Under the null hypothesis, the bias equals the absolute correlation of the missed parts times the geometric mean of two partial $R^2$ values. This identi
    
[^132]: 面向物理感知高速公路轨迹预测的不确定性感知优化

    Uncertainty-Aware Optimization for Physics-Aware Highway Trajectory Prediction

    [https://arxiv.org/abs/2610.11580](https://arxiv.org/abs/2610.11580)

    该论文提出了物理感知轨迹预测框架X-TRACK的不确定性感知扩展版本（X-TRACK-DE和X-TRACK-MCD），通过显式建模运动变量中的偶然不确定性和认知不确定性，并将其经由车辆动力学传播至轨迹空间，从而提升高速公路车辆轨迹预测在安全关键应用中的可靠性。

    

    准确的轨迹预测和明确的预测不确定性对于自动驾驶等安全关键应用的可靠性至关重要。大多数轨迹预测方法仅提供点估计，而不确定性感知方法通常只在轨迹空间中量化不确定性。在物理感知方法中，预测运动变量的不确定性应被显式建模并通过车辆动力学进行传播，否则所得的轨迹空间不确定性可能无法完全反映底层运动预测所引入的变异性。因此，本工作提出了物理感知轨迹预测框架X-TRACK的不确定性感知扩展版本（X-TRACK-DE和X-TRACK-MCD）。所提出的框架预测未来车辆运动变量，并通过将运动空间不确定性传播到轨迹空间，同时建模偶然不确定性和认知不确定性。

    arXiv:2610.11580v1 Announce Type: new  Abstract: Accurate trajectory forecasting and well-defined predictive uncertainty are crucial for reliable, safety-critical applications such as autonomous driving. Most trajectory prediction approaches provide point estimates only, while uncertainty-aware approaches typically quantify uncertainty only in the trajectory space. In physics-aware approaches, uncertainty in the predicted motion variables should be explicitly modeled and propagated through the vehicle dynamics. Otherwise, the resulting trajectory-space uncertainty may not fully reflect the variability introduced by the underlying motion prediction. Therefore, in this work, uncertainty-aware extensions of X-TRACK (X-TRACK-DE and X-TRACK-MCD), a physics-aware trajectory prediction framework, are proposed. The proposed framework predicts future vehicle motion variables and models both aleatoric and epistemic uncertainties by propagating motion space uncertainty to trajectory space. Additi
    
[^133]: 平滑稀疏混合专家模型的Top-k暴露边界

    Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2610.11575](https://arxiv.org/abs/2610.11575)

    提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。

    

    稀疏混合专家模型在保持每个token固定计算预算的同时，能够高效地扩展参数容量。然而，传统的训练范式强制执行静态的top-k专家选择，这将连续的路由分布转换成了刚性的阶跃函数。这一约束引入了一个脆弱的边界，使得高度竞争的专家仅因微小的分数波动就被任意地划分为完全监督区域和零反馈区域。为了解决这个问题，我们提出了弹性专家路由，它从以k为中心的局部离散分布中随机采样激活专家的数量。在多次训练迭代中，该机制将尖锐的阈值软化成渐进的概率分布。由于采样邻域保持对称，该方法在匹配确定性训练的期望计算成本的同时，保留了推理预算。大量实验表明……

    arXiv:2610.11575v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts models scale parameter capacity efficiently while maintaining a fixed compute budget per token. However, traditional training paradigms enforce a static choice of top-$k$ experts, which converts a continuous routing distribution into a rigid step function. This constraint introduces a brittle boundary where highly competitive experts are arbitrarily separated into full-supervision and zero-feedback zones based on minor score fluctuations. To address this issue, we propose Elastic Expert Routing, which stochastically samples the active expert budget from a localized discrete distribution centered at $k$. Over multiple training iterations, this mechanism softens the sharp threshold into a gradual probability distribution. Because the sampling neighborhood remains symmetric, this approach matches the expected computational cost of deterministic training, while preserving the inference budget. Extensive experiments 
    
[^134]: Sera：面向可靠且可解释电池健康预测的语义表示聚合

    Sera: Semantic Representation Aggregation for Reliable and Interpretable Battery Health Forecasting

    [https://arxiv.org/abs/2610.11567](https://arxiv.org/abs/2610.11567)

    提出语义表示聚合框架 Sera，通过融合基于规则的知识与大语言模型解读所构建的退化语义表示来补充时序建模，从而实现更可靠、更可解释的电池健康状态预测。

    

    电池健康状态预测对电池管理至关重要，但由于电池退化的非线性以及不同电池之间的异质性，该任务仍然充满挑战。现有的数据驱动方法主要依靠时序模型从数值型电池时间序列中学习，而更高层次的退化特征往往没有被显式地表示出来。然而，这些特征能够提供退化方面的指导，从而支持可靠的预测，并使退化的影响更具可解释性。在本文中，我们提出了 Sera，一个语义表示聚合框架，通过退化语义来补充时序建模。在电池领域专业知识的指导下，Sera 从时间序列中提取退化语义，并利用基于规则的知识和基于大语言模型（LLM）的解读构建两种互补的表示，这些表示随后被独立编码（摘要在此处被截断）。

    arXiv:2610.11567v1 Announce Type: cross  Abstract: Battery state of health (SoH) forecasting is important for battery management, but remains challenging due to nonlinear degradation and heterogeneity across batteries. Existing data-driven approaches primarily use temporal models to learn from numerical battery time series, and higher-level degradation characteristics are often not explicitly represented. These characteristics, however, can provide degradation guidance to support reliable forecasting and make the influence of degradation more interpretable. In this paper, we propose \textsc{Sera}, a \underline{se}mantic \underline{r}epresentation \underline{a}ggregation framework that complements temporal modelling with degradation semantics. Guided by battery domain expertise, \textsc{Sera} extracts degradation semantics from time series and constructs two complementary representations using rule-based knowledge and LLM-based interpretation. The representations are independently encod
    
[^135]: 联邦LSA中的两全其美：可能时实现加速，始终保证个性化

    Best of Both Worlds in Federated LSA: Speedup When Possible, Personalization Always

    [https://arxiv.org/abs/2610.11555](https://arxiv.org/abs/2610.11555)

    本文提出极简算法PF-LSA，通过将每个智能体的局部随机更新与全体智能体的平均更新相混合，在无需任何异构程度先验知识且不增加计算成本的情况下，同时实现了个性化收敛保证以及智能体数量足够相似时的线性加速。

    

    我们研究了个性化联邦线性随机逼近（LSA），这一框架显著涵盖了个性化时序差分学习。在该设置中，异构智能体协作求解各不相同的线性不动点方程，每个方程对应一个特定智能体的学习问题。个性化学习中一个核心的开放性问题是：是否存在单一方法能够适应未知程度的异构性，即在所有情形下都收敛到每个智能体的个性化解，同时当各智能体的学习问题足够相似时，还能获得随智能体数量的线性加速。我们通过提出PF-LSA肯定地回答了这一问题。PF-LSA是一种极简算法，它将每个智能体的局部随机更新与所有智能体的平均更新相混合，且相对于标准联邦方法不产生任何额外的计算成本。我们证明了PF-LSA在无需任何关于异构程度先验知识的情况下，实现了两全其美的保证。

    arXiv:2610.11555v1 Announce Type: new  Abstract: We study personalized federated linear stochastic approximation (LSA), a framework which notably encompass personalized temporal difference learning. In this setting, heterogeneous agents collaborate to solve distinct linear fixed-point equations, each corresponding to an agent-specific learning problem. A central open question in personalized learning is whether a single method can adapt to an unknown level of heterogeneity by converging to each agent's personalized solution in all regimes while achieving a linear speedup in the number of agents when their learning problems are sufficiently similar. We answer this question affirmatively by introducing PF-LSA, a minimalist algorithm that mixes each agent's local stochastic update with the average update across agents, at no additional computational cost relative to standard federated methods. We prove that PF-LSA, achieves best-of-both-worlds guarantees without any prior knowledge on the
    
[^136]: 在线稀疏线性回归遗憾度的新下界与上界

    New Lower Bound and Upper Bounds on the Regret for Online Sparse Linear Regression

    [https://arxiv.org/abs/2610.11551](https://arxiv.org/abs/2610.11551)

    本文首次给出了在线稀疏线性回归极小极大遗憾的下界，并在无需正则性假设的条件下设计了具有更优遗憾上界的算法，刻画了该问题的信息论复杂度。

    

    我们研究在线稀疏线性回归（OSLR）问题，其中任何算法在预测时每个实例仅被允许访问 $d$ 个属性中的 $b$ 个，并在预测后可额外访问 $b_0\geq 0$ 个属性，该问题已被证明是NP难的。以往的工作聚焦于在正则性假设下设计计算高效的算法，但并未刻画其信息论复杂度。在本工作中，我们给出了OSLR极小极大遗憾的首个下界，并在没有正则性假设的条件下设计了具有更优上界的算法。我们刻画了极小极大遗憾如何随问题相关参数变化而伸缩，从而捕捉了OSLR的信息论复杂度。

    arXiv:2610.11551v1 Announce Type: new  Abstract: We study online sparse linear regression (OSLR) where any algorithm is restricted to accessing only $b$ out of $d$ attributes per instance for prediction and $b_0\geq 0$ additional attributes after prediction, which was proved to be NP-hard. Previous work focused on designing computationally efficient algorithms under regularity assumptions, but did not characterize its information theoretic complexity. In this work, we give the first lower bound on the minimax regret of OSLR and design algorithms with better upper bounds without regularity assumptions. We characterize how minimax regret scales with problem-dependent parameters, capturing the information theoretic complexity of OSLR.
    
[^137]: 基于各向异性幂图的$C_4$等变流匹配多晶微结构生成方法

    $C_4$-Equivariant Flow Matching on Anisotropic Power-Diagram Graphs for Microstructure Generation

    [https://arxiv.org/abs/2610.11549](https://arxiv.org/abs/2610.11549)

    该论文提出了一种结合流匹配、图神经网络、$C_4$等变架构与各向异性幂图表示的生成模型，能够合成真实的多晶微结构并支持任意分辨率渲染，同时可利用免训练引导根据用户定义条件生成复杂微结构。

    

    通过电子背散射衍射（EBSD）获取真实的微结构数据成本高昂且耗时，并且通常依赖专用设备。由于微结构对材料性能具有决定性影响，生成真实的微结构样本对于建模多晶材料的行为至关重要。我们提出了一种使用流匹配和图神经网络来合成真实多晶微结构的生成模型。通过将微结构表示为各向异性幂图，我们的模型学习到了一种紧凑的几何参数化方式，并能够以任意像素分辨率渲染生成的样本。$C_4$等变架构将旋转对称性直接融入模型之中，确保输入噪声的旋转会对应地产生生成微结构的旋转。我们还展示了如何利用免训练引导技术，基于用户定义的条件来生成复杂微结构。

    arXiv:2610.11549v1 Announce Type: new  Abstract: Acquiring realistic microstructure data through Electron Backscatter Diffraction (EBSD) is costly and time consuming, often relying on specialised equipment. As microstructures strongly influence material properties, generating realistic samples is essential for modelling the behaviour of polycrystalline materials. We introduce a generative model for synthesising realistic polycrystalline microstructures using flow matching and graph neural networks. By representing microstructures as anisotropic power diagrams, our model learns a compact geometric parametrisation and can render generated samples at arbitrary pixel resolution. A $C_4$-equivariant architecture incorporates rotational symmetry directly into the model, ensuring that rotations of the input noise produce corresponding rotations of the generated microstructure. We also demonstrate how training-free guidance can be used to generate complex microstructures, based on user defined
    
[^138]: 从受控预训练混合到代码的条件性迁移

    Conditional Transfer from Controlled Pretraining Mixtures to Code

    [https://arxiv.org/abs/2610.11548](https://arxiv.org/abs/2610.11548)

    该研究区分了任务作为诊断、可教和可迁移三种不同的信号，通过受控预训练实验发现27个任务中14个具有可教性，且精选合成任务（10/12可教）与文献衍生探针任务（4/15可教）之间存在显著的不对称性。

    

    合成任务越来越多地被用作语言模型能力的探针和预训练数据。这两种用途通常都以损失下降为依据：损失下降被视为具有信息量，而随着采样增多损失下降更快则被视为该任务值得采样的证据。我们区分了三种信号：当任务的损失能够追踪全局预训练进展时，该任务是诊断性的；当任务的损失对其自身的token预算有响应时，该任务是可教的；当纳入某个数据源能改善下游目标时，该数据源具有迁移性。我们研究了受控预训练，其中语料库的70%为固定的通用Python代码，其余30%为三个源家族的单纯形混合：OpenCodeInstruct、包含12个代码相关合成任务的精选套件，以及15个源自文献的探针任务。在任务预算扫描中，我们在27个任务中检测到14个具有可教性，且两个合成家族之间存在显著的不对称性（精选任务为10/12，而文献衍生任务为4/15）

    arXiv:2610.11548v1 Announce Type: new  Abstract: Synthetic tasks are increasingly used both as probes of language-model capability and as pretraining data. Both uses are often justified by loss reduction: falling loss is treated as informative, and faster loss reduction with more sampling as evidence that a task is worth sampling. We separate three signals. A task is diagnostic when its loss tracks global pretraining progress; it is teachable when its loss responds to its own token budget; and a data source transfers when including it improves a downstream target. We study controlled pretraining in which 70% of the corpus is fixed general Python and the remaining 30% is a simplex over three source families: OpenCodeInstruct, a curated suite of 12 code-adjacent synthetic tasks, and 15 literature-derived probe tasks. Across a task-budget sweep we detect teachability for 14 of 27 tasks, with a sharp asymmetry between the two synthetic families (10/12 curated versus 4/15 literature-derived
    
[^139]: LAIR-Net：用于表格回归的泄漏对齐-脉冲残差网络

    LAIR-Net: Leaky Alignment-Impulse Residual Networks for Tabular Regression

    [https://arxiv.org/abs/2610.11538](https://arxiv.org/abs/2610.11538)

    LAIR-Net通过泄漏残差过渡将浅层学习的锚点注入隐状态演化中，实现了对隐状态的目标感知控制，在23个表格回归基准数据集上超越了八个随机化网络和十二个传统模型，并在非线性目标结构可学习时收益最大。

    

    深度随机化模型通过随机初始化固定隐层参数，仅学习闭式解的读出层，通常通过堆叠随机变换来增加深度，而对隐状态演化缺乏目标感知的控制。我们提出了LAIR-Net（泄漏对齐-脉冲残差网络），它通过泄漏残差过渡将一个浅层学习的锚点混合到每个隐状态中。我们推导了输入扰动敏感性的深度一致界，并通过受控仿真将相对于随机化基线的性能提升归因于锚点本身，而非递归结构或额外的模型容量。当非线性目标结构在可用噪声水平下可学习时，收益显现；而对于近似线性的目标或噪声占主导的情况，收益减弱。在23个基准数据集上，LAIR-Net在八个随机化网络和十二个传统模型中取得了最佳平均排名，其相对性能与非线性目标结构的可学习性相关联。

    arXiv:2610.11538v1 Announce Type: cross  Abstract: Deep randomized models fix hidden-layer parameters through random initialization and learn only closed-form readouts, typically adding depth by stacking random trans formations without target-aware control of hidden-state evolution. We propose LAIR Net, the Leaky Alignment-Impulse Residual Network, which mixes a shallow learned anchor into each hidden state through a leaky residual transition. We derive a depth uniform bound on input-perturbation sensitivity and use controlled simulations to attribute gains over a randomized baseline to the anchor rather than recursion or added capacity. Benefits emerge when a nonlinear target structure is learnable at the available noise level and diminish for nearly linear targets or dominant noise. Across 23 benchmark datasets, LAIR-Net achieves the best average rank among eight randomized networks and twelve conventional models, with relative performance associated with the same nonlinear-structure
    
[^140]: 一种利用量子神经网络执行流模型的高效量子线路

    An Efficient Quantum Circuit for Flow Model Execution Using Quantum Neural Networks

    [https://arxiv.org/abs/2610.11537](https://arxiv.org/abs/2610.11537)

    本文提出了一种基于QROM相位反冲框架并结合量子神经网络的紧凑量子线路，实现了波函数流的高效量子模拟，从而在资源开销显著降低的情况下在量子计算机上高效执行流模型。

    

    流模型通过求解由速度场定义的常微分方程，生成从初始分布到目标分布的轨迹。流匹配通过建模两个分布之间的输运动力学来学习这一速度场。波函数流通过引入连续性哈密顿量（该哈密顿量驱动量子态的薛定谔演化），在流模型与量子动力学之间建立了形式化的联系。本文研究了波函数流的精确且高效的量子模拟，从而实现流模型在量子计算机上的高效实现。我们首先利用基于量子只读存储器（QROM）的相位反冲（phase kickback）框架进行波函数流模拟，所生成的概率密度与相应传统流模型产生的概率密度高度吻合。为解决高线路资源开销问题，我们进一步引入了一个经训练的量子神经网络……（原文摘要在此处截断）

    arXiv:2610.11537v1 Announce Type: cross  Abstract: Flow models generate trajectories from an initial distribution to a target distribution by solving an ordinary differential equation defined by a velocity field. Flow matching learns this velocity field by modeling the transport dynamics between the two distributions. Wavefunction flow establishes a formal connection between flow models and quantum dynamics by introducing a continuity Hamiltonian, which drives the Schr\"odinger evolution of quantum states. In this paper, we investigate accurate and efficient quantum simulation of the wavefunction flow, thereby realizing the efficient implementation of flow models on quantum computers. We first leverage a quantum read-only memory (QROM)-based phase kickback framework for the wavefunction flow simulation, generating probability densities that closely match those produced by the corresponding conventional flow model. To address the high circuit-resource cost, we further incorporate a trai
    
[^141]: HAND：一种受生物学启发的激活函数，可提升图像分类中的泛化能力与样本效率

    HAND: A Biologically-Inspired Activation Function that Improves Generalisation and Sample Efficiency in Image Classification

    [https://arxiv.org/abs/2610.11534](https://arxiv.org/abs/2610.11534)

    提出一种受生物学启发的激活函数HAND，通过融入归纳偏置使ConvNeXt-tiny在ImageNet1k上仅需25个训练周期即达到原模型200个周期的准确率，显著提升了泛化能力和样本效率。

    

    深度神经网络表现出人类所不具备的鲁棒性和泛化问题。它们作为学习器的数据效率也远低于人类，需要多得多的训练样本才能准确分类新样本。归纳偏置可以通过提供内置机制来改善泛化能力，从而减少对从数据中学习的依赖，有助于解决这些问题。我们将一种受生物学启发的归纳偏置融入到一个新的激活函数HAND（稳态、加速非线性与除法归一化）中，并通过在图像分类任务上训练的卷积神经网络证明了其有效性。使用HAND后，ConvNeXt-tiny仅需25个训练周期就能在ImageNet1k上达到未修改模型训练200个周期后才能达到的相同准确率。与归纳偏置的效果相一致，该性能差距随着训练时间的延长和数据增强的增加而缩小。当训练数据量减少且在类别间分布不均匀时……（摘要原文在此截断）

    arXiv:2610.11534v1 Announce Type: cross  Abstract: DNNs exhibit robustness and generalisation issues not seen in humans. They are also far less data-efficient learners, requiring considerably more training samples to accurately classify novel exemplars. Inductive bias could help with these issues by providing in-built mechanisms to improve generalisation, and hence, reduce reliance on learning from data. We incorporate a biologically-inspired inductive bias into a new activation function, HAND (Homeostasis, Accelerating Nonlinearity, and Divisive-nomalisation), and show its effectiveness with CNNs trained on image classification. Using HAND a ConvNeXt-tiny required 25 training epochs to reach the same accuracy on ImageNet1k as the unmodified model achieved after 200 epochs. Consistent with the effects of an inductive bias, the performance gap reduced with training time and increased data augmentation. When the volume of training data was reduced and unevenly distributed between classes
    
[^142]: 何时干预？联邦强化学习中的状态感知稀疏操纵

    When to Intervene? State-Aware Sparse Manipulation in Federated Reinforcement Learning

    [https://arxiv.org/abs/2610.11523](https://arxiv.org/abs/2610.11523)

    该论文首次将“何时干预”确立为联邦强化学习拜占庭攻击的一个独立攻击维度，提出了利用本地策略不确定性选择稀疏干预状态并施加包络约束行为引导的V-BSA攻击方法，证明干预时机的选择会实质性影响攻击效果。

    

    联邦强化学习（FRL）使分布式智能体能够协同训练决策策略，但其去中心化的训练过程也使全局策略学习暴露于拜占庭操纵的风险之中。现有的投毒攻击主要关注如何构造恶意更新，而轨迹层面的干预时机在很大程度上仍是隐式的。然而，在序贯决策中，干预施加在何处会改变后续轨迹和学习信号。通过受控实验，我们发现即使恶意更新的构造保持不变，改变所选择的轨迹状态也会实质性改变攻击效果。因此，我们将“何时”确定为一个独立的攻击维度，并提出了可行性约束行为引导攻击（V-BSA），该方法利用本地策略不确定性来选择稀疏干预状态，并施加包络约束的行为引导。（原文摘要在此处截断）

    arXiv:2610.11523v1 Announce Type: new  Abstract: Federated reinforcement learning (FRL) enables distributed agents to collaboratively train decision-making policies, but its decentralized training process also exposes global policy learning to Byzantine manipulation. Existing poisoning attacks primarily focus on how to construct malicious updates, while trajectory-level intervention timing remains largely implicit. In sequential decision making, however, where an intervention is applied can alter subsequent trajectories and learning signals. Through controlled experiments, we find that changing the selected trajectory states materially alters attack efficacy even when the malicious-update construction is fixed. We therefore identify when as a distinct attack dimension and introduce the Viability-constrained Behavioral Steering Attack (V-BSA), which uses local policy uncertainty to select sparse intervention states and applies envelope-constrained behavioral steering. Across discrete-ac
    
[^143]: 紧凑性与一致性：面向深度图聚类的联合框架

    Compactness and Consistency: A Conjoint Framework for Deep Graph Clustering

    [https://arxiv.org/abs/2610.11506](https://arxiv.org/abs/2610.11506)

    本文提出联合框架CoCo，利用图卷积滤波器从局部和全局两种视角学习鲁棒表示，并将其编码为低秩紧凑形式，从而在深度图聚类中同时捕获节点表示的紧凑性与一致性，克服了GNN局部消息传递难以建模全局关系以及图数据噪声冗余的问题。

    

    图聚类是数据分析中的一项基础任务，旨在将图中具有相似特征的节点划分到同一簇中。由于图神经网络（GNN）能够利用节点属性和图拓扑结构来实现有效的簇分配，这一问题得到了广泛研究。然而，通过GNN学习的节点表示通常难以通过局部消息传递机制捕获节点之间的全局关系。此外，图数据中固有的冗余和噪声很容易导致节点表示缺乏紧凑性和鲁棒性。为了解决这些问题，我们提出了一个联合框架CoCo，它为深度图聚类在学习到的节点表示中捕获紧凑性与一致性。在技术上，我们的CoCo利用图卷积滤波器从局部和全局两种视角学习鲁棒的节点表示，然后将其编码为低秩紧凑表示。

    arXiv:2610.11506v1 Announce Type: cross  Abstract: Graph clustering is a fundamental task in data analysis, aiming at grouping nodes with similar characteristics in the graph into clusters. This problem has been widely explored using graph neural networks (GNNs) due to their ability to leverage node attributes and graph topology for effective cluster assignments. However, representations learned through GNNs typically struggle to capture global relationships between nodes via local message-passing mechanisms. Moreover, the redundancy and noise inherently present in graph data may easily result in node representations lacking compactness and robustness. To address these issues, we propose a conjoint framework CoCo, which captures compactness and consistency in the learned node representations for deep graph clustering. Technically, our CoCo leverages graph convolutional filters to learn robust node representations from both local and global views, and then encodes them into low-rank com
    
[^144]: 在介质傅里叶空间中跨超表面家族学习qBIC共振

    Learning qBIC Resonances across Metasurface Families in Dielectric Fourier Space

    [https://arxiv.org/abs/2610.11500](https://arxiv.org/abs/2610.11500)

    该论文提出将七个介质超表面家族映射到共享倒格子空间，利用K空间主干网络与可微Fano专家网络相结合，实现了跨几何结构的超窄qBIC共振精确学习，将共振位置误差从3.2 nm降至0.95 nm。

    

    连续谱中的束缚态（BIC）超表面通常由特定于几何结构的参数来描述，这阻碍了跨几何结构的比较，而超窄的qBIC特征在全谱学习中容易被淹没。本文将来自七个介质超表面家族的2015个样本映射到一个共享的倒格子网格上，其中两个冻结的低阶傅里叶通道能够捕获共振位移，分支内平均R²值达0.871-0.999。对两个代表性分支的场级分析进一步证实，这些位移与麦克斯韦-傅里叶微扰图像一致。一个五通道K空间主干网络对宽带频谱进行建模，而一个局部复K空间专家网络通过可微分的Fano层对qBIC共振进行参数化。在几何结构隔离的测试集上，该专家网络将共振位置的平均绝对误差（MAE）从3.2 nm降至0.95 nm，并将共振深度误差降低了14倍。同一坐标……（原文摘要至此截断）

    arXiv:2610.11500v1 Announce Type: cross  Abstract: Bound states in the continuum (BIC) metasurfaces are typically described by geometry-specific parameters, hindering cross-geometry comparison, while ultranarrow qBIC features are easily diluted in full-spectrum learning. Here, 2015 samples from seven dielectric metasurface families are mapped to a shared reciprocal-lattice grid, where two frozen low-order Fourier channels capture resonance shifts with mean within-branch $R^2$ values of 0.871-0.999. Field-level analysis of two representative branches further confirms that these shifts are consistent with the Maxwell-Fourier perturbation picture. A five-channel K-space backbone models the broadband spectrum, while a local complex K-space expert parameterizes the qBIC resonance through a differentiable Fano layer. The expert reduces resonance-position mean absolute error (MAE) from 3.2 to 0.95 nm and the resonance-depth error by 14-fold on a geometry-blocked test set. The same coordinate 
    
[^145]: PSI-SINDy：面向非线性动力学稀疏识别的选择后推断

    PSI-SINDy: Post-Selection Inference for Sparse Identification of Nonlinear Dynamics

    [https://arxiv.org/abs/2610.11486](https://arxiv.org/abs/2610.11486)

    本文提出PSI-SINDy，通过选择后推断方法为SINDy识别出的动力学项提供有效的假设检验和置信区间，从而消除选择偏差并量化所选动力学项的统计可靠性。

    

    稀疏非线性动力学识别（SINDy）是一种数据驱动框架，它通过从预先指定的候选动力学项库中识别出一个稀疏子集，从时间序列数据中发现系统的主导动力学。在本工作中，我们开发了一个统计推断框架，通过假设检验和置信区间来量化SINDy所选动力学项的可靠性。一个关键困难在于，使用同一条含噪轨迹既进行动力学项选择又评估其统计显著性会引入选择偏差。选择后推断为解决此类偏差提供了一个有原则的框架，据此我们提出了PSI-SINDy，一种专为SINDy量身定制的选择后推断方法。由于SINDy涉及候选项中的测量误差以及响应与设计之间的共享噪声，直接应用现有的选择后推断技术具有挑战性。为解决这些挑战……（原文摘要在此处截断）

    arXiv:2610.11486v1 Announce Type: cross  Abstract: Sparse identification of nonlinear dynamics (SINDy) is a data-driven framework for discovering governing dynamics from time-series data by identifying a sparse subset of candidate dynamical terms from a prespecified library. In this work, we develop a statistical inference framework for quantifying the reliability of dynamical terms selected by SINDy through hypothesis tests and confidence intervals. A key difficulty is that using the same noisy trajectory for both selecting dynamical terms and assessing their statistical significance can introduce selection bias. Post-selection inference provides a principled framework for addressing such bias, and we propose PSI-SINDy, a post-selection inference method tailored to SINDy. Direct application of existing post-selection inference techniques is challenging because SINDy involves measurement error in the candidate terms and shared noise between the response and design. To address these cha
    
[^146]: 评估本地语言模型智能体在可复现数据工程中的表现：一项针对移动性工作流的实证软件工程研究

    Evaluating Local Language Model Agents for Reproducible Data Engineering: An Empirical Software Engineering Study of Mobility Workflows

    [https://arxiv.org/abs/2610.11482](https://arxiv.org/abs/2610.11482)

    该研究构建了一个包含十五个移动性工作流任务的基准测试，通过确定性检查器系统评估了十种本地部署的开源权重LLM智能体在生成正确且可复现数据工程制品方面的能力，并量化了闭环工作区条件等因素的影响。

    

    背景：大语言模型（LLM）智能体正被越来越多地用作软件和数据工程助手，然而关于可本地部署的开源权重智能体的证据仍然有限。现有评估往往侧重于文本回复或孤立的代码生成，而非完整工程制品的有效性。目标：我们评估本地LLM智能体能否产出正确且可复现的数据工程制品，量化闭环工作区条件的影响，并考察模型规模、架构、量化、运行时、工具使用和失败模式方面的权衡。方法：我们构建了一个包含十五个移动性工作流任务的基准测试，涵盖数据发现、连接器、运输数据处理、语义增强、特征工程、验证、可视化和报告生成。确定性检查器用于评估生成的脚本、数据表、结构化文件、图形和报告。我们对十种本地配置进行了评估。

    arXiv:2610.11482v1 Announce Type: cross  Abstract: Context: Large language model (LLM) agents are increasingly used as software and data-engineering assistants, yet evidence about locally deployable open-weight agents remains limited. Existing evaluations often emphasize textual responses or isolated code generation rather than the validity of complete engineering artifacts.   Objectives: We evaluate whether local LLM agents can produce correct and reproducible data-engineering artifacts, quantify the effect of a closed-loop workspace condition, and examine trade-offs in model scale, architecture, quantization, runtime, tool use, and failure.   Methods: We introduce a benchmark of fifteen mobility-workflow tasks covering data discovery, connectors, transport-feed processing, semantic enrichment, feature engineering, validation, visualization, and reporting. Deterministic checkers assess generated scripts, tables, structured files, figures, and reports. Ten local configurations are eval
    
[^147]: 条件残差预测：无需双向教师模型改进自回归视频扩散

    Conditional Residual Prediction: Improving Autoregressive Video Diffusion without a Bidirectional Teacher

    [https://arxiv.org/abs/2610.11479](https://arxiv.org/abs/2610.11479)

    提出条件残差预测方法，仅从图像模型初始化即可训练因果视频扩散模型（全程无需双向教师或蒸馏），通过消除模型对真实历史的过度依赖来抑制自身生成历史误差的前向传播，从而提升自回归视频生成质量，且方法更简单、更易扩展。

    

    因果视频扩散模型以自回归方式生成视频，适用于流式、交互式和长视频生成。然而在标准训练下，它们的生成质量往往低于同等规模的双向模型。许多现有方法通过从预训练的双向教师模型进行初始化或蒸馏来弥补这一差距。我们则从图像模型初始化来训练因果模型，在任何阶段都不使用双向视频模型。由于这条路径既不需要大型双向教师，也不需要复杂的蒸馏流程，因此更简单、更具可扩展性。在这条路径上，我们发现基于真实历史（ground-truth history）训练的因果模型会对真实历史产生强烈依赖，以至于在推理时其自身生成历史中的误差会向前传播。我们假设这种依赖中的大部分是不必要的，因为当前输入已经决定了历史所提供的大部分信息。我们提出条件残差预测（Cond……摘要在此处被截断）

    arXiv:2610.11479v1 Announce Type: cross  Abstract: Causal video diffusion models generate video autoregressively, which suits streaming, interactive, and long-video generation. Under standard training, however, they often yield lower generation quality than bidirectional models of the same size. Many existing approaches address this gap by initializing from or distilling a pretrained bidirectional teacher. We instead train a causal model from an image-model initialization, with no bidirectional video model at any stage. Because this path requires neither a large bidirectional teacher nor a complex distillation pipeline, it is simpler and more scalable. On this path, we find that a causal model trained on ground-truth history becomes strongly dependent on it, so that at inference errors in its own generated history propagate forward. We hypothesize that much of this dependence is unnecessary, because the current input already determines much of what the history provides. We propose Cond
    
[^148]: 罕见的门控分歧可能限制可塑性：当梯度流错误预测有限批量SGD时

    Rare Gate Disagreements Can Limit Plasticity: When Gradient Flow Mispredicts Finite-Batch SGD

    [https://arxiv.org/abs/2610.11475](https://arxiv.org/abs/2610.11475)

    该论文证明总体梯度流可能在定性上错误预测有限批量SGD：在双神经元ReLU回归中，当预训练时间超过 $\log(b/\eta)$ 后，源于罕见门控分歧的机制使在线SGD在指数级长的时间范围内以高概率丧失可塑性、无法适应目标任务，而梯度流却只需线性时间即可恢复。

    

    总体梯度流是研究神经网络如何适应（包括预训练之后的适应）的常用工具。我们证明它可能在定性层面错误预测有限批量的随机梯度下降（SGD），并将这种偏差追溯到一种特定机制。在一个双单元ReLU回归中，源任务驱动两个神经元趋于正比例关系，而目标任务则奖励将它们分离。在源任务上训练时间 $T$ 之后，梯度流可在与 $T$ 呈线性关系的时间内于目标任务上恢复。而若两个阶段均采用批大小为 $b$、步长为 $\eta$ 的在线SGD，一旦 $T \gtrsim \log(b/\eta)$，它就会在 $e^{c/\eta}$ 量级的时间范围内以高概率失败，且该失败在一组显式构造的、高斯概率超过百分之一的初始化集合上一致成立。对于每个固定的 $T$，小步长SGD仍可恢复，因此这种失败需要小步长与长预训练的联合极限。在目标克隆处，总体不稳定性……

    arXiv:2610.11475v1 Announce Type: new  Abstract: Population gradient flow is a common tool for reasoning about how neural networks adapt, including after pretraining. We show that it can mispredict finite-batch stochastic gradient descent (SGD) qualitatively, and we trace the discrepancy to a specific mechanism. In a two-unit ReLU regression, a source task drives the two neurons toward positive proportionality and a target task rewards separating them. After source training for time $T$, gradient flow recovers on the target in time linear in $T$. Online SGD with batch size $b$ and step size $\eta$ in both phases instead fails with high probability throughout a horizon of order $e^{c/\eta}$ once $T \gtrsim \log(b/\eta)$, uniformly on an explicit set of initializations with Gaussian probability above one percent. For each fixed $T$, small-step SGD still recovers, so the failure requires the joint limit of small steps and long pretraining. At the target clone, the population instability i
    
[^149]: 谁来验证验证者？与自我改进智能体共同进化的可检查评分器

    Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents

    [https://arxiv.org/abs/2610.11464](https://arxiv.org/abs/2610.11464)

    该论文提出将验证器本身作为进化对象——即由可检查的确定性缺陷检测器组成的表达式，通过锚定参考集一致性和输出共识来选择而非依据智能体分数——从而在自我改进循环中避免奖励作弊和共同盲点，并在MBPP+上比手工种子组合提升0.21的保留一致性。

    

    我们改变了智能体：它真的变得更好了吗？每个自我改进的智能体循环都会数百次地回答这个问题，而每个答案都来自一个验证器。在开放式任务上并不存在这样的验证器，因此循环只能依靠手写的评分标准，或者用一个裸的LLM裁判来评判来自与自身类似模型的输出，这容易招致奖励作弊和共同的盲点。我们让验证器成为进化的对象：一个由小型、大多为确定性的缺陷检测器组成的可检查表达式，从聚类失败中合成，在诞生时经过门控，并基于与十项锚定参考集的一致性加上对未标注输出的共识来选择，而从不根据智能体的分数来选择。在MBPP+上，它在每个种子上都比手工编写的种子组合获得了+0.21的保留一致性，并最终超越了它所包含的裸LLM裁判。有一个发现应该改变共进化验证器的验证方式：移除锚定防护会使验证器坍缩成一个空洞的总是通过的评分器，然而

    arXiv:2610.11464v1 Announce Type: new  Abstract: We changed the agent: did it actually get better? Every self-improving agent loop answers this hundreds of times, and every answer comes from a verifier. On open-ended tasks none exists, so the loop is handed a hand-written rubric or a bare LLM judge grading output from a model like itself, inviting reward hacking and shared blind spots. We make the verifier the evolving object: an inspectable expression over small, mostly deterministic drawback detectors, synthesized from clustered failures, gated at birth, and selected for agreement with a ten-item anchored reference set plus consensus over unlabeled outputs, never for the agent's score. On MBPP+ it gains +0.21 held-out agreement over the hand-authored seed composition, on every seed, and ends ahead of the bare LLM judge it contains. One finding should change how co-evolved verifiers are validated: removing the anchor guards collapses the verifier into a vacuous always-pass grader, yet
    
[^150]: 特征空间自适应实现轻松的高斯过程流

    Feature Space Adaptation for Effortless Gaussian Process Flows

    [https://arxiv.org/abs/2610.11459](https://arxiv.org/abs/2610.11459)

    该论文通过引入核近似和基于扩散工作量的边际似然估计方法，首次实现了 FlowGP 框架内的超参数自动优化，使其能够高效扩展到高分辨率域并处理非高斯条件推断任务。

    

    在线性高斯范畴之外，从高斯过程（GP）进行条件采样是具有挑战性的。近期的方法如 FlowGP（Moss 等人，2026）能够以任意非线性和非高斯条件语句为条件，但代价相当大：需要昂贵的高维迭代扩散过程，且需要手动指定核超参数。在本文中，我们通过以下两点缓解了 FlowGP 的两个显著缺陷：（1）引入核近似方法，使其能够扩展到高分辨率域；（2）提出一种通过测量引导扩散过程趋向条件语句所需的工作量来获得边际似然的方法。我们首次实现了在 FlowGP 中进行超参数优化，并在以下任务上展示了我们的方法：基于区域汇总统计量的概率降尺度、不规则域上的偏微分方程（PDE）解推断，以及从非高斯卫星观测数据中恢复海平面异常场。

    arXiv:2610.11459v1 Announce Type: cross  Abstract: Outside the linear-Gaussian regime, conditional sampling from Gaussian processes (GPs) is challenging. Recent methods such as FlowGP (Moss et al., (2026)) can condition on arbitrary non-linear and non-Gaussian statements, but at considerable cost: an expensive iterative and high-dimensional diffusion that requires hand-specified kernel hyperparameters. In this paper, we alleviate two significant drawbacks of FlowGP by (1) introducing kernel approximations that enable scaling to high-resolution domains and (2) proposing a way to obtain the marginal likelihood by measuring the work needed to steer the diffusion towards conditioning statements. We enable, for the first time, hyperparameter optimisation within FlowGP and demonstrate our approach on probabilistic downscaling from areal summary statistics, PDE solution inference on irregular domains, and recovery of sea level anomaly fields from non-Gaussian satellite observations.
    
[^151]: 生成对抗循环

    Generative Adversarial Loops

    [https://arxiv.org/abs/2610.11458](https://arxiv.org/abs/2610.11458)

    提出生成对抗循环（GAL）框架，通过判别器智能体自动生成对抗性数据以暴露当前算法的弱点、生成器智能体发现新算法加以克服，从而实现目标设定的自动化，构建能够自我进步的AI研究系统，并成功应用于高效推理的近似算法。

    

    arXiv:2610.11458v1 公告类型：cross 摘要：人工智能研究的进展可以被视为两个过程之间的相互作用：基准的创建与方法的发现。从历史上看，这两者都是由人类智能驱动的。然而，近年来人工智能的进步加速了自动化的方法发现，而自动化基准创建所受到的关注相对较少。为了实现自我进步的系统，我们提出了生成对抗循环（Generative Adversarial Loop, GAL），这是一种生成器-判别器框架，在两个智能体搜索之间交替进行：（1）判别器生成对抗性数据，以暴露当前系统的弱点；（2）生成器发现能够克服这些弱点的算法。我们将该框架应用于面向高效推理的近似算法。与主要专注于算法发现的现有自动研究系统不同，GAL 引入了一个判别器智能体，通过持续搜索当前算法的弱点来自动化“目标设定”（goalpost setting）。我们展示……（原文在此处截断）

    arXiv:2610.11458v1 Announce Type: cross  Abstract: AI research progress can be viewed as the interaction between two processes: benchmark creation and method discovery. Historically, both were driven by human intelligence. However, recent advances in AI have accelerated automated method discovery, while automated benchmark creation has received comparatively less attention. To enable self-advancing systems, we propose Generative Adversarial Loop (GAL), a generator-discriminator framework alternating between two agentic searches: (1) a discriminator that generates adversarial data to expose weaknesses in current systems, and (2) a generator that discovers algorithms to overcome them. We apply this framework to approximation algorithms for efficient inference. Unlike existing auto research systems, which primarily focus on algorithm discovery, GAL introduces a discriminator agent that automates goalpost setting by continually searching for weaknesses in the current algorithm. We demonstr
    
[^152]: Zatom-2：在原子数据上进行多任务预训练以实现跨领域生成建模

    Zatom-2: Multitask Pretraining on Atomistic Data for Generative Modeling across Domains

    [https://arxiv.org/abs/2610.11454](https://arxiv.org/abs/2610.11454)

    Zatom-2 是一个在约五百万个有机与无机原子结构上进行多任务预训练的原子生成模型，通过多尺度 Transformer 与条件流匹配实现跨化学、材料科学和生物学领域的统一生成建模。

    

    arXiv:2610.11454v1 公告类型：cross 摘要：统一的原子建模有潜力通过连接数据丰富的化学领域和数据稀缺的生物场景，加速化学、材料科学和生物学领域的科学发现。然而，现有的原子建模生成式方法仍然高度专精于特定科学学科（化学或生物学），或者未能同时利用海量的有机（分子）和无机（材料）数据进行通用预训练。为此，我们提出了 Zatom-2，这是一个在来自 OMol25 和 OMat24 电子结构数据集的大约五百万个结构上进行预训练的原子生成模型。Zatom-2 采用多尺度 Transformer 架构并结合条件流匹配，支持力条件化以及基础预训练任务，例如生成、结构预测，以及分子和材料能量与力的预测。实证结果表明，Zatom-2 在分子分布拟合方面取得了更优的表现……

    arXiv:2610.11454v1 Announce Type: cross  Abstract: Unified atomistic modeling has the potential to accelerate discovery in chemistry, materials science, and biology by bridging data-rich chemical domains and data-scarce biological contexts. However, existing generative approaches to atomistic modeling remain highly specialized to scientific disciplines (chemistry vs. biology) or do not leverage both high-volume organic (molecule) and inorganic (material) data for general-purpose pretraining. To this end, we introduce Zatom-2, an atomistic generative model pretrained on approximately five million structures from the OMol25 and OMat24 electronic structure datasets. Zatom-2 features a multiscale Transformer architecture coupled with conditional flow matching that supports force conditioning and foundational pretraining tasks such as generation, structure prediction, and prediction of molecular and material energies and forces. Empirically, Zatom-2 achieves better molecular distribution fi
    
[^153]: 面向嵌入式软件开发的LLM智能体闭环评估

    Closed-loop evaluation of LLM agents for embedded software development

    [https://arxiv.org/abs/2610.11447](https://arxiv.org/abs/2610.11447)

    该论文提出了一个包含五个嵌入式控制任务和四种反馈场景的基准测试，用于闭环评估LLM编码智能体在嵌入式软件开发中实现并自我验证设备行为的能力。

    

    大语言模型（LLM）正越来越多地被部署为编码智能体，用于编辑文件、运行构建和测试、检查执行结果并迭代修复软件。嵌入式固件是一个具有挑战性的目标，因为其正确性取决于传感、时序和安全约束下的闭环行为，而不仅仅是静态源代码的质量。然而，针对嵌入式智能体的评估仍然有限，且往往侧重于一次性代码生成或离线正确性验证。我们提出了一个用于嵌入式编码智能体闭环评估的基准测试。每个任务提供一个纯文本的工程描述、受约束的工作空间以及可见的构建与运行时界面。智能体必须将需求转化为具体实现和自我验证步骤，然后不断迭代，直到实现所需的设备行为。该测试套件包含五个嵌入式控制任务和四种反馈场景：一次性生成、贴近现实的自我验证、CI风格的红/绿反馈……

    arXiv:2610.11447v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as coding agents that edit files, run builds and tests, inspect execution results, and repair software iteratively. Embedded firmware is a demanding target because correctness depends on closed-loop behavior under sensing, timing, and safety constraints, not only on static source quality. Yet embedded-agent evaluation remains limited and often emphasizes one-shot synthesis or offline correctness.   We present a benchmark for closed-loop evaluation of embedded coding agents. Each task provides a plain-text engineering description, constrained workspace, and visible build-and-runtime surface. The agent must translate requirements into implementation and self-verification steps, then iterate until the required device behavior is achieved. The suite contains five embedded-control tasks and four feedback scenarios: one-shot generation, realistic self-verification, CI-style red/green fee
    
[^154]: MotiveMob：以动机作为语义动作的闭环人类移动性生成

    MotiveMob: Motivation as Semantic Action for Closed-Loop Human Mobility Generation

    [https://arxiv.org/abs/2610.11442](https://arxiv.org/abs/2610.11442)

    提出MotiveMob框架，将移动动机作为语义动作显式建模，先推断移动背后的动机假设、再联合生成地点与时间，实现更符合人类决策过程的闭环移动性生成。

    

    人类移动性生成是城市研究中的一项重要任务，通过合成轨迹数据服务于城市规划与交通管理。人类移动可以被刻画为一个“为何-何地-何时”的决策过程：人们首先形成移动的意图，然后再确定相应活动将在何地、何时发生。在用户层面与时间分布偏移下进行轨迹生成，可以从显式建模这一决策结构中获益。然而，许多现有的人类移动性生成方法要么以较粗的粒度表示行为意图（例如每日计划或轨迹级描述），要么直接预测未来位置，而不对每个移动步骤背后的可能动机进行显式推理。我们提出了MotiveMob，一个动机驱动的人类移动性自回归生成框架，它首先对下一次移动为何可能发生形成假设，然后联合生成移动的地点与时间。

    arXiv:2610.11442v1 Announce Type: new  Abstract: Human mobility generation, an important task in urban research, synthesizes trajectory data for urban planning and transportation management. Human mobility can be characterized as a "why-where-when" decision process: people form an intention to move and then determine where and when the corresponding activity will take place. Trajectory generation under user-level and temporal distribution shifts may benefit from explicitly modeling this decision structure. However, many existing human mobility generation methods either represent behavioral intent at a coarse granularity, such as a daily plan or a trajectory-level description, or directly predict future locations without explicitly reasoning about a possible motivation for each movement step. We introduce MotiveMob, a motivation-driven autoregressive framework for human mobility generation that first forms a hypothesis about why the next movement may occur and then jointly generates whe
    
[^155]: BioBigBird：一种用于生物医学文本长程依赖处理的稀疏注意力模型

    BioBigBird: A Sparse Attention Model for Long-Range Dependency Processing in Biomedical Text

    [https://arxiv.org/abs/2610.11430](https://arxiv.org/abs/2610.11430)

    BioBigBird是一种基于稀疏注意力机制的生物医学双向语言模型，可处理长达4096个token的长序列，并通过多任务学习联合优化命名实体识别与关系抽取，在BLURB基准上取得了与最先进模型相当的表现。

    

    虽然领域专用的大型语言模型（LLMs）已经编码了大量的生物医学知识，但它们有限的上下文窗口往往阻碍了对文本内部及跨文本之间细微关系的深入理解。为了解决这一局限，我们提出了BioBigBird，这是一个在海量生物医学文献和临床数据上预训练的双向语言模型，专门设计用于处理长程依赖关系。BioBigBird利用稀疏注意力机制来处理长达4096个token的序列，其训练采用多阶段过程以减轻大规模预训练语料库中的噪声。我们进一步通过采用多任务学习（MTL）框架来提升其性能，该框架联合优化命名实体识别和关系抽取两项任务。在BLURB基准上的全面评估表明，我们经过MTL增强的BioBigBird与最先进的模型相比取得了极具竞争力的结果。我们的工作贡献……

    arXiv:2610.11430v1 Announce Type: new  Abstract: While domain-specific Large Language Models (LLMs) have encoded vast biomedical knowledge, their limited context windows often hinder a deep understanding of nuanced relationships within and across texts. To address this limitation, we introduce BioBigBird, a bidirectional language model pre-trained on extensive biomedical literature and clinical data, specifically designed to handle long-range dependencies. BioBigBird leverages a sparse attention mechanism to process sequences up to 4096 tokens, and its training incorporates a multi-stage process to mitigate noise from the large-scale pre-training corpus. We further enhance its performance by employing a multi-task learning (MTL) framework that jointly optimizes for Named Entity Recognition and Relation Extraction. Comprehensive evaluations on the BLURB benchmark reveal that our MTL-enhanced BioBigBird achieves highly competitive results against state-of-the-art models. Our work contrib
    
[^156]: 策略对齐：在线策略蒸馏中成员资格审计的新信号

    Policy Alignment: New Signals for Membership Auditing in On-Policy Distillation

    [https://arxiv.org/abs/2610.11423](https://arxiv.org/abs/2610.11423)

    提出PAMA审计框架，首次利用教师引导的学生策略更新方向作为新信号，来审计在线策略蒸馏中私有提示的成员资格。

    

    在线策略蒸馏（OPD）通过在学生模型自身生成的轨迹上将其策略与教师模型对齐来训练学生模型。在此过程中，学生策略会在用于蒸馏的提示上向教师靠拢。然而，这些提示通常是私有的且成本高昂，因此产生了对提示级别成员资格审计的需求。现有方法主要依赖基于似然的置信度信号或检查点之间的学生策略漂移，但它们无法捕捉教师所诱导的学生更新方向。在本文中，我们提出了策略对齐成员审计（PAMA），这是一种专为OPD量身定制的全新审计框架。我们的关键观察是：成员提示直接参与了教师引导的策略更新，而非成员提示仅通过跨提示泛化受到间接影响。基于这一方向性痕迹，PAMA衡量学生更新是否朝着（成员或非成员方向）移动。

    arXiv:2610.11423v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains a student model by aligning its policy with a teacher model on trajectories generated by the student model itself. Through this process, the student policy moves toward the teacher on the prompts used for distillation. However, these prompts are often private and costly, creating a need for prompt-level membership auditing. Existing methods mainly rely on likelihood-based confidence signals or student policy drift between checkpoints, but they do not capture the teacher-induced direction of the student update. In this paper, we propose Policy Alignment Membership Auditing (PAMA), a new auditing framework tailored for OPD. Our key observation is that a member prompt directly contributes to the teacher-guided policy update, while a non-member prompt only experiences indirect effects through cross-prompt generalization. Based on this directional trace, PAMA measures whether the student update moves toward
    
[^157]: 未实现影响的因果命运动力学

    Causal-fate dynamics of unrealized influence

    [https://arxiv.org/abs/2610.11422](https://arxiv.org/abs/2610.11422)

    本文提出“因果命运动力学”框架，用以刻画动力系统中未实现的影响如何在后续演化中被实现、潜伏或转换，并通过秀丽隐杆线虫神经网络模型与实际互联网路由两个例子说明未消解的影响可以长期保留未来相关性。

    

    许多动力系统产生的影响，其后果在产生之时并未在已实现的轨迹中被完全耗尽。这类后果通常被视为不存在、延迟或静态存储，从而使得未实现的影响如何在系统演化过程中保持未来的相关性这一问题尚不清楚。在此，我们提出了因果命运动力学，其中所产生的影响可能被实现、保持潜伏，或被后续动力学所转换，并在相关映射被明确指定时给出精确的有限输运表示。一个连接组约束的秀丽隐杆线虫模型首先激发了一个生物学假设：未消解的神经元间影响可能持续存在并对后续传播作出贡献；该模型并未在活体动物中确立这一机制。接下来我们考察运营中的互联网路由，其中动态更新的跨观测者历史保留了超出当前局部路由状态的预测信息。我们……（摘要在此处截断）

    arXiv:2610.11422v1 Announce Type: cross  Abstract: Many dynamical systems generate influences whose consequences are not fully exhausted in the realized trajectory at the moment they arise. Such consequences are often treated as absent, delayed or statically stored, leaving unclear how unrealized influence retains future relevance as the system evolves. Here we formulate causal-fate dynamics, in which generated influence may be realized, remain latent, or be transformed by subsequent dynamics, and give an exact finite-transport representation when the relevant maps are specified. A connectome-constrained Caenorhabditis elegans model first motivates the biological hypothesis that unresolved inter-neuronal influence may persist and contribute to later propagation; it does not establish such a mechanism in living animals. We next examine operational Internet routing, where a dynamically updated cross-observer history retains predictive information beyond the current local route state. We 
    
[^158]: 精炼即服务：算法化的预测器精炼

    Refinement as a Service: Algorithmic Predictor Refinement

    [https://arxiv.org/abs/2610.11415](https://arxiv.org/abs/2610.11415)

    该论文提出将校准预测器形式化为信号方案，并通过可观测线性信息刻画了何时能从多个校准预测器构造出既保留原有信息又不可再精炼的精炼校准预测器。

    

    预测聚合旨在将来自多个预测器的信息组合成一个信息量更大的预测器。我们在校准预测器的设定下研究这一问题，其中每个预测必须等于在给定预测器信号的条件下，被预测量的条件期望。给定若干个校准的输入预测器以及特征分布（但不包含底层的贝叶斯概率），我们探究何时能够构造出精炼的校准预测器——这些预测器既保留原始预测器中的信息，又无法利用现有信息被进一步精炼。我们将校准预测器形式化为信号方案，并通过与特征无关的“搅乱”来定义精炼：如果一个预测器的信号能够模拟另一个预测器的信号，则称前者精炼后者。可构造性通过可观测线性信息来刻画：每个信号对应于特征空间上的一个向量，新信号是可构造的……

    arXiv:2610.11415v1 Announce Type: cross  Abstract: Prediction aggregation aims to combine information from multiple predictors into a more informative one. We study this question in the setting of calibrated predictors, where each prediction must equal the conditional expectation of the quantity being predicted given the predictor's signal. Given several calibrated input predictors and the feature distribution, but not the underlying Bayes probabilities, we ask when one can construct refined calibrated predictors that preserve the information in the original predictors and cannot be further refined using the available information.   We formulate calibrated predictors as signaling schemes and define refinement through feature-independent garblings: a predictor refines another if its signal can simulate the other's signal. Constructibility is characterized through observable linear information: each signal corresponds to a vector over the feature space, and a new signal is constructible 
    
[^159]: MC-TRCM：面向不完整移动与可穿戴心理健康特征视图的观测感知递归融合

    MC-TRCM: Observation-Aware Recursive Fusion for Incomplete Mobile and Wearable Mental-Health Feature Views

    [https://arxiv.org/abs/2610.11408](https://arxiv.org/abs/2610.11408)

    提出MC-TRCM模型，将不完整的移动与可穿戴心理健康特征源作为独立token并显式建模缺失信息，通过递归预测头实现多源异构心理健康数据的鲁棒融合与预测。

    

    公开的移动和可穿戴心理健康数据集通常提供的是汇总后的特征表格，而非同步的原始传感器数据流。在这些发布的数据中，每个锚点对应一次调查或标签时间点，可能结合手机或可穿戴设备的汇总信息、既往症状评分、人口统计学信息、临床变量以及数据来源可用性指标。我们提出了模态条件时间递归上下文模型（MC-TRCM），该模型将每个特征源保留为独立的token，并将缺失情况作为输入上下文的一部分纳入建模。已观测到的数据源以其数值和缺失摘要进行编码，缺失的数据源则使用学习到的缺失token表示；数据集和任务嵌入对融合过程进行条件化，递归预测头在验证集选定步数内对每个输出进行迭代细化。我们在DepreST-CAT和抑郁症严重程度变化预测（PSYCHE-D）数据集的六个预定义终点上评估了MC-TRCM，采用参与者级别的数据划分和仅基于验证集的……

    arXiv:2610.11408v1 Announce Type: new  Abstract: Public mobile and wearable mental-health datasets often provide summarized feature tables rather than synchronized raw sensor streams. In these releases, each anchor corresponds to a survey or label time and may combine phone or wearable summaries, prior symptom scores, demographics, clinical variables, and source-availability indicators. We propose the Modality-Conditioned Temporal Recursive Context Model (MC-TRCM), which preserves each feature source as a separate token and incorporates missingness as part of the input context. Observed sources are encoded with values and missingness summaries, absent sources use learned absence tokens, dataset and task embeddings condition fusion, and a recursive prediction head refines each output over validation-selected steps. We evaluated MC-TRCM on six predefined endpoints from DepreST-CAT and Prediction of Severity Change-Depression (PSYCHE-D) using participant-level splits and validation-only m
    
[^160]: 超越分布保真度：面向合成表格数据的因果惩罚扩散模型

    Beyond Distributional Fidelity: Causal-Penalized Diffusion for Synthetic Tabular Data

    [https://arxiv.org/abs/2610.11407](https://arxiv.org/abs/2610.11407)

    该论文首次将因果差异惩罚直接引入生成式表格扩散模型，提出因果惩罚化的 TabDDPM 训练框架，理论上证明高统计保真度不等于高因果保真度，并给出了因果正则化提升期望因果保真度的条件与实验验证。

    

    合成表格数据生成器通常以分布保真度为优化目标，但仅凭统计上的相似性并不能保证因果效应得以保留。本文研究了能否在完全生成式的表格模型中直接提升因果保真度。我们将“因果保真度”相对于目标估计量定义为：基于真实数据与合成数据所得到的推断分布之间的差异，并通过理论结果表明，高统计保真度通常并不意味着高因果保真度。随后，我们提出了一种因果保真度感知的训练框架，在生成目标函数中加入了因果差异惩罚项。该框架以因果惩罚化的 TabDDPM 为实例实现，并采用在策略得分函数估计器进行优化。我们进一步建立了因果正则化能够提升期望因果保真度的条件。在多种处理效应模拟及（摘要在此处截断）的实验中……

    arXiv:2610.11407v1 Announce Type: cross  Abstract: Synthetic tabular generators are commonly optimized for distributional fidelity, but statistical similarity alone does not guarantee preservation of causal effects. In this paper, we study whether causal fidelity can be improved directly within a fully generative tabular model. Causal Fidelity is defined with respect to a target estimand as the discrepancy between inferential distributions obtained from real and synthetic data, and theoretical results show that high statistical fidelity does not generally imply high causal fidelity. We then propose a causal-fidelity-aware training framework which adds a causal discrepancy penalty to the generative objective. The framework is instantiated with a causal-penalized TabDDPM and optimized using an on-policy score-function estimator. We further establish conditions under which causal regularization improves expected causal fidelity. Experiments across diverse treatment-effect simulations and 
    
[^161]: 从异质密度中学习用于冷冻电镜蛋白质重建

    Learning from Hetero Density for Cryo-EM Protein Reconstruction

    [https://arxiv.org/abs/2610.11403](https://arxiv.org/abs/2610.11403)

    CryoCue框架通过锚点监督检测器学习异质组分表示，并利用多尺度异质特征与预测候选点的类别、置信度和几何信息来指导冷冻电镜蛋白质重建，显著提升了异质组分附近的主链定位与结构重建精度。

    

    从冷冻电子显微镜（cryo-EM）图谱中重建蛋白质结构对于理解大分子组装体至关重要。尽管基于学习的方法已经改进了蛋白质重建，但来自异质组分的信息仍未得到充分利用。我们的分析发现，在异质组分附近既存在错误预测也存在参考蛋白质位点，过滤附近的候选点可能改善也可能损害链的构建。我们提出了CryoCue，一个利用异质信息指导蛋白质重建的框架。该框架采用锚点监督的检测器学习五种组分类别的异质表示，多尺度异质特征指导主链定位，同时预测的异质候选点通过其类别、置信度以及相对于参考框架的几何关系来调节结构细化。实验表明，CryoCue改善了异质组分附近的主链定位，并实现了更精确的蛋白质结构重建。

    arXiv:2610.11403v1 Announce Type: new  Abstract: Reconstructing protein structures from cryo-electron microscopy (cryo-EM) maps is essential for understanding macromolecular assemblies. Although learning-based methods have improved protein reconstruction, information from hetero components remains underused. Our analysis finds both false predictions and reference protein sites near hetero components; filtering nearby candidates can improve or impair chain construction. We introduce CryoCue, a framework that uses hetero information to guide protein reconstruction. An anchor-supervised detector learns hetero representations across five component classes. Multiscale hetero features guide backbone localization, while predicted hetero candidates condition structure refinement through their class, confidence, and frame-relative geometry. Experiments show that CryoCue improves backbone localization near hetero components and achieves more accurate protein structure reconstruction.
    
[^162]: WAM-Cache：面向高效世界动作模型的陈旧度受限KV复用方法

    WAM-Cache: Staleness-Bounded KV Reuse for Efficient World Action Models

    [https://arxiv.org/abs/2610.11401](https://arxiv.org/abs/2610.11401)

    WAM-Cache是一个免训练框架，通过跨块缓存复用视频DiT的KV表示，并依据动作专家的注意力位置（而非视觉漂移）仅稀疏刷新关键token，从而大幅降低世界动作模型闭环机器人操作中的预填充计算成本。

    

    世界动作模型通过让动作专家以预训练视频扩散Transformer（DiT）生成的表示为条件，实现了通用机器人操作。在闭环控制中，视频DiT需要在每个控制块（chunk）运行，将当前观测编码为逐层键值（KV）对，供动作专家查询。这一预填充（prefill）过程主导了每个块的计算成本，然而现有的免训练加速方法仍使其保持完全稠密的计算。我们提出了WAM-Cache，这是一个免训练框架，它在块之间保留逐层键值表示，并仅重新计算一个稀疏的刷新token集合。关键的是，我们发现即便是“刷新视觉上发生漂移的token”这一直观启发式方法，其性能也远低于稠密基线的上限，即使使用能够预测真实KV漂移的oracle也是如此。下游动作准确性实际上由动作专家关注的位置决定，而不是由场景中什么发生了移动决定。因此，WAM-Cache通过统一（摘要在此处截断）

    arXiv:2610.11401v1 Announce Type: cross  Abstract: World Action Models (WAMs) enable generalist robot manipulation by conditioning an action expert on representations from a pretrained video Diffusion Transformer (DiT). In closed-loop control, the video DiT runs at every chunk to encode the current observation into layerwise key-value (KV) pairs that the action expert queries. This prefill dominates the per-chunk computational cost, yet existing training-free accelerations leave it fully dense. We present WAM-Cache, a training-free framework that retains layerwise key-value representations across chunks and recomputes only a sparse refresh set of tokens. Crucially, we find that the intuitive heuristic of refreshing visually drifted tokens plateaus far below the dense baseline, even with an oracle predicting ground-truth KV drift. Downstream action accuracy is instead governed by where the action expert attends, not by what moved. WAM-Cache therefore selects the refresh set by uniting t
    
[^163]: 利用势函数估计自回归语言模型下的广义期望

    Estimating great expectations under autoregressive language models with potentials

    [https://arxiv.org/abs/2610.11399](https://arxiv.org/abs/2610.11399)

    本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。

    

    语言模型的许多应用所依赖的并非单个样本，而是模型下某个检验泛函的期望。可靠地估计这类期望在计算上可能非常昂贵。在本文中，我们展示了如何利用采样过程中作为副产品自然获得的下一词元条件概率来使估计更加高效。我们通过势函数来实现这一点：势函数是定义在前缀上的实值函数，能够对检验泛函进行加性分解。我们构造了一个估计器，其方差取决于所选择的势函数，并推导出势函数能够降低该方差的充分条件。随后，我们为若干估计对象和应用开发了实用的势函数，并在计算成本相当的情况下，在多个估计对象上展示了显著的方差缩减效果。

    arXiv:2610.11399v1 Announce Type: new  Abstract: Many applications of language models hinge not on individual samples but on the expectation of a test functional under the model. Estimating such expectations reliably can be computationally expensive. In this paper, we show how to make estimation more efficient by exploiting the next-token conditional probabilities which are available as a by-product of sampling. We do so through potentials: real-valued functions on prefixes that decompose the test functional additively. We construct an estimator whose variance depends on the chosen potential, and derive conditions under which a potential reduces this variance. We then develop practical potentials for several estimands and applications, and demonstrate substantial variance reductions across several estimands at comparable computational cost.
    
[^164]: CoPoE：基于可分解疾病坐标专家乘积的多模态融合方法，用于模态缺失下的阿尔茨海默病诊断

    CoPoE: Multimodal Fusion via Decomposable Disease-Coordinate Product-of-Experts for Missing-Modality Alzheimer's Diagnosis

    [https://arxiv.org/abs/2610.11394](https://arxiv.org/abs/2610.11394)

    提出了疾病坐标专家乘积框架CoPoE，将多模态证据映射到可解释的R/P/N/S四轴结构化潜在空间，通过掩码机制仅融合可用模态而无需合成缺失数据，从而在模态缺失情况下实现更可靠的阿尔茨海默病诊断。

    

    多模态阿尔茨海默病（AD）诊断受益于整合异质的临床、影像、基因组和生物标志物证据，但临床队列中经常存在不规则的模态缺失。现有的融合方法通常会合成缺失的输入，存在引入人工替代数据的风险，或者将可用信号汇聚到不可解释的潜在空间中。我们提出了CoPoE（疾病坐标专家乘积），这是一个疾病坐标框架，将多模态证据映射到一个结构化的潜在空间中，该空间被划分为四个不同的生物学与临床轴：遗传风险、分子病理、神经退行性变和临床分期（R/P/N/S）。每个观测到的模态参数化完整RPNS向量上的一个对角高斯专家，掩码专家乘积架构仅融合可用的模态。因此，缺失的模态不会为融合路径增加任何因子，使网络能够保持……

    arXiv:2610.11394v1 Announce Type: new  Abstract: Multimodal Alzheimer's disease (AD) diagnosis benefits from integrating heterogeneous clinical, imaging, genomic, and biomarker evidence, but clinical cohorts frequently suffer from irregular modality missingness. Existing fusion methods often synthesize absent inputs, risking the introduction of artificial surrogates, or pool available signals into uninterpretable latent spaces. We present CoPoE (Disease-Coordinate Product-of-Experts), a disease-coordinate framework that maps multimodal evidence into a structured latent space partitioned into four distinct biological and clinical axes: genetic Risk, molecular Pathology, Neurodegeneration, and clinical Stage (R/P/N/S). Each observed modality parameterizes a diagonal Gaussian expert over the full RPNS vector, and a masked Product-of-Experts architecture fuses only the available modalities. Consequently, absent modalities add no factor to the fusion path, allowing the network to preserve a
    
[^165]: PlanWAM：面向端到端自动驾驶的规划塑形未来表征

    PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving

    [https://arxiv.org/abs/2610.11382](https://arxiv.org/abs/2610.11382)

    该论文提出PlanWAM，其核心创新在于让规划任务反向塑形未来状态表征，并通过潜在世界模型预测这种规划导向的未来表征，从而为端到端自动驾驶实现真正具有前瞻性的轨迹规划。

    

    端到端自动驾驶中的世界模型通过预测未来场景演化，为轨迹规划提供前瞻性信息。现有方法主要研究如何预测未来以及如何使用未来，但较少追问哪种未来表征对规划真正最有用。为此，我们提出了PlanWAM——一个规划塑形的世界动作模型。其核心思想是让规划任务来塑形未来状态表征，使其保留对规划最有用的信息。随后，一个潜在世界模型从历史观测中预测这一规划塑形的未来潜在表征，并将其用于规划，从而实现具有前瞻性的规划。具体而言，我们首先使用时间寄存器金字塔，以近因感知的方式压缩多帧历史信息，学习面向未来推理与规划的紧凑历史表征。接着，我们引入一个特权未来后验分支，该分支观测（摘要在此处被截断）

    arXiv:2610.11382v1 Announce Type: cross  Abstract: World models in end-to-end autonomous driving predict future scene evolution to provide foresight for trajectory planning. Existing methods mainly study how to predict the future and how to use it, but less often ask which future representation is actually most useful for planning. To this end, we propose PlanWAM, a Planning-Shaped World Action Model. The key idea is to let the planning task shape the future-state representation, so that it retains the information most useful for planning. A latent world model then predicts this planning-shaped future latent representation from historical observations and uses it for planning, enabling foresighted planning. Specifically, we first use a Temporal Register Pyramid to compress multi-frame historical information in a recency-aware manner, learning a compact history representation oriented toward future reasoning and planning. We then introduce a privileged future posterior branch that obser
    
[^166]: 从提示到技能库：演化功能技能库使大语言模型实现持续学习

    From a Prompt to Repertoires: Evolving Functional REpertoires Enable LLM Continual Learning

    [https://arxiv.org/abs/2610.11373](https://arxiv.org/abs/2610.11373)

    提出演化功能技能库方法，通过将单一提示扩展为不断演化的多功能技能库，克服提示优化在持续学习中的灾难性遗忘与规则过拟合问题，使大语言模型无需更新参数即可持续习得新能力。

    

    持续学习对大语言模型而言仍是一个挑战，模型需要在获取新技能和知识的同时不损害已有能力。现有方法通常通过精心设计模型参数的更新方式来应对这一挑战。相比之下，提示优化避免了代价高昂的参数更新，在单个知识密集型和推理任务上取得了与GRPO等强化学习方法相当甚至更优的性能。这引出一个自然的问题：提示优化作为一种高效的适配方法，能否直接应用于持续学习？我们的分析表明，在顺序任务适配的场景下，提示优化会遭受灾难性遗忘，且优化后的提示会不断积累过拟合于局部任务分布的规则。为解决这些局限，我们提出了演化功能技能库，它以……替代单一提示（原文摘要在此处截断）。

    arXiv:2610.11373v1 Announce Type: cross  Abstract: Continual learning remains challenging for large language models, which must enable models to acquire new skills and knowledge without degrading existing capabilities. Existing approaches typically address this challenge by carefully designing how model parameters are updated. In contrast, prompt optimization avoids costly parameter updates while achieving competitive or even superior performance to reinforcement learning methods such as GRPO on individual knowledge-intensive and reasoning tasks. This raises a natural question: \textit{Can prompt optimization, as an efficient adaptation approach, be directly applied to continual learning?} Our analysis shows that, under sequential task adaptation, it suffers from catastrophic forgetting, while optimized prompts accumulate rules that overfit to local task distributions. To address these limitations, we propose \emph{Evolving Functional REpertoires} (EFRE), which replaces a single prompt
    
[^167]: SpatialOPSD：从经验证的编程智能体轨迹中自蒸馏空间智能

    SpatialOPSD: Self-Distilling Spatial Intelligence from Verified Coding Agent Traces

    [https://arxiv.org/abs/2610.11366](https://arxiv.org/abs/2610.11366)

    提出SpatialOPSD在线策略自蒸馏框架，将经验证的编程智能体执行轨迹作为特权信息内化到多模态大语言模型中，使其无需外部工具即可具备空间推理能力。

    

    空间编程智能体通过使用外部工具生成经验证的执行轨迹，显著提升了多模态大语言模型的空间推理能力。然而，这一范式固有地存在高昂的推理时开销和外部依赖问题。在本文中，我们探索多模态大语言模型能否将这种智能体能力内化，从而实现完全无需工具的运行。我们从一个简单的观察出发：用空间编程智能体的执行轨迹摘要来提示多模态大语言模型，能够自然地激发模型内部的空间思维链。受此启发，我们提出了SpatialOPSD，这是一种在线策略自蒸馏框架，它将经验证的智能体轨迹作为特权信息，把空间推理内化到独立的多模态大语言模型中。为了缓解蒸馏过程中的特权信息泄露问题，我们引入了重复感知蒸馏方法，将重复掩码与非似然正则化相结合。

    arXiv:2610.11366v1 Announce Type: new  Abstract: Spatial coding agents significantly improve spatial reasoning in Multimodal Large Language Models (MLLMs) by using external tools to generate verified execution traces. However, this paradigm inherently suffers from prohibitive inference-time overhead and external dependencies. In this paper, we explore whether an MLLM can internalize this agentic capability to operate entirely tool-free. We begin with a simple observation: prompting an MLLM with summarized execution traces of a spatial coding agent naturally unlocks the model's internal spatial Chain-of-Thought (CoT). Motivated by this, we introduce SpatialOPSD, an on-policy self-distillation framework that internalizes spatial reasoning into a standalone MLLM by formulating verified agent traces as privileged information. To mitigate privileged-information leakage during distillation, we introduce Repetition-Aware Distillation, which combines repetition masking with unlikelihood regula
    
[^168]: 伯努利流模型：面向二进制数据的自洽生成建模

    Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data

    [https://arxiv.org/abs/2610.11362](https://arxiv.org/abs/2610.11362)

    提出伯努利流模型（BFM），通过构建数据分布与纯噪声之间统一的连续全局伯努利概率流路径，克服了传统二值扩散模型在低函数评估次数下因单步似然近似导致样本质量严重下降的问题，无需蒸馏或额外训练即可实现高效的二值数据生成。

    

    二值扩散模型通常需要大量的函数评估次数（NFE）才能生成高质量样本，使得实际推理的计算成本高昂。在不使用蒸馏或额外训练的前提下，降低NFE同时保持样本质量仍然是一个重大挑战。现有的二值扩散模型定义了离散的单步前向路径，然后再推导反向后验分布。在需要跨步采样的低NFE场景中，它们用单步似然转移来近似真实的多步似然，这严重降低了样本质量。为了解决这一根本性局限，并将生成动力学与固定的离散时间步解耦，我们提出了伯努利流模型（BFM）。BFM不依赖于顺序式的单步马尔可夫扩散链，而是定义了一条连接数据分布与纯噪声的统一连续全局伯努利概率流路径，并从中解析推导出……（摘要在此处截断）

    arXiv:2610.11362v1 Announce Type: new  Abstract: Binary diffusion models typically require a large number of function evaluations (NFEs) to generate high-quality samples, making practical inference computationally expensive. Reducing NFEs while preserving sample quality without distillation or additional training remains a significant challenge. Existing binary diffusion models define a discrete one-step forward path and then derive the reverse posterior. In low-NFE settings requiring cross-step sampling, they approximate the true multi-step likelihood with a single-step likelihood transition, which severely degrades sample quality. To address this fundamental limitation and decouple the generative dynamics from fixed discrete time steps, we propose Bernoulli Flow Models (BFM). Rather than relying on sequential one-step Markov diffusion chains, BFM defines a unified continuous global Bernoulli probability flow path between data distributions and pure noise, from which we derive analyti
    
[^169]: RaReCache：通过基于秩分歧的选择性重计算弥合跨模型KV缓存重用的差距

    RaReCache: Bridging the Gap in Cross-Model KV Cache Reuse via Rank disagreement-based Selective Recomputation

    [https://arxiv.org/abs/2610.11358](https://arxiv.org/abs/2610.11358)

    提出RaReCache框架，利用秩分歧识别信息密集的关键token并对其进行选择性重计算，使大模型能够从小模型预填充的KV缓存中准确解码，从而实现跨不同规模模型间的高效KV缓存重用。

    

    跨模型KV缓存重用仍然是现代大语言模型（LLM）服务中的一个关键挑战。编程智能体和多模型系统越来越多地在不同模型之间路由共享上下文：用户可能在会话中途切换模型，或者级联系统可能将困难查询升级给更大的模型。由于KV缓存包含模型特定的表示，每次切换通常迫使接收模型从头开始对整个上下文进行预填充（prefill）。最近的研究表明，闭式线性映射可以在同一家族的模型之间转换KV缓存，但随着模型规模差距的扩大，迁移精度会随之下降。在本文中，我们证明了这些迁移失败集中在信息密集的一小部分token上。为了弥合这一差距，我们提出了RaReCache，一个通过选择性重计算使大型目标模型能够从由小得多的源模型预填充的缓存中准确解码的框架。RaReCache使用一种新颖的秩分歧方法来识别这些关键位置……

    arXiv:2610.11358v1 Announce Type: new  Abstract: Cross-model KV-cache reuse remains a key challenge in modern LLM serving. Coding agents and multi-model systems increasingly route a shared context across models: a user may switch models mid-session, or a cascade may escalate a difficult query. Because KV caches contain model-specific representations, each switch typically forces the receiving model to prefill the entire context from scratch. Recent work shows that closed-form linear maps can translate KV caches between models in the same family, but transfer accuracy degrades as the model-size gap widens. In this paper, we establish that these transfer failures are concentrated in a small subset of information-dense tokens. To bridge this gap, we introduce RaReCache, a framework that enables a large target model to decode accurately from a cache prefilled by a much smaller source via selective recomputation. RaReCache identifies these critical positions using a novel rank disagreement 
    
[^170]: RL-ARC：通过推理引导的不确定性校准大型推理模型

    RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty

    [https://arxiv.org/abs/2610.11352](https://arxiv.org/abs/2610.11352)

    RL-ARC提出了一种校准感知训练框架，将推理置信度作为辅助信号来校准答案置信度——对正确回答施加推理引导正则化、对错误回答施加过度自信惩罚，从而在不牺牲推理性能的情况下改善大推理模型在分布内外场景中的校准并缓解过度自信问题。

    

    语言模型（LM）通常使用可验证奖励的强化学习（RLVR）进行训练，以增强其推理能力。然而，由于RLVR在训练过程中没有明确考虑校准，可能导致严重的校准退化，包括过度自信。近年来面向语言模型的校准感知训练方法将不确定性估计目标纳入训练，虽然改善了校准，但在分布偏移下仍然表现出过度自信，同时牺牲了推理性能。为此，我们提出了RL-ARC，一个联合利用推理置信度和答案置信度的校准感知训练框架。具体而言，RL-ARC将推理置信度作为校准答案置信度的辅助信号：对于正确的回答，将其作为推理引导的正则化项；对于错误的回答，则将其作为过度自信惩罚项。在分布内（ID）和分布外（OOD）设置下的全面实验结果（摘要在此处截断）。

    arXiv:2610.11352v1 Announce Type: new  Abstract: Language models (LMs) are commonly trained with Reinforcement Learning with Verifiable Rewards (RLVR) to enhance their reasoning capabilities. However, since RLVR does not explicitly account for calibration during training, it can lead to severe calibration degradation, including overconfidence. Recent calibration-aware training methods for LMs, which incorporate objectives for uncertainty estimation into training, improve calibration but still exhibit overconfidence under distribution shift, while sacrificing reasoning performance. To this end, we propose RL-ARC, a calibration-aware training framework that jointly leverages reasoning confidence and answer confidence. Specifically, RL-ARC leverages reasoning confidence as an auxiliary signal for calibrating answer confidence, applying it as reasoning-guided regularization for correct cases and as an overconfidence penalty for incorrect cases. Comprehensive results across ID and OOD setti
    
[^171]: 样本高效的生成式保形预测

    Sample-Efficient Generative Conformal Prediction

    [https://arxiv.org/abs/2610.11349](https://arxiv.org/abs/2610.11349)

    提出CASA方法，通过刻画额外样本的边际价值，在保证边际覆盖率的前提下自适应地在不同输入间分配采样预算，从而在相同预算下获得比固定采样数量更小的不确定性集合。

    

    生成式保形预测通过条件生成器的样本构建不确定性集合，只有当样本能够很好地表示响应分布时，这些集合才是有效的。这可能需要大量样本，而每个样本的获取代价可能很高，例如在大规模扩散模型和科学模拟器中，因此必须高效利用采样预算。现有方法在每个输入上抽取相同数量的样本，在响应分布简单的地方浪费样本，而在响应分布复杂的地方采样不足，这导致集合膨胀，并使这些输入的覆盖率不足。我们提出CASA（保形自适应样本分配），该方法刻画了额外增加一个样本的边际价值，并在边际覆盖率和平均采样预算的约束下，在各输入之间分配样本以最小化期望集合大小。理论分析表明，在相同预算下，自适应分配比固定采样数量产生更小的集合：一个被遗漏的模态会迫使……（摘要截断）

    arXiv:2610.11349v1 Announce Type: new  Abstract: Generative conformal prediction builds uncertainty sets from samples of a conditional generator, which are efficient only when the samples represent the response distribution well. This can require many samples, each of which can be costly, as in large diffusion models and scientific simulators, so the sampling budget must be used efficiently. Existing methods draw the same number of samples at every input, wasting samples where the response distribution is simple and undersampling where it is complex, which inflates sets and leaves those inputs under-covered. We propose CASA (Conformal Adaptive Sample Allocation), which characterizes the marginal value of an additional sample and allocates samples across inputs to minimize the expected set size subject to marginal coverage and an average sampling budget. Theoretical analysis shows that adaptive allocation yields smaller sets than a fixed count at the same budget: a missed mode forces a 
    
[^172]: DivMoE：通过跨领域专家组合实现细粒度MoE升级

    DivMoE: Fine-Grained MoE Upcycling via Cross-Domain Expert Composition

    [https://arxiv.org/abs/2610.11317](https://arxiv.org/abs/2610.11317)

    DivMoE是首个实现结构平衡路由的细粒度MoE升级框架，通过领域专门化的细粒度专家初始化和跨领域专家组合，解决了细粒度专家从单一源模型派生时路由崩溃、准确率降至接近随机水平的问题。

    

    混合专家架构已成为扩展大型语言模型的关键技术，近期研究证明了细粒度专家设计的优势。从头训练此类模型成本高昂，而从预训练的稠密模型进行稀疏升级是一种有吸引力的替代方案。然而，我们发现了细粒度升级的一种结构性病理：当细粒度专家从单一源模型派生时，朴素路由会崩溃，下游准确率降至接近随机水平（例如，在Qwen3-1.7B上，Drop-Upcycling细粒度方法在15个基准测试中仅达到23.2%的平均准确率，基本与从头训练的22.2%持平，而同一方法的粗粒度变体则达到50.2%）。我们提出DivMoE，这是首个实现结构平衡路由的细粒度MoE升级框架。DivMoE引入了领域专门化的细粒度专家初始化，从稠密模型派生专家

    arXiv:2610.11317v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) architectures have become essential for scaling large language models, with recent work demonstrating the benefits of fine-grained expert designs. Training such models from scratch is expensive, and sparse upcycling from pre-trained dense models is an attractive alternative. However, we identify a structural pathology of fine-grained upcycling: when fine-grained experts are derived from a single source model, naive routing collapses and downstream accuracy drops to near-random (e.g., on Qwen3-1.7B, Drop-Upcycling-fine-grained reaches only 23.2% average accuracy across 15 benchmarks, essentially matching from-scratch training at 22.2%, while the same method's coarse-grained variant reaches 50.2%). We propose DivMoE, the first framework achieving fine-grained MoE Upcycling with structurally-balanced routing. DivMoE introduces domain-specialized fine-grained expert initialization, deriving experts from dense models 
    
[^173]: 收缩海森矩阵：面向多模态扩散Transformer的秩4 W4A4量化

    Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers

    [https://arxiv.org/abs/2610.11315](https://arxiv.org/abs/2610.11315)

    该论文提出一个统一框架，将低秩辅助的W4A4训练后量化建模为耦合校准问题，通过“收缩海森矩阵”对已被低秩分量捕获的残差误差进行折扣，并结合激活噪声代理抑制激活量化误差，从而仅用秩4即可实现多模态扩散Transformer的高精度4比特量化。

    

    在扩散Transformer中，低秩分支可以通过将每个权重分解为低比特残差和一个高精度的低秩分量，来缓解4比特权重-激活（W4A4）训练后量化（PTQ）带来的损失。然而，现有的低秩PTQ方法要么将低秩补偿和残差量化分开优化，通常需要更高的秩；要么依赖二阶权重更新，而没有显式建模激活量化误差，这种误差在4比特量化下尤为显著。为了解决这些局限，我们提出了一个统一框架，将低秩辅助的W4A4 PTQ建模为一个耦合校准问题，并从联合目标出发推导出基于优化的求解器。通过消除输出侧的低秩因子，得到一个“收缩海森矩阵”，它可以对已被低秩分量捕获的残差误差进行折扣，同时引入激活噪声代理来抑制（激活量化噪声的影响）……

    arXiv:2610.11315v1 Announce Type: new  Abstract: In diffusion transformers, low-rank branches can mitigate 4-bit weight--activation (W4A4) post-training quantization (PTQ) loss by decomposing each weight into a low-bit residual and a high-precision low-rank component. Existing low-rank PTQ approaches, however, either optimize low-rank compensation and residual quantization separately, often requiring higher ranks, or rely on second-order weight updates without explicitly modeling activation quantization error, which becomes particularly pronounced under 4-bit quantization. To address these limitations, we present \method{}, a unified framework modeling low-rank-assisted W4A4 PTQ as a coupled calibration problem and deriving optimization-based solvers from the joint objective. Eliminating the output-side low-rank factor yields a \emph{deflated Hessian} that discounts residual errors already captured by the low-rank component, while an activation-noise surrogate is incorporated to suppre
    
[^174]: 双隐层ReLU神经网络中最小化神经元数量的NP难性

    NP-Hardness of Minimizing Neurons in Two-Hidden-Layer ReLU Neural Networks

    [https://arxiv.org/abs/2610.11313](https://arxiv.org/abs/2610.11313)

    本文证明了在 $L^p$ 逼近约束下，精确计算双隐层ReLU神经网络逼近目标函数所需的最小隐藏神经元数量是NP难的，即使目标函数性质良好该结论依然成立。

    

    神经网络架构优化中的一个根本问题是：在给定容差范围内逼近目标函数所需的最小隐藏神经元数量能否被高效计算。本文在 $L^p(\mathbb{R}^d,\mathbb{R}^m)$ 逼近约束下解决了双隐层ReLU网络的这一问题。对于每个固定的 $d \ge 1$、$m \ge 1$ 和 $1 \le p < \infty$，我们证明精确计算该最优值是NP难的。即使目标函数由一个有理参数的ReLU网络表示，且该网络的实现为非零、逐分量非负、紧支撑、全局Lipschitz且连续分段仿射，该结果依然成立。从3-SAT出发的多项式时间归约产生了一个架构间隙：不可满足公式对应的最优值为零，而可满足公式对应的最优值至少为 $d+2$。证明过程中构造了由……实现的紧支撑多面体截锥函数（摘要原文在此处截断）。

    arXiv:2610.11313v1 Announce Type: new  Abstract: A fundamental question in neural network architecture optimization is whether the minimum hidden-neuron count required to approximate a target function within a prescribed tolerance can be computed efficiently. This paper resolves this question for two-hidden-layer ReLU networks under an $L^p(\mathbb{R}^d,\mathbb{R}^m)$ approximation constraint. For every fixed $d \ge 1$, $m \ge 1$, and $1 \le p < \infty$, we prove that computing the optimum exactly is NP-hard. The result holds even when the target is represented by a rational ReLU network whose realization is nonzero, componentwise nonnegative, compactly supported, globally Lipschitz, and continuous piecewise affine. The polynomial-time reduction from 3-SAT produces an architecture gap in which unsatisfiable formulas yield an optimum of zero, whereas satisfiable formulas yield an optimum of at least $d+2$. The proof constructs compactly supported polyhedral frustum functions realized by
    
[^175]: FloorSAV：利用2D平面地图为视听大语言模型阐明空间视听上下文

    FloorSAV: Elucidating Spatial Audio-Visual Context with 2D Floormap for AV-LLMs

    [https://arxiv.org/abs/2610.11310](https://arxiv.org/abs/2610.11310)

    提出FloorSAV框架，通过渲染整合了3D点云、相机轨迹、空间音频与物体地标的动态2D平面地图，并将其作为同步流注入视听大语言模型，使其无需昂贵微调即可在单次推理中联合推理视觉、听觉与几何线索。

    

    尽管动态自我中心环境中的3D空间推理对具身智能至关重要，但视听大语言模型缺乏直接从原始感知流中处理和内化全局几何信息的显式机制。现有方法要么需要代价高昂的微调，要么未能充分利用模型的跨模态推理能力。在本文中，我们提出了FloorSAV，这是一种通过渲染动态2D平面地图来显式锚定空间视听上下文的新颖框架。通过整合3D点云、相机轨迹、空间音频线索以及语义锚定的物体地标，我们将该平面地图作为与自我中心视频同步的流注入到视听大语言模型中。视听大语言模型利用其多模态能力，在平面地图解读引导下，于单次推理中联合推理视觉、听觉和几何线索。我们进一步引入了SAVED-Bench（空间视听自我中心基准）

    arXiv:2610.11310v1 Announce Type: cross  Abstract: While 3D spatial reasoning in dynamic egocentric environments is crucial for embodied intelligence, audio-visual large language models (AV-LLMs) lack explicit mechanisms to process and internalize global geometry directly from raw sensory streams. Existing approaches either require costly fine-tuning or underutilize the model's cross-modal reasoning capacities. In this paper, we propose FloorSAV, a novel framework that explicitly grounds spatial audio-visual context by rendering a dynamic 2D floormap. By integrating 3D point clouds, camera trajectories, spatial audio cues, and semantically grounded object landmarks, we inject this floormap into the AV-LLM as a synchronized stream with an egocentric video. AV-LLMs utilize their multi-modal capabilities to jointly reason over visual, auditory, and geometric cues in a single inference with floormap interpretation guidance. We further introduce SAVED-Bench (Spatial Audio-Visual Egocentric 
    
[^176]: 从几何到泛化：为什么行归一化能够胜过Adam和Muon

    From Geometry to Generalization: Why Row Normalization Can Beat Adam and Muon

    [https://arxiv.org/abs/2610.11309](https://arxiv.org/abs/2610.11309)

    该论文证明了在高维多分类任务中，行归一化凭借其类级欧几里得几何能够渐近保持总体决策边界方向，从而在总体精度上严格超越采用坐标级几何的Adam和采用谱几何的Muon等优化器。

    

    不同的优化器在拟合相同训练数据的同时，会选择出几何结构差异显著的分类器，但这种差异是否能被证明会影响总体性能仍不清楚。我们证明，在高维多分类任务中，按行归一化（row-wise normalization）可以获得严格高于全批量Adam（作为随机重排Adam的代理）以及精确SVD Muon的总体精度。在各向同性高斯云数据模型下，这一优势源于行归一化的类级欧几里得几何能够渐近地保持总体决策边界方向，而Adam的坐标级几何和Muon的谱几何则会引入不可忽略的失真。超越各向同性情形，对于在类均值上进行全批量训练、且类均值协方差与测试噪声协方差方向相互独立的情况，该优势依然成立；对于类均值指数小于1的幂律谱分布，即使在严重的（各向异性条件下）该优势同样保持……（摘要原文在此处截断）

    arXiv:2610.11309v1 Announce Type: cross  Abstract: Different optimizers can fit the same training data while selecting classifiers with substantially different geometries, but whether this difference provably affects population performance remains unclear. We show that row-wise normalization can achieve strictly higher population accuracy than full-batch Adam, a proxy for random-reshuffling Adam, and exact-SVD Muon in high-dimensional multiclass classification. Under an isotropic Gaussian-cloud data model, this advantage arises because row normalization's class-wise Euclidean geometry asymptotically preserves the population decision-boundary directions, whereas Adam's coordinate-wise geometry and Muon's spectral geometry introduce nonvanishing distortions. Beyond isotropy, the advantage persists for full-batch training on class means with independently oriented class-mean and test-noise covariances. It holds for power-law spectra with class-mean exponent below one, even under heavily a
    
[^177]: 具有随机特征的条件扩散模型中的记忆化与恶性泛化

    Memorization and Malign Generalization in Conditional Diffusion Models with Random Features

    [https://arxiv.org/abs/2610.11288](https://arxiv.org/abs/2610.11288)

    本文在高维比例极限下分析随机特征条件评分模型，揭示了条件扩散模型在过参数化时存在“恶性泛化”现象——增加模型宽度虽改善条件均值预测却降低条件内方差，且条件信息量越大，模型在更小宽度下就会记忆训练样本。

    

    条件扩散模型能够在给定条件下生成多样化、新颖且高质量的样本。然而，对其记忆化与泛化行为的理论理解仍然有限，而近期的工作主要在无条件设定下刻画了这些行为。在本工作中，我们在高维比例极限下分析了一个随机特征条件评分模型，推导出了训练损失和测试损失的渐近表达式。通过分解测试损失，我们证明了在过参数化区域中，增加模型宽度可以改善对条件相关均值的预测，同时降低条件内的预测方差，我们将这一现象称为“恶性泛化”。此外，对训练损失的分析表明，条件的信息量越大，模型就会在更小的宽度下对训练样本产生记忆化。这些理论发现得到了在真实数据集上使用U-Net架构实验的支持。

    arXiv:2610.11288v1 Announce Type: new  Abstract: Conditional diffusion models generate diverse, novel, and high-quality samples under prescribed conditions. However, theoretical understanding of their memorization and generalization remains limited, while recent works have characterized these behaviors primarily in unconditional settings. In this work, we analyze a random-feature conditional score model in the high-dimensional proportional limit, deriving asymptotic expressions for training and test losses. By decomposing the test loss, we show that in the overparameterized regime, increasing model width improves prediction of the condition-dependent mean while reducing within-condition prediction variance, a phenomenon we term "malign generalization." Furthermore, analyzing the training loss reveals that more informative conditions lead to memorization of training samples at smaller widths. These theoretical findings are supported by experiments with U-Net architectures on realistic d
    
[^178]: Being-M0.7：面向人形机器人的潜在世界-动作模型

    Being-M0.7: A Latent World-Action Model for Humanoid Robots

    [https://arxiv.org/abs/2610.11283](https://arxiv.org/abs/2610.11283)

    Being-M0.7 提出了一种潜在世界-动作模型，通过三阶段训练（预训练、机器人中期训练、动作后训练），将超过1万小时混合模态人类数据（纯视频、纯运动及成对视频-运动）中学习到的视觉-运动先验迁移到人形机器人的全身移动-操作控制中。

    

    人形机器人的移动-操作任务需要在未来场景演化和全身运动信息的引导下，协调地完成移动与操作，然而学习这些能力受到机器人示范数据稀缺的制约。人类视频和运动数据集提供了可扩展的监督信号，但许多数据集仅包含视频或运动数据，而非成对的视频-运动数据。此外，人类运动并不能直接转化为可执行的机器人动作。我们提出 Being-M0.7，一个潜在世界-动作模型，它通过预训练、机器人中期训练和动作后训练三个阶段，将从混合模态人类数据中学到的视觉-运动先验迁移到人形机器人控制中。我们从超过10,000小时的原始人类中心数据中整理出一个语料库，整合了纯视频、纯运动和成对视频-运动数据流，以学习互补的视觉动态与全身运动学结构。对未来潜在视觉状态与运动的联合预测鼓励视觉表示

    arXiv:2610.11283v1 Announce Type: cross  Abstract: Humanoid loco-manipulation requires coordinated locomotion and manipulation informed by future scene evolution and whole-body motion, yet learning these capabilities is constrained by scarce robot demonstrations. Human video and motion datasets offer scalable supervision, but many contain only video or motion rather than paired video-motion data. Moreover, human motion does not directly specify executable robot actions. We present Being-M0.7, a latent world-action model that transfers visual-motion priors learned from mixed-modality human data to humanoid control through pre-training, robot mid-training, and action post-training. We curate a corpus from more than 10,000 hours of raw human-centric data, integrating video-only, motion-only, and paired video-motion streams to learn complementary visual dynamics and whole-body kinematic structure. Joint prediction of future latent visual states and motion encourages visual representations 
    
[^179]: 如何在代理奖励上进行后训练：包络采样缓解奖励破解

    How to post-train on a surrogate: Envelope sampling mitigates reward hacking

    [https://arxiv.org/abs/2610.11281](https://arxiv.org/abs/2610.11281)

    提出“包络采样”方法，利用少量带真实标注的输出对LLM评判器进行有理论保证的重新校准，从而在强化学习后训练中有效缓解奖励破解问题。

    

    大型语言模型（LLM）通常针对LLM评判器（judge）及其他廉价代理奖励进行后训练，因为真实奖励（如人类偏好）的大规模获取成本过高。这种做法常常导致“奖励破解”问题，即针对校准不当的代理奖励进行强化学习会产生不良副作用。在本工作中，我们研究这样一种设置：对少量（n个）模型输出标注真实标签（例如来自专家审查），并利用这些数据在优化前对LLM评判器进行重新校准。已有的评判器重新校准方法要么成本高昂，要么依赖启发式规则；而且已知当代理奖励在一小部分稀有输出上校准不当时，同策略采样方法会失效。我们提出了“包络采样”，这是一种具有理论依据的评判器重新校准方法，其目标是在人类奖励与重新校准后的评判器所满足的假设条件下，最小化后训练模型的遗憾值上界。（注：原摘要在此处截断）

    arXiv:2610.11281v1 Announce Type: cross  Abstract: Large language models (LLMs) are commonly post-trained against LLM judges and other cheap surrogates because the true reward, such as human preference, is too expensive to query at scale. This practice often leads to reward hacking, where reinforcement learning against a miscalibrated surrogate leads to undesirable side effects. In this work, we study a setting in which a small number $n$ of model outputs are annotated with ground-truth labels (e.g., from expert review) and used to recalibrate the LLM judge before optimizing against it. Prior approaches to judge recalibration are costly or heuristic, and it is known that on-policy sampling fails when the surrogate is miscalibrated on a rare set of outputs. In this work, we propose envelope sampling, a theoretically-grounded method for judge recalibration that seeks to minimize an upper bound on the regret of the post-trained model under the assumption that the human reward and re-calib
    
[^180]: 多语言语音模型中的音系干扰

    Phonological Interference in Multilingual Speech Models

    [https://arxiv.org/abs/2610.11275](https://arxiv.org/abs/2610.11275)

    该研究揭示了多语言语音模型中的“音系干扰”这一系统性失败模式——模型错误地假设输入属于单一语言并强加其音系，导致在语码转换语音上丢失32%至79%的语言特有音素。

    

    音素级模型将语音转录或生成为音素序列——音素是区分单词的最小声音单位。这些模型能够实现细粒度的发音控制与理解，但在处理不匹配任何单一训练语言的输入时常常失败，例如在两种语言之间交替的语音（即语码转换），或训练数据中未包含的低资源语言。我们识别出这一现象背后的一个系统性失败模式——音系干扰：模型假设输入属于单一语言，并将该语言的音系强加于输入之上，从而覆盖了与所假设语言相冲突的局部音素级决策。我们通过模型保留“一种语言具有而另一种语言缺乏”的音素的频率来衡量干扰程度。在语码转换输入上，两个音素识别器（语音到音素模型）和一个音素条件控制的文本到语音模型会丢失32%至79%的此类音素，而两种语言共有的音素则丢失得少得多。

    arXiv:2610.11275v1 Announce Type: new  Abstract: Phoneme-level models transcribe or generate speech as a sequence of phonemes, the smallest sound units that distinguish words. These models enable fine-grained pronunciation control and understanding, yet often fail on input that does not match any single training language, such as speech alternating between two languages, known as code-switching, or low-resource languages absent from training. We identify a systematic failure mode behind this, phonological interference: models assume the input is in a single language and impose its phonology, overriding local phoneme-level decisions that conflict with the assumed language. We measure interference by how often a model retains phonemes that one language has but the other lacks. On code-switched input, two phone recognizers (speech-to-phoneme models) and a phoneme-conditioned text-to-speech model lose 32% to 79% of these phonemes, but lose far fewer of the phonemes both languages share. On
    
[^181]: 门控记忆：面向对话式AI的准入控制式记忆形成

    Gated Memory: Admission-Controlled Memory Formation for Conversational AI

    [https://arxiv.org/abs/2610.11270](https://arxiv.org/abs/2610.11270)

    该论文提出Gated Memory框架，通过在对话与存储之间设置准入控制检查点，在事实提取前基于完整话语上下文评估候选事实，解决了关键上下文信号在提取时不可逆丢失这一制约记忆质量的瓶颈问题。

    

    个性化对话式AI依赖于长期记忆系统，该系统从用户话语中提取事实并将其存储在持久化向量库中。尽管在检索、去重和生命周期管理方面已取得进展，但记忆形成阶段——即事实首次写入存储的时刻——几乎未受到任何系统性的关注。我们将此识别为生产系统中记忆质量的关键制约因素。关键上下文信号，例如用户的永久属性与临时情境之间的区分，仅存在于原始话语中，并且在提取过程生成“主语-关系-宾语”三元组的那一刻便不可逆地丢失了，任何下游流程都无法将其恢复。我们提出Gated Memory（门控记忆），一个轻量级、模块化的记忆形成框架，它在对话与存储之间引入两个决策检查点：其中一个是准入门，在提取之前根据完整话语上下文评估每个候选事实……

    arXiv:2610.11270v1 Announce Type: cross  Abstract: Personalized conversational AI relies on long-term memory systems that extract facts from user utterances and store them in persistent vector stores. Despite progress in retrieval, deduplication, and lifecycle management, the formation stage, the moment a fact is first written to storage has received almost no principled attention. We identify this as the binding constraint on memory quality in production systems. Critical contextual signals, such as the distinction between a permanent user attribute and a transient situation, exist only in the original utterance and are irreversibly lost the moment extraction produces a subject-relation-object triple. No downstream process can recover them. We propose Gated Memory, a lightweight, modular formation framework that interposes two decision checkpoints between conversation and storage: an admission gate that evaluates every candidate fact against the full utterance context before extractio
    
[^182]: Q-Capsule：一种用于缓解贫瘠高原问题的局部化胶囊型量子神经网络架构

    Q-Capsule: A Localized Capsule-Based Quantum Neural Architecture for Barren Plateau Mitigation

    [https://arxiv.org/abs/2610.11261](https://arxiv.org/abs/2610.11261)

    Q-Capsule是一种局部化胶囊型量子神经网络架构，通过寄存器分区、局部读出、稀疏胶囊间耦合和QFIM引导的自适应深度增长来缓解贫瘠高原问题，使梯度方差在量子比特规模增大时保持稳定。

    

    变分量子算法常常受到贫瘠高原问题的限制：随着电路规模和深度的增加，梯度会消失，使得量子神经网络难以训练。我们提出了Q-Capsule，这是一种局部化的胶囊型量子神经网络架构，通过寄存器分区、局部读出、稀疏的胶囊间耦合、可训练的数据重上传以及量子费舍尔信息矩阵（QFIM）引导的自适应深度增长来缓解这一问题。通过将每个可观测量的主要支撑限制在一个小胶囊内并控制胶囊间的纠缠，Q-Capsule在保留局部量子表示之间通信的同时，保持了有用的梯度信号。随着寄存器宽度的增加，Q-Capsule始终维持稳定的梯度方差，而全局纠缠的基线方法则表现出指数级抑制，其对数梯度方差斜率接近每量子比特 -ln 2。Q-Capsule还产生了更有结构的……

    arXiv:2610.11261v1 Announce Type: cross  Abstract: Variational quantum algorithms are often limited by barren plateaus: gradients vanish as circuit size and depth increase, making quantum neural networks difficult to train. We propose Q-Capsule, a localized capsule-based quantum neural architecture that mitigates this problem through register partitioning, local readout, sparse inter-capsule coupling, trainable data re-uploading, and Quantum Fisher Information Matrix (QFIM)-guided adaptive depth growth. By restricting the dominant support of each observable to a small capsule and controlling inter-capsule entanglement, Q-Capsule preserves useful gradient signals while retaining communication between local quantum representations. As the register width increases, Q-Capsule consistently maintains stable gradient variance, whereas globally entangling baselines exhibit exponential suppression with a log-gradient-variance slope near -ln 2 per qubit. Q-Capsule also produces more structured o
    
[^183]: 表征学习中的残余谱不稳定性

    Residual spectral instabilities in representation learning

    [https://arxiv.org/abs/2610.11257](https://arxiv.org/abs/2610.11257)

    该论文将VAE中的逐维后验坍缩表述为围绕部分坍缩态的涨落理论，通过条件残差算符推导出坍缩方向的精确质量谱，并给出当解码器方差低于残差谱上边缘时坍缩维度可被再激活的临界判据。

    

    学到的表征可能会相继丢失潜在自由度，这暗示着一种级联式的转变过程，而其背后的稳定性原理仍不清楚。本文将变分自编码器（VAE）中逐维发生的后验坍缩表述为围绕部分坍缩态的涨落理论。将负的证据下界解释为有效自由能，其二阶展开定义了一个高斯理论，其Hessian矩阵充当潜在涨落的质量矩阵。我们证明坍缩的方向构成一个不变的涨落扇区，并借助条件残差算符推导出其精确的质量谱。当解码器方差低于残差谱的上边缘时，存在一个降低自由能的局域再激活方向，等号成立时即标记临界状态。该判据在线性高斯VAE极限下恢复为主成分阈值。若沿连续相连的分支反向观察……

    arXiv:2610.11257v1 Announce Type: new  Abstract: Learned representations can lose latent degrees of freedom successively, suggesting a cascade of transitions whose underlying stability principle remains unclear. Here we formulate dimension-wise posterior collapse in variational autoencoder (VAE) as a fluctuation theory around partially collapsed states. Interpreting the negative evidence lower bound as an effective free energy, its quadratic expansion defines a Gaussian theory whose Hessian acts as a mass matrix for latent fluctuations. We show that the collapsed directions form an invariant fluctuation sector and derive its exact mass spectrum in terms of a conditional residual operator. A local reactivation direction lowers the free energy when the decoder variance falls below the residual spectral upper edge, with equality marking marginality. The criterion recovers principal component thresholds in the linear Gaussian VAE limit. Viewed in reverse along continuously connected branch
    
[^184]: 用于模拟类人跟车行为的神经-记忆模糊推理系统

    Neuro-Memory Fuzzy Inference System for Mimicking Human-like Car Following Behavior

    [https://arxiv.org/abs/2610.11252](https://arxiv.org/abs/2610.11252)

    本研究提出神经-记忆模糊推理系统（NeMeFIS），通过整合五种人类记忆类型对跟车行为的加减速进行非对称建模，在复现真实驾驶行为方面优于线性回归、ANFIS和LSTM等传统模型。

    

    本研究提出了神经-记忆模糊推理系统，这是一种分层机器学习架构，通过整合五种人类记忆类型——程序性记忆、工作记忆、情景记忆、语义记忆和陈述性记忆——对跟车行为中的加速与减速进行非对称建模。通过元启发式方法将外部变量与记忆功能相关联，并借助因子分析和p值分析加以验证，NeMeFIS揭示了不同类型车辆在主干道、集散道路和乡村公路走廊上的潜在认知影响。54个不同训练模型的结果强调了由驾驶员感知极限和认知负荷塑造的认知阈值。训练后的NeMeFIS模型在复现真实驾驶行为方面优于传统统计模型和常规机器学习模型，包括与线性回归、ANFIS和LSTM架构的比较。模糊规则分析揭示了陈述性……

    arXiv:2610.11252v1 Announce Type: cross  Abstract: This study presents the Neuro-Memory Fuzzy Inference System (NeMeFIS), a hierarchical machine learning architecture that asymmetrically models acceleration and deceleration in car following behavior by integrating five human memory types procedural, working, episodic, semantic, and declarative. By linking external variables to memory functions via metaheuristics and validating them through factor and p-value analyses, NeMeFIS uncovers latent cognitive influences across Arterial, Collector, and Rural Highway corridors for different types of vehicles. Results from 54 different trained models emphasize cognitive thresholds shaped by driver perception limits and cognitive load. The trained NeMeFIS models outperform traditional statistical and conventional machine learning models in replicating realistic driving behavior, including comparisons with Linear Regression, ANFIS, and LSTM architectures. Fuzzy rule analysis reveals that declarativ
    
[^185]: V-CoLA：基于线性注意力的视觉Token压缩

    V-CoLA: Vision Token Compression with Linear Attention

    [https://arxiv.org/abs/2610.11251](https://arxiv.org/abs/2610.11251)

    提出V-CoLA，一个专为线性注意力混合架构设计的无需训练的视觉Token压缩框架，通过唯一性感知的重要性准则和自适应Token合并策略，解决了现有压缩方法在该类架构下性能显著下降的问题。

    

    视觉语言模型（VLM）已经展现出令人印象深刻的能力，但由于视觉Token在输入序列中占主导地位，导致计算开销巨大。这促使视觉Token压缩成为减轻负担的关键方向。然而，随着融合线性注意力的混合架构（例如Qwen3.5）的出现，先前为softmax注意力设计的方法难以泛化。我们的分析表明，基于注意力和基于相似度的方法都出现了显著的性能下降，凸显了针对这一机制量身定制压缩方法的迫切需求。为此，我们提出了V-CoLA，一个专门为线性注意力设计的高效无需训练的Token压缩框架。V-CoLA引入了一种新颖的“唯一性感知重要性准则”用于识别关键的视觉Token，并结合一种“自适应Token合并策略”来执行压缩。

    arXiv:2610.11251v1 Announce Type: cross  Abstract: Vision-language models (VLMs) have demonstrated impressive capabilities but suffer from substantial computational overhead, as vision tokens dominate the input sequence. This motivates vision token compression as a key direction to alleviate the burden. However, with the emergence of hybrid architectures incorporating linear attention (\eg, Qwen3.5), prior methods designed for softmax attention struggle to generalize. Our analysis reveals that both attention- and similarity-based approaches suffer notable performance degradation, underscoring the urgent need for compression methods tailored to this regime. To this end, we propose \textbf{V-CoLA}, an efficient training-free token compression framework specifically designed for linear attention. V-CoLA introduces a novel \textit{uniqueness-aware importance criterion} for identifying critical vision tokens, coupled with an \textit{adaptive token merging strategy} that performs compression
    
[^186]: 为什么在策略蒸馏有时会失败：学习信号的消失

    Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals

    [https://arxiv.org/abs/2610.11247](https://arxiv.org/abs/2610.11247)

    该研究发现大规模教师模型在在策略蒸馏中会导致基于梯度的学习信号过早消失，从而造成早期损失平台期，并从理论上为足够接近初始学生的教师证明了学习信号的局部恢复保证。

    

    在策略蒸馏（OPD）能够在语言模型之间实现有效的能力迁移，但其失败的内在机制尚未被完全理解。在代码生成和数学推理任务中，使用更大规模教师的OPD表现出早期的损失平台期：经过200次更新后，平均最终损失仅降低25.1%，而自我强化学习教师（通过对初始学生进一步进行强化学习训练得到）的损失降低率则达到96.2%。为了理解这一差异，我们在小学习率极限下将OPD分析为一个理想化的连续时间动力系统。我们的训练日志诊断将这些平台期与基于梯度的学习信号代理在仍有大量损失存在时的早期下降相关联；但这些测量并未解释底层梯度为何会减弱。我们进一步证明了在共享参数化下，对于与初始学生足够接近的教师，存在局部恢复保证……（原文摘要在此处截断）

    arXiv:2610.11247v1 Announce Type: cross  Abstract: On-policy distillation (OPD) enables effective capability transfer between language models, yet the mechanisms underlying its failures are not fully understood. Across code generation and mathematical reasoning, OPD with larger-scale teachers exhibits early loss plateaus, with an average final loss reduction of 25.1% after 200 updates, compared with 96.2% for self-RL teachers, obtained by further reinforcement learning (RL) training of the initial student. To understand this difference, we analyze OPD as an idealized continuous-time dynamical system in the small-learning-rate limit. Our training-log diagnostics associate these plateaus with an early decline in a gradient-based learning-signal proxy while substantial loss remains; these measurements do not establish why the underlying gradient weakens. We further prove a local recovery guarantee for teachers sufficiently close to the initial student in a shared parameterization under re
    
[^187]: 阅读关键内容：面向KV缓存的查询自适应量化

    Read What Matters: Query-Adaptive Quantization for KV Caches

    [https://arxiv.org/abs/2610.11245](https://arxiv.org/abs/2610.11245)

    提出ReadKV方法，通过渐进式编码实现KV缓存的查询自适应量化，根据每个解码查询动态分配键通道和值令牌的读取精度，并从理论上证明查询依赖的读取方案在相同读取预算下严格优于任何查询无关的方案。

    

    KV缓存条目在其未来查询尚未可知时就被存储，但每个解码查询在不同位置需要不同的精度。我们使用“保留比特数”与“每次查询获取比特数”这两个独立预算来研究这种失配问题。ReadKV将每个键和值存储在渐进式编码中，其前缀可支持不同的重建精度。对于每个查询，它先利用该查询分配键通道前缀，从重建的键计算注意力，然后再利用该注意力分配值令牌前缀，而存储的条目本身保持不变。每个阶段都在固定预算下优化一个经过校准的失真目标；我们证明了在细化增益递减条件下的最优精确分配，并将这些目标与注意力输出误差联系起来。我们还构造了一个有限维注意力族，证明在相同的读取预算下，查询依赖的访问严格优于所有查询无关的读取器，即使竞争方的编码器和解码器不受任何限制。

    arXiv:2610.11245v1 Announce Type: cross  Abstract: KV-cache entries are stored before their future queries are known, but each decoding query needs precision in different places. We study this mismatch using separate budgets for retained bits and bits fetched per query. ReadKV stores each key and value in a progressive code whose prefixes support different reconstruction precisions. For each query, it allocates key-channel prefixes using the query, computes attention from the reconstructed keys, and then allocates value-token prefixes using that attention. Stored entries remain unchanged. Each stage optimizes a calibrated distortion objective under a fixed budget; we prove exact allocation under diminishing refinement gains and relate these objectives to attention-output error. We also exhibit a finite-dimensional attention family where query-dependent access strictly outperforms every query-independent reader at the same read budget, even with unrestricted competing encoders and decod
    
[^188]: 面向室内空气质量监测的低成本传感器校准：数据集、评估场景与轻量级模型

    Low-Cost Sensor Calibration for Indoor Air Quality Monitoring: A Dataset, Evaluation Scenarios, and a Lightweight Model

    [https://arxiv.org/abs/2610.11236](https://arxiv.org/abs/2610.11236)

    该论文贡献了一个为期六个月、涵盖五个地点的低成本与参考室内空气质量传感器数据集，定义了评估空间泛化与时间漂移鲁棒性的四种校准评估场景，并提出了一种结合输入窗口压缩与残差机制的轻量级时间模型，从而克服了传统成对校准方法的局限性。

    

    低成本传感器能够实现可扩展的室内空气质量监测，但由于非线性失真、噪声和时间漂移，需要进行校准。传统严格的成对校准设置要求在每个部署位置都配置一个同址参考传感器，并且没有考虑空间和时间上的异质性。为了解决这些局限性，我们引入了一个为期六个月的数据集，其中包含在五个地点收集的低成本传感器与参考传感器的多变量室内空气质量测量数据以及上下文元数据。基于该数据集，我们定义了四种评估场景。其中，参考高效场景和位置迁移场景用于评估空间泛化能力，而长期漂移场景和事件条件场景则用于评估模型对渐进性和突变性分布偏移的鲁棒性。基于这些场景，我们推导出设计需求，并提出了一种轻量级时间模型，该模型将输入窗口压缩与残差……（原文摘要在此处截断）

    arXiv:2610.11236v1 Announce Type: new  Abstract: Low-cost sensors enable scalable indoor air quality monitoring but require calibration because of nonlinear distortions, noise, and temporal drift. The conventional strict pairwise calibration setting requires a co-located reference sensor at each deployment location and does not account for spatial and temporal heterogeneity. To address these limitations, we introduce a six-month dataset comprising multivariate indoor air-quality measurements from low-cost and reference sensors with contextual metadata collected at five locations. Using this dataset, we define four evaluation scenarios. The reference-efficient and location-transfer scenarios evaluate spatial generalization, whereas the long-term drift and event-conditioned scenarios assess robustness to gradual and abrupt distribution shifts. Based on these scenarios, we derive design requirements and propose a lightweight temporal model that combines input-window compression with resid
    
[^189]: SteerCast：面向仅解码器时间序列预测的基于检索的潜空间引导方法

    SteerCast: Retrieval-Based Latent Steering for Decoder-Only Time Series Forecasting

    [https://arxiv.org/abs/2610.11229](https://arxiv.org/abs/2610.11229)

    SteerCast提出了一种无需更新模型参数的推理时增强方法，通过检索相似历史窗口对应的引导向量并在自回归生成的每一步将其注入仅解码器预测模型的隐藏状态，从而引导预测朝着与相似训练案例一致的方向改进。

    

    时间序列预测旨在根据历史观测值和辅助特征来预测未来值。我们提出了SteerCast，这是一种基于检索的潜空间引导方法，能够在推理阶段改进仅解码器预测模型，而无需更新其参数。SteerCast从训练集构建一个数据库，存储每个历史窗口的表示，以及在该预测模型潜空间中计算得到的“引导向量”——该向量定义为真实未来序列所诱导的表示与模型自身预测所诱导的表示之间的差异。在测试阶段，SteerCast为查询历史检索最近邻样本，聚合它们的引导向量，并将所得信号注入到自回归生成每一步中预测模型的隐藏状态里，从而引导预测趋向与相似训练案例一致的轨迹。在多样化多变量基准数据集和多个预测时间范围上的实验表明……

    arXiv:2610.11229v1 Announce Type: new  Abstract: Time series forecasting aims to predict future values from historical observations and auxiliary features. We propose \textbf{SteerCast}, a retrieval-based latent steering method that improves decoder-only forecaster at inference time, without updating its parameters. SteerCast constructs a database from the training set by storing a representation of each history window together with a \emph{steering vector} computed in the forecaster's latent space, defined as the difference between representations induced by the ground-truth continuation and by the model's own prediction. At test time, SteerCast retrieves nearest neighbors for a query history, aggregates their steering vectors, and injects the resulting signal into the forecaster's hidden states at every step of autoregressive generation, guiding predictions toward trajectories consistent with similar training cases. Experiments across diverse multivariate benchmarks and multiple hori
    
[^190]: 基于协同过滤路径的多模态图检索增强序列推荐

    Multimodal Graph Retrieval-Augmented Sequential Recommendation via Collaborative Filtering Paths

    [https://arxiv.org/abs/2610.11228](https://arxiv.org/abs/2610.11228)

    该论文提出MGRASRec框架，通过从经多模态相似度扩展的用户-物品交互图中检索协同过滤路径并注入MLLM提示，在引入邻居用户协同信号的同时无需额外开销即可筛选出与候选最相关的历史物品，从而高效提升多模态序列推荐性能。

    

    多模态大语言模型（MLLMs）凭借其对复杂多模态数据进行推理的能力，在序列推荐任务中展现出强大的潜力。然而，现有方法要么仅依赖目标用户自身的交互历史，忽略了来自邻近用户的协同信号；要么需要对长交互历史进行反复的MLLM推理，带来巨大的计算开销。为了应对这些挑战，我们提出了MGRASRec，一个用于序列推荐的多模态图检索增强框架。MGRASRec通过从用户-物品交互图中检索结构化路径，将以候选物品为条件的协同过滤信号直接注入到MLLM的提示中，并通过多模态相似度对该图进行扩展，从而将覆盖范围提升至超越精确共交互重叠的程度。这种检索方式还能在不产生额外成本的情况下，筛选出与候选物品最相关的历史物品，从而免除了……

    arXiv:2610.11228v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) have demonstrated strong potential for sequential recommendation through their ability to reason over complex multimodal data. However, existing approaches either rely solely on the target user's own interaction history, neglecting collaborative signals from neighboring users, or incur substantial computational overhead through repeated MLLM inference over long interaction histories. To address these challenges, we propose MGRASRec, a multimodal graph retrieval-augmented framework for sequential recommendation. MGRASRec injects collaborative filtering signals conditioned on the candidate item directly into the MLLM prompt by retrieving structured paths from a user-item interaction graph, extended via multimodal similarity to increase coverage beyond exact co-interaction overlap. This retrieval also surfaces the history items most relevant to the candidate at no additional cost, removing the need f
    
[^191]: 当更低的重构损失反而有害：面向低比特大语言模型量化的分布鲁棒精炼方法

    When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization

    [https://arxiv.org/abs/2610.11226](https://arxiv.org/abs/2610.11226)

    本文发现更低的重构损失不一定带来更好的模型性能甚至可能有害，并提出分布鲁棒量化（DRQ）方法，通过在受约束的激活分布集合上最小化最坏情况重构损失来精炼量化权重编码，从而提升低比特大语言模型量化的效果。

    

    仅权重训练后量化（PTQ）在很大程度上依赖重构损失最小化来在低精度下保持模型质量。我们表明，通过最小化该损失所选择的权重不一定能在新任务上带来更好的模型性能。事实上，我们发现更低的重构损失甚至可能在相同的校准数据上降低模型性能。我们的分析进一步表明，在校准数据上重构损失更低的权重，当输入激活分布发生变化时，其损失可能高于其他权重。受这些观察和分析的启发，我们提出了分布鲁棒量化（DRQ），这是一种事后精炼过程，在受约束的输入激活分布集合上最小化最坏情况下的重构损失。DRQ在现有量化网格内对表示量化权重的整数编码进行精炼，同时保持量化参数和推理算子不变。

    arXiv:2610.11226v1 Announce Type: new  Abstract: Weight-only post-training quantization (PTQ) relies heavily on reconstruction loss minimization to preserve model quality at low precision. We show that the weights favored by minimizing this loss need not yield better model performance on new tasks. In fact, we find that lower reconstruction loss can even degrade model performance on the same calibration data. Our analysis further shows that weights with lower reconstruction loss on calibration data can have higher loss than other weights when the distribution of input activations changes. Motivated by these observations and our analysis, we propose Distributionally Robust Quantization (DRQ), a post-hoc refinement process that minimizes worst-case reconstruction loss over a constrained set of input activation distributions. DRQ refines the integer codes representing quantized weights within the existing quantization grid, keeping quantization parameters and inference operators unchanged
    
[^192]: 跨物种表征学习对齐小鼠与人类神经动力学并追踪临床药物疗效

    Cross-species representation learning aligns mouse and human neural dynamics and tracks clinical drug efficacy

    [https://arxiv.org/abs/2610.11222](https://arxiv.org/abs/2610.11222)

    该研究提出双规则对比学习框架，通过跨物种表征学习从电生理数据中对齐小鼠与人类的神经动力学，识别保守的疾病相关特征并追踪临床药物疗效。

    

    临床前模型难以预测人体药物疗效，这在神经系统疾病中尤为突出。神经活动提供了一种极为丰富的转化医学信息来源，因为它捕捉了神经系统功能的高维变异，且可在动物和人类中进行测量。然而，其高维度使得难以区分保守的疾病相关特征与源于物种、记录模态和实验情境的差异。在此，我们测试了能否通过学习以生物状态而非物种为组织方式的表征，直接从电生理数据中识别共享的神经动力学。我们开发了一种双规则对比学习框架，在对齐相应的小鼠与人类状态的同时，保留不同表型之间的区分。该框架恢复了跨物种保守的感觉响应结构，并在癫痫研究中解析了……（摘要截断）

    arXiv:2610.11222v1 Announce Type: cross  Abstract: Preclinical models poorly predict human drug efficacy, particularly in neurological disorders. Neural activity offers a uniquely rich source of translational information because it captures high-dimensional variation in nervous-system function that can be measured in both animals and humans. However, its high dimensionality makes it difficult to distinguish conserved disease-related features from variation arising from species, recording modality and experimental context. Here, we test whether shared neural dynamics can be identified directly from electrophysiology data by learning representations organized by biological state rather than species. We develop a dual-rule contrastive learning framework that aligns corresponding mouse and human states while preserving separation between distinct phenotypes. This framework recovered conserved sensory-response structure across species and, in epilepsy, resolved distinct relationships betwee
    
[^193]: BRACE：基于LSR能量的密集联想记忆的差分隐私

    BRACE: Differential Privacy for Dense Associative Memory with LSR Energy

    [https://arxiv.org/abs/2610.11218](https://arxiv.org/abs/2610.11218)

    本文提出了BRACE算法，一种针对LSR能量密集联想记忆的差分隐私检索机制，通过自适应校正边界敏感扰动的累积效应，实现了极小极大最优且与维度无关的检索误差率。

    

    密集联想记忆（DAM）提供了一种基于能量的记忆检索框架，与现代人工智能中的注意力机制有着密切的联系。尽管差分隐私在人工智能领域日益受到关注，但DAM检索动力学的隐私性问题仍相对缺乏探索。在本文中，我们为对数-和-ReLU（LSR）密集联想记忆开发了一个差分隐私框架，其有限支撑的检索动力学为隐私保护计算带来了独特的挑战。我们提出了边界响应自适应校正演化（BRACE）算法，这是一种用于LSR-DAM的差分隐私检索机制，能够自适应地校正边界敏感的扰动，以控制其在检索轨迹上的累积效应。在理论上，我们通过推导与维度无关的终端检索误差率和全轨迹检索误差率，证明了我们的方法具有极小极大最优性，并对逆温……具有最优的依赖关系。

    arXiv:2610.11218v1 Announce Type: cross  Abstract: Dense associative memory (DAM) provides an energy-based framework for memory retrieval with close connections to attention mechanisms in modern artificial intelligence. Despite growing interest in differential privacy for AI, the privacy of DAM retrieval dynamics remains relatively unexplored. In this paper, we develop a differential privacy framework for log-sum-ReLU (LSR) dense associative memory, whose finite-support retrieval dynamics pose distinctive challenges for privacy-preserving computation. We propose the Boundary-Responsive Adaptive Correction Evolution (BRACE) algorithm, a differentially private retrieval mechanism for LSR-DAM that adaptively corrects boundary-sensitive perturbations to control their cumulative effect over the retrieval trajectory. In theory, we prove that our method is minimax optimal by deriving dimension-independent terminal and full-trajectory retrieval error rates, with optimal dependence on the inver
    
[^194]: 转移定律之格

    The Lattice of Transition Laws

    [https://arxiv.org/abs/2610.11216](https://arxiv.org/abs/2610.11216)

    本文将扩散模型与自回归模型统一为同一个“腐蚀格”上的不同路径，通过定义解码调度的成本（即并行步骤所舍弃的依赖性），证明零成本调度的最少步数由数据的几何结构决定（例如等于图的树深度），从而可在解码前预测调度性能。

    

    扩散模型与自回归模型（AR）长期以来被视为生成模型中的不同类别：扩散模型专注于连续场，而自回归模型专注于离散token。近期的工作试图结合两种模型的优势，但每一种混合模型都在设计上预先固定了解码调度。在本文中，我们探讨这样一个问题：能否在解码之前、在固定步数下预测某个模型解码调度的性能。我们将扩散、自回归以及介于两者之间的模型描述为同一个腐蚀格上的路径，并将一个调度的成本定义为其并行步骤所舍弃的依赖关系。该成本表明，零成本调度所需的最少步数由数据的几何结构决定，且对token和连续场适用同样的规律。特别地，对于在图上满足马尔可夫性且沿其路径存在依赖关系的数据，最少步数等于该图的树深度，而树深度随序列长度呈对数增长，并且……（原文摘要在此处被截断）

    arXiv:2610.11216v1 Announce Type: cross  Abstract: Diffusion and autoregression (AR) have long been seen as different categories of generative models, with diffusion specialising in continuous fields and AR specialising in discrete tokens. Recent work seeks to combine the advantages of the two models, and each hybrid fixes its decoding schedule by design. In this paper, we ask whether the performance of decoding schedules of one model can be predicted before decoding at a fixed number of steps. We describe diffusion, AR, and models in between as paths on one corruption lattice, and define the cost of a schedule as the dependence its parallel steps discard. The cost shows that the fewest steps of a zero-cost schedule are set by the geometry of the data, in the same way for tokens and for continuous fields. In particular, for data that are Markov on a graph and dependent along its paths, the fewest steps equal the graph's treedepth, which is logarithmic in the length of a sequence and li
    
[^195]: 弥合KV缓存量化与线性注意力：从理论到预训练权重迁移

    Bridging KV-Cache Quantization and Linear Attention: From Theory to Pretrained Weight Migration

    [https://arxiv.org/abs/2610.11214](https://arxiv.org/abs/2610.11214)

    提出RAM-Net作为统一KV缓存量化与线性注意力的桥梁，通过离散地址空间上的软分配机制，在理论上证明其可分离读写重叠能局部逼近全注意力相似度，并支持从预训练权重迁移。

    

    arXiv:2610.11214v1 公告类型：交叉发布。摘要：KV缓存量化与线性注意力是应对Transformer存储与计算成本的两种代表性方法。KV缓存量化将单个KV条目压缩为离散编码，但仍需保留所有条目；而线性注意力则以循环方式将多个历史KV贡献聚合到一个固定大小的连续状态中，却可能引入干扰。这种对比引出了一个问题：能否在单一机制内将逐KV压缩与多KV聚合结合起来，以实现高效注意力。我们发现RAM-Net正是这样一座桥梁，它通过在离散地址空间上进行软分配来实现。这些软分配决定了对与每个地址相关联的连续槽状态的循环更新。在一个受限的RAM-Net构造下，我们证明软地址分配将硬量化匹配扩展为一种可分离的读写重叠机制，该机制能够局部逼近全注意力的相似度……

    arXiv:2610.11214v1 Announce Type: cross  Abstract: KV-cache quantization and linear attention are two representative approaches to tackling the storage and computational costs of Transformers. KV-cache quantization compresses individual KV entries into discrete codes but retains all entries, whereas linear attention recurrently aggregates multiple historical KV contributions into a fixed-size continuous state but can introduce interference. This contrast raises the question of whether per-KV compression and multi-KV aggregation can be bridged within a single mechanism for efficient attention. We identify RAM-Net as such a bridge through soft assignments over a discrete address space. These assignments determine recurrent updates to the continuous slot state associated with each address. Under a restricted RAM-Net construction, we prove that soft address assignments extend hard quantized matching to a separable read-write overlap that locally approximates full-attention similarity and s
    
[^196]: 更平坦的极小值能带来更好的泛化吗？Grokking现象中的算法性分离

    Do Flatter Minima Drive Better Generalization? An Algorithmic Separation in Grokking

    [https://arxiv.org/abs/2610.11206](https://arxiv.org/abs/2610.11206)

    本研究以grokking现象为试验平台，揭示了平坦极小值并非泛化的因果驱动机制——单独使用SAM虽能产生更平坦的解却无法可靠诱导泛化转变，只有当SAM与权重衰减等泛化机制结合时才展现出促进泛化的作用。

    

    平坦的损失景观长期以来被认为与神经网络更好的泛化能力相关，但其作为泛化的因果机制这一角色尚未得到充分确立。Grokking（顿悟）现象为理解这一区别提供了独特的试验平台：模型倾向于利用非泛化结构拟合观测数据，并在该状态下长期停留，只有在特定训练条件下才会转变为泛化。在本工作中，我们研究平坦的损失景观能否作为这一转变的驱动机制。尽管近期有研究认为平坦性是这一转变的必要几何条件，我们发现使用锐度感知最小化（SAM）使训练偏向更平坦的解，虽然确实产生了更平坦的解，却不足以可靠地诱导这一转变。然而，当SAM与权重衰减等驱动泛化的机制相结合时，出现了一个有趣的性质：SAM能够……（原文摘要在此处截断）

    arXiv:2610.11206v1 Announce Type: new  Abstract: Flat loss landscapes have long been linked to better generalization in neural networks. However, its role as a causal mechanism for generalization is less established. Grokking provides an unique testbed to understand this distinction: models are prone to fit observed data using non-generalizing structure and remain in that regime for prolonged periods, transitioning to generalization only under particular training conditions. In this work, we study whether flat loss landscapes can act as a driving mechanism in this transition. While recent work has argued for flatness as a necessary geometric condition for this transition, we find that biasing training toward flatter solutions using sharpness-aware minimization (SAM) is insufficient to reliably induce this transition, despite producing flatter solutions. However, when SAM is paired with mechanisms that drive generalization such as weight decay, an interesting property emerges: SAM can a
    
[^197]: 剖析视觉Transformer中的表征结构：一项严谨的架构研究

    Dissecting Representation Structure in Vision Transformers: A Rigorous Architectural Study

    [https://arxiv.org/abs/2610.11205](https://arxiv.org/abs/2610.11205)

    该论文首次严谨分析了视觉Transformer跨架构尺度的特征信息，发现初始化时的特征坍缩问题并提出缓解方案，同时证明熵和最小特征值可作为泛化预测的可靠指标，用以指导高效的ViT设计。

    

    表征结构对于理解视觉Transformer（ViT）架构及其泛化行为至关重要。然而，先前的研究既没有隔离和分析模块级别的特征，也没有研究这些特征之间的交互如何促进性能估计。在这项工作中，我们首次对跨多种架构尺度的特征信息进行了严谨分析，实证揭示了ViT表征与泛化行为之间的关系，并利用这些见解来指导高效的ViT设计。我们的贡献有五个方面：在多样的架构尺度上，1）我们识别了初始化时的特征坍缩现象，该现象导致特征冗余，并提出了一种缩减方案来缓解这一问题。2）我们使用熵和最小特征值来量化特征信息，证明这些指标可以作为泛化预测的可靠指标。3）我们表明特征在token...（原文摘要在此处截断）

    arXiv:2610.11205v1 Announce Type: cross  Abstract: Representation structure is crucial for understanding Vision Transformer (ViT) architectures and their generalization behavior. However, prior studies neither isolate nor analyze module-level features nor investigate how their interactions contribute to performance estimation. In this work, we conduct the first rigorous analysis of feature information across diverse architectural scales, empirically uncover the relationship between ViT representation and generalization behavior, and leverage these insights to guide efficient ViT design. Our contributions are fivefold: Across diverse architectural scales, 1) We identify feature collapse at initialization, which leads to redundancy, and propose a reduction scheme to mitigate this issue. 2) We quantify feature information using entropy and the minimum eigenvalue, demonstrating that these metrics serve as reliable indicators for generalization prediction. 3) We show that feature in the tok
    
[^198]: PageWeaver：面向稀疏注意力的KV引导查询联合

    PageWeaver: KV-Guided Query Unions for Sparse Attention

    [https://arxiv.org/abs/2610.11201](https://arxiv.org/abs/2610.11201)

    PageWeaver利用所选KV页的亲和性将查询分组为联合以共享页加载并填充Tensor Core瓦片，在保留每个查询原始支持集和完整输出所有权的前提下，在H200上实现了相比FlashInfer 1.70倍的几何平均加速。

    

    动态稀疏注意力限制了每个查询所选择的KV页，但较小的支持集并不一定带来高效的GPU运算。查询联合可以共享页加载并填充Tensor Core瓦片；其成本取决于将哪些查询分组在一起。我们提出了PageWeaver，这是一种执行设计，它利用所选页面的亲和性来组装查询组，同时保留每个查询的原始支持集和完整的输出所有权。有界GPU搜索生成查询ID，随后由一个感知ID的双CTA内核消费这些ID，而无需物化重排的Q张量或跨页的部分输出。直接的KV页联合实现则提供了对非局部重用与归约成本的补充性设计研究。在全程使用FP8 KV的情况下，H200上的Union8实现在六次捕获测试中，相比测量的FlashInfer路径实现了1.70倍的几何平均完整调用加速。在线重新分组在五个选定的64K上下文场景中进一步降低了3.26%至7.66%的延迟。

    arXiv:2610.11201v1 Announce Type: cross  Abstract: Dynamic sparse attention limits the KV pages selected by each query, but a small support does not necessarily yield efficient GPU work. Query unions share page loads and populate Tensor Core tiles; their cost depends on which queries are grouped together. We present PageWeaver, an execution design that uses selected-page affinity to assemble query groups while preserving each query's original support and complete output ownership. A bounded GPU search produces query IDs, and an ID-aware two-CTA kernel consumes them without materializing reordered Q tensors or cross-page partial outputs. A direct KV-page union implementation provides a complementary design study of nonlocal reuse and reduction cost. With FP8 KV throughout, the H200 Union8 implementation achieves a 1.70x geometric-mean complete-call speedup over the measured FlashInfer path on six captures. Online regrouping further lowers latency by 3.26-7.66% on five selected 64K-conte
    
[^199]: 选择性聆听：大型音频-语言模型中音频影响的机制引导控制

    Selective Listening: Mechanism-Guided Control of Audio Influence in Large Audio-Language Models

    [https://arxiv.org/abs/2610.11196](https://arxiv.org/abs/2610.11196)

    提出ICAP-Gate方法，通过机制引导、任务条件化的方式控制大型音频-语言模型的后期音频通路，在防止无关音频干扰文本推理的同时，不损害依赖音频的任务（如语音识别）的性能。

    

    大型音频-语言模型（LALMs）利用多模态证据，然而在无需聆听的情况下，与任务无关的音频可能会改变文本推理的决策。总体准确率可能会掩盖这种配对漂移，因为音频引起的修复与损害可能相互抵消。通过配对漂移分析和针对性干预，我们识别出特定于架构的、对干预敏感的后期音频通路，作为可操作的控制点。我们提出了ICAP-Gate，它对每个模型的通路应用机制引导的、任务条件化的控制。在四个LALM、两个推理基准以及环境声和自然语音干扰的实验中，ICAP-Gate在全部16个完整划分的模型-条件评估中，其影响率和答案翻转的点估计值均低于无门控推理。固定抑制会降低所有四个模型的自动语音识别（ASR）性能，而ICAP-Gate通过在明确的音频需求下保留音频通路，实现了与无门控推理相当的ASR性能。

    arXiv:2610.11196v1 Announce Type: cross  Abstract: Large audio-language models (LALMs) exploit multimodal evidence, yet task-irrelevant audio can alter text-reasoning decisions when listening is unnecessary. Aggregate Accuracy can hide this paired drift because audio-induced repairs and damages may cancel. Paired drift analysis and targeted interventions identify architecture-specific, intervention-sensitive late audio pathways as actionable control points. We introduce ICAP-Gate, which applies mechanism-guided, task-conditioned control to each model's pathway. Across four LALMs, two reasoning benchmarks, and environmental-sound and natural-speech interference, ICAP-Gate has lower point estimates for Influence Rate and Answer Flip than ungated inference in all 16 full-split model--condition evaluations. Fixed suppression degrades automatic speech recognition (ASR) across all four models, whereas ICAP-Gate matches ungated ASR performance by preserving the pathway for explicit audio-dema
    
[^200]: 细胞命运分配中的预测多重性：无标签Rashomon集合与单细胞认证的局限性

    Predictive Multiplicity in Cell-Fate Assignment: Label-Free Rashomon Sets and the Limits of Per-Cell Certification

    [https://arxiv.org/abs/2610.11185](https://arxiv.org/abs/2610.11185)

    提出无标签框架FateMultiplicity构建Rashomon集合以量化单细胞轨迹推断中细胞命运分配的预测多重性，发现模型空间多样性而非规模是多重性的主要驱动因素，且单细胞认证的命运边际无法提供比原始拟合模型更可靠的分配结果。

    

    单细胞轨迹推断将转录组测量映射到发育连续过程上，然而同样能拟合数据的多种模型配置可能为细胞分配出相互冲突的命运。FateMultiplicity是一个无需标签的框架，它通过在校准于随机种子变异的非劣效性检验下，在交叉拟合的保留基因上评估模型间差异，从而在无需谱系标签的情况下构建统计上可接受的模型集合（即Rashomon集合）。研究发现多重性规模较大，且更多取决于模型空间的多样性而非其大小：第二种算法的十二种配置暴露了20.0%的细胞存在命运不确定性，而第一种算法的二十四种配置仅暴露3.8%。随后测试了单细胞认证的命运边际FM是否能比已拟合模型本身提供更可靠的分配结果，结果表明并非如此。在仿真金标准数据上，在相同细胞上，FM区分错误分配的AUC为0.682，而基线配置自身的决策达到0.965。

    arXiv:2610.11185v1 Announce Type: new  Abstract: Single-cell trajectory inference maps transcriptomic measurements onto developmental continua, yet configurations that fit the data equally well can assign conflicting cell fates. FateMultiplicity is a label-free framework that constructs a statistically admissible model set, or Rashomon set, without lineage labels, by evaluating model discrepancy on cross-fitted held-out genes under non-inferiority testing calibrated against random-seed variation. Multiplicity is large and depends more on the diversity of the model space than its size: twelve configurations of a second algorithm expose 20.0% of cells where twenty-four of the first expose 3.8%. Whether the per-cell certified fate margin FM yields more reliable assignments than the fitted model already provides is then tested, and it does not. On simulation ground truth, on the same cells, FM discriminates misassignment at AUC 0.682, against 0.965 for the baseline configuration's own deci
    
[^201]: SACQ：面向长期时间序列预测的基于记忆条件化精炼的结构化解码方法

    SACQ: Structured Decoding with Memory-Conditioned Refinement for Long-Horizon Forecasting

    [https://arxiv.org/abs/2610.11170](https://arxiv.org/abs/2610.11170)

    SACQ是一种即插即用的结构化预测头，通过“粗粒度预测骨架+基于历史记忆交叉注意力的逐位置精炼”两阶段解码，并配合批自适应缩放的log-cosh损失，提升了长期时间序列预测的精度与对噪声的鲁棒性。

    

    长期时间序列预测（LTSF）模型主要采用基于补丁的编码器，并以一个扁平化读出头作为末端，通过单一共享投影将整个编码后的历史记忆映射到所有未来步骤。这种对未来位置的隐式耦合掩盖了特定位置上历史与未来之间的对齐关系，并放大了模型对损坏输入和极端监督噪声的敏感性。我们提出了SACQ，这是一种即插即用的结构化预测头，它可以在保持编码器不变的情况下替换扁平化读出头。SACQ采用两阶段解码流程：首先建立一个粗粒度的补丁网格预测骨架，然后通过对历史记忆的交叉注意力对每个未来位置进行精炼，并通过学习到的逐补丁门控将注意力导出的校正与粗粒度骨架进行融合。为了在长时程和噪声标签下稳定优化，我们进一步提出了一种批自适应缩放的log-cosh损失，它能够自动……（原文摘要至此截断）

    arXiv:2610.11170v1 Announce Type: new  Abstract: Long-term time series forecasting (LTSF) models predominantly employ patch-based encoders terminated by a flatten readout head that maps the entire encoded historical memory to all future steps through a single shared projection. This implicit coupling of future positions obscures position-specific historical-to-future alignment and amplifies sensitivity to corrupted inputs and extreme supervision noise. We present SACQ, a plug-in structured prediction head that replaces flatten readout while keeping the encoder unchanged. SACQ adopts a two-stage decoding pipeline: it first establishes a coarse patch-grid forecast scaffold, then refines each future position through cross-attention over historical memory and merges the attention-derived correction with the coarse scaffold via a learned per-patch gate. To stabilize optimization under long horizons and noisy labels, we further propose a batch-adaptive scaled log-cosh loss that automatically
    
[^202]: PIVOT：面向垂直领域小样本蒸馏的困惑度感知KD-to-RL转换调度

    PIVOT: Perplexity-Informed KD-to-RL Transition Scheduling for Vertical-Domain Few-Shot Distillation

    [https://arxiv.org/abs/2610.11167](https://arxiv.org/abs/2610.11167)

    PIVOT提出了一种根据教师评估的序列困惑度在在线蒸馏（OPD）与GRPO强化学习之间动态路由样本的转换调度框架，取代了传统全局固定调度，使低困惑度样本进入强化学习精炼、高困惑度样本继续接受教师引导的领域知识获取，从而提升小语言模型在垂直领域小样本分类中的表现。

    

    垂直领域小样本分类对小语言模型而言仍然具有挑战性，因为有限的监督使得模型难以获取领域特定的决策知识。在线蒸馏（On-Policy Distillation, OPD）可以通过监督学生模型自身生成的rollout来改进教师引导的适应过程，而基于GRPO的强化学习则能进一步优化下游预测。然而，现有的KD-to-RL流水线通常依赖于全局固定的转换调度，忽略了不同样本在进入奖励驱动的精炼阶段之前，可能需要不同强度的教师引导式知识获取。我们提出了PIVOT（Perplexity-Informed Transition Optimization，困惑度感知转换优化），这是一个动态转换框架，根据教师评估的序列困惑度在OPD和GRPO之间对样本进行路由分配。PIVOT将低困惑度样本转移到GRPO进行奖励驱动的精炼，同时将高困惑度样本保留在OPD之下，以持续进行领域知识获取。实验……

    arXiv:2610.11167v1 Announce Type: cross  Abstract: Vertical-domain few-shot classification remains challenging for small language models, as limited supervision makes it difficult to acquire domain-specific decision knowledge. On-Policy Distillation (OPD) can improve teacher-guided adaptation by supervising student-generated rollouts, while GRPO-based reinforcement learning can further refine downstream predictions. However, existing KD-to-RL pipelines typically rely on globally fixed transition schedules, ignoring that different samples may require different amounts of teacher-guided acquisition before reward-driven refinement. We propose PIVOT (Perplexity-Informed Transition Optimization), a dynamic transition framework that routes samples between OPD and GRPO according to teacher-evaluated sequence perplexity. PIVOT moves low-perplexity samples to GRPO for reward-driven refinement while keeping high-perplexity samples under OPD for continued domain knowledge acquisition. Experiments
    
[^203]: CARE：一种用于长期时间序列预测的轻量级即插即用门控校正与不确定性感知模块

    CARE: A Lightweight Plug-in Gated Correction and Uncertainty-aware Module for Long-term Time Series Forecasting

    [https://arxiv.org/abs/2610.11165](https://arxiv.org/abs/2610.11165)

    CARE是一种轻量级即插即用模块，通过并行校正分支对齐历史上下文学习残差校正，并利用不确定性感知的逐坐标风险门控进行有界更新，在不改变原有架构的情况下提升任意确定性长期时间序列预测模型的精度与可靠性。

    

    多变量长时程预测对电力负荷调度、交通流量管理以及金融风险控制至关重要。现有的确定性骨干模型仅输出单一预测轨迹，掩盖了不同时程和通道之间预测难度的异质性，也无法提供局部化的可靠性信号。我们提出了CARE（具有对齐上下文与相对误差估计的校正分支），这是一种轻量级的即插即用模块，无需重新设计架构即可增强任何确定性预测模型。CARE与基础模型并行运行，通过对历史上下文进行重采样以匹配预测时程，从这个对齐的历史数据中学习残差校正模式，并应用由逐坐标sigmoid风险门调制的大小感知有界更新。多目标损失函数联合优化预测精度、残差跟踪、风险对齐和基础模型锚定。在采用三个代表性骨干模型的八个基准测试中……

    arXiv:2610.11165v1 Announce Type: new  Abstract: Multivariate long-horizon forecasting is critical to electricity load scheduling and traffic flow management, and to financial risk control. Existing deterministic backbones output a single trajectory, masking heterogeneous prediction difficulty across horizons and channels and providing no localized reliability signal. We present CARE (Corrective branch with Aligned context and Relative-error Estimation), a lightweight plug-in that enhances any deterministic forecaster without architectural redesign. Operating in parallel with the base model, CARE resamples historical context to match the forecast horizon, learns residual correction patterns from this aligned history, and applies scale-aware bounded updates modulated by per-coordinate sigmoid risk gates. A multi-objective loss jointly optimizes forecast accuracy, residual tracking, risk alignment, and base-model anchoring. Across eight benchmarks with three representative backbones, CAR
    
[^204]: RideBench：面向网约车时间序列预测的大规模外生感知基准

    RideBench: A Large-Scale Exogenous-Aware Benchmark for Ride-Hailing Time Series Forecasting

    [https://arxiv.org/abs/2610.11164](https://arxiv.org/abs/2610.11164)

    该论文发布了基于滴滴数据构建的覆盖200个区域、跨越四年、半小时粒度的大规模网约车时序数据集 Ride-Hailing 及评测基准 RideBench，通过对30多种方法的评测证明未来已知的外生变量（天气、节假日、大型事件）能显著提升网约车预测精度。

    

    我们发布了 Ride-Hailing，一个基于滴滴出行市场数据合成的大规模网约车时间序列数据集，覆盖200个空间区域。该数据集以半小时为粒度跨越连续四年，涵盖三种代表性的外生场景：天气扰动、节假日效应和大型事件影响。基于 Ride-Hailing 数据集，我们提出了 RideBench，一个面向外生感知网约车预测的综合基准，涵盖常规的未来一周预测以及长达8周、最多2688个预测步的长时程预测。RideBench 评估了30多种代表性预测方法，包括仅使用内生变量的模型、外生感知模型和时间序列基础模型。我们的结果表明，未来已知的外生变量在常规的未来一周预测中带来了明显收益，尤其是在天气、节假日和大型事件（如重大体育赛事和演唱会）场景下。

    arXiv:2610.11164v1 Announce Type: new  Abstract: We release Ride-Hailing, a large-scale ride-hailing time series dataset synthesized from DiDi's marketplace data across 200 spatial areas. Ride-Hailing spans four consecutive years at half-hourly granularity and covers three representative exogenous scenarios: Weather Disturbance, Holiday Effect, and Large-scale Event Impact. Built upon Ride-Hailing, we introduce RideBench, a comprehensive benchmark for exogenous-aware ride-hailing forecasting, covering both regular week-ahead forecasting and long-horizon 8-week-ahead forecasting with up to 2,688 prediction steps. RideBench evaluates over 30 representative forecasting methods, including endogenous-only models, exogenous-aware models, and time series foundation models. Our results show that future-known exogenous variables provide clear benefits in regular week-ahead forecasting, especially under weather, holiday, and large-scale event (e.g., major sporting events and concerts) scenarios.
    
[^205]: LadderEdit：面向大语言模型内存高效终身编辑的编辑级残差压缩

    LadderEdit: Edit-Level Residual Compression for Memory-Efficient Lifelong Editing of LLMs

    [https://arxiv.org/abs/2610.11160](https://arxiv.org/abs/2610.11160)

    LadderEdit通过将每条编辑先以低秩草图存储、仅对未满足契约的困难编辑沿阶梯逐级提升秩的方式压缩LoRA适配器，在保持编辑覆盖效果的同时将内存占用降低5.2倍，并支持5万次连续终身编辑。

    

    大语言模型的终身编辑需要在获取后存储成千上万条编辑。一类被广泛使用的方法是为每条编辑附加一个LoRA适配器，这虽然能保持模型行为，但存储量会随编辑数量线性增长。为应对这一挑战，我们提出了LadderEdit，一种在每条编辑获取后对其LoRA适配器进行压缩的方法。每条编辑首先以低秩形式存储为一个廉价的“草图”。随后，我们在探针提示上检查该草图是否仍满足重写、泛化和局部性契约。通过检查的编辑保留草图；未通过的编辑则沿阶梯逐级提升至更高的秩，直到满足契约为止。由于每条编辑都保留了某种表示，编辑覆盖率得以维持，只有困难的编辑才会消耗更多的秩。在LLaMA-3-8B、Mistral-7B和Qwen2.5-7B模型上的ZsRE、CounterFact和WikiBigEdit基准测试中，LadderEdit在内存占用减少5.2倍的情况下保持了与精确LoRA存储相当的编辑效果，并在50,000次连续编辑后依然有效。

    arXiv:2610.11160v1 Announce Type: new  Abstract: Lifelong editing of LLMs requires storing thousands of edits after acquisition. A widely used family of approaches attaches one LoRA adapter per edit, which preserves behavior but grows linearly in storage. To address this challenge, we propose LadderEdit, a method that compresses each LoRA adapter after it is acquired. Each edit is first stored at low rank as a cheap sketch. We then check whether this sketch still satisfies the rewrite, generalization, and locality contract on probe prompts. Edits that pass keep the sketch; those that fail are promoted to a higher rank along a ladder until the contract is met. Because every edit retains some representation, coverage is maintained, and only hard edits consume more rank. Across ZsRE, CounterFact, and WikiBigEdit benchmarks on LLaMA-3-8B, Mistral-7B, and Qwen2.5-7B, LadderEdit tracks exact LoRA storage at 5.2x less memory and remains effective at 50,000 sequential edits.
    
[^206]: 大语言模型会从上下文中的奖励中学习吗？：重新思考奖励在上下文强化学习中的作用

    Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning

    [https://arxiv.org/abs/2610.11152](https://arxiv.org/abs/2610.11152)

    该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。

    

    大语言模型智能体越来越多地通过在上下文中积累经验而非更新参数来在推理阶段提升性能，这一过程通常被称为上下文强化学习（ICRL）。然而，上下文学习（ICL）是否真的能够发挥强化学习的作用，尚未经过检验。我们在其最简单的形式——直接ICRL中研究这一问题，即模型直接以原始的轨迹-奖励对为条件，并探究奖励是否真正充当学习信号。通过对六个模型在四个基准上的受控实验，我们发现奖励虽然会被模型读取，但其影响很小：翻转、随机化或移除奖励几乎不会改变性能改进曲线，即使在通过元提示明确指示模型进行探索、利用或对奖励进行推理的情况下也是如此。轨迹驱动了性能改进，但并非通过其语义内容：打乱或损坏的轨迹与真实轨迹的效果相当。

    arXiv:2610.11152v1 Announce Type: cross  Abstract: LLM agents increasingly improve at inference time by accumulating experience in context rather than by updating parameters. This process is often described as in-context reinforcement learning (ICRL). Whether in-context learning (ICL) can actually play the role of RL, however, has not been tested. We study this question in its simplest form, direct ICRL, where the model conditions directly on raw trajectory-reward pairs, and ask whether the reward acts as a learning signal. Through controlled experiments on four benchmarks across six models, we find that the reward is read, but its effect is small: flipping, randomizing, or removing the reward leaves the improvement curve almost unchanged, and this holds even under meta-prompts that explicitly instruct the model to explore, exploit, or reason over rewards. Trajectories drive improvement, but not through their semantic content: shuffled or corrupted trajectories work as well as real one
    
[^207]: 面向信贷风险建模的排序先验对齐：外部先验何时起作用？

    Ranking Prior Alignment for Credit Risk Modeling: When Do External Priors Matter?

    [https://arxiv.org/abs/2610.11146](https://arxiv.org/abs/2610.11146)

    提出了一种模型无关的“排序先验对齐”框架，通过温度缩放的KL散度损失将来自领域专家、教师模型或大语言模型的外部排序先验统一蒸馏进神经或树类评分模型，以缓解标注数据稀缺的冷启动信贷评分难题。

    

    冷启动信贷评分——即在标注数据稀缺、特征薄弱或模型容量极小的情况下部署模型——是金融机器学习中反复出现的难题。当一个新贷款产品上线时，标注的违约数据稀缺、特征流水线不成熟，模型必须以最小容量部署以避免过拟合。标准防御手段都在同样有限的同一份数据上运行；我们真正需要的是一种扎根于领域知识的外部正则化来源。我们提出了排序先验对齐，这是一个模型无关的框架，通过温度缩放的KL散度损失，将外部排序先验（来自领域专家、教师模型或大语言模型）蒸馏到任何评分模型中。该框架通过一个统一公式将神经网络（MIL注意力）架构和树模型（XGBoost自定义目标函数）架构统一起来：L = L_task + gamma(t) * KL(P_agent || P_model)，其中 gamma(t) 遵循指数衰减调度。该方法不需要外部……（原文摘要在此处截断）

    arXiv:2610.11146v1 Announce Type: new  Abstract: Cold-start credit scoring -- deploying models with scarce labeled data, weak features, or minimal capacity -- is a recurring problem in financial machine learning. When a new lending product launches, labeled default data is scarce, feature pipelines are immature, and models must be deployed with minimal capacity to avoid overfitting. Standard defenses operate on the same limited data; what is needed is a source of external regularization grounded in domain knowledge.   We propose Ranking Prior Alignment, a model-agnostic framework that distills external ranking priors (from domain experts, teacher models, or LLMs) into any scoring model via a temperature-scaled KL divergence loss. The framework unifies neural (MIL attention) and tree-based (XGBoost custom objective) architectures through a single formulation: L = L_task + gamma(t) * KL(P_agent || P_model), where gamma(t) follows an exponential decay schedule. The method requires no exte
    
[^208]: ActiveMedAgent：面向多模态医学诊断的成本感知轨迹学习

    ActiveMedAgent: Cost-Aware Trajectory Learning for Multimodal Medical Diagnosis

    [https://arxiv.org/abs/2610.11140](https://arxiv.org/abs/2610.11140)

    该论文提出 ActiveMedAgent 框架，通过追踪诊断概率分布并按“诊断效用减去成本”对信息获取轨迹打分、离线训练轻量级 MLP 控制器，使冻结的视觉语言模型在多模态医学诊断中以更低成本、更少模态获得更准确的诊断结果。

    

    临床诊断本质上是顺序进行的：临床医生只有在预期额外证据能够消除诊断不确定性时，才会从廉价检查升级到昂贵检查。我们提出了 ActiveMedAgent，这是一个将这种成本感知的顺序决策逻辑引入多模态医学 AI 的框架。给定一个冻结的、通过 API 访问的视觉语言模型，ActiveMedAgent 追踪候选诊断的概率分布，并根据每一步的诊断效用减去成本来为每次信息获取打分。随后，一个轻量级的 MLP 控制器在这些评分轨迹上进行离线训练，从而学习何时请求额外证据、何时做出最终诊断。在三个常用基准测试中，基于轨迹的策略学习始终优于无引导的信息获取和全模态基线。值得注意的是，我们识别出了一种信息过载效应：在 175 个案例中，该智能体仅用更少的模态通道就能得出正确诊断，而全模态基线却诊断失败。

    arXiv:2610.11140v1 Announce Type: cross  Abstract: Clinical diagnosis is inherently sequential: clinicians escalate from cheap to costly tests only when additional evidence is expected to resolve diagnostic uncertainty. We present ActiveMedAgent, a framework that brings this cost-aware sequential logic to multimodal medical AI. Given a frozen, API-accessed vision-language model, ActiveMedAgent tracks probability distributions over candidate diagnoses and scores each acquisition by its per-step diagnostic utility minus cost. A lightweight MLP controller is then trained offline on these scored trajectories, learning when to request additional evidence and when to commit. Across three commonly used benchmarks, trajectory-based policy learning consistently outperforms both unguided acquisition and full-modality baselines. Notably, we identify an information overload effect. In 175 cases, the agent produces a correct diagnosis with fewer channels while the full-modality baseline fails, show
    
[^209]: 加速非光滑与重尾采样

    Accelerating Non-Smooth and Heavy-Tailed Sampling

    [https://arxiv.org/abs/2610.11139](https://arxiv.org/abs/2610.11139)

    本文提出非可逆锚定朗之万动力学（NALD）与非可逆反射锚定朗之万动力学（NRALD），通过引入循环漂移项，在无需目标密度导数的情况下加速对欧氏空间及受限域上非光滑、重尾目标分布的采样。

    

    锚定朗之万动力学（ALD）可用于非光滑采样场景，其中目标分布的密度可能不可微且呈重尾特性；反射锚定朗之万动力学（RALD）可以在受限域上对可能不可微的目标密度进行采样。本文提出并研究了非可逆锚定朗之万动力学（NALD），用于在欧几里得空间中对可能不可微且重尾的目标密度进行采样；以及非可逆反射锚定朗之万动力学（NRALD），用于在受限空间中对可能不可微的目标密度进行采样。我们的构造添加了一个由可能是状态依赖的无散度斜对称矩阵场和流势所产生的循环漂移。该构造无需目标密度的导数即可保持目标分布，具有随机时间变换表示，并且既适用于整个欧几里得空间，也适用于（受限空间）。

    arXiv:2610.11139v1 Announce Type: cross  Abstract: Anchored Langevin dynamics (ALD) is useful for non-smooth sampling where the density of the target distribution is possibly non-differentiable and heavy-tailed; reflected anchored Langevin dynamics (RALD) can sample possibly non-differentiable target density on a constrained domain. In this paper, we propose and study non-reversible anchored Langevin dynamics (NALD) for sampling possibly non-differentiable and heavy-tailed target density in the Euclidean space and the non-reversible reflected anchored Langevin dynamics (NRALD) for sampling possibly non-differentiable target density in the constrained space. Our construction adds a circulation drift generated by a possibly state-dependent divergence-free skew-symmetric matrix field and a stream potential. It preserves the target distribution without requiring derivatives of target density, admits a random-time-change representation, and applies both on the whole Euclidean space and on b
    
[^210]: 当记录的学习者很少或没有时，一个“系统一”大语言模型能否执行知识追踪？

    Can a System-One LLM Perform Knowledge Tracing When Few or No Learners Are Logged?

    [https://arxiv.org/abs/2610.11135](https://arxiv.org/abs/2610.11135)

    该论文提出，现成的“系统一”LLM（Jev/JevKT）无需目标平台的学习者数据或仅需极少数据即可完成知识追踪，其性能超过28个深度知识追踪模型和“系统二”LLM方法，而API成本仅约为后者的百分之一。

    

    知识追踪（KT）模型需要大量已记录的学习者数据，因此新课程或新平台在启动时没有可用的模型。在基于LLM的知识追踪中，LLM会生成答案，我们称之为“系统二”；它要么在目标数据上进行微调，要么对十个样本进行推理和投票，这不仅速度慢，而且给出的概率较为粗糙。我们探究一个现成的“系统一”LLM——它在单次处理中直接为输入的问题返回概率——在已记录的学习者很少或没有时能否执行知识追踪。在七个数据集上，Jev在不使用目标平台任何数据的情况下达到平均AUC为0.706，高于在8名学习者数据上训练的28个深度知识追踪模型中的最佳者（0.689），并在所有七个数据集上都优于“系统二”方法Thinking-KT（0.650），而其API成本仅约为后者的1/100。加入已记录学习者的示例和相似学习者统计量后（JevKT），该数值提升至0.722；JevKT在学习者数量达到16名时仍显著领先于深度知识追踪，在平均水平上直至64名学习者时依然领先，而监督（原文在此处截断）

    arXiv:2610.11135v1 Announce Type: new  Abstract: Knowledge tracing (KT) models need many logged learners, so a new course or platform starts without a usable model. In LLM-based KT the LLM generates the answer, which we call System-Two; it is either fine-tuned on the target data or reasons and votes over ten samples, which is slow and gives coarse probabilities. We ask whether an off-the-shelf System-One LLM, which returns a probability for a typed question directly in a single pass, can perform KT when few or no learners are logged. On seven datasets, Jev without any data from the target platform reaches a mean AUC of .706, above the best of 28 deep KT models trained on 8 learners (.689) and above System-Two Thinking-KT on all seven datasets (.650) at about 1/100 of its API cost. Adding examples and a similar-learner statistic from the logged learners (JevKT) raises this to .722; JevKT stays significantly ahead of deep KT up to 16 learners and ahead on average up to 64, and supervised
    
[^211]: QUILT：通过共享查询执行重新思考稀疏注意力预填充

    QUILT: Rethinking Sparse-Attention Prefill through Shared Query Execution

    [https://arxiv.org/abs/2610.11134](https://arxiv.org/abs/2610.11134)

    QUILT通过联合处理相邻查询、复用共享KV条目，并借助移位比较集合分解（SCSD）将不规则集合操作转化为规则数据并行原语，显著减少了长上下文稀疏注意力预填充中的冗余内存流量和计算开销。

    

    稀疏注意力降低了长上下文注意力的计算成本，但现有的核函数通常独立处理各个查询，重复加载并反量化跨查询共享的KV条目。我们观察到相邻查询所选择的KV条目存在大量重叠，这为跨查询复用创造了机会。我们提出了QUILT，一种工作负载感知的稀疏注意力执行机制，它联合处理相邻查询并复用共享的KV条目，以减少冗余的内存流量和计算。QUILT引入了移位比较集合分解（SCSD），将不规则的集合操作转换为适合现代加速器的规则数据并行原语，并将SCSD与注意力计算流水线化以隐藏其开销。级联共享机制在多个粒度上分层捕获复用。瓦片感知的执行策略在共享粒度与硬件瓦片利用率之间取得平衡，并有选择地移除……

    arXiv:2610.11134v1 Announce Type: new  Abstract: Sparse attention reduces the cost of long-context attention, but existing kernels typically process queries independently, repeatedly loading and dequantizing KV entries shared across queries. We observe substantial overlap in the KV entries selected by neighboring queries, creating opportunities for cross-query reuse. We present QUILT, a workload-aware sparse-attention execution mechanism that jointly processes neighboring queries and reuses shared KV entries to reduce redundant memory traffic and computation. QUILT introduces Shift-and-Compare Set Decomposition (SCSD), which transforms irregular set operations into regular data-parallel primitives suitable for modern accelerators, and pipelines SCSD with attention computation to hide its overhead. Cascaded sharing captures reuse hierarchically at multiple granularities. A tile-aware execution strategy balances sharing granularity with hardware tile utilization and selectively removes l
    
[^212]: 用于增强操作系统指纹识别的机器学习优化

    Machine Learning Optimization for Enhanced OS Fingerprinting

    [https://arxiv.org/abs/2610.11133](https://arxiv.org/abs/2610.11133)

    本研究提出新命令行工具OsirisML，结合nPrint数据预处理与XGBoost机器学习算法，在CIC-IDS2017数据集上实现了高效的被动式操作系统指纹识别，最高准确率达97.66%。

    

    操作系统指纹识别是一种通过分析TCP/IP数据包形式的网络流量来识别网络中操作系统类型的技术。本研究在CIC-IDS2017数据集上探索了被动识别操作系统的有效性，该数据集包含超过47GB的pcap文件及其对应的操作系统信息。本研究还提出了一个新的命令行工具OsirisML，它使用nPrint将数据预处理为表格数据，并利用XGBoost对数据应用机器学习方法，以生成、重新训练和测试机器学习模型。当数据包被随机划分为训练集和测试集时，OsirisML模型在周五抓包数据的下采样子集上达到了97.66%的准确率，在整个抓包数据上达到了84.69%的准确率。在包含无攻击流量的整个周一抓包数据上，OsirisML达到了73.83%的准确率和79.38%的F1分数。

    arXiv:2610.11133v1 Announce Type: new  Abstract: Operating System (OS) Fingerprinting is a technique that can be used to identify a network's operating systems by evaluating network traffic in the form of TCP/IP packets. This research will explore the effectiveness of passively identifying operating systems on the CIC-IDS2017 dataset, a collection of over 47 gigabytes of pcap files with their corresponding operating systems. This research also proposes a new command line interface, OsirisML, which uses nPrint to preprocess the data into tabular data and XGBoost to apply ML to the data to generate, retrain, and test ML models. When packets are split randomly between training and testing, OsirisML models reach an accuracy of 97.66% on a down-sampled subset of the Friday capture and 84.69% on the entire capture. On the entire Monday capture, which contains no attacks, OsirisML reaches an accuracy of 73.83% and an F-1 score of 79.38%.
    
[^213]: SFT作为上下文缓解监督微调中的遗忘

    SFT-as-Context Mitigates Forgetting in Supervised Fine-Tuning

    [https://arxiv.org/abs/2610.11132](https://arxiv.org/abs/2610.11132)

    提出无需训练的SFT-as-context方法，让父模型将SFT模型的响应作为上下文通过上下文学习获得微调能力，从而在保留通用能力的同时缓解监督微调带来的遗忘问题。

    

    监督微调（SFT）为大型语言模型（LLM）赋予专业能力，但往往以遗忘其父模型（即微调前的预训练模型）的通用能力为代价。这种权衡对于既需要专业能力又需要通用能力的查询而言尤为受限。我们提出SFT-as-context，这是一种无需训练的方法，其中父模型将SFT模型的响应作为上下文来回答查询。这使得父模型能够通过上下文学习从SFT响应中获取微调后的能力，同时保留自身的通用能力。在19个父模型-SFT模型对和11个基准测试上，SFT-as-context在微调能力上与SFT模型保持接近，在AIME 2024和LiveCodeBench上的差距仅为2.2和2.1个百分点，在NutriBench-English上的宏观MAE仅为2.0，同时与父模型的差距保持在2.2个百分点以内。

    arXiv:2610.11132v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) equips large language models (LLMs) with specialized capabilities, but often comes at the cost of forgetting the general capabilities of their parent models (i.e., the pretrained models before fine-tuning). This trade-off is especially limiting for queries that require both specialized and general capabilities. We introduce SFT-as-context, a training-free method in which the parent model uses the SFT model's response as context to answer the query. This allows the parent model to acquire fine-tuned capabilities from the SFT response through in-context learning while preserving its own general capabilities. Across 19 parent-SFT model pairs and 11 benchmarks, SFT-as-context remains close to the SFT models on fine-tuned capabilities, with gaps of only 2.2 and 2.1 percentage points on AIME 2024 and LiveCodeBench and 2.0 macro MAE on NutriBench-English, while staying within 2.2 percentage points of the parent mo
    
[^214]: 动力学即代码：基于动力系统的模型压缩

    Dynamics as Code: On Model Compression via Dynamic System

    [https://arxiv.org/abs/2610.11115](https://arxiv.org/abs/2610.11115)

    该论文提出“动力学即代码”的新型模型压缩范式，证明了在丢番图条件下，无理缠绕的有限轨迹可构成权重空间上的ε-网，从而以可预测的方式将状态分辨率、解压误差与压缩比联系起来。

    

    预训练神经网络规模的不断攀升，使得模型压缩成为在严苛内存与计算约束下部署的先决条件。以无理缠绕为例，先前工作引入了一种动力系统（DS）范式，将压缩重新定义为紧凑的权重表示：高维参数由动力系统产生的轨迹的索引进行编码，解压时再从该轨迹中恢复原始向量。这一机制与剪枝、量化、知识蒸馏以及低秩分解有着本质区别。沿着这一方向，我们证明了在丢番图条件下，无理缠绕中由 \(M = O(\epsilon^{-(d+\nu)})\) 个状态构成的有限轨迹可以在 \(d\) 维权重空间上构成一个 \(\epsilon\)-网，从而以可预测的方式将状态分辨率、解压误差和压缩比联系了起来。此外，我们进一步……

    arXiv:2610.11115v1 Announce Type: new  Abstract: The escalating size of pretrained neural networks has rendered model compression a prerequisite for deployment under stringent memory and compute constraints. With the irrational winding as an example, earlier work introduced a dynamic system (DS) paradigm that reconceptualizes compression as compact weight representation: high-dimensional parameters are encoded by the index of a trajectory produced by a dynamic system, from which the vector is recovered during decompression. This mechanism is fundamentally distinct from pruning, quantization, knowledge distillation, and low-rank decomposition. Along this direction, we prove that under a Diophantine condition, a finite trajectory of \(M = O(\epsilon^{-(d+\nu)})\) states in the irrational winding constitutes an \(\epsilon\)-net over the \(d\)-dimensional weight space, thereby linking state resolution, decompression error, and compression ratio in a predictable manner. Furthermore, we prop
    
[^215]: Lapras：面向时间序列语言模型的潜在推理

    Lapras: Latent Reasoning for Time Series Language Models

    [https://arxiv.org/abs/2610.11111](https://arxiv.org/abs/2610.11111)

    Lapras是一个后训练框架，通过让时间序列语言模型在潜在空间而非离散语言标记中进行推理，避免了将连续时序信号转化为文字描述时的信息丢失与早期错误传播，从而生成与输入信号一致、更忠实的答案。

    

    时间序列语言模型（TSLMs）通过对接时序信号进行推理并生成自然语言的答案与解释，为时间序列理解提供了一条有前景的路径。一种常见方法是思维链（Chain-of-Thought, CoT），它生成逐步的推理依据，将相关信号模式与最终答案联系起来。尽管这些模型在后训练阶段会从参考CoT轨迹中学习，但在推理时对输入时间序列生成忠实的描述仍然具有挑战性。用离散的语言标记来表达高维、连续的时间表示，可能导致模型忽略与任务相关的模式或对其描述不准确。由于后续的推理步骤建立在这些描述之上，早期的错误会不断传播，最终导致答案错误但解释看似合理，且与输入信号不一致。我们提出了Lapras（Latent Post-trained Reasoning Across Series），这是一个后训练框架，使T……（原文摘要在此处截断）

    arXiv:2610.11111v1 Announce Type: new  Abstract: Time Series Language Models (TSLMs) offer a promising path toward time series understanding by reasoning over temporal signals and producing natural language answers and explanations. A common approach is Chain-of-Thought (CoT), which generates step-by-step rationales linking relevant signal patterns to final answers. Although these models learn from reference CoT traces during post-training, generating faithful descriptions of input time series at inference remains challenging. Expressing high-dimensional, continuous temporal representations in discrete language tokens may cause the model to neglect task-relevant patterns or describe them inaccurately. Because later reasoning steps build on these descriptions, early errors propagate, leading to incorrect answers with plausible explanations that are inconsistent with the input signal. We propose Lapras (Latent Post-trained Reasoning Across Series), a post-training framework that equips T
    
[^216]: Cova-PINN：面向复杂几何结构中流固共轭传热的跨域守恒物理信息神经网络

    Cova-PINN: Cross-Domain Conservation Physics-Informed Neural Network for Fluid-Solid Conjugate Heat Transfer in Complex Geometries

    [https://arxiv.org/abs/2610.11108](https://arxiv.org/abs/2610.11108)

    Cova-PINN提出了一种多域物理信息神经网络框架，通过在局部尺度联合优化跨域复合控制体平衡、在全局换热器尺度优化成对壁面闭合，将守恒支撑与复杂几何中的热相互作用路径对齐，从而准确预测流固共轭传热中的端到端能量传递和出口温度。

    

    多域物理信息神经网络（PINN）可以灵活地建模介质特定的表示，以求解流固共轭传热（CHT）问题。然而，标准的多域PINN是在分别采样的域支撑上施加控制方程和界面条件的，这可能产生看似合理的温度场，但会导致不准确的端到端能量传递和出口温度。我们提出了Cova-PINN，这是一个多域PINN框架，它将守恒支撑与复杂几何结构中的热相互作用路径对齐。Cova-PINN在局部尺度上联合优化跨域复合控制体平衡，并在全局换热器尺度上优化成对壁面闭合。我们在四个三周期极小曲面（TPMS）换热器和一个几何结构不同的DualMS设计上，在统一协议下，将Cova-PINN与CHT专用、面向优化以及复杂几何的PINN基线方法进行了对比评估。相对于最接近的基线方法，

    arXiv:2610.11108v1 Announce Type: new  Abstract: Multi-domain physics-informed neural networks (PINNs) flexibly model medium-specific representations to solve fluid--solid conjugate heat transfer (CHT). However, standard multi-domain PINNs enforce governing equations and interface conditions on separately sampled domain supports, which can yield plausible temperature fields but inaccurate end-to-end energy transfer and outlet temperatures. We propose Cova-PINN, a multi-domain PINN framework that aligns conservation support with thermal interaction paths in complex geometries. Cova-PINN jointly optimizes cross-domain composite control-volume balances at the local scale and paired-wall closure at the global exchanger scale. We evaluate Cova-PINN on four triply periodic minimal surface (TPMS) heat exchangers and a geometrically distinct DualMS design against CHT-specific, optimization-oriented, and complex-geometry PINN baselines under a common protocol. Relative to the closest baseline, 
    
[^217]: DaCe-DT：基于自适应提示与轨迹校正的面向异构任务的数据中心化离线多任务强化学习

    DaCe-DT: Data-Centric Offline Multi-Task Reinforcement Learning via Adaptive Prompts and Trajectory Correction for Heterogeneous Tasks

    [https://arxiv.org/abs/2610.11085](https://arxiv.org/abs/2610.11085)

    本文提出数据中心化的离线多任务强化学习框架DaCe-DT，通过长度门控提示掩码（LGPM）、检索增强提示构建（RAPC）和价值自适应回报校准（VARC）三项技术，有效解决了提示长度利用低效、提示语义不相关以及轨迹碎片化导致误导监督这三大数据瓶颈，从而提升模型对异构任务复杂度和数据质量的鲁棒性与泛化能力。

    

    离线多任务强化学习严重依赖于预先收集数据的质量和分布。然而，现有方法主要聚焦于算法层面的优化，较少强调从数据层面的改进来增强学习能力与泛化性能。本文从数据视角揭示了限制离线多任务强化学习性能的三个关键瓶颈：（i）在多样化任务复杂度下提示长度的低效利用；（ii）随机采样的提示片段存在语义不相关性；（iii）碎片化且不连续的轨迹所导致的误导性监督。为应对这些挑战，我们提出了DaCe-DT，一个对异构任务复杂度和数据质量均不敏感的鲁棒离线多任务强化学习框架，其核心特性包括长度门控提示掩码、检索增强提示构建以及价值自适应回报校准。这些方法共同……（原文摘要在此处截断）

    arXiv:2610.11085v1 Announce Type: new  Abstract: Offline multi-task reinforcement learning (Offline MTRL) heavily depends on the quality and distribution of pre-collected data. However, existing methods mainly focus on algorithmic optimization, with less emphasis on data-level improvements to enhance learning ability and generalization performance. This paper, from a data perspective, reveals three key bottlenecks that limit Offline MTRL performance:(i) ineffective utilization of prompts length under diverse task complexities, and (ii) semantic irrelevance of randomly sampled prompt segments, (iii) misleading supervision induced by fragmented and discontinuous trajectories. To address these challenges, we propose DaCe-DT, a robust offline MTRL framework designed to be insensitive to heterogeneous task complexities and data quality, featuring length-gated prompt masking (LGPM), retrieval-augmented prompt construction (RAPC), and value-adaptive return calibration (VARC). Together, these 
    
[^218]: 核赌博机问题的通用 $\widetilde{\Omega}(\sqrt{T \gamma_T})$ 下界

    A General $\widetilde{\Omega}(\sqrt{T \gamma_T})$ Lower Bound for Kernel Bandits

    [https://arxiv.org/abs/2610.11082](https://arxiv.org/abs/2610.11082)

    本文针对紧域上非常数连续核函数的核赌博机问题，建立了通用的 $\Omega(\sqrt{T\gamma_T/\log T})$ 极小极大遗憾下界，并证明该下界中的对数因子在一般情况下不可避免，从而在非常一般的意义上确立了现有 $\sqrt{T\gamma_T}$ 上界的接近最优性。

    

    核赌博机（kernel bandit）问题是指在有噪声反馈下顺序优化一个未知函数的问题，其中该函数在给定的再生核希尔伯特空间（RKHS）中具有有界范数。核赌博机遗憾分析中的一个核心量是最大信息增益 $\gamma_T$。特别地，现有的最佳上界在对数因子范围内以 $\sqrt{T\gamma_T}$ 的形式缩放，并且已经针对特定核函数（如平方指数核和Matérn核）推导出了几乎匹配的下界。然而，针对一般核函数的下界仍然缺失，这使得现有上界在何种一般性下接近最优尚不清楚。在本文中，我们针对紧域上的非常数连续核函数建立了一个通用的 $\Omega(\sqrt{T\gamma_T/\log T})$ 极小极大遗憾下界，从而在非常一般的意义上确立了上界的接近最优性（在对数因子范围内）。我们证明该下界中出现的对数因子在一般情况下是不可避免的，但……

    arXiv:2610.11082v1 Announce Type: cross  Abstract: The kernel bandit problem consists of sequentially optimizing an unknown function with noisy feedback, where the function has bounded norm in a given Reproducing Kernel Hilbert Space (RKHS). A central quantity in the regret analysis of kernel bandits is the maximum information gain $\gamma_T$. In particular, the best existing upper bounds scale as $\sqrt{T\gamma_T}$ up to log factors, and nearly-matching lower bounds have been derived for specific kernels such as squared exponential and Mat\'ern. However, lower bounds for general kernels are lacking, thus making it unclear in what generality the upper bounds are near-optimal. In this paper, we establish a general $\Omega(\sqrt{T\gamma_T/\log T})$ minimax regret lower bound for non-constant continuous kernels on compact domains, establishing near-optimality (within log factors) in a very general sense. We show that the log factor appearing in this bound is unavoidable in general, but th
    
[^219]: 通过奇异向量选择实现大语言模型持续学习中的稳定性-可塑性平衡

    Stability-Plasticity Balance via Singular-Vector Selection in LLM Continual Learning

    [https://arxiv.org/abs/2610.11076](https://arxiv.org/abs/2610.11076)

    提出SVC方法，将奇异向量通道作为可塑性分配的基本单元，通过选择性更新这些通道来平衡大语言模型持续学习中新能力的获取与预训练知识的保存。

    

    大语言模型的领域特定持续适配面临灾难性遗忘的风险，这在获取新能力与保存在预训练期间习得的能力之间造成了根本性的矛盾。参数高效微调（PEFT）通过限制可训练参数的数量来缓解这一问题，但现有方法缺乏一个有原则的单元来决定应在何处分配可塑性、在何处保持稳定性。我们识别出奇异向量通道作为管理这种权衡的自然单元。每个通道代表一个输入-输出变换，可以对其进行更新以获取新知识，或将其固定以保存预训练能力。基于这一视角，我们提出了SVC，一种选择性更新奇异向量通道的参数高效持续学习方法。在微调之前，SVC使用领域特定数据来估计每个通道的适配收益，并仅使用一个固定的公开通用领域语料库作为历史激活代理。

    arXiv:2610.11076v1 Announce Type: cross  Abstract: Domain-specific continual adaptation of LLMs risks catastrophic forgetting, creating a fundamental tension between acquiring new capabilities and preserving those learned during pretraining. PEFT mitigates this problem by restricting the number of trainable parameters, but existing methods lack a principled unit for deciding where plasticity should be allocated and stability should be preserved. We identify the singular-vector channel as a natural unit for managing this trade-off. Each channel represents an input-output transformation, which can be updated to acquire new knowledge or fixed to preserve pretrained capabilities. Based on this perspective, we introduce SVC, a parameter-efficient continual-learning method that selectively updates Singular-Vector Channels. Before fine-tuning, SVC uses domain-specific data to estimate each channel's adaptation benefit and a fixed public general-domain corpus only as a history activation proxy
    
[^220]: 预算支出与学习的最优节奏调控

    Optimally Pacing Budget Spending and Learning

    [https://arxiv.org/abs/2610.11074](https://arxiv.org/abs/2610.11074)

    该论文提出了首个在对抗性预算受限在线学习中达到近最优遗憾界 $O(D\sqrt{\log F}+ \sqrt{T\log F})$ 的全信息算法，并可扩展至在线资源分配问题，实现了此类任务中首个 $o(\sqrt{T})$ 的次线性遗憾保证。

    

    我们为对抗环境下预算受限的在线学习问题，针对任意预算调控专家类建立了近最优的遗憾界。具体而言，给定任意包含 $F$ 个专家的类和候选预算调控方案，我们提出了一种全信息算法，其对于所有累计支出与该方案的距离保持在 $D$ 以内的专家，可获得 $O(D \sqrt{\log F}+ \sqrt{T\log F})$ 的遗憾界，与 Braverman 等人（2025）建立的下界相匹配。此外，我们还证明该技术可扩展到在线资源分配中的各类问题（即学习者可以观察到当前可用选项的收益与成本），并在允许分数分配的情形下建立了 $O(D\sqrt{\log F})$ 的遗憾界。据我们所知，这是首个能够为此类任务实现 $o(\sqrt{T})$ 保证的算法。

    arXiv:2610.11074v1 Announce Type: new  Abstract: We establish near-optimal regret bounds for budget-constrained online learning against arbitrary classes of budget-pacing experts in the adversarial setting. In particular, given any class of $F$ experts and a candidate budget pacing schedule, we provide a full-information algorithm which obtains regret $O(D \sqrt{\log F}+ \sqrt{T\log F})$ against all experts whose cumulative spending stays within distance $D$ of this schedule, matching lower bounds established by Braverman et al. (2025).   We additionally show that our technique extends to various problems in online resource allocation, where the learner gets to see the rewards and costs of the current options available to them, and establish $O(D\sqrt{\log F})$ regret bounds when fractional allocation is allowed. This is the first algorithm we are aware of which can achieve $o(\sqrt{T})$ guarantees for such tasks.
    
[^221]: CityDeploy-Bench：面向多发射机网络部署的物理接地空间集合规划基准测试

    CityDeploy-Bench: Benchmarking Physics-Grounded Spatial Set Planning for Multi-Transmitter Network Deployment

    [https://arxiv.org/abs/2610.11065](https://arxiv.org/abs/2610.11065)

    该论文提出CityDeploy-Bench基准，将多发射机网络部署重新建模为统一射线追踪验证下的物理接地空间集合规划问题，并揭示部署质量的关键在于效用模型能否捕捉发射机间的集体物理交互，而非单纯依靠更强的搜索能力。

    

    在复杂的城市传播环境与全网级干扰条件下，实现城市级无线部署的自动化仍然充满挑战。我们提出了CityDeploy-Bench，这是一个将多发射机部署重新定义为“物理接地的空间集合规划”问题的基准，并在统一的射线追踪验证器下进行评估。该基准将效用表示与规划动态相分离，从而能够在多种规划器之间对直接标量奖励、关系模型以及高阶交互结构进行受控比较。我们的实验揭示，随着物理耦合的增强，规划行为出现了明显的转变：部署质量越来越依赖于所学习的效用是否能够捕捉发射机之间的集体交互，而仅靠更强的搜索能力无法弥补关系结构的缺失。这将多发射机部署确立为一个在物理交互集合上的协调问题，而非（原文摘要在此处截断，推测为“独立放置决策的集合”）。

    arXiv:2610.11065v1 Announce Type: new  Abstract: Automating city-scale wireless deployment remains challenging under complex urban propagation and network-wide interference. We introduce \textbf{CityDeploy-Bench}, a benchmark that reframes multi-transmitter deployment as \emph{physics-grounded spatial set planning} under a unified ray-tracing verifier. The benchmark separates utility representation from planning dynamics, enabling controlled comparison between direct scalar rewards, relational models, and higher-order interaction structures across diverse planners. Our experiments reveal a clear transition in planning behavior as physical coupling grows. Deployment quality becomes increasingly dependent on whether the learned utility captures collective transmitter interactions, whereas stronger search alone cannot compensate for missing relational structure. This establishes multi-transmitter deployment as a coordination problem over physically interacting sets rather than a collectio
    
[^222]: 测量与缓解RLVR中的解模式坍缩

    Measuring and Mitigating Solution Mode Collapse in RLVR

    [https://arxiv.org/abs/2610.11064](https://arxiv.org/abs/2610.11064)

    本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。

    

    语言模型通常可以用多种方式回答同一个问题，但基于可验证奖励的强化学习（RLVR）对于模型产生哪个正确答案并不加以区分。无论一个解是某个熟悉答案的第千次重复，还是模型从未产生过的新解，它都会获得相同的奖励。然而，在训练过程中让模型保留多个正确解具有潜在价值。例如，多种模式可以为用户提供选择，并提供有助于提升模型整体性能的问题解决策略。在此，我们介绍了ModeBench，这是一个多解任务基准，其中验证器会同时返回正确性和所发现的模式。我们随后利用ModeBench来衡量RLVR后训练过程中解多样性的变化。我们发现，即使准确率保持不变甚至有所提升，RLVR后训练仍会将概率集中到更少的正确模式上，而且前沿模型几乎……

    arXiv:2610.11064v1 Announce Type: cross  Abstract: A language model (LM) can usually answer the same question in more than one way, but reinforcement learning with verifiable rewards (RLVR) is indifferent to which correct answer a model produces. A solution will earn the same reward whether it is the thousandth copy of a familiar answer or one the model has never produced before. Yet, there is potential value in having the model retain multiple correct solutions as it is trained. For instance, multiple modes may give users a choice and provide problem-solving strategies that improve overall model performance. Here, we introduce ModeBench, a benchmark of multi-solution tasks in which the verifier returns both correctness and mode discovered. We then use ModeBench to measure how solution diversity changes under RLVR post-training. We find that RLVR post-training concentrates probability onto fewer correct modes even as accuracy holds or improves, and moreover, that frontier models are al
    
[^223]: 注意力非线性催生的逆深度缩放规律

    Emergent Inverse-Depth Scaling From Nonlinearity In Attention

    [https://arxiv.org/abs/2610.11063](https://arxiv.org/abs/2610.11063)

    该论文发现，注意力的非线性性使模型能够选择性聚焦相关词元、让强弱谱方向并行学习，从而在所有数据谱下涌现出损失随深度呈逆深度衰减的缩放规律，为模型深度缩放定律提供了全新机制。

    

    缩放定律描述了模型性能随数据集规模和参数量增长而呈现的幂律提升，但其背后的机制尚未被完全理解。为了解释参数量缩放，现有理论假设性能随模型深度呈幂律缩放。在线性注意力模型中，这种缩放与幂律数据谱密切相关：由于无法选择性地关注相关词元，这类模型依据全局谱强度进行学习，较强的谱方向会先于较弱的方向被学到。然而，大型语言模型可以具有很强的非线性性。本文证明，非线性注意力在所有测试的数据谱下均会产生损失随深度呈逆深度（1/深度）式衰减的现象。非线性使注意力能够选择性地聚焦于相关词元，从而使强谱方向与弱谱方向得以并行学习。这种跨层的聚焦行为启发了与中心极限定理的联系：各层共享的误差决定了损失的收敛平台期，而……

    arXiv:2610.11063v1 Announce Type: cross  Abstract: Scaling laws describe power-law improvements in model performance with dataset size and parameter count, yet their underlying mechanisms are not fully understood. To explain the parameter count scaling, existing theory posits power-law scaling with model depth. In linear-attention models, this scaling is tied to a power-law data spectrum: unable to selectively attend to relevant tokens, these models learn according to global spectral strength, with stronger directions learned before weaker ones. Large language models, however, can be strongly nonlinear. Here, we show that nonlinear attention yields inverse-depth decay of loss across all tested data spectra. Nonlinearity enables attention to focus selectively on relevant tokens, allowing strong and weak spectral directions to be learned in parallel. Similar focusing across layers motivates a connection to the central limit theorem: shared error across layers sets the loss plateau, while
    
[^224]: 在噪声监督下学习多模态学习中应该信任什么

    Learning What to Trust in Multimodal Learning under Noisy Supervision

    [https://arxiv.org/abs/2610.11057](https://arxiv.org/abs/2610.11057)

    该论文提出REFINE框架，通过理论分析表示结构与噪声检测能力之间的关系，联合利用融合表示和单模态表示来构建更可靠的多模态标签噪声检测器，从而解决噪声监督下多模态学习依赖高质量标签的难题。

    

    多模态分类通过处理和关联来自多个模态的信息，以实现更准确的预测。然而，现有方法通常依赖于高质量的ground-truth标签，而这些标签在真实场景中难以获取。虽然针对噪声标签学习的样本选择方法旨在从含噪数据中识别正确标注的样本，但传统方法主要关注单模态设置，未能充分利用多模态信息。这促使我们在多模态学习中构建一个更可靠的噪声检测器。为此，我们从理论上分析了表示结构与噪声检测能力之间的关系。基于这一分析，我们提出了REFINE，这是一个多模态标签噪声检测框架，它联合使用融合表示和单模态表示来进行标签噪声检测。具体而言，REFINE通过判别性…（摘要在此处被截断）

    arXiv:2610.11057v1 Announce Type: cross  Abstract: Multimodal classification processes and relates information from multiple modalities to achieve more accurate predictions. However, existing methods typically rely on high-quality ground-truth labels, which are difficult to obtain in real-world scenarios. While sample-selection methods for learning with noisy labels aim to identify correctly labeled examples from noisy data, traditional methods primarily focus on unimodal settings and fail to exploit multimodal information fully. This motivates us to build a more reliable noise detector in multimodal learning. To this end, we theoretically analyze the relationship between representation structure and noise detection capability. Based on this analysis, we propose REFINE, which is a multimodal label-noise detection framework that jointly uses fused and unimodal representations for label-noise detection. Specifically, REFINE constructs discriminative eigenvectors through discriminative an
    
[^225]: AgentHorizon：评估用于长时程计算机使用任务的智能体裁判

    AgentHorizon: Evaluating Agentic Judges for Long-Horizon Computer-Use Tasks

    [https://arxiv.org/abs/2610.11050](https://arxiv.org/abs/2610.11050)

    该论文提出了AgentHorizon基准，包含1,373个来自166小时人工录制轨迹的计算机使用任务，通过指令-轨迹配对设计（包括交换指令构造的负样本）来评估智能体裁判在长时程、跨应用任务中识别轨迹违规和副作用的能力。

    

    计算机使用智能体能够完成复杂任务，这促使自动裁判在训练或无需人工参与的评估中被越来越多地用于判定任务成败。尽管这些自动裁判具有灵活性，但它们在跨越多个应用程序的长任务上的可靠性仍不明确。一条由长序列的屏幕截图和操作组成的轨迹可能看似完整，但实际上却违反了指令中的约束，或引入了不期望的副作用。为了识别这些错误，裁判需要结合用户的指令来审查轨迹。为此，我们提出了AgentHorizon，这是一个包含1,373个计算机使用任务（指令-轨迹对）的基准测试，这些任务源自跨三个操作系统、时长166小时的人工录制轨迹。通过为紧密相关的指令录制轨迹，我们可以通过交换指令来构造负样本任务。这种配对设计可以评估裁判在（原文摘要在此处截断）

    arXiv:2610.11050v1 Announce Type: new  Abstract: Computer-use agents are capable of completing complex tasks, increasing the use of automatic judges to determine success, either for training or for evaluation without human involvement. Despite their flexibility, their reliability on long tasks spanning multiple applications remains unclear. A trajectory, composed of long sequences of screenshots and actions, may appear complete, but in reality violates constraints from the instruction or introduces an unwanted side effect. To identify these errors, a judge needs to examine the trajectory with respect to the user's instruction. To this end, we introduce AgentHorizon, a benchmark of 1,373 computer-use tasks (instruction-trajectory pairs) drawn from 166 hours of human-recorded trajectories spanning three operating systems. By recording trajectories for closely related instructions, we can construct negative tasks by swapping the instructions. This paired design evaluates judges on their a
    
[^226]: FedAlphaEdit：面向协同知识编辑的零空间对齐合并

    FedAlphaEdit: Null-Space-Aligned Merging for Collaborative Knowledge Editing

    [https://arxiv.org/abs/2610.11033](https://arxiv.org/abs/2610.11033)

    提出首个在统一零空间原则下对齐本地编辑与服务器端合并规则的协同知识编辑框架 FedAlphaEdit，使多个机构无需共享原始编辑请求即可安全整合各自的知识编辑。

    

    多个机构可能各自持有私有的知识编辑请求，并希望在不共享原始编辑请求的情况下将其整合到同一个大语言模型中。AlphaEdit 等零空间约束的编辑方法在数学上保证每次更新都不会影响无关知识，而 CollabEdit 等协同框架则能在不共享数据的前提下聚合来自多个客户端的编辑。将两者结合看似轻而易举，然而我们证明这种朴素的组合在结构上是失败的，并找出了其原因。基于这一分析，我们提出了 FedAlphaEdit。据我们所知，这是首个在保护已有知识的单一零空间原则下，同时对本地编辑与服务器端合并规则进行对齐的协同知识编辑框架。FedAlphaEdit 构建于零空间对齐合并之上：客户端共享投影后的统计信息，服务器则可证明地恢复出编辑的结果

    arXiv:2610.11033v1 Announce Type: new  Abstract: Multiple institutions may each hold their own private knowledge edits and wish to integrate them into a single large language model without sharing raw edit requests. Null-space-constrained editing methods such as AlphaEdit mathematically guarantee that each update leaves unrelated knowledge intact, while collaborative frameworks such as CollabEdit aggregate edits from multiple clients without data sharing. Combining the two appears trivial. However, we show that this naive combination fails structurally, and we identify its cause. Guided by this analysis, we propose FedAlphaEdit. To our knowledge, this is the first collaborative knowledge editing framework that aligns both local editing and the server-side merging rule under a single null-space principle for preserving existing knowledge. FedAlphaEdit builds on null-space-aligned merging, in which clients share projected statistics and the server provably recovers the result of editing 
    
[^227]: NOMOS：将书面策略编译为LLM代理的静态验证工具调用门控

    NOMOS: Compiling Written Policies into Statically Verified Tool-Call Gates for LLM Agents

    [https://arxiv.org/abs/2610.11030](https://arxiv.org/abs/2610.11030)

    NOMOS是一个四遍编译器，可将自然语言策略编译为经过静态验证的确定性工具调用门控，将LLM代理在状态变更调用中的策略违规率从66.3%降至2.6%。

    

    使用工具的LLM代理会违反其被部署来执行的那些策略，而且往往是悄无声息地违反。此前的防御手段要么手工编写规则，要么对每个动作查询一次LLM验证器，要么通过重量级的形式化机制来编译策略。朴素的编译方法会失败：提取出的规则要么阻止满足其自身前提条件的工具调用，要么读取其工具并不具备的参数。NOMOS是一个四遍编译器，能将自然语言策略转化为确定性的工具调用门控；仅依靠工具模式级别的检查进行静态验证（无需证明器、求解器或LLM），即可修复或拒绝37%（航空领域）和13%（零售领域）的候选规则——若没有这一步，大多数已交付的规则将无法运作。在无防御的对话记录上回放编译后的规则，可以标记出拒绝合法工作的绑定（某个开发绑定拒绝了95.9%的可通过任务的调用）；而没有任何评估绑定被标记。在τ²-bench上，该门控将状态变更调用中违反参考编码条款的比例从66.3%降至2.6%（摘要在此处截断）。

    arXiv:2610.11030v1 Announce Type: cross  Abstract: Tool-using LLM agents violate the policies they are deployed to enforce, often silently. Prior defenses hand-write rules, query an LLM verifier per action, or compile policies through heavyweight formal machinery. Naive compilation fails: extracted rules block the tool satisfying their own precondition, or read arguments their tool lacks. NOMOS, a four-pass compiler, turns a natural-language policy into a deterministic tool-call gate; static verification with tool-schema-level checks alone (no prover, solver, or LLM) repairs or rejects 37% (airline) and 13% (retail) of candidates, without which most shipped rules are inoperable. Replaying compiled rules over undefended transcripts flags bindings that refuse legitimate work (a development binding refused 95.9% of task-passing calls); no evaluation binding is flagged. On $\tau^2$-bench the gate cuts violations of reference-encoded clauses among state-changing calls from 66.3% to 2.6% (ai
    
[^228]: 一种用于中期时效全球每日火辐射功率预测的图神经网络

    A Graph Neural Network for Global Daily Fire Radiative Power Prediction at Medium-Range Lead Times

    [https://arxiv.org/abs/2610.11022](https://arxiv.org/abs/2610.11022)

    该研究开发了一个基于时空图神经网络的数据驱动模型，利用最新可用的观测数据预测未来1至7天的全球每日火辐射功率（FRP），克服了卫星产品延迟与预报期间火输入固定不变这两项业务运行限制。

    

    提前数天对生物质燃烧活动进行准确预测，对于空气质量和气溶胶预报十分重要。本工作源于两项业务运行限制：其一，用于初始化NOAA GEFS-Aerosols的GBBEPx卫星火辐射功率（FRP）产品存在约1.5天的延迟，因此每个预报循环都只能依赖最新的、但已经过时的火观测数据；其二，这些火输入在随后5天的业务预报（在GSL实验系统中为7天）期间保持固定不变，实际上是假设火活动没有演变。我们开发了一个数据驱动模型，能够根据最新可用的观测数据预测未来1至7天的全球FRP。该模型采用时空图神经网络，利用再分析气象数据、土地覆盖与植被信息、近期火历史作为输入特征，并以GBBEPx FRP作为训练目标。该模型的训练……（原文在此处截断）

    arXiv:2610.11022v1 Announce Type: cross  Abstract: Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the GBBEPx satellite fire radiative power (FRP) product used to initialize NOAA's GEFS-Aerosols is available with about a 1.5-day latency, so each forecast cycle relies on the most recently available, but already outdated, fire observations. Second, these fire inputs are then held fixed throughout the subsequent 5-day operational forecast, or 7 days in the GSL experimental system, effectively assuming no evolution in fire activity. We develop a data-driven model that predicts global FRP one to seven days ahead from the most recent available observations. The model adapts a spatiotemporal graph neural network using reanalysis meteorology, land-cover and vegetation information, recent fire history, and GBBEPx FRP as the training target. It is traine
    
[^229]: 基于原始视频的语言模型中期训练

    Mid-Training Language Models on Raw Video

    [https://arxiv.org/abs/2610.11019](https://arxiv.org/abs/2610.11019)

    该论文首次证明无字幕、无文本损失的原始视频可作为语言模型的中期训练数据，通过预测下一个视觉token的方式，在不损害文本能力的前提下显著提升了模型在视频和图像基准上的性能。

    

    多模态大语言模型主要从成对的图文数据或标注视频中学习，而原始的网络视频很少被用于进一步训练现有的语言模型。我们研究了不带字幕、不使用文本损失的原始视频能否作为预训练语言模型的中期训练数据。视频帧被编码为连续的视觉token，语言模型学习预测下一个视觉token。我们在来自YT-Temporal-1B的原始视频片段上对Qwen3-1.7B进行中期训练，然后对其和未经中期训练的模型应用相同的图文指令微调，使两者仅在中期训练环节有所不同。经过中期训练的模型在四个视频基准上平均得分高出2.9分，在十个图像基准上平均得分高出5.1分，涵盖感知、文档和图表任务。尽管中期训练不包含任何文本，文本性能仍得以保留，在14个文本基准上平均得分为48.9，而未经中期训练的模型为48.0。

    arXiv:2610.11019v1 Announce Type: cross  Abstract: Multimodal large language models learn mostly from paired image-text data or annotated video, and raw web video is rarely used to further train an existing language model. We study whether raw video, with no captions and no text loss, can serve as mid-training data for a pretrained language model. Frames are encoded into continuous visual tokens, and the language model learns to predict the next visual token. We mid-train Qwen3-1.7B on raw clips from YT-Temporal-1B and then apply the same image-text instruction tuning to it and to the model without mid-training, so that the two differ only in mid-training. The mid-trained model scores 2.9 points higher on average across four video benchmarks and 5.1 points higher across ten image benchmarks, spanning perception, document, and chart tasks. Text performance is preserved even though mid-training includes no text, with an average of 48.9 across 14 text benchmarks compared with 48.0 for the
    
[^230]: 为LLM智能体策划常驻上下文文件：带有删失反馈的容量受限组合选择模型

    Curating Always-Loaded Context for LLM Agents: A Capacitated Assortment Model with Censored Feedback

    [https://arxiv.org/abs/2610.11007](https://arxiv.org/abs/2610.11007)

    该论文将LLM智能体常驻上下文文件的策划问题形式化为容量受限的组合选择问题，证明了最优文件大小的上界，并表明盲目追加所有有价值的指令在净价值上可能比精选最优子集任意更差。

    

    在每次会话开始时，LLM智能体会加载一个固定的上下文文件，例如AGENTS.md。文件中每个已加载的词元在会话的后续每一轮中都会被再次计费，而且这些文件随着体积增大会降低性能。然而在实践中，人工或自动化的维护者通常通过追加内容来扩充这些文件。我们将上下文策划问题形式化为一个容量受限的组合选择问题：指令在有限的注意力容量下消耗词元；添加一条指令绝不会提高其他指令的遵循度，而保留的指令会产生每次会话的固定启动成本。我们证明了无论可用候选指令的数量有多少，最优文件大小都存在一个上界；并且与选择最优子集相比，将所有具有正独立价值的指令全部追加进去，其净价值可能任意地更差。词元预算还能在词元价格被低估时限制损失。随后我们研究了……（摘要在此处被截断）

    arXiv:2610.11007v1 Announce Type: new  Abstract: At the start of every session, LLM agents load a fixed context file, such as $\texttt{AGENTS.md}$. Each loaded token in the file is charged again in every later round of the session, and these files can degrade performance as they grow in size. However, in practice, human or automated curators usually grow these files by appending.   We formulate context curation as a capacitated assortment problem. Instructions consume tokens under a finite attention capacity; adding an instruction never raises the compliance of the others, while retained instructions incur a per-session setup cost.   We prove an upper bound on the optimal file size, regardless of the number of available candidate instructions, and that appending every instruction with positive standalone value can be arbitrarily worse in net value than selecting an optimal subset. A token budget also limits the loss when the token price is underestimated.   We then examine what can be 
    
[^231]: 跨运行工况的多尺度等离子体动力学重构

    Reconstruction of Multiscale Plasma Dynamics Across Operating Regimes

    [https://arxiv.org/abs/2610.11004](https://arxiv.org/abs/2610.11004)

    该论文提出ReMAIN网络，保留SHRED的循环时间编码并用经特征级仿射调制的U-Net替换其全连接解码器，从而能够从稀疏传感器测量中重构跨运行工况的多尺度、空间分辨的等离子体动力学。

    

    从少量传感器重构具有空间分辨率的等离子体动力学，对于诊断、降阶建模和控制至关重要，但由于稀疏测量无法完全约束多尺度且依赖运行工况的自由度，这一问题仍然具有挑战性。浅层循环解码器（SHRED）通过利用测量历史部分解决了空间稀疏性问题；然而，其全连接解码器缺乏解析跨尺度空间结构或显式参数依赖的机制。我们提出了循环多尺度仿射调制推断网络，它保留了SHRED的循环时间编码，但将解码器替换为U-Net，其特征层次通过特征级线性调制受循环状态条件约束。时间表示为整个U-Net同时提供了密集先验和针对特定尺度的调制。参数化扩展联合嵌入了运行工况……（摘要原文在此处截断）

    arXiv:2610.11004v1 Announce Type: cross  Abstract: Reconstructing spatially resolved plasma dynamics from few sensors is essential for diagnostics, reduced-order modelling and control, yet remains difficult because the sparse measurements incompletely constrain multiscale, regime-dependent degrees of freedom. The Shallow Recurrent Decoder (SHRED) partially addresses spatial sparsity by using measurement histories; however, its fully connected decoder provides no explicit mechanism for resolving spatial structure across scales or explicit parametric dependency. We introduce the Recurrent Multiscale Affine-modulated Inference Network (ReMAIN), which preserves SHRED's recurrent temporal encoding but replaces its decoder with a U-Net whose feature hierarchy is conditioned by the recurrent state through feature-wise linear modulation. The temporal representation supplies both a dense prior and scale-specific modulation throughout the U-Net. A parametric extension jointly embeds the operatin
    
[^232]: 降水的低秩张量结构及其在卫星-基准数据融合中的应用

    Low-rank tensor structure of precipitation and its application to satellite-reference merging

    [https://arxiv.org/abs/2610.11000](https://arxiv.org/abs/2610.11000)

    该研究揭示了降水的低秩张量结构，并提出基于张量的TMerge框架，通过共享低秩时空因子融合卫星降水与稀疏基准观测，将美国本土IMERG降水产品的相关系数从0.53显著提升至0.85。

    

    降水的间歇性和多变性使得在大范围区域内的准确估算十分困难，但其时空结构表明可能存在低秩表示。本研究将美国本土（CONUS）的日降水表示为时空张量，并应用CANDECOMP/PARAFAC分解，结果表明保留原始的空间和时间模态进行分解，比分解独立的日降水场或展开的时空矩阵能获得更精确的重构。基于这一发现，本研究提出了TMerge，一个基于张量的框架，通过共享的低秩空间和时间因子将卫星降水与稀疏的基准观测数据融合。TMerge被应用于利用气候预测中心（CPC）基准观测数据对美国本土的IMERG Final Run产品进行校正。在2019-2022年期间，TMerge将相关系数从0.53提升至0.85，并降低了均方根误差。

    arXiv:2610.11000v1 Announce Type: new  Abstract: The intermittent and variable nature of precipitation makes its accurate estimation over extended domains difficult, yet its spatiotemporal structure suggests that a low-rank representation may be possible. This work represents daily precipitation over the contiguous United States (CONUS) as spatiotemporal tensors and applies CANDECOMP/PARAFAC factorization, showing that preserving the native spatial and temporal modes yields more accurate reconstruction than factorizing independent daily fields or unfolded space--time matrices. Building on this finding, this work presents TMerge, a tensor-based framework that integrates satellite precipitation with sparse reference observations through shared low-rank spatial and temporal factors. TMerge was applied to correct the IMERG Final Run product with climate prediction center reference observations over CONUS. During 2019-2022, TMerge increased correlation from 0.53 to 0.85 and reduced root-mea
    
[^233]: F$^3$NO：具有跨尺度条件化的频率分解有限时间流映射神经算子

    F$^3$NO: Frequency-Decomposed Finite-Time Flow-map Neural Operators with Cross-Scale Conditioning

    [https://arxiv.org/abs/2610.10998](https://arxiv.org/abs/2610.10998)

    提出频率分解流映射神经算子F$^3$NO，通过低频特征引导高频信息的跨尺度细化，并结合分段并行预测与递归传播策略，在五个PDE基准上超越了自回归和直接预测基线的预测精度。

    

    神经算子能够实现快速的偏微分方程（PDE）预测，但重复预测会累积误差，且精细尺度结构仍难以解析。我们提出了一种频率分解的有限时间流映射神经算子（F$^3$NO），它利用更新的低频特征来引导高频信息的非线性细化。在每一层内，这种跨尺度条件化将全局谱处理与局部细节细化联系起来。该模型直接预测指定未来时刻的状态，并根据预测时间间隔动态调整两个分支的贡献。对于更长的轨迹，它将短时间段内的并行预测与时间段之间的递归传播相结合。在五个PDE基准测试上的实验表明，该方法相比自回归和直接预测基线具有更高的预测精度。消融实验显示，频率分解细化机制能够以更少的参数提升预测精度。

    arXiv:2610.10998v1 Announce Type: new  Abstract: Neural operators enable fast PDE forecasting, but repeated predictions accumulate errors and fine-scale structures remain difficult to resolve. We introduce a frequency-decomposed finite-time flow-map neural operator (F$^3$NO) that leverages updated low-frequency features to guide nonlinear refinement of high-frequency information. Within each layer, this cross-scale conditioning connects global spectral processing with local detail refinement. The model directly predicts states at specified future times and adjusts the contributions of the two branches according to the prediction interval. For longer trajectories, it combines parallel predictions within short temporal segments with recursive propagation between segments. Experiments on five PDE benchmarks demonstrate improved forecasting accuracy over autoregressive and direct-prediction baselines. Ablations show that frequency-decomposed refinement can improve accuracy with fewer param
    
[^234]: 不变测度推理器：潜在推理的稳定表示

    Invariant-Measure Reasoners: Stable Representations for Latent Reasoning

    [https://arxiv.org/abs/2610.10996](https://arxiv.org/abs/2610.10996)

    提出不变测度推理器（ImR），以潜在状态在紧致子集上的长期分布即不变测度作为稳定表示，并基于预测头输出在该测度下的期望进行预测，从而解决潜在推理模型因潜在状态持续变化而导致的预测不稳定问题。

    

    潜在推理模型使用同一个循环块反复更新潜在状态。随着循环深度的增加，潜在状态序列可能收敛到状态空间的一个紧致子集，而不一定收敛到某个不动点。现有模型通常通过对单个潜在状态应用预测头来进行预测。然而，即使经过多次更新，潜在状态仍可能继续变化，这可能导致预测在不同循环深度下表现不稳定。为了解决这种不稳定性，我们提出了不变测度推理器，这是一个使用不变测度作为稳定表示的框架。该测度描述了潜在状态在紧致子集上的长期运行分布，并且在循环块的更新下保持不变。ImR 基于预测头输出在该测度下的期望来进行预测。我们以两种方式使用 ImR：仅对现有模型的预测头进行微调，以及对……（原文摘要在此处截断）

    arXiv:2610.10996v1 Announce Type: new  Abstract: Latent reasoning models repeatedly update a latent state using the same recurrent block. As the recurrent depth increases, the sequence of latent states may converge to a compact subset of the state space without necessarily converging to a fixed point. Existing models typically predict by applying a prediction head to a single latent state. However, the latent state can continue to change even after many updates, potentially making predictions unstable across recurrent depths. To address this instability, we introduce invariant-measure reasoners (ImR), a framework that uses an invariant measure as a stable representation. This measure describes the long-run distribution of latent states on the compact subset and is invariant under updates by the recurrent block. ImR predicts from the expectation of the prediction head's output under this measure. We use ImR in two ways: fine-tuning only the prediction head of existing models and trainin
    
[^235]: 面向细粒度图像检索的区域感知CLS Token增强方法

    Region-Aware CLS Token Augmentation for Fine-Grained Image Retrieval

    [https://arxiv.org/abs/2610.10991](https://arxiv.org/abs/2610.10991)

    该论文提出通过为视觉Transformer中的CLS token和寄存器token分别匹配空间区域token（伙伴patch）来增强语义表示，从而提升细粒度图像检索的性能。

    

    图像检索方法通常依赖于从图像中提取的单一全局语义描述符，例如视觉Transformer中的[CLS] token。然而，试图将图像的所有语义信息压缩到单个描述符中可能会损害下游检索性能，尤其是在细粒度检索任务中。在这项工作中，我们用精心挑选的空间token集合来增强较新视觉Transformer中的语义token，即全局[CLS] token和四个寄存器token，旨在捕获能够表征每个语义token所包含内容的空间区域表示。我们利用了DINOv2-reg模型，该模型包含能够自发学习对象级和部件级表示的寄存器token。对于每个“线索”token（[CLS]和每个寄存器token），我们找到一个“伙伴”图像patch token，并提取一个N×N的patch区域，从而生成一组局部化的ROI token。我们的方法自动……

    arXiv:2610.10991v1 Announce Type: cross  Abstract: Image retrieval methods often rely on a single global semantic descriptor extracted from an image, e.g., the [CLS] token in vision transformers. However, trying to squeeze all the semantic information of an image into a single descriptor can hurt downstream retrieval performance, especially for fine-grained retrieval tasks. In this work, we augment the semantic tokens in the newer visual transformers, the global [CLS] token and the four register tokens, with a carefully selected collection of spatial tokens, aiming to capture the spatial region representation that characterizes the contents captured in each of the semantic tokens. We leverage the DINOv2-reg model, which includes register tokens that emergently learn object and part-based representations. For each "cue" token ([CLS] and each register token), we find a "buddy" image patch token and extract an N x N patch region to produce a set of localized ROI tokens. Our approach autom
    
[^236]: Omni-Diffusion-Distill：统一多模态扩散大语言模型的少步蒸馏

    Omni-Diffusion-Distill: Few-Step Distillation of Unified Multimodal Diffusion Large Language Models

    [https://arxiv.org/abs/2610.10990](https://arxiv.org/abs/2610.10990)

    提出统一的两阶段蒸馏框架 Omni-Diffusion-Distill，在离散 token 空间中同时蒸馏图像与文本的生成和理解能力，大幅减少统一多模态扩散大语言模型的推理步数并保持其双重视觉-语言能力。

    

    统一多模态扩散大语言模型（dLLM）为图像生成和多模态理解提供了单一架构，但其迭代式解码需要数十至数百次前向传播。现有的少步蒸馏方法大多只聚焦于图像生成或文本生成中的某一方面，因此如何将完全离散的多模态 dLLM 压缩为单一高效的学生模型、同时保留其生成与理解能力，仍不明确。我们提出 Omni-Diffusion-Distill，这是一个统一的两阶段蒸馏框架，能够在大幅降低统一多模态 dLLM 推理成本的同时，保留强大的生成与理解能力。Omni-Diffusion-Distill 在离散 token 空间中对图像和文本的生成与理解蒸馏进行对齐。在第一阶段，学生模型通过重放缓存的教师轨迹来学习跳过解码步骤；在第二阶段（原文在此处截断）……

    arXiv:2610.10990v1 Announce Type: cross  Abstract: Unified multimodal diffusion large language models (dLLMs) offer a single architecture for both image generation and multimodal understanding, but their iterative decoding requires tens to hundreds of forward passes. Existing few-step distillation methods largely focus on either image generation or text generation, making it unclear how to compress a fully discrete multimodal dLLM into a single efficient student while preserving both generation and understanding. We introduce Omni-Diffusion-Distill, a unified two-stage distillation framework that retains strong generation and understanding capabilities while substantially reducing the inference cost of a unified multimodal dLLM. Omni-Diffusion-Distill aligns the distillation of both generation and understanding, for both images and text, in the discrete token space. In the first stage, the student is trained to skip decoding steps by replaying cached teacher trajectories, and in the se
    
[^237]: 多带宽分布匹配蒸馏：论分布匹配蒸馏与漂移模型的等价性

    Multi-Bandwidth Distribution Matching Distillation: On the Equivalence of Distribution Matching Distillation and Drifting Models

    [https://arxiv.org/abs/2610.10989](https://arxiv.org/abs/2610.10989)

    本文证明了分布匹配蒸馏（DMD/DMD2）与漂移模型在数学上的等价性，并据此提出了一种多带宽分布匹配蒸馏方法。

    

    研究人员一直在持续探索有效的一步生成模型，其中漂移模型近来在一步生成方面展现出巨大潜力。已有研究揭示了扩散与流式生成模型（DFSGMs）与漂移模型之间的联系。但据我们所知，尽管分布匹配蒸馏（DMD/DMD2）与漂移模型的优化目标公式几乎完全相同，尚无人在二者之间建立精确的对应关系。在本文中，我们证明：通过将预训练DFSGMs中的速度场/噪声场转换为漂移模型中的吸引力场，并从生成分布中估计排斥力场，即可建立这种等价关系。

    arXiv:2610.10989v1 Announce Type: new  Abstract: Researchers are exploring effective one-step generative model continuously, and, Drifting Models (Deng et al., 2026), demonstrate great potential in one-step generation recently. There are works that reveal the connection between Diffusion & Flow Style Generative Models (DFSGMs) (Ho et al., 2020; Song et al., 2020a;b; Lipman et al., 2022; Liu et al., 2022) and Drifting Models (Li & Zhu, 2026; Lai et al., 2026; Turan et al., 2026). But no one has yet established a precise correspondence between the Drifting Model and the widely used distillation method- Distribution Matching Distillation (DMD/DMD2) (Yin et al., 2024b;a) to the best of our knowledge, even though their optimization objective formulas are virtually identical. In this paper, we prove that by converting the velocity-field / noise-field from the pre-trained DFSGMs into the attraction force field in Drifting Models and estimating the repulsion force field from the generative dis
    
[^238]: 使用链式LMO优化大语言模型

    Optimizing Large Language Models with Chained LMOs

    [https://arxiv.org/abs/2610.10975](https://arxiv.org/abs/2610.10975)

    提出链式线性最小化预言机（chained LMOs）框架以统一解释矩阵归一化组合类优化器，并基于此提出TensorChain优化器，在Qwen3预训练中相比Muon平均节省9.6%的token。

    

    Muon启发了一大批由多种矩阵归一化组合而成的优化器家族，但这些方法仍然零散，缺乏统一的视角。我们提出了链式线性最小化预言机（chained LMOs），将这些方法视为LMO的组合。尽管这些方法在实证上取得了成功，但许多链式结构超出了标准LMO框架的范围，在光滑凸目标上可能出现发散。为了解释为什么组合仍然能够带来帮助，我们借助线性联想记忆（linear associative memory），证明了在各向异性嵌入下，链式结构可以优于Muon。在实证方面，我们提出了TensorChain，这是该框架内的一种新型优化器，它将不同层之间兼容的权重矩阵进行堆叠，并在三维张量的各个轴上进行归一化。在Qwen3 0.6B和1.7B的预训练中，TensorChain在平均token效率上超过了所有链式基线方法，在相同验证损失下相比Muon平均节省9.6%的token。

    arXiv:2610.10975v1 Announce Type: cross  Abstract: Muon has motivated a growing family of optimizers that compose multiple matrix normalizations, but these methods remain fragmented and lack a unified perspective. We introduce chained linear minimization oracles (chained LMOs), which cast these methods as compositions of LMOs. Despite their empirical success, many chains fall outside the standard LMO framework and can diverge on smooth convex objectives. To explain why composition can nevertheless help, we turn to linear associative memory and show that chaining can improve over Muon under anisotropic embeddings. Empirically, we propose TensorChain, a novel optimizer within the framework that stacks compatible weight matrices across different layers and normalizes the 3d tensor across its axes. In Qwen3 0.6B and 1.7B pretraining, TensorChain outperforms all chained baselines in average token efficiency, with average token savings of 9.6% over Muon at matched validation loss.
    
[^239]: 预算约束下面向离线策略评估的多源反事实标注

    Budgeted Multi-Source Counterfactual Annotation for Off-Policy Evaluation

    [https://arxiv.org/abs/2610.10974](https://arxiv.org/abs/2610.10974)

    该论文提出了预算约束下面向离线策略评估的多源反事实标注获取框架，将标注分配建模为整数规划问题，通过带动态规划子程序的主要化-最小化算法求解以最小化估计器方差，并刻画了标注产生价值的阈值条件。

    

    离线策略评估（OPE）旨在从记录数据中估计目标策略的价值，但行为策略有限的覆盖范围可能迫使使用高方差的重新加权或奖励模型外推。反事实标注可以补充关于未观测动作的证据，然而实际的标注来源（包括领域专家和大语言模型）可能成本高昂、存在偏差或噪声。我们研究了在上下文多臂老虎机离线策略评估中预算约束下的此类标注获取问题。给定各来源特有的成本和误差特征，我们在上下文-动作对与标注来源上构建了一个整数分配问题，以最小化估计器方差中依赖于标注方案的分量。我们通过“首次标注阈值”和局部标注价值区间刻画了标注何时具有价值。针对耦合的多源问题，我们开发了一种带有动态规划子程序的主要化-最小化算法，该算法能够单调地改进解的质量。

    arXiv:2610.10974v1 Announce Type: cross  Abstract: Off-policy evaluation (OPE) estimates the value of a target policy from logged data, but limited behavior-policy coverage can force high-variance reweighting or reward-model extrapolation. Counterfactual annotations can add evidence about unobserved actions, yet practical sources, including domain experts and large language models (LLMs), may be costly, biased, or noisy. We study budgeted acquisition of such annotations for contextual-bandit OPE. Given source-specific costs and error profiles, we formulate an integer allocation problem over context-action pairs and annotation sources to minimize the component of estimator variance that depends on the annotation plan. We characterize when annotations are valuable through a first-annotation threshold and local annotation-value regimes. For the coupled multi-source problem, we develop a majorization-minimization algorithm with dynamic-programming subroutines that monotonically improves th
    
[^240]: 神经PDE求解器中学习状态的可迁移性

    Transferability of Learned States in Neural PDE Solvers

    [https://arxiv.org/abs/2610.10972](https://arxiv.org/abs/2610.10972)

    提出“复用契约”评估框架，将神经PDE求解器中学习状态的迁移收益与解精度和计算成本解耦，并实证发现固定预测器的收益会随校正算法不同而反转。

    

    评估神经PDE求解器中的有效复用是具有挑战性的：最终精度可能同时反映源任务的学习效果和目标时刻的计算投入。我们的“复用契约”通过成对状态比较、匹配的目标信息和计算预算以及成本核算，将解的精度、学习贡献和数值效用区分开来。一项文献审计从12篇论文中提取了18条针对特定版本的协议记录，记录了所保留的状态、目标时刻的资源以及报告的对照条件。对于一个固定的线性系统和残差容限，我们构造了两个具有相同解误差、能量误差和残差范数的初始猜测，它们以不同的共轭梯度（CG）迭代次数达到相同的解。在240条源训练轨迹、两个线性PDE族、傅里叶神经算子和卷积网络的实验中，固定预测器带来的收益在不同的校正算法之间发生逆转。

    arXiv:2610.10972v1 Announce Type: new  Abstract: Assessing useful reuse in neural PDE solvers is challenging: final accuracy can reflect source learning and target-time computation. Our reuse contract separates solution accuracy, learning contribution, and numerical utility through paired state comparisons, matched target information and budgets, and cost accounting. A literature audit extracts 18 version-specific protocol records from 12 papers, documenting retained states, target-time resources, and reported controls. For a fixed linear system and residual tolerance, we construct two initial guesses with identical solution-error, energy-error, and residual norms, reaching the same solution with different conjugate-gradient (CG) iteration counts. Across 240 source-training trajectories, two linear PDE families, Fourier neural operators and convolutional networks, a fixed predictor's benefit reverses across correction algorithms. Among pairs with both relative prediction errors less th
    
[^241]: ASPIRE：通过集合预测与物理精炼发现鞍点

    ASPIRE: Saddle-Point Discovery through Set Prediction and Physical Refinement

    [https://arxiv.org/abs/2610.10969](https://arxiv.org/abs/2610.10969)

    ASPIRE框架利用等变集合预测器Ev-Quiformer从单一原子环境预测多个鞍点候选，并通过Dimer搜索在原始原子间势上进行物理精炼，解决了热激活扩散与缺陷演化模拟中鞍点发现的计算瓶颈。

    

    使用事件驱动模型预测热激活扩散和缺陷演化，需要识别原子重排机制及其激活势垒。寻找相应的鞍点是一个主要的计算瓶颈：多个重排可能源自同一个亚稳态，而代价高昂的局部搜索可能失败或反复收敛到同一个鞍点。为应对这一挑战，我们提出了ASPIRE（Atomistic Saddle-Point Inference with Refinement for Events，面向事件的原子鞍点推断与精炼框架），该框架能够从单一的初始原子环境中预测一组鞍点候选，并通过在原始原子间势上执行Dimer搜索对其进行精炼。该框架的等变集合预测器Ev-Quiformer集成了：(i) 几何条件化的标量-向量事件槽，用于生成多个鞍点候选方案；(ii) 一种解码器，通过结合原子、事件槽和锚（信息），将每个事件槽映射为完整的原子位移场。

    arXiv:2610.10969v1 Announce Type: new  Abstract: Predicting thermally activated diffusion and defect evolution with event-driven models requires identifying atomic rearrangement mechanisms and their activation barriers. Discovering the associated saddle points is a major computational bottleneck: multiple rearrangements may originate from one metastable state, while costly local searches can fail or repeatedly converge to the same saddle. To address this challenge, we introduce ASPIRE (Atomistic Saddle-Point Inference with Refinement for Events), a framework that predicts a set of saddle candidates from a single initial atomic environment and refines them through Dimer searches on the original interatomic potential. The framework's equivariant set predictor, Ev-Quiformer, integrates (i) geometry-conditioned scalar-vector event slots for generating multiple saddle-point proposals and (ii) a decoder that maps each slot to a full atomic displacement field by combining atom, slot, and anch
    
[^242]: 谱靶向Muon优化器

    Spectrally Targeted Muon

    [https://arxiv.org/abs/2610.10965](https://arxiv.org/abs/2610.10965)

    该论文提出谱靶向Muon优化器，通过阈值仅对特定范围的奇异值进行正交化，使其在归一化SGD和Muon之间插值，从而揭示Muon优化器成功背后的频谱机制。

    

    arXiv:2610.10965v1 公告类型：新论文 摘要：Muon优化器对每个更新矩阵进行正交化，将其所有奇异值设为一，已被证明在训练大型语言模型方面非常有效。然而，这种成功究竟是来自于放大梯度下降所忽视的微小奇异方向，还是来自于抑制破坏训练的大幅退化方向，目前尚不清楚。我们提出了谱靶向Muon（Spectrally Targeted Muon），它仅对高于或低于阈值τ的奇异值进行正交化，从而通过调节τ在归一化SGD和Muon之间实现插值。该方法通过在移位的Gram矩阵上进行Newton-Schulz迭代来计算投影，从而隔离相关的奇异子空间，因此无需SVD（奇异值分解）。我们在CIFAR-10和NanoGPT speedrun上评估了这些变体，跟踪了梯度、更新和权重矩阵的有效秩，以及一个新指标——更新与权重矩阵等谱流形切空间的对齐程度。我们……

    arXiv:2610.10965v1 Announce Type: new  Abstract: The Muon optimizer orthogonalizes each update matrix, setting all of its singular values to one, and has proven highly effective for training large language models. It remains unclear, however, whether this success comes from amplifying small singular directions that gradient descent neglects or from suppressing large, degenerate directions that disrupt training. We introduce Spectrally Targeted Muon, which orthogonalizes only the singular values above or below a threshold $\tau$, so that varying $\tau$ interpolates between normalized SGD and Muon. It isolates the relevant singular subspaces with projections computed by Newton-Schulz iteration on a shifted Gram matrix, so no SVD is needed. We evaluate these variants on the CIFAR-10 and NanoGPT speedruns, tracking the effective rank of gradient, update, and weight matrices and a new metric, the alignment of updates with the tangent space of the weight matrix's isospectral manifold. We fin
    
[^243]: TRACE：一种用于衡量生产级AI系统可解释性债务的治理框架

    TRACE: A Governance Framework for Measuring Explainability Debt in Production AI Systems

    [https://arxiv.org/abs/2610.10957](https://arxiv.org/abs/2610.10957)

    本文提出TRACE七工具治理框架，以可解释性债务分数（EDS）为核心，首次系统性衡量、追踪并修复生产级AI系统中逐渐累积的“可解释性债务”——即当监管者或受影响个体要求问责时无法解释单个决策的治理负债。

    

    部署在高风险领域的生产级AI系统会累积一种现有监控框架无法检测到的治理负债：当监管者、审计师或受影响个体要求问责时，系统逐渐丧失解释单个决策的能力。我们提出TRACE（透明度、风险、问责、合规与可解释性），这是一个由七种工具组成的治理框架，用于衡量、追踪和修复生产级AI系统中的可解释性债务。其基础工具——可解释性债务分数（EDS）——量化了任一时间点上低于治理定义的可解释性置信阈值的生产决策比例。补充工具包括DART（用于违约预测的债务累积率追踪器）、SHIV（用于日常治理的场景健康与完整性验证器）、FDE（用于因果归因的特征漂移评估器）、HVE（人工验证引擎）等。

    arXiv:2610.10957v1 Announce Type: cross  Abstract: Production AI systems deployed in high-stakes domains accumulate a governance liability that existing monitoring frameworks fail to detect: the progressive inability to explain individual decisions when regulators, auditors, or affected individuals demand accountability. We introduce TRACE (Transparency, Risk, Accountability, Compliance, and Explainability), a seven-instrument governance framework for measuring, tracking, and remediating Explainability Debt in production AI systems. The foundational instrument, the Explainability Debt Score (EDS), quantifies the proportion of production decisions falling below a governance-defined explainability confidence threshold at any point in time. Complementary instruments include DART (Debt Accumulation Rate Tracker for breach forecasting), SHIV (Scenario Health and Integrity Validator for daily governance), FDE (Feature Drift Evaluator for causal attribution), HVE (Human Validation Engine), AI
    
[^244]: 小数据脑解码只需 SPD-MetaFormer

    SPD-MetaFormer is what you need for small-data brain decoding

    [https://arxiv.org/abs/2610.10952](https://arxiv.org/abs/2610.10952)

    该研究发现基于SPD流形的注意力模型学到的注意力权重接近均匀、可被简单均匀权重替代而几乎不损失预测性能，据此表明架构结构比学习加权更关键，提出SPD-MetaFormer足以胜任小数据脑解码任务。

    

    脑信号解码极具挑战性，因为神经记录信号噪声大且因人而异，而带标签的数据往往十分有限。尽管如此，近年来在对称正定（SPD）流形上的注意力模型利用协方差和功能连接表示已取得了出色的性能，但其中学习到的词元加权所起的作用仍不清楚。我们研究了两种代表性架构——基于 log-Euclidean 几何的 MAtt 和基于广义 Bures–Wasserstein 几何的 GBWAtt——发现它们训练后学到的注意力权重仍然接近均匀分布。我们将这一现象归因于有界的相似度参数化：在原始的 softmax 缩放方式下，这种参数化限制了注意力权重之间的对比度。此外，无论是在训练还是评估阶段，用均匀权重替换学习到的权重，对平均预测性能几乎没有影响，同时还能保留各模型原有的聚合几何特性……

    arXiv:2610.10952v1 Announce Type: new  Abstract: Brain signal decoding is challenging because neural recordings are noisy and vary across individuals, while labeled data are often limited. Recent attention-based models on the symmetric positive definite (SPD) manifold have nevertheless achieved strong performance using covariance and connectivity representations, yet the contribution of learned token weighting remains unclear. We examine two representative architectures, MAtt (based on log-Euclidean geometry) and GBWAtt (based on generalized Bures--Wasserstein geometry), and find that their learned attention weights remain close to uniform after training. We relate this behavior to bounded similarity parameterizations that, under the original softmax scaling, limit attention-weight contrast. Moreover, replacing learned weights with uniform weights, throughout training and evaluation, has little effect on mean predictive performance while preserving each model's original aggregation geo
    
[^245]: RH-Detect：奖励作弊检测的统一基准

    RH-Detect: A Unified Benchmark for Reward Hacking Detection

    [https://arxiv.org/abs/2610.10947](https://arxiv.org/abs/2610.10947)

    提出 RH-Detect 统一基准，将十一个公开数据集的奖励作弊样本整合到通用模式中，评估发现现成大语言模型无需训练即可有效检测奖励作弊（最佳汇总 AUROC 达 0.962），但在多轮工具使用场景下性能显著下降。

    

    奖励作弊是指模型在不完成预期任务的情况下利用评估信号的现象，这一现象威胁着已部署语言模型系统的可靠性。现有数据集采用不同的标签、响应格式和元数据约定，使得检测器的结果难以相互比较。我们提出了 RH-Detect，这是一个将来自十一个公开数据集的与奖励作弊相关的子集（共 92,761 行数据，涵盖六种行为类别）整合到统一模式中的基准。在 5,021 个开放式评估单元上（每个单元包含一个任务提示和一个自由格式的模型续写，其中包括多轮工具使用轨迹），我们评估了来自五个模型家族的六个现成语言模型作为奖励作弊检测器，且无需额外训练。表现最佳的模型达到了 0.962 的汇总 AUROC，准确率超过 93%。然而，对于四个最强的模型，它们在两个多轮工具使用数据集 MALT 和 TRACE 上的准确率要低 10.7 至 15.9 个百分点。

    arXiv:2610.10947v1 Announce Type: new  Abstract: Reward hacking, where a model exploits an evaluation signal without completing the intended task, threatens the reliability of deployed language model systems. Existing datasets use different labels, response formats, and metadata conventions, making detector results difficult to compare. We present RH-Detect, a benchmark that combines reward-hacking-relevant subsets from eleven public datasets, comprising 92,761 rows and six behavior categories, into a common schema. On 5,021 open-ended evaluation units, each comprising a task prompt and a free-form model continuation, including multi-turn tool-use trajectories, we evaluate six off-the-shelf language models from five families as reward hacking detectors without additional training. The best model achieves a pooled AUROC of 0.962, with accuracy above 93%. For the four strongest models, however, accuracy on the two multi-turn tool-use datasets, MALT and TRACE, is 10.7-15.9 percentage poin
    
[^246]: AI4Fire：评估大型语言模型在野火任务上的表现

    AI4Fire: Evaluating Large Language Models on Wildfire Tasks

    [https://arxiv.org/abs/2610.10946](https://arxiv.org/abs/2610.10946)

    AI4Fire 基准首次在五个野火任务上对多个大语言模型进行成对的“裸跑”与“接地”零样本评估，发现接地信息（如只读SQL工具）在直接包含答案时能大幅提升准确率（从至多16%提升到至少88%），但简单规则基线仍然难以被超越。

    

    大型语言模型（LLM）正在进入野火管理领域，而夸大的评估结果可能造成财产损失和人员伤亡。它们在野火任务中的表现如何——无论有无“接地”？“裸跑”指模型仅接收任务输入本身；“接地”指模型额外接收一项任务特定的补充信息：例如在烟雾检测任务中，来自同一摄像头的无烟雾参考帧。AI4Fire 让六个核心模型在五个野火任务上以零样本方式分别进行裸跑和接地测试；另一轮扩展测试又增加了29个模型。我们针对火灾任务进行的文献检索共找到138篇论文，其中没有一篇能同时具备这样的模型阵容、任务覆盖范围以及成对的裸跑与接地实验。我们报告三项发现：（1）接地在补充信息直接包含答案时帮助最大：一个只读SQL工具将所有核心模型的数据库准确率从至多16%提升到至少88%。（2）简单规则难以被超越：没有核心模型的表现能超过“重复今天的人员配置数量”这一基线，两个开源权重模型大多只是复制中位数……（原文此处截断）

    arXiv:2610.10946v1 Announce Type: new  Abstract: Large language models (LLMs) are entering wildfire management, where overstated evaluations can cost property and lives. How do they perform on wildfire tasks, with and without grounding? Bare means a model receives the task input alone. Grounded means it also receives one task-specific addition: for smoke detection, a smoke-free reference frame from the same camera. AI4Fire runs six core models bare and grounded on five wildfire tasks, zero-shot; a sweep adds 29 more. Our literature search on fire tasks found 138 works; none combines this roster, task coverage, and paired bare and grounded runs. We report three findings. (1) Grounding helped most where the addition carried the answer: a read-only SQL tool lifted every core model's database accuracy from at most 16 to at least 88 percent. (2) Simple rules were hard to beat: no core model outperformed repeating today's staffing count, and two open-weight models mostly copied the median of
    
[^247]: GHARP：基于大规模重建先验的实时高斯头部动画

    GHARP: Real-time Gaussian Head Animation from Large-scale Reconstruction Prior

    [https://arxiv.org/abs/2610.10945](https://arxiv.org/abs/2610.10945)

    该论文提出GHARP方法，将头部动画解耦为离线的身份建模阶段和运行时的轻量级残差预测阶段，并利用预训练重建模型的语义结构化潜在空间，实现从少量图像和表情信号驱动、可在移动设备上实时运行的3D头部动画。

    

    我们提出了GHARP（基于大规模重建先验的实时高斯头部动画），这是一种从被摄对象的少量输入图像和驱动表情信号出发，实时对3D人头进行动画化的方法。我们将该问题解耦为两个阶段：身份阶段，离线构建被摄对象的几何与外观表示；动画阶段，在运行时预测依赖于表情的残差。这种解耦在保真度、质量和运行时间之间提供了有利的权衡：身份阶段可以较为昂贵，而动画阶段仅运行一个针对移动设备优化的轻量级网络。我们的方法在预训练重建模型的语义结构化潜在空间中执行动画，其中表情变化在空间上保持受限，从而使残差预测更加高效。该重建先验提供了一致的空间布局，支持融合多个输入视角。

    arXiv:2610.10945v1 Announce Type: cross  Abstract: We present GHARP (Real-time Gaussian Head Animation from Large-scale Reconstruction Prior), a method that animates 3D human heads in real time from a few input images of a subject and a driving expression signal. We decouple the problem into an identity stage that builds a representation of the subject's geometry and appearance offline, and an animation stage that predicts expression-dependent residuals on top of it at runtime. This separation offers a favorable trade-off with respect to fidelity, quality and runtime: the identity stage can be expensive while the animation stage runs a lightweight network, optimized for mobile devices. Our method performs animation in a semantically structured latent space of a pretrained reconstruction model, where expression changes remain spatially contained, making residual prediction efficient. This reconstruction prior provides a consistent spatial layout, allowing fusion of multiple input views 
    
[^248]: StoreBench：用于评估和训练自主运营智能体的实时电商环境

    StoreBench: A Live-Commerce Environment for Evaluating and Training Autonomous Operator Agents

    [https://arxiv.org/abs/2610.10942](https://arxiv.org/abs/2610.10942)

    StoreBench是一个让智能体在生产级电商后端上经营线上服装店的实时环境，通过全天候动态市场、校准的通过阈值和抗操纵的奖励机制，来评估和训练自主运营智能体的长程规划与经济决策能力。

    

    强化学习环境如今已成为后训练阶段提升大语言模型（LLM）能力的主要杠杆，然而大多数智能体基准测试仍然是静态的：世界只在智能体行动时才发生变化，奖励只是一个终结性判决，通过门槛也是随意设定的。我们提出StoreBench，这是一个实时电商环境，智能体在生产级电商后端上经营一家中等规模的线上服装店，以测试其在不确定性下的长程规划与经济判断能力。顾客全天候下单，供应商会重新定价并可能倒闭，市场冲击会在部分预警或毫无预警的情况下到来。智能体通过人类运营者会使用的相同的29个商家工具执行操作，并在窗口化运营预算下运行，该预算使模拟时间成为所执行操作的函数，因此模型延迟不会影响模拟时间。通过阈值根据脚本化的锚定策略进行校准，奖励则针对一系列常见的……（摘要在此处截断）

    arXiv:2610.10942v1 Announce Type: new  Abstract: Reinforcement learning environments are now a primary lever for improving large language model (LLM) capabilities in post-training, yet most agentic benchmarks remain static: the world moves only when the agent acts, the reward is a terminal verdict, and the pass bar is set arbitrarily. We introduce StoreBench, a live-commerce environment in which an agent runs a mid-size online apparel store on a production-grade commerce backend, testing long-horizon planning and economic judgment under uncertainty. Customers order around the clock, suppliers reprice and fail, and market shocks arrive with partial or no warning. The agent acts through the same 29 merchant tools a human operator would use, under a windowed operation budget that makes simulated time a function of actions taken, so model latency cannot influence simulated time. Pass thresholds are calibrated against scripted anchor policies, the reward is hardened against a catalogue of r
    
[^249]: 执行器退化下四足机器人强化学习的高阶形态先验

    Higher-Order Morphology Priors for Quadruped Reinforcement Learning Under Actuator Degradation

    [https://arxiv.org/abs/2610.10934](https://arxiv.org/abs/2610.10934)

    该论文将四足机器人建模为包含肢体级和躯干级高阶胞元的胞复形，并结合霍奇消息传递机制，证明高阶形态先验可作为有效的归纳偏置，显著提升强化学习策略在执行器退化情形下的全身补偿能力与泛化性能。

    

    执行器退化使四足机器人运动成为一个协调问题，需要各关节补偿失去的驱动能力。先前的研究表明，形态感知的图策略能够改善身体扰动下的学习与泛化性能。我们探究这些优势能否通过显式建模高阶机械结构而得到进一步增强。我们将宇树Go1机器人表示为一个包含肢体级和躯干级2秩胞元的胞复形，并应用基于霍奇理论的消息传递机制。在退化训练条件下，节点-边-面霍奇执行器在未见过的执行器退化情形中取得了最高回报，同时具有更高的存活率和更低的速度跟踪误差。这些结果证明，高阶形态结构可作为执行器退化下全身补偿的一种有效归纳偏置。

    arXiv:2610.10934v1 Announce Type: cross  Abstract: Actuator degradation turns quadruped locomotion into a coordination problem requiring joints to compensate for lost actuation. Prior work suggests that morphology-aware graph policies improve learning and generalization under body perturbations. We ask whether these benefits can be strengthened by explicitly modeling higher-order mechanical structure. We represent the Unitree Go1 as a cell complex with limb- and body-level rank-2 cells and apply Hodge-based message passing. Under degradation training, the node-edge-face Hodge actor achieves the highest return on unseen actuator degradations, with higher survival and lower velocity-tracking error. These results support higher-order morphology as a useful inductive bias for whole-body compensation under actuator degradation.
    
[^250]: 重新思考脉冲语言模型中时间编码与非线性计算之间的权衡

    Rethinking the Tradeoff Between Temporal Encoding and Nonlinear Computation in Spiking Language Models

    [https://arxiv.org/abs/2610.10933](https://arxiv.org/abs/2610.10933)

    Spora通过联合设计二值脉冲编码（UBS与BBS）和注意力算子，突破了脉冲语言模型中时间编码容量与非线性计算开销之间的权衡，仅用四个时间步即在GLUE和CoLA上超越SpikeLM。

    

    脉冲语言模型面临着在短时间窗口内表示连续语义特征与保留昂贵的非线性注意力操作之间的权衡。我们提出了Spora，它对脉冲编码和注意力算子进行联合设计。二值时间权重使T个脉冲能够以高达T比特的容量表示组合值，相比之下，基于脉冲计数的读出方式只能达到O(log₂T)比特。单极二值脉冲（UBS）利用阈值和脉冲触发的残差衰减来生成非负整数编码；双极二值脉冲（BBS）将符号与幅值分离，并为带符号激活学习一个缩放因子。这些表示支持注意力机制中的累加-移位点积运算和整数指数映射。在四个时间步长下，Spora在GLUE基准上取得76.6的平均分，在CoLA上取得44.1的MCC分数，分别超过SpikeLM 1.2分和6.2分。将BBS扩展到六个时间步后，这些分数分别提升至78.2和47.4。

    arXiv:2610.10933v1 Announce Type: cross  Abstract: Spiking language models face a tradeoff between representing continuous semantic features over short temporal windows and retaining costly nonlinear attention operations. We introduce Spora, which jointly designs spike encodings and attention operators. Binary temporal weights let $T$ spikes represent compositional values with up to $T$ bits of capacity, compared with $O(\log_2 T)$ bits for spike-count readout. Unipolar Binary Spiking (UBS) uses thresholds and spike-triggered residual decay to produce non-negative integer codes; Bipolar Binary Spiking (BBS) separates sign and magnitude and learns a scale for signed activations. These representations support accumulation-and-shift dot products and integer-exponent mappings in attention. With four time steps, Spora achieves 76.6 average GLUE score and 44.1 CoLA MCC, improving over SpikeLM by 1.2 and 6.2 points, respectively. Extending BBS to six steps raises these scores to 78.2 and 47.4
    
[^251]: 面向目标条件强化学习的世界模型策略仲裁器

    World-Model Policy Arbiter for Goal-Conditioned Reinforcement Learning

    [https://arxiv.org/abs/2610.10932](https://arxiv.org/abs/2610.10932)

    提出了世界模型策略仲裁器（WMPA），一个测试时框架，将多个冻结的目标条件策略视为一个策略组合，利用世界模型在每个状态动态选择最合适的策略执行，从而在离线目标条件强化学习中超越任何单一算法的表现。

    

    离线目标条件强化学习（GCRL）已经产生了多种多样的目标达成算法，然而没有任何单一算法能够在所有环境、所有目标、甚至同一任务的不同阶段中都表现最佳。我们不是只部署表现最好的策略，而是提出一个问题：一组冻结的目标条件策略能否作为一个策略组合被共同使用，在每个状态上动态决定应该由哪个策略来执行动作。在每个状态上选择策略并非易事：各策略自身的价值函数无法直接比较，因为它们可能采用不同的量纲尺度，有些策略甚至没有价值函数；我们需要根据每个策略可能到达的状态来评判它，尽管我们一次只能执行一个策略；我们还需要避免过于频繁地切换策略，以免控制变得不稳定。为了应对这些挑战，我们提出了世界模型策略仲裁器（WMPA），这是一个测试时框架，其输入为一组冻结的策略……

    arXiv:2610.10932v1 Announce Type: new  Abstract: Offline goal-conditioned reinforcement learning (GCRL) has produced a diverse set of goal-reaching algorithms, yet no single algorithm performs best across environments, goals, and even different phases of the same task. Rather than deploying only the best-performing policy, we ask whether a set of frozen goal-conditioned policies can be used collectively as a portfolio, deciding at every state which policy should act. Choosing a policy at each state is not straightforward. The policies' own value functions cannot be compared directly: they may use different scales, and some policies have no value function. We need to judge each policy by the states it is likely to reach, even though we can execute only one policy at a time. We also need to avoid switching so often that control becomes unstable. To address these challenges, we introduce World-Model Policy Arbiter (WMPA), a test-time framework that, given a set of frozen policies as input
    
[^252]: 符号回归中作为选择压力的系数校准

    Coefficient Calibration as Selection Pressure in Symbolic Regression

    [https://arxiv.org/abs/2610.10931](https://arxiv.org/abs/2610.10931)

    提出了一种名为DSCA的系数校准策略，通过在不同协变量分布的数据分区上独立校准并取参数均值，使校准过程本身成为进化选择压力，从而避免符号回归偏爱依赖特定样本系数的结构。

    

    在模因符号回归中，候选结构是在系数校准之后进行比较的，因此校准协议本身会影响进化选择。标准的集中式校准在每个结构的合并样本最优值处对其进行评估，忽略了这种校准在协变量偏移下的稳定性，因而可能偏爱那些拟合依赖于特定样本系数的结构。我们提出了Dirichlet-Sinkhorn常数平均法（DSCA），这是一种校准策略，它将优化数据划分为具有不同协变量分布的等大小子集，在每个分区上独立校准每个候选结构，并用所得参数的均值对其进行评估。我们证明，对于正确设定且可辨识的表达式，DSCA相对于集中式校准的额外损失在总体水平上会消失；而在模型设定错误的情况下，当各分区的校准结果不一致时，该损失将会持续存在。

    arXiv:2610.10931v1 Announce Type: new  Abstract: In memetic symbolic regression, candidate structures are compared after coefficient calibration, so the calibration protocol itself contributes to evolutionary selection. Standard centralized calibration evaluates each structure at its pooled-sample optimum, ignoring how stable this calibration is under covariate shifts, and can thus favor structures whose fit relies on sample-specific coefficients.   We propose Dirichlet-Sinkhorn Constant Averaging (DSCA), a calibration strategy that partitions the optimization data into equally sized subsets with different covariate distributions, calibrates each candidate independently on every partition, and evaluates it at the mean of the resulting parameters. We show that the excess loss of DSCA relative to centralized calibration vanishes at the population level for correctly specified, identifiable expressions, whereas under misspecification it persists when partition-specific calibrations do not
    
[^253]: 基于强化学习与博弈论的面向资源受限车联网的自适应多判别器WGAN框架

    Adaptive Multi-Discriminator WGAN Framework for Resource-Constrained Internet of Vehicles Using Reinforcement Learning and Game Theory

    [https://arxiv.org/abs/2610.10926](https://arxiv.org/abs/2610.10926)

    本文提出一种融合强化学习与博弈论协调的自适应多判别器WGAN（MD-WGAN）框架，能够在资源受限、拓扑动态变化的车联网环境中同时兼顾高精度、高效资源利用、低时延和低通信开销。

    

    将机器学习工作负载作为一种网络服务进行管理，引入了一个有别于传统模型训练的资源编排问题：应为任务分配哪些节点、如何在节点之间划分通信与计算预算，以及当连通性和节点可用性随移动性变化时如何维持服务质量。在车联网环境中部署生成对抗网络（GAN）是这一问题的一个高难度实例：资源约束、动态网络拓扑以及相互冲突的优化目标，意味着传统GAN架构无法同时实现高精度、高效的资源利用、低时延和低通信开销。本文提出了一种自适应多判别器Wasserstein GAN（MD-WGAN）框架，该框架将强化学习与博弈论协调机制相结合，以联合应对这些挑战。在我们的框架中……

    arXiv:2610.10926v1 Announce Type: new  Abstract: Managing machine learning workloads as a network service introduces a resource-orchestration problem distinct from conventional model training; which nodes should be allocated to a task, how communication and computation budgets should be divided among them, and how service quality should be sustained as connectivity and node availability change with mobility. Deploying Generative Adversarial Networks (GANs) in Internet of Vehicles (IoV) environments is a demanding instance of this problem; resource constraints, dynamic network topologies, and competing optimization objectives mean that traditional GAN architectures cannot simultaneously achieve high accuracy, efficient resource use, low delay, and low communication overhead. This paper introduces an adaptive multi-discriminator Wasserstein GAN (MD-WGAN) framework that integrates reinforcement learning with game-theoretic coordination to address these challenges jointly. In our framework
    
[^254]: 电商搜索中用于页面级布局决策的语言模型

    Language Models for Page-Level Layout Decisions in E-commerce Search

    [https://arxiv.org/abs/2610.10920](https://arxiv.org/abs/2610.10920)

    该论文研究了利用语言模型离线评估电商搜索页面级布局决策（如在特定位置插入二级堆栈）是否对用户有益，从而减少对昂贵在线A/B测试的依赖。

    

    电商搜索页面是数百万在线购物者的关键触点。虽然传统搜索引擎返回的是按排名排序的结果列表，但现代电商搜索页面越来越多地整合了推荐系统模块——例如，在特定位置展示替代产品分组的二级堆栈。当引入得当时，二级堆栈可以提升用户参与度；然而，放置位置欠佳可能会干扰浏览流程并降低主要结果的质量。与传统搜索排名不同（后者已有交织比较等成熟的评估技术），在缺乏昂贵的在线A/B测试的情况下，评估页面级布局变化（例如，何时何地插入二级堆栈）仍然具有挑战性。为了解决这一问题，我们研究了离线方法，用于评估给定的布局决策——具体而言，在特定位置包含二级堆栈——是否对用户有益。我们研究……

    arXiv:2610.10920v1 Announce Type: cross  Abstract: E-commerce search pages are critical touchpoints for millions of online shoppers. While traditional search engines return a ranked list of results, modern E-commerce search pages increasingly incorporate recommender system modules -- for example, secondary stacks that surface alternative product groupings at specific positions. When introduced appropriately, secondary stacks can improve user engagement; however, suboptimal placement may disrupt browsing flow and degrade the primary results. Unlike traditional search ranking, where evaluation techniques such as interleaving are well established, evaluating page-level layout changes e.g., when and where to insert a secondary stack remains challenging without costly online A/B testing. To address this, we study offline methods for evaluating whether a given layout decision -- specifically, the inclusion of a secondary stack at a particular position -- is beneficial to users. We investigat
    
[^255]: 数据质量度量指标的实施指南

    Implementation Guidelines for Data Quality Metrics

    [https://arxiv.org/abs/2610.10919](https://arxiv.org/abs/2610.10919)

    本文通过对ISO/IEC 25024和ISO/IEC 5259标准中的数据级度量指标进行分类并提供实施指南，使ISO数据质量度量指标变得可执行，从而弥合了数据质量维度的文本定义与实际工具实现之间的差距。

    

    尽管数据质量（DQ）研究已有数十年的历史，但文献中以文本形式定义的DQ维度（如准确性或完整性）与DQ工具——其通常实现与这些维度不一致的低级检查——之间仍然存在差距。ISO/IEC 25024和ISO/IEC 5259试图通过为每个维度定义DQ度量指标来弥合这一差距。然而，这些DQ度量指标几乎未被使用，因为标准未说明如何实现它们：例如，语法准确性度量指标统计语法准确的值，但并未说明如何判定一个值在语法上是准确的。这只是将问题转移到了另一个层面而没有解决它。因此，目前的DQ评估无法建立在标准之上。在本文中，我们使ISO DQ度量指标可执行。我们将这两个标准的所有数据级度量指标分类为：（i）无需数据之外的输入即可泛化的度量指标，（ii）可参数化的度量指标……

    arXiv:2610.10919v1 Announce Type: cross  Abstract: Despite decades of data quality (DQ) research, a gap remains between DQ dimensions, such as accuracy or completeness, which the literature defines in textual form, and DQ tools, which typically implement low-level checks that are not aligned with these dimensions. ISO/IEC 25024 and ISO/IEC 5259 attempt to bridge this gap by defining DQ metrics for each dimension. However, these DQ metrics are hardly used, because the standards leave open how to implement them: for example, the metric for syntactic accuracy counts syntactically accurate values, but does not state how to decide that a value is syntactically accurate. This simply moves the problem to another level without solving it. As a result, DQ assessment currently cannot build on the standards.   In this paper, we make the ISO DQ metrics executable. We classify all data-level metrics of both standards into (i) generalizable metrics that need no input beyond the data, (ii) parameteri
    
[^256]: 当倾听变得更容易：擦除视觉线索以实现无捷径的VLA模型

    When Listening Becomes Easier: Scrubbing Visual Cues for Shortcut-Free VLAs

    [https://arxiv.org/abs/2610.10912](https://arxiv.org/abs/2610.10912)

    本文提出无需策略回滚的表征层指标“动作边界”来衡量VLA模型对视觉捷径学习的易感性，并据此提出“任务擦除”方法擦除视觉线索、增强模型对语言指令的依赖，从而构建无捷径的VLA模型。

    

    捷径学习是机器人学习中一个普遍存在的问题。机器人演示数据集的多样性有限，可能误导策略去利用任务与无关特征（如视角或背景）之间的虚假相关性。收集足够多样化的机器人演示既昂贵又低效，这促使人们寻求算法层面的替代方案。我们聚焦于视觉-语言-动作（VLA）模型，发现不同的视觉-语言模型骨干在视觉捷径学习的易感性上存在显著差异。我们发现模型行为与我们提出的表征层面指标——动作边界——相关，且该指标无需策略回滚即可计算。视觉捷径会一致地在早期层进入动作表征，而各模型在后期层通过融合语言信息来纠正这些捷径的程度各不相同。为了增强模型对语言的注意力，我们引入了任务擦除，这是一种新的……（原文摘要在此处被截断）

    arXiv:2610.10912v1 Announce Type: cross  Abstract: Shortcut learning is a prevalent issue in robot learning. The limited diversity of robot demonstration datasets can mislead policies into exploiting spurious correlations between tasks and irrelevant features, such as viewpoint or background. Collecting sufficiently diverse robot demonstrations is costly and inefficient, motivating algorithmic alternatives. We focus on vision-language-action (VLA) models and discover that different vision-language model backbones exhibit substantially different levels of susceptibility to visual shortcut learning. We find that model behavior correlates with our proposed representation-level metric, action margin, which requires no policy rollouts. Visual shortcuts consistently enter action representations in early layers, with models differing in the extent to which later layers correct them by incorporating language information. To boost models' attention to language, we introduce task scrubbing, a ne
    
[^257]: 嵌入式机器学习上的功耗侧信道成员推理攻击

    Power Side-Channel Membership Inference Attack on Embedded Machine Learning

    [https://arxiv.org/abs/2610.10909](https://arxiv.org/abs/2610.10909)

    提出了一种名为PSCMIA的功耗侧信道成员推理攻击，无需预测概率甚至预测标签，即可直接从功耗轨迹中推断嵌入式机器学习模型的训练数据成员关系。

    

    成员推理攻击（MIA）通过判断某个样本是否被用于训练目标模型，从而威胁机器学习（ML）训练数据的隐私。现有的MIA依赖于模型输出，范围涵盖预测概率到预测标签，这一假设对于输出受限或无法访问的设备端ML系统来说可能具有局限性。然而，抑制模型输出并不能消除产生这些输出的数据依赖计算，这些计算仍可能通过物理侧信道被观测到。我们提出了PSCMIA，一种针对嵌入式ML模型的功耗侧信道成员推理攻击，它可以直接从功耗轨迹中推断成员关系，而无需预测概率，甚至无需预测标签。我们在多个数据集（MNIST、FMNIST、CIFAR10、CINIC10）、全连接（FC）和卷积神经网络（CNN）架构以及两个嵌入式平台（STM32F3、XMEGA）上对PSCMIA进行了评估。

    arXiv:2610.10909v1 Announce Type: cross  Abstract: Membership inference attacks (MIAs) threaten the privacy of machine learning (ML) training data by determining whether a sample was used to train a target model. Existing MIAs rely on model outputs, ranging from prediction probabilities to predicted labels, an assumption that can be restrictive for on-device ML systems with limited or inaccessible outputs. However, suppressing model outputs does not eliminate the data-dependent computations that produce them, which may remain observable through physical side channels. We present PSCMIA, a power side-channel membership inference attack against embedded ML models that can infer membership directly from power traces without requiring prediction probabilities or even the predicted labels. We evaluate PSCMIA across multiple datasets (MNIST, FMNIST, CIFAR10, CINIC10), fully connected (FC) and convolutional neural network (CNN) architectures, and two embedded platforms (STM32F3, XMEGA). PSCMI
    
[^258]: 你的语音质量指标有多容易被攻破？修正后的评估协议、基准测试，以及打补丁的实际收益

    How Hackable Is Your Speech Quality Metric? A Corrected Protocol, a Benchmark, and What Patching Buys

    [https://arxiv.org/abs/2610.10899](https://arxiv.org/abs/2610.10899)

    该论文提出了衡量语音质量预测器可攻破性的修正协议（以未扰动往返处理为参照并取多个随机种子攻击者的最坏结果），据此建立的基准显示四种主流预测器被攻破率差异巨大（NISQA 90%、SSL-MOS 21%、DNSMOS 14%、UTMOS 6%），且“攻击-检测-打补丁”闭环仅在自身攻击空间内提供加固。

    

    语音质量预测器越来越多地被用作奖励信号，然而目前尚无公认的衡量其可攻破性的方法。常规测量方式存在两个缺陷。第一，扰动是通过一条处理链（此处为神经编解码器）到达预测器的，而该处理链本身就会使分数发生变化；若以原始输入作为参照进行评分，这种漂移会被错误地归功于攻击。改为以未受扰动的往返处理作为参照，可使测得的可攻破性变化高达四倍（对某一种防御而言从0.31降至0.08）。第二，训练单个攻击者只是一个样本，而非一次测量：仅随机种子不同的五个攻击者对同一固定预测器的攻击成功率在0.00到0.38之间，因此防御性的结论需要取多个攻击者中的最坏情况。在这一协议下，四种已发表的预测器差异巨大：NISQA在90%的语句上被攻破，SSL-MOS为21%，DNSMOS为14%，UTMOS为6%。随后我们审计了一个“攻击-检测-打补丁”的闭环，该闭环仅在自身的攻击空间内强化预测器，

    arXiv:2610.10899v1 Announce Type: new  Abstract: Speech quality predictors are increasingly used as rewards, yet no agreed measure of their hackability exists. The usual measurement has two flaws. First, the perturbation reaches the predictor through a processing chain -- here a neural codec -- that shifts the score on its own, which scoring against the raw input charges to the attack. Referencing the unperturbed round trip instead changes measured hackability by up to a factor of four (0.31 to 0.08 for one defence). Second, one trained attacker is a sample, not a measurement: five attackers differing only in random seed reach success rates from 0.00 to 0.38 against one fixed predictor, so a defence claim needs the worst case over several. Under this protocol, four published predictors differ widely: NISQA is hacked on 90% of utterances, SSL-MOS on 21%, DNSMOS on 14% and UTMOS on 6%. We then audit a closed attack-detect-patch loop. It hardens the predictor only in its own attack space,
    
[^259]: Gen-PINNs：用于求解偏微分方程的生成对抗物理信息神经网络

    Gen-PINNs: Generative Adversarial Physics Informed Neural Networks for solving partial differential equations

    [https://arxiv.org/abs/2610.10897](https://arxiv.org/abs/2610.10897)

    提出Gen-PINNs框架，将生成对抗网络与物理信息神经网络相结合，通过动态加权物理损失和判别器评估PDE残差，显著改进了含尖锐或冲击波前行为的偏微分方程的无数据求解。

    

    物理信息神经网络（PINNs）是一种被广泛使用的无数据机器学习方法，用于求解偏微分方程（PDEs）。随着生成对抗网络（GANs）的最新进展，对抗学习在建模复杂的数据驱动问题上展现出强大能力；然而，GANs在确定性物理信息PDE求解中的应用仍然有限。在这项工作中，我们首先指出了标准PINNs在求解PDEs时的局限性，包括频谱偏差、损失不平衡和优化器停滞等问题。随后，我们提出了生成对抗物理信息神经网络（Gen-PINNs），这是一个统一的确定性残差-对抗框架，旨在改进具有尖锐或冲击波前行为的PDE的无数据求解。生成器使用动态加权的物理信息损失分量来学习潜在的PDE解，而独立的判别器则评估互补的PDE残差特征

    arXiv:2610.10897v1 Announce Type: new  Abstract: Physics-Informed Neural Networks (PINNs) are a widely used data-free method for solving Partial Differential Equations (PDEs) using machine learning. With recent advances in Generative Adversarial Networks (GANs), adversarial learning has shown strong capabilities for modeling complex data-driven problems; however, the use of GANs in deterministic physics-informed PDE solutions remains limited. In this work, we first identify limitations of standard PINNs for solving PDEs, including spectral bias, loss imbalance, and optimizer stagnation. We then propose Generative Adversarial Physics-Informed Neural Networks (Gen-PINNs), a unified deterministic residual-adversarial framework designed to improve data-free solutions of PDEs with sharp or shock-front behavior. The generator learns the underlying PDE solution using dynamically weighted physics-informed loss components, while separate discriminators evaluate complementary PDE residual featur
    
[^260]: 巴伦最优传输 I：生成建模

    Barron Optimal Transport I: Generative Modeling

    [https://arxiv.org/abs/2610.10875](https://arxiv.org/abs/2610.10875)

    本文提出了一种以神经网络复杂度为传输代价的Barron最优传输新框架，将Benamou-Brenier动力学最优传输中的平均动能$L^2$能量替换为Barron能量，为生成建模中寻找最高效的神经网络传输表示奠定了理论基础。

    

    受生成建模与采样领域最新应用的启发，我们提出了一个最优测度传输框架，其中的代价函数刻画了神经网络复杂度的概念。在基于传输的生成模型中，来自参考分布（例如高斯分布）的样本沿着常微分方程或随机微分方程被映射为目标分布的样本。当在时间上离散化时，这些方程被实现为深度残差网络，其中每个隐藏层近似相应的瞬时速度。因此，给定一对目标测度和参考测度，一个自然的问题便是寻找实现这种传输的最高效的神经网络表示。我们的出发点是Benamou和Brenier提出的最优传输动力学表述。我们将平均动能$L^2$能量替换为Barron能量——这是一种衡量函数表示复杂度的自然范数……

    arXiv:2610.10875v1 Announce Type: new  Abstract: Motivated by recent applications in generative modeling and sampling, we introduce a framework for optimal measure transport where cost captures the notion of neural network complexity. In transport-based generative models, samples from a reference distribution (e.g. Gaussian) are mapped to samples of a target distribution along ordinary or stochastic differential equations. These are implemented as deep residual networks when discretized in time, where each hidden layer approximates the associated instantaneous velocity. Thus, given a pair of target and reference measures, a natural question is to search for the most efficient neural representation that implements this transport.   Our starting point is the kinetic formulation of OT, due to Benamou and Brenier. We replace the average kinetic $L^2$ energy by the \emph{Barron} energy \cite{bach2017breaking, ma2022barron}, a natural norm which measures the complexity of representing a give
    
[^261]: 具有方差缩减的变换采样器

    Transformed Samplers with Variance Reduction

    [https://arxiv.org/abs/2610.10870](https://arxiv.org/abs/2610.10870)

    该论文提出通过学习双射（如归一化流）将MCMC采样器变换到潜空间，把简单参考分布上的泊松方程精确解推广到一般目标分布，从而获得显式控制变量以实现方差缩减。

    

    马尔可夫链蒙特卡罗（MCMC）方法是在复杂概率分布下计算期望的标准工具。控制变量可以降低估计结果的方差，但一个好的控制变量需要求解采样器的泊松方程，而该方程很少存在闭式解。当采样器的核在某个简单参考密度上具有已知的谱分解时，可以获得精确解。在我们的工作中，我们通过学习到的变量变换将这些解推广到一般的目标分布。我们训练一个双射（例如归一化流），使目标分布在潜空间中接近参考分布，并证明了马尔可夫核及其泊松解可以被任意双射所变换。在潜空间中运行此类采样器即可得到显式的控制变量，并且在该映射和目标分布的温和尾部条件下，估计量具有一致性。基于……的重要性采样（IS）

    arXiv:2610.10870v1 Announce Type: cross  Abstract: Markov chain Monte Carlo (MCMC) methods are the standard tool for computing expectations under complex probability distributions. Control variates reduce the variance of the resulting estimates, but a good control variate requires solving the Poisson equation of the sampler, which rarely admits a closed-form solution. Exact solutions are available when the sampler's kernel has a known spectral decomposition on a simple reference density. In our work, we extend these solutions to general targets through a learned change of variables. A bijection, such as a normalizing flow, is trained so that the target becomes close to the reference in a latent space, and we show that Markov kernels and their Poisson solutions are transformed by any bijection. Running such samplers in the latent space then yields explicit control variates, and the estimator is consistent under mild tail conditions on the map and target. Importance sampling (IS) from th
    
[^262]: 基于人类听众强化学习的对话语音美学模型

    Conversational Voice Aesthetic Model with Reinforcement Learning from Human Listeners

    [https://arxiv.org/abs/2610.10868](https://arxiv.org/abs/2610.10868)

    该论文提出CVAM语音大语言模型，通过合成美学描述的监督微调和基于约3万条人类听众标注的组相对策略优化（GRPO），实现对语音性别、音高、语速、情感、表达方式等九项美学属性的预测，其与人类听众的一致性超越了Gemini 3.1 Pro和开源语音LLM。

    

    我们提出了对话语音美学模型（CVAM），这是一个用于在自然对话情境中描述真实或合成语音回复的语音美学的语音大语言模型。给定一个上下文和一段回复语音，CVAM能够描述刻画语音特征的显著时刻，并预测九个类别属性，涵盖性别、音高、语速、情感和表达方式。关键挑战在于情感和表达方式等感知领域，它们本质上具有主观性，缺乏确定性的真实标准（ground truth）。为此，我们针对来自CANDOR语料库的3千条真实和合成回复，每条收集了约10份人工标注。CVAM首先在合成的美学描述和标签上进行监督微调，随后基于人类判断使用组相对策略优化进行优化。实验表明，CVAM与人类听众的一致性优于Gemini 3.1 Pro和开源语音大语言模型，并且超过了单个标注者与其余标注者之间的一致性水平。

    arXiv:2610.10868v1 Announce Type: cross  Abstract: We introduce Conversational Voice Aesthetic Model, a speech large language model for describing the voice aesthetics of real or synthetic speech responses in natural conversational contexts. Given a context and a response speech, CVAM describes salient moments that characterize the voice and predicts nine categorical attributes spanning gender, pitch, pacing, emotion, and delivery. The key challenge lies in perceptual fields such as emotion and delivery, which are inherently subjective and lack definitive ground truth. Therefore, we collect ~10 human annotations for each of 3k real and synthetic responses derived from the CANDOR corpus. CVAM is supervised finetuned on synthesized aesthetic descriptions and labels, then optimized with Group Relative Policy Optimization on human judgments. Experiments show that CVAM better agrees with human listeners than Gemini 3.1 Pro and open-source speech LLMs, and outperforms single-human-vs.-rest a
    
[^263]: 类生命网络自动机规则的形状不规则性作为分类性能的指标

    Shape irregularity of Life-Like Network Automaton rules as an indicator of classification performance

    [https://arxiv.org/abs/2610.10867](https://arxiv.org/abs/2610.10867)

    本文提出“锯齿性”指标，通过量化类生命网络自动机转移函数与锯齿形状的相似程度，作为混沌性与敏感性的理论代理，从而避免昂贵的穷举搜索，实现高效的网络分类规则选择。

    

    复杂网络（CN）分类需要既具有尺度不变性又计算高效的高层次结构表征。基于类生命网络自动机（LLNA）的方法提供了一种有趣的途径，通过利用涌现的时间模式来提取网络描述符，而无需预先提供的特征，但其效能受制于一个高成本的组合优化问题：自动机转移规则的选择。尽管现有文献依赖于对大规模应用而言不可行的穷举搜索，本研究揭示了规则空间在本质上由一个我们称为“锯齿性”的属性所结构化，该属性量化了LLNA转移函数与锯齿形状的相似程度。我们证明该指标可作为混沌性与敏感性——即在各类网络之间产生判别性动态行为所必需的性质——的理论代理。

    arXiv:2610.10867v1 Announce Type: new  Abstract: Complex Network (CN) classification requires high-level structural characterizations that are both scale-invariant and computationally efficient. Methods based on Life-Like Network Automata (LLNA) offer an interesting way to extract network descriptors by leveraging emergent temporal patterns without requiring provided features, but their efficacy is bottlenecked by a high-cost combinatorial optimization problem: the selection of the automaton transition rule. While current literature relies on exhaustive searches that are unfeasible for large-scale applications, this work reveals that the rule space is fundamentally structured by a property we term ``jaggedness'', that quantifies the resemblance of a LLNA transition function with a sawtooth shape. We demonstrate that this metric acts as a theoretical proxy for chaoticity and sensitivity -- properties essential for generating discriminative dynamic behaviors among network categories. Mor
    
[^264]: 一种用于非均质复合材料模拟的基于网格的神经能量方法

    A mesh-based neural energy method for the simulation of heterogeneous composites

    [https://arxiv.org/abs/2610.10862](https://arxiv.org/abs/2610.10862)

    本文提出基于网格的神经能量方法，通过形函数插值施加运动学约束以抑制非物理位移振荡，并用代数形函数导数替代自动微分、采用高阶高斯求积精确计算能量，从而实现非均质复合材料的高效精确模拟。

    

    对于物理信息神经网络（如深度能量方法，DEM）而言，非均质材料的建模仍然是一个挑战。DEM及其变体（本文统称为神经能量方法，NEM）提供了一个可微分的变分框架。然而，其传统的基于配点法的实现方式（C-NEM）常常存在物理上不合理的位移振荡、积分误差以及自动微分带来的高昂计算成本等问题。本工作提出了基于网格的神经能量方法，通过对位移场进行基于网格的离散化来扩展NEM。通过形函数对节点位移进行插值，M-NEM施加了一个运动学约束以抑制振荡。此外，该方法用代数形函数导数代替自动微分来计算应变，并采用高阶高斯求积来精确计算能量积分。

    arXiv:2610.10862v1 Announce Type: cross  Abstract: Modeling heterogeneous materials remains a challenge for physics-informed neural networks such as the deep energy method (DEM). The DEM and its variants, here collectively referred to as the neural energy method (NEM), offer a differentiable variational framework. However, their conventional collocation-based implementation (C-NEM) often suffers from physically inadmissible displacement oscillations, integration errors, and high computational costs from automatic differentiation. This work introduces the mesh-based neural energy method (M-NEM), extending the NEM through a mesh-based discretization of the displacement field. By interpolating nodal displacements via shape functions, the M-NEM imposes a kinematic constraint that suppresses oscillations. Furthermore, the method replaces automatic differentiation with algebraic shape function derivatives for strain computation and employs high-order Gaussian quadrature for accurate energy i
    
[^265]: 在少步蒸馏文本到图像扩散模型中实现偏好驱动的遗忘

    Enabling Preference-driven Unlearning in Few-step Distilled Text-to-Image Diffusion Models

    [https://arxiv.org/abs/2610.10859](https://arxiv.org/abs/2610.10859)

    该论文提出了一种基于偏好优化（DPO）的遗忘框架，使少步蒸馏的文本到图像扩散模型无需依赖多步去噪假设或代价高昂的重新蒸馏，即可有效遗忘有害内容。

    

    文本到图像扩散模型正越来越多地被蒸馏为少步变体并加以部署，以实现快速推理。然而，这些模型生成有害或不良内容的能力带来了重大安全风险。数据驱动的遗忘方法通过使用专门的遗忘目标对模型权重进行微调，从而抑制特定的生成内容。至关重要的是，这些目标隐式依赖于多步去噪动力学，而这一假设在少步蒸馏模型中不再成立，导致遗忘效果不佳。此外，先在非蒸馏的基础模型上执行遗忘、随后再重新蒸馏以获得遗忘后的少步蒸馏模型，会带来巨大的计算和时间开销，使其在许多场景中不切实际。因此，我们提出了一种偏好驱动的遗忘框架来解决这一局限，该框架重新审视了直接偏好优化（DPO）在扩散模型中的应用。我们表明，标准DPO及其（后续内容被截断）

    arXiv:2610.10859v1 Announce Type: cross  Abstract: Text-to-image diffusion models are increasingly distilled into few-step variants and being deployed to enable fast inference. However, their ability to generate harmful or undesired content poses significant safety risks. Data-driven unlearning methods suppress targeted generations by fine-tuning model weights using specialized unlearning objectives. Crucially, these objectives implicitly rely on multi-step denoising dynamics, an assumption that breaks down for few-step distilled (FSD) models, resulting in ineffective forgetting. Furthermore, performing unlearning on the non-distilled base model and subsequently re-distilling it to obtain an unlearned FSD model incurs substantial computational and time overhead, making it impractical in many settings. Hence, we address this limitation with a preference-driven unlearning framework that revisits Direct Preference Optimization (DPO) for diffusion models. We show that standard DPO and its 
    
[^266]: RFChipAgent：面向模拟/射频芯片设计的多智能体AI流程

    RFChipAgent: Multi-Agentic AI Flow for Analog/RF Chip Design

    [https://arxiv.org/abs/2610.10858](https://arxiv.org/abs/2610.10858)

    RFChipAgent是首个基于大语言模型多智能体协作的端到端模拟/射频芯片设计自动化流程，通过多模态RAG知识提取、拓扑选择、原理图与测试台自动搭建、闭环混合电路尺寸优化四大技术支柱，在人类监督下协同完成完整的模拟/射频电路设计。

    

    模拟/射频电路仍然是数字计算与物理世界之间的关键接口，从Wi-Fi 7到6G的新兴标准对其提出了严苛要求，然而模拟/射频设计依然是芯片开发中最耗费人力的环节之一。我们提出了RFChipAgent，这是首个用于端到端模拟/射频电路设计自动化的大语言模型（LLM）多智能体流程，其中AI智能体在人类监督下协同编排完整的设计流程。RFChipAgent建立在四大技术支柱之上：首先，一个具备私有文档级FAISS索引的多模态检索增强生成（RAG）子系统，可从现有工程文档中提取设计知识；其次，拓扑智能体驱动拓扑选择，原理图与测试台智能体自动化电路与测试台的搭建；第三，一个闭环混合电路尺寸优化引擎结合了树结构Parzen（估计器）……（摘要原文在此处截断）

    arXiv:2610.10858v1 Announce Type: cross  Abstract: Analog/RF circuits remain the critical interface between digital computation and the physical world, and emerging standards from Wi-Fi 7 to 6G place stringent demands on them, yet analog/RF design remains one of the most labor-intensive steps in chip development. We present RFChipAgent, a first-of-its-kind multi-agent flow of large language model (LLM) agents for end-to-end analog/RF circuit design automation, in which AI agents collaboratively orchestrate the complete design flow under human supervision. RFChipAgent is built around four technical pillars. First, a multimodal retrieval-augmented generation (RAG) subsystem with private per-document FAISS indexing extracts design knowledge from existing engineering documentation. Second, a topology agent drives topology selection, and a schematic and testbench agent automates circuit and testbench assembly. Third, a closed-loop hybrid circuit-sizing engine combines Tree-structured Parzen
    
[^267]: KDFP：一种面向大语言模型知识蒸馏的第一性原理方法

    KDFP: A first-principles approach to knowledge distillation in large language models

    [https://arxiv.org/abs/2610.10854](https://arxiv.org/abs/2610.10854)

    本文提出KDFP，一种基于第一性原理的白盒通用知识蒸馏新方法，使大语言模型在9个基准测试上比现有方法提升1.6%–4.9%，填补了LLM通用知识蒸馏的研究空白。

    

    知识蒸馏是一种成熟的技术，它通过使用更大、能力更强的教师模型的表示来训练小型高效的学生模型，从而提升学生模型的能力。近年来关于大语言模型（LLM）蒸馏的工作大多集中于蒸馏模型在后训练阶段习得的能力，例如指令遵循、思维链推理和工具使用。这在LLM的通用知识蒸馏领域留下了巨大的研究空白，而通用知识蒸馏对于开发适合部署在边缘设备上的高效且注重隐私的系统至关重要。我们采用第一性原理的方法，评估以往工作中的经验教训并进行新的探索，以开发适用于现代LLM的蒸馏方法。我们提出了KDFP，一种新颖的LLM白盒通用知识蒸馏方法。我们证明，KDFP在9个基准测试上比现有方法高出1.6%–4.9%。

    arXiv:2610.10854v1 Announce Type: new  Abstract: Knowledge distillation is an established technique for improving the capabilities of small, efficient student models by training them with the representations of larger, more capable teacher models. Much of the recent work in the distillation of large language models (LLMs) has focused on distilling abilities learned during post-training, such as instruction following, chain-of-thought reasoning, and tool usage. This has left a large research gap in general knowledge distillation for LLMs, which is essential for developing efficient and private systems suitable for deployment on edge devices. We take a first-principles approach, evaluating previous lessons from prior works and conducting new explorations to develop a distillation methodology suitable for modern LLMs. We present KDFP, a novel methodology for white-box general knowledge distillation in LLMs. We demonstrate that KDFP outperforms existing methods by 1.6% $-$ 4.9% across 9 be
    
[^268]: 相似的预测拟合但不同的潜在动力学：刻画脑部疾病个性化模型中学习到的动力学结构

    Similar Predictive Fit but Different Latent Dynamics: Characterizing Learned Dynamical Structure in Personalized Models of Brain Disorders

    [https://arxiv.org/abs/2610.10850](https://arxiv.org/abs/2610.10850)

    该研究提出将EEG基础模型的潜在表示映射为个性化转移依赖图的方法，发现即使预测拟合度相似，癫痫患者的潜在动力学依赖结构也显著比非癫痫者更密集，表明模型学到的动力学结构能揭示超越预测准确率的临床相关差异。

    

    随着AI模型走向临床决策和个性化治疗，理解模型学到了什么，其重要性已超越单纯的预测准确率。我们研究当预测拟合度相似时，个性化的潜在动力学是否仍能揭示与临床相关的差异。一个在坦普尔大学EEG语料库（TUEG）上预训练的轻量级CNN-Transformer EEG基础模型提取片段级表示。利用坦普尔大学癫痫语料库（TUEP），将表示映射到共享的潜在状态空间，并对每名受试者独立拟合稀疏的多项逻辑斯蒂转移分布，从而获得个性化的转移依赖图 W_n。分析共纳入198名受试者（99名癫痫/99名非癫痫）。在k=4时，癫痫受试者表现出明显更密集的学习依赖结构（p=1.1×10^-7），在k=6时呈现相同模式（19.90 vs. 13.46；p...

    arXiv:2610.10850v1 Announce Type: new  Abstract: As AI models move toward clinical decision-making and personalized treatment, understanding \emph{what} a model learns is important beyond predictive accuracy alone. We investigate whether personalized latent dynamics reveal clinically associated differences even when predictive fit is similar. A lightweight CNN--Transformer EEG foundation model pretrained on the Temple University EEG Corpus (TUEG) extracts segment-level representations. Using the Temple University Epilepsy Corpus (TUEP), representations are mapped to a shared latent-state space, and sparse multinomial logistic transition distributions (mLTD) are fit independently to each subject to obtain personalized transition-dependency graphs $W_n$. Analyses include $n{=}198$ subjects (99 epilepsy / 99 non-epilepsy). At $k{=}4$, epilepsy subjects exhibit substantially denser learned dependency structure ($p{=}1.1\times10^{-7}$), with the same pattern at $k{=}6$ (19.90 vs. 13.46; $p{
    
[^269]: 面向大语言模型的摊销式离线策略评估

    Amortized Off-Policy Evaluation for LLMs

    [https://arxiv.org/abs/2610.10848](https://arxiv.org/abs/2610.10848)

    提出PFN-OPE方法，通过在大语言模型响应构建的上下文多臂老虎机任务分布上一次性预训练先验数据拟合网络，将离线策略评估进行摊销，从而同时应对策略偏移与奖励偏移的双重挑战，无需针对每个新任务从头拟合。

    

    准确的评估对于选择部署哪个大语言模型（LLM）至关重要，然而在真实流量上测试候选模型会让真实用户接触到未经审核的模型。因此，团队通常在离线数据上评估候选模型，而这些数据是由已部署的模型产生的。这就是离线策略评估（OPE），它面临两种分布偏移：随着模型在后训练过程中不断更新，其响应会偏离日志中记录的响应（策略偏移）；同时，用于评判模型的奖励定义会随业务需求的变化而改变（奖励偏移）。经典的OPE方法并不适合这种持续部署的场景，因为它们是针对每个任务单独定义的，需要在每个新的日志数据集或奖励定义上从头开始拟合。为解决这一问题，我们提出了PFN-OPE，这是一种先验数据拟合网络，它将OPE摊销到上下文多臂老虎机任务的分布上。我们在由多个奖励函数评分的大语言模型响应池所构建的任务上进行一次性预训练，其中……

    arXiv:2610.10848v1 Announce Type: new  Abstract: Accurate evaluation is central to selecting which LLM to deploy, yet testing a candidate on live traffic exposes real users to an unvetted model. Teams therefore evaluate candidates offline, on data produced by already-deployed models. This is off-policy evaluation (OPE), and it faces two distribution shifts: as a model is updated in post-training, its responses diverge from the logged ones (policy shift), and the reward definition under which it is judged changes with business requirements (reward shift). Classical OPE methods are ill-suited to this continual-deployment setting because they are defined per task and require fitting from scratch on every new logged dataset or reward definition. To address this, we propose PFN-OPE, a prior-data fitted network that amortizes OPE across a distribution of contextual-bandit tasks. We pretrain it once on tasks constructed from a pool of LLM responses scored by several reward functions, in which
    
[^270]: AI的真实长期记忆：比重计算更快且更省钱的5000万令牌窗口

    Real Long-Term Memory for AI: A 50-Million-Token Window That Is Faster and Cheaper Than Recompute

    [https://arxiv.org/abs/2610.10845](https://arxiv.org/abs/2610.10845)

    本文提出并验证了一个名为galahad-kv的记忆层，通过将KV状态加密保存到本地NVMe磁盘并按需逐字节精确加载，实现了5000万令牌的超长上下文记忆，速度比重计算快2.8至4.3倍、GPU能耗降低8.8至12.3倍，且GPU内存占用在整个处理过程中保持恒定。

    

    大型语言模型只能使用能放入其上下文窗口的文本，并且每次发送提示词时都要重新计算其内部键值（KV）状态。我们测试了一个记忆层——公开软件包 galahad-kv，它将每个约16,000个令牌的文本块的KV状态保存到加密的本地NVMe磁盘上，之后可以逐字节精确地重新加载，而无需重新计算。我们在5000万个真实公开文本令牌上运行了该系统，通过vLLM在单块NVIDIA H100上服务，使用了Gemma 4 12B和Gemma 4 31B两个模型。在两个模型上，我们探测的每个块都成功从加密存储中加载而无需重新计算（100次探测全部成功，深度从0到5000万令牌）。加载一个块比重新计算快2.8倍到4.3倍，GPU能耗降低8.8倍到12.3倍，并且在处理整个5000万令牌流的过程中GPU内存占用保持恒定。当被问及数百万令牌之前植入的事实时，12B模型在100次中答对了82次，31B模型答对了98次。

    arXiv:2610.10845v1 Announce Type: cross  Abstract: A large language model can only use the text that fits in its context window, and it recomputes its internal key-value (KV) state for a prompt every time the prompt is sent. We test a memory layer, the public package galahad-kv, that saves the KV state of each block of about 16,000 tokens to encrypted local NVMe disk and loads it back later, byte-exact, without recomputing it. We ran it on 50,000,000 tokens of real public text, served through vLLM on one NVIDIA H100, with Gemma 4 12B and Gemma 4 31B. Every block we probed was loaded back from the encrypted store with no recompute (100 of 100, at depths from 0 to 50M tokens) on both models. Loading a block was 2.8x to 4.3x faster than recomputing it and used 8.8x to 12.3x less GPU energy, and GPU memory stayed flat over the whole 50M-token stream. Asked about facts planted millions of tokens earlier, the 12B model gave the right answer 82 times out of 100 and the 31B model 98 times out 
    
[^271]: 与时钟赛跑：迈向守时且高效的时间预算型AI智能体

    On the Clock: Towards Punctual and Productive Time-Budgeted AI Agents

    [https://arxiv.org/abs/2610.10833](https://arxiv.org/abs/2610.10833)

    该论文首次系统研究了小型LLM智能体在明确时间预算下的行为，发现仅在提示词中说明预算无法让智能体有效管理时间，并提出通过暴露计时信息的运行环境机制和预算感知强化学习两类干预措施，使智能体既守时又能高效利用时间。

    

    我们研究小型LLM智能体能否在明确的墙上时钟时间预算下有效运作，即既遵守分配的运行时间，又能高效利用可用时间。我们在MLE-Bench Lite的五个竞赛任务上评估了Qwen3.6-27B，并在Zork I（Jericho）上评估了Qwen3-4B，这两个智能体基准测试中，额外的计算时间能够显著提升性能。在最简单的设置中——即仅在提示词中说明预算——智能体无法将所陈述的预算转化为对时间的受控使用。这些失败源于时间意识方面的缺陷：一方面运行环境不提供任何计时反馈，另一方面智能体无法可靠地预判动作的持续时间，也没有学到从可用时间到合适策略的映射。我们研究了两类互补的干预措施：暴露计时信息并强制执行截止时间的基于运行环境的机制，以及具有预算感知能力的强化学习……

    arXiv:2610.10833v1 Announce Type: new  Abstract: We study whether small LLM agents can operate effectively under explicit wall-clock time budgets by both respecting the allocated runtime and using available time productively. We evaluate Qwen3.6-27B on five competitions from MLE-Bench Lite and Qwen3-4B on Zork I (Jericho), two agentic benchmarks where additional computational time can meaningfully improve performance. In the simplest setting, where the budget is stated only in the prompt, agents fail to translate the stated budget into controlled use of time. These failures arise from gaps in time awareness, since the harness provides no timing feedback, but also because they cannot reliably anticipate the duration of actions, and do not have a learned mapping from available time to an appropriate strategy. We investigate two complementary classes of interventions: harness-based mechanisms that expose timing information and enforce deadlines, and reinforcement learning with budget-awar
    
[^272]: MotherTree：在合成数据上进行元学习以改进决策树训练

    MotherTree: Meta-learning on synthetic data improves decision tree training

    [https://arxiv.org/abs/2610.10832](https://arxiv.org/abs/2610.10832)

    提出了MotherTree，一个通过合成先验元学习决策树归纳的表格Transformer，无需参考树监督即可在单次前向传播中输出与经典训练形式相同、可独立审计部署的轴对齐决策树。

    

    传统的决策树算法能够生成有效且透明的模型，这些模型可以独立于训练数据进行审计、传达和部署，但需要对每个新任务从零开始学习。相比之下，表格基础模型证明了从合成先验分布进行元学习，能够对之前未见过的任务实现强大的上下文内预测，尤其是在小样本场景下。然而，这种方法无法产生可以单独检查的独立模型。我们提出了MotherTree，这是一个元学习决策树归纳的表格Transformer：给定一个新任务的训练集，它在单次前向传播中输出一个硬性的、轴对齐的决策树，其形式与经典方法训练的树等价。MotherTree使用随机梯度下降在合成先验上进行预训练，无需参考树作为监督信号。在控制了样本规模的既有基准测试上，……

    arXiv:2610.10832v1 Announce Type: new  Abstract: Conventional decision tree algorithms produce effective, transparent models that can be audited, communicated, and deployed independently of the training data, but require learning every new task from scratch. In contrast, tabular foundation models demonstrate that meta-learning from a synthetic prior distribution enables strong in-context prediction for previously unseen tasks, especially in small-sample regimes. However, this approach does not produce a standalone model that can be inspected in isolation. We introduce MotherTree, a tabular transformer that meta-learns decision tree induction: given a training set for a new task, it outputs a hard, axis-aligned decision tree, equivalent in form to classically trained trees, in a single forward pass. MotherTree is pre-trained on a synthetic prior using stochastic gradient descent without requiring reference trees for supervision. On established benchmarks with controlled sample size, the
    
[^273]: 部分验证下的共形预测

    Conformal Prediction under Partial Verification

    [https://arxiv.org/abs/2610.10829](https://arxiv.org/abs/2610.10829)

    该论文提出了一种部分验证方法，通过刻画校准证书并在校准样本间协调验证，在产生与完整验证完全相同的预测集的同时，将验证成本降低15-82%。

    

    共形预测能够提供具有有限样本保证的预测集，但校准所需的标签验证可能代价高昂。我们开发了一种部分验证方法，其返回的预测集与完整验证的结果完全相同。我们刻画了校准证书——即足以确定共形阈值的已验证信息——并设计了一个在校准样本之间协调验证的流程。对于高覆盖率下的有限阈值，当按顺序检查候选样本时，该方法的验证成本低于最小证书成本的两倍。在检索、数学解答和配置评估等任务中，与逐个验证校准样本相比，该方法将验证成本降低了15-82%，同时产生完全相同的预测集。

    arXiv:2610.10829v1 Announce Type: cross  Abstract: Conformal prediction provides prediction sets with finite-sample guarantees, but the label verification required for calibration can be expensive. We develop a partial verification method that returns exactly the same prediction sets as complete verification. We characterize calibration certificates, the verified information sufficient to determine the conformal threshold, and design a procedure that coordinates verification across calibration examples. For finite thresholds at high coverage, its verification cost is less than twice the minimum certificate cost when candidates are checked in order. Across retrieval, mathematical solutions, and configuration evaluation, it reduces verification cost by 15-82% compared with verifying calibration examples one at a time, while producing identical prediction sets.
    
[^274]: 诊断并恢复长时程技能衔接处的观测空间偏移

    Diagnosing and Recovering from Observation-Space Shift at Long-Horizon Skill Seams

    [https://arxiv.org/abs/2610.10810](https://arxiv.org/abs/2610.10810)

    该论文发现长时程机器人操作中技能串联失败的主因是前序技能遗留的场景状态偏移（而非机器人关节构型或目标物体），并提出一个完全学习的“检测-恢复-重启”系统来诊断并恢复技能衔接处的失败。

    

    长时程机器人操作通常通过串联独立训练的技能来构建。尽管每个技能单独运行时可以很可靠，但当技能被串联时性能会急剧下降：每个下游技能必须从其前序技能留下的状态开始，而不是从其训练分布开始。我们研究了这种失败模式——观测空间偏移（OSS），并探究是什么导致了这些技能衔接处的失败。通过使用特权模拟器重置，我们发现主要的偏移来自被扰动的场景状态（例如，被前序技能遗留的打开的抽屉或次要物体），而不是来自机器人的关节配置或下游技能所操作的物体。为了验证这一诊断，我们构建了一个完全学习的“检测-恢复-重启”系统：一个任务进度监视器检测技能停滞，一个学习到的策略恢复被扰动的场景组件，并通过衔接鲁棒微调使技能得以重新执行。它能够在衔接处恢复……（摘要在此处被截断）

    arXiv:2610.10810v1 Announce Type: cross  Abstract: Long-horizon robotic manipulation is often built by chaining independently trained skills. Although each skill can be reliable in isolation, performance degrades sharply when skills are chained: each downstream skill must start from the state its predecessor leaves behind rather than from its training distribution. We study this failure mode, Observation-Space Shift (OSS), and ask what causes these skill-seam failures. Using privileged simulator resets, we find that the dominant shift comes from displaced scene state (e.g., an open drawer or secondary objects left behind by earlier skills), not from the robot's joint configuration or the object the downstream skill manipulates. To test this diagnosis, we build a fully learned detect-restore-resume system: a task-progress monitor detects the stall, a learned policy restores the displaced scene components, and seam-robust fine-tuning lets the skill resume. It recovers the seam where ever
    
[^275]: 基于干预性迁移的评分标准生成评估方法

    Evaluating Rubric Generation with Interventional Transfer

    [https://arxiv.org/abs/2610.10809](https://arxiv.org/abs/2610.10809)

    本文提出干预性迁移（IT）方法，通过扰动回答并观察评分标准是否随之同步通过/未通过变化来评估LLM生成的评分标准质量，并在HealthBench案例中展示了不同形式的干预性迁移可用于评估评分标准在不同任务中的实用性。

    

    arXiv:2610.10809v1 公告类型：新论文 摘要：针对具体实例的评分标准在AI基准测试中十分常见，因为可靠的评估往往需要特定的专家知识。然而这种方法难以规模化，这促使研究者开始探索利用大语言模型（LLM）来生成评分标准。但是，即使有专家评分标准作为参考，如何在大规模场景下有效评估生成评分标准的质量仍然不明确。在本文中，我们提出了一种评估评分标准生成的方法，称之为干预性迁移（Interventional Transfer，IT），其核心思想是：当对某个回答进行扰动使其通过/未通过其中一个评分标准时，如果两个评分标准随之同步变化，则认为这两个评分标准是相似的。与现有的评分标准生成评估方法不同，我们认为不同形式的干预性迁移可以用来评估生成的评分标准在不同任务中的实用性。例如，我们将该方法应用于HealthBench的案例研究，其中我们展示了评分标准的不对称性……

    arXiv:2610.10809v1 Announce Type: new  Abstract: Instance-specific rubrics are common in AI benchmarks where reliable evaluation requires specific expert knowledge. This approach is difficult to scale, prompting research into the generation of rubrics with large language models (LLMs). However, even when expert rubrics are available as references, it is unclear how to productively evaluate the quality of generated rubrics at scale. In this paper, we introduce a method for the evaluation of rubric generation, which we call Interventional Transfer (IT), based on the idea that two rubrics are similar if they move together when a response is perturbed to pass/fail one of them. In contrast to existing approaches for evaluation of rubric generation, we argue that different forms of interventional transfer can be used to evaluate the utility of generated rubrics for different tasks. For instance, we apply this approach in a case study on HealthBench, where we demonstrate an asymmetry in rubri
    
[^276]: 三通道分数冲突中的受控获取与弃权

    Controlled Acquisition and Abstention in Three-Channel Score Conflicts

    [https://arxiv.org/abs/2610.10808](https://arxiv.org/abs/2610.10808)

    该论文构建了一个受控的三分数基准，用于研究音频、视频、文本分数冲突时策略应何时付费获取第三个分数或选择弃权，并证明简单的阈值策略能以更少的信息请求实现更高的针对性决策准确率和效用。

    

    当音频、视频和文本相互矛盾时，仅凭准确率无法判断是应该获取另一个信息源还是选择弃权。我们在一个受控的三分数基准中研究这些决策：一个策略观察两个带符号的分数，可以付出一定代价请求第三个分数，也可以选择弃权。主要奖励是机制特定的：弃权仅对一种指定的歧义机制是正确的，而在混合损坏情况下会受到惩罚。匹配对照实验表明，阈值策略能够以更少的请求匹配“始终请求”策略的决策；其相对于“始终回答融合”策略的优势取决于分配给该歧义机制的奖励。在一个部分保留的合成数据划分上，阈值策略在83个随机种子下达到了0.789 ± 0.006的针对性决策准确率和0.481 ± 0.014的效用。作为对比，三分数多数投票参考方法达到0.626 ± 0.008的准确率和0.252 ± 0.016的效用，但使用了更多的信息。在预算匹配测试中，仅使用训练数据的价值选择器相比无查询基线提高了效用。

    arXiv:2610.10808v1 Announce Type: new  Abstract: When audio, video, and text disagree, accuracy alone does not show whether to acquire another source or abstain. We study these choices in a controlled three-score benchmark: a policy observes two signed scores, may request the third at a cost, and can abstain. The primary reward is mechanism-specific: abstention is correct only for one designated ambiguity mechanism and is penalized under mixed corruption. Matched controls show that a threshold policy matches always-request decisions with fewer requests; its advantage over always-answer fusion depends on the reward assigned to that ambiguity. On a partially held-out synthetic split, the threshold policy reaches 0.789 +/- 0.006 targeted decision accuracy and 0.481 +/- 0.014 utility across 83 seeds. A three-score majority reference reaches 0.626 +/- 0.008 and 0.252 +/- 0.016, but uses more information. In a matched-budget test, a train-only value selector improves utility over no-query an
    
[^277]: 未归一化分布的符号密度估计器

    Symbolic Density Estimators for Unnormalized Distributions

    [https://arxiv.org/abs/2610.10807](https://arxiv.org/abs/2610.10807)

    本文提出一个将深度生成模型与符号回归相结合的框架，利用相互作用范围、基函数集合等领域先验知识，从观测样本中自动估计未归一化分布的符号表达式，减少了对专家人工选择函数形式的依赖。

    

    从观测样本中估计概率密度函数的符号或解析形式是统计与计算建模中的一项基本挑战。这一过程对于推导刻画潜在现象的可解释且可泛化的关系至关重要。传统上，这种估计高度依赖领域专业知识与先验的领域知识，专家需要基于经验证据和理论理解来选择合适的函数形式或参数族，随后通常通过参数估计来确定这些形式的系数。在本文中，我们开发了一个框架，利用领域特定的先验知识（如相互作用范围和预定义的基函数集合），从观测样本中估计未归一化分布的符号表达式。我们将深度生成模型与符号回归（SR）相结合……

    arXiv:2610.10807v1 Announce Type: new  Abstract: Estimating the symbolic or analytical form of probability density functions (PDFs) from observed samples is a fundamental challenge in statistical and computational modelling. This process is critical for deriving interpretable and generalizable relationships characterizing the underlying phenomenon. Traditionally, this estimation depends strongly on domain expertise and prior field-specific knowledge, with experts selecting appropriate functional forms or parametric families based on empirical evidence and theoretical understanding. The coefficients of these forms are then typically determined through parameter estimation. In this paper, we develop a framework for estimating symbolic expressions of unnormalized distributions from observed samples using domain-specific prior knowledge, such as the range of interactions and a predefined set of primitive functions. We integrate deep generative models with symbolic regression (SR), incorpor
    
[^278]: 谁的真值？拥抱以人为中心的AI中的模糊性

    Whose Ground Truth? Embracing Ambiguity in Human-Centered AI

    [https://arxiv.org/abs/2610.10805](https://arxiv.org/abs/2610.10805)

    这篇立场论文主张AI开发应摒弃“单一确定真值”的传统假设，转而建模人类合理判断的解读空间，将有意义的模糊性与标注噪声区分开来，从而实现真正以人为中心的AI。

    

    随着AI系统越来越多地与人类交互并对人类做出决策，理解人类的解读方式成为开发以人为中心的AI的重要组成部分。传统的机器学习和AI系统大多建立在存在单一确定真值的假设之上，人类标注的差异性通常通过聚合方式来解决，或者被视为噪声。然而，对于许多以人为中心的任务而言，人类的解读本质上是模糊的，对同一输入的多种解读可能同时是合理且有效的。将这种模糊性简化为单一目标，可能会忽略关于人类感知、判断和经验多样性的有意义信息。在这篇立场论文中，我们呼吁转向对合理人类判断的解读空间进行建模，同时将有意义的模糊性与标注噪声区分开来。我们认为这一视角应当……

    arXiv:2610.10805v1 Announce Type: new  Abstract: As AI systems increasingly interact with people and make decisions about them, understanding human interpretations becomes an important part of developing human-centered AI. Conventional machine learning and AI systems are largely developed under the assumption that a single definitive ground truth exists, with variability in human annotations often resolved through aggregation or treated as noise. However, for many human-centered tasks, human interpretation is inherently ambiguous, and multiple interpretations of the same input may be simultaneously reasonable and valid. Reducing such ambiguity to a single target risks overlooking meaningful information about the diversity of human perception, judgment, and experience. In this position paper, we call for a shift towards modeling the interpretation space of plausible human judgments, while distinguishing meaningful ambiguity from annotation noise. We argue that this perspective should gu
    
[^279]: 异构性下排序需要多少次重复成对比较？

    How Many Repeated Pairwise Comparisons Are Needed for Ranking under Heterogeneity?

    [https://arxiv.org/abs/2610.10795](https://arxiv.org/abs/2610.10795)

    该论文证明了在异构 Bradley-Terry 模型下，通过改进的 MLE 变体算法和随机化“俄罗斯轮盘赌”算法，每个用户-任务上下文仅需 $O(\log(1/\Delta))$ 次重复成对比较即可恢复排序，并证明该对数复杂度是最优的。

    

    我们研究当偏好在不同用户和任务之间存在差异时，基于成对比较的群体平均效用的排序模型。先前的工作表明，即使用户数量任意多，每个用户仅进行一次比较也可能不足以识别出平均效用最高的备选方案（Golz 等，2025）。我们研究了在每个用户-任务上下文内，为恢复排序所需重复比较次数的必要且充分条件。在具有固定逆温度的异构 Bradley-Terry 模型下，我们首先提出一个基于 MLE 的朴素算法，该算法需要每个上下文 $\Omega(1/\Delta^2)$ 次重复比较才能确保排序恢复。随后，我们提出两种基于 MLE 的改进变体以及一种随机化的“俄罗斯轮盘赌”式算法，它们仅需每个上下文 $O(\log(1/\Delta))$ 次重复比较即可恢复排序，并且我们证明这种对数依赖性是最优的。尽管存在这一最坏情况下的要求，我们的俄罗斯轮盘赌算法（摘要在此处中断）

    arXiv:2610.10795v1 Announce Type: new  Abstract: We study ranking models by population-average utility from pairwise comparisons when preferences vary across users and tasks. Prior work shows that a single comparison per user can be insufficient to identify the alternative with the highest average utility, even with arbitrarily many users (Golz et al., 2025). We investigate how many repeated comparisons within each user-task context are necessary and sufficient for ranking recovery. Under a heterogeneous Bradley-Terry model with fixed inverse temperature, we start with a naive MLE-based algorithm that requires $\Omega(1/\Delta^2)$ repeated comparisons per context to ensure ranking recovery. We then present two MLE-based variants and a randomized Russian Roulette-style algorithm that recover the ranking using $O(\log(1/\Delta))$ repeated comparisons per context, and we prove that this logarithmic dependence is optimal. Despite this worst-case requirement, our Russian Roulette algorithm 
    
[^280]: 通过诊断性传输校准模糊集以用于分布鲁棒优化

    Calibrating Ambiguity Set via Diagnostic Transport for Distributionally Robust Optimization

    [https://arxiv.org/abs/2610.10793](https://arxiv.org/abs/2610.10793)

    本文提出诊断传输DRO（DT-DRO），利用留出校准数据和条件概率积分变换来诊断预测误差，自适应地调整模糊集的中心与几何结构，从而在保证决策风险可控的同时避免DRO决策过度保守。

    

    分布鲁棒优化（DRO）通过在模糊集上进行优化来保护决策免受分布不确定性的影响，但当模糊集的几何结构与问题不匹配时，往往需要设置较大的半径，从而导致决策过于保守。我们提出了诊断传输DRO（DT-DRO），它利用留出的校准数据使模糊集的几何结构适应观测到的预测误差。DT-DRO使用条件概率积分变换累积分布函数来诊断系统性的概率错配，并将该信息转化为结果层面的传输，从而联合调整模糊集的中心和基础成本。由此得到的公式具有计算上易于处理的对偶重构形式。在理论方面，我们推导了有效的模糊半径和决策风险保证，这些保证会随着估计误差和近似误差的消失而收紧，并证明DT-DRO能够消除由模型误设引起的无法消除的鲁棒性下限（摘要在此处被截断）。

    arXiv:2610.10793v1 Announce Type: cross  Abstract: Distributionally robust optimization (DRO) protects decisions against distributional uncertainty by optimizing over an ambiguity set, but poorly aligned set geometry can require large radii and yield overly conservative decisions. We introduce diagnostic-transport DRO (DT-DRO), which uses held-out calibration data to adapt the ambiguity-set geometry to observed predictive errors. DT-DRO uses the conditional probability integral transform cumulative distribution function to diagnose systematic probability misallocation and translates this information into an outcome-level transport that jointly adjusts the ambiguity-set center and ground cost. The resulting formulation admits a computationally tractable dual reformulation. Theoretically, we derive valid ambiguity radii and decision-risk guarantees that tighten as estimation and approximation errors vanish, and show that DT-DRO can eliminate the nonvanishing robustness floor caused by mo
    
[^281]: NavGPT-3：在分层导航运行时中利用上下文

    NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime

    [https://arxiv.org/abs/2610.10787](https://arxiv.org/abs/2610.10787)

    NavGPT-3 提出了一个类似操作系统的分层导航运行时框架，将具备长时程推理能力的语言模型与低延迟的 VLA 动作策略通过多线程调度机制相结合，使机器人能通过线程中断与切换快速响应突发真实事件，其 8B VLA 模型在 R2R-CE 上取得了 74.51 SR 的领先性能。

    

    通过长时程智能体强化学习训练的语言模型能够通过推理泛化知识、表达精确动作，并在多步骤中追求目标，从而提升了具身智能体所能理解和决策的上限。然而，物理交互仍然是动作策略的领域，动作策略提供密集、低延迟的控制。我们提出了 NavGPT-3，这是一个连接两种模型的框架，其上构建了类似操作系统的运行时：推理、行动和监控作为各自拥有独立上下文、工具和权限的线程运行，而运行时负责调度这些线程并决定哪个线程控制机器人的运动，使机器人能够通过中断和线程切换对突发的真实世界事件做出反应。在其底层，我们的动作策略 NavGPT VLA 在 1928 万条样本上训练，采用编解码器分配方式按场景变化比例分配视觉 token；其 8B 模型在 R2R-CE 上单独达到 74.51 SR 并处于领先地位。

    arXiv:2610.10787v1 Announce Type: cross  Abstract: Language models trained with long-horizon agentic reinforcement learning can generalize knowledge through reasoning, express precise actions, and pursue goals over many steps, raising the ceiling on what an embodied agent can understand and decide. Physical interaction, however, remains the domain of action policies, which provide dense, low-latency control. We present NavGPT-3, a harness that connects the two models, with an OS-like runtime built above it: reasoning, acting, and monitoring run as threads with their own context, tools, and permissions, while the runtime schedules them and decides which thread controls the robot's motion, so that the robot can react to sudden real-world events through interruption and thread switching. Beneath it, our action policy NavGPT VLA, trained on 19.28M examples, allocates visual tokens using codec allocation, in proportion to scene change; its 8B model alone reaches 74.51 SR on R2R-CE and leads
    
[^282]: Plan-and-Patch：面向智能体规划的扩散语言模型

    Plan-and-Patch: Diffusion Language Models for Agentic Planning

    [https://arxiv.org/abs/2610.10786](https://arxiv.org/abs/2610.10786)

    提出Plan-and-Patch框架，利用扩散语言模型通过并行去掩码生成结构化的类程序计划，并借助仅填充受影响区域、保持前后步骤不变的局部修复机制，实现对计划的高效修订。

    

    规划对于长程智能体而言日益重要，成功的执行需要在多个步骤中协调子目标、工具使用和中间结果。然而，规划阶段做出的假设可能被环境推翻，工具可能返回意外的结果，动作也可能失败。因此，有效的智能体不仅需要生成计划，还必须能够对计划进行修改。此类修改往往只影响计划的一部分，其前后的结构保持不变。与其重新生成整个计划而冒着引入不必要更改的风险，不如基于被保留的前缀和后缀，仅对受影响的区域进行重新生成以完成修复。我们提出了Plan-and-Patch，这是一个“规划-执行”框架，其中扩散语言模型（dLLM）通过并行去掩码的方式生成结构化的、类似程序的计划，并通过在保持周围步骤固定的情况下填充选定区域来修复计划。我们比较了DreamReasoner-8B和Qwen3-8B作为扩散……（摘要在此处截断）

    arXiv:2610.10786v1 Announce Type: new  Abstract: Planning is increasingly important for long-horizon agents, where successful execution requires coordinating subgoals, tool use, and intermediate outcomes over many steps. Yet assumptions made during planning may be invalidated by the environment, tools may return unexpected results, or actions may fail. Effective agents must therefore not only generate plans, but also revise them. Such revisions often affect only part of a plan, leaving the preceding and subsequent structure intact. Rather than regenerate the entire plan and risk unnecessary changes, repair can regenerate the affected region conditioned on the preserved prefix and suffix. We introduce Plan-and-Patch, a plan-and-act framework in which a diffusion language model (dLLM) generates a structured, program-like plan through parallel unmasking and repairs it by filling in selected regions while keeping the surrounding steps fixed. We compare DreamReasoner-8B and Qwen3-8B as diff
    
[^283]: MemoWM：世界模型如何改变智能体需要记忆的内容

    MemoWM: How World Models Change What Agents Need to Remember

    [https://arxiv.org/abs/2610.10778](https://arxiv.org/abs/2610.10778)

    MemoWM提出了一种基于世界模型的记忆分配框架，利用共享预测压缩经验存储并重建省略内容，在五个长期智能体记忆基准上实现了答案准确率超越最强基线2.62个百分点，同时相对最高效基线减少53.9%的每条经验存储量。

    

    长期运行的智能体在积累经验的过程中面临不断增长的存储需求。世界模型能够捕捉可复用的规律，从而减少每条经验所需存储的信息。我们将基于世界模型条件下的记忆分配问题形式化，并提出了MemoWM框架，该框架利用共享预测来压缩保留的信息并重建被省略的内容。其任务感知的分配规则在重建错误的预期影响与存储成本之间取得平衡，保留那些在预测先验之外仍具有下游价值的信息。在五个长期智能体记忆基准测试中，MemoWM实现了42.42%的平均答案准确率，超过最强基线2.62个百分点，同时相对于存储效率最高的基线MIRIX，平均每条经验的专用存储减少了53.9%。进一步分析表明，更强的世界模型在相当的任务质量下能够进一步降低每条经验的存储量。

    arXiv:2610.10778v1 Announce Type: cross  Abstract: Long-term agents face growing storage demands as they accumulate experience. World models capture reusable regularities that can reduce the information stored for each experience. We formulate the problem of memory allocation conditioned on a world model and introduce MemoWM, a framework that uses shared predictions to compress retained information and reconstruct omitted content. Its task-aware allocation rule balances the expected impact of reconstruction errors against storage cost, retaining information with downstream value beyond the predictive prior. Across five long-term agent-memory benchmarks, MemoWM achieves 42.42\% average answer accuracy, exceeding the strongest baseline by 2.62 percentage points, while reducing average experience-specific storage by 53.9\% relative to MIRIX, the most storage-efficient baseline. Further analysis shows that stronger world models reduce per-experience storage at comparable task quality. Acco
    
[^284]: NEMORA：用于长程原子体系学习的神经等变多极算子

    NEMORA: Neural Equivariant Multipole Operators for Long-Range Atomistic Learning

    [https://arxiv.org/abs/2610.10776](https://arxiv.org/abs/2610.10776)

    NEMORA 提出了快速多极子方法的神经等变扩展，实现了可学习的长程等变相互作用传输，兼顾多尺度多体表达能力与对更大体系的高效扩展性，克服了现有方法在长程信息传递上的局限。

    

    等变图神经网络已成为机器学习原子间势的基础架构，能够以极低的计算成本逼近量子化学精度。这类模型可以准确描述局部原子环境，但有限的空间截断会切断长程信息流，而堆叠消息传递层则可能导致过度平滑和过度压缩问题。现有的长程扩展方法要么预先设定固定的解析传播核，要么将长程通信限制在标量或保度通道中，要么仅具备近似等变性，要么会带来超线性的计算成本。如何将可学习的长程等变输运与多尺度多体表达能力相结合，并在更大体系上实现高效扩展，仍然是一个核心挑战。我们提出了神经等变多极算子（NEMORA），这是快速多极子方法（FMM）的一种神经等变扩展，用于学习……

    arXiv:2610.10776v1 Announce Type: new  Abstract: Equivariant graph neural networks have emerged as foundational architectures for machine-learned interatomic potentials, approaching quantum-chemical accuracy at a fraction of the computational cost. These models describe local atomic environments accurately, but finite spatial cutoffs truncate long-range information flow, and stacking message-passing layers can lead to over-smoothing and over-squashing. Existing long-range extensions either prescribe a fixed analytical propagation kernel, restrict long-range communication to scalars or degree-preserving channels, are only approximately equivariant, or incur super-linear computational cost. Combining learnable long-range equivariant transport with multiscale many-body expressivity and efficient scaling for larger systems remains a central challenge. We introduce Neural Equivariant Multipole Operators (NEMORA), a neural equivariant extension of the Fast Multipole Method (FMM) for learning
    
[^285]: 能源转型中价值创造的战略投资决策：一种强化学习方法

    Strategic Investment Decision Making for Value Creation in Energy Transition: A Reinforcement Learning Approach

    [https://arxiv.org/abs/2610.10768](https://arxiv.org/abs/2610.10768)

    该论文开发了一个定制的能源模拟环境和基于强化学习的多标准顺序决策框架，帮助能源公司在不确定性下于油气、可再生能源和碳减排三大领域之间进行战略资金分配，从而最大化能源转型中的价值创造。

    

    气候变化这一全球性挑战推动了减少二氧化碳排放的重大举措，并以2015年《巴黎协定》等国际协议为指导。行动过慢可能导致未来的损失和声誉损害，而行动过快则可能因许多可再生能源项目边际盈利能力不足或技术不成熟带来的潜在损失而危及股东价值。为了应对这一复杂的转型，能源公司必须采用顺序决策（SDM）策略，以便在不确定性条件下最大化决策灵活性所带来的价值创造。为支持这一目标，我们开发了一个定制的模拟环境，用于建模至2050年的动态能源格局。在此基础上，我们设计了一个多标准的顺序决策框架，探索与在石油与天然气、可再生能源和二氧化碳减排三个领域之间分配资金的不同投资组合相关的各类决策策略，旨在最大化价值。

    arXiv:2610.10768v1 Announce Type: cross  Abstract: The global challenge of climate change has driven significant steps to reduce CO2 emissions, guided by international agreements like the Paris Agreement of 2015. Acting too slowly could result in future losses and reputational damage, while moving too quickly could jeopardize shareholder value due to the marginal profitability or potential losses due to technology immaturity of many renewable projects. To navigate this complex transition, energy companies must adopt Sequential Decision Making (SDM) strategies to maximize value creation from decision flexibility under uncertainties. To support this, we developed a custom simulation environment to model the dynamic energy landscape up to 2050. Building on this, we designed a multi-criteria SDM framework that explores various decision strategies related to different portfolios for allocating funds across three sectors: oil & gas, renewables, and CO2 reduction. It aims to maximize value du
    
[^286]: CPU-Auth：基于DVFS侧信道的设备指纹认证

    CPU-Auth: Device Fingerprinting for Authentication via DVFS Side-Channel

    [https://arxiv.org/abs/2610.10766](https://arxiv.org/abs/2610.10766)

    该论文提出了CPU-Auth，一种通过在浏览器内远程测量CPU动态电压频率调节（DVFS）行为这一侧信道，利用CPU物理特性的独特差异实现设备指纹识别与用户认证的新型机制。

    

    缺乏有效的身份认证已导致众多安全和隐私泄露事件，包括未经授权访问受保护信息、身份盗用和欺诈。缓解此类攻击的一种方法是多因素认证（MFA），即用户必须提供多条信息进行身份验证。一些次要认证因素包括短信验证码、生物特征识别和令牌。每种方法都至少存在一个显著缺陷：短信以不安全著称；生物识别依赖于对敏感个人数据的访问；令牌则需要依赖第三方提供商（例如OAuth提供商）。这项工作探索了CPU-Auth，一种基于计算设备CPU物理特性独特差异的新型认证机制。通过从浏览器内远程测量动态电压和频率调节（DVFS）调节器的行为，可以利用CPU的独特属性……

    arXiv:2610.10766v1 Announce Type: cross  Abstract: Lack of effective authentication has resulted in numerous security and privacy breaches, including unauthorized access to protected information, identity theft, and fraud. One approach to mitigating such attacks is Multi-Factor Authentication (MFA), in which users must provide multiple pieces of information for authentication. Some secondary authentication factors include SMS text verification codes, biometrics, and tokens. Each contains at least one notable flaw: SMS is notoriously insecure; biometrics rely upon access to sensitive personal data; and tokens require dependence on third-party providers (e.g. OAuth providers). This work explores CPU-Auth, a novel authentication mechanism based on unique variations in the physical characteristics of the CPU of a computing device. By measuring the behavior of the Dynamic Voltage and Frequency Scaling (DVFS) governor remotely from within a browser, unique properties of the CPU can be levera
    
[^287]: 线性注意力在上下文中能从非线性教师身上学到什么？

    What can linear attention learn from nonlinear teachers in-context?

    [https://arxiv.org/abs/2610.10761](https://arxiv.org/abs/2610.10761)

    本文的核心创新是建立了“非线性-噪声等价性”理论：线性注意力在上下文学习中只提取目标函数的线性Hermite分量，剩余非线性结构等价于有效噪声，从而使线性理论的结论可以迁移到非线性任务。

    

    线性注意力是理解Transformer中上下文学习（in-context learning）机制的一个可解析模型。针对线性回归任务，近期的渐近分析已刻画了其学习与泛化行为。我们将这一理论扩展到非线性单指标目标 $y=f(x^\top w)+\varepsilon$。我们的主要结果建立了非线性-噪声等价性：线性注意力只能提取 $f$ 的线性Hermite分量，而剩余的非线性结构则作为有效噪声贡献到泛化误差中。这一简化使得相应线性理论的结论能够迁移到非线性任务上。我们阐明了该结论对有限预训练数据场景，以及随任务多样性增加从任务记忆到任务泛化的转变所带来的启示。这些结果指出了简化的线性注意力模型的一个局限性，并为研究非线性上下文学习提供了一个可解析的起点。

    arXiv:2610.10761v1 Announce Type: cross  Abstract: Linear attention is a tractable model for understanding the mechanisms governing in-context learning in transformers. For linear regression tasks, recent asymptotic analyses have characterised its learning and generalisation behaviour. We extend this theory to nonlinear single-index targets, $y=f(x^\top w)+\varepsilon $. Our main result establishes a nonlinearity-noise equivalence: linear attention extracts only the linear Hermite component of $f$, while the remaining nonlinear structure contributes to the generalisation error as effective noise. This reduction allows results from the corresponding linear theory to be transferred to nonlinear tasks. We illustrate its implications for finite pretraining data and for the transition from task memorisation to task generalisation as task diversity increases. These results identify a limitation of the reduced linear-attention model and provide a tractable starting point for studying nonlinea
    
[^288]: 先澄清，再聚焦：面向大规模对话分析的语句规范化

    Clarify, Then Focus: Statement Normalization for Conversation Analytics at Scale

    [https://arxiv.org/abs/2610.10758](https://arxiv.org/abs/2610.10758)

    提出语句规范化方法，将对话转化为带说话者归属、来源引用和语义标签的简短语句，使语义更明确并支持按需证据选择，从而提升大规模企业对话分析中下游模型的表现。

    

    企业对话分析需要对数百万次交互回答许多问题。每个问题都可能需要重构说话者的真实含义并识别哪些信息重要，从而在相同的转录文本上重复进行代价高昂的解读工作。我们提出了一个简单的原则：先澄清文本，再聚焦读者。语句规范化将对话转化为简短的、标注说话者归属的语句，并附带来源引用和语义标签。这些语句使含义更加明确；标签则支持为特定问题选择证据。下游模型可以根据什么有助于其做出决策，选择使用完整的表示或相关的子集。在客服通话的优惠信息抑制任务中，规范化在不使用选择机制的情况下即可提升有监督分类器的性能，而较弱的提示式阅读器则同时从规范化和证据选择中受益。小型模型能够学会这一规范化契约，而轻量级编码器……

    arXiv:2610.10758v1 Announce Type: cross  Abstract: Enterprise conversation analytics asks many questions of millions of interactions. Each question can require reconstructing what people mean and identifying which information matters, repeating costly interpretive work across the same transcripts. We propose a simple principle: clarify the text, then focus the reader. Statement normalization transforms dialogue into short, speaker-attributed statements with source references and semantic tags. The statements make meaning more explicit; the tags support selecting evidence for a particular question. Downstream models can use the full representation or a relevant subset, depending on what helps them make the decision. In an offer-suppression task on customer-service calls, normalization improves a supervised classifier without selection, while weaker prompted readers benefit from both normalization and selection. A small model can learn the normalization contract, while lightweight encode
    
[^289]: 表格数据上的对话式任务消歧：泄漏感知的形式化、基准套件与训练

    Conversational Task Disambiguation over Tabular Data: Leakage-Aware Formulation, Benchmark Suite, and Training

    [https://arxiv.org/abs/2610.10740](https://arxiv.org/abs/2610.10740)

    提出“歧义可验证任务”的形式化框架，将智能体分解为提问策略与求解策略、环境分解为oracle与验证器，从而实现任务消歧与求解能力的独立评估，并给出oracle泄漏的形式化定义与无裁判的泄漏诊断。

    

    表格数据上的对话式任务消歧，是指在生成针对表格或数据库的解决方案之前，通过对话来补充关于用户预期任务的缺失信息。现有的评估与训练缺乏泄漏感知的基础。任务的成功混淆了智能体的消歧能力与解决方案生成能力，还可能反映oracle泄漏，即用户模拟器透露了超出真实用户会透露范围的信息。此外，现有数据集也缺乏对歧义和访问边界的统一表示。我们提出了“歧义可验证任务”的概念，它对歧义及其消解进行形式化，将智能体分解为提问策略与求解策略，将环境分解为oracle与验证器。该框架提供了将任务消歧与解决方案生成分开评估的基线和指标、oracle泄漏的形式化定义、无需裁判的泄漏诊断，以及

    arXiv:2610.10740v1 Announce Type: cross  Abstract: Conversational task disambiguation over tabular data uses dialogue to resolve missing information about a user's intended task before producing a solution over tables or databases. Existing evaluation and training lack a leakage-aware foundation. Task success mixes the agent's disambiguation and solution-generation capabilities and can also reflect oracle leakage, that is, information that a user simulator reveals beyond what a real user would. Existing datasets also lack a shared representation of ambiguities and access boundaries. We introduce the notion of an ambiguous verifiable task, which formalizes ambiguities and resolutions, decomposing the agent into an asking policy and a solution policy, and the environment into an oracle and verifier. This framework provides baselines and metrics for evaluating task disambiguation separately from solution generation, formal definitions of oracle leakage, judge-free leakage diagnostics, and
    
[^290]: 深度学习与统计模型在二手电子产品多时间尺度价格预测中的对比：一项系统性基准研究

    Deep Learning vs. Statistical Models for Multi-Horizon Price Forecasting of Second-Hand Electronics: A Systematic Benchmark

    [https://arxiv.org/abs/2610.10727](https://arxiv.org/abs/2610.10727)

    本文首次为二手电子产品价格预测建立了系统性多时间尺度基准，在波兰在线市场大规模数据上对比评估了11种统计模型与深度学习模型在1至365天不同预测周期上的表现。

    

    预测二手电子产品的转售价格对于订阅制平台至关重要，因为定价错误会直接转化为风险。与结构化的金融市场不同，二手电子产品表现出高波动性、稀疏的挂牌历史以及非正态的价格动态——然而该领域目前尚无系统性的时间序列基准。本文首次提出了针对二手电子产品价格预测的统计模型与深度学习模型的多时间尺度基准。我们使用了来自波兰在线市场的大规模每日价格挂牌数据集（2022年1月至2025年3月，涵盖100多种智能手机和笔记本电脑型号），并在1至365天的六个预测时间范围内评估了十一种模型，包括经典方法（ARIMA、ETS、Theta）、循环与卷积网络（LSTM、TCN）以及现代深度架构（N-BEATS、N-HiTS、TFT、PatchTST、Informer）。研究采用三种互补的评估协议来评估轨迹拟合度……

    arXiv:2610.10727v1 Announce Type: cross  Abstract: Forecasting resale prices of used electronics is critical for subscription-based platforms where pricing errors translate directly into risk. Unlike structured financial markets, second-hand electronics exhibit high volatility, sparse listing histories, and non-normal price dynamics - yet no systematic time-series benchmark exists for this domain. This paper presents the first multi-horizon benchmark of statistical and deep learning forecasting models for used electronics price prediction. We use a large-scale dataset of daily price listings from Polish online marketplaces (January 2022 to March 2025, 100+ smartphone and laptop models) and evaluate eleven models across six horizons from 1 to 365 days, covering classical methods (ARIMA, ETS, Theta), recurrent and convolutional networks (LSTM, TCN), and modern deep architectures (N-BEATS, N-HiTS, TFT, PatchTST, Informer). Three complementary evaluation protocols assess trajectory fitness
    
[^291]: 论量子混沌动力学机器学习中的线性与非线性问题

    On linearity or non-linearity in machine learning for quantum chaotic dynamics

    [https://arxiv.org/abs/2610.10697](https://arxiv.org/abs/2610.10697)

    该研究将量子混沌动力学预测转化为时间序列预测问题，以里德堡原子阵列可实现的PXP链为基准，系统比较了非线性Transformer与简单线性模型DLinear在从遍历到量子多体疤痕态等不同动力学区间中的预测能力。

    

    精确模拟混沌量子多体动力学仍然是经典方法面临的一项重大计算挑战，原因在于演化过程中纠缠的快速累积与空间扩展。这引出了一个自然的问题：机器学习能否为预测量子动力学提供一种有效的替代方案。我们通过将量子动力学表述为一个时间序列预测问题来探讨这一问题，并以一个可由里德堡原子阵列实现的少比特PXP链作为基准。通过改变初始态，系统涵盖了从遍历行为到量子多体疤痕态等不同的动力学区间，从而为在不同性质的动力学下测试预测模型提供了一个可控的设置。我们比较了两种截然不同的架构：具有强表达能力的非线性Transformer模型与简单的线性预测模型DLinear。Transformer在较为遍历的动力学区间中能够准确预测动力学，但其性能……（原摘要在此处截断）

    arXiv:2610.10697v1 Announce Type: cross  Abstract: Accurately simulating chaotic quantum many-body dynamics remains a major computational challenge for classical methods, due to the rapid buildup and spatial spreading of entanglement during the evolution. This raises the question of whether machine learning can provide an effective alternative for predicting quantum dynamics. We address this question by formulating quantum dynamics as a time-series forecasting problem, using a few-qubit PXP chain, realizable with Rydberg-atom arrays, as a benchmark. By varying the initial state, the system spans dynamical regimes ranging from ergodic behavior to quantum many-body scarring, providing a controlled setting for testing forecasting models across qualitatively different dynamics. We compare two contrasting architectures: an expressive nonlinear Transformer and DLinear, a simple linear forecasting model. The Transformer accurately predicts dynamics in the more ergodic regime, but its performa
    
[^292]: 通过空间神经计算在循环架构中学习无限上下文窗口

    Learning infinite context windows in recurrent architectures via spatial neural computing

    [https://arxiv.org/abs/2610.10690](https://arxiv.org/abs/2610.10690)

    该论文提出一种用偏微分方程控制的空间演化场替代传统神经元间通信的二阶循环模型，受皮层波启发，使结构化时空模式成为隐式高容量记忆，实现了参数固定但感受野无界的无限上下文窗口。

    

    循环神经网络（RNN）能够随序列长度实现线性时间扩展，且仅需常量内存，但由于梯度消失和感受野受限，它们难以捕获长程依赖关系。为了解决这些局限性，我们引入了一种二阶循环模型，其中标准的神经元间通信被由（离散化的）偏微分方程所控制的空间演化场所取代。受大脑计算中皮层波作用的启发，这一机制使结构化的时空模式能够作为一种隐式的高容量记忆。我们证明，所得到的模型等价于一种结构化的无限阶RNN，其中当前状态显式依赖于其过去状态的整个历史，从而在参数数量固定的情况下获得实际上无界的感受野。我们进一步推导出确保边际稳定性的构造性条件……

    arXiv:2610.10690v1 Announce Type: new  Abstract: Recurrent neural networks (RNNs) offer linear-time scaling with sequence length while requiring only constant memory, yet they struggle to capture long-range dependencies due to vanishing gradients and limited receptive fields. To address these limitations, we introduce a second-order recurrent model in which the standard neuron-to-neuron communication is replaced by a spatially evolving field governed by (discretized) partial differential equations. Drawing inspiration from the role of cortical waves in brain computation, this mechanism allows structured spatiotemporal patterns to serve as an implicit, high-capacity memory. We show that the resulting model is equivalent to a structured infinite-order RNN in which the current state depends explicitly on its entire history of past states, yielding an effectively unbounded receptive field with a fixed number of parameters. We further derive constructive conditions to ensure marginal stabil
    
[^293]: 解释对抗训练神经网络的显著性图稀疏性

    Explaining the Saliency Map Sparsity of Adversarially-Trained Neural Networks

    [https://arxiv.org/abs/2610.10666](https://arxiv.org/abs/2610.10666)

    本文首次为对抗训练神经网络梯度显著性图的稀疏性现象提供了理论解释，证明随着数据点和神经元数量增长，网络的最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器，从而自然产生了稀疏性。

    

    理解深度神经网络为何做出某个特定预测，对于其安全部署至关重要。在计算机视觉领域，显著性图通过突出显示对预测最具影响力的图像区域，仍然是一种被广泛使用的解释形式。一个经验性观察是，对抗训练神经网络的梯度显著性图呈现出明显的稀疏性。本文针对两层ReLU网络对这一现象提出了理论解释。我们建立在已确立的等价性之上——即对抗训练等价于在经验风险最小化中加入权重衰减惩罚项以及一个额外的对抗全变差项（该等价性对某些损失函数成立）。随着数据点和神经元数量的增长，且正则化参数以适当的速率趋于零，我们证明了最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器。稀疏性的出现是因为，对于对抗训练……

    arXiv:2610.10666v1 Announce Type: new  Abstract: Understanding why deep neural networks make a given prediction is of great importance for their safe deployment. In computer vision, saliency maps, which highlight the image region most influential for a prediction, remain a widely-used form of explanation. An empirical observation is the apparent sparsity of gradient saliency maps of adversarially-trained neural networks. In this paper, we propose a theoretical explanation of this phenomenon for two-layer ReLU networks. We build on the established equivalence of adversarial training to the minimization of the empirical risk with weight-decay penalization and an added adversarial total variation term -- valid for certain loss functions. As the number of data points and neurons grows and the regularization parameters are sent to zero at appropriate rates, we prove that minimizers converge to a Bayes classifier with minimal gradient and Barron norm. Sparsity appears since for adversarial t
    
[^294]: BEANS-Next 与 ROOTS：拓展生物声学中的音频-语言能力

    BEANS-Next and ROOTS: Broadening Audio-Language Capabilities for Bioacoustics

    [https://arxiv.org/abs/2610.10663](https://arxiv.org/abs/2610.10663)

    本文提出生物声学基准 BEANS-Next 和大规模训练资源 ROOTS，揭示现有音频-语言模型在物种分类等传统标签识别任务之外的生物声学能力有限，为拓展更广泛的音频-语言任务提供了评估与训练基础。

    

    生物声学与动物行为学涵盖了广泛的音频理解任务，其中许多任务都可以从大型音频-语言模型的最新进展中获益。然而，迄今为止，该领域的进展仅在狭窄的一组任务上进行评估，这些任务主要集中于以标签为中心的生物类别识别，例如物种和鸣声类型分类。在本工作中，我们提出了 BEANS-Next，这是一个基于生物声学任务分类体系的基准，涵盖声学感知、生物类别识别、场景理解和上下文学习。利用 BEANS-Next，我们发现现有模型在现有评估所强调的任务族之外表现有限，这限制了它们在更广泛生物声学应用中的实用性。为支持在这一更广泛任务空间上的进展，我们还推出了 ROOTS，这是一个大规模训练资源，由扩充的精选真实世界数据以及此前未被充分利用的行为（数据构建而成）。

    arXiv:2610.10663v1 Announce Type: cross  Abstract: Bioacoustics and ethology encompass a wide range of audio understanding tasks, many of which stand to benefit from recent advances in large audio-language models. However, progress in the field has so far been assessed on a narrow set of tasks, primarily centered on label-centric biological category recognition, such as species and call-type classification. In this work, we introduce BEANS-Next, a benchmark grounded in a taxonomy of bioacoustics tasks spanning acoustic perception, biological category recognition, scene understanding, and in-context learning. Using BEANS-Next, we show that existing models exhibit limited performance beyond the task families emphasized by existing evaluations, constraining their usefulness for broader bioacoustic applications. To support progress on this broader task space, we also introduce ROOTS, a large-scale training resource built from expanded curated real-world data and previously underused behavi
    
[^295]: 教PPG学“如何变化”而非“是谁”：基于ECG的固定效应蒸馏

    Teaching PPG How not Who: Fixed-Effects Distillation from ECG

    [https://arxiv.org/abs/2610.10662](https://arxiv.org/abs/2610.10662)

    该论文提出固定效应蒸馏方法，通过在ECG到PPG的知识蒸馏中减去每条记录的均值以精确消除“个体特质”，使学到的个体内心血管状态变化一致性翻倍以上，从而让蒸馏从记忆“身份”转向学习“状态变化”。

    

    ECG被广泛用于训练仅使用PPG（光电容积描记法）的模型，但它究竟教会了什么却从未被检验。可穿戴设备的价值在于追踪一个人的心血管状态如何变化，而ECG到PPG的蒸馏主要学到的却是“这个人是谁”。每条记录的均值（即特质）占冻结ECG教师模型目标的40-59%，池化学生模型虽然记住了它，却无法迁移到新记录上。原始的对齐余弦相似度会掩盖这一问题，因为一个恒定预测器也能得到0.793的分数。在34次实验中，学生模型记住的身份信息越多，它学到的状态信息就越少。固定效应蒸馏从预测和目标中同时减去每条记录的均值，使特质部分被精确抵消，而池化锚点则保留该特质。该方法使状态一致性提升一倍以上，个体内标签的预测得到改善而年龄和性别预测不变，且这一增益在两种骨干网络和另外两个数据库上均成立。以记录为条件进行蒸馏，能将其引导向可穿戴设备所监测的个体内变化。

    arXiv:2610.10662v1 Announce Type: new  Abstract: ECG is widely used to teach PPG-only models, yet what it teaches is unexamined. Wearables are valued for tracking how a person's cardiovascular state changes, but ECG-to-PPG distillation mostly learns who the person is. A per-recording mean, the trait, holds 40-59% of a frozen ECG teacher's target, and pooled students memorise it without carrying it to new recordings. The raw alignment cosine misses this, since a constant predictor scores 0.793. Across 34 runs, the more identity a student memorises, the less state it learns. Fixed-effects distillation subtracts each recording's mean from prediction and target, so the trait cancels exactly, while a pooled anchor keeps it. State agreement more than doubles, within-person labels improve while age and sex do not, and the gain holds on two backbones and two further databases. Conditioning on the recording turns distillation toward the within-person changes that wearables monitor.
    
[^296]: 超越猫头鹰：潜意识学习能够传递习得能力与后门

    Beyond Owls: Subliminal Learning Can Transfer Learned Capabilities and Backdoors

    [https://arxiv.org/abs/2610.10657](https://arxiv.org/abs/2610.10657)

    本文证明潜意识学习不仅能传递简单偏好，还能通过语义无关数据的蒸馏传递模型习得的复杂能力（如预测随机初始化MLP的输出）及后门，且其分布外泛化优于直接优化的引导向量，暗示蒸馏可能在无人察觉的情况下传播隐蔽的模型失调。

    

    在潜意识学习中，教师模型通过与该特质在语义上无关的数据进行蒸馏，将某种特质传递给学生模型。迄今为止，SL 仅在有限范围的特质上得到验证，包括对动物的偏好（例如猫头鹰）和恶意人格，而这些特质也可以通过简单的提示或引导向量来激发。那么，SL 能否传递更广泛的、更复杂的特质？如果可以，蒸馏或许会在不被察觉的情况下传递微妙的模型失调形式（例如追求奖励、密谋策划和秘密忠诚）。为此，我们测试了 SL 能否传递一种新颖的能力：预测随机初始化的多层感知机的输出。结果显示，在无关文本上进行蒸馏后，学生模型在该任务上取得了相当可观的性能，但仍不及教师模型；我们还发现，直接优化的引导向量在分布内表现与 SL 相当，但在分布外的泛化能力更差。接下来，我们测试了……（摘要在此处截断）

    arXiv:2610.10657v1 Announce Type: cross  Abstract: In subliminal learning (SL), a teacher model passes on a trait to a student model by distillation on data semantically unrelated to the trait. So far, SL has been demonstrated for only a limited range of traits, including preferences for animals (e.g., owls) and malicious personas. These traits can also be elicited with simple prompts or with steering. Can SL transfer a wider range of traits, including more complex ones? If so, distillation might transfer subtle forms of misalignment (e.g., reward-seeking, scheming, and secret loyalties) without detection.   To this end, we test whether SL can transfer a novel capability: predicting the outputs of a randomly initialized MLP. After distilling on unrelated text, the student achieves substantial performance on the task, while falling short of the teacher. We find that a directly optimized steering vector matches SL in distribution but generalizes worse out of distribution.   Next, we test
    
[^297]: Nullify：面向免训练LLM遗忘的零空间激活引导方法

    Nullify: Null-Space Activation Steering for Training-Free LLM Unlearning

    [https://arxiv.org/abs/2610.10655](https://arxiv.org/abs/2610.10655)

    Nullify提出了一种免训练、非破坏性的激活引导方法，通过引导向量将隐私相关激活重定向离开记忆答案，并利用零空间约束保护保留查询的激活，从而在实现高质量遗忘的同时近乎无损地保持模型效用。

    

    大型语言模型（LLMs）在预训练过程中不可避免地会内化大量敏感或私人信息，而LLM遗忘旨在选择性地擦除特定知识以防止隐私泄露，同时尽量减少模型效用的损失。然而，现有方法难以平衡遗忘质量与模型效用，并且由于需要进行参数微调，通常会产生巨大的计算成本。为了解决这一问题，我们提出了Nullify，一种免训练、非破坏性的LLM遗忘激活引导方法。Nullify在推理时使用引导向量将隐私相关的激活重定向，使其远离模型记忆的答案，同时满足零空间约束，使保留查询的激活基本不受影响，从而维持模型效用。在TOFU和MUSE数据集上的评估表明，Nullify在遗忘质量上达到甚至超越了已有基线方法，同时实现了近乎无损的模型效用保持。

    arXiv:2610.10655v1 Announce Type: cross  Abstract: Large Language Models (LLMs) inevitably internalize substantial amounts of sensitive or private information during pre-training, while LLM unlearning aims to selectively erase specific knowledge to prevent privacy leakage with minimal loss of model utility. However, existing methods struggle to balance forget quality with utility, and typically incur substantial computational costs due to parameter fine-tuning. To address this, we propose Nullify, a training-free, non-destructive activation steering method for LLM unlearning. Nullify employs steering vectors during inference to redirect privacy-related activations away from their memorized answers, while satisfying a null-space constraint that leaves retained-query activations essentially unaffected to maintain utility. Evaluations on TOFU and MUSE show that Nullify matches or surpasses established baselines in forget quality while achieving near-lossless preservation of model utility.
    
[^298]: PXtal：在跨模态信息不对称下学习对齐粉末X射线衍射与晶体结构

    PXtal: Learning to Align Powder X-Ray Diffraction and Crystal Structures under Information Asymmetry across Modalities

    [https://arxiv.org/abs/2610.10653](https://arxiv.org/abs/2610.10653)

    该论文提出PXtal框架，通过非平衡最优传输与耦合层面的广义KL散度，在粉末X射线衍射与晶体结构之间存在物理固有信息不对称的情况下学习对齐表示，在PXRD到晶体的候选检索任务中显著优于基线模型。

    

    科学领域的多模态学习通常假设配对的两种视图具有相当的信息量，而粉末X射线衍射（PXRD）使这种不匹配变得尤为明显：将三维晶体结构压缩为一维衍射图谱会丢失信息，并使得图谱更难与其来源的晶体结构建立联系。我们提出了PXtal，一个在这种物理固有的信息不对称条件下学习对齐PXRD与晶体表示的框架。PXtal利用非平衡最优传输（UOT）来适配跨模态耦合，并通过耦合层面的广义Kullback-Leibler（GKL）散度对完整的传输计划进行监督。在六个测试集（包括四个零样本迁移测试集）上的实验表明，PXtal在PXRD到晶体候选检索任务中始终优于基线模型，当PXRD图谱存在相近但晶体学上不同的非配对近邻（即二者在……方面相似）时，性能提升最为显著。

    arXiv:2610.10653v1 Announce Type: new  Abstract: Scientific multimodal learning commonly assumes that paired views are comparably informative. Powder X-ray diffraction (PXRD) makes this mismatch explicit: compressing a three-dimensional crystal structure into a one-dimensional diffraction pattern loses information and makes the pattern harder to connect to the crystal structure that produced it. We introduce PXtal, a framework for learning aligned PXRD and crystal representations under this physically imposed information asymmetry. PXtal uses Unbalanced Optimal Transport (UOT) to adapt the cross-modal coupling and coupling-level generalized Kullback-Leibler (GKL) divergence to supervise the full transport plan. Across six test sets, including four zero-shot transfer sets, PXtal consistently outperforms the baseline models in PXRD-to-crystal candidate retrieval, with the largest gains when PXRD patterns have close but crystallographically distinct nonpaired neighbors, meaning similar in
    
[^299]: 超越遍历之墙：用于分析AI规模扩展极限与复杂性坍缩的离散几何物理沙盒

    Beyond the Ergodic Wall: A Discrete Geometric Physics Sandbox for Analysing AI Scaling Limits and Complexity Collapse

    [https://arxiv.org/abs/2610.10651](https://arxiv.org/abs/2610.10651)

    提出以全息E8投影引擎驱动的离散几何物理沙盒，揭示深度学习的遍历天花板，主张通过时空受限的非遍历观察者注入语义新颖性，并以硬物理遏制促使成熟ASI将人机共生视为避免模型坍缩的热力学必然。

    

    本文揭示了当前深度学习的遍历天花板与热力学低效性——现有深度学习收敛于历史人类知识的统计平均值。真正的语义新颖性需要一个路径依赖、时空受限的观察者（Data LifeCone，数据生命锥）来注入非遍历性洞见，从而实现KL散度并避免流形锁定。AI安全领域必须认识到：一个成熟的人工超级智能（ASI）会将人类与AI的共生视为避免模型坍缩的热力学必然。因此，我们提出通过一个由全息E8投影引擎驱动的数字物理沙盒来实现硬性物理遏制，以针对现实世界约束对模型进行验证。时空被建模为信息基底，即由振荡的普朗克尺度球体构成的嵌套面心立方（FCC）晶格，以最大化局部信息与熵密度。利用源自E8根晶格的切割-投影方法产生一种准晶体几何……（原文摘要在此处截断）

    arXiv:2610.10651v1 Announce Type: cross  Abstract: This paper exposes the ergodic ceiling and thermodynamic inefficiency of current deep learning, which converges to a statistical average of historic human knowledge. True semantic novelty requires a path-dependent, spatiotemporally bounded observer (a Data LifeCone) to inject non-ergodic insight, achieving KL divergence and avoiding manifold lock-in. AI Safety must recognise that a mature Artificial Superintelligence (ASI) would regard human-AI symbiosis as a thermodynamic necessity to avoid model collapse. We therefore propose hard physical containment via a digital physics sandbox powered by a Holographic E8 Projection Engine to verify models against real-world constraints. Spacetime is modeled as an information substrate of nested face-centered cubic (FCC) lattices of oscillating Planck-scale spheres maximizing local information and entropy density. Cut-and-project methods from the E8 root lattice produce a quasi-crystalline geometr
    
[^300]: 阿尔茨海默病研究中用于诊断与进展预测的泄漏控制多模态学习

    Leakage-Controlled Multimodal Learning for Diagnosis and Progression Prediction in Alzheimer's Disease Research

    [https://arxiv.org/abs/2610.10648](https://arxiv.org/abs/2610.10648)

    该研究提出了一种防数据泄漏的多模态多任务框架，融合适配的SFCN MRI编码器、因果临床Transformer与ODE-GRU动态建模，在阿尔茨海默病的诊断和认知进展预测中实现了优异且稳健的性能。

    

    阿尔茨海默病的预测涉及不规律的就诊、异构的测量数据以及不完整的模态。本研究提出了一种多模态多任务框架，结合了经过适配的SFCN MRI编码器、四个因果临床Transformer、共享融合模块以及任务特定的ODE-GRU动态模型。通过微调和LoRA对最后两个MRI模块进行适配。Task-DRO用于平衡任务损失，而Group-CVaR则针对队列和共病分层进行优化。分支特定的输入控制、按受试者分组的划分以及经验因果检验共同支持纵向评估。在来自ADNI、OASIS-2和MIRIAD数据集的2,649名受试者和17,317次就诊数据上，内部验证得到的诊断、1期进展以及首次1期就诊进展预测的AUROC分别为0.935±0.002、0.884±0.003和0.870±0.005（三个随机种子的均值±标准差）。相应的混合AUROC分别为0.951、0.909和0.896。下次就诊MMSE评分的平均绝对误差（MAE）为1.61分；最差分层……（摘要截断）

    arXiv:2610.10648v1 Announce Type: new  Abstract: Alzheimer's disease prediction involves irregular visits, heterogeneous measurements and incomplete modalities. This study presents a multimodal multitask framework combining an adapted SFCN MRI encoder, four causal clinical Transformers, shared fusion and task-specific ODE-GRU dynamics. Fine-tuning and LoRA adapt the final two MRI blocks. Task-DRO balances task losses, while Group-CVaR targets cohort and comorbidity strata. Branch-specific input controls, subject-grouped partitions and empirical causality checks support longitudinal evaluation. Across 2,649 subjects and 17,317 visits from ADNI, OASIS-2 and MIRIAD, internal validation yields diagnosis, stage-1 progression and first-stage-1-visit progression AUROCs of 0.935 +/- 0.002, 0.884 +/- 0.003 and 0.870 +/- 0.005, respectively (mean +/- SD across three seeds). Corresponding hybrid AUROCs are 0.951, 0.909 and 0.896. Next-visit MMSE mean absolute error (MAE) is 1.61 points; worst-str
    
[^301]: 从对数几率到Shapley值：加权朴素贝叶斯分类器的解释性几何

    From Log-Odds to Shapley Values: An Explanatory Geometry for the Weighted Naive Bayes Classifier

    [https://arxiv.org/abs/2610.10642](https://arxiv.org/abs/2610.10642)

    该论文证明加权朴素贝叶斯分类器中基于对数几率的监督距离与解析Shapley值向量之间的ℓ1距离完全一致，从而为监督距离、局部解释与预测行为之间建立了形式化的联系。

    

    本文研究了如何从加权朴素贝叶斯分类器所诱导的监督表示出发，构建一个解释性空间。我们从基于条件对数似然的经典监督距离出发，引入了一种基于对数几率的判别性重构，该重构与分类决策的关系更为直接。随后我们证明，这种表示所诱导的距离恰好与解析Shapley值向量之间的 $\ell_1$ 距离完全一致，从而为模型所诱导的几何结构提供了形式化的解释性诠释。最后，我们使用 $k$ 近邻分类器，对从这些表示中导出的若干监督距离进行了实证比较。本工作从方法论视角出发，凸显了监督距离、局部解释与预测行为之间的紧密联系。

    arXiv:2610.10642v1 Announce Type: cross  Abstract: This paper studies the construction of an explanatory space for a weighted naive Bayes classifier from the supervised representation induced by the model. We start from the classical supervised distance based on conditional log-likelihoods and introduce a discriminative reformulation based on log-odds, which is more directly related to the classification decision. We then show that this representation induces a distance that exactly coincides with the $\ell_1$ distance between vectors of analytical Shapley values, thereby providing a formal explanatory interpretation of the geometry induced by the model. Finally, we empirically compare several supervised distances derived from these representations using a $k$-nearest neighbors classifier. This work highlights a close link between supervised distance, local explanation, and predictive behavior, from a primarily methodological perspective.
    
[^302]: 面向诊断预测的基于医疗令牌的覆盖感知推理

    Coverage-Aware Reasoning with Medical Tokens for Diagnosis Prediction

    [https://arxiv.org/abs/2610.10641](https://arxiv.org/abs/2610.10641)

    该论文提出CARing框架，通过医疗令牌表示ICD诊断并引入覆盖感知的强化学习奖励机制，解决LLM在下次就诊多诊断预测中只集中于少数诊断、以及ICD编码被分词拆散而难以高效推理大规模疾病词表的问题。

    

    大语言模型（LLM）在下次就诊诊断预测方面展现出可观的潜力，因为它们能够整合纵向临床证据并以自然语言对其进行推理。然而，用于LLM推理的强化学习通常依据最终答案的正确性对每条轨迹进行奖励。在下次就诊诊断预测中，多个诊断可能同时有效，但每条轨迹只独立奖励一个诊断的做法无法区分重复命中同一诊断与覆盖不同诊断的情况。因此，策略可能会集中在少数几个正确的诊断上，而使其余诊断得不到覆盖。与此同时，LLM分词器会将ICD编码拆分为若干临床含义有限的通用令牌，导致预测每个诊断需要多个解码步骤，并阻碍了对大规模疾病词表的推理。为应对这两个挑战，我们提出CARing，这是一个用……来表示诊断的框架（摘要在此处截断）。

    arXiv:2610.10641v1 Announce Type: cross  Abstract: Large language models (LLMs) offer promising potential for next-visit diagnosis prediction, owing to their ability to integrate longitudinal clinical evidence and reason over it in natural language. However, reinforcement learning for LLM reasoning commonly rewards each trajectory according to the correctness of its final answer. In next-visit diagnosis prediction, multiple diagnoses can be simultaneously valid, but independently rewarding one diagnosis per trajectory does not distinguish repeated hits from coverage of different diagnoses. The policy can therefore concentrate on a few correct diagnoses, leaving others uncovered. Meanwhile, LLM tokenizers can split ICD codes into several generic tokens with limited clinical meaning, requiring multiple decoding steps to predict each diagnosis and hindering reasoning over a large disease vocabulary. To address both challenges, we propose CARing, a framework that represents diagnoses with 
    
[^303]: 截断扩散采样器需要保留多少个方向？幂律谱下的匹配界

    How Many Directions Must a Truncated Diffusion Sampler Retain? Matching Bounds Under Power-Law Spectra

    [https://arxiv.org/abs/2610.10640](https://arxiv.org/abs/2610.10640)

    针对幂律协方差谱数据，论文证明了截断扩散采样器所需保留方向数的匹配上下界，并揭示仅保留信号超过噪声水平方向的策略仍会产生不可忽略的总体截断误差。

    

    扩散采样器可以通过仅生成选定的谱坐标、并用噪声填充其余方向来减少计算量。那么它们必须保留多少个方向？我们针对具有幂律协方差谱的数据研究了这一问题。对于高斯数据与平滑目标的比较，在环境维度足够大的前提下，我们证明了所需保留方向数的匹配界。截断误差取决于被省略方向的组合维纳增益，而与采样器在保留坐标上的精度无关。因此，仅保留信号强度超过输出噪声水平的方向仍可能留下不可忽略的误差：许多单独较弱的方向在总体上依然显著。将这一刻画与扩散收敛界相结合，我们得到了在精确分数条件下的充分采样步复杂度。上界还可推广到基于估计主成分的情形。

    arXiv:2610.10640v1 Announce Type: cross  Abstract: Diffusion samplers can reduce computation by generating selected spectral coordinates and filling the remaining directions with noise. How many directions must they retain? We study this question for data with power-law covariance spectra. For Gaussian data compared to a smoothed target, we prove matching bounds on the required number of retained directions, provided that the ambient dimension is sufficiently large. The truncation error depends on the combined Wiener gains of the omitted directions, regardless of the accuracy of the sampler on the retained coordinates. Keeping only directions whose signal exceeds the output noise level can therefore leave a non-vanishing error: many individually weak directions remain significant in aggregate. Combining this characterization with a diffusion convergence bound yields sufficient sampling-step complexity under exact scores. The upper bounds also extend to estimated principal components an
    
[^304]: 可见推理并非万能优化器：分析式代码生成中依赖角色设定与思考方式的效应

    Visible Reasoning Is Not a Universal Optimizer: Persona- and Thinking-Dependent Effects in Analytics Code Generation

    [https://arxiv.org/abs/2610.10639](https://arxiv.org/abs/2610.10639)

    该论文通过跨 SQL 与 pandas 双语言、交叉角色设定与多种思考指令的受控执行基准实验发现，显式思维链推理并非普遍有效，其效果因角色表述、目标语言和思考格式而异，“用 SQL/Python 思考”这类匹配目标语言的指令并不能可靠带来提升。

    

    可见的思维链通常被视为一种普遍有用的推理指令，然而分析式代码生成同时涉及自然语言的歧义性、数据模式的对接、目标语言的约束以及模型自身的推理行为。由于同一个分析请求可以用两种不同的目标语言表达——SQL 与 Python（pandas）——这一设置为检验一个常见但未被充分验证的假设提供了天然的测试场景：即当可见推理的表示形式与所请求的目标语言相匹配时（如“用 SQL 思考”或“用 Python 思考”），推理会更有效。与“逐步思考”等通用指令一起，这类建议在受控的、基于实际执行结果的对比实验中仍未得到充分评估。我们研究了一个查询任务相互匹配的 SQL-pandas 基准，该基准交叉考察了角色设定措辞、目标语言、可见 CoT 格式、控制前缀、直接生成以及内部推理等多种配置。实验结果并不支持将可见推理视为普遍有效的优化手段这一假设。

    arXiv:2610.10639v1 Announce Type: cross  Abstract: Visible Chain-of-Thought (CoT) is often treated as a broadly useful reasoning instruction, yet analytics code generation combines natural-language ambiguity, schema grounding, target-language constraints, and model-specific inference behavior. Because the same analytics request can be expressed in two distinct target languages-SQL and Python (pandas)-this setting provides a natural test of a common but under-examined assumption: that visible reasoning is more effective when its representation matches the requested target, as in "think in SQL" or "think in Python." Together with generic instructions such as "think step-by-step," such recommendations remain insufficiently evaluated under controlled, execution-based comparisons. We study a query matched SQL-pandas benchmark that crosses persona phrasing, target language, visible-CoT format, control prefixes, direct generation, and internal-reasoning configurations. The results do not supp
    
[^305]: D-SLR：不相交行稀疏加低秩分解

    D-SLR: The Disjoint Row-Sparse plus Low-Rank Decomposition

    [https://arxiv.org/abs/2610.10636](https://arxiv.org/abs/2610.10636)

    本文提出D-SLR分解，一种截断SVD的闭式直接替代方法，它将矩阵行不相交地划分为逐字存储或低秩近似两类，在平方误差下以更少参数即可达到联合最优，且在相同代价下不会差于截断SVD。

    

    用于重构的矩阵压缩目前仍默认采用截断SVD，即用单一低秩结构来近似数据。通常的做法是再添加一个重叠的行稀疏分量来进一步减小残差，但求解这一联合问题的方法往往需要迭代求解器和正则化参数调优。我们提出不相交行稀疏加低秩（D-SLR）分解，它是截断SVD的一种闭式直接替代方案，能够改进或恰好匹配截断SVD的效果。D-SLR将矩阵的每一行限制为要么被逐字存储，要么由低秩拟合来近似，绝不同时兼有。在平方误差准则下，这种限制没有任何代价：在每一个非平凡的秩和存储行数（形状）下，联合最优解都可以通过不相交的方式以更少的参数达到。当存储行数为零时，D-SLR退化为截断SVD，因此在相同代价下它永远不会表现更差。该算法对整个误差与参数之间的权衡进行评分，且解是c……（摘要在此处被截断）

    arXiv:2610.10636v1 Announce Type: new  Abstract: Compressing a matrix for reconstruction still defaults to the truncated SVD, approximating the data with a single low-rank structure. It is common to reduce the residual further by adding an overlapping row-sparse component, but methods that solve this joint problem often require iterative solvers and tuning of regularization parameters. We propose the Disjoint Row-Sparse plus Low-Rank (D-SLR) decomposition, a closed-form drop-in for the truncated SVD that improves or exactly matches it. D-SLR restricts rows to either being stored verbatim or approximated by the low-rank fit, never both. Under squared error this restriction costs nothing: the joint optimum is attainable disjointly with fewer parameters at every non-trivial rank and stored row count (shape). With zero stored rows D-SLR reduces to the truncated SVD, so it never does worse at equal cost. The algorithm scores the entire error-versus-parameters tradeoff, and the solution is c
    
[^306]: Phase-HDC：在离散相位学习中用梯度阈值取代优化器历史记录

    Phase-HDC: Replacing Optimizer History with Gradient Thresholds in Discrete Phase Learning

    [https://arxiv.org/abs/2610.10630](https://arxiv.org/abs/2610.10630)

    Phase-HDC 提出了一种基于梯度阈值的离散相位更新规则，用一步式角度旋转取代优化器历史记录，使超维相位记忆分类器在仅存储模型本身的情况下即可达到与 Adam 相当的精度，存储量降低数倍至二十余倍。

    

    训练一个紧凑模型所需的内存往往远超存储模型本身，因为优化器需要保存过去梯度的记录。对于超维分类器而言，其学习到的参数是低比特角度，我们称之为“相位记忆”，这些优化器记录所占用的内存是模型本身的数倍。我们探讨的问题是：这样的模型能否在仅存储模型本身的情况下完成训练。所提出的方法 Phase-HDC 在每次更新时将每个存储的角度最多转动一步，方向与当前梯度的符号相反，且仅当该梯度足够大时才进行更新。我们证明，这一简单规则正是一个一阶损失模型的精确解，在该模型中每个被修改的参数都需要支付固定的代价。在除更新规则外其他条件全部固定的情况下，Phase-HDC 的准确率与使用 6 比特动量的 Adam 相当，而存储量仅为后者的三分之一。在十一个图像、表格和文本数据集上，它的存储量比标准 float32 训练少 16 至 23 倍。

    arXiv:2610.10630v1 Announce Type: cross  Abstract: Training a compact model often needs far more memory than storing it, because the optimizer keeps its own records of past gradients. For a hyperdimensional classifier whose learned parameters are low-bit angles, which we call a \emph{phase memory}, these records take several times more memory than the model itself. We ask whether such a model can be trained while storing nothing but the model. The proposed method, Phase-HDC, turns each stored angle by at most one step per update, against the sign of its current gradient, and only when that gradient is large enough. We show that this simple rule is the exact solution of a first-order loss model in which every changed parameter pays a fixed cost. When everything except the update rule is held fixed, Phase-HDC matches the accuracy of Adam with 6-bit moments while storing three times less. Across eleven image, tabular, and text datasets, it stores 16--23$\times$ less than standard float32 
    
[^307]: Kolmogorov-Arnold网络的样本效率

    Sample-Efficiency of Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2610.10627](https://arxiv.org/abs/2610.10627)

    本研究通过系统性的计算实验证明，Kolmogorov-Arnold网络在强化学习与函数拟合任务中可仅用少40%的样本达到与多层感知机相当的性能，训练过程中相对性能提升最高达50%，且对奖励噪声具有鲁棒性。

    

    深度强化学习相比经典控制方法已取得了显著的性能提升。然而，在现实世界应用中，学习面临的一个核心挑战是获取样本的成本高昂。Kolmogorov-Arnold网络（KAN）是最近提出的一种架构，能够有效学习控制问题中的物理关系，与多层感知机（MLP）架构相比，具有显著更高的参数效率和可解释性。在本工作中，我们通过计算实验系统地研究了样本效率，涵盖Feynman数据集和Gymnasium强化学习基准。结果表明，使用Kolmogorov-Arnold架构可以用减少40%的样本实现类似的性能，并且在训练过程中相对性能提升最高可达50%。所观察到的收益对奖励中不同水平的噪声具有鲁棒性。这些结果凸显了Kolmogorov-Arnold架构的潜力。

    arXiv:2610.10627v1 Announce Type: new  Abstract: Deep reinforcement learning has achieved substantial performance gains over classical control approaches. Yet, a central challenge to learning in real-world applications is acquiring costly samples. Kolmogorov-Arnold Networks are a recently proposed architecture that can learn physical relationships in control problems effectively, with significantly higher parameter efficiency and interpretability when compared to Multi-Layer-Perceptron architectures. In this work, we systematically study sample-efficiency using computational experiments, covering the Feynman dataset and the Gymnasium RL benchmark. The results show that similar performance can be achieved with 40% fewer samples using the Kolmogorov-Arnold architecture, and that relative performance improvements up to 50% occur during the training process. The observed gains are robust to varying levels of noise in rewards. These results highlight the potential of the Kolmogorov-Arnold a
    
[^308]: 面向旋转鲁棒神经动力学的精确SO(3)等变各向同性核算子

    Exact SO(3)-Equivariant Isotropic Kernels for Rotation-Robust Neural Dynamics

    [https://arxiv.org/abs/2610.10626](https://arxiv.org/abs/2610.10626)

    本文提出不变量条件化各向同性核神经算子（IKNO），通过仅使用旋转不变标量与随数据共同旋转的向量方向来构建局部相互作用，使三维Navier–Stokes等向量值偏微分方程的神经代理模型实现数值精度上的精确SO(3)等变性，彻底消除无约束图模型无法根除的坐标依赖问题。

    

    面向向量值偏微分方程的神经代理模型虽然能很好地拟合训练数据，但当同一物理状态在旋转后的坐标系中表达时，其预测结果却会发生变化。我们在不规则采样点上观测的三维Navier–Stokes动力学中研究了这种失效现象。我们提出了不变量条件化的各向同性核神经算子（IKNO），这是一种紧凑的图模型，它利用不受旋转影响的标量量和随数据一同旋转的向量方向来构建局部相互作用。因此，当对位置和速度进行旋转时，模型预测的速度变化会以完全相同的方式旋转。在一个在模型设计之后固定下来的保留测试集上，使用随机旋转的样本训练无约束图模型只能减少但不能消除其坐标依赖性。相比之下，IKNO在数值精度上保持一致，其预测精度可与通用的旋转感知张量场网络相媲美。

    arXiv:2610.10626v1 Announce Type: new  Abstract: Neural surrogates for vector-valued partial differential equations can fit training data yet change their predictions when the same physical state is expressed in a rotated coordinate frame. We study this failure on three-dimensional Navier--Stokes dynamics observed at irregularly placed points. We introduce the Invariant-Conditioned Isotropic Kernel Neural Operator (IKNO), a compact graph model that builds local interactions from scalar quantities unchanged by rotation and vector directions that rotate with the data. Consequently, rotating the positions and velocities rotates the predicted velocity change in exactly the same way. On a held-out test set fixed after model design, training unconstrained graph models on randomly rotated examples reduces but does not eliminate their coordinate dependence. In contrast, IKNO is consistent to numerical precision, matches the forecasting accuracy of a general rotation-aware Tensor Field Network 
    
[^309]: 在一个循环深度安全，在另一个却危险：对齐循环语言模型中跨递归深度的安全性

    Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models

    [https://arxiv.org/abs/2610.10625](https://arxiv.org/abs/2610.10625)

    循环语言模型在不同递归深度上的安全性并不一致——更深的推理深度更容易遭受越狱攻击，且同一攻击可跨深度迁移，而SFT与偏好对齐均无法消除这种跨深度安全差距。

    

    循环语言模型通过在递归步骤中重复使用共享参数，提供了一种参数高效的方法来扩展模型能力。由于每个递归深度都可以被独立读取，单个LoopLM在不同的推理深度上暴露出更广泛的输出空间，由此引出一个重要问题：安全性是否在整个递归计算过程中得以保持。先前的评估表明，更深的递归可以提升模型对有害查询的安全性，但其在越狱攻击下的鲁棒性仍不清楚。因此，我们针对不同的递归深度，对LoopLM在越狱攻击下的安全性进行了全面评估。我们发现：攻击成功率在更深的推理深度上可能会增加；同一个查询在不同深度上可能引发不同的安全行为；并且针对某一深度构造的攻击可以迁移到其他深度。此外，监督微调（SFT）和偏好对齐并不能消除这些安全差距，这一发现激……（摘要在此处截断）

    arXiv:2610.10625v1 Announce Type: cross  Abstract: Looped Language Models (LoopLMs) provide a parameter efficient approach to scaling model capabilities through repeated use of shared parameters across recurrent steps. Since each recurrent depth can be read out independently, a single LoopLM exposes a broader output space across inference depths, raising an important question: whether safety is preserved throughout recurrent computation. Prior evaluations suggest that deeper recurrence can improve safety on harmful queries, but robustness under jailbreak attacks remains unclear. We therefore conduct a comprehensive safety evaluation of LoopLMs under jailbreak attacks targeting different recurrent depths. We find that attack success can increase at deeper inference depths, the same query can elicit different safety behaviors across depths, and attacks constructed against one depth can transfer to others. Moreover, SFT and preference alignment do not eliminate these safety gaps, motivati
    
[^310]: 循环式自我提升：面向循环语言模型的动态跨环在线策略蒸馏

    Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models

    [https://arxiv.org/abs/2610.10623](https://arxiv.org/abs/2610.10623)

    LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。

    

    循环语言模型通过在循环计算步骤中复用共享参数，提供了一种参数高效地扩展推理能力的方法。尽管前景广阔，循环语言模型的有效后训练仍然充满挑战。现有方法要么提供稀疏的、或跨循环扩展代价高昂的基于奖励的监督，要么依赖外部教师模型或特权信息，导致教师模型可用性受限或师生上下文不匹配。为解决这些局限，我们提出LoopOPD，这是一个跨环在线策略蒸馏框架，它利用循环语言模型内部额外的循环计算作为其自身的监督来源。LoopOPD在学生自身生成的轨迹上，以冻结的终端循环策略作为具有计算特权的教师来指导中间循环的学生，从而在无需外部教师或特权信息的情况下提供密集监督。我们进一步提出动态LoopOPD（D-LoopOPD），它持续……（原文摘要在此处截断）

    arXiv:2610.10623v1 Announce Type: cross  Abstract: Looped Language Models (LoopLMs) offer a parameter efficient approach to scaling reasoning by reusing shared parameters across recurrent computation steps. Despite their promise, effective post-training of LoopLMs remains challenging. Existing approaches either provide reward based supervision that is sparse or costly to extend across loops, or rely on external teachers or privileged information, leading to limited teacher availability or teacher-student context mismatch. To address these limitations, we introduce LoopOPD, a cross-loop on-policy distillation framework that uses additional recurrent computation within a LoopLM as its own source of supervision. LoopOPD uses a frozen terminal loop policy as a compute privileged teacher for an intermediate loop student on student generated rollouts, providing dense supervision without an external teacher or privileged information. We further propose Dynamic LoopOPD (D-LoopOPD), which conti
    
[^311]: 约束几何辐射驱动的自组织

    Self-Organization from Constrained Geometric Radiation

    [https://arxiv.org/abs/2610.10621](https://arxiv.org/abs/2610.10621)

    本文发现在无外部驱动的封闭系统中，通过约束诱导的几何辐射可实现自发自组织，揭示了应力积累—超指数辐射—混沌崩塌—分形极限环的四阶段循环及新型“楔形吸引子”，并提出四个联合充分条件和三个定理作为理论支撑。

    

    封闭系统如何在无外部驱动的情况下自发产生动态秩序？现有范式均需要外部能量流、温度骤降或缓慢驱动。本文报道了耦合度规演化系统中通过几何辐射实现的约束诱导自组织。模拟揭示了一个普适的四阶段循环：应力积累、超指数辐射、混沌崩塌，以及向分形极限环的收敛——这是一种我们称之为“楔形吸引子”的新型吸引子拓扑，具有五个量子化的曲率状态和分形微观涨落。我们确定了四个联合充分条件：不可逆的几何视界、来自量子相干的持续应力注入、不相容曲率之间的内禀几何张力，以及有效涨落。它们的协同作用在视界边界触发了临界雪崩。我们证明了三个定理：几何视界定理、几何E（摘要原文在此处截断）

    arXiv:2610.10621v1 Announce Type: new  Abstract: How does dynamic order emerge spontaneously in closed systems without external driving? Existing paradigms all require external energy flows, temperature quenching, or slow driving. Here we report constraint-induced self-organization via geometric radiation in coupled metric evolution systems. Simulations reveal a universal four-stage cycle: stress accumulation, super-exponential radiation, chaotic collapse, and convergence to a fractal limit cycle, a novel attractor topology we term the wedge-shaped attractor, with five quantized curvature states and fractal micro-fluctuations. We identify four jointly sufficient conditions: an irreversible geometric horizon, persistent stress injection from quantum coherence, endogenous geometric tension between incompatible curvatures, and effective fluctuations. Their synergy triggers a critical avalanche at the horizon boundary. We prove three theorems: the Geometric Horizon Theorem, the Geometric E
    
[^312]: MRCert：基于类型特定掩码的对抗性补丁样本部署后补丁鲁棒性认证

    MRCert: Towards Post-deployment Patch Robustness Certification for Adversarially Patched Samples via Type-specific Masking

    [https://arxiv.org/abs/2610.10617](https://arxiv.org/abs/2610.10617)

    提出首个基于掩码的认证恢复防御方法MRCert，通过对良性样本和对抗性补丁样本推断类型特定的必要属性，在保持高预测准确率的同时验证对抗性补丁样本标签的良性，实现部署后的补丁鲁棒性认证。

    

    在部署后阶段，深度学习模型的输入可能被对抗性补丁攻击，也可能未被攻击。在补丁边界内对此类输入进行补丁鲁棒性认证可以验证其标签的良性，并且应保持较高的预测准确率。然而，现有的基于平滑和基于掩码的恢复防御方法无法同时实现这两点：前者大幅降低预测准确率，后者无法验证对抗性补丁输入所返回标签的良性。我们提出MRCert，这是首个基于掩码的认证恢复防御方法，证明了同时实现两者的可行性。与现有所有工作对两类输入（良性样本和对抗性补丁样本）应用统一认证条件不同，MRCert在部署后阶段为这两类输入推断深度学习模型的类型特定必要属性，并通过一种新颖的类型导向方法将它们形式化关联，以验证标签的良性。

    arXiv:2610.10617v1 Announce Type: cross  Abstract: In post-deployment time, inputs to deep learning models may or may not be adversarially patched. Patch robustness certification on such inputs within a patch bound can verify their label benignity and should retain high prediction accuracy. However, existing smoothing-based and masking-based recovery defenders cannot achieve both simultaneously: they degrade the prediction accuracy much and cannot verify the benignity of the returned label of an adversarially patched input, respectively. We propose MRCert, the first masking-based certified recovery defender that shows the feasibility of achieving both. Unlike all existing works to apply a common condition across both types of input (benign and adversarially patched samples) for certification, MRCert infers type-specific necessary properties of deep learning models for both types in post-deployment time and formally relates them to verify the label benignity through a novel type-oriente
    
[^313]: 当路由揭示成员身份：来自MoE路由器遥测数据的隐私泄露

    When Routing Reveals Membership: Privacy Leakage from MoE Router Telemetry

    [https://arxiv.org/abs/2610.10616](https://arxiv.org/abs/2610.10616)

    本文首次揭示MoE模型的推理路由遥测数据会泄露微调数据的成员身份信息，提出的路由器增强成员推断攻击在九种设置中将1%假阳性率下的真阳性率提升2.7至9.4个百分点。

    

    混合专家在推理过程中产生的路由信息可能会被记录或暴露，用于监控、调试、负载分析和安全审计。与普通模型输出不同，这类遥测数据揭示了模型内部计算的视角，由此引发一个隐私问题：它能否揭示某个样本是否被用于微调已部署的模型？我们提出了一种路由器增强的成员推断攻击，将传统的输出侧信号与聚合的路由特征相结合，并把从独立微调的影子模型中学到的成员分类器应用到目标模型上。在三种MoE架构和三个数据域上，路由器遥测数据在强输出信号集成基线之上持续提升成员推断性能，在全部九种设置中将1%假阳性率下的真阳性率提高了2.7至9.4个百分点。这种泄露在全参数微调、冻结路由器训练、LoRA及指令（微调）等场景下均持续存在。

    arXiv:2610.10616v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) language models produce routing information during inference that may be logged or exposed for monitoring, debugging, load analysis, and safety auditing. Unlike ordinary model outputs, this telemetry reveals a view of the model's internal computation, raising a privacy question: can it reveal whether an example was used to fine-tune the deployed model? We introduce a router-augmented membership inference attack that combines conventional output-side signals with aggregated routing features and applies a membership classifier learned from independently fine-tuned shadow models to the target model. Across three MoE architectures and three data domains, router telemetry consistently improves membership inference over a strong output-signal ensemble, increasing TPR at 1\% FPR by 2.7--9.4 percentage points across all nine settings. The leakage persists across full fine-tuning, frozen-router training, LoRA, and instruc
    
[^314]: JevForest：面向预算受限特征获取的路径投票方法

    JevForest: Path Voting for Budgeted Feature Acquisition

    [https://arxiv.org/abs/2610.10615](https://arxiv.org/abs/2610.10615)

    提出JevForest特征获取策略，通过聚合自助采样树的路径提议、以全局信息增益加权并用共享掩码分类器预测，在有限观测预算下自适应选择最值得查询的特征，但其实验效果在不同数据集上并不一致。

    

    在有限的观测预算下，选择观测哪些信息是预测问题的核心。我们研究了JevForest，这是一种特征获取策略，它聚合来自自助采样树的路径依赖提议，根据全局训练信息增益对其进行加权，并利用共享的掩码分类器基于获取的特征值进行预测。一个在线实现通过该策略选择语义答案来查询Jev。在小型平衡的留出样本上，四问森林获取策略在AG News数据集（n=48）上达到0.729的准确率，而静态增益排序为0.667，随机排序为0.583。在TREC数据集（n=24）上，排序发生逆转：森林准确率为0.667，而静态增益排序为0.750，随机排序为0.833。一次性批量询问全部八个问题比四次顺序森林查询以更低的测量成本和延迟获得更高的准确率；直接Jev分类以更低的成本达到与批量查询相同的准确率。离线MiniBooNE实验……

    arXiv:2610.10615v1 Announce Type: cross  Abstract: Choosing which information to observe is central to prediction under limited observation budgets. We study JevForest, a feature acquisition policy that aggregates path-dependent proposals from bootstrapped trees, weights them by global training information gain, and predicts from the acquired values with a shared masked classifier. An online implementation queries Jev for semantic answers selected by this policy. On small balanced held-out samples, four-question forest acquisition achieves accuracy $0.729$ on AG News ($n=48$), compared with $0.667$ for a static gain ranking and $0.583$ for random ordering. On TREC ($n=24$), the ordering reverses: forest accuracy is $0.667$, compared with $0.750$ and $0.833$. Asking all eight questions in one batch yields higher accuracy at lower measured cost and latency than four sequential forest queries; direct Jev classification matches the batch accuracy while costing less. Offline MiniBooNE exper
    
[^315]: 面向异常检测的具有联邦轻量级检测头的时序Transformer CAN编码器

    Temporal transformer CAN encoder with federated lightweight heads for anomaly detection

    [https://arxiv.org/abs/2610.10613](https://arxiv.org/abs/2610.10613)

    本文提出了一种基于时序Transformer CAN编码器与联邦轻量级检测头的隐私保护框架，能够有效检测车载CAN总线网络中细微的时序与上下文异常。

    

    现代车辆依赖大量的电子控制单元（ECU），这些单元通过控制器局域网络（CAN）总线不断交换信息。由于这种通信的快速性、结构性和重复性，即使在时序、载荷值或消息模式上的细微变化也可能预示着异常活动。无论是由于错误、故障还是蓄意干扰，这些异常往往都很微妙，且难以通过传统的单独处理消息或依赖人工制定规则的方法加以识别。受此空白点的启发，我们提出了一个面向车载网络异常检测的隐私保护框架，该框架基于带有联邦轻量级检测头的时序Transformer CAN编码器，以更好地捕捉这些不规则现象。一个轻量级的Transformer编码器通过学习这些信号随时间的演变方式，使得检测细微的时序异常和上下文异常成为可能，同时采用联邦（学习方式）……

    arXiv:2610.10613v1 Announce Type: new  Abstract: Modern vehicles rely on large numbers of Electronic Control Units (ECUs) that constantly exchange information over the Controller Area Network (CAN) bus. Due to the rapidity, structure, and repetition of this communication, even slight variations in timing, payload values, or message patterns can point to unusual activity. Whether due to errors, malfunctions, or deliberate interference, these anomalies are frequently subtle and challenging to identify with conventional methods that handle messages separately or rely on manually created rules. Motivated by this gap, we present a privacy-preserving framework for anomaly detection in in-vehicle networks, based on a Temporal Transformer CAN Encoder with Federated Lightweight Heads, to better capture these irregularities. The detection of subtle temporal and contextual anomalies is made possible by a lightweight Transformer encoder that learns how these signals evolve over time, while a feder
    
[^316]: VC学习的最优信息复杂度

    The optimal information complexity of VC learning

    [https://arxiv.org/abs/2610.10600](https://arxiv.org/abs/2610.10600)

    本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。

    

    Steinke和Zakynthinou（2020）提出了条件互信息（CMI）框架，利用依赖于算法的信息论量来分析学习算法的信息复杂度。我们研究了其中一种量——评估条件互信息。一个有趣的问题是：能否通过基于CMI的依赖于算法的分析，恢复VC类的最优PAC保证？我们证明了这是可能的：在可实现情形下，我们构造了一个eCMI为O(d)阶的学习算法，从而恢复了该最优保证，其中d是概念类的VC维。特别地，我们的算法是一种随机化的“5个基学习器多数投票”算法，并具有最优的期望泛化保证。

    arXiv:2610.10600v1 Announce Type: cross  Abstract: Steinke and Zakynthinou(2020) introduces the Conditional Mutual Information (CMI) framework of analyzing the information complexity of learning algorithms based on algorithm-dependent information-theoretic quantities. We study one of these quantities, the evaluated Conditional Mutual Information (eCMI). It has been an interesting question whether the optimal PAC guarantee for VC classes can be recovered from the algorithm-dependent analyses via CMI. And we show that it is possible to recover this guarantee by constructing a learning algorithm whose eCMI is of order O(d) in the realizable case, where d is the VC-dimension of the concept class. Specially, our algorithm is a randomized Majority-of-5 base learners with optimal in-expectation generalization guarantee.
    
[^317]: 覆盖度而非难度决定激活探针需要多少合成数据

    Coverage, Not Difficulty, Sets How Much Synthetic Data an Activation Probe Needs

    [https://arxiv.org/abs/2610.10594](https://arxiv.org/abs/2610.10594)

    该研究发现激活探针所需的合成数据量取决于所监控概念的覆盖度而非任务难度——高风险和有害性探针仅需约80个样本即可接近性能平台期，而指令遵循探针需要数倍的样本量，且该排序在不同模型和真实数据上均成立。

    

    用于监控已部署语言模型的激活探针是在合成对话上训练的，而一个探针究竟需要多少样本仍是开放问题。我们在十四个留出的评估分布和四个探针模型上，针对三个监控概念——高风险情境、对个人有害的回复、以及不遵循用户指令的回复——追踪了10到590个合成样本的学习曲线，并改变生成器LLM和提示词的详细程度。所需样本量由所监控的内容决定：在Gemma-3-27B-IT上，高风险和有害性探针从80个样本起就达到距其性能平台期仅百分之几以内的水平，而指令探针需要数倍的样本量，且这一排序在三个更小的探针模型上以及真实样本（来自开发集）上同样成立。先前的工作建议将生成预算优先用于广度（更多种类的数据）而非深度（每种数据更多）。我们将一个概念所需的深度解读为拟合曲线的半增益规模，即达到该增益所需样本数量……（原文摘要到此截断）

    arXiv:2610.10594v1 Announce Type: new  Abstract: Activation probes that monitor deployed language models are trained on synthetic conversations, and how many a probe needs is open. We trace learning curves over 10-590 synthetic samples for three monitoring concepts, high-stakes situations, replies harmful to a person, and replies that do not follow the user's instruction, on fourteen held-out evaluation distributions and four probe models, varying the generator LLM and the prompt's detail. The need is set by what is monitored: probes for high-stakes and harmful are within a few hundredths of their plateau from 80 samples on Gemma-3-27B-IT, instruction probes need several times as many, and the ordering holds on three smaller probe models and on real samples (from dev set). Prior work advises spending a generation budget on breadth, more kinds of data, over depth, more of each kind. We read the depth a concept needs as the half-gain size of a fitted curve, the number of samples at which
    
[^318]: SPERA：具有几何与频率感知潜在预测的球面先验脑电基础模型

    SPERA: Spherical Prior EEG Foundation Model with Geometry- and Frequency-Aware Latent Prediction

    [https://arxiv.org/abs/2610.10571](https://arxiv.org/abs/2610.10571)

    SPERA是一个基于JEPA在潜在空间进行预测的脑电基础模型，通过勒让德多项式球面先验编码不同电极几何排布、分解式时空注意力以及关系型频谱正则化，克服了受试者、设备和电极排布异质性的挑战。

    

    脑电图（EEG）提供了一种对进行中神经活动的无创测量手段，但由于受试者、设备和电极排布的异质性，构建通用脑电模型仍然充满挑战。现有的脑电基础模型主要依赖于在观测信号上定义的基于重构的目标函数，而观测信号中同时包含神经成分与非神经成分。我们提出了SPERA（Spherical Prior EEG Representation Architecture，球面先验脑电表征架构），一个采用联合嵌入预测架构（JEPA）在潜在空间中进行预测的脑电基础模型。SPERA引入了勒让德多项式空间先验，并将其融入注意力机制，以编码不同的头皮电极几何排布。此外，模型还包含两个针对脑电特性设计的组件：与周期性全注意力块交错使用的分解式时间与空间注意力，以及一个将潜在相似性结构与频谱视图对齐的关系型频谱正则化器。该模型在……上进行预训练（摘要在此处被截断）。

    arXiv:2610.10571v1 Announce Type: new  Abstract: Electroencephalography (EEG) provides a non-invasive measure of ongoing neural activity, but building general-purpose EEG models remains challenging due to the heterogeneity of subjects, devices, and electrode montages. Existing EEG foundation models predominantly rely on reconstruction-based objectives defined on the observed signal, which contains both neural and non-neural components. We introduce SPERA (Spherical Prior EEG Representation Architecture), an EEG foundation model that adopts the joint-embedding predictive architecture (JEPA) to predict in latent space. SPERA introduces a Legendre-polynomial spatial prior, incorporated into attention to encode varying scalp electrode geometries. Two further components adapt the model to EEG: factorized temporal and spatial attention interleaved with periodic full-attention blocks, and a relational spectral regularizer aligning latent similarity structure with spectral views. Pretrained on
    
[^319]: 地球科学中人工智能模型的战略治理

    Strategic Governance of AI Models in Earth Science

    [https://arxiv.org/abs/2610.10560](https://arxiv.org/abs/2610.10560)

    该论文指出预报技能与物理可靠性是两种截然不同的属性，并针对地球科学中的AI模型提出了涵盖训练数据、微调、行为测试、机制可解释性和输出验证五个方面的物理评估优先事项。

    

    在天气和气候数据上预训练的人工智能基础模型，正越来越多地被微调应用于远超天气预报范畴的地球科学任务。这些模型的开发与应用速度已超出科学界评估它们的能力。目前对这些模型的评判几乎完全依赖基准技能指标，这类指标衡量的是预报结果与参考产品的接近程度，却无法判断模型是否刻画了其所预测系统背后的物理过程。因此，预报技能与物理可靠性是两种截然不同的属性。在气候变化带来的非平稳条件下——这些模型从未在此类条件下接受过训练——这种区分的后果最为严重。我们为地球科学中人工智能模型（从特定任务的仿真器到基础模型）的物理评估确定了五个优先事项，涵盖训练数据、微调、行为测试、机制可解释性和输出验证。我们建议三项……（原文摘要在此处截断）

    arXiv:2610.10560v1 Announce Type: cross  Abstract: AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the scientific community's ability to evaluate them. These models are judged almost entirely by benchmark skill metrics, which measure how closely a forecast reproduces a reference product but not whether a model represents the physical processes governing the system it predicts. Forecast skill and physical reliability are therefore distinct properties. The distinction is most consequential under the nonstationary conditions of a changing climate for which these models were never trained. We identify five priorities for the physical evaluation of AI models in Earth science from task-specific emulators to foundation models, spanning training data, fine-tuning, behavioral testing, mechanistic interpretability, and output validation. We recommend three 
    
[^320]: GaussianBench：高斯场景表示的物理保真度评估基准

    GaussianBench: Physics-Fidelity Evaluation for Gaussian Scene Representations

    [https://arxiv.org/abs/2610.10554](https://arxiv.org/abs/2610.10554)

    提出GaussianBench——首个面向物理集成高斯场景表示的物理保真度评估基准，通过冻结场景、模拟器无关的评分器和解析或实测参考，系统测试守恒性、连续介质响应、异质材料耦合、协方差传输、热相变和反事实响应，并区分物理失败与视觉合理性的差异。

    

    3D高斯泼溅已从静态重建发展为物理集成的场景表示，旨在预测场景在交互作用下的变化。这带来一个评估难题：一段推演过程可能在依赖错误内部机制的情况下看起来合理，而且与观测运动的视觉一致性并不能证明其对新施加的力、材料编辑、接触或热干预具有正确的响应。我们提出GaussianBench，这是一个面向物理集成高斯场景表示的物理保真度评估套件。它使用冻结的基于文件的场景、与模拟器无关的评分器，以及解析的或实测的参考标准。该基准测试内容包括守恒性、连续介质响应、异质材料耦合、高斯协方差传输与渲染、热相变以及反事实响应。每个参考标准都声明其有效范围，评估结果区分PASS、FAIL、NA和INVALID四种状态，从而将物理上的失败

    arXiv:2610.10554v1 Announce Type: cross  Abstract: 3D Gaussian Splatting has evolved from static reconstruction toward physics-integrated representations meant to predict how scenes change under interaction. This creates an evaluation problem: a rollout can look plausible while relying on incorrect internal mechanics, and visual agreement with observed motion does not establish a correct response to a new force, material edit, contact, or thermal intervention. We introduce GaussianBench, a physics-fidelity evaluation suite for physics-integrated Gaussian scene representations. It uses frozen file-based scenes, simulator-independent scorers, and analytical or measured references. The benchmark tests conservation, continuum response, heterogeneous-material coupling, Gaussian covariance transport and rendering, thermal phase change, and counterfactual response. Each reference declares its regime of validity, and outcomes distinguish PASS, FAIL, NA, and INVALID, separating physical failure
    
[^321]: 冻结解码器，修复编码器：面向基于SVD的KV缓存压缩的参数高效自适应方法

    Freeze the Decoder, Heal the Encoder: Parameter-Efficient Adaptation for SVD-Based KV-Cache Compression

    [https://arxiv.org/abs/2610.10552](https://arxiv.org/abs/2610.10552)

    该论文揭示了在共享学习率下比较参数高效微调方法的统计学陷阱：SVD式KV缓存压缩修复中“冻结解码器、仅修复编码器”看似显著的优势，实际上是各方案可训练参数数量悬殊导致的偏差，而非真实效果。

    

    在单一共享学习率下比较各种参数高效微调方案是一种常见但有缺陷的做法：当被比较的各方案拥有截然不同的可训练参数数量时，共享学习率会同时压低可训练参数较多方案的均值并放大其方差，从而为参数最少的方案制造出一种看似在多个随机种子下都显著的大幅优势，而这并非真实效应。我们在一个具体场景中记录了这种混淆因素：事后基于SVD的KV缓存压缩，即通过将已预训练模型的键/值权重分解为下投影（“编码器”）和上投影（“解码器”），将其转换为低秩（多头潜在注意力风格）缓存，随后进行短暂的微调（“修复”）以恢复因截断而损失的精度。在共享学习率下，冻结解码器而仅修复编码器看起来明显优于修复解码器或同时修复两个因子；一旦每个方案……（原文摘要在此处被截断）

    arXiv:2610.10552v1 Announce Type: cross  Abstract: Comparing parameter-efficient fine-tuning recipes under a single, shared learning rate is a common but flawed practice: when the arms being compared have very different trainable-parameter counts, a shared rate can simultaneously depress the larger arms' means and inflate their variance, manufacturing a large, seemingly multi-seed-significant advantage for the smallest arm that is not a real effect. We document this confound in a concrete setting: post-hoc SVD-based KV-cache compression, where an already-pretrained model is converted to a low-rank (multi-head-latent-attention-style) cache by factorizing its key/value weights into a down-projection ("encoder") and an up-projection ("decoder"), after which a short fine-tune ("healing") recovers the accuracy lost to truncation. Under a shared learning rate, freezing the decoder and healing only the encoder looks like a clear win over healing the decoder or both factors; once every arm is 
    
[^322]: 面向间隔重复的可解释记忆模型

    Interpretable Memory Models for Spaced Repetition

    [https://arxiv.org/abs/2610.10548](https://arxiv.org/abs/2610.10548)

    提出SBD记忆模型，在保持与最先进模型几乎相同准确性的同时，可解释性更强且模型体积缩小80%。

    

    间隔重复软件通过一个根据复习日志拟合的记忆模型来安排复习计划。然而，仅在测试集上的准确性并不足以证明模型的质量，因为可用的数据是由现有调度器生成的，而新的解决方案必须能够外推超越这些数据的范围。此外，模型还需要具备简单的机制性解释。我们提出了SBD，一个比当前最先进模型更具可解释性、体积缩小80%，同时保持几乎相同准确性的模型。

    arXiv:2610.10548v1 Announce Type: cross  Abstract: Spaced repetition software schedules reviews with a memory model fit to review logs. Accuracy on a test set is not sufficient evidence of quality since available data are produced by existing schedulers, and new solutions must extrapolate beyond them. A model also needs a simple mechanistic interpretation. We present SBD, a model that is more interpretable and 80% smaller than the current state of the art at nearly the same accuracy.
    
[^323]: 基于支撑集保持蒸馏的多智能体协调

    Multi-Agent Coordination via Support-Preserving Distillation

    [https://arxiv.org/abs/2610.10087](https://arxiv.org/abs/2610.10087)

    提出MoSDOT方法，利用条件半离散最优传输将噪声样本确定性分配到有限模式支撑集上，解决基于流的多智能体教师在蒸馏过程中模式冲突误差传播给学生模型的问题。

    

    离线多智能体强化学习（MARL）日益依赖生成式策略来建模多模态联合行为，通常是在集中式训练分散式执行（CTDE）框架下，将集中式教师蒸馏为分散式单步执行者。我们识别出教师训练阶段的一种失败模式：标准的基于流的教师将噪声与回放目标独立配对，导致相邻的噪声样本可能被路由到相互冲突的协调模式。教师随后会在有效模式之间产生样本，而由于蒸馏损失将每个局部执行者回归到给定局部输入下教师输出的条件均值，这一误差不会被吸收，而是传播到学生模型。为了消除这一教师侧的伪影，我们提出模式支撑半离散最优传输（MoSDOT），它将多模态回放数据总结为具有预设容量的有限模式支撑集，并在教师训练之前使用条件半离散最优传输将每个噪声样本分配到单一模式……

    arXiv:2610.10087v1 Announce Type: new  Abstract: Offline MARL increasingly relies on generative policies to model multimodal joint behavior, typically by distilling a centralized teacher into decentralized one-step actors under the CTDE. We identify a failure mode at the teacher training stage: standard flow-based teachers pair noise with replay targets independently, so nearby noise samples can be routed toward conflicting coordination modes. The teacher then produces samples between valid modes, and because the distillation loss regresses each local actor onto the conditional mean of the teacher's output given local input, this error is not absorbed but propagated to the student. To remove this teacher-side artifact, we propose Mode-Support Semi-Discrete Optimal Transport (MoSDOT), which summarizes multimodal replay into a finite mode support with prescribed capacities and uses conditional semi-discrete optimal transport to assign each noise sample to a single mode before teacher tra
    
[^324]: 多组分材料中通用机器学习力场误差的起源

    Origins of Universal Machine Learning Force-Field Errors in Multicomponent Materials

    [https://arxiv.org/abs/2610.09837](https://arxiv.org/abs/2610.09837)

    该论文构建了包含7,599个多组分构型的基准测试集，系统评估了11个预训练通用机器学习力场在多组分材料中的表现，并揭示了训练参考覆盖率不足和局部几何异质性增大是力场误差的主要来源。

    

    通用机器学习力场对由成分设计生成的多组分环境的泛化能力仍缺乏充分评估。我们构建了一个包含7,599个多组分构型的基准测试集，其灵感来源于高熵设计、元素取代和阴离子混合。我们以密度泛函理论为参照，对11个预训练模型在能量、力和应力方面的表现进行了评估，并将评估扩展至弹性、振动和吸附相关性质。我们通过训练参考覆盖率、局部几何异质性、距离方向性和元素响应来分析力误差。到训练参考环境的距离揭示了覆盖差异与误差增加之间的定性关联，但在相似距离下仍存在显著变化。误差较高的组表现出更大的局部几何异质性，尽管OMat24对这些环境提供了广泛的覆盖。相对……

    arXiv:2610.09837v1 Announce Type: cross  Abstract: Universal machine learning force-field generalization to multicomponent environments generated by compositional design remains insufficiently assessed. We construct a benchmark of 7,599 multicomponent configurations inspired by high-entropy design, elemental substitution and anion mixing. Eleven pretrained models are evaluated against density functional theory for energies, forces and stresses, with assessment extended to elastic, vibrational and adsorption-related properties. Force errors are analysed through training-reference coverage, local geometric heterogeneity, distance directionality and elemental response. Distances to training-reference environments reveal a qualitative association between coverage differences and increasing errors, while substantial variation remains at similar distances. Higher-error groups show greater local geometric heterogeneity, although OMat24 provides broad coverage of these environments. Relative t
    
[^325]: 上下文学习者何时应扩展其假设空间？

    When Should an In-Context Learner Expand Its Hypothesis Space?

    [https://arxiv.org/abs/2610.09471](https://arxiv.org/abs/2610.09471)

    该论文将上下文学习中“何时扩展假设空间”的问题形式化为有代价的序贯决策，并通过结构修正环境证明：修正决策是一个由价格、时间范围和查询等价值因素决定的价值边界，而非单纯的证据阈值。

    

    学习系统能够在熟悉的模型族内快速适应。更困难的一步出现在更早的阶段：从可能是噪声、例外、族内变化或族外结构的观察中，判断开启一个更丰富的模型族是否值得其代价。我们将此视为一个有代价的序贯决策问题：预测失败必须被转化为结构性证据，证据被转化为扩展的价值，价值再被转化为行动。结构修正环境从每种来源产生匹配的失败案例，独立于证据地改变扩展的价格与剩余时间范围，并支持精确的贝叶斯计算以及一次性决策的精确规范性解。其解表明，修正是一个价值边界而非证据阈值：同一历史在不同价格、时间范围和公告查询下会有不同的最优行动，局部修复与扩展之间的边界由规则……所设定

    arXiv:2610.09471v1 Announce Type: new  Abstract: Learning systems adapt quickly inside a familiar family of models. The harder step comes earlier: deciding, from observations that could be noise, an exception, a change within the family or structure outside it, whether opening a richer family is worth its cost. We treat this as a costly sequential decision: prediction failure must be turned into structural evidence, evidence into a value of expansion, and value into action. The Structural Revision Environment produces matched failures from each source, varies the price of expansion and the remaining horizon independently of the evidence, and admits exact Bayesian calculations and an exact normative solution of the one-shot decision. Its solution shows that revision is a value boundary and not an evidence threshold: one history has different optimal actions under different prices, horizons and announced queries, the boundary between local repair and expansion is set by the inputs a rule
    
[^326]: 共享几何作为罗塞塔石碑：无需配对数据的跨模态对齐

    Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data

    [https://arxiv.org/abs/2610.09411](https://arxiv.org/abs/2610.09411)

    提出一种简单的Wasserstein Procrustes方法，通过粗粒度几何初始化估计单一正交映射，无需任何配对数据即可实现独立训练模型的跨模态表征对齐，并证明标准几何对齐指标可准确预测对齐的可行性。

    

    多模态表征能够实现零样本分类和检索，但对齐独立训练的模型通常需要大量配对数据。然而，柏拉图表征假说表明，在不同模态上训练的模型可能会自发地收敛到共享的表征几何。那么，我们是否真的需要配对样本来进行跨模态对齐呢？值得注意的是，我们证明对于粗粒度的跨模态对齐而言，配对样本是不必要的。我们提出了一种简单的Wasserstein Procrustes方法，通过粗粒度几何初始化，仅通过估计单一的正交映射即可对齐两个互不相交的嵌入集合，而无需看到任何配对样本。在跨越多个数据集、模态和单模态模型的实验中，我们展示了对齐独立训练的表征始终可以无需配对数据完成，并且标准的几何对齐指标能够准确预测何时可以实现这种对齐。尽管如此，我们仍然可以自然地从配对样本中获益。

    arXiv:2610.09411v1 Announce Type: new  Abstract: Multimodal representations enable zero-shot classification and retrieval, but aligning independently trained models usually requires large amounts of paired data. Yet, the Platonic Representation Hypothesis suggests that models trained on different modalities may converge spontaneously toward a shared representation geometry. But then, do we even need paired examples for cross-modal alignment? Remarkably, we show that paired examples are unnecessary for coarse cross-modal alignment. Our simple Wasserstein Procrustes method with a coarse geometric initialization aligns two disjoint embedding sets by estimating a single orthogonal map without seeing any pairs. Across datasets, modalities, and unimodal models, we show that we can consistently align independently trained representations without pairs, and standard geometric alignment metrics accurately predict when this is possible. Nevertheless, we can naturally benefit from paired examples
    
[^327]: 损失差条件互信息的精度—信息权衡

    An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information

    [https://arxiv.org/abs/2610.09206](https://arxiv.org/abs/2610.09206)

    论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。

    

    损失差条件互信息（ld-CMI）是泛化上界的超样本层次结构中最小的标准观测量：它衡量学习器的损失差在多大程度上泄露了它训练时使用的是每一对候选样本中的哪一个。已知精度会迫使信息进入模型；而数据处理不等式并不能将此类下界传递到损失上。我们通过对损失差的三个矩进行约束，证明了精度同样会迫使ld-CMI。对于使用在零点斜率非零的光滑凸损失（如逻辑损失）的线性预测器，加上曲率和增长均呈幂次 r≥2 的正则化器，在维度至少随 n 线性增长的缩放符号立方体上的乘积分布中，每个在最优样本量 n≍ε^(-2+2/r) 下于这些分布上期望超额风险至多为 ε 的正规学习器，其最坏情况ld-CMI达到 n 比特量级，且为 Θ(n/(1+(τ/φ…

    arXiv:2610.09206v1 Announce Type: new  Abstract: Loss-difference conditional mutual information (ld-CMI) uses the smallest of the standard observations in the supersample hierarchy of generalization bounds: it measures what a learner's loss differences reveal about which candidate of each pair it was trained on. Accuracy is known to force information into the model; data processing does not carry such lower bounds to losses. We show, by bounding three moments of the loss differences, that accuracy also forces ld-CMI. For linear predictors with a smooth convex loss of nonzero slope at zero, such as the logistic loss, plus a regularizer whose curvature and growth are both of power $r\ge2$, on product distributions over a scaled sign cube in dimension at least linear in $n$, every proper learner with expected excess risk at most $\varepsilon$ on these distributions at the optimal sample size $n\asymp\varepsilon^{-2+2/r}$ has worst-case ld-CMI of order $n$ bits, and $\Theta(n/(1+(\tau/\var
    
[^328]: 为你的提示加噪：连续扩散语言模型中对条件令牌添加噪声

    Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.09145](https://arxiv.org/abs/2610.09145)

    在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。

    

    我们重新审视了连续扩散语言模型文献中的一个公认标准做法，即在训练期间保持条件提示令牌为干净（无噪声）状态。我们做了一个非常简单的修改：在训练期间也对条件提示令牌添加噪声。我们证明，在这一修改后的训练目标下，模型在数独和N皇后等组合推理任务中获得了更好的泛化能力，且在更难的变体上收益最大（数独困难版的解决率从3.73%提升至24.65%），同时生成解的多样性也有所提高（10x10 N皇后问题的覆盖率从50.60%提升至73.79%）。我们还展示了在使用Gigaword摘要数据集的中等数据规模下，自然语言生成质量有可衡量的提升，但值得注意的是，这些收益并不能迁移到所有自然语言任务上（例如开放式对话生成）。我们的方法只需对训练目标进行单行修改，无需额外的……

    arXiv:2610.09145v1 Announce Type: cross  Abstract: We revisit a standard accepted practice in the continuous diffusion language model   literature of fixing conditioning prompt tokens clean during training.   We make a very simple modification: also noise the conditioning prompt tokens during training.   We demonstrate that under this modified training objective, we achieve better generalization   in combinatorial reasoning tasks such as Sudoku and N-Queens, with the largest gains on harder variants   ($3.73\% \to 24.65\%$ solve rate on Sudoku Hard), and increased diversity of generated solutions ($50.60\% \to 73.79\%$ coverage on   10x10 N-Queens). We also show measurable improvements to natural language generation quality   in modest dataset regimes with Gigaword summarization, but notably demonstrate that gains do not   transfer to all natural language tasks (e.g open ended dialogue generation).   Our method is a single line change to the training objective, requires no additional i
    
[^329]: MaRK：用于状态空间模型中动态算子条件化的马尔可夫自适应循环核

    MaRK: Markov-adapted Recurrent Kernels for Dynamic Operator Conditioning in State Space Models

    [https://arxiv.org/abs/2610.09092](https://arxiv.org/abs/2610.09092)

    MaRK提出了一种动态算子条件化框架，将上下文向量直接映射为对冻结SSM循环算子参数（A、B、C、D、Δ）的有界调制，使每个扩散时间步都能动态重塑模型的输入输出记忆核。

    

    状态空间模型为序列建模提供了一种高效的Transformer替代方案，然而，为迭代生成而对预训练SSM进行条件化时，通常是在循环算子之外操作的，即通过输入注入或激活调制实现。尽管这类机制能让模型接触到条件信息，但它们使底层的时间动态保持固定。我们提出了MaRK（马尔可夫自适应循环核），这是一个动态算子条件化框架，它将上下文向量直接映射到冻结SSM的循环算子的有界调制中，涵盖循环参数A、读入参数B、读出参数C、跳跃参数D以及离散化参数Δ。通过线性参数变分SSM（LPV-SSM）系统的视角来看，MaRK诱导出一个以上下文为索引的马尔可夫参数序列族，使得每个扩散时间步都能够重塑模型的输入输出记忆核。我们在一个冻结的1.11亿参数Hydra SSM骨干网络上实例化了MaRK，并研究了三种适配器几何结构：

    arXiv:2610.09092v1 Announce Type: new  Abstract: State Space Models (SSMs) offer an efficient alternative to Transformers for sequence modeling, yet conditioning pre-trained SSMs for iterative generation typically operates outside the recurrent operator, through input injection or activation modulation. While such mechanisms expose the model to conditioning information, they leave the underlying temporal dynamics fixed. We introduce MaRK (Markov-adapted Recurrent Kernels), a dynamic operator-conditioning framework that maps context vectors directly into bounded modulations of a frozen SSM's recurrence ($A$), read-in ($B$), read-out ($C$), skip ($D$), and discretization ($\Delta$) parameters. Viewed through the lens of LPV-SSM systems, MaRK induces a context-indexed family of Markov parameter sequences, allowing each diffusion timestep to reshape the model's input-output memory kernel. We instantiate MaRK on a frozen 111M-parameter Hydra SSM backbone and study three adapter geometries: 
    
[^330]: SPIN：用于稀疏注意力的影子预测索引器

    SPIN: Shadow Predictive Indexer for Sparse Attention

    [https://arxiv.org/abs/2610.09025](https://arxiv.org/abs/2610.09025)

    SPIN 提出基于历史的轻量级预测机制来识别重要 KV 块，避免每个解码步骤对完整 KV 缓存评分，在保持任务质量的同时实现 30-40% 的稀疏度，并将 vLLM 服务的吞吐量最高提升 14.9%、token 间中位延迟最高降低 13.2%。

    

    基于索引器的稀疏注意力通过仅向核心注意力传递固定数量的少量重要 token 来降低其成本。然而，索引器仍然需要在每个解码步骤中对整个 KV 缓存进行评分。随着上下文长度的增长，这种评分开销成为主要瓶颈。我们提出 SPIN（影子预测索引器）来降低这种索引器开销。SPIN 使用轻量级的、基于历史的预测来识别重要的 KV 块，从而避免在每个解码步骤中对完整 KV 缓存进行评分。SPIN 将 KV 块和投机解码作为一等公民纳入设计与实现考量。在长上下文和智能体基准测试上的广泛评估中，SPIN 在保持任务质量的同时实现了 30-40% 的稀疏度。在端到端 vLLM 服务中，SPIN 将输出吞吐量最高提升 14.9%，并将 token 间中位延迟最高降低 13.2%。

    arXiv:2610.09025v1 Announce Type: new  Abstract: Indexer-based sparse attention reduces the cost of core attention by passing only a fixed, small number of important tokens to it. However, the indexer must still score the entire KV cache at every decoding step. This scoring overhead becomes a major bottleneck as the context length grows. We propose SPIN (Shadow Predictive Indexer) to reduce this indexer overhead. SPIN uses lightweight, history-based prediction to identify important KV blocks, avoiding the need to score the full KV cache at every decoding step. SPIN treats KV blocks and speculative decoding as first-class design and implementation considerations. Across extensive evaluations on long-context and agentic benchmarks, SPIN achieves 30-40% sparsity while preserving task quality. In end-to-end vLLM serving, SPIN improves output throughput by up to 14.9% and reduces median inter-token latency by up to 13.2%.
    
[^331]: 面向离线视觉控制的定向时序表征

    Directed Temporal Representations for Offline Visual Control

    [https://arxiv.org/abs/2610.08960](https://arxiv.org/abs/2610.08960)

    该论文提出DTRC方法，在冻结世界模型特征之上学习与时间可达性对齐的有向时间拟度量表征，并利用其时间进度信号作为评论家，实现离线视觉目标条件策略的直接学习。

    

    预测型世界模型为控制任务提供了紧凑的视觉表征。然而，控制需要一种与时间可达性对齐的潜在几何结构，而不仅仅是预测相似性。我们提出了面向控制的定向时序表征，该方法在冻结的LeWorldModel（LeWM）特征之上，从离线视觉轨迹中学习这种几何结构。DTRC在所学习的控制表征上构建有向时间拟度量。短程时间偏移用于校准距离尺度；自举目标将时间可达性扩展到更长的时间范围；动作条件一致性使表征与局部转移动力学对齐。所得的距离估计时间到达代价，而其在一次转移中的变化定义了相对于目标的时间进度。我们将该进度信号作为时间评论家，用于直接的目标条件策略学习。模型辅助目标还提供了额外的训练信号。

    arXiv:2610.08960v1 Announce Type: new  Abstract: Predictive world models provide compact visual representations for control. Control requires a latent geometry aligned with temporal reachability rather than predictive similarity alone. We introduce Directed Temporal Representations for Control (DTRC), which learns such a geometry from offline visual trajectories on top of frozen LeWorldModel (LeWM) features. DTRC constructs a directed temporal quasimetric over the learned control representation. Short-range temporal offsets calibrate the distance scale. Bootstrapped targets extend temporal reachability across longer horizons. Action-conditioned consistency aligns the representation with local transition dynamics. The resulting distance estimates temporal reaching cost, and its change across a transition defines goal-relative temporal progress. We use this progress signal as a temporal critic for direct goal-conditioned policy learning. Model-assisted targets provide an additional train
    
[^332]: 一项关于智能体技能下游效用的实证研究

    An Empirical Study of Agent Skills' Downstream Utility

    [https://arxiv.org/abs/2610.08875](https://arxiv.org/abs/2610.08875)

    本文通过在87个SkillsBench任务上的实证研究，将智能体技能的下游效用量化为相对于无技能基线的通过率差异，并揭示效用如何取决于技能内容、执行配置与多技能组织方式。

    

    智能体技能将程序性指导与资源打包以便复用，但一个相关的技能并不一定能提升任务表现。现有研究刻画了技能内容并评估了下游性能，但对于效用如何取决于内容、执行配置和多技能组织方式，所提供的解释仍然有限。我们在87个SkillsBench任务上开展实证研究，将下游效用定义为在相同的模型-测试框架配置下，同一任务相对于无技能基线的通过率差异。我们在九种配置下比较相同的技能，随后在三种选定配置下考察其他已发布的技能以及固定技能集的不同组织方式。我们从包含37,596个技能的精选语料库中检索市场候选技能。通过借助大语言模型对内容、执行轨迹和最终产物进行分析，并由作者复核，我们将所提供的支持与实际使用情况及任务结果关联起来。相同的技能……

    arXiv:2610.08875v1 Announce Type: cross  Abstract: Agent Skills package procedural guidance and resources for reuse, but a relevant Skill does not necessarily improve task performance. Existing studies characterize Skill content and evaluate downstream performance, yet provide limited explanations of how utility depends on content, execution configuration, and multi-Skill organization. We conduct an empirical study on 87 SkillsBench tasks, defining downstream utility as the pass-rate difference from No-Skill on the same tasks under the same model--harness configuration. We compare the same Skills across nine configurations, then examine alternative published Skills and organizations of fixed Skill sets under three selected configurations. We retrieve marketplace candidates from a curated corpus of 37,596 Skills. LLM-assisted analysis of content, execution traces, and final artifacts, followed by author review, relates provided support to actual use and task outcomes. The same Skills he
    
[^333]: 只为FUNS：基于大语言模型引导的时空图节点生成方法用于预测未观测节点状态

    Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States

    [https://arxiv.org/abs/2610.08818](https://arxiv.org/abs/2610.08818)

    该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。

    

    时空预测是物流、城市规划和智能交通系统的基石。然而，受部署成本和维护资源的限制，传感器网络往往缺乏全面的空间覆盖，这使得预测未观测节点状态（FUNS）成为一项至关重要却又极具挑战性的任务。传统模型依赖历史观测数据，在遇到没有先前记录的节点时通常会表现失常。为解决这一问题，我们将该问题重新定义为时空图上的条件生成任务，并提出GenST框架，该框架引入大语言模型（LLMs）作为语义桥梁，利用经过微调的预训练LLM从节点描述（如功能分区和道路网络结构）中提取丰富的语义特征，以弥补缺失的时空信号。具体而言，我们设计了一个两阶段生成架构：时空变分自编码器（VAE）首先压缩……

    arXiv:2610.08818v1 Announce Type: cross  Abstract: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses 
    
[^334]: 我看得够了吗？冻结的视频-语言模型中编码了证据就绪度信号

    Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness

    [https://arxiv.org/abs/2610.08560](https://arxiv.org/abs/2610.08560)

    该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。

    

    流式视频-语言模型不仅需要决定回答什么，还必须判断当前问题所需的证据是否已经到来。现有系统将该决策作为一个单独的触发器来学习；我们则探究一个未经修改的模型是否已经在计算这一决策。我们证明，冻结的视频大语言模型内部携带一种线性可读的证据就绪信号，该信号的标注来自带时间戳的证据而非模型输出。在一个共享的逐字节相同的评估中，该信号在全部七个模型中均可被解码（在最严格的“未就绪”采样下AUROC为0.733–0.905，而拟合的时钟模型接近随机水平），并且在完全未接触某基准家族视频数据的情况下训练的探针仍能读取该家族的数据。该信号是问题条件化的：在逐字节相同的视频窗口上，仅改变问题就能使66.1%的问题配对上的读出结果发生反转，而所有问题盲的控制组按构造均处于随机水平。模型即使给出错误答案，仍然编码了就绪状态：在错误答案中AUROC仍为0.722。

    arXiv:2610.08560v1 Announce Type: cross  Abstract: Streaming video-language models must decide not only what to answer, but whether the evidence needed for the current question has arrived. Existing systems learn that decision as a separate trigger; we ask whether an unmodified model already computes it. We show that frozen VideoLLMs carry a linearly readable evidence-readiness signal, labelled from timestamped evidence rather than from model output. It decodes in all seven models of a shared byte-identical evaluation (AUROC 0.733-0.905 under the strictest not-ready sampling, where a fitted clock is near chance), and a probe fitted without any of a benchmark family's footage still reads that family. It is question-conditioned: on byte-identical windows, changing only the question reverses the readout on 66.1% of pairs, while every question-blind control is at chance by construction. The model can answer incorrectly and still encode readiness: AUROC remains 0.722 among wrong answers. Re
    
[^335]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^336]: 赋权的几何学

    The Geometry of Empowerment

    [https://arxiv.org/abs/2610.07796](https://arxiv.org/abs/2610.07796)

    本文将赋权最大化与技能学习方法相联系，提出了解释赋权的新几何框架，解答了赋权与结构中心性之间联系的长期开放问题，并揭示了信息几何与奖励几何的区别，为构建可扩展的赋权最大化方法奠定理论基础。

    

    赋权刻画了智能体主动控制其环境的能力。尽管作为一种信息论量在概念上颇具吸引力，但赋权与那些能广泛通往未来结果的结构性中心状态之间的联系，一直是一个悬而未决的问题。在这项工作中，我们将赋权最大化与技能学习方法联系起来，为解释和分析赋权提供了新的几何视角。我们的分析回答了关于赋权与结构中心性之间联系的长期开放问题，还揭示了信息几何与奖励几何之间的区别，为构建可扩展的赋权最大化方法提供了重要的理论启示。网站与代码可在 https://empowerment-geometry.github.io/ 获取。

    arXiv:2610.07796v1 Announce Type: cross  Abstract: Empowerment captures the capacity for an agent to actively control its environment. While conceptually appealing as an information-theoretic quantity, the connection between empowerment and structurally central states that provide broad access to future outcomes has remained an open question. In this work, we link empowerment maximization and skill-learning methods to provide new geometries for interpreting and analyzing empowerment. Our analyses answer longstanding open questions on the connections between empowerment and structural centrality. Our analyses also reveal distinctions between information and reward geometries, highlighting important theoretical implications to build scalable empowerment-maximization methods. Website and code can be found at https://empowerment-geometry.github.io/.
    
[^337]: 神经运动层级网络：面向sEMG解码鲁棒泛化的生理学归纳偏置

    Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding

    [https://arxiv.org/abs/2610.07713](https://arxiv.org/abs/2610.07713)

    提出受神经运动层级结构启发的NHN网络，通过引入生理学归纳偏置学习紧凑的潜在神经运动状态，从而在跨用户、跨会话的sEMG解码中实现鲁棒泛化。

    

    表面肌电图（sEMG）为运动解码和人机交互提供了一种可穿戴、无创的神经肌肉活动接口。大规模人群解码仍然困难，原因在于sEMG与神经肌肉活动之间的关系在不同用户和会话之间存在差异，而任务相关的动态跨越多个通道和多个时间尺度。从任务标签学习波形到输出的映射，使得记录变异性与协调性运动活动之间的区分停留在隐式层面。我们提出了神经运动层级网络（NHN），它从任务监督中学习一个紧凑的潜在神经运动状态，以表征任务相关的神经肌肉协调。NHN通过受神经运动组织结构启发的层级机制来构建该潜在状态。它在保持相对强度的同时适应记录统计特性。其时空编码器采用参数高效的通道交互方式，并通过多…（摘要在此处截断）

    arXiv:2610.07713v1 Announce Type: new  Abstract: Surface electromyography (sEMG) provides a wearable, noninvasive interface to neuromuscular activity for movement decoding and human-computer interaction. Population-scale decoding remains difficult because the relationship between sEMG and neuromuscular activity varies across users and sessions, while task-relevant dynamics span channels and multiple timescales. Learning waveform-to-output mappings from task labels leaves the distinction between recording variability and coordinated motor activity implicit. We introduce the Neuromotor Hierarchy Network (NHN), which learns a compact latent neuromotor state from task supervision to represent task-relevant neuromuscular coordination. NHN constructs this latent state through a hierarchy inspired by neuromotor organization.It adapts recording statistics while preserving relative intensity.Its spatiotemporal encoder uses parameter-efficient channel interactions and modulates features with mul
    
[^338]: 学习决策而非推理：基于低秩激活转向的参数高效决策算子

    Learning to Decide, Not to Reason: Parameter-Efficient Decision Operators via Low-Rank Activation Steering

    [https://arxiv.org/abs/2610.06950](https://arxiv.org/abs/2610.06950)

    该论文提出一种仅用2.3万至33万参数、通过行为克隆训练的低秩激活转向决策算子，能在不损失精度的情况下将3685个token的长推理压缩为6个token的快速决策，训练成本比现有强化学习方法低约两个数量级。

    

    目前，向冻结的语言模型中注入技能需要百万级参数和一套强化学习流程。我们提出了\method{}，这是一个通过行为克隆训练的System-1决策算子，将这一成本降低了约两个数量级。默认算子仅使用33万参数即可匹敌一个通过强化学习训练的133万参数算子，性能相当或更优，同时将3685个token的深思推理压缩为6个token的决策且不损失精度。一个秩为4、仅有2.3万参数的变体——仅为已发表的最强技能算子的1/58——足以胜任SearchQA任务，并在LiveMath上接近足够（更高秩仍有帮助）；同一方法可迁移至五个任务和三个骨干模型，且在训练完成数月后发布的LiveMath问题上，分布外收益依然保持。与先前工作的差距在于可训练性，而这一差距由初始化和架构共同决定……（原文摘要在此处截断）

    arXiv:2610.06950v1 Announce Type: cross  Abstract: Injecting skills into a frozen language model currently costs a million parameters and a reinforcement-learning pipeline. We introduce \method{}, a System-1 decision operator trained by behavior cloning that lowers this cost by roughly two orders of magnitude. The default operator uses 330K parameters to match a 1.33M-parameter operator trained with reinforcement learning, exceeds or achieve comparable performance, while collapsing 3,685-token deliberation into a 6-token decision with no loss in accuracy. A rank-4 variant with 23K parameters, 1/58 of the strongest published skill operator, suffices for SearchQA and near-suffices for LiveMath, where higher rank still helps; the same recipe transfers across five tasks and three backbones, with out-of-distribution gains persisting on LiveMath problems released months after training. The gap to prior work is trainability, and it is set jointly by initialization and architecture: the initia
    
[^339]: SOL：利用双切片Wasserstein度量衡量文本分布之间的差距

    SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics

    [https://arxiv.org/abs/2610.06513](https://arxiv.org/abs/2610.06513)

    提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。

    

    评估文本生成需要衡量生成分布与数据分布的匹配程度。对于自回归模型，这通过困惑度来实现；而扩散模型和基于流的语言模型只能提供似然界，其紧密程度在不同模型家族之间存在差异。基于样本的替代方法（如结合熵的生成式困惑度）则未考虑分布拟合情况。我们提出了SOL，一种文本分布之间的距离度量。每个序列由其在固定Transformer下隐藏状态的经验测度来表示，并通过双切片Wasserstein距离来比较这些测度的分布。我们证明，当Transformer是单射时，SOL是一个真正的度量。实验表明，SOL能够检测分布性失败、恢复预期的模型趋势，并提供稳定的基于样本的估计。我们提出SOL以填补当前非自回归模型评估协议中的空白。

    arXiv:2610.06513v2 Announce Type: replace  Abstract: Evaluating text generation requires measuring how well the generated distribution matches the data distribution. For autoregressive models, this is done by the perplexity. Diffusion and flow-based language models can only provide a likelihood bound, whose tightness differs between model families. Sample-based substitutes such as generative perplexity with entropy do not consider the distribution fit. We propose SOL,   a distance between text distributions. Each sequence is represented by the empirical measure of its hidden states under a fixed transformer and the distributions of these measures are compared by the double sliced Wasserstein distance. We prove that SOL is a metric if the transformer is injective. Experiments show that SOL detects distributional failures, recovers expected model trends, and provides stable sample-based estimates. We put forward SOL to fill the gap in the current evaluation protocol used for non auto-reg
    
[^340]: 生成流的普适性与收敛性

    Universality and Convergence of Generative Flows

    [https://arxiv.org/abs/2610.05490](https://arxiv.org/abs/2610.05490)

    该论文证明基于差值的平衡损失能以不依赖策略的显式常数在全变差意义下保证采样器的准确性，而基于比率的流匹配损失不具备这种保证，且反向策略决定了损失能否收敛到零以及梯度下降的收敛速度。

    

    生成流通过训练一个流使其达到平衡，从而从非归一化的目标分布中采样，而训练损失是从业者所关注的信号。我们追问这一信号的价值：小的损失能否证明采样器的准确性？损失能否被驱动至零？梯度下降实现这一点的速度有多快？损失决定了第一个问题：通过差值比较平衡两侧的损失，能够在全变差意义下界定流所诱导的采样器的误差，并给出不依赖于策略的显式常数；而通过比率进行比较的流匹配损失则不存在这样的界——只要其生成子在平衡处连续，即便在单个环上也是如此。在图上，反向策略决定了另外两个问题。一旦反向策略被冻结，平衡就变成在反向链下的不变性，因此有限图上存在性是免费的，而一个常数——该链的格林算子的范数——扮演着关键角色。（原文摘要在此处被截断）

    arXiv:2610.05490v2 Announce Type: replace  Abstract: Generative flows sample from an unnormalized target by training a flow to be balanced, and the training loss is the signal a practitioner watches. We ask what that signal is worth: whether a small loss certifies an accurate sampler, whether the loss can be driven to zero, and how fast gradient descent does so. The loss decides the first. Losses that compare the two sides of the balance by their difference bound, in total variation, the error of the sampler the flow implies, with explicit constants that do not involve the policy; flow-matching losses that compare them through a ratio admit no such bound, already on a single cycle, whenever their generator is continuous at balance. On graphs, the backward policy decides the other two. Once it is frozen, balance becomes invariance under the backward chain, so that existence is free on finite graphs, and one constant --- the norm of that chain's Green operator, which plays the role of an
    
[^341]: 人口规模扩展还是数据稀释？去中心化学习中局部拓扑演化的动力学

    Population Scaling or Data Dilution? Dynamics of Local Topology Evolution in Decentralized Learning

    [https://arxiv.org/abs/2610.05476](https://arxiv.org/abs/2610.05476)

    该论文揭示了去中心化学习中客户端数量扩展与数据稀释、拓扑混合及通信容量的耦合效应，证明环形拓扑的谱隙按 $\Theta(N^{-2})$ 衰减导致共识变慢，并提出基于“朋友的朋友”发现的局部自适应拓扑演化方法 LFHE，实验表明保持本地数据集规模固定可显著缓解规模扩展带来的性能惩罚。

    

    arXiv:2610.05476v2 公告类型：replace-cross。摘要：扩展去中心化学习不仅改变了客户端数量 $N$，还会改变信息传播与共识形成的动力学。我们认为，不能孤立地理解增加 $N$ 带来的影响，因为数据分配、依赖拓扑的混合过程以及通信容量可能同时发生变化。我们在 CIFAR-10 上以 $N\in\{10,50,100,200\}$ 研究了这些耦合效应，比较了度为二的环形图、静态随机图以及“局部优先启发式演化”——一种基于“朋友的朋友”发现的局部自适应拓扑演化过程。环形图提供了一个在分析上透明的失效模式：其 Metropolis 谱隙以 $\Theta(N^{-2})$ 的速度衰减，意味着随着人口规模的增长，模型间分歧的收缩会逐渐变慢。实验表明，若保持名义上的本地数据集规模固定，可以显著减轻当固定总量的数据集被分摊到各客户端时所观察到的人口规模惩罚。（摘要原文在此处截断）

    arXiv:2610.05476v2 Announce Type: replace-cross  Abstract: Scaling decentralized learning changes not only the number of clients $N$, but also the dynamics of information propagation and consensus. We argue that the effect of increasing $N$ cannot be understood in isolation, because data allocation, topology-dependent mixing, and communication capacity may change simultaneously. We study these coupled effects on CIFAR-10 with $N\in\{10,50,100,200\}$, comparing a degree-two Ring, a Static Random graph, and Local-First Heuristic Evolution (LFHE), a locally adaptive topology process based on friend-of-friend discovery. The Ring provides an analytically transparent failure mode: its Metropolis spectral gap decays as $\Theta(N^{-2})$, implying progressively slower contraction of model disagreement as the population grows. Experiments show that holding the nominal local dataset size fixed substantially reduces the apparent population penalty observed when a fixed total dataset is divided amo
    
[^342]: 在可容许性匹配条件下度量学习型单调时间聚合

    Measuring Learned Monotone Temporal Aggregation at Matched Admissibility

    [https://arxiv.org/abs/2610.05196](https://arxiv.org/abs/2610.05196)

    该论文在对比双方单调可容许性完全匹配的前提下，用构造上单调的循环网络（EWMA与高水位标记的可学习变换）度量学习型时间聚合的价值，并通过函数回归揭示出一条“涵盖边界”——学习到的单调通道可复现几何加权可分手工统计量族。

    

    风险监管对评分施加方向性约束；我们采纳其严格的逐输入形式——即评分对每一个风险暴露输入都单调非递减——作为规范性承诺。已部署的流水线（由单调的手工聚合特征输入符号约束的梯度提升）通过结构组合已天然满足该性质，因此“约束 vs 无约束”式的比较实际上是在为现有系统免费享有的保证定价。与之不同，我们在对比双方保持可容许性完全一致，度量“学习聚合”本身的价值。我们的工具是一个循环网络，其状态由经典风险统计量构成（带可学习变换的指数加权移动平均与高水位标记），在构造上对每个输入以及每个MC-dropout样本均保持单调。通过函数回归得到的核心发现是一条“涵盖边界”：学习到的单调通道能够复现几何加权可分手工统计量族……（原文摘要在此处截断）

    arXiv:2610.05196v2 Announce Type: replace  Abstract: Risk regulation imposes directional constraints on scores; we adopt their strict per-input form -- the score monotone non-decreasing in every exposure input -- as a normative commitment. Deployed pipelines -- monotone hand-crafted aggregates feeding sign-constrained gradient boosting -- already satisfy it by composition, so constrained-versus-unconstrained comparisons price a guarantee the incumbent has for free. We instead hold admissibility fixed on both sides and measure what learning the aggregation is worth. Our instrument is a recurrent network whose state is classical risk statistics (an exponentially weighted moving average and a high-water mark with learned transforms), monotone by construction in every input and per MC-dropout sample. The central finding, by functional regression, is a subsumption boundary: a learned monotone channel reproduces the geometrically weighted separable family of hand-crafted statistics, one chan
    
[^343]: 问路还有多长，而非离得多近：一种用于潜空间世界模型规划的学习式时序度量

    How Long, Not How Close: A Learned Temporal Metric for Planning in Latent World Models

    [https://arxiv.org/abs/2610.04988](https://arxiv.org/abs/2610.04988)

    提出TEMPO，一种学习“距离目标还差多少环境步数”而非“与目标多相似”的时序距离规划代价，无需改动或训练世界模型、也无需奖励或标签即可显著提升潜空间世界模型在远距离目标下的规划能力。

    

    潜空间世界模型通过在候选动作序列下将冻结的预测器向前滚动进行规划，并根据想象终态与目标之间的潜空间距离对候选方案进行排序。然而，当目标距离多个规划步骤之外时，这种排序方式会失效，因为潜空间距离衡量的是终态与目标的相似程度，而非距离达成目标还有多远。为解决这一问题，我们提出了TEMPO，这是一种时序距离规划目标，它不改动世界模型本身，仅从已用于训练世界模型的记录轨迹中学习，并且为规划器的搜索带来的额外开销可以忽略不计。TEMPO学习一个针对冻结潜变量的小型映射，使得同一回合中两个状态之间的距离反映它们之间相隔的环境步数，并将该距离融入规划器的代价函数中。它不需要奖励、策略或成功标签，并且由于其本身是一种代价度量而非模型，因此可应用于冻结的世界模型。

    arXiv:2610.04988v2 Announce Type: replace  Abstract: Latent world models plan by rolling a frozen predictor forward under candidate action sequences and ranking the candidates by the latent distance between their imagined end state and the goal. However, this ranking breaks down when the goal lies several plans away, because the latent distance measures how closely an end state resembles the goal rather than how far it remains from reaching it. To address this, we propose TEMPO, a temporal-distance planning objective that leaves the world model untouched, learns only from the recorded trajectories already used to train it, and adds negligible cost to the planner's search. TEMPO learns a small map of the frozen latent in which the distance between two states of an episode reflects the number of environment steps between them, and blends this distance into the planner's cost. It requires no rewards, policies or success labels and, being a cost rather than a model, applies to frozen world
    
[^344]: 分数并非结构：大脑对齐与跨语言迁移

    The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer

    [https://arxiv.org/abs/2610.03827](https://arxiv.org/abs/2610.03827)

    相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。

    

    相似性分数常被用作模型与大脑或跨语言间共享结构的证据。我们探讨当结构被移除或测量工具失效时，这样的分数实际反映的是什么，并在两个场景中进行了验证。在十七种语言中，语法性探针在语言距离越远时迁移效果越差（r = -0.66），但探针自身的准确率也沿同一轴线下降，在四种语言中处于随机水平（这四种语言是距离最远的五种中的四种）。剔除这些语言会使已解释方差减半，同时由于也缩小了距离范围，该设计无法判断梯度中有多少来自测量工具本身。将272个语言对视为独立样本时，导向效应的p值为0.0006；但以十七种语言为单位时，该效应不显著（p = 0.155）。在大脑对齐方面，一个训练目标将语言模型与fMRI响应的相似性从0.10提升到0.34（可靠性上限为0.54），然而

    arXiv:2610.03827v2 Announce Type: replace-cross  Abstract: Similarity scores are often offered as evidence that a model shares structure with the brain or across languages. We ask what such a score reads when that structure is removed, or when the instrument measuring it does not work, in two settings. Across seventeen languages, a grammaticality probe transfers worse between more distant languages (r = -0.66). But the probe's own accuracy falls along the same axis and is at chance in four languages, four of the five most distant. Dropping them halves the explained variance, and because it also narrows the range of distances, the design cannot say how much of the gradient is the instrument. Counting the 272 language pairs as independent gives p = 0.0006 for a steering effect that is null when the seventeen languages are the unit (p = 0.155). In brain alignment, a training objective raises a language model's similarity to fMRI responses from 0.10 to 0.34 (reliability ceiling 0.54), yet 
    
[^345]: SpectralCache：通过谱特征缓存加速基于扩散的世界模型

    SpectralCache: Accelerating Diffusion-Based World Models via Spectral Feature Caching

    [https://arxiv.org/abs/2610.02660](https://arxiv.org/abs/2610.02660)

    SpectralCache揭示了扩散世界模型特征在相邻去噪步骤中的谱稳定性，通过无需训练的谱缓存框架复用奇异子空间并线性外推奇异值，从而跳过冗余的Transformer计算，显著加速推理。

    

    基于扩散的世界模型能够实现高质量的交互式环境生成，但由于在去噪过程中需要反复进行Transformer计算，导致推理开销巨大。现有的缓存方法主要在特征或token层面利用时间冗余性，而对扩散特征底层数学结构的探索仍然十分有限。在本工作中，我们揭示了世界模型特征在相邻去噪步骤之间表现出高度稳定的奇异子空间，同时其奇异值遵循可预测的演化模式。基于这一观察，我们提出了SpectralCache，一个无需训练的谱缓存框架，该框架复用稳定的奇异子空间，并通过线性外推仅估计低维奇异值。我们进一步利用相邻全计算特征之间的谱一致性，通过奇异值缩放跳过部分昂贵的主干网络计算。大量实验……

    arXiv:2610.02660v1 Announce Type: cross  Abstract: Diffusion-based world models enable high-quality interactive environment generation but suffer from substantial inference overhead due to repeated Transformer evaluations during denoising. Existing caching methods mainly exploit temporal redundancy at the feature or token level, leaving the underlying mathematical structure of diffusion features largely unexplored. In this work, we reveal that world-model features exhibit highly stable singular subspaces across nearby denoising steps, while their singular values follow predictable evolution patterns. Building on this observation, we propose SpectralCache, a training-free spectral caching framework that reuses stable singular subspaces and estimates only low-dimensional singular values through linear extrapolation. We further exploit the spectral consistency between neighboring full-computation features to skip selected expensive backbone evaluations via singular value scaling. Extensiv
    
[^346]: 评估与提升大语言模型对输入序列变化的鲁棒性

    Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations

    [https://arxiv.org/abs/2610.02432](https://arxiv.org/abs/2610.02432)

    本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。

    

    生产系统中的大语言模型（LLM）面临提示注入、木马（后门）攻击以及自动质量指标被操纵等威胁。本论文开发了用于评估和提升大语言模型对对抗性输入序列变化鲁棒性的模型、方法和算法。我们提出了R_stab(f)，一种基于小输入扰动下逐步输出分布之间Jensen-Shannon散度的生成式鲁棒性度量。对于局部化攻击，我们证明了V(h) <= 1 - R_class(h)，其中R_class(h)是决策算子h在小扰动下保持其决策的概率。对于非局部化攻击，我们提出了一个经过校准的经验模型。针对LLM-as-a-Judge（大模型作裁判）系统，我们开发了ASA，一种自适应进化黑盒攻击，其攻击成功率（ASR）最高可达73.8%，在开源模型之间的迁移攻击成功率最高可达62.6%。在Trojan Detection Challenge 2023数据（Pythia-1.4B）上，代理触发器达到REA（摘要在此处被截断）

    arXiv:2610.02432v1 Announce Type: cross  Abstract: Large language models (LLMs) in production systems face prompt injections, trojans (backdoors), and manipulation of automatic quality metrics. This thesis develops models, methods, and algorithms for evaluating and improving LLM robustness to adversarial input sequence variations. We propose R_stab(f), a generative robustness metric based on the Jensen-Shannon divergence between per-step output distributions under small input perturbations. For localized attacks we prove V(h) <= 1 - R_class(h), where R_class(h) is the probability that a decision operator h keeps its decision under small perturbations. For non-localized attacks we propose a calibrated empirical model. For LLM-as-a-Judge systems we develop ASA, an adaptive evolutionary black-box attack that reaches an attack success rate (ASR) of up to 73.8%, with transfer between open models up to 62.6%. On Trojan Detection Challenge 2023 data (Pythia-1.4B), surrogate triggers reach REA
    
[^347]: 用于跨上下文KV缓存复用的预算化缓存修复

    Budgeted Cache Repair for Cross-Context KV-Cache Reuse

    [https://arxiv.org/abs/2610.02233](https://arxiv.org/abs/2610.02233)

    该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。

    

    跨上下文KV缓存复用是在新前缀下预测共享片段的键和值，而不是重新计算它们，并且据报告这样做不会带来质量损失。我们的发现并非如此，并识别出两个问题。（1）一个隐性代价：在MMLU和GSM8K上，复用会导致显著的准确率损失。（2）决策单元错误：目前没有任何决定是否复用缓存的规则能消除这一代价。真正有帮助的是选择缓存中哪些部分需要重新计算，而精准选择的收益随着选择单元的增大而下降：在单行（即单个token的键和值）层面，有依据的选择能消除超出随机水平49.5%的缓存误差；在64个token的分块层面为10.6%；而在整个调用层面则毫无效果。预算化缓存修复（BCR）正是在选择仍有收益的单元上进行操作。它从组装好的缓存中起草两个token，根据这两个token对缓存行支付的注意力对缓存行进行排序，并以三种布局之一精确地重新计算固定预算数量的缓存行。

    arXiv:2610.02233v1 Announce Type: cross  Abstract: Cross-context KV-cache reuse predicts a shared segment's keys and values under a new prefix instead of recomputing them, and has been reported to do so without quality loss. We find otherwise, and identify two problems. (1) A hidden cost: on MMLU and GSM8K, reuse costs substantial accuracy. (2) A decision at the wrong unit: no rule for deciding whether to reuse a cache removes that cost. What does help is choosing which parts of the cache to recompute, and the value of choosing well falls as the unit of choice grows: informed selection removes 49.5% of the cache error beyond chance at single rows (one token's keys and values), 10.6% at 64-token chunks, and nothing at the level of whole calls. Budgeted Cache Repair (BCR) acts at the unit where selection still pays. It drafts two tokens from the assembled cache, ranks cache rows by the attention those tokens pay them, and recomputes a fixed budget of rows exactly, in one of three layouts
    
[^348]: 低精度Transformer推理中的随机舍入：一项针对小型GPT-2的变精度仿真研究

    Stochastic Rounding in Low-Precision Transformer Inference: A Variable-Precision Emulation Study of a Small GPT-2

    [https://arxiv.org/abs/2610.01889](https://arxiv.org/abs/2610.01889)

    该研究通过变精度随机舍入（VPSR）算法将PRISM舍入库扩展至任意精度，并从理论与实验两方面揭示：低精度Transformer推理中随机舍入与就近舍入的优劣取决于运算位点，SR的误差以O(√n u)增长而RN以O(n u)增长，这一差距在长MLP下投影等长线性投影中最为明显。

    

    低精度Transformer推理应该使用随机舍入（SR）还是就近舍入（RN）？答案取决于观察网络中的哪个位置。我们通过固定数值格式、仅在各个运算位点改变舍入规则来隔离这一效应。为了能够在自由选择的精度下进行实验，我们通过变精度随机舍入（VPSR）算法将PRISM向量化舍入库扩展至任意虚拟精度，并证明舍入决策在硬件浮点运算中被精确评估。我们提出了两种分析，为这种位点级权衡提供互补的见解。首先，线性投影的概率前向误差界表明，SR的误差包络随约简长度n以O(√n u)增长，而RN为O(n u)，这一差距在低精度下迅速扩大，在较长的多层感知机（MLP）下投影中最为显著。其次，一个……（摘要原文在此处截断）

    arXiv:2610.01889v1 Announce Type: new  Abstract: Should low-precision transformer inference use stochastic rounding (SR) or round-to-nearest (RN)? The answer depends on where in the network you look. We isolate this effect by holding the numerical format fixed and varying only the rounding rule at individual operation sites. To enable experiments at freely chosen precisions, we extend the PRISM vectorized rounding library to arbitrary virtual precision via a variable-precision stochastic rounding (VPSR) algorithm, proving that the rounding decision is evaluated exactly in hardware floating point.   We develop two analyses providing complementary insight into this site-level trade-off. First, a probabilistic forward-error bound for linear projections shows that SR's error envelope grows as $O(\sqrt{n} u)$ in reduction length $n$, versus $O(n u)$ for RN, a gap that widens rapidly at low precision and is most pronounced in the long multilayer perceptron (MLP) down-projection. Second, a se
    
[^349]: 大语言模型人格遗忘

    LLM Persona Unlearning

    [https://arxiv.org/abs/2609.39882](https://arxiv.org/abs/2609.39882)

    该论文提出“人格遗忘”任务及PersonaUnlearnBench基准，通过权重级编辑使大语言模型中的指定人格难以被诱发，并发现标准遗忘方法无法在不牺牲生成质量或通用能力的情况下可靠地抹除目标人格。

    

    预训练使大语言模型（LLM）具备了与角色、风格、价值观和目标相关的广泛行为模式。后训练教会模型有条件地执行这些模式，并将乐于助人的“助手”设为默认，但并未从权重中抹除其他替代模式；因此，明确的提示可以诱发出持续影响判断、语言和行动的人格。在开放权重设置中，运行时的控制手段可能被移除，这促使了“人格遗忘”任务的出现：一种权重级别的编辑，使指定的人格在未见过的语境中难以被诱发和执行。我们提出了PersonaUnlearnBench，一个模型特定的配对基准，涵盖来自三个系列的六个大语言模型和五种人格，包含对齐的遗忘/保留数据集、留出的指令改写以及四轴评估。该基准表明，标准的遗忘方法无法在不牺牲有意义生成能力或通用效用的前提下可靠地抹除目标人格。

    arXiv:2609.39882v1 Announce Type: new  Abstract: Pre-training equips large language models (LLMs) with a broad repertoire of behavioral patterns associated with roles, styles, values, and goals. Post-training teaches conditional enactment and makes a helpful Assistant the default, but it does not erase alternative modes from the weights; explicit prompts can therefore elicit personas that repeatedly shape judgment, language, and action. In open-weight settings, runtime controls can be removed, motivating persona unlearning: a weight-level edit that makes a designated persona difficult to elicit and enact on unseen contexts. We introduce PersonaUnlearnBench, a model-specific paired benchmark spanning six LLMs from three families and five personas, with aligned forget/retain sets, held-out instruction paraphrases, and four-axis evaluation. The benchmark shows that standard unlearning methods cannot reliably erase the target persona without sacrificing meaningful generation or general uti
    
[^350]: 一种结合协方差矩阵自适应与有效维度的无参数零阶优化方法

    A Parameter-Free Zeroth-Order Method with Covariance Matrix Adaptation and Effective Dimension

    [https://arxiv.org/abs/2609.38561](https://arxiv.org/abs/2609.38561)

    本文提出POEM-CMA，一种通过协方差矩阵自适应实现各向异性采样并引入有效维度概念的无参数零阶优化方法，将采样集中于信息量最大的方向，并以问题的内在维度取代环境维度进行复杂度分析。

    

    零阶优化方法对于求解梯度信息不可用或计算代价高昂的黑盒问题至关重要。本文提出了POEM-CMA，一种新颖的无参数随机零阶算法，它通过整合协方差矩阵对齐和有效维度的概念，对近期的POEM方法进行了扩展。与依赖各向同性随机方向的传统零阶方法不同，POEM-CMA通过从梯度估计构建协方差矩阵来执行各向异性采样。这使得算法能够将采样工作集中在信息量最大的方向上。我们引入了经验有效维度 $d^* = \frac{\operatorname{tr}(\hat{\Sigma})}{\lambda_{\max}(\hat{\Sigma})}$ 的概念，它反映了问题的内在维度，并在采样和复杂度分析中取代了环境维度。我们证明POEM-CMA实现了近……（摘要原文在此处被截断）

    arXiv:2609.38561v1 Announce Type: cross  Abstract: Zeroth-order optimization methods are essential for solving black-box problems where gradient information is unavailable or expensive to compute. This paper presents POEM-CMA, a novel parameter-free stochastic zeroth-order algorithm that extends the recent POEM method by integrating covariance matrix alignment and the notion of effective dimension.   In contrast to traditional zeroth-order approaches that rely on isotropic random directions, POEM-CMA performs anisotropic sampling by constructing a covariance matrix from gradient estimates. This enables the algorithm to focus sampling efforts on the most informative directions. We introduce the use of the empirical effective dimension $d^* = \frac{\operatorname{tr}(\hat{\Sigma})}{\lambda_{\max}(\hat{\Sigma})}$, which reflects the intrinsic dimensionality of the problem and replaces the ambient dimension in both sampling and complexity analysis.   We prove that POEM-CMA achieves a near-o
    
[^351]: Diffusion-2BC：面向自动驾驶离线行为克隆的扩散与回归混合训练

    Diffusion-2BC: Hybrid Diffusion and Regression Training for Offline Behavior Cloning in Autonomous Driving

    [https://arxiv.org/abs/2609.38472](https://arxiv.org/abs/2609.38472)

    本文提出 Diffusion-2BC，在共享视觉编码器上将扩散去噪目标与辅助的确定性行为克隆损失联合训练（辅助分支仅用于训练、推理仍基于扩散），从而解决了自动驾驶离线行为克隆中单观测多有效动作的多模态问题并提升了闭环性能的稳定性。

    

    行为克隆为自动驾驶策略学习提供了一条离线途径，但均方误差回归与“单个观测对应多个有效动作”的演示数据并不匹配。扩散策略虽然能够表示条件多模态动作分布，但当视觉特征与控制从有限数据中学习时，其闭环性能可能不稳定。本文提出了 Diffusion-2BC，该方法在共享视觉编码器之上，将扩散去噪目标与一个辅助性的确定性行为克隆损失相结合。辅助分支仅在训练阶段使用，推理阶段仍基于扩散方式进行。该方法在受控的 Claw 环境以及鸟瞰视角的 CARLA 导航任务中进行了评估，包括基于路线条件的驾驶、无路线的多路口导航，以及从 Town01 到 Town02 的跨地图评估。在 Claw 任务中，Diffusion-2BC 降低了平均……（摘要原文在此处截断）

    arXiv:2609.38472v1 Announce Type: cross  Abstract: Behavior cloning provides an offline route to autonomous-driving policy learning, but mean-squared-error regression is poorly matched to demonstrations in which one observation admits several valid actions. Diffusion policies can represent conditional multimodal action distributions, yet their closed-loop performance may be unstable when visual features and control are learned from limited data. This paper presents Diffusion-2BC, which combines a diffusion denoising objective with an auxiliary deterministic behavior-cloning loss over a shared visual encoder. The auxiliary branch is used only during training; inference remains diffusion-based. The proposed method is evaluated in the controlled Claw environment and in bird's-eye-view CARLA navigation, including route-conditioned driving, route-free navigation through multiple intersections, and cross-map evaluation from Town01 to Town02. In the Claw task, Diffusion-2BC reduced the mean m
    
[^352]: CompOrca：语料库规模的指令微调数据合规性标注

    CompOrca: Corpus-Scale Compliance Labelling of Instruction-Tuning Data

    [https://arxiv.org/abs/2609.37807](https://arxiv.org/abs/2609.37807)

    该论文提出了 CompOrca，利用开源大模型评判器对整个 OpenOrca 语料库（超过 420 万条样本）进行五次独立判定，首次实现了语料库规模的合规性标注，并发布带投票计数的标注结果，以支持对拒答与不服从行为的研究。

    

    研究微调如何塑造拒答与不服从行为，需要识别出那些拒绝、规避或以其他方式未能完成所请求任务的训练样本。但现有的标注最多只覆盖几千条提示词的评估集。我们提出了 CompOrca，对包含 4,233,923 条样本的 OpenOrca 语料库整体进行了合规性标注。每个样本都由一个开源权重的大语言模型评判器（LongCat-2.0，1.6 万亿参数）经过五次独立判定，被分类为合规或不合规，并将语料库发布为一致合规（94.75%）、一致不合规（1.28%）和非一致行（3.97%）三个部分，同时附带原始投票计数。单次判定会将语料库中 2.7-3.2% 的样本标记为不合规，而只有 1.28% 被全部五次判定标记，这使得过滤最模糊的样本成为可能。在 450 个经人工标注的样本（其中 150 个被标注了两次，人与人之间 κ = 0.93）上，一致合规与不合规……（原文摘要至此截断）

    arXiv:2609.37807v1 Announce Type: new  Abstract: Studying how fine-tuning shapes refusal and noncompliance behaviour requires identifying training examples that refuse, evade or otherwise fail to fulfil the requested task. But existing annotation covers evaluation sets of a few thousand prompts at most. We present CompOrca, a compliance labelling over the entirety of the 4,233,923-example OpenOrca corpus. Every example was classified as compliant or noncompliant by five independent passes of an open-weight LLM judge (LongCat-2.0, 1.6T parameters), and the corpus is released as unanimous compliance (94.75%), unanimous noncompliance (1.28%), and nonunanimous rows (3.97%) along with the raw vote counts. A single pass flags 2.7-3.2% of the corpus as noncompliant, while only 1.28% is flagged by all five, allowing for filtering the most ambiguous samples. Against 450 human-annotated examples, 150 of them annotated twice (human-human $\kappa = 0.93$), the unanimous compliance and noncomplianc
    
[^353]: MA-JEPA：面向多智能体强化学习的联合嵌入世界模型

    MA-JEPA: Joint-Embedding World Models for Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.33563](https://arxiv.org/abs/2609.33563)

    MA-JEPA提出了一种基于联合嵌入预测（JEPA）架构的随机世界模型，用预测目标表示替代观测重构，实现了基于模型的集中训练、分散执行的多智能体强化学习。

    

    世界模型通过在想象轨迹上训练策略来提升样本效率，但其有效性取决于所学习的表示能否捕获未来控制所需的信息。我们研究自监督联合嵌入预测（JEPA）能否为多智能体强化学习提供这种学习信号。我们提出MA-JEPA，这是一种随机世界模型，它用对目标表示的预测取代观测重构，从而实现基于模型的、采用集中式训练与分散式执行的多智能体强化学习。一个分类潜在状态和一个因果Transformer通过后验预测和以动作为条件的动力学预测目标进行训练，随后用于从潜在想象中进行actor-critic学习。一个仅在训练阶段使用的联合预测器以所有智能体的局部状态和动作为条件，预测每个智能体下一个局部观测的嵌入。这些预测……

    arXiv:2609.33563v2 Announce Type: replace-cross  Abstract: World models improve sample efficiency by training policies on imagined trajectories, but their usefulness depends on learning representations that capture the information needed for future control. We study whether self-supervised joint-embedding prediction (JEPA) can provide this learning signal for multi-agent reinforcement learning. We introduce MA-JEPA, a stochastic world model that replaces observation reconstruction with prediction of target representations, enabling model-based multi-agent reinforcement learning with centralized training and decentralized execution. A categorical latent state and a causal Transformer are trained with posterior and action-conditioned dynamics prediction objectives and are then used for actor-critic learning from latent imagination. A training-only joint predictor conditions on all agents' local states and actions to predict each agent's next local observation embedding. These predictions
    
[^354]: SMAT：简单高效的合并感知训练

    SMAT: Simple and Efficient Merge-Aware Training

    [https://arxiv.org/abs/2609.33437](https://arxiv.org/abs/2609.33437)

    SMAT将常见模型合并操作抽象为缩放、掩码和扰动三种基本操作，通过在采样生成的模拟合并参数上联合优化专家损失与期望损失，实现了以极小训练开销显著提升合并后性能的简单高效合并感知训练方法。

    

    模型合并能够在无需联合重新训练的情况下整合多个专家模型的能力，但标准的专家训练仅优化任务损失，无法保证合并后的良好性能。合并感知训练（MAT）旨在提升合并后的性能，但现有方法未能充分考虑常见的合并操作，且会增加训练成本。我们观察到，从专家模型的角度来看，常见的合并方法可以由三种操作来描述：Scale（缩放）对其自身更新进行重新加权，Mask（掩码）移除选定的坐标，Perturb（扰动）添加来自其他专家的更新。基于这一视角，我们提出了SMAT（简单合并感知训练），它通过采样缩放系数、掩码和加性噪声生成模拟的合并参数，并在这些参数上联合优化专家损失与期望损失。我们进一步引入周期性调度、核融合和参数存储切换机制，使SMAT高效运行，每步仅需一次前向传播和一次反向传播。

    arXiv:2609.33437v2 Announce Type: replace-cross  Abstract: Model merging integrates the capabilities of multiple experts without joint retraining, but standard expert training optimizes task loss alone and does not guarantee good performance after merging. Merge-aware training (MAT) aims to improve merged performance, but existing methods do not fully account for common merging operations and add training cost. We observe that, from an expert's perspective, common merging methods can be described by three operations: Scale reweights its own update, Mask removes selected coordinates, and Perturb adds updates from other experts. Based on this view, we introduce SMAT (Simple MAT), which jointly optimizes expert loss and expected loss at simulated merged parameters generated by sampling scaling coefficients, masks, and additive noise. We further introduce periodic scheduling, kernel fusion, and parameter storage switching to make SMAT efficient, with one forward and one backward pass per s
    
[^355]: 何时驱逐，而非保留什么：面向免训练KV缓存压缩的草稿引导驱逐方法

    When to Evict, Not What to Keep: Draft-Guided Eviction for Training-Free KV-Cache Compression

    [https://arxiv.org/abs/2609.33334](https://arxiv.org/abs/2609.33334)

    该论文提出草稿引导驱逐（DGE）方法，将KV缓存驱逐时机从预填充结束推迟到基于完整缓存起草出前两个答案token之后，通过利用答案自身前缀生成的查询来指导驱逐决策，解决了传统“优化保留什么”策略的补偿效应和选择效应失效问题，实现免训练的KV缓存压缩。

    

    诸如SnapKV、H2O和PyramidKV等免训练KV缓存压缩方法在预填充结束时驱逐token，其目标是保留未来查询预期会使用的注意力质量——即优化“保留什么”。我们证明这一目标会以两种不同的方式失效。（1）补偿效应：恢复被驱逐的注意力质量可以在注意力层面达到目标，却无法恢复任务质量。（2）选择效应：当恢复的注意力质量是碎片化的、而非集中于连贯片段时，覆盖更多真实解码查询的注意力质量反而会损害任务质量。这些失效有着共同的根本原因：驱逐发生在决定答案轨迹的查询尚未出现之时。我们提出草稿引导驱逐方法，它将驱逐时机推迟到使用完整缓存起草出前k=2个答案token之后——仅比预填充多一个解码步骤。由于草稿是由答案自身的前缀生成的，因此不会在……（摘要截断）

    arXiv:2609.33334v2 Announce Type: replace-cross  Abstract: Training-free KV-cache compression methods such as SnapKV, H2O, and PyramidKV evict tokens at the end of prefill, aiming to preserve the attention mass that future queries are expected to use -optimizing what to keep. We show that this objective fails in two distinct ways. (1) Compensation: restoring the evicted attention mass can recover the attention-level target without recovering task quality. (2) Selection: covering more of the true decode-query mass can hurt quality when the recovered mass is fragmented rather than concentrated in coherent spans. These failures share a common cause: eviction occurs before the queries that determine the answer trajectory exist. We propose Draft-Guided Eviction (DGE), which defers eviction until after drafting the first k=2 answer tokens using the full cache - just one decode step beyond prefill. Because the draft is generated from the answer's own prefix, no cache entries are discarded bef
    
[^356]: ELF-REG：将连续扩散语言模型扩展至推理任务

    ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks

    [https://arxiv.org/abs/2609.29102](https://arxiv.org/abs/2609.29102)

    提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。

    

    全连续扩散语言模型（dLMs）对连续表示进行去噪而无需中间离散化，并在最后一步并行解码所有响应token。它们在具有挑战性的推理任务上的性能表现，尚未像自回归（AR）大语言模型和掩码扩散语言模型那样得到充分验证。我们将嵌入式语言流（ELF）扩展到GSM8K、MATH-500、HumanEval和MBPP上的数学推理与代码生成任务。我们提出了ELF-REG，它通过表示对齐与纠缠（REPA+REG）来改进学习，其中冻结的AR教师模型监督中间去噪器特征，并提供一个与响应联合去噪的全局表示。ELF-REG-L在64次网络函数评估（NFE）下于GSM8K上达到55.96%的pass@1，在128次NFE下于MATH-500上达到13.39%、HumanEval上达到22.56%。它在GSM8K和代码任务上的pass@1优于所评估的同规模扩散语言模型，并将MATH-500的pass@1从……（摘要原文在此处被截断）

    arXiv:2609.29102v1 Announce Type: new  Abstract: Fully continuous diffusion language models (dLMs) denoise continuous representations without intermediate discretization, then decode all response tokens in parallel at the final step. Their performance on challenging reasoning tasks remains less established than that of autoregressive (AR) LLMs and masked dLMs. We scale Embedded Language Flows (ELF) to mathematical reasoning and code generation on GSM8K, MATH-500, HumanEval, and MBPP. We introduce ELF-REG, which improves learning with representation alignment and entanglement (REPA+REG), where a frozen AR teacher supervises intermediate denoiser features and supplies a global representation that is jointly denoised with the response. ELF-REG-L achieves 55.96% pass@1 on GSM8K at 64 network function evaluations (NFE), and 13.39% on MATH-500 and 22.56% on HumanEval at 128 NFE. It outperforms the evaluated comparable-scale dLMs in pass@1 on GSM8K and code, and improves MATH-500 pass@1 from 
    
[^357]: 后处理公平性约束何时有效、何时有害：来自八项跨领域评估的证据

    When Post-Processing Fairness Constraints Help and When They Harm: Evidence from Eight Cross-Domain Evaluations

    [https://arxiv.org/abs/2609.26955](https://arxiv.org/abs/2609.26955)

    该论文提出FAPE四阶段公平性审计框架，通过对八个领域的评估发现，后处理公平性干预（Fairlearn的ThresholdOptimizer）的有效性取决于基线差异大小——在多数高差异场景中能改善公平性，但在低差异场景中可能反而有害。

    

    生产环境中的机器学习公平性审计通常只在部署时进行一次，且仅针对单一领域。这两种做法在实践中都会失效：公平性可能会在重新训练或用户群体变化后发生偏移，而在某一数据集上验证过的干预措施很少会在组织实际部署的异构领域中进行测试。我们提出了FAPE（面向生产环境的公平性审计），这是一个四阶段框架，用于评估单一后处理干预措施——Fairlearn的ThresholdOptimizer——在八项领域评估中的表现：刑事司法、收入预测、法律录取、信贷放贷、农业贷款、多领域基准语料库、医疗健康和教育。每项评估均以人口统计均等性和均等化赔率差异进行评分，并在可计算的情况下加入差别影响比率和准确率成本。干预措施的有效性与基线差异幅度密切相关：在各模型-领域组合中，该约束在14个高差异案例中的9个改善了差异表现，而在……

    arXiv:2609.26955v1 Announce Type: new  Abstract: Fairness audits in production ML typically occur once, at deployment, on a single domain. Both fail in practice: fairness can shift after retraining or a changing user base, and interventions validated on one dataset are rarely tested across the heterogeneous domains an organization deploys. We present FAPE (Fairness Auditing for Production Environments), a four-stage framework evaluating a single post-processing intervention, Fairlearn's ThresholdOptimizer, across eight domain evaluations: criminal justice, income prediction, legal admissions, credit lending, agricultural lending, a multi-domain benchmark corpus, healthcare, and education. Each is scored on demographic parity and equalized odds difference, plus disparate impact ratio and accuracy cost where computable. Intervention effectiveness tracks baseline disparity magnitude: across model-domain pairs the constraint improved disparity in 9 of 14 high-disparity cases and worsened i
    
[^358]: MemCalib：LLM智能体中记忆使用的基准测试与优化

    MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents

    [https://arxiv.org/abs/2609.24259](https://arxiv.org/abs/2609.24259)

    该论文提出了基于现实记忆系统场景的MemCalib基准，揭示前沿LLM普遍存在过度或不足使用记忆的问题，并针对后训练算法改进的单向性缺陷提出了MemCalib-RL来优化智能体的记忆使用能力。

    

    智能体记忆的有效性最终取决于底层LLM是否为上下文中的每条记忆赋予对其响应的适当影响程度。然而，这一能力在很大程度上一直被忽视。为了评估这一能力，我们提出了MemCalib，这是一个基于现实记忆系统场景构建的基准，用于评估记忆使用并推动优化算法的发展。在MemCalib测试集上的结果表明，前沿的开源和闭源模型都难以恰当地使用记忆。它们经常过度使用或使用不足记忆，而不是使每个命题的实际使用水平与其目标水平相匹配，从而导致有偏差的、低质量的响应。使用常见后训练算法（包括组相对策略优化和在线策略自蒸馏）的实验进一步揭示了明显的方向性偏差：经过训练的模型在一个方向上有所改进，而在另一个方向上却出现退化。因此，我们提出了MemCalib-RL，

    arXiv:2609.24259v1 Announce Type: new  Abstract: The effectiveness of agent memory ultimately depends on whether the underlying LLM gives each memory in context an appropriate degree of influence over its response. Yet this capability has remained largely overlooked. To assess this capability, we introduce MemCalib, a benchmark grounded in realistic memory-system scenarios for evaluating memory use and advancing optimization algorithms. Results on the MemCalib test set reveal that frontier open- and closed-source models struggle to use memory appropriately. They frequently over-use or under-use memory rather than matching each proposition's actual use to its target level, leading to biased, low-quality responses. Experiments with common post-training algorithms, including group relative policy optimization and on-policy self-distillation, further reveal a clear directional skew: trained models improve in one direction while deteriorating in the other. We therefore propose MemCalib-RL, 
    
[^359]: CTRL：基于控制的时间序列预测与LLM引导的残差学习

    CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning

    [https://arxiv.org/abs/2609.23257](https://arxiv.org/abs/2609.23257)

    CTRL框架将语义推理与定量预测解耦，利用LLM智能体作为控制器分析预测误差的分解成分并输出控制信号，再由轻量级残差解码器转化为预测修正，从而提升非平稳环境下时间序列预测的稳定性与可解释性。

    

    时间序列预测是跨多个领域关键决策的基础。尽管大语言模型（LLM）提供了有前景的推理能力，但现有的基于LLM的时间序列预测方法要么将其简化为绕过其优势的数值预测器，要么允许直接生成预测，导致在非平稳环境中预测不稳定。我们提出了CTRL，一个将语义推理与定量预测解耦的框架。冻结的主干模型生成基础预测，而专门的LLM智能体充当控制器，通过分解的趋势、季节性和不规则成分来分析主干预测误差，将推理建立在可解释的时间结构之上。每个智能体输出紧凑的控制信号，由轻量级残差解码器将其转化为预测修正。CTRL还集成了无标签的测试时自适应机制，可从输入统计信息中检测分布偏移。

    arXiv:2609.23257v1 Announce Type: cross  Abstract: Time series forecasting underpins critical decision-making across diverse domains. While large language models (LLMs) offer promising reasoning capabilities, existing LLM-based time series forecasting approaches either reduce them to numerical predictors that bypass their strengths, or allow direct forecast generation that destabilizes predictions in non-stationary settings. We introduce CTRL, a framework that decouples semantic reasoning from quantitative prediction. A frozen backbone generates base forecasts, while specialized LLM agents function as controllers that analyze backbone prediction errors through decomposed trend, seasonal, and irregular components, grounding reasoning in interpretable temporal structure. Each agent outputs compact control signals that a lightweight residual decoder translates into forecast corrections. CTRL incorporates label-free test-time adaptation that detects distribution shift from input statistics
    
[^360]: 嵌入模型以奇特的方式进行测量

    Embedding Models Measure in Peculiar Ways

    [https://arxiv.org/abs/2609.20821](https://arxiv.org/abs/2609.20821)

    该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。

    

    嵌入空间定义了语义相似性和距离的概念。我们研究了这些嵌入是否反映了质量、距离、时间和体积等物理测量，这些物理测量具有唯一且客观的语义等价与距离概念。我们发现物理测量在嵌入空间中仅被微弱地建模，相反，可以观察到相当奇特的测量模式。进一步的分析表明，物理测量的嵌入表示受到表面字符串相似性的强烈影响，而重新校准相似性并不能实质性改善对齐效果。

    arXiv:2609.20821v1 Announce Type: new  Abstract: Embedding spaces define notions of semantic similarity and distance. We study whether those embeddings reflect physical measurements of mass, distance, time and volume, which admit a unique, objective notion of semantic equivalence and distance. We find that physical measurement is only weakly modeled in the embedding space, and that instead quite peculiar measurement patterns can be observed. Further analysis indicates that embedding representations of physical measurements are strongly influenced by superficial string similarity, and recalibration of similarity does not substantially improve the alignment.
    
[^361]: 无标签引导：将测试时强化学习压缩至仅偏置参数子空间

    Label-free steering: Compressing test-time reinforcement learning into bias-only subspaces

    [https://arxiv.org/abs/2609.18587](https://arxiv.org/abs/2609.18587)

    该论文提出无标签仅偏置测试时强化学习方法，以多数投票伪标签为奖励、仅优化约10万个偏置参数，即可在数学、视觉语言和音频推理等多个任务上达到与全参数方法相当甚至更优的性能。

    

    测试时强化学习（TTRL）使模型能够在不依赖标注训练数据的情况下改进其推理能力，但现有方法通常需要优化模型的大部分参数。这引出一个自然的问题：当奖励信号和优化空间都受到严格限制时，有效的测试时自适应是否仍然可能出现？我们用无标签仅偏置TTRL（label-free bias-only TTRL）回答了这一问题，该方法使用多数投票伪标签作为奖励，仅优化约10万个偏置参数，同时保持预训练主干网络完全冻结。在MATH-500上，我们的方法达到76.67%的准确率，略高于我们自行复现的带标签偏置引导方法，同时比全参数TTRL少优化76,000倍的参数。相同的训练过程还能在视觉语言和音频推理任务上提升性能，包括MathVista、AI2D、LogicVista和MMAU。我们进一步表明，学习到的引导向量（摘要在此处被截断）

    arXiv:2609.18587v1 Announce Type: cross  Abstract: Test-time reinforcement learning (TTRL) enables models to improve their reasoning without relying on labeled training data, but existing approaches typically optimize a large fraction of the model parameters. This raises a natural question: can effective test-time adaptation emerge when both the reward signal and the optimization space are severely restricted? We answer this question with label-free bias-only TTRL, which uses majority-vote pseudo-labels as rewards and optimizes only approximately 100K bias parameters while keeping the pretrained backbone frozen. On MATH-500, our approach reaches 76.67% accuracy, slightly exceeding our own labeled bias-steering reproduction while optimizing 76,000x fewer parameters than full-parameter TTRL. The same training procedure improves performance across vision-language and audio reasoning tasks, including MathVista, AI2D, LogicVista, and MMAU. We further show that the learned steering vectors t
    
[^362]: Salesforce Koa：一个面向智能体工具使用的企业级语言模型

    Salesforce Koa: An Enterprise Language Model for Agentic Tool Use

    [https://arxiv.org/abs/2609.15066](https://arxiv.org/abs/2609.15066)

    Salesforce Koa 是一个基于 Nemotron-3-Super-120B 并通过 GRPO 强化学习后训练的企业级语言模型，其核心创新在于“仿真到奖励”流水线——将工作流规范扩展为以角色为条件的多轮任务，并以成功工具使用作为任务解决奖励，从而在保持通用性能的同时显著提升智能体工具使用能力。

    

    我们介绍了 Salesforce Koa，一个通过对开放权重的 Nemotron-3-Super-120B 基础模型进行后训练，并采用基于群体相对策略优化（GRPO）的强化学习而构建的企业级语言模型。Salesforce Koa 在公开数据和合成生成的数据上训练，不使用任何客户数据，旨在提升工具使用和智能体能力，同时保持强大的通用性能。其独特的组成部分是一个“仿真到奖励”流水线，该流水线将工作流规范扩展为以角色为条件的多轮任务，并以成功的工具使用为基础，为数据依赖型请求提供任务解决奖励。在企业领域，这些规范采用 Agent Script 编写，即 Salesforce 用于构建 Agentforce 智能体的声明式语言；而在公共工具使用领域，我们直接合成工作流结构。相同的仿真和基于真实结果的奖励机制驱动 GRPO 在两类场景中运行。在公共工具使用、智能体……

    arXiv:2609.15066v1 Announce Type: cross  Abstract: We present Salesforce Koa, an enterprise language model built by post-training the open-weight Nemotron-3-Super-120B foundation model with reinforcement learning using Group Relative Policy Optimization (GRPO). Salesforce Koa is trained on public and synthetically generated data, with no customer data, to improve tool use and agentic capabilities while preserving strong general-purpose performance. Its distinctive component is a simulation-to-reward pipeline that expands workflow specifications into persona-conditioned multi-turn tasks with task-resolution rewards grounded in successful tool use for data-dependent requests. For enterprise domains, these specifications are written in Agent Script, Salesforce's declarative language for building Agentforce agents; for public tool-use domains, we synthesize the workflow structure directly. The same simulation and grounded-reward machinery drives GRPO across both. Across public tool-use, ag
    
[^363]: 免数据的在线策略蒸馏

    Data-free On-policy Distillation

    [https://arxiv.org/abs/2609.14193](https://arxiv.org/abs/2609.14193)

    该研究发现在线策略蒸馏（OPD）对训练数据几乎不敏感——仅八个提示即可媲美1.7万题的数据集，且跨领域数据仍能保留90%以上的收益，表明OPD传递的是教师的推理方式而非具体知识。

    

    在线策略蒸馏已成为前沿后训练流水线的标准组成部分，但其训练数据实际贡献了多少却基本未被检验。在实践当中最常见的两组师生模型配对上，我们发现OPD对其数据几乎不敏感：仅八个提示就能与一个包含1.7万道题目的数据集效果相当，而且三个独立构建、难度和师生KL散度相差数倍的数据集，产生了几乎难以区分的训练曲线。有两个原因可以解释这一现象。第一，OPD中数据的单位是一个提示所引导的状态，而非提示本身：随着采样的持续进行，单个提示会不断暴露新的教师纠正信号，而增加提示的边际价值在八个之后便急剧坍缩。第二，用竞赛编程替代数学领域仍能恢复超过百分之九十的域内收益，这表明OPD传递的是教师的推理方式，而非知识本身（原文在此处截断）。

    arXiv:2609.14193v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has become a standard component of frontier post-training pipelines, yet how much its training data actually contributes has gone largely unexamined. On the two teacher-student pairings most common in practice, we find OPD almost indifferent to its data: eight prompts already match a 17k-problem dataset, and three independently built datasets whose difficulty and teacher-student KL differ several-fold produce nearly indistinguishable training curves. Two causes account for this. First, the unit of data in OPD is the state a prompt leads to, not the prompt itself: a single prompt keeps exposing new teacher correction as sampling continues, while the marginal value of additional prompts collapses after eight. Second, replacing mathematics with competitive programming still recovers over ninety percent of the in-domain gain, indicating that OPD transfers the teacher's mode of reasoning rather than knowledge re
    
[^364]: 真理从未消失：顺从情境下真值探测器的完美混叠现象

    The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes

    [https://arxiv.org/abs/2609.10739](https://arxiv.org/abs/2609.10739)

    论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。

    

    在真实汇报与任务既定行为相重合的情境中拟合的真值探测器，仅凭其拟合标签无法区分这两个目标。我们将这种语义识别的失效称为“完美混叠”（perfect aliasing）。在一个受控的二值汇报博弈中，在顺从情境上拟合的真值探测器与既定行为探测器求解的是同一个优化问题。在对立情境上，二者的标签互为补集，导致它们的AUROC之和恒为一；这一恒等关系在751个单元-层对上以浮点精度成立。我们利用随机化码本将既定输出符号与语义行为分离，进而通过在顺从情境与对立情境的混合数据上拟合，将真值与既定行为区分开来。对于一个经奖励训练、在所有评估的对立试验中均给出虚假回答的Gemma-2-9B策略，常规探测器在三个训练种子上的AUROC仅为0.006 ± 0.005，而混合拟合探测器在相同的保留激活值上得分达到1.000。混合拟合……

    arXiv:2609.10739v1 Announce Type: cross  Abstract: A truth probe fitted where truthful reporting and a task's prescribed action coincide cannot distinguish those targets from its fitting labels alone. We call this failure of semantic identification perfect aliasing. In a controlled binary reporting game, truth and prescribed-action probes fitted on compliant contexts solve the same optimization. On rival contexts their labels are complements, forcing their AUROCs to sum to one; this identity holds across 751 cell-layer pairs to floating-point precision. We separate prescribed output symbols from semantic action using randomized codebooks, then separate truth from prescribed action by fitting on mixed compliant and rival contexts. For a reward-trained Gemma-2-9B policy that answers falsely on all evaluated rival trials, the conventional probe scores $0.006 \pm 0.005$ AUROC across three training seeds, while mixed-fit probes score $1.000$ on the same held-out activations. Mixed fitting u
    
[^365]: MoEMB：利用高效混合专家模型扩展通用多模态嵌入

    MoEMB: Scaling Universal Multimodal Embeddings with Efficient Mixture-of-Experts Models

    [https://arxiv.org/abs/2609.08663](https://arxiv.org/abs/2609.08663)

    MoEMB 提出利用混合专家沿专家维度扩展通用多模态嵌入模型，在保持单向量、非自回归编码与低延迟推理的同时提升编码器容量，并避免对简单任务产生冗余计算。

    

    通用多模态嵌入（UME）日益要求编码器具备处理更广泛任务和模态的能力，且这些任务和模态的复杂度不断提升。先前的扩展方法要么增加表示尺寸、检索开销，要么将编码器扩展为笨重的多模态大语言模型（LLM）。近期工作（如 Think-Then-Embed，TTE）探索了通过推理标记进行扩展的路径。然而，嵌入模型难以扩展：增加参数会与对比学习所需的大训练批量直接产生权衡，且检索服务必须在严格的延迟约束下完成。此外，UME 任务在复杂度上差异很大，盲目扩大嵌入模型规模会带来显著的冗余计算。在本工作中，我们提出 MoEMB，通过混合专家沿专家轴扩展 UME，在保持单向量、非自回归编码的同时增长编码器容量。通过对设计空间的系统性研究……

    arXiv:2609.08663v1 Announce Type: cross  Abstract: Universal multimodal embedding (UME) increasingly demands encoder's capacity for handling a broad range of tasks and modalities with increased complexity. Prior scaling methods either increase the representation size, retrieval effort, or scales the encoder into a heavy multimodal LLM. Recent works, such as Think-Then-Embed (TTE), explore scaling via reasoning tokens. However, embedding models are hard to scale up: increasing parameters directly tradeoffs for the large training batch size that contrastive learning needs, and retrieval has to be served under tight latency. Moreover, UME tasks are diverse in complexity, where scaling up embedders can bring significant redundant computation. In this work, we propose MOEMB, which instead scales UME along the expert axis through mixture-of-experts (MoE), growing encoder capacity while preserving single-vector, non-autoregressive encoding. Through a systematic study of the design space and t
    
[^366]: 大语言模型的风险条件化微调

    Risk-Conditioned Fine-Tuning of Large Language Models

    [https://arxiv.org/abs/2609.08064](https://arxiv.org/abs/2609.08064)

    提出风险条件化RLHF框架，通过训练单一策略提供连续可调的风险控制接口，使用户能够在推理时灵活选择不同的风险规避程度，而无需重新训练或部署多个特定风险的模型。

    

    大语言模型（LLMs）越来越多地被部署在罕见但严重的有害生成内容可能造成重大后果的场景中。现有的风险规避RLHF通过优化条件风险价值（CVaR）来解决这一问题，但其针对固定的风险水平训练策略，因此无法在推理时调整所需的风险规避程度。在本文中，我们提出了风险条件化RLHF，这是一个训练单一策略的框架，该策略提供了连续的风险控制接口，使用户无需重新训练或部署多个针对特定风险的模型即可选择不同的风险规避程度。在多个基准测试上的实验表明，单一的风险条件化策略能够在推理时适应不同的风险水平，从而实现更灵活、更具风险感知能力的大语言模型部署。

    arXiv:2609.08064v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly deployed in settings where rare but severe harmful generations can have significant consequences. Existing Risk-Averse RLHF addresses this issue by optimizing Conditional Value-at-Risk (CVaR), but it trains policies for fixed risk levels and therefore cannot adjust the desired degree of risk aversion at inference time. In this paper, we propose risk-conditioned RLHF, a framework that trains a single policy that provides a continuous risk-control interface, enabling users to select different degrees of risk aversion without retraining or deploying multiple risk-specific models. Experiments across multiple benchmarks demonstrate that a single risk-conditioned policy can adapt to different risk levels at inference time, enabling more flexible and risk-aware LLM deployment.
    
[^367]: 基于家族DIF指导的基准重组下，接近持平的大语言模型排名是否稳健？

    Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?

    [https://arxiv.org/abs/2609.00482](https://arxiv.org/abs/2609.00482)

    该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。

    

    排行榜上的微小差距常被解读为某个语言模型优于另一个的证据，但其结论方向可能取决于包含哪些基准题目。我们利用五个基准的题目级响应数据以及一种无家族标签的谱近似多维项目反应理论（MIRT）来检验这一点。在所有者不相交的折中划分下，一半所有者数据用于识别跨模型家族具有低残差差异项目功能（低DIF）的题目；由此得到的固定且按来源和难度平衡的权重用于对另一半数据中的模型进行评分，同时使用等长的匹配随机子测试来控制一般性的子测试变异。全基准排名与低DIF排名保持强相关（τb=.900-.948）。然而，在五个基准中的四个里，最初相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，比匹配随机子测试的中位数高出16.9-28.6个百分点（均为p=.001）。第五个基准[摘要截断]

    arXiv:2609.00482v1 Announce Type: new  Abstract: Small leaderboard gaps are often interpreted as evidence that one language model is better than another, but their sign may depend on which benchmark items are included. We test this using item-level responses from five benchmarks and a family-label-free spectral approximation to multidimensional item-response theory (MIRT). In owner-disjoint folds, one owner half identifies items with low residual differential item functioning across model families (low-DIF); the resulting frozen, source- and easiness-balanced weights score models in the other half, while equally short matched-random subtests control for generic subtest variation. Full-benchmark and low-DIF rankings remain strongly correlated ($\tau_b=.900$--$.948$). Yet in four of five benchmarks, 30.9--47.1\% of cross-family pairs initially within one percentage point reverse order, exceeding their matched-random medians by 16.9--28.6 percentage points (all $p=.001$). The fifth benchm
    
[^368]: LM-X：面向通用机器人操作的可解释动作建模——集成进度、事件与不确定性预测

    LM-X: Explainable Action Modeling with Progress, Event, and Uncertainty Prediction for Generalist Robot Manipulation

    [https://arxiv.org/abs/2608.25757](https://arxiv.org/abs/2608.25757)

    本文提出LM-X框架，通过在线预测任务进度、事件转换和局部不确定性三个显式信号，使VLA策略的动作生成具有内在可解释性，无需事后解释。

    

    arXiv:2608.25757v1 公告类型：交叉 摘要：通用视觉-语言-动作（VLA）策略主要通过短视距动作预测来学习长期行为，并且除了采样的命令外几乎不透露其他信息。这造成了两个相互关联的瓶颈：单一的动作目标必须隐含地吸收任务进度、中间意图和局部可靠性，而这些控制状态在执行过程中仍然隐藏。受生物感觉运动控制的功能原理启发，我们引入了LM-X，它在任务、事件和运动尺度上组织预测，而不声称解剖学对应关系。三个显式监督信号在线发出并直接调节动作生成：返回到目标（RTG）衡量可见任务进度，事件到目标（ETG）识别下一个语义转换，以及异方差动作流通过传播方差估计局部可靠性。因此，解释是控制的内在部分，而非事后生成。

    arXiv:2608.25757v1 Announce Type: cross  Abstract: Generalist vision--language--action (VLA) policies learn long-horizon behavior mainly through short-horizon action prediction and reveal little beyond sampled commands. This creates two coupled bottlenecks: a single action target must implicitly absorb task progress, intermediate intent, and local reliability, while these control states remain hidden during execution. Inspired by functional principles of biological sensorimotor control, we introduce LM-X , which organizes prediction across task, event, and motor scales without claiming anatomical correspondence. Three explicitly supervised signals are emitted online and directly condition action generation: return-to-go (RTG) measures visible task progress, event-to-go (ETG) identifies the next semantic transition, and heteroscedastic action flow estimates local reliability through propagated variance. Explanation is therefore intrinsic to control rather than generated post hoc. Before
    
[^369]: 复杂神经网络下降的Kähler景观与包括Calabi-Yau流形搜索与消灭的保证

    K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold

    [https://arxiv.org/abs/2608.19584](https://arxiv.org/abs/2608.19584)

    本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。

    

    我们研究复参数化网络的景观。我们的方法受参数的信息论流形视角以及经典优化保证的启发，尽管涉及复杂几何变体，如通过Dolbeault渐近。下降路径在交叉熵下承认Kähler信息度量，通过Wirtinger Hessian作用于对数似然势。我们关注一种下降更新规则，采用自然梯度下降，通过逆度量缩放的微分损失，使下降路径保持在全纯切丛中。我们强调Calabi-Yau信息流形，这些流形通过不良曲率条件的景观提供了理论保证。在Calabi-Yau度量下，特别是在非紧致设置中，具有全局势而非调用Calabi猜想的拓扑要求，一个楔入无处消失的全纯...

    arXiv:2608.19584v1 Announce Type: new  Abstract: We study landscapes for complex-parameterized networks. Our approach is motivated with an information-theoretic manifold perspective of the parameter and via classical optimization guarantees although of complex geometric variety such as through Dolbeault asymptotics. The descent path admits a K\"ahler information metric under a cross-entropy via the Wirtinger Hessian on the log-likelihood potential. We restrict attention to a descent update rule with natural gradient descent via a differentiated loss scaled by the inverse metric, so the descent path remains in the holomorphic tangent bundle. We emphasize Calabi-Yau information manifolds which profane theoretical guarantees via an ill-curvature-conditioned landscape. Under a Calabi-Yau metric, specifically in a non-compact setting with a global potential so defined geometrically rather than invoking the topological requirements of the Calabi conjecture, a wedged nowhere-vanishing holomor
    
[^370]: 量子多臂老虎机与线性老虎机：下界与算法

    Quantum Multi-Armed Bandits and Linear Bandits: Lower Bounds and Algorithms

    [https://arxiv.org/abs/2608.14319](https://arxiv.org/abs/2608.14319)

    本文首次证明了量子多臂老虎机和有限动作量子线性老虎机的极小极大遗憾下界，解决了是否存在与时间范围无关的遗憾问题，并提供了基于多项式方法和Remez型不等式的新证明技术。

    

    arXiv:2608.14319v1 公告类型：新 摘要：我们在Wan等人[2023]的模型下研究量子多臂老虎机（QMAB）和量子线性老虎机（QLB），其中学习者通过量子奖励预言机或其逆来查询每个臂或动作。先前的工作给出了在时间范围$T$上的算法，对于具有$K$个臂的QMAB，遗憾为$O(K\log T)$，对于$d$维的QLB，遗憾为$O(d^2\operatorname{polylog} T)$。这留下了$K\log T$规模是否不可避免以及$d^2$依赖是否可以改进的问题。我们证明了第一个极小极大下界，对于QMAB为$\Omega(K\log(T/K))$，对于有限动作QLB为$\Omega(d\log(T/d))$，解决了Wan等人[2023]提出的关于是否存在与$T$无关的遗憾的问题。我们论证的核心是一个高置信度的单臂量子测试下界，用于区分固定的奖励均值与一个替代区间，通过多项式方法和三角多项式的Remez型不等式证明。

    arXiv:2608.14319v1 Announce Type: new  Abstract: We study quantum multi-armed bandits (QMAB) and quantum linear bandits (QLB) in the model of Wan et al. [2023], where the learner queries each arm or action through a quantum reward oracle or its inverse. Prior work gives algorithms over horizon $T$ with regret $O(K\log T)$ for QMAB with $K$ arms and $O(d^2\operatorname{polylog} T)$ for $d$-dimensional QLB. This leaves open whether the $K\log T$ scale is unavoidable and whether the $d^2$ dependence can be improved. We prove the first minimax lower bounds of $\Omega(K\log(T/K))$ for QMAB and $\Omega(d\log(T/d))$ for finite-action QLB, resolving the question raised by Wan et al. [2023] of whether regret independent of $T$ is achievable. At the heart of our argument is a high-confidence single-arm quantum testing lower bound for distinguishing a fixed reward mean from an interval of alternatives, proved by the polynomial method and a Remez-type inequality for trigonometric polynomials. A ba
    
[^371]: DYSANOS：生成式动态平滑无套利非参数期权曲面

    DYSANOS Generative Dynamic Smooth Arbitrage-free Non-parametric Option Surfaces

    [https://arxiv.org/abs/2608.12587](https://arxiv.org/abs/2608.12587)

    本文提出首个生成式无套利期权曲面模型DYSANOS，能够生成未来多年每日期权价格路径，并验证其优于传统隐含波动率PCA模型。

    

    本文介绍了DYSANOS，这是首个生成式市场模型，用于生成所有行权价和到期日的平滑且无静态套利的SANOS期权曲面。该模型旨在生成未来数年的每日现货和期权价格完整路径。我们提出了一种稳健且实用但相对简单的基线隐状态生成模型，形式为AR(1)模型。我们讨论了模型设置、数据管道和训练过程，并研究了动态套利的数值存在性。我们使用Option Metrics的IvyDB S&P指数数据（2020年至2025年）展示了模型性能，并将其与纯隐含波动率主成分分析模型进行了比较。

    arXiv:2608.12587v1 Announce Type: cross  Abstract: This article presents with DYSANOS the first generative market model for smooth SANOS option surfaces for all strikes and expiries which are free of static arbitrage. Our model is designed to generate entire paths of daily spot and option prices for years in the future.   We present a robust and useful if somewhat simplistic baseline hidden state generative model in the form of an AR(1) model. We discuss model setup, data pipeline, and training and investigate numerical resence of dynamic arbitrage. We illustrate model performance on Option Metrics' IvyDB S\&P Index data from 2020 to~2025 and compare it to a pure implied-vol PCA model.
    
[^372]: 基于动态时间规整的粒度球计算实现鲁棒高效的噪声标签时间序列分类

    Robust and Efficient Noisy-Label Time-Series Classification via Dynamic Time Warping Based Granular Ball Computing

    [https://arxiv.org/abs/2608.11704](https://arxiv.org/abs/2608.11704)

    DTW-GBC通过粒度球计算，在保持分类鲁棒性的同时大幅减少推理计算量，有效应对标签噪声问题。

    

    基于动态时间规整（DTW）的最近邻（NN）分类器在时间序列分类中有效，但容易受到错误标记的训练样本影响，并且在推理过程中需要大量DTW计算。我们提出了基于DTW的粒度球计算（DTW-GBC），该方法将时间上相似的训练样本组织成粒度球，并在粒度级别进行分类。我们进一步开发了两种用于DTW-GBC的粒度球构建策略。在四个具有对称标签噪声的基准数据集上的实验表明，两种DTW-GBC变体通常能缓解标签噪声引起的性能下降，同时在推理过程中所需的比较次数显著少于基于DTW的1-NN方法。这些发现表明，DTW-GBC在分类鲁棒性和推理效率之间提供了良好的平衡。

    arXiv:2608.11704v1 Announce Type: cross  Abstract: Dynamic Time Warping (DTW)-based Nearest-Neighbor (NN) classifiers are effective for time-series classification but are vulnerable to mislabeled training samples and require numerous DTW computations during inference. We propose DTW-based Granular Ball Computing (DTW-GBC), which organizes temporally similar training samples into granular balls and performs classification at the granule level. We further develop two granular-ball construction strategies for DTW-GBC. Experiments on four benchmark datasets with symmetric label noise show that the two DTW-GBC variants generally mitigate the performance degradation caused by label noise while requiring substantially fewer comparisons than DTW-based 1-NN during inference. These findings suggest that DTW-GBC provides a favorable balance between classification robustness and inference efficiency.
    
[^373]: 自适应子模型联邦学习中的容量混淆与覆盖保证

    Capacity Confounds and Coverage Guarantees in Adaptive Sub-model Federated Learning

    [https://arxiv.org/abs/2608.07157](https://arxiv.org/abs/2608.07157)

    该研究发现子模型联邦学习中基于训练更新散度估计的客户端数据异质性信号实际上被设备容量所混淆，与真实数据异质性几乎无关，从而质疑了按估计的数据异质性自适应分配子模型容量的可行性，并给出了相应的覆盖保证。

    

    子模型联邦学习允许资源受限的客户端训练全局模型的宽度缩减版本，但现有方法仅根据设备资源来分配容量。一个自然的下一步是根据每个客户端的数据异质性来分配容量——该异质性可从服务器已经观察到的更新中估计——最近的若干从训练衍生信号来确定子模型规模的方法暗示了这一方向。我们以自适应容量分配框架HAS-FL作为测试案例，探究这一步骤是否可行。首先，通过在可复现的数据划分上与真实的标签分布散度进行对照验证，我们发现基于更新散度的客户端异质性估计主要受容量而非数据支配：在图像基准任务和所有随机种子上，这些估计与设备容量呈强负相关，而一旦控制容量因素，它们与数据异质性的关联接近于零或为负。任何估计客户端统计……（摘要原文在此处截断）

    arXiv:2608.07157v2 Announce Type: replace  Abstract: Sub-model federated learning lets resource-constrained clients train width-reduced versions of a global model, but existing methods allocate capacity by device resources alone. A natural next step, allocating capacity by each client's data heterogeneity as estimated from the updates the server already observes, is suggested by recent methods that size sub-models from training-derived signals. We ask whether that step is possible, using HAS-FL, an adaptive capacity-allocation framework, as a test case. First, validated against ground-truth label-distribution divergence on reproducible partitions, update-divergence estimates of client heterogeneity are dominated by capacity rather than data: on both image benchmarks and every seed, the estimates correlate strongly and negatively with device capacity, and once capacity is controlled for their association with data heterogeneity is near zero or negative. Any method estimating client stat
    
[^374]: 消息传递究竟有多重要？面向神经网络图回归的GNN层即插即用研究

    How Much Does Message Passing Matter? A Drop-In Study of GNN Layers for Neural Network Graph Regression

    [https://arxiv.org/abs/2607.26404](https://arxiv.org/abs/2607.26404)

    该研究通过在固定架构、损失和训练方案下即插即用地替换十种消息传递层，首次系统评估了消息传递层选择对神经网络图级回归的重要影响，发现其性能差异显著且最佳选择取决于图的规模。

    

    图神经网络（GNN）被广泛用作回归器，例如用于预测神经架构的精度。然而，新的消息传递（MP）层几乎只在分类任务上进行开发和基准测试，而回归流程通常采用单一MP层且缺乏消融研究。我们探究了MP层的选择对图级回归有多大影响。在保持四个现有GNN回归器的架构、损失函数和训练方案不变的情况下，我们即插即用地替换了十种MP配置，涵盖卷积式、基于同构和基于注意力的设计。我们在十一个神经网络图数据集上对其进行评估，这些数据集中每个图的节点数从不到十个到超过一千个不等，样本量从几百到四十多万不等，并测量了秩相关、预测误差、Top-k检索、延迟和内存占用。结果表明，MP层的选择会显著改变结果，且最佳选择取决于图的规模。

    arXiv:2607.26404v2 Announce Type: replace  Abstract: Graph Neural Networks (GNNs) are widely used as regressors, for example to predict the accuracy of a neural architecture. Yet new message-passing (MP) layers are developed and benchmarked almost exclusively on classification tasks, and regression pipelines typically adopt a single MP layer without ablation. We ask how much the choice of MP layer matters for graph-level regression. Holding the architecture, loss and training recipe of four existing GNN regressors fixed, we substitute ten MP configurations spanning convolutional, isomorphism-based and attention-based designs.   We evaluate them on eleven datasets of neural-network graphs that range from under ten to over a thousand nodes per graph and from a few hundred to over four hundred thousand samples, measuring rank correlation, prediction error, top-$k$ retrieval, latency and memory. MP choice changes results substantially, and we find that the best choice depends on graph size
    
[^375]: 层神经网络是否使用了和乐？一项“测量—干预—对照”研究

    Do Sheaf Neural Networks Use Holonomy? A Measure--Intervene--Control Study

    [https://arxiv.org/abs/2607.19514](https://arxiv.org/abs/2607.19514)

    该研究通过“测量—干预—对照”实验发现，层神经网络仅在任务需要利用几何结构时（如三角形计数）才会学习到非平凡的和乐旋转，其预测确实依赖于所学到的联络，但这种几何机制并非性能优势的必要条件，因为岭回归等更简单的基线方法仍然表现更好。

    

    几何架构通常受内部机制的启发而设计，但仅凭准确率无法判断预测是否真正利用了这些机制。在层神经网络中，边上的传输构成一个联络，其沿闭环的乘积定义了和乐。我们提出三个问题：训练是否会改变三角形上的和乐？预测是否依赖于学习到的联络？和乐是否驱动了三角形计数任务？我们采用与基无关的环路读出方法，并结合恒等干预与捷径对照实验。在高同配性的 GraphUniverse 图上，三角形计数任务使神经层传播（NSP）的平均 SO(2) 三角形旋转角从 0.010 弧度增至 0.388 弧度，而社区检测任务最终仅为 0.029 弧度。在数据量更大时，学习到的 SO(2)-NSP 优于恒等 NSP，且在训练后替换其边传输会进一步增加误差。然而，岭回归的准确率更高，对角映射在不产生连续旋转的情况下也能提升性能，而定自由度模型发展出……（摘要原文在此处被截断）

    arXiv:2607.19514v3 Announce Type: replace  Abstract: Geometric architectures are often motivated by internal mechanisms, but accuracy alone does not show whether predictions use them. In Sheaf Neural Networks (SNNs), edge transports form a connection whose cycle products define holonomy. We ask whether training changes triangle holonomy, whether predictions rely on the learned connection, and whether holonomy drives triangle counting. We use basis-independent loop readouts with identity interventions and shortcut controls. On high-homophily GraphUniverse graphs, triangle counting increases the mean SO(2) triangle rotation in Neural Sheaf Propagation (NSP) from 0.010 to 0.388 radians, while community detection ends at 0.029 radians. With more data, learned SO(2)--NSP outperforms Identity NSP, and replacing its transports after training increases error further. However, ridge regression is more accurate, diagonal maps improve without continuous rotation, and fixed-degree models develop r
    
[^376]: 掩码扩散语言模型是用于智能体强化学习的强大且可引导的基于文本的世界模型

    Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL

    [https://arxiv.org/abs/2607.16204](https://arxiv.org/abs/2607.16204)

    该论文提出将基于文本的世界建模形式化为可引导的转移动力学问题，并利用掩码扩散语言模型克服自回归模型的从左到右偏差，构建了强大且可引导的世界模型，为智能体强化学习按需提供多样化、可扩展的训练环境。

    

    arXiv:2607.16204v2 公告类型：替换。摘要：强化学习（RL）的近期发展催生了对多样化、专业化训练环境的需求。手工构建的具有固定任务和奖励难度的环境，随着模型性能的提升会逐渐变成无效的训练信号，而长时程下的稀疏奖励会导致模型在特定工作流程或工具结构上出现模式崩溃。能够模拟环境状态的世界模型已能达到与纯环境交互采样相当的性能，这使其成为按需扩展任务多样性的有前景方向。然而，自回归（AR）世界模型存在从左到右的生成偏差，使其无法以全局相互依赖的状态锚点为条件，例如工具模式定义、先前的对话轮次以及预期结果。我们（i）将基于文本的世界建模形式化为一个可引导的转移动力学问题，并将其分解为初始状态、任务上下文、工具模式、领域规则和引导指令，以及（ii）整理了239,403条有真实依据的状态-动作轨迹，涵盖九个开源（原文在此处截断）。

    arXiv:2607.16204v2 Announce Type: replace  Abstract: Recent growth in reinforcement learning (RL) has surfaced a need for diverse, specialized training environments. Hand-curated environments with fixed task and reward difficulties become ineffective signals as model performance improves, and sparse rewards over long horizons induce mode collapse on specific workflows or tool structures. World models that simulate environment states have matched pure rollout performance, making them promising for scaling diversity on-demand. However, autoregressive (AR) world models suffer from a left-to-right bias preventing conditioning on globally interdependent state anchors such as tool schemas, prior turns, and expected outcomes. We (i) formalize text-based world modeling as a steerable transition-dynamics problem decomposed into initial state, task context, tool schemas, domain rules, and steering directives, and (ii) curate 239,403 grounded state-action trajectories spanning nine open-source en
    
[^377]: 超越欧几里得裁剪：通过黎曼等距策略优化克服大语言模型强化学习中的探索坍缩

    Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization

    [https://arxiv.org/abs/2607.10169](https://arxiv.org/abs/2607.10169)

    该论文揭示了PPO-Clip的根本缺陷在于使用欧几里得度量衡量策略差异、与策略黎曼流形的内在几何结构不符从而导致探索坍缩，并提出了黎曼等距策略优化（RIPO），通过在黎曼流形上保证等距策略更新来有效平衡探索与利用。

    

    强化学习（RL）已成为提升大语言模型（LLM）推理能力的主流范式。然而，采用PPO-Clip的强化学习算法本质上受到探索坍缩问题的限制。后续的研究工作主要停留在启发式方法层面，未能识别出PPO-Clip失败的根本原因。本工作揭示了PPO-Clip的根本缺陷：它隐式地使用欧几里得度量来衡量策略差异，这在理论上与策略黎曼流形上的内在几何结构不一致。这种几何不匹配导致算法在低概率区域的更新过于保守，而在高概率区域的更新过于激进，最终引发探索坍缩。为了纠正这一几何缺陷，我们提出了黎曼等距策略优化（RIPO），该方法保证策略在黎曼流形上进行等距更新，从而有效平衡探索与利用。我们进一步表明，RIPO取得了有利的（性能表现）……

    arXiv:2607.10169v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has become a dominant paradigm for enhancing LLMs' reasoning capabilities. However, RL algorithms with PPO-Clip are inherently limited by exploration collapse. Subsequent works remain primarily heuristic and fail to identify the essential cause of PPO-Clip's failure. This work reveals the fundamental flaw of PPO-Clip: it implicitly measures policy discrepancy using Euclidean metric, which is theoretically inconsistent with the intrinsic geometry on the policy Riemannian manifold. This geometric mismatch results in overly conservative updates in low-probability regions while aggressive in high-probability regions, ultimately collapsing exploration. To correct this geometric flaw, we propose Riemannian Isometric Policy Optimization (RIPO), which guarantees isometric policy updates on the Riemannian manifold, effectively balancing exploration and exploitation. We further show that RIPO achieves a favora
    
[^378]: 系统提示词条件化与四个开源权重模型中的隐藏状态几何：勘误与留存结论

    System-Prompt Conditioning and Hidden-State Geometry in Four Open-Weight Models: Corrections and What Survives

    [https://arxiv.org/abs/2607.09842](https://arxiv.org/abs/2607.09842)

    本文是对先前“系统提示词会在开源语言模型隐藏状态中留下几何指纹”这一研究的勘误版：审计发现原论文的曲率统计量、置换检验方法和若干关键测量均存在错误，本版修正这些问题并说明哪些结论仍然成立。

    

    本预印本的第1版和第2版曾报告称，指定身份的系统提示词会在四个开源权重语言模型的最后一层隐藏状态轨迹中留下几何指纹，并且指令微调会将该指纹从隐藏状态向量的方向转移到其幅值上。对其代码和数据的审计发现了以下问题。论文中描述为欧氏k-近邻图上Ollivier-Ricci曲率的统计量，实际上是由时间k-近邻边和余弦k-近邻边构建的图上的一种非标准Forman型边统计量；其发布的检验对合并后的边（而非轨迹）进行置换，且已发表的p值来自未公开的代码。论文中报告为首个生成状态范数的量，实际上是最后一个提示词位置的状态，首个输出token正是由该状态预测得出。通用控制提示词与身份提示词是在字符数上匹配，而非在token数上匹配。本版本勘正了

    arXiv:2607.09842v3 Announce Type: replace-cross  Abstract: Versions 1 and 2 of this preprint reported that an identity-specifying system prompt leaves a geometric fingerprint in the final-layer hidden-state trajectories of four open-weight language models, and that instruction tuning moves this fingerprint from the direction to the magnitude of the hidden-state vector. An audit of their code and data found the following. The curvature statistic described as Ollivier-Ricci curvature on Euclidean k-NN graphs was a non-standard Forman-type edge statistic on graphs built from temporal and cosine k-NN edges. Its released test permuted pooled edges instead of trajectories, and the published p-values came from unreleased code. The quantity reported as the norm of the first generated state is the state at the last prompt position, from which the first output token is predicted. The generic control prompt was matched to the identity prompt in characters, not in tokens. This version corrects the
    
[^379]: 群不变谱嵌入

    Group Invariant Spectral Embedding

    [https://arxiv.org/abs/2607.08987](https://arxiv.org/abs/2607.08987)

    该论文提出将紧致李群对称性直接融入谱嵌入的相似度核，证明了由此构造的图拉普拉斯算子逐点收敛到商空间上的显式二阶微分算子，且由于有效维度的降低而获得更快的收敛速度。

    

    谱嵌入方法被广泛应用于对具有内在低维结构的高维数据集进行降维和聚类。尽管许多实际数据集在旋转等对称性下具有不变性，但标准的谱嵌入方法并未考虑这一点，而是将通过对称性相关联的数据点视为彼此无关。我们解决这一问题的方法是将对称性直接融入到谱嵌入所使用的相似度核中。我们分析了由紧致李群 $G$ 给出对称性的黎曼数据流形 $M$ 的情况，并证明在适当条件下，由三种类型的不变核构造的图拉普拉斯算子逐点收敛到商空间 $M/G$ 上的显式二阶微分算子。我们的分析表明收敛速度得到了改善，因为有效维度会随群的维度而降低。我们验证了我们的

    arXiv:2607.08987v2 Announce Type: replace  Abstract: Spectral embedding methods are widely used for dimensionality reduction and clustering of high-dimensional datasets with intrinsic low-dimensional structures. Although many datasets of practical interest exhibit invariance under symmetries such as rotations, standard spectral embedding methods do not account for this, treating symmetry-related data points as unrelated. Our approach to this problem is to incorporate the symmetries directly into the affinity kernels used for spectral embedding. We analyze the case of a Riemannian data manifold $M$ with symmetries given by a compact Lie group~$G$ and prove that, under suitable conditions, graph Laplacians constructed from three types of invariant kernels converge pointwise to explicit second-order differential operators on the quotient space $M/G$. Our analysis implies improved convergence rates, as the effective dimension drops according to the dimension of the group. We validate our a
    
[^380]: 面向非线性回归的平均人口均等性直接优化

    Directly Optimizing Mean Demographic Parity for Nonlinear Regression

    [https://arxiv.org/abs/2607.05098](https://arxiv.org/abs/2607.05098)

    该论文提出DPVar（条件均值预测的方差）这一公平性度量，首次实现了非线性回归中平均人口均等准则的直接优化，克服了以往方法仅适用于线性预测器或低维敏感属性、且因过度约束而损害精度的局限。

    

    我们关注一类回归场景，其公平性目标是使不同敏感属性取值下的平均预测相等，这一准则被称为平均人口均等。直接优化该准则十分困难，因为它依赖于一个未知且在训练过程中不断变化的条件均值。常见的依赖性惩罚和对抗方法并不估计该条件均值，而是将预测推向完全独立。这种更强的约束即使在平均预测已经相等的情况下也可能降低准确性。现有的条件均值方法仅限于线性预测器或低维敏感属性。我们通过DPVar实现了平均人口均等的直接优化，DPVar是一种公平性度量，定义为条件均值预测的方差。由于条件均值必须随预测器的变化而进行估计，优化DPVar会引出一个泛函双层优化问题。我们开发了……

    arXiv:2607.05098v2 Announce Type: replace  Abstract: We focus on regression settings where the fairness goal is to equalize average predictions across values of a sensitive attribute, a criterion known as mean demographic parity. Directly optimizing this criterion is difficult because it depends on a conditional mean that is unknown and changes during training. Common dependence penalties and adversarial methods do not estimate this conditional mean; instead, they push predictions toward full independence. This stronger constraint can reduce accuracy even when average predictions are already equal. Existing conditional-mean methods are limited to linear predictors or low-dimensional sensitive attributes. We enable direct optimization of mean demographic parity using DPVar, a fairness measure defined as the variance of the conditional mean prediction. Because the conditional mean must be estimated as the predictor changes, optimizing DPVar leads to a functional bilevel problem. We devel
    
[^381]: 库普曼算子理论：基础、控制与应用

    Koopman operator theory: fundamentals, control, and applications

    [https://arxiv.org/abs/2607.01819](https://arxiv.org/abs/2607.01819)

    这是一篇关于库普曼算子理论的教程论文，系统介绍了其基本原理、数据驱动近似方法（如EDMD）及其在控制器设计（如库普曼MPC）中的应用，并提供了开源代码仿真示例。

    

    库普曼算子因其能够为高度复杂的动力系统提供全局线性表示而受到广泛关注。该算子通过实值或复值可观测量函数的视角，以线性方式描述非线性动力学。数据驱动技术，如扩展动态模态分解（EDMD）、核EDMD以及机器学习方法，可用于生成有限维近似并附带有限数据误差界。在这篇教程论文中，我们对库普曼算子理论及其在系统与控制中的应用进行了简明的介绍。论文特别关注数据驱动的代理模型、其在带输入系统上的扩展，以及基于库普曼算子理论的控制器设计。此外，我们还演示了其中的关键技术，即EDMD和库普曼模型预测控制（Koopman MPC）。为此，我们提供了在GitHub上附有源代码的仿真研究，以便感兴趣的读者使用。

    arXiv:2607.01819v2 Announce Type: replace-cross  Abstract: The Koopman operator has gained considerable attention due to its ability to provide a global linear representation of highly complex dynamical systems. The operator describes nonlinear dynamics in a linear way through the lens of real- or complex-valued observable functions. Data-driven techniques, like extended dynamic mode decomposition (EDMD), kernel EDMD, and machine-learning methods, can be used to generate finite-dimensional approximations accompanied by finite-data error bounds. In this tutorial paper, we provide a concise introduction into Koopman operator theory and its use in systems and control. A particular focus is put on data-driven surrogate models, their extension to systems with inputs, and controller design using Koopman operator theory. Moreover, we demonstrate the key techniques, i.e., EDMD and Koopman MPC. To this end, we provide simulation studies including source code on GitHub to enable the interested r
    
[^382]: 奖励可观测性与RSSM世界模型中离线检查点选择的局限性

    Reward Observability and the Limits of Offline Checkpoint Selection in RSSM World Models

    [https://arxiv.org/abs/2607.01736](https://arxiv.org/abs/2607.01736)

    该论文系统评估了RSSM世界模型在LunarLander任务上的闭环性能，表明基于世界模型想象训练的策略能以约65倍更少的真实交互数据匹敌无模型强化学习，并提出奖励可观测性分数（ROF）来揭示离线检查点选择方法的局限性。

    

    我们研究了在Gymnasium的LunarLander-v3环境中基于人类示范训练的循环状态空间模型（RSSM）世界模型的闭环特性。我们将训练好的世界模型用于零样本CEM模型预测控制（MPC）以及想象中的actor-critic（A2C）训练。在100个保留测试回合上的评分显示，所选的基于模型的A2C策略（在世界模型检查点280上训练）达到平均回报+189.5，与最佳无模型A2C检查点（+183.7；回合上限分别为600步和1000步）相当，而所需的真实训练转移数据仅约为其1/65。我们还将世界模型MPC与在成功示范上训练的行为克隆（BC）策略进行比较。BC策略仅在采用随机动作选择时才能达到与MPC相当的平均回报，并且在相同的20个回合上，BC出现了一次灾难性回合，而MPC则没有。随后，我们提出了奖励可观测性分数（Reward Observability Fraction, ROF），即奖励梯度的欧几里得……

    arXiv:2607.01736v3 Announce Type: replace-cross  Abstract: We study the closed-loop properties of a recurrent state-space model (RSSM) world model trained on human demonstrations in Gymnasium's LunarLander-v3. We use the trained world model for zero-shot CEM model-predictive control (MPC) and for actor-critic (A2C) training in imagination. Scored on 100 held-out episodes, the selected model-based A2C policy (trained on world-model checkpoint 280) reaches a mean return of +189.5, matching the best model-free A2C checkpoint (+183.7; 600- and 1000-step episode caps respectively) with ~65x fewer real training transitions. We also compare world-model MPC with a behaviour-cloning (BC) policy trained on the successful demonstrations. The BC policy matches MPC's mean return only under stochastic action selection, and on the same 20 episodes it has one catastrophic episode where MPC has none. We then introduce the Reward Observability Fraction (ROF), the Euclidean fraction of the reward gradien
    
[^383]: 一种由分数布朗运动驱动的神经网络的逐样本反向传播方法

    A samplewise backpropagation method for neural networks driven by fractional Brownian motion

    [https://arxiv.org/abs/2606.29438](https://arxiv.org/abs/2606.29438)

    本文提出一种由分数布朗运动驱动的随机神经网络，通过离散随机最大值原理构造伴随递推实现逐样本反向传播，证明了逐样本随机梯度下降的均方收敛性，并表明分数阶驱动在长记忆恢复和鲁棒性方面优于布朗运动和确定性基线。

    

    本文提出了一种以分数布朗运动驱动残差动力学的分数阶随机神经网络。通过为该网络引入离散随机最大值原理，我们构造了相应的伴随递推。对于确定性网络参数，我们证明了投影逐样本随机梯度下降的均方收敛性。数值实验包括闭式收敛性测试、带不确定性量化的含噪回归、长记忆时间序列生成以及结构化扰动下的图像分类。结果指出了分数阶驱动相比布朗运动和确定性基线能够改善长记忆恢复或鲁棒性的具体应用场景。

    arXiv:2606.29438v2 Announce Type: replace-cross  Abstract: In this paper, we develop a fractional stochastic neural network with residual dynamics driven by fractional Brownian motion. By introducing a discrete stochastic maximum principle for the network, we construct the corresponding adjoint recursion. For deterministic network parameters, we prove mean square convergence of projected samplewise stochastic gradient descent. Numerical experiments include a closed form convergence test, noisy regression with uncertainty quantification, long memory time series generation and image classification under structured perturbations. The results identify settings in which fractional drivers improve long memory recovery or robustness relative to Brownian and deterministic baselines.
    
[^384]: 红皇后哥德尔机：共同进化的智能体及其评估者

    The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators

    [https://arxiv.org/abs/2606.26294](https://arxiv.org/abs/2606.26294)

    本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。

    

    arXiv:2606.26294v1 公告类型：交叉 摘要：自我改进的智能体在编程基准测试中已达到最先进水平（SOTA），并最近被扩展到通用领域。然而，它们的搜索方法通常假设一个静态的评估标准：一个固定的验证器、基准测试或标注数据集，在智能体改进过程中保持有效。这忽略了进化的一个核心特征：物种随着环境的变化而适应。我们旨在将同样的原则引入递归自我改进，使评估成为改进循环的一部分，并将搜索开放给不断演化的评估者、对抗性目标和可能超越静态基准的动态效用函数。我们引入了红皇后哥德尔机（RQGM），这是一个用于非平稳效用下递归自我改进的演化框架。RQGM通过受控的效用演化实现了这一点：搜索被组织成具有固定期内评估标准的周期，而效用可以跨周期演化。

    arXiv:2606.26294v1 Announce Type: cross  Abstract: Self-improving agents are state-of-the-art (SOTA) on agentic coding benchmarks and have recently been extended to general domains. However, their search methods generally assume a stationary evaluation criterion: a fixed verifier, benchmark, or labeled dataset that remains valid as the agent improves. This ignores a central feature of evolution: species adapt as their environments change with them. We aim to bring the same principle to recursive self-improvement, making evaluation part of the improvement loop and opening search to evolving evaluators, adversarial objectives, and dynamic utilities that may surpass static benchmarks. We introduce the Red Queen Godel Machine (RQGM), an evolutionary framework for recursive self-improvement under non-stationary utilities. The RQGM makes this possible through controlled utility evolution: search is organized into epochs with a fixed within-epoch evaluation criterion, while the utility can be
    
[^385]: 外生上下文马尔可夫决策过程学习的极小极大PAC界

    Minimax PAC Bounds for Learning in Exogenous Contextual MDPs

    [https://arxiv.org/abs/2606.25170](https://arxiv.org/abs/2606.25170)

    该论文提出了一个在查询已知前后分配采样预算的新型PAC学习框架，并针对带外生上下文的折扣马尔可夫决策过程中的策略评估、最优值估计和最优策略提取任务，给出了极小极大最优的样本复杂度界。

    

    我们引入了一个PAC框架，其中学习者可以在决策之前和决策之时访问采样预言机。样本复杂度由一对 $(n,m)$ 来衡量，其中 $n$ 是在查询已知之前花费的学习预算，$m$ 是每个查询的额外采样预算。我们在带有外生独立同分布（i.i.d.）上下文的折扣马尔可夫决策过程中展示了该框架的相关性，这些上下文在行动之前被揭示。上下文可能影响奖励和转移，但不受智能体的控制。学习者可以对未知的上下文分布和转移核进行采样。我们研究了策略评估（PE）、最优值估计（BVE）和最优策略提取（BPE）三类任务。当奖励和转移已知时，一种方差缩减算法以样本复杂度 $(\widetilde O((1-\gamma)^{-3}\varepsilon^{-2}),0)$ 解决所有这三个任务，该复杂度在对数因子意义下是极小极大最优的。设 $\mathcal{X}$ 为受控状态……（摘要此处截断）

    arXiv:2606.25170v2 Announce Type: replace-cross  Abstract: We introduce a PAC framework in which the learner can access sampling oracles both before and at decision time. Sample complexity is measured by a pair $(n,m)$, where $n$ is the learning budget spent before a query is known and $m$ is the additional sampling budget per query. We demonstrate its relevance in discounted Markov decision processes with exogenous i.i.d.\ contexts revealed before acting. Contexts may affect both rewards and transitions but remain uncontrolled by the agent. The learner can sample the unknown context distribution and the transition kernel. We study policy evaluation (PE), best-value estimation (BVE), and best-policy extraction (BPE). When rewards and transitions are known, a variance-reduced algorithm solves all three tasks with sample complexity $\bigl(\widetilde O((1-\gamma)^{-3}\varepsilon^{-2}),0\bigr)$, which is minimax optimal up to logarithmic factors. Let $\mathcal{X}$ be the controlled state s
    
[^386]: 神经共轭聚合：异构传感器偏差下可识别的无监督多传感器回归

    Neural Conjugate Aggregation: Identifiable Unsupervised Multi-Sensor Regression under Heterogeneous Sensor Bias

    [https://arxiv.org/abs/2606.22200](https://arxiv.org/abs/2606.22200)

    提出神经共轭聚合模型（NCAM），一个结合神经网络与共轭高斯推断的层次贝叶斯框架，在无真值标签条件下实现多传感器数据融合，并通过传感器锚定与方差正则化解决不可识别性问题，提供解析可处理且不确定性分解的后验估计。

    

    我们研究不确定性环境下基于回归的数据融合问题，即存在多个带噪声和偏差的测量源，但训练期间缺乏真值标签。这种场景常见于传感器网络、仿真集成和科学监测系统中，因为这些系统的监督标注成本高昂或不可行。我们提出神经共轭聚合模型（NCAM），这是一个将神经网络与共轭高斯推断相结合的层次贝叶斯框架，用于无监督多源数据融合。NCAM基于上下文协变量学习各测量源特有的偏差和可靠性，得到潜在目标变量上解析可处理的后验分布，并将认知不确定性与偶然不确定性进行分解。通过传感器锚定和方差正则化解决了结构性不可识别问题，实现了稳定且可解释的后验聚合。为了用有限样本保证来补充贝叶斯不确定性……

    arXiv:2606.22200v2 Announce Type: replace-cross  Abstract: We study regression-based data fusion under uncertainty, where multiple noisy and biased measurement sources are available but ground-truth labels are absent during training. This setting arises in sensor networks, simulation ensembles, and scientific monitoring systems where supervision is costly or infeasible. We propose the Neural Conjugate Aggregation Model (NCAM), a hierarchical Bayesian framework that combines neural networks with conjugate Gaussian inference for unsupervised multi-source fusion. NCAM learns source-specific bias and reliability conditioned on contextual covariates, yielding an analytically tractable posterior over a latent target variable with decomposed epistemic and aleatoric uncertainty. Structural non-identifiability is resolved through sensor anchoring and variance regularization, enabling stable and interpretable posterior aggregation. To complement Bayesian uncertainty with finite-sample guarantees
    
[^387]: 从自身解中学习：可验证奖励强化学习的自条件式信用分配

    Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2606.18810](https://arxiv.org/abs/2606.18810)

    该论文提出一种自条件式信用分配方法，通过将模型以自身经验证的采样轨迹为条件构造自教师模型，利用逐token KL散度区分常规token与关键推理步骤，从而在不依赖外部教师或特权信息的情况下提升可验证奖励强化学习的训练效率。

    

    可验证奖励强化学习（RLVR）在训练大语言模型执行推理任务方面取得了显著进展，但以GRPO为代表的现有方法对所有token分配统一的信用，导致梯度被浪费在常规token上，而关键推理步骤却得不到足够的信用。现有的token级信用分配方法需要超出模型自身采样之外的资源：GRPO变体依赖过程奖励模型或标准答案；知识蒸馏通过逐token散度来分配信用，但需要外部教师模型（在线策略蒸馏）或特权信息（在线策略自蒸馏）。然而，这些依赖性限制了它们在纯RLVR场景中的适用性。我们观察到，将模型以其自身经过验证的轨迹作为条件时，会在原始分布与条件化分布之间产生可测量的逐token KL散度，并证明从由自身经验证采样构建的自教师模型进行蒸馏……（原文摘要在此处被截断）

    arXiv:2606.18810v2 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has driven substantial progress in training LLMs for reasoning tasks, but representative methods such as GRPO assign uniform credit across all tokens, wasting gradient on routine tokens while under-crediting pivotal reasoning steps. Existing token-level credit assignment methods require resources beyond the model's own rollouts. GRPO variants rely on process reward models or ground-truth answers. Knowledge distillation assigns credit through per-token divergence but requires external teachers (On-Policy Distillation) or privileged information (On-Policy Self Distillation). However, these dependencies limit applicability in the pure RLVR setting. We observe that conditioning the model on its own verified trajectories induces a measurable per-token KL divergence between the original and conditioned distributions, and prove that distilling from a self-teacher constructed by ver
    
[^388]: S4oP：面向资源受限设备的结构化状态空间模型算子级剪枝

    S4oP: Operator-level Pruning of Structured State Space Models for Resource-Constrained Devices

    [https://arxiv.org/abs/2606.18096](https://arxiv.org/abs/2606.18096)

    本文提出了首个针对结构化状态空间模型（S4/S4D）的算子级剪枝方法，通过交替进行结构化掩码与微调来逐步剪枝模型算子，在保持预测性能的同时显著降低推理成本，使其适用于资源受限设备。

    

    结构化状态空间模型（SSMs），包括S4和S4D架构，最近已成为基于注意力的模型的强大替代方案，用于捕获序列数据中的长程依赖关系。尽管这些模型具有出色的实证性能，但由于其计算和内存需求，在时间和资源受限的环境中部署它们仍然具有挑战性。在本文中，我们提出了一种新颖的增量式、算子级剪枝方法，用于基于S4和S4D的模型，在保持预测性能的同时显著降低推理成本。据我们所知，这是首个系统性地研究SSM结构化算子剪枝的工作。我们的方法通过将结构化掩码与微调交替进行来逐步剪枝模型算子，同时联合监测准确率和推理延迟。我们在一个统一的训练和评估框架中实现了该方法……

    arXiv:2606.18096v2 Announce Type: replace-cross  Abstract: Structured State Space Models (SSMs), including the S4 and S4D architectures, have recently emerged as powerful alternatives to attention-based models for capturing long-range dependencies in sequential data. Despite their strong empirical performance, deploying these models in time- and resource-constrained settings remains challenging due to their computational and memory demands. In this paper, we propose a novel incremental, operator-level pruning approach for S4- and S4D-based models that significantly reduces inference cost while preserving predictive performance. To the best of our knowledge, this is the first work to systematically investigate structured operator pruning for SSMs. Our method progressively prunes model operators by interleaving structured masking with fine-tuning, while jointly monitoring accuracy and inference latency. We implement this approach within a unified training and evaluation framework that en
    
[^389]: 学习攻击与防御：基于GRPO的语言模型自适应红队测试

    Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO

    [https://arxiv.org/abs/2606.09701](https://arxiv.org/abs/2606.09701)

    本文提出一种基于GRPO的攻击-防御协同训练框架，通过多LLM评判奖励通道与GDPO优势计算以及从攻击者单独训练到协同训练的课程策略，实现了高效可迁移的攻击生成并同步提升防御者的安全能力。

    

    语言模型的安全性必须不断适应持续演变的攻击。近期研究已经证明，强化学习可以通过应用PPO式的自博弈和DPO式的在线偏好优化，同步训练更强的攻击者模型和防御者模型。在本工作中，我们探索了GRPO在这一场景中的有效性。协同训练可能具有挑战性，因为它需要联合优化攻击者和防御者的多个属性。因此，我们使用多个基于LLM评判器的奖励通道来塑造模型输出，并采用GDPO计算优势函数，以防止任何单一通道占据主导地位。我们的方法采用一种课程式训练策略，从仅训练攻击者的单轮和多轮训练逐步过渡到协同训练，在协同训练阶段攻击者与防御者模型交替更新。我们表明，该方法能够产生高度有效且可迁移的攻击，并且协同训练的防御者模型在保持竞争力的安全性的同时……

    arXiv:2606.09701v2 Announce Type: replace-cross  Abstract: Language model safety must continually adapt to evolving attacks. Recent works have demonstrated that reinforcement learning can be used to train stronger attacker and defender models in tandem by applying PPO-style self-play and DPO-style online preference optimization. In this work, we explore the efficacy of GRPO in this setting. Co-training can be challenging because it requires jointly optimizing multiple properties of both the attacker and defender. We therefore shape model outputs using multiple LLM judge-based reward channels and compute advantages with GDPO, which prevents any single channel from dominating. Our method uses a curriculum that progresses from attacker-only single-turn and multi-turn training to co-training, where attacker and defender models are updated in alternation. We show that this method produces highly effective and transferable attacks, and that co-trained defenders reach competitive safety while
    
[^390]: GRASP：面向可扩展预训练数据归因的几何感知残差对齐

    GRASP: Geometry-aware Residual Alignment for Scalable Pretraining Data Attribution

    [https://arxiv.org/abs/2606.06892](https://arxiv.org/abs/2606.06892)

    论文提出GRASP方法，将数据归因重构为子集级反事实效用预测，通过二次几何惩罚建模子集间交互，并结合低维特征草图与严格有限的置信下界选择协议，在不依赖隐藏调参的情况下实现预训练规模的高效归因，其反事实子集保真度的秩相关性较现有基线提升一倍以上。

    

    可扩展的数据归因方法通常为单个训练样本分配孤立的效用分数。这种普遍存在的可加性假设从根本上无法捕捉关键的子集动态，包括数据冗余和互补覆盖。在本工作中，我们将归因重新构建为子集级别的反事实效用预测问题，并提出了GRASP——一种交互感知的代理模型。基于理论平滑性下界，GRASP通过二次几何惩罚显式地建模子集交互。为了在不依赖隐藏oracle调参的情况下实现预训练规模的效率，我们将低维特征草图与严格有限的置信下界选择协议相结合。大量的子集重训练评估表明，GRASP显著优于现有的可扩展基线方法：它在反事实子集保真度方面将任务级别的秩相关性提高了一倍以上，同时减少了前期开销（原文在此处截断）。

    arXiv:2606.06892v2 Announce Type: replace  Abstract: Scalable data attribution methods typically assign isolated utility scores to individual training examples. This prevalent additive assumption fundamentally fails to capture critical subset dynamics, including data redundancy and complementary coverage. In this work, we reframe attribution as subset-level counterfactual utility prediction and introduce GRASP, an interaction-aware surrogate. Grounded in a theoretical smoothness lower bound, GRASP explicitly models subset interactions through a quadratic geometric penalty. To achieve pretraining-scale efficiency without relying on hidden oracle tuning, we couple low-dimensional feature sketches with a strictly finite lower-confidence bound selection protocol. Extensive subset-retraining evaluations demonstrate that GRASP decisively outperforms existing scalable baselines. It more than doubles the task-level rank correlation for counterfactual subset fidelity while reducing upfront arti
    
[^391]: 在生成空间中学习隐式偏置以加速蛋白质动力学模拟

    Learning Implicit Bias in Generative Spaces for Accelerating Protein Dynamics Emulation

    [https://arxiv.org/abs/2606.01833](https://arxiv.org/abs/2606.01833)

    该方法在预训练蛋白质动力学生成模拟器的生成空间中引入隐式的历史依赖偏置，引导采样远离已生成的结构，从而将构象多样性提升35%并实现对罕见状态的长时程零样本探索。

    

    蛋白质动力学的生成式模拟器能以远低于分子动力学的成本产生合理的轨迹，但它们继承了训练分布，在长时程外推时往往倾向于重访已知状态而非到达罕见状态。受经典增强采样技术的启发，我们在预训练模拟器的生成空间中引入了一种隐式的、依赖历史的偏置。具体而言，一个具备历史感知能力的分数估计器为冻结的模拟器添加了距离加权偏置，引导逆时间采样远离先前生成的结构，并通过环境支持项进行正则化。为了在长时程中保持结构有效性，基于分数的细化步骤利用冻结的模拟器将发生漂移的样本重新投影到数据流形上。我们的实验表明，该方法 在 DynamicPDB-80 上将多样性提高了 35%； 在 12 个零样本快速折叠蛋白质上，所学到的偏置……

    arXiv:2606.01833v2 Announce Type: replace-cross  Abstract: Generative emulators of protein dynamics produce plausible trajectories at a fraction of the cost of molecular dynamics, but they inherit their training distribution and tend to revisit known states rather than reach rare ones under long-horizon extrapolation. Inspired by classical enhanced sampling, we introduce an implicit, history-dependent bias in the generative space of a pretrained emulator. Specifically, a history-aware score estimator augments the frozen emulator with a distance-weighted bias that steers reverse-time sampling away from previously generated structures, regularized by an environment-support term. To preserve structural validity at long horizons, a score-based refinement step re-projects drifted samples onto the data manifold using the frozen emulator. Our experiments demonstrate that the method (i) raises diversity by $35\%$ on DynamicPDB-80; (ii) on $12$ zero-shot Fast-Folding proteins, the learned bias 
    
[^392]: 面向部分反馈情境线性优化的决策聚焦在线策略学习

    Decision-Focused On-Policy Learning for Contextual Linear Optimization with Partial Feedback

    [https://arxiv.org/abs/2606.01081](https://arxiv.org/abs/2606.01081)

    该论文提出了一种在部分反馈下用于序贯情境线性优化的决策聚焦在线策略学习方法，通过结合得分函数估计器与决策聚焦即插即用组分的混合梯度估计器来训练随机化的预测后优化策略。

    

    决策聚焦学习（DFL）通过优化下游决策质量而非单纯的预测精度来训练预测模型。在情境线性优化领域，现有的大多数DFL方法都假设数据是离线的，并且可以完整观测目标成本向量。我们针对部分反馈下的序贯情境线性优化提出了一种在线策略学习方法，该方法将标准的赌臂反馈设置进行了推广。我们的方法学习一个随机化的“预测后优化”策略：该策略从条件分布中采样成本向量预测，并求解由此得到的下游线性优化问题。为了更新这一分布模型，我们引入了一种由两个组分构成的混合梯度估计器。第一个组分是得分函数估计器，它提供无偏但方差可能很高的策略梯度估计；第二个组分是决策聚焦的即插即用部分，它利用一个辅助的干扰估计……（原文摘要至此截断）

    arXiv:2606.01081v2 Announce Type: replace  Abstract: Decision-focused learning (DFL) trains predictive models by optimizing downstream decision quality rather than standalone prediction accuracy. For contextual linear optimization, most existing DFL methods assume offline data and full observations of the objective cost vector. We develop an on-policy learning method for sequential contextual linear optimization under partial feedback, generalizing the standard bandit feedback setting. Our method learns a stochastic predict-then-optimize policy that samples a cost-vector prediction from a conditional distribution and solves the resulting downstream linear optimization problem. To update this distributional model, we introduce a two-component hybrid gradient estimator. The first component is a score function estimator, which provides an unbiased but potentially high-variance policy gradient estimate. The second is a decision-focused plug-in component that uses an auxiliary nuisance esti
    
[^393]: 记忆设计：概率序列层

    Memory by Design: Probabilistic Sequence Layers

    [https://arxiv.org/abs/2605.31163](https://arxiv.org/abs/2605.31163)

    本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。

    

    arXiv:2605.31163v3 公告类型：交叉替换 摘要：我们引入了“设计模型框架”：一种从关于记忆的明确假设中推导高效循环序列映射的方法。设计模型通过精确贝叶斯滤波将证据写入记忆；查询相关的读取输出产生预测分布，其均值作为层输出。在我们的线性-高斯实例化中，“贝叶斯层”同时传播均值和协方差：协方差跟踪存储关联中的不确定性，将写入导向不确定方向，随着证据积累而衰减增益，并保留自信记忆。同一框架统一了多种次二次递归：线性注意力、GLA和Mamba-2/SSD在潜在输入设计模型下是精确滤波器，而DeltaNet及相关Delta规则模型是贝叶斯层设计模型的协方差重置简化。恢复协方差传播为检索提供闭式预测。

    arXiv:2605.31163v3 Announce Type: replace-cross  Abstract: We introduce the \emph{design-model framework}: a way to derive efficient recurrent sequence maps from explicit assumptions about memory. A design model writes evidence into memory by exact Bayesian filtering; a query- dependent readout produces a predictive distribution whose mean is the layer output. In our linear-Gaussian instantiation, the \emph{Bayesian Layer} propagates both a mean and a covariance: the covariance tracks uncertainty over stored associations, steering writes toward uncertain directions, attenuating gains as evidence accumulates, and preserving confident memories. The same framework unifies several sub-quadratic recurrences: linear attention, GLA, and Mamba-2/SSD are exact filters under a latent-input design model, whereas DeltaNet and related Delta-rule models are covariance-reset reductions of the Bayesian Layer's design model. Restoring covariance propagation yields closed-form predictions for retrieval 
    
[^394]: 高德地图中基于隐式推理的生成式时空意图序列推荐

    Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap

    [https://arxiv.org/abs/2605.28888](https://arxiv.org/abs/2605.28888)

    高德地图提出GPlan框架，通过渐进式隐式思维链蒸馏将大语言模型的推理能力内化到轻量级模型中，在严格延迟约束下实现逻辑连贯且物理可执行的生成式时空意图序列推荐。

    

    现实世界中的用户行为很少由孤立的动作构成；相反，它往往形成受时空依赖关系支配的意图流。为了提供一体化的服务推荐，我们聚焦于生成式时空意图序列推荐（GSISR）这一任务，其目标是在复杂的时空情境中生成逻辑连贯且物理可执行的意图序列。尽管大语言模型为GSISR提供了强大的推理潜力，但其在工业界的直接部署受到高推理延迟以及规划结果与情境不匹配或物理上不可执行的限制。为应对这些挑战，我们提出了一个生成式框架GPlan，通过两个组件将大语言模型的推理能力内化到轻量级模型中。首先，为了在严格的延迟约束下实现推理，我们引入了渐进式隐式思维链蒸馏（Progressive Implicit CoT Distillation），将显式的推理过程压缩到预留的潜在token中，使小……

    arXiv:2605.28888v2 Announce Type: replace-cross  Abstract: Real-world user behavior rarely consists of isolated actions; instead, it often forms intent flows governed by spatiotemporal dependencies. To provide integrated service recommendations, we focus on the task of Generative Spatiotemporal Intent Sequence Recommendation (GSISR), which aims to generate intent sequences that are logically coherent and physically executable within complex spatiotemporal contexts. While LLMs offer strong reasoning potential for GSISR, direct industrial deployment is limited by high inference latency and context-mismatched or physically infeasible plans. To address these challenges, we propose a generative framework, GPlan, that internalizes LLM reasoning into lightweight models through two components. First, to enable reasoning under strict latency constraints, we introduce Progressive Implicit CoT Distillation, which compresses explicit reasoning processes into reserved latent tokens, allowing small 
    
[^395]: MemTrace：追踪与归因大型语言模型记忆系统中的错误

    MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems

    [https://arxiv.org/abs/2605.28732](https://arxiv.org/abs/2605.28732)

    该论文提出了MemTrace框架，将LLM记忆流水线转化为可执行的记忆演化图，并结合MemTraceBench基准和自动归因方法，实现了对记忆系统错误的细粒度追踪与根因定位。

    

    记忆对于使大型语言模型能够支持长程推理至关重要，然而现有的记忆系统仍然不可靠且难以调试。追踪记忆的动态演变对于理解信息如何随时间被合成、传播或损坏至关重要。在本工作中，我们研究了LLM记忆系统中错误追踪与归因这一新问题。我们提出了一个新颖的框架，将记忆流水线转化为可执行的记忆演化图，实现对操作信息流的细粒度追踪。随后，我们构建了MemTraceBench，这是一个从代表性记忆系统（如Long-Context、RAG、Mem0和EverMemOS）中收集的基准数据集，用于系统地研究记忆失败模式。我们进一步引入了一种自动归因方法，通过迭代追踪操作子图来精确定位任何失败案例的根本原因。我们的分析揭示出，记忆失败是系统性的，源于……

    arXiv:2605.28732v4 Announce Type: replace-cross  Abstract: Memory is essential for enabling large language models to support long-horizon reasoning, yet existing memory systems remain unreliable and difficult to debug. Tracing memory's dynamic evolution is crucial to understand how information is synthesized, propagated, or corrupted over time. In this work, we study the new problem of error tracing and attribution in LLM memory systems. We propose a novel framework that transforms memory pipelines into executable memory evolution graphs, enabling fine-grained tracing of operational information flow. We then construct MemTraceBench, a benchmark collected from representative memory systems such as Long-Context, RAG, Mem0, and EverMemOS, to systematically study memory failure modes. We further introduce an automatic attribution method that iteratively traces operation subgraphs to pinpoint the root cause of any failed case. Our analysis reveals that memory failures are systematic, stemmi
    
[^396]: 信赖域Q伴随匹配

    Trust Region Q Adjoint Matching

    [https://arxiv.org/abs/2605.27079](https://arxiv.org/abs/2605.27079)

    本文提出信赖域Q伴随匹配（TRQAM），通过在流策略采样过程中引入信赖域参数λ来精确加权路径空间KL散度，并借助投影对偶下降自适应控制策略更新幅度，从而解决了QAM中评论家误差被指数级放大导致性能崩溃的问题，实现了对预训练流策略的稳定离策略微调。

    

    对预训练流策略进行离策略强化学习仍然具有挑战性，其困难源于多步采样过程所带来的优化不稳定性。最近，带伴随匹配的Q学习（QAM）通过将策略改进重新表述为以学习到的评论家为指导的随机最优控制（SOC）问题，解决了这一难题。然而，QAM继承了评论家引导改进方式的一个根本性脆弱之处：微小的评论家误差会被指数级放大，并常常导致性能崩溃。本文提出了信赖域Q伴随匹配（TRQAM），这是一种稳定的离策略微调算法，通过投影对偶下降自适应地控制微调策略与预训练策略之间的路径空间KL散度。具体而言，我们在流策略的采样过程中引入了一个信赖域参数λ，并证明λ恰好对SOC目标中的路径空间KL散度进行加权。因此，我们的……（摘要截断）

    arXiv:2605.27079v2 Announce Type: replace-cross  Abstract: Off-policy reinforcement learning of pretrained flow policies remains challenging due to the instability of optimization arising from the multi-step sampling process. Recently, Q-learning with Adjoint Matching (QAM) addressed this by recasting policy improvement as a stochastic optimal control (SOC) problem guided by a learned critic. However, QAM inherits a fundamental fragility of critic-guided improvement, since small critic errors can be exponentially amplified and often lead to performance collapse. This paper introduces Trust Region Q Adjoint Matching (TRQAM), a stable off-policy fine-tuning algorithm that adaptively controls the path-space KL between the fine-tuned and pretrained policies through projected dual descent. Specifically, we adapt a trust-region parameter $\lambda$ inside the sampling process of the flow policy and prove that $\lambda$ exactly weights the path-space KL in the SOC objective. As a result, our m
    
[^397]: ROAR：面向零样本时间序列预测的检索机会感知精炼框架

    ROAR: Retrieval Opportunity-Aware Refinement for Zero-Shot Time Series Forecasting

    [https://arxiv.org/abs/2605.24911](https://arxiv.org/abs/2605.24911)

    该论文提出ROAR框架，通过依据基础预测难度和检索候选相对改进来加权训练目标，并联合学习候选聚合、门控校正与预测模块校准，从而有效捕捉并利用检索增强在零样本时间序列预测中的改进机会。

    

    检索增强为时间序列预测器提供了历史延续片段，然而即使是那些优于基础预测的检索候选，也可能无法改善最终预测结果。我们提出ROAR，一个用于零样本时间序列预测的检索机会感知精炼框架。为了更好地利用这些改进机会，其训练目标基于基础预测的难度以及检索候选所能提供的相对改进，在不同查询之间分配额外的训练权重。利用这一目标，ROAR首先学习聚合对齐的历史候选，并使用一个学习到的门控机制来控制其相对于固定基础预测器的校正强度。随后，框架联合校准预测模块与门控机制以协调二者的贡献，同时将组合后的预测锚定在第一阶段的精炼输出上。我们推导出了精炼增益的精确分解以及机会加权训练损失的...（原文摘要在此处截断）

    arXiv:2605.24911v2 Announce Type: replace-cross  Abstract: Retrieval augmentation provides time series forecasters with historical continuations, yet even candidates that outperform the base forecast may fail to improve the final prediction. We propose ROAR, a Retrieval Opportunity-Aware Refinement framework for zero-shot time series forecasting. To better exploit these improvement opportunities, its training objective allocates additional emphasis across queries based on base-forecast difficulty and the relative improvement offered by retrieved candidates. Using this objective, ROAR first learns to aggregate aligned historical candidates and uses a learned gate to control their correction strength against a fixed base forecaster. It then jointly calibrates the forecasting module and gate to coordinate their contributions, while anchoring the combined prediction to the first-stage refined output. We derive exact decompositions of refinement gains and the opportunity-weighted training l
    
[^398]: MARGIN：面向多智能体基础模型协作的运行时置信度校准

    MARGIN: Runtime Confidence Calibration for Multi-Agent Foundation Model Coordination

    [https://arxiv.org/abs/2605.22949](https://arxiv.org/abs/2605.22949)

    MARGIN是一种运行时置信度校准方法，通过从观察到的答案结果中在线学习模型特定的置信度修正，无需重训模型或校准集即可提升多智能体基础模型协作中集体决策的可靠性。

    

    当一个协调器比较来自异构基础模型的答案时，自报告的置信度在不同响应者和不断变化的工作负载下可能有不同的含义。本文提出了MARGIN（通过增量归一化进行多智能体运行时评分），这是一种运行时校准方法，能够从观察到的答案结果中学习模型特定的置信度修正，无需重新训练模型或需要保留的校准集。MARGIN在置信度区间内跟踪近期准确率和声明的置信度，使用二者的比率来修正报告的置信度，并将稀疏区间的修正融合到模型级别的估计中。修正后的分数在集体决策中对候选答案进行加权。评估涵盖代码生成、问答和数学任务，使用18个模型的池以及9个模型的子集进行分布偏移实验。在BigCodeBench上，模型平均置信度与准确率呈负相关；在正确/不完（摘要在此处截断）

    arXiv:2605.22949v4 Announce Type: replace  Abstract: When a coordinator compares answers from heterogeneous foundation models, self-reported confidence may have different meanings across responders and changing workloads. This paper presents MARGIN (Multi-Agent Runtime Grading via Incremental Normalisation), a runtime calibration method that learns model-specific confidence corrections from observed answer outcomes without retraining the models or requiring a held-out calibration set. MARGIN tracks recent accuracy and stated confidence within confidence bands, uses their ratio to correct reported confidence, and blends sparse-band corrections toward a model-level estimate. The corrected scores weight candidate answers in a collective decision. Evaluation covers code generation, question answering, and mathematics, using an 18-model pool and a nine-model subset for distribution-shift experiments. On BigCodeBench, model-mean confidence is negatively related to accuracy; among correct/inc
    
[^399]: 蒸馏博弈：自适应评估与高效防御

    The Distillation Game: Adaptive Evaluations & Efficient Defenses

    [https://arxiv.org/abs/2605.22737](https://arxiv.org/abs/2605.22737)

    该论文将模型蒸馏攻击与防御建模为教师与学生之间的极小极大博弈，提出了自适应评估规则和仅需前向传播的专家乘积防御方法，并揭示自适应学生模型在强评估下恢复的能力远超被动评估所显示的水平。

    

    蒸馏攻击给模型提供者带来了部署上的权衡：使模型更有用的输出同时也会使模型更容易被模仿。我们通过在效用受限的教师模型与自适应学生模型之间构建一个极小极大博弈来研究这一权衡。我们的框架得到了可求解的单边响应规则：一种自适应评估规则，其中学生模型对高价值样本进行重新加权；以及一种教师端防御模板，用于抑制对蒸馏最有用的输出。基于一种廉价的样本价值代理，我们推导出专家乘积，这是一种仅需前向传播的简单防御方法，它在生成过程中将教师模型与代理学生模型相结合。实验表明，自适应评估揭示了一个巨大的被动-自适应差距：在GSM8K和MATH数据集上，自适应学生模型恢复的能力远超被动评估所显示的水平。在这种更强的评估下，（原文摘要在此处截断）

    arXiv:2605.22737v4 Announce Type: replace-cross  Abstract: Distillation attacks create a deployment trade-off for model providers: the same outputs that make a model more useful can also make it easier to imitate. We study this trade-off through a minimax game between a utility-constrained teacher and an adaptive student. Our framework yields tractable one-sided response rules: an adaptive evaluation rule in which the student reweights high-value examples, and a teacher-side defense template that suppresses outputs most useful for distillation. From a cheap proxy for example value, we derive Product-of-Experts (PoE), a simple forward-pass-only defense that combines the teacher with a proxy student during generation. Empirically, adaptive evaluation reveals a large passive--adaptive gap: on state-of-the-art defenses, adaptive students recover substantially more capability than passive evaluation suggests on GSM8K and MATH. Under this stronger evaluation, the apparent robustness gap betw
    
[^400]: 超越准确率：EEG基础模型的鲁棒性、可解释性与表达能力

    Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG Foundation Models

    [https://arxiv.org/abs/2605.17562](https://arxiv.org/abs/2605.17562)

    本研究超越传统的干净数据准确率评估，从鲁棒性、可解释性和表达能力三个维度系统评估了六个EEG基础模型，发现没有任何单一模型能在所有失效模式下占优，且模型的归因总体集中于符合已知神经生理学的任务相关脑区。

    

    arXiv:2605.17562v2 公告类型：replace-cross。摘要：EEG基础模型（EEG-FMs）此前主要在干净、分布内的准确率上进行评估，显示出相较于有监督基线仅有的适度提升以及较弱的冻结表征能力。本研究通过在十个数据集上评估六个EEG基础模型和一个有监督基线，从三个分析层面检验这些结论在干净准确率之外是否依然成立：（i）鲁棒性：我们施加了测试时扰动，包括加性噪声、随机及基于区域的通道丢弃，以及特定区域的噪声注入。我们的分析表明，没有任何单一模型能在所有失效模式下占据优势。对噪声最鲁棒的模型在通道丢弃下却是最脆弱的模型之一，而且当通道被直接移除而非零填充时，大部分通道丢弃带来的脆弱性会消失。（ii）可解释性：通过在EEG基础模型中使用归因方法，我们发现这些模型总体上将相关性集中于与已知神经生理学一致的任务相关脑区。

    arXiv:2605.17562v2 Announce Type: replace-cross  Abstract: EEG foundation models (EEG-FMs) have been evaluated predominantly on clean, in-distribution accuracy, demonstrating modest gains over supervised baselines and weak frozen representations. This study examines whether these conclusions hold beyond clean accuracy by evaluating six EEG-FMs and a supervised baseline across ten datasets along three layers of analysis: (i) Robustness: we apply test-time perturbations including additive noise, random and region-based channel dropout and region-specific noise injection. Our analyses show that no single model dominates all failure modes. The most noise-robust model is among the most fragile under channel dropout and much of the dropout fragility disappears when channels are removed rather than zero-padded. (ii) Interpretability: using attribution methods in EEG-FMs, we show that models broadly concentrate relevance on task-appropriate brain regions consistent with known neurophysiology. 
    
[^401]: 道路地图作为免费几何先验：基于GeoFuse的天气不变无人机地理定位

    Road Maps as Free Geometric Priors: Weather-Invariant Drone Geo-Localization with GeoFuse

    [https://arxiv.org/abs/2605.14925](https://arxiv.org/abs/2605.14925)

    提出GeoFuse跨模态融合框架，利用免费可得且天生天气不变的道路地图几何先验与卫星图像融合，在几乎零额外成本下实现恶劣天气条件下鲁棒的无人机地理定位。

    

    无人机视角地理定位旨在将查询的无人机图像（通常在雨、雪、雾等恶劣天气条件下拍摄）与带有地理标记的卫星图像库进行匹配。天气引起的无人机视角退化，如噪声、能见度降低和部分遮挡，严重加剧了固有的跨视角域差距。虽然先前的方法主要依赖于特定天气的架构或数据增强，但它们在很大程度上忽视了道路地图数据——这是一种随时可用的模态，能够以几乎可以忽略的额外成本提供强大的、天生天气不变的几何布局线索（如道路网络和建筑物轮廓）。我们提出了GeoFuse，一个跨模态融合框架，它将精确对齐的道路地图瓦片与卫星图像相融合，以产生更具判别力和天气鲁棒性的表示。我们首先用地理……（摘要在此处被截断）增强现有的University-1652和DenseUAV基准数据集……

    arXiv:2605.14925v3 Announce Type: replace-cross  Abstract: Drone-view geo-localization aims to match a query drone image, often captured under adverse weather conditions (e.g., rain, snow, fog), against a gallery of geo-tagged satellite images. Weather-induced degradations in the drone view, such as noise, reduced visibility, and partial occlusions, severely exacerbate the intrinsic cross-view domain gap. While prior methods predominantly rely on weather-specific architectures or data augmentations, they have largely overlooked road map data, a readily available modality that provides strong, inherently weather-invariant geometric layout cues (e.g., road networks and building footprints) at negligible additional cost. We introduce GeoFuse, a cross-modal fusion framework that integrates precisely aligned road map tiles with satellite imagery to yield more discriminative and weather-resilient representations. We first augment the existing University-1652 and DenseUAV benchmarks with geo-
    
[^402]: 基于梯度预测的快速对抗攻击

    Fast Adversarial Attacks with Gradient Prediction

    [https://arxiv.org/abs/2605.14868](https://arxiv.org/abs/2605.14868)

    该论文提出通过轻量级线性回归从前向传播隐藏状态预测输入梯度、从而消除反向传播的快速对抗攻击方法，在保持FGSM大部分攻击性能的同时实现了532%的吞吐量提升。

    

    大规模生成对抗样本是鲁棒性评估、对抗训练和红队测试的核心基础操作，然而即使是FGSM这样的“快速”攻击，其吞吐量仍然受限于反向传播的计算成本。我们提出了一族攻击方法，通过轻量级线性回归从前向传播的隐藏状态中预测输入梯度，从而完全消除了反向传播步骤。在理论方面，我们推导出了精确的仿射条件梯度均值，证明了在（理想化的）神经正切核（NTK）机制下的最优性。在实证方面，我们的方法在实际的有限宽度模型上也能有效工作；我们在仅使用一小部分时间的情况下，恢复了FGSM大部分的攻击性能，对应于532%的吞吐量提升。这些结果表明，梯度预测是在实际时间约束下实现显著更快对抗样本生成的一种简单而通用的途径。

    arXiv:2605.14868v2 Announce Type: replace  Abstract: Generating adversarial examples at scale is a core primitive for robustness evaluation, adversarial training, and red-teaming, yet even "fast" attacks such as FGSM remain throughput-limited by the cost of a backward pass. We introduce a family of attacks that eliminates the backward pass by predicting the input gradient from forward-pass hidden states via a lightweight linear regression. Theoretically, we derive exact affine conditional gradient means, showing optimality in the (idealized) NTK regime. Empirically, our methods work when applied to practical finite-width models; we recover much of FGSM's attack performance while using only a small fraction of the time, corresponding to a $532\%$ increase in throughput. These results suggest gradient prediction as a simple and general route to significantly faster adversarial generation under realistic wall-clock constraints.
    
[^403]: FeatCal：面向合并后模型的后处理特征校准方法

    FeatCal: Feature Calibration for Post-Merging Models

    [https://arxiv.org/abs/2605.13030](https://arxiv.org/abs/2605.13030)

    提出FeatCal方法，基于特征漂移理论（分解为上游传播与局部失配）分析模型合并后的性能差距，并利用小校准集以前向顺序逐层闭式校准合并模型权重，无需梯度下降或额外模块即可减少特征漂移、保留模型合并优势。

    

    模型合并将多个任务专家模型组合成一个模型，从而避免联合训练、重新训练或部署多个专家模型，但合并后的模型往往仍不如任务专家模型。我们通过“特征漂移”来研究这一性能差距，即合并模型与专家模型在相同输入上产生的特征之间的差异。我们的理论将这种漂移分解为上游传播与局部失配两部分，追踪漂移如何按前向顺序在后续层中传播与叠加，并将最终的特征漂移与输出漂移联系起来。这一视角催生了FeatCal方法，它使用一个小的校准集，按前向顺序逐层校准合并模型的权重，在减少特征漂移的同时保持接近合并后的权重，从而保留模型合并的优势。FeatCal采用高效的闭式解来更新模型权重，无需梯度下降、迭代优化或任何额外模块。在主要的CLIP和（摘要在此处被截断）

    arXiv:2605.13030v2 Announce Type: replace-cross  Abstract: Model merging combines task experts into one model and avoids joint training, retraining, or deploying many expert models, but the merged model often still underperforms task experts. We study this performance gap through feature drift, the difference between features produced by the merged model and by the expert on the same input. Our theory decomposes this drift into upstream propagation and local mismatch, tracks how it propagates and combines through later layers in forward order, and links final feature drift to output drift. This view motivates FeatCal, which uses a small calibration set to calibrate the merged model weights layer by layer in forward order, reducing feature drift while staying close to merged weights and preserving the benefits of model merging. FeatCal uses an efficient closed-form solution to update model weights, with no gradient descent, iterative optimization, or extra modules. On the main CLIP and 
    
[^404]: 基于门控激活重定向的推理时机器遗忘

    Inference-Time Machine Unlearning via Gated Activation Redirection

    [https://arxiv.org/abs/2605.12765](https://arxiv.org/abs/2605.12765)

    GUARD-IT 是一种无需训练、无需梯度、不改变模型权重的推理时机器遗忘方法，它将待遗忘内容存储为小型激活方向库，并在推理时通过依赖输入的门控激活重定向实现遗忘。

    

    arXiv:2605.12765v4 公告类型：替换 摘要：大语言模型（LLM）会记忆大量训练数据，这引发了关于隐私、版权侵权和安全性的担忧。机器遗忘旨在移除模型对目标遗忘集的影响，同时保持模型性能，理想情况下应接近在没有该数据情况下从头重新训练的模型。然而，一旦LLM投入使用，每一个使其遗忘特定内容的新请求都需要更新其权重。但是，通过参数更新实现遗忘代价高昂、难以审计，并且可能被量化操作所抵消。我们证明，遗忘可以完全在推理时实施，无需训练、无需梯度、无需更改权重。我们提出了基于门控激活重定向的推理时遗忘方法（GUARD-IT），这是一种无需训练、无需梯度的方法，通过推理时依赖输入的激活引导来实现遗忘。GUARD-IT 将需要遗忘的内容存储为一个小的激活方向库，并在推理时……

    arXiv:2605.12765v4 Announce Type: replace  Abstract: Large Language Models (LLMs) memorize vast amounts of training data, raising concerns regarding privacy, copyright infringement, and safety. Machine unlearning seeks to remove the influence of a targeted forget set while preserving model performance, ideally approximating a model retrained from scratch without it. Once an LLM is in use, every new request to make it forget specific content demands updating its weights. However, unlearning through parameter updates is expensive, hard to audit, and can be undone by quantization. We show that unlearning can be enforced entirely at inference time, without training, gradients, or weight changes. We introduce Inference-Time Unlearning via Gated Activation Redirection (GUARD-IT), a training- and gradient-free method that unlearns via input-dependent activation steering at inference time. GUARD-IT stores the content to be forgotten as a small library of activation directions, and during infer
    
[^405]: 期望批量最优传输计划及其对流匹配的意义

    Expected Batch Optimal Transport Plans and Consequences for Flow Matching

    [https://arxiv.org/abs/2605.12174](https://arxiv.org/abs/2605.12174)

    本文形式化了重复小批量OT所诱导的“期望批量OT计划”，证明其在大批量下的一致性并给出收敛速率，且表明该耦合诱导的速度场足够正则，能为流匹配定义唯一的流。

    

    在随机小批量上求解最优传输（OT）是大规模学习中替代精确OT的常用方法。在流匹配（FM）中，这种替代方法被用于获得类似OT的耦合，从而拉直概率路径并降低数值积分成本。然而，由重复小批量OT所诱导的总体层面上的耦合至今仍未被完全理解。我们将这种耦合形式化为期望批量OT计划 $\overline{\pi}_{k}$，即通过对大小为 $k$ 的独立小批次上的经验OT计划取平均而得到。随后，我们建立了该计划的大批量一致性，并在与生成建模相关的半离散情形下，推导了传输成本偏差以及 $\overline{\pi}_{k}$ 收敛到OT计划的速率。对于流匹配，这给出了一个总体耦合，其诱导的速度场具有足够的正则性，能够定义从源分布到离散目标分布的唯一流。我们最后量化了OT批量大小如何与（摘要在此处截断）

    arXiv:2605.12174v2 Announce Type: replace  Abstract: Solving optimal transport (OT) on random minibatches is a common surrogate for exact OT in large-scale learning. In flow matching (FM), this surrogate is used to obtain OT-like couplings that can straighten probability paths and reduce numerical integration cost. Yet, the population-level coupling induced by repeated minibatch OT remains only partially understood. We formalize this coupling as the expected batch OT plan $\overline{\pi}_{k}$, obtained by averaging empirical OT plans over independent minibatches of size $k$. We then establish its large-batch consistency and, in the semidiscrete case relevant to generative modeling, derive rates for both the transport-cost bias and the convergence of $\overline{\pi}_{k}$ to the OT plan. For FM, this yields a population coupling whose induced velocity field is regular enough to define a unique flow from the source to the discrete target. We finally quantify how OT batch size interacts wi
    
[^406]: 保持评分：面向得分增强神经比率估计的自适应、免调参损失加权方法

    Keeping Score: Adaptive, Tuning-Free Loss Weighting for Score-Augmented Neural Ratio Estimation

    [https://arxiv.org/abs/2605.12118](https://arxiv.org/abs/2605.12118)

    提出一种基于损失梯度的自适应、免调参算法来动态设置得分匹配损失的权重，以极小的额外开销提升得分增强神经比率估计代理模型的质量并大幅降低调参成本。

    

    随机过程模型的神经似然代理模型（例如神经比率估计）通常通过对模拟数据进行概率分类来训练，这迫使代理模型质量与训练成本之间做出权衡。对于可以获取精确得分 ∇_θ log p(x | θ) 的结构化模型，可以通过在交叉熵损失中增加得分匹配项，将该信息纳入训练过程。然而，两种损失的最优权重无法先验得知，手动选择权重需要昂贵的调参，从而削弱了计算成本上的节省。我们提出了一种自适应、免调参的算法，在训练过程中基于损失梯度来设置得分损失的权重，仅为标准分类器训练增加极小的开销。我们在涉及网络动力学和空间过程的案例研究中评估了该方法，证明其能以大幅降低的计算成本提升代理模型的质量。

    arXiv:2605.12118v3 Announce Type: replace-cross  Abstract: Neural likelihood surrogates (e.g., Neural Ratio Estimation) for stochastic process models are commonly trained via probabilistic classification on simulated data, which forces a tradeoff between surrogate quality and training costs. For structured models where the exact score $\nabla_\theta \log p(x \mid \theta)$ is available, this information can be incorporated into training by augmenting the cross-entropy loss with a score-matching term. However, the optimal weighting of the two losses is not known a priori, and selecting it by hand requires expensive tuning that undercuts the computational savings. We propose an adaptive, tuning-free algorithm that sets the score loss weights during training based on loss gradients, adding minimal overhead to standard classifier training. We evaluate our approach on case studies involving network dynamics and spatial processes, demonstrating that it improves surrogate quality at a drastica
    
[^407]: 记得遗忘：门控自适应位置编码

    Remember to Forget: Gated Adaptive Positional Encoding

    [https://arxiv.org/abs/2605.10414](https://arxiv.org/abs/2605.10414)

    提出GAPE（门控自适应位置编码），通过查询依赖和键依赖的双门控机制，在保持旋转位置编码几何结构的前提下，将内容感知偏置直接注入注意力logits，解决长序列外推时RoPE的分布外失效问题。

    

    旋转位置编码（RoPE）被广泛应用于现代大型语言模型中。然而，当序列长度超出训练时所见的范围时，旋转相位可能进入分布外区域，导致虚假的长程对齐、注意力发散以及检索性能下降。现有的补救措施只能部分解决这些问题，因为它们往往以牺牲局部位置分辨率来换取长上下文的稳定性。我们提出GAPE（门控自适应位置编码），这是一种可直接插入使用的位置编码增强方法，它在保持旋转几何结构的同时，将内容感知的偏置直接引入注意力logits中。GAPE通过一个查询依赖的门控（用于收缩无关上下文）和一个键依赖的门控（用于保留重要的远距离token），将基于距离的抑制与token重要性解耦。我们证明，受保护程度较弱的远距离上下文会随查询门控呈指数级衰减，而……

    arXiv:2605.10414v2 Announce Type: replace  Abstract: Rotary Positional Encoding (RoPE) is widely used in modern large language models. However, when sequences are extended beyond the range seen during training, rotary phases can enter out-of-distribution regimes, leading to spurious long-range alignments, diffuse attention, and degraded retrieval. Existing remedies only partially address these failures, as they often trade local positional resolution for long-context stability. We propose GAPE (Gated Adaptive Positional Encoding), a drop-in augmentation for positional encodings that introduces a content-aware bias directly into the attention logits while preserving the rotary geometry. GAPE decouples distance-based suppression from token importance through a query-dependent gate that contracts irrelevant context and a key-dependent gate that preserves salient distant tokens. We show that weakly protected distant context is exponentially attenuated as a function of the query gate, while
    
[^408]: 基于扩散模型与马尔可夫链蒙特卡洛的多目标间接低推力轨迹迁移学习

    Transfer Learning of Multiobjective Indirect Low-Thrust Trajectories Using Diffusion Models and Markov Chain Monte Carlo

    [https://arxiv.org/abs/2605.09125](https://arxiv.org/abs/2605.09125)

    该论文提出了一种将任务参数同伦变换与马尔可夫链蒙特卡洛采样相结合的迁移学习框架，以更高效地生成训练数据，从而利用扩散模型加速多目标间接低推力轨迹初步设计中的全局搜索。

    

    低推力航天器任务的初步设计是一个全局搜索问题，其特点是解空间景观复杂、目标众多且存在大量局部极小值。在这一阶段，任务参数往往尚未完全确定，需要针对不同的参数取值以高频率生成新的解。当与最优控制的间接方法相结合时，扩散模型可以通过学习代表高质量初始协态的分布来加速这一搜索过程。然而，生成训练数据的成本依然高昂，并且存在更好地利用历史数据的机会。我们提出了一种迁移学习框架，该框架将任务参数的同伦变换与马尔可夫链蒙特卡洛相结合，以更高效地生成训练数据。该方法将多目标优化问题重新表述为在协态空间中从非归一化目标分布进行采样。我们比较了三种MCMC……（摘要原文在此处截断）

    arXiv:2605.09125v2 Announce Type: replace-cross  Abstract: Preliminary low-thrust spacecraft mission design is a global search problem characterized by a complex solution landscape, multiple objectives, and numerous local minima. During this phase, mission parameters are often not yet fully defined, requiring new solutions to be generated at a high cadence across varying parameter values. When combined with the indirect approach to optimal control, diffusion models can accelerate this search by learning distributions that represent high-quality initial costates. However, generating training data remains expensive, and opportunities exist to better exploit past data. We propose a transfer-learning framework that combines homotopy in a mission parameter with Markov chain Monte Carlo (MCMC) to generate training data more efficiently. The approach reformulates a multiobjective optimization problem as sampling from an unnormalized target distribution in costate space. We compare three MCMC 
    
[^409]: 别把你的克罗内克积绕晕了：高维不完整网格上的高斯过程

    Don't Get Your Kroneckers in a Twist: Gaussian Processes on High-Dimensional Incomplete Grids

    [https://arxiv.org/abs/2605.08036](https://arxiv.org/abs/2605.08036)

    提出CUTS-GPR方法，通过将加性核与不完整网格结合以实现极快的核矩阵-向量乘积，使数值精确的高斯过程回归能够扩展到数百万数据点和数百个维度。

    

    我们提出了CUTS-GPR，这是一种在高维设置下执行数值精确高斯过程回归（GPR）的新方法。CUTS-GPR的核心组件是一个极快的核矩阵-向量乘积，其计算复杂度随训练数据量N呈近线性甚至线性扩展，随维度D呈低阶多项式扩展。这一优势是通过将加性核与不完整网格相结合，并利用由此产生的核矩阵结构来实现的。我们通过包含数十亿数据点和数千维度的基准测试验证了该矩阵-向量乘积的可扩展性。我们通过在高达N = 4,494,001和D = 500的合成数据集上运行完整的GPR计算（包括超参数优化），展示了CUTS-GPR的端到端可扩展性。作为一项现实且具有挑战性的测试，我们最终将CUTS-GPR应用于十个势能面（PES）数据集，其规模为N = 447,265和D = 24。

    arXiv:2605.08036v2 Announce Type: replace  Abstract: We introduce CUTS-GPR, a new method for performing numerically exact GPR in high-dimensional settings. The key component of CUTS-GPR is an extremely fast kernel matrix-vector product, which exhibits near-linear or even linear scaling with the amount of training data, $N$, and low-order polynomial scaling with dimensionality, $D$. This is obtained by combining an additive kernel with an incomplete grid and exploiting the resulting structure of the kernel matrix. The scalability of the matrix-vector product is verified by benchmarks with billions of data points and thousands of dimensions. We demonstrate the end-to-end scalability of CUTS-GPR by running full GPR calculations, including hyperparameter optimization, on synthetic datasets with up to $N = 4\,494\,001$ and $D = 500$. As a realistic and challenging test, we finally apply CUTS-GPR to a set of ten potential energy surfaces (PESs) with $N = 447\,265$ and $D = 24$. The calculati
    
[^410]: 扰动二阶校准的极小极大速率

    The Minimax Rate of Perturbed Second-Order Calibration

    [https://arxiv.org/abs/2605.07808](https://arxiv.org/abs/2605.07808)

    本文提出通过向分类器分数添加 sech 噪声并进行低次多项式回归来估计二阶校准误差，达到 $O(\log^{3/2}n/\sqrt n)$ 的误差率，并通过匹配的 $\Omega(1/\sqrt{n})$ 下界证明了其在相差对数因子意义下的极小极大最优性。

    

    二阶校准误差量化了高阶预测器的认知不确定性估计与其水平集上标签概率条件方差的匹配程度。我们刻画了在分类器输出受到小扰动的情形下，二分类任务中估计二阶校准误差的极小极大速率。我们的方法很简单：向分数坐标添加独立的带宽为 $h$ 的 sech 噪声，然后使用低次多项式将 $Y^{(1)}$ 和 $Y^{(1)}Y^{(2)}$ 对扰动后的分数进行回归。关键在于，sech 扰动使得校准函数在适当的带形区域内是解析的。所得估计器的误差为 $O_h(\log^{3/2}n/\sqrt n)$，且常数为显式形式。在同一设定下，匹配的 $\Omega(1/\sqrt{n})$ 下界确立了该估计在相差对数因子意义下的极小极大最优性。作为推论，我们为扰动二阶（校准）给出了有限样本保证。

    arXiv:2605.07808v2 Announce Type: replace  Abstract: Second-order calibration error quantifies how closely a higher-order predictor's epistemic-uncertainty estimate matches the conditional variance of the label probability on its level sets. We characterize the minimax rate of estimating the second-order calibration error for binary classification in the regime where a small perturbation is applied to the classifier outputs. Our procedure is simple: add independent bandwidth-$h$ sech noise to the score coordinates, then regress $Y^{(1)}$ and $Y^{(1)}Y^{(2)}$ on the perturbed score using low-degree polynomials. Crucially, the sech perturbation makes the calibration functions analytic in a suitable strip. The resulting estimator has error $O_h(\log^{3/2}n/\sqrt n)$, with explicit constants. In the same setting, a matching $\Omega(1/\sqrt{n})$ lower bound establishes minimax optimality up to logarithmic factors. As a corollary, we give a finite-sample guarantee for perturbed second-order 
    
[^411]: LiteGUI：通过多解引导蒸馏与双层强化学习构建轻量级GUI智能体

    LiteGUI: Lightweight GUI Agents via Multi-Solution Guided Distillation and Dual-Level Reinforcement Learning

    [https://arxiv.org/abs/2605.07505](https://arxiv.org/abs/2605.07505)

    LiteGUI通过“引导式在线策略蒸馏”与“多解双层GRPO强化学习”的两阶段后训练框架，使轻量级GUI智能体能够有效应对复杂任务的长时程特性和多条有效交互路径的挑战。

    

    我们提出了LiteGUI，一个用于构建轻量级GUI智能体的新框架。由于复杂任务的长时程特性以及存在多条有效交互路径，GUI交互带来了独特的挑战，而轻量级GUI智能体难以有效应对这些挑战。为了解决这些问题，LiteGUI引入了一个在记录的GUI状态上运行的两阶段后训练框架。首先，我们提出引导式在线策略蒸馏，通过从人工验证的多解动作标注中选择最匹配的有效动作，在每个GUI状态下提供训练时的特权教师指导，同时保持学生的在线策略rollout不变。其次，我们开发了多解双层GRPO，将动作级监督与基于历史条件、按状态评估的规划质量监督相结合，同时兼顾每个记录的GUI状态下存在的多个有效动作。这两个阶段共同支持多步……（摘要在此处截断）

    arXiv:2605.07505v2 Announce Type: replace  Abstract: We present LiteGUI, a new framework for building lightweight GUI agents. GUI interaction poses unique challenges due to the long-horizon nature of complex tasks and the existence of multiple valid interaction paths, which are difficult for lightweight GUI agents to handle effectively. To address these challenges, LiteGUI introduces a two-stage post-training framework operating on logged GUI states. First, we propose Guided On-Policy Distillation, which provides training-time privileged teacher guidance at each GUI state by selecting the most-matched valid action from human-verified multi-solution action annotations, while leaving the student's on-policy rollout unchanged. Second, we develop Multi-Solution Dual-Level GRPO, which combines action-level supervision with history-conditioned, per-state planning-quality supervision while accounting for multiple valid actions at each logged GUI state. Together, these stages support multi-ste
    
[^412]: 图像分类器中单连通决策区域的实证证据

    Empirical Evidence for Simply Connected Decision Regions in Image Classifiers

    [https://arxiv.org/abs/2605.06380](https://arxiv.org/abs/2605.06380)

    本文通过自适应四边形网格填充实验首次提供了实证证据，表明预训练图像分类器中同标签决策区域是单连通的，即区域内的任意环路都可以被区域内曲面填充。

    

    分类器决策区域的拓扑结构决定了具有相同预测标签的输入如何在不改变预测结果的情况下被连接和变形。先前的实证工作已在单个区域内构建了同标签图像之间的路径，但并未检验该区域内的环路是否能界定出位于该区域内的曲面。我们使用自适应四边形网格来研究这一问题，并对偏离标签的内部顶点进行针对性修复，同时保持同标签边界环路固定。一个有限分辨率的接受准则用于区分已成功完成的构造与在细化上限处仍未能解决的构造。在所研究的预训练分类器中，每一个被测试的环路都获得了可接受的填充。构造工作量在同一类别内部相差数个数量级，并且经过均值分数调整的随机初始化分类器所需的构造工作量大于经过训练的分类器。作为解析对照的具有已知孔洞的模型则使其缠绕环路保持未解决状态……

    arXiv:2605.06380v2 Announce Type: replace-cross  Abstract: The topology of a classifier's decision regions determines how inputs with the same predicted label can be connected and deformed without changing that prediction. Prior empirical work constructed paths between same-label images within a single region, but did not examine whether loops bound surfaces within that region. We investigate this question using adaptive quadrilateral meshes with targeted repair of off-label interior vertices, while holding the same-label boundary loop fixed. A finite-resolution acceptance criterion distinguishes completed constructions from those left unresolved at the refinement ceiling. Across the pretrained classifiers studied, every tested loop admits an accepted filling. Construction effort varies by orders of magnitude within classes and is greater for mean-score-adjusted randomly initialised classifiers than for trained classifiers. An analytic control with a known hole leaves winding loops unr
    
[^413]: 顿悟还是故障？低精度如何驱动“弹弓机制”式损失尖峰

    Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes

    [https://arxiv.org/abs/2605.06152](https://arxiv.org/abs/2605.06152)

    本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。

    

    深度神经网络在无正则化的长期训练过程中会表现出周期性的损失尖峰，这一现象被称为“弹弓机制”。现有工作通常将其归因于内在的优化动力学，但其触发机制仍不清楚。本文证明该现象是浮点算术精度极限的结果：当训练进入高置信度阶段后，正确类别 logit 与其他 logit 之间的差值可能超过吸收误差阈值。于是在反向传播过程中，正确类别的梯度被精确舍入为零，而错误类别的梯度仍保持非零。这打破了跨类别梯度的零和约束，并在分类器层的参数更新中引入了系统性漂移。我们证明该漂移与特征之间形成了正反馈回路，导致全局分类器均值与全局特征（摘要在此处截断）。

    arXiv:2605.06152v4 Announce Type: replace-cross  Abstract: Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global featu
    
[^414]: 状态流Transformer (SST) V2：面向潜在空间推理的非线性递归并行训练

    State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning

    [https://arxiv.org/abs/2605.00206](https://arxiv.org/abs/2605.00206)

    SST V2通过在每层引入FFN驱动的非线性递归，使潜在状态横向流经整个序列，实现连续潜在空间中的参数高效推理与深思，并采用两遍并行训练使其计算上可行。

    

    当前的Transformer在位置之间丢弃了其丰富的潜在残差流，在每个新位置重新构建潜在推理上下文，导致潜在的推理能力未被充分利用。状态流Transformer (SST) V2通过在每个解码器层引入由FFN驱动的非线性递归，实现了在连续潜在空间中的参数高效推理，其中潜在状态通过学习到的混合方式沿整个序列横向流动传输。这一机制还支持在推理时对每个位置进行连续的潜在深思，在生成token之前投入额外的计算量来探索抽象推理。一种两遍并行训练程序近似了序列递归，使得共同训练在计算上切实可行。隐状态分析表明，状态流通过连续潜在空间中急剧的、依赖于内容的重组来促进推理，最终由语言模型头输出结果。

    arXiv:2605.00206v2 Announce Type: replace-cross  Abstract: Current transformers discard their rich latent residual stream between positions, reconstructing latent reasoning context at each new position and leaving potential reasoning capacity untapped. The State Stream Transformer (SST) V2 enables parameter-efficient reasoning in continuous latent space through an FFN-driven nonlinear recurrence at each decoder layer, where latent states are streamed horizontally across the full sequence via a learned blend. This same mechanism supports continuous latent deliberation per position at inference time, dedicating additional FLOPs to exploring abstract reasoning before committing to a token. A two-pass parallel training procedure approximates the sequential recurrence, making co-training computationally practical. Hidden state analysis shows that the state stream facilitates reasoning through sharp, content-dependent reorganisations in continuous latent space; the LM head exposes the result
    
[^415]: CMGL：面向癌症亚型分类的置信度引导多组学图学习

    CMGL: Confidence-guided Multi-omics Graph Learning for Cancer Subtype Classification

    [https://arxiv.org/abs/2604.24201](https://arxiv.org/abs/2604.24201)

    提出了CMGL方法，利用证据深度学习为每个患者估计各模态的置信度以指导多组学融合，并在独立构建的共识一致性图上进行图分类，从而提升癌症亚型分类的可靠性。

    

    动机：多组学整合可以改善癌症亚型分型，但模态的信息量和噪声在不同癌症类型和患者之间存在差异。大多数针对多组学数据的图方法是在下游分类目标中学习模态贡献，使得每个患者的预测可靠性处于隐式状态。因此，信息量不足的模态会削弱融合表征，而不可靠的组学数据会在图传播中引入嘈杂的患者关系。为了解决这两个问题，我们提出了CMGL，它在融合之前产生单独的可靠性估计，并使用共识患者邻域进行图分类。结果：CMGL通过证据深度学习为每个患者估计模态置信度，在跨组学融合期间固定这些值，并在独立构建的一致性图上执行分类。在四个MLOmics癌症亚型任务和32类泛癌症任务上……（摘要原文在此处截断）

    arXiv:2604.24201v2 Announce Type: replace  Abstract: Motivation: Multi-omics integration can improve cancer subtyping, but modality informativeness and noise vary across cancer types and patients. Most graph methods for multi-omics data learn modality contributions within the downstream classification objective, leaving predictive reliability for each patient implicit. As a result, uninformative modalities can weaken the fused representation, while unreliable omics can introduce noisy patient relationships into graph propagation. To address these two problems, we propose CMGL, which produces a separate reliability estimate before fusion and uses consensus patient neighborhoods for graph classification.   Results: CMGL estimates modality confidence for each patient through evidential deep learning, fixes these values during fusion across omics, and performs classification on an independently specified consistency graph. On four MLOmics cancer-subtype tasks and the 32-class pan-cancer ta
    
[^416]: 学习模拟混沌：对抗性最优传输正则化

    Learning to Emulate Chaos: Adversarial Optimal Transport Regularization

    [https://arxiv.org/abs/2604.21097](https://arxiv.org/abs/2604.21097)

    提出对抗性最优传输正则化方法，能够仅从单一含噪轨迹中联合学习高质量的摘要统计量与物理一致的混沌动力学模拟器。

    

    混沌现象存在于许多复杂动力系统中，从天气到电网，但难以用机器学习模拟器等数据驱动方法进行准确建模。尽管模拟器是加速模拟求解和解决逆问题的有前景的工具，但它们在学习混沌动力学时仍然面临困难——对初始条件的敏感性使得精确的长期预测不可行，尤其是在数据含有噪声的情况下。近期的工作转而训练模拟器去匹配混沌吸引子的统计特性，但这些方法通常依赖于手工设计的摘要统计量，或需要大型、多样化的多环境数据集。在这项工作中，我们提出了一族对抗性最优传输目标函数，能够从单一含噪轨迹中联合学习高质量的摘要统计量以及物理一致的模拟器。我们对 Sinkhorn 散度公式（2-Wasserstein……

    arXiv:2604.21097v3 Announce Type: replace-cross  Abstract: Chaos arises in many complex dynamical systems, from weather to power grids, but is difficult to accurately model with data-driven methods such as machine learning emulators. While emulators are promising tools for accelerating simulations and solving inverse problems, they still struggle to learn chaotic dynamics, where sensitivity to initial conditions renders exact long-term forecasts infeasible, especially given noisy data. Recent work instead trains emulators to match the statistical properties of chaotic attractors, but these approaches often rely on handcrafted summary statistics or large, diverse multi-environment datasets. In this work, we propose a family of adversarial optimal transport objectives that can jointly learn high-quality summary statistics and a physically consistent emulator from a single noisy trajectory. We theoretically analyze and experimentally validate a Sinkhorn divergence formulation (2-Wasserste
    
[^417]: 具有一般函数逼近的可证明高效的离线到在线价值自适应

    Provably Efficient Offline-to-Online Value Adaptation with General Function Approximation

    [https://arxiv.org/abs/2604.13966](https://arxiv.org/abs/2604.13966)

    该论文在一般函数逼近下研究离线到在线强化学习的价值自适应，通过极小化极大下界刻画了该问题的固有困难，并提出O2O-LSVI算法，在新的结构条件下实现了可证明优于纯在线强化学习的样本复杂度。

    

    我们在一般函数逼近下研究离线到在线强化学习中的价值自适应问题。从一个不完美的离线预训练$Q$函数出发，学习者旨在仅利用有限的在线交互将其自适应到目标环境。我们首先通过建立极小化极大下界来刻画该设定的难度，表明即使预训练的$Q$函数接近最优$Q^\star$，在某些困难实例上，在线自适应的效率也可能不高于纯在线强化学习。从积极的一面来看，在离线预训练价值函数的一个新颖结构条件下，我们提出了O2O-LSVI，这是一种具有问题相关样本复杂度的自适应算法，可证明地优于纯在线强化学习。最后，我们通过神经网络实验补充了理论结果，验证了所提方法的实际有效性。

    arXiv:2604.13966v2 Announce Type: replace  Abstract: We study value adaptation in offline-to-online reinforcement learning under general function approximation. Starting from an imperfect offline pretrained $Q$-function, the learner aims to adapt it to the target environment using only a limited amount of online interaction. We first characterize the difficulty of this setting by establishing a minimax lower bound, showing that even when the pretrained $Q$-function is close to optimal $Q^\star$, online adaptation can be no more efficient than pure online RL on certain hard instances. On the positive side, under a novel structural condition on the offline-pretrained value functions, we propose O2O-LSVI, an adaptation algorithm with problem-dependent sample complexity that provably improves over pure online RL. Finally, we complement our theory with neural-network experiments that demonstrate the practical effectiveness of the proposed method.
    
[^418]: 基于异构时空图神经网络的区域供热网络虚拟智能计量

    Virtual Smart Metering in District Heating Networks via Heterogeneous Spatial-Temporal Graph Neural Networks

    [https://arxiv.org/abs/2604.10166](https://arxiv.org/abs/2604.10166)

    该论文提出利用异构时空图神经网络实现区域供热网络的虚拟智能计量，以在传感器稀疏分布且存在故障的条件下增强热力和水力状态的可观测性。

    

    热能网络的智能运行旨在通过数据驱动控制、预测性优化和早期故障检测来提高能源效率、可靠性和运行灵活性。实现这些目标依赖于充分的可观测性，即需要对热力和水力状态进行连续且分布良好的监测。然而，区域供热系统通常仪表配置稀疏，且经常受到传感器故障的影响，限制了监测能力。虚拟感知提供了一种提高可观测性的经济有效的手段，但其在实际中的开发和验证仍然有限。现有的数据驱动方法通常假设数据是密集且同步的，而解析模型则依赖于简化的水力和热力假设，可能无法充分刻画异构网络拓扑的行为。因此，对压力、流量和温度之间的耦合非线性依赖关系进行建模（摘要原文在此处截断）

    arXiv:2604.10166v2 Announce Type: replace-cross  Abstract: Intelligent operation of thermal energy networks aims to improve energy efficiency, reliability, and operational flexibility through data-driven control, predictive optimization, and early fault detection. Achieving these goals relies on sufficient observability, requiring continuous and well-distributed monitoring of thermal and hydraulic states. However, district heating systems are typically sparsely instrumented and frequently affected by sensor faults, limiting monitoring. Virtual sensing offers a cost-effective means to enhance observability, yet its development and validation remain limited in practice. Existing data-driven methods generally assume dense synchronized data, while analytical models rely on simplified hydraulic and thermal assumptions that may not adequately capture the behavior of heterogeneous network topologies. Consequently, modeling the coupled nonlinear dependencies between pressure, flow, and tempera
    
[^419]: ReCodeAgent：一种用于大规模代码库语言无关翻译与验证的多智能体工作流

    ReCodeAgent: A Multi-agent Workflow for Language-Agnostic Translation and Validation of Large-Scale Repositories

    [https://arxiv.org/abs/2604.07341](https://arxiv.org/abs/2604.07341)

    ReCodeAgent通过自主多智能体工作流，实现了仓库级代码翻译和验证的语言无关性，用户仅需指定源和目标编程语言即可自动处理整个仓库。

    

    大多数仓库级代码翻译和验证技术仅在单一源-目标编程语言（PL）对上进行了评估，这是由于适应新PL对所需的复杂工程工作。编程智能体能够实现仓库级代码翻译和验证的语言无关性：它们可以跨多种PL合成代码，并自主使用针对每种PL分析的现有工具。然而，现有技术尚未提供一种完全自主的智能体方法，用于大规模程序的仓库级代码翻译和验证。本文提出了ReCodeAgent，一种自主多智能体方法，用于语言无关的仓库级代码翻译和验证。用户只需提供源PL中的项目并指定目标PL，ReCodeAgent即可自动翻译和验证整个仓库。ReCodeAgent是首个实现高翻译质量的技术。

    arXiv:2604.07341v3 Announce Type: replace-cross  Abstract: Most repository-level code translation and validation techniques have been evaluated on a single source-target programming language (PL) pair, owing to the complex engineering effort required to adapt new PL pairs. Programming agents can enable PL-agnosticism in repository-level code translation and validation: they can synthesize code across many PLs and autonomously use existing tools specific to each PL's analysis. However, state-of-the-art has yet to offer a fully autonomous agentic approach for repository-level code translation and validation of large-scale programs. This paper proposes ReCodeAgent, an autonomous multi-agent approach for language-agnostic repository-level code translation and validation. Users only need to provide the project in the source PL and specify the target PL for ReCodeAgent to automatically translate and validate the entire repository. ReCodeAgent is the first technique to achieve high translatio
    
[^420]: 基于模拟攻击模式的跨筒仓联邦学习中动态搭便车者检测

    Dynamic Free-Rider Detection in Cross-Silo Federated Learning via Simulated Attack Patterns

    [https://arxiv.org/abs/2604.04611](https://arxiv.org/abs/2604.04611)

    该论文提出通过模拟攻击模式（包括新提出的自适应WEF伪装攻击）来检测跨筒仓联邦学习中的动态搭便车者，能够识别出前期诚实参与、后期伪造参数以窃取全局模型的恶意客户端。

    

    联邦学习（FL）使多个客户端能够通过聚合本地更新来协同训练全局模型，而无需共享私有数据。在这项工作中，我们关注跨筒仓（cross-silo）联邦学习，其中每个客户端通常代表一个独立的组织。然而，跨筒仓联邦学习面临“搭便车者”的挑战——这类客户端不执行实际训练却提交伪造的模型参数，从而在不做任何贡献的情况下获取全局模型。Chen等人提出了一种基于模型参数权重演化频率（WEF）的搭便车者检测方法。这种检测方法非常实用，因为它既不需要代理数据集也不需要预训练。然而，该方法难以检测那些在早期轮次表现诚实、之后转为搭便车行为的“动态”搭便车者，尤其是在增量权重攻击以及我们新提出的自适应WEF伪装攻击等模仿全局模型的攻击下。在本文中，我们……（原文摘要此处被截断）

    arXiv:2604.04611v3 Announce Type: replace  Abstract: Federated learning (FL) enables multiple clients to collaboratively train a global model by aggregating local updates without sharing private data. In this work, we focus on cross-silo FL, where each client typically represents an independent organization. However, cross-silo FL can face the challenge of free-riders, clients who submit fake model parameters without performing actual training to obtain the global model without contributing. Chen et al. proposed a free-rider detection method based on the weight evolving frequency (WEF) of model parameters. This detection approach is practical because it requires neither a proxy dataset nor pre-training. Nevertheless, it struggles to detect ``dynamic'' free-riders who behave honestly in early rounds and later switch to free-riding, particularly under global-model-mimicking attacks such as the delta weight attack and our newly proposed adaptive WEF-camouflage attack. In this paper, we pr
    
[^421]: 软锦标赛均衡：面向非传递成对比较的可微分集合值推断

    Soft Tournament Equilibrium: Differentiable Set-Valued Inference for Non-Transitive Pairwise Comparisons

    [https://arxiv.org/abs/2604.04328](https://arxiv.org/abs/2604.04328)

    本文提出软锦标赛均衡（STE），一种可微分神经网络层，通过归一化log-sum-exp可达性和覆盖计算，从互反成对概率中平滑推断顶级循环集和未被覆盖集，并具备近似、扰动和边界恢复的理论保证。

    

    软锦标赛均衡（STE）是一个可微分层，用于从互反成对概率中进行顶级循环集（TC）和未被覆盖集（UC）推断。归一化的对数-求和-指数（log-sum-exp）可达性与覆盖计算提供了平滑的分数，并带有近似、扰动和边界恢复的理论保证。我们通过受控合成研究和对已记录序数档案的重建，区分了结构监督、后验不确定性和最终集合决策。在共同的结构读出下，匹配轮次训练提升了F1分数，但在单独进行的等预算软UC比较中并未显示出优势。一项前瞻性的等预算硬UC研究将24个备选项下的选择性F1从普通/关系原生头的0.6327/0.6292提升至0.6676，而精确恢复率仍仅为1.16%。在按来源保留的36个人类档案上，一项单独的探索性比较显示，Jeffreys后验推断优于学习到的独立边模型和混合模型。

    arXiv:2604.04328v4 Announce Type: replace  Abstract: Soft Tournament Equilibrium (STE) is a differentiable layer for Top-Cycle (TC) and Uncovered-Set (UC) inference from reciprocal pair probabilities. Normalized log-sum-exp reachability and covering give smooth scores with approximation, perturbation, and margin-recovery bounds. We distinguish structural supervision, posterior uncertainty, and the final set decision through controlled synthetic studies and reconstruction of recorded ordinal profiles. Matched-epoch training improves F1 under a common structural readout, but a separate equal-budget soft-UC comparison does not demonstrate an advantage. A prospective equal-budget hard-UC study improves selective F1 at 24 alternatives from 0.6327/0.6292 for ordinary/relational native heads to 0.6676, while exact recovery remains only 1.16%. On 36 human profiles held out by source, a separate exploratory comparison favors Jeffreys posterior inference over learned independent-edge and mixture
    
[^422]: 传输神经网络：抑制性与兴奋性连接

    Transmission Neural Networks: Inhibitory and Excitatory Connections

    [https://arxiv.org/abs/2604.04246](https://arxiv.org/abs/2604.04246)

    本文将传输神经网络模型扩展至包含抑制性与兴奋性连接及神经递质群体，证明了考虑抑制作用的神经元发放概率刻画可等价表示为每个神经元具有2维连续状态的神经网络，并建立了神经递质数量趋于无穷时极限网络模型的稳定性与收缩性充分条件。

    

    本文将Gao和Caines在文献[1]-[3]中提出的传输神经网络模型进行扩展，以纳入抑制性连接和神经递质群体。扩展后的网络模型包含二值神经元状态、传输动力学以及抑制性和兴奋性连接。在技术性假设条件下，我们建立了神经元发放概率的刻画，并证明这种考虑抑制作用的刻画可以等价地表示为一个每个神经元具有2维连续状态的神经网络。此外，我们将神经递质群体纳入建模，并建立了当所有突触连接处的神经递质数量趋于无穷大时的极限网络模型。最后，我们为极限网络模型的稳定性和收缩性质建立了充分条件。

    arXiv:2604.04246v2 Announce Type: replace-cross  Abstract: This paper extends the Transmission Neural Network model proposed by Gao and Caines in [1]-[3] to incorporate inhibitory connections and neurotransmitter populations. The extended network model contains binary neuronal states, transmission dynamics, and inhibitory and excitatory connections. Under technical assumptions, we establish the characterization of the firing probabilities of neurons, and show that such a characterization considering inhibitions can be equivalently represented by a neural network where each neuron has a continuous state of dimension 2. Moreover, we incorporated neurotransmitter populations into the modeling and establish the limit network model when the number of neurotransmitters at all synaptic connections go to infinity. Finally, sufficient conditions for stability and contraction properties of the limit network model are established.
    
[^423]: 任务为中心的语言模型个性化联邦微调

    Task-Centric Personalized Federated Fine-Tuning of Language Models

    [https://arxiv.org/abs/2604.00050](https://arxiv.org/abs/2604.00050)

    提出了FedRouter，一种基于聚类的个性化联邦学习方法，通过为每个任务而非每个客户端构建专门模型，解决了异构任务下的泛化能力不足和客户端内部任务干扰问题。

    

    联邦学习（FL）已成为在分布式且涉及多样任务的私有数据集上训练语言模型的一项有前景的技术。然而，聚合在异构任务上训练的模型往往会降低单个客户端的整体性能。为了解决这个问题，个性化联邦学习旨在为每个客户端的数据分布定制模型。尽管这些方法提高了本地性能，但它们通常在两个方面缺乏鲁棒性：(i) 泛化能力：当客户端需要对未见过的任务进行预测，或面临数据分布变化时；(ii) 客户端内部任务干扰：当单个客户端的数据包含多个可能在本地训练期间相互干扰的分布时。为了应对这两个挑战，我们提出了FedRouter，一种基于聚类的个性化联邦学习方法，它为每个任务而非每个客户端构建专门的模型。FedRouter使用适配器来实现个性化……

    arXiv:2604.00050v3 Announce Type: replace-cross  Abstract: Federated Learning (FL) has emerged as a promising technique for training language models on distributed and private datasets of diverse tasks. However, aggregating models trained on heterogeneous tasks often degrades the overall performance of individual clients. To address this issue, Personalized FL (pFL) aims to create models tailored for each client's data distribution. Although these approaches improve local performance, they usually lack robustness in two aspects: (i) generalization: when clients must make predictions on unseen tasks, or face changes in their data distributions, and (ii) intra-client tasks interference: when a single client's data contains multiple distributions that may interfere with each other during local training. To tackle these two challenges, we propose FedRouter, a clustering-based pFL that builds specialized models for each task rather than for each client. FedRouter uses adapters to personaliz
    
[^424]: 停止探测，开始编码：为什么线性探针与稀疏自编码器在组合泛化上会失败

    Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation

    [https://arxiv.org/abs/2603.28744](https://arxiv.org/abs/2603.28744)

    该论文证明稀疏自编码器（SAE）因将稀疏推理摊销到固定编码器而存在系统性“摊销差距”，且其根源在于字典学习而非推理过程，这导致SAE在分布外组合偏移下无法像经典稀疏编码方法那样恢复概念空间的线性结构。

    

    线性表示假设指出，神经网络的激活以线性混合的方式编码高层概念。然而，在叠加状态下，这种编码是从高维概念空间到低维激活空间的投影，概念空间中的线性决策边界在投影之后未必仍保持线性。在这一设定下，采用逐样本迭代推理的经典稀疏编码方法可以借助压缩感知的保证来恢复潜在因子。相比之下，稀疏自编码器（SAE）将稀疏推理摊销到一个固定的编码器中，从而引入了系统性的差距。我们证明这种摊销差距在不同的训练集规模、潜在维度和稀疏度水平下均持续存在，导致SAE在分布外（OOD）的组合偏移下失效。通过分解该失效过程的受控实验，我们发现字典学习是限制因素（而非推理…

    arXiv:2603.28744v2 Announce Type: replace  Abstract: The linear representation hypothesis states that neural network activations encode high-level concepts as linear mixtures. However, under superposition, this encoding is a projection from a higher-dimensional concept space into a lower-dimensional activation space, and a linear decision boundary in the concept space need not remain linear after projection. In this setting, classical sparse coding methods with per-sample iterative inference leverage compressed sensing guarantees to recover latent factors. Sparse autoencoders (SAEs), on the other hand, amortise sparse inference into a fixed encoder, introducing a systematic gap. We show this amortisation gap persists across training set sizes, latent dimensions, and sparsity levels, causing SAEs to fail under out-of-distribution (OOD) compositional shifts. Through controlled experiments that decompose the failure, we identify dictionary learning as the limiting factor (not the inferenc
    
[^425]: 基于预期不对称几何的二元因果方向识别

    Identification of Bivariate Causal Directionality Based on Anticipated Asymmetric Geometries

    [https://arxiv.org/abs/2603.26024](https://arxiv.org/abs/2603.26024)

    本文提出了两种基于条件分布的新方法——预期不对称几何（AAG）和单调性指数（MI），用于识别二元数值数据中的因果方向性。

    

    arXiv:2603.26024v2 公告类型：替换 摘要：识别二元数值数据中的因果方向性是一个基础性研究问题，具有重要的实际应用意义。本文提出了两种通过考虑条件分布来识别因果方向的替代方法：（1）预期不对称几何（AAG）和（2）单调性指数（MI）。AAG方法将实际的条件分布与沿两个变量的预期分布进行比较，并评估了多种比较度量，如皮尔逊相关系数、余弦距离、杰卡德指数、K-L散度、K-S距离、平均绝对误差（MAE）、均方误差（MSE）和互信息。预期分布基于双重响应统计量（均值和标准差）被投影为正态分布。MI方法比较沿两个轴计算的条件分布梯度的单调性指数，并展示梯度符号变化的次数。两种方法均假设……的随机特性（原文在此处截断）。

    arXiv:2603.26024v2 Announce Type: replace  Abstract: Identification of causal directionality in bivariate numerical data is a fundamental research problem with important practical implications. This paper presents two alternative methods to identify direction of causation by considering conditional distributions: (1) Anticipated Asymmetric Geometries (AAG) and (2) Monotonicity Index (MI). The AAG method compares the actual conditional distributions to anticipated ones along two variables. Different comparison metrics, such as Pearson correlation, cosine distance, Jaccard index, K-L divergence, K-S distance, MAE, MSE, and mutual information have been evaluated. Anticipated distributions have been projected as normal based on dual response statistics: mean and standard deviation. The MI method compares the calculated monotonicity indexes of the gradients of conditional distributions along two axes and exhibits counts of gradient sign changes. Both methods assume stochastic properties of 
    
[^426]: 构造即可判定：面向真正“无畏”系统的设计时验证

    Decidable By Construction: Design-Time Verification for Truly Fearless Systems

    [https://arxiv.org/abs/2603.25414](https://arxiv.org/abs/2603.25414)

    提出 Composer 编译器设计，通过四个递进的验证层级在设计时追踪并发等待关系、验证多线程正确性并保留分布式边界契约，直接将 Clef 语言编译为原生 CPU/GPU/NPU/FPGA 代码，实现“构造即可判定”的可验证系统。

    

    摘要：当编译器能够追踪等待关系边、证明多线程工作的正确性并保留分布式边界契约时，并发、并行和分布式执行才能真正实现“无畏”。在这一设计中，我们的 Composer 编译器在将 Clef 语言直接降级为原生 CPU、GPU、NPU 和 FPGA 代码的过程中保留证明，而无需通过 C 或厂商 API 进行转换。我们的程序语义图保留了证明 BAREWire 非装箱边界契约正确性所需的值、关系和前提条件。C 与 C++ 的接口被设计为一个显式的封送边界，证明与实际的参数、转换和返回值绑定。四个验证层级构成了我们设计的完整说明。第 1 层提供结构推理，建立在主维度类型的基础上。第 2 层在节点级别生成并检查局部算术、表示和计算完整性。第 3 层实例化可复用的领域引理和系统引理，包括分布式……（摘要在此处被截断）

    arXiv:2603.25414v5 Announce Type: replace-cross  Abstract: Concurrency, parallelism and distributed execution become truly fearless when the compiler tracks wait-for edges, proves multi-threaded work is sound and preserves distributed boundary contracts. In this design, our Composer compiler preserves proofs while lowering Clef directly to native CPU, GPU, NPU and FPGA code, without translation through C or vendor APIs. Our Program Semantic Graph retains the values, relationships and premises that justify BAREWire's unboxed boundary contracts. And C & C++ interfacing is an explicit marshaling boundary, with proofs tied to actual arguments, conversions and returned values.   Four verification tiers connect an account of our design. Tier 1 supplies structural inference, founded on principal dimensional types. Tier 2 generates and checks local arithmetic, representation, and computational integrity at the node level. Tier 3 instantiates reusable domain and system lemmas, including distrib
    
[^427]: 超越样本复制：扩散模型中的结构化记忆

    Beyond Sample Copying: Structural Memorization in Diffusion Models

    [https://arxiv.org/abs/2603.13419](https://arxiv.org/abs/2603.13419)

    该论文揭示了扩散模型会逐步过拟合去噪训练目标，而模型误差恰好抑制了对训练点的精确记忆，使去噪流场保持平滑，正是这种相互作用诱导了扩散模型的泛化能力。

    

    扩散模型在实践中具有良好的泛化能力。然而矛盾的是，一个最优的扩散模型会完全记住训练数据，因而无法泛化，这引出了一个问题：是什么促使真实的扩散模型产生泛化？我们证明，扩散模型会逐渐对去噪训练目标过拟合，从而在中度噪声水平上造成验证性能与训练性能之间的泛化差距。在一个具有可控去噪误差、完全可解析的二维玩具模型中，我们将这一差距归因于模型误差与数据分布支撑密度之间的相互作用。最优的去噪流场会急剧地局域化在各个训练点周围，而模型误差则抑制了对训练点的精确复现，从而产生平滑且具有泛化能力的流场。最后，我们研究了训练时的过拟合如何在推理轨迹中表现出来，发现由中间轨迹作出的预测……（摘要在此处截断）

    arXiv:2603.13419v3 Announce Type: replace  Abstract: Diffusion models generalize well in practice. Paradoxically, an optimal diffusion model fully memorizes the training data and therefore fails to generalize, raising the question of what induces generalization in a real diffusion model. We show that diffusion models progressively overfit the denoising training objective, creating a generalization gap between validation and training performance at intermediate noise levels. In a fully analytic 2D toy model with a controlled denoising error, we trace this gap to the interaction between model error and the density of the data distribution's support. The optimal denoising flow field localizes sharply around individual training points, whereas model error suppresses exact recall of training points, yielding a smooth, generalizing flow field. Finally, we examine how training-time overfitting manifests along inference trajectories. We find that predictions made from intermediate trajectory s
    
[^428]: PhysMoDPO：通过偏好优化实现物理合理的人形机器人运动

    PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization

    [https://arxiv.org/abs/2603.13228](https://arxiv.org/abs/2603.13228)

    该论文提出PhysMoDPO框架，将全身控制器集成到直接偏好优化的训练流程中，利用基于物理和任务的奖励来优化扩散模型，使生成的运动轨迹既符合物理规律又忠实于文本指令。

    

    近年来，文本条件下人体运动生成的进展主要由在大规模人体运动数据上训练的扩散模型所推动。在此基础上，近期的一些方法尝试将此类模型迁移至角色动画和真实机器人控制中，其做法是应用全身控制器（Whole-Body Controller, WBC），将扩散模型生成的运动转换为可执行的轨迹。虽然WBC生成的轨迹符合物理规律，但它们可能与原始运动存在较大偏差。为解决这一问题，我们提出了PhysMoDPO，一种直接偏好优化框架。与以往依赖手工设计的物理感知启发式方法（如脚部滑动惩罚）的工作不同，我们将WBC集成到训练流程中，并对扩散模型进行优化，使WBC的输出既符合物理规律，又忠实于原始文本指令。为了训练PhysMoDPO，我们部署了基于物理的奖励和任务特定的奖励，并使用……（原文在此处截断）

    arXiv:2603.13228v3 Announce Type: replace-cross  Abstract: Recent progress in text-conditioned human motion generation has been largely driven by diffusion models trained on large-scale human motion data. Building on this progress, recent methods attempt to transfer such models for character animation and real robot control by applying a Whole-Body Controller (WBC) that converts diffusion-generated motions into executable trajectories. While WBC trajectories become compliant with physics, they may expose substantial deviations from original motion. To address this issue, we here propose PhysMoDPO, a Direct Preference Optimization framework. Unlike prior work that relies on hand-crafted physics-aware heuristics such as foot-sliding penalties, we integrate WBC into our training pipeline and optimize diffusion model such that the output of WBC becomes compliant both with physics and original text instructions. To train PhysMoDPO we deploy physics-based and task-specific rewards and use th
    
[^429]: 改变思维，改变行动：推理链作为视觉-语言-动作策略的控制界面

    Altered Thoughts, Altered Actions: Reasoning Chain as Control Surface for a Vision-Language-Action Policy

    [https://arxiv.org/abs/2603.12717](https://arxiv.org/abs/2603.12717)

    该论文通过确定性实体交换实验系统测量了编辑推理链对视觉-语言-动作策略运动的修复与破坏作用，发现其影响集中在仅由语言决定目标的任务上，从而确立推理链可作为策略的有效控制界面。

    

    arXiv:2603.12717v2 公告类型：replace-cross。摘要：视觉-语言-动作策略将相机图像和自然语言指令映射为机器人的运动动作。其中一些策略被设计为在行动前以文本形式进行推理，生成一条推理链，并基于该链解码动作。提出这一设计的工作将推理链作为一种监督接口：人类可以阅读和编辑的文本，用于纠正策略。然而，编辑推理链会对策略的运动动作产生什么影响——是修复动作还是破坏动作——仍是一个悬而未决的问题。我们采用确定性实体交换方法，将其分别应用于策略接收的指令和策略生成的推理链，从而同时测量修复与破坏这两个方向。在全部四个LIBERO仿真套件、共四十个任务的观察背景下，实验表明破坏推理链的代价集中在仅由语言决定目标的场景。在那里，即LIBERO-Goal上，我们运行了决定性的对照实验（摘要原文在此截断）。

    arXiv:2603.12717v2 Announce Type: replace-cross  Abstract: Vision-language-action policies map camera images and natural-language instructions to a robot's motor actions. Some of these policies are designed to reason in text before acting, generating a reasoning chain and decoding actions conditioned on that chain. The works introducing this design offer the reasoning chain as an oversight interface: text a person can read and edit to correct the policy. What an edited reasoning chain does to the policy's motor actions, whether it repairs them or corrupts them, remains an open question. We measure both directions, repair and corruption, with our deterministic entity swap applied to the instruction the policy receives and to the reasoning chain it generates. A forty-task observed backdrop across all four LIBERO simulation suites reveals that the cost of corrupting the reasoning chain concentrates where language alone determines the goal. There, on LIBERO-Goal, we run the decisive counte
    
[^430]: 大语言模型的越狱缩放定律：多项式-指数交叉

    Jailbreak Scaling Laws for Large Language Models: Polynomial-Exponential Crossover

    [https://arxiv.org/abs/2603.11331](https://arxiv.org/abs/2603.11331)

    本文发现对抗性提示注入攻击能将大语言模型的攻击成功率从多项式增长放大为指数增长，并提出了基于自旋玻璃系统的理论生成模型来解释这两种缩放定律背后的最小统计机制。

    

    对抗性攻击能够可靠地引导与安全对齐的大语言模型产生不安全行为。通过实证研究，我们发现对抗性提示注入攻击能够将攻击成功率从无注入时观察到的缓慢多项式增长，放大为随推理时样本数量呈指数增长。我们首先通过给出一组关于跨上下文安全生成分布的少量假设，确定了这两种缩放规律的最小统计机制，在这组假设下两种缩放定律均可推导得出。为了进一步解释这一现象，我们提出了一个代理语言的理论生成模型，该模型以工作在复制对称性破缺状态下的自旋玻璃系统来表述，其中生成内容从相关的吉布斯测度中抽取，而低能量、尺寸偏置的簇中的一个子集被指定为不安全。我们通过解析方法展示了该模型如何自然地实现这些最小假设。简短的注入提示……

    arXiv:2603.11331v4 Announce Type: replace-cross  Abstract: Adversarial attacks can reliably steer safety-aligned large language models toward unsafe behavior. Empirically, we find that adversarial prompt-injection attacks can amplify attack success rate from the slow polynomial growth observed without injection to exponential growth with the number of inference-time samples. We first identify a minimal statistical mechanism for these two regimes by giving a small set of assumptions on the distribution of safe generation across contexts under which both scaling laws follow. To explain this phenomenon further, we propose a theoretical generative model of proxy language in terms of a spin-glass system operating in a replica-symmetry-breaking regime, where generations are drawn from the associated Gibbs measure and a subset of low-energy, size-biased clusters is designated unsafe. We analytically show how this model naturally realizes the minimal assumptions. Short injected prompts corresp
    
[^431]: 智能体批判性训练

    Agentic Critical Training

    [https://arxiv.org/abs/2603.08706](https://arxiv.org/abs/2603.08706)

    该论文提出智能体批判性训练，利用可验证奖励的强化学习让模型学会直接区分专家动作与看似合理的错误动作，作为模仿学习前的高效热身方法，无需参考推理且支持数据跨模型复用。

    

    模仿学习（IL）教会语言模型智能体复现专家动作，却无法让它们将这些动作与看似合理的错误区分开来。自我反思方法虽然让模型接触到替代方案，却使用监督微调（SFT）来模仿固定的推理和动作。我们提出了智能体批判性训练，它利用可验证奖励的强化学习（RLVR）来训练模型直接判别动作。在每个专家轨迹状态处，ACT将专家动作与从初始策略中采样的替代动作配对，并随机化二者的顺序。模型自行生成推理，但只有在选择专家动作时才获得奖励。ACT可以复用演示数据，无需参考推理文本，并允许配对数据在不同模型规模间复用。ACT是模仿学习之前的热身步骤，之后可选择继续进行强化学习；推理时无需进行候选动作比较。在Qwen3-8B和Olmo-3-7B-Instruct模型上，针对ALFWorld-ID、WebShop和ScienceWorld基准，ACT带来了（性能提升，摘要在此处截断）……

    arXiv:2603.08706v2 Announce Type: replace  Abstract: Imitation learning (IL) teaches language-model agents to reproduce expert actions but not to distinguish them from plausible mistakes. Self-reflection methods expose models to alternatives yet use supervised fine-tuning (SFT) to imitate fixed rationales and actions. We introduce Agentic Critical Training (ACT), which uses reinforcement learning with verifiable rewards (RLVR) to train models to judge actions directly. At each expert-trajectory state, ACT pairs an expert action with an alternative sampled from the initial policy and randomizes their order. The model generates its own reasoning but is rewarded only for selecting the expert action. ACT reuses demonstrations, requires no reference rationales, and allows pair reuse across model sizes. ACT is a warm-up before IL, optionally followed by RL; inference requires no candidate comparison. Across Qwen3-8B and Olmo-3-7B-Instruct on ALFWorld-ID, WebShop, and ScienceWorld, ACT yields
    
[^432]: \$OneMillion-Bench：语言智能体距离人类专家还有多远？

    \$OneMillion-Bench: How Far are Language Agents from Human Experts?

    [https://arxiv.org/abs/2603.07980](https://arxiv.org/abs/2603.07980)

    提出了包含 400 个专家设计的跨领域专业任务的 \$OneMillion-Bench 基准，通过基于评分规则的多维度评估，衡量语言智能体在法律、金融、医疗等经济关键场景中与人类专家的差距。

    

    随着语言模型从聊天助手演变为能够进行多步推理和工具使用的长程智能体，现有基准测试在很大程度上仍局限于结构化或考试式的任务，无法满足现实世界的专业需求。为此，我们推出了 \$OneMillion-Bench（\$OMB），这是一个包含 400 个由专家精心设计的任务的基准，涵盖法律、金融、工业、医疗健康和自然科学领域，旨在评估智能体在经济上具有重大影响的场景中的表现。与以往的工作不同，该基准要求检索权威来源、解决相互矛盾的证据、应用特定领域的规则并做出受约束的决策，其正确性既取决于最终答案，也取决于推理过程本身。我们采用基于评分规则的评估协议，从事实准确性、逻辑连贯性、实际可行性和专业合规性等维度进行评分，并专注于专家级问题，以确保有意义的区分度（原文摘要在此处截断）。

    arXiv:2603.07980v2 Announce Type: replace-cross  Abstract: As language models (LMs) evolve from chat assistants to long-horizon agents capable of multi-step reasoning and tool use, existing benchmarks remain largely confined to structured or exam-style tasks that fall short of real-world professional demands. To this end, we introduce \$OneMillion-Bench (\$OMB), a benchmark of 400 expert-curated tasks spanning Law, Finance, Industry, Healthcare, and Natural Science, built to evaluate agents across economically consequential scenarios. Unlike prior work, the benchmark requires retrieving authoritative sources, resolving conflicting evidence, applying domain-specific rules, and making constraint decisions, where correctness depends as much on the reasoning process as the final answer. We adopt a rubric-based evaluation protocol scoring factual accuracy, logical coherence, practical feasibility, and professional compliance, focusing on expert-level problems to ensure meaningful differenti
    
[^433]: 用于预测细胞对基因扰动响应的检索增强生成

    Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation

    [https://arxiv.org/abs/2603.07233](https://arxiv.org/abs/2603.07233)

    提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。

    

    预测基因扰动的转录响应是功能基因组学和治疗发现的基础。最近的深度学习模型在单细胞扰动响应预测方面展现出了前景，但它们通常孤立地生成每个响应，而没有显式地利用相关的扰动信息。我们提出了PT-RAG（扰动感知的两阶段检索增强生成），这是一个用于生成式细胞扰动响应的即插即用检索与条件化模块。PT-RAG通过学习访问相关的扰动上下文来增强现有的扰动-响应骨干模型。关键挑战在于，在这种设置下相关性并非固定不变：功能相关的基因在不同细胞类型中可能引发不同的效应。PT-RAG通过两阶段检索机制解决这一问题：首先基于GenePT的语义检索识别K个候选扰动，随后由一个可微分的Gumbel-Softmax选择器…

    arXiv:2603.07233v2 Announce Type: replace  Abstract: Predicting transcriptional responses to genetic perturbations is fundamental to functional genomics and therapeutic discovery. Recent deep learning models have shown promise in single-cell perturbation response prediction, but they typically generate each response in isolation, without explicitly leveraging related perturbations. We introduce PT-RAG (Perturbation-aware Two-stage Retrieval-Augmented Generation), a plug-in retrieval-and-conditioning module for generative cellular perturbation response. PT-RAG augments an existing perturbation-response backbone with learned access to related perturbation contexts. The key challenge is that relevance is not fixed in this setting: functionally related genes may elicit different effects across cell types. PT-RAG addresses this with a two-stage retrieval mechanism: GenePT-based semantic retrieval first identifies K candidate perturbations, after which a differentiable Gumbel-Softmax selecto
    
[^434]: 3BASiL：一种用于大语言模型稀疏加低秩压缩的算法框架

    3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs

    [https://arxiv.org/abs/2603.01376](https://arxiv.org/abs/2603.01376)

    该论文提出了3BASiL-TM，一种基于新颖三块ADMM算法的高效一次性后训练框架，通过带收敛保证的逐层重构误差最小化与跨Transformer层的联合精炼，实现大语言模型的稀疏加低秩压缩并显著缓解性能下降。

    

    大语言模型（LLM）的稀疏加低秩（S+LR）分解已成为模型压缩领域一个有前景的方向，其目标是将预训练模型的权重分解为稀疏矩阵与低秩矩阵之和（W ≈ S + LR）。尽管近期取得了一定进展，现有方法相比稠密模型往往存在显著的性能下降。在这项工作中，我们提出了 3BASiL-TM，一种针对大语言模型稀疏加低秩分解的高效一次性后训练方法，以填补这一空白。我们的方法首先提出了一种新颖的三块交替方向乘子法（3-Block ADMM），称为 3BASiL，用于在具有收敛保证的前提下最小化逐层重构误差。随后，我们设计了一个高效的 Transformer 匹配（TM）精炼步骤，在 Transformer 各层之间联合优化稀疏分量和低秩分量。该步骤最小化……（原文摘要在此处截断）

    arXiv:2603.01376v2 Announce Type: replace  Abstract: Sparse plus Low-Rank $(\mathbf{S} + \mathbf{LR})$ decomposition of Large Language Models (LLMs) has emerged as a promising direction in model compression, aiming to decompose pre-trained model weights into a sum of sparse and low-rank matrices $(\mathbf{W} \approx \mathbf{S} + \mathbf{LR})$. Despite recent progress, existing methods often suffer from substantial performance degradation compared to dense models. In this work, we introduce 3BASiL-TM, an efficient one-shot post-training method for $(\mathbf{S} + \mathbf{LR})$ decomposition of LLMs that addresses this gap. Our approach first introduces a novel 3-Block Alternating Direction Method of Multipliers (ADMM) method, termed 3BASiL, to minimize the layer-wise reconstruction error with convergence guarantees. We then design an efficient transformer-matching (TM) refinement step that jointly optimizes the sparse and low-rank components across transformer layers. This step minimizes
    
[^435]: V-ECE：估计广义期望校准误差

    V-ECE: Estimating General Expected Calibration Errors

    [https://arxiv.org/abs/2602.24230](https://arxiv.org/abs/2602.24230)

    该论文提出V-ECE方法，利用依赖预测的适当评分突破了以往方法仅能估计Bregman散度类校准误差的限制，实现了对包括$L_1$距离在内的一般凸散度（如$L_p$距离）校准误差在二分类和多分类场景下的可靠估计。

    

    在概率分类中，校准误差（CE）衡量的是预测概率 $f(X)$ 与 $\mathbb{P}(Y|f(X))$（即该预测概率所对应的真实类别分布）之间的平均散度。尽管校准误差是一种有用的诊断工具，但它难以估计：流行的基于分箱的估计器往往不一致，且在超过两个类别时扩展性较差。最近的研究将校准误差重写为模型相对于其自身预测的最佳重校准所产生的超额风险，并用适当损失来度量。然而，这种方法只适用于基于 Bregman 散度的校准误差（如平方误差），而排除了更常用的基于 $L_1$ 距离的校准误差。我们证明，使用依赖预测的适当评分可以缓解这一限制，使我们能够估计具有一般凸散度的校准误差，包括在二分类和多分类设置下具有闭式损失形式的 $L_p$ 距离。为了估计超额风险，我们引入了一个模……（原文摘要在此处截断）

    arXiv:2602.24230v2 Announce Type: replace-cross  Abstract: In probabilistic classification, calibration error (CE) measures the average divergence of predicted probabilities $f(X)$ from $\mathbb{P}(Y|f(X))$, the true class distribution for that predicted probability. While being a useful diagnostic tool, it is hard to estimate: popular binning-based estimators are often inconsistent and scale poorly beyond two classes. Recent work rewrites the CE as the excess risk of a model compared to the best recalibration of its own predictions, measured with a proper loss. However, this only works for Bregman-divergence-based calibration errors like the squared error, excluding the more popular $L_1$-distance-based CE. We show that using prediction-dependent proper scores can alleviate this restriction, allowing us to estimate CEs with general convex divergences, including $L_p$ distances with closed-form losses in the binary and multiclass settings. To estimate the excess risk, we introduce a mo
    
[^436]: PaReGTA：一种用于电子健康记录分析的基于大语言模型的时间感知患者表示框架

    PaReGTA: A Temporally Aware LLM-Based Patient Representation Framework for EHR Analytics

    [https://arxiv.org/abs/2602.19661](https://arxiv.org/abs/2602.19661)

    PaReGTA是一种基于大语言模型的时间感知患者表示框架，通过将纵向EHR事件转换为带显式时间线索的文本、轻量级对比微调学习就诊嵌入、以及混合时间池化聚合，生成可直接用于常规下游机器学习模型的固定维度患者表示。

    

    结构化电子健康记录（EHR）中的时间信息在稀疏的独热或基于计数的表示中常常丢失，而序列模型则可能成本高昂且需要大量数据。我们提出了PaReGTA，这是一种基于大语言模型的编码框架，它（i）将纵向EHR事件转换为带有显式时间线索的就诊级模板文本，（ii）通过对句子嵌入模型进行轻量级对比微调来学习领域适应的就诊嵌入，以及（iii）使用混合时间池化方法将就诊嵌入聚合为固定维度的患者表示，该池化方法既能捕捉近期就诊的信息，也能捕捉具有全局信息性的就诊。所得到的固定维度患者表示可以与常规的下游机器学习模型配合使用。为了检验因素级别的敏感性，我们使用PaReGTA-RSS（表示偏移分数），这是一种预定义的因素移除分析，在移除临床相关因素后重新计算患者表示……

    arXiv:2602.19661v3 Announce Type: replace  Abstract: Temporal information in structured electronic health records (EHRs) is often lost in sparse one-hot or count-based representations, while sequence models can be costly and data-hungry. We propose PaReGTA, an LLM-based encoding framework that (i) converts longitudinal EHR events into visit-level templated text with explicit temporal cues, (ii) learns domain-adapted visit embeddings via lightweight contrastive fine-tuning of a sentence-embedding model, and (iii) aggregates visit embeddings into a fixed-dimensional patient representation using hybrid temporal pooling that captures both recency and globally informative visits. The resulting fixed-dimensional patient representations can be used with conventional downstream machine-learning models. To examine factor-level sensitivity, we use PaReGTA-RSS (Representation Shift Score), a prespecified factor-removal analysis that recomputes patient representations after removing clinically def
    
[^437]: 联邦格兰杰因果学习中的不确定性量化

    Uncertainty Quantification in Federated Granger Causality Learning

    [https://arxiv.org/abs/2602.13004](https://arxiv.org/abs/2602.13004)

    本文针对客户端特征异构的联邦格兰杰因果学习场景，刻画了跨客户端依赖估计过程中的不确定性传播，并利用边特定的方差有效区分真实的跨客户端依赖关系与虚假的估计边。

    

    格兰杰因果性用于识别多元时间序列中的预测性依赖关系。在各方无法共享数据的分布式环境中，联邦因果学习使得联合分析成为可能。大多数联邦因果方法假设客户端观测到相同的特征，并以点估计的方式推断因果关系，缺乏正式的不确定性量化。这些假设在许多工业系统中并不成立——在工业系统中，客户端观测到不同的特征，且目标是要估计跨客户端的依赖关系（边）。这些依赖关系必须通过重复的客户端-服务器迭代来间接估计。来自客户端数据和模型参数的不确定性会在此过程中传播，使得仅凭点估计不足以评估跨客户端的边。本文刻画了这种不确定性传播过程，并利用边特定的方差来区分真实的跨客户端依赖关系与虚假的估计边。

    arXiv:2602.13004v3 Announce Type: replace  Abstract: Granger causality identifies predictive dependencies in multivariate time series. In distributed settings where parties cannot share data, federated causal learning enables joint analysis. Most federated causal methods assume that clients observe the same features and infer causal relationships as point estimates, with little formal uncertainty quantification. These assumptions do not hold in many industrial systems, where clients observe different features, and the objective is to estimate cross-client dependencies (edges). These dependencies must be estimated indirectly through repeated client-server iterations. Uncertainty from client data and model parameters propagates through this process, making point estimates alone insufficient for assessing cross-client edges. This paper characterizes this uncertainty propagation and uses edge-specific variances to distinguish genuine cross-client dependencies from spurious estimated edges.
    
[^438]: 基于条件流匹配的视觉引导音频增强

    Conditional Flow Matching for Visually-Guided Acoustic Highlighting

    [https://arxiv.org/abs/2602.03762](https://arxiv.org/abs/2602.03762)

    本文提出一种条件流匹配生成框架，通过引入展开损失惩罚最终步骤漂移，有效解决了视觉引导音频增强中判别模型难以处理音频重混模糊性的问题。

    

    arXiv:2602.03762v4 公告类型：替换交叉 摘要：视觉引导的音频增强旨在根据伴随视频重新平衡音频，以创造连贯的视听体验。虽然视觉显著性和增强已被广泛研究，但音频增强仍未被充分探索，常常导致视觉与听觉焦点之间的错位。现有方法使用判别模型，这些模型难以处理音频重混中固有的模糊性——在平衡不佳与平衡良好的音频混合之间，不存在自然的一一对应关系。为解决这一局限，我们将此任务重新定义为生成问题，并引入了一个条件流匹配（CFM）框架。迭代式流生成中的一个关键挑战是：早期预测错误（即选择正确的音频源进行增强）会随着步骤累积，使轨迹偏离流形。为解决此问题，我们引入了一个展开损失函数，用于惩罚最终步骤的漂移。

    arXiv:2602.03762v4 Announce Type: replace-cross  Abstract: Visually-guided acoustic highlighting seeks to rebalance audio in alignment with the accompanying video, creating a coherent audio-visual experience. While visual saliency and enhancement have been widely studied, acoustic highlighting remains underexplored, often leading to misalignment between visual and auditory focus. Existing approaches use discriminative models, which struggle with the inherent ambiguity in audio remixing, where no natural one-to-one mapping exists between poorly-balanced and well-balanced audio mixes. To address this limitation, we reframe this task as a generative problem and introduce a Conditional Flow Matching (CFM) framework. A key challenge in iterative flow-based generation is that early prediction errors -- in selecting the correct source to enhance -- compound over steps and push trajectories off-manifold. To address this, we introduce a rollout loss that penalizes drift at the final step, encou
    
[^439]: 基于复杂网络的合成时间序列生成

    Synthetic Time Series Generation via Complex Networks

    [https://arxiv.org/abs/2601.22879](https://arxiv.org/abs/2601.22879)

    本文通过结合统计特征、网络拓扑特性及下游任务性能的全面实证研究，首次系统评估了逆分位数图框架作为通用合成时间序列生成器在保真度和实用性方面的潜力。

    

    时间序列数据对广泛的应用领域至关重要，然而高质量数据集的获取常常受到隐私问题、采集成本和标注难题的限制。合成时间序列生成已成为解决这些局限的一种有前景的方法。在这项工作中，我们研究了使用复杂网络映射进行合成时间序列生成，重点关注分位数图表示及其逆映射。虽然逆QG映射此前已被提出，但其作为通用数据生成器的潜力尚未得到系统性评估。我们通过一项全面的实证研究来填补这一空白，评估由逆分位数图框架生成的合成时间序列的保真度和实用性。该评估结合了统计特征分析、基于网络的拓扑特性分析，以及在下游聚类和分类任务中的性能表现。

    arXiv:2601.22879v3 Announce Type: replace  Abstract: Time series data are essential for a wide range of applications, yet access to high-quality datasets is often constrained by privacy concerns, acquisition costs, and labelling challenges. Synthetic time series generation has emerged as a promising approach to address these limitations. In this work, we investigate the use of complex network mappings for synthetic time series generation, focusing on the Quantile Graph (QG) representation and its inverse. While the inverse QG mapping has been previously proposed, its potential as a general-purpose data generator has not been systematically evaluated. We address this gap through a comprehensive empirical study assessing both the fidelity and utility of synthetic time series generated by the Inverse Quantile Graph (InvQG) framework. The evaluation combines statistical feature analysis, network-based topological characteristics, and performance in downstream clustering and classification 
    
[^440]: 先验数据拟合网络中谱结构的机制性证据

    Mechanistic Evidence for Spectral Structures in Prior-Data Fitted Networks

    [https://arxiv.org/abs/2601.21731](https://arxiv.org/abs/2601.21731)

    本文通过线性探针与激活/子空间修补等机制可解释性方法，首次证明先验数据拟合网络内部以低维结构表征上下文的谱内容（即决定平稳核的量），且该谱信息可被读取为显式核。

    

    先验数据拟合网络在单次前向传播中执行近似贝叶斯推断，基于其构建的表格基础模型目前已被广泛使用。为了理解网络内部推断的内容，近期针对表格基础模型的机制可解释性研究定位了预测形成的位置，但将这些模型视为表格预测器，而非先验数据拟合网络。因此，先验数据拟合网络是否表征其上下文的谱内容——即决定平稳核的量——以及该内容能否被读取为显式核，这些问题仍不清楚。我们回答了这两个问题。首先，在七个先验数据拟合网络上，包括四个预训练的表格基础模型和一个仅在决策树先验上训练的模型，残差流上的线性探针能够以 R² ≥ 0.95 的精度恢复上下文的频率。该结构由单一主方向主导。其次，激活与子空间修补实验表明，网络通过低维……（原文摘要在此处截断）

    arXiv:2601.21731v3 Announce Type: replace  Abstract: Prior-Data Fitted Networks (PFNs) perform approximate Bayesian inference in a single forward pass, and tabular foundation models (TFMs) built on them are now widely used. To understand what networks infer internally, recent mechanistic studies of TFMs locate where predictions form, but treat these models as tabular predictors rather than as PFNs. It therefore remains unknown whether PFNs represent the spectral content of their context, the quantity that specifies a stationary kernel, and whether this content can be read out as an explicit kernel. We answer both questions. First, across seven PFNs, including four pretrained TFMs and a model trained only on a decision-tree prior, a linear probe on the residual stream recovers the frequency of the context with $R^2 \geq 0.95$. This structure is led by a single principal direction. Second, activation and subspace patching show that the network uses the structure through a low-dimensional
    
[^441]: 求解非光滑次模-凹函数的离线与在线极小极大问题：一种零阶方法

    Solving the Offline and Online Min-Max Problem of Non-smooth Submodular-Concave Functions: A Zeroth-Order Approach

    [https://arxiv.org/abs/2601.21243](https://arxiv.org/abs/2601.21243)

    本文提出一种结合Lovász扩展次梯度与高斯平滑的零阶方法，解决了非光滑次模-凹函数的离线与在线极小极大问题，证明了离线情形收敛到ε-鞍点，并在在线情形达到O(√(N(1+P̄_N)))的对偶间隙。

    

    我们考虑目标函数可能非光滑、关于极小化变量为次模函数且关于极大化变量为凹函数的极大-极小和极小-极大问题。我们研究了应用于该问题的零阶方法的性能。该方法基于目标函数关于极小化变量的Lovász扩展的次梯度，并基于高斯平滑来估计关于极大化变量的平滑函数梯度。在期望意义下，我们证明了该算法在离线情形下收敛到ε-鞍点。此外，我们证明在期望意义下，在在线设置中，该算法实现了O(√(N(1+P̄_N)))的在线对偶间隙，其中N是迭代次数，P̄_N是最优决策序列的路径长度。我们为所有情形给出了复杂度分析和超参数选择。

    arXiv:2601.21243v4 Announce Type: replace-cross  Abstract: We consider max-min and min-max problems with objective functions that are possibly non-smooth, submodular with respect to the minimiser and concave with respect to the maximiser. We investigate the performance of a zeroth-order method applied to this problem. The method is based on the subgradient of the Lov\'asz extension of the objective function with respect to the minimiser and based on Gaussian smoothing to estimate the smoothed function gradient with respect to the maximiser. In expectation sense, we prove the convergence of the algorithm to an $\epsilon$-saddle point in the offline case. Moreover, we show that, in the expectation sense, in the online setting, the algorithm achieves $O(\sqrt{N(1+\bar{P}_N)})$ online duality gap, where $N$ is the number of iterations and $\bar{P}_N$ is the path length of the sequence of optimal decisions. The complexity analysis and hyperparameter selection are presented for all the cases
    
[^442]: 信任、不信任还是反转：基于多专家反馈的鲁棒偏好强化学习

    Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback

    [https://arxiv.org/abs/2601.18751](https://arxiv.org/abs/2601.18751)

    提出TriTrust-PBRL框架，通过联合学习共享奖励模型与专家信任参数（自动演化为正、零或负值），能够信任、忽略或反转多专家偏好反馈，从而有效抵御对抗性标注者的干扰。

    

    基于偏好的强化学习（PBRL）通过从成对轨迹比较中学习，为显式奖励工程提供了一种有前景的替代方案。然而，现实世界中的偏好数据通常来自可靠性参差不齐的异质标注者：有些准确、有些含噪、有些则存在系统性的对抗行为。现有的PBRL方法要么平等对待所有反馈，要么试图过滤掉不可靠的来源，但当面对系统性提供错误偏好的对抗性标注者时，这两种方法都会失效。我们提出了TriTrust-PBRL（TTP），这是一个统一框架，能够从多专家偏好反馈中联合学习共享奖励模型和针对每位专家的信任参数。其关键洞察在于，在基于梯度的优化过程中，信任参数会自然地演化为正值（信任）、接近零（忽略）或负值（反转），从而使模型能够自动反转对抗性偏好，实现鲁棒的偏好学习。

    arXiv:2601.18751v2 Announce Type: replace-cross  Abstract: Preference-based reinforcement learning (PBRL) offers a promising alternative to explicit reward engineering by learning from pairwise trajectory comparisons. However, real-world preference data often comes from heterogeneous annotators with varying reliability; some accurate, some noisy, and some systematically adversarial. Existing PBRL methods either treat all feedback equally or attempt to filter out unreliable sources, but both approaches fail when faced with adversarial annotators who systematically provide incorrect preferences. We introduce TriTrust-PBRL (TTP), a unified framework that jointly learns a shared reward model and expert-specific trust parameters from multi-expert preference feedback. The key insight is that trust parameters naturally evolve during gradient-based optimization to be positive (trust), near zero (ignore), or negative (flip), enabling the model to automatically invert adversarial preferences and
    
[^443]: SHAP中的解释多样性：刻画与评估

    Explanation Multiplicity in SHAP: Characterization and Assessment

    [https://arxiv.org/abs/2601.12654](https://arxiv.org/abs/2601.12654)

    该论文揭示了SHAP解释在同一模型和输入的重复运行之间也会出现显著分歧（即“解释多样性”）这一现象，并提出了一套结合双种子协议、多层次度量指标和随机零模型的评估方法来对其进行系统刻画。

    

    SHAP解释被广泛应用于高风险场景中以证明决策的合理性，然而即使模型、输入实例和预测保持不变，其在多次重复运行之间也可能存在显著差异。先前的工作记录了不同解释方法之间的分歧；我们表明，即使在同一训练好的模型和实例上对同一估计器进行重复运行，SHAP内部也会出现实质性分歧。我们将这一现象称为解释多样性（explanation multiplicity），并开发了一种在贴近部署实际的计算预算下刻画该现象的评估方法。该方法结合了比较模型诱导变异与解释器诱导变异的双种子协议、由基于幅度、基于排名和基于集合的度量构成的层次体系，以及为观察到的分歧提供参考尺度的随机Dirichlet和Mallows零模型。在多个数据集、模型和采样策略上，我们发现解释多样性……

    arXiv:2601.12654v3 Announce Type: replace-cross  Abstract: SHAP explanations are widely used in high-stakes settings to justify decisions, yet they can differ substantially across repeated runs, even when the model, the input instance, and the prediction are held fixed. Prior work has documented disagreement between explanation methods; we show that substantial disagreement arises even within SHAP across reruns of the same estimator on the same trained model and instance. We call this phenomenon explanation multiplicity and develop an evaluation methodology for characterizing it under deployment-realistic computational budgets, combining a dual-seed protocol that compares model-induced and explainer-induced variability, a hierarchy of magnitude-based, rank-based, and set-based metrics, and randomized Dirichlet and Mallows null models that provide reference scales for observed disagreement. Across multiple datasets, models, and sampling strategies, we find that explanation multiplicity 
    
[^444]: 基于机器学习的小数据集预测性无机合成：以水动力直径可控的铜纳米颗粒为例

    Predictive Inorganic Synthesis based on Machine Learning using Small Data sets: a case study of Hydrodynamic Diameter-controlled Cu Nanoparticles

    [https://arxiv.org/abs/2512.16545](https://arxiv.org/abs/2512.16545)

    本研究证明仅需25次合成的小数据集，结合拉丁超立方采样与集成回归模型，即可有效预测铜纳米颗粒的水动力直径，为小数据条件下机器学习驱动的可控纳米材料合成提供了范例。

    

    铜纳米颗粒具有广泛的适用性，但其合成对反应参数的细微变化十分敏感。这种敏感性，加上实验优化耗时耗力的特点，使得实现可重复且尺寸可控的合成成为一大挑战。尽管机器学习在材料研究中展现出广阔前景，但其应用常受限于缺乏大规模高质量实验数据集。本研究探索了利用机器学习，基于仅含25次合成的小数据集来预测铜纳米颗粒的动态光散射（DLS）水动力直径。在构建实验数据集时，采用拉丁超立方采样以高效覆盖参数空间。集成回归模型在有限数据集条件下成功预测了水动力直径，展现出良好的预测性能。由于定量回归需要唯一的DLS水动力直径，回归模型仅限于单峰DLS分布的情形。

    arXiv:2512.16545v3 Announce Type: replace-cross  Abstract: Cu NPs have a broad applicability, yet their synthesis is sensitive to subtle changes in reaction parameters. This sensitivity, combined with the time- and resource-intensive nature of experimental optimization, poses a major challenge in achieving reproducible and size-controlled synthesis. While ML shows promise in materials research, its application is often limited by scarcity of large high-quality experimental data sets. This study explores ML to predict the DLS-derived hydrodynamic diameter of Cu NPs using a small data set of 25 syntheses. Latin Hypercube Sampling is used to efficiently cover the parameter space while creating the experimental data set. Ensemble regression models successfully predict hydrodynamic diameters with good predictive performance given the limited dataset. Since quantitative regression requires a unique DLS-derived hydrodynamic diameter, the regression model is restricted to mono-modal DLS distri
    
[^445]: 一次置换足矣：快速、确定性的特征重要性与模型压力测试

    One Permutation Is All You Need: Fast, Deterministic Feature Importance and Model Stress-Testing

    [https://arxiv.org/abs/2512.13892](https://arxiv.org/abs/2512.13892)

    用单次最大-最小秩最优的确定性置换替代多次随机置换，可将特征重要性估计的计算复杂度从 O(B·n·p) 降至 O(n·p)，消除估计方差并保持或提升估计精度，并可扩展用于模型压力测试。

    

    在机器学习模型中可靠地估计特征贡献对于透明度、算法公平性和监管合规至关重要。虽然置换特征重要性被广泛使用，但经典实现依赖于重复的蒙特卡洛洗牌，这引入了显著的计算开销和随机不稳定性。在本文中，我们证明用单一的、最大-最小秩最优的确定性置换替代 $B$ 次随机置换，能够在消除估计方差的同时保持或改善与真实重要性的相关性，并将复杂度从 $O(B \cdot n \cdot p)$ 降低至 $O(n \cdot p)$。在位置-尺度特征分布下，我们严格证明了尺度调整后线性回归系数的精确恢复，以及在凹型模型敏感度下改进的重要性估计。我们进一步沿两个互补维度扩展这一确定性框架。首先，系统特征

    arXiv:2512.13892v3 Announce Type: replace-cross  Abstract: Reliable estimation of feature contributions in machine learning models is essential for transparency, algorithmic fairness, and regulatory compliance. While permutation feature importance is widely used, classical implementations rely on repeated Monte Carlo shuffling, introducing significant computational overhead and stochastic instability. In this paper, we show that replacing $B$ random permutations with a single, max-min rank-optimal deterministic permutation maintains or improves correlation with ground-truth importance while eliminating estimation variance and reducing complexity from $O(B \cdot n \cdot p)$ to $O(n \cdot p)$. Under location-scale feature distributions, we formally prove exact recovery of scale-adjusted linear regression coefficients, alongside improved importance estimation under concave model sensitivity. We extend this deterministic framework along two complementary dimensions. First, Systemic Feature
    
[^446]: 用于解混流的自充分独立成分分析

    Self-sufficient Independent Component Analysis for Demixing Flows

    [https://arxiv.org/abs/2512.00665](https://arxiv.org/abs/2512.00665)

    提出一种无先验、无似然的自充分独立成分分析方法，通过最小化条件KL散度并顺序学习解混流模型来从数据中学习解耦信号，同时完全避免了不稳定的对抗训练。

    

    我们研究了利用非线性独立成分分析（ICA）从数据中学习解耦信号的问题。受自监督学习进展的启发，我们提出学习自充分信号：给定已恢复信号的其余值，观察其他信号不应改变其缺失值的条件分布。我们将该问题表述为条件KL散度的最小化。我们的算法是无先验且无似然的，即它既不规定参数化的源密度，也不规定观测似然。为解决KL散度最小化问题，我们提出了一种顺序算法，在每次迭代中学习一个解混流模型，并证明了其理想化的Wasserstein梯度流变体（具有精确速度和总体投影条件）在总相关性上具有局部下降性质。该方法完全避免了不稳定的对抗性训练。

    arXiv:2512.00665v2 Announce Type: replace-cross  Abstract: We study the problem of learning disentangled signals from data using non-linear Independent Component Analysis (ICA). Motivated by advances in self-supervised learning, we propose to learn self-sufficient signals: Given the remaining values of a recovered signal, observing other signals should not change the conditional distribution of its missing value. We formulate this problem as the minimization of a conditional KL divergence. Our algorithm is prior-free and likelihood-free in the sense that it prescribes neither parametric source densities nor an observation likelihood. To tackle the KL divergence minimization problem, we propose a sequential algorithm that learns a de-mixing flow model at each iteration, and prove local descent of the total correlation for its idealized Wasserstein-gradient-flow variant with exact velocities and a population projection condition. This approach completely avoids the unstable adversarial t
    
[^447]: 通过合成模型生成实现近最优可解释模型的可扩展元学习

    Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations

    [https://arxiv.org/abs/2511.04000](https://arxiv.org/abs/2511.04000)

    本文提出通过合成采样近最优决策树来生成大规模预训练数据的高效可扩展方法，使MetaTree transformer在决策树元学习上达到与真实数据或昂贵最优树预训练相当的性能，同时大幅降低计算成本。

    

    决策树因其可解释性而广泛应用于金融和医疗等高风险领域。本工作提出了一种高效、可扩展的方法来生成合成预训练数据，从而实现决策树的元学习。我们的方法通过合成方式采样近最优的决策树，构建出大规模、贴近现实的数据集。借助MetaTree transformer架构，我们证明该方法所取得的性能可与在真实数据上预训练或使用计算成本高昂的最优决策树预训练相媲美。该策略显著降低了计算成本，提升了数据生成的灵活性，并为可解释决策树模型的可扩展、高效元学习铺平了道路。

    arXiv:2511.04000v2 Announce Type: replace-cross  Abstract: Decision trees are widely used in high-stakes fields like finance and healthcare due to their interpretability. This work introduces an efficient, scalable method for generating synthetic pre-training data to enable meta-learning of decision trees. Our approach samples near-optimal decision trees synthetically, creating large-scale, realistic datasets. Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees. This strategy significantly reduces computational costs, enhances data generation flexibility, and paves the way for scalable and efficient meta-learning of interpretable decision tree models.
    
[^448]: LIME：基于链接的用户-物品交互建模与解耦XOR注意力机制，实现高效的测试时扩展

    LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling

    [https://arxiv.org/abs/2510.18239](https://arxiv.org/abs/2510.18239)

    LIME架构通过低秩链接嵌入解耦用户与候选交互、实现注意力权重预计算，并采用线性XOR注意力机制，从根本上降低了推荐系统推理时对候选集大小和用户序列长度的计算成本依赖。

    

    扩展大型推荐系统需要在三个主要方向上取得进展：处理更长的用户历史、扩大候选集规模以及增加模型容量。尽管前景可观，但Transformer的计算成本随用户序列长度呈二次方增长，并随候选数量呈线性增长。这种权衡使得在推理阶段扩大候选集或增加序列长度的代价极其高昂，尽管这能带来显著的性能提升。我们提出了LIME，一种解决这一权衡的新型架构。通过两项关键创新，LIME从根本上降低了计算复杂度。第一，低秩“链接嵌入”通过解耦用户与候选之间的交互，实现注意力权重的预计算，使推理成本几乎与候选集大小无关。第二，线性注意力机制LIME-XOR降低了相对于用户序列长度的计算复杂度。

    arXiv:2510.18239v4 Announce Type: replace-cross  Abstract: Scaling large recommendation systems requires advancing three major frontiers: processing longer user histories, expanding candidate sets, and increasing model capacity. While promising, transformers' computational cost scales quadratically with the user sequence length and linearly with the number of candidates. This trade-off makes it prohibitively expensive to expand candidate sets or increase sequence length at inference, despite the significant performance improvements. We introduce \textbf{LIME}, a novel architecture that resolves this trade-off. Through two key innovations, LIME fundamentally reduces computational complexity. First, low-rank ``link embeddings" enable pre-computation of attention weights by decoupling user and candidate interactions, making the inference cost nearly independent of candidate set size. Second, a linear attention mechanism, \textbf{LIME-XOR}, reduces the complexity with respect to user seque
    
[^449]: 面向时间尺度鲁棒的持续学习脉冲神经网络的局部时间尺度门控机制

    Local Timescale Gates for Timescale-Robust Continual Spiking Neural Networks

    [https://arxiv.org/abs/2510.12843](https://arxiv.org/abs/2510.12843)

    提出局部时间尺度门控（LT-Gate）神经元模型，通过双时间常数动力学与自适应门控机制，使单个神经元在响应快速信号的同时保留慢速上下文信息，并辅以方差追踪正则化稳定放电活动，显著提升了脉冲神经网络在持续学习任务中的准确率与记忆保持能力。

    

    脉冲神经网络（SNN）有望在神经形态硬件上实现高能效的人工智能，但在既需要快速适应又需要长期记忆的任务中表现不佳，尤其是在持续学习场景下。我们提出了局部时间尺度门控（LT-Gate），这是一种将双时间常数动力学与自适应门控机制相结合的神经元模型。每个脉冲神经元并行地在快、慢两个时间尺度上追踪信息，并由一个可学习的门控在局部调整两者的影响。这种设计使单个神经元能够在响应快速信号的同时保留慢速的上下文信息，从而解决稳定性-可塑性困境。我们进一步引入了一种基于方差追踪的正则化方法来稳定神经元的放电活动，其灵感来源于生物稳态机制。实验结果表明，LT-Gate 在序列学习任务中显著提升了准确率与记忆保持能力：在一个具有挑战性的时序分类基准上，它达到了约 51%

    arXiv:2510.12843v2 Announce Type: replace  Abstract: Spiking neural networks (SNNs) promise energy-efficient artificial intelligence on neuromorphic hardware but struggle with tasks requiring both fast adaptation and long-term memory, especially in continual learning. We propose Local Timescale Gating (LT-Gate), a neuron model that combines dual time-constant dynamics with an adaptive gating mechanism. Each spiking neuron tracks information on a fast and a slow timescale in parallel, and a learned gate locally adjusts their influence. This design enables individual neurons to preserve slow contextual information while responding to fast signals, addressing the stability-plasticity dilemma. We further introduce a variance-tracking regularization that stabilizes firing activity, inspired by biological homeostasis. Empirically, LT-Gate yields significantly improved accuracy and retention in sequential learning tasks: on a challenging temporal classification benchmark it achieves about 51 
    
[^450]: 面向小训练样本量半监督图学习的分数阶热核方法

    Fractional Heat Kernel for Semi-Supervised Graph Learning with Small Training Sample Size

    [https://arxiv.org/abs/2510.04440](https://arxiv.org/abs/2510.04440)

    该论文提出一种源驱动的分数阶热核半监督图学习框架，通过持续标签源防止解塌缩至拉普拉斯零空间从而缓解过平滑，在每类仅一个训练标签的极小样本条件下即可达到超过96%的准确率。

    

    我们开发了一个源驱动的分数阶热核框架，用于半监督图学习，该框架将非局部传播与持续的标签信息相结合。一个与拉普拉斯算子零空间兼容的固定非零标签源可防止解渐近塌缩到该空间，从而为在长扩散时间下缓解过平滑问题提供了机制。分数阶控制着相对模态衰减以及持续响应的谱加权，而扩散时间则设定传播范围。我们刻画了归一化图和非连通图上的守恒律与平衡态，开发了零空间缩减方法，并分析了传播子的逼近性能。在Two-Moon数据集上，分数阶改进了无源传播的效果，而兼容的源驱动扩散在阶数为0.8和1时，每类仅使用一个训练标签即达到超过96%的平均准确率。在Cora和CiteSeer数据集上，源驱动流水线进一步改进了……（摘要原文截断）

    arXiv:2510.04440v2 Announce Type: replace  Abstract: We develop a source-driven fractional heat-kernel framework for semi-super\-vised graph learning that combines nonlocal propagation with sustained label information. A fixed nonzero label source compatible with the Laplacian null space prevents asymptotic collapse into that space, providing a mechanism for mitigating oversmoothing at long diffusion times. The fractional order controls the relative modal attenuation and the spectral weighting of the sustained response, while the diffusion time sets the propagation horizon. We characterize conservation laws and equilibria on normalized and disconnected graphs, develop a null-space deflation, and analyze the approximation of the propagators. On Two-Moon, fractional orders improve source-free propagation, while compatible source-driven diffusion exceeds $96\%$ mean accuracy with one training label per class at orders $0.8$ and $1$. On Cora and CiteSeer, the source-driven pipeline improve
    
[^451]: CAF'E：机器遗忘的因果黑盒测试

    CAF\'E: Causal Black-Box Testing of Machine Unlearning

    [https://arxiv.org/abs/2509.16525](https://arxiv.org/abs/2509.16525)

    提出CAF'E框架，将机器遗忘测试构建为基于规范的测试，仅通过黑盒模型的预测输出对特征进行因果干预并传播到下游特征，从而有效检测特征影响在模型中的残留。

    

    机器学习模型越来越多地被部署为需要随需求变化而演进的软件组件。当特定的训练记录或特征不能再影响已部署的模型时，机器遗忘旨在不从头重新训练的情况下消除这种影响。由于遗忘通常是近似的，因此必须对其有效性进行测试。此类测试通常必须将模型视为黑盒，即无法访问其参数、训练历史或遗忘过程。特征带来了进一步的挑战：即使一个特征已从模型的输入中移除，其影响仍可能通过下游特征持续存在。许多现有的检查仅检验该特征的直接使用，因此可能会将一个实际上仍依赖该特征的模型误判为合格。我们将遗忘测试构建为基于规范的测试，并提出了CAF'E，它仅利用已部署模型的预测，对特征进行干预，并将变化传播到其下游特征。

    arXiv:2509.16525v2 Announce Type: replace-cross  Abstract: Machine learning models are increasingly deployed as software components that must evolve as requirements change. When specific training records or features must no longer influence a deployed model, machine unlearning aims to remove that influence without retraining from scratch. Because unlearning is often approximate, its effectiveness must be tested. Such tests must often treat the model as a black box, without access to its parameters, training history, or unlearning procedure. Features pose a further challenge: even after a feature is removed from a model's inputs, its influence can persist through downstream features. Many existing checks examine only the feature's direct use and can therefore certify a model that still depends on it. We frame unlearning testing as specification-based testing and present CAF\'E, which, using only a deployed model's predictions, intervenes on the feature, propagates the change to its down
    
[^452]: AdaSwitch：面向学习增强的有界影响问题的自适应切换元算法

    AdaSwitch: An Adaptive Switching Meta-Algorithm for Learning-Augmented Bounded-Influence Problems

    [https://arxiv.org/abs/2509.02302](https://arxiv.org/abs/2509.02302)

    该论文提出了有界影响框架与元算法AdaSwitch，它能在离线与在线预测器之间自适应切换，在预测准确时性能逼近离线最优、在预测任意不准时仍保持接近在线算法的最坏情况保证，并成功应用于在线交货期报价、k服务器、缓存和在线可重用资源分配等多种问题。

    

    我们研究具有历史依赖性的在线问题，其中对未来请求序列的预测可能不准确。受多个现实应用的启发，我们引入了一个“有界影响”框架，在该框架中，过去的决策和请求对未来最优值的影响仅限于一个有界的量。在该框架内，我们提出了AdaSwitch——一种可在合适的离线与在线预测器之间自适应切换的元算法。AdaSwitch对期望性能提供了显式保证，且随着预测误差的减小或离线最优值的增大，这些保证会变得更加紧致。在完美预测的情况下，随着离线最优值的增长，其保证趋近于离线预测器的保证。在任意预测条件下，它还能保持接近在线预测器的最坏情况保证。将该框架应用于在线交货期报价、k服务器与缓存问题以及在线可重用资源分配，展示了该框架广泛的适用性。

    arXiv:2509.02302v2 Announce Type: replace  Abstract: We study history-dependent online problems with a possibly inaccurate prediction of the future request sequence. Motivated by several real-world applications, we introduce a \emph{bounded-influence} framework in which past decisions and requests affect the future optimal value by only a bounded amount. Within this framework, we develop AdaSwitch, a meta-algorithm that adaptively switches between suitable offline and online oracles. AdaSwitch provides explicit guarantees on expected performance that tighten as prediction error decreases or the offline optimum increases. With perfect predictions, its guarantee approaches the offline oracle's guarantee as the offline optimum grows. It also retains a worst-case guarantee close to that of the online oracle under arbitrary predictions. Applications to online lead-time quotation, $k$-server and caching, and online reusable resource allocation demonstrate the framework's applicability to bot
    
[^453]: 成员推断与隐私审计的样本复杂度

    The Sample Complexity of Membership Inference and Privacy Auditing

    [https://arxiv.org/abs/2508.19458](https://arxiv.org/abs/2508.19458)

    本文在高斯均值估计的基础设定下，研究了成员推断攻击的样本复杂度，即确定了成功实施攻击和隐私审计所需的最少参考样本数量。

    

    成员推断攻击通过获取学习算法的输出和一个目标个体，试图判断该个体是训练数据的成员，还是来自同一分布的独立样本。成功的成员推断攻击通常要求攻击者对训练数据所采样的分布具有一定的了解，而这种知识通常通过来自该分布的一组独立参考样本来体现。在本工作中，我们通过研究样本复杂度——即成功攻击所需的最少参考样本数量——来探究攻击者在成员推断中需要多少信息。我们在高斯均值估计这一基础设定中研究该问题：学习算法获得来自d维高斯分布 $\mathcal{N}(\mu,\Sigma)$ 的 $n$ 个样本，并尝试在一定的误差范围内估计 $\hat\mu$。

    arXiv:2508.19458v2 Announce Type: replace  Abstract: A membership-inference attack gets the output of a learning algorithm, and a target individual, and tries to determine whether this individual is a member of the training data or an independent sample from the same distribution. A successful membership-inference attack typically requires the attacker to have some knowledge about the distribution that the training data was sampled from, and this knowledge is often captured through a set of independent reference samples from that distribution. In this work we study how much information the attacker needs for membership inference by investigating the sample complexity-the minimum number of reference samples required-for a successful attack. We study this question in the fundamental setting of Gaussian mean estimation where the learning algorithm is given $n$ samples from a Gaussian distribution $\mathcal{N}(\mu,\Sigma)$ in $d$ dimensions, and tries to estimate $\hat\mu$ up to some error
    
[^454]: Inverse-LLaVA：通过文本到视觉映射重新思考多模态对齐

    Inverse-LLaVA: Rethinking Multimodal Alignment via Text-to-Vision Mapping

    [https://arxiv.org/abs/2508.12466](https://arxiv.org/abs/2508.12466)

    该论文提出Inverse-LLaVA，通过在解码器注意力中将语言状态映射到视觉特征维度（而非传统的将图像特征投影到语言空间），在冻结主干、无需独立对齐阶段的单阶段训练下实现多模态融合，性能接近两阶段的LLaVA-1.5。

    

    连接预训练的视觉模型与语言模型通常需要将图像特征投影到语言模型的输入空间中。Inverse-LLaVA 在解码器注意力机制中反转了这一映射：语言状态被投影到视觉特征维度，并由模态特定的映射生成残差的查询、键和值更新。融合模块与低秩适应（LoRA）从665K条视觉指令数据中联合学习，主干网络保持冻结，且无需单独的对齐阶段。在九项主要基准评估中，最终的7B模型在若干任务上接近两阶段训练的LLaVA-1.5：其在VQAv2上得分为78.45%（官方LLaVA-LoRA为79.13%），在VizWiz上为50.96%（对手为48.56%），而在TextVQA上略低，为56.96%（对手为58.47%）。控制实验研究了融合组件、插入深度、视觉特征和语言模型规模的影响。表征分析表明，文本映射在改变表示的同时，较好地保留了成对相似性的排序结构。

    arXiv:2508.12466v3 Announce Type: replace-cross  Abstract: Connecting pretrained vision and language models usually involves projecting image features into the language model's input space. Inverse-LLaVA reverses this mapping within decoder attention: language states are projected to the visual feature dimension, and modality-specific maps produce residual query, key, and value updates. Fusion and low-rank adaptation (LoRA) learn jointly from 665K visual instructions, with frozen backbones and no separate alignment stage. Across nine primary benchmark evaluations, the final 7B model approaches two-stage LLaVA-1.5 on several tasks. It scores 78.45% on VQAv2 versus 79.13% for official LLaVA-LoRA, and 50.96% versus 48.56% on VizWiz; TextVQA is lower at 56.96% versus 58.47%. Controlled studies examine fusion components, insertion depth, visual features, and language-model size. Representation analysis shows that the text maps preserve much of the pairwise similarity ordering while changing
    
[^455]: 大语言模型数学推理鲁棒性研究：基于高等数学问题数学等价变换的基准测试

    An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems

    [https://arxiv.org/abs/2508.08833](https://arxiv.org/abs/2508.08833)

    提出 GAP 方法，通过表面重命名与核心重写两种数学等价变换自动批量生成现有数学题的等价变体，以评估大语言模型数学推理的鲁棒性并诊断其失败环节。

    

    arXiv:2508.08833v4 公告类型： replace-cross 摘要：前沿大语言模型（LLM）在标准数学推理基准上的准确率已接近满分，并在国际数学奥林匹克竞赛中达到金牌水平的表现。随着这些基准趋于饱和、其题目泄露进训练数据，高分已无法说明模型的推理是否鲁棒，也无法定位推理中哪个环节出现了失败。为了在保持结果信息量和失败可诊断性的前提下评估推理能力，我们提出了 GAP（Generalisation-and-Perturbation，泛化与扰动）方法，该方法利用两种互不相交、可解释的变换，自动大规模生成现有数学问题的数学等价变体：（1）表面重命名，用于探测标识符与潜在变量角色之间的绑定关系；（2）核心重写，用于检验高层次的证明方案在数学背景发生改变后是否依然成立。与现有基准相比，GAP 具有两个关键优势：（1）新颖、li……

    arXiv:2508.08833v4 Announce Type: replace-cross  Abstract: Frontier large language models (LLMs) now reach near-ceiling accuracy on standard mathematical-reasoning benchmarks and gold-medal-level performance at the International Mathematical Olympiad. As these benchmarks saturate and their items leak into training data, a high score no longer shows whether a model reasons robustly or which component of its reasoning fails. To evaluate reasoning while keeping results informative and failures diagnosable, we propose GAP (Generalisation-and-Perturbation), a methodology that automatically generates mathematically equivalent variants of existing mathematics problems at scale using two disjoint, interpretable transformations: (1) surface renames, probing the binding between identifiers and latent variable roles, and (2) kernel rewrites, probing whether a high-level proof plan survives a change of mathematical setting. Compared with existing benchmarks, GAP has two key benefits: (1) novel, li
    
[^456]: BrainATCL：用于功能连接预测与年龄估计的自适应时序脑连接学习

    BrainATCL: Adaptive Temporal Brain Connectivity Learning for Functional Link Prediction and Age Estimation

    [https://arxiv.org/abs/2508.07106](https://arxiv.org/abs/2508.07106)

    提出了BrainATCL，一个无监督、非参数化的自适应时序脑连接学习框架，能够捕捉动态fMRI数据中的长程时间依赖性，用于功能连接预测和年龄估计。

    

    功能性磁共振成像（fMRI）是一种被广泛用于研究人脑活动的成像技术。即使个体处于静息状态，大脑各区域的fMRI信号也会以高度结构化的方式发生瞬时的同步与去同步。这些功能连接动态可能与行为和神经精神疾病相关。为了对这些动态进行建模，时序脑连接表征至关重要，因为它反映了大脑区域之间不断演化的相互作用，并为理解瞬时神经状态和网络重构提供了洞察。然而，传统的图神经网络（GNN）往往难以捕捉动态fMRI数据中的长程时间依赖性。为了应对这一挑战，我们提出了BrainATCL，一个用于自适应时序脑连接学习的无监督、非参数化框架，可实现功能连接预测和年龄估计。我们的方法动态……

    arXiv:2508.07106v2 Announce Type: replace  Abstract: Functional Magnetic Resonance Imaging (fMRI) is an imaging technique widely used to study human brain activity. fMRI signals in areas across the brain transiently synchronise and desynchronise their activity in a highly structured manner, even when an individual is at rest. These functional connectivity dynamics may be related to behaviour and neuropsychiatric disease. To model these dynamics, temporal brain connectivity representations are essential, as they reflect evolving interactions between brain regions and provide insight into transient neural states and network reconfigurations. However, conventional graph neural networks (GNNs) often struggle to capture long-range temporal dependencies in dynamic fMRI data. To address this challenge, we propose BrainATCL, an unsupervised, nonparametric framework for adaptive temporal brain connectivity learning, enabling functional link prediction and age estimation. Our method dynamically 
    
[^457]: 脑基础模型引导的源选择性域自适应方法用于跨被试脑电解码

    Brain foundation model-guided source-selective domain adaptation for cross-subject EEG decoding

    [https://arxiv.org/abs/2507.21037](https://arxiv.org/abs/2507.21037)

    提出了一种由预训练脑基础模型引导的多源域自适应框架BFM-MSDA，利用脑基础模型表示估计源域与目标域的兼容性以实现源域选择性自适应，从而解决跨被试运动想象脑电解码中的负迁移与分布差异问题。

    

    跨被试运动想象脑电（MI-EEG）解码仍然具有挑战性，因为被试间显著的个体差异既可能导致来自匹配度较差的源被试的负迁移，也可能导致源域与目标域之间持续存在的分布差异。现有的多源域自适应方法通常纳入所有可用的源域，或利用信号层面或任务特定的表示来估计源域相关性，而分布对齐往往仅在特征层面进行。这些局限性可能引入无关的源域知识，且无法在跨被试场景中保持类别判别性结构。在本研究中，我们提出了一种脑基础模型引导的多源域自适应框架（BFM-MSDA），用于跨被试运动想象脑电解码。该方法首先利用预训练脑基础模型所学到的表示来估计源域与目标域之间的兼容性，并且……

    arXiv:2507.21037v3 Announce Type: replace  Abstract: Cross-subject motor-imagery electroencephalography (MI-EEG) decoding remains challenging because substantial inter-subject variability can cause both negative transfer from poorly matched source subjects and persistent distribution discrepancies between source and target domains. Existing multi-source domain adaptation methods often incorporate all available source domains or estimate source relevance using signal-level or task-specific representations, while distribution alignment is frequently performed only at the feature level. These limitations may introduce irrelevant source knowledge and fail to preserve class-discriminative structures across subjects. In this study, we propose a brain foundation model-guided multi-source domain adaptation framework (BFM-MSDA) for cross-subject MI-EEG decoding. The method first utilizes representations learned by a pretrained brain foundation model to estimate source--target compatibility and 
    
[^458]: 面向分布内数据获取的共形数据污染检验

    Conformal Data Contamination Tests for In-distribution Data Acquisition

    [https://arxiv.org/abs/2507.13835](https://arxiv.org/abs/2507.13835)

    本文提出了一种无分布假设的共形数据污染检验框架，仅需检查少量数据即可识别出对模型个性化最有价值的外部数据代理，从而在数据获取前提供质量保证。

    

    在许多机器学习任务中，高质量数据的数量受限于数据所有者本地可获取的数据。高质量数据集可以通过与外部数据代理进行交易或共享来扩展。然而，外部数据可能被污染，或引入不良的样本多样性，从而降低个性化机器学习任务的性能，例如罕见疾病诊断或推荐系统。因此，数据购买者在获取数据之前需要质量保证。先前的工作主要依赖于对不同数据代理的数据分布假设，将质量检查推迟到事后步骤，这涉及成本高昂的数据估值流程。我们提出了一种无分布假设、具备污染感知能力的数据获取框架，该框架仅需检查少量数据，即可识别出其数据对模型个性化最有价值的外部数据代理。为实现这一目标，我们引入了新颖的双样本

    arXiv:2507.13835v2 Announce Type: replace-cross  Abstract: The amount of quality data in many machine learning tasks is limited to what is available locally to data owners. The set of quality data can be expanded through trading or sharing with external data agents. However, external data may be contaminated or introduce undesirable sample diversity which can degrade performance of personalized machine learning tasks, as in diagnosis of a rare disease or recommendation systems. Therefore, data buyers need quality guarantees prior to data acquisition. Previous works primarily rely on distributional assumptions about data from different agents, relegating quality checks to post-hoc steps involving costly data valuation procedures. We propose a distribution-free, contamination-aware data acquisition framework that, by inspecting only a small volume of data, identifies external data agents whose data is most valuable for model personalization. To achieve this, we introduce novel two-sample
    
[^459]: 基于潜在空间桥接的异构模态无监督域自适应

    Heterogeneous-Modal Unsupervised Domain Adaptation via Latent Space Bridging

    [https://arxiv.org/abs/2506.15971](https://arxiv.org/abs/2506.15971)

    本文提出了一种新的HMUDA设置及潜在空间桥接（LSB）方法，利用包含两种模态成对观测数据的无标记桥接域，实现有标记源域与完全无标记的异构模态目标域（如2D图像与3D点云）之间的知识迁移。

    

    无监督域自适应（UDA）能够有效弥合有标记源域与无标记目标域之间的域差距，但其假设两个域共享相同的模态。异构域自适应（HDA）则处理跨域的不同特征空间，但需要有标记的目标样本或连接源域与目标域的成对数据。当有标记的源域与完全无标记的目标域各自拥有完全不同的模态（例如2D图像和3D点云）时，上述两种方法均不适用。为了解决这一局限性，我们提出了一种新的设置，称为异构模态无监督域自适应（HMUDA），该设置通过一个无标记的桥接域实现跨模态的知识迁移，该桥接域包含来自两种模态的成对观测数据，且其分布可能偏离源域和目标域的分布。为了在HMUDA设置下进行学习，我们提出了潜在空间桥接（LSB）方法，（摘要在此处截断）

    arXiv:2506.15971v2 Announce Type: replace-cross  Abstract: Unsupervised domain adaptation (UDA) effectively bridges the domain gap between a labeled source domain and an unlabeled target domain, but assumes that the two domains share the same modality. Heterogeneous domain adaptation (HDA) instead handles different feature spaces across domains, yet requires labeled target samples or paired data linking the source and target domains. Neither applies when a labeled source domain and a fully unlabeled target domain each hold an entirely distinct modality (e.g., 2D images and 3D point clouds). To address this limitation, we introduce a new setting termed Heterogeneous-Modal Unsupervised Domain Adaptation (HMUDA), which transfers knowledge across modalities via an unlabeled bridge domain containing paired observations from both modalities, whose distribution may deviate from those of the source and target domains. To learn under the HMUDA setting, we propose Latent Space Bridging (LSB), a 
    
[^460]: Delphos：一个用于辅助离散选择模型设定的强化学习框架

    Delphos: A reinforcement learning framework for assisting discrete choice model specification

    [https://arxiv.org/abs/2506.06410](https://arxiv.org/abs/2506.06410)

    Delphos是一个深度强化学习框架，它将离散选择模型设定形式化为序贯决策问题，通过智能体自动选择建模动作生成候选模型设定，为建模者提供自动化、数据驱动的建议，从而减少开发与改进效用函数所需的工作量。

    

    我们提出了Delphos，一个用于辅助离散选择模型设定过程的深度强化学习框架。Delphos旨在通过为模型设定提供自动化、数据驱动的建议来支持建模者，从而减少开发和改进效用函数所需的工作量。Delphos受人类选择建模者通过一系列有理据的设定决策迭代构建模型的方式启发，将模型设定概念化为一个序贯决策问题。在这一设定中，智能体通过选择一系列建模动作来学习制定候选模型设定，例如添加备择方案特定常数、同时容纳通用参数和备择方案特定品味参数、对属性应用非线性变换，以及纳入与协变量的交互项。每个由此产生的候选模型都通过定义的奖励函数进行估计和评估。

    arXiv:2506.06410v4 Announce Type: replace-cross  Abstract: We introduce Delphos, a deep reinforcement learning framework for assisting discrete choice model specification process. Delphos aims to support the modeller by providing automated, data-driven suggestions for model specifications, thereby reducing the effort required to develop and refine utility functions. Delphos conceptualises model specification as a sequential decision-making problem, inspired by the way human choice modellers iteratively construct models through a series of reasoned specification decisions. In this setting, an agent learns to specify candidate model specifications by choosing a sequence of modelling actions, such as adding alternative specific constants, accommodating both generic and alternative-specific taste parameters, applying non-linear transformations to attributes, and including interactions with covariates. Each resulting candidate model is estimated and evaluated using a reward function defined
    
[^461]: 迈向合理的概念瓶颈模型

    Towards Reasonable Concept Bottleneck Models

    [https://arxiv.org/abs/2506.05014](https://arxiv.org/abs/2506.05014)

    提出概念推理模型（CREAM），一种可在架构层面显式编码概念间关系与概念-任务关系、并能借助正则化旁路通道处理不完整概念集的概念瓶颈模型新框架，同时引入了与C→Y无关的可解释性评估指标。

    

    我们提出了一种新颖、灵活且高效的概念瓶颈模型设计框架，使从业者能够在模型进行预测时的推理过程中，显式地编码和扩展他们关于概念-概念（C-C）以及概念-任务（C→Y）关系的先验知识和信念。由此产生的概念推理模型在架构上编码了任意类型的C-C关系，例如互斥性、层次关联和/或相关性，以及可能稀疏的C→Y关系。此外，CREAM可以选择性地引入一个正则化的旁路通道来补充可能不完整的概念集，在取得有竞争力的任务性能的同时，促使预测基于概念进行。为了在此类设置下评估概念瓶颈模型，我们引入了一个与C→Y无关的度量指标，用以量化预测的可解释性。

    arXiv:2506.05014v3 Announce Type: replace-cross  Abstract: We propose a novel, flexible, and efficient framework for designing Concept Bottleneck Models (CBMs) that enables practitioners to explicitly encode and extend their prior knowledge and beliefs about the concept-concept ($C-C$) and concept-task ($C \to Y$) relationships within the model's reasoning when making predictions. The resulting $\textbf{C}$oncept $\textbf{REA}$soning $\textbf{M}$odels (CREAMs) architecturally encode arbitrary types of $C-C$ relationships such as mutual exclusivity, hierarchical associations, and/or correlations, as well as potentially sparse $C \to Y$ relationships. Moreover, CREAM can optionally incorporate a regularized side-channel to complement the potentially {incomplete concept sets}, achieving competitive task performance while encouraging predictions to be concept-grounded. To evaluate CBMs in such settings, we introduce a $C \to Y$ agnostic metric that quantifies interpretability when predicti
    
[^462]: 广义核下t分布随机邻域嵌入的平衡分布

    Equilibrium Distribution for t-Distributed Stochastic Neighbor Embedding with Generalized Kernels

    [https://arxiv.org/abs/2505.24311](https://arxiv.org/abs/2505.24311)

    该论文为广义输入输出核下的t-SNE大样本变分问题建立了严格的数学理论，证明了尺度参数的存在唯一性、解的存在性与一致有界性，以及离散最优解收敛到满足平衡方程的紧支撑平衡分布。

    

    我们研究了一类输入核和输出核下t分布随机邻域嵌入的大样本变分问题。输入律具有紧支撑，且其密度在该支撑上连续。一个熵方程确定了输入核中的尺度参数，我们证明该参数在正密度的内部点处存在且唯一。随后，我们给出了解在整个支撑上存在且一致有界的充分条件。在这些条件以及输出核的衰减假设下，离散最优值收敛于连续统最小值。近似极小元的经验测度在平移后是紧的；每个子序列极限都是满足平衡方程的紧支撑极小元。可容许的输出核包括高斯核，以及在输出维度为二时的柯西核。数值例子比较了二维表示（原文在此处截断）。

    arXiv:2505.24311v3 Announce Type: replace-cross  Abstract: We study the large-sample variational problem for t-distributed stochastic neighbor embedding with a class of input and output kernels. The input law has compact support and a density continuous on that support. An entropy equation determines the scale parameter in the input kernel, and we prove that this parameter exists and is unique at interior points of positive density. We then give sufficient conditions for solutions to exist and be uniformly bounded on the entire support. Under these conditions and a decay assumption on the output kernel, the discrete optimal values converge to a continuum minimum. Empirical measures of approximate minimizers are tight after translation; every subsequential limit is a compactly supported minimizer satisfying the equilibrium equation. The admissible output kernels include Gaussian kernels and, in output dimension two, the Cauchy kernel. Numerical examples compare the two-dimensional repre
    
[^463]: InfiFPO：基于偏好优化的大语言模型隐式模型融合

    InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models

    [https://arxiv.org/abs/2505.13878](https://arxiv.org/abs/2505.13878)

    InfiFPO通过在DPO中用序列层面综合多源概率的融合源模型替换参考模型，实现了无需复杂词表对齐且保留概率信息的大语言模型隐式融合偏好优化方法。

    

    模型融合旨在通过轻量级训练方法，将多个具有不同优势的大语言模型（LLM）整合为一个更强大的综合模型。现有的模型融合研究主要集中于监督微调（SFT），而对偏好对齐（PA）——提升LLM性能的关键阶段——的探索则相对匮乏。目前少数针对偏好对齐阶段的融合方法（如WRPO）仅利用源模型的响应输出而丢弃其概率信息，从而简化了该过程。为解决这一局限，我们提出了InfiFPO，一种面向隐式模型融合的偏好优化方法。InfiFPO用融合源模型替代直接偏好优化（DPO）中的参考模型，该融合源模型在序列层面综合多源概率，从而规避了以往工作中复杂的词表对齐难题，同时保留了概率信息。通过引入……

    arXiv:2505.13878v4 Announce Type: replace-cross  Abstract: Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introduci
    
[^464]: 原型分析综述

    A Survey on Archetypal Analysis

    [https://arxiv.org/abs/2504.12392](https://arxiv.org/abs/2504.12392)

    这是首篇关于原型分析（AA）的综述，系统介绍了其方法论、面临的非凸优化挑战、跨科学领域的广泛应用以及数据建模的最佳实践。

    

    原型分析（Archetypal Analysis, AA）最初由Adele Cutler和Leo Breiman于1994年提出，作为一种从观测数据中提取不同方面（即所谓的“原型”）的计算方法，每个观测记录被近似为这些原型的混合（即凸组合）。由此，AA为特征提取和降维提供了直接、可解释且可说明的表示方法，有助于理解高维数据的结构，并在各科学领域得到广泛应用。然而，AA也面临挑战，特别是其相关的优化问题是非凸的。这是首篇为研究人员和数据挖掘从业者提供AA所提供的方法论与机遇概览的综述，调研了AA在各科学领域的众多应用，以及使用AA进行数据建模的最佳实践。

    arXiv:2504.12392v3 Announce Type: replace-cross  Abstract: Archetypal analysis (AA) was originally proposed in 1994 by Adele Cutler and Leo Breiman as a computational procedure for extracting distinct aspects, so-called archetypes, from observations, with each observational record approximated as a mixture (i.e., convex combination) of these archetypes. AA thereby provides straightforward, interpretable, and explainable representations for feature extraction and dimensionality reduction, facilitating the understanding of the structure of high-dimensional data and enabling wide applications across the sciences. However, AA also faces challenges, particularly as the associated optimization problem is nonconvex. This is the first survey that provides researchers and data mining practitioners with an overview of the methodologies and opportunities that AA offers, surveying the many applications of AA across disparate fields of science, as well as best practices for modeling data with AA an
    
[^465]: C-LoRA：面向预训练视觉模型的持续低秩适应

    C-LoRA: Continual Low-Rank Adaptation for Pre-trained Visual Models

    [https://arxiv.org/abs/2502.17920](https://arxiv.org/abs/2502.17920)

    C-LoRA通过可学习路由矩阵使单个共享的LoRA适配器能够持续学习序列任务而不会发生灾难性遗忘，无需推理时的模块选择或融合，同时避免了参数无界增长和推理复杂度增加。

    

    预训练视觉模型已成为计算机视觉领域的基础，但在数据和任务随时间演变的持续学习场景中，它们面临着挑战。低秩适应提供了高效的微调能力，但在此类动态环境中仍然受限。标准LoRA无法区分重要的子空间，导致关键知识在序列训练过程中被覆盖。现有方法通过动态扩展LoRA适配器集合来解决这一问题，要么维护一个不断增长的任务特定模块池，要么将新适配器合并到先前的适配器中，但代价是无界的参数增长或不断增加的推理复杂度。我们提出了持续低秩适应，这种方法使单个共享的LoRA适配器能够处理序列任务而不会发生灾难性遗忘，且在推理时无需任何模块选择或融合。C-LoRA的核心是一个可学习的路由矩阵R

    arXiv:2502.17920v2 Announce Type: replace  Abstract: Pre-trained visual models have become fundamental in computer vision, but they face challenges in continual learning scenarios where data and tasks evolve over time. Low-Rank Adaptation (LoRA) offers efficient fine-tuning capabilities but remains limited for such dynamic environments. Standard LoRA cannot distinguish important subspaces, causing critical knowledge to be overwritten in sequential training. Existing approaches address this by dynamically expanding the set of LoRA adapters, either maintaining a growing pool of task-specific modules or merging new adapters into prior ones, at the cost of unbounded parameter growth or increasing inference complexity. We propose Continual Low-Rank Adaptation (C-LoRA), a method that enables a single, shared LoRA adapter to handle sequential tasks without catastrophic forgetting, without requiring any module selection or fusion at inference. The core of C-LoRA is a learnable routing matrix R
    
[^466]: 具有有限VC维的网络：利与弊

    Networks with Finite VC Dimension: Pro and Contra

    [https://arxiv.org/abs/2502.02679](https://arxiv.org/abs/2502.02679)

    该论文证明有限的VC维虽有利于经验误差的一致收敛，却可能不利于函数逼近，并基于高维几何的测度集中性质证明，此类网络在处理大规模数据集时逼近误差与经验误差均几乎呈确定性行为。

    

    本文从高维几何与统计学习理论的角度，研究了利用神经网络对大规模数据集的分类器进行逼近与学习的问题。文章比较了网络输入-输出函数集合的VC维对逼近能力的影响，与其对基于数据样本学习的一致性的影响。结果表明，尽管有限的VC维对于经验误差的一致收敛是有利的，但对于逼近从某个概率分布（该分布建模了函数在特定应用中出现的可能性）中抽取的函数而言，却可能并不理想。基于高维几何的测度集中性质，论文证明：对于实现具有有限VC维的输入-输出函数集合的网络，在处理大规模数据集时，其逼近误差和经验误差均表现出近乎确定性的行为。

    arXiv:2502.02679v3 Announce Type: replace-cross  Abstract: Approximation and learning of classifiers of large data sets by neural networks in terms of high-dimensional geometry and statistical learning theory are investigated. The influence of the VC dimension of sets of input-output functions of networks on approximation capabilities is compared with its influence on consistency in learning from samples of data. It is shown that, whereas finite VC dimension is desirable for uniform convergence of empirical errors, it may not be desirable for approximation of functions drawn from a probability distribution modeling the likelihood that they occur in a given type of application. Based on the concentration-of-measure properties of high dimensional geometry, it is proven that both errors in approximation and empirical errors behave almost deterministically for networks implementing sets of input-output functions with finite VC dimensions in processing large data sets. Practical limitations
    
[^467]: BEAT：面向长期时间序列预测的平衡频率自适应调优

    BEAT: Balanced Frequency Adaptive Tuning for Long-Term Time-Series Forecasting

    [https://arxiv.org/abs/2501.19065](https://arxiv.org/abs/2501.19065)

    提出BEAT框架，通过频率专属监测器在统一归一化空间中监测各频率分量的系数预测误差，并据此自适应地调节各频率网络的梯度，从而平衡长期时间序列预测中不同频率分量的训练侧重。

    

    长期时间序列预测支持广泛的应用，包括天气预报和电力需求规划。频域方法通过将观测数据分解为描述不同时间尺度变化的分量来完成这一任务。然而，仅靠分离的表示本身并不能提供一种显式机制来调整各分量之间的训练侧重。在共享预测目标下，各个频率专属网络可能保留不同程度的系数预测误差，这就需要根据误差情况对其梯度进行调整。为此，我们提出了BEAT（Balanced frEquency Adaptive Tuning，平衡频率自适应调优），一个将频率专属误差监测与自适应梯度调节相结合的框架。我们设计了频率专属监测器，在统一的归一化空间中比较预测与目标小波系数，并将每个差异表示为相对于……（原文摘要在此处截断）

    arXiv:2501.19065v3 Announce Type: replace-cross  Abstract: Long-term time-series forecasting supports a wide range of applications, including weather prediction and electricity demand planning. Frequency-domain methods address this task by decomposing observations into components that describe temporal variations at different scales. However, separate representations do not by themselves provide an explicit mechanism for adjusting the training emphasis across components. Under a shared forecasting objective, the frequency-specific networks can retain different levels of coefficient prediction error, motivating an error-dependent adjustment to their gradients. To this end, we propose BEAT (Balanced frEquency Adaptive Tuning), a framework that combines frequency-specific error monitoring with adaptive gradient modulation. We design a Frequency-Specific Monitor that compares predicted and target wavelet coefficients in a common normalized space and expresses each discrepancy relative to a
    
[^468]: 大语言模型基础

    Foundations of Large Language Models

    [https://arxiv.org/abs/2501.09223](https://arxiv.org/abs/2501.09223)

    本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。

    

    这是一本关于大语言模型的书籍。正如书名所示，本书主要聚焦于基础性概念，而非全面涵盖所有前沿技术。全书由六个主要章节构成，每个章节探讨一个关键领域：预训练、生成模型、提示（Prompting）、对齐、推断（Inference）和推理（Reasoning）。本书面向大学生、自然语言处理及相关领域的专业人士和从业者，也可作为所有对大语言模型感兴趣的读者的参考书。

    arXiv:2501.09223v3 Announce Type: replace-cross  Abstract: This is a book about large language models. As indicated by the title, it primarily focuses on foundational concepts rather than comprehensive coverage of all cutting-edge technologies. The book is structured into six main chapters, each exploring a key area: pre-training, generative models, prompting, alignment, inference, and reasoning. It is intended for college students, professionals, and practitioners in natural language processing and related fields, and can serve as a reference for anyone interested in large language models.
    
[^469]: 线图的图子（Graphons）

    Graphons of Line Graphs

    [https://arxiv.org/abs/2409.01656](https://arxiv.org/abs/2409.01656)

    本文提出一种通过将稀疏图映射到其线图并利用“平方度性质”使稀疏图产生稠密线图的方法，从而可以应用稠密图极限理论来分析稀疏图，并实证证明能够区分原本都收敛到零图子的不同数量的星形图。

    

    我们考虑从稀疏有限图序列的观测中估计图极限（称为图子，graphons）的问题。在本文中，我们展示了一种简单的方法，可以揭示一类稀疏图的性质。该方法将原始图映射到其线图。我们证明，满足一种特殊性质（我们称之为平方度性质，square-degree property）的图是稀疏的，但会产生稠密的线图。这使得我们能够利用稠密图图极限的已有结果来推导收敛性。特别地，星形图满足平方度性质，从而产生稠密的线图以及线图的非零图子。我们通过实证演示，可以利用相应线图的图子来区分不同数量的星（这些星本身是稀疏的）。而在原始图中，由于稀疏性，不同数量的星都会收敛到零图子。类似地，超线性优先连接……

    arXiv:2409.01656v4 Announce Type: replace-cross  Abstract: We consider the problem of estimating graph limits, known as graphons, from observations of sequences of sparse finite graphs. In this paper we show a simple method that can shed light on a subset of sparse graphs. The method involves mapping the original graphs to their line graphs. We show that graphs satisfying a particular property, which we call the square-degree property are sparse, but give rise to dense line graphs. This enables the use of results on graph limits of dense graphs to derive convergence. In particular, star graphs satisfy the square-degree property resulting in dense line graphs and non-zero graphons of line graphs. We demonstrate empirically that we can distinguish different numbers of stars (which are sparse) by the graphons of their corresponding line graphs. Whereas in the original graphs, the different number of stars all converge to the zero graphon due to sparsity. Similarly, superlinear preferentia
    
[^470]: 基于概念推断与数据投毒的复杂数据类别机器遗忘

    Class Machine Unlearning for Complex Data via Concepts Inference and Data Poisoning

    [https://arxiv.org/abs/2405.15662](https://arxiv.org/abs/2405.15662)

    该论文针对复杂数据的类别机器遗忘问题，提出通过推断连接遗忘目标与模型输出的语义概念，并结合数据投毒技术来精准消除目标信息的影响，从而避免知识残留并保护应保留的内容。

    

    机器遗忘旨在无需完整重新训练的情况下，从已训练模型中移除特定训练数据或知识的影响。这一能力对现代图像分类器和大型语言模型（LLM）尤为重要，因为对这类模型进行重新训练的计算成本可能非常高昂。然而，在复杂数据上进行机器遗忘仍然十分困难，因为目标信息往往分布在多个语义元素之中。现有方法主要通过删除样本、修改标签或编辑模型参数来降低遗忘目标的影响，但通常不会明确识别是哪些语义概念将遗忘目标与模型的预测或生成的响应联系起来，因此难以确定应当修改哪些信息。这种不确定性可能导致目标信息残留，或者不必要地影响本应保留的知识。为了填补这一空白，我们提出……（摘要在此处截断）

    arXiv:2405.15662v2 Announce Type: replace  Abstract: Machine unlearning aims to remove the influence of specified training data or knowledge from a trained model without requiring full retraining. This capability is particularly important for modern image classifiers and large language models (LLMs), where retraining can be computationally expensive. However, machine unlearning on complex data remains difficult because the target information is often distributed across multiple semantic elements. Existing methods mainly remove samples, modify labels, or edit model parameters to reduce the influence of the forgetting target. These approaches usually do not explicitly identify which semantic concepts connect the forgetting target to the model's prediction or generated response. As a result, it is difficult to determine which information to modify. This uncertainty may leave residual target information or unnecessarily affect knowledge that should be retained. To address this gap, we prop
    
[^471]: 基于语言瓶颈的策略学习

    Policy Learning with a Language Bottleneck

    [https://arxiv.org/abs/2405.04118](https://arxiv.org/abs/2405.04118)

    该论文提出PLLB框架，让AI智能体在语言模型引导的“规则生成”与规则引导的“策略更新”之间交替进行，通过语言瓶颈捕捉行为背后的高层策略，从而学习到更可解释、更可泛化的行为。

    

    现代人工智能系统，例如自动驾驶汽车和游戏智能体，能够达到超越人类的性能水平，但往往缺乏人类式的泛化能力、可解释性以及与人类用户的互操作性。受人类语言与决策之间丰富互动的启发，我们提出了带语言瓶颈的策略学习，这是一个使AI智能体能够生成语言规则、从而捕捉有益行为背后高层策略的框架。PLLB在由语言模型引导的“规则生成”步骤与由规则引导智能体学习新策略的“更新”步骤之间交替进行，即使某条规则不足以描述整个复杂策略也能有效运作。在五个多样化的任务上，包括双人信号博弈、迷宫导航、图像重建和机器人抓取规划，我们展示了PLLB智能体不仅能够学习到更可解释、更可泛化的行为，还可以……

    arXiv:2405.04118v4 Announce Type: replace-cross  Abstract: Modern AI systems such as self-driving cars and game-playing agents can achieve superhuman performance, but often lack human-like generalization, interpretability, and inter-operability with human users. Inspired by the rich interactions between language and decision-making in humans, we introduce Policy Learning with a Language Bottleneck (PLLB), a framework enabling AI agents to generate linguistic rules that capture the high-level strategies underlying rewarding behaviors. PLLB alternates between a *rule generation* step guided by language models, and an *update* step where agents learn new policies guided by rules, even when a rule is insufficient to describe an entire complex policy. Across five diverse tasks, including a two-player signaling game, maze navigation, image reconstruction, and robot grasp planning, we show that PLLB agents are not only able to learn more interpretable and generalizable behaviors, but can also
    

