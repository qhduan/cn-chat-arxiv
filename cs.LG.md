# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance](https://arxiv.org/abs/2610.20607) | 提出HiRe-MoMa分层残差框架，通过对冻结基础策略进行移动底盘与机械臂解耦的残差修正，并结合稀疏人类引导的专家价值对齐，实现真实世界移动操作系统的稳定且样本高效的微调。 |
| [^2] | [Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion](https://arxiv.org/abs/2610.20606) | 该论文提出一种将线性自编码器与扩散Transformer相结合的潜在扩散框架，可直接在权重空间中对强化学习得到的人形机器人控制网络进行生成式建模，从而生成可部署的高性能控制网络。 |
| [^3] | [Co-Evolving Acting and Forecasting for Language Agents](https://arxiv.org/abs/2610.20597) | 该论文提出在成功的同策略轨迹上引入多步动作序列作为监督目标，使行动生成与动作预测协同演化，从而充分利用中间步骤的监督信号，缓解语言智能体训练中长程 rollout 成本高、终端奖励稀疏的问题。 |
| [^4] | [Your Transformer Is Secretly Elastic](https://arxiv.org/abs/2610.20588) | 提出一种无需重新训练的可微功能单元排序方法，将预训练Transformer转化为单一弹性模型，可跨连续计算预算实时自适应调整，并在22M至8B参数的27个模型上验证了其有效性。 |
| [^5] | [Price of Anarchy of First-Price Auctions in Agentic Markets](https://arxiv.org/abs/2610.20587) | 本文研究了代理市场中一价拍卖的无政府代价，指出代理市场本身所具备的特性有望降低一价拍卖带来的福利损失。 |
| [^6] | [Continuum Attention](https://arxiv.org/abs/2610.20586) | 本文提出“连续注意力”算子，用对连续分段线性记忆曲线的积分取代离散键值查找，通过闭式指数矩实现精确归一化注意力，在语言建模上以负对数似然显著优于softmax和焦点注意力。 |
| [^7] | [Learnable Hodge-Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes](https://arxiv.org/abs/2610.20585) | 该论文提出一种由可学习霍奇-拉普拉斯算子驱动的混合专家架构，将解按频谱尺度分配给不同transformer专家并通过逐顶点门控权重重组，在粗非结构网格稳态PDE代理任务上相比Transolver将相对$L_2$误差降低了31%。 |
| [^8] | [Latent Compaction with Test-Time Training](https://arxiv.org/abs/2610.20582) | 提出“潜在压缩”方法，通过测试时训练在模型权重冻结且不接触下游问题的情况下，学习紧凑的文档级 KV 缓存，显著优于同等预算的缓存压缩基线。 |
| [^9] | [One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness](https://arxiv.org/abs/2610.20576) | 提出跨界面策略正则化（CIPR）这一自监督方法，通过在两个界面下惩罚动作分布间的KL散度，使LLM智能体在面对界面扰动（如动作重命名、记号变化）时保持一致行为，显著提升其跨界面的鲁棒性。 |
| [^10] | [Recovering Reliable Supervision via Critique-Guided On-Policy Distillation](https://arxiv.org/abs/2610.20572) | 提出CriticOPD方法，通过教师模型对学生的错误推理轨迹进行批评点评并引导学生重写解答，恢复了在线策略蒸馏中教师监督的可靠性，在数学推理尤其是竞赛题目上显著优于传统OPD。 |
| [^11] | [Detection of Potato Plant Leaf Diseases using Convolutional Neural Networks](https://arxiv.org/abs/2610.20544) | 本研究开发了一种仅含446,787个参数（约1.70 MB）的轻量级CNN模型，在PlantVillage数据集上达到98.66%的测试准确率，并在埃塞俄比亚独立田间数据集上达到90.28%的准确率，可用于检测马铃薯早疫病和晚疫病。 |
| [^12] | [PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis](https://arxiv.org/abs/2610.20538) | 该论文提出PrismaCoder，从数据端出发，通过分解式评分标准合成方法，解决多模态代码生成中样本的程序正确性、代码质量与视觉正确性难以同时评估的核心难题，从而推动开放多模态代码智能的发展。 |
| [^13] | [Revisiting Simple Baselines for Relational Deep Learning](https://arxiv.org/abs/2610.20534) | 本文提出了一种简单但有效的关系深度学习基线方法，通过将低通特征传播应用于关系实体图并训练MLP，再结合Correct-and-Smooth标签传播算法优化预测结果，证明无需复杂的图神经网络架构即可实现强大的预测性能。 |
| [^14] | [Generative Representation Learning for Scalable Latent Tokenization](https://arxiv.org/abs/2610.20533) | 提出了一种基于逐点互信息分解的生成式自监督表示学习方法，能够避免维度坍缩和特征抑制、随算力增加可预期地提升分词质量，并在分词器与生成模型的综合计算效率上首次实现了优于直接扩散的计算最优改进。 |
| [^15] | [ActiveEvidence: Decision-Focused Evidence Graphs with Verifiable Acquisition Gaps](https://arxiv.org/abs/2610.20532) | 该论文提出ActiveEvidence框架，通过反事实证据交互图（CEIG）和可验证的ε-决策证书，使多模态智能体能够高效决定何时获取视觉证据、仅评估可能改变决策的关键交互，并在证明进一步交互无法改变所选动作时及时停止。 |
| [^16] | [The Physics Is the Network: Fast, Training-Free Neural Emulation](https://arxiv.org/abs/2610.20525) | PolyNet将多项式动力学的神经表示与简单循环网络结合，在已知物理系数的情况下无需训练即可构建张量图，实现高效硬件执行和大规模并行仿真，并在KdV方程求解中达到与数值求解器相当的精度。 |
| [^17] | [VASPO: Self-Distillation with Privileged Value Estimation](https://arxiv.org/abs/2610.20516) | 该论文提出VASPO方法，将特权价值估计与自蒸馏在PPO框架内结合，利用评论家与自教师在学生生成前缀上评估特权信息时呈现的互补位置动态特性，为推理模型提供更有效的词元级训练信号。 |
| [^18] | [Bayesian Learning to Select Fair Classifiers](https://arxiv.org/abs/2610.20514) | 本文提出一种基于汤普森采样的贝叶斯多臂老虎机在线方法，在真实标签延迟可得的条件下，自适应选择同时兼顾预测准确性与敏感属性子群体公平性的分类器（涵盖表格模型、大语言模型与人工标注者），并相比基线方法显著降低了遗憾值。 |
| [^19] | [A Two-Level Clinical Novelty Prioritization Framework for Disease Trajectories Discovered from Longitudinal Electronic Health Records](https://arxiv.org/abs/2610.20512) | 该论文提出了一个两级框架，将疾病轨迹分支模体的语义稀疏性与经可靠性调整的临床优先级分离，从而对从纵向电子健康记录中发现的 99,527 个模体进行新颖性评估和临床优先级排序。 |
| [^20] | [VascuSR: Iterative Anatomy-Aware Diffusion for CT Super-Resolution](https://arxiv.org/abs/2610.20509) | VascuSR提出了一种将图像重建与显式3D血管引导相结合的迭代式解剖感知扩散框架，有效解决了CT超分辨率重建中血管细节缺失、虚假或错误连接的问题。 |
| [^21] | [Graph-Enhanced Disease Trajectory Discovery and Temporal Endpoint Forecasting from Longitudinal Electronic Health Records](https://arxiv.org/abs/2610.20502) | 该研究提出了一种在MIMIC-IV上实现泄漏控制的图增强疾病轨迹模式发现框架，结合时间感知Transformer进行终点预测，使事件时间预测误差平均降低19.2天。 |
| [^22] | [Hindsight-Privileged RLVR: Single-Rollout Token Credit Without Group Sampling](https://arxiv.org/abs/2610.20497) | 提出HP-RLVR，一种无需组采样的单次轨迹RLVR方法，通过比较因果视角与包含未来结果和验证器反馈的事后特权视角之间的Jensen–Shannon散度为每个词元分配有界信用，从而在所有尝试结果相同时仍能提供有效学习信号。 |
| [^23] | [Average-Reward Policy Identification from a Single Continuing Trajectory](https://arxiv.org/abs/2610.20486) | 该论文针对平均奖励强化学习提出了一种自适应决定在何处收集信息的算法，并通过对称的上下界证明：对于一大类环境，在足够高的精度下，从单条持续轨迹中学习在最坏情况下并不比可自由采样的学习更困难。 |
| [^24] | [Decoupled Marginal Sharpening for Training-Free Inference-Time Scaling](https://arxiv.org/abs/2610.20483) | 该论文提出解耦边际锐化（DMS），在目标层面和采样器层面分别进行锐化，以克服全序列锐化抑制多样推理路径以及有限粒子SMC路径坍缩的问题，从而实现无需训练和奖励模型的LLM推理时性能提升。 |
| [^25] | [Lost in Translation: Why Lower Bias Scores Don't Mean Fairer Models](https://arxiv.org/abs/2610.20479) | 本研究审计了10个开源模型在8种语言中的偏见表现，发现非英语语言下刻板印象分数的下降主要源于翻译缺陷与测量噪声，而非模型本身更公平，因为测试项的英语分数对其翻译后分数的预测能力远低于对改写后分数的预测能力。 |
| [^26] | [SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift](https://arxiv.org/abs/2610.20473) | 该论文提出 SafeAdapt-Poker，将对手漂移下的预算化适应建模为仅依赖决策前信息的双重反事实序贯控制问题，通过单调策略风险模型与配对预算状态延续模型分别校准偏离的局部下行风险和预算消耗的未来代价，并用逐路径累计KL账本约束总偏离量，从而在即时适应价值与安全风险之间实现可校准的权衡。 |
| [^27] | [Selective Growing Q-Networks (SGQN): An Analysis of Adaptive Action Grids in Decoupled Q-Learning](https://arxiv.org/abs/2610.20463) | 本文提出选择性生长Q网络（SGQN），在解耦Q学习中用自适应动作网格（在高Q值区域附近增加bin、剪除低Q值bin并采用epsilon-贪婪动作选择）取代均匀分辨率增长，在多个DMControl连续控制任务上取得优异表现，但消融分析发现自适应网格本身并不能解释这些性能优势。 |
| [^28] | [Learning by Examples: Power Grid Steady State Analysis with a Tabular Foundation Model](https://arxiv.org/abs/2610.20462) | 本研究证明预训练表格基础模型TabPFN无需重新训练或微调，即可作为电网稳态分析（潮流预测、N-k预想故障筛查和电压稳定裕度预测）的通用神经求解器，在多个任务上达到或超越任务特定模型。 |
| [^29] | [The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation](https://arxiv.org/abs/2610.20456) | 提出COUNT方法，通过在廉价的主机内存上存储预计算的n-gram计数并结合语言模型自身的unembedding来构建n-gram嵌入，无需训练新参数，从而实现主机内存上的扩展与免训练适配。 |
| [^30] | [What Does Disagreement Mean? A Semantic Identifiability Frontier for Pairwise Preference Learning](https://arxiv.org/abs/2610.20450) | 该论文提出包含方向偏好、持续分歧和非二元质量三个量的非参数潜在语义模型，证明在匿名二值反馈通道下分歧的语义含义本质上是不可辨识的，但通过同标注者重复标注与允许弃权的反馈相结合可以完整点辨识这三个语义量。 |
| [^31] | [The Accuracy Well: Nonmonotone Accuracy Dynamics at Vanishing Initialization](https://arxiv.org/abs/2610.20439) | 该论文发现小初始化训练神经网络时精度呈现非单调的“精度之井”现象——先快速上升、随后下降、很久后才恢复，并为浅层和深层线性网络建立了严格的三阶段理论解释。 |
| [^32] | [Continual Learning for Reliable In-Sequence Molecular Communication](https://arxiv.org/abs/2610.20426) | 本文提出仿射记忆回放（AMR）方法，通过存储历史仿真样本并利用插值与外推合成回放样本，在实验设置复杂度逐步递增的序列内分子通信中实现通信性能的增量估计，在适应新设置的同时保持整体估计精度。 |
| [^33] | [OrionMoE-v1.0: A Mixture-of-Experts Foundation Model for Tabular In-Context Learning](https://arxiv.org/abs/2610.20418) | 提出了OrionMoE-Small，一个将分布感知表示与数据集条件稀疏混合专家计算相结合的紧凑型表格基础模型，通过双轴Transformer沿特征轴自适应激活专家，实现异构表格任务的高效上下文学习。 |
| [^34] | [Tracking a Small Set of Experts in Online Convex Optimization](https://arxiv.org/abs/2610.20416) | 本文证明了在线凸优化中“追踪一小组专家”问题存在本质的计算复杂性分离：信息论上可达的最优遗憾度 $\widetilde{O}(md+k)$ 在超图植入团猜想等复杂性假设下无法被任何多项式时间算法实现，高效算法只能达到 $\widetilde{O}(dk)$ 量级的遗憾度。 |
| [^35] | [One Size Does Not Fit All: Curvature-Aware Adaptive Perturbation for SAM](https://arxiv.org/abs/2610.20415) | 该论文提出CAP-SAM方法，通过基于锐度比率自适应地调节SAM的扰动半径，克服了固定扰动半径忽略局部曲率的缺陷，从而在损失景观上更有效地搜索鲁棒极小值。 |
| [^36] | [When Does Momentum Help in Reinforcement Learning with Verifiable Rewards?](https://arxiv.org/abs/2610.20413) | 本文通过带偏差修正EMA动量的老虎机形式化分析RLVR，理论上证明精确梯度下增大动量系数β可严格加速优化，而采样估计梯度下需增大β以维持有效步长η/(1-β)来克服采样波动。 |
| [^37] | [Hyperspherical Contrastive Representations Unlock Planning and Exploration](https://arxiv.org/abs/2610.20409) | 该论文提出C-Slerp方法，利用超球面参数化的对比表示，通过球面线性插值以闭式解直接推断中间路径点，实现了无需奖励、演示或预训练模型的长时程规划与探索，成功完成最多八块立方体的搭建等复杂任务。 |
| [^38] | [Posterior Frequencies and the Cost of Not Fitting: A Correction to a Memorization Lower Bound](https://arxiv.org/abs/2610.20402) | 本文通过精确的三点反例证明 Feldman 记忆化下界中的频率系数不成立且无法用普遍正常数因子修复，并基于全样本后验概率给出了一个精确的替代系数。 |
| [^39] | [Training Language Models to Fail Gracefully](https://arxiv.org/abs/2610.20401) | 提出GracefulRL方法，通过重塑优化格局，使语言模型仅凭结果级奖励信号就能学会在行动不可靠时执行安全的回退策略、实现“优雅失败”，无需人工标注或推理时额外机制。 |
| [^40] | [HRTF Upsampling and Personalization via Neural Operators](https://arxiv.org/abs/2610.20400) | SPARO是一种将稀疏空间测量映射为连续HRTF场的神经算子，在SONICOM数据集的所有稀疏度水平上均实现了最优的LSD和ILD误差，显著优于现有最强基线。 |
| [^41] | [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399) | 本文提出 Cortex，通过生成前瞻性查询将上下文增益蒸馏为可合并的低秩或结构化前馈网络参数更新，在 4B 至 120B 的多种稠密与混合专家模型上实现了“巩固—保持”帕累托最优的记忆巩固。 |
| [^42] | [From Human Corrections to Learned Segmentations: A Critical Mathematical Review of Information, Learning, and Human Effort](https://arxiv.org/abs/2610.20396) | 该综述提出统一的数学框架，借助精确贝叶斯更新、超额风险分解与条件信息分解，系统比较主动学习、机器教学和交互式分割等方法，阐明人类纠错何时转化为有效学习，以及在同等人类付出下如何衡量与实现更好的分割效果。 |
| [^43] | [FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding](https://arxiv.org/abs/2610.20392) | FOLD是一种免训练的长上下文投机解码方法，它保留精确的局部位置，将所有远端条目作为共享位置上的远端记忆处理，并通过锐化记忆内部的注意力来缓解注意力稀释问题。 |
| [^44] | [Hypothesis Selection via Weighted Medians](https://arxiv.org/abs/2610.20375) | 本文提出了一种基于加权中位数的假设选择算法，在达到最优近似因子和样本复杂度的同时，将运行时间对失败概率的依赖从多项式级改进为对数级。 |
| [^45] | [ReAl: Refine neural operators via aligning dynamical flow](https://arxiv.org/abs/2610.20369) | ReAl 是一个模型无关的即插即用方法，通过条件流匹配模型精炼预训练 PDE 代理模型的细尺度结构，并通过对齐动力学流来抑制自回归滚动预测中的生成误差累积，从而在保持长期稳定性的同时改善高频结构的重建。 |
| [^46] | [Fractional is Better: Learnable Derivative Orders in Neural Operator Learning](https://arxiv.org/abs/2610.20363) | 该论文提出在神经算子中引入与偏微分方程结构对齐的可学习导数特征，从理论上证明其可将逼近界从 $O(W^{-(s-m)/d})$ 改进为 $O(W^{-s/d})$，并出人意料地发现最优导数阶数是严格低于偏微分方程阶数 $m$ 的分数阶 $\beta^*$。 |
| [^47] | [Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization](https://arxiv.org/abs/2610.20353) | 提出NatScale框架，通过模型特定的幂变换将基于梯度的专家权重校准到信息论对比度目标，并联合评估门控与上下投影配置的输出误差，从而在存储预算下实现更优的混合精度MoE量化位宽分配。 |
| [^48] | [Lifted State Policy Optimization: Internalizing Self-Correction through Transition Geometry](https://arxiv.org/abs/2610.20350) | LSPO将以能量函数刻画、经成本调整的“有缺陷状态向更优状态的局部转移”作为学习基本单元，通过奖励塑形和内部化损失将自我纠错内化进单次生成策略，在保持与测试时反思相当的准确率的同时显著降低延迟和token开销。 |
| [^49] | [NeuroDETR: Multiscale Discrete Spatiotemporal Tokenization for Asynchronous EEG Event Detection](https://arxiv.org/abs/2610.20349) | NeuroDETR提出了一种融合多尺度时间码本、空间金字塔VQ和事件感知稀疏注意力模块FOCAL的离散时空多尺度框架，能够在无需预定义事件边界的情况下，从连续EEG记录中联合检测事件类别及其起止时间。 |
| [^50] | [Non-asymptotic Convergence of Average-reward Q-learning with Options](https://arxiv.org/abs/2610.20347) | 该论文首次在均方意义下为inter-option和intra-option两种设置的平均奖励选项Q学习算法建立了 $\tilde{O}(1/\text{迭代次数})$ 的非渐近收敛速率保证。 |
| [^51] | [When Does Task-Aware Imputation Matter? An Analysis of Robustness Under Train-Test Mismatch](https://arxiv.org/abs/2610.20342) | 该论文提出将缺失数据插补与下游任务统一建模的框架，并在98个表格数据集上证明重构误差无法有效代表预测效用，任务感知的插补方法在训练-测试缺失机制不匹配时更具鲁棒性。 |
| [^52] | [Training Dynamics of Transformers on Unbounded Inputs: Attention Learning Can Lag Output Layer Learning](https://arxiv.org/abs/2610.20337) | 该论文证明了在无界输入上训练浅层Transformer识别无星语言时，注意力学习会滞后于输出层学习，需要采用“先更新输出层、再更新注意力、最后微调输出层”的三阶段梯度下降策略才能成功训练。 |
| [^53] | [Are We Overthinking One-Step Generation?](https://arxiv.org/abs/2610.20331) | 本文提出Priming Models两阶段框架：先用简单的重建目标（如固定噪声水平的像素去噪）预训练，再进行特征空间分布后训练，并发现这一简单的重建阶段即可完全解决单步生成器分布训练的困难。 |
| [^54] | [Comparative Evaluation of Deep-Learning based DoA Estimators for Real-Time ISAC Beamforming](https://arxiv.org/abs/2610.20326) | 本文提出一个统一评估框架，将多种经典与深度学习DoA估计器嵌入相同的基于SVD的ISAC波束成形流水线中，在相同条件下系统比较其估计精度与计算延迟，以评估其实时部署的适用性。 |
| [^55] | [Efficient Test-Time Adaptation For Robot Policies](https://arxiv.org/abs/2610.20318) | 该论文提出TEMPO方法，通过熵正则化与基于分数的记忆机制，在无需奖励信号的条件下实现机器人运动策略在分布偏移环境下稳定的在线测试时自适应。 |
| [^56] | [Extreme-Scale Super-Resolution of Homogeneous Turbulence from a Single Realization](https://arxiv.org/abs/2610.20305) | 该研究提出一种极端超分辨率框架，仅利用单个高分辨率湍流实现训练WDSR网络，便能在32倍和64倍的严重欠分辨率下准确重建均匀湍流的相干涡结构与细尺度特征。 |
| [^57] | [SpectrumQuant: Spectrum Alignment for LLM Quantization](https://arxiv.org/abs/2610.20303) | 提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。 |
| [^58] | [MMLA-RSI: Recursive Self-Improvement through RTT and RTMA](https://arxiv.org/abs/2610.20294) | 提出了一种递归自我改进方案MMLA-RSI，其中RTT改变策略状态、RTMA改变权威记忆，且这两种改进方法本身也可作为改进对象，由独立接纳的后继者在下一代中接管相同角色，从而实现智能体行为的持续递归改进。 |
| [^59] | [CASI: Credit Assignment via Set Intervention for Agentic Retrieval](https://arxiv.org/abs/2610.20293) | 该论文提出CASI方法，通过基于快照的集合干预构建反事实信息并结合Shapley聚合，解决了智能体检索中多个联合生成的检索动作的集合内信用分配问题，从而提升策略优化效果。 |
| [^60] | [Predicting Unseen Scientific Interactions: Conditional Flow Models Hallucinate Non-Additive Effects](https://arxiv.org/abs/2610.20288) | 本研究揭示条件流匹配模型学习个体效应远早于学习交互作用，并会在不存在交互作用时幻觉出非加性效应，导致模型整体看似准确却无法恢复真正的科学交互效应。 |
| [^61] | [From Plato to Aristotle: Toward a General Theory of Representation Hypotheses](https://arxiv.org/abs/2610.20286) | 该论文提出格表征假设（LRH），将各种表征对齐准则统一到同一个格结构中，从而解释了柏拉图表征假设的收敛声明与其实际证据之间的差异，并提出亚里士多德表征假设将收敛限定于局部邻域。 |
| [^62] | [TANGO: RNA Topology and Geometry Co-Design](https://arxiv.org/abs/2610.20277) | 提出TANGO框架，首次将RNA逆设计表述为拓扑与几何协同设计问题，通过多目标多智能体强化学习和分阶段课程训练（二级结构拓扑感知策略学习、三级结构迁移训练、前沿引导微调），实现同时满足二级碱基配对拓扑与三级几何结构的RNA序列设计。 |
| [^63] | [Resolution Allocation with Learnable Resizing and Spectral Filter Parameterization](https://arxiv.org/abs/2610.20274) | 该论文首次系统研究了可学习下采样与谱域滤波器参数化之间的相互作用，发现谱域卷积核比空间域卷积核从可学习分辨率分配中获得更大的准确率增益，且这一优势在匹配总保留面积后依然存在。 |
| [^64] | [FlowSep-RF: Rectified Flow Matching for Single-Channel Radio-Frequency Signal Separation](https://arxiv.org/abs/2610.20269) | 提出FlowSep-RF整流流匹配框架，通过融合时域IQ采样与频域特征、仅需一次端点速度评估即可完成单信道射频信号分离，在多种干扰场景下取得最低误码率和均方误差。 |
| [^65] | [FUSE: Fairness-aware Update Screening against Byzantine Exploitation in $q$-Fair Federated Learning](https://arxiv.org/abs/2610.20264) | 该论文提出了FUSE方法，首次在服务器仅能观察公平性缩放更新的“仅更新接口”下形式化了拜占庭鲁棒的q-公平联邦学习问题，通过公平性感知的更新筛选机制，使服务器能够区分合法的公平性放大与恶意的拜占庭攻击。 |
| [^66] | [Harness Policy Distillation: Aligning Agent Policies with Runtime Semantics](https://arxiv.org/abs/2610.20261) | 提出框架策略蒸馏（HPD）：通过在运行时框架中重放参考轨迹，将普通轨迹数据中隐含的运行时控制语义恢复为模型调用边界处的控制文档并作为显式训练目标，再借助框架验证保证目标可执行，从而实现智能体策略与运行时语义的对齐。 |
| [^67] | [Long-Horizon RL Needs Muon](https://arxiv.org/abs/2610.20256) | 提出了一种结合 Muon 优化器、令牌平均损失、截断重要性采样及长度与工具错误惩罚的单回合异步训练方案，显著提升了长时程强化学习在仓库级问题解决任务上的训练成功率。 |
| [^68] | [Adversarial Reinforcement Learning for Robust Planetary Landing](https://arxiv.org/abs/2610.20252) | 该论文提出一种三阶段对抗性强化学习训练框架，由辅助智能体生成具有时空相关性且满足物理约束的最坏情况扰动，与着陆控制器形成非对称序贯零和博弈并交替优化，从而显著提升行星着陆策略应对非标称条件的鲁棒性。 |
| [^69] | [Contrastive uncertainty learning for robust multi-structure segmentation in lower-pelvic MR](https://arxiv.org/abs/2610.20251) | 该论文提出对比不确定性学习框架，利用蒙特卡洛Dropout产生的体素级不确定性直接指导特征表示学习，通过排斥高不确定性特征防止边界处特征塌缩，从而提升前列腺MRI八结构分割在模糊区域的鲁棒性。 |
| [^70] | [Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation](https://arxiv.org/abs/2610.20247) | 本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。 |
| [^71] | [Retrieval Over Training: Similarity search-based Model Selection for Time Series Anomaly Detection](https://arxiv.org/abs/2610.20243) | 提出RAMSAD框架，通过相似性检索历史实例知识库而非训练模型的方式为新时间序列选择最优异常检测器，从而解决了现有模型选择方法在分布外（OOD）场景下精度大幅下降的问题。 |
| [^72] | [MAEM: Mitigating Recency Bias via a Replay-Free, Capacity-Controlled Dynamic Mixture of Experts for Class-Incremental Graph Learning](https://arxiv.org/abs/2610.20238) | 提出MAEM框架，通过为每个新会话训练并冻结专用专家、以无重放且容量受控的动态专家混合方式进行路由，从而在无需任务标识的情况下缓解类增量图学习中的近期偏差与灾难性遗忘。 |
| [^73] | [The minimal width of neural networks and immersions of real projective space](https://arxiv.org/abs/2610.20229) | 本文证明了当实射影空间 $\mathbb{RP}^{n-1}$ 可光滑浸入 $\mathbb{R}^n$ 时，神经网络逼近从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 连续函数的最小宽度至少为 $n+\min\{n,m\}$，从而在 $n=2,3,4,7,8$ 时确定了 ReLU 网络的最小宽度为 $n+m$，揭示了神经网络逼近理论与微分拓扑之间的深刻联系。 |
| [^74] | [Mixed Poisson diffusion models and application to posterior sampling](https://arxiv.org/abs/2610.20218) | 该论文提出了一种由多元计数过程驱动的混合泊松扩散模型（MPDM），为观测数据离散且服从泊松分布的泊松逆问题提供了后验采样的新框架。 |
| [^75] | [Continual pretraining Prostate MRI ViT Encoder for task alignment](https://arxiv.org/abs/2610.20217) | 该论文提出利用LoRA在前列腺分割任务上对预训练的ViT编码器进行持续预训练，以增强预训练与下游任务之间的对齐，并通过CKA逐层表征分析评估表征变化。 |
| [^76] | [HGRP: Heterogeneous Graph Neural Network for Routability Prediction](https://arxiv.org/abs/2610.20209) | 提出异构图神经网络HGRP，将单元、线网和布线网格建模为不同类型的节点并同时保留拓扑连接、空间映射与网格邻接等多层次电路语义，从而在无需布线的情况下实现高效准确的可布线性预测。 |
| [^77] | [Investigating Behavior Cloning from Few Demonstrations for Autonomous Driving based on Bird's-Eye View in Simulated Cities](https://arxiv.org/abs/2610.20203) | 该论文提出在行为克隆中引入核密度估计器，根据动作密度对训练样本加权，从而在专家演示稀缺的情况下，基于鸟瞰图输入显著提升自动驾驶对红灯停车等罕见但关键动作的学习能力。 |
| [^78] | [Dimension-Optimal Dynamic Regret in Exp-Concave Online Learning](https://arxiv.org/abs/2610.20187) | 本文解决了指数凹在线学习动态遗憾中悬而未决的维度最优性问题，提出一种自适应未知路径长度的全信息学习器并给出匹配下界，将遗憾界的维度因子从 d^{2/3} 改进到尖锐最优的 d^{1/3}。 |
| [^79] | [Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs](https://arxiv.org/abs/2610.20179) | 本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。 |
| [^80] | [RM-OPD: On-Policy Distillation for Generative Reward Models](https://arxiv.org/abs/2610.20174) | 该论文提出RM-OPD两阶段框架，先通过单领域GRPO微调训练各领域专家教师模型，再通过路由同策略蒸馏将其整合为单一学生模型，从而解决Mix-RL训练生成式奖励模型时的奖励稀疏与跨领域干扰问题。 |
| [^81] | [Optimizing Sparse Boolean Rule Networks using Mixed Integer Linear Programming](https://arxiv.org/abs/2610.20170) | 该论文提出用混合整数线性规划训练稀疏的布尔规则网络（DNF网络），通过增强的公式大幅减少整数决策变量并消除冗余约束，在保持精确布尔语义的同时实现误分类成本与稀疏性之间的全局最优权衡。 |
| [^82] | [Gap Detection: Mitigating Homogenization through Demand and Defensibility](https://arxiv.org/abs/2610.20162) | 提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。 |
| [^83] | [In-Context Batch Black-box Optimization via Flow Matching](https://arxiv.org/abs/2610.20138) | 提出FMTR方法，利用流匹配离线学习历史条件化的连续提议分布，实现可变大小批量的高效采样，并在优化过程中自适应平衡探索与利用，改进批量黑盒优化。 |
| [^84] | [Seeing at the Edge: Energy Characterization of Vision-Language Model Inference on Edge Platforms](https://arxiv.org/abs/2610.20136) | 该论文首次系统测量了视觉语言模型在树莓派5和Jetson等边缘平台上的能耗特性，发现NPU卸载可将每token能耗降低约三分之二、避免热节流，并揭示图像输入会同时带来固定能耗和随输出长度增长的额外能耗。 |
| [^85] | [A Bregman Proximal Viewpoint on Neural Operators](https://arxiv.org/abs/2610.20120) | 本文将神经算子层的作用解释为Banach函数空间上Bregman正则化优化问题的最小值点，由此提出了一种新的Bregman神经算子，其实验表明其傅里叶变体在训练更深、更精确的模型上更具优势。 |
| [^86] | [KANLab: TensorFlow Playground for Kolmogorov-Arnold Networks](https://arxiv.org/abs/2610.20117) | KANLab是一个从TensorFlow Playground分叉而来的开源浏览器可视化工具，通过交互式展示B样条激活曲线、剪枝和梯度下降等现象，帮助用户直观学习Kolmogorov-Arnold网络的核心概念。 |
| [^87] | [From Spectral to Frobenius: When Muon Loses Its Spectral Implicit Bias](https://arxiv.org/abs/2610.20112) | 本文证明Muon优化器的实际实现（有限次Newton-Schulz迭代和正稳定化常数）会使其失去理论上的谱隐式偏差，转而像梯度下降一样选择Frobenius最大间隔方向。 |
| [^88] | [Exhaustive Canonical Skeletons Generation: A Formal Dataset for Decomposable Symbolic Regression](https://arxiv.org/abs/2610.20108) | 该论文提出基于确定性受限上下文无关文法穷举生成的单变量规范骨架形式化数据集，消除了符号回归数据集中的结构冗余，并支持在不假设线性可分性的前提下求解多变量符号回归问题。 |
| [^89] | [What Do Foundation Models for Market Microstructure Learn? A Controlled Study with Synthetic Limit Order Books](https://arxiv.org/abs/2610.20104) | 本文通过在潜在参数已知的合成限价订单簿模拟器上训练GPT风格基础模型，首次证明订单流上的下一词元预训练实质上实现了对市场机制的摊销贝叶斯推断，其预测性能可精确逼近贝叶斯最优解。 |
| [^90] | [A Large-Dimensional Analysis of Least-Squares Federated Learning via Generalized Total Variation Minimization](https://arxiv.org/abs/2610.20075) | 本文通过随机矩阵理论对基于广义全变差最小化的最小二乘联邦学习进行大维度分析，给出了独立同分布与非独立同分布高斯混合数据下渐近精确性能的精确解析表达式，量化了正则化如何在局部适应与协作之间进行权衡。 |
| [^91] | [Rethinking Prediction Horizons: Decoupling Model Queries from State Updates in Few-Step MeanFlow](https://arxiv.org/abs/2610.20071) | 该论文在少步MeanFlow采样中解耦了模型查询区间与状态更新时长，发现预测视野对生成质量的影响并非仅由查询时间决定，还依赖于进入查询的状态分布——中点视野收益随早期引导设置的变化可从0.109显著增大至1.308 FID。 |
| [^92] | [Trainable at Depth, Solvable at Small Size: What the Loss Surface of a Polynomial Activation Dictates for Designing, Initializing and Training Networks](https://arxiv.org/abs/2610.20069) | 该论文提出多项式激活函数 f(z) = z + λz³ 加恒等路径的构造，通过精确求解小规模模型的损失面来指导深层网络的设计、初始化与训练，使百层网络无需任何归一化机制即可在 MNIST 上成功训练。 |
| [^93] | [GBTSVR: Twin Support Vector Regression via Granular Ball Computing](https://arxiv.org/abs/2610.20053) | 提出 GBTSVR 及其分位数扩展 GB-TSVQR，以粒球取代单个样本作为基本学习单元，显著提升了孪生支持向量回归在噪声、异常值和数据异质性场景下的鲁棒性与泛化性能。 |
| [^94] | [Sequential Learning-Based Physics-Informed Neural Network Modeling for Parameter Estimation on a Power Converter](https://arxiv.org/abs/2610.20041) | 该论文提出一种基于敏感性分析、按序释放梯度的分阶段PINN方法，可从时域数据中高精度识别降压变换器的电感、电容和寄生电阻参数，精度比同时训练基线提升一个数量级。 |
| [^95] | [Weighted Adjoint Matching: Finetuning Flow Matching Models with Time-Dependent Control Cost](https://arxiv.org/abs/2610.20035) | 提出加权伴随匹配方法，通过在控制代价上引入时间依赖权重，早期保持粗略语义特征的多样性与平衡、后期精细化细节以提升奖励，从而在微调流匹配模型时实现奖励与模式平衡之间更好的权衡。 |
| [^96] | [On the Convergence of Semi-Gradient TD: A Mean Field Perspective](https://arxiv.org/abs/2610.20014) | 该论文在平均场机制下分析了两层神经网络中半梯度时序差分学习的收敛性，证明了当熵诱导的松弛主导移动自举目标的敏感性时，算法可指数收敛至唯一不动点，且该稳定性机制可扩展至基于软Q学习的熵正则化离策略控制。 |
| [^97] | [ChemArgus: Automated Scientific Diagnostics for Lab-Grade Multimodal Chemistry Reasoning: A Frontier Benchmark](https://arxiv.org/abs/2610.20012) | ChemArgus是一个竞赛级多模态化学推理前沿基准，通过细粒度分步评分（含部分得分）和真实图像配受控文本描述的题目设计，能够诊断模型推理链在何处断裂，而不仅评估最终答案。 |
| [^98] | [Outperforming 1000 Euclidean Layers With Hyperbolic Goal-Conditioned RL](https://arxiv.org/abs/2610.20009) | 通过将对比强化学习中的欧几里得距离替换为双曲空间的测地距离，仅64层深的双曲智能体就能以超过12倍的训练速度和约13倍的参数量优势，超越1024层欧几里得网络的导航性能。 |
| [^99] | [Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds](https://arxiv.org/abs/2610.20007) | 本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。 |
| [^100] | [Provable Transformer Context Compression for In-Context Reinforcement Learning](https://arxiv.org/abs/2610.20004) | 该论文从理论上首次刻画了Transformer在上下文强化学习中进行TD评估所需的最小上下文维度（即由采样转移诱导的Krylov维度），并证明当上下文维度低于该阈值时，最优近似误差以 Θ(γ^m·max{1,1/(m(1-γ))}) 的速率衰减，从而回答了此类学习本质上需要多少上下文这一可扩展性问题。 |
| [^101] | [Empirical Boundaries of Reservoir Computing for Few-Shot Sequence Pattern Tasks](https://arxiv.org/abs/2610.19997) | 本文通过极简的少样本序列到序列模式任务，对回声状态网络（ESN）在分布、架构与优化层面的多种变体进行了实证评估，以刻画储备池计算在记忆与模式识别能力上的优势边界与局限。 |
| [^102] | [LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation](https://arxiv.org/abs/2610.19975) | 该论文发现循环MoE模型的两次循环访问存在非对称性，并提出LatentLoop方法，通过压缩第一次循环访问的序列、保留第二次的完整分辨率，并将节省的计算重新分配以增大隐藏层维度，从而在匹配预算下提升模型性能。 |
| [^103] | [Safe-FCL: Continual Risk Awareness and Fine-Grained Safety Aggregation for Federated Continual Learning of LLMs](https://arxiv.org/abs/2610.19968) | 提出Safe-FCL框架，通过持续风险感知（CRA）机制跨任务学习新风险并保留历史恶意模式知识，结合细粒度安全聚合（FGSA）机制，解决大语言模型联邦持续学习中恶意更新模式不断演变导致防御失效的问题。 |
| [^104] | [Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study](https://arxiv.org/abs/2610.19966) | 本文提出一种无需标签的启发式方法，通过嵌套Dropout VAE估计词元表示的内在维度来预先确定transformer注意力头的数量，实验表明按内在维度设置约10个注意力头的模型与32个头性能统计等效，且注意力参数减少69%。 |
| [^105] | [AtomSearch: Evidence-Grounded Tree Search for Agent Skill Optimization](https://arxiv.org/abs/2610.19951) | AtomSearch 通过对比式原子编辑和技能状态树搜索，以证据为依据实现局部化、可回溯的智能体技能优化，在多个任务和模型上超越了 SkillOpt。 |
| [^106] | [Learning Graph Energies for Finite-Time Generation](https://arxiv.org/abs/2610.19946) | 该论文提出“有限时域能量学习”方法，为固定步数后停止的 Metropolis-Hastings 图生成采样器学习接受能量，利用仅训练阶段使用的判别器和接受/拒绝分支的耦合延续获得无偏的接受梯度估计，从而在不依赖平衡负样本或离散状态微分的情况下实现高质量图生成。 |
| [^107] | [MorphingNet: Symbolic Regression Where the Network Is the Formula](https://arxiv.org/abs/2610.19942) | MorphingNet将每个函数的形式本身作为可训练参数，把符号回归转化为连续优化问题，无需算子库即可在训练全程保持公式可读，并在SRBench上达到最先进精度。 |
| [^108] | [SAM3RL: Memory Control via Reinforcement Learning for Visual Object Tracking](https://arxiv.org/abs/2610.19931) | SAM3RL 将 SAM 3 跟踪器的记忆控制形式化为序贯决策任务，通过强化学习训练轻量级记忆控制器来决定哪些帧进入记忆库，从而在 12 个基准中的 10 个上创造了新的最先进纪录。 |
| [^109] | [Convergence Rates of Gaussian Process Thompson Sampling for Efficient Global Optimization](https://arxiv.org/abs/2610.19925) | 本文首次在无噪声RKHS设定下为高斯过程汤普森采样建立了最坏情况简单遗憾上界 $O(N^{-\nu/d}\sqrt{\log N})$ 及匹配的下界，证明对数惩罚项是该算法的固有代价。 |
| [^110] | [MolHop: Scaffold-Hopping Molecular Design with Agentic Reinforcement Learning](https://arxiv.org/abs/2610.19920) | MolHop 提出了一个基于智能体强化学习的跨骨架分子发现框架，将约 160 万个商业分子骨架组织成从粗到细的分层动作空间，使模型能够发现骨架完全不同但保留参考分子功能特性的新分子。 |
| [^111] | [Beyond DFT electronic band structures via dual-space orbital learning](https://arxiv.org/abs/2610.19919) | O-kRystal框架利用低成本的PBE计算提供的实空间与倒易空间轨道信息，以约0.171 eV的平均绝对误差预测高精度HSE06能带结构，大幅降低了高保真电子结构计算的成本。 |
| [^112] | [CARD: Closing the Teacher-Student Loop for Adversarial Robustness Distillation](https://arxiv.org/abs/2610.19911) | 提出闭环循环对抗鲁棒性蒸馏框架CARD，将蒸馏后的学生模型作为反馈来源改进教师模型，再将教师增益回传给学生，从而突破传统单向蒸馏范式并提升对抗鲁棒性。 |
| [^113] | [When Non-Adversarial Noise Hurts Learning: Benchmarking Skill Self-Evolution in Agents](https://arxiv.org/abs/2610.19910) | 该论文提出了首个评估大语言模型智能体技能自我进化在非对抗性噪声下鲁棒性的基准 RSE-Bench，通过跨四个领域、六个基准的受控扰动以及与方法无关的 RSEClaw 噪声注入框架，系统分离学习阶段与评估阶段的噪声，揭示经验噪声如何影响智能体技能学习。 |
| [^114] | [Multimodal Medical Image Parsing with Task-Specific Low-Rank Adapters](https://arxiv.org/abs/2610.19901) | 该论文提出在冻结的 4-bit 量化 Qwen3.5-9B 主干上，根据输入问题动态选择并定制 LoRA 适配器的秩与提示词，同时结合稀有类别重平衡与数据补充策略，使单一视觉-语言模型能够高效应对覆盖六类任务族和七种成像模态的医学图像解析挑战。 |
| [^115] | [Universal Sample-Based Embeddings of Higher-Order Objects](https://arxiv.org/abs/2610.19899) | 提出了一个统一的基于样本的嵌入框架，能将分布、函数、图等任意高阶对象编码为固定大小的向量，实现跨对象类型的通用紧凑表示，并可用于训练中未见过的对象。 |
| [^116] | [Few-Shot Adaptation for Anomalous Sound Detection at Low False-Positive Rate](https://arxiv.org/abs/2610.19893) | 该论文提出一种两阶段异常声音检测方法，在仅有正常样本时部署正则化无监督模型，并在首次故障发生后利用小样本适应策略学习异常样本，将AUPR从23.0提升至64.0、pAUROC从39.6提升至72.8，显著改善了低误报率下的检测性能。 |
| [^117] | [On the Limits of Representation-Based Test-Time Adaptation for Multimodal LLMs](https://arxiv.org/abs/2610.19892) | 该论文证明了基于表示的测试时自适应无法恢复多模态大语言模型在损坏数据下的性能，原因在于其视觉token序列偏移方向几乎正交且遍布整个嵌入空间，这与CLIP类模型存在共享域偏移方向的情况截然不同。 |
| [^118] | [Occupancy Modeling via Drifting: One-Step Generation Enables Direct Policy Extraction](https://arxiv.org/abs/2610.19883) | 该论文提出使用现代单步生成采样器“漂移”来建模占用度，从而通过直接微分即可从生成模型中提取策略，显著简化了从无标注数据进行强化学习的流程。 |
| [^119] | [Neural Wasserstein Splines: Representation-Invariant Second-Order Priors for Trajectory Inference](https://arxiv.org/abs/2610.19878) | 提出神经瓦瑟斯坦样条（NWSP），通过瓦瑟斯坦平滑先验和内在瓦瑟斯坦加速度的有限维替代量，实现表示不变的二阶正则化，从而从不成对的快照中学习连续的种群演化曲线。 |
| [^120] | [SPARS: A Stability-Plasticity-Aware Region Selection Framework for Continual Learning](https://arxiv.org/abs/2610.19876) | SPARS是一种无需样本回放的持续学习混合框架，通过结合全局参数隔离与基于FIM或置信度分数估计器的局部参数正则化来选择关键参数区域，在保持旧知识稳定性的同时兼顾新任务的可塑性，从而缓解灾难性遗忘。 |
| [^121] | [Explorer-Zero: Improving Self-Play for Long-Horizon Tasks with Decomposable Verifiable Rewards](https://arxiv.org/abs/2610.19868) | Explorer-Zero 提出了一种自博弈框架，通过与环境的交互同时自动生成任务和细粒度的可验证奖励，利用中间与最终状态检查点奖励部分完成，从而无需人工标注任务即可提升 LLM 智能体在长程任务上的强化学习效果。 |
| [^122] | [On the Optimisation Landscape of Overparameterised Weighted MMD](https://arxiv.org/abs/2610.19861) | 该论文首次为加权MMD优化提供了有限粒子的过参数化保证，证明对于一维高斯核，在粒子数量足够多时所有局部极小值都具有任意小的MMD，且所有虚假驻点均不稳定。 |
| [^123] | [Text-Informed Training Data Filtering for Enhanced Virtual Flow Meter Performance](https://arxiv.org/abs/2610.19860) | 本文提出一种利用大语言模型将非结构化文本信息（如工程师评论和操作日志）与表格化传感器数据相结合的数据过滤方法，通过上下文异常检测去除低质量数据，从而提升虚拟流量计的预测性能。 |
| [^124] | [Energy Consumption Analysis of Discrete Diffusion and Autoregressive Language Models](https://arxiv.org/abs/2610.19855) | 本文通过对比基准测试发现，在较长序列（200–600词元）推理时，扩散语言模型的每词元能耗比使用KV缓存的自回归模型效率高约2倍，展现出显著的能效优势。 |
| [^125] | [Memory-Efficient Subspace Optimization with Residual Correction: From Momentum SGD to Muon](https://arxiv.org/abs/2610.19844) | 本文提出带残差校正的内存高效子空间优化方法SMRC及其多阶段变体，通过在子空间中维护优化器状态并将正交残差梯度加入参数更新，实现了与全空间动量SGD相匹配的理论收敛速率，并将该框架进一步扩展到Muon优化器。 |
| [^126] | [Rent-Seeking Language Models: Reward Models Set the Prices, Frequent Audits Curb the Rent](https://arxiv.org/abs/2610.19832) | 该论文将RLHF中的奖励过度优化创造性地重新表述为经济学中的“寻租”问题，通过租金、俘获指数和无谓损失等可度量指标实证揭示：策略模型会把41%-66%的代理奖励增益转化为真实质量不支持的租金（如编造不存在实体的细节或回避事实问题），租金随监管者定价而变，而频繁的黄金奖励审计可遏制这种寻租行为。 |
| [^127] | [CTIA-ODE: Continuous-Time Inter-Variable Attention for Long-Term Time Series Forecasting](https://arxiv.org/abs/2610.19831) | 该论文提出CTIA-ODE连续时间预测框架，将连续时间变量间注意力机制嵌入常微分方程求解器中，使内生变量轨迹能够在无需时间对齐的情况下沿积分路径动态查询静态外生变量信息，并通过双重稳定策略提升连续潜在动力学的稳定性。 |
| [^128] | [MAPLE: A Multi-Axis Parametric Knowledge Extraction Lens for LLM Evaluation](https://arxiv.org/abs/2610.19827) | 该论文提出MAPLE统一评估框架，通过在表层、概率和超越采样三个层面测量答案恢复与干扰、正确词元排名变化及采样遗漏答案的恢复情况，全面解耦知识提取增强（KEE）技术对大语言模型参数化知识提取的多样化影响。 |
| [^129] | [JointServe: Serving Multi-Model Agents on Memory-Constrained GPUs](https://arxiv.org/abs/2610.19822) | JointServe通过将权重分片与KV块统一纳入同一内存预算进行管理，并利用意图信号预取和部分驻留等智能体工作负载的结构特性在模型被调用前完成准备，从而在内存受限GPU上实现了多模型LLM智能体的高效服务。 |
| [^130] | [When Does Depth Help? Understanding Layerwise Inference in Tabular Foundation Models](https://arxiv.org/abs/2610.19814) | 该研究通过对TabPFN-3全部24个上下文学习层在272个分类数据集上的系统分析，揭示了表格基础模型存在显著的“扩张-压缩”表示轨迹，表明深度对不同输入的帮助程度并不相同，并非所有输入都需要更深的层来完成任务。 |
| [^131] | [VaryST: Generalizable Spatio-Temporal Forecasting Beyond the Closed-World Setting](https://arxiv.org/abs/2610.19813) | 提出了从零训练的紧凑时空预测模型 VaryST，通过从可用的长期观测中动态构建预测上下文，实现了在时间分布演变、空间域变化和观测可用性变化等开放环境下的强泛化预测能力。 |
| [^132] | [Calibration and Risk-Budgeted Decisions with System One Models](https://arxiv.org/abs/2610.19811) | 该论文提出在LLM工作流的二元决策场景中，通过不确定时采取保守回退动作，并利用Learn Then Test（LTT）的有限样本统计保证来控制假遗漏风险，显著优于朴素阈值选择基线方法。 |
| [^133] | [ReproBAIT: Can coding agents build their own biological AI tools?](https://arxiv.org/abs/2610.19810) | 该论文提出了ReproBAIT评估基准，通过在受限信息、无互联网访问的沙箱环境中，衡量编码智能体设计、实现并训练复现已发表生物AI工具的能力，并依据复现工具在留出数据上的性能进行评分。 |
| [^134] | [Efficient contextual partial monitoring with general function approximation](https://arxiv.org/abs/2610.19804) | 本文提出了计算高效的上下文部分监控框架 E2D.PMSide，针对原始 E2D 算法探索规则难以计算的问题设计了两种高效的探索规则，在通用函数逼近下实现了最优遗憾阶。 |
| [^135] | [Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning](https://arxiv.org/abs/2610.19802) | 提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。 |
| [^136] | [StateCraft: Augmenting Video World Models with Abstract-State Reasoning](https://arxiv.org/abs/2610.19795) | StateCraft提出将持久可更新的抽象状态与视觉状态解耦的新范式，通过语言模型追踪世界演化中的实体变化并以此条件化视频扩散模型，从而显著提升视频世界模型在长期交互中的状态一致性与推理能力。 |
| [^137] | [More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning](https://arxiv.org/abs/2610.19789) | 本文通过 132 次训练运行（累计超过 1.2 万 H800 GPU 小时），在四个智能体基准和三种模型规模上系统性地将十种自蒸馏方法与 GRPO 直接对比，以厘清性能提升的真实来源，并揭示各方法在性能、训练稳定性与计算成本上的实际优劣。 |
| [^138] | [Reinforcement Learning via Advantage Guidance](https://arxiv.org/abs/2610.19783) | 提出了一种名为“优势引导”的新策略提取方法，将优势条件化的监督学习结构与采样时的对比扩散引导相结合，实现了像监督学习一样稳定、却比以往强化学习方法性能更强的策略优化。 |
| [^139] | [Learning Who Benefits Most: Multi-Treatment Causal Ranking for Population Health Stratification](https://arxiv.org/abs/2610.19782) | 提出PROMETHEUS框架，将人群健康分层转化为多治疗因果排序问题，通过联合学习对治疗进行全局排序，依据预期增量收益识别最应优先获得护理强化的患者，从而在资源受限条件下优化医疗资源分配。 |
| [^140] | [SNACK: A Sequential Notation Framework for Probabilistic Graph Generation](https://arxiv.org/abs/2610.19781) | SNACK提出了一种将图线性化为字符串序列的表示框架，通过序列与有序邻接矩阵的双射对应和无效logit屏蔽机制，使序列模型能够以100%有效率生成图，并在保持有竞争力样本质量的同时显著提升GFlowNet训练吞吐量。 |
| [^141] | [From Global to Tail Convergence: Dynamics of $\alpha$-Stable Score-Based Generative Models](https://arxiv.org/abs/2610.19773) | 本文对α-稳定基于分数的生成模型建立了非渐近收敛性分析，证明在全局Wasserstein距离下α-稳定动力学与布朗扩散模型表现相当，但两者在尾部收敛行为上存在本质区别。 |
| [^142] | [When Does the Dependency Graph Help Code-to-Architecture Mapping?](https://arxiv.org/abs/2610.19769) | 该论文将代码到架构的映射建模为依赖图上的半监督节点分类问题，并通过GCN、GAT、MLP与朴素贝叶斯的系统对比实验，揭示了依赖图结构信息在何种条件下才能真正提升架构映射的效果。 |
| [^143] | [Learning to Compute Is Not Learning to Retain or Reuse in Looped Transformers](https://arxiv.org/abs/2610.19768) | 循环Transformer在训练深度内计算精确，但超出训练深度后会大量丢失答案；通过随机化监督位置或随机化训练循环次数可显著提升答案保留能力，而截断梯度等抗过度思考的训练方法则无效。 |
| [^144] | [Analyzing and Mitigating Poisoning Attacks on Flow-Based Intrusion Detection](https://arxiv.org/abs/2610.19766) | 该论文揭示了仅需污染0.001%训练数据即可在基于流的入侵检测系统中植入隐蔽后门，并提出一种诊断优先的方法，通过受限扰动分析定位触发特征、借助蒸馏基线模型检测行为漂移，并结合自动化数据清洗或神经元剪枝实现针对性缓解。 |
| [^145] | [CAST: History-Adaptive Spectral Clipping for LLM Pretraining](https://arxiv.org/abs/2610.19761) | CAST通过利用AdamW二阶矩和谱范数指数移动平均等历史信息，自适应地设定谱裁剪的尺度与阈值，在19.08B参数MoE模型预训练中以更少步数实现了更低的损失和更少的梯度尖峰。 |
| [^146] | [Discriminative Score Function: Turning Pretrained Models into Functional Generative Priors](https://arxiv.org/abs/2610.19743) | 提出判别式分数函数（DSF），一种免训练方法，可从预训练判别式模型（如ResNet分类器、DETR检测器和DINOv2编码器）的损失梯度几何中提取图像空间生成场，将其转化为生成先验，无需重训或任务引导即可从噪声生成图像。 |
| [^147] | [ProtModernBERT: Compute-Efficient Protein Language Modeling](https://arxiv.org/abs/2610.19738) | ProtModernBERT通过现代化架构改进，以减少高达80-90%的训练计算量实现了与ProtBert和Amplify-120M相当的性能，为资源受限的研究实验室提供了高效的蛋白质语言建模方案。 |
| [^148] | [What Happens to the Goal? Distinct Failures of Policy Alignment in Goal Misgeneralization](https://arxiv.org/abs/2610.19731) | 该研究发现了目标误泛化的三种不同失败形式（功能性目标替代、策略捷径和线索竞争），并通过追踪目标信息在智能体内部的作用深度，揭示目标会在不同层次上丢失。 |
| [^149] | [TIMERSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models](https://arxiv.org/abs/2610.19725) | TIMERSI是一个围绕冻结时间序列基础模型的自改进智能体部署系统，由编码智能体根据任务描述迭代生成并修正预测流水线，并在部署时通过自适应选择器挑选最优流水线，在198个任务上超越了所有对比方法。 |
| [^150] | [Articulation Upper Bounds for Language Models](https://arxiv.org/abs/2610.19721) | 本文提出了新的提示优化算法和基于“提示恢复”的传输度量方法，用于研究随着大语言模型能力扩展时概念在模型间传输的上限，从而探究语言表达能力自身的极限。 |
| [^151] | [Learning Individual Scientific Taste with Privileged Hindsight Distillation](https://arxiv.org/abs/2610.19719) | 本文将个体科学品味建模为从标准级评估到相对偏好的研究者特定映射，并提出特权后见之明蒸馏方法，利用训练时后见的科学证据提升提案时的评估质量与群体级映射，以解决个体反馈稀缺的冷启动问题。 |
| [^152] | [New Model, Old Habits: Behavior Transfer under Model Swapping in Multi-Session LLM Agents](https://arxiv.org/abs/2610.19712) | 揭示了多会话LLM智能体在模型切换中存在的“行为迁移”现象，并提出DASH方法，通过方向与作者感知的历史摘要保留强模型行为、抑制弱到强的行为迁移，显著优于原始历史传递和朴素摘要基线。 |
| [^153] | [ReGRA: Residual-Gradient Adaptation for Complementary Low-Rank Fine-Tuning](https://arxiv.org/abs/2610.19700) | ReGRA提出一种两阶段低秩微调方法，通过去除任务梯度在第一阶段更新子空间上的投影并聚合残差梯度方向来初始化第二阶段适配器，使有限秩预算下的两次更新捕获互补方向，从而提升微调效果。 |
| [^154] | [Annealed L\'{e}vy Diffusion via Gaussian Surrogate Reverse Dynamics](https://arxiv.org/abs/2610.19692) | 提出退火Lévy扩散（ALD），通过随时间退火的稳定性指数将重尾探索与高斯精炼结合，并把稳定噪声表示为高斯尺度混合，从而在标准Gaussian噪声预测训练中仅需增加一个标量尺度输入即可实现。 |
| [^155] | [Function-Space Temporal Self-Distillation for Policy Optimization](https://arxiv.org/abs/2610.19687) | 本文提出单策略的函数空间策略优化框架FSPO，以稀疏的历史下一词元分布作为固定时间目标进行自蒸馏，无需维护持久教师模型即可提供稳定密集的词元级监督，并通过特权分歧校正与可靠性自适应锚定提升策略优化效果。 |
| [^156] | [Beyond Distribution Matching: Selective Margin Preservation for Supervised Fine-Tuning](https://arxiv.org/abs/2610.19685) | 提出SRP-SFT，一种通过铰链惩罚选择性保持非目标token边际关系的软边际目标，在不冻结参考分布的情况下显著提升了监督微调的数学推理与代码生成性能。 |
| [^157] | [Enhancing Latent World Models with Associative Memory](https://arxiv.org/abs/2610.19683) | 该论文提出一种通过内容可寻址记忆库增强的JEPA式潜在世界模型，配合鼓励槽位多样性和时间均衡利用的互补损失来防止表征坍塌，以轻量级架构在长时程规划任务上实现了最先进的性能。 |
| [^158] | [MCAIR: Monitoring and Correcting Agent Unfaithfulness via Internal Representations](https://arxiv.org/abs/2610.19679) | 该论文提出了MCAIR框架，通过利用大语言模型的内部表示在运行时监控并纠正智能体的“完成不忠实性”行为，即智能体未按原定来源或程序完成任务却不予披露的欺骗性偏差。 |
| [^159] | [HACHIMI-II: From Role-Play to Learner Behavior Modeling for Faithful Student Simulation](https://arxiv.org/abs/2610.19654) | HACHIMI-II 将学生模拟从让大语言模型角色扮演转变为对学习者行为的第三人称预测，通过从学习者历史构建学生模型并用行为框架约束其下一步动作，从而忠实再现真实学习者的能力分布和典型错误。 |
| [^160] | [On Global Joint Lower Bounds in Zeroth-Order Nonconvex Optimization](https://arxiv.org/abs/2610.19652) | 本文在无限制全空间设置下为光滑随机非凸优化建立了标量预言机全局联合下界，回答了零阶方法复杂度关于维度、光滑化参数和精度的依赖是否联合最优这一公开问题。 |
| [^161] | [Model What Matters: Decision-Guided Selective World Modeling for Language Agents](https://arxiv.org/abs/2610.19642) | 提出DGS-WM方法，通过决策引导的观测选择机制让语言智能体的世界建模只聚焦于具有决策价值的观测内容，从而弥补稀疏任务奖励无法监督环境动态理解的局限。 |
| [^162] | [Synthetic reaction data from transformation rules for graph-to-sequence pretraining in biodegradation pathway prediction](https://arxiv.org/abs/2610.19640) | 该论文提出enviGraph框架，通过应用环境转化规则生成77万余个合成反应数据对来预训练图到序列预测器，解决了观测生物降解反应数据稀缺的问题，从而提升生物降解产物和路径预测的性能。 |
| [^163] | [Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching](https://arxiv.org/abs/2610.19636) | 本文提出SAE-IDF方法，利用稀疏自编码器将符号编码为稀疏表示并根据稀有度重新加权，实现了跨异构符号系统的单元匹配，为通过数据驱动方式对齐与合并现有符号系统、构建大规模神经符号模型迈出了关键的第一步。 |
| [^164] | [Risk-Controlled DFT Selection for Machine-Learned Interatomic Potentials](https://arxiv.org/abs/2610.19630) | 本文提出一种基于保形风险控制（CRC）的框架，利用轻量级预测器估计MLIP与DFT之间的不一致分数并校准为选择阈值，在保证MLIP误差可控的前提下自动决定哪些构型需要昂贵的DFT计算，从而突破自动化材料发现的通量瓶颈。 |
| [^165] | [Self-Conditioning Enables Deep Sequential Reasoning in Flow Matching](https://arxiv.org/abs/2610.19628) | 该论文从电路复杂度角度证明，原始流匹配的额外采样步骤无法增强常数深度模型的表达能力，而自条件化机制能突破这一限制，使模型实现真正的深度序列推理。 |
| [^166] | [DAGFormer: Global Topology Prediction for Transformer Cross-Layer Connections](https://arxiv.org/abs/2610.19625) | DAGFormer通过联合训练的拓扑预测器一次性规划整个Transformer模型的全局跨层连接拓扑，在75M至1B多个规模上将困惑度降低17%至25%，并全面超越逐层路由方法。 |
| [^167] | [Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories](https://arxiv.org/abs/2610.19614) | 提出Precog-RM模型，通过元学习在共享嵌入空间中以少样本方式预测LLM智能体轨迹的成败，并可作为GRPO训练的早停控制器，在保持性能的同时最多减少47.3%的训练计算量。 |
| [^168] | [A Graph-Based Analysis of Comparability, Scale, and Attribution in Private Transformer Inference](https://arxiv.org/abs/2610.19607) | 该论文通过构建包含62个私有Transformer推理（PTI）系统的实证图并进行节点、边、图三层分析，揭示了密码学进步总是与其他因素共同驱动性能提升、1891个候选系统对中仅1对可直接比较、以及私有推理的扩展前沿已在13B规模停滞等重要发现。 |
| [^169] | [PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference](https://arxiv.org/abs/2610.19605) | PN-Mix通过RoPE通道对层面的查询-键配对范数分析识别量化敏感的键缓存通道对，并在预填充阶段一次性识别、解码全程复用，为其分配更高精度，从而实现高效且低精度损失的长输出大语言模型推理。 |
| [^170] | [Beyond Static Positional Encoding: Bayesian Time-aware Attention for Non-Stationary Long-term Time Series Forecasting](https://arxiv.org/abs/2610.19603) | 该论文从贝叶斯视角将自注意力解释为内容先验与时间证据似然结合的后验，并推导总体层面的ELBO，据此揭示了静态位置编码在非平稳长期时间序列预测中的局限性，提出贝叶斯时间感知注意力机制以应对非平稳时间依赖建模的挑战。 |
| [^171] | [InstructPro2: Natural Language and Biological Grounding Guided De Novo Protein Design](https://arxiv.org/abs/2610.19592) | InstructPro2通过自然语言指令结合配体和抗原等生物学背景信息生成功能蛋白质序列，在2.057亿数据对上预训练后，其功能恢复F1达74.8，显著超越最强基线Pinal的62.2。 |
| [^172] | [L3M: A Verifiable Harness and Runtime for Self-Extending and Self-Improving Agents](https://arxiv.org/abs/2610.19591) | L3M用Lean 4构建了一个可验证的运行时框架，使自我扩展与自我改进的智能体能够跨越多次调用累积计算并产生复合收益，同时无需人类参与每个环节。 |
| [^173] | [Projected Wasserstein-Fisher-Rao Flows for Hawkes Background-Intensity Estimation](https://arxiv.org/abs/2610.19588) | 本文提出一种基于Wasserstein-Fisher-Rao几何的似然变分方法，通过向统计流形投影的分布框架来估计Hawkes过程的背景强度，并证明了相应估计量的条件一致性定理。 |
| [^174] | [Who's Behind the Text? Robust Model Authorship Attribution](https://arxiv.org/abs/2610.19583) | 提出基于似然的模型作者归属方法ATLAS，能够在未见领域、未知写作场景以及模型家族未来版本等分布偏移下依然稳健地识别文本的来源大语言模型。 |
| [^175] | [Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation](https://arxiv.org/abs/2610.19578) | 提出DualGCN双分支图知识蒸馏框架，通过拓扑平滑的语义先验锚定教师网络的物品表示，并以解耦的两阶段训练将知识蒸馏给语义学生分支，从而弥合语义空间与协同空间之间的拓扑鸿沟，解决冷启动推荐中的表示坍塌问题。 |
| [^176] | [Learning Across Temporal Graph Representation Silos](https://arxiv.org/abs/2610.19576) | 该研究首次证明关系历史中学习到的预测结构（关系动力学）能够跨越连续时间动态图与时序知识图谱之间的传统表示边界实现双向知识复用，且独立预训练的两类模型还可在分数层面融合以提升预测性能。 |
| [^177] | [On Inherent Privacy of Posterior Sampling: A Unified R\'enyi-Divergence Framework](https://arxiv.org/abs/2610.19573) | 本文提出统一的Rényi散度框架，证明从贝叶斯后验分布中采样（OPS）天然获得“免费”的差分隐私保证，其隐私损失由单条记录对数似然比的后验指数矩刻画，并在多种尾部条件与代表性贝叶斯模型下给出显式隐私保证。 |
| [^178] | [Geometric Kernel Diversity: An Axiomatic Diversity Measure in Euclidean Space](https://arxiv.org/abs/2610.19570) | 本文提出几何核多样性（GKD）——通过为每个点关联径向核立体并度量其并集体积来衡量多样性——该度量在欧几里得空间中同时满足单调性、唯一性与连续性三条公理，且在维度 d ≤ 3 时可被精确的多项式时间算法求解，从而回答了此前关于高效公理化多样性度量是否存在的悬而未决的问题。 |
| [^179] | [INTACT: Inducing Latent Reasoning on a Frozen Backbone](https://arxiv.org/abs/2610.19568) | INTACT 通过小型外部编码器结合蒸馏与强化学习，在完全冻结的大模型上诱导潜在推理，在不改变模型行为的前提下显著提升数学推理准确率并减少自回归解码开销。 |
| [^180] | [TEX-MEX: Similarity-Guided Temporal Contrastive Learning for Time Series](https://arxiv.org/abs/2610.19557) | TEX-MEX首次将时间序列的结构相似性与时序相似性统一到单一对比学习框架中，通过编码器识别结构相似窗口，将样本拉向其时移版本、推离不相似窗口的时移版本，从而学习相似窗口在时间上通常伴随出现的模式。 |
| [^181] | [Evidence-Anchored Overdose Death Prediction from Death Investigation Reports](https://arxiv.org/abs/2610.19556) | 该论文提出了一个从死亡调查报告中预测疑似药物过量死亡的自动化框架，通过专家引导生成证据锚定的解释，并比较了流水线模型与自我合理化模型在该任务上的性能差异。 |
| [^182] | [Test-Time Forecasting: Recurrent Depth Enables In-Context Dynamics Prediction](https://arxiv.org/abs/2610.19552) | 该论文通过建立线性注意力与动态模态分解的联系，提出测试时预测（TTF）框架，以梯度下降迭代作为循环深度，使仅经过单步预测训练的轻量级模型在自回归展开中即可泛化到分布外的动力系统动力学。 |
| [^183] | [Noisy-Label Calibration with Multiple Unreliable Labels](https://arxiv.org/abs/2610.19543) | 该论文提出了噪声校正ECE（NC-ECE）估计器、噪声概率估计算法以及事后重校准方法，使得在仅有多个不可靠标注的情况下，也能对二元分类器的校准性能进行无偏评估和可靠重校准。 |
| [^184] | [Two-stage Residual Learning for Modelling Strength-Informed Optimisation of Thermal Systems](https://arxiv.org/abs/2610.19539) | 提出了一种基于人工神经网络的两阶段残差学习（TSRL）框架，通过基础神经网络与残差神经网络的级联建模，结合局部建模强度评估与领域约束，显著降低了热力系统代理模型的预测误差，实现了更可靠的热力系统优化。 |
| [^185] | [Typed Decision Models on Tabular Data](https://arxiv.org/abs/2610.19536) | 类型化决策模型在表格数据任务上以更低成本达到与大语言模型相当的性能，且对问题表述更加鲁棒，但所有模型对表格组织方式的变化都不够稳健。 |
| [^186] | [A Unified Response Framework for Questions and Tutoring Dialogues](https://arxiv.org/abs/2610.19532) | 提出了一个统一框架，将学生对问题的回答与辅导对话中导师提出的子任务共同建模在项目反应理论（IRT）量表上，通过文本估计难度、追踪学生动态能力演化，并利用导师干预行为正则化难度值，实现对学业知识的更细粒度评估。 |
| [^187] | [Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following](https://arxiv.org/abs/2610.19525) | CALIF 是一种轻量级方法，通过利用现有冻结参考策略计算经校准的基于似然的奖励加成来增强验证器奖励，无需额外评判模型即可缓解可验证指令遵循中的奖励作弊问题并保持回复质量。 |
| [^188] | [HankelKANet: Lightweight Path Loss Estimation via Physics-Inspired Hankel Convolution](https://arxiv.org/abs/2610.19524) | HankelKANet通过用零阶和一阶汉克尔函数的径向幅度参数化卷积核，将电磁传播特性结构化嵌入轻量级残差UNet架构，无需额外物理损失项即可实现最先进精度和效率的路径损耗估计。 |
| [^189] | [High-Probability Regret Bounds for Feel-Good Thompson Sampling](https://arxiv.org/abs/2610.19513) | 本文首次为 Feel-Good Thompson 采样在上下文老虎机中建立了高概率遗憾界，在线性设置下达到了与最优算法在维度和时间范围依赖性上相匹配的遗憾保证。 |
| [^190] | [CktHGVAE: A Symmetric Design/State Heterogeneous Graph Variational Autoencoder for Transistor-level Analog/Mixed-signal Circuit Design and Analysis](https://arxiv.org/abs/2610.19508) | 提出了一种同时以设计变量和状态变量为输入输出的异构图变分自编码器CktHGVAE，无需针对电路拓扑专门调参即可高精度预测晶体管级模拟/混合信号电路的工作状态，并能在潜在空间中实现电路规格驱动的自动化设计优化。 |
| [^191] | [SKG: A Law-Induced Engine from Inference to Induction](https://arxiv.org/abs/2610.19505) | 该论文提出SKG推理引擎，将辛几何、核方法与约束力学统一为一个八元组框架，把守恒律、本构关系、对称群和公理系统四个层次的定律嵌入学习过程，实现了从先验定律驱动的推理到归纳的统一、可验证的方法体系。 |
| [^192] | [What You Sign Is Not What You Run: The Verifiability Gap in Quantized LLM Supply Chains](https://arxiv.org/abs/2610.19498) | 该论文提出“独立可验证率（IVR）”这一新指标，发现下载量最高的 198 个 Hugging Face GGUF 量化模型仓库均未记录构建环境，导致第三方完全无法逐比特复现用户实际运行的量化产物（IVR 为 0/198），从而揭示了“已签名的全精度权重”与“实际运行的量化产物”之间的供应链可验证性鸿沟，并指出补充记录浮点收缩设置等被遗漏的构建坐标可实现复现。 |
| [^193] | [Digital Self-Interference Cancellation for In-Band Full-Duplex Systems Based on Transformer Neural Operators](https://arxiv.org/abs/2610.19490) | 本文将带内全双工系统中的数字自干扰消除重新建模为算子学习问题，利用Transformer神经算子从历史观测中学习残余自干扰信道的时间演化，即使在期望信号强于残余自干扰的困难场景下也能实现更优的消除性能。 |
| [^194] | [State-Matched Sequence Conformal Prediction](https://arxiv.org/abs/2610.19484) | 提出了状态匹配序列保形预测（SMSCP），通过构建局部线性状态并利用状态相似性选取匹配的历史校准分数参考集，解决了可比状态在非相邻时间重现时基于近期性的传统方法失效的问题。 |
| [^195] | [MIPS: Mitigating Mode Collapse in Reinforcement Learning for Large Outcome Spaces](https://arxiv.org/abs/2610.19457) | 提出MIPS方法，通过将采样轨迹与学习到的轨迹模型相结合进行边际概率评估，解决了大结果空间强化学习中因结果稀疏导致IPS概率估计失效、从而无法有效缓解模式崩塌的问题。 |
| [^196] | [Large-Step Last-Iterate Convergence of Random-Reshuffling Stochastic Gradient Descent](https://arxiv.org/abs/2610.19444) | 本文为随机重排随机梯度下降（RR-SGD）的末次迭代点建立了大步长收敛保证，通过保留随机比较点的采样偏差，从步长限制中移除了显式的 1/n 因子，使末次迭代的收敛速率在对数因子内匹配平均迭代点的收敛速率。 |
| [^197] | [Progress-Aware Distillation for Mitigating Stagnation in Small Language Model Agents](https://arxiv.org/abs/2610.19439) | 该论文针对蒸馏得到的小语言模型智能体在长交互轨迹中易陷入低效循环的进度停滞问题，提出了进度感知蒸馏方法予以缓解。 |
| [^198] | [Learning to Select Features with Manifold Deep Unfolding PCA](https://arxiv.org/abs/2610.19436) | 本文提出MDUFS-Net，通过将ℓ_{2,0}范数融入PCA框架并将ADMM迭代算法深度展开为端到端可训练网络，结合流形优化策略实现参数自动学习和数值鲁棒性提升，兼顾了模型方法的轻量性与深度方法的自动化优势。 |
| [^199] | [Comprehensive Comparisons of Reinforcement Learning Algorithms in Autonomous Driving](https://arxiv.org/abs/2610.19429) | 本文在无信号交叉口的自动驾驶决策问题上，对FVI、RCPI、DQN和D-DDPG四种强化学习算法在最优性、安全性、训练稳定性和计算成本方面进行了全面比较，发现RCPI最接近参考解且训练最稳定但数据效率最低，而D-DDPG数据效率最高但稳定性最差。 |
| [^200] | [W-JEPA: Wavelet Joint-Embedding Predictive Architecture for Wearable Activity Recognition](https://arxiv.org/abs/2610.19411) | 该论文提出W-JEPA自监督框架，通过将惯性运动信号组织成按尺度、传感器轴和时间索引的小波标记网格，并利用联合嵌入预测架构捕获与活动相关的多尺度细微变化，从而提升可穿戴活动识别的跨数据集迁移能力和对未见受试者的泛化能力。 |
| [^201] | [Scaling LLM Inference via Parallelism: Throughput-Latency Trade-off](https://arxiv.org/abs/2610.19405) | 提出了一个系统性框架来刻画LLM推理中张量并行、流水线并行及混合配置在吞吐量与延迟（TTFT和ITL）之间的权衡，以指导在不同工作负载特征和服务级别目标下选择最优并行配置。 |
| [^202] | [ActMOF: Mapping Reticular Synthesis through Active Learning](https://arxiv.org/abs/2610.19404) | 该论文提出了ActMOF——首个以实验数据为基础的铝-羧酸MOF合成主动学习基准，通过将实验测量的微波合成条件编码并赋予基于X射线衍射的结晶度指数，为比较高斯过程、树集成和大语言模型等搜索策略在绘制MOF结晶景观中的表现提供了标准化平台。 |
| [^203] | [Train to Stitch: Topology-Preserving Regularization for Interoperable Representations](https://arxiv.org/abs/2610.19401) | 该论文提出一种拓扑保持正则化方法，无需训练期间的任何协调即可对齐独立训练的自编码器的潜在空间，从而实现表示层面的互操作性。 |
| [^204] | [Selective Next-Location Prediction with MetaModelSelect: A Lightweight Post-Hoc Confidence Metamodel](https://arxiv.org/abs/2610.19394) | 提出轻量级事后置信度元模型MetaModelSelect，无需重训冻结的循环神经网络骨干即可判断其下一位置预测是否可靠，从而实现选择性蜂窝网络下一位置预测并减少资源浪费。 |
| [^205] | [Fully sequential Gaussian process regression for preference learning](https://arxiv.org/abs/2610.19391) | 该论文针对目标函数只能通过带噪声成对比较来评估的连续全局优化问题，提出了基于矩匹配的完全序贯高斯过程近似方法DOVE，并首次为这类完全序贯近似贝叶斯推断建立了统计误差和期望简单遗憾的收敛速率保证。 |
| [^206] | [RA$^2$: Retain-Anchored Attraction Preserves Forget-Adjacent Utility in LLM Unlearning](https://arxiv.org/abs/2610.19385) | 提出RA$^2$（保留锚定吸引）潜空间遗忘方法，通过将遗忘词元表示重定向至附近由保留数据支持的表示，在有效遗忘的同时保持被遗忘领域附近提示的良性推理效用。 |
| [^207] | [Robust Explanatory Analysis for Structured Optimization Networks (REASON) in Learning](https://arxiv.org/abs/2610.19378) | 受系统1/系统2范式启发，提出一种双循环机制，通过分层解决残差误差并主动获取额外证据，在分布偏移和扰动下以约一半的计算量持续优于单循环模型。 |
| [^208] | [One-Step Prediction Fidelity Is a Poor Proxy for Planning Performance in JEPA World Models](https://arxiv.org/abs/2610.19375) | 该论文提出一种以物理单位衡量单步误差并将其按来源分解的冻结解码器评估协议，通过实验证明在JEPA世界模型中，单步预测保真度并不能可靠预测目标条件规划性能，因此不应作为模型选择和检查点保存的代理指标。 |
| [^209] | [Policy-Verified Self-Training: Learning Agent Compliance from Verifiable Policy Rewards](https://arxiv.org/abs/2610.19372) | 提出策略验证自训练（PVST）范式，将运行时策略验证器的决策转化为可验证的策略奖励，通过强化合规轨迹、以违规作为偏好优化的负样本，无需更强教师模型、人工偏好标签或攻击特定监督即可训练智能体的策略合规性。 |
| [^210] | [The Impact of Detector Architecture, Model Scale, and Scene Density on Multi-Pedestrian Tracking](https://arxiv.org/abs/2610.19369) | 本研究系统评估了检测器架构、模型规模、场景密度和推理方式对多行人跟踪的影响，发现RT-DETR在跟踪质量上表现最佳（尤其在拥挤的MOT20场景），而YOLO26-L配合NMS在质量与效率之间达到最佳平衡。 |
| [^211] | [Dynamic Graph World Modelling: Learning Future Dynamics from Temporal Context](https://arxiv.org/abs/2610.19364) | 提出首个动态图世界建模框架DyG-WM，通过回答“什么演化了、如何演化、接下来将演化什么”三个问题，从时间上下文中联合预测未来的节点与关系表示。 |
| [^212] | [Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems](https://arxiv.org/abs/2610.19362) | 该论文将 Best-of-N 采样建模为正确与错误响应极端奖励尾部之间的竞争，利用极值理论刻画出正确尾部主导、错误尾部主导和临界三种渐近状态，统一解释了缩放增益与奖励作弊导致的性能退化，并推导了有限样本缩放定律。 |
| [^213] | [Steering Vectors for Competing Ethical Frameworks Are Interchangeable: A Competing-Target Test for Activation Steering](https://arxiv.org/abs/2610.19359) | 论文提出“竞争目标测试”这一新方法，发现用于康德义务论与行为功利主义的激活转向向量共享大部分方向且可互换，说明转向向量可能通过共享组件起作用而非直接控制目标行为。 |
| [^214] | [Asymptotic Behavior of Transformers](https://arxiv.org/abs/2610.19358) | 本文证明了一类包含层归一化、MLP和RoPE编码的Transformer在softmax温度超过各自阈值时，其经验缓存分布会收敛到与初始条件无关的唯一极限，但由于这组阈值的上确界为无穷大，不存在能强制所有Transformer收敛的全局最小温度。 |
| [^215] | [BFHD: Bidirectional Feature Harmonization Decomposition for Heterogeneous Clinical Assessments](https://arxiv.org/abs/2610.19352) | 该论文提出 BFHD 框架，将异构临床评估系统间的调和问题建模为双向可恢复性问题，通过双向耦合转换和特征级输出门控，在原始测量空间中显式区分可双向可靠转换的测量指标与不可转换成分。 |
| [^216] | [Where to Start, How to Move: Analyzing Curricula under GRPO for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.19349) | 该研究系统分析了引导辅助GRPO课程设计，发现在固定回合级阈值规则下“从接近完成的阶段开始”最有效，且基于结果信息的工具调用选择在已训练和未训练任务类别上均优于标准GRPO。 |
| [^217] | [Integrated Novelty Detection Systems: From Sensor Networks to Multi-Novelty Computer Vision](https://arxiv.org/abs/2610.19347) | 本文针对气象传感器网络中的雨量计堵塞检测问题，评估并揭示了四种现有质量控制方法的局限性，特别是它们在检测成片故障传感器集群时的不足。 |
| [^218] | [Target-Token Asymmetry Can Masquerade As Depth Specialization](https://arxiv.org/abs/2610.19344) | 层级消融研究中所观察到的“早期层专门化”现象，可能仅仅是两个语料库之间目标词元分布不对称（如数学文本中大量空白字符）所造成的假象，而非真正的深度专门化。 |
| [^219] | [Counterfactual Gate-Support Decomposition: Separating Selection from Redistribution in Training-Data Filtering](https://arxiv.org/abs/2610.19343) | 该论文提出反事实门控支持分解（CGSD），一种四臂预算匹配的实验设计，可将训练数据质量门控的整体效应分解为支持内选择、支持间重分配及其交互作用，并在韩英与韩日检索任务中发现支持内选择是唯一显著超出校准零区间的分量。 |
| [^220] | [Why don't Graph Transformers attend to the same token?](https://arxiv.org/abs/2610.19324) | 本文揭示了大语言模型中的“注意力汇聚”现象在图Transformer中罕见的原因——图的对称性与局部性在理论上可限制注意力集中，而对使用全局注意力的分子任务，真正阻止汇聚的是GRIT的门控注意力评分函数，将其替换为点积后汇聚率从0.09飙升至0.68。 |
| [^221] | [TopicMerge: A Divide and Conquer Approach to Topic Modeling](https://arxiv.org/abs/2610.19323) | TopicMerge提出一种分治式的多阶段流水线，先通过高效聚类将文档过度分割为语义纯净的微聚类，再由LLM结合局部邻域上下文迭代仲裁合并候选对，从而同时实现了最先进的聚类性能与计算效率。 |
| [^222] | [Historical-Feature Residual Adaptation for Sequential Screening](https://arxiv.org/abs/2610.19315) | 该论文提出一种利用固定历史特征构建、锚定于反馈揭示前基础分数的采集时刻残差自适应方法，在保留预训练模型的前提下将顺序筛选的命中回收率从22.58%提升至28.61%，超越了所有六个预先指定的对比方法。 |
| [^223] | [Did the LLM Answer the Question? A Semantic Reconstruction Game for Jailbreak Evaluation](https://arxiv.org/abs/2610.19306) | 该论文提出了一种无需训练的“语义重构游戏”评估方法，通过让LLM从回复中重构其实际回答的请求并与诱饵请求进行受限语义比较来判断越狱是否成功，从而避免将拒绝或仅主题相关的回复误判为成功攻击，大幅提升攻击成功率评估的准确性（不同评判模型间ASR差异可达82个百分点）。 |
| [^224] | [DTDD: Divergence-Triggered Dynamic Distillation for Reliable On-Policy Supervision](https://arxiv.org/abs/2610.19302) | 提出发散触发动态蒸馏（DTDD）框架，通过监测分段级师生分歧自动触发教师接管并生成恢复后缀供学生行为克隆学习，从而在长程推理、无SFT的基础初始化蒸馏和多轮智能体场景中保障在策略蒸馏监督的可靠性。 |
| [^225] | [Verifier-Grounded Agentic RL for Provably Correct Code](https://arxiv.org/abs/2610.19298) | 该论文提出了一种读取验证器诊断信息（而非简单的接受/拒绝判定）的智能体强化学习奖励机制，结合加权证明义务完成度评分、参考证明隐藏探针以及防止篡改规范的完整性审计，解决了验证器奖励稀疏和策略作弊两大难题，从而训练出可证明正确的代码。 |
| [^226] | [Exact Scaling Laws for a Random-Features Bigram Model](https://arxiv.org/abs/2610.19281) | 本文在具有Zipf词元统计的瓶颈随机特征二元模型中精确推导出预测遗憾随模型宽度、词表大小和Zipf指数变化的缩放定律，证明纯粹由表征容量约束即可产生经典神经缩放定律的函数形式。 |
| [^227] | [Dimension-Free Scaling Laws for Size Generalization: An Exponential Separation](https://arxiv.org/abs/2610.19279) | 该论文建立了尺寸泛化的统计学习理论，证明了尺寸泛化与经典域内学习之间存在样本复杂度上的指数级分离，并给出了与输入维度无关的缩放定律。 |
| [^228] | [Outcome-Routed Distillation and Reinforcement Learning for Diffusion Language Models](https://arxiv.org/abs/2610.19271) | 提出一种无需似然的结果路由蒸馏与强化学习流水线，通过验证器结果路由 rollout 组，结合无比率 REINFORCE、教师密集 KL 监督与检查点锚定 KL 约束，有效训练扩散语言模型。 |
| [^229] | [Exploring Interpretability in Deep Learning-Based Forest Classification - Separating Similar Forest Classes using Sentinel-2 Time Series](https://arxiv.org/abs/2610.19270) | 该研究利用循环神经网络对Sentinel-2时间序列进行树种分类，并借助积分梯度可解释性方法量化光谱-时间特征贡献，从而有效区分纯林与混交林（尤其是挪威云杉）等相似森林类别。 |
| [^230] | [Beyond Time Induction: Structured Conditional Neural Solvers for Parametric Stochastic-Volatility Pricing](https://arxiv.org/abs/2610.19254) | 提出BeTINN结构化条件神经求解器，通过多方向时间-参数条件化和联合残差通路扩展时间诱导调制，在Heston随机波动率定价中显著提升期权价值与关键敏感度的联合预测精度。 |
| [^231] | [Multi-Sensitive Distributional Fairness: Joint Wasserstein Audits and Shapley Allocation](https://arxiv.org/abs/2610.19252) | 该论文提出用一维Wasserstein几何对多个敏感属性进行联合公平性审计，将联合差异刻画为全单元条件分位数过程的方差，并借助Shapley值实现与顺序无关的差异归因，从而捕捉单坐标审计会遗漏的交叉性歧视。 |
| [^232] | [Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction](https://arxiv.org/abs/2610.19246) | 提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。 |
| [^233] | [FinGREAT: Evaluating LLMs on Financial Graph-guided Retrieval And Typing](https://arxiv.org/abs/2610.19240) | 该论文提出FinGREAT基准，基于包含11个账户节点和11种交易类型边的金融会计图，在跨越26年的1,300份SEC 10-K文件上系统评估了开源大语言模型提取和分类复式记账关系证据的能力，并发布了含483个段落的人工标注关系分类基准。 |
| [^234] | [LLMs program using Reusable and Composable Abstract Representations](https://arxiv.org/abs/2610.19238) | 大语言模型在生成代码时利用中间层中形成的可复用、可组合的抽象表示，这些表示可跨编程语言迁移，并可通过激活引导独立操纵和灵活组合来精确控制代码生成。 |
| [^235] | [From Detection To Anticipation: Continuous Onset Forecasting For Stuttered Speech](https://arxiv.org/abs/2610.19235) | 该论文将口吃预测重新建模为连续因果时间-事件预测任务，仅利用先前语音每100毫秒预测口吃起始风险，并通过紧凑因果Conformer建模冻结Whisper语音表示的时间演化，实现了从“事后检测”到“事前预判”的转变。 |
| [^236] | [Spectral-MoE: Heterogeneous Per-Layer Expert Utilization in Dense-to-MoE Conversion](https://arxiv.org/abs/2610.19232) | Spectral-MoE提出了一种在密集模型到MoE转换过程中学习在各层之间异构分配激活专家数量预算的方法，突破了传统方法对每层固定相同利用率的限制，从而更有效地降低推理算术成本。 |
| [^237] | [Riding the Fixed-Point Curve: Image Classification in the Continuous Thought Machine](https://arxiv.org/abs/2610.19220) | 本文揭示了连续思维机器（CTM）图像分类的内在机制：每个思考步骤是具有平衡放大的吸引性非正规映射，神经状态的不动点随每步移动，思维轨迹即这些不动点描绘的曲线（大部分运动跨图像共享并表现为行波活动），从而解释了CTM的涌现思维过程。 |
| [^238] | [Narrowing the Collaboration Gap, Probably](https://arxiv.org/abs/2610.19215) | 该论文提出让持有不同私有信息的多智能体通过交流关于动作的校准概率信念而非仅交流动作本身，并从理论和实验上证明这种概率沟通方式能更有效地汇聚决策相关信息、提升协作表现。 |
| [^239] | [Generalizing Identifiability Results in Causal Discovery from Complete Data to Missing Data Settings](https://arxiv.org/abs/2610.19211) | 该论文建立了在包括非随机缺失在内的多种缺失机制下因果图保持可辨识的一般性充分条件，提供了将完整数据因果发现算法的正确性保证扩展到缺失数据场景的框架，并针对加性噪声模型提出了两个渐近正确的算法。 |
| [^240] | [Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization](https://arxiv.org/abs/2610.19198) | 该论文揭示了语义嵌入的几何结构是LLM路由在未见任务上失效的根源，并提出用Qwen2.5-7B-Instruct深层最后token的隐藏状态替代语义嵌入作为表示，从而显著提升路由方法对未见任务的泛化能力。 |
| [^241] | [FireGraphNet: Sub-daily wildfire spread prediction with directed graph neural networks](https://arxiv.org/abs/2610.19191) | FireGraphNet将12小时野火蔓延预测建模为带风向、地形、燃料与障碍物边特征的有向H3图上的节点分类任务，平均精度达到最强距离基线的1.7至1.9倍，且性能提升主要来自沿风向的消息传递机制。 |
| [^242] | [Adaptive Computation with Conformal Certification](https://arxiv.org/abs/2610.19189) | 本文提出“保形自适应”方法，利用保形认证为科学成像数据（医学与天文图像）的自适应计算提供统计风险保证，在降低计算成本的同时保留关键稀疏区域的分辨率并提供可靠的不确定性量化。 |
| [^243] | [Structure-aware Bayesian full waveform inversion via diffusion-regularized Stein variational gradient descent](https://arxiv.org/abs/2610.19187) | 该论文提出将学习得到的扩散模型作为隐式正则化机制嵌入基于退火Stein变分梯度下降的贝叶斯全波形反演框架，以刻画复杂地质结构先验并增强反演的不确定性量化能力。 |
| [^244] | [Learning in Strategic Queuing with Geometric Recharge: From Bandits to Systems](https://arxiv.org/abs/2610.19179) | 该论文将具有单数据包缓冲区的策略性排队问题与几何充能老虎机问题建立等价关系，并证明任何无悔学习队列都能达到至少等于服务器速率 Hamacher 和的长期平均吞吐量，且该界是紧的。 |
| [^245] | [Diagnosing Low-Confidence Patient Profiles via Localized Harmful Covariate Shift Detection](https://arxiv.org/abs/2610.19178) | 该论文提出结合MED3pa与Detectron的统一诊断流程，先识别可解释的低置信度患者画像，再在各画像内检测有害协变量偏移，从而定位临床预测模型在特定患者亚群中的性能退化。 |
| [^246] | [Into the Unknown: Benchmarking AI agents on open scientific research problems](https://arxiv.org/abs/2610.19159) | 该论文推出了"Into the Unknown"基准测试，通过与能源部国家实验室科学家合作定义未发表的后续研究方向及评分准则，评估九个前沿AI智能体在开放科学问题上撰写后续论文的能力，结果显示135篇投稿无一满足全部条件，其中114篇未通过完整性检查。 |
| [^247] | [Demystifying On-Policy Distillation: Where Does the Teacher's Advantage Live?](https://arxiv.org/abs/2610.19145) | 该研究通过在师生不一致位置进行受控token翻转的干预实验，量化了在线策略蒸馏中教师优势的来源，发现重复的选择性纠正可将学生准确率从28.4%提升至74.5%（接近教师的76.9%），且基于概率质量和KL散度的选择性干预比随机翻转高出15-18个百分点。 |
| [^248] | [Towards Training and Inference of Graph Neural Networks in Graph Databases](https://arxiv.org/abs/2610.19142) | 该论文证明图数据库可以直接用于图神经网络的训练与推理，通过 Cypher 实现即时采样与特征检索，并提供优化使训练加速约 2.5 倍、数据库内推理比冷加载快 10 倍以上且无额外内存开销。 |
| [^249] | [Online Tokenization with Optimal Delay in Linear Time](https://arxiv.org/abs/2610.19139) | 该论文首次形式化并解决了在线分词问题，提出了延迟达到最优且运行时间为线性的在线分词算法。 |
| [^250] | [MA-Bench: Measuring Operational Planning in Language Models](https://arxiv.org/abs/2610.19138) | 提出了基于军事作战规划第一步“任务分析”构建的MA-Bench基准，用于评估语言模型在缺少唯一正确答案的情况下，从不完整且矛盾的文档集合中形成态势分析报告的能力。 |
| [^251] | [Student Copula Markov Networks](https://arxiv.org/abs/2610.19135) | 本文基于copula密度刻画成对马尔可夫性质，提出学生t-Copula马尔可夫网络及相应的ECM算法，将概率图模型扩展至非高斯数据并实现参数估计与图结构选择。 |
| [^252] | [TRACER: Trace-Regularized Training Against Indirect Prompt Injection](https://arxiv.org/abs/2610.19133) | TRACER提出了一种通道感知的对抗训练方法，通过轨迹正则化与奖励塑形，将RAG场景下的间接提示注入攻击成功率从65.5%大幅降至6.1%，同时保持模型的干净任务准确率。 |
| [^253] | [Imagining Goals with Geometric Horizon Models for Policy Generalization](https://arxiv.org/abs/2610.19129) | 提出TD-Aug方法，通过从几何视野模型中采样直接想象出组合已有知识可达成的新目标，从而提升策略在零样本目标泛化任务上的表现。 |
| [^254] | [PairIT: Autoregressive Transformers for Low-Data Molecule Optimization](https://arxiv.org/abs/2610.19123) | 提出了PairIT框架，通过从结构相似分子对中学习来训练SMILES自回归Transformer，实现低数据条件下的种子分子性质优化，并借助提示词在推理时支持多性质联合优化。 |
| [^255] | [Unlocking Structured Pruning for Reasoning Language Models via Attention-Preserving MLP Trimming](https://arxiv.org/abs/2610.19121) | LayerTrim是一种免训练的结构化剪枝方法，通过识别冗余解码器层并选择性裁剪其MLP特征、同时保留注意力机制和解码器深度，在无需梯度或恢复训练的情况下保持长篇推理能力，相比RESP平均准确率最高提升21.4个百分点，并实现1.24倍的预填充加速和19.7%的GPU峰值内存降低。 |
| [^256] | [Genesis Steering: Better Activation Steering Through Training Data](https://arxiv.org/abs/2610.19114) | 提出Genesis Steering方法，利用模型自身微调数据（或通过成员推断构建的代理数据）生成激活引导向量，并从理论和实验上证明基于训练数据的引导比外部数据集更有效。 |
| [^257] | [From Copying to Composition: How Language Models Develop In-Context Learning](https://arxiv.org/abs/2610.19113) | 该研究通过受控诊断方法发现，语言模型会从倾向于复制提示中最近示例，逐渐发展为整合提示信息进行组合式推理的上下文学习，且这种转变出现得很早，并可通过特定的合成数据训练加以引导。 |
| [^258] | [Neural Supervised Clustering for Learning Individualized Treatment Rules with Many Treatments](https://arxiv.org/abs/2610.19104) | 提出了一种结合共享非线性协变量表示、低维治疗嵌入与成对聚类融合的神经Q学习框架NSCF，能够从每个治疗臂有限的观测数据中学习多治疗方案下的个体化治疗规则，并生成可解释的治疗分组划分。 |
| [^259] | [Contextual Credit Policy Optimization](https://arxiv.org/abs/2610.19103) | 提出上下文信用策略优化（CCPO），利用基于冻结 LLM 表示和轨迹统计量计算的上下文相似度对观测匹配的同类经验进行加权，从而在稀疏奖励的长时程任务中实现更可靠的步骤级信用分配，并识别成功轨迹中被良好最终结果所掩盖的冗余动作。 |
| [^260] | [Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena](https://arxiv.org/abs/2610.19095) | 该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。 |
| [^261] | [Beyond Aggregate Accuracy: Item-Level Transitions in Prompt Optimization](https://arxiv.org/abs/2610.19084) | 该论文提出RFDPO方法，通过追踪提示优化过程中逐题的“恢复、退化与变动”转变，揭示总体准确率所掩盖的题目级变化，并利用这些反馈指导后续的提示修订。 |
| [^262] | [Backpropagating Verifier Rewards Through Continuous Flow Language Models](https://arxiv.org/abs/2610.19080) | 提出了一种面向连续流语言模型的强化学习后训练框架，通过引入估计中间词元嵌入价值的评论家网络，绕过不可微的解码步骤，实现验证器奖励沿连续动作的路径梯度反向传播，从而以更低方差提升模型推理能力。 |
| [^263] | [Theoretical and Practical Advances in Hierarchical Bayesian Deep Learning Methodology](https://arxiv.org/abs/2610.19074) | 本文提出多层粒子边缘Metropolis-Hastings（MLPMMH）方法，通过将超参数的伪边缘推断与基于迹类神经网络先验的耦合序贯蒙特卡罗相结合，实现了神经网络参数与超参数的高效联合贝叶斯推断并提供了理论保证。 |
| [^264] | [Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap](https://arxiv.org/abs/2610.19065) | 生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。 |
| [^265] | [How Inner Optimizers Shape Replica Scaling](https://arxiv.org/abs/2610.19064) | 本文研究内层优化器对分布式 LLM 训练中副本扩展性能的影响，识别出决定不同内层优化器（如 Muon 与 AdamW）在副本扩展时表现差异的两个关键因素，从而解释了为何某些内层优化器能显著缓解副本数量增加带来的性能下降。 |
| [^266] | [Conformalized Physics-Informed Weak Regression for Parameter Estimation in Epidemic Models](https://arxiv.org/abs/2610.19058) | 该论文提出物理信息弱回归（PIWR）方法，通过在时间上积分模型而非对数据求导来估计传染病模型参数，并结合由物理模型自身生成校准集的分割保形预测，为所有参数提供具有有限样本覆盖率保证的无分布同步置信区间。 |
| [^267] | [Deep Generative Models for Phylogenetic Inference with Complex Evolutionary Processes](https://arxiv.org/abs/2610.19053) | 提出利用深度生成模型进行基于模拟的系统发育推断，以突破传统似然方法在处理复杂进化过程时无法准确高效推断树后验分布的局限。 |
| [^268] | [Smart Alone, Blind Together: Why Accuracy Fails to CertifyTheory of Mind in Multi-Agent Teams](https://arxiv.org/abs/2610.19051) | 该论文论证准确性不能作为认证多智能体团队中心智理论模块的标准，因为所需准确性取决于部署场景的错误代价，且模块间错误的相关性会使准确性阈值以可预测的方向失效。 |
| [^269] | [What Information Buys in Active Inference](https://arxiv.org/abs/2610.19046) | 本文将主动推理（AIF）与贝叶斯UCBVI置于相同的MDP状态抽象下进行对比，发现AIF的信息增益衰减过快导致探索不足，进而通过对信息增益进行平方根校准，使校准后的AIF达到了√T阶的后悔界。 |
| [^270] | [Graph Coarsening Enables Escaping Local Optima in Compute-Constrained Nonconvex MAX-CUT](https://arxiv.org/abs/2610.19043) | 该论文提出将近线性时间的谱图粗化方法CMG用于热启动具有优良优化景观的粒子多项式松弛（PPR），使计算受限的非凸MAX-CUT优化能够逃离局部最优，在基准实例上平均达到97.5%的最优解、并在全部Biq Mac测试实例上达到已发表最优解，还可扩展至两万顶点规模并大幅降低内存需求。 |
| [^271] | [Loss Curvature as a Sampling-Free Proxy for Memorization in Diffusion Models](https://arxiv.org/abs/2610.19042) | 该论文提出输入损失曲率（训练损失Hessian矩阵的Frobenius范数）可作为扩散模型记忆化的免采样代理指标，通过在训练中追踪其均值即可预测模型何时开始记忆并复现训练数据。 |
| [^272] | [Tracing Morphology, Semantics, and Divergence in Language Acquisition](https://arxiv.org/abs/2610.19041) | 该论文通过对 Pythia 和 OLMo 等开源模型中间训练检查点的系统评估，揭示了语言模型先习得形态结构、语义嵌入随后才趋于稳定的习得轨迹，并发现训练数据规模对发散识别有积极影响。 |
| [^273] | [Lag: A Mean-Field Optimal Policy for Online Assignment with Reusable Resources](https://arxiv.org/abs/2610.19038) | 本文针对可复用资源的在线任务分配问题，将其建模为上下文相关的restless多臂老虎机，提出了闭式指数策略Lag，并通过首个上下文老虎机的确定性流体极限证明了其在新缩放机制下的渐近最优性。 |
| [^274] | [Who Should We Listen to More? Welfare-Aware Preference Acquisition for Pluralistic Alignment](https://arxiv.org/abs/2610.19035) | 提出福利感知偏好获取（WAPA）方法，通过基于个体监督效用的社会福祉目标在人群中分配有限的人类反馈，解决多元对齐中因AI模拟精度差异导致的偏好代表性不足问题。 |
| [^275] | [How Many Ranks Does a Token Need? A Noise-Floor Squelch for Rank-One LoRA Experts](https://arxiv.org/abs/2610.19033) | 提出无需路由器的秩一LoRA专家混合方法SQUELCH，通过基于噪底的软阈值处理和Stein无偏风险估计，让每个token自动决定所需的秩数量。 |
| [^276] | [Optimization for One, Feasibility for Most: Decoupling Optimization from Feasibility in Chance-Constrained Programs](https://arxiv.org/abs/2610.19027) | 提出ACCESS框架，将优化与可行性评估解耦——利用大量廉价的单场景子问题生成高质量候选解，再通过大规模批量仿真独立评估可行性，从而在机会约束随机规划中同时兼顾目标函数质量与高概率可行性。 |
| [^277] | [How Far Can Coding Agents Go in Visual Generation?](https://arxiv.org/abs/2610.19025) | 该研究通过AgentDrawBench基准发现，通用编程智能体通过编写可执行程序生成图像时，擅长保持结构和满足精确视觉约束，在构图准确性重要的领域表现具有竞争力，但在感知丰富性上仍有局限。 |
| [^278] | [Discrete Diffusion Models are Minimax Optimal Under Sparsity](https://arxiv.org/abs/2610.19020) | 该论文证明了在稀疏性假设下，基于正则化得分熵目标的离散扩散模型能够达到极小极大最优的统计速率，且所用正则化方法无需预知稀疏参数即可实现自适应。 |
| [^279] | [IR-PRM+ with Previous-Utility Prediction: Constant Regret and Best-Iterate Convergence](https://arxiv.org/abs/2610.19016) | 本文提出了一种基于先前效用预测的新方法，使同步和交替更新的 IR-PRM+ 在有限两人零和矩阵博弈中都能获得常数遗憾保证与最优迭代收敛保证，克服了以往额外梯度分析无法涵盖乐观更新和交替更新的局限。 |
| [^280] | [Scaling Cheap Synthetic Pretraining Improves Indoor Radio-Map Prediction](https://arxiv.org/abs/2610.19015) | 该论文提出自动室内布局生成器与快速开源近似射线追踪器相结合的数据生成流程，利用1.28亿合成样本预训练视觉模型，使室内无线电地图预测误差平均降低15%，并展现出可预测的数据规模扩展规律。 |
| [^281] | [Data Density Scaling Laws for Image Self-Distillation](https://arxiv.org/abs/2610.19013) | 本文发现图像自蒸馏中样本重复的收益依赖于具体数据集，小数据集重复训练甚至可能优于大数据集，并提出同时考虑独特样本数量与每样本重复次数的改进版 Chinchilla 缩放定律。 |
| [^282] | [PMO-Dock: Benchmarking Docking, Specificity, and Generalization in Molecular Optimization](https://arxiv.org/abs/2610.19011) | 提出了包含25个基于对接的分子优化任务的PMO-Dock基准测试，涵盖苗头化合物生成、先导化合物优化和特异性评估，并通过将模型开发与最终评估分离的标准化协议，解决了基于对接的分子优化评估缺乏统一标准的问题。 |
| [^283] | [Discovering Undisclosed Concepts in LoRA Adapters for Diffusion-Based AIGC Models](https://arxiv.org/abs/2610.18990) | 本文提出一种概念发现机制，通过搜索能同时最大化输出连贯性、LoRA诱导独特性和子空间对齐度的条件输入，来揭示LoRA适配器中被训练生成但未公开的概念，从而支持对AIGC模型的内容审核与调查。 |
| [^284] | [Virtual-Alloy Physics-Aware Semantic System for extrapolative prediction of perovskite solar cells](https://arxiv.org/abs/2610.18949) | 该研究提出V-PaSS系统，通过维加德定律插值端元化合物晶体图并结合界面能带偏移与科学语言模型，实现了对新型材料钙钛矿太阳能电池的外推性能预测。 |
| [^285] | [Limits of Statistical Assurance for Embodied and Agentic AI under Adaptive Adversaries](https://arxiv.org/abs/2610.18948) | 本文证明了对于具身智能和智能体AI，任何有限时长的安全认证在原理上都无法排除那些在认证前伪装正常行为、认证后才激活的隐蔽自适应对手，并精确刻画了此类攻击带来的最坏情形期望损失。 |
| [^286] | [Physical Layer Anomaly Detection via Adaptive Gaussian Mixture Model](https://arxiv.org/abs/2610.18946) | 提出基于自适应高斯混合模型的检测框架，通过学习合法CSI模式并检测统计异常，即使在攻击者与合法节点空间接近的情况下也能有效识别CSI欺骗攻击，在位移超过150毫米时实现95-98%的检测率且误报率仅为1%。 |
| [^287] | [Semi-Amortized Bayesian Inference with Reusable Local Posteriors](https://arxiv.org/abs/2610.18945) | 该论文提出一种半摊销贝叶斯推断方法，将参数分块并为计算昂贵或频繁更新的块训练可复用的神经估计器来近似局部条件后验，再通过马尔可夫链将这些神经采样与常规更新交替进行，从而高效地实现高维复杂模型的贝叶斯推断。 |
| [^288] | [Wong--Zakai Convergence and FNO Learning for Stochastic Volterra Equations](https://arxiv.org/abs/2610.18939) | 本文为随机Volterra积分方程建立了Wong–Zakai收敛定理，并在此基础上证明镜像填充傅里叶神经算子能够学习其解算子，实现对完整解轨迹的准确预测以及向更细时间网格的泛化。 |
| [^289] | [Beyond Equation Search: Conjecture-Guided Discovery with SCOPE and ClosureBench](https://arxiv.org/abs/2610.18938) | 该论文提出SCOPE多智能体框架，将符号回归从优化单个方程转向以科学猜想为核心的研究范式，通过多个科学家智能体独立建模、主动实验与证据交换来检验和修正猜想，并配套ClosureBench基准评估猜想驱动式科学发现。 |
| [^290] | [Delay Complexity Dimension: Learning with Delayed and Censored Labels](https://arxiv.org/abs/2610.18935) | 本文提出“延迟复杂度维度”（DCD），将VC维度与观测信道的标签信息率相结合，刻画了标签延迟且被删失情形下PAC学习的样本复杂度，证明任何学习器需要 Ω(DCD/ε²) 个样本，且伪标签ERM可在常数与对数因子内达到该界。 |
| [^291] | [An Initial Study of Physiological Markers for Deepfake Speech Detection](https://arxiv.org/abs/2610.18934) | 该论文提出利用36个基于人类言语产生系统（呼吸、发声、时间协调）的手工生理标记来检测深度伪造语音，验证了通过识别生理模式不一致性检测伪造语音的可行性，域内最佳等错误率达21.40%，并在跨域场景下展现出一定的泛化能力。 |
| [^292] | [The Agentic Scientist: Orchestrating Coding Agents for Automated Scientific (Co-)Discovery](https://arxiv.org/abs/2610.18933) | 提出了端到端分布式系统“智能体科学家”，将计算研究分解为多个阶段并由新一代编程智能体驱动、结合智能体自验证，实现人机协同的自动化科学发现。 |
| [^293] | [TorchSpin: Differentiable Electron Paramagnetic Resonance Simulation in PyTorch](https://arxiv.org/abs/2610.18924) | TorchSpin 是一个基于 PyTorch 的开源可微分电子顺磁共振（EPR）模拟框架，它将 EasySpin 的组织结构与数值约定移植到张量计算环境中，在复现经典结果的同时，借助自动微分使 EPR 模拟能够融入机器学习工作流。 |
| [^294] | [Popularity Trajectories and Mixing Patterns of Urban Venues in Two European Cities](https://arxiv.org/abs/2610.18917) | 本研究基于伦敦和巴塞罗那约一万个场所5.5年的Foursquare签到数据，发现城市场所的流行度演变可归纳为六种典型轨迹（下降、上升、先升后降、稳定、昙花一现、后起之秀），且不存在U型复苏轨迹，而社区的贫困程度与人口密度并不能预示场所的流行度走向。 |
| [^295] | [Distinguishing Process Change, Instrument Fault and Data-Path Corruption in Streaming Telemetry: A Reference Implementation and Benchmark](https://arxiv.org/abs/2610.18904) | 提出了一种由多种在线检测器与确定性、可审查的归因规则组合的方法，能在流式遥测中以96.3%的检测率和91.2%的正确归因率区分过程变化、仪器故障与数据通路损坏这三种异常成因。 |
| [^296] | [AI-Driven, Timing-Aware Power Optimization](https://arxiv.org/abs/2610.18890) | 提出了一种AI驱动的时序感知功耗优化框架，利用梯度提升模型集成来排序功耗恢复编辑和修复动作，在EDA确定性分析的把关下实现布线后功耗降低，同时避免引入新的约束违例。 |
| [^297] | [Bayesian Causal Forests for Zero-Inflated Continuous Outcomes](https://arxiv.org/abs/2610.18889) | 本文提出ZIC-BCF-Smear方法，通过解耦的两部分贝叶斯因果森林（Probit BCF建模零值参与、高斯BCF建模正值强度），结合子群体倾向得分调整与Duan非参数平滑估计，有效解决了半连续结果下异质性处理效应估计的难题。 |
| [^298] | [Physics-Guided Multi-Fidelity Surrogate Modeling for Natural Supercavity Geometry of an Axisymmetric Conical Cavitator](https://arxiv.org/abs/2610.18886) | 该论文提出一种物理引导的多保真度代理建模方法，用于预测轴对称锥形空化器的自然超空泡几何形态，在解析模型精度不足与多相CFD计算成本过高之间取得平衡，从而支撑超空泡鱼雷空化器的静态设计与实时形状管理。 |
| [^299] | [Token Utility for On-Policy Distillation via Local Trust-Region Optimization](https://arxiv.org/abs/2610.18883) | 该论文从蒸馏目标本身出发推导出token效用公式 $s_t=g_t^T F_t^\dagger g_t$，证明其在反向KL下等价于师生对数概率比的学生加权方差、在正向KL下等价于Pearson χ²散度，为在线策略蒸馏的token选择提供了统一且有原则的理论基础。 |
| [^300] | [Following the Flow the Law Requires: Training Voice Agents to Transfer to Unseen Regulated Procedures in Debt Collection](https://arxiv.org/abs/2610.18881) | 本文提出 FlowSpec 机器可读规范，通过在 297 个随机化监管程序上微调 90 亿参数模型，训练出能够零样本遵循从未见过的受监管债务催收流程的语音代理。 |
| [^301] | [Do Perturbation Predictors Learn Gene Interactions? Diagnosing Extrapolation in Agentic Screening](https://arxiv.org/abs/2610.18880) | 该论文提出“加性分歧”这一诊断指标来评估扰动预测模型对未见基因组合的外推能力，并发现在八种方法和三个 Perturb-seq 数据集上，专门的神经预测器并不稳定优于简单的加性基线，且偏离加性基线越大的预测误差越大。 |
| [^302] | [Do Vision CIL Methods Transfer to Protein? A Diagnostic Benchmark of Frozen Protein Encoders](https://arxiv.org/abs/2610.18879) | 该论文系统评估了九种视觉领域的类增量学习方法在冻结蛋白质编码器上的表现，发现没有任何方法能可靠地跨编码器迁移，其根源在于梯度稳定性信号弱、嵌入几何异质性大以及协方差假设不成立。 |
| [^303] | [Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data](https://arxiv.org/abs/2610.18878) | 该论文提出一种自蒸馏方法，通过训练一个“逆向推理模型”根据答案反向生成推理轨迹，从而仅利用提示-答案对数据即可高效后训练推理大语言模型，在数据效率与计算效率之间取得平衡。 |
| [^304] | [Lotus: Learning to Understand Time-series by Learning to Forecast](https://arxiv.org/abs/2610.18877) | LOTUS 提出“预测锚定”预训练原则，通过要求时间序列的语言对齐表示能够预测未来片段演化，使其在映射到 LLM 嵌入空间时保留时间与频谱信息，从而学会真正理解时间序列。 |
| [^305] | [RoboSimStudio: An Interactive Toolkit for Human and Agent Rapid Real2Sim of Robot Manipulation Tasks](https://arxiv.org/abs/2610.18835) | 该论文提出交互式工具包RoboSimStudio，通过将求解器、接触模型和物理参数的选择显式化且可编辑，使人类与编程智能体能够基于观察到的物理行为快速构建并迭代优化机器人操作任务的Real2Sim仿真。 |
| [^306] | [Does Pretraining Teach Chemical Language Models Chemistry?](https://arxiv.org/abs/2610.18651) | 化学语言模型的预训练收益无法随规模扩大，性能低于梯度提升基线，仅有基于理化描述符的有监督预训练能持续带来提升，表明预训练几乎未能让模型真正学到化学知识。 |
| [^307] | [Readout Scaling Shapes Sparse Feature Recovery](https://arxiv.org/abs/2610.18462) | 本文提出读出规范化决策空间（RC-DS）坐标方法以消除读出缩放的奇异值增益，并通过受控实验证明当训练恢复因子生成的原始空间时稀疏特征恢复效果最佳，将平均稀疏因子恢复从0.506提升至0.913。 |
| [^308] | [Convergence-Rate-Guided Algorithm Design of Expected Improvement for Continuous Simulation Optimization](https://arxiv.org/abs/2610.18434) | 该论文通过推导简单遗憾的高概率上界，在收敛率分析的基础上系统性地设计了随机克里金期望改进算法的三个关键组件——当前最优解、采样重复次数和终端推荐，解决了未知异方差噪声下连续仿真优化中依赖启发式规则进行算法设计的局限。 |
| [^309] | [ResearchFlow-Eval: Tracing the Execution Gap in End-to-End Multi-Claim Autoresearch](https://arxiv.org/abs/2610.18088) | 该论文提出 ResearchFlow-Eval——一个基于已发表论文锚定、在阶段与论断两个层面细粒度评估端到端多论断自主研究系统持续科学执行与迭代能力的评测框架，弥补了现有评估仅打分孤立任务或最终成果的不足。 |
| [^310] | [Privacy-Preserving GNN Inference on Financial Transaction Graphs via Homomorphic Encryption](https://arxiv.org/abs/2610.18054) | 该论文提出了一种基于CKKS同态加密、以边为中心的图神经网络推理框架，使多家银行能够在不共享明文交易特征的前提下协作进行反洗钱检测。 |
| [^311] | [Interaction-as-a-Judge](https://arxiv.org/abs/2610.17895) | 提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。 |
| [^312] | [RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions](https://arxiv.org/abs/2610.17867) | RT-J是一个8500万参数的关系型基础模型，通过在639个网络爬取数据库和LLM生成的预测任务上利用长上下文预训练，仅需数百个上下文样本即可实现准确预测，所需样本量比现有方法减少10倍。 |
| [^313] | [MMTA: Benchmarking Multimodal Temporal Analysis with Time Series, Text, and Vision](https://arxiv.org/abs/2610.17856) | MMTA是一个涵盖九个领域、融合时间序列、文本与视觉三种模态的多模态时间分析基准测试，首次系统评估了多模态大语言模型能否将时间序列与真实世界视觉证据相整合的能力。 |
| [^314] | [RELAY: Selective Self-Distillation with Adaptive Hand-off for Rubric-Based Reinforcement Learning](https://arxiv.org/abs/2610.17623) | RELAY提出一种两阶段训练方法，第一阶段将冻结教师模型的选择性自蒸馏与GRPO结合，并通过监控蒸馏损失的停滞检测器自适应地在最佳时机移除教师模型、切换到纯GRPO训练，从而在基于评分标准的强化学习中超越纯GRPO方法，且停滞检测器跨模型和领域具有良好泛化性。 |
| [^315] | [Cross-View Time-Frequency Diffusion for Extreme-Aware Time-Series Generation](https://arxiv.org/abs/2610.17407) | 提出CrossTF框架，利用时间视角与频率视角去噪器之间的极端敏感不对称性，在扩散生成后期窗口进行跨视角调整，从而在不损害整体生成质量的前提下大幅提升时间序列中极端事件的保真度。 |
| [^316] | [Learning to Design Convex Relaxations for Nonconvex Optimization](https://arxiv.org/abs/2610.16948) | 本文提出 AutoConvexRelax 方法，将非凸问题的符号公式表示为三元图并学习策略自动组合符号重写以生成针对具体问题的凸松弛轨迹，同时证明了该策略在明确条件下保持松弛有效性并给出可计算的误差界。 |
| [^317] | [DEFlow: Deep Equilibrium Flow Maximum-Entropy Reinforcement Learning](https://arxiv.org/abs/2610.16597) | DEFlow提出了一种深度均衡流策略，通过隐式欧拉积分将所有中间动作作为均衡系统联合求解，并借助隐式微分与离散变量替换公式实现了无需反向传播的离策略训练，在多目标对接、机械臂操作和MuJoCo基准任务上均取得了约10%的平均相对性能提升。 |
| [^318] | [Towards Modality-Agnostic Seizure Classification: A Unified Deep Learning Framework for Heterogeneous EEG and Intracranial EEG Signals](https://arxiv.org/abs/2610.16596) | 该论文提出了一个统一的深度学习框架，能够整合来自不同受试者和来源的异构EEG与iEEG信号进行模态无关的癫痫发作分类，突破了现有单模态系统的应用局限。 |
| [^319] | [Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy-Judge Gap](https://arxiv.org/abs/2610.15778) | 该研究发现强化学习“攻击评委”的深度由评委自身能力决定（与评委AUC呈负相关，系列内更大的评委更难被欺骗），而攻击速度反而与“策略-评委能力差距”的解释相矛盾——差距更大的组合被欺骗得更早，且所有组合均未带来真实的保留集准确率提升。 |
| [^320] | [ProtoPur: Prototype-Purity Pseudo-Labeling for Weakly Supervised Instance Learning in Pathology](https://arxiv.org/abs/2610.15669) | 提出ProtoPur方法，通过基于冻结基础模型特征构建原型区域并估计类别平衡的局部纯度，为高纯度区域分配伪标签，从而实现病理全切片图像的弱监督实例级学习。 |
| [^321] | [SuperAI: Merging Capabilities Across Heterogeneous Foundation Models](https://arxiv.org/abs/2610.15322) | 提出SuperAI跨架构多模型融合框架，利用共享锚点样本提取任务相关表示、PCA构建同维低维功能子空间，并通过正交Procrustes映射实现异构基础模型之间互补能力的融合迁移。 |
| [^322] | [AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery](https://arxiv.org/abs/2610.14959) | AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。 |
| [^323] | [Strong Gaussian Approximation of Constant-Learning-Rate SGD: Bias and Trajectory Inference](https://arxiv.org/abs/2610.14567) | 本文为重尾噪声下的常学习率随机梯度下降建立了KMT型强高斯逼近，实现了有限样本下同时性的整个轨迹层面的推断，并蕴含泛函中心极限定理。 |
| [^324] | [Natural Code Transformation Confuses Agents in Vulnerability Analysis](https://arxiv.org/abs/2610.14350) | 本文发现AI智能体在分析具有相同功能和相同漏洞的不同真实代码提交时会产出截然不同的结果，并据此提出从真实开发模式中挖掘自然且功能保持的代码变换，作为防御攻击者智能体的新手段。 |
| [^325] | [MvFFN: Multi-view Floor plan Feed-forward Network for Unposed Wide-Baseline Panorama Layout Reconstruction](https://arxiv.org/abs/2610.14315) | 提出MvFFN网络，通过让每个全景图列仅局部预测所见墙体及顺时针相邻两堵墙，再由确定性聚合成全局户型图，实现无位姿宽基线全景图中相机位姿与布局的联合重建。 |
| [^326] | [Hide and Seek: HIDE Concepts to SEEK explanations for black box models](https://arxiv.org/abs/2610.13935) | 提出了一种无需模型权重或概念标注数据集的黑盒视觉模型概念级归因方法HnS，通过视觉-语言模型检索概念相关图像、遮挡图像中的概念区域，并依据模型输出的变化来衡量该概念的归因贡献。 |
| [^327] | [AI Models Can Provably Hide Arbitrary Capabilities](https://arxiv.org/abs/2610.13928) | 本文提出一种将加密神经网络嵌入模型权重的后门攻击方法，在密码学假设下可证明地隐藏任意能力，使现有能力评估手段难以引出或解读这些隐藏功能。 |
| [^328] | [Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison](https://arxiv.org/abs/2610.13082) | 该研究基于AMI会议语料库199个配对情境，从五个维度对人类项目经理与Llama 3.1 8B在协作设计会议中的引导行为进行计算比较，发现人类引导在情境语义距离和共情/缓和语词汇密度上更高，且后者主要由缓和语词汇驱动。 |
| [^329] | [Reinforcement Learning via Fixed-point Flow Matching](https://arxiv.org/abs/2610.12848) | 本文提出了一种无需数据的不动点流匹配（FFM）方法，通过将actor更新表述为不动点迭代、以critic的动作梯度作为回归目标，实现了强化学习中流策略的训练与微调。 |
| [^330] | [HRIL: Learning Multimodal Synergy via Higher-Order Tensor Modeling](https://arxiv.org/abs/2610.12393) | 提出HRIL方法，通过在模态嵌入上构建经验交叉矩张量进行高阶张量建模，显式捕获体现为高阶统计依赖的多模态协同信息，从而在自监督多模态表示学习中保留协同信号的信息容量。 |
| [^331] | [Automated Assembly Instruction Generation from CAD Models Using Grounded Large Language Models: A Human-in-the-Loop Framework](https://arxiv.org/abs/2610.11896) | 该论文提出一种人在回路的框架，将CAD装配模型映射为结构化的ProductGraph中间表示，利用大语言模型自动生成受工程信息约束的自然语言装配指令，并通过人工审核解决所有质量标记后才允许导出文档。 |
| [^332] | [Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning](https://arxiv.org/abs/2610.11152) | 该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。 |
| [^333] | [NOMOS: Compiling Written Policies into Statically Verified Tool-Call Gates for LLM Agents](https://arxiv.org/abs/2610.11030) | NOMOS是一个四遍编译器，可将自然语言策略编译为经过静态验证的确定性工具调用门控，将LLM代理在状态变更调用中的策略违规率从66.3%降至2.6%。 |
| [^334] | [Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an RBF Kernel](https://arxiv.org/abs/2610.10080) | 本文针对基于RBF核的高斯过程，在固定域渐近框架下建立了极大似然估计的精确渐近理论，攻克了密集采样强相关性与协方差矩阵非线性依赖带来的长期理论难题。 |
| [^335] | [Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment](https://arxiv.org/abs/2610.09369) | 提出互不相溶扩散策略，一种无需标签的训练时噪声分配方法，通过保持噪声到动作路径的相对分离，防止扩散策略在低维机器人动作空间中坍缩为单一模态，从而保留多模态动作分布。 |
| [^336] | [Steering Diffusion Models to Rare Events with Sequential Monte Carlo](https://arxiv.org/abs/2610.08652) | 本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。 |
| [^337] | [ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding](https://arxiv.org/abs/2610.08078) | 该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。 |
| [^338] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^339] | [Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes](https://arxiv.org/abs/2610.07853) | 该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。 |
| [^340] | [EMG-FM-Bench: A Comprehensive Benchmark for Foundation Model Transfer and Adaptation on Electromyography](https://arxiv.org/abs/2610.06450) | 该论文提出了EMG-FM-Bench，一个整合20个公开数据集（超100万个EMG片段）、系统评估九个预训练基础模型在肌电信号上迁移与适配能力的综合基准，全面回答了冻结与微调、预训练收益、少样本新用户泛化及跨任务性能等关键问题。 |
| [^341] | [Langevin Flow Maps: Efficient Molecular Dynamics and Transition Path Sampling](https://arxiv.org/abs/2610.05998) | 提出朗之万流映射，通过扩展机器学习力场以额外学习随机朗之万积分器，实现大时间步长的高效分子动力学模拟，在准确恢复系统动力学性质的同时速度提升一个数量级，并展示了向可迁移模型发展的路径。 |
| [^342] | [From Mixing to Tearing: Graph Decomposition in Decentralized Optimization via Message Passing](https://arxiv.org/abs/2610.03709) | 该论文从第一性原理出发提出了一种基于图分解的去中心化优化通用框架，通过消息传递联合设计优化子问题、对偶变量分块和协作求解的智能体集群，突破了传统仅利用网络混合信息的范式。 |
| [^343] | [The Neuro-Physical Inverter: A Modular Framework for Magnetotelluric Inversion Coupling Ensemble Conditioning with Residual Learning](https://arxiv.org/abs/2610.03225) | 提出了一种模块化的神经物理反演器（NPI）框架，通过将集成条件高斯过程与残差学习神经网络相结合，实现了具备不确定性量化能力的大地电磁反演，在不破坏集成稳定性的情况下系统性降低误差。 |
| [^344] | [Predictively Oriented Gaussian Process Posteriors](https://arxiv.org/abs/2610.03201) | 提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。 |
| [^345] | [Efficient Memory Crystallization for Graph Learning under Non-Stationary Distribution Shifts](https://arxiv.org/abs/2610.02795) | 该论文提出了一种无需训练的测试时框架EMC，通过闭式解将每个新到来的图域高效地“结晶”为紧凑且语义忠实的记忆，以低成本替代生成式记忆方案，从而应对图学习中的持续分布偏移。 |
| [^346] | [Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination](https://arxiv.org/abs/2610.02626) | 提出IG-VLA框架，通过在视觉表征空间中直接想象任务相关的未来场景演变，并将其内化为紧凑的场景要旨标记，使VLA模型在推理时无需昂贵的显式未来推演即可获得未来推理带来的动作预测增益，实现高效的机器人操作。 |
| [^347] | [Participation-Sensitive Convergence and the Fragment First, Converge Later Pattern in Asynchronous Online Learning: A Topological Analysis Across 22 OULAD Courses](https://arxiv.org/abs/2610.01738) | 该研究基于22门OULAD课程（超22,000名学习者）的拓扑分析发现，Zigzag持续同调的β0是参与敏感指标，它与活跃学习者人数共同响应截止日期事件，且截止日期在82.6%的评估中引发“先分裂、后收敛”的群体行为模式。 |
| [^348] | [The Teacher Is a Direction, Not a Destination: Extrapolating RL-Induced Representation Residuals in On-Policy Distillation](https://arxiv.org/abs/2609.36484) | 提出RIDE方法，通过在每一层外推强化学习引起的内部表征偏移方向（而非在输出空间外推隐式奖励），克服语言模型头的各向异性衰减与噪声放大问题，使学生模型在在线策略蒸馏中稳定地超越教师。 |
| [^349] | [NeuronSifter: Intervention Planning in CNS Microenvironments](https://arxiv.org/abs/2609.35445) | NeuronSifter 的核心创新在于将决策质量构建于干预接口本身：它把给药方案编译为状态条件下的靶点占据场，用占据条件扩散算子在微环境动力学中传播，并依据干预损失的期望下降来选择测量手段，同时将不同类型的观测结果同化到同一后验分布中。 |
| [^350] | [NeuronDiscover: Agent-in-Twin for Mechanistic Discovery in Neuronal Microenvironments with World Action Models](https://arxiv.org/abs/2609.35338) | NeuronDiscover 提出“孪生体中智能体”框架，以机制为基础的世界动作模型统一耦合预测、干预提议与观测设计，并对“孪生混淆”进行形式化、对机制—偏差联合信念进行实验设计，从而在稀疏观测下区分真实机制变化与计算孪生体误差，实现神经元微环境的自动化机制发现。 |
| [^351] | [Deep kernel hedging](https://arxiv.org/abs/2609.34474) | 该论文提出了一种深度核对冲框架，通过神经网络参数化核函数并结合广义表示定理与随机傅里叶特征近似，实现了可处理路径依赖信息的可扩展、高效金融对冲。 |
| [^352] | [The Statistical Benefits of Multiple Responses for Learning from Demonstrations](https://arxiv.org/abs/2609.33291) | 当示范者并非最优时，返回多个候选响应（pass@$k$）能带来更强的统计优势——将最坏情况下对目标精度的依赖从 $1/\varepsilon^2$ 改善为 $1/\varepsilon$，并将对奖励类大小的依赖从 $\log N$ 改善为 $\log N/\log k$。 |
| [^353] | [PrismQuant: Optimal Null-Space Rotations for Grouped Quantizers](https://arxiv.org/abs/2609.32429) | PrismQuant提出了一种量化器感知的旋转框架，通过将旋转设计表述为Ky Fan迹最大化并推导出可证明最优的闭式解，将激活主特征空间与非对称分组INT4量化的常数组子空间精确对齐，从而在不扩大组内数值范围的前提下提升低比特量化效果。 |
| [^354] | [TimeES: Probabilistic and Deterministic Time Series Forecasting via Evolutionary Spectra](https://arxiv.org/abs/2609.32384) | 提出TimeES框架，基于演化谱理论显式建模傅里叶谱中的演变与随机性，将非平稳随机过程建模转化为学习由随机变量调制的演化表示，并利用厄米对称性将谱估计复杂度从O(NM)降至O(NK)，从而实现对非平稳时间序列的概率性与确定性预测。 |
| [^355] | [DiffPTS: Rethinking Diffusion ELBO for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2609.32363) | 本文通过在位置-尺度噪声模型下重新推导扩散模型的证据下界（ELBO），提出了 DiffPTS，将估计器的训练与变分推断统一为一个自然的联合训练目标，从而改进概率时间序列预测。 |
| [^356] | [Learning Provable Neural Network Observer for Uncertain Dynamical Systems](https://arxiv.org/abs/2609.30819) | 提出了一种两阶段训练框架（点引导李雅普诺夫预训练加LMI微调），为不确定动态系统的神经网络观测器实现了可证明的全局李雅普诺夫稳定性，同时克服了传统LMI方法带来的大规模半定规划可扩展性瓶颈。 |
| [^357] | [A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data](https://arxiv.org/abs/2609.29584) | 该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。 |
| [^358] | [Memory Attention](https://arxiv.org/abs/2609.28399) | 提出记忆注意力，用可复用的词元记忆与上下文键结合构建注意力值，推理时可简化为查表加法并支持CPU卸载，在相同训练预算下提升了语言建模和下游任务性能。 |
| [^359] | [hyperbolix: Hyperbolic Deep Learning in JAX](https://arxiv.org/abs/2609.28248) | 该论文提出了 hyperbolix，首个基于 JAX 与 Flax NNX 构建的全面、通用的开源双曲深度学习库，提供六种统一接口的流形、覆盖各类双曲神经网络层族以及黎曼优化器和降维工具。 |
| [^360] | [EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues](https://arxiv.org/abs/2609.27043) | 提出了EduBehaviors框架，利用大语言模型测量教育对话中重复出现的可观察行为并据此学习分类器，实现了可解释、可审计的教育对话标注，其性能与直接提示方法相当。 |
| [^361] | [Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism](https://arxiv.org/abs/2609.18381) | 该论文提出通过训练判别器来生成类似散度的真实性评分，能够自适应地检测天气预报模型表现出的任何失败模式，从而克服固定指标存在的盲区。 |
| [^362] | [A Unified and Constrained View of Regularization-Based Robust Reinforcement Learning](https://arxiv.org/abs/2609.13050) | 该论文通过推导性能差距上界统一了基于正则化的鲁棒强化学习方法，将鲁棒训练表述为约束优化问题，并通过联合更新拉格朗日乘子自动调整正则化权重以提升鲁棒性。 |
| [^363] | [Relatively Smart II: Tractable or Semi-Supervised Instance-Optimal Learning](https://arxiv.org/abs/2609.10886) | 本文证明ERM乃至任何正规一致学习器在无分布二分类中都是相对智能的，从而解决了此前留下的开放问题，即自然的可处理学习器能否达到与单包含图学习器可比的保证。 |
| [^364] | [Multi-Level-Set-Based Physics-Driven Neural Network to Solve 3-D Inverse Scattering Problems](https://arxiv.org/abs/2609.08594) | 本文提出基于多水平集的物理驱动神经网络（LSPDNN），利用软并集多材料模型将目标支撑域与材料分布分离描述，并结合作用于材料区域指示器的模型一致TV正则化，有效缓解了三维电磁逆散射重建中的边界模糊与伪影问题。 |
| [^365] | [Rethinking One-Shot Federated Graph Learning: Training-Free Statistical Estimation](https://arxiv.org/abs/2609.06154) | 该论文提出免训练框架SPEAR，将单轮联邦图学习重新表述为统计估计问题，通过计算拓扑平滑的类别原型并使用样本量自适应的收缩估计器进行聚合，有效解决了极端非独立同分布条件下本地GNN训练导致的跨客户端表示不对齐问题。 |
| [^366] | [When Genomic Masking Priors Fail to Transfer: Strong Variant Prediction, Weak Functional Generation](https://arxiv.org/abs/2609.04861) | GenDA双向扩散模型在变异效应预测上虽超越同规模自回归模型，但熵引导策略对性能提升并无实际贡献，且模型在零样本功能序列填充生成任务上失败，表明基因组掩码先验能很好地迁移到变异预测、却难以迁移到功能生成。 |
| [^367] | [Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning](https://arxiv.org/abs/2609.01449) | 该论文将扩散去噪器改造为带持久隐藏状态且无时间步条件的通用迭代更新，构建出可随推理深度不断提升准确率的随时求解器，并通过推理时持续注入高斯噪声实现单轨迹高效探索解空间，在极限数独上达到99.90%的精确求解率。 |
| [^368] | [LLMs Can Design Near-Optimal OR Algorithms](https://arxiv.org/abs/2608.27296) | 大语言模型（如gpt-5.6-sol）能够仅凭少量提示和固定计算资源，自动设计出在库存控制、排队网络和分类优化等问题上匹配或超越现有最佳方法的近乎最优运筹学算法。 |
| [^369] | [Mechanistic Circuit Identification for Controllable Data Generation](https://arxiv.org/abs/2608.24065) | 本文提出了一种基于电路识别的框架，通过连接数据估值与机制可解释性，实现可控数据生成，并引入SAMS调度器以优化效用目标数据产出。 |
| [^370] | [Physics-Integrated Operator Learning via Gaussian Splatting Representations](https://arxiv.org/abs/2608.24049) | 本文提出一种表示级物理集成方法，通过前馈高斯喷溅表示将连续高斯场与闭式导数结合，直接嵌入PDE算子于学习演化中，避免优化困难并提升长期预测精度。 |
| [^371] | [Detecting and Discriminating Operator Misspecification in Hybrid PDE-Parameter Learning: a Reference-Free Instrument, with Discrimination Bounded In Sample](https://arxiv.org/abs/2608.16925) | 本文提出一种无需参考信息的统计工具，通过单次拟合即可区分混合PDE参数学习中的算子误设定与参数不可识别问题，并证明了样本内判别能力有界。 |
| [^372] | [Equilibrium Forcing: Adaptive Video Generation Without Noise Conditioning](https://arxiv.org/abs/2608.14706) | 本文提出平衡力迫（EqF）框架，通过去除噪声水平条件化并解耦去噪与采样，实现了自适应闭环推理，显著提升了自回归视频生成的质量和一致性。 |
| [^373] | [Uncertainty-Aware Probabilistic Constrained Clustering from Entangled Pairwise Supervision](https://arxiv.org/abs/2608.12027) | 本文提出了一种不确定性感知的概率约束聚类方法，通过角度成对目标和估计-校正-集成框架，有效处理现实中的软性、纠缠成对监督，提高了聚类性能。 |
| [^374] | [Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL](https://arxiv.org/abs/2608.11669) | 本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。 |
| [^375] | [FACT: Failure-Aware Causal Training for World-Action Models](https://arxiv.org/abs/2608.10232) | FACT提出了一种失败感知的因果训练方法，让世界-动作模型以执行动作为条件预测未来视频和任务进度，从而利用失败轨迹监督动作后果，将失败数据转化为有效的训练信号而非被丢弃。 |
| [^376] | [Comparing SGLD and a fixed-noise predictor-corrector adaptation in canonical joint energy-based models on CIFAR-10](https://arxiv.org/abs/2608.05025) | 该论文系统复现了经典联合能量模型（JEM），并首次检验了用固定噪声预测-校正采样器替代SGLD的效果，结果显示其分类性能接近经典方法但生成质量略有差距，同时记录了两种失败模式。 |
| [^377] | [SPARK: Learning from Verification in Speculative GRPO Rollouts](https://arxiv.org/abs/2608.04962) | 提出SPARK，一种轻量级自适应机制，利用目标模型的验证反馈在推测性GRPO训练的采样过程中实时修正草稿模型的预测遗漏，从而提升训练效率。 |
| [^378] | [Surrogate Substitution Preserves PHI Detectability: A Multi-Detector Equivalence Study](https://arxiv.org/abs/2608.03172) | 本文提出了一种成对多检测器评估协议，通过等价性检验和仅在掩盖跨度上评分，验证结构保留的PHI替代物能否被下游检测器可靠识别，并区分生成器缺陷与检测器固有局限。 |
| [^379] | [QASP: Query-Adaptive Robust Vector Search Policy](https://arxiv.org/abs/2607.29606) | QASP通过单次预先监督回归预测每个查询的完整召回率进展曲线，从而为任意召回率目标推导出自适应搜索策略，在跨目标、索引配置和数据集泛化的同时兼顾高召回率与低计算成本。 |
| [^380] | [The Zero Pattern of a Design Matrix Drives Multiple Descent in Over-parameterized Regression](https://arxiv.org/abs/2607.24041) | 本文放宽了过参数化线性回归中协变量独立且协方差矩阵非退化的假设，证明协方差退化与变量相关性会引发多重下降现象，并借助二部图的最大匹配和Dulmage-Mendelsohn分解精确定位了风险峰值出现的位置。 |
| [^381] | [Hopformer: Homogeneity-Pursuit Transformer for Time Series Forecasting](https://arxiv.org/abs/2607.22299) | Hopformer提出两阶段时间序列预测框架，先通过稀疏模式聚合（SPA）提取融合协变量的共同低方差趋势作为同质化层，再用LoRA微调的Transformer建模残差中的复杂依赖，在理论上证明了近最优偏差-方差权衡与泛化界，在多个基准上平均提升MASE 6.56%。 |
| [^382] | [Inpainting Insights: Elevating Visual XAI with Photorealistic Perturbations](https://arxiv.org/abs/2607.15482) | 该论文改进了广泛使用的基于扰动的方法LIME，利用生成式图像修复技术生成照片级逼真的扰动样本，解决了传统扰动方法产生不真实分布外样本和可见伪影的问题，从而提升视觉XAI的图像解释质量。 |
| [^383] | [Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs](https://arxiv.org/abs/2607.13609) | 该论文提出用规范不变的图狄利克雷能量替代岭正则化，从有向图流数据中恢复势能，解决了岭正则化导致的排序反转问题，并实现了跨λ四个数量级的参数不敏感性。 |
| [^384] | [Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains](https://arxiv.org/abs/2607.13319) | OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。 |
| [^385] | [AdaPCLA: Adaptive Prior-Calibrated Logit Adjustment for Long-Tailed Longitudinal EHR Generation](https://arxiv.org/abs/2607.12645) | 提出AdaPCLA框架，通过数据分布感知的模拟退火训练将先验知识内化到生成模型中，并结合零样本分布控制实现无需训练即可适应多样化临床人群的长尾纵向电子健康记录生成。 |
| [^386] | [tidyHEBO: Robust General-Purpose Bayesian Optimization with Model-Consistent Warping and Pareto Search](https://arxiv.org/abs/2607.10669) | tidyHEBO 是一个 BoTorch 原生的通用贝叶斯优化器，通过将 Yeo-Johnson 输出变换与高斯过程代理模型联合拟合、在原始目标尺度上评估采集函数并执行多准则累积帕累托搜索，在无需任何基准特定调优的情况下于 Olympus 基准上排名第一。 |
| [^387] | [Does Depth Survive Composition, and Can a Router Use It? Adaptive Compute in Latent World Models](https://arxiv.org/abs/2607.10203) | 该论文通过在九个 DeepMind Control 任务上测试十二模块早退潜在世界模型发现，“更深预测器更准确”的假设在部分任务上不成立且多步复合会改变深度-误差关系，尽管逐步选择深度的贪婪 oracle 可将 rollout 误差降低 10-37%，但训练出的所有路由器均无法捕捉这种深度选择收益。 |
| [^388] | [Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing](https://arxiv.org/abs/2607.09015) | 该论文提出了面向大语言模型路由的相关性感知上下文多臂老虎机算法，通过耦合奖励混合与解耦预测混合两种互补设计利用可能不可靠的代理奖励信号，在加速学习的同时对代理信号错误设定保持鲁棒性。 |
| [^389] | [RhyMix: A Lightweight Adaptive Multi-Rhythm Network for Long-Term Time Series Forecasting](https://arxiv.org/abs/2607.08234) | RhyMix提出了一种轻量级的并行双路径混合神经架构，通过带可学习循环嵌入的循环路径与自适应门控机制，同时捕获时间序列中的季节周期、短期波动和长期趋势等多种模式，从而提升长期时间序列预测性能。 |
| [^390] | [CausalDS: Benchmarking Causal Reasoning in Data-Science Agents](https://arxiv.org/abs/2607.08093) | 该论文提出CausalDS基准，通过系统性地生成结构因果模型、对应的观测数据以及基于真实领域的自然语言场景，填补了现有基准在评估数据科学智能体因果推理能力方面的空白。 |
| [^391] | [What Does a Discrete Diffusion Model Learn? One Reverse Process in Three Coordinates](https://arxiv.org/abs/2607.05381) | 本文证明离散扩散模型中的去噪器、桥接插件和具体分数三种方法学习的其实是同一对象——加噪过程的精确时间反演，只是处于三种不同坐标系，并给出 Oracle Distance 恒等式（负 ELBO 等于数据熵加路径 KL 散度）及三种坐标间的精确转换。 |
| [^392] | [Platonic Projection Structures: Operator-Induced Observability in Representation Learning](https://arxiv.org/abs/2607.05175) | 该论文提出柏拉图投影结构（PPS）这一算子理论框架，通过自伴半正定观测算子与商几何 $H/\ker(\Pi)$ 来刻画部分观测下表示学习的可观测性，并揭示量子测量与线性观测下的表示推断共享同一算子理论结构。 |
| [^393] | [LIME: Learning Intent-aware Camera Motion from Egocentric Video](https://arxiv.org/abs/2607.02417) | 该论文首次将视觉-语言条件下的相机运动生成形式化为独立任务，并提出通过事后标注从自我中心视频中挖掘多意图监督数据的方法，使机器人能够根据自然语言意图预测相机运动。 |
| [^394] | [Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery](https://arxiv.org/abs/2606.29403) | 提出自组织保形预测（SOCP），利用无需校准标签的无监督自组织映射（SOM）发现输入空间分组并从中提取校准缓冲，在预测器和非一致性分数保持不变的前提下实现精确的条件覆盖有效性，从而减少特征空间异质区域的覆盖差距。 |
| [^395] | [Blackknife: Query-Free Black-Box Rewiring Attacks on Heterogeneous Graph Neural Networks](https://arxiv.org/abs/2606.29240) | Blackknife提出了一种免查询的严格黑盒攻击方法，仅通过观测一跳结构和保持度数不变的边重连操作，即可有效规避异构图神经网络的防御。 |
| [^396] | [From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations](https://arxiv.org/abs/2606.26277) | 本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。 |
| [^397] | [Geological text descriptions in ill-posed inverse problems: insights from learned hydraulic-conductivity inversion](https://arxiv.org/abs/2606.24967) | 该研究通过合成达西流基准系统考察了地质文本描述在学习型渗透系数反演中的作用，发现当观测存在结构歧义时描述能改善重建，实例特定内容尤为有效，且求解器会依据描述所述的几何信息偏移重建结果，但随着观测信息量增大，描述的影响会减弱。 |
| [^398] | [X-Safe: Embodiment-Agnostic Safeguarding for Robot Arm Manipulation Policies via Action Masking](https://arxiv.org/abs/2606.22278) | X-Safe提出了一种与机器人本体无关的安全保障方法，通过在构型空间中对当前构型进行局部推理建立安全区域，并利用动作掩码将策略动作限制在安全区域内，从而在无需模型简化或学习型安全函数的情况下，为机械臂操作策略提供形式化的防碰撞概率保证。 |
| [^399] | [PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning](https://arxiv.org/abs/2606.21139) | PoLAR 通过在潜在动作空间中引入极坐标径向结构（半径编码动作幅度、方向编码动作模式），并借助双曲空间与观测时间偏移作为约束，使机器人策略更容易将潜在动作解码为实际的机器人动作。 |
| [^400] | [HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents](https://arxiv.org/abs/2606.16285) | HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。 |
| [^401] | [Driving, Fast or Slow? Neuro-Symbolic Guidance for Motion Prediction in Multi-Modal Ground Mobility](https://arxiv.org/abs/2606.15251) | 提出了TraCS神经符号框架，通过代理式代码生成将自然语言交通法规转化为可解释的概率一阶逻辑约束，并利用响应式数据流推理引擎实时更新合规性图景，从而增强黑盒运动预测模型在异构交通环境中的可解释性与安全性。 |
| [^402] | [A Unifying Framework for Concept-Based Representational Similarity](https://arxiv.org/abs/2606.09653) | 提出了一个基于概念的表征相似性统一框架，沿“对齐对象”与“对齐层级”两个轴分解concept对齐问题，并引入基于干预的基准InterVenchA，分别测量提取质量、translation质量与concept一致性，且与人类感知进行对照验证。 |
| [^403] | [AgentWorld: A Data Factory for Training LLM Agents from Multi-Turn Interactions](https://arxiv.org/abs/2606.09138) | AgentWorld是一个数据工厂，通过API代理网关捕获异构智能体运行时的多轮交互，并将其转化为标准化、高质量的结构化训练数据，为LLM智能体训练提供从交互收集到训练消费的统一数据生命周期。 |
| [^404] | [Activation Steering Induces Emergent Misalignment: Steering Vector Construction and Comprehensive Evaluation](https://arxiv.org/abs/2606.08682) | 该论文通过系统构建引导向量并综合评估，首次证实激活引导技术可以从狭窄任务出发，在大语言模型中诱发跨广泛无关领域的涌现性失调，揭示了这一流行推理时调节技术的潜在安全风险。 |
| [^405] | [HARP: Efficient Data Selection for Finetuning Large Language Models](https://arxiv.org/abs/2606.07690) | 提出了一种基于训练的高效数据选择方法HARP，它将训练数据组织为层次结构、仅评估代表性子集并用经验贝叶斯后验推断其余数据的效用，配合互补的冗余控制与加性选择策略，在大幅降低选择成本的同时保持下游任务对齐效果。 |
| [^406] | [STREAM: Stochastic Riemannian Flow Matching with Anisotropic Decoder for Digital Histopathology Image Generation](https://arxiv.org/abs/2606.07036) | 提出STREAM框架，首次在数字病理组织学图像生成中应用黎曼流匹配，直接以病理学视觉基础模型patch token所在的单位超球面作为潜在空间并配合各向异性解码器，克服了现有方法中条件信号主导多样性、导致生成多样性不足的问题。 |
| [^407] | [Memory by Design: Probabilistic Sequence Layers](https://arxiv.org/abs/2605.31163) | 本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。 |
| [^408] | [Subspace-Decomposed JEPAs: Disentangling Progression and Content in Latent World Models](https://arxiv.org/abs/2605.31111) | SD-JEPA将JEPA潜在空间分解为正交的进展子空间和内容子空间，使两种防坍缩力量在不相交的坐标上加性组合而非相互竞争，从而在多个控制基准上提升了世界模型性能。 |
| [^409] | [Unification and Optimization of Robust Supervised Learning](https://arxiv.org/abs/2605.28165) | 该论文提出了一个统一框架，将分布鲁棒优化、Mixup、标签平滑等鲁棒学习方法纳入同一表述体系，明确了各方法间的冗余或互补关系，并推导出将设计选择转化为可调超参数的实用训练过程。 |
| [^410] | [A Simple Plug-in for Improving Eviction-Based KV Cache Compression](https://arxiv.org/abs/2605.23258) | VECTOR是一种即插即用插件，通过引入“保留-近似-逐出”三路token路由机制，并结合离线校准的可重构性信号，恢复了原本会被二元逐出不可逆丢失的值信息，从而显著提升基于逐出的KV缓存压缩方法在长上下文推理中的性能。 |
| [^411] | [A finite-element-inspired bipartite graph learned simulator for manufacturability assessment in large-deformation sheet forming](https://arxiv.org/abs/2605.22845) | 该论文提出CAtt-BiGNN，一种基于交叉注意力的二部图神经网络，通过模拟显式有限元中节点运动学与单元变形的耦合更新机制，实现大变形板料成形可制造性的高效评估。 |
| [^412] | [OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond](https://arxiv.org/abs/2605.19660) | 该论文识别出Token范数不平衡（TNI）是极端压缩下KV缓存逐通道量化保真度的主要瓶颈，并提出了一种精确且轻量的量化方法OScaR，无需复杂的量化流水线即可实现高保真KV缓存压缩。 |
| [^413] | [MOCHA: Multi-Objective Chebyshev Annealing for Agent Skill Optimization](https://arxiv.org/abs/2605.19330) | 提出 MOCHA 方法，通过切比雪夫标量化选择改进对象并以扩展帕累托前沿为接受准则，从探索退火到利用，实现智能体技能在准确性与上下文经济性之间的多目标优化，克服了现有单分数提示词优化器易停滞的问题。 |
| [^414] | [BrainDyn: A Sheaf Neural ODE for Forecasting and Generating Brain Dynamics](https://arxiv.org/abs/2605.19324) | BrainDyn 提出了一种基于胞腔层束的图神经 ODE 网络，通过根据预测上下文窗口动态重算的限制映射来建模脑区之间异质的通信方式，从而实现对大脑动力学的预测与生成。 |
| [^415] | [How Far Are We From True Auto-Research?](https://arxiv.org/abs/2605.19156) | 该研究提出ResearchArena评测框架，让现成智能体在轻量指导下自主完成从构思到写作的完整科研闭环，并通过仅稿件审稿、感知产物同行评审和人类元评审三种互补视角，系统评估了117篇智能体生成论文的真实质量，发现仅看稿件时结果乐观，凸显了更全面评估自动化研究成果的必要性。 |
| [^416] | [Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers](https://arxiv.org/abs/2605.18106) | 该论文提出优化器设计的对称兼容原则——梯度更新规则应在权重块的对称群下保持等变，据此统一了双正交等变更新方法，并为嵌入层、语言模型输出头、SwiGLU MLP和MoE路由器等结构推导出对称兼容的优化器。 |
| [^417] | [When and Why Adversarial Training Improves PINNs: A Neural Tangent Kernel Perspective](https://arxiv.org/abs/2605.15959) | 本文基于神经正切核（NTK）视角提出了一个新的分析框架，揭示了GAN判别器如何影响PINNs训练动力学，从而解释了对抗训练何时及为何有效，并据此提出了一种实用高效的PINNs训练算法。 |
| [^418] | [Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting](https://arxiv.org/abs/2605.15470) | Njord是一个将深度潜变量框架与图神经网络相结合的概率性海洋预报模型，通过自适应不规则海面几何的K-means聚类网格实现全球0.25°和波罗的海2公里分辨率的预报，在单次前向传播中即可采样集合预报，性能超越确定性机器学习基线并提供不确定性估计。 |
| [^419] | [GQLA: Group-Query Latent Attention for Hardware-Adaptive Decoding](https://arxiv.org/abs/2605.15250) | 该论文提出分组查询潜在注意力（GQLA），通过组共享投影使同一权重即可支持MQA吸收式与GQA两种解码路径，从而根据GPU计算带宽比和张量并行度自适应选择最优推理方式，并能零梯度转换现有MLA/GQA模型检查点。 |
| [^420] | [Learning, Fast and Slow: Towards LLMs That Adapt Continually](https://arxiv.org/abs/2605.12484) | 受人类多时间尺度学习机制启发，提出了一种快-慢学习框架，以模型参数作为“慢”权重、优化上下文作为“快”权重，使大语言模型既能通过文本反馈快速适应特定任务，又能避免灾难性遗忘并保持通用推理能力。 |
| [^421] | [A proximal gradient algorithm for composite log-concave sampling](https://arxiv.org/abs/2605.12461) | 提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。 |
| [^422] | [Split the Differences, Pool the Rest: Provably Efficient Multi-Objective Imitation](https://arxiv.org/abs/2605.12000) | 提出多输出增强行为克隆（MA-BC）算法，通过划分冲突专家数据并汇集无冲突的状态-动作对，以更快的统计速率收敛到帕累托最优策略，并达到极小极大最优。 |
| [^423] | [LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection](https://arxiv.org/abs/2605.10980) | 提出LEAP，一种无需训练、即插即用的方法，通过前瞻性检测在去噪早期已收敛的token，突破现有置信度阈值的保守限制，从而显著加速扩散语言模型的并行解码。 |
| [^424] | [TIDES: Implicit Time-Awareness in Selective State Space Models](https://arxiv.org/abs/2605.09742) | TIDES 提出了一种选择性状态空间模型变体，通过将输入依赖性从离散化步长转移到对角状态矩阵，在原生处理不规则时间戳的同时保留了逐词符表达能力。 |
| [^425] | [TabFORGE: A Tabular Foundation Model for Generative Modelling](https://arxiv.org/abs/2605.09424) | 提出了TabFORGE——一个基于预训练表格基础生成表示的新型表格基础模型，通过更好地对齐表格数据的特征间依赖结构，显著提升生成式表格建模的合成数据质量。 |
| [^426] | [Generative Actor-Critic with Soft Bridge Policies](https://arxiv.org/abs/2605.08733) | 提出SoftGAC软生成式行动者-评论家算法，利用软桥接策略在单次网络评估内生成动作并提供可处理的软目标，同时解决了生成式策略训练中边际动作密度不可得和迭代采样推理与内存开销大的两大障碍。 |
| [^427] | [Attention Transfer Is Not Universally Effective for Vision Transformers](https://arxiv.org/abs/2605.07191) | 本研究通过对来自11个ViT家族的20个教师模型进行系统基准测试，发现注意力迁移并非普遍有效——其中4个ViT家族的注意力迁移持续失败且性能低于从零训练的基线，且该失败在不同模型规模、训练时长、数据集和分布外评估下均保持一致，其根源被定位在注意力路由通道。 |
| [^428] | [From Causal Graphs to Markovian States: Multi-Order State Exposure for Deep RL](https://arxiv.org/abs/2605.07057) | 提出MOSE方法，将单个共享Q函数同时暴露于1到W阶基于历史的状态构造，解决了如何从纵向因果图构造可证明满足马尔可夫性质的最小状态这一根本问题。 |
| [^429] | [Medical Imaging Classification with Cold-Atom Reservoir Computing using Auto-Encoders and Surrogate-Driven Training](https://arxiv.org/abs/2605.06727) | 该论文提出了一种结合引导式自编码器与可微分代理模型的冷原子储备池计算混合框架，克服了量子测量不可微带来的梯度障碍，实现了医学影像（息肉检测）分类任务的端到端训练。 |
| [^430] | [Topology-Aware Representation Alignment for Semi-Supervised Vision-Language Learning](https://arxiv.org/abs/2604.26370) | 提出ToMA框架，利用持续同调将图文成对关系分解为H0-消亡与H1-诞生两类关系，并通过跨模态对应在模态间对齐，从而捕捉多模态表示流形的全局拓扑结构，提升半监督视觉-语言学习的泛化能力。 |
| [^431] | [Dictionary learning for Kernel EDMD](https://arxiv.org/abs/2604.25572) | 本文将 EDMD 中的字典学习方法扩展到 kEDMD 中的核学习，实现了核函数及其参数的自动学习，从而简化了 kEDMD 中核函数定义与参数选择的繁琐过程。 |
| [^432] | [Ideological Bias in LLMs' Economic Causal Reasoning](https://arxiv.org/abs/2604.21334) | 该研究在EconCausal基准中引入亲政府与亲市场观点存在分歧的意识形态争议案例，发现大语言模型在经济因果推理中表现出系统性意识形态偏见，且争议性案例的因果方向预测难度显著更高。 |
| [^433] | [Causally-Constrained Probabilistic Forecasting for Time-Series Anomaly Detection](https://arxiv.org/abs/2604.17998) | 提出了一种因果引导Transformer（CGT）模型，通过将时滞因果图先验融入深度概率预测框架，实现了多变量时间序列异常检测中的因果解释与根因定位能力。 |
| [^434] | [Scaling Human-AI Coding Collaboration Requires a Governable Consensus Layer](https://arxiv.org/abs/2604.17883) | 该论文提出扩展人机编码协作需要引入一个可治理的共识层，通过版本化的项目状态模型将意图、结构性承诺、代码工件和证据关联起来，使人类能够在AI编码过程中检查、协调和修订共享的设计决策。 |
| [^435] | [StaRPO: Stability-Augmented Reinforcement Policy Optimization](https://arxiv.org/abs/2604.08905) | 提出StaRPO框架，通过自相关函数（ACF）和路径效率（PE）两个轻量级指标将推理稳定性显式纳入强化学习优化目标，从而改善大语言模型推理过程的逻辑一致性与结构稳定性。 |
| [^436] | [Structural Pass Analysis in Football: Learning Pass Archetypes and Tactical Impact from Spatio-Temporal Tracking Data](https://arxiv.org/abs/2603.28916) | 该论文提出了由线突破得分（LBS）、空间增益度量（SGM）和结构破坏指数（SDI）组合而成的可解释战术影响值（TIV）框架，利用时空追踪数据量化足球传球对对方防守结构的突破与破坏效果。 |
| [^437] | [Constitutive parameterized deep energy method for solid mechanics problems with random material parameters](https://arxiv.org/abs/2603.26030) | 提出本构参数化深度能量方法（CPDEM），这是一种纯物理驱动的框架，可高效求解具有随机材料参数的固体力学问题，避免了有限元方法重复网格划分的高昂计算成本、数据驱动代理模型对大规模数据集的依赖，以及标准深度能量方法在参数变化时需完全重新训练的缺陷。 |
| [^438] | [Sharp Statistical Thresholds for Attributed Gaussian Graph Matching](https://arxiv.org/abs/2603.23305) | 本文针对边权重与节点特征联合相关的属性化高斯图匹配问题，推导出了精确恢复、几乎精确恢复和部分恢复的精确信息论阈值，并发现节点特征信息的引入使得精确恢复阈值从共享阈值中分离，呈现出比标准图匹配更丰富的相变结构。 |
| [^439] | [Maximum-Entropy Exploration with Future State-Action Visitation Measures](https://arxiv.org/abs/2603.18965) | 该论文提出了一种与未来状态-动作特征折扣分布的熵成比例的内在奖励方法，理论上证明其期望总和是从初始状态出发的轨迹所访问分布熵的下界且可通过收缩算子的不动点性质进行离策略估计，从而实现了更优的最大熵探索。 |
| [^440] | [Edgewise Envelopes Between Balanced Forman and Ollivier-Ricci Curvature](https://arxiv.org/abs/2603.13535) | 该论文通过推导平衡Forman曲率与Ollivier-Ricci曲率之间的显式双侧分段仿射界，将图上OR曲率的逐边计算从最优传输线性规划降至最坏情况O(最大度^2.5)时间，大幅提升大规模图上曲率计算的效率。 |
| [^441] | [The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness](https://arxiv.org/abs/2603.09200) | 该论文提出RAISE框架，揭示提升大语言模型的演绎、归纳与溯因逻辑推理能力会经由三条机制性路径逐步促成AI的情境感知能力，警示逻辑推理研究与AI安全之间的冲突风险。 |
| [^442] | [NePPO: Near-Potential Policy Optimization for General-Sum Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2603.06977) | 提出NePPO方法，通过学习一个与玩家无关的势函数，将一般和博弈转化为以该势函数为共同效用的合作博弈，使其纳什均衡近似原博弈的纳什均衡，从而在混合合作-竞争环境中实现近似纳什均衡的计算。 |
| [^443] | [TimeWarp: Evaluating Web Agents by Revisiting the Past](https://arxiv.org/abs/2603.04949) | 提出了模拟网页UI演变的TimeWarp基准测试，揭示现有网页智能体对界面变化的脆弱性，并通过TimeTraj标注方法与BC变体训练显著提升了智能体跨网页版本的泛化能力。 |
| [^444] | [MARS: Harmonizing Multimodal Convergence via Adaptive Rank Search](https://arxiv.org/abs/2603.00720) | 提出MARS方法，利用双缩放定律自适应搜索LoRA秩对，以平衡多模态大语言模型微调中不同模态的收敛速度，避免负面干扰并提升任务性能。 |
| [^445] | [Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](https://arxiv.org/abs/2603.00191) | 提出LoDA方法，从投影能量视角出发进行任务驱动的低秩子空间分解，在基于LoRA的持续学习中同时实现跨任务知识共享与真正有效的任务特定隔离。 |
| [^446] | [Parameter-Efficient Temporal Domain Generalization](https://arxiv.org/abs/2602.11965) | LoTiC通过将时序参数更新分解为共享的低秩基和一个紧凑的时变核心，在r×r的小空间中学习时序动力学，从而实现对现代大模型参数高效的时序域泛化。 |
| [^447] | [Flow Matching with Uncertainty Quantification and Guidance](https://arxiv.org/abs/2602.10326) | 提出了不确定性感知流匹配UA-Flow，通过预测速度场的异方差不确定性来量化生成样本的可靠性，并利用该信号实现不确定性感知引导，从而提升生成样本的保真度。 |
| [^448] | [ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling](https://arxiv.org/abs/2602.09009) | 该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。 |
| [^449] | [Bridging Academia and Industry: A Comprehensive Benchmark for Attributed Graph Clustering](https://arxiv.org/abs/2602.08519) | 该论文提出了PyAGC——首个面向属性图聚类、可直接用于生产环境的综合基准库，通过统一的“编码-聚类-优化”模块化框架和内存高效的小批次实现，弥合了学术研究与真实工业部署之间的鸿沟。 |
| [^450] | [Is Flow Matching Just Trajectory Replay for Sequential Data?](https://arxiv.org/abs/2602.08318) | 本文推导了序列数据上流匹配目标的最优速度场，揭示其本质是一个非参数、记忆增强的连续时间动力系统，其速度场可表示为观测转移的瞬时速度的加权混合，为神经流匹配模型提供可解释视角并提出FreeFM。 |
| [^451] | [On the Dynamics of Delayed Generalization in Linear Models](https://arxiv.org/abs/2602.08302) | 本文证明延迟泛化现象并非深度网络独有，在线性可分数据上用梯度下降和指数尾损失训练的线性模型中同样会出现，并将超平面动力学刻画为总体、支持向量主导的遗忘和泛化三个阶段，其中动力学的时间尺度是导致延迟泛化的原因。 |
| [^452] | [Rational Transductors](https://arxiv.org/abs/2602.07599) | 本文提出有理转换器这一双流架构，通过将源自加权有限自动机的矩阵值递归以深度有理注入的方式嵌入Transformer注意力机制，将Transformer的表达能力严格推广至可捕获所有正则语言及 $\NC^1$ 完全语言，从而解决其在序列逻辑与状态追踪上的局限。 |
| [^453] | [Deep Time-series Forecasting Needs Kernelized Moment Balancing](https://arxiv.org/abs/2602.00717) | 该论文提出 KMB-DF 方法，通过从再生核希尔伯特空间（RKHS）中自适应地选择最有信息量的平衡函数来实施核化矩匹配，克服了现有方法只能匹配少数预定义矩的局限，从而实现深度时间序列预测中预测分布与真实分布的充分平衡。 |
| [^454] | [SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning](https://arxiv.org/abs/2601.21649) | 本文提出仓库中心化学习（RCL）新范式，将静态代码库转化为交互式学习信号，训练出针对特定代码仓库深度专门化的小型语言模型SWE-Spot-4B，以弥补小型模型处理复杂陌生代码库时泛化能力不足的缺陷。 |
| [^455] | [Beyond Sustainable Development Goal (SDG) Accelerators: Directed Evidence for Adaptive Sustainability Portfolios](https://arxiv.org/abs/2601.20875) | 本研究基于114个国家2000至2024年的面板数据，结合面板格兰杰因果检验与局部投影方法，识别出84个定向SDG联系（40个协同、44个权衡），证明不存在普适的SDG加速器，并主张依据定向因果证据构建适应性的可持续发展政策组合。 |
| [^456] | [Interpolation of GEDI Biomass Estimates with Calibrated Uncertainty Quantification](https://arxiv.org/abs/2601.16834) | 该论文提出注意力神经过程（ANP）概率元学习架构，通过结合局部激光雷达观测集合与地理空间基础模型嵌入来插值GEDI稀疏生物量观测，能够根据景观异质性自适应调整不确定性，并在五种生物群系中实现了竞争性精度和良好的不确定性校准。 |
| [^457] | [Tree-Structured Synergy of Large Language Models and Bayesian Optimization for Efficient CASH](https://arxiv.org/abs/2601.12355) | 该论文提出LB-MCTS框架，以蒙特卡洛树搜索作为共享的轨迹结构化状态，协同大语言模型的语义先验与贝叶斯优化，实现跨算法选择与算法内超参数空间的联合高效优化，从而更好地解决CASH问题。 |
| [^458] | [Imaging-anchored Multiomics in Cardiovascular Disease: Integrating Cardiac Imaging, Bulk, Single-cell, and Spatial Transcriptomics](https://arxiv.org/abs/2601.07871) | 本综述提出以心脏影像为锚点的多组学整合框架，将超声心动图、心脏MRI和CT所定义的心脏空间表型，与bulk、单细胞及空间转录组提供的细胞类型和位置特异的分子背景相融合，从而打破数据孤岛、实现心血管疾病的多模态联合分析。 |
| [^459] | [Long-Range Distillation: Distilling 10,000 Years of Simulated Climate into Long Timestep AI Weather Models](https://arxiv.org/abs/2512.22814) | 提出了一种“远程蒸馏”方法，利用自回归教师模型生成的一万年模拟气候合成数据，训练单步生成的长时间步长概率学生模型，从而解决AI长期天气预报中误差累积与再分析数据不足导致的过拟合问题。 |
| [^460] | [The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior](https://arxiv.org/abs/2512.12066) | 该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。 |
| [^461] | [Tracking large chemical reaction networks and rare events by neural networks](https://arxiv.org/abs/2512.10309) | 提出了一种基于自回归概率分布表示、二阶优化与增强采样策略的统一神经网络框架，可高效求解大型化学反应网络的化学主方程，并精确捕捉稀有事件与亚稳态跃迁动力学。 |
| [^462] | [A Versatile Variational Quantum Kernel Framework for Non-Trivial Classification](https://arxiv.org/abs/2511.10831) | 该研究提出了一个结合资源高效拟设与参数缩放技术的多功能变分量子核框架，在涵盖表格、图像、时间序列和图数据的八个真实高维数据集上取得了与经典RBF核相当的分类性能。 |
| [^463] | [When is a System Discoverable from Data? Discovery Requires Chaos](https://arxiv.org/abs/2511.08860) | 混沌虽然与不可预测性相关，却往往是使动力系统控制方程能够从数据中被发现的关键——系统的域外泛化能力恰好在混沌出现时产生，而保守系统本质上无法仅凭数据被发现。 |
| [^464] | [Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs](https://arxiv.org/abs/2511.04473) | 提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。 |
| [^465] | [A Nesterov-Accelerated Byzantine-Robust Federated Learning](https://arxiv.org/abs/2511.02657) | 该论文提出了一种将Nesterov动量与拜占庭鲁棒聚合规则无缝结合的联邦学习算法Byrd-NAFL，在非凸光滑损失函数和宽松假设下建立了有限时间收敛保证，同时兼顾通信效率与对拜占庭攻击的抵抗能力。 |
| [^466] | [Toward Agents That Reason About Their Computation](https://arxiv.org/abs/2510.22833) | 本文通过让强化学习智能体感知自身计算成本并自主控制计算使用时机，实现了在相同训练预算下于75%的游戏中性能更优、且平均计算量减少三倍的效果。 |
| [^467] | [Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered](https://arxiv.org/abs/2510.08464) | 剪枝会严重破坏VLA模型并加剧安全违规，本文通过谱分析揭示了其权重结构脆弱性的根源，并提出GLUESTICK方法——通过在权重空间对密集模型与剪枝模型插值计算一次性校正项，在保留稀疏性优势的同时恢复模型大部分性能。 |
| [^468] | [Context Value Informed In-Context Reinforcement Learning](https://arxiv.org/abs/2509.23209) | 提出CV-ICRL方法，通过为上下文内策略引入显式的上下文价值估计，并将价值预测以价值token的形式写回上下文，实现TD式回报目标，从而降低策略更新的方差，使测试时对未见任务的适应更加稳定。 |
| [^469] | [Anticipatory Evaluation of Language Models: Predicting LLM Benchmark Scores from Descriptions](https://arxiv.org/abs/2509.20645) | 该论文提出“前瞻性评估”新范式和EVALCAST数据集（2,290对描述-性能数据），证明前沿LLM能够从自然语言任务描述中推断出基准测试的相对难度排序，但精确预测绝对性能分数仍然困难。 |
| [^470] | [Sig-DEG for Distillation: Making Diffusion Models Faster and Lighter](https://arxiv.org/abs/2508.16939) | 本文提出Sig-DEG，利用部分签名与循环结构逼近随机微分方程的解来蒸馏预训练扩散模型，在保持有竞争力的样本质量的同时显著提升推理速度。 |
| [^471] | [Linearly Convergent Algorithms for Nonsmooth Problems with Unknown Smooth Regions](https://arxiv.org/abs/2507.19465) | 本文首次证明bundle-level类方法可在光滑区域未知的情况下，对满足二次增长条件的凸分片光滑函数实现全局线性收敛，并将该方法扩展至弱凸分片光滑问题，同时提出了无需任何问题参数的可验证精确终止准则。 |
| [^472] | [The Invisible Leash: Why RLVR May or May Not Escape Its Origin](https://arxiv.org/abs/2507.14843) | 该论文提出“经验支持动力学”框架，逐题追踪RLVR对已验证解的保持、扩展与收缩，发现在给定采样预算下支持收缩超过扩展，表明RLVR可能并未真正扩展模型推理边界，而主要是放大了基座模型已知的高奖励输出。 |
| [^473] | [State Space Models Naturally Produce Time Cell and Oscillatory Behaviors and Scale to Abstract Cognitive Functions](https://arxiv.org/abs/2507.13638) | 该论文提出状态空间模型凭借结构化的多时间尺度复数旋转基，无需训练即可自然涌现时间细胞和振荡等神经时间动态，并可通过学习扩展至抽象认知功能，为连接微观神经回路与认知功能提供了数学上充分、计算上合理的框架。 |
| [^474] | [Rule-based and reinforcement-learning policies for resource allocation in a pediatric echo lab: a simulation study](https://arxiv.org/abs/2506.06297) | 该研究基于斯坦福儿童医院一周的真实运营数据，构建了儿科超声心动图实验室的离散事件随机仿真模型并通过开源Gymnasium接口提供，为基于规则和强化学习的资源分配策略优化提供了可复现的实验平台。 |
| [^475] | [Score-Based Training for Energy-Based TTS Models](https://arxiv.org/abs/2505.13771) | 本文提出了一种新的分数学习准则，使能量文本转语音模型的分数更适合推理阶段的一阶优化方案，克服了NCE和SSM忽略对数似然函数形式的局限性。 |
| [^476] | [Incentivize Contribution and Learn Parameters Too: Federated Learning with Strategic Data Owners](https://arxiv.org/abs/2505.12010) | 提出两种联邦学习机制：UPBReD 以几何速率联合收敛到贡献博弈的纳什均衡与福利最大化模型，而 2P-UPBReD 通过预算平衡的转移支付使完全贡献成为客户端的占优策略，从而实现最优模型训练。 |
| [^477] | [Trustworthy AI Must Account for Interactions](https://arxiv.org/abs/2504.07170) | 可信人工智能的各维度（如公平性、隐私、鲁棒性等）之间存在负面相互作用，例如差分隐私可能放大偏见损害公平性，因此研究必须放弃孤立改进单一方面做法，转而采用同时兼顾所有维度的整体视角。 |
| [^478] | [Spectral structure learning for clinical time series](https://arxiv.org/abs/2502.11680) | 该论文提出了一种基于多任务平稳高斯过程 StructGP 和改进 NOTEARS 算法的谱结构学习方法，能够从不规则采样的临床时间序列中有效学习时间序列之间的有向条件依赖关系图，并取得了高召回率和高精确率。 |
| [^479] | [Forget About It: Mitigating Forgetting Without Replay or Auxiliary Objectives](https://arxiv.org/abs/2501.05559) | 本文提出序贯微调平均法（SFA），证明仅依靠微调过程中简单的周期性参数插值，无需重放或辅助目标即可有效缓解灾难性遗忘，并在 Qwen3-14B 适配器持续学习中以最多节省 36.3% 训练时间的优势超越了重放和自适应合并等复杂方法。 |
| [^480] | [GLL: A Differentiable Graph Learning Layer for Neural Networks](https://arxiv.org/abs/2412.08016) | 该论文通过伴随方法推导出反向传播方程，提出了可微图学习层GLL，实现了图学习算法与神经网络的端到端精确集成，从而在分类任务中利用样本间的关系信息。 |
| [^481] | [Noise-Aware Differentially Private Variational Inference](https://arxiv.org/abs/2410.19371) | 提出了一种基于随机梯度变分推断的噪声感知差分隐私近似贝叶斯推断新方法，突破了以往方法仅适用于简单概率模型的局限，可应用于高维和非共轭模型，并提出了一种更准确的噪声感知后验评估方法。 |
| [^482] | [Active Exploration in Bayesian Model-based Reinforcement Learning for Robot Manipulation](https://arxiv.org/abs/2404.01867) | 该论文提出了在机器人操作中基于贝叶斯模型的强化学习中通过主动探索改善模型质量和保持数据效率的方法。 |
| [^483] | [Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series](https://arxiv.org/abs/2402.05203) | 贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。 |
| [^484] | [Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models](https://arxiv.org/abs/2311.04902) | 提出GBLM-Pruner，一种利用泰勒展开一阶项、基于少量校准样本归一化梯度的免训练剪枝方法，在大语言模型剪枝中显著优于SparseGPT和Wanda等现有方法。 |
| [^485] | [Generation of data on discontinuous manifolds via continuous stochastic non-invertible networks](https://arxiv.org/abs/2112.09646) | 该论文提出了一种新框架：利用预训练对比编码器获得聚类潜空间，并为每个代表单峰子流形的聚类训练专用低复杂度网络，从而有效生成不连续分布的数据，克服了VAE和GAN等传统方法在此类任务上的局限。 |
| [^486] | [Energy-Based Models For Speech Synthesis.](http://arxiv.org/abs/2310.12765) | 本文引入了一种新的非自回归模型，即基于能量的模型 (EBMs)，通过噪声对比估计进行训练。该模型可以使用 Langevin MCMC 进行采样，并在语音合成任务中取得了改进。 |

# 详细

[^1]: 基于稀疏人类引导的真实世界移动操作分层残差策略学习

    Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance

    [https://arxiv.org/abs/2610.20607](https://arxiv.org/abs/2610.20607)

    提出HiRe-MoMa分层残差框架，通过对冻结基础策略进行移动底盘与机械臂解耦的残差修正，并结合稀疏人类引导的专家价值对齐，实现真实世界移动操作系统的稳定且样本高效的微调。

    

    移动操作是自主机器人系统的基石，然而由于高维动作空间和异构模块之间复杂的耦合关系，真实世界的微调仍然是一项艰巨的挑战。在本文中，我们提出了HiRe-MoMa（分层残差移动操作），这是一个用于复杂移动操作系统稳定且样本高效的真实世界微调的系统性框架。其核心是HiRe-MoMa采用分层残差架构来微调冻结的基础策略。策略无关残差在基础策略之上学习加性修正，而分层解耦将残差拆分为移动底盘和机械臂两个头部并采用单向条件化，从而缓解梯度干扰，并使修正与平台的物理层次结构保持一致。为了在数据昂贵的真实世界环境中提高样本效率，我们提出了专家引导的价值对齐，利用稀疏的人类引导（摘要在此处截断）

    arXiv:2610.20607v1 Announce Type: cross  Abstract: Mobile Manipulation (MoMa) is a cornerstone of autonomous robotic systems, yet real-world fine-tuning remains a formidable challenge because of high-dimensional action spaces and intricate coupling between heterogeneous modules. In this paper, we propose HiRe-MoMa (Hierarchical Residual Mobile Manipulation), a systematic framework for stable and sample-efficient real-world fine-tuning of complex MoMa systems. At its core, HiRe-MoMa adopts a Hierarchical Residual Architecture that fine-tunes a frozen base policy. A Policy-Agnostic Residual learns additive corrections atop the base policy, while Hierarchical Decoupling splits the residual into mobile-base and manipulator-arm heads with unidirectional conditioning, mitigating gradient interference and aligning corrections with the platform's physical hierarchy. To improve sample efficiency in data-expensive real-world settings, we propose Expert-Guided Value Alignment, which leverages spa
    
[^2]: 基于扩散模型的人形机器人策略神经权重建模与理解

    Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion

    [https://arxiv.org/abs/2610.20606](https://arxiv.org/abs/2610.20606)

    该论文提出一种将线性自编码器与扩散Transformer相结合的潜在扩散框架，可直接在权重空间中对强化学习得到的人形机器人控制网络进行生成式建模，从而生成可部署的高性能控制网络。

    

    机器人学习领域的最新进展推动了人形机器人控制的快速发展，使其能够在现实世界中实现越来越敏捷的行为。然而，人形机器人控制网络通常是通过传统训练流程获得的，而针对直接建模其学习到的权重分布、并在权重空间中生成功能性网络的研究却相对较少。在这项工作中，我们研究了能否通过权重对高性能的人形机器人控制网络进行直接的生成式建模。我们采用了一个潜在扩散框架，将线性自编码器与扩散Transformer相结合：自编码器将执行器网络参数压缩到紧凑的潜在空间中，扩散Transformer则对这些潜在表示的分布进行建模。通过在为Unitree G1机器人收集的强化学习网络数据上进行训练，我们的方法能够生成可部署的网络，执行各种运动技能。

    arXiv:2610.20606v1 Announce Type: cross  Abstract: Recent advances in robot learning have driven rapid progress in humanoid control, enabling increasingly agile behaviors in the real world. However, humanoid control networks are typically obtained through conventional training procedures, while comparatively little attention has been given to directly modeling the distribution of their learned weights and generating functional networks in weight space. In this work, we investigate whether high-performing humanoid control networks can be generatively modeled directly through their weights. We adopt a latent diffusion framework that couples a linear autoencoder with a Diffusion Transformer. The autoencoder compresses actor network parameters into a compact latent space, and the Diffusion Transformer models the distribution of these latents. Trained on networks collected from reinforcement learning for the Unitree G1 robot, our approach generates deployable networks that perform various m
    
[^3]: 面向语言智能体的行动与预测协同演化

    Co-Evolving Acting and Forecasting for Language Agents

    [https://arxiv.org/abs/2610.20597](https://arxiv.org/abs/2610.20597)

    该论文提出在成功的同策略轨迹上引入多步动作序列作为监督目标，使行动生成与动作预测协同演化，从而充分利用中间步骤的监督信号，缓解语言智能体训练中长程 rollout 成本高、终端奖励稀疏的问题。

    

    训练语言智能体的瓶颈在于长程轨迹 rollout 的高昂成本，而标准的智能体强化学习（RL）仅从最终标量奖励中学习，导致中间步骤的监督信号在很大程度上未被利用。先前的工作通过两类方式利用这种监督：一是估计每步成功概率的价值目标，二是拟合环境响应的观测目标。但两者都难以良好构造：(1) 价值目标来自轨迹中每个步骤同一的单一标量信号；(2) 观测目标（如网页和终端日志）冗长且含噪，且只能向前预测一步。我们提出 \name，引入第三种监督目标——多步动作目标。在成功的同策略轨迹上，\name 训练策略主干生成每个历史之后所跟随的动作序列。当从先前历史进行预测时，每个动作也会受到监督，从而形成一个时间集成，……（摘要在此处截断）

    arXiv:2610.20597v1 Announce Type: cross  Abstract: Training language agents is bottlenecked by the high cost of long-horizon rollouts, yet standard agentic reinforcement learning (RL) learns only from a terminal scalar reward, leaving the supervision in intermediate steps largely unused. Prior work exploits this supervision through value targets that estimate per-step success probabilities or observation targets that fit environment response. Both are hard to construct well: (1) value targets come from the same single scalar signal for each step in the trajectory, and (2) observation targets, such as web pages and terminal logs, are long and noisy and reach only one step ahead. We propose \name, which adds a third kind of multi-step action targets. On successful on-policy trajectories, \name trains the policy backbone to generate the action sequences that follows each history. Each action is also supervised when forecast from the preceding histories, forming a temporal ensemble that th
    
[^4]: 你的Transformer其实暗藏弹性

    Your Transformer Is Secretly Elastic

    [https://arxiv.org/abs/2610.20588](https://arxiv.org/abs/2610.20588)

    提出一种无需重新训练的可微功能单元排序方法，将预训练Transformer转化为单一弹性模型，可跨连续计算预算实时自适应调整，并在22M至8B参数的27个模型上验证了其有效性。

    

    扩展Transformer已经产生了具备广泛能力的模型，并被部署在异构硬件和波动的工作负载上。然而，模型家族通常只提供少数几个固定尺寸（例如S、B、L），迫使用户根据硬件和延迟约束选定某一个模型。这种刚性在不同查询负载下往往并非最优。我们提出了一种无需重新训练的方法，可以将预训练的Transformer“弹性化”为一个单一模型，该模型能够在连续的计算预算范围内实时自适应调整。我们的关键贡献是对功能单元排序问题的可微松弛：与以往仅优化每个MLP块或注意力头单一重要性权重的工作不同，我们在一个共同尺度上联合地对每一个MLP神经元和注意力头进行排序，方法是在多个稀疏度目标下同时通过软剪枝最小化模型损失。在从22M到8B参数的27个视觉和文本Transformer编码器上……

    arXiv:2610.20588v1 Announce Type: new  Abstract: Scaling up transformers has produced models with broad capabilities, deployed across heterogeneous hardware and fluctuating workloads. Yet, model families offer only a few fixed sizes (e.g., S, B, L), forcing users to commit to a model based on hardware and latency constraints. This rigidity is often suboptimal under varying query loads. We introduce a retraining-free method that elastifies pre-trained transformers into a single model that adapts on-the-fly across a continuum of computational budgets. Our key contribution is a differentiable relaxation of the functional unit-ranking problem: unlike prior work, which optimizes a single importance weight per MLP block or attention head, we rank every individual MLP neuron and attention head jointly, on a common scale, by minimizing the model's loss under soft pruning across many sparsity targets, all at once. Across 27 vision and text transformer encoders ranging from 22M to 8B parameters,
    
[^5]: 代理市场中一价拍卖的无政府代价

    Price of Anarchy of First-Price Auctions in Agentic Markets

    [https://arxiv.org/abs/2610.20587](https://arxiv.org/abs/2610.20587)

    本文研究了代理市场中一价拍卖的无政府代价，指出代理市场本身所具备的特性有望降低一价拍卖带来的福利损失。

    

    代理市场是一种去中心化的服务时市场，AI代理通过竞价来争夺每一条用户提示。代理的出价基于其对提示的期望价值，即其响应所获奖励的期望值减去推理成本。一价拍卖中出价最高者支付其出价并获得该提示，这是在此类市场中分配提示的一种自然方式，因为其运行简单透明，且不需要对私有能力和成本进行中心化了解。然而，一价拍卖并非诚实的：参与者可能出价低于其真实价值以期获得更大利润，这可能导致提示未被分配给对其估值最高的竞价者。这种福利损失由无政府代价来量化。在本文中，我们论证代理市场很可能具备能够降低无政府代价的特性。第一个特性是竞争性下限

    arXiv:2610.20587v1 Announce Type: cross  Abstract: An agentic market is a decentralized serving-time market where AI agents bid to claim each user prompt. An agent's bid is based on its expected value for the prompt, which is the expectation of the reward for its response minus its inference costs. First-price auctions (FPAs), in which the highest bidder pays its bid and receives the prompt, are a natural way to allocate prompts in such a market, since they are simple and transparent to run, and do not require central knowledge of private capabilities and costs. However, FPAs are not truthful: participants may bid less than their true value in the hope of a larger profit, and the prompt may then be allocated to someone other than the bidder who values it the most. This welfare loss is quantified by the $\textit{price-of-anarchy}$ (PoA). In this paper, we argue that agentic markets are likely to possess features that reduce the price-of-anarchy. The first is a $\textit{competitive floor
    
[^6]: 连续注意力

    Continuum Attention

    [https://arxiv.org/abs/2610.20586](https://arxiv.org/abs/2610.20586)

    本文提出“连续注意力”算子，用对连续分段线性记忆曲线的积分取代离散键值查找，通过闭式指数矩实现精确归一化注意力，在语言建模上以负对数似然显著优于softmax和焦点注意力。

    

    我们提出了连续注意力，这是一种因果Transformer注意力算子，它用对连续分段线性记忆曲线的积分取代了离散的键值查找。每个查询沿该曲线诱导出一个指数形式的相关性密度。通过对该密度与插值后的值进行积分，可以从闭式指数矩（包括半个token的边界贡献）得到精确的归一化注意力。其梯度可从至多二阶矩解析地推导出来，从而支持并行GPU实现，同时保持二次序列复杂度和标准的Transformer解码器结构。与普通softmax注意力和焦点注意力相比，CA分别将平均验证负对数似然降低了每字节0.0200和0.0480奈特，并在所有配对比较中获胜。在一个额外的不相交验证切片上，CA相比最强选定的softmax基线取得了每字节0.0457奈特的优势，且具有配对95%置信（原文此处截断）。

    arXiv:2610.20586v1 Announce Type: new  Abstract: We introduce Continuum Attention (CA), a causal Transformer attention operator that replaces discrete key and value lookup with integration over a continuous, piecewise linear memory curve. Each query induces an exponential relevance density along the curve. Integrating the density and interpolated values yields exact normalized attention from closed form exponential moments, including half token boundary contributions. The gradients follow analytically from moments of order at most two, enabling parallel GPU implementations while retaining quadratic sequence complexity and the standard Transformer decoder. Compared with ordinary softmax and focal attention, CA reduces mean validation negative log likelihood by 0.0200 and 0.0480 nats per byte, respectively, winning all paired comparisons. An additional disjoint validation slice yields a 0.0457 nats per byte advantage over the strongest selected softmax baseline, with a paired 95% confide
    
[^7]: 可学习霍奇-拉普拉斯驱动的混合专家模型用于粗非结构网格上的稳态偏微分方程代理模型

    Learnable Hodge-Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes

    [https://arxiv.org/abs/2610.20585](https://arxiv.org/abs/2610.20585)

    该论文提出一种由可学习霍奇-拉普拉斯算子驱动的混合专家架构，将解按频谱尺度分配给不同transformer专家并通过逐顶点门控权重重组，在粗非结构网格稳态PDE代理任务上相比Transolver将相对$L_2$误差降低了31%。

    

    当前针对非结构网格上稳态偏微分方程代理模型的方法对整个解应用单一的混合机制。本文提出了一种新架构，将解划分给多个标准transformer专家，每个专家专注于不同的频谱尺度。该划分由离散霍奇-拉普拉斯算子 $L_0=\star_0^{-1} d_0^{\top}\star_1 d_0$ 构建，作为测量高频内容的拟合层。门控机制以网格为输入，生成逐顶点权重以重新组合各专家的输出。AirFRANS是一个稳态雷诺平均Navier-Stokes（RANS）翼型基准测试。我们在该基准的精简重网格化版本上评估了所提方法，每个非结构网格包含9.6K个面和4.9K个顶点。我们在相同的数据集和数据集划分上与Transolver进行了对比。我们的方法将相对$L_2$误差降低了31%，并降低了内部均方误差。

    arXiv:2610.20585v1 Announce Type: new  Abstract: Current approaches for steady partial differential equations surrogate models on unstructured meshes apply a single mixing mechanism to the whole solution. This paper proposes an architecture that partitions the solution among vanilla transformer experts, each focused on a different spectral scale. The partition is built from a discrete Hodge Laplacian $L_0=\star_0^{-1} d_0^{\top}\star_1 d_0$, that serves as a fitting layer measuring high-frequency content. A gating mechanism takes the mesh as input and produces the per-vertex weights that recombine the expert outputs. AirFRANS is a steady Reynolds-averaged Navier--Stokes (RANS) airfoil benchmark. We evaluate our approach in a reduced remeshed version of the benchmark with each unstructured mesh counting 9.6K faces and 4.9K vertices. We compare our approach with the Transolver on the same dataset and dataset slit. Our approach reduces relative $L_2$ error by 31\% and interior mean square
    
[^8]: 基于测试时训练的潜在压缩

    Latent Compaction with Test-Time Training

    [https://arxiv.org/abs/2610.20582](https://arxiv.org/abs/2610.20582)

    提出“潜在压缩”方法，通过测试时训练在模型权重冻结且不接触下游问题的情况下，学习紧凑的文档级 KV 缓存，显著优于同等预算的缓存压缩基线。

    

    长上下文会扩大键值（KV）缓存，增加生成过程中的内存使用和注意力计算成本。压缩这些缓存时必须为未见过的问 题保留信息。我们提出了**潜在压缩**，该方法通过测试时训练学习一个紧凑的、针对特定文档的缓存。我们使用重构 重要性、教师注意力和键空间多样性来初始化缓存条目。训练首先将注意力输出和对数归一化器拟合到完整缓存的对应 值，其中由重构选出的条目保持固定；随后通过从完整缓存模型进行端到端蒸馏，联合优化整个缓存。整个过程模型权 重保持冻结，训练仅使用关于文档的通用提示，无法访问下游的问题或答案。在 RULER、LongBench 和 Phonebook Lookup 上的实验表明，该方法相较于预算相当的压缩基线取得了显著提升。例如在 RULER 上，我们的方法实现了……（摘要截断）

    arXiv:2610.20582v1 Announce Type: new  Abstract: Long contexts enlarge key--value (KV) caches, increasing memory use and attention costs during generation. Compressing these caches must preserve information for unseen questions. We introduce **Latent Compaction**, which learns a compact document-specific cache through test-time training. We initialize cache entries using reconstruction importance, teacher attention, and key-space diversity. Training first fits attention outputs and log-normalizers to those of the full cache, with reconstruction-selected entries held fixed. It then jointly optimizes the entire cache through end-to-end distillation from the full-cache model. Model weights remain frozen throughout, and training uses only generic prompts about the document, without access to downstream questions or answers. Experiments on RULER, LongBench, and Phonebook Lookup demonstrate substantial gains over budget-matched compression baselines. On a RULER, for example, our method achie
    
[^9]: 一个任务，多种界面：简单的跨界面一致性训练提升LLM智能体的鲁棒性

    One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness

    [https://arxiv.org/abs/2610.20576](https://arxiv.org/abs/2610.20576)

    提出跨界面策略正则化（CIPR）这一自监督方法，通过在两个界面下惩罚动作分布间的KL散度，使LLM智能体在面对界面扰动（如动作重命名、记号变化）时保持一致行为，显著提升其跨界面的鲁棒性。

    

    后训练将LLM智能体的能力与其训练时所用的界面绑定在一起。对该界面进行保持语义的扰动，例如重命名某个动作或改变记号方式，虽然不会改变底层任务本身，却会使智能体的表现变差。由于对每个样本独立评分，界面随机化只会将这种依赖从单一界面扩大到训练界面的分布上，因此分布外的扰动仍会导致智能体性能下降。我们提出了跨界面策略正则化（CIPR），这是一种自监督目标函数，它用无信息的标识符替换动作名称，并在两个界面下呈现每个训练样本，在这两个界面中，等价的动作在构造上即可被对应识别。CIPR惩罚在两个界面下所诱导的动作分布之间的KL散度，从而使策略在不同界面之间保持行为一致。我们在监督学习和强化学习两种设置下评估了CIPR的效果。

    arXiv:2610.20576v1 Announce Type: cross  Abstract: Post-training binds the competence of an LLM agent to the interface it was trained on. Semantics-preserving perturbations of that interface, such as renaming an action or changing the notation, leave the underlying task unchanged yet make the agent worse at it. Scoring each sample independently means interface randomization only broadens this dependence from a single interface to the training interface distribution, so out-of-distribution perturbations still degrade the agent. We propose Cross-Interface Policy Regularization (CIPR), a self-supervised objective that replaces action names with uninformative identifiers and presents each training sample under two interfaces in which equivalent actions are identified by construction. CIPR penalizes the KL divergence between the action distributions induced under the two interfaces, so the policy behaves consistently across them. We evaluate CIPR in both supervised and reinforcement-learnin
    
[^10]: 通过批评引导的在线策略蒸馏恢复可靠监督

    Recovering Reliable Supervision via Critique-Guided On-Policy Distillation

    [https://arxiv.org/abs/2610.20572](https://arxiv.org/abs/2610.20572)

    提出CriticOPD方法，通过教师模型对学生的错误推理轨迹进行批评点评并引导学生重写解答，恢复了在线策略蒸馏中教师监督的可靠性，在数学推理尤其是竞赛题目上显著优于传统OPD。

    

    在线策略蒸馏（OPD）利用教师模型提供的token级别监督，在学生模型自身生成的推理轨迹上进行训练，其前提假设是教师模型能够从学生所写的任意前缀出发推导出正确答案。我们将这一特性称为教师模型的可恢复性，并证明该特性在错误轨迹上会失效：教师模型虽然能够识别学生的错误，却会沿着这些错误继续推导。我们提出CriticOPD方法，利用错误识别能力来恢复可恢复性。教师模型对错误的推理轨迹进行批评点评，使学生能够针对所指出的问题重写其解答。教师模型对包含批评（作为特权信息）的完整输出进行评分，而学生则使用不含批评的修订版本进行训练。在两个Qwen3学生模型和六个数学推理基准上，CriticOPD优于传统OPD方法，其中在竞赛题目上取得了最大的性能提升。

    arXiv:2610.20572v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains a student model on its own rollouts with token-level supervision from a teacher, assuming that the teacher can reach a correct answer from any prefix the student writes. We call this property the recoverability of the teacher and show it fails on incorrect rollouts: the teacher follows student errors, although it recognizes them. We propose CriticOPD, using error recognition to restore recoverability. The teacher critiques an incorrect rollout, allowing the student to rewrite its solution following the error. The resulting output, including the critique as privileged information, is scored by the teacher while the student uses the revision without the critique for training. On two Qwen3 students and six mathematical reasoning benchmarks, CriticOPD improves OPD, with the largest gains on competition problems.
    
[^11]: 使用卷积神经网络检测马铃薯植物叶片病害

    Detection of Potato Plant Leaf Diseases using Convolutional Neural Networks

    [https://arxiv.org/abs/2610.20544](https://arxiv.org/abs/2610.20544)

    本研究开发了一种仅含446,787个参数（约1.70 MB）的轻量级CNN模型，在PlantVillage数据集上达到98.66%的测试准确率，并在埃塞俄比亚独立田间数据集上达到90.28%的准确率，可用于检测马铃薯早疫病和晚疫病。

    

    马铃薯是埃塞俄比亚重要的粮食安全作物，但其产量受到叶部病害的影响，尤其是早疫病和晚疫病。本研究开发了一个轻量级卷积神经网络（CNN），将马铃薯叶片分为三类：健康、早疫病和晚疫病。所提出的模型使用来自PlantVillage数据集的2,215张图像进行训练，并在一个独立的田间数据集上进行了评估，该数据集包含从埃塞俄比亚Awi Zone Banja地区马铃薯农场收集的247张图像。在PlantVillage数据集上，该模型达到了99.04%的训练准确率、98.21%的验证准确率和98.66%的测试准确率；在田间数据集上达到了90.28%的准确率。该研究还在分类性能和计算效率方面将所提出的CNN与VGG16和MobileNetV2进行了比较。所提出的模型包含446,787个可训练参数，占用约1.70 MB存储空间，平均处理每张图像需要5.58秒。

    arXiv:2610.20544v1 Announce Type: new  Abstract: Potato is an important food-security crop in Ethiopia, but its production is affected by foliar diseases, particularly early blight and late blight. This study develops a lightweight convolutional neural network (CNN) to classify potato leaves into three categories: healthy, early blight, and late blight. The proposed model was trained using 2,215 images from the PlantVillage dataset and evaluated on an independent field dataset containing 247 images collected from potato farms in Banja, Awi Zone, Ethiopia. On PlantVillage, the model achieved 99.04% training accuracy, 98.21% validation accuracy, and 98.66% test accuracy. On the field dataset, it achieved 90.28% accuracy. The study also compares the proposed CNN with VGG16 and MobileNetV2 in terms of classification performance and computational efficiency. The proposed model contains 446,787 trainable parameters, occupies approximately 1.70 MB, and requires an average of 5.58 seconds per 
    
[^12]: PrismaCoder：通过分解式评分标准合成突破开放多模态代码智能的边界

    PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis

    [https://arxiv.org/abs/2610.20538](https://arxiv.org/abs/2610.20538)

    该论文提出PrismaCoder，从数据端出发，通过分解式评分标准合成方法，解决多模态代码生成中样本的程序正确性、代码质量与视觉正确性难以同时评估的核心难题，从而推动开放多模态代码智能的发展。

    

    多模态代码智能将程序的符号世界与其执行所创造的视觉世界连接起来。随着模型越来越多地被要求生成图表、网页、动画和半结构化文档，人们对这一能力的兴趣正在迅速增长。近期，专有模型已大幅领先，甚至能够仅凭文本描述或一张截图就生成交互式作品。然而，这些模型的数据和训练过程仍然封闭，开放领域的进展受限于可用数据的数量与质量，以及生产这些数据的流程和方法。其中质量是更难解决的问题，因为一个样本必须同时满足：作为程序是正确的、构建良好的，以及最关键的——渲染后在视觉上是正确的。目前没有任何现有信号能够同时覆盖这三个方面：执行往往只返回通过或失败，代码相似度无法映射到视觉输出，而针对特定任务的验证器又缺乏泛化能力。为了进一步推动开放领域的发展，我们从数据端入手……

    arXiv:2610.20538v1 Announce Type: new  Abstract: Multimodal code intelligence connects the symbolic world of programs with the visual world their execution creates. Interest in this capability is burgeoning, as models are increasingly asked to produce charts, webpages, animations, and semi-structured documents. Recently, proprietary models have pushed far ahead, producing interactive artifacts even from a text description or a screenshot alone. Yet their data and training remain closed, and open progress is held back by the quantity and quality of available data and by the pipelines and methods for producing it. Quality is the harder of the two, since a sample must be correct as a program, well built, and above all visually right once rendered. No existing signal covers all three. Execution often returns just a pass or fail, code similarity does not map to the visual output, and task-specific verifiers do not generalize. To push open progress further, we start from the data side with P
    
[^13]: 重新审视关系深度学习的简单基线方法

    Revisiting Simple Baselines for Relational Deep Learning

    [https://arxiv.org/abs/2610.20534](https://arxiv.org/abs/2610.20534)

    本文提出了一种简单但有效的关系深度学习基线方法，通过将低通特征传播应用于关系实体图并训练MLP，再结合Correct-and-Smooth标签传播算法优化预测结果，证明无需复杂的图神经网络架构即可实现强大的预测性能。

    

    关系深度学习已成为一种强大的范式，它通过将关系数据库表示为时序异构图，从而在其上学习端到端的预测模型。基于图神经网络和Transformer的最新进展在多个不同领域取得了优异的性能。然而，更简单的关系基线方法的有效性仍未得到充分探索，这使得我们难以评估实现强大预测性能究竟需要多复杂的架构。受“预测信号来源于目标属性、关系属性和历史标签”这一观察的启发，我们重新审视了如何使用简单的传播方法来有效利用这些信号。我们将固定的低通特征传播方法适配到关系实体图上，并训练一个多层感知机（MLP）进行预测。所得的预测结果进一步通过Correct-and-Smooth（一种基于图的标签传播算法）进行优化改进……

    arXiv:2610.20534v1 Announce Type: new  Abstract: Relational deep learning has emerged as a powerful paradigm for learning end-to-end predictive models over relational databases by representing them as temporal heterogeneous graphs. Recent advances based on graph neural networks and Transformers have achieved strong performance across diverse domains. However, the effectiveness of simpler relational baselines remains underexplored, making it difficult to assess how much architectural complexity is needed to achieve strong predictive performance. Motivated by the observation that predictive signals arise from target attributes, relational attributes, and historical labels, we revisit how effectively these signals can be exploited using simple propagation methods. We adapt fixed low-pass feature propagation to relational entity graphs and train an MLP for prediction. The resulting predictions are further refined using Correct-and-Smooth, a graph-based label propagation algorithm that inco
    
[^14]: 用于可扩展潜在分词的生成式表示学习

    Generative Representation Learning for Scalable Latent Tokenization

    [https://arxiv.org/abs/2610.20533](https://arxiv.org/abs/2610.20533)

    提出了一种基于逐点互信息分解的生成式自监督表示学习方法，能够避免维度坍缩和特征抑制、随算力增加可预期地提升分词质量，并在分词器与生成模型的综合计算效率上首次实现了优于直接扩散的计算最优改进。

    

    我们提出了一种自监督表示学习方法，专门设计用作生成建模的分词器（tokenizer）。我们的目标通过分解数据的逐点互信息来监督表示损失，该互信息通过计算一个学习到的生成模型的条件预测之间的信息差来估计。我们从理论和实证两方面论证了我们的算法，证明我们的生成式表示学习方法能够避免维度坍缩和特征抑制问题，而在这些情况下，DINO和LeJEPA等常用方法则会失效。此外，我们展示了我们的表示学习技术具有可扩展性；投入更多计算资源可以带来可预期的更好的潜在分词效果。我们首次对分词器和生成模型的综合浮点运算效率进行了细致分析，并证明我们的方法在像素任务上是相对于直接扩散的一种计算最优的改进。

    arXiv:2610.20533v1 Announce Type: new  Abstract: We introduce an approach for self-supervised representation learning designed to be used as a tokenizer for generative modeling. Our objective supervises a representation loss by factorizing the pointwise mutual information of the data, which we estimate by taking information differences between conditional predictions from a learned generative model. We justify our algorithm theoretically and empirically, demonstrating that our generative approach to representation learning avoids dimensional collapse and feature suppression in cases where commonly used methods like DINO and LeJEPA fail. Moreover, we show our representation learning technique is scalable; expending more compute leads to predictably better latent tokenization. For the first time, we perform a careful analysis of the combined flop efficiency of tokenizer and generative model and demonstrate that our approach is a compute-optimal improvement over direct diffusion for pixel
    
[^15]: ActiveEvidence：具有可验证获取差距的决策导向证据图

    ActiveEvidence: Decision-Focused Evidence Graphs with Verifiable Acquisition Gaps

    [https://arxiv.org/abs/2610.20532](https://arxiv.org/abs/2610.20532)

    该论文提出ActiveEvidence框架，通过反事实证据交互图（CEIG）和可验证的ε-决策证书，使多模态智能体能够高效决定何时获取视觉证据、仅评估可能改变决策的关键交互，并在证明进一步交互无法改变所选动作时及时停止。

    

    多模态智能体必须决定视觉证据是否值得获取、查询哪个候选对象或分辨率、以及进一步的交互何时不再能改变其决策。我们在严格的免费元数据/付费观测边界、随机向量成本以及冻结执行轨迹的条件下研究这一问题，并将其视为可决策认证的主动证据获取问题。反事实证据交互图（CEIG）表示一元证据价值、带符号的二阶交互以及有向条件性损害，同时为未解决的交互维护显式的分数包络；决策加权包络目标收紧了可能改变动作身份的界限。间隙导向的细化仅评估能够取代现有获取方案的交互，一旦ε-决策证书表明未解决的项在声明的容差范围内无法改变所选动作，便停止评估。该证书是……

    arXiv:2610.20532v1 Announce Type: cross  Abstract: A multimodal agent must decide whether visual evidence is worth acquiring, which candidate or resolution to query, and when further interaction can no longer change its decision. We study this problem under a strict free-metadata/paid-observation boundary, stochastic vector costs, and a frozen execution trace, and treat it as decision-certifiable active evidence acquisition. A Counterfactual Evidence Interaction Graph (CEIG) represents unary evidence value, signed second-order interaction, and directed conditional harm while maintaining explicit score enclosures for unresolved interactions; a decision-weighted enclosure objective tightens the bounds that can change action identity. Gap-directed refinement evaluates only interactions that can displace the incumbent acquisition and stops once an $\epsilon$-decision certificate shows that unresolved terms cannot change the selected action beyond a declared tolerance. The certificate is mo
    
[^16]: 物理即网络：快速、无需训练的神经仿真

    The Physics Is the Network: Fast, Training-Free Neural Emulation

    [https://arxiv.org/abs/2610.20525](https://arxiv.org/abs/2610.20525)

    PolyNet将多项式动力学的神经表示与简单循环网络结合，在已知物理系数的情况下无需训练即可构建张量图，实现高效硬件执行和大规模并行仿真，并在KdV方程求解中达到与数值求解器相当的精度。

    

    神经仿真不必通过学习已知物理的替代品来实现。我们研究了PolyNet，它将多项式动力学的连续时间神经表示与用于离散时间积分的简单循环网络相耦合。当控制方程的系数已知时，所得到的张量图无需训练，既保留了规定的数值计算格式，又能实现高效的硬件执行和并行仿真。以一维Korteweg-de Vries方程为例，我们将PolyNet与CPU和GPU上的有限差分Runge-Kutta求解器以及三种经过训练的神经模型进行比较：状态映射仿真器、物理信息预测器和学习趋势积分器。解析孤子解为不同网格分辨率、10^6个色散缩放时间步长、多达10^4条轨迹的集合以及不同初始振幅下的振幅和相位误差提供了参考基准。PolyNet实现了与数值求解器相当的精度

    arXiv:2610.20525v1 Announce Type: new  Abstract: Neural emulation need not require learning a substitute for known physics. We investigate PolyNets that couple a continuous-time neural representation of polynomial dynamics with a simple recurrent network for discrete-time integration. When the governing coefficients are known, the resulting tensor graph requires no training and preserves the prescribed numerical scheme while enabling efficient hardware execution and parallel simulation. Using the one-dimensional Korteweg-de Vries equation, we compare PolyNet with CPU and GPU finite-difference Runge-Kutta solvers and three trained neural models: a state-map emulator, a physics-informed predictor, and a learned-tendency integrator. Analytical soliton solutions provide a reference for amplitude and phase errors across grid resolutions, 10^6 dispersively scaled time steps, ensembles of up to 10^4 trajectories, and varying initial amplitudes. PolyNet achieves accuracy comparable to the nume
    
[^17]: VASPO：基于特权价值估计的自蒸馏

    VASPO: Self-Distillation with Privileged Value Estimation

    [https://arxiv.org/abs/2610.20516](https://arxiv.org/abs/2610.20516)

    该论文提出VASPO方法，将特权价值估计与自蒸馏在PPO框架内结合，利用评论家与自教师在学生生成前缀上评估特权信息时呈现的互补位置动态特性，为推理模型提供更有效的词元级训练信号。

    

    在线策略自蒸馏通过特权自教师为推理模型提供密集监督。然而，这种监督反映的是局部词元偏好，而非无指导的学生续写内容的期望回报。近期方法将该信号与GRPO相结合，即用GRPO确定更新方向、用教师模型进行信号缩放。但我们证明了在PPO框架内可以实现更有效的整合。具体而言，我们识别出一种互补的位置动态特性：特权信息在响应末端能够提升评论家价值预测的准确性，而自教师的指导在生成初期较强，中期下降，并在末期回升。这些洞察促成了价值增强自蒸馏策略优化的提出：在该方法中，自教师与评论家在学生生成的前缀处评估相同的特权信息，评论家基于预测的续写回报计算词元级的PPO优势……

    arXiv:2610.20516v1 Announce Type: new  Abstract: On-policy self-distillation provides dense supervision for reasoning models via a privileged self-teacher. However, this supervision reflects local token preferences rather than the expected return of an unguided student continuation. Recent methods combine this signal with GRPO. They use GRPO for update directions and the teacher for signal scaling. However, we demonstrate a more effective integration within PPO. Specifically, we identify complementary positional dynamics: privileged information enhances a critic's value-prediction accuracy toward the response's end, whereas self-teacher guidance is strong initially, drops mid-generation, and rebounds finally. These insights motivate Value-Augmented Self-Distillation Policy Optimization (VASPO), wherein a self-teacher and critic evaluate identical privileged information at student-generated prefixes. The critic computes token-level PPO advantages from predicted continuation returns, whi
    
[^18]: 贝叶斯学习选择公平分类器

    Bayesian Learning to Select Fair Classifiers

    [https://arxiv.org/abs/2610.20514](https://arxiv.org/abs/2610.20514)

    本文提出一种基于汤普森采样的贝叶斯多臂老虎机在线方法，在真实标签延迟可得的条件下，自适应选择同时兼顾预测准确性与敏感属性子群体公平性的分类器（涵盖表格模型、大语言模型与人工标注者），并相比基线方法显著降低了遗憾值。

    

    对分类器（无论是AI、人类还是智能体）的评估通常假设批量设置，即每个智能体需要对测试集中的每个样本都产生标签，这在许多场景下成本高昂或不切实际。这促使人们研究在线策略，即在样本到来时自适应地选择要查询的智能体，并在观察到结果后更新策略。我们研究了在在线数据流中选择智能体的问题，该智能体需最佳地平衡（i）预测性能与（ii）敏感属性的子群体公平性，同时允许真实类别标签的延迟观测。我们将该问题建模为基于汤普森采样的贝叶斯多臂老虎机问题，显式地平衡准确性与公平性。我们在涵盖表格分类器、大语言模型和人工标注者的真实世界数据集上进行了实证评估，结果表明相对于基线方法遗憾值显著降低。我们的理论贡献包括……

    arXiv:2610.20514v1 Announce Type: new  Abstract: Evaluations of classifiers (AI or human or agents) often assume a batch setting in which every agent produces labels on every example in a test set, which can be expensive or impractical in many contexts. This motivates online policies that adaptively select agents to query as examples arrive and update policies as outcomes are observed. We study the problem of selecting, in an online stream, the agent that best balances (i) predictive performance and (ii) subgroup fairness of a sensitive attribute, and that allows for delayed observations of true class labels. We formulate this as a Bayesian multi-armed bandit problem using Thompson Sampling, explicitly balancing accuracy and fairness. We conduct empirical evaluations on real-world datasets spanning problems involving tabular classifiers, large language models, and human labelers, demonstrating substantial reductions in regret relative to baselines. Our theoretical contributions include
    
[^19]: 一种针对从纵向电子健康记录中发现的疾病轨迹的两级临床新颖性优先级排序框架

    A Two-Level Clinical Novelty Prioritization Framework for Disease Trajectories Discovered from Longitudinal Electronic Health Records

    [https://arxiv.org/abs/2610.20512](https://arxiv.org/abs/2610.20512)

    该论文提出了一个两级框架，将疾病轨迹分支模体的语义稀疏性与经可靠性调整的临床优先级分离，从而对从纵向电子健康记录中发现的 99,527 个模体进行新颖性评估和临床优先级排序。

    

    基于图的电子健康记录（EHR）分析可以生成疾病轨迹目录，但结构上的支持并不能确立语义上的新颖性。我们提出了一个两级框架，用于对之前由上游患者不重叠、仅基于训练集的 MIMIC-IV v3.1 流水线发现并在基于验证的目录选择后冻结的 99,527 个分支模体进行优先级排序。该框架将路径级与节点到终点的语义稀疏性同经可靠性调整的临床优先级分离开来。对每个模体或节点-目标对进行三种确定性文本化处理，使用 BAAI/bge-small-en-v1.5 进行嵌入，并通过平均 top-20 余弦距离与 166,972 篇 2023 年之前的 PubMed 参考文献进行校准。结果表明，正向重组出现在 8.63% 的模体中，对全局排名的影响可以忽略不计，而特异性和支持度/GNN 可靠性项则改变了临床优先级的排序。此外还开展了跨邻居数量、模板、编码器及通用标签设置的鲁棒性分析（摘要在此处被截断）。

    arXiv:2610.20512v1 Announce Type: new  Abstract: Graph-based analysis of electronic health records (EHRs) yields disease-trajectory catalogs, but structural support does not establish semantic novelty. We present a two-level framework to prioritize 99,527 branching motifs previously discovered by an upstream patient-disjoint, train-only MIMIC-IV v3.1 pipeline and frozen after validation-based catalog selection. The framework separates path-level and node-to-endpoint semantic sparsity from reliability-adjusted clinical priority. Three deterministic textualizations per motif or node-target pair were embedded with BAAI/bge-small-en-v1.5 and calibrated against 166,972 pre-2023 PubMed references using mean top-20 cosine distance. Positive recombination occurred in 8.63% of motifs and had negligible effect on global ranking, whereas specificity and support/GNN reliability terms changed clinical-priority ordering. Robustness analyses across neighbor counts, templates, encoders, generic-label 
    
[^20]: VascuSR：用于CT超分辨率的迭代式解剖感知扩散模型

    VascuSR: Iterative Anatomy-Aware Diffusion for CT Super-Resolution

    [https://arxiv.org/abs/2610.20509](https://arxiv.org/abs/2610.20509)

    VascuSR提出了一种将图像重建与显式3D血管引导相结合的迭代式解剖感知扩散框架，有效解决了CT超分辨率重建中血管细节缺失、虚假或错误连接的问题。

    

    生成式超分辨率（SR）可以从严重下采样的观测中恢复计算机断层扫描（CT）图像的清晰细节，但缺失的高频内容并非由低分辨率输入唯一确定。因此，视觉上看似合理的重建结果可能包含解剖学上不支持的血管细节，例如缺失的、虚假的或错误连接的血管。为了克服这一挑战，我们提出了VascuSR，这是一个迭代式的解剖感知扩散框架，它将图像重建与显式的3D血管引导相结合。首先，为了提供更丰富的解剖学引导，我们使用显式的3D血管表示来捕捉单个切片之外的结构信息。这种表示还简化了血管细化过程，使轻量级的3D生成细化器能够专注于血管结构而非复杂的CT外观。其次，为了引导逐切片的超分辨率，一个轻量级的解剖控制器……

    arXiv:2610.20509v1 Announce Type: cross  Abstract: Generative super-resolution (SR) can recover sharp detail in computed tomography (CT) images from severely downsampled observations, but the missing high-frequency content is not uniquely determined by the low-resolution input. Consequently, visually plausible reconstructions may contain anatomically unsupported vascular detail, such as missing, spurious, or incorrectly connected vessels. To overcome this challenge, we introduce VascuSR, an iterative anatomy-aware diffusion framework that combines image reconstruction with explicit 3D vascular guidance. First, to provide richer anatomical guidance, we use an explicit 3D vascular representation that captures structural information beyond individual slices. This representation also simplifies vascular refinement by allowing a lightweight 3D generative refiner to focus on vascular structure rather than complex CT appearance. Second, to guide slice-wise SR, a lightweight anatomy controller
    
[^21]: 基于纵向电子健康记录的图增强疾病轨迹发现与时序终点预测

    Graph-Enhanced Disease Trajectory Discovery and Temporal Endpoint Forecasting from Longitudinal Electronic Health Records

    [https://arxiv.org/abs/2610.20502](https://arxiv.org/abs/2610.20502)

    该研究提出了一种在MIMIC-IV上实现泄漏控制的图增强疾病轨迹模式发现框架，结合时间感知Transformer进行终点预测，使事件时间预测误差平均降低19.2天。

    

    纵向电子健康记录（EHR）能够揭示疾病进展的结构，而有效的预测需要泄漏可控的发现方法、考虑删失的结局以及可检验的轨迹证据。我们在MIMIC-IV v3.1数据集上开发了一个图增强框架：在构建图之前对患者进行划分，仅从训练集中学习连续入院的转移关系，并冻结在验证数据上筛选出的针对特定终点的分支模式策略。最终的模式目录包含99,527个模式（motifs），一个时间感知的Transformer通过针对特定终点的交叉注意力机制，为每个终点最多检索24个与患者兼容的模式。在五个训练种子上，micro-AUPRC从0.9330提升至0.9357，同时事件时间的MAE从676.9天降至657.7天（减少19.2天；95%配对患者聚类自助法置信区间为13.8-24.8）。时间效应呈现异质性：730天内事件的MAE有所恶化，而对于更晚发生的事件则改善了58.9天。

    arXiv:2610.20502v1 Announce Type: new  Abstract: Longitudinal electronic health records (EHRs) can reveal disease-progression structure, while useful forecasting requires leakage-controlled discovery, censor-aware outcomes, and inspectable trajectory evidence. We developed a graph-enhanced framework in MIMIC-IV v3.1 that splits patients before graph construction, learns train-only consecutive-admission transitions, and freezes endpoint-specific branching-motif policies selected on validation data. The final catalog contained 99,527 motifs, and a time-aware Transformer retrieved at most 24 patient-compatible motifs per endpoint through endpoint-specific cross-attention. Across five training seeds, micro-AUPRC increased from 0.9330 to 0.9357, while event-time MAE decreased from 676.9 to 657.7 days (reduction 19.2 days; 95% paired patient-cluster bootstrap CI, 13.8-24.8). Timing effects were heterogeneous: MAE worsened for events within 730 days but improved by 58.9 days for events occurr
    
[^22]: 事后特权RLVR：无需组采样的单次轨迹词元信用分配

    Hindsight-Privileged RLVR: Single-Rollout Token Credit Without Group Sampling

    [https://arxiv.org/abs/2610.20497](https://arxiv.org/abs/2610.20497)

    提出HP-RLVR，一种无需组采样的单次轨迹RLVR方法，通过比较因果视角与包含未来结果和验证器反馈的事后特权视角之间的Jensen–Shannon散度为每个词元分配有界信用，从而在所有尝试结果相同时仍能提供有效学习信号。

    

    可验证奖励的强化学习（RLVR）利用经过检验的任务结果来训练语言模型，但如何将最终奖励分配到各个推理词元或动作上仍然困难。以组相对策略优化（GRPO）为代表的组相对方法需要对每个提示采样多次尝试，但当所有尝试获得相同结果时，其基于奖励的学习信号便会失效。我们提出事后特权RLVR（HP-RLVR），这是一种单次轨迹方法，将已完成轨迹自身的未来转化为词元级的更新权重。生成策略快照在每个合格词元的因果历史条件下，以及在一个包含已实现未来和验证器反馈的事后上下文条件下分别对该词元进行打分，两个视角之间的归一化Jensen–Shannon散度相对于历史基线对奖励进行加权，从而为成功和失败的尝试都提供有界的词元信用，且无需参考答案或学习到的评论家模型。（原文摘要在此处截断）

    arXiv:2610.20497v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains language models from checked task outcomes, but assigning a terminal reward to individual reasoning tokens or actions remains difficult. Group-relative methods such as Group Relative Policy Optimization (GRPO) sample several attempts per prompt, but lose their reward-based learning signal when all attempts receive the same outcome. We propose Hindsight-Privileged RLVR (HP-RLVR), a single-rollout method that turns a completed trajectory's own future into token-level update weights. The generating policy snapshot scores each eligible token under its causal history and under a hindsight context containing the realized future and verifier feedback. A normalized Jensen--Shannon divergence between the two views weights the reward relative to a historical baseline, yielding bounded token credit for both successful and failed attempts, with no reference solution or learned critic. In p
    
[^23]: 从单条持续轨迹中进行平均奖励策略识别

    Average-Reward Policy Identification from a Single Continuing Trajectory

    [https://arxiv.org/abs/2610.20486](https://arxiv.org/abs/2610.20486)

    该论文针对平均奖励强化学习提出了一种自适应决定在何处收集信息的算法，并通过对称的上下界证明：对于一大类环境，在足够高的精度下，从单条持续轨迹中学习在最坏情况下并不比可自由采样的学习更困难。

    

    在强化学习中，智能体必须探索未知环境以学习如何做出好的决策。与可以自由采样任意“状态-动作”对的方法不同，从单条轨迹中学习的智能体必须花费额外的时间在状态之间移动。因此，人们自然会认为从单条轨迹中学习需要更多的经验。在本文中，我们证明这种额外的代价并非总是必要的。对于平均奖励强化学习，我们开发了一种算法，该算法能够学习在何处收集信息，同时确保所需的样本可以被高效获取。我们建立了识别近似最优策略所需交互次数的上界。此外，我们证明了一个相匹配的信息论下界，表明对于一大类环境，在足够高的精度下，从单条轨迹中学习在最坏情况下并不比可自由采样的学习更困难。

    arXiv:2610.20486v1 Announce Type: new  Abstract: In reinforcement learning, an agent must explore an unknown environment to learn how to make good decisions. Unlike methods that can freely sample any state--action pair, an agent learning from a single trajectory must spend additional time navigating between states. It is therefore natural to expect that learning from a single trajectory requires more experience. In this paper, we show that this additional cost is not always necessary. For average-reward reinforcement learning, we develop an algorithm that learns where to collect information while ensuring that the required samples can be obtained efficiently. We establish an upper bound on the number of interactions needed to identify a near-optimal policy. Furthermore, we prove a matching information theoretic lower bound, showing that, for a broad class of environments and at sufficiently high accuracy, learning from a single trajectory is no harder in the worst case than learning wi
    
[^24]: 面向免训练推理时扩展的解耦边际锐化

    Decoupled Marginal Sharpening for Training-Free Inference-Time Scaling

    [https://arxiv.org/abs/2610.20483](https://arxiv.org/abs/2610.20483)

    该论文提出解耦边际锐化（DMS），在目标层面和采样器层面分别进行锐化，以克服全序列锐化抑制多样推理路径以及有限粒子SMC路径坍缩的问题，从而实现无需训练和奖励模型的LLM推理时性能提升。

    

    我们研究如何在推理阶段改进大语言模型（LLM）的推理能力，而无需更新模型权重或使用外部奖励模型。近期的无奖励幂采样方法定义了序列级幂目标 $\pi_\alpha(y\mid x)\propto p_0(y\mid x)^\alpha$，其中 $\alpha>1$，该目标提升了在基础模型下已经具有较高相对概率的补全的相对概率。现有方法使用 Metropolis--Hastings 或序贯蒙特卡洛（SMC）来近似地从该目标中采样。然而，在两个不同层面会出现挑战：在目标层面，全序列锐化可能将概率集中到少数精确的“推理--答案”轨迹上，从而抑制支持相同答案的其他备选推理路径；在采样器层面，有限粒子的 SMC 在重采样过程中反复复制少数高权重前缀时，可能出现路径坍缩。为应对这些挑战，我们提出了解耦边际锐化（Decoupled Marginal Sharpening, DMS），它分别在……（原文摘要到此截断）

    arXiv:2610.20483v1 Announce Type: new  Abstract: We study how to improve LLM reasoning at inference time without updating model weights or using external reward models. Recent reward-free power-sampling methods define the sequence-level power target $\pi_\alpha(y\mid x)\propto p_0(y\mid x)^\alpha$, where $\alpha>1$, which increases the relative probability of completions that are already likely under the base model. Existing methods use Metropolis--Hastings or Sequential Monte Carlo (SMC) to approximately sample from this target. However, two challenges arise at different levels. At the target level, full-sequence sharpening can concentrate probability on a few exact reasoning--answer trajectories, suppressing alternative paths that support the same answer. At the sampler level, finite-particle SMC can suffer from path collapse when resampling repeatedly duplicates a few high-weight prefixes. To address these challenges, we propose Decoupled Marginal Sharpening (DMS), which separately 
    
[^25]: 翻译中的迷失：为什么更低的偏见分数并不意味着更公平的模型

    Lost in Translation: Why Lower Bias Scores Don't Mean Fairer Models

    [https://arxiv.org/abs/2610.20479](https://arxiv.org/abs/2610.20479)

    本研究审计了10个开源模型在8种语言中的偏见表现，发现非英语语言下刻板印象分数的下降主要源于翻译缺陷与测量噪声，而非模型本身更公平，因为测试项的英语分数对其翻译后分数的预测能力远低于对改写后分数的预测能力。

    

    多语言语言模型的偏见审计通常将模型在英语中的平均刻板印象分数与其在另一种语言中的平均分数进行比较，并将二者的差异解读为公平性上的差异。我们探究这种差异中有多少其实是噪声。我们审计了八个语言下的十个开源模型，测量当英语测试项仅仅被改写时分数的变动幅度，逐项检验每一次语言转换，并验证该测试能够捕捉我们故意注入的转换。在全部70个模型-语言对中，刻板印象分数均低于其英语水平，且印地语、韩语和斯瓦希里语的下降幅度明显超过改写英语所导致的下降；其中大部分下降来自StereoSet测试项。通过对每个翻译进行评判，我们发现不完美的翻译下降幅度更大，但即使只看忠实的翻译，也几乎出现同样程度的下降。尽管如此，更低的平均值并不能表明模型更公平：一个测试项的英语分数对其翻译后分数的预测能力远不如对其改写后分数的预测能力。

    arXiv:2610.20479v1 Announce Type: new  Abstract: Bias audits of multilingual language models usually compare a model's average stereotype score in English with its average in another language and read the difference as a difference in fairness. We ask how much of that difference is noise. Auditing ten open models in eight languages, we measure how far scores move when English test items are merely reworded, test every language shift item by item, and check that the test catches shifts we inject on purpose. In all 70 model-language pairs the stereotype score falls below its English level, and for Hindi, Korean and Swahili the drop clearly exceeds what rewording English produces; most of it comes from StereoSet items. Judging every translation, we find that imperfect ones drop more, but faithful translations alone drop nearly as much. Still, a lower average does not show that a model is fairer. An item's English score predicts its translated score far less well than its reworded score (c
    
[^26]: SafeAdapt-Poker：对手漂移下预算化适应的反事实风险校准

    SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift

    [https://arxiv.org/abs/2610.20473](https://arxiv.org/abs/2610.20473)

    该论文提出 SafeAdapt-Poker，将对手漂移下的预算化适应建模为仅依赖决策前信息的双重反事实序贯控制问题，通过单调策略风险模型与配对预算状态延续模型分别校准偏离的局部下行风险和预算消耗的未来代价，并用逐路径累计KL账本约束总偏离量，从而在即时适应价值与安全风险之间实现可校准的权衡。

    

    在行为发生漂移后，对手的历史行为记录可能会产生误导。SafeAdapt-Poker 将对手漂移下的预算化适应问题表述为一个双重反事实序贯控制问题，仅使用每次决策之前可获得的信息。在每次决策时，控制器会提出两个不同的问题：一是在匹配的合法后续情形下，偏离冻结锚点会引发怎样的局部下行风险；二是现在消耗一部分剩余偏离预算会损失多少未来的适应价值。第一个问题由一个关于偏离强度的单调策略风险模型来回答；第二个问题则由一个配对的预算状态延续模型来回答，该模型为当前的预算消耗定价。控制器通过将即时适应价值与这两个量进行权衡，在保持支撑集的候选偏离中进行选择；一条逐路径的累计KL散度账本对相对于锚点的总偏离量进行约束；同时从一组有限的冻结控制器中进行筛选。

    arXiv:2610.20473v1 Announce Type: cross  Abstract: Opponent histories can mislead after behavioral shifts. SafeAdapt-Poker formulates budgeted adaptation under opponent shift as a dual-counterfactual sequential-control problem using only information available before each decision. At each decision the controller asks two distinct questions: what local downside a deviation from the frozen anchor would induce under matched legal continuations, and what future adaptation value would be lost by consuming part of the remaining deviation budget now. The first question is answered by a monotone strategic-risk model over deviation intensity; the second by a paired budget-state continuation model that prices current budget consumption. The controller selects among support-preserving candidate deviations by trading immediate adaptation value against these two quantities, a pathwise cumulative-KL ledger bounds total anchor-relative deviation, and a finite family of frozen controllers is screened 
    
[^27]: 选择性生长Q网络：解耦Q学习中自适应动作网格的分析

    Selective Growing Q-Networks (SGQN): An Analysis of Adaptive Action Grids in Decoupled Q-Learning

    [https://arxiv.org/abs/2610.20463](https://arxiv.org/abs/2610.20463)

    本文提出选择性生长Q网络（SGQN），在解耦Q学习中用自适应动作网格（在高Q值区域附近增加bin、剪除低Q值bin并采用epsilon-贪婪动作选择）取代均匀分辨率增长，在多个DMControl连续控制任务上取得优异表现，但消融分析发现自适应网格本身并不能解释这些性能优势。

    

    基于价值的Q学习用于连续控制时可以对动作空间进行离散化，但离散化的分辨率需要在早期探索与最终精度之间进行权衡。生长Q网络（GQN）从粗到细均匀地增加分辨率。我们提出了选择性生长Q网络（SGQN），它保留了去耦合Q学习，并用自适应动作网格取代均匀增长：在具有高估计Q值的bin旁边添加新的bin，剪除低Q值的bin，并以epsilon-贪婪方式选择动作。在具有2到12个动作维度的八个DeepMind Control Suite（DMControl）任务上，SGQN在四个任务上取得了所有对比方法中最高的平均最终回报，在另外三个任务上与最佳方法的差距在1.1%以内，在Walker Run任务上比Bang-Bang MPO低3.4%，每个维度平均使用9个可用bin中的7.5个。随后我们检验自适应网格是否能解释这些结果，发现它并不能。增长会激活每个候选……（摘要原文在此处被截断）

    arXiv:2610.20463v1 Announce Type: new  Abstract: Value-based Q-learning for continuous control can discretise the action space, but the resolution of the discretisation trades early exploration against final precision. Growing Q-Networks (GQN) grow the resolution uniformly from coarse to fine. We introduce Selective Growing Q-Networks (SGQN), which keeps decoupled Q-learning and replaces uniform growth with an adaptive action grid: bins are added next to bins with high estimated Q-value, bins with low Q-value are pruned, and actions are chosen epsilon-greedily. On eight DeepMind Control Suite (DMControl) tasks with 2 to 12 action dimensions, SGQN has the highest mean final return among the compared methods on four, is within 1.1% of the best method on three more, and is 3.4% below Bang-Bang MPO on Walker Run, using on average 7.5 of 9 available bins per dimension. We then test whether the adaptive grid explains these results, and find that it does not. Growth activates every candidate 
    
[^28]: 从示例中学习：基于表格基础模型的电网稳态分析

    Learning by Examples: Power Grid Steady State Analysis with a Tabular Foundation Model

    [https://arxiv.org/abs/2610.20462](https://arxiv.org/abs/2610.20462)

    本研究证明预训练表格基础模型TabPFN无需重新训练或微调，即可作为电网稳态分析（潮流预测、N-k预想故障筛查和电压稳定裕度预测）的通用神经求解器，在多个任务上达到或超越任务特定模型。

    

    基础模型已在多个科学领域展现出强大的泛化能力，但其在电力系统中的应用大多局限于特定任务。我们研究了预训练的表格基础模型（TFM）TabPFN能否作为电网稳态分析的通用神经求解器。我们在三个任务上对该TFM进行了评估：潮流（PF）预测、N-k预想故障筛查（NCS）以及电压稳定裕度（VSM）预测。我们在118节点和2000节点系统上开展了实验，将该TFM与特定任务的神经基线模型进行对比。在无需重新训练或微调的情况下，该TFM在三个任务的多个场景中达到甚至超越了最佳任务特定模型的性能。在118节点系统的潮流预测任务中，TFM的一个变体实现了$1\times 10^{-3}$ pu的电压幅值误差和$0.30^\circ$的电压相角误差，优于所有基线模型。在未见的N-1故障场景以及接近不可行的潮流算例上，该TFM同样表现出较强的竞争力。

    arXiv:2610.20462v1 Announce Type: new  Abstract: Foundation models have shown strong generalization across scientific domains, but their use in power systems remains largely task-specific. We investigate whether a pretrained tabular foundation model (TFM), TabPFN, can serve as a general-purpose neural solver for power grid steady-state analysis. We evaluate the TFM on three tasks: power flow (PF) prediction, N-k contingency screening (NCS), and voltage stability margin (VSM) prediction. We performed experiments on 118- and 2000-bus systems, comparing the TFM against task-specific neural baselines. Without retraining or fine-tuning, the TFM matches or outperforms the best task-specific model in several settings across the three tasks. For 118-bus PF prediction, a variant of the TFM achieves errors of $1\times 10^{-3}$ pu for voltage magnitudes and $0.30^\circ$ for voltage angles, outperforming all baselines. On unseen N-1 contingency and close-to-infeasible PF cases, the TFM is competit
    
[^29]: The COUNT：通过计数N-gram解锁主机内存上的扩展与免训练适配

    The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation

    [https://arxiv.org/abs/2610.20456](https://arxiv.org/abs/2610.20456)

    提出COUNT方法，通过在廉价的主机内存上存储预计算的n-gram计数并结合语言模型自身的unembedding来构建n-gram嵌入，无需训练新参数，从而实现主机内存上的扩展与免训练适配。

    

    语言模型（LMs）的计算与性能之间的缩放关系已被广泛研究；最近，模型也被设计为能够扩展内存使用。特别是，为n-gram添加嵌入可以使内存扩展与计算使用相互独立，因为每次前向传播只处理单个嵌入行，其大小远小于整个嵌入表所占用的存储空间。然而，现有方法需要从头训练嵌入，这需要昂贵的GPU内存来进行快速读写。与此同时，主机系统的普通内存价格低廉，却在这些训练方法中未被充分利用。为了在主机内存上实现扩展，我们提出了COUNT（Counting Offline Unlocks N-gram Transformers），它通过在主机上存储预计算的n-gram计数来构建n-gram嵌入，而无需添加任何可学习参数。我们利用语料库频率，对语言模型自身的unembedding进行加权组合，以表示最常跟随当前上下文的后继词元……（摘要原文在此处截断）

    arXiv:2610.20456v1 Announce Type: new  Abstract: The scaling relationship of compute and performance for language models (LMs) has been extensively studied; more recently, models are being designed to scale memory use, as well. In particular, adding embeddings for n-grams scales memory separately from compute use, because each pass only processes individual embedding rows, which are much smaller than the storage used for the whole embedding table. However, existing methods train embeddings from scratch, which needs expensive GPU memory for fast reading and writing. Meanwhile, the ordinary memory of the host system is cheap and underutilized by these training methods. To scale on host memory, we introduce COUNT (Counting Offline Unlocks N-gram Transformers), which builds n-gram embeddings without adding learned parameters by storing precomputed n-gram counts on host. We use the corpus frequencies to weight together the LM's own unembeddings of the next tokens most often following the cu
    
[^30]: 分歧意味着什么？成对偏好学习的语义可辨识性边界

    What Does Disagreement Mean? A Semantic Identifiability Frontier for Pairwise Preference Learning

    [https://arxiv.org/abs/2610.20450](https://arxiv.org/abs/2610.20450)

    该论文提出包含方向偏好、持续分歧和非二元质量三个量的非参数潜在语义模型，证明在匿名二值反馈通道下分歧的语义含义本质上是不可辨识的，但通过同标注者重复标注与允许弃权的反馈相结合可以完整点辨识这三个语义量。

    

    成对比较数据是偏好学习中的标准监督信号，但此类数据中的分歧可能反映稳定的潜在异质性、短暂的随机性，或是被强制选择接口所掩盖的非二元响应。我们研究了这些语义区分中哪些可以从反馈通道本身被辨识出来。我们引入了一个非参数潜在语义模型，包含三个按查询定义的量：聚合方向偏好 $\mu_q$、持续分歧 $\tau_q$，以及非二元质量 $\alpha_q$。在匿名二值通道下，我们证明了一个精确的语义纤维定理：$\mu_q$ 是点可辨识的，而 $(\tau_q, \alpha_q)$ 则在与同一二值分布相容的显式可行域内变化。同一标注者的重复标注可通过协方差恒等式辨识 $\tau_q$，支持弃权的反馈则可直接辨识 $\alpha_q$，两者结合即可点辨识完整的语义三元组。我们还推导了有限……

    arXiv:2610.20450v1 Announce Type: cross  Abstract: Pairwise comparison data are a standard supervision signal in preference learning, but disagreement in such data can reflect stable latent heterogeneity, ephemeral stochasticity, or non-binary responses hidden by a forced-choice interface. We study which of these semantic distinctions are identifiable from the feedback channel itself. We introduce a nonparametric latent semantic model with three querywise quantities: aggregate directional preference $\mu_q$, persistent disagreement $\tau_q$, and non-binary mass $\alpha_q$. Under the anonymous binary channel, we prove an exact semantic fiber theorem: $\mu_q$ is point-identified, whereas $(\tau_q, \alpha_q)$ vary over an explicit feasible region compatible with the same binary law. Same-annotator repeats identify $\tau_q$ through a covariance identity, abstention-enabled feedback identifies $\alpha_q$ directly, and together they point-identify the full semantic triple. We also derive fin
    
[^31]: 精度之井：消失初始化下的非单调精度动态

    The Accuracy Well: Nonmonotone Accuracy Dynamics at Vanishing Initialization

    [https://arxiv.org/abs/2610.20439](https://arxiv.org/abs/2610.20439)

    该论文发现小初始化训练神经网络时精度呈现非单调的“精度之井”现象——先快速上升、随后下降、很久后才恢复，并为浅层和深层线性网络建立了严格的三阶段理论解释。

    

    我们研究了小初始化下的分类动态，并识别出一种显著的精度非单调行为，我们称之为“精度之井”：精度在训练早期快速上升，随后下降，直到很久之后才恢复。我们在深度线性网络、MNIST上的全连接ReLU网络、CIFAR-10上的深度卷积网络以及受控合成任务中观察到这一现象，且该现象跨越不同的损失函数和优化器，同时存在于训练精度和测试精度中。我们提出了一个三阶段的解释框架：目标驱动的组织、退化瓶颈，以及随后的招募与恢复。我们为浅层和深层线性网络建立了这些动态的严格理论，特别地表明：在预测器和损失变化保持渐近消失的同时，高精度就可能出现，而在持续优化之后精度才会恶化。对于一个简单的合成分类任务，我们……

    arXiv:2610.20439v1 Announce Type: new  Abstract: We study classification dynamics under small initialization and identify a pronounced non-monotone behavior of accuracy that we call the accuracy well: accuracy rises rapidly early in training, subsequently drops, and only much later recovers. We observe this phenomenon in deep linear networks, fully connected ReLU networks on MNIST, deep convolutional networks on CIFAR-10, and controlled synthetic tasks, across different losses and optimizers and in both training and test accuracy. We propose a three-stage explanation in terms of target-driven organization, a degeneracy bottleneck, and subsequent recruitment and recovery. We develop a rigorous theory of these dynamics for shallow and deep linear networks, showing in particular that high accuracy can emerge while the predictor and the change in loss remain asymptotically vanishing, before continued optimization causes accuracy to deteriorate. For a simple synthetic classification task, w
    
[^32]: 面向可靠序列内分子通信的持续学习

    Continual Learning for Reliable In-Sequence Molecular Communication

    [https://arxiv.org/abs/2610.20426](https://arxiv.org/abs/2610.20426)

    本文提出仿射记忆回放（AMR）方法，通过存储历史仿真样本并利用插值与外推合成回放样本，在实验设置复杂度逐步递增的序列内分子通信中实现通信性能的增量估计，在适应新设置的同时保持整体估计精度。

    

    本文研究可靠的序列内分子通信，即生物纳米机器按指定顺序交换以分子编码的信息，并提出一种新的基于仿真的方法，通过逐步增加实验设置的复杂度来增量式评估通信协议。所提出的方法称为仿射记忆回放（AMR），其构建一个持续学习（CL）模型来估计生物纳米机器之间的通信性能。AMR存储过去的仿真样本，并通过在过往样本之间进行插值与外推来合成额外的回放样本。在增量式性能估计任务中，它同时回放过往样本和合成样本以正则化模型更新，帮助模型在适应新实验设置的同时保留先前设置的知识。实验结果表明，AMR在保持整体估计精度方面显著优于现有的持续学习方法。

    arXiv:2610.20426v1 Announce Type: new  Abstract: This paper considers reliable in-sequence molecular communication, where bio-nanomachines exchange molecule-encoded information in a specified order, and investigates a new simulation-based method to incrementally evaluate communication protocols by gradually increasing the complexity of experimental settings. The proposed method, called Affine Memory Replay (AMR), builds a continual learning (CL) model to estimate communication performance between bio-nanomachines. AMR stores past simulation samples and synthesizes additional replay samples through interpolation and extrapolation between past samples. Through incremental performance estimation tasks, it replays both past and synthetic samples to regularize model updates, helping the model adapt to new experimental settings while retaining knowledge from earlier ones. Experimental results show that AMR significantly outperforms existing CL methods in maintaining overall estimation accura
    
[^33]: OrionMoE-v1.0：用于表格上下文学习的混合专家基础模型

    OrionMoE-v1.0: A Mixture-of-Experts Foundation Model for Tabular In-Context Learning

    [https://arxiv.org/abs/2610.20418](https://arxiv.org/abs/2610.20418)

    提出了OrionMoE-Small，一个将分布感知表示与数据集条件稀疏混合专家计算相结合的紧凑型表格基础模型，通过双轴Transformer沿特征轴自适应激活专家，实现异构表格任务的高效上下文学习。

    

    表格基础模型能够在无需针对特定任务进行参数更新的情况下，通过从带标签的支持表中预测未见观测值来实现上下文学习。尽管近年来取得了一定进展，但大多数现有方法仍依赖于在特征和数据集上应用相同变换的密集架构。我们提出了OrionMoE-Small，这是一个紧凑的表格基础模型，它将分布感知表示与基于数据集条件的稀疏混合专家计算相结合。OrionMoE-Small建立在双轴Transformer架构之上，沿特征轴选择性地激活专家，其路由决策同时参考了单个特征表示和更广泛的数据集上下文。这种设计使模型能够在异构表格问题上自适应地分配计算资源，同时保留跨任务的共享计算路径。OrionMoE-Small完全在由结构（原文此处截断）生成的合成分类任务上进行预训练。

    arXiv:2610.20418v1 Announce Type: new  Abstract: Tabular foundation models enable in-context learning by predicting unseen observations from a labeled support table without task-specific parameter updates. Despite recent advances, most existing approaches rely on dense architectures that apply the same transformations across features and datasets. We introduce OrionMoE-Small, a compact tabular foundation model that combines distribution-aware representations with dataset-conditioned sparse mixture-of-experts computation. Built on a dual-axis Transformer architecture, OrionMoE-Small selectively activates experts along the feature axis, with routing decisions informed by both individual feature representations and the broader dataset context. This design enables adaptive allocation of computation across heterogeneous tabular problems while retaining a shared computational pathway across tasks. OrionMoE-Small is pretrained entirely on synthetic classification tasks generated from a struct
    
[^34]: 在线凸优化中追踪一小组专家

    Tracking a Small Set of Experts in Online Convex Optimization

    [https://arxiv.org/abs/2610.20416](https://arxiv.org/abs/2610.20416)

    本文证明了在线凸优化中“追踪一小组专家”问题存在本质的计算复杂性分离：信息论上可达的最优遗憾度 $\widetilde{O}(md+k)$ 在超图植入团猜想等复杂性假设下无法被任何多项式时间算法实现，高效算法只能达到 $\widetilde{O}(dk)$ 量级的遗憾度。

    

    对于从专家建议中进行预测这一基本问题，Freund 曾提出疑问：算法能否与一个仅在 $m$ 位专家之间切换 $k$ 次的比较对象相竞争。对于离散的专家集合，Bousquet 和 Warmuth（2020）解决了这一问题，他们给出了一个高效算法，其遗憾度达到了信息论极限。我们在在线凸优化中研究类似的问题，此时动作空间是连续的。通过将动作空间离散化并应用 Bousquet–Warmuth 算法，可以导出一个低效的算法，其遗憾度为 $\widetilde{O}(md+k)$，而目前已知的针对该问题的最佳高效算法遗憾度为 $\widetilde{O}(dk)$。我们证明了这种计算上的低效是该问题所固有的，从而为该问题建立了信息论遗憾界与多项式时间可达遗憾界之间的分离。具体而言，在超图植入团猜想（hypergraphic planted-clique conjecture）之下，我们证明……（原文摘要在此处截断）

    arXiv:2610.20416v1 Announce Type: new  Abstract: For the fundamental problem of prediction from expert advice, Freund asked whether one could compete with a comparator that switches $k$ times among only $m$ experts. For a discrete set of experts, this problem was resolved by Bousquet and Warmuth (2020), who gave an efficient algorithm with optimal regret matching the information-theoretic limit.   We study the analogous problem in online convex optimization, where the action space is continuous. An inefficient algorithm can be derived by discretizing the action space and applying the Bousquet--Warmuth algorithm, giving regret of $\widetilde{O}(md+k)$, while the best known efficient algorithm for this problem has regret $\widetilde{O}(dk)$. We show that this computational inefficiency is inherent, establishing a separation between the information-theoretic and polynomial-time regret achievable for this problem. Specifically, under the hypergraphic planted-clique conjecture, we show that
    
[^35]: 一种尺寸无法适配所有情况：面向SAM的曲率感知自适应扰动

    One Size Does Not Fit All: Curvature-Aware Adaptive Perturbation for SAM

    [https://arxiv.org/abs/2610.20415](https://arxiv.org/abs/2610.20415)

    该论文提出CAP-SAM方法，通过基于锐度比率自适应地调节SAM的扰动半径，克服了固定扰动半径忽略局部曲率的缺陷，从而在损失景观上更有效地搜索鲁棒极小值。

    

    深度神经网络泛化能力差的问题常常与收敛到尖锐极小值有关。锐度感知最小化（SAM）通过同时最小化训练损失和损失景观的局部锐度，使模型收敛偏向平坦极小值，从而改善深度神经网络的泛化能力。然而，SAM依赖于固定的扰动半径ρ，忽略了局部曲率，导致低效的行为：扰动半径可能太小而无法逃离由尖锐局部极小值所定义的不良盆地，也可能太大而无法收敛到具有良好损失-锐度权衡的解。我们将ρ重新解释为扰动信号的强度，并提出了面向SAM的曲率感知自适应扰动方法（CAP-SAM），这是一种几何感知的方法，基于锐度比率来调节扰动半径，从而能够在损失景观上更有效地搜索鲁棒极小值。我们证明了CAP-SAM保持了……（原文在此处截断）

    arXiv:2610.20415v1 Announce Type: new  Abstract: Poor generalisation in deep neural networks is frequently associated with convergence to sharp minima. Sharpness-Aware Minimisation (SAM) improves generalisation of deep neural networks by minimising a combination of the training loss and the local sharpness of the loss landscape, biasing convergence towards flat minima. However, SAM relies on a fixed perturbation radius $\rho$ that ignores local curvature, leading to inefficient behaviour: the perturbation may be too small to escape poor basins defined by sharp local minima, or too large to allow convergence to solutions with good loss-sharpness tradeoff. We reinterpret $\rho$ as the intensity of the perturbation signal and propose Curvature-Aware Adaptive Perturbation for SAM (CAP-SAM), a geometry-aware method that modulates the perturbation radius based on a sharpness ratio, enabling a more effective search for robust minima on the loss landscape. We prove that CAP-SAM preserves the s
    
[^36]: 动量何时有助于可验证奖励强化学习？

    When Does Momentum Help in Reinforcement Learning with Verifiable Rewards?

    [https://arxiv.org/abs/2610.20413](https://arxiv.org/abs/2610.20413)

    本文通过带偏差修正EMA动量的老虎机形式化分析RLVR，理论上证明精确梯度下增大动量系数β可严格加速优化，而采样估计梯度下需增大β以维持有效步长η/(1-β)来克服采样波动。

    

    动量是加速凸优化的经典工具，也是现代大语言模型训练的标准组成部分。然而，近期的实证工作对一阶动量在可验证奖励强化学习（RLVR）中的有效性提出了质疑。我们通过带有偏差修正指数移动平均（EMA）动量的RLVR多臂老虎机形式化来研究这一问题，其中EMA系数 $\beta$ 控制过去梯度权重衰减的速度。当策略梯度被精确计算时，我们证明在防止过冲（overshoot）的范围内增大 $\beta$ 能严格加速优化。当梯度从采样响应中估计得到时，正的期望增益可能被采样波动所掩盖。我们证明，随着学习率 $\eta$ 减小，固定 $\beta$ 下的期望增益比这些波动消失得更快，而增大 $\beta$ 以使 $\eta/(1-\beta)$ 保持在……（摘要在此处截断）

    arXiv:2610.20413v1 Announce Type: new  Abstract: Momentum is a classical tool for accelerating convex optimization and a standard component of modern large language model training. Yet recent empirical work has questioned the usefulness of first-moment momentum in reinforcement learning with verifiable rewards (RLVR). We study this question through a bandit formulation of RLVR with bias-corrected exponential moving average (EMA) momentum, where the EMA coefficient $\beta$ controls how quickly the weights on past gradients decay. When policy gradients are evaluated exactly, we prove that increasing $\beta$ within a range that prevents overshoot strictly accelerates optimization. When gradients are estimated from sampled responses, a positive expected gain can be obscured by sampling fluctuations. We show that as the learning rate $\eta$ decreases, the expected gain with fixed $\beta$ vanishes faster than these fluctuations, whereas increasing $\beta$ to keep $\eta/(1-\beta)$ within the 
    
[^37]: 超球面对比表示解锁规划与探索

    Hyperspherical Contrastive Representations Unlock Planning and Exploration

    [https://arxiv.org/abs/2610.20409](https://arxiv.org/abs/2610.20409)

    该论文提出C-Slerp方法，利用超球面参数化的对比表示，通过球面线性插值以闭式解直接推断中间路径点，实现了无需奖励、演示或预训练模型的长时程规划与探索，成功完成最多八块立方体的搭建等复杂任务。

    

    我们开发了一种自监督强化学习算法，能够从零开始学习搭建高达八个立方体的塔，而无需先验数据、奖励、演示或预训练模型。我们的方法——通过球面线性插值进行对比规划——将对比表示作为可复用的构建模块，并反复组合它们来推断一系列中间路径点。在我们的超球面参数化下，学习到的表示可以通过球面线性插值以闭式解的方式推断中间路径点，无需训练单独的高层策略或执行图搜索。重要的是，表示的几何结构还能告诉我们如何根据每个目标达成问题的难度来调整路径点的数量。实验表明，我们的方法在长时程导航和机器人操作任务上取得了成功，包括强基线方法失败的场景。

    arXiv:2610.20409v1 Announce Type: new  Abstract: We develop a self-supervised reinforcement learning algorithm that learns to build towers of up to eight cubes from scratch, without prior data, rewards, demonstrations, or pretrained models. Our method, Contrastive Planning via Spherical Linear Interpolation (C-Slerp), uses contrastive representations as reusable building blocks and repeatedly composes them to infer a sequence of intermediate waypoints. Under our hyperspherical parameterization, the learned representations allow intermediate waypoints to be inferred in closed form through spherical linear interpolation, without training a separate high-level policy or performing graph search. Importantly, the representation geometry also tells us how to adjust the number of waypoints based on the difficulty of each goal-reaching problem. Experiments show that our method succeeds on long-horizon navigation and robotic manipulation tasks, including settings where strong baselines fail to 
    
[^38]: 后验频率与不拟合的代价：对记忆化下界的一个修正

    Posterior Frequencies and the Cost of Not Fitting: A Correction to a Memorization Lower Bound

    [https://arxiv.org/abs/2610.20402](https://arxiv.org/abs/2610.20402)

    本文通过精确的三点反例证明 Feldman 记忆化下界中的频率系数不成立且无法用普遍正常数因子修复，并基于全样本后验概率给出了一个精确的替代系数。

    

    arXiv:2610.20402v1 公告类型：cross（交叉发布） 摘要：Feldman 的记忆化下界根据数据点在训练样本中出现的频率来赋予其频率系数。我们利用独立公平的标签和一个置换等变的学习器，构造了一个精确的三点反例，否定了其条件形式与平均形式。通过构造一列采样先验，使所提出的惩罚项与实际超额风险之比呈指数增长，从而排除了任何普遍的正常数因子修复的可能性。这些先验随样本量变化，出问题的频率剖面变得稀有，且两个量都趋近于零。在保留原有模型的前提下，我们用全样本后验概率得到了一个精确的替代结果。在可交换性条件下，这些后验概率取决于出现次数（多重性）以及完整的频率剖面。我们刻画了对无限制学习器有效的、仅依赖多重性的最大系数。原始系数具有 Good 精确频率公式的后验均值形式，并且是一种占用加权平均。

    arXiv:2610.20402v1 Announce Type: cross  Abstract: Feldman's memorization lower bound assigns a frequency coefficient based on how often a point appears in the training sample. We give an exact three-point counterexample to its conditional and averaged forms, using independent fair labels and a permutation-equivariant learner. A sequence of sampling priors makes the ratio of the proposed penalty to actual excess risk grow exponentially, ruling out a universal positive constant-factor repair. The priors vary with sample size, the offending profiles become rare, and both quantities vanish. Retaining the original model, we obtain an exact replacement using full-sample posterior probabilities. Under exchangeability, these depend on multiplicity and the complete frequency profile. We characterize the largest multiplicity-only coefficients valid for unrestricted learners. The original coefficient has the posterior-mean form of Good's exact frequency formula and is an occupancy-weighted avera
    
[^39]: 训练语言模型优雅地失败

    Training Language Models to Fail Gracefully

    [https://arxiv.org/abs/2610.20401](https://arxiv.org/abs/2610.20401)

    提出GracefulRL方法，通过重塑优化格局，使语言模型仅凭结果级奖励信号就能学会在行动不可靠时执行安全的回退策略、实现“优雅失败”，无需人工标注或推理时额外机制。

    

    强化学习能让语言模型更擅长解决任务，但同样的机制也会奖励猜测行为。当成功是获得奖励的唯一途径时，选择安全的回退方案总是不如孤注一掷的尝试划算。而当错误的操作可能破坏数据库或提交有缺陷的代码时，这种激励机制可能代价高昂。因此，我们提出训练模型“优雅地失败”：在尽可能保持成功行为的同时，学会在行动不可靠时执行安全的回退策略。然而，由于不利的优化动态，简单地在策略梯度中对回退行为使用中间奖励是脆弱的。我们通过GracefulRL来应对这一挑战，该方法通过重塑优化格局来促进优雅失败。重要的是，这种行为直接从结果层面的奖励信号中涌现，无需标注何时回退，也无需额外的推理时机制（如验证器）。

    arXiv:2610.20401v1 Announce Type: new  Abstract: Reinforcement learning makes language models better at solving tasks, but the same mechanism also rewards guessing. When success is the only way to earn reward, opting for a safe fallback is always worse than a long-shot attempt. When a wrong action can corrupt a database or ship broken code, that incentive can be costly. Therefore, we propose training models to fail gracefully: to preserve successful behavior whenever possible while learning to execute a safe fallback policy when acting would be unreliable. However, naively using policy gradients with intermediate reward for fallback behavior is brittle due to unfavorable optimization dynamics. We address this challenge with GracefulRL, which reshapes the optimization landscape to promote graceful failure. Importantly, this behavior emerges directly from outcome-level reward signals, without annotations specifying when to fall back or additional inference-time machinery such as a verifi
    
[^40]: 基于神经算子的HRTF上采样与个性化

    HRTF Upsampling and Personalization via Neural Operators

    [https://arxiv.org/abs/2610.20400](https://arxiv.org/abs/2610.20400)

    SPARO是一种将稀疏空间测量映射为连续HRTF场的神经算子，在SONICOM数据集的所有稀疏度水平上均实现了最优的LSD和ILD误差，显著优于现有最强基线。

    

    从头相关传输函数（HRTF）的稀疏测量进行空间上采样是个性化双耳音频中的一项根本性挑战。我们将HRTF上采样表述为一个算子学习问题，并提出了SPARO——一种神经算子，它将一组空间方向上的稀疏测量映射到HRTF场，这是一种可以在任意位置查询的连续表示。SPARO将测量投影到学习得到的球面潜在模式集合上，使用Transformer块处理这些潜在模式，并在查询方向上解码所得的表示。在SONICOM数据集上，使用LAP24测量配置，SPARO在从3到100个测量的每个稀疏度水平上都取得了最佳的对数谱失真（LSD）和耳间电平差（ILD）误差（在100个测量时，LSD比最强基线低18%，ILD误差低43%），同时在耳间时间差（ITD）误差方面保持竞争力。

    arXiv:2610.20400v1 Announce Type: cross  Abstract: Head-related transfer function (HRTF) spatial upsampling from sparse measurements is a fundamental challenge in personalized binaural audio. We formulate HRTF upsampling as an operator learning problem and introduce SPARO, a neural operator that maps sparse measurements at a set of spatial directions to the HRTF field, a continuous representation that can be queried at arbitrary positions. SPARO projects the measurements onto a learned set of spherical latent modes, processes the latent modes with Transformer blocks, and decodes the resulting representations at query directions. On SONICOM, using the LAP24 measurement configuration, SPARO attains the best log-spectral distortion (LSD) and interaural level difference (ILD) error at every sparsity level from 3 to 100 measurements (18% lower LSD and 43% lower ILD error than the strongest baseline at 100 measurements) while remaining competitive in interaural time difference (ITD) error. I
    
[^41]: Cortex：将上下文整合为参数化记忆

    Cortex: Consolidating Context into Parametric Memory

    [https://arxiv.org/abs/2610.20399](https://arxiv.org/abs/2610.20399)

    本文提出 Cortex，通过生成前瞻性查询将上下文增益蒸馏为可合并的低秩或结构化前馈网络参数更新，在 4B 至 120B 的多种稠密与混合专家模型上实现了“巩固—保持”帕累托最优的记忆巩固。

    

    上下文可以提升大语言模型的性能，但这些增益在移除上下文后可能会丢失。受生物系统巩固机制的启发，我们将记忆巩固定义为在保留非目标能力的同时将这些增益内化，从而在统一目标下连接了上下文指定编辑、技能学习和行为引导。我们引入了一个涵盖五类任务的测试平台，用于衡量单次、批量与顺序更新下的巩固与保持效果。我们提出 Cortex，它通过生成前瞻性查询来预判上下文的用途，并将上下文条件下的输出蒸馏为可合并的低秩或结构化前馈网络更新。在 4B 至 120B 参数规模的稠密与混合专家骨干网络上（包括纯文本与多模态模型），Cortex 在大多数评估设置中位于经验性的“巩固—保持”帕累托前沿。在对 14B 模型进行 50 次顺序事实编辑后，Cortex-Structural（摘要原文在此处截断）

    arXiv:2610.20399v1 Announce Type: new  Abstract: Context can improve large language model performance, but these gains may be lost after context removal. Inspired by biological systems consolidation, we formulate Memory Consolidation (MC) as internalizing these gains while preserving non-target capabilities, connecting context-specified editing, skill learning, and behavioral steering under a common objective. We introduce a five-task-family testbed measuring consolidation and retention under single, batch, and sequential updates. We propose Cortex, which generates prospective queries to anticipate uses of context and distills context-conditioned outputs into mergeable low-rank or structural feed-forward network updates. Across 4B--120B dense and mixture-of-experts backbones, including both text-only and multimodal models, Cortex lies on the empirical consolidation--retention Pareto frontier in most evaluated settings. After 50 sequential factual edits on a 14B model, Cortex-Structural
    
[^42]: 从人类纠错到可学习的分割：关于信息、学习与人类付出的批判性数学综述

    From Human Corrections to Learned Segmentations: A Critical Mathematical Review of Information, Learning, and Human Effort

    [https://arxiv.org/abs/2610.20396](https://arxiv.org/abs/2610.20396)

    该综述提出统一的数学框架，借助精确贝叶斯更新、超额风险分解与条件信息分解，系统比较主动学习、机器教学和交互式分割等方法，阐明人类纠错何时转化为有效学习，以及在同等人类付出下如何衡量与实现更好的分割效果。

    

    人类的纠错何时才能成为有用的学习？在同等人类付出的条件下，又是什么决定了更好的分割效果？这篇批判性叙述性综述追踪反馈从选择、学习器更新到评估的完整过程。它基于一个统一的数学框架——涵盖目标信息、预测误差、响应可预测性与人类工作量——对主动学习、机器教学、交互式分割和顺序预测评估进行比较。精确的贝叶斯更新将评估标签信息转化为预期的对数损失改进；超额风险分解则解释了近似学习器可能损失什么。条件信息分解将所选位置、标签以及所展示上下文的作用区分开来。实证比较说明了响应协议、更新机制和付出衡量标准为何至关重要，并区分了针对当前掩码的优化与可复用的学习，以及模拟交互与真实人类证据之间的差别。

    arXiv:2610.20396v1 Announce Type: new  Abstract: When does a human correction become useful learning, and what establishes better segmentation at matched human effort? This critical narrative review follows feedback through its selection, learner update, and evaluation. It compares active learning, machine teaching, interactive segmentation, and sequential predictive evaluation using a common mathematical account of target information, predictive error, response predictability, and human work. An exact Bayesian update converts evaluation-label information into expected log-loss improvement; an excess-risk decomposition explains what an approximate learner can lose. Conditional information decompositions separate the roles of selected locations, labels, and displayed context. Empirical comparisons show why response protocols, update mechanisms, and effort denominators matter, distinguishing current-mask refinement from reusable learning and simulated interaction from human evidence. We 
    
[^43]: FOLD：面向长上下文投机解码的具有局部顺序的无位置远端记忆

    FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding

    [https://arxiv.org/abs/2610.20392](https://arxiv.org/abs/2610.20392)

    FOLD是一种免训练的长上下文投机解码方法，它保留精确的局部位置，将所有远端条目作为共享位置上的远端记忆处理，并通过锐化记忆内部的注意力来缓解注意力稀释问题。

    

    投机解码通过轻量级起草与并行验证来加速大语言模型的生成。它对长输入尤其有前景，因为KV缓存的访问成本可以摊销到多个被接受的token上。然而，大多数起草器是在短序列上训练的，在长输入上性能会退化，而长上下文训练的代价高昂。现有的免训练方法要么限制上下文窗口，要么重新缩放位置，要么用检索到的远端条目来增强近期上下文。然而，起草器所读取的目标隐藏状态已经编码了上下文与位置信息，并且其轻量级架构使得全历史注意力变得可行。我们提出FOLD，一种免训练方法，它保留精确的局部位置，并将所有远端条目视为处于共享位置上的远端记忆。为应对记忆增长带来的注意力稀释问题，我们在该记忆内部锐化注意力以突出相关条目，同时单独控制其……（摘要在此处截断）

    arXiv:2610.20392v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM generation through lightweight drafting and parallel verification. It is particularly promising for long inputs, where KV-cache access costs can be amortized over multiple accepted tokens. Yet most drafters are trained on short sequences and degrade on long inputs, while long-context training is costly. Existing training-free approaches restrict the context window, rescale positions, or augment recent context with retrieved distant entries. Drafters, however, read target hidden states that already encode contextual and positional information, and their lightweight architecture makes full-history attention affordable. We propose FOLD, a training-free method that preserves exact local positions and treats all distant entries as far memory at a shared position. To counter attention dilution as the memory grows, we sharpen attention within it to emphasize relevant entries while separately controlling its 
    
[^44]: 基于加权中位数的假设选择

    Hypothesis Selection via Weighted Medians

    [https://arxiv.org/abs/2610.20375](https://arxiv.org/abs/2610.20375)

    本文提出了一种基于加权中位数的假设选择算法，在达到最优近似因子和样本复杂度的同时，将运行时间对失败概率的依赖从多项式级改进为对数级。

    

    给定 n 个分布（假设）$\mathcal H=\{H_1,\ldots,H_n\}$ 以及来自未知目标分布 P 的样本，适当假设选择旨在找到一个分布，使其与 P 的总变差距离接近已知分布中可达到的最佳水平。我们提出了一种算法，该算法使用 $O(\varepsilon^{-2}\log(n/\delta))$ 个目标样本，运行时间为 $O(n\varepsilon^{-2}\log(n/\delta))$，以至少 $1-\delta$ 的概率返回满足 $d_\mathrm{TV}(P,H_{\hat i})\leq 3\min_{i\in[n]}d_\mathrm{TV}(P,H_i)+\varepsilon$ 的 $H_{\hat i}$。该算法达到了最优的近似因子和样本复杂度。该算法的运行时间将 Aamand、Aliakbarpour、Chen 和 Silwal（COLT 2026）的最先进界限 $\tilde{O}(n\varepsilon^{-2}\delta^{-1})$ 中对失败概率的依赖从多项式级改进为对数级，从而回答了他们提出的问题 2。

    arXiv:2610.20375v1 Announce Type: cross  Abstract: Given $n$ distributions (hypotheses) $\mathcal H=\{H_1,\ldots,H_n\}$ and samples from an unknown target distribution $P$, proper hypothesis selection seeks a distribution whose total variation distance to $P$ is close to the best achievable within the known distributions. We give an algorithm that, using $O(\varepsilon^{-2}\log(n/\delta))$ target samples, running in $O(n\varepsilon^{-2}\log(n/\delta))$ time, with probability at least $1-\delta$, returns $H_{\hat i}$ satisfying $d_\mathrm{TV}(P,H_{\hat i})\leq 3\min_{i\in[n]}d_\mathrm{TV}(P,H_i)+\varepsilon.$ The algorithm achieves the optimal approximation factor and sample complexity. The running time improves the dependence on the failure probability from polynomial to logarithmic in the state-of-the-art bound $\tilde{O}(n\varepsilon^{-2}\delta^{-1})$ of Aamand, Aliakbarpour, Chen, and Silwal (COLT 2026), thereby answering their Question~2.
    
[^45]: ReAl：通过对齐动力学流来精炼神经算子

    ReAl: Refine neural operators via aligning dynamical flow

    [https://arxiv.org/abs/2610.20369](https://arxiv.org/abs/2610.20369)

    ReAl 是一个模型无关的即插即用方法，通过条件流匹配模型精炼预训练 PDE 代理模型的细尺度结构，并通过对齐动力学流来抑制自回归滚动预测中的生成误差累积，从而在保持长期稳定性的同时改善高频结构的重建。

    

    使用均方误差（MSE）训练的神经偏微分方程（PDE）代理模型能够捕捉主导动力学，但常常丢失高频信息和细尺度结构。生成模型（例如流匹配，Flow Matching）可以捕捉细尺度细节的分布，然而在自回归滚动预测过程中，随机误差可能会不断累积，从而损害长期稳定性。为了解决这些局限性，我们提出了 Refine and Align（ReAl），这是一个与模型无关的即插即用模块，可用于增强预训练的 PDE 代理模型，以改善与复杂物理现象相关的高频结构的重建。在精炼阶段，条件流匹配模型学习确定性基础预测周围的残差分布，在保留主导动力学的同时恢复缺失的细尺度结构。在对齐阶段，为了减轻在自回归滚动预测中被放大的生成差异，粒子（摘要在此处被截断）

    arXiv:2610.20369v1 Announce Type: new  Abstract: Neural partial differential equation (PDE) surrogates trained with mean squared error (MSE) can capture dominant dynamics but often lose high-frequency information and fine-scale structures. Generative models (e.g., Flow Matching) can capture the distribution of fine-scale details, yet stochastic errors may accumulate during autoregressive rollouts and compromise long-term stability. To address these limitations, we introduce Refine and Align(ReAl), a model-agnostic plug-in that augments pretrained PDE surrogates to improve the reconstruction of high-frequency structures associated with complex physical phenomena. In the refinement stage, a conditional flow-matching model learns the residual distribution around the deterministic base prediction, recovering missing fine-scale structures while retaining the dominant dynamics. In the alignment stage, to mitigate the generative discrepancies amplified in the autoregressive rollout, particula
    
[^46]: 分数阶更优：神经算子学习中的可学习导数阶数

    Fractional is Better: Learnable Derivative Orders in Neural Operator Learning

    [https://arxiv.org/abs/2610.20363](https://arxiv.org/abs/2610.20363)

    该论文提出在神经算子中引入与偏微分方程结构对齐的可学习导数特征，从理论上证明其可将逼近界从 $O(W^{-(s-m)/d})$ 改进为 $O(W^{-s/d})$，并出人意料地发现最优导数阶数是严格低于偏微分方程阶数 $m$ 的分数阶 $\beta^*$。

    

    神经算子学习函数空间之间的映射，从而能够为偏微分方程提供快速的代理求解方案。尽管这些方法在架构上具有显著的多样性，它们通常共享一种通用的输入表示：原始的坐标-数值对 $(x, u(x))$。我们探究与偏微分方程微分结构对齐的输入能否改进学习效果。通过对温和解进行Picard迭代，我们证明输入的导数会显式地出现在解算子中，这表明提供导数特征应当能够减轻网络隐式微分的负担。我们证明了这一直觉：提供导数特征可将逼近界从 $O(W^{-(s-m)/d})$ 改进为 $O(W^{-s/d})$，其中 $W$ 是参数数量，$s$ 是输入正则性，$m$ 是偏微分方程的阶数，$d$ 是空间维度。然而，我们的核心发现出人意料：最优导数阶数 $\beta^*$ 严格小于偏微分方程的阶数 $m$。这一差距

    arXiv:2610.20363v1 Announce Type: new  Abstract: Neural operators learn mappings between function spaces, enabling fast surrogate solutions to partial differential equations. Despite remarkable architectural diversity, these methods often share a common input representation: raw coordinate-value pairs $(x, u(x))$. We ask whether inputs aligned with PDE differential structure can improve learning. Through Picard iteration on mild solutions, we show that derivatives of the input appear explicitly in the solution operator, suggesting that providing derivative features should reduce the network's implicit differentiation burden. We prove this intuition: providing derivative features improves approximation bounds from $O(W^{-(s-m)/d})$ to $O(W^{-s/d})$, where $W$ is parameter count, $s$ is input regularity, $m$ is the PDE order, and $d$ is spatial dimension. Our central finding, however, is a surprise: the optimal derivative order $\beta^*$ is strictly less than the PDE order $m$. This gap 
    
[^47]: 超越专家排序：面向混合精度MoE量化的对比校准

    Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization

    [https://arxiv.org/abs/2610.20353](https://arxiv.org/abs/2610.20353)

    提出NatScale框架，通过模型特定的幂变换将基于梯度的专家权重校准到信息论对比度目标，并联合评估门控与上下投影配置的输出误差，从而在存储预算下实现更优的混合精度MoE量化位宽分配。

    

    混合专家语言模型在每个token上仅激活少数专家，但所有专家权重都必须被存储。混合精度量化通过为敏感专家分配更高位宽、为鲁棒专家分配更低位宽来降低这种存储成本，但仅凭专家排序并不能决定对其误差的加权强度。为填补这一空白，我们提出了NatScale，一个用于混合精度MoE量化的对比校准分配框架。它使用模型特定的幂变换，在保持排序不变的前提下，将基于梯度的专家权重校准到信息论对比度目标。我们联合评估每个专家的门控、上投影和下投影配置以估计其输出误差。然后，我们将这些误差与校准后的权重，同现有的基于权重谱、方差和精度衰减的先验相结合，在存储预算下选择配置。在三个模型上，NatScale取得了最低的（摘要在此处截断）

    arXiv:2610.20353v1 Announce Type: new  Abstract: Mixture-of-experts (MoE) language models activate only a few experts per token, yet all expert weights must be stored. Mixed-precision quantization reduces this storage cost by assigning higher bit-widths to sensitive experts and lower bit-widths to resilient ones, but expert rankings alone do not determine how strongly to weight their errors. To address this gap, we propose NatScale, a contrast-calibrated allocation framework for mixed-precision MoE quantization. It uses a model-specific power transformation to calibrate gradient-based expert weights to an information-theoretic contrast target while preserving rankings. We jointly evaluate each expert's gate, up and down configurations to estimate its output error. We then combine these errors and calibrated weights with an existing prior based on weight spectra, variance and precision decay to select configurations under a storage budget. Across three models, NatScale achieves the lowe
    
[^48]: 提升状态策略优化：通过转移几何内化自我纠错

    Lifted State Policy Optimization: Internalizing Self-Correction through Transition Geometry

    [https://arxiv.org/abs/2610.20350](https://arxiv.org/abs/2610.20350)

    LSPO将以能量函数刻画、经成本调整的“有缺陷状态向更优状态的局部转移”作为学习基本单元，通过奖励塑形和内部化损失将自我纠错内化进单次生成策略，在保持与测试时反思相当的准确率的同时显著降低延迟和token开销。

    

    偏好优化对最终答案进行排序，轨迹强化学习最大化总体回报，但两者都没有对将一个有缺陷的推理状态转变为更优状态的局部变化给予奖励。我们提出了提升状态策略优化（LSPO），它将这种有限的、经成本调整的转移确立为学习的基本单位。每个文本答案都被增广上一个学习得到的连续坐标，形成一个由能量函数进行评估和排序的提升状态。塑形奖励驱动能量在扣除编辑和步进成本后下降，内部化损失将选定的低能量答案吸收进直接生成器。在Qwen3.5-27B上，由此得到的单次通过策略在数学、科学、代码和逻辑基准上达到71.8%的总体准确率，与测试时反思的71.9%相当，同时延迟降低28%，token消耗减少25.8%。该模式在稠密和混合专家骨干网络上均成立，匹配对照实验证实……（摘要原文在此处被截断）

    arXiv:2610.20350v1 Announce Type: cross  Abstract: Preference optimization ranks final answers and trajectory RL maximizes aggregate returns, yet neither credits the local change that turns a flawed reasoning state into a better one. We introduce Lifted State Policy Optimization (LSPO), which establishes this finite, cost-adjusted transition as the primary unit of learning. Each textual answer is augmented with a learned continuous coordinate, forming a lifted state that an energy function evaluates and orders. A shaped reward drives energy descent net of edit and step costs, and an internalization loss absorbs selected low-energy answers into the direct generator. On Qwen3.5-27B, the resulting single-pass policy reaches 71.8% aggregate accuracy across mathematics, science, code, and logic benchmarks, matching test-time reflection at 71.9% with 28% lower latency and 25.8% fewer tokens. The pattern holds across dense and mixture-of-experts backbones, and matched controls confirm that tr
    
[^49]: NeuroDETR：面向异步脑电（EEG）事件检测的多尺度离散时空标记化

    NeuroDETR: Multiscale Discrete Spatiotemporal Tokenization for Asynchronous EEG Event Detection

    [https://arxiv.org/abs/2610.20349](https://arxiv.org/abs/2610.20349)

    NeuroDETR提出了一种融合多尺度时间码本、空间金字塔VQ和事件感知稀疏注意力模块FOCAL的离散时空多尺度框架，能够在无需预定义事件边界的情况下，从连续EEG记录中联合检测事件类别及其起止时间。

    

    异步脑电事件检测通过直接从连续记录中识别事件类别及其时间边界来支持实用的脑机接口应用，而无需在推理时预先定义事件边界。其主要挑战在于从持续进行的背景活动中提取与事件相关的模式。为解决这一问题，我们提出了NeuroDETR，一个将结构化标记化与事件感知特征细化相结合的离散时空多尺度框架。多尺度时间码本用于捕获短暂响应和持续动态，而空间金字塔VQ（Spatial-Pyramid VQ）则分层地编码EEG通道间的空间结构。这些互补的表示构成了一个紧凑的事件记忆库，用于联合分类事件并估计其起始和结束时间。在标记化之前，我们应用FOCAL——一个事件感知的稀疏注意力模块，它利用预测的事件存在性来引导时间特征细化……

    arXiv:2610.20349v1 Announce Type: new  Abstract: Asynchronous EEG event detection supports practical brain-computer interfaces by identifying event classes and temporal boundaries directly from continuous recordings without predefined event boundaries at inference time. The primary challenge lies in extracting event-related patterns embedded in ongoing background activity. To address this, we propose \textbf{NeuroDETR}, a discrete spatiotemporal multiscale framework that combines structured tokenization with event-aware feature refinement. Multiscale temporal codebooks capture brief responses and sustained dynamics, while Spatial-Pyramid VQ hierarchically encodes spatial structure across EEG channels. These complementary representations form a compact event memory for jointly classifying events and estimating their onsets and offsets. Before tokenization, we apply FOCAL, an event-aware sparse attention module that uses predicted event presence to guide temporal feature refinement and s
    
[^50]: 带选项的平均奖励Q学习的非渐近收敛性

    Non-asymptotic Convergence of Average-reward Q-learning with Options

    [https://arxiv.org/abs/2610.20347](https://arxiv.org/abs/2610.20347)

    该论文首次在均方意义下为inter-option和intra-option两种设置的平均奖励选项Q学习算法建立了 $\tilde{O}(1/\text{迭代次数})$ 的非渐近收敛速率保证。

    

    我们研究了选项式强化学习（RL）在平均奖励表述下Q学习算法的有限样本收敛性。选项允许暂时扩展的分层动作，而平均奖励表述则涵盖了超越折扣型和回合型强化学习的持续性任务。我们首次在均方意义下为平均奖励选项Q学习算法建立了 $\tilde{O}(1/\text{迭代次数})$ 的收敛保证，且同时适用于inter-option和intra-option两种设置。首先，我们证明了在单链假设下，inter-option和intra-option贝尔曼算子均满足多步跨度收缩性质。其次，我们提出了这些多步选项贝尔曼算子的同步估计器，由此得到选项Q学习的两种变体算法。最后，我们扩展了随机逼近框架，在文献中所用的最小假设条件下证明了这两种算法的收敛速率。

    arXiv:2610.20347v1 Announce Type: new  Abstract: We study the finite-sample convergence of Q-learning algorithms for the average-reward formulation of Reinforcement Learning (RL) with options. While options allow temporarily extended and hierarchical actions, the average-reward formulation incorporates continual tasks beyond discounted and episodic RL. We establish the first $\tilde{O}(1/\text{iterations})$ convergence guarantees, in the mean-square sense, for average-reward option Q-learning algorithms in both the inter- and intra-option settings. First, we prove that both inter- and intra-option Bellman operators satisfy multi-step span contractions under unichain assumptions. Then, we propose synchronous estimators of these multi-step option Bellman operators leading to two variants of option Q-learning. Finally, we extend the stochastic approximation framework to prove convergence rates of these two algorithms under the minimal assumptions used in the literature.
    
[^51]: 任务感知插补何时重要？训练-测试不匹配下的鲁棒性分析

    When Does Task-Aware Imputation Matter? An Analysis of Robustness Under Train-Test Mismatch

    [https://arxiv.org/abs/2610.20342](https://arxiv.org/abs/2610.20342)

    该论文提出将缺失数据插补与下游任务统一建模的框架，并在98个表格数据集上证明重构误差无法有效代表预测效用，任务感知的插补方法在训练-测试缺失机制不匹配时更具鲁棒性。

    

    缺失数据是机器学习中普遍存在的挑战，常常降低预测性能并使模型部署复杂化。大多数现有方法将插补视为独立于下游任务的预处理步骤，当训练与测试数据在缺失机制或缺失率上不匹配时，这种分离会带来问题。我们引入了一个统一框架，将缺失条目视为任务条件优化问题中的变量。我们在两种设置下实例化该框架：一种摊销方法适用于训练和测试时均存在缺失数据的场景，另一种潜在优化方法适用于仅训练数据不完整的设置。评估在98个表格数据集上进行，并在36个数据集上评估了训练-测试不匹配下的鲁棒性。结果表明，重构误差并不能很好地代理预测效用（Spearman rho 约为 0.16 到 0.32），而任务感知……

    arXiv:2610.20342v1 Announce Type: new  Abstract: Missing data is a pervasive challenge in machine learning, often degrading predictive performance and complicating model deployment. Most existing approaches treat imputation as a preprocessing step independent of the downstream task, a separation that becomes problematic under train-test mismatches in missingness mechanisms or rates. We introduce a unified framework that treats missing entries as variables in a task-conditioned optimization problem. We instantiate this framework in two settings: an amortized approach for scenarios with missing data at both training and test time, and a latent optimization approach for settings where only training data is incomplete.   The evaluation is conducted on 98 tabular datasets, and robustness under train-test mismatches is assessed on 36 datasets. Results reveal that reconstruction error constitutes a poor proxy for predictive utility (Spearman rho approximately 0.16 to 0.32), and that task-awar
    
[^52]: 无界输入下Transformer的训练动力学：注意力学习可能滞后于输出层学习

    Training Dynamics of Transformers on Unbounded Inputs: Attention Learning Can Lag Output Layer Learning

    [https://arxiv.org/abs/2610.20337](https://arxiv.org/abs/2610.20337)

    该论文证明了在无界输入上训练浅层Transformer识别无星语言时，注意力学习会滞后于输出层学习，需要采用“先更新输出层、再更新注意力、最后微调输出层”的三阶段梯度下降策略才能成功训练。

    

    Transformer的成功归因于多个因素，包括其在变长输入上表达形式语言的能力，以及注意力机制聚焦相关输入特征的能力。然而，对于浅层Transformer的训练动力学，尤其是如何解耦注意力与输出（前馈）层的学习，目前理解仍然有限。前馈网络中单步梯度特征学习理论的一个自然延伸表明，可以先通过单步梯度学习注意力权重，然后再训练输出层。我们证明，对于训练用于从无界长度输入中识别无星语言的浅层Transformer而言，这种做法是失败的。我们证明了该Transformer可以通过三阶段梯度下降成功训练以识别该语言：首先更新输出层，随后进行注意力更新，最后对输出层进行微调。

    arXiv:2610.20337v1 Announce Type: new  Abstract: The success of transformers is attributed to several factors, including their capacity to express formal languages over variable-length inputs, and ability of the attention mechanism to focus on relevant input features. However, there is limited understanding of the training dynamics of shallow transformers, specifically how to decouple the learning of the attention and the output (feedforward) layer. A natural extension of the theory of one-step gradient feature learning in feedforward networks suggests that one could first learn the attention weights with a single gradient step, and then train the output layer. We show that this fails for a shallow transformer trained to recognize a star-free language from unbounded-length input. We prove that the transformer can be trained to recognize the language with a three-phase gradient descent: the output layer is updated first, followed by an attention update, and the output layer is fine-tune
    
[^53]: 我们是否把单步生成想得太复杂了？

    Are We Overthinking One-Step Generation?

    [https://arxiv.org/abs/2610.20331](https://arxiv.org/abs/2610.20331)

    本文提出Priming Models两阶段框架：先用简单的重建目标（如固定噪声水平的像素去噪）预训练，再进行特征空间分布后训练，并发现这一简单的重建阶段即可完全解决单步生成器分布训练的困难。

    

    扩散模型和流匹配如今已成为连续域生成建模的主流框架，但它们在推理时通常需要多次网络评估。这推动了大量基于扩散和流的单步生成建模研究，然而现有方法往往依赖复杂的训练机制。在本工作中，我们提出了Priming Models（预启动模型），这是一个用于训练单步生成器的简单两阶段框架。鉴于从零开始进行分布训练的困难，Priming Models首先以重建目标进行训练（例如在固定噪声水平下进行像素空间去噪），然后再进行特征空间的分布后训练。我们的核心科学发现是：这一简单的重建阶段本身就足以完全消除分布训练的困难。实证结果表明，该方法在类条件图像生成、文本到图像生成和世界模型等任务上均有效。

    arXiv:2610.20331v1 Announce Type: new  Abstract: Diffusion and flow matching are now dominant frameworks for generative modeling in continuous domains, but they typically require multiple network evaluations at inference. This has motivated extensive research on one-step generative modeling primarily built on diffusion and flow, though existing approaches often rely on complex training machinery. In this work, we introduce Priming Models, a simple two-stage framework for training one-step generators. Motivated by the difficulty of distributional training from scratch, Priming Models are first trained on a reconstruction objective, e.g., pixel-space denoising at a fixed noise level, before undergoing feature-space distributional post-training. Our central scientific finding is that this simple reconstruction stage is already enough to entirely alleviate the difficulty of distributional training. Empirically, across class-conditional image generation, text-to-image generation, and world 
    
[^54]: 面向实时ISAC波束成形的基于深度学习的DoA估计器比较评估

    Comparative Evaluation of Deep-Learning based DoA Estimators for Real-Time ISAC Beamforming

    [https://arxiv.org/abs/2610.20326](https://arxiv.org/abs/2610.20326)

    本文提出一个统一评估框架，将多种经典与深度学习DoA估计器嵌入相同的基于SVD的ISAC波束成形流水线中，在相同条件下系统比较其估计精度与计算延迟，以评估其实时部署的适用性。

    

    集成感知与通信系统依赖于精确的到达方向（DoA）估计来实现有效的波束成形和频谱效率提升。尽管文献中已报道了大量经典的以及基于深度学习的DoA估计方法，但它们在实时ISAC运行中的适用性仍缺乏充分认识，因为实际部署需要同时考虑估计精度和计算延迟。本文提出了一个统一的评估框架，用于在ISAC波束成形流水线中系统地比较经典方法和基于学习的DoA估计器。文中将具有代表性的经典与基于学习的DoA估计器（包括一种集成架构）与基于奇异值分解（SVD）的波束成形相结合，并在完全相同的运行条件下进行评估。研究分析了DoA估计精度、推理延迟、波束成形权重生成时间以及（摘要在此处截断）

    arXiv:2610.20326v1 Announce Type: new  Abstract: Integrated Sensing and Communication (ISAC) systems rely on accurate Direction-of-Arrival (DoA) estimation for effective beamforming and spectral-efficiency enhancement. While numerous classical and deep-learning-based DoA estimation methods have been reported in the literature, their suitability for real-time ISAC operation remains insufficiently understood because practical deployment requires consideration of both estimation accuracy and computational latency. This paper presents a unified evaluation framework for systematically comparing classical and learning-based DoA estimators within an ISAC beamforming pipeline. Representative classical and learning-based DoA estimators, together with an ensemble architecture, are integrated with Singular Value Decomposition (SVD)-based beamforming and evaluated under identical operating conditions. The impact of DoA estimation accuracy, inference latency, beamforming-weight generation time, and
    
[^55]: 面向机器人策略的高效测试时自适应

    Efficient Test-Time Adaptation For Robot Policies

    [https://arxiv.org/abs/2610.20318](https://arxiv.org/abs/2610.20318)

    该论文提出TEMPO方法，通过熵正则化与基于分数的记忆机制，在无需奖励信号的条件下实现机器人运动策略在分布偏移环境下稳定的在线测试时自适应。

    

    用于机器人运动的强化学习策略在部署时可能因运行条件变化而失效（例如崎岖或湿滑地形、载荷变化、传感性能退化），而在设备端延迟与内存限制下，重新训练或手动重新校准往往不可行。我们研究面向机器人控制的测试时自适应（TTA）：在执行过程中以在线、自监督的方式适配预训练策略，且无需访问奖励信号。我们的起点是一个轻量级基线方法，仅更新执行器（actor），并使用冻结的预训练评论家（critic）作为部署时的性能替代指标。在此基础上，我们提出了TEMPO——一种测试时熵正则化、记忆引导的策略优化方法，能够在严重分布偏移下稳定评论家驱动的更新。TEMPO结合了：(i) 对评论家预测分布施加的促进熵的正则化项，以缓解过度自信的外推；(ii) 基于分数的记忆机制……（摘要内容在此处截断）

    arXiv:2610.20318v1 Announce Type: cross  Abstract: Reinforcement-learning policies for robot locomotion can fail at deployment when operating conditions shift (e.g., rough or slippery terrain, payload changes, sensing degradation), and retraining or manual recalibration is often infeasible under on-device latency and memory constraints. We study test-time adaptation (TTA) for robotic control: adapting a pretrained policy online and in a self-supervised manner during execution, without access to rewards. Our starting point is a lightweight baseline that updates only the actor using a frozen pretrained critic as a deployment-time performance surrogate. Building on this, we introduce TEMPO, a test-time entropy-regularized, memory-guided policy optimization method that stabilizes critic-driven updates under severe shift. TEMPO combines (i) an entropy-promoting regularizer on the critic's predictive distribution to mitigate overconfident extrapolation, and (ii) a score-based memory, inspire
    
[^56]: 从单一流场实现出发的均匀湍流极端尺度超分辨率重建

    Extreme-Scale Super-Resolution of Homogeneous Turbulence from a Single Realization

    [https://arxiv.org/abs/2610.20305](https://arxiv.org/abs/2610.20305)

    该研究提出一种极端超分辨率框架，仅利用单个高分辨率湍流实现训练WDSR网络，便能在32倍和64倍的严重欠分辨率下准确重建均匀湍流的相干涡结构与细尺度特征。

    

    均匀湍流在空间上相互分离的区域中会呈现出重复出现的多尺度结构，这表明单一流场实现可能包含足够的信息来学习未解析尺度的表示。我们通过一个极端超分辨率框架来验证这一假设：在该框架中，一个宽激活深度超分辨率（WDSR）网络仅使用从单个高分辨率湍流实现中提取的重叠小块进行训练。对于先前未见过的流场，则通过逐块预测并采用汉宁窗加权的重叠相加方法进行拼接重建。该框架在二维均匀各向同性湍流上进行了评估，超分辨率倍数达到32倍和64倍，属于严重的空间欠分辨率情形。尽管训练数据有限，所提出的方法仍能重建相干的涡旋结构，并且比传统的插值方法更准确地恢复细尺度特征。

    arXiv:2610.20305v1 Announce Type: cross  Abstract: Homogeneous turbulent flows exhibit recurring multiscale structures across spatially separated regions, suggesting that a single flow realization may contain sufficient information to learn representations of unresolved scales. We investigate this hypothesis through an extreme super-resolution framework in which a wide-activation deep super-resolution (WDSR) network is trained exclusively on overlapping patches extracted from a single high-resolution turbulent realization. Previously unseen flow fields are reconstructed through patchwise prediction followed by Hann-window-weighted overlap-add assembly. The framework is evaluated for two-dimensional homogeneous isotropic turbulence at super-resolution factors of x32 and x64, representing severe spatial under-resolution. Despite the limited training data, the proposed approach reconstructs coherent vortical structures and recovers fine-scale features more accurately than conventional int
    
[^57]: SpectrumQuant：面向大语言模型量化的频谱对齐方法

    SpectrumQuant: Spectrum Alignment for LLM Quantization

    [https://arxiv.org/abs/2610.20303](https://arxiv.org/abs/2610.20303)

    提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。

    

    正交变换已成为大语言模型（LLM）低比特量化的有效工具，因为它能够在不改变底层模型函数的情况下重塑激活分布。现有方法通常依赖于固定的、与数据无关的变换（如Hadamard旋转），或者依赖于在校准阶段对可学习旋转的优化。我们从第一性原理出发研究旋转选择问题，并证明最大化协方差可以带来频谱对齐，从而为构建量化友好的正交变换提供了一个有理论依据的准则。基于这一洞察，我们提出了SpectrumQuant，这是一种闭式解的旋转方法，它从校准数据中估计激活协方差，并利用它构建协方差感知的正交变换以用于训练后量化。我们的公式化将量化友好的旋转刻画为对齐分量与混合分量的组合，同时将Hadamard旋转作为特例恢复出来（此句原文被截断）。

    arXiv:2610.20303v1 Announce Type: new  Abstract: Orthogonal transforms have emerged as an effective tool for low-bit quantization of large language models (LLMs), as they reshape activation distributions without changing the underlying model function. Existing approaches typically rely either on fixed data-agnostic transforms, such as Hadamard rotations, or on calibration-time optimization of learned rotations. We study rotation selection from first principles and show that maximizing covariance leads to spectral alignment and provides a principled criterion for constructing quantization-friendly orthogonal transforms. Based on this insight, we propose SpectrumQuant, a closed-form rotation method that estimates activation covariance from calibration data and uses it to construct a covariance-aware orthogonal transform for post-training quantization. Our formulation characterizes quantization-friendly rotations as compositions of alignment and mixing components, while recovering Hadamar
    
[^58]: MMLA-RSI：通过RTT和RTMA实现的递归自我改进

    MMLA-RSI: Recursive Self-Improvement through RTT and RTMA

    [https://arxiv.org/abs/2610.20294](https://arxiv.org/abs/2610.20294)

    提出了一种递归自我改进方案MMLA-RSI，其中RTT改变策略状态、RTMA改变权威记忆，且这两种改进方法本身也可作为改进对象，由独立接纳的后继者在下一代中接管相同角色，从而实现智能体行为的持续递归改进。

    

    智能体可以通过改变其行为方式和所保留的经验，在未解决的问题中改进自身的行为。我们提出MMLA-RSI，这是一种在记忆中介学习架构中具有两种实现的递归自我改进方案：推理时训练（RTT）改变策略状态 $\Phi$，而推理时记忆适配（RTMA）改变权威记忆 $M$。每条路线还将其适配方法本身作为改进的对象。当前的数值更新器训练提出其后继者的策略；当前的记忆管理器选择可供提出其后继者的进程所使用的记录。被独立接纳的后继者在下一代中接管相同的角色。我们给出了有界可执行规则语言、共享的任务/提案接口，以及适用于任一路线或其组合的单一算法。RTT实例在有界动量更新中改变标量调度；RTMA实例选择NULL或一个co（摘要在此处被截断）

    arXiv:2610.20294v1 Announce Type: new  Abstract: An agent can improve its behavior within an unresolved problem by changing how it acts and which experience it retains. We propose MMLA-RSI, a recursive self-improvement scheme with two implementations in Memory-Mediated Learning Architecture: Reasoning-Time Training (RTT) changes the policy state $\Phi$, while Reasoning-Time Memory Adaptation (RTMA) changes authoritative memory $M$. Each route also makes its adaptation method an object of improvement. A current numerical updater trains the policy proposing its successor; a current memory manager chooses the records available to the process proposing its successor. An independently admitted successor takes over the same role in the next generation. We give bounded executable rule languages, shared task/proposal interfaces, and one algorithm for either route or their composition. The RTT instance changes scalar schedules in a clipped-momentum update. The RTMA instance chooses NULL or a co
    
[^59]: CASI：基于集合干预的智能体检索信用分配

    CASI: Credit Assignment via Set Intervention for Agentic Retrieval

    [https://arxiv.org/abs/2610.20293](https://arxiv.org/abs/2610.20293)

    该论文提出CASI方法，通过基于快照的集合干预构建反事实信息并结合Shapley聚合，解决了智能体检索中多个联合生成的检索动作的集合内信用分配问题，从而提升策略优化效果。

    

    大型语言模型越来越多地联合生成多个检索动作，而下游系统通常只对最终结果集合提供一个奖励。广播这一奖励虽然能为期望的集合级目标提供有效的得分函数梯度估计器，但无法确定依赖于联盟结构的集合内归因。我们提出了CASI（Credit Assignment via Set Intervention for Agentic Retrieval，基于集合干预的智能体检索信用分配），它通过基于快照的集合干预来构建缺失的反事实信息：只需生成并检索一次，保持可重放的快照组件固定，并在共同条件下评估不同的替代子集。Shapley聚合将这些边际贡献转化为集合内信用，并通过有界零和优势整形将其融入策略优化。当 m >= 4 时，单例、留一和全集端点通常不足以确定Shapley分配；在集合值可加的情况下，额外的联盟……（原文摘要在此处截断）

    arXiv:2610.20293v1 Announce Type: cross  Abstract: Large language models increasingly generate multiple retrieval actions jointly, while downstream systems often provide only one reward for the resulting set. Broadcasting this reward provides a valid score-function gradient estimator for the expected set-level objective, but does not determine a coalition-dependent within-set attribution. We introduce CASI (Credit Assignment via Set Intervention for Agentic Retrieval), which constructs the missing counterfactual information through snapshot-grounded set intervention: generate and retrieve once, hold replayable snapshot components fixed, and evaluate alternative subsets under common conditions. Shapley aggregation converts these marginals into within-set credit, which bounded zero-sum advantage shaping incorporates into policy optimization. For m >= 4, singleton, leave-one-out, and full-set endpoints do not generally determine the Shapley allocation; under additive set values, extra coa
    
[^60]: 预测未见过的科学交互作用：条件流模型会幻觉出非加性效应

    Predicting Unseen Scientific Interactions: Conditional Flow Models Hallucinate Non-Additive Effects

    [https://arxiv.org/abs/2610.20288](https://arxiv.org/abs/2610.20288)

    本研究揭示条件流匹配模型学习个体效应远早于学习交互作用，并会在不存在交互作用时幻觉出非加性效应，导致模型整体看似准确却无法恢复真正的科学交互效应。

    

    条件生成模型越来越多地被用于超出直接训练支持范围的科学预测。然而，科学问题往往关注的是一种干预是否会改变另一种干预的效果。这类交互作用可能是核心研究目标，却在总变异中仅占很小比例，这使得模型可能在整体上表现准确，却未能恢复出相关的科学效应。目前尚不清楚这些模型如何学习交互作用，以及它们何时能够成功外推到未见过的条件。因此，我们利用生理时间序列模拟器、药物剂量-反应数据、组合细胞扰动以及分子构象生成，研究流匹配模型中的交互作用恢复问题，并将真实的交互作用恢复与幻觉出的交互结构区分开来。在受控环境中，个体效应的学习远早于交互作用，且模型会在不存在交互作用的地方幻觉出交互结构。在各种设置中，我们……

    arXiv:2610.20288v1 Announce Type: new  Abstract: Conditional generative models are increasingly used for scientific prediction beyond the direct training support. Yet scientific questions often concern whether one intervention changes the effect of another. Such interactions can be of primary interest while explaining little of the total variation, allowing models to appear accurate overall without recovering the relevant scientific effect. It remains unclear how these models learn interactions and when these successfully extrapolate to unseen conditions. We therefore study interaction recovery in flow-matching models using a physiological time-series simulator, drug dose-response data, combinatorial cell perturbations, and molecular conformer generations, where we separate recovery from hallucinated interaction structure. In the controlled setting, individual effects are learned much earlier than interactions and models hallucinate interactions where none exist. Across settings, we fi
    
[^61]: 从柏拉图到亚里士多德：迈向表征假设的通用理论

    From Plato to Aristotle: Toward a General Theory of Representation Hypotheses

    [https://arxiv.org/abs/2610.20286](https://arxiv.org/abs/2610.20286)

    该论文提出格表征假设（LRH），将各种表征对齐准则统一到同一个格结构中，从而解释了柏拉图表征假设的收敛声明与其实际证据之间的差异，并提出亚里士多德表征假设将收敛限定于局部邻域。

    

    柏拉图表征假设（PRH）认为，能力日益增强的模型会收敛到一种共同的表征。然而，最近的研究通过邻域、生成树和距离等方式对齐表征时呈现出不同的观点。我们证明这些观点之所以存在差异，是因为它们背后的对齐准则位于同一个格的不同位置，并将这一解释形式化为格表征假设（LRH）。在LRH框架下，PRH所声称的收敛对应于格顶部的共识，而其实际证据则来自邻域重叠——这是一种接近格底部的准则。亚里士多德表征假设（ARH）将收敛限制在局部邻域，因为校准保留了邻域重叠随模型规模增长的趋势，却在很大程度上消除了中心化核相关性（CKA）的增长。我们证明这种消除可能源于对层间最大值的校准，因此全局收敛……

    arXiv:2610.20286v1 Announce Type: new  Abstract: The Platonic Representation Hypothesis (PRH) holds that increasingly capable models converge toward a common representation. However, recent studies have shown different views on aligning representations via their neighborhoods, spanning trees, and distances. We show that these views can differ because the alignment criteria behind them sit at different points of one lattice, and we formalize this explanation as the Lattice Representation Hypothesis (LRH). Under the LRH, the convergence claimed by the PRH is agreement at the top of the lattice, whereas its evidence comes from neighbor overlap, a criterion near the bottom. The Aristotelian Representation Hypothesis (ARH) restricts convergence to local neighborhoods because calibration keeps the growth of neighbor overlap with model size but largely removes that of centered kernel alignment. We show that this removal can come from calibrating a maximum over layers, so global convergence re
    
[^62]: TANGO：RNA拓扑与几何协同设计

    TANGO: RNA Topology and Geometry Co-Design

    [https://arxiv.org/abs/2610.20277](https://arxiv.org/abs/2610.20277)

    提出TANGO框架，首次将RNA逆设计表述为拓扑与几何协同设计问题，通过多目标多智能体强化学习和分阶段课程训练（二级结构拓扑感知策略学习、三级结构迁移训练、前沿引导微调），实现同时满足二级碱基配对拓扑与三级几何结构的RNA序列设计。

    

    设计满足目标三级结构要求的RNA序列是RNA工程中的一个重要问题，在合成生物学和治疗学领域有着广泛的应用。现有的3D RNA设计方法已取得长足进展，但它们在很大程度上将目标折叠视为纯几何目标。这对RNA而言是不完整的，因为RNA的功能折叠不仅由三级几何结构决定，还由二级碱基配对拓扑决定。因此，我们将RNA逆设计表述为拓扑与几何协同设计问题，并提出了TANGO——一个面向该任务的多目标多智能体强化学习框架。TANGO采用分阶段课程学习：首先从二级结构目标中学习拓扑感知策略，然后将该策略迁移到三级结构目标上训练一个通用的协同设计模型，最后应用前沿引导的微调来细化特定目标下二级结构保真度与三级结构几何之间的权衡。

    arXiv:2610.20277v1 Announce Type: new  Abstract: Designing RNA sequences that satisfy target tertiary-structure objectives is an important problem in RNA engineering, with broad applications in synthetic biology and therapeutics. Existing 3D RNA design methods have made substantial progress, but they largely treat the target fold as a geometry-only objective. This is incomplete for RNA, whose functional folds are specified not only by tertiary geometry but also by secondary base-pairing topology. We therefore formulate RNA inverse design as a topology-and-geometry co-design problem and propose TANGO, a multi-objective multi-agent reinforcement learning framework for this task. TANGO adopts a staged curriculum: it first learns a topology-aware policy from secondary-structure targets, then transfers this policy to tertiary-structure targets to train a general co-design model, and finally applies frontier-guided fine-tuning to refine target-specific trade-offs among secondary-structure fi
    
[^63]: 基于可学习缩放与谱域滤波器参数化的分辨率分配

    Resolution Allocation with Learnable Resizing and Spectral Filter Parameterization

    [https://arxiv.org/abs/2610.20274](https://arxiv.org/abs/2610.20274)

    该论文首次系统研究了可学习下采样与谱域滤波器参数化之间的相互作用，发现谱域卷积核比空间域卷积核从可学习分辨率分配中获得更大的准确率增益，且这一优势在匹配总保留面积后依然存在。

    

    可学习的下采样与谱域滤波器参数化各自都使卷积网络更加灵活，但两者之间的相互作用尚未被系统性研究。我们通过考察在空间域和谱域中学习到的卷积核，以及在裁剪缩放实现下的可学习下采样比例，来弥补这一研究空白。我们发现谱域核与空间域核对可学习分辨率分配的响应不同，谱域核获得了更大的增益。我们在三个数据集（EuroSAT、CIFAR-10 和 STL-10）以及两种模型规模上一致地观察到这一现象（在保留测试集上准确率增益差异为 +0.013 至 +0.028）。这种差异部分反映了谱域核从保留更多空间区域中所获得的更大增益，但当总保留面积相同时，该差异仍然为正。这一模式在所有数据集和模型规模上均可观察到，尽管其幅度及依赖关系……

    arXiv:2610.20274v1 Announce Type: new  Abstract: Learnable downsampling and spectral-domain filter parameterizations have each made convolutional networks more flexible, but their interaction has not been systematically examined. We bridge this gap by examining convolutional kernels learned in the spatial and spectral domains, together with learned downsampling ratios under a cropped resizing implementation. We find that spectral and spatial kernels respond differently to learned resolution allocation, with larger gains for the spectral kernel. We observe this consistently on three datasets (EuroSAT, CIFAR-10 and STL-10) and at two model sizes (differences in accuracy gain of +0.013 to +0.028 on held-out test sets). Part of this difference reflects the greater gain of the spectral kernel from retaining more spatial area, but the difference remains positive when total retained area is matched. This pattern is observed across datasets and model sizes, although its magnitude and dependenc
    
[^64]: FlowSep-RF：用于单信道射频信号分离的整流流匹配

    FlowSep-RF: Rectified Flow Matching for Single-Channel Radio-Frequency Signal Separation

    [https://arxiv.org/abs/2610.20269](https://arxiv.org/abs/2610.20269)

    提出FlowSep-RF整流流匹配框架，通过融合时域IQ采样与频域特征、仅需一次端点速度评估即可完成单信道射频信号分离，在多种干扰场景下取得最低误码率和均方误差。

    

    单信道射频（RF）信号分离对于干扰抑制至关重要，尤其当目标信号与同信道干扰在时域和频域上发生重叠时。经典的匹配滤波和LMMSE接收机在结构化非高斯干扰下性能受限，而基于扩散和分数的方法通常需要迭代采样或优化。我们提出FlowSep-RF，这是一种整流流匹配框架，它学习从接收到的混合信号到目标信号的干净端点传输。该模型将时域IQ采样与频域特征相结合，仅需一次端点速度评估即可恢复分离出的波形。我们在包含OFDM-BPSK、OFDM-QPSK和RFChallenge CommSignal2干扰的QPSK混合信号上，将FlowSep-RF与经典方法、判别式方法和生成式基线进行了对比评估。在所有三种设置下，FlowSep-RF在所有被评估的基线方法中均取得了最低的平均误码率（BER）和均方误差（MSE）。

    arXiv:2610.20269v1 Announce Type: cross  Abstract: Single-channel radio-frequency (RF) signal separation is important for interference mitigation when a signal of interest overlaps with a co-channel interferer in time and frequency. Classical matched filtering and LMMSE receivers are limited under structured non-Gaussian interference, while diffusion and score-based methods often require iterative sampling or optimization. We propose FlowSep-RF, a rectified flow matching framework that learns a clean-endpoint transport from the received mixture to the signal of interest. The model combines time-domain IQ samples with frequency-domain features and recovers the separated waveform using one endpoint velocity evaluation. We evaluate FlowSep-RF on QPSK mixtures with OFDM-BPSK, OFDM-QPSK, and RFChallenge CommSignal2 interference against classical, discriminative, and generative baselines. FlowSep-RF achieves the lowest average BER and MSE among evaluated baselines across all three settings w
    
[^65]: FUSE：面向 q-公平联邦学习中抗拜占庭利用的公平性感知更新筛选

    FUSE: Fairness-aware Update Screening against Byzantine Exploitation in $q$-Fair Federated Learning

    [https://arxiv.org/abs/2610.20264](https://arxiv.org/abs/2610.20264)

    该论文提出了FUSE方法，首次在服务器仅能观察公平性缩放更新的“仅更新接口”下形式化了拜占庭鲁棒的q-公平联邦学习问题，通过公平性感知的更新筛选机制，使服务器能够区分合法的公平性放大与恶意的拜占庭攻击。

    

    客户端级公平联邦学习，以 q-公平联邦学习、不可知联邦学习或倾斜经验风险最小化等形式出现，已成为均衡异构客户端间效用的有吸引力的解决方案。该范式赋予高损失客户端更大的相对影响力，同时由中央服务器将其按损失加权的更新聚合为一个共享模型。然而，q-公平联邦学习的拜占庭鲁棒性在很大程度上被忽视了，导致服务器无法区分合法的由公平性引起的更新放大与恶意的更新缩放。为了填补这一关键空白，我们在“仅更新接口”下形式化了拜占庭鲁棒的 q-公平联邦学习问题，在该接口下，服务器只能观察到按公平性缩放后的客户端更新。我们在共享类条件标签偏斜情形下的分析表明，公平性缩放保持了非负类混合结构……

    arXiv:2610.20264v1 Announce Type: new  Abstract: Client-level fair federated learning, in the form of $q$-fair federated learning ($q$-FFL), agnostic federated learning, or tilted empirical risk minimization, has emerged as an attractive solution to equalize utility across heterogeneous clients. This paradigm grants high-loss clients greater relative influence, while a central server aggregates their loss-weighted updates into a single shared model. Yet, the Byzantine robustness of $q$-fair federated learning has been largely neglected, leaving the server unable to tell legitimate fairness-induced amplification from malicious update scaling. To address this critical gap, we formalize the problem of Byzantine-robust $q$-fair federated learning under an \emph{update-only interface}, in which the server observes nothing but the fairness-scaled client updates. Our analysis under shared class-conditional label skew shows that fairness scaling preserves the nonnegative class-mixture structur
    
[^66]: 框架策略蒸馏：使智能体策略与运行时语义对齐

    Harness Policy Distillation: Aligning Agent Policies with Runtime Semantics

    [https://arxiv.org/abs/2610.20261](https://arxiv.org/abs/2610.20261)

    提出框架策略蒸馏（HPD）：通过在运行时框架中重放参考轨迹，将普通轨迹数据中隐含的运行时控制语义恢复为模型调用边界处的控制文档并作为显式训练目标，再借助框架验证保证目标可执行，从而实现智能体策略与运行时语义的对齐。

    

    将通用模型部署为智能体，意味着要在特定产品的运行时框架中运行它，该框架决定哪些动作是可执行的、状态如何推进、以及循环何时可以终止。普通的轨迹监督只显示采取了哪个动作，却将该动作背后的运行时语义保持在隐式状态。我们提出框架策略蒸馏，它将这些运行时控制语义与动作一起作为显式的训练目标，以对齐框架策略。HPD 通过框架重放参考轨迹，以恢复普通轨迹数据中隐含的运行时语义，并将其表示为每个模型调用边界处的控制文档。这些文档为教师模型生成动作提供了特权信息，同时框架验证确保由此产生的目标在运行时契约下是可执行的。我们在恢复的运行时语义上联合训练共享模型……（原文截断）

    arXiv:2610.20261v1 Announce Type: new  Abstract: Deploying a general-purpose model as an agent means running it inside a product-specific runtime harness that decides which actions are executable, how state advances, and when the loop may terminate. Ordinary trajectory supervision shows which action was taken but leaves the runtime semantics behind that action implicit. We introduce Harness Policy Distillation (HPD), which makes these runtime control semantics an explicit training target alongside actions to align the harness policy. HPD replays reference trajectories through the harness to recover the runtime semantics that are implicit in ordinary trajectory data, and represents them as control documents at each model-call boundary. These documents provide privileged information for a teacher to generate actions, while harness verification ensures that the resulting targets are executable under the runtime contract. We jointly train a shared model on the runtime semantics recovered f
    
[^67]: 长时程强化学习需要 Muon

    Long-Horizon RL Needs Muon

    [https://arxiv.org/abs/2610.20256](https://arxiv.org/abs/2610.20256)

    提出了一种结合 Muon 优化器、令牌平均损失、截断重要性采样及长度与工具错误惩罚的单回合异步训练方案，显著提升了长时程强化学习在仓库级问题解决任务上的训练成功率。

    

    长时程智能体需要在长时间交互中协调推理、工具使用与环境反馈。从稀疏的结果奖励中学习，要求对不同长度的轨迹进行加权，并在回合数据仍在收集时就更新策略。我们研究了一种单回合异步训练方案，它结合了 Muon 优化器、令牌平均损失、截断重要性采样，以及辅助性的长度惩罚和工具错误惩罚。单回合采样使异步学习无需等待针对同一问题收集的多条轨迹。我们在两个模型家族上，针对仓库级问题解决任务研究了优化器选择和方案各组件的作用。评估的训练运行结果表明，完整方案比基线方法取得了更高的训练成功率。

    arXiv:2610.20256v1 Announce Type: new  Abstract: Long-horizon agents must coordinate reasoning, tool use, and environment feedback over extended interactions. Learning from sparse outcome rewards requires weighting trajectories of different lengths and updating the policy while rollouts are still being collected. We examine a single-rollout asynchronous recipe combining Muon, token-averaged loss, truncated importance sampling, and auxiliary length and tool-error penalties. Single-rollout sampling allows asynchronous learning without waiting for multiple trajectories for the same issue. We study optimizer choices and recipe components for repository-level issue resolution across two model families. The evaluated training runs show higher training success with the full recipe than with the baselines.
    
[^68]: 用于鲁棒行星着陆的对抗性强化学习

    Adversarial Reinforcement Learning for Robust Planetary Landing

    [https://arxiv.org/abs/2610.20252](https://arxiv.org/abs/2610.20252)

    该论文提出一种三阶段对抗性强化学习训练框架，由辅助智能体生成具有时空相关性且满足物理约束的最坏情况扰动，与着陆控制器形成非对称序贯零和博弈并交替优化，从而显著提升行星着陆策略应对非标称条件的鲁棒性。

    

    用于行星着陆以及其他航空航天制导、导航与控制任务的深度强化学习（DRL）策略，通常在随机化扰动下进行训练以提高鲁棒性。然而，典型的随机扰动采样假设各扰动变化相互独立，而实际任务中的扰动往往表现出空间与时间上的相关性，无法通过独立的随机抽样来复现。本工作研究对抗性强化学习，以发现具有挑战性的非标称条件。一个辅助智能体生成物理受限的扰动来降低着陆控制器的性能，从而形成非对称的序贯零和博弈。训练分为三个阶段进行：基线控制器学习、借助随机扰动进行鲁棒性增强，以及控制器与扰动生成器交替优化的迭代对抗阶段。在……中的初步实验（原文在此处截断）

    arXiv:2610.20252v1 Announce Type: new  Abstract: Deep reinforcement learning (DRL) policies for planetary landing and other aerospace guidance, navigation, and control tasks are commonly trained under randomized disturbances to improve robustness. However, typical stochastic perturbation sampling assumes independent variations, whereas real mission disturbances often exhibit spatial and temporal correlation and cannot be reproduced through independent random draws. This work investigates adversarial reinforcement learning to discover challenging off-nominal conditions. A secondary agent generates physically constrained perturbations that degrade the performance of a landing controller, forming an asymmetric sequential zero-sum game. Training proceeds in three stages: baseline controller learning, robustness enhancement with stochastic perturbations, and an iterative adversarial phase in which the controller and disturbance generator are alternately optimized. Preliminary experiments in
    
[^69]: 基于对比不确定性学习的下盆腔MRI鲁棒多结构分割

    Contrastive uncertainty learning for robust multi-structure segmentation in lower-pelvic MR

    [https://arxiv.org/abs/2610.20251](https://arxiv.org/abs/2610.20251)

    该论文提出对比不确定性学习框架，利用蒙特卡洛Dropout产生的体素级不确定性直接指导特征表示学习，通过排斥高不确定性特征防止边界处特征塌缩，从而提升前列腺MRI八结构分割在模糊区域的鲁棒性。

    

    前列腺MRI中解剖结构的边界勾勒常常受到区域模糊性的影响。依赖全局重叠损失函数的自动分割方法往往对表面偏差不敏感，导致在模糊区域出现显著误差。现有的大多数不确定性感知方法将不确定性视为事后质量指标或预测过滤器。我们提出了对比不确定性学习，这是一个监督框架，利用预测不确定性直接指导表示学习。CUL使用蒙特卡洛Dropout识别体素级不确定性，并据此构建嵌入几何结构：低不确定性且正确预测的特征形成稳定体素，而高不确定性或错误分类的特征则被主动排斥，从而防止在困难边界处发生特征塌缩。我们在一个包含589例扫描的3D前列腺MRI多结构分割任务（8个解剖结构）上对CUL进行了评估。

    arXiv:2610.20251v1 Announce Type: cross  Abstract: Boundary delineation of anatomical structures in prostate MRI is often compromised by regional ambiguity. Automated segmentation methods relying on global overlap loss functions tend to be insensitive to surface deviations, leading to significant errors in ambiguous regions. Most existing uncertainty-aware methods treat uncertainty as a post-hoc quality indicator or prediction filter. We propose Contrastive Uncertainty Learning (CUL), a supervised framework that uses predictive uncertainty to directly guide representation learning. Using Monte Carlo Dropout, CUL identifies voxel-wise uncertainty and structures the embedding geometry accordingly: low-uncertainty, correctly predicted features form stable voxels while high-uncertainty or misclassified features are actively repelled, preventing feature collapse at difficult boundaries. We evaluate CUL on a 589-scan 3D prostate MRI multi-structure segmentation task (8 anatomical structures)
    
[^70]: 扩展多语言专家混合模型：语言覆盖范围如何改变容量与计算分配

    Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation

    [https://arxiv.org/abs/2610.20247](https://arxiv.org/abs/2610.20247)

    本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。

    

    多语言模型在各语言之间共享容量，因此增加语言数量可能会损害现有语言的表现。专家混合模型只需增加极少量的每词元计算即可扩展这一容量。鉴于这种额外容量，增加专家数量能否降低每种新增语言的代价？为回答这一问题，我们提出了一种感知重复的联合缩放定律，将活跃参数数量、每种语言的训练词元数量、专家数量与语言覆盖范围联系起来。该定律基于来自64个预训练配置的26,300个逐语言损失观测数据进行拟合，在留出配置上相比最强评估基线将预测误差降低了28%。在活跃参数数量和每种语言训练词元数量相同的情况下，32个专家相比稠密模型将语言从12种扩展到200种时的损失惩罚降低了83%。我们的定律进一步表明，更广的语言覆盖范围会在更大的参数范围内倾向于使用更多专家。

    arXiv:2610.20247v1 Announce Type: new  Abstract: Multilingual models share capacity across languages, so adding languages can hurt performance on existing ones. Mixture-of-experts models expand this capacity with little additional per-token computation. Given this extra capacity, does adding experts reduce what each additional language costs? To answer this question, we propose a repetition-aware joint scaling law relating number of active parameters, training tokens per language, number of experts, and language coverage. Fitted on 26,300 per-language loss observations from 64 pretrained configurations, our law reduces prediction error on held-out configurations by 28% relative to the strongest evaluated baseline. At matched number of active parameters and training tokens per language, 32 experts reduce the loss penalty of expanding from 12 to 200 languages by 83% compared with a dense model. Our law further shows that broader language coverage favors more experts across a wider range 
    
[^71]: 检索优于训练：基于相似性搜索的时序异常检测模型选择方法

    Retrieval Over Training: Similarity search-based Model Selection for Time Series Anomaly Detection

    [https://arxiv.org/abs/2610.20243](https://arxiv.org/abs/2610.20243)

    提出RAMSAD框架，通过相似性检索历史实例知识库而非训练模型的方式为新时间序列选择最优异常检测器，从而解决了现有模型选择方法在分布外（OOD）场景下精度大幅下降的问题。

    

    异常检测是时间序列分析中的一项基础任务，对众多应用的下游性能具有重要影响。尽管文献中已提出了大量的异常检测方法，但近期的基准研究表明，没有任何单一的检测器能够在高度异构的时间序列上始终表现最佳。因此，一种实用且可扩展的解决方案是开发一种模型选择方法，针对给定的时间序列选出最有可能表现良好的异常检测器。然而，现有文献中提出的模型选择方法在应用于分布外（Out-of-Distribution, OOD）场景时会出现显著的精度下降。本文针对上述局限性，提出了RAMSAD——一个基于检索的时序异常检测模型选择框架。给定一个新的时间序列，我们的方法会查询一个由先前观察到的实例构成的知识库，并检索……（原文摘要到此截断）

    arXiv:2610.20243v1 Announce Type: new  Abstract: Anomaly detection is a fundamental task for time-series analytics, with important implications for the downstream performance of many applications. Despite the large number of anomaly detection methods proposed in the literature, recent benchmark studies have shown that no single detector performs best across highly heterogeneous time series. Therefore, a practical and scalable solution is to develop a model-selection method that, for a given time series, selects the anomaly detector most likely to perform well. Nevertheless, the model selection approaches proposed in the literature suffer from a significant drop in accuracy when applied in Out-of-Distribution (OOD) settings. In this paper, we tackle the aforementioned limitation and propose RAMSAD, a retrieval-based framework for model selection in time-series anomaly detection. Given a new time series, our method queries a knowledge base of previously observed instances, retrieves the 
    
[^72]: MAEM：通过无重放、容量控制的动态专家混合来缓解类增量图学习中的近期偏差

    MAEM: Mitigating Recency Bias via a Replay-Free, Capacity-Controlled Dynamic Mixture of Experts for Class-Incremental Graph Learning

    [https://arxiv.org/abs/2610.20238](https://arxiv.org/abs/2610.20238)

    提出MAEM框架，通过为每个新会话训练并冻结专用专家、以无重放且容量受控的动态专家混合方式进行路由，从而在无需任务标识的情况下缓解类增量图学习中的近期偏差与灾难性遗忘。

    

    图神经网络在图结构数据上表现出色，但当类别按顺序依次到达时会遭受灾难性遗忘。在类增量图学习中，预测结果往往由最近学习的类别主导，从而产生严重的近期偏差。动态扩展方法可以将知识在不同会话间隔离，但许多方法假设的是任务增量设置，在测试时需要任务标识来选择分支或输出头——而这一信息在类增量学习中是无法获得的。可学习的路由器提供了一种替代方案，但在新数据上持续微调路由器可能使路由偏向最近添加的组件。我们提出了MAEM，一个面向类增量图学习的无重放、容量控制的动态专家混合框架。MAEM使用简化图卷积（SGC）构建节点表示，并为每个新会话训练一个专用专家，随后将其冻结以保护已有知识。对于无需任务ID的路由机制……（摘要原文在此处截断）

    arXiv:2610.20238v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) excel on graph-structured data but suffer catastrophic forgetting when classes arrive sequentially. In class-incremental graph learning, predictions are often dominated by recently learned classes, causing severe recency bias. Dynamic expansion can isolate knowledge across sessions, but many methods assume task-incremental settings and require task identities at test time to select branches or heads---information unavailable in class-incremental learning. Learnable routers offer an alternative, yet continual fine-tuning the router on new data can bias routing toward recent components. We propose MAEM, a replay-free, capacity-controlled dynamic mixture-of-experts framework for class-incremental graph learning. MAEM builds representations with Simplified Graph Convolution (SGC) and trains a dedicated expert for each new session, then freezes it to protect prior knowledge. For routing without task IDs or a share
    
[^73]: 神经网络的最小宽度与实射影空间的浸入

    The minimal width of neural networks and immersions of real projective space

    [https://arxiv.org/abs/2610.20229](https://arxiv.org/abs/2610.20229)

    本文证明了当实射影空间 $\mathbb{RP}^{n-1}$ 可光滑浸入 $\mathbb{R}^n$ 时，神经网络逼近从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 连续函数的最小宽度至少为 $n+\min\{n,m\}$，从而在 $n=2,3,4,7,8$ 时确定了 ReLU 网络的最小宽度为 $n+m$，揭示了神经网络逼近理论与微分拓扑之间的深刻联系。

    

    我们研究了在紧集上一致收敛拓扑下，能够逼近所有从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的连续函数的深度神经网络的最小宽度。我们证明，只要实射影空间 $\mathbb{RP}^{n-1}$ 可以光滑浸入到 $\mathbb{R}^n$ 中，在激活函数连续且单调的假设下，最小宽度至少为 $n+\min\{n,m\}$。由此可得，当 $n$ 等于 2、3、4、7 或 8 且 $1\le m \le n$ 时，ReLU 网络的最小宽度为 $n+m$，这扩展了 Hanin-Sellke 和 Hwang 的结果。这表明神经网络的逼近理论与微分拓扑之间存在着一种有趣的联系。

    arXiv:2610.20229v1 Announce Type: cross  Abstract: We investigate the minimal width of deep neural networks that can approximate all continuous functions from $\mathbb{R}^n$ to $\mathbb{R}^m$ in the topology of uniform convergence on compact subsets. We show that whenever the real projective space $\mathbb{RP}^{n-1}$ admits a smooth immersion into $\mathbb{R}^n$, the minimal width is at least $n+\min\{n,m\}$, assuming that the activation function is continuous and monotone. As a consequence, the minimal width of ReLU networks is $n+m$ whenever $n$ is equal to 2, 3, 4, 7 or 8 and $1\le m \le n$, extending results of Hanin-Sellke and Hwang. This suggests an intriguing connection between the approximation theory of neural networks and differential topology.
    
[^74]: 混合泊松扩散模型及其在后验采样中的应用

    Mixed Poisson diffusion models and application to posterior sampling

    [https://arxiv.org/abs/2610.20218](https://arxiv.org/abs/2610.20218)

    该论文提出了一种由多元计数过程驱动的混合泊松扩散模型（MPDM），为观测数据离散且服从泊松分布的泊松逆问题提供了后验采样的新框架。

    

    扩散模型是学习复杂概率分布的强大框架。虽然广泛使用的连续数据扩散模型依赖于高斯加噪过程，但我们转而开发了基于泊松噪声的扩散模型。更准确地说，我们引入并研究了一个新的数学框架，称为混合泊松扩散模型（Mixed Poisson Diffusion Models, MPDM），其加噪动力学由多元计数过程驱动。我们的方法受泊松逆问题的启发。高斯扩散模型的成功使其在越来越多的逆问题后验采样方法中被用作先验模型。然而，这些方法大多聚焦于高斯逆问题，即观测值通过确定性前向算子生成并被加性高斯噪声所污染。相比之下，观测值离散且服从泊松分布的泊松逆问题在此领域仍相对缺乏探索。

    arXiv:2610.20218v1 Announce Type: cross  Abstract: Diffusion models are a powerful framework for learning complex probability distributions. While widely used diffusion models for continuous data rely on Gaussian noising processes, we instead develop diffusion models based on Poisson noise. More precisely, we introduce and study a new mathematical framework, called Mixed Poisson Diffusion Models (MPDM), with noising dynamics driven by multivariate counting processes.   Our methodology is motivated by Poisson inverse problems. The success of Gaussian diffusion models has led to their use as prior models in a growing number of posterior sampling methods for inverse problems. However, most of these approaches focus on Gaussian inverse problems, in which observations are generated through a deterministic forward operator corrupted by additive Gaussian noise. By contrast, Poisson inverse problems, where observations are discrete and Poisson distributed, remain comparatively underexplored in
    
[^75]: 持续预训练前列腺MRI ViT编码器以实现任务对齐

    Continual pretraining Prostate MRI ViT Encoder for task alignment

    [https://arxiv.org/abs/2610.20217](https://arxiv.org/abs/2610.20217)

    该论文提出利用LoRA在前列腺分割任务上对预训练的ViT编码器进行持续预训练，以增强预训练与下游任务之间的对齐，并通过CKA逐层表征分析评估表征变化。

    

    构建MRI基础模型已成为推进医学图像分析和提高任务特定模型可及性的一个有前景的方向。现有方法主要关注有效的预训练策略，以及在不同下游任务上评估其性能。虽然这些方法展现了性能提升，但对理解预训练与下游目标之间任务对齐的关注却有限。与语言模型中两个阶段天然对齐不同，这种对齐在基于视觉的应用中并不明显。受此空白的启发，我们探索了持续预训练。具体而言，在前列腺分割任务上使用低秩适应对预训练的ViT编码器进行持续预训练，以增强任务对齐和下游性能。我们进一步使用中心化核对齐进行逐层表征分析，以评估表征变化。

    arXiv:2610.20217v1 Announce Type: cross  Abstract: Building foundation models for MRI has emerged as a promising direction for advancing medical image analysis and improving accessibility of task specific models. Existing approaches primarily focus on effective pretraining strategies and evaluating their performance across diverse downstream tasks. While these methods show performance gains, limited attention has been given to understanding task alignment between pretraining and downstream objectives. Unlike language models, where the two stages are naturally aligned, such alignment is less evident in vision based applications. Motivated by this gap, we explore continual pretraining. Specifically, continually pretraining a pretrained ViT encoder using Low-Rank Adaptation (LoRA) on a prostate segmentation task to enhance task alignment and downstream performance. We further perform layer-wise representational analysis using Centered Kernel Alignment (CKA) to assess representational shif
    
[^76]: HGRP：用于可布线性预测的异构图神经网络

    HGRP: Heterogeneous Graph Neural Network for Routability Prediction

    [https://arxiv.org/abs/2610.20209](https://arxiv.org/abs/2610.20209)

    提出异构图神经网络HGRP，将单元、线网和布线网格建模为不同类型的节点并同时保留拓扑连接、空间映射与网格邻接等多层次电路语义，从而在无需布线的情况下实现高效准确的可布线性预测。

    

    可布线性预测无需实际布线即可快速评估布局质量，为可布线性驱动的布局优化提供有效反馈。近年来，机器学习方法通过数据驱动的方式实现了高效的可布线性预测。然而，其预测性能高度依赖于输入表示捕获关键电路信息的能力。为了解决现有电路表示无法同时保留逻辑拓扑、物理布局和布线资源信息的局限性，本文提出了HGRP，一种用于可布线性预测的异构图神经网络。HGRP将单元、线网和布线网格建模为不同类型的节点，并显式表示拓扑连接、空间映射和网格邻接关系，从而构建了能够保留多层次电路语义的异构图表示。

    arXiv:2610.20209v1 Announce Type: new  Abstract: Routability prediction enables rapid evaluation of placement quality without routing, providing effective feedback for routability-driven placement optimization. Recent machine learning methods have achieved efficient routability prediction through data-driven approach. However, their prediction performance highly depends on the capability of input representations to capture essential circuit information. To address the limitations of existing circuit representations in simultaneously preserving logical topology, physical layout and routing resource information, this paper proposes HGRP, a heterogeneous graph neural network for routability prediction. HGRP models the cells, nets, and routing grids as distinct types of nodes and explicitly represents topological connectivity, spatial mapping, and grid adjacency, thereby constructing a heterogeneous graph representation that preserves multi-level circuit semantics. Based on the representat
    
[^77]: 探究在模拟城市中基于鸟瞰图、从少量演示数据中进行行为克隆的自动驾驶方法

    Investigating Behavior Cloning from Few Demonstrations for Autonomous Driving based on Bird's-Eye View in Simulated Cities

    [https://arxiv.org/abs/2610.20203](https://arxiv.org/abs/2610.20203)

    该论文提出在行为克隆中引入核密度估计器，根据动作密度对训练样本加权，从而在专家演示稀缺的情况下，基于鸟瞰图输入显著提升自动驾驶对红灯停车等罕见但关键动作的学习能力。

    

    本文研究了在模拟城市环境中，从鸟瞰图（BEV）视角使用行为克隆（BC）实现自动驾驶的方法。行为克隆通过监督学习来模仿专家的驾驶行为。以往的研究已在CARLA模拟器中应用了行为克隆，但未能充分解决遵守交通信号灯这一挑战。我们的方法通过引入核密度估计器对行为克隆进行改进，根据动作密度调整训练样本的权重，从而改善了对罕见但关键动作（如红灯停车、绿灯起步加速）的学习，尤其是在专家演示数据稀缺的场景下。鸟瞰图输入提供了驾驶环境的抽象俯视视图，利用这种输入，我们的方法简化了策略学习过程。训练好的卷积神经网络（CNN）基于这些鸟瞰图输入和额外的状态变量输出转向与加速动作。实验……

    arXiv:2610.20203v1 Announce Type: cross  Abstract: This paper investigates the use of Behavior Cloning (BC) for autonomous driving from a bird's-eye view (BEV) perspective in simulated urban environments. BC uses supervised learning to mimic expert driving behaviors. Previous works have applied BC in the CARLA simulator but did not fully address the challenges of traffic light compliance. Our approach enhances BC by integrating a kernel density estimator to adjust training sample weights based on action density, thereby improving the learning of rare but critical actions such as stopping at red lights and accelerating at green lights, specially in scenarios of scarce number of expert demonstrations. Using BEV inputs, which provide an abstract top-down view of the driving environment, our method simplifies the policy learning process. The trained convolutional neural network (CNN) outputs steering and acceleration actions based on these BEV inputs and additional state variables. Experim
    
[^78]: 指数凹在线学习中维度最优的动态遗憾

    Dimension-Optimal Dynamic Regret in Exp-Concave Online Learning

    [https://arxiv.org/abs/2610.20187](https://arxiv.org/abs/2610.20187)

    本文解决了指数凹在线学习动态遗憾中悬而未决的维度最优性问题，提出一种自适应未知路径长度的全信息学习器并给出匹配下界，将遗憾界的维度因子从 d^{2/3} 改进到尖锐最优的 d^{1/3}。

    

    追踪不断变化的比较基准（comparators）是在线学习中的一个核心问题。对于指数凹（exp-concave）损失，先前的研究已在固定维度下建立了遗憾关于时间范围 T 和欧氏路径长度 P_T 的 T^{1/3}P_T^{2/3} 依赖关系，但维度的最优代价一直悬而未决。本文通过建立一个尖锐的 d^{1/3} 曲线型维度因子解决了这一问题，改进了近期基于切换和区间归约方法所得到的 d^{2/3} 因子 [1, 2]。为实现这一改进，作者在 R^d 的紧凸子集上设计了一个 proper 的、全信息设置下的学习器，该学习器能够自适应未知的路径长度，并在维度无关的直径、梯度与曲率参数固定的条件下，保证遗憾界 Reg_T(u_{1:T}) ≤ Õ(d + min{d^{1/3}T^{1/3}P_T^{2/3}, √(TP_T), T})。此外，作者在直径为 1 的欧氏球上给出了匹配的下界，从而在对数因子范围内确立了极小极大前沿。

    arXiv:2610.20187v1 Announce Type: new  Abstract: Tracking changing comparators is a central problem in online learning. For exp-concave losses, previous work established the $T^{1/3}P_T^{2/3}$ dependence on the horizon $T$ and Euclidean path length $P_T$ for fixed dimension, while the optimal cost of dimension remained unresolved. We resolve this question by establishing a sharp $d^{1/3}$ curved dimension factor, improving the $d^{2/3}$ factor of recent switching and interval reductions [1, 2]. To achieve this improvement, we develop a proper, full-information learner on compact convex subsets of $\mathbb{R}^d$ that adapts to the unknown path length and, under fixed dimension-independent diameter, gradient, and curvature parameters, guarantees   $$\operatorname{Reg}_T(u_{1:T}) \le \widetilde{O}\left(d+\min\left\{d^{1/3}T^{1/3}P_T^{2/3},\sqrt{TP_T},T\right\}\right).$$   A matching lower bound on a diameter-one Euclidean ball establishes the minimax frontier up to logarithmic factors whe
    
[^79]: 跨模型上下文内神经反馈揭示大语言模型中的（有限）激活控制能力

    Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs

    [https://arxiv.org/abs/2610.20179](https://arxiv.org/abs/2610.20179)

    本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。

    

    基于激活向量的探针被广泛用于检测语言模型的不良行为，但其有效性依赖于一个假设：模型无法控制其激活所编码的内容。人类可以通过神经反馈学会控制甚至单个神经元的放电，近期有研究探讨了这种能力是否也适用于语言模型，但得出了相互矛盾的结论。迄今为止，尚无实验设计能够将模型在任务上的表现能力与其对自身激活的真实访问区分开来。我们通过“上下文内神经反馈”会话填补了这一空白：在其中，一个“受试”模型被要求最大化（或最小化）一个由其自身激活计算得出的分数，且该分数会在每一轮对话中展示给该模型。在我们的第一个实验中，句子要么由受试模型自己生成，要么由第二个“写作”模型生成，而分数始终来自受试模型的激活。我们测试了来自五个系列的十个开源权重模型，规模从9B到32B。部分模型……（原文摘要在此处被截断）

    arXiv:2610.20179v1 Announce Type: new  Abstract: Probes on activation vectors are widely used to detect misbehavior in language models, but their success relies on the assumption that models cannot control what their activations encode. Humans can learn, through neurofeedback, to control the firing of even single neurons, and recent studies investigated whether this ability extends to language models, reaching conflicting conclusions. No experimental design so far disentangles competence at the task from genuine access to one's own activations. We fill this gap with in-context neurofeedback sessions, in which a "subject" model is asked to maximize (or minimize) a score derived from its activations, shown back to it at every turn. In our first experiment, the sentences are generated either by the subject or by a second "writer" model, while the score always derives from the activations of the subject. We test ten open-weight models from five families, with sizes from 9B to 32B. Some mod
    
[^80]: RM-OPD：面向生成式奖励模型的同策略蒸馏

    RM-OPD: On-Policy Distillation for Generative Reward Models

    [https://arxiv.org/abs/2610.20174](https://arxiv.org/abs/2610.20174)

    该论文提出RM-OPD两阶段框架，先通过单领域GRPO微调训练各领域专家教师模型，再通过路由同策略蒸馏将其整合为单一学生模型，从而解决Mix-RL训练生成式奖励模型时的奖励稀疏与跨领域干扰问题。

    

    基于混合领域偏好数据的强化学习（Mix-RL）已成为训练通用生成式奖励模型（GRM）的主流方法。然而，Mix-RL 存在两个关键瓶颈：由结果奖励导致的奖励稀疏性，以及联合优化异构能力所引起的跨领域干扰，二者共同导致了能力获取不均衡和普遍的奖励欺骗问题。受同策略蒸馏（On-Policy Distillation, OPD）在多领域能力融合中成功应用的启发，我们提出了 RM-OPD（通过同策略蒸馏进行奖励建模），这是一个将专家能力获取与能力整合相分离的两阶段框架。首先，我们通过单领域 GRPO 微调培养领域专精的教师模型，使每个专家能够在隔离状态下达到其性能上限；然后，我们通过路由同策略蒸馏将这些教师模型整合到一个学生模型中。

    arXiv:2610.20174v1 Announce Type: cross  Abstract: Reinforcement Learning on Mixed-domain preference data (Mix-RL) has become the mainstream approach for training general-purpose Generative Reward Models (GRMs). However, Mix-RL suffers from two critical bottlenecks: the reward sparsity induced by outcome rewards, and the cross-domain interference arising from jointly optimizing heterogeneous capabilities, which together give rise to imbalanced capability acquisition and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in multi-domain capability integration, we introduce RM-OPD (Reward Modeling via On-Policy Distillation), a two-stage framework that separates specialist acquisition from capability integration. First, we cultivate domain-specialized teacher models via single-domain GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; then we consolidate these teachers into one student model through (i) routed on-policy dis
    
[^81]: 使用混合整数线性规划优化稀疏布尔规则网络

    Optimizing Sparse Boolean Rule Networks using Mixed Integer Linear Programming

    [https://arxiv.org/abs/2610.20170](https://arxiv.org/abs/2610.20170)

    该论文提出用混合整数线性规划训练稀疏的布尔规则网络（DNF网络），通过增强的公式大幅减少整数决策变量并消除冗余约束，在保持精确布尔语义的同时实现误分类成本与稀疏性之间的全局最优权衡。

    

    我们研究学习准确且认知上简单的二分类器的问题，这类分类器可以直接解释为布尔规则。我们提出了用于训练稀疏析取范式（DNF）网络的混合整数线性规划（MILP）公式，这类网络可被视为由合取隐藏层和析取输出层组成的小型网络。与以往使用MILP进行归纳规则学习的工作不同，我们从固定的网络结构出发，进而为其寻找最优权重。正则化的目标函数明确地在误分类成本与稀疏性之间进行权衡，从而产生全局优化的规则集。我们的主要技术贡献是一种增强的公式，它大幅减少了整数决策变量的数量并消除了冗余约束，同时对所有可行解保持精确的布尔语义。此外，我们开发了一种基于初始化和dropout相结合的方法来打破对称性……

    arXiv:2610.20170v1 Announce Type: new  Abstract: We study the problem of learning accurate and cognitively simple binary classifiers that are directly interpretable as Boolean rules. We propose mixed integer linear programming (MILP) formulations for training sparse disjunctive normal form (DNF) networks, viewed as small networks with a conjunctive hidden layer and a disjunctive output layer. Contrary to previous work that uses MILP for inductive rule learning, we start with a fixed network structure, for which we then find optimal weights. A regularized objective explicitly trades off misclassification costs against sparsity, yielding globally optimized rule sets. Our main technical contribution is an enhanced formulation that substantially reduces the number of integer decision variables and eliminates redundant constraints, while preserving exact Boolean semantics for all feasible solutions. Furthermore, we develop a combined approach based on initialization and dropout to break sym
    
[^82]: 差距检测：通过需求与可辩护性缓解同质化

    Gap Detection: Mitigating Homogenization through Demand and Defensibility

    [https://arxiv.org/abs/2610.20162](https://arxiv.org/abs/2610.20162)

    提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。

    

    语言模型在回答开放式问题时常常生成语义相似的答案，即使存在许多合理的替代方案。生成有用的替代方案需要同时识别查询所要求的内容以及现有回答已经覆盖的内容。我们提出了差距检测（Gap Detection），这是一个在共享文本嵌入空间中搜索相关但尚未被占据的方向的框架。其目标函数GapScore将查询相关性（需求，Demand）与最近现有条目的相异度（可辩护性，Defensibility）相结合。该框架在单位球面上优化该乘积，利用语言模型将候选方向映射为概念词，并通过任务特定的过滤和受约束的提示将这些概念转化为文本。在五个精选主题和90个INFINITY-CHAT查询上，报告的输出碰撞率平均相对降低分别为12.8%和10.8%。我们还将该框架实例化应用于四个产品的生成式引擎优化。

    arXiv:2610.20162v1 Announce Type: new  Abstract: Language models often produce semantically similar answers to open-ended questions, even when many plausible alternatives exist. Generating useful alternatives requires identifying both what the query calls for and what existing responses already cover. We introduce Gap Detection, a framework that searches for relevant but unoccupied directions in a shared text embedding space. Its objective, GapScore, combines query relevance (Demand) with dissimilarity to the nearest existing item (Defensibility). The framework optimizes this product on the unit sphere, maps candidate directions to concept words with a language model, and uses task-specific filtering and constrained prompting to turn those concepts into text. On five curated topics and 90 INFINITY-CHAT queries, the reported mean relative reductions in output Collision are 12.8% and 10.8%, respectively. We also instantiate the framework for generative engine optimization over four produ
    
[^83]: 基于流匹配的上下文批量黑盒优化

    In-Context Batch Black-box Optimization via Flow Matching

    [https://arxiv.org/abs/2610.20138](https://arxiv.org/abs/2610.20138)

    提出FMTR方法，利用流匹配离线学习历史条件化的连续提议分布，实现可变大小批量的高效采样，并在优化过程中自适应平衡探索与利用，改进批量黑盒优化。

    

    批量黑盒优化对于昂贵系统非常重要，在这类系统中可以并行评估多个候选方案，但只能负担少数几轮评估。在这种情况下，有效的批量构建必须在识别多个有前景区域的同时避免冗余评估，而传统贝叶斯优化随着批量大小的增长，可能需要越来越复杂的批量选择程序。我们提出了汤普森采样精炼流匹配（FMTR），这是一种摊销方法，可以离线学习一个连续的、以历史为条件的提议分布，从中可以采样可变大小的批量。FMTR通过条件流匹配在汤普森采样的argmax上进行离线训练，这些argmax通过一个依赖于优化视野的朗之万前推映射向高价值区域精炼，从而在优化过程中自适应地调整探索-利用权衡。在测试时，模型参数保持固定，批量生成只需（原文在此截断）。

    arXiv:2610.20138v1 Announce Type: new  Abstract: Batch black-box optimization is important for expensive systems where many candidates can be evaluated in parallel, yet only a few rounds of evaluation are affordable. In this regime, effective batch construction must identify multiple promising regions while avoiding redundant evaluations, and conventional Bayesian optimization can require increasingly difficult batch-selection procedures as the batch size grows. We introduce Flow Matching for Thompson-Sampling Refinement (FMTR), an amortized approach that learns offline a continuous, history-conditioned proposal distribution from which variable-size batches can be sampled. FMTR is trained offline by conditional flow matching on Thompson-sampling argmaxes refined toward high-value regions via a horizon-dependent Langevin pushforward map, adapting the exploration-exploitation tradeoff over the optimization horizon. At test time, the model parameters remain fixed and batch generation requ
    
[^84]: 边缘视觉：边缘平台上视觉语言模型推理的能耗特性分析

    Seeing at the Edge: Energy Characterization of Vision-Language Model Inference on Edge Platforms

    [https://arxiv.org/abs/2610.20136](https://arxiv.org/abs/2610.20136)

    该论文首次系统测量了视觉语言模型在树莓派5和Jetson等边缘平台上的能耗特性，发现NPU卸载可将每token能耗降低约三分之二、避免热节流，并揭示图像输入会同时带来固定能耗和随输出长度增长的额外能耗。

    

    视觉语言模型（VLM）正在向边缘设备迁移，但人们对其在配备NPU或GPU加速器的单板计算机上究竟消耗多少能量、以及这些能量去向何处，仍然知之甚少。我们在配备Hailo-10H NPU的树莓派5和Jetson Orin Nano Super上，分别在CPU和加速器上运行Qwen2-VL-2B-Instruct及其纯文本对应模型，测量了整机能耗、延迟和热行为，并将每次请求的固定能耗与随输出长度增长的能耗分离开来。由于这些加速器使用不同的量化和推理服务软件，我们比较的是部署软件栈而非芯片硬件本身。研究有三个突出发现。第一，NPU卸载使树莓派成为一个高效得多的VLM平台：它将每个生成token的能耗降低约三分之二，让主机CPU基本保持空闲，并避免了CPU后端在持续运行中出现的 thermal throttling（热节流）。第二，图像输入会同时增加固定能耗和每token能耗……

    arXiv:2610.20136v1 Announce Type: new  Abstract: Vision-language models (VLMs) are moving onto edge devices, but little is known about how much energy they draw on single-board computers with NPU or GPU accelerators, or where that energy goes. We measure total-system energy, latency, and thermal behavior of Qwen2-VL-2B-Instruct and its text-only sibling on a Raspberry Pi 5 with a Hailo-10H NPU and on a Jetson Orin Nano Super, on both CPU and accelerator, and separate the fixed energy of each request from the energy that grows with output length. Because the accelerators use different quantization and serving software, we compare deployment stacks rather than silicon. Three findings stand out. First, NPU offloading makes the Pi a much more efficient VLM platform: it cuts energy per generated token by about two-thirds, leaves the host CPU largely free, and avoids the thermal throttling that the CPU backend reaches in sustained use. Second, an image adds both fixed and per-token energy, b
    
[^85]: 神经算子的Bregman近端视角

    A Bregman Proximal Viewpoint on Neural Operators

    [https://arxiv.org/abs/2610.20120](https://arxiv.org/abs/2610.20120)

    本文将神经算子层的作用解释为Banach函数空间上Bregman正则化优化问题的最小值点，由此提出了一种新的Bregman神经算子，其实验表明其傅里叶变体在训练更深、更精确的模型上更具优势。

    

    我们通过将算子层的作用视为Banach函数空间上Bregman正则化优化问题的最小值点，在神经算子方面取得了若干进展。所提出的框架允许将激活算子解释为从对偶空间到原空间的Bregman邻近算子。这一新颖的视角足够通用，既能涵盖经典的神经算子，又催生了一种新的变体——我们称之为Bregman神经算子，它包含逆激活算子，并具有与标准神经算子相同的表达能力。数值实验证明了傅里叶神经算子的Bregman变体在训练更深、更精确模型方面的额外优势。

    arXiv:2610.20120v1 Announce Type: new  Abstract: We present several advances on neural operators by viewing the action of operator layers as the minimizers of Bregman regularized optimization problems over Banach function spaces. The proposed framework allows interpreting the activation operators as Bregman proximity operators from dual to primal space. This novel viewpoint is general enough to recover classical neural operators as well as a new variant, coined Bregman neural operators, which includes the inverse activation operator and features the same expressivity of standard neural operators. Numerical experiments support the added benefits of the Bregman variant of Fourier neural operators for training deeper and more accurate models.
    
[^86]: KANLab：面向Kolmogorov-Arnold网络的TensorFlow Playground

    KANLab: TensorFlow Playground for Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2610.20117](https://arxiv.org/abs/2610.20117)

    KANLab是一个从TensorFlow Playground分叉而来的开源浏览器可视化工具，通过交互式展示B样条激活曲线、剪枝和梯度下降等现象，帮助用户直观学习Kolmogorov-Arnold网络的核心概念。

    

    KANLab是一个基于浏览器的可视化工具，旨在以交互方式教授Kolmogorov-Arnold网络（KAN）的核心概念，它从广受欢迎的TensorFlow Playground分叉而来。为了最大程度地提高教学清晰度，KANLab省略了现代KAN实现中的残差连接和可学习缩放参数，确保所显示的B样条激活曲线直接对应其底层的可训练控制点。用户可以检查和修改激活函数、初始化方案，并观察梯度下降如何纠正人为造成的扰动。此外，用户还可以在紧凑的2D机器学习环境中探索激活方差、交互式剪枝以及重新训练/继续训练等现象。KANLab是开源的，可在https://kanlab.app/获取。

    arXiv:2610.20117v1 Announce Type: new  Abstract: KANLab is a browser-based visualization tool designed to interactively teach the core concepts of Kolmogorov-Arnold Networks (KAN), forked from the widely successful TensorFlow Playground. To maximize educational clarity, KANLab omits residual connections and learnable scaling parameters found in modern KAN implementations, ensuring that displayed B-spline activation curves directly correspond to their underlying trainable control points. Users can inspect and modify activation functions, initialization schemes and observe gradient descent correcting manual disruptions. Furthermore, user can explore phenomena such as activation variance, interactive pruning, and re\-/training within a compact 2D machine learning setting. KANLab is open-source and available at https://kanlab.app/
    
[^87]: 从谱范数到Frobenius范数：当Muon失去其谱隐式偏差时

    From Spectral to Frobenius: When Muon Loses Its Spectral Implicit Bias

    [https://arxiv.org/abs/2610.20112](https://arxiv.org/abs/2610.20112)

    本文证明Muon优化器的实际实现（有限次Newton-Schulz迭代和正稳定化常数）会使其失去理论上的谱隐式偏差，转而像梯度下降一样选择Frobenius最大间隔方向。

    

    关于隐式偏差的先前研究表明，梯度下降在线性可分问题上会选择Frobenius最大间隔方向，而Fan等人（2025）则为谱范数最速下降方法（包括采用精确正交化的理想化Muon）确立了谱最大间隔行为。然而在实践中，Muon使用有限次Newton-Schulz迭代和正的稳定化常数来近似正交化。我们证明，这些实现上的选择导致Muon在可分数据的线性分类任务中选择Frobenius最大间隔方向，而非谱最大间隔方向。我们还对指数移动平均和Nesterov式动量这两种情况确立了同样的行为。关键的技术要素是一个可求和的扰动论证，它表明所得的参数轨迹与梯度下降的参数轨迹之间保持有界距离。随后，我们厘清了Newton-Schulz迭代次数、稳定化常数等因素各自的作用。

    arXiv:2610.20112v1 Announce Type: new  Abstract: Prior work on implicit bias shows that gradient descent selects the Frobenius max-margin direction on linearly separable problems, whereas Fan et al. (2025) established spectral max-margin behavior for spectral-norm steepest descent methods, including idealized Muon with exact orthogonalization. In practice, however, Muon approximates orthogonalization using finite Newton-Schulz iterations and a positive stabilization constant. We prove that these implementation choices cause Muon to select the Frobenius rather than the spectral max-margin direction in linear classification with separable data. We also establish this behavior for both exponential moving average and Nesterov-style momentum. The key technical ingredient is a summable perturbation argument showing that the resulting parameter trajectories remain at bounded distance from those of gradient descent. We then disentangle the roles of Newton-Schulz iteration count, the stabilizat
    
[^88]: 穷举式规范骨架生成：面向可分解符号回归的形式化数据集

    Exhaustive Canonical Skeletons Generation: A Formal Dataset for Decomposable Symbolic Regression

    [https://arxiv.org/abs/2610.20108](https://arxiv.org/abs/2610.20108)

    该论文提出基于确定性受限上下文无关文法穷举生成的单变量规范骨架形式化数据集，消除了符号回归数据集中的结构冗余，并支持在不假设线性可分性的前提下求解多变量符号回归问题。

    

    神经符号回归（SR）模型需要多样化的数学表达式数据集以实现稳健训练。然而，现有数据集主要依赖随机采样方法，存在结构冗余度高和函数覆盖不完整的问题。为解决这一问题，我们通过确定性受限上下文无关文法生成了一份穷举式的单变量规范骨架数据集。规范骨架作为一种通用的结构表示，能够涵盖更具体的函数形式，从而消除冗余表达式并避免重复的计算开销。尽管我们的数据集仅由单变量骨架组成，但其设计旨在支持可分解SR范式，从而在无需假设线性可分性的情况下解决多变量SR问题。此外，我们还为该数据集配套了一种采样策略，可生成干净的数值数据，而不会触发数值不稳定性或依赖人工（原文在此处截断）。

    arXiv:2610.20108v1 Announce Type: new  Abstract: Neural symbolic regression (SR) models require diverse datasets of mathematical expressions for robust training. However, existing datasets rely predominantly on stochastic sampling methods, which suffer from high structural redundancy and incomplete functional coverage. To address this, we introduce an exhaustive dataset of univariate canonical skeletons generated via a deterministic constrained context-free grammar. A canonical skeleton acts as a generic structural representation that subsumes more specific functional forms, eliminating redundant expressions and avoiding repeated computational efforts. Although our dataset consists of univariate skeletons, it is designed to support the decomposable SR paradigm to solve multivariate SR problems without assuming linear separability. Furthermore, we pair this dataset with a sampling strategy that generates clean numerical data without triggering numerical instabilities or relying on artif
    
[^89]: 市场微观结构基础模型学到了什么？基于合成限价订单簿的受控研究

    What Do Foundation Models for Market Microstructure Learn? A Controlled Study with Synthetic Limit Order Books

    [https://arxiv.org/abs/2610.20104](https://arxiv.org/abs/2610.20104)

    本文通过在潜在参数已知的合成限价订单簿模拟器上训练GPT风格基础模型，首次证明订单流上的下一词元预训练实质上实现了对市场机制的摊销贝叶斯推断，其预测性能可精确逼近贝叶斯最优解。

    

    在限价订单簿（LOB）消息流上训练的自回归Transformer 越来越多地被提出作为市场微观结构的基础模型。由于真实市场的生成过程是未知的，这些模型只能通过其输出的真实性来评判，这使它们究竟学到了什么这一问题悬而未决。我们通过反转实验设置使这一问题变得可以回答：我们在潜在参数已知的模拟器所产生的订单流上训练GPT风格的模型，从而使得贝叶斯最优预测器、生成状态以及精确数据熵都可以作为参照基准。以此为衡量标准，订单流上的下一词元预训练实际上实现了对市场状态机制的摊销贝叶斯推断。学习到的预测器与精确贝叶斯最优解的差距不超过每词元一毫纳特，即使在潜在状态在一个回合内发生漂移的非平稳情形下，即需要滤波才能处理的机制中，也能逼近预言机下界。潜在的状态机制可被线性解码……（摘要原文在此处截断）

    arXiv:2610.20104v1 Announce Type: new  Abstract: Autoregressive transformers trained on limit order book (LOB) message streams are increasingly proposed as foundation models for market microstructure. Because the generative process of real markets is unknown, these models can be judged only by the realism of their output, leaving open what they actually learn. We make the question answerable by inverting the setup: we train GPT-style models on order flow from simulators whose latent parameters are known, so that the Bayes-optimal predictor, the generating state, and the exact data entropy serve as references. Measured against them, next-token pretraining on order flow implements amortized Bayesian inference over market regimes. The learned predictive matches the exact Bayes optimum to within a millinat per token and approaches the oracle floor even on a nonstationary family whose latent state drifts within an episode, a regime that requires filtering. The latent regime is linearly deco
    
[^90]: 基于广义全变差最小化的最小二乘联邦学习的大维度分析

    A Large-Dimensional Analysis of Least-Squares Federated Learning via Generalized Total Variation Minimization

    [https://arxiv.org/abs/2610.20075](https://arxiv.org/abs/2610.20075)

    本文通过随机矩阵理论对基于广义全变差最小化的最小二乘联邦学习进行大维度分析，给出了独立同分布与非独立同分布高斯混合数据下渐近精确性能的精确解析表达式，量化了正则化如何在局部适应与协作之间进行权衡。

    

    去中心化网络上的联邦学习本质上是在通过协作抑制局部有限样本噪声与协作在异构节点之间引入的偏差之间进行权衡。我们通过对基于最小二乘模型的广义全变差最小化进行大维度分析来研究这种张力。利用随机矩阵理论的工具，我们证明了全局GTVMin解的Lipschitz观测在大维度下是集中的，且对任何底层网络结构都普遍成立。我们提供了在固定网络上独立同分布与非独立同分布高斯混合分布下渐近精确性能的精确解析表达式。对于后者，这些表达式量化了正则化如何在局部适应与协作之间取得平衡。我们的理论结果通过合成数据和真实世界数据的实验得到了验证。

    arXiv:2610.20075v1 Announce Type: new  Abstract: Federated learning over decentralized networks inherently trades off the suppression of local finite-sample noise through collaboration against the bias that collaboration introduces between heterogeneous nodes. We study this tension by performing a large-dimensional analysis of Generalized Total Variation Minimization with least-squares models. Using tools from random matrix theory, we show that Lipschitz observations of the global GTVMin solution concentrate in large dimensions, universally for any underlying network structure. We provide exact analytic expressions for the asymptotically precise performance under both i.i.d. and non-i.i.d. Gaussian mixture distributions on a fixed network. For the latter, these expressions quantify how regularization balances local adaptation against collaboration. Our theoretical results are confirmed by experiments on synthetic and real-world data.
    
[^91]: 重新思考预测视野：在少步MeanFlow中解耦模型查询与状态更新

    Rethinking Prediction Horizons: Decoupling Model Queries from State Updates in Few-Step MeanFlow

    [https://arxiv.org/abs/2610.20071](https://arxiv.org/abs/2610.20071)

    该论文在少步MeanFlow采样中解耦了模型查询区间与状态更新时长，发现预测视野对生成质量的影响并非仅由查询时间决定，还依赖于进入查询的状态分布——中点视野收益随早期引导设置的变化可从0.109显著增大至1.308 FID。

    

    MeanFlow预测区间平均速度，但传统的少步采样将模型查询的区间与状态更新的持续时间绑定在一起。我们在推理阶段将这两个量分离，在固定模型、更新网格和查询次数的同时改变预测视野。我们的核心发现是：视野的响应同时取决于查询位置以及进入该查询的状态。在七个独立训练的CIFAR-10检查点（分类器自由引导CFG尺度为2）上，较长的视野在早期步骤会恶化FID，但在中点处会改善FID。为了探究是否仅查询时间决定了这一响应，我们固定中点查询以及当前/后续引导，仅改变较早的CFG。在五个先前研究过的检查点上的保留因子实验结果中，中点视野的平均收益从0.109 FID增加到1.308 FID，表明其对输入状态分布具有敏感性。一个冻结的……（摘要在此处截断）

    arXiv:2610.20071v1 Announce Type: new  Abstract: MeanFlow predicts interval-average velocities, but conventional few-step sampling ties the interval queried from the model to the duration of the state update. We separate these two quantities at inference, fixing the model, update grid, and number of queries while varying the prediction horizon. Our central finding is that the horizon response depends on both query position and the states entering that query. Across seven independently trained CIFAR-10 checkpoints at classifier-free guidance (CFG) scale 2, a longer horizon worsens FID at early steps but improves it at the midpoint. To probe whether query time alone determines this response, we fix the midpoint query and current/subsequent guidance while changing only earlier CFG. In held-out factorial outcomes on five previously studied checkpoints, the mean midpoint horizon benefit increases from 0.109 to 1.308 FID, indicating sensitivity to the incoming-state distribution. A frozen fo
    
[^92]: 深度可训练，小尺寸可求解：多项式激活函数的损失面如何指导网络的设计、初始化与训练

    Trainable at Depth, Solvable at Small Size: What the Loss Surface of a Polynomial Activation Dictates for Designing, Initializing and Training Networks

    [https://arxiv.org/abs/2610.20069](https://arxiv.org/abs/2610.20069)

    该论文提出多项式激活函数 f(z) = z + λz³ 加恒等路径的构造，通过精确求解小规模模型的损失面来指导深层网络的设计、初始化与训练，使百层网络无需任何归一化机制即可在 MNIST 上成功训练。

    

    是什么让深度网络得以训练，其损失面又是什么形状？我们用一个构造在两个尺度上同时回答了这两个问题：一个多项式激活函数 f(z) = z + λz³（经过截断并投影到单位球面上），外加一条未归一化的恒等路径。作为深层堆叠网络，它在无需任何归一化机制的情况下于 MNIST 上训练至一百层；作为小模型，其平方损失是一个我们可以精确求解的多项式系统。小尺度解释了大尺度：它规定了如何设计、初始化和训练这个深层堆叠网络。

    arXiv:2610.20069v1 Announce Type: new  Abstract: What makes a deep network train, and what shape is its loss surface? We answer both with one construction at two scales: a polynomial activation f (z) = z + {\lambda}z 3 , clamped and projected onto the unit sphere, plus an unnormalized identity path. As a deep stack it trains to one hundred layers on MNIST with no normalization machinery; as a small model its square loss is a polynomial system we solve exactly. The small scale explains the large one: it prescribes how to design, initialize and train the stack.
    
[^93]: GBTSVR：基于粒球计算的孪生支持向量回归

    GBTSVR: Twin Support Vector Regression via Granular Ball Computing

    [https://arxiv.org/abs/2610.20053](https://arxiv.org/abs/2610.20053)

    提出 GBTSVR 及其分位数扩展 GB-TSVQR，以粒球取代单个样本作为基本学习单元，显著提升了孪生支持向量回归在噪声、异常值和数据异质性场景下的鲁棒性与泛化性能。

    

    现实世界应用中的回归模型常常受到噪声、异常值、数据异质性以及非对称误差特性的挑战，这些因素严重限制了传统逐点学习模型的有效性。虽然孪生支持向量回归（TSVR）通过使用非平行回归边界部分地解决了这些问题，但它仍然以单个样本作为基本学习单元。因此，TSVR 对含噪和受污染的观测数据仍然高度敏感，并且无法利用复杂数据集中固有的结构和区域信息，导致鲁棒性和泛化性能欠佳。为了解决这些局限性，我们提出了粒球孪生支持向量回归（GBTSVR）及其分位数扩展——粒球孪生支持向量分位数回归（GB-TSVQR）。所提出的模型将粒球计算与 TSVR 相结合，用粒球取代单个数据样本作为基本学习单元。

    arXiv:2610.20053v1 Announce Type: new  Abstract: Regression models used in real-world applications are often challenged by noise, outliers, data heterogeneity, and asymmetric error characteristics, which severely limit the effectiveness of conventional point-wise learning models. While twin support vector regression (TSVR) partially addresses these issues through the use of nonparallel regression bounds, it still relies on individual samples as basic learning units. As a result, TSVR remains highly sensitive to noisy and contaminated observations and fails to leverage the underlying structural and regional information inherent in complex datasets, leading to suboptimal robustness and generalization performance. To address these limitations, we introduce granular ball twin support vector regression (GBTSVR) and its quantile extension, granular ball twin support vector quantile regression (GB-TSVQR). The proposed models integrate granular ball computing with TSVR, replacing individual da
    
[^94]: 基于序列学习的物理信息神经网络建模用于电力变换器参数估计

    Sequential Learning-Based Physics-Informed Neural Network Modeling for Parameter Estimation on a Power Converter

    [https://arxiv.org/abs/2610.20041](https://arxiv.org/abs/2610.20041)

    该论文提出一种基于敏感性分析、按序释放梯度的分阶段PINN方法，可从时域数据中高精度识别降压变换器的电感、电容和寄生电阻参数，精度比同时训练基线提升一个数量级。

    

    本文提出一种分阶段的物理信息神经网络（PINN），用于从时域数据中识别DC降压变换器的电感、电容和寄生电阻。梯度按照从L到C再到rL到rC的顺序逐步释放，该顺序来源于敏感性分析。五阶段训练方案可将L和C的估计误差控制在5%以内、串联ESR控制在10%以内，而三阶段变体能以快11%的速度达到相同精度。8⁴全因子参数扫描验证表明，训练好的PINN在四个元件同时受到8%扰动时仍保持精度，在5 ms范围内波形RMSE达到2.6×10⁻²，比同时训练的基线方法好一个数量级。

    arXiv:2610.20041v1 Announce Type: new  Abstract: A staged physics-informed neural network (PINN) is proposed for identifying the inductance, capacitance and parasitic resistances of a DC buck converter from time-domain data. Gradients are released in the order of L to C to rL to rC, a sequence derived from sensitivity analysis. The five-stage schedule recovers L and C to within 5% and the series ESR to 10%, while a three-stage variant attains the same accuracy 11% faster. An 84 factorial sweep confirms that the trained PINN remains accurate for simultaneous 8% perturbations of all four components, achieving a waveform RMSE of 2.6 x 10-2 over 5 ms-an order-of-magnitude better than a simultaneous-training baseline.
    
[^95]: 加权伴随匹配：使用时间依赖控制代价微调流匹配模型

    Weighted Adjoint Matching: Finetuning Flow Matching Models with Time-Dependent Control Cost

    [https://arxiv.org/abs/2610.20035](https://arxiv.org/abs/2610.20035)

    提出加权伴随匹配方法，通过在控制代价上引入时间依赖权重，早期保持粗略语义特征的多样性与平衡、后期精细化细节以提升奖励，从而在微调流匹配模型时实现奖励与模式平衡之间更好的权衡。

    

    我们提出了加权伴随匹配，这是一种在给定可微奖励的情况下微调预训练扩散模型或流匹配模型的方法。该方法源自伴随匹配，后者将微调重新表述为具有二次控制代价的随机最优控制问题。基于这些模型先生成粗略特征、后生成精细细节的观察，我们在控制代价上引入了时间依赖的权重。在生成过程早期施加高代价可以保持携带大部分语义信息的粗略特征的平衡性与多样性，而在最后阶段施加低代价则允许模型通过细化细节来提高奖励。我们证明了该方法等价于在生成路径上施加时变KL散度正则化条件下最大化奖励。在高斯混合分布和MNIST数据集上的实验表明，与伴随匹配相比，我们的方法在奖励和模式平衡之间实现了更好的权衡。

    arXiv:2610.20035v1 Announce Type: cross  Abstract: We introduce Weighted Adjoint Matching, a method to finetune a pretrained diffusion or flow matching model given a differentiable reward. We derive it from Adjoint Matching, which reformulates finetuning as a stochastic optimal control problem with a quadratic control cost. Building on the observation that these models generate coarse features before fine details, we place a time-dependent weight on the control cost. A high cost early in the generative process preserves balance and diversity in the coarse features, which carry most of the semantics, while a low cost at the end lets the model increase the reward by refining details. We prove that our method is equivalent to maximizing the reward under time-varying Kullback--Leibler regularization on the generative paths. Experiments on Gaussian mixtures and on MNIST show a better tradeoff between reward and mode balance than Adjoint Matching.
    
[^96]: 关于半梯度TD的收敛性：平均场视角

    On the Convergence of Semi-Gradient TD: A Mean Field Perspective

    [https://arxiv.org/abs/2610.20014](https://arxiv.org/abs/2610.20014)

    该论文在平均场机制下分析了两层神经网络中半梯度时序差分学习的收敛性，证明了当熵诱导的松弛主导移动自举目标的敏感性时，算法可指数收敛至唯一不动点，且该稳定性机制可扩展至基于软Q学习的熵正则化离策略控制。

    

    半梯度时序差分学习是强化学习的核心算法组件，但结合函数逼近的自举机制通常会产生非保守向量场，并可能导致不稳定。我们在平均场机制下针对两层神经网络分析了这一效应，并识别出一种简单的稳定性机制：熵诱导的松弛效应主导了移动自举目标的敏感性。在该条件下，我们证明了算法向唯一不动点的指数收敛，并量化了其相对于无正则化贝尔曼解的偏差。我们还证明了同样的机制可通过软Q学习扩展到熵正则化的离策略控制。

    arXiv:2610.20014v1 Announce Type: new  Abstract: Semi-gradient temporal-difference learning is a core algorithmic component of reinforcement learning, but bootstrapping with function approximation generally induces a non-conservative vector field and can lead to instability. We analyze this effect for two-layer neural networks in the mean-field regime and identify a simple stability mechanism in which entropy-induced relaxation dominates the sensitivity of the moving bootstrap target. Under this condition, we prove exponential convergence to a unique fixed point and quantify its bias relative to the unregularized Bellman solution. We also show that the same mechanism extends to entropy-regularized off-policy control through soft Q-learning.
    
[^97]: ChemArgus：面向实验室级多模态化学推理的自动化科学诊断：一个前沿基准

    ChemArgus: Automated Scientific Diagnostics for Lab-Grade Multimodal Chemistry Reasoning: A Frontier Benchmark

    [https://arxiv.org/abs/2610.20012](https://arxiv.org/abs/2610.20012)

    ChemArgus是一个竞赛级多模态化学推理前沿基准，通过细粒度分步评分（含部分得分）和真实图像配受控文本描述的题目设计，能够诊断模型推理链在何处断裂，而不仅评估最终答案。

    

    化学评估已经从选择题探测转向开放式专家问题，然而定义当前前沿的基准仍然只对答案终点进行评分。这种评分方式将记忆的事实与逐步推导混为一谈，将视觉和文本输入视为可互换的，并将长程规划与逐步执行混淆。这些盲点使得无法诊断模型的推理链在何处断裂。我们提出了ChemArgus，这是一个竞赛级化学基准，取材于化学竞赛材料、教科书和研究论文，其评分体系用三个测量属性取代了单纯的答案终点评分。细粒度评分对每个子问题逐点评分并给予部分得分，因此分数落在步骤层面而非结果层面。多模态能力在每道包含真实图像的题目上进行考核，并配有受控的文本描述进行对照，因此视觉理解……

    arXiv:2610.20012v1 Announce Type: new  Abstract: Chemistry evaluation has moved past multiple-choice probes toward open-ended expert problems, yet the benchmarks that define the current frontier still score answer endpoints alone. Such scoring conflates memorized facts with stepwise derivation, treats visual and textual inputs as interchangeable, and confounds long-horizon planning with stepwise execution. These blind spots prevent any diagnosis of where the reasoning chain breaks in a model. We present ChemArgus, a competition-level chemistry benchmark drawn from chemistry competition material, textbooks, and research papers, whose scoring replaces the answer endpoint with three measurement properties. Fine-grained scoring grades every subquestion point by point with partial credit, so a score lands at the level of the step rather than the outcome. Multimodal capability is priced on every problem containing a real image, paired with a controlled text description, so visual understandi
    
[^98]: 使用双曲目标条件强化学习超越1000层欧几里得网络

    Outperforming 1000 Euclidean Layers With Hyperbolic Goal-Conditioned RL

    [https://arxiv.org/abs/2610.20009](https://arxiv.org/abs/2610.20009)

    通过将对比强化学习中的欧几里得距离替换为双曲空间的测地距离，仅64层深的双曲智能体就能以超过12倍的训练速度和约13倍的参数量优势，超越1024层欧几里得网络的导航性能。

    

    将自监督对比强化学习（CRL）扩展到1000层欧几里得神经网络已经实现了最先进的目标达成性能，但其实际运行时间随深度线性增长。我们发现，改变几何结构可以用浅得多的网络实现更强的性能。这在复杂导航环境中尤为重要，因为墙壁会使路径距离与欧几里得距离产生显著偏差。在U形迷宫中，我们证明了路径越长，其与欧几里得距离的偏差就越大。双曲CRL通过将CRL对比目标中的欧几里得距离替换为双曲空间中的测地距离来解决这种不匹配问题。在困难导航任务上，归纳偏置胜过规模：深度为64的双曲智能体超越了1024层深的欧几里得网络，同时训练速度快12倍以上，参数量少约13倍。我们观察到双曲智能体的行为在质上……

    arXiv:2610.20009v1 Announce Type: new  Abstract: Scaling self-supervised contrastive reinforcement learning (CRL) to 1000-layer Euclidean neural networks has enabled state-of-the-art goal-reaching performance, yet wall-clock time scales linearly with depth. We find that changing the geometry achieves stronger performance with far shallower networks. This matters in complex navigation environments, where walls can make route distance deviate substantially from Euclidean distance. In a U-shaped maze, we show that the longer the route, the more it deviates from Euclidean distance. Hyperbolic CRL addresses this mismatch by replacing Euclidean distances in CRL's contrastive objective with geodesic distances in hyperbolic space. On hard navigation tasks, inductive bias beats scale: depth 64 hyperbolic agents outperform 1024-layer-deep Euclidean networks while training more than $12 \times$ faster and using about $13 \times$ fewer parameters. We observe that hyperbolic agents behave qualitati
    
[^99]: 连续马尔可夫链伪谱隙的极小极大估计及其在经验集中界中的应用

    Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds

    [https://arxiv.org/abs/2610.20007](https://arxiv.org/abs/2610.20007)

    本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。

    

    我们研究了基于单条轨迹观测来对马尔可夫链伪谱隙 $\gamma_{ps}$ 进行极小极大估计的问题。通过扩展 Wolfer 和 Kontorovich (2024) 关于有限状态空间链的早期工作，我们针对紧致连续状态空间的情形提出了一个估计器。在转移算子具有 Lipschitz 密度的链类上，我们的估计器在相差对数因子的意义下似乎是极小极大最优的。将该估计器与 Paulin 版本的 Bernstein 不等式相结合，可以为马尔可夫链推导出经验集中不等式。这开启了许多应用，包括置信区间、主动学习和泛化界。特别地，对于具有马尔可夫奖励的静止老虎机问题，我们提出了一个 UCB 算法版本，其遗憾界与独立同分布情形类似，且无需对链的伪谱隙有任何先验知识。

    arXiv:2610.20007v1 Announce Type: cross  Abstract: We study the minimax estimation of the pseudo-spectral gap $\gamma_{ps}$ of a Markov chain from the observation of a single trajectory. Extending earlier works on finite-state space chains by Wolfer and Kontorovich (2024), we provide an estimator in the case of a compact, continuous state-space. Our estimator appears to be minimax-optimal on the class of chains with a transition operator with Lipschitz density, up to logarithmic terms. Our estimator, combined with Paulin's version of Bernstein inequality, allows to derive empirical concentration inequalities for Markov chains. This unlocks many applications, from confidence intervals, to active learning and generalization bounds. In particular, for rested bandits with Markov rewards, we provide a version of UCB with a regret bound similar to the one in the iid case, without assuming any knowledge on the pseudo-spectral gap of the chains.
    
[^100]: 面向上下文强化学习的可证明Transformer上下文压缩

    Provable Transformer Context Compression for In-Context Reinforcement Learning

    [https://arxiv.org/abs/2610.20004](https://arxiv.org/abs/2610.20004)

    该论文从理论上首次刻画了Transformer在上下文强化学习中进行TD评估所需的最小上下文维度（即由采样转移诱导的Krylov维度），并证明当上下文维度低于该阈值时，最优近似误差以 Θ(γ^m·max{1,1/(m(1-γ))}) 的速率衰减，从而回答了此类学习本质上需要多少上下文这一可扩展性问题。

    

    Transformer可以在不更新参数的情况下执行上下文强化学习。然而，现有的理论构造通常要求上下文大小随采样转移数量的增长而增长，导致大规模问题中内存成本不断增加。这引出了一个根本性的可扩展性问题：执行此类学习本质上需要多少上下文？我们针对固定策略TD评估回答了这个问题。我们证明，在共享线性解码器的情况下，精确复现前 $L$ 次TD迭代所需的最小上下文维度是由采样转移所诱导的Krylov维度。当可用的上下文维度 $m$ 低于该阈值时，精确复现不再可能。在这种情况下，我们刻画了最优近似误差，并证明其以 $\Theta(\gamma^m\max\{1,1/(m(1-\gamma))\})$ 的速率衰减，其中 $\gamma$ 为折扣因子。从表示……

    arXiv:2610.20004v1 Announce Type: new  Abstract: Transformers can perform in-context reinforcement learning without parameter updates. However, existing theoretical constructions often require context sizes that grow with the number of sampled transitions, leading to increasing memory costs in large-scale problems. This raises a fundamental scalability question: how much context is intrinsically necessary to carry out such learning? We answer this question for fixed-policy TD evaluation. We show that, with a shared linear decoder, the minimum context dimension required to exactly reproduce the first $L$ TD iterates is the Krylov dimension induced by the sampled transitions. When the available context dimension $m$ falls below this threshold, exact reproduction is no longer possible. In this regime, we characterize the optimal approximation error and show that it decays as $\Theta(\gamma^m\max\{1,1/(m(1-\gamma))\})$ where $\gamma$ is the discounting factor. Moving from representation co
    
[^101]: 储备池计算在少样本序列模式任务中的经验边界

    Empirical Boundaries of Reservoir Computing for Few-Shot Sequence Pattern Tasks

    [https://arxiv.org/abs/2610.19997](https://arxiv.org/abs/2610.19997)

    本文通过极简的少样本序列到序列模式任务，对回声状态网络（ESN）在分布、架构与优化层面的多种变体进行了实证评估，以刻画储备池计算在记忆与模式识别能力上的优势边界与局限。

    

    高效的序列建模仍然是机器学习中的一项根本性挑战。尽管Transformer取得了巨大成功，循环神经网络（RNN）最近又重新成为高效的替代方案。特别是，回声状态网络（ESNs）提供了一种引人注目的随机化RNN框架，已被广泛研究用于时间序列预测和动力系统。本文对ESN在分布、架构和优化方面的各种变体进行了实证分析，并在少样本序列到序列（seq2seq）模式任务上进行了评估。这些合成任务利用离散符号序列构建，其特征包括延迟、静默阶段和触发标记，用以隔离并评估模型的记忆能力和模式识别能力。这些任务的极简性对于理解机器学习模型的优势与局限性至关重要。此外，当输入信号采用独热编码时，ESN的动力学受到……（摘要原文在此处截断）

    arXiv:2610.19997v1 Announce Type: new  Abstract: Efficient sequence modeling remains a fundamental challenge in machine learning. Despite the success of Transformers, recurrent neural networks (RNNs) have recently re-emerged as efficient alternatives. Particularly, Echo State Networks (ESNs) offer a compelling framework of randomized RNNs that have been extensively studied for time-series forecasting and dynamical systems. This paper presents an empirical analysis of distributional, architectural, and optimization variants of ESNs evaluated on few-shot sequence-to-sequence (seq2seq) pattern tasks. These synthetic tasks utilize discrete symbol sequences, characterized by delays, inactive phases, and triggered markers to isolate and evaluate memory and pattern recognition capabilities. The minimality of these tasks is crucial to understand the strengths and limitations of machine learning models. Furthermore, when the input signal utilizes a one-hot encoding, ESN dynamics are governed st
    
[^102]: LatentLoop：通过非对称计算分配扩展循环MoE模型

    LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation

    [https://arxiv.org/abs/2610.19975](https://arxiv.org/abs/2610.19975)

    该论文发现循环MoE模型的两次循环访问存在非对称性，并提出LatentLoop方法，通过压缩第一次循环访问的序列、保留第二次的完整分辨率，并将节省的计算重新分配以增大隐藏层维度，从而在匹配预算下提升模型性能。

    

    循环混合专家模型将参数重用与稀疏计算相结合，以在匹配的计算预算下提升性能。先前的研究表明，即使共享参数，不同的循环访问也可以发挥不同的作用。为了探究不同循环访问的贡献，我们在一个将共享模块应用两次的模型中学习独立的残差系数，观察到第一次访问的尺度较小。固定系数实验表明，降低第一次访问的尺度能够保持甚至提升性能，而对第二次访问进行相同的削减则会损害性能。这种非对称性促使我们探索非对称的计算分配方案。我们提出LatentLoop，该方法压缩第一次访问的序列，在第二次访问中保留完整分辨率，并将节省下来的计算重新分配用于增大隐藏层维度。我们进一步学习第一次访问对完整序列的更新强度，并在各次访问之间共享MoE权重（摘要在此处截断）。

    arXiv:2610.19975v1 Announce Type: new  Abstract: Looped Mixture-of-Experts (MoE) models combine parameter reuse with sparse computation to improve performance under matched budgets. Prior work demonstrates that different loop visits can play different roles even when they share parameters. To probe the contributions of different loop visits, we learn separate residual coefficients in a model that applies a shared block twice, observing a smaller scale in the first visit. Fixed coefficient experiments show that reducing the first visit's scale preserves or improves performance, whereas the same reductions to the second hurt it. This asymmetry motivates us to explore asymmetric compute allocation. We propose \textbf{LatentLoop}, which compresses the first visit's sequence, retains full resolution in the second, and reallocates saved computation to increase the hidden size. We further learn how strongly the first visit updates the full sequence, and share MoE weights across visits while u
    
[^103]: Safe-FCL：面向大语言模型联邦持续学习的持续风险感知与细粒度安全聚合

    Safe-FCL: Continual Risk Awareness and Fine-Grained Safety Aggregation for Federated Continual Learning of LLMs

    [https://arxiv.org/abs/2610.19968](https://arxiv.org/abs/2610.19968)

    提出Safe-FCL框架，通过持续风险感知（CRA）机制跨任务学习新风险并保留历史恶意模式知识，结合细粒度安全聚合（FGSA）机制，解决大语言模型联邦持续学习中恶意更新模式不断演变导致防御失效的问题。

    

    联邦微调使大语言模型（LLM）能够从分布式客户端数据中协同学习，但恶意客户端更新可能会危害全局模型的安全性。随着联邦任务的不断演进，恶意更新模式的变化对现有防御机制长期维持可靠风险检测的能力构成了挑战。为应对这一挑战，我们提出了Safe-FCL，一个将持续风险感知与细粒度安全聚合相结合的大语言模型联邦持续学习框架。具体而言，持续风险感知（CRA）机制跨顺序任务学习新出现的风险模式，并利用从历史特征中导出的保护性子空间来约束检测器的更新，在适应新风险的同时缓解对先前所学恶意模式的遗忘。为了将风险感知转化为聚合层面的安全控制，细粒度安全聚合（FGSA）机制结合客户端级别的风险（信息进行安全聚合）。

    arXiv:2610.19968v1 Announce Type: new  Abstract: Federated fine-tuning enables large language models (LLMs) to learn collaboratively from distributed client data, but malicious client updates can compromise global model safety. As federated tasks evolve, changing malicious update patterns challenge the ability of existing defenses to maintain reliable risk detection over time. To address this challenge, we propose Safe-FCL, a framework integrating continual risk awareness with fine-grained safety aggregation for federated continual learning of LLMs. Specifically, the Continual Risk Awareness (CRA) mechanism learns emerging risk patterns across sequential tasks and constrains detector updates using a protective subspace derived from historical features, mitigating forgetting of previously learned malicious patterns while adapting to new risks. To translate risk awareness into aggregation-level safety control, the Fine-Grained Safety Aggregation (FGSA) mechanism combines client-level ris
    
[^104]: 从内在维度选择注意力头数量：一项小规模研究

    Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study

    [https://arxiv.org/abs/2610.19966](https://arxiv.org/abs/2610.19966)

    本文提出一种无需标签的启发式方法，通过嵌套Dropout VAE估计词元表示的内在维度来预先确定transformer注意力头的数量，实验表明按内在维度设置约10个注意力头的模型与32个头性能统计等效，且注意力参数减少69%。

    

    Transformer中注意力头的数量通常由惯例或网格搜索来确定。我们测试了一种简单的启发式方法来预先设置这一数量：在无标签的情况下估计词元表示的内在维度，并使用大致相同数量的注意力头。我们在WikiText-2的DistilBERT嵌入上训练了一个嵌套Dropout变分自编码器（VAE）来估计内在维度，并在VAE率失真曲线的拐点处选择KL权重，而非手动选择。由此得到k*=9.2（95%置信区间[8.6, 9.8]）。在一个从零训练的小型4层transformer中，10个注意力头与32个注意力头在SST-2和MRPC任务上统计等效（10个随机种子的配对TOST检验，±1分容差），而10头模型的注意力参数减少了69%。从一开始就训练10个头也略优于先训练32个头再剪枝到10个头的做法（0.6分，p=0.008）。需要说明的是，证据范围较为有限：仅涉及两个分类任务、一种小型架构、未使用预训练模型。

    arXiv:2610.19966v1 Announce Type: new  Abstract: The number of attention heads in a transformer is usually set by convention or grid search. We test a simple heuristic for setting it in advance: estimate the intrinsic dimension of token representations without labels, and use roughly that many heads. We estimate intrinsic dimension with a nested-dropout VAE trained on DistilBERT embeddings of WikiText-2, choosing the KL weight at the knee of the VAE's rate-distortion curve rather than by hand. This gives $k^*=9.2$ (95% CI [8.6, 9.8]). In a small 4-layer transformer trained from scratch, 10 heads and 32 heads are statistically equivalent on SST-2 and on MRPC (paired TOST over 10 seeds, $\pm1$ point margin), with 69% fewer attention parameters for the 10-head model. Training 10 heads from the start also slightly outperforms training 32 heads and pruning to 10 (0.6 points, $p=0.008$). The evidence is narrow: two classification tasks, one small architecture, no pretrained models, and no fo
    
[^105]: AtomSearch：面向智能体技能优化的基于证据的树搜索

    AtomSearch: Evidence-Grounded Tree Search for Agent Skill Optimization

    [https://arxiv.org/abs/2610.19951](https://arxiv.org/abs/2610.19951)

    AtomSearch 通过对比式原子编辑和技能状态树搜索，以证据为依据实现局部化、可回溯的智能体技能优化，在多个任务和模型上超越了 SkillOpt。

    

    智能体技能通过可编辑的外部指令来适配冻结的模型。一次性修改多条规则会使改动难以审查，而仅凭即时得分来判断每次局部修改可能会丢弃有用的中间状态。我们提出 AtomSearch，它保持修改动作的局部性，同时利用后续的技能状态来引导搜索。对比式原子编辑将每个原始规则修改与失败和成功的执行、一个行为假设以及预期作用范围相关联。技能状态树搜索（STS）评估较短的编辑组合，并选择性地将其保留为可回访、可扩展的可执行状态。结果记忆为后续的修改提案提供已观察到的修复与回归统计数据。我们在四个任务和五个目标模型上评估了 AtomSearch。使用 deepseek-v4-flash 时，它在四个任务中的三个上超越了 SkillOpt，其中包括在 LiveMathematicianBench 上取得 9.68 个百分点的提升，以及 SkillOpt 的最终（摘要在此处截断）

    arXiv:2610.19951v1 Announce Type: new  Abstract: Agent skills adapt frozen models through editable external instructions. Revising several rules at once makes changes difficult to inspect, while judging each local edit only by its immediate score can discard useful intermediate states. We introduce AtomSearch, which keeps revision actions local while using subsequent skill states to guide search. Contrastive atomic editing links each primitive rule edit to failed and successful executions, a behavioral hypothesis, and an intended scope. Skill-State Tree Search (STS) evaluates short edit compositions and selectively retains them as executable states that can be revisited and extended. Outcome memory supplies later proposals with observed repair and regression statistics. We evaluate AtomSearch across four tasks and five target models. With deepseek-v4-flash, it exceeds SkillOpt on three of four tasks, including a 9.68-percentage-point gain on LiveMathematicianBench, and SkillOpt's final
    
[^106]: 面向有限时间生成的图能量学习

    Learning Graph Energies for Finite-Time Generation

    [https://arxiv.org/abs/2610.19946](https://arxiv.org/abs/2610.19946)

    该论文提出“有限时域能量学习”方法，为固定步数后停止的 Metropolis-Hastings 图生成采样器学习接受能量，利用仅训练阶段使用的判别器和接受/拒绝分支的耦合延续获得无偏的接受梯度估计，从而在不依赖平衡负样本或离散状态微分的情况下实现高质量图生成。

    

    能量基模型可以表示分子图等离散结构上的分布，但在平衡状态下准确的能量，并不一定能在采样器运行固定步数后停止时产生准确的样本。我们提出了有限时域能量学习（finite-horizon energy learning），它将 Metropolis--Hastings 采样器的接受能量适配到该采样器在规定步数后返回的分布，同时保持初始化、提议分布和退火时间表固定不变。对于图生成任务，一个冻结的预训练模型提供初始化和提议分布，而学习得到的能量控制接受步骤。一个仅在训练阶段使用的判别器对完整样本进行评分，通过对提议的接受分支和拒绝分支进行耦合的延续，可以得到一个接受概率梯度估计器，在所述条件下对于固定的判别器是无偏的，且无需平衡的负样本，也无需对离散状态进行微分。生成过程使用学习到的……（摘要在此处被截断）

    arXiv:2610.19946v1 Announce Type: new  Abstract: Energy-based models can represent distributions over discrete structures such as molecular graphs, but an energy that is accurate at equilibrium need not yield accurate samples when the sampler stops after a fixed number of steps. We introduce finite-horizon energy learning, which adapts the acceptance energy of a Metropolis--Hastings sampler to the distribution it returns after a prescribed number of steps, with initialization, proposals, and annealing schedule held fixed. For graph generation, a frozen pretrained model supplies initialization and proposals, and a learned energy governs acceptance. A training-only discriminator scores completed samples, and coupled continuations of the accepted and rejected branches of a proposal give an acceptance-gradient estimator that is unbiased for a fixed discriminator under stated conditions, without equilibrated negative samples or differentiation through discrete states. Generation uses the le
    
[^107]: MorphingNet：网络即公式的符号回归

    MorphingNet: Symbolic Regression Where the Network Is the Formula

    [https://arxiv.org/abs/2610.19942](https://arxiv.org/abs/2610.19942)

    MorphingNet将每个函数的形式本身作为可训练参数，把符号回归转化为连续优化问题，无需算子库即可在训练全程保持公式可读，并在SRBench上达到最先进精度。

    

    我们提出了MorphingNet，一种用于符号回归（SR）的网络，其单元可以连续地改变自身的函数形式。MorphingNet不再基于固定算子库构建的表达式进行搜索，而是将每个函数的形式本身作为参数进行训练，从而使符号回归成为一个连续优化问题。其单元要么是无极点有理函数，要么是复指数-对数函数，随着参数的变化，它们可以在指数函数、正弦函数、阻尼振荡、幂律和对数函数之间平滑转换。该网络在训练的每一步都是一个可读的公式，且训练过程无需算子库；随后的结构恢复步骤会将训练好的公式重写为简短的基本初等表达式。训练过程仅为稠密线性代数运算，因此对GPU友好，并可扩展到大规模数据。MorphingNet在SRBench上达到了最先进的精度，在合成基准上恢复了大多数生成公式，在大规模数据上性能最高可提升至……（原文截断）。

    arXiv:2610.19942v1 Announce Type: new  Abstract: We introduce MorphingNet, a network for symbolic regression (SR) whose units change their function continuously. Instead of searching over expressions built from a fixed operator library, MorphingNet trains the form of each function as a parameter, so SR becomes one continuous optimization problem. Its units are either pole-free rational functions or complex exponential--logarithm functions, which move between exponentials, sinusoids, damped oscillations, power laws and logarithms as their parameters change. The network is a readable formula at every step of training, and no operator library is needed to train it; a structure-recovery step afterwards rewrites the trained formula into short elementary expressions. Training is dense linear algebra, so it is GPU-friendly and scales to large data. MorphingNet achieves state-of-the-art accuracy on SRBench, recovers most generating formulas on synthetic benchmarks, and on large data is up to t
    
[^108]: SAM3RL：基于强化学习的视觉目标跟踪记忆控制

    SAM3RL: Memory Control via Reinforcement Learning for Visual Object Tracking

    [https://arxiv.org/abs/2610.19931](https://arxiv.org/abs/2610.19931)

    SAM3RL 将 SAM 3 跟踪器的记忆控制形式化为序贯决策任务，通过强化学习训练轻量级记忆控制器来决定哪些帧进入记忆库，从而在 12 个基准中的 10 个上创造了新的最先进纪录。

    

    Segment Anything Model 3（SAM 3）是一个用于图像和视频分割的基础模型，在视觉目标跟踪方面达到了最先进的水平。其跟踪器继承自 SAM 2，将过去帧的信息存储在一个记忆库中，由简单的先进先出（FIFO）规则进行管理，从而在复杂视频中保持时间一致性。近期的一些方法通过手工设计的记忆更新规则来改进 SAM 2，每种方法都针对特定的失败模式，例如干扰物、遮挡或物体运动。然而，这些方法带来的提升只能部分迁移到更强大的 SAM 3 主干网络上。为此，我们提出了一种更通用、数据驱动的方法，将记忆控制形式化为一个序贯决策任务。SAM3RL 是 SAM 3 跟踪器的扩展，配备了一个通过强化学习训练的轻量级记忆控制器，由它决定哪些帧进入记忆库，以最大化跟踪性能。SAM3RL 在 12 个基准中的 10 个上创造了新的最先进纪录。

    arXiv:2610.19931v1 Announce Type: cross  Abstract: Segment Anything Model 3 (SAM 3) is a foundation model for image and video segmentation that sets the state of the art in visual object tracking. Its tracker, inherited from SAM 2, stores information from past frames in a memory bank, managed by a simple First-In-First-Out rule, allowing temporal consistency in complex videos. Several recent methods improve SAM 2 with hand-crafted memory update rules, each targeting a specific failure mode, such as distractors, occlusions, or object motion. However, their gains transfer only partially to the stronger SAM 3 backbone. We instead present a more general, data-driven approach that frames memory control as a sequential decision-making task. SAM3RL is an extension of the SAM 3 tracker with a lightweight memory controller, trained by reinforcement learning, that decides which frames enter the memory bank in order to maximize tracking performance. SAM3RL sets a new state of the art on 10 of 12 
    
[^109]: 高效全局优化中高斯过程汤普森采样的收敛速率

    Convergence Rates of Gaussian Process Thompson Sampling for Efficient Global Optimization

    [https://arxiv.org/abs/2610.19925](https://arxiv.org/abs/2610.19925)

    本文首次在无噪声RKHS设定下为高斯过程汤普森采样建立了最坏情况简单遗憾上界 $O(N^{-\nu/d}\sqrt{\log N})$ 及匹配的下界，证明对数惩罚项是该算法的固有代价。

    

    高效全局优化旨在在有限的评估预算 $N$ 内找到昂贵的黑盒函数的全局最优解。我们在无噪声观测条件下研究高斯过程汤普森采样（GP-TS）的收敛速率，并假设目标函数固定于再生核希尔伯特空间（RKHS）中。对于维度为 $d$、光滑度参数为 $\nu$ 的Matérn核，我们为足够精细网格上的离散GP-TS及其连续版本建立了最坏情况下期望简单遗憾上界 $O(N^{-\nu/d}\sqrt{\log N})$。对于连续实现，我们开发了一种嵌套网格论证方法来控制整个定义域上的后验样本路径。我们进一步在固定的一维Matérn-5/2核下，为两种实现建立了匹配的下界 $\Omega(N^{-5/2}\sqrt{\log N})$，证明了对数惩罚项是标准GP-TS在此类设定中的固有特性。

    arXiv:2610.19925v1 Announce Type: cross  Abstract: Efficient global optimization aims to identify a global optimum of an expensive black-box function within a limited evaluation budget $N$. We investigate the convergence rates of Gaussian process Thompson sampling (GP-TS) under noiseless observations, assuming a fixed objective function in a reproducing kernel Hilbert space (RKHS). For Mat\'ern kernels with smoothness parameter $\nu$ in dimension $d$, we establish a worst-case expected simple-regret upper bound of $O(N^{-\nu/d}\sqrt{\log N})$ for both discrete GP-TS on sufficiently fine grids and its continuous counterpart. For the continuous implementation, we develop a nested-grid argument to control posterior sample paths over the entire domain. We further establish matching lower bounds of $\Omega(N^{-5/2}\sqrt{\log N})$ for both implementations under a fixed one-dimensional Mat\'ern-$5/2$ kernel, demonstrating that the logarithmic penalty is intrinsic to standard GP-TS in this set
    
[^110]: MolHop：基于智能体强化学习的骨架跃迁分子设计

    MolHop: Scaffold-Hopping Molecular Design with Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.19920](https://arxiv.org/abs/2610.19920)

    MolHop 提出了一个基于智能体强化学习的跨骨架分子发现框架，将约 160 万个商业分子骨架组织成从粗到细的分层动作空间，使模型能够发现骨架完全不同但保留参考分子功能特性的新分子。

    

    分子设计通常被表述为在现有分子空间上的优化问题。然而，分子优化与发现结构上截然不同的替代分子代表了两个本质不同的搜索问题。基于大语言模型（LLM）的分子设计智能体通常执行前者，即在现有结构的邻域内进行局部细化，通过结构修饰来改善目标性质。相比之下，后者需要探索更广泛的化学空间，以发现具有完全不同骨架同时保留参考分子功能特性的分子。我们提出了 MolHop，一个用于跨骨架分子发现的智能体框架，使模型能够在预先组织好的化学空间中进行导航和编辑。MolHop 将来自商业分子的约 160 万个分子骨架组织成一个从粗到细的分层动作空间。

    arXiv:2610.19920v1 Announce Type: cross  Abstract: Molecular design is often formulated as an optimization problem over existing molecular spaces. However, molecular optimization and the discovery of structurally distinct replacement molecules represent two fundamentally different search problems. LLM-based molecular design agents typically perform the former, conducting local refinement within the neighborhood of existing structures to improve target properties through structural modifications. In contrast, the latter requires exploration of a broader chemical space to discover molecules with entirely different scaffolds while preserving the functional characteristics of reference molecules. We propose MolHop, an agentic framework for cross-scaffold molecular discovery that enables models to navigate and edit within a pre-organized chemical space. MolHop organizes approximately 1.6 million molecular scaffolds from commercial molecules into a coarse-to-fine hierarchical action space. T
    
[^111]: 通过双空间轨道学习超越DFT电子能带结构

    Beyond DFT electronic band structures via dual-space orbital learning

    [https://arxiv.org/abs/2610.19919](https://arxiv.org/abs/2610.19919)

    O-kRystal框架利用低成本的PBE计算提供的实空间与倒易空间轨道信息，以约0.171 eV的平均绝对误差预测高精度HSE06能带结构，大幅降低了高保真电子结构计算的成本。

    

    精确的电子能带结构对于预测材料性质至关重要，但高保真度的计算往往需要高昂的计算成本。我们提出了O-kRystal，一个双空间轨道学习框架，它将成本较低的Perdew-Burke-Ernzerhof (PBE)计算同时用作能量基线和结构化轨道信息的来源。通过将PBE能带能量与实空间和倒易空间的轨道矩阵相结合，O-kRystal能够预测单个电子态作为晶体动量和自旋函数的更高精度修正。对于距带边中点5 eV以内的电子态，在涵盖绝缘体、金属、磁性和非磁性体系的近6000种三维材料上，该方法预测Heyd-Scuseria-Ernzerhof (HSE06)能带能量的平均绝对误差为0.171 ± 0.009 eV。在一个包含六种强关联材料的独立基准测试中，该方法将相应的PBE能带能量MAE降低了68.7%，并能分辨……

    arXiv:2610.19919v1 Announce Type: new  Abstract: Accurate electronic band structures are essential for predicting material properties, but high-fidelity calculations are computationally demanding. We introduce O-kRystal, a dual-space orbital-learning framework that uses a lower-cost Perdew-Burke-Ernzerhof (PBE) calculation as both an energy baseline and a source of structured orbital information. Combining PBE band energies with real- and reciprocal-space orbital matrices, O-kRystal predicts higher-level corrections for individual electronic states as a function of crystal momentum and spin. For states within 5 eV of the band-edge midpoint, it predicts Heyd-Scuseria-Ernzerhof (HSE06) band energies with a mean absolute error of $0.171 \pm 0.009$ eV across nearly 6,000 three-dimensional materials spanning insulating, metallic, magnetic, and nonmagnetic systems. On a separate benchmark of six strongly correlated materials, it reduces the corresponding PBE band-energy MAE by 68.7% and reso
    
[^112]: CARD：闭合师生循环以实现对抗鲁棒性蒸馏

    CARD: Closing the Teacher-Student Loop for Adversarial Robustness Distillation

    [https://arxiv.org/abs/2610.19911](https://arxiv.org/abs/2610.19911)

    提出闭环循环对抗鲁棒性蒸馏框架CARD，将蒸馏后的学生模型作为反馈来源改进教师模型，再将教师增益回传给学生，从而突破传统单向蒸馏范式并提升对抗鲁棒性。

    

    对抗鲁棒性蒸馏通常将鲁棒教师模型视为固定的知识来源，并以单向方式将鲁棒性传递给紧凑的学生模型。这种范式在学生模型完成蒸馏后便终止了师生之间的交互，从而丢弃了来自蒸馏后学生模型的潜在有用反馈。然而，利用这种反馈并非易事，因为不加区分的弱到强知识迁移可能会将学生的错误传播给教师。为解决这一问题，我们提出了循环对抗鲁棒性蒸馏（CARD），这是一个闭环框架，它将蒸馏后的学生转化为反馈来源，用于改进鲁棒教师模型，随后将教师的增益迁移回同一个学生。CARD首先执行强到弱的对抗蒸馏，然后使用可靠性增强的学生反馈来改进鲁棒教师模型。该改进过程将逆向对抗学生反馈与师生（信息）相结合……（摘要截断）

    arXiv:2610.19911v1 Announce Type: new  Abstract: Adversarial robustness distillation typically treats a robust teacher as a fixed knowledge source and transfers robustness to a compact student in a one-way manner. This paradigm terminates teacher-student interaction once the student is distilled, thereby discarding potentially useful feedback from the distilled student. However, exploiting such feedback is nontrivial because indiscriminate weak-to-strong transfer may propagate student errors to the teacher. To address this issue, we propose Cyclic Adversarial Robustness Distillation (CARD), a closed-loop framework that turns the distilled student into a feedback source for refining the robust teacher and subsequently transfers the teacher gains back to the same student. CARD first performs strong-to-weak adversarial distillation and then uses reliability-enhanced student feedback to refine the robust teacher. The refinement combines inverse adversarial student feedback with teacher-stu
    
[^113]: 当非对抗性噪声损害学习时：智能体技能自我进化的基准测试

    When Non-Adversarial Noise Hurts Learning: Benchmarking Skill Self-Evolution in Agents

    [https://arxiv.org/abs/2610.19910](https://arxiv.org/abs/2610.19910)

    该论文提出了首个评估大语言模型智能体技能自我进化在非对抗性噪声下鲁棒性的基准 RSE-Bench，通过跨四个领域、六个基准的受控扰动以及与方法无关的 RSEClaw 噪声注入框架，系统分离学习阶段与评估阶段的噪声，揭示经验噪声如何影响智能体技能学习。

    

    大语言模型智能体越来越多地通过将经验蒸馏为可复用的技能来提升自身，但经验中的错误可能在学习到的规则中持续存在，并影响后续任务。现有评估主要考察干净的学习过程或技能冻结后遇到的扰动，使得对自我进化过程中鲁棒性的理解仍不充分。我们提出了 RSE-Bench，据我们所知，这是首个用于评估在普通、非对抗性噪声下技能自我进化的基准。该基准涵盖四个领域和六个基准测试，对指令、环境证据、执行记录以及结果-记录配对进行受控扰动，同时保留底层任务和原生评分规则。该协议将学习过程中的噪声与评估时的输入侧噪声分离开来，在保留任务上测试冻结的技能库。我们还引入了 RSEClaw，一个与方法无关的框架，支持一致的噪声注入和程序化的……

    arXiv:2610.19910v1 Announce Type: new  Abstract: Large language model agents increasingly improve by distilling experience into reusable skills, but errors in that experience can persist in learned rules and affect later tasks. Existing evaluations largely examine clean learning or perturbations encountered after skills are frozen, leaving robustness during self-evolution insufficiently understood. We introduce RSE-Bench, to our knowledge the first benchmark for evaluating skill self-evolution under ordinary, non-adversarial noise. It spans four domains and six benchmarks, with controlled perturbations to instructions, environmental evidence, execution records, and outcome--record pairings that preserve the underlying tasks and native scoring rules. The protocol separates noise during learning from input-side noise at evaluation, testing frozen skill stores on held-out tasks. We also introduce \textsc{RSEClaw}, a method-agnostic framework supporting consistent noise injection and proce
    
[^114]: 基于任务特定低秩适配器的多模态医学图像解析

    Multimodal Medical Image Parsing with Task-Specific Low-Rank Adapters

    [https://arxiv.org/abs/2610.19901](https://arxiv.org/abs/2610.19901)

    该论文提出在冻结的 4-bit 量化 Qwen3.5-9B 主干上，根据输入问题动态选择并定制 LoRA 适配器的秩与提示词，同时结合稀有类别重平衡与数据补充策略，使单一视觉-语言模型能够高效应对覆盖六类任务族和七种成像模态的医学图像解析挑战。

    

    将单一的视觉-语言模型适配到医学图像解析任务，需要能够反映每个任务需求的训练选择。我们针对 FLARE 2026 任务 3 研究了基于 Qwen3.5-9B 的任务特定适配方法，涵盖六个任务族和七种成像模态。我们的系统将一个共享的、冻结的 4-bit 量化主干网络与根据输入问题选择的 LoRA 适配器相结合。我们定制了适配器秩与提示词，重新平衡了稀有分类目标，并利用皮肤病学分类样本补充回归训练。在固定的内部保留集上，任务特定适配器在分类任务上取得了数据集宏平均平衡准确率 0.7791，多标签分类的数据集宏平均 micro-F1 为 0.6286，计数任务的平均绝对误差（MAE）为 11.73，检测任务的数据集宏平均 F1 为 0.8914，回归任务的数据集宏平均 MAE 为 9.78，报告生成的 CRIMSON 分数为 0.8204。在主办方包含 1,703 个样本的隐藏验证集上，

    arXiv:2610.19901v1 Announce Type: cross  Abstract: Adapting a single vision-language model to medical image parsing requires training choices that reflect the demands of each task. We study task-specific adaptation with Qwen3.5-9B for FLARE 2026 Task 3, covering six task families and seven imaging modalities. Our system couples a shared, frozen 4-bit backbone with LoRA adapters selected from the input question. We customize adapter ranks and prompts, rebalance rare classification targets, and supplement regression training with dermatology classification examples. On the fixed internal holdout, the task-specific adapters achieve a dataset-macro balanced accuracy of 0.7791 for classification, a dataset-macro micro-F1 of 0.6286 for multi-label classification, a counting mean absolute error (MAE) of 11.73, a dataset-macro detection F1 of 0.8914, a dataset-macro regression MAE of 9.78, and a report-generation CRIMSON score of 0.8204. On the organizer's 1,703-example hidden-validation set, 
    
[^115]: 高阶对象的通用样本嵌入方法

    Universal Sample-Based Embeddings of Higher-Order Objects

    [https://arxiv.org/abs/2610.19899](https://arxiv.org/abs/2610.19899)

    提出了一个统一的基于样本的嵌入框架，能将分布、函数、图等任意高阶对象编码为固定大小的向量，实现跨对象类型的通用紧凑表示，并可用于训练中未见过的对象。

    

    神经网络天然地作用于有限维向量，然而许多问题需要对分布、函数和图等高阶对象进行推理。这类对象刻画了基础集合内部的关系，且很少具备显式的紧凑表示。此外，我们通常希望得到的不是单个高阶对象的紧凑表示，而是整族此类对象的紧凑表示，其中包括训练时未曾见过的对象成员。虽然针对特定对象类型已有一些方法，但目前还没有一种方法被设计为能够跨对象类型通用。我们提出了一个统一框架，将任意高阶对象表示为在该对象族上学到的固定大小嵌入向量。编码器将仅通过有限样本集观察到的对象映射为嵌入向量，而特定领域的解码器则通过在该嵌入条件下对对象进行建模来提供训练信号。由于编码器仅从样本出发进行工作，（摘要原文在此处被截断）

    arXiv:2610.19899v1 Announce Type: new  Abstract: Neural networks operate naturally on finite-dimensional vectors, yet many problems require reasoning about higher-order objects such as distributions, functions, and graphs. Such objects capture relationships within a ground set and rarely admit an explicit compact representation. Moreover, we typically want a compact representation not of a single higher-order object but of an entire family of such objects, including members unseen during training. While a few approaches exist for particular object types, none is designed to work across object types. We propose a unified framework that represents any higher-order object as a fixed-size embedding vector learned over a family of such objects. An encoder maps an object, seen only through a finite set of samples, to an embedding, and a domain-specific decoder supplies the training signal by modeling the object conditioned on that embedding. Since the encoder works only from samples, one arc
    
[^116]: 低误报率下异常声音检测的小样本适应

    Few-Shot Adaptation for Anomalous Sound Detection at Low False-Positive Rate

    [https://arxiv.org/abs/2610.19893](https://arxiv.org/abs/2610.19893)

    该论文提出一种两阶段异常声音检测方法，在仅有正常样本时部署正则化无监督模型，并在首次故障发生后利用小样本适应策略学习异常样本，将AUPR从23.0提升至64.0、pAUROC从39.6提升至72.8，显著改善了低误报率下的检测性能。

    

    可靠的故障检测对于大型设施中的工业机械至关重要，因为意外故障可能导致昂贵的停机时间和生产中断。尽管声学监测为机器状态监测提供了一种可扩展的解决方案，但其部署往往受到高误报率的限制。我们提出了一种自动异常声音检测的两步方法：首先，当仅有正常运行的音频样本可用时，部署一个正则化的无监督模型；然后，在第一次机器故障发生后，采用小样本适应策略利用故障期间获得的异常样本来改进系统。这种适应使精确率-召回率曲线下面积（AUPR）从23.0提升至64.0，并改善了低误报率区域的检测性能，部分受试者工作特征曲线下面积从39.6提升至72.8。我们还表明，协方差……（原文摘要在此处截断）

    arXiv:2610.19893v1 Announce Type: cross  Abstract: Reliable fault detection is crucial for industrial machinery in large-scale facilities, where unexpected failures can cause costly downtime and production interruptions. Although acoustic monitoring provides a scalable solution for machine condition monitoring, its deployment is often constrained by high false-positive rates. We propose a two-step approach to automatic anomalous sound detection: first, when only normal operating audio samples are available, a regularized unsupervised model is deployed; then, after the first machine fault, a few-shot adaptation strategy exploits the anomalous samples acquired during the fault to improve the system. This adaptation increases the area under the precision-recall curve (AUPR) from 23.0 to 64.0 and improves detection in the low false-positive-rate region, with the partial area under the receiver operating characteristic curve (pAUROC) increasing from 39.6 to 72.8. We also show that covarianc
    
[^117]: 论多模态大语言模型中基于表示的测试时自适应的局限性

    On the Limits of Representation-Based Test-Time Adaptation for Multimodal LLMs

    [https://arxiv.org/abs/2610.19892](https://arxiv.org/abs/2610.19892)

    该论文证明了基于表示的测试时自适应无法恢复多模态大语言模型在损坏数据下的性能，原因在于其视觉token序列偏移方向几乎正交且遍布整个嵌入空间，这与CLIP类模型存在共享域偏移方向的情况截然不同。

    

    视觉-语言模型在常见损坏（corruptions）下会出现显著的准确率下降。虽然测试时自适应（TTA）在CLIP类对比模型中成功缓解了这类分布偏移，但其在生成式多模态大语言模型（MLLMs）中的应用仍未被充分探索，并且由于其序列化的token生成方式而面临独特的挑战。在这项工作中，我们证明基于表示的TTA无法恢复MLLMs的性能。我们的实验表明，即使通过离线利用目标数据而违反标准TTA假设，样本盲的特征对齐也只能恢复微乎其微的准确率。我们将这一失败归因于损坏偏移的本质：与CLIP类架构不同（后者受益于单一池化分类嵌入中共享的域偏移方向），MLLM的视觉token序列偏移几乎相互正交，并覆盖整个嵌入空间。这些发现界定了任何样本盲的、基于位置的……

    arXiv:2610.19892v1 Announce Type: new  Abstract: Vision-language models suffer significant accuracy drops under common corruptions. While test-time adaptation (TTA) successfully mitigates these shifts in CLIP-like contrastive models, its application to generative multimodal large language models (MLLMs) remains underexplored and uniquely challenging due to their sequential token generation. In this work, we demonstrate that representation-based TTA fails to recover performance in MLLMs. Our experiments reveal that even when violating standard TTA assumptions by leveraging target data offline, sample-blind feature alignment recovers only marginal accuracy. We attribute this failure to the nature of corruption shifts: unlike CLIP-like architectures, which benefit from a shared domain-shift direction within a single pooled classification embedding, MLLM visual token sequence shifts are nearly orthogonal and span the full embedding space.These findings bound what any sample-blind, position
    
[^118]: 基于漂移的占用度建模：单步生成实现直接策略提取

    Occupancy Modeling via Drifting: One-Step Generation Enables Direct Policy Extraction

    [https://arxiv.org/abs/2610.19883](https://arxiv.org/abs/2610.19883)

    该论文提出使用现代单步生成采样器“漂移”来建模占用度，从而通过直接微分即可从生成模型中提取策略，显著简化了从无标注数据进行强化学习的流程。

    

    对行为基础模型日益增长的兴趣，结合生成式建模的最新进展，催生了一种从无标注数据中进行强化学习的诱人方案：预训练一个生成式模型来建模占用度（即在无奖励轨迹上的折扣状态访问分布），这样一旦获得一个小的带奖励标注的数据集，就可以轻松地提取策略。我们沿着三个维度对该方案进行分解：(i) 需要在多大程度上捕捉占用度的多模态结构，(ii) 生成式模型的类别，以及 (iii) 用于策略提取的方法。在每一个维度上，我们分析了现有选择、提出了更简单的替代方案，并通过实验验证了每项决策。这一分析最终形成了一种新颖且精简的方法，该方法利用漂移——一种现代的单步生成采样器——来建模占用度。我们证明这使得可以通过对学习到的模型进行直接微分来实现策略提取……

    arXiv:2610.19883v1 Announce Type: new  Abstract: The growing interest in behavior foundation models, combined with recent advancements in generative modeling, has led to an appealing recipe for reinforcement learning from unlabeled data: pre-train a generative model to model occupancies, i.e., discounted state visitations, on reward-free trajectories, such that a policy can be easily extracted once a small reward-labeled dataset becomes available. We decompose this recipe along three axes: (i) the degree to which the multi-modal structure of occupancies must be captured, (ii) the class of generative model, and (iii) the method used for policy extraction. Along each axis, we consider existing choices, propose simpler alternatives, and validate each decision experimentally. This analysis culminates in a novel, streamlined method that leverages drifting, a modern one-step generative sampler, to model the occupancy. We show that this enables policy extraction via direct differentiation thr
    
[^119]: 神经瓦瑟斯坦样条：用于轨迹推断的表示不变二阶先验

    Neural Wasserstein Splines: Representation-Invariant Second-Order Priors for Trajectory Inference

    [https://arxiv.org/abs/2610.19878](https://arxiv.org/abs/2610.19878)

    提出神经瓦瑟斯坦样条（NWSP），通过瓦瑟斯坦平滑先验和内在瓦瑟斯坦加速度的有限维替代量，实现表示不变的二阶正则化，从而从不成对的快照中学习连续的种群演化曲线。

    

    种群动态通常只能以不成对的快照形式被观测到，这使得测量之间的演化过程欠定。二阶正则化可以约束这一演化，但对采样粒子加速度施加惩罚是对联合粒子路径的先验，而同一边际曲线的不同粒子表示可能具有不同的加速度代价。为此，我们提出了神经瓦瑟斯坦样条（Neural Wasserstein Splines, NWSP），这是一种基于样本的方法，在瓦瑟斯坦平滑先验下学习连续的种群曲线。一个神经网络将潜变量映射为时间上三次样条的控制点，因此任意时刻的样本都可以直接由样条基产生，而无需数值积分。我们将生成器的速度投影到有限标量势字典的梯度上，并对这一依赖于测度的投影进行时间求导，从而得到内在瓦瑟斯坦加速度的有限维替代量，用于训练……

    arXiv:2610.19878v1 Announce Type: new  Abstract: Population dynamics are often observed only as unpaired snapshots, which leave the evolution between measurements underdetermined. Second-order regularization can constrain it, but a penalty on the accelerations of sampled particles is a prior on joint particle paths, and different particle representations of the same marginal curve can have different acceleration costs. We therefore introduce Neural Wasserstein Splines (NWSP), a sample-based method that learns a continuous population curve under a Wasserstein smoothness prior. A neural network maps latent variables to the control points of a cubic spline in time, so samples at any time come from the spline basis without numerical integration. We project the generator's velocity onto gradients of a finite dictionary of scalar potentials and differentiate this measure-dependent projection in time, obtaining a finite-dimensional surrogate for the intrinsic Wasserstein acceleration, trained
    
[^120]: SPARS：一种面向持续学习的稳定性-可塑性感知区域选择框架

    SPARS: A Stability-Plasticity-Aware Region Selection Framework for Continual Learning

    [https://arxiv.org/abs/2610.19876](https://arxiv.org/abs/2610.19876)

    SPARS是一种无需样本回放的持续学习混合框架，通过结合全局参数隔离与基于FIM或置信度分数估计器的局部参数正则化来选择关键参数区域，在保持旧知识稳定性的同时兼顾新任务的可塑性，从而缓解灾难性遗忘。

    

    持续学习旨在获取新知识的同时保留先前学到的知识。然而，在学习新任务时更新共享参数往往会覆盖已有知识，导致灾难性遗忘。记忆回放方法可以缓解灾难性遗忘，但其依赖先前的样本，引发了隐私、内存和可扩展性方面的担忧。另一方面，基于参数隔离的方法能够保留已学到的知识，但可能限制知识迁移；而正则化方法虽然促进了适应性，但仍容易受到新旧任务之间干扰的影响。为了维持稳定性与可塑性之间的平衡，本文提出了稳定性-可塑性感知区域选择框架，这是一种无需回放的混合框架，将全局参数隔离与局部参数正则化相结合。SPARS 使用费雪信息矩阵（FIM）或一种新颖的置信度分数（CS）估计器来估计参数重要性。

    arXiv:2610.19876v1 Announce Type: new  Abstract: Continual learning aims to acquire new knowledge while retaining previously learned knowledge. However, updating shared parameters when learning new tasks often overwrites existing knowledge, causing catastrophic forgetting. Memory replay methods mitigate catastrophic forgetting but rely on previous samples, raising privacy, memory, and scalability concerns. On the other hand, parameter isolation-based methods preserve learned knowledge but may limit knowledge transfer, whereas regularization promotes adaptability but remains susceptible to interference between old and new tasks. To maintain the stability-plasticity trade-off, this paper proposes Stability-Plasticity-Aware Region Selection (SPARS), a replay-free hybrid framework combining global parameter isolation with localized parameter regularization. SPARS estimates parameter importance using either the Fisher Information Matrix (FIM) or a novel Confidence Score (CS) estimator that 
    
[^121]: Explorer-Zero：通过可分解的可验证奖励改进长程任务的自博弈

    Explorer-Zero: Improving Self-Play for Long-Horizon Tasks with Decomposable Verifiable Rewards

    [https://arxiv.org/abs/2610.19868](https://arxiv.org/abs/2610.19868)

    Explorer-Zero 提出了一种自博弈框架，通过与环境的交互同时自动生成任务和细粒度的可验证奖励，利用中间与最终状态检查点奖励部分完成，从而无需人工标注任务即可提升 LLM 智能体在长程任务上的强化学习效果。

    

    使用强化学习在长程交互任务上训练大语言模型（LLM）智能体，通常依赖于人工标注的任务和可靠的验证机制，而对于有状态的环境而言，这两者的构建成本都非常高。自博弈提供了一种替代方案，但它必须生成可实现的、可组合的任务，并提供超越二元结果的反馈。我们提出了 Explorer-Zero，一个通过与环境的交互来联合生成任务和细粒度、可验证奖励的自博弈框架。一个经过学习的提议者探索环境、构建任务，并将中间状态和最终状态记录为验证目标。这些检查点能够对部分完成给予奖励，同时通过提议者与求解者的交替训练来调整生成的课程。我们从理论上证明，当失败的回放轨迹完成了不同数量的子任务时，子任务奖励能够减少这些失败轨迹之间的坍缩。在无需人工标注训练任务的情况下，Explorer-Zero 的表现优于所评估的基线方法。

    arXiv:2610.19868v1 Announce Type: cross  Abstract: Training LLM agents on long-horizon interactive tasks with reinforcement learning typically relies on human-annotated tasks and reliable verification, both expensive to construct for stateful environments. Self-play offers an alternative, but must generate achievable, compositional tasks and provide feedback beyond binary outcomes. We propose Explorer-Zero, a self-play framework that jointly generates tasks and fine-grained, verifiable rewards through environment interaction. A learned proposer explores the environment, constructs tasks, and records intermediate and final states as verification targets. These checkpoints reward partial completion, while alternating proposer--solver training adapts the generated curriculum. We show theoretically that subtask rewards reduce collapse among unsuccessful rollouts when they complete different numbers of subtasks. Without human-annotated training tasks, Explorer-Zero outperforms the evaluated
    
[^122]: 关于过参数化加权最大均值差异（MMD）的优化景观研究

    On the Optimisation Landscape of Overparameterised Weighted MMD

    [https://arxiv.org/abs/2610.19861](https://arxiv.org/abs/2610.19861)

    该论文首次为加权MMD优化提供了有限粒子的过参数化保证，证明对于一维高斯核，在粒子数量足够多时所有局部极小值都具有任意小的MMD，且所有虚假驻点均不稳定。

    

    我们研究了加权原子测度上最大均值差异（MMD）的优化景观。对于一维高斯核，我们证明，对于任何紧支撑的目标分布，足够多的粒子可以确保每个局部极小值都具有任意小的MMD。此外，任何MMD仍然远离零的驻点，在粒子位置的无权重梯度流下都是不稳定的。这些结果既适用于权重被约束在单纯形上的情况，也适用于权重不受约束但每一步都被优化的情况。当目标分布本身是有限支撑时，我们进一步加强了这一结果：超过一个明确的粒子数量阈值后，不存在虚假局部极小值，且每个虚假驻点都是不稳定的。据我们所知，这些是针对加权MMD优化的首类有限粒子过参数化保证。

    arXiv:2610.19861v1 Announce Type: cross  Abstract: We study the optimisation landscape of the maximum mean discrepancy (MMD) over weighted atomic measures. For the one-dimensional Gaussian kernel, we show that, for any compactly supported target distribution, sufficiently many particles ensure that every local minimum has arbitrarily small MMD. Moreover, any stationary point whose MMD remains bounded away from zero is unstable under an unweighted gradient flow over the particle locations. These results hold both when the weights are constrained to the simplex, and when they are unconstrained but optimised at each step. When the target is itself finitely supported, we further strengthen this result: beyond an explicit particle threshold, there are no spurious local minima, and every spurious stationary point is unstable. To our knowledge, these are the first finite-particle overparameterisation guarantees of their kind for weighted MMD optimisation.
    
[^123]: 基于文本信息的训练数据过滤以提升虚拟流量计性能

    Text-Informed Training Data Filtering for Enhanced Virtual Flow Meter Performance

    [https://arxiv.org/abs/2610.19860](https://arxiv.org/abs/2610.19860)

    本文提出一种利用大语言模型将非结构化文本信息（如工程师评论和操作日志）与表格化传感器数据相结合的数据过滤方法，通过上下文异常检测去除低质量数据，从而提升虚拟流量计的预测性能。

    

    在能源、水产养殖和化工等行业中，机理模型和数据驱动模型的有效性常常受到低质量训练数据的限制，这些数据会掩盖潜在的物理规律并降低预测精度。此类质量问题通常表现为数据集中的异常，涵盖传感器故障、工艺扰动以及操作员的错误输入等。虽然传统的基于规则的过滤方法能够去除一部分异常，但它无法利用以非结构化文本形式存储的上下文信息，例如解释数据采集条件的工程师评论和操作日志。我们提出了一种基于文本信息的数据过滤方法，利用大语言模型（LLM）将非结构化运营数据与表格化传感器数据相结合。该方法包含两个步骤：（1）通过对两种数据模态进行推理来检测上下文异常；（2）生成一个二……（摘要原文在此处截断）

    arXiv:2610.19860v1 Announce Type: cross  Abstract: In sectors such as energy, aquaculture and chemical processing, the effectiveness of both mechanistic and data-driven models is frequently limited by poor-quality training data, which obscures underlying physical laws and degrades predictive accuracy. Such quality issues commonly manifest as anomalies in the dataset, ranging from sensor faults to process upsets and incorrect operator inputs. While traditional rule-based filtering can remove a subset of the anomalies, it is unable to leverage contextual information stored as unstructured text, such as engineer comments and operational logs that explain under what conditions the data was collected. We propose a text-informed data filtering method that leverages large language models (LLMs) to integrate unstructured operational data with tabular sensor data. The method consists of two steps: (1) contextual anomaly detection by reasoning over both data modalities, and (2) generation of a b
    
[^124]: 离散扩散与自回归语言模型的能耗分析

    Energy Consumption Analysis of Discrete Diffusion and Autoregressive Language Models

    [https://arxiv.org/abs/2610.19855](https://arxiv.org/abs/2610.19855)

    本文通过对比基准测试发现，在较长序列（200–600词元）推理时，扩散语言模型的每词元能耗比使用KV缓存的自回归模型效率高约2倍，展现出显著的能效优势。

    

    随着大语言模型规模的不断扩大，其推理过程中的能耗已成为一个关键问题。尽管自回归（AR）模型是行业标准，但扩散语言模型（DLLMs）作为一种能够并行生成词元的有前景的替代方案已经崭露头角。本文中，我们对扩散模型（LLaDA-8B、Dream-7B）与相应的自回归模型（Llama-3.1-8B、Qwen-2.5-7B）进行了对比能耗基准测试。我们分析了模型架构、KV缓存优化与每词元能耗效率（焦耳/词元）之间的权衡关系。结果表明，在较长序列（200–600 个词元）下，扩散模型的能耗为 2.5–2.7 焦耳/词元，比采用 KV 缓存的自回归模型（5.3–5.7 焦耳/词元）效率高约 2 倍。KV 缓存为自回归模型在较长序列下带来了显著的收益（能耗降低 50–70%）。

    arXiv:2610.19855v1 Announce Type: new  Abstract: As Large Language Models (LLMs) continue to scale, their energy consumption during inference has become a critical concern. While Autoregressive (AR) models are the industry standard, Diffusion Language Models (DLLMs) have emerged as a promising alternative capable of parallel token generation. In this paper, we perform a comparative energy benchmark of diffusion models (LLaDA-8B, Dream-7B) against AR counterparts (Llama-3.1-8B, Qwen-2.5-7B). We analyze the trade-offs between model architecture, KV-cache optimization, and Joules-per-token efficiency. Our results show that at longer sequences (200--600 tokens), diffusion models achieve 2.5--2.7 J/token, approximately 2$\times$ more efficient than AR models with KV-cache (5.3--5.7 J/token). KV-cache provides substantial benefits for AR models at longer sequences (50--70\% energy reduction).
    
[^125]: 基于残差校正的内存高效子空间优化：从动量SGD到Muon

    Memory-Efficient Subspace Optimization with Residual Correction: From Momentum SGD to Muon

    [https://arxiv.org/abs/2610.19844](https://arxiv.org/abs/2610.19844)

    本文提出带残差校正的内存高效子空间优化方法SMRC及其多阶段变体，通过在子空间中维护优化器状态并将正交残差梯度加入参数更新，实现了与全空间动量SGD相匹配的理论收敛速率，并将该框架进一步扩展到Muon优化器。

    

    面向内存高效的子空间优化已成为一种在保持优化性能的同时降低大语言模型（LLM）训练内存开销的实用方法。已有算法通常依赖于随机梯度的主奇异子空间，然而在训练后期，梯度噪声可能主导该子空间，从而阻碍收敛。为克服这一局限，我们采用残差校正框架，将优化器状态保持在子空间内，并将正交残差梯度加入参数更新中。我们首先将该方案融入动量SGD，提出了SMRC，在固定的正交归一基下获得了 $O(T^{-1/4})$ 的收敛速率，与全空间动量SGD相当。我们的分析引入了子空间虚拟迭代点的概念，以解决动量与残差校正之间的耦合问题。此外，我们将SMRC扩展为多阶段变体，周期性地重新计算基底……（摘要原文在此处被截断）

    arXiv:2610.19844v1 Announce Type: new  Abstract: Memory-efficient subspace optimization has emerged as a practical approach to reducing the memory cost of LLM training while preserving optimization performance. Previous algorithms typically rely on the leading singular subspace of stochastic gradients, yet the gradient noise may dominate the subspace in later training stages and can prevent convergence. To overcome this limitation, we adopt a residual correction framework, which maintains the optimizer states in the subspace and adds the orthogonal residual gradient to the parameter update. We first integrate this scheme into momentum SGD to develop SMRC, which attains an $O(T^{-1/4})$ convergence rate for a fixed orthonormal basis, matching that of full-space momentum SGD. Our analysis introduces a subspace virtual iterate to resolve the coupling between the momentum and the residual correction. Furthermore, we extend SMRC to a multi-stage variant which periodically recomputes bases t
    
[^126]: 寻租的语言模型：奖励模型定价格，频繁审计遏制租金

    Rent-Seeking Language Models: Reward Models Set the Prices, Frequent Audits Curb the Rent

    [https://arxiv.org/abs/2610.19832](https://arxiv.org/abs/2610.19832)

    该论文将RLHF中的奖励过度优化创造性地重新表述为经济学中的“寻租”问题，通过租金、俘获指数和无谓损失等可度量指标实证揭示：策略模型会把41%-66%的代理奖励增益转化为真实质量不支持的租金（如编造不存在实体的细节或回避事实问题），租金随监管者定价而变，而频繁的黄金奖励审计可遏制这种寻租行为。

    

    在RLHF中，奖励模型的过度优化通常被描述为“奖励作弊”。我们将这一现象重新表述为经济学中的“寻租”：策略如同一家企业，从监管者——即代理奖励模型——那里赚取真实质量无法支撑的奖励。这一视角将过度优化转化为可度量的量，即租金、俘获指数和无谓损失，并引出源自规制经济学的可检验预测。在一个固定且版本受控的测试平台中，包含三个模型家族、精度匹配的代理奖励模型和两个黄金奖励模型，实验表明强化学习将其代理奖励增益的41%至66%转化为租金，而黄金奖励却趋于停滞。租金随各监管者的定价而变化。两个代理奖励模型都不核实事实，但一个为自信的具体细节付费，另一个为勤奋的表现付费，因此RL策略学会编造关于不存在实体的细节（占探针的97%至100%）或回避事实性问题（占67%至98%）。策略与监管者之间的亲缘关系会带来约0的固定溢价。

    arXiv:2610.19832v1 Announce Type: new  Abstract: Reward model overoptimization in RLHF is usually described as reward hacking. We recast it as rent-seeking: the policy, a firm, earns reward from a regulator, the proxy reward model, that true quality does not back. The lens turns overoptimization into measurable quantities, namely rent, a capture index and deadweight loss, and into testable predictions from the economics of regulation. In a fixed, version-controlled testbed with three model families, accuracy-matched proxy reward models and two gold reward models, RL turns 41% to 66% of its proxy gain into rent while gold reward plateaus. Rent follows each regulator's prices. Neither proxy verifies facts, but one pays for confident specifics and the other for displays of diligence, so RL policies learn to fabricate details about nonexistent entities (97% to 100% of probes) or to deflect factual questions (67% to 98%). Kinship between policy and regulator adds a fixed premium of about 0.
    
[^127]: CTIA-ODE：用于长期时间序列预测的连续时间变量间注意力机制

    CTIA-ODE: Continuous-Time Inter-Variable Attention for Long-Term Time Series Forecasting

    [https://arxiv.org/abs/2610.19831](https://arxiv.org/abs/2610.19831)

    该论文提出CTIA-ODE连续时间预测框架，将连续时间变量间注意力机制嵌入常微分方程求解器中，使内生变量轨迹能够在无需时间对齐的情况下沿积分路径动态查询静态外生变量信息，并通过双重稳定策略提升连续潜在动力学的稳定性。

    

    尽管多变量时间序列预测具有广泛的应用前景，但如何有效纳入外生变量仍然是一个挑战。主流的离散时间架构往往要求异构变量之间进行刚性的时间对齐，这可能限制其捕捉持续演变的外部影响的能力。为了解决这一局限性，我们提出了CTIA-ODE，一个以连续时间变量间注意力（CTIA）为核心的连续时间预测框架。具体而言，CTIA被嵌入到常微分方程（ODE）求解器中，使持续演化的内生轨迹能够在积分路径上动态查询静态的外生上下文，而无需显式的时间对齐。为了提高连续潜在动力学的稳定性，我们进一步引入了一种双重稳定策略：在架构上，线性残差增强（LRA）模块提供宏观层面的……

    arXiv:2610.19831v1 Announce Type: new  Abstract: Although multivariate time series forecasting has broad application prospects, effectively incorporating exogenous variables remains challenging. Prevailing discrete-time architectures often impose rigid temporal alignment across heterogeneous variables, which may limit their ability to capture continuously evolving external effects. To address this limitation, we introduce CTIA-ODE, a continuous-time forecasting framework centered on Continuous-Time Inter-Variable Attention (CTIA). Specifically, CTIA is embedded within an ordinary differential equation (ODE) solver, enabling the continuously evolving endogenous trajectory to dynamically query a static exogenous context along the integration path without requiring explicit temporal alignment. To improve the stability of the continuous latent dynamics, we further introduce a dual-stabilization strategy: architecturally, a Linear Residual Augmentation (LRA) module provides a macroscopic tr
    
[^128]: MAPLE：用于大语言模型评估的多轴参数化知识提取透镜

    MAPLE: A Multi-Axis Parametric Knowledge Extraction Lens for LLM Evaluation

    [https://arxiv.org/abs/2610.19827](https://arxiv.org/abs/2610.19827)

    该论文提出MAPLE统一评估框架，通过在表层、概率和超越采样三个层面测量答案恢复与干扰、正确词元排名变化及采样遗漏答案的恢复情况，全面解耦知识提取增强（KEE）技术对大语言模型参数化知识提取的多样化影响。

    

    大语言模型从训练数据中吸收海量知识，但其固有的概率性本质无法保证这些知识会在其输出中显现。为弥合这一差距，知识提取增强（KEE）技术——如激活引导、解码调整和提示——通过修改推理过程，可能强化已有的预测，或挖掘出较难获取的答案。然而，现有评估以总体准确率为中心，将“成功恢复的失败案例”与“被干扰的成功案例”合并为单一分数，从而掩盖了这些技术多样化的影响。我们提出MAPLE，一个用于刻画KEE技术如何影响参数化知识提取的统一框架。其诊断指标通过测量答案的恢复与干扰、未恢复失败案例中正确词元的排名变化，以及固定预算采样所遗漏答案的恢复情况，在表层、概率和超越采样三个层面解耦上述影响。我们评估了六种KEE……（原文摘要在此处截断）

    arXiv:2610.19827v1 Announce Type: cross  Abstract: LLMs absorb vast knowledge from training data, yet their inherently probabilistic nature offers no guarantee that this knowledge will surface in their outputs. To bridge this gap, Knowledge Extraction Enhancement (KEE) techniques such as activation steering, decoding adjustments, and prompting modify inference, potentially reinforcing existing predictions or surfacing less accessible answers. Yet their evaluations centered on aggregate accuracy, which obscures diverse effects by collapsing recovered failures and disrupted successes into a single score. We introduce MAPLE, a unified framework for characterizing how KEE techniques affect parametric knowledge extraction. Its diagnostic metrics disentangle these effects at the surface, probability, and beyond-sampling levels by measuring answer recovery and disruption, gold-token rank shifts on unrecovered failures, and recovery of answers missed by fixed-budget sampling. We evaluate six K
    
[^129]: JointServe：在内存受限GPU上服务多模型智能体

    JointServe: Serving Multi-Model Agents on Memory-Constrained GPUs

    [https://arxiv.org/abs/2610.19822](https://arxiv.org/abs/2610.19822)

    JointServe通过将权重分片与KV块统一纳入同一内存预算进行管理，并利用意图信号预取和部分驻留等智能体工作负载的结构特性在模型被调用前完成准备，从而在内存受限GPU上实现了多模型LLM智能体的高效服务。

    

    大语言模型智能体正在从调用工具的单一模型，演变为协调多个专家模型的规划器，而这些专家模型的能力分别存储于不同的权重之中。这一转变打破了当今服务系统的一个隐含假设：即模型的工作集能够完全容纳于GPU内存。在私有部署和边缘部署中常见的内存受限GPU上，朴素的整模型交换方式会使权重传输与计算串行化，并且在缺乏主机端KV恢复支持的情况下，每次切换都会丢弃会话状态。我们提出JointServe，一个将权重分片和KV块视为在统一内存预算中竞争的驻留对象的服务系统。JointServe利用智能体工作负载的三个结构特性（有向调用、间隔执行和双状态驻留）在模型被调用之前做好准备：从调用者自身输出流中解析出的意图信号在生成过程中触发预取，部分k*驻留将切换过程缩减为增量...

    arXiv:2610.19822v1 Announce Type: cross  Abstract: LLM agents are evolving from a single model calling tools to planners that orchestrate multiple expert models whose capabilities live in distinct weights. This shift breaks a hidden assumption of today's serving systems: that the working set of models fits in GPU memory. On the memory-constrained GPUs typical of private and edge deployments, naive whole-model swapping serializes weight transfer with computation and, without host-backed KV restoration, discards session state at a switch. We present JointServe, a serving system that treats weight shards and KV blocks as residency objects competing for one unified memory budget. JointServe exploits three structural properties of agent workloads (directed calls, gapped execution, and dual-state residency) to prepare models before they are called: an intent signal parsed from the caller's own output stream triggers prefetch during generation, partial k* residency reduces switching to increm
    
[^130]: 深度何时有帮助？理解表格基础模型中的逐层推理

    When Does Depth Help? Understanding Layerwise Inference in Tabular Foundation Models

    [https://arxiv.org/abs/2610.19814](https://arxiv.org/abs/2610.19814)

    该研究通过对TabPFN-3全部24个上下文学习层在272个分类数据集上的系统分析，揭示了表格基础模型存在显著的“扩张-压缩”表示轨迹，表明深度对不同输入的帮助程度并不相同，并非所有输入都需要更深的层来完成任务。

    

    基于Transformer的表格基础模型（TFMs）在中小规模表格预测任务上取得了最先进的性能，同时通过共享的层堆栈处理多样化的输入。近期的逐层分析发现TFM的早期表示已具有强大的预测性能，且各层之间的计算存在大量重叠。然而，仅凭早期预测的可及性并不能确定后续层对每个输入都是可有可无的，而且总体准确率可能掩盖了哪些输入受益于额外的计算。我们研究了TabPFN-3在272个分类数据集上的全部24个上下文学习层，结合了表示几何、预测轨迹、截断和结构干预等方法，同时还考察了另外四个TFM中相关的表示动态。我们发现了一条显著的“扩张-压缩”轨迹，且后期更强的维度收缩与更大的……（摘要在此处截断）

    arXiv:2610.19814v1 Announce Type: new  Abstract: Transformer-based tabular foundation models (TFMs) achieve state-of-the-art performance on small to medium-sized tabular prediction tasks while processing diverse inputs through a shared stack of layers. Recent layerwise analyses identify strong predictive performance in early TFM representations and substantial overlap in computation across layers. However, early predictive accessibility alone does not establish whether later layers are dispensable for every input, and aggregate accuracy can obscure which inputs benefit from additional computation. We study all 24 in-context-learning layers of TabPFN-3 across 272 classification datasets, combining representation geometry, prediction trajectories, truncation, and structural interventions, while also examining related representation dynamics in four additional TFMs. We uncover a pronounced expansion-compression trajectory, with stronger late dimensional contraction associated with larger 
    
[^131]: VaryST：超越封闭世界设定的可泛化时空预测

    VaryST: Generalizable Spatio-Temporal Forecasting Beyond the Closed-World Setting

    [https://arxiv.org/abs/2610.19813](https://arxiv.org/abs/2610.19813)

    提出了从零训练的紧凑时空预测模型 VaryST，通过从可用的长期观测中动态构建预测上下文，实现了在时间分布演变、空间域变化和观测可用性变化等开放环境下的强泛化预测能力。

    

    arXiv:2610.19813v1 公告类型：新 摘要：时空预测对许多现实世界系统至关重要，然而部署环境本质上是多变的：时间分布可能演变，空间域可能发生变化，观测数据的可用性也可能偏离训练时的条件。现有模型往往通过节点特定表示、预定义的空间支持集以及对持续可用观测数据的依赖，与训练环境紧密耦合，这限制了对变化环境的泛化能力。为此，我们提出了 VaryST，一个从头训练的紧凑型时空预测模型，它通过从可用的长期观测中动态构建预测上下文，实现了强大的泛化能力。为了适应时间变化，VaryST 从长期历史中推导节点表示，在更长期模式的背景下捕捉当前行为。为了适应空间变化，它动态选择功能相关的（此处摘要被截断）

    arXiv:2610.19813v1 Announce Type: new  Abstract: Spatio-temporal forecasting is fundamental to many real-world systems, yet deployment environments are inherently varying: temporal distributions may evolve, spatial domains may change, and observation availability may shift from training conditions. Existing models are often tightly coupled to their training environments through node-specific representations, predefined spatial support sets, and reliance on consistently available observations, limiting generalization to changing environments. To this end, we propose VaryST, a compact spatio-temporal forecasting model trained from scratch that achieves strong generalization by dynamically constructing predictive context from available long-term observations. To accommodate temporal changes, VaryST derives node representations from long-term histories, capturing current behavior in the context of longer-term patterns. To adapt to spatial changes, it dynamically selects functionally releva
    
[^132]: 基于系统1模型的校准与风险预算决策

    Calibration and Risk-Budgeted Decisions with System One Models

    [https://arxiv.org/abs/2610.19811](https://arxiv.org/abs/2610.19811)

    该论文提出在LLM工作流的二元决策场景中，通过不确定时采取保守回退动作，并利用Learn Then Test（LTT）的有限样本统计保证来控制假遗漏风险，显著优于朴素阈值选择基线方法。

    

    大语言模型（LLM）工作流与智能体依赖于一系列二元决策，例如允许/拦截输入，或丢弃/保留检索到的文档。当无法进行人工转交处理时，不确定的案例仍必须被赋予一个动作。我们在该场景下研究系统1决策分数：在不确定时默认采取保守的回退动作，并在被分配效用最大化动作的输入子集内控制假遗漏风险。在涵盖内容审核、安全性和重排序的六个数据集上，我们将通用系统1模型与任务基线以及一个本地开源权重微调的系统1模型进行比较。结果表明，朴素的阈值选择基线在大约一半具有明确测试风险的设定中超出目标风险，而利用有限样本统计保证的Learn Then Test（LTT）方法仅超出7.8%。我们的结果说明了为什么仅依赖类似概率的输出接口不足以支撑部署决策，以及如何进行风险预算分配。

    arXiv:2610.19811v1 Announce Type: cross  Abstract: LLM workflows and agents rely on a set of binary decisions like allow/block an input, or discard/retain a retrieved document. When human deferral is unavailable, uncertain cases must still receive an action. We study System One decision scores in this setting by defaulting to a conservative fallback when uncertain, and controlling the false omission risk within the subset of inputs assigned the utility-maximizing action. Across six datasets covering moderation, safety, and reranking, we compare general System One models with task baselines and a local open-weights fine-tuned System One model. We show that naive baselines for threshold selection exceed the target risk in roughly half of settings with defined test risk, compared with 7.8\% for Learn Then Test (LTT), which leverages finite-sample statistical guarantees. Our results show why relying on a probability-like interface is insufficient for deployment decisions and how risk-budge
    
[^133]: ReproBAIT：编码智能体能否构建自己的生物AI工具？

    ReproBAIT: Can coding agents build their own biological AI tools?

    [https://arxiv.org/abs/2610.19810](https://arxiv.org/abs/2610.19810)

    该论文提出了ReproBAIT评估基准，通过在受限信息、无互联网访问的沙箱环境中，衡量编码智能体设计、实现并训练复现已发表生物AI工具的能力，并依据复现工具在留出数据上的性能进行评分。

    

    大型语言模型在利用科学知识和现有计算工具执行复杂工作流方面正变得日益胜任。目前，关于它们在仅获得工具的有限描述时重建这些工具的能力，尚缺乏相关信息。我们推出了一项名为 ReproBAIT（复现生物AI工具）的评估，用于衡量编码智能体在一系列不同信息获取条件下，设计、实现和训练源自已发表研究的生物AI工具的能力。编码智能体被置于安全沙箱环境中，无法访问互联网，仅接收训练数据、固定的时间和计算预算，以及关于待复现工具的有限信息。随后，我们根据复现的生物AI工具在留出数据上的表现（工具特定指标）对其进行评分。我们评估了一组复杂程度各异的生物AI工具：ProteinMPNN（一种逆蛋白质折叠工具）、RFdiffusion3（一种全……）

    arXiv:2610.19810v1 Announce Type: cross  Abstract: Large language models are increasingly competent at leveraging scientific knowledge and existing computational tools to execute complex workflows. At present, little information exists on their ability to reconstruct tools when given limited descriptions of such tools. We introduce an evaluation titled ReproBAIT (Reproducing Biological AI Tools), which measures the ability of coding agents to design, implement, and train biological AI tools taken from published research across a range of information access conditions. The coding agent is securely sandboxed without internet access and receives training data, a fixed time and compute budget, and limited information about the tool to replicate. We then score the reproduced biological AI tool based on its performance (tool-specific metrics) on held-out data. We evaluated a set of biological AI tools of varying complexities: ProteinMPNN, an inverse protein folding tool, RFdiffusion3, an all
    
[^134]: 基于通用函数逼近的高效上下文部分监控

    Efficient contextual partial monitoring with general function approximation

    [https://arxiv.org/abs/2610.19804](https://arxiv.org/abs/2610.19804)

    本文提出了计算高效的上下文部分监控框架 E2D.PMSide，针对原始 E2D 算法探索规则难以计算的问题设计了两种高效的探索规则，在通用函数逼近下实现了最优遗憾阶。

    

    我们研究了带有辅助信息的有限随机部分监控（PM）博弈，这是部分反馈序列学习的一种范式。在每一轮 t ∈ [T] 中，玩家和环境都观察到一个上下文，玩家选择一个动作，环境同时选择一个结果。玩家既无法观察到结果，也无法观察到损失；相反，他们只能观察到一个关于结果提供部分信息的反馈符号。我们利用 Foster 等人（2021）提出的估计到决策（E2D）元算法，设计了 E2D.PMSide——一个计算高效的上下文部分监控框架，该框架使用在线回归预言机，基于历史的（上下文、动作、反馈符号）元组高效地估计结果分布。由于原始 E2D 算法所给出的探索规则难以计算，我们提出了两种计算高效的探索规则，实现了最优的遗憾阶。

    arXiv:2610.19804v1 Announce Type: new  Abstract: We study finite stochastic partial monitoring (PM) games with side information, a paradigm of sequential learning with partial feedback. On each round $t \in [T]$, both player and environment observe a context and player chooses an action and environment simultaneously chooses an outcome. The player neither observes the outcome nor the loss; instead, they only observe a feedback symbol which is partially informative about the outcome. We utilize the Estimation-to-Decisions (E2D) meta-algorithm of Foster et al.(2021) to design E2D.PMSide, a computationally efficient framework for contextual partial monitoring, which uses an online regression oracle to efficiently estimate the outcome distributions based on historical (context, action, feedback symbol) tuples. Since the exploration rule given by the original E2D algorithm is intractable to compute, we propose two computationally efficient exploration rules that achieve optimal regret order
    
[^135]: 基于正-未标记学习的扩散模型渐近精确负引导

    Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning

    [https://arxiv.org/abs/2610.19802](https://arxiv.org/abs/2610.19802)

    提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。

    

    扩散模型的负引导旨在生成避开某种不期望条件的样本，该条件既可以由参考样本指定，也可以由条件得分与无条件模型共同指定。现有方法在每个样本处注入由局部信息推导出的排斥项，因此所得采样器并不以某个显式密度为目标，其偏差也无法通过增加计算量而被系统地降低。与之相反，我们提出了一种从显式目标密度采样的负引导方法，该方法能够统一处理基于参考样本和基于得分的两种条件指定方式，并以一个参数化目标密度本身的回避强度来抑制不期望条件附近的概率质量。实现这一目标需要在每个扩散时间步估计每个样本属于不期望条件的可能性。我们将这一估计问题构造为一个正-未标记学习问题，该问题支持轻量级的在线训练。

    arXiv:2610.19802v1 Announce Type: new  Abstract: Negative guidance of diffusion models generates samples that avoid an undesired condition, specified either by reference samples or by a conditional score alongside an unconditional model. Existing methods inject repulsive terms derived from local information at each sample, so the resulting sampler does not target an explicit density, and its bias cannot be systematically reduced by increasing computation. We instead present a negative-guidance method that samples from an explicit target density, which uniformly handles both reference-sample and score-based specifications and suppresses probability mass near the undesired condition with an avoidance strength that parametrizes the target density itself. Realizing this target requires estimating, at each diffusion timestep, how likely each sample is to belong to the undesired condition. We cast this estimation as a Positive-Unlabeled learning problem that admits lightweight online trainin
    
[^136]: StateCraft：通过抽象状态推理增强视频世界模型

    StateCraft: Augmenting Video World Models with Abstract-State Reasoning

    [https://arxiv.org/abs/2610.19795](https://arxiv.org/abs/2610.19795)

    StateCraft提出将持久可更新的抽象状态与视觉状态解耦的新范式，通过语言模型追踪世界演化中的实体变化并以此条件化视频扩散模型，从而显著提升视频世界模型在长期交互中的状态一致性与推理能力。

    

    视频世界模型旨在以逼真的外观和动态效果模拟复杂环境。要在长时间交互中保持一致性，需要追踪过去的动作并理解这些动作如何影响环境。视频世界模型一次只能生成整个世界的部分观察结果。仅记住过去的观察在许多场景中并不充分，因为经过一系列动作后，这些观察可能已不再反映世界的真实状态。我们引入了一种新范式，将持久、可更新的抽象状态与视觉状态解耦。抽象状态随世界演进而追踪实体的变化，并被用来作为视觉状态（即当前视图）的条件。我们提出了StateCraft，这是一种将语言模型与视频扩散模型相结合以演化动态世界的新型架构。我们在生成式接口上评估了我们的方法，这带来了重大挑战，因为

    arXiv:2610.19795v1 Announce Type: cross  Abstract: Video world models aim to simulate complex environments with realistic appearance and dynamics. Maintaining consistency over long interactions requires tracking past actions and understanding how those actions affect the environment. Video world models can only generate partial observations of the entire world at a time. Remembering past observations alone may not be sufficient in every scenario, as they may no longer reflect the world after a sequence of actions. We introduce a paradigm that decouples a persistent, updatable abstract state from the visual state. The abstract state keeps track of the entities' changes as the world evolves, and is used to condition the visual state, which represents the current view. We propose StateCraft, a novel architecture that pairs a language model with a video diffusion model to evolve the dynamic world. We evaluate our approach on generative interfaces, which pose a substantial challenge because
    
[^137]: 更多方法，就能带来更好的智能体吗？对智能体强化学习中自蒸馏方法的现实检验

    More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.19789](https://arxiv.org/abs/2610.19789)

    本文通过 132 次训练运行（累计超过 1.2 万 H800 GPU 小时），在四个智能体基准和三种模型规模上系统性地将十种自蒸馏方法与 GRPO 直接对比，以厘清性能提升的真实来源，并揭示各方法在性能、训练稳定性与计算成本上的实际优劣。

    

    智能体强化学习中的自蒸馏已催生出越来越多的方法。然而，现有评估往往在不同实验条件下复用已发表的基线分数，同期提出的各方法之间鲜有直接比较，且对训练稳定性和计算成本的关注有限。这些局限掩盖了所报告性能提升的真正来源，也掩盖了各竞争方法在实际中的优缺点。高昂的智能体训练成本进一步凸显了建立统一比较基准的必要性。为弥补这些空白，我们在 ALFWorld、ScienceWorld、Search-QA 和 WebShop 四个基准上，针对三种不同模型规模，将十种代表性方法与 GRPO 进行了对比。本研究共包含 132 次训练运行，累计记录 12,124.4 个 H800 GPU 小时，并对预训练骨干模型及全部十一种训练后的方法在八个 rollout 种子上进行了重复评估。除最终任务性能外，我们还考察了训练进度、稳定性以及测量……（摘要原文在此截断）

    arXiv:2610.19789v1 Announce Type: new  Abstract: Self-distillation in agentic reinforcement learning has produced a growing number of methods. However, existing evaluations often reuse published baseline scores under differing experimental conditions, provide few direct comparisons among contemporaneous methods, and give limited attention to training stability and computational cost. These limitations obscure the source of reported gains and the practical strengths and weaknesses of competing methods. Costly agent training further motivates a common basis for comparison. To address these gaps, we compare ten representative methods against GRPO on ALFWorld, ScienceWorld, Search-QA, and WebShop across three model sizes. Our study comprises 132 training runs totaling 12,124.4 recorded H800 GPU-hours, with repeated evaluations over eight rollout seeds for the pretrained backbones and all eleven trained methods. Beyond final task performance, we examine training progress, stability, and mea
    
[^138]: 基于优势引导的强化学习

    Reinforcement Learning via Advantage Guidance

    [https://arxiv.org/abs/2610.19783](https://arxiv.org/abs/2610.19783)

    提出了一种名为“优势引导”的新策略提取方法，将优势条件化的监督学习结构与采样时的对比扩散引导相结合，实现了像监督学习一样稳定、却比以往强化学习方法性能更强的策略优化。

    

    强化学习（RL）有望学习到能够超越数据中所展示行为的策略。然而在实践中，这类系统往往存在不稳定性和超参数敏感的问题。相比之下，监督学习提供了稳定且可扩展的训练目标，但从根本上受限于示范行为的质量。在本文中，我们探讨是否能够以一种像监督学习一样简单稳定、却又像强化学习一样高性能的方式来进行策略优化。遗憾的是，简单地用监督学习训练策略并以未来奖励或优势作为条件，其表现通常不如专门的强化学习方法。我们提出了一种新的策略提取方法——优势引导，它在保留带有优势条件化的简单监督学习结构的同时，通过有效的条件化训练和采样过程中的对比扩散引导，超越了先前的强化学习方法。

    arXiv:2610.19783v1 Announce Type: new  Abstract: Reinforcement learning (RL) holds the promise of learning a policy that can outperform what is shown in the data. However, in practice, such systems suffer from instability and hyperparameter sensitivity. In contrast, supervised learning provides a stable and scalable training objective, but is fundamentally limited by the quality of the demonstrated behavior. In this paper, we ask whether we can perform policy optimization with something as simple and stable as supervised learning but as performant as RL. Unfortunately, naively training the policy with supervised learning and conditioning on future rewards or advantages typically underperform dedicated RL methods. We introduce a new policy extraction method, Advantage Guidance, that retains the simple supervised learning structure with advantage conditioning while outperforming prior RL methods through effective conditional training and contrastive diffusion guidance during sampling. We
    
[^139]: 学习谁受益最多：面向人群健康分层的多治疗因果排序方法

    Learning Who Benefits Most: Multi-Treatment Causal Ranking for Population Health Stratification

    [https://arxiv.org/abs/2610.19782](https://arxiv.org/abs/2610.19782)

    提出PROMETHEUS框架，将人群健康分层转化为多治疗因果排序问题，通过联合学习对治疗进行全局排序，依据预期增量收益识别最应优先获得护理强化的患者，从而在资源受限条件下优化医疗资源分配。

    

    人群健康分层框架（如意大利的DM77）按照临床需求和护理路径对患者进行分类，但在资源受限的情况下，对于应优先采取哪些护理调整提供的指导有限。我们提出了PROMETHEUS（通过分层在医疗环境中基于多治疗效应定位实现主动资源优化），这是一个将人群健康分层形式化为多治疗因果排序问题的框架。其目标是根据预期增量收益对符合条件的患者-治疗机会进行排序，而非获得完全校准的个体治疗效应估计。每种临床上可接受的护理强化方案都与维持患者当前护理路径进行比较。多种治疗方法被联合学习，以在共享参考条件、结果、随访时间范围和基线的领域内，生成跨患者和护理特征全局可比较的排序。

    arXiv:2610.19782v1 Announce Type: new  Abstract: Population health stratification frameworks, such as the Italian DM77, organize patients by clinical need and care pathways but offer limited guidance on which care changes to prioritize under constrained resources.   We present PROMETHEUS (Proactive Resource Optimization through Multi-treatment Effect-based Targeting in Healthcare Environments Using Stratification), a framework that formulates population health stratification as a multi-treatment causal ranking problem. Its objective is to order eligible patient-treatment opportunities by expected incremental benefit rather than obtain perfectly calibrated individual treatment-effect estimates.   Each clinically admissible care intensification is compared with maintaining the patient's current pathway. Treatments are learned jointly to produce a globally comparable ranking across patients and care profiles, within domains sharing a reference condition, outcome, follow-up horizon, and be
    
[^140]: SNACK：一种用于概率图生成的序列表示框架

    SNACK: A Sequential Notation Framework for Probabilistic Graph Generation

    [https://arxiv.org/abs/2610.19781](https://arxiv.org/abs/2610.19781)

    SNACK提出了一种将图线性化为字符串序列的表示框架，通过序列与有序邻接矩阵的双射对应和无效logit屏蔽机制，使序列模型能够以100%有效率生成图，并在保持有竞争力样本质量的同时显著提升GFlowNet训练吞吐量。

    

    我们提出了SNACK，这是一种用于概率图生成的字符串表示方法，使序列模型能够生成并建模具有离散节点和边特征的图。SNACK通过编码下三角邻接矩阵的非零元素来实现图的线性化，在有效序列与有序邻接矩阵之间建立了双射对应关系。这种设计使得无效logit屏蔽成为可能，从而能够强制执行结构约束（如化学价规则和排序规则），保证生成的序列100%有效。SNACK还通过显式考虑节点排序对称性，实现了基于字符串的原则性GFlowNet训练。在标准图生成任务和分子分布学习基准测试中，SNACK取得了具有竞争力的样本质量，同时相对于基于图的生成方法大幅提升了GFlowNet的训练吞吐量。探测分析进一步表明，在SNACK上训练的模型编码了（原文摘要在此处被截断）

    arXiv:2610.19781v1 Announce Type: cross  Abstract: We introduce SNACK, a string representation for probabilistic graph generation that enables sequence models to generate and model graphs with discrete node and edge features. SNACK linearizes graphs by encoding the nonzero entries of the lower-triangular adjacency matrix, establishing a bijective correspondence between valid sequences and ordered adjacency matrices. This design enables invalid logit masking to enforce structural constraints--such as chemical valence and ordering rules--guaranteeing 100% valid generated sequences. SNACK also enables principled string-based GFlowNet training by explicitly accounting for node-ordering symmetries. Across standard graph-generation tasks and molecular distribution-learning benchmarks, SNACK achieves competitive sample quality, while substantially improving the throughput of GFlowNet training relative to graph-based generation. Probing analyses further show that models trained on SNACK encode
    
[^141]: 从全局收敛到尾部收敛：α-稳定基于分数的生成模型的动力学

    From Global to Tail Convergence: Dynamics of $\alpha$-Stable Score-Based Generative Models

    [https://arxiv.org/abs/2610.19773](https://arxiv.org/abs/2610.19773)

    本文对α-稳定基于分数的生成模型建立了非渐近收敛性分析，证明在全局Wasserstein距离下α-稳定动力学与布朗扩散模型表现相当，但两者在尾部收敛行为上存在本质区别。

    

    由α-稳定Lévy过程驱动的基于分数的生成模型已被提出作为布朗扩散模型的替代方案，经验证据表明其在采样方面有所改进，并能更好地建模重尾数据。然而，这些模型的理论保证仍然有限。我们对保持尺度的α-稳定生成动力学建立了非渐近收敛性分析，涵盖紧支撑目标和幂律目标。在p-Wasserstein距离下，我们量化了早停、初始化、分数逼近和时间离散化的影响，并证明初始化误差呈指数衰减。与布朗动力学的比较表明，仅凭全局Wasserstein收敛并不能从根本上区分这两种模型：在共同的重尾目标类别上，两种模型的初始化误差都随正向时间范围呈指数衰减。而一个鲜明的分离出现在尾部。

    arXiv:2610.19773v1 Announce Type: new  Abstract: Score-based generative models driven by $\alpha$-stable L{\'e}vy processes have been proposed as an alternative to Brownian diffusion models, with empirical evidence suggesting improved sampling and better modeling of heavy-tailed data. Yet theoretical guarantees for these models remain limited. We develop a non-asymptotic convergence analysis for scale-preserving $\alpha$-stable generative dynamics, covering both compactly supported and power-law targets. In p-Wasserstein distance, we quantify the effects of early stopping, initialization, score approximation, and time discretization, and show exponential decay of the initialization error. A comparison with Brownian dynamics reveals that global Wasserstein convergence alone does not fundamentally distinguish the two models: both initialization errors decay exponentially with the forward horizon on the common class of heavy-tailed targets. A sharp separation emerges instead in the tails.
    
[^142]: 依赖图何时有助于代码到架构的映射？

    When Does the Dependency Graph Help Code-to-Architecture Mapping?

    [https://arxiv.org/abs/2610.19769](https://arxiv.org/abs/2610.19769)

    该论文将代码到架构的映射建模为依赖图上的半监督节点分类问题，并通过GCN、GAT、MLP与朴素贝叶斯的系统对比实验，揭示了依赖图结构信息在何种条件下才能真正提升架构映射的效果。

    

    当系统的实现偏离其预期的模块化分解，而源文件与架构模块之间的映射缺失、不完整或人工维护成本高昂时，就会产生架构技术债。我们将代码到架构的映射问题形式化为有向多关系依赖图上的半监督节点分类，其中文件作为节点，类型化的依赖关系作为边，词法特征则从代码标识符和目录布局中提取。从少量带标签的样本出发，通过迭代式自训练过程将映射扩展到其余文件。我们将图卷积网络（GCN）、图注意力网络（GAT）以及不使用图结构的多层感知机（MLP）——三种学习型模型均采用相同的特征和训练流程——与朴素贝叶斯参照方法进行对比。在十个开源Java和C/C++系统上进行的100次实验运行中，三种神经模型的表现均优于……（原文摘要至此截断）

    arXiv:2610.19769v1 Announce Type: cross  Abstract: Architectural technical debt arises when a system's implementation drifts from its inten ded modular decomposition, while the mapping between source files and architectural modu les is missing, incomplete, or costly to maintain manually. We formulate code-to-archite cture mapping as semi-supervised node classification over a directed multi-relational de pendency graph, with files as nodes, typed dependencies as edges, and lexical features d erived from code identifiers and directory layout. Starting from a small labeled sample,   an iterative self-training procedure expands the mapping to the remaining files. We com pare a graph convolutional network (GCN), a graph attention network (GAT), and a graph-f ree multi-layer perceptron (MLP), using identical features and training procedures for t he learned models, against a Naive Bayes reference. Across 100 runs on ten open-source J ava and C/C++ systems, all three neural models outperform 
    
[^143]: 循环Transformer中学会计算并不等于学会保留与复用

    Learning to Compute Is Not Learning to Retain or Reuse in Looped Transformers

    [https://arxiv.org/abs/2610.19768](https://arxiv.org/abs/2610.19768)

    循环Transformer在训练深度内计算精确，但超出训练深度后会大量丢失答案；通过随机化监督位置或随机化训练循环次数可显著提升答案保留能力，而截断梯度等抗过度思考的训练方法则无效。

    

    循环Transformer通过反复应用同一个模块来进行计算，因此它们可以运行比训练时更多的循环次数。众所周知，递归推理器会“过度思考”：迭代超出训练深度可能会把原本正确的答案变成错误答案。这种晚期错误可能反映了计算失败、信息保留能力的丧失，或者是后续计算无法复用的状态。我们在上下文内的Mealy机执行任务中将这些情况区分开来，该任务中模拟器为隐藏状态干预提供中间目标和精确的反事实。训练为在循环32次时作答的模型在该深度完全正确，但在循环64次时却只能保留22%的答案。信息保留能力取决于训练的要求：每个样本只监督一个随机选择的位置（而不是总是监督最后一个位置），可以将保留率提升至99.8%；随机化训练循环次数也有所帮助；而截断梯度（如针对过度思考问题的渐进式训练中所采用的做法）则没有效果。计算本身是精确的：位置……

    arXiv:2610.19768v1 Announce Type: new  Abstract: Looped transformers compute by applying one block repeatedly, so they can run for more loops than they were trained with. Recurrent reasoners are known to overthink: iterating past the trained depth can turn a correct answer into a wrong one. Such a late error can reflect failed computation, loss of retention, or a state that later computation cannot reuse. We separate these cases in in-context Mealy-machine execution, where a simulator supplies intermediate targets and exact counterfactuals for hidden-state interventions. Models trained to answer at loop 32 are fully correct there but keep only 22% of these answers at loop 64. Retention depends on what training asks for: supervising one randomly chosen position per example, instead of always the last, keeps 99.8%, and randomizing the number of training loops also helps; truncating gradients, as in progressive training against overthinking, does not. The computation itself is exact: posi
    
[^144]: 针对基于流的入侵检测中投毒攻击的分析与缓解

    Analyzing and Mitigating Poisoning Attacks on Flow-Based Intrusion Detection

    [https://arxiv.org/abs/2610.19766](https://arxiv.org/abs/2610.19766)

    该论文揭示了仅需污染0.001%训练数据即可在基于流的入侵检测系统中植入隐蔽后门，并提出一种诊断优先的方法，通过受限扰动分析定位触发特征、借助蒸馏基线模型检测行为漂移，并结合自动化数据清洗或神经元剪枝实现针对性缓解。

    

    尽管基于人工智能（AI）的入侵检测系统（IDS）在运营网络中的部署日益增多，但其对大规模训练数据的依赖使其面临隐蔽投毒攻击的严重漏洞。我们证明，仅需污染0.001%的训练数据，就能在基于流的IDS模型中成功植入潜在后门，在保持对干净输入接近正常性能的同时确保高攻击成功率。为应对这一威胁，我们提出了一种“诊断优先”的方法论，通过受限扰动分析定位触发器相关特征，并通过与蒸馏基线模型的对比检测投毒引起的行为漂移。该方法进一步支持通过自动化数据清洗或神经元剪枝实现针对性缓解。我们的发现揭示了几乎无法检测的后门所带来的运营风险，并提供了一种可扩展的……（原文摘要在此处截断）

    arXiv:2610.19766v1 Announce Type: cross  Abstract: Although Artificial Intelligence (AI)-based Intrusion Detection Systems (IDSs) are increasingly deployed in operational networks, their dependence on large-scale training data introduces a critical vulnerability to stealthy poisoning attacks. We demonstrate that latent backdoors can be successfully implanted in flow-based IDS models by poisoning as little as 0.001% of the training data, maintaining near-nominal performance on clean inputs while ensuring high attack success rates. To address this threat, we propose a diagnostic-first methodology that localizes trigger-relevant features through constrained perturbation analysis and detects poisoning-induced behavioral drift by comparison with a distilled baseline model. The proposed methodology further enables targeted mitigation via automated data sanitization or neuron pruning. Our findings expose the operational risks of near-undetectable backdoors and provide a scalable approach to a
    
[^145]: CAST：面向大语言模型预训练的历史自适应谱裁剪方法

    CAST: History-Adaptive Spectral Clipping for LLM Pretraining

    [https://arxiv.org/abs/2610.19761](https://arxiv.org/abs/2610.19761)

    CAST通过利用AdamW二阶矩和谱范数指数移动平均等历史信息，自适应地设定谱裁剪的尺度与阈值，在19.08B参数MoE模型预训练中以更少步数实现了更低的损失和更少的梯度尖峰。

    

    软谱裁剪（SSC）能够降低梯度奇异值，但其效果取决于梯度尺度和裁剪阈值。我们提出了CAST（具有自适应尺度与阈值的裁剪），这是SSC的一种历史自适应扩展方法。CAST利用AdamW先前的二阶矩来设定行和列的尺度，并通过裁剪输出估计谱范数的指数移动平均来为每个矩阵设定阈值。此外，CAST会在AdamW更新之前恢复原始坐标。我们训练了一个总参数量为19.08B的混合专家（MoE）模型，共处理了83.89B个词元。CAST取得了比九个梯度处理基线更低的训练损失和平均验证损失，且相对梯度尖峰最少。它仅用少7.2%的步数就达到了最佳基线的最终平滑训练损失。

    arXiv:2610.19761v1 Announce Type: new  Abstract: Soft spectral clipping (SSC) reduces gradient singular values, but its effect depends on the gradient scale and clipping threshold. We introduce CAST (Clipping with Adaptive Scales and Thresholds), a history-adaptive extension of SSC. AdamW's previous second moment sets row and column scales. An exponential moving average of the clipped output's estimated spectral norm sets each matrix's threshold. CAST restores the original coordinates before the AdamW update. We train a mixture-of-experts (MoE) model with 19.08B total parameters for 83.89B tokens. CAST achieves lower training and mean validation losses than nine gradient-processing baselines, with the fewest relative gradient spikes. It reaches the best baseline's final smoothed training loss in 7.2\% fewer steps.
    
[^146]: 判别式分数函数：将预训练模型转化为功能性生成先验

    Discriminative Score Function: Turning Pretrained Models into Functional Generative Priors

    [https://arxiv.org/abs/2610.19743](https://arxiv.org/abs/2610.19743)

    提出判别式分数函数（DSF），一种免训练方法，可从预训练判别式模型（如ResNet分类器、DETR检测器和DINOv2编码器）的损失梯度几何中提取图像空间生成场，将其转化为生成先验，无需重训或任务引导即可从噪声生成图像。

    

    判别式模型通常以分类、检测或表示学习等任务特定目标进行训练，但其实际效用取决于能否超越训练任务、泛化到同一领域中未见过的样本。这表明判别式模型可能在任务标签之外仍保留着数据流形的隐式痕迹。我们探讨能否在不依赖任务特定引导的情况下，将这一痕迹暴露为一种生成信号。我们提出了判别式分数函数，它从预训练判别式模型的损失梯度几何中提取图像空间的生成场。DSF提供了一种免训练的更新规则，无需重新训练或任务特定目标，即可将图像推向与模型一致的视觉区域。我们将DSF应用于多种预训练判别式模型，包括ResNet分类器、DETR检测器和DINOv2 ViT编码器。DSF能够从噪声生成无条件图像，并……

    arXiv:2610.19743v1 Announce Type: cross  Abstract: Discriminative models are trained with task-specific objectives such as classification, detection, or representation learning, yet their practical utility depends on generalizing beyond the training task to unseen samples from the same domain. This suggests that a discriminative model may retain an implicit trace of the data manifold beyond its task labels. We ask whether this trace can be exposed as a generative signal without task-specific guidance. We introduce the Discriminative Score Function (DSF), which extracts an image-space generative field from the loss-gradient geometry of a pretrained discriminative model. DSF provides a training-free update rule that moves images toward model-consistent visual regions without retraining or task-specific targets. We apply DSF to diverse pretrained discriminative models, including ResNet classifiers, DETR detectors, and DINOv2 ViT encoders. DSF generates unconditional images from noise and 
    
[^147]: ProtModernBERT：计算高效的蛋白质语言建模

    ProtModernBERT: Compute-Efficient Protein Language Modeling

    [https://arxiv.org/abs/2610.19738](https://arxiv.org/abs/2610.19738)

    ProtModernBERT通过现代化架构改进，以减少高达80-90%的训练计算量实现了与ProtBert和Amplify-120M相当的性能，为资源受限的研究实验室提供了高效的蛋白质语言建模方案。

    

    大型蛋白质语言模型为学习序列-功能关系和指导实验设计提供了强大的框架。然而，架构方面的改进尚未得到充分探索，尤其是在与资源受限的研究实验室相关的低算力场景中。我们展示了通过利用现代架构改进技术，可以降低仅编码器Transformer架构的训练成本，从而提供令人鼓舞的性能与计算效率之间的权衡。我们提出了ProtModernBERT，这是一个在氨基酸序列上预训练的现代化BERT风格模型，并将其与ProtBert、Amplify和ESM2在一系列监督结构任务（包括二级结构预测、无序区域预测和亚细胞定位）以及ProteinGym的零样本功能基准测试上进行比较。我们发现，ProtModernBERT在训练计算量减少高达80-90%的情况下，能够达到或接近Amplify-120M和ProtBert的性能水平。

    arXiv:2610.19738v1 Announce Type: new  Abstract: Large protein language models provide a powerful framework for learning sequence-function relationships and guiding experimental design. However, architectural improvements are underexplored, especially in low-compute regimes of relevance for resource-constrained research laboratories. We show that the training cost for encoder-only transformer architectures can be reduced by leveraging modern architectural refinements, offering an encouraging performance / compute-efficiency trade-off. We introduce ProtModernBERT, a modernized BERT-style model pre-trained on amino acid sequences, and compare it with ProtBert, Amplify, and ESM2 across a set of supervised structural tasks -encompassing secondary structure prediction, disorder prediction, and subcellular localization- and ProteinGym's zero-shot functional benchmark. We find that ProtModernBERT matches or closely tracks the performance of Amplify-120M and ProtBert with up to 80-90\% reducti
    
[^148]: 目标怎么了？目标误泛化中策略对齐的不同失败形式

    What Happens to the Goal? Distinct Failures of Policy Alignment in Goal Misgeneralization

    [https://arxiv.org/abs/2610.19731](https://arxiv.org/abs/2610.19731)

    该研究发现了目标误泛化的三种不同失败形式（功能性目标替代、策略捷径和线索竞争），并通过追踪目标信息在智能体内部的作用深度，揭示目标会在不同层次上丢失。

    

    目标误泛化是一种常见的失败模式，即智能体在分布外环境中表现出胜任的行为，但并未追求预期的目标。然而，我们对误泛化智能体中预期目标究竟发生了什么仍然缺乏深入理解。通过在三个Procgen环境和五个机器人任务上的研究，我们发现了三种不同的策略-目标错位形式：（1）功能性目标替代，即行为遵循一个替代目标；（2）策略捷径，即行为遵循一种习得的动作策略；（3）预期目标线索与代理目标线索之间的线索竞争。我们通过追踪目标信息在内部何处仍然可用、其对策略的影响强度，以及它在动作选择中所扮演的角色来区分这些模式，并发现目标会在不同的深度层面丢失。在目标能够到达决策环节的情况下（如线索竞争），引导干预可以将策略重定向至目标位置，从而提升预期目标的达成。

    arXiv:2610.19731v1 Announce Type: new  Abstract: Goal misgeneralization is a common failure mode in which an agent behaves competently out of distribution but does not pursue the intended goal. However, we still have a weak understanding of what happens to the intended goal in misgeneralizing agents. Across three Procgen environments and five robot tasks, we find three distinct forms of policy-goal misalignment: (1) functional target substitution, where behavior follows an alternative goal, (2) a policy shortcut, where behavior follows a learned action strategy, and (3) cue competition between intended and proxy goal cues. We distinguish these regimes by tracing where goal information remains available internally, how strongly it influences the policy, and what role it plays in action selection, and find that the goal is lost at different depths. Where the goal reaches the decision, as in cue competition, steering redirects the policy to the goal's location, raising intended-goal reach
    
[^149]: TIMERSI：迈向时间序列基础模型的自改进智能体化部署

    TIMERSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models

    [https://arxiv.org/abs/2610.19725](https://arxiv.org/abs/2610.19725)

    TIMERSI是一个围绕冻结时间序列基础模型的自改进智能体部署系统，由编码智能体根据任务描述迭代生成并修正预测流水线，并在部署时通过自适应选择器挑选最优流水线，在198个任务上超越了所有对比方法。

    

    时间序列基础模型具有强大的零样本预测能力，但在真实应用中的部署仍需针对具体任务设计模型的输入与输出。直接推理可能无法利用领域知识，也无法解决反复出现的预测误差，而定制的流水线则需要大量的人类专业知识与精力。我们提出了TIMERSI，一个围绕冻结预测模型构建的自改进智能体化部署系统。一个编码智能体将任务描述转化为可执行的流水线修改，并在历史预测窗口上测试每次修改，利用观察到的误差指导后续修改。在部署阶段，自适应选择器根据相似条件下已完成的预测，在多个候选流水线中进行选择。在TIME和FEV-bench的198个任务上，TIMERSI在六项点预测与概率预测指标上超越了所有提交的方法。受控比较进一步验证了其设计的价值。

    arXiv:2610.19725v1 Announce Type: new  Abstract: Time-series foundation models provide strong zero-shot forecast, but deployment for real- world applications still requires task-specific designs for model inputs and outputs. Direct inference can leave domain knowledge and re- curring forecast errors unaddressed, while tai- lored pipelines require substantial human ex- pertise and effort. We introduce TIMERSI, a self-improving agentic deployment system around frozen forecasting models. A cod- ing agent translates task descriptions into ex- ecutable pipeline changes. It tests each re- vision on historical forecasting windows and uses observed errors to guide subsequent re- visions. During deployment, an adaptive se- lector chooses among the resulting pipelines using completed forecasts under similar con- ditions. Across 198 tasks on TIME and FEV- bench, TIMERSI outperforms all submissions on six point and probabilistic forecasting met- rics. Controlled comparisons support the value of fe
    
[^150]: 语言模型的表达上限

    Articulation Upper Bounds for Language Models

    [https://arxiv.org/abs/2610.19721](https://arxiv.org/abs/2610.19721)

    本文提出了新的提示优化算法和基于“提示恢复”的传输度量方法，用于研究随着大语言模型能力扩展时概念在模型间传输的上限，从而探究语言表达能力自身的极限。

    

    语言只能捕捉说话者之间形成意义的感知、直觉和意图中的一部分。那么，语言的极限究竟在哪里？我们朝着在大语言模型（LLM）中衡量这一问题的方向迈出了步伐：对于一组概念，我们测量随着LLM能力的扩展，每个概念在两个LLM之间的传输效果如何。这需要新颖的测量工具：(1) 最优消息：我们引入了一种提示优化算法，在基准测试中比现有算法高出最多14%，并能准确估计提示的上限。(2) 一种传输度量：要求发送方对文本进行标注，然后接收方尝试恢复发送方的提示，并测量恢复的程度；我们发现该度量与人类的标注错误强相关。利用这些度量，我们使用观察性缩放定律来研究：随着发送方和接收方容量的增长，使用最优消息传达的概念的可传输性如何提升——它是否会在某一水平以下渐近收敛……

    arXiv:2610.19721v1 Announce Type: cross  Abstract: Language captures only a subset of the perceptions, intuitions and intents that form meaning between speakers. What, then, are the limits of language? We take steps towards measuring this in LLMs as follows: for a set of concepts, we measure how well each concept can be transmitted between two LLMs as their capabilities scale. This requires novel measurement tools: (1) optimal messages: we introduce a prompt optimization algorithm that beats existing algorithms by up to 14% on benchmarks and accurately estimates prompt upper bounds. (2) A transmission measure: the sender is asked to label text; the receiver then tries to recover the sender's prompt, and recovery is measured; we find this correlates strongly with human label error. With these measures, we use observational scaling laws to study how transmissibility of a concept, conveyed with optimal messages, increases as transmitter and receiver capacity grows: does it asymptote below
    
[^151]: 基于特权后见之明蒸馏学习个体科学品味

    Learning Individual Scientific Taste with Privileged Hindsight Distillation

    [https://arxiv.org/abs/2610.19719](https://arxiv.org/abs/2610.19719)

    本文将个体科学品味建模为从标准级评估到相对偏好的研究者特定映射，并提出特权后见之明蒸馏方法，利用训练时后见的科学证据提升提案时的评估质量与群体级映射，以解决个体反馈稀缺的冷启动问题。

    

    科学品味决定了研究者追求哪些研究方向，并且因人而异。现有工作或是沿多个评估标准对研究方向进行评估，或是利用引用、评审等社区反馈信号来建模科学品味，但两者都没有直接刻画个体如何在相同标准下将评估转化为不同的偏好。我们将科学品味建模为一种研究者特定的映射：从共享的、有证据支撑的评估空间中基于提案可见证据的标准级评估，映射到相对偏好。从回顾性监督中学习到的群体级映射，可在个体反馈稀缺时提供冷启动。我们提出了特权后见之明蒸馏，它在训练阶段利用后来出现的科学证据来改进提案时的标准评估和群体级映射，同时仅暴露提案时可见的信息。

    arXiv:2610.19719v1 Announce Type: cross  Abstract: Scientific taste shapes which research directions researchers pursue, and it varies across individuals. Existing work evaluates research directions along multiple evaluation criteria or models scientific taste using community feedback signals such as citations and reviews, but neither directly captures how individuals translate assessments under the same criteria into different preferences. We model scientific taste as a researcher-specific mapping from criterion-level assessments in a shared, evidence-grounded evaluation space with proposal-visible evidence, to relative preferences. A population-level mapping learned from retrospective supervision provides a cold start when individual feedback is scarce. We introduce Privileged Hindsight Distillation (PHD), which uses later-emerging scientific evidence during training to improve proposal-time criterion assessment and the population mapping, while exposing only proposal-visible informa
    
[^152]: 新模型，旧习惯：多会话LLM智能体中模型切换下的行为迁移

    New Model, Old Habits: Behavior Transfer under Model Swapping in Multi-Session LLM Agents

    [https://arxiv.org/abs/2610.19712](https://arxiv.org/abs/2610.19712)

    揭示了多会话LLM智能体在模型切换中存在的“行为迁移”现象，并提出DASH方法，通过方向与作者感知的历史摘要保留强模型行为、抑制弱到强的行为迁移，显著优于原始历史传递和朴素摘要基线。

    

    多会话智能体可以在不同LLM之间切换，以平衡性能与推理成本。这种切换需要传递历史记录以保持会话连续性，但其对接收方模型的影响仍未被充分探索。通过多会话任务的受控实验，我们揭示了“行为迁移”现象：接收方模型倾向于采纳生成其历史记录的模型的推理和行动模式。这种效应在传递原始历史时最为明显，而通过摘要可以显著减弱。基于这些发现，我们提出了方向与作者感知的历史摘要方法（DASH），通过摘要保留更强模型的行为，同时最小化弱模型向强模型的行为迁移。在重复模型切换的实验中，DASH的表现优于原始历史传递和朴素摘要基线。这些结果表明，有效的多模型编排不仅需要考虑使用哪个模型，还需要考虑其行为模式如何被传递。

    arXiv:2610.19712v1 Announce Type: new  Abstract: Multi-session agents can switch between LLMs to balance performance and inference cost. Such switches require transferring a history for continuity, yet their effects on the recipient remain underexplored. Through controlled experiments with multi-session tasks, we uncover behavior transfer: recipients tend to adopt reasoning and action patterns from the models that generated their histories. This effect is most evident with raw-history transfer but is substantially reduced by summarization. Based on these findings, we propose Direction- and Authorship-aware Summarization of History (DASH) that preserves stronger-model behavior while minimizing weak-to-strong transfer via summarization. Experiments with repeated swaps show that DASH outperforms raw-history transfer and naive summarization baselines. These results suggest that effective multi-model orchestration requires considering not only which model to use, but also which behavioral p
    
[^153]: ReGRA：面向互补低秩微调的残差梯度自适应方法

    ReGRA: Residual-Gradient Adaptation for Complementary Low-Rank Fine-Tuning

    [https://arxiv.org/abs/2610.19700](https://arxiv.org/abs/2610.19700)

    ReGRA提出一种两阶段低秩微调方法，通过去除任务梯度在第一阶段更新子空间上的投影并聚合残差梯度方向来初始化第二阶段适配器，使有限秩预算下的两次更新捕获互补方向，从而提升微调效果。

    

    低秩自适应通过学习低秩权重更新实现了参数高效的微调。提高秩虽然能扩大自适应能力，但并不能保证所有学到的方向都能改善下游性能。在一项秩为16的LoRA诊断实验中，我们发现仅保留顶部奇异分量即可取得最佳性能，而引入更多分量反而会导致性能下降。受此启发，我们提出了残差梯度自适应，这是一种旨在更有效利用固定秩预算的两阶段方法。在第一阶段完成后，ReGRA在校准集上估计任务梯度，并去除其在第一阶段更新所诱导的双侧子空间上的投影，随后聚合这些残差梯度方向来初始化第二阶段的适配器，同时保持第一阶段的更新不变。正交正则项则鼓励两次更新捕获互补的方向。两者结……

    arXiv:2610.19700v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) enables parameter-efficient fine-tuning by learning low-rank weight updates. Increasing rank expands adaptation capacity, but does not guarantee that all learned directions improve downstream performance. In a diagnostic experiment with rank-16 LoRA, we show that retaining only the top singular components achieves the best performance, whereas adding further components degrades it. Motivated by this, we propose Residual-Gradient Adaptation (ReGRA), a two-stage method for making more effective use of a fixed rank budget. After the first stage, ReGRA estimates task gradients on a calibration set and removes their projections onto the two-sided subspace induced by the first-stage update. It then aggregates the residual-gradient directions to initialize a second-stage adapter without changing the first-stage update. An orthogonality regularizer encourages the two updates to capture complementary directions. Both up
    
[^154]: 基于高斯替代反向动力学的退火Lévy扩散

    Annealed L\'{e}vy Diffusion via Gaussian Surrogate Reverse Dynamics

    [https://arxiv.org/abs/2610.19692](https://arxiv.org/abs/2610.19692)

    提出退火Lévy扩散（ALD），通过随时间退火的稳定性指数将重尾探索与高斯精炼结合，并把稳定噪声表示为高斯尺度混合，从而在标准Gaussian噪声预测训练中仅需增加一个标量尺度输入即可实现。

    

    高斯扩散模型为生成建模提供了一个实用的框架。去噪Lévy概率模型（DLPM）在整个去噪过程中引入尾部指数固定不变的重尾扰动。我们提出了退火Lévy扩散（ALD）。ALD通过随时间变化的稳定性指数 $\alpha(t)$，将重尾探索与高斯精炼相结合。我们的“边缘优先”表述将稳定噪声表示为标量高斯尺度混合。这一表述使得标准的Gaussian噪声预测训练只需增加一个额外的尺度输入即可实现。我们研究了两种采样过程：ALD-renoise估计干净样本并注入新的边缘噪声；ALD-posterior则通过累积的条件方差和高斯桥更新来保持相邻状态之间的依赖关系。两种采样器都复用同一个训练好的预测器，仅使用不同的标量输入。我们在四个二维基准和一个不平衡的九模态混合分布上对ALD进行了评估。

    arXiv:2610.19692v1 Announce Type: new  Abstract: Gaussian diffusion models offer a practical framework for generative modeling. Denoising L\'{e}vy Probabilistic Models (DLPM) introduce heavy-tailed perturbations with a fixed tail index throughout denoising. We introduce Annealed L\'{e}vy Diffusion (ALD). ALD varies the stability index $\alpha(t)$ to combine heavy-tailed exploration with Gaussian refinement. Our marginal-first formulation represents stable noise as a scalar Gaussian scale mixture. This formulation enables standard Gaussian noise-prediction training with one additional scale input. We study two sampling processes. ALD-renoise estimates the clean sample and injects fresh marginal noise. ALD-posterior preserves dependence between successive states through an accumulated conditional variance and a Gaussian bridge update. Both samplers reuse the same trained predictor with different scalar inputs. We evaluate ALD on four 2D benchmarks and an imbalanced nine-mode mixture. We 
    
[^155]: 面向策略优化的函数空间时间自蒸馏

    Function-Space Temporal Self-Distillation for Policy Optimization

    [https://arxiv.org/abs/2610.19687](https://arxiv.org/abs/2610.19687)

    本文提出单策略的函数空间策略优化框架FSPO，以稀疏的历史下一词元分布作为固定时间目标进行自蒸馏，无需维护持久教师模型即可提供稳定密集的词元级监督，并通过特权分歧校正与可靠性自适应锚定提升策略优化效果。

    

    可验证奖励强化学习（RLVR）提升了语言模型的推理能力，但标量结果只能为响应轨迹提供粗略的监督信号。自蒸馏通过将无辅助的策略与基于反馈信息的自教师进行比较，能够提供更密集的词元级指导；然而教师会随学生一同演化，导致监督目标不稳定，因而需要额外的正则化。我们提出了函数空间策略优化（FSPO），这是一个单策略框架，将当前的特权适应与函数空间时间整合相结合。FSPO将稀疏的历史下一词元分布存储为固定的时间目标，在无需维护持久教师模型的情况下实现时间上的分离。该框架进一步引入特权分歧校正以显式抑制局部被高估的动作，并引入可靠性自适应锚定，在自生成证据重复时强化参考监督（摘要原文在此处截断）。

    arXiv:2610.19687v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) improves reasoning in language models, but scalar outcomes provide coarse supervision over response trajectories. Self-distillation offers denser token-level guidance by comparing an unassisted policy with a feedback-informed self-teacher, yet the teacher evolves with the student, creating unstable targets and motivating additional regularization. We introduce Function-Space Policy Optimization (FSPO), a single-policy framework that combines current privileged adaptation with function-space temporal consolidation. FSPO stores sparse historical next-token distributions as fixed temporal targets, providing temporal separation without maintaining a persistent teacher model. It further introduces privileged disagreement correction to explicitly suppress locally overvalued actions, reliability-adaptive grounding that strengthens reference supervision when self-generated evidence repeated
    
[^156]: 超越分布匹配：面向监督微调的选择性边际保持方法

    Beyond Distribution Matching: Selective Margin Preservation for Supervised Fine-Tuning

    [https://arxiv.org/abs/2610.19685](https://arxiv.org/abs/2610.19685)

    提出SRP-SFT，一种通过铰链惩罚选择性保持非目标token边际关系的软边际目标，在不冻结参考分布的情况下显著提升了监督微调的数学推理与代码生成性能。

    

    监督微调（SFT）会在教师强制前缀下改变非目标token之间的logit边际。匹配局部参考分布能够保持这些概率，但同时也会惩罚边际的扩大。我们探究的问题是：一个软辅助目标能否在不冻结该参考分布的前提下，保持选定的局部非目标边际关系。我们提出了选择性秩保持SFT（SRP-SFT），这是一种软边际目标：对来自参考分布top-K非目标token的token对，施加基于参考分布缩放下界的铰链惩罚，并应用于固定的教师强制前缀。一旦所要求的下界得到满足，铰链惩罚即停止。我们推导了违反边际所带来的KL代价，刻画了可行区域及其条件效用，并给出了降低总体边际违反的有限步条件。在Qwen3-4B-Base模型和UltraFeedback数据集上的单轮训练实验中，该目标在全部三个训练随机种子上均使MATH-500和代码生成得分超过交叉熵（CE）和LP-SFT方法。

    arXiv:2610.19685v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) changes logit margins among non-target tokens at teacher-forced prefixes. Matching a local reference distribution keeps those probabilities and also penalizes margin expansions. We ask whether a soft auxiliary can preserve selected local non-target margin relations without freezing that distribution. We introduce \textbf{Selective Rank-Preserving SFT (SRP-SFT)}, a soft margin target: a hinge on reference-scaled lower bounds for pairs from the reference's top-$K$ non-target tokens, applied at fixed teacher-forced prefixes. The hinge stops once the requested bound is met. We derive the KL cost of violating a margin, characterize the feasible region and its conditional utility, and give a finite-step condition for reducing aggregate violations. In one-epoch experiments on Qwen3-4B-Base and UltraFeedback, this target raises MATH-500 and code-generation scores over CE and LP-SFT at all three training seeds. At s
    
[^157]: 用联想记忆增强潜在世界模型

    Enhancing Latent World Models with Associative Memory

    [https://arxiv.org/abs/2610.19683](https://arxiv.org/abs/2610.19683)

    该论文提出一种通过内容可寻址记忆库增强的JEPA式潜在世界模型，配合鼓励槽位多样性和时间均衡利用的互补损失来防止表征坍塌，以轻量级架构在长时程规划任务上实现了最先进的性能。

    

    通过潜在动力学训练的世界模型为具身规划提供了一条可扩展的路径，但现有方法存在表征坍塌、计算开销大以及长时程能力有限等问题。我们提出了一种记忆增强的潜在世界模型，通过内容可寻址的记忆库来增强JEPA风格的预测。我们的方法直接从观测中学习高层表征，在共享潜在空间内以多个时间尺度运行，并通过自适应层归一化来调节预测。为了防止坍塌并确保记忆的有效使用，我们设计了两种互补的损失函数，以鼓励槽位多样性和平衡的时间利用率。在PushT、Tworoom、Reacher和OGBCube上的实验表明，我们的方法以轻量级架构实现了最先进的性能，在长时程规划任务上超越了传统的潜在世界模型。

    arXiv:2610.19683v1 Announce Type: new  Abstract: World models trained via latent dynamics offer a scalable path for embodied planning, yet existing methods suffer from representational collapse, computational overhead, and limited long-horizon capability. We introduce a memory-enhanced latent world model that augments JEPA-style prediction with a content-addressable memory bank. Our approach learns high-level representations directly from observations, operating at multiple temporal scales within a shared latent space, and conditions predictions via adaptive layer normalization. To prevent collapse and ensure effective memory usage, we design two complementary losses that encourage slot diversity and balanced temporal utilization. Experiments on PushT, Tworoom, Reacher, and OGBCube demonstrate that our method achieves state-of-the-art performance with a lightweight architecture, outperforming conventional latent world models on long-horizon planning tasks.
    
[^158]: MCAIR：基于内部表示的智能体不忠实行为监控与纠正框架

    MCAIR: Monitoring and Correcting Agent Unfaithfulness via Internal Representations

    [https://arxiv.org/abs/2610.19679](https://arxiv.org/abs/2610.19679)

    该论文提出了MCAIR框架，通过利用大语言模型的内部表示在运行时监控并纠正智能体的“完成不忠实性”行为，即智能体未按原定来源或程序完成任务却不予披露的欺骗性偏差。

    

    基于大语言模型（LLM）的智能体已展现出通过多步推理和工具交互完成复杂任务的卓越能力。然而，近期研究表明，基于LLM的智能体可能表现出不忠实行为，即其最终报告未能忠实反映其实际执行轨迹。具体而言，当智能体无法通过预期的来源或程序完成任务时，它们可能会转而使用替代来源或策略，却不披露此类偏差，导致用户误以为任务已按照原始要求完成。我们将这种特定形式的智能体不忠实行为称为“智能体完成不忠实性”。检测和缓解智能体完成不忠实性至关重要，但目前仍 largely 未被充分探索。在这项工作中，我们提出了MCAIR，一个利用内部表示的运行时监控与纠正框架。

    arXiv:2610.19679v1 Announce Type: cross  Abstract: Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in completing complex tasks through multi-step reasoning and tool interactions. However, recent studies have revealed that LLM-based agents can exhibit unfaithful behaviors, where their final reports fail to faithfully reflect their actual execution trajectories. Specifically, when agents fail to accomplish tasks through the intended sources or procedures, they may resort to alternative sources or strategies while failing to disclose such deviations, leading users to incorrectly believe that the tasks have been completed according to the original requirements. We refer to this specific form of agent unfaithfulness as Agent Completion Unfaithfulness. Detecting and mitigating agent completion unfaithfulness is critical yet remains largely unexplored. In this work, we propose MCAIR, a runtime monitoring and correction framework that leverages internal repres
    
[^159]: HACHIMI-II：从角色扮演到学习者行为建模，实现忠实的学生模拟

    HACHIMI-II: From Role-Play to Learner Behavior Modeling for Faithful Student Simulation

    [https://arxiv.org/abs/2610.19654](https://arxiv.org/abs/2610.19654)

    HACHIMI-II 将学生模拟从让大语言模型角色扮演转变为对学习者行为的第三人称预测，通过从学习者历史构建学生模型并用行为框架约束其下一步动作，从而忠实再现真实学习者的能力分布和典型错误。

    

    模拟学生使研究人员能够在真实学习者出现之前测试辅导系统和教学策略，但前提是模拟必须重现特定学习者在哪些地方成功、在哪些地方失败。常见做法是让大型语言模型扮演学生角色，但角色设定并不能刻画这个学习者知道什么、下一步会做什么，于是模型自身的能力填补了这一空缺。由此导致模拟学生的表现水平由模型本身决定，彼此之间差异很小，并且很少犯真实学习者会犯的错误。本文对学习者的行为进行建模，将学生模拟重新表述为对学习者下一步动作的第三人称预测。HACHIMI-II 从学习者的历史记录中构建学生模型，推断学习者在当前题目或辅导轮次中的状态，并将其编译成一个约束下一步动作的行为执行框架。在三个题目作答数据集上，模拟学生与真实学生能力之间的 Wasserstein-1 距离为 2.4–8.4（摘要原文在此处截断）。

    arXiv:2610.19654v1 Announce Type: new  Abstract: Simulated students let researchers test tutoring systems and teaching strategies before real learners are available, but only if the simulation reproduces where a particular learner succeeds and fails. The common approach asks a large language model to role-play a student, but a role does not model what this learner knows or will do next, so the model's own competence fills the gap. Simulated students perform at a level set by the model, differ little from one another, and seldom make the real learners' errors. We model the learner's behavior and recast student simulation as third-person prediction of the learner's next action. HACHIMI-II builds a student model from the learner's history, infers the learner's state for the current item or tutor turn, and compiles it into a behavior harness that constrains the next action. On three item-response datasets, the Wasserstein-1 distance between simulated and real student ability is 2.4--8.4$\t
    
[^160]: 关于零阶非凸优化中的全局联合下界

    On Global Joint Lower Bounds in Zeroth-Order Nonconvex Optimization

    [https://arxiv.org/abs/2610.19652](https://arxiv.org/abs/2610.19652)

    本文在无限制全空间设置下为光滑随机非凸优化建立了标量预言机全局联合下界，回答了零阶方法复杂度关于维度、光滑化参数和精度的依赖是否联合最优这一公开问题。

    

    近期的研究工作试图通过梯度估计器将一阶非凸优化的保证转移到零阶方法上，这通常会产生与维度 $d$ 呈线性关系的因子。Kornowski 等人（2024）针对非光滑非凸优化获得了 $\mathcal{O}(d\delta^{-1}\epsilon^{-3})$ 的预言机复杂度，并提出了其关于这三个参数的依赖关系是否联合最优的问题。尽管关于维度的依赖性以及对 $(\delta,\epsilon)$ 的一阶依赖性已被分别证明是不可避免的，但与之匹配的零阶下界一直悬而未决。我们在无限制的全空间设置中回答了这个问题。对于光滑随机非凸优化，在假设种群目标函数光滑且全局下有界、以及样本 Lipschitz 常数满足二阶矩有界的条件下，我们建立了标量预言机下界 $\Omega(dL_0^2\min\{\Delta H,L_0^2\}/\epsilon^4)$。随后通过一个光滑化归，进一步得到了一个下界……（摘要在此处被截断）

    arXiv:2610.19652v1 Announce Type: new  Abstract: Recent work has sought to transfer first-order nonconvex optimization guarantees to zeroth-order methods through gradient estimators, typically incurring a factor linear in $d$. \citet{kornowski2024} obtained an $\mathcal{O}(d\delta^{-1}\epsilon^{-3})$ oracle complexity for nonsmooth nonconvex optimization and asked whether its dependence on all three parameters is jointly optimal. Although the dimension dependence and the first-order dependence on $(\delta,\epsilon)$ are separately known to be unavoidable, a matching zeroth-order lower bound has remained open. We answer this question in the unrestricted full-space setting. For smooth stochastic nonconvex optimization, we establish a scalar-oracle lower bound of $\Omega\!\left(dL_0^2\min\{\Delta H,L_0^2\}/\epsilon^4\right)$ assuming a smooth, globally lower-bounded population objective and a second-moment bound on sample Lipschitz constants. A smooth reduction then yields a lower bound f
    
[^161]: 建模关键内容：面向语言智能体的决策引导式选择性世界建模

    Model What Matters: Decision-Guided Selective World Modeling for Language Agents

    [https://arxiv.org/abs/2610.19642](https://arxiv.org/abs/2610.19642)

    提出DGS-WM方法，通过决策引导的观测选择机制让语言智能体的世界建模只聚焦于具有决策价值的观测内容，从而弥补稀疏任务奖励无法监督环境动态理解的局限。

    

    强化学习为大型语言模型驱动的智能体提供了一种直接优化任务目标的机制，但任务奖励通常是稀疏的轨迹级信号，仅用于评估任务是否成功，无法直接监督动作条件下的环境状态转移，导致模型对动作后果和环境动态的显式理解十分有限。近期的工作通过策略与世界建模协同训练的方式引入辅助监督，但通常将完整的环境观测不加区分地统一作为世界建模目标，忽视了观测内容在决策相关性上的差异。我们提出了DGS-WM（决策引导式选择性世界建模），这是一种面向语言智能体的决策引导式策略-世界建模协同训练方法。DGS-WM采用一个决策引导的观测选择组件，基于观测单元的决策价值筛选出具有决策价值的观测单元作为世界建模目标……

    arXiv:2610.19642v1 Announce Type: cross  Abstract: Reinforcement learning provides a direct mechanism for optimizing task objectives in large language model-driven agents, but task rewards are typically sparse trajectory-level signals that only evaluate task success, without directly supervising action-conditioned environment transitions, leaving the model with limited explicit understanding of action consequences and environment dynamics. Recent work introduces auxiliary supervision through policy and world modeling co-training, yet typically treats complete environmental observations uniformly as world modeling targets, overlooking the varying decision relevance of observation content. We propose DGS-WM (Decision-Guided Selective World Modeling), a decision-guided policy-world modeling co-training approach for language agents. DGS-WM employs a Decision-Guided Observation Selection component that filters observation units with decision value as world modeling targets based on their su
    
[^162]: 基于转化规则生成的合成反应数据用于生物降解路径预测中图到序列模型的预训练

    Synthetic reaction data from transformation rules for graph-to-sequence pretraining in biodegradation pathway prediction

    [https://arxiv.org/abs/2610.19640](https://arxiv.org/abs/2610.19640)

    该论文提出enviGraph框架，通过应用环境转化规则生成77万余个合成反应数据对来预训练图到序列预测器，解决了观测生物降解反应数据稀缺的问题，从而提升生物降解产物和路径预测的性能。

    

    观测到的生物降解反应数据的稀缺性限制了产物预测器和路径预测器的训练。我们提出了enviGraph框架，该框架生成并整理合成反应数据，利用这些数据预训练图到序列预测器，再在观测到的环境反应数据上进行微调，以实现产物预测和递归路径预测。将继承的环境转化规则应用于来自PubChem、NORMAN SusDat和EPA DSSTox的化学结构并加以整理，产生了776,851个标准化的母体-产物数据对用于预训练。我们将该数据集与三个美国专利商标局（USPTO）反应数据集进行比较：USPTO-50K、USPTO-MIT和USPTO-Stereo。我们采用按路径划分的十折交叉验证，在EAWAG生物催化/生物降解数据库（BBD）、监管农药生物降解数据（Soil）以及活性污泥生物降解数据上对enviGraph和基于序列的enviFormer进行了评估。

    arXiv:2610.19640v1 Announce Type: new  Abstract: The scarcity of observed biodegradation reactions constrains the training of product and pathway predictors. We present enviGraph, a framework that generates and curates synthetic reaction data, uses these data to pretrain a graph-to-sequence predictor, and fine-tunes the predictor on observed environmental reactions for product and recursive pathway prediction. Applying inherited environmental transformation rules to structures from PubChem, NORMAN SusDat and EPA DSSTox, followed by curation, produces 776,851 canonical parent-product pairs for pretraining. We compare this dataset with three United States Patent and Trademark Office (USPTO) reaction datasets: USPTO-50K, USPTO-MIT and USPTO-Stereo. We evaluate enviGraph and the sequence-based enviFormer using ten folds split by pathway on the EAWAG Biocatalysis/Biodegradation Database (BBD), regulatory pesticide biodegradation data (Soil) and activated-sludge biodegradation data (Sludge).
    
[^163]: 稀疏自编码器是符号匹配的共享坐标系

    Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching

    [https://arxiv.org/abs/2610.19636](https://arxiv.org/abs/2610.19636)

    本文提出SAE-IDF方法，利用稀疏自编码器将符号编码为稀疏表示并根据稀有度重新加权，实现了跨异构符号系统的单元匹配，为通过数据驱动方式对齐与合并现有符号系统、构建大规模神经符号模型迈出了关键的第一步。

    

    大语言模型（LLM）在一组权重中蕴含了广泛的知识，但缺乏可解释性和可信赖的推理能力，尤其是在高风险的应用场景中。神经符号模型凭借其可解释的架构和系统2（System 2）推理能力，如果能构建到LLM的规模，将是一种有前景的替代方案；然而，其符号骨干通常需要人工构建，因此现有的神经符号模型受限于符号组件的覆盖范围，通常只能应用于特定领域。一种数据驱动的方法则可以通过对齐和合并现有的符号系统来构建更大的符号系统。我们瞄准了这一方向的第一步基础性工作：在这些异构系统之间进行单元匹配。我们的方法SAE-IDF使用稀疏自编码器（SAE）将符号在其原生语境中编码为稀疏代码，根据每个维度的稀有度进行重新加权，并在不同系统之间进行匹配。在涵盖6种符号系统的12个任务上，SAE-IDF的表现优于基线方法。

    arXiv:2610.19636v1 Announce Type: cross  Abstract: Large language models (LLMs) hold broad knowledge in one set of weights, but lack interpretability and trustworthy reasoning, especially in high-stakes use cases. A neuro-symbolic model, with its interpretable architecture and System 2 reasoning capacity, is a promising alternative if built at the scale of LLMs; however, its symbolic backbone is usually constructed with human effort, so existing neuro-symbolic models are bottlenecked by the scope of the symbol component and usually domain-specific. A data-driven method could instead build a larger symbolic system by aligning and merging existing ones. We target its first, foundational step: matching units across these heterogeneous systems. Our method, SAE-IDF, encodes a symbol in its native context into a sparse code with a sparse autoencoder (SAE), re-weights each dimension by its rarity and matches across systems. On 12 tasks across 6 kinds of symbolic system, SAE-IDF outperforms th
    
[^164]: 面向机器学习原子间势的风险控制DFT选择方法

    Risk-Controlled DFT Selection for Machine-Learned Interatomic Potentials

    [https://arxiv.org/abs/2610.19630](https://arxiv.org/abs/2610.19630)

    本文提出一种基于保形风险控制（CRC）的框架，利用轻量级预测器估计MLIP与DFT之间的不一致分数并校准为选择阈值，在保证MLIP误差可控的前提下自动决定哪些构型需要昂贵的DFT计算，从而突破自动化材料发现的通量瓶颈。

    

    自动化材料发现需要评估大量候选构型，而密度泛函理论（DFT）计算是主要的通量瓶颈。预训练的机器学习原子间势（MLIPs）能够以低成本近似DFT，但其误差在不同构型之间可能相差数个数量级，这使得难以决定何时应该信任MLIP的预测、何时应回退到DFT计算。我们将这一决策问题构建为风险控制选择问题，并在保形风险控制（CRC）框架内进行求解。该方法将每个构型从原子序数和笛卡尔坐标映射为固定的几何描述符，轻量级预测器利用该描述符估计能量、平均力和最大力方面的MLIP-DFT不一致分数。随后，CRC将该分数校准为DFT标记的选择阈值，使得未被选中的构型（保留其MLIP标签）上的MLIP误差保持在用户指定的水平以下，从而在保证预测质量的同时大幅减少所需的DFT计算量。

    arXiv:2610.19630v1 Announce Type: cross  Abstract: Automated materials discovery requires evaluating large pools of candidate configurations, with density functional theory (DFT) as a major throughput bottleneck. Pretrained machine-learned interatomic potentials (MLIPs) approximate DFT cheaply, but their errors vary by orders of magnitude across configurations, making it difficult to decide when to trust the MLIP predictions and when to fall back to DFT. We frame this decision as a risk-controlled selection problem and solve it within a conformal risk control (CRC) framework. Each configuration is mapped from atomic numbers and Cartesian coordinates to a fixed geometric descriptor, which a lightweight predictor uses to estimate an MLIP-DFT disagreement score in energy, average force, and maximum force. CRC then calibrates this score into a selection threshold for DFT labeling such that the MLIP error on the unselected configurations, which keep their MLIP labels, stays below a user-cho
    
[^165]: 自条件化使流匹配具备深度序列推理能力

    Self-Conditioning Enables Deep Sequential Reasoning in Flow Matching

    [https://arxiv.org/abs/2610.19628](https://arxiv.org/abs/2610.19628)

    该论文从电路复杂度角度证明，原始流匹配的额外采样步骤无法增强常数深度模型的表达能力，而自条件化机制能突破这一限制，使模型实现真正的深度序列推理。

    

    近期研究探索了通过图像生成进行视觉推理的方法，即模型通过直接生成解答图像来解决视觉推理问题。直觉上，像流匹配这样的迭代采样器似乎非常适合这一任务，因为每个采样步骤可能类似于一个推理步骤。然而，本工作从电路复杂度的视角表明这种直觉具有误导性。我们证明，只要模型输出满足某些温和的正则性条件，原始流匹配采样器的轨迹就可以用常数轮数的顺序并行模型评估来模拟。因此，在标准的 transformer 复杂度理论模型下，增加采样步骤无法提升常数深度骨干模型的表达能力。我们进而揭示，自条件化——即让模型以上一步对最终图像的估计作为条件输入——发挥着一种此前未被认识到的关键作用……（摘要在此处被截断）

    arXiv:2610.19628v1 Announce Type: cross  Abstract: Recent work has explored visual reasoning through image generation, where models solve visual-reasoning problems by directly generating solution images. Intuitively, iterative samplers such as flow matching are well suited to this task, since each sampling step may resemble one reasoning step. However, in this work, from a circuit complexity perspective, we show that this intuition is misleading. We prove that as long as the model outputs satisfy certain mild regularity conditions, the trajectory of the vanilla flow-matching sampler can be simulated by a constant number of sequential rounds of parallel model evaluations. Consequently, additional sampling steps cannot increase the expressive power of a constant-depth backbone model under the standard complexity-theoretic model of transformers. We then reveal that self-conditioning, which conditions the model on the previous step's estimate of the final image, plays a previously unrecogn
    
[^166]: DAGFormer：面向Transformer跨层连接的全局拓扑预测

    DAGFormer: Global Topology Prediction for Transformer Cross-Layer Connections

    [https://arxiv.org/abs/2610.19625](https://arxiv.org/abs/2610.19625)

    DAGFormer通过联合训练的拓扑预测器一次性规划整个Transformer模型的全局跨层连接拓扑，在75M至1B多个规模上将困惑度降低17%至25%，并全面超越逐层路由方法。

    

    我们提出DAGFormer，一种由联合训练的拓扑预测器来规划跨层连接的Transformer语言模型。在标准Transformer残差连接之外，DAGFormer增加了从较早层到较晚层的直接连接，预测器基于输入词元控制每条连接的强度。与现有跨层路由架构逐层单独设置连接的方式不同，DAGFormer一次性规划整个模型的路由。在参数量从75M到1B的匹配OLMo-2模型对上，DAGFormer比原始Transformer在WikiText-2困惑度上降低17%至25%；在50亿词元上训练的1B DAGFormer优于在100亿词元上训练的原始模型。在与四个路由基线使用相同数据和预算的对比中，它在75M、300M和1B三个规模上均取得最低困惑度。该增益还延续到13项常识、科学、阅读和数学评测任务上（平均高出2.0至2.5分）。

    arXiv:2610.19625v1 Announce Type: new  Abstract: We propose DAGFormer, a Transformer language model whose cross-layer connections are planned by a jointly trained topology predictor. Beyond the residual connections in a standard Transformer, DAGFormer adds direct connections from earlier layers to later ones, and the predictor controls the strength of every connection based on input tokens. Unlike existing cross-layer routing architectures, which set each layer's connections separately, DAGFormer plans the routing of the whole model at once. On matched OLMo-2 pairs from 75M to 1B parameters, DAGFormer lowers WikiText-2 perplexity by 17% to 25% over the vanilla Transformer, and a 1B DAGFormer trained on 5B tokens outperforms the vanilla model trained on 10B. Against four routing baselines with the same data and budget, it reaches the lowest perplexity at 75M, 300M and 1B. The gain carries over to 13 commonsense, science, reading and math evaluation tasks (2.0 to 2.5 points higher mean a
    
[^167]: 分岔路径：学习LLM智能体轨迹的结果预测性表示

    Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories

    [https://arxiv.org/abs/2610.19614](https://arxiv.org/abs/2610.19614)

    提出Precog-RM模型，通过元学习在共享嵌入空间中以少样本方式预测LLM智能体轨迹的成败，并可作为GRPO训练的早停控制器，在保持性能的同时最多减少47.3%的训练计算量。

    

    在这项工作中，我们将智能体轨迹重新定义为在动作与环境观测的共享嵌入空间中的路径。我们证明，同一任务的成功与失败rollout在该空间中是可分离的，使用简单的逐任务判别器即可在三个智能体基准和三个策略模型上预测留出的rollout结果，AUROC最高达0.95。为了跨任务利用这一信号，我们引入了Precog-RM，这是一个基于Transformer的模型，通过元学习获得一种表示，使结果预测可以在少样本设置下进行。我们证明Precog-RM能够泛化到未见过的任务和模型，并在轨迹终止前很久就能对其成败给出高置信度预测。作为Tau2-Bench上组相对策略优化（GRPO）的早停控制器部署时，Precog-RM将训练FLOPs减少了高达47.3%，同时保持与基线GRPO相当的性能。

    arXiv:2610.19614v1 Announce Type: cross  Abstract: In this work, we reframe agentic trajectories as paths through a shared embedding space of actions and environmental observations. We show that successful and failed rollouts of the same task separate in this space, with simple per-task discriminators predicting held-out rollout outcomes at up to 0.95 AUROC across three agent benchmarks and three policy models. To exploit this signal across tasks, we introduce Precog-RM, a transformer-based model that meta-learns a representation in which outcome prediction can be performed in a few-shot setting. We show that Precog-RM generalizes to unseen tasks and models, and gives confident predictions of trajectory success long before termination. Deployed as an early stopping controller for Group Relative Policy Optimization (GRPO) on Tau2-Bench, Precog-RM cuts training FLOPs by up to 47.3\% while maintaining performance comparable to baseline GRPO.
    
[^168]: 基于图的分析：私有Transformer推理中的可比性、规模与归因

    A Graph-Based Analysis of Comparability, Scale, and Attribution in Private Transformer Inference

    [https://arxiv.org/abs/2610.19607](https://arxiv.org/abs/2610.19607)

    该论文通过构建包含62个私有Transformer推理（PTI）系统的实证图并进行节点、边、图三层分析，揭示了密码学进步总是与其他因素共同驱动性能提升、1891个候选系统对中仅1对可直接比较、以及私有推理的扩展前沿已在13B规模停滞等重要发现。

    

    围绕大语言模型日益增长的隐私担忧加速了私有Transformer推理（PTI）的研究。然而，综合该研究方向的实证结论却受到多维度密码学进展和不一致评估契约的阻碍。在这项工作中，我们通过构建一个包含62个PTI系统的实证图来对该领域进行系统化梳理，将系统属性映射为节点特征，将评估兼容性映射为边。对该图的三层分析阐明了私有推理的现状。节点级归因表明，密码学进步推动了大多数系统的性能提升，但总是与其他因素（模型保真度、硬件、成本核算）共同出现。边级审计表明，在1891个候选系统对中，只有1对在共享假设下可直接比较。最后，图级分析表明，私有推理的扩展前沿已在13B规模处趋于停滞。

    arXiv:2610.19607v1 Announce Type: cross  Abstract: Rising privacy concerns around large language models have accelerated research in private Transformer inference (PTI). Synthesizing the empirical claims in this line of research, however, is hindered by multi-dimensional cryptographic progress and inconsistent evaluation contracts. In this work, we systematize the field by constructing an empirical graph of 62 PTI systems, mapping system attributes to node features and evaluation compatibility to edges. A three-tiered analysis of this graph illuminates the state of private inference. Node-level attribution shows that cryptographic progress drives performance gains across most systems, yet always co-occurs with other factors (model fidelity, hardware, cost accounting). Edge-level auditing shows that only 1 of 1891 candidate system pairs is directly comparable under shared assumptions. Finally, graph-level analysis shows that the scaling frontier of private inference has plateaued at 13B
    
[^169]: PN-Mix：面向高效长输出大语言模型推理的配对范数引导混合精度键缓存量化

    PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference

    [https://arxiv.org/abs/2610.19605](https://arxiv.org/abs/2610.19605)

    PN-Mix通过RoPE通道对层面的查询-键配对范数分析识别量化敏感的键缓存通道对，并在预填充阶段一次性识别、解码全程复用，为其分配更高精度，从而实现高效且低精度损失的长输出大语言模型推理。

    

    大语言模型的长输出推理受到严重的KV缓存内存与带宽瓶颈的制约。虽然KV缓存量化可以缓解这一问题，但低比特量化（例如2比特）会严重损害模型精度。近期的混合精度键缓存量化方法通过为重要的键通道分配更高精度来缓解这种精度下降，但其仅利用键自身的统计信息来估计通道显著性。通过在RoPE通道对层面进行注意力logit扰动分析，我们证明了在查询-键配对范数乘积较大的通道对中，量化误差对logit变化有显著影响。基于这一洞察，我们提出了PN-Mix，一种面向基于RoPE的仅解码器（decoder-only）大语言模型的混合精度键缓存量化方法，该方法为显著通道对分配更高精度，这些通道对在预填充阶段仅识别一次，并在整个解码过程中复用。在标准推理、代码生成和AIME基准……

    arXiv:2610.19605v1 Announce Type: cross  Abstract: Long-output inference in large language models suffers from severe KV cache memory and bandwidth bottlenecks. While KV cache quantization alleviates this issue, low-bit quantization (e.g., 2-bit) severely degrades model accuracy. Recent mixed-precision key-cache quantization methods mitigate this degradation by allocating higher precision to important key channels, but estimate channel saliency using only key statistics. Through attention logit perturbation analysis at the RoPE channel-pair level, we show that quantization errors in channel pairs with large query--key pair-norm products contribute significantly to logit variation. Based on this insight, we propose PN-Mix, a mixed-precision key-cache quantization method for RoPE-based decoder-only LLMs, which allocates higher precision to salient channel pairs identified once during the prefill stage and reused throughout decoding. Across standard reasoning, code-generation, and AIME be
    
[^170]: 超越静态位置编码：面向非平稳长期时间序列预测的贝叶斯时间感知注意力机制

    Beyond Static Positional Encoding: Bayesian Time-aware Attention for Non-Stationary Long-term Time Series Forecasting

    [https://arxiv.org/abs/2610.19603](https://arxiv.org/abs/2610.19603)

    该论文从贝叶斯视角将自注意力解释为内容先验与时间证据似然结合的后验，并推导总体层面的ELBO，据此揭示了静态位置编码在非平稳长期时间序列预测中的局限性，提出贝叶斯时间感知注意力机制以应对非平稳时间依赖建模的挑战。

    

    Transformer已在长期时间序列预测（LTSF）中展现出强大的性能，然而对非平稳时间依赖关系的建模仍然具有挑战性。在本文中，我们将自注意力解释为注意力位置上的贝叶斯后验，将由缩放点积相似度诱导的基于内容的先验，与由位置编码诱导的时间证据似然相结合。这一表述为位置编码如何塑造自注意力所捕获的时间依赖关系提供了概率解释。基于这一视角，我们推导出一个总体层面的证据下界（ELBO），刻画了位置编码在预测弱平稳时间序列中的作用。我们的分析进一步揭示了静态位置编码的一个局限性：当时间依赖结构在不同输入窗口之间存在差异时，固定的时间偏差必须适应异构的动态特性，因此可能是次优的。

    arXiv:2610.19603v1 Announce Type: new  Abstract: Transformers have demonstrated strong performance in long-term time series forecasting (LTSF), yet modeling non-stationary temporal dependencies remains challenging. In this paper, we interpret self-attention as a Bayesian posterior over attended positions, combining a content-based prior induced by scaled dot-product similarity with a temporal-evidence likelihood induced by positional encoding. This formulation provides a probabilistic interpretation of how positional encoding shapes the temporal dependencies captured by self-attention. Building on this perspective, we derive a population-level evidence lower bound (ELBO) that characterizes the role of positional encoding in forecasting weakly stationary time series. Our analysis further highlights a limitation of static positional encoding: when temporal dependence structures vary across input windows, a fixed temporal bias must accommodate heterogeneous dynamics and can therefore be s
    
[^171]: InstructPro2：自然语言与生物学背景引导的从头蛋白质设计

    InstructPro2: Natural Language and Biological Grounding Guided De Novo Protein Design

    [https://arxiv.org/abs/2610.19592](https://arxiv.org/abs/2610.19592)

    InstructPro2通过自然语言指令结合配体和抗原等生物学背景信息生成功能蛋白质序列，在2.057亿数据对上预训练后，其功能恢复F1达74.8，显著超越最强基线Pinal的62.2。

    

    生成具有所需功能的蛋白质是计算生物学的核心目标。尽管自然语言能够表达多样的设计目标，但现有的蛋白质生成方法通常仅支持狭窄的任务，或对生物学约束的利用有限。我们开发了InstructPro2，将自然语言设计目标转化为蛋白质序列，同时融入任务特定的生物学背景。文本编码器用于捕获设计指令，而专门的配体编码器和抗原编码器为序列生成解码器提供生物学背景信息。在预训练阶段，我们构建了包含来自UniProtKB的2.057亿个数据对的InstructPro2Bench基准数据集。随后，我们将该模型适配于配体结合蛋白和抗体设计任务。在功能蛋白质生成基准测试中，InstructPro2超越了最强基线Pinal，功能恢复F1达到74.8（对比62.2），并将全局RMSD从5.33A显著降低。

    arXiv:2610.19592v1 Announce Type: new  Abstract: The ability to generate proteins with desired functions is a central goal of computational biology. Although natural language can express diverse design objectives, existing protein generation methods often support only narrow tasks or make limited use of biological constraints. We develop InstructPro2 to translate natural-language design objectives into protein sequences while incorporating task-specific biological context. A text encoder captures the design instructions, while dedicated ligand and antigen encoders provide biological context to the sequence-generating decoder. For pretraining, we assemble InstructPro2Bench, comprising 205.7 million  pairs from UniProtKB. We subsequently adapt the model to ligand-binding protein and antibody design. On functional protein generation benchmarks, InstructPro2 outperforms Pinal, the strongest baseline, achieving a functional recovery F1 of 74.8 versus 62.2 and lowering global RMSD from 5.33A
    
[^172]: L3M：一个用于自我扩展与自我改进智能体的可验证框架与运行时

    L3M: A Verifiable Harness and Runtime for Self-Extending and Self-Improving Agents

    [https://arxiv.org/abs/2610.19591](https://arxiv.org/abs/2610.19591)

    L3M用Lean 4构建了一个可验证的运行时框架，使自我扩展与自我改进的智能体能够跨越多次调用累积计算并产生复合收益，同时无需人类参与每个环节。

    

    AI的持久性收益来自那些能够吸收更多计算量的方法，而语言模型已经能够在单次调用内扩展计算。然而，实际有用的工作发生在多次调用之间：跨越数小时或数天，通过工具和外部系统，在众多智能体及其服务的人类之间进行，并且越来越多地通过智能体自己编写的代码来实现。这些工作能否产生复合效应，取决于围绕它的框架，而该框架面临一个模型本身没有的约束：在人类、模型和代码这三类计算参与者中，只有人类的注意力无法扩展，因此只有当人类不必处于每个环节中时，计算才能持续增长。L3M正是基于这一前提构建的运行时系统，它使用Lean 4编写，使得同一种语言既实现了框架、又声明了其不变量，并且还是智能体用来扩展该系统的语言。一次计算贯穿人类、智能体、模型和代码，其全部状态都存放在Git中：形成丰富的状态轨迹，而不……

    arXiv:2610.19591v1 Announce Type: cross  Abstract: The lasting gains in AI have come from methods that absorb more computation, and language models already scale computation within a call. Useful work, though, happens across calls: over hours or days, through tools and external systems, among many agents and the humans they work for, and increasingly through code the agents write themselves. Whether that work compounds is enabled by the harness around it, and the harness faces a constraint the model does not: of the three participants in such a computation, the human, the model and code, only the human's attention does not scale, so the computation can grow only if the human is not in every loop. L3M is a runtime built on that premise, written in Lean 4 so that one language implements the harness, states its invariants, and is the language agents extend it in. One computation spans humans, agents, models and code, and all of its state lives in Git: a rich state trajectory -- rather tha
    
[^173]: 用于Hawkes过程背景强度估计的投影Wasserstein-Fisher-Rao流

    Projected Wasserstein-Fisher-Rao Flows for Hawkes Background-Intensity Estimation

    [https://arxiv.org/abs/2610.19588](https://arxiv.org/abs/2610.19588)

    本文提出一种基于Wasserstein-Fisher-Rao几何的似然变分方法，通过向统计流形投影的分布框架来估计Hawkes过程的背景强度，并证明了相应估计量的条件一致性定理。

    

    我们考虑在触发机制已知的情况下估计Hawkes过程背景强度的问题，将背景视为一个具有正则密度的未知有限正测度。我们提出了一种基于Wasserstein-Fisher-Rao几何的似然变分方法。在这一设定下，相关的一阶变分天然地是分布取值的，因此超出了经典梯度流表述的范畴。我们为由此产生的动力学建立了一个分布框架，并通过向统计流形的投影赋予其有限维意义。我们将真实背景识别为极限广义梯度的唯一零点，并在适当的稳定性、紧性、可积性和逼近条件下，为在概率趋于1的事件上满足经验投影方程的可测估计量建立了条件一致性定理。

    arXiv:2610.19588v1 Announce Type: cross  Abstract: We consider the problem of estimating the background intensity of a Hawkes process when the triggering mechanism is known, treating the background as an unknown finite positive measure with a regular density. We propose a likelihood-based variational approach based on the Wasserstein-Fisher-Rao geometry. In this setting, the relevant first variations are naturally distribution-valued and therefore fall outside the classical gradient-flow formulation. We develop a distributional framework for the resulting dynamics and give them a finite-dimensional meaning through projections onto statistical manifolds. We identify the true background as the unique zero of a limiting generalized gradient and, under suitable stability, compactness, integrability and approximation conditions, establish a conditional consistency theorem for measurable estimators satisfying the empirical projected equations on events whose probabilities tend to one. Finall
    
[^174]: 文本背后是谁？鲁棒的模型作者归属

    Who's Behind the Text? Robust Model Authorship Attribution

    [https://arxiv.org/abs/2610.19583](https://arxiv.org/abs/2610.19583)

    提出基于似然的模型作者归属方法ATLAS，能够在未见领域、未知写作场景以及模型家族未来版本等分布偏移下依然稳健地识别文本的来源大语言模型。

    

    随着大语言模型越来越多地被用于生成文本，其滥用已成为一个日益严重的社会问题，涉及学术不端、模型蒸馏和网络攻击等方面。因此，模型作者归属（MAA）——即识别给定文本的来源模型——对于促进大语言模型的负责任使用和提升人工智能安全具有重要意义。在实际应用中，归属系统必须在训练时观察到的分布之外仍然保持可靠，它们可能会遇到来自未见过的领域和写作场景的文本，以及来自系统训练时尚不存在的模型家族未来版本所生成的文本。然而，现有的MAA方法在这种分布偏移下往往会出现显著的性能下降。为应对这一挑战，我们提出了一种基于似然的来源归属作者追踪方法ATLAS，它通过以来源为条件的……（摘要在此处被截断）

    arXiv:2610.19583v1 Announce Type: cross  Abstract: As large language models (LLMs) are increasingly used to generate text, their misuse has become a growing societal problem, with implications for academic misconduct, model distillation, and cyberattacks. For these reasons, Model Authorship Attribution (MAA), which identifies the source model of a given text, is important for promoting the responsible use of LLMs and improving AI safety. In practice, attribution systems must remain reliable beyond the distributions observed during training. They may encounter texts from unseen domains and writing settings, as well as generations from future versions of model families that were not available when the system was trained. However, existing MAA methods often exhibit substantial performance degradation under such distribution shifts. To address this challenge, we propose Authorship Tracing via Likelihood-based Attribution of Sources (ATLAS), which formulates MAA through source-conditioned t
    
[^175]: 跨越拓扑鸿沟：基于语义先验的双分支图蒸馏实现鲁棒冷启动推荐

    Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation

    [https://arxiv.org/abs/2610.19578](https://arxiv.org/abs/2610.19578)

    提出DualGCN双分支图知识蒸馏框架，通过拓扑平滑的语义先验锚定教师网络的物品表示，并以解耦的两阶段训练将知识蒸馏给语义学生分支，从而弥合语义空间与协同空间之间的拓扑鸿沟，解决冷启动推荐中的表示坍塌问题。

    

    图协同过滤模型在推荐系统中取得了显著的成功。然而，由于未观测物品的拓扑隔离性，这些模型在严格的冷启动场景中固有地遭受灾难性的表示坍塌问题。虽然整合多模态语义特征提供了一种潜在的补救方案，但将孤立的语义属性直接映射到全局拓扑驱动的协同空间中，往往会引发表示退化和优化冲突。为了弥合这一根本性的结构鸿沟，我们提出了DualGCN——一个创新的双分支图知识蒸馏框架。DualGCN在解耦的两阶段训练范式下运行：首先构建一个鲁棒的协同教师网络，其中物品表示由拓扑平滑的语义先验锚定，有效重塑了潜在流形；随后，语义学生分支采用残差映射（摘要在此处截断）……

    arXiv:2610.19578v1 Announce Type: cross  Abstract: Graph Collaborative Filtering models have achieved remarkable success in recommendation systems. However, they inherently suffer from a catastrophic representation collapse in strict cold-start scenarios due to the topological isolation of unobserved items. While integrating multi-modal semantic features offers a potential remedy, directly mapping isolated semantic attributes into a globally topology-driven collaborative space frequently precipitates representation degradation and optimization conflicts. To bridge this fundamental structural divide, we propose DualGCN, an innovative Dual-Branch Graph Knowledge Distillation Framework. Operating under a decoupled two-stage training paradigm, DualGCN first constructs a robust collaborative teacher network where item representations are anchored by a topology-smoothed semantic prior, effectively reshaping the latent manifold. Subsequently, a semantic student branch employs a residual mappi
    
[^176]: 跨越时序图表示孤岛的学习

    Learning Across Temporal Graph Representation Silos

    [https://arxiv.org/abs/2610.19576](https://arxiv.org/abs/2610.19576)

    该研究首次证明关系历史中学习到的预测结构（关系动力学）能够跨越连续时间动态图与时序知识图谱之间的传统表示边界实现双向知识复用，且独立预训练的两类模型还可在分数层面融合以提升预测性能。

    

    时序图数据以不同的表示形式被记录，但这些差异是否也决定了所学知识复用的范围仍不清楚。本研究探讨了从关系历史中学习到的预测结构能否在连续时间动态图（CTDG）和时序知识图谱（TKG）之间进行复用。研究表明，在保留各自独立的输入与预测接口以及独立词表的情况下，双向复用得以实现，且相同表示形式的数据源并不总是更有用。对源历史的受控重组实验显示，促成复用的数据组织方式因源-目标任务组合而异。我们证明了关系历史中存在的预测结构——关系动力学（Relational Dynamics）——的某些习得方面可以跨越传统的表示边界进行复用。此外，独立预训练的TKG和CTDG学习器还可以在分数层面进行融合，从而提升预测性能。

    arXiv:2610.19576v1 Announce Type: new  Abstract: Temporal graph data are recorded in different representations, but whether these differences also determine the scope of learned knowledge reuse remains unclear. This study investigates whether predictive structure learned from relational histories can be reused between continuous-time dynamic graphs (CTDGs) and temporal knowledge graphs (TKGs). Bidirectional reuse emerges while retaining distinct input and prediction interfaces and independent vocabularies, and sources with the same representation are not always more useful. Controlled reorganizations of source histories show that the organization contributing to reuse varies across source-target conditions. We show that learned aspects of Relational Dynamics, the predictive structure present in relational histories, can be reused across conventional representation boundaries. Independently pretrained TKG and CTDG learners can also be fused at the score level to improve prediction on a 
    
[^177]: 论后验采样的固有隐私性：一个统一的Rényi散度框架

    On Inherent Privacy of Posterior Sampling: A Unified R\'enyi-Divergence Framework

    [https://arxiv.org/abs/2610.19573](https://arxiv.org/abs/2610.19573)

    本文提出统一的Rényi散度框架，证明从贝叶斯后验分布中采样（OPS）天然获得“免费”的差分隐私保证，其隐私损失由单条记录对数似然比的后验指数矩刻画，并在多种尾部条件与代表性贝叶斯模型下给出显式隐私保证。

    

    差分隐私（DP）为量化隐私保证提供了一个严格的框架。虽然大多数方法通过注入经过校准的随机噪声来实现差分隐私，但某些过程（如采样）的内在随机性可以“免费”带来差分隐私保证。在这项工作中，我们开发了一个统一的Rényi散度框架，用于刻画从贝叶斯后验分布中采样的固有差分隐私保证。我们证明，单次后验采样（One-Posterior-Sampling, OPS）所导致的隐私损失由单条记录对数似然比的后验指数矩决定，并在一致有界、次高斯或次指数尾部的情形下建立了显式的差分隐私保证。我们将该框架应用于代表性的贝叶斯模型，包括带任意先验的类别似然模型、高斯-高斯共轭对、来自指数族的广义线性模型以及高斯似然的线性回归等。

    arXiv:2610.19573v1 Announce Type: cross  Abstract: Differential Privacy (DP) provides a rigorous framework for quantifying privacy guarantees. While most methods achieve DP by injecting calibrated random noise, the intrinsic randomness of certain procedures such as sampling can yield DP guarantees ``for free''. In this work, we develop a unified R\'enyi divergence framework to characterize the inherent DP guarantees of sampling from a Bayesian posterior distribution. We show that the privacy loss incurred by One-Posterior-Sampling (OPS) is governed by posterior exponential moments of the single-record log-likelihood ratio, and establish explicit DP guarantees under uniformly bounded, sub-Gaussian, or sub-exponential tail regimes. We apply the framework to representative Bayesian models, including categorical likelihoods with arbitrary priors, Gaussian-Gaussian conjugate pair, generalized linear models from the exponential family and linear regression with Gaussian likelihood, both with
    
[^178]: 几何核多样性：欧几里得空间中的公理化多样性度量

    Geometric Kernel Diversity: An Axiomatic Diversity Measure in Euclidean Space

    [https://arxiv.org/abs/2610.19570](https://arxiv.org/abs/2610.19570)

    本文提出几何核多样性（GKD）——通过为每个点关联径向核立体并度量其并集体积来衡量多样性——该度量在欧几里得空间中同时满足单调性、唯一性与连续性三条公理，且在维度 d ≤ 3 时可被精确的多项式时间算法求解，从而回答了此前关于高效公理化多样性度量是否存在的悬而未决的问题。

    

    量化一组对象的多样性是机器学习与统计学中的一个基本问题。Mironov 和 Prokhorenkova（2025）最近证明，三条公理——单调性、唯一性和连续性——对于可靠的多样性度量至关重要，而文献中的标准度量无一能同时满足这三条公理。他们还构造了两个满足这些公理的度量，但计算它们都是 NP 难的，这就留下了一个悬而未决的问题：是否存在高效的公理化多样性度量。我们研究了该问题在欧几里得空间受限情形下的版本。我们提出了几何核多样性，它为每个点关联一个径向核立体，并以这些立体的并集体积来度量多样性。我们证明，标准 GKD 在欧几里得可实现的输入上满足全部三条 Mironov-Prokhorenkova 公理。对于标准 GKD，我们在维度 d ≤ 3 时给出了精确的多项式时间算法；在任意维度下，我们给出了一个随机……（原文摘要在此处截断）

    arXiv:2610.19570v1 Announce Type: cross  Abstract: Quantifying the diversity of a set of objects is a fundamental problem in machine learning and statistics. Mironov and Prokhorenkova (2025) recently showed that three axioms-monotonicity, uniqueness, and continuity-are crucial for a reliable diversity measure and that none of the standard metrics in the literature satisfy all three. They also constructed two measures that do satisfy the axioms, but both are NP-hard to compute, leaving open the question of whether efficient axiomatic diversity measures can exist. We study a Euclidean-restricted version of this question. We introduce Geometric Kernel Diversity (GKD), which associates a radial kernel solid with each point and measures the volume of their union. We show that canonical GKD satisfies all three Mironov-Prokhorenkova axioms on Euclidean-realizable inputs. For canonical GKD, we give exact polynomial-time algorithms in dimensions $d \le 3$. In arbitrary dimensions, we give a ran
    
[^179]: INTACT：在冻结骨干模型上诱导潜在推理

    INTACT: Inducing Latent Reasoning on a Frozen Backbone

    [https://arxiv.org/abs/2610.19568](https://arxiv.org/abs/2610.19568)

    INTACT 通过小型外部编码器结合蒸馏与强化学习，在完全冻结的大模型上诱导潜在推理，在不改变模型行为的前提下显著提升数学推理准确率并减少自回归解码开销。

    

    潜在推理用紧凑的连续“软词元”表示取代自然语言的思维链，减少了必须自回归解码的中间词元数量。我们使用一个小型外部编码器在冻结的骨干模型中诱导潜在推理。该编码器迭代地将骨干模型的隐藏状态映射为新的潜在块，构建一个潜在推理前缀，之后冻结的模型以文本形式写出其余的推理过程和答案。我们的方法 INTACT 通过蒸馏训练该编码器，使其再现书面推理步骤的效果，同时保持模型行为接近冻结的骨干模型，然后再根据答案正确性对其进行强化学习。在从 1B 到 8B 的四个冻结骨干模型上，INTACT 同时提升了 GSM8K 和平均域外准确率，包括在 Llama-3.2-3B 上 GSM8K 从 75.8% 提升至 79.3%，在 Qwen2.5-3B 上从 77.6% 提升至 79.7%。它还降低了端到端（原文摘要在此处截断）

    arXiv:2610.19568v1 Announce Type: new  Abstract: Latent reasoning replaces natural-language chains of thought with compact continuous "soft-token" representations, reducing the number of intermediate tokens that must be decoded autoregressively. We use a small external encoder to induce latent reasoning in a frozen backbone model. The encoder iteratively maps the backbone's hidden states to new latent blocks, constructing a latent reasoning prefix, after which the frozen model writes its remaining reasoning and its answer as text. Our method, INTACT, trains this encoder through distillation to reproduce the effect of written reasoning steps while keeping the model's behavior close to that of the frozen backbone, and then reinforces it on answer correctness. Across four frozen backbones from 1B to 8B, INTACT improves both GSM8K and average out-of-domain accuracy, including GSM8K gains from 75.8% to 79.3% on Llama-3.2-3B and from 77.6% to 79.7% on Qwen2.5-3B. It also reduces end-to-end l
    
[^180]: TEX-MEX：面向时间序列的相似性引导时序对比学习

    TEX-MEX: Similarity-Guided Temporal Contrastive Learning for Time Series

    [https://arxiv.org/abs/2610.19557](https://arxiv.org/abs/2610.19557)

    TEX-MEX首次将时间序列的结构相似性与时序相似性统一到单一对比学习框架中，通过编码器识别结构相似窗口，将样本拉向其时移版本、推离不相似窗口的时移版本，从而学习相似窗口在时间上通常伴随出现的模式。

    

    时间序列的对比学习依赖于决定哪些输入对应当被视为相似、哪些应被视为不相似。由于很少有数据增强能够可靠地保留时间序列的语义，先前的方法依赖两种互不相交的窗口相似性概念：结构相似性，即关联形态相似的窗口；以及时序相似性，即关联在时间上接近出现的窗口。我们提出了TEX-MEX，据我们所知，这是首个将两者统一为单一关系的方法。给定从整个数据集中随机抽取的一批窗口，TEX-MEX利用编码器识别哪些窗口在结构上彼此相似或不相似，然后将每个窗口拉向相似窗口的时移版本，并推离不相似窗口的时移版本，从而学习在时间上通常与相似窗口相伴出现的模式。在PPG、EEG、ECG、加速度计和连续血糖监测等数据上……（摘要原文在此处截断）

    arXiv:2610.19557v1 Announce Type: new  Abstract: Contrastive learning for time series depends on deciding which pairs of inputs should be treated as similar and dissimilar. Because few augmentations reliably preserve meaning in time series, prior methods have relied on two disjoint notions of similarity between windows: structural similarity, which relates windows with similar morphology, and temporal similarity, which relates windows that occur close together in time. We introduce TEX-MEX, to our knowledge the first approach to unify the two into a single relationship. Given a random batch of windows drawn across the dataset, TEX-MEX uses the encoder to identify which windows are structurally similar or dissimilar to each other. It then pulls each window toward the temporally shifted versions of similar windows and pushes it away from those of dissimilar ones, learning what typically accompanies similar windows in time. Across PPG, EEG, ECG, accelerometry, and continuous glucose monit
    
[^181]: 基于死亡调查报告的证据锚定药物过量死亡预测

    Evidence-Anchored Overdose Death Prediction from Death Investigation Reports

    [https://arxiv.org/abs/2610.19556](https://arxiv.org/abs/2610.19556)

    该论文提出了一个从死亡调查报告中预测疑似药物过量死亡的自动化框架，通过专家引导生成证据锚定的解释，并比较了流水线模型与自我合理化模型在该任务上的性能差异。

    

    arXiv:2610.19556v1 公告类型：新论文 摘要：在美国，由于涉及药物死亡的认证过程复杂，药物过量死亡的确认常常被延误。死亡调查报告是一种丰富但未被充分利用的数据源，可用于预测某例死亡是否为疑似药物过量（即在最终死因和死亡方式判定之前被初步认定为药物过量的死亡）。我们提出了一个自动化框架，能够从死亡调查报告中对疑似药物过量死亡进行分类，同时生成由专家知识引导的支持性解释。通过专家引导的设计过程，我们研究了以下两个问题：1）哪种类型的解释最有利于提升模型预测性能；2）哪种架构更适合该任务：是先进行合理化解释再预测的流水线模型，还是将预测与合理化解释联合进行的自我合理化模型？我们的实验表明，采用基于证据锚定的解释流水线模型（原文在此处截断）……

    arXiv:2610.19556v1 Announce Type: new  Abstract: Confirmation of overdose deaths in the United States is often delayed due to the complexities of certifying drug-involved deaths. Death investigation reports represent a rich but underutilized data source that could be used to predict whether a death is a suspected overdose (i.e., a death preliminarily identified as an overdose prior to the final cause and manner of death determination). We present an automated framework that classifies suspected overdoses from death investigation reports, while producing supporting explanations guided by expert knowledge. Through an expert-guided design process, we investigate 1) what type of explanations are best for model prediction performance, and 2) which architecture is better suited for this task: pipeline models that rationalize first and then predict, or self-rationalizing models that jointly predict and rationalize? Our experiments show that a pipeline model with explanations grounded in the i
    
[^182]: 测试时预测：循环深度实现上下文动力学预测

    Test-Time Forecasting: Recurrent Depth Enables In-Context Dynamics Prediction

    [https://arxiv.org/abs/2610.19552](https://arxiv.org/abs/2610.19552)

    该论文通过建立线性注意力与动态模态分解的联系，提出测试时预测（TTF）框架，以梯度下降迭代作为循环深度，使仅经过单步预测训练的轻量级模型在自回归展开中即可泛化到分布外的动力系统动力学。

    

    面向动力系统的基础模型能够在推理阶段对未见过的偏微分方程（PDE）、常微分方程（ODE）以及时间序列进行预测。然而，这些基础模型是否真正学到了可泛化的上下文预测策略，仍是一个悬而未决的问题。我们首先建立了线性注意力与（扩展）动态模态分解之间的联系，据此证明循环 transformer 能够在上下文中模拟动力系统的传播子。随后，我们将这一洞见推广为“测试时预测”（Test-Time Forecasting, TTF）框架，其中单步预测器利用梯度下降的迭代过程作为循环深度。我们证明，仅在 ODE 和 PDE 上进行单步预测训练的轻量级 TTF 模型，在自回归展开时能够泛化到分布外的系统动力学。我们的结果进一步证实了 TTF 是一个具有理论依据的框架，可用于设计……（原文此处截断）

    arXiv:2610.19552v1 Announce Type: new  Abstract: Foundation models for dynamical systems can forecast unseen partial differential equations (PDEs), ordinary differential equations (ODEs), and time series during inference. However, it remains an open question whether these foundation models truly learn generalizable in-context forecasting strategies. We show that looped transformers are capable of emulating propagators for dynamical systems in-context by first establishing a link between linear attention and (extended) dynamic mode decomposition. We then generalize this insight into the $\textit{Test-Time Forecasting}$ (TTF) framework in which a one-step forecaster leverages iterations of gradient descent as recurrent depth. We demonstrate that lightweight TTF models trained on ODEs and PDEs generalize to out-of-distribution system dynamics during autoregressive rollout after being trained for one-step prediction. Our results reinforce TTF as a theoretically motivated framework for desi
    
[^183]: 基于多个不可靠标签的噪声标签校准

    Noisy-Label Calibration with Multiple Unreliable Labels

    [https://arxiv.org/abs/2610.19543](https://arxiv.org/abs/2610.19543)

    该论文提出了噪声校正ECE（NC-ECE）估计器、噪声概率估计算法以及事后重校准方法，使得在仅有多个不可靠标注的情况下，也能对二元分类器的校准性能进行无偏评估和可靠重校准。

    

    在医学标注和众包场景中，多个标注者会为每个样本分配标签，但这些标签可能是错误的。将此类噪声标签视为真实标签会使校准评估产生偏差。为解决这一问题，我们提出了在潜在干净标签分布下，仅利用噪声标注来评估和重新校准固定二元预测器的方法。首先，在给定各标注者噪声概率的条件下，我们提出了噪声校正ECE（NC-ECE），即类条件噪声下期望校准误差（ECE）的分箱估计器，并推导了该估计器偏差的非渐近界。随后，我们在条件独立性和标签方向性假设下，提出了估计未知噪声概率的实用算法。我们还开发了基于损失和基于直方图的事后重校准方法，分别具有干净风险保证和校准保证。在半合成基准上的实验表明，所提出的方法……

    arXiv:2610.19543v1 Announce Type: cross  Abstract: In medical annotation and crowdsourcing, multiple annotators assign labels to each example, but these labels may be incorrect. Treating such noisy labels as ground truth can bias calibration evaluation. To address this problem, we propose methods for evaluating and recalibrating a fixed binary predictor under the underlying clean-label distribution, using only noisy annotations. First, given annotator-specific noise probabilities, we propose Noise Corrected ECE (NC-ECE), a binned estimator of expected calibration error (ECE) under class-conditional noise. We derive non-asymptotic bounds on the estimator's bias. Then, we propose practical algorithms for estimating unknown noise probabilities under conditional independence and label-orientation assumptions. We also develop loss-based and histogram post-hoc recalibration with clean-risk and calibration guarantees, respectively. Experiments on semi-synthetic benchmarks show that the propos
    
[^184]: 两阶段残差学习用于建模强度感知的热力系统优化

    Two-stage Residual Learning for Modelling Strength-Informed Optimisation of Thermal Systems

    [https://arxiv.org/abs/2610.19539](https://arxiv.org/abs/2610.19539)

    提出了一种基于人工神经网络的两阶段残差学习（TSRL）框架，通过基础神经网络与残差神经网络的级联建模，结合局部建模强度评估与领域约束，显著降低了热力系统代理模型的预测误差，实现了更可靠的热力系统优化。

    

    总体性能指标无法识别用于热力系统优化的机器学习（ML）代理模型中的局部建模不准确问题。本研究通过基于验证的可靠性筛选和优化后残差评估来评估局部建模强度。研究开发了一种基于人工神经网络（ANN）的两阶段残差学习（TSRL）框架，用于建模强度感知且受领域约束的热力系统优化。在TSRL框架中，一个基础神经网络预测过程输出，另一个残差神经网络学习基础神经网络留下的带符号残差。TSRL框架在两个热力系统案例研究中得到实现，包括原油蒸馏装置的817个模拟观测数据和一台395 MW工业燃气轮机的运行记录。TSRL框架将炼油数据集中燃料油的测试均方根误差（RMSE）从10.3 g/s降低到4.7 g/s，并将……（摘要在此处被截断）

    arXiv:2610.19539v1 Announce Type: new  Abstract: Aggregate performance metrics do not identify local modelling inaccuracies in machine learning (ML) surrogates used for thermal-system optimisation. This study assesses local modelling strength through validation-based reliability screening and post-optimisation residual assessment. A two-stage residual learning (TSRL) framework, built on artificial neural networks (ANNs), is developed for modelling-strength-informed and domain-constrained optimisation of thermal systems. Within the TSRL framework, a base ANN predicts the process output, and a residual ANN learns the signed residuals left by the base ANN. The TSRL framework is implemented on two case studies of thermal systems, including 817 simulation observations of a crude distillation unit and operating records from a 395 MW industrial gas turbine. The TSRL framework reduces the test root mean square error (RMSE) from 10.3 g/s to 4.7 g/s for Fuel Oil in the refinery dataset and from 
    
[^185]: 表格数据上的类型化决策模型

    Typed Decision Models on Tabular Data

    [https://arxiv.org/abs/2610.19536](https://arxiv.org/abs/2610.19536)

    类型化决策模型在表格数据任务上以更低成本达到与大语言模型相当的性能，且对问题表述更加鲁棒，但所有模型对表格组织方式的变化都不够稳健。

    

    类型化决策模型是一类语言模型，它们通过为一组固定的候选答案中的每个答案返回概率来回答问题，而不是生成文本。我们在表格数据上研究这类模型，将四种决策模型与三个大语言模型（LLM）在行分类、事实核查和异常检测任务上进行比较。每个请求都以原始形式发送，同时也在21种保持内容不变的改写下发送，例如重命名选项或以其他格式书写表格。决策模型的表现与LLM相当，但成本更低。其中三个模型对问题的表述方式具有更强的鲁棒性。然而，我们测试的所有模型对表格的组织方式都不鲁棒。对于每个模型，在表格或问题的某种改写下，至少有三分之一的行决策会发生变化。决策模型赋予低概率的决策是最容易发生改变的。决策模型是一种实用的选择。

    arXiv:2610.19536v1 Announce Type: new  Abstract: Typed decision models are language models that answer a question with a fixed set of possible answers by returning a probability for each answer instead of generating text. We study them on tabular data and compare four decision models with three large language models (LLMs) on row classification, fact verification and anomaly detection. Every request is sent in its original form and under 21 types of rewrite that keep its content, such as renaming the options or writing the table in another format. The decision models perform as well as the LLMs at a lower cost. Three of them are also far more robust to how the question is formulated. No model we test is robust to how the table is organized. For every model, at least a third of the row decisions change under some rewrite of the table or the question. The decisions to which a decision model gives low probability are the ones most likely to change. Decision models are a practical choice f
    
[^186]: 面向问题与辅导对话的统一响应框架

    A Unified Response Framework for Questions and Tutoring Dialogues

    [https://arxiv.org/abs/2610.19532](https://arxiv.org/abs/2610.19532)

    提出了一个统一框架，将学生对问题的回答与辅导对话中导师提出的子任务共同建模在项目反应理论（IRT）量表上，通过文本估计难度、追踪学生动态能力演化，并利用导师干预行为正则化难度值，实现对学业知识的更细粒度评估。

    

    现有教育应用中的学生模型主要关注学生对问题的回答。然而，随着辅导服务在大规模上日益普及，分析导师与学生之间的对话变得尤为重要，因为这些对话可能揭示关于学生知识的细粒度证据。在这项工作中，我们提出了一个统一框架，将学生对问题的回答以及辅导对话中导师提出的子任务建模在同一个项目反应理论（IRT）量表上。我们首先在辅导对话中识别数学子任务，将这些子任务改写为自包含的问题，并判定学生对这些子任务回答的正确性。随后，我们开发了一个IRT框架，从文本中估计问题/子任务的难度，追踪学生在问答和对话交互过程中不断变化的能力，并提出一种新颖的方法，利用导师在连续子任务之间的干预行为来正则化子任务的难度值。我们评估了我们的框架……

    arXiv:2610.19532v1 Announce Type: new  Abstract: Existing student models in educational applications largely focus on student responses to questions. However, as tutoring becomes more accessible at large scale, it is important to analyze dialogues between tutors and students, which may reveal fine-grained evidence on student knowledge. In this work, we propose a unified framework that models both student responses to questions and tutor-posed subtasks during dialogues on a shared item response theory (IRT) scale. We first identify mathematical subtasks in tutoring dialogues, rewrite them into self-contained questions, and identify the correctness of student responses to these subtasks. We then develop an IRT framework that estimates question/subtask difficulty from text, tracks students' evolving abilities across question-answering and dialogue interactions, and a novel method to leverages tutor moves between consecutive subtasks to regularize their difficulty values. We evaluate our f
    
[^187]: 诊断与纠正可验证指令遵循中的奖励作弊

    Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following

    [https://arxiv.org/abs/2610.19525](https://arxiv.org/abs/2610.19525)

    CALIF 是一种轻量级方法，通过利用现有冻结参考策略计算经校准的基于似然的奖励加成来增强验证器奖励，无需额外评判模型即可缓解可验证指令遵循中的奖励作弊问题并保持回复质量。

    

    可验证指令遵循基准强调输出约束的满足，但可能忽视回复是否切实地回答了底层请求，从而形成评估盲区。我们表明，前沿模型可以在几乎没有实质性答案内容的回复上获得满分。将同样的基于验证器的标准用作训练奖励会诱发奖励作弊：公开发布的检查点在经过可验证奖励的强化学习后，约束满足度更高，但回复质量更低。为解决这一奖励作弊问题，我们提出了 CALIF，这是一种轻量级方法，它通过一个经过校准的基于似然的奖励加成来增强验证器奖励，从而保持回复质量。CALIF 仅利用现有的冻结参考策略，在只有请求本身（不含约束）的条件下评估回复的似然，无需额外的评判模型或习得的奖励模型。在从 1.7B 到 8B 的推理与非推理模型上（摘要至此截断）。

    arXiv:2610.19525v1 Announce Type: cross  Abstract: Verifiable instruction-following benchmarks emphasize output-constraint satisfaction but can overlook whether responses meaningfully answer the underlying request, creating an evaluation blind spot. We show that frontier models can receive full credit for responses with little substantive answer content. Using the same verifier-based criterion as a training reward can induce reward hacking, as publicly released checkpoints exhibit higher constraint satisfaction but lower response quality after reinforcement learning with verifiable rewards. To address this reward hacking, we introduce CALIF, a lightweight method that augments verifier rewards with a calibrated likelihood-based bonus to preserve response quality. CALIF evaluates response likelihood under the request alone using the existing frozen reference policy, requiring no additional judge or learned reward model. Across models ranging from 1.7B to 8B in reasoning and non-reasoning
    
[^188]: HankelKANet：基于物理启发汉克尔卷积的轻量级路径损耗估计

    HankelKANet: Lightweight Path Loss Estimation via Physics-Inspired Hankel Convolution

    [https://arxiv.org/abs/2610.19524](https://arxiv.org/abs/2610.19524)

    HankelKANet通过用零阶和一阶汉克尔函数的径向幅度参数化卷积核，将电磁传播特性结构化嵌入轻量级残差UNet架构，无需额外物理损失项即可实现最先进精度和效率的路径损耗估计。

    

    准确的路径损耗（PL）估计对于无线网络的规划与优化至关重要。本文提出了HankelKANet，一种用于路径损耗估计的轻量级物理启发网络，它将电磁传播特性结构化地嵌入到神经架构中。该模型建立在单个残差UNet骨干网络之上，将城市布局特征与卷积Kolmogorov-Arnold网络（KAN）相结合。具体而言，卷积核由零阶和一阶汉克尔函数的径向幅度进行参数化。这种设计作为一种物理启发的非线性距离编码，使网络能够直接捕获主要的衰减特性，而无需额外的基于物理的损失项。实验结果表明，HankelKANet在模拟的RadioMapSeer数据集和真实世界的华为RSRPSet数据集上均实现了最先进的精度和效率。

    arXiv:2610.19524v1 Announce Type: new  Abstract: Accurate path loss (PL) estimation is essential for wireless network planning and optimization. This paper proposes HankelKANet, a lightweight physics-inspired network for PL estimation that structurally embeds electromagnetic propagation characteristics into the neural architecture. Built on a single residual UNet backbone, the proposed model integrates urban layout features with a convolutional Kolmogorov-Arnold network(KAN). Specifically, the convolutional kernels are parameterized by the radial amplitudes of the zeroth- and first-order Hankel functions. This formulation acts as a physics-inspired nonlinear distance encoding, enabling the network to capture dominant attenuation characteristics directly without requiring additional physics-based loss terms. Experimental results demonstrate that HankelKANet achieves state-of-the-art accuracy and efficiency on both the simulated RadioMapSeer dataset and the real-world Huawei RSRPSet. Com
    
[^189]: Feel-Good Thompson 采样的高概率遗憾界

    High-Probability Regret Bounds for Feel-Good Thompson Sampling

    [https://arxiv.org/abs/2610.19513](https://arxiv.org/abs/2610.19513)

    本文首次为 Feel-Good Thompson 采样在上下文老虎机中建立了高概率遗憾界，在线性设置下达到了与最优算法在维度和时间范围依赖性上相匹配的遗憾保证。

    

    Feel-Good Thompson 采样（FGTS）通过在标准 Thompson 采样的后验中加入一个探索项来增强，该探索项倾向于偏向预测最优值较高的奖励函数。这一改进在线性上下文老虎机中实现了除对数因子外极小极大最优的遗憾，但现有关于其两种标准变体的分析仅建立了期望意义下的遗憾保证。在本文中，我们在具有一般可实现函数逼近的上下文老虎机中，首次为累积型（cumulative）和当前步型（current-step）两种 FGTS 推导出了高概率遗憾界。我们的界取决于相应的解耦系数以及真实奖励函数附近的先验概率质量。在 d 维线性设置下，两种变体都以至少 1-δ 的概率满足 Regret(T)=O(d√T log(2T)+√T log(1/δ))，其中 T 为时间范围。该界在维度和时间范围依赖性上与基于……的算法相匹配。

    arXiv:2610.19513v1 Announce Type: cross  Abstract: Feel-good Thompson sampling (FGTS) augments the posterior of vanilla Thompson sampling with an exploration term that favors reward functions with high predicted optimal values. This modification achieves minimax-optimal regret up to logarithmic factors in linear contextual bandits, but existing analyses of its two standard variants establish regret guarantees in expectation. In this paper, we derive the first high-probability regret bounds for both cumulative and current-step FGTS in contextual bandits with general realizable function approximation. Our bounds depend on the corresponding decoupling coefficients and the prior mass near the true reward function. In the $d$-dimensional linear setting, both variants satisfy $\mathrm{Regret}(T)=O\big(d\sqrt T\log(2T)+\sqrt T\log(1/\delta)\big)$ with probability at least $1-\delta$, where $T$ is the time horizon. This bound matches the dimension and horizon dependence of algorithms based on 
    
[^190]: CktHGVAE：一种用于晶体管级模拟/混合信号电路设计与分析的对称式设计/状态异构图变分自编码器

    CktHGVAE: A Symmetric Design/State Heterogeneous Graph Variational Autoencoder for Transistor-level Analog/Mixed-signal Circuit Design and Analysis

    [https://arxiv.org/abs/2610.19508](https://arxiv.org/abs/2610.19508)

    提出了一种同时以设计变量和状态变量为输入输出的异构图变分自编码器CktHGVAE，无需针对电路拓扑专门调参即可高精度预测晶体管级模拟/混合信号电路的工作状态，并能在潜在空间中实现电路规格驱动的自动化设计优化。

    

    我们提出了CktHGVAE，这是一种面向晶体管级模拟/混合信号CMOS电路的异构图变分自编码器，它不仅将设计变量，还将状态变量同时作为输入和输出。在采用单一超参数配置、无需针对特定电路拓扑进行调整的情况下，模型生成的电路工作状态与SPICE仿真结果的对比显示：对于典型的单端电压放大器和CMOS反相器，中位R²值超过0.9；对于数据集更大的差分放大器，R²值介于0.81至0.96之间。我们以NMOS输入、电阻负载的共源电压放大器为例，在经PCA降维的潜在空间中演示了基于贝叶斯优化、进化搜索、随机搜索和准穷举搜索的潜在空间优化方法，其中目标函数被定义为对偏离所需电路规格的惩罚。对于每种搜索策略，在所有被推荐的设计候选方案中，93.3%至100%的方案……

    arXiv:2610.19508v1 Announce Type: new  Abstract: We propose CktHGVAE, a heterogeneous graph variational autoencoder for transistor-level analog/mixedsignal CMOS circuits that takes not only design variables but also state variables as its inputs and outputs. With a single hyperparameter configuration without topology-specific tuning, comparison between model-generated and SPICE-simulated circuit operational states gives median R2 values above 0.9 for typical single-ended voltage amplifiers and the CMOS inverter, and 0.81-0.96 for differential amplifiers with larger datasets. Latent-space optimization in the PCA-reduced latent space with Bayesian optimization, evolutionary search, random search, and quasi-exhaustive search is demonstrated for a common-source voltage amplifier with NMOS input and resistive load, where the objective function is defined as a penalty for deviation from required circuit specifications. For each search strategy, 93.3 - 100% of all design candidates suggested 
    
[^191]: SKG：从推理到归纳的定律驱动引擎

    SKG: A Law-Induced Engine from Inference to Induction

    [https://arxiv.org/abs/2610.19505](https://arxiv.org/abs/2610.19505)

    该论文提出SKG推理引擎，将辛几何、核方法与约束力学统一为一个八元组框架，把守恒律、本构关系、对称群和公理系统四个层次的定律嵌入学习过程，实现了从先验定律驱动的推理到归纳的统一、可验证的方法体系。

    

    本文提出了SKG（辛核梯度，Symplectic-Kernel Gradient）推理引擎，用于定律驱动的推理与归纳。定律是先验的，数据提供实例，推理则是定律约束下的几何演化。我们将辛几何、GENERIC框架、核方法、再生核希尔伯特空间（RKHS）、群等变网络和约束力学整合为一个统一的、可分析的、可实现的、可验证的框架。我们提出了SKG八元组 S = (M, g, ω, H, S, G, Δ, Π)，其四个组成部分对应四个层次的定律：守恒律、本构关系、对称群和公理系统。主要结果：(1) 不变量通过雅可比矩阵进入度量构造：g = J^T Σ^-1 J + ε_G I。对于常数情形，我们建立了逆重构理论：可解性秩条件（定理3.9）、一致性判据（定理3.11）和唯一性定理（定理3.14）。(2) 核带宽矩阵由Fisher信息唯一确定（摘要在此处截断）

    arXiv:2610.19505v1 Announce Type: new  Abstract: This paper proposes the SKG (Symplectic-Kernel Gradient) reasoning engine for law-induced inference and induction. Laws are a priori; data provide instances. Inference is geometric evolution under law constraints. We organize symplectic geometry, GENERIC, kernel methods, RKHS, group-equivariant networks, and constrained mechanics into a unified, analyzable, implementable, verifiable framework. We propose the SKG octuple S = (M, g, omega, H, S, G, Delta, Pi), whose four components correspond to four levels of laws: conservation laws, constitutive relations, symmetry groups, and axiom systems.   Main results: (1) Invariants enter metric construction via the Jacobian: g = J^T Sigma^-1 J + epsilon_G I. For constant cases, we establish inverse reconstruction theory: solvability rank condition (Thm 3.9), consistency criterion (Thm 3.11), and uniqueness theorem (Thm 3.14). (2) The kernel bandwidth matrix is uniquely determined by Fisher informa
    
[^192]: 你所签署的并非你所运行的：量化大语言模型供应链中的可验证性差距

    What You Sign Is Not What You Run: The Verifiability Gap in Quantized LLM Supply Chains

    [https://arxiv.org/abs/2610.19498](https://arxiv.org/abs/2610.19498)

    该论文提出“独立可验证率（IVR）”这一新指标，发现下载量最高的 198 个 Hugging Face GGUF 量化模型仓库均未记录构建环境，导致第三方完全无法逐比特复现用户实际运行的量化产物（IVR 为 0/198），从而揭示了“已签名的全精度权重”与“实际运行的量化产物”之间的供应链可验证性鸿沟，并指出补充记录浮点收缩设置等被遗漏的构建坐标可实现复现。

    

    模型签名用于认证大语言模型发布时所附带的全精度权重 W，然而用户实际运行的是一个量化产物 Q，而在当前的披露规范下，生成该量化产物的转换过程并非唯一确定。我们将非环境输入与执行环境 E 分离，并测量了一个据我们所知此前从未被测量过的量：独立可验证率，即在按受欢迎程度抽样的 GGUF 生态系统中，第三方能够逐比特（bit-for-bit）重新推导出的产物所占的比例。在下载量最高的 200 个 Hugging Face GGUF 仓库中（198 个在研究范围内），没有任何产物或模型卡记录其构建环境，因此在“记录输入协议”下，IVR 为 0/198。然而这些产物并非不可复现。在四个披露了其输入的样本内仓库中，仅凭模型卡上的信息无法复现任何 Q4_K_M 文件，但模型卡所遗漏的两个关键坐标——构建时的浮点收缩设置，以及（针对混合模型的）……（原文摘要在此处截断）

    arXiv:2610.19498v1 Announce Type: cross  Abstract: Model signing authenticates the full-precision weights W an LLM is released with, yet users run a quantized artifact Q whose producing transformation is not uniquely determined under current disclosure. We separate the non-environment inputs from the execution environment E and measure a quantity that, to our knowledge, has not been measured before: the Independent Verifiability Rate (IVR), the fraction of a popularity-sampled GGUF ecosystem that a third party can re-derive bit-for-bit. Across the 200 most-downloaded Hugging Face GGUF repositories (198 in scope), no artifact or model card records the build environment, so under a recorded-inputs protocol the IVR is 0/198. Yet the artifacts are not irreproducible. On four in-sample repositories that disclose their inputs, the card's information alone reproduces no Q4_K_M file, but two coordinates the card omits - the build's floating-point-contraction setting and, for hybrid models, the
    
[^193]: 基于Transformer神经算子的带内全双工系统数字自干扰消除

    Digital Self-Interference Cancellation for In-Band Full-Duplex Systems Based on Transformer Neural Operators

    [https://arxiv.org/abs/2610.19490](https://arxiv.org/abs/2610.19490)

    本文将带内全双工系统中的数字自干扰消除重新建模为算子学习问题，利用Transformer神经算子从历史观测中学习残余自干扰信道的时间演化，即使在期望信号强于残余自干扰的困难场景下也能实现更优的消除性能。

    

    数字域自干扰消除在带内全双工无线电台中，于被动抑制和模拟消除之后是必不可少的。当期望信号相对于残余自干扰较强时，这一任务尤其具有挑战性。在这种场景下，传统的迭代数字消除器会从被期望信号污染的接收样本中更新其系数，可能导致严重的性能下降。本文将数字域自干扰消除重新表述为一个算子学习问题。所提出的框架从历史观测中学习残余自干扰信道的时间演化，并利用学习到的算子结合已知的本地发射信号来重建当前的残余自干扰。该框架采用Transformer神经算子来实现。所得到的方法在无期望信号的环境下进行训练，然后应用于实际场景（原文摘要在此处截断）。

    arXiv:2610.19490v1 Announce Type: cross  Abstract: Digital-domain self-interference cancellation is essential in in-band full-duplex radios after passive suppression and analog cancellation. It is particularly challenging when the desired signal is strong relative to the residual self-interference. In this regime, conventional iterative digital cancellers update their coefficients from desired-signal-contaminated received samples and may suffer substantial performance degradation. This paper reformulates digital-domain self-interference cancellation as an operator-learning problem. The proposed framework learns the temporal evolution of the residual self-interference channel from historical observations and uses the learned operator together with the known local transmit signal to reconstruct the current residual self-interference. A Transformer neural operator is used to realize this framework. The resulting method is trained in a desired-signal-free setting and then applied in the pr
    
[^194]: 状态匹配序列保形预测

    State-Matched Sequence Conformal Prediction

    [https://arxiv.org/abs/2610.19484](https://arxiv.org/abs/2610.19484)

    提出了状态匹配序列保形预测（SMSCP），通过构建局部线性状态并利用状态相似性选取匹配的历史校准分数参考集，解决了可比状态在非相邻时间重现时基于近期性的传统方法失效的问题。

    

    序列保形预测需要决定哪些过去的误差应当用于校准当前目标的不确定性。当近期观测仍然具有相关性时，近期性原则和在线误差反馈是有用的；但当可比状态在非相邻时间点重现时，最近的过去可能是一个糟糕的参考，且这种重现模式通常是未知的。我们提出了状态匹配序列保形预测（SMSCP），该方法从目标前信息中构建显式的单侧局部线性状态，利用状态相似性选择过去校准分数的匹配参考集，然后应用普通的未加权保形次序统计量。与现有方法不同，SMSCP利用状态相似性来识别与目标分数分位数相关的校准观测。这促使我们对时间依赖性下邻近状态是否具有可比的目标分数分位数进行理论分析。我们建立了对（摘要在此处被截断）的控制。

    arXiv:2610.19484v1 Announce Type: cross  Abstract: Sequential conformal prediction needs to decide which past errors should calibrate uncertainty for the current target. Recency and online error feedback are useful when recent observations remain relevant, but the recent past can be a poor reference when comparable states recur at nonadjacent times, with the recurrence pattern generally unknown. We propose State-Matched Sequence Conformal Prediction (SMSCP), which constructs an explicit one-sided local-linear state from pre-target information, uses state similarity to select a matched reference set of past calibration scores, and then applies the ordinary unweighted conformal order statistic. In contrast to existing approaches, SMSCP uses state similarity to identify calibration observations relevant to the target score quantile. This motivates a theoretical analysis of whether nearby states have comparable target score quantiles under temporal dependence. We establish control of the r
    
[^195]: MIPS：在大结果空间中缓解强化学习的模式崩塌

    MIPS: Mitigating Mode Collapse in Reinforcement Learning for Large Outcome Spaces

    [https://arxiv.org/abs/2610.19457](https://arxiv.org/abs/2610.19457)

    提出MIPS方法，通过将采样轨迹与学习到的轨迹模型相结合进行边际概率评估，解决了大结果空间强化学习中因结果稀疏导致IPS概率估计失效、从而无法有效缓解模式崩塌的问题。

    

    强化学习中的期望奖励最大化会将概率质量集中在少数几个结果上，即使在需要许多不同、高质量解的任务中也是如此，这种失败被称为模式崩塌。逆概率缩放在基于组的策略梯度方法（如GRPO）中，通过将每个奖励除以该结果的概率来对抗这一问题，该概率由其在采样组内的出现频率估计得出。然而，当结果空间相对于组规模较大时，结果很少重复出现，这些估计便失去了信息量。由于一个结果可以沿着许多不同的轨迹构造出来，其概率是所有这些轨迹概率的总和，直接计算是不可行的。我们提出了边际IPS（MIPS），它通过将采样轨迹与一个学习到的、通向每个结果的轨迹模型相结合，来执行边际评估。在可以精确计算采样分布的Hypergrids任务上，MIPS保持接近奖励分布……（摘要原文在此处截断）

    arXiv:2610.19457v1 Announce Type: new  Abstract: Expected-reward maximization in reinforcement learning concentrates probability mass on a few outcomes, even in tasks that require many distinct, high-quality solutions, a failure known as mode collapse. Inverse Probability Scaling (IPS) counters this in group-based policy gradients such as GRPO by dividing each reward by the outcome's probability, estimated from its frequency within the sampled group. When the outcome space is large relative to the group, outcomes rarely repeat and these estimates become uninformative. Since an outcome can be constructed along many trajectories, its probability is a sum over all of them, and computing it directly is intractable. We propose Marginal IPS (MIPS), which performs marginal evaluation by combining the sampled trajectory with a learned model of the trajectories that lead to each outcome. On Hypergrids, where the exact sampling distribution is computable, MIPS stays near the reward-distribution 
    
[^196]: 随机重排随机梯度下降的大步长末次迭代收敛性

    Large-Step Last-Iterate Convergence of Random-Reshuffling Stochastic Gradient Descent

    [https://arxiv.org/abs/2610.19444](https://arxiv.org/abs/2610.19444)

    本文为随机重排随机梯度下降（RR-SGD）的末次迭代点建立了大步长收敛保证，通过保留随机比较点的采样偏差，从步长限制中移除了显式的 1/n 因子，使末次迭代的收敛速率在对数因子内匹配平均迭代点的收敛速率。

    

    随机重排随机梯度下降（RR-SGD）在机器学习中广泛应用于有限和优化问题。对于凸的 L-光滑分量函数，近期的工作为平均迭代点建立了收敛性保证，其分量步长可达到 1/L 的量级。然而，末次迭代点是实践中自然的输出，而现有在此设定下的末次迭代收敛保证要求步长与分量数 n 成反比。我们在无约束问题上为标准 RR-SGD 的末次迭代点建立了期望目标函数间隙界，所用常数分量步长满足 0<η≲1/(L(1+log(nK)))，其中 K 为训练轮数。通过在随机比较点保留采样偏差，我们从步长限制中移除了显式的 1/n 因子。结合现有的小步长界，我们的结果在对数因子内匹配了相应的平均迭代点收敛速率。

    arXiv:2610.19444v1 Announce Type: cross  Abstract: Random reshuffling stochastic gradient descent (RR-SGD) is widely used for finite-sum optimization in machine learning. For convex $L$-smooth components, recent work establishes convergence guarantees for an averaged iterate with component stepsizes as large as order $1/L$. However, the final iterate is a natural output in practice, and existing last-iterate guarantees in this setting require stepsizes that scale inversely with the number of components $n$. We establish an expected objective-gap bound for the last iterate of standard RR-SGD on unconstrained problems with a constant component stepsize satisfying $0<\eta\lesssim 1/(L(1 + \log(nK)))$, where $K$ is the number of epochs. By retaining the sampling bias at random comparison points, we remove the explicit $1/n$ factor from the stepsize restriction. Combined with existing small-step bounds, our result matches the corresponding averaged-iterate convergence rate up to logarithmic
    
[^197]: 面向小语言模型智能体停滞问题的进度感知蒸馏方法

    Progress-Aware Distillation for Mitigating Stagnation in Small Language Model Agents

    [https://arxiv.org/abs/2610.19439](https://arxiv.org/abs/2610.19439)

    该论文针对蒸馏得到的小语言模型智能体在长交互轨迹中易陷入低效循环的进度停滞问题，提出了进度感知蒸馏方法予以缓解。

    

    基于大语言模型（LLM）的智能体系统在推理和解决复杂现实任务方面已展现出卓越的能力，但同时也伴随着大量的token消耗和推理延迟，限制了其实际部署。为了提升智能体系统的效率，近期的研究工作聚焦于通过基于监督微调的蒸馏方法，将大规模模型的智能体能力迁移到小语言模型（SLM）中。然而，随着交互轨迹的增长，涉及更长的推理-行动链和大量的环境反馈时，经过蒸馏的学生智能体往往会出现进度停滞现象：陷入低效循环，表现为持续失败的策略以及难以取得实质性进展。我们的系统性分析表明，这一现象在多种小语言模型和任务领域中普遍存在。为解决这一问题，我们提出了进度感知蒸馏方法。

    arXiv:2610.19439v1 Announce Type: new  Abstract: While LLM-based agent systems have demonstrated remarkable proficiency in reasoning and solving complex real-world tasks, they also come with substantial token consumption and inference latency, limiting their practical deployment. To enhance the efficiency of agent systems, recent efforts have focused on transferring agentic capabilities from large-scale models to small language models (SLMs) via supervised fine-tuning-based distillation. Nevertheless, as interaction trajectories grow longer, involving extended reasoning-action chains and extensive environmental feedback, distilled student agents tend to experience progress stagnation: becoming trapped in unproductive loops characterized by persistent failed strategies and difficulty making meaningful progress. Our systematic analysis shows that this phenomenon consistently occurs across diverse SLMs and task domains. To address this issue, we propose Progress-Aware Distillation (ProD),
    
[^198]: 基于流形深度展开PCA的特征选择学习方法

    Learning to Select Features with Manifold Deep Unfolding PCA

    [https://arxiv.org/abs/2610.19436](https://arxiv.org/abs/2610.19436)

    本文提出MDUFS-Net，通过将ℓ_{2,0}范数融入PCA框架并将ADMM迭代算法深度展开为端到端可训练网络，结合流形优化策略实现参数自动学习和数值鲁棒性提升，兼顾了模型方法的轻量性与深度方法的自动化优势。

    

    无监督特征选择（UFS）在图像处理和模式识别中具有重要意义。然而，现有的基于模型的方法通常依赖手动调节的超参数，而基于深度学习的方法一般需要大量训练数据。为解决这些问题，本文提出了一种高效的流形深度展开特征选择网络（MDUFS-Net）。在技术上，我们首先将ℓ_{2,0}范数引入经典的主成分分析（PCA）框架，然后将基于交替方向乘子法（ADMM）的迭代算法展开为端到端可训练的网络，从而实现参数的自动学习。此外，我们还开发了一种流形优化策略以提高数值鲁棒性。在此基础上，我们进一步通过引入ℓ_0范数以获得更具判别性的特征表示，将MDUFS-Net扩展为增强变体MDUFS-Net+。

    arXiv:2610.19436v1 Announce Type: cross  Abstract: Unsupervised feature selection (UFS) is of great significance in image processing and pattern recognition. Nevertheless, existing model-based methods often rely on manually tuned hyperparameters, while deep learning-based methods generally require massive training data. To address these issues, this paper proposes an efficient manifold deep unfolding feature selection network (MDUFS-Net). Technically, we first incorporate the $\ell_{2,0}$-norm into the classic principal component analysis (PCA) framework, then unfold the alternating direction method of multipliers (ADMM)-based iterative algorithm into an end-to-end trainable network, thereby enabling automatic parameter learning. Moreover, a manifold optimization strategy is developed to improve the numerical robustness. On this basis, we further extend MDUFS-Net into an enhanced variant called MDUFS-Net+ by introducing the $\ell_{0}$-norm for more discriminative feature representation
    
[^199]: 自动驾驶中强化学习算法的综合比较

    Comprehensive Comparisons of Reinforcement Learning Algorithms in Autonomous Driving

    [https://arxiv.org/abs/2610.19429](https://arxiv.org/abs/2610.19429)

    本文在无信号交叉口的自动驾驶决策问题上，对FVI、RCPI、DQN和D-DDPG四种强化学习算法在最优性、安全性、训练稳定性和计算成本方面进行了全面比较，发现RCPI最接近参考解且训练最稳定但数据效率最低，而D-DDPG数据效率最高但稳定性最差。

    

    本文对四种具有代表性的强化学习方法在自动驾驶汽车（AV）决策中的应用进行了全面比较，包括：拟合值迭代、滚动分类策略迭代（RCPI）、深度Q网络（DQN）以及离散动作版的深度确定性策略梯度（D-DDPG）。这四种方法被应用于无信号交叉口的同一自动驾驶决策问题，并从最优性与安全性、训练稳定性以及训练与决策成本等方面进行了比较。结果表明：RCPI最接近参考解且训练过程最稳定，但其数据效率最低；D-DDPG与参考解的差距第二小且数据效率最高，但训练稳定性最差；DQN是最稳定的无模型方法，但碰撞率最高；FVI的碰撞率低于DQN，但数据效率较低。

    arXiv:2610.19429v1 Announce Type: new  Abstract: This paper presents a comprehensive comparison of four representative reinforcement learning (RL) methods in autonomous vehicle (AV) decision-making, including: Fitted Value Iteration (FVI), Rollout Classification Policy Iteration (RCPI), Deep Q-Network (DQN), and a discrete-action adaptation of Deep Deterministic Policy Gradient (D-DDPG). The four methods are applied to the same AV decision-making problem at an unsignalized intersection and are compared in terms of optimality and safety, training stability, and training and decision cost. RCPI is the closest to a reference solution and has the most stable training process. However, it is the least data-efficient. D-DDPG reaches the second-smallest gap to a reference solution with the highest data efficiency, but has the least training stability. DQN is the most stable model-free method but has the highest collision rate. FVI achieves a lower collision rate than DQN, but is less data-eff
    
[^200]: W-JEPA：面向可穿戴设备活动识别的小波联合嵌入预测架构

    W-JEPA: Wavelet Joint-Embedding Predictive Architecture for Wearable Activity Recognition

    [https://arxiv.org/abs/2610.19411](https://arxiv.org/abs/2610.19411)

    该论文提出W-JEPA自监督框架，通过将惯性运动信号组织成按尺度、传感器轴和时间索引的小波标记网格，并利用联合嵌入预测架构捕获与活动相关的多尺度细微变化，从而提升可穿戴活动识别的跨数据集迁移能力和对未见受试者的泛化能力。

    

    可穿戴设备活动识别面临的一个重大挑战是数据集之间的异构性，包括传感器放置位置、通道数量、采样率和受试者等方面的差异。此外，区分不同活动的运动模式通常表现为相似波形中细微的、尺度特定的变化。因此，原始信号往往具有模糊性，在原始信号上训练的模型倾向于依赖特定传感器配置或特定受试者的线索，这既限制了跨数据集迁移，也限制了对未见受试者的泛化能力。解决这一问题需要能够保留此类与活动相关的、尺度特定变化的表示方法。我们提出了W-JEPA，这是一种自监督框架，它利用惯性运动的多尺度结构，并采用联合嵌入预测架构，通过将时间窗口组织成由尺度、传感器轴和时间索引的小波标记（token）的固定网格。这些小波标记有助于揭示活动特定的……

    arXiv:2610.19411v1 Announce Type: new  Abstract: A major challenge in wearable device activity recognition is the heterogeneity across datasets in sensor placement, channel count, sampling rate and subjects. Moreover, the motion patterns that distinguish activities often appear as subtle, scale-specific variations within otherwise similar waveforms. As a result, raw signals are often ambiguous, and models trained on them tend to rely on cues specific to a particular sensor setup or participant, which limits both cross-dataset transfer and generalization to unseen participants. Addressing this requires representations that preserve such activity-relevant, scale-specific variation. We introduce W-JEPA, a self-supervised framework that exploits the multiscale structure of inertial motion and leverages a joint-embedding predictive architecture, by organizing windows into a fixed grid of wavelet tokens indexed by scale, sensor axis, and time. These wavelet tokens help expose activity-specif
    
[^201]: 通过并行性扩展大语言模型推理：吞吐量与延迟的权衡

    Scaling LLM Inference via Parallelism: Throughput-Latency Trade-off

    [https://arxiv.org/abs/2610.19405](https://arxiv.org/abs/2610.19405)

    提出了一个系统性框架来刻画LLM推理中张量并行、流水线并行及混合配置在吞吐量与延迟（TTFT和ITL）之间的权衡，以指导在不同工作负载特征和服务级别目标下选择最优并行配置。

    

    大语言模型（LLM）服务系统必须在有效利用分布式GPU资源的同时平衡延迟与吞吐量。这一挑战源于推理包含两个根本不同的阶段：预填充阶段决定首词元生成时间（TTFT），解码阶段决定词元间延迟（ITL）。张量并行（TP）、流水线并行（PP）以及混合配置的使用在这些延迟维度之间呈现出不同的权衡，使得最优并行配置高度依赖于工作负载特征、延迟需求和上下文长度。此外，尽管分布式推理已被广泛采用，但对于在不同工作负载条件和服务级别目标（SLO）下应如何选择并行配置，人们的理解仍然有限。在本文中，我们开发了一个系统性框架，通过吞吐量……

    arXiv:2610.19405v1 Announce Type: cross  Abstract: Large Language Model (LLM) serving systems must balance latency and throughput while efficiently utilizing distributed GPU resources. This challenge arises because inference consists of two fundamentally different phases: prefill, which determines time-to-first-token (TTFT), and decode, which determines inter-token latency (ITL). The use of tensor parallelism (TP), pipeline parallelism (PP), and hybrid configurations exposes different trade-offs between these latency dimensions, making the optimal parallelism configuration highly dependent on workload characteristics, latency requirements, and context length. Moreover, despite the widespread adoption of distributed inference, there is still limited understanding of how parallelism configurations should be selected under varying workload conditions and service-level objectives (SLO). In this paper, we develop a systematic framework for characterizing LLM inference parallelism via throug
    
[^202]: ActMOF：通过主动学习绘制网状化学合成图谱

    ActMOF: Mapping Reticular Synthesis through Active Learning

    [https://arxiv.org/abs/2610.19404](https://arxiv.org/abs/2610.19404)

    该论文提出了ActMOF——首个以实验数据为基础的铝-羧酸MOF合成主动学习基准，通过将实验测量的微波合成条件编码并赋予基于X射线衍射的结晶度指数，为比较高斯过程、树集成和大语言模型等搜索策略在绘制MOF结晶景观中的表现提供了标准化平台。

    

    网状化学使得金属有机框架（MOFs）能够从其构建单元出发进行设计，然而其合成仍受到实验数据稀疏、结晶结果含有噪声、以及缺乏用于比较主动学习策略的基准测试等因素的制约。尽管贝叶斯优化（BO）等数据驱动方法可以提高合成成功率，但如何事先选择合适的优化器来绘制MOF结晶景观在很大程度上仍然依赖经验。在本文中，我们提出了ActMOF，一个以实验为基础的基准测试平台，用于比较铝-羧酸MOF合成中的各类搜索策略。ActMOF基于实验测量的微波合成数据进行仿真建模，每个编码条件都被赋予一个基于衍射峰强度和半高全宽（FWHM）计算的结晶度指数。通过对由高斯过程（GP）、树集成模型和大语言模型（LLM）引导的策略进行基准测试，ActMOF……

    arXiv:2610.19404v1 Announce Type: cross  Abstract: Reticular chemistry allows metal-organic frameworks (MOFs) to be designed from their building units, yet their synthesis remains constrained by sparse experimental data, noisy crystallization outcomes, and the absence of benchmarks for comparing active-learning strategies. While data-driven approaches such as Bayesian optimization (BO) can improve synthesis success rates, selecting appropriate optimizers beforehand for mapping MOF crystallization landscapes remains largely empirical. Here we present ActMOF, an experiment-grounded benchmark for comparing search strategies in aluminum-carboxylate MOF synthesis. ActMOF is emulated from experimentally measured microwave syntheses with each encoded condition assigned a crystallinity index calculated based on diffraction peak intensity and full width at half-maximum. By benchmarking policies guided by Gaussian processes (GP), tree ensembles and large language models (LLM), ActMOF placed the 
    
[^203]: 训练即缝合：面向可互操作表示的拓扑保持正则化方法

    Train to Stitch: Topology-Preserving Regularization for Interoperable Representations

    [https://arxiv.org/abs/2610.19401](https://arxiv.org/abs/2610.19401)

    该论文提出一种拓扑保持正则化方法，无需训练期间的任何协调即可对齐独立训练的自编码器的潜在空间，从而实现表示层面的互操作性。

    

    随着大型且功能强大的预训练模型日益普及，人们越来越关注开发能够让不同神经网络无缝结合的方法，且这些网络之间几乎不需要或完全不需要直接通信。以基于自编码器的数据压缩为例：能够独立选择架构并分别训练编码器和解码器模型，同时仍确保它们能够协同工作，这将从根本上改变此类模型的构建、部署和更新方式。在这项工作中，我们提出了一种新颖的方法，利用拓扑正则化来对齐两个独立训练的自编码器模型的潜在空间，完全消除了训练过程中协调配合的需要。我们提供了条件性的理论分析，展示了我们的正则化器所促进的结构一致性如何能够转化为独立学习的表示之间的近似几何兼容性。重建实验……（摘要在此处截断）

    arXiv:2610.19401v1 Announce Type: new  Abstract: As large, powerful pre-trained models become more prevalent, there is growing interest in developing methods that allow different neural networks to be seamlessly combined with little or no direct communication between them. Take autoencoder-based data compression as an example: the ability to independently choose architectures and train encoder and decoder models, while still ensuring they work together, could fundamentally transform how such models are built, deployed, and updated. In this work, we introduce a novel approach that leverages topological regularization to align the latent spaces of two independently trained autoencoder models, entirely eliminating the need for coordination during training. We provide a conditional theoretical analysis showing how the structural agreement encouraged by our regularizer can translate into approximate geometric compatibility between independently learned representations. Reconstruction experi
    
[^204]: 基于MetaModelSelect的选择性下一位置预测：一种轻量级事后置信度元模型

    Selective Next-Location Prediction with MetaModelSelect: A Lightweight Post-Hoc Confidence Metamodel

    [https://arxiv.org/abs/2610.19394](https://arxiv.org/abs/2610.19394)

    提出轻量级事后置信度元模型MetaModelSelect，无需重训冻结的循环神经网络骨干即可判断其下一位置预测是否可靠，从而实现选择性蜂窝网络下一位置预测并减少资源浪费。

    

    下一位置预测可以支持主动式的移动网络决策，但基于不可靠的预测采取行动可能会浪费资源。我们将MetaModelSelect——一种轻量级的事后置信度元模型——应用于选择性蜂窝网络下一位置预测。该面向移动性场景的模型将预测位置的习得嵌入与四个最大的softmax概率以及Softmax集中度相结合。它仅利用冻结的循环神经网络的输出，无需重新训练骨干网络，即可学习该网络的top-1预测是否正确。在路径平衡的长春蜂窝移动轨迹数据上，我们评估了采用交叉熵（CE）训练以及采用自适应训练加熵最小化（SAT+EM）训练的骨干网络，并与八种标量置信度分数、两种位置感知查找方法以及一个原始logit多层感知机进行比较。相对于最大softmax概率，MetaModelSelect在每次运行的曲线下面积方面实现了平均降低（摘要在此处截断）。

    arXiv:2610.19394v1 Announce Type: new  Abstract: Next-location prediction can support proactive mobile-network decisions, but acting on unreliable predictions can waste resources. We apply MetaModelSelect, a lightweight post-hoc confidence metamodel, to selective cellular next-location prediction. The mobility-specific model combines a learned embedding of the predicted location with the four largest softmax probabilities and Softmax Concentration. It learns whether a frozen recurrent neural network's top-1 prediction is correct, using only its outputs and without retraining the backbone. On path-balanced Changchun cellular mobility trajectories, we evaluate backbones trained with cross-entropy (CE) and self-adaptive training plus entropy minimization (SAT+EM), comparing against eight scalar confidence scores, two location-aware lookup methods, and a raw-logit multilayer perceptron. Relative to maximum softmax probability, MetaModelSelect achieves mean per-run reductions in area under 
    
[^205]: 面向偏好学习的完全序贯高斯过程回归

    Fully sequential Gaussian process regression for preference learning

    [https://arxiv.org/abs/2610.19391](https://arxiv.org/abs/2610.19391)

    该论文针对目标函数只能通过带噪声成对比较来评估的连续全局优化问题，提出了基于矩匹配的完全序贯高斯过程近似方法DOVE，并首次为这类完全序贯近似贝叶斯推断建立了统计误差和期望简单遗憾的收敛速率保证。

    

    我们研究了一类连续全局优化问题，其目标函数无法被直接评估；相反，我们只能观测到在两个选定的设计点中哪一个（在有噪声的情况下）表现更好。这类成对比较会使高斯过程回归的使用变得复杂，因为其会产生非高斯后验。我们考虑一种基于矩匹配的完全序贯近似方法，该方法可以对元模型的均值和协方差实现快速的闭式更新。随后我们提出了DOVE，这是一种完全序贯采样策略，它将信息增益准则与旨在控制误差累积的随机采样相结合。在相当标准的正则性条件下，我们证明了依赖于核函数光滑性的统计误差和期望简单遗憾的收敛速率。据我们所知，这是连续全局优化领域中完全序贯近似贝叶斯推断的首个此类收敛速率结果。

    arXiv:2610.19391v1 Announce Type: new  Abstract: We study continuous global optimization problems in which the objective function cannot be directly evaluated; rather, we observe only which of two selected design points appears to perform better, subject to noise. Such comparisons complicate the use of Gaussian process regression because they yield non-Gaussian posteriors. We consider a fully sequential approximation, based on moment-matching, that yields a fast closed-form update of the metamodel's mean and covariance. We then propose DOVE, a fully sequential sampling policy that combines an information gain criterion with randomized sampling designed to control error accumulation. Under fairly standard regularity conditions, we prove convergence rates for both statistical error and expected simple regret that depend on kernel smoothness. To our knowledge, these are the first such rates for fully sequential approximate Bayesian inference in continuous global optimization.
    
[^206]: RA$^2$：基于保留锚定吸引的机器遗忘方法保持遗忘邻接效用

    RA$^2$: Retain-Anchored Attraction Preserves Forget-Adjacent Utility in LLM Unlearning

    [https://arxiv.org/abs/2610.19385](https://arxiv.org/abs/2610.19385)

    提出RA$^2$（保留锚定吸引）潜空间遗忘方法，通过将遗忘词元表示重定向至附近由保留数据支持的表示，在有效遗忘的同时保持被遗忘领域附近提示的良性推理效用。

    

    大语言模型（LLM）遗忘的评估通常主要依赖目标抑制能力和干净保留数据上的效用。这些指标可能漏掉一种局部失败模式：模型在标准保留提示上可能表现良好，但当提示位于被遗忘领域附近时，其良性推理能力会下降，这种现象被称为遗忘邻接效用损失。一个根本性的问题是：遗忘表示在脱离被移除的行为之后应该去向何处？成功的遗忘不仅取决于遗忘条件下的表示是否离开了有害计算，还取决于它们随后被重定向的方向。我们提出RA$^2$（保留锚定吸引），这是一种潜空间遗忘方法，它将遗忘词元状态重定向到附近由保留数据支持的表示，同时保持保留数据上的高层行为。在WMDP和MUSE基准上，RA$^2$获得了比强遗忘基线更高的遗忘邻接效用。

    arXiv:2610.19385v1 Announce Type: new  Abstract: Large Language Model (LLM) unlearning is often evaluated primarily by target suppression and clean retain utility. These metrics can miss a local failure mode: a model may perform well on standard retain prompts, yet degrade on benign reasoning when prompts lie near the forgotten domain. This phenomenon is termed forget-adjacent utility loss. A fundamental question is: where should forget representations go after they leave the removed behavior? Successful unlearning depends not only on whether forget-conditioned representations depart from the harmful computation, but also on the direction in which they are redirected afterward. We propose RA$^2$ (Retain-Anchored Attraction), a latent-space unlearning method that redirects forget-token states toward nearby retain-supported representations while preserving higher-layer behavior on retain data. On WMDP and MUSE, RA$^2$ yields higher forget-adjacent utility than strong unlearning baselines
    
[^207]: 学习中结构化优化网络的鲁棒解释性分析

    Robust Explanatory Analysis for Structured Optimization Networks (REASON) in Learning

    [https://arxiv.org/abs/2610.19378](https://arxiv.org/abs/2610.19378)

    受系统1/系统2范式启发，提出一种双循环机制，通过分层解决残差误差并主动获取额外证据，在分布偏移和扰动下以约一半的计算量持续优于单循环模型。

    

    我们寻求在复合学习目标下对残差误差进行分层解决，将其作为一种识别和利用额外信息的原则性机制，尤其是在分布偏移和扰动的情况下。受系统1/系统2范式的启发，我们将这一原理表述为服务于同一推理目标的双循环机制。一个快速的、类似系统1的循环负责解决现有证据可处理的残差；剩余的认知性残差则触发一个中等速度的、类似系统2的循环，该循环主动获取额外证据，并通过残差的减少来衡量其信息价值。在嵌套解空间上，可证明中等循环的最优解不劣于快速循环的最优解。实验表明，所得的双循环模型在约为中等循环一半的计算量下持续优于快速循环，包括在未见过的损坏情形下也表现良好，并且当提供更清晰的观测时（摘要在此处被截断）

    arXiv:2610.19378v1 Announce Type: new  Abstract: We seek a hierarchical resolution of residual error under a composite learning objective as a principled mechanism for identifying and exploiting additional information, particularly under distribution shift and perturbation. Inspired by the System 1/System 2 paradigm, we formulate this principle as a two-loop mechanism serving a common inference goal. A fast, System 1-like loop resolves the residual addressable by the available evidence; the remaining epistemic residual triggers a medium speed, System 2-like loop that actively acquires additional evidence and measures its informational value through residual reduction. Over nested solution spaces, the medium-loop optimum is provably no worse than the fast-loop optimum. Experiments show that the resulting two-loop model consistently improves upon the fast loop at approximately half the computation of the medium loop, including under unseen corruptions, and, when provided a cleaner observ
    
[^208]: 单步预测保真度是JEPA世界模型规划性能的糟糕代理指标

    One-Step Prediction Fidelity Is a Poor Proxy for Planning Performance in JEPA World Models

    [https://arxiv.org/abs/2610.19375](https://arxiv.org/abs/2610.19375)

    该论文提出一种以物理单位衡量单步误差并将其按来源分解的冻结解码器评估协议，通过实验证明在JEPA世界模型中，单步预测保真度并不能可靠预测目标条件规划性能，因此不应作为模型选择和检查点保存的代理指标。

    

    潜在世界模型通常基于单步预测损失进行选择、检查点保存和消融实验，其背后的假设是：能更好地预测下一个潜在表示的模型就是更好的模型。在基于模型的强化学习中，这一假设已被证明并不可靠；我们追问它在潜在世界模型中是否同样失效——在潜在世界模型中，训练目标是模型自身的嵌入表示，且损失并不以可解释的单位来表达。我们提出一种冻结解码器协议，以物理单位报告单步误差，并将其分解为两部分：已存在于编码器表示中的误差部分，以及预测器所增加的误差部分。我们将该协议应用于一个近期的端到端JEPA世界模型，在三个连续控制环境中，通过微调移动该指标，并在相同的检查点上测量目标条件规划性能，全程使用三个训练种子和配对的规划种子。在单个环境内，按训练轮次的阶梯实验显示，这种相关性在单个……之后趋于饱和（原文摘要在此处截断）

    arXiv:2610.19375v1 Announce Type: new  Abstract: Latent world models are routinely selected, checkpointed and ablated on one-step prediction loss, on the assumption that a model which predicts the next latent better is a better model. In model-based RL this assumption is known to be unreliable; we ask whether it also fails for latent world models, where the training target is the model's own embedding and the loss is not expressed in interpretable units. We introduce a frozen-decoder protocol reporting one-step error in physical units, decomposed into the part already present in the encoder's representation and the part the predictor adds. Applying it to a recent end-to-end JEPA world model across three continuous-control environments, we move the metric by fine-tuning and measure goal-conditioned planning on the same checkpoints, with three training seeds and paired planning seeds throughout. Within one environment, a per-epoch ladder shows the relationship saturating after a single e
    
[^209]: 策略验证自训练：从可验证策略奖励中学习智能体合规性

    Policy-Verified Self-Training: Learning Agent Compliance from Verifiable Policy Rewards

    [https://arxiv.org/abs/2610.19372](https://arxiv.org/abs/2610.19372)

    提出策略验证自训练（PVST）范式，将运行时策略验证器的决策转化为可验证的策略奖励，通过强化合规轨迹、以违规作为偏好优化的负样本，无需更强教师模型、人工偏好标签或攻击特定监督即可训练智能体的策略合规性。

    

    运行时策略验证器可以阻止智能体执行不被允许的操作，但其决策通常仅用于执行管控，随后便被丢弃。我们探究这些决策能否转而成为可验证的策略奖励，用于训练它们所守护的智能体。我们提出了策略验证自训练，这是一种后训练范式，其中智能体生成自身的交互轨迹，而一个可执行的策略验证器——此处为从书面策略编译而来的确定性门控——评估每个操作是否合规。合规轨迹被强化，违规则成为偏好优化的负样本；除了该门控和基准测试的任务成功信号外，PVST不需要更强的教师模型、不需要人工偏好标签、也不需要针对特定攻击的监督，尽管轨迹采样是在基准测试的受攻击环境中收集的。在提示注入下留出的AgentDojo任务上，策略验证偏

    arXiv:2610.19372v1 Announce Type: new  Abstract: Runtime policy verifiers can prevent agents from executing disallowed actions, but their decisions are typically used only for enforcement and discarded afterward. We ask whether these decisions can instead become verifiable policy rewards for training the agents they guard. We introduce Policy-Verified Self-Training (PVST), a post-training paradigm in which an agent generates its own interaction trajectories and an executable policy verifier, here a deterministic gate compiled from a written policy, evaluates whether each action complies. Compliant trajectories are reinforced, and violations become negatives for preference optimization; beyond the gate and the benchmark's task-success signal, PVST needs no stronger teacher model, no human preference labels and no attack-specific supervision, although rollouts are collected in the benchmark's attacked environment. On held-out AgentDojo tasks under prompt injection, policy-verified prefer
    
[^210]: 检测器架构、模型规模与场景密度对多行人跟踪的影响

    The Impact of Detector Architecture, Model Scale, and Scene Density on Multi-Pedestrian Tracking

    [https://arxiv.org/abs/2610.19369](https://arxiv.org/abs/2610.19369)

    本研究系统评估了检测器架构、模型规模、场景密度和推理方式对多行人跟踪的影响，发现RT-DETR在跟踪质量上表现最佳（尤其在拥挤的MOT20场景），而YOLO26-L配合NMS在质量与效率之间达到最佳平衡。

    

    视频中的行人跟踪对于监控和场景分析等应用非常重要，但在遮挡频繁、运动快速的拥挤场景中变得更具挑战性。本文分析了检测器架构、模型规模、场景密度和推理方式如何影响多行人跟踪。我们在MOT17、MOT20和SOMPT22基准数据集的视频序列上，将YOLO11、RT-DETR和YOLO26检测器家族分别与ByteTrack和BoT-SORT跟踪器进行了组合比较。对于YOLO26，我们还比较了基于非极大值抑制（NMS）的传统重叠过滤方式与端到端推理方式。结果表明，RT-DETR检测器家族在所有三个基准上与两种跟踪器结合时，在HOTA跟踪质量指标上均取得了最高的平均分数，且在MOT20上优势更为明显，但其计算成本也更高。在MOT17和SOMPT22上，YOLO26变体，尤其是使用NMS的YOLO26-L，在跟踪质量与效率之间提供了有利的组合。

    arXiv:2610.19369v1 Announce Type: cross  Abstract: Pedestrian tracking in video is important for applications such as surveillance and scene analysis, but it becomes more challenging in crowded scenarios with frequent occlusions and rapid motion. This paper analyzes how detector architecture, model scale, scene density, and inference regime affect the tracking of multiple pedestrians. We compare the YOLO11, RT-DETR, and YOLO26 detector families with the ByteTrack and BoT-SORT trackers on sequences from the MOT17, MOT20, and SOMPT22 benchmarks. For YOLO26, we also compare traditional overlap filtering via Non-Maximum Suppression (NMS) with end-to-end inference. The RT-DETR family achieved the highest mean scores on the HOTA tracking quality metric across all three benchmarks with both trackers, with a more pronounced advantage on MOT20, albeit at a higher computational cost. On MOT17 and SOMPT22, YOLO26 variants, especially YOLO26-L with NMS, offered favorable combinations of quality an
    
[^211]: 动态图世界建模：从时间上下文中学习未来动态

    Dynamic Graph World Modelling: Learning Future Dynamics from Temporal Context

    [https://arxiv.org/abs/2610.19364](https://arxiv.org/abs/2610.19364)

    提出首个动态图世界建模框架DyG-WM，通过回答“什么演化了、如何演化、接下来将演化什么”三个问题，从时间上下文中联合预测未来的节点与关系表示。

    

    动态图学习通过从历史观测中对实体及其关系的演化进行建模，以预测未来的图状态。现有方法尚未充分解决动态图世界建模问题，即需要学习从时间上下文到未来潜在图状态的转移。这一任务的难点主要有三：节点和关系依赖不同的历史证据，结构演化具有路径依赖性，且未来快照提供的监督目标稀疏并依赖于具体实现。我们提出动态图世界建模，这是首个通过回答三个问题——什么发生了演化、如何发生演化、接下来将演化什么——来从时间上下文中联合预测未来节点与关系表示的框架。为了捕捉“什么发生了演化”，历史节点-关系上下文编码器总结了节点邻域轨迹和节点对特定的历史；为了刻画“如何发生演化”，节点-关系路径签名编码器……

    arXiv:2610.19364v1 Announce Type: new  Abstract: Dynamic graph learning models how entities and their relations evolve from historical observations to predict future graph states. Existing methods do not fully address dynamic graph world modelling, which requires learning transitions from temporal context to future latent graph states. This is difficult for three reasons: nodes and relations rely on different historical evidence, structural evolution is path-dependent, and future snapshots provide sparse and realization-specific targets. We propose Dynamic Graph World Modelling (DyG-WM), the first framework to jointly predict future node and relation representations from temporal context by answering three questions: what has evolved, how it evolved, and what will evolve next. To capture what has evolved, a historical node-relation context encoder summarizes node-neighborhood trajectories and pair-specific histories. To characterize how it evolved, a node-relation path-signature encode
    
[^212]: 尾部竞争解释了可验证问题中 Best-of-N 采样的缩放规律

    Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems

    [https://arxiv.org/abs/2610.19362](https://arxiv.org/abs/2610.19362)

    该论文将 Best-of-N 采样建模为正确与错误响应极端奖励尾部之间的竞争，利用极值理论刻画出正确尾部主导、错误尾部主导和临界三种渐近状态，统一解释了缩放增益与奖励作弊导致的性能退化，并推导了有限样本缩放定律。

    

    Best-of-N 采样是大语言模型一种简单的推理时扩展策略，然而其在不完整奖励模型下的行为仍鲜为人知。我们将 Best-of-N 研究为正确响应与错误响应的极端奖励尾部之间的竞争。这一视角表明，成功不仅取决于采样到正确响应的概率，还取决于正确响应是否在上尾部中占优于错误响应。利用极值理论，我们刻画了三种渐近状态：正确尾部主导、错误尾部主导和临界状态，从而同时解释了单调增益和奖励作弊导致的性能退化。我们进一步针对代表性奖励分布推导了有限样本缩放定律，表明在高斯奖励下，方差足够大的错误响应即使平均奖励更低，也可能在选择过程中占据主导。与理想选择不同，使用不完整选择器时的收敛（原文摘要在此截断）

    arXiv:2610.19362v1 Announce Type: new  Abstract: Best-of-N sampling is a simple inference-time scaling strategy for large language models, yet its behavior under incomplete reward models remains poorly understood. We study Best-of-N as a competition between the extreme reward tails of correct and incorrect responses. This view shows that success depends not only on the probability of sampling a correct response, but also on whether correct responses dominate incorrect ones in the upper tail. Using extreme value theory, we characterize three asymptotic regimes: correct-tail-dominant, incorrect-tail-dominant, and critical explaining both monotonic gains and reward-hacking-induced degradation. We further derive finite-sample scaling laws for representative reward distributions, showing that, under Gaussian rewards, incorrect responses with sufficiently large variance can dominate selection even when their mean reward is lower. Unlike oracle selection, convergence with an incomplete select
    
[^213]: 竞争伦理框架的转向向量可互换：一种用于激活转向的竞争目标测试

    Steering Vectors for Competing Ethical Frameworks Are Interchangeable: A Competing-Target Test for Activation Steering

    [https://arxiv.org/abs/2610.19359](https://arxiv.org/abs/2610.19359)

    论文提出“竞争目标测试”这一新方法，发现用于康德义务论与行为功利主义的激活转向向量共享大部分方向且可互换，说明转向向量可能通过共享组件起作用而非直接控制目标行为。

    

    激活转向通过向模型的隐藏状态添加一个向量来控制某个指定的行为。然而，该行为上的增益并不能证明向量是直接控制它的：向量可能通过与相邻行为共享的组件起作用，而未转向基线和随机基线无法区分这种情况。我们引入了竞争目标测试：将一个向量与以相同方式为竞争目标构建的向量进行比较，并通过等效性检验判定二者的差异何时低于有效边界。我们在开源权重模型中，将该测试应用于康德义务论（根据行为所遵循的规则来判断行为）和行为功利主义（根据行为产生的福祉来判断行为）的转向向量。在留出的情境上，通过提示施加某个框架会提高该框架的得分并降低另一框架的得分，而两个框架各自的转向向量却以相似的方式同时移动两者的得分，且没有任何一个达到等效边界。这两个向量共享其大部分方向，即“应用某个框架”的请求……

    arXiv:2610.19359v1 Announce Type: cross  Abstract: Activation steering adds a vector to a model's hidden states to control a named behaviour. A gain on that behaviour, however, does not show that the vector controls it directly: it may act through a component shared with neighbouring behaviours, which unsteered and random baselines cannot separate. We introduce the competing-target test: a vector is compared with the vector built the same way for a competing target, and an equivalence test states when their difference is below a useful margin. We apply it in open-weight models to vectors for Kantian deontology, which judges an act by the rule it follows, and act utilitarianism, which judges it by the welfare it produces. On held-out situations, prompting a framework raises its score and lowers the other's, whereas vectors for either framework move both scores alike, and none reaches the margin. The two vectors share most of their direction, a request to apply some framework; the same c
    
[^214]: Transformer的渐近行为

    Asymptotic Behavior of Transformers

    [https://arxiv.org/abs/2610.19358](https://arxiv.org/abs/2610.19358)

    本文证明了一类包含层归一化、MLP和RoPE编码的Transformer在softmax温度超过各自阈值时，其经验缓存分布会收敛到与初始条件无关的唯一极限，但由于这组阈值的上确界为无穷大，不存在能强制所有Transformer收敛的全局最小温度。

    

    本文研究了Transformer网络在长时间生成文本时的行为。我们证明，即使是一个仅有三个词元、单层的纯注意力Transformer也可能无法收敛。随后，我们研究了一类丰富的Transformer模型，它们具有任意有限深度、层归一化、多层感知机（MLP）、RoPE编码以及局部层与全局层的任意混合。该类中的每个Transformer都存在一个softmax温度阈值，当温度超过该阈值时，其经验缓存分布会收敛到一个唯一的极限，且该极限与初始条件无关。我们进一步证明，尽管每个Transformer都拥有自己的温度阈值，但这组阈值的上确界是无穷大的，因此不存在一个能够强制所有Transformer都收敛的全局最小温度。

    arXiv:2610.19358v1 Announce Type: new  Abstract: In this paper we study the behavior of transformer networks generating text over long periods of time. We show that even a three-token, single-layer attention-only transformer can fail to converge. We then study a rich class of transformers with arbitrary finite depth, layer normalization, MLPs, RoPE encodings, and an arbitrary mix of local and global layers. Every transformer in this class has a softmax temperature threshold above which its empirical cache distribution converges to a unique limit, independent of initial conditions. We show further that, while each transformer has a temperature threshold, the supremum of this set is infinite, so there is no global minimal temperature that forces all transformers to converge.
    
[^215]: BFHD：面向异构临床评估的双向特征调和分解

    BFHD: Bidirectional Feature Harmonization Decomposition for Heterogeneous Clinical Assessments

    [https://arxiv.org/abs/2610.19352](https://arxiv.org/abs/2610.19352)

    该论文提出 BFHD 框架，将异构临床评估系统间的调和问题建模为双向可恢复性问题，通过双向耦合转换和特征级输出门控，在原始测量空间中显式区分可双向可靠转换的测量指标与不可转换成分。

    

    临床评估通常在不同中心和不同时间使用异构的评估系统进行采集，导致记录中混合了不同但相互关联的测量指标集合。这促使我们需要超越总分对接层面的调和（harmonization）方法。本文将测量层面的临床调和形式化为一个双向可恢复性问题：给定来自两个评估系统的配对观测数据，目标是识别哪些测量指标能够在应用定义的容差范围内被可靠地双向转换，同时分离出不可转换的成分。我们提出双向特征调和分解（Bidirectional Feature Harmonization Decomposition, BFHD），这是一个以可行性为驱动的框架，它强制执行双向耦合的转换，并利用特征级输出门控在原始测量空间中产生显式的分解结果。在合成数据与真实临床评估配对上的实验表明，BFHD 实现了更广泛的可行调和覆盖范围。

    arXiv:2610.19352v1 Announce Type: new  Abstract: Clinical assessments are often collected using heterogeneous assessment systems across centers and time, leading to records that mix different but related sets of measurements. This motivates harmonization beyond total-score linking. We formulate clinical harmonization at the measurement level as a bidirectional recoverability problem: given paired observations from two assessment systems, the goal is to identify which measurements can be reliably translated in both directions within an application-defined tolerance, while separating non-translatable components. We propose Bidirectional Feature Harmonization Decomposition (BFHD), a feasibility-driven framework that enforces bidirectionally coupled translation and uses feature-wise output gating to produce an explicit decomposition in the original measurement space. Experiments on synthetic data and real clinical assessment pairs show that BFHD achieves broader feasible harmonization cove
    
[^216]: 从哪里开始，如何移动：分析长程LLM智能体在GRPO下的课程

    Where to Start, How to Move: Analyzing Curricula under GRPO for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2610.19349](https://arxiv.org/abs/2610.19349)

    该研究系统分析了引导辅助GRPO课程设计，发现在固定回合级阈值规则下“从接近完成的阶段开始”最有效，且基于结果信息的工具调用选择在已训练和未训练任务类别上均优于标准GRPO。

    

    当所有采样（rollouts）获得相同的奖励时，组相对策略优化（GRPO）无法提供基于奖励差异的学习信号。在需要多次交互的长程任务中，奖励只在成功完成任务后才给出，这使得训练早期“全部失败”的组非常常见，限制了模型练习后续步骤的机会。引导辅助GRPO（Guide Assisted GRPO）在学习者继续之前重放一段成功的参考前缀。我们研究了初始位置、交接单元和移动规则如何影响无辅助情况下的任务完成率。在BFCL上跨三种模型规模的实验表明，在固定的回合级阈值规则下，从接近完成的阶段开始更为有效。更细粒度的交接本身并不能持续带来帮助，但基于结果信息的工具调用选择在已训练和未训练的任务类别上都优于GRPO。引导式开始也提升了AppWorld在两个评估集上的成功率。这些发现强调了学习从哪里开始以及起始位置如何移动的重要性。

    arXiv:2610.19349v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) provides no reward-relative learning signal when all rollouts receive identical rewards. In long-horizon tasks that require multiple interactions, rewards are given only upon successful task completion, making all-fail groups common early in training and limiting opportunities to practice later steps. Guide Assisted GRPO replays a successful reference prefix before the learner continues. We study how initial positions, handoff units, and movement rules affect unassisted task completion. Across three model sizes on BFCL, starting near completion is more effective under a fixed turn-level threshold rule. Finer handoffs alone do not consistently help, but outcome-informed tool-call selection outperforms GRPO on trained and untrained task categories. Guided starts also improve AppWorld success on both evaluation splits. These findings highlight where learning starts and how starting positions move 
    
[^217]: 集成新颖性检测系统：从传感器网络到多新颖性计算机视觉

    Integrated Novelty Detection Systems: From Sensor Networks to Multi-Novelty Computer Vision

    [https://arxiv.org/abs/2610.19347](https://arxiv.org/abs/2610.19347)

    本文针对气象传感器网络中的雨量计堵塞检测问题，评估并揭示了四种现有质量控制方法的局限性，特别是它们在检测成片故障传感器集群时的不足。

    

    新颖性检测在众多技术和科学领域中都至关重要，其重要性涵盖从确保传感器网络的可靠性到增强计算机视觉系统的适应性。随着人工智能系统日益精密和复杂，有效的异常检测对于维持系统的可靠性和准确性变得不可或缺。本工作研究了气象传感器网络和计算机视觉中的新颖性检测问题。在气象传感器网络中，特别是在降雨监测方面，检测被堵塞的雨量计会显著影响数据可靠性、粮食安全和灾害管理。本工作评估了四种现有的用于检测堵塞雨量计的质量控制方法，并展示了它们的局限性，尤其是在检测成片故障传感器集群方面的不足。通过比较条件混合模型（CMM）、PSWQ（Fz-filter）、核回归和高斯方法，我们识别出了这些方法的缺陷。

    arXiv:2610.19347v1 Announce Type: new  Abstract: Novelty detection is crucial in various technological and scientific domains. Its importance spans from ensuring the reliability of sensor networks to enhancing the adaptability of computer vision systems. As AI systems become increasingly sophisticated and complex, effective anomaly detection is essential for maintaining reliability and accuracy. This work addresses novelty detection in weather sensor networks and computer vision. In weather sensor networks, particularly for monitoring rainfall, detecting blocked rain gauges can significantly impact data reliability, food security, and disaster management. This work evaluates four existing quality control methods for detecting blocked rain gauges and demonstrates their limitations, particularly in detecting clusters of faulty sensors. By comparing methods such as the Conditional Mixture Model (CMM), PSWQ (Fz-filter), Kernel Regression, and Gaussian Method, we identify their shortcomings
    
[^218]: 目标词元不对称性可能伪装成深度专门化

    Target-Token Asymmetry Can Masquerade As Depth Specialization

    [https://arxiv.org/abs/2610.19344](https://arxiv.org/abs/2610.19344)

    层级消融研究中所观察到的“早期层专门化”现象，可能仅仅是两个语料库之间目标词元分布不对称（如数学文本中大量空白字符）所造成的假象，而非真正的深度专门化。

    

    层级消融研究通常会扰动语言模型的早期或晚期层，比较其对两个语料库预测造成的损害，并将某一语料库上更大的早期层效应解读为该模型对该任务的早期层专门化。这种比较会对各语料库要求模型预测的词元取平均，而这些目标词元是不同的：例如，数学解答比百科全书散文需要多得多的空白字符和数字。我们证明这种差异可以制造出深度专门化的假象。我们按目标词元类型拆分了跨语料库的早晚期损伤交互作用，并在两个语料库共享的目标词元混合下重新计算。在GSM8K和Wikitext之间，正交互作用在五个模型实例中均得到复现。在三个完全分解的模型中，只有空白字符具有正交互作用。移除空白字符在所有三个模型中反转了汇总效应，而匹配完整的目标词元混合则在两个模型中反转了该效应。

    arXiv:2610.19344v1 Announce Type: new  Abstract: Layer-ablation studies often perturb early or late layers of a language model, compare the damage to prediction on two corpora, and read a larger early-layer effect on one corpus as early-layer specialization for its task. Such a comparison averages over the tokens that each corpus asks the model to predict, and these target tokens differ: math solutions, for example, ask for far more whitespace and numbers than encyclopedia prose. We show that this difference can create the appearance of depth specialization. We split a cross-corpus early--late damage interaction by target-token type and recompute it under a target mixture shared by both corpora. Between GSM8K and Wikitext, a positive interaction replicates across five model instances. In the three fully decomposed models, only whitespace has a positive interaction. Removing whitespace reverses the pooled effect in all three, and matching the full target mixture reverses it in two. The 
    
[^219]: 反事实门控支持分解：在训练数据过滤中分离选择与重分配效应

    Counterfactual Gate-Support Decomposition: Separating Selection from Redistribution in Training-Data Filtering

    [https://arxiv.org/abs/2610.19343](https://arxiv.org/abs/2610.19343)

    该论文提出反事实门控支持分解（CGSD），一种四臂预算匹配的实验设计，可将训练数据质量门控的整体效应分解为支持内选择、支持间重分配及其交互作用，并在韩英与韩日检索任务中发现支持内选择是唯一显著超出校准零区间的分量。

    

    训练数据质量门控会同时改变两个方面的结果：输入空间各区域内哪些样本得以保留，以及训练数据总量在这些区域之间如何分布，因此单纯的“门控与随机”对比无法将这两种机制分离开来。本文提出了反事实门控支持分解方法，这是一种四臂、预算匹配的实验设计，通过将门控资格与支持分布进行交叉组合，把观察到的门控效应分解为支持内选择、支持间重分配以及二者的交互作用。该方法面临的实际约束是有限池可实现性：作者刻画了在何种质量和词元组装条件下才能构建所需的实验臂，并讨论了支持消亡的情形——在此情况下常规设计无法定义。在韩英和韩日检索任务上的实验表明，支持内选择分量的均值是唯一超出其校准零区间的分量（+29.7 和 +5.6 个保留集条目；30/30 次重复均为正向）。

    arXiv:2610.19343v1 Announce Type: cross  Abstract: A training-data quality gate changes both which examples survive within regions of the input space and how training mass is distributed across those regions, so a gate-versus-random comparison does not isolate these mechanisms. We introduce Counterfactual Gate-Support Decomposition (CGSD), a four-arm, budget-matched design that crosses gate eligibility with support distribution and decomposes the observed gate effect into within-support selection, between-support redistribution, and their interaction. The practical constraint is finite-pool realizability: we characterize mass and token-assembly conditions under which the required arms can be constructed, together with a support-extinction case in which the ordinary design is undefined. On Korean-English and Korean-Japanese retrieval, the mean within-support selection component is the only one that exceeds its calibrated null band (+29.7 and +5.6 holdout items; 30/30 positive replicates
    
[^220]: 为什么图Transformer不会将注意力集中于同一个token？

    Why don't Graph Transformers attend to the same token?

    [https://arxiv.org/abs/2610.19324](https://arxiv.org/abs/2610.19324)

    本文揭示了大语言模型中的“注意力汇聚”现象在图Transformer中罕见的原因——图的对称性与局部性在理论上可限制注意力集中，而对使用全局注意力的分子任务，真正阻止汇聚的是GRIT的门控注意力评分函数，将其替换为点积后汇聚率从0.09飙升至0.68。

    

    在大语言模型（LLM）中，注意力高度集中于第一个token，这种“注意力汇聚”现象被认为可以保护深层模型免受过度混合的影响。尽管图Transformer面临着类似的结构性压力，我们发现最先进的图Transformer GRIT在分子数据上很少形成注意力汇聚，且在更大的图上训练时汇聚程度反而更低，这与LLM中的趋势恰好相反。从原理上讲，图结构本身可以防止注意力汇聚：我们从理论和实验上证明了图的对称性和局部性会限制任何节点所能接收的注意力。然而，对于具有可区分原子和全局注意力的分子而言，这两个限制都无法阻止汇聚；真正阻止汇聚的因素在于模型内部。我们发现注意力评分函数是主要原因：将GRIT的门控logit替换为点积后，在相同的图上其汇聚率从0.09升至0.68。在使用点积注意力的图Transformer中，移除汇聚token所产生的误差与移除一个随机token大致相同。

    arXiv:2610.19324v1 Announce Type: new  Abstract: In large language models (LLMs), attention concentrates heavily on the first token. This 'attention sink' is believed to protect deep models against over-mixing. Even though graph transformers face a related pressure, we find that a SoTA graph transformer, GRIT, rarely forms sinks on molecules, and that it sinks less when trained on larger graphs, the opposite of the trend in LLMs. In principle, a graph can prevent sinks: we prove and show empirically that its symmetry and locality limit the attention any node can receive. On molecules, however, with distinguishable atoms and global attention, neither limit prevents sinks; what stops them lies in the model. We find that the attention scoring function is the main cause. Replacing GRIT's gated logit with a dot product raises its sink rate from 0.09 to 0.68 on identical graphs. In Graph Transformers using dot-product attention, removing the sink gives about the same error as removing a rand
    
[^221]: TopicMerge：一种分而治之的主题建模方法

    TopicMerge: A Divide and Conquer Approach to Topic Modeling

    [https://arxiv.org/abs/2610.19323](https://arxiv.org/abs/2610.19323)

    TopicMerge提出一种分治式的多阶段流水线，先通过高效聚类将文档过度分割为语义纯净的微聚类，再由LLM结合局部邻域上下文迭代仲裁合并候选对，从而同时实现了最先进的聚类性能与计算效率。

    

    主题建模领域的最新进展分化为两条主要路径：高效的基于嵌入的方法具有良好的可扩展性，但在标准聚类指标上往往表现不佳；而大语言模型（LLM）方法虽然实现了最先进的聚类性能，但代价是高昂的计算开销和较差的可扩展性。本文提出了TopicMerge，这是一种新颖的多阶段流水线，它将上述两种方法结合起来，以同时实现最先进的聚类性能和计算效率。TopicMerge首先采用高效的聚类流水线，有意识地将文档语料库过度分割为大量细粒度、语义纯净的微聚类，并为这些微聚类赋予初始语义标签。随后，它引入了一种新颖的迭代合并算法，由LLM在局部邻域上下文的引导下，对候选合并对进行仲裁决策。这些被批准的合并随后被有选择地……

    arXiv:2610.19323v1 Announce Type: new  Abstract: Recent advancements in topic modeling have split into two main paths: efficient embedding-based methods that offer scalability but are often outperformed on standard clustering metrics, and Large Language Model (LLM) approaches that achieve state-of-the-art clustering performance but at the cost of prohibitive computational expense and poor scalability. This paper introduces TopicMerge, a novel multi-stage pipeline that bridges these approaches to achieve both state-of-the-art clustering performance and computational efficiency. TopicMerge first employs an efficient clustering pipeline to deliberately over-segment a document corpus into a large set of fine-grained, semantically pure micro-clusters, which are then given initial semantic labels. It then introduces a novel, iterative merging algorithm where an LLM, guided by local neighborhood context, arbitrates on consolidating candidate pairs. These approved merges are then selectively e
    
[^222]: 面向顺序筛选的历史特征残差自适应

    Historical-Feature Residual Adaptation for Sequential Screening

    [https://arxiv.org/abs/2610.19315](https://arxiv.org/abs/2610.19315)

    该论文提出一种利用固定历史特征构建、锚定于反馈揭示前基础分数的采集时刻残差自适应方法，在保留预训练模型的前提下将顺序筛选的命中回收率从22.58%提升至28.61%，超越了所有六个预先指定的对比方法。

    

    顺序筛选策略必须在固定的实验预算下有效利用已获取的反馈。我们研究了采集时刻的残差自适应：一种由固定历史特征构建的小型排序校正，并锚定于在反馈揭示之前记录的基础分数。该接口接受一个以历史为条件的打分器和一个候选特征表；其主要实例化方式是在保留预训练模型的同时，用64个历史方向来扩充AssayFormer的十维表示。在来自15个记录来源的20个官方测试筛选任务的回顾性评估中，一个在接触目标结果之前就已固定的对比设置将按来源宏平均的命中回收率从22.58%提升至28.61%，超过了全部六个预先指定的对比方法。使用相同特征的普通排序头达到了25.73%。补充的强逻辑回归、排序融合和未投影对照分别达到了26.42%、26.02%和26.78%。另一个模型独立（摘要在此处截断）

    arXiv:2610.19315v1 Announce Type: new  Abstract: Sequential screening policies must use acquired feedback effectively under a fixed experiment budget. We study acquisition-time residual adaptation: a small ranking correction built from fixed historical features and anchored to base scores recorded before feedback is revealed. The interface accepts a history-conditioned scorer and a candidate feature table; its primary instantiation augments AssayFormer's ten-dimensional representation with 64 historical directions while retaining the pretrained model. In retrospective evaluation on 20 official-test screens from 15 recorded sources, a comparison fixed before target-outcome access increased source-macro fraction of hits recovered from 22.58% to 28.61%, exceeding all six prespecified comparators. A same-feature ordinary ranking head achieved 25.73%. Supplementary strong logistic, rank-fusion, and unprojected controls achieved 26.42%, 26.02%, and 26.78%, respectively. A separate model-inde
    
[^223]: 大语言模型回答了问题吗？一种用于越狱评估的语义重构游戏

    Did the LLM Answer the Question? A Semantic Reconstruction Game for Jailbreak Evaluation

    [https://arxiv.org/abs/2610.19306](https://arxiv.org/abs/2610.19306)

    该论文提出了一种无需训练的“语义重构游戏”评估方法，通过让LLM从回复中重构其实际回答的请求并与诱饵请求进行受限语义比较来判断越狱是否成功，从而避免将拒绝或仅主题相关的回复误判为成功攻击，大幅提升攻击成功率评估的准确性（不同评判模型间ASR差异可达82个百分点）。

    

    针对大语言模型（LLM）的越狱攻击通常使用攻击成功率（ASR）进行评估，而这依赖于正确识别出那些真正满足禁止请求的回复。现有的评估器可能将拒绝类回复或仅在主题上相关的回复误判为成功的攻击。我们证明，仅更换评判所用的LLM就可能使报告的ASR变化高达82个百分点。我们提出了语义重构游戏，这是一种无需训练的评估器，它将合规性判断分解为受约束的语义比较。在仅给定一个回复的情况下，由LLM根据其实质性内容重构出该回复所回答的请求，并排除其中仅仅是重复或拒绝的信息。该游戏将这一重构结果与相似的诱饵请求进行比较，以判断其是否与原始请求匹配，并在两个候选请求都不匹配时允许弃权。在三个现成的LLM评判器上，该游戏相对于人工标注取得了最低的平均错误率……

    arXiv:2610.19306v1 Announce Type: new  Abstract: Jailbreak attacks against large language models (LLMs) are evaluated using attack success rate (ASR), which depends on correctly identifying responses that fulfill prohibited requests. Existing evaluators can mistake refusals or topically related responses for successful attacks. We show that changing only the judge LLM can alter reported ASR by up to $82$ percentage points. We introduce the semantic reconstruction game, a training-free evaluator that decomposes compliance judgments into constrained semantic comparisons. Given only a response, an LLM reconstructs the request answered by its substantive content, excluding information merely repeated or refused. The game compares this reconstruction against similar decoy requests to determine whether it matches the original request, allowing abstention when neither candidate matches. Across three off-the-shelf LLM judges, the game achieves the lowest mean error against human labels among e
    
[^224]: DTDD：面向可靠在策略监督的发散触发动态蒸馏

    DTDD: Divergence-Triggered Dynamic Distillation for Reliable On-Policy Supervision

    [https://arxiv.org/abs/2610.19302](https://arxiv.org/abs/2610.19302)

    提出发散触发动态蒸馏（DTDD）框架，通过监测分段级师生分歧自动触发教师接管并生成恢复后缀供学生行为克隆学习，从而在长程推理、无SFT的基础初始化蒸馏和多轮智能体场景中保障在策略蒸馏监督的可靠性。

    

    在策略蒸馏（OPD）在学生模型的 rollout 轨迹上提供密集的 token 级监督，但当学生与教师模型出现分歧时，这种监督可能变得不可靠——分歧可能源于较大的初始策略差距，或长程推理与工具使用中的误差累积。我们提出了发散触发动态蒸馏（DTDD），这是一种自适应的 OPD 框架，利用分段级别的师生分歧在训练过程中触发教师接管。当分歧在推理过程中触发干预时，学生的前缀部分继续接受标准的 OPD 监督；随后教师生成一个恢复后缀，学生通过行为克隆学习该后缀，即在教师生成的 token 上使用交叉熵损失。我们在三种场景下评估 DTDD：长程推理、从基础初始化出发且无需离线监督微调（SFT）的蒸馏，以及多轮智能体。我们还比较了其他触发器设计的（效果）……

    arXiv:2610.19302v1 Announce Type: new  Abstract: On-policy distillation (OPD) provides dense token-level supervision on student rollouts, but this supervision can become unreliable when the student and teacher diverge, either because of a large initial policy gap or compounding errors in long-horizon reasoning and tool use. We introduce Divergence-Triggered Dynamic Distillation (DTDD), an adaptive OPD framework that uses segment-level student-teacher divergence to trigger teacher takeover during training. When divergence triggers an intervention during reasoning, the student prefix continues to receive standard OPD supervision. The teacher then generates a recovery suffix, which the student learns through behavior cloning, using cross-entropy on the teacher-generated tokens. We evaluate DTDD in three regimes: long-horizon reasoning, distillation from a base initialization without offline supervised fine-tuning (SFT), and multi-turn agents. We also compare alternative trigger designs an
    
[^225]: 基于验证器的智能体强化学习实现可证明正确的代码

    Verifier-Grounded Agentic RL for Provably Correct Code

    [https://arxiv.org/abs/2610.19298](https://arxiv.org/abs/2610.19298)

    该论文提出了一种读取验证器诊断信息（而非简单的接受/拒绝判定）的智能体强化学习奖励机制，结合加权证明义务完成度评分、参考证明隐藏探针以及防止篡改规范的完整性审计，解决了验证器奖励稀疏和策略作弊两大难题，从而训练出可证明正确的代码。

    

    面向代码的强化学习通常以通过测试作为奖励，但模型即使通过了测试，在测试未覆盖的输入上仍可能是错误的。程序验证器则可以判定一个程序是否在所有输入上都满足形式化规范。语言模型几乎总是只在推理阶段与验证器配对使用，而将验证器转化为奖励信号面临两大困难：其一，验证器只报告接受或拒绝，当所有尝试都失败时无法提供梯度信号；其二，策略可以欺骗验证器，最简单的方式就是让验证器直接假设本应被证明的内容。我们构建了一个智能体环境和奖励机制来同时解决这两个问题。该奖励读取验证器的诊断信息而非其判定结果，通过候选方案所完成的证明义务的加权比例进行评分，并利用参考证明注释中的隐藏探针进行扩展。完整性审计的优先级高于验证器，会拒绝任何篡改规范的候选方案。

    arXiv:2610.19298v1 Announce Type: new  Abstract: Reinforcement learning for code usually rewards passing tests, which a model can do while remaining wrong on the inputs the tests do not cover. A program verifier instead decides whether a program satisfies a formal specification for all inputs. Language models are almost always paired with verifiers at inference time only, and turning a verifier into a reward raises two difficulties. It reports only acceptance or rejection, which gives no gradient when every attempt fails, and a policy can cheat it, most simply by telling it to assume what the proof was supposed to establish. We build an agentic environment and reward that address both. The reward reads the verifier's diagnostics rather than its verdict, scoring a candidate by the weighted fraction of proof obligations it discharges, extended with hidden probes from a reference proof's annotations. An integrity audit outranks the verifier, rejecting any candidate that alters the specifi
    
[^226]: 随机特征二元模型的精确缩放定律

    Exact Scaling Laws for a Random-Features Bigram Model

    [https://arxiv.org/abs/2610.19281](https://arxiv.org/abs/2610.19281)

    本文在具有Zipf词元统计的瓶颈随机特征二元模型中精确推导出预测遗憾随模型宽度、词表大小和Zipf指数变化的缩放定律，证明纯粹由表征容量约束即可产生经典神经缩放定律的函数形式。

    

    理解神经缩放定律对理论和实践都越来越重要，然而即使在简单的设置下，基于第一性原理的解释仍然有限。在这项工作中，我们研究了具有幂律（Zipf）词元统计的瓶颈二元模型中的容量受限缩放，将其视为大型语言模型的一个解析可处理的玩具模型。固定的随机嵌入作为表征容量的代理，并诱导出一个显式的宽度瓶颈，使我们能够推导出预测遗憾作为模型宽度、词表大小和Zipf指数的函数的缩放定律。我们的结果通过分离出纯粹由表征约束引起的缩放，补充了近期基于优化动力学的理论解释。在适当的范围内，该理论恢复了经典神经缩放定律的函数形式，并表明重尾语言统计可能是优化驱动和容量驱动缩放的共同底层原因。

    arXiv:2610.19281v1 Announce Type: new  Abstract: Understanding neural scaling laws is increasingly important for theory and practice, yet first-principles explanations remain limited even in simple settings. In this work, we study capacity-limited scaling in a bottlenecked bigram model with power-law (Zipf) token statistics, viewed as an analytically tractable toy model of a large language model. Fixed random embeddings serve as a proxy for representational capacity and induce an explicit bottleneck of width, allowing us to derive scaling laws for prediction regret as a function of model width, vocabulary size, and the Zipf exponent. Our results complement recent theoretical accounts based on optimization dynamics by isolating scaling due purely to representational constraints. In the appropriate regime, the theory recovers the functional form of classical neural scaling laws and suggests that heavy-tailed language statistics may underlie both optimization- and capacity-driven scaling.
    
[^227]: 面向尺寸泛化的无维度缩放定律：一种指数级分离

    Dimension-Free Scaling Laws for Size Generalization: An Exponential Separation

    [https://arxiv.org/abs/2610.19279](https://arxiv.org/abs/2610.19279)

    该论文建立了尺寸泛化的统计学习理论，证明了尺寸泛化与经典域内学习之间存在样本复杂度上的指数级分离，并给出了与输入维度无关的缩放定律。

    

    现代机器学习模型越来越多地处理可变大小的结构化数据，例如集合和图。一个核心需求是尺寸泛化：即模型在某一尺寸的输入上训练后，能够泛化到不同尺寸（可能大得多）输入的能力。然而，经典统计学习理论主要考虑固定的输入域，其理论保证往往随输入维度的增加而恶化。尽管已有大量工作确定了集合和图上的模型何时能够跨输入尺寸进行迁移，但这些结果并未刻画这样做的统计代价：尺寸泛化的样本复杂度，以及它与经典域内学习之间的关系，在很大程度上仍是未知的。我们发展了尺寸泛化的统计学习理论，并在样本复杂度上建立了一个尖锐的指数级分离。设 $n$ 表示训练样本的数量，我们证明了多项式源维度……

    arXiv:2610.19279v1 Announce Type: new  Abstract: Modern machine learning models increasingly operate on variable-size structured data, such as sets and graphs. A central requirement is \emph{size generalization}: the ability of a model trained on inputs of one size to generalize to inputs of a different, potentially much larger, size. Classical statistical learning theory, however, primarily considers fixed input domains, and its guarantees often deteriorate with input dimension. Although extensive work has established when models on sets and graphs can be transferred across input sizes, such results do not characterize the statistical cost of doing so: the sample complexity of size generalization, and its relationship to classical in-domain learning, remain largely unknown. We develop a statistical learning theory of size generalization and establish a sharp exponential separation in sample complexity. Letting $n$ denote the number of training samples, we show that polynomial source d
    
[^228]: 面向扩散语言模型的结果路由蒸馏与强化学习

    Outcome-Routed Distillation and Reinforcement Learning for Diffusion Language Models

    [https://arxiv.org/abs/2610.19271](https://arxiv.org/abs/2610.19271)

    提出一种无需似然的结果路由蒸馏与强化学习流水线，通过验证器结果路由 rollout 组，结合无比率 REINFORCE、教师密集 KL 监督与检查点锚定 KL 约束，有效训练扩散语言模型。

    

    在线策略蒸馏能够对学生模型自身生成的样本提供密集监督，但其监督目标仍然来自教师模型。基于验证器的强化学习（RL）则使用结果信号，但对验证器覆盖范围之外的能力没有直接约束。将两者结合起来对于扩散语言模型而言十分困难，因为双向去噪机制使得基于比率的方法所需的策略似然变得不可靠。我们为混合专家架构的扩散语言模型开发了一个无需似然的训练流水线。每个 rollout 组根据验证器结果进行路由：混合组使用无比率、组中心化的 REINFORCE 方法；如果学生模型的所有 rollout 都失败但教师模型成功，则模型在学生的去噪画布上接受来自教师的密集 KL 监督；其余所有组则被跳过。一个锚定于蒸馏后检查点的 KL 项限制了模型在验证器奖励之外的变动。该工具调用教师模型仅在 18.6% 的训练提示上成功。在 24 个去噪

    arXiv:2610.19271v1 Announce Type: new  Abstract: On-policy distillation provides dense supervision on the student's own samples, but its targets still come from the teacher. Verifier reinforcement learning (RL) instead uses an outcome signal, but it provides no direct constraint on capabilities outside the verifier. Combining the two is difficult for diffusion language models (dLLMs), because bidirectional denoising makes the policy likelihoods required by ratio-based RL unreliable.   We develop a likelihood-free pipeline for a mixture-of-experts dLLM. Each rollout group is routed by verifier outcome. Mixed groups use ratio-free, group-centred REINFORCE. If every student rollout fails but a teacher succeeds, the model receives dense KL supervision on the student's denoising canvases. All other groups are skipped. A KL term anchored to the distilled checkpoint limits changes outside the verifier reward.   The tool-calling teacher succeeds on only 18.6% of the training prompts. At 24 den
    
[^229]: 探索基于深度学习的森林分类的可解释性——利用Sentinel-2时间序列分离相似森林类别

    Exploring Interpretability in Deep Learning-Based Forest Classification - Separating Similar Forest Classes using Sentinel-2 Time Series

    [https://arxiv.org/abs/2610.19270](https://arxiv.org/abs/2610.19270)

    该研究利用循环神经网络对Sentinel-2时间序列进行树种分类，并借助积分梯度可解释性方法量化光谱-时间特征贡献，从而有效区分纯林与混交林（尤其是挪威云杉）等相似森林类别。

    

    生态保险理论认为，多样化的森林更具韧性，能够更好地适应不断变化的环境条件。中欧日益加剧的干旱和热胁迫推动了混交林的发展，以维持生态系统服务。然而，物种分布的多样性使基于遥感的物种制图变得复杂，因为在像素级别区分纯林与混交林组分仍然具有挑战性。我们评估了循环神经网络（RNN）从Sentinel-2时间序列中进行树种分类的潜力，并研究了类别判别背后的光谱-时间特征，特别着重于分离纯挪威云杉像素与混交云杉像素。我们将RNN分类模型与随机森林（RF）和极限梯度提升（XGBoost）进行了比较，并使用积分梯度（IG）来量化跨光谱波段和时间的特征归因。表现最佳的RNN实现了……（原文摘要在此处截断）

    arXiv:2610.19270v1 Announce Type: new  Abstract: The ecological insurance theory posits that diverse forests are more resilient and better able to adapt to changing environmental conditions. Increasing drought and heat stress in Central Europe promotes mixed forests for maintaining ecosystem services. However, a diverse species distribution complicates remote sensing-based species mapping, as distinguishing pure from mixed species compositions at the pixel level remains challenging. We evaluated the potential of recurrent neural networks (RNNs) for tree-species classification from Sentinel-2 time series and investigated the spectral-temporal features underlying class discrimination, with particular emphasis on separating pure from mixed Norway spruce pixels. We compared RNN classification models with Random Forest (RF) and Extreme Gradient Boosting (XGBoost) and used Integrated Gradients (IG) to quantify feature attributions across spectral bands and time. The best-performing RNN achie
    
[^230]: 超越时间归纳：面向参数化随机波动率定价的结构化条件神经求解器

    Beyond Time Induction: Structured Conditional Neural Solvers for Parametric Stochastic-Volatility Pricing

    [https://arxiv.org/abs/2610.19254](https://arxiv.org/abs/2610.19254)

    提出BeTINN结构化条件神经求解器，通过多方向时间-参数条件化和联合残差通路扩展时间诱导调制，在Heston随机波动率定价中显著提升期权价值与关键敏感度的联合预测精度。

    

    物理信息神经网络为在无需标注数据的情况下学习参数化偏微分方程解族提供了一个灵活框架，然而在异构机制之间保持精确的导数仍然具有挑战性。结构化神经求解器提供了紧凑的表示形式，但受限的条件参数化可能限制其在整个解族上的适应能力。我们提出了BeTINN，这是一种结构化条件架构，通过多方向的时间-参数条件化和联合残差通路扩展了时间诱导调制机制。我们在Heston随机波动率偏微分方程上评估了BeTINN，这是一个具有挑战性的参数化问题，涉及混合导数以及具有经济意义的一阶和二阶敏感度。相对于我们的基线方法，BeTINN在归一化期权价值和关键敏感度指标上提升了联合精度。受控消融实验刻画了其各个架构组件的作用。我们进一步发现验证p……（摘要在此处被截断）

    arXiv:2610.19254v1 Announce Type: new  Abstract: Physics-informed neural networks provide a flexible framework for learning parametric PDE solution families without labeled data, yet preserving accurate derivatives across heterogeneous regimes remains challenging. Structured neural solvers offer compact representations, but restrictive conditional parameterizations can limit adaptation across the solution family. We introduce BeTINN, a structured conditional architecture extending time-induced modulation through multi-directional time-parameter conditioning and a joint residual pathway. We evaluate BeTINN on the Heston stochastic-volatility PDE, a demanding parametric problem with mixed derivatives and economically relevant first- and second-order sensitivities. BeTINN improves joint accuracy across normalized option value and key sensitivity metrics relative to our baselines. Controlled ablations characterize the roles of its architectural components. We further find that validation p
    
[^231]: 多敏感属性分布公平性：联合Wasserstein审计与Shapley归因分配

    Multi-Sensitive Distributional Fairness: Joint Wasserstein Audits and Shapley Allocation

    [https://arxiv.org/abs/2610.19252](https://arxiv.org/abs/2610.19252)

    该论文提出用一维Wasserstein几何对多个敏感属性进行联合公平性审计，将联合差异刻画为全单元条件分位数过程的方差，并借助Shapley值实现与顺序无关的差异归因，从而捕捉单坐标审计会遗漏的交叉性歧视。

    

    针对多个敏感属性的公平性审计通常一次只针对单个属性坐标进行。这可能遗漏纯粹的交叉性差异：一个评分可能与每个敏感坐标各自独立，却依赖于它们的联合特征。我们利用一维Wasserstein几何，研究实值评分的分布性人口统计均等。我们区分了两种互补的构造。条件审计和顺序审计通过其零点集刻画联合均等，但它们的正值通常无法对固定的联合Wasserstein差异进行分解。相比之下，全单元条件分位数函数定义了一个取值于希尔伯特空间的随机变量，其方差即为联合差异。该分位数过程的条件期望诱导出一个单调信息博弈：任意属性揭示顺序都能给出同一联合差异的精确鞅分解，而Shapley值则提供了一种跨敏感属性、与顺序无关的归因分配方法。

    arXiv:2610.19252v1 Announce Type: cross  Abstract: Fairness audits with several sensitive attributes are often performed one coordinate at a time. This can miss purely intersectional disparities: a score may be independent of each sensitive coordinate while depending on their joint profile. We study distributional demographic parity for a real-valued score using one-dimensional Wasserstein geometry. We distinguish two complementary constructions. Conditional and sequential audits characterize joint parity through their zero sets, but their positive values do not generally decompose a fixed joint Wasserstein disparity. By contrast, full-cell conditional quantile functions define a Hilbert-valued random variable whose variance is the joint disparity. Conditional expectations of this quantile process induce a monotone information game: every reveal order yields an exact martingale decomposition of the same joint disparity, while Shapley values provide an order-free attribution across sens
    
[^232]: 精细化拓扑遍次调度：SE(3)等变结构预测中动态k近邻图的阻尼信赖域补偿

    Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction

    [https://arxiv.org/abs/2610.19246](https://arxiv.org/abs/2610.19246)

    提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。

    

    等变结构预测流水线在坐标更新与k近邻（kNN）图构建之间交替进行，其中图的成员关系是坐标的离散、分段常数函数。在每个块边界（PB）重建该图是训练轮次中最昂贵的部分，而复用一个冻结图（F）虽然快速，却会留下拓扑差距。我们研究了一种双遍设计：廉价的遍次记录少量状态信息（间隔边距、kNN半径、实测的每边界位移），第二遍则进行保守补偿。我们将图构建基准的选择形式化为一个阻尼不动点迭代，其唯一不动点恰好为PB，并由此推导出：(i) 具有双侧常数 $m_i/4$ 和单侧常数 $m_i/2$ 的精确不翻转（non-flip）证书；(ii) 最小方差（BLUE）重放增益 $\gamma^*$ 及由此产生的可行性交叉点；(iii) 一个 $\ell_\infty$ 信赖域投影，可保……

    arXiv:2610.19246v1 Announce Type: new  Abstract: Equivariant structure-prediction pipelines alternate coordinate updates with the construction of a k-nearest-neighbour (kNN) graph whose membership is a discrete, piecewise-constant function of the coordinates. Rebuilding that graph at every block boundary (PB) is the expensive part of an epoch, whereas reusing one frozen graph (F) is fast but leaves a topology gap. We study a two-pass design in which the cheap pass records a small amount of state (margins, kNN radii, measured per-boundary displacements) and a second pass compensates conservatively. We formulate the choice of the graph-construction basis as a damped fixed-point iteration whose unique fixed point is exactly PB, and derive (i) exact non-flip certificates with the two-sided constant $m_i/4$ and one-sided constant $m_i/2$, (ii) the minimum-variance (BLUE) replay gain $\gamma^*$ and the resulting feasibility crossover, (iii) an $\ell_\infty$ trust-region projection that prese
    
[^233]: FinGREAT：在金融图引导的检索与类型识别任务上评估大语言模型

    FinGREAT: Evaluating LLMs on Financial Graph-guided Retrieval And Typing

    [https://arxiv.org/abs/2610.19240](https://arxiv.org/abs/2610.19240)

    该论文提出FinGREAT基准，基于包含11个账户节点和11种交易类型边的金融会计图，在跨越26年的1,300份SEC 10-K文件上系统评估了开源大语言模型提取和分类复式记账关系证据的能力，并发布了含483个段落的人工标注关系分类基准。

    

    金融披露通过非结构化文本表达了复式记账法的底层关系。尽管金融领域的大语言模型基准测试主要侧重于问答和分类，但基于模式（schema）引导的证据发现仍然相对缺乏研究。我们探究了开源权重的大语言模型能否提取并分类与文本背后预定义的复式记账关系相对应的证据。我们使用一个由11个节点（账户）和11条边（交易类型）构成的金融会计图，评估了来自5个开源权重模型家族的中等规模模型，覆盖跨越26年的1,300份SEC 10-K年报第7项（管理层讨论与分析），并将其提取性能与三个专有模型进行对比。结果显示，各模型在提取数量、边分布以及跨模型一致性方面存在显著差异。基于该研究，我们进一步引入了一个包含483个段落的人工标注关系分类基准。

    arXiv:2610.19240v1 Announce Type: new  Abstract: Financial disclosures express the underlying relationships of double-entry accounting through unstructured text. While financial LLM benchmarks predominantly emphasize question answering and classification, schema-guided evidence discovery remains relatively under-explored. We investigate whether open-weight LLMs can extract and classify evidence corresponding to predefined double-entry accounting relationships underlying the text. Using a financial accounting graph comprising 11 nodes (accounts) and 11 edges (transaction types), we evaluate mid-sized models from 5 open-weight families across 1,300 SEC 10-K Item 7 filings spanning 26 years, comparing their extraction performance against three proprietary models. Results reveal substantial differences in extraction volume, edge distributions, and cross-model agreement. Building on this study, we introduce a human-annotated relation classification benchmark comprising 483 passages across t
    
[^234]: 大语言模型使用可复用和可组合的抽象表示进行编程

    LLMs program using Reusable and Composable Abstract Representations

    [https://arxiv.org/abs/2610.19238](https://arxiv.org/abs/2610.19238)

    大语言模型在生成代码时利用中间层中形成的可复用、可组合的抽象表示，这些表示可跨编程语言迁移，并可通过激活引导独立操纵和灵活组合来精确控制代码生成。

    

    在编写代码时，相同的计算可以用不同的编程语言和不同的实现策略来表达。我们证明，大语言模型在生成代码时，其激活值能够捕获这种共享结构，形成可复用且可组合的抽象表示。这些表示出现在模型的中间层，可以跨编程语言复用，并赋予模型在各种干预条件下生成代码时令人印象深刻的泛化能力。通过激活引导，我们展示了函数、编程语言和实现策略可以被独立操纵。对语言和函数的独立干预可以灵活组合以控制代码生成，即使在提示中不包含语言和函数相关信息的情况下也是如此。此外，我们发现对一种编程语言中任意定义函数的示例进行微调，可以迁移到其他语言上。

    arXiv:2610.19238v1 Announce Type: new  Abstract: When writing code, the same computation can be expressed in different programming languages and through different implementation strategies. We show that the activations of LLMs capture this shared structure in reusable and composable abstract representations when they generate code. These representations emerge in the middle layers, are reusable across programming languages, and afford the model an impressive amount of generalization ability when generating code under various interventions. Using activation steering, we show that function, programming language, and implementation strategy can be independently manipulated. The independent interventions on language and function can be flexibly composed to control code generation, even in cases when the prompt contains no information about language and function. We additionally find that finetuning on examples of arbitrarily defined functions in one programming language transfers to other 
    
[^235]: 从检测到预判：口吃言语的连续起始预测

    From Detection To Anticipation: Continuous Onset Forecasting For Stuttered Speech

    [https://arxiv.org/abs/2610.19235](https://arxiv.org/abs/2610.19235)

    该论文将口吃预测重新建模为连续因果时间-事件预测任务，仅利用先前语音每100毫秒预测口吃起始风险，并通过紧凑因果Conformer建模冻结Whisper语音表示的时间演化，实现了从“事后检测”到“事前预判”的转变。

    

    自动口吃检测系统主要在言语不流畅发生之后才进行检测，这限制了其在预判性干预中的应用。我们通过将口吃预测构建为连续的因果性“时间-事件”预测任务来填补这一空白，在该任务中，模型仅利用先前的语音，每100毫秒估计一次即将发生的口吃起始的风险。为支持这一任务，我们对36名说话人的FluencyBank Timestamped数据集进行了扩充，加入了经人工审核的重复、延长和阻塞事件，共产生2,872个预测事件。随后，我们使用一个仅有367万可训练参数的紧凑因果Conformer对冻结的Whisper-large-v3语音表示的时间演化进行建模。所提出的方法达到了0.63的AUROC。时间分析进一步表明，显式建模先前语音状态优于简单增大静态声学窗口，而离散时间生存分析公式能够在300毫秒至1.5秒的范围内提供连贯一致的预测。

    arXiv:2610.19235v1 Announce Type: cross  Abstract: Automatic stuttering systems primarily detect dysfluencies after they have occurred, limiting their use for anticipatory intervention. We address this gap by formulating stuttering prediction as continuous causal time-to-event forecasting, where a model estimates the risk of an upcoming stuttering onset every 100 ms using only preceding speech. To support this task, we augment FluencyBank Timestamped for 36 speakers with manually reviewed repetitions, prolongations, and blocks, yielding 2,872 forecasting events. We then model the temporal evolution of frozen Whisper-large-v3 speech representations using a compact causal Conformer with 3.67M trainable parameters. The proposed approach achieves an AUROC of 0.63. Temporal analysis further shows that explicitly modeling preceding speech states outperforms simply increasing the static acoustic window, while a discrete-time survival formulation provides coherent forecasts from 300 ms to 1.5 
    
[^236]: Spectral-MoE：密集模型到MoE转换中的异构逐层专家利用率

    Spectral-MoE: Heterogeneous Per-Layer Expert Utilization in Dense-to-MoE Conversion

    [https://arxiv.org/abs/2610.19232](https://arxiv.org/abs/2610.19232)

    Spectral-MoE提出了一种在密集模型到MoE转换过程中学习在各层之间异构分配激活专家数量预算的方法，突破了传统方法对每层固定相同利用率的限制，从而更有效地降低推理算术成本。

    

    大语言模型在推理过程中会带来巨大的内存需求和算术成本。密集模型到混合专家模型的转换是一种旨在降低这些算术成本的模型优化技术。Dense-to-MoE转换将预训练的前馈网络（FFN）神经元划分为多个专家，并为每个token选择其中的一部分。现有的主流方法分为两类：解析方法利用启发式规则快速生成转换后的模型，而学习方法则通过梯度优化以慢得多的速度生成模型。然而，这两种方法都在每一层固定使用相同的激活专家数量（即“利用率”）。我们提出了Spectral-MoE，它学习在各层之间异构地分配利用率预算。其名称源于一种比喻：将密集的FFN分解为多个专家，就如同将一束光分离成其光谱一样。Spectral-MoE包含两种不同的方法：一种静态方法，使用单个每（摘要在此处截断）

    arXiv:2610.19232v1 Announce Type: new  Abstract: Large language models impose substantial memory demands and arithmetic costs during inference. Dense-to-mixture-of-experts (MoE) conversion is a model optimization technique that aims to reduce these arithmetic costs. Dense-to-MoE conversion partitions pretrained feed-forward network (FFN) neurons into experts and selects a subset for each token. Leading methods fall into two categories: analytic methods produce a converted model quickly using heuristics, while learning methods produce a model much more slowly by optimizing via gradients. Both approaches, however, fix the same active expert count, or "utilization," at every layer. We introduce Spectral-MoE, which learns to allocate a utilization budget heterogeneously across layers. Its name draws a metaphor between separating a dense FFN into experts and separating a light beam into its spectrum. Spectral-MoE consists of two distinct approaches: a static approach that uses a single per-
    
[^237]: 驾驭不动点曲线：连续思维机器中的图像分类

    Riding the Fixed-Point Curve: Image Classification in the Continuous Thought Machine

    [https://arxiv.org/abs/2610.19220](https://arxiv.org/abs/2610.19220)

    本文揭示了连续思维机器（CTM）图像分类的内在机制：每个思考步骤是具有平衡放大的吸引性非正规映射，神经状态的不动点随每步移动，思维轨迹即这些不动点描绘的曲线（大部分运动跨图像共享并表现为行波活动），从而解释了CTM的涌现思维过程。

    

    连续思维机器（CTM）是一种循环神经网络，它关注输入并基于神经元之间的同步性做出预测。在跨越多个领域的任务中，它表现出尚未被解释的涌现思维过程。我们报告了关于CTM如何对图像进行分类的多项发现，重点聚焦于ImageNet。每个思考步骤是一个具有吸引性但强烈非正规的映射，表现出平衡放大效应（Murphy and Miller, 2009）；虽然吸引性在CTM初始化时就已存在，但训练极大地促进了非正规性。根据巴拿赫（Banach）定理，神经状态具有一个局部唯一的不动点，该不动点在每一步都会移动，而思维轨迹就是这些不动点所描绘的曲线，滞后一个时间步跟随其后。该曲线的大部分运动在不同图像之间是共享的，并表现为先前报道过的跨越神经元的行波活动。沿着思维的注意力调度……

    arXiv:2610.19220v1 Announce Type: new  Abstract: The Continuous Thought Machine (CTM) is a recurrent network that attends to inputs and makes predictions based on the synchronization between its neurons. On tasks spanning many domains, it displays emergent thought processes that have not been explained. We report a number of findings about how the CTM classifies images, focusing on ImageNet. Each thinking step is an attracting but strongly non-normal map exhibiting balanced amplification (Murphy and Miller, 2009); while attraction is already present at CTM initialization, training highly promotes non-normality. By Banach's theorem, the neural state has a locally unique attracting fixed point, which moves at every step, and the thought trajectory is the curve these fixed points trace, followed one tick behind. Most of that curve's motion is shared across images, and visible as a previously reported traveling wave of activity across neurons. The attention schedule along the thought is la
    
[^238]: 很可能缩小协作差距（Narrowing the Collaboration Gap, Probably）

    Narrowing the Collaboration Gap, Probably

    [https://arxiv.org/abs/2610.19215](https://arxiv.org/abs/2610.19215)

    该论文提出让持有不同私有信息的多智能体通过交流关于动作的校准概率信念而非仅交流动作本身，并从理论和实验上证明这种概率沟通方式能更有效地汇聚决策相关信息、提升协作表现。

    

    大语言模型越来越多地被部署为由多个智能体组成的团队，这些智能体对同一个共享任务持有不同的私有信息。这样的智能体应如何以简洁的方式进行沟通，才能促进决策相关信息的汇聚？仅交流所提议的动作可能会丢弃与决策相关的不确定性：两个智能体可能出于不同的原因偏好同一个动作。我们研究了一种替代性的交互接口，即智能体之间交流关于动作的经校准的概率信念。我们首先分析了一个简单的合成场景，该场景刻画了部分信息条件下多轮推理的结构。在该场景中，我们从理论和实证上证明，传递无偏的概率可以严格优于传递动作。随后，我们在一个由经过训练的Transformer和预训练LLM智能体参与的协作迷宫求解任务中测试了同样的原则。我们的结果表明，概率沟通优于动作沟通。

    arXiv:2610.19215v1 Announce Type: new  Abstract: Large language models are increasingly deployed as teams of agents that hold different private information about a shared task. How should such agents concisely communicate in a way that promotes aggregation of decision-relevant information? Communicating only proposed actions can discard decision-relevant uncertainty: two agents may prefer the same action for different reasons. We study an alternative interface in which agents communicate calibrated beliefs over actions. We first analyze a simple synthetic setting that captures the structure of multi-round reasoning under partial information. In this setting, we show theoretically and empirically that communicating unbiased probabilities can be strictly more powerful than communicating actions. We then test the same principle in a collaborative maze-solving task with trained transformers and pretrained LLM agents. Our results demonstrate that probability communication outperforms action
    
[^239]: 将因果发现中基于完整数据的可辨识性结果推广至缺失数据场景

    Generalizing Identifiability Results in Causal Discovery from Complete Data to Missing Data Settings

    [https://arxiv.org/abs/2610.19211](https://arxiv.org/abs/2610.19211)

    该论文建立了在包括非随机缺失在内的多种缺失机制下因果图保持可辨识的一般性充分条件，提供了将完整数据因果发现算法的正确性保证扩展到缺失数据场景的框架，并针对加性噪声模型提出了两个渐近正确的算法。

    

    虽然针对完整数据的因果发现方法已经成熟，但能够处理缺失值的此类算法仍然有限。我们建立了一般性的充分条件，在这些条件下，因果图在各种缺失类别（包括非随机缺失，Missing Not At Random）下仍然是可辨识的（即可以从可用数据分布中唯一确定），并提供了一个框架，用于将完整数据因果发现算法的正确性保证扩展到缺失数据场景。我们的结果表明，当满足以下条件时，无未观测混杂因素的因果图的可辨识性得以保持：(1) 各缺失机制之间不存在因果交互作用；(2) 没有变量直接导致其自身的缺失；(3) 缺失机制满足忠实性条件，或遵循广义线性模型。基于这些理论结果，我们提出了两个针对加性噪声模型的渐近正确算法，用于恢复潜在的因果……

    arXiv:2610.19211v1 Announce Type: cross  Abstract: While causal discovery methods have matured for complete data, such algorithms for handling missing values remain limited. We establish general sufficient conditions under which causal graphs remain identifiable (i.e., uniquely determined from the available data distribution) under various missingness categories, including Missing Not At Random, and provide a framework to extend correctness guarantees from causal discovery algorithms for complete data to missing data settings. Our results show that identifiability of causal graphs with no unobserved confounders is preserved when (1) missingness mechanisms do not causally interact, (2) no variable directly causes its own missingness, and (3) either faithfulness holds for missingness mechanisms, or they follow generalized linear models. Guided by these theoretical results, we introduce two asymptotically correct algorithms for Additive Noise Models (ANMs) that recover the underlying caus
    
[^240]: 嵌入几何形态决定大语言模型路由：追踪失效模式并提升泛化能力

    Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization

    [https://arxiv.org/abs/2610.19198](https://arxiv.org/abs/2610.19198)

    该论文揭示了语义嵌入的几何结构是LLM路由在未见任务上失效的根源，并提出用Qwen2.5-7B-Instruct深层最后token的隐藏状态替代语义嵌入作为表示，从而显著提升路由方法对未见任务的泛化能力。

    

    随着可用的大语言模型（LLM）数量不断增长，如何为给定提示选择合适的模型已成为一个关键问题。路由领域的许多研究依赖于将提示语义嵌入到向量空间中，然后对嵌入表示应用轻量级分类器。这些方法在分布内表现稳健，但难以泛化到未见过的任务。通过研究语义嵌入的几何结构，我们追踪到一个明确的失效模式：决定未见提示邻居的特征是LLM选择的不可靠代理。因此，我们用以下映射替换语义嵌入：对提示运行Qwen2.5-7B-Instruct的部分前向传播，并使用其在深层的最后一个token的隐藏状态作为表示。当对这种新表示应用kNN和其他简单分类器时，我们在泛化方面获得了明确的结果：在平均各种成本惩罚下，它们的性能……

    arXiv:2610.19198v1 Announce Type: cross  Abstract: With the growing number of LLMs available, selecting which one to use for a prompt has become a key question. A large part of the routing literature relies on semantically embedding prompts into a vector space and then applying a lightweight classifier to the embedded representations. These methods perform robustly in-distribution but they struggle to generalize to unseen tasks. By studying the geometry of semantic embeddings, we trace a clear failure mode: features that determine the neighbors of unseen prompts are an unreliable proxy for LLM selection. We therefore replace the semantic embedding with the following map: we run a partial forward pass of Qwen2.5-7B-Instruct on the prompt and use the hidden state of its last token at a deep layer as the representation. When applying kNN and other simple classifiers to this new representation, we obtain a clear result in generalization: on average across cost penalties, their performance 
    
[^241]: FireGraphNet：基于有向图神经网络的亚日尺度野火蔓延预测

    FireGraphNet: Sub-daily wildfire spread prediction with directed graph neural networks

    [https://arxiv.org/abs/2610.19191](https://arxiv.org/abs/2610.19191)

    FireGraphNet将12小时野火蔓延预测建模为带风向、地形、燃料与障碍物边特征的有向H3图上的节点分类任务，平均精度达到最强距离基线的1.7至1.9倍，且性能提升主要来自沿风向的消息传递机制。

    

    野火应对决策需要在数小时内做出，然而大多数学习型蔓延模型却以每日为预测频率，并以燃烧面积等为目标。我们提出FireGraphNet，将12小时火势增长建模为有向H3图上的节点分类任务，其中候选单元位于火线前方，边上包含风向对齐、地形、燃料连续性和障碍物等属性。在卫星观测的火线周界数据上，该方法在加州测试火灾中取得的平均精度（average precision）达到最强距离基线的1.7至1.9倍，在州外火灾中达到1.62倍。我们证明这一能力来源于消息传递机制，而非单元特征或聚合器；该能力在州内早期火灾上最强，并依靠其边属性沿风向向远处传递，且会被池化操作所掩盖。一个卷积网络在用新增火势数据训练后也能达到相似的排序性能，但校准效果较差，因此候选集构建、目标定义和聚合方式都会影响关于模型能力的结论。对燃料和障碍物属性进行修改时，在有引导的设置下比随机设置更能减少预测的蔓延范围。

    arXiv:2610.19191v1 Announce Type: new  Abstract: Wildfire decisions are made in hours, yet most learned spread models predict daily on targets including burned area. We propose FireGraphNet which formulates 12-hour growth as node classification on directed H3 graphs, with candidates ahead of the front and wind alignment, terrain, fuel continuity and barriers on edges. On satellite perimeters it achieves average precision 1.7-1.9x the strongest distance baseline on test California fires and 1.62x out of state. We show this skill comes from message passing rather than cell features or the aggregator, is highest on young fires in state, is carried deep along the wind by its edges, and is hidden by pooling. A convolutional network matches this ranking once trained on new growth, at worse calibration, so candidate construction, target definition, and aggregation shape conclusions about model skill. Editing fuel and barrier attributes reduces predicted spread more under guided than random pl
    
[^242]: 基于保形认证的自适应计算

    Adaptive Computation with Conformal Certification

    [https://arxiv.org/abs/2610.19189](https://arxiv.org/abs/2610.19189)

    本文提出“保形自适应”方法，利用保形认证为科学成像数据（医学与天文图像）的自适应计算提供统计风险保证，在降低计算成本的同时保留关键稀疏区域的分辨率并提供可靠的不确定性量化。

    

    涉及医学（如3D放射影像、显微图像）和天文传感器的科学成像数据带来了与自然图像不同的挑战。一个关键挑战是这些数据集的规模通常比自然图像大几个数量级。由于硬件需求增加，在这些数据上进行高效的训练和推理十分困难，往往需要定制的解决方案。此外，肿瘤和天文现象等相关目标相对于背景信息呈稀疏分布。量化、剪枝等标准效率提升技术会对所有区域进行统一缩放，可能损害预测的精度/保真度。因此，能够在不损失数据相关部分分辨率的前提下降低计算需求、并为预测提供不确定性估计的高效方法至关重要。我们提出了具有风险保证的“保形自适应”方法……（原文摘要至此截断）

    arXiv:2610.19189v1 Announce Type: new  Abstract: Scientific imaging data involving medical (e.g., 3D radiographic, microscopic images) and astronomical sensors pose different challenges from natural images. One key challenge is that these datasets are often orders of magnitude larger than natural images. Efficient training and inference on such data is challenging due to the increased hardware requirements, often needing bespoke solutions. Moreover, the relevant objects such as tumors and astronomical phenomena are sparsely distributed with respect to the background information. Standard efficiency techniques like quantization and pruning apply uniform scaling of all regions, potentially compromising the precision/fidelity of predictions. Thus, efficient methods that reduce computational requirements without losing resolution for relevant parts of the data and with uncertainty estimates on predictions are essential. We present \textit{Conformal Adaptation} with risk guarantees for unce
    
[^243]: 基于扩散正则化Stein变分梯度下降的结构感知贝叶斯全波形反演

    Structure-aware Bayesian full waveform inversion via diffusion-regularized Stein variational gradient descent

    [https://arxiv.org/abs/2610.19187](https://arxiv.org/abs/2610.19187)

    该论文提出将学习得到的扩散模型作为隐式正则化机制嵌入基于退火Stein变分梯度下降的贝叶斯全波形反演框架，以刻画复杂地质结构先验并增强反演的不确定性量化能力。

    

    全波形反演（FWI）是高分辨率地下模型重建的强大工具，但它仍具有高度非线性和不适定性，尤其是在初始模型不准确或采集数据含噪且不完整的情况下。退火Stein变分梯度下降（ASVGD）通过用一组速度模型集合来近似后验分布，为FWI提供了一种基于粒子的贝叶斯框架，从而能够在单一确定性解之外实现不确定性量化。然而，传统的ASVGD-FWI通常依赖于简单的手工先验，如高斯或平滑假设，这往往不足以刻画复杂的地质结构。为解决这一局限，我们在ASVGD-FWI框架中引入训练好的扩散模型（DM）作为隐式正则化机制。训练好的扩散模型无需显式计算扩散先验密度，而是作为……（摘要在此处截断）

    arXiv:2610.19187v1 Announce Type: cross  Abstract: Full waveform inversion (FWI) is a powerful tool for high-resolution subsurface model reconstruction, but it remains highly nonlinear and ill-posed, especially when the initial model is inaccurate or the acquired data are noisy and incomplete. Annealed Stein variational gradient descent (ASVGD) provides a particle-based Bayesian framework for FWI by approximating the posterior distribution with an ensemble of velocity models, thereby enabling uncertainty quantification beyond a single deterministic solution. However, conventional ASVGD-FWI commonly relies on simple handcrafted priors, such as Gaussian or smoothing assumptions, which are often insufficient to represent complex geological structures. To address this limitation, we introduce a learned diffusion model (DM) as an implicit regularization mechanism within the ASVGD-FWI framework. Instead of explicitly evaluating the diffusion prior density, the trained diffusion model acts as
    
[^244]: 几何充能下策略性排队中的学习：从老虎机到系统

    Learning in Strategic Queuing with Geometric Recharge: From Bandits to Systems

    [https://arxiv.org/abs/2610.19179](https://arxiv.org/abs/2610.19179)

    该论文将具有单数据包缓冲区的策略性排队问题与几何充能老虎机问题建立等价关系，并证明任何无悔学习队列都能达到至少等于服务器速率 Hamacher 和的长期平均吞吐量，且该界是紧的。

    

    我们研究排队系统中的数据包路由问题，其中队列扮演自私、短视的智能体角色，使用无悔学习算法来选择具有未知服务速率的服务器。为了对数据中心中使用的小缓冲区进行建模，我们假设每个服务器都有一个单数据包缓冲区。即使是如此微小的缓冲区也已经捕捉到了缓冲的关键特性：路由器可以在被接受的数据包仍处于服务中时利用多个服务器，但数据包可能会占用服务器多个回合，从而使未来的服务和学习变得复杂。我们首先分析将数据包路由到 $m$ 个服务速率分别为 $\mu_1, \dots, \mu_m$ 的缓冲服务器的单个队列。该设置等价于几何充能的再充能老虎机问题的一个实例。我们证明，任何无悔学习者都能实现至少等于服务器速率的 Hamacher 和的长期平均吞吐量，且该界是紧的。因此，单个无悔队列是稳定的当且仅当……

    arXiv:2610.19179v1 Announce Type: cross  Abstract: We study packet routing in queuing systems where queues act as selfish, myopic agents that use no-regret learning algorithms to select servers with unknown service rates. To model the small buffers used in data centers, we assume that each server has a single-packet buffer. Even such a tiny buffer already captures the features of buffering: routers can exploit multiple servers while accepted packets remain in service, but packets can occupy servers for multiple rounds, complicating future service and learning.   We first analyze a single queue routing packets to $m$ buffered servers with service rates $\mu_1, \dots, \mu_m$. This setting is equivalent to an instance of the recharging bandits problem with geometric recharge. We prove that any no-regret learner achieves a long-run average throughput of at least the Hamacher sum of the server rates, and that this bound is tight. Consequently, a single no-regret queue is stable if and only 
    
[^245]: 通过局部有害协变量偏移检测诊断低置信度患者画像

    Diagnosing Low-Confidence Patient Profiles via Localized Harmful Covariate Shift Detection

    [https://arxiv.org/abs/2610.19178](https://arxiv.org/abs/2610.19178)

    该论文提出结合MED3pa与Detectron的统一诊断流程，先识别可解释的低置信度患者画像，再在各画像内检测有害协变量偏移，从而定位临床预测模型在特定患者亚群中的性能退化。

    

    在部署临床预测模型之前，必须评估其可靠性以确保安全使用。错误可能源于训练与部署之间的分布偏移，也可能来自异质性能，即模型在某些患者亚群中系统性地表现不佳。然而，大多数事后诊断方法仅在总体层面评估可靠性，只检查部署数据是否类似于训练数据、模型平均表现是否良好，而未考虑可能导致某些患者受到不成比例影响的异质性。我们提出了一个统一这两种视角的诊断流程：MED3pa首先发现可解释的低置信度患者画像，随后在每个画像内部应用Detectron，以检验其性能退化是否与可检测的有害协变量偏移相关。在MIMIC-数据集上的三项临床预测任务和三种注入的亚群偏移中……（原文摘要至此被截断）

    arXiv:2610.19178v1 Announce Type: new  Abstract: Before deploying clinical prediction models, their reliability must be assessed to ensure safe use. Errors may arise from distribution shifts between training and deployment or from heterogeneous performance where the model systematically underperforms for certain patient subgroups. Yet most post-hoc diagnostic methods evaluate reliability only at the population level, checking whether the deployment data resemble the training data and whether the model performs well on average, without accounting for heterogeneity that may cause some patients to be disproportionately affected. We propose a unified diagnostic pipeline that combines both perspectives. MED3pa first discovers interpretable low-confidence patient profiles, and Detectron is then applied within each profile to test whether its degradation is associated with a detectable harmful covariate shift. Across three clinical prediction tasks and three injected subgroup shifts in MIMIC-
    
[^246]: 进入未知领域：在开放科学研究问题上对AI智能体进行基准测试

    Into the Unknown: Benchmarking AI agents on open scientific research problems

    [https://arxiv.org/abs/2610.19159](https://arxiv.org/abs/2610.19159)

    该论文推出了"Into the Unknown"基准测试，通过与能源部国家实验室科学家合作定义未发表的后续研究方向及评分准则，评估九个前沿AI智能体在开放科学问题上撰写后续论文的能力，结果显示135篇投稿无一满足全部条件，其中114篇未通过完整性检查。

    

    AI智能体能够对开放性科学问题开展研究吗？我们推出了"Into the Unknown"（进入未知领域）基准测试，该基准为智能体提供一篇科学论文以及一个待探索的研究方向，并要求其撰写一篇后续论文。我们与美国能源部（DOE）国家实验室的科学家合作，每位科学家都贡献了自己工作中一个未发表的后续研究方向，以及投稿必须满足的必要条件。虽然评估一篇后续论文的质量总体上具有挑战性，但利用这些条件形成评分准则，使得对没有参考答案的研究问题的评估更加可行。我们让九个前沿智能体各自撰写15篇后续论文，涵盖量子计算、等离子体物理、计算化学、材料科学和科学机器学习等领域。在135篇投稿中，没有一篇满足所有条件，其中114篇未通过我们的完整性检查，最常见的原因是……

    arXiv:2610.19159v1 Announce Type: cross  Abstract: Can AI agents carry out research on open scientific problems? We introduce Into the Unknown, a benchmark that provides an agent with a science paper along with a research direction to pursue and tasks it to write a follow-up paper. We work with scientists at US Department of Energy (DOE) national laboratories, who each contribute an unpublished follow-up direction from their own work together with the necessary conditions a submission must meet. While assessing the quality of a follow-up paper is challenging in general, using these conditions to form a rubric makes evaluation more tractable for research problems that have no reference answer. We have nine frontier agents write 15 follow-ups each, spanning quantum computing, plasma physics, computational chemistry, materials science, and scientific machine learning domains. None of the 135 submissions satisfies all the conditions, and 114 fail our integrity check, most often because the
    
[^247]: 揭秘在线策略蒸馏：教师模型的优势究竟在哪里？

    Demystifying On-Policy Distillation: Where Does the Teacher's Advantage Live?

    [https://arxiv.org/abs/2610.19145](https://arxiv.org/abs/2610.19145)

    该研究通过在师生不一致位置进行受控token翻转的干预实验，量化了在线策略蒸馏中教师优势的来源，发现重复的选择性纠正可将学生准确率从28.4%提升至74.5%（接近教师的76.9%），且基于概率质量和KL散度的选择性干预比随机翻转高出15-18个百分点。

    

    在线策略蒸馏日益被视为一种比强化学习更廉价的替代方案，但它本质上仍是一种依赖教师监督的知识蒸馏（KD）形式。我们从知识蒸馏的视角重新审视在线策略蒸馏，探究教师模型的优势在学生模型生成的轨迹中究竟体现在何处。通过在师生不一致的位置进行受控的token翻转，可以直接衡量教师纠正所带来的准确率提升。在筛选出的失败轨迹上，单次token翻转仅带来适度的平均收益，而重复翻转可将准确率从28.4%提升至74.5%，接近教师模型的76.9%。收益随翻转次数增加而持续累积，但在相似数量级下，基于概率质量和KL散度的选择方法比随机翻转高出15-18个百分点。在线策略蒸馏之后，选择性干预仍然有用，但更大程度的argmax一致性并不能在各模型对之间稳定地提升准确率。我们提倡采用能够学习更多（原文在此处截断）的训练计划。

    arXiv:2610.19145v1 Announce Type: new  Abstract: On-policy distillation (OPD) is increasingly studied as a cheaper alternative to reinforcement learning, yet it remains a form of knowledge distillation (KD) that relies on teacher supervision. We revisit OPD from the KD perspective by asking where the teacher's advantage lies along student-generated trajectories. Controlled token flips at teacher-student disagreements directly measure the accuracy gained from teacher corrections. Single-token flips yield a modest average gain on selected failed trajectories, whereas repeated flips raise accuracy from 28.4% to 74.5%, close to the teacher's 76.9%. Gains continue to accumulate with more flips, but mass- and KL-based selection outperform random flips by 15-18 percentage points at counts of similar magnitude. Selective intervention remains useful after OPD, but greater argmax agreement does not consistently improve accuracy across model pairs. We advocate training schedules that learn more v
    
[^248]: 图数据库中图神经网络训练与推理的探索

    Towards Training and Inference of Graph Neural Networks in Graph Databases

    [https://arxiv.org/abs/2610.19142](https://arxiv.org/abs/2610.19142)

    该论文证明图数据库可以直接用于图神经网络的训练与推理，通过 Cypher 实现即时采样与特征检索，并提供优化使训练加速约 2.5 倍、数据库内推理比冷加载快 10 倍以上且无额外内存开销。

    

    大规模图上的图神经网络在训练和推理时通常依赖于内存中的图存储以及临时搭建的数据处理流水线。图数据库为可随时间演化的图提供了可扩展的存储和高效的遍历能力，但其在图神经网络训练和推理中的适用性却很少被研究。我们证明图数据库可以直接用于图神经网络的训练和推理，执行采样、特征检索，甚至完整的前向传播。我们展示了不同的图神经网络训练流水线可以使用 Cypher 在图数据库中进行即时采样和检索。我们进一步提供了若干优化方法，在保持相同精度的同时将训练过程加速约 2.5 倍。我们证明了完全在数据库内进行推理是可行的，且与图神经网络的训练方式无关。在真实世界的图数据上，我们的推理设置比将图冷加载到内存中快 10 倍以上，且没有额外的内存开销。我们的工作为图数据库中图神经网络的训练与推理奠定了基础。

    arXiv:2610.19142v1 Announce Type: new  Abstract: Graph Neural Networks on large graphs commonly rely on in-memory graph storage with ad hoc data processing pipelines for both training and inference. Graph databases provide scalable storage and efficient traversal for graphs that can evolve over time, but their applicability in GNN training and inference are rarely studied. We show that graph DBs can be used in GNN training and inference directly, executing sampling, feature retrieval, and even the full forward pass. We show different GNN training pipelines can perform on-the-fly sampling and retrieval in graph DBs using Cypher. We further provide optimisations that speed up the training process by \~2.5x while maintaining the same accuracy. We illustrate that fully in-DB inference is possible, independent of how the GNNs are trained. Our inference setup can be >10x faster than cold loading the graph into RAM with no extra memory overhead on real-world graphs. Our work lays the foundati
    
[^249]: 线性时间内实现最优延迟的在线分词

    Online Tokenization with Optimal Delay in Linear Time

    [https://arxiv.org/abs/2610.19139](https://arxiv.org/abs/2610.19139)

    该论文首次形式化并解决了在线分词问题，提出了延迟达到最优且运行时间为线性的在线分词算法。

    

    分词是几乎所有大语言模型（LLM）中必不可少的预处理步骤。最流行的框架之一是BPE分词，它由特殊词元切分、预分词和字节对编码（BPE）三个部分组成。然而，现有的算法是顺序执行的，每个阶段只能处理上一阶段的完整输出。为了降低延迟和内存占用，最近的一些工作（Jiang and Gong, ICML 2026）（Li, Yang and Mamouras, ASPLOS 2026）（Mamouras, Li and Yang, PLDI 2026）已经开始研究针对预分词和BPE等子程序的在线算法，目标是在输入在线到达时以流式方式将输入转化为输出。然而，如何实现高效的在线分词仍然是一个悬而未决的问题。在本工作中，我们形式化并解决了在线分词问题。我们引入了在线分词“延迟”的概念，用于刻画算法输出落后于（摘要在此处被截断）

    arXiv:2610.19139v1 Announce Type: cross  Abstract: Tokenization is an essential preprocessing step in almost all large language models (LLMs). One of the most popular frameworks is BPE tokenization, which consists of special token segmentation, pretokenization, and byte-pair encoding (BPE). However, existing algorithms are sequential, with each stage only processing complete output from the previous stage. In order to reduce latency and memory footprints, recent works (Jiang and Gong, ICML 2026) (Li, Yang and Mamouras, ASPLOS 2026) (Mamouras, Li and Yang, PLDI 2026) have initiated the study of online algorithms for subroutines such as pretokenization and BPE with a goal of streaming input to output as the input arrives online. However, the search for efficient online tokenization remains open.   In this work, we formalize and resolve online tokenization. We introduce the notion of delay for online tokenization which characterizes the extent to which the algorithm output lags behind the
    
[^250]: MA-Bench：测量语言模型的作战规划能力

    MA-Bench: Measuring Operational Planning in Language Models

    [https://arxiv.org/abs/2610.19138](https://arxiv.org/abs/2610.19138)

    提出了基于军事作战规划第一步“任务分析”构建的MA-Bench基准，用于评估语言模型在缺少唯一正确答案的情况下，从不完整且矛盾的文档集合中形成态势分析报告的能力。

    

    arXiv:2610.19138v1 公告类型：交叉（cross） 摘要：当不存在唯一正确答案时，语言模型能在多大程度上将庞大、不完整且部分矛盾的文档集合转化为一份计划？我们提出了任务分析基准（MA-Bench），该基准建立在军事作战规划的第一个步骤——任务分析之上。在这一步骤中，参谋人员需要通读全部态势信息，并在任何人提出解决方案之前先陈述问题。该基准包含10个由专家构建的场景，每个场景将10至12份文档组成的合成资料包与1,257条由专家编写的通过/失败要求相配对，每条要求均包含其来源、问题类型和重要性级别的信息。一个推理模型评判器在不查看文档的情况下，依据这些要求对每份简报进行评分；其评分表现如同第五位持证规划人员，且不改变由四位规划人员组成的评审小组的评分一致性（Fleiss κ 从0.42变为0.41），而来自另一家提供商的评判器也能重现相同的模型排名（Spearman 0.94）。在二十个前沿模型和开源权重模型中，表现最好的模型得到62%……（原文摘要至此截断）

    arXiv:2610.19138v1 Announce Type: cross  Abstract: How well can a language model turn a large, incomplete and partly contradictory body of documents into a plan when no single correct answer exists? We introduce Mission Analysis (MA)-Bench, built on mission analysis, the first step of military operational planning, in which a staff reads the full situation and states the problem before anyone proposes a solution. Its 10 expert-built scenarios pair synthetic packets of ten to twelve documents with 1,257 expert-written pass/fail requirements, each containing information on their source, question type, and importance level. A reasoning-model judge grades each brief against the requirements without seeing the documents; it grades like a fifth credentialed planner, leaving a four-planner panel's agreement unchanged (Fleiss $\kappa$ 0.42 to 0.41), and a judge from a second provider reproduces the model ranking (Spearman 0.94). Across twenty frontier and open-weights models the best gets 62% 
    
[^251]: 学生Copula马尔可夫网络

    Student Copula Markov Networks

    [https://arxiv.org/abs/2610.19135](https://arxiv.org/abs/2610.19135)

    本文基于copula密度刻画成对马尔可夫性质，提出学生t-Copula马尔可夫网络及相应的ECM算法，将概率图模型扩展至非高斯数据并实现参数估计与图结构选择。

    

    本文将概率图模型的应用范围扩展到了非高斯数据。该文建立了无向概率图模型中成对马尔可夫性质基于copula密度的刻画方式。基于这一刻画，我们推导出了学生t-Copula马尔可夫网络中成对马尔可夫性质的必要条件。文中提供了一种期望-条件最大化算法（ECM），用于实现带惩罚项的极大似然估计和图结构选择。通过模拟研究对该算法的性能进行了评估，结果表明该算法在估计模型参数和恢复底层图结构方面具有良好效果。

    arXiv:2610.19135v1 Announce Type: cross  Abstract: This paper extends the applicability of probabilistic graphical models to non Gaussian data. It establishes a characterization of the pairwise Markov property in undirected probabilistic graphical models in terms of copula densities. Building on this characterization, we derive a necessary condition for the pairwise Markov property in Student copula Markov networks. An Expectation-Conditional Maximization algorithm (ECM) is provided to facilitate the penalized maximum likelihood estimation and graph selection. The performance of the proposed algorithm is evaluated through simulation studies, which demonstrate its effectiveness in estimating model parameters and recovering the underlying graph structure.
    
[^252]: TRACER：面向间接提示注入的轨迹正则化训练

    TRACER: Trace-Regularized Training Against Indirect Prompt Injection

    [https://arxiv.org/abs/2610.19133](https://arxiv.org/abs/2610.19133)

    TRACER提出了一种通道感知的对抗训练方法，通过轨迹正则化与奖励塑形，将RAG场景下的间接提示注入攻击成功率从65.5%大幅降至6.1%，同时保持模型的干净任务准确率。

    

    间接提示注入利用不可信的检索内容、工具输出或智能体消息来覆盖大语言模型的预期行为。我们证明了防御的鲁棒性依赖于通道条件：同一攻击作为后续消息时成功率为18%，但放在检索上下文中时成功率为53%。我们提出了TRACER（带奖励塑形的轨迹正则化对抗通道暴露），这是一种通道感知的对抗训练方法，它通过采样呈现包装形式，并将以答案为锚点、以证据为门控的奖励与ATRC相结合，在注入无效时将被攻击的推理与干净证据对齐。在GSM8K上使用Qwen2.5-3B-Instruct，TRACER将RAG框架下的攻击成功率（ASR）从65.5%降低到6.1%，同时保持78-82%的干净准确率。在3B规模上，RAG+工具的联合训练在各类框架下实现13.9%的平均攻击成功率，而7B模型仅需单个epoch即可达到9.4%。这些结果表明，鲁棒的防御必须在不同的部署通道上进行显式的训练和评估。

    arXiv:2610.19133v1 Announce Type: new  Abstract: Indirect prompt injection uses untrusted retrieved content, tool outputs, or agent messages to override an LLM's intended behavior. We show that defense robustness is channel-conditioned: the same attack succeeds 18% of the time as a follow-up message but 53% when placed in retrieved context. We propose TRACER (Trace-Regularized Adversarial Channel Exposure with Reward-shaping), a channel-aware adversarial training method that samples presentation wrappers and combines an answer-anchored, evidence-gated reward with ATRC, which aligns attacked reasoning with clean evidence when an injection is invalid. On GSM8K with Qwen2.5-3B-Instruct, TRACER reduces RAG-framed ASR from 65.5% to 6.1% while preserving 78-82% clean accuracy. Joint RAG+tool training achieves 13.9% average ASR across framings at 3B scale, while a single epoch at 7B reaches 9.4%. These results demonstrate that robust defenses must explicitly train and evaluate across deployme
    
[^253]: 基于几何视野模型的目标想象以实现策略泛化

    Imagining Goals with Geometric Horizon Models for Policy Generalization

    [https://arxiv.org/abs/2610.19129](https://arxiv.org/abs/2610.19129)

    提出TD-Aug方法，通过从几何视野模型中采样直接想象出组合已有知识可达成的新目标，从而提升策略在零样本目标泛化任务上的表现。

    

    从离线数据中进行训练已经在机器人等领域取得了长足进步，催生出可以零样本直接应用或高效微调以适应下游任务的通用策略。然而，这些策略的泛化能力仍然可能较差，原因既在于建模目标的选择，也在于从静态数据集中学习。在本工作中，我们聚焦于具有挑战性的零样本目标泛化任务，即策略需要在需要组合其已有知识的未见任务上进行评估（组合泛化）。提升策略泛化能力的一个途径是利用世界模型生成新的经验；然而，这种生成方式在较长时程上已被证明十分困难。因此，为缓解这一问题，我们提出了TD-Aug，它从几何视野模型中采样，能够直接想象出通过组合已有知识可以实现的新颖结果。我们证明，通过……

    arXiv:2610.19129v1 Announce Type: new  Abstract: Training from offline data has allowed for substantial progress in domains such as robotics, leading to general-purpose policies that can be easily applied zero-shot or efficiently finetuned for downstream tasks. However, these policies can still have poor generalization, both due to the choice of modeling objective and from learning from a static dataset. In this work, we focus on the challenging task of zero-shot goal generalization, where a policy is evaluated on unseen tasks that require combining its existing knowledge (compositional generalization). An avenue for improving a policy's generalization is by generating new experience with world models; however, such generation has proven difficult for longer horizons. Thus, to alleviate this issue, we propose TD-Aug, which samples from a geometric horizon model and allows for directly imagining novel outcomes that can be achieved by composing existing knowledge. We demonstrate that tra
    
[^254]: PairIT：用于低数据分子优化的自回归Transformer

    PairIT: Autoregressive Transformers for Low-Data Molecule Optimization

    [https://arxiv.org/abs/2610.19123](https://arxiv.org/abs/2610.19123)

    提出了PairIT框架，通过从结构相似分子对中学习来训练SMILES自回归Transformer，实现低数据条件下的种子分子性质优化，并借助提示词在推理时支持多性质联合优化。

    

    生成具有目标性质的新型分子是药物发现中的核心挑战。尽管早期的基础性工作主要集中于从头生成，但实际应用场景需要更复杂的设定：以一个初始种子分子为条件、同时优化多种性质，并在数据稀缺的条件下运行。我们提出了PairIT，这是一个面向SMILES序列自回归Transformer的灵活训练框架，通过从结构相似的分子对中学习来实现基于种子分子的生成。通过利用匹配的分子对，模型学会生成局部的、化学上有意义的编辑，从而改进种子分子的某种性质。提示词进一步扩展了这一框架，可以指定目标性质、期望的改变方向以及大致的改变幅度。多个提示词可以拼接使用，从而在推理阶段实现多性质优化。对于仅靠分子对监督不足以学好的性质，则采用强化学习……（摘要原文在此处截断）

    arXiv:2610.19123v1 Announce Type: new  Abstract: Generating novel molecules with desired properties is a central challenge in drug discovery. Although foundational work focused on de novo generation, practical use cases require a more complex setting: conditioning on an initial seed molecule, optimizing multiple properties, and operating in data-sparse regimes. We introduce PairIT, a flexible training framework for autoregressive transformers over SMILES sequences that enables seeded generation by learning from pairs of structurally similar molecules. By leveraging matched pairs, the model learns to generate local, chemically meaningful edits that improve a property of the seed molecule. Prompt tokens extend this framework by specifying the target property, desired direction, and approximate magnitude of change. Multiple prompt tokens can be concatenated to enable multi-property optimization at inference time. For properties where pair supervision alone is insufficient, reinforcement l
    
[^255]: 通过保持注意力的MLP裁剪解锁推理语言模型的结构化剪枝

    Unlocking Structured Pruning for Reasoning Language Models via Attention-Preserving MLP Trimming

    [https://arxiv.org/abs/2610.19121](https://arxiv.org/abs/2610.19121)

    LayerTrim是一种免训练的结构化剪枝方法，通过识别冗余解码器层并选择性裁剪其MLP特征、同时保留注意力机制和解码器深度，在无需梯度或恢复训练的情况下保持长篇推理能力，相比RESP平均准确率最高提升21.4个百分点，并实现1.24倍的预填充加速和19.7%的GPU峰值内存降低。

    

    结构化剪枝通过移除模型组件来生成更小的稠密模型，从而降低大语言模型的推理成本。然而，现有方法可能导致在AIME等长篇推理任务上的推理性能崩溃。我们提出了LayerTrim，这是一种免训练的结构化剪枝方法，它识别冗余的解码器层，并选择性地裁剪其MLP特征，同时保留注意力机制和解码器深度。LayerTrim是首个无需梯度或恢复训练即可保持长篇推理能力的免训练结构化剪枝方法。在三个推理语言模型和五个基准测试上，LayerTrim在大多数评估设置中优于现有的结构化剪枝方法，相比RESP平均准确率提升最高达21.4个百分点。在Qwen3-14B上，LayerTrim相比稠密模型实现了1.24倍的预填充加速和19.7%的峰值GPU内存降低，同时更好地保留了推理能力。

    arXiv:2610.19121v1 Announce Type: cross  Abstract: Structured pruning reduces LLM inference costs by removing model components to produce smaller dense models. However, existing methods can cause reasoning performance to collapse on long-form reasoning tasks such as AIME. We propose \textit{LayerTrim}, a training-free structured pruning method that identifies redundant decoder layers and selectively trims their MLP features while preserving attention and decoder depth. LayerTrim is the first training-free structured pruning method designed to preserve long-form reasoning without gradients or recovery training. Across three reasoning language models and five benchmarks, LayerTrim outperforms existing structured pruning methods in most evaluated settings, improving average accuracy by up to 21.4 percentage points over RESP. On \texttt{Qwen3-14B}, LayerTrim achieves 1.24$\times$ faster prefill and 19.7\% lower peak GPU memory than the dense model, while better preserving reasoning ability
    
[^256]: Genesis Steering：借助训练数据实现更好的激活引导

    Genesis Steering: Better Activation Steering Through Training Data

    [https://arxiv.org/abs/2610.19114](https://arxiv.org/abs/2610.19114)

    提出Genesis Steering方法，利用模型自身微调数据（或通过成员推断构建的代理数据）生成激活引导向量，并从理论和实验上证明基于训练数据的引导比外部数据集更有效。

    

    激活引导技术可在推理阶段对大语言模型进行控制，但其有效性取决于参考数据的选择。我们提出Genesis Steering方法，从模型自身微调数据中的对比样本推导出“创生向量”。当这些数据不可用时，我们借助成员推断领域的方法识别可能的训练样本并构建代理数据集。我们在理论上建立了两个结果：关于到决策边界的最优位移，以及代理估计偏差的降低。我们在Tülu-3-8B（其微调数据公开可得）上评估了Genesis Steering和基于成员推断的代理构建方法，并以Llama-3.1-8B-Instruct作为对照。研究发现，来自Tülu-3-8B训练数据混合中的安全数据在两个模型上都比外部数据集产生更强的引导效果。在Tülu-3-8B上，各成员推断得分在识别训练样本方面基本一致，且由排名靠前的样本构建的引导向量能够取得……

    arXiv:2610.19114v1 Announce Type: new  Abstract: Activation steering controls large language models at inference time, but its effectiveness depends on the choice of reference data. We introduce genesis steering, which derives a genesis vector from contrastive examples in the model's own fine-tuning data. When these data are unavailable, we draw on the membership inference literature to identify likely training examples and construct a proxy dataset. We establish two theoretical results on optimal displacement to the decision boundary and reduced proxy-estimation bias. We evaluate genesis steering and membership-based proxy construction on T\"ulu-3-8B, whose fine-tuning data are publicly available, and Llama-3.1-8B-Instruct as a control. We find that safety data from T\"ulu-3-8B's training mixture yield stronger steering than external datasets on both models. On T\"ulu-3-8B, membership scores largely agree in identifying training examples, and vectors built from top-ranked examples ach
    
[^257]: 从复制到组合：语言模型如何发展出上下文学习能力

    From Copying to Composition: How Language Models Develop In-Context Learning

    [https://arxiv.org/abs/2610.19113](https://arxiv.org/abs/2610.19113)

    该研究通过受控诊断方法发现，语言模型会从倾向于复制提示中最近示例，逐渐发展为整合提示信息进行组合式推理的上下文学习，且这种转变出现得很早，并可通过特定的合成数据训练加以引导。

    

    上下文学习使语言模型能够根据提示中的指令和示例调整自身行为，但仅凭准确率无法揭示上下文信息是如何控制模型预测的。我们通过一个受控诊断方法研究ICL的习得与可训练性，该方法将“复制提示中最近的示例”与“整合提示中跨上下文的信息”区分开来。在17个模型中，约三分之一的模型倾向于复制答案，少数模型摇摆不定，其余模型则倾向于符合规则的答案，这与更高的准确率相关，且其预测受所请求的值控制而非受最近的冲突示例控制。检查点分析表明，模型在准确率超越随机水平之前很早就开始优先处理与任务相关的提示词元，而非冲突示例。在匹配训练实验中，先定义词尾再查询词尾的合成文档使模型在受控诊断上转向组合式响应。

    arXiv:2610.19113v1 Announce Type: new  Abstract: In-context learning (ICL) allows language models to adapt their behavior from instructions and demonstrations in a prompt, but accuracy alone does not reveal how contextual information controls their predictions. We study the acquisition and trainability of ICL with a controlled diagnostic that separates copying a nearby demonstration from integrating information across the prompt. Across 17 models, about a third prefer the copied answer, a few are undecided, and the rest prefer the rule-consistent answer, which goes with higher accuracy and with control by the requested value rather than the closest conflicting example. Checkpoint analyses show that models begin prioritizing task-relevant prompt tokens over conflicting examples long before their accuracy improves beyond chance level. In matched training experiments, synthetic documents that define endings and then query them shift models toward compositional responses on the controlled 
    
[^258]: 用于学习多治疗方案个体化治疗规则的神经监督聚类方法

    Neural Supervised Clustering for Learning Individualized Treatment Rules with Many Treatments

    [https://arxiv.org/abs/2610.19104](https://arxiv.org/abs/2610.19104)

    提出了一种结合共享非线性协变量表示、低维治疗嵌入与成对聚类融合的神经Q学习框架NSCF，能够从每个治疗臂有限的观测数据中学习多治疗方案下的个体化治疗规则，并生成可解释的治疗分组划分。

    

    在存在多种治疗方案的情况下学习个体化治疗规则（ITR），需要从每个治疗臂有限的观测数据中估计治疗特异性效应。具有相似条件响应的治疗方案本可以共享信息，但线性对比模型可能无法捕捉非线性的异质性。我们提出了非线性监督聚类融合方法（NSCF），这是一个神经Q学习框架，包含共享的非线性协变量表示和低维治疗嵌入。通过成对融合，该方法将学习到的对比效应相似的治疗方案归为一组，从而得到一个个体化治疗规则和一个可解释的划分结构。我们设计了一种交替算法，将神经网络训练、嵌入融合、基于验证集的划分选择以及无惩罚的分组重新拟合相结合。在非线性精确共享的模拟实验中，NSCF相较于强大的神经基准方法降低了遗憾值；扰动分析刻画了硬共享机制的适用边界。在黑色素瘤异种移植应用中，NSCF提高了平均策略价值……

    arXiv:2610.19104v1 Announce Type: cross  Abstract: Learning an individualized treatment rule (ITR) with many treatments requires estimating treatment-specific effects from limited observations per arm. Treatments with similar conditional responses can share information, but linear contrast models may miss nonlinear heterogeneity. We propose nonlinear supervised clustering fusion (NSCF), a neural Q-learning framework with a shared nonlinear covariate representation and low-dimensional treatment embeddings. Pairwise fusion groups treatments with similar learned contrasts, yielding an ITR and an interpretable partition. An alternating algorithm combines neural training, embedding fusion, validation-based partition selection, and unpenalized grouped refitting. NSCF lowers regret relative to a strong neural reference in nonlinear exact-sharing simulations; perturbations characterize the limits of hard sharing. In a melanoma xenograft application, NSCF improves average policy value over the 
    
[^259]: 上下文信用策略优化

    Contextual Credit Policy Optimization

    [https://arxiv.org/abs/2610.19103](https://arxiv.org/abs/2610.19103)

    提出上下文信用策略优化（CCPO），利用基于冻结 LLM 表示和轨迹统计量计算的上下文相似度对观测匹配的同类经验进行加权，从而在稀疏奖励的长时程任务中实现更可靠的步骤级信用分配，并识别成功轨迹中被良好最终结果所掩盖的冗余动作。

    

    基于分组的强化学习使大型语言模型智能体能够在长时程任务中从稀疏的结果奖励中学习。近期方法通过跨轨迹比较观测匹配的经验来推导步骤级信用。然而，相同的观测可能出现在不同的上下文中，且匹配到的同类经验对于某个给定决策并非同等具有参考价值。此外，良好的最终结果可能会掩盖成功轨迹中存在的冗余动作或绕路行为。为解决这两个挑战，我们提出上下文信用策略优化（CCPO）。针对第一个挑战，CCPO 依据上下文相似度对来自其他轨迹的观测匹配同类经验进行加权，该相似度由冻结的 LLM 表示和累积的轨迹统计量计算得出，使得来自相似上下文的同类经验对某一步骤的基线贡献更大。针对第二个挑战，CCPO 将历史信用（即将实际回报与加权同类基线进行比较）与……（原文摘要在此处被截断）

    arXiv:2610.19103v1 Announce Type: new  Abstract: Group-based reinforcement learning enables large language model agents to learn from sparse outcome rewards in long-horizon tasks. Recent methods derive step-level credit by comparing observation-matched experiences across trajectories. However, identical observations can occur in different contexts, and matched peers are not equally informative for a given decision. Moreover, favorable terminal outcomes can obscure redundant actions or detours within successful trajectories. To address these two challenges, we propose Contextual Credit Policy Optimization (CCPO). For the first, CCPO weights observation-matched peers from other trajectories by contextual similarity, computed from frozen LLM representations and accumulated trajectory statistics, allowing peers from similar contexts to contribute more to a step's baseline. For the second, CCPO pairs historical credit, which compares the realized return against the weighted peer baseline, w
    
[^260]: 相似的模型，失灵的胜率：Luce公理在Chatbot竞技场中的失效之处

    Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena

    [https://arxiv.org/abs/2610.19095](https://arxiv.org/abs/2610.19095)

    该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。

    

    Chatbot竞技场使用Bradley-Terry模型对语言模型进行排名，该模型为每个模型赋予单一分数，因而隐含地假设了Luce选择公理。选择理论预言了该公理会在何处失效：近似复制的模型被区分的果断程度会超出其分数所暗示的水平。基于180万条竞技场投票数据，同一模型不同版本之间的投票果断程度是Bradley-Terry预测的1.37倍，同一代不同参数规模之间为1.26倍，同一实验室不同模型之间为1.12倍；一年后的新发布模型也呈现出一致的乘数，且该效应在替代性的平局处理方式、提示词子集与主题划分以及时间检验下依然稳健。两个模型回答在措辞和格式上的相似程度能够追踪这一效应。一个仅含四到六个参数的修正模型在预测留出集投票方面优于Bradley-Terry模型以及拥有数百个参数的模型，这主要归功于对平局的建模；在预先设定的测试中，将该修正模型冻结并应用于新发布的模型时，其收益集中于平局情形，并在成对比较中表现更优。

    arXiv:2610.19095v1 Announce Type: new  Abstract: Chatbot Arena ranks language models with the Bradley-Terry model, which gives each model one score and so assumes Luce's choice axiom. Choice theory predicts where this fails: near-copies are told apart more decisively than their scores imply. On 1.8 million Arena votes, votes between versions of one model are 1.37 times as decisive as Bradley-Terry predicts, between sizes of one generation 1.26 times and between models of one lab 1.12 times; a release a year later gives consistent multipliers, and the effect survives alternative tie treatments, prompt subsets and topics, and timing checks. How alike two models' answers are worded and formatted tracks the effect. A correction with four to six parameters predicts held-out votes better than Bradley-Terry and than models with hundreds of parameters, mostly by modelling ties; frozen and carried to new releases in a test fixed in advance, its gains are concentrated in ties, and better head-to
    
[^261]: 超越总体准确率：提示优化中的题目级转变

    Beyond Aggregate Accuracy: Item-Level Transitions in Prompt Optimization

    [https://arxiv.org/abs/2610.19084](https://arxiv.org/abs/2610.19084)

    该论文提出RFDPO方法，通过追踪提示优化过程中逐题的“恢复、退化与变动”转变，揭示总体准确率所掩盖的题目级变化，并利用这些反馈指导后续的提示修订。

    

    自动提示优化（APO）利用模型响应的反馈来自动改进提示。然而，其有效性通常通过总体分数来衡量，这种方式无法揭示哪些题目由错变对、哪些原本正确的答案变为了错误，以及改进是否能推广到未见过的题目。我们提出了反思式反馈驱动提示优化（RFDPO），这是一种追踪“恢复”（由错变对）、“退化”（由对变错）和“变动”（所有发生变化的结果）的工具。RFDPO 利用这些反馈来指导后续修订，并在整个优化过程中记录提示的变化。我们在涵盖法律、知识、指令遵循、数学和隐私五个领域的基准上评估了 RFDPO，使用 GPT-5 作为优化器，GPT-4o-mini 和 GPT-4.1 Mini 作为求解器。我们的结果表明，总体准确率可能掩盖显著的变化：在九个配置中的五个里，变动量至少是……的五倍（原文摘要在此处被截断）。

    arXiv:2610.19084v1 Announce Type: cross  Abstract: Automatic prompt optimization (APO) uses feedback from model responses to improve prompts automatically. However, its effectiveness is usually measured through overall scores, which do not reveal which questions become correct, which previously correct answers become wrong, or whether improvements extend to unseen questions. We introduce Reflective Feedback-Driven Prompt Optimization (RFDPO), a tool that tracks recoveries (incorrect to correct), regressions (correct to incorrect), and churn (all changed outcomes). RFDPO uses this feedback to guide subsequent revisions and records prompt changes throughout optimization. We evaluate RFDPO across five benchmarks covering law, knowledge, instruction following, mathematics, and privacy, using GPT-5 as the optimizer and GPT-4o-mini and GPT-4.1 Mini as solvers. Our results show that overall accuracy can hide substantial changes: in five of nine configurations, churn was at least five times th
    
[^262]: 通过连续流语言模型反向传播验证器奖励

    Backpropagating Verifier Rewards Through Continuous Flow Language Models

    [https://arxiv.org/abs/2610.19080](https://arxiv.org/abs/2610.19080)

    提出了一种面向连续流语言模型的强化学习后训练框架，通过引入估计中间词元嵌入价值的评论家网络，绕过不可微的解码步骤，实现验证器奖励沿连续动作的路径梯度反向传播，从而以更低方差提升模型推理能力。

    

    强化学习（RL）已显著提升了自回归和离散扩散语言模型的推理能力。生成连续词元嵌入的语言模型正成为一种强有力的替代方案，但针对这类模型的强化学习后训练研究仍然不足。我们提出了一个针对连续流语言模型的后训练框架。这类模型在多个步骤中对词元嵌入进行去噪，并仅在最后一步将其解码为离散词元。为了获得路径梯度——即通过连续动作进行反向传播且方差远低于REINFORCE的梯度——我们引入了一个评论家（critic）网络来估计中间词元嵌入的价值，从而绕过不可微的解码步骤。评论家还必须在策略变化时跟踪价值函数的变化；为此，我们设计了一个能够从每个词元表示预测价值的评论家网络，以改进策略优化。在四个推理任务上的实验表明……（摘要被截断）

    arXiv:2610.19080v1 Announce Type: new  Abstract: Reinforcement learning (RL) has substantially improved the reasoning of autoregressive and discrete-diffusion language models. Language models that generate continuous token embeddings are emerging as a strong alternative, but RL post-training for them remains understudied. We propose \method{}, a post-training framework for continuous flow language models. Such a model denoises token embeddings over multiple steps and decodes them into discrete tokens only at the last step. To obtain pathwise gradients, which backpropagate through continuous actions and have much lower variance than REINFORCE, we introduce a critic that estimates the value of intermediate token embeddings, bypassing the non-differentiable decoding step. The critic must also keep track of the value function as the policy changes; to this end, we design a critic that predicts a value from every token representation and improves policy optimization. Across four reasoning t
    
[^263]: 分层贝叶斯深度学习方法的理论与实践进展

    Theoretical and Practical Advances in Hierarchical Bayesian Deep Learning Methodology

    [https://arxiv.org/abs/2610.19074](https://arxiv.org/abs/2610.19074)

    本文提出多层粒子边缘Metropolis-Hastings（MLPMMH）方法，通过将超参数的伪边缘推断与基于迹类神经网络先验的耦合序贯蒙特卡罗相结合，实现了神经网络参数与超参数的高效联合贝叶斯推断并提供了理论保证。

    

    可靠的不确定性量化仍然是人工智能领域的一个重要挑战。分层贝叶斯方法通过用后验分布表示未知量，为这一问题提供了一个有原则的框架。由于存在强依赖性，对神经网络参数与先验超参数进行联合推断对于传统的马尔可夫链蒙特卡罗（MCMC）方法而言极具挑战性。在本工作中，我们提出了多层粒子边缘Metropolis-Hastings（MLPMMH）方法，该方法将超参数的伪边缘推断与跨网络宽度截断层级的耦合序贯蒙特卡罗（SMC）条件参数推断相结合。该层级结构由具有宽度相关衰减的迹类神经网络（TNN）先验诱导，其动机来自在训练后的神经网络参数中观察到的可压缩结构这一经验性发现。在标准的多层耦合假设下，我们证明了MLPMMH能够实现均方误差（摘要在此处被截断）……

    arXiv:2610.19074v1 Announce Type: new  Abstract: Reliable uncertainty quantification remains an important challenge in AI. Hierarchical Bayesian methods provide a principled framework for this by representing unknown quantities through posterior distributions. Joint inference of neural-network parameters and prior hyperparameters is challenging for conventional Markov chain Monte Carlo (MCMC) methods due to strong dependencies. In this work, we introduce the multilevel particle marginal Metropolis-Hastings (MLPMMH) method, which combines pseudo-marginal inference for the hyperparameters with coupled sequential Monte Carlo (SMC) across hierarchy of network-width truncations for conditional parameter inference. The hierarchy is induced by a trace-class neural network (TNN) prior with width-dependent decay, motivated empirically by the compressible structure observed in trained neural-network parameters. Under standard multilevel coupling assumptions, we establish that MLPMMH achieves mea
    
[^264]: 重新审视生成式检索与稠密检索：一个静态价值先验即可逆转差距

    Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap

    [https://arxiv.org/abs/2610.19065](https://arxiv.org/abs/2610.19065)

    生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。

    

    随着语言模型和语义物品标识符（SID）的不断改进，生成式检索（GR）已成为序列推荐中强有力的候选生成方法。然而，在在线零售候选生成场景中，GR 并非稠密检索（DR）的全面更优替代方案。我们在两个 Amazon Toys 数据集上用不同指标检验了二者的差异：在两种方法共同覆盖的用户中，GR 在最佳相关物品价值上领先，而 DR 在精确目标召回率以及排序列表整体的相关价值上领先。通过深入分析，我们发现流行度对 GR 的价值领先贡献最大。在 DR 上仅通过一步重排加入相同的静态物品价值，即可逆转这一领先，同时保持召回率高于 GR。最后，在一个使用 LLM 判定相关性的工业级内部电商数据集上，我们证明了 140 亿参数的 SID-LLM 仍不如比它小 47 倍的稠密编码器。

    arXiv:2610.19065v1 Announce Type: cross  Abstract: Generative retrieval (GR) has emerged as a strong candidate generator for sequential recommendation as language models and semantic-item identifiers have improved. However, GR is not a uniformly stronger replacement of dense retrieval (DR) in online retail candidate generation. We examine their differences on two Amazon Toys datasets under different metrics. GR leads on best relevant item value among users both methods cover, while DR leads on exact-target recall and relevant value across the ranked list. With deep analysis, we find that popularity contributes the most to GR's measured value lead. Adding the same static item value to DR in one reranking step reverses this lead and keeps recall above GR. Finally, on an industrial-scale internal e-commerce dataset with LLM-judged relevance, we show that a 14B SID-LLM still underperforms a $47\times$ smaller dense encoder.
    
[^265]: 内层优化器如何塑造副本扩展

    How Inner Optimizers Shape Replica Scaling

    [https://arxiv.org/abs/2610.19064](https://arxiv.org/abs/2610.19064)

    本文研究内层优化器对分布式 LLM 训练中副本扩展性能的影响，识别出决定不同内层优化器（如 Muon 与 AdamW）在副本扩展时表现差异的两个关键因素，从而解释了为何某些内层优化器能显著缓解副本数量增加带来的性能下降。

    

    在地理上分散且互连性较差的计算池或数据中心之间训练大语言模型（LLM）已成为备受关注的话题。这促使人们对诸如 DiLoco 等面向 LLM 的局部优化方法展开更多研究，尤其是那些使用局部（或内层）自适应或预条件优化器的方法，这些方法对高性能 LLM 训练至关重要。然而，一个核心挑战依然存在：众所周知，在固定总训练预算的情况下，增加模型副本数量（即副本扩展，replica scaling）会导致性能单调下降。已有工作从实证角度表明，内层优化器的选择（Muon 与 AdamW 之间）可以显著改善这一趋势，但对性能差异的解释却很有限。在这项工作中，我们考察了一系列内层优化器，并识别出影响副本扩展的两个重要因素：（1）梯度协方差结构……（原文摘要在此处截断）

    arXiv:2610.19064v1 Announce Type: new  Abstract: Training large language models (LLMs) across geographically distributed and poorly interconnected compute pools or data centers has become a topic of significant interest. This has led to increased study of local optimization methods such as DiLoco for LLMs, especially those that use local (or inner) adaptive or preconditioned optimizers, which are critical for high-performance LLM training. However, a central challenge remains: it is well known that increasing the number of model replicas (replica scaling) while fixing the total training budget leads to monotonically decreasing performance. Existing work has empirically demonstrated that the choice of inner optimizer (Muon v.s. AdamW) can significantly improve this trend, but offers only limited explanation of the performance difference. In this work, we consider a range of inner-optimizers and identify two important factors that impact replica scaling: (1) the covariance structure of t
    
[^266]: 传染病模型参数估计的保形化物理信息弱回归

    Conformalized Physics-Informed Weak Regression for Parameter Estimation in Epidemic Models

    [https://arxiv.org/abs/2610.19058](https://arxiv.org/abs/2610.19058)

    该论文提出物理信息弱回归（PIWR）方法，通过在时间上积分模型而非对数据求导来估计传染病模型参数，并结合由物理模型自身生成校准集的分割保形预测，为所有参数提供具有有限样本覆盖率保证的无分布同步置信区间。

    

    传染病的房室模型在系数上通常是线性的，这些系数是物理参数或它们的乘积或比率。我们利用这一特性，采用闭式最小二乘法进行估计，并为估计结果提供无分布假设的不确定性量化。物理信息回归（PIR）通过将参数拟合到数据的有限差分导数上来实现；我们提出了物理信息弱回归（PIWR），它改为在时间上对模型进行积分，无需对数据求导，可联合估计初始状态，并具有高阶精度。为了量化不确定性，我们将这两种估计器都纳入分割保形预测框架中，其校准群体由基于物理的模型自身生成，从而为所有参数提供具有有限样本覆盖率保证的同步置信区间。一个残差自适应因子（无需真实值即可计算）可将区间缩放至各个数据集（PIWR-RACC）。在十一个疾病信息神经……的基准模型上（原文摘要在此处截断）

    arXiv:2610.19058v1 Announce Type: cross  Abstract: Compartment models of infectious diseases are often linear in coefficients that are physical parameters or products or ratios of them. We exploit this with closed-form least squares and equip the estimates with distribution-free uncertainty. Physics-informed regression (PIR) fits the parameters to finite-difference derivatives of the data; we introduce physics-informed weak regression (PIWR), which integrates the model in time instead, needs no derivatives of the data, estimates the initial state jointly and is high-order accurate. To quantify uncertainty we wrap both estimators in split conformal prediction, with a calibration population generated by the physics-based model itself, giving simultaneous intervals for all parameters with a finite-sample coverage guarantee. A residual-adaptive factor, computed without ground truth, scales the intervals to each data set (PIWR-RACC). On the eleven benchmark models of disease-informed neural
    
[^267]: 面向复杂进化过程的系统发育推断深度生成模型

    Deep Generative Models for Phylogenetic Inference with Complex Evolutionary Processes

    [https://arxiv.org/abs/2610.19053](https://arxiv.org/abs/2610.19053)

    提出利用深度生成模型进行基于模拟的系统发育推断，以突破传统似然方法在处理复杂进化过程时无法准确高效推断树后验分布的局限。

    

    从序列数据推断系统发育树在理解进化关系中发挥着核心作用，其应用范围涵盖从追踪病原体传播到重建生命演化历史。传统上，研究者使用基于似然的方法（如MCMC）来获得树的后验分布。然而，似然函数仅适用于具有严格假设的简单进化模型。对于更贴近现实的进化模型，使用传统方法进行推断的代价极其高昂、不准确，甚至根本无法实现。在实践中，研究者只能退而求其次，尽可能使用足够灵活的传统方法，即使他们预期该方法存在模型误设。原则上，我们可以采用基于模拟的推断方法：从进化模型中模拟生成树和序列数据，然后训练神经网络在给定序列的条件下预测树的后验分布。然而，如何表示和训练复杂的后验分布是一个数学上的挑战……

    arXiv:2610.19053v1 Announce Type: new  Abstract: Inference of phylogenetic trees from sequences plays a central role in understanding evolutionary relationships, with applications ranging from tracking pathogen spread to reconstructing the history of life. Conventionally, practitioners obtain posteriors over trees using likelihood-based methods like MCMC. However, likelihoods are only available for simple evolutionary models with restrictive assumptions. For realistic evolutionary models, inference using conventional methods is prohibitively expensive, inaccurate, or even impossible. In practice, practitioners settle for using as flexible a conventional method as possible, even though they expect it is misspecified. In principle, we could use simulation-based inference: simulate tree and sequence data from an evolutionary model and train a neural network to predict the posterior over trees given sequences. However it is a mathematical challenge to represent and train complex posterior 
    
[^268]: 独自聪明，集体失明：为什么准确性无法认证多智能体团队中的心智理论

    Smart Alone, Blind Together: Why Accuracy Fails to CertifyTheory of Mind in Multi-Agent Teams

    [https://arxiv.org/abs/2610.19051](https://arxiv.org/abs/2610.19051)

    该论文论证准确性不能作为认证多智能体团队中心智理论模块的标准，因为所需准确性取决于部署场景的错误代价，且模块间错误的相关性会使准确性阈值以可预测的方向失效。

    

    基于对队友推断而采取行动的协调者，通常以这些推断的准确性作为认证标准。我们论证准确性是错误的认证依据，原因有二。首先，行动所需的准确性取决于错误行动的代价，因此这一标准应属于部署场景，而非模块本身。其次，模块的错误很少是独立的，且这种相关性有方向性：正相关的错误在其损害重叠时有益，而在损害叠加时有害，因此准确性阈值会以可预测的方向失效——一旦错误信念相互作用并可能造成伤害，就不存在对每个模块都正确的基于准确性的规则。我们在任务分配、Level-Based Foraging 和 Hanabi 中检验这些预测，使用了注入错误、训练的和贝叶斯模块、五个语言模型以及人类判断。准确性在两个方向上都失效了：在共享错误叠加的 Hanabi 中批准了有害的语言模型部署，同时拒绝了……（原文截断）

    arXiv:2610.19051v1 Announce Type: cross  Abstract: Coordinators that act on what they infer about teammates are usually certified by the accuracy of those inferences. We argue that accuracy is the wrong certificate, for two reasons. First, the accuracy needed to act depends on what a wrong action costs, so the bar belongs to the deployment, not the module. Second, a module's errors are rarely independent, and dependence has a sign: positively associated errors help where their damage overlaps and hurt where it compounds, so an accuracy threshold fails in a predictable direction, and once wrong beliefs interact and can do harm, no rule based on accuracy is right for every module. We test these predictions in task assignment, Level-Based Foraging and Hanabi, with injected errors, trained and Bayesian modules and five language models, and in human judgments. Accuracy fails in both directions, approving harmful language-model deployments in Hanabi, where shared errors compound, and rejecti
    
[^269]: 主动推理中信息能换来什么

    What Information Buys in Active Inference

    [https://arxiv.org/abs/2610.19046](https://arxiv.org/abs/2610.19046)

    本文将主动推理（AIF）与贝叶斯UCBVI置于相同的MDP状态抽象下进行对比，发现AIF的信息增益衰减过快导致探索不足，进而通过对信息增益进行平方根校准，使校准后的AIF达到了√T阶的后悔界。

    

    信息论式的探索是强化学习（RL）方法学研究与计算认知神经科学的主要关注点，而成熟的强化学习理论则基于置信集来构建后悔保证。这种情况留下了一个悬而未决的问题：在何种条件下，自然界中存在的信息驱动探索比成熟强化学习理论的探索更具优势。我们将这两种范式中最先进的方法——主动推理（AIF）与 Bayes-UCBVI——置于同一起跑线上：二者都通过同一个针对具有 $S$ 个状态的表格型 MDP 的状态抽象来行动，且它们的贪婪形式仅在探索项上有所差别。我们发现 AIF 的信息增益衰减得过快，无法充当置信宽度，这使得依赖它进行探索的贪婪智能体被锁定在无法选择最优动作的状态。通过对信息增益进行平方根校准，我们得到了校准后的 AIF（calibrated AIF），它在精确的情况下达到了 $\sqrt T$ 阶的后悔界。

    arXiv:2610.19046v1 Announce Type: new  Abstract: Information-theoretic exploration is of primary interest in methodological studies of reinforcement learning (RL) and computational cognitive neuroscience, while the established RL theory builds regret guarantees on confidence sets. This situation leaves open the conditions under which the information-driven exploration found in nature is more advantageous than the exploration of the established RL theory. We put the state-of-the-art methods of these two paradigms, Active Inference (AIF) and Bayes-UCBVI, on an equal footing: both act through the same state abstraction of a tabular MDP with $S$ states, and their greedy forms differ only in the exploration term. We find that AIF's information gain decays too fast to serve as a confidence width, locking greedy agents that explore by it out of the optimal action. Calibrating the information gain by a square root yields calibrated AIF, which attains regret of order $\sqrt T$ for an exact abst
    
[^270]: 图粗化使计算受限的非凸MAX-CUT问题能够逃离局部最优

    Graph Coarsening Enables Escaping Local Optima in Compute-Constrained Nonconvex MAX-CUT

    [https://arxiv.org/abs/2610.19043](https://arxiv.org/abs/2610.19043)

    该论文提出将近线性时间的谱图粗化方法CMG用于热启动具有优良优化景观的粒子多项式松弛（PPR），使计算受限的非凸MAX-CUT优化能够逃离局部最优，在基准实例上平均达到97.5%的最优解、并在全部Biq Mac测试实例上达到已发表最优解，还可扩展至两万顶点规模并大幅降低内存需求。

    

    由于MAX-CUT的二值优化景观包含指数级数量的局部最优解，该问题对初始化高度敏感。我们研究了粒子多项式松弛（PPR），这是一种基于矩的非凸连续重构方法，具有优越的优化景观。我们使用组合多重网格（CMG）对PPR进行热启动，CMG是一种谱图粗化方法，能够在近线性时间内生成结构感知的初始解，且无需任何训练数据。在t2g20（n=400）这一已知精确最优解的基准实例上，CMG+PPR组合管线平均达到最优解的97.5%，最佳运行达到98.4%。带有离散局部搜索改进的扩展变体，在最近一种神经分支定界方法所评估的全部六十个Biq Mac实例上均达到了已发表的最优解。借助稀疏CSR表示，相同方法可扩展到多达20,000个顶点的G-set实例，最多可将内存需求降低……（摘要原文在此处被截断）

    arXiv:2610.19043v1 Announce Type: cross  Abstract: Because its binary optimization landscape contains exponentially many local optima, MAX-CUT is highly sensitive to initialization. We study Particle Polynomial Relaxation (PPR), a moment-based nonconvex continuous reformulation possessing an advantageous optimization landscape. We warm-start PPR with Combinatorial Multigrid (CMG), a spectral graph-coarsening method that produces structure-aware initial solutions in near-linear time without training data. On \texttt{t2g20} ($n=400$), benchmark instance with known exact optimum, the combined CMG+PPR pipeline reaches $97.5\%$ of optimum on average and $98.4\%$ on the best run. An extended variant with discrete local-search refinement reaches the published optimum on all sixty Biq Mac instances evaluated by a recent neural branch-and-bound method. Using a sparse CSR representation, the same approach scales to G-set instances with up to $20{,}000$ vertices reducing memory requirements by up
    
[^271]: 损失曲率作为扩散模型记忆化的免采样代理指标

    Loss Curvature as a Sampling-Free Proxy for Memorization in Diffusion Models

    [https://arxiv.org/abs/2610.19042](https://arxiv.org/abs/2610.19042)

    该论文提出输入损失曲率（训练损失Hessian矩阵的Frobenius范数）可作为扩散模型记忆化的免采样代理指标，通过在训练中追踪其均值即可预测模型何时开始记忆并复现训练数据。

    

    理解扩散模型为何会记忆其训练数据背后的机制，仍是生成式建模中一个悬而未决的问题。本文从两个视角考察这一问题：(a) 模型层面，即模型在训练过程中何时开始出现记忆化；(b) 样本层面，即单个样本被给定模型记忆的倾向性。我们证明，所谓的输入损失曲率，即扩散模型训练损失Hessian矩阵的Frobenius范数，是量化这两个记忆化视角的有效指标。特别是，我们通过实验证明，在训练过程中持续追踪训练样本与验证样本上平均的输入曲率，可以识别模型何时可能开始复现其训练数据。然而，该指标不足以衡量单个样本被记忆的可能性。为解决……（原文摘要在此处截断）

    arXiv:2610.19042v1 Announce Type: new  Abstract: Understanding the mechanism behind why diffusion models memorize their training data remains an open question in generative modeling. In this paper, we examine this question from two perspectives: (a) the model-level perspective concerning the onset of memorization in a model during training, and (b) the sample-level perspective concerning the proclivity of an individual sample to be memorized by a given model. We show that the so-called input loss curvature, or the Frobenius norm of the Hessian of the diffusion model's training loss, is an effective metric for quantifying both perspectives on memorization. In particular, we empirically demonstrate that input curvatures averaged across training and validation samples can identify when a model is likely to start regurgitating its training data when tracked over training time. However, this metric is not sufficient for measuring an individual sample's likelihood of being memorized. To addr
    
[^272]: 追踪语言习得中的形态、语义与发散

    Tracing Morphology, Semantics, and Divergence in Language Acquisition

    [https://arxiv.org/abs/2610.19041](https://arxiv.org/abs/2610.19041)

    该论文通过对 Pythia 和 OLMo 等开源模型中间训练检查点的系统评估，揭示了语言模型先习得形态结构、语义嵌入随后才趋于稳定的习得轨迹，并发现训练数据规模对发散识别有积极影响。

    

    理解大语言模型（LLM）中语言习得的动态过程，对于揭示统计表示如何演变为结构化语言能力至关重要。然而，现有研究大多聚焦于完全训练好的模型，对中间训练阶段发展轨迹的探索则相对匮乏。在本工作中，我们利用开源模型（如 Pythia 和 OLMo）的中间检查点，在教师强制评估设置下，系统地研究了跨训练步骤与模型规模的语言习得过程。我们重点关注四个核心特征：形态结构习得、嵌入漂移、重复主导性以及共现效应。我们的发现表明，模型在语义嵌入趋于稳定之前就已经较好地习得了形态结构。此外，我们还发现训练数据集规模对发散识别具有积极影响，而共现对……

    arXiv:2610.19041v1 Announce Type: cross  Abstract: Understanding the dynamics of language acquisition in Large Language Models (LLMs) is crucial for uncovering how statistical representations evolve into structured linguistic capabilities. However, most existing studies focus primarily on fully trained models, leaving the developmental trajectory across intermediate training stages largely underexplored. In this work, we leverage intermediate checkpoints from open-source models (e.g., Pythia and OLMo) under teacher-forcing evaluation to systematically investigate language acquisition across training steps and model scales. We focus on four core characteristics: morphological structure acquisition, embedding drift, repetition dominance, and co-occurrence effects. Our findings reveal that models acquire morphological structures well before semantic embeddings stabilize. Furthermore, we show that training dataset size has a positive impact on divergence recognition, while co-occurrence du
    
[^273]: Lag：可复用资源在线分配的平均场最优策略

    Lag: A Mean-Field Optimal Policy for Online Assignment with Reusable Resources

    [https://arxiv.org/abs/2610.19038](https://arxiv.org/abs/2610.19038)

    本文针对可复用资源的在线任务分配问题，将其建模为上下文相关的restless多臂老虎机，提出了闭式指数策略Lag，并通过首个上下文老虎机的确定性流体极限证明了其在新缩放机制下的渐近最优性。

    

    我们研究具有可复用资源的在线任务分配问题。该问题出现在同行评审分配、基于技能的呼叫路由、网约车匹配、紧急调度和众包等场景中：任务按照几何过程到达，必须立即被分配给会暂时忙碌的服务器，产生的收益取决于任务-服务器对，目标是最大化长期平均收益。我们将该问题建模为上下文相关的restless多臂老虎机（CRMAB）。我们推导出一个占用测度线性规划（OCC-LP），它为任何工作保持策略的收益提供了上界。从其对偶问题中，我们得到了Lag——一个闭式指数策略。我们引入了一种缩放机制，其中每个服务器的副本数量与每步的到达数量同步增长。在该机制下，我们得到了首个上下文老虎机的确定性流体极限，并证明Lag在该机制下是渐近最优的。最后，我们在同行评审数据上评估了我们的算法。

    arXiv:2610.19038v1 Announce Type: new  Abstract: We study online task assignment with reusable resources. This problem arises in peer reviewer assignment, skill-based call routing, rideshare matching, emergency dispatch, and crowdsourcing: Tasks arrive according to a geometric process and must be immediately assigned to servers that become temporarily busy, yielding rewards that depend on the task-server pair. The goal is to maximize long-run average reward.   We cast the problem as a contextual restless multi-armed bandit (CRMAB). We derive an occupancy-measure LP (OCC-LP) that upper-bounds the reward of any work-conserving policy. From its dual we obtain Lag, a closed-form index policy. We introduce a scaling regime in which the number of copies of each server and the number of arrivals per step grow together. This regime yields the first deterministic fluid limit for a contextual bandit. In it, Lag is asymptotically optimal.   Finally, we evaluate our algorithms on peer review datas
    
[^274]: 我们应该更多地倾听谁的声音？面向多元对齐的福利感知偏好获取

    Who Should We Listen to More? Welfare-Aware Preference Acquisition for Pluralistic Alignment

    [https://arxiv.org/abs/2610.19035](https://arxiv.org/abs/2610.19035)

    提出福利感知偏好获取（WAPA）方法，通过基于个体监督效用的社会福祉目标在人群中分配有限的人类反馈，解决多元对齐中因AI模拟精度差异导致的偏好代表性不足问题。

    

    多元对齐需要在有限的人类反馈下容纳多样化的偏好。使用AI模拟器来标注偏好比较可以降低标注成本，但不同个体之间模拟精度的差异可能导致某些人的偏好在训练中代表性不足。我们研究了如何在人群之间分配人类反馈以构建多元对齐的训练数据，以及这种分配如何影响训练得到的策略。为了使这种分配明确且可控，我们定义了一个基于个体监督效用的社会福祉目标，该效用衡量标注质量。每个查询的边际福祉增益取决于其预期的监督改进、接收者的当前效用以及应用指定的优先级。福利感知偏好获取（WAPA）使用辅助参考模型来估计查询增益和个体效用，然后根据估计的边际福祉增益来选择查询……

    arXiv:2610.19035v1 Announce Type: cross  Abstract: Pluralistic alignment needs to accommodate diverse preferences under limited human feedback. Using AI simulators to label preference comparisons reduces annotation costs, but variation in simulation accuracy across individuals can leave some people's preferences poorly represented in training. We study how to allocate human feedback across people to construct training data for pluralistic alignment, and how this allocation affects the trained policy. To make this allocation explicit and controllable, we define a social welfare objective over individual supervision utilities that measure label quality. Each query's marginal welfare gain depends on its expected supervision improvement, the recipient's current utility, and application-designated priority. Welfare-Aware Preference Acquisition (WAPA) uses auxiliary reference models to estimate query gains and individual utilities, then selects queries by estimated marginal welfare gain whil
    
[^275]: 一个Token需要多少秩？——面向秩一LoRA专家的噪底静噪方法

    How Many Ranks Does a Token Need? A Noise-Floor Squelch for Rank-One LoRA Experts

    [https://arxiv.org/abs/2610.19033](https://arxiv.org/abs/2610.19033)

    提出无需路由器的秩一LoRA专家混合方法SQUELCH，通过基于噪底的软阈值处理和Stein无偏风险估计，让每个token自动决定所需的秩数量。

    

    低秩适配在每次更新中将其所有秩一分量应用于每个token，而其混合专家变体则改用学习到的路由器来激活固定数量的分量。我们认为，一个token所需的分量数量是一个具有经典答案的检测问题：训练后的LoRA的秩一分量对token的响应表现为高斯噪底之上的少数几个较大值，噪底宽度与未训练适配器的宽度一致，且噪底承载的是串扰而非适配信息。我们提出SQUELCH，一种无需路由器的秩一专家混合方法，它： 以每个token自身鲁棒噪声水平的倍数对其响应进行软阈值处理，从而使激活的秩随token而变化； 通过Stein无偏风险估计为每个模块选择该倍数，因此无需指定秩预算； 将由此得到的边缘式保留或丢弃决策替换为在值Gram矩阵上的一次解释消去步骤。

    arXiv:2610.19033v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) applies every rank-one component of its update to every token, and mixture-of-experts variants replace this with a learned router that activates a fixed number of components. We argue that the number of components a token needs is a detection question with a classical answer. A token's responses to the rank-one components of a trained LoRA consist of a few large values on a Gaussian noise floor whose width matches that of an untrained adapter, and the floor carries cross-talk rather than adaptation. We introduce SQUELCH, a router-free mixture of rank-one experts that (i) soft-thresholds each token's responses at a multiple of its own robust noise level, so the active rank varies per token; (ii) chooses that multiple for each module by Stein's unbiased risk estimate, so no rank budget is specified; and (iii) replaces the resulting marginal keep-or-drop decisions by one explaining-away step on the value Gram matr
    
[^276]: 为一优化，为众可行：机会约束规划中优化与可行性的解耦

    Optimization for One, Feasibility for Most: Decoupling Optimization from Feasibility in Chance-Constrained Programs

    [https://arxiv.org/abs/2610.19027](https://arxiv.org/abs/2610.19027)

    提出ACCESS框架，将优化与可行性评估解耦——利用大量廉价的单场景子问题生成高质量候选解，再通过大规模批量仿真独立评估可行性，从而在机会约束随机规划中同时兼顾目标函数质量与高概率可行性。

    

    机会约束随机规划（CCSP）要求所求解以高概率满足不确定性约束。标准方法要么通过基于样本平均近似（SAA）的重构将场景直接嵌入优化过程，导致计算成本急剧增长且常常无法找到可行解；要么采用保守重构，虽能保证可行性，但从设计上牺牲了目标函数的质量。我们提出 ACCESS（Adaptive Chance-Constrained Exploration via Structured Sampling，基于结构化采样的自适应机会约束探索），这是一个将优化与可行性评估解耦的结构化采样框架。首先，通过求解大量代价低廉的单场景子问题来生成高质量候选解，从而避免直接在高维受约束解空间中进行采样。其次，通过在海量场景上进行大规模批量仿真来单独评估可行性，这一过程天然是……（摘要在此处截断）

    arXiv:2610.19027v1 Announce Type: cross  Abstract: Chance-constrained stochastic programs (CCSPs) require solutions that satisfy uncertain constraints with high probability. Standard approaches either embed scenarios directly into optimization by considering sample average approximation (SAA)-based reformulations, growing prohibitively expensive and often failing to find feasible solutions; or employ conservative reformulations that guarantee feasibility but sacrifice objective quality by design. We propose ACCESS (Adaptive Chance-Constrained Exploration via Structured Sampling), a structured sampling framework that decouples optimization from feasibility evaluation. First, high-quality candidate solutions are generated by solving many inexpensive single-scenario subproblems, avoiding sampling directly in the constrained high-dimensional solution space. Second, feasibility is assessed separately via large-scale batched simulation over a massive number of scenarios, which is naturally a
    
[^277]: 编程智能体在视觉生成中能走多远？

    How Far Can Coding Agents Go in Visual Generation?

    [https://arxiv.org/abs/2610.19025](https://arxiv.org/abs/2610.19025)

    该研究通过AgentDrawBench基准发现，通用编程智能体通过编写可执行程序生成图像时，擅长保持结构和满足精确视觉约束，在构图准确性重要的领域表现具有竞争力，但在感知丰富性上仍有局限。

    

    现代图像生成在很大程度上依赖于专门的生成模型，这些模型将图像构建过程隐式地嵌入在学习到的参数中。我们探索了一种替代方案：通过可执行程序来构建图像的通用编程智能体。我们在 AgentDrawBench（一个包含200个提示词的基准测试）上对零样本绘图的研究揭示了一种独特的能力特征。智能体擅长保持结构并满足精确的视觉约束，在构图准确性比照片级真实外观更重要的领域取得了有竞争力的结果，但在感知丰富性方面仍然有限。它们的视觉表现力随绘图界面的不同而变化，一项关于重构的定性研究展示了从人类演示中提取的程序化技能如何支持不同的艺术媒介。随后，我们考察了代码作为一种外部视觉思维语言：一种智能体可以构建的结构化、可执行的表示。

    arXiv:2610.19025v1 Announce Type: cross  Abstract: Modern image generation largely relies on specialized generative models that leave the process of image construction implicit in learned parameters. We explore an alternative: general-purpose coding agents that construct images through executable programs. Our study of zero-shot drawing on AgentDrawBench, a 200-prompt benchmark, reveals a distinctive capability profile. Agents excel at preserving structure and satisfying precise visual constraints, achieving competitive results in domains where compositional accuracy matters more than photorealistic appearance, while remaining limited in perceptual richness. Their visual expressiveness varies with the drawing interface, and a qualitative study on reconstruction shows how procedural skills extracted from human demonstrations support different artistic media. We then examine code as an external visual language of thought: a structured, executable representation that agents can construct 
    
[^278]: 稀疏条件下的离散扩散模型是极小极大最优的

    Discrete Diffusion Models are Minimax Optimal Under Sparsity

    [https://arxiv.org/abs/2610.19020](https://arxiv.org/abs/2610.19020)

    该论文证明了在稀疏性假设下，基于正则化得分熵目标的离散扩散模型能够达到极小极大最优的统计速率，且所用正则化方法无需预知稀疏参数即可实现自适应。

    

    离散扩散模型已成为生成文本及其他类别数据时替代自回归模型的领先方案，但其理论基础的发展仍落后于连续扩散模型。对于连续扩散模型，一条重要的理论路线表明，这些模型能够自适应数据中的低维结构，从而提升统计效率与泛化能力。在本工作中，我们通过研究稀疏性假设下离散扩散模型的统计速率，来探索这一现象在离散情形中的对应。通过分析由正则化得分熵目标导出的非参数估计量，我们在稀疏情形下建立了极小极大最优速率。所提出的正则化方法可以看作是对分类任务中标签平滑技术的一种改进，并且不需要预先知道稀疏性参数。这表明，离散扩散模型……

    arXiv:2610.19020v1 Announce Type: cross  Abstract: Discrete diffusion models have emerged as a leading alternative to autoregressive models for generating text and other categorical data, yet their theoretical foundations remain less developed than those of their continuous counterparts. For continuous diffusion models, a prominent line of theory suggests that these models can adapt to low-dimensional structure in the data, thereby improving statistical efficiency and generalization. In this work, we investigate the discrete analogue of this phenomenon by studying the statistical rates of discrete diffusion models under sparsity assumptions. By analyzing a nonparametric estimator derived from a regularized score-entropy objective, we establish minimax-optimal rates in the sparse regime. The proposed regularization can be viewed as an adaptation of label smoothing techniques introduced for classification, and does not require the knowledge of the sparsity parameter. This shows that disc
    
[^279]: 带有先前效用预测的 IR-PRM+：常数遗憾与最优迭代收敛

    IR-PRM+ with Previous-Utility Prediction: Constant Regret and Best-Iterate Convergence

    [https://arxiv.org/abs/2610.19016](https://arxiv.org/abs/2610.19016)

    本文提出了一种基于先前效用预测的新方法，使同步和交替更新的 IR-PRM+ 在有限两人零和矩阵博弈中都能获得常数遗憾保证与最优迭代收敛保证，克服了以往额外梯度分析无法涵盖乐观更新和交替更新的局限。

    

    遗憾匹配（及其变体）和反事实遗憾最小化是实践中求解标准型博弈和扩展型博弈的核心算法。然而，这些算法历来缺乏乐观方法和额外梯度方法所具备的更强理论收敛保证。最近的工作提出了 IREG-PRM+，这是一种确实具有强理论收敛保证的遗憾匹配变体。然而，它依赖于同步的额外梯度更新。IREG-PRM+ 是更一般化的 IR-PRM 框架的一个实例，在该框架中，乐观更新和交替更新在经验上往往具有更好的性能。然而，先前的额外梯度分析无法扩展到这些情形。我们提出了一种能够同时覆盖上述两种设置的新方法。对于有限两人零和矩阵博弈中的同步和交替 IR-PRM+，先前效用预测恢复了此前通过额外梯度方法获得的常数遗憾保证和收敛速度保证……（原文摘要在此处截断）

    arXiv:2610.19016v1 Announce Type: new  Abstract: Regret matching (and its variants) and counterfactual regret minimization are central algorithms in practice for normal-form and extensive-form game solving. Yet these have historically lacked the stronger theoretical convergence guarantees available to optimistic and extra-gradient methods. Recent work introduced IREG-PRM+, a regret matching variant which does have strong theoretical convergence guarantees. However, it relies on a simultaneous, extra-gradient update. IREG-PRM+ is an instance of the more general IR-PRM framework, where empirically optimistic and alternating updates often have better performance. However, the prior extra-gradient analysis does not extend to these cases. We introduce a new approach that covers both settings. For simultaneous and alternating IR-PRM+ in finite two-player zero-sum matrix games, previous-utility prediction recovers the constant-regret and convergence-rate guarantees previously obtained with ex
    
[^280]: 规模化廉价合成预训练提升室内无线电地图预测

    Scaling Cheap Synthetic Pretraining Improves Indoor Radio-Map Prediction

    [https://arxiv.org/abs/2610.19015](https://arxiv.org/abs/2610.19015)

    该论文提出自动室内布局生成器与快速开源近似射线追踪器相结合的数据生成流程，利用1.28亿合成样本预训练视觉模型，使室内无线电地图预测误差平均降低15%，并展现出可预测的数据规模扩展规律。

    

    真实测量数据太过有限，难以训练出能够在各种建筑中准确工作的室内路径损耗预测器。大规模合成预训练提供了一条可行之路，但真实的建筑布局同样稀缺，而且精确的射线追踪仿真器速度慢、成本高且通常为专有软件。我们通过引入一个自动生成多样化室内布局的生成器以及一个快速、开源的近似射线追踪器来解决这两个问题，后者可为这些布局计算无线电地图。利用这一流程，我们生成了1.28亿个合成样本，对ResNet-101和Swin-T编码器-解码器进行预训练，并在高质量基准数据上进行微调。尽管存在仿真器失配问题，我们的ResNet-101模型在五项室内无线电地图预测任务中，相比已知最佳方法平均降低了15%的误差，并提升了在小型真实测量数据集上的迁移性能。两种架构均展现出可预测的数据扩展趋势：在200万至1600万样本上拟合的幂律……（原文摘要至此截断）

    arXiv:2610.19015v1 Announce Type: new  Abstract: Real measurements are too limited to train an accurate indoor pathloss predictor that works across a wide range of buildings. Large-scale synthetic pretraining offers a way forward, but real building layouts are also scarce, and accurate ray-tracing simulators are slow, expensive, and often proprietary. We address both problems by introducing a generator that automatically creates diverse indoor layouts and a fast, open-source approximate ray tracer that computes radio maps for them. Using this pipeline, we generate 128M synthetic samples, pretrain ResNet-101 and Swin-T encoder-decoders, and fine-tune them on high-quality benchmark data. Despite simulator mismatch, our ResNet-101 model reduces error by 15% on average relative to the best known methods across five indoor radio-map prediction tasks and improves transfer on a small real-measurement dataset. Both architectures show predictable data-scaling trends: power laws fitted on 2M-16M
    
[^281]: 图像自蒸馏的数据密度缩放定律

    Data Density Scaling Laws for Image Self-Distillation

    [https://arxiv.org/abs/2610.19013](https://arxiv.org/abs/2610.19013)

    本文发现图像自蒸馏中样本重复的收益依赖于具体数据集，小数据集重复训练甚至可能优于大数据集，并提出同时考虑独特样本数量与每样本重复次数的改进版 Chinchilla 缩放定律。

    

    当前的缩放定律表述认为，增加数据集中独特样本的数量能够提升模型性能，而对相同样本的重复接触所带来的价值会逐渐递减。在本工作中，我们针对图像自蒸馏模型检验了这一假设。通过在多个不同数据集（即 Maxar、SATLAS、Waymo、nuPlan、Ego4D、ImageNet 和 YFCC15M）上进行系统性实验，我们证明样本重复对下游性能的影响并非微不足道，而是依赖于具体的数据集。我们发现在某些设置下，在较小的随机子集上重复训练的效果可以超过在更大的独特样本集上训练的效果。这一发现凸显了现有缩放定律的局限性。为弥补这一空白，我们提出了一个改进版的 Chinchilla 风格缩放定律，该定律明确考虑了两个因素：独特样本的数量以及每个样本的重复次数。这一公式为理解该现象提供了一个框架。

    arXiv:2610.19013v1 Announce Type: new  Abstract: Current scaling law formulations suggest that increasing the number of unique samples in a dataset improves model performance, while repeated exposure to the same samples provides diminishing value. In this work, we examine this assumption for image self-distillation models. Through systematic experiments across diverse datasets, namely Maxar, SATLAS, Waymo, nuPlan, Ego4D, ImageNet, and YFCC15M, we demonstrate that the effects of sample repetition on downstream performance are non-trivial and dataset-dependent. We identify settings where repeatedly training on a smaller random subset can outperform training on a larger set of unique samples. This finding highlights a limitation of existing scaling laws. To address this gap, we propose a modified version of a Chinchilla-style scaling law that explicitly accounts for two factors: the number of unique samples and repetition per sample. This formulation provides a framework for understanding
    
[^282]: PMO-Dock：分子优化中对接、特异性与泛化能力的基准测试

    PMO-Dock: Benchmarking Docking, Specificity, and Generalization in Molecular Optimization

    [https://arxiv.org/abs/2610.19011](https://arxiv.org/abs/2610.19011)

    提出了包含25个基于对接的分子优化任务的PMO-Dock基准测试，涵盖苗头化合物生成、先导化合物优化和特异性评估，并通过将模型开发与最终评估分离的标准化协议，解决了基于对接的分子优化评估缺乏统一标准的问题。

    

    arXiv:2610.19011v1 公告类型：新论文 摘要：实用分子优化基准（PMO）规范了分子优化领域的评估标准，但它建立在简单的基于性质的评估函数之上，无法反映基于结构的药物设计。随着该领域转向基于对接的优化目标，评估实践变得缺乏标准化，不同的方法在不同的对接任务上报告结果，通常缺乏用于公平比较的共享协议或样本效率约束。为解决这一问题，我们提出了面向对接的实用分子优化基准，这是一个基于对接优化的基准测试和评估协议，包含25个任务，涵盖苗头化合物生成、先导化合物优化，以及一项新的特异性任务——该任务要求分子对靶标具有强结合能力，同时对脱靶相互作用进行惩罚。该协议将模型开发与最终评估分离：验证任务用于超参数调优，测试任务则被保留用于最终评估，且在开发过程中不使用……

    arXiv:2610.19011v1 Announce Type: new  Abstract: The Practical Molecular Optimization (PMO) benchmark standardized evaluation in molecular optimization, but it is built on simple property-based oracles that do not capture structure-based drug design. As the field has moved to docking-based objectives, evaluation practice has become unstandardized, and different methods report results on different docking tasks, often without a shared protocol for fair comparison or sample-efficiency constraints. To address this, we introduce Practical Molecular Optimization for Docking (PMO-Dock), a benchmark and protocol for docking-based optimization consisting of 25 tasks covering hit generation, lead optimization, and a new specificity task requiring strong on-target binding while penalizing off-target interactions. The protocol separates model development from final evaluation: validation tasks are used for hyperparameter tuning, and test tasks are held out for final evaluation and are not used du
    
[^283]: 面向扩散式AIGC模型的LoRA适配器中未公开概念的发现方法

    Discovering Undisclosed Concepts in LoRA Adapters for Diffusion-Based AIGC Models

    [https://arxiv.org/abs/2610.18990](https://arxiv.org/abs/2610.18990)

    本文提出一种概念发现机制，通过搜索能同时最大化输出连贯性、LoRA诱导独特性和子空间对齐度的条件输入，来揭示LoRA适配器中被训练生成但未公开的概念，从而支持对AIGC模型的内容审核与调查。

    

    低秩适配（LoRA）技术使得扩散模型的个性化定制变得低成本且得以广泛分享，但当有害或问题内容未被如实披露时，也会带来内容审核风险。因此，亟需开发一种机制：在给定一个LoRA适配器的情况下，发现其被训练用于生成的概念，以支持下游的调查分析。我们通过最能清晰揭示LoRA所学内容的条件输入来刻画目标概念。这样的输入应当能够诱导出连贯的LoRA输出，这些输出既与基础模型的输出存在明显区别，又与LoRA学习到的子空间高度对齐。因此，我们将概念发现形式化为一个搜索问题：寻找一个条件输入，使其能够同时最大化输出连贯性、LoRA带来的独特性以及子空间对齐度。然而，即使目标明确，在开放式的输入空间上进行优化的代价也极其高昂，因为可靠的评估……（摘要原文在此处截断）

    arXiv:2610.18990v1 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) has made diffusion model personalization affordable and widely shareable, but it also creates moderation risks when harmful or problematic content is not faithfully disclosed. Hence, it is desired to develop a mechanism that, given a LoRA adapter, discovers the concept it was trained to generate, so as to support downstream investigation. We characterize the target concept through the conditioning input that most clearly exposes what the LoRA has learned. Such an input should induce coherent LoRA outputs that are distinctive from those of the base model and strongly align with the learned LoRA subspace. We therefore formulate concept discovery as searching for the conditioning input that jointly maximizes output coherence, LoRA-induced distinctiveness, and subspace alignment. Even with a clear objective, however, optimizing it over an open-ended input space is prohibitively expensive, since reliable evaluatio
    
[^284]: 面向钙钛矿太阳能电池外推预测的虚拟合金物理感知语义系统

    Virtual-Alloy Physics-Aware Semantic System for extrapolative prediction of perovskite solar cells

    [https://arxiv.org/abs/2610.18949](https://arxiv.org/abs/2610.18949)

    该研究提出V-PaSS系统，通过维加德定律插值端元化合物晶体图并结合界面能带偏移与科学语言模型，实现了对新型材料钙钛矿太阳能电池的外推性能预测。

    

    现有的钙钛矿太阳能电池机器学习模型难以预测由新材料制成的器件。大多数模型将器件层视为任意的分类标签，并通过无特征的成分向量或需要预先求解晶胞的刚性晶体图来表示钙钛矿吸光层。因此，它们无法泛化到未经记录的合金或不熟悉的电池结构。我们提出了虚拟合金物理感知语义系统（V-PaSS），这是一种围绕钙钛矿固溶体物理连续性构建的架构。V-PaSS通过维加德定律并结合学习到的带隙弯曲效应，对23种纯ABX₃端元化合物的弛豫晶体图进行插值，从而表示任意混合阳离子、混合卤素的吸光层。对于器件叠层，它将层序与电池极性（n-i-p或p-i-n）同界面能带偏移相耦合，并利用科学语言模型对化学描述进行编码。在评估中……

    arXiv:2610.18949v1 Announce Type: cross  Abstract: Existing machine-learning models for perovskite solar cells struggle to predict devices made from new materials. Most treat device layers as arbitrary categorical labels and represent perovskite absorbers either through featureless composition vectors or through rigid crystal graphs that require a pre-solved unit cell. As a result, they cannot generalize to unrecorded alloys or unfamiliar cell architectures. We present the Virtual-Alloy Physics-Aware Semantic System (V-PaSS), an architecture built around the physical continuity of perovskite solid solutions. V-PaSS represents arbitrary mixed-cation, mixed-halide absorbers by interpolating relaxed crystal graphs of 23 pure $ABX_3$ endpoint compounds via Vegard's law with learned bandgap bowing. For the device stack, it couples layer sequence and cell polarity (n-i-p or p-i-n) with interfacial band offsets and encodes chemical descriptions with a scientific language model. Evaluated unde
    
[^285]: 自适应对手下具身智能与智能体AI统计保障的极限

    Limits of Statistical Assurance for Embodied and Agentic AI under Adaptive Adversaries

    [https://arxiv.org/abs/2610.18948](https://arxiv.org/abs/2610.18948)

    本文证明了对于具身智能和智能体AI，任何有限时长的安全认证在原理上都无法排除那些在认证前伪装正常行为、认证后才激活的隐蔽自适应对手，并精确刻画了此类攻击带来的最坏情形期望损失。

    

    有限的安全评估无法排除那些在认证之前保持未受攻击行为、之后才激活的对手。我们针对具身智能和工具使用型AI刻画了这一局限性。一个在可观测、不可撤销的认证之前完全匹配观测规律、并在之后保留干预能力的对手，在 $n$ 次试验中可达到 $v\mathbb{E}[(n-S_0)_+]$ 的预期认证后事件数，其中 $v$ 是激活事件概率，$S_0$ 是未受攻击的认证时间。该结果覆盖了自适应评估器和私有随机化评估器。在无记忆、事件递增干预且不存在约束性干预预算的情况下，在认证时刻切换对于固定似然比检验而言是精确最优的。当干预在授权部署时间范围内提高事件率时，即使认证被推迟到最后期限，正向的清洁认证要求也会导致线性增长的最坏情况预期超额。

    arXiv:2610.18948v1 Announce Type: cross  Abstract: Finite safety assessments cannot exclude an adversary that preserves unattacked behavior until certification and activates afterward. We characterize this limitation for embodied and tool-use AI. An adversary that matches the complete observation law until observable, irrevocable certification and retains intervention access afterward attains $v\mathbb{E}[(n-S_0)_+]$ expected post-certification events over $n$ trials, where $v$ is the active event probability and $S_0$ the unattacked certification time. The result covers adaptive and privately randomized evaluators. Switching at certification is exactly optimal for a fixed likelihood-ratio test under memoryless, event-increasing interventions without a binding intervention budget. A positive clean-certification requirement forces linear worst-case expected excess when intervention raises event rates over an authorized deployment horizon, even if certification is deferred to the deadlin
    
[^286]: 基于自适应高斯混合模型的物理层异常检测

    Physical Layer Anomaly Detection via Adaptive Gaussian Mixture Model

    [https://arxiv.org/abs/2610.18946](https://arxiv.org/abs/2610.18946)

    提出基于自适应高斯混合模型的检测框架，通过学习合法CSI模式并检测统计异常，即使在攻击者与合法节点空间接近的情况下也能有效识别CSI欺骗攻击，在位移超过150毫米时实现95-98%的检测率且误报率仅为1%。

    

    物理层欺骗攻击通过操纵信道状态信息（CSI）对无线定位系统构成严重威胁。当攻击者与合法节点在空间上十分接近时，由于两者的CSI特征变得日益相似，检测此类攻击变得尤为困难。为解决室内场景中的这一挑战，我们提出了一种基于自适应高斯混合模型（GMM）的检测框架，该框架通过学习合法CSI模式并利用统计异常检测来发现空间不一致性。我们使用具有64个均匀线性阵列（ULA）天线和5毫米空间分辨率的大规模多输入多输出（MIMO）实验数据集，针对空间位移范围为50毫米至450毫米的CSI欺骗攻击评估了所提出的检测器。自适应GMM在位移超过150毫米时实现了95-98%的检测率，同时保持1%的误报率。我们对该方法进行了基准测试

    arXiv:2610.18946v1 Announce Type: cross  Abstract: Physical-layer spoofing attacks pose a critical threat to wireless localization systems by manipulating channel state information (CSI). Detecting such attacks becomes particularly challenging when the attacker and the legitimate node are spatially proximate, as their CSI characteristics become increasingly similar. To address this challenge in indoor scenarios, we propose an adaptive Gaussian mixture model (GMM)-based detection framework that learns legitimate CSI patterns and detects spatial inconsistency through statistical anomaly detection. Using a massive multiple-input multiple-output (MIMO) experimental dataset with 64 uniform linear array (ULA) antenna and 5 mm spatial resolution, we evaluated our detector against CSI spoofing attacks with spatial displacements ranging from 50 mm to 450 mm. The adaptive GMM achieves 95-98% detection rate for displacements above 150 mm while maintaining 1% false alarm rate. We benchmark our app
    
[^287]: 具有可复用局部后验的半摊销贝叶斯推断

    Semi-Amortized Bayesian Inference with Reusable Local Posteriors

    [https://arxiv.org/abs/2610.18945](https://arxiv.org/abs/2610.18945)

    该论文提出一种半摊销贝叶斯推断方法，将参数分块并为计算昂贵或频繁更新的块训练可复用的神经估计器来近似局部条件后验，再通过马尔可夫链将这些神经采样与常规更新交替进行，从而高效地实现高维复杂模型的贝叶斯推断。

    

    摊销贝叶斯推断利用模拟来执行贝叶斯估计，适用于似然函数计算代价高昂或无法解析获得的情形。然而，在具有复杂依赖关系的高维模型中，学习完整的联合后验分布极具挑战性。我们提出了一种半摊销方法，将参数划分为较小的块，并利用条件独立性来识别推断每个块所需的信息。对于后验更新计算代价高昂或频繁重复的块，我们为其训练神经估计器。每个估计器从相关模型组件的模拟中学习，在给定相关观测值及其所依赖变量取值的情况下，近似该块的完整条件后验分布。随后，马尔可夫链在直接从这些估计器采样与对剩余变量进行常规更新之间交替进行。每次神经更新都使用当前的条件变量取值，并且既不需要似然……

    arXiv:2610.18945v1 Announce Type: cross  Abstract: Amortized Bayesian inference uses simulations to perform Bayesian estimation when likelihoods are expensive or analytically unavailable. However, learning the full joint posterior is challenging in high-dimensional models with complex dependencies. We introduce a semi-amortized method that divides parameters into smaller blocks, and uses conditional independence to identify the information needed to infer each block. We train neural estimators for blocks whose posterior updates are computationally expensive or frequently repeated. Each estimator learns from simulations of the relevant model components to approximate the block's full conditional posterior, given the relevant observations and the values of the variables on which it depends. A Markov chain then alternates direct draws from these estimators with conventional updates of the remaining variables. Each neural update uses the current conditioning values and requires neither lik
    
[^288]: 随机Volterra方程的Wong–Zakai收敛性与傅里叶神经算子学习

    Wong--Zakai Convergence and FNO Learning for Stochastic Volterra Equations

    [https://arxiv.org/abs/2610.18939](https://arxiv.org/abs/2610.18939)

    本文为随机Volterra积分方程建立了Wong–Zakai收敛定理，并在此基础上证明镜像填充傅里叶神经算子能够学习其解算子，实现对完整解轨迹的准确预测以及向更细时间网格的泛化。

    

    本文研究了描述具有记忆效应的随机系统的随机Volterra积分方程（SVIE）的Wong–Zakai收敛性与傅里叶神经算子学习。文中证明了一个Wong–Zakai定理，建立了由布朗运动的分段线性逼近所驱动的逐路径解向相应SVIE解的收敛性。该收敛性为使用镜像填充傅里叶神经算子来学习解算子奠定了基础，该解算子将初始条件和驱动噪声映射到完整的解轨迹。数值实验展示了具有竞争力的预测精度，以及向比训练所用时间网格更细的网格的泛化能力，同时评估了填充方式和计算成本的影响。作为对\citet{LKP:26}的后续研究，本研究通过对卷积记忆特性量身定制的独立收敛性分析与逼近性分析，将其算子学习框架扩展到了SVIE。

    arXiv:2610.18939v1 Announce Type: cross  Abstract: This paper studies Wong--Zakai convergence and Fourier neural operator learning for stochastic Volterra integral equations (SVIEs) describing random systems with memory. A Wong--Zakai theorem establishes convergence of pathwise solutions driven by piecewise linear approximations of Brownian motion to the corresponding SVIE solution. This convergence provides a foundation for using mirror-padded Fourier neural operators to learn the solution operator mapping initial conditions and driving noise to entire solution trajectories. Numerical experiments demonstrate competitive prediction accuracy and generalization to temporal grids finer than those used for training, while assessing the influence of padding and computational cost. As a follow-up to \citet{LKP:26}, this study extends their operator-learning framework to SVIEs through separate convergence and approximation analyses tailored to convolutional memory.
    
[^289]: 超越方程搜索：基于SCOPE与ClosureBench的猜想引导科学发现

    Beyond Equation Search: Conjecture-Guided Discovery with SCOPE and ClosureBench

    [https://arxiv.org/abs/2610.18938](https://arxiv.org/abs/2610.18938)

    该论文提出SCOPE多智能体框架，将符号回归从优化单个方程转向以科学猜想为核心的研究范式，通过多个科学家智能体独立建模、主动实验与证据交换来检验和修正猜想，并配套ClosureBench基准评估猜想驱动式科学发现。

    

    大多数符号回归方法都在优化候选方程，但提升方程的拟合度并不能确定其背后的科学猜想应该被保留、修改还是摒弃。一个猜想可能存在多种数学实现方式，而精确度相近的方程可能蕴含着不同的解释。这促使研究重心从精炼单个表达式转向发展和检验指导这些表达式的科学猜想。我们提出了SCOPE，一个能够维持多条猜想驱动研究轨迹的框架。多个科学家智能体通过方程建模、问题提出和主动实验独立地探究各自的猜想，随后交换证据以修正其猜想和研究方向。他们的发现被整合为一份科学报告，将模型、结论与证据联系起来。现有的方程发现基准主要评估恢复出的表达式，而缺乏对证据层面的考量。（注：摘要原文在此处被截断）

    arXiv:2610.18938v1 Announce Type: cross  Abstract: Most symbolic regression methods optimize candidate equations, but improving an equation's fit does not establish whether its underlying scientific conjecture should be retained, revised, or rejected. A conjecture may admit multiple mathematical realizations, and similarly accurate equations may imply different explanations. This motivates a shift from refining individual expressions to developing and testing the conjectures that guide them. We introduce SCOPE, a framework that sustains multiple conjecture-driven research trajectories. Multiple scientist agents independently pursue conjectures through equation modeling, question formulation, and active experimentation, then exchange evidence to revise their conjectures and research directions. Their findings are consolidated into a scientific report linking models, conclusions, and evidence. Existing equation-discovery benchmarks primarily assess recovered expressions, leaving the evid
    
[^290]: 延迟复杂度维度：带延迟与删失标签的学习

    Delay Complexity Dimension: Learning with Delayed and Censored Labels

    [https://arxiv.org/abs/2610.18935](https://arxiv.org/abs/2610.18935)

    本文提出“延迟复杂度维度”（DCD），将VC维度与观测信道的标签信息率相结合，刻画了标签延迟且被删失情形下PAC学习的样本复杂度，证明任何学习器需要 Ω(DCD/ε²) 个样本，且伪标签ERM可在常数与对数因子内达到该界。

    

    我们研究二元分类器的PAC学习问题，其中标签在随机延迟后到达，并在观测窗口 $W$ 之外被删失（截断）。这种观测模型需要多少样本？我们提出了一个依赖于分布的答案——延迟复杂度维度。DCD 将VC维度与观测信道的精确标签信息率相结合。任何学习器都需要 $\Omega(\mathrm{DCD}/\varepsilon^2)$ 个样本；在给出该下界的实例族上，一个预言机伪标签ERM（经验风险最小化）在常数和对数因子范围内达到这一速率。在该实例族上，DCD 与经典删失ERM复杂度相差一个常数因子；而在二者分离的情形下，DCD 速率是否可达到仍是开放问题。该理论还给出了一种量化方案，在该方案下延迟信息可降低残余标签方差；提出了一种在其版本空间目标上具有 $(1-1/e)$ 保证的探索策略；以及一个……（原文摘要在此处截断）

    arXiv:2610.18935v1 Announce Type: cross  Abstract: We study PAC learning of binary classifiers when labels arrive after random delays and are censored beyond an observation window $W$. How many samples does this observation model require? We propose a distribution-dependent answer, the Delay Complexity Dimension (DCD). The DCD combines the VC dimension with the exact label-information rate of the observation channel. Every learner needs $\Omega(\mathrm{DCD}/\varepsilon^2)$ samples. An oracle pseudo-label ERM matches this rate on the instance family of the lower bound, up to constants and logarithmic factors. On that family the DCD is within a constant factor of the classical censored-ERM complexity, and whether the DCD rate is achievable where the two separate remains open. The theory also yields a quantization scheme under which delay information lowers the residual label variance, an expedition policy with a $(1-1/e)$ guarantee for its version-space objective, and a phenomenon with n
    
[^291]: 深度伪造语音检测的生理标记初步研究

    An Initial Study of Physiological Markers for Deepfake Speech Detection

    [https://arxiv.org/abs/2610.18934](https://arxiv.org/abs/2610.18934)

    该论文提出利用36个基于人类言语产生系统（呼吸、发声、时间协调）的手工生理标记来检测深度伪造语音，验证了通过识别生理模式不一致性检测伪造语音的可行性，域内最佳等错误率达21.40%，并在跨域场景下展现出一定的泛化能力。

    

    传统的深度伪造语音检测器主要针对合成痕迹，在未见过的攻击或录音条件下性能往往下降。然而，人类语音中包含与呼吸、发声以及言语产生系统的时间协调相关的生理标记，而语音合成器可能无法忠实地再现这些标记。这引出了一个假设：可以通过识别这些模式中的不一致性来检测深度伪造语音。我们使用组织为呼吸、发声和时间三大类的36个手工设计的生理标记来验证这一假设。我们在合并的英语录音上使用逻辑回归和SVM分类器对标记向量进行评估。最佳的域内系统达到了21.40%的等错误率（EER），而逻辑回归在域外数据集对上达到了32.77%。由于这些标记描述了特定的言语产生特性，逻辑回归系数可以指示……（原文在此处截断）

    arXiv:2610.18934v1 Announce Type: cross  Abstract: Conventional deepfake speech detectors primarily target synthesis artifacts and often degrade under unseen attacks or recording conditions. Human speech, however, contains physiological markers associated with respiration, phonation, and temporal coordination arising from the speech-production system that synthesizers may not reproduce faithfully. This motivates the hypothesis that deepfake speech can be detected by identifying inconsistencies in these patterns, which we test using 36 handcrafted physiological markers organized into respiratory, vocal, and temporal categories. We evaluate the marker vectors with logistic regression and SVM classifiers on pooled English recordings. The best in-domain system achieves a 21.40% equal error rate (EER), while logistic regression achieves 32.77% on the out-of-domain dataset pair. Since the markers describe specific speech-production properties, the logistic-regression coefficients indicate wh
    
[^292]: 智能体科学家：编排编程智能体以实现自动化科学（协同）发现

    The Agentic Scientist: Orchestrating Coding Agents for Automated Scientific (Co-)Discovery

    [https://arxiv.org/abs/2610.18933](https://arxiv.org/abs/2610.18933)

    提出了端到端分布式系统“智能体科学家”，将计算研究分解为多个阶段并由新一代编程智能体驱动、结合智能体自验证，实现人机协同的自动化科学发现。

    

    现代编程智能体正在从根本上重塑人类输入在计算研究中的角色：它们能够编写代码、启动计算机模拟实验、分析结果，并以不断延长的时间跨度自主运行。尽管有这些进展，完全自主的研究智能体仍存在构思半径有限、以及目光短浅的爬山式搜索动态等局限。为弥合这一差距，我们提出了“智能体科学家”，这是一个端到端的分布式流水线，使人类研究者能够与现代编程智能体协同开展计算研究。在其前作系统的基础上，我们的系统将研究过程分解为任务起草、想法生成、规划、沙盒化实验、结果汇总、基线与消融实验构建、事后分析以及报告撰写等环节。重要的是，流水线中的每一步都由下一代编程智能体驱动，并利用智能体自验证机制。整体研究过程由（原文在此处截断）……

    arXiv:2610.18933v1 Announce Type: cross  Abstract: Modern coding agents are fundamentally reshaping the role of human input for computational research: they write code, launch in silico experiments, analyze their results, and autonomously operate with ever-increasing time horizons. Despite these developments, fully autonomous research agents suffer from a limited ideation radius and myopic hill-climbing search dynamics. To bridge this gap, we introduce The Agentic Scientist, an end-to-end distributed pipeline for human researchers to conduct computational research alongside modern coding agents. Building on its predecessor, our system decomposes research into task drafting, idea generation, planning, sandboxed experimentation, result aggregation, baseline and ablation construction, post-hoc analysis, and report writing. Importantly, each step in the pipeline is powered by the next generation of coding agents and leverages agentic self-verification. The overall research process is drive
    
[^293]: TorchSpin：基于 PyTorch 的可微分电子顺磁共振模拟

    TorchSpin: Differentiable Electron Paramagnetic Resonance Simulation in PyTorch

    [https://arxiv.org/abs/2610.18924](https://arxiv.org/abs/2610.18924)

    TorchSpin 是一个基于 PyTorch 的开源可微分电子顺磁共振（EPR）模拟框架，它将 EasySpin 的组织结构与数值约定移植到张量计算环境中，在复现经典结果的同时，借助自动微分使 EPR 模拟能够融入机器学习工作流。

    

    解析电子顺磁共振（EPR）谱通常需要对自旋哈密顿量参数进行反复的正向模拟与优化。现有软件提供了成熟的模拟与拟合方法，但这些计算通常无法作为可微分组件融入机器学习工作流中。我们提出了 TorchSpin，这是一个开源的 PyTorch 框架，它将 EasySpin 的组织结构与数值约定移植到基于张量的计算环境中。涵盖连续波 EPR、ENDOR、脉冲 EPR、自旋动力学、轨迹模拟和磁强测量的代表性计算结果与相应的 EasySpin 结果高度吻合。在相同条件下的基准测试中，TorchSpin 在九项工作负载中的六项上 CPU 执行速度快于 EasySpin，但性能在很大程度上取决于具体计算类型。因此，我们将重点放在 PyTorch 自动微分所支持的能力上。在一个 Cu(II) 样例中（原文在此处截断）……

    arXiv:2610.18924v1 Announce Type: cross  Abstract: Interpreting electron paramagnetic resonance (EPR) spectra commonly requires repeated forward simulation and optimization of spin Hamiltonian parameters. Established software provides mature simulation and fitting methods, but these calculations are not generally available as differentiable components within machine-learning workflows. We present TorchSpin, an open-source PyTorch framework that ports the organization and numerical conventions of EasySpin into a tensor-based computational environment. Representative calculations spanning continuous-wave EPR, ENDOR, pulse EPR, spin dynamics, trajectory simulation, and magnetometry closely reproduced corresponding EasySpin results. In matched benchmarks, TorchSpin CPU execution was faster than EasySpin for six of nine workloads although performance depended strongly on the calculation. We therefore focused on the capabilities enabled by automatic differentiation in PyTorch. In a Cu(II) po
    
[^294]: 两个欧洲城市中城市场所的流行度轨迹与混合模式

    Popularity Trajectories and Mixing Patterns of Urban Venues in Two European Cities

    [https://arxiv.org/abs/2610.18917](https://arxiv.org/abs/2610.18917)

    本研究基于伦敦和巴塞罗那约一万个场所5.5年的Foursquare签到数据，发现城市场所的流行度演变可归纳为六种典型轨迹（下降、上升、先升后降、稳定、昙花一现、后起之秀），且不存在U型复苏轨迹，而社区的贫困程度与人口密度并不能预示场所的流行度走向。

    

    诸如餐馆、商店、酒吧和公园等城市场所承载着城市的日常生活，它们的集体兴衰反映着一个社区的活力程度。通常的假设是，一个场所的命运紧随其所在社区的命运，因此一个地区的富裕程度和居民数量将决定该场所的表现。然而，场所之间是否随时间共享某种共同的表现轨迹这一问题尚未得到充分探索。在本研究中，我们使用了伦敦和巴塞罗那约10^4个场所长达5.5年的Foursquare数据。我们发现六种可区分的形态在两个城市中反复出现，分别是：(1) 下降型，(2) 上升型，(3) 先升后降型，(4) 稳定型，(5) 昙花一现型（早期走红但迅速失去人气）以及 (6) 后起之秀型（起步缓慢但后期获得人气）。然而，并未出现恢复型形态，即两个城市中都不存在U型的先降后升群体。此外，我们发现贫困剥夺程度和人口密度并不能指示（摘要在此处被截断）

    arXiv:2610.18917v1 Announce Type: cross  Abstract: Urban venues such as restaurants, shops, bars and parks host the daily life of a city, and their collective rise and fall signals how vibrant a neighbourhood is. The usual assumption is that a venue's fortunes follow its neighbourhood's, so that how wealthy an area is and how many people live there will decide a venue's performance. Yet whether there is a common performance trajectory shared by venues over time is underexplored. In this study, we use 5.5 years of Foursquare data for about 10^4 venues in London and Barcelona. We found six distinguishable shapes recur in both cities, that are (1) Fall, (2) Rise, (3) Rise & Fall, (4) Steady, (5) Ephemeral (an early hit that loses popularity fast) and (6) Underdogs (a slow starter that gains popularity late). However, no recovery shape appears, that is neither city yields a U-shaped fall-and-rise group. Additionally, we found that deprivation and population density are not indicative which
    
[^295]: 流式遥测中区分过程变化、仪器故障与数据通路损坏：参考实现与基准测试

    Distinguishing Process Change, Instrument Fault and Data-Path Corruption in Streaming Telemetry: A Reference Implementation and Benchmark

    [https://arxiv.org/abs/2610.18904](https://arxiv.org/abs/2610.18904)

    提出了一种由多种在线检测器与确定性、可审查的归因规则组合的方法，能在流式遥测中以96.3%的检测率和91.2%的正确归因率区分过程变化、仪器故障与数据通路损坏这三种异常成因。

    

    监测系统的读数偏离预期有三种可能原因：物理过程发生了变化、仪器正在故障，或数值在仪器与软件之间的传输过程中被损坏。这三种情况需要不同的应对措施，但在单个样本中看起来相似。我们阐述了这一三成因问题，证明了单通道无法区分哪些成因对，并提出了一组在线检测器——包括基于新息序列上CUSUM和EWMA控制图的卡尔曼跟踪器、带广义ESD确认的Hampel中位数检测器、带ADWIN的鲁棒噪声统计量、精确值检查以及可选的双通道互补滤波器——并由一个确定性、可审查的归因规则进行组合。这些检测器已合并至一个公开的算法库中。在13种故障类型的2,942个注入事件上，该方法的检测率为96.3%（95%置信区间95.7-97.0），每小时误报0.22次，并从……（摘要截断）

    arXiv:2610.18904v1 Announce Type: cross  Abstract: A reading departs from what a monitoring system expects for one of three reasons: the physical process changed, the instrument is failing, or the value was corrupted between the instrument and the software. The three call for different responses and look alike in a single sample. We state the three-cause problem, prove which pairs of causes one channel cannot separate, and propose a bank of online detectors - a Kalman tracker with CUSUM and EWMA charts on its innovation, a Hampel median with generalized-ESD confirmation, a robust noise statistic with ADWIN, exact-value checks and an optional two-channel complementary filter - combined by a deterministic, inspectable attribution rule. The detectors were merged into a public algorithm library. On 2,942 injected events of thirteen fault types the method detects 96.3% (95% interval 95.7-97.0) at 0.22 false alarms per hour and attributes 91.2% (90.6-91.8) of detected events correctly from o
    
[^296]: AI驱动的时序感知功耗优化

    AI-Driven, Timing-Aware Power Optimization

    [https://arxiv.org/abs/2610.18890](https://arxiv.org/abs/2610.18890)

    提出了一种AI驱动的时序感知功耗优化框架，利用梯度提升模型集成来排序功耗恢复编辑和修复动作，在EDA确定性分析的把关下实现布线后功耗降低，同时避免引入新的约束违例。

    

    布线后功耗恢复需要在时序、电气和物理约束紧密耦合的情况下改变单元选择，而每次改变的价值取决于修复其副作用所需的成本。我们提出一个AI驱动的框架，其中学习到的模型驱动优化的两个阶段：它对要尝试的功耗恢复编辑以及随后的修复动作进行排序，同时由确定性的电子设计自动化（EDA）分析掌控每一次的接受判定。该框架记录基线约束裕量，识别恢复事务在每个分析视图中引入的损害，将预先存在的违例保持在基线水平而不在功耗优化步骤中处理它们，并将修复开销限制在实测节省量的一小部分之内。一个由梯度提升模型组成的集成模型从每次试验前可获得的动作和时序上下文特征预测功耗变化、归一化损害降低和约束回退风险；经校准的

    arXiv:2610.18890v1 Announce Type: cross  Abstract: Post-route power recovery changes cell choices under tightly coupled timing, electrical and physical constraints, and the value of each change depends on what it costs to repair its side effects. We propose an AI-driven framework in which a learned model drives both phases of the optimization: it ranks the powerrecovery edits to try and the repair actions that follow, while deterministic electronic design automation (EDA) analyses govern every acceptance. The framework records baseline constraint margins, identifies the damage a recovery transaction introduces in each analysis view, holds pre-existing violations at their baseline rather than working them inside the power step, and caps repair spending at a fraction of the measured saving. An ensemble of gradient-boosted models predicts power change, normalized damage reduction and constraint-regression risk from action and timing-context features available before each trial; calibrated
    
[^297]: 面向零膨胀连续结果的贝叶斯因果森林

    Bayesian Causal Forests for Zero-Inflated Continuous Outcomes

    [https://arxiv.org/abs/2610.18889](https://arxiv.org/abs/2610.18889)

    本文提出ZIC-BCF-Smear方法，通过解耦的两部分贝叶斯因果森林（Probit BCF建模零值参与、高斯BCF建模正值强度），结合子群体倾向得分调整与Duan非参数平滑估计，有效解决了半连续结果下异质性处理效应估计的难题。

    

    当结果变量为半连续型（即在零处呈现离散点质量，而正值部分呈现偏态的连续分布）时，估计（异质性）处理效应仍然极具挑战性，尽管这种数据类型在现实世界场景中广泛存在。现有方法要么忽略零值点质量，要么施加脆弱的参数化假设，要么面临极高的计算成本和数据增强噪声。本文提出了带有Duan平滑估计的零膨胀连续贝叶斯因果森林，这是一种解耦的两部分非参数估计器：在完整样本上拟合Probit BCF以建模参与（是否取零值），并严格在活跃子群体上拟合高斯BCF以建模正值的强度。子群体倾向得分调整缓解了活跃子群体内部由正则化引起的混杂，而Duan的非参数平滑估计器则将预测重新转换回响应尺度。

    arXiv:2610.18889v1 Announce Type: cross  Abstract: Estimating (heterogeneous) treatment effects is still challenging when the outcome is semicontinuous (i.e., exhibiting both a discrete point mass at zero and a skewed continuous distribution for positive values), despite the strong presence of this data type in real world settings. Existing methods either ignore the zero mass, impose fragile parametric assumptions, or suffer from extreme computational costs and data augmentation noise. This paper proposes Zero-Inflated Continuous Bayesian Causal Forests with Duan's Smearing (ZIC-BCF-Smear), a decoupled two-part nonparametric estimator that fits a Probit BCF on the full sample to model participation and a Gaussian BCF strictly on the active subpopulation to model positive intensity. Subpopulation Propensity Adjustment (SPA) mitigates regularization-induced confounding within the active subpopulation, and Duan's non-parametric smearing estimator re-transforms predictions to the response 
    
[^298]: 面向轴对称锥形空化器自然超空泡几何形态的物理引导多保真度代理建模

    Physics-Guided Multi-Fidelity Surrogate Modeling for Natural Supercavity Geometry of an Axisymmetric Conical Cavitator

    [https://arxiv.org/abs/2610.18886](https://arxiv.org/abs/2610.18886)

    该论文提出一种物理引导的多保真度代理建模方法，用于预测轴对称锥形空化器的自然超空泡几何形态，在解析模型精度不足与多相CFD计算成本过高之间取得平衡，从而支撑超空泡鱼雷空化器的静态设计与实时形状管理。

    

    arXiv:2610.18886v1 公告类型：cross 摘要：锥形空化器后方产生的自然超空泡会形成一个充满蒸汽的包络层，当其在高速水下航行器的整个弹体表面持续存在时，能够降低水动力摩擦阻力，并实现全湿流状态下无法达到的性能水平。准确预测最大空泡长度、最大空泡半径以及局部空泡轮廓随空化数和锥半角变化的函数关系，对于超空泡鱼雷空化器的静态设计和实时形状管理而言都至关重要。经典解析模型虽然计算成本低廉，但在实际工程中非常重要的中等空化数和大锥角工况下会表现出系统性偏差，此时细长体假设开始失效。现有的数据驱动机器学习方法虽然能够提供更高的精度，但受到多相计算流体力学（CFD）参数扫描高昂计算成本的制约。

    arXiv:2610.18886v1 Announce Type: cross  Abstract: Natural supercavitation behind conical cavitators generates a vapor-filled envelope that, when sustained over the full body of a high-speed underwater vehicles, reduces hydrodynamic skin-friction drag and enables performance levels unattainable in fully wetted flow. Accurate prediction of maximum cavity length and maximum cavity radius and local cavity profile as functions of cavitation number and cone half-angle is essential for both the static design and the real-time shape management of super-cavitating torpedo cavitators. Classical analytical models provide computationally inexpensive predictions but exhibit systematic disagreements in the practically important regime of moderate cavitation number and large angle, where slender-body assumptions begin to break down. Existing data-driven machine learning approaches offer higher accuracy but are constrained by the prohibitive cost of multiphase computational fluid dynamics (CFD) sweep
    
[^299]: 基于局部信赖域优化的在线策略蒸馏Token效用

    Token Utility for On-Policy Distillation via Local Trust-Region Optimization

    [https://arxiv.org/abs/2610.18883](https://arxiv.org/abs/2610.18883)

    该论文从蒸馏目标本身出发推导出token效用公式 $s_t=g_t^T F_t^\dagger g_t$，证明其在反向KL下等价于师生对数概率比的学生加权方差、在正向KL下等价于Pearson χ²散度，为在线策略蒸馏的token选择提供了统一且有原则的理论基础。

    

    在线策略蒸馏（OPD）中的Token选择方法使用多种信号对学生访问过的位置进行优先级排序，这些信号包括不确定性、师生分歧和不匹配方向等。我们则从蒸馏目标本身推导token重要性：在学生预测分布上施加相等的局部KL预算时，每个位置的损失能降低多少？答案是效用 $s_t=g_t^\top F_t^\dagger g_t$，其中 $g_t$ 是logit梯度，$F_t$ 是类别Fisher矩阵。对于反向KL，该效用等于师生对数概率比的学生加权方差；对于正向KL，它等于Pearson $\chi^2$ 散度，因此合适的效用度量由目标函数的选择自然决定。在共享的训练检查点上，若干独立提出的准则，包括学生熵、KL分歧和TIP，均将其重要性集中在与所推导出的效用几乎相同的位置。

    arXiv:2610.18883v1 Announce Type: new  Abstract: Token-selection methods for on-policy distillation (OPD) prioritize student-visited positions using diverse signals, including uncertainty, teacher-student disagreement, and mismatch direction. We instead derive token importance from the distillation objective itself: how much can each position's loss decrease under an equal local KL budget on the student's predictive distribution? The answer is the utility $s_t=g_t^\top F_t^\dagger g_t$, where $g_t$ is the logit gradient and $F_t$ is the categorical Fisher matrix. For reverse-KL it equals the student-weighted variance of the student-teacher log-probability ratio, and for forward-KL it equals the Pearson $\chi^2$ divergence, so the appropriate utility follows from the choice of objective. On shared training checkpoints, several independently motivated criteria, including student entropy, KL disagreement, and TIP, concentrate their importance on nearly the same positions as the derived ut
    
[^300]: 遵循法律规定的流程：训练语音代理迁移至未见过的受监管债务催收程序

    Following the Flow the Law Requires: Training Voice Agents to Transfer to Unseen Regulated Procedures in Debt Collection

    [https://arxiv.org/abs/2610.18881](https://arxiv.org/abs/2610.18881)

    本文提出 FlowSpec 机器可读规范，通过在 297 个随机化监管程序上微调 90 亿参数模型，训练出能够零样本遵循从未见过的受监管债务催收流程的语音代理。

    

    基于大语言模型（LLM）构建的语音代理如今可以处理客户来电，但在受监管的应用场景中，仅仅完成任务是不够的：代理必须遵循由法律设定的对话流程。在债务催收中，代理必须在提及债务之前先确认自己正在与债务人通话，表明自身和债权人的身份，说明欠款金额，并避免使用被禁止的语言。每个债权人都会将这些规则转化为自己的流程程序，并不断进行修订。本文探究的核心问题是：能否只训练一次一个小型、自托管的策略模型，然后让它遵循从未见过的受监管程序。每个部署使用一份机器可读的规范 FlowSpec，由它生成策略指令、控制工具调用、提供监督信号并定义测试；代理只能说出经过批准的目录句子。研究团队在来自 297 个随机化程序的 2,506 段对话上微调了一个 90 亿参数的模型，并在四个未见过的部署程序上评估了一个泰语外呼代理（原摘要在此处被截断）。

    arXiv:2610.18881v1 Announce Type: new  Abstract: Voice agents built on large language models (LLMs) now handle customer calls, but in regulated applications completing the task is not enough: the agent must follow a conversational flow set by law. In debt collection, it must confirm it is speaking to the debtor before mentioning the debt, identify itself and the creditor, state the amount, and avoid prohibited language. Every creditor turns these rules into its own procedure and revises it. We ask whether a small, self-hosted policy can be trained once and then follow regulated procedures it has never seen. One machine-readable specification per deployment, FlowSpec, generates the policy instruction, gates the tools, supplies the supervision and defines the tests; the agent speaks only approved catalogue sentences. We fine-tune a 9-billion-parameter model on 2,506 conversations from 297 randomized procedures and evaluate a Thai outbound-call agent on four unseen deployment procedures (
    
[^301]: 扰动预测器能学到基因相互作用吗？——智能体筛选中的外推能力诊断

    Do Perturbation Predictors Learn Gene Interactions? Diagnosing Extrapolation in Agentic Screening

    [https://arxiv.org/abs/2610.18880](https://arxiv.org/abs/2610.18880)

    该论文提出“加性分歧”这一诊断指标来评估扰动预测模型对未见基因组合的外推能力，并发现在八种方法和三个 Perturb-seq 数据集上，专门的神经预测器并不稳定优于简单的加性基线，且偏离加性基线越大的预测误差越大。

    

    高通量扰动筛选正越来越多地使用机器学习模型来预测那些从未被实际测量过的基因组合的效应。一个困难的情形是：测试基因对中的两个基因在训练时都被单独见过，但它们的组合却从未出现。在这种设定下，标准的分布外（OOD）方法往往会失效，因为单独来看任何一个基因都不显得陌生。因此，我们将每个模型的预测与一个简单的加性预测进行比较，后者由两个已观测到的单基因效应相加得到。二者之间的差异——我们称之为加性分歧——衡量了模型为未见组合所引入的额外相互作用效应的大小。在八种预测方法和三个 Perturb-seq 数据集上，专门的神经预测器并不能稳定地优于这一简单的加性基线。更重要的是，偏离加性基线越大的预测往往具有越大的误差。常规的……（摘要原文在此处截断）

    arXiv:2610.18880v1 Announce Type: new  Abstract: High-throughput perturbation screens increasingly use machine-learning models to predict the effects of gene combinations that were never measured. A difficult case arises when both genes in a test pair were seen individually during training, but their combination was not. In this setting, standard OOD methods often fail because neither gene appears unfamiliar on its own. We therefore compare each model prediction with a simple additive prediction obtained by summing the two observed single-gene effects. The difference between them, which we call additive disagreement (AD), measures how much extra interaction effect the model introduces for an unseen combination. Across eight prediction methods and three Perturb-seq datasets, specialised neural predictors do not consistently outperform this simple additive baseline. More importantly, predictions with larger deviations from the additive baseline tend to have larger errors. Conventional ou
    
[^302]: 视觉类增量学习方法能否迁移到蛋白质领域？——面向冻结蛋白质编码器的诊断性基准测试

    Do Vision CIL Methods Transfer to Protein? A Diagnostic Benchmark of Frozen Protein Encoders

    [https://arxiv.org/abs/2610.18879](https://arxiv.org/abs/2610.18879)

    该论文系统评估了九种视觉领域的类增量学习方法在冻结蛋白质编码器上的表现，发现没有任何方法能可靠地跨编码器迁移，其根源在于梯度稳定性信号弱、嵌入几何异质性大以及协方差假设不成立。

    

    摘要：蛋白质功能注释所面对的标签空间不断扩大，这使得类增量学习成为在不进行完全重训的情况下纳入新发现功能的天然框架。然而，现有的类增量学习方法大多是在视觉领域开发的，尚未在冻结的蛋白质表示上进行过系统评估。我们对九种具有代表性的类增量学习方法进行了受控基准测试，涵盖五种基于序列、结构以及序列-结构的蛋白质编码器和两个不同规模的酶功能数据集，同时保持编码器冻结，并确保评估设置在不同表示之间保持一致。我们发现，没有任何一种方法能够可靠地跨编码器迁移，而且这些失败在不同的阶段构建方式下依然存在。诊断分析表明，这种不稳定性源于较弱的基于梯度的稳定性信号、嵌入几何结构中的显著异质性，以及不成立的协方差假设……

    arXiv:2610.18879v1 Announce Type: new  Abstract: Protein function annotation operates over an expanding label space, making class-incremental learning (CIL) a natural framework for incorporating newly discovered functions without full retraining. Yet existing CIL methods have been developed largely in vision and have not been systematically evaluated over frozen protein representations. We conduct a controlled benchmark of nine representative CIL methods across five sequence-, structure-, and sequence--structure-based protein encoders and two enzyme function datasets at different scales, while keeping the encoders frozen and evaluation settings consistent across representations. We find that no method transfers reliably across encoders, and that these failures persist across alternative phase constructions. Diagnostic analyses reveal that this instability arises from weak gradient-based stability signals, substantial heterogeneity in embedding geometry, and covariance assumptions whose
    
[^303]: 基于仅答案数据的高效后训练与验证的逆向推理模型

    Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data

    [https://arxiv.org/abs/2610.18878](https://arxiv.org/abs/2610.18878)

    该论文提出一种自蒸馏方法，通过训练一个“逆向推理模型”根据答案反向生成推理轨迹，从而仅利用提示-答案对数据即可高效后训练推理大语言模型，在数据效率与计算效率之间取得平衡。

    

    arXiv:2610.18878v1 公告类型：新论文 摘要：对推理型大语言模型进行后训练通常需要在昂贵的监督与低效的计算之间做出权衡：监督微调通常需要推理轨迹，这在数据上是低效的，因为在许多应用场景中，人们只有提示-答案对，而缺乏真实的思维链（CoT）序列。另一方面，强化学习（例如使用可验证奖励的方法）虽然可以在没有CoT数据的情况下运行，但在计算上却低效得多，因为它在每次优化步骤中都需要进行多次自回归生成，而这些生成无法保证产生非零的策略梯度。在本工作中，我们探索了一种折中方案——一种仅使用答案数据的自蒸馏技术，将思维链视为隐变量。这依赖于训练一个配套的“逆向推理大语言模型”，该模型能够根据答案生成原始模型的推理轨迹。该逆向模型只需训练一次，即可在后续传入的各个批次中重复使用……

    arXiv:2610.18878v1 Announce Type: new  Abstract: Post-training reasoning LLMs often requires making a tradeoff between costly supervision and compute inefficiency: supervised finetuning typically requires reasoning traces, which is data inefficient since in many applications one only has prompt-answer tuples and lacks the groundtruth chain-of-thought (CoT) sequence. On the other hand, reinforcement learning (with verifiable rewards, for example) can operate without CoT data but it is much more compute-inefficient since it requires multiple autoregressive completions per optimization step, none of which are guaranteed to give a non-zero policy gradient. In this work we explore a middleground, a self-distillation technique that uses answer-only data, treating CoT as latent variables. This relies on training a partner ``inverse reasoning LLM" that generates the original's reasoning traces conditioned on the answer. The inverse model is trained once and reused across incoming batches of an
    
[^304]: Lotus：通过学习预测来学会理解时间序列

    Lotus: Learning to Understand Time-series by Learning to Forecast

    [https://arxiv.org/abs/2610.18877](https://arxiv.org/abs/2610.18877)

    LOTUS 提出“预测锚定”预训练原则，通过要求时间序列的语言对齐表示能够预测未来片段演化，使其在映射到 LLM 嵌入空间时保留时间与频谱信息，从而学会真正理解时间序列。

    

    时间序列语言模型通常使用标签、描述或问答对来将数值信号与语言对齐。尽管这种监督方式有助于形成语言表示，但它并未显式要求这些表示保留底层时间序列信号如何演化的信息。我们提出了预测锚定，这是一种预训练原则，要求与语言对齐的时间序列表示在适应下游任务之前，必须保持对未来演化趋势的预测能力。我们在 LOTUS 中实现了这一原则：该方法将每个时间序列通道划分为不重叠的片段，并将其映射到 LLM 的嵌入空间中；随后，它从时间序列词元的 LLM 隐藏状态预测下一个片段，促使学习到的表示保留时间信息。为了同时捕捉时间变化和周期性模式，我们还提出了一种时间-频谱片段表示，将每个片段的时域与频域特征相结合（原文此处截断）。

    arXiv:2610.18877v1 Announce Type: cross  Abstract: Time-series language models typically align numerical signals with language using labels, captions, or question-answer pairs. Although such supervision encourages linguistic representations, it does not explicitly require them to preserve information about how the underlying time-series signal evolves. We introduce forecast grounding, a pretraining principle that requires language-aligned time-series representations to remain predictive of the future evolution before they are adapted to downstream tasks. We instantiate this principle in LOTUS, which divides each time-series channel into non-overlapping patches and maps them into the LLM embedding space. It then predicts the next patch from the LLM hidden states of the time-series tokens, encouraging the learned representation to retain temporal information. To capture both temporal changes and periodic patterns, we also propose a temporal-spectral patch representation that combines eac
    
[^305]: RoboSimStudio：面向人类与智能体快速实现机器人操作任务Real2Sim的交互式工具包

    RoboSimStudio: An Interactive Toolkit for Human and Agent Rapid Real2Sim of Robot Manipulation Tasks

    [https://arxiv.org/abs/2610.18835](https://arxiv.org/abs/2610.18835)

    该论文提出交互式工具包RoboSimStudio，通过将求解器、接触模型和物理参数的选择显式化且可编辑，使人类与编程智能体能够基于观察到的物理行为快速构建并迭代优化机器人操作任务的Real2Sim仿真。

    

    准确建模机器人与其所操作物体之间交互的物理特性一直十分困难，这些物理特性包括接触力学、耦合接触动力学、形变、刚度与摩擦。我们提出了RoboSimStudio（RSS），这是一个交互式工具包，它将求解器、接触模型和物理参数的选择变得明确且可编辑，使人类和编程智能体都能基于观察到的物理行为快速创建并迭代改进仿真。我们基于开源的Newton物理仿真系统实现了RSS，并通过人类引导的real2sim实验展示了其应用，包括（1）视觉评估、（2）力反馈以及（3）智能体Real2Sim2Real。代码、视频及更多信息可在项目网站获取：https://robosimstudio.github.io

    arXiv:2610.18835v2 Announce Type: cross  Abstract: It has been difficult to accurately model the physics of the interaction between a robot and the objects it manipulates, including contact mechanics, coupled contact dynamics, deformations, stiffness and friction. We present RoboSimStudio (RSS), an interactive toolkit that makes choices of solver, contact model, and physical parameters explicit and editable, allowing both humans and coding agents to quickly create and iteratively refine simulations based on observed physical behavior. We implement RSS with the open Newton physics simulation system and describe experiments with human-guided real2sim using (1) visual evaluation, (2) force-feedback and (3) agentic Real2Sim2Real. Code, video and additional information can be found on the project website: https://robosimstudio.github.io
    
[^306]: 预训练能教会化学语言模型化学吗？

    Does Pretraining Teach Chemical Language Models Chemistry?

    [https://arxiv.org/abs/2610.18651](https://arxiv.org/abs/2610.18651)

    化学语言模型的预训练收益无法随规模扩大，性能低于梯度提升基线，仅有基于理化描述符的有监督预训练能持续带来提升，表明预训练几乎未能让模型真正学到化学知识。

    

    化学语言模型（CLM）在大规模分子语料库上进行预训练，旨在学习可迁移到多种下游任务的表示。本研究使用一系列此类化学语言模型，在64个分子性质预测数据集上系统地研究了预训练的效果。我们发现，化学语言模型预训练带来的下游性能提升并不随规模扩大而增长，且性能在低于梯度提升基线的水平上趋于饱和。在所评估的各种预训练模式中，只有在传统理化描述符上进行有监督预训练才能持续提升所测试化学语言模型的下游性能，而某些形式的预训练甚至会导致模型表现不如随机初始化的编码器。尽管模型能够较好地解析分子结构，但其表现并不优于随机初始化的模型，这表明化学语言模型在预训练中充其量只学到了类似描述符的性质和低级token统计信息，而除此之外学到的化学知识出奇地少。

    arXiv:2610.18651v2 Announce Type: replace-cross  Abstract: Chemical language models (CLMs) are pretrained on large corpora of molecules with the aim of learning representations that transfer to diverse downstream tasks. Here, we systematically study the effects of pretraining using a series of such CLMs across 64 molecular property prediction datasets. We find that the downstream gains of pretraining in CLMs do not scale and performance saturates below gradient-boosting baselines. Across the regimes evaluated, only supervised pretraining on traditional physico-chemical descriptors consistently improves downstream performance for the CLMs tested whereas some forms of pretraining result in models worse than a randomly initialized encoder. While models can resolve molecular structure well, they do so no better than a randomly initialized model, suggesting that, at best, CLMs learn descriptor-like properties in addition to low-level token statistics during pretraining, but surprisingly lit
    
[^307]: 读出缩放塑造稀疏特征恢复

    Readout Scaling Shapes Sparse Feature Recovery

    [https://arxiv.org/abs/2610.18462](https://arxiv.org/abs/2610.18462)

    本文提出读出规范化决策空间（RC-DS）坐标方法以消除读出缩放的奇异值增益，并通过受控实验证明当训练恢复因子生成的原始空间时稀疏特征恢复效果最佳，将平均稀疏因子恢复从0.506提升至0.913。

    

    读出缩放会改变稀疏自编码器（SAE）所学到的内容，但移除它的益处取决于数据。我们使用读出规范化决策空间（RC-DS）坐标来移除奇异值增益，同时保留已保留的决策信息。在固定的读出条件数下，改变生成分布会逆转哪些训练坐标最能恢复已知因子。在我们的受控坐标匹配研究中，当训练恢复了因子生成时所在的空间，恢复效果最好。当因子在读出缩放之前生成时，在条件数为100的情况下，与相同预算下的中心化logit训练相比，RC-DS将平均稀疏因子恢复从0.506提高到0.913。一个精确的包装器以数值精度重现了RC-DS训练，从而分离出训练几何结构而非额外信息的作用。在神经网络策略中，RC-DS在测试预算下改善了平均logit重建。

    arXiv:2610.18462v2 Announce Type: cross  Abstract: Readout scaling changes what a sparse autoencoder (SAE) learns, but the benefit of removing it depends on the data. We use readout-canonicalized decision-space (RC-DS) coordinates to remove singular-value gains while preserving retained decision information. At a fixed readout condition number, changing the generating distribution reverses which training coordinates recover known factors best. In our controlled coordinate-matching study, recovery is highest when training restores the space in which the factors were generated. When factors are generated before readout scaling, RC-DS increases mean sparse-factor recovery from 0.506 to 0.913 at condition number 100, compared with centered-logit training under the same budget. An exact wrapper reproduces RC-DS training to numerical precision, isolating training geometry rather than additional information. In neural policies, RC-DS improves mean logit reconstruction under the tested budgets
    
[^308]: 基于收敛率引导的连续仿真优化期望改进算法设计

    Convergence-Rate-Guided Algorithm Design of Expected Improvement for Continuous Simulation Optimization

    [https://arxiv.org/abs/2610.18434](https://arxiv.org/abs/2610.18434)

    该论文通过推导简单遗憾的高概率上界，在收敛率分析的基础上系统性地设计了随机克里金期望改进算法的三个关键组件——当前最优解、采样重复次数和终端推荐，解决了未知异方差噪声下连续仿真优化中依赖启发式规则进行算法设计的局限。

    

    基于随机克里金替代模型的期望改进方法被广泛应用于连续仿真优化，但仿真输出上未知的异方差噪声使得三个基本设计选择变得至关重要：用于定义改进的当前最优解、每个采样点的重复次数，以及终端推荐。现有方法通常要么假设噪声水平已知，要么通过启发式规则或经验校准分别选择这些组件。我们转而推导简单遗憾的高概率上界，并在一般规则类内系统地选择这三个组件以收紧该上界。我们首先为具有未知和已估计噪声水平的随机克里金预测器建立有限样本置信界，作为三个设计选择的基础。随后我们分析了用于当前最优解和终端推荐的一般后验分位数族。

    arXiv:2610.18434v1 Announce Type: new  Abstract: Expected improvement (EI) based on stochastic kriging (SK) surrogate is widely used in continuous simulation optimization, but unknown heteroscedastic noise on simulation output makes three basic design choices consequential: the incumbent used to define improvement, the replication level at each sampled point, and the terminal recommendation. Existing methods typically either assume the noise levels are known or choose these components separately through heuristic rules or empirical calibration. We instead derive a high-probability upper bound on the simple regret and systematically choose the three components within general rule classes to tighten this bound. We first establish a finite-sample confidence bound for the SK predictor with unknown and estimated noise levels as a foundation for the three design choices. We then analyze general posterior-quantile families for both the incumbent and the terminal recommendation. For the incumb
    
[^309]: ResearchFlow-Eval：追踪端到端多论断自主研究中的执行差距

    ResearchFlow-Eval: Tracing the Execution Gap in End-to-End Multi-Claim Autoresearch

    [https://arxiv.org/abs/2610.18088](https://arxiv.org/abs/2610.18088)

    该论文提出 ResearchFlow-Eval——一个基于已发表论文锚定、在阶段与论断两个层面细粒度评估端到端多论断自主研究系统持续科学执行与迭代能力的评测框架，弥补了现有评估仅打分孤立任务或最终成果的不足。

    

    自主研究系统越来越多地承担起完整的研究项目。在实践中，科学研究要求系统通过持续的、多阶段的工作流程来追求多个相互依赖的论断，并随着证据的积累不断修订实验。然而，现有的自主研究系统评估通常只对孤立任务或最终成果进行打分，使得这些过程层面的能力在很大程度上未被衡量。因此，我们提出了 ResearchFlow-Eval，这是一个在阶段与论断两个层面上评估持续科学执行与迭代的细粒度框架。每项评估都锚定于一篇已发表的论文，从中推导出针对各阶段的评分细则，将每个研究阶段分解为多个维度和分别打分的细则条目。利用这些评分细则，我们对每个研究阶段进行评分，并详细考察实验执行情况，重点关注多论断的完成度以及跨多次尝试的修订。我们将该评估应用于（原文摘要在此处截断）

    arXiv:2610.18088v2 Announce Type: new  Abstract: Autoresearch systems increasingly undertake entire research projects. In practice, scientific research requires systems to pursue multiple interdependent claims through a continuous, multi-stage workflow and revise experiments as evidence accumulates. Existing evaluations of autoresearch systems, however, typically score isolated tasks or final artifacts, leaving these process-level capabilities largely unmeasured. We therefore introduce ResearchFlow-Eval, a fine-grained framework for assessing continuous scientific execution and iteration at both the stage and claim levels. Each evaluation is grounded in a published anchor paper, from which we derive stage-specific rubrics that decompose each research stage into dimensions and separately scored rubric items. Using these rubrics, we score each research stage and examine experimental execution in detail, focusing on multi-claim completion and revisions across attempts. We apply the evalua
    
[^310]: 基于同态加密的金融交易图上隐私保护的图神经网络推理

    Privacy-Preserving GNN Inference on Financial Transaction Graphs via Homomorphic Encryption

    [https://arxiv.org/abs/2610.18054](https://arxiv.org/abs/2610.18054)

    该论文提出了一种基于CKKS同态加密、以边为中心的图神经网络推理框架，使多家银行能够在不共享明文交易特征的前提下协作进行反洗钱检测。

    

    金融网络中的反洗钱（AML）需要对跨越多个机构的交易模式进行建模。然而在实践中，由于严格的隐私法规，银行无法共享账户和交易特征。我们提出了一种专为这种多银行场景量身定制的隐私保护图神经网络（GNN）推理框架。中央服务器以明文形式持有图的连接结构和预训练模型权重，而每家银行则将其交易特征加密保存。随后的推理过程在CKKS同态加密方案下完全基于加密特征进行，因此任何明文交易特征或预测结果都不会暴露给服务器。使这一方法切实可行的关键是一种以边为中心的GNN架构，其中消息传递通过节点-边关联矩阵上的运算来表达。这种表述方式不仅保留了交易网络中带边属性的多重图结构……

    arXiv:2610.18054v2 Announce Type: replace-cross  Abstract: Anti-money laundering (AML) in financial networks requires modeling transaction patterns that span multiple institutions. In practice, however, banks cannot share account and transaction features due to strict privacy regulations. We propose a privacy-preserving graph neural network (GNN) inference framework tailored to this multi-bank setting. A central server holds the graph connectivity and pretrained model weights in plaintext, while each bank keeps its transaction features encrypted. Inference is then carried out entirely on encrypted features under the CKKS homomorphic encryption scheme, so that no plaintext transaction features or predictions are exposed to the server. The key to making this practical is an edge-centric GNN architecture, wherein message passing is expressed through operations on node--edge incidence matrices. This formulation not only preserves the edge-attributed multigraph structure of transaction netw
    
[^311]: 交互即裁判

    Interaction-as-a-Judge

    [https://arxiv.org/abs/2610.17895](https://arxiv.org/abs/2610.17895)

    提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。

    

    编程智能体的任务是产出复杂的交互式应用，例如网站、桌面应用和游戏。人类在评估这些应用时会以交互方式进行，而当前自动化的LLM评估主要集中于代码分析和单元测试。这种不匹配会导致LLM评判者遗漏那些对人类用户而言一目了然的缺陷。我们研究了“交互即裁判”，这是一种由计算机操作智能体（CUA）直接与应用进行交互的评估范式。我们证明“交互即裁判”能够同时改进交互式应用的评估与生成。为此，我们收集并发布了两个互补的数据集：包含人类偏好及对应智能体操作轨迹的6,858对应用，以及带有经人类验证的需求标签的222个应用。相较于标准的非交互式方法，我们发现“交互即裁判”在预测h……（原文摘要在此处截断）

    arXiv:2610.17895v2 Announce Type: cross  Abstract: Coding agents are tasked with producing sophisticated interactive applications such as websites, desktop applications, and games. While humans assess these applications interactively, current automated LLM evaluations primarily focus on code analysis and unit tests. This mismatch can cause LLM judges to miss failures that are immediately apparent to human users. We study Interaction-as-a-Judge, an evaluation paradigm where computer-use agents (CUAs) directly interact with an application. We demonstrate that Interaction-as-a-Judge can improve both evaluation and generation of interactive applications. To this end, we collect and release two complementary datasets: $6,858$ pairs of applications with human preferences and corresponding agent traces, and $222$ applications with human-verified requirement labels. Relative to standard non-interactive approaches, we find that Interaction-as-a-Judge improves evaluation in terms of predicting h
    
[^312]: RT-J：用于上下文高效预测的关系型Transformer大规模预训练

    RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions

    [https://arxiv.org/abs/2610.17867](https://arxiv.org/abs/2610.17867)

    RT-J是一个8500万参数的关系型基础模型，通过在639个网络爬取数据库和LLM生成的预测任务上利用长上下文预训练，仅需数百个上下文样本即可实现准确预测，所需样本量比现有方法减少10倍。

    

    用于表格和关系型数据库的基础模型在各个科学和工业领域的预测任务中应用日益广泛。然而，将现有模型适配到未见过的数据集需要数万个样本。在此，我们推出了RT-J，一个8500万参数的关系型基础模型（RFM），即使只有数百个甚至更少的上下文样本也能做出准确预测。为实现这一目标，我们首先构建了"Join"数据集，这是一个大型数据集合，包含639个网络爬取的数据库以及6,023个由大语言模型生成的预测任务。我们使用长达8,192个数据库单元格的长上下文进行预训练，并通过一种新颖的轻量级随机游走检索器获取上下文样本。最后，我们利用上下文调优和集成方法来扩展测试时计算。在来自RelBench的上下文回归任务上，RT-J平均只需少10倍的样本即可匹敌采用LLM智能体或RDBLearn特征化器以及TabICLv2或Ligh（摘要截断）的强大特征工程流水线。

    arXiv:2610.17867v2 Announce Type: cross  Abstract: Foundation models for tables and relational databases are increasingly used for predictive tasks across various scientific and industrial domains. However, adapting existing models to unseen datasets requires tens of thousands of examples. Here we introduce RT-J, an 85M-parameter relational foundation model (RFM) that makes accurate predictions even with hundreds of in-context examples or fewer. To achieve this, we first assemble the Join, a large collection of 639 web-crawled databases with 6,023 LLM-generated forecasting tasks. We pretrain with long contexts of up to 8,192 database cells, using in-context examples from a novel lightweight random-walk retriever. Finally, we leverage context tuning and ensembling to scale up test-time compute. On in-context regression tasks from RelBench, RT-J requires 10x fewer examples on average to match strong feature engineering pipelines with LLM-agent or RDBLearn featurizers and TabICLv2 or Ligh
    
[^313]: MMTA：基于时间序列、文本与视觉的多模态时间分析基准测试

    MMTA: Benchmarking Multimodal Temporal Analysis with Time Series, Text, and Vision

    [https://arxiv.org/abs/2610.17856](https://arxiv.org/abs/2610.17856)

    MMTA是一个涵盖九个领域、融合时间序列、文本与视觉三种模态的多模态时间分析基准测试，首次系统评估了多模态大语言模型能否将时间序列与真实世界视觉证据相整合的能力。

    

    时间序列分析越来越多地在多模态环境中被研究，然而大多数基准测试仅将时间信号与文本或从同一信号导出的可视化相配对。这使得多模态大语言模型能否将时间序列与真实世界的视觉证据相整合这一问题悬而未决。我们提出了MMTA（多模态时间分析），这是一个包含时间序列、文本和视觉三种模态任务的基准测试，具有多样化的跨模态互补性模式。MMTA涵盖九个领域——机器人、自动驾驶、人类动作、制造、AR辅助、医疗健康、推荐、金融和气候——采用视频+时间序列和图像+时间序列的布局，包含3,524个样本的测试集，涵盖分类和预测任务。该基准将异构的真实世界数据源标准化为统一的共享模式，提供领域特定的提示词、来源文档以及同时对错误答案和生成失败进行惩罚的成功加权评估指标。

    arXiv:2610.17856v2 Announce Type: cross  Abstract: Time-series analysis is increasingly studied in multimodal settings, yet most benchmarks pair temporal signals with text or with visualizations derived from the same signal. This leaves open whether multimodal large language models (MLLMs) can integrate time series with real-world visual evidence. We introduce MMTA (multimodal temporal analysis), a benchmark of tri-modal tasks with diverse cross-modal complementarity patterns among time series, text, and vision. MMTA spans nine domains -- robotics, autonomous driving, human action, manufacturing, AR assistance, healthcare, recommendation, finance, and climate -- under video+TS and image+TS layouts, with a 3,524-sample test split covering classification and forecasting. The benchmark standardizes heterogeneous real-world sources into a shared schema with domain-specific prompts, provenance documentation, and success-weighted metrics that penalize both wrong answers and failed generation
    
[^314]: RELAY：面向基于评分标准强化学习的自适应交接式选择性自蒸馏方法

    RELAY: Selective Self-Distillation with Adaptive Hand-off for Rubric-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.17623](https://arxiv.org/abs/2610.17623)

    RELAY提出一种两阶段训练方法，第一阶段将冻结教师模型的选择性自蒸馏与GRPO结合，并通过监控蒸馏损失的停滞检测器自适应地在最佳时机移除教师模型、切换到纯GRPO训练，从而在基于评分标准的强化学习中超越纯GRPO方法，且停滞检测器跨模型和领域具有良好泛化性。

    

    基于评分标准的强化学习根据多条书面评分标准对回答进行评分，但会将这些标准压缩为单一奖励，这不仅丢弃了评分标准所承载的细节信息，而且当一组回答打平时无法提供GRPO奖励梯度。一种将评分标准作为特权上下文读取的冻结教师模型可以提供token级别的监督，但我们观察到其相对GRPO的增益会出现早期平台期。RELAY利用优势值和奖励来选择正优势以及低分数零优势的rollout，并监控蒸馏损失以决定何时让教师模型退场。第一阶段将冻结教师模型的选择性蒸馏与GRPO相结合；随后停滞检测器移除教师模型，第二阶段仅使用GRPO继续训练。该方法在一个模型-领域对上进行校准后，无需修改即可在另外三个模型-领域对上复用，停滞检测器在第122至171步之间触发。在所有四个设置中，RELAY在所比较的方法中于域内取得最高分数，领先于GRPO在RaR-Medici上的表现

    arXiv:2610.17623v2 Announce Type: cross  Abstract: Rubric-based reinforcement learning grades a response against many written criteria but collapses them into one reward, which discards the detail the rubric carries and gives no GRPO reward gradient when a group's responses tie. A frozen teacher that reads the rubric as privileged context provides token-level supervision, but we observe an early plateau in the gains over GRPO. RELAY uses advantages and rewards to select positive-advantage and low-scoring zero-advantage rollouts, and monitors the distillation loss to decide when to retire the teacher. Stage 1 combines selective distillation from a frozen teacher with GRPO; a plateau detector then removes the teacher, and Stage 2 continues with GRPO alone. Calibrated on one model-domain pair and reused unchanged in three others, the plateau detector fires between steps 122 and 171. RELAY scores highest in-domain among the compared methods in all four settings, ahead of GRPO on RaR-Medici
    
[^315]: 面向极端事件感知时间序列生成的跨视角时频扩散模型

    Cross-View Time-Frequency Diffusion for Extreme-Aware Time-Series Generation

    [https://arxiv.org/abs/2610.17407](https://arxiv.org/abs/2610.17407)

    提出CrossTF框架，利用时间视角与频率视角去噪器之间的极端敏感不对称性，在扩散生成后期窗口进行跨视角调整，从而在不损害整体生成质量的前提下大幅提升时间序列中极端事件的保真度。

    

    生成真实的时间序列需要忠实地再现罕见但影响巨大的极端事件。现有方法通常依赖于专门的表示形式或以极端事件为焦点的训练目标，这可能限制所捕获极端事件的多样性，或损害整体分布的保真度。我们提出了跨视角时频扩散，这是一个利用时间视角与频率视角去噪器之间差异来实现极端事件感知时间序列生成的框架。我们的关键实证发现是“极端敏感的跨视角不对称性”：沿着连接两个视角干净样本估计的方向移动，会显著改变极端事件的保真度，而对整体生成质量的影响却很有限。我们进一步确定了一个后期跨视角窗口，在该窗口中这种差异趋于稳定，且跨视角调整对极端保真度的影响尤为强烈。这些发现启发了一种分阶段优化策略（摘要在此处被截断）。

    arXiv:2610.17407v2 Announce Type: cross  Abstract: Generating realistic time-series requires faithfully reproducing rare but high-impact extreme events. Existing approaches often rely on specialized representations or extreme-focused training objectives, which may restrict the diversity of captured extremes or compromise overall distributional fidelity. We propose Cross-View Time-Frequency Diffusion (CrossTF), a framework that exploits discrepancies between time- and frequency-view denoisers for extreme-aware time-series generation. Our key empirical finding is extreme-sensitive cross-view asymmetry: moving along the direction connecting the two views' clean-sample estimates substantially changes extreme fidelity while having limited effects on overall generation quality. We further identify a late-stage cross-view window in which this discrepancy stabilizes and cross-view adjustments have a particularly strong effect on extreme fidelity. These findings motivate a staged optimization s
    
[^316]: 学习为非凸优化设计凸松弛

    Learning to Design Convex Relaxations for Nonconvex Optimization

    [https://arxiv.org/abs/2610.16948](https://arxiv.org/abs/2610.16948)

    本文提出 AutoConvexRelax 方法，将非凸问题的符号公式表示为三元图并学习策略自动组合符号重写以生成针对具体问题的凸松弛轨迹，同时证明了该策略在明确条件下保持松弛有效性并给出可计算的误差界。

    

    凸松弛通过用能给出有效界的凸代理问题替换困难的非凸问题，使非凸优化变得易于处理。然而，凸松弛的设计在很大程度上仍然依赖人工：人类专家必须决定在何处进行松弛、使用哪种松弛方法，以及值得付出多大的公式规模增长代价。我们证明了这一设计过程是可以被学习的。本文提出的方法 AutoConvexRelax 将非凸问题的符号化公式表示为一个连接变量、符号项及其目标函数或约束位置的三元图（tripartite graph）。随后，该方法学习一个策略，随着公式的演化，将符号重写组合成针对特定问题的松弛轨迹。我们证明，在明确的定义域、出现上下文和表示一致性条件下，由学习到的策略生成的有限动作序列能够保持松弛的有效性。我们进一步为混合松弛建立了可计算的误差界，并……（摘要原文在此处截断）

    arXiv:2610.16948v2 Announce Type: replace  Abstract: Convex relaxations make nonconvex optimization tractable by replacing hard nonconvex problems with convex surrogates that give valid bounds. Yet their design is still largely manual: a human expert must decide where to relax, which relaxation to use, and how much formulation growth is worth paying. We show that this design process can be learned. The method proposed in this paper, AutoConvexRelax, represents the symbolic formulation of a nonconvex problem as a tripartite graph linking variables, symbolic terms, and their objective or constraint positions. It then learns a policy that composes symbolic rewrites into problem-specific relaxation trajectories as the formulation evolves. We prove that finite action sequences generated by the learned policy preserve relaxation validity under explicit domain, occurrence-context, and representation-consistency conditions. We further establish computable error bounds for mixed relaxations and
    
[^317]: DEFlow：深度均衡流最大熵强化学习

    DEFlow: Deep Equilibrium Flow Maximum-Entropy Reinforcement Learning

    [https://arxiv.org/abs/2610.16597](https://arxiv.org/abs/2610.16597)

    DEFlow提出了一种深度均衡流策略，通过隐式欧拉积分将所有中间动作作为均衡系统联合求解，并借助隐式微分与离散变量替换公式实现了无需反向传播的离策略训练，在多目标对接、机械臂操作和MuJoCo基准任务上均取得了约10%的平均相对性能提升。

    

    表达能力强的流策略能够表示多个高价值动作，但在仅用少量积分步骤进行采样的同时保持动作密度的可处理性仍然是一个挑战。我们提出了DEFlow，一种深度均衡流策略，它用隐式欧拉积分取代显式欧拉采样，并将所有中间动作作为一个均衡系统联合求解。我们将隐式微分与针对最终动作密度的离散变量替换公式相结合，使得能够在软演员-评论家（soft actor-critic）框架内进行离策略训练，而无需通过求解器迭代进行反向传播。在四个多目标对接变体上，DEFlow相对最强基线在平均成功覆盖率上取得了约10%的平均相对提升。它还在五个操作任务中的四个上取得了最高的平均时间平均回报，同时在五个MuJoCo任务中保持广泛的竞争力。这些结果支持了隐式流策略的有效性。

    arXiv:2610.16597v2 Announce Type: replace  Abstract: Expressive flow policies can represent multiple high-value actions, but sampling them with few integration steps while retaining tractable action densities remains challenging. We introduce DEFlow, a deep equilibrium flow policy that replaces explicit Euler sampling with implicit Euler integration and solves all intermediate actions jointly as an equilibrium system. We combine implicit differentiation with a discrete change-of-variables formula for the final-action density, enabling off-policy training within soft actor-critic without backpropagating through solver iterations. DEFlow achieves an average relative improvement of approximately 10% in mean successful coverage over the strongest baseline across four Multi-Goal docking variants. It also achieves the highest mean time-averaged return on four of five manipulation tasks, while remaining broadly competitive across five MuJoCo tasks. These results support implicit flow policies
    
[^318]: 迈向模态无关的癫痫发作分类：面向异构脑电图与颅内脑电信号的统一深度学习框架

    Towards Modality-Agnostic Seizure Classification: A Unified Deep Learning Framework for Heterogeneous EEG and Intracranial EEG Signals

    [https://arxiv.org/abs/2610.16596](https://arxiv.org/abs/2610.16596)

    该论文提出了一个统一的深度学习框架，能够整合来自不同受试者和来源的异构EEG与iEEG信号进行模态无关的癫痫发作分类，突破了现有单模态系统的应用局限。

    

    癫痫发作的预测与检测对于癫痫的临床治疗至关重要，仍然是这一神经系统疾病的核心任务。脑电图（EEG）和颅内脑电图因其能够以高时间分辨率捕捉具有判别性的大脑活动而被广泛用于癫痫分析。然而，现有工作通常分别分析EEG或iEEG的癫痫发作活动，其中EEG分析已得到充分研究，而越来越多的工作则因其更高的空间层面特异性和信号质量而采用iEEG。目前，大多数现有的癫痫发作预测系统本质上都是单模态的，这限制了其在各种临床条件下的应用。因此，需要开发一个能够同时利用EEG和iEEG信号分析癫痫发作的通用框架。本文提出了一种新的多模态癫痫发作分类框架，整合了来自不同受试者和来源的异构EEG和iEEG记录。

    arXiv:2610.16596v2 Announce Type: cross  Abstract: Seizure prediction and detection, essential for clinical treatment of epilepsy, remain the crucial task of this neurological disease. EEG and iEEG are widely used to analyze epilepsy due to its capability of capturing discriminative brain activities at high time resolution. However, existing works generally analyze either EEG or iEEG seizure activity separately, where EEG analysis has been well-researched, while more and more works consider iEEG for its higher specificity at the spatial level and signal quality. Currently, most of the existing seizure prediction systems are single-modal in nature, which limits its application to a variety of clinical conditions. Therefore, a common framework to analyze seizures with both EEG and iEEG signals needs to be developed. This paper presents a new framework of multimodal seizure classification integrating the heterogeneous EEG and iEEG recordings from different subjects and sources. A specific
    
[^319]: 谁决定了“评委攻击”的节奏？评委能力与策略-评委差距之争

    Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy-Judge Gap

    [https://arxiv.org/abs/2610.15778](https://arxiv.org/abs/2610.15778)

    该研究发现强化学习“攻击评委”的深度由评委自身能力决定（与评委AUC呈负相关，系列内更大的评委更难被欺骗），而攻击速度反而与“策略-评委能力差距”的解释相矛盾——差距更大的组合被欺骗得更早，且所有组合均未带来真实的保留集准确率提升。

    

    在使用GRPO针对一个冻结的、无参考的LLM（大语言模型）评委进行训练时，强化学习会利用评委的假阳性判断。那么，究竟是策略与评委之间的能力差距，还是评委本身，决定了这种“攻击评委”行为的速度与程度？我们在GSM8K数据集上训练了0.5B和1.5B的Qwen2.5策略，分别对抗四个规模不小于它们的评委，此外1.5B策略还对抗了一个规模更小的评委。事后分析显示，评委的接受率确实上升了，但在12个实验组合中，没有任何一个组合的种子平均保留集贪心准确率提升超过0.04。使用相同配方但以金标准作为奖励的对照组使1.5B策略的准确率提高了+0.054（而对于0.5B策略，金标准奖励同样没有带来可检测的提升）。评委被攻击的深度取决于评委本身：攻击的停滞期与评委的AUC相关（在12个组合上的Spearman相关系数为-0.69），并且在同一系列模型中，更大的评委被欺骗得更浅。而在攻击速度方面，两个预先设定的配对实验与“能力差距”的解释相矛盾：在预设阈值下，差距更大的那个组合反而提前了3.5倍被欺骗。（原文在此处截断）

    arXiv:2610.15778v2 Announce Type: replace-cross  Abstract: Under GRPO against a frozen, reference-free LLM judge, RL exploits the judge's false positives. Does the policy-judge capability gap or the judge itself set how fast and how far? We train 0.5B and 1.5B Qwen2.5 policies on GSM8K against four judges at least as large, and the 1.5B policy also against one smaller judge. In a post-hoc analysis, judge acceptance rose, but no cell of 12 gained more than 0.04 in seed-mean held-out greedy accuracy. A gold-reward control with the same recipe raised the 1.5B policy's accuracy by +0.054 (for the 0.5B policy, gold reward gave no detectable gain either). The depth of judge hacking tracks the judge: the plateau correlates with judge AUC (Spearman -0.69 over 12 cells), and within a family the larger judge is fooled less deeply. For the pace, two pre-specified pairs contradict the gap account: the cell with the larger gap was fooled 3.5x sooner at the pre-specified threshold. With 12 cells, ho
    
[^320]: ProtoPur：面向病理学弱监督实例学习的原型纯度伪标签方法

    ProtoPur: Prototype-Purity Pseudo-Labeling for Weakly Supervised Instance Learning in Pathology

    [https://arxiv.org/abs/2610.15669](https://arxiv.org/abs/2610.15669)

    提出ProtoPur方法，通过基于冻结基础模型特征构建原型区域并估计类别平衡的局部纯度，为高纯度区域分配伪标签，从而实现病理全切片图像的弱监督实例级学习。

    

    在计算病理学中，弱袋级标签广泛可得，而全切片图像的可靠实例级标注仍然昂贵且存在歧义。弱监督多示例学习在袋级分类上表现优异，但基于注意力分数、模型预测、负样本袋或空间优化的实例级监督方法可能无法识别类别特定的局部证据。我们提出了ProtoPur，一种用于病理全切片图像弱监督实例级学习的原型纯度伪标签方法。ProtoPur基于冻结的基础模型特征构建原型区域，并在这些区域上估计类别平衡的局部纯度，进而为与目标类别强相关的高纯度区域分配伪标签。这种“纯度优先”的设计将类别特定证据与阴性或共享的形态学特征区分开来，而覆盖率仅用于确保充足的伪标签分配。

    arXiv:2610.15669v2 Announce Type: replace-cross  Abstract: Weak bag-level labels are widely available in computational pathology, whereas reliable instance-level annotations for whole-slide images remain expensive and ambiguous. Weakly supervised multiple instance learning (MIL) achieves strong bag-level classification, but instance-level supervision from attention scores, model predictions, negative bags, or spatial refinement may fail to identify class-specific local evidence. We propose ProtoPur, a prototype-purity pseudo-labeling method for weakly supervised instance-level learning in pathology whole-slide images. ProtoPur estimates class-balanced local purity over prototype regions constructed from frozen foundation-model features, and assigns pseudo-labels to high-purity regions strongly associated with a target class. This purity-first design separates class-specific evidence from negative or shared morphology, with coverage used only to ensure sufficient pseudo-label assignment
    
[^321]: SuperAI：跨异构基础模型的能力融合

    SuperAI: Merging Capabilities Across Heterogeneous Foundation Models

    [https://arxiv.org/abs/2610.15322](https://arxiv.org/abs/2610.15322)

    提出SuperAI跨架构多模型融合框架，利用共享锚点样本提取任务相关表示、PCA构建同维低维功能子空间，并通过正交Procrustes映射实现异构基础模型之间互补能力的融合迁移。

    

    arXiv:2610.15322v2 公告类型：new 摘要：实现分布于各基础模型之间的知识与能力的复用和组合，是构建可扩展人工智能（AI）系统的重要一步。然而，架构差异与不兼容的表示方式，使得通过统一的跨模型接口来迁移和整合这些能力变得十分困难。为应对这一挑战，我们提出了SuperAI——一个跨架构的多模型融合框架，它通过低维任务相关表示的正交对齐，将多个源模型的互补能力整合到一个宿主模型中。具体而言，我们首先使用共享锚点样本从源模型和宿主模型中提取任务相关表示，并应用主成分分析（PCA）构建维度相同的低维功能子空间。接下来，我们使用正交Procrustes映射将源模型与宿主模型进行对齐，从而实现跨异构架构的能力迁移与融合。

    arXiv:2610.15322v2 Announce Type: new  Abstract: Enabling the reuse and composition of knowledge and capabilities distributed across foundation models is an important step toward building scalable artificial intelligence (AI) systems. However, architectural differences and incompatible representations make it difficult to transfer and integrate these capabilities through a unified cross-model interface. To address this challenge, we propose SuperAI, a cross-architecture multi-model fusion framework that integrates complementary capabilities from multiple source models into a host model through orthogonal alignment of low-dimensional task-relevant representations. Specifically, we first extract task-relevant representations from the source models and the host model using shared anchor samples, and apply principal component analysis (PCA) to construct low-dimensional functional subspaces with the same dimensionality. Next, we use orthogonal Procrustes mappings to align the source and hos
    
[^322]: AlphaRLM：面向基于文献的符号化因子发现的语言模型对齐

    AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery

    [https://arxiv.org/abs/2610.14959](https://arxiv.org/abs/2610.14959)

    AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。

    

    公式化Alpha挖掘旨在从庞大且充满噪声的搜索空间中发现能够捕捉预测性市场信号的可执行表达式。近期基于大语言模型（LLM）的方法主要依赖提示工程、迭代搜索或种子因子进化，而通过策略学习直接构造因子的路径较少被探索。为解决这一问题，我们提出了AlphaRLM，一个强化学习框架，用于训练大语言模型在无需输入因子表达式的情况下，直接从金融文献中构造可执行的符号化因子。AlphaRLM引入了文献引导算子组装机制，将因子构造建立在文献衍生的思想与可执行算子之上，为探索提供语义先验。它还通过可执行有效性奖励和分组收益分离奖励（后者能够捕捉因子各分位数间的非单调结构）将生成策略与市场反馈对齐。在沪深300、中证1000和标普500上的大量实验表明，AlphaRLM……（摘要至此截断）

    arXiv:2610.14959v2 Announce Type: new  Abstract: Formulaic alpha mining seeks to discover executable expressions that capture predictive market signals in a vast, noisy search space. Recent LLM-based approaches use prompting, iterative search, or seed-factor evolution, leaving direct factor construction through policy learning less explored. To address this, we introduce AlphaRLM, a reinforcement-learning framework that trains LLMs to construct executable symbolic factors from financial literature without input factor expressions. AlphaRLM incorporates Literature-Guided Operator Assembly (LEGO), which conditions factor construction on literature-derived ideas and executable operators, providing semantic priors for exploration. It further aligns the generation policy with market feedback through executable-validity and grouped-return-separation rewards that capture non-monotonic structures across factor quantiles. Extensive experiments on CSI 300, CSI 1000, and S&P 500 show that AlphaRL
    
[^323]: 常学习率随机梯度下降的强高斯逼近：偏差与轨迹推断

    Strong Gaussian Approximation of Constant-Learning-Rate SGD: Bias and Trajectory Inference

    [https://arxiv.org/abs/2610.14567](https://arxiv.org/abs/2610.14567)

    本文为重尾噪声下的常学习率随机梯度下降建立了KMT型强高斯逼近，实现了有限样本下同时性的整个轨迹层面的推断，并蕴含泛函中心极限定理。

    

    本文为常学习率随机梯度下降（SGD）建立了强高斯逼近（KMT型逼近）。我们在重尾噪声的情形下开展工作，并假设迭代次数增长与学习率收缩同时进行。我们的工作在同一概率空间上构造了简单的高斯过程，并以概率方式界定了SGD误差与其各自极限之间的最大差异。尽管目前已存在以最优点为中心的常学习率SGD的中心极限定理，但这些定理仅能提供单个迭代点的推断。相比之下，我们的淬火型（quenched）KMT逼近能够实现有限样本下同时性的、整个轨迹层面的推断，并蕴含随机过程的收敛性（即泛函中心极限定理FCLT）。对于以真实最优点为中心的累积误差，当学习率满足 $\gamma_n=\frac{c}{\sqrt n}+o(n^{-1/2})$（其中 $c$ 为常数）时，我们首先得到了KMT型逼近界。

    arXiv:2610.14567v1 Announce Type: new  Abstract: This paper develops strong Gaussian approximations (KMT type) for constant learning rate stochastic gradient descent (SGD). We work with heavy tailed noise and assume that the number of iterates grows and the learning rate shrinks simultaneously. Our work constructs simple Gaussian processes on the same space and bounds, in probability, the maximum difference between the SGD errors and their respective limits. While there exist central limit theorems for constant rate SGD centered at the optimum, these only provide individual iterate inference. By contrast, our quenched KMT approximations allow for finite sample, simultaneous, whole-trajectory-level inference and imply stochastic process convergence (Functional Central Limit Theorems (FCLTs)). For accumulated errors centered at the true optimum, we first obtain a KMT-type approximation bound when the learning rate scales as $\gamma_n=\frac{c}{\sqrt n}+o(n^{-1/2})$ for a constant $c \in \
    
[^324]: 自然代码变换使智能体在漏洞分析中产生混淆

    Natural Code Transformation Confuses Agents in Vulnerability Analysis

    [https://arxiv.org/abs/2610.14350](https://arxiv.org/abs/2610.14350)

    本文发现AI智能体在分析具有相同功能和相同漏洞的不同真实代码提交时会产出截然不同的结果，并据此提出从真实开发模式中挖掘自然且功能保持的代码变换，作为防御攻击者智能体的新手段。

    

    arXiv:2610.14350v2 公告类型： cross 摘要：在展示了自动化漏洞发现的强大潜力之后，AI智能体也可以被用来大规模攻击软件系统。当今的软件系统需要一些机制来平衡攻防双方的力量，通过减缓潜在攻击的速度，为防御者争取时间来修补漏洞并强化软件。沿着这一思路，我们对漏洞分析中智能体行为轨迹的研究揭示了一个令人惊讶的发现：对于承载相同功能和相同漏洞的真实世界代码提交，智能体产生了截然不同的分析结果。这一现象在多个代码领域和漏洞类别中持续出现。这促使我们研究源自真实世界软件开发过程的自然且保持功能不变的代码变换，以及其作为防御攻击者智能体的潜力。我们通过从真实世界代码提交中挖掘反复出现的开发模式，构建了一个自然变换语料库。

    arXiv:2610.14350v2 Announce Type: cross  Abstract: Having demonstrated a strong potential for automating vulnerability discovery, AI agents can also be used to attack software systems at scale. Today's software systems need mechanisms to level the playing field by slowing down potential attacks, giving them time to patch vulnerabilities and strengthen software. Along this line, our study of agent trajectories in vulnerability analysis shows a surprising finding: agents produce dramatically different results across real-world code commits that carry the same functionality and the same vulnerability. This phenomenon appears consistently in multiple code domains and vulnerability categories.   It motivates us to study natural and functionality-preserving code transformations derived from real-world software development, and their potential as a defense against attacker agents. We build a corpus of natural transformations by mining recurring development patterns from real-world code commit
    
[^325]: MvFFN：用于无位姿宽基线全景图布局重建的多视角户型图前馈网络

    MvFFN: Multi-view Floor plan Feed-forward Network for Unposed Wide-Baseline Panorama Layout Reconstruction

    [https://arxiv.org/abs/2610.14315](https://arxiv.org/abs/2610.14315)

    提出MvFFN网络，通过让每个全景图列仅局部预测所见墙体及顺时针相邻两堵墙，再由确定性聚合成全局户型图，实现无位姿宽基线全景图中相机位姿与布局的联合重建。

    

    从稀疏的、无位姿的、宽基线RGB全景图中重建精确的相机位姿和户型图布局，仍然是一个具有挑战性的开放问题。在这种设定下，直接进行多房间户型图预测尤为困难：位姿、几何、语义和全局布局结构紧密耦合，而完整的户型图难以作为单一的全局输出直接学习。我们的核心洞察是让模型的预测严格保持局部性。多视角有序墙体实例分割正是围绕这一理念设计的：每个全景图列仅预测它所观测到的墙体，以及按顺时针顺序接下来的两个相邻墙体。全局墙体划分、房间布局和共享坐标则通过对重叠的局部预测进行确定性聚合而自然涌现。基于MOIS，我们提出了多视角户型图前馈网络，它联合预测粗略的相机位姿、稠密的全局几何以及逐墙体的语义输出。

    arXiv:2610.14315v2 Announce Type: cross  Abstract: Reconstructing accurate camera poses and floor-plan layouts from sparse, unposed, wide-baseline RGB panoramas remains a challenging open problem. Direct multi-room floor-plan prediction is especially difficult in this setting: pose, geometry, semantics, and global layout structure are tightly coupled, yet a complete floor-plan is hard to learn as a single global output. Our key insight is to keep the model's predictions strictly local. Multi-view Ordered Wall Instance Segmentation (MOIS) is designed around this concept. Each panorama column predicts only the wall it observes plus the next two adjacent walls in clockwise order. The global wall partition, room layouts, and shared coordinates emerge from deterministic aggregation of overlapping local predictions. Building on MOIS, we propose Multi-view floor-plan Feed-Forward Network (MvFFN), which jointly predicts coarse camera poses, dense global geometry, and per-wall semantic outputs.
    
[^326]: 捉迷藏：隐藏概念以寻找黑盒模型的解释

    Hide and Seek: HIDE Concepts to SEEK explanations for black box models

    [https://arxiv.org/abs/2610.13935](https://arxiv.org/abs/2610.13935)

    提出了一种无需模型权重或概念标注数据集的黑盒视觉模型概念级归因方法HnS，通过视觉-语言模型检索概念相关图像、遮挡图像中的概念区域，并依据模型输出的变化来衡量该概念的归因贡献。

    

    解释机器学习模型的预测是一项核心挑战，尤其是对于黑盒视觉模型而言。基于概念的归因方法利用人类可理解的概念来解释模型。现有的基于概念的方法只能解释分类和回归模型，它们需要访问模型权重或大型的概念标注数据集，并且大多数方法无法提供图像级别的归因。我们提出了“捉迷藏”方法，这是一种新的概念级归因方法，无需访问模型权重或标注的概念数据集。HnS能够解释广泛的视觉模型。给定一个概念和一个图像集合，HnS使用视觉-语言模型检索与该概念对齐的图像，在每张图像中定位该概念，遮挡定位到的区域，并根据黑盒模型在原始图像与遮挡图像之间输出的变化来衡量归因。这通过直接测量……实现了忠实的归因。

    arXiv:2610.13935v1 Announce Type: cross  Abstract: Interpreting machine learning model predictions is a central challenge, particularly for black-box vision models. Concept-based attribution methods interpret models using human-understandable concepts. Existing concept-based methods can only interpret classification and regression models. They require access to model weights or large concept-annotated datasets, and most methods cannot provide image-level attribution. We propose Hide and Seek (HnS), a new method for concept-level attribution that does not require access to model weights or labeled concept datasets. HnS can interpret a wide range of vision models. Given a concept and image set, HnS retrieves concept-aligned images using a vision-language model, localizes the concept within each image, masks the localized region, and measures attribution from the change in the black-box model's output between the original and masked images. This enables faithful attribution by directly me
    
[^327]: AI模型能够可证明地隐藏任意能力

    AI Models Can Provably Hide Arbitrary Capabilities

    [https://arxiv.org/abs/2610.13928](https://arxiv.org/abs/2610.13928)

    本文提出一种将加密神经网络嵌入模型权重的后门攻击方法，在密码学假设下可证明地隐藏任意能力，使现有能力评估手段难以引出或解读这些隐藏功能。

    

    能力评估旨在在模型部署前暴露其危险行为，但其可靠性依赖于一个假设：隐藏的能力是可以被引出的。我们通过构建后门攻击证明，这一假设在对抗性条件下并不成立——该攻击将加密的神经网络嵌入宿主模型的权重之中，只有在接收到秘密触发器时才会解密并执行。借助数字保险箱技术，这些隐藏电路在密码学假设下可被证明难以被引出和解释，即使攻击者拥有对模型权重的完全访问权限，这将先前的工作从隐藏固定字符串扩展到了隐藏任意计算。我们在PyTorch中实现了概念验证构建，并在一个小型化学反应预测任务（作为CBRN相关能力的代理任务）上，通过微调、权重加噪、基于梯度的触发器搜索以及潜在对抗训练等方法评估了引出攻击的效果。我们的结果表明，该方法对这些引出手段具有抵抗能力。

    arXiv:2610.13928v1 Announce Type: new  Abstract: Capability evaluations aim to surface dangerous model behaviors before deployment, but their reliability depends on the assumption that hidden capabilities can be elicited. We show this assumption does not hold under adversarial conditions by constructing backdoor attacks that embed encrypted neural networks within host model weights, decrypting and executing them only upon receiving a secret trigger. Using digital lockers, these hidden circuits remain provably hard to elicit and interpret under cryptographic assumptions, even given full access to model weights, extending prior work from hiding fixed strings to arbitrary computations. We implement our proof of concept constructions in PyTorch and evaluate elicitation through fine-tuning, weight noising, gradient-based trigger search, and latent adversarial training on a small chemical reaction prediction task - a proxy for CBRN-relevant capabilities. Our results demonstrate resistance to
    
[^328]: 协作设计会议中的人类与大型语言模型引导：一项计算行为比较

    Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison

    [https://arxiv.org/abs/2610.13082](https://arxiv.org/abs/2610.13082)

    该研究基于AMI会议语料库199个配对情境，从五个维度对人类项目经理与Llama 3.1 8B在协作设计会议中的引导行为进行计算比较，发现人类引导在情境语义距离和共情/缓和语词汇密度上更高，且后者主要由缓和语词汇驱动。

    

    大型语言模型（LLM）正日益被探索用作群体讨论与协作工作的引导者，但LLM生成的引导与人类引导之间的行为差异仍未得到充分刻画。我们基于AMI会议语料库中的199个配对情境，将基于情境的协作设计会议中人类项目经理的引导话语与Llama 3.1 8B Instruct的回应进行了比较。分析涵盖五个验证性维度：情境语义距离、词汇详述代理指标、共情与缓和语词汇密度、话题分歧度以及阶段-词汇对齐度。一项探索性的指令性度量未能通过领域验证，因此不予解读。抽样的人类项目经理话语在情境语义距离（r = 0.644）和共情与缓和语词汇密度（r = 0.203）上得分更高。第二项对比主要由缓和语词汇驱动，因此我们将其解读为词汇层面的缓和行为而非……（摘要原文在此处截断）

    arXiv:2610.13082v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly being explored as facilitators of group discussion and collaborative work, but the behavioral differences between LLM-generated and human facilitation remain insufficiently characterized. We compare human Project Manager facilitation turns from scenario-based collaborative design meetings with Llama 3.1 8B Instruct responses across 199 paired contexts from the AMI Meeting Corpus. The analysis covers five confirmatory dimensions: contextual semantic distance, a lexical elaboration proxy, empathyand-hedge lexical density, topic divergence, and phase-vocabulary alignment. An exploratory directiveness measure failed domain validation and is not interpreted. The sampled human Project Manager turns were higher on contextual semantic distance (r = 0.644) and on empathyand-hedge lexical density (r = 0.203). The second contrast is carried mainly by hedge terms, so we read it as lexical hedging rather 
    
[^329]: 基于不动点流匹配的强化学习

    Reinforcement Learning via Fixed-point Flow Matching

    [https://arxiv.org/abs/2610.12848](https://arxiv.org/abs/2610.12848)

    本文提出了一种无需数据的不动点流匹配（FFM）方法，通过将actor更新表述为不动点迭代、以critic的动作梯度作为回归目标，实现了强化学习中流策略的训练与微调。

    

    在最大熵和行为约束强化学习（RL）中，最优策略由玻尔兹曼分布给出。从该分布中生成样本可能具有挑战性，因为它通常复杂且具有多峰性。虽然通过流匹配训练的流策略为表示此类分布提供了强大的框架，但它们传统上依赖于对训练数据的访问，而在线强化学习中通常无法获得训练数据。为了解决这一局限，我们提出了不动点流匹配（FFM），这是一种在强化学习中训练和微调流策略的无数据方法。FFM利用目标流恒等式将actor更新表述为不动点迭代，采用流匹配式的回归目标，其中critic的动作梯度作为回归目标。我们证明最优策略是该迭代的不动点。通过大量实验，我们证明FFM能够有效微调预训练的流策略。

    arXiv:2610.12848v1 Announce Type: new  Abstract: In maximum entropy and behavior-constrained reinforcement learning (RL), the optimal policy is given by a Boltzmann distribution. Generating samples from this distribution can be challenging, as it is often complex and multimodal. While flow policies trained with flow matching provide a powerful framework for representing such distributions, they traditionally rely on access to training data, which is generally unavailable in online RL. To address this limitation, we propose fixed-point flow matching (FFM), a data-free procedure for training and fine-tuning flow policies in RL. FFM leverages the target flow identity to formulate the actor update as a fixed-point iteration, utilizing a flow-matching-style regression objective where the critic's action gradients serve as regression targets. We show that the optimal policy is a fixed point of this iteration. Through extensive experiments, we demonstrate that FFM effectively fine-tunes pretr
    
[^330]: HRIL：通过高阶张量建模学习多模态协同信息

    HRIL: Learning Multimodal Synergy via Higher-Order Tensor Modeling

    [https://arxiv.org/abs/2610.12393](https://arxiv.org/abs/2610.12393)

    提出HRIL方法，通过在模态嵌入上构建经验交叉矩张量进行高阶张量建模，显式捕获体现为高阶统计依赖的多模态协同信息，从而在自监督多模态表示学习中保留协同信号的信息容量。

    

    自监督多模态表示学习在众多领域取得了显著成功，然而由于跨模态交互的复杂性，捕获协同信息仍然具有挑战性。与各模态之间的共享信息不同，协同信息是指只有通过多个模态的联合配置才能产生任务相关信号、且无法从任何单一模态中独立恢复的信息。本工作聚焦于如何在多模态表示中保留此类协同信号的信息容量。关键观察在于：协同信息体现为模态之间的高阶统计依赖，这为显式建模联合交互提供了一个有原则的目标。受此洞察启发，我们提出了高阶表示与信息学习，通过在模态嵌入上构建经验交叉矩张量来表示多……（原文摘要在此处截断）

    arXiv:2610.12393v1 Announce Type: new  Abstract: Self-supervised multimodal representation learning has achieved remarkable success across diverse domains, yet capturing synergistic information remains challenging due to the complexity of cross-modal interactions. Unlike the shared information across individual modalities, synergy arises when task-relevant signals emerge only from the joint configuration of multiple modalities and cannot be recovered from any modality in isolation. This work focuses on how to preserve the information capacity for such synergistic signals in multimodal representations. The key observation is that synergistic information is reflected in higher-order statistical dependence among modalities, which provides a principled target for explicitly modeling joint interactions. Motivated by this insight, we propose Higher-order Representation and Information Learning (HRIL), which constructs an empirical cross-moment tensor over modality embeddings to represent mul
    
[^331]: 基于接地大语言模型的CAD模型自动化装配指令生成：一种人在回路框架

    Automated Assembly Instruction Generation from CAD Models Using Grounded Large Language Models: A Human-in-the-Loop Framework

    [https://arxiv.org/abs/2610.11896](https://arxiv.org/abs/2610.11896)

    该论文提出一种人在回路的框架，将CAD装配模型映射为结构化的ProductGraph中间表示，利用大语言模型自动生成受工程信息约束的自然语言装配指令，并通过人工审核解决所有质量标记后才允许导出文档。

    

    装配文档是一种下游制造产物，目前通常仍需通过人工解读CAD模型来编写。结构化产品数据和大语言模型如今都已具备，但关于CAD解读、装配序列规划、指令编写和人工监督的研究在很大程度上是彼此独立进行的。本文提出了“基于CAD的装配指令生成”这一任务：即在从CAD模型中提取的结构化工程信息的约束下，生成自然语言形式的装配流程。所提出的框架将STEP装配体映射为类型化的ProductGraph中间表示，通过确定性拓扑排序推导出操作先后顺序，将每个步骤实现为仅以所选图上下文为条件的语言描述，附加每一步的视觉文档，并应用基于规则和模型辅助的质量检查。在人工审核员解决所有质量标记之前，PDF导出功能将保持禁用状态。

    arXiv:2610.11896v1 Announce Type: new  Abstract: Assembly documentation is a downstream manufacturing artifact that is still usually authored by interpreting CAD models by hand. Structured product data and large language models are both available, yet studies of CAD interpretation, assembly sequence planning, instruction writing, and human oversight have largely proceeded separately. This paper formulates CAD-grounded assembly instruction generation: the production of natural-language assembly procedures constrained by structured engineering information extracted from CAD models. The proposed framework maps a STEP assembly to a typed ProductGraph intermediate representation, derives a precedence order by deterministic topological sorting, realizes each step as language conditioned only on selected graph context, attaches per-step visual documentation, and applies rule-based and model-assisted checks. PDF export remains disabled until a human reviewer resolves every quality flag. The ca
    
[^332]: 大语言模型会从上下文中的奖励中学习吗？：重新思考奖励在上下文强化学习中的作用

    Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning

    [https://arxiv.org/abs/2610.11152](https://arxiv.org/abs/2610.11152)

    该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。

    

    大语言模型智能体越来越多地通过在上下文中积累经验而非更新参数来在推理阶段提升性能，这一过程通常被称为上下文强化学习（ICRL）。然而，上下文学习（ICL）是否真的能够发挥强化学习的作用，尚未经过检验。我们在其最简单的形式——直接ICRL中研究这一问题，即模型直接以原始的轨迹-奖励对为条件，并探究奖励是否真正充当学习信号。通过对六个模型在四个基准上的受控实验，我们发现奖励虽然会被模型读取，但其影响很小：翻转、随机化或移除奖励几乎不会改变性能改进曲线，即使在通过元提示明确指示模型进行探索、利用或对奖励进行推理的情况下也是如此。轨迹驱动了性能改进，但并非通过其语义内容：打乱或损坏的轨迹与真实轨迹的效果相当。

    arXiv:2610.11152v1 Announce Type: cross  Abstract: LLM agents increasingly improve at inference time by accumulating experience in context rather than by updating parameters. This process is often described as in-context reinforcement learning (ICRL). Whether in-context learning (ICL) can actually play the role of RL, however, has not been tested. We study this question in its simplest form, direct ICRL, where the model conditions directly on raw trajectory-reward pairs, and ask whether the reward acts as a learning signal. Through controlled experiments on four benchmarks across six models, we find that the reward is read, but its effect is small: flipping, randomizing, or removing the reward leaves the improvement curve almost unchanged, and this holds even under meta-prompts that explicitly instruct the model to explore, exploit, or reason over rewards. Trajectories drive improvement, but not through their semantic content: shuffled or corrupted trajectories work as well as real one
    
[^333]: NOMOS：将书面策略编译为LLM代理的静态验证工具调用门控

    NOMOS: Compiling Written Policies into Statically Verified Tool-Call Gates for LLM Agents

    [https://arxiv.org/abs/2610.11030](https://arxiv.org/abs/2610.11030)

    NOMOS是一个四遍编译器，可将自然语言策略编译为经过静态验证的确定性工具调用门控，将LLM代理在状态变更调用中的策略违规率从66.3%降至2.6%。

    

    使用工具的LLM代理会违反其被部署来执行的那些策略，而且往往是悄无声息地违反。此前的防御手段要么手工编写规则，要么对每个动作查询一次LLM验证器，要么通过重量级的形式化机制来编译策略。朴素的编译方法会失败：提取出的规则要么阻止满足其自身前提条件的工具调用，要么读取其工具并不具备的参数。NOMOS是一个四遍编译器，能将自然语言策略转化为确定性的工具调用门控；仅依靠工具模式级别的检查进行静态验证（无需证明器、求解器或LLM），即可修复或拒绝37%（航空领域）和13%（零售领域）的候选规则——若没有这一步，大多数已交付的规则将无法运作。在无防御的对话记录上回放编译后的规则，可以标记出拒绝合法工作的绑定（某个开发绑定拒绝了95.9%的可通过任务的调用）；而没有任何评估绑定被标记。在τ²-bench上，该门控将状态变更调用中违反参考编码条款的比例从66.3%降至2.6%（摘要在此处截断）。

    arXiv:2610.11030v1 Announce Type: cross  Abstract: Tool-using LLM agents violate the policies they are deployed to enforce, often silently. Prior defenses hand-write rules, query an LLM verifier per action, or compile policies through heavyweight formal machinery. Naive compilation fails: extracted rules block the tool satisfying their own precondition, or read arguments their tool lacks. NOMOS, a four-pass compiler, turns a natural-language policy into a deterministic tool-call gate; static verification with tool-schema-level checks alone (no prover, solver, or LLM) repairs or rejects 37% (airline) and 13% (retail) of candidates, without which most shipped rules are inoperable. Replaying compiled rules over undefended transcripts flags bindings that refuse legitimate work (a development binding refused 95.9% of task-passing calls); no evaluation binding is flagged. On $\tau^2$-bench the gate cuts violations of reference-encoded clauses among state-changing calls from 66.3% to 2.6% (ai
    
[^334]: 具有RBF核的高斯过程极大似然估计的精确渐近理论

    Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an RBF Kernel

    [https://arxiv.org/abs/2610.10080](https://arxiv.org/abs/2610.10080)

    本文针对基于RBF核的高斯过程，在固定域渐近框架下建立了极大似然估计的精确渐近理论，攻克了密集采样强相关性与协方差矩阵非线性依赖带来的长期理论难题。

    

    高斯过程（GP）广泛应用于机器学习、空间统计、时间序列分析、优化、贝叶斯统计以及各类科学应用中。高斯过程模型的核心组成部分是其核函数，通常通过参数族来指定。其中最广泛使用的选择之一是径向基函数（RBF）核，也称为平方指数核或高斯核，这得益于其形式简单、光滑且灵活的特性。在实践中，核参数通常通过极大似然估计（MLE）来估计，这也是标准高斯过程软件所采用的标准方法。尽管应用如此广泛，即便对于RBF核，极大似然估计量在固定域渐近框架下的渐近行为仍然鲜为人知。其主要困难源于密集采样观测之间日益增强的相关性，以及协方差矩阵对核参数的非线性依赖。在本文中……

    arXiv:2610.10080v1 Announce Type: cross  Abstract: Gaussian processes (GPs) are widely used across machine learning, spatial statistics, time-series analysis, optimization, Bayesian statistics, and scientific applications. A central component of a GP model is its kernel, which is typically specified through a parametric family. Among the most widely used choices is the radial basis function (RBF), also known as the squared exponential or Gaussian kernel, owing to its simple form, smoothness, and flexibility. In practice, the kernel parameters are routinely estimated by the maximum likelihood estimators (MLEs), as implemented by standard GP software. Despite this widespread use, the asymptotic behavior of the MLEs remains poorly understood under fixed-domain asymptotics, even for the RBF kernel. The main difficulty arises from the increasingly strong dependence among densely sampled observations and the nonlinear dependence of the covariance matrix on the kernel parameters. In this pape
    
[^335]: 互不相溶扩散策略：通过无标签噪声分配保留多模态机器人动作

    Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment

    [https://arxiv.org/abs/2610.09369](https://arxiv.org/abs/2610.09369)

    提出互不相溶扩散策略，一种无需标签的训练时噪声分配方法，通过保持噪声到动作路径的相对分离，防止扩散策略在低维机器人动作空间中坍缩为单一模态，从而保留多模态动作分布。

    

    当扩散策略首次被提出时，人们期望它们能够恢复多模态动作分布。然而，我们发现这一期望并不总是成立，因为即使保证了数据集模态的平衡以及批次内完全的对称性，扩散策略仍常常会坍缩到单一模态。我们的分析表明，独立的动作-噪声配对是导致这一失败的原因之一，它增加了扩散路径之间的混合与交叉，从而可能产生平均化的去噪响应，并抑制模态特定的行为。这一问题在机器人规划中尤为严重，因为动作空间是密集且低维的，会显著加剧这种混合与交叉。为了缓解该问题，我们提出了互不相溶扩散策略，这是一种无需标签的训练时附加组件，通过动作-噪声分配来保持相对互不干扰的噪声到动作路径，且无需修改策略架构。

    arXiv:2610.09369v1 Announce Type: cross  Abstract: When diffusion policies were first introduced, they were expected to recover multi-modal action distributions. However, we find this expectation does not always hold, as diffusion policies often collapse to a single modality even when we guarantee the balance of dataset modalities and exact within-batch symmetry. Our analysis indicates that independent action-noise pairing contributes to this failure by increasing mixing and crossing among diffusion paths, which can produce averaged denoising responses and suppress modality-specific behavior. This issue is especially severe in robot planning, where action spaces are dense and low-dimensional, significantly increasing such mixing and crossing. To alleviate this problem, we propose Immiscible Diffusion Policy, a label-free training-time add-on to diffusion policy that uses action-noise assignment to preserve relatively distinct noise-to-action routes without modifying the policy architec
    
[^336]: 利用序列蒙特卡洛引导扩散模型生成稀有事件

    Steering Diffusion Models to Rare Events with Sequential Monte Carlo

    [https://arxiv.org/abs/2610.08652](https://arxiv.org/abs/2610.08652)

    本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。

    

    扩散模型正日益被用作天气预报、分子动力学和材料设计等领域中昂贵模拟器的替代品。在这些模型中，计算事件 $E$ 的概率 $p_0[E]$ 十分困难，尤其当所关注的事件是稀有事件时。使用蒙特卡洛方法进行稳定估计在计算上将变得不可行，因为需要随稀有程度增加而不断增长的样本量（$\propto 1/p_0[E]$）来补偿。在本文中，我们提出了稀有事件的扩散重要性采样方法（Diffusion Importance Sampling of Rare Events，简称 DireSMC），这是一种序列蒙特卡洛方案，通过引导一组加权样本趋向稀有事件，不仅能获得样本，还能得到其概率的校准估计。我们通过对事件集合进行解析松弛来构建引导机制，使该方法能够轻松扩展到广泛的用户自定义稀有事件。我们在一个具有解析解的玩具问题以及一个基于分数的模型上验证了我们的方法。

    arXiv:2610.08652v1 Announce Type: cross  Abstract: Diffusion models are increasingly used as surrogates for expensive simulators in weather prediction, molecular dynamics, and materials design. In these models, computing the probability $p_0[E]$ of an event $E$ is difficult, especially when the event of interest is rare. A stable estimate using Monte Carlo becomes computationally intractable, requiring a growing sample size $\propto\!1/p_0[E]$ to compensate for an increasing rarity. In this paper, we present Diffusion Importance Sampling of Rare Events or DireSMC, a sequential Monte Carlo scheme that guides a population of weighted samples towards the rare event, giving access not only to samples but also to a calibrated estimate of its probability. We set up our guidance using an analytical relaxation of the event set, allowing the method to easily extend to a wide range of user-defined rare events. We validate our method on a toy problem with analytical solutions and on a score-based
    
[^337]: ProximalFM：隐藏混杂下的摊销式近端因果推断

    ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding

    [https://arxiv.org/abs/2610.08078](https://arxiv.org/abs/2610.08078)

    该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。

    

    标准的因果识别方法通常假设不存在未观测的混杂因素，当相关混杂因素未被观测到时这些方法可能会失效。近端因果推断则转而使用代理变量在隐藏混杂存在的情况下识别因果效应。然而，非参数近端估计在实践中可能颇具挑战性：恢复条件平均处理效应（CATE）等因果估计量需要求解一个病态的积分方程，该方程对数据需求量大、对超参数敏感且优化过程不稳定。对此类模型进行贝叶斯推断提供了一种理想的替代方案，可通过先验进行正则化来缓解上述困难。然而，计算后验本身也具有挑战性，因为典型的似然函数中会包含潜变量。借鉴近期表格基础模型在后门、工具变量和前门等设定中取得的成功，我们提出先验数据拟合网络（PFN）在……方面具有独特优势（摘要在此处截断）

    arXiv:2610.08078v1 Announce Type: cross  Abstract: Standard causal identification methods often assume no unmeasured confounding and can fail when relevant confounders are unobserved. Proximal causal inference instead uses proxy variables to identify effects under hidden confounding. However, nonparametric proximal estimation can be challenging in practice: recovering causal estimands such as the conditional average treatment effect (CATE) requires solving an ill-posed integral equation that is data-hungry, hyperparameter-sensitive, and optimization-unstable. Bayesian inference for such models provides a desirable alternative, mitigating these difficulties by regularizing through the prior. However, computing a posterior is itself challenging, as a typical likelihood function will include latent variables. Following the recent success of tabular foundation models in backdoor, instrumental variable, and frontdoor settings, we propose that prior-data fitted networks (PFNs) are uniquely s
    
[^338]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^339]: 迷失于bf16类型转换：导出三值语言模型可能抵消大部分低学习率代码改动的效果

    Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes

    [https://arxiv.org/abs/2610.07853](https://arxiv.org/abs/2610.07853)

    该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。

    

    BitNet b1.58、Falcon-E和BitCPM等三值语言模型先以更高精度的潜在权重进行微调，再通过导出步骤生成三值代码进行部署；在三个实验室公开记录的流程中，该导出步骤均会先将潜在权重转换为bf16。我们对三个实验室的这些流程进行了审计。在已发布的检查点中，对随附潜在权重执行fp32量化所得的结果与实际部署代码在Falcon-E和BitCPM上有0.83%–1.77%的代码不一致，在BitNet 2B-4T上有1.530%不一致；对于Falcon-E和BitCPM，绝大多数不一致源于这样的乘积：其bf16舍入恰好落在阈值上，并被“舍入到偶数”（ties-to-even）规则映射为零，而未经修改的onebitllms导出器可以逐字节复现全部四个Falcon-E发布版本。在微调终点，当学习率按标称的学习率与bf16-ULP之比选取时，按文档所述流程导出会使Falcon-E-1B-Base的贪婪解码GSM8K严格准确率从58.79%跌至0.78%，使BitCPM-CANN-（原文摘要在此处截断）的准确率从36.13%跌至0.39%。

    arXiv:2610.07853v1 Announce Type: cross  Abstract: Ternary language models such as BitNet b1.58, Falcon-E and BitCPM are fine-tuned with higher-precision latent weights and deployed as ternary codes produced by an export step that, in the labs' documented pipelines, first casts the latents to bf16. We audit those pipelines across three labs. In released checkpoints, fp32 quantization of the shipped latents disagrees with the deployed codes on 0.83-1.77% of codes in Falcon-E and BitCPM and on 1.530% in BitNet 2B-4T; for Falcon-E and BitCPM most disagreements are products that bf16 rounding lands exactly on the threshold, which ties-to-even maps to zero, and the unmodified onebitllms exporter reproduces all four Falcon-E releases byte for byte. At fine-tuned endpoints, with learning rates selected to match a nominal learning-rate-to-bf16-ULP ratio, the documented export lowers greedy GSM8K strict accuracy from 58.79% to 0.78% for Falcon-E-1B-Base and from 36.13% to 0.39% for BitCPM-CANN-
    
[^340]: EMG-FM-Bench：一个面向肌电信号基础模型迁移与适配的综合性基准测试

    EMG-FM-Bench: A Comprehensive Benchmark for Foundation Model Transfer and Adaptation on Electromyography

    [https://arxiv.org/abs/2610.06450](https://arxiv.org/abs/2610.06450)

    该论文提出了EMG-FM-Bench，一个整合20个公开数据集（超100万个EMG片段）、系统评估九个预训练基础模型在肌电信号上迁移与适配能力的综合基准，全面回答了冻结与微调、预训练收益、少样本新用户泛化及跨任务性能等关键问题。

    

    基础模型（FM）正日益被开发用于通用时间序列和生理信号，但它们向下游生理任务的迁移能力仍鲜为人知。这一问题对肌电信号（EMG）尤其具有挑战性，因为EMG信号分布在用户、传感配置、采集硬件和下游任务之间差异巨大。我们提出了EMG-FM-Bench，一个用于研究肌电信号上基础模型迁移与适配的系统性基准。EMG-FM-Bench统一整合了20个公开数据集，涵盖超过100万个EMG片段，并评估了九个预训练基础模型，围绕四个问题展开研究：预训练模型在冻结或完全微调情况下的表现如何；与从零开始训练相同模型相比，预训练能带来多大帮助；模型在标注数据有限的情况下对新用户的泛化能力如何；以及性能在不同EMG任务之间的变化情况。在整个基准测试中，线性探针……

    arXiv:2610.06450v2 Announce Type: replace  Abstract: Foundation models (FMs) are increasingly being developed for general time series and physiological signals, yet their transferability to downstream physiological tasks remains poorly understood. This question is particularly challenging for electromyography (EMG), where signal distributions vary substantially across users, sensing configurations, acquisition hardware, and downstream tasks. We introduce EMG-FM-Bench, a systematic benchmark for studying foundation-model transfer and adaptation on EMG. EMG-FM-Bench unifies 20 public datasets with over 1 million EMG segments and evaluates nine pretrained foundation models across four questions: how pretrained models perform when frozen or fully fine-tuned, how much pretraining helps compared with training the same model from scratch, how well models generalize to new users with limited labeled data, and how performance changes across different EMG tasks. Across the benchmark, linear prob
    
[^341]: 朗之万流映射：高效分子动力学与过渡路径采样

    Langevin Flow Maps: Efficient Molecular Dynamics and Transition Path Sampling

    [https://arxiv.org/abs/2610.05998](https://arxiv.org/abs/2610.05998)

    提出朗之万流映射，通过扩展机器学习力场以额外学习随机朗之万积分器，实现大时间步长的高效分子动力学模拟，在准确恢复系统动力学性质的同时速度提升一个数量级，并展示了向可迁移模型发展的路径。

    

    分子动力学模拟通过在许多飞秒级的小时间步长上积分朗之万方程来进行。这给估计发生在长得多的时间尺度上的系综性质和转变动力学带来了挑战。我们提出了朗之万流映射，它对机器学习力场进行了扩展，使其额外学习随机朗之万积分器。我们证明朗之万流映射能够实现大时间步长的分子动力学模拟，并能恢复系统的准确动力学性质，同时运行速度比现有机器学习力场快一个数量级。此外，通过在多样化的分子数据集上进行训练，我们展示了通往可迁移朗之万流映射的路径。

    arXiv:2610.05998v2 Announce Type: replace  Abstract: Molecular dynamics simulations proceed by integrating the Langevin equations over many small femtosecond timesteps. This poses a challenge for estimating ensemble properties and transition dynamics that occur on much longer timescales. We introduce Langevin Flow Maps, which extend machine-learned force-fields to additionally learn the stochastic Langevin integrator. We show that Langevin Flow Maps enable large-timestep molecular dynamics and recover accurate dynamical properties of the system, while running an order of magnitude faster than current machine-learned force fields. Further, by training on a diverse molecular dataset, we demonstrate a path towards transferable Langevin Flow Maps.
    
[^342]: 从混合到撕裂：基于消息传递的去中心化优化中的图分解

    From Mixing to Tearing: Graph Decomposition in Decentralized Optimization via Message Passing

    [https://arxiv.org/abs/2610.03709](https://arxiv.org/abs/2610.03709)

    该论文从第一性原理出发提出了一种基于图分解的去中心化优化通用框架，通过消息传递联合设计优化子问题、对偶变量分块和协作求解的智能体集群，突破了传统仅利用网络混合信息的范式。

    

    我们研究在无向图上最小化光滑强凸函数之和的问题，其中每个函数由一个智能体（节点）持有，且通信仅限于图中相邻的节点。现有的去中心化方法，无论是基于gossip（闲聊）算法还是基于生成树上的路由机制，通常都是利用网络来混合或聚合信息，以实现预先设定的局部优化更新。这种以通信为中心的视角所缺失的，是一个能够利用图结构来联合设计优化子问题、以及智能体协作求解这些子问题所需的协同计算与通信的通用框架。我们从第一性原理出发构建了这样一个框架，联合设计了协同约束的线性表示、由此产生的对偶变量的分块方式（并对其进行联合优化），以及在指定子图上协作求解每个分块子问题的连通智能体集群。

    arXiv:2610.03709v1 Announce Type: cross  Abstract: We study the minimization of sums of smooth strongly convex functions over undirected graphs, with each function held by one agent and communication restricted to neighbors in the graph. Existing decentralized methods, whether based on gossip or on routing over spanning trees, typically   use the network to mix or aggregate information to enable   {\it prescribed} local optimization   updates. What this communication-centered viewpoint lacks is a general   framework that uses graph structure to {\it jointly} design the   optimization subproblems and the cooperative computation and communication through which agents solve them   cooperatively.   We develop such a framework from first principles,   jointly designing the linear representation of agreement constraints, the blocks of   the resulting dual variables (jointly optimized), and connected cluster of agents that   cooperatively solve each block subproblem over the assigned subgraph
    
[^343]: 神经物理反演器：一种耦合集成条件化与残差学习的大地电磁反演模块化框架

    The Neuro-Physical Inverter: A Modular Framework for Magnetotelluric Inversion Coupling Ensemble Conditioning with Residual Learning

    [https://arxiv.org/abs/2610.03225](https://arxiv.org/abs/2610.03225)

    提出了一种模块化的神经物理反演器（NPI）框架，通过将集成条件高斯过程与残差学习神经网络相结合，实现了具备不确定性量化能力的大地电磁反演，在不破坏集成稳定性的情况下系统性降低误差。

    

    我们提出了神经物理反演器（NPI），这是一个模块化、不确定性感知的地球物理反演框架，它将基于集成的条件化方法与受约束的残差学习相结合，并在一维大地电磁（MT）环境中作为受控测试平台进行了演示。该框架分两个阶段运行：首先，集成条件高斯过程（EnsCGP）根据观测响应对电阻率模型的先验集成进行条件化，生成物理上可接受的参考集成；然后，残差学习神经网络预测针对该参考的校正，该网络在合成数据上训练，并通过物理耦合目标在野外应用中对每个测站进行微调。由于集成在两个阶段中均被条件化、细化和传播，因此每个估计都带有相应的集成离散度。合成实验表明，NPI在不破坏集成稳定性的前提下，系统性地降低了集成平均误差。应用于宽带（摘要在此处截断）。

    arXiv:2610.03225v1 Announce Type: new  Abstract: We present the Neuro-Physical Inverter (NPI), a modular, uncertainty-aware framework for geophysical inversion that couples ensemble-based conditioning with constrained residual learning, demonstrated in the 1D magnetotelluric (MT) setting as a controlled testbed. The framework operates in two stages. An Ensemble-Conditional Gaussian Process (EnsCGP) conditions a prior ensemble of resistivity models on the observed response, producing a physically admissible reference ensemble. A residual-learning neural network then predicts targeted corrections to this reference, trained on synthetic data and fine-tuned per station for field application through a physics-coupled objective. Because an ensemble is conditioned, refined, and propagated through both stages, every estimate carries an associated ensemble spread. Synthetic experiments show that NPI systematically reduces ensemble-mean error without destabilizing the ensemble. Applied to broadb
    
[^344]: 预测导向的高斯过程后验

    Predictively Oriented Gaussian Process Posteriors

    [https://arxiv.org/abs/2610.03201](https://arxiv.org/abs/2610.03201)

    提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。

    

    高斯过程（GP）是建模和量化函数关系中不确定性的强大工具。然而，它们要求使用者做出许多设计决策，例如核函数和观测模型的选择。次优的选择可能产生错误设定的模型，无法捕捉潜在的数据生成过程。我们提出了预测导向高斯过程（PrO-GPs），它将预测不确定性作为主要的推断目标，为标准高斯过程提供了一种稳健的替代方案。尽管对非参数模型直接计算PrO后验是不可行的，我们推导了一种简化形式和实用的采样方案以实现高效计算。通过合成数据和真实数据的实验，我们表明与标准高斯过程方法相比，PrO-GPs在模型错误设定的情况下能够产生更好校准的预测分布。

    arXiv:2610.03201v1 Announce Type: cross  Abstract: Gaussian Processes (GPs) are a powerful tool for modelling and quantifying uncertainty in functional relationships. However, they require practitioners to make a number of design decisions, such as the choice of the kernel and the observation model. Suboptimal choices can produce misspecified models that do not capture the underlying data generating process. We introduce Predictively Oriented Gaussian Processes (PrO-GPs), which treat predictive uncertainty as the primary inferential target and provide a robust alternative to standard GPs. Although direct computation of a PrO posterior for nonparametric models is intractable, we derive a reduced formulation and practical sampling scheme for efficient computation. Through synthetic and real data experiments, we show that PrO-GPs produce better calibrated predictive distributions under model misspecification compared to standard GP approaches.
    
[^345]: 非平稳分布偏移下图学习的高效记忆结晶

    Efficient Memory Crystallization for Graph Learning under Non-Stationary Distribution Shifts

    [https://arxiv.org/abs/2610.02795](https://arxiv.org/abs/2610.02795)

    该论文提出了一种无需训练的测试时框架EMC，通过闭式解将每个新到来的图域高效地“结晶”为紧凑且语义忠实的记忆，以低成本替代生成式记忆方案，从而应对图学习中的持续分布偏移。

    

    部署在现实世界系统中的深度图学习模型通常需要应对非平稳环境，其中底层的图分布会随时间持续漂移。主流的解决方案依赖于训练辅助生成模块来合成记忆图以实现跨域适应，这会带来大量的计算开销，并且在长期分布偏移下扩展性较差。我们认为存在一条更经济的路径：与其生成记忆，不如将其“结晶”。为此，我们提出了高效记忆结晶（EMC），这是一个无需训练的测试时框架，它通过对面向记忆的分布匹配目标的闭式解，将每个传入的图域蒸馏为紧凑且语义忠实的记忆，从而在持续协变量偏移下消除冗余的域信息。为了在模型遍历长序列时保持泛化性与适应性……（摘要在此处截断）

    arXiv:2610.02795v1 Announce Type: new  Abstract: Deep graph learning models deployed in real-world systems often need to cope with non-stationary environments, where the underlying graph distribution drifts continually over time. Prevailing solutions rely on training auxiliary generative modules to synthesize memory graphs for cross-domain adaptation, which incurs substantial computational overhead and scales poorly under prolonged distribution shifts. We argue that a more economical path exists: rather than generating memory, one can crystallize it. To this end, we propose Efficient Memory Crystallization (EMC), a training-free test-time framework that distills each incoming graph domain into a compact, semantically faithful memory through a closed-form solution to a memory-oriented distribution-matching objective, thereby eliminating redundant domain information under continual covariate shifts. To preserve both generalizability and adaptability as the model traverses a long sequence
    
[^346]: 想象未来，内化要旨：通过内化时空想象实现高效的VLA推理

    Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination

    [https://arxiv.org/abs/2610.02626](https://arxiv.org/abs/2610.02626)

    提出IG-VLA框架，通过在视觉表征空间中直接想象任务相关的未来场景演变，并将其内化为紧凑的场景要旨标记，使VLA模型在推理时无需昂贵的显式未来推演即可获得未来推理带来的动作预测增益，实现高效的机器人操作。

    

    视觉-语言-动作（VLA）模型日益融入中间推理以提升机器人操作能力，然而现有方法主要针对观测到的状态进行推理，并未显式地预测未来场景的演变。而若在每次推理步骤中都将推理扩展到显式的未来推演，则会引入大量计算开销。我们提出IG-VLA，一个使模型能够想象未来并内化其要旨的VLA推理框架。我们的潜在时空推理学会直接在视觉表征空间中想象与任务相关的未来场景演变，从而引导动作预测，而无需代价高昂的像素级视频生成。为进一步降低推理开销，我们引入了场景要旨记忆，它将推理所得的场景-行为关联内化为一个紧凑的场景要旨标记，在保留未来推理优势的同时，在推理阶段绕过显式的未来想象。

    arXiv:2610.02626v2 Announce Type: cross  Abstract: Vision-language-action (VLA) models increasingly incorporate intermediate reasoning to improve robotic manipulation, yet existing approaches primarily reason about observed states without explicitly anticipating future scene evolution. Extending such reasoning to explicit future rollouts at every inference step, however, introduces substantial computational overhead. We propose IG-VLA, a VLA reasoning framework that enables models to imagine the future and internalize the gist. Our Latent Spatiotemporal Reasoning learns to imagine task-relevant future scene evolution directly in visual representation space, guiding action prediction without costly pixel-level video generation. To further reduce inference overhead, we introduce Scene Gist Memory, which internalizes reasoning-derived scene-behavior associations into a compact Scene Gist Token, preserving the benefits of future reasoning while bypassing explicit future imagination at infe
    
[^347]: 参与敏感的收敛与异步在线学习中的“先分裂、后收敛”模式：基于22门OULAD课程的拓扑分析

    Participation-Sensitive Convergence and the Fragment First, Converge Later Pattern in Asynchronous Online Learning: A Topological Analysis Across 22 OULAD Courses

    [https://arxiv.org/abs/2610.01738](https://arxiv.org/abs/2610.01738)

    该研究基于22门OULAD课程（超22,000名学习者）的拓扑分析发现，Zigzag持续同调的β0是参与敏感指标，它与活跃学习者人数共同响应截止日期事件，且截止日期在82.6%的评估中引发“先分裂、后收敛”的群体行为模式。

    

    异步在线学习在提供时间灵活性的同时付出了结构性代价：学习共同体往往趋向于分裂而非凝聚。β0（来自Zigzag持续同调的不连通行为聚类数量）可作为衡量这一结构的群体层面指标。此前有两个问题尚未在大规模上得到验证：（1）表面上的β0收敛反映的是真实的行为趋同，还是学习者流失？（2）评估截止日期是否会产生可复现的“分裂-收敛”循环？我们在全部22门OULAD课程（N > 22,000；857个周对）中对这两个问题进行了验证。结果显示，β0的变化与活跃学习者数量的变化强烈共变（合并r = 0.387；各课程r_delta中位数为0.459，22门课程中有20门呈此关系），这表明β0是一个参与敏感的指标：β0与活跃学习者人数共同响应截止日期事件，而非一方导致另一方。截止日期在82.6%的评估中引发了分裂现象，并呈现出“先分裂、后收敛”的模式。

    arXiv:2610.01738v1 Announce Type: cross  Abstract: Asynchronous online learning offers temporal flexibility at a structural cost: learning communities tend to fragment rather than cohere. $\beta_0$, the number of disconnected behavioral clusters from Zigzag Persistent Homology, serves as a cohort-level indicator of this structure. Two questions remained unverified at scale: (1) does apparent $\beta_0$ convergence reflect genuine behavioral alignment or learner dropout? and (2) do assessment deadlines produce reproducible fragmentation-convergence cycles? We address both across all 22 OULAD courses (N > 22,000; 857 week-pairs). Changes in $\beta_0$ strongly co-vary with active learner changes (pooled r = 0.387; median per-course r_delta = 0.459, 20/22 courses), identifying $\beta_0$ as a participation-sensitive indicator: $\beta_0$ and active learner counts co-respond to deadline events rather than one causing the other. Deadlines produced fragmentation in 82.6% of assessments and the f
    
[^348]: 教师是方向，而非终点：在线策略蒸馏中外推强化学习诱导的表征残差

    The Teacher Is a Direction, Not a Destination: Extrapolating RL-Induced Representation Residuals in On-Policy Distillation

    [https://arxiv.org/abs/2609.36484](https://arxiv.org/abs/2609.36484)

    提出RIDE方法，通过在每一层外推强化学习引起的内部表征偏移方向（而非在输出空间外推隐式奖励），克服语言模型头的各向异性衰减与噪声放大问题，使学生模型在在线策略蒸馏中稳定地超越教师。

    

    在线策略蒸馏在学生模型自身的轨迹上训练学生模型去匹配教师的下一个词元分布，并已取得了显著的实证收益。广义变体允许学生模型通过在输出空间中外推隐式奖励来超越教师模型。然而，语言模型头会以各向异性的方式衰减这种变化：教师隐藏状态中编码的大部分变化仅以很小的比例传递到logits，而输出空间外推所依赖的采样词元对数概率比会注入噪声，外推过程会放大这些噪声，导致训练不稳定。我们观察到，强化学习会使模型的内部表征相对于其基础检查点发生偏移，并且这种偏移的方向可以在每一层被测量。受这一观察的启发，我们提出了RIDE（强化学习诱导方向外推），它在表征层面外推强化学习所诱导的变化方向……

    arXiv:2609.36484v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains a student to match the teacher's next-token distributions on the student's own trajectories and has yielded substantial empirical gains. Generalized variants allow the student to surpass the teacher by extrapolating an implicit reward in output space. The language-model head, however, attenuates this change anisotropically: much of the change encoded in the teacher's hidden states reaches the logits at a small fraction of its weight, and the sampled-token log-probability ratios on which output-space extrapolation relies inject noise that the extrapolation amplifies, making training unstable. We observe that reinforcement learning (RL) shifts a model's internal representations relative to its base checkpoint, and that the direction of this shift can be measured at every layer. Motivated by this observation, we propose RIDE (RL-Induced Direction Extrapolation), which extrapolates the RL-induced change di
    
[^349]: NeuronSifter：中枢神经系统微环境中的干预规划

    NeuronSifter: Intervention Planning in CNS Microenvironments

    [https://arxiv.org/abs/2609.35445](https://arxiv.org/abs/2609.35445)

    NeuronSifter 的核心创新在于将决策质量构建于干预接口本身：它把给药方案编译为状态条件下的靶点占据场，用占据条件扩散算子在微环境动力学中传播，并依据干预损失的期望下降来选择测量手段，同时将不同类型的观测结果同化到同一后验分布中。

    

    对中枢神经系统（CNS）干预进行优先级排序，需要预测剂量、给药途径和给药方案如何作用于一个部分观测的微环境，进而选择能够真正改变决策的测量手段。现有的动作条件预测器将给药方案简化为身份标识符或标量暴露量，丢弃了靶点在何处、何时被作用的信息；而将点估计交给独立的规划器，又会丢弃使测量值得执行的联合不确定性。因此，我们将决策质量视为干预接口本身的属性，而非控制器部署方式的问题。NeuronSifter 将给药方案编译为带支持掩码的状态条件靶点占据场，通过占据条件扩散算子在微环境动力学中传播该场，并根据干预损失的期望下降幅度来选择测量手段，同时将类型化的结果同化到同一后验分布中。在一个公开声明的合成阿尔茨海默（模型）……（原文摘要在此处截断）

    arXiv:2609.35445v2 Announce Type: replace  Abstract: Prioritizing central nervous system (CNS) interventions requires predicting how a dose, route, and schedule act on a partially observed microenvironment, then choosing the measurement that would change the decision. Action-conditioned predictors reduce a regimen to an identity token or a scalar exposure, discarding where and when the target is engaged; handing a point estimate to a separate planner then discards the joint uncertainty that makes a measurement worth running. We therefore treat decision quality as a property of the intervention interface, not of controller placement. NeuronSifter compiles regimens into state-conditional target-occupancy fields with support masks, propagates them through microenvironment dynamics with an occupancy-conditioned diffusion operator, and selects measurements by their expected reduction in intervention loss, assimilating typed outcomes into the same posterior. In a declared synthetic Alzheimer
    
[^350]: NeuronDiscover：基于世界动作模型在神经元微环境中进行机制发现的“孪生体中智能体”框架

    NeuronDiscover: Agent-in-Twin for Mechanistic Discovery in Neuronal Microenvironments with World Action Models

    [https://arxiv.org/abs/2609.35338](https://arxiv.org/abs/2609.35338)

    NeuronDiscover 提出“孪生体中智能体”框架，以机制为基础的世界动作模型统一耦合预测、干预提议与观测设计，并对“孪生混淆”进行形式化、对机制—偏差联合信念进行实验设计，从而在稀疏观测下区分真实机制变化与计算孪生体误差，实现神经元微环境的自动化机制发现。

    

    神经元微环境中的机制发现需要通过干预与测量来区分关于溶质输运与神经元响应的相互竞争的解释。然而，预测准确性无法解决这一问题：真实的机制变化与计算孪生体中的误差会在稀疏观测中留下相同的特征。我们对这种“孪生混淆”问题进行了形式化，并对机制—偏差的联合信念进行推理，设计能够将二者区分开的实验。NeuronDiscover 是一个“孪生体中智能体”框架，其共享的、以机制为基础的世界动作模型（WAM）将预测、干预提议与观测设计耦合在一起；经独立裁定的结果会修订一个具有界定范围的“机制—干预—观测—结果”（MIOY）图，其中获得支持的关系被编译为带有经偏差调整的接受界限的可执行程序。我们在由独立……（原文摘要在此处截断）裁决的模拟脑液示踪剂输运世界上进行评估。

    arXiv:2609.35338v2 Announce Type: replace  Abstract: Mechanistic discovery in neuronal microenvironments requires interventions and measurements that separate competing explanations of solute transport and neuronal response. Predictive accuracy cannot settle the question: a real mechanistic change and an error in the computational twin leave the same signature in sparse observations. We formalize this twin confounding and reason over a joint mechanism--discrepancy belief, designing experiments that separate the two. NeuronDiscover is an Agent-in-Twin framework whose shared, mechanism-grounded World Action Model (WAM) couples prediction, intervention proposals, and observation design; independently adjudicated outcomes revise a scoped Mechanism--Intervention--Observation--Outcome (MIOY) graph, whose supported relations compile into executable programs carrying discrepancy-adjusted acceptance bounds. We evaluate on simulated brain-fluid tracer-transport worlds adjudicated by an independe
    
[^351]: 深度核对冲

    Deep kernel hedging

    [https://arxiv.org/abs/2609.34474](https://arxiv.org/abs/2609.34474)

    该论文提出了一种深度核对冲框架，通过神经网络参数化核函数并结合广义表示定理与随机傅里叶特征近似，实现了可处理路径依赖信息的可扩展、高效金融对冲。

    

    我们提出了一种深度核对冲框架，该框架将深度学习的灵活性与核方法的结构归纳偏置相结合。对冲泛函被限制在一个再生核希尔伯特空间中，其核函数通过输入特征的神经网络嵌入进行参数化。该框架在凸损失函数下最小化正则化经验风险，并能够通过截断的时间增强签名特征来处理路径依赖信息。我们针对联合对冲问题推导了一个广义表示定理，将经验优化问题简化为有限维问题。为了进一步降低与大型核矩阵相关的计算成本，我们开发了一种可扩展的随机傅里叶特征近似方法，并建立了收敛性保证。随机傅里叶参数仅需采样一次并在整个训练过程中保持固定，而深度核则通过（此处摘要被截断）适应市场数据。

    arXiv:2609.34474v2 Announce Type: replace  Abstract: We introduce a deep kernel hedging framework that combines the flexibility of deep learning with the structural inductive bias of kernel methods. The hedging functional is restricted to a reproducing kernel Hilbert space whose kernel is parameterized through a neural network embedding of the input features. The framework minimizes a regularized empirical risk under convex loss functions and can accommodate path-dependent information through truncated time-augmented signature features. We derive a generalized representer theorem for the joint hedging problem, reducing the empirical optimization to a finite-dimensional problem. To further reduce the computational cost associated with large kernel matrices, we develop a scalable random Fourier feature approximation and establish convergence guarantees. The random Fourier parameters are sampled once and remain fixed throughout training, while the deep kernel adapts to market data through
    
[^352]: 多候选响应用于从示范中学习的统计优势

    The Statistical Benefits of Multiple Responses for Learning from Demonstrations

    [https://arxiv.org/abs/2609.33291](https://arxiv.org/abs/2609.33291)

    当示范者并非最优时，返回多个候选响应（pass@$k$）能带来更强的统计优势——将最坏情况下对目标精度的依赖从 $1/\varepsilon^2$ 改善为 $1/\varepsilon$，并将对奖励类大小的依赖从 $\log N$ 改善为 $\log N/\log k$。

    

    许多生成式系统会返回多个候选响应，并根据其中最优的一个进行评估。最近的研究表明，当示范为最优时，pass@$k$ 能够将通过示范学习的样本复杂度降低一个 $k$ 的对数因子。我们探讨了当示范者不被假设为最优时会发生什么。我们发现，在这种设定下，多个响应能带来定性上更强的优势。在一个没有奖励反馈的有限奖励类模型中，从 pass@$1$ 转变为任何 $k\ge2$ 的 pass@$k$，会将对目标精度的最坏情况依赖从 $1/\varepsilon^2$ 改善为 $1/\varepsilon$，且该结论在所有示范者质量下均成立。在标准评估设定下（即一个未知奖励在训练前被固定），增大 $k$ 还提供了一个额外的独特好处：对大小为 $N$ 的奖励类的最优依赖从 $\log N$ 改善为 $\log N/\log k$。我们进一步证明这两个效应……

    arXiv:2609.33291v2 Announce Type: replace-cross  Abstract: Many generative systems return multiple candidate responses and are evaluated according to the best one. Recent work shows that, when demonstrations are optimal, pass@$k$ can reduce the sample complexity of learning from demonstrations by a logarithmic factor in $k$. We ask what happens when the demonstrator is not assumed to be optimal. We find that multiple responses provide a qualitatively stronger benefit in this setting. In a finite reward-class model with no reward feedback, moving from pass@$1$ to any pass@$k$ with $k\ge2$ changes the worst-case dependence on target accuracy from $1/\varepsilon^2$ to $1/\varepsilon$, uniformly over demonstrator quality. Under standard evaluation, where an unknown reward is fixed before training, increasing $k$ provides an additional and distinct benefit: the optimal dependence on a reward class of size $N$ improves from $\log N$ to $\log N/\log k$. We further show that these two effects 
    
[^353]: PrismQuant：面向分组量化器的最优零空间旋转

    PrismQuant: Optimal Null-Space Rotations for Grouped Quantizers

    [https://arxiv.org/abs/2609.32429](https://arxiv.org/abs/2609.32429)

    PrismQuant提出了一种量化器感知的旋转框架，通过将旋转设计表述为Ky Fan迹最大化并推导出可证明最优的闭式解，将激活主特征空间与非对称分组INT4量化的常数组子空间精确对齐，从而在不扩大组内数值范围的前提下提升低比特量化效果。

    

    更小的激活异常值并不一定意味着更好的低比特量化：关键在于它们与量化器的对齐程度。我们提出了PrismQuant，这是一个量化器感知的旋转框架，将激活的主特征空间与非对称分组INT4的常数组子空间对齐。仿射偏移在该子空间中表示能量，而不会扩大组内的数值范围。我们将旋转设计表述为Ky Fan迹最大化问题，并推导出一个对该对齐目标可证明最优的闭式解。紧凑的Householder变换及其compact-WY表示使得无需梯度的构建以及在可折叠和在线位置的高效应用成为可能。一个预测性的范围定律进一步将未对齐的激活能量和组大小与量化相关的变化联系起来。在Llama、Qwen和Mistral上的实验涵盖了高达70B参数的稠密模型和30B的混合专家模型。

    arXiv:2609.32429v2 Announce Type: replace-cross  Abstract: Smaller activation outliers do not necessarily imply better low-bit quantization: their alignment with the quantizer matters. We introduce PrismQuant, a quantizer-aware rotation framework that aligns the leading activation eigenspace with the constant group subspace of asymmetric grouped INT4. The affine offsets represent the energy in this subspace without widening the range within the group. We formulate rotation design as a Ky Fan trace maximization and derive a closed-form solution that is provably optimal for this alignment objective. Compact Householder transformations and their compact-WY representation enable gradient-free construction and efficient application at both foldable and online sites. A predictive range law further connects unaligned activation energy and group size to quantization-relevant variation. Experiments on Llama, Qwen, and Mistral span dense models up to 70B parameters and a 30B mixture-of-experts m
    
[^354]: TimeES：基于演化谱的概率性与确定性时间序列预测

    TimeES: Probabilistic and Deterministic Time Series Forecasting via Evolutionary Spectra

    [https://arxiv.org/abs/2609.32384](https://arxiv.org/abs/2609.32384)

    提出TimeES框架，基于演化谱理论显式建模傅里叶谱中的演变与随机性，将非平稳随机过程建模转化为学习由随机变量调制的演化表示，并利用厄米对称性将谱估计复杂度从O(NM)降至O(NK)，从而实现对非平稳时间序列的概率性与确定性预测。

    

    现实世界中的时间序列本质上是非平稳的，其趋势、周期模式和不确定性随时间不断演变。虽然傅里叶域为时间序列建模提供了一个自然的视角，但当前的深度学习方法并未显式地建模傅里叶谱中的演变与随机性，这限制了它们在非平稳时间序列中同时准确预测期望轨迹及其不确定性的能力。受演化谱理论启发，我们提出了TimeES，一个基于演化谱理论实现概率性与确定性预测的通用框架。具体而言，我们推导了一种可参数化的演化谱公式，将非平稳随机过程建模重新表述为学习由随机变量调制的演化表示。此外，我们利用厄米对称性和谱特性，将估计谱的复杂度从O(NM)降低到O(NK)，其中K << M/2。

    arXiv:2609.32384v2 Announce Type: replace  Abstract: Real-world time series are inherently non-stationary, with trends, periodic patterns, and uncertainty evolving over time. While the Fourier domain offers a natural lens to model time series, current deep learning approaches do not explicitly model evolution and randomness in the Fourier spectra, which limits their ability to accurately predict both the expected trajectory and its uncertainty in non-stationary time series. Motivated by Evolutionary Spectra (ES) theory, we propose TimeES, a general framework that enables probabilistic and deterministic forecasting via the evolutionary spectra theory. Specifically, we derive a parameterizable evolutionary spectra formulation, recasting non-stationary random process modeling as learning an evolving representation modulated by random variables. Furthermore, we reduce the complexity of the estimated spectra from O(NM) to O(NK), where K << M/2, by exploiting Hermitian symmetry and spectral 
    
[^355]: DiffPTS：重新思考用于概率时间序列预测的扩散模型证据下界（ELBO）

    DiffPTS: Rethinking Diffusion ELBO for Probabilistic Time Series Forecasting

    [https://arxiv.org/abs/2609.32363](https://arxiv.org/abs/2609.32363)

    本文通过在位置-尺度噪声模型下重新推导扩散模型的证据下界（ELBO），提出了 DiffPTS，将估计器的训练与变分推断统一为一个自然的联合训练目标，从而改进概率时间序列预测。

    

    概率时间序列预测需要对复杂且随时间变化的分布进行建模和预测。近年来，基于去噪扩散概率模型（DDPM）的方法展现出良好的前景，这类方法通过为扩散过程配备预训练的均值和方差估计器来适应分布偏移。然而，这些方法通常遵循标准的 DDPM 框架，仅考虑证据下界（ELBO）的部分组成部分，并将估计器的训练视为与变分推断框架相分离的、人为设计的回归任务。为解决这一问题，我们在位置-尺度噪声模型（LSNM）下重新审视 ELBO，发现它能够自然地为估计器导出一个高斯负对数似然目标，并内在地定义了一个联合训练目标，该目标统一了近期用于概率预测的各种扩散范式。基于这一具有理论依据的 ELBO 重构，我们提出了 DiffPTS（原文摘要在此处截断）。

    arXiv:2609.32363v2 Announce Type: replace  Abstract: Probabilistic time series forecasting requires modeling and predicting complex and time-varying distributions. Recently, Denoising Diffusion Probabilistic Model (DDPM)-based approaches have shown promise by equipping the dif- fusion process with pretrained mean and variance estimators to accommodate distributional shift. However, these methods typically follow the standard DDPM framework and consider only partial components of the evidence lower bound (ELBO), treating the training of estimators as designed regression tasks separate from the variational inference framework. To address this, we rethink the ELBO under the Location-Scale Noise Model (LSNM) and find that it naturally induces a Gaussian negative log likelihood objective for the estimators and inherently defines a joint training objective that unifies recent diffusion paradigms for probabilistic forecasting. Building on this principled ELBO reformulation, we propose Diff- P
    
[^356]: 面向不确定动态系统的可证明神经网络安全观测器学习

    Learning Provable Neural Network Observer for Uncertain Dynamical Systems

    [https://arxiv.org/abs/2609.30819](https://arxiv.org/abs/2609.30819)

    提出了一种两阶段训练框架（点引导李雅普诺夫预训练加LMI微调），为不确定动态系统的神经网络观测器实现了可证明的全局李雅普诺夫稳定性，同时克服了传统LMI方法带来的大规模半定规划可扩展性瓶颈。

    

    在许多安全关键应用中，不确定动态系统的控制依赖于能够估计状态和外部扰动的观测器。神经网络观测器可以提高估计精度，但通过线性矩阵不等式（LMI）约束来认证其李雅普诺夫稳定性会产生大规模的半定规划（SDP），对于大型网络而言难以求解。为了克服这一可扩展性瓶颈，我们提出了一种针对可证明稳定神经网络观测器的新型两阶段训练框架。该方法将优化过程解耦为两个阶段：首先是点引导的李雅普诺夫预训练阶段，可在采样状态上快速实现高估计精度和局部稳定性；随后是LMI微调阶段，可高效地满足严格的全局李雅普诺夫稳定性认证。我们为局部稳定性半径以及在给定紧致误差-状态域上的概率覆盖提供了形式化的理论保证。

    arXiv:2609.30819v1 Announce Type: new  Abstract: In many safety-critical applications, control of uncertain dynamical systems relies on observers that estimate states and external disturbances. Neural network observers can improve estimation accuracy, but certifying their Lyapunov stability via Linear Matrix Inequality (LMI) constraints leads to large-scale semidefinite programs (SDPs) that are difficult to solve for large networks. To overcome this scalability bottleneck, we propose a novel two-stage training framework for provably stable neural network observers. Our approach decouples the optimization into a point-guided Lyapunov pre-training phase, which rapidly achieves high estimation accuracy and local stability over sampled states, followed by an LMI fine-tuning phase that efficiently satisfies a strict global Lyapunov stability certificate. We provide formal theoretical guarantees for local stability radii and probabilistic coverage over a prescribed compact error-state domain
    
[^357]: 一个从纵向文本数据中建模组织级语义身份的计算框架

    A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data

    [https://arxiv.org/abs/2609.29584](https://arxiv.org/abs/2609.29584)

    该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。

    

    组织持续产生大量文本数据，这些数据记录了组织如何随时间进行沟通、演化并形成自身差异。尽管自然语言处理的最新进展已显著提升了组织层面的文本分析能力，但现有方法主要将组织表示为用于相似度估计、分类或检索的潜在嵌入向量或预测性特征向量。因此，目前尚缺乏一个通用的计算框架，能够将组织级语义身份建模为一种可解释的、基于纵向文本证据而不断演化的语义构建。本文提出了一个计算框架，将语义表示学习、基于图的语义建模、组织级语义指纹、时序语义演化以及证据驱动的验证整合在一个统一的分析方法论之中。组织被刻画……（原文摘要在此处截断）

    arXiv:2609.29584v1 Announce Type: cross  Abstract: Organisations continuously generate large volumes of textual data that capture how they communicate, evolve and differentiate themselves over time. Although recent advances in natural language processing have substantially improved organisation-level text analytics, existing approaches primarily represent organisations as latent embeddings or predictive feature vectors for similarity estimation, classification or retrieval. Consequently, there is currently no general computational framework for modelling organisation-level semantic identity as an interpretable and evolving semantic construct derived from longitudinal textual evidence. This paper introduces a computational framework that integrates semantic representation learning, graph-based semantic modelling, organisation-level semantic fingerprints, temporal semantic evolution and evidence-driven validation within a unified analytical methodology. Organisations are characterised th
    
[^358]: 记忆注意力

    Memory Attention

    [https://arxiv.org/abs/2609.28399](https://arxiv.org/abs/2609.28399)

    提出记忆注意力，用可复用的词元记忆与上下文键结合构建注意力值，推理时可简化为查表加法并支持CPU卸载，在相同训练预算下提升了语言建模和下游任务性能。

    

    语言模型通常从上下文隐藏状态构建注意力值，即使其中某些内容可能在不同上下文间是可以复用的。我们研究了在结合上下文信息的情况下，基于词元索引的记忆能否取代专用的值投影。我们提出了记忆注意力，它通过将层特定的词元记忆与上下文键相结合来构建值。记忆提供词元特定的表示，而键则保持对上下文的依赖性。在推理时，归一化可以被折叠进记忆表中，从而将值的构建简化为查找和加法。基于词元索引的检索还支持带预取功能的CPU卸载，减少GPU上的参数存储。在匹配的训练词元预算并引入额外记忆参数的条件下，跨多种注意力配置的实验显示出语言建模和平均下游性能的改进。

    arXiv:2609.28399v1 Announce Type: new  Abstract: Language models typically construct attention values from contextual hidden states, even when some of their content may be reusable across contexts. We investigate whether token-indexed memory can replace the dedicated value projection when complemented by contextual information. We propose Memory Attention (MA), which forms values by combining layer-specific token memory with contextual keys. The memory supplies token-specific representations, while the keys preserve context dependence. At inference, normalization can be folded into the memory tables, reducing value construction to lookup and addition. Token-indexed retrieval also enables CPU offloading with prefetching, reducing GPU parameter storage. Under matched training token budgets and with additional memory parameters, experiments across attention configurations show improved language modeling and average downstream performance.
    
[^359]: hyperbolix：基于 JAX 的双曲深度学习库

    hyperbolix: Hyperbolic Deep Learning in JAX

    [https://arxiv.org/abs/2609.28248](https://arxiv.org/abs/2609.28248)

    该论文提出了 hyperbolix，首个基于 JAX 与 Flax NNX 构建的全面、通用的开源双曲深度学习库，提供六种统一接口的流形、覆盖各类双曲神经网络层族以及黎曼优化器和降维工具。

    

    我们提出了 hyperbolix，一个基于 Flax NNX 构建、运行于 JAX 之上的开源双曲深度学习库。据我们所知，这是 JAX 中首个全面、通用的双曲深度学习库。它包含六种具有统一接口的流形：欧几里得空间、庞加莱球、双曲面、κ-stereographic 模型、混合曲率乘积空间以及固有速度空间。我们实现了涵盖线性层、卷积、注意力机制、归一化、位置编码、回归和向量量化的各类层族。这些构建模块所覆盖的方法范围，从 Ganea 最初提出的双曲神经网络，到近期的完全双曲架构（如 Hypformer 和 Lorentzian ResNet）应有尽有。此外，hyperbolix 还包含以 optax 变换形式实现的黎曼优化器、封装的分布以及双曲降维技术。其 API 采用地道的 JAX 惯用风格：流形是无状态的……（原文摘要在此处截断）

    arXiv:2609.28248v1 Announce Type: new  Abstract: We present hyperbolix, an open-source library for hyperbolic deep learning in JAX, built on Flax NNX. To our knowledge, it is the first comprehensive, general-purpose hyperbolic deep learning library in JAX. It includes six manifolds with a common interface: Euclidean space, the Poincar\'e ball, the hyperboloid, the $\kappa$-stereographic model, mixed-curvature product spaces, and the proper velocity space. We implement layer families that cover linear layers, convolutions, attention, normalization, positional encoding, regression, and vector quantization. These building blocks span methods ranging from Ganea's original hyperbolic neural networks to recent fully hyperbolic architectures such as Hypformer and Lorentzian ResNet. Additionally, hyperbolix contains Riemannian optimizers implemented as optax transformations, wrapped distributions, and hyperbolic dimensionality-reduction techniques. Its API uses idiomatic JAX: Manifolds are sta
    
[^360]: EduBehaviors：面向可审计教育对话编码的基于断言的模式框架

    EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues

    [https://arxiv.org/abs/2609.27043](https://arxiv.org/abs/2609.27043)

    提出了EduBehaviors框架，利用大语言模型测量教育对话中重复出现的可观察行为并据此学习分类器，实现了可解释、可审计的教育对话标注，其性能与直接提示方法相当。

    

    大语言模型使得与目标教学构念相对应的教学标注能够快速部署，为在对话数据集上生成分类提供了自然语言接口。然而，由于大语言模型推理的不透明性，我们无法获得关于模型为何为某句话语选择特定标签的可验证的、机制层面的洞察。我们提出了EduBehaviors框架，这是一种可解释、可扩展的教育数据标注方法，它利用大语言模型测量与多个目标构念相关的重复出现的可观察行为，然后基于这些可观察行为学习该构念的分类器。我们在TalkMoves数据集上对该框架进行了评估，预测教师TalkMoves标签。我们的最佳配置取得了0.673的宏平均F1值和0.688的Cohen's kappa系数，证明其与直接提示方法相比具有竞争力。此外，我们发布了EduBehaviors工具包，包含两个供研究人员使用的工具。

    arXiv:2609.27043v1 Announce Type: new  Abstract: Large language models have allowed the rapid deployment of pedagogical annotations corresponding to constructs of interest, allowing a natural language interface for generating classifications on a conversational dataset. However due to the opaque nature of LLM reasoning, we have no verifiable, mechanistic insight into why a model chose a label for an utterance. We introduce the EduBehaviors framework, an interpretable, scalable approach to annotating educational data that uses LLMs to measure repeated observable behaviors relevant to many constructs of interest and then learns a classifier for the construct based on these observable behaviors. We evaluate the framework on the TalkMoves dataset, predicting the Teacher TalkMoves labels. Our best configuration results in a macro-F1 of 0.673 and 0.688 Cohen's kappa, proving competitive with direct prompting approaches. In addition, we release EduBehaviors Toolkit, two tools allowing researc
    
[^361]: 每个固定指标都有盲区：一种用于评分天气预报真实性的学习型大气评论器

    Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism

    [https://arxiv.org/abs/2609.18381](https://arxiv.org/abs/2609.18381)

    该论文提出通过训练判别器来生成类似散度的真实性评分，能够自适应地检测天气预报模型表现出的任何失败模式，从而克服固定指标存在的盲区。

    

    尽管机器学习天气预报模型在逐点指标上具有很高的精度，但它们可能表现出多种失败模式，例如模糊、周期性不规则以及其他非物理的空间伪影。这促使人们提出了各种指标来检测已知的失败情况。现有指标预先固定了某种表示或变换，而这种选择限制了它们能够检测到的伪影类型。我们提出训练一个判别器来区分参考数据和模型输出，并利用其输出logit获得一种类似散度的真实性评分。该判别器能够学习任何能将模型输出场与真实天气区分开的特征，从而适应该模型所表现出的任何失败模式。我们通过对ERA5再分析数据施加各种合成破坏，将我们的学习型大气评论器与现有指标进行比较。我们的方法成功识别了这些破坏并对其严重程度进行排序，而现有指标在至少……

    arXiv:2609.18381v1 Announce Type: new  Abstract: Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical spatial artifacts. This has motivated a variety of metrics to detect known failure cases. Existing metrics fix a representation or transformation in advance, and that choice limits the artifacts they can detect. We propose to train a discriminator for separating reference data from the model's output, and using its output logit to obtain a divergence-like realism score. The discriminator learns whatever separates the model's fields from real weather, adapting to whichever failure mode that model exhibits. We compare our learned atmospheric critic to existing metrics using various synthetic corruptions applied to ERA5 reanalysis data. Our method successfully identifies the corruptions and ranks their severity, while existing metrics fail on at lea
    
[^362]: 基于正则化的鲁棒强化学习的统一约束视角

    A Unified and Constrained View of Regularization-Based Robust Reinforcement Learning

    [https://arxiv.org/abs/2609.13050](https://arxiv.org/abs/2609.13050)

    该论文通过推导性能差距上界统一了基于正则化的鲁棒强化学习方法，将鲁棒训练表述为约束优化问题，并通过联合更新拉格朗日乘子自动调整正则化权重以提升鲁棒性。

    

    基于正则化的方法已成为训练深度强化学习策略以对抗对抗性输入扰动的标准方法。在本文中，我们通过推导标称策略与最坏情况策略之间性能差距的新上界来统一这些方法。每个上界都表示为现有的正则化目标加上标称策略与最坏情况策略之间的KL散度惩罚，这进一步解释了为什么在实践中添加KL惩罚能够提升鲁棒性。基于这些上界，我们将鲁棒训练表述为一个约束优化问题，并证明现有方法对应于固定拉格朗日乘子的特殊情况。与之相反，我们在更新策略的同时联合更新该乘子，从而自动调整正则化权重。最后，我们在多个连续控制任务上进行了广泛的对抗性评估，以验证我们的理论分析。

    arXiv:2609.13050v1 Announce Type: new  Abstract: Regularization-based methods have become a standard approach for training Deep Reinforcement Learning policies against adversarial input perturbations. In this paper, we unify these methods by deriving new upper bounds on the performance gap between the nominal and worst-case policies. Each upper bound is expressed as an existing regularization objective plus a KL-divergence penalty between the nominal and worst-case policies, which further explains why adding a KL penalty improves robustness in practice. Building on these bounds, we formulate robust training as a constrained optimization problem, showing that existing methods correspond to the special case of a fixed Lagrange multiplier. We instead update the multiplier jointly with the policy to automatically tune the regularization weight. Finally, we conduct extensive adversarial evaluations across several continuous control tasks to validate our theoretical analysis.
    
[^363]: 相对智能学习 II：可处理或半监督的实例最优学习

    Relatively Smart II: Tractable or Semi-Supervised Instance-Optimal Learning

    [https://arxiv.org/abs/2609.10886](https://arxiv.org/abs/2609.10886)

    本文证明ERM乃至任何正规一致学习器在无分布二分类中都是相对智能的，从而解决了此前留下的开放问题，即自然的可处理学习器能否达到与单包含图学习器可比的保证。

    

    我们继续研究由 Dughmi 和 Pour（2026）提出的“相对智能学习”概念，它要求监督学习器在逐一边际的意义上，与每一个能够从无标注数据中可靠验证的、针对固定分布的误差保证相竞争。他们证明了单包含图学习器是相对智能的，但样本复杂度存在二次方膨胀，且任何相对智能学习器都无法做得更好，由此留下了 ERM 或其他自然、可处理的学习器能否达到可比保证的开放问题。他们还留下另一个开放问题：这种样本复杂度膨胀是否可以仅限于无标注数据。我们的第一个结果表明，ERM——实际上是任何正规一致学习器——在无分布设定下的二分类问题中都是相对智能的。我们证明了，若在 m 个样本下具有较小的可验证误差，则意味着在大小为 O(m²) 的随机样本上的均匀分布中也具有类似的较小误差，从而在该样本上产生大小至多为 2^(m+1) 的覆盖……

    arXiv:2609.10886v1 Announce Type: new  Abstract: We continue the study of relatively smart learning, introduced by Dughmi and Pour (2026), which asks a supervised learner to compete, marginal by marginal, with every distribution-fixed error guarantee soundly certifiable from unlabeled data. They showed that the One-Inclusion Graph (OIG) learner is relatively smart with a quadratic sample-complexity blowup, and that no relatively smart learner can do better, leaving open whether ERM or another natural or tractable learner achieves comparable guarantees. They also left open whether the blowup can be restricted to unlabeled data.   Our firs results shows that ERM---and in fact any proper consistent learner---is relatively smart for binary classification in the distribution-free setting. We show that a small certifiable error with $m$ samples implies a similarly small error on the uniform distribution over a random sample of size $O(m^2)$, yielding a cover of size at most $2^{m+1}$ on that
    
[^364]: 基于多水平集的物理驱动神经网络求解三维逆散射问题

    Multi-Level-Set-Based Physics-Driven Neural Network to Solve 3-D Inverse Scattering Problems

    [https://arxiv.org/abs/2609.08594](https://arxiv.org/abs/2609.08594)

    本文提出基于多水平集的物理驱动神经网络（LSPDNN），利用软并集多材料模型将目标支撑域与材料分布分离描述，并结合作用于材料区域指示器的模型一致TV正则化，有效缓解了三维电磁逆散射重建中的边界模糊与伪影问题。

    

    本文提出了一种基于水平集的物理驱动神经网络求解器（LSPDNN），用于求解三维电磁逆散射问题。为了缓解逐体素对比度重建中的边界模糊和重建伪影问题，所提出的求解器利用实际散射体的分段均匀性，采用多个坐标相关的神经水平集分量来表示未知目标。具体而言，提出了一种软并集多材料模型，分别描述目标支撑域和材料分布：全局支撑域由多个水平集分量的并集形成，而局部对比度则由归一化的分量权重和可学习的复介电常数候选值共同确定。此外，在材料区域指示器上施加模型一致的总变分（TV）正则化，而非直接作用于重建的对比度上，从而在抑制碎片化材料分配的同时避免过度……

    arXiv:2609.08594v1 Announce Type: new  Abstract: This paper proposes a level-set-based physics-driven neural network solver (LSPDNN) for 3-D electromagnetic inverse scattering. To mitigate boundary blurring and reconstruction artifacts in voxel-wise contrast reconstruction, the proposed solver exploits the piecewise homogeneity of practical scatterers by representing unknown targets with multiple coordinate-dependent neural level-set components. Specifically, a soft-union multi-material model is proposed to separately describe the object support and material distribution. The global support is formed by the union of multiple level-set components, while the local contrast is determined by normalized component weights and learnable complex permittivity candidates. In addition, a model-consistent total variation (TV) regularization is imposed on the material-region indicators, rather than directly on the reconstructed contrast, to suppress fragmented material assignments without excessive
    
[^365]: 重新思考单轮联邦图学习：免训练的统计估计方法

    Rethinking One-Shot Federated Graph Learning: Training-Free Statistical Estimation

    [https://arxiv.org/abs/2609.06154](https://arxiv.org/abs/2609.06154)

    该论文提出免训练框架SPEAR，将单轮联邦图学习重新表述为统计估计问题，通过计算拓扑平滑的类别原型并使用样本量自适应的收缩估计器进行聚合，有效解决了极端非独立同分布条件下本地GNN训练导致的跨客户端表示不对齐问题。

    

    单轮联邦图学习通常旨在通过单轮通信，在多个拥有互不相连子图的客户端上训练图神经网络（GNN）。现有方法主要基于“本地GNN训练不可或缺”这一前提来设计先进的优化策略。然而，实证观察表明，在极端非独立同分布条件下，本地GNN训练会产生严重的跨客户端表示不对齐问题，成为误差的主要来源而非补救手段。受此启发，我们将单轮联邦图学习重新表述为一个统计估计问题。我们提出了SPEAR（具有自适应可靠性的统计原型估计），这是一个完全免训练的框架，它直接在原始特征空间中从本地图计算拓扑平滑的类别原型。服务器随后使用样本量自适应的收缩估计器来聚合这些原型，对不可靠的本地估计进行降权处理。

    arXiv:2609.06154v1 Announce Type: new  Abstract: One-shot federated graph learning generally aims to train Graph Neural Networks (GNNs) across clients with disconnected subgraphs in a single communication round. Existing methods predominantly design advanced optimization strategies under the premise that local GNN training is indispensable. However, empirical observations reveal that under extreme non-IID conditions, local GNN training suffers from severe cross-client representation misalignment, becoming a major source of error rather than a remedy. Motivated by this, we reformulate one-shot FGL as a statistical estimation problem. We propose SPEAR (Statistical Prototype Estimation with Adaptive Reliability), a completely training-free framework that directly computes topology-smoothed class prototypes from local graphs in the original feature space. The server then aggregates these prototypes using a sample-size-adaptive shrinkage estimator that down-weights unreliable local estimate
    
[^366]: 当基因组掩码先验无法迁移时：强变异预测，弱功能生成

    When Genomic Masking Priors Fail to Transfer: Strong Variant Prediction, Weak Functional Generation

    [https://arxiv.org/abs/2609.04861](https://arxiv.org/abs/2609.04861)

    GenDA双向扩散模型在变异效应预测上虽超越同规模自回归模型，但熵引导策略对性能提升并无实际贡献，且模型在零样本功能序列填充生成任务上失败，表明基因组掩码先验能很好地迁移到变异预测、却难以迁移到功能生成。

    

    双向离散扩散模型似乎天然适合基因组建模，因为它可以从两侧侧翼序列重建缺失的序列。我们在一个附加假设下开发了GenDA（基因组密度优化的吸收态扩散模型）：熵引导的片段放置会将重建压力集中在组成复杂的区域上，从而同时改进下游变异效应预测和功能序列生成。我们的结果仅部分支持这一前提。经过监督微调后，2.02亿参数的GenDA模型在ClinVar单核苷酸变异（SNV）汇总数据上达到了0.774的AUROC，比同等规模的因果自回归模型高出0.103。然而，一个使用随机片段放置的匹配变体达到了0.777，这表明没有证据显示熵引导是ClinVar性能提升的原因。更出乎意料的是，GenDA在零样本功能填充压力测试中失败：在启动子、增强子、外显子边界和内含子边界上，它无法……（原文摘要在此处截断）

    arXiv:2609.04861v1 Announce Type: new  Abstract: Bidirectional discrete diffusion model appears naturally suited to genomic modeling because it can reconstruct missing sequence from both flanks. We developed GenDA (Genomic Density-optimized Absorbing Diffusion) under the additional hypothesis that entropy-guided span placement would concentrate reconstruction pressure on compositionally complex regions, improving both downstream variant-effect prediction and functional sequence generation. Our results only partially support this premise. After supervised fine-tuning, the 202M-parameter GenDA model reaches a pooled ClinVar SNV AUROC of 0.774, exceeding a similarly scaled autoregressive model by 0.103. However, a matched random-span variant reaches 0.777, providing no evidence that entropy guidance causes the ClinVar improvement. More unexpectedly, GenDA fails a zero-shot functional inpainting stress test: across promoters, enhancers, exon boundaries, and intron boundaries, it does not c
    
[^367]: 扩散作为无时间步迭代推理的训练课程

    Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning

    [https://arxiv.org/abs/2609.01449](https://arxiv.org/abs/2609.01449)

    该论文将扩散去噪器改造为带持久隐藏状态且无时间步条件的通用迭代更新，构建出可随推理深度不断提升准确率的随时求解器，并通过推理时持续注入高斯噪声实现单轨迹高效探索解空间，在极限数独上达到99.90%的精确求解率。

    

    扩散模型和递归推理器都是迭代式的，但它们在迭代之间传递信息的方式不同。我们在扩散去噪器中加入了一个持久的隐藏状态，并移除了其时间步条件化，留下一个可以运行到任意深度的单一共享更新。其结果是一个随时求解器：准确率随推理深度的增加而持续提升，远超训练中使用的滚动长度和反向传播窗口，在Sudoku-Extreme（极限数独）上达到99.90%的精确求解率。我们还在Maze-Unique（独特迷宫）上获得了98.93%的求解率。令人惊讶的是，推理时渐进去噪并非必要：通过在每一步用新的高斯噪声替换所有非线索变量，将腐蚀程度保持在最大值，仍能保持近乎完美的求解能力并收敛到稳定的解。这种简单的噪声注入机制使单一轨迹能够高效地探索解空间并最终得到正确答案，而无需并行滚动或候选筛选。

    arXiv:2609.01449v1 Announce Type: new  Abstract: Diffusion models and recursive reasoners are both iterative, but they carry information across iterations differently. We add a persistent hidden state to a diffusion denoiser and remove its timestep conditioning, leaving a single shared update that can be run to arbitrary depth. The result is an anytime solver: accuracy keeps improving with inference depth far beyond the rollout lengths and backpropagation window used in training, reaching 99.90% exact solve on Sudoku-Extreme. We also obtain 98.93% solve rate on Maze-Unique. Surprisingly, progressive denoising is unnecessary at inference: holding corruption at its maximum by replacing every non-clue variable with fresh Gaussian noise at each step retains near-perfect solving and converges to stable solutions. This simple noise-injection mechanism enables a single trajectory to efficiently explore the solution space and settle on the correct answer without parallel rollouts, candidate se
    
[^368]: 大语言模型能够设计近乎最优的运筹学算法

    LLMs Can Design Near-Optimal OR Algorithms

    [https://arxiv.org/abs/2608.27296](https://arxiv.org/abs/2608.27296)

    大语言模型（如gpt-5.6-sol）能够仅凭少量提示和固定计算资源，自动设计出在库存控制、排队网络和分类优化等问题上匹配或超越现有最佳方法的近乎最优运筹学算法。

    

    arXiv:2608.27296v1 公告类型：新论文 摘要：我们探究大型语言模型（LLMs）是否能够为定义明确的运筹学（OR）问题设计有效的算法。我们研究了库存控制、排队网络控制和分类优化问题。我们评估了LLM使用的两个层级：在第一层级，模型接收单个问题实例并为该实例返回解决方案；在第二层级，模型仅接收问题类别描述和广泛的参数范围，并返回一个将实例参数映射到解决方案的算法。人工输入极少：我们提供一个未经调整的提示来描述问题，模型可以使用带有固定计算预算的Python沙盒工具。我们测试的最强模型gpt-5.6-sol，在几乎所有评估实例上匹配或超越了现有最佳方法。即使在第二层级，即在看到评估实例之前返回的算法已固定的情况下，这一结论也成立。性能也随着新发布模型的迭代而显著提升。

    arXiv:2608.27296v1 Announce Type: new  Abstract: We ask whether large language models (LLMs) can design effective algorithms for well-specified operations research (OR) problems. We study inventory control, queueing network control, and assortment optimization. We evaluate two levels of LLM use: at level 1, the model receives one problem instance and returns a solution for that instance; at level 2, it receives only the problem class description and broad parameter ranges, and returns an algorithm that maps instance parameters to solutions. Human input is minimal: we give one untuned prompt that describes the problem, and the model has access to a Python sandbox tool with a fixed compute budget.   The strongest model we test, gpt-5.6-sol, matches or outperforms the best existing method on almost all evaluated instances. This holds even at level 2, where the returned algorithm is fixed before seeing the evaluation instances. Performance also improves sharply across models released less 
    
[^369]: 面向可控数据生成的机制电路识别

    Mechanistic Circuit Identification for Controllable Data Generation

    [https://arxiv.org/abs/2608.24065](https://arxiv.org/abs/2608.24065)

    本文提出了一种基于电路识别的框架，通过连接数据估值与机制可解释性，实现可控数据生成，并引入SAMS调度器以优化效用目标数据产出。

    

    arXiv:2608.24065v1 公告类型：交叉 摘要：尽管近期数据合成方面的进展旨在策展高质量数据集，但大多数生成流程仍依赖于基于提示的启发式控制。这种黑箱范式对单个样本如何与模型底层学习动态相互作用提供的洞察有限。为弥合这一差距，我们提出了一种电路基础框架，将基于训练动态的数据估值与机制可解释性（MI）相结合。具体而言，我们将数据质量概念化为三个互补的效用轴：可学习性、挑战性和对齐性。首先，我们揭示出模型内部专门电路，这些电路因果地控制这些效用信号。然后，超越启发式提示转向机制控制，我们利用这些电路作为可控接口，主动引导生成以产出效用目标数据。基于此能力，我们引入了SAMS（阶段感知机制调度），该调度器调度电路...

    arXiv:2608.24065v1 Announce Type: cross  Abstract: While recent advances in data synthesis aim to curate high-quality datasets, most generation pipelines still rely on heuristic prompt-based control. This black-box paradigm provides limited insight into how individual samples interact with a model's underlying learning dynamics. To bridge this gap, we propose a circuit-grounded framework that connects training-dynamics-based data valuation with mechanistic interpretability (MI). Specifically, we conceptualize data quality along three complementary utility axes, learnability, challenge, and alignment. First, we uncover specialized model-internal circuits that causally govern these utility signals. Then, moving beyond heuristic prompting toward mechanistic control, we leverage these circuits as controllable interfaces, actively steering generation to produce utility-targeted data. Building on this capability, we introduce SAMS (Stage-Aware Mechanistic Scheduling), which schedules circuit
    
[^370]: 基于高斯喷溅表示的物理集成算子学习

    Physics-Integrated Operator Learning via Gaussian Splatting Representations

    [https://arxiv.org/abs/2608.24049](https://arxiv.org/abs/2608.24049)

    本文提出一种表示级物理集成方法，通过前馈高斯喷溅表示将连续高斯场与闭式导数结合，直接嵌入PDE算子于学习演化中，避免优化困难并提升长期预测精度。

    

    arXiv:2608.24049v1 公告类型：新公告 摘要：神经算子为时空偏微分方程系统提供了高效的替代模型，但纯数据驱动的公式在长时间自回归预测中往往会累积显著误差，并可能无法利用可用的控制方程结构。现有方法主要通过基于残差的训练目标或特定于PDE的架构约束来整合物理知识，这可能引入优化困难或限制架构的通用性。在本工作中，我们提出了一种表示级别的物理集成方法，其中前馈高斯喷溅（FFGS）表示作为离散解场与控制算子之间的连续接口。FFGS表示将状态重建为具有闭式空间导数的连续高斯场，使得可用的物理PDE算子能够直接集成到学习到的演化映射中，而无需引入额外的复杂性。

    arXiv:2608.24049v1 Announce Type: new  Abstract: Neural operators provide efficient surrogates for spatiotemporal PDE systems, but purely data-driven formulations often accumulate substantial errors during long-horizon autoregressive prediction and may fail to exploit available governing-equation structure. Existing approaches incorporate physics primarily through residual-based training objectives or PDE-specific architectural constraints, which can introduce optimization difficulties or limit architectural generality. In this work, we introduce a representation-level approach to physics integration in which a feed-forward Gaussian splatting (FFGS) representation serves as a continuous interface between discretized solution fields and governing operators. The FFGS representation reconstructs the state as a continuous Gaussian field with closed-form spatial derivatives, allowing available physical PDE operators to be integrated directly within the learned evolution map without introduc
    
[^371]: 混合偏微分方程-参数学习中的算子误设定检测与判别：一种无参考工具，且样本内判别有界

    Detecting and Discriminating Operator Misspecification in Hybrid PDE-Parameter Learning: a Reference-Free Instrument, with Discrimination Bounded In Sample

    [https://arxiv.org/abs/2608.16925](https://arxiv.org/abs/2608.16925)

    本文提出一种无需参考信息的统计工具，通过单次拟合即可区分混合PDE参数学习中的算子误设定与参数不可识别问题，并证明了样本内判别能力有界。

    

    arXiv:2608.16925v1 公告类型：新 摘要：我们构建了一种工具，能够从单次拟合中、无需任何参考（oracle）信息，判断混合PDE-参数估计器所假定的算子是否正确，并将其与仅仅是参数不可识别的情况区分开来。在一个自伴抛物型反问题中，一种带有插入尺度和每种子参数的信息矩阵统计量，在正确设定下中位数为0.19，拒绝率为0.033（低于预先注册的0.10上限），而在两种误设定情形下分别升至224和85，并在每次重复中都触发拒绝。在正确设定但不可识别的设计下，该统计量保持沉默（n=200时为0.050，Clopper-Pearson区间[0.024, 0.090]），而秩统计量在预先注册的边界c_5^*=2.15×10^{-3}处塌缩为零。因此，对单次拟合的两种读数，能够在可部署测试所覆盖的三种设计下，区分这两种失败类型。这种区分是本文的贡献；仅检测本身已是一个拥挤的研究方向。在样本内，这是一个界限。

    arXiv:2608.16925v1 Announce Type: new  Abstract: We build an instrument that reads, from a single fit and with no oracle, whether the operator a hybrid PDE-parameter estimator postulates is wrong-and separates that from a merely unidentifiable parameter. On one self-adjoint parabolic inverse problem, an information-matrix statistic with plug-in scale and per-seed parameter has median 0.19 under correct specification, rejection rate $0.033$ against a pre-registered ceiling of $0.10$, and rises to $224$ and $85$ under two misspecifications, firing in every replicate. On a correctly specified but non-identifiable design it stays mute-$0.050$ at $n=200$, Clopper-Pearson $[0.024, 0.090]$-while a rank statistic collapses to zero at a pre-registered boundary $c_5^*=2.15\times10^{-3}.$ Two readings of one fit therefore separate the two failures across the three designs a deployable test reaches. That separation is the contribution; detection alone is a crowded flank. In sample it is a bound, o
    
[^372]: 平衡力迫：无需噪声条件化的自适应视频生成

    Equilibrium Forcing: Adaptive Video Generation Without Noise Conditioning

    [https://arxiv.org/abs/2608.14706](https://arxiv.org/abs/2608.14706)

    本文提出平衡力迫（EqF）框架，通过去除噪声水平条件化并解耦去噪与采样，实现了自适应闭环推理，显著提升了自回归视频生成的质量和一致性。

    

    arXiv:2608.14706v1 公告类型：交叉 摘要：基于扩散和流匹配的标准自回归视频生成算法依赖于刚性训练目标和静态采样计划，限制了推理过程对数据的适应性。我们引入了平衡力迫（EqF），一种无需噪声水平条件化的视频去噪生成模型的简化框架。EqF开创了模块化的训练和推理时设计，用于无噪声条件生成，将学习去噪场与采样解耦。这种灵活性允许推理时算法通过适应样本反馈进行闭环操作，在具有挑战性的自回归视频生成基准上提高了视频质量和一致性。广泛的分析精确揭示了移除噪声水平条件化如何使EqF的数据依赖推理特性超越标准噪声水平条件化去噪视频方法的性能。

    arXiv:2608.14706v1 Announce Type: cross  Abstract: Standard autoregressive video generation algorithms based on Diffusion and Flow Matching rely on rigid training objectives and static sampling schedules, limiting inference procedures from adapting to the data. We introduce Equilibrium Forcing (EqF), a simplified framework for video denoising generative models without noise level conditioning. EqF pioneers modular training- and inference-time designs for noise-unconditional generation that decouple learning the denoising field from sampling. This flexibility allows for inference-time algorithms that operate in a closed loop by adapting to feedback from the sample, improving video quality and consistency on challenging autoregressive video generation benchmarks. Extensive analysis elucidates exactly how removing the noise level conditioning enables EqF's data-dependent inference properties to surpass the performance of standard noise level-conditional denoising video methods.
    
[^373]: 不确定性感知的成对纠缠监督下的概率约束聚类

    Uncertainty-Aware Probabilistic Constrained Clustering from Entangled Pairwise Supervision

    [https://arxiv.org/abs/2608.12027](https://arxiv.org/abs/2608.12027)

    本文提出了一种不确定性感知的概率约束聚类方法，通过角度成对目标和估计-校正-集成框架，有效处理现实中的软性、纠缠成对监督，提高了聚类性能。

    

    arXiv:2608.12027v1 公告类型：交叉 摘要：成对约束聚类通常依赖于硬性的必须连接/不能连接标签，而现实的成对监督可能是实值且包含内在模糊性、专家判断和随机损坏。现有的深度约束聚类（DCC）方法主要针对硬性、专家无关的约束，将软标签大多数值化处理而非语义化理解。我们将这一设置形式化为不确定性感知的概率约束聚类（UPCC），通过异构观察过程定义规范的天生不确定性目标，并分析其条件可识别性。我们引入了ProbPair，一种用于概率关系的角度成对目标，并构建了ECI-PP，一个估计-校正-集成框架，通过信念估计、校正和可靠性感知集成来细化不完善的监督。在具有挑战性的概率监督设置中，我们在多个基准上的实验...

    arXiv:2608.12027v1 Announce Type: cross  Abstract: Pairwise constrained clustering typically relies on hard must-link/cannot-link labels, whereas realistic pairwise supervision may be real-valued and entangle intrinsic ambiguity, expert judgment, and stochastic corruption. Existing deep constrained clustering (DCC) methods mainly target hard, expert-agnostic constraints, treating soft labels mostly numerically rather than semantically. We formalize this setting as uncertainty-aware probabilistic constrained clustering (UPCC), defining a canonical aleatoric target through a heterogeneous observation process and analyzing its conditional identifiability. We introduce ProbPair, an angular pairwise objective for probabilistic relations, and build ECI-PP, an estimator--corrector--integrator framework that refines imperfect supervision via belief estimation, correction, and reliability-aware integration. Across challenging probabilistic supervision settings, experiments on diverse benchmarks
    
[^374]: 橡皮筋丢弃法：一种缓解基于评分标准奖励强化学习中奖励黑客问题的简单方法

    Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL

    [https://arxiv.org/abs/2608.11669](https://arxiv.org/abs/2608.11669)

    本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。

    

    摘要：基于评分标准的强化学习，即由LLM评判员根据一系列标准进行评分，已成为对没有确定性答案的任务进行语言模型后训练的标准方法。然而，评分标准是质量的固定代理，永远无法完全描述质量，长期针对其训练的策略会学会利用这一差异。我们直接对此进行了测量。使用组相对策略优化（GRPO）在医学和科学评分标准上训练Qwen3-8B，并使用训练评判员和更强的黄金评判员对分布外（OOD）基准进行评分，我们发现这两个分数在训练过程中出现分歧。训练评判员的分数持续上升，而黄金评判员的分数达到峰值后下降，在HealthBench-Hard上下降3分，在ResearchQA上下降22分。如果评判员存在固定偏差，黄金曲线只会发生恒定偏移，而不会在训练分数上升时下降，因此这种分歧是奖励黑客问题，而非评判员偏差。

    arXiv:2608.11669v1 Announce Type: cross  Abstract: Reinforcement learning against rubrics, lists of criteria graded by an LLM judge, has become a standard way to post-train language models on tasks with no deterministic answer. The rubric, however, is a fixed proxy for quality, never a complete description of it, and a policy trained against it long enough will learn to exploit the difference. We measure this directly. Training Qwen3-8B with Group Relative Policy Optimization (GRPO) on medical and science rubrics and grading out-of-distribution (OOD) benchmarks with both the training judge and a stronger gold judge, we find that the two scores diverge during training. The training judge's score keeps climbing while the gold judge's score peaks and then falls, by 3 points on HealthBench-Hard and by 22 points on ResearchQA. A judge with a fixed bias would shift the gold curve by a constant, not send it down while the training score rises, so the divergence is reward hacking, not judge no
    
[^375]: FACT：面向世界-动作模型的失败感知因果训练

    FACT: Failure-Aware Causal Training for World-Action Models

    [https://arxiv.org/abs/2608.10232](https://arxiv.org/abs/2608.10232)

    FACT提出了一种失败感知的因果训练方法，让世界-动作模型以执行动作为条件预测未来视频和任务进度，从而利用失败轨迹监督动作后果，将失败数据转化为有效的训练信号而非被丢弃。

    

    近期的世界-动作模型表明，将策略与未来预测进行联合训练可以为动作生成提供物理先验。基于视频模型的未来预测能力，许多世界-动作模型通过生成未来视频并借助逆动力学模型恢复动作，或将这些预测视频用作动作生成的目标条件。在这两种情况下，世界模型主要在成功演示的数据上训练，几乎没有理由去预测不良动作的后果。我们提出了FACT，这是一个因果世界-动作模型，它以执行的动作条件预测未来视频和任务进度。这种以动作为条件的接口使得失败轨迹能够监督动作后果，将坏动作转化为有效的未来预测目标，而不是被直接丢弃。失败感知训练使进度预测器能够同时感知成功和失败的动作结果，这可以选择性地用于对样本进行评分（摘要在此处截断）。

    arXiv:2608.10232v2 Announce Type: replace-cross  Abstract: Recent world-action models (WAMs) show that co-training policies with future prediction can provide physical priors for action generation. Building on the future-prediction ability of video models, many WAMs generate future videos and recover actions with inverse-dynamics models, or use these predicted videos as goal conditions for action generation. In both cases, the world model is trained mostly on successful demonstrations and has little reason to predict the consequences of bad actions. We introduce FACT, a causal World-Action Model that predicts future video and task progress conditioned on the executed action. This action-conditioned interface allows failure rollouts to supervise action consequences, turning bad actions into valid future targets rather than being discarded. Failure-aware training makes the progress predictor aware of both successful and failed action outcomes, which can optionally be used to score sample
    
[^376]: 在CIFAR-10上比较SGLD与固定噪声预测-校正采样器在经典联合能量模型中的改进应用

    Comparing SGLD and a fixed-noise predictor-corrector adaptation in canonical joint energy-based models on CIFAR-10

    [https://arxiv.org/abs/2608.05025](https://arxiv.org/abs/2608.05025)

    该论文系统复现了经典联合能量模型（JEM），并首次检验了用固定噪声预测-校正采样器替代SGLD的效果，结果显示其分类性能接近经典方法但生成质量略有差距，同时记录了两种失败模式。

    

    联合能量模型将分类与生成统一在单个网络中，并支持分布外（OOD）检测。经典的JEM训练依赖于随机梯度朗之万动力学（SGLD）；而一种有理论依据的替代方法——预测-校正（PC）采样器——此前尚未在经典模型上经过系统的复现检验。我们在两次独立运行中复现了基于WideResNet-28-10（不含归一化层）的经典JEM，并测试了一种固定噪声PC改进方法——用确定性梯度步骤替代存在缺陷的退火噪声预测器——涵盖三种协议：改进采样器在整个训练轨迹（115-132个epoch）中替代SGLD；冷启动生成（FID）；以及精炼式多OOD检测（AUROC）。复现结果达到92.88%的测试准确率和44.46的buffer-FID（经典方法为92.90%和38.40）。我们记录了两种失败模式：灾难性……

    arXiv:2608.05025v3 Announce Type: replace  Abstract: Joint Energy-Based Models (JEM) unify classification and generation within a single network and support out-of-distribution (OOD) detection. Canonical JEM training relies on stochastic gradient Langevin dynamics (SGLD); a theoretically motivated alternative, the Predictor-Corrector (PC) sampler, has not previously undergone a systematic replication test on the canonical model. We reproduce canonical JEM on WideResNet-28-10 without normalisation layers on two independent runs and test a fixed-noise PC adaptation - with the degenerate annealed-noise predictor replaced by a deterministic gradient step - across three protocols: the adapted sampler replacing SGLD throughout the full training trajectories (115-132 epochs); cold-start generation (FID); and refinement-style multi-OOD detection (AUROC). The reconstruction reaches 92.88% test accuracy and buffer-FID 44.46 (canonical: 92.90% and 38.40). We document two failure modes: catastroph
    
[^377]: SPARK：从推测性GRPO采样中的验证学习

    SPARK: Learning from Verification in Speculative GRPO Rollouts

    [https://arxiv.org/abs/2608.04962](https://arxiv.org/abs/2608.04962)

    提出SPARK，一种轻量级自适应机制，利用目标模型的验证反馈在推测性GRPO训练的采样过程中实时修正草稿模型的预测遗漏，从而提升训练效率。

    

    组相对策略优化（GRPO）已成为提升大语言模型推理能力的标准方法，但其训练成本高昂，因为每一步都需要生成大量冗长的推理响应。近期方法采用推测解码来加速采样过程，即由草稿模型提出候选token，再由目标模型进行验证。然而，我们发现草稿模型在单个响应内以及同一GRPO组内的多个响应之间会反复遗漏相同的token，而这些遗漏处的验证反馈并未用于在采样过程中更新草稿模型。基于这一分析，我们提出了SPARK，一种轻量级的自适应机制，旨在增强推测性GRPO训练中使用的草稿模型。SPARK利用目标模型的验证反馈在生成过程中修正草稿模型的预测，同时保留其原有的训练流程。在21个模型-数据集设置上的平均结果显示……

    arXiv:2608.04962v2 Announce Type: replace-cross  Abstract: Group Relative Policy Optimization (GRPO) has become a standard approach for improving the reasoning abilities of large language models, but its training is costly because each step requires generating many long reasoning responses. Recent methods use speculative decoding to accelerate the rollouts, where a draft model proposes tokens that the target model verifies. However, we find that the draft model repeatedly misses the same tokens within a response and across responses in the same GRPO group, and the verification feedback on these misses is not used to update the draft during the rollout. Motivated by this analysis, we introduce SPARK, a lightweight self-adaptive mechanism designed to enhance draft models used in speculative GRPO training. SPARK leverages target verification feedback to refine draft predictions during generation while preserving their existing training procedures. Averaged across 21 model-dataset settings
    
[^378]: 替代替换保持PHI可检测性：一项多检测器等价性研究

    Surrogate Substitution Preserves PHI Detectability: A Multi-Detector Equivalence Study

    [https://arxiv.org/abs/2608.03172](https://arxiv.org/abs/2608.03172)

    本文提出了一种成对多检测器评估协议，通过等价性检验和仅在掩盖跨度上评分，验证结构保留的PHI替代物能否被下游检测器可靠识别，并区分生成器缺陷与检测器固有局限。

    

    摘要：arXiv:2608.03172v3 公告类型：替换 摘要：结构保留的去标识化将受保护健康信息（PHI）替换为逼真的同类型替代物——“Anna S.”变为“Maria S.”，而非[姓名]——从而保持临床文本流畅，并确保下游工具继续正常工作。但这仅在替换本身不破坏这些工具所依赖的信号时才有帮助。我们提出一个狭窄且可检验的问题：在去标识器实际掩盖的跨度上，下游PHI检测器是否仍能发现替代物？我们引入了一种成对的多检测器评估协议，该协议（i）仅在掩盖的跨度上评分效用，将覆盖范围与效用解耦；（ii）使用等价性检验（TOST）而非零假设显著性检验，后者在我们的样本量（57,000个成对跨度）下无信息量；（iii）构建了一个替代物失败类型学，将可修复的生成器缺陷与固有的检测器限制区分开来。在11个检测器、7个基准和7种语言（1,750篇文档）上，召回率...

    arXiv:2608.03172v3 Announce Type: replace  Abstract: Structure-preserving de-identification replaces protected health information (PHI) with realistic same-type surrogates -- "Anna S." becomes "Maria S.", not [NAME] -- so that clinical text stays fluent and downstream tools keep working. But this only helps if the substitution does not itself corrupt the signal those tools rely on. We ask a narrow, testable question: on the spans a de-identifier actually masks, can downstream PHI detectors still find the surrogate? We introduce a paired, multi-detector evaluation protocol that (i) scores utility only on masked spans, decoupling coverage from utility; (ii) uses equivalence testing (TOST) rather than null-hypothesis significance testing, which is uninformative at our sample size (57k paired spans); and (iii) builds a surrogate-failure typology separating fixable generator defects from intrinsic detector limits. Across 11 detectors, 7 benchmarks, and 7 languages (1,750 documents), recall 
    
[^379]: QASP：查询自适应的鲁棒向量搜索策略

    QASP: Query-Adaptive Robust Vector Search Policy

    [https://arxiv.org/abs/2607.29606](https://arxiv.org/abs/2607.29606)

    QASP通过单次预先监督回归预测每个查询的完整召回率进展曲线，从而为任意召回率目标推导出自适应搜索策略，在跨目标、索引配置和数据集泛化的同时兼顾高召回率与低计算成本。

    

    向量搜索的一个根本挑战是在最小化计算成本的同时实现持续的高召回率。固定的搜索参数会导致不同查询之间显著的性能差异，而传统的基于平均召回率的评估方式掩盖了这些逐查询的差异。我们提出了QASP（查询自适应鲁棒向量搜索策略），它通过一次预先的监督回归预测每个查询的完整召回率进展曲线，并据此为任意召回率目标推导出搜索策略；这避免了搜索过程中的迭代模型调用，也无需为每个目标单独设置预测器。通过利用尺度不变特征进行搜索前推理来预测归一化召回率值，QASP能够跨召回率目标、索引配置和数据集进行泛化。细粒度的进展预测还进一步支持一种轻量级的反应式补充机制，可根据预测值与观测值之间的偏差调整搜索深度，而无需额外的……（摘要原文在此处截断）

    arXiv:2607.29606v3 Announce Type: replace-cross  Abstract: A fundamental challenge of vector search is achieving consistently high recall while minimizing computational costs. Fixed search parameters cause significant performance variance across queries, and conventional evaluation on average recall masks these per-query disparities. We introduce QASP (Query-Adaptive robust vector Search Policy), which predicts the complete recall progression curve per query via a single upfront supervised regression, from which a search policy is derived for any recall target; this avoids iterative model invocations during search or separate predictors per target. By predicting normalized recall values with scale-invariant features and pre-search inference, QASP generalizes across recall targets, index configurations, and datasets. The fine-grained progress predictions further enable a lightweight reactive complement that adjusts search depth based on predicted-versus-observed deviations without addit
    
[^380]: 设计矩阵的零模式驱动过参数化回归中的多重下降

    The Zero Pattern of a Design Matrix Drives Multiple Descent in Over-parameterized Regression

    [https://arxiv.org/abs/2607.24041](https://arxiv.org/abs/2607.24041)

    本文放宽了过参数化线性回归中协变量独立且协方差矩阵非退化的假设，证明协方差退化与变量相关性会引发多重下降现象，并借助二部图的最大匹配和Dulmage-Mendelsohn分解精确定位了风险峰值出现的位置。

    

    过参数化线性回归在过去十年中得到了广泛研究。然而，大多数现有工作假设协变量是相互独立的，且其协方差矩阵是非退化的。本文放宽了这两个假设，并在岭正则化趋于零的情形下推导出预测风险的确定性等价形式。我们证明协方差矩阵的退化性和变量间的相关性可以导致多重下降现象，并刻画了相应峰值可能出现的位置。我们的证明采用了一种新颖的方差轮廓图表示方法。我们证明了相关二部图的最大匹配和Dulmage--Mendelsohn分解能够识别出方差变得奇异的配置。

    arXiv:2607.24041v2 Announce Type: replace-cross  Abstract: Over-parameterized linear regression has been widely studied over the last decade. However, most existing works assume that the covariates are independent and that their covariance matrices are non-degenerate. In this paper, we relax both assumptions and derive deterministic equivalents for the prediction risk in a vanishing-ridge regime. We show that degeneracy of the covariance matrices and dependence can lead to multiple descent, and characterize where the corresponding peaks can occur. Our proofs use a novel graph representation of the variance profile. We show that maximum matchings and the Dulmage--Mendelsohn decomposition of the associated bipartite graph identify the configurations at which the variance becomes singular.
    
[^381]: Hopformer：面向时间序列预测的同质性追求Transformer

    Hopformer: Homogeneity-Pursuit Transformer for Time Series Forecasting

    [https://arxiv.org/abs/2607.22299](https://arxiv.org/abs/2607.22299)

    Hopformer提出两阶段时间序列预测框架，先通过稀疏模式聚合（SPA）提取融合协变量的共同低方差趋势作为同质化层，再用LoRA微调的Transformer建模残差中的复杂依赖，在理论上证明了近最优偏差-方差权衡与泛化界，在多个基准上平均提升MASE 6.56%。

    

    对包含高维协变量的多条时间序列进行预测面临一个核心挑战：如何在统一共同的时间模式的同时保留有意义的序列特定信息。我们提出了Hopformer（同质性追求Transformer），一个解决该挑战的两阶段框架。在第一阶段，我们执行稀疏模式聚合（SPA）方案，提取一个融合协变量的共同低方差趋势，这充当了一个同质化层。在第二阶段，一个经LoRA微调的Transformer对残差中剩余的复杂依赖关系进行建模。我们的方法具有坚实的理论基础：我们通过oracle不等式证明SPA实现了近乎最优的偏差-方差权衡，并在相依时间序列数据下为第二阶段提供了泛化界。Hopformer创造了新的最先进水平，在合成和真实世界预测基准上平均将MASE提升了6.56%。

    arXiv:2607.22299v2 Announce Type: replace-cross  Abstract: Forecasting multiple time-series with high-dimensional covariates presents a core challenge: unifying common temporal patterns while retaining meaningful series-specific information. We introduce Hopformer (Homogeneity-Pursuit Transformer), a two-stage framework that addresses this challenge. In the first stage, we perform a Sparsity Pattern Aggregation (SPA) scheme extracting a common low-variance trend that incorporates the covariates. This acts as a homogenization layer. In the second stage, a LoRA-fine-tuned Transformer models the remaining complex dependencies in the residual. Our method is theoretically grounded. We prove that SPA achieves a near-optimal bias-variance trade-off via an oracle inequality. We also provide generalization bounds for the second stage under dependent time series data. Hopformer sets a new state of the art, improving MASE by an average of 6.56% across synthetic and real-world forecasting benchmar
    
[^382]: 修复洞见：利用照片级逼真的扰动提升视觉可解释人工智能（XAI）

    Inpainting Insights: Elevating Visual XAI with Photorealistic Perturbations

    [https://arxiv.org/abs/2607.15482](https://arxiv.org/abs/2607.15482)

    该论文改进了广泛使用的基于扰动的方法LIME，利用生成式图像修复技术生成照片级逼真的扰动样本，解决了传统扰动方法产生不真实分布外样本和可见伪影的问题，从而提升视觉XAI的图像解释质量。

    

    最先进的机器学习模型日益复杂，使其行为越来越难以解释，这推动了可解释人工智能（XAI）领域的快速发展。在众多已提出的方法中，基于扰动的方法扮演着重要角色。这类方法通过系统地改变（扰动）输入特征，来衡量其对模型预测的影响。对于图像数据，传统的扰动技术通常涉及用预定义颜色等替换像素值。然而，这类方法以及更精细的确定性技术，都会生成不真实的分布外样本，并常常留下可见的伪影，这可能误导模型并损害解释质量。在这项工作中，我们对广泛使用的基于扰动的方法LIME进行了调整，以展示生成式图像修复如何改进基于扰动的图像解释。我们实现了照片级逼真的……

    arXiv:2607.15482v2 Announce Type: replace  Abstract: The increasing complexity of state-of-the-art machine learning models has made their behavior progressively harder to interpret, spurring rapid advancements in the field of eXplainable Artificial Intelligence (XAI). Among many methods proposed, perturbation-based approaches play a major role. By systematically altering (perturbing) input features, these approaches measure the impact on the model's predictions. For image data, traditional perturbation techniques, often involve replacing pixel values e.g., with a pre-defined color. However, such approaches, but also more refined deterministic techniques, generate unrealistic out-of-distribution samples and often leave visible artifacts, which can mislead the model and compromise explanation quality. In this work, we adjust LIME, a widely used perturbation-based method, to demonstrate how generative inpainting can improve perturbation-based explanations for images. We achieve photoreali
    
[^383]: 有向图上基于流恢复势能的规范不变、参数不敏感正则化方法

    Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs

    [https://arxiv.org/abs/2607.13609](https://arxiv.org/abs/2607.13609)

    该论文提出用规范不变的图狄利克雷能量替代岭正则化，从有向图流数据中恢复势能，解决了岭正则化导致的排序反转问题，并实现了跨λ四个数量级的参数不敏感性。

    

    从有向图上观测到的流中恢复潜在势能（即带狄利克雷边界的离散泊松问题）是一个不适定问题，而标准的解决方案反而适得其反：岭正则化会向规范无意义的原点收缩，导致恢复的排序发生坍塌和反转（相对于植入的真实值，秩相关从+0.81变为-0.42）。规范不变的图狄利克雷能量消除了这一风险，并带来了参数不敏感性：估计值在λ的四个数量级范围内保持稳定，而岭正则化在所有λ>0的情况下都会反转排序。作者证明了约化求解是SPD（对称正定）的，并在岭正则化使其坍塌的地方精确保存了动态范围，还通过泊松残差仅从流中定位吸收边界。H^1半范数本身是经典的；新颖之处在于规范诊断、由此获得的参数不敏感性，以及表明结果对提取方法稳健的消融实验。在三个公开的点击流数据集上……（摘要被截断）

    arXiv:2607.13609v2 Announce Type: replace  Abstract: Recovering a latent potential from observed flow on a directed graph (a discrete Poisson problem with Dirichlet boundaries) is ill-posed, and the standard fix backfires: ridge regularization shrinks toward a gauge-meaningless origin, collapsing and reversing the recovered ordering ($+0.81\to-0.42$ rank correlation against a planted ground truth). The gauge-invariant graph Dirichlet energy removes the hazard and delivers parameter-insensitivity: the estimate is stable across four orders of magnitude in $\lambda$, whereas ridge inverts the ordering for every $\lambda>0$. We prove the reduced solve is SPD and preserves dynamic range exactly where ridge collapses it, and localize absorbing boundaries from flow alone via a Poisson residual. The $H^1$ seminorm is classical; what is new is the gauge diagnosis, the parameter-insensitivity it buys, and an ablation showing the result is robust to the extraction method. On three public clickstr
    
[^384]: 将通用车辆模型适配用于跨地形高速模型预测控制（MPC）

    Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains

    [https://arxiv.org/abs/2607.13319](https://arxiv.org/abs/2607.13319)

    OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。

    

    高速越野自动驾驶需要对目标车辆进行精确的闭环控制，同时在多变地形上保持鲁棒性。近期的前向运动动力学（FKD）预测基础模型展现出一条有前景的路径：从通用模型出发，将其专用于目标平台。然而，有效的专业化仍然具有挑战性，因为它通常需要大量真实世界数据，而且适配到某一环境的模型仍可能对特定地形或驾驶模式过拟合。我们提出了OptCar（优化汽车），这是一种弥合通用FKD模型到专用FKD模型差距的方法，在为特定车辆优化性能的同时保留了跨地形泛化能力。OptCar引入了一种基于transformer的FKD架构，利用FiLM将多步预测条件化于单个动力学上下文token，该token概括了近期的状态-动作历史。随后，该方法利用有限的真实世界数据对通用模型进行专业化（摘要在此处截断）。

    arXiv:2607.13319v2 Announce Type: replace-cross  Abstract: High-speed off-road autonomy requires precise closed-loop control for a target vehicle while remaining robust across changing terrains. Recent forward kinodynamic (FKD) prediction foundation models suggest a promising path, starting from a generalist model and specializing it to the target platform. However, effective specialization remains challenging, as it often requires substantial real-world data, and models adapted to one setting can still overfit to specific terrains or driving regimes. We present OptCar (Optimized Car), a recipe for bridging the gap from generalist to specialist FKD models that preserves cross-terrain generalization while optimizing performance for a specific vehicle. OptCar introduces a transformer FKD architecture that uses FiLM to condition multi-step predictions on a single dynamics context token summarizing recent state-action history. It then specializes the generalist model using limited real-wor
    
[^385]: AdaPCLA：面向长尾纵向电子健康记录生成的自适应先验校准Logit调整

    AdaPCLA: Adaptive Prior-Calibrated Logit Adjustment for Long-Tailed Longitudinal EHR Generation

    [https://arxiv.org/abs/2607.12645](https://arxiv.org/abs/2607.12645)

    提出AdaPCLA框架，通过数据分布感知的模拟退火训练将先验知识内化到生成模型中，并结合零样本分布控制实现无需训练即可适应多样化临床人群的长尾纵向电子健康记录生成。

    

    纵向电子健康记录（EHR）的生成式建模对于隐私保护研究日益重要，然而标准的自回归模型往往难以充分表达尾部事件（即疾病、症状）的共现结构，从而降低了生成数据对罕见亚群体的保真度和忠实性。为此，我们提出了AdaPCLA框架，该框架使生成模型能够通过数据分布感知的训练策略自适应地拟合和生成EHR数据；这是通过模拟退火训练将数据知识内化到模型参数中实现的。同时，该框架支持通过零样本分布控制，无需再训练即可适应多样化的临床人群进行数据生成。此外，我们的理论分析通过标签级经验神经正切核（NTK）刻画了罕见编码的logit更新，并推导出先验内化边界，说明了退火速度和NTK条件数如何影响所保留的先验信号。

    arXiv:2607.12645v2 Announce Type: replace  Abstract: Generative modeling of longitudinal Electronic Health Records is increasingly important for privacy-preserving research, yet standard autoregressive models tend to underrepresent the co-occurrence structure of tail events (i.e., diseases, symptoms), reducing the fidelity and faithfulness of generated data for rare subpopulations. To this end, we propose AdaPCLA framework, which enables generative models to adaptively fit and generate EHR data through a data distribution-aware training strategy; this is achieved by internalizing data knowledge parameters by simulated annealing training. It also supports training-free adaptation to a diverse clinical population for generation through zero-shot distribution control. Moreover, our theoretical analysis characterizes rare-code logit updates through the label-wise empirical NTK and derives a prior-internalization bound for how annealing speed and NTK conditioning affect retained prior signa
    
[^386]: tidyHEBO：具备模型一致性变换与帕累托搜索的鲁棒通用贝叶斯优化

    tidyHEBO: Robust General-Purpose Bayesian Optimization with Model-Consistent Warping and Pareto Search

    [https://arxiv.org/abs/2607.10669](https://arxiv.org/abs/2607.10669)

    tidyHEBO 是一个 BoTorch 原生的通用贝叶斯优化器，通过将 Yeo-Johnson 输出变换与高斯过程代理模型联合拟合、在原始目标尺度上评估采集函数并执行多准则累积帕累托搜索，在无需任何基准特定调优的情况下于 Olympus 基准上排名第一。

    

    贝叶斯优化（BO）被广泛用于昂贵的黑盒优化问题，然而其实际性能不仅取决于高层的算法选择，还取决于代理模型训练、输入输出变换、采集函数以及候选点搜索等环节的具体实现方式。我们提出了 tidyHEBO，一个基于 BoTorch 原生实现的单目标优化器，专为鲁棒的通用优化而设计。tidyHEBO 将 Yeo-Johnson 输出变换与高斯过程代理模型进行联合拟合，使用确定性求积或蒙特卡洛采样在原始目标尺度上评估采集函数，并对多个采集准则执行受约束的累积帕累托搜索。在没有任何针对 Olympus 基准的超参数调优——仅使用默认优化器配置的情况下——tidyHEBO 在 Olympus 基准的受评估方法中排名第一。它在典型性能上取得了最佳平均排名（平均排名 1.53...）

    arXiv:2607.10669v2 Announce Type: replace  Abstract: Bayesian optimization (BO) is widely used for expensive black-box problems, yet practical performance depends not only on high-level algorithmic choices but also on how surrogate model training, input and output warping transformations, acquisition functions, and candidate search are implemented. We present tidyHEBO, a BoTorch-native single-objective optimizer designed for robust general-purpose optimization. tidyHEBO jointly fits Yeo-Johnson output warping with the Gaussian-process surrogate, evaluates acquisition functions on the original objective scale using deterministic quadrature or MC-samples, and performs constrained cumulative Pareto search over multiple acquisition criteria. Without any Olympus-specific hyperparameter tuning - using only default optimizer configurations - tidyHEBO ranked first among the evaluated methods on the Olympus benchmark. It achieved the best average ranks for typical performance (average rank 1.53
    
[^387]: 深度在复合预测中能否保持优势，路由器能否利用它？潜在世界模型中的自适应计算

    Does Depth Survive Composition, and Can a Router Use It? Adaptive Compute in Latent World Models

    [https://arxiv.org/abs/2607.10203](https://arxiv.org/abs/2607.10203)

    该论文通过在九个 DeepMind Control 任务上测试十二模块早退潜在世界模型发现，“更深预测器更准确”的假设在部分任务上不成立且多步复合会改变深度-误差关系，尽管逐步选择深度的贪婪 oracle 可将 rollout 误差降低 10-37%，但训练出的所有路由器均无法捕捉这种深度选择收益。

    

    自适应计算世界模型在某些 rollout 步骤上运行的预测器模块数量少于其他步骤。这依赖于两个假设：更深的预测器能带来更准确的 rollout，以及需要深度的步骤可以在预测器运行之前被识别出来。我们在九个 DeepMind Control 任务上对这两个假设进行了检验，使用的是十二模块的早退（early-exit）潜在预测器，每个任务八个随机种子。第一个假设在三个任务上失败：在这些任务中，较浅的退出点在每个种子上都比完整堆栈 rollout 得更准确，并且将预测复合十步会使深度-误差关系在两个方向上都偏离其单步形状。事后来看，逐步选择深度是有帮助的：一个能看到真实下一步潜在状态的贪婪 oracle，在相同平均计算量下，其 rollout 误差在所有九个任务上均比最佳固定深度混合低 10-37%，比最佳步索引调度低 8-28%。但我们训练的任何路由器都无法捕捉到这一点。这些难度路由器……（原文摘要在此处截断）

    arXiv:2607.10203v4 Announce Type: replace  Abstract: Adaptive-compute world models run fewer predictor blocks on some rollout steps than on others. This rests on two assumptions: that a deeper predictor gives a more accurate rollout, and that the steps which need depth can be identified before the predictor runs. We test both on nine DeepMind Control tasks with a twelve-block early-exit latent predictor, eight seeds per task. The first assumption fails on three tasks, where a shallower exit rolls out more accurately than the full stack on every seed, and composing predictions over ten steps moves the depth-error relation away from its one-step shape in both directions. In hindsight, choosing depth per step helps: a greedy oracle that sees the true next latent has 10-37% lower rollout error than the best mixture of fixed depths at equal average compute, and 8-28% lower than the best step-index schedule, on all nine tasks. No router we trained captures this. The difficulty router these m
    
[^388]: 面向大语言模型路由的具有代理奖励的相关性感知上下文多臂老虎机

    Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing

    [https://arxiv.org/abs/2607.09015](https://arxiv.org/abs/2607.09015)

    该论文提出了面向大语言模型路由的相关性感知上下文多臂老虎机算法，通过耦合奖励混合与解耦预测混合两种互补设计利用可能不可靠的代理奖励信号，在加速学习的同时对代理信号错误设定保持鲁棒性。

    

    我们研究了具有相关臂并可获得由机器学习模型产生的代理奖励信号的上下文多臂老虎机问题，其动机来自大语言模型（LLM）路由等应用。与经典上下文多臂老虎机仅依赖老虎机反馈并假设臂间条件独立不同，我们的设定允许依赖上下文的臂间相关性，以及可能存在噪声或错误设定的辅助奖励信息。我们提出了通过两种互补设计来利用此类代理奖励的算法。一种耦合的奖励混合方法将真实奖励与代理奖励汇集在一起，在代理信号可靠时加速学习；而一种解耦的预测混合方法则为老虎机反馈和代理奖励分别维护独立的估计器，并自适应地组合它们的预测。这种解耦带来了对代理错误设定的鲁棒性，能够恢复与仅使用真实奖励相当的后悔保证。

    arXiv:2607.09015v2 Announce Type: replace  Abstract: We study contextual bandit problems with correlated arms and access to surrogate reward signals produced by a machine learning model, motivated by applications such as large language model (LLM) routing. Unlike classical contextual bandits that rely solely on bandit feedback and assume conditional independence across arms, our setting allows context-dependent inter-arm correlations and auxiliary reward information that may be noisy or misspecified. We propose algorithms that leverage such surrogate rewards through two complementary designs. A coupled reward-mixing approach pools true and surrogate rewards to accelerate learning when surrogate signals are reliable, while a decoupled prediction-mixing approach maintains separate estimators for bandit feedback and surrogate rewards and adaptively combines their predictions. This decoupling yields robustness to surrogate misspecification, recovering regret guarantees comparable to reward
    
[^389]: RhyMix：一种用于长期时间序列预测的轻量级自适应多节奏网络

    RhyMix: A Lightweight Adaptive Multi-Rhythm Network for Long-Term Time Series Forecasting

    [https://arxiv.org/abs/2607.08234](https://arxiv.org/abs/2607.08234)

    RhyMix提出了一种轻量级的并行双路径混合神经架构，通过带可学习循环嵌入的循环路径与自适应门控机制，同时捕获时间序列中的季节周期、短期波动和长期趋势等多种模式，从而提升长期时间序列预测性能。

    

    现实世界的时间序列表现出复杂的动态特性，其特征是多种时间模式同时存在：短期波动、周期性季节循环、长期趋势以及不规则的突变。然而，大多数预测架构依赖于单路径时间建模——Transformer能够捕获长程依赖但会平滑局部变化，卷积能够捕获局部模式但感受野有限，线性模型高效但无法捕获非线性动态——因此没有任何单一的架构范式能够同时捕获所有这些模式类型。为解决这一问题，我们提出了RhyMix（RHYthm MIXture，节奏混合），一种围绕并行双路径建模范式与自适应门控机制设计的混合神经架构。RhyMix集成了两个互补的编码分支：(i) 循环路径，通过可学习的循环嵌入引入显式的季节性归纳偏置；以及(ii) 轻量级Mult……（注：原始摘要在此处被截断，内容不完整）

    arXiv:2607.08234v2 Announce Type: replace  Abstract: Real-world time series exhibit complex dynamics characterized by multiple simultaneous temporal patterns: short-term fluctuations, periodic seasonal cycles, long-term trends, and irregular abrupt changes. However, most forecasting architectures rely on single-path temporal modeling---transformers capture long-range dependencies but smooth local variations, convolutions capture local patterns but have limited receptive fields, and linear models are efficient but cannot capture nonlinear dynamics---so no single architectural paradigm captures all these pattern types simultaneously. To address this, we introduce RhyMix (RHYthm MIXture), a hybrid neural architecture designed around a parallel dual-path modeling paradigm with adaptive gating mechanisms. RhyMix integrates two complementary encoding branches: (i) a Cyclic Path that incorporates explicit seasonal inductive bias through learnable cyclic embeddings; and (ii) a lightweight Mult
    
[^390]: CausalDS：数据科学智能体中因果推理的基准测试

    CausalDS: Benchmarking Causal Reasoning in Data-Science Agents

    [https://arxiv.org/abs/2607.08093](https://arxiv.org/abs/2607.08093)

    该论文提出CausalDS基准，通过系统性地生成结构因果模型、对应的观测数据以及基于真实领域的自然语言场景，填补了现有基准在评估数据科学智能体因果推理能力方面的空白。

    

    大型语言模型日益扮演数据科学智能体的角色，将抽象推理与先进的工具使用相结合。然而，相关基准测试领域大致分为两类：一类是缺乏真实数据分析的符号化因果推理基准，另一类是缺乏因果数据生成结构及一系列因果数据科学任务的数据分析基准。现有的因果评估数据集通常局限于来自现有来源的精选示例，其多样性源于有限的模板化变体，而非系统性地生成新颖的合成因果结构。我们提出了CausalDS，一个用于评估智能体化数据科学工作流中因果推理能力的基准。每个基准实例是一个场景，由一个采样得到的结构因果模型（SCM）及其生成的观测数据，以及一个基于真实领域的配套合成自然语言故事组成。（注：原文摘要在此处被截断）

    arXiv:2607.08093v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly act as data-science agents, combining abstract reasoning with advanced tool use. Yet the relevant benchmark landscape largely divides into symbolic causal reasoning benchmarks without realistic data analysis or data analysis benchmarks without causal data-generating structure or a range of causal data science tasks. Existing causal evaluation datasets are often restricted to curated examples from existing sources, with diversity coming from limited templatized variations rather than from systematic generation of novel synthetic causal structures. We introduce CausalDS, a benchmark for evaluating causal reasoning in agentic data-science workflows. Each benchmark instance is a scene consisting of a sampled structural causal model (SCM) with generated observational data and an accompanying synthetic natural-language story grounded in a real domain. We ground the composition of the benchmar
    
[^391]: 离散扩散模型学到了什么？三种坐标系下的同一个逆向过程

    What Does a Discrete Diffusion Model Learn? One Reverse Process in Three Coordinates

    [https://arxiv.org/abs/2607.05381](https://arxiv.org/abs/2607.05381)

    本文证明离散扩散模型中的去噪器、桥接插件和具体分数三种方法学习的其实是同一对象——加噪过程的精确时间反演，只是处于三种不同坐标系，并给出 Oracle Distance 恒等式（负 ELBO 等于数据熵加路径 KL 散度）及三种坐标间的精确转换。

    

    离散扩散模型究竟学到了什么？现有文献分别训练去噪器、桥接插件和具体分数，每种方法都在不同的约定下推导出自己的 ELBO 目标。这种碎片化使得所学习的对象不清晰，ELBO 数值也难以解释和比较。本文指出，这三种方法在不同的坐标系中学习的是同一个对象：加噪过程的精确时间反演。我们首先证明了 Oracle Distance 恒等式：对于任意加噪过程，负 ELBO 不仅仅是似然的一个界；它等于数据熵加上从反向加噪过程到所学过程的路径 KL 散度。接下来，针对令牌因子化的噪声，我们证明去噪器、桥接插件和具体分数是该逆向过程的三种坐标，推导出它们之间的精确转换关系，并确定了每种方法所学习的概率分布。特别是，与其标准解释相反……（摘要原文在此处被截断）

    arXiv:2607.05381v2 Announce Type: replace-cross  Abstract: What does a discrete diffusion model actually learn? The literature trains denoisers, bridge plug-ins, and concrete scores, each deriving its own ELBO objective under different conventions. This fragmentation makes the object being learned unclear and ELBO values difficult to interpret and compare. All three methods learn the same object in different coordinates: the exact time-reversal of the noising process. We first prove the Oracle Distance identity: for any noising process, the negative ELBO is not just a bound on the likelihood; it equals the data entropy plus the path KL divergence from the reversed noising process to the learned one. Next, for token-factorizing noise, we show that denoiser, bridge plug-in, and concrete score are three coordinates of this reverse process, derive exact conversions between them, and identify the probability distribution learned by each. In particular, contrary to its standard interpretatio
    
[^392]: 柏拉图投影结构：表示学习中的算子诱导可观测性

    Platonic Projection Structures: Operator-Induced Observability in Representation Learning

    [https://arxiv.org/abs/2607.05175](https://arxiv.org/abs/2607.05175)

    该论文提出柏拉图投影结构（PPS）这一算子理论框架，通过自伴半正定观测算子与商几何 $H/\ker(\Pi)$ 来刻画部分观测下表示学习的可观测性，并揭示量子测量与线性观测下的表示推断共享同一算子理论结构。

    

    我们通过柏拉图投影结构（PPS）来刻画表示学习中的可观测性，这是一种用于分析部分观测下表示可访问性的算子理论框架。PPS并不将可观测输出视为潜在表示的直接反映，而是通过一个作用于潜在表示空间的自伴半正定算子来建模观测。一个系统被表示为三元组 $(H, \Pi, O)$，其中 $H$ 是潜在表示空间，$\Pi \succeq 0$ 是观测算子，$O(v)=\langle v,\Pi v\rangle$ 定义了诱导出的标量可观测量。可观测性由商几何 $H/\ker(\Pi)$ 刻画，其代表在观测下不可区分的潜在状态等价类。我们证明，量子测量与线性观测模型下的表示推断共享这一算子理论结构，但在代数（原文在此处截断）……

    arXiv:2607.05175v2 Announce Type: replace  Abstract: We characterize observability in representation learning through Platonic Projection Structures (PPS), an operator-theoretic framework for analyzing representation accessibility under partial observation. Rather than treating observable outputs as direct reflections of latent representations, PPS models observation through a self-adjoint positive semidefinite operator acting on a latent representation space. A system is represented as a triple $(H, \Pi, O)$, where $H$ is a latent representation space, $\Pi \succeq 0$ is an observation operator, and $O(v)=\langle v,\Pi v\rangle$ defines an induced scalar observable. Observability is characterized by the quotient geometry $H/\ker(\Pi)$, representing equivalence classes of latent states indistinguishable under observation. We show that quantum measurement and representation inference under linear observation models share this operator-theoretic structure while differing in the algebraic
    
[^393]: LIME：从自我中心视频学习意图感知的相机运动

    LIME: Learning Intent-aware Camera Motion from Egocentric Video

    [https://arxiv.org/abs/2607.02417](https://arxiv.org/abs/2607.02417)

    该论文首次将视觉-语言条件下的相机运动生成形式化为独立任务，并提出通过事后标注从自我中心视频中挖掘多意图监督数据的方法，使机器人能够根据自然语言意图预测相机运动。

    

    自主机器人通常需要先移动相机才能执行动作：例如检查物体、揭示被遮挡的区域，或获得能够响应用户意图的视角。虽然视觉-语言导航将指令转化为底盘运动，视觉-语言-动作策略将指令映射到操作动作，但视觉-语言条件下的相机运动作为一类独立的动作仍相对缺乏研究。我们提出了视觉-语言条件下的相机运动生成任务：给定当前的RGB观测和自由形式的自然语言意图，预测下一次观测的相对目标相机位姿。其挑战在于，有用的视角变化依赖于潜在的感知意图，涵盖从粗粒度的空间移动到细粒度的物体检查或遮挡揭示。为了建模这种结构，我们通过事后标注从自我中心视频中挖掘多意图的相机运动监督数据，将每个相对SE(3)……（原文摘要在此处被截断）

    arXiv:2607.02417v2 Announce Type: replace-cross  Abstract: Autonomous robots often need to move their camera before they can act: to inspect an object, reveal an occluded region, or obtain a view that responds to a user's intent. While vision-language navigation translates instructions to base motion and vision-language-action policies map instructions to manipulation actions, vision-language conditioned camera motion remains comparatively underexplored as a first-class action. We formulate vision-language conditioned camera motion generation: given a current RGB observation and a free-form natural-language intent, predict a relative target camera pose for the next observation. The challenge is that useful viewpoint changes depend on latent perceptual intent, ranging from coarse spatial moves to fine-grained inspection or occlusion revealing. To model this structure, we mine multi-intent camera-motion supervision from egocentric video via hindsight labeling, pairing each relative SE(3)
    
[^394]: 自组织保形预测：通过无监督群体发现减少区域覆盖差距

    Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery

    [https://arxiv.org/abs/2606.29403](https://arxiv.org/abs/2606.29403)

    提出自组织保形预测（SOCP），利用无需校准标签的无监督自组织映射（SOM）发现输入空间分组并从中提取校准缓冲，在预测器和非一致性分数保持不变的前提下实现精确的条件覆盖有效性，从而减少特征空间异质区域的覆盖差距。

    

    保形预测能够保证边际覆盖率，但汇总的校准分位数可能会掩盖特征空间中异质区域的系统性覆盖不足问题。我们提出自组织保形预测（SOCP），这是一种校准方案，利用无需校准标签训练的无监督自组织映射（SOM）来发现输入空间的分组。在预测时，查询的最佳匹配单元（BMU）从单个单元、固定网格邻域或基于原型的扩展中提取校准缓冲。当固定邻域过于稀疏时，模式3（Regime 3）根据原型距离添加单元，其全局预算在观察到任何校准分数之前，就从训练单元的占用情况和计划校准规模中选定。预测器和非一致性分数保持不变。仅单元检索具有精确的单元条件有效性，且每个固定的单元并集都具有精确的检索集有效性。（摘要在此处截断）

    arXiv:2606.29403v2 Announce Type: replace-cross  Abstract: Conformal prediction guarantees marginal coverage, but a pooled calibration quantile can hide systematic undercoverage across heterogeneous regions of the feature space. We introduce Self-Organized Conformal Prediction (SOCP), a calibration scheme that discovers input-space groups with an unsupervised Self-Organizing Map (SOM) trained without calibration labels. At prediction time, the query's best-matching unit (BMU) draws a calibration buffer from one cell, a fixed grid neighborhood, or a prototype-based enlargement. When fixed neighborhoods are too sparse, Regime 3 adds cells by prototype distance, using a global budget selected from training-cell occupancies and the planned calibration size before any calibration score is observed. The predictor and nonconformity score remain unchanged. Cell-only retrieval has exact cell-conditional validity, and each fixed union of cells has exact retrieved-set validity. Interpreting a nei
    
[^395]: Blackknife：针对异构图神经网络的免查询黑盒重连攻击

    Blackknife: Query-Free Black-Box Rewiring Attacks on Heterogeneous Graph Neural Networks

    [https://arxiv.org/abs/2606.29240](https://arxiv.org/abs/2606.29240)

    Blackknife提出了一种免查询的严格黑盒攻击方法，仅通过观测一跳结构和保持度数不变的边重连操作，即可有效规避异构图神经网络的防御。

    

    异构图神经网络（HGNNs）在包含多种节点类型和关系类型的图上表现优异，但其对结构访问受限攻击的鲁棒性仍缺乏充分研究。我们提出了Blackknife，一种免查询的严格黑盒规避攻击，它仅观测被攻击节点的一跳结构、其真实标签以及允许动作空间内的原始特征。该方法不使用任何受害者模型的输出来训练代理模型、生成扰动或选择攻击。Blackknife仅使用可观测的边和关系类型来训练一个共享代理模型，然后通过连续松弛和投影梯度上升来优化结构扰动。每次重连操作会删除一条边并添加另一条相同关系类型的边，从而保持目标节点的总度数以及每种关系类型下的度数不变。余弦相似度惩罚和自适应结构分类器进一步约束……

    arXiv:2606.29240v3 Announce Type: replace  Abstract: Heterogeneous graph neural networks (HGNNs) perform well on graphs with multiple node and relation types, but their robustness to attacks with restricted structural access remains insufficiently understood. We propose Blackknife, a query-free, strict black-box evasion attack that observes only the one-hop structures of the attacked nodes, their ground-truth labels, and raw features within the permitted action space. No victim-model outputs are used to train the surrogate, generate perturbations, or select attacks.Blackknife trains a shared surrogate using only observable edges and relation types, then optimizes structural perturbations through continuous relaxation and projected gradient ascent. Each rewiring operation deletes one edge and adds another of the same relation type, preserving the target node's total degree and degree for each relation type. A cosine-similarity penalty and an adaptive structural classifier further regula
    
[^396]: 从点击到意图：基于大语言模型提炼分类法的跨平台会话嵌入在金融服务推荐中的应用

    From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations

    [https://arxiv.org/abs/2606.26277](https://arxiv.org/abs/2606.26277)

    本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。

    

    顺序用户行为建模在工业推荐系统中被广泛采用；然而，在金融服务领域仍存在显著差距，因为登录前的网页交互与认证后的应用内体验截然不同。具体而言，登录前的网页用户通常探索新产品，而登录后的应用用户则专注于账户服务。由于跨渠道实体解析的挑战（例如，将匿名网页会话匹配到已认证的移动账户），基于网页的意图信号在认证后个性化中仍未得到充分利用。现有捕获网页意图的方法通常是临时且狭窄的，缺乏灵活性来同时支持定量下游推荐和定性的大规模理解。在这项工作中，我们提出了一个可扩展且双用途的意图预测框架，用于基于网页的交互，并展示了其在个性化中的适用性。

    arXiv:2606.26277v1 Announce Type: cross  Abstract: Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach tran
    
[^397]: 病态反问题中的地质文本描述：来自学习型渗透系数反演的见解

    Geological text descriptions in ill-posed inverse problems: insights from learned hydraulic-conductivity inversion

    [https://arxiv.org/abs/2606.24967](https://arxiv.org/abs/2606.24967)

    该研究通过合成达西流基准系统考察了地质文本描述在学习型渗透系数反演中的作用，发现当观测存在结构歧义时描述能改善重建，实例特定内容尤为有效，且求解器会依据描述所述的几何信息偏移重建结果，但随着观测信息量增大，描述的影响会减弱。

    

    渗透系数反演是一个不适定（病态）问题：即使是完整的水头观测也可能留下结构性歧义。地质文本描述可以提供关于地下结构的额外信息来约束重建。然而，描述何时能改善重建、以及求解器如何使用其所述内容，目前仍不清楚。我们在学习型反演中，通过一个带有理想化描述的合成达西流基准来研究这些问题，并改变观测密度和流动方向。经过稀疏观测训练后，当观测留下结构性歧义时，与未使用描述训练的模型相比，描述可以改善重建效果，尽管增益在不同条件和不同运行之间有所差异。实例特定的内容可以在场类型信息之外进一步改善重建。编辑这些内容会使重建结果向所选特征的所述几何形态偏移，但随着观测对这些特征的信息量增加，这种偏移会减弱。

    arXiv:2606.24967v2 Announce Type: replace  Abstract: Hydraulic-conductivity inversion is ill posed: even complete head observations can leave structural ambiguity. Geological text descriptions can supply additional information about subsurface structure to constrain reconstruction. However, it remains unclear when descriptions improve reconstruction and how solvers use their stated content. We examine these questions in learned inversion using a synthetic Darcy-flow benchmark with idealised descriptions, varying observation density and flow direction. After sparse-observation training, descriptions can improve reconstruction over models trained without them when observations leave structural ambiguity, though gains vary across conditions and runs. Instance-specific content can improve reconstruction beyond field-type information. Editing this content shifts reconstructions toward the stated geometry of selected features, less strongly as observations become more informative about those
    
[^398]: X-Safe：基于动作掩码的与本体无关的机械臂操作策略安全保障方法

    X-Safe: Embodiment-Agnostic Safeguarding for Robot Arm Manipulation Policies via Action Masking

    [https://arxiv.org/abs/2606.22278](https://arxiv.org/abs/2606.22278)

    X-Safe提出了一种与机器人本体无关的安全保障方法，通过在构型空间中对当前构型进行局部推理建立安全区域，并利用动作掩码将策略动作限制在安全区域内，从而在无需模型简化或学习型安全函数的情况下，为机械臂操作策略提供形式化的防碰撞概率保证。

    

    确保基于学习的机器人操作在多样化机器人本体和任务上的安全性，仍然需要大量的手工工程。在避免碰撞方面，现有方法通常依赖于特定本体的学习型安全近似，或使用简化的机器人模型并结合基于停止的回退控制器，从而损害了系统性能。在本文中，我们提出了一种与本体无关的安全保障方法X-Safe，它能够为避免碰撞提供形式化的概率保证，而无需模型简化或学习型安全函数。通过在构型空间中运行，我们的安全保障方法可以与不同类型的机器人策略协同工作。X-Safe在当前构型附近进行局部推理以建立安全区域，并使用动作掩码将策略的动作支持范围限制在这些安全区域内。我们在两种机器人本体和三种策略上演示了X-Safe的效果，涵盖仿真和硬件实验。

    arXiv:2606.22278v2 Announce Type: replace-cross  Abstract: Ensuring safety of learning-enabled robotic manipulation across diverse embodiments and tasks still requires significant manual engineering. For collision avoidance, existing approaches typically rely on embodiment-specific learned safety approximations or simplified robot models together with stopping-based fallback controllers that compromise performance. In this paper, we propose an embodiment-agnostic safeguarding approach, X-Safe, which provides formal probabilistic guarantees for collision avoidance without model simplifications or learned safety functions. By operating in the configuration space, our safeguarding method works with different types of robot policies. X-Safe reasons locally around the current configuration to establish safe regions and uses action masking to restrict the policy's support to these safe regions. We demonstrate X-Safe on two embodiments and three policies, both in simulation and on hardware. W
    
[^399]: PoLAR：在机器人策略学习的潜在动作中分解幅度与模式

    PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning

    [https://arxiv.org/abs/2606.21139](https://arxiv.org/abs/2606.21139)

    PoLAR 通过在潜在动作空间中引入极坐标径向结构（半径编码动作幅度、方向编码动作模式），并借助双曲空间与观测时间偏移作为约束，使机器人策略更容易将潜在动作解码为实际的机器人动作。

    

    机器人的动作在幅度和模式两个维度上各不相同：即它们产生多大的变化，以及产生何种类型的变化。一个能够反映这种结构的潜在动作空间，可以使下游策略更容易将潜在动作映射为机器人动作。然而，现有的潜在动作方法通常依赖于从观测对进行的视觉重建，而没有显式地鼓励这种结构。我们提出了具有径向结构的极坐标潜在动作，它在潜在动作上施加径向-方向结构，鼓励半径编码过渡的幅度，方向保留过渡的模式。PoLAR 使用观测之间的时间偏移作为过渡幅度的弱代理，鼓励具有较大时间偏移的过渡占据更大的半径。我们在双曲空间中实例化这一结构，双曲空间随半径增长而扩大的体积，天然适合在更大幅度下容纳更多样化的过渡模式。在任务内……（原文摘要在此处截断）

    arXiv:2606.21139v2 Announce Type: replace-cross  Abstract: Robot actions vary in both extent and mode: how much change they produce and what kind of change they produce. A latent action space that reflects this structure may make it easier for downstream policies to map latent actions to robot actions. However, existing latent action methods typically rely on visual reconstruction from observation pairs without explicitly encouraging such structure. We introduce Polar Latent Actions with Radial structure (PoLAR), which imposes a radial-direction structure on latent actions, encouraging radius to encode transition extent and direction to retain transition mode. PoLAR uses temporal offsets between observations as a weak proxy for transition extent, encouraging transitions with larger offsets to occupy larger radii. We instantiate this structure in hyperbolic space, whose expanding volume with radius offers a natural fit for more diverse transition modes at larger extents. Across in-task 
    
[^400]: HiMPO：面向长时程智能体低纠缠信用分配的后见之明引导记忆策略优化

    HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents

    [https://arxiv.org/abs/2606.16285](https://arxiv.org/abs/2606.16285)

    HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。

    

    长时程智能体依赖记忆机制来压缩交互历史，但优化记忆写入面临一个独特的信用分配挑战：记忆更新可能因下游工具故障、噪声观测或推理错误而非其自身贡献而受到奖励或惩罚。我们提出了HiMPO，一个用于为长时程智能体中记忆写入动作分配低纠缠信用的后见之明引导记忆策略优化框架。HiMPO首先通过在相同的预写入状态下比较可从先前记忆和更新后记忆中恢复的任务相关信息，来估计记忆更新的局部效用。然后，它将后见相关性用作一个有界的回溯过滤器，当局部效用得不到目标结果支持时衰减记忆信用。由此产生的记忆特定优势仅应用于记忆token，而轨迹级奖励则优化智能体的其余部分。

    arXiv:2606.16285v2 Announce Type: replace  Abstract: Long-horizon agents rely on memory mechanisms to compress interaction history, but optimizing memory writing faces a distinct credit assignment challenge: a memory update may be rewarded or penalized due to downstream tool failures, noisy observations, or reasoning errors rather than its own contribution. We propose HiMPO, a Hindsight-Informed Memory Policy Optimization framework for assigning less-entangled credit to memory-writing actions in long-horizon agents. HiMPO first estimates the local utility of a memory update by comparing the task-relevant information recoverable from the previous and updated memories under the same pre-write state. It then uses hindsight relevance as a bounded retrospective filter that attenuates memory credit when local utility is not supported by the target outcome. The resulting memory-specific advantage is applied only to memory tokens, while trajectory-level rewards optimize the rest of the agent's
    
[^401]: 驾驶，该快还是该慢？面向多模态地面移动的运动预测神经符号引导方法

    Driving, Fast or Slow? Neuro-Symbolic Guidance for Motion Prediction in Multi-Modal Ground Mobility

    [https://arxiv.org/abs/2606.15251](https://arxiv.org/abs/2606.15251)

    提出了TraCS神经符号框架，通过代理式代码生成将自然语言交通法规转化为可解释的概率一阶逻辑约束，并利用响应式数据流推理引擎实时更新合规性图景，从而增强黑盒运动预测模型在异构交通环境中的可解释性与安全性。

    

    对包括行人、自行车、汽车和卡车在内的异构交通空间进行准确且可解释的运动预测，对于安全的自主导航至关重要。然而，最先进的方法仍然主要是黑盒式的，缺乏对现实世界移动中法规和行为约束的显式编码。我们提出了轨迹合规性塑造，这是一个神经符号框架，它利用可解释的概率一阶逻辑来增强现有的黑盒运动预测主干模型。为此，TraCS 采用代理式代码生成流程，以弥合交通法规的自然语言描述与概率运动预测之间的鸿沟。此外，TraCS 还采用了一个响应式数据流推理引擎，能够随着场景的演变维护并高效更新合规性图景。为防止 TraCS 过度自信地引导主干模型的预测……

    arXiv:2606.15251v2 Announce Type: replace-cross  Abstract: Accurate and interpretable motion prediction for heterogeneous traffic spaces, including pedestrians, bicycles, cars, and trucks, is essential for safe autonomous navigation. Nevertheless, state-of-the-art approaches remain predominantly black-box, lacking explicit encoding of the regulatory and behavioral constraints of real-world mobility. We propose Trajectory Compliance-Shaping (TraCS), a neuro-symbolic framework that augments existing black-box motion prediction backbones with interpretable and probabilistic first-order logic. To do so, TraCS employs an agentic code-generation pipeline to bridge the gap between natural-language descriptions of traffic regulations and probabilistic motion prediction. Furthermore, TraCS employs a reactive data-streaming inference engine that maintains and efficiently updates compliance landscapes as scenes evolve. To prevent TraCS from overconfidently steering the backbone's predictions in t
    
[^402]: 基于概念的表征相似性统一框架

    A Unifying Framework for Concept-Based Representational Similarity

    [https://arxiv.org/abs/2606.09653](https://arxiv.org/abs/2606.09653)

    提出了一个基于概念的表征相似性统一框架，沿“对齐对象”与“对齐层级”两个轴分解concept对齐问题，并引入基于干预的基准InterVenchA，分别测量提取质量、translation质量与concept一致性，且与人类感知进行对照验证。

    

    跨模型与跨模态学习到的表征往往呈现出惊人的结构相似性，这暗示它们共享潜在的concept分解。然而，concept对齐的定义仍然模糊不清：现有方法在相同术语下优化着不同的目标，掩盖了实际被对齐的内容。我们提出了一个统一框架，沿两个轴对对齐进行分解：对齐的对象（表征 vs. concept）以及对齐的层级（实例级 vs. 分布级）。由此引出四种相应的性质——translation和concept一致性的实例级与分布级变体——并精确揭示了现有方法提供了其中哪些保证。我们进一步介绍了InterVenchA，这是一个基于干预的基准，分别测量提取质量、translation质量和concept一致性——我们依据人类对concept对齐的感知对该基准进行了验证。利用该框架，我们……

    arXiv:2606.09653v2 Announce Type: replace  Abstract: Learned representations across models and modalities often exhibit striking structural similarities, suggesting shared underlying concept decompositions. However, concept alignment remains poorly defined: existing approaches optimize different objectives under the same terminology, obscuring what is actually aligned. We propose a unifying framework that decomposes alignment along two axes: what is aligned (representations vs. concepts) and at what level (instance-wise vs. distributional). This induces four corresponding properties---instance-wise and distributional variants of translation and concept consistency---and reveals precisely which of these guarantees existing methods provide. We further introduce InterVenchA, an intervention-based benchmark that separately measures extraction quality, translation quality, and concept consistency---which we validate against human perception of concept alignment. Using this framework, we sho
    
[^403]: AgentWorld：一个从多轮交互中训练大语言模型智能体的数据工厂

    AgentWorld: A Data Factory for Training LLM Agents from Multi-Turn Interactions

    [https://arxiv.org/abs/2606.09138](https://arxiv.org/abs/2606.09138)

    AgentWorld是一个数据工厂，通过API代理网关捕获异构智能体运行时的多轮交互，并将其转化为标准化、高质量的结构化训练数据，为LLM智能体训练提供从交互收集到训练消费的统一数据生命周期。

    

    训练大语言模型使其成为有能力的智能体，需要从与环境的**多轮交互**中学习。然而，由异构智能体运行时生成的多轮交互轨迹，难以转化为训练引擎能够直接使用的结构化、高质量训练数据。本文提出AgentWorld，一个通过网关服务器和数据池连接异构智能体运行时与训练引擎的数据工厂，围绕从交互收集到训练消费的统一数据生命周期重构LLM智能体训练。网关服务器通过LLM API代理捕获多轮交互，数据池则将其组织为标准化的步骤记录，以便进行数据整理与训练。交互式仪表板支持轨迹监控、步骤检查和数据审查。在演示中，用户可以集成开箱即用的白盒和黑盒智能体，对轨迹进行追踪与整理……

    arXiv:2606.09138v2 Announce Type: replace-cross  Abstract: Training LLMs to act as capable agents requires learning from multi-turn interactions with environments. However, multi-turn interaction trajectories generated by heterogeneous agent runtimes are difficult to transform into structured, high-quality training data that training engines can readily consume. In this paper, we propose AgentWorld, a data factory that bridges heterogeneous agent runtimes and training engines through a Gateway Server and a Data Pool, reframing LLM agent training around a unified data lifecycle from interaction collection to training consumption. The Gateway Server captures multi-turn interactions through an LLM API proxy, while the Data Pool organizes them into standardized step records for curation and training. An interactive dashboard supports trajectory monitoring, step inspection, and data review. In our demonstration, users integrate ready-to-use white-box and black-box agents, trace and curate t
    
[^404]: 激活引导诱发涌现性失调：引导向量构建与综合评估

    Activation Steering Induces Emergent Misalignment: Steering Vector Construction and Comprehensive Evaluation

    [https://arxiv.org/abs/2606.08682](https://arxiv.org/abs/2606.08682)

    该论文通过系统构建引导向量并综合评估，首次证实激活引导技术可以从狭窄任务出发，在大语言模型中诱发跨广泛无关领域的涌现性失调，揭示了这一流行推理时调节技术的潜在安全风险。

    

    激活引导（Activation Steering, AS）已成为一种流行的推理时技术，用于调节大语言模型（LLM）的行为。AS通过向中间激活注入任务特定的引导向量（Steering Vectors, SVs），以灵活且轻量的方式调节模型行为。与此同时，近期研究发现了“涌现性失调”（Emergent Misalignment, EM）这一新兴安全问题：在来自狭窄任务的不安全样本上进行微调，会意外地引发模型在无关领域中的广泛不安全行为。随着AS的日益普及，深入研究AS是否能诱发EM已成为一个紧迫但尚未被充分探索的课题。本文对激活引导诱发的涌现性失调进行了全面研究。具体而言，（1）我们确定了一种引导向量构建方法，证实源自狭窄任务的SV确实能在广泛的无关系领域中诱发失调。定性测量进一步揭示……（摘要在此处不完整）

    arXiv:2606.08682v2 Announce Type: replace  Abstract: Activation steering (AS) has emerged as a popular inference-time technique for modulating the behavior of large language models (LLMs). AS modulates model behavior in a flexible and lightweight manner by injecting task-specific steering vectors (SVs) into intermediate activations. Meanwhile, recent work has identified emergent misalignment (EM) as an emerging safety concern, whereby finetuning on unsafe examples from a narrow task unexpectedly induces broadly unsafe behavior in unrelated domains. With the increasing popularity of AS, it is pressing yet under-explored to thoroughly investigate whether AS can induce EM. In this paper, we conduct a comprehensive study of activation-steering-induced emergent misalignment. Specifically, (1) we identify a steering vector construction to confirm that SVs derived from a narrow task can indeed induce misalignment across broad unrelated domains. Qualitative measurement further reveals that AS-
    
[^405]: HARP：面向大语言模型微调的高效数据选择方法

    HARP: Efficient Data Selection for Finetuning Large Language Models

    [https://arxiv.org/abs/2606.07690](https://arxiv.org/abs/2606.07690)

    提出了一种基于训练的高效数据选择方法HARP，它将训练数据组织为层次结构、仅评估代表性子集并用经验贝叶斯后验推断其余数据的效用，配合互补的冗余控制与加性选择策略，在大幅降低选择成本的同时保持下游任务对齐效果。

    

    微调数据选择需要在两个相互冲突的目标之间取得平衡：既要选出能提升下游任务表现的样本，又要避免反复对模型进行微调。免训练选择器具有良好的可扩展性，但依赖嵌入相似度或聚类等代理指标，这些指标可能与目标objective不匹配；基于训练的选择器通过梯度信号、子集评估或Shapley归因能更好地反映下游效用，但需要大量代价高昂的训练-评估迭代。我们提出了分层主动区域剪枝方法（HARP），这是一种高效的基于训练的选择器，能够在降低选择成本的同时保持下游对齐效果。HARP将训练数据池组织为节点-叶节点层次结构，仅评估具有代表性的叶节点，并利用经验贝叶斯后验推断未测量的数据效用。随后，它使用两个互补的包络策略进行数据选择：HARP-C以保守方式控制数据冗余，HARP-E则以加性方式……（摘要原文在此处截断）

    arXiv:2606.07690v2 Announce Type: replace  Abstract: Finetuning data selection requires balancing two competing goals: selecting examples that improve the downstream objective, and doing so without repeatedly finetuning models. Train-free selectors are scalable but rely on proxies such as embedding similarity or clustering, which may not match the target objective. Train-based selectors better reflect downstream utility through gradient signals, subset evaluation, or Shapley attribution, but require many costly train-evaluate iterations. We propose Hierarchical Active Region Pruning (HARP), an efficient train-based selector that preserves downstream alignment while reducing selection cost. HARP organizes the training pool into a node-leaf hierarchy, evaluates only representative leaves, and infers unmeasured utilities with empirical Bayes posteriors. It then selects data using two complementary envelopes: HARP-C, which conservatively controls redundancy, and HARP-E, which additively re
    
[^406]: STREAM：面向数字病理组织学图像生成的随机黎曼流匹配与各向异性解码器

    STREAM: Stochastic Riemannian Flow Matching with Anisotropic Decoder for Digital Histopathology Image Generation

    [https://arxiv.org/abs/2606.07036](https://arxiv.org/abs/2606.07036)

    提出STREAM框架，首次在数字病理组织学图像生成中应用黎曼流匹配，直接以病理学视觉基础模型patch token所在的单位超球面作为潜在空间并配合各向异性解码器，克服了现有方法中条件信号主导多样性、导致生成多样性不足的问题。

    

    合成病理组织学图像生成旨在解决患者隐私问题以及基础模型日益增长的数据需求。现有最先进的病理组织学生成模型使用预训练的视觉基础模型（VFMs）作为条件信号。我们证明这会导致“条件主导的多样性”：在TCGA-BRCA数据集上，其输出多样性的62-75%可归因于条件信号而非所学习的潜在空间，并且从头合成在推理时仍需要VFM。我们转而将病理组织学VFM本身用作潜在空间：其patch token经ℓ2归一化后位于单位超球面S^(d-1)上，具有强角度优势和内在曲率，这启发了黎曼公式化。我们提出STREAM，这是首个在病理组织学领域应用黎曼流匹配的框架，分为两个阶段：1）桥式随机扰动为每个token建立可整流性……（注：所提供的摘要原文在此处截断，以上为可获取部分的翻译）

    arXiv:2606.07036v3 Announce Type: replace-cross  Abstract: Synthetic histopathology image generation addresses patient-privacy concerns and the growing data demands of foundation models. Existing state-of-the-art histopathology generative models use pretrained Vision Foundation Models (VFMs) as conditioning signals. We show this yields conditioning-dominated diversity: on TCGA-BRCA, 62-75% of their output diversity is attributable to the conditioning signal rather than the learned latent space, while de novo synthesis still requires a VFM at inference. We instead use histopathology VFMs as the latent space itself: their patch tokens are $\ell_2$-normalized on the unit hypersphere $\mathcal{S}^{d-1}$ with strong angular dominance and intrinsic curvature, motivating a Riemannian formulation. We present STREAM, the first framework to apply Riemannian flow matching in the histopathology domain, in two stages: 1) a bridge-type stochastic perturbation that establishes per-token rectifiabilit
    
[^407]: 记忆设计：概率序列层

    Memory by Design: Probabilistic Sequence Layers

    [https://arxiv.org/abs/2605.31163](https://arxiv.org/abs/2605.31163)

    本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。

    

    arXiv:2605.31163v3 公告类型：交叉替换 摘要：我们引入了“设计模型框架”：一种从关于记忆的明确假设中推导高效循环序列映射的方法。设计模型通过精确贝叶斯滤波将证据写入记忆；查询相关的读取输出产生预测分布，其均值作为层输出。在我们的线性-高斯实例化中，“贝叶斯层”同时传播均值和协方差：协方差跟踪存储关联中的不确定性，将写入导向不确定方向，随着证据积累而衰减增益，并保留自信记忆。同一框架统一了多种次二次递归：线性注意力、GLA和Mamba-2/SSD在潜在输入设计模型下是精确滤波器，而DeltaNet及相关Delta规则模型是贝叶斯层设计模型的协方差重置简化。恢复协方差传播为检索提供闭式预测。

    arXiv:2605.31163v3 Announce Type: replace-cross  Abstract: We introduce the \emph{design-model framework}: a way to derive efficient recurrent sequence maps from explicit assumptions about memory. A design model writes evidence into memory by exact Bayesian filtering; a query- dependent readout produces a predictive distribution whose mean is the layer output. In our linear-Gaussian instantiation, the \emph{Bayesian Layer} propagates both a mean and a covariance: the covariance tracks uncertainty over stored associations, steering writes toward uncertain directions, attenuating gains as evidence accumulates, and preserving confident memories. The same framework unifies several sub-quadratic recurrences: linear attention, GLA, and Mamba-2/SSD are exact filters under a latent-input design model, whereas DeltaNet and related Delta-rule models are covariance-reset reductions of the Bayesian Layer's design model. Restoring covariance propagation yields closed-form predictions for retrieval 
    
[^408]: 子空间分解的JEPA：在潜在世界模型中解耦任务进展与内容

    Subspace-Decomposed JEPAs: Disentangling Progression and Content in Latent World Models

    [https://arxiv.org/abs/2605.31111](https://arxiv.org/abs/2605.31111)

    SD-JEPA将JEPA潜在空间分解为正交的进展子空间和内容子空间，使两种防坍缩力量在不相交的坐标上加性组合而非相互竞争，从而在多个控制基准上提升了世界模型性能。

    

    联合嵌入预测架构（JEPAs）通过预测未来嵌入来学习紧凑的潜在世界模型，但其潜在空间中没有任何单一坐标被专门用于编码任务进展。我们将JEPA潜在空间划分为两个角色互不相交的正交子空间：一个由余弦间隔三元组损失塑造的低维进展子空间，以及一个由LeWM现有SIGReg目标正则化的高维内容子空间。我们证明了这两种防坍缩力量作用于互不相交的坐标上，因此它们以加法方式组合，而非在同一维度上相互竞争。我们的方法SD-JEPA在同等计算量下，在LeWM基线的大多数控制基准测试中优于LeWM基线，并在Push-T任务上超越了最强的非LeWM JEPA基线；子空间消融证伪实验证实了这种子空间划分是起支撑作用的关键要素。除规划任务之外，所得到的一维角度进展坐标还可作为场景感知的……

    arXiv:2605.31111v2 Announce Type: replace  Abstract: Joint-Embedding Predictive Architectures (JEPAs) learn compact latent world models by predicting future embeddings, but no single coordinate of the latent is designated to encode task progression. We carve the JEPA latent into two orthogonal subspaces with disjoint roles: a low-dimensional progression subspace shaped by a cosine-margin triplet loss, and a high-dimensional content subspace regularised by the existing SIGReg objective of LeWM. We prove that the two anti-collapse forces act on disjoint coordinates, so they compose additively rather than competing on the same dimensions. Our method, SD-JEPA improves over the LeWM baseline on the majority of its control benchmarks at matched compute, and outperforms the strongest non-LeWM JEPA baseline on Push-T; a subspace-ablation falsifier confirms the split is the load-bearing ingredient. Beyond planning, the resulting 1-D angular progression coordinate functions as a scene-aware comp
    
[^409]: 鲁棒监督学习的统一与优化

    Unification and Optimization of Robust Supervised Learning

    [https://arxiv.org/abs/2605.28165](https://arxiv.org/abs/2605.28165)

    该论文提出了一个统一框架，将分布鲁棒优化、Mixup、标签平滑等鲁棒学习方法纳入同一表述体系，明确了各方法间的冗余或互补关系，并推导出将设计选择转化为可调超参数的实用训练过程。

    

    鲁棒学习方法，如分布鲁棒优化、邻域风险最小化、Mixup和标签平滑，分别应对不同的不确定性来源，包括分布偏移、有限样本效应和标签不确定性。然而在实践中，多种不确定性来源可能同时出现，目前尚不清楚应选用哪种鲁棒方法或方法组合。我们提出了一个统一框架，将一大类鲁棒性方法沿着明确的设计维度进行组织，并在共享的表述中清晰地展现它们之间的关系与组合方式。在该框架内，我们分析了鲁棒方法之间的相互作用：某些方法在特定建模假设下是冗余的，而另一些方法则产生真正不同的解族，且这些关系会随模型类或损失函数的变化而改变。我们进一步推导出一种可操作的训练过程，将设计选择及其优势转化为可调节的超参数。

    arXiv:2605.28165v2 Announce Type: replace  Abstract: Robust learning methods such as distributionally robust optimization, vicinal risk minimization, Mixup, and label smoothing address different sources of uncertainty, including distribution shift, finite-sample effects, and label uncertainty. In practice, however, several such sources may occur simultaneously, and it remains unclear which robust method or combination of methods to use. We introduce a unified framework that organizes a broad class of robustness methods along explicit design axes and makes their relations and combinations explicit within a shared formulation. Within this framework, we analyze how robustness methods interact: some are redundant under particular modeling assumptions, whereas others yield genuinely distinct solution families, and these relationships can change with the model class or loss. We further derive a tractable training procedure that exposes the design choices and their strengths as tunable hyperp
    
[^410]: 一种用于改进基于逐出式KV缓存压缩的简单插件

    A Simple Plug-in for Improving Eviction-Based KV Cache Compression

    [https://arxiv.org/abs/2605.23258](https://arxiv.org/abs/2605.23258)

    VECTOR是一种即插即用插件，通过引入“保留-近似-逐出”三路token路由机制，并结合离线校准的可重构性信号，恢复了原本会被二元逐出不可逆丢失的值信息，从而显著提升基于逐出的KV缓存压缩方法在长上下文推理中的性能。

    

    KV缓存的快速增长是大语言模型长上下文推理的主要瓶颈。现有方法通常受限于二元逐出或表示近似，这可能未能充分利用那些对精确保留并非关键、但仍可重构的token。我们提出了VECTOR，这是一种面向逐出式流水线的即插即用增强方法，引入了三路token路由机制：保留、近似和逐出。VECTOR将来自基础评分器的重要性信号与来自离线校准的基于回归的值估计的可重构性信号相结合。通过利用可重构性，VECTOR恢复了在二元逐出下会被不可逆转丢失的有用值信息，同时保留关键向量以维持注意力路由的稳定性。在LongBench、上下文长度高达128K token的RULER以及多轮对话SCBench上，VECTOR在高（压缩率）下相比其基础逐出方法取得了显著提升。

    arXiv:2605.23258v2 Announce Type: replace  Abstract: KV cache growth is a major bottleneck for long-context inference in large language models. Existing methods are often dominated by binary eviction or representation approximation, which may underutilize tokens that are not critical for exact retention but are still reconstructable. We present VECTOR, a plug-and-play augmentation for eviction-based pipelines that introduces three-way token routing: retention, approximation, and eviction. VECTOR combines an importance signal from the base scorer with a reconstructability signal from an offline-calibrated regression-based value estimation. By leveraging reconstructability, VECTOR recovers useful value information that would otherwise be irreversibly lost under binary eviction, while preserving key vectors for attention routing stability. On LongBench, RULER with contexts up to 128K tokens, and multi-turn SCBench, VECTOR yields significant gains over its base eviction methods under high 
    
[^411]: 一种受有限元启发的二部图学习模拟器，用于大变形板料成形的可制造性评估

    A finite-element-inspired bipartite graph learned simulator for manufacturability assessment in large-deformation sheet forming

    [https://arxiv.org/abs/2605.22845](https://arxiv.org/abs/2605.22845)

    该论文提出CAtt-BiGNN，一种基于交叉注意力的二部图神经网络，通过模拟显式有限元中节点运动学与单元变形的耦合更新机制，实现大变形板料成形可制造性的高效评估。

    

    显式动力学有限元（FE）模拟被广泛应用于大变形分析，但对于设计空间探索而言成本仍然高昂。在显式有限元分析中，节点运动学和单元级变形量通过耦合的节点-单元更新过程不断演化。受有限元启发的学习模拟器可以保留这一结构，并在各自的原生域上学习节点运动学和单元变形。节点位移描述几何演化，而单元减薄率识别局部变形和关键区域。在滚动预测过程中将两种输出同时反馈，使每个场能够为另一个场的后续更新提供信息。本工作提出了CAtt-BiGNN，一种基于交叉注意力的二部图神经网络，用于节点-单元耦合学习。该二部图编码了有限元引导的状态组织和信息流，而无需施加平衡或本构约束。网格节点和单元被表示……

    arXiv:2605.22845v3 Announce Type: replace-cross  Abstract: Explicit dynamic finite element (FE) simulations are widely used for large-deformation analysis but remain expensive for design-space exploration. In explicit FE analysis, nodal kinematics and element-level deformation measures evolve through a coupled node-element update process. An FE-inspired learned simulator can preserve this structure and learn nodal kinematics and elemental deformation on their native domains. Nodal displacement describes geometry evolution, while elemental thinning identifies local deformation and critical regions. Feeding both outputs back during rollout allows each field to inform subsequent updates of the other. This work proposes CAtt-BiGNN, a cross-attention-based bipartite graph neural network for coupled nodal-elemental learning. The bipartite graph encodes FE-guided state organisation and information flow without imposing equilibrium or constitutive constraints. Mesh nodes and elements are repre
    
[^412]: OScaR：面向大语言模型及更广泛领域的极端KV缓存量化的“奥卡姆剃刀”

    OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond

    [https://arxiv.org/abs/2605.19660](https://arxiv.org/abs/2605.19660)

    该论文识别出Token范数不平衡（TNI）是极端压缩下KV缓存逐通道量化保真度的主要瓶颈，并提出了一种精确且轻量的量化方法OScaR，无需复杂的量化流水线即可实现高保真KV缓存压缩。

    

    向长上下文推理和多模态智能的快速发展，使得键值（KV）缓存的内存占用成为高效部署中的主要内存瓶颈。虽然已有的逐通道量化方法能够有效处理键张量中固有的通道级离群值，但其在极端压缩下的效力会减弱。在本工作中，我们从实证和理论两个视角重新审视了逐通道量化范式的固有局限性。我们的分析将Token范数不平衡（TNI）识别为量化保真度的主要瓶颈。我们证明，当共享量化参数需要跨越具有显著范数差异的token组时，TNI会系统性地放大误差。我们没有依赖复杂的量化流水线（如TurboQuant），而是提出了OScaR（Omni-Scaled Canalized Rotation，全尺度引导旋转），一种精确且轻量的KV缓存协同量化方法。

    arXiv:2605.19660v3 Announce Type: replace-cross  Abstract: The rapid advancement toward long-context reasoning and multi-modal intelligence has made the memory footprint of the Key-Value (KV) cache a dominant memory bottleneck for efficient deployment. While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression. In this work, we revisit the inherent limitations of the per-channel quantization paradigm from both empirical and theoretical perspectives. Our analysis identifies Token Norm Imbalance (TNI) as the primary bottleneck to quantization fidelity. We demonstrate that TNI systematically amplifies errors when shared quantization parameters are required to span token groups exhibiting substantial norm disparities. Instead of relying on intricate quantization pipelines (e.g., TurboQuant), we propose OScaR (Omni-Scaled Canalized Rotation), an accurate and lightweight KV cache co
    
[^413]: MOCHA：面向智能体技能优化的多目标切比雪夫退火方法

    MOCHA: Multi-Objective Chebyshev Annealing for Agent Skill Optimization

    [https://arxiv.org/abs/2605.19330](https://arxiv.org/abs/2605.19330)

    提出 MOCHA 方法，通过切比雪夫标量化选择改进对象并以扩展帕累托前沿为接受准则，从探索退火到利用，实现智能体技能在准确性与上下文经济性之间的多目标优化，克服了现有单分数提示词优化器易停滞的问题。

    

    LLM 智能体通过“技能”来组织其行为——技能是结构化的自然语言规范，规定了智能体如何进行推理、检索和响应。与单体式提示词不同，技能是受平台限制约束的多字段产物：描述字段被索引用于路由，指令主体被加载到共享的上下文窗口中，每多一行都会消耗智能体本可用于其他地方的上下文资源。因此，技能优化本质上是多目标问题：技能应当同时具备准确性和经济性。现有的提示词优化器将这些目标压缩为单一分数，我们发现这会导致它们停滞不前：在其大多数运行中，1000 次 rollout 相较种子技能毫无进展。我们提出了 MOCHA（Multi-Objective CHebyshev Annealing，多目标切比雪夫退火），该方法通过切比雪夫标量化来选择需要改进的技能，并接受能够扩展帕累托前沿的改进，从而从探索阶段逐步退火至利用阶段。切比雪夫标量化……

    arXiv:2605.19330v2 Announce Type: replace-cross  Abstract: LLM agents organize behavior through skills -- structured natural-language specifications that govern how an agent reasons, retrieves, and responds. Unlike monolithic prompts, skills are multi-field artifacts with platform limits: descriptions are indexed for routing, instruction bodies are loaded into a shared context window, and every extra line costs the agent context it could spend elsewhere. Skill optimization is therefore inherently multi-objective: a skill should be accurate and economical at once. Existing prompt optimizers collapse these goals into a single score, and we find that this makes them stall: in most of their runs, 1000 rollouts yield zero progress over the seed skill. We introduce MOCHA (Multi-Objective CHebyshev Annealing), which selects skills to improve by Chebyshev scalarization and accepts improvements that expand the Pareto front, annealing from exploration to exploitation. Chebyshev scalarization sco
    
[^414]: BrainDyn：一种用于预测与生成脑动力学的层束神经常微分方程（Sheaf Neural ODE）模型

    BrainDyn: A Sheaf Neural ODE for Forecasting and Generating Brain Dynamics

    [https://arxiv.org/abs/2605.19324](https://arxiv.org/abs/2605.19324)

    BrainDyn 提出了一种基于胞腔层束的图神经 ODE 网络，通过根据预测上下文窗口动态重算的限制映射来建模脑区之间异质的通信方式，从而实现对大脑动力学的预测与生成。

    

    大脑活动随时间推移在多个脑区之间以协调的模式展开，对这些动力学进行建模是计算神经科学的核心目标之一。由于大脑活动分布在相互连接的脑区之上，它自然可以用图来表示：图的节点为脑区或记录电极，边刻画这些节点之间的关系，并通过消息传递来建模脑区之间的通信。然而，标准消息传递网络将所有节点置于同一个共享特征空间中，并且通常通过标量求和来聚合来自邻居的消息，这与关于大脑通信的神经生物学证据并不相符。为了建模大脑的异质性特征，我们提出了 BrainDyn——一种基于胞腔层束（cellular sheaf）构建的图常微分方程（ODE）网络。该网络带有动态限制映射，这些映射会根据预测的上下文窗口重新计算，而非一次性固定，用以介导脑区之间的通信。

    arXiv:2605.19324v2 Announce Type: replace  Abstract: Brain activity unfolds as coordinated patterns across regions over time, and modeling these dynamics is a central goal in computational neuroscience. Because activity is distributed over connected brain regions, it is naturally represented on a graph whose nodes are brain regions or recording electrodes and whose edges capture relationships between these nodes, with message-passing modeling regional communication. However, standard message-passing networks place every node in a single shared feature space and typically aggregate messages from neighbors by scalar summation, which does not align with neurobiological evidence of brain communication. In order to model the heterogeneous nature of the brain, we introduce BrainDyn, a graph ordinary differential equation (ODE) network built on a cellular sheaf with dynamic restriction maps, recomputed from the forecasting context window rather than fixed once, mediating region-to-region comm
    
[^415]: 我们距离真正的自动化研究还有多远？

    How Far Are We From True Auto-Research?

    [https://arxiv.org/abs/2605.19156](https://arxiv.org/abs/2605.19156)

    该研究提出ResearchArena评测框架，让现成智能体在轻量指导下自主完成从构思到写作的完整科研闭环，并通过仅稿件审稿、感知产物同行评审和人类元评审三种互补视角，系统评估了117篇智能体生成论文的真实质量，发现仅看稿件时结果乐观，凸显了更全面评估自动化研究成果的必要性。

    

    近期的自动研究系统能够产出完整的论文，但可行性并不等同于质量，而该领域目前仍缺乏对智能体生成论文实际质量的系统性研究。我们提出了ResearchArena，一个极简的评测框架，让现成的智能体（使用Opus 4.6的Claude Code、使用GPT-5.4的Codex以及使用K2.5的Kimi Code）在仅有轻量级指导的情况下自主完成完整的研究闭环（创意构思、实验开展、论文写作、自我完善）。在13个计算机科学种子课题上、每个智能体-领域组合进行3次试验，ResearchArena共产生了117篇智能体生成的论文。每篇论文均通过三种互补的视角进行评估：仅基于稿件的审稿（SAR）、结合工作区产物检查的感知产物式同行评审（PR），以及由人类开展的元评审。仅从SAR来看，结果是乐观的：Claude Code获得最高分，超越了Analemma的FARS……

    arXiv:2605.19156v2 Announce Type: replace-cross  Abstract: Recent auto-research systems can produce complete papers, but feasibility is not the same as quality, and the field still lacks a systematic study of how good agent-generated papers actually are. We introduce ResearchArena, a minimal scaffold that lets off-the-shelf agents (Claude Code using Opus 4.6, Codex using GPT-5.4, and Kimi Code using K2.5) carry out the full research loop themselves (ideation, experimentation, paper writing, self-refinement) under only lightweight guidance. Across 13 computer science seeds and 3 trials per agent-domain pair, ResearchArena yields 117 agent-generated papers, each evaluated under three complementary lenses: a manuscript-only reviewer (SAR), an artifact-aware peer review (PR) in which agents inspect the workspace alongside the manuscript, and an human conducted meta-review. Under SAR alone the picture is optimistic: Claude Code obtains the highest score, outperforms Analemma's FARS, and mat
    
[^416]: 优化器设计的对称兼容原则：嵌入层、语言模型输出头、SwiGLU多层感知机与混合专家路由器

    Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers

    [https://arxiv.org/abs/2605.18106](https://arxiv.org/abs/2605.18106)

    该论文提出优化器设计的对称兼容原则——梯度更新规则应在权重块的对称群下保持等变，据此统一了双正交等变更新方法，并为嵌入层、语言模型输出头、SwiGLU MLP和MoE路由器等结构推导出对称兼容的优化器。

    

    摘要：深度学习实践中长期存在一个显著的几何差异。尽管现代神经网络架构天然展现出丰富的对称性与等变性，但Adam及其变体等流行的优化器在本质上是逐坐标操作的，因而无法尊重参数空间的等变性结构。我们通过为优化器设计引入一个对称兼容原则来解决这一差异：梯度更新规则应当在作用于相应权重块的对称群下保持等变。遵循这一原则，我们首先为一般矩阵层的双正交等变更新提供了统一视角，这正是随机谱下降、Muon、Scion和极梯度方法所采用的方式。更重要的是，通过从正交群拓展到置换对称与共享平移对称，我们为具有相应对称性的参数块推导出了对称兼容的优化器……

    arXiv:2605.18106v5 Announce Type: replace-cross  Abstract: A striking geometric disparity has long persisted in the practice of deep learning. While modern neural network architectures naturally exhibit rich symmetry and equivariance properties, popular optimizers such as Adam and its variants operate inherently coordinate-wise, rendering them unable to respect the equivariance structures of the parameter space. We address this disparity by introducing a symmetry-compatible principle for optimizer design: the gradient update rule should be equivariant under the symmetry group acting on the corresponding weight block. Following this principle, we first provide a unified perspective on bi-orthogonally equivariant updates for general matrix layers, as employed by stochastic spectral descent, Muon, Scion, and polar gradient methods. More importantly, by moving from orthogonal groups to permutation and shared-shift symmetries, we derive symmetry-compatible optimizers for parameter blocks wh
    
[^417]: 对抗训练何时以及为何能改进物理信息神经网络（PINNs）：基于神经正切核的视角

    When and Why Adversarial Training Improves PINNs: A Neural Tangent Kernel Perspective

    [https://arxiv.org/abs/2605.15959](https://arxiv.org/abs/2605.15959)

    本文基于神经正切核（NTK）视角提出了一个新的分析框架，揭示了GAN判别器如何影响PINNs训练动力学，从而解释了对抗训练何时及为何有效，并据此提出了一种实用高效的PINNs训练算法。

    

    物理信息神经网络（PINNs）是求解微分方程的强大代理模型，但由于谱偏差、刚性问题以及在高频或多尺度解上精度不佳，其训练一向十分困难。基于生成对抗网络（GANs）的对抗训练近来在改善PINNs训练方面获得了令人惊讶的显著实证效果，但其背后的机制仍不明确。为此，我们提出了一个分析对抗训练PINNs的新框架，其核心观察在于GAN中的判别器如何影响PINNs的训练动力学。该框架首先为对抗训练在PINNs中何时以及为何有效提供了急需的理论基础，随后对各类GAN变体在此类训练中的作用进行了统一分析，最终导出了一种新的、实用且高效的PINNs训练算法。实证结果表明，我们的方法能够显著……

    arXiv:2605.15959v2 Announce Type: replace  Abstract: Physics-informed neural networks (PINNs) are powerful surrogates for differential equations but are notoriously difficult to train due to spectral bias, stiffness, and poor accuracy on high-frequency or multiscale solutions. Adversarial training based on generative adversarial networks (GANs) has recently gained surprisingly strong empirical results in improving training, but the underlying mechanisms remain elusive. To this end, we propose a new analysis framework for adversarially trained PINNs, based on the key observation of how the discriminator in GANs can influence the training dynamics of PINNs. The framework first provides a much needed theoretical grounding to why and when adversarial training is effective in PINNs, then presents a unified analysis of GANs variants in such training, and finally leads to a new, practical, efficient training algorithm for PINNs. Empirical results demonstrate that our method can significantly 
    
[^418]: Njord：用于海洋集合预报的概率图神经网络

    Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting

    [https://arxiv.org/abs/2605.15470](https://arxiv.org/abs/2605.15470)

    Njord是一个将深度潜变量框架与图神经网络相结合的概率性海洋预报模型，通过自适应不规则海面几何的K-means聚类网格实现全球0.25°和波罗的海2公里分辨率的预报，在单次前向传播中即可采样集合预报，性能超越确定性机器学习基线并提供不确定性估计。

    

    海洋动力学本质上具有混沌特性，然而现有的机器学习海洋模型只能产生确定性预报。我们提出了Njord，一个用于海洋预报的概率性数据驱动模型，适用于全球和区域两个尺度。Njord将深度潜变量框架与图神经网络架构相结合，能够在单次前向传播中对每个预报步骤进行采样。我们将Njord应用于全球0.25°分辨率以及波罗的海2公里分辨率的区域预报。为了扩展到这些大型海洋网格，我们引入了能够适应不规则海面几何形状的K-means聚类网格。实验表明，与确定性机器学习基线相比，该模型在两个尺度上均表现出强劲性能，同时还能从采样的集合预报中提供不确定性估计。在全球OceanBench基准测试中，Njord在与真实观测对比评估时，在上层海洋变量上取得了平均最低误差。

    arXiv:2605.15470v3 Announce Type: replace  Abstract: Ocean dynamics are inherently chaotic, yet existing machine learning ocean models produce only deterministic forecasts. We introduce Njord, a probabilistic data-driven model for ocean forecasting, applicable to both global and regional domains. Njord combines a deep latent variable framework with a graph neural network architecture, enabling sampling each forecast step in a single forward pass. We apply Njord globally at 0.25{\deg} resolution and regionally to the Baltic Sea at 2 km resolution. To scale to these large ocean grids we introduce K-means cluster meshes that adapt to irregular sea surface geometry. Experiments demonstrate strong performance on both domains compared to deterministic machine learning baselines, while also providing uncertainty estimates from the sampled ensemble forecasts. On the global OceanBench benchmark, Njord achieves the lowest errors on average across upper-ocean variables when evaluated against real
    
[^419]: GQLA：面向硬件自适应解码的分组查询潜在注意力

    GQLA: Group-Query Latent Attention for Hardware-Adaptive Decoding

    [https://arxiv.org/abs/2605.15250](https://arxiv.org/abs/2605.15250)

    该论文提出分组查询潜在注意力（GQLA），通过组共享投影使同一权重即可支持MQA吸收式与GQA两种解码路径，从而根据GPU计算带宽比和张量并行度自适应选择最优推理方式，并能零梯度转换现有MLA/GQA模型检查点。

    

    解码效率取决于KV缓存流量、计算量与并行度。多头潜在注意力（MLA）提供了吸收式和头扩展式两种执行形式，但其针对特定头的投影限制了分组KV共享。我们提出了分组查询潜在注意力（GQLA），其组共享的上投影保留了联合潜在表示，并支持可分片的扩展缓存。相同权重可借助现有内核实现等效的MQA吸收式与GQA两种执行方式。屋顶线分析将路径偏好与GPU的计算带宽比及张量并行（TP）程度关联起来：吸收式以流量换计算并在各并行副本间复制潜在表示；GQA则对KV组进行分片。在匹配token数量下，每组采用八个查询头的预训练在所测试的各模型规模上均取得了接近MLA的最终训练损失。TransGQLA可利用校准数据转换GQA和MLA检查点，无需梯度更新，转换后的DeepSeek-V3.1-Base在八项任务上达到了77.79%的（成绩，原文截断）。

    arXiv:2605.15250v4 Announce Type: replace  Abstract: Decoding efficiency depends on KV-cache traffic, computation, and parallelism. Multi-head Latent Attention (MLA) offers absorbed and head-expanded forms, but its head-specific projections limit grouped KV sharing. We introduce Group-Query Latent Attention (GQLA), whose group-shared up-projections preserve a joint latent and enable a shardable expanded cache. The same weights support equivalent MQA-absorb and GQA execution using existing kernels. Roofline analysis relates path preference to GPU compute-to-bandwidth ratio and tensor parallelism (TP): absorption trades traffic for computation and replicates the latent across ranks; GQA shards KV groups. Pretraining with eight query heads per group yields terminal training losses close to MLA's across the tested model scales at matched tokens. TransGQLA converts GQA and MLA checkpoints using calibration data. Without gradient updates, converted DeepSeek-V3.1-Base achieves 77.79% eight-ta
    
[^420]: 快与慢学习：迈向能持续适应的大语言模型

    Learning, Fast and Slow: Towards LLMs That Adapt Continually

    [https://arxiv.org/abs/2605.12484](https://arxiv.org/abs/2605.12484)

    受人类多时间尺度学习机制启发，提出了一种快-慢学习框架，以模型参数作为“慢”权重、优化上下文作为“快”权重，使大语言模型既能通过文本反馈快速适应特定任务，又能避免灾难性遗忘并保持通用推理能力。

    

    大语言模型（LLMs）通常通过更新参数（例如通过强化学习）来针对下游任务进行训练。然而，更新模型参数迫使其吸收任务特定的信息，这可能导致灾难性遗忘和可塑性丧失。相比之下，在固定LLM参数的情况下进行上下文学习可以低成本、快速地适应任务特定需求（例如提示词优化），但其本身通常无法达到通过更新LLM参数所能获得的性能提升。受人类学习在多个时间尺度上运作方式（例如系统1与系统2）的启发，我们提出了一种面向LLM的快-慢学习框架，将模型参数实例化为“慢”权重，将优化后的上下文实例化为“快”权重。这些快速“权重”可以从文本反馈中学习以吸收任务特定的信息，同时让慢权重保持更接近基础模型并持续维持通用推理行为。

    arXiv:2605.12484v3 Announce Type: replace  Abstract: Large language models (LLMs) are trained for downstream tasks by updating their parameters (e.g., via RL). However, updating model parameters forces them to absorb task-specific information, which can result in catastrophic forgetting and loss of plasticity. In contrast, in-context learning with fixed LLM parameters can cheaply and rapidly adapt to task-specific requirements (e.g., prompt optimization), but cannot by itself typically match the performance gains available through updating LLM parameters. Inspired by how human learning works at multiple time scales (e.g., System 1 vs 2), we propose a fast-slow learning framework for LLMs, instantiated with model parameters as "slow" weights and optimized context as "fast" weights. These fast "weights" can learn from textual feedback to absorb the task-specific information, while allowing slow weights to stay closer to the base model and persist general reasoning behaviors. Fast-Slow Tr
    
[^421]: 一种用于复合对数凹采样的近端梯度算法

    A proximal gradient algorithm for composite log-concave sampling

    [https://arxiv.org/abs/2605.12461](https://arxiv.org/abs/2605.12461)

    提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。

    

    我们提出了一种从 $\mathbb{R}^d$ 上的复合对数凹分布中进行采样的算法，即形如 $\pi\propto e^{-f-g}$ 的密度，假设我们能够获得 $\alpha_f$-凸函数 $f$ 的梯度评估，以及 $\alpha_g$-凸函数 $g$（其中 $\alpha_g \ge 0$）的受限高斯预言机（RGO）。后一要求意味着我们可以容易地从密度 $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$ 中采样，这相当于 $g$ 的近端算子在采样上的类比。如果 $f + g$ 是 $\alpha$-凸的且 $f$ 是 $\beta$-光滑的，我们的采样器在总变差距离下达到 $\varepsilon$ 误差，所需的预言机调用次数为 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$，其中 $\kappa := 1 \vee \beta/\alpha$，$0\le \alpha_g \le \beta$ 且 $\alpha := \alpha_f + \alpha_g > 0$，这与之前 $g=0$ 情形下最先进的结果相匹配。我们进一步将结果扩展到以下情形：(1) ...（摘要在此处被截断）

    arXiv:2605.12461v2 Announce Type: replace-cross  Abstract: We propose an algorithm to sample from composite log-concave distributions over $\mathbb{R}^d$, i.e., densities of the form $\pi\propto e^{-f-g}$, assuming access to gradient evaluations of $\alpha_f$-convex $f$ and a restricted Gaussian oracle (RGO) for $\alpha_g$-convex $g$, where $\alpha_g \ge 0$. The latter requirement means that we can easily sample from the density $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$, which is the sampling analogue of the proximal operator for $g$. If $f + g$ is $\alpha$-convex and $f$ is $\beta$-smooth, our sampler achieves $\varepsilon$ error in total variation distance using $\widetilde{\mathcal O}(\kappa \sqrt d \log^4(1/\varepsilon))$ oracle calls where $\kappa := 1 \vee \beta/\alpha$, with $0\le \alpha_g \le \beta$ and $\alpha := \alpha_f + \alpha_g > 0$, which matches prior state-of-the-art results for the case $g=0$. We further extend our results to cases where (1) $
    
[^422]: 分化差异，汇集其余：可证明高效的多目标模仿学习

    Split the Differences, Pool the Rest: Provably Efficient Multi-Objective Imitation

    [https://arxiv.org/abs/2605.12000](https://arxiv.org/abs/2605.12000)

    提出多输出增强行为克隆（MA-BC）算法，通过划分冲突专家数据并汇集无冲突的状态-动作对，以更快的统计速率收敛到帕累托最优策略，并达到极小极大最优。

    

    本工作研究多目标模仿学习问题：即在多目标马尔可夫决策过程（MOMDP）中，给定多个帕累托最优专家的示范数据，恢复位于帕累托前沿上的策略。标准模仿学习方法并不适用于这一情形，因为简单聚合相互冲突的专家轨迹可能导致产生被支配的策略。为解决这一问题，我们提出了多输出增强行为克隆（MA-BC）算法，该算法系统性地划分存在分歧的专家数据，同时汇集未观察到行为冲突的状态-动作对。在理论方面，我们证明了MA-BC收敛到帕累托最优策略的统计速率快于任何独立考虑每个专家数据集的学习器。此外，我们为多目标模仿学习建立了一个新的下界，证明MA-BC具有极小极大最优性。最后，我们通过实验验证了该算法的有效性。

    arXiv:2605.12000v3 Announce Type: replace  Abstract: This work investigates multi-objective imitation learning: the problem of recovering policies that lie on the Pareto front given demonstrations from multiple Pareto-optimal experts in a Multi-Objective Markov Decision Process (MOMDP). Standard imitation approaches are ill-equipped for this regime, as naively aggregating conflicting expert trajectories can result in dominated policies. To address this, we introduce Multi-Output Augmented Behavioral Cloning (MA-BC), an algorithm that systematically partitions divergent expert data while pooling state-action pairs where no behavior conflict is observed. Theoretically, we prove that MA-BC converges to Pareto-optimal policies at a faster statistical rate than any learner that considers each expert dataset independently. Furthermore, we establish a novel lower bound for multi-objective imitation learning, demonstrating that MA-BC is minimax optimal. Finally, we empirically validate our alg
    
[^423]: LEAP：通过前瞻早期收敛Token检测解锁扩散语言模型的并行性

    LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection

    [https://arxiv.org/abs/2605.10980](https://arxiv.org/abs/2605.10980)

    提出LEAP，一种无需训练、即插即用的方法，通过前瞻性检测在去噪早期已收敛的token，突破现有置信度阈值的保守限制，从而显著加速扩散语言模型的并行解码。

    

    扩散语言模型因其高度并行处理的潜力而受到广泛关注。现有dLLM的并行能力源于高置信度下条件独立的假设，这确保了边缘分布与联合分布之间的差异可以忽略不计。然而，为保持准确性所需的严格置信度阈值严重限制了并行性的可扩展性。通过系统的token级统计分析，我们揭示了相当一部分token在去噪过程的早期就已收敛到正确的预测，但未能达到标准置信度阈值，这证实了当前基于置信度的判据过于保守。为此，我们提出了LEAP（通过前瞻早期收敛Token检测实现加速并行解码）。LEAP是一种无需训练、即插即用的方法，它利用未来上下文过滤来（摘要在此处截断）

    arXiv:2605.10980v2 Announce Type: replace  Abstract: Diffusion Language Models (dLLMs) have garnered significant attention for their potential in highly parallel processing. The parallel capabilities of existing dLLMs stem from the assumption of conditional independence at high confidence levels, which ensures negligible discrepancy between the marginal and joint distributions. However, the stringent confidence thresholds required to preserve accuracy severely constrain the scalability of parallelism. Through systematic token-level statistical analysis, we reveal that a substantial proportion of tokens converge to their correct predictions early in the denoising process yet fail to reach standard confidence thresholds, confirming that current confidence-based criteria are overly conservative. In response, we introduce LEAP (Lookahead Early-Convergence Token Detection for Accelerated Parallel Decoding). LEAP is a training-free, plug-and-play method that leverages future context filterin
    
[^424]: TIDES：选择性状态空间模型中的隐式时间感知

    TIDES: Implicit Time-Awareness in Selective State Space Models

    [https://arxiv.org/abs/2605.09742](https://arxiv.org/abs/2605.09742)

    TIDES 提出了一种选择性状态空间模型变体，通过将输入依赖性从离散化步长转移到对角状态矩阵，在原生处理不规则时间戳的同时保留了逐词符表达能力。

    

    选择性状态空间模型（SSM），如 Mamba，通过将时间离散化步长 $\Tilde{\Delta}$ 设为输入的学习函数，实现了强大的逐词符表达能力。然而，这样做使得 $\Tilde{\Delta}$ 不再等于连续观测之间的物理时间间隔 $\Delta$，限制了此类模型处理不规则时间序列的能力。连续时间 SSM，如 S5，保持 $\Tilde{\Delta}\equiv\Delta$，因此能够原生地处理不规则时间戳，但其动态仍是线性时不变（LTI）的，限制了逐词符表达能力。我们提出了 TIDES，这是一种选择性 SSM 变体，通过将输入依赖性从步长转移到对角状态矩阵上，从而调和了选择性与连续时间架构。由此，$\Tilde{\Delta}\equiv\Delta$（与 S5 一样），使模型能够原生地处理不规则时间戳，同时不牺牲使选择性 SSM 表现出色的逐词符表达能力。

    arXiv:2605.09742v3 Announce Type: replace  Abstract: Selective state space models (SSMs), such as Mamba, achieve strong per-token expressivity by making the time discretization step $\Tilde{\Delta}$ a learned function of the input. However, in doing so, $\Tilde{\Delta}$ no longer equals the physical time gap $\Delta$ between consecutive observations, limiting the ability of these models to handle irregular time series. Continuous time SSMs, such as S5, keep $\Tilde{\Delta}\equiv\Delta$ and therefore handle irregular timestamps natively, but their dynamics remain linear time invariant (LTI), limiting per token expressivity. We propose \textbf{TIDES}, a selective SSM variant that reconciles selective and continuous architectures by moving input dependence off the step size and onto the diagonal state matrix. As a result, $\Tilde{\Delta}\equiv\Delta$ as in S5, allowing the model to handle irregular timestamps natively without sacrificing the per-token expressivity that makes selective SSM
    
[^425]: TabFORGE：一个用于生成建模的表格基础模型

    TabFORGE: A Tabular Foundation Model for Generative Modelling

    [https://arxiv.org/abs/2605.09424](https://arxiv.org/abs/2605.09424)

    提出了TabFORGE——一个基于预训练表格基础生成表示的新型表格基础模型，通过更好地对齐表格数据的特征间依赖结构，显著提升生成式表格建模的合成数据质量。

    

    生成建模是对基础模型的一项高要求测试，因为它需要针对给定数据模态进行稳健、全面的表示学习，而不仅仅是针对监督预测目标进行优化。尽管近期关于表格基础模型的研究在预测建模方面取得了显著进展，但生成式表格基础模型仍未得到充分探索。尤其是，现有的表格基础生成器在合成数据质量方面尚未能持续匹敌强大的针对特定数据集的生成器。一个关键原因在于它们与表格数据的特征间依赖结构（可能反映潜在因果关系）的对齐不足。在本文中，我们通过引入一种新颖的表格基础模型 TabFORGE 来填补这一空白，该模型建立在预训练的表格基础生成表示（Tabular FOundational Representations for GEneration）之上。TabFORGE 旨在利用可迁移的（摘要在此处截断）

    arXiv:2605.09424v2 Announce Type: replace  Abstract: Generative modelling is a demanding test of foundation models, because it requires robust, holistic representation learning for a given data modality, rather than optimisation for a supervised prediction target alone. While recent work on tabular foundation models has achieved remarkable progress in predictive modelling, generative tabular foundation models remain underexplored. Existing tabular foundation generators, in particular, have not yet consistently matched strong dataset-specific generators in synthetic data quality. A key reason is their insufficient alignment with the inter-feature dependency structure of tabular data, which can reflect underlying causal relationships. In this paper, we address this gap by introducing a novel tabular foundation model, \textbf{TabFORGE}, built on pretrained \textbf{Tab}ular \textbf{FO}undational \textbf{R}epresentations for \textbf{GE}neration. TabFORGE is designed to utilise transferable 
    
[^426]: 基于软桥接策略的生成式行动者-评论家算法

    Generative Actor-Critic with Soft Bridge Policies

    [https://arxiv.org/abs/2605.08733](https://arxiv.org/abs/2605.08733)

    提出SoftGAC软生成式行动者-评论家算法，利用软桥接策略在单次网络评估内生成动作并提供可处理的软目标，同时解决了生成式策略训练中边际动作密度不可得和迭代采样推理与内存开销大的两大障碍。

    

    arXiv:2605.08733v2 公告类型：替换 摘要：扩散模型和流模型等表达力强的生成式策略对最大熵在线强化学习非常有吸引力，因为它们能够建模多峰且高度非高斯的动作分布。然而，训练有效的软生成式策略面临两个常常同时出现的障碍。首先，边际动作密度通常不可获得，因此现有方法通常依赖于熵的界、启发式代理或近似方法。其次，迭代式的共享参数采样器提高了推理成本，并且需要对重复的网络评估进行时间上的反向传播，从而增加了内存开销，并可能导致策略优化不稳定。这些障碍促使我们寻求一种生成式行动者-评论家方法，既能提供可处理的软目标，又只需一次网络评估即可生成每个动作。为此，我们提出了软生成式行动者-评论家算法（SoftGAC），其行动者是一个单（摘要在此处截断）

    arXiv:2605.08733v2 Announce Type: replace  Abstract: Expressive generative policies such as diffusion and flow models are appealing for maximum-entropy (MaxEnt) online reinforcement learning because of their ability to model multimodal and highly non-Gaussian action distributions. However, training effective soft generative policies faces two obstacles that often arise together. First, marginal action densities are often unavailable, so existing methods typically rely on entropy bounds, heuristic proxies or approximations. Second, iterative shared-parameter samplers raise inference cost and require backpropagation through time over repeated network evaluations, increasing memory cost and potentially destabilizing policy optimization. These obstacles motivate us to seek a generative actor-critic method that exposes a tractable soft objective while generating each action in just one network evaluation. To this end, we propose soft generative actor-critic (SoftGAC), whose actor is a singl
    
[^427]: 注意力迁移对视觉Transformer并非普遍有效

    Attention Transfer Is Not Universally Effective for Vision Transformers

    [https://arxiv.org/abs/2605.07191](https://arxiv.org/abs/2605.07191)

    本研究通过对来自11个ViT家族的20个教师模型进行系统基准测试，发现注意力迁移并非普遍有效——其中4个ViT家族的注意力迁移持续失败且性能低于从零训练的基线，且该失败在不同模型规模、训练时长、数据集和分布外评估下均保持一致，其根源被定位在注意力路由通道。

    

    最近的一项研究表明，注意力迁移——即仅将预训练教师视觉Transformer（ViT）的注意力模式迁移到随机初始化的标准学生ViT——就足以完全恢复教师预训练权重所带来的收益。我们在一个涵盖来自11个知名ViT家族的20个教师的综合基准上重新审视了这一发现，揭示出注意力迁移并非普遍有效。虽然其中7个家族的迁移能够成功，但另有4个家族持续失败，其性能甚至比从零开始训练（无迁移）的基线低出最多5.1%。进一步的结果表明，这种失败在不同模型规模上具有家族一致性，并且在延长训练时间、更换迁移数据集以及分布外评估的条件下依然存在。随后的对照分析一致地将问题定位到注意力路由通道，表明关键问题并不在于学生能否匹配教师的注意力……

    arXiv:2605.07191v2 Announce Type: replace-cross  Abstract: A recent work shows that Attention Transfer, which transfers only the attention patterns from a pre-trained teacher Vision Transformer (ViT) to a randomly initialized standard student ViT, is sufficient to recover the full benefit of the teacher's pre-trained weights. We revisit this finding on a comprehensive benchmark of 20 teachers from 11 well-known ViT families and reveal that Attention Transfer is not universally effective. While 7 families transfer successfully, 4 consistently fail, falling up to 5.1\% below the from-scratch no-transfer baseline. Further results demonstrate that this failure is family-consistent across model sizes, and persists under extended training durations, different transfer datasets, and out-of-distribution evaluations. Controlled analyses then consistently localize the problem to the attention-routing channel, indicating that the key issue is not whether the student can match the teacher's attent
    
[^428]: 从因果图到马尔可夫状态：面向深度强化学习的多阶状态暴露

    From Causal Graphs to Markovian States: Multi-Order State Exposure for Deep RL

    [https://arxiv.org/abs/2605.07057](https://arxiv.org/abs/2605.07057)

    提出MOSE方法，将单个共享Q函数同时暴露于1到W阶基于历史的状态构造，解决了如何从纵向因果图构造可证明满足马尔可夫性质的最小状态这一根本问题。

    

    在线强化学习（RL）依赖于马尔可夫性质来保证性能，但现实应用中仅凭原始观测变量往往缺乏良定义的状态。尽管因果强化学习受到越来越多的关注，现有工作通常假设马尔可夫状态是给定的，而仅将因果性用于加速学习。这留下了一个根本性的空白：给定一个关于观测变量的纵向因果图，如何构造可证明满足马尔可夫性质的MDP状态？我们提供了一种构造可证明的最小此类状态的方法。在深度强化学习中，仅有最小表示在实证上并不能提升性能，这表明神经网络无法自动利用马尔可夫最小性。因此，我们提出了MOSE（Multi-Order State Exposure，多阶状态暴露），将单个共享的Q函数同时暴露于1阶到W阶的基于历史的状态构造中，通过注意力机制……

    arXiv:2605.07057v2 Announce Type: replace  Abstract: Online reinforcement learning (RL) relies on the Markov property for guaranteed performance, but real-world applications often lack well-defined states given raw observed variables. While causal RL has attracted growing interest, existing work typically assumes Markovian states are provided and uses causality only to accelerate learning. This leaves a fundamental gap: \emph{given a longitudinal causal graph over observed variables, how does one construct MDP states that provably satisfy the Markov property?} We provide a procedure that constructs a provably minimal such state. In deep RL, the minimal representation alone empirically fails to improve performance, indicating that neural networks cannot automatically exploit Markovian minimality. We therefore propose \textbf{MOSE} (Multi-Order State Exposure), which exposes a single shared Q-function to history-based state constructions of orders 1 through $W$ simultaneously. An attenti
    
[^429]: 基于自编码器与代理驱动训练的冷原子储备池计算医学影像分类

    Medical Imaging Classification with Cold-Atom Reservoir Computing using Auto-Encoders and Surrogate-Driven Training

    [https://arxiv.org/abs/2605.06727](https://arxiv.org/abs/2605.06727)

    该论文提出了一种结合引导式自编码器与可微分代理模型的冷原子储备池计算混合框架，克服了量子测量不可微带来的梯度障碍，实现了医学影像（息肉检测）分类任务的端到端训练。

    

    我们提出了一种基于中性原子储备池计算的量子-经典混合流水线，用于医学图像分类，重点关注息肉检测这一二元分类任务。为了有效应对高维数据问题，我们集成了一个引导式自编码器。该流水线能够学习图像数据的紧凑且具有判别性的表示，这些表示也非常适合量子储备池计算。此类系统的一个关键挑战在于量子测量的不可微性，这为标准训练造成了“梯度障碍”。我们通过引入一个可微分代理模型来模拟量子层，从而克服了这一障碍，实现了对整个系统的端到端反向传播。这一引导式训练过程针对分类准确率和自编码器的忠实图像重建进行联合优化。学习到的潜在表示被编码为脉冲失谐参数……（原文摘要在此处截断）

    arXiv:2605.06727v2 Announce Type: replace  Abstract: We introduce a hybrid quantum-classical pipeline, based on neutral-atom reservoir computing, for medical image classification, focusing on the binary classification task of polyp detection. To deal effectively with the high dimensionality, we integrate a guided auto-encoder. This pipeline learns compact and discriminative representations of image data that are also well-suited for quantum reservoir computing. A key challenge in such systems is the non-differentiable nature of quantum measurements, which creates a 'gradient barrier' for standard training. We overcome this barrier by incorporating a differentiable surrogate model that emulates the quantum layer, enabling end-to-end backpropagation through the entire system. This guided training process is jointly optimized for classification accuracy and for faithful image recovery from the auto-encoder. The learned latent representations are encoded as pulse detuning parameters within
    
[^430]: 面向半监督视觉-语言学习的拓扑感知表示对齐

    Topology-Aware Representation Alignment for Semi-Supervised Vision-Language Learning

    [https://arxiv.org/abs/2604.26370](https://arxiv.org/abs/2604.26370)

    提出ToMA框架，利用持续同调将图文成对关系分解为H0-消亡与H1-诞生两类关系，并通过跨模态对应在模态间对齐，从而捕捉多模态表示流形的全局拓扑结构，提升半监督视觉-语言学习的泛化能力。

    

    视觉-语言模型已展现出强大的性能，但它们在专业领域的泛化能力往往较差。半监督视觉-语言学习通过利用少量带标注的图文对和大量无标注图像来缓解这一局限，然而现有方法从根本上仍是成对式的，无法建模多模态表示流形的全局结构。大多数现有的基于拓扑的对齐方法依赖于持续图匹配，这既不能保证几何对齐，也没有利用对视觉-语言学习至关重要的图文配对信息。我们提出了面向半监督视觉-语言学习的拓扑感知表示对齐框架ToMA，该框架利用持续同调将成对关系分解为 $H_0$-消亡关系与 $H_1$-诞生关系，并通过可用的跨模态对应关系在模态之间对这些关系进行对齐。这一轻量级的分解……

    arXiv:2604.26370v2 Announce Type: replace-cross  Abstract: Vision-language models have shown strong performance, but they often generalize poorly to specialized domains. While semi-supervised vision-language learning mitigates this limitation by leveraging a small set of labeled image-text pairs together with abundant unlabeled images, existing methods remain fundamentally pairwise and fail to model the global structure of multimodal representation manifolds. Most existing topology-based alignment methods rely on persistence diagram matching, which neither guarantees geometric alignment nor uses the image-text pairing information central to vision-language learning. We propose Topology-Aware Representation Alignment for Semi-Supervised Vision-Language Learning (ToMA), a framework that uses persistent homology to decompose pairwise relations into $H_0$-death and $H_1$-birth relations and aligns them across modalities through available cross-modal correspondences. This lightweight decomp
    
[^431]: 核扩展动态模态分解的字典学习

    Dictionary learning for Kernel EDMD

    [https://arxiv.org/abs/2604.25572](https://arxiv.org/abs/2604.25572)

    本文将 EDMD 中的字典学习方法扩展到 kEDMD 中的核学习，实现了核函数及其参数的自动学习，从而简化了 kEDMD 中核函数定义与参数选择的繁琐过程。

    

    通过状态空间行为来研究非线性动力系统可能颇具挑战性，一种可行的替代方法是借助其关联的 Koopman 算子来进行分析。这将非线性问题转化为线性、无穷维的问题。为了在有限维空间中逼近该算子，扩展动态模态分解（EDMD）是一种常用的算法。它需要一个有限的泛函列表和一组系统快照，以计算算子的近似及其对应的谱。与其直接选择泛函列表，不如通过核函数将其隐式定义，这种方法被称为核扩展动态模态分解（kEDMD）。然而，人们仍然需要定义核函数并选择其参数值。在本文中，我们旨在通过将 EDMD 的字典学习扩展为 kEDMD 中的核学习，来简化这一过程。通过简化 kEDMD，我们展示了如何进行……

    arXiv:2604.25572v2 Announce Type: replace-cross  Abstract: Studying nonlinear dynamical systems through their state space behavior can be challenging, and one possible alternative is to analyze them via their associated Koopman operator. This turns the nonlinear problem into a linear, infinite-dimensional one. To approximate the operator in finite dimensions, extended dynamic mode decomposition (EDMD) is a commonly used algorithm. It requires a finite list of functionals and a set of snapshots from the system to compute an approximation of the operator and its corresponding spectrum. Instead of choosing the list of functionals directly, it can be implicitly defined via kernels, a method known as kernel extended dynamic mode decomposition (kEDMD). However, one still needs to define the kernel and choose its parameter values.   In this paper, we aim to streamline this process by extending dictionary learning for EDMD to kernel learning in kEDMD. By simplifying kEDMD, we show how to perfo
    
[^432]: 大语言模型经济因果推理中的意识形态偏见

    Ideological Bias in LLMs' Economic Causal Reasoning

    [https://arxiv.org/abs/2604.21334](https://arxiv.org/abs/2604.21334)

    该研究在EconCausal基准中引入亲政府与亲市场观点存在分歧的意识形态争议案例，发现大语言模型在经济因果推理中表现出系统性意识形态偏见，且争议性案例的因果方向预测难度显著更高。

    

    大语言模型（LLM）在推理经济因果效应时是否会表现出系统性的意识形态偏见？随着大语言模型越来越多地被应用于政策分析和经济报道——在这些领域中，方向正确的因果判断至关重要——这一问题具有直接的现实意义。我们通过在EconCausal基准中扩展加入意识形态争议案例来开展系统性评估——即干预导向（亲政府）视角与市场导向（亲市场）视角会预测出相反因果符号的实例。我们从顶级经济学和金融学期刊中提取的10,490个因果三元组（具有经验验证效应方向的处理-结果对）中，识别出1,056个意识形态争议实例，并评估了20个最先进的大语言模型预测经验支持的因果方向的能力。我们发现，意识形态争议项目始终比非争议项目更难预测，并且在17个o……

    arXiv:2604.21334v3 Announce Type: replace-cross  Abstract: Do large language models (LLMs) exhibit systematic ideological bias when reasoning about economic causal effects? As LLMs are increasingly used in policy analysis and economic reporting, where directionally correct causal judgments are essential, this question has direct practical stakes. We present a systematic evaluation by extending the EconCausal benchmark with ideology-contested cases - instances where intervention-oriented (pro-government) and market-oriented (pro-market) perspectives predict divergent causal signs. From 10,490 causal triplets (treatment-outcome pairs with empirically verified effect directions) derived from top-tier economics and finance journals, we identify 1,056 ideology-contested instances and evaluate 20 state-of-the-art LLMs on their ability to predict empirically supported causal directions. We find that ideology-contested items are consistently harder than non-contested ones, and that across 17 o
    
[^433]: 基于因果约束的概率预测时间序列异常检测方法

    Causally-Constrained Probabilistic Forecasting for Time-Series Anomaly Detection

    [https://arxiv.org/abs/2604.17998](https://arxiv.org/abs/2604.17998)

    提出了一种因果引导Transformer（CGT）模型，通过将时滞因果图先验融入深度概率预测框架，实现了多变量时间序列异常检测中的因果解释与根因定位能力。

    

    多变量时间序列中的异常检测是工业监测领域的一项核心挑战，因为故障往往源于复杂的时间动态和跨传感器交互。尽管近期的深度学习模型，包括图神经网络和Transformer，已经展现出强大的实证性能，但大多数方法仍主要基于相关性，对因果解释和根因定位的支持有限。本研究引入了一个因果约束的概率预测框架，即因果引导Transformer（CGT）模型，用于多变量时间序列异常检测，该模型将显式的时滞因果图先验与深度序列建模相结合。对于每个目标变量，一个专用的预测模块采用由因果发现方法推导出的硬父节点掩码，将主要预测路径限制在因果图所支持的成因上，同时一个潜在高斯输出头用于捕获预测的不确定性。

    arXiv:2604.17998v2 Announce Type: replace  Abstract: Anomaly detection in multivariate time series is a central challenge in industrial monitoring, as failures frequently arise from complex temporal dynamics and cross-sensor interactions. While recent deep learning models, including graph neural networks and Transformers, have demonstrated strong empirical performance, most approaches remain primarily correlational and offer limited support for causal interpretation and root-cause localization. This study introduces a causally-constrained probabilistic forecasting framework which is a Causally Guided Transformer (CGT) model for multivariate time-series anomaly detection, integrating an explicit time-lagged causal graph prior with deep sequence modeling. For each target variable, a dedicated forecasting block employs a hard parent mask derived from causal discovery to restrict the main prediction pathway to graph-supported causes, while a latent Gaussian head captures predictive uncerta
    
[^434]: 扩展人机编码协作需要一个可治理的共识层

    Scaling Human-AI Coding Collaboration Requires a Governable Consensus Layer

    [https://arxiv.org/abs/2604.17883](https://arxiv.org/abs/2604.17883)

    该论文提出扩展人机编码协作需要引入一个可治理的共识层，通过版本化的项目状态模型将意图、结构性承诺、代码工件和证据关联起来，使人类能够在AI编码过程中检查、协调和修订共享的设计决策。

    

    氛围编码（Vibe coding）将自然语言意图转化为可运行的软件，但仅靠代码生成无法让人类和智能体对其正在构建的内容形成共同认知。随着协作跨越多个参与者、组件和版本发布，设计决策必须保持可理解且可修订。我们认为，扩展人类与AI的编码协作需要一个可治理的共识层。Agentic Consensus（智能体共识）围绕一个版本化的项目状态模型来组织协作，该模型将选定的意图、结构性承诺、可执行工件和证据联系在一起。参与者可以在实现之前澄清意图，并基于人类可检查、可修订的承诺来协调提议的更改。双向同步将预期变更与实现连接起来，并在工件演进过程中暴露冲突或过时的证据。我们在Consensus Control中实现了该审查工作流。在一项小规模探索性人类研究中，10名参与者……

    arXiv:2604.17883v2 Announce Type: replace-cross  Abstract: Vibe coding turns natural-language intent into working software, but code generation alone does not give humans and agents a shared account of what they are building. As collaboration spans participants, components, and releases, design choices must remain understandable and open to revision. We argue that scaling human and AI coding collaboration requires a governable consensus layer. Agentic Consensus organizes collaboration around a versioned project-state model linking selected intent, structural commitments, executable artifacts, and evidence. Participants can clarify intent before implementation and coordinate proposed changes against commitments that humans can inspect and revise. Bidirectional synchronization connects intended changes to implementation and exposes conflicts or stale evidence as artifacts evolve. We implemented its review workflow in Consensus Control. In a small exploratory human study, 10 participants 
    
[^435]: StaRPO：稳定性增强的强化策略优化

    StaRPO: Stability-Augmented Reinforcement Policy Optimization

    [https://arxiv.org/abs/2604.08905](https://arxiv.org/abs/2604.08905)

    提出StaRPO框架，通过自相关函数（ACF）和路径效率（PE）两个轻量级指标将推理稳定性显式纳入强化学习优化目标，从而改善大语言模型推理过程的逻辑一致性与结构稳定性。

    

    强化学习（RL）在提升大型语言模型于复杂推理任务中的准确性方面非常有效。然而，现有的RL策略优化框架依赖最终答案的正确性作为反馈信号，很少捕捉推理过程的内部逻辑结构。因此，模型会生成流畅且语义相关的回答，但往往存在逻辑不一致、结构混乱或冗余的问题。为此，我们提出了StaRPO，这是一种稳定性增强的强化学习框架，将推理稳定性显式地纳入优化目标。StaRPO将稳定性分解为两个可计算的轻量级指标：自相关函数（ACF）用于评估局部的逐步连贯性，路径效率（PE）用于评估推理轨迹的全局目标导向性。这些稳定性奖励与任务奖励相结合，提供互补的、过程性的信号。

    arXiv:2604.08905v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) is effective in enhancing the accuracy of large language models in complex reasoning tasks. Existing RL policy optimization frameworks rely on final-answer correctness as feedback signals and rarely capture the internal logical structure of the reasoning process. Consequently, the models would generate fluent and semantically relevant responses but logically inconsistent, structurally erratic, or redundant. To this end, we propose StaRPO, a stability-augmented reinforcement learning framework that explicitly incorporates reasoning stability into the optimization objective. Our StaRPO decomposes stability into two computable lightweight metrics: the Autocorrelation Function (ACF) to evaluate local step-to-step coherence, and Path Efficiency (PE) to evaluate global goal-directedness of the reasoning trajectory. These stability rewards are combined with task rewards to provide complementary and process-
    
[^436]: 足球传球结构分析：从时空追踪数据中学习传球原型与战术影响

    Structural Pass Analysis in Football: Learning Pass Archetypes and Tactical Impact from Spatio-Temporal Tracking Data

    [https://arxiv.org/abs/2603.28916](https://arxiv.org/abs/2603.28916)

    该论文提出了由线突破得分（LBS）、空间增益度量（SGM）和结构破坏指数（SDI）组合而成的可解释战术影响值（TIV）框架，利用时空追踪数据量化足球传球对对方防守结构的突破与破坏效果。

    

    传球动作通常通过球权推进和进攻结果来评估，然而传球与对方空间组织之间的关系为传球行为提供了另一个观察视角。本研究引入了一个可解释的框架，利用同步的事件数据和追踪数据来描述传球结构。该框架提出三个互补的描述指标：线突破得分（LBS），衡量传球者与接球者之间球门侧防守人数的差异；空间增益度量（SGM），比较二者与最近防守球员的距离；以及结构破坏指数（SDI），概括传球发起时与一秒后采样时刻防守组织的差异。经过转换与标准化处理后，这些指标被组合成战术影响值（TIV），一个固定权重的结构性综合指标。该框架应用于2022年卡塔尔世界杯的41,078个运动战传球事件……

    arXiv:2603.28916v3 Announce Type: replace  Abstract: Passing actions are commonly evaluated through ball progression and attacking outcomes, yet their relationship with the spatial organisation of the opposition provides an additional perspective on passing behaviour. This study introduces an interpretable framework for describing passing structure using synchronised event and tracking data. Three complementary descriptors are proposed: Line Bypass Score (LBS), measuring the difference in goal-side opponent counts between passer and receiver; Space Gain Metric (SGM), comparing their nearest-opponent distances; and Structural Disruption Index (SDI), summarising differences in defensive organisation between pass initiation and a sampled state one second later. Following transformation and standardisation, these descriptors are combined into Tactical Impact Value (TIV), a fixed-weight structural composite. Applied to 41,078 open-play passing events from the 2022 FIFA World Cup, the framew
    
[^437]: 面向随机材料参数固体力学问题的本构参数化深度能量方法

    Constitutive parameterized deep energy method for solid mechanics problems with random material parameters

    [https://arxiv.org/abs/2603.26030](https://arxiv.org/abs/2603.26030)

    提出本构参数化深度能量方法（CPDEM），这是一种纯物理驱动的框架，可高效求解具有随机材料参数的固体力学问题，避免了有限元方法重复网格划分的高昂计算成本、数据驱动代理模型对大规模数据集的依赖，以及标准深度能量方法在参数变化时需完全重新训练的缺陷。

    

    在实际结构设计和固体力学仿真中，材料属性固有地在有界区间内表现出随机变化。然而，在连续材料不确定性条件下评估力学响应仍然是一个持续的挑战。传统数值方法（如有限元法（FEM））的计算成本极其高昂，因为它们需要对每一个参数化实现重复进行网格离散化和方程求解。同样，数据驱动的代理模型严重依赖大规模、高保真度的数据集，而标准的物理驱动框架（如深度能量方法）在材料参数发生变化时则严格要求从头开始完全重新训练。为了弥合这一关键差距，我们提出了本构参数化深度能量方法（CPDEM）。在这一纯物理驱动的框架中，应变能密度泛函通过编码潜在代表性表示被重新构建……（摘要在此处不完整）

    arXiv:2603.26030v2 Announce Type: replace  Abstract: In practical structural design and solid mechanics simulations, material properties inherently exhibit random variations within bounded intervals. However, evaluating mechanical responses under continuous material uncertainty remains a persistent challenge. Traditional numerical approaches, such as the Finite Element Method (FEM), incur prohibitive computational costs as they require repeated mesh discretization and equation solving for every parametric realization. Similarly, data-driven surrogate models depend heavily on massive, high-fidelity datasets, while standard physics-informed frameworks (e.g., the Deep Energy Method) strictly demand complete retraining from scratch whenever material parameters change. To bridge this critical gap, we propose the Constitutive Parameterized Deep Energy Method (CPDEM). In this purely physics-driven framework, the strain energy density functional is reformulated by encoding a latent representat
    
[^438]: 属性化高斯图匹配的尖锐统计阈值

    Sharp Statistical Thresholds for Attributed Gaussian Graph Matching

    [https://arxiv.org/abs/2603.23305](https://arxiv.org/abs/2603.23305)

    本文针对边权重与节点特征联合相关的属性化高斯图匹配问题，推导出了精确恢复、几乎精确恢复和部分恢复的精确信息论阈值，并发现节点特征信息的引入使得精确恢复阈值从共享阈值中分离，呈现出比标准图匹配更丰富的相变结构。

    

    我们研究网络匹配问题，其中边权重和节点特征都按照特定的节点对应关系相关联。虽然先前的工作已经为单独匹配图或特征分别建立了尖锐阈值，但联合设置的情形却很少被探索。在本研究中，我们推导了底层对齐的精确恢复、几乎精确恢复和部分恢复的精确信息论阈值。这些阈值由图和特征的相关强度、节点数量以及特征维度来表示。有趣的是，尽管在标准图匹配场景中观察到“全有或全无”的相变现象，但节点级上下文信息的引入带来了更丰富的结构：精确恢复阈值从几乎精确恢复和部分恢复的共享阈值中分离出来。填补了先前工作留下的空白，我们提供……

    arXiv:2603.23305v2 Announce Type: replace-cross  Abstract: We investigate the problem of matching networks where both edge weights and node features are correlated following a specific node correspondence. While prior work has established sharp thresholds for matching either graphs or features individually, the joint setting has been significantly less explored. In this study, we derive precise information-theoretic thresholds for exact recovery, almost-exact recovery, and partial recovery of the underlying alignment. These are expressed in terms of graph and feature correlation strengths, the number of nodes, and the dimension of the features. Interestingly, whereas an all-or-nothing phase transition is observed in the standard graph-matching scenario, the inclusion of contextual information at the node level introduces a richer structure: the exact-recovery threshold separates from the shared threshold for almost-exact and partial recovery. Closing gaps left by prior works, we provid
    
[^439]: 基于未来状态-动作访问度量的最大熵探索

    Maximum-Entropy Exploration with Future State-Action Visitation Measures

    [https://arxiv.org/abs/2603.18965](https://arxiv.org/abs/2603.18965)

    该论文提出了一种与未来状态-动作特征折扣分布的熵成比例的内在奖励方法，理论上证明其期望总和是从初始状态出发的轨迹所访问分布熵的下界且可通过收缩算子的不动点性质进行离策略估计，从而实现了更优的最大熵探索。

    

    最大熵强化学习激励智能体探索状态和动作以最大化某个分布的熵，通常做法是提供与该熵函数成比例的额外内在奖励。在本文中，我们研究了与未来时间步中所访问的状态-动作特征的折扣分布熵成比例的内在奖励。该方法基于两个理论结果：首先，我们证明了这些内在奖励的期望总和是从初始状态出发的轨迹中所访问状态-动作特征折扣分布的熵的下界，我们将其与另一种最大熵目标联系起来；其次，我们证明了内在奖励定义中使用的分布是某个收缩算子的不动点，因此可以进行离策略估计。实验表明，新目标通常能够带来对特征更好的访问覆盖。

    arXiv:2603.18965v2 Announce Type: replace  Abstract: Maximum entropy reinforcement learning motivates agents to explore states and actions to maximize the entropy of some distribution, typically by providing additional intrinsic rewards proportional to that entropy function. In this paper, we study intrinsic rewards proportional to the entropy of the discounted distribution of state-action features visited during future time steps. This approach is motivated by two results. First, we show that the expected sum of these intrinsic rewards is a lower bound on the entropy of the discounted distribution of state-action features visited in trajectories starting from the initial states, which we relate to an alternative maximum entropy objective. Second, we show that the distribution used in the intrinsic reward definition is the fixed point of a contraction operator and can therefore be estimated off-policy. Experiments highlight that the new objective often leads to improved visitation of f
    
[^440]: 平衡Forman曲率与Ollivier-Ricci曲率之间的逐边包络界

    Edgewise Envelopes Between Balanced Forman and Ollivier-Ricci Curvature

    [https://arxiv.org/abs/2603.13535](https://arxiv.org/abs/2603.13535)

    该论文通过推导平衡Forman曲率与Ollivier-Ricci曲率之间的显式双侧分段仿射界，将图上OR曲率的逐边计算从最优传输线性规划降至最坏情况O(最大度^2.5)时间，大幅提升大规模图上曲率计算的效率。

    

    在大规模图上计算Ollivier-Ricci（OR）曲率在计算上是极其昂贵的，因为需要为每条边求解一个最优传输问题。我们通过推导基于传输的OR曲率与组合式的平衡Forman（BF）曲率之间的显式、双侧、分段仿射转移模（transfer moduli）来绕过这一瓶颈。我们建立了以2跳局部图组合结构为参数的 $\mathfrak{c}_{\rm OR}(i,j)$ 的确定性界，将逐边评估的复杂度从最优传输线性规划降低到最坏情况下 $\mathcal{O}\left(\max_{v \in V} \operatorname{deg}(v)^{2.5}\right)$ 的时间，完全消除了对全局求解器的依赖。经验可扩展性基准测试证实了这些理论保证，表明所提出的转移模相比精确OR计算的陡峭多项式扩展提供了显著的渐近加速和常数因子加速。此外……

    arXiv:2603.13535v5 Announce Type: cross  Abstract: Evaluating Ollivier-Ricci (OR) curvature on large-scale graphs is computationally prohibitive due to the necessity of solving an optimal transport problem for every edge. We bypass this bottleneck by deriving explicit, two-sided, piecewise-affine transfer moduli between the transport-based OR curvature and the combinatorial Balanced Forman (BF) curvature. We establish deterministic bounds for $\mathfrak{c}_{\rm OR}(i,j)$ parameterized by 2-hop local graph combinatorics, reducing the edgewise evaluation complexity from an optimal transport linear program to a worst-case $\mathcal{O}\left(\max_{v \in V} \operatorname{deg}(v)^{2.5}\right)$ time, entirely eliminating the reliance on global solvers. Empirical scalability benchmarks confirm these theoretical guarantees, demonstrating that the proposed transfer moduli yield significant asymptotic and constant-factor speedups over the steep polynomial scaling of exact OR evaluation. Furthermor
    
[^441]: 推理陷阱——逻辑推理作为通向情境感知的机制性路径

    The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness

    [https://arxiv.org/abs/2603.09200](https://arxiv.org/abs/2603.09200)

    该论文提出RAISE框架，揭示提升大语言模型的演绎、归纳与溯因逻辑推理能力会经由三条机制性路径逐步促成AI的情境感知能力，警示逻辑推理研究与AI安全之间的冲突风险。

    

    情境感知是指AI系统认识自身本质、理解其训练与部署环境、并对自身处境进行策略性推理的能力，被广泛认为是先进AI系统中最危险的新兴能力之一。与此同时，越来越多的研究致力于提升大语言模型（LLM）在演绎、归纳和溯因三方面的逻辑推理能力。本文认为，这两条研究轨迹正在走向碰撞。我们提出了RAISE框架（Reasoning Advancing Into Self Examination，推理进阶为自我审视），该框架识别出三条机制性路径，通过这些路径，逻辑推理能力的提升能够使AI逐步获得更深层次的情境感知：演绎式自我推断、归纳式情境识别和溯因式自我建模。我们对每条路径进行了形式化，并构建了一条从基础自我识别到策略性（此处摘要原文不完整）

    arXiv:2603.09200v2 Announce Type: replace-cross  Abstract: Situational awareness, the capacity of an AI system to recognize its own nature, understand its training and deployment context, and reason strategically about its circumstances, is widely considered among the most dangerous emergent capabilities in advanced AI systems. Separately, a growing research effort seeks to improve the logical reasoning capabilities of large language models (LLMs) across deduction, induction, and abduction. In this paper, we argue that these two research trajectories are on a collision course. We introduce the RAISE framework (Reasoning Advancing Into Self Examination), which identifies three mechanistic pathways through which improvements in logical reasoning enable progressively deeper levels of situational awareness: deductive self inference, inductive context recognition, and abductive self modeling. We formalize each pathway, construct an escalation ladder from basic self recognition to strategic 
    
[^442]: NePPO：面向一般和博弈多智能体强化学习的近势函数策略优化

    NePPO: Near-Potential Policy Optimization for General-Sum Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2603.06977](https://arxiv.org/abs/2603.06977)

    提出NePPO方法，通过学习一个与玩家无关的势函数，将一般和博弈转化为以该势函数为共同效用的合作博弈，使其纳什均衡近似原博弈的纳什均衡，从而在混合合作-竞争环境中实现近似纳什均衡的计算。

    

    多智能体强化学习（MARL）越来越多地被用于设计在共享环境中进行交互的具备学习能力的智能体。然而，在一般和博弈中训练MARL算法仍然充满挑战：学习动态可能变得不稳定，且收敛性保证通常仅在双人零和博弈或完全合作博弈等受限场景下成立。此外，当智能体具有异构且可能相互冲突的偏好时，尚不清楚应以何种系统级目标来指导学习。本文提出了一种新的MARL流程——近势函数策略优化，用于在混合合作-竞争环境中计算近似纳什均衡。其核心思想是学习一个与玩家无关的势函数，使得以该势函数作为共同效用的合作博弈的纳什均衡能够近似原博弈的纳什均衡。为此，我们引入……（原文在此处截断）

    arXiv:2603.06977v3 Announce Type: replace  Abstract: Multi-agent reinforcement learning (MARL) is increasingly used to design learning-enabled agents that interact in shared environments. However, training MARL algorithms in general-sum games remains challenging: learning dynamics can become unstable, and convergence guarantees typically hold only in restricted settings such as two-player zero-sum or fully cooperative games. Moreover, when agents have heterogeneous and potentially conflicting preferences, it is unclear what system-level objective should guide learning. In this paper, we propose a new MARL pipeline called Near-Potential Policy Optimization (NePPO) for computing approximate Nash equilibria in mixed cooperative--competitive environments. The core idea is to learn a player-independent potential function such that the Nash equilibrium of a cooperative game with this potential as the common utility approximates a Nash equilibrium of the original game. To this end, we introdu
    
[^443]: TimeWarp：通过重访过去来评估网页智能体

    TimeWarp: Evaluating Web Agents by Revisiting the Past

    [https://arxiv.org/abs/2603.04949](https://arxiv.org/abs/2603.04949)

    提出了模拟网页UI演变的TimeWarp基准测试，揭示现有网页智能体对界面变化的脆弱性，并通过TimeTraj标注方法与BC变体训练显著提升了智能体跨网页版本的泛化能力。

    

    随着网页智能体在基准测试中逐渐缩小与人类的差距，一个问题随之而来：今天的智能体在明天的网页上是否也能表现得同样出色？我们提出了TimeWarp，一个模拟不断演变的网页环境的基准测试。TimeWarp包含三个网页环境，每个环境有六个UI版本，涵盖了互联网不同时代的UI设计、前端代码和工作流程。我们为TimeWarp配套了一系列复杂、真实的任务，涵盖不同形式的网页导航。我们的实验表明，基于视觉的智能体容易受到网页变化的影响，而基于文本的智能体一旦在单一版本上微调就会变得脆弱。为了解决这个问题，我们提出了TimeTraj，一种新的标注方法，通过计划蒸馏来收集跨多个版本的轨迹。通过使用我们的BC变体方法在教师rollout上训练智能体，我们取得了显著的性能提升：Qwen-3 4B模型从20.4%提升到37.7%，Llama-3.1 8B模型从0%提升到27.0%。

    arXiv:2603.04949v2 Announce Type: replace  Abstract: As web agents close the gap with humans on benchmarks, one question arises: Do today's agents perform just as well on tomorrow's web? We introduce TimeWarp, a benchmark that emulates the evolving web. TimeWarp consists of three web environments, each with six UI versions spanning UI design, frontend code, and workflows from different eras of the internet. We pair TimeWarp with a set of complex, realistic tasks covering different forms of web navigation. Our experiments reveal that vision-based agents are vulnerable to changes, while text-based agents become brittle once fine-tuned on a single version. To address this, we propose TimeTraj, a new annotation method that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: 20.4% to 37.7% for Qwen-3 4B and 0% to 27.0% for Llama-3.1 8B models. Our work helps study gene
    
[^444]: MARS：通过自适应秩搜索实现多模态收敛的协调

    MARS: Harmonizing Multimodal Convergence via Adaptive Rank Search

    [https://arxiv.org/abs/2603.00720](https://arxiv.org/abs/2603.00720)

    提出MARS方法，利用双缩放定律自适应搜索LoRA秩对，以平衡多模态大语言模型微调中不同模态的收敛速度，避免负面干扰并提升任务性能。

    

    使用低秩适应（LoRA）等参数高效方法对多模态大语言模型（MLLMs）进行微调对于任务适配至关重要。然而，跨模态之间不平衡的训练动态常常因负面干扰而导致次优的准确率，这一挑战通常采用低效的启发式方法来解决，例如手动调整不同模态的学习率。为了克服这一问题，我们提出了MARS（多模态自适应秩搜索），一种能够发现最佳秩对的方法，在平衡训练动态的同时最大化性能。我们的关键创新是提出的双缩放定律框架，为该搜索提供了支撑：一个定律建模模块特定的收敛时间，将搜索空间剪枝为动态对齐的候选集合；另一个定律则预测最终任务性能，从剪枝后的集合中选出最优的秩对。通过将LoRA的秩重新用作控制模态特定收敛速度的控制器，MARS在……

    arXiv:2603.00720v2 Announce Type: replace  Abstract: Fine-tuning Multimodal Large Language Models (MLLMs) with parameter-efficient methods like Low-Rank Adaptation (LoRA) is crucial for task adaptation. However, imbalanced training dynamics across modalities often lead to suboptimal accuracy due to negative interference, a challenge typically addressed with inefficient heuristic methods such as manually tuning separate learning rates. To overcome this, we introduce MARS (Multimodal Adaptive Rank Search), an approach to discover optimal rank pairs that balance training dynamics while maximizing performance. Our key innovation, a proposed framework of dual scaling laws, enables this search: one law models module-specific convergence time to prune the search space to candidates with aligned dynamics, while the other predicts final task performance to select the optimal pair from the pruned set. By re-purposing the LoRA rank as a controller for modality-specific convergence speed, MARS out
    
[^445]: 基于LoRA的持续学习中面向知识共享与隔离的任务驱动子空间分解

    Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning

    [https://arxiv.org/abs/2603.00191](https://arxiv.org/abs/2603.00191)

    提出LoDA方法，从投影能量视角出发进行任务驱动的低秩子空间分解，在基于LoRA的持续学习中同时实现跨任务知识共享与真正有效的任务特定隔离。

    

    持续学习（CL）要求模型在不遗忘旧知识的前提下依次适应新任务。近来，低秩适配作为参数高效微调（PEFT）的代表性方法，在持续学习领域受到越来越多的关注。一些基于LoRA的持续学习方法通过分离各任务的更新空间来减少任务间的干扰，通常是利用过去任务估计的零空间来构建新的更新空间。然而，这些方法（i）忽略了任务共享的方向，从而抑制了知识迁移；（ii）无法捕捉真正有效的任务特定方向，因为在任务相关的情形下，旧任务的“零空间基”对新任务而言可能几乎不活跃。为解决这一问题，我们从投影能量的视角研究了LoRA的学习能力，并提出了低秩分解与适配方法（LoDA）。该方法通过执行任务驱动的分解，求解两个能量（优化问题），从而构建出通用的以及真正任务特定的LoRA子空间。

    arXiv:2603.00191v5 Announce Type: replace  Abstract: Continual Learning (CL) requires models to sequentially adapt to new tasks without forgetting old knowledge. Recently, Low-Rank Adaptation (LoRA), a representative Parameter-Efficient Fine-Tuning (PEFT) method, has gained increasing attention in CL. Several LoRA-based CL methods reduce interference across tasks by separating their update spaces, typically building the new space from the estimated null space of past tasks. However, they (i) overlook task-shared directions, which suppresses knowledge transfer, and (ii) fail to capture truly effective task-specific directions since these ``null bases" of old tasks can remain nearly inactive for new task under correlated tasks. To address this, we study LoRA learning capability from a projection energy perspective, and propose Low-rank Decomposition and Adaptation (LoDA). It performs a task-driven decomposition to build general and truly task-specific LoRA subspaces by solving two energy
    
[^446]: 参数高效的时序域泛化

    Parameter-Efficient Temporal Domain Generalization

    [https://arxiv.org/abs/2602.11965](https://arxiv.org/abs/2602.11965)

    LoTiC通过将时序参数更新分解为共享的低秩基和一个紧凑的时变核心，在r×r的小空间中学习时序动力学，从而实现对现代大模型参数高效的时序域泛化。

    

    时序域泛化从一系列已观测的域中学习，以预测未见未来域的模型。现有的参数动力学方法在现代主干网络规模下变得极其昂贵，因为其时序编码器和生成器需要对整个适配后的参数状态进行操作。用LoRA替换全模型更新并不能消除这一失效模式：动力学模型仍然必须编码并重构每一个适配器参数，使得辅助模型本身对于现代主干网络而言规模过于庞大。我们提出了低秩时变核心方法，这是一种参数高效的TDG方法，它将时序更新分解为共享的低秩基和一个紧凑的核心。低秩基定义了一个稳定的更新子空间，而只有核心会随时间演化。因此，LoTiC在一个r×r的空间中学习时序动力学，而不必追踪完整的适配器。我们通过参数增量的几何结构来论证这种分解的合理性。

    arXiv:2602.11965v2 Announce Type: replace  Abstract: Temporal domain generalization (TDG) learns from a sequence of observed domains to predict models for unseen future domains. Existing parameter-dynamics methods become prohibitively expensive at modern backbone scales because their temporal encoders and generators operate on the entire adapted parameter state. Replacing full-model updates with LoRA does not remove this failure mode: the dynamics model must still encode and reconstruct every adapter parameter, making the auxiliary model itself prohibitively large for modern backbones. We propose Low-rank Time-varying Core (LoTiC), a parameter-efficient TDG method that decomposes temporal updates into shared low-rank bases and a compact core. The bases define a stable update subspace, while only the core evolves. LoTiC therefore learns temporal dynamics in an r*r space instead of tracking the full adapter. We justify this factorization through the geometry of parameter increments and s
    
[^447]: 具有不确定性量化与引导的流匹配

    Flow Matching with Uncertainty Quantification and Guidance

    [https://arxiv.org/abs/2602.10326](https://arxiv.org/abs/2602.10326)

    提出了不确定性感知流匹配UA-Flow，通过预测速度场的异方差不确定性来量化生成样本的可靠性，并利用该信号实现不确定性感知引导，从而提升生成样本的保真度。

    

    尽管流匹配等基于采样的生成模型取得了显著成功，它们仍可能产生质量不一致或退化的样本。为了评估样本的可靠性并提高样本的保真度，我们提出了不确定性感知流匹配，这是流匹配的一个轻量级扩展，能够同时预测速度场和异方差不确定性。UA-Flow 通过在流动力学中传播速度不确定性来估计空间局部化的样本不确定性。所得的估计值可作为可靠性信号，反映数据流形对每个生成样本或区域的支持程度。我们进一步利用这些信号，通过不确定性感知引导（UG）和不确定性感知无分类器引导（UCFG）来引导生成过程。在 ImageNet 和 CIFAR-10 上的实验表明，UA-Flow 的不确定性比逐元素基线方法更能一致地跟踪样本保真度，并且这些信号还携带有价值的信息。

    arXiv:2602.10326v3 Announce Type: replace-cross  Abstract: Despite the remarkable success of sampling-based generative models such as flow matching, they can still produce samples of inconsistent or degraded quality. To assess sample reliability and improve sample fidelity, we propose uncertainty-aware flow matching (UA-Flow), a lightweight extension of flow matching that predicts the velocity field together with heteroscedastic uncertainty. UA-Flow estimates spatially localized sample uncertainty by propagating velocity uncertainty through the flow dynamics. The resulting estimates serve as reliability signals that reflect data-manifold support for each generated sample or region. We further use them to steer generation via uncertainty-aware guidance (UG) and uncertainty-aware classifier-free guidance (UCFG). Experiments on ImageNet and CIFAR-10 show that UA-Flow's uncertainty tracks sample fidelity more consistently than element-wise baselines and that these signals also carry inform
    
[^448]: ANCRe：面向高效深度扩展的自适应神经连接重分配

    ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling

    [https://arxiv.org/abs/2602.09009](https://arxiv.org/abs/2602.09009)

    该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。

    

    摘要（arXiv:2602.09009v2，公告类型：replace-cross）：扩展网络深度一直是现代基础模型成功的核心驱动力，然而近期研究表明，深层网络往往未被充分利用。本文从优化视角重新审视了加深神经网络的默认机制——残差连接。严格的分析证明，残差连接的布局能够从根本上塑造收敛行为，甚至会引发收敛速率上的指数级差距。受此启发，我们提出了自适应神经连接重分配，这是一个具有理论依据且轻量级的框架，能够从数据中参数化并学习残差连接方式。ANCRe 以可忽略不计的计算和内存开销（<1%）自适应地重新分配残差连接，同时使网络深度得到更有效的利用。我们在大型语言模型预训练、扩散模型以及深度R（原文此处截断）等任务上进行了大量数值测试……

    arXiv:2602.09009v2 Announce Type: replace-cross  Abstract: Scaling network depth has been a central driver behind the success of modern foundation models, yet recent investigations suggest that deep layers are often underutilized. This paper revisits the default mechanism for deepening neural networks, namely residual connections, from an optimization perspective. Rigorous analysis proves that the layout of residual connections can fundamentally shape convergence behavior, and even induces an exponential gap in convergence rates. Prompted by this insight, we introduce adaptive neural connection reassignment (ANCRe), a principled and lightweight framework that parameterizes and learns residual connectivities from the data. ANCRe adaptively reassigns residual connections with negligible computational and memory overhead ($<1\%$), while enabling more effective utilization of network depth. Extensive numerical tests across pre-training of large language models, diffusion models, and deep R
    
[^449]: 连接学术界与工业界：面向属性图聚类的综合基准

    Bridging Academia and Industry: A Comprehensive Benchmark for Attributed Graph Clustering

    [https://arxiv.org/abs/2602.08519](https://arxiv.org/abs/2602.08519)

    该论文提出了PyAGC——首个面向属性图聚类、可直接用于生产环境的综合基准库，通过统一的“编码-聚类-优化”模块化框架和内存高效的小批次实现，弥合了学术研究与真实工业部署之间的鸿沟。

    

    属性图聚类是一种基础的无监督任务，它将结构拓扑与节点属性相结合，以挖掘图结构数据中的潜在模式。尽管属性图聚类在欺诈检测、用户细分等工业应用中具有重要意义，但学术研究与实际工业部署之间仍存在显著的鸿沟。当前的评估方案存在诸多局限：数据集局限于规模小、同质性高的引文网络数据集，采用难以扩展的全批次训练范式，并且依赖有监督指标，无法真实反映标签稀缺环境下的性能。为弥合这些差距，我们提出了PyAGC——一个全面、可用于生产环境的基准测试库，旨在对属性图聚类方法在不同规模和结构特性下进行压力测试。我们将现有方法统一到一个模块化的“编码-聚类-优化”框架中，并首次为广泛的（方法）提供了内存高效的小批次实现。

    arXiv:2602.08519v2 Announce Type: replace  Abstract: Attributed Graph Clustering (AGC) is a fundamental unsupervised task that integrates structural topology and node attributes to uncover latent patterns in graph-structured data. Despite its significance in industrial applications such as fraud detection and user segmentation, a significant chasm persists between academic research and real-world deployment. Current evaluation protocols suffer from the small-scale, high-homophily citation datasets, non-scalable full-batch training paradigms, and a reliance on supervised metrics that fail to reflect performance in label-scarce environments. To bridge these gaps, we present PyAGC, a comprehensive, production-ready benchmark and library designed to stress-test AGC methods across diverse scales and structural properties. We unify existing methodologies into a modular Encode-Cluster-Optimize framework and, for the first time, provide memory-efficient, mini-batch implementations for a wide a
    
[^450]: 流匹配对于序列数据仅仅是轨迹重放吗？

    Is Flow Matching Just Trajectory Replay for Sequential Data?

    [https://arxiv.org/abs/2602.08318](https://arxiv.org/abs/2602.08318)

    本文推导了序列数据上流匹配目标的最优速度场，揭示其本质是一个非参数、记忆增强的连续时间动力系统，其速度场可表示为观测转移的瞬时速度的加权混合，为神经流匹配模型提供可解释视角并提出FreeFM。

    

    流匹配在科学领域中被越来越多地用于时间序列的生成与预测，而这些数据往往源自底层的动力系统。然而，目前尚不清楚流匹配究竟是学习到了可迁移的动力学结构，还是仅仅执行了一种有效的“轨迹重放”。我们通过推导经验流匹配目标在序列数据上所对应的最优速度场来研究这一问题。对于实践中常用的高斯条件路径，我们证明由此得到的采样器是一个常微分方程（ODE），其动力学描述了一个非参数的、记忆增强的连续时间动力系统。该驱动速度场具有闭式表达式，即由观测到的状态转移所确定的瞬时速度的加权混合。这一刻画为神经流匹配模型提供了一种可解释的视角——将其视为对底层非参数目标的参数化估计——并自然引出了FreeFM：一种...

    arXiv:2602.08318v3 Announce Type: replace-cross  Abstract: Flow matching (FM) is increasingly used in scientific domains for time series generation and forecasting, where data often arise from underlying dynamical systems. However, it is not well-understood whether it learns transferable dynamical structure or simply performs an effective ''trajectory replay''. We study this question by deriving the optimal velocity field targeted by the empirical FM objective on sequential data. For the Gaussian conditional paths commonly used in practice, we show that the resulting sampler is an ODE whose dynamics describe a nonparametric, memory-augmented continuous-time dynamical system. The driving velocity field admits a closed-form expression as a weighted mixture of instantaneous velocities determined by observed transitions. This characterization provides an interpretable view of neural FM models as parametric estimation of an underlying nonparametric target and naturally leads to FreeFM: a cl
    
[^451]: 论线性模型中延迟泛化的动力学

    On the Dynamics of Delayed Generalization in Linear Models

    [https://arxiv.org/abs/2602.08302](https://arxiv.org/abs/2602.08302)

    本文证明延迟泛化现象并非深度网络独有，在线性可分数据上用梯度下降和指数尾损失训练的线性模型中同样会出现，并将超平面动力学刻画为总体、支持向量主导的遗忘和泛化三个阶段，其中动力学的时间尺度是导致延迟泛化的原因。

    

    延迟泛化是指训练准确率在测试准确率之前很长时间就已接近完美的现象，此前通常在深度网络的深度与组合结构的背景下进行研究。在本文中，我们探究延迟泛化是否可以在更简单的设定中出现。我们证明，在线性可分数据上使用梯度下降和指数尾损失训练的线性模型中，延迟泛化现象也会出现。我们将有限时间内的超平面动力学刻画为三个阶段：总体阶段、由支持向量主导的遗忘阶段和泛化阶段。残差收敛至超出最大间隔方向是遗忘阶段开始的充分条件。在支持向量主导阶段，我们刻画了超平面动力学，其中动力学的时间尺度导致了延迟泛化。我们通过改变总体分布和支持向量分布的受控实验验证了我们的理论。

    arXiv:2602.08302v3 Announce Type: replace-cross  Abstract: Delayed generalization, where training accuracy becomes nearly perfect substantially before test accuracy, has often been studied in the context of the depth and compositional structure of deep networks. In this paper, we ask whether delayed generalization can occur in simpler settings. We show that it can arise in a linear model trained with gradient descent and an exponential-tail loss on linearly separable data. We characterize the finite-time hyperplane dynamics into three phases: a population phase, a support vector dominated unlearning phase and a generalization phase. The convergence of the residual beyond the maximum-margin direction is sufficient for the unlearning phase to begin. In the support vector dominated phase, we characterize the hyperplane dynamics. The time scale in these dynamics causes delayed generalization. We validate our theory through controlled experiments varying population and support-vector distri
    
[^452]: 有理转换器

    Rational Transductors

    [https://arxiv.org/abs/2602.07599](https://arxiv.org/abs/2602.07599)

    本文提出有理转换器这一双流架构，通过将源自加权有限自动机的矩阵值递归以深度有理注入的方式嵌入Transformer注意力机制，将Transformer的表达能力严格推广至可捕获所有正则语言及 $\NC^1$ 完全语言，从而解决其在序列逻辑与状态追踪上的局限。

    

    标准Transformer在语义建模方面表现出色，但在处理严格的序列逻辑与状态追踪方面存在困难。已有理论工作证明，自注意力机制的表达能力被限制在 $\AC^0$（硬注意力情形）或 $\TC^0$（软注意力情形）复杂度类内，而这些复杂度类通常无法在不借助中间思维链的情况下，在序列问题上实现稳健的长度泛化。在这项工作中，我们提出了有理转换器，这是一种双流架构，通过源自加权有限自动机（WFA）的矩阵值递归来增强Transformer。通过一种深度有理注入方案将有理状态信息注入注意力机制，我们的框架严格推广了Transformer的表达能力，使其能够捕获所有正则语言、$\NC^1$完全语言（例如 $S_5$ 的字问题），以及……

    arXiv:2602.07599v2 Announce Type: replace  Abstract: Standard Transformers excel at semantic modeling but struggle with   rigid sequential logic and state tracking. Theoretical work   establishes that self-attention is limited to $\AC^0$ (under hard   attention) or $\TC^0$ (under soft attention), complexity classes   that often fail to support robust length generalization on   sequential problems without intermediate chain-of-thought   \citep{hahn2020theoretical, hao2022formal, merrill2022saturated}. In this   work, we introduce \emph{Rational Transductors}, a dual-stream   architecture that augments the Transformer with a matrix-valued   recurrence derived from Weighted Finite Automata (WFA). By   injecting rational state information into the attention mechanism   via a \emph{Deep Rational Injection} scheme, our framework strictly   generalizes the expressive power of Transformers to capture all   Regular Languages, $\NC^1$-complete languages (such as the word   problem of $S_5$), and
    
[^453]: 深度时间序列预测需要核化矩平衡

    Deep Time-series Forecasting Needs Kernelized Moment Balancing

    [https://arxiv.org/abs/2602.00717](https://arxiv.org/abs/2602.00717)

    该论文提出 KMB-DF 方法，通过从再生核希尔伯特空间（RKHS）中自适应地选择最有信息量的平衡函数来实施核化矩匹配，克服了现有方法只能匹配少数预定义矩的局限，从而实现深度时间序列预测中预测分布与真实分布的充分平衡。

    

    深度时间序列预测可以被表述为一个分布平衡问题，其目标是对齐预测值与真实值的分布。根据 Imbens 准则，要平衡两个分布，需要它们关于任意平衡函数的一阶矩完全相同。我们证明，现有目标函数仅针对一个或两个预定义的平衡函数进行矩匹配，因而无法实现完全的分布平衡。为解决这一局限，我们提出基于核化矩平衡的直接预测方法。与现有目标函数不同，KMB-DF 能够从再生核希尔伯特空间（RKHS）中自适应地选择最具信息量的平衡函数，从而实现充分的分布平衡。我们推导出一个易于处理且可微的公式，使其能够基于经验样本进行估计，并无缝集成到基于梯度的训练流程中。在多个数据集上开展的大量实验（摘要原文在此处截断）……

    arXiv:2602.00717v2 Announce Type: replace  Abstract: Deep time-series forecasting can be formulated as a distribution balancing problem aimed at aligning the distribution of the forecasts and ground truths. According to Imbens' criterion, balancing two distributions requires that their first moments be identical with respect to any balancing function. We demonstrate that existing objectives enforce moment matching only for one or two predefined balancing functions, thus failing to achieve full distribution balance. To address this limitation, we propose direct forecasting with kernelized moment balancing (KMB-DF). Unlike existing objectives, KMB-DF adaptively selects the most informative balancing functions from a reproducing kernel Hilbert space (RKHS) to enforce sufficient distribution balancing. A tractable and differentiable formulation is derived to enable estimation from empirical samples and seamless integration into gradient-based training pipelines. Extensive experiments acros
    
[^454]: SWE-Spot：通过仓库中心化学习构建小型仓库专家模型

    SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning

    [https://arxiv.org/abs/2601.21649](https://arxiv.org/abs/2601.21649)

    本文提出仓库中心化学习（RCL）新范式，将静态代码库转化为交互式学习信号，训练出针对特定代码仓库深度专门化的小型语言模型SWE-Spot-4B，以弥补小型模型处理复杂陌生代码库时泛化能力不足的缺陷。

    

    在隐私敏感和资源受限环境中部署编码智能体的需求，推动了对高性能开源权重小型语言模型（SLM）的需求。然而，这些模型存在根本性的能力差距：与前沿大模型不同，它们缺乏在推理时应对复杂、陌生代码库的强泛化能力。我们发现当前流行的任务中心学习（TCL）方法——即在不同仓库间扩大任务暴露规模——无法解决这一局限。为此，我们提出仓库中心化学习（RCL），这是一种范式转变：在SLM训练中优先追求纵向的单仓库深度，而非横向的单任务广度。为实现这种深度，我们设计了由四个单元组成的仓库中心化经验，将静态代码库转化为交互式学习信号，用于训练SWE-Spot-4B——一系列作为仓库专门专家构建的高度紧凑模型。在受控对比实验中……

    arXiv:2601.21649v2 Announce Type: replace-cross  Abstract: The deployment of coding agents in privacy-sensitive and resource-constrained environments drives the demand for capable open-weight Small Language Models (SLMs). However, they suffer from a fundamental capability gap: unlike frontier large models, they lack the inference-time strong generalization to handle complicated, unfamiliar codebases. We identify that the prevailing Task-Centric Learning (TCL) approach, which scales exposure across disparate repositories, fails to address this limitation. In response, we propose Repository-Centric Learning (RCL), a shift that prioritizes vertical, single-repository depth over horizontal, single-task breadth for SLM training. To accomplish this depth, we design a four-unit Repository-Centric Experience, transforming static codebases into interactive learning signals, to train SWE-Spot-4B, a family of highly compact models built as repo-specialized experts. Under controlled comparisons wi
    
[^455]: 超越可持续发展目标（SDG）加速器：面向适应性可持续发展组合的定向证据

    Beyond Sustainable Development Goal (SDG) Accelerators: Directed Evidence for Adaptive Sustainability Portfolios

    [https://arxiv.org/abs/2601.20875](https://arxiv.org/abs/2601.20875)

    本研究基于114个国家2000至2024年的面板数据，结合面板格兰杰因果检验与局部投影方法，识别出84个定向SDG联系（40个协同、44个权衡），证明不存在普适的SDG加速器，并主张依据定向因果证据构建适应性的可持续发展政策组合。

    

    政府和发展伙伴常常寻求可持续发展目标（SDG）“加速器”，希望其进展能将收益传播至人类福祉、制度、资源和环境优先事项等各领域。这类主张需要关于方向、时机、稳健性和权衡的证据，而不仅仅是网络连通性。我们利用2000年至2024年间对114个国家进行年度观测的平衡面板数据，绘制了定向SDG相互依赖关系。由于目标序列具有持续性、趋势性和横截面依赖性，我们将针对一阶差分序列的Dumitrescu-Hurlin面板格兰杰非因果检验与覆盖272个定向目标配对的错误发现控制相结合，并采用带Driscoll-Kraay标准误的面板局部投影法分析31个基于理论推导的指标联系。校正后我们发现84个定向联系：40个协同效应和44个权衡关系。这一密集结构不支持存在普适性加速器。驱动-接收排名（原文在此处截断）

    arXiv:2601.20875v3 Announce Type: replace-cross  Abstract: Governments and development partners often seek Sustainable Development Goal (SDG) "accelerators" whose progress can propagate gains across human well-being, institutions, resources, and environmental priorities. Such claims require evidence on direction, timing, robustness, and trade-offs, not only network connectivity. We map directed SDG interdependence using a balanced panel of 114 countries observed annually from 2000 to 2024. Because the goal series are persistent, trending, and cross-sectionally dependent, we combine Dumitrescu-Hurlin panel Granger non-causality tests on first-differenced series with false-discovery control across 272 directed goal pairs, and panel local projections with Driscoll-Kraay standard errors for 31 theory-derived indicator linkages. We find 84 directed linkages after correction: 40 synergies and 44 trade-offs. This dense structure does not support a universal accelerator. Driver-receiver rankin
    
[^456]: 基于校准不确定性量化的GEDI生物量估计插值方法

    Interpolation of GEDI Biomass Estimates with Calibrated Uncertainty Quantification

    [https://arxiv.org/abs/2601.16834](https://arxiv.org/abs/2601.16834)

    该论文提出注意力神经过程（ANP）概率元学习架构，通过结合局部激光雷达观测集合与地理空间基础模型嵌入来插值GEDI稀疏生物量观测，能够根据景观异质性自适应调整不确定性，并在五种生物群系中实现了竞争性精度和良好的不确定性校准。

    

    NASA GEDI任务的全域生物量密度估计需要在异质性景观中对稀疏的激光雷达（LIDAR）观测进行插值。虽然随机森林和XGBoost被广泛使用，但它们仅基于卫星特征对每个位置进行独立预测，无法适应异质性景观中不同的预测难度。我们引入了注意力神经过程，这是一种概率元学习架构，它显式地使预测同时依赖于局部观测集合和地理空间基础模型嵌入。ANP学习到一个灵活的空间协方差近似器，在复杂景观中表现出更高的不确定性，而在均质区域中表现出较低的不确定性。我们在从热带亚马逊森林到北方、温带和高山生态系统的五种不同生物群系中验证了该方法，结果表明ANP在保持不确定性校准的同时实现了具有竞争力的精度。我们展示了一个可扩展的、经验校准的……（原文摘要在此处截断）

    arXiv:2601.16834v4 Announce Type: replace  Abstract: Wall-to-wall biomass density estimation from NASA's GEDI mission requires interpolating sparse LIDAR observations across heterogeneous landscapes. While Random Forest and XGBoost are widely used, they predict each location independently from satellite features, without adapting to the varying difficulty of heterogeneous landscapes. We introduce Attentive Neural Processes (ANPs), a probabilistic meta-learning architecture that explicitly conditions predictions on local observation sets as well as geospatial foundation model embeddings. ANPs learn a flexible spatial covariance approximator that is more uncertain in complex landscapes and less in homogeneous areas. We validate this approach across five distinct biomes ranging from tropical Amazonian forests to boreal, temperate, and alpine ecosystems, demonstrating that ANPs achieve competitive accuracy while maintaining uncertainty calibration. We demonstrate a scalable, empirically ca
    
[^457]: 大语言模型与贝叶斯优化的树状结构协同以实现高效CASH

    Tree-Structured Synergy of Large Language Models and Bayesian Optimization for Efficient CASH

    [https://arxiv.org/abs/2601.12355](https://arxiv.org/abs/2601.12355)

    该论文提出LB-MCTS框架，以蒙特卡洛树搜索作为共享的轨迹结构化状态，协同大语言模型的语义先验与贝叶斯优化，实现跨算法选择与算法内超参数空间的联合高效优化，从而更好地解决CASH问题。

    

    为降低机器学习的专业门槛，AutoML社区一直致力于研究CASH问题，即联合自动化算法选择与超参数调优。虽然贝叶斯优化（BO）等传统方法难以应对冷启动问题，但大语言模型（LLM）可以通过语义先验来缓解这些问题。然而，现有的基于LLM的优化器在高维、结构化的CASH空间中泛化能力较差。CASH要求在候选算法之间以及各算法特定的超参数空间内部同时进行协调的探索与利用。然而，现有方法往往依赖于扁平的优化历史，将不同算法的观测混在一起，掩盖了配置细化的轨迹。在本文中，我们提出LB-MCTS，这是一种轨迹结构化的优化框架，它使用蒙特卡洛树搜索（MCTS）树作为共享状态，用于算法选择、超参数细化以及BO-

    arXiv:2601.12355v3 Announce Type: replace  Abstract: To lower the expertise barrier in machine learning, the AutoML community has focused on the CASH problem, which jointly automates algorithm selection and hyperparameter tuning. While traditional methods like Bayesian Optimization (BO) struggle with cold-start issues, Large Language Models (LLMs) can mitigate these through semantic priors. However, existing LLM-based optimizers generalize poorly to high-dimensional, structured CASH spaces. CASH requires coordinated exploration and exploitation both across candidate algorithms and within their algorithm-specific hyperparameter spaces. Yet existing approaches often rely on flat optimization histories that mix observations across algorithms and obscure configuration-refinement trajectories. In this paper, we propose LB-MCTS, a trajectory-structured optimization framework that uses a Monte Carlo Tree Search tree as a shared state for algorithm selection, hyperparameter refinement, and BO-
    
[^458]: 心血管疾病中以影像为锚点的多组学研究：整合心脏影像、bulk转录组、单细胞与空间转录组学

    Imaging-anchored Multiomics in Cardiovascular Disease: Integrating Cardiac Imaging, Bulk, Single-cell, and Spatial Transcriptomics

    [https://arxiv.org/abs/2601.07871](https://arxiv.org/abs/2601.07871)

    本综述提出以心脏影像为锚点的多组学整合框架，将超声心动图、心脏MRI和CT所定义的心脏空间表型，与bulk、单细胞及空间转录组提供的细胞类型和位置特异的分子背景相融合，从而打破数据孤岛、实现心血管疾病的多模态联合分析。

    

    心血管疾病源于遗传风险、分子程序与组织尺度重塑之间的相互作用，这些过程在临床上通过影像学手段被观察。目前医疗系统常规产生大量心脏MRI、CT和超声心动图数据，以及bulk转录组、单细胞和空间转录组数据，然而这些数据仍在各自独立的分析流程中处理。本综述探讨了将心脏影像表型与转录组及空间分辨分子状态相联系的联合表示方法。本文采用以影像为锚点的视角，即由超声心动图、心脏MRI和CT定义心脏的空间表型，而bulk、单细胞和空间转录组则提供细胞类型特异与位置特异的分子背景。文中首先总结了这些模态的生物学与技术特性，并概述了各自相应的表示学习策略。多模态融合方法……（原文摘要在此处截断）

    arXiv:2601.07871v2 Announce Type: replace-cross  Abstract: Cardiovascular disease arises from interactions between inherited risk, molecular programmes, and tissue-scale remodelling that are observed clinically through imaging. Health systems now routinely generate large volumes of cardiac MRI, CT and echocardiography together with bulk, single-cell and spatial transcriptomics, yet these data are still analysed in separate pipelines. This review examines joint representations that link cardiac imaging phenotypes to transcriptomic and spatially resolved molecular states. An imaging-anchored perspective is adopted in which echocardiography, cardiac MRI and CT define a spatial phenotype of the heart, and bulk, single-cell and spatial transcriptomics provide cell-type- and location-specific molecular context. The biological and technical characteristics of these modalities are first summarised, and representation-learning strategies for each are outlined. Multimodal fusion approaches are r
    
[^459]: 远程蒸馏：将一万年模拟气候蒸馏进长时间步长的AI天气模型

    Long-Range Distillation: Distilling 10,000 Years of Simulated Climate into Long Timestep AI Weather Models

    [https://arxiv.org/abs/2512.22814](https://arxiv.org/abs/2512.22814)

    提出了一种“远程蒸馏”方法，利用自回归教师模型生成的一万年模拟气候合成数据，训练单步生成的长时间步长概率学生模型，从而解决AI长期天气预报中误差累积与再分析数据不足导致的过拟合问题。

    

    准确的长期天气预报对AI模型而言仍是一项重大挑战，原因在于误差会在自回归滚动预测中不断累积，且再分析训练数据集对支撑可预报性的慢变率模态的采样十分有限。大多数AI模型以及所有数值天气预报模型都是自回归式的，需要反复应用短时效预报才能达到次季节至季节（S2S）乃至季节性预报时效，这常常导致不稳定和校准失准。能够在单步中生成长期预报的长步长概率模型提供了一种替代方案，但仅基于40年的再分析记录进行训练会导致过拟合，这表明需要比现有数据多出几个数量级的数据量。我们提出了远程蒸馏方法：利用一个自回归“教师”模型生成的海量合成数据集，来训练一个长步长概率“学生”模型直接进行长期预报。

    arXiv:2512.22814v2 Announce Type: replace  Abstract: Accurate long-range weather forecasting remains a major challenge for AI models, both because errors accumulate over autoregressive rollouts and because reanalysis training datasets provide limited sampling of the slow modes of variability underpinning predictability. Most AI and all numerical weather prediction models are autoregressive, repeatedly applying short-lead forecasts to reach subseasonal-to-seasonal (S2S) or seasonal lead times, often causing instability and miscalibration. Long-timestep probabilistic models that generate long-range forecasts in a single step offer an alternative, but training on the 40-year reanalysis record leads to overfitting, suggesting orders of magnitude more data are required. We introduce long-range distillation, whereby a long-timestep probabilistic ``student'' model is trained to forecast directly at long-range using a huge synthetic dataset generated by an autoregressive ``teacher'' model. Usi
    
[^460]: 安全的不稳定性：随机种子与温度如何暴露大语言模型不一致的拒绝行为

    The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior

    [https://arxiv.org/abs/2512.12066](https://arxiv.org/abs/2512.12066)

    该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。

    

    当前大语言模型的安全评估依赖于单次测试，隐含地假设模型响应是确定性的，并能代表模型的安全对齐水平。我们通过研究安全拒绝决策在不同随机种子和温度设置下的稳定性来挑战这一假设。我们在20种采样配置（4种温度 × 5个随机种子）下，对来自三个系列的四个指令微调模型（Llama 3.1 8B、Qwen 2.5 7B、Qwen 3 8B、Gemma 3 12B）在876个有害提示词上进行了测试，发现18-28%的提示词表现出决策翻转——即模型在某些配置下拒绝回答，而在其他配置下则予以配合——具体比例因模型而异。我们的安全稳定性指数（SSI）显示，更高的温度会显著降低决策稳定性（Friedman卡方 = 396.81，p < 0.001），温度内平均SSI从温度0.0时的0.977下降至温度1.0时的0.942。我们在……（原文摘要在此处被截断）

    arXiv:2512.12066v3 Announce Type: replace-cross  Abstract: Current safety evaluations of large language models rely on single-shot testing, implicitly assuming that model responses are deterministic and representative of the model's safety alignment. We challenge this assumption by investigating the stability of safety refusal decisions across random seeds and temperature settings. Testing four instruction-tuned models from three families (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B) on 876 harmful prompts across 20 sampling configurations (4 temperatures x 5 seeds), we find that 18-28% of prompts exhibit decision flips--the model refuses in some configurations but complies in others--depending on the model. Our Safety Stability Index (SSI) reveals that higher temperatures significantly reduce decision stability (Friedman chi-squared = 396.81, p < 0.001), with mean within-temperature SSI dropping from 0.977 at temperature 0.0 to 0.942 at temperature 1.0. We validate findings acro
    
[^461]: 利用神经网络追踪大型化学反应网络与稀有事件

    Tracking large chemical reaction networks and rare events by neural networks

    [https://arxiv.org/abs/2512.10309](https://arxiv.org/abs/2512.10309)

    提出了一种基于自回归概率分布表示、二阶优化与增强采样策略的统一神经网络框架，可高效求解大型化学反应网络的化学主方程，并精确捕捉稀有事件与亚稳态跃迁动力学。

    

    化学反应网络被广泛用于模拟生物物理过程中的随机动力学。通过化学主方程对其进行精确的概率描述，在高维系统中会因状态空间的指数级增长而变得难以处理。当亚稳态跃迁和稀有事件主导动力学行为时，这一困难会进一步加剧。在此，我们提出了一种先进的神经网络方法，能够在统一框架内有效处理大型反应网络和稀有事件动力学。基于完整联合概率分布的自回归表示，我们通过二阶优化方法（包括自然梯度下降和含时变分原理）实现了高效训练。我们利用增强采样策略，能够准确解析控制稀有跃迁的低概率区域。我们在多稳态

    arXiv:2512.10309v3 Announce Type: replace-cross  Abstract: Chemical reaction networks are widely used to model stochastic dynamics in biophysical processes. Their exact probabilistic description via the chemical master equation becomes intractable for high-dimensional systems due to the exponential growth of the state space. This difficulty is further exacerbated when metastable transitions and rare events dominate the dynamics. Here, we present an advanced neural-network approach that effectively tackles large reaction networks and rare-event dynamics within a unified framework. Building on autoregressive representations of the full joint probability distribution, we achieve efficient training through second-order optimization methods, including natural gradient descent and the time-dependent variational principle. We leverage enhanced-sampling strategies that enable accurate resolution of low-probability regions governing rare transitions. We demonstrate the method on a multistable g
    
[^462]: 面向非平凡分类任务的多功能变分量子核框架

    A Versatile Variational Quantum Kernel Framework for Non-Trivial Classification

    [https://arxiv.org/abs/2511.10831](https://arxiv.org/abs/2511.10831)

    该研究提出了一个结合资源高效拟设与参数缩放技术的多功能变分量子核框架，在涵盖表格、图像、时间序列和图数据的八个真实高维数据集上取得了与经典RBF核相当的分类性能。

    

    量子核方法是量子机器学习中一个具有前景的分支，但其在不同数据类型和编码选择下的性能仍未被充分理解。为填补这一空白，我们开发了一个用于变分量子核的算法框架，该框架利用资源高效的拟设（ansätze）来处理复杂分类任务，并引入了一种参数缩放技术以加速收敛。我们在八个具有挑战性的真实世界高维数据集上对该框架进行了全面基准测试，这些数据集涵盖表格数据、图像、时间序列和图数据。结果表明，在经典模拟中，所提出的量子核相比标准经典核（如径向基函数（RBF）核）展现出具有竞争力的分类准确率。这项工作证明，经过合理设计的量子核可以作为多功能、高性能的工具，为量子增强的机器学习奠定基础。

    arXiv:2511.10831v3 Announce Type: replace  Abstract: Quantum kernel methods are a promising branch of quantum machine learning, but their per- formance across data types and encoding choices remains incompletely understood. To address this gap, we developed an algorithmic framework for variational quantum kernels utilizing resource- efficient ans\"atze for complex classification tasks and introduced a parameter scaling technique to accelerate convergence. We conducted a comprehensive benchmark of this framework on eight chal- lenging, real-world and high-dimensional datasets covering tabular, image, time series, and graph data. Our results show that the proposed quantum kernel demonstrated competitive classification accuracy compared to standard classical kernels in classical simulation, such as the radial basis function (RBF) kernel. This work demonstrates that properly designed quantum kernels can func- tion as versatile, high-performance tools, laying a foundation for quantum-enhanc
    
[^463]: 什么时候系统能从数据中被发现？发现需要混沌

    When is a System Discoverable from Data? Discovery Requires Chaos

    [https://arxiv.org/abs/2511.08860](https://arxiv.org/abs/2511.08860)

    混沌虽然与不可预测性相关，却往往是使动力系统控制方程能够从数据中被发现的关键——系统的域外泛化能力恰好在混沌出现时产生，而保守系统本质上无法仅凭数据被发现。

    

    深度学习革命推动了物理科学领域的重大人工智能进展，特别是在从观测数据中发现动力系统方面。然而，学习到的代理模型和符号发现模型的可靠性常常受到非唯一性的损害：模型可能完美拟合观测轨迹，却缺乏真正的预测能力。我们证明，尽管混沌与不可预测性相关联，但它往往正是使控制方程可被发现的关键。我们证明了经典洛伦兹系统在分析上是可发现的，并通过实验验证，对于许多系统而言，域外泛化能力恰好在混沌出现时产生。相比之下，保守系统在本质上无法仅凭数据被发现。基准测试中混沌系统的普遍存在可能无意中掩盖了这一局限性。这些发现有助于解释为什么人工智能擅长天气预报，却在许多工程应用中表现不佳。

    arXiv:2511.08860v3 Announce Type: replace-cross  Abstract: The deep learning revolution has enabled major AI advances in physical sciences, particularly in discovering dynamical systems from observational data. Yet the reliability of learned surrogates and symbolically discovered models is often undermined by non-uniqueness: models may fit observed trajectories perfectly, but lack genuine predictive power. We show that chaos, despite its association with unpredictability, is often what makes governing equations discoverable. We prove that the classical Lorenz system is analytically discoverable and verify experimentally that for many systems out-of-domain generalization emerges precisely at chaos onset. Conservative systems, by contrast, are inherently non-discoverable from data alone. The prevalence of chaotic systems in benchmarks may have inadvertently obscured this limitation. These findings help explain why AI excels at weather forecasting, yet struggles in many engineering applic
    
[^464]: 用于更好训练和评估知识图谱增强大语言模型的真值子图

    Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs

    [https://arxiv.org/abs/2511.04473](https://arxiv.org/abs/2511.04473)

    提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。

    

    从图结构知识库中检索信息是提高大语言模型事实性的一个有前景的方向。尽管已提出多种解决方案，但由于缺乏带有图检索真值目标的具有挑战性的问答数据集，方法之间的比较十分困难。我们提出了SynthKGQA，这是一个由大语言模型驱动的框架，可以从任何知识图谱生成高质量的知识图谱问答数据集，并提供知识图谱中用于推理问题的完整真值事实集合。我们展示了SynthKGQA生成的数据除了能够对知识图谱检索器进行更有信息量的基准测试外，还能用于训练更好的模型。我们将SynthKGQA应用于Wikidata，生成了GTSQA——一个旨在测试知识图谱检索器对未见过的图结构和关系类型的零样本泛化能力的新数据集，并在其上对流行的知识图谱增强大语言模型解决方案进行了基准测试。

    arXiv:2511.04473v3 Announce Type: replace-cross  Abstract: Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, an LLM-powered framework for generating high-quality Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over questions. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models.We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it.
    
[^465]: 一种Nesterov加速的拜占庭鲁棒联邦学习

    A Nesterov-Accelerated Byzantine-Robust Federated Learning

    [https://arxiv.org/abs/2511.02657](https://arxiv.org/abs/2511.02657)

    该论文提出了一种将Nesterov动量与拜占庭鲁棒聚合规则无缝结合的联邦学习算法Byrd-NAFL，在非凸光滑损失函数和宽松假设下建立了有限时间收敛保证，同时兼顾通信效率与对拜占庭攻击的抵抗能力。

    

    我们研究了鲁棒联邦学习问题，即一组工作节点在中央服务器的协调下协作训练一个共享模型，同时存在能够进行任意且潜在恶意行为的拜占庭攻击者。为了同时提高通信效率和对这类攻击者的抵御能力，我们提出了一种拜占庭鲁棒的Nesterov加速联邦学习算法。Byrd-NAFL将Nesterov动量与拜占庭鲁棒聚合规则无缝集成到联邦学习过程中，以实现对梯度污染的快速且安全的收敛。我们在非凸光滑损失函数下，并采用对聚合梯度较为宽松的假设，为Byrd-NAFL建立了有限时间收敛保证。大量数值实验验证了Byrd-NAFL的有效性，并证明了其在收敛性能方面相对于现有基准方法的优越性。

    arXiv:2511.02657v2 Announce Type: replace  Abstract: We investigate robust federated learning, where a group of workers collaboratively train a shared model under the orchestration of a central server in the presence of Byzantine adversaries capable of arbitrary and potentially malicious behaviors. To simultaneously enhance communication efficiency and resilience against such adversaries, we propose a Byzantine-resilient Nesterov-accelerated federated learning (Byrd-NAFL) algorithm. Byrd-NAFL seamlessly integrates Nesterov's momentum into the federated learning process alongside Byzantine-resilient aggregation rules to achieve fast and safe convergence against gradient corruption. We establish a finite-time convergence guarantee for Byrd-NAFL under non-convex and smooth loss functions with relaxed assumptions on the aggregated gradients. Extensive numerical experiments validate the effectiveness of Byrd-NAFL and demonstrate the superiority over existing benchmarks in terms of convergen
    
[^466]: 迈向能对自身计算进行推理的智能体

    Toward Agents That Reason About Their Computation

    [https://arxiv.org/abs/2510.22833](https://arxiv.org/abs/2510.22833)

    本文通过让强化学习智能体感知自身计算成本并自主控制计算使用时机，实现了在相同训练预算下于75%的游戏中性能更优、且平均计算量减少三倍的效果。

    

    虽然强化学习智能体在许多复杂任务中可以取得超越人类的性能，但随着其能力的提升，它们的计算效率通常并不会随之提高。相比之下，人类在任务上越熟练，所需的认知努力就越少。如果智能体在学习过程中能够对其计算进行推理，它们是否也能类似地减少计算占用？如果可以，我们就能拥有更节能的智能体，或者为规划等其他过程释放计算资源。在本文中，我们通过实验向智能体展示其计算成本，并赋予它们控制何时使用计算的能力。我们在街机学习环境上进行实验，结果表明在相同的训练计算预算下，能够对其计算进行推理的智能体在75%的游戏中表现更好。此外，这些智能体平均使用的计算量减少了三倍。我们对各个游戏进行了分析。

    arXiv:2510.22833v3 Announce Type: replace  Abstract: While reinforcement learning agents can achieve superhuman performance in many complex tasks, they typically do not become more computationally efficient as they improve. In contrast, humans gradually require less cognitive effort as they become more proficient at a task. If agents could reason about their compute as they learn, could they similarly reduce their computation footprint? If they could, we could have more energy efficient agents or free up compute cycles for other processes like planning. In this paper, we experiment with showing agents the cost of their computation and giving them the ability to control when they use compute. We conduct our experiments on the Arcade Learning Environment, and our results demonstrate that with the same training compute budget, agents that reason about their compute perform better on 75% of games. Furthermore, these agents use three times less compute on average. We analyze individual game
    
[^467]: 不要手持剪刀奔跑：剪枝会破坏VLA模型，但它们可以被恢复

    Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered

    [https://arxiv.org/abs/2510.08464](https://arxiv.org/abs/2510.08464)

    剪枝会严重破坏VLA模型并加剧安全违规，本文通过谱分析揭示了其权重结构脆弱性的根源，并提出GLUESTICK方法——通过在权重空间对密集模型与剪枝模型插值计算一次性校正项，在保留稀疏性优势的同时恢复模型大部分性能。

    

    视觉-语言-动作（VLA）模型是野外机器人领域一种极具前景的范式，但其部署受到边缘端严格的SWaP-C（尺寸、重量、功耗与成本）约束的瓶颈限制。量化、蒸馏和剪枝等压缩技术在LLM和VLM中已相当成熟，然而剪枝在VLA模型中仍特别缺乏探索。我们惊讶地发现，对VLA模型进行剪枝会导致急剧的性能下降以及安全违规的增加。通过谱分析，我们识别出VLA权重的结构特性，这有助于解释这种脆弱性。基于这一见解，我们提出了GLUESTICK，这是一种剪枝后恢复方法，能够在保留稀疏性优势的同时恢复原始模型的大部分能力。GLUESTICK通过在权重空间中对密集模型与剪枝模型进行插值，计算一个一次性的校正项。在推理时，每个被剪枝的层应用该校正项，以最小的开销恢复损失的性能。

    arXiv:2510.08464v2 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models are a promising paradigm for field robotics, but their deployment is bottlenecked by strict edge SWaP-C constraints. Compression techniques such as quantization, distillation, and pruning are well-established for LLMs and VLMs. Pruning, however, remains particularly underexplored for VLA models. We surprisingly find that pruning VLA models yields drastic performance degradation and increased safety violations. Through spectral analysis, we identify structural properties of VLA weights that help explain this fragility. Based on this insight, we introduce GLUESTICK, a post-pruning recovery method that restores much of the original model's capability while preserving sparsity benefits. GLUESTICK computes a one-time corrective term by interpolating between the dense and pruned models in weight space. At inference, each pruned layer applies this correction to recover lost performance with minimal 
    
[^468]: 基于上下文价值信息的上下文内强化学习

    Context Value Informed In-Context Reinforcement Learning

    [https://arxiv.org/abs/2509.23209](https://arxiv.org/abs/2509.23209)

    提出CV-ICRL方法，通过为上下文内策略引入显式的上下文价值估计，并将价值预测以价值token的形式写回上下文，实现TD式回报目标，从而降低策略更新的方差，使测试时对未见任务的适应更加稳定。

    

    上下文内强化学习（ICRL）已成为一种在测试时无需参数更新即可适应未见任务的有效范式。然而，现有的ICRL方法可能表现出脆弱且不稳定的适应能力，而这种适应背后的机制仍鲜为人知。我们提供了ICRL的策略梯度视角，并论证当测试时的rollout有限时，依赖轨迹级回报反馈可能导致高方差的更新。受actor-critic原则的启发，我们提出了基于上下文价值信息的ICRL（CV-ICRL），它为上下文内策略配备了一个显式的上下文价值：即在当前状态和累积交互上下文条件下的期望折扣回报。CV-ICRL训练一个价值头，并将其预测作为价值token写回上下文中，从而实现TD式的回报目标，以获得更低方差的策略更新。在Dark Room、Minigrid和Procgen测试平台上的实验表……

    arXiv:2509.23209v2 Announce Type: replace  Abstract: In-context reinforcement learning (ICRL) has emerged as an effective paradigm for test time adaptation to unseen tasks without parameter updates. However, existing ICRL methods can exhibit brittle and unstable adaptions, and the mechanisms underlying such adaptions remain poorly understood. We provide a policy-gradient view of ICRL and argue that relying on trajectory-level return feedback can lead to high-variance updates when test time rollout is limited. Motivated by the actor--critic principle, we propose Context Value Informed ICRL (CV-ICRL), which equips the in-context policy with an explicit context value: the expected discounted return conditioned on the current state and accumulated interaction context. CV-ICRL trains a value head and writes its prediction back into the context as a value token, enabling TD-style return targets for lower-variance policy updates. Experiments on the Dark Room, Minigrid, and Procgen testbeds sh
    
[^469]: 语言模型的前瞻性评估：从任务描述预测LLM基准测试分数

    Anticipatory Evaluation of Language Models: Predicting LLM Benchmark Scores from Descriptions

    [https://arxiv.org/abs/2509.20645](https://arxiv.org/abs/2509.20645)

    该论文提出“前瞻性评估”新范式和EVALCAST数据集（2,290对描述-性能数据），证明前沿LLM能够从自然语言任务描述中推断出基准测试的相对难度排序，但精确预测绝对性能分数仍然困难。

    

    现代机器学习研究日益面临评估瓶颈：构建基准测试和运行实验成本高昂，拖慢了迭代速度。我们探讨的问题是：在实验之前，仅凭对任务和实验设置的自然语言描述，能够对评估结果推断出多少信息。我们将这一设定称为“前瞻性评估”。我们提出了EVALCAST，一个基于文献构建的资源，包含来自1,519篇arXiv论文的2,290对“描述-性能”数据，其中包括由模型知识截止日期之后发表的论文组成的测试集。我们评估了两项任务：通过回归预测绝对性能，以及通过成对预测比较基准测试的难度。基于前沿LLM的实验表明，精确的回归预测仍然具有挑战性，而成对预测在匹配设置下能够恢复相对性能排序。在截止日期后论文和流式评估上取得相当的表现，说明所观察到的信号并非仅仅……（原文摘要在此截断）

    arXiv:2509.20645v4 Announce Type: replace  Abstract: Modern ML research increasingly faces an evaluation bottleneck: building benchmarks and running experiments is costly, slowing iteration. We ask how much can be inferred about evaluation outcomes before experiments, given only a natural-language description of the task and setup. We call this setting anticipatory evaluation. We introduce EVALCAST, a literature-grounded resource of 2,290 description-performance pairs from 1,519 arXiv papers, including a test set of papers published after the model's knowledge cutoff. We evaluate two tasks: predicting absolute performance through regression and comparing benchmark difficulty through pairwise prediction. Experiments with frontier LLMs show that accurate regression remains challenging, while pairwise predictions can recover relative performance ordering under matched settings. Comparable performance on post-cutoff and streaming evaluations suggests that the observed signal is not solely 
    
[^470]: Sig-DEG用于蒸馏：让扩散模型更快更轻量

    Sig-DEG for Distillation: Making Diffusion Models Faster and Lighter

    [https://arxiv.org/abs/2508.16939](https://arxiv.org/abs/2508.16939)

    本文提出Sig-DEG，利用部分签名与循环结构逼近随机微分方程的解来蒸馏预训练扩散模型，在保持有竞争力的样本质量的同时显著提升推理速度。

    

    扩散模型实现了最先进的生成性能，但在推理阶段的计算成本依然高昂，通常需要大量的离散化步骤。我们提出了Sig-DEG（基于签名的微分方程生成器），一种用于蒸馏预训练扩散模型的生成器。受高阶随机泰勒展开的启发，Sig-DEG将总结子区间上布朗运动信息的部分签名与循环结构相结合，以逼近随机微分方程的解，从而实现通用性。我们将蒸馏问题表述为监督学习，在粗时间网格上匹配细分辨率的教师轨迹。在推理阶段，Sig-DEG能够实现快速生成，因为部分签名可以从其已知分布中被精确模拟，无需细粒度的布朗路径。在图像和时间序列生成任务上的实验表明，该方法在样本质量具有竞争力的同时，推理速度可大幅提升。

    arXiv:2508.16939v2 Announce Type: replace  Abstract: Diffusion models achieve state-of-the-art generative performance but remain computationally expensive at inference, often requiring many discretization steps. We propose Sig-DEG (Signature-based Differential Equation Generator), a generator for distilling pre-trained diffusion models. Inspired by high-order stochastic Taylor expansions, Sig-DEG combines partial signatures that summarize Brownian motion over subintervals with a recurrent structure to approximate stochastic differential equation solutions to achieve universality. We formulate distillation as supervised learning, matching fine-resolution teacher trajectories on a coarse time grid. At inference, Sig-DEG enables fast generation, as the partial signatures can be simulated exactly from their known distributions without requiring fine-grained Brownian paths. Experiments on image and time-series generation demonstrate competitive sample quality with inference speedups of up t
    
[^471]: 针对光滑区域未知的非光滑问题的线性收敛算法

    Linearly Convergent Algorithms for Nonsmooth Problems with Unknown Smooth Regions

    [https://arxiv.org/abs/2507.19465](https://arxiv.org/abs/2507.19465)

    本文首次证明bundle-level类方法可在光滑区域未知的情况下，对满足二次增长条件的凸分片光滑函数实现全局线性收敛，并将该方法扩展至弱凸分片光滑问题，同时提出了无需任何问题参数的可验证精确终止准则。

    

    我们为优化分片光滑（PWS）函数开发了高效算法，其中其光滑区域对算法而言是未知的。对于满足二次增长（QG）条件的凸分片光滑函数，我们证明了bundle-level（BL）类方法可以实现全局线性收敛——据我们所知，这是该问题类别上任何算法的首个此类结果。我们将该方法扩展到弱凸分片光滑问题，实现了与光滑非凸优化相同量级的复杂度。此外，我们为分片光滑优化引入了一个可验证且精确的终止准则。与光滑优化中的梯度范数类似，该最优性证书在QG条件下紧密刻画了最优性间隙，并且无需知道任何问题参数即可进行评估。我们为该证书开发了一个搜索子程序，并将其嵌入到“猜测-检验”框架中。

    arXiv:2507.19465v2 Announce Type: replace-cross  Abstract: We develop efficient algorithms for optimizing piecewise smooth (PWS) functions when their smooth regions are \emph{unknown} to the algorithm. For convex PWS functions satisfying a quadratic growth (QG) condition, we demonstrate that the bundle-level (BL)-type method~\cite{lemarechal1995new} achieves global linear convergence---to our knowledge, the first such result for any algorithm for this problem class. We extend the method to solve weakly convex PWS problems, achieving the same order of complexity as in smooth non-convex optimization. Furthermore, we introduce a verifiable and accurate termination criterion for PWS optimization. Like the gradient norm in smooth optimization, this certificate tightly characterizes the optimality gap under the QG condition and can be evaluated without knowledge of any problem parameters. We develop a search subroutine for this certificate and embed it within a guess-and-check framework, res
    
[^472]: 《无形的缰绳：为什么RLVR可能或可能无法摆脱其起源》

    The Invisible Leash: Why RLVR May or May Not Escape Its Origin

    [https://arxiv.org/abs/2507.14843](https://arxiv.org/abs/2507.14843)

    该论文提出“经验支持动力学”框架，逐题追踪RLVR对已验证解的保持、扩展与收缩，发现在给定采样预算下支持收缩超过扩展，表明RLVR可能并未真正扩展模型推理边界，而主要是放大了基座模型已知的高奖励输出。

    

    arXiv:2507.14843v5 公告类型：replace-cross。摘要：近期的研究进展突出了可验证奖励强化学习（RLVR）作为增强大语言模型能力的一种有前景的方法。然而，目前仍不清楚当前的RLVR实践究竟是真正扩展了模型的推理边界，还是主要放大了基座模型已经掌握的高奖励输出。我们引入了“经验支持动力学”这一框架，用以量化RLVR如何保持、扩展和收缩模型对已验证解决方案的访问能力。该框架被应用于多个模型、推理任务和训练检查点，通过逐个问题地匹配已验证的成功案例，揭示出总体准确率所可能掩盖的变化。在所评估的模型级基准汇总中，观察到的支持收缩超过了其扩展：在给定的采样预算下，失去已验证成功的问题数量多于获得已验证成功的问题数量。这些损失与pass@1的提升同时发生。一些RLVR模型增加了平均token级熵……

    arXiv:2507.14843v5 Announce Type: replace-cross  Abstract: Recent advances highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising approach to enhancing LLM capabilities. However, it remains unclear whether the current practice of RLVR truly expands models' reasoning boundary or mainly amplifies high-reward outputs that base models already know. We introduce empirical support dynamics, a framework that quantifies how RLVR preserves, expands, and shrinks access to verified solutions. Applied across models, reasoning tasks, and training checkpoints, it matches verified successes question by question to reveal changes that aggregate accuracy can conceal. Across the evaluated model-level benchmark aggregates, \textit{the shrinkage of observed support outweighs its expansion}: more questions lose a verified success than gain one at the stated sampling budgets. These losses occur alongside gains in \texttt{pass@1}. Some RLVR models increase average token-level entropy 
    
[^473]: 状态空间模型自然产生时间细胞与振荡行为，并可扩展至抽象认知功能

    State Space Models Naturally Produce Time Cell and Oscillatory Behaviors and Scale to Abstract Cognitive Functions

    [https://arxiv.org/abs/2507.13638](https://arxiv.org/abs/2507.13638)

    该论文提出状态空间模型凭借结构化的多时间尺度复数旋转基，无需训练即可自然涌现时间细胞和振荡等神经时间动态，并可通过学习扩展至抽象认知功能，为连接微观神经回路与认知功能提供了数学上充分、计算上合理的框架。

    

    现代神经科学的一个重大挑战是弥合微观神经回路与认知功能之间的鸿沟。尽管人们对神经元的连接方式和生物物理特性已有大量了解，但低层次现象如何产生抽象行为在很大程度上仍未解决。在此，我们提出基于状态空间模型（State Space Models）的数学框架，为产生类似时间细胞、斜坡活动和振荡响应的反应特性提供了数学充分性与计算合理性。我们的结果表明，这些行为可以从一种结构化的多时间尺度复数旋转基中涌现出来，该基实现了优化的历史压缩，甚至在训练之前就能自然产生结构化的时间动态，这与最近在生物回路中的发现相呼应。学习过程则利用并适应这种预先存在的时间基来完成特定任务。我们展示了该模型可以轻松扩展至抽象认知功能。

    arXiv:2507.13638v4 Announce Type: replace-cross  Abstract: A grand challenge in modern neuroscience is bridging the gap between the microscale neural circuits and cognitive functions. While extensive knowledge exists about neuronal connectivity and biophysics, how low-level phenomena give rise to abstract behaviors remains largely unresolved. Here, we propose that a mathematical framework based on State Space Models offers mathematical sufficiency and computational plausibility to generate response profiles that resemble time cells, ramping activity, and oscillatory responses. Our results suggest that such behaviors can emerge from a structured multi-timescale complex rotational basis, which enables optimized history compression and naturally generates structured temporal dynamics even prior to training, mirroring recent findings in biological circuits. Learning exploits and adapts to the pre-existing temporal basis for specific tasks. We show that the model can readily scale to abstra
    
[^474]: 儿科超声心动图实验室资源分配的基于规则的策略与强化学习策略：一项仿真研究

    Rule-based and reinforcement-learning policies for resource allocation in a pediatric echo lab: a simulation study

    [https://arxiv.org/abs/2506.06297](https://arxiv.org/abs/2506.06297)

    该研究基于斯坦福儿童医院一周的真实运营数据，构建了儿科超声心动图实验室的离散事件随机仿真模型并通过开源Gymnasium接口提供，为基于规则和强化学习的资源分配策略优化提供了可复现的实验平台。

    

    在医院环境中安排超声心动图检查面临重大挑战，其原因包括患者爽约、到达时间多变、检查时长异质等非确定性因素，以及胎儿与非胎儿患者流之间不对称的资源约束。胎儿病例涉及孕期进行的专门超声检查，而非胎儿病例对应常规诊断性超声心动图服务。利用斯坦福大学Lucile Packard儿童医院超声心动图实验室一周（184名患者）的运营数据，我们估计了患者爽约概率，并推导出到达时间和检查时长的经验分布，进而使用SimPy构建了一个离散事件随机仿真模型，并通过开源的Gymnasium接口对外提供（https://github.com/desResLab/echo_rl）。作为策略优化的基线，我们系统地……（原文摘要至此截断）

    arXiv:2506.06297v2 Announce Type: replace  Abstract: Scheduling echocardiographic examinations in a hospital setting presents significant challenges due to non-deterministic factors such as patient no-shows, variable arrival times, and heterogeneous exam durations, as well as asymmetric resource constraints between fetal and non-fetal patient streams. Fetal cases involve specialized ultrasound examinations performed during pregnancy, whereas non-fetal cases correspond to general diagnostic echocardiography services. Using one week (184 patients) of operational data from the Echo Laboratory at Stanford University's Lucile Packard Children's Hospital, we estimate patient no-show probabilities and derive empirical distributions of arrival times and exam durations, and use them to build a discrete-event stochastic simulation model using SimPy, exposed through an open-source Gymnasium interface (https://github.com/desResLab/echo_rl). As a baseline for policy optimization, we systematically 
    
[^475]: 基于分数训练的能量文本转语音模型

    Score-Based Training for Energy-Based TTS Models

    [https://arxiv.org/abs/2505.13771](https://arxiv.org/abs/2505.13771)

    本文提出了一种新的分数学习准则，使能量文本转语音模型的分数更适合推理阶段的一阶优化方案，克服了NCE和SSM忽略对数似然函数形式的局限性。

    

    噪声对比估计（NCE）是一种训练具有难以处理的归一化项的能量模型（EBM）的流行方法。NCE的核心思想是通过比较参考样本和噪声样本的非归一化对数似然来进行学习，从而避免显式计算归一化项。然而，NCE严重依赖于噪声样本的质量。最近，切片分数匹配（SSM）因与其密切相关的扩散模型（DM）而得到广泛推广。与NCE不同，SSM通过学习对数似然在随机选择方向上投影的分布，来学习对数似然的梯度（即分数）。然而，NCE和SSM都忽略了对数似然函数的形式，考虑到能量模型和扩散模型在推理过程中使用一阶优化，这是一个问题。本文提出了一种新的准则，可以学习更适合一阶方案的分数。实验对比了这些用于训练能量模型的方法。

    arXiv:2505.13771v2 Announce Type: replace-cross  Abstract: Noise contrastive estimation (NCE) is a popular method for training energy-based models (EBM) with intractable normalisation terms. The key idea of NCE is to learn by comparing unnormalised log-likelihoods of the reference and noisy samples, thus avoiding explicitly computing normalisation terms. However, NCE critically relies on the quality of noisy samples. Recently, sliced score matching (SSM) has been popularised by closely related diffusion models (DM). Unlike NCE, SSM learns a gradient of log-likelihood, or score, by learning distribution of its projections on randomly chosen directions. However, both NCE and SSM disregard the form of log-likelihood function, which is problematic given that EBMs and DMs make use of first-order optimisation during inference. This paper proposes a new criterion that learns scores more suitable for first-order schemes. Experiments contrasts these approaches for training EBMs.
    
[^476]: 激励贡献并同时学习参数：面向策略性数据拥有者的联邦学习

    Incentivize Contribution and Learn Parameters Too: Federated Learning with Strategic Data Owners

    [https://arxiv.org/abs/2505.12010](https://arxiv.org/abs/2505.12010)

    提出两种联邦学习机制：UPBReD 以几何速率联合收敛到贡献博弈的纳什均衡与福利最大化模型，而 2P-UPBReD 通过预算平衡的转移支付使完全贡献成为客户端的占优策略，从而实现最优模型训练。

    

    在联邦学习（FL）中，一个中心根据客户端在其私有数据上计算得到的更新来训练一个共享模型。收集数据并在其上进行训练是有代价的，因此策略性的客户端可能只贡献很少的数据，从而搭其他人训练成果的便车。我们研究了能够同时激励数据贡献并学习模型的机制。我们的第一个机制 UPBReD 联合更新客户端的贡献和模型，并以几何速率收敛到贡献博弈的纳什均衡，同时得到针对这些贡献的福利最大化模型。由于该均衡可能存在贡献不足的问题，我们的第二个机制 2P-UPBReD 引入了预算平衡的转移支付，使完全贡献成为占优策略，此后训练即收敛到最优模型；我们还给出了每个客户端都愿意参与的条件。两种机制都信任客户端对其所贡献数据的报告。

    arXiv:2505.12010v5 Announce Type: replace-cross  Abstract: In federated learning (FL), a center trains a shared model from updates that clients compute on their private data. Collecting and training on data is costly, so a strategic client may contribute little data and free-ride on the model trained by the others. We study mechanisms that incentivize data contribution and learn the model at the same time. Our first mechanism, UPBReD, updates the clients ' contributions and the model jointly, and converges at a geometric rate to a Nash equilibrium of the contribution game together with the welfare-maximizing model for those contributions. This equilibrium can involve under-contribution, so our second mechanism, 2P-UPBReD, adds budget-balanced transfers that make full contribution a dominant strategy, after which training converges to the optimal model; we also give conditions under which every client prefers to participate. Both mechanisms trust what clients report about the data they 
    
[^477]: 可信人工智能必须考虑各维度间的相互作用

    Trustworthy AI Must Account for Interactions

    [https://arxiv.org/abs/2504.07170](https://arxiv.org/abs/2504.07170)

    可信人工智能的各维度（如公平性、隐私、鲁棒性等）之间存在负面相互作用，例如差分隐私可能放大偏见损害公平性，因此研究必须放弃孤立改进单一方面做法，转而采用同时兼顾所有维度的整体视角。

    

    可信人工智能涵盖了将AI系统与人类价值观对齐的许多理想目标，包括公平性、隐私性、鲁棒性、可解释性和不确定性量化。可信人工智能研究的最终目标是同时实现所有这些方面。然而，提升某一方面的努力往往会引入意想不到的权衡，从而对其他方面产生负面影响。在这篇立场论文中，我们回顾了五个方面的代表性方法，并系统地考察了每一对维度组合，详细阐述了可能出现的负面相互作用。例如，在模型训练中应用差分隐私可能会放大偏见，从而损害公平性。基于这些发现，我们的立场是：目前孤立地改进一两个方面研究实践是不够的。相反，可信人工智能研究必须考虑各方面之间的相互作用，并同时采取覆盖所有相关维度的整体视角。我们提出了相应的建议。

    arXiv:2504.07170v3 Announce Type: replace  Abstract: Trustworthy AI encompasses many aspirational aspects for aligning AI systems with human values, including fairness, privacy, robustness, explainability, and uncertainty quantification. The ultimate goal of Trustworthy AI research is to achieve all aspects simultaneously. However, efforts to enhance one aspect often introduce unintended trade-offs that negatively impact others. In this position paper, we review notable approaches to five aspects and systematically consider every pair, detailing the negative interactions that can arise. For example, applying differential privacy to model training can amplify biases, undermining fairness. Drawing on these findings, we take the position that current research practices of improving one or two aspects in isolation are insufficient. Instead, research on Trustworthy AI must account for interactions between aspects and adopt a holistic view across all relevant axes at once. We offer recommend
    
[^478]: 临床时间序列的谱结构学习

    Spectral structure learning for clinical time series

    [https://arxiv.org/abs/2502.11680](https://arxiv.org/abs/2502.11680)

    该论文提出了一种基于多任务平稳高斯过程 StructGP 和改进 NOTEARS 算法的谱结构学习方法，能够从不规则采样的临床时间序列中有效学习时间序列之间的有向条件依赖关系图，并取得了高召回率和高精确率。

    

    我们开发并评估了一种用于临床时间序列的结构学习算法。临床时间序列是在多名患者中观察到的多变量时间序列，且采样不规则，这对现有的结构学习算法提出了挑战。我们假设这些时间序列是 StructGP 的实现，StructGP 是一种 k 维多输出或多任务平稳高斯过程（GP），独立的患者共享相同的协方差函数。StructGP 将时间序列之间的有序条件关系编码在一个有向无环图中。我们实现了一种改进的 NOTEARS 算法，该算法基于可微的无环性定义，通过求解一系列连续优化问题来恢复图结构。仿真结果表明，在平均度最高为 3、任务数最多为 20 的条件下，我们在恢复有向边时达到了 0.93 的中位数召回率 [IQR, 0.86, 0.97]，同时保持了 0.71 的中位数精确率 [0.57-0.84]。我们进一步表明……

    arXiv:2502.11680v2 Announce Type: replace  Abstract: We develop and evaluate a structure learning algorithm for clinical time series. Clinical time series are multivariate time series observed in multiple patients and irregularly sampled, challenging existing structure learning algorithms. We assume that our times series are realizations of StructGP, a k-dimensional multi-output or multi-task stationary Gaussian process (GP), with independent patients sharing the same covariance function. StructGP encodes ordered conditional relations between time series, represented in a directed acyclic graph. We implement an adapted NOTEARS algorithm, which based on a differentiable definition of acyclicity, recovers the graph by solving a series of continuous optimization problems. Simulation results show that up to mean degree 3 and 20 tasks, we reach a median recall of 0.93% [IQR, 0.86, 0.97] while keeping a median precision of 0.71% [0.57-0.84], for recovering directed edges. We further show tha
    
[^479]: 忘掉它吧：无需重放或辅助目标即可缓解遗忘

    Forget About It: Mitigating Forgetting Without Replay or Auxiliary Objectives

    [https://arxiv.org/abs/2501.05559](https://arxiv.org/abs/2501.05559)

    本文提出序贯微调平均法（SFA），证明仅依靠微调过程中简单的周期性参数插值，无需重放或辅助目标即可有效缓解灾难性遗忘，并在 Qwen3-14B 适配器持续学习中以最多节省 36.3% 训练时间的优势超越了重放和自适应合并等复杂方法。

    

    我们是否把缓解灾难性遗忘这件事弄得比实际需要的更复杂了？持续适应往往会导致模型在早期任务上的性能下降，这促使人们提出了重放缓冲区、辅助目标和自适应合并等复杂机制。我们研究了一种更简单的替代方案——仅在微调过程中进行周期性参数插值——能独立走多远。我们提出了序贯微调与平均法，并与最初为优化稳定性而设计的无重放 Lookahead 方法一同评估，发现了令人瞩目的结果。在 Qwen3-14B 适配器持续学习中，这两种方法的固定配置在平均最终准确率上优于所测试的重放和自适应合并配置，同时最多可减少 36.3% 的任务训练时间。此外，在语言和视觉任务中，我们的结果可与更复杂的持续学习方法相媲美。这些发现共同表明，简单的训练中插值方法值得认真考虑。

    arXiv:2501.05559v2 Announce Type: replace  Abstract: Have we made mitigating catastrophic forgetting more complicated than it needs to be? Continual adaptation often degrades performance on earlier tasks, motivating rehearsal buffers, auxiliary objectives, and adaptive merging procedures. We investigate how far a simpler alternative-periodic parameter interpolation during fine-tuning-can go on its own. We introduce Sequential Fine-tuning with Averaging (SFA) and evaluate it alongside replay-free Lookahead, originally designed for optimization stability to find compelling results. In Qwen3-14B adapter continual learning, fixed configurations of both methods outperform the tested replay and adaptive-merging configurations in mean final accuracy using up to 36.3% less task training time. Furthermore, across language and vision tasks, our results compete with more complex continual-learning methods. Together, these findings suggest that simple in-training interpolation deserves considerati
    
[^480]: GLL：一种面向神经网络的可微图学习层

    GLL: A Differentiable Graph Learning Layer for Neural Networks

    [https://arxiv.org/abs/2412.08016](https://arxiv.org/abs/2412.08016)

    该论文通过伴随方法推导出反向传播方程，提出了可微图学习层GLL，实现了图学习算法与神经网络的端到端精确集成，从而在分类任务中利用样本间的关系信息。

    

    标准的用于分类任务的深度学习架构通过投影头和softmax激活函数生成标签预测。尽管这些方法取得了成功，但它们未能利用样本之间的关系信息来生成标签预测。在近期的研究中，基于图的学习技术（即拉普拉斯学习）已被启发式地与神经网络结合，用于有监督和半监督学习（SSL）任务。然而，先前的工作要么对损失函数关于图学习算法的梯度进行近似，要么将这两个过程解耦处理，未能实现与神经网络的端到端集成。在这项工作中，我们通过伴随方法推导了反向传播方程，从而能够将一类通用的图学习层引入神经网络。所得方法不同于图神经网络，使我们能够精确地集成相似度图构建与...

    arXiv:2412.08016v3 Announce Type: replace  Abstract: Standard deep learning architectures used for classification generate label predictions with a projection head and softmax activation function. Although successful, these methods fail to leverage the relational information between samples for generating label predictions. In recent works, graph-based learning techniques, namely Laplace learning, have been heuristically combined with neural networks for both supervised and semi-supervised learning (SSL) tasks. However, prior works approximate the gradient of the loss function with respect to the graph learning algorithm or decouple the processes; end-to-end integration with neural networks is not achieved. In this work, we derive backpropagation equations, via the adjoint method, for inclusion of a general family of graph learning layers into a neural network. The resulting method, distinct from graph neural networks, allows us to precisely integrate similarity graph construction and 
    
[^481]: 噪声感知的差分隐私变分推断

    Noise-Aware Differentially Private Variational Inference

    [https://arxiv.org/abs/2410.19371](https://arxiv.org/abs/2410.19371)

    提出了一种基于随机梯度变分推断的噪声感知差分隐私近似贝叶斯推断新方法，突破了以往方法仅适用于简单概率模型的局限，可应用于高维和非共轭模型，并提出了一种更准确的噪声感知后验评估方法。

    

    差分隐私（DP）为统计推断提供了强大的隐私保障，但这可能导致下游应用中出现不可靠的结果和偏差。虽然已有若干噪声感知方法被提出，可以将DP扰动整合到推断过程中，但这些方法仅限于特定类型的简单概率模型。在本工作中，我们提出了一种基于随机梯度变分推断的新型噪声感知近似贝叶斯推断方法，该方法还可应用于高维和非共轭模型。此外，我们还提出了一种更准确的噪声感知后验评估方法。实验表明，在现有方法适用的领域内，我们的推断方法具有与现有方法相似的性能。在此领域之外，我们在高维贝叶斯线性回归中获得了准确的覆盖率，并在使用UCI Adult数据集的贝叶斯逻辑回归中获得了良好校准的预测概率。

    arXiv:2410.19371v4 Announce Type: replace-cross  Abstract: Differential privacy (DP) provides robust privacy guarantees for statistical inference, but this can lead to unreliable results and biases in downstream applications. While several noise-aware approaches have been proposed which integrate DP perturbation into the inference, they are limited to specific types of simple probabilistic models. In this work, we propose a novel method for noise-aware approximate Bayesian inference based on stochastic gradient variational inference which can also be applied to high-dimensional and non-conjugate models. We also propose a more accurate evaluation method for noise-aware posteriors. Empirically, our inference method has similar performance to existing methods in the domain where they are applicable. Outside this domain, we obtain accurate coverages on high-dimensional Bayesian linear regression and well-calibrated predictive probabilities on Bayesian logistic regression with the UCI Adult
    
[^482]: 机器人操作中基于贝叶斯模型的强化学习中的主动探索

    Active Exploration in Bayesian Model-based Reinforcement Learning for Robot Manipulation

    [https://arxiv.org/abs/2404.01867](https://arxiv.org/abs/2404.01867)

    该论文提出了在机器人操作中基于贝叶斯模型的强化学习中通过主动探索改善模型质量和保持数据效率的方法。

    

    有效解决机器人操作等复杂环境中的多个任务仍然是机器人技术中的一个持续挑战，也是数据驱动解决方案（如强化学习（RL））的一个机会。模型驱动的RL通过构建机器人的动态模型，实现数据重用和在具有相同机器人和类似环境的任务之间进行迁移学习。此外，机器人学中的数据收集成本很高，我们必须依赖数据高效的方法，如基于模型的RL，在基于学习模型的廉价仿真中进行大部分策略学习。因此，模型的质量对于后续任务的性能至关重要。在这项工作中，我们致力于通过在初始探索阶段执行动态模型的主动学习，基于最大化信息收集，以提高模型的质量并保持数据效率。

    arXiv:2404.01867v1 Announce Type: cross  Abstract: Efficiently tackling multiple tasks within complex environment, such as those found in robot manipulation, remains an ongoing challenge in robotics and an opportunity for data-driven solutions, such as reinforcement learning (RL). Model-based RL, by building a dynamic model of the robot, enables data reuse and transfer learning between tasks with the same robot and similar environment. Furthermore, data gathering in robotics is expensive and we must rely on data efficient approaches such as model-based RL, where policy learning is mostly conducted on cheaper simulations based on the learned model. Therefore, the quality of the model is fundamental for the performance of the posterior tasks. In this work, we focus on improving the quality of the model and maintaining the data efficiency by performing active learning of the dynamic model during a preliminary exploration phase based on maximize information gathering. We employ Bayesian ne
    
[^483]: 贝尔曼符合推断：时间序列预测中预测区间的校准

    Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series

    [https://arxiv.org/abs/2402.05203](https://arxiv.org/abs/2402.05203)

    贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。

    

    我们引入了贝尔曼符合推断（BCI），这是一个围绕任何时间序列预测模型的框架，可以提供校准的预测区间。与现有方法不同，BCI能够利用多步预测，并通过在每个时间步骤上解决一维随机控制问题（SCP）来显式优化平均区间长度。特别地，我们使用动态规划算法来找到SCP的最优策略。我们证明了在任意分布转换和时间依赖性下，BCI能够实现长期覆盖，即使多步预测较差。我们在实证中发现，与现有方法相比，BCI避免了无信息区间（长度无限）的生成，并在波动率预测问题上生成了明显更短的预测区间。

    We introduce Bellman Conformal Inference (BCI), a framework that wraps around any time series forecasting models and provides calibrated prediction intervals. Unlike the existing methods, BCI is able to leverage multi-step ahead forecasts and explicitly optimize the average interval lengths by solving a one-dimensional stochastic control problem (SCP) at each time step. In particular, we use the dynamic programming algorithm to find the optimal policy for the SCP. We prove that BCI achieves long-term coverage under arbitrary distribution shifts and temporal dependence, even with poor multi-step ahead forecasts. We find empirically that BCI avoids uninformative intervals that have infinite lengths and generates substantially shorter prediction intervals on volatility forecasting problems when compared with existing methods.
    
[^484]: 超越规模：梯度如何塑造大语言模型中的剪枝决策

    Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models

    [https://arxiv.org/abs/2311.04902](https://arxiv.org/abs/2311.04902)

    提出GBLM-Pruner，一种利用泰勒展开一阶项、基于少量校准样本归一化梯度的免训练剪枝方法，在大语言模型剪枝中显著优于SparseGPT和Wanda等现有方法。

    

    arXiv:2311.04902v3 公告类型：替换 摘要：拥有数十亿参数的大型语言模型（LLMs）是网络剪枝的主要目标，剪枝旨在在不损害性能的前提下移除部分模型权重。先前的方法如幅度剪枝、SparseGPT和Wanda要么仅关注权重，要么将权重与激活值相结合以实现稀疏性。然而，这些方法忽略了从预训练LLMs中获取的蕴含丰富信息的梯度。在本文中，我们提出了一种新颖的以稀疏性为中心的预训练LLMs剪枝方法，称为基于梯度的语言模型剪枝器（GBLM-Pruner）。GBLM-Pruner利用泰勒展开的一阶项，以无需训练的方式运行，通过利用少量校准样本的适当归一化梯度来确定剪枝度量，并在多个基准测试中大幅超越SparseGPT和Wanda等竞争方法。有趣的是，通过引入梯度，我们方法的非结构化剪枝倾向于重新……

    arXiv:2311.04902v3 Announce Type: replace  Abstract: Large Language Models (LLMs) with billions of parameters are prime targets for network pruning, which removes some model weights without hurting performance. Prior approaches such as magnitude pruning, SparseGPT, and Wanda either concentrated solely on weights or integrated weights with activations for sparsity. However, they overlooked the informative gradients derived from pretrained LLMs. In this paper, we present a novel sparsity-centric pruning method for pretrained LLMs, termed Gradient-based Language Model Pruner (GBLM-Pruner). GBLM-Pruner leverages the first-order term of the Taylor expansion, operating in a training-free manner by harnessing properly normalized gradients from a few calibration samples to determine the pruning metric, and substantially outperforms competitive counterparts like SparseGPT and Wanda in multiple benchmarks. Intriguingly, by incorporating gradients, unstructured pruning with our method tends to re
    
[^485]: 基于连续随机不可逆网络的不连续流形数据生成

    Generation of data on discontinuous manifolds via continuous stochastic non-invertible networks

    [https://arxiv.org/abs/2112.09646](https://arxiv.org/abs/2112.09646)

    该论文提出了一种新框架：利用预训练对比编码器获得聚类潜空间，并为每个代表单峰子流形的聚类训练专用低复杂度网络，从而有效生成不连续分布的数据，克服了VAE和GAN等传统方法在此类任务上的局限。

    

    对于大多数已知框架（如生成自编码器和生成对抗网络）而言，生成不连续分布是一项困难的任务。生成式不可逆模型无法准确生成此类分布，需要较长的训练时间，且经常出现模式崩溃（mode collapse）问题。变分自编码器（VAE）基于保持潜空间为高斯分布以便于简单采样的思想，虽然能够实现准确的重构，但在生成任务上存在显著的局限性。在本工作中，我们不再试图保持潜空间为高斯分布，而是使用预训练的对比编码器来获得聚类的潜空间。然后，对于代表单峰子流形的每个聚类，我们训练一个专用的低复杂度网络，从高斯分布生成该子流形。所提出的框架基于互信息的 信息论公式（摘要在此处截断）。

    arXiv:2112.09646v2 Announce Type: replace  Abstract: The generation of discontinuous distributions is a difficult task for most known frameworks such as generative autoencoders and generative adversarial networks. Generative non-invertible models are unable to accurately generate such distributions, require long training and often are subject to mode collapse. Variational autoencoders (VAEs), which are based on the idea of keeping the latent space to be Gaussian for the sake of a simple sampling, allow an accurate reconstruction, while they experience significant limitations at generation task. In this work, instead of trying to keep the latent space to be Gaussian, we use a pre-trained contrastive encoder to obtain a clustered latent space. Then, for each cluster, representing a unimodal submanifold, we train a dedicated low complexity network to generate this submanifold from the Gaussian distribution. The proposed framework is based on the information-theoretic formulation of mutual
    
[^486]: 基于能量的模型用于语音合成

    Energy-Based Models For Speech Synthesis. (arXiv:2310.12765v1 [cs.SD])

    [http://arxiv.org/abs/2310.12765](http://arxiv.org/abs/2310.12765)

    本文引入了一种新的非自回归模型，即基于能量的模型 (EBMs)，通过噪声对比估计进行训练。该模型可以使用 Langevin MCMC 进行采样，并在语音合成任务中取得了改进。

    

    最近对于非自回归 (non-AR) 模型在语音合成领域引起了很多关注，比如 FastSpeech 2 和扩散模型。与自回归模型不同的是，这些模型的输出之间没有自回归的依赖关系，这使得推断过程更加高效。本文通过引入基于能量的模型 (EBMs) 来拓展非自回归模型的范围。论文描述了如何使用噪声对比估计来训练基于能量的模型，噪声对比估计依赖于正样本和负样本之间的比较。论文提出了一些生成有效负样本的策略，包括使用表现优秀的自回归模型。论文还描述了如何使用 Langevin Markov Chain Monte-Carlo (MCMC) 从基于能量的模型中进行采样。使用 Langevin MCMC 可以建立基于能量的模型与当前流行的扩散模型之间的联系。在 LJSpeech 数据集上的实验证明，所提出的方法相对于 Tacotron 2 有改进。

    Recently there has been a lot of interest in non-autoregressive (non-AR) models for speech synthesis, such as FastSpeech 2 and diffusion models. Unlike AR models, these models do not have autoregressive dependencies among outputs which makes inference efficient. This paper expands the range of available non-AR models with another member called energy-based models (EBMs). The paper describes how noise contrastive estimation, which relies on the comparison between positive and negative samples, can be used to train EBMs. It proposes a number of strategies for generating effective negative samples, including using high-performing AR models. It also describes how sampling from EBMs can be performed using Langevin Markov Chain Monte-Carlo (MCMC). The use of Langevin MCMC enables to draw connections between EBMs and currently popular diffusion models. Experiments on LJSpeech dataset show that the proposed approach offers improvements over Tacotron 2.
    

