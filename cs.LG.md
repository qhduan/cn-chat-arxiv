# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hide and Seek: HIDE Concepts to SEEK explanations for black box models](https://arxiv.org/abs/2610.13935) | 提出了一种无需模型权重或概念标注数据集的黑盒视觉模型概念级归因方法HnS，通过视觉-语言模型检索概念相关图像、遮挡图像中的概念区域，并依据模型输出的变化来衡量该概念的归因贡献。 |
| [^2] | [Constructive Bounds for the Topological Expressivity of Neural Networks](https://arxiv.org/abs/2610.13931) | 本文提出用神经网络输出的贝蒂数（而非决策边界的贝蒂数）来度量拓扑表示能力，并构造性地证明了特定架构的 ReLU 网络可以表达同伦等价于具有任意贝蒂数的球体楔和的空间，从而为拓扑表示能力给出了构造性上界。 |
| [^3] | [AI Models Can Provably Hide Arbitrary Capabilities](https://arxiv.org/abs/2610.13928) | 本文提出一种将加密神经网络嵌入模型权重的后门攻击方法，在密码学假设下可证明地隐藏任意能力，使现有能力评估手段难以引出或解读这些隐藏功能。 |
| [^4] | [KL Before and After Mixing: A Round-Complexity Separation in Parallel Sampling](https://arxiv.org/abs/2610.13924) | 该论文证明了并行采样中两种KL散度准则之间的轮数复杂度分离：在均衡二进制串的均匀分布上，按调度平均散度准则需要 $\Theta((\log n)^2)$ 轮，而按输出分布散度准则仅需双重对数阶（约 $\log_2\ln n$）轮。 |
| [^5] | [Goal-Oriented Bundle Management and Flow Control for Deep-Space Communications via Reinforcement Learning](https://arxiv.org/abs/2610.13923) | 本文提出一种基于强化学习的闭环框架，用于接触驱动深空网络中继节点的捆绑包调度与丢弃决策，通过优化接收端信息年龄（信息新鲜度）来提升月球与火星数据向地球传输的任务效用与公平性。 |
| [^6] | [DyssomniaNN: Multi-Scale Spiking Neural Modeling of Human Sleep Disorders and Perceived Restedness](https://arxiv.org/abs/2610.13918) | 提出多尺度脉冲神经网络框架DyssomniaNN，将睡眠稳态双过程模型、下丘脑“跷跷板”回路与皮层-丘脑脉冲网络耦合，成功模拟五种临床睡眠障碍并复现真实脑电特征，为客观睡眠与主观休息感之间的分歧提供了机理解释。 |
| [^7] | [Designing Protein Structures from Natural Language](https://arxiv.org/abs/2610.13917) | 该论文提出首个支持自然语言指令的全原子蛋白质结构生成模型Prompteina，通过构建包含180万结构和5000多万条文本说明的配对数据集进行训练，实现了类似“文本生成图像”的蛋白质设计范式，并在18项结构特性中的15项上超越现有基线模型。 |
| [^8] | [Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping](https://arxiv.org/abs/2610.13909) | 该论文系统性地重新评估了表格深度回归中的各类目标函数，发现分位数损失是唯一能相对调优后的MSE基线带来一致提升的目标函数，并进一步揭示了由于不同噪声水平的区域需要不同的训练时机，全局早停策略存在根本性局限。 |
| [^9] | [Hitting-Time Bounds for Asynchronous Q-Learning](https://arxiv.org/abs/2610.13901) | 该论文在仅要求状态-动作链不可约（允许周期性）的弱条件下，证明了零初始化、常数步长的异步Q学习只需 $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ 次转移即可高概率达到 $\varepsilon$ 上确界范数精度，且初始化成本以反向链最大期望命中时间 $\mathfrak h_\star$ 的加性形式出现。 |
| [^10] | [Prediction Is Not a Sufficient Memory State](https://arxiv.org/abs/2610.13881) | 本文证明仅能复现当前预测的记忆状态不足以支持后续学习：在精确累积学习中，固定维度的循环状态仅当损失为多项式时存在，且即使在当前预测完全相同的历史之间，继续学习仍可能额外需要 $r-d$ 个状态坐标（例如岭回归中必须存储设计矩阵）。 |
| [^11] | [Partially Informed Stochastic Optimization via Zeroth-Order Residual Tracking](https://arxiv.org/abs/2610.13871) | 提出了PISO方法，将可获取的梯度分量与残差的运行估计相结合进行零阶残差追踪，使随机探测产生的方差取决于追踪误差而非残差幅值，从而在光滑非凸目标上建立了无需对不可获取梯度分量施加一致界的收敛保证。 |
| [^12] | [Finite-Sample Guarantees for Fully Discrete Conditional Diffusion Models](https://arxiv.org/abs/2610.13851) | 本文为以有限去噪步数实现的条件扩散模型建立了首个有限样本理论保证，通过将分数训练目标与采样网格对齐来联合分析统计估计与数值离散化误差，获得了近优的KL散度与总变差误差收敛速率。 |
| [^13] | [Inference-Time Steering Across Scaffolds and Epitopes for Sample-Efficient De Novo Nanobody Design](https://arxiv.org/abs/2610.13844) | 该论文将纳米抗体设计中的条件选择（支架与表位）重新表述为推理时资源分配问题，提出全局SMC和改编BaSE两种动态预算分配器，使每GPU小时可设计候选产量提升近5倍。 |
| [^14] | [Returning to Development States: When Full Sandbox Snapshots Help Coding Agents](https://arxiv.org/abs/2610.13841) | 基于完整沙箱快照的 Go-Explore 式搜索可以提升编码智能体的解决率与效率，其效果主要依赖于代码仓库之外的状态（如数据库、服务及其他运行时状态）。 |
| [^15] | [Specialists Learn Faster: Knowledge Organization in Pre-training Shapes Knowledge Update Efficiency](https://arxiv.org/abs/2610.13835) | 研究发现按时间顺序组织预训练数据所形成的“专家”模型比随机打乱顺序训练的“通才”模型能以更少的更新步骤更高效地获取新知识，并从第一性原理推导出解释该现象的定律，据此设计了保留稳定历史知识的预训练数据组织策略。 |
| [^16] | [On the Design Space of Continuous Diffusion Language Models](https://arxiv.org/abs/2610.13834) | 本研究揭示了连续扩散语言模型中自条件化通过纠正训练/推理不匹配而生效的机制，并提出了无需额外前向传播的二级插值方法DoubleCat，从而实现更廉价且更有效的设计。 |
| [^17] | [Shared Canonical Component Removal Beyond Expert Views](https://arxiv.org/abs/2610.13819) | 本文提出共享典型成分移除（SCCR）审计框架，通过仅移除原始嵌入与专家嵌入之间强烈共享的方向，量化并揭示黑盒模型超越专家特征所利用的残余预测信号。 |
| [^18] | [TabMF: Learning Dataset Representations with Tabular Foundation Models](https://arxiv.org/abs/2610.13818) | TabMF提出了一种框架，通过架构感知的数据集级读出层和基于合成数据随机视图的对比后训练，将预训练表格基础模型（TabICLv2与TabPFN-2.5）重新用作有监督数据集的编码器，从而在数据集检索和分布偏移监测等任务上学习到强大的数据集级表示。 |
| [^19] | [Dream to Remember: Continual Steering of Generalist Robot Policies with World Models](https://arxiv.org/abs/2610.13816) | 该论文提出D2R方法，利用冻结的预训练世界模型通过想象推演来温习旧任务，从而在持续引导冻结的通用机器人策略学习新任务的同时避免灾难性遗忘，且无需重放缓冲区或新数据。 |
| [^20] | [Spanning Diffusion and Flow-based Posterior Sampling via Exact Gaussian Conditioning](https://arxiv.org/abs/2610.13812) | 该论文提出 β-Flow，一种基于精确高斯条件化（卡尔曼更新）的通用即插即用后验采样框架，通过相关参数 β 和置信度参数 κ 统一了流匹配、扩散及去相关先验等多种现有引导采样方法。 |
| [^21] | [QTestBench: A Test Framework for Evaluating LLM-Generated Quantum Code](https://arxiv.org/abs/2610.13811) | 该论文提出了QTestBench框架，首次通过将大语言模型生成的量子测试判定结果与独立的基准检查器对比，来评估这些测试能否有效区分正确与错误的量子代码实现，而非将测试本身当作评估标准。 |
| [^22] | [Diversity-Aware Ensemble of Quantum-Classical Models via Combinatorial Fusion](https://arxiv.org/abs/2610.13808) | 本文提出基于组合融合分析（CFA）的多样性感知方法，通过秩-得分函数和认知多样性将多个架构各异的量子电路与经典模型进行融合集成，在四个二分类基准上持续超越最佳单一模型。 |
| [^23] | [FORTE: A Production Framework for Feature Evaluation and Deprecation in Industrial Recommender Systems](https://arxiv.org/abs/2610.13806) | FORTE是一个与模型无关的生产级框架，通过高效的扰动实现支持每日特征评估，并利用分组消融处理特征交互，从而实现工业推荐系统中特征的系统化发现、验证与淘汰。 |
| [^24] | [Constrained Probabilistic Numerics: Gaussian Process Solvers for Linear PDEs under Linear Inequality Constraints](https://arxiv.org/abs/2610.13804) | 该论文提出了一种支持线性不等式约束（如非负性、单调性）的高斯过程PDE求解方法，证明等式与不等式条件化可以顺序进行，并通过截断高斯矩实现了非高斯后验的可计算性。 |
| [^25] | [Adaptive Variance Reduction without Knowing Oracle Smoothness](https://arxiv.org/abs/2610.13799) | 提出了一种两阶段自适应方差缩减算法，无需预知光滑常数或方差界，即可同时适应平均光滑性、标准光滑性和无噪声三种设置，分别达到 $O(T^{-1/3})$、$O(T^{-1/4})$ 和 $O(T^{-1/2})$ 的收敛速率。 |
| [^26] | [Randomized Momentum Transport with Cheap Predictors](https://arxiv.org/abs/2610.13789) | 本文提出随机动量传输方法，以概率 $q$ 执行昂贵的传输校正、其余时候使用廉价预测并配合重要性加权，在保持 $O(T^{-1/3})$ 平稳性收敛界的同时显著降低了动量方法的oracle调用成本。 |
| [^27] | [Optimal Algorithms for Nonsmooth Nonconvex Stochastic Optimization with Constraints](https://arxiv.org/abs/2610.13787) | 该论文提出了求解带凸约束的非光滑非凸随机优化问题的新算法，首次在多种设定下达到最优收敛率：使用随机梯度时复杂度为 Õ(δ⁻¹ε⁻³)，使用零阶查询时复杂度为 Õ(dδ⁻¹ε⁻³)。 |
| [^28] | [Your Discrete Graph Diffusion Model is Secretly Memorizing](https://arxiv.org/abs/2610.13786) | 本文首次证明离散图扩散模型存在“条件记忆化”现象——当反向扩散轨迹以损坏的训练图为条件时，模型几乎能完美恢复该训练图，而这种记忆化在常规推理和样本级评估指标中均无法被察觉。 |
| [^29] | [ReplaySSM: Cache SSM Inputs, Not State](https://arxiv.org/abs/2610.13778) | ReplaySSM 通过将 SSM 状态表示为周期性更新的检查点加近期输入缓冲区，将状态内存流量约减半，使投机解码回滚变为简单的缓冲区指针操作，并支持以低精度检查点进行输出计算。 |
| [^30] | [Geometry-Aware Multi-Objective Optimization: Norm-Constrained Oracles with Entropic Weight Updates](https://arxiv.org/abs/2610.13773) | 该论文提出一种将范数约束线性最小化预言机与任务权重熵更新相结合的单循环多目标优化方法，对任意有限维范数对实现了确定性 $O(T^{-1/2})$ 与随机 $O(T^{-1/4})$ 的遍历平稳性速率，且随机情形的主项与任务数量无关。 |
| [^31] | [From Pixels to Privacy-Sensitive Features: An Information-Theoretic Framework for Reconstruction Bounds](https://arxiv.org/abs/2610.13768) | 该论文提出了一个基于率失真理论的信息论框架，能在训练前计算出 DP-SGD 训练模型的重构误差下界，即使面对可观测训练过程、知晓其余训练数据和总体分布的最坏情况攻击者依然成立，并允许分析者选择面向隐私敏感特征的误差度量表示。 |
| [^32] | [DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs](https://arxiv.org/abs/2610.13767) | 提出了免校准的双尺度训练后量化框架DualQuant，利用最小化权重重建误差所得的缩放因子作为激活缩放，无需校准数据即可同时完成大语言模型的权重与激活低比特量化。 |
| [^33] | [Shortcut Learning in Geospatial Foundation Model Embeddings for Cross-Disaster Damage Assessment](https://arxiv.org/abs/2610.13766) | 该研究揭示了地理空间基础模型在灾害损毁评估中存在捷径学习问题——留出整个灾害事件测试时性能大幅下降，且仅凭建筑物数量或灾前图像就能取得较高准确率，说明模型可能依赖事件/场景特征而非真正可迁移的损毁证据。 |
| [^34] | [When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Priors](https://arxiv.org/abs/2610.13765) | 本文指出离散图扩散模型的E-R随机图先验是GNN去噪器失效的根本原因，并提出潜在社交性先验（LSP）与联合扩散过程，使节点社交性变量与图结构共同演化，从而恢复生成图中的节点异质性和高阶结构。 |
| [^35] | [How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?](https://arxiv.org/abs/2610.13764) | 本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。 |
| [^36] | [Matched-Context Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2610.13755) | MCGPO通过匹配上下文从已有轨迹中构建加权动作偏好，在无需critic和额外rollout的情况下，为稀疏终端反馈下长时程LLM智能体的中间决策分配信用。 |
| [^37] | [Studying the Effects of Task Diversity Using Small Mathematical Worlds](https://arxiv.org/abs/2610.13750) | 本文提出了两个程序化生成的数学任务生态系统IntWorld和PermWorld，通过精确控制任务数量，实现了对任务多样性如何影响神经网络训练的系统研究。 |
| [^38] | [A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning](https://arxiv.org/abs/2610.13744) | 本文发现自蒸馏中的评分信号既无法定位学生推理中的错误，也无法可靠区分正确与错误的rollout，并指出错误定位本质上是搜索问题，因而提出批评条件化的在策略自蒸馏方法来弥补“只会打分、不会搜索”的评分教师的结构性缺陷。 |
| [^39] | [SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices](https://arxiv.org/abs/2610.13740) | SmolWAM通过复用20亿参数教师模型的预训练主干并精简推理接口，仅用3.6 GPU小时训练出一个轻量潜在世界-动作模型，在边缘设备上以快11倍的速度实现95.5%的操作成功率，同时仍保留预测场景运动的能力。 |
| [^40] | [SafeDiSTL: Decomposing Joint Constraints for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.13737) | SafeDiSTL利用信号时序逻辑鲁棒性与可微动力学提供的动作敏感度，将团队级联合约束代价分解并分配到各智能体，解决了联合约束下多智能体强化学习的信用分配难题。 |
| [^41] | [Orthros: Phase-Aware Heterogeneous Attention for Efficient Transformers](https://arxiv.org/abs/2610.13735) | Orthros注意力通过相位感知的方式在序列处理的不同阶段动态切换计算模式，实现了线性复杂度的预填充与完整注意力的解码相结合，从而在重预填充、轻解码的长文本场景下比原始Transformer具有更低的渐近复杂度。 |
| [^42] | [SeoulMMOD: A Large-Scale Multimodal Origin-Destination Flow Benchmark](https://arxiv.org/abs/2610.13730) | SeoulMMOD是首个覆盖首尔全市6种出行方式、提供三年逐小时OD流数据及多类辅助信号的大规模多模态出行基准数据集，并在统一协议下系统评测了14种预测基线模型。 |
| [^43] | [Amortized Counterfactual Prediction with Graph Neural Networks](https://arxiv.org/abs/2610.13724) | 提出反事实图神经过程（CFGNP），利用图神经网络端到端地直接从观测数据回答反事实查询，无需先估计因果图，从而避免因果图估计误差向下游反事实预测的传播。 |
| [^44] | [Machine Unlearning for Federated Socially Fair k-Means Clustering](https://arxiv.org/abs/2610.13717) | 本文提出了一种联邦社会公平k-均值聚类的精确机器遗忘方法，通过为每个群体维护独立的本地摘要，使得在全局公平权重变化时未受影响机构的摘要依然有效，并在理想采样下证明了与客户端群体比例无关的初始化近似保证。 |
| [^45] | [Towards Understanding the Training of GFlowNets: A Theoretical Perspective](https://arxiv.org/abs/2610.13711) | 该论文为GFlowNets的流匹配、细致平衡和轨迹平衡三种训练目标建立了首个统一理论框架，用期望平衡损失定量刻画了采样分布误差，并证明了在多种采样分布与反向策略设置下的有限时间收敛保证。 |
| [^46] | [Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy](https://arxiv.org/abs/2610.13708) | Adaptive DHiSparse 在分层稀疏 KV 架构上支持推测解码，并根据负载动态调整 KV 缓存驻留策略，在 100K 上下文、128 并发下相比 HiSparse 将解码吞吐提升至 1.45 倍。 |
| [^47] | [Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks](https://arxiv.org/abs/2610.13693) | 该论文提出通过双层 Stackelberg 优化将校准器与主干网络端到端联合训练，无需单独的校准数据集或专用校准损失，仅引入校准器学习率这一个超参数，即可同时学习到既准确又校准良好的神经网络。 |
| [^48] | [Variance-Reduced Policy Gradient is Nearly Minimax-Optimal](https://arxiv.org/abs/2610.13692) | 该论文提出LVR-PG统一理论框架，证明了一类结合递归更新与防御性重要性采样的一阶方差缩减策略梯度算法，在弱梯度支配等最小假设下、无需梯度预处理器，即可达到 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 样本复杂度的近乎极小极大最优最后一次迭代全局收敛。 |
| [^49] | [Transformer-based In-Context Score Operator Learning for Diffusion Models](https://arxiv.org/abs/2610.13690) | 本文提出一种基于Transformer的上下文内分数算子学习框架，使扩散模型仅需少量示例样本即可在推理时适应从未见过的分布，无需再训练或参数更新就能完成反向扩散采样。 |
| [^50] | [X-AVDD: Cross-Attentive Audio-Visual Dataset Distillation](https://arxiv.org/abs/2610.13687) | X-AVDD提出基于交叉注意力的音视频数据集蒸馏方法，通过将跨模态对齐与联合优化解耦，将端到端合成成本降低约8倍GPU小时，同时生成高度对齐的合成数据并显著优于现有蒸馏方法。 |
| [^51] | [On the Encoding of Calendar Features in Covariate-Informed Time Series Foundation Models](https://arxiv.org/abs/2610.13676) | 本研究首次系统比较了七种日历特征编码方式在时间序列基础模型中的表现，发现编码方式的选择会显著影响模型的零样本预测精度。 |
| [^52] | [Not All Dimensions Are Equal: Outlier-Aware Attention Linearization with Grouped Quadratic Kernels](https://arxiv.org/abs/2610.13674) | 该论文发现预训练注意力中少数离群维度主导了线性化的近似误差，并提出维度感知的OAL方法，通过Hadamard乘积表述将全局系数放宽为维度特定的线性项与跨维度二次交互，再以分组二次核在更少参数下实现高保真的注意力线性化。 |
| [^53] | [A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models](https://arxiv.org/abs/2610.13663) | 本文提出一个由分析师、编码员和记录员组成的多智能体LLM框架，在保持整体网络结构不变的前提下自动生成并调优指定关键组件的代码，并可迁移至脉冲神经网络分类与PINN求解高阶微分方程等不同类型模型。 |
| [^54] | [Pretraining and Test-Time Training Budgets Matter for ARC: A Controlled Study of VARC and TRM](https://arxiv.org/abs/2610.13660) | 本文在统一协议下控制变量地对比视觉型VARC与递归型TRM模型，证明预训练与测试时训练预算（尤其是增加TTT样本呈现次数）对ARC性能有决定性影响，提示性能差异不能简单归因于架构本身。 |
| [^55] | [Scalable Sparse Autoregressive Variational Inference for Non-Gaussian Spatial Fields](https://arxiv.org/abs/2610.13656) | 该论文提出了一种可扩展的变分推断框架，通过在变分族中镜像稀疏最近邻自回归先验结构、使用归一化流条件分布和共享神经条件器摊销参数，并以双随机优化实现对数多项式迭代复杂度，从而高效解决大规模非高斯空间场的后验推断难题。 |
| [^56] | [TabReliBench: Evaluating Tabular Foundation Models Beyond Predictive Accuracy](https://arxiv.org/abs/2610.13654) | 提出了 TabReliBench，一个从校准、分布质量与区间效率等可靠性维度超越单纯预测精度来系统评估表格基础模型的大型分类与回归基准，发现模型排名随可靠性指标而变化，且良好的边际覆盖并不代表有效的条件覆盖。 |
| [^57] | [A Critical Look at Power Sampling when Test Time Sharpening LLMs](https://arxiv.org/abs/2610.13653) | 该论文证明基于MCMC/SMC的幂采样方法在现实计算预算下无法收敛到目标幂分布，其收益可由简单的Best-of-N采样复现，表明其成功源于筛选高概率序列而非真正近似幂分布。 |
| [^58] | [Electron-Density-Informed Molecular Property Prediction Using Surrogate Functionals](https://arxiv.org/abs/2610.13649) | 该方法将基态电子密度作为模型输入并强制其满足替代泛函最小化条件，在QM9和QM9数据集的广延分子性质预测上超越了现有最先进模型，并在更大分子上展现出更强的泛化能力。 |
| [^59] | [SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data](https://arxiv.org/abs/2610.13644) | 该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。 |
| [^60] | [PANDA: Prior-guided Attentional Dual-path Architecture](https://arxiv.org/abs/2610.13635) | 提出了PANDA——一种可插入两阶段消息传递超图神经网络的先验引导双路径架构，通过将接收者归一化注意力与结构先验插值的主路径和非竞争性辅助路径相结合，克服了固定权重与归一化注意力在异质场景下的局限性。 |
| [^61] | [Private and Distributionally Robust Federated Multiclass Classification](https://arxiv.org/abs/2610.13631) | 该论文提出了一个差分隐私且分布鲁棒的联邦多分类框架，基于ADMM算法求解联邦Wasserstein分布鲁棒目标，并通过注入校准高斯噪声为客户端数据提供样本级隐私保护和对标签噪声的鲁棒性。 |
| [^62] | [JitterVLA: A Replay Benchmark for Motion Quality in VLA-Driven Human-Robot Manipulation](https://arxiv.org/abs/2610.13628) | 提出JitterVLA重放基准，通过在不同控制器间重放相同的指令和到达时间轨迹来隔离控制器对运动质量的影响，并证明简单的插值方法即可将人机操作中的p99加加速度降低83%（代价仅为2厘米跟踪误差）。 |
| [^63] | [Kernel-based External Validity of Experimental Studies](https://arxiv.org/abs/2610.13625) | 该论文提出用核方法（最大均值差异MMD）形式化定义实验研究的外部有效性，并建立了研究规模与MMD分位数之间的渐近幂律关系，从而能够从小规模初步研究估计出具有外部有效性的研究所需规模。 |
| [^64] | [Some remarks on the connection between generative diffusion and optimal transport](https://arxiv.org/abs/2610.13614) | 本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。 |
| [^65] | [Vocal Intensity Conversion with Continuous SPL Control via Adversarial Training](https://arxiv.org/abs/2610.13608) | 本文提出一种基于对抗训练的语音强度转换方法，首次实现对连续分贝刻度上任意目标声压级的精确控制，使用WavLM特征时达到了目标强度变化的83%。 |
| [^66] | [CorrFlow: Exact and Permutation-Equivariant Flow Matching on the Correlation Elliptope](https://arxiv.org/abs/2610.13606) | CorrFlow通过广义Fisher变换将满秩相关矩阵双射映射到无约束坐标，再用置换等变的图流匹配网络学习其经验分布，从而无需投影即可精确保证输出位于相关椭圆面上，并对变量排序保持等变性。 |
| [^67] | [A small cognitive core explains human behavior at scale](https://arxiv.org/abs/2610.13603) | 该研究将多个经典认知原理整合为一个统一的计算模型CORE，其在超过700万个人类选择数据上拟合后不仅优于黑箱模型，还能再现75%的已报告行为效应，表明一个小型认知核心即可解释大部分人类行为。 |
| [^68] | [EchoTab: Learning Measurement-Grounded Representations for Echocardiographic Diagnosis through Tabular Reconstruction](https://arxiv.org/abs/2610.13599) | EchoTab提出了一种以测量为基础的表示学习框架，将测量特定的token作为超声视角与诊断之间的中间表示，并通过表格重建任务学习测量基础表示，从而提升多视角超声心动图诊断的准确性。 |
| [^69] | [Lower Bounds for Randomized Linear-Oracle Optimization over Strongly Convex Sets](https://arxiv.org/abs/2610.13590) | 本文针对强凸集合上基于线性极小化预言机的优化问题，分析了现有 $1/T^2$ 最坏情况误差下界的脆弱性，并证明即使引入随机化，此类方法在 T 次 LMO 调用后仍无法突破 $1/T^2$ 的误差下界。 |
| [^70] | [Probabilistic Forecasting of Electric Vehicle Charging in a Time-varying Hierarchical Infrastructure](https://arxiv.org/abs/2610.13588) | 提出将分层预测与处理缺失数据的保形预测相结合，在持续演化的电动汽车充电基础设施分层结构中，为从单个充电点到国家级的每个聚合节点生成有效的预测区间。 |
| [^71] | [RARA: Robust Policy Fine-Tuning through Retrieval-Anchored Representation Alignment for Robotic Manipulation](https://arxiv.org/abs/2610.13578) | 本文提出RARA，一种两阶段微调方法，通过检索预训练中任务意图和运动相似的表示特征并对齐目标演示的表示，在少量演示适应新任务的同时保护预训练表示不被扭曲，从而保持模型在分布外偏移下的鲁棒泛化能力。 |
| [^72] | [$L_0$ Feature Selection via Monte Carlo Tree Search in High-Dimensional Financial Regression](https://arxiv.org/abs/2610.13574) | 本文提出一种基于广义嵌套 Rollout 策略自适应蒙特卡洛树搜索（GNRPA）的 $L_0$ 特征选择框架，将特征选择建模为以样本外验证性能为评分准则的序贯决策过程，从而克服 Lasso 收缩偏差与精确稀疏回归计算困难的问题。 |
| [^73] | [Simultaneous Competitive and Regret Guarantees for TCP Acknowledgment](https://arxiv.org/abs/2610.13571) | 该论文为TCP确认问题提出了一种随机化在线算法，在无需预先知道报文数量或离线最优成本的情况下，同时实现了(2+ε)的竞争比保证和相对于最优专家的次线性期望遗憾。 |
| [^74] | [Rationally Linearized MLPs](https://arxiv.org/abs/2610.13568) | RaLiMPs利用低次有理近似替代标准激活函数来高效线性化MLP，在从隐式3D模型到大语言模型的多种应用中，以更低的推理成本实现了具有竞争力甚至更优的精度。 |
| [^75] | [Probabilistic Ship Trajectory Forecasting using Transformer Models on AIS Data](https://arxiv.org/abs/2610.13557) | 该论文利用AIS数据、基于核密度估计的热图编码以及分位数损失训练的Transformer和Performer模型，实现了受限水域中船舶轨迹的概率预测，并给出模型感知的预测区间。 |
| [^76] | [Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning](https://arxiv.org/abs/2610.13536) | 提出了一种无需预先知晓目标特征的无监督防御方法，仅通过分析学生模型的参数更新即可发现、解释并抑制从共享基础参数的教师模型阈下继承的失对齐行为特征。 |
| [^77] | [S-Space: Exploring Spatial Workspace in Multimodal Language Models](https://arxiv.org/abs/2610.13527) | 该论文发现多模态语言模型内部存在一个固有的线性子空间"S-Space"，作为模型的内部空间工作空间，以连续坐标编码物体位置，并支撑感知编码、视角变换和言语输出等空间推理的全过程。 |
| [^78] | [Simulating Training Spikes of Deep Neural Networks by Layer-wise Random Feature Models](https://arxiv.org/abs/2610.13524) | 本文提出用逐层随机特征二次模型来模拟深度神经网络训练中的损失尖峰现象，成功预测了Adam优化器 $\beta_2$ 接近1时Transformer语言模型的训练尖峰，并据此推导出能提升性能的 $\beta_2$ 调度策略。 |
| [^79] | [Asymptotic Analysis of Discrete-Time Self-Attention Dynamics in Transformers](https://arxiv.org/abs/2610.13518) | 本文首次对Transformer的离散时间自注意力动力学进行了系统研究，给出了单头与多头情形下平衡点的完整分类和稳定性分析，并证明其主要吸引子是与值矩阵主特征向量对齐的共识平衡点。 |
| [^80] | [FedPCD: Priority-Constrained Federated Aggregation](https://arxiv.org/abs/2610.13508) | FedPCD提出了一种优先级约束的联邦学习优化框架，在保持传统联邦聚合为主要目标的同时，仅施加满足客户端性能约束所需的最小校正，将全局优化与客户端公平保护相分离，在异构数据场景下显著优于现有最先进方法。 |
| [^81] | [Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning](https://arxiv.org/abs/2610.13504) | 该论文提出解释性世界模型（EWM）框架，用语言模型生成的因果解释取代传统的时间邻近性来进行强化学习中的信用分配，通过保守分层算子递归重分配奖励并保持回合总回报不变，从而提升生成式推荐系统中长时程决策的样本效率。 |
| [^82] | [A Scalable Learned Iterative Solver for Nonlinear MPC via Temporal Decomposition](https://arxiv.org/abs/2610.13480) | 本文通过在LISCO求解器架构中利用问题的时间分解结构，使神经网络规模与预测时域无关，从而实现了对大型非线性系统可扩展的实时非线性模型预测控制。 |
| [^83] | [Harmonic analysis and circuit approximation of Sobolev functions on quantum input spaces](https://arxiv.org/abs/2610.13477) | 该论文通过紧群上的调和分析刻画了量子态与量子幺正操作所在空间上Sobolev函数的Peter–Weyl分解，并结合谱截断与基于Solovay–Kitaev定理的有限深度量子电路逼近，建立了将谱截断误差与电路逼近误差分离的定量逼近速率。 |
| [^84] | [SqueezeBench: Saturation-Resilient Evaluation of Mathematical Capability](https://arxiv.org/abs/2610.13461) | SqueezeBench提出了一种抗饱和的数学能力评估基准，要求AI不断提出并用Lean机器验证证明整数函数的上下界记录，使每次成功都提高门槛而非消除问题。 |
| [^85] | [Learning by Compression: Risk Estimates Attain the Law-of-Large-Numbers Rate](https://arxiv.org/abs/2610.13442) | 本文证明基于优先压缩的学习算法的风险界中并非固有存在对数因子，其风险估计对所有复杂度取值一致地达到 $\boldsymbol{k}/N + O(1/\sqrt{N})$ 的大数定律速率，与经验估计速率相同。 |
| [^86] | [Revisiting Accent Classification with DistilHuBERT: Speaker-Disjoint Evaluation and Cross-Corpus Generalization](https://arxiv.org/abs/2610.13440) | 该研究发现文件级评估会大幅高估基于DistilHuBERT的口音分类模型性能，说话人分离评估才能更真实地反映模型对未见说话人的泛化能力，且仅用外部数据训练的模型难以可靠泛化到L2-ARCTIC语料库。 |
| [^87] | [PPG-Bench: A Unified Benchmark for PPG Foundation Models](https://arxiv.org/abs/2610.13435) | PPG-Bench是一个统一的开源基准测试，在九个数据集和26个任务上，通过四种下游评估模式对五个PPG基础模型进行了系统、公平的比较。 |
| [^88] | [Does Financial Speech Need Continual Learning Regularization? A Study of EWC and mini-KFAC on Earnings Call ASR](https://arxiv.org/abs/2610.13432) | 该研究发现在将Whisper适配到财报电话会议语音识别时，新任务与旧任务之间存在互补性而非灾难性遗忘，因而正则化反而有害；但当确需正则化时，逐层mini-KFAC比对角EWC能取得更低的词错误率和更稳定的优化效果。 |
| [^89] | [One Thing Leads to Another: The Essentials of Time Series Causal Discovery](https://arxiv.org/abs/2610.13430) | 本教程论文首次为统计学、动力系统和机器学习等不同研究社区的时间序列因果发现方法提供了统一概述，系统介绍了其关键原理、局限性、假设、评估基准方法以及在农业、地球科学、医疗保健和人工智能系统等下游任务中的重要性。 |
| [^90] | [In-Context Learning for System Identification: Probabilistic Formulation, Long-Sequence Handling, and Benchmark Validation](https://arxiv.org/abs/2610.13411) | 本文提出一种统一的概率化上下文学习元辨识架构，将基于Transformer的编码器-解码器与长序列处理机制相结合，实现长时域校准预测，并通过基准测试验证了其有效性。 |
| [^91] | [Accelerating Linear Bandits with Persistent Candidate Sets](https://arxiv.org/abs/2610.13407) | PC-LinUCB 通过维护少量活跃动作并每轮均匀采样挑战者，将线性老虎机每轮打分评估次数从 O(K) 降至 O(k+q)，其额外的发现成本不随时间范围增长，并在均匀采样下达到了 Ω((K−k)/q) 的遗憾下界。 |
| [^92] | [Desertification Forecasting in Semi-Arid Regions: Using Deep Learning to Probe the Observability Limits of Land Degradation Dynamics](https://arxiv.org/abs/2610.13406) | 该研究用ConvLSTM模型基于卫星观测数据预测巴西塞阿拉州土地退化指数的逐年演变，发现该指数的像元级变化以难以预测的逐年跳变为主（梯度提升模型仅能解释4.7%的方差），而ConvLSTM仍能相对持续性基线将提前一至四年预测的RMSE降低1.6%至4.4%，从而揭示了土地退化动态的可观测性极限。 |
| [^93] | [Why Linear Models Can Still Work in a Complex and Changing World](https://arxiv.org/abs/2610.13404) | 该论文论证了当世界的复杂性超出有限、部分观测数据所能可靠学习的范围时，灵活的非线性模型会过度拟合特定环境的细节而难以迁移，因此省略不稳定非线性成分的线性模型在分布变化的新环境中反而能取得更低的预测误差。 |
| [^94] | [Model-free Bayesian Clustering and Multimodal Posterior Summarisation](https://arxiv.org/abs/2610.13397) | 本文提出了一种无模型、噪声感知的贝叶斯聚类框架，通过鞅后验避免MCMC采样，并利用Wasserstein型摘要方法将多模态后验高效压缩为少量加权代表，同时边缘化了密度水平的不确定性。 |
| [^95] | [PlacePulse: Learning Transferable Place Representations from Structural Geographic Data](https://arxiv.org/abs/2610.13390) | PlacePulse是一个结构性地理空间基础模型，它将美国超过105万个H3区域的572个静态结构属性压缩为可迁移的256维嵌入表示，并配套提出了区分内在表示质量与下游预测性能的双轴评估基准。 |
| [^96] | [BLOKS: Scaling Kernel Solvers with Sketched Block Preconditioning](https://arxiv.org/abs/2610.13382) | BLOKS提出了一种将块Jacobi近似与草图化相结合的新颖预条件子，在加速核岭回归近似求解收敛速度的同时有效控制了内存成本，兼顾了效率与模型表达能力。 |
| [^97] | [Overcoming Data Imbalances with Expert Selection Biases](https://arxiv.org/abs/2610.13370) | 该论文提出在MoE模型中引入基于类别的专家选择偏差，将不同类别的词元优先路由到专门的专家子集，从而在数据类别不平衡的情况下实现软性类别对齐的专家专门化，有效缓解少数类别容量受限的问题。 |
| [^98] | [Opening the Black Box of Classifier-Free Guidance via Information Bottleneck](https://arxiv.org/abs/2610.13369) | 该论文从信息论视角将扩散生成轨迹建模为随时间变化的信息瓶颈，揭示了条件信息的有效性在经过一个关键阶段后会显著增强，从而为无分类器引导（CFG）的激活时机提供了原则性的理论基础。 |
| [^99] | [AgentSWE: Can Coding Agents Build the Agent You Actually Want?](https://arxiv.org/abs/2610.13361) | 提出了首个检验编码智能体能否忠实构建出真正满足用户需求的智能体的基准 AgentSWE，其以“委托”形式覆盖创建、编辑、优化三个开发阶段共25个任务，并通过保留用例和需求驱动的严格评分标准进行验收。 |
| [^100] | [EmoGavel: Grounding-Aware Reinforcement Learning with Verifiable Rewards for Multimodal Emotion Preference Learning](https://arxiv.org/abs/2610.13353) | 提出EmoGavel框架，通过将定位感知监督微调与可验证奖励强化学习相结合，在多模态情感偏好学习中实现偏好预测、错误定位、修复和缺失证据恢复的联合优化。 |
| [^101] | [Hidden Data Challenges in Clinical Machine Learning](https://arxiv.org/abs/2610.13350) | 该论文以MIMIC-IV为案例，系统揭示并量化了临床机器学习中广泛复用的预处理流水线在时间、语义和人群对齐上的隐性假设违规——包括时间泄漏、特征-任务错位和有偏插补——并说明这些模糊的数据处理选择会如何改变模型可见的信息并影响预测解释。 |
| [^102] | [Understanding Mamba-3 Through Oscillatory Dynamics](https://arxiv.org/abs/2610.13341) | Mamba-3 中依赖输入的旋转可被理解为欠阻尼二阶动力学，这种振荡机制通过调节信息的积累方式，提升了状态跟踪、长上下文预测和长度泛化能力。 |
| [^103] | [AgentTune: A Unified Training Lifecycle for Tool-Using LLM Agents](https://arxiv.org/abs/2610.13335) | 提出了AgentTune——一个统一的开源Python框架，通过两层EventLog表示整合了工具使用型LLM智能体从构建、训练、评估到蒸馏、维护的全生命周期流程。 |
| [^104] | [PDE-PFN: Prior-Data Fitted Neural PDE Solver](https://arxiv.org/abs/2610.13334) | 提出了PDE-PFN，一种通过上下文学习近似PDE解的后验预测均值的先验数据拟合神经求解器，它利用物理信息神经网络生成的带噪声近似解进行预训练，实现了跨异构方程的泛化能力以及对噪声先验的鲁棒性。 |
| [^105] | [Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning](https://arxiv.org/abs/2610.13330) | 本文提出首个基于掩码扩散语言模型的化学推理智能体ChemDiffAgent，将化学推理建模为对推理与工具使用轨迹的迭代去噪过程，使工具使用决策能够在全轨迹上下文中联合确定，突破了传统自回归范式在双向分子理解和多源证据整合上的局限。 |
| [^106] | [DuoRate: Two-Rate Per-Expert Precision Allocation for Mixture-of-Experts Serving](https://arxiv.org/abs/2610.13328) | 针对MoE模型中存储占用与解码流量这两种不同速率需求之间的“存储-读取差距”，DuoRate提出离线存储单一精度层级、在线依据工作负载自身的预填充特征求解解码泛函下的率失真分配，实现逐专家精度的双速率优化。 |
| [^107] | [Taking Tokenization off the Host for Agentic LLM Serving](https://arxiv.org/abs/2610.13325) | 提出 GraphTok，一个将完整分词流水线封装为可重放 CUDA 图并逐位复现参考结果的 GPU 分词器，比 CPU 快 4.7-6.0 倍，可将分词从主机彻底卸载，从而显著降低智能体场景下 LLM 服务的首 token 延迟并提升 GPU 集群可扩展性。 |
| [^108] | [SBNO : Schr\"odinger Bridge Neural Operator for the Forward and Inverse Problems under Missing Information](https://arxiv.org/abs/2610.13317) | SBNO提出了一种免方程的薛定谔桥匹配神经算子框架，在观测稀疏且物理信息部分缺失的情况下，以单一模型同时解决PDE的正问题与逆问题。 |
| [^109] | [JEPA-kMMVAE: CSI Prediction for Time-Varying Wireless Channels](https://arxiv.org/abs/2610.13315) | 提出JEPA-kMMVAE框架，将联合嵌入预测架构与k记忆马尔可夫变分自编码器结合，通过时间结构化的潜在表示从含噪观测中预测时变多天线信道的未来CSI，其中残差校正在中高信噪比下优于基线方法。 |
| [^110] | [Sustainable Cloud-Fog-Edge Systems for Smart Buildings: Algorithmic Adaptation under Memory Constraints and Dynamic Orchestration Strategies](https://arxiv.org/abs/2610.13307) | 该论文提出以“建筑本身即数据中心”为核心理念的可持续云-雾-边缘系统，通过低成本Arduino弹性数据采集、内存约束下的轻量级算法自适应与动态编排策略，实现智能建筑中自主、弹性且保护隐私的边缘学习系统。 |
| [^111] | [Structure-Aware Numerical Stabilization for Half-Precision Proximal Policy Optimization](https://arxiv.org/abs/2610.13306) | 提出PPO-FP16方法，通过全局标量保护、策略变化残差保护和高效持久更新保护三个结构感知模块，解决了FP16直接训练PPO时数值误差反馈累积的问题，在保持原有目标函数和训练设置的同时实现了稳定高效的半精度训练。 |
| [^112] | [MORAD: Reinforcement Learning with Multi-Objective Rewards for RNA Inverse Design](https://arxiv.org/abs/2610.13305) | MORAD提出了一个在线强化学习框架，通过联合优化涵盖三级几何、碱基配对、集合行为和序列组成的六项多目标奖励，为骨架条件RNA扩散模型训练单一可复用的共享策略，用于RNA逆向设计。 |
| [^113] | [DAG Structure Learning with Node-Varying Heavy-Tailed Disturbances](https://arxiv.org/abs/2610.13304) | 提出了节点异质的Student-t DAGMA方法，可联合估计DAG图结构与跨节点变化的重尾扰动参数，在异质重尾噪声下显著提升了图结构恢复的准确性。 |
| [^114] | [Sonic-AMG: Streaming Human Motion Generation from Spatial Audio](https://arxiv.org/abs/2610.13300) | 提出了首个从空间音频流式生成人体动作的框架Sonic-AMG，通过将声学内容与身体相对的声源几何信息分别编码为独立标记，由因果变换器生成动作潜变量，并引入排练训练来缓解训练与推理之间的历史不匹配问题。 |
| [^115] | [Decoding Register Tokens Preserves Global Scene Details](https://arxiv.org/abs/2610.13299) | 该研究发现，让表示自编码器（RAE）的解码器通过联合自注意力同时处理寄存器标记与全部补丁标记，能够保留光照、模糊、风格等全局场景细节，从而提升图像重建质量。 |
| [^116] | [Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks](https://arxiv.org/abs/2610.13292) | 本文通过拉格朗日神经网络精确学习势能函数，并结合信号感知采样策略，首次证明仅从轨迹数据即可恢复无法用成对项之和表示的不可约三体相互作用（Axilrod-Teller三偶极相互作用）。 |
| [^117] | [Do Prompt Optimizers Break Requirements They Are Not Scored On? A Preregistered Study](https://arxiv.org/abs/2610.13287) | 预注册研究发现，提示词优化器在提升目标指标的同时确实会破坏未被评分的要求（如引用和格式），但事后分析表明，表面的性能退化主要源于推理过程挤占输出空间导致答案被截断而无法提取，而非真正的内容质量下降。 |
| [^118] | [MHR-Router: Multimodal Heterogeneous Retrieval-Enhanced Adaptive Routing for VLM Inference](https://arxiv.org/abs/2610.13278) | 提出MHR-Router框架，通过融合冻结的多模态特征构建查询表示，并检索专家模型的成功与失败记忆作为局部证据，实现将多模态查询自适应且准确地路由到最合适的VLM专家模型。 |
| [^119] | [CopRoute: Dependence-Aware Conformal Routing for Language Models](https://arxiv.org/abs/2610.13276) | CopRoute通过显式建模模型失败之间的依赖结构，按每个模型的边际新增正确回答量对模型排序并进行共形校准，从而在控制全错风险的同时提升语言模型集合路由的覆盖率。 |
| [^120] | [CV-OPD: Contrastive Vision On-policy Distillation](https://arxiv.org/abs/2610.13264) | 该论文发现传统在策略蒸馏会让学生视觉语言模型走捷径、未能真正依赖视觉证据（即使视觉区域被遮挡仍能答对问题），并提出CV-OPD方法，通过在相同学生前缀条件下对比正负视觉裁剪所诱导的教师分布，并引入基于间隔的对比约束，迫使学生模型真正利用视觉信息进行推理。 |
| [^121] | [Data-Adaptive Mahalanobis Metric Learning for Cross-Head Attention in Transformers](https://arxiv.org/abs/2610.13261) | 本文提出DAMCHA注意力机制，通过学习完整的、依赖输入的跨头马氏度量矩阵来泛化多头注意力，显式捕捉头间交互，并使注意力几何结构随数据动态自适应。 |
| [^122] | [Neural Surrogate Modeling of Ultrasonic Wave Propagation in Heterogeneous Media](https://arxiv.org/abs/2610.13258) | 提出了一种将傅里叶神经算子与局部积分重采样融入U-Net架构的新型神经代理模型，能够在有限数据下高效、鲁棒地模拟非均匀介质中的高频超声波传播，并适用于不同空间尺度。 |
| [^123] | [CAST: Credit Alignment for Cross-Tokenizer On-Policy Distillation](https://arxiv.org/abs/2610.13257) | CAST提出了一种跨分词器在线蒸馏中的“移动或保持”信用对齐框架，利用预测不确定性和教师信号平滑度将块级优势精确分配到学生令牌上，并对无教师对应的令牌施加锚定约束。 |
| [^124] | [Tabular Foundation Models Are Effectively Shallow](https://arxiv.org/abs/2610.13253) | 表格基础模型（TFMs）实际上是“浅层”的——其 transformer 块中高达 96% 可以被一个闭式、无需标签的仿射转换器替换而不损失性能，说明这类模型存在大量架构冗余，可以被大幅简化。 |
| [^125] | [State-OPD: State-Grounded On-Policy Distillation for Multi-Turn Agents](https://arxiv.org/abs/2610.13238) | 该论文提出State-OPD，通过交互状态将学生推理区分为先验推理与当前推理两种角色——先验推理会扭曲教师监督、当前推理影响同状态下的动作多样性——从而实现更有效的多轮智能体在线策略蒸馏。 |
| [^126] | [The Predictability Bias of Joint-Embedding Predictive Learning in Medical Imaging](https://arxiv.org/abs/2610.13237) | 该论文提出TAP-JEPA模型并首次揭示了联合嵌入预测学习中的“可预测性偏差”——预测路径对可从上下文推断的解剖结构预测能力保持良好，但对临床诊断至关重要的散发性局灶病理的预测能力显著下降。 |
| [^127] | [Learning Schwarzschild Geodesic Motion with Lagrangian Neural Networks](https://arxiv.org/abs/2610.13234) | 本文提出解耦的位置-速度归一化方案与能量约束的对数均匀采样数据生成策略两项协同改进，攻克了拉格朗日神经网络学习史瓦西测地线运动这一广义相对论领域的开放性挑战。 |
| [^128] | [ReSFlow: Rectifying Categorical Flows on Statistical Manifolds for One-Step Generation](https://arxiv.org/abs/2610.13225) | 提出ReSFlow模型，通过在统计流形上校正类别流使其轨迹逼近测地线，实现离散数据的一步生成，大幅降低推理计算成本。 |
| [^129] | [Tensor-based Fusion of Multi-modal Data for Predicting Cancer Risk Levels](https://arxiv.org/abs/2610.13221) | 本文提出一种集成特征提取、张量分解与聚类的无监督机器学习框架，通过融合异构的基因组与临床多模态数据提取具有生物学意义的潜在特征，从而提升癌症风险等级预测的准确性。 |
| [^130] | [Golden Anchor for Video Generation](https://arxiv.org/abs/2610.13217) | 该论文揭示了文生图模型直接生成的图像并不适合作为视频生成的锚点，并通过基于图像奖励微调T2I模型来生成结构和角色细节更丰富的“黄金锚点”，从而提升下游视频生成质量。 |
| [^131] | [Monarch-MoE: Structured Experts for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2610.13215) | Monarch-MoE用固定置换连接的块对角因子替代LoRA专家中的稠密因子，通过块级梯度隔离缓解多模态持续指令微调中的灾难性遗忘，同时将适配器参数减少约75%和50%。 |
| [^132] | [Verifiable Sampling for Delegated Generative Quantum Machine Learning](https://arxiv.org/abs/2610.13211) | 该论文提出了一种盲化且可验证的采样协议，用于安全委托生成式量子机器学习模型的推理阶段，并扩展了可验证的委托变分量子算法以覆盖训练阶段的委托。 |
| [^133] | [How Much Computation Does Every Forecast Need? Structured Block Skipping in Two Time-Series Foundation Models](https://arxiv.org/abs/2610.13208) | 该研究表明时间序列基础模型可以在推理时按预测难度有选择地跳过整个 Transformer 块，在几乎不损失预测质量的前提下将推理延迟降低约 11%–13%。 |
| [^134] | [Rethinking Relative Policy Differences Transfer of On-Policy Distillation](https://arxiv.org/abs/2610.13205) | 该论文提出了一个统一框架，将专家与参考策略之间的相对策略差异视为 KL 正则化策略优化的隐式奖励，从而突破了教师必须显著强于目标模型的限制，实现不同规模、训练阶段和代际模型之间的任意到任意在线策略蒸馏迁移。 |
| [^135] | [S$^3$R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding](https://arxiv.org/abs/2610.13202) | S³R提出自稀疏推测解码方法，用稀疏注意力起草token、全注意力验证并采用精确推测采样，在严格保持稠密策略采样分布（保证训练忠实性）的同时大幅加速长上下文RLVR的rollout生成。 |
| [^136] | [Does Learning Recurrent Networks Matter in Partially Observable RL?](https://arxiv.org/abs/2610.13197) | 该研究发现冻结（无需训练）的循环网络即可为部分可观测强化学习提供数百步的记忆能力，并在多样化的POMDP任务中取得与训练循环网络相当的性能，从而质疑了在部分可观测RL中学习循环网络的必要性。 |
| [^137] | [Prediction-Informed Landing Window Planning for UAV Landing on USV in High Sea States](https://arxiv.org/abs/2610.13190) | 本文提出一种预测信息驱动的分层控制框架，利用预训练模型显式预测无人艇姿态，从而识别着陆窗口并确定最佳触舰时机，实现高海况下无人机在无人艇上的自主着陆。 |
| [^138] | [WaSA: Wavelet Scattering Autoencoders for Unsupervised Offline Detection of Malicious Clients in Federated Learning](https://arxiv.org/abs/2610.13186) | 提出了完全无监督的离线检测框架WaSA，通过小波散射自编码器对客户端数据频谱摘要进行评分，无需攻击标签即可识别任意类型的联邦学习投毒攻击。 |
| [^139] | [Neurograf v1: An End-to-End Romanian-First Language Model Family](https://arxiv.org/abs/2610.13134) | 该论文提出了Neurograf v1——首个从零开始为罗马尼亚语这一低资源欧洲语言构建的端到端语言模型家族，涵盖635 GB语料库、分词器、五个基础模型、指令微调助手和嵌入模型，验证了低资源语言从零训练大模型的可行性。 |
| [^140] | [Looking Ahead and Back: Anticipatory Generative Modeling for Road-Network Trajectories](https://arxiv.org/abs/2610.13131) | 提出了一种新的图约束生成范式“预期性生成建模”，将路网轨迹生成从位置模拟转向行为建模，通过预期未来路线来解决路口转向选择问题，避免轨迹偏离路网。 |
| [^141] | [Fast Gradient-Based Approximation of Continuous Dynamic Time Warping](https://arxiv.org/abs/2610.13129) | 本文提出了 FastCDTW，一种高效的连续动态时间规整（CDTW）近似求解器，证明了其在温和条件下的可微性，并针对分段线性规整函数提供精确闭式解、针对平滑规整函数提供无偏蒙特卡洛估计两种计算策略，大幅降低了计算成本。 |
| [^142] | [Why Heavy-Tailed Weight Spectra Predict Model Quality](https://arxiv.org/abs/2610.13123) | 该论文通过概率框架推导出仅基于DNN最后一层权重的边际似然，首次从统计学角度解释了为什么训练权重矩阵中的重尾谱分布能够有效预测模型质量。 |
| [^143] | [Continually Post-training Small Models via Synthetic Self-distillation](https://arxiv.org/abs/2610.13113) | 该论文提出合成自蒸馏方法，通过让略大的同族对齐模型生成同分布的合成示范，并结合离线策略监督与拒绝采样，使缺乏强上下文学习能力的小模型也能在持续后训练中学习新能力的同时保留已有知识。 |
| [^144] | [Does Location Matter? A Black-Box Causal Audit Reveals Model-Dependent Audio Grounding in LALM Hallucination](https://arxiv.org/abs/2610.13112) | 该研究提出一种黑盒因果审计方法，通过CLAP相似度定位并用谱门控移除音频证据区域，发现定位消融在控制时长后稳定优于随机移除，但其幻觉检测效果存在模型依赖性，且先前空白波形对比法的表面优势实为时长混淆所致。 |
| [^145] | [Scaling RNA Foundation Models through Iterative Refinement](https://arxiv.org/abs/2610.13106) | Eco-RNA是一个8000万参数的RNA基础模型，通过学习式迭代精化使推理深度可调，在固定参数下增加迭代次数即可显著提升RNA适应性预测性能。 |
| [^146] | [Can Failed Trajectories Reveal Where Agents Go Wrong? Agentic Failure Attribution Without Step-Level Labels](https://arxiv.org/abs/2610.13105) | 提出 SOAP 框架，通过谱评分与注意力引导传播，仅从无步骤标签的失败轨迹中定位 LLM 智能体导致失败的关键步骤，无需人工标注或成功轨迹参考集。 |
| [^147] | [The Limits of Query Monitoring against Distributed Model Extraction](https://arxiv.org/abs/2610.13083) | 该研究基于百万条真实对话数据表明，仅凭查询监控检测分布式模型提取的能力从根本上受普通用户协同行为所限，只有当提取活动比同一周内最独特的正常用户群体更加独特时才能被发现。 |
| [^148] | [Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison](https://arxiv.org/abs/2610.13082) | 该研究基于AMI会议语料库199个配对情境，从五个维度对人类项目经理与Llama 3.1 8B在协作设计会议中的引导行为进行计算比较，发现人类引导在情境语义距离和共情/缓和语词汇密度上更高，且后者主要由缓和语词汇驱动。 |
| [^149] | [Double-Take: Improving Capacity, Binding, and Selectivity In Long-Context Within Bidirectional Memory](https://arxiv.org/abs/2610.13073) | 提出 Double-Take 方法，通过多步非固定更新和跨位置机制，修复大块测试时训练中单步写入累积干扰限制容量、以及自关联写入无法形成跨位置绑定这两大基本失败，从而提升长上下文记忆的容量、绑定与选择性。 |
| [^150] | [CB-Orchestrator: Adaptive Workflow Optimization for LLM Agents via Contextual Bandits](https://arxiv.org/abs/2610.13064) | 提出CB-Orchestrator自适应框架，将工作流生成与选择解耦，通过进化搜索构建多样化候选工作流池，并将工作流选择建模为上下文老虎机问题，实现样本高效学习，兼顾性能与成本效益。 |
| [^151] | [CoBe: A Benchmark For Conversational Counterfactual Text Editing](https://arxiv.org/abs/2610.13057) | 该论文提出了CoBe基准，用于评估模型的对话式反事实文本编辑能力，要求模型通过因果理解判断假设性变化会影响的范围，在修改文本的同时保留未受因果影响的事实，而非仅生成统计上相关联的文本。 |
| [^152] | [Space Over Statistic, Retraining Over Ranking: Auditing Training Labels Under Label-Only Backdoors](https://arxiv.org/abs/2610.13051) | 该研究通过受控实验证明，防御仅标签后门攻击（如FLIP）的成败关键在于特征空间而非检测方法本身——同一个最近邻检测在冻结的DINOv2自监督特征上AUROC达0.967，而在投毒模型自身特征上仅0.546，并且应以重训练后的效果而非样本排序结果来评判防御方法。 |
| [^153] | [Lightweight Time-Frequency Vibration Sensing for Consumer Device Condition Monitoring](https://arxiv.org/abs/2610.13040) | 本文提出一种轻量级AI振动感知框架，通过CWT和HHT将一维振动信号转换为二维时频表示，并借助MobileNetV3-Small等轻量级深度学习模型进行分析，实现消费类旋转设备在真实干扰下的智能状态监测与预测性维护。 |
| [^154] | [Towards Easier-to-Tune Training Algorithms: New Results from AlgoPerf](https://arxiv.org/abs/2610.13036) | 在更新版AlgoPerf V1基准测试的自调优规则下，Schedule-Free AdamW在13个提交算法中排名第一，同时分析表明矩阵预条件和双循环方法虽样本效率优异，但需更激进的计算优化才能在实际运行时间上具竞争力。 |
| [^155] | [Target-Constrained Pooling for Trajectory Extrapolation](https://arxiv.org/abs/2610.13034) | 提出目标约束池化（TCP）方法，利用已完成的相关轨迹从两种互补池化信号中估计轨迹终点并修正自由预测，在五个领域的25,245个目标上将平均RMSE降低35.8%至42.1%。 |
| [^156] | [Low-Rank Kolmogorov-Arnold Generator Flows for Interpretable Operator Learning](https://arxiv.org/abs/2610.13028) | 提出KAFlow方法，用由一元Kolmogorov-Arnold函数构成、秩受系数函数个数约束的低秩符号来学习物理算子的生成元，从而将算子学习的代价从随分辨率平方增长降下来，并直接获得可解释的闭合形式。 |
| [^157] | [Trajectory-Aware Policy Optimization: Closing the State Distribution Gap in Long-Horizon LLM RL](https://arxiv.org/abs/2610.13020) | 提出轨迹感知策略优化（TAPO），通过将理论推导的状态分布修正项融入强化学习目标，并配合几何均值方差缩减与长度感知动态裁剪，弥合长程 LLM 强化学习中的状态分布偏移，解决训练不稳定与崩溃问题。 |
| [^158] | [Generalized Proximal Policy Optimization: Unifying LLM RL Algorithms from the Perspective of Divergence](https://arxiv.org/abs/2610.13015) | 该论文提出广义近端策略优化（GPPO），基于带单调性能改进保证的广义 f-散度信赖域理论，从散度几何视角统一理解和设计大语言模型强化学习算法，克服了 PPO 固定截断机制无法自适应策略偏移的缺陷。 |
| [^159] | [Evidential Multimodal Transformers with Dempster-Shafer Fusion for Emotion Recognition](https://arxiv.org/abs/2610.13013) | 该论文提出EMT-DS模型，利用Dirichlet分布将预测建模为主观意见，并通过Dempster组合规则融合各模态证据，将模态间的矛盾显式量化为冲突、区分认知不确定性与偶然不确定性，从而避免了Softmax方法在传感器数据损坏或矛盾时仍给出过高置信度的问题，提升了多模态情感识别的可靠性与鲁棒性。 |
| [^160] | [Scaling Empowered Agents](https://arxiv.org/abs/2610.13002) | 该论文提出了一种简单可扩展的赋能度最大化方法——将赋能度视为某状态下可执行技能的数量，引导智能体到达多种技能的良好起点状态——从而首次在大规模环境中实现了理论预测的赋能度涌现特性，如收集资源、改造环境、生存，以及在 Crafter 中建造房屋。 |
| [^161] | [What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training](https://arxiv.org/abs/2610.12995) | 提出CIG-SFT信息论框架，用条件信息增益而非模型兼容性来决定SFT应学习哪些词元，从而提升大语言模型后训练的泛化能力。 |
| [^162] | [Gaussian-Width Bounds for Offline-to-Online Linear Bandits](https://arxiv.org/abs/2610.12990) | 该论文利用高斯向量Lipschitz函数的集中性改进了对奖励预测误差的控制，将部分覆盖离线数据下线性赌博机的遗憾界从 $\widetilde O(\sqrt{dkT})$ 提升至最优的 $\widetilde O(k\sqrt T)$，并给出了匹配的上下界。 |
| [^163] | [Forecasting Corpus-Sampling Disagreement in Masked Text Reconstruction](https://arxiv.org/abs/2610.12987) | 该论文推导出一个一阶公式，可仅凭参考语料库预测两个独立语料库上训练的掩码文本重建预测器之间的期望分歧，并发现语料库扩大四倍仅使分歧降低5–40%，远低于1/n缩放律的预期，其根源在于罕见上下文的平滑效应。 |
| [^164] | [Scan-Parallel Variational Inference for event-driven Latent Jump Models](https://arxiv.org/abs/2610.12986) | 提出潜在跳跃模型（LJMs），通过泊松事件驱动的可学习瞬时跳跃来显式建模不规则时间序列中未观测冲击的时间与幅度，并配套开发了扫描并行变分推断方法用于高效训练。 |
| [^165] | [NucleoNet: A Nuclear-Aware Determinant Ansatz for Multi-Geometry Neural Variational Monte Carlo](https://arxiv.org/abs/2610.12985) | NucleoNet 提出一种核感知的神经行列式拟设，将电子与原子核纳入统一的共享骨干网络中联合演化，使单一模型可跨多个核几何构型联合优化，从而以更少的参数准确描绘分子势能面。 |
| [^166] | [Diffusion Precipitation Generation with Medium-range Weather Forecasting](https://arxiv.org/abs/2610.12976) | 该论文提出“动力学-降水解耦”即插即用框架，利用以大气状态和卫星-雨量计降水估计为条件的扩散生成器，将冻结天气基础模型的预报映射为与观测对齐的每日降水，从而克服再分析数据带来的区域性和强度相关偏差。 |
| [^167] | [Minions: A Dynamic and Distributed Virtual GPU Cloud Built on Idle Apple Silicon Devices](https://arxiv.org/abs/2610.12970) | 本文提出Minions系统，将闲置的Apple Silicon Mac设备聚合为动态分布式虚拟GPU云，用于执行大规模机器学习推理任务，无需额外采购GPU硬件。 |
| [^168] | [DataCupid: Stable Matching Repair for Training Data Acquisition with Delayed Feedback](https://arxiv.org/abs/2610.12960) | 该论文提出剩余负载修复（RLR）方法，通过将可靠的结果证书与批量匹配调度相结合，解决训练数据获取中带延迟反馈的稳定匹配修复问题，并证明其发布轮次界限和单次修复耦合能保证认证时间、观测发布量及检查点时间均不增加。 |
| [^169] | [JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents](https://arxiv.org/abs/2610.12957) | 本文提出JoyInside-VLM-Agent，一个通过产品条件化轨迹构建、闭环沙盒数据生成与分阶段策略学习（含新颖的DPER强化学习方法）训练的ReAct式多模态具身智能体，能够将多模态观察与用户请求转化为可执行动作，并适应产品约束、目标变化与执行反馈。 |
| [^170] | [From Imbalance to Evidence: Deep Nonparametric Testing for Complex Data under Severe Sample Imbalance](https://arxiv.org/abs/2610.12946) | 提出增强型不平衡感知深度分类器双样本检验，创新性地将严重样本不平衡从分类器训练的不利因素转化为可重复利用的检验证据来源，从而在高不平衡情形下实现对图等复杂数据更可靠的分布比较。 |
| [^171] | [Sparse Dictionary Learning for Measure-Valued Data via Linearized Optimal Transport: Efficient Algorithms and Sample Complexity](https://arxiv.org/abs/2610.12924) | 该论文提出一个基于线性化最优传输的稀疏字典学习框架，将概率测度嵌入希尔伯特空间并通过稀疏自编码器学习数据自适应字典，实现对高光谱图像、点云及大语言模型文档嵌入等测度值数据的高效可解释表示，同时给出以原子类度量复杂度刻画的泛化界和样本复杂度理论保证。 |
| [^172] | [Beyond Second-Order Information: Paid Logistic Regression with Noisy Features](https://arxiv.org/abs/2610.12919) | 该文提出带付费噪声特征的在线逻辑回归问题，证明与付费线性回归不同，仅凭协方差这类二阶信息已不足以刻画最优损失——相同协方差的噪声分布可产生不同损失、更低方差甚至导致更高损失，因此需要超越二阶信息的新结构刻画。 |
| [^173] | [Controlled-Minimax: Stabilizing Adversarial Optimization via Control Theory](https://arxiv.org/abs/2610.12915) | 本文将极小极大优化的训练过程建模为二阶常微分方程，揭示了Hessian矩阵特征结构导致非收敛的根本原因，并提出Controlled-Minimax框架，通过设计反馈控制器实现了与Hessian谱无关的局部指数稳定性理论保证。 |
| [^174] | [BLEND: Balancing Personalization vs. Generalization in Federated Vision-Language Models](https://arxiv.org/abs/2610.12907) | BLEND框架通过结合客户端共享的全局视觉与文本适配器、每个客户端专属的个性化视觉投影适配器，以及融合全局与个性化输出的新型损失函数，在联邦视觉语言模型中同时实现了个性化与泛化能力的提升。 |
| [^175] | [GIST: Generator-state-Informed Supervision for Tabular Foundation Models](https://arxiv.org/abs/2610.12904) | 提出GIST方法，利用生成器中间状态在条件标签分布上对查询损失进行平均，从而消除表格基础模型预训练中的采样变异性，并在不改变架构和推理的情况下提升TabArena与TALENT基准上的性能。 |
| [^176] | [Rethinking the Roles of Diffusion and Decoding: Predictive-State Flows for Parallel Language Generation](https://arxiv.org/abs/2610.12898) | 该论文提出预测态流（PSF），重新划分扩散先验与解码器的生成职责，让整流流先验仅生成基于前文的高层预测态，再由块因果解码器并行完成词元级选择，从而实现并行语言生成。 |
| [^177] | [Generalized Determinantal Point Processes for Attraction and Repulsion with Distinct Interaction Kernels](https://arxiv.org/abs/2610.12894) | 提出行列式比点过程（DRPP），利用两个对称正定核行列式之比分别建模排斥与吸引，在保持特征可解释性的同时，借助辅助变量表示实现高效MCMC采样与最大似然学习。 |
| [^178] | [Personalized Federated Direct Preference Optimization via LoRA Adapters](https://arxiv.org/abs/2610.12885) | 提出pFedDPO方法，让每个客户端在通过FedAvg同步的共享全局基础模型上保留本地LoRA适配器以捕获个性化偏好，从而将FedDPO理论界中的梯度多样性项κ²替换为更小的残余异构性项κ²_eff。 |
| [^179] | [CoGameReg: Cooperative-Game-Based Collaborative Semi-Supervised Regression for Seismic Reservoir Property Prediction](https://arxiv.org/abs/2610.12884) | 提出了基于合作博弈的半监督回归框架CoGameReg，通过三重博弈Shapley机制为异构回归器分配非负贡献权重，利用贡献加权伪标签在井标签稀疏条件下实现更准确的地震储层物性预测。 |
| [^180] | [HART: Hybrid Attention-RWKV Time series Foundation Model](https://arxiv.org/abs/2610.12883) | 本文发现RWKV-7在非自回归多步预测中因“占位符空转衰减（PID）”机制导致历史信息被削弱而产生长时程退化，并据此提出了融合注意力机制与RWKV的混合时间序列基础模型HART来解决该问题。 |
| [^181] | [When Solvers Get Moody: Quantifying and Predicting Performance Variability in Mixed Integer Programming](https://arxiv.org/abs/2610.12876) | 本文提出加权变异性分数（WVS）来量化MIP求解器在重复运行中的性能波动，并利用静态特征和根节点级特征构建预测器来预测这种变异性，同时首次证明了开放实例neos-5045105-creuse的最优性。 |
| [^182] | [Robust Langevin Sampling under State-Dependent Heavy-Tailed Noise with Applications to LLM Fine-Tuning](https://arxiv.org/abs/2610.12865) | 该论文提出R-LMC、A-LMC和RA-LMC三种朗之万采样算法，通过引入广义光滑性和状态依赖重尾噪声假设，放松了传统SG-LMC对全局Lipschitz梯度与有界噪声方差的要求，并将其应用于大语言模型的贝叶斯微调。 |
| [^183] | [Universal Bounds for Self-Normalized Kullback-Leibler Estimation](https://arxiv.org/abs/2610.12851) | 本文证明了基于归一化重要性权重的KL散度估计器存在通用有限样本界——其期望超过目标-提议KL散度至多 $\log 2$，且无需重要性权重有界或高阶矩有限，并将该结果应用于生成模型的Soft Best-of-$N$测试时对齐与相对熵编码。 |
| [^184] | [Two-way Guidance: Simultaneous Learning of Summary Statistics for Generative Bayes](https://arxiv.org/abs/2610.12850) | 提出“双向引导”机制，通过汇总统计量与后验分布的同步联合学习，使基于模拟的贝叶斯推断在观测规模不断增大时仍能有效恢复集中化的后验分布。 |
| [^185] | [Reinforcement Learning via Fixed-point Flow Matching](https://arxiv.org/abs/2610.12848) | 本文提出了一种无需数据的不动点流匹配（FFM）方法，通过将actor更新表述为不动点迭代、以critic的动作梯度作为回归目标，实现了强化学习中流策略的训练与微调。 |
| [^186] | [When Generative Models Act as Simulators](https://arxiv.org/abs/2610.12847) | 该论文研究生成模型通过“可编辑性”充当模拟器的条件，提出两种在模型潜在状态与模拟器状态之间学习映射的方法（探针解码加引导、逆映射），并发现逆映射方法即使在引导失败时也能成功。 |
| [^187] | [APEX: Asynchronous Prefix Denoising for Extreme Compression](https://arxiv.org/abs/2610.12843) | APEX通过将分层潜在空间与按通道组的异步前缀去噪相结合，解决了极致潜在压缩中的“重建-生成”两难困境，在64倍和128倍压缩下全面超越现有方法。 |
| [^188] | [COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering](https://arxiv.org/abs/2610.12837) | 提出COFFEQA方法，通过反事实引导为少样本蒸馏中教师决策边界附近的不确定样本提供额外监督，从而提升大语言模型问答任务的蒸馏效果。 |
| [^189] | [Cross-Behavior Policy Optimization: Unified Learning from Offline Demonstrations and On- and Off-Policy Rollouts via a Shared Regression Target](https://arxiv.org/abs/2610.12830) | CBPO通过固定参考的共享回归目标，将离线演示、在策略与离策略采样统一到同一回归损失和回放缓冲区中，无需分别协调SFT与RL目标、重要性加权或评论家网络即可实现数据的持续复用与统一学习。 |
| [^190] | [Scalable Learning of Determinantal Point Processes with Kernel-Structured Interactions](https://arxiv.org/abs/2610.12829) | 该论文提出一种利用非线性核随机特征进行随机优化的可扩展行列式点过程学习方法，通过全秩交互建模与替代函数设计，突破了结构约束的局限，显著改善了表达能力与计算可扩展性之间的权衡，并以极高的参数效率获得了更好的泛化性能。 |
| [^191] | [TreeHCA: Hindsight-Guided Exploration and Credit Assignment for Multi-Turn LLM Agents](https://arxiv.org/abs/2610.12828) | TreeHCA利用参考答案似然的变化引导探索，构建推理与工具交互树，并通过逆后见之明重要性校正和归一化树回备将终端结果传播为轮级优势，为多轮LLM智能体实现细粒度的后见之明信用分配，克服了GRPO对所有token分配相同优势的缺陷。 |
| [^192] | [Inverse Synthesis of Shock Time Series from Shock Response Spectra Using Ensembles of Conditional Variational Autoencoders](https://arxiv.org/abs/2610.12820) | 该论文提出一种基于条件变分自编码器集成（涵盖不同小波母函数与分解层级）的逆向合成方法，通过循环解码器预测多尺度小波系数重构波形，为同一目标冲击响应谱生成多个候选冲击波形，其谱误差低于先前的直接时域CVAE模型。 |
| [^193] | [Deploying RDMA for Large-Scale Inference and Training Workloads](https://arxiv.org/abs/2610.12815) | 本文总结了字节跳动在大规模推理与训练系统中部署基于 RDMA 的软件通信层 ByteExpress 的三条经验：高效 RDMA 需要跨应用与传输层的缓冲区管理、可扩展通信需要对连接与接收容量和在途数据的协同管理、服务可用性需要在传输层之上实现故障恢复。 |
| [^194] | [VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails](https://arxiv.org/abs/2610.12813) | 提出VEPA框架，将形式化防护栏的确定性执行能力与从部署经验中持续学习和演化安全策略的机制相结合，为LLM智能体提供终身的、可验证的自适应安全防护。 |
| [^195] | [The Expressive Power of Random-Feature MPNNs Across Graph Sizes](https://arxiv.org/abs/2610.12809) | 该论文提出了一个总体层面的规模一致逼近分析框架，通过将连续随机特征量化为有限种颜色，证明了具有图规模无关Lipschitz界的MPNN不具备跨图规模的通用逼近能力。 |
| [^196] | [Trainable Quantum-Native Language Models: Dissipative Memory, Capacity Limits, and Hardware Validation](https://arxiv.org/abs/2610.12806) | 该论文提出了以小型量子寄存器为隐藏状态、通过可学习词元条件量子信道转移并直接由玻恩规则测量输出概率的量子原生语言模型，发现可学习的词元条件耗散门控是实现超越单词元有用记忆的关键，其性能优于参数量匹配的单词元对照模型但不及双词元对照模型，从而揭示了此类量子语言模型的容量极限并通过硬件实验加以验证。 |
| [^197] | [Polyconvex neural-network surrogates with learned paired directional features for anisotropic hyperelastic homogenization](https://arxiv.org/abs/2610.12801) | 提出了一种具有可学习配对方向特征的多凸物理增强神经网络，能够在构造上保证客观性、多凸性与增长条件等力学一致性的前提下，高效逼近各向异性超弹性均匀化的有效响应。 |
| [^198] | [Meta-Reflective Evolution Effectively Improves Production AI Systems](https://arxiv.org/abs/2610.12794) | 提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。 |
| [^199] | [Coordinate-Free Automatic Differentiation](https://arxiv.org/abs/2610.12776) | 该论文提出一个与坐标（基）无关的统一导数算子，可完全由现有自动微分原语实现，从而以可复用、可组合的方式表达旋转、体积形式、曲率等几何微分运算，并与 JAX 的批处理、编译和硬件加速兼容。 |
| [^200] | [Credit Attribution in Timestamped Soccer Networks Using Multivariate Hawkes Processes](https://arxiv.org/abs/2610.12775) | 该论文提出利用多元霍克斯过程对足球比赛中带时间戳的传球和射门序列建模，量化每次传球对后续进攻事件的贡献，从而将进球功劳合理分配给进攻组织中的多名球员，弥补了传统助攻统计的局限性。 |
| [^201] | [REDaCT: Collapse-Free Unlearning in Large Reasoning Models](https://arxiv.org/abs/2610.12769) | REDaCT是一种面向大型推理模型的推理时遗忘学习框架，通过泄漏检测器与纠错嵌入注入机制，在不引发推理坍塌的前提下从思维链和最终答案中移除遗忘集信息。 |
| [^202] | [Provenance-grouped protocol for evaluating machine learning in perovskite photovoltaics](https://arxiv.org/abs/2610.12766) | 该论文指出钙钛矿光伏机器学习评估中随机数据划分会因海量重复数据导致信息泄漏，提出基于机构来源分组的严格评估协议，并揭示树集成模型仅擅长记忆熟悉配方、而成分感知神经架构在真正未知场景下泛化能力更强的规律。 |
| [^203] | [NV-Tesseract-AD: Curriculum-Enhanced Conditional Diffusion for Multivariate Time-Series Anomaly Detection](https://arxiv.org/abs/2610.12754) | 提出NV-Tesseract-AD框架，通过在掩码严重程度与扩散噪声水平两个互补轴上同时实施由易到难的三阶段双课程学习，增强了基于Transformer的条件扩散模型在多元时间序列异常检测中的优化稳定性与残差判别能力，并实现了无需逐数据集调优的严格零样本迁移。 |
| [^204] | [Trajectory Prediction for Autonomous Driving in Road Construction Zones Using Human-in-the-Loop Driving Data](https://arxiv.org/abs/2610.12747) | 该论文构建了首个针对道路施工区域的多模态驾驶数据集WZDrive-Data，并提出了多模态自车轨迹预测器WZDrive，有效解决了施工区域车道布局临时变化导致的车辆轨迹难以预测的难题。 |
| [^205] | [Static Gradient Attribution Underperforms a Density-Matched Random Mask on Loss-Based Forgetting Within LoRA B-Matrix](https://arxiv.org/abs/2610.12744) | 该研究揭示了梯度掩码与随机掩码比较中存在的参数池混淆问题（零初始化LoRA下A的梯度为零），并发现即使限定在同一B矩阵参数池内进行公平比较，密度匹配的均匀随机掩码在基于损失的遗忘指标上仍始终优于静态梯度归因。 |
| [^206] | [Neural expectation-maximization and meta-models](https://arxiv.org/abs/2610.12741) | 该论文将Transformer解释为最大乘积环状信念传播与期望最大化混合概率推断的可微近似，并据此提出“元模型”这一新设计范式，由输入生成实例特定的参数。 |
| [^207] | [Minimax Rates for Estimating Powered Sliced Wasserstein Functionals](https://arxiv.org/abs/2610.12740) | 本文建立了幂次切片Wasserstein泛函两样本估计的极小极大速率——当 $r<2p$ 时为 $N^{-(r-p)/r}$ 阶，当 $r>2p$ 时为 $N^{-1/2}$ 阶——并证明自然的代入估计量在两种情形下均达到速率最优。 |
| [^208] | [KITE: A Knowledge-Injected Temporal Encoder for Time-Series Event Forecasting](https://arxiv.org/abs/2610.12738) | 该论文提出KITE架构，通过可切换的协变量和注意力偏置两种机制将工程师掌握的语义先验知识注入自监督事件预测器，无需架构搜索即可适配各类先验，在标签稀缺场景下显著提升罕见事件预测性能（域内最高提升47%、跨域提升53%）。 |
| [^209] | [Kernelized Parameter Sensitivity Atlases for Amortized Conditional Interpretability](https://arxiv.org/abs/2610.12737) | 该论文提出 KPSA，通过经验神经正切核的分块分解构建参数组敏感性的核化图谱，用轻量级核函数摊销式地为任意新输入估计敏感性，从而以远低于逐样本反向传播的成本实现输入条件化的模型可解释性。 |
| [^210] | [StereoEngine: Scaling Stereo Bimanual Robot Data without Stereo Cameras](https://arxiv.org/abs/2610.12734) | StereoEngine通过“深度估计+重投影与视频扩散模型补全将单目演示提升为立体视频对”和“仿真立体渲染联合训练”两种互补方法合成逼真的机器人操作立体数据，无需立体相机即可大规模扩展双手机器人立体数据集。 |
| [^211] | [A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges](https://arxiv.org/abs/2610.12728) | 该论文提出一个范畴论统一框架，将数百种参数高效微调方法视为从小参数空间到权重空间的光滑映射，并通过像、纤维与基变换三个不变量统一刻画方法的表达能力、训练动力学以及量化、合并、重启等操作。 |
| [^212] | [Zebra3D: Efficient Hybrid Foundation Models For Scalable 3D Reconstruction](https://arxiv.org/abs/2610.12727) | Zebra3D提出一种逐层稀疏-线性混合注意力框架，通过为容忍的早期阶段分配线性注意力、为敏感的中后期阶段分配瓦片稀疏注意力，并利用闭式仿射拟合实现免训练转换，从而在无需从头训练的情况下显著提升预训练3D基础模型在大规模视图集上的重建效率。 |
| [^213] | [Spatial Context or Memory? Sizing the Inputs of Neural Surrogates for Partially Observed Fields](https://arxiv.org/abs/2610.12723) | 本文以贝叶斯最优误差为基准，首次精确量化了部分观测场神经代理模型中空间传感器窗口与历史快照长度之间的权衡，证明对于具有局部动力学的稀疏含噪观测场，扩大空间窗口比增加记忆更有效，且仅读取空间窗口的模型即可接近最优预测性能。 |
| [^214] | [A Distributional Framework for Behavioral Characterization and Risk Management of Large Language Models](https://arxiv.org/abs/2610.12718) | 该论文提出一个基于蒙特卡洛模拟构建经验响应分布的框架，通过语义干预量化分布变化，将大语言模型的随机行为刻画与模型风险管理、治理以及黑盒模型监控联系起来。 |
| [^215] | [Byzantine-Resilient Federated Large Language Model via Hypergraph Message Passing](https://arxiv.org/abs/2610.12711) | 本文提出HMP-GAE框架，将低秩模型更新表示为超图节点，通过迭代消息传递利用多智能体间的高阶语义关系来识别并抑制恶意更新，从而在不访问私有数据的前提下防御拜占庭攻击、保障联邦大语言模型聚合的可信性。 |
| [^216] | [PVBRF-ID: Private, Verifiable and Byzantine-Robust Federated Intrusion Detection for Oilfield Enterprise Networks](https://arxiv.org/abs/2610.12704) | PVBRF-ID提出了一种双服务器联邦学习协议，通过站点对更新进行秘密共享、服务器联合检验范数与余弦相似度并结合承诺、Merkle根和同态审计，在不暴露更新的前提下同时实现隐私保护、公开可验证性与抗拜占庭鲁棒性，在油田入侵检测任务中达到与FedAvg相当的精度并能抵御投毒攻击。 |
| [^217] | [Bridging Human and Model Attention: Explainability Analysis of CNN, Mamba, and ViT Architectures with Gaze-Based Validation](https://arxiv.org/abs/2610.12703) | 该论文的核心贡献是首次以眼动追踪数据为真值，通过标准化归一化流程和三种互补的相似度指标，系统量化了CNN、ViT和Vision Mamba三种架构的模型注意力与人类视觉注意力在多个视觉任务上的一致性。 |
| [^218] | [Product Monte Carlo: Expectation Under a Product of Densities](https://arxiv.org/abs/2610.12702) | 该论文提出乘积蒙特卡罗（PMC）算法，通过将难以处理的密度乘积期望转化为全贝叶斯回归问题，仅利用各分量密度的蒙特卡罗样本即可计算任意期望，并具备一致性保证与不确定性量化能力。 |
| [^219] | [Primal-Dual Flow Matching for Sample-Wise Constrained Generation](https://arxiv.org/abs/2610.12699) | 提出原始-对偶流匹配（PDFM），在流匹配框架中引入原始-对偶优化，仅依靠二值约束满足反馈即可直接控制逐样本约束的满足概率，无需可微距离、投影或凸性等约束集结构假设。 |
| [^220] | [Toward Maximizing Algebraic Connectivity via Density-Dependent Backbone Construction and Reinforcement Learning Completion](https://arxiv.org/abs/2610.12697) | 提出一种密度依赖的两阶段方法——先依据密度选择完全多部图或凯莱图作为骨架，再用强化学习补全剩余边——在固定顶点和边数条件下高效构建代数连通度最大的图拓扑，性能优于现有方法且运行时间可扩展性良好。 |
| [^221] | [Critical Mass: Structure-Disjoint Splits Drift With the Libraries They Are Built On](https://arxiv.org/abs/2610.12694) | 论文指出分子机器学习中用于分布外评估的结构不相交划分标准（基于MCES距离阈值10的单链接聚类）是建立在29,000分子规模上的一次性惯例，随分子库增长到22万余分子后已无法计算验证，为此提出结合可采纳界与并查集跳过的优化算法，将计算量降低2,259倍，使该评估标准首次可在现代公共分子库规模上被检验。 |
| [^222] | [Expert-level detection of low-voltage intervals in EEG as a robust biomarker of neonatal encephalopathy](https://arxiv.org/abs/2610.12680) | 该论文提出仅由振幅和持续时间定义的低电压间期（LVIs）作为新生儿脑电图不连续性的全新指标，并开发了基于ConvNeXt的AI检测模型，在异常分级、HIE分级、癫痫预测等多项任务中验证了LVIs作为新生儿脑病稳健生物标志物的专家级检测能力。 |
| [^223] | [Amortised Inference through One-Step Implicit Sampling](https://arxiv.org/abs/2610.12679) | 提出单步隐式采样器（OSIS），利用短时MCMC动力学定义局部目标使建模分布成为MCMC核的不动点，从而实现高维贝叶斯后验分布的单步摊销采样。 |
| [^224] | [Recurrent Harness for Agentic Continual Learning](https://arxiv.org/abs/2610.12668) | 提出 ReHarness，一种循环式智能体持续学习框架，在严格在线流式协议下，将每个实例处理后的轨迹作为反馈向后传播，修订该实例对应的反思状态，实现前向读写与后向修订相结合的在线适应。 |
| [^225] | [MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis](https://arxiv.org/abs/2610.12659) | 该论文发现主流RLVR数学语料库中约88%的问题因过于简单或过难而无法为目标策略提供有效学习信号（即零优势陷阱），并提出MagicBeanStalk方法，通过递归数据合成生成既蕴含难题所需推理操作、又难度适中的有效训练问题。 |
| [^226] | [Data-driven Optimal Sensor Placement for Aerosol Monitoring in the UK](https://arxiv.org/abs/2610.12658) | 该研究提出一个将卷积高斯神经过程与带惰性评估和缓存的遗传优化相结合的数据驱动传感器布置框架，使英国气溶胶监测的计算时间减少60%、联合负对数似然提升约22%，为环境观测系统的数据驱动扩展提供了计算高效的方案。 |
| [^227] | [Kronecker Product Attention: Memory-Efficient KV Compression via Structured Factorization](https://arxiv.org/abs/2610.12657) | 提出克罗内克积注意力（KPA），将键值激活表示为紧凑的词元相关矩阵因子的克罗内克积之和，严格推广了张量积注意力（TPA），在保持表达能力的同时实现了内存高效的KV缓存压缩。 |
| [^228] | [SynAct: Synergistic Active Feedback Learning for LLM Agents](https://arxiv.org/abs/2610.12651) | 提出SynAct主动学习框架，将专家注意力视为稀缺约束资源，利用动作与最终结果之间的互信息对决策排序，在最能改变结果的关键决策点精准请求专家反馈并提炼为可迁移规则，从而弥补LLM智能体的知识缺口。 |
| [^229] | [Machine Learning and Time-Series Foundation Models for Day-Ahead Electricity Forecasting in a University Building](https://arxiv.org/abs/2610.12638) | 该研究在一栋大学建筑上对统计、机器学习、循环神经网络和预训练TimesFM等日前电力预测模型进行了统一基准测试，发现带协变量的TimesFM 3.0总体误差最低（平均绝对误差较每周季节性朴素预测降低约43%），而XGBoost仍具竞争力。 |
| [^230] | [STAPLE: Static Sparse Upcycling of Pretrained Checkpoints via Embedding Modules](https://arxiv.org/abs/2610.12599) | STAPLE 通过将基于 token 索引的嵌入作为残差路径加入预训练模型，在保留原有 FFN 计算的同时以可扩展的方式学习互补的 token 特定信息，从而实现高效的静态稀疏升级。 |
| [^231] | [SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire](https://arxiv.org/abs/2610.12596) | SpireBench让20个大语言模型完整游玩《杀戮尖塔》以评估长时程战略推理能力，发现前沿模型虽达到普通人类水平但远逊于专家，且随着游戏进程延长差距不断扩大，所有模型都误判长线风险，倾向保守保血而非牺牲血量换取成长。 |
| [^232] | [Realism VS Accuracy: Event Sequence Forecasting from a Generative Modeling Perspective](https://arxiv.org/abs/2610.12586) | 该论文提出从条件生成建模视角研究事件序列多步预测，构建了同时评估预测准确性与分布真实性的基准，揭示了两者之间的根本性权衡，并提出Wasserstein BaryBooster算法，将真实性强的扩散模型样本聚合到最先进的预测器中。 |
| [^233] | [DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models](https://arxiv.org/abs/2610.12572) | DiffusionART 通过将每个提示词的奖励聚合为共享的奖励提升方向以在构造上消除前向-反向一致性力，并结合信任域掩码的二次更新实现无需 EMA 调参的漂移自适应恢复，从而实现扩散模型在线强化学习的高效、稳定与可扩展。 |
| [^234] | [An all-transistor physical-learning transformer can be trained to perform in-context learning](https://arxiv.org/abs/2610.12571) | 该论文提出了一种全晶体管物理学习Transformer，通过在双电路中引入额外电路元件恢复双向信息流，并利用亚阈值MOSFET的指数电流特性构建注意力机制，使物理系统首次能够被训练执行上下文学习任务。 |
| [^235] | [Sample-Adaptive Zeroth-Order Post-Training](https://arxiv.org/abs/2610.12566) | 提出样本自适应零阶优化（SAZO），通过在线估计样本重要性并自适应分配查询预算来降低估计方差、加速大语言模型的零阶微调收敛，且无需额外模型评估、仅增加 O(N) 的标量状态。 |
| [^236] | [Fly-by-Code: Embodied Coding Agents for Aerial Manipulation with Active Visual and Physical Feedback](https://arxiv.org/abs/2610.12549) | 该论文提出主动反馈框架，通过主动“观察与探测”获取缺失的视觉和物理证据，解决了空中操作中部分可观测性和交互鲁棒性受限导致失败原因无法诊断的问题，从而支持具身编码智能体有效地改进程序策略。 |
| [^237] | [AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits](https://arxiv.org/abs/2610.12545) | AutoIndexer提出了一种模拟文本编辑器的“编辑链”训练与解码机制，使因果自回归模型能够通过插入、替换和删除操作灵活地事后修改已生成的输出，从而支持非因果顺序的灵活解码。 |
| [^238] | [Transformer-Based Masked Diffusion Models Provably Learn Distributions with Latent Structure](https://arxiv.org/abs/2610.12544) | 本文首次为基于Transformer的掩码扩散模型建立了端到端统计保证，证明其能以由潜在结构（隐马尔可夫树混合模型）决定的样本复杂度学习结构化离散分布，突破了以往理论中 $V^L$ 维度灾难的限制。 |
| [^239] | [Equal-Budget Monte Carlo Tree Search versus Randomized-Restart List Scheduling for DAG Workflows: A Controlled Equal-Budget Computational Study](https://arxiv.org/abs/2610.12542) | 在严格控制的等计算预算实验中，简单的随机重启列表调度方法在DAG工作流调度上比蒙特卡洛树搜索（MCTS）好2.88%，表明更复杂的搜索方法并不必然优于简单基线。 |
| [^240] | [Enriched Holomorphic Neural Networks for 2D and 3D Laplace Problems](https://arxiv.org/abs/2610.12541) | 本文提出将全纯神经网络与可训练的奇异增强函数相结合，利用构造性调和性免去内部残差、仅凭边界数据求解含角点奇异性的二维与三维拉普拉斯问题，精度较现有最优方法提升数个数量级至约1.7-2.1倍。 |
| [^241] | [Efficient Learning of Survival Analysis Models Under Outcome-Dependent Truncation and Censoring](https://arxiv.org/abs/2610.12538) | 该论文证明了在结果依赖的截断和右删失数据下，当基线风险函数已知时，非凸的Cox回归可以通过一阶优化方法高效学习，并为需要联合学习基线风险函数的情形构建了可识别且设定正确的模型。 |
| [^242] | [Market Impact Estimation via Causal Foundation Models](https://arxiv.org/abs/2610.12537) | 本文提出一种在合成市场先验上训练的连续时间因果基础模型（PFN），用于无偏估计市场冲击，证明经典价格-交易流回归方法在知情信息流混淆下会高估市场冲击（最高达68%），而该模型即使在高混淆情形下也能逼近无偏真值。 |
| [^243] | [Structural and Representational Agency in LLM-Driven Timbre Generation and Interpolation](https://arxiv.org/abs/2610.12532) | 本研究通过对一个实际运行的LLM驱动音色界面进行测量，首次证明用户真实可控的声音空间（可精确刻画为维度随放置声音数递增的zonotope）与界面中由语言模型自我报告的位置（误差约为该空间宽度的三倍）是可分离的，揭示了LLM创意界面中“实际能动性”与“表征能动性”之间的脱节。 |
| [^244] | [CacheScope: Exact-Compute Allocation for Training-Free Diffusion Acceleration](https://arxiv.org/abs/2610.12530) | 提出CacheScope，通过以输入位移归一化探测输出变化来在线识别高风险缓存复用步骤，并选择性回退到完整去噪计算，实现无需训练的扩散模型加速。 |
| [^245] | [GlucoPRISM: Protocol-Supervised Pretraining for Cross-Cohort Glucose Transfer](https://arxiv.org/abs/2610.12528) | GlucoPRISM 提出协议监督的预训练方法，将 CGM 表示划分为对日期和传感器均不变的特质带、仅随日期变化的状态带以及跟踪设备的传感器带，使部署时无需目标数据即可通过固定投影去除损害跨队列迁移的测量成分。 |
| [^246] | [Behind-the-Meter Photovoltaic Visibility from Net Smart Meters: Deployment Regimes and Operational Insights](https://arxiv.org/abs/2610.12527) | 本文提出四种部署模式来系统评估基于深度学习的净智能电表光伏分解方法，揭示了模型在已知用户、新用户、未来时期和新区域等不同场景下的性能差异，为电力运营商提供了实际部署的关键运营洞察。 |
| [^247] | [MolP2SBench: Do LLMs Really Reason from Molecular Properties to Structure?](https://arxiv.org/abs/2610.12522) | 提出MolP2SBench基准，通过遮蔽分子骨架、性质打乱与反事实测试系统评估大语言模型从分子性质到结构的推理能力，发现模型虽能利用物理性质预测结构，但未必真正依赖正确的性质-结构配对。 |
| [^248] | [CircuitTree: Composing Circuits into A Syllable Transcriber](https://arxiv.org/abs/2610.12520) | 该论文提出CircuitTree数据结构及贪心密度缩放组合算法，将39个从HuBERT中提取的音素分类电路高效组合成单个模型，实现了任何单个电路都不具备的音节转写能力，以64.8%的准确率转写开音节，仅比完全不共享电路的上限低1.2个百分点。 |
| [^249] | [Looking Under the Streetlight: Evaluation in Generative Molecular Dynamics](https://arxiv.org/abs/2610.12518) | 这篇立场论文指出，生成式分子动力学领域的评估实践因基准真值获取代价高昂而结构性偏离了科学目标，导致评估依赖便利的替代指标，并据此提出更贴合领域追求的具体可观测量、报告标准和计算与实验基准设计方案。 |
| [^250] | [CORAL-KV: Compact Cross-Layer Prediction and Residual Coding for Autoregressive Video KV Caches](https://arxiv.org/abs/2610.12514) | 提出Coral-KV框架，利用视频模型KV缓存中强烈的跨层依赖关系，通过紧凑的块对角加低秩预测器跨层预测键值并对预测残差进行量化，从而在自回归长视频生成中高效压缩KV缓存并抑制重建误差随视频增长而累积的问题。 |
| [^251] | [EarLog: Opportunistic Ear Physiology Sensing toward Personalized Hearing Risk Awareness](https://arxiv.org/abs/2610.12501) | EarLog将耳机播放内容本身转化为机会式探测信号，通过内容条件化残差感知在日常聆听中识别微弱的声反射，实现对个性化听力损伤风险的实时感知。 |
| [^252] | [Acoustic Biomarkers for Alzheimer's Disease Detection from Spontaneous Speech](https://arxiv.org/abs/2610.12500) | 本研究证明从“饼干失窃”图片描述等自发言语中提取的韵律、频谱和时域声学特征，结合机器学习分类器，可以有效区分阿尔茨海默病患者与健康人群，为AD的早期筛查提供了一种具有临床意义的语音生物标志物方法。 |
| [^253] | [An Input-Dependent Readout Phase for Resonant Sequence Models](https://arxiv.org/abs/2610.12496) | 本文将 ResonatorLM 中每个头固定的读取相位改为当前 token 的函数，仅在读取环节引入输入依赖性，在保持精确卷积和仅增加 0.2% 参数的同时提升了模型能力，并分析了未训练振荡器组的相位可编码时间但线性重建能力受限（约 16 个 token）。 |
| [^254] | [Kairos: Toward a Byte-Level Foundation Model for Every Modality](https://arxiv.org/abs/2610.12495) | Kairos 提出了一种在统一字节空间上处理文本、图像、音频、视频、激光雷达等多模态的小型基础模型架构，融合门控 DeltaNet 线性注意力、滑动窗口注意力、注意力-残差聚合、稀疏 MoE 和多尺度字节级编解码器，并通过单机上的小规模过拟合收敛测试验证该组合能否有效优化。 |
| [^255] | [Algebraic Integer Quantization for Machine Learning Applications](https://arxiv.org/abs/2610.12494) | 本文提出代数整数数格式（AINF）及面向大语言模型推理的代数整数量化（AIQ），该框架可将乘累加运算化归为纯整数运算以实现无误差计算，并允许针对具体应用灵活选择代数整数，在性能与精度上均具有竞争力。 |
| [^256] | [Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance](https://arxiv.org/abs/2610.12493) | 本文是一篇立场论文，指出当前面向特征重要性的可解释AI忠实性度量存在结构性错位，并强调XAI实际上更多扮演调试工具的角色而非建立信任的手段。 |
| [^257] | [TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers](https://arxiv.org/abs/2610.12474) | 本文首次对十亿参数规模的TTS扩散变换器开展4比特权重与激活（W4A4）训练后量化研究，揭示了量化主要损害说话人音色而非可懂度这一现象及其来源，并提出感知引导的量化修复方法。 |
| [^258] | [Neuromorphic Satellite Intelligence: A Formal Framework for Event-Driven Architectures, Spike-Based Algorithms, and Quantitative Energy-Accuracy Trade-offs](https://arxiv.org/abs/2610.12471) | 本文建立了卫星脉冲神经计算的形式化数学模型，提出三种原创脉冲算法（用于姿态估计的SKCF、用于健康监测的TCAD、用于认知卫星链路的SAMC），并引入定量的能耗-精度-延迟（EAL）权衡曲面。 |
| [^259] | [On the estimation and validity of AI time horizons---a statistical look at the METR plot](https://arxiv.org/abs/2610.12466) | 本文用样条函数和项目反应理论重新估计了METR的AI时间视野，发现人类时间与AI难度的关系并非全局线性（在2至30分钟区间近乎平坦），因此同样10倍的提升，从3分钟到30分钟远比从30分钟到5小时容易实现。 |
| [^260] | [asdex: Automatic Sparse Differentiation in JAX](https://arxiv.org/abs/2610.12336) | 本文提出了 JAX 中的自动稀疏微分工具 asdex，通过稀疏模式检测、图着色、压缩微分与解压缩四个步骤，利用雅可比矩阵的稀疏结构，使自动微分传递次数与问题维度无关，从而大幅提升计算效率。 |
| [^261] | [Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read](https://arxiv.org/abs/2610.12133) | 该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。 |
| [^262] | [Efficient quadratic entropy with distance sketches](https://arxiv.org/abs/2610.11976) | 本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。 |
| [^263] | [Beyond QAOA: A Review of AI and Quantum Computing for Adaptive Combinatorial Optimization](https://arxiv.org/abs/2610.11759) | 本综述提出“自适应量子优化”的概念，按被学习的决策类型系统梳理了AI赋能量子优化、量子赋能AI优化和AI-量子协同优化三种范式，并通过对119篇论文的分析发现AI辅助量子优化的证据显著强于量子辅助AI优化。 |
| [^264] | [Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts](https://arxiv.org/abs/2610.11575) | 提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。 |
| [^265] | [Sera: Semantic Representation Aggregation for Reliable and Interpretable Battery Health Forecasting](https://arxiv.org/abs/2610.11567) | 提出语义表示聚合框架 Sera，通过融合基于规则的知识与大语言模型解读所构建的退化语义表示来补充时序建模，从而实现更可靠、更可解释的电池健康状态预测。 |
| [^266] | [When to Intervene? State-Aware Sparse Manipulation in Federated Reinforcement Learning](https://arxiv.org/abs/2610.11523) | 该论文首次将“何时干预”确立为联邦强化学习拜占庭攻击的一个独立攻击维度，提出了利用本地策略不确定性选择稀疏干预状态并施加包络约束行为引导的V-BSA攻击方法，证明干预时机的选择会实质性影响攻击效果。 |
| [^267] | [Estimating great expectations under autoregressive language models with potentials](https://arxiv.org/abs/2610.11399) | 本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。 |
| [^268] | [Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers](https://arxiv.org/abs/2610.11315) | 该论文提出一个统一框架，将低秩辅助的W4A4训练后量化建模为耦合校准问题，通过“收缩海森矩阵”对已被低秩分量捕获的残差误差进行折扣，并结合激活噪声代理抑制激活量化误差，从而仅用秩4即可实现多模态扩散Transformer的高精度4比特量化。 |
| [^269] | [Dissecting Representation Structure in Vision Transformers: A Rigorous Architectural Study](https://arxiv.org/abs/2610.11205) | 该论文首次严谨分析了视觉Transformer跨架构尺度的特征信息，发现初始化时的特征坍缩问题并提出缓解方案，同时证明熵和最小特征值可作为泛化预测的可靠指标，用以指导高效的ViT设计。 |
| [^270] | [Ranking Prior Alignment for Credit Risk Modeling: When Do External Priors Matter?](https://arxiv.org/abs/2610.11146) | 提出了一种模型无关的“排序先验对齐”框架，通过温度缩放的KL散度损失将来自领域专家、教师模型或大语言模型的外部排序先验统一蒸馏进神经或树类评分模型，以缓解标注数据稀缺的冷启动信贷评分难题。 |
| [^271] | [Dynamics as Code: On Model Compression via Dynamic System](https://arxiv.org/abs/2610.11115) | 该论文提出“动力学即代码”的新型模型压缩范式，证明了在丢番图条件下，无理缠绕的有限轨迹可构成权重空间上的ε-网，从而以可预测的方式将状态分辨率、解压误差与压缩比联系起来。 |
| [^272] | [Measuring and Mitigating Solution Mode Collapse in RLVR](https://arxiv.org/abs/2610.11064) | 本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。 |
| [^273] | [Democratizing MoE inference on commodity GPUs with CoMoE](https://arxiv.org/abs/2610.09424) | CoMoE通过新颖的以主机为中心的路由机制，将主机转变为主动路由枢纽，解决了消费级GPU互连带宽受限的通信瓶颈，使MoE模型推理能够在低成本的商品级GPU上高效部署。 |
| [^274] | [Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data](https://arxiv.org/abs/2610.09117) | 提出Workhorse框架，通过在同一套无需重定向的真人演示数据上分别训练视觉规划器和强化学习全身跟踪器，并以互相模仿对方部署误差的方式增强训练数据，使Unitree G1人形机器人能从第一人称视觉实现鲁棒的全身移动操作并具备抗干扰恢复能力。 |
| [^275] | [Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization](https://arxiv.org/abs/2610.08969) | 该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。 |
| [^276] | [TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models](https://arxiv.org/abs/2610.07767) | TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。 |
| [^277] | [Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$](https://arxiv.org/abs/2610.07623) | 该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。 |
| [^278] | [BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation](https://arxiv.org/abs/2610.07594) | BiGym 2.0是一个针对宇树G1人形机器人20个家庭操作任务的全身控制基准测试平台，实验表明视觉-语言-动作微调方法总体表现最佳，而编码智能体开发的程序优于所有演示驱动的强化学习基线。 |
| [^279] | [TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning](https://arxiv.org/abs/2610.07043) | 提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。 |
| [^280] | [Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning](https://arxiv.org/abs/2610.05745) | 该研究揭示量化VLA模型在环境扰动下虽保持分布内性能却变得脆弱，发现动作差异集中于少数脆弱状态且教师指导在高差异状态提升泛化、在低差异状态反而损害泛化，据此提出面向脆弱状态的选择性调优方法以恢复量化模型的泛化能力。 |
| [^281] | [Green-Routed Neural Operators: Physics Determines Where the Network Reads](https://arxiv.org/abs/2610.05337) | 提出格林路由神经算子（GRNO），利用控制方程决定潜在特征的采样位置，在五个PDE系统中的四个上取得最低的自回归预测误差。 |
| [^282] | [Evolving Hybrid Quantum-Classical Architectures for Image Classification](https://arxiv.org/abs/2610.03220) | 该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。 |
| [^283] | [Hamiltonian locality testing and certification do not achieve the Heisenberg limit](https://arxiv.org/abs/2610.03205) | 该论文证明了哈密顿量k-局域性测试和哈密顿量认证问题都需要Ω(1/ε²)的总演化时间，首次为哈密顿量学习与测试中的自然问题建立了排除1/ε海森堡极限标度的下界。 |
| [^284] | [Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination](https://arxiv.org/abs/2610.02626) | 提出IG-VLA框架，通过在视觉表征空间中直接想象任务相关的未来场景演变，并将其内化为紧凑的场景要旨标记，使VLA模型在推理时无需昂贵的显式未来推演即可获得未来推理带来的动作预测增益，实现高效的机器人操作。 |
| [^285] | [Selection-Based Structured Reasoning: Toward Efficient Multimodal Search Agents](https://arxiv.org/abs/2610.01892) | 提出基于选择的结构化推理框架SSR，将开放式推理生成转化为从预定义推理候选中的似然选择，并借助共享上下文KV缓存实现并行评分，从而大幅提升小模型多模态搜索智能体的推理效率。 |
| [^286] | [Exploring More, Reasoning Better: Stepwise Risk-Sensitive GRPO for Diffusion Language Models](https://arxiv.org/abs/2610.00661) | 提出StepRS-GRPO方法，通过在去噪状态间动态调整群组优势变换的风险系数，使扩散语言模型在数学推理任务上同时提升pass@1准确率、pass@k覆盖率和答案多样性。 |
| [^287] | [Useful to Whom? Sample Value Is Defined Only Relative to the Learner](https://arxiv.org/abs/2610.00221) | 该论文通过冻结样本子集、只改变学习器的受控实验证明，核心集选择中“易样本优先”与“几何覆盖”准则之间的交叉边界并非由数据固定，而是随学习器容量（如ResNet-18宽度、输入网格分辨率）而移动，说明样本的价值只有相对于特定学习器才有意义。 |
| [^288] | [Sharp Statistical Rates for Asynchronous TD Learning with Markovian Data](https://arxiv.org/abs/2609.38880) | 该论文为基于单条马尔可夫轨迹的表格型TD学习建立了精确的末次迭代样本复杂度界 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$，且该界对常数步长与递减步长调度均成立。 |
| [^289] | [OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime](https://arxiv.org/abs/2609.38345) | 提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。 |
| [^290] | [Robust LassoNet: Enhancing Feature Selection in Neural Networks via Robust Loss Functions](https://arxiv.org/abs/2609.38263) | 本文提出Robust LassoNet，通过在LassoNet中引入Huber、Cauchy、Tukey双平方等鲁棒损失函数，在保留原始优化框架的同时，显著提升了神经网络在数据污染情况下的特征选择准确性与预测性能。 |
| [^291] | [Graph World Models for Constrained Epidemic Policy Planning](https://arxiv.org/abs/2609.35545) | EpiMind是一个用于受限疫情政策规划的图世界模型框架，通过图因子化循环状态空间模型和图-时间ADMM实现跨区域协调干预，在保证共享资源预算可行性的同时将入院预测误差降低29%。 |
| [^292] | [ThinkNet: Evolutionary Deep Ensembles for Efficient Motor-Imagery EEG Classification](https://arxiv.org/abs/2609.33967) | ThinkNet提出了一种验证控制的框架，通过仅训练数据归一化、验证引导的进化搜索和验证门控推理，在严格防止测试受试者信息泄露的前提下，实现面向新用户的高效、紧凑的运动想象脑电分类。 |
| [^293] | [LLM4Trust: Exploring the Capabilities of Large Language Models for Trust Evaluation](https://arxiv.org/abs/2609.33521) | 本文提出首个系统性基准框架LLM4Trust，通过构建信任图建模五种基本信任属性，并在九种提示方法下评估八个代表性大语言模型的属性理解能力，系统探索了LLM在信任评估中的潜力。 |
| [^294] | [Benchmarking Attention for Tabular Foundation Models](https://arxiv.org/abs/2609.31306) | 该论文针对表格基础模型中独特的二维行/列注意力模式构建了可复现的基准测试，并在多种注意力后端上系统评估其性能，填补了高效注意力研究在二维表格场景中的空白。 |
| [^295] | [RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking](https://arxiv.org/abs/2609.23457) | 本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。 |
| [^296] | [Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies](https://arxiv.org/abs/2609.19844) | 论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。 |
| [^297] | [A multimodal large language model for evidence-based autism spectrum disorder screening](https://arxiv.org/abs/2609.16464) | 本研究提出ASDchat——一个采用双分支架构的多模态大语言模型，能够基于视频、音频和对话进行自闭症筛查，同时生成与ADOS-2临床标准对齐的可追溯、带时间戳的行为证据，在中国27个站点1,035名参与者的数据集上达到0.953的AUC。 |
| [^298] | [Predicting Social Media Engagement using Machine Learning](https://arxiv.org/abs/2609.16082) | 本研究通过提取家具公司Facebook图片帖子的视觉、文本和时间特征，并运用随机森林、LightGBM等机器学习模型来预测社交媒体互动度。 |
| [^299] | [Nonsmooth Optimization via Orthogonalized Momentum](https://arxiv.org/abs/2609.13677) | 本文首次在与反向传播兼容的广义导数框架下证明：对于任意固定动量因子β∈[0,1)，正交化动量优化器Muon在凸Lipschitz非光滑目标上几乎从任何初始化出发都可能无法收敛到全局最优解，揭示了其超越光滑优化的根本局限。 |
| [^300] | [What Makes Adversarial Examples Transfer Across Deepfake Detectors?](https://arxiv.org/abs/2609.10002) | 该研究通过对涵盖六种骨干网络、两种预训练方案和五种训练数据配置的60个深度伪造检测器进行受控评估，发现当攻击者的源模型与目标模型共享相同的骨干网络、架构家族、预训练方案或训练数据时，黑盒对抗迁移攻击的成功率显著更高，且这种兼容性效应因攻击方法而异。 |
| [^301] | [SUN: Reaching for Novelty in Reinforcement Learning](https://arxiv.org/abs/2609.08642) | 本文提出基于后继价值函数的SUN指标，可同时衡量目标的新颖性与可达性，该可达性感知的目标选择框架能无缝集成到任意离策略强化学习算法中，并具有恢复计数奖励、约束命中概率和拒绝不可达目标的理论保证。 |
| [^302] | [Protocol effects on feature-based hardware-Trojan detection across Trust-Hub families](https://arxiv.org/abs/2609.07199) | 该论文发现基于特征的硬件木马检测性能严重依赖测试集划分协议——当测试门电路来自训练中已见过的宿主电路时性能被高估，而按宿主家族留出时 F1/AP 会从约 0.91/0.98 骤降至约 0.46/0.58。 |
| [^303] | [Differentiable Hybrid Modelling for Learning and Optimising Chemical Transport Processes from Experimental Data](https://arxiv.org/abs/2609.04011) | 提出一个通用的可微分混合建模框架，将JAX有限体积群体平衡求解器与可学习的神经网络组件相结合，能够从真实实验数据中同时发现本构定律并拟合初始条件，用于学习和优化化学传输过程。 |
| [^304] | [DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving](https://arxiv.org/abs/2608.30386) | 提出衰减感知状态压缩（DASC），通过从模型权重中推导不同头和通道的“保留视野”来压缩混合线性注意力模型的持久状态检查点，从而缓解内存压力、减少缓存驱逐和重复预填充。 |
| [^305] | [Preference Elicitation for Policy Optimization and Application to Aligning Heart Transplantation with Human Values](https://arxiv.org/abs/2608.28620) | 提出一种两阶段的线性效用偏好引出算法，通过成对比较学习切割平面并可证明收敛到用户效用函数，从而直接从分配结果学习效用函数用于策略优化，并成功应用于心脏移植分配的价值对齐问题。 |
| [^306] | [Enhancing Tabular Learners with Context-Aware Semantic Embeddings](https://arxiv.org/abs/2608.03565) | 提出CASE框架，通过用代表性行样本预填充基于Gemma 3的表格语言模型的KV缓存来为表格行嵌入提供数据集级语义上下文，从而弥合大语言模型语义理解与表格学习器统计能力之间的差距，解决文本特征的语义歧义问题。 |
| [^307] | [DRIFT: Derailing Denoising Trajectories of Flow-Matching VLAs with Adversarial Patch Attack](https://arxiv.org/abs/2608.03207) | 本文提出DRIFT通用对抗性贴片攻击，揭示流匹配VLA模型（如pi0）的鲁棒性只是错觉——仅对第一个去噪步骤施加攻击即可高效地使现成策略失效。 |
| [^308] | [Particle-based Generalised Stochastic Optimisation](https://arxiv.org/abs/2608.02844) | 本文提出了针对不可解梯度损失函数的基于扩散的广义随机粒子优化框架，通过平均场动力学及其交互粒子近似统一了多种现有算法，证明了指数收敛性与非渐近误差界，并据此构建了高阶Langevin和动量优化器。 |
| [^309] | [The Fourth Quadrant: A Stylized View of Benign Misfitting](https://arxiv.org/abs/2608.01032) | 该论文揭示了“良性失配”现象——在单尖峰线性回归模型的某个训练集大小范围内，任何泛化良好的线性预测器对训练数据的拟合都必须比零预测器更差，从而刻画了泛化能力与训练拟合之间一种反直觉的权衡关系。 |
| [^310] | [First-Order In-Context Shapley: Scalable Data Auditing for LLM Alignment and Evaluation](https://arxiv.org/abs/2607.22766) | 该论文提出了一种仅需推理、无需重训练的一阶上下文内 Shapley 数据估值方法，可在 O(k) 次前向传播内完成对大语言模型对齐数据的可扩展审计，以 19 倍更少的计算量实现了与蒙特卡洛上下文内 Shapley 在全部目标样本上一致的估值效果。 |
| [^311] | [Learning from All-Failed Groups via First-Visit Coverage in Long-Horizon Agentic Tasks](https://arxiv.org/abs/2607.22724) | 提出CovGPO方法，通过在整组rollout全部失败时引入首次访问观测覆盖，为欠采样的状态改变动作提供学习信号，从而打破长程智能体任务中采样不平衡导致的自我延续信用陷阱。 |
| [^312] | [MIRROR: Learning from the Other View for Multi-Modal Reasoning](https://arxiv.org/abs/2607.21552) | 本文提出MIRROR强化学习方法，通过在同一问题的不同模态视角中选取模型表现最佳者作为教师，实现跨视角自蒸馏，从而提升多模态推理能力。 |
| [^313] | [Medical foundation models converge less under label supervision](https://arxiv.org/abs/2607.20274) | 该论文通过对75万余张医学图像、18个公开模型和101个自训练模型的大规模分析发现，医学基础模型——尤其是经过标签监督训练的模型——对相同图像的表征彼此差异巨大，其近邻重叠度甚至接近随机初始化的水平，因此不同基础模型的嵌入不可互换使用。 |
| [^314] | [Planning with Transformers: Chain of Computation and Structured Context Windows](https://arxiv.org/abs/2607.17710) | 提出计算链（COC）计算架构，通过将基于Transformer的语言模型置于迭代循环中并利用结构化上下文窗口作为外部记忆，以弥合LLMs理论计算能力与实际规划性能之间的差距。 |
| [^315] | [Equality, Equity, and Causality in Fairness Research: A Commentary on Cheng (2026)](https://arxiv.org/abs/2607.17679) | 该评论在Cheng（2026）对心理测量学与AI/ML公平性系统比较的基础上，进一步探讨了平等与公平的概念区别以及因果性在公平性研究中的作用，为跨学科公平性研究的未来方向提供了指引。 |
| [^316] | [H-OPD: Confidence Aware Heterogeneous Multi-Teacher Multimodal On-policy Distillation](https://arxiv.org/abs/2607.02592) | 提出 H-OPD 框架，以置信度感知的 token 级异构多教师仲裁取代传统静态的任务/样本级教师路由，并通过视觉到语言的描述迁移让纯文本教师获取关键视觉语义，从而显著提升多模态推理在线蒸馏的效果。 |
| [^317] | [Improving Certified Robustness via Adversarial Distillation](https://arxiv.org/abs/2606.31653) | 本文提出了结合对抗蒸馏的认证训练方法AD-CERT，通过将对抗训练目标与IBP松弛界相结合，在标准准确率与认证鲁棒性之间取得更好的权衡。 |
| [^318] | [FlowAWR: Online Adaptive Flow Reinforcement via Advantage-Weighted Rectification](https://arxiv.org/abs/2606.30376) | FlowAWR提出将连续生成流模型的策略优化重构为朝向理论最优速度场的监督回归范式，通过优势加权矫正实现幅度感知的自适应优化，解决了现有方法中训练-推理不一致及固定幅度校正的局限。 |
| [^319] | [Hierarchical Experimentalist Agents](https://arxiv.org/abs/2606.29315) | 提出分层实验智能体（HExA），一种以实验为中心的上下文内自我改进框架，使智能体能够通过主动实验收集查询相关证据、发现未知环境的一般规律，并将交互经验转化为可复用技能。 |
| [^320] | [Erased, but Not Gone: Output Forgetting Is Not True Forgetting](https://arxiv.org/abs/2606.25001) | 该论文提出以重训练模型群体作为参考分布的“重训练一致的表示遗忘”评估范式，揭示现有机器遗忘方法虽在输出层面看似遗忘成功，但在表示空间中仍与理想遗忘存在结构性偏差，即输出遗忘并不等于真正的遗忘。 |
| [^321] | [RoboMME-Interference: Benchmarking Robot Memory Under Interference](https://arxiv.org/abs/2606.22338) | 本文提出了RoboMME-Interference，一个跨会话机器人记忆基准，用于评估长上下文和干扰下VLA模型的记忆性能，并发现现有感知记忆变体在干扰下表现有限。 |
| [^322] | [Decoupling the Declarative from the Procedural in Vision-Language-Action Models](https://arxiv.org/abs/2606.21496) | 该论文指出现有视觉-语言-动作模型无法将陈述性知识（概念与实体语义）与程序性知识（操作方法）解耦，导致其对空间、语义和任务变化十分脆弱、难以泛化到训练物体之外，这是通用机器人技能迁移的根本瓶颈。 |
| [^323] | [VLA-FAIL: Efficient Task Failure Detection for Finetuned Vision-Language-Action Models](https://arxiv.org/abs/2606.21386) | 提出VLA-FAIL框架，通过最后一层马氏距离（LLMD）和动作块一致性（ACC）两个轻量级检测器，在无需失败数据和最小开销的情况下高效检测视觉-语言-动作模型的运行时任务失败。 |
| [^324] | [Urban Power Grid Topology and Hierarchy Identification from Open Data](https://arxiv.org/abs/2606.21352) | 本文提出一个基于开放数据的框架，通过整合公共基础设施数据与地理空间机器学习聚类建筑负荷，实现了从高压输电到单个建筑连接的城市电网完整拓扑与层级识别。 |
| [^325] | [TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability](https://arxiv.org/abs/2606.19821) | 提出TelcoAgent框架，通过自动构建3GPP知识图谱、零样本时间序列基础模型预测和领域可解释推理三个组件，实现了无需站点特定训练的准确、可扩展且可解释的5G多KPM预测。 |
| [^326] | [Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence](https://arxiv.org/abs/2606.19386) | 本文勘误并重新审视了先前报告的“状态饱和陷阱”（实为动作间 dt=0 导致指数衰减从未运行的纯累加器结果），揭示出以采样时间校准与以墙钟时间校准的监测器动态在固定速率流上等价、但在动作间隔差异巨大的智能体流上根本不同，因此墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制。 |
| [^327] | [Scalable Circuit Learning for Interpreting Large Language Models](https://arxiv.org/abs/2606.16939) | 提出基于稀疏线性回归的CircuitLasso方法，以极低的计算成本学习到与最先进干预方法结构准确性相当的电路，并能高效揭示人类可理解的SAE语义特征在大语言模型中的传播及其对预测的影响。 |
| [^328] | [PACT: Privileged Trace Co-Training for Multi-Turn Tool-Use Agents](https://arxiv.org/abs/2606.16215) | PACT框架提出将专家轨迹仅作为训练时的优化信号而非推理提示，通过轨迹条件化强化学习代理目标和带退火过程监督的组件感知SFT损失两个互补信号进行协同训练，同时保持推理时仅使用提示词生成，从而兼顾强化学习的推理一致性与监督微调的密集过程监督优势。 |
| [^329] | [Lesion-DDPM: Lesion-Enhanced 3D Diffusion for MS MRI Synthesis](https://arxiv.org/abs/2606.15457) | 提出Lesion-DDPM，一种结合多层次解剖掩码注入与病灶加权重建损失的3D条件扩散框架，能够在合成多发性硬化症FLAIR MRI时有效保留小而稀疏的病灶。 |
| [^330] | [Learning Transfers: Kan Extensions for Neural Invariants](https://arxiv.org/abs/2606.07627) | 本文通过范畴论中的Kan扩展形式化定义了表示迁移所需保持的结构，并提出了一种新的迁移差异度量，以任务改变所强制的不变量为目标进行评分。 |
| [^331] | [Agents' Last Exam](https://arxiv.org/abs/2606.05405) | 本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。 |
| [^332] | [The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show](https://arxiv.org/abs/2606.05328) | 该论文通过逆转扩散采样过程恢复模型内部轨迹，首次证明视频扩散模型内部确实线性编码了物理合理性信息，其可从扩散Transformer状态中以约81.27%的准确率解码出来，超越了专用表征学习基线。 |
| [^333] | [When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation](https://arxiv.org/abs/2606.03532) | 本研究发现在自我在线策略蒸馏中，教师更新的隔离期（更新之间完全冻结教师）而非教师年龄是实现稳定学习的关键结构属性，并提出诊断框架揭示了状态遗忘崩溃现象——短程最优的固定更新时间表在长程训练中会因时钟驱动的刷新将漂移的学生一次性不可逆地复制进教师而导致灾难性失败。 |
| [^334] | [A Unified Benchmark for Dynamic Medical Treatment Reinforcement Learning](https://arxiv.org/abs/2606.01028) | 该论文提出了MedGym——一个基于物理信息神经网络、在连续时间框架下由临床数据构建的可配置动态治疗推荐强化学习基准，用于评估强化学习方法处理不规则测量间隔、个体化治疗反应和测量点间安全性的能力。 |
| [^335] | [Unsupervised Identification and Removal of Spurious Correlations During Fine-Tuning](https://arxiv.org/abs/2605.27676) | 本文证明可以在无监督条件下从朴素LoRA微调的权重中识别出微调引入的伪相关潜在因素，并提出应消除伪相关本身而非潜在因素，以避免破坏预训练模型依赖该因素获得的有效任务信号。 |
| [^336] | [Is TabPFN the Silver Bullet for Insurance Pricing?](https://arxiv.org/abs/2605.22892) | 本文首次实证评估了TabPFN在机动车保险定价中的表现，发现它并非“灵丹妙药”——在索赔频率预测上落后于GLM且表现依赖上下文训练集规模，仅在索赔强度预测上具有竞争力。 |
| [^337] | [Do Deep Ensembles Actually Capture Uncertainty in Graph Neural Networks?](https://arxiv.org/abs/2605.22593) | 研究发现深度集成在图神经网络中并不能有效捕获认知不确定性，其边际改进仅来自平均化优化噪声，而独立训练的网络因“认知坍缩”现象收敛到过于相似的预测，缺乏集成方法所需的多样性。 |
| [^338] | [Stochastic MeanFlow Policies: One-Step Generative Control with Entropic Mirror Descent](https://arxiv.org/abs/2605.21282) | SMFP是一种将MeanFlow噪声到动作映射与高斯重参数化相结合的单步生成式策略，通过解析熵下界实现熵正则化镜像下降强化学习，在七个MuJoCo基准上优于高斯策略和其他生成式策略。 |
| [^339] | [SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front](https://arxiv.org/abs/2605.20619) | 该论文提出SURF方法，通过将权重变化时解在帕累托前沿上的移动速度积分得到的累积弧长归一化为CDF，并利用其反函数调控标量化权重的采样，从而实现对帕累托前沿的均匀覆盖。 |
| [^340] | [Quadratic Characterizations for Reachability Analysis of Neural Networks](https://arxiv.org/abs/2605.20482) | 本文提出一个通过凸二次规划局部生成候选二次不等式、再用平方和证书全局验证的框架，为二维平面上的标量关系构建可靠过近似的二次刻画，可直接应用于基于二次约束和积分二次约束的神经网络可达性分析。 |
| [^341] | [Cross-View Attention Fusion Net: A Prior-Guided Dual-View Representation Learning for Cardiac Output Estimation from Short-Term PPG Signals](https://arxiv.org/abs/2605.19666) | 提出跨视图注意力融合网络CVAF-Net，通过跨视图注意力将原始PPG的时间视图与特征序列图的先验引导视图相融合，实现从短期PPG信号准确估计心输出量。 |
| [^342] | [Sampling-Based Safe Reinforcement Learning](https://arxiv.org/abs/2605.19469) | 该论文提出基于采样的安全强化学习算法SBSRL，通过在有限动力学样本上联合施加约束来近似最坏情况优化，并结合认知不确定性约束的探索策略，提供了学习全过程的高概率安全保证和有限时间样本复杂度界，在仿真与真实机器人上实现了安全高效的探索。 |
| [^343] | [Stochastic Compositional Optimization via Hybrid Momentum Frank--Wolfe](https://arxiv.org/abs/2605.15350) | 该论文提出了混合动量随机Frank--Wolfe这一无投影算法，首次系统性地处理不可微的凸外层函数（如最大值、CVaR和范数）的随机复合优化问题，在有限方差噪声下对非凸目标达到K^{-1/4}收敛率、在模型下界条件下达到K^{-1/3}的末次迭代误差，并将分析扩展到无需截断的重尾噪声情形。 |
| [^344] | [Bayesian Nonparametric Mixed-Effect ODEs with Gaussian Processes](https://arxiv.org/abs/2605.13088) | 提出MEGPODE模型，利用高斯过程先验将每个受试者的ODE向量场分解为共享种群成分与个体偏差，并通过状态空间高斯过程和单一变分目标避免重复求解ODE，从而在稀疏不规则观测下实现灵活且稳健的非参数动力学建模。 |
| [^345] | [On the Approximation Complexity of Matrix Product Operator (MPO) Born Machines](https://arxiv.org/abs/2605.11471) | 本文证明了用多项式键维MPO玻恩机器近似任意连续目标是NP难的，但同时表明当损失诱导哈密顿量局部且具有常数谱隙时，满足对数Sobolev不等式的目标可获得带可证明KL保证的多项式键维MPO-BM近似，且多项式数量的局部分数查询足以维持这些保证。 |
| [^346] | [Vision2Code: A Multi-Domain Benchmark for Evaluating Image-to-Code Generation](https://arxiv.org/abs/2605.11307) | 提出了Vision2Code——一个覆盖15个数据集共539个样例、无需参考源代码的多领域图像到代码生成基准，通过执行生成代码并结合领域定制评分标准进行评估，其评分与人类评分的相关性显著优于通用评估指标。 |
| [^347] | [Model Capacity Determines Grokking through Competing Memorisation and Generalisation Speeds](https://arxiv.org/abs/2605.09724) | 该论文提出一个信息论解释框架，指出顿悟现象并非在模型容量足以记住训练集时立即发生，而是由记忆化速度与泛化速度这两个随参数量变化的可测量时间尺度的相对竞争所决定。 |
| [^348] | [FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](https://arxiv.org/abs/2605.09081) | FactoryNet是一个包含1.13亿数据点、覆盖7种机器形态和4项任务的大规模工业时间序列预训练数据集，通过统一的S-E-F-C控制理论模式使指令意图与实际响应可在不同机器间比较，为工业时间序列基础模型奠定了基础。 |
| [^349] | [UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence](https://arxiv.org/abs/2605.09040) | UxSID 通过语义 ID 与双层注意力机制构建语义组共享兴趣记忆，为超长用户行为序列建模提供兼顾效率与效果的第三条路径，实现 SOTA 性能并在大规模广告 A/B 测试中带来 0.337% 的收入提升。 |
| [^350] | [Reinforcement learning for inverse silhouette design and rapid laser cutting of compact parallelogram quad kirigami prototypes](https://arxiv.org/abs/2605.08098) | 本文提出RL-Kirigami框架，将最优传输条件流匹配与强化学习（GRPO）相结合，实现紧凑平行四边形四联剪纸的逆向轮廓设计，单个样本即达到94.2%的sIoU并优于求解器基线，可支持快速激光切割制作原型。 |
| [^351] | [Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph](https://arxiv.org/abs/2605.08037) | 本文提出 GraphDPO，将 DPO 从成对偏好比较推广到有向无环偏好图结构，充分利用多个 rollout 之间的传递性偏好关系进行聚合监督，从而实现更稳定、更充分的语言模型对齐。 |
| [^352] | [Convex Optimization of Spiking Neural Networks with Global and Approximation Guarantees](https://arxiv.org/abs/2605.08022) | 该论文将脉冲神经网络（包括LIF-SNN）的训练问题首次等价转化为凸优化问题，证明了零对偶间隙与全局最优保证，并提出具有高概率近似保证的随机化替代方法以解决凸规划变量指数级增长的问题。 |
| [^353] | [FactoryBench: Evaluating Industrial Machine Understanding](https://arxiv.org/abs/2605.07675) | FactoryBench基于珀尔因果阶梯构建了四个因果层级的问答基准，包含7万余个问答项，用于评估时间序列模型和大语言模型对工业机器人遥测数据的理解能力，零样本评测显示当前前沿大语言模型在结构化任务上不超过50%、在决策任务上不超过18%。 |
| [^354] | [Constrained Triple-Head Price Optimization (C3PO) Network for Discrete-Choice Models](https://arxiv.org/abs/2605.06941) | 提出了基于Transformer的约束三头价格优化网络C3PO，通过融合价格模仿学习、多任务收入预测、上下文弹性学习以及轻量级可微分约束层，在离散选择环境下实现无需重训练、可执行定价约束的跨领域实时定价优化。 |
| [^355] | [Playing the network backward: A Game Theoretic Attribution Framework](https://arxiv.org/abs/2605.06212) | 该论文提出将神经网络的反向归因方法（如梯度法和LRP）统一重构为扩展网络图上的双人博弈框架，使各类归因图成为特定均衡下博弈轨迹的积分投影，并可通过策略正则化、风险规避等博弈论概念定制期望的解释性质。 |
| [^356] | [Training Non-Differentiable Networks via Optimal Transport](https://arxiv.org/abs/2605.01928) | 本文提出PolyStep方法，通过最优传输理论仅利用前向传播即可训练具有跳跃性损失的非可微网络，解决了反向传播不适用的问题。 |
| [^357] | [Fairness-Guided DQN-Based TXOP Control for NR-U/Wi-Fi Coexistence](https://arxiv.org/abs/2605.00457) | 本文提出一种公平性引导的深度Q网络方法，通过根据每节点吞吐量比率动态调整NR-U的传输机会（TXOP）时长，有效改善NR-U与Wi-Fi在非授权频谱中共存的公平性与吞吐量性能。 |
| [^358] | [Occam's Razor is Only as Sharp as Your ELBO](https://arxiv.org/abs/2604.25984) | 本文证明在贝叶斯线性回归中，基于ELBO的超参数学习会因高斯近似后验协方差结构的不同而产生系统性偏差：对角或低秩加对角协方差会低估信噪比导致欠拟合，而低秩协方差会高估信噪比导致过拟合。 |
| [^359] | [Approximating Uniform Random Rotations by Two-Stage Structured Hadamard Rotations in High Dimensions](https://arxiv.org/abs/2604.23418) | 该论文为两阶段结构化Hadamard旋转逼近高维均匀随机旋转建立了首个非渐近理论保证，证明了坐标分布距离的 $\sqrt{\log d/d}$ 阶一致上界以及 $d^{-1/2}$ 阶下界。 |
| [^360] | [Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI](https://arxiv.org/abs/2604.08015) | 提出CATMIL训练目标，通过成分自适应Tversky加权和病灶级多示例学习两个辅助损失项，在不修改网络架构的前提下显著提升脑部MRI中小病灶的分割召回率。 |
| [^361] | [Koopman-Based Nonlinear Identification and Model Predictive Control of a Turbofan Engine](https://arxiv.org/abs/2604.01730) | 本文提出基于元启发式扩展动态模态分解的Koopman模型辨识方法，并据此开发了带扰动观测器的自适应Koopman模型预测控制器，实现涡扇发动机的多变量控制，且该单一模型可跨多种控制策略复用。 |
| [^362] | [MVNN: A Measure-Valued Neural Network for Learning McKean-Vlasov Dynamics from Particle Data](https://arxiv.org/abs/2604.00333) | 该论文提出一种测度值神经网络MVNN，可直接从粒子轨迹数据中学习McKean-Vlasov动力学中依赖测度的相互作用项，并提供了适定性、混沌传播以及万能逼近与定量逼近速率等理论保证。 |
| [^363] | [Mixture-Greedy for Online Generative Model Selection: Is UCB Necessary in Diversity-Aware Multi-Armed Bandits?](https://arxiv.org/abs/2603.21716) | 本文提出Mixture-Greedy策略，并发现在多样性感知的多臂老虎机生成模型选择问题中，UCB探索奖励并非必要——去除UCB反而收敛更快、样本效率更高、性能更好。 |
| [^364] | [Weight Clustering Reveals the Importance of Relative Rank in Large Language Models](https://arxiv.org/abs/2603.17917) | 利用标量 K 均值聚类将大语言模型权重压缩为 16-64 个共享值并可在不重训的情况下保持性能，并通过扰动实验揭示保持权重间的相对排序比保持精确数值更重要。 |
| [^365] | [Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](https://arxiv.org/abs/2603.07233) | 提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。 |
| [^366] | [On the Rate of Convergence of GD in Non-linear Neural Networks: An Adversarial Robustness Perspective](https://arxiv.org/abs/2603.02095) | 该论文在双神经元ReLU网络的最简二分类设置中证明，梯度下降虽能收敛到最优对抗鲁棒性边际，但收敛速度极慢（严格为 Θ(1/ln(t))），并首次建立了非线性模型中鲁棒性边际收敛速度的显式下界。 |
| [^367] | [Bridge Matching Sampler: Scalable Sampling via Generalized Fixed-Point Diffusion Matching](https://arxiv.org/abs/2603.00530) | 本文提出桥匹配采样器（BMS），将最小二乘匹配方法统一推广为基于Nelson关系的不动点迭代，以单一、稳定且可扩展的目标函数学习任意先验与目标分布间的随机传输映射，并通过阻尼正则化变体缓解模式崩溃、实现前所未有规模的采样。 |
| [^368] | [Beyond State-Wise Mirror Descent: Offline Policy Optimization with Parametric Policies](https://arxiv.org/abs/2602.23811) | 该工作将离线强化学习的理论保证从有限小动作空间扩展到大/连续动作空间上的参数化策略类，识别出将镜像下降推广至参数化策略时的核心难点“上下文耦合”，并据此提出了计算上高效（oracle高效）的独立参数化离线策略优化算法。 |
| [^369] | [Locally Private Parametric Methods for Change-Point Detection](https://arxiv.org/abs/2602.13619) | 本文研究了本地差分隐私下的参数化变点检测问题，建立了以Chernoff信息为根本限制的有限样本极小极大上下界，并据此设计了分别适用于低隐私和高隐私机制的最优本地隐私变点检测算法，从而精确量化了隐私保护带来的统计代价。 |
| [^370] | [Federated Learning of Nonlinear Temporal Dynamics with Graph Attention-based Cross-Client Interpretability](https://arxiv.org/abs/2602.13485) | 本文提出一种垂直联邦学习框架，在原始数据不可共享、客户端观测异构且各客户端模型固定不可重训的约束下，利用图注意力机制实现跨客户端的非线性时间动态一步预测，并具备跨客户端可解释性。 |
| [^371] | [ERIS: Enhancing the Privacy-Utility-Scalability Trade-Off in Federated Learning via Federated Shard Aggregation](https://arxiv.org/abs/2602.08617) | ERIS提出联邦分片聚合（FSA）机制，将客户端更新分割为互不重叠的分片并分散至多个客户端聚合器进行聚合，从而在十亿参数规模的联邦学习中同时改善隐私、可扩展性与模型效用之间的权衡。 |
| [^372] | [RanSOM: Second-Order Momentum with Randomized Scaling for Constrained and Unconstrained Optimization](https://arxiv.org/abs/2602.06824) | 该论文提出RanSOM方法，通过将优化步长随机化（指数或Beta分布），利用Stein型恒等式使新迭代点处单次Hessian-向量积（且其前向传播可与下一次梯度计算共享）成为梯度变化的无偏估计，从而在无需Lipschitz Hessian或额外前向反向传播的情况下，对约束（Frank-Wolfe）和无约束优化均达到最优的oracle复杂度。 |
| [^373] | [Fast Rates for Nonstationary Weighted Risk Minimization](https://arxiv.org/abs/2602.05742) | 本文将分布漂移下的超额风险分解为学习误差与漂移误差，对非平稳加权经验风险最小化建立了在任意权重类上一致成立的快速oracle界，并在结构突变与平滑漂移过程以及线性模型、基近似和神经网络等回归问题中刻画了最优权重。 |
| [^374] | [COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation](https://arxiv.org/abs/2602.03806) | COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。 |
| [^375] | [HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos](https://arxiv.org/abs/2602.02473) | HumanX通过协同设计的XGen数据生成流水线与XMimic交互优先学习框架，将人类视频直接转化为无需任务定制设计、可泛化的人形机器人敏捷交互技能。 |
| [^376] | [Beyond Prediction-Set Size: Quantifying Conformal Uncertainty through Credal Geometry](https://arxiv.org/abs/2602.01667) | 该论文提出以信度集合几何作为超越预测集大小的共形不确定性量化新视角，为多种信度不确定性度量给出解析刻画与排序后线性时间的精确算法，通过 Tsallis 熵集合大小惩罚将其与经典共形摘要统一，并证明了常见分离式共形回归中不确定性摘要无法区分输入的不可能性定理。 |
| [^377] | [A Generalizable Machine Learning Framework for Half-Heusler Thermoelectrics](https://arxiv.org/abs/2602.01149) | 本文提出了一种可泛化的机器学习框架，通过基于PCA的数据划分策略、贝叶斯超参数优化与特征选择，能够准确预测半哈斯勒化合物（包括未见成分）的热电品质因数zT，并将RMSE降低16-27%。 |
| [^378] | [RPP: A Certified Poisoned-Sample Detection Framework for Backdoor Attacks under Dataset Imbalance](https://arxiv.org/abs/2602.00183) | 本文首次揭示了数据集不平衡会加剧后门攻击的脆弱性，并提出了一种仅在黑盒设置下利用模型输出概率运行的认证型中毒样本检测框架RPP，该框架可提供可证明的域内可检测性保证和误报率的概率上界。 |
| [^379] | [Preference-Conditioned Multi-Objective Reinforcement Learning for Tactical Decision Making in Heavy-Duty Vehicles](https://arxiv.org/abs/2601.18783) | 提出一种基于偏好条件的多目标强化学习框架，结合广义策略改进与近端策略优化，在安全性、能效和时间三个冲突目标之间逼近帕累托前沿，从而实现重型车辆的战术决策优化。 |
| [^380] | [Learning to accelerate Krasnosel'skii-Mann fixed-point iterations with guarantees](https://arxiv.org/abs/2601.07665) | 本文提出一个学习优化框架，通过向Krasnosel'skii-Mann不动点迭代注入可求和扰动来提升其平均情况性能，同时在理论上保证局部线性收敛性，并可推广用于加速多种算子分裂方法。 |
| [^381] | [Probabilistic Multi-Agent Aircraft Landing Time Prediction](https://arxiv.org/abs/2512.08281) | 提出一个概率多智能体着陆时间预测框架，将多架飞机的着陆时间以概率分布形式输出，同时兼顾轨迹不确定性及空域中飞机间的交互影响。 |
| [^382] | [MetaJoin: Adaptive Join Selections for Heterogeneous Tables](https://arxiv.org/abs/2512.02444) | MetaJoin提出了一个无需标签的自适应连接选择框架，能够在执行后依据专家的输出证据动态决定保留、比较、升级或修复连接结果，并引入基于互最近邻覆盖率搜索字符串转换的MNN Join专家。 |
| [^383] | [Bayesian dynamic scheduling of multipurpose batch processes under incomplete look-ahead information](https://arxiv.org/abs/2512.01093) | 该论文提出了一种基于从扰动概率分布中学习贝叶斯网络的动态调度方法，使多用途批处理过程能够在仅拥有不完整前瞻信息的现实条件下有效应对扰动，从而最小化长期成本和系统紧张性。 |
| [^384] | [Infinitesimal Operator World Models: Finite-Sample Guarantees for Offline Reinforcement Learning](https://arxiv.org/abs/2511.10383) | 该论文将强化学习与HJB方程相联系，提出基于受控扩散过程无穷小生成元（在再生核希尔伯特空间中学习）的算子理论世界模型，在生成元近似误差足够小时证明了值函数的全局收敛性，并为离线强化学习提供了有限样本保证。 |
| [^385] | [A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)](https://arxiv.org/abs/2510.22809) | 提出了一种基于信息论不确定性（惊奇度）的“无模型”学习框架，将学习视为对不确定性的测量，可直接从原始数据进行推理，支持数据的编辑与删除，并在异常检测等任务上达到最先进性能。 |
| [^386] | [Second-Order Optimization with Asynchronous Curvature](https://arxiv.org/abs/2510.15714) | 提出Split-Client异步任务并行设计，使Hessian分解在后台进行而不阻塞梯度更新，并证明了依赖曲率平均延迟（而非最大延迟）的一阶驻点收敛界，其墙钟时间性能可与最优调参的Lazy Hessian方法相媲美且无需调参重用周期。 |
| [^387] | [ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios](https://arxiv.org/abs/2510.10998) | 该论文提出了ABLEIST评估框架，通过五个健全主义伤害指标和三个交叉性伤害指标，对六个LLM在2,820个招聘场景中的表现进行审计，揭示了针对残障候选人（尤其在性别、种姓等交叉身份维度）的显著偏见伤害，而许多先进模型无法检测到这些伤害。 |
| [^388] | [Surrogate Graph Partitioning for Spatial Prediction](https://arxiv.org/abs/2510.07832) | 本文提出将代理图划分问题建模为混合整数二次规划，用于构建使分段内预测方差之和最小的空间分段，并针对其计算复杂度过高的问题开发了高效的近似求解方案。 |
| [^389] | [Mixture of Many Zero-Compute Experts: A High-Rate Quantization Theory Perspective](https://arxiv.org/abs/2510.03151) | 本文借助高率量化理论，分析了一种由推理时零计算的单参数常数专家构成的MoE模型，推导了其在一维与多维输入下的测试误差逼近结果以及专家参数的统计学习性质。 |
| [^390] | [Approximation of differential entropy in Bayesian optimal experimental design](https://arxiv.org/abs/2510.00734) | 该论文针对贝叶斯最优实验设计提出了一种高效计算方法：当似然熵与设计无关或可显式计算时，期望信息增益的嵌套积分可简化为单积分，并通过蒙特卡洛或拟蒙特卡洛代理近似证据密度来高效估计微分熵。 |
| [^391] | [Multidimensional Bayesian Active Machine Learning of Working Memory Task Performance](https://arxiv.org/abs/2510.00375) | 该研究提出一种基于高斯过程后验不确定性的双轴贝叶斯主动分类方法，在沉浸式虚拟环境中同时控制空间负荷与特征绑定负荷两个维度来评估工作记忆表现，其效果与传统单维阶梯式自适应方法相当。 |
| [^392] | [Discovering Phase Space Structure in Learned Hamiltonian Systems](https://arxiv.org/abs/2509.23003) | 提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。 |
| [^393] | [User Trajectory Prediction Unifying Global and Local Temporal Information](https://arxiv.org/abs/2508.02161) | 该论文提出一种融合MLP、多尺度卷积神经网络和交叉注意力的轨迹预测模型，统一提取轨迹数据中的全局与局部时间信息，在保持可接受推理时间的同时降低预测误差。 |
| [^394] | [Regression-adjusted Monte Carlo Estimators for Shapley Values and Probabilistic Values](https://arxiv.org/abs/2506.11849) | 本文提出一种将蒙特卡洛采样与回归调整相结合的新方法来估计Shapley值等概率值，允许用任意可高效计算概率值的函数族（如XGBoost）替代线性回归，并通过交叉拟合实现无偏且性能领先的概率值估计。 |
| [^395] | [UniConFlow: A Unified Constrained Flow-Matching Framework for Certified Motion Planning](https://arxiv.org/abs/2506.02955) | 本文提出UniConFlow，一个统一的约束流匹配轨迹生成框架，通过新颖的预设时间归零函数系统性地处理等式与不等式约束，实现可认证的机器人运动规划。 |
| [^396] | [Physics-inspired Generative AI models via real hardware-based noisy quantum diffusion](https://arxiv.org/abs/2505.22193) | 本文提出两种受物理启发的量子扩散模型方案——利用量子随机行走中量子与经典动力学的相互作用降低生成图像的FID，以及仅用四个量子比特借助真实IBM量子硬件的固有噪声生成图像——为可扩展的量子生成式AI算法开辟了新路径。 |
| [^397] | [Optimizable Model Averaging for Ensembling Heterogeneous Model Outputs](https://arxiv.org/abs/2505.21857) | 提出可优化模型平均（OMA），一种黑盒集成方法，通过最小化正则化预测熵来学习非负组合权重以融合异构架构模型的预测概率，无需微调或参数合并即可改善性能与校准。 |
| [^398] | [Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal](https://arxiv.org/abs/2505.15738) | 该论文主张，由于基于微调的提示注入防御缺乏形式化保证且可能带来虚假安全感，开发者应在防御部署给用户之前用强大的对抗性攻击对其进行压力测试。 |
| [^399] | [Heterogeneity-Aware Client Sampling for Optimal and Efficient Federated Learning](https://arxiv.org/abs/2505.11304) | 本文首次对一般异构联邦学习进行了统一理论分析，揭示了通信异构性与计算异构性各自导致目标不一致的不同机制，并据此提出异构感知的客户端采样策略，以实现最优且高效的联邦学习。 |
| [^400] | [EmbodiTTA: Resource-Efficient Test-Time Adaptation for Embodied Visual Systems](https://arxiv.org/abs/2505.00986) | 本文提出按需测试时自适应框架OD-TTA，仅在检测到显著域偏移时才触发自适应，并通过轻量级域偏移检测、源域选择和解耦BN更新三项技术，在边缘设备上实现了精准且资源高效的自适应。 |
| [^401] | [Fast localized urban microclimate emulator: wind and temperatures in unseen urban morphologies](https://arxiv.org/abs/2503.19708) | 本文提出基于傅里叶神经算子的快速局地城市微气候模拟器FLUME，通过引入多方向距离特征（MDDF）编码建筑几何信息，实现了数据高效、可泛化到未见城市形态的三维风场与温度场快速预测。 |
| [^402] | [Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models](https://arxiv.org/abs/2502.08598) | 提出了一种总方差/信噪比解耦（TV/SNR）框架，使扩散模型噪声调度中的总方差和信噪比可被独立控制，并发现恒定总方差调度能够持续改进指数爆炸总方差的调度。 |
| [^403] | [Bandits with Anytime Knapsacks](https://arxiv.org/abs/2501.18560) | 本文提出了SUAK算法，用于解决带任意时刻成本约束的背包老虎机这一新问题，通过战略性地低利用预算并在必要时自适应跳过轮次，实现了与经典背包老虎机问题相同的 $O(K \log T)$ 问题相关遗憾上界。 |
| [^404] | [A Sampling Complexity-aware Framework for Discrete-time Fractional-Order Dynamical System Identification](https://arxiv.org/abs/2501.17499) | 本文提出了一种采样复杂度感知的离散时间分数阶动力系统辨识算法，通过将辨识问题转化为最小二乘估计并避免求解高阶多项式方程，有效解决了噪声数据下的系统参数辨识问题，并给出了参数估计误差的理论界。 |
| [^405] | [Understanding Transformer-based Vision Models through Inversion](https://arxiv.org/abs/2412.06534) | 本文提出一种新颖的模块化特征反演方法，能更高效地从大规模Transformer视觉模型的中间表示重建语义更连贯的图像，并借此揭示了DETR与ViT等模型内部机制随网络深度的差异。 |
| [^406] | [Conditional Vendi Score: Prompt-Aware Diversity Evaluation for Generative AI Models and LLMs](https://arxiv.org/abs/2411.02817) | 该论文提出了基于条件熵的条件Vendi和条件RKE两种新度量方法，能够在提示引导的生成任务中分离出模型自身引起的多样性，为生成式AI模型和大语言模型提供了提示感知的多样性评估工具。 |
| [^407] | [Deep End-to-End Survival Analysis with Temporal Consistency](https://arxiv.org/abs/2410.06786) | 该论文提出了一种受深度Q网络启发的新型生存分析框架，通过延迟目标网络实现时间一致性约束下的端到端训练，能够高效处理大规模纵向数据并建模复杂的时间依赖关系。 |
| [^408] | [Instance-Level Difficulty: A Missing Perspective in Machine Unlearning](https://arxiv.org/abs/2410.03043) | 本文从实例级角度研究机器遗忘的难度问题，总结出使遗忘某个数据点困难的四个因素，并证明这些因素仅取决于目标模型及其训练数据，而与具体的遗忘算法无关。 |
| [^409] | [Provably Accurate Shapley Value Estimation via Leverage Score Sampling](https://arxiv.org/abs/2410.01917) | 提出Leverage SHAP，通过对Kernel SHAP的轻量级改进，利用杠杆分数采样将模型评估次数从指数级降至O(n log n)，实现了可证明精确的Shapley值估计。 |
| [^410] | [Data-Driven Stochastic Optimal Control in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2407.16407) | 该论文提出一种完全数据驱动的方法，在动力学和成本函数均未知的情况下，通过在再生核希尔伯特空间中学习马尔可夫转移算子，并与随状态维度线性扩展的凸算子理论HJB递归相结合，实现非线性随机扩散系统的最优控制。 |
| [^411] | [TernaryLLM: Ternarized Large Language Model](https://arxiv.org/abs/2406.07177) | 本文提出TernaryLLM，通过双重可学习三值化（DLT）和对离群值友好的特征知识蒸馏（OFF）两项创新技术，克服了权重与激活中离群值的挑战，实现了大语言模型的高效三值化，从而大幅降低计算成本和内存需求。 |
| [^412] | [Linearized Wasserstein dimensionality reduction with approximation guarantees](https://arxiv.org/abs/2302.07373) | 该论文提出LOT Wassmap算法，利用Sinkhorn距离和线性化最优传输等近似方案避免计算两两Wasserstein距离矩阵，从而高效揭示概率测度数据集在Wasserstein空间中的低维结构，并为此类近似下的嵌入质量提供了理论保证。 |

# 详细

[^1]: 捉迷藏：隐藏概念以寻找黑盒模型的解释

    Hide and Seek: HIDE Concepts to SEEK explanations for black box models

    [https://arxiv.org/abs/2610.13935](https://arxiv.org/abs/2610.13935)

    提出了一种无需模型权重或概念标注数据集的黑盒视觉模型概念级归因方法HnS，通过视觉-语言模型检索概念相关图像、遮挡图像中的概念区域，并依据模型输出的变化来衡量该概念的归因贡献。

    

    解释机器学习模型的预测是一项核心挑战，尤其是对于黑盒视觉模型而言。基于概念的归因方法利用人类可理解的概念来解释模型。现有的基于概念的方法只能解释分类和回归模型，它们需要访问模型权重或大型的概念标注数据集，并且大多数方法无法提供图像级别的归因。我们提出了“捉迷藏”方法，这是一种新的概念级归因方法，无需访问模型权重或标注的概念数据集。HnS能够解释广泛的视觉模型。给定一个概念和一个图像集合，HnS使用视觉-语言模型检索与该概念对齐的图像，在每张图像中定位该概念，遮挡定位到的区域，并根据黑盒模型在原始图像与遮挡图像之间输出的变化来衡量归因。这通过直接测量……实现了忠实的归因。

    arXiv:2610.13935v1 Announce Type: cross  Abstract: Interpreting machine learning model predictions is a central challenge, particularly for black-box vision models. Concept-based attribution methods interpret models using human-understandable concepts. Existing concept-based methods can only interpret classification and regression models. They require access to model weights or large concept-annotated datasets, and most methods cannot provide image-level attribution. We propose Hide and Seek (HnS), a new method for concept-level attribution that does not require access to model weights or labeled concept datasets. HnS can interpret a wide range of vision models. Given a concept and image set, HnS retrieves concept-aligned images using a vision-language model, localizes the concept within each image, masks the localized region, and measures attribution from the change in the black-box model's output between the original and masked images. This enables faithful attribution by directly me
    
[^2]: 神经网络拓扑表达能力的构造性界

    Constructive Bounds for the Topological Expressivity of Neural Networks

    [https://arxiv.org/abs/2610.13931](https://arxiv.org/abs/2610.13931)

    本文提出用神经网络输出的贝蒂数（而非决策边界的贝蒂数）来度量拓扑表示能力，并构造性地证明了特定架构的 ReLU 网络可以表达同伦等价于具有任意贝蒂数的球体楔和的空间，从而为拓扑表示能力给出了构造性上界。

    

    神经网络 Φ 的表示能力描述了该网络所能表达的函数类别。“拓扑表示能力”这一术语指的是一类特别适合理解二分类问题的度量，通常以决策边界 β(Φ⁻¹(0)) 的贝蒂数（Betti numbers）来表述。尽管这一度量颇受欢迎，但它对于回归以及其他不涉及决策边界的问题并不适用。作为替代，我们提出用神经网络输出的贝蒂数 β(Φ(X)) 来研究拓扑表示能力。我们严格证明了一个架构为 (d, Σₖ₌₁ᵈ βₖ(3k+2), Σₖ₌₁ᵈ βₖ(k+1)) 的 ReLU 网络，能够表达同伦等价于具有任意贝蒂数的球体楔和的空间。这为任意连通（β₀=1）的贝蒂数序列给出了上界，而通过构造……

    arXiv:2610.13931v1 Announce Type: cross  Abstract: The expressivity of a neural network $\Phi$ describes the class of functions that a network can express. The term topological expressivity refers to a family of measures particularly suited to understand binary classification and is often formulated in terms of the Betti numbers of a decision boundary $\beta (\Phi^{-1}(0))$. Despite its popularity, this measure is not as useful for regression and other problems that do not involve a decision boundary. Instead, we propose to study topological expressivity in terms of the Betti numbers of the output of a neural network $\beta(\Phi(X))$. We formally prove that a ReLU network with architecture $(d,\sum\limits_{k=1}^{d   } \beta_k(3k+2),\sum\limits_{k=1}^{d}\beta_k(k+1))$ can express spaces homotopy equivalent to wedge sums of spheres with arbitrary Betti numbers. This gives upper bounds for any arbitrary connected ($\beta_0=1$) sequence of Betti numbers, but by constructing spaces homotopy
    
[^3]: AI模型能够可证明地隐藏任意能力

    AI Models Can Provably Hide Arbitrary Capabilities

    [https://arxiv.org/abs/2610.13928](https://arxiv.org/abs/2610.13928)

    本文提出一种将加密神经网络嵌入模型权重的后门攻击方法，在密码学假设下可证明地隐藏任意能力，使现有能力评估手段难以引出或解读这些隐藏功能。

    

    能力评估旨在在模型部署前暴露其危险行为，但其可靠性依赖于一个假设：隐藏的能力是可以被引出的。我们通过构建后门攻击证明，这一假设在对抗性条件下并不成立——该攻击将加密的神经网络嵌入宿主模型的权重之中，只有在接收到秘密触发器时才会解密并执行。借助数字保险箱技术，这些隐藏电路在密码学假设下可被证明难以被引出和解释，即使攻击者拥有对模型权重的完全访问权限，这将先前的工作从隐藏固定字符串扩展到了隐藏任意计算。我们在PyTorch中实现了概念验证构建，并在一个小型化学反应预测任务（作为CBRN相关能力的代理任务）上，通过微调、权重加噪、基于梯度的触发器搜索以及潜在对抗训练等方法评估了引出攻击的效果。我们的结果表明，该方法对这些引出手段具有抵抗能力。

    arXiv:2610.13928v1 Announce Type: new  Abstract: Capability evaluations aim to surface dangerous model behaviors before deployment, but their reliability depends on the assumption that hidden capabilities can be elicited. We show this assumption does not hold under adversarial conditions by constructing backdoor attacks that embed encrypted neural networks within host model weights, decrypting and executing them only upon receiving a secret trigger. Using digital lockers, these hidden circuits remain provably hard to elicit and interpret under cryptographic assumptions, even given full access to model weights, extending prior work from hiding fixed strings to arbitrary computations. We implement our proof of concept constructions in PyTorch and evaluate elicitation through fine-tuning, weight noising, gradient-based trigger search, and latent adversarial training on a small chemical reaction prediction task - a proxy for CBRN-relevant capabilities. Our results demonstrate resistance to
    
[^4]: 混合前后的KL散度：并行采样中的轮数复杂度分离

    KL Before and After Mixing: A Round-Complexity Separation in Parallel Sampling

    [https://arxiv.org/abs/2610.13924](https://arxiv.org/abs/2610.13924)

    该论文证明了并行采样中两种KL散度准则之间的轮数复杂度分离：在均衡二进制串的均匀分布上，按调度平均散度准则需要 $\Theta((\log n)^2)$ 轮，而按输出分布散度准则仅需双重对数阶（约 $\log_2\ln n$）轮。

    

    并行采样器可以随机选择将哪些坐标一起生成。对于目标分布 $p$ 以及为每次输出重新抽取的调度 $U$ 所对应的输出分布 $q_U$，我们比较（以奈特为单位）调度平均散度 $\mathbb{E}_U D_{\mathrm{KL}}(p\|q_U)$ 与输出分布散度 $D_{\mathrm{KL}}(p\|\mathbb{E}_U q_U)$。每轮从轮次开始时冻结的精确条件边缘分布中进行独立抽取，每次输出的调度在采样前从调度的有限混合中抽取，并且每个被调度的轮次均被计入成本。对于长度为偶数 $n$ 且恰有 $n/2$ 个 1 的二进制串上的均匀分布，在固定容差 $\varepsilon>0$ 下，第一种准则所需的最少期望轮数为 $\Theta_\varepsilon((\log n)^2)$，而第二种准则下仅为 $(e^{-\varepsilon}+o(1))\log_2\ln n$；并且这一双重对数阶可在每次运行中以相同的轮数实现。对于该目标分布，一个基于计数状态的递归……（摘要原文在此处截断）

    arXiv:2610.13924v1 Announce Type: new  Abstract: A parallel sampler can choose at random which coordinates to generate together. For a target $p$ and the output law $q_U$ of a schedule $U$ drawn afresh for each output, we compare the schedule-averaged divergence $\mathbb{E}_U D_{\mathrm{KL}}(p\|q_U)$ with the output-law divergence $D_{\mathrm{KL}}(p\|\mathbb{E}_U q_U)$, in nats. Rounds commit independent draws from exact conditional marginals frozen at the round's start, each output's schedule is drawn before sampling from a finite mixture of schedules, and every scheduled round is charged. For the uniform distribution on binary strings of even length $n$ with $n/2$ ones, the least expected number of rounds at a fixed tolerance $\varepsilon>0$ is $\Theta_\varepsilon((\log n)^2)$ under the first criterion but $(e^{-\varepsilon}+o(1))\log_2\ln n$ under the second, and the double-logarithmic order is attainable with the same round count on every run. For this target, a count-state recursi
    
[^5]: 基于强化学习的面向目标的深空通信捆绑包管理与流量控制

    Goal-Oriented Bundle Management and Flow Control for Deep-Space Communications via Reinforcement Learning

    [https://arxiv.org/abs/2610.13923](https://arxiv.org/abs/2610.13923)

    本文提出一种基于强化学习的闭环框架，用于接触驱动深空网络中继节点的捆绑包调度与丢弃决策，通过优化接收端信息年龄（信息新鲜度）来提升月球与火星数据向地球传输的任务效用与公平性。

    

    深空通信在间歇性连接、长传播时延和中继存储约束的条件下运行。在这类网络中，捆绑包协议支持“存储-携带-转发”的通信方式，要求中继节点管理有限的存储资源和稀缺的接触机会。中继侧的调度与丢弃决策本质上是面向目标的：传输资源应当分配给那些对下游任务目标最有价值的捆绑包。传统调度器（如先进先出FIFO）并不显式地优化任务效用，并且可能在异构的行星数据源之间提供不公平的服务。本文针对接触驱动的深空网络，提出了一种用于中继侧捆绑包与流量管理的闭环建模与学习框架，其中月球和火星的数据更新通过共享缓冲区的中继节点转发至地球。我们使用过剩年龄 Δ_i^e(t)=[Δ_i(t)-d_i(t)…（原文摘要在此处截断）来量化接收端的信息新鲜度。

    arXiv:2610.13923v1 Announce Type: cross  Abstract: Deep-space communications operate under intermittent connectivity, long propagation delays, and relay storage constraints. In such networks, the Bundle Protocol enables store-carry-forward communication, requiring relay nodes to manage limited storage and scarce contact opportunities. Relay-side scheduling and dropping decisions are inherently goal-oriented: transmission resources should be allocated to bundles most valuable to downstream mission objectives. Conventional schedulers, such as FIFO, do not explicitly optimize task utility and may provide unfair service across heterogeneous planetary sources. This paper presents a closed-loop modeling and learning framework for relay-side bundle and flow management in a contact-driven deep-space network, where lunar and Martian updates are forwarded through a shared-buffer relay to Earth. We quantify destination-side information freshness using excess age $\Delta_i^e(t)=[\Delta_i(t)-d_i(t)
    
[^6]: DyssomniaNN：人类睡眠障碍与主观休息感的多尺度脉冲神经建模

    DyssomniaNN: Multi-Scale Spiking Neural Modeling of Human Sleep Disorders and Perceived Restedness

    [https://arxiv.org/abs/2610.13918](https://arxiv.org/abs/2610.13918)

    提出多尺度脉冲神经网络框架DyssomniaNN，将睡眠稳态双过程模型、下丘脑“跷跷板”回路与皮层-丘脑脉冲网络耦合，成功模拟五种临床睡眠障碍并复现真实脑电特征，为客观睡眠与主观休息感之间的分歧提供了机理解释。

    

    临床睡眠医学反复记录到客观睡眠时长与主观日间功能损害之间的深刻分歧，典型例证包括矛盾性失眠（paradoxical insomnia）中的睡眠状态误感知，以及1型发作性睡病中的边界不稳定现象。在此，我们提出DyssomniaNN：一个多尺度生物物理计算框架，该框架将Borbély的睡眠稳态与昼夜节律驱动双过程调节模型，与Saper的下丘脑相互抑制“跷跷板”回路以及一个递归的皮层-丘脑漏积分放电（leaky integrate-and-fire）脉冲神经网络相耦合。我们在相同的72小时光照周期下形式化了五个临床队列：健康对照、昼夜节律睡眠-觉醒障碍（延迟时相型）、慢性失眠、矛盾性失眠和1型发作性睡病。合成局部场电位在所有典型频段上与来自PhysioNet Sleep-EDF数据库的真实人类脑电图频谱高度吻合。

    arXiv:2610.13918v1 Announce Type: cross  Abstract: Clinical sleep medicine repeatedly documents profound divergences between objective sleep duration and subjective daytime impairment, notably exemplified by sleep state misperception in paradoxical insomnia and boundary instability in narcolepsy type 1. Here, we present DyssomniaNN: a multi-scale biophysical computational framework coupling Borb\'ely's two-process regulation of sleep homeostasis and circadian drive with Saper's hypothalamic reciprocal flip-flop circuit and a recurrent corticothalamic leaky integrate-and-fire spiking neural network. We formalize five clinical cohorts across an identical 72-hour photoperiod: healthy control, circadian rhythm sleep-wake disorder (delayed-phase), chronic insomnia, paradoxical insomnia, and narcolepsy type 1. The synthetic local field potential converges closely with empirical human electroencephalography spectra from the PhysioNet Sleep-EDF database across all canonical frequency bands (sp
    
[^7]: 从自然语言设计蛋白质结构

    Designing Protein Structures from Natural Language

    [https://arxiv.org/abs/2610.13917](https://arxiv.org/abs/2610.13917)

    该论文提出首个支持自然语言指令的全原子蛋白质结构生成模型Prompteina，通过构建包含180万结构和5000多万条文本说明的配对数据集进行训练，实现了类似“文本生成图像”的蛋白质设计范式，并在18项结构特性中的15项上超越现有基线模型。

    

    从头蛋白质设计的一个核心挑战是生成具有所需结构和功能特性的蛋白质。现有的设计模型通常需要为每种特性提供专门的输入表示，这限制了它们所支持的设计目标。我们开发了Prompteina，这是首个支持以自然语言指定的新型、更广泛设计目标的全原子结构生成模型，类似于文本到图像的生成。为了训练Prompteina，我们构建了Prompteina-Data数据集，这是一个文本—蛋白质结构配对数据集，包含180万个结构和超过5000万条描述全局及残基特异性结构与功能特性的文本说明。我们通过文本条件预训练和微调，将残基与其对应的文本对齐，从而适配预训练的结构生成模型Proteina-Complexa。Prompteina在18项结构特性中的15项上超越了基线模型，并生成了与文本描述更加匹配的蛋白质。

    arXiv:2610.13917v1 Announce Type: new  Abstract: A central challenge in de novo protein design is generating proteins with desired structural and functional properties. Existing design models typically require a dedicated input representation for each property, limiting the design objectives they support. We develop Prompteina, the first all-atom structure generation model that supports new and broader design objectives specified in natural language, analogous to text-to-image generation. To train Prompteina, we construct Prompteina-Data, a paired text--protein structure dataset of 1.8M structures and 50M+ captions describing global and residue-specific structural and functional properties. We adapt Proteina-Complexa, a pretrained structure generation model, through text-conditioned pretraining and finetuning that aligns residues with their corresponding text. Prompteina outperforms baselines on 15 of 18 structural properties and generates proteins that more closely match descriptions 
    
[^8]: 重新审视表格数据深度回归：回归目标函数与全局早停的局限

    Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping

    [https://arxiv.org/abs/2610.13909](https://arxiv.org/abs/2610.13909)

    该论文系统性地重新评估了表格深度回归中的各类目标函数，发现分位数损失是唯一能相对调优后的MSE基线带来一致提升的目标函数，并进一步揭示了由于不同噪声水平的区域需要不同的训练时机，全局早停策略存在根本性局限。

    

    尽管面向表格数据的深度学习模型近来受到大量研究关注，但回归任务中目标函数的选择仍是一个研究不足的自由度。若干研究方向提出了标准均方误差（MSE）的替代方案，但由于数据集、骨干网络和评估协议的差异，其实际收益难以相互比较。在本工作中，我们通过在广泛收集的成熟表格数据集和多种神经骨干网络上对回归目标函数进行系统性重新评估来填补这一空白。我们的比较表明，许多已提出的目标函数相对经过恰当调参的MSE基线只能带来有限或不一致的收益，而分位数损失在所测试的各个骨干网络上均能相对MSE取得一致的总体提升。随后，我们识别出即使采用这种强目标函数仍然存在的一个局限：具有不同目标噪声水平的区域会在不同的时机达到t……

    arXiv:2610.13909v1 Announce Type: cross  Abstract: While deep learning models for tabular data have recently received significant research attention, the choice of objective functions for regression tasks remains an understudied degree of freedom. Several lines of research offer alternatives to the standard Mean Squared Error (MSE), but differences in datasets, backbones, and evaluation protocols make their practical benefits difficult to compare. In this work, we address this gap by performing a systematic re-evaluation of regression objectives across a broad collection of established tabular datasets and multiple neural backbones. Our comparison reveals that many proposed objectives provide limited or inconsistent gains over a properly tuned MSE baseline, while quantile loss yields consistent aggregate improvements over MSE across the tested backbones. We then identify a limitation that remains even with this strong objective: regions with different levels of target noise can reach t
    
[^9]: 异步Q学习的命中时间界

    Hitting-Time Bounds for Asynchronous Q-Learning

    [https://arxiv.org/abs/2610.13901](https://arxiv.org/abs/2610.13901)

    该论文在仅要求状态-动作链不可约（允许周期性）的弱条件下，证明了零初始化、常数步长的异步Q学习只需 $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ 次转移即可高概率达到 $\varepsilon$ 上确界范数精度，且初始化成本以反向链最大期望命中时间 $\mathfrak h_\star$ 的加性形式出现。

    

    我们研究了从有限马尔可夫决策过程的单条轨迹出发的标准异步Q学习的最后一次迭代。该轨迹遵循一个固定的行为策略，其状态-动作链是不可约的，且可能是周期的。对于折扣因子 $\gamma$，记 $H=(1-\gamma)^{-1}$，设 $\mu_{\min}$ 为最小的平稳状态-动作概率，$\mathfrak h_\star$ 为反向状态-动作链的最大期望命中时间。我们证明，对于 $0<\varepsilon\le1$，采用常数步长、零初始化的Q学习能够以高概率在 $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ 次转移内实现上确界范数误差至多 $\varepsilon$。统计项保留了标准Q学习对有效视野的四次方依赖，而初始化成本以 $\mathfrak h_\star$ 的加性形式出现。对于非周期行为链，通过与混合时间的比较，可得到一个瞬态项 $\widehat{O}$……（摘要原文在此处被截断）

    arXiv:2610.13901v1 Announce Type: cross  Abstract: We study the last iterate of standard asynchronous Q-learning from a single trajectory of a finite Markov decision process. The trajectory follows a fixed behavior policy whose state--action chain is irreducible and may be periodic. For discount factor $\gamma$, write $H=(1-\gamma)^{-1}$, and let $\mu_{\min}$ be the minimum stationary state--action probability and $\mathfrak h_\star$ the largest expected hitting time of the reverse state--action chain. We prove that zero-initialized Q-learning with a constant step size achieves sup-norm error at most $\varepsilon$ with high probability using $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ transitions, for $0<\varepsilon\le1$. The statistical term retains the quartic effective-horizon dependence of standard Q-learning, while initialization cost is additive in $\mathfrak h_\star$. For aperiodic behavior chains, a comparison with mixing time yields a transient term $\wide
    
[^10]: 预测并不构成充分的记忆状态

    Prediction Is Not a Sufficient Memory State

    [https://arxiv.org/abs/2610.13881](https://arxiv.org/abs/2610.13881)

    本文证明仅能复现当前预测的记忆状态不足以支持后续学习：在精确累积学习中，固定维度的循环状态仅当损失为多项式时存在，且即使在当前预测完全相同的历史之间，继续学习仍可能额外需要 $r-d$ 个状态坐标（例如岭回归中必须存储设计矩阵）。

    

    一个能够复现所有当前预测的记忆，可能仍然已经忘记了如何从下一个观测中学习。我们针对精确累积学习刻画了这一区别。对于具有无约束实值特征与标签的光滑凸残差损失，固定维度的连续循环状态当且仅当损失为多项式时才存在。对于普通的未加权观测，$D$ 次多项式损失恰好需要 $r=\binom{d+D}{D}-1$ 个坐标；即使在具有同一固定当前预测器的历史之中，继续学习仍可能恰好需要 $r-d$ 个坐标。非多项式损失在这类“相同预测”的历史上具有无界的状态需求。有界标量逻辑斯蒂（logistic）的例子在固定任意有限阶导数喷后仍保留这一障碍。对于岭回归，隐藏的证据就是设计矩阵：精确状态需要 $dp+d(d+1)/2$ 的代价，而预测器仅需 $dp$。在一个良态的零预测族上，最优化……（摘要在此处截断）

    arXiv:2610.13881v1 Announce Type: new  Abstract: A memory that reproduces every current prediction may still have forgotten how to learn from the next observation. We characterize this distinction for exact cumulative learning and attention. For smooth convex residual losses with unrestricted real features and labels, a continuous recurrent state of fixed dimension exists exactly when the loss is polynomial. With ordinary, unweighted observations, degree $D$ requires precisely $r=\binom{d+D}{D}-1$ coordinates; even among histories with one fixed current predictor, exactly $r-d$ coordinates can remain necessary for continuation. Nonpolynomial losses have unbounded requirements on such equal-prediction histories. Bounded scalar logistic examples retain this obstruction after fixing any finite derivative jet. For ridge regression, the hidden evidence is the design matrix: the exact state costs $dp+d(d+1)/2$, versus $dp$ for the predictor. On a well-conditioned zero-prediction family, opti
    
[^11]: 基于零阶残差追踪的部分信息随机优化

    Partially Informed Stochastic Optimization via Zeroth-Order Residual Tracking

    [https://arxiv.org/abs/2610.13871](https://arxiv.org/abs/2610.13871)

    提出了PISO方法，将可获取的梯度分量与残差的运行估计相结合进行零阶残差追踪，使随机探测产生的方差取决于追踪误差而非残差幅值，从而在光滑非凸目标上建立了无需对不可获取梯度分量施加一致界的收敛保证。

    

    许多优化问题只能直接获取梯度的一部分，而剩余的分量必须通过函数值评估来推断。这种结构出现在执行性预测、执行性强化学习以及带有黑盒下层问题的双层优化中。我们提出了部分信息随机优化，它将可获得的梯度分量与残差的运行估计相结合。通过将每次零阶观测以该估计为中心，PISO使得由随机探测引起的方差取决于追踪误差而非残差的幅值。这产生了一种具有显式收缩机制的随机化草图迭代，以及对动量的原理性解释。对于光滑非凸目标，我们建立了依赖于残差变化量和初始幅值的收敛保证，而无需对不可获取的梯度分量施加一致界。

    arXiv:2610.13871v1 Announce Type: new  Abstract: Many optimization problems provide direct access to only part of the gradient, while the remaining component must be inferred from function evaluations. This structure arises in performative prediction, performative reinforcement learning, and bilevel optimization with a black-box lower level. We introduce Partially Informed Stochastic Optimization (PISO), which combines the available gradient component with a running estimate of the residual. By centring each zeroth-order observation on this estimate, PISO makes the variance due to random probing depend on the tracking error rather than the residual's magnitude. This yields a randomized sketch iteration with an explicit contraction mechanism and a principled interpretation of momentum. For smooth nonconvex objectives, we establish convergence guarantees that depend on the residual's variation and initial magnitude, without requiring a uniform bound on the unavailable gradient component.
    
[^12]: 全离散条件扩散模型的有限样本保证

    Finite-Sample Guarantees for Fully Discrete Conditional Diffusion Models

    [https://arxiv.org/abs/2610.13851](https://arxiv.org/abs/2610.13851)

    本文为以有限去噪步数实现的条件扩散模型建立了首个有限样本理论保证，通过将分数训练目标与采样网格对齐来联合分析统计估计与数值离散化误差，获得了近优的KL散度与总变差误差收敛速率。

    

    我们为从数据中学习并以有限个去噪步骤实现的条件扩散模型建立了有限样本理论保证。我们的方法将分数训练目标与采样网格对齐，从而能够对统计估计误差与数值离散化误差进行联合分析。在光滑条件密度模型下，我们推导出所生成分布误差的高概率上界，并量化了所需的分数评估次数。对于每一个光滑度指标 $\beta>0$，选择合适的网格后，无论是否采用提前停止，均可获得相对于原始目标的平均条件 Kullback–Leibler (KL) 散度 $\widetilde O_p(n^{-2\beta/(d_x+d_y+2\beta)})$。完整采样器与提前停止采样器均能达到平均目标总变差 (TV) 误差 $\widetilde O_p(n^{-\beta/(d_x+d_y+2\beta)})$。其中 $n$ 为样本量，$d_x,d_y$ 分别为数据维度与条件变量维度，误差在（原文此处截断）上取平均。

    arXiv:2610.13851v1 Announce Type: cross  Abstract: We establish finite-sample guarantees for conditional diffusion models learned from data and implemented with finitely many denoising steps. Our approach aligns the score-training objective with the sampling grid, allowing statistical estimation and numerical discretization errors to be analyzed together. Under a smooth conditional density model, we derive high-probability bounds on the generated distribution's error and quantify the required number of score evaluations. For every smoothness index $\beta>0$, a suitable grid yields averaged conditional Kullback--Leibler (KL) divergence $\widetilde O_p(n^{-2\beta/(d_x+d_y+2\beta)})$ relative to the original target with or without early stopping. Both the full and early-stopped samplers attain average target total-variation (TV) error $\widetilde O_p(n^{-\beta/(d_x+d_y+2\beta)})$. Here $n$ is the sample size, $d_x,d_y$ are the data and conditioning dimensions, and errors are averaged over
    
[^13]: 跨支架与表位的推理时引导以实现样本高效的从头纳米抗体设计

    Inference-Time Steering Across Scaffolds and Epitopes for Sample-Efficient De Novo Nanobody Design

    [https://arxiv.org/abs/2610.13844](https://arxiv.org/abs/2610.13844)

    该论文将纳米抗体设计中的条件选择（支架与表位）重新表述为推理时资源分配问题，提出全局SMC和改编BaSE两种动态预算分配器，使每GPU小时可设计候选产量提升近5倍。

    

    生成式分子设计通常依赖于过度生成与事后过滤。像Feynman-Kac（FK）引导这样的推理时引导方法通过将生成轨迹偏向高奖励候选来提高效率，但它们只能在单一固定配置内进行优化。在从头纳米抗体设计中，预先确定支架类型或目标表位等条件选择会造成瓶颈。由于产量在不同条件之间差异很大，对没有前景的设置进行引导会浪费计算资源。我们将生成式条件选择重新表述为一个推理时资源分配问题。我们提出了两种动态预算分配器：一种是联合的全局序贯蒙特卡罗（SMC）方法，可以在不同配置之间并发地进行引导和计算资源分配；另一种是改编自批处理逐次消除（BaSE）的解耦方法。实验表明，这些动态分配器将每GPU小时可设计候选的产量提升了近5倍。

    arXiv:2610.13844v1 Announce Type: new  Abstract: Generative molecular design typically relies on over-generation and post-hoc filtering. Inference-time steering methods like Feynman-Kac (FK) steering improve efficiency by biasing trajectories toward high-reward candidates, yet they optimize within a single fixed configuration. In de novo nanobody design, upfront commitment to conditioning choices like scaffold identity or target epitope creates a bottleneck. Since yield varies across these conditions, steering unpromising setups wastes compute. We recast generative condition selection as an inference-time resource-allocation problem. We propose two dynamic budget allocators: a joint Global Sequential Monte Carlo (SMC) formulation that steers and allocates compute across configurations concurrently, and a decoupled approach adapting Batched Successive Elimination (BaSE). Empirically, these dynamic allocators increase the yield of designable candidates per GPU-hour nearly 5$\times$ witho
    
[^14]: 回归开发状态：完整沙箱快照何时能帮助编码智能体

    Returning to Development States: When Full Sandbox Snapshots Help Coding Agents

    [https://arxiv.org/abs/2610.13841](https://arxiv.org/abs/2610.13841)

    基于完整沙箱快照的 Go-Explore 式搜索可以提升编码智能体的解决率与效率，其效果主要依赖于代码仓库之外的状态（如数据库、服务及其他运行时状态）。

    

    编码智能体在失败之前往往会取得有用的进展，但从头重试会丢弃在此过程中创建的机器状态。这使得故障恢复成为一个有趣的搜索问题：如果中间环境状态保留了有意义的进展，从这些状态恢复并分支可能优于从初始状态反复重启。我们在完整的 Daytona 快照上实现了 Go-Explore 风格的搜索，发现更高且更高效的解决率主要依赖于位于代码仓库之外的进展，例如数据库、服务和其他运行时状态。我们的结果表明，当长周期的编码工作跨越多个服务、且其状态无法从单个 diff 重建时，完整快照搜索最为有效。

    arXiv:2610.13841v1 Announce Type: new  Abstract: Coding agents often make useful progress before they fail, but retry-from-scratch discards the machine state created along the way. This makes failure recovery an interesting search problem: if intermediate environment states preserve meaningful progress, restoring and branching from them could outperform repeatedly restarting from the initial state. We implement Go-Explore-style search over full Daytona snapshots and find that higher and more efficient solve rates depend mainly on progress that sits outside the repository, such as databases, services, and other runtime state. Our results suggest that full-snapshot search is most effective when long-horizon coding work spans multiple services whose state cannot be reconstructed from a single diff.
    
[^15]: 专家学得更快：预训练中的知识组织方式决定知识更新效率

    Specialists Learn Faster: Knowledge Organization in Pre-training Shapes Knowledge Update Efficiency

    [https://arxiv.org/abs/2610.13835](https://arxiv.org/abs/2610.13835)

    研究发现按时间顺序组织预训练数据所形成的“专家”模型比随机打乱顺序训练的“通才”模型能以更少的更新步骤更高效地获取新知识，并从第一性原理推导出解释该现象的定律，据此设计了保留稳定历史知识的预训练数据组织策略。

    

    理想的（语言）模型必须能够持续更新其知识。现有研究主要关注如何在训练完成后使这些更新变得高效。我们研究了预训练期间时变知识的组织方式如何影响后续更新的效率。按时间顺序组织知识进行训练的模型会专精于最近接触到的知识，我们将其称为“专家”（specialists）；相比之下，使用随机打乱顺序训练相同知识的模型在所有呈现的时间段上均表现良好，我们将其称为“通才”（generalists）。令人惊讶的是，专家模型获取新知识所需的更新步骤远少于通才模型。为解释这一优势，我们从第一性原理出发推导出一个定律，该定律将更新成本与模型的优化动力学联系起来，并能很好地预测实验结果。基于这些发现，我们设计了一种预训练数据组织策略，以保留稳定的历史知识……

    arXiv:2610.13835v1 Announce Type: new  Abstract: An ideal (language) model must continually update its knowledge. Existing work largely focuses on making these updates efficient after training. We investigate how the organization of time-varying knowledge during pre-training shapes the efficiency of subsequent updates. Models trained on chronologically ordered knowledge specialize in the most recently encountered knowledge; we call them specialists. In contrast, models trained on the same knowledge in randomly shuffled order perform well across all represented time periods; we call them generalists. Surprisingly, specialists require far fewer update steps than generalists to acquire new knowledge. To explain this advantage, we derive a law from first principles that relates update cost to the model's optimization dynamics and that closely predicts empirical results. Guided by these findings, we design a strategy for organizing pre-training data that preserves stable historical knowledg
    
[^16]: 连续扩散语言模型的设计空间

    On the Design Space of Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.13834](https://arxiv.org/abs/2610.13834)

    本研究揭示了连续扩散语言模型中自条件化通过纠正训练/推理不匹配而生效的机制，并提出了无需额外前向传播的二级插值方法DoubleCat，从而实现更廉价且更有效的设计。

    

    连续扩散语言模型（CDLMs）有望将连续流建模的优势引入语言领域，包括加速采样和可控生成。该领域发展迅速，CDLMs的能力已开始与离散扩散模型相媲美。然而，与图像或视频等连续模态不同，我们对训练CDLMs时哪些关键因素起决定作用的认知仍然有限。在本工作中，我们聚焦于其中两个因素，即自条件化和嵌入几何，并揭示了它们背后的一些机制，这些机制进而启发了更廉价、更有效的设计。我们首先证明，自条件化通过让模型接触到推理时才会出现、而在训练过程中较为罕见的状态，从而纠正了训练与推理之间的不匹配。这些增益可以通过一种二级插值方法——我们称之为 $\mathrm{DoubleCat}$——在不增加额外前向传播的情况下获得，该方法在TinyGSM任务上的准确率提升更……（原文摘要在此处截断）

    arXiv:2610.13834v1 Announce Type: new  Abstract: Continuous diffusion language models (CDLMs) promise to bring the benefits of continuous flow modelling to language, including accelerated sampling and controllable generation. The field has grown rapidly, and the capabilities of CDLMs have begun to rival those of discrete diffusion. Yet, unlike for continuous modalities such as images or videos, our grasp of the levers that matter when training CDLMs remains limited. In this work, we focus on two of these levers, namely self-conditioning and embedding geometry, and shed light on some of the mechanisms behind them, which in turn suggest cheaper and more effective designs. We begin by showing that self-conditioning corrects a train/inference mismatch by exposing the model to inference-like states that are otherwise rare during training. These gains can be recovered without an extra forward pass by a two-level interpolant, coined $\mathrm{DoubleCat}$, which improves TinyGSM accuracy more t
    
[^17]: 超越专家视图的共享典型成分移除

    Shared Canonical Component Removal Beyond Expert Views

    [https://arxiv.org/abs/2610.13819](https://arxiv.org/abs/2610.13819)

    本文提出共享典型成分移除（SCCR）审计框架，通过仅移除原始嵌入与专家嵌入之间强烈共享的方向，量化并揭示黑盒模型超越专家特征所利用的残余预测信号。

    

    在临床实践中，专家通常首先将原始输入转换为一组专家特征，然后再做决策。尽管在原始输入上训练的黑盒模型可以超越在专家视图上训练的模型，但目前尚不清楚原始模型在专家视图所捕捉的信息之外还利用了什么预测信号。我们引入了一个审计框架，使用共享典型成分移除（SCCR）来量化这一差距。与标准的残差化或概念擦除方法不同，SCCR 仅选择性地移除学习到的原始嵌入与专家嵌入之间强烈共享的方向。在合成数据、冠心病（CAD）检测、抗抑郁治疗脱落预测以及使用 SkinCon 标注的皮肤病变分类上的评估表明，SCCR 能够识别专家视图的不足之处，并恢复残余的预测信号，同时最小化与专家视图的冗余。

    arXiv:2610.13819v1 Announce Type: new  Abstract: In clinical practice, experts often begin by converting raw inputs into a set of expert features before making a decision. While black-box models trained on raw input can outperform models trained on expert views, it remains unclear what predictive signal the raw model exploits beyond what the expert view captures. We introduce an audit framework to quantify this gap using Shared Canonical Component Removal (SCCR). Unlike standard residualization or concept erasure, SCCR selectively removes only those directions strongly shared between learned raw and expert embeddings.   Evaluations on synthetic data, coronary artery disease (CAD) detection, antidepressant treatment dropout prediction, and skin lesion classification using SkinCon annotations show that SCCR identifies the ways in which expert views are insufficient and recovers residual predictive signal while minimizing redundancy with the expert view.
    
[^18]: TabMF：基于表格基础模型学习数据集表示

    TabMF: Learning Dataset Representations with Tabular Foundation Models

    [https://arxiv.org/abs/2610.13818](https://arxiv.org/abs/2610.13818)

    TabMF提出了一种框架，通过架构感知的数据集级读出层和基于合成数据随机视图的对比后训练，将预训练表格基础模型（TabICLv2与TabPFN-2.5）重新用作有监督数据集的编码器，从而在数据集检索和分布偏移监测等任务上学习到强大的数据集级表示。

    

    有监督数据集的表示可以支持更高层次的机器学习任务，例如数据集检索和分布偏移监测。然而，现有方法通常要么依赖手工设计的元特征，要么依赖专门为表示学习训练的数据集编码器。我们提出了TabMF，这是一个将预训练表格基础模型（TFMs）重新用作有监督数据集编码器的框架。TabMF为预测型TFM配备了一个架构感知的数据集级读出层，并通过对合成有监督数据集的随机视图进行对比后训练，实现模型的端到端适配。我们在TabICLv2和TabPFN-2.5上实例化了TabMF，并在未见过的真实数据集上通过六项任务评估了所得的表示。简单的冻结TFM读出层即可产生具有竞争力的表示，这表明预测性预训练本身已经诱导出了实质性的数据集级结构。两个TabMF编码器取得了最强的……（原文摘要在此处截断）

    arXiv:2610.13818v1 Announce Type: new  Abstract: Representations of supervised datasets support higher-level machine-learning tasks such as dataset retrieval and distribution-shift monitoring. However, existing approaches typically rely either on hand-crafted meta-features or on dataset encoders trained specifically for representation learning. We introduce TabMF, a framework that repurposes pretrained tabular foundation models (TFMs) as encoders of supervised datasets. TabMF equips a predictive TFM with an architecture-aware dataset-level readout and adapts the model end-to-end using contrastive post-training on stochastic views of synthetic supervised datasets. We instantiate TabMF on TabICLv2 and TabPFN-2.5 and evaluate the resulting representations across six tasks on unseen real datasets. Simple frozen TFM readouts yield competitive representations, indicating that predictive pretraining already induces substantial dataset-level structure. The two TabMF encoders obtain the stronge
    
[^19]: 梦境记忆：基于世界模型的通用机器人策略持续引导

    Dream to Remember: Continual Steering of Generalist Robot Policies with World Models

    [https://arxiv.org/abs/2610.13816](https://arxiv.org/abs/2610.13816)

    该论文提出D2R方法，利用冻结的预训练世界模型通过想象推演来温习旧任务，从而在持续引导冻结的通用机器人策略学习新任务的同时避免灾难性遗忘，且无需重放缓冲区或新数据。

    

    通用机器人策略通过模仿学习进行训练，其能力受限于演示数据的质量和覆盖范围。强化学习可以改进和扩展这些策略，方法要么是微调策略权重，要么是训练一个小型模块来引导冻结的生成式策略的输入噪声。当任务按顺序到来时，这两种方法都会遭受灾难性遗忘：在某个任务上进行微调会损害先前任务的表现，而引导方法只是将遗忘转移到了共享的引导模块上。我们提出了 Dream to Remember（D2R），它能够将冻结的通用策略持续引导至新任务，同时保留其先前的技能。当引导模块学习新任务时，一个冻结的预训练世界模型通过想象推演来温习先前任务，将引导模块锚定在其先前的行为上，无需重放缓冲区或新数据。当世界模型足够准确时，D2R 可以完全在想象中学习新任务。否则……（摘要原文在此处截断）

    arXiv:2610.13816v1 Announce Type: cross  Abstract: Generalist robot policies are trained with imitation learning and are bounded by the quality and coverage of their demonstrations. Reinforcement learning can improve and extend them, either by fine-tuning the policy weights or by training a small module to steer the input noise of a frozen generative policy. When tasks arrive sequentially, both approaches suffer from catastrophic forgetting. Fine-tuning on a task degrades prior tasks, and steering only shifts forgetting to the shared steering module. We propose Dream to Remember (D2R), which continually steers a frozen generalist policy toward new tasks, while retaining its earlier skills. As the steering module learns a new task, a frozen pretrained world model rehearses prior tasks through imagined rollouts, anchoring the module to its earlier behavior without a replay buffer or new data. When the world model is faithful enough, D2R learns the new task entirely in imagination. Otherw
    
[^20]: 通过精确高斯条件化统一扩散与基于流的后验采样

    Spanning Diffusion and Flow-based Posterior Sampling via Exact Gaussian Conditioning

    [https://arxiv.org/abs/2610.13812](https://arxiv.org/abs/2610.13812)

    该论文提出 β-Flow，一种基于精确高斯条件化（卡尔曼更新）的通用即插即用后验采样框架，通过相关参数 β 和置信度参数 κ 统一了流匹配、扩散及去相关先验等多种现有引导采样方法。

    

    流匹配和扩散模型越来越多地被用作线性逆问题的先验，然而大多数方法都是针对某一类轨迹先验量身定制的。虽然已有即插即用方法被提出，但这些方法会隐式地去相关化引导轨迹，导致先验过于宽泛，在数据稀疏时表现不佳。本工作提出了 β-Flow：一种通用的即插即用方法，由相关参数 β 索引，可泛化流、扩散以及去相关先验；并由置信度参数 κ 控制路径分布被施加的强度。每一步只需要对去噪网络进行一次前向评估，随后执行一个高斯条件化步骤。该步骤是一个精确的卡尔曼更新，不依赖于网络梯度。我们证明了许多现有的引导采样器都可以被解释为我们方法的实例或近似，每种方法都隐式地使用了不同的轨迹先验。

    arXiv:2610.13812v1 Announce Type: new  Abstract: Flow-matching and diffusion models are increasingly used as priors for linear inverse problems, however, most methods are bespoke to one class of trajectory prior. Plug-and-play methods have been proposed, however these implicitly decorrelate the guided trajectory, resulting in broad priors that can struggle when data is sparse. This work proposes $\beta$-Flow: a general plug-and-play method indexed by a correlation parameter $\beta$ that generalises over flow, diffusion, and decorrelated priors, and a confidence parameter $\kappa$ that controls how strongly the path distribution is enforced. Each step requires only one forward evaluation of a denoising network, followed by a Gaussian conditioning step. This step is an exact Kalman update that is not dependent on network gradients. We show that many existing guided samplers can be interpreted as instances or approximations of our method, each implicitly using a different trajectory prior
    
[^21]: QTestBench：一个用于评估大语言模型生成的量子代码的测试框架

    QTestBench: A Test Framework for Evaluating LLM-Generated Quantum Code

    [https://arxiv.org/abs/2610.13811](https://arxiv.org/abs/2610.13811)

    该论文提出了QTestBench框架，首次通过将大语言模型生成的量子测试判定结果与独立的基准检查器对比，来评估这些测试能否有效区分正确与错误的量子代码实现，而非将测试本身当作评估标准。

    

    大语言模型（LLMs）在经典代码生成方面已展现出优异表现，并越来越多地被用于编写保护代码质量的单元测试。然而，它们能否在量子软件领域做到同样的事情仍不明确，因为量子程序具有概率性、依赖于数值容差，并由特定领域的检查器进行评判，因此一个格式良好的测试仍可能与基准测试的判定结果不一致。现有的量子基准测试衡量的是模型能否产生正确的解决方案，但并未衡量其生成的测试能否区分正确的实现与错误的实现。我们提出了QTestBench，这是一个将大语言模型生成的量子测试套件与独立的基准检查器进行对比评估的框架，绝不将测试本身视为真值。对于每个任务，QTestBench会提示模型生成语法、语义和行为层面的测试，在运行时注入候选解决方案，并将每个测试的判定结果与检查器的标签进行对比。

    arXiv:2610.13811v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong results in classical code generation and are increasingly used to write the unit tests that safeguard that code. Whether they can do the same for quantum software is less clear, because quantum programs are probabilistic, depend on numerical tolerances, and are judged by domain-specific checkers, so a well-formed test can still disagree with the benchmark's verdict. Existing quantum benchmarks measure whether a model produces a correct solution, but not whether its tests can distinguish correct implementations from faulty ones. We introduce QTestBench, a framework that evaluates LLM-generated quantum test suites against independent benchmark checkers, never treating the tests themselves as ground truth. For each task, QTestBench prompts models to produce syntactic, semantic, and behavioral tests, injects candidate solutions at runtime, and compares each test verdict with the checker's labe
    
[^22]: 基于组合融合的多样性感知量子-经典模型集成

    Diversity-Aware Ensemble of Quantum-Classical Models via Combinatorial Fusion

    [https://arxiv.org/abs/2610.13808](https://arxiv.org/abs/2610.13808)

    本文提出基于组合融合分析（CFA）的多样性感知方法，通过秩-得分函数和认知多样性将多个架构各异的量子电路与经典模型进行融合集成，在四个二分类基准上持续超越最佳单一模型。

    

    变分量子电路（VQC）对电路架构高度敏感，然而大多数量子机器学习流程仅选择单一最佳设计并丢弃其余候选方案。本文研究了一个问题：量子电路搜索空间中的结构多样性是否可以通过模型融合来产生更强的预测。我们应用组合融合分析（CFA）——一个利用秩-得分函数（RSF）和认知多样性（CD）来选择和组合多个评分系统的模型融合框架——对一个包含29个架构各异的经典模型和量子电路的模型池进行处理。通过一种多样性感知的模型选择方法，我们识别出具有最大多样性的模型子集，并结合得分组合与秩组合两种方式对它们进行融合。在涵盖不同领域的四个二分类基准数据集上的实验表明，基于CFA的混合经典-量子融合方法持续优于最佳的单个模型。

    arXiv:2610.13808v1 Announce Type: cross  Abstract: Variational quantum circuits (VQCs) are highly sensitive to circuit architecture, yet most quantum machine learning pipelines select a single best design and discard the remaining candidates. This paper addresses whether the structural diversity across quantum circuit search spaces can be used to produce stronger predictions through model fusion. We apply Combinatorial Fusion Analysis (CFA), a model fusion framework that selects and combines multiple scoring systems using rank-score function (RSF) and cognitive diversity (CD), to a pool of 29 architecturally diverse classical models and quantum circuits. Using a diversity-aware model selection method, we identify subsets of maximally diverse models and combine them with both score and rank combination. Experiments on four binary classification benchmarks covering different domains demonstrate that CFA-based hybrid classical-quantum fusion consistently outperform the best individual mod
    
[^23]: FORTE：一个用于工业推荐系统中特征评估与淘汰的生产级框架

    FORTE: A Production Framework for Feature Evaluation and Deprecation in Industrial Recommender Systems

    [https://arxiv.org/abs/2610.13806](https://arxiv.org/abs/2610.13806)

    FORTE是一个与模型无关的生产级框架，通过高效的扰动实现支持每日特征评估，并利用分组消融处理特征交互，从而实现工业推荐系统中特征的系统化发现、验证与淘汰。

    

    生产环境的推荐系统存在特征生命周期的不对称性：添加新特征必须在某些指标上产生收益，而移除特征则不能使任何指标发生退化。在实践中，特征淘汰难以被优先考虑，因为一个特征可能在以下情况下无法被移除：它与其他特征存在交互作用、其重要性每天都在变化、在生产模型变更或数据漂移发生之前无法对其进行评估、它与其他发布项目争夺A/B测试空间、它在在线A/B测试中导致指标退化，或者它被多个需要协调其淘汰的模型共同使用。我们提出了FORTE，一个与模型无关、用于发现、验证和移除特征的生产级框架。其高效的扰动实现支持每日特征评估。为了考虑特征交互作用，它通过分组消融来评估从所有低秩特征构建的候选特征集，其离线结果足够预……

    arXiv:2610.13806v1 Announce Type: new  Abstract: Production recommender systems have feature-lifecycle asymmetry: adding new features must produce gains on some metric, while feature removal must not regress any metric. In practice, feature deprecation is hard to prioritize as a feature can fail to be removed if it interacts with other features, its importance changes daily, it cannot be evaluated before production models change or data drift occurs, it is in contention for A/B test space with other launches, it causes regressions in an online A/B test, or it is consumed by multiple models that need to coordinate its deprecation. We present FORTE, a model-agnostic production framework for discovering, validating, and removing features. Its efficient perturbation implementation allows daily feature evaluation. To account for interactions, it evaluates candidate feature sets constructed from all the low-rank features through grouped ablation and its offline results are sufficiently predi
    
[^24]: 约束概率数值方法：线性不等式约束下线性偏微分方程的高斯过程求解器

    Constrained Probabilistic Numerics: Gaussian Process Solvers for Linear PDEs under Linear Inequality Constraints

    [https://arxiv.org/abs/2610.13804](https://arxiv.org/abs/2610.13804)

    该论文提出了一种支持线性不等式约束（如非负性、单调性）的高斯过程PDE求解方法，证明等式与不等式条件化可以顺序进行，并通过截断高斯矩实现了非高斯后验的可计算性。

    

    arXiv:2610.13804v1 公告类型： cross 摘要：线性偏微分方程（PDE）的高斯过程求解器返回的是后验分布而非点估计，但它们仅以等式信息为条件：PDE残差、边界数据和测量值。然而，许多感兴趣的解还满足不等式约束，包括非负性、单调性和无套利性，而这些约束无法通过等式条件来强制执行。我们证明，将高斯过程先验同时条件于线性等式观测和一个有限维不等式事件，等价于在解空间上以Borel测度的形式进行顺序条件化。因此，融合物理信息的后验可以被视为虚拟点处不等式更新的有效先验。组合后的分布虽然是非高斯的，但可以通过截断高斯矩进行计算，且PDE条件化和不等式更新均无需任何修改。我们在一个泊松基准问题、期权定价……上对该构造方法进行了评估。

    arXiv:2610.13804v1 Announce Type: cross  Abstract: Gaussian process solvers for linear partial differential equations (PDEs) return posteriors rather than point estimates, but they condition only on equality information: the PDE residual, boundary data, and measurements. Many solutions of interest also obey inequality constraints, including nonnegativity, monotonicity, and absence of arbitrage, which equality conditioning cannot enforce. We prove that conditioning a Gaussian process prior jointly on linear equality observations and a finite-dimensional inequality event is equivalent to conditioning sequentially in terms of Borel measures on the solution space. The physics-informed posterior is therefore a valid prior for the inequality update at virtual points. The combined law is non-Gaussian yet computable through truncated-Gaussian moments, and neither the PDE conditioning nor the inequality update requires modification. We evaluated the construction on a Poisson benchmark, option-p
    
[^25]: 无需知道Oracle光滑度的自适应方差缩减

    Adaptive Variance Reduction without Knowing Oracle Smoothness

    [https://arxiv.org/abs/2610.13799](https://arxiv.org/abs/2610.13799)

    提出了一种两阶段自适应方差缩减算法，无需预知光滑常数或方差界，即可同时适应平均光滑性、标准光滑性和无噪声三种设置，分别达到 $O(T^{-1/3})$、$O(T^{-1/4})$ 和 $O(T^{-1/2})$ 的收敛速率。

    

    在平均光滑性条件下，方差缩减技术可以改善随机非凸优化的收敛速率。现有的自适应方差缩减方法专为平均光滑性设计，无法适应标准光滑性。在本文中，我们研究了一个算法如何在不了解光滑常数或方差界的情况下，自适应地适应这两种设置以及无噪声设置。给定梯度预算 $T$，我们开发了一种两阶段方法。第一阶段比较两个独立的梯度，当它们的差超过阈值时切换到归一化动量。这一阶段既提供了具有标准光滑和无噪声保证的候选解，也为第二阶段的方差缩减确保了有界的期望函数间隙。由此得到的收敛速率为：在平均光滑性下为 $O(T^{-1/3})$，在标准光滑性下为 $O(T^{-1/4})$，在无噪声情况下为 $O(T^{-1/2})$。当 $T$ 未知时，w

    arXiv:2610.13799v1 Announce Type: cross  Abstract: Variance reduction improves the convergence rate of stochastic non-convex optimization under the average smoothness condition. Existing adaptive variance reduction methods are especially designed for average smoothness and do not adapt to standard smoothness. In this paper, we study how one algorithm can adapt to these two settings and the noiseless setting without knowing the smoothness constants or the variance bound. Given the gradient budget $T$, we develop a two-phase method. The first phase compares two independent gradients and switches to normalized momentum when their difference exceeds a threshold. This phase provides both a candidate with standard smooth and noiseless guarantees and also ensures a bounded expected function gap for the second variance reduction phase. The resulting rates are $O(T^{-1/3})$ under average smoothness, $O(T^{-1/4})$ under standard smoothness, and $O(T^{-1/2})$ without noise. When $T$ is unknown, w
    
[^26]: 带廉价预测器的随机动量传输

    Randomized Momentum Transport with Cheap Predictors

    [https://arxiv.org/abs/2610.13789](https://arxiv.org/abs/2610.13789)

    本文提出随机动量传输方法，以概率 $q$ 执行昂贵的传输校正、其余时候使用廉价预测并配合重要性加权，在保持 $O(T^{-1/3})$ 平稳性收敛界的同时显著降低了动量方法的oracle调用成本。

    

    诸如STORM和二阶动量等动量方法通过在动量上添加传输校正来跟踪梯度，但其代价是每次迭代需要一次额外的oracle调用。我们引入了随机动量传输方法，该方法以概率 $q$ 计算校正项，否则使用廉价的预测，并通过重要性加权保持条件均值，同时将 $(q^{-1}-1)$ 倍的预测损失加到方差上。在传输偏差、二阶矩和残差控制假设下，线性最小化oracle（LMO）更新——包括归一化方向、符号方向和精确谱方向（Muon）——在单梯度初始化下，对于每个固定的 $q>0$ 都能保持 $O(T^{-1/3})$ 的平稳性界。对于未归一化的STORM，当且仅当梯度差噪声模量超过光滑度常数时，某个 $q<1$ 才能降低充分梯度预算的首项系数，降幅最多约 $10\%$。在线对角预…（摘要原文在此处截断）

    arXiv:2610.13789v1 Announce Type: new  Abstract: Momentum methods such as STORM and second-order momentum track the gradient by adding a transport correction to momentum, at the cost of an extra oracle call per iteration. We introduce randomized momentum transport, which evaluates the correction with probability $q$ and otherwise uses a cheap prediction, with importance weighting that preserves the conditional mean and adds $(q^{-1}-1)$ times the prediction loss to the variance. Under transport-bias, second-moment, and residual-control assumptions, linear minimization oracle (LMO) updates, including normalized, sign, and exact spectral (Muon) directions, retain an $O(T^{-1/3})$ stationarity bound for each fixed $q>0$ with a single-gradient initialization. For unnormalized STORM, some $q<1$ lowers the leading coefficient of our sufficient gradient budget if and only if the gradient-difference noise modulus exceeds the smoothness constant, by at most about $10\%$. An online diagonal pred
    
[^27]: 带约束的非光滑非凸随机优化的最优算法

    Optimal Algorithms for Nonsmooth Nonconvex Stochastic Optimization with Constraints

    [https://arxiv.org/abs/2610.13787](https://arxiv.org/abs/2610.13787)

    该论文提出了求解带凸约束的非光滑非凸随机优化问题的新算法，首次在多种设定下达到最优收敛率：使用随机梯度时复杂度为 Õ(δ⁻¹ε⁻³)，使用零阶查询时复杂度为 Õ(dδ⁻¹ε⁻³)。

    

    我们针对一类随机优化问题提出了新的算法，该类问题的目标函数既不光滑也不凸，且受到凸约束集的限制。我们的算法能够返回一个 (δ,ε)-Goldstein 驻点（该概念是 Liu 等人于 2024 年针对约束问题最近定义的），并具有改进的复杂度，且是首个在若干设定下针对该任务达到最优收敛率的算法。特别地，我们设计了基于投影和无投影（projection-free）的两类算法：使用随机梯度时可达到 Õ(δ⁻¹ε⁻³) 的速率，我们证明了该速率是最优的；使用零阶查询时可达到 Õ(dδ⁻¹ε⁻³) 的速率，该速率在维度 d 的依赖关系上也是最优的，并改进了先前的工作。值得注意的是，我们的下界是以 Frank-Wolfe 间隙的形式给出的，并且即使对于光滑问题也成立，这具有独立的学术价值。这些结果表明……（原文摘要在此处截断）

    arXiv:2610.13787v1 Announce Type: cross  Abstract: We present new algorithms for stochastic optimization problems in which the objective is neither smooth nor convex, subject to a convex constraint set. Our algorithms return a $(\delta,\varepsilon)$-Goldstein stationary point, as recently defined for constrained problems by Liu et al. (2024), with improved complexities, and are the first to achieve optimal convergence rates for this task in several settings. In particular, we design projection-based and projection-free algorithms that achieve a $\widetilde O(\delta^{-1}\varepsilon^{-3})$ rate using stochastic gradients, which we show to be optimal, and a $\widetilde O(d\delta^{-1}\varepsilon^{-3})$ rate using zeroth-order queries, which is also optimal in its dependence on the dimension $d$ and improves over prior work. Notably, our lower bound is given in terms of the Frank-Wolfe gap and holds even for smooth problems, which is of independent interest. These results show that in nonco
    
[^28]: 你的离散图扩散模型正在秘密记忆

    Your Discrete Graph Diffusion Model is Secretly Memorizing

    [https://arxiv.org/abs/2610.13786](https://arxiv.org/abs/2610.13786)

    本文首次证明离散图扩散模型存在“条件记忆化”现象——当反向扩散轨迹以损坏的训练图为条件时，模型几乎能完美恢复该训练图，而这种记忆化在常规推理和样本级评估指标中均无法被察觉。

    

    离散图扩散模型（GDMs）在马尔可夫噪声模型下，通过独立地向节点和边添加噪声来破坏图结构。在没有节点特征的无条件设置下，在单个图上训练的等变去噪器能够生成新颖的图，并且边重叠和WL核相似度等样本级指标未发现记忆化的证据。在这项工作中，我们挑战了这一观点，并首次证明图扩散模型确实表现出记忆化。通过分析在双块随机块模型（SBM）上训练的GDM的反向扩散轨迹，我们发现存在强烈的训练数据偏差：与保留的测试图相比，该模型几乎完美地恢复了被损坏的训练图。我们将这种现象称为“条件记忆化”，因为它仅在反向轨迹以损坏的训练图为条件时才会显现，而在常规推理过程中并不明显。它与不断增长的基因（原文在此处截断）

    arXiv:2610.13786v1 Announce Type: cross  Abstract: Discrete graph diffusion models (GDMs) corrupt the graph structure by adding noise independently to nodes and edges under a Markovian noise model. In the unconditional setting without node features, an equivariant denoiser trained on a single graph is able to generate novel graphs and sample-level metrics such as edge overlap and WL kernel similarity find no evidence of memorization. In this work, we challenge this view and show for the first time that graph diffusion models indeed exhibit memorization. Analyzing the reverse diffusion trajectories of GDMs trained on two-block stochastic block models (SBMs), we find a strong train-data bias: the model recovers a corrupted training graph almost perfectly compared to a held-out test graph. We term this conditional memorization, as it surfaces only when the reverse trajectory is conditioned on a corrupted training graph and is not apparent during inference. It coincides with a growing gene
    
[^29]: ReplaySSM：缓存 SSM 输入，而非状态

    ReplaySSM: Cache SSM Inputs, Not State

    [https://arxiv.org/abs/2610.13778](https://arxiv.org/abs/2610.13778)

    ReplaySSM 通过将 SSM 状态表示为周期性更新的检查点加近期输入缓冲区，将状态内存流量约减半，使投机解码回滚变为简单的缓冲区指针操作，并支持以低精度检查点进行输出计算。

    

    状态空间模型（SSM）维护一个固定大小的循环状态，但在每个解码步骤都对其进行读写可能成为混合模型的瓶颈。这种急切的状态管理会导致大量低算术强度的状态 I/O，使投机解码无法进行低成本的回滚，并将状态量化误差引入后续的循环更新中。我们提出了 ReplaySSM，它通过展开循环，将状态表示为一个检查点和一个近期 SSM 输入的缓冲区，从而将输出计算与检查点更新分离开来。通过周期性地更新检查点，ReplaySSM 在大多数步骤中避免了状态写入，并大约将主导性的状态内存流量减半。保留近期输入还使投机回滚成为缓冲区指针操作，并支持仅输出的验证，而无需具体化中间状态。同样的分离机制还允许低精度的检查点在大多数步骤中服务于输出计算，而更高精度的……（摘要原文在此处被截断）

    arXiv:2610.13778v1 Announce Type: cross  Abstract: State-space models (SSMs) maintain a fixed-size recurrent state, but reading and writing it at every decoding step can bottleneck hybrid models. This eager state management causes large state I/O with low arithmetic intensity, leaves speculative decoding no cheap rollback, and feeds state quantization error into subsequent recurrent updates. We introduce ReplaySSM, which unrolls the recurrence to represent the state as a checkpoint and a buffer of recent SSM inputs. This separates output computation from checkpoint updates. By updating the checkpoint periodically, ReplaySSM avoids state writes on most steps and roughly halves the dominant state memory traffic. Keeping recent inputs also makes speculative rollback a buffer-pointer operation and enables output-only verification without materializing intermediate states. The same separation allows a low-precision checkpoint to serve output computation on most steps while a higher-precisio
    
[^30]: 几何感知的多目标优化：范数约束预言机与熵权重更新

    Geometry-Aware Multi-Objective Optimization: Norm-Constrained Oracles with Entropic Weight Updates

    [https://arxiv.org/abs/2610.13773](https://arxiv.org/abs/2610.13773)

    该论文提出一种将范数约束线性最小化预言机与任务权重熵更新相结合的单循环多目标优化方法，对任意有限维范数对实现了确定性 $O(T^{-1/2})$ 与随机 $O(T^{-1/4})$ 的遍历平稳性速率，且随机情形的主项与任务数量无关。

    

    多梯度下降通常在欧几里得几何中进行表述，而矩阵感知的优化器则使用由其他范数诱导的方向。我们分析了一种单循环多目标方法，该方法将范数约束的线性最小化预言机与任务权重的熵更新相结合。对于任意有限维范数对，我们在确定性设定下建立了 $O(T^{-1/2})$ 的遍历平稳性速率，并在未正则化的情形下从光滑性与下有界性推导出所需的轨迹梯度界。在随机梯度与有界梯度假设下，仅需一个持久的混合梯度动量缓冲区即可获得 $O(T^{-1/4})$ 的速率，其主项对任务数量没有显式依赖。我们还对内层加权目标在任意有界梯度序列上建立了静态遗憾界，并且在引入熵正则化和权重下限后，可在预热阶段之后实现逐迭代的内层间隙控制，同时保持……（原文摘要在此处截断）

    arXiv:2610.13773v1 Announce Type: cross  Abstract: Multiple-gradient descent is typically formulated in Euclidean geometry, whereas matrix-aware optimizers use directions induced by other norms. We analyze a single-loop multi-objective method combining a norm-constrained linear minimization oracle with entropic updates of task weights. For any finite-dimensional norm pair, we establish an $O(T^{-1/2})$ ergodic stationarity rate in the deterministic setting, deriving the required trajectory gradient bound from smoothness and lower-boundedness in the unregularized case. With stochastic gradients and a bounded-gradient assumption, one persistent blended-gradient momentum buffer yields an $O(T^{-1/4})$ rate whose leading term has no explicit dependence on the number of tasks. We also establish static regret for the inner weighting objective on arbitrary bounded gradient sequences and, with entropy regularization and a weight floor, iteration-wise inner-gap control after burn-in while retai
    
[^31]: 从像素到隐私敏感特征：重构误差界的信息论框架

    From Pixels to Privacy-Sensitive Features: An Information-Theoretic Framework for Reconstruction Bounds

    [https://arxiv.org/abs/2610.13768](https://arxiv.org/abs/2610.13768)

    该论文提出了一个基于率失真理论的信息论框架，能在训练前计算出 DP-SGD 训练模型的重构误差下界，即使面对可观测训练过程、知晓其余训练数据和总体分布的最坏情况攻击者依然成立，并允许分析者选择面向隐私敏感特征的误差度量表示。

    

    arXiv:2610.13768v1 公告类型：cross 摘要：对于一个使用 DP-SGD 训练的模型，攻击者究竟能够恢复出什么？其 $(\varepsilon,\delta)$ 差分隐私保证并不能直接回答这个问题。我们推导出一个可以在训练之前计算的失真界，并且该界甚至适用于一类很强的攻击者：他们可以在训练过程中观察参数轨迹、知晓所有其他训练记录，以及训练记录所来自的总体分布。分析者可以自由选择度量误差所用的表示方式，使该误差界能够针对特定应用中真正涉及隐私敏感性的重构方面。我们的信息论框架将两项贡献分离开来：其一是通过条件互信息度量的、训练过程所泄露的关于某条记录的信息量；其二是通过信源熵度量的、数据分布中原本存在的不确定性。率失真理论将二者结合，得到对所有重构方法期望误差的一个下界。

    arXiv:2610.13768v1 Announce Type: cross  Abstract: What can an adversary recover from a model trained with DP-SGD? Its $(\varepsilon,\delta)$ guarantee does not answer this question directly. We derive a distortion bound that can be computed before training and applies even to an adversary who observes the parameter trajectory during training, knows every other training record and the population distribution from which the training records are drawn. The analyst may choose the representation in which error is measured, allowing the bound to target the aspects of reconstruction that are privacy-sensitive in a given application. Our information-theoretic framework separates two contributions: the information about a record revealed by training, measured through conditional mutual information, and the uncertainty already present in the data distribution, measured through source entropy. Rate-distortion theory combines them into a lower bound on the expected error of every reconstruction a
    
[^32]: DualQuant：面向大语言模型的双尺度免校准权重与激活量化

    DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs

    [https://arxiv.org/abs/2610.13767](https://arxiv.org/abs/2610.13767)

    提出了免校准的双尺度训练后量化框架DualQuant，利用最小化权重重建误差所得的缩放因子作为激活缩放，无需校准数据即可同时完成大语言模型的权重与激活低比特量化。

    

    在原生支持4比特的加速器上进行低比特推理，需要同时对权重和激活进行量化，才能充分发挥这些格式所提供的吞吐量提升。虽然训练后权重量化已得到广泛研究，但激活量化仍然更具挑战性，且通常需要校准数据。在这项工作中，我们提出了DUALQUANT，一个双尺度、免校准的训练后量化框架，它直接最小化权重重建的均方误差（MSE），并作为直接副产品，在激活量化之前产生逐通道的激活缩放因子。我们的关键观察是，通过最小化权重重建误差获得的缩放因子能够提供有效的激活缩放，从而在无需校准数据的情况下实现具有竞争力的激活量化。我们将DUALQUANT表述为一个不可微的权重重建问题，并开发了一个高效的求解器。

    arXiv:2610.13767v1 Announce Type: cross  Abstract: Low-bit inference on accelerators with native 4-bit support requires quantizing both weights and activations to realize the full throughput gains these formats provide. While post-training weight quantization has been extensively studied, activation quantization remains considerably more challenging and typically requires calibration data. In this work, we introduce DUALQUANT, a dual-scale, calibration-free post-training quantization framework that directly minimizes the weight-reconstruction Mean Squared Error (MSE) and, as an immediate by-product, yields per-channel activation rescalings prior to activation quantization. Our key observation is that the scaling factors obtained by minimizing weight-reconstruction error provide effective activation rescalings, enabling competitive activation quantization without calibration data. We formulate DUALQUANT as a non-differentiable weight-reconstruction problem and develop an efficient solve
    
[^33]: 地理空间基础模型嵌入中的捷径学习：面向跨灾害的损毁评估

    Shortcut Learning in Geospatial Foundation Model Embeddings for Cross-Disaster Damage Assessment

    [https://arxiv.org/abs/2610.13766](https://arxiv.org/abs/2610.13766)

    该研究揭示了地理空间基础模型在灾害损毁评估中存在捷径学习问题——留出整个灾害事件测试时性能大幅下降，且仅凭建筑物数量或灾前图像就能取得较高准确率，说明模型可能依赖事件/场景特征而非真正可迁移的损毁证据。

    

    地理空间基础模型可以支持灾害损毁评估，但在熟悉灾害上的高准确率可能隐藏着捷径学习：即基于事件或场景特征进行预测，而非依据能够迁移到新灾害的证据。我们研究了 TerraMind、Prithvi-EO-2.0、DOFA 和 DINOv3 在多灾害数据集 xView2 和 BRIGHT 上的嵌入表示。与随机划分图像切片相比，留出整个灾害进行测试会使 xView2 上的平均平衡准确率从 0.807 降至 0.638，BRIGHT 上的从 0.818 降至 0.568。四组实验检验了何种信息支持预测：事件与采集背景信息、灾前/灾后图像输入、建筑物数量以及正确的图像配对。结果表明灾前图像本身仍具有预测能力，而仅使用标注建筑物数量的分类器即可达到 0.629 和 0.653 的平衡准确率。在 xView2 上，结合灾前/灾后特征平均表现最佳，但并未超过仅使用灾前特征的表现。

    arXiv:2610.13766v1 Announce Type: cross  Abstract: Geospatial foundation models can support disaster damage assessment, but high accuracy on familiar disasters may hide shortcut learning: prediction from event or scene characteristics rather than evidence that transfers to a new disaster. We study embeddings from TerraMind, Prithvi-EO-2.0, DOFA, and DINOv3 on the multi-disaster datasets xView2 and BRIGHT. Holding out an entire disaster, instead of randomly splitting image tiles, reduces mean balanced accuracy from 0.807 to 0.638 on xView2 and from 0.818 to 0.568 on BRIGHT. Four sets of tests examine what information supports prediction: event and acquisition context, before/after image inputs, building count, and correct image pairing. Pre-event images remain predictive, and a classifier given only the annotated number of buildings reaches balanced accuracy of 0.629 and 0.653. Combining before/after features performs best on average on xView2, but does not improve over pre-only feature
    
[^34]: 当边独立性失效时：基于潜在社交性先验的联合图扩散

    When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Priors

    [https://arxiv.org/abs/2610.13765](https://arxiv.org/abs/2610.13765)

    本文指出离散图扩散模型的E-R随机图先验是GNN去噪器失效的根本原因，并提出潜在社交性先验（LSP）与联合扩散过程，使节点社交性变量与图结构共同演化，从而恢复生成图中的节点异质性和高阶结构。

    

    离散图扩散模型通过具有马尔可夫噪声模型的独立边更新来破坏图结构。结合对输出分布的交换性要求，终端先验必然是Erdős–Rényi（E-R）随机图。这对无节点特征的无条件图生成具有根本性的影响：在大图极限下，任何应用于E-R图的GNN去噪器都近似输出E-R图，使得有效去噪变得困难。在实践中，生成的图无法恢复有意义的节点异质性和高阶结构。以往的模型引入专门的节点特征来弥补这一问题，将其归结为表达能力问题，而先验的作用则未被审视。我们转而从根本原因入手，引入潜在社交性先验（LSP）和一种联合扩散过程，使潜在节点社交性变量与图结构共同演化。

    arXiv:2610.13765v1 Announce Type: cross  Abstract: Discrete graph diffusion models corrupt graph structure through independent edge updates with a Markovian noise model. Combined with the exchangeability requirement on the output distribution, the terminal prior is necessarily an Erd\H{o}s--R\'enyi (E-R) random graph. This has a fundamental consequence for unconditional graph generation without node features: in the large graph limit, any GNN denoiser applied to an E-R graph approximately outputs an E-R graph, making effective denoising difficult. In practice, generated graphs fail to recover meaningful node heterogeneity and higher-order structure. Previous models introduced specialized node features to compensate, framing this as an expressivity problem, while the role of the prior remained unexamined. We instead address the root cause by introducing the Latent Sociability Prior (LSP) and a join} diffusion process that co-evolves latent node sociability variables and graph structure,
    
[^35]: 先验数据拟合网络能多好地摊销贝叶斯推断？

    How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?

    [https://arxiv.org/abs/2610.13764](https://arxiv.org/abs/2610.13764)

    本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。

    

    先验数据拟合网络（PFNs）被训练用于近似合成先验的贝叶斯后验预测分布（PPD），然而关于其近似质量的研究结果却出人意料地稀少。从经验风险最小化的视角出发，我们将PFNs的部署误差分解为先验误差、上下文误差和摊销误差，并在现实先验下相对于参考后验预测分布来度量摊销误差。结果表明，PFNs通常能很好地恢复目标后验预测分布。然而，我们发现部分残余摊销误差仍然存在，尤其是在先验预训练分布中的罕见设置下，即使扩大模型规模也无法消除，这表现为独立预训练的PFNs之间的分歧，并在推理时给预测器留下不确定性。作为补救措施，我们研究了多种类型的贝叶斯模型平均方法，发现这些方法能够持续减少这种不确定性并改善预测性能。

    arXiv:2610.13764v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) are trained to approximate the Bayesian posterior predictive distribution (PPD) of a synthetic prior, yet results about the approximation quality are surprisingly thin. Taking an empirical risk minimization perspective, we decompose the deployment error of PFNs into prior, context, and amortization error, and measure the latter against a reference PPD under a realistic prior. Results show that PFNs generally recover the targeted PPD very well. Nevertheless, we find that parts of the residual amortization error persist, especially in \textit{rare} settings under the prior pretraining distribution, even when scaling the model size, which manifests in disagreement between independently pretrained PFNs and leaves uncertainty about the predictor at inference time. As a remedy, we study multiple types of Bayesian model averages and find that these can consistently reduce this uncertainty and improve predicti
    
[^36]: 面向大语言模型智能体训练的匹配上下文群组策略优化

    Matched-Context Group Policy Optimization for LLM Agent Training

    [https://arxiv.org/abs/2610.13755](https://arxiv.org/abs/2610.13755)

    MCGPO通过匹配上下文从已有轨迹中构建加权动作偏好，在无需critic和额外rollout的情况下，为稀疏终端反馈下长时程LLM智能体的中间决策分配信用。

    

    训练长时程语言模型智能体需要从稀疏的终端反馈中为中间决策分配信用。在有限的同批rollout群组内，有用的动作比较取决于其上下文、两个备选方案的证据支持，以及这些证据如何进入学习过程。我们提出匹配上下文群组策略优化（MCGPO），这是一种从已收集轨迹中构建加权动作偏好的方法。MCGPO通过动作前的状态摘要和观察到的评分标准进展来定义比较上下文，并由任务规则进行标注。双动作支持机制在有序的精确匹配与粗略匹配规则中进行选择。固定折将候选决策与用于定向其偏好的终端结果分离开来，而依赖支持度的证据权重则在共享的GraphGPO任务学习之外，对辅助成对目标进行缩放。MCGPO复用rollout数据以及每个响应的原始历史，保持无critic（价值函数）的训练方式（摘要在此处截断）。

    arXiv:2610.13755v1 Announce Type: cross  Abstract: Training long-horizon language model agents requires assigning credit to intermediate decisions from sparse terminal feedback. Within a finite group of sibling rollouts, useful action comparisons depend on their context, evidence for both alternatives, and how that evidence enters learning. We propose Matched-Context Group Policy Optimization (MCGPO), a procedure for constructing weighted action preferences from already collected trajectories. MCGPO defines comparison contexts with pre-action state summaries and observed rubric progress, annotated by task rules. Two-action support selects among ordered exact and coarse matching rules. Fixed folds separate candidate decisions from the terminal outcomes used to orient their preferences, and support-dependent evidence weights scale an auxiliary pairwise objective alongside shared GraphGPO task learning. MCGPO reuses rollout data and each response's original history, preserving critic-free
    
[^37]: 使用小型数学世界研究任务多样性的影响

    Studying the Effects of Task Diversity Using Small Mathematical Worlds

    [https://arxiv.org/abs/2610.13750](https://arxiv.org/abs/2610.13750)

    本文提出了两个程序化生成的数学任务生态系统IntWorld和PermWorld，通过精确控制任务数量，实现了对任务多样性如何影响神经网络训练的系统研究。

    

    摘要：像模加法这样的基础数学任务一直是揭示神经网络行为某些方面的宝贵工具，而这些行为在嘈杂、不可验证、大规模的现实世界环境中往往难以观察。以往这些研究通常只聚焦于单一任务。然而，现代语言模型尽管仅通过下一个词预测目标进行训练，却接触到大量不同的任务。在这项工作中，我们引入了两个数学任务生态系统——IntWorld和PermWorld，它们由整数和排列上彼此不同但相互关联的任务构建而成。两者都被设计用于在一个内聚的数学“世界”中模拟任务多样性。由于数据集是程序化生成的，我们可以精确地改变任务的数量，从而开始将任务多样性的作用与训练中的其他因素区分开来。在描述了这些数据集之后，我们提供了从这些数据集中获得的一些初步结果。

    arXiv:2610.13750v1 Announce Type: new  Abstract: Basic mathematical tasks like modular addition have been a valuable tool for illuminating aspects of neural network behavior that are challenging to see in noisy, unverifiable, large-scale, real-world settings. These studies have typically focused on a single task. However, modern language models are exposed to a vast number of tasks despite being trained with a next-token prediction objective. In this work, we introduce two mathematical task ecosystems, IntWorld and PermWorld, built from distinct but related tasks on integers and permutations. Both are designed to simulate task diversity within a cohesive mathematical 'world' where tasks are distinct but related. Because the datasets are procedurally generated, we can precisely vary the number of tasks, allowing us to begin to disentangle the role of task variety from other factors in training. After describing the datasets, we provide some initial results obtained from these datasets w
    
[^38]: 只会打分而不会搜索的特权教师：面向推理的批评条件化在策略自蒸馏

    A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning

    [https://arxiv.org/abs/2610.13744](https://arxiv.org/abs/2610.13744)

    本文发现自蒸馏中的评分信号既无法定位学生推理中的错误，也无法可靠区分正确与错误的rollout，并指出错误定位本质上是搜索问题，因而提出批评条件化的在策略自蒸馏方法来弥补“只会打分、不会搜索”的评分教师的结构性缺陷。

    

    在策略自蒸馏有望用密集的token级信用信号取代可验证奖励强化学习中稀疏的结果信号：一个以参考解为条件的教师对学生在策略生成的rollout进行评分，由此产生的分布偏移再被蒸馏回策略中。我们研究这种评分信号是否真的包含纠正性信用。利用学生rollout中的黄金错误区间，我们检验了自蒸馏目标所依赖的两种信号——全分布KL散度和token对数比——是否具备两个关键性质：能否定位错误，以及能否将概率从错误的学生token上推开。结果表明，这两种信号均不具备上述性质，也无法在同一问题内可靠地区分正确与错误的rollout。我们认为这种失败是结构性的：在推理轨迹中定位错误本身就是一个搜索问题，而评分教师从未被明确要求执行这一任务。

    arXiv:2610.13744v1 Announce Type: new  Abstract: On-policy self-distillation promises to replace the sparse outcome signal of verifiable-reward RL with dense token-level credit: a teacher conditioned on the reference solution scores the student's own rollout, and the resulting distributional shift is distilled back into the policy. We investigate whether this scoring signal actually contains corrective credit. Using gold error spans in student rollouts, we test the two signals consumed by self-distillation objectives, the full-distribution KL and the token log-ratio, for two properties: whether they localize errors and whether they push probability away from erroneous student tokens. They exhibit neither of these properties nor can they reliably distinguish correct from incorrect rollouts within the same problem. We argue that this failure is structural: locating an error in a reasoning trace is itself a search problem, while a scoring teacher is never explicitly asked to perform that 
    
[^39]: SmolWAM：面向边缘设备机器人操作的潜在世界-动作模型

    SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices

    [https://arxiv.org/abs/2610.13740](https://arxiv.org/abs/2610.13740)

    SmolWAM通过复用20亿参数教师模型的预训练主干并精简推理接口，仅用3.6 GPU小时训练出一个轻量潜在世界-动作模型，在边缘设备上以快11倍的速度实现95.5%的操作成功率，同时仍保留预测场景运动的能力。

    

    世界-动作模型（WAM）能够同时预测机器人的动作以及这些动作将产生的观测结果。然而，目前最强的模型依赖于视频扩散Transformer，其计算开销对于资源受限的边缘设备而言过于昂贵。我们提出了SmolWAM，这是一种由20亿参数的Cosmos Policy衍生而来的潜在世界-动作模型：我们保留了其预训练主干，并重建了其推理接口，包括单帧视觉潜变量、保留主干注意力汇的缓存文本特征、28个Transformer块中的12个，以及1,008个token——动作和未来视觉潜变量可在一次整流流步骤中从中读出。仅需3.6 GPU小时的训练，SmolWAM便在400个LIBERO任务中取得95.5%的成功率，每个16动作块仅需57毫秒和2.3 TFLOP的计算量，GPU显存占用仅2.4 GiB，比其教师模型快11倍、FLOP减少18.5倍、显存减少3.2倍，且其世界模型头仍然能够预测场景的运动。

    arXiv:2610.13740v1 Announce Type: cross  Abstract: World-action models (WAMs) predict a robot's actions together with the observations those actions will produce. However, the strongest models rely on video diffusion transformers that are far too computationally expensive for resource-constrained edge devices. We present SmolWAM, a latent WAM that we derive from the 2B-parameter Cosmos Policy by keeping its pretrained trunk and rebuilding its inference interface: single-frame visual latents, cached text features that preserve the trunk's attention sinks, 12 of 28 transformer blocks, and 1,008 tokens from which actions and future visual latents are read out in one rectified-flow step. Trained in 3.6 GPU-hours, SmolWAM succeeds in 95.5% of 400 LIBERO episodes at 57 ms and 2.3 TFLOP per 16-action chunk with 2.4 GiB of GPU memory, 11x faster than its teacher with 18.5x fewer FLOPs and 3.2x less memory, and its world head still forecasts scene motion.
    
[^40]: SafeDiSTL：面向安全多智能体强化学习的联合约束分解

    SafeDiSTL: Decomposing Joint Constraints for Safe Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2610.13737](https://arxiv.org/abs/2610.13737)

    SafeDiSTL利用信号时序逻辑鲁棒性与可微动力学提供的动作敏感度，将团队级联合约束代价分解并分配到各智能体，解决了联合约束下多智能体强化学习的信用分配难题。

    

    协作型多智能体任务通常对共享目标或集体安全施加约束。团队层面的违规代价函数可以衡量这种约束结果，但无法区分哪些智能体的动作对该违规的影响最大。我们提出了SafeDiSTL，这是一个约束多智能体强化学习框架，即使在仅有联合约束度量的情况下也能实现信用分配。我们利用信号时序逻辑（STL）的鲁棒性，通过运行最大代价增量来衡量回合级的违规程度。平滑的全回合鲁棒性和可微动力学提供了动作敏感度，这些敏感度在同一决策时刻将每个代价增量分配到各个智能体上。在加权各智能体对每个代价增量的份额时，这些敏感度还考虑了延迟效应。我们在协作载荷运输、联网导航和网络化交通控制等任务中对该框架进行了研究。SafeDiSTL实现了1.6-5.1倍的（性能提升）。

    arXiv:2610.13737v1 Announce Type: cross  Abstract: Cooperative multi-agent tasks often impose constraints on a shared objective or collective safety. A team level violation cost function can measure this constraint outcome, but does not distinguish which agents' actions most strongly influence that violation. We present SafeDiSTL, a constrained multi-agent reinforcement learning framework that allows for credit assignment even with joint constraint measurements. We use Signal Temporal Logic (STL) robustness to measure episode-level violation through running-maximum cost increments. Smooth full-episode robustness and differentiable dynamics provide action sensitivities, which allocate each cost increment across agents at the same decision time. These sensitivities account for delayed effects when weighting agents' shares of each cost increment. We study the framework in cooperative payload transport, connected navigation, and networked traffic control tasks. SafeDiSTL achieves 1.6-5.1x 
    
[^41]: Orthros：面向高效Transformer的相位感知异构注意力

    Orthros: Phase-Aware Heterogeneous Attention for Efficient Transformers

    [https://arxiv.org/abs/2610.13735](https://arxiv.org/abs/2610.13735)

    Orthros注意力通过相位感知的方式在序列处理的不同阶段动态切换计算模式，实现了线性复杂度的预填充与完整注意力的解码相结合，从而在重预填充、轻解码的长文本场景下比原始Transformer具有更低的渐近复杂度。

    

    诸如线性注意力和滑窗注意力等高效注意力机制，旨在将Transformer的计算复杂度从二次方降至线性。然而，这些机制在上下文记忆能力上的退化往往需要交错插入完整的注意力层，导致长文本处理中的二次方瓶颈依然未得到解决。为解决这一问题，我们提出了Orthros注意力，它将效率范式从结构性分配转变为相位感知解耦。作为一个统一模块，它能够在序列处理的不同阶段动态切换计算模式，实现线性复杂度的预填充和完整注意力的解码。我们的设计使Orthros注意力在以“重预填充、轻解码”为特征的常见长文本场景中，相比原始Transformer具有更低的渐近复杂度。除了理论推导和复杂度分析之外，大量实证评估也证明了Orthros的有效性……

    arXiv:2610.13735v1 Announce Type: cross  Abstract: Efficient attention mechanisms, such as linear and sliding window attention, aim to reduce the computational complexity of Transformers from quadratic to linear. However, their degraded in-context recall often necessitates interleaving full attention layers, leaving the quadratic bottleneck unresolved in long-context processing. To address this, we introduce Orthros attention, which shifts the efficiency paradigm from structural allocation to phase-aware decoupling. As a unified module, it dynamically switches its computation pattern across different phases of sequence processing, enabling linear complexity prefilling and full attention decoding. Our design grants Orthros attention a lower asymptotic complexity compared to vanilla Transformers in prevalent long-context scenarios characterized by heavy prefilling and light decoding. Beyond theoretical formulation and complexity analysis, extensive empirical evaluations demonstrate Orthr
    
[^42]: SeoulMMOD：一个大规模多模态起讫点（OD）流基准数据集

    SeoulMMOD: A Large-Scale Multimodal Origin-Destination Flow Benchmark

    [https://arxiv.org/abs/2610.13730](https://arxiv.org/abs/2610.13730)

    SeoulMMOD是首个覆盖首尔全市6种出行方式、提供三年逐小时OD流数据及多类辅助信号的大规模多模态出行基准数据集，并在统一协议下系统评测了14种预测基线模型。

    

    起讫点（OD）流预测通过预测城市中人口流动的规模与模式，为城市出行决策提供支持。然而，现有的大多数公开OD基准数据集只能提供城市出行的局部视角，通常仅覆盖单一交通服务或至多两种出行方式。由于出行方式覆盖不完整，观测到的需求变化难以区分是不同交通方式之间的转移，还是总出行量的变化。我们提出了SeoulMMOD，一个面向首尔全市多模态出行的大规模公开基准数据集。SeoulMMOD提供了三年的逐小时OD流数据，估计了首尔25个区和426个分区中6种城市出行方式的总出行量，并同时提供出行时间、出行距离、日历、降雨量以及兴趣点（POI）等辅助信号。基于SeoulMMOD，我们在统一协议下对14种预测基线模型进行了基准测试，并分析了空间尺度、联合训练与分方式训练以及跨年……（摘要原文在此处截断）等方面的影响。

    arXiv:2610.13730v1 Announce Type: new  Abstract: Origin-destination (OD) flow forecasting supports urban mobility decisions by predicting the volume and patterns of human movement across a city. Most existing public OD benchmarks provide only a partial view of citywide mobility, covering a single transport service or at most two modes. With partial mode coverage, observed demand changes are difficult to separate into shifts across transport modes and changes in total travel. We introduce SeoulMMOD, a large-scale public benchmark for citywide multimodal mobility in Seoul. SeoulMMOD provides three years of hourly OD flows estimating total mobility for 6 urban travel modes across 25 districts and 426 sub-districts, together with travel-time, travel-distance, calendar, rainfall, and point-of-interest (POI) signals. Using SeoulMMOD, we benchmark 14 forecasting baselines under a common protocol and analyse the effects of spatial scaling, joint versus mode-separate training, and cross-year ge
    
[^43]: 基于图神经网络的摊销式反事实预测

    Amortized Counterfactual Prediction with Graph Neural Networks

    [https://arxiv.org/abs/2610.13724](https://arxiv.org/abs/2610.13724)

    提出反事实图神经过程（CFGNP），利用图神经网络端到端地直接从观测数据回答反事实查询，无需先估计因果图，从而避免因果图估计误差向下游反事实预测的传播。

    

    反事实推断研究观察到的结果在不同情形下可能有何不同，例如患者如果接受不同的治疗是否会康复。只要决策依赖于未观察到的替代方案的结果，此类问题就会出现。回答反事实查询通常需要了解潜在的因果结构，而这在实践中往往是不可得的。因此，一种常见的做法是先从观测数据中估计因果图，然后再进行反事实推断。然而，估计出的因果图中的误差可能会传播到下游的反事实预测中。为了应对这一挑战，我们提出了反事实图神经过程（CFGNP），这是一种端到端的方法，可以直接从观测数据回答反事实查询，而无需先估计因果图。CFGNP 利用图神经网络（GNN）来发掘对偶……（原文在此处截断）

    arXiv:2610.13724v1 Announce Type: cross  Abstract: Counterfactual inference asks how an observed outcome might have differed under different circumstances, such as whether a patient would have recovered had they received a different treatment. Such questions arise whenever decisions depend on the outcomes of alternatives that were not observed. Answering counterfactual queries generally requires knowledge of the underlying causal structure, which is often unavailable in practice. A common approach is therefore to first estimate the causal graph from observational data and then perform counterfactual inference. However, errors in the estimated causal graph can propagate to downstream counterfactual predictions. To address this challenge, we introduce the Counterfactual Graph Neural Process (CFGNP), an end-to-end approach that answers counterfactual queries directly from observational data without first estimating a causal graph. CFGNP uses Graph Neural Networks (GNNs) to exploit the dua
    
[^44]: 面向联邦社会公平k-均值聚类的机器遗忘

    Machine Unlearning for Federated Socially Fair k-Means Clustering

    [https://arxiv.org/abs/2610.13717](https://arxiv.org/abs/2610.13717)

    本文提出了一种联邦社会公平k-均值聚类的精确机器遗忘方法，通过为每个群体维护独立的本地摘要，使得在全局公平权重变化时未受影响机构的摘要依然有效，并在理想采样下证明了与客户端群体比例无关的初始化近似保证。

    

    联邦聚类可以支持跨医院的患者分层以及跨分布式金融数据集的客户细分等应用。负责任的人工智能要求对各人口群体进行公平表示，并支持数据删除，这在一定程度上源于《通用数据保护条例》（GDPR）中的删除权等权利。这些目标之间相互作用：删除记录可能会改变群体之间的平衡，并改变剩余数据总体的聚类结果。我们研究了联邦社会公平k-均值聚类的精确机器遗忘问题，其目标是最小化到最近中心的最大群体平均平方距离。我们的方法为每个群体保留单独的本地摘要，因此当全局公平权重发生变化时，未受影响的机构的摘要仍然有效。对于两个群体的情况，我们证明了在理想采样下与客户端群体比例无关的初始化近似保证，并且我们表明同种子重放可以复现重训练的结果。

    arXiv:2610.13717v1 Announce Type: new  Abstract: Federated clustering can support applications such as patient stratification across hospitals and customer segmentation across distributed financial datasets. Responsible AI calls for fair representation of demographic groups and support for data deletion, motivated in part by rights such as the GDPR's right to erasure. These goals interact: deleting records can change the balance between groups and alter the clustering of the remaining population. We study exact machine unlearning for federated socially fair $k$-means, whose objective is to minimize the largest group-average squared distance to the nearest center. Our approach keeps separate local summaries for each group, so summaries at unaffected institutions remain valid as global fairness weights change. For two groups, we prove an initialization approximation guarantee independent of client group proportions under ideal sampling, and we show that same-seed replay reproduces retrai
    
[^45]: 迈向理解GFlowNets的训练：一个理论视角

    Towards Understanding the Training of GFlowNets: A Theoretical Perspective

    [https://arxiv.org/abs/2610.13711](https://arxiv.org/abs/2610.13711)

    该论文为GFlowNets的流匹配、细致平衡和轨迹平衡三种训练目标建立了首个统一理论框架，用期望平衡损失定量刻画了采样分布误差，并证明了在多种采样分布与反向策略设置下的有限时间收敛保证。

    

    生成流网络（GFlowNets）学习以与正奖励成比例的方式采样离散对象的策略。它们通过最小化平衡目标来训练，精确满足平衡目标可以保证正确的采样。然而，将非零平衡损失和近似平稳性与采样精度联系起来的系统性理论，尤其是在采样分布和反向策略不断演化的情形下，仍然缺乏。为了弥合这一差距，我们为该情形下的流匹配、细致平衡和轨迹平衡建立了一个统一理论。我们首先用期望平衡损失的平方根来约束终端对称KL散度。然后，我们证明近似平稳性可以控制这种分布误差，误差由一个衡量参数化不完备性的切空间残差给出。最后，我们在固定或受控采样，以及固定、平衡训练或独立更新的反向策略条件下，建立了有限时间收敛性。在标准假设下……

    arXiv:2610.13711v1 Announce Type: new  Abstract: Generative Flow Networks (GFlowNets) learn policies that sample discrete objects in proportion to a positive reward. They are trained by minimizing balance objectives, whose exact satisfaction guarantees correct sampling. However, a comprehensive theory linking nonzero balance loss and approximate stationarity to sampling accuracy, especially under evolving sampling distributions and backward policies, is still lacking. To bridge this gap, we develop a unified theory for Flow Matching, Detailed Balance, and Trajectory Balance in this regime. We first bound terminal symmetric KL by the square root of the expected balance loss. Then, we show that approximate stationarity controls this distribution error up to a tangent-space residual measuring parameterization incompleteness. Finally, we establish finite-time convergence under fixed or controlled sampling and fixed, balance-trained, or separately updated backward policies. Under standard s
    
[^46]: 自适应 DHiSparse：基于负载自适应 KV 分层结构的推测解码

    Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy

    [https://arxiv.org/abs/2610.13708](https://arxiv.org/abs/2610.13708)

    Adaptive DHiSparse 在分层稀疏 KV 架构上支持推测解码，并根据负载动态调整 KV 缓存驻留策略，在 100K 上下文、128 并发下相比 HiSparse 将解码吞吐提升至 1.45 倍。

    

    长上下文推理服务往往在硬件达到有效吞吐量之前就耗尽了 KV 缓存内存。HiSparse 将历史 KV 存储在主机内存中，仅取回被选中的条目，但它在任何负载下都使用单一的缓存预算，并且不支持诸如 DSpark 和 DFlash2 等基于草稿模型的推测解码。我们提出了 Adaptive DHiSparse，它支持在分层稀疏 KV 架构上进行推测解码，并根据负载自适应地调整 KV 驻留策略。它保护必需的工作集，在 GPU 内存有富余时保留更多的历史记录，并通过精确的融合检索恢复其余部分。DSpark 和 DFlash2 的提议与接受规则保持不变：查询共享缺失行的检索，草稿 KV 使用窗口受限的环形缓冲区，验证暂存在层组之间被复用。在 100K 上下文和 128 并发条件下，GLM-5.3 配合 DFlash2 在每个解码 GPU 上达到 793 输出 token/秒，是 HiSparse 的 1.45 倍、非推测自适应服务的 1.27 倍。（摘要在此处截断）

    arXiv:2610.13708v1 Announce Type: new  Abstract: Long-context serving runs out of KV memory before the hardware reaches its useful throughput. HiSparse stores history in host memory and fetches only the selected entries, but it uses one cache budget at every load and does not handle draft-model speculative decoding such as DSpark and DFlash2. We present Adaptive DHiSparse, which supports speculative decoding over hierarchical sparse KV and adapts residency to load. It protects the required working set, retains additional history while GPU memory is spare, and recovers the rest through exact fused retrieval. DSpark and DFlash2 keep their proposal and acceptance rules: queries share missing-row retrieval, draft KV uses a window-bounded ring, and verification staging is reused across layer groups. At 100K context and concurrency 128, GLM-5.3 with DFlash2 delivers 793 output tokens/s per decode GPU, 1.45$\times$ HiSparse and 1.27$\times$ non-speculative adaptive serving. Held to the same l
    
[^47]: 一步到位的校准：联合学习既准确又校准良好的神经网络

    Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks

    [https://arxiv.org/abs/2610.13693](https://arxiv.org/abs/2610.13693)

    该论文提出通过双层 Stackelberg 优化将校准器与主干网络端到端联合训练，无需单独的校准数据集或专用校准损失，仅引入校准器学习率这一个超参数，即可同时学习到既准确又校准良好的神经网络。

    

    神经网络（NN）虽然具有高度的灵活性和通用性，但往往校准不佳。事后校准方法使用单独的校准数据集和额外的校准参数来调整已经训练好的模型的输出。这一过程增加了额外开销，需要对校准器进行单独优化以及后续的模型选择。在本文中，我们证明了标准的（事后）校准器可以通过双层 Stackelberg 公式与主干网络进行端到端联合训练。该公式通过使用交叉熵同时优化准确性和校准性能，消除了对单独校准集或专用校准损失函数的需求，仅引入了一个额外的超参数：校准器学习率。尽管我们的框架支持从温度缩放、矩阵缩放到神经网络层等各种校准器，但我们的结果表明，使用最终的分类层本身作为校准器即可得到既准确又校准良好的神经网络。

    arXiv:2610.13693v1 Announce Type: cross  Abstract: Neural networks (NNs), while highly flexible and versatile, are often poorly calibrated. Post-hoc calibration methods adjust the outputs of an already trained model using a separate calibration dataset and additional calibration parameters. This process adds overhead that requires separate optimization of the calibrator and subsequent model selection. In this paper, we show that standard (post-hoc) calibrators can be trained end-to-end with the backbone via a bilevel Stackelberg formulation. This formulation eliminates the need for a separate calibration set or a dedicated calibration loss by jointly optimizing accuracy and calibration using cross-entropy, introducing only one additional hyperparameter: the calibrator learning rate. Although our framework supports calibrators ranging from temperature and matrix scaling to NN layers, our results show that using the final classification layer itself as the calibrator yields accurate, wel
    
[^48]: 方差缩减策略梯度近乎极小极大最优

    Variance-Reduced Policy Gradient is Nearly Minimax-Optimal

    [https://arxiv.org/abs/2610.13692](https://arxiv.org/abs/2610.13692)

    该论文提出LVR-PG统一理论框架，证明了一类结合递归更新与防御性重要性采样的一阶方差缩减策略梯度算法，在弱梯度支配等最小假设下、无需梯度预处理器，即可达到 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 样本复杂度的近乎极小极大最优最后一次迭代全局收敛。

    

    标准策略梯度（PG）算法由于依赖蒙特卡洛更新而存在高估计方差的问题。虽然离策略方差缩减技术可以缓解这一问题，但要在当前建立最优样本复杂度，要么需要限制性假设（如一致有界的重要性权重），要么需要不实用的二阶操作。在本工作中，我们提出了无环方差缩减策略梯度（LVR-PG），这是一个统一的理论框架，抽象出一类将递归更新与防御性重要性采样相结合的一阶估计器，其中包括已知的估计器如PAGE-PG和STORM-PG。我们证明，在一般参数化策略的弱梯度支配条件下，该算法类能够实现 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 的最后一次迭代全局收敛。我们仅使用纯一阶更新，在最小假设条件下且无需梯度预处理器即可建立这一保证。

    arXiv:2610.13692v1 Announce Type: cross  Abstract: Standard Policy Gradient (PG) algorithms suffer from high estimation variance because they rely on Monte Carlo updates. While off-policy variance reduction mitigates this problem, establishing optimal sample complexity currently requires either restrictive assumptions, such as uniformly bounded importance weights, or impractical second-order operations. In this work, we introduce Loopless Variance-Reduced PG (LVR-PG), a unified theoretical framework that abstracts a class of first-order estimators that integrate recursive updates with defensive importance sampling, including known estimators such as PAGE-PG and STORM-PG. We prove that this algorithmic class achieves $\tilde{\mathcal{O}}(\epsilon^{-2})$ last-iterate global convergence under weak gradient domination for general parametric policies. We establish this guarantee using purely first-order updates, with minimal assumptions and without gradient preconditioners. By extending sta
    
[^49]: 基于Transformer的扩散模型上下文内分数算子学习

    Transformer-based In-Context Score Operator Learning for Diffusion Models

    [https://arxiv.org/abs/2610.13690](https://arxiv.org/abs/2610.13690)

    本文提出一种基于Transformer的上下文内分数算子学习框架，使扩散模型仅需少量示例样本即可在推理时适应从未见过的分布，无需再训练或参数更新就能完成反向扩散采样。

    

    扩散模型通常学习单一目标分布或由训练时固定的条件变量所索引的一族分布的生成动力学，这限制了它们仅凭少量样本适应前所未见分布的能力。我们为扩散模型引入了一种上下文内解算子学习框架，其中每个目标分布被视为从元分布中抽取的一个任务，并在推理时由经验上下文样本集来指定。我们的方法学习一个上下文内分数算子，该算子将上下文样本、带噪声的查询点以及扩散时间映射到依赖任务的分数，从而无需再训练或参数更新即可在未见任务上进行反向扩散采样。一个关键观察是，依赖任务的分数具有高斯核条件均值表示形式，其经验Nadaraya-Watson近似可以由一个Transformer精确表示。

    arXiv:2610.13690v1 Announce Type: new  Abstract: Diffusion models typically learn the generative dynamics of a single target distribution or a family of distributions indexed by conditioning variables fixed during training. This limits their ability to adapt to a previously unseen distribution from only a small set of examples. We introduce an in-context solution operator learning framework for diffusion models, in which each target distribution is treated as a task drawn from a meta-distribution and is specified at inference time by an empirical context set. Our method learns an in-context score operator that maps the context samples, a noisy query point, and the diffusion time to the task-dependent score, enabling reverse-diffusion sampling on unseen tasks without retraining or parameter updates. A key observation is that the task-dependent score admits a Gaussian-kernel conditional-mean representation. Its empirical Nadaraya--Watson approximation can be represented exactly by a mode
    
[^50]: X-AVDD：基于跨注意力的音视频数据集蒸馏

    X-AVDD: Cross-Attentive Audio-Visual Dataset Distillation

    [https://arxiv.org/abs/2610.13687](https://arxiv.org/abs/2610.13687)

    X-AVDD提出基于交叉注意力的音视频数据集蒸馏方法，通过将跨模态对齐与联合优化解耦，将端到端合成成本降低约8倍GPU小时，同时生成高度对齐的合成数据并显著优于现有蒸馏方法。

    

    音视频数据集蒸馏旨在从大规模配对的音视频数据集中合成一个小型数据集，同时几乎完整保留其对神经网络的训练效用。现有方法（如DM和AVDD）面临两个关键局限：1）内存消耗大，难以扩展到复杂架构和更高的每类样本数（SPC）设置；2）往往无法保持强跨模态语义对应关系。为解决这些问题，我们提出了X-AVDD，一种基于交叉注意力的蒸馏方法，在合成过程中显式捕获模态间的关系。通过将跨模态对齐与联合优化解耦，X-AVDD在内存效率上显著提升，将端到端合成成本降低了约8倍GPU小时，同时生成高度对齐的合成数据。实验表明，X-AVDD显著优于其他音视频蒸馏方法。在VGGSound这一音视频数据集上……（摘要原文在此处截断）

    arXiv:2610.13687v1 Announce Type: cross  Abstract: Audio-visual dataset distillation aims to synthesize a small dataset from a large paired audio-visual dataset, while nearly preserving the training utility for neural networks. Existing approaches such as DM and AVDD face two critical limitations: 1) they are memory-intensive, making them difficult to scale to complex architectures and higher samples-per-class (SPC) settings, and 2) they often fail to preserve strong cross-modal semantic correspondence. To tackle these issues, we propose X-AVDD, a cross-attention-based distillation method that explicitly captures inter-modal relationships during synthesis. By decoupling cross-modal alignment from joint optimization, X-AVDD is substantially more memory-efficient, reducing end-to-end synthesis cost by $\sim 8\times$ GPU-hours, while producing highly aligned synthetic data. Empirically, X-AVDD significantly outperforms other audio-visual distillation methods. On VGGSound, an audio-visual 
    
[^51]: 关于协变量感知时间序列基础模型中日历特征编码的研究

    On the Encoding of Calendar Features in Covariate-Informed Time Series Foundation Models

    [https://arxiv.org/abs/2610.13676](https://arxiv.org/abs/2610.13676)

    本研究首次系统比较了七种日历特征编码方式在时间序列基础模型中的表现，发现编码方式的选择会显著影响模型的零样本预测精度。

    

    时间序列基础模型日益提供协变量支持，其中日历信息是实践中最常见的分类协变量类型，因为它在任何时间点都可用。然而，这些特征的编码方式在很大程度上被视为实现细节而非研究对象，尽管编码是预训练模型在推理阶段所剩无几的设计选择之一。本研究证明，信息传入模型的方式会显著影响时间序列基础模型的预测精度。因此，我们在来自多个领域的九个数据集上，通过零样本滚动评估，对四种最先进的时间序列基础模型系统地比较了七种日历编码方式，包括三种传统的与目标无关的表示（序数编码、独热编码和周期编码）以及四种目标编码。这九个数据集受到周末和节假日等社会活动的影响。

    arXiv:2610.13676v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) increasingly offer covariate support, with calendar information being the most common type of categorical covariate in practice, as it is available for every point in time. However, how these features are encoded is largely treated as an implementation detail rather than as an object of study, despite encoding being one of the few design choices left for pretrained models at inference time. With the present study, we demonstrate that the way information is propagated into the model substantially affects forecast accuracy of TSFMs. Therefore, we systematically compare seven calendar encodings, including three conventional target-agnostic representations (ordinal, one-hot, and cyclic) and four target encodings, across four state-of-the-art TSFMs on nine datasets from multiple domains in a zero-shot rolling evaluation. The nine datasets are influenced by social activities, such as weekends and holidays,
    
[^52]: 并非所有维度生而平等：基于分组二次核的离群值感知注意力线性化

    Not All Dimensions Are Equal: Outlier-Aware Attention Linearization with Grouped Quadratic Kernels

    [https://arxiv.org/abs/2610.13674](https://arxiv.org/abs/2610.13674)

    该论文发现预训练注意力中少数离群维度主导了线性化的近似误差，并提出维度感知的OAL方法，通过Hadamard乘积表述将全局系数放宽为维度特定的线性项与跨维度二次交互，再以分组二次核在更少参数下实现高保真的注意力线性化。

    

    线性化softmax注意力为高效的Transformer推理提供了一条途径，但如何在近似中保留预训练注意力仍然充满挑战。我们识别出一个被标准线性化方法所忽略的重要近似误差来源：一小部分离群维度对查询-键点积的贡献不成比例地大，并占据了近似误差的很大一部分。受这种异质性的启发，我们提出了离群值感知注意力线性化（Outlier-Aware Attention Linearization, OAL），这是一种用于构建二次注意力近似的维度感知方法。从二阶泰勒近似出发，我们引入了一种Hadamard乘积表述，使各个查询-键的独立贡献得以显式呈现。该表述放宽了全局共享系数的限制，以支持维度特定的线性项和跨维度的二次交互，同时保持可分离的计算方式。为了以更少的参数保留这种灵活性，OAL将维度进行分组（原文在此处截断）。

    arXiv:2610.13674v1 Announce Type: cross  Abstract: Linearizing softmax attention offers a route to efficient Transformer inference, but preserving pretrained attention remains challenging. We identify an important source of approximation error overlooked by standard linearization methods: a small number of outlier dimensions contribute disproportionately to query-key dot products and account for a large fraction of the approximation error. Motivated by this heterogeneity, we propose Outlier-Aware Attention Linearization (OAL), a dimension-aware method for constructing quadratic attention approximations. Starting from a second-order Taylor approximation, we introduce a Hadamard-product formulation that makes individual query-key contributions explicit. This formulation relaxes globally shared coefficients to support dimension-specific linear terms and cross-dimensional quadratic interactions while preserving separable evaluation. To retain this flexibility with fewer parameters, OAL gro
    
[^53]: 用于专用神经网络模型自动调优的多智能体LLM系统

    A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models

    [https://arxiv.org/abs/2610.13663](https://arxiv.org/abs/2610.13663)

    本文提出一个由分析师、编码员和记录员组成的多智能体LLM框架，在保持整体网络结构不变的前提下自动生成并调优指定关键组件的代码，并可迁移至脉冲神经网络分类与PINN求解高阶微分方程等不同类型模型。

    

    许多机器学习方法都是两级结构：专家先选定一类方法，而在该类方法内部，关键组件的选择至关重要且通常依赖人工。本文致力于实现这些组件的自动调优，并提出了一种可跨不同性质模型迁移的框架。该多智能体大语言模型（LLM）框架包含三个角色——分析师、编码员和记录员——并配备实验日志和沙盒执行环境。与优化整个机器学习脚本或在离散空间中搜索架构的系统不同，该框架在一个开放但语义上狭窄的空间中运作：智能体仅为指定的组件生成代码，而保持整体网络结构不变。该方法在两个方向上得到验证：调优基于相关性学习的分类脉冲神经网络，以及使用物理信息神经网络（PINN）求解高阶微分方程。

    arXiv:2610.13663v1 Announce Type: new  Abstract: Many machine learning approaches are two-level: an expert selects a class of methods, within this class the choice of key components is critical and usually manual. This paper addresses the automatic tuning of these components and proposes a framework transferable across models of different nature. The multi-agent large language model (LLM) framework has three roles---analyst, coder, and recorder~---with an experiment log and a sandboxed execution environment. Unlike systems that optimize the entire ML script or search architectures in a discrete space, the framework operates over an open but semantically narrow space: the agent generates code only for the specified components, leaving the overall network structure unchanged. The approach is validated in two directions: tuning a classification spiking neural network with correlation learning, and solving high-order differential equations with Physics-Informed Neural Networks (PINN). Acro
    
[^54]: 预训练与测试时训练预算对ARC至关重要：VARC与TRM的对照研究

    Pretraining and Test-Time Training Budgets Matter for ARC: A Controlled Study of VARC and TRM

    [https://arxiv.org/abs/2610.13660](https://arxiv.org/abs/2610.13660)

    本文在统一协议下控制变量地对比视觉型VARC与递归型TRM模型，证明预训练与测试时训练预算（尤其是增加TTT样本呈现次数）对ARC性能有决定性影响，提示性能差异不能简单归因于架构本身。

    

    抽象与推理语料库（ARC）越来越多地被用于评估神经网络能否从少量示例中推断出任务特定的变换。近期的ARC模型不仅在架构上存在差异，在更广泛的训练与评估协议上也各不相同，包括预训练数据、数据增强、测试时设置和优化预算，这使得关于架构优势的结论备受争议。在本工作中，我们在ARC-AGI-1上采用统一的归纳协议，研究视觉启发的VARC模型与基于递归的TRM模型的性能如何受预训练和测试时训练（TTT）预算的影响。我们将五个预训练预算与每个测试任务的四个TTT预算进行交叉组合，每个预算规定了任务变体数量和总样本呈现次数，从而定义了受控的训练预算。我们观察到，在所有设置下，增加TTT样本呈现次数均能提升两种模型的标准ARC pass@2指标。

    arXiv:2610.13660v1 Announce Type: cross  Abstract: The Abstraction and Reasoning Corpus (ARC) is increasingly used to evaluate whether neural networks can infer task-specific transformations from few demonstrations. Recent ARC models differ in architecture, but also in broader training and evaluation protocols, including pretraining data, data augmentation, test-time setup, and optimization budgets, making conclusions about architecture benefits debatable. In this work, we study how the performance of the vision-inspired VARC and recursion-based TRM models is affected by pretraining and test-time training (TTT) budgets under a common inductive protocol on ARC-AGI-1. We cross five pretraining budgets with four TTT budgets for each test task. Each budget specifies a task-variant count and total sample presentations, defining the controlled training budgets. We observe that increasing TTT sample presentations improves the standard ARC pass@2 metric in all settings for both models. Increas
    
[^55]: 面向非高斯空间场的可扩展稀疏自回归变分推断

    Scalable Sparse Autoregressive Variational Inference for Non-Gaussian Spatial Fields

    [https://arxiv.org/abs/2610.13656](https://arxiv.org/abs/2610.13656)

    该论文提出了一种可扩展的变分推断框架，通过在变分族中镜像稀疏最近邻自回归先验结构、使用归一化流条件分布和共享神经条件器摊销参数，并以双随机优化实现对数多项式迭代复杂度，从而高效解决大规模非高斯空间场的后验推断难题。

    

    潜场模型是空间统计学中的重要工具，但即使采用高斯过程（GP）先验，只要似然函数是非高斯的，潜场的后验分布在解析上便是不可求解的。现有的近似方法对于大规模非高斯数据集而言，要么速度太慢，要么限制过多。我们提出了一种可扩展的变分推断框架，该框架在变分族中镜像了先验的稀疏最近邻自回归结构，使用归一化流条件分布以增强表达能力，并通过单个共享的神经条件器对所有变分参数进行摊销。结合缩减祖先集的双随机优化方法，实现了每次迭代的对数多项式复杂度。

    arXiv:2610.13656v1 Announce Type: cross  Abstract: Latent field models are essential for spatial statistics, but the posterior distribution of the latent field, even with a Gaussian process (GP) prior, is analytically intractable whenever the likelihood is non-Gaussian. Existing approximations are either too slow or too restrictive for large non-Gaussian datasets. We present a scalable variational inference framework that mirrors the sparse nearest-neighbor autoregressive structure of the prior in the variational family, uses normalizing-flow conditionals for expressiveness, and amortizes all variational parameters via a single shared neural conditioner. Doubly stochastic optimization with reduced ancestor sets achieves polylogarithmic complexity per iteration.
    
[^56]: TabReliBench：超越预测精度的表格基础模型可靠性评估

    TabReliBench: Evaluating Tabular Foundation Models Beyond Predictive Accuracy

    [https://arxiv.org/abs/2610.13654](https://arxiv.org/abs/2610.13654)

    提出了 TabReliBench，一个从校准、分布质量与区间效率等可靠性维度超越单纯预测精度来系统评估表格基础模型的大型分类与回归基准，发现模型排名随可靠性指标而变化，且良好的边际覆盖并不代表有效的条件覆盖。

    

    表格基础模型（TFM）因其强大的预测性能而日益流行。其中许多模型还提供后验预测分布（PPD），但其可靠性仍然较少被研究。我们通过点预测、分布质量、边际与子群校准以及区间效率来评估可靠性。随后，我们提出了 TabReliBench，一个用于上述比较以及数据集层面分析的大型分类与回归基准。我们发现，TFM 的排名在 TabArena 与我们的基准之间、以及在不同可靠性指标之间会发生变化。回归类 TFM 通常能够达到接近名义水平的边际覆盖率并保持较窄的区间，但这种良好的边际表现并不意味着有效的条件覆盖。数据集层面的分析显示，分类难度往往伴随着较差的校准，而回归任务的预测性能并不能可靠地指示其校准情况。集成方法可以提升预测性……（原文在此截断）

    arXiv:2610.13654v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) are increasingly popular for their strong predictive performance. Many also provide posterior predictive distributions (PPDs), whose reliability remains less studied. We assess reliability through point prediction, distributional quality, marginal and subgroup calibration, and interval efficiency. We then introduce TabReliBench, a large classification and regression benchmark for these comparisons and dataset-level analyses. We find that TFMs' rankings change between TabArena and our benchmark and across reliability metrics. Regression TFMs usually achieve near-nominal marginal coverage and retain narrow intervals, but this favorable marginal behavior does not imply valid conditional coverage. Dataset-level analyses show that classification difficulty often accompanies poor calibration, whereas regression predictive performance is not a reliable indicator of calibration. Ensembling improves predictive per
    
[^57]: 对测试时锐化大语言模型中幂采样的批判性审视

    A Critical Look at Power Sampling when Test Time Sharpening LLMs

    [https://arxiv.org/abs/2610.13653](https://arxiv.org/abs/2610.13653)

    该论文证明基于MCMC/SMC的幂采样方法在现实计算预算下无法收敛到目标幂分布，其收益可由简单的Best-of-N采样复现，表明其成功源于筛选高概率序列而非真正近似幂分布。

    

    带可验证奖励的强化学习（RLVR）已显著提升了大语言模型的推理能力，然而越来越多的证据表明，它主要进行的是分布锐化：将概率质量集中在基础模型本身已经能够产生的推理路径上。因此，近期研究试图在不需要奖励信号的情况下获得RLVR的收益，其方法是通过马尔可夫链蒙特卡洛（MCMC）或序列蒙特卡洛（SMC）从幂分布 $\pi_\alpha \propto \pi_{\text{base}}^\alpha$ 中进行采样。我们证明，在现实的计算预算下，这些采样器无法收敛到 $\pi_\alpha$，并且它们的收益可以通过简单的 Best-of-$N$ 采样（配合适当的评分）大体复现。这表明它们成功的原因在于选择了高概率序列，而非近似 $\pi_\alpha$，由此引出一个自然的问题：幂分布本身是否是一个值得追求的目标？我们通过微调基础模型使其匹配 $\pi_\alpha$ 来直接检验这一假设，从而绕开了采样器带来的复杂性。

    arXiv:2610.13653v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has substantially improved LLM reasoning, yet growing evidence suggests it mainly performs distribution sharpening: concentrating probability mass on reasoning paths the base model already produces. Recent work therefore seeks RLVR's gains without needing reward signals, by sampling from the power distribution $\pi_\alpha \propto \pi_{\text{base}}^\alpha$ via Markov chain Monte Carlo or sequential Monte Carlo. We show that these samplers fail to converge to $\pi_\alpha$ under realistic compute budgets, and that their gains are largely recreated via simple Best-of-$N$ with appropriate grading . This suggests they succeed by selecting high-probability sequences, not by approximating $\pi_\alpha$, raising a natural question: is the power distribution itself a worthwhile target? We test this hypothesis directly by fine-tuning the base model to match $\pi_\alpha$, sidestepping sampler co
    
[^58]: 基于替代泛函的电子密度信息分子性质预测

    Electron-Density-Informed Molecular Property Prediction Using Surrogate Functionals

    [https://arxiv.org/abs/2610.13649](https://arxiv.org/abs/2610.13649)

    该方法将基态电子密度作为模型输入并强制其满足替代泛函最小化条件，在QM9和QM9数据集的广延分子性质预测上超越了现有最先进模型，并在更大分子上展现出更强的泛化能力。

    

    机器学习方法正越来越多地被用于直接从分子结构预测化学性质，以绕过量子力学计算中常常令人望而却步的成本。近期研究表明，将基态电子密度作为机器学习模型的输入可以提高此类任务的预测精度。本工作采用了最初为无轨道密度泛函理论开发的替代泛函框架，并将其扩展到分子性质预测任务中。我们证明，将基态密度作为输入传递给模型，同时强制该密度成为替代泛函的最小值，在QM9数据集的七个广延性质中有六个超越了最先进的性能。此外，该方法在来自QM40数据集的更大分子上表现出更好的泛化能力。在两个数据集的所有广延性质上，我们的方法均优于现有模型。

    arXiv:2610.13649v1 Announce Type: cross  Abstract: Machine learning methods are increasingly used to predict chemical properties directly from the molecular structure to bypass the often prohibitive cost of quantum mechanical calculations. Recent work has shown that including the ground-state electron density as input to the machine learning model can improve accuracy in this setting. This work employs the recent surrogate functional framework originally developed for orbital-free density functional theory and extends it to predicting molecular properties. We show that passing the ground-state density as input to the model, while simultaneously enforcing it to be the minimum of the surrogate functional, surpasses state-of-the-art performance for six out of seven extensive properties of the QM9 dataset. Additionally, the approach shows improved generalization to larger molecules taken from the QM40 dataset. On all extensive properties of both datasets, our approach outperforms models th
    
[^59]: SelectArena：面向真实世界表格数据的现代特征选择基准测试

    SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data

    [https://arxiv.org/abs/2610.13644](https://arxiv.org/abs/2610.13644)

    该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。

    

    从表格数据集中选择特征子集（列）是现实应用中普遍存在的挑战。行业专家希望最小化模型训练成本，气候科学家需要可解释的因素子集来理解极端天气事件，而临床医生由于财务和时间限制需要在各类诊断测试中做出选择。当从业者查阅科学文献以确定应使用哪种特征选择方法时，他们面对的是一个缺乏代表性的基准测试现状。尽管相关研究众多，现有的特征选择基准测试所能提供的指导却十分有限，因为它们依赖于过时的模型、临时设计的评估协议，或无法反映真实问题的人工生成数据。为了弥合科学基准测试与现实应用之间的差距，我们推出了 SelectArena，一个现代化的特征选择基准测试。我们比较了15种广泛使用的特征选择方法……

    arXiv:2610.13644v1 Announce Type: cross  Abstract: Selecting a subset of features (columns) from a tabular dataset is a ubiquitous challenge in real-world applications. Industry experts seek to minimize model training costs, climate scientists require an interpretable subset of factors to understand extreme weather events, and clinicians need to choose among diagnostic tests due to financial and time constraints. When looking at the scientific literature to determine which feature selection method to use, practitioners across domains face an unrepresentative landscape of benchmarking. Despite extensive research, existing feature selection benchmarks provide limited guidance because they rely on outdated models, ad hoc evaluation protocols, or artificially generated data that do not reflect real-world problems. To close the gap between scientific benchmarking and real-world applications, we introduce SelectArena, a modern benchmark for feature selection. We compare 15 widely used featur
    
[^60]: PANDA：先验引导的注意力双路径架构

    PANDA: Prior-guided Attentional Dual-path Architecture

    [https://arxiv.org/abs/2610.13635](https://arxiv.org/abs/2610.13635)

    提出了PANDA——一种可插入两阶段消息传递超图神经网络的先验引导双路径架构，通过将接收者归一化注意力与结构先验插值的主路径和非竞争性辅助路径相结合，克服了固定权重与归一化注意力在异质场景下的局限性。

    

    超图神经网络（HNNs）旨在建模超越成对交互的高阶关系，其通常依赖于固定的结构权重或归一化的注意力机制。固定权重在异质（heterophilic）场景中可能传播误导性信息，而归一化会使注意力系数具有竞争性——即在消息传递框架中，增大某个发送者的权重必然会降低同一聚合集合中其他发送者的权重。我们的主要贡献是提出了PANDA（先验引导的注意力双路径架构，Prior-guided AttentioNal Dual-path Architecture），这是一个适用于两阶段消息传递超图神经网络的插件，可同时应用于无向和有向设置。其主路径在学习到的接收者归一化注意力与结构关联先验之间进行插值，而非竞争性（非归一化）的辅助路径则允许每个发送者贡献其自身变换后的信息。我们还证明，在固定结构系数的情况下……（摘要原文在此处截断）

    arXiv:2610.13635v1 Announce Type: new  Abstract: Hypergraph Neural Networks (HNNs), designed to model higher-order relations beyond pairwise interactions, typically rely either on fixed structural weights or on normalised attention. Fixed weights can propagate misleading information in heterophilic settings, while normalisation makes attention coefficients competitive, in the sense that, in a message-passing framework, increasing the weight of one sender must reduce the weight of others in the same aggregation set. Our main contribution is the introduction of PANDA (Prior-guided AttentioNal Dual-path Architecture), a plug-in for two-stage message passing HNNs applicable to both undirected and directed settings. The main path interpolates between learned receiver-normalised attention and structural incidence priors, while a non-competitive (not normalised) auxiliary path lets each sender contribute its own transformed information. We also show that, with fixed structural coefficients, P
    
[^61]: 隐私保护与分布鲁棒的联邦多分类

    Private and Distributionally Robust Federated Multiclass Classification

    [https://arxiv.org/abs/2610.13631](https://arxiv.org/abs/2610.13631)

    该论文提出了一个差分隐私且分布鲁棒的联邦多分类框架，基于ADMM算法求解联邦Wasserstein分布鲁棒目标，并通过注入校准高斯噪声为客户端数据提供样本级隐私保护和对标签噪声的鲁棒性。

    

    资产密集型行业中的故障诊断需要在多个地理分布的客户端站点上采集的含噪、敏感数据来训练多分类模型。联邦学习能够在不集中数据的情况下实现协同训练，但大多数方法无法为存储在客户端的单条记录提供正式的隐私保证，并且在标签噪声下性能可能退化。为解决这些局限性，我们开发了一个差分隐私、分布鲁棒的联邦多分类框架，为存储在每个客户端的记录提供样本级隐私保护。我们基于带有Crammer-Singer损失的联邦Wasserstein分布鲁棒多分类目标函数，并使用基于交替方向乘子法（ADMM）的算法进行求解。为保护发布的客户端更新，我们的方法向本地模型以及（截断）的近端线性化客户端更新中注入经过校准的高斯噪声。

    arXiv:2610.13631v1 Announce Type: new  Abstract: Fault diagnosis in asset-intensive industries requires multiclass models trained on noisy, sensitive data collected across geographically distributed client sites. Federated learning enables collaborative training without centralizing data, but most approaches do not provide formal privacy guarantees for individual records stored at client sites and can degrade under label noise. To address these limitations, we develop a differentially private, distributionally robust framework for federated multiclass classification that provides sample-level privacy for records stored at each client. We build on a federated Wasserstein distributionally robust multiclass objective with the Crammer-Singer loss, solved using an Alternating Direction Method of Multipliers (ADMM)-based algorithm. To protect the released client updates, our method injects calibrated Gaussian noise into the proximal linearized client updates of both the local model and the l
    
[^62]: JitterVLA：一个用于评估VLA驱动人机操作中运动质量的重放基准

    JitterVLA: A Replay Benchmark for Motion Quality in VLA-Driven Human-Robot Manipulation

    [https://arxiv.org/abs/2610.13628](https://arxiv.org/abs/2610.13628)

    提出JitterVLA重放基准，通过在不同控制器间重放相同的指令和到达时间轨迹来隔离控制器对运动质量的影响，并证明简单的插值方法即可将人机操作中的p99加加速度降低83%（代价仅为2厘米跟踪误差）。

    

    视觉-语言-动作（VLA）策略发出的目标较为稀疏，且其到达机器人控制器的时机存在延迟和不规则性，因此即使预测的动作合理，也可能在人体附近产生突兀的运动。现有的补救措施往往在不同的数据流、延迟条件和指标下进行评估。JitterVLA通过替代控制器重放完全相同的指令和到达时间轨迹，从而隔离出控制器本身的独立影响。作者将该基准应用于合成抓取动作、来自20名受试者的手部轨迹、三个开源VLA流水线（共838个回合）以及一台物理UR3机器人。结果表明，插值方法降低了所有数据来源的建模p99加加速度，在200毫秒延迟下的人体运动中降低了83%（从615降至106 m/s^3），代价是2.0厘米的跟踪误差，且该效果在598个新回合上得到了复现；限制加加速度的生成方法能以更大的跟踪代价进一步降低加加速度。熵与运动不连续性之间的关联在两个流水线中呈现出相反但均可复现的符号，熵门控在一个流水线中有效而在另一个中无效。在UR3上，插值降低了加速度……（摘要截断）

    arXiv:2610.13628v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies issue sparse targets whose arrival at the robot controller is delayed and irregular, so a reasonable predicted action can still produce abrupt motion near people. Remedies are evaluated on different streams, delays and metrics. JitterVLA replays identical commands and arrival traces through alternative controllers, isolating the controller's effect. We apply it to synthetic reaches, hand trajectories from 20 people, three open VLA pipelines (838 episodes) and a physical UR3. Interpolation lowered modeled p99 jerk for every source, by 83% (615 to 106 m/s^3) on human motion with 200 ms delay, at a 2.0 cm tracking cost, and the effect replicated on 598 new episodes; jerk-limited generation lowered jerk further at a larger cost. Entropy-discontinuity associations had opposite, replicated signs in two pipelines, and entropy gating helped in one but not the other. On the UR3, interpolation lowered accele
    
[^63]: 基于核方法的实验研究外部有效性

    Kernel-based External Validity of Experimental Studies

    [https://arxiv.org/abs/2610.13625](https://arxiv.org/abs/2610.13625)

    该论文提出用核方法（最大均值差异MMD）形式化定义实验研究的外部有效性，并建立了研究规模与MMD分位数之间的渐近幂律关系，从而能够从小规模初步研究估计出具有外部有效性的研究所需规模。

    

    一项研究的外部有效性是指其研究结果能够泛化到未见过的实验条件（如数据集）的程度。我们将研究结果及其研究问题共同形式化，并将外部有效性定义为：该研究的两次独立且规模相同的重复实验所产生的结果之间的最大均值差异（MMD）低于给定容差的概率。这一方法以结果空间的选择以及适用于MMD的合适再生核为核心，足够灵活，可应用于多种不同类型的结果和研究问题。此外，我们建立了研究规模与MMD分位数之间的渐近幂律关系，并利用该关系从小规模的初步研究中估计出具有外部有效性的研究所需的规模。我们将该框架应用于带并列的排名问题，并讨论了适合不同研究问题的核函数。

    arXiv:2610.13625v1 Announce Type: cross  Abstract: The external validity of a study is the extent to which its findings generalize to unseen experimental conditions, such as datasets. We formalize the results of a study together with its research question, and define external validity as the probability that two independent and same-sized replications of the study produce results whose maximum mean discrepancy (MMD) is below a given tolerance. This approach, centered on the choice of a space of results and of an appropriate reproducing kernel for the MMD, is flexible enough to be deployed on multiple different types of results and research questions. Additionally, we establish an asymptotic power-law relation between the size of a study and the quantiles of the MMD, which we use to estimate the desired size of an externally valid study from a small-scale preliminary study. We deploy our framework to rankings with ties and discuss kernels suited to different research questions. Experime
    
[^64]: 关于生成扩散与最优传输之间联系的一些注记

    Some remarks on the connection between generative diffusion and optimal transport

    [https://arxiv.org/abs/2610.13614](https://arxiv.org/abs/2610.13614)

    本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。

    

    我们讨论了生成扩散与最优传输之间的三个简单联系。首先，数据与其 Ornstein-Uhlenbeck 加噪过程的联合分布在每一时刻都是某个熵正则化最优传输问题的解，其温度由噪声水平决定。因此，反向扩散可以被解读为一种逐渐降低熵正则化的退火过程。其次，我们研究了物种形成转变（speciation transition）——即数据分布的各个模式在生成过程中如何浮现——并比较了它在扩散模型、流匹配与最优传输中的发生方式。以高斯混合分布作为高维例子，我们表明扩散与流匹配借以分离各模式的动力学不稳定性，在最优传输情形下变为生成初始阶段一个指数级薄的边界层，而该边界层会被熵正则化所平滑。第三，考虑一个带有一般前向动力学的……（摘要原文在此处截断）

    arXiv:2610.13614v1 Announce Type: cross  Abstract: We discuss three simple connections between generative diffusion and optimal transport. First, the joint law of the data and of their Ornstein-Uhlenbeck noising is, at every time, the solution of an entropic optimal-transport problem, at a temperature set by the noise level. Backward diffusion can consequently be read as an annealing which gradually decreases the entropic regularization. Second, we study the speciation transition--how the modes of the data distribution emerge during generation--and compare how it happens in diffusion and flow matching with respect to optimal transport. Using Gaussian mixtures as high-dimensional example, we show that the dynamical instability through which diffusion and flow matching separate the modes becomes, for optimal transport, an exponentially thin boundary layer at the beginning of generation, which is rounded by entropic regularization. Third, considering a generic forward dynamics with a gene
    
[^65]: 基于对抗训练的连续声压级控制语音强度转换

    Vocal Intensity Conversion with Continuous SPL Control via Adversarial Training

    [https://arxiv.org/abs/2610.13608](https://arxiv.org/abs/2610.13608)

    本文提出一种基于对抗训练的语音强度转换方法，首次实现对连续分贝刻度上任意目标声压级的精确控制，使用WavLM特征时达到了目标强度变化的83%。

    

    语音强度转换是指将一段录音语音以不同的语音强度呈现出来，而不是简单地缩放其幅度，同时保留其内容和说话人身份。现有系统通过参考语音或少数离散类别来控制强度。与之不同，我们使用连续分贝刻度上任意设定的目标声压级（SPL）来控制转换过程。一个轻量级转换器以条件循环一致性生成对抗网络（GAN）的形式进行训练，无需平行数据，其转换冻结的语音编码器所提取的特征，再由预训练的声码器重新合成语音。我们比较了三种表示方式：log-mel频谱图、WavLM和SpeechTokenizer，并提出了一种客观评估协议，在校准语料库上测量实际达到的强度变化与目标强度变化的匹配程度。使用WavLM时，转换器实现了目标强度变化的83%，并且还能同时改变基频。感知评估……

    arXiv:2610.13608v1 Announce Type: cross  Abstract: Vocal intensity conversion renders a recorded utterance at a different vocal intensity, rather than simply scaling its amplitude, while preserving its content and speaker identity. Existing systems control intensity through a reference utterance or a few discrete categories. Instead, we control the conversion with any target sound pressure level (SPL) on a continuous dB scale. A lightweight converter, trained as a conditional cycle-consistent GAN without parallel data, transforms the features of a frozen speech encoder, from which a pretrained vocoder resynthesizes the speech. We compare three representations, log-mel spectrograms, WavLM, and SpeechTokenizer, and introduce an objective protocol that measures the achieved intensity change against the target one on a calibrated corpus. With WavLM, the converter achieves a measured 83% of the target change in intensity and also shifts the fundamental frequency. A perceptual evaluation con
    
[^66]: CorrFlow：相关椭圆面上的精确且置换等变的流匹配

    CorrFlow: Exact and Permutation-Equivariant Flow Matching on the Correlation Elliptope

    [https://arxiv.org/abs/2610.13606](https://arxiv.org/abs/2610.13606)

    CorrFlow通过广义Fisher变换将满秩相关矩阵双射映射到无约束坐标，再用置换等变的图流匹配网络学习其经验分布，从而无需投影即可精确保证输出位于相关椭圆面上，并对变量排序保持等变性。

    

    相关矩阵同时结合了非线性正定性约束与行列同时置换的对称性，因此其生成涉及两个截然不同的问题：第一，无论变量如何排序，每个输出都必须位于相关椭圆面上；第二，生成器必须学习经验数据在该有效空间内所占据的特定领域分布。现有的金融图像生成器将这两个问题混在一起：它们学习有序的矩阵元素，然后再事后修复无效的采样结果。我们提出了CorrFlow，将这两个问题分离开来。一个广义Fisher变换将满秩相关矩阵双射地映射到无约束坐标，从而无需投影即可保证输出的有效性。一个图流匹配网络在这些坐标中学习经验分布，并且具有置换等变性：对变量重新排序只会使输出相应地重新排序。我们比较了重新实现的条件CorrGAN……

    arXiv:2610.13606v1 Announce Type: cross  Abstract: Correlation matrices combine nonlinear positive-definiteness constraints with a simultaneous row-column permutation symmetry. Their generation therefore involves two distinct problems. First, every output must remain on the correlation elliptope, irrespective of how its variables are ordered. Second, the generator must learn the domain-specific distribution occupied by empirical data inside that valid space. Existing financial image generators mix these problems: they learn ordered matrix entries and repair invalid draws afterward. We introduce CorrFlow, which separates them. A generalized Fisher transform maps full-rank correlation matrices bijectively to unconstrained coordinates, guaranteeing valid outputs without projection. A graph flow-matching network learns the empirical distribution in those coordinates and is permutation equivariant: reordering the variables only reorders the output. We compare a conditional CorrGAN reimpleme
    
[^67]: 一个小型认知核心可大规模解释人类行为

    A small cognitive core explains human behavior at scale

    [https://arxiv.org/abs/2610.13603](https://arxiv.org/abs/2610.13603)

    该研究将多个经典认知原理整合为一个统一的计算模型CORE，其在超过700万个人类选择数据上拟合后不仅优于黑箱模型，还能再现75%的已报告行为效应，表明一个小型认知核心即可解释大部分人类行为。

    

    心理学已经提出了许多认知原理。然而，每条原理都是在孤立状态下发展的，它们组合起来的解释力尚不清楚。我们将一小部分已确立的原理整合为一个模型，称之为情境检索与编码模型（COntextual Retrieval and Encoding，CORE）。CORE由一个情景记忆的层级结构组成，预测误差被写入其中，其编码、检索和遗忘均依赖于情境。该模型对来自137个实验的7,036,442个选择进行联合拟合后，对人类选择的预测优于黑箱模型，并且在模拟时能够再现75%的已报告行为效应。此外，其潜在变量呈现出贝叶斯学习和事件分割的经典特征。综合来看，这些发现表明一个小型认知核心即可解释人类行为的大部分，同时也通过在全部实验数据上构建计算模型，为使心理学超越各部分之和开辟了一条道路。

    arXiv:2610.13603v1 Announce Type: new  Abstract: Psychology has proposed many principles of cognition. Each, however, was developed in isolation, leaving their combined explanatory power unknown. We assembled a small set of established principles into one model, which we call COntextual Retrieval and Encoding (CORE). CORE consists of a hierarchy of episodic memories into which prediction errors are written and whose encoding, retrieval, and forgetting depend on context. Fitted jointly to 7,036,442 choices from 137 experiments, the model predicts human choices better than black-box models and reproduces 75% of reported behavioral effects when simulated. Moreover, its latent variables show classic signatures of Bayesian learning and event segmentation. Taken together, these findings suggest that a small cognitive core accounts for much of human behavior. They also open a path toward making psychology more than the sum of its parts, by building computational models on the entire experimen
    
[^68]: EchoTab：通过表格重建学习以测量为基础的超声心动图诊断表示

    EchoTab: Learning Measurement-Grounded Representations for Echocardiographic Diagnosis through Tabular Reconstruction

    [https://arxiv.org/abs/2610.13599](https://arxiv.org/abs/2610.13599)

    EchoTab提出了一种以测量为基础的表示学习框架，将测量特定的token作为超声视角与诊断之间的中间表示，并通过表格重建任务学习测量基础表示，从而提升多视角超声心动图诊断的准确性。

    

    在超声心动图等多视角医学影像检查中，诊断信息分布于各个视角之中，每个视角捕捉心脏结构与功能的不同方面。在临床实践中，这些视角首先通过定量测量进行解读，进而支持具体诊断。然而，现有的超声心动图表示学习方法并未将测量显式建模为视角与诊断之间的中间表示。因此，单项测量无法以下游诊断任务可有选择性地利用的形式得到显式表示，从而阻碍了准确诊断。受这一分层临床工作流程的启发，我们提出了EchoTab，一个以测量为基础的表示学习框架，它使用测量特定的token（标记）作为超声视角与诊断之间的中间表示，来表征多视角超声心动图检查。每个测量token可选择性地……（原文摘要至此截断）

    arXiv:2610.13599v1 Announce Type: new  Abstract: Diagnostic information in multi-view medical imaging studies, like echocardiography, is distributed across views, each capturing different aspects of cardiac structure and function. In clinical practice, these views are first interpreted through quantitative measurements, which then support specific diagnoses. However, existing echo representation learning methods do not explicitly model measurements as an intermediate representation between views and diagnosis. As a result, individual measurements are not explicitly represented in a form that downstream diagnostic tasks can selectively use, hindering accurate diagnosis.   Inspired by this hierarchical clinical workflow, we introduce EchoTab, a measurement-grounded representation learning framework that represents multi-view echocardiography studies using measurement-specific tokens as an intermediate representation between echo views and diagnosis. Each measurement token selectively att
    
[^69]: 强凸集合上随机化线性预言机优化的下界

    Lower Bounds for Randomized Linear-Oracle Optimization over Strongly Convex Sets

    [https://arxiv.org/abs/2610.13590](https://arxiv.org/abs/2610.13590)

    本文针对强凸集合上基于线性极小化预言机的优化问题，分析了现有 $1/T^2$ 最坏情况误差下界的脆弱性，并证明即使引入随机化，此类方法在 T 次 LMO 调用后仍无法突破 $1/T^2$ 的误差下界。

    

    近期的研究结果 [arXiv:2602.04378, arXiv:2602.22608] 建立了在 T 次线性极小化预言机（LMO）调用之后阶为 $1/T^2$ 的最坏情况目标误差下界，该下界在其对 T 的依赖关系上与上界 [arXiv:1406.1305] 相匹配，从而解决了一个长期悬而未决的开放性问题。然而，经过仔细审视，这两个下界都存在一定的脆弱性。第一个结果 [arXiv:2602.04378] 仅适用于标准的 Frank–Wolfe 算法，并且正如我们所证明的，其规定的缓慢精确线搜索轨迹可以将足够小的初始化误差指数级放大，相当于每次迭代大约需要额外一位初始化精度。第二个结果 [arXiv:2602.22608] 通过一个能够对抗性地适应算法查询的抵抗预言机，为任何基于 LMO 的确定性一阶方法获得了下界。在这两种情况下，可以设想随机化或许能够绕过这些下界；特别是……（摘要在此处截断）

    arXiv:2610.13590v1 Announce Type: cross  Abstract: Recent results [arXiv:2602.04378, arXiv:2602.22608] have established worst-case objective-error lower bounds of order $1/T^2$ after $T$ linear minimization oracle (LMO) calls, matching the upper bound [arXiv:1406.1305] in its dependence on $T$ and thereby resolving a long-standing open question. Upon close inspection, however, both lower bounds are somewhat brittle. The first [arXiv:2602.04378] applies only to the standard Frank--Wolfe algorithm, and, as we show, its prescribed slow exact-line-search trajectories can amplify sufficiently small initialization errors exponentially, corresponding to roughly one additional bit of initialization accuracy per iteration. The second [arXiv:2602.22608] obtains lower bounds for any deterministic LMO-based first-order method via a resisting oracle that adversarially adapts to the queries of the algorithm. In both cases it is conceivable that randomization may circumvent these lower bounds; in par
    
[^70]: 时变分层基础设施中电动汽车充电的概率预测

    Probabilistic Forecasting of Electric Vehicle Charging in a Time-varying Hierarchical Infrastructure

    [https://arxiv.org/abs/2610.13588](https://arxiv.org/abs/2610.13588)

    提出将分层预测与处理缺失数据的保形预测相结合，在持续演化的电动汽车充电基础设施分层结构中，为从单个充电点到国家级的每个聚合节点生成有效的预测区间。

    

    对电动汽车（EV）充电需求进行不确定性量化，对于将电动汽车普及给电力系统带来的日益增长的压力转化为灵活性来源至关重要。我们研究了苏格兰公共充电站的一个大规模数据集，其中需求呈现出分层结构，跨越多个聚合层级组织，从单个充电点到区域和国家级尺度。充电基础设施在不断演变，新站点不断部署，现有站点偶尔被移除，这就需要针对这种演变结构量身定制的预测和不确定性量化方法。为了应对这些挑战，我们将分层预测与处理缺失数据的保形预测最新进展相结合。所考虑的保形方法在层次结构的每个节点上都能产生有效的预测区间，其中CP-MNR-Nested算法成功地处理了时变的分层结构。

    arXiv:2610.13588v1 Announce Type: cross  Abstract: Uncertainty quantification for electric vehicle (EV) charging demand is essential to turn the increasing pressure of EV adoption on power systems into a source of flexibility. We study a large-scale dataset of public charging stations in Scotland, where demand exhibits a hierarchical structure, organized across multiple aggregation levels, from individual charging points to regional and national scales. The charging infrastructure is continuously evolving, with new stations deployed and existing ones occasionally removed, which calls for forecasting and uncertainty quantification methods tailored to this evolving structure. To address these challenges, we combine hierarchical forecasting with recent advances in conformal prediction with missing data. The conformal procedures considered produce valid prediction intervals at every node of the hierarchy and the algorithm CP-MNR-Nested successfully accounts for the time-varying hierarchica
    
[^71]: RARA：通过检索锚定表示对齐实现机器人操作的鲁棒策略微调

    RARA: Robust Policy Fine-Tuning through Retrieval-Anchored Representation Alignment for Robotic Manipulation

    [https://arxiv.org/abs/2610.13578](https://arxiv.org/abs/2610.13578)

    本文提出RARA，一种两阶段微调方法，通过检索预训练中任务意图和运动相似的表示特征并对齐目标演示的表示，在少量演示适应新任务的同时保护预训练表示不被扭曲，从而保持模型在分布外偏移下的鲁棒泛化能力。

    

    预训练的机器人策略（如视觉-语言-动作（VLA）模型）为通用机器人操作提供了坚实的基础，但仅凭少量演示将它们适应到新任务仍然具有挑战性。直接微调预训练策略通常能在分布内取得出色的性能，但这可能会扭曲预训练期间学到的表示，并削弱模型在分布外偏移下的泛化能力。为了解决这个问题，我们提出了检索锚定表示对齐，这是一种两阶段适应方法。首先，RARA从预训练中检索具有相似任务意图和运动的表示特征，并将策略对目标演示的表示与检索到的演示的预训练表示进行对齐。其次，RARA在目标演示上对策略进行微调，同时限制其对检索到的演示的表示发生偏移的程度……

    arXiv:2610.13578v1 Announce Type: cross  Abstract: Pre-trained robot policies, such as Vision-Language-Action (VLA) models, provide a strong foundation for generalist robotic manipulation, but adapting them to new tasks with only a few demonstrations remains challenging. Directly fine-tuning a pre-trained policy often achieves strong in-distribution performance, yet it can distort the representations learned during pre-training and weaken the model's generalization ability under out-of-distribution shifts. We propose Retrieval-Anchored Representation Alignment (RARA), a two-stage adaptation method, to address this issue. First, RARA retrieves representation features from pretraining with similar task intent and motion, and aligns the policy's representations of the target demonstrations with the pre-trained representations of the retrieved demonstrations. Second, RARA fine-tunes the policy on the target demonstrations while limiting how much its representations of the retrieved demonst
    
[^72]: 基于蒙特卡洛树搜索的高维金融回归 $L_0$ 特征选择

    $L_0$ Feature Selection via Monte Carlo Tree Search in High-Dimensional Financial Regression

    [https://arxiv.org/abs/2610.13574](https://arxiv.org/abs/2610.13574)

    本文提出一种基于广义嵌套 Rollout 策略自适应蒙特卡洛树搜索（GNRPA）的 $L_0$ 特征选择框架，将特征选择建模为以样本外验证性能为评分准则的序贯决策过程，从而克服 Lasso 收缩偏差与精确稀疏回归计算困难的问题。

    

    特征选择是高维金融回归中的核心问题，大量工程化变量以及预测变量之间的强相关性会降低模型的稳定性和可解释性。$L_1$ 正则化（Lasso）被广泛用于稀疏回归，但当有效信号分散在相互关联的变量之间时，其收缩效应可能具有局限性。精确稀疏回归可以避免这种收缩，但需要求解一个计算上非常困难的组合优化问题，且通常以最小化样本内误差而非样本外误差为目标进行拟合。本文提出了一种基于带广义嵌套 Rollout 策略自适应的蒙特卡洛树搜索的特征选择框架。特征选择被表述为一个序贯决策过程，候选特征子集依据其样本外验证性能而非训练误差进行评分。该方法还支持字典序……

    arXiv:2610.13574v1 Announce Type: new  Abstract: Feature selection is a central problem in high-dimensional financial regression, where large numbers of engineered variables and strong correlations among predictors reduce model stability and interpretability. $L_1$ regularization (Lasso) is widely used for sparse regression, but its shrinkage can be restrictive when informative signals are spread across correlated variables. Exact sparse regression avoids this shrinkage but requires solving a combinatorial optimization problem that is computationally difficult and is typically fit to minimize in-sample rather than out-of-sample error. This paper presents a feature selection framework based on Monte Carlo Tree Search with Generalized Nested Rollout Policy Adaptation (GNRPA). Feature selection is formulated as a sequential decision process, and candidate subsets are scored by their out-of-sample validation performance rather than training error. The method also supports a lexicographic o
    
[^73]: TCP确认问题的同时竞争比与遗憾保证

    Simultaneous Competitive and Regret Guarantees for TCP Acknowledgment

    [https://arxiv.org/abs/2610.13571](https://arxiv.org/abs/2610.13571)

    该论文为TCP确认问题提出了一种随机化在线算法，在无需预先知道报文数量或离线最优成本的情况下，同时实现了(2+ε)的竞争比保证和相对于最优专家的次线性期望遗憾。

    

    我们研究了TCP确认问题的同时竞争比与遗憾保证。每次确认会服务所有待处理的报文，目标是最小化总确认成本与等待成本。众所周知，没有任何确定性在线算法能够实现优于2的竞争比。给定k个随机化在线专家和0<ε<1，我们构建了一个随机化在线算法，其实际总成本始终至多为离线最优成本OPT的(2+ε)倍，且相对于实际成本最小的专家，其期望遗憾为O(√(OPT·log(k+1)) + log(1/ε))。该算法无需预先知道报文数量或OPT。我们的算法控制专家之间的切换成本，并利用允许的竞争界限与当前成本之间的差距来决定何时进行学习。我们还证明了下界，表明当k≥……（原文摘要在此处截断）

    arXiv:2610.13571v1 Announce Type: new  Abstract: We study simultaneous competitive and regret guarantees for TCP acknowledgment. Each acknowledgment serves all pending packets, and the objective is to minimize the total acknowledgment and waiting costs. It is well known that no deterministic online algorithm achieves a competitive ratio better than $2$. Given $k$ randomized online experts and $0<\varepsilon<1$, we construct a randomized online algorithm whose realized total cost is always at most $(2+\varepsilon)$ times the offline optimal cost $\mathrm{OPT}$ and whose expected regret against the expert with the smallest realized cost is $O(\sqrt{\mathrm{OPT}\log(k+1)}+\log(1/\varepsilon))$. The algorithm requires no advance knowledge of the number of packets or $\mathrm{OPT}$. Our algorithm controls switching costs between experts and uses the gap between the permitted competitive bound and its current cost to determine when to learn. We also prove lower bounds showing that, for $k\ge
    
[^74]: 有理线性化多层感知机

    Rationally Linearized MLPs

    [https://arxiv.org/abs/2610.13568](https://arxiv.org/abs/2610.13568)

    RaLiMPs利用低次有理近似替代标准激活函数来高效线性化MLP，在从隐式3D模型到大语言模型的多种应用中，以更低的推理成本实现了具有竞争力甚至更优的精度。

    

    我们提出了有理线性化多层感知机，这是一种利用标准激活函数的低次有理近似来高效线性化MLP的新方法。受基于多项式的MLP线性化方法启发，我们证明有理近似能够以显著更低的次数提供精确的表示。RaLiMPs利用这些近似构建具有窄瓶颈的核表示，从而实现跨MLP层的高效组合。我们为所得近似提供了理论保证，并在从隐式3D模型到大语言模型等多种应用上对RaLiMPs进行了评估。实验表明，RaLiMPs具有良好的精度-效率权衡，在大幅降低推理成本的同时达到了具有竞争力或更优的精度。

    arXiv:2610.13568v1 Announce Type: new  Abstract: We present Rationally Linearized Multilayer Perceptrons(RaLiMPs), a new approach to efficiently linearizing MLPs using low-degree rational approximations of standard activation functions. Inspired by polynomial-based MLP linearization, we show that rational approximations can provide accurate representations at substantially lower degree. RaLiMPs exploit these approximations to construct kernel representations with narrow bottlenecks, enabling efficient composition across MLP layers. We provide theoretical guarantees for the resulting approximation and evaluate RaLiMPs across applications ranging from implicit 3D models to large language models. Our experiments demonstrate favorable accuracy--efficiency tradeoffs, with RaLiMPs achieving competitive or improved accuracy while substantially reducing inference cost.
    
[^75]: 基于AIS数据使用Transformer模型的概率船舶轨迹预测

    Probabilistic Ship Trajectory Forecasting using Transformer Models on AIS Data

    [https://arxiv.org/abs/2610.13557](https://arxiv.org/abs/2610.13557)

    该论文利用AIS数据、基于核密度估计的热图编码以及分位数损失训练的Transformer和Performer模型，实现了受限水域中船舶轨迹的概率预测，并给出模型感知的预测区间。

    

    预测船舶轨迹是海上航行中的一项重要任务，这既适用于人类航海员，也适用于自主水面船舶。轨迹预测能够使诸如避碰等子程序及时评估局势并做出相应反应。本文研究了用于受限水域中水面船舶轨迹预测的Transformer和Performer模型。利用广泛可得的自动识别系统（AIS）位置数据，对候选模型进行训练与评估。每个智能体周围局部区域内其他船舶的历史运动信息通过位置热图进行传递，热图采用核密度估计（KDE）生成，并在输入Transformer模型之前使用卷积神经网络（CNN）进行编码。模型使用分位数损失进行训练，以高效获得模型感知的预测区间，因为这对时间……（原文摘要在此处截断）

    arXiv:2610.13557v1 Announce Type: new  Abstract: Anticipation of vessel trajectories is an essential task when navigating at sea. This applies to both human navigators and autonomous surface vessels. Trajectory forecasting enables sub-routines such as collision avoidance to assess the situation and react accordingly in time. This paper investigates Transformer and Performer models for trajectory prediction of surface vessels in confined waters. Using widely available position data from the Automatic Identification System (AIS), candidate models are trained and evaluated. Historical information on how other vessels have moved in the local area surrounding each agent is conveyed using positional heat maps. The heat maps are generated using Kernel Density Estimation (KDE) and encoded with a Convolutional Neural Network (CNN) before entering the Transformer models. The models are trained with quantile loss to efficiently obtain a model-aware prediction interval, as this is crucial for time
    
[^76]: 防御蒸馏的黑魔法：阈下学习的无监督发现

    Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning

    [https://arxiv.org/abs/2610.13536](https://arxiv.org/abs/2610.13536)

    提出了一种无需预先知晓目标特征的无监督防御方法，仅通过分析学生模型的参数更新即可发现、解释并抑制从共享基础参数的教师模型阈下继承的失对齐行为特征。

    

    当语言模型从共享其基础参数的教师模型蒸馏而来时，学生模型可能会继承教师模型的行为特征——包括失对齐——即使训练数据在语义上与该特征无关。现有的所有防御方法都需要预先知道该特征。我们移除了这一要求，并证明通过利用一个关键的不对称性——阈下传递的特征在教师模型生成的多样化数据集上保持一致表达，而任务特定能力和共享能力则各不相同——可以仅凭参数更新，在无监督条件下恢复该特征。由此得到的估计器会返回一个权重空间方向，放大该方向即可发现并解释该特征，或将其投影出去以抑制该特征。在阈下学习基准测试中，我们的方法在没有任何特征标签的情况下成功恢复了传递的动物偏好特征，并抑制了其表达。

    arXiv:2610.13536v1 Announce Type: cross  Abstract: When a language model is distilled from a teacher that shares its base parameters, the student can inherit behavioural traits of the teacher---including misalignment---even from data that is semantically unrelated to the trait (Cloud et al., 2026). All existing defences require knowing the trait. We remove this requirement and prove that a subliminally transferred trait can be recovered from the parameter updates alone, without supervision, by exploiting a single asymmetry: the trait is expressed consistently across diverse datasets generated by the teacher, whereas task-specific and shared capabilities vary. The resulting estimator returns a weight-space direction that can be amplified to discover and interpret the trait or projected out to suppress it. Applied to the subliminal-learning benchmarks of (Cloud et al., 2026), our method recovers the transmitted animal-preference traits without any trait labels and suppresses their expres
    
[^77]: S-Space：探索多模态语言模型中的空间工作空间

    S-Space: Exploring Spatial Workspace in Multimodal Language Models

    [https://arxiv.org/abs/2610.13527](https://arxiv.org/abs/2610.13527)

    该论文发现多模态语言模型内部存在一个固有的线性子空间"S-Space"，作为模型的内部空间工作空间，以连续坐标编码物体位置，并支撑感知编码、视角变换和言语输出等空间推理的全过程。

    

    多模态语言模型（MLLMs）展现出有前景的空间推理能力，然而支持这些能力的内部机制仍然知之甚少。我们发现了S-Space——一个线性表示子空间，其中物体标记的表示沿着水平、垂直和距离三个轴编码连续坐标。尽管我们通过对物体标记激活求空间答案得分的梯度来识别这些轴，但S-Space是模型内在的，而非由特定数据或提示诱导产生，在非空间问题下依然保持可读性。我们证明S-Space充当着多模态语言模型的内部空间工作空间：感知过程将物体位置编码到其中，形成可直接读取以支持准确空间判断的空间表示；内部推理过程对其进行变换以实现视角转换；言语响应则选择性地访问它——因果干预实验证实了这一点，干预会改变空间判断，同时（在很大程度上不影响其他能力）。

    arXiv:2610.13527v1 Announce Type: cross  Abstract: Multimodal language models (MLLMs) exhibit promising spatial reasoning capabilities, yet the internal mechanisms supporting them remain poorly understood. We uncover S-Space, a linear representation subspace in which object-token representations encode continuous coordinates along horizontal, vertical, and distance axes. Although we identify these axes by differentiating spatial-answer scores with respect to object-token activations, S-Space is intrinsic rather than induced by specific data or prompts, remaining readable under non-spatial questions. We show that S-Space functions as an internal spatial workspace of MLLMs. Perception encodes object locations into it, forming spatial representations that can be directly read out to support accurate spatial judgments. Internal reasoning transforms it for changing viewpoints. Verbal responses selectively access it, evidenced by causal interventions that alter spatial judgments while largel
    
[^78]: 基于逐层随机特征模型模拟深度神经网络的训练尖峰现象

    Simulating Training Spikes of Deep Neural Networks by Layer-wise Random Feature Models

    [https://arxiv.org/abs/2610.13524](https://arxiv.org/abs/2610.13524)

    本文提出用逐层随机特征二次模型来模拟深度神经网络训练中的损失尖峰现象，成功预测了Adam优化器 $\beta_2$ 接近1时Transformer语言模型的训练尖峰，并据此推导出能提升性能的 $\beta_2$ 调度策略。

    

    在大语言模型（LLM）预训练过程中，损失函数间歇性出现的“尖峰”现象在理论和实证层面都仍然是一个谜。本文证明，具有重尾幂律分布输入的简单二次模型能够复现这些尖峰。我们引入了一个理论框架和模拟算法，利用此类二次模型来刻画深度神经网络（DNN）的逐层训练动态。在一个关于Adam超参数配置的案例研究中（其中 $\beta_2$ 被设置为接近 $1$，已知该配置会在Transformer语言模型中诱发尖峰），我们的模拟成功地从给定检查点预测出指示尖峰行为的凸起。此外，我们从观测到的指标中推导出合适的 $\beta_2$ 调度策略，实证结果表明该策略能提升模型性能。

    arXiv:2610.13524v1 Announce Type: new  Abstract: Intermittent ``spikes'' in loss, during the pre-training of Large Language Models (LLMs), remains a mysterious phenomenon both theoretically and empirically. In this paper, we show that simple quadratic models with heavy-tailed, power-law distributed inputs can replicate these spikes. We introduce a theoretical framework and simulation algorithm to model per-layer training dynamics of Deep Neural Networks (DNNs), using such quadratic models. In a case study on an Adam hyperparameter configuration (where $\beta_2$ is set close to $1$) known to induce spikes in Transformer LMs, our simulation successfully predicts bumps indicating spiky behavior from a given checkpoint. Furthermore, we derive an appropriate $\beta_2$ schedule from observed metrics, which empirically improves performance.
    
[^79]: Transformer中离散时间自注意力动力学的渐近分析

    Asymptotic Analysis of Discrete-Time Self-Attention Dynamics in Transformers

    [https://arxiv.org/abs/2610.13518](https://arxiv.org/abs/2610.13518)

    本文首次对Transformer的离散时间自注意力动力学进行了系统研究，给出了单头与多头情形下平衡点的完整分类和稳定性分析，并证明其主要吸引子是与值矩阵主特征向量对齐的共识平衡点。

    

    在机器学习中，自注意力动力学描述了Transformer的注意力机制，它通过将Transformer的层索引视为时间索引而得到。尽管该动力系统本质上是离散时间的，但以往通常在连续时间框架下对其进行研究。本文旨在对离散时间版本的自注意力动力学进行全面深入的研究：在单头和多头两种情形下，给出了其平衡点的完整分类，并研究了这些平衡点的稳定性。我们证明，在这两种情形下，动力学的主要吸引子是与值矩阵主特征向量对齐的共识平衡点（单头情形下对应值矩阵，多头情形下对应权重矩阵）；同时，至少在单头情形下，还可以证明存在其他局部渐近稳定的二部平衡点。通过观察学习到的……（该行为得到了验证）

    arXiv:2610.13518v1 Announce Type: cross  Abstract: In machine learning, a self-attention dynamics describes the attention mechanism of transformers, and is obtained treating the layer index of a transformer as time index. Even though naturally discrete-time, this dynamical system is most often studied in continuous-time. Scope of this paper is to provide a thorough investigation of the discrete-time version of the self-attention dynamics: a complete classification of its equilibria is given, and their stability properties are studied, in both the single-head and the multihead case. We show that in both cases the main attractor of the dynamics is a consensus equilibrium aligned with the principal eigenvector of the value matrix (in the single-head case, a corresponding matrix of weights in the multihead case), even though other locally asymptotically stable bipartite equilibria can be show to exist at least in the single-head situation. The behavior is confirmed by looking at the learne
    
[^80]: FedPCD：优先级约束的联邦聚合

    FedPCD: Priority-Constrained Federated Aggregation

    [https://arxiv.org/abs/2610.13508](https://arxiv.org/abs/2610.13508)

    FedPCD提出了一种优先级约束的联邦学习优化框架，在保持传统联邦聚合为主要目标的同时，仅施加满足客户端性能约束所需的最小校正，将全局优化与客户端公平保护相分离，在异构数据场景下显著优于现有最先进方法。

    

    联邦学习（FL）旨在从分散的客户端数据中优化共享的全局模型。虽然该方法对平均性能有效，但可能在客户端之间产生不公平的性能差异，尤其是在统计异构的数据分布下。现有的公平性感知和多目标联邦学习方法通过客户端重新加权、梯度校正或共同下降公式来解决这一问题，但往往会改变全局优化方向或对称地对待各客户端目标。我们提出了FedPCD，这是一种优先级约束的联邦优化框架，将全局优化目标与客户端保护分离开来。FedPCD保持传统的联邦聚合作为主要目标，同时仅允许满足客户端性能约束所需的最小校正。在异构联邦学习基准测试中，FedPCD始终优于最先进的基线方法，实现了最高的全局性能。

    arXiv:2610.13508v1 Announce Type: new  Abstract: Federated learning (FL) seeks to optimize a shared global model from decentralized client data. While effective for average performance, this process can produce unfair performance across clients, particularly under statistically heterogeneous data distributions. Existing fairness-aware and multi-objective FL methods address this through client re-weighting, gradient correction, or common-descent formulations, but often change the global optimization direction or treat client objectives symmetrically. We propose FedPCD, a priority-constrained federated optimization framework that separates the global optimization objective from client protection. FedPCD preserves the conventional federated aggregate as the primary objective, while allowing only the minimum correction necessary to meet client performance constraints. Across heterogeneous FL benchmarks, FedPCD consistently outperforms state-of-the-art baselines, achieving the highest globa
    
[^81]: 超越时间邻近性：面向强化学习中信用分配的解释性世界模型

    Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning

    [https://arxiv.org/abs/2610.13504](https://arxiv.org/abs/2610.13504)

    该论文提出解释性世界模型（EWM）框架，用语言模型生成的因果解释取代传统的时间邻近性来进行强化学习中的信用分配，通过保守分层算子递归重分配奖励并保持回合总回报不变，从而提升生成式推荐系统中长时程决策的样本效率。

    

    arXiv:2610.13504v1 公告类型：交叉 摘要：生成式推荐系统越来越多地采用强化学习（RL）来优化那些可能在决策发生数百甚至数千次交互之后才显现的结果。标准的强化学习依据时间邻近性而非显式相关性来传播奖励信号，这削弱了分配给那些距离遥远但至关重要的动作的信号，并成为样本效率低下的一个根源。我们提出了解释性世界模型，这是一个归因框架，利用解释来识别可能对最终已实现结果有贡献的早期事件。我们用一个前沿语言模型来实例化EWM，并通过一个保守的分层算子将其生成的解释用作学习信号，该算子在保持未折扣回合回报的前提下递归地重新分配奖励。为了探究什么样的解释才是有用的，我们受Halpern的实际因果理论、Lombrozo的解释简洁性理论以及De……（原文摘要在此处截断）

    arXiv:2610.13504v1 Announce Type: cross  Abstract: Generative Recommenders increasingly use Reinforcement Learning (RL) to optimize outcomes that may appear hundreds or thousands of interactions after a decision. Standard RL propagates reward according to temporal proximity rather than explicit relevance, attenuating the signal assigned to distant but important actions and creating a source of sample inefficiency. We introduce Explanatory World Models (EWM), an attribution framework that uses explanations to identify the earlier events that plausibly contributed to a realized outcome. We instantiate EWM with a frontier language model and use its explanations as learning signals using a conservative hierarchical operator that recursively redistributes reward while preserving the undiscounted episode return. To investigate what constitutes a useful explanation, we operationalize three complementary variants inspired by Halpern's actual causation, Lombrozo's explanatory simplicity, and De
    
[^82]: 基于时间分解的非线性模型预测控制可扩展学习迭代求解器

    A Scalable Learned Iterative Solver for Nonlinear MPC via Temporal Decomposition

    [https://arxiv.org/abs/2610.13480](https://arxiv.org/abs/2610.13480)

    本文通过在LISCO求解器架构中利用问题的时间分解结构，使神经网络规模与预测时域无关，从而实现了对大型非线性系统可扩展的实时非线性模型预测控制。

    

    非线性模型预测控制（NMPC）可以应用于具有复杂目标的大型非线性系统，同时施加约束条件，但求解底层优化问题的计算成本限制了其在快动态或高维状态系统中的实时应用。神经网络可以通过逼近控制律来降低在线计算成本，但可能违反约束，且通常需要大量预先计算的优化器解用于训练。基于学习的约束优化迭代求解器（LISCO）则通过自监督训练学习预测原始-对偶步以满足Karush-Kuhn-Tucker（KKT）条件，无需预先计算的解，但其密集的整体式网络架构会随预测时域的增长而扩大。我们通过在求解器架构中利用问题的时间结构消除了这种对时域的依赖，使网络规模与时域无关，同时……（原文在此处截断）

    arXiv:2610.13480v1 Announce Type: cross  Abstract: Nonlinear model predictive control (NMPC) can be applied to large nonlinear systems with complex objectives while enforcing constraints, but the cost of solving the underlying optimization problems limits real-time application for systems with fast dynamics or many states. Neural networks can reduce this online cost by approximating the control law, but may violate constraints and typically require many precomputed optimizer solutions for training. The learning-based iterative solver for constrained optimization (LISCO) instead learns to predict primal-dual steps toward satisfaction of the Karush-Kuhn-Tucker (KKT) conditions through self-supervised training without precomputed solutions, but its dense, monolithic architecture grows with the prediction horizon. We remove this horizon dependence by exploiting the temporal structure of the problem in the solver architecture, which makes the network size independent of the horizon while le
    
[^83]: 量子输入空间上Sobolev函数的调和分析与量子电路逼近

    Harmonic analysis and circuit approximation of Sobolev functions on quantum input spaces

    [https://arxiv.org/abs/2610.13477](https://arxiv.org/abs/2610.13477)

    该论文通过紧群上的调和分析刻画了量子态与量子幺正操作所在空间上Sobolev函数的Peter–Weyl分解，并结合谱截断与基于Solovay–Kitaev定理的有限深度量子电路逼近，建立了将谱截断误差与电路逼近误差分离的定量逼近速率。

    

    我们推导了定义在表示量子态的复单位球面与复射影空间上、以及表示量子幺正操作的特殊幺正群与射影幺正群上的Sobolev函数的逼近速率。通过利用紧群与紧齐性空间上的调和分析，我们借助不可约表示和Laplace–Beltrami算子刻画了相应的Peter–Weyl分解与Sobolev空间。结合谱截断与有限深度量子电路逼近（并援引Solovay–Kitaev定理），我们建立了能够将谱截断误差与电路逼近误差分离的定量逼近速率。

    arXiv:2610.13477v1 Announce Type: cross  Abstract: We derive approximation rates for Sobolev functions defined on the complex unit sphere and complex projective space, representing quantum states, and on the special and projective unitary groups, representing quantum unitary operations. By using harmonic analysis on compact groups and compact homogeneous spaces, we characterize the corresponding Peter--Weyl decompositions and Sobolev spaces via irreducible representations and the Laplace--Beltrami operator. Combining spectral truncation with finite-depth quantum-circuit approximation, invoking the Solovay--Kitaev theorem, we establish quantitative rates that separate the spectral truncation error from the circuit approximation error.
    
[^84]: SqueezeBench：抗饱和的数学能力评估

    SqueezeBench: Saturation-Resilient Evaluation of Mathematical Capability

    [https://arxiv.org/abs/2610.13461](https://arxiv.org/abs/2610.13461)

    SqueezeBench提出了一种抗饱和的数学能力评估基准，要求AI不断提出并用Lean机器验证证明整数函数的上下界记录，使每次成功都提高门槛而非消除问题。

    

    人工智能系统在数学领域正取得快速进展，但这种进展难以衡量。针对已解决问题的基准会饱和，而在开放问题的基准中，每个问题只能被首次解决一次。我们提出了SqueezeBench，这是一个对饱和具有韧性的基准：一次成功会提高门槛，而不是移除问题。每个问题固定一个整数函数 $f(n)$，其取值仅在较小的 $n$ 时已知，该基准为它维护一个下界记录和一个上界记录。记录是一个可计算的Lean函数，并附带机器验证的证明表明它约束着 $f$。一次提交需要提出一个新函数，并证明它约束着 $f$、在任何 $n$ 处都不差于当前记录、且在某个 $n$ 处严格更优。一旦被接受，它就成为下一次提交必须超越的记录。因此，该系统既选择了界，也选择了其证明，并且可以在单个 $n$ 处改进记录，或改进其增长率……

    arXiv:2610.13461v1 Announce Type: new  Abstract: AI systems are making rapid progress in mathematics, but this progress is hard to measure. Benchmarks of solved problems saturate, and in benchmarks of open problems each problem can be resolved for the first time only once. We present SqueezeBench, a benchmark that is resilient to saturation: a success raises the bar instead of removing the problem. Each problem fixes an integer function $f(n)$ whose values are known only for small $n$, and the benchmark keeps a lower and an upper record on it. A record is a computable Lean function with a machine-checked proof that it bounds $f$. A submission proposes a new function and proves that it bounds $f$, that it is no worse than the current record at any $n$, and that it is strictly better at some $n$. Once accepted, it becomes the record that the next submission has to beat. The system thus chooses the bound as well as its proof, and it may improve the record at a single $n$ or its growth rat
    
[^85]: 通过压缩学习：风险估计达到大数定律速率

    Learning by Compression: Risk Estimates Attain the Law-of-Large-Numbers Rate

    [https://arxiv.org/abs/2610.13442](https://arxiv.org/abs/2610.13442)

    本文证明基于优先压缩的学习算法的风险界中并非固有存在对数因子，其风险估计对所有复杂度取值一致地达到 $\boldsymbol{k}/N + O(1/\sqrt{N})$ 的大数定律速率，与经验估计速率相同。

    

    压缩方案为推导学习算法的数据相关风险界提供了一个通用框架。对于基于优先压缩的学习规则，现有结果以一个样本相关的复杂度 $\boldsymbol{k}$ 来上界风险，该复杂度定义为压缩的基数再加上重构假设不适用的训练样本数量。然而，这些结果给出的裕量为 $\boldsymbol{k}/N$ 之上 $O(\sqrt{\ln N/N})$ 的量级。本文证明该对数因子并非内在固有。我们证明，对于每个固定的置信水平 $1-\delta$，风险以至少 $1-\delta$ 的概率被上界为 $$ \frac{\boldsymbol{k}}{N} + O\left(\frac{1}{\sqrt{N}}\right), $$ 且该界对所有可能的 $\boldsymbol{k}$ 值一致成立。因此，基于优先压缩的学习算法的风险估计达到了与经验估计相同的大数定律速率。

    arXiv:2610.13442v1 Announce Type: cross  Abstract: Compression schemes provide a general framework for deriving data-dependent risk bounds for learning algorithms. For learning rules based on preferent compression, existing results upper-bound the risk in terms of a sample-dependent complexity $\boldsymbol{k}$, defined as the cardinality of the compression augmented by the training examples on which the reconstructed hypothesis is inappropriate. These results, however, yield a margin of order $O(\sqrt{\ln N/N})$ above $\boldsymbol{k}/N$. In this paper, we establish that the logarithmic factor is not intrinsic. We prove that, for every fixed confidence level $1-\delta$, the risk is bounded with probability at least $1-\delta$ by $$ \frac{\boldsymbol{k}}{N} + O\left(\frac{1}{\sqrt{N}}\right), $$ uniformly over all possible values of $\boldsymbol{k}$. Thus, risk estimates for learning algorithms based on preferent compression attain the same law-of-large-numbers rate as the empirical esti
    
[^86]: 基于DistilHuBERT重访口音分类：说话人分离评估与跨语料库泛化

    Revisiting Accent Classification with DistilHuBERT: Speaker-Disjoint Evaluation and Cross-Corpus Generalization

    [https://arxiv.org/abs/2610.13440](https://arxiv.org/abs/2610.13440)

    该研究发现文件级评估会大幅高估基于DistilHuBERT的口音分类模型性能，说话人分离评估才能更真实地反映模型对未见说话人的泛化能力，且仅用外部数据训练的模型难以可靠泛化到L2-ARCTIC语料库。

    

    最近的口音分类系统在标准的语料库内评估中可以达到很高的准确率。然而，当训练集和测试集之间的说话人存在重叠时，与真实部署场景相比，这样的结果可能高估了模型的鲁棒性。本研究使用L2-ARCTIC和多个外部英语口音数据集，重新审视了基于DistilHuBERT的口音分类的鲁棒性。比较涵盖了文件级和说话人分离的数据划分，以及平衡的外部数据训练和师生伪标签方法。语料库内评估和外部评估之间的巨大性能差距表明，文件级评估大大高估了性能，而说话人分离评估能为模型对未见说话人的泛化提供更贴近现实的估计。在本研究中，仅用外部数据训练的模型无法可靠地泛化到L2-ARCTIC。在所测试的配置中，平衡的外部……

    arXiv:2610.13440v1 Announce Type: cross  Abstract: Recent accent classification systems can achieve high accuracy in standard in-corpus evaluations. However, when speakers overlap between the training and test sets, such results may overestimate model robustness compared with real-world deployment scenarios. This study revisits the robustness of DistilHuBERT-based accent classification by using L2-ARCTIC and multiple external English accent datasets. The comparison covers file-level and speaker-disjoint splits, along with balanced external-data training and teacher-student pseudo-labeling. The large performance gap between in-corpus and external evaluations suggests that file-level evaluation substantially overestimates performance, while speaker-disjoint evaluation can provide a more realistic estimate of the generalization to unseen speakers. In this study, models trained on external data alone do not reliably generalize to L2-ARCTIC. Among the tested configurations, balanced externa
    
[^87]: PPG-Bench：一个面向PPG基础模型的统一基准测试

    PPG-Bench: A Unified Benchmark for PPG Foundation Models

    [https://arxiv.org/abs/2610.13435](https://arxiv.org/abs/2610.13435)

    PPG-Bench是一个统一的开源基准测试，在九个数据集和26个任务上，通过四种下游评估模式对五个PPG基础模型进行了系统、公平的比较。

    

    光电容积脉搏波（PPG）正越来越多地被可穿戴设备和临床设备大规模记录，并且已开发出越来越多的PPG基础模型（FMs），用于为各种下游任务计算可复用的表征。然而这些模型难以相互比较：不同的评估使用不同的数据集、任务定义、预处理方法、下游模型头、超参数调整和基线。我们提出了PPG-Bench，一个统一的开源基准测试，评估了五个公开的PPG基础模型，涵盖九个开放获取数据集和26个任务，涉及心血管状态估计、活动、睡眠、疾病筛查和身份识别。我们在四种适应性递增的下游评估模式（线性探测、随机森林、冻结MLP探测和端到端微调）下对模型进行评估，并采用针对各模型优化的超参数调节和受试者分离的交叉验证。我们还将其与三种对照进行了比较：随机/常数基线，以及……

    arXiv:2610.13435v1 Announce Type: new  Abstract: Photoplethysmography (PPG) is increasingly recorded by wearable and clinical devices at enormous scale, and a growing number of PPG foundation models (FMs) have been developed to compute reusable representations for various downstream tasks. These models are difficult to compare: different evaluations use different datasets, task definitions, preprocessing, downstream heads, hyperparameter tuning, and baselines. We present PPG-Bench, a unifying open-source benchmark evaluating five public PPG FMs across nine open-access datasets and 26 tasks spanning cardiovascular state estimation, activity, sleep, screening, and identification. We evaluate models under four downstream regimes of increasing adaptability (linear probing, random forests, frozen MLP probes, and end-to-end finetuning), with model-specific hyperparameter optimization and subject-disjoint cross-validation. We also evaluate against three controls: random/constant baseline, a d
    
[^88]: 金融语音需要持续学习正则化吗？EWC与mini-KFAC在财报电话会议语音识别上的研究

    Does Financial Speech Need Continual Learning Regularization? A Study of EWC and mini-KFAC on Earnings Call ASR

    [https://arxiv.org/abs/2610.13432](https://arxiv.org/abs/2610.13432)

    该研究发现在将Whisper适配到财报电话会议语音识别时，新任务与旧任务之间存在互补性而非灾难性遗忘，因而正则化反而有害；但当确需正则化时，逐层mini-KFAC比对角EWC能取得更低的词错误率和更稳定的优化效果。

    

    财报电话会议转录呈现了一个现实的持续学习场景：自动语音识别（ASR）系统在用新录音进行更新时，存在灾难性遗忘之前学到的金融语音模式的风险。将Whisper-small从通用领域语音语料库适配到Earnings22基准的实验却揭示了相反的结果：所有被评估的方法在适配过程中对同领域保留集的金融语音表现出提升而非遗忘。这种任务互补性使得正则化反而适得其反。然而，当正则化确有必要时，逐层的mini-Kronecker-Factored近似曲率（mini-KFAC）方法似乎比标准的对角弹性权重固化（EWC）实现了更低的词错误率（WER）和更稳定的优化。研究结果表明，在应用持续学习约束之前应先评估任务互补性，并优先选择mini-KFAC。

    arXiv:2610.13432v1 Announce Type: cross  Abstract: Earnings call transcription presents a realistic continual learning scenario where Automatic Speech Recognition (ASR) systems are updated with new recordings but risk catastrophic forgetting of previously learned financial speech patterns. Adapting Whisper-small from a general-domain speech corpus to the Earnings22 benchmark reveals the opposite: all evaluated methods exhibit improvement on held-out financial speech from the same domain during adaptation rather than forgetting. This task complementarity makes regularization counterproductive. Nevertheless, when regularization is warranted, a layer-wise mini-Kronecker-Factored Approximate Curvature (mini-KFAC) approximation seems to achieve lower word error rates (WERs) and more stable optimization than standard diagonal Elastic Weight Consolidation (EWC). The results suggest that task complementarity should be assessed before applying continual learning constraints, and prefer mini-KFA
    
[^89]: 一因导一果：时间序列因果发现的核心要点

    One Thing Leads to Another: The Essentials of Time Series Causal Discovery

    [https://arxiv.org/abs/2610.13430](https://arxiv.org/abs/2610.13430)

    本教程论文首次为统计学、动力系统和机器学习等不同研究社区的时间序列因果发现方法提供了统一概述，系统介绍了其关键原理、局限性、假设、评估基准方法以及在农业、地球科学、医疗保健和人工智能系统等下游任务中的重要性。

    

    时间序列因果发现——即检测时间序列之间因果关系的过程——起源于半个多世纪前的信号处理研究。此后，它一直受到积极而广泛的关注，并在统计学、动力系统和机器学习等领域演化出不同的形式。进入这个碎片化且不断发展的研究领域可能令人望而生畏。本教程为跨研究社区的时间序列因果发现提供了统一的概述。我们介绍了现代发现方法所采用的关键原理，并讨论了它们的局限性和假设。我们还讨论了如何对发现方法进行评估、基准测试和微调。我们进一步讨论了时间序列因果发现如何在农业与地球科学、医疗保健以及人工智能系统的下游任务中变得至关重要。我们相信本教程将作为一个有价值的……

    arXiv:2610.13430v1 Announce Type: cross  Abstract: Time series causal discovery --- the process of detecting causation between time series --- has its roots in signal processing research from over a half century ago. It has since received active and widespread interest, evolving into different forms across domains such as statistics, dynamical systems and machine learning. Entering this fragmented and constantly evolving research landscape can be daunting. This tutorial provides a unified overview of time series causal discovery across research communities. We introduce the key principles invoked by modern discovery methods, and we discuss their limitations and assumptions. We also discuss how discovery methods are evaluated, benchmarked and fine-tuned. We further discuss how causal discovery for time series is becoming crucial for downstream tasks in agriculture and Earth science, in healthcare, and in artificial intelligence systems. We believe that this tutorial will serve as a valu
    
[^90]: 面向系统辨识的上下文学习：概率化建模、长序列处理与基准验证

    In-Context Learning for System Identification: Probabilistic Formulation, Long-Sequence Handling, and Benchmark Validation

    [https://arxiv.org/abs/2610.13411](https://arxiv.org/abs/2610.13411)

    本文提出一种统一的概率化上下文学习元辨识架构，将基于Transformer的编码器-解码器与长序列处理机制相结合，实现长时域校准预测，并通过基准测试验证了其有效性。

    

    传统的系统辨识通常包含两个步骤：首先，对训练数据进行处理以估计目标系统的模型；然后，利用该模型对新输入查询进行输出预测。相比之下，近期的上下文学习方法学习一个元模型，以零样本方式直接从训练数据和查询输入映射到输出，从而绕过显式的模型估计。该方法利用在大量合成生成数据上离线训练的元模型，来模拟系统辨识算法的行为。在本工作中，我们从三个方向推进了这一面向系统辨识的元学习框架。首先，我们开发了一个统一的概率化元辨识架构，该架构将基于Transformer的编码器-解码器与专为处理长训练序列设计的机制相结合，实现了长时域上的校准预测。其次，我们展示了如何利用预训练的……（摘要原文在此处截断）

    arXiv:2610.13411v1 Announce Type: cross  Abstract: Traditional system identification typically involves two steps: initially, training data is processed to estimate a model of the system of interest; then, the model is used to predict outputs for new input queries. In contrast, recent in-context learning approaches learn a meta model that maps directly from training data and query inputs to outputs in a zero-shot fashion, bypassing explicit model estimation. This approach leverages a meta model trained offline, on large sets of synthetically generated data, to emulate the behavior of a system identification algorithm. In this work, we advance this meta-learning framework for system identification in three directions. First, we develop a unified probabilistic meta-identification architecture that combines a Transformer-based encoder-decoder with tailored mechanisms for handling long training sequences, enabling calibrated predictions over long horizons. Second, we show how a pre-trained
    
[^91]: 利用持久候选集加速线性老虎机问题

    Accelerating Linear Bandits with Persistent Candidate Sets

    [https://arxiv.org/abs/2610.13407](https://arxiv.org/abs/2610.13407)

    PC-LinUCB 通过维护少量活跃动作并每轮均匀采样挑战者，将线性老虎机每轮打分评估次数从 O(K) 降至 O(k+q)，其额外的发现成本不随时间范围增长，并在均匀采样下达到了 Ω((K−k)/q) 的遗憾下界。

    

    在具有大量固定动作集（K 个动作）的线性老虎机问题中，穷举式的乐观打分会使动作选择代价高昂，每轮需要 O(K) 次打分评估。我们提出了持久候选 LinUCB（PC-LinUCB），该方法维护 k 个活跃动作，并在每一轮从其余 K−k 个动作中均匀采样 q 个挑战者。由于只需对这 k+q 个候选动作进行打分，PC-LinUCB 将每轮的打分评估次数降低至 O(k+q)。我们建立了与问题无关和与问题相关的高概率遗憾界，该遗憾界由标准的线性老虎机学习项以及一个在固定置信度下量级为 O((K−k)log(K−k)/q) 的额外发现成本组成。值得注意的是，这一额外成本并不随时间范围增长。对于固定动作间隔和足够长的时间范围，我们在均匀挑战者采样下建立了 Ω((K−k)/q) 的遗憾下界，表明……

    arXiv:2610.13407v1 Announce Type: new  Abstract: Exhaustive optimistic scoring can make action selection costly in linear bandits with a large fixed set of $K$ actions, requiring $\mathcal{O}(K)$ score evaluations per round. We introduce Persistent Candidate LinUCB (\textsc{PC-LinUCB}), which maintains $k$ active actions and samples $q$ challengers uniformly from the remaining $K-k$ actions at each round. By scoring only these $k+q$ candidates, \textsc{PC-LinUCB} reduces the number of score evaluations per round to $\mathcal{O}(k+q)$. We establish problem-independent and problem-dependent high-probability regret bounds consisting of a standard linear-bandit learning term and an additive discovery cost of order $\mathcal{O}\left((K-k)\log(K-k)/q\right)$ at fixed confidence. Notably, this additional cost does not grow with the horizon. For fixed gaps and sufficiently long horizons, we establish an $\Omega\left((K-k)/q\right)$ regret lower bound under uniform challenger sampling, showing 
    
[^92]: 半干旱地区的荒漠化预测：利用深度学习探究土地退化动态的可观测性极限

    Desertification Forecasting in Semi-Arid Regions: Using Deep Learning to Probe the Observability Limits of Land Degradation Dynamics

    [https://arxiv.org/abs/2610.13406](https://arxiv.org/abs/2610.13406)

    该研究用ConvLSTM模型基于卫星观测数据预测巴西塞阿拉州土地退化指数的逐年演变，发现该指数的像元级变化以难以预测的逐年跳变为主（梯度提升模型仅能解释4.7%的方差），而ConvLSTM仍能相对持续性基线将提前一至四年预测的RMSE降低1.6%至4.4%，从而揭示了土地退化动态的可观测性极限。

    

    荒漠化的预测所受到的关注远少于荒漠化制图。我们使用一个卷积长短期记忆网络，在巴西半干旱的塞阿拉州上，基于2000至2023年的十种年度地球观测（EO）变量，以约2.2公里分辨率的256×206网格进行训练，以衡量卫星数据能在多大程度上预测MapBiomas衍生退化指数（IDV）的逐年演变。该网络在最后观测状态之上预测一个有界的变化量，采用两阶段训练（先单步预测，再递归展开），并在连续的空间与时间保留集上进行评估。2000年至2023年间，全州平均IDV从0.238上升至0.350，约每年0.005，但像元级别的变化以逐年跳变为主，一个输入全部通道的梯度提升模型仅能解释其方差的4.7%。相对于这一上限，ConvLSTM将提前一年预测的RMSE相对持续性基线降低了1.6%，提前四年降低了4.4%……

    arXiv:2610.13406v1 Announce Type: cross  Abstract: Forecasting desertification has received far less attention than mapping it. We use a convolutional LSTM (ConvLSTM) trained on ten annual Earth Observation (EO) variables (2000-2023) over the semi-arid state of Cear\'a, Brazil, on a 256x206 grid at roughly 2.2 km resolution, to measure how much of the year-to-year evolution of a MapBiomas-derived degradation index (IDV) can be predicted from satellite data. The network predicts a bounded change on top of the last observed state, is trained in two stages (single-step, then recursive rollout) and is evaluated on a contiguous spatial and temporal holdout. State-wide mean IDV rose from 0.238 to 0.350 between 2000 and 2023, about 0.005 per year, but cell-level change is dominated by year-to-year jumps, and a gradient-boosting model given all channels explains only 4.7% of its variance. Against this ceiling, the ConvLSTM lowers RMSE relative to persistence by 1.6% one year ahead and 4.4% fou
    
[^93]: 为什么线性模型在复杂多变的世界中仍然有效

    Why Linear Models Can Still Work in a Complex and Changing World

    [https://arxiv.org/abs/2610.13404](https://arxiv.org/abs/2610.13404)

    该论文论证了当世界的复杂性超出有限、部分观测数据所能可靠学习的范围时，灵活的非线性模型会过度拟合特定环境的细节而难以迁移，因此省略不稳定非线性成分的线性模型在分布变化的新环境中反而能取得更低的预测误差。

    

    即使底层机制是非线性的，线性模型在复杂领域中也能做出良好的预测。我们认为，这是因为世界的复杂性超过了从有限的、部分观测的数据中能够可靠学习的复杂性。灵活的模型可以利用曲率、阈值和交互效应，这些特征可能恰好适用于某个医院、某个时期、某个人群、某种政策环境或某种测量过程，但在其他环境中却会失效。当这种不稳定性主要存在于非线性成分中时，忽略该成分反而能让线性模型在新环境中获得更低的预测误差。我们将这一论点与关于简单线性规则、偏差与方差、正则化、最佳线性预测、测量误差、分布偏移、不变性、Rashomon集合以及欠规范性的研究联系起来。命题1给出了一个简单的平方误差情形：在训练环境中学习的非线性规则因其迁移能力太弱，反而提高了目标环境中的误差；有限样本估计……

    arXiv:2610.13404v1 Announce Type: new  Abstract: Linear models can predict well in complex domains even when underlying mechanisms are nonlinear. We argue that this can occur because the complexity of the world exceeds the complexity that can be learned reliably from finite, partially observed data. Flexible models can exploit curvature, thresholds, and interactions that fit one hospital, period, population, policy setting, or measurement process but fail elsewhere. When this instability lies mainly in the nonlinear component, omitting it can give linear models lower prediction error in a new environment. We connect this claim to work on simple linear rules, bias and variance, regularization, best linear prediction, measurement error, distribution shift, invariance, Rashomon sets, and underspecification. Proposition 1 gives a simple squared-error case in which a nonlinear rule learned in the training environment raises target error because it transfers too weakly; finite-sample estimat
    
[^94]: 无模型贝叶斯聚类与多模态后验摘要

    Model-free Bayesian Clustering and Multimodal Posterior Summarisation

    [https://arxiv.org/abs/2610.13397](https://arxiv.org/abs/2610.13397)

    本文提出了一种无模型、噪声感知的贝叶斯聚类框架，通过鞅后验避免MCMC采样，并利用Wasserstein型摘要方法将多模态后验高效压缩为少量加权代表，同时边缘化了密度水平的不确定性。

    

    我们开发了一个无模型、噪声感知的贝叶斯聚类框架，该框架从预测贝叶斯的视角构建聚类划分上的鞅后验。这一方法避免了显式的似然-先验模型设定、昂贵的MCMC采样以及寻求众数的变分近似，同时能够适应不规则的簇形状，并支持后验样本的并行生成。为了解释高维、噪声感知聚类上的后验分布，我们引入了一种Wasserstein类型的摘要方法。利用一种新颖的噪声感知损失，该方法将大量后验样本压缩为少量的加权代表，从而捕捉不同的簇结构与噪声分配情况。此外，我们对密度水平的不确定性进行边缘化处理，避免了对特定密度水平选择的依赖。我们建立了所提损失的理论性质，并为水平边缘化后验近似提供了理论保证。

    arXiv:2610.13397v1 Announce Type: cross  Abstract: We develop a model-free, noise-aware Bayesian clustering framework that constructs a martingale posterior over clusterings from a predictive Bayesian perspective. This avoids explicit likelihood-prior model specification, expensive MCMC sampling, and mode-seeking variational approximations, while accommodating irregular cluster shapes and enabling parallel generation of posterior draws. To interpret the posterior over high-dimensional, noise-aware clusterings, we introduce a Wasserstein-type summarization method. Using a novel noise-aware loss, this approach compresses the large collection of posterior draws into a small set of weighted representatives, capturing distinct cluster structures and noise assignments. Further, we marginalise the uncertainty in the density level, avoiding reliance on a specific choice. We establish theoretical properties of the proposed loss and guarantees for the level-marginal posterior approximation. Resu
    
[^95]: PlacePulse：从结构性地理数据中学习可迁移的地点表示

    PlacePulse: Learning Transferable Place Representations from Structural Geographic Data

    [https://arxiv.org/abs/2610.13390](https://arxiv.org/abs/2610.13390)

    PlacePulse是一个结构性地理空间基础模型，它将美国超过105万个H3区域的572个静态结构属性压缩为可迁移的256维嵌入表示，并配套提出了区分内在表示质量与下游预测性能的双轴评估基准。

    

    地理空间机器学习通常依赖于对大量社会人口、商业和环境变量进行特定任务的特征工程。地理空间基础模型虽然提供了可复用的地点表示，但主流方法通常编码的是动态信号（如人口流动和搜索活动）或来自卫星影像的视觉信号。很少有模型将地点的静态结构组成（谁在那里生活和工作、哪些企业在此经营、以及它们面临哪些风险）表示为精细空间分辨率下的可迁移嵌入。我们提出了PlacePulse，这是一个结构性地理空间基础模型，它将572个属性（通过H3邻域聚合扩展为2,288个输入）压缩为256维的单位范数嵌入，覆盖美国1,055,537个H3分辨率-7单元。我们还引入了一个可复用的双轴基准，将内在表示质量与下游预测（性能）分离开来。

    arXiv:2610.13390v1 Announce Type: new  Abstract: Geospatial machine learning often depends on task-specific feature engineering over large collections of sociodemographic, business, and environmental variables. Geospatial foundation models offer reusable representations of place, but leading approaches typically encode dynamic signals, such as mobility and search activity, or visual signals from satellite imagery. Few models represent the static structural composition of places (who lives and works there, which businesses operate there, and which risks they face) as transferable embeddings at fine spatial resolution. We introduce PlacePulse, a structural geospatial foundation model that compresses 572 attributes, expanded to 2,288 inputs through H3 neighborhood aggregates, into 256-dimensional unit-norm embeddings for 1,055,537 H3 resolution-7 cells in the United States. We also introduce a reusable two-axis benchmark that separates intrinsic representation quality from downstream pred
    
[^96]: BLOKS：通过草图化块预条件子扩展核方法求解器

    BLOKS: Scaling Kernel Solvers with Sketched Block Preconditioning

    [https://arxiv.org/abs/2610.13382](https://arxiv.org/abs/2610.13382)

    BLOKS提出了一种将块Jacobi近似与草图化相结合的新颖预条件子，在加速核岭回归近似求解收敛速度的同时有效控制了内存成本，兼顾了效率与模型表达能力。

    

    核岭回归（KRR）是一种理论上严谨的非参数学习算法，也常被用于研究深度学习的性质。然而，精确的KRR公式会带来高昂的时间和内存成本，使其难以应用于大规模数据集，这推动了近似求解器的发展。这些算法依赖于：（i）迭代方法，以避免直接求解的计算负担；（ii）草图化或批处理技术，以减小问题规模；（iii）预条件技术，以加速收敛。尽管预条件对于快速收敛至关重要，但其带来的内存成本可以通过减少模型的自由参数数量来控制，不过这可能以牺牲模型的表达能力进而牺牲精度为代价。为应对这一挑战，我们提出了BLOKS，这是一种基于共轭梯度法的Nyström草图化KRR求解器，其核心是一种新颖的预条件子，该预条件子将块Jacobi近似与草图化技术相结合。

    arXiv:2610.13382v1 Announce Type: new  Abstract: Kernel Ridge Regression (KRR) is a theoretically sound algorithm for nonparametric learning, which is also often used to investigate properties of deep learning. However, the exact KRR formulation incurs time and memory costs preventing applicability to large datasets, motivating the development of approximate solvers. These algorithms rely on (i) iterative methods, to avoid the computational burden of direct solutions; (ii) sketching or batching, to reduce the problem size; and (iii) preconditioning, to accelerate convergence. While crucial for fast convergence, preconditioning incurs memory costs that can be managed by lowering the number of free parameters of the model, potentially at the expense of expressivity and hence accuracy. To address this challenge, we propose BLOKS, a Conjugate Gradient-based solver for Nystr\"om-sketched KRR built around a novel preconditioner that combines a block Jacobi approximation with sketching. The B
    
[^97]: 利用专家选择偏差克服数据不平衡

    Overcoming Data Imbalances with Expert Selection Biases

    [https://arxiv.org/abs/2610.13370](https://arxiv.org/abs/2610.13370)

    该论文提出在MoE模型中引入基于类别的专家选择偏差，将不同类别的词元优先路由到专门的专家子集，从而在数据类别不平衡的情况下实现软性类别对齐的专家专门化，有效缓解少数类别容量受限的问题。

    

    混合专家模型已成为前沿大语言模型的标准架构。每个Transformer块中的多个并行专家在理论上能够在预训练期间实现专家按领域、语言或模态等类别进行专门化。然而，当数据在各类别之间分布不平衡时，少数类别可能会集中在专家池中的一小部分，或者在专家之间低效地分散，从而实际上限制了它们可用的模型容量。在这项工作中，我们证明了基于类别的专家选择偏差可以将来自不同类别的词元优先路由到专门的专家池。通过根据输入类别施加偏差，我们提供了一种诱导软性类别对齐专门化的简单方法。这种干预促使各类别趋向于不同的专家子集，同时在有用时保留跨类别的共享。更重要的是，它使得过度分配

    arXiv:2610.13370v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models have become the standard architecture for frontier LLMs. The multiple parallel experts in each transformer block theoretically enable expert specialization according to categories such as domains, languages, or modalities during pretraining. However, when data is imbalanced across categories, minority categories can become concentrated in a small fraction of the expert pool or spread inefficiently across them, effectively limiting their available capacity. In this work, we demonstrate that category-conditioned expert-selection biases can preferentially route tokens from different categories toward specialized expert pools. By applying biases based on input category, we provide a simple way to induce soft category-aligned specialization. This intervention encourages categories toward distinct expert subsets while retaining cross-category sharing when useful. More importantly, it enables the over-allocation 
    
[^98]: 基于信息瓶颈打开无分类器引导的黑箱

    Opening the Black Box of Classifier-Free Guidance via Information Bottleneck

    [https://arxiv.org/abs/2610.13369](https://arxiv.org/abs/2610.13369)

    该论文从信息论视角将扩散生成轨迹建模为随时间变化的信息瓶颈，揭示了条件信息的有效性在经过一个关键阶段后会显著增强，从而为无分类器引导（CFG）的激活时机提供了原则性的理论基础。

    

    无分类器引导（CFG）是提升扩散模型条件生成质量的标准技术，但其激活调度通常依赖启发式选择。在实践中，大多数现有基于CFG的方法在整个推理过程中依赖于预定义的激活策略，而近期的动态变体则往往依赖手工设计的调度，缺乏关于条件信息在何时最为有用的原则性理论解释。为了理解条件输入与得分函数之间的信息相关性在生成过程中如何演化，我们从信息论的视角探讨了引导应当何时被激活。在本工作中，我们将扩散轨迹形式化为一个随时间变化的信息瓶颈，并证明条件信息的有效性在整个采样过程中并非均匀分布，而是在一个以Fishe（费雪信息）为特征的关键阶段之后显著增强。

    arXiv:2610.13369v1 Announce Type: new  Abstract: Classifier-free guidance (CFG) is a standard technique for improving conditional generation in diffusion models, yet its activation schedule is usually selected heuristically. In practice, most existing CFG-based methods rely on a pre-defined activation policy throughout inference, while recent dynamic variants often depend on hand-designed schedules without a principled theoretical explanation of when conditional information is most useful. In order to understand how the informational relevance between conditioning inputs and the score function evolves over the course of generation, we explore when guidance should be activated from an information-theoretic perspective. In this work, we formulate the diffusion trajectory as a time-dependent information bottleneck and show that the effectiveness of conditional information is not uniform throughout sampling, but becomes significantly stronger after a critical stage characterized by a Fishe
    
[^99]: AgentSWE：编码智能体能构建出你真正想要的智能体吗？

    AgentSWE: Can Coding Agents Build the Agent You Actually Want?

    [https://arxiv.org/abs/2610.13361](https://arxiv.org/abs/2610.13361)

    提出了首个检验编码智能体能否忠实构建出真正满足用户需求的智能体的基准 AgentSWE，其以“委托”形式覆盖创建、编辑、优化三个开发阶段共25个任务，并通过保留用例和需求驱动的严格评分标准进行验收。

    

    随着软件的生产逐渐交由软件本身来完成，编码智能体越来越多地被委托去构建贴合用户自身工作流的智能体。这种委托隐含着一个未被言明的假设：交付的智能体将忠实服务于它被构建时所要满足的需求，而不仅仅是表面上看起来如此。目前尚无任何基准测试检验这一假设。我们提出 AgentSWE，一个面向智能体软件工程的基准，涵盖智能体开发生命周期的三个阶段：根据自然语言需求创建完整智能体、编辑生产级智能体代码库，以及朝向可度量目标进行优化。其25个任务中每一个都被设计为一次“委托”而非一场“考试”：构建者仅针对少量示例用例进行迭代，而验收环节则在保留用例上对冻结提交的成果进行评估，这些保留用例旨在覆盖完整需求，并依据源自需求的评分标准以严格的通过/……（原文摘要在此处截断）

    arXiv:2610.13361v1 Announce Type: cross  Abstract: As the production of software passes into the hands of software itself, coding agents are increasingly commissioned to build agents fitted to their users' own workflows. This delegation carries a quiet assumption, that the delivered agent will faithfully serve the requirement it was built for, rather than merely appear to. No existing benchmark tests that assumption. We introduce AgentSWE, a benchmark for agent software engineering spanning three stages of the agent development lifecycle: Creation of a complete agent from a natural-language requirement, Editing of production agent codebases, and Optimization toward a measurable target. Each of its 25 tasks is framed as a commission rather than an exam: the builder iterates against a handful of illustrative cases, while acceptance exercises the frozen submission on held-out cases designed to span the full requirement, grading behavior against requirement-derived rubrics under hard pass/
    
[^100]: EmoGavel：基于可验证奖励的定位感知强化学习用于多模态情感偏好学习

    EmoGavel: Grounding-Aware Reinforcement Learning with Verifiable Rewards for Multimodal Emotion Preference Learning

    [https://arxiv.org/abs/2610.13353](https://arxiv.org/abs/2610.13353)

    提出EmoGavel框架，通过将定位感知监督微调与可验证奖励强化学习相结合，在多模态情感偏好学习中实现偏好预测、错误定位、修复和缺失证据恢复的联合优化。

    

    多模态情感偏好学习通过成对比较来识别哪些开放式情感描述更符合人类判断。然而，现有方法主要关注最终的偏好决策，对于定位和纠正错误内容几乎不提供显式监督。为了解决这一局限性，我们提出了EmoGavel，一个用于多模态情感偏好学习的定位感知可验证奖励强化学习框架。我们首先引入定位监督微调，利用受控错误构建信息来联合学习偏好预测、错误定位、修复和缺失证据恢复。然后，我们通过定位感知组相对策略优化来优化这些可验证的结果，将偏好门控定位奖励与动态零方差救援机制相结合，以优先保证偏好正确性并恢复。

    arXiv:2610.13353v1 Announce Type: cross  Abstract: Multimodal emotion preference learning uses pairwise comparisons to identify which open-ended emotion descriptions align more closely with human judgments. However, existing methods primarily focus on the final preference decision and provide little explicit supervision for localizing and correcting erroneous content. To address this limitation, we propose EmoGavel, a Grounding-Aware reinforcement learning framework with Verifiable rewards for multimodal Emotion preference Learning. We first introduce grounded supervised fine-tuning (Grounded SFT), which exploits controlled-error construction information to jointly learn preference prediction, error localization, repair, and missing-evidence recovery. We then optimize these verifiable outcomes with grounding-aware group relative policy optimization (GA-GRPO), combining preference-gated grounding rewards with dynamic zero-variance rescue to prioritize preference correctness and recover 
    
[^101]: 临床机器学习中的隐性数据挑战

    Hidden Data Challenges in Clinical Machine Learning

    [https://arxiv.org/abs/2610.13350](https://arxiv.org/abs/2610.13350)

    该论文以MIMIC-IV为案例，系统揭示并量化了临床机器学习中广泛复用的预处理流水线在时间、语义和人群对齐上的隐性假设违规——包括时间泄漏、特征-任务错位和有偏插补——并说明这些模糊的数据处理选择会如何改变模型可见的信息并影响预测解释。

    

    被广泛复用的电子健康记录预处理流水线可能隐含着关于信息何时可用、标签代表何种临床事件片段、以及缺失值含义的假设。这些假设很少得到系统性审计，并且当预处理流水线被复用或改编时，可能会在后续研究中传播。我们以MIMIC-IV为案例研究，识别并量化了三种互补的预处理对齐形式（即时间对齐、语义对齐和人群对齐）的违规情况，它们分别表现为时间泄漏、特征-任务错位和有偏插补。我们展示了时间戳语义、事件片段定义、终点构建、模态选择和缺失值处理中的模糊性如何改变模型可获取的信息，并影响对其预测结果的解释。通过对MIMIC-IV数据的分析和实证案例研究，我们证明了这些数据处理选择可能仍然……

    arXiv:2610.13350v1 Announce Type: new  Abstract: Widely reused preprocessing pipelines for electronic health records can encode assumptions about when information became available, what clinical episode a label represents, and what missingness means. These assumptions are rarely audited systematically and may propagate across subsequent studies when preprocessing pipelines are reused or adapted. Using MIMIC-IV as a case study, we identify and quantify violations of three complementary forms of preprocessing alignment, i.e., temporal, semantic, and population alignment, which manifest as temporal leakage, feature-task misalignment, and biased imputation. We show how ambiguities in timestamp semantics, episode definitions, endpoint construction, modality selection, and missingness handling can alter the information available to a model and affect the interpretation of its predictions. Through analyses of MIMIC-IV data and empirical case studies, we demonstrate that such choices may remai
    
[^102]: 通过振荡动力学理解 Mamba-3

    Understanding Mamba-3 Through Oscillatory Dynamics

    [https://arxiv.org/abs/2610.13341](https://arxiv.org/abs/2610.13341)

    Mamba-3 中依赖输入的旋转可被理解为欠阻尼二阶动力学，这种振荡机制通过调节信息的积累方式，提升了状态跟踪、长上下文预测和长度泛化能力。

    

    Mamba-3 在循环状态更新中引入了依赖输入的旋转，但其在状态保持之外的贡献仍不明确。我们将这些旋转解释为欠阻尼二阶动力学，为研究振荡在序列建模中的作用提供了一个框架。受控对比实验表明，欠阻尼动力学在状态跟踪任务上优于非振荡的阻尼机制，同时语言模型预训练结果显示其对长上下文预测和长度泛化具有益处。对预训练模型的干预实验进一步证实了学习到的相位的重要性，状态分析表明旋转调节了保留信息与新输入更新的结合方式。综合来看，这些发现将欠阻尼动力学确认为循环计算的一种有效机制，并表明振荡通过调节信息积累的方式支持长上下文处理。

    arXiv:2610.13341v1 Announce Type: new  Abstract: Mamba-3 introduces input-dependent rotations into recurrent state updates, but their contribution beyond state retention remains unclear. We interpret these ro- tations as underdamped second-order dynamics, providing a framework for in- vestigating the role of oscillation in sequence modeling. Controlled comparisons show that underdamped dynamics outperform non-oscillatory damping regimes on state-tracking tasks, while language-model pretraining demonstrates benefits for long-context prediction and length generalization. Interventions in pretrained models further establish the importance of learned phase, and state analysis sug- gests that rotation regulates how retained information combines with incoming up- dates. Taken together, these findings identify underdamped dynamics as a useful mechanism for recurrent computation and suggest that oscillation supports long- context processing by regulating how information accumulates.
    
[^103]: AgentTune：面向工具使用型大语言模型智能体的统一训练生命周期

    AgentTune: A Unified Training Lifecycle for Tool-Using LLM Agents

    [https://arxiv.org/abs/2610.13335](https://arxiv.org/abs/2610.13335)

    提出了AgentTune——一个统一的开源Python框架，通过两层EventLog表示整合了工具使用型LLM智能体从构建、训练、评估到蒸馏、维护的全生命周期流程。

    

    工具使用型智能体日益需要一个共同的底层基础来构建、训练、评估、蒸馏和维护智能体系统，然而现有工具通常只是孤立地处理这些阶段。我们提出了AgentTune，一个源码开放的Python库，它为开发和优化工具使用型智能体提供了统一框架。AgentTune的核心是一种两层EventLog表示，它在保留智能体轨迹的同时，支持为训练、评估和审计提供一致的投影。基于这一底层基础，该库提供了共享的智能体运行时、工具调用强化学习、智能体蒸馏、检索增强生成、决策工作流、评估工具，以及通往自我修复智能体的设计路径。这些组件通过可组合的接口暴露出来，使智能体策略、训练方法、评估流程和恢复机制能够灵活组合使用。

    arXiv:2610.13335v1 Announce Type: new  Abstract: Tool-using agents increasingly require a common substrate for building, training, evaluating, distilling, and maintaining agentic systems, yet existing tooling typically addresses these stages in isolation. We present AgentTune, a source-available Python library that provides a unified framework for developing and optimizing tool-using agents. At its core, AgentTune introduces a two-tier EventLog representation that preserves agent trajectories while supporting consistent projections for training, evaluation, and auditing. Building on this substrate, the library provides a shared agent runtime, tool-calling reinforcement learning, agentic distillation, retrieval-augmented generation, decision workflows, evaluation utilities, and a designed path toward self-healing agents. These components are exposed through composable interfaces so that agent strategies, training methods, evaluation procedures, and recovery mechanisms can be combined wi
    
[^104]: PDE-PFN：基于先验数据拟合的神经PDE求解器

    PDE-PFN: Prior-Data Fitted Neural PDE Solver

    [https://arxiv.org/abs/2610.13334](https://arxiv.org/abs/2610.13334)

    提出了PDE-PFN，一种通过上下文学习近似PDE解的后验预测均值的先验数据拟合神经求解器，它利用物理信息神经网络生成的带噪声近似解进行预训练，实现了跨异构方程的泛化能力以及对噪声先验的鲁棒性。

    

    受大语言模型（LLMs）具有广泛泛化能力以及对噪声或不可靠预训练数据的鲁棒性这一成功的启发，我们探索使用近似预训练数据对PDE解场进行上下文内预测。此外，受先验数据拟合网络中基于后验预测均值的上下文学习推理机制的启发，我们提出了PDE-PFN——一种通过上下文学习来近似PDE解的后验预测均值的先验数据拟合神经求解器。PDE-PFN构建于具有Transformer自注意力和交叉注意力机制的PFN架构之上，并在由物理信息神经网络生成的带噪声近似解上进行预训练，这些解作为多样但未必精确的先验。通过在一系列二维PDE上的实验，我们证明PDE-PFN在异构方程上实现了经验泛化，在噪声先验下表现出鲁棒性……

    arXiv:2610.13334v1 Announce Type: new  Abstract: Motivated by the success of large language models (LLMs) with broad generalizability and robustness to noisy or unreliable pre-training data, we explore in-context prediction of PDE solution fields using approximate pre-training data. In addition, inspired by the posterior predictive mean-based inference mechanism of the in-context learning in prior-data fitted networks (PFNs), we propose PDE-PFN, a prior-data fitted neural solver that approximates the posterior predictive mean of PDE solutions via in-context learning. PDE-PFN builds on a PFN architecture with self- and cross-attention mechanisms of Transformer and is pre-trained on noisy approximate solutions generated by physics-informed neural networks, serving as diverse but not necessarily exact priors. Through experiments on a range of two-dimensional PDEs, we demonstrate that PDE-PFN achieves empirical generalization across heterogeneous equations, robustness under noisy priors, a
    
[^105]: 面向工具集成化学推理的掩码扩散语言智能体

    Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning

    [https://arxiv.org/abs/2610.13330](https://arxiv.org/abs/2610.13330)

    本文提出首个基于掩码扩散语言模型的化学推理智能体ChemDiffAgent，将化学推理建模为对推理与工具使用轨迹的迭代去噪过程，使工具使用决策能够在全轨迹上下文中联合确定，突破了传统自回归范式在双向分子理解和多源证据整合上的局限。

    

    为复杂化学任务设计工具集成的推理智能体系统仍然是一个根本性的挑战。尽管近期的化学智能体通过集成外部工具展现出有前景的性能，但它们仍建立在自回归范式之上，这种范式本质上受限于因果注意力和从左到右的生成方式。因此，它们在双向分子理解、全局协调的工具规划以及多源证据整合方面常常表现不佳。为应对这一挑战，我们提出了ChemDiffAgent，这是首个用于工具集成化学推理的掩码扩散语言智能体。它将化学推理重新表述为对交错的推理与工具使用轨迹的迭代去噪过程，使得工具使用决策能够在完整轨迹的上下文中联合确定，从而更好地将智能体的决策边界与其知识边界对齐。我们进一步分析了如何……（原文摘要在此处截断）

    arXiv:2610.13330v1 Announce Type: new  Abstract: Designing tool-integrated reasoning agent systems for complex chemical tasks remains a fundamental challenge. Although recent chemical agents have demonstrated promising performance through the integration of external tools, they are still built on the autoregressive paradigm, which is inherently constrained by causal attention and left-to-right generation. As a result, they often struggle with bidirectional molecular understanding, global coordinated tool planning and multi-source evidence integration. To address this challenge, we introduce ChemDiffAgent, the first masked diffusion language agent for tool-integrated chemical reasoning. It reformulates chemical reasoning as iterative denoising over interleaved reasoning and tool-use trajectories, enabling tool-use decisions to be jointly determined under full-trajectory context and thereby better aligning the agent's decision boundary with its knowledge boundary. We further analyze how 
    
[^106]: DuoRate：面向混合专家模型服务的双速率逐专家精度分配

    DuoRate: Two-Rate Per-Expert Precision Allocation for Mixture-of-Experts Serving

    [https://arxiv.org/abs/2610.13328](https://arxiv.org/abs/2610.13328)

    针对MoE模型中存储占用与解码流量这两种不同速率需求之间的“存储-读取差距”，DuoRate提出离线存储单一精度层级、在线依据工作负载自身的预填充特征求解解码泛函下的率失真分配，实现逐专家精度的双速率优化。

    

    服务混合专家模型需要为专家精度付出两次代价：常驻内存占用需要为路由器可达的每一个专家付费，而每个批量大小为1的解码步骤只需为其选中的专家付费。这是作用在相同的逐专家宽度上的两个不同的速率泛函，然而已发表的逐专家MoE分配器只约束了前者。我们将这种差异称为“存储-读取差距”（store-read gap）；在已发布的BitsMoE检查点中，每权重2.54比特（bpw）的存储量在解码时变成了每权重7.50比特的流量。我们提出DuoRate，它只存储一次，而按工作负载读取。离线阶段，它存储一个前缀可读权重流的统一精度层级，并预先计算一个失真张量。在线阶段，它无需标签地剖析工作负载自身的预填充过程，并在解码泛函下求解拉格朗日率失真分配问题；可以证明路由频率在每个逐专家决策中都被消去，求解成本仅为每个专家一次点积和一次二分查找。

    arXiv:2610.13328v1 Announce Type: new  Abstract: Serving a Mixture-of-Experts (MoE) model pays for expert precision twice: the resident footprint pays for every expert the router can reach, while each batch-one decode step pays only for the experts it selects. These are two different rate functionals over the same per-expert widths, yet published per-expert MoE allocators constrain only the first. We call the difference the store-read gap; in the released BitsMoE checkpoint, 2.54 bits per weight (bpw) of storage become 7.50 bpw of decode traffic. We propose DuoRate, which stores once and reads per workload. Offline, it stores one uniform tier of a prefix-readable weight stream and bakes a distortion tensor. Online, it profiles a workload's own prefill without labels and solves a Lagrangian rate-distortion allocation under the decode functional; routing frequency provably cancels from every per-expert decision, and the solve costs one dot product per expert and a bisection. A deployment
    
[^107]: 面向智能体化 LLM 服务的分词任务主机卸载方案

    Taking Tokenization off the Host for Agentic LLM Serving

    [https://arxiv.org/abs/2610.13325](https://arxiv.org/abs/2610.13325)

    提出 GraphTok，一个将完整分词流水线封装为可重放 CUDA 图并逐位复现参考结果的 GPU 分词器，比 CPU 快 4.7-6.0 倍，可将分词从主机彻底卸载，从而显著降低智能体场景下 LLM 服务的首 token 延迟并提升 GPU 集群可扩展性。

    

    智能体工作负载会在多次调用间重复发送几乎不变的长上下文。前缀缓存如今已能跳过大部分预填充阶段，但在查询缓存之前，每个请求仍必须完整地进行分词。分词这一长期以来被视为可忽略不计的 CPU 阶段，如今已成为首 token 时间（TTFT）中占比最大的部分，而为一个 GPU 集群进行分词的前端也限制了其所能支撑的 GPU 数量。我们提出了 GraphTok，一个 GPU 分词器，它将整个编码流水线——从预分词、字节对编码（BPE）合并到 token ID 组装——作为一个可重放的 CUDA 图来运行。GraphTok 在 GPT-2、Llama 3、Qwen 2.5、DeepSeek-V3 和 Gemma 3 上逐位复现了参考 token ID，在 GPU 无法占优的短输入场景下将其交给 CPU 引擎处理，并可无缝接入未经修改的 vLLM 和 NVIDIA Dynamo。GraphTok 在包含数据传输的情况下，仅用 1.7-3.2 毫秒即可编码一百万个 GPT-2 token，在 Llama 3 上比 giga（原文截断）快 4.7-6.0 倍。

    arXiv:2610.13325v1 Announce Type: cross  Abstract: Agentic workloads resend long contexts that barely change between calls. Prefix caching now skips most of their prefill, yet every request must still be tokenized in full before the cache can even be consulted. Tokenization, a CPU stage long treated as negligible, becomes the largest part of time-to-first-token (TTFT), and a frontend that tokenizes for a whole GPU fleet limits how many GPUs it can feed. We present GraphTok, a GPU tokenizer that runs the entire encoding pipeline, from pre-tokenization through byte-pair encoding (BPE) merging to token-ID assembly, as one replayable CUDA graph. GraphTok reproduces the reference token IDs bit-for-bit on GPT-2, Llama 3, Qwen 2.5, DeepSeek-V3, and Gemma 3, hands short inputs to a CPU engine where the GPU cannot win, and plugs into unmodified vLLM and NVIDIA Dynamo. GraphTok encodes one million GPT-2 tokens in 1.7-3.2 ms including data transfers, and on Llama 3 it is 4.7-6.0x faster than giga
    
[^108]: SBNO：面向信息缺失场景下正问题与逆问题的薛定谔桥神经算子

    SBNO : Schr\"odinger Bridge Neural Operator for the Forward and Inverse Problems under Missing Information

    [https://arxiv.org/abs/2610.13317](https://arxiv.org/abs/2610.13317)

    SBNO提出了一种免方程的薛定谔桥匹配神经算子框架，在观测稀疏且物理信息部分缺失的情况下，以单一模型同时解决PDE的正问题与逆问题。

    

    偏微分方程（PDE）代理建模通常依赖于稠密的观测数据，或者可以获取控制方程及物理约束。而在实际应用中，观测往往是稀疏且不完整的，底层物理机理也可能仅被部分给定或完全不可得。我们提出了SBNO，这是一个免方程的、基于成对数据的扩散桥匹配框架，能够在部分观测条件下为PDE的正问题和逆问题学习条件算子。SBNO利用扩散薛定谔桥匹配，从一种掩码感知的端点耦合中学习一对时间依赖的漂移项，该耦合在训练过程中保留了经验上的输入-解配对关系，并在单一框架内同时支持两种PDE任务方向。在推理阶段，SBNO通过积分由学习到的漂移对构建的概率流常微分方程（ODE），生成确定性的点重建结果，无需在推理时进行观测引导或PDE引导权重的调节。在五个PDE基准测试上（原文在此处截断）……

    arXiv:2610.13317v1 Announce Type: new  Abstract: Partial differential equation (PDE) surrogate modeling often relies on either dense observations or access to governing equations/physics constraints. In practice, observations can be sparse and partial, and the underlying physics may be partially specified or unavailable. We introduce SBNO, an equation-free paired-data diffusion bridge matching framework that learns conditional operators for both forward and inverse PDE problems under partial observations. SBNO uses diffusion Schr\"odinger bridge matching to learn a pair of time-dependent drifts from a mask-aware endpoint coupling that preserves empirical input-solution pairing during training and supports both PDE-task directions within a single framework. At inference time, SBNO produces a deterministic point reconstruction by integrating the probability-flow ODE constructed from the learned drift pair, without inference-time observation or PDE guidance-weight tuning. On five PDE benc
    
[^109]: JEPA-kMMVAE：时变无线信道的信道状态信息预测

    JEPA-kMMVAE: CSI Prediction for Time-Varying Wireless Channels

    [https://arxiv.org/abs/2610.13315](https://arxiv.org/abs/2610.13315)

    提出JEPA-kMMVAE框架，将联合嵌入预测架构与k记忆马尔可夫变分自编码器结合，通过时间结构化的潜在表示从含噪观测中预测时变多天线信道的未来CSI，其中残差校正在中高信噪比下优于基线方法。

    

    本文研究了利用神经网络从含噪历史观测中对时变多天线信道进行信道状态信息（CSI）预测的问题。主要挑战在于如何利用时间相关性，同时区分信道演化与观测噪声。我们提出了JEPA-kMMVAE，该方法将联合嵌入预测架构与k记忆马尔可夫变分自编码器相结合，通过具有时间结构的潜在表示来预测未来CSI。该框架支持直接预测以及相对于最新潜在状态和含噪CSI观测的残差校正两种模式。仿真结果表明，残差配置在中高输入CSI信噪比下优于所评估的基线方法，而在较强噪声环境下序列基线方法仍能保持更高的精度。

    arXiv:2610.13315v1 Announce Type: cross  Abstract: This paper considers channel state information (CSI) prediction using neural networks for time-varying multi-antenna channels from noisy historical observations. The main challenge is to exploit temporal correlation while distinguishing channel evolution from observation noise. We propose JEPA-kMMVAE, which combines a joint-embedding predictive architecture with a k-memory Markov variational autoencoder to predict future CSI through temporally structured latent representations. The framework supports direct prediction and residual corrections relative to the latest latent state and noisy CSI observation. Simulation results show that the residual configuration outperforms the evaluated baselines at moderate and high input CSI signal-to-noise ratios, while sequence baselines remain more accurate under stronger noise.
    
[^110]: 面向智能建筑的可持续云-雾-边缘系统：内存约束下的算法自适应与动态编排策略

    Sustainable Cloud-Fog-Edge Systems for Smart Buildings: Algorithmic Adaptation under Memory Constraints and Dynamic Orchestration Strategies

    [https://arxiv.org/abs/2610.13307](https://arxiv.org/abs/2610.13307)

    该论文提出以“建筑本身即数据中心”为核心理念的可持续云-雾-边缘系统，通过低成本Arduino弹性数据采集、内存约束下的轻量级算法自适应与动态编排策略，实现智能建筑中自主、弹性且保护隐私的边缘学习系统。

    

    信息通信技术（ICT）行业不断加剧的环境足迹——全球数据中心用电量到2030年可能达到3%——以及智能建筑对实时响应能力的迫切需求，要求向边缘计算进行根本性转变。本文提出了一种整体的架构与算法方法，用于在智能建筑内构建自主、弹性且保护隐私的学习系统，推进了“建筑本身即数据中心”这一核心理念。与传统的集中式系统相比，我们的贡献展现了原创性，并围绕三大支柱展开：1）使用价格实惠的Arduino生态系统（Uno R4 WiFi/BME280）验证了弹性数据采集链，并提供了安全的持久化解决方案，包括集中式归档（Node-RED/MySQL）和安全远程传输（Python/rsync/SSH）；2）提出了一种轻量级……

    arXiv:2610.13307v1 Announce Type: new  Abstract: The escalating environmental footprint of the Information and Communication Technol- ogy (ICT) sectorwith global Data Center electric- ity usage potentially reaching 3% by 2030 and the imperative for real-time responsiveness in smart buildings demand a foundational shift towards Edge Computing. This article proposes a holis- tic architectural and algorithmic approach for cre- ating autonomous, resilient, and privacy-preserving learning systems within smart buildings, advancing the core concept of the building itself as the data center. Our contributions, demonstrating originality compared to conventional centralized systems, are structured around three main pillars: 1) Validation of resilient data acquisition chains using the af- fordable Arduino ecosystem (Uno R4 WiFi/BME280) and secure persistence solutions including centralized archiving (Node-RED/MySQL) and secure remote transfer (Python/rsync/SSH). 2). The proposal of a lightweight,
    
[^111]: 面向半精度近端策略优化的结构感知数值稳定化方法

    Structure-Aware Numerical Stabilization for Half-Precision Proximal Policy Optimization

    [https://arxiv.org/abs/2610.13306](https://arxiv.org/abs/2610.13306)

    提出PPO-FP16方法，通过全局标量保护、策略变化残差保护和高效持久更新保护三个结构感知模块，解决了FP16直接训练PPO时数值误差反馈累积的问题，在保持原有目标函数和训练设置的同时实现了稳定高效的半精度训练。

    

    arXiv:2610.13306v1 公告类型：新论文。摘要：在资源受限平台上的本地在线训练需要较低的计算开销和训练状态开销。我们研究了半精度近端策略优化（PPO）在商用GPU上的数值稳定性与计算效率。直接将PPO训练转换为FP16是不可靠的，因为数值误差会通过策略更新和后续的数据收集不断反馈累积，而PPO需要反复将当前策略与固定的行为策略进行比较。我们提出了包含三个结构感知模块的PPO-FP16方法：全局标量保护用于稳定在小批量数据或参数更新之间共享的统计量；策略变化残差保护用于提取并保留当前策略与行为策略之间的微小差异；高效持久更新保护用于稳定优化器状态和参数回写，同时减少状态存储。这些改动完整保留了PPO的目标函数、超参数和训练预算。在六个连续……

    arXiv:2610.13306v1 Announce Type: new  Abstract: Local online training on resource-constrained platforms requires low computational and training-state overhead. We study the numerical stability and computational efficiency of half-precision proximal policy optimization (PPO) on commodity GPUs. Directly converting PPO training to FP16 is unreliable because numerical errors feed back through policy updates and subsequent data collection, while PPO repeatedly compares a current policy with a fixed behavior policy. We propose PPO-FP16 with three structure-aware modules. Global scalar protection stabilizes statistics shared across a minibatch or parameter update. Policy-change residual protection extracts and preserves small differences between the current and behavior policies. Efficient persistent-update protection stabilizes optimizer state and parameter writeback while reducing state storage. These changes preserve the PPO objective, hyperparameters, and training budget. Across six cont
    
[^112]: MORAD：基于多目标奖励的RNA逆向设计强化学习

    MORAD: Reinforcement Learning with Multi-Objective Rewards for RNA Inverse Design

    [https://arxiv.org/abs/2610.13305](https://arxiv.org/abs/2610.13305)

    MORAD提出了一个在线强化学习框架，通过联合优化涵盖三级几何、碱基配对、集合行为和序列组成的六项多目标奖励，为骨架条件RNA扩散模型训练单一可复用的共享策略，用于RNA逆向设计。

    

    我们提出了MORAD，这是一个面向骨架条件RNA扩散模型的在线强化学习框架，能够在多种目标骨架上训练单一的共享策略。结构条件的RNA逆向设计旨在寻找与目标三维骨架相兼容的核苷酸序列，然而现有的强化学习方法要么为每个目标骨架单独优化一个策略，要么依赖离线偏好优化，使得可复用的骨架条件模型的在线后训练在很大程度上尚未被探索。MORAD结合了骨架条件组采样、组相对奖励归一化以及前向过程扩散目标。超越现有主要针对三级结构质量的强化学习奖励，MORAD联合优化了涵盖三级几何、碱基配对、集合行为和序列组成的六项测量指标。为了在尺度差异下聚合这些异构目标……

    arXiv:2610.13305v1 Announce Type: new  Abstract: We introduce MORAD, an online reinforcement learning framework for backbone-conditioned RNA diffusion models that trains a single shared policy across diverse target backbones. Structure-conditioned RNA inverse design seeks nucleotide sequences compatible with a target three-dimensional backbone, yet existing reinforcement learning approaches either optimize a separate policy for each target backbone or rely on offline preference optimization, leaving online post-training of a reusable backbone-conditioned model largely unexplored. MORAD combines backbone-conditioned group sampling, group-relative reward normalization, and a forward-process diffusion objective. Beyond existing RL rewards that primarily target tertiary structural quality, MORAD jointly optimizes six measurements spanning tertiary geometry, base pairing, ensemble behavior, and sequence composition. To aggregate these heterogeneous objectives despite differences in scale an
    
[^113]: 具有节点异质重尾扰动的DAG结构学习

    DAG Structure Learning with Node-Varying Heavy-Tailed Disturbances

    [https://arxiv.org/abs/2610.13304](https://arxiv.org/abs/2610.13304)

    提出了节点异质的Student-t DAGMA方法，可联合估计DAG图结构与跨节点变化的重尾扰动参数，在异质重尾噪声下显著提升了图结构恢复的准确性。

    

    DAG（有向无环图）结构可以通过引入保证无环性的平滑约束（如DAGMA的对数行列式函数）从观测数据中学习得到。通常情况下，观测数据与潜在的DAG通过线性结构方程模型相关联，并对每个节点的扰动假设服从高斯分布。如何扩展DAG学习方法以考虑跨节点变化的扰动尺度与尾部行为，仍然是一个挑战。我们提出了节点异质的Student-t DAGMA方法，能够联合估计图结构、节点特定的扰动尺度以及自由度参数。借助高斯尺度混合表示，我们得到了一个迭代重加权的求解形式，可对较大的标准化残差进行降权处理。基于已有结果，我们还建立了图结构与扰动参数的可识别性。合成实验表明，在异质重尾扰动条件下，该方法在结构恢复方面表现更为优异。

    arXiv:2610.13304v1 Announce Type: new  Abstract: DAG structures can be learned from observations by incorporating acyclicity via smooth constraints, such as DAGMA's log-determinant function. Typically, observations and the underlying DAG are related through linear structural equation models, with Gaussianity assumption on the disturbances at each node. Extending DAG learning methods that account for scale and tail behavior of disturbances that vary across nodes remains a challenge. We propose a node-varying Student-$t$ DAGMA that jointly estimates the graph structure, node-specific disturbance scales, and the degrees-of-freedom parameter. A Gaussian scale-mixture representation yields an iteratively reweighted formulation that down-weights large standardized residuals. Building on existing results, we also establish identifiability of the graph and disturbance parameters. Synthetic experiments demonstrate improved structural recovery under heterogeneous heavy-tailed disturbances, and e
    
[^114]: Sonic-AMG：基于空间音频的流式人体动作生成

    Sonic-AMG: Streaming Human Motion Generation from Spatial Audio

    [https://arxiv.org/abs/2610.13300](https://arxiv.org/abs/2610.13300)

    提出了首个从空间音频流式生成人体动作的框架Sonic-AMG，通过将声学内容与身体相对的声源几何信息分别编码为独立标记，由因果变换器生成动作潜变量，并引入排练训练来缓解训练与推理之间的历史不匹配问题。

    

    对空间音频建模人类反应需要同时理解所听到的内容以及声音的来源方位。先前的工作是基于完整的声学与空间输入序列离线生成动作。然而，人类是在不预知接下来会听到什么的情况下对声音做出反应的。我们提出Sonic-AMG，据我们所知，这是首个从空间音频进行流式人体动作生成的框架。该模型将声学内容与身体相对的声源几何信息编码为相互独立的标记。以这些标记为条件，一个带有流（flow）头的因果变换器生成连续的动作潜变量，再由一个因果解码器从中重建动作片段。标准的教师强制训练从真值动作中推导动作历史和身体相对的声源位置，而流式推理则依赖生成的动作来获得二者，从而造成训练与推理之间的不匹配。为减小这种历史差异，我们引入了排练训练，该方法将……（摘要在此处截断）

    arXiv:2610.13300v1 Announce Type: cross  Abstract: Modeling human reactions to spatial audio requires understanding both what is heard and where it comes from. Prior work generates motion offline from complete acoustic and spatial input sequences. Humans, however, react to sounds without knowing what they will hear next. We present Sonic-AMG, to our knowledge the first framework for streaming human motion generation from spatial audio. The model encodes acoustic content and body-relative source geometry as separate tokens. Conditioned on these tokens, a causal transformer with a flow head generates continuous motion latents, from which a causal decoder reconstructs motion segments. Standard teacher-forced training derives motion histories and body-relative source locations from ground-truth motion, whereas streaming inference relies on generated motion for both, resulting in a training-inference mismatch. To reduce this history discrepancy, we introduce rehearsal training, which combin
    
[^115]: 解码寄存器标记以保留全局场景细节

    Decoding Register Tokens Preserves Global Scene Details

    [https://arxiv.org/abs/2610.13299](https://arxiv.org/abs/2610.13299)

    该研究发现，让表示自编码器（RAE）的解码器通过联合自注意力同时处理寄存器标记与全部补丁标记，能够保留光照、模糊、风格等全局场景细节，从而提升图像重建质量。

    

    预训练的视觉Transformer学习到跨越空间补丁标记与非空间标记的结构化图像表示。近期分析表明，非空间的寄存器标记编码了诸如光照、模糊和风格等全局场景属性。然而，基于补丁的图像解码器会丢弃这些输出，使得解码器只能依靠补丁来恢复图像外观。我们研究了在完整空间补丁表示已经可用的情况下，保留寄存器特征能否改善解码效果。我们使用带有冻结的、配备寄存器编码器的表示自编码器（RAEs）来研究这一问题。我们的解码器通过联合自注意力机制，将四个寄存器标记与所有补丁标记一起处理。外观探针、配对重建误差分析以及固定解码器下的寄存器替换实验，检验了哪些全局图像属性可以从寄存器中读取，以及它们的取值如何影响重建的像素。对于生成（摘要在此处截断）

    arXiv:2610.13299v1 Announce Type: cross  Abstract: Pretrained vision transformers learn structured image representations spanning spatial patch tokens and non-spatial tokens. Recent analyses suggest that non-spatial register tokens encode global scene properties such as illumination, blur, and style. Yet patch-based image decoders discard these outputs, leaving the decoder to recover image appearance from patches alone. We investigate whether retaining register features improves decoding when the complete spatial patch representation is already available. We study this question using representation autoencoders (RAEs) with frozen register-equipped encoders. Our decoder processes four register tokens together with all patch tokens through joint self-attention. Appearance probes, paired reconstruction-error analysis, and fixed-decoder register substitutions examine which global image properties are accessible from registers and how their values affect reconstructed pixels. For generation
    
[^116]: 使用拉格朗日神经网络恢复不可约三体相互作用

    Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks

    [https://arxiv.org/abs/2610.13292](https://arxiv.org/abs/2610.13292)

    本文通过拉格朗日神经网络精确学习势能函数，并结合信号感知采样策略，首次证明仅从轨迹数据即可恢复无法用成对项之和表示的不可约三体相互作用（Axilrod-Teller三偶极相互作用）。

    

    拉格朗日神经网络（LNNs）通过参数化一个标量拉格朗日量并经由欧拉-拉格朗日方程推导动力学，从轨迹数据中学习力学系统，使模型偏向遵循力学变分结构的运动。迄今为止所研究的相互作用均为成对可加形式，尚不清楚LNN能否仅凭运动本身恢复不可约的多体相互作用。我们采用Axilrod-Teller三偶极相互作用——即主要的三体色散项——来解决这个问题，该相互作用对三个粒子集体几何构型的依赖无法用任何成对项之和来表示。我们精确规定动能，仅学习势能，将其表示为粒子间距离和角度的函数。结合一种信号感知的采样策略，将训练数据集中在三体信号最强的区域，同时保持对构型空间的广泛覆盖，该方法……

    arXiv:2610.13292v1 Announce Type: new  Abstract: Lagrangian Neural Networks (LNNs) learn mechanical systems from trajectory data by parameterizing a scalar Lagrangian and deriving the dynamics through the Euler-Lagrange equations, biasing the model toward motion that respects the variational structure of mechanics. The interactions studied so far have been pairwise-additive, leaving open whether LNNs can recover an irreducible many-body interaction from motion alone. We address this using the Axilrod-Teller triple-dipole interaction, the leading three-body dispersion term, whose dependence on the collective geometry of three particles cannot be represented by any sum of pairwise terms. We prescribe the kinetic energy exactly and learn only the potential, as a function of interparticle distances and angles. Combined with a signal-aware sampling strategy that concentrates training data where the three-body signal is strongest while preserving broad coverage of configuration space, this r
    
[^117]: 提示词优化器是否会破坏其未被评分的要求？一项预注册研究

    Do Prompt Optimizers Break Requirements They Are Not Scored On? A Preregistered Study

    [https://arxiv.org/abs/2610.13287](https://arxiv.org/abs/2610.13287)

    预注册研究发现，提示词优化器在提升目标指标的同时确实会破坏未被评分的要求（如引用和格式），但事后分析表明，表面的性能退化主要源于推理过程挤占输出空间导致答案被截断而无法提取，而非真正的内容质量下降。

    

    提示词优化器会重写提示词以提升某个目标指标，但提示词中还包含该指标并不评分的要求，例如输出格式、引用来源或拒答行为。在一项预注册研究中，我们在HotpotQA、SQuAD 2.0和IFEval上运行GEPA、MIPROv2和TextGrad，并通过确定性检查逐项统计对未评分要求的违反情况。按照预注册的评判标准，三种优化器均产生了性能退化，在HotpotQA上尤为显著：引用有效性的净退化达到10.7至41.6个百分点，而精确匹配分数则上升了21至34个百分点。然而，事后分解分析改变了这一解读：优化后的提示词会在要求的JSON对象之前引出推理过程，而在全程使用的200个token输出上限下，13.9%至47.0%的回复在任何答案出现之前就已被截断。在可解析的输出中，净退化在零附近3个百分点以内。当token上限提高到1024时，提取失败现象消失，内容退化保持在……

    arXiv:2610.13287v1 Announce Type: new  Abstract: Prompt optimizers rewrite a prompt to raise one target metric, but prompts also state requirements the metric does not score, such as an output format, citations, or abstention. In a preregistered study, we run GEPA, MIPROv2, and TextGrad on HotpotQA, SQuAD 2.0, and IFEval and count per-item violations of unscored requirements with deterministic checks. By the preregistered criterion, all three optimizers produce regressions, substantially on HotpotQA, where net regression in citation validity reaches 10.7 to 41.6 points while exact match rises by 21 to 34 points. A post-hoc decomposition changes this reading. The optimized prompts elicit reasoning before the required JSON object, and under the 200-token output cap used throughout, 13.9% to 47.0% of responses end before any answer appears. Among parseable outputs, net regression is within 3 points of zero. With a 1,024-token cap, extraction failures disappear, content regression stays be
    
[^118]: MHR-Router：面向视觉语言模型推理的多模态异构检索增强自适应路由

    MHR-Router: Multimodal Heterogeneous Retrieval-Enhanced Adaptive Routing for VLM Inference

    [https://arxiv.org/abs/2610.13278](https://arxiv.org/abs/2610.13278)

    提出MHR-Router框架，通过融合冻结的多模态特征构建查询表示，并检索专家模型的成功与失败记忆作为局部证据，实现将多模态查询自适应且准确地路由到最合适的VLM专家模型。

    

    视觉语言模型（VLM）展现出互补的能力，然而由于异构的多模态输入以及各专家模型能力的差异，针对给定的图像-问题查询选择最合适的专家模型仍然具有挑战性。现有的路由方法依赖于基于文本的线索或静态的模型画像，无法捕捉细粒度的视觉证据以及专家模型特定的失败模式。为了解决这一问题，我们提出了MHR-Router，一个面向VLM推理的多模态异构检索增强自适应路由框架。它通过局部检索证据，将每个多模态查询准确分发到最合适的VLM专家模型。MHR-Router由两个阶段组成：（1）多模态路由表示，将冻结的多模态特征融合为全面的查询表示；（2）异构检索增强自适应路由，检索专家模型特定的成功与失败记忆作为局部证据，并采用共享评分模型（原文在此处截断）。

    arXiv:2610.13278v1 Announce Type: new  Abstract: Vision-language models (VLMs) exhibit complementary capabilities, yet selecting the most suitable expert for a given image-question query remains challenging due to heterogeneous multimodal inputs and varying expert strengths. Existing routing approaches rely on text-based cues or static model profiles, failing to capture fine-grained visual evidence and expert-specific failure patterns. To address this, we propose MHR-Router, a multimodal heterogeneous retrieval-enhanced adaptive routing framework for VLM inference. It accurately dispatches each multimodal query to the most suitable VLM expert via local retrieval evidence. MHR-Router consists of two stages: (1) Multimodal Routing Representation, which fuses frozen multimodal features into a comprehensive query representation; and (2) Heterogeneous Retrieval-Enhanced Adaptive Routing, which retrieves expert-specific success and failure memories as local evidence and employs a shared scor
    
[^119]: CopRoute：面向语言模型的依赖感知共形路由

    CopRoute: Dependence-Aware Conformal Routing for Language Models

    [https://arxiv.org/abs/2610.13276](https://arxiv.org/abs/2610.13276)

    CopRoute通过显式建模模型失败之间的依赖结构，按每个模型的边际新增正确回答量对模型排序并进行共形校准，从而在控制全错风险的同时提升语言模型集合路由的覆盖率。

    

    基于集合的语言模型路由旨在提高“至少有一个模型能正确回答查询”这一事件的覆盖率，同时控制推理调用的次数。现有的风险受控路由策略通过校准单一的共形阈值并将其应用于一个嵌套的模型集合族，来约束全错风险（即所选集合中所有模型都失败的概率）。由于每个模型都是被孤立地评分的，这些策略忽略了模型失败之间的依赖结构。共享训练数据或架构的模型往往在相同的查询上失败，因此两个各自很强的模型联合起来可能是冗余的，而一个较弱但互补的模型却能覆盖其他模型错过的查询。为了顾及这种被现有风险受控策略所忽略的依赖性，我们提出了CopRoute，它根据每个模型在已选模型之外还能额外正确回答多少新查询来对模型进行排序，并针对每个查询决定……

    arXiv:2610.13276v1 Announce Type: new  Abstract: Set-based language model routing aims to increase the coverage of the event where at least one model answers a query correctly while controlling the number of inference calls made. Existing risk-controlled routing policies bound the all-miss risk (the probability that every model in the selected set fails) by calibrating a single conformal threshold and applying it to a nested family of model sets. Because each model is scored in isolation, these policies ignore the dependence structure of model failures. Models that share training data or architecture tend to fail on the same queries, so two individually strong models can be jointly redundant while a weaker but complementary model covers the queries that the others miss. To account for this dependence, which existing risk-controlled policies ignore, we propose CopRoute, which orders models by how many new queries each one answers correctly beyond those already selected, decides per quer
    
[^120]: CV-OPD：对比视觉在策略蒸馏

    CV-OPD: Contrastive Vision On-policy Distillation

    [https://arxiv.org/abs/2610.13264](https://arxiv.org/abs/2610.13264)

    该论文发现传统在策略蒸馏会让学生视觉语言模型走捷径、未能真正依赖视觉证据（即使视觉区域被遮挡仍能答对问题），并提出CV-OPD方法，通过在相同学生前缀条件下对比正负视觉裁剪所诱导的教师分布，并引入基于间隔的对比约束，迫使学生模型真正利用视觉信息进行推理。

    

    在策略蒸馏在视觉语言模型后训练中展现出多功能性与便利性，能够为学生模型提供来自教师模型的密集监督。然而，我们观察到一种异常现象：蒸馏得到的视觉语言模型即使在相关视觉区域被遮蔽或替换为噪声的情况下，仍然能够准确回答问题。基于这一观察，我们的定量与定性实验表明，以答案作为提示的在策略蒸馏未能教会学生模型真正地寻找并依赖视觉证据。我们的理论分析指出，这种失败源于蒸馏过程中缺乏视觉约束而产生的捷径。为解决该问题，我们提出了对比视觉在策略蒸馏，该方法在以相同的学生前缀为条件的情况下，比较由正样本裁剪与负样本裁剪所诱导的教师分布，并通过基于间隔的对比约束来增强优化，从而（摘要在此处截断）

    arXiv:2610.13264v1 Announce Type: cross  Abstract: On-policy distillation (OPD) reveals versatility and convenience during VLM post-training, providing dense supervision for student from teacher. However, we observed an anomaly that the distilled VLM could still answer the questions accurately even when the related visual regions were masked or replaced with noise. Building on this observation, our quantitative and qualitative experiments demonstrate that OPD, which uses the answer as a hint, has failed to teach students to genuinely seek out and rely on visual evidence. Our theoretical analysis suggests that this failure arises from a shortcut enabled by the lack of visual constraints during distillation. To address the issue, we introduce Contrastive Vision On-policy Distillation (CV-OPD), which compares teacher distributions induced by positive and negative crops while conditioning on the same student prefix and augments optimization with a margin-based contrastive constraint, which
    
[^121]: 面向Transformer跨头注意力的数据自适应马氏度量学习

    Data-Adaptive Mahalanobis Metric Learning for Cross-Head Attention in Transformers

    [https://arxiv.org/abs/2610.13261](https://arxiv.org/abs/2610.13261)

    本文提出DAMCHA注意力机制，通过学习完整的、依赖输入的跨头马氏度量矩阵来泛化多头注意力，显式捕捉头间交互，并使注意力几何结构随数据动态自适应。

    

    本文提出了数据自适应马氏跨头注意力（DAMCHA），这是一种简单而有效的注意力机制，从度量学习的视角对多头注意力（MHA）进行了泛化。标准MHA隐式地学习一种静态的、逐头的欧氏相似度，其对应于统一跨头度量矩阵的对角块，从而丢弃了所有跨头的非对角交互。DAMCHA通过两项关键创新解决了这一局限：首先，它学习完整的跨头度量矩阵，诱导出一种类似马氏距离的相似度，显式地捕捉超出MHA受限块对角结构之外的头间交互；其次，它将度量矩阵参数化为依赖输入的函数，使注意力几何结构能够动态适应数据的内在流形结构，而不是在训练结束后保持固定。此外，作者还引入了有原则的正则化策略（摘要在此处截断）。

    arXiv:2610.13261v1 Announce Type: new  Abstract: In this paper, we propose Data-Adaptive Mahalanobis Cross-Head Attention (DAMCHA), a simple and effective attention mechanism that generalizes Multi-Head Attention (MHA) through the lens of metric learning. Standard MHA implicitly learns a static, head-wise Euclidean similarity corresponding to the diagonal blocks of a unified cross-head metric matrix, discarding all off-diagonal interactions across heads. DAMCHA addresses this limitation via two key innovations. First, it learns the full cross-head metric matrix, inducing a Mahalanobis-like similarity that explicitly captures inter-head interactions beyond the restricted block-diagonal structure of MHA. Second, it parameterizes the metric matrix as an input-dependent function, enabling the attention geometry to dynamically adapt to the intrinsic manifold structure of the data rather than remaining fixed after training. We further introduce principled regularization strategies, most nota
    
[^122]: 非均匀介质中超声波传播的神经代理模型

    Neural Surrogate Modeling of Ultrasonic Wave Propagation in Heterogeneous Media

    [https://arxiv.org/abs/2610.13258](https://arxiv.org/abs/2610.13258)

    提出了一种将傅里叶神经算子与局部积分重采样融入U-Net架构的新型神经代理模型，能够在有限数据下高效、鲁棒地模拟非均匀介质中的高频超声波传播，并适用于不同空间尺度。

    

    神经代理模型为波传播提供了计算高效的自回归建模，可实现诸如快速逆问题求解等应用。尽管近来已有进展，但现有的代理模型仍无法满足超声传感的需求，而超声传感的特点正是高频和非均匀的传播介质。我们展示了基于傅里叶神经算子的方法如何无法捕捉重要的信号特征（如反射），而基于U-Net的方法虽然能很好地学习波场，却缺乏在不同空间尺度之间传递信息的能力。随后，我们提出了一种新的代理模型，通过将FNO与局部积分重采样整合到U-Net架构中，兼取两者之长。我们证明该模型可以在有限数据上完成训练，其预测对输入噪声具有鲁棒性，并能在不同空间尺度上有效工作。

    arXiv:2610.13258v1 Announce Type: new  Abstract: Neural surrogates provide computationally efficient auto-regressive modeling of wave propagation, enabling e.g. fast inverse solutions. Despite recent progress, the current surrogates are not yet sufficient for the needs of ultrasonic sensing, characterized by high frequency and heterogeneous propagation media. We show how methods building on Fourier Neural Operators (FNO) fail to capture essential signal features, such as reflections, whereas U-Net -based methods learn the wave fields well but lack the ability to transfer information across different spatial scales. We then propose a new surrogate that captures the best of the two worlds, by integrating FNOs and local-integral resampling into a U-Net architecture. We show it can be trained on limited data, and that the predictions are robust to input noise and work across spatial scales.
    
[^123]: CAST：面向跨分词器在线策略蒸馏的信用对齐

    CAST: Credit Alignment for Cross-Tokenizer On-Policy Distillation

    [https://arxiv.org/abs/2610.13257](https://arxiv.org/abs/2610.13257)

    CAST提出了一种跨分词器在线蒸馏中的“移动或保持”信用对齐框架，利用预测不确定性和教师信号平滑度将块级优势精确分配到学生令牌上，并对无教师对应的令牌施加锚定约束。

    

    跨分词器在线策略蒸馏（OPD）需要在对教师与学生模型的分数进行比较之前，先对齐二者的序列。基于解码文本的对齐方式可以在块级别提供这种对应关系，但遗留了一个尚未解决的关键问题：信用对齐。块级信号只指定了某个片段应该改变多少，却没有说明该信号应如何在学生的各个令牌之间分配，同时缺乏教师对应关系的令牌则不受任何约束。我们提出了CAST（基于块感知的信用分配与选择性令牌锚定），一个“移动或保持”（shift-or-stay）框架，同时解决上述两个问题。对于已对齐的块，CAST利用预测不确定性和局部教师信号平滑度，将每个块的优势重新分配到学生令牌上，在保留块总优势的同时，强调由局部一致教师反馈所支持的学习机会。所得权重精确求解了一个经KL正则化的信用分配问题。

    arXiv:2610.13257v1 Announce Type: cross  Abstract: Cross-tokenizer on-policy distillation (OPD) requires aligning teacher and student sequences before their scores can be compared. Decoded-text alignment provides such correspondence at the chunk level, but leaves a distinct problem unresolved: credit alignment. A chunk-level signal specifies how much a span should change, but not how that signal should be distributed across student tokens, while tokens without teacher correspondence remain unconstrained. We propose CAST (Chunk-aware credit Assignment with Selective Token Anchoring), a shift-or-stay framework that addresses both issues. For aligned chunks, CAST redistributes each chunk's advantage across student tokens using predictive uncertainty and local teacher-signal smoothness, emphasizing learning opportunities supported by locally consistent teacher feedback while preserving the chunk's total advantage. The resulting weights exactly solve a KL-regularized credit-allocation probl
    
[^124]: 表格基础模型实际上是浅层的

    Tabular Foundation Models Are Effectively Shallow

    [https://arxiv.org/abs/2610.13253](https://arxiv.org/abs/2610.13253)

    表格基础模型（TFMs）实际上是“浅层”的——其 transformer 块中高达 96% 可以被一个闭式、无需标签的仿射转换器替换而不损失性能，说明这类模型存在大量架构冗余，可以被大幅简化。

    

    基于上下文学习的表格基础模型（TFMs）特别适用于资源受限或存在隐私约束的场景，因为它们无需额外训练即可解决新任务。然而，其架构由多个 transformer 块组成，导致高昂的计算和内存开销。与此同时，梯度提升决策树（GBDTs）等传统替代方法仍极具竞争力，并且可以在 CPU 上高效运行，这表明表格任务可能本质上并不需要深层架构。这种对比引出了一个根本性问题：能否通过消除架构冗余来简化 TFMs，而不牺牲性能？为了回答这个问题，我们提出了一种闭式、无需标签的仿射转换器，它可以替换连续窗口中高达 96% 的块，而其余部分保持不变。在 171 个分类数据集上对八个前沿 TFMs 进行评估后，我们发现表格 transformer 模型存在严重的参数冗余。

    arXiv:2610.13253v1 Announce Type: cross  Abstract: In-context-learning Tabular Foundation Models (TFMs) are uniquely suited to scenarios with limited resources or privacy constraints, as they solve new tasks without additional training. However, their architecture consists of multiple transformer blocks, resulting in high computational and memory footprints. Meanwhile, traditional alternatives like Gradient-Boosted Decision Trees (GBDTs) remain highly competitive and run efficiently on CPUs, suggesting that tabular tasks may not inherently require deep architectures. This contrast raises a fundamental question: can TFMs be simplified by removing architectural redundancies without sacrificing performance? To answer this, we introduce a closed-form, label-free affine translator that replaces a contiguous window of up to 96% of the blocks, leaving the rest unchanged. Evaluating eight frontier TFMs on 171 classification datasets, we reveal that tabular transformers are vastly overparameter
    
[^125]: State-OPD：面向多轮智能体的状态锚定在线策略蒸馏

    State-OPD: State-Grounded On-Policy Distillation for Multi-Turn Agents

    [https://arxiv.org/abs/2610.13238](https://arxiv.org/abs/2610.13238)

    该论文提出State-OPD，通过交互状态将学生推理区分为先验推理与当前推理两种角色——先验推理会扭曲教师监督、当前推理影响同状态下的动作多样性——从而实现更有效的多轮智能体在线策略蒸馏。

    

    在线策略蒸馏（OPD）已成为将前沿模型的推理能力迁移到更小模型的一种强大范式，但将其成功扩展到多轮智能体环境仍具挑战性。我们通过交互状态来研究智能体OPD，将客观任务和已执行的动作-观察记录与学生推理的内部语义轨迹区分开来。透过这一视角，我们识别出学生推理在多轮OPD中的两个角色：先验推理会扭曲教师监督，而当前推理则塑造在同一状态下所采样的动作备选方案。具体而言，在匹配的交互状态下，移除先验的学生推理可以提高教师续写的成功率并改变教师的token级目标，且在越靠后的轮次中续写收益越大。同时，重采样完整的推理-动作响应比重采样……（原文摘要在此处截断）

    arXiv:2610.13238v1 Announce Type: new  Abstract: On-Policy Distillation (OPD) has emerged as a powerful paradigm for transferring reasoning capabilities from frontier models to smaller models, while extending its success to multi-turn agentic environments remains challenging. We study agentic OPD through interaction states, distinguishing the objective task and executed action-observation record from the internal semantic trajectory of student reasoning. Through this lens, we identify two roles of student reasoning in multi-turn OPD: prior reasoning distorts teacher supervision, while current reasoning shapes the action alternatives sampled at the same state. Specifically, at matched interaction states, removing prior student reasoning improves teacher continuation success and changes the teacher's token-level targets, with larger continuation gains at later turns. Meanwhile, resampling complete reasoning-action responses yields greater action diversity at the same state than resamplin
    
[^126]: 医学影像中联合嵌入预测学习的可预测性偏差

    The Predictability Bias of Joint-Embedding Predictive Learning in Medical Imaging

    [https://arxiv.org/abs/2610.13237](https://arxiv.org/abs/2610.13237)

    该论文提出TAP-JEPA模型并首次揭示了联合嵌入预测学习中的“可预测性偏差”——预测路径对可从上下文推断的解剖结构预测能力保持良好，但对临床诊断至关重要的散发性局灶病理的预测能力显著下降。

    

    arXiv:2610.13237v1 公告类型：交叉 摘要：联合嵌入预测架构（JEPAs）通过从可见上下文预测被掩码区域的教师表示来进行学习。点预测是对条件目标分布的总结，无法再现上下文未能解析的变异性。我们将这一约束称为“可预测性偏差”。在计算机断层扫描（CT）中，重复出现的解剖结构可以从上下文中预测出来，而决定临床解读的散发性、局灶性病理往往无法预测。我们提出了TAP-JEPA，这是一个在113,611个CT体积数据上训练的ViT-B/3D模型，在扫描级分类任务上与最先进的编码器（COLIPRI、Merlin、VoxelFM）具有竞争力。随后，我们通过重放训练期间所使用的精确掩码、教师和预测器计算来审计所学到的预测路径。预测将病理的平均Skill分数（患病率归一化的平均精度）从0.388降低至0.271，而解剖结构的Skill分数仅从0.921降至0.873。沿着（原文摘要在此处截断）

    arXiv:2610.13237v1 Announce Type: cross  Abstract: Joint-embedding predictive architectures (JEPAs) learn by predicting teacher representations of masked regions from visible context. Point prediction summarizes the conditional target distribution and cannot reproduce variation that the context does not resolve. We call this constraint predictability bias. In computed tomography (CT), recurrent anatomy is predictable from context, whereas the sporadic, localized pathology that determines clinical interpretation often is not. We introduce TAP-JEPA, a ViT-B/3D trained on 113,611 CT volumes that is competitive with state-of-the-art encoders (COLIPRI, Merlin, VoxelFM) in scan-level classification. We then audit the learned predictive pathway by replaying the exact masking, teacher, and predictor computations used during training. Prediction lowers the mean Skill (prevalence-normalized average precision) from 0.388 to 0.271 for pathology, but only from 0.921 to 0.873 for anatomy. Along the 
    
[^127]: 基于拉格朗日神经网络的史瓦西测地线运动学习

    Learning Schwarzschild Geodesic Motion with Lagrangian Neural Networks

    [https://arxiv.org/abs/2610.13234](https://arxiv.org/abs/2610.13234)

    本文提出解耦的位置-速度归一化方案与能量约束的对数均匀采样数据生成策略两项协同改进，攻克了拉格朗日神经网络学习史瓦西测地线运动这一广义相对论领域的开放性挑战。

    

    拉格朗日神经网络（LNN）通过参数化一个标量拉格朗日量，并利用欧拉-拉格朗日方程推导加速度，从轨迹数据中学习动力学，从而提供了一种变分归纳偏置，相比纯数据驱动模型提升了物理一致性。最近，LNN已被扩展到广义相对论的场景中，而史瓦西测地线仍是一个尚未解决的开放挑战。在复杂相对论系统上进行训练受到以下因素的阻碍：动力学的多样性、数值不稳定性，以及难以充分采样物理上相关且多样化的轨道区域。我们引入了两项协同改进：一是解耦的位置-速度归一化方案，该方案独立地重新缩放每个坐标及其时间导数，并对加速度目标进行重新加权，以平衡所有通道的贡献；二是一种能量约束的对数均匀采样数据集生成策略，该策略基于内（摘要在此处被截断，内容不完整）

    arXiv:2610.13234v1 Announce Type: new  Abstract: Lagrangian Neural Networks (LNNs) learn dynamics from trajectory data by parameterizing a scalar Lagrangian and deriving accelerations through the Euler--Lagrange equation, providing a variational inductive bias that improves physical consistency over purely data-driven models. LNNs have recently been extended to general relativistic settings, leaving Schwarzschild geodesics as an open challenge. Training on complex relativistic systems is hindered by the diversity of the dynamics, as well as numerical instabilities and the difficulty of adequately sampling physically relevant, diverse orbital regimes. We introduce two coordinated improvements: a decoupled position--velocity normalization scheme that independently rescales each coordinate and its time derivatives, reweighting acceleration targets to balance contributions across all channels; and an energy-constrained log-uniformly sampled dataset generation strategy informed by the inner
    
[^128]: ReSFlow：在统计流形上校正类别流以实现一步生成

    ReSFlow: Rectifying Categorical Flows on Statistical Manifolds for One-Step Generation

    [https://arxiv.org/abs/2610.13225](https://arxiv.org/abs/2610.13225)

    提出ReSFlow模型，通过在统计流形上校正类别流使其轨迹逼近测地线，实现离散数据的一步生成，大幅降低推理计算成本。

    

    在统计流形上学习连续向量场，将简单先验分布传输至目标离散分布，极大地推动了离散数据的生成建模。然而，由于这些连续时间流不受沿统计流形测地线运动的约束，因此需要进行多步数值积分来模拟，导致推理阶段产生高昂的计算成本。在本工作中，受连续欧几里得空间中校正流“直而快”特性的启发，我们提出了校正统计流模型，该模型通过显式学习遵循内蕴黎曼几何的测地线轨迹，实现了统计流形上的一步类别生成。通过在统计流形上校正流，我们从理论上保证了先验分布与目标分布之间的轨迹逼近测地线。

    arXiv:2610.13225v1 Announce Type: cross  Abstract: Generative modeling for discrete data has been significantly advanced by learning continuous vector fields on the statistical manifolds to transport simple priors to target discrete distributions. However, because these continuous-time flows are not constrained to follow geodesics on the statistical manifold, thereby requiring multi-step numerical integration for simulation and leading to substantial computational costs during inference. In this work, drawing inspiration from the "straight and fast" properties of Rectified Flow in continuous Euclidean space, we propose the Rectified Statistical Flow (ReSFlow) model, which enables one-step categorical generation on statistical manifolds by explicitly learning geodesic trajectories that adhere to the intrinsic Riemannian geometry. By rectifying the flow on the statistical manifold, we theoretically guarantee that trajectories between prior and target distributions approximate geodesics. 
    
[^129]: 基于张量的多模态数据融合用于预测癌症风险等级

    Tensor-based Fusion of Multi-modal Data for Predicting Cancer Risk Levels

    [https://arxiv.org/abs/2610.13221](https://arxiv.org/abs/2610.13221)

    本文提出一种集成特征提取、张量分解与聚类的无监督机器学习框架，通过融合异构的基因组与临床多模态数据提取具有生物学意义的潜在特征，从而提升癌症风险等级预测的准确性。

    

    arXiv:2610.13221v1 公告类型：新 摘要：癌症的诊断与治疗日益依赖于生成高维数据集的多模态诊断技术。尽管这些数据集蕴含丰富的病理信息，但它们在解释、降维和临床适用性方面带来了相当大的挑战。传统的机器学习方法往往无法提取有意义的跨模态特征，从而限制了患者风险分层中的预测准确性。为了解决这些局限性，本文提出了一种无监督机器学习框架，该框架集成了特征提取、张量分解和聚类，以改进癌症风险预测。该框架采用张量分解来融合异构的基因组与临床数据，从而能够提取出优于单模态特征集的、具有生物学意义的潜在特征。本文研究了多种特征提取策略，所得特征……

    arXiv:2610.13221v1 Announce Type: new  Abstract: Cancer diagnosis and treatment increasingly rely on multi-modal diagnostic technologies that generate high-dimensional datasets. Although these datasets are rich in pathological information, they pose considerable challenges in interpretation, dimensionality reduction, and clinical applicability. Conventional machine learning approaches often fail to extract meaningful cross-modal features, thereby limiting predictive accuracy in patient risk stratification. To address these limitations, this paper proposes an unsupervised machine learning framework that integrates feature extraction, tensor decomposition, and clustering to improve cancer risk prediction. The framework employs tensor decomposition to fuse heterogeneous genomic and clinical data, enabling the extraction of latent, biologically meaningful features that outperform unimodal feature sets. Multiple feature extraction strategies are investigated, and the resulting features are 
    
[^130]: 视频生成的黄金锚点

    Golden Anchor for Video Generation

    [https://arxiv.org/abs/2610.13217](https://arxiv.org/abs/2610.13217)

    该论文揭示了文生图模型直接生成的图像并不适合作为视频生成的锚点，并通过基于图像奖励微调T2I模型来生成结构和角色细节更丰富的“黄金锚点”，从而提升下游视频生成质量。

    

    视频生成支持数字内容创作，其应用涵盖虚拟化身、AI驱动的电影制作、交互式游戏引擎和世界模型。最先进的系统主要通过图生视频（I2V）生成来实现可控性，其中注入的锚点图像决定了生成视频的身份、构图和风格。在现代流程中，该锚点由文生图（T2I）生成器合成。这引出了一个很大程度上被忽视的问题：直接从T2I模型获得的图像是否真正适合当前的视频模型？我们的分析表明，锚点的美学和质量分数与下游视频质量仅有微弱的相关性。而结构-纹理分数在视觉和运动两个维度上均与I2V视频质量呈负相关。相应地，基于图像奖励对T2I模型进行微调，可以产生结构和角色细节明显更丰富的锚点：它们的得分……（摘要在此处被截断）

    arXiv:2610.13217v1 Announce Type: cross  Abstract: Video generation supports digital content creation, with applications in virtual avatars, AI-driven filmmaking, interactive game engines, and world models. State-of-the-art systems achieve controllability primarily through image-to-video (I2V) generation, in which an injected anchor image determines the identity, composition, and style of the resulting video. In modern pipelines, this anchor is synthesized by a text-to-image (T2I) generator. This raises a largely overlooked question: Are images obtained directly from the T2I model truly suitable for current video models? Our analysis shows that anchor aesthetic and quality scores are weakly correlated with downstream video quality. The structure-and-texture score is negatively correlated with I2V video quality in both the visual and motion dimensions. Fine-tuning the T2I model on an image reward accordingly yields anchors with visibly denser structural and character detail: they score 
    
[^131]: Monarch-MoE：面向多模态持续指令微调的结构化专家

    Monarch-MoE: Structured Experts for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2610.13215](https://arxiv.org/abs/2610.13215)

    Monarch-MoE用固定置换连接的块对角因子替代LoRA专家中的稠密因子，通过块级梯度隔离缓解多模态持续指令微调中的灾难性遗忘，同时将适配器参数减少约75%和50%。

    

    在多模态持续指令微调中学习新任务可能导致对已获得知识的灾难性遗忘。尽管多专家LoRA将学习分散到各个专家中，但对稠密低秩专家的更新仍可能影响许多输出特征。我们提出Monarch-MoE，用通过固定置换连接的块对角因子替代LoRA专家中的稠密因子。这种结构提供了块级梯度隔离：单个块对的更新仅影响一个输出块，同时保持所有输入块与输出块之间的连接。为了使这些紧凑的专家在内存受限的硬件上具备实用性，我们结合了批处理计算、融合前向算子和优化的反向传播。在LLaVA-1.5-7B上于MLLM-DCL和UCIT两个数据集对三种持续学习方法的评估中，Monarch-MoE在将适配器参数分别减少约75%和50%的同时，提升了对首个任务的记忆保持能力。

    arXiv:2610.13215v1 Announce Type: new  Abstract: Learning new tasks in multimodal continual instruction tuning can cause catastrophic forgetting of previously acquired knowledge. Although multi-expert LoRA distributes learning across experts, updates to a dense low-rank expert can still affect many output features. We introduce Monarch-MoE, which replaces the dense factors in LoRA experts with block-diagonal factors connected by a fixed permutation. This structure provides blockwise gradient isolation: individual block-pair updates affect only one output block, while connections between all input and output blocks are maintained. To make these compact experts practical on memory-limited hardware, we combine batched computation, fused forward operators, and an optimized backward pass. Across three continual-learning methods evaluated on MLLM-DCL and UCIT with LLaVA-1.5-7B, Monarch-MoE improves first-task retention while reducing adapter parameters by approximately 75% and 50%, respectiv
    
[^132]: 委托式生成量子机器学习中的可验证采样

    Verifiable Sampling for Delegated Generative Quantum Machine Learning

    [https://arxiv.org/abs/2610.13211](https://arxiv.org/abs/2610.13211)

    该论文提出了一种盲化且可验证的采样协议，用于安全委托生成式量子机器学习模型的推理阶段，并扩展了可验证的委托变分量子算法以覆盖训练阶段的委托。

    

    生成式机器学习在众多跨学科任务中展现出卓越成果，鉴于量子生成模型能够生成经典计算机难以处理的分布，大量研究正致力于将其作为实现超越经典性能的路径。然而，将此类计算安全地委托给强大的远程量子服务器这一问题却相对较少受到关注。在本工作中，我们提出了一种盲化且可验证的采样协议，可用于此类量子模型的推理阶段。该协议继承了通用盲量子计算的盲性，并通过利用基于陷阱方案的已知方法实现可验证性。同时，为了委托训练阶段，我们讨论了对先前提出的可验证端到端委托变分量子算法（Inajetovic等人，Physical Review Research, 2026）的扩展。除了通常的接受……

    arXiv:2610.13211v1 Announce Type: cross  Abstract: Generative machine learning has shown remarkable results across a wide range of interdisciplinary tasks, motivating significant efforts toward Quantum Generative Models as a path to beyond-classical performance, given their capability of generating classically intractable distributions. However, comparatively little attention has been paid to securely delegating such computations to a powerful remote quantum server. In this work, we propose a blind and verifiable protocol for sampling that can be used for the inference phase of such quantum models. Such a protocol inherits blindness from Universal Blind Quantum Computing and achieves verifiability by exploiting known approaches based on trapification schemes. Alongside, to delegate the training phase, we discuss an extension of the previously proposed verifiable end-to-end Delegated Variational Quantum Algorithms (Inajetovic et al., Phys. Rev. Research, 2026).   Beyond the usual Accept
    
[^133]: 每次预测需要多少计算量？两个时间序列基础模型中的结构化块跳过

    How Much Computation Does Every Forecast Need? Structured Block Skipping in Two Time-Series Foundation Models

    [https://arxiv.org/abs/2610.13208](https://arxiv.org/abs/2610.13208)

    该研究表明时间序列基础模型可以在推理时按预测难度有选择地跳过整个 Transformer 块，在几乎不损失预测质量的前提下将推理延迟降低约 11%–13%。

    

    时间序列基础模型对每个输入都使用相同的网络深度，尽管预测窗口在结构和难度上各不相同。我们测试了在两个预训练模型（Chronos-Bolt 和 TimesFM 2.5）中是否可以在推理时移除整个 Transformer 块。我们的旁路方法避免了原始块的计算，并且在所有块都启用时能够完全保留稠密模型的输出。掩码仅在校准窗口上进行选择，并在来自 ETTh1、Traffic、Weather 和 Exchange 数据集的 1,088 个留出窗口上进行评估。在中等稀疏度下，使用 7/11 候选块的 Chronos-Bolt 和使用 16/20 块的 TimesFM 在各数据集上的平均相对预测质量几乎没有变化，同时将 A100 单批次中位延迟分别降低了 11.2% 和 13.4%。更激进的 TimesFM 跳过策略进一步加速了推理，但显著损害了 Traffic 和 Weather 数据集上的表现。在两个模型中，不同的等预算掩码在不同的窗口上表现更优，且回溯性 oracle（摘要在此处截断）

    arXiv:2610.13208v1 Announce Type: new  Abstract: Time-series foundation models use the same network depth for every input, although forecasting windows differ in structure and difficulty. We test whether whole Transformer blocks can be removed at inference time in two pretrained models: Chronos-Bolt and TimesFM 2.5. Our bypasses avoid the original block computation and preserve the dense model exactly when all blocks are enabled. Masks are selected only on calibration windows and evaluated on 1,088 held-out windows from ETTh1, Traffic, Weather, and Exchange. At moderate sparsity, Chronos-Bolt with 7/11 candidate blocks and TimesFM with 16/20 blocks show little change in mean relative forecasting quality across datasets while reducing median A100 batch-1 latency by 11.2% and 13.4%. More aggressive TimesFM skipping accelerates inference further but substantially harms Traffic and Weather. Across both models, different equal-budget masks win on different windows, and a retrospective oracl
    
[^134]: 重新思考在线策略蒸馏中的相对策略差异迁移

    Rethinking Relative Policy Differences Transfer of On-Policy Distillation

    [https://arxiv.org/abs/2610.13205](https://arxiv.org/abs/2610.13205)

    该论文提出了一个统一框架，将专家与参考策略之间的相对策略差异视为 KL 正则化策略优化的隐式奖励，从而突破了教师必须显著强于目标模型的限制，实现不同规模、训练阶段和代际模型之间的任意到任意在线策略蒸馏迁移。

    

    在线策略蒸馏（OPD）利用来自更强教师模型的 token 级监督，在目标策略自身生成的 rollout 上对其进行训练，但通常依赖于教师模型显著强于目标模型这一前提。最近的工作通过迁移专家模型与参考策略之间的相对策略差异、而非专家模型的绝对分布，放宽了这一要求。然而，究竟哪些差异能提供有用的监督信号，以及哪些目标模型能够从中受益，仍不清楚。我们提出了一个统一的视角：代理专家与代理参考模型之间的对数概率比率定义了 KL 正则化策略优化的一种隐式奖励。这一表述为目标策略带来了在线策略更新，将传统 OPD 恢复为特例，并实现了在不同规模、训练阶段或代际的模型之间的任意到任意迁移。我们的系统性实验得出了三个发现。首先，更强的专家不再……（摘要原文在此处截断）

    arXiv:2610.13205v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains a target policy on its own rollouts using token-level supervision from a stronger teacher, but typically relies on the teacher being substantially more capable than the target. Recent work relaxes this requirement by transferring the relative policy difference between an expert and a reference policy rather than the expert's absolute distribution. Yet it remains unclear which differences provide useful supervision and which targets can benefit from them. We develop a unified view in which the log-probability ratio between a proxy expert and a proxy reference defines an implicit reward for KL-regularized policy optimization. This formulation induces an on-policy update for the target, recovers conventional OPD as a special case, and enables any-to-any transfer across models that differ in scale, training stage or generation. Our systematic experiments yield three findings. First, a stronger expert is no
    
[^135]: S$^3$R：基于自稀疏推测解码的快速且忠实的强化学习轨迹生成

    S$^3$R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding

    [https://arxiv.org/abs/2610.13202](https://arxiv.org/abs/2610.13202)

    S³R提出自稀疏推测解码方法，用稀疏注意力起草token、全注意力验证并采用精确推测采样，在严格保持稠密策略采样分布（保证训练忠实性）的同时大幅加速长上下文RLVR的rollout生成。

    

    针对带可验证奖励的长上下文强化学习（RLVR），快速生成rollout需要在不改变用于策略优化的响应组的前提下减少KV缓存流量。在我们分析的RLVR工作负载中，注意力计算占自回归解码内核时间的72–76%。稀疏注意力可以减少这些读取，但直接从稀疏策略中采样会改变用于学习的响应组。对于组相对优化，我们证明即使采用精确的响应级重加权，通常也无法恢复期望的稠密策略更新，因为每个响应的优势取决于组内其他成员的奖励。我们提出了自稀疏推测轨迹生成（S$^3$R）：它使用稀疏注意力起草token，然后通过全注意力验证和精确推测采样生成响应。起草与验证共享同一策略参数，从而无需训练或更新单独的……

    arXiv:2610.13202v1 Announce Type: cross  Abstract: Fast rollout generation for long-context reinforcement learning with verifiable rewards (RLVR) requires reducing KV-cache traffic without changing the response groups used for policy optimization. In our profiled RLVR workloads, attention accounts for 72--76\% of autoregressive decoding kernel time. Sparse attention reduces these reads, but sampling directly from a sparse policy changes the response groups used for learning. For group-relative optimization, we show that even exact response-wise reweighting does not generally recover the expected dense-policy update because the advantage of each response depends on the rewards of other group members. We propose Self-Sparse Speculative Rollouts (S$^3$R). It uses sparse attention to propose tokens, then generates responses through full-attention verification and exact speculative sampling. Drafting and verification share the same policy parameter, eliminating the need to train or update a
    
[^136]: 在部分可观测强化学习中，学习循环网络真的重要吗？

    Does Learning Recurrent Networks Matter in Partially Observable RL?

    [https://arxiv.org/abs/2610.13197](https://arxiv.org/abs/2610.13197)

    该研究发现冻结（无需训练）的循环网络即可为部分可观测强化学习提供数百步的记忆能力，并在多样化的POMDP任务中取得与训练循环网络相当的性能，从而质疑了在部分可观测RL中学习循环网络的必要性。

    

    部分可观测条件下的强化学习（RL）通常依赖循环网络来编码时间信息，循环模块往往与策略或价值函数联合训练。这引出了一个自然的问题：学习循环网络真的有必要吗？在本文中，我们研究了冻结的循环网络能否为求解部分可观测马尔可夫决策过程（POMDP）提供足够的记忆能力。受储备池计算和广义储备池计算理论基础的启发，我们研究了一种将固定循环网络与可训练的非线性输出模块相结合的简单架构。我们在具有不同记忆需求、观测模态和动作空间的各种任务上对该方法进行了评估。令人惊讶的是，冻结的循环网络能够支持数百步的记忆回溯，并在多样化的POMDP任务中取得具有竞争力的控制性能……

    arXiv:2610.13197v1 Announce Type: cross  Abstract: Reinforcement learning (RL) under partial observability typically re- lies on recurrent networks to encode temporal information, with the recurrent module trained jointly with the policy or value function. This raises a natural question: is learning the recurrent network actually necessary? In this paper, we investigate whether a frozen recurrent network can provide sufficient memory for solving par- tially observable Markov decision processes (POMDPs). Motivated by the theoretical foundations of Reservoir Computing (RC) and Generalized Reservoir Computing (GRC), we study a simple archi- tecture that combines a fixed recurrent network with a trainable nonlinear output module. We evaluate this approach across tasks with varying memory demands, observation modalities, and action spaces. Surprisingly, frozen recurrent networks support recall over hundreds of steps and achieve competitive control performance across diverse POMDPs, while us
    
[^137]: 高海况下基于预测信息的无人机在无人水面艇上的着陆窗口规划

    Prediction-Informed Landing Window Planning for UAV Landing on USV in High Sea States

    [https://arxiv.org/abs/2610.13190](https://arxiv.org/abs/2610.13190)

    本文提出一种预测信息驱动的分层控制框架，利用预训练模型显式预测无人艇姿态，从而识别着陆窗口并确定最佳触舰时机，实现高海况下无人机在无人艇上的自主着陆。

    

    在高海况下，无人机（UAV）在无人水面艇（USV）上的自主着陆十分困难。现有的分层着陆方法通常仅从短时观测历史中隐式地推断未来的USV运动，这限制了其适应剧烈且快速变化海况的能力。本文提出了一种预测信息驱动的分层控制框架。该方法开发了一个预训练模型，在高层着陆决策期间显式预测USV的姿态。该预测有助于识别合适的着陆窗口并选择恰当的触舰时机，同时由常规的低层飞行控制器维持无人机的稳定性并跟踪运动指令。为了在真实扰动下评估该框架，研究构建了一个实船-仿真验证平台，将在真实海况下采集的USV运动数据迁移到MuJoCo仿真USV中。实验证明了使用显式预……

    arXiv:2610.13190v1 Announce Type: cross  Abstract: Autonomous landing of unmanned aerial vehicle (UAV) on unmanned surface vehicle (USV) is difficult in high sea states. Existing hierarchical landing methods often infer future USV motion only implicitly from short observation histories, which limits their ability to adapt to severe and rapidly changing sea conditions. This paper presents a prediction-informed hierarchical control framework. A pretrained model is developed to explicitly predict the USV attitude during high-level landing decisions. The prediction helps identify suitable landing windows and choose the right touchdown time, while a conventional low-level flight controller maintains UAV stability and tracks motion commands. To evaluate the framework under realistic disturbances, a real-sim validation setup is constructed in which USV-motion data collected in real sea conditions are transferred to a MuJoCo USV. The experiments demonstrate the feasibility of using explicit at
    
[^138]: WaSA：用于联邦学习中恶意客户端无监督离线检测的小波散射自编码器

    WaSA: Wavelet Scattering Autoencoders for Unsupervised Offline Detection of Malicious Clients in Federated Learning

    [https://arxiv.org/abs/2610.13186](https://arxiv.org/abs/2610.13186)

    提出了完全无监督的离线检测框架WaSA，通过小波散射自编码器对客户端数据频谱摘要进行评分，无需攻击标签即可识别任意类型的联邦学习投毒攻击。

    

    联邦学习（FL）能够在不暴露原始数据的情况下实现协作模型训练，但仍然容易受到注入中毒数据集的恶意客户端的攻击。现有的防御方法通常是被动的（在训练过程中运行）或监督式的（需要预先了解攻击类型）——这些假设在开放式物联网部署中很少成立。我们提出了WaSA，这是一个完全无监督的恶意客户端离线检测框架。在训练之前，每个客户端向服务器传输一份单一的、保护隐私的数据频谱摘要。一个仅使用干净辅助数据训练的去噪自编码器，根据客户端与所学习的良性流形的偏差对其进行评分，无论攻击类型如何都能识别数据损坏。WaSA不需要攻击标签，能够泛化到未见过的异常，并引入了一种用于客户端选择的玻尔兹曼采样策略，可以在硬排除和标准聚合之间平滑插值。

    arXiv:2610.13186v1 Announce Type: new  Abstract: Federated Learning (FL) enables collaborative model training without exposing raw data, yet remains vulnerable to malicious clients injecting poisoned datasets. Existing defenses are typically reactive (operating during training) or supervised (requiring prior knowledge of attack types) - assumptions that rarely hold in open-ended IoT deployments. We propose WaSA, a fully unsupervised, offline framework for malicious client detection. Before training, each client transmits a single, privacy-preserving spectral summary of its data to the server. A denoising autoencoder, trained solely on clean auxiliary data, scores clients based on their deviation from the learned benign manifold, identifying corruptions regardless of the attack type. WaSA requires no attack labels, generalizes to unseen anomalies, and introduces a Boltzmann sampling strategy for client selection that smoothly interpolates between hard exclusion and standard aggregation.
    
[^139]: Neurograf v1：一个端到端的罗马尼亚语优先语言模型家族

    Neurograf v1: An End-to-End Romanian-First Language Model Family

    [https://arxiv.org/abs/2610.13134](https://arxiv.org/abs/2610.13134)

    该论文提出了Neurograf v1——首个从零开始为罗马尼亚语这一低资源欧洲语言构建的端到端语言模型家族，涵盖635 GB语料库、分词器、五个基础模型、指令微调助手和嵌入模型，验证了低资源语言从零训练大模型的可行性。

    

    我们提出了Neurograf v1，这是一个实验性的端到端罗马尼亚语优先语言模型流水线，旨在研究从零开始为低资源欧洲语言训练语言模型的可行性。该流水线包括：RoSfera，一个635 GB的罗马尼亚语语料库，包含1.757亿个文档；一个8万词表规模的字节级BPE分词器；五个参数规模从1.6亿到60亿不等的仅解码器模型，在1310亿个token上训练；三个经过指令微调的助手模型；以及两个通过对比学习训练的嵌入模型。训练和评估均使用ELI-NP/IFIN-HH的计算基础设施完成。该模型家族表现出验证损失随模型规模一致缩放的趋势，且罗马尼亚语事实性和文化知识随模型规模增大而不断提升。在包含1000个问题的WWTBM基准测试中，Neurograf-6B在罗马尼亚语背景问题上达到69.89%的准确率，Neurograf-3.2B达到67.38%。计算最优分析得出了参数缩放指数为（原文此处截断）

    arXiv:2610.13134v1 Announce Type: new  Abstract: We present Neurograf v1, an experimental end-to-end Romanian-first language-model pipeline developed to investigate the feasibility of training language models from scratch for a lower-resource European language. The pipeline comprises RoSfera, a 635 GB Romanian corpus containing 175.7 million documents; an 80k byte-level BPE tokenizer; five decoder-only models ranging from 160M to 6B parameters, trained on 131 billion tokens; three instruction-tuned assistants; and two contrastively trained embedding models. Training and evaluation were conducted using computing infrastructure at ELI-NP/IFIN-HH.   The model family exhibits consistent scaling of validation loss and increasing Romanian factual and cultural knowledge with model size. On the 1,000-question WWTBM benchmark, Neurograf-6B achieves 69.89% accuracy on Romanian-context questions, while Neurograf-3.2B reaches 67.38%. Compute-optimal analysis yields a parameter-scaling exponent of 
    
[^140]: 瞻前顾后：面向路网轨迹的预期性生成建模

    Looking Ahead and Back: Anticipatory Generative Modeling for Road-Network Trajectories

    [https://arxiv.org/abs/2610.13131](https://arxiv.org/abs/2610.13131)

    提出了一种新的图约束生成范式“预期性生成建模”，将路网轨迹生成从位置模拟转向行为建模，通过预期未来路线来解决路口转向选择问题，避免轨迹偏离路网。

    

    轨迹生成是交通管理和城市规划等城市应用的基础。在道路网络中，车辆沿路段的运动在很大程度上是确定性的，而在交叉路口的转向决策则至关重要——多条出路口的路径可能看起来同样合理，但它们却可能通向截然不同的下游路线。现有的生成方法（如扩散模型）仅模拟位置序列，难以完成下一步转向选择，甚至会产生偏离道路网络的轨迹。为填补这一空白，我们主张一个根本性的转变：从位置模拟转向道路网络上的行为建模。在本文中，我们从生成式路径规划的视角重新审视轨迹生成，引入了一种新的图约束生成范式——预期性生成建模，通过结合对未来路线的预期来解决下一步转向选择问题。

    arXiv:2610.13131v1 Announce Type: new  Abstract: Trajectory generation underpins urban applications such as traffic management and urban planning. Within road networks, vehicle movement along road segments is largely deterministic, whereas transitions at intersections are critical--multiple outgoing paths may appear equally plausible, yet they can lead to substantially different downstream routes. Existing generative methods such as diffusion models simulate location sequences and fail short of next transition selection, even producing trajectories that drift off the road network. To fill this gap, we advocate a fundamental shift: moving beyond location simulation toward behavior modeling on the road network. In this paper, we revisit trajectory generation through the lens of generative route planning, and introduce a new graph-constrained generative paradigm, Anticipatory Generative Modeling, which resolves next transition selection by incorporating the Anticipation of how the future 
    
[^141]: 连续动态时间规整的快速梯度近似

    Fast Gradient-Based Approximation of Continuous Dynamic Time Warping

    [https://arxiv.org/abs/2610.13129](https://arxiv.org/abs/2610.13129)

    本文提出了 FastCDTW，一种高效的连续动态时间规整（CDTW）近似求解器，证明了其在温和条件下的可微性，并针对分段线性规整函数提供精确闭式解、针对平滑规整函数提供无偏蒙特卡洛估计两种计算策略，大幅降低了计算成本。

    

    动态时间规整（DTW）是度量时间序列之间差异的常用方法。它通过动态规划在时间戳之间寻找最优规整路径，但这一过程速度较慢且不可微分。因此，已有多种变体被提出。其中，Soft-DTW 提出了 DTW 的平滑形式，使其能够进行梯度计算；而连续动态时间规整（CDTW）则通过可参数化的规整函数引入了连续形式。然而，这两种方法的计算成本都很高，对于现代机器学习流水线中的大规模数据集而言并不实用。本文从多个方面重新审视了 CDTW。在证明了其在温和条件下的可微性之后，我们提出了 FastCDTW——一个高效的近似求解器，具有两种不同的计算策略：针对分段线性规整函数的精确闭式方案，以及针对平滑规整函数的无偏蒙特卡洛估计器……

    arXiv:2610.13129v1 Announce Type: cross  Abstract: Dynamic Time Warping (DTW) is a common choice for measuring the discrepancy between time series. It proceeds by finding an optimal warping between timestamps through dynamic programming, which can be slow and is not differentiable. Consequently, several variants have been proposed. Among them, Soft-DTW proposes a smooth formulation of DTW, amenable to gradient computation, while Continuous-DTW (CDTW) introduces a continuous formulation through parametrizable warping functions. Yet, computational costs remain high for both methods and impractical for large datasets in modern machine learning pipelines. This paper revisits CDTW in several ways. After establishing its differentiability under mild conditions, we introduce FastCDTW, which is an efficient approximate solver with two different computational strategies: an exact closed-form scheme for piecewise-linear warping functions, and an unbiased Monte Carlo estimator for smooth ones, su
    
[^142]: 为什么重尾权重谱能够预测模型质量

    Why Heavy-Tailed Weight Spectra Predict Model Quality

    [https://arxiv.org/abs/2610.13123](https://arxiv.org/abs/2610.13123)

    该论文通过概率框架推导出仅基于DNN最后一层权重的边际似然，首次从统计学角度解释了为什么训练权重矩阵中的重尾谱分布能够有效预测模型质量。

    

    对深度神经网络（DNN）预测成功的坚实理解仍然难以捉摸。例如，虽然存在许多试图预测DNN模型质量的数据相关指标，但这些指标在大型数据集上的计算成本可能很高，和/或在训练数据集未公开时可能根本无法计算。一个实用且有效的模型质量预测方法涉及分析DNN权重矩阵，因为先前的工作已经证明，训练后权重矩阵中的重尾谱分布能够对模型质量产生很强的预测能力。然而，这类指标缺乏严格的统计学论证。在这项工作中，我们旨在从统计学角度理解这些指标为何表现良好。为此，我们采用概率框架来推导DNN的边际似然，且仅使用最后一层权重进行评估。在大型卷积DNN和大型语言模型（LLM）上，我们发现这种边际似然（原文摘要在此处截断）

    arXiv:2610.13123v1 Announce Type: cross  Abstract: A solid understanding of the predictive success of deep neural networks (DNNs) remains elusive. For example, while many data-dependent metrics exist that seek to predict model quality of DNNs, these metrics may be expensive to compute for large datasets and/or may be impossible to compute when training datasets are not released. A practical and effective predictor of model quality involves analyzing DNN weight matrices, as prior work has established that heavy-tailed spectral distributions in trained weight matrices can be strongly predictive of model quality. However, such metrics lack a rigorous statistical justification. In this work, we aim to understand why these metrics perform well, from a statistical perspective. To do so, we employ a probabilistic framework to derive the marginal likelihood for a DNN, evaluated using only the final layer weights. Across large-scale convolutional DNNs and LLMs, we find that this marginal likeli
    
[^143]: 通过合成自蒸馏对小模型进行持续后训练

    Continually Post-training Small Models via Synthetic Self-distillation

    [https://arxiv.org/abs/2610.13113](https://arxiv.org/abs/2610.13113)

    该论文提出合成自蒸馏方法，通过让略大的同族对齐模型生成同分布的合成示范，并结合离线策略监督与拒绝采样，使缺乏强上下文学习能力的小模型也能在持续后训练中学习新能力的同时保留已有知识。

    

    持续后训练面临获取与保持的双重挑战：模型必须在学习新能力的同时保留已有知识。虽然自蒸馏微调通过使用上下文示例的自我监督缓解了这一问题，但它依赖于强大的上下文学习和指令遵循能力。因此，SDFT 在对低延迟、端侧应用至关重要的小模型上会失效。为了在更小规模上实现持续学习，我们首先找出了失败的根本原因。我们证明了模型获取能力差和遗忘问题与后训练模型的分布偏移呈线性相关，这揭示了同分布学习的必要性。因此，我们提出了合成自蒸馏，通过将同分布重写任务外包给一个略大的同族对齐模型来模仿 SDFT 的做法。结合离线策略监督和拒绝采样，该方法提供了高质量的监督信号。

    arXiv:2610.13113v1 Announce Type: cross  Abstract: Continual post-training faces a dual challenge of acquisition and retention: models must learn new capabilities while preserving prior knowledge. While Self-Distillation Fine-Tuning (SDFT) mitigates this using self-supervision with in-context demonstrations, it relies on strong in-context learning and instruction-following abilities. Consequently, SDFT fails on smaller models critical for low-latency, on-device use. To enable continual learning at smaller scales, we first identify the root cause of failure. We show that poor acquisition and forgetting correlate linearly with the post-trained model's distribution shift, revealing the necessity of in-distribution learning. We thus introduce Synthetic Self-Distillation (SynthSelf), imitating SDFT's in-distribution rewrite by outsourcing the task to a minimally larger, same-family aligner. Coupled with off-policy supervision and rejection sampling, this provides the high-quality supervisio
    
[^144]: 位置重要吗？黑盒因果审计揭示LALM幻觉中模型依赖的音频接地

    Does Location Matter? A Black-Box Causal Audit Reveals Model-Dependent Audio Grounding in LALM Hallucination

    [https://arxiv.org/abs/2610.13112](https://arxiv.org/abs/2610.13112)

    该研究提出一种黑盒因果审计方法，通过CLAP相似度定位并用谱门控移除音频证据区域，发现定位消融在控制时长后稳定优于随机移除，但其幻觉检测效果存在模型依赖性，且先前空白波形对比法的表面优势实为时长混淆所致。

    

    大型音频语言模型（LALMs）经常表现出对象幻觉，即确认并不存在的声音。主流的缓解方法——音频感知解码（AAD）——将真实音频下的logits与全零的“空白”波形进行对比，而此前已有综述指出这种空白波形属于分布外输入。我们引入了一种黑盒因果测试，直接隔离所声称的证据本身：通过CLAP相似度定位某个声明的证据区域，用谱门控方法将其移除，并在三个LALMs上以2x2x2的实验网格设计将其与随机窗口对照进行比较。在Qwen2-Audio-7B上，定位消融的原始AUROC（0.613）低于空白方法（0.721）；但这属于时长混淆，而非定位失败，因为空白方法移除的是整个音频片段，而网格实验仅针对2-4秒的片段。在控制时长一致的条件下，定位方法在两个模型的所有实验单元中均优于随机移除。原始检测价值仍依赖于具体模型，在Audio Flamingo 3上尤为明显。同样的证据可用于纠正解码：对比解码组合……（摘要在此处截断）

    arXiv:2610.13112v1 Announce Type: new  Abstract: Large audio-language models (LALMs) frequently exhibit object hallucination, affirming absent sounds. The dominant mitigation, Audio-Aware Decoding (AAD), contrasts logits under real audio against an all-zero "blank" waveform, flagged by prior review as out-of-distribution. We introduce a black-box causal test that isolates the claimed evidence itself: localize a claim's evidence region via CLAP similarity, remove it by spectral gating, and compare against a random-window control across a 2x2x2 grid on three LALMs. On Qwen2-Audio-7B, localized ablation's raw AUROC trails blank's (0.613 vs. 0.721); this is a duration confound, not a localization failure, since blank removes the entire clip versus the grid's 2-4 s. Holding duration fixed, localization beats random removal at every cell on both models. Raw detection value remains model-dependent, especially on Audio Flamingo 3. The same evidence corrects decoding: contrastive decoding combi
    
[^145]: 通过迭代精化扩展RNA基础模型

    Scaling RNA Foundation Models through Iterative Refinement

    [https://arxiv.org/abs/2610.13106](https://arxiv.org/abs/2610.13106)

    Eco-RNA是一个8000万参数的RNA基础模型，通过学习式迭代精化使推理深度可调，在固定参数下增加迭代次数即可显著提升RNA适应性预测性能。

    

    RNA中的生物功能在多个尺度上进行编码，从局部序列基序到长程结构依赖关系。传统的RNA基础模型在固定架构中将参数容量与推理深度耦合在一起，为每次推理规定固定的计算图。在此，我们提出了Eco-RNA，一个8000万参数的RNA基础模型，它通过学习式迭代精化使推理深度可调。其共享的、输入条件化的Transformer核心通过反复应用来精化序列表示。迭代次数在训练完成后选定，并在模型参数固定的情况下控制有效深度。我们在不同循环深度上进行训练，采用掩码重建、状态正则化，以及一种将更深重建锚定于浅层参考的单侧损失。在包含70项测定的RNAGym适应性基准测试中，将精化迭代从一次增加到四次，使总体绝对Spearman相关系数从0.242得到提升（摘要此处原文截断）。

    arXiv:2610.13106v1 Announce Type: new  Abstract: Biological function in RNA is encoded across scales, from local sequence motifs to long-range structural dependencies. Conventional RNA foundation models couple parameter capacity and inference depth within a fixed architecture, prescribing a fixed computation graph for each inference. Here we present Eco-RNA, an 80M-parameter RNA foundation model that makes inference depth adjustable through learned iterative refinement. Its shared, input-conditioned Transformer core refines sequence representations through repeated application. The iteration count is selected after training and controls effective depth at fixed model parameters. We train across recurrent depths with masked reconstruction, state regularization, and a one-sided loss that anchors deeper reconstruction to a shallow reference. On the 70-assay RNAGym fitness benchmark, increasing refinement from one to four iterations raises aggregate absolute Spearman correlation from 0.242
    
[^146]: 失败轨迹能否揭示智能体在何处出错？无需步骤级标签的智能体失败归因

    Can Failed Trajectories Reveal Where Agents Go Wrong? Agentic Failure Attribution Without Step-Level Labels

    [https://arxiv.org/abs/2610.13105](https://arxiv.org/abs/2610.13105)

    提出 SOAP 框架，通过谱评分与注意力引导传播，仅从无步骤标签的失败轨迹中定位 LLM 智能体导致失败的关键步骤，无需人工标注或成功轨迹参考集。

    

    arXiv:2610.13105v1 公告类型：新 摘要：LLM 智能体的失败归因旨在识别导致任务失败的关键步骤，这对于调试和改进智能体系统至关重要。现有方法依赖于成本高昂的 LLM 提示、步骤级错误标注，或缺乏失败信号辅助的已验证成功轨迹。本文研究如何从无步骤标签的失败轨迹中进行失败归因，其中每条轨迹已知已失败，但其关键步骤未知。我们提出了 SOAP（基于注意力引导传播的谱评分，Spectral scOring with Attention-guided Propagation），这是一个有效的框架，既不需要步骤级训练标签，也不需要单独的仅成功参考集。SOAP 首先从 LLM 表示中构建谱参考空间，并根据每一步的投影行为为其分配基础错误分数。重要的是，后续步骤可能表现出更强的失败信号，因为它们继承了较早的错误，这会导致独立评分器失效……

    arXiv:2610.13105v1 Announce Type: new  Abstract: Failure attribution for LLM agents aims to identify decisive steps responsible for task failure, which is critical for debugging and improving agentic systems. Existing methods rely on costly LLM prompting, step-level error annotations, or verified successful trajectories that lack failure signals for assistance. In this paper, we study failure attribution from step-unlabeled failure trajectories, where each trajectory is known to have failed but its decisive step is unknown. We propose SOAP (Spectral scOring with Attention-guided Propagation), an effective framework requiring neither step-level training labels nor a separate success-only reference set. SOAP first constructs a spectral reference space from LLM representations and assigns each step a base error score according to its projection behavior. Importantly, later steps may exhibit stronger failure signals because they inherit an earlier mistake, causing independent scorers to fa
    
[^147]: 针对分布式模型提取的查询监控的局限性

    The Limits of Query Monitoring against Distributed Model Extraction

    [https://arxiv.org/abs/2610.13083](https://arxiv.org/abs/2610.13083)

    该研究基于百万条真实对话数据表明，仅凭查询监控检测分布式模型提取的能力从根本上受普通用户协同行为所限，只有当提取活动比同一周内最独特的正常用户群体更加独特时才能被发现。

    

    模型提供商如今面临着通过数千个账户开展的蒸馏攻击活动，其中每个账户看起来都像是一名普通用户。当网络与支付元数据无法将这些账户关联起来时，剩下的唯一证据就是它们的查询内容。我们研究了提供商仅凭这些查询能够获知什么、不能获知什么：我们使用一百万条真实对话轮次进行方法开发，并采用之后17周内任何设计决策都未曾接触过的数据进行单次冻结评估。研究结论表明，检测结果更多取决于良性的用户协同行为，而非提取活动本身的特殊性：只有当提取活动比同一周内最独特的普通用户群体更为独特时，它才是可被发现的。我们对这一结论给出了精确刻画，证明提取活动与普通用户之间微小但为正的差异所带来的是检测时间而非可检测性的提升，并识别出了最难检测的提取活动。在实证方面，经过校准的监控器能够捕获逐账户监控器所遗漏的朴素分布式提取（0.86-1.00 ...

    arXiv:2610.13083v1 Announce Type: cross  Abstract: Model providers now face distillation campaigns run through thousands of accounts, each of which looks like an ordinary user. When network and payment metadata cannot link those accounts, the only evidence left is what they ask. We study what a provider can and cannot learn from those queries alone, using one million real chat turns for development and 17 later weeks that no design choice touched for a single frozen evaluation. The answer is set less by anything specific to extraction than by benign coordination: a campaign is visible only if it is more distinctive than the most distinctive group of ordinary users in the same week. We make this precise, show that a small but positive difference between campaign and users buys a detection time rather than detectability, and identify the campaign that is hardest to detect. Empirically, a calibrated monitor catches naive distributed extraction that per-account monitors miss (0.86-1.00 of 
    
[^148]: 协作设计会议中的人类与大型语言模型引导：一项计算行为比较

    Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison

    [https://arxiv.org/abs/2610.13082](https://arxiv.org/abs/2610.13082)

    该研究基于AMI会议语料库199个配对情境，从五个维度对人类项目经理与Llama 3.1 8B在协作设计会议中的引导行为进行计算比较，发现人类引导在情境语义距离和共情/缓和语词汇密度上更高，且后者主要由缓和语词汇驱动。

    

    大型语言模型（LLM）正日益被探索用作群体讨论与协作工作的引导者，但LLM生成的引导与人类引导之间的行为差异仍未得到充分刻画。我们基于AMI会议语料库中的199个配对情境，将基于情境的协作设计会议中人类项目经理的引导话语与Llama 3.1 8B Instruct的回应进行了比较。分析涵盖五个验证性维度：情境语义距离、词汇详述代理指标、共情与缓和语词汇密度、话题分歧度以及阶段-词汇对齐度。一项探索性的指令性度量未能通过领域验证，因此不予解读。抽样的人类项目经理话语在情境语义距离（r = 0.644）和共情与缓和语词汇密度（r = 0.203）上得分更高。第二项对比主要由缓和语词汇驱动，因此我们将其解读为词汇层面的缓和行为而非……（摘要原文在此处截断）

    arXiv:2610.13082v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly being explored as facilitators of group discussion and collaborative work, but the behavioral differences between LLM-generated and human facilitation remain insufficiently characterized. We compare human Project Manager facilitation turns from scenario-based collaborative design meetings with Llama 3.1 8B Instruct responses across 199 paired contexts from the AMI Meeting Corpus. The analysis covers five confirmatory dimensions: contextual semantic distance, a lexical elaboration proxy, empathyand-hedge lexical density, topic divergence, and phase-vocabulary alignment. An exploratory directiveness measure failed domain validation and is not interpreted. The sampled human Project Manager turns were higher on contextual semantic distance (r = 0.644) and on empathyand-hedge lexical density (r = 0.203). The second contrast is carried mainly by hedge terms, so we read it as lexical hedging rather 
    
[^149]: Double-Take：在双向记忆中提升长上下文的容量、绑定与选择性

    Double-Take: Improving Capacity, Binding, and Selectivity In Long-Context Within Bidirectional Memory

    [https://arxiv.org/abs/2610.13073](https://arxiv.org/abs/2610.13073)

    提出 Double-Take 方法，通过多步非固定更新和跨位置机制，修复大块测试时训练中单步写入累积干扰限制容量、以及自关联写入无法形成跨位置绑定这两大基本失败，从而提升长上下文记忆的容量、绑定与选择性。

    

    处理长序列并不能保证远处的信息依然可用。随着上下文的增长，固定大小的记忆必须保留大量关联、跨位置地连接信息，并保存相关的结构，而不被其所观察到的一切内容所淹没。这些是彼此不同的需求：记忆可能存储了更多信息，却没有表示出检索所需的关系；它也可能传递了这些关系，却没有保留其中重要的部分。因此，挑战不仅仅在于扩展上下文长度，而在于构建一个随着上下文增长仍保持可用的记忆。我们在大块测试时训练中研究这一问题，并识别出两种基本失败模式：单步写入会累积干扰，从而限制容量；而自关联写入无法形成跨位置的绑定。我们提出了 Double-Take (DT)，通过多步非固定更新与跨位置机制修复这些失败。（注：原文摘要在此处被截断）

    arXiv:2610.13073v1 Announce Type: new  Abstract: Processing a long sequence does not guarantee that distant information remains usable. As context grows, a fixed-size memory must retain many associations, connect information across positions, and preserve relevant structure without being overwhelmed by everything it observes. These are distinct requirements: a memory may store more information without representing the relationships needed for retrieval, and it may transport those relationships without preserving the ones that matter. Thus, the challenge is not simply extending context length, but constructing a memory that remains usable as context grows. We study this problem in large-chunk test-time training (TTT) and identify two basic failures: single-step writes accumulate interference, limiting capacity, while self-associative writes cannot form cross-position bindings. We introduce Double-Take (DT), which repairs these failures with multi-step unpinned updates and cross-position
    
[^150]: CB-Orchestrator：基于上下文老虎机的LLM智能体自适应工作流优化

    CB-Orchestrator: Adaptive Workflow Optimization for LLM Agents via Contextual Bandits

    [https://arxiv.org/abs/2610.13064](https://arxiv.org/abs/2610.13064)

    提出CB-Orchestrator自适应框架，将工作流生成与选择解耦，通过进化搜索构建多样化候选工作流池，并将工作流选择建模为上下文老虎机问题，实现样本高效学习，兼顾性能与成本效益。

    

    大语言模型（LLM）智能体通过智能体工作流在解决复杂任务方面展现出了卓越的能力。然而，人工设计这些工作流既费时费力，又缺乏适应多样化查询的灵活性。虽然自动化工作流优化是一个有前景的方向，但现有方法在训练阶段的反馈收集过程中往往会产生高昂的API成本。此外，它们无法自适应地复用高质量的工作流，这对性能和成本效益都形成了瓶颈。为了解决这些局限性，我们提出了CB-Orchestrator，这是一个将工作流生成与选择解耦的自适应框架。在第一阶段，我们通过进化搜索构建了一个多样化的候选工作流池。在第二阶段，我们将工作流选择形式化为一个上下文老虎机问题，通过平衡探索与利用实现样本高效的学习，从而显著……

    arXiv:2610.13064v1 Announce Type: new  Abstract: Large Language Model (LLM) agents have demonstrated remarkable capabilities in tackling complex tasks through agentic workflows. However, manually designing these workflows is labor-intensive and lacks flexibility to adapt to diverse queries. While automated workflow optimization is a promising direction, existing methods often incur expensive API costs during training-phase feedback collection. Furthermore, they fail to adaptively reuse high-quality workflows, which creates a bottleneck for both performance and cost-effectiveness. To address these limitations, we propose CB-Orchestrator, an adaptive framework that decouples workflow generation from selection. In the first stage, we construct a diverse pool of candidate workflows via evolutionary search. In the second stage, we formulate workflow selection as a Contextual Bandit problem, which enables sample-efficient learning by balancing exploration and exploitation, thereby significan
    
[^151]: CoBe：面向对话式反事实文本编辑的基准测试

    CoBe: A Benchmark For Conversational Counterfactual Text Editing

    [https://arxiv.org/abs/2610.13057](https://arxiv.org/abs/2610.13057)

    该论文提出了CoBe基准，用于评估模型的对话式反事实文本编辑能力，要求模型通过因果理解判断假设性变化会影响的范围，在修改文本的同时保留未受因果影响的事实，而非仅生成统计上相关联的文本。

    

    反事实文本编辑是指根据一个假设性变化来修改给定文本的能力，同时保留未受该变化因果影响的事实。当被问及“如果医生没有推荐手术会发生什么”时，模型输出应保留患者的病史，仅修改该决定所带来的后果。这需要因果理解能力：模型必须在没有被明确指示的情况下，自行判断哪些事件受到假设性变化的影响，哪些内容应保持不变。这项技能是人类推理的核心，尤其在医疗或法律等安全关键领域中至关重要。相比之下，关联性文本生成任务只是生成与改变后的条件在统计上相关联的文本。在上述医疗示例中，关联性响应可能会让患者的症状变得更轻，因为较轻的病例通常与不做手术相关。这样的响应会产生一个看似合理的场景……

    arXiv:2610.13057v1 Announce Type: cross  Abstract: Counterfactual text editing is the ability to revise a given text in light of a hypothetical change while preserving facts not causally affected by that change. When asked what would have happened if a doctor had not recommended surgery, the model output should retain the patient's medical history and modify only the consequences of the decision. This requires causal understanding: a model must determine, without being explicitly instructed to, what events are affected by the hypothetical change and what should remain fixed. This skill is central to human reasoning, especially in safety-critical domains like healthcare or law. By contrast, the task of associational text generation merely generates text statistically associated with the altered condition. In the medical example, an associational response might make the patient's symptoms less severe because milder cases correlate with no surgery. Such responses produce a plausible scena
    
[^152]: 空间重于统计，重训练重于排序：仅标签后门攻击下的训练标签审计

    Space Over Statistic, Retraining Over Ranking: Auditing Training Labels Under Label-Only Backdoors

    [https://arxiv.org/abs/2610.13051](https://arxiv.org/abs/2610.13051)

    该研究通过受控实验证明，防御仅标签后门攻击（如FLIP）的成败关键在于特征空间而非检测方法本身——同一个最近邻检测在冻结的DINOv2自监督特征上AUROC达0.967，而在投毒模型自身特征上仅0.546，并且应以重训练后的效果而非样本排序结果来评判防御方法。

    

    有些后门攻击从不触碰图像本身。FLIP只修改标签：在CIFAR-10上，仅需翻转3%的标签，就能让一个ResNet-32把99.7%携带触发器的源类图像归入攻击者指定的类别。常规防御手段是在用这些（被污染的）标签训练出的模型内部寻找投毒数据。那么，当这些防御失效时，问题究竟出在它们运行的检测方法上，还是它们所处的特征空间？我们通过一项受控交换实验来回答：固定检测方法、标签与评价指标，仅更换特征空间，并通过让受害模型在其保留的数据上重新训练来评判每种防御的优劣。结果证明，特征空间起决定性作用：在dog→cat（狗→猫）攻击上，同一个最近邻检测在投毒模型自身特征上的AUROC仅为0.546，在从未见过任何标签的冻结DINOv2特征上为0.967，而在用干净标签训练的同一网络的特征上达到0.995——后者是预言机式的对照，而非防御方法。我们以90.94/0.97的指标复现了FLIP作者推荐的防御SPECTRE（对比其发表的91.47/1.7），随后对其代价进行衡量：在truck（卡车）……（原文摘要在此处截断）

    arXiv:2610.13051v1 Announce Type: cross  Abstract: Some backdoor attacks never touch an image. FLIP only changes labels: on CIFAR-10, flipping 3% of them makes a ResNet-32 send 99.7% of triggered source-class images to the attacker's class. The usual defenses look for the poison inside a model trained on those same labels. So when they struggle, is the problem the test they run, or the space they run it in?   We answer with a controlled swap: we keep the test, the labels and the metric fixed, change only the feature space, and judge each defense by retraining the victim on what it keeps. The space decides. On dog$\to$cat the same nearest-neighbor check scores AUROC 0.546 on the poisoned model's features, 0.967 on frozen DINOv2 features that never saw a label, and 0.995 on the same network trained on clean labels, an oracle control rather than a defense. We reproduce SPECTRE, the defense FLIP's authors recommend, to 90.94 / 0.97 against its published 91.47 / 1.7, then price it: on truck
    
[^153]: 面向消费设备状态监测的轻量级时频振动感知

    Lightweight Time-Frequency Vibration Sensing for Consumer Device Condition Monitoring

    [https://arxiv.org/abs/2610.13040](https://arxiv.org/abs/2610.13040)

    本文提出一种轻量级AI振动感知框架，通过CWT和HHT将一维振动信号转换为二维时频表示，并借助MobileNetV3-Small等轻量级深度学习模型进行分析，实现消费类旋转设备在真实干扰下的智能状态监测与预测性维护。

    

    包括洗衣机、空调、电风扇、扫地机器人和家用健身器材在内的消费级旋转设备，越来越依赖振动传感来实现智能状态监测与预测性维护。要确保这些资源受限的电器可靠运行，需要高效的轻量级人工智能（AI）解决方案，能够在真实运行干扰下处理非平稳信号。本文提出了一种轻量级的AI辅助振动感知框架，该框架采用连续小波变换（CWT）和希尔伯特-黄变换（HHT）将一维振动时间序列信号转换为二维时频表示，随后利用轻量级深度学习架构对这些表示进行分析，具体包括MobileNetV3-Small、ConvNeXt-Tiny以及You Only Look Once第11版（YOLOv11）分类模型。

    arXiv:2610.13040v1 Announce Type: cross  Abstract: Consumer rotating devices, including washing machines, air conditioners, electric fans, robotic vacuum cleaners, and home fitness equipment, increasingly rely on vibration sensing for intelligent condition monitoring and predictive maintenance. Ensuring reliable operation in these resource-constrained appliances requires efficient, lightweight artificial intelligence (AI) solutions capable of processing non-stationary signals under real-world operational interference. This paper proposes a lightweight AI-assisted vibration sensing framework that employs continuous wavelet transform (CWT) and Hilbert-Huang transform (HHT) to convert one-dimensional (1-D) vibration time-series signals into two-dimensional (2-D) time-frequency representations. These representations are subsequently analyzed using lightweight deep learning architectures, specifically MobileNetV3-Small, ConvNeXt-Tiny, and the You Only Look Once version 11 classification mod
    
[^154]: 迈向更易调优的训练算法：来自AlgoPerf的新结果

    Towards Easier-to-Tune Training Algorithms: New Results from AlgoPerf

    [https://arxiv.org/abs/2610.13036](https://arxiv.org/abs/2610.13036)

    在更新版AlgoPerf V1基准测试的自调优规则下，Schedule-Free AdamW在13个提交算法中排名第一，同时分析表明矩阵预条件和双循环方法虽样本效率优异，但需更激进的计算优化才能在实际运行时间上具竞争力。

    

    评估深度学习优化器是一项具有挑战性的任务，因为超参数调优和优化器-模型协同设计等混杂因素往往会掩盖算法的真实性能。我们展示了更新后的AlgoPerf V1基准测试的结果，在固定硬件上对一组新的训练算法在九个工作负载上的到达目标所需的实际运行时间进行了评估。本报告重点关注该基准测试自调优规则集的结果，该规则要求提交的算法在所有工作负载上使用单一配置。在基于PyTorch和JAX的13个提交算法中，Schedule-Free AdamW排名第一。除了实际运行时间得分外，我们还分析了步数效率与实际运行时间之间的权衡，发现矩阵预条件和双循环方法具有良好的样本效率，但需要更激进的计算优化才能在运行时间上具备竞争力。最后，我们的超参数显著性和保留分析

    arXiv:2610.13036v1 Announce Type: new  Abstract: Evaluating deep learning optimizer is challenging because confounding factors such as hyperparameter tuning and optimizer-model co-design often obscure true algorithmic performance. We present results of the updated AlgoPerf V1 benchmark, in which we evaluate a new set of training algorithms on wall-clock time-to-target across nine workloads on fixed hardware. This report focuses on the results of the benchmark's self-tuning ruleset, which requires submissions to use a single configuration across all workloads. Out of the 13 submissions in PyTorch and JAX, Schedule-Free AdamW ranked first. In addition to the wall-clock scores, we also analyze the trade-offs between step efficiency and wall-clock execution, revealing that matrix-preconditioned and two-loop methods achieve good sample efficiency but require more aggressive computational optimizations to be competitive in runtime. Finally, our hyperparameter saliency and retention analyses 
    
[^155]: 面向轨迹外推的目标约束池化

    Target-Constrained Pooling for Trajectory Extrapolation

    [https://arxiv.org/abs/2610.13034](https://arxiv.org/abs/2610.13034)

    提出目标约束池化（TCP）方法，利用已完成的相关轨迹从两种互补池化信号中估计轨迹终点并修正自由预测，在五个领域的25,245个目标上将平均RMSE降低35.8%至42.1%。

    

    从部分目标轨迹进行轨迹外推是一个欠定问题，但已完成的相关轨迹能提供有关未确定延续部分的信息。我们提出了目标约束池化（TCP），该方法将有界时间范围的终点与延续形态分离，并从两个互补的源信号中估计该终点。均匀池化捕捉最终响应水平，而匹配增量池化在对齐源与目标状态后捕捉剩余变化。TCP选择代表性源，将两个终点估计投影到与目标一致的区域，将其转化为对自由预测的修正，并通过强度校准和退化感知加权来控制其影响。在来自五个领域、涵盖两个预测方向的25,245个目标上，TCP使所有十个仅基于目标的预测器的平均RMSE降低了35.8%至42.1%，并改进了全部100个预测器、数据集与方向的组合。

    arXiv:2610.13034v1 Announce Type: new  Abstract: Trajectory extrapolation from a partial target trajectory is underdetermined, but completed related trajectories provide information about the unresolved continuation. We propose target-constrained pooling (TCP), which separates the finite-horizon endpoint from continuation shape and estimates that endpoint from two complementary source signals. Uniform pooling captures final response levels, while matched-increment pooling captures the remaining change after aligning source and target states. TCP selects representative sources, projects both endpoint estimates onto a target-consistent region, converts them into corrections of the free forecast, and controls their influence through strength calibration and degradation-aware weighting. Across 25,245 targets from five domains and both forecasting directions, TCP reduces mean RMSE by 35.8% to 42.1% for all ten target-only forecasters and improves all 100 forecaster, dataset, and direction c
    
[^156]: 用于可解释算子学习的低秩Kolmogorov-Arnold生成元流

    Low-Rank Kolmogorov-Arnold Generator Flows for Interpretable Operator Learning

    [https://arxiv.org/abs/2610.13028](https://arxiv.org/abs/2610.13028)

    提出KAFlow方法，用由一元Kolmogorov-Arnold函数构成、秩受系数函数个数约束的低秩符号来学习物理算子的生成元，从而将算子学习的代价从随分辨率平方增长降下来，并直接获得可解释的闭合形式。

    

    从数据中学习物理定律的神经算子应当能够泛化到训练分布之外，并以闭合形式呈现所学到的定律。Kolmogorov-Arnold神经算子通过在所有位置和频率上学习一个稠密的伪微分符号（symbol），在这两方面均有所实现，但其计算代价随分辨率的平方增长，且其闭合形式必须依靠人工读出。本文指出这种稠密符号是不必要的：有限阶变系数微分算子的符号，其分离秩由系数函数的个数所界定，与分辨率无关，而该网络在其自身基准测试上学到的符号秩仅为二。作者提出了Kolmogorov-Arnold流（KAFlow），它将物理算子的生成元学习为由一元Kolmogorov-Arnold函数构建的低秩符号，并通过固定这些函数的规范，使得每一项都具有唯一……（摘要原文在此处被截断）

    arXiv:2610.13028v1 Announce Type: new  Abstract: Neural operators that learn physical laws from data should generalize beyond their training distribution and expose the learned law in closed form. The Kolmogorov-Arnold neural operator achieves both for position-dependent dynamics by learning a dense pseudo-differential symbol over all positions and frequencies, at a cost that grows with the square of the resolution and with closed forms that must be read out by hand. This dense symbol is unnecessary: a finite-order variable-coefficient differential operator has a symbol whose separation rank is bounded by the number of its coefficient functions, independently of the resolution, and the symbols that this network learns on its own benchmarks have rank two. We introduce the Kolmogorov-Arnold flow (KAFlow), which learns the generator of a physical operator as a low-rank symbol built from univariate Kolmogorov-Arnold functions, fixes the gauge of these functions so that every term has a uni
    
[^157]: 轨迹感知策略优化：弥合长程大语言模型强化学习中的状态分布差距

    Trajectory-Aware Policy Optimization: Closing the State Distribution Gap in Long-Horizon LLM RL

    [https://arxiv.org/abs/2610.13020](https://arxiv.org/abs/2610.13020)

    提出轨迹感知策略优化（TAPO），通过将理论推导的状态分布修正项融入强化学习目标，并配合几何均值方差缩减与长度感知动态裁剪，弥合长程 LLM 强化学习中的状态分布偏移，解决训练不稳定与崩溃问题。

    

    强化学习（RL）已成为提升大语言模型推理能力的主流范式。然而，长程自回归 rollout 引入了一个根本性限制：局部策略失配会在轨迹中不断累积，演变为严重的状态分布偏移。现有的 LLM 强化学习方法（如 GRPO 和 GSPO）在 token 或序列层面控制策略失配，但忽略了这种由轨迹引起的偏移，导致随着长度的增长出现训练不稳定乃至最终的训练崩溃。为了弥合状态分布差距，我们提出了轨迹感知策略优化（TAPO），这是一个有原则性的框架，从离策略策略梯度中理论推导出状态分布修正项，并将其纳入强化学习目标之中。TAPO 还进一步采用几何均值估计器来降低方差，并结合长度感知的动态裁剪机制，防止更新在整条轨迹上过度偏离。大量实验表……（摘要在此处截断）

    arXiv:2610.13020v1 Announce Type: new  Abstract: Reinforcement learning (RL) has become a dominant paradigm for enhancing LLMs' reasoning capabilities. Yet long-horizon autoregressive rollouts introduce a fundamental limitation: local policy mismatch can compound into severe state distribution shift over trajectories. Existing LLM RL methods, such as GRPO and GSPO, control policy mismatch at the token or sequence level, but overlook this trajectory-induced shift, causing training instability and eventual collapse as the horizon grows. To close the state distribution gap, we propose Trajectory-Aware Policy Optimization (TAPO), a principled framework that theoretically derives the state-distribution correction from the off-policy policy gradient and incorporates it into the RL objective. TAPO further employs a geometric-mean estimator for variance reduction, together with a length-aware dynamic clipping that prevents overly deviated updates across the trajectory. Extensive experiments de
    
[^158]: 广义近端策略优化：从散度视角统一大语言模型强化学习算法

    Generalized Proximal Policy Optimization: Unifying LLM RL Algorithms from the Perspective of Divergence

    [https://arxiv.org/abs/2610.13015](https://arxiv.org/abs/2610.13015)

    该论文提出广义近端策略优化（GPPO），基于带单调性能改进保证的广义 f-散度信赖域理论，从散度几何视角统一理解和设计大语言模型强化学习算法，克服了 PPO 固定截断机制无法自适应策略偏移的缺陷。

    

    强化学习已成为提升大语言模型（LLM）推理能力的基础范式。然而，主流方法近端策略优化（PPO）依赖一个几何上退化的截断代理，无法自适应地调整对策略偏移的抵抗，从而导致过早截断和探索能力退化。后续的各类变体则通过零散且启发式的修改来缓解这些问题。我们提出了广义近端策略优化（GPPO），这是一个有原则的统一框架，通过散度的视角来理解和设计大语言模型强化学习算法。我们建立了具有单调性能改进保证的广义 $f$-散度信赖域理论，并证明了不同的 $f$-散度会诱导出不同的几何结构与策略更新动态。我们进一步引入了新颖的估计器拓扑结构，相比传统的单样本蒙特卡洛方法，能够提供更忠实的散度估计。

    arXiv:2610.13015v1 Announce Type: cross  Abstract: Reinforcement learning has become the foundational paradigm for enhancing the reasoning capabilities of LLMs. Yet the dominant approach, Proximal Policy Optimization (PPO), relies on a geometrically degenerate clipping proxy that cannot adapt its resistance to policy shifts, leading to premature clipping and degraded exploration. Subsequent variants alleviate these issues through fragmented and heuristic modifications. We propose Generalized Proximal Policy Optimization (GPPO), a principled and unified framework for understanding and designing LLM RL algorithms through the lens of divergence. We establish a generalized $f$-divergence trust-region theory with a monotonic performance improvement guarantee, and show that different $f$-divergences induce distinct geometries and policy update dynamics. We further introduce novel estimator topologies that provide more faithful divergence estimates than the conventional single-sample Monte Ca
    
[^159]: 基于Dempster-Shafer融合的证据性多模态Transformer情感识别方法

    Evidential Multimodal Transformers with Dempster-Shafer Fusion for Emotion Recognition

    [https://arxiv.org/abs/2610.13013](https://arxiv.org/abs/2610.13013)

    该论文提出EMT-DS模型，利用Dirichlet分布将预测建模为主观意见，并通过Dempster组合规则融合各模态证据，将模态间的矛盾显式量化为冲突、区分认知不确定性与偶然不确定性，从而避免了Softmax方法在传感器数据损坏或矛盾时仍给出过高置信度的问题，提升了多模态情感识别的可靠性与鲁棒性。

    

    基于Softmax的多模态情感识别方法即使在传感器数据损坏或相互矛盾的情况下（例如面部表情显示积极效价而语音韵律表明消极唤醒时），仍常常给出过高的置信度。我们提出了一种带Dempster-Shafer融合的证据性多模态Transformer（EMT-DS），通过形式化的不确定性量化来解决这一问题。EMT-DS将预测建模为主观意见（Dirichlet分布），从而将认知不确定性（证据缺乏）与偶然不确定性（传感器噪声）分离开来。特定模态的Transformer编码器产生信念质量分布，并通过Dempster组合规则进行融合，该规则使用冲突系数1/(1-c)对联合信念质量进行归一化，从而将矛盾证据显式量化为冲突，使融合后的预测变得弥散，而不是被迫达成缺乏依据的一致结论。

    arXiv:2610.13013v1 Announce Type: new  Abstract: Softmax-based multimodal emotion recognition methods often assign high confidence even when sensor data are corrupted or contradictory, for example when facial expressions suggest positive valence while vocal prosody indicates negative arousal. We present an Evidential Multimodal Transformer with Dempster-Shafer Fusion (EMT-DS) that addresses this problem through formal uncertainty quantification. EMT-DS models predictions as Subjective Opinions (Dirichlet distributions), separating epistemic uncertainty (lack of evidence) from aleatoric uncertainty (sensor noise). Modality-specific transformer encoders produce belief mass distributions that are fused with Dempster's Rule of Combination, which normalizes the joint belief mass by a conflict coefficient 1/(1-c), so that contradictory evidence is explicitly quantified as conflict and the fused prediction becomes diffuse rather than being forced into an unjustified consensus. Under subject-i
    
[^160]: 规模化赋能智能体

    Scaling Empowered Agents

    [https://arxiv.org/abs/2610.13002](https://arxiv.org/abs/2610.13002)

    该论文提出了一种简单可扩展的赋能度最大化方法——将赋能度视为某状态下可执行技能的数量，引导智能体到达多种技能的良好起点状态——从而首次在大规模环境中实现了理论预测的赋能度涌现特性，如收集资源、改造环境、生存，以及在 Crafter 中建造房屋。

    

    赋能度衡量的是智能体主动控制其环境的能力。在本工作中，我们阐明了技能学习与赋能度之间的联系。虽然技能学习并不等于赋能度，但我们提出了一种简单且可扩展的赋能度最大化方法：赋能度对应于从某一状态出发可执行的技能数量，因此最大化赋能度就等价于到达那些能够作为众多技能良好起点的状态。我们的方法适用于任意底层技能学习方法。以往关于赋能度的研究大多局限于表格型设置，而该方法使我们能够在大规模环境中研究赋能度，从而实现理论上已被预测、但经验上难以观测到的赋能度特性（例如收集资源、改造环境、生存等）。实验表明，赋能度最大化能够解锁有趣的涌现特性，例如在 Crafter 环境中建造房屋。

    arXiv:2610.13002v1 Announce Type: new  Abstract: Empowerment measures the capacity of an agent to actively control its environment. In this work, we explain the connection between skill learning and empowerment. While skill learning is not empowerment, we introduce a simple scalable recipe for empowerment maximization: empowerment corresponds to the number of executable skills from a state, so maximizing empowerment corresponds to reaching states that are good starting points for many skills. Our recipe applies to arbitrary underlying skill learning methods. Whereas most prior work on empowerment looks at tabular settings, this recipe enables us to study empowerment at a scale where we can realize theoretically-predicted but empirically-elusive properties of empowerment (e.g., gathering resources, environment modification, survival). Experiments show that empowerment maximization unlocks interesting emergent properties, such as building a house in Crafter. Website and code at \href{htt
    
[^161]: SFT应该学习什么？用于可泛化后训练的条件信息增益

    What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training

    [https://arxiv.org/abs/2610.12995](https://arxiv.org/abs/2610.12995)

    提出CIG-SFT信息论框架，用条件信息增益而非模型兼容性来决定SFT应学习哪些词元，从而提升大语言模型后训练的泛化能力。

    

    监督微调（SFT）仍然是大语言模型后训练中广泛使用的方法，因为固定的示范可以直接引入期望的推理策略和行为模式。然而，强迫模型模仿外部提供的轨迹也可能干扰预训练已经支持的行为，从而限制模型在训练分布之外的泛化能力。近期的动态SFT方法通过根据模型侧信号（如预测概率、先验支持度或分布兼容性）对词元级监督进行重新加权来缓解这一问题。然而，模型兼容性并不等同于学习价值：一个当前模型支持度较弱的词元，可能是不希望出现的不匹配，也可能恰恰是示范意图教授的任务特定决策。我们提出CIG-SFT，这是一个信息论框架，用于估计任务条件下的信息量……

    arXiv:2610.12995v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) remains a widely used approach for post-training large language models, because fixed demonstrations can directly introduce desired reasoning strategies and behavioral patterns. Yet forcing a model to imitate externally provided trajectories may also interfere with behaviors already supported by pretraining, limiting generalization beyond the training distribution. Recent dynamic SFT methods mitigate this issue by reweighting token-level supervision according to model-side signals such as predictive probability, prior support, or distributional compatibility. However, model compatibility is not equivalent to learning value: a token that is weakly supported by the current model may either be an undesirable mismatch or precisely the task-specific decision that the demonstration is intended to teach. We propose CIG-SFT, an information-theoretic framework that estimates the task-conditioned information carried 
    
[^162]: 离线到在线线性赌博机的高斯宽度界

    Gaussian-Width Bounds for Offline-to-Online Linear Bandits

    [https://arxiv.org/abs/2610.12990](https://arxiv.org/abs/2610.12990)

    该论文利用高斯向量Lipschitz函数的集中性改进了对奖励预测误差的控制，将部分覆盖离线数据下线性赌博机的遗憾界从 $\widetilde O(\sqrt{dkT})$ 提升至最优的 $\widetilde O(k\sqrt T)$，并给出了匹配的上下界。

    

    我们研究离线数据如何帮助随机线性赌博机中的在线学习。我们关注部分覆盖的离线数据，即 $d$ 个特征方向中只有 $k$ 个方向的覆盖较差。现有基于分阶段消除和乐观原则的算法分析在此情形下给出的遗憾界为 $\widetilde O(\sqrt{dkT})$ 量级。我们证明这种依赖关系源于对奖励预测误差的宽松控制，并利用高斯向量的Lipschitz函数的集中性获得了更紧的界。基于此分析，我们开发了一种遗憾为 $\widetilde O(k\sqrt T)$ 的分阶段消除算法。对于单位欧氏球，我们证明了部分覆盖下匹配的遗憾下界，以及在均匀离线覆盖下匹配的上下界。我们还研究了利用离线数据的纯探索问题，为固定设计算法和自适应算法分别提供了简单遗憾的算法及匹配的上下界。

    arXiv:2610.12990v1 Announce Type: new  Abstract: We study how offline data can help online learning in stochastic linear bandits. We focus on partially covered offline data, where only $k$ of the $d$ feature directions are poorly covered. Existing analyses of algorithms based on phased elimination and optimism give regret bounds of order $\widetilde O(\sqrt{dkT})$ in this regime. We show that this dependence arises from loose control of the reward prediction error, and obtain tighter bounds using concentration of Lipschitz functions of Gaussian vectors. Based on this analysis, we develop a phased elimination algorithm with regret $\widetilde O(k\sqrt T)$. For the unit Euclidean ball, we prove a matching regret lower bound under partial coverage, as well as matching upper and lower bounds under uniform offline coverage. We also study pure exploration with offline data, providing algorithms and matching upper and lower bounds on simple regret for fixed-design algorithms and also for adap
    
[^163]: 掩码文本重建中语料库采样分歧的预测

    Forecasting Corpus-Sampling Disagreement in Masked Text Reconstruction

    [https://arxiv.org/abs/2610.12987](https://arxiv.org/abs/2610.12987)

    该论文推导出一个一阶公式，可仅凭参考语料库预测两个独立语料库上训练的掩码文本重建预测器之间的期望分歧，并发现语料库扩大四倍仅使分歧降低5–40%，远低于1/n缩放律的预期，其根源在于罕见上下文的平滑效应。

    

    如果从同一总体中再次收集一个文本语料库，预测器的输出会发生多大变化？我们针对掩码文本重建任务研究了这一问题，使用了一种由邻近词对计数构建的平滑预测器，并在完整的50,257词元的GPT-2词表上进行评估。将每个源文档视为一个观测值，我们推导出了拟合于两个独立语料库的预测器之间期望分歧的一阶公式。该公式考虑了可变的文档长度、随机上下文计数以及跨词元偏移的相关性，并且可以在不访问任何目标语料库的情况下，仅凭一个单独的参考语料库进行计算。在TinyStories、WikiText-103和CNN/DailyMail数据集上，将每个语料库从512个块扩展到2,048个块（每块64个词元）仅使分歧降低了5–40%，远低于1/n缩放律所隐含的75%。该预测准确地追踪了这种缓慢的下降趋势，我们将其归因于对罕见上下文的平滑处理。

    arXiv:2610.12987v1 Announce Type: new  Abstract: If a second text corpus were collected from the same population, how much would a predictor's outputs change? We study this question for masked-text reconstruction using a smoothed predictor built from nearby token-pair counts and evaluated over the full 50,257-token GPT-2 vocabulary. Treating each source document as one observation, we derive a first-order formula for the expected disagreement between predictors fitted to two independent corpora. The formula accounts for variable document length, random context counts, and correlations across token offsets, and it can be evaluated from a separate reference corpus without access to either target corpus. On TinyStories, WikiText-103, and CNN/DailyMail, growing each corpus from 512 to 2,048 chunks of 64 tokens lowers disagreement by only 5--40\%, far short of the 75\% implied by $1/n$ scaling. The forecast tracks this slow decline, which we trace to smoothing of rarely seen contexts. Forec
    
[^164]: 面向事件驱动潜在跳跃模型的扫描并行变分推断

    Scan-Parallel Variational Inference for event-driven Latent Jump Models

    [https://arxiv.org/abs/2610.12986](https://arxiv.org/abs/2610.12986)

    提出潜在跳跃模型（LJMs），通过泊松事件驱动的可学习瞬时跳跃来显式建模不规则时间序列中未观测冲击的时间与幅度，并配套开发了扫描并行变分推断方法用于高效训练。

    

    连续时间潜在状态空间模型被广泛应用于不规则观测的时间序列，通常通过一个随机初始状态加平滑的常微分方程（ODE）动力学，或通过随机微分方程（SDE）累积的布朗扰动来表示不确定性。虽然这些构造对于平滑或扩散系统而言是自然的，但它们并不能直接编码一组数量有限的未观测冲击，而这些冲击的发生时间、幅度及其对记忆的影响本身都是不确定的。我们转而提出潜在跳跃模型（LJMs）：潜在状态在泊松驱动的各事件时刻之间遵循闭式确定性动力学，并在每个事件时刻发生一次学习得到的瞬时跳跃，其强度和标记分布依赖于状态。这定义了一个时间齐次的马尔可夫过程，其跳跃是内在的状态转移，而非数值积分步长。在训练方面，我们推导了一种可处理的事件增广（原文摘要至此截断）

    arXiv:2610.12986v1 Announce Type: cross  Abstract: Continuous-time latent state-space models are widely used for irregularly observed time series, typically representing uncertainty through a random initial state followed by smooth ordinary differential equation (ODE) dynamics, or Brownian perturbations accumulated by a stochastic differential equation (SDE). While these constructions are natural for smooth or diffusive systems, they do not directly encode a finite set of unobserved shocks whose timing, magnitude, and effect on memory are themselves uncertain. We instead introduce latent jump models (LJMs): the latent state follows closed-form deterministic dynamics between Poisson-driven event times and undergoes a learned instantaneous jump at each event, with state-dependent intensity and mark distributions. This defines a time-homogeneous Markov process whose jumps are intrinsic state transitions rather than numerical integration steps. For training, we derive a tractable event-aug
    
[^165]: NucleoNet：面向多几何构型神经变分蒙特卡洛的核感知行列式拟设

    NucleoNet: A Nuclear-Aware Determinant Ansatz for Multi-Geometry Neural Variational Monte Carlo

    [https://arxiv.org/abs/2610.12985](https://arxiv.org/abs/2610.12985)

    NucleoNet 提出一种核感知的神经行列式拟设，将电子与原子核纳入统一的共享骨干网络中联合演化，使单一模型可跨多个核几何构型联合优化，从而以更少的参数准确描绘分子势能面。

    

    神经网络变分蒙特卡洛为求解电子薛定谔方程提供了一条精确的途径，但许多神经波函数仅针对单个核几何构型进行优化，需要分别进行多次计算才能绘制出势能面。我们提出了 NucleoNet，这是一种核感知的行列式拟设，其中电子与原子核构成一个统一的粒子集合，通过共享的电子-原子核骨干网络进行传播，该骨干网络联合演化单体与双体表示。原子核携带可学习的表示，与电子表示一同演化并对轨道构建进行条件化，而反对称性仅由电子行列式施加。在平衡构型的小分子基准测试中，NucleoNet 以显著更少的参数量达到了与现有神经波函数基线相当的精度。对于每个分子体系，一个跨多个几何构型联合优化的单一模型即可描述该分子的势能面。

    arXiv:2610.12985v1 Announce Type: cross  Abstract: Neural-network variational Monte Carlo provides an accurate route to solving the electronic Schr\"odinger equation, but many neural wavefunctions are optimized for a single nuclear geometry, requiring separate calculations to map out a potential energy surface. We introduce NucleoNet, a nuclear-aware determinant ansatz in which electrons and nuclei form a unified particle set propagated through a shared electron--nucleus backbone that jointly evolves one-particle and two-particle representations. Nuclei carry learned representations that evolve alongside the electronic representations and condition orbital construction, while antisymmetry is imposed only by electronic determinants. On equilibrium small-molecule benchmarks, NucleoNet achieves accuracy comparable to established neural-wavefunction baselines with substantially fewer parameters. For each molecular system, a single model optimized jointly across geometries describes the N$_
    
[^166]: 结合中期天气预报的扩散模型降水生成

    Diffusion Precipitation Generation with Medium-range Weather Forecasting

    [https://arxiv.org/abs/2610.12976](https://arxiv.org/abs/2610.12976)

    该论文提出“动力学-降水解耦”即插即用框架，利用以大气状态和卫星-雨量计降水估计为条件的扩散生成器，将冻结天气基础模型的预报映射为与观测对齐的每日降水，从而克服再分析数据带来的区域性和强度相关偏差。

    

    准确的每日降水预报对于对天气敏感的决策至关重要。尽管近期的天气基础模型已经取得了显著进展，但它们是在再分析数据上训练的，这些数据存在区域性和强度相关的偏差。由于在基于观测的目标上重新训练WFMs成本高昂，我们研究了动力学-降水解耦，这是一个即插即用的框架，利用单独训练的生成器将冻结天气模型的非降水预报映射为与观测对齐的每日降水。其关键挑战在于捕获异构的降雨分布，并将每日累积降水与随空间和时间演变的大气信号对齐。为了解决这些问题，我们训练了一个以大气状态和卫星-雨量计降水估计为条件的扩散生成器。多尺度循环编码器用于总结大气演变过程，而交叉注意力机制则将大气与降水信息对齐。

    arXiv:2610.12976v1 Announce Type: new  Abstract: Accurate daily precipitation forecasts are important for weather-sensitive decisions. Although recent Weather Foundation Models (WFMs) have achieved significant progress, they are trained on reanalysis data, which exhibit regional and intensity-dependent biases. Since retraining WFMs on observation-based targets is costly, we study Dynamics-Precipitation Decoupling, a plug-and-play framework that uses a separately trained generator to map non-precipitation forecasts from a frozen weather model to observation-aligned daily precipitation. Its key challenges are to capture heterogeneous rainfall distributions and align daily accumulations with atmospheric signals that evolve across space and time. To address them, we train a diffusion generator conditional on atmospheric states and satellite-gauge estimates of precipitation. A multi-scale recurrent encoder summarizes atmospheric evolution, while cross-attention aligns atmospheric and precip
    
[^167]: Minions：基于闲置Apple Silicon设备构建的动态分布式虚拟GPU云

    Minions: A Dynamic and Distributed Virtual GPU Cloud Built on Idle Apple Silicon Devices

    [https://arxiv.org/abs/2610.12970](https://arxiv.org/abs/2610.12970)

    本文提出Minions系统，将闲置的Apple Silicon Mac设备聚合为动态分布式虚拟GPU云，用于执行大规模机器学习推理任务，无需额外采购GPU硬件。

    

    对GPU加速的机器学习推理需求日益增长，这暴露了集中式GPU基础设施的局限性，包括高成本、弹性有限以及资源受限时的长时间等待。与此同时，现代基于Apple Silicon的macOS设备配备了强大的GPU，但这些GPU在日常使用中大部分时间处于闲置状态。本文提出了Minions，一个动态分布式虚拟GPU云，它聚合macOS设备上闲置的Apple Silicon GPU来执行大规模机器学习推理工作负载。Minions使团队能够利用现有硬件，无需额外采购GPU。该系统提供灵活的GPU推理任务执行、容错能力和高效的资源利用率。我们的经验表明，在集中式GPU资源受限的环境中，闲置的macOS设备可以被转化为实用的分布式GPU推理平台。

    arXiv:2610.12970v1 Announce Type: cross  Abstract: The increasing demand for GPU-accelerated machine learning inference has exposed limitations in centralized GPU infrastructures, including high cost, limited elasticity, and long wait times under resource constraints. At the same time, modern Apple Silicon-based macOS devices are equipped with powerful GPUs that remain largely idle during normal usage. This paper presents Minions, a dynamic and distributed virtual GPU cloud that aggregates idle Apple Silicon GPUs across macOS devices to execute large-scale machine learning inference workloads. Minions enables teams to leverage existing hardware without requiring additional GPU procurement. The system provides flexible GPU inference task execution, fault tolerance, and efficient resource utilization. Our experience shows that idle macOS devices can be transformed into a practical distributed GPU inference platform in environments where centralized GPU resources are constrained.
    
[^168]: DataCupid：面向具有延迟反馈的训练数据获取的稳定匹配修复

    DataCupid: Stable Matching Repair for Training Data Acquisition with Delayed Feedback

    [https://arxiv.org/abs/2610.12960](https://arxiv.org/abs/2610.12960)

    该论文提出剩余负载修复（RLR）方法，通过将可靠的结果证书与批量匹配调度相结合，解决训练数据获取中带延迟反馈的稳定匹配修复问题，并证明其发布轮次界限和单次修复耦合能保证认证时间、观测发布量及检查点时间均不增加。

    

    arXiv:2610.12960v1 公告类型： new 摘要：效用变化后的匹配修复可以复用有效的观测结果，但新的评估必须遵守匹配容量限制，且可能乱序返回结果。我们研究具有有界的（可能依赖于奖励的）报告延迟的已宣布失效问题，以及不占用未来匹配容量的待处理评估。剩余负载修复（RLR）将可靠的结果证书与针对不均等样本缺口的批量匹配调度相结合。其完整的二元配额调度实现了相对于终端剩余端点负载、系数 cb < 1.607 的尖锐发布轮次界限，并具有独立的延迟界限和证据生命周期界限。RLR-P 监控完整的发布顺序前缀，在接受之前不改变采样计划。在确定性纯预言机下，来自相同耗尽初始状态的单次修复耦合保证认证时间、发布的观测数量或达到可复用检查点的时间均不会增加。我们还推导出一个信息……（摘要在此处截断）

    arXiv:2610.12960v1 Announce Type: new  Abstract: Matching repair after utility changes can reuse valid observations, but new evaluations must respect matching capacity and may return out of order. We study announced invalidations with bounded, possibly reward-dependent reporting delays and pending evaluations that do not occupy future matching capacity. Residualload repair (RLR) combines sound outcome certificates with batched matching schedules for unequal sample deficits. Its complete dyadic quota schedule admits a sharp issue-round bound with coefficient cb < 1.607 relative to the terminal residual endpoint load, together with separate delay and evidence-lifetime bounds. RLR-P monitors complete issue-order prefixes without changing the sampling plan before acceptance. With a deterministic pure oracle, a single-repair coupling from the same drained initial state guarantees no increase in certification time, issued observations, or time to a reusable checkpoint. We also derive an info
    
[^169]: JoyInside-VLM-Agent：面向具身智能体的产品条件化多模态策略学习

    JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents

    [https://arxiv.org/abs/2610.12957](https://arxiv.org/abs/2610.12957)

    本文提出JoyInside-VLM-Agent，一个通过产品条件化轨迹构建、闭环沙盒数据生成与分阶段策略学习（含新颖的DPER强化学习方法）训练的ReAct式多模态具身智能体，能够将多模态观察与用户请求转化为可执行动作，并适应产品约束、目标变化与执行反馈。

    

    具身智能体必须将多模态观察和用户请求转化为可执行的动作，同时适应产品约束、不断变化的目标以及执行反馈。我们提出了JoyInside-VLM-Agent，这是一个ReAct风格的多模态智能体，通过面向机器人平台和智能硬件的产品条件化轨迹构建与分阶段策略学习开发而成。一个闭环沙盒生成涵盖目标修订和失败恢复的多模态交互轨迹，并依据执行结果和语义标准进行校验。在强化学习之前，先进行监督微调和多教师在线策略蒸馏；强化学习采用DPER（双源特权在线策略自蒸馏与token级证据路由）方法。DPER在冻结的特权教师与共享推理策略参数的特权学生之间路由token级证据，在调节由奖励推导的token优势的同时保持其符

    arXiv:2610.12957v1 Announce Type: cross  Abstract: Embodied agents must translate multimodal observations and user requests into executable actions while adapting to product constraints, changing goals, and execution feedback. We present JoyInside-VLM-Agent, a ReAct-style multimodal agent developed through product-conditioned trajectory construction and staged policy learning for robotic platforms and smart hardware. A closed-loop sandbox generates multimodal interaction trajectories covering goal revisions and failure recovery, which are checked against execution outcomes and semantic criteria. Supervised fine-tuning and multi-teacher on-policy distillation precede reinforcement learning with DPER (Dual-Source Privileged On-Policy Self-Distillation with Token-Level Evidence Routing). DPER routes token-level evidence between a frozen privileged teacher and a privileged student sharing the rollout policy's parameters, modulating reward-derived token advantages while preserving their sig
    
[^170]: 从不平衡到证据：严重样本不平衡下复杂数据的深度非参数检验

    From Imbalance to Evidence: Deep Nonparametric Testing for Complex Data under Severe Sample Imbalance

    [https://arxiv.org/abs/2610.12946](https://arxiv.org/abs/2610.12946)

    提出增强型不平衡感知深度分类器双样本检验，创新性地将严重样本不平衡从分类器训练的不利因素转化为可重复利用的检验证据来源，从而在高不平衡情形下实现对图等复杂数据更可靠的分布比较。

    

    在严重样本不平衡的情况下比较分布，需要有效利用数量占多数的丰富观测数据，同时对涉及较小样本的差异保持敏感性。这一挑战对于图等复杂的非欧几里得数据尤为突出，因为其结构特征需要灵活的检验方法。基于分类器的双样本检验提供了这种灵活性，但现有方法主要是针对平衡或规模相近的样本设计的。本文研究了高度不平衡情形下的C2ST，并表明严重的样本不平衡可能会显著降低检验性能。与传统的将样本不平衡仅视为分类器训练中不利因素的不平衡学习方法不同，我们将其转化为可重复检验证据的来源。基于这一见解，我们提出了一种增强型不平衡感知深度分类器双样本检验，这是一种分类……

    arXiv:2610.12946v1 Announce Type: cross  Abstract: Comparing distributions under severe sample imbalance requires effectively using abundant majority observations while retaining sensitivity to differences involving the smaller sample. This challenge is particularly relevant for complex, non-Euclidean data, such as graphs, whose structural characteristics call for flexible testing procedures. Classifier two-sample tests (C2STs) provide such flexibility, but existing procedures are primarily designed for balanced or comparably sized samples. In this paper, we study C2STs in a highly imbalanced regime, and show that severe sample imbalance may substantially degrade testing performance. Different from conventional imbalanced learning approaches that treat sample imbalance merely as a nuisance for classifier training, we turn it into a source of repeated testing evidence. Building on this insight, we propose an Enhanced Imbalance-Aware Deep Classifier Two-Sample Test (EIAD-C2ST), a classif
    
[^171]: 基于线性化最优传输的测度值数据稀疏字典学习：高效算法与样本复杂度

    Sparse Dictionary Learning for Measure-Valued Data via Linearized Optimal Transport: Efficient Algorithms and Sample Complexity

    [https://arxiv.org/abs/2610.12924](https://arxiv.org/abs/2610.12924)

    该论文提出一个基于线性化最优传输的稀疏字典学习框架，将概率测度嵌入希尔伯特空间并通过稀疏自编码器学习数据自适应字典，实现对高光谱图像、点云及大语言模型文档嵌入等测度值数据的高效可解释表示，同时给出以原子类度量复杂度刻画的泛化界和样本复杂度理论保证。

    

    我们通过使用源自最优传输几何的特征将概率测度嵌入到希尔伯特空间中，研究概率测度值数据的字典学习问题。我们通过稀疏自编码器参数化学习数据自适应的字典，将概率测度表示为快速、可扩展的稀疏有限维编码。所得框架将最优传输丰富的几何结构与稀疏表示学习的可解释性和高效性相结合。我们在多种数据模态上展示了该框架的实用性，包括高光谱图像、几何点云以及大语言模型潜在空间中的文档嵌入。我们还针对广泛的 $L^2$ 值测度嵌入类别，证明了稀疏字典学习目标的泛化界，该泛化界以原子类的度量复杂度来表示。对于最优传输和熵正则化传输映射字典，我们推导了……（原文摘要在此处截断）

    arXiv:2610.12924v1 Announce Type: cross  Abstract: We study dictionary learning for probability measure-valued data by embedding measures into Hilbert spaces using features derived from optimal transport geometry. We learn data-adapted dictionaries through sparse autoencoder parameterizations, producing fast, scalable representations of probability measures as sparse finite-dimensional codes. The resulting framework combines the rich, geometric structure of optimal transport with the interpretability and efficiency of sparse representation learning. We demonstrate its utility across diverse data modalities, including hyperspectral images, geometric point clouds, and document embeddings in large language model latent spaces. We also prove generalization bounds for sparse dictionary learning objectives on broad classes of $L^2$-valued measure embeddings, expressed in terms of the metric complexity of the atom class. For optimal transport and entropic transport map dictionaries, we derive
    
[^172]: 超越二阶信息：带噪声特征的付费逻辑回归

    Beyond Second-Order Information: Paid Logistic Regression with Noisy Features

    [https://arxiv.org/abs/2610.12919](https://arxiv.org/abs/2610.12919)

    该文提出带付费噪声特征的在线逻辑回归问题，证明与付费线性回归不同，仅凭协方差这类二阶信息已不足以刻画最优损失——相同协方差的噪声分布可产生不同损失、更低方差甚至导致更高损失，因此需要超越二阶信息的新结构刻画。

    

    我们研究具有付费随机特征的在线二分类预测问题，其中学习者可以花费资源来获得对潜在特征的更好观测。这一设定受医学检测等应用的启发：额外花费可以提供对患者特征更精确的测量，从而更好地预测二元结果。在每一轮中，一个潜在特征被独立同分布地抽取，响应通过逻辑（logistic）模型生成，学习者在观测到带噪声的特征版本之前选择支付多少费用。学习者的目标是对预测损失和特征获取成本进行联合控制。与以往关于带付费特征的在线线性回归的结果不同，仅有协方差信息已不再足够：两个具有相同协方差的噪声分布可能导致不同的最优逻辑损失，甚至更低的方差反而会带来更高的损失。在一种结构性排序下，更大的支付…（原文摘要在此处截断）

    arXiv:2610.12919v1 Announce Type: cross  Abstract: We consider online binary prediction with paid stochastic features, where the learner can spend resources to obtain better observations of the underlying features. This setting is motivated by applications such as medical testing, where additional cost can provide more precise measurements of patient's features for predicting a binary outcome. At each round, a latent feature is drawn i.i.d., the response is generated through a logistic model, and the learner chooses how much to pay before observing a noisy version of the feature. The goal of learner is to jointly control prediction loss and acquisition cost. In contrast to the former result on the online linear regression with paid features, covariance information alone is no longer sufficient: two noise distributions with the same covariance may induce different optimal logistic losses, and lower variance can even lead to higher loss. Under a structural ordering in which larger paymen
    
[^173]: 受控极小极大：基于控制理论的对抗优化稳定化方法

    Controlled-Minimax: Stabilizing Adversarial Optimization via Control Theory

    [https://arxiv.org/abs/2610.12915](https://arxiv.org/abs/2610.12915)

    本文将极小极大优化的训练过程建模为二阶常微分方程，揭示了Hessian矩阵特征结构导致非收敛的根本原因，并提出Controlled-Minimax框架，通过设计反馈控制器实现了与Hessian谱无关的局部指数稳定性理论保证。

    

    极小极大优化是对抗学习的基石，然而其面临严重的稳定性挑战，常表现为持续的极限环或发散。本文从连续时间动力系统的视角研究了这些不稳定性。通过将训练动力学建模为二阶常微分方程（ODE），我们严格地识别出非收敛的根本原因：损失函数的Hessian矩阵在平衡点附近存在问题的特征结构。为解决这一问题，我们提出了Controlled-Minimax，一种新颖的稳定化框架，它将优化轨迹重新表述为一个可控的物理系统。与启发式正则化技术不同，我们的方法推导出一个特定的反馈控制器，无论底层Hessian谱如何，该控制器都能提供局部指数稳定性的理论保证。因此，我们的方法能够稳定传统方法难以处理的一般极小极大问题。

    arXiv:2610.12915v1 Announce Type: new  Abstract: Minimax optimization is fundamental to adversarial learning yet suffers from severe stability challenges, often manifesting as persistent limit cycles or divergence. This paper investigates these instabilities through the lens of continuous-time dynamical systems. By modeling training dynamics as a second-order Ordinary Differential Equation (ODE), we rigorously identify the root cause of non-convergence: the problematic eigenstructure of the loss function's Hessian near equilibrium points. To address this, we introduce Controlled-Minimax, a novel stabilization framework that reformulates the optimization trajectory as a controllable physical system. Unlike heuristic regularization techniques, our approach derives a specific feedback controller that provides theoretical guarantees of local exponential stability, regardless of the underlying Hessian spectrum. Consequently, our method stabilizes general minimax problems where traditional g
    
[^174]: BLEND：在联邦视觉语言模型中平衡个性化与泛化能力

    BLEND: Balancing Personalization vs. Generalization in Federated Vision-Language Models

    [https://arxiv.org/abs/2610.12907](https://arxiv.org/abs/2610.12907)

    BLEND框架通过结合客户端共享的全局视觉与文本适配器、每个客户端专属的个性化视觉投影适配器，以及融合全局与个性化输出的新型损失函数，在联邦视觉语言模型中同时实现了个性化与泛化能力的提升。

    

    联邦视觉语言模型是机器学习的新兴前沿领域之一，旨在将联邦学习整合到视觉语言模型的微调流程中。在这项工作中，我们提出了BLEND，一个旨在同时增强联邦视觉语言模型个性化与泛化能力的新框架。BLEND提出了一种选择性的VLM参数微调与聚合策略，包含两个部分：(i) 在客户端之间共享的全局视觉和文本适配器，(ii) 为每个客户端量身定制的个性化视觉投影适配器。BLEND进一步引入了一种新颖的损失函数，通过融合全局适配器和个性化适配器的输出来实现个性化，同时通过零样本VLM投影头实现的原理性锚点设计来促进泛化。大量实验表明，BLEND始终优于最先进的基线方法。

    arXiv:2610.12907v1 Announce Type: cross  Abstract: Federated vision-language models (FedVLMs) represent one of the emerging frontiers of machine learning, aiming to integrate federated learning within the fine-tuning pipelines of vision-language models (VLMs). In this work, we introduce BLEND, a new framework designed to jointly enhance the personalization and generalization capabilities of FedVLMs. BLEND proposes a selective VLM parameter fine-tuning and aggregation strategy that consists of (i) global vision and text adapters shared across clients, and (ii) a personalized vision projection adapter tailored to each client. BLEND further introduces a novel loss function that blends the outputs of the global and personalized adapters to achieve personalization, while simultaneously promoting generalization through a principled anchor design implemented via the projection heads of the zero-shot VLM. Extensive experiments demonstrate that BLEND consistently outperforms state-of-the-art ba
    
[^175]: GIST：面向表格基础模型的生成器状态感知监督

    GIST: Generator-state-Informed Supervision for Tabular Foundation Models

    [https://arxiv.org/abs/2610.12904](https://arxiv.org/abs/2610.12904)

    提出GIST方法，利用生成器中间状态在条件标签分布上对查询损失进行平均，从而消除表格基础模型预训练中的采样变异性，并在不改变架构和推理的情况下提升TabArena与TALENT基准上的性能。

    

    基于先验数据拟合网络的表格基础模型（TFMs）通常使用采样得到的硬查询标签，这在预训练中留下了可避免的采样变异性。一个关键观察是，中间生成器状态能够揭示超出单个采样结果的标签不确定性。我们提出了面向表格基础模型的生成器状态感知监督（GIST），利用这些状态在条件标签分布上对查询损失进行平均。GIST联合设计了任务生成与查询监督，使用熵校准编码在支持的标签相关分支上获得可处理的、与输入一致的目标。我们证明这种平均方式保持了总体目标，并且在每个生成分布内固定参数的条件下不会产生更大的梯度协方差。在匹配的预训练调度下，GIST在不改变架构或推理的情况下，提升了三种架构在TabArena和TALENT上的平均排名。在这些基准上……

    arXiv:2610.12904v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) based on prior-data fitted networks typically use sampled hard query labels, leaving avoidable sampling variability in pretraining. A key observation is that intermediate generator states can reveal label uncertainty beyond a sampled outcome. We propose Generator-state-Informed Supervision for TFMs (GIST), which uses these states to average query losses over conditional label distributions. GIST jointly designs task generation and query supervision, using entropy-calibrated encodings to obtain tractable, input-consistent targets on supported label-dependent branches. We show that this averaging preserves the population objective and yields no larger gradient covariance at fixed parameters within each generating distribution. Under matched pretraining schedules, GIST improves mean rank across three architectures on both TabArena and TALENT without changing architecture or inference. Across these benchmarks
    
[^176]: 重新思考扩散与解码的角色：用于并行语言生成的预测态流

    Rethinking the Roles of Diffusion and Decoding: Predictive-State Flows for Parallel Language Generation

    [https://arxiv.org/abs/2610.12898](https://arxiv.org/abs/2610.12898)

    该论文提出预测态流（PSF），重新划分扩散先验与解码器的生成职责，让整流流先验仅生成基于前文的高层预测态，再由块因果解码器并行完成词元级选择，从而实现并行语言生成。

    

    在连续扩散语言模型（DLMs）中，由扩散先验生成的潜变量几乎确定性地被映射到词元，映射方式要么通过解码器，要么通过词元嵌入矩阵。与图像生成不同——图像中视觉细节可以连续变化——扩散先验必须在连续空间中捕捉细粒度的离散词元选择，这可能使得联合潜分布难以建模。我们主张应当重新划分生成角色：扩散先验应当生成高层次的预测信息，而解码器则负责解决剩余的词元级选择。基于这一观点，我们提出了预测态流（Predictive-State Flows, PSF），该方法将序列划分为多个块，并将每个块与一个仅由前文上下文计算得到的预测态相关联。一个整流流先验生成预测态序列，随后一个块因果解码器并行生成所有块。

    arXiv:2610.12898v1 Announce Type: new  Abstract: In continuous diffusion language models (DLMs), latents generated by the diffusion prior are mapped almost deterministically to tokens, either by a decoder or through a token embedding matrix. Unlike in image generation, where visual details can vary continuously, the diffusion prior must capture fine-grained discrete token choices in continuous space, which can make the joint latent distribution difficult to model. We argue that generative roles should instead be divided: the diffusion prior should generate high-level predictive information, while the decoder should resolve the remaining token-level choices. Based on this view, we introduce Predictive-State Flows (PSF), which partitions a sequence into blocks and associates each block with a predictive state computed solely from the preceding context. A rectified-flow prior generates the sequence of predictive states, after which a block-causal decoder generates all blocks in parallel, 
    
[^177]: 面向吸引与排斥、具有不同相互作用核的广义行列式点过程

    Generalized Determinantal Point Processes for Attraction and Repulsion with Distinct Interaction Kernels

    [https://arxiv.org/abs/2610.12894](https://arxiv.org/abs/2610.12894)

    提出行列式比点过程（DRPP），利用两个对称正定核行列式之比分别建模排斥与吸引，在保持特征可解释性的同时，借助辅助变量表示实现高效MCMC采样与最大似然学习。

    

    行列式点过程（DPPs）通过排斥作用对子集选择概率进行建模，但许多应用中还表现出由成对关系或共享潜在结构所产生的吸引作用。非对称DPPs（NDPPs）虽然能够同时捕捉这两种效应，但其核函数难以解释和设定。我们提出行列式比点过程（DRPP），它通过两个对称正定核的行列式之比来建模排斥与吸引。这一表述在保留标准DPPs基于特征的解释性的同时，允许对两种效应分别进行设定。此外，我们引入了一种辅助变量表示，可实现高效的MCMC采样，并给出包括归一化常数在内的关键统计量的可计算期望公式，从而共同支持最大似然学习。我们还从理论上刻画了适合DRPPs的相互作用结构：对于聚类（原文在此处截断）

    arXiv:2610.12894v1 Announce Type: cross  Abstract: Determinantal point processes (DPPs) model subset selection probability with repulsion, but many applications also exhibit attraction arising from pairwise relations or shared latent structures. Nonsymmetric DPPs (NDPPs) capture both effects, but their kernels are difficult to interpret and specify. We propose the determinant-ratio point process (DRPP), which models repulsion and attraction via a ratio of the determinants of two symmetric positive-definite kernels. This formulation preserves the feature-based interpretation of standard DPPs while allowing the two effects to be specified separately. Furthermore, we introduce an auxiliary-variable representation that enables efficient MCMC sampling and yields tractable expectation formulas for key statistics including the normalizing constant, which together facilitate maximum-likelihood learning. We theoretically characterize the interaction structures well suited to DRPPs: for clustere
    
[^178]: 基于LoRA适配器的个性化联邦直接偏好优化

    Personalized Federated Direct Preference Optimization via LoRA Adapters

    [https://arxiv.org/abs/2610.12885](https://arxiv.org/abs/2610.12885)

    提出pFedDPO方法，让每个客户端在通过FedAvg同步的共享全局基础模型上保留本地LoRA适配器以捕获个性化偏好，从而将FedDPO理论界中的梯度多样性项κ²替换为更小的残余异构性项κ²_eff。

    

    联邦直接偏好优化使分布式智能体能够在无需集中敏感偏好数据的情况下，以保护隐私的方式对语言模型进行对齐。然而，现有的FedDPO分析假设所有客户端收敛到一个单一的共享策略，这与实践中存在的异构偏好分布从根本上相矛盾。我们提出了个性化联邦直接偏好优化（pFedDPO），其中每个客户端在共享的全局基础模型之上维护一个小的私有LoRA适配器。在pFedDPO中，基础模型每轮通过FedAvg进行同步，而LoRA适配器保持本地化并捕获客户端特定的偏好模式。我们证明，这种分解将FedDPO理论界中的梯度多样性κ²替换为残余异构性κ²_eff，当适配器吸收客户端特定方向时，κ²_eff不大于κ²，其代价是引入一个额外的适配器偏差项，该项会随着（摘要在此处被截断）……

    arXiv:2610.12885v1 Announce Type: new  Abstract: Federated Direct Preference Optimization (FedDPO) enables privacy-preserving alignment of language models across distributed agents without centralizing sensitive preference data. However, existing FedDPO analysis assumes all clients converge to a single shared policy, which is fundamentally at odds with the heterogeneous preference distributions present in practice. We propose Personalized FedDPO (pFedDPO), in which each client maintains a small private LoRA adapter atop a shared global base model. In pFedDPO, the base model is synchronized via FedAvg each round, while LoRA adapters remain local and capture client-specific preference patterns. We prove that this decomposition replaces the gradient diversity $\kappa^2$ in the FedDPO bound with a residual heterogeneity $\kappa^2_{\mathrm{eff}}$, which is no larger than $\kappa^2$ when adapters absorb client-specific directions, at the cost of an additive adapter-bias term that vanishes as
    
[^179]: CoGameReg：基于合作博弈的协同半监督回归用于地震储层物性预测

    CoGameReg: Cooperative-Game-Based Collaborative Semi-Supervised Regression for Seismic Reservoir Property Prediction

    [https://arxiv.org/abs/2610.12884](https://arxiv.org/abs/2610.12884)

    提出了基于合作博弈的半监督回归框架CoGameReg，通过三重博弈Shapley机制为异构回归器分配非负贡献权重，利用贡献加权伪标签在井标签稀疏条件下实现更准确的地震储层物性预测。

    

    当测井标签稀疏而三维地震观测数据丰富时，从地震数据中准确估计储层物性仍然具有挑战性。我们提出了CoGameReg，这是一个基于合作博弈的半监督回归框架，利用异构回归器的贡献加权预测来构建共享伪标签。这种设计使每个回归器的影响在稀疏井控条件下为无标签地震样本生成的学习目标中得以显式体现。三重博弈Shapley机制将秩-MSE性能博弈与成对一致性博弈相结合，从有标签训练数据中估计非负的回归器权重。这些权重在协同训练期间保持固定，并被重复用于最终的预测融合。在重新训练之前，采用共同的局部k近邻误差减少准则对候选伪标签进行筛选。我们在10个公共回归数据集上对CoGameReg进行了评估（摘要此处截断）。

    arXiv:2610.12884v1 Announce Type: new  Abstract: Accurate estimation of reservoir properties from seismic data remains challenging when well-log labels are sparse but three-dimensional seismic observations are abundant. We present CoGameReg, a cooperative-game-based semi-supervised regression framework that uses contribution-weighted predictions of heterogeneous regressors to construct shared pseudo-labels. This design makes each regressor's influence explicit in the learning targets generated for unlabeled seismic samples under sparse well control. The Tri-Game Shapley mechanism combines a rank-MSE performance game with a pairwise consistency game to estimate nonnegative regressor weights from labeled training data. These weights remain fixed during collaborative training and are reused for final prediction fusion. Candidate pseudo-labels are screened using a common local k-nearest-neighbor error-reduction criterion before retraining. We evaluate CoGameReg on 10 public regression data
    
[^180]: HART：混合注意力-RWKV时间序列基础模型

    HART: Hybrid Attention-RWKV Time series Foundation Model

    [https://arxiv.org/abs/2610.12883](https://arxiv.org/abs/2610.12883)

    本文发现RWKV-7在非自回归多步预测中因“占位符空转衰减（PID）”机制导致历史信息被削弱而产生长时程退化，并据此提出了融合注意力机制与RWKV的混合时间序列基础模型HART来解决该问题。

    

    arXiv:2610.12883v1 公告类型：新论文 摘要：非自回归多步预测（NAF）使时间序列基础模型能够在单次前向传播中预测整个未来时间范围。尽管RWKV-7具有强大的序列建模能力，但在我们的NAF实验中，它表现出长时程退化。在这种设置下，未来补丁被表示为缺失输入，这要求循环状态在没有新观测的情况下在预测范围内持续演化。我们对RWKV-7广义增量规则的分析揭示了一种我们称之为占位符空转衰减（PID）的机制：当占位符驱动的写入无法提供足够补偿时，重复的状态转换会削弱历史信息。逐层分析显示，大多数层中的状态注入范数有所降低，而历史区域与预测区域之间的转换矩阵范数保持相似，这与PID机制相符。基于这一分析，我们提出了HART，一种混合注意力-RWKV时间序列基础模型……

    arXiv:2610.12883v1 Announce Type: new  Abstract: Non-autoregressive multi-step forecasting (NAF) enables time series foundation models to predict an entire future horizon in a single forward pass. Despite its strong sequence modeling capabilities, RWKV-7 exhibits long-horizon degradation in our NAF experiments. In this setup, future patches are represented as missing inputs, requiring the recurrent state to evolve across the prediction horizon without new observations. Our analysis of RWKV-7's generalized delta rule suggests a mechanism we term Placeholder Idling Decay (PID): repeated state transitions can attenuate historical information when placeholder-driven writes provide insufficient compensation. Layer-wise analysis reveals reduced state-injection norms in most layers, while transition-matrix norms remain similar between historical and prediction regions, consistent with PID. Motivated by this analysis, we propose HART, a Hybrid Attention-RWKV Time-series Foundation Model that a
    
[^181]: 当求解器变得喜怒无常：量化与预测混合整数规划中的性能变异性

    When Solvers Get Moody: Quantifying and Predicting Performance Variability in Mixed Integer Programming

    [https://arxiv.org/abs/2610.12876](https://arxiv.org/abs/2610.12876)

    本文提出加权变异性分数（WVS）来量化MIP求解器在重复运行中的性能波动，并利用静态特征和根节点级特征构建预测器来预测这种变异性，同时首次证明了开放实例neos-5045105-creuse的最优性。

    

    混合整数规划（MIP）求解器被广泛使用，但即使在看似中性的扰动下，其性能也可能出现显著波动。这种变异性给基准测试和计算研究结果的解读带来了困难。我们提出了加权变异性分数（WVS），这是一个稳健的多组件指标，它综合了重复运行中归一化运行时间、最终最优性间隙、LP迭代次数以及分支定界节点数的离散程度。在MIPLIB 2017实例的一个大型子集上（每个实例在四小时时限内进行100次随机化运行求解），WVS揭示出一种具有显著上尾的右偏分布：许多实例基本保持稳定，但部分实例表现出无法仅用基本属性解释的大幅变异性。值得注意的是，该实验在两次SCIP运行中成功证明了长期开放实例'neos-5045105-creuse'的最优性。利用静态描述符和根节点级特征，基于树的预测器结合门控尾部（机制），（摘要在此处截断）

    arXiv:2610.12876v1 Announce Type: cross  Abstract: Mixed-integer programming (MIP) solvers are widely used, yet their performance can vary substantially even under seemingly neutral perturbations. Such variability complicates benchmarking, and the interpretation of computational studies. We introduce the Weighted Variability Score (WVS), a robust multi-component metric that aggregates dispersion in normalized runtime, final optimality gap, LP iterations, and branch-and-bound nodes across repeated runs. On a large subset of MIPLIB 2017 instances, each solved in 100 randomized runs under a four-hour limit, WVS reveals a right-skewed distribution with a substantial upper tail: many instances are essentially stable, but some exhibit large variability that is not explained by basic properties alone. Notably, this experiment proved optimality for the open instance 'neos-5045105-creuse' in two SCIP runs. Using static descriptors and root-level features, a tree-based predictor with a gated tai
    
[^182]: 状态依赖重尾噪声下的鲁棒朗之万采样及其在大语言模型微调中的应用

    Robust Langevin Sampling under State-Dependent Heavy-Tailed Noise with Applications to LLM Fine-Tuning

    [https://arxiv.org/abs/2610.12865](https://arxiv.org/abs/2610.12865)

    该论文提出R-LMC、A-LMC和RA-LMC三种朗之万采样算法，通过引入广义光滑性和状态依赖重尾噪声假设，放松了传统SG-LMC对全局Lipschitz梯度与有界噪声方差的要求，并将其应用于大语言模型的贝叶斯微调。

    

    现有针对非对数凹采样的随机梯度朗之万蒙特卡洛（SG-LMC）方法的收敛性保证，通常假设梯度全局Lipschitz连续且噪声方差一致有界。这两个假设即使在基本的采样问题中也可能不成立，而放松它们能更好地反映诸如大语言模型（LLM）贝叶斯微调等实际应用场景。我们通过广义光滑性以及所提出的状态依赖重尾预言机（具有有限p阶矩，且包含Blum-Gladyshev（BG-0）条件）来放松这两个假设。我们分别以及联合地研究这些放松，并提出了三种方法：在梯度全局Lipschitz连续的条件下，鲁棒LMC（R-LMC）通过几何中位数中心化和残差裁剪来处理所提出的重尾预言机；在精确梯度条件下，自适应LMC（A-LMC）在广义光滑性下控制离散化误差；鲁棒自适应LMC（RA-LMC）则同时应对这两种放松（摘要原文在此处截断）。

    arXiv:2610.12865v1 Announce Type: new  Abstract: Existing convergence guarantees for non-log-concave sampling with stochastic gradient Langevin Monte Carlo (SG-LMC) typically assume globally Lipschitz gradients and uniformly bounded noise variance. Both assumptions can fail even in elementary sampling problems, and relaxing them better reflects practical applications such as Bayesian fine-tuning of large language models (LLMs). We relax both assumptions through generalized smoothness and our proposed state-dependent heavy-tailed oracle with finite $p$-th moments, $1<2$ and includes the Blum-Gladyshev (BG-0) condition. We study these relaxations separately and jointly, proposing three methods. Under globally Lipschitz gradients, robust LMC (R-LMC) handles the proposed oracle through geometric-median centering and residual clipping. With exact gradients, adaptive LMC (A-LMC) controls discretization error under generalized smoothness. Robust adaptive LMC (RA-LMC) addresses both relaxation
    
[^183]: 自归一化Kullback-Leibler估计的通用界

    Universal Bounds for Self-Normalized Kullback-Leibler Estimation

    [https://arxiv.org/abs/2610.12851](https://arxiv.org/abs/2610.12851)

    本文证明了基于归一化重要性权重的KL散度估计器存在通用有限样本界——其期望超过目标-提议KL散度至多 $\log 2$，且无需重要性权重有界或高阶矩有限，并将该结果应用于生成模型的Soft Best-of-$N$测试时对齐与相对熵编码。

    

    本研究探讨了一种基于熵的Kullback-Leibler（KL）散度估计器，该估计器由归一化的重要性权重构造而成。我们证明了一个通用的有限样本界，表明对于任意样本量，该估计器的期望超过目标-提议分布间KL散度的幅度至多为 $\log 2$。随后，我们刻画了由重要性加权重采样所诱导的分布与提议分布之间的KL散度，证明其上界同时由基于熵的估计器的期望和目标-提议分布间的KL散度约束。重要的是，这些界既不要求重要性权重有界，也不要求高阶矩有限。我们将这些结果应用于通过Soft Best-of-$N$采样实现生成模型的测试时对齐（这是Verdun等人（2025）提出的一种基于重要性采样的扩展技术），以及应用于相对熵编码，得到了有限样本策略散度证书和无需矩条件的编码成本界。

    arXiv:2610.12851v1 Announce Type: cross  Abstract: This work studies an entropy-based estimator of Kullback-Leibler (KL) divergence constructed from normalized importance weights. We prove a universal finite-sample bound showing that its expectation can exceed the target-proposal KL divergence by at most $\log 2$ for any sample size. We then characterize the KL divergence between the distribution induced by importance-weighted resampling and the proposal, showing that it is upper bounded by both the expectation of the entropy-based estimator and the target-proposal KL divergence. Importantly, these bounds require neither bounded importance weights nor finite higher moments. We apply these results to test-time alignment of generative models via Soft Best-of-$N$ sampling, an importance-sampling-based scaling technique proposed by Verdun et al. (2025), and to relative entropy coding, obtaining finite-sample policy-divergence certificates and moment-free bounds on coding cost.
    
[^184]: 双向引导：生成式贝叶斯中汇总统计量的同步学习

    Two-way Guidance: Simultaneous Learning of Summary Statistics for Generative Bayes

    [https://arxiv.org/abs/2610.12850](https://arxiv.org/abs/2610.12850)

    提出“双向引导”机制，通过汇总统计量与后验分布的同步联合学习，使基于模拟的贝叶斯推断在观测规模不断增大时仍能有效恢复集中化的后验分布。

    

    贝叶斯基于模拟的推断（SBI）在观测规模增大时，可能难以恢复越来越集中的后验分布。我们考虑将充分汇总统计量与后验分布进行联合学习，以此作为在各种SBI方法中实现可扩展推断的通用原则。SBI中常见的做法是通过预先确定的汇总量进行降维，但这可能会部分丢弃后验信息。我们的方法则对后验学习器如何依赖于所学习的汇总量（例如亲和性）施加结构约束，并与后验同时学习该汇总量。我们识别出一种名为“双向引导”的重要机制：在汇总量受到结构限制的条件下，所学汇总量的后验充分性简化了后验学习，而合适的后验目标又引导所学汇总量趋向后验充分性。该机制解释了一个反直觉的发现，即结构限制……（摘要原文在此处截断）

    arXiv:2610.12850v1 Announce Type: cross  Abstract: Bayesian simulation-based inference (SBI) can struggle to recover increasingly concentrated posteriors as observation size grows. We consider learning sufficient summary statistics jointly with the posterior as a general principle for scalable inference in various SBI methods. A common measure taken in SBI is dimension reduction through predetermined summaries, which may partially discard posterior information. Our approach instead imposes structure on how the posterior learner depends on a learned summary (e.g., affinity), and learns the summary simultaneously with the posterior. We identify an important mechanism named two-way guidance: under a structural restriction on the summary, the posterior sufficiency of the learned summary simplifies posterior learning, whereas a suitable posterior objective guides the learned summary towards posterior sufficiency. This mechanism explains a counterintuitive finding that the structural restric
    
[^185]: 基于不动点流匹配的强化学习

    Reinforcement Learning via Fixed-point Flow Matching

    [https://arxiv.org/abs/2610.12848](https://arxiv.org/abs/2610.12848)

    本文提出了一种无需数据的不动点流匹配（FFM）方法，通过将actor更新表述为不动点迭代、以critic的动作梯度作为回归目标，实现了强化学习中流策略的训练与微调。

    

    在最大熵和行为约束强化学习（RL）中，最优策略由玻尔兹曼分布给出。从该分布中生成样本可能具有挑战性，因为它通常复杂且具有多峰性。虽然通过流匹配训练的流策略为表示此类分布提供了强大的框架，但它们传统上依赖于对训练数据的访问，而在线强化学习中通常无法获得训练数据。为了解决这一局限，我们提出了不动点流匹配（FFM），这是一种在强化学习中训练和微调流策略的无数据方法。FFM利用目标流恒等式将actor更新表述为不动点迭代，采用流匹配式的回归目标，其中critic的动作梯度作为回归目标。我们证明最优策略是该迭代的不动点。通过大量实验，我们证明FFM能够有效微调预训练的流策略。

    arXiv:2610.12848v1 Announce Type: new  Abstract: In maximum entropy and behavior-constrained reinforcement learning (RL), the optimal policy is given by a Boltzmann distribution. Generating samples from this distribution can be challenging, as it is often complex and multimodal. While flow policies trained with flow matching provide a powerful framework for representing such distributions, they traditionally rely on access to training data, which is generally unavailable in online RL. To address this limitation, we propose fixed-point flow matching (FFM), a data-free procedure for training and fine-tuning flow policies in RL. FFM leverages the target flow identity to formulate the actor update as a fixed-point iteration, utilizing a flow-matching-style regression objective where the critic's action gradients serve as regression targets. We show that the optimal policy is a fixed point of this iteration. Through extensive experiments, we demonstrate that FFM effectively fine-tunes pretr
    
[^186]: 当生成模型充当模拟器时

    When Generative Models Act as Simulators

    [https://arxiv.org/abs/2610.12847](https://arxiv.org/abs/2610.12847)

    该论文研究生成模型通过“可编辑性”充当模拟器的条件，提出两种在模型潜在状态与模拟器状态之间学习映射的方法（探针解码加引导、逆映射），并发现逆映射方法即使在引导失败时也能成功。

    

    模拟器将环境总结为一个用户可以读取和修改的状态。生成模型无需显式指定即可学习世界的表示，但没有提供编辑该世界状态的内置方法。我们研究了生成模型在何种条件下可以通过“可编辑性”充当模拟器：即改变模型的潜在表示是否能产生与编辑模拟器状态后相同的下一步输出。简而言之，我们的目标是在生成模型的潜在状态与模拟器状态之间找到一种映射，其中模型仅通过观测数据进行训练，而映射则是在此之后学习的。我们测试了两种映射方法，两者都是从与模拟器状态配对的观测数据中学习得到的。第一种方法是训练探针从模型中解码出模拟器状态，然后沿着这些方向进行引导；第二种方法是通过逆映射直接学习该映射，这种方法即使在引导失败的情况下也能取得成功。

    arXiv:2610.12847v1 Announce Type: cross  Abstract: A simulator summarizes an environment in a state that users can read and change. Generative models learn representations of the world without explicit specification, but offer no built-in way to edit that world's state. We investigate the conditions under which a generative model can act as a simulator through $\textit{editability}$: whether changing a model's latent representation produces the next-step outputs we get by editing a simulator's state. Plainly stated, our goal is to find a mapping between a generative model's latent state and the simulator state, where the model is trained on observations alone and the mapping is learned afterward. We test two kinds of mappings, both learned from observations paired with simulator states. In one, we train probes to decode the simulator state from the model, then steer along them. In the other, we learn the mapping directly with an inverse map, which can succeed even where steering fails.
    
[^187]: APEX：面向极致压缩的异步前缀去噪

    APEX: Asynchronous Prefix Denoising for Extreme Compression

    [https://arxiv.org/abs/2610.12843](https://arxiv.org/abs/2610.12843)

    APEX通过将分层潜在空间与按通道组的异步前缀去噪相结合，解决了极致潜在压缩中的“重建-生成”两难困境，在64倍和128倍压缩下全面超越现有方法。

    

    更少的潜在标记意味着更低成本的扩散。然而，激进的潜在压缩带来了一个“重建-生成”的两难困境：增加通道数可以改善重建质量，却使生成变得更困难。我们提出了APEX（Asynchronous Prefix denoising for EXtreme compression，面向极致压缩的异步前缀去噪），它将分层潜在空间与按通道组的异步去噪相结合。我们的自动编码器将通道前缀按照从结构到细节的顺序排列，同时由一个共享的去噪器在错开的局部时间生成它们，且无需增加标记数量。在ImageNet-512上，在64倍和128倍压缩下，APEX在每个评估的通道数上都优于现有方法。在匹配的训练和引导采样计算量下，它在每个评估的计算预算下也优于256标记模型，且在较低计算量下增益更大。代码可在 https://daviddinkevich.github.io/APEX 获取。

    arXiv:2610.12843v1 Announce Type: cross  Abstract: Fewer latent tokens promise cheaper diffusion. Yet aggressive latent compression creates a reconstruction-generation dilemma: adding channels improves reconstruction but makes generation harder. We introduce APEX, short for Asynchronous Prefix denoising for EXtreme compression, which pairs a hierarchical latent space with asynchronous denoising over channel groups. Our autoencoder orders channel prefixes from structure to detail, while one shared denoiser generates them at staggered local times without adding tokens. On ImageNet-512, APEX outperforms existing methods at every evaluated channel count under 64x and 128x compression. At matched training and guided sampling compute, it also outperforms a 256-token model at every evaluated budget, with larger gains at lower compute. Code is available at https://daviddinkevich.github.io/APEX.
    
[^188]: COFFEQA：面向大语言模型问答的反事实引导少样本蒸馏

    COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering

    [https://arxiv.org/abs/2610.12837](https://arxiv.org/abs/2610.12837)

    提出COFFEQA方法，通过反事实引导为少样本蒸馏中教师决策边界附近的不确定样本提供额外监督，从而提升大语言模型问答任务的蒸馏效果。

    

    知识蒸馏旨在训练一个较小的学生模型来模仿较大的教师模型，以适应资源受限的环境。然而，蒸馏过程通常数据密集且计算成本高昂，尤其是对于生成式任务。少样本蒸馏提供了一种替代方案，即使用有限的训练样本来迁移教师模型的知识。然而，我们观察到少样本蒸馏的性能在很大程度上取决于这些训练样本的选择。事实上，我们证明了使用教师决策边界附近更不确定的样本会显著恶化蒸馏性能。基于这一观察，我们提出了开放书本问答中的反事实引导少样本蒸馏（COFFEQA），这是一种新颖的策略，巧妙地利用反事实为具有挑战性的事实样本提供额外的监督。在生成式问答设置中，我们定义了一个反事实……

    arXiv:2610.12837v1 Announce Type: cross  Abstract: Knowledge distillation aims to train a smaller student model to mimic a larger teacher model for resource-constrained environments. However, the process of distillation is typically data-intensive and computationally expensive, particularly for generative tasks. Few-shot distillation offers an alternative by transferring teacher knowledge using limited training samples. However, we observe that few-shot distillation performance is heavily dependent on the choice of these training samples. In fact, we demonstrate that using more uncertain samples near the teacher's decision boundary can significantly deteriorate distillation performance. Motivated by this observation, we propose Counterfactual-guided Few-shot Distillation in Open-book Question Answering (COFFEQA), a novel strategy that aptly uses counterfactuals to provide additional supervision around challenging factual samples. In the generative QA setting, we define a counterfactual
    
[^189]: 跨行为策略优化：通过共享回归目标从离线演示与在策略及离策略采样中实现统一学习

    Cross-Behavior Policy Optimization: Unified Learning from Offline Demonstrations and On- and Off-Policy Rollouts via a Shared Regression Target

    [https://arxiv.org/abs/2610.12830](https://arxiv.org/abs/2610.12830)

    CBPO通过固定参考的共享回归目标，将离线演示、在策略与离策略采样统一到同一回归损失和回放缓冲区中，无需分别协调SFT与RL目标、重要性加权或评论家网络即可实现数据的持续复用与统一学习。

    

    推理后训练依赖离线演示数据和策略生成的采样数据，然而传统的先监督微调（SFT）后强化学习（RL）的方法将两者的目标分离，并限制了已收集回复的持续复用。我们提出跨行为策略优化（CBPO），通过固定参考目标使这些数据源能够兼容于同一种回归损失。由于生成策略不进入该目标，正负离线补全、参考回复以及当前与历史采样数据可以共享统一的奖励标注记录格式和经验回放缓冲区。随着新回复进入缓冲区，优化始终保持相同的回归目标，无需协调分离的SFT和RL目标，也无需存储行为策略概率、重要性加权或学习评论家网络。通过限制可用的数据源，可以得到离线、在策略和离策略等变体。

    arXiv:2610.12830v1 Announce Type: new  Abstract: Reasoning post-training draws on offline demonstrations and policy-generated rollouts, yet conventional supervised fine-tuning (SFT) followed by reinforcement learning (RL) separates their objectives and limits continued reuse of collected responses. We propose \textbf{Cross-Behavior Policy Optimization} (CBPO), which makes these sources compatible with one regression loss through a fixed-reference target. Because the generating policy does not enter this target, positive and negative offline completions, reference responses, and current and historical rollouts can share a common reward-labeled record format and replay buffer. As new responses enter the buffer, optimization retains the same regression objective without coordinating separate SFT and RL objectives or requiring stored behavior-policy probabilities, importance weighting, or a learned critic. Restricting the available sources gives offline, on-policy, and off-policy variants,
    
[^190]: 具有核结构交互的可扩展行列式点过程学习

    Scalable Learning of Determinantal Point Processes with Kernel-Structured Interactions

    [https://arxiv.org/abs/2610.12829](https://arxiv.org/abs/2610.12829)

    该论文提出一种利用非线性核随机特征进行随机优化的可扩展行列式点过程学习方法，通过全秩交互建模与替代函数设计，突破了结构约束的局限，显著改善了表达能力与计算可扩展性之间的权衡，并以极高的参数效率获得了更好的泛化性能。

    

    有限集合上的行列式点过程（DPP）是一种展现排斥性的概率模型，其中项目间的相似性由一个正定核矩阵编码。将DPP学习扩展到大规模数据集通常需要对核矩阵施加结构约束，例如低秩加对角形式。然而，这种结构性假设在表达能力与计算可扩展性之间引入了严重的权衡。我们的研究通过以下方式应对这一挑战：（a）通过非线性核函数和潜在特征对全秩交互进行建模；（b）为最大似然估计器设计替代函数；（c）采用基于非线性核随机特征的随机优化方法。在真实世界数据集上的数值实验表明，我们的方法显著改善了这一权衡，以卓越的参数效率实现了更好的泛化性能。

    arXiv:2610.12829v1 Announce Type: cross  Abstract: Determinantal point processes (DPPs) on a finite set are probability models that exhibit repulsion where inter-item similarities are encoded by a positive definite kernel matrix. Scaling DPP learning to large datasets typically requires imposing structural constraints on the kernel matrix, such as a low-rank-plus-diagonal form. However, structural assumptions introduce a severe trade-off between expressive power and computational scalability. Our study addresses this challenge by (a) modeling the full-rank interactions via a nonlinear kernel function and latent features, (b) designing a surrogate function for the maximum likelihood estimator, and (c) adopting stochastic optimization using random features of the nonlinear kernel. Numerical experiments on real-world datasets demonstrate that our method significantly improves this trade-off, achieving better generalization performance with remarkable parameter efficiency.
    
[^191]: TreeHCA：面向多轮LLM智能体的后见之明引导探索与信用分配

    TreeHCA: Hindsight-Guided Exploration and Credit Assignment for Multi-Turn LLM Agents

    [https://arxiv.org/abs/2610.12828](https://arxiv.org/abs/2610.12828)

    TreeHCA利用参考答案似然的变化引导探索，构建推理与工具交互树，并通过逆后见之明重要性校正和归一化树回备将终端结果传播为轮级优势，为多轮LLM智能体实现细粒度的后见之明信用分配，克服了GRPO对所有token分配相同优势的缺陷。

    

    基于结果监督的组相对策略优化（GRPO）为LLM智能体轨迹中的每个token分配相同的优势值，导致有用和无用的决策无法被区分。后见之明信用分配（HCA）可以细化这种反馈，但其估计依赖于在关键历史节点处包含信息性备选方案的轨迹。我们提出TreeHCA，将后见之明引导的探索与后见之明信用分配相结合。TreeHCA利用参考答案似然的变化来回访中间历史节点并采样替代性延续，从而构建一棵由推理和工具交互组成的树。由于这种探索改变了采样延续的分布，我们推导了相应的逆后见之明重要性校正，其中包括一个精确的有限池恒等式。我们将这一结构用于归一化的树回备中，将终端结果传播为轮级优势。这种共享的后见之明结构……（摘要在此处截断）

    arXiv:2610.12828v1 Announce Type: new  Abstract: Outcome-supervised group-relative policy optimization (GRPO) assigns the same advantage to every token in an LLM agent trajectory, leaving useful and unhelpful decisions undifferentiated. Hindsight credit assignment (HCA) can refine this feedback, but its estimates rely on trajectories containing informative alternatives at consequential histories. We introduce TreeHCA, which couples hindsight-guided exploration with hindsight credit assignment. TreeHCA uses changes in reference-answer likelihood to revisit intermediate histories and sample alternative continuations, forming a tree of reasoning and tool interactions. Because this exploration changes the distribution of sampled continuations, we derive the corresponding inverse-hindsight importance correction, including an exact finite-pool identity. We use this structure in a normalized tree backup that propagates terminal outcomes into turn-level advantages. This shared hindsight struct
    
[^192]: 使用条件变分自编码器集成从冲击响应谱逆向合成冲击时间序列

    Inverse Synthesis of Shock Time Series from Shock Response Spectra Using Ensembles of Conditional Variational Autoencoders

    [https://arxiv.org/abs/2610.12820](https://arxiv.org/abs/2610.12820)

    该论文提出一种基于条件变分自编码器集成（涵盖不同小波母函数与分解层级）的逆向合成方法，通过循环解码器预测多尺度小波系数重构波形，为同一目标冲击响应谱生成多个候选冲击波形，其谱误差低于先前的直接时域CVAE模型。

    

    冲击响应谱（SRS）表征单自由度系统对瞬态加速度的峰值响应。由于从加速度时间序列（波形）到其冲击响应谱的映射是多对一的，逆向合成旨在寻找与给定谱一致的候选波形。我们扩展了先前的条件变分自编码器（CVAE）框架，采用循环解码器预测多尺度小波系数，并通过逆离散小波变换重构波形。由于单个模型在潜变量采样下表现出的变异性有限，我们构建了由独立训练的模型组成的集成，这些模型涵盖不同的小波母函数和分解层级。每个成员模型为同一目标冲击响应谱生成一个独立的候选波形。在四个留出的基准数据集上的评估表明，所展示的小波模型的平均和中位数谱误差低于先前的直接时域CVAE模型。

    arXiv:2610.12820v1 Announce Type: new  Abstract: The shock response spectrum (SRS) characterizes the peak responses of single-degree-of-freedom systems to transient acceleration. Because the mapping from an acceleration time series (waveform) to its SRS is many-to-one, inverse synthesis seeks candidate waveforms consistent with a prescribed spectrum. We extend an earlier conditional variational autoencoder (CVAE) framework with a recurrent decoder that predicts multiscale wavelet coefficients and reconstructs waveforms through an inverse discrete wavelet transform. Individual models exhibit limited variability under latent sampling, motivating an ensemble of independently trained models spanning different mother wavelets and decomposition levels. Each member produces a separate candidate waveform for the same target SRS. Evaluation on four held-out benchmark datasets shows lower mean and median spectral errors for the illustrated wavelet model than for the earlier direct time-domain CV
    
[^193]: 在大规模推理与训练工作负载中部署 RDMA

    Deploying RDMA for Large-Scale Inference and Training Workloads

    [https://arxiv.org/abs/2610.12815](https://arxiv.org/abs/2610.12815)

    本文总结了字节跳动在大规模推理与训练系统中部署基于 RDMA 的软件通信层 ByteExpress 的三条经验：高效 RDMA 需要跨应用与传输层的缓冲区管理、可扩展通信需要对连接与接收容量和在途数据的协同管理、服务可用性需要在传输层之上实现故障恢复。

    

    RDMA 可以降低分布式机器学习系统中的通信开销，但在生产环境中实现这些收益并非简单地用 RDMA 替换 TCP。可变大小的消息、并发连接、突发性的 incast 以及服务故障，带来了仅靠硬件卸载无法解决的挑战。本文报告了我们在字节跳动基于参数服务器的推理与训练系统中部署 ByteExpress——一种构建在硬件 RDMA 之上的软件通信层——的实践经验。我们总结出部署中的三条经验教训：高效的 RDMA 需要跨应用层与传输层的缓冲区管理；可扩展的通信需要对连接、接收容量和在途数据进行协同管理；服务可用性需要在传输层之上实现故障恢复。ByteExpress 结合了 RDMA 感知的内存分配器、零拷贝传输、QP 多路复用、接收信用流控、软件发送窗口以及 RPC 辅助的恢复机制（摘要在此处截断）。

    arXiv:2610.12815v1 Announce Type: new  Abstract: RDMA can reduce communication overhead in distributed machine learning systems, but realizing these benefits in production requires more than replacing TCP. Variable-size messages, concurrent connections, bursty incasts, and service failures introduce challenges that hardware offload alone cannot address. This paper reports our experience deploying ByteExpress, a software communication layer over hardware RDMA, in ByteDance's parameter-server-based inference and training systems. We present three lessons from the deployment: efficient RDMA requires buffer management across application and transport layers; scalable communication requires coordinated management of connections, receive capacity, and outstanding data; and service availability requires recovery above the transport layer. ByteExpress combines an RDMA-aware allocator, zero-copy transfers, QP multiplexing, receive-credit flow control, a software send window, and RPC-assisted re
    
[^194]: VEPA：面向终身LLM智能体防护栏的经过验证的演化形式化策略

    VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails

    [https://arxiv.org/abs/2610.12813](https://arxiv.org/abs/2610.12813)

    提出VEPA框架，将形式化防护栏的确定性执行能力与从部署经验中持续学习和演化安全策略的机制相结合，为LLM智能体提供终身的、可验证的自适应安全防护。

    

    LLM智能体越来越多地运行于敏感资源之上，例如私人文档和可变的系统状态，这可能带来严重的安全风险。因此，运行时防护栏应在潜在不可逆的智能体动作执行之前，做出可检查、可复现且确定性的允许或拒绝决策。然而，安全需求在部署之前很少是完整的，因此防护栏需要随着新风险的出现而不断适应。现有的自适应防护栏从部署经验中学习，但它们通常通过LLM推理来应用安全策略，这可能导致决策不一致和策略解释模糊。另一方面，形式化防护栏以确定性方式执行策略，但它们通常依赖于部署前或特定任务的策略，并且缺乏适应新观察到的风险的机制。我们提出了VEPA，一个用于形式化智能体防护栏经过验证演化的框架……

    arXiv:2610.12813v1 Announce Type: new  Abstract: LLM agents increasingly operate over sensitive resources, such as private documents and mutable system states, which can create serious safety and security risks. Runtime guardrails should therefore make inspectable, reproducible, and deterministic allow or refuse decisions before potentially irreversible agent actions are executed. However, safety and security requirements are rarely complete before deployment, so guardrails need to keep adapting as new risks emerge. Existing adaptive guardrails learn from deployment experience, but they usually apply safety policies through LLM inference, which can lead to inconsistent decisions and ambiguous policy interpretations. Formal guardrails, on the other hand, enforce policies deterministically, but they usually rely on pre-deployment or task-specific policies and have no mechanism to adapt to newly observed risks. We introduce VEPA, a framework for the verified evolution of formal agent guar
    
[^195]: 随机特征MPNN在不同图规模上的表达能力

    The Expressive Power of Random-Feature MPNNs Across Graph Sizes

    [https://arxiv.org/abs/2610.12809](https://arxiv.org/abs/2610.12809)

    该论文提出了一个总体层面的规模一致逼近分析框架，通过将连续随机特征量化为有限种颜色，证明了具有图规模无关Lipschitz界的MPNN不具备跨图规模的通用逼近能力。

    

    连续随机节点特征可以提升消息传递神经网络的表达能力，但现有的理论分析还不够精细，无法刻画其局限性，尤其是在跨图规模逼近函数或在固定数量随机比特条件下的情形。我们建立了一个用于规模一致逼近的总体层面分析框架。对于初始特征具有与图规模无关的Lipschitz界的MPNN（包括标准的均值、最大值以及阶归一化和等架构），我们证明连续随机特征可以被量化为有限多种颜色，同时使期望输出上的均匀误差任意小。这一约简使我们能够将有限颜色可区分性上的限制转化为连续随机特征下规模一致逼近的限制。我们利用该框架证明了这类MPNN不具备规模一致的通用性，并分析了颜色和独立随机特征所带来的表达能力提升。

    arXiv:2610.12809v1 Announce Type: new  Abstract: Continuous random node features increase the expressivity of message-passing neural networks (MPNNs), but existing theoretical analyses are not fine-grained enough to capture their limitations, particularly in approximating functions across graph sizes or under a fixed number of random bits. We develop a population-level framework for size-uniform approximation. For MPNNs with a graph-size-independent Lipschitz bound in their initial features, including standard mean, max, and order-normalized sum architectures, we show that continuous random features can be quantized to finitely many colors with arbitrarily small uniform error in expected output. This reduction lets us transfer limitations on finite-color distinguishability to limitations on size-uniform approximation with continuous random features. We use this framework to prove that these MPNNs are not size-uniformly universal, analyze the expressivity gains from colors and independe
    
[^196]: 可训练的原生量子语言模型：耗散记忆、容量极限与硬件验证

    Trainable Quantum-Native Language Models: Dissipative Memory, Capacity Limits, and Hardware Validation

    [https://arxiv.org/abs/2610.12806](https://arxiv.org/abs/2610.12806)

    该论文提出了以小型量子寄存器为隐藏状态、通过可学习词元条件量子信道转移并直接由玻恩规则测量输出概率的量子原生语言模型，发现可学习的词元条件耗散门控是实现超越单词元有用记忆的关键，其性能优于参数量匹配的单词元对照模型但不及双词元对照模型，从而揭示了此类量子语言模型的容量极限并通过硬件实验加以验证。

    

    我们研究了一种字节级自回归语言模型，其隐藏状态是一个小型量子寄存器，其状态转移是通过学习得到的、以词元为条件的量子信道，其下一字节概率直接来自玻恩规则测量，而无需学习经典词表投影或softmax层。在固定宽度的循环设置中，幺正动力学无法收缩或重置已存储的区分信息；实验表明，幺正循环在所有测试变体中均不敌经过重新训练的单词元对照模型。引入可学习的耗散，进而引入以词元为条件的耗散门控，使模型产生了超越单个词元的有用记忆。在上下文长度为12时，门控模型相比参数量匹配的重新训练单词元对照模型，将测试负对数似然改善了0.20 ± 0.03 nats/byte；在更长的上下文中改善幅度为0.29–0.33，但始终无法超过重新训练的双词元对照模型。键值召回基准测试显示出依赖于宽度的阈值：一个存储的键值对满足p……（原文摘要在此处截断）

    arXiv:2610.12806v1 Announce Type: cross  Abstract: We study byte-level autoregressive language models whose hidden state is a small quantum register, whose transitions are learned token-conditioned quantum channels, and whose next-byte probabilities come directly from Born-rule measurements without a learned classical vocabulary projection or softmax. In the fixed-width recurrent setting, unitary dynamics cannot contract or reset stored distinctions; experimentally, unitary recurrence loses to its retrained one-token control in every tested variant. Introducing learned dissipation, then token-conditioned dissipative gating, yields useful memory beyond one token. The gated model improves test negative log-likelihood by 0.20 +/- 0.03 nats/byte over retrained parameter-matched one-token controls at context length 12, and by 0.29-0.33 at longer contexts, but never beats retrained two-token controls. A key-value recall benchmark shows a width-dependent threshold: one stored pair meets the p
    
[^197]: 基于学习型配对方向特征的多凸神经网络代理模型用于各向异性超弹性均匀化

    Polyconvex neural-network surrogates with learned paired directional features for anisotropic hyperelastic homogenization

    [https://arxiv.org/abs/2610.12801](https://arxiv.org/abs/2610.12801)

    提出了一种具有可学习配对方向特征的多凸物理增强神经网络，能够在构造上保证客观性、多凸性与增长条件等力学一致性的前提下，高效逼近各向异性超弹性均匀化的有效响应。

    

    逼近代表性体积元（RVE）的有效各向异性超弹性响应，需要本构表达能力、力学一致性和计算效率。本工作开发了一种基于配对方向特征的多凸物理增强神经网络（PANN）构造，其方向、拉伸幂次和行列式指数能够自适应于RVE的响应。这些特征与凸的、逐分量非递减的核心函数组合，实现为输入凸神经网络（ICNN）或输入凸Kolmogorov-Arnold网络（ICKAN）。完整的能量函数在构造上即满足客观性、精确的无应力参考构型、二维多凸性以及规定的增长条件，并对所选模型另行证明了能量非负。在多空腔和单空腔周期性单胞上，将这些能量与以相同方式训练的无约束能量模型进行比较（摘要在此处截断）。

    arXiv:2610.12801v1 Announce Type: cross  Abstract: Approximating the effective anisotropic hyperelastic response of a representative volume element (RVE) requires constitutive expressiveness, mechanical consistency, and computational efficiency. This work develops a polyconvex physics-augmented neural network (PANN) construction based on paired directional features whose orientations, stretch powers, and determinant exponents adapt to the RVE response. The features are composed with convex, componentwise nondecreasing cores, realized as an input-convex neural network (ICNN) or an input-convex Kolmogorov-Arnold network (ICKAN). The complete energies satisfy objectivity, an exactly stress-free reference, two-dimensional polyconvexity, and prescribed growth conditions by construction, and nonnegative energy is certified separately for the selected models. On a multicavity and a single-cavity periodic cell, these energies are compared with an Unconstrained energy, trained identically witho
    
[^198]: 元反思式进化有效改进生产级AI系统

    Meta-Reflective Evolution Effectively Improves Production AI Systems

    [https://arxiv.org/abs/2610.12794](https://arxiv.org/abs/2610.12794)

    提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。

    

    arXiv:2610.12794v1 公告类型：交叉 摘要：由大语言模型（LLM）引导的进化优化已成功改进了提示词、程序，乃至整个AI系统。大多数优化方法从搜索前沿中选取一个候选，然后仅基于其失败情况，在单次LLM调用中对其进行变异。这种对搜索前沿的关注在生产级AI系统上可能陷入停滞，因为孤立的失败很难揭示复杂的隐藏交互。我们提出了CoCoEvolve（Cortex Code Evolve），一个围绕我们称之为“元反思”（meta-reflection）机制构建的进化优化器，其中智能体变异器（Cortex Code）会选择性地从整个进化历史中检索证据，并在验证和提出变异之前探测实际运行的系统。我们将CoCoEvolve应用于优化三类生产级AI系统：数据智能体、基于LLM的数据流水线以及数据工程工作流。在DABStep上，CoCoEvolve登顶公开排行榜，在留出难题集上达到92.8%的准确率，相比之下基线仅为78.4%（摘要原文在此处截断）。

    arXiv:2610.12794v1 Announce Type: cross  Abstract: Large Language Model (LLM)-guided evolutionary optimization has successfully improved prompts, programs, and even entire AI systems. Most optimization methods select a candidate from the search frontier, then mutate it in one LLM pass based only on its failures. This focus on the frontier can stall on production AI systems, where isolated failures reveal little about complex, hidden interactions. We introduce CoCoEvolve (Cortex Code Evolve), an evolutionary optimizer built around what we call meta-reflection, wherein an agentic mutator (Cortex Code) selectively retrieves evidence from the entire evolutionary history and probes the live system before verifying and proposing mutations. We apply CoCoEvolve to optimize three classes of production AI systems: data agents, LLM-powered data pipelines, and data engineering workflows. On DABStep, CoCoEvolve tops the public leaderboard, reaching 92.8% held-out hard accuracy compared to 78.4% for
    
[^199]: 坐标无关的自动微分

    Coordinate-Free Automatic Differentiation

    [https://arxiv.org/abs/2610.12776](https://arxiv.org/abs/2610.12776)

    该论文提出一个与坐标（基）无关的统一导数算子，可完全由现有自动微分原语实现，从而以可复用、可组合的方式表达旋转、体积形式、曲率等几何微分运算，并与 JAX 的批处理、编译和硬件加速兼容。

    

    自动微分（AD）是现代机器学习的基石。由于所有自动微分框架都隐式地固定了欧几里得空间的一组基来定义梯度和雅可比矩阵，它们天然处理的是由一组固定坐标构成的数组型数学对象。这使得表达旋转、体积形式和曲率等对象变得十分笨拙，而这类对象恰恰出现在诸如人工智能应用于科学的场景中。为此定制的变通方法缺乏可复用性。我们证明：(i) 单一的、与基无关的导数算子可以表示多种几何微分运算；(ii) 该算子可以完全用现有的自动微分原语来表达。这种可复用的变换能够与 JAX 的批处理、编译以及硬件加速相组合。几何以代数数据的形式进入系统。我们通过实验演示了该算子在不同流形算子之间的复用、在时空反问题处理流水线中的组合运用，以及……（原文摘要在此处截断）

    arXiv:2610.12776v1 Announce Type: new  Abstract: Automatic differentiation (AD) is a cornerstone of modern machine learning. Since all AD frameworks implicitly fix a basis of Euclidean space to define gradients and Jacobians, they natively process mathematical objects that are arrays of a fixed set of coordinates. This makes it awkward to express objects such as rotations, volume forms, and curvatures that arise, for example, in applications of AI to science. Bespoke workarounds lack reusability. We show that (i) a single, basis-independent derivative operator can represent many geometric differential operations, and (ii) this operator can be expressed entirely in terms of existing AD primitives. This reusable transformation can be composed with JAX batching, compilation, and hardware acceleration. Geometry enters as algebraic data. We experimentally demonstrate its reuse across different manifold operators, its composition through a spacetime inverse-problem pipeline, and its use in o
    
[^200]: 使用多元霍克斯过程在带时间戳的足球网络中进行贡献归因

    Credit Attribution in Timestamped Soccer Networks Using Multivariate Hawkes Processes

    [https://arxiv.org/abs/2610.12775](https://arxiv.org/abs/2610.12775)

    该论文提出利用多元霍克斯过程对足球比赛中带时间戳的传球和射门序列建模，量化每次传球对后续进攻事件的贡献，从而将进球功劳合理分配给进攻组织中的多名球员，弥补了传统助攻统计的局限性。

    

    足球中两种主要的传统球员统计数据是进球和助攻，当一名球员的传球直接导致队友进球时，该球员获得助攻。然而，仅将功劳归于进球前最后一次传球的球员，无法体现多名球员在进球组织过程中的贡献。因此，传统统计数据只能有限地反映球员的参与程度。在本文中，我们使用多元霍克斯过程（MHP）对带时间戳的传球和射门序列进行建模，这些序列构成连续时间网络。MHP 提供了一种分支机制，其中一个事件（传球或射门）可以作为前一个事件的子事件生成，这使我们能够量化每次传球对后续传球或射门事件的贡献。这种方法将功劳分配到进攻序列中的多次传球上，使其能够适用于广泛的场景。基于这一框架，我们……

    arXiv:2610.12775v1 Announce Type: cross  Abstract: The two main traditional player statistics used in soccer are goals scored and assists, with an assist credited when a player's pass immediately leads to a teammate scoring. Attributing credit solely to the player who makes the final pass before a goal fails to account for the contributions of multiple players in the buildup to the goal. Consequently, traditional statistics provide only a limited view of player involvement. In this paper, we use multivariate Hawkes processes (MHPs) to model timestamped sequences of passes and shots, which form continuous-time networks. MHPs provide a branching mechanism where an event (a pass or shot) can be generated as a child of a previous event, which allows us to quantify each pass's contribution to a subsequent pass or shot. This approach distributes credit across multiple passes within an attacking sequence, making it applicable across a wide range of situations. Building on this framework, we p
    
[^201]: REDaCT：大型推理模型中的无坍塌遗忘学习

    REDaCT: Collapse-Free Unlearning in Large Reasoning Models

    [https://arxiv.org/abs/2610.12769](https://arxiv.org/abs/2610.12769)

    REDaCT是一种面向大型推理模型的推理时遗忘学习框架，通过泄漏检测器与纠错嵌入注入机制，在不引发推理坍塌的前提下从思维链和最终答案中移除遗忘集信息。

    

    大型推理模型（LRM）中的遗忘学习要求同时从推理轨迹和最终答案中移除遗忘集信息。现有方法在实现这一目标时以推理坍塌为代价，即模型无法生成连贯的思维链（CoT），也无法正确终止其推理过程。为应对这一挑战，我们提出了REDaCT（通过检测与纠错微调实现再生成），一种面向大型推理模型的推理时遗忘学习框架。REDaCT包含两个关键组件：（1）一个二元泄漏检测器，用于识别推理轨迹中泄露遗忘集信息的句子；（2）一个嵌入优化模块，离线学习一个纠错嵌入向量，在推理过程中首次检测到泄漏时将其注入，从而引导后续推理生成不泄露遗忘集信息的句子。实验结果表明，REDaCT完全避免了推理坍塌。

    arXiv:2610.12769v1 Announce Type: cross  Abstract: Unlearning in large reasoning models (LRMs) requires removing forget-set information from both the reasoning trace and the final answer. Existing approaches achieve this objective at the cost of reasoning collapse, where the model fails to produce a coherent chain-of-thought (CoT) and does not properly terminate its reasoning process. To address this challenge, we propose REDaCT (REgenerate via Detection and Corrective Tuning), an inference-time unlearning framework for LRMs. REDaCT consists of two key components: (1) a binary leak detector that identifies sentences in the reasoning trace that leak forget-set information, and (2) an embedding optimization module that learns a corrective embedding offline, such that when injected at the first detected leak during inference, it guides subsequent reasoning towards sentences that do not leak forget-set information. Experimental results show that REDaCT completely avoids reasoning collapse,
    
[^202]: 钙钛矿光伏中机器学习评估的来源分组协议

    Provenance-grouped protocol for evaluating machine learning in perovskite photovoltaics

    [https://arxiv.org/abs/2610.12766](https://arxiv.org/abs/2610.12766)

    该论文指出钙钛矿光伏机器学习评估中随机数据划分会因海量重复数据导致信息泄漏，提出基于机构来源分组的严格评估协议，并揭示树集成模型仅擅长记忆熟悉配方、而成分感知神经架构在真正未知场景下泛化能力更强的规律。

    

    用于钙钛矿太阳能电池的机器学习模型通常采用随机数据划分进行评估，我们证明了这种做法会因大规模的内部数据重复而系统性地泄漏信息。由于整理后的器件中近三分之二与另一个器件共享完全相同的制备配方，随机划分主要奖励的是对历史数据的简单记忆，而非检验真正的物理泛化能力。为解决这一问题，我们绘制了钙钛矿数据库的机构来源图谱，并建立了一套严格的分组划分层级，逐步隔离重复的器件、出版物和机构。对十三个模型的基准测试在所有划分方式上建立了新的最先进性能，并揭示了一个关键的性能格局转变：树集成模型凭借对熟悉配方的识别在随机划分中占据主导，但在面对真正未见过的机构和未来新发现时排名大幅下滑，而成分感知的神经架构在这种情况下表现更为出色。此外，

    arXiv:2610.12766v1 Announce Type: new  Abstract: Machine-learning models for perovskite solar cells are typically evaluated using random data splits, which we show systematically leak information through massive internal duplication. Because nearly two-thirds of curated devices share their exact recipe with another, random splits primarily reward the mere recall of historical data rather than testing true physical generalisation. To solve this, we mapped the institutional provenance of the Perovskite Database and established a rigorous hierarchy of grouped splits that progressively isolate duplicate devices, publications, and institutions. Benchmarking thirteen models establishes new state-of-the-art performance across all splits and reveals a critical regime shift: while tree ensembles dominate random splits by recognising familiar recipes, their ranking collapses on genuinely unseen institutions and future discoveries, where composition-aware neural architectures excel. Furthermore, 
    
[^203]: NV-Tesseract-AD：面向多元时间序列异常检测的课程增强条件扩散模型

    NV-Tesseract-AD: Curriculum-Enhanced Conditional Diffusion for Multivariate Time-Series Anomaly Detection

    [https://arxiv.org/abs/2610.12754](https://arxiv.org/abs/2610.12754)

    提出NV-Tesseract-AD框架，通过在掩码严重程度与扩散噪声水平两个互补轴上同时实施由易到难的三阶段双课程学习，增强了基于Transformer的条件扩散模型在多元时间序列异常检测中的优化稳定性与残差判别能力，并实现了无需逐数据集调优的严格零样本迁移。

    

    我们提出了NV-Tesseract-AD，一个用于多元时间序列异常检测的课程学习增强的条件扩散框架。我们的核心假设是：重建难度可以沿着两个互补的轴（掩码严重程度和扩散噪声水平）进行显式结构化，并且同时沿这两个轴从易到难进行训练，可以稳定优化过程并提高残差的可辨别性。我们将基于Transformer的扩散骨干网络与覆盖掩码比例和扩散时间步的三阶段双课程学习相结合，在整个训练过程中保留异常样本，并依靠结构化破坏来驱动自监督表示学习。我们在两种设置下评估NV-Tesseract-AD：在TSB-UAD-M语料库上进行分布内训练，以及在不进行每个数据集调优的情况下，严格零样本迁移到十个EasyTSAD基准数据族。双轴课程产生了掩码不变的重建（摘要在此处截断）

    arXiv:2610.12754v1 Announce Type: new  Abstract: We propose NV-Tesseract-AD, a curriculum learning-enhanced conditional diffusion framework for multivariate time-series anomaly detection. Our central hypothesis is that reconstruction difficulty can be explicitly structured along two complementary axes, masking severity and diffusion noise level, and that training from easier to harder tasks along both axes simultaneously stabilizes optimization and improves residual discriminability. We integrate a Transformer-based diffusion backbone with a three-stage dual curriculum over masking ratios and diffusion timesteps, retaining anomalous samples throughout training and relying on structured corruption to drive self-supervised representation learning. We evaluate NV-Tesseract-AD under two regimes: in-distribution training on the TSB-UAD-M corpus and strict zero-shot transfer to ten EasyTSAD benchmark families without per-dataset tuning. The dual-axis curriculum produces masking-invariant rec
    
[^204]: 基于人在环路驾驶数据的道路施工区域自动驾驶轨迹预测

    Trajectory Prediction for Autonomous Driving in Road Construction Zones Using Human-in-the-Loop Driving Data

    [https://arxiv.org/abs/2610.12747](https://arxiv.org/abs/2610.12747)

    该论文构建了首个针对道路施工区域的多模态驾驶数据集WZDrive-Data，并提出了多模态自车轨迹预测器WZDrive，有效解决了施工区域车道布局临时变化导致的车辆轨迹难以预测的难题。

    

    arXiv:2610.12747v1 公告类型：cross 摘要：在城市道路施工区域中进行自车轨迹预测仍然是自动驾驶研究中一个未被充分探索的课题，专用数据和系统性评估都很有限。道路施工区域涉及车道布局和道路连接性的临时变化，使与周围车辆和工人的交互变得复杂，这些变化使得未来的车辆轨迹难以预测。为了应对这些挑战，我们收集了WZDrive-Data，这是一个多模态数据集，包含来自33名驾驶员的1,183个施工区域驾驶片段，使用带有预录制工人动画和VR控制工人的人在环路驾驶模拟器进行采集。驾驶员控制和注视信号从人类参与者处记录，而车辆轨迹和场景信息从模拟器中获取。我们提出的多模态自车轨迹预测器WZDrive实现了平均minADE (K=6)为1.09米和minFDE (K=6)为2.12米的性能。与表现最好的（摘要在此处被截断）

    arXiv:2610.12747v1 Announce Type: cross  Abstract: Ego-trajectory prediction in urban roadway work zones remains an underexplored topic in autonomous-driving research, with limited dedicated data and systematic evaluation. Roadway work zones involve temporary changes in the lane layouts and connectivity of the roads, complicating interactions with surrounding vehicles and workers. These changes make future vehicle trajectories difficult to predict. To address these challenges, we collect WZDrive-Data, a multimodal dataset comprising 1,183 work-zone driving segments from 33 drivers, using a human-in-the-loop driving simulator with prerecorded worker animations and VR-controlled workers. Driver control and gaze signals are recorded from human participants, while vehicle trajectories and scene information are obtained from the simulator. Our proposed multimodal ego-trajectory predictor, WZDrive, achieves a mean minADE (K=6) of 1.09 m and minFDE (K=6) of 2.12 m. Compared with the best-perf
    
[^205]: 静态梯度归因在LoRA B矩阵内的基于损失的遗忘上表现不如密度匹配的随机掩码

    Static Gradient Attribution Underperforms a Density-Matched Random Mask on Loss-Based Forgetting Within LoRA B-Matrix

    [https://arxiv.org/abs/2610.12744](https://arxiv.org/abs/2610.12744)

    该研究揭示了梯度掩码与随机掩码比较中存在的参数池混淆问题（零初始化LoRA下A的梯度为零），并发现即使限定在同一B矩阵参数池内进行公平比较，密度匹配的均匀随机掩码在基于损失的遗忘指标上仍始终优于静态梯度归因。

    

    参数高效持续学习通过更新受限的适配器参数子集来纳入顺序任务，同时限制灾难性遗忘。梯度归因是选择该子集的常见规则，但它需要对校准数据进行一次额外的反向传播。我们在标准零初始化LoRA（B=0）下，识别出初始化时的梯度掩码与无限制随机掩码之间比较中的一个参数池混淆因素：关于A的梯度恒为零（∂L/∂A=0），因此梯度掩码严格地从B矩阵中选择，而无限制随机掩码则同时从A和B中抽取。这样的比较同时改变了选择规则和候选参数池。通过将两种方法限制在相同的参数池内来隔离选择标准，我们发现了一个负面结果：均匀随机掩码在基于损失的后向迁移（遗忘）上始终实现更低的数值……（摘要原文在此处截断）

    arXiv:2610.12744v1 Announce Type: new  Abstract: Parameter-efficient continual learning updates a restricted subset of adapter parameters to incorporate sequential tasks while limiting catastrophic forgetting. Gradient attribution is a common rule for selecting that subset, but it requires an additional backward pass over calibration data. We identify a parameter-pool confound in comparisons between initialization-time gradient masks and unrestricted random masks under standard zero-initialized LoRA ($B=0$): the gradient with respect to $A$ is identically zero ($\partial\mathcal{L}/\partial A=0$), so gradient masks select strictly from the $B$ matrix whereas unrestricted random masks draw from both $A$ and $B$. Such a comparison changes both the selection rule and the candidate parameter pool. Isolating the selection criterion by restricting both approaches to the same parameter pool, we find a negative result: a uniform random mask consistently achieves lower loss-based backward trans
    
[^206]: 神经期望最大化与元模型

    Neural expectation-maximization and meta-models

    [https://arxiv.org/abs/2610.12741](https://arxiv.org/abs/2610.12741)

    该论文将Transformer解释为最大乘积环状信念传播与期望最大化混合概率推断的可微近似，并据此提出“元模型”这一新设计范式，由输入生成实例特定的参数。

    

    我们将Transformer网络解释为一种混合概率推断过程的学习近似，该过程结合了最大乘积环状信念传播（LBP）与期望最大化（EM）。在这种视角下，Transformer在部分可观测的马尔可夫随机场（MRF）中执行推断，其结构和参数并非先验给定，而是从数据中推断得到。我们证明自注意力机制在输入条件化的、动态结构的MRF上实现了对最大乘积LBP的可微近似，其中注意力权重编码了学习到的结构亲和性。我们进一步在深度层面建立了与EM的对应关系：每一层在类似E步的前向传播中更新潜在表示，而层间转移θ^(l) → θ^(l+1)则充当隐式的M步。基于这一解释，我们提出了元模型，这是一种设计范式，其中输入生成实例特定的参数……

    arXiv:2610.12741v1 Announce Type: new  Abstract: We interpret Transformer networks as learned approximations to a hybrid probabilistic-inference procedure that combines max-product Loopy Belief Propagation (LBP) with Expectation--Maximization (EM). In this view, a Transformer performs inference in partially observed Markov Random Fields (MRFs) whose structure and parameters are not given a priori but are inferred from data. We show that self-attention implements a differentiable approximation to max-product LBP on an input-conditioned, dynamically structured MRF, with attention weights encoding learned structural affinities. We further establish an EM correspondence at the level of depth: each layer updates latent representations in an E-step-like forward pass, while the transition $\theta^{(l)} \to \theta^{(l+1)}$ across layers acts as an implicit M-step. Building on this interpretation, we introduce \emph{meta-models}, a design paradigm in which inputs generate instance-specific para
    
[^207]: 幂次切片Wasserstein泛函估计的极小极大速率

    Minimax Rates for Estimating Powered Sliced Wasserstein Functionals

    [https://arxiv.org/abs/2610.12740](https://arxiv.org/abs/2610.12740)

    本文建立了幂次切片Wasserstein泛函两样本估计的极小极大速率——当 $r<2p$ 时为 $N^{-(r-p)/r}$ 阶，当 $r>2p$ 时为 $N^{-1/2}$ 阶——并证明自然的代入估计量在两种情形下均达到速率最优。

    

    我们研究了在投影 $r$ 阶矩假设下幂次切片Wasserstein泛函的两样本估计问题。令 $N=n\wedge m$，其中 $n$ 和 $m$ 为两个样本量。我们建立了非临界情形下的极小极大速率：当 $r<2p$ 时，极小极大风险为 $N^{-(r-p)/r}$ 阶；而当 $r>2p$ 时，为 $N^{-1/2}$ 阶。在这两种情形下，自然的代入估计量均达到极小极大速率最优。在端点 $r=p$ 处，极小极大风险具有不趋于零的下界，因此一致相合估计是不可能实现的。在临界阶 $r=2p$ 处，我们得到了代入估计量的精确最坏情形速率以及极小极大风险的下界，但上下界之间存在一个对数差距，该临界情形下的精确极小极大速率仍是悬而未决的问题。

    arXiv:2610.12740v1 Announce Type: cross  Abstract: We study the two-sample estimation of the powered Sliced Wasserstein functional under projected $r$-th moment assumptions. Let $N=n\wedge m$, where $n$ and $m$ are two sample sizes. We establish the minimax rates in the noncritical regimes. When $p<2p$, the minimax risk is of order $N^{-(r-p)/r}$, while for $r>2p$ it is of order $N^{-1/2}$. In both cases, the natural plug-in estimator is minimax rate-optimal. At the endpoint $r=p$, the minimax risk is bounded away from zero, so uniformly consistent estimation is impossible. At the critical order $r=2p$, we obtain the exact worst-case rate of the plug-in estimator and a lower bound for the minimax risk, leaving a logarithmic gap between the lower and upper bounds. The exact minimax rate in this critical case remains open.
    
[^208]: KITE：一种用于时间序列事件预测的知识注入型时间编码器

    KITE: A Knowledge-Injected Temporal Encoder for Time-Series Event Forecasting

    [https://arxiv.org/abs/2610.12738](https://arxiv.org/abs/2610.12738)

    该论文提出KITE架构，通过可切换的协变量和注意力偏置两种机制将工程师掌握的语义先验知识注入自监督事件预测器，无需架构搜索即可适配各类先验，在标签稀缺场景下显著提升罕见事件预测性能（域内最高提升47%、跨域提升53%）。

    

    在多元时间序列系统中预测罕见且高代价的事件十分困难，因为这些事件的前兆很少被观测到，因此纯粹的数据驱动模型几乎没有可学习的信息。然而，运维这些系统的工程师掌握着先于此类事件发生的知识——这些信息在稀缺的标签中并不存在。我们将其称为语义先验，并通过两种可切换的机制将其注入自监督事件预测器中：协变量机制和注意力偏置机制，每种机制在零状态下均为恒等映射，因此单一架构即可服务于所有类型的先验，无需针对每个数据集进行架构搜索。当先验能够提供观测窗口中尚未包含的信息时，它才能发挥作用：在短上下文情境下，信息丰富的先验在域内优于无先验基线（汇总h-AUROC相对无先验基线：周期性时钟提升14%，p<0.001；守恒残差在0.25标签比例下提升47%），且有效的先验可以跨域迁移（提升53%）……

    arXiv:2610.12738v1 Announce Type: cross  Abstract: Forecasting rare, high-cost events in multivariate time-series systems is hard because their precursors are seldom observed, so a purely data-driven model has little to learn from. The engineers who run these systems, however, hold knowledge that precedes such events, information absent from the scarce labels. We call this a semantic prior and inject it into a self-supervised event forecaster through two switchable mechanisms, a covariate mechanism and an attention-bias mechanism, each the identity at its null, so one architecture serves every prior type with no per-dataset architecture search. A prior helps when it supplies information the observed window does not already contain: in a short context an informative prior beats no prior in-domain (pooled h-AUROC over the no-prior baseline: a periodicity clock +14%, p<0.001; a conservation residual +47% at a 0.25 label fraction), and a working prior transfers out-of-domain (+53%), substi
    
[^209]: 面向摊销式条件可解释性的核化参数敏感性图谱

    Kernelized Parameter Sensitivity Atlases for Amortized Conditional Interpretability

    [https://arxiv.org/abs/2610.12737](https://arxiv.org/abs/2610.12737)

    该论文提出 KPSA，通过经验神经正切核的分块分解构建参数组敏感性的核化图谱，用轻量级核函数摊销式地为任意新输入估计敏感性，从而以远低于逐样本反向传播的成本实现输入条件化的模型可解释性。

    

    基于梯度的归因方法能够提供关于模型参数的输入特定信息，但通过反向传播为大量样本计算归因代价高昂，而静态重要性得分则丢弃了对输入的依赖性。我们从经验神经正切核的分块分解出发推导出 KPSA（核化参数敏感性图谱）。它度量归一化的参数组敏感性，将这些敏感性来源嵌入一个显式的查询空间，并使用轻量级的核函数为未见过的输入估计敏感性。我们推导了区分有限来源误差与不可约表示误差的理论界，并给出了基于间隔（margin）的 top-k 恢复保证。本文研究了两个可解释性应用。其一，KPSA 在视觉、时序预测和蛋白质建模等多种设置中识别出对扰动高度敏感的参数组。在参数预算跨越五倍范围（0.02%–0.1%）的情况下，其选定的参数组可达到扰动效果的 39%–85%（原文在此处截断）。

    arXiv:2610.12737v1 Announce Type: new  Abstract: Gradient-based attribution provides input-specific information about model parameters, but computing it via backpropagation for many samples is expensive, while static importance scores discard input dependence. We derive KPSA (Kernelized Parameter Sensitivity Atlas) from a block decomposition of the empirical neural tangent kernel. It measures normalized parameter-group sensitivity, embeds those sources in an explicit query space, and uses a lightweight kernel to estimate sensitivities for unseen inputs. We derive bounds that separate finite-source from irreducible representation error and give margin-based top-$k$ recovery. This paper studies two interpretability applications. Firstly, KPSA identifies parameter groups with high susceptibility to perturbations across vision, forecasting, and protein-modeling setups. Across a fivefold range of parameter budgets ($0.02$-$0.1\%$), its selected groups attain $39$-$85\%$ of the perturbation 
    
[^210]: StereoEngine：无需立体相机即可扩展立体双手机器人数据

    StereoEngine: Scaling Stereo Bimanual Robot Data without Stereo Cameras

    [https://arxiv.org/abs/2610.12734](https://arxiv.org/abs/2610.12734)

    StereoEngine通过“深度估计+重投影与视频扩散模型补全将单目演示提升为立体视频对”和“仿真立体渲染联合训练”两种互补方法合成逼真的机器人操作立体数据，无需立体相机即可大规模扩展双手机器人立体数据集。

    

    双目立体视觉在机器人操作中的应用日益增多，越来越多的研究将其融入视觉-语言-动作（VLA）模型中。然而，大规模真实世界立体数据集的稀缺仍然是扩展此类策略的根本瓶颈。近期的生成式单目转立体方法主要面向VR内容和自动驾驶等场景，此前尚无工作能够为机器人操作生成逼真的合成立体数据，而该领域的立体监督信号主要来自仿真。我们提出StereoEngine，引入两种互补的方法为机器人操作合成立体数据。在第一种方法中，我们利用估计的深度将单目（左目）视频重投影（warp）到右目视角，并使用视频扩散模型对去遮挡（disoccluded）区域进行修复补全，从而无需在真实立体数据上训练即可将单目演示提升为立体视频对。在第二种方法中，我们在仿真中渲染立体视频对，并联合……（原文摘要在此处截断）

    arXiv:2610.12734v1 Announce Type: cross  Abstract: Stereo vision is increasingly used for robotic manipulation, with a growing body of work incorporating it into Vision-Language-Action (VLA) models. However, the scarcity of large-scale, real-world stereo datasets remains a fundamental bottleneck for scaling such policies. Recent generative monocular-to-stereo methods target settings such as VR content and autonomous driving. No prior work produces realistic synthetic stereo data for robotic manipulation, where stereo supervision has come predominantly from simulation. We propose StereoEngine, which introduces two complementary methods to synthesize stereo data for robotic manipulation. In the first, we warp a monocular (left) video into the right view using estimated depth and inpaint the disoccluded regions with a video diffusion model, lifting monocular demonstrations into stereo pairs without training on real stereo. In the second, we render a stereo pair in simulation and jointly g
    
[^211]: 参数高效微调的范畴图册：锥、轨道与规范

    A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges

    [https://arxiv.org/abs/2610.12728](https://arxiv.org/abs/2610.12728)

    该论文提出一个范畴论统一框架，将数百种参数高效微调方法视为从小参数空间到权重空间的光滑映射，并通过像、纤维与基变换三个不变量统一刻画方法的表达能力、训练动力学以及量化、合并、重启等操作。

    

    参数高效微调（PEFT）已有数百种方法，通常通过参数量和基准测试分数来进行比较。我们将这些方法视为映射来比较：预训练检查点是权重空间中的一个基点，而权重空间方法是从一个小参数空间出发、以该点为起点的光滑映射。这些映射构成一个切片范畴，其中冻结模型是始对象，全量微调是终对象。该映射的三个不变量承担了大部分刻画工作：其在基点附近的像通过一阶像、维度、关于秩的最小割界，以及该点是光滑点还是顶点，决定了模型的表达能力；其纤维连同优化器的度量决定了训练动力学；基变换移动基点，并涵盖了初始化拆分、量化、重启与合并等操作。例如，我们证明零初始化的LoRA，以及在给定秩与奇偶性条件下从其默认起点出发的Householder反射适配，都从顶点开始……（摘要原文在此处被截断）

    arXiv:2610.12728v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning has hundreds of methods, usually compared by parameter count and benchmark score. We compare them as maps. A pretrained checkpoint is a base point in weight space, and a weight-space method is a smooth map from a small parameter space that starts at that point. These maps form a slice category in which the frozen model is initial and full fine-tuning terminal. Three invariants of the map do most of the work. Its image near the base point governs expressivity, through the first-order image, the dimension, a min-cut bound on rank and whether that point is smooth or an apex. Its fibres, with the optimizer's metric, govern training dynamics. Base change moves the base point and covers splitting initialisations, quantisation, restarts and merging. We prove, for example, that zero-initialised LoRA, and Householder reflection adaptation at its default start under stated rank and parity conditions, begin at the ap
    
[^212]: Zebra3D：面向可扩展三维重建的高效混合基础模型

    Zebra3D: Efficient Hybrid Foundation Models For Scalable 3D Reconstruction

    [https://arxiv.org/abs/2610.12727](https://arxiv.org/abs/2610.12727)

    Zebra3D提出一种逐层稀疏-线性混合注意力框架，通过为容忍的早期阶段分配线性注意力、为敏感的中后期阶段分配瓦片稀疏注意力，并利用闭式仿射拟合实现免训练转换，从而在无需从头训练的情况下显著提升预训练3D基础模型在大规模视图集上的重建效率。

    

    前馈式3D基础模型能够从无位姿图像中重建场景，但全局注意力的平方级计算成本限制了其在大规模视图集合上的可扩展性。我们提出了Zebra3D，这是一种逐层的稀疏-线性混合注意力框架，能够在无需从头训练的情况下提升预训练3D基础模型的效率。在VGGT、HunyuanWorld-Mirror和π³上，我们观察到一种普遍一致的深度模式：早期阶段对线性替换相对容忍，而敏感的中后期阶段往往表现出更集中的注意力。Zebra3D利用这一结构，为容忍阶段分配以线性注意力为中心的注意力，为敏感阶段分配瓦片稀疏注意力，并使用地标摘要来保留未被选择的上下文。闭式仿射拟合实现了无需训练的转换，而可选的转换后训练则将任务监督与输出及关系特征蒸馏相结合。

    arXiv:2610.12727v1 Announce Type: cross  Abstract: Feed-forward 3D foundation models reconstruct scenes from unposed images, but the quadratic cost of global attention limits their scalability to large view sets. We present Zebra3D, a layerwise sparse-linear hybrid attention framework that improves the efficiency of pretrained 3D foundation models without training from scratch. Across VGGT, HunyuanWorld-Mirror, and $\pi^3$, we observe a broadly consistent depth-wise pattern: early stages are relatively tolerant to linear replacement, while sensitive middle-to-late stages tend to exhibit more concentrated attention. Zebra3D exploits this structure by assigning linear-centered attention to tolerant stages and tile-sparse attention to sensitive stages, using landmark summaries to preserve unselected context. Closed-form affine fitting enables training-free conversion, while optional post-conversion training combines task supervision with output and relational feature distillation. Across 
    
[^213]: 空间上下文还是记忆？确定部分观测场神经代理模型的输入规模

    Spatial Context or Memory? Sizing the Inputs of Neural Surrogates for Partially Observed Fields

    [https://arxiv.org/abs/2610.12723](https://arxiv.org/abs/2610.12723)

    本文以贝叶斯最优误差为基准，首次精确量化了部分观测场神经代理模型中空间传感器窗口与历史快照长度之间的权衡，证明对于具有局部动力学的稀疏含噪观测场，扩大空间窗口比增加记忆更有效，且仅读取空间窗口的模型即可接近最优预测性能。

    

    从稀疏、含噪传感器预测物理场的神经代理模型会读取一个邻近传感器窗口和一段过去快照的历史。实践者通常通过相互比较拟合好的模型来选择这两者，但这种比较无法将模型自身的不足与信息的缺失区分开来。我们以相同读数下的贝叶斯最优误差为基准来衡量窗口与历史之间的权衡。精确的高斯分析表明，对于具有局部动力学的场，含噪或稀疏的观测使得最优预测只能通过空间相关性来读取远处的传感器。在一个具有已知最优解的受迫线性场上，仅读取窗口的训练模型即可达到与最优解相差1%以内的水平。随后，我们在粘性Burgers流和混沌的Kuramoto-Sivashinsky（KS）方程上训练此类模型，且所有决定性比较均在任何测试评估之前予以固定。在除KS之外的所有场中，八张快照所消除的单快照误差比例随窗口长度增加而下降（摘要原文在此处截断）。

    arXiv:2610.12723v1 Announce Type: new  Abstract: Neural surrogates that forecast a physical field from sparse, noisy sensors read a window of nearby sensors and a history of past snapshots. Practitioners choose both by comparing fitted models with each other. Such comparisons cannot tell a model's shortfall from missing information. We measure the trade-off between window and history against the Bayes-optimal error of the same readings. An exact Gaussian analysis shows that noisy or sparse observation of a field with local dynamics makes the optimal forecast read distant sensors only through spatial correlation. On a forced linear field with a known optimum, trained models that read only the window come within 1% of it. We then train such models on viscous Burgers flow and the chaotic Kuramoto-Sivashinsky (KS) equation, with every deciding comparison fixed before any test evaluation. In every field except KS, eight snapshots remove a smaller fraction of the one-snapshot error as the le
    
[^214]: 大语言模型行为刻画与风险管理的分布框架

    A Distributional Framework for Behavioral Characterization and Risk Management of Large Language Models

    [https://arxiv.org/abs/2610.12718](https://arxiv.org/abs/2610.12718)

    该论文提出一个基于蒙特卡洛模拟构建经验响应分布的框架，通过语义干预量化分布变化，将大语言模型的随机行为刻画与模型风险管理、治理以及黑盒模型监控联系起来。

    

    大语言模型本质上是随机的，在相同实验条件下重复执行可能产生不同的响应。我们提出了一个可应用于模型全生命周期的框架，从任务设计到残差行为分析和持续监控。该方法通过蒙特卡洛模拟估计来构建经验响应分布，并引入语义干预来研究这些分布如何变化。我们用两个不同的任务，借款人分类和音乐专辑推荐，来演示该框架，并量化由此产生的分布变化。该框架通过测量鲁棒性和控制有效性、刻画残差行为，与模型风险管理和治理相联系。由于它仅依赖于观测到的响应，因此也可应用于通过商业API访问的黑盒模型。

    arXiv:2610.12718v1 Announce Type: cross  Abstract: Large language models are stochastic by nature. Repeated executions under the same experimental conditions may produce different responses. We propose a framework that can be applied across the model lifecycle, from task design to residual behavior and continuous monitoring. The method consists of building an empirical response distribution estimated through Monte Carlo simulation. Semantic interventions are introduced to study how these distributions change. We illustrate the framework with two different tasks, borrower classification and music album recommendation, and quantify the resulting distributional changes. The framework is linked to model risk management and governance by measuring robustness and control effectiveness, and characterizing residual behavior. Since it relies only on observed responses, it can also be applied to black-box models accessed through commercial APIs.
    
[^215]: 基于超图消息传递的拜占庭容错联邦大语言模型

    Byzantine-Resilient Federated Large Language Model via Hypergraph Message Passing

    [https://arxiv.org/abs/2610.12711](https://arxiv.org/abs/2610.12711)

    本文提出HMP-GAE框架，将低秩模型更新表示为超图节点，通过迭代消息传递利用多智能体间的高阶语义关系来识别并抑制恶意更新，从而在不访问私有数据的前提下防御拜占庭攻击、保障联邦大语言模型聚合的可信性。

    

    联邦大语言模型有望提供保护隐私的协作智能。然而，去中心化的训练过程容易受到拜占庭攻击，恶意模型更新会破坏语义一致性与事实可靠性。本文提出了一种新的弹性防御策略，防止语义被破坏的知识在联邦聚合过程中传播。论文将防御问题形式化为同时最大化全局准确性并抑制被操纵模型更新的优化问题，从而在不访问私有本地数据的情况下实现可信的知识聚合。为了增强联邦大语言模型的抗攻击能力，我们开发了一种新颖的超图消息传递图自编码器框架，其中低秩模型更新被表示为超图中的节点，并通过迭代消息传递挖掘多个智能体之间的高阶语义关系。通过利用历史检查点和模型更新的演化信息（原文摘要至此截断）。

    arXiv:2610.12711v1 Announce Type: new  Abstract: Federated Large Language Models (FedLLMs) are promising to deliver privacy-preserving, collaborative intelligence. The decentralized training process is vulnerable to Byzantine attacks that corrupt semantic consistency and factual resilience through malicious model updates. This paper presents a new resilience strategy that prevents semantically corrupted knowledge from propagating during federated aggregation. A defense problem is formulated to jointly maximize global accuracy and suppress manipulated model updates, enabling trustworthy knowledge aggregation without accessing private local data. To strengthen the resilience of FedLLMs, we develop a novel Hypergraph Message Passing Graph Autoencoder (HMP-GAE) framework, where low-rank model updates are represented as nodes in a hypergraph and high-order semantic relationships among multiple agents are exploited through iterative message passing. By leveraging historical checkpoints and c
    
[^216]: PVBRF-ID：面向油田企业网络的隐私保护、可验证且抗拜占庭攻击的联邦入侵检测

    PVBRF-ID: Private, Verifiable and Byzantine-Robust Federated Intrusion Detection for Oilfield Enterprise Networks

    [https://arxiv.org/abs/2610.12704](https://arxiv.org/abs/2610.12704)

    PVBRF-ID提出了一种双服务器联邦学习协议，通过站点对更新进行秘密共享、服务器联合检验范数与余弦相似度并结合承诺、Merkle根和同态审计，在不暴露更新的前提下同时实现隐私保护、公开可验证性与抗拜占庭鲁棒性，在油田入侵检测任务中达到与FedAvg相当的精度并能抵御投毒攻击。

    

    联邦学习使分散的油田站点无需汇集网络流量即可共同训练共享的入侵检测器，但其聚合步骤迫使我们只能在隐私与鲁棒性之间二选一。抗拜占庭攻击的规则必须以明文读取每个更新，而安全聚合则将更新隐藏起来，使其无法接受检测投毒的检查。我们提出了PVBRF-ID，这是一种双服务器协议：各站点对其更新进行秘密共享，两台服务器联合检验每个更新的范数及其与在一个小的干净根集上计算所得参考值之间的余弦相似度，且每个站点仅泄露一个接受比特。通过承诺机制、Merkle根和同态审计，被接受更新的纳入过程具备公开可验证性。在一个包含15个类别的Edge-IIoTset任务上，十个站点持有不同分布的数据并使用十个随机种子进行实验，PVBRF-ID在干净数据上与FedAvg表现相当（macro-F1为0.708对0.706），并能抵御符号翻转攻击（0.696对0.091）。面对控制……的自适应攻击者……

    arXiv:2610.12704v1 Announce Type: cross  Abstract: Federated learning allows dispersed oilfield sites to train a shared intrusion detector without pooling network traffic, yet its aggregation step forces a choice between privacy and robustness. Byzantine-robust rules must read each update in the clear, whereas secure aggregation hides updates from the checks that detect poisoning. We present PVBRF-ID, a two-server protocol in which sites secret-share their updates and the servers jointly test each update's norm and its cosine similarity to a reference computed on a small clean root set, revealing one accept bit per site. Commitments, Merkle roots and a homomorphic audit make the inclusion of accepted updates publicly verifiable. On a 15-class Edge-IIoTset task with ten sites holding differently distributed data and ten random seeds, PVBRF-ID matched FedAvg on clean data (macro-F1 0.708 vs. 0.706) and withstood sign-flip attacks (0.696 vs. 0.091). Against an adaptive attacker controllin
    
[^217]: 桥接人类与模型注意力：基于眼动注视验证的CNN、Mamba与ViT架构可解释性分析

    Bridging Human and Model Attention: Explainability Analysis of CNN, Mamba, and ViT Architectures with Gaze-Based Validation

    [https://arxiv.org/abs/2610.12703](https://arxiv.org/abs/2610.12703)

    该论文的核心贡献是首次以眼动追踪数据为真值，通过标准化归一化流程和三种互补的相似度指标，系统量化了CNN、ViT和Vision Mamba三种架构的模型注意力与人类视觉注意力在多个视觉任务上的一致性。

    

    深度学习在高风险应用场景中需要对模型决策过程进行精确刻画。我们对卷积神经网络（CNN）、视觉Transformer（ViT）以及视觉Mamba（ViM）进行了系统性的可解释性评估。与以往以定性分析为主的研究不同，我们利用眼动追踪数据作为真值基准，量化了模型注意力与人类视觉感知之间的对齐程度。我们通过一个标准化的归一化流程将这一比较形式化，该流程将事后归因图和注视热图映射到空间概率分布上。我们采用针对各架构定制的XAI方法，在细粒度分类、医学影像和情绪识别任务上评估对齐程度。我们使用Jensen-Shannon相似度、卡方相似度和Pearson相关系数这三种互补的估计量来衡量保真度，它们分别捕捉支持域重叠、质量差异和结构共变关系。我们的结果揭示了各架构独特的注意力动态特征（摘要原文在此处截断）。

    arXiv:2610.12703v1 Announce Type: cross  Abstract: High-risk applications of deep learning require precise characterization of model decision-making. We conduct a systematic explainability evaluation across Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Vision Mamba (ViM). Unlike predominantly qualitative studies, we quantify the alignment between model attention and human visual perception using eye-tracking ground truth. We formalize this comparison via a standardized normalization pipeline mapping post-hoc attribution maps and gaze heatmaps onto spatial probability distributions. Alignment is evaluated across fine-grained classification, medical imaging, and emotion recognition using architecture-specific XAI methods. We measure fidelity using Jensen-Shannon similarity, Chi-square similarity, and Pearson correlation, three complementary estimators capturing support overlap, mass discrepancy, and structural co-variation. Our results reveal distinct attention dy
    
[^218]: 乘积蒙特卡罗：密度乘积下的期望计算

    Product Monte Carlo: Expectation Under a Product of Densities

    [https://arxiv.org/abs/2610.12702](https://arxiv.org/abs/2610.12702)

    该论文提出乘积蒙特卡罗（PMC）算法，通过将难以处理的密度乘积期望转化为全贝叶斯回归问题，仅利用各分量密度的蒙特卡罗样本即可计算任意期望，并具备一致性保证与不确定性量化能力。

    

    聚合独立计算的子样本推断是大规模贝叶斯计算、联邦学习和意见汇聚领域公认的难题。其核心问题是近似一个密度正比于各个体密度乘积的分布。我们提出了乘积蒙特卡罗（Product Monte Carlo, PMC），这是一种仅利用来自各个体密度的蒙特卡罗样本、即可计算密度乘积下任意期望的算法。我们通过一族可计算的条件期望，将难以直接处理的乘积期望转化为一个回归问题。除了点估计之外，我们的全贝叶斯回归框架还提供了连贯的不确定性量化以及有原则的诊断方法。我们为PMC估计量建立了一致性方面的理论保证，并刻画了其收敛速率。在改变子样本数量、维度和异质性的模拟与真实数据实验中，该方法均表现出良好效果。

    arXiv:2610.12702v1 Announce Type: cross  Abstract: Aggregating independently computed sub-sample inferences is a well-known challenge in large-scale Bayesian computation, federated learning, and opinion pooling. The core problem is to approximate a distribution whose density is proportional to a product of individual densities. We propose Product Monte Carlo (PMC), an algorithm for computing arbitrary expectations under a product of densities using only Monte Carlo draws from individual densities. We turn the intractable product expectation into a regression problem by expressing it through a family of computable conditional expectations. Besides the point estimate, our fully Bayesian regression formulation provides coherent uncertainty quantification and principled diagnostics. We establish theoretical guarantees on consistency and characterize convergence rates for the PMC estimator. Across simulated and real-data experiments varying the number, dimensionality, and heterogeneity of t
    
[^219]: 面向逐样本约束生成的原始-对偶流匹配

    Primal-Dual Flow Matching for Sample-Wise Constrained Generation

    [https://arxiv.org/abs/2610.12699](https://arxiv.org/abs/2610.12699)

    提出原始-对偶流匹配（PDFM），在流匹配框架中引入原始-对偶优化，仅依靠二值约束满足反馈即可直接控制逐样本约束的满足概率，无需可微距离、投影或凸性等约束集结构假设。

    

    我们考虑在生成数据的同时满足预先指定的逐样本约束的问题，并要求约束满足概率达到用户指定的水平。从总体层面的约束负对数似然最小化出发，采用流匹配参数化，在正则性假设以及速度场函数族具有充分表达能力的条件下，我们为相应的原始-对偶优化算法推导了约束满足概率的理论保证。我们给出了该方法的一个实用实现，称为原始-对偶流匹配，它能够直接控制约束满足率，并且对于约束是否满足仅需二值成员资格谕示反馈，从而无需可微的距离度量、投影操作或约束集的特殊结构（如凸性）。我们证明，PDFM 相比朴素 FM 以及……（摘要在此处被截断）能够提升约束满足率。

    arXiv:2610.12699v1 Announce Type: new  Abstract: We consider the problem of learning to generate data while satisfying a prescribed \emph{sample-wise} constraint with a user-specified probability of constraint satisfaction. Starting with a population-level constrained negative log-likelihood minimization and using a flow-matching (FM) parameterization, under regularity assumptions and sufficient expressivity of the velocity field family, we derive guarantees on the probability of constraint satisfaction for the corresponding primal-dual optimization algorithm. We provide a practical implementation of this approach, called Primal-Dual Flow Matching (PDFM), which has direct control over constraint satisfaction rate and requires only binary membership oracle feedback for constraint satisfaction, avoiding the need for differentiable distances, projections, or special structure of the constraint set, such as convexity. We show that PDFM improves constraint satisfaction over vanilla FM and a
    
[^220]: 基于密度依赖骨架构建与强化学习补全的代数连通度最大化研究

    Toward Maximizing Algebraic Connectivity via Density-Dependent Backbone Construction and Reinforcement Learning Completion

    [https://arxiv.org/abs/2610.12697](https://arxiv.org/abs/2610.12697)

    提出一种密度依赖的两阶段方法——先依据密度选择完全多部图或凯莱图作为骨架，再用强化学习补全剩余边——在固定顶点和边数条件下高效构建代数连通度最大的图拓扑，性能优于现有方法且运行时间可扩展性良好。

    

    本文研究了一个开放的网络设计问题，即在顶点数和边数固定的条件下，构建具有最大代数连通度的图拓扑结构。文章提出了一种基于密度的方法，将设计任务分解为两个阶段：高质量、低成本的骨架结构构建阶段和强化学习补全阶段。该方法根据图的密度选择骨架，在稠密情形下使用完全多部图，在低到中等密度情形下使用凯莱图，然后通过基于学习的过程来完善剩余的边。数值结果表明，所提出的方法能够实现较高的代数连通度，且运行时间具有良好的可扩展性，优于现有方法。这些结果支持了一种面向可扩展网络设计的“结构族优先”策略。

    arXiv:2610.12697v1 Announce Type: cross  Abstract: This paper studies an open network design problem of constructing graph topologies with maximum algebraic connectivity under fixed numbers of vertices and edges. A density-dependent approach is proposed that decomposes the design task into two stages: a high-quality, low-cost structural backbone construction and a reinforcement learning completion stage. The method selects the backbone based on density, using complete multipartite graphs in dense regimes and Cayley graphs in low-to-intermediate regimes, and then refines the remaining edges through a learning-based procedure. Numerical results show that the proposed method achieves high algebraic connectivity with favorable runtime scalability, outperforming existing methods. These results support a structural-family-first strategy for scalable network design.
    
[^221]: 临界质量：结构不相交的数据划分随其构建所依赖的分子库规模而漂移

    Critical Mass: Structure-Disjoint Splits Drift With the Libraries They Are Built On

    [https://arxiv.org/abs/2610.12694](https://arxiv.org/abs/2610.12694)

    论文指出分子机器学习中用于分布外评估的结构不相交划分标准（基于MCES距离阈值10的单链接聚类）是建立在29,000分子规模上的一次性惯例，随分子库增长到22万余分子后已无法计算验证，为此提出结合可采纳界与并查集跳过的优化算法，将计算量降低2,259倍，使该评估标准首次可在现代公共分子库规模上被检验。

    

    分子机器学习通过结构不相交划分来评估分布外性能，而这一构建规则在分子库恰好具备的规模上被一次性固定下来：先对成对距离设定阈值，再将整个连通分量分配到各折中。在从质谱预测分子结构的任务中，该领域已收敛于在短视的最大公共边子图（MCES）距离下以阈值10进行单链接聚类。这一惯例是在29,000个分子上确立的，但公共分子库现已包含227,690个分子。由于MCES对每一对分子的计算都是NP难的，且分子对数量随规模呈二次增长，因此该标准从未在这一规模上被评估过。我们提出了该标准的一个优化版本，使其能够在公共分子库上以计算可行的方式应用：通过可采纳的界和并查集跳过机制，将2.6×10¹⁰个分子对减少到仅1.15×10⁷个整数规划问题，节省了2,259倍的计算量，同时可证明……（原文摘要在此处截断）

    arXiv:2610.12694v1 Announce Type: new  Abstract: Molecular machine learning evaluates out-of-distribution performance with structure disjoint splits, a construction rule fixed once at the scale its library happened to have: threshold a pairwise distance, then assign whole connected components to folds. For predicting structures from mass spectra, the field has converged on single-linkage clustering under the myopic Maximum Common Edge Subgraph (MCES) distance at a threshold of 10. This convention was set on 29,000 molecules but public libraries now hold 227,690 molecules. MCES is NP-hard per pair over a quadratically growing set, and so the criterion has never been evaluated at that scale. We propose an optimized version of this criterion, allowing for computationally feasible application on the public libraries: Admissible bounds and a union-find skip reduce $2.6 \times 10^{10}$ molecular pairs to only $1.15 \times 10^{7}$ integer programs, a $2{,}259\times$ saving, while provably pro
    
[^222]: 脑电图低电压间期的专家级检测：一种新生儿脑病的稳健生物标志物

    Expert-level detection of low-voltage intervals in EEG as a robust biomarker of neonatal encephalopathy

    [https://arxiv.org/abs/2610.12680](https://arxiv.org/abs/2610.12680)

    该论文提出仅由振幅和持续时间定义的低电压间期（LVIs）作为新生儿脑电图不连续性的全新指标，并开发了基于ConvNeXt的AI检测模型，在异常分级、HIE分级、癫痫预测等多项任务中验证了LVIs作为新生儿脑病稳健生物标志物的专家级检测能力。

    

    背景：新生儿脑电图的不连续性是脑功能障碍的关键标志，但难以通过视觉观察进行量化。爆发间期（IBIs）是衡量不连续性的一个关键特征，然而其定义不易直接转化为AI检测任务。研究目标：本文引入低电压间期（LVIs）这一概念，其仅由振幅和持续时间定义，作为IBI的独立于上下文的替代方案；开发一种针对LVIs的AI检测方法，并评估其作为脑损伤标志物的临床实用性。方法：我们开发了一个ConvNeXt模型来检测LVIs，并将其性能与现有方法进行比较。随后，我们在四项应用中检验了所检测LVIs的临床相关性：1）在一个公开基准数据集上对脑电图背景异常进行分级；2）对临床缺氧缺血性脑病（HIE）分级并与正常脑电图数据集进行对比；3）癫痫发作预测；4）与癫痫发作负荷的关联分析。结果：（原文摘要在此处被截断）

    arXiv:2610.12680v1 Announce Type: new  Abstract: Background: Discontinuity in neonatal EEG is a critical marker of brain dysfunction but can be difficult to quantify visually. Inter-burst intervals (IBIs) are a key characteristic of discontinuity, yet their definition does not easily translate to AI detection. Objectives: To introduce low-voltage intervals (LVIs), defined exclusively by amplitude and duration, as a context-independent alternative to IBIs, develop an AI detection method for LVIs, and assess their clinical utility as a marker of brain injury. Methods: We developed a ConvNeXt model to detect LVIs and compared its performance with existing methods. We then tested the clinical relevance of detected LVIs across four applications: 1) grading background EEG abnormalities on an open benchmark dataset, 2) classifying clinical hypoxic-ischaemic encephalopathy (HIE) grades and comparing with a normal EEG dataset, 3) seizure prediction, and 4) association with seizure burden. Resul
    
[^223]: 通过单步隐式采样实现摊销式推断

    Amortised Inference through One-Step Implicit Sampling

    [https://arxiv.org/abs/2610.12679](https://arxiv.org/abs/2610.12679)

    提出单步隐式采样器（OSIS），利用短时MCMC动力学定义局部目标使建模分布成为MCMC核的不动点，从而实现高维贝叶斯后验分布的单步摊销采样。

    

    尽管在理论和算法方面取得了进展，从高维贝叶斯后验分布中进行高效采样仍然是一个难题。通常需要多个步骤才能抽取无偏样本，例如MCMC算法的迭代，或基于扩散的摊销采样器中的积分步骤。然而，可扩展的单步摊销推断仍然是一个具有挑战性的开放问题。在本工作中，我们建立在数据驱动生成建模的最新进展之上，为单步摊销采样提供了一种可能的解决方案。我们的单步隐式采样器（OSIS）利用短时MCMC动力学为隐式建模分布的演化定义局部目标。在渐近极限下，所建模的分布成为MCMC核的不动点，因此其样本即来自目标分布。我们分析了OSIS的收敛性质，并展示了它与其他生成建模和采样算法之间的联系。我们研究了各种……

    arXiv:2610.12679v1 Announce Type: new  Abstract: Efficient sampling from Bayesian posterior distributions in high dimensions remains a hard problem, despite advances in theory and algorithms. Multiple steps are usually required to draw unbiased samples, for example, iterations of an MCMC algorithm or integration steps in a diffusion-based amortised sampler. However, scalable one-step amortised inference remains a challenging open problem. In this work, we build upon recent progress in data-driven generative modelling to offer a possible solution for one-step amortised sampling. Our one-step implicit sampler (OSIS) uses short-time MCMC dynamics to define a local target for the evolution of an implicitly modelled distribution. In the asymptotic limit, the modelled distribution is a fixed point of the MCMC kernel and thus samples from the target. We analyse the convergence properties of OSIS and show connections with other generative modelling and sampling algorithms. We study various des
    
[^224]: 面向智能体持续学习的循环式执行框架

    Recurrent Harness for Agentic Continual Learning

    [https://arxiv.org/abs/2610.12668](https://arxiv.org/abs/2610.12668)

    提出 ReHarness，一种循环式智能体持续学习框架，在严格在线流式协议下，将每个实例处理后的轨迹作为反馈向后传播，修订该实例对应的反思状态，实现前向读写与后向修订相结合的在线适应。

    

    语言模型智能体日益需要在长程交互流上运行，这要求通过持续学习对分布偏移进行在线适应。然而，现有的智能体框架通常要么仅通过前向信息流来演化其状态，要么依赖额外的回放或验证数据来进行框架优化。为解决这一问题，我们提出 ReHarness，一个面向智能体持续学习的循环式框架，它在严格的在线流式协议下，在前向读写之外补充了后向修订机制。我们提出这样一个问题：在框架状态中，哪些信息本可以以不同的方式被捕获，以便更好地服务于流中的当前实例。具体而言，在每个实例处理完成后，我们利用所产生的轨迹作为反馈，对为该实例检索到的反思进行修订，从而将反馈通过文本化的框架状态向后传播。该框架在三个持续学习基准和三个语言模型上进行了验证。

    arXiv:2610.12668v1 Announce Type: cross  Abstract: Language model agents increasingly operate over long interaction streams, requiring online adaptation to distribution shifts through continual learning. However, existing agent harnesses typically either evolve their states only through forward information flow or rely on additional replay or validation data for harness optimization. To address this, we propose ReHarness, a recurrent harness framework for agentic continual learning that complements forward read and write with backward revision under a strictly online streaming protocol. We ask what information could have been captured differently in the harness state to better inform the current instance in the stream. Specifically, after each instance, we leverage the resulting trajectory as feedback to revise the reflections retrieved for that instance, thereby propagating feedback backward through the textual harness state. Across three continual learning benchmarks and three langua
    
[^225]: MagicBeanStalk：通过递归数据合成摆脱RLVR零优势陷阱

    MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis

    [https://arxiv.org/abs/2610.12659](https://arxiv.org/abs/2610.12659)

    该论文发现主流RLVR数学语料库中约88%的问题因过于简单或过难而无法为目标策略提供有效学习信号（即零优势陷阱），并提出MagicBeanStalk方法，通过递归数据合成生成既蕴含难题所需推理操作、又难度适中的有效训练问题。

    

    以数据为中心的RLVR扩展一直聚焦于收集或生成可验证的训练问题。然而，语料库的规模本身并不能决定该数据集能否为目标策略提供有用的学习信号。为了指导有效的RLVR数据合成，我们通过实验将数据集的有效性刻画为其问题、目标策略和训练rollout预算的函数。在这一刻画下，我们展示了一个汇集自广泛使用的RLVR数据集的大型开源数学语料库几乎完全陷入了零优势陷阱：在固定的目标策略和rollout预算下，大约88%的问题要么在每次rollout中都被解决，要么在每次rollout中都失败。过滤可以浓缩剩余部分，但它只能进行选择，从而丢弃了难题。因此，有效的合成问题应当需要解决那些难题所必需的推理操作及其组合，但其难度必须加以控制（原文在此处截断）。

    arXiv:2610.12659v1 Announce Type: new  Abstract: Data-centric scaling of RLVR has focused on collecting or generating verifiable training problems. However, corpus size alone does not determine whether the dataset provides a useful learning signal for a target policy. To guide the synthesis of effective RLVR data, we experimentally characterize dataset effectiveness as a function of its problems, the target policy, and the training rollout budget. Under this characterization, we show that a large open-source math corpus pooled from widely used RLVR datasets falls almost entirely into a zero-advantage trap: for a fixed target policy and rollout budget, roughly 88% of its problems are solved in every rollout or failed in every rollout. Filtering concentrates what remains, but it can only select, discarding the hard problems. Effective synthetic problems should therefore require the reasoning operations and compositions required to solve those hard problems. Yet their difficulty must be c
    
[^226]: 英国气溶胶监测的数据驱动最优传感器布置

    Data-driven Optimal Sensor Placement for Aerosol Monitoring in the UK

    [https://arxiv.org/abs/2610.12658](https://arxiv.org/abs/2610.12658)

    该研究提出一个将卷积高斯神经过程与带惰性评估和缓存的遗传优化相结合的数据驱动传感器布置框架，使英国气溶胶监测的计算时间减少60%、联合负对数似然提升约22%，为环境观测系统的数据驱动扩展提供了计算高效的方案。

    

    我们开发了一个数据驱动的传感器布置框架，并将其应用于英国（UK）的气溶胶监测。训练好的卷积高斯神经过程被嵌入到DeepSensor优化框架中。采用惰性评估和缓存的遗传优化将平均计算时间减少了60%（10±3分钟对比25±7分钟），同时产生的顺序布置性能与贪婪求解器相当。利用CAMS参考场构建回溯性神谕基准，在确定五个气溶胶站点位置后，联合互信息驱动的遗传布置的联合负对数似然（Joint NLL）提升了约22%。季节性和风场条件的分析表明，基于模型的准则与神谕基准之间的一致性随气象输送条件而变化。这些结果展示了一种计算高效的方法，可用于数据驱动的环境观测系统扩展。

    arXiv:2610.12658v1 Announce Type: new  Abstract: We develop a data-driven sensor-placement framework and apply it to aerosol monitoring in the United Kingdom (UK). A trained convolutional Gaussian neural process is embedded within the DeepSensor optimisation framework. Genetic optimisation with lazy evaluation and caching reduces mean computational time by 60\% (10$\pm$3 versus 25$\pm$7 minutes) while producing sequential placement performance comparable to the greedy solver. Using the CAMS reference field to construct a retrospective oracle benchmark, Joint NLL improves by about 22\% for Joint-MI-driven genetic placement after five locations for aerosol stations are identified. Seasonal and wind-regime analyses show that agreement between model-based criteria and oracle benchmarks varies with meteorological transport conditions. These results show a computationally efficient approach for data-informed expansion of environmental observing systems.
    
[^227]: 克罗内克积注意力：基于结构化分解的内存高效KV缓存压缩

    Kronecker Product Attention: Memory-Efficient KV Compression via Structured Factorization

    [https://arxiv.org/abs/2610.12657](https://arxiv.org/abs/2610.12657)

    提出克罗内克积注意力（KPA），将键值激活表示为紧凑的词元相关矩阵因子的克罗内克积之和，严格推广了张量积注意力（TPA），在保持表达能力的同时实现了内存高效的KV缓存压缩。

    

    高效的自回归生成日益受到键值（KV）缓存内存占用的限制，该缓存随序列长度、批次大小和模型深度线性增长。现有方法如多查询注意力（MQA）、分组查询注意力（GQA）、多头潜在注意力（MLA）和张量积注意力（TPA）通过不同的权衡来降低这一成本，但往往牺牲了表达能力、需要额外的位置编码机制，或强加了限制性的低秩结构。我们提出克罗内克积注意力（KPA），它将键和值激活表示为紧凑的、依赖于词元的矩阵因子的克罗内克积之和。KPA严格推广了TPA：秩一向量外积分解作为一种退化的克罗内克分裂出现，而非退化的矩阵因子能够表示更丰富的块结构模式。我们进一步证明，与之匹配的结构化旋转位置编码算子……

    arXiv:2610.12657v1 Announce Type: cross  Abstract: Efficient autoregressive generation is increasingly limited by the memory footprint of the key-value (KV) cache, which grows linearly with sequence length, batch size, and model depth. Existing approaches such as Multi-Query Attention (MQA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), and Tensor Product Attention (TPA) reduce this cost through different trade-offs, but often sacrifice expressivity, require additional positional-encoding machinery, or impose restrictive low-rank structure. We propose Kronecker Product Attention (KPA), which represents key and value activations as sums of Kronecker products of compact, token-dependent matrix factors. KPA strictly generalizes TPA: rank-one vector-outer-product factorization arises as a degenerate Kronecker split, while non-degenerate matrix factors can represent richer block-structured patterns. We further show that a structured rotary embedding operator matched to t
    
[^228]: SynAct：面向大语言模型智能体的协同主动反馈学习

    SynAct: Synergistic Active Feedback Learning for LLM Agents

    [https://arxiv.org/abs/2610.12651](https://arxiv.org/abs/2610.12651)

    提出SynAct主动学习框架，将专家注意力视为稀缺约束资源，利用动作与最终结果之间的互信息对决策排序，在最能改变结果的关键决策点精准请求专家反馈并提炼为可迁移规则，从而弥补LLM智能体的知识缺口。

    

    arXiv:2610.12651v1 公告类型：新论文 摘要：执行多步骤工作流的大语言模型（LLM）智能体以两种根本不同的方式失败：可恢复的失误（进一步的尝试可以修复）和知识缺口（重试无法解决）。专家反馈可以解决后者，但专家的注意力是稀缺资源。因此，关键挑战不在于是否请求反馈，而在于在何处请求反馈才能改变结果。现有方法要么完全避免使用专家反馈，导致智能体不断强化自身的错误；要么在任务层面分配反馈，而未能识别出关键的决策点。我们提出了SynAct，这是一个主动学习框架，它将专家注意力视为约束性瓶颈资源，并在单个决策的层面分配反馈。SynAct通过在多次重复rollout中计算所选动作与最终结果之间的互信息来对决策进行排序，在价值最高的决策处向专家查询，并将所得的修正提炼为可迁移的规则。

    arXiv:2610.12651v1 Announce Type: new  Abstract: LLM agents executing multi-step workflows fail in two fundamentally different ways: recoverable slips, which further attempts can repair, and knowledge gaps, which retrying cannot resolve. Expert feedback can address the latter, but expert attention is scarce. The key challenge is therefore not whether to ask for feedback, but where feedback can change the outcome. Existing approaches either avoid expert feedback, allowing agents to reinforce their own errors, or allocate feedback at the task level without identifying consequential decisions. We present SynAct, an active learning framework that treats expert attention as the binding constraint and allocates feedback at the level of individual decisions. SynAct ranks decisions by the mutual information between the selected action and the terminal outcome across repeated rollouts, queries an expert at the highest-value decisions, and distills the resulting corrections into transferable rul
    
[^229]: 面向大学建筑日前电力预测的机器学习与时间序列基础模型

    Machine Learning and Time-Series Foundation Models for Day-Ahead Electricity Forecasting in a University Building

    [https://arxiv.org/abs/2610.12638](https://arxiv.org/abs/2610.12638)

    该研究在一栋大学建筑上对统计、机器学习、循环神经网络和预训练TimesFM等日前电力预测模型进行了统一基准测试，发现带协变量的TimesFM 3.0总体误差最低（平均绝对误差较每周季节性朴素预测降低约43%），而XGBoost仍具竞争力。

    

    日前电力预测可以帮助建筑运营者根据预期需求来安排供暖、通风及其他服务。选择预测模型还需要在预测精度与其预测时所需输入、本地训练和计算需求之间进行权衡。我们针对一栋大学建筑，对统计模型、机器学习模型、循环神经网络模型以及预训练的TimesFM模型进行了基准测试。我们采用按时间顺序划分的验证方式，并在相同的57个测试日上评估所有配置，比较各模型在工作日、周末和节假日的精度，同时比较本地拟合与预测所需时间。天气协变量采用回溯观测数据。带协变量的TimesFM 3.0取得了最低的总体误差，与每周季节性朴素预测相比，平均绝对误差降低了约43%。XGBoost仍具有竞争力，尽管逐日配对比较并未证明TimesFM在平均绝对误差上具有明确优势。

    arXiv:2610.12638v1 Announce Type: cross  Abstract: Day-ahead electricity forecasts can help building operators schedule heating, ventilation, and other services around anticipated demand. Choosing a forecasting model also requires balancing accuracy with its forecast-time inputs, local training, and computational requirements.   We benchmark statistical, machine-learning, recurrent neural-network, and pretrained TimesFM models for a university building. We use chronological validation and evaluate all configurations on the same 57 test days, comparing accuracy across workdays, weekends, and holidays alongside local fitting and forecasting time. Weather covariates use retrospective observations.   TimesFM 3.0 with covariates achieves the lowest overall errors, reducing mean absolute error by approximately 43\% relative to weekly seasonal-naive forecasting. XGBoost remains competitive, although paired daily comparisons do not establish a clear mean-absolute-error advantage for TimesFM. T
    
[^230]: STAPLE：通过嵌入模块实现预训练检查点的静态稀疏升级

    STAPLE: Static Sparse Upcycling of Pretrained Checkpoints via Embedding Modules

    [https://arxiv.org/abs/2610.12599](https://arxiv.org/abs/2610.12599)

    STAPLE 通过将基于 token 索引的嵌入作为残差路径加入预训练模型，在保留原有 FFN 计算的同时以可扩展的方式学习互补的 token 特定信息，从而实现高效的静态稀疏升级。

    

    稀疏计算可以在不按比例增加活跃计算量的情况下扩展大语言模型的参数容量，但预训练稀疏架构仍然成本高昂且需要大量数据。现有的稀疏升级方法主要针对混合专家模型，其中专家同质性、样本效率低和系统开销等问题可能限制其实际可扩展性。静态稀疏内存提供了可预测的访问方式和针对特定 token 的更新，有助于应对这些挑战，但将其简单地集成到预训练计算中收益甚微。在这项工作中，我们提出了 STAPLE，一种建立在 STEM 基础之上、用于将静态稀疏内存集成到预训练模型中的方法。STEM 用基于 token 索引的嵌入替换 FFN 上投影，而 STAPLE 则保留原有的 FFN 上投影，并将这些嵌入作为残差路径加入，在以可扩展的方式学习互补的 token 特定信息的同时，保留了预训练的计算能力。在 OLMo-2 1B 模型上，STAPLE 提升了……（摘要内容在此处截断）

    arXiv:2610.12599v1 Announce Type: new  Abstract: Sparse computation can scale the parametric capacity of LLMs without proportionally increasing active computation, but pretraining sparse architectures remains expensive and data-hungry. Existing sparse upcycling methods mainly target Mixture-of-Experts models, where expert homogeneity, sample inefficiency, and system overheads can limit practical scalability. Static sparse memory offers predictable access and token-specific updates that can help address these challenges, but naively integrating it into pretrained computation yields poor returns. In this work, we introduce STAPLE, a recipe built upon STEM for integrating static sparse memory into pretrained models. Whereas STEM replaces FFN up-projections with token-indexed embeddings, STAPLE retains them and adds the embeddings as a residual pathway, preserving pretrained computation while learning complementary token-specific information in a scalable manner. On OLMo-2 1B, STAPLE impro
    
[^231]: SpireBench：利用《杀戮尖塔》评估智能体模型的长期战略博弈能力

    SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire

    [https://arxiv.org/abs/2610.12596](https://arxiv.org/abs/2610.12596)

    SpireBench让20个大语言模型完整游玩《杀戮尖塔》以评估长时程战略推理能力，发现前沿模型虽达到普通人类水平但远逊于专家，且随着游戏进程延长差距不断扩大，所有模型都误判长线风险，倾向保守保血而非牺牲血量换取成长。

    

    arXiv:2610.12596v1 公告类型：新论文 摘要：智能体大语言模型如今已能处理复杂任务，但针对长时程的战略推理研究仍然不足：大多数智能体基准测试短小且情节化，而且模型在长任务上的表现难以分解为存储知识、推理能力和运气。我们提出SpireBench，让模型完整地游玩商业卡牌构筑游戏《杀戮尖塔》，在该游戏中早期决策决定长期成败。在最难难度下对20个大语言模型进行测试后发现：前沿模型击败了启发式基线，达到普通人类玩家的平均水平，但仍远远落后于专家玩家。这一差距随着单局进程的延长而扩大：最强的模型在第一幕的胜率与专家相当，但随后逐渐落后。区分强弱模型的关键在于它们打好每一场战斗的能力；而所有模型在整个局中都会误判风险，倾向于保留生命值而非花费生命值来换取成长。对游戏中所有术语进行全面且自洽的重命名实验表明……（摘要至此截断）

    arXiv:2610.12596v1 Announce Type: new  Abstract: Agentic large language models now tackle complex tasks, yet strategic reasoning over long horizons is under-studied: most agentic benchmarks are short and episodic, and performance on long tasks is hard to dissect into stored knowledge, reasoning and luck. We introduce SpireBench, in which models play the complete commercial deck-builder Slay the Spire (StS), where early decisions decide long-term success. Across 20 large language models (LLMs) at the game's highest difficulty, frontier models beat heuristic baselines and match the average human player but remain far behind experts. This gap widens as the run lengthens: the strongest models win the first act as often as experts, then fall behind. What separates strong models from weak is how well they play each fight; all of them misjudge risk over a run, preserving health rather than spending it to grow stronger. A comprehensive, self-consistent renaming of every game term shows that pa
    
[^232]: 真实性 VS 准确性：从生成建模视角看事件序列预测

    Realism VS Accuracy: Event Sequence Forecasting from a Generative Modeling Perspective

    [https://arxiv.org/abs/2610.12586](https://arxiv.org/abs/2610.12586)

    该论文提出从条件生成建模视角研究事件序列多步预测，构建了同时评估预测准确性与分布真实性的基准，揭示了两者之间的根本性权衡，并提出Wasserstein BaryBooster算法，将真实性强的扩散模型样本聚合到最先进的预测器中。

    

    事件序列是由随机事件组成的链条，每个事件由异构（数值型和类别型）特征描述。事件序列预测问题出现在金融、医疗保健、业务流程智能等领域。在这项工作中，我们将事件序列的多步预测研究为一个条件生成建模问题。我们不仅从预测准确性方面评估预测结果，还从分布真实性方面进行评估。为此，我们提出了一个基准，包含针对这两方面质量的度量指标、强大的统计基线模型、从相邻领域借鉴的近期模型，以及精选的真实世界数据集。细致的评估表明，没有任何方法能在准确性和真实性两方面都表现出色，这凸显了两者之间根本性的权衡。最后，我们提出了Wasserstein BaryBooster，这是一种将来自真实性强的扩散模型的样本聚合到最先进预测器中的算法，证明了分布（原文在此处被截断）……

    arXiv:2610.12586v1 Announce Type: new  Abstract: Event sequences are chains of random events, each described by heterogeneous (numerical and categorical) features. The problem of event sequence forecasting occurs in finance, healthcare, business process intelligence, and other fields. In this work, we study multi-horizon forecasting of event sequences as a conditional generative modeling problem. We evaluate forecasts not only for prediction accuracy, but also for distributional realism. To this end, we present a benchmark comprising metrics for both quality aspects, strong statistical baselines, recent models adapted from adjacent fields, and curated real-world datasets. Careful evaluation reveals that no method excels at both accuracy and realism, highlighting a fundamental trade-off between the two. Finally, we introduce Wasserstein BaryBooster, an algorithm that aggregates samples from a realism-strong diffusion model into a state-of-the-art predictor, demonstrating that distributi
    
[^233]: DiffusionART：高效、稳定且可扩展的扩散模型强化方法

    DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models

    [https://arxiv.org/abs/2610.12572](https://arxiv.org/abs/2610.12572)

    DiffusionART 通过将每个提示词的奖励聚合为共享的奖励提升方向以在构造上消除前向-反向一致性力，并结合信任域掩码的二次更新实现无需 EMA 调参的漂移自适应恢复，从而实现扩散模型在线强化学习的高效、稳定与可扩展。

    

    在线强化学习（RL）利用黑盒奖励对扩散模型进行微调，但诸如 Flow-GRPO 这类反向过程方法的效率受限于糟糕的信用分配，而诸如 DiffusionNFT 这类前向过程方法虽然高效，但在缺乏精心调整的指数移动平均（EMA）离线策略训练时并不稳定。我们证明了 NFT 风格的目标函数中包含一种隐藏的前向-反向一致性力，联合的正负样本训练只能部分抵消这种力，并且 EMA 是通过恢复机制而非离线策略采样来稳定训练的。我们提出了 DiffusionART（基于聚合优先奖励倾斜的扩散模型强化），该方法将每个提示词的奖励聚合为一个共享的奖励提升方向并作用于反向过程状态，从而在构造上消除一致性力，同时增加一个由信任域掩码控制的第二次更新，实现无需 EMA 调参的漂移自适应恢复。该方法不需要可微分奖励……

    arXiv:2610.12572v1 Announce Type: new  Abstract: Online reinforcement learning (RL) fine-tunes diffusion models with black-box rewards, but reverse-process methods such as Flow-GRPO are limited in efficiency by poor credit assignment, while forward-process methods such as DiffusionNFT are efficient yet unstable without carefully tuned exponential-moving-average (EMA) off-policy training. We show that NFT-style objectives contain a hidden forward--reverse consistency force that joint positive--negative training cancels only partially, and that EMA stabilizes training through restoration rather than off-policy rollouts. We introduce DiffusionART (Diffusion Reinforcement with Aggregation-First Reward Tilting), which aggregates each prompt's rewards into a shared reward-improving direction for reverse-process states, removing the consistency force by construction, and adds a trust-region-masked second update for drift-adaptive restoration without EMA tuning. It requires no differentiable r
    
[^234]: 一种全晶体管物理学习Transformer可以被训练执行上下文学习

    An all-transistor physical-learning transformer can be trained to perform in-context learning

    [https://arxiv.org/abs/2610.12571](https://arxiv.org/abs/2610.12571)

    该论文提出了一种全晶体管物理学习Transformer，通过在双电路中引入额外电路元件恢复双向信息流，并利用亚阈值MOSFET的指数电流特性构建注意力机制，使物理系统首次能够被训练执行上下文学习任务。

    

    大多数通过平衡传播或耦合学习训练的系统都属于与多层感知器相同的类别：虽然它们可能是通用函数逼近器，但它们难以学习那些对Transformer而言很容易的关联任务。平衡传播依赖于信息在物理系统中输入与输出之间的双向流动。而更复杂的非线性系统会破坏这种反向流动，例如执行乘法运算的系统，而乘法正是表达点积注意力所必需的。在这里，我们通过在双电路（dyadic circuit）中引入额外的电路元件，显式地恢复了这种双向流动。随后，我们利用这些想法构建了一种注意力机制，该机制利用了亚阈值MOSFET器件的指数电流定律。我们通过仿真表明，尽管使用了现实的后向传输组件（摘要在此处截断），该电路仍可以被训练来执行一个简单的上下文学习任务。

    arXiv:2610.12571v1 Announce Type: new  Abstract: Most systems trained by equilibrium propagation or coupled learning are in the same class as multi-layer perceptrons: while they may be universal function approximators, they struggle to learn associative tasks which are easy for transformers. Equilibrium propagation relies on a bidirectional flow of information through the physical system between input and output. The reverse flow is broken by more complex nonlinear systems, such as those performing multiplication which is needed to express dot-product attention. Here we explicitly restore this bidirectional flow by introducing extra circuit elements for this purpose into a double or dyadic circuit. We then use these ideas to construct an attention mechanism exploiting the exponential current law of sub-threshold MOSFET devices. We show that in simulations this circuit can be trained to perform a simple in-context learning task, despite using realistic reverse components which transmit 
    
[^235]: 样本自适应零阶后训练

    Sample-Adaptive Zeroth-Order Post-Training

    [https://arxiv.org/abs/2610.12566](https://arxiv.org/abs/2610.12566)

    提出样本自适应零阶优化（SAZO），通过在线估计样本重要性并自适应分配查询预算来降低估计方差、加速大语言模型的零阶微调收敛，且无需额外模型评估、仅增加 O(N) 的标量状态。

    

    零阶（ZO）优化为微调大语言模型提供了一种内存高效的替代方案。然而，现有的零阶方法通常均匀地采样训练样本，忽略了样本不断演变的重要性，留下了可避免的估计方差。为解决这一局限，我们提出了样本自适应零阶优化（SAZO），它在线学习样本重要性并自适应地分配查询以加速收敛。SAZO从现有的方向差分中提取梯度范数信息来估计样本重要性，而无需访问真实梯度。此外，为了从稀疏的小批量反馈中追踪不断演变的重要性，SAZO为每个样本维护指数移动平均，以跨更新聚合观测结果。该方法不需要额外的模型评估，与MeZO相比仅增加 $O(N)$ 的标量状态。我们证明SAZO渐近地匹配领先的（摘要在此处截断）

    arXiv:2610.12566v1 Announce Type: new  Abstract: Zeroth-order (ZO) optimization offers a memory-efficient alternative to backpropagation for fine-tuning large language models. However, existing ZO methods typically sample training examples uniformly, overlooking their evolving importance and leaving avoidable estimation variance. To address this limitation, we propose Sample-Adaptive Zeroth-Order Optimization (SAZO), which learns sample importance online and adaptively allocates queries to accelerate convergence. SAZO extracts gradient-norm information from existing directional differences to estimate sample importance without access to true gradients. Furthermore, to track evolving importance from sparse minibatch feedback, SAZO maintains per-example exponential moving averages that aggregate observations across updates. The method requires no additional model evaluations and only $O(N)$ additional scalar state compared with MeZO. We prove that SAZO asymptotically matches the leading 
    
[^236]: Fly-by-Code（代码飞行）：基于主动视觉与物理反馈的空中操作具身编码智能体

    Fly-by-Code: Embodied Coding Agents for Aerial Manipulation with Active Visual and Physical Feedback

    [https://arxiv.org/abs/2610.12549](https://arxiv.org/abs/2610.12549)

    该论文提出主动反馈框架，通过主动“观察与探测”获取缺失的视觉和物理证据，解决了空中操作中部分可观测性和交互鲁棒性受限导致失败原因无法诊断的问题，从而支持具身编码智能体有效地改进程序策略。

    

    具身编码智能体通过生成可执行程序并根据执行反馈对其进行改进，从而从自然语言指令构建机器人策略。这种改进过程假设执行反馈能够揭示任务是否成功，并且在任务失败后能提供足够的证据来指导程序修订。空中操作在两个方面违背了这一假设：载荷约束限制了机载相机的覆盖范围，使任务关键结果处于机器人视野之外（部分可观测性）；同时强烈的动态耦合和有限的控制裕度使物理交互变得脆弱（交互鲁棒性受限），导致失败原因难以确定。重新检查相同的反馈既无法恢复从未收集到的观测数据，也无法区分现有证据无法区分的失败原因。为了获取这些缺失的证据，我们引入了主动反馈，一个主动的“观察与探测”框架。……

    arXiv:2610.12549v1 Announce Type: cross  Abstract: Embodied coding agents construct robot policies from natural-language instructions by generating executable programs and refining them from execution feedback. This refinement assumes that execution feedback reveals whether the task succeeded and, after a failure, provides sufficient evidence to guide program revision. Aerial manipulation violates this assumption in two ways. Payload constraints restrict onboard camera coverage and leave task-critical outcomes outside the robot's view (partial observability), while strong dynamic coupling and limited control margins make physical interactions fragile (limited interaction robustness) and leave failure causes unresolved. Re-examining the same feedback cannot recover observations that were never collected or separate failure causes that the available evidence does not distinguish. To acquire this missing evidence, we introduce Active Feedback (AF), an active "View and Probe" framework. Wh
    
[^237]: AutoIndexer：通过在编辑链上训练因果模型实现灵活顺序解码

    AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits

    [https://arxiv.org/abs/2610.12545](https://arxiv.org/abs/2610.12545)

    AutoIndexer提出了一种模拟文本编辑器的“编辑链”训练与解码机制，使因果自回归模型能够通过插入、替换和删除操作灵活地事后修改已生成的输出，从而支持非因果顺序的灵活解码。

    

    自回归解码不太适合非因果顺序的数据，例如交叉引用或未来依赖关系，因为其输出无法事后修改，这与迭代式的人类推理不同。扩散式大语言模型部分缓解了这一限制，但仍无法灵活地在预先生成的上下文中插入或删除任意数量的标记。我们将AutoIndexer设计为一个具有新颖训练和解码机制的生成模型，该机制模拟“编辑链”，使模型能够通过标记的插入、替换和删除来灵活地事后修改其先前的输出。该模型学习预测用于开启和关闭编辑的标记token，同时将光标移动到现有token序列中的位置，类似于文本编辑器。训练序列是通过对真实标签施加逆链式编辑操作进行扰动而获得的，模型通过撤销这些逆操作来学习编辑。扰动过程……

    arXiv:2610.12545v1 Announce Type: cross  Abstract: Autoregressive decoding is less suited to non-causally ordered data, e.g., cross-references or future dependencies, since their outputs cannot be revised post hoc, unlike iterative human reasoning. Diffusion LLMs partially mitigate this limitation but still cannot flexibly insert or remove arbitrary numbers of tokens from a pre-generated context. We design AutoIndexer as a generative model with a novel training and decoding mechanism that simulates a Chain of Edits, allowing the model to flexibly modify its prior outputs in hindsight via token insertions, substitutions, and deletions. The model learns to predict marker tokens for opening and closing edits, simultaneously moving a cursor to positions in the existing token sequence, analogous to a text editor. Training sequences are obtained by perturbing ground-truth labels with inverse chained edit operations, and the model learns to edit by undoing those inverses. The perturbation tra
    
[^238]: 基于Transformer的掩码扩散模型可证明地学习具有潜在结构的分布

    Transformer-Based Masked Diffusion Models Provably Learn Distributions with Latent Structure

    [https://arxiv.org/abs/2610.12544](https://arxiv.org/abs/2610.12544)

    本文首次为基于Transformer的掩码扩散模型建立了端到端统计保证，证明其能以由潜在结构（隐马尔可夫树混合模型）决定的样本复杂度学习结构化离散分布，突破了以往理论中 $V^L$ 维度灾难的限制。

    

    掩码扩散模型（MDMs）通过使用学习到的掩码预测器迭代地并行解除多个词元的掩码来生成离散数据。其输出包含两类误差：由并行采样导致的并行化误差，以及由有限样本学习掩码预测器所产生的估计误差。现有理论将这两类误差分开处理，且已有的统计界通常随词表大小为 $V$、长度为 $L$ 的序列总数 $V^L$ 而扩展，饱受维度灾难之苦。在本工作中，我们建立了端到端的统计保证，表明基于Transformer的掩码扩散模型能够学习结构化分布，其样本复杂度由数据的潜在结构决定。具体而言，我们通过隐马尔可夫树的混合模型来刻画这种结构，该模型能够捕捉组份内部的潜在依赖关系以及组份之间的异质性。当数据分布能够被此类混合模型很好地近似时，基于Transformer的……

    arXiv:2610.12544v1 Announce Type: new  Abstract: Masked diffusion models (MDMs) generate discrete data by iteratively unmasking multiple tokens in parallel using a learned mask predictor. Their output carries two errors: parallelization error due to parallel sampling, and estimation error from the mask predictor learned with finite samples. Existing theory treats these errors separately, and the available statistical bounds typically scale with the number $V^L$ of length-$L$ sequences over a vocabulary of size $V$, suffering from the curse of dimensionality. In this work, we establish end-to-end statistical guarantees showing that transformer-based MDMs learn structured distributions with a sample complexity governed by their latent structure. Specifically, we model this structure through mixtures of hidden Markov trees, which capture latent dependencies within components and heterogeneity across components. When the data distribution is well approximated by such a mixture, a transform
    
[^239]: 等预算蒙特卡洛树搜索与随机重启列表调度在DAG工作流上的对比：一项受控等预算计算研究

    Equal-Budget Monte Carlo Tree Search versus Randomized-Restart List Scheduling for DAG Workflows: A Controlled Equal-Budget Computational Study

    [https://arxiv.org/abs/2610.12542](https://arxiv.org/abs/2610.12542)

    在严格控制的等计算预算实验中，简单的随机重启列表调度方法在DAG工作流调度上比蒙特卡洛树搜索（MCTS）好2.88%，表明更复杂的搜索方法并不必然优于简单基线。

    

    蒙特卡洛树搜索（MCTS）常被提出用于调度有向无环图（DAG）工作流，但大多数研究从未进行过可能推翻该结论的对比实验——即给予一个简单得多的方法相同的搜索预算。我们在受控协议下进行了这项对比实验。所有方法分支获得完全相同的预算——构建和评估的候选调度数量相同：一个MCTS调度方法（预注册为主要分支）；一个不含搜索树的随机重启列表调度器（作为对照组）；以及一个教科书式UCT的MCTS分支（作为在同一领域上的注册后续实验）。分析计划在数据收集之前就已锁定；一名独立检查员确认了确认阶段的全部768个调度均有效且完成时间计算正确。在主预算为2048个被评估调度的条件下，MCTS比随机重启方法差2.88%（95%置信区间[-3.62%, -2.15%]；在96个保留实例中有72个相对于基线存在正向提升空间）。这一性能劣势并非源于预算不足（摘要在此处被截断）。

    arXiv:2610.12542v1 Announce Type: new  Abstract: Monte Carlo Tree Search (MCTS) is often proposed for scheduling directed-acyclic-graph (DAG) workflows, yet most studies never run the comparison that could refute it - giving a much simpler method the same search effort. We run that comparison under a controlled protocol. All arms receive an identical budget - the same number of candidate schedules built and evaluated: an MCTS scheduling method, preregistered as the main arm; a randomized-restart list scheduler with no search tree as the control; and a textbook-UCT MCTS arm as a registered follow-up on the same domain. The analysis plan was locked before data collection; an independent checker confirmed all 768 confirmation-stage schedules valid with correct finish times. At the main budget of 2048 evaluated schedules, MCTS is 2.88% worse than the restarts (95% CI [-3.62%,-2.15%]; the 72 of 96 held-out instances with positive baseline-relative headroom). The deficit is not budget starva
    
[^240]: 面向二维与三维拉普拉斯问题的增强全纯神经网络

    Enriched Holomorphic Neural Networks for 2D and 3D Laplace Problems

    [https://arxiv.org/abs/2610.12541](https://arxiv.org/abs/2610.12541)

    本文提出将全纯神经网络与可训练的奇异增强函数相结合，利用构造性调和性免去内部残差、仅凭边界数据求解含角点奇异性的二维与三维拉普拉斯问题，精度较现有最优方法提升数个数量级至约1.7-2.1倍。

    

    拉普拉斯方程的边值问题在角点、边缘和交汇处附近会产生奇异行为，而光滑逼近器对此难以很好地处理。我们将全纯神经网络与由局部几何和边界条件导出的奇异函数相结合。该表示在构造上即为调和函数，因此无需内部残差：网络仅依靠边界数据拟合光滑余项，而增强项负责捕捉奇异行为。奇异指数是可训练的，因此即使局部展开形式事先未知，该构造方法依然适用。我们利用沿复零方向求值的独立网络，将该方法扩展至三维。在所有测试问题上，增强模型都是精度最高的方法。当增强项能精确捕捉奇异性时，其优势达到数个数量级；在一个CdZnTe辐射探测器问题上，其精度比次优方法高出约1.7-2.1倍。

    arXiv:2610.12541v1 Announce Type: new  Abstract: Boundary-value problems for the Laplace equation develop singular behaviour near corners, edges and junctions that smooth approximators resolve poorly. We combine holomorphic neural networks with singular functions derived from the local geometry and boundary conditions. The representation is harmonic by construction, so no interior residual is needed: the network fits the smooth remainder from boundary data alone, and the enrichment terms capture the singular behaviour. Singular exponents are trainable, so the construction also applies when the local expansion is not known in advance. We extend the approach to three dimensions using independent networks evaluated along complex null directions. The enriched model is the most accurate tested method on every problem. Its margin is several orders of magnitude where the enrichment captures the singularity exactly, and a factor of about 1.7-2.1 over the next-best method on a CdZnTe radiation 
    
[^241]: 结果依赖截断与删失下生存分析模型的高效学习

    Efficient Learning of Survival Analysis Models Under Outcome-Dependent Truncation and Censoring

    [https://arxiv.org/abs/2610.12538](https://arxiv.org/abs/2610.12538)

    该论文证明了在结果依赖的截断和右删失数据下，当基线风险函数已知时，非凸的Cox回归可以通过一阶优化方法高效学习，并为需要联合学习基线风险函数的情形构建了可识别且设定正确的模型。

    

    现代机器学习模型通常是在假设训练数据能够代表模型未来所面对的预测任务的前提下进行训练的，但实际数据集往往违背这一假设。我们关注结果依赖的截断问题，即一个观测是否出现在训练数据中取决于其标签值。我们的工作连接了两个基本相互独立的文献领域：截断数据下的高效学习与生存分析。现有的截断生存分析方法并未刻画优化问题的整体结构（优化景观），而是依赖于期望最大化（EM）等迭代算法。我们填补了这一空白，证明了当基线风险函数已知时（尽管问题是非凸的），Cox回归可以在截断和右删失数据下通过一阶优化方法被高效学习。相反，当基线风险函数需要联合学习时，我们构建了可识别且设定正确的模型，其负对数似然……（摘要在此处截断）

    arXiv:2610.12538v1 Announce Type: new  Abstract: Modern machine learning models are typically trained on data assumed to be representative of the predictive tasks the model will face in the future, but practical datasets violate this assumption. We focus on outcome-dependent truncation in which an observation appears in the training data depending on its label. Our work connects two largely distinct literatures: efficient learning under truncation and survival analysis. Existing methods for truncated survival analysis do not characterize the optimization landscape and instead rely on iterative procedures such as expectation maximization. We address this gap and prove that Cox regression can be efficiently learned under truncated and right-censored data using first-order optimization when the baseline hazard is known (despite nonconvexity). In contrast, when the baseline hazard is learned jointly, we construct identifiable and correctly specified models whose negative log-likelihood con
    
[^242]: 基于因果基础模型的市场冲击估计

    Market Impact Estimation via Causal Foundation Models

    [https://arxiv.org/abs/2610.12537](https://arxiv.org/abs/2610.12537)

    本文提出一种在合成市场先验上训练的连续时间因果基础模型（PFN），用于无偏估计市场冲击，证明经典价格-交易流回归方法在知情信息流混淆下会高估市场冲击（最高达68%），而该模型即使在高混淆情形下也能逼近无偏真值。

    

    当交易者执行大额订单时，相关的问题不是“价格将会是多少？”，而是“如果我没有交易，价格本会是多少？”因此，市场冲击本质上是一个因果量，然而经典模型（Almgren-Chriss模型、传播模型）将价格对交易流进行回归，当知情交易流同时通过信息和交易行为推动价格时，这种回归是有偏的。我们提出了一个概念验证性的“因果市场模拟器”：一个连续时间的因果先验数据拟合网络（PFN），该网络在合成市场先验上训练，其中隐藏的信息流在构造上混淆了交易流与价格的关系，且混淆强度是一个精确的、可逐回合调节的旋钮。在该先验上，朴素估计器的偏差遵循一个我们经实证确认的封闭形式（高达68%的高估），而PFN在高混淆情况下仍能逼近无偏的真实参数预言机。其“do消融孪生模型”——即移除干预输入的同一模型——则坍缩至……（摘要在此处被截断）

    arXiv:2610.12537v1 Announce Type: new  Abstract: When a trader executes a large order, the relevant question is not \emph{what will the price be?} but \emph{what would the price have been had I not traded?} Market impact is thus a causal quantity, yet classical models (Almgren--Chriss, propagators) regress price on flow, which is biased whenever informed flow moves prices through information as well as trading. We present a proof-of-concept \emph{Causal Market Simulator}: a continuous-time causal prior-data fitted network (PFN) trained on a synthetic market prior in which a hidden informational flow confounds the flow--price relation by construction, with confounding strength an exact per-episode knob. On this prior, the naive estimator's bias follows a closed form we confirm empirically (up to 68\% overestimation), while the PFN tracks the unbiased true-parameter oracle at high confounding. Its \emph{do-ablated twin}, the same model with the intervention inputs removed, collapses onto
    
[^243]: 大语言模型驱动的音色生成与插值中的结构能动性与表征能动性

    Structural and Representational Agency in LLM-Driven Timbre Generation and Interpolation

    [https://arxiv.org/abs/2610.12532](https://arxiv.org/abs/2610.12532)

    本研究通过对一个实际运行的LLM驱动音色界面进行测量，首次证明用户真实可控的声音空间（可精确刻画为维度随放置声音数递增的zonotope）与界面中由语言模型自我报告的位置（误差约为该空间宽度的三倍）是可分离的，揭示了LLM创意界面中“实际能动性”与“表征能动性”之间的脱节。

    

    生成式创意界面通常会报告用户在可能性空间中所处的位置。这类显示暗示所展示的量值是从作品本身测量得到的；但在越来越多的系统中，这些量值实际上是由一个描述其自身输出的语言模型预测出来的，而两者在屏幕上无法区分。大多数创意领域缺乏公认的测量标准来检验这种自我报告；而音频领域恰好存在这样的标准。在一项针对实际运行的大语言模型驱动音色界面的测量研究中，用户所能实际产生的效果被证明与界面所报告的内容是可分离的。前者可以被精确刻画：可达集是一个zonotope（区域多胞体），每放置一个声音其维度增加一，因此模型可以在其中行动的空间显然属于用户自身。后者则不准确：显示的位置与实际测量值相差约为该区域宽度的三倍，且该误差有两个独立的来源：模……（原文摘要在此截断）

    arXiv:2610.12532v1 Announce Type: cross  Abstract: Generative creative interfaces routinely report the user's position within a space of possibilities. Such displays imply that the quantities shown were measured from the artefact; in a growing class of systems they are instead predicted by a language model describing its own output, and the two are indistinguishable on screen. Most creative domains lack an agreed measurement against which such a self-report can be checked; audio admits one. In a measurement study of a working LLM-driven timbre interface, what a user can cause proves separable from what the interface reports. The first is exactly characterisable: the reachable set is a zonotope whose dimension increases by one per sound placed, so the space the model may act within is demonstrably the user's own. The second is inaccurate. The displayed position differs from measurement by approximately three times the width of that region, and the error has two independent sources: mode
    
[^244]: CacheScope：面向无训练扩散加速的精确计算分配方法

    CacheScope: Exact-Compute Allocation for Training-Free Diffusion Acceleration

    [https://arxiv.org/abs/2610.12530](https://arxiv.org/abs/2610.12530)

    提出CacheScope，通过以输入位移归一化探测输出变化来在线识别高风险缓存复用步骤，并选择性回退到完整去噪计算，实现无需训练的扩散模型加速。

    

    无训练特征复用通过在选定步骤用缓存的近似结果替代完整的去噪计算来加速扩散Transformer的推理。现有方法主要致力于改进被跳过特征的重构，但我们发现，一旦采用了稳定的一阶重构器，继续增加重构复杂度带来的收益十分有限，而精确计算步骤的放置位置对保真度的影响要大得多。更重要的是，绝对的浅层探测变化同时受输入轨迹位移和网络响应强度的共同影响，因此并不能直接反映缓存风险。我们通过用相应的输入位移对探测输出变化进行归一化来解决这一混淆问题，从而得到每单位轨迹移动的方向性增益。基于这一信号，我们提出CacheScope，它能够在线识别高风险的缓存复用步骤，并选择性地回退到完整的去噪计算。

    arXiv:2610.12530v1 Announce Type: new  Abstract: Training-free feature reuse accelerates diffusion transformer inference by replacing full denoising computation at selected steps with cached approximations. Existing methods mainly improve the reconstruction of skipped features, but we find that once a stable first-order reconstructor is used, increasing reconstruction complexity brings limited gains, whereas the placement of exact-computation steps has a much larger effect on fidelity. More importantly, absolute shallow-probe variation is jointly affected by input-trajectory displacement and network response strength, and therefore does not directly reflect cache risk. We address this confounding by normalizing probe-output variation with the corresponding input displacement, yielding a directional gain per unit trajectory movement. Based on this signal, we propose CacheScope, which identifies high-risk reuse steps online and selectively falls back to full denoising computation while l
    
[^245]: GlucoPRISM：面向跨队列血糖迁移的协议监督预训练

    GlucoPRISM: Protocol-Supervised Pretraining for Cross-Cohort Glucose Transfer

    [https://arxiv.org/abs/2610.12528](https://arxiv.org/abs/2610.12528)

    GlucoPRISM 提出协议监督的预训练方法，将 CGM 表示划分为对日期和传感器均不变的特质带、仅随日期变化的状态带以及跟踪设备的传感器带，使部署时无需目标数据即可通过固定投影去除损害跨队列迁移的测量成分。

    

    连续血糖监测（CGM）基础模型将每一天编码为一个向量，该向量混合了这个人是谁、那天发生了什么以及传感器如何进行测量，因此测量差异会随表示一同进入新的队列。我们探究预训练能否将损害迁移能力的测量成分置于已知坐标中，使部署阶段无需目标数据、仅用固定投影即可将其去除。GlucoPRISM 通过“响应契约”来监督 GlucoFM 骨干网络，即规定仅改变日期或仅改变传感器时表示的各个部分应如何变化：64 维的特质带对两者均不应变化，48 维的状态带仅跨天变化，而 16 维的传感器带应跟踪设备。通过在配对的 Dexcom 与 Libre 记录上校准的合成传感器视图、不同日期的配对以及血糖摘要目标来提供这种监督，且无需任何临床标签。在十二个跨队列（实验中）……

    arXiv:2610.12528v1 Announce Type: new  Abstract: Continuous glucose monitoring (CGM) foundation models encode each day as one vector that mixes who the person is, what happened that day and how the sensor measured it, so measurement differences travel with the representation into new cohorts. We ask whether pretraining can place the measurement component that harms transfer in known coordinates, so that deployment drops them with a fixed projection and no target data. GlucoPRISM supervises a GlucoFM backbone with response contracts, rules for how each part of the representation should change when only the day or only the sensor changes: a 64-dimensional Trait band should change with neither, a 48-dimensional State band only across days, and a 16-dimensional Sensor band should track the device. Synthetic sensor views calibrated on paired Dexcom and Libre recordings, different-day pairs and glucose-summary targets supply this supervision without clinical labels. Across twelve cross-cohor
    
[^246]: 从净智能电表实现表后光伏发电可见性：部署模式与运营洞察

    Behind-the-Meter Photovoltaic Visibility from Net Smart Meters: Deployment Regimes and Operational Insights

    [https://arxiv.org/abs/2610.12527](https://arxiv.org/abs/2610.12527)

    本文提出四种部署模式来系统评估基于深度学习的净智能电表光伏分解方法，揭示了模型在已知用户、新用户、未来时期和新区域等不同场景下的性能差异，为电力运营商提供了实际部署的关键运营洞察。

    

    配电网运营商需要对表后光伏（PV）发电的可见性，然而大多数住宅用户仅通过净智能电表进行观测。基于深度学习的光伏分解技术可以恢复这种可见性，但其运营价值取决于模型是服务于已知用户、新用户、未来时期还是新区域。我们利用四种部署模式来量化这种依赖关系，这些模式将家庭间迁移与时间外推区分开来，应用于Ausgrid智能电表数据，并进行跨数据集适应到Pecan Street数据。采用固定的训练协议，我们在与部署模式匹配的评估下，对代表性的深度序列架构进行光伏分解基准测试。在固定的48小时上下文长度下，最佳模型的日间平均绝对误差（MAE）从已见家庭在未来测试期的0.104千瓦上升到同期未见家庭的0.185千瓦，并且在固定训练条件下……

    arXiv:2610.12527v1 Announce Type: new  Abstract: Distribution operators need visibility of behind-the-meter photovoltaic (PV) generation, yet most residential customers are observed only through net smart meters. Deep-learning-based PV disaggregation can restore this visibility, but its operational value depends on whether the model serves known customers, new customers, future periods, or new regions. We quantify this dependence using four deployment regimes that separate household transfer from temporal extrapolation, applied to Ausgrid smart-meter data with cross-dataset adaptation to the Pecan Street data. Using a fixed training protocol, we benchmark representative deep sequence architectures for PV disaggregation under regime-matched evaluation. At a fixed 48-hour context length, daylight mean absolute error (MAE) of the best model rises from 0.104~kW on seen households in the future test period to 0.185~kW on unseen households over the same period, and under a fixed training con
    
[^247]: MolP2SBench：大语言模型真的能从分子性质推理出分子结构吗？

    MolP2SBench: Do LLMs Really Reason from Molecular Properties to Structure?

    [https://arxiv.org/abs/2610.12522](https://arxiv.org/abs/2610.12522)

    提出MolP2SBench基准，通过遮蔽分子骨架、性质打乱与反事实测试系统评估大语言模型从分子性质到结构的推理能力，发现模型虽能利用物理性质预测结构，但未必真正依赖正确的性质-结构配对。

    

    大语言模型（LLM）在分子发现和结构解析中的应用日益增多，而这些应用中的决策依赖于物理观测结果。现有评估主要集中于端到端的分子重建，在这种设置下，正确的预测并不一定意味着模型真正依赖于所提供的性质信息。我们提出了MolP2SBench，一个用于评估大语言模型“性质到结构”（P2S）推理能力的基准。每个任务给出一个带有遮蔽部分的分子骨架及其性质信息，要求模型从合理的候选结构中选出缺失的结构。我们通过打乱或替换性质信息，来测试模型对正确“性质-结构”配对的依赖程度以及对反事实性质变化的响应能力。我们采用基于通道的排行榜，使用“性质锚定恢复率”（PGR）指标，对原始恢复和反事实恢复给予同等权重。通过对七个模型的评估，我们发现大语言模型能够利用物理性质进行结构预测，但性能表现各异……

    arXiv:2610.12522v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in molecular discovery and structure elucidation, where decisions depend on physical observations. Existing evaluations primarily focus on end-to-end molecular reconstruction, where correct predictions do not necessarily indicate reliance on the supplied properties. We introduce MolP2SBench, a benchmark for molecular property-to-structure (P2S) reasoning in LLMs. Each task presents a masked molecular scaffold with given properties and asks the model to select the missing structure from plausible candidates. We shuffle or replace properties to test reliance on correct property-structure pairing and responsiveness to counterfactual property changes. A channel-wise leaderboard uses Property-Grounded Recovery (PGR), giving equal weight to source and counterfactual recovery. Evaluating seven models, we find that LLMs can use physical properties for structure prediction, but performance vari
    
[^248]: CircuitTree：将电路组合成音节转写器

    CircuitTree: Composing Circuits into A Syllable Transcriber

    [https://arxiv.org/abs/2610.12520](https://arxiv.org/abs/2610.12520)

    该论文提出CircuitTree数据结构及贪心密度缩放组合算法，将39个从HuBERT中提取的音素分类电路高效组合成单个模型，实现了任何单个电路都不具备的音节转写能力，以64.8%的准确率转写开音节，仅比完全不共享电路的上限低1.2个百分点。

    

    电路发现的目标是将庞大的单一神经网络分解为微小的模块化子网络，这些子网络是可解释的，因为每个子网络执行一个可识别的子任务。我们开发了一种将任意数量的电路组合成单个模型的实用方法。我们提出了一种贪心的、密度缩放的组合算法，克服了以往通过简单并集来组合电路的尝试的局限性，然后通过CircuitTree将其推广到N个电路，CircuitTree是一种基于Θ(N log N)树组合调度构建的新数据结构。我们将从HuBERT中剪枝得到的用于分类单个音素的电路组合成一个音节分类器，这是39个音素电路中任何一个单独都不具备的能力。将CircuitTree串联成电路层次结构流水线，我们以64.8%的准确率转写开音节，与完全不共享任何电路的66.0%上限相差仅1.2个百分点，同时共享了99.5%的电路组件。

    arXiv:2610.12520v1 Announce Type: cross  Abstract: The goal of circuit discovery is to decompose a monolithic neural network into tiny modular subnetworks which are interpretable in the sense that each does an identifiable subtask. We develop a practical method for composing arbitrarily many circuits into a single model. We propose a greedy, density-scaled composition algorithm that overcomes limitations of previous attempts to compose circuits by naive union, then generalize it to $N$ circuits via CircuitTree, a new data structure built on a $\Theta(N\log N)$ tree-compose schedule. We compose circuits pruned from HuBERT for classifying individual phonemes into a syllable classifier, a capability none of the 39 phoneme circuits has individually. Chaining CircuitTrees into a hierarchy-of-circuits pipeline, we transcribe open syllables at 64.8\% accuracy, within 1.2 points of a 66.0\% ceiling that shares no circuitry at all, with 99.5\% of circuit components shared.
    
[^249]: 在路灯下寻找：生成式分子动力学的评估问题

    Looking Under the Streetlight: Evaluation in Generative Molecular Dynamics

    [https://arxiv.org/abs/2610.12518](https://arxiv.org/abs/2610.12518)

    这篇立场论文指出，生成式分子动力学领域的评估实践因基准真值获取代价高昂而结构性偏离了科学目标，导致评估依赖便利的替代指标，并据此提出更贴合领域追求的具体可观测量、报告标准和计算与实验基准设计方案。

    

    本立场论文认为，生成式分子动力学社区未能正确评估其研究进展：当前的评估方式使得模型只需匹配所报告的指标，而在所声称的科学任务上却可能出现任意大的偏差，这带来了夸大对经典基线方法优势的风险。这反映了常见评估实践与该领域为自身设定的科学任务之间存在结构性错位。究其原因，作为基准真值的、对目标系统进行穷尽式分子动力学模拟在计算上代价高昂，而这一成本恰恰是推动该领域发展的动力。由于基准真值难以获得，评估便逐渐滑向便利的替代指标，由此形成的评估规范也容易固化为惯例。我们识别出了作为社区层面均衡状态而持续存在的错位评估模式，并提出具体的可观测量、报告标准以及基准测试设计方案（涵盖计算与实验两方面），以使评估能更好地反映生成式分子动力学所追求的真正目标。

    arXiv:2610.12518v1 Announce Type: cross  Abstract: This position paper argues that the generative molecular dynamics (GenMD) community is not properly evaluating its progress: current evaluations let a model match what is reported while being arbitrarily wrong on what was claimed, which risks over-claiming advantage over classical baselines. This reflects a structural misalignment between common evaluation practices and the scientific tasks the field sets for itself. The reason is that the ground truth---exhaustive molecular dynamics on the systems of interest---is computationally costly, and this cost is what motivates the field. With ground truth inaccessible, evaluation drifts toward convenient proxies, and the resulting norms entrench easily. We identify misaligned evaluation patterns that persist as a community-level equilibrium, and propose concrete observables, reporting standards, and benchmark designs, both computational and experimental, that better reflect what GenMD aspires
    
[^250]: CORAL-KV：面向自回归视频KV缓存的紧凑跨层预测与残差编码

    CORAL-KV: Compact Cross-Layer Prediction and Residual Coding for Autoregressive Video KV Caches

    [https://arxiv.org/abs/2610.12514](https://arxiv.org/abs/2610.12514)

    提出Coral-KV框架，利用视频模型KV缓存中强烈的跨层依赖关系，通过紧凑的块对角加低秩预测器跨层预测键值并对预测残差进行量化，从而在自回归长视频生成中高效压缩KV缓存并抑制重建误差随视频增长而累积的问题。

    

    arXiv:2610.12514v1 公告类型：cross 摘要：自回归视频扩散模型维护着持续增长的注意力键值缓存，使内存成为长视频生成的关键瓶颈。近来，基于聚类的量化已成为压缩视频KV缓存的有效方法，它通过将相似的键和值向量分组、再量化残差误差来实现压缩。然而，当后续帧关注被压缩的历史内容时，微小的重建误差会随视频变长而不断累积。除了聚类所利用的层内相似性之外，我们发现视频模型的缓存各层之间存在强烈的跨层依赖关系。这促使我们提出Coral-KV——离线校准、残差编码、局部自适应——一个KV缓存压缩框架，它利用每一层的缓存状态来预测后续层的键和值，然后对剩余的预测残差进行量化。我们构建了紧凑的块对角加低秩预测器：块对角分量建模通道关系……

    arXiv:2610.12514v1 Announce Type: cross  Abstract: Autoregressive video diffusion models maintain a growing attention key-value (KV) cache, making memory a key bottleneck for long video generation. Recently, clustering-based quantization has emerged as an effective method to compress video KV caches by grouping similar key and value vectors then quantizing residual errors. However, when later frames attend to the compressed history, small reconstruction errors can compound as the video grows. Beyond the within-layer similarity exploited by clustering, we find strong dependencies across cache layers in video models. This motivates Coral-KV - Calibrated Offline, Residual-coded, and Adapted Locally - a KV-cache compression framework that uses cache states in each layer to predict keys and values in subsequent layers, then quantizes the remaining prediction residuals. We construct compact block-diagonal-plus-low-rank (block-DPLR) predictors: block-diagonal components model channel relation
    
[^251]: EarLog：面向个性化听力风险感知的机会式耳部生理感知

    EarLog: Opportunistic Ear Physiology Sensing toward Personalized Hearing Risk Awareness

    [https://arxiv.org/abs/2610.12501](https://arxiv.org/abs/2610.12501)

    EarLog将耳机播放内容本身转化为机会式探测信号，通过内容条件化残差感知在日常聆听中识别微弱的声反射，实现对个性化听力损伤风险的实时感知。

    

    许多耳机用户在高负荷聆听过程中面临不可逆听力损伤的风险，然而目前的保护手段仍依赖于声压限制。临床检测只能在损伤已经可测量之后才发现问题，使得风险累积的过程处于未被观测的状态。声反射（Acoustic Reflex, AR）是一种对高声学负荷的非自主生理反应，标志着耳朵进入保护状态的时机。然而AR信号十分微弱，现有的测量方法需要专用探头和受控条件，而这些条件在日常音频播放中难以维持。我们提出了EarLog，它将播放内容——最主要的干扰源——转变为机会式的探测信号。EarLog引入了内容条件化的残差感知方法，以抑制由非平稳音频内容、个体耳部声学特性差异和设备响应所引起的变异。该方法通过响度相关的变化以及残差中的局部频谱不规则性来识别微弱的AR信号。一种共振约束的稀疏有限冲激响应（FIR）滤波器……（摘要在此处截断）

    arXiv:2610.12501v1 Announce Type: cross  Abstract: Many headphone users risk irreversible hearing damage during high-load listening, yet current protection still relies on sound-pressure limits. Clinical tests detect damage after it becomes measurable, leaving the risk-accumulating process unobserved. The acoustic reflex (AR), an involuntary response to high acoustic load, marks when the ear enters a protective state. AR traces are weak, and existing measurements require dedicated probes and controlled conditions that are difficult to maintain during everyday audio. We present EarLog, which turns playback, the dominant interference, into an opportunistic probe. EarLog introduces content-conditioned residual sensing to suppress variations caused by nonstationary content, individual ear acoustics, and device response. It identifies weak AR through loudness-dependent changes and localized spectral irregularities in the residual. A resonance-constrained sparse finite impulse response (FIR)
    
[^252]: 基于自发言语的声学生物标志物用于阿尔茨海默病检测

    Acoustic Biomarkers for Alzheimer's Disease Detection from Spontaneous Speech

    [https://arxiv.org/abs/2610.12500](https://arxiv.org/abs/2610.12500)

    本研究证明从“饼干失窃”图片描述等自发言语中提取的韵律、频谱和时域声学特征，结合机器学习分类器，可以有效区分阿尔茨海默病患者与健康人群，为AD的早期筛查提供了一种具有临床意义的语音生物标志物方法。

    

    阿尔茨海默病（AD）仍然是一个重大的公共卫生挑战，早期识别对于及时干预和护理规划至关重要。最近的研究表明，基于语音的数字生物标志物能够捕捉嗓音产生、韵律和时序方面的细微变化，这些变化往往先于或伴随认知衰退出现。在本研究中，我们探讨从自发言语中提取的声学特征是否能够可靠地区分阿尔茨海默病患者与健康对照者。我们使用源自“饼干失窃”图片描述任务的结构化语音数据集，提取了涵盖韵律、频谱和时域的具有临床价值的低级描述符，并评估了多种机器学习分类器。我们的结果表明，声学生物标志物具有很强的判别价值，表现最佳的模型在诊断组之间实现了具有临床意义的区分。这些发现支持了使用……（原文摘要至此截断）

    arXiv:2610.12500v1 Announce Type: cross  Abstract: Alzheimer's disease (AD) remains a major public health challenge, and early identification is crucial for timely intervention and care planning. Recent work has shown that speech-based digital biomarkers can capture subtle changes in vocal production, prosody, and timing that often precede or accompany cognitive decline. In this study, we investigate whether acoustic features extracted from spontaneous speech can reliably distinguish individuals with AD from healthy controls. Using a structured speech dataset derived from the Cookie Theft picture-description task, we extract clinically informative low-level descriptors spanning prosodic, spectral, and temporal domains, and evaluate a range of machine learning classifiers. Our results indicate that acoustic biomarkers carry strong discriminative value, with the best-performing models achieving clinically meaningful separation between diagnostic groups. These findings support the use of 
    
[^253]: 谐振序列模型的输入依赖读取相位

    An Input-Dependent Readout Phase for Resonant Sequence Models

    [https://arxiv.org/abs/2610.12496](https://arxiv.org/abs/2610.12496)

    本文将 ResonatorLM 中每个头固定的读取相位改为当前 token 的函数，仅在读取环节引入输入依赖性，在保持精确卷积和仅增加 0.2% 参数的同时提升了模型能力，并分析了未训练振荡器组的相位可编码时间但线性重建能力受限（约 16 个 token）。

    

    具有固定大小状态的序列模型避免了注意力机制的二次方计算开销，但面临一个设计选择：输入依赖的状态转移能提供内容相关的记忆，但代价是需要基于FFT的训练；而固定转移则保留了 O(T log T) 的卷积路径，却对每个输入都以相同方式读取记忆。ResonatorLM 是最近提出的属于第二类的模型，它通过每个注意力头固定的学习相位来读取一组阻尼复振荡器。我们将该读取相位改为当前 token 的函数，从而仅在读取过程中引入输入依赖性，同时保持卷积的精确性，且仅增加 0.2% 的参数。对未训练振荡器组的分析表明，相位确实编码了经过的时间，但普通 token 会在 2-8 步内衰减到背景水平以下，且任何阻尼选择都无法将线性重建扩展到约 16 个 token 之外。在 WikiText-2 上使用 600 万参数的字符级模型实验中，输入依赖相位与原模型在（原文摘要在此处被截断）

    arXiv:2610.12496v1 Announce Type: cross  Abstract: Sequence models with a fixed-size state avoid the quadratic cost of attention, but face a design choice. An input-dependent state transition gives content-dependent memory at the cost of FFT-based training, while a fixed transition keeps the O(T log T) convolutional path but reads memory the same way for every input. ResonatorLM, a recent model of the second kind, reads a bank of damped complex oscillators through a fixed learned phase per head. We make that readout phase a function of the current token, which adds input dependence to the read only, keeps the convolution exact, and costs 0.2 percent more parameters. An analysis of the untrained bank shows that phase does encode elapsed time, but an ordinary token falls below the background within 2-8 steps and no choice of damping extends linear reconstruction beyond about 16 tokens. In 6M-parameter character-level models on WikiText-2, the input-dependent phase matches the original at
    
[^254]: Kairos：迈向适用于所有模态的字节级基础模型

    Kairos: Toward a Byte-Level Foundation Model for Every Modality

    [https://arxiv.org/abs/2610.12495](https://arxiv.org/abs/2610.12495)

    Kairos 提出了一种在统一字节空间上处理文本、图像、音频、视频、激光雷达等多模态的小型基础模型架构，融合门控 DeltaNet 线性注意力、滑动窗口注意力、注意力-残差聚合、稀疏 MoE 和多尺度字节级编解码器，并通过单机上的小规模过拟合收敛测试验证该组合能否有效优化。

    

    在单台本地机器上训练的小型语言模型几乎没有留下架构试错的空间。Kairos 结合了基于共享投影的双向门控 DeltaNet 线性注意力与滑动窗口注意力（LiZAttention2）、分块的注意力-残差聚合、稀疏混合专家（MoE）前馈层，以及多尺度字节级编解码器，在文本、图像、音频、视频、激光雷达和控制流共享的统一字节空间上，以均匀噪声去噪目标进行训练；本报告在追问这种组合是否有用之前，先追问它究竟能否正常优化。我们给出了每个组件由代码推导出的算法，包括一个跨会话的循环状态记忆门控，随后进行收敛性健全性检查：在 1–15M 参数规模下对单个 440 字节的句子进行过拟合测试（仅文本），其中对编解码器和注意力选择使用三个随机种子，并对另外七个因素进行单次运行的参数扫描。所测试的两个注意力变体在……

    arXiv:2610.12495v1 Announce Type: cross  Abstract: Small language models trained on a single local machine leave little room for architectural trial and error. Kairos combines bidirectional gated-DeltaNet linear attention and sliding-window attention over shared projections (LiZAttention2), block-wise Attention-Residual aggregation, a sparse Mixture-of-Experts (MoE) feed-forward layer and a multi-scale byte-level codec, trained with a uniform-noise denoising objective over one byte space shared by text, image, audio, video, lidar and control streams; this report asks whether the combination optimizes at all before asking whether it is useful. We give code-derived algorithms for each component, including a cross-session recurrent-state memory gate, then run convergence sanity checks: overfittinga single 440-byte sentence at 1--15M parameters, text only, with three seeds for codec and attention choices and single-run sweeps for seven further factors. The two attention variants tested at 
    
[^255]: 面向机器学习应用的代数整数量化

    Algebraic Integer Quantization for Machine Learning Applications

    [https://arxiv.org/abs/2610.12494](https://arxiv.org/abs/2610.12494)

    本文提出代数整数数格式（AINF）及面向大语言模型推理的代数整数量化（AIQ），该框架可将乘累加运算化归为纯整数运算以实现无误差计算，并允许针对具体应用灵活选择代数整数，在性能与精度上均具有竞争力。

    

    本文提出代数整数数格式（AINF）作为一种高度可配置且精确的压缩计算框架。AINF的一个关键特性是它能够直接化归为纯整数运算，从而实现无误差的乘累加操作。我们开发了一个利用这一特性的矩阵乘法引擎，在现代硬件加速器上获得了与最高效数值格式相比在可预测因子范围内的性能。AIQ的另一个关键优势在于可以自由选择适配特定用例数值特征的代数整数，提供了浮点或定点格式从根本上无法实现的优化机会。基于这些特性，我们提出了代数整数量化（AIQ），一种面向大语言模型（LLM）推理的有损压缩格式，并提供了实验结果证明AIQ能够实现具有竞争力的准确率。

    arXiv:2610.12494v1 Announce Type: new  Abstract: In this article, we propose algebraic integer number formats (AINF) as a framework for highly configurable and accurate compressed computation. A key property of AINF is its straightforward reduction to purely integer arithmetic, enabling error-free multiply-accumulate operations. We develop a matrix multiplication engine that exploits this property, obtaining performance within a predictable factor of the most efficient numerical counterparts on modern hardware accelerators. Another key advantage of AIQ is the freedom to select algebraic integers tailored to the numerical characteristics of a given use case, offering optimization opportunities that are fundamentally unavailable in floating-point or fixed-point formats. Motivated by these properties, we introduce algebraic integer quantization (AIQ), a lossy compression format for large language model (LLM) inference, and provide results demonstrating that AIQ achieves competitive accura
    
[^256]: 无锚点的忠实性：关于可解释AI特征重要性忠实性度量结构性错位的立场论文

    Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance

    [https://arxiv.org/abs/2610.12493](https://arxiv.org/abs/2610.12493)

    本文是一篇立场论文，指出当前面向特征重要性的可解释AI忠实性度量存在结构性错位，并强调XAI实际上更多扮演调试工具的角色而非建立信任的手段。

    

    可解释人工智能（XAI）工具通常被呈现为一种通过使机器学习模型算法的某些方面更易于人类理解来提升对模型信任的方法。然而，过去十年积累的证据表明，这一目标总体上未能实现，反而促进了对错误预测的过度依赖，进而可能导致算法厌恶（algorithmic aversion）。实际上，XAI似乎主要作为一种调试工具发挥作用，有助于获得关于模型所学到的特征等方面的经验性洞察。造成这一局限性的主要问题之一在于，XAI方法常常过度简化模型的内部工作机制，导致解释与模型本身之间出现错位——这种行为被称为不忠实性（unfaithfulness）或不忠（infidelity）。忠实性可以被理解为解释与模型之间的近似程度。以严格的方式计算忠实性……

    arXiv:2610.12493v1 Announce Type: new  Abstract: Explainable Artificial Intelligence (XAI) tools have often been presented as methods to increase trust in machine learning models by making aspects of their algorithms more human-understandable. However, evidence accumulated over the last decade points to an overall failure of this objective, instead promoting overreliance on incorrect predictions, which can subsequently lead to algorithmic aversion. Rather, XAI appears to function primarily as a debugging tool, useful for obtaining anecdotal insights into, for example, features learned by the model. One of the main issues underlying this limitation is that XAI methods often oversimplify the inner workings of the model, resulting in misalignment between the explanation and the model itself - a behavior referred to as unfaithfulness or infidelity. Faithfulness can be interpreted as the degree of approximation between an explanation and the model. Computing faithfulness in a rigorous manne
    
[^257]: TTS-Q：面向文本到语音扩散变换器的感知引导训练后量化

    TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers

    [https://arxiv.org/abs/2610.12474](https://arxiv.org/abs/2610.12474)

    本文首次对十亿参数规模的TTS扩散变换器开展4比特权重与激活（W4A4）训练后量化研究，揭示了量化主要损害说话人音色而非可懂度这一现象及其来源，并提出感知引导的量化修复方法。

    

    零样本文本到语音（TTS）技术只需几秒钟的音频即可克隆一个声音。然而其高质量的生成依赖于数十亿参数的扩散变换器，其庞大的模型规模和多步采样过程使得最需要该技术的端侧设备、辅助技术和实时应用难以使用。训练后量化（PTQ）无需重新训练即可压缩此类模型，但现有方法是为语言和视觉领域设计的，而非语音领域。在4比特精度下，强量化器虽然能让语音可懂度接近全精度水平，但会持续损害说话人音色，导致错误率保持较低的同时，声音却偏离了目标说话人。我们探究了为什么量化会损害语音、损伤存在于何处，以及能否在不重新训练的情况下进行修复。为此，我们对1B和3.5B规模的TTS扩散变换器开展了首个4比特权重与激活（W4A4）训练后量化研究。研究发现，音色损失同时来自权重量化和激活量化，但只有激活部分具有时间维度的特性……（原文摘要在此处截断）

    arXiv:2610.12474v1 Announce Type: cross  Abstract: Zero-shot text-to-speech (TTS) can clone a voice from seconds of audio. Yet the quality comes from billion-parameter diffusion transformers whose size and multi-step sampling put them out of reach of the on-device, assistive, and real-time applications that need them most. Post-training quantization (PTQ) shrinks such models without retraining, but it has been built for language and vision, not speech. At 4 bits, a strong quantizer leaves intelligibility close to full precision but consistently erodes speaker timbre, so error rates stay low while the voice drifts from the target speaker. We ask why quantization hurts speech, where the damage lives, and whether it can be repaired without retraining. To find out, we run the first 4-bit weight-and-activation (W4A4) PTQ study of a TTS diffusion transformer at 1B and 3.5B. The timbre cost comes from both the weight and the activation quantization, but only the activation part has the timest
    
[^258]: 神经形态卫星智能：面向事件驱动架构、基于脉冲的算法与定量化能耗-精度权衡的形式化框架

    Neuromorphic Satellite Intelligence: A Formal Framework for Event-Driven Architectures, Spike-Based Algorithms, and Quantitative Energy-Accuracy Trade-offs

    [https://arxiv.org/abs/2610.12471](https://arxiv.org/abs/2610.12471)

    本文建立了卫星脉冲神经计算的形式化数学模型，提出三种原创脉冲算法（用于姿态估计的SKCF、用于健康监测的TCAD、用于认知卫星链路的SAMC），并引入定量的能耗-精度-延迟（EAL）权衡曲面。

    

    卫星星载计算面临一个不可回避的矛盾：任务智能需求持续增长，而功耗、质量和辐射预算却保持固定。本文在三个方面推进了该领域的最新进展。首先，我们建立了面向卫星工作负载的脉冲神经计算的形式化数学模型，推导出能耗关于网络稀疏度、放电率和突触扇出的闭式表达式。其次，我们提出了三种原创算法——用于姿态估计的脉冲卡尔曼-互补滤波器（SKCF）、用于航天器健康监测的时间对比异常检测器（TCAD）、以及用于认知卫星链路的基于脉冲的自适应调制与编码（SAMC）方案——每种算法均附有伪代码和解析复杂度界限。第三，我们引入了一个定量的能耗-精度-延迟（EAL）权衡曲面，并利用从已发表硅基测量数据推导的数值估计对其进行填充。

    arXiv:2610.12471v1 Announce Type: cross  Abstract: Satellite onboard computing faces an irreducible tension: mission intelligence requirements grow monotonically while power, mass, and radiation budgets remain fixed. This paper advances the state of the art on three fronts. First, we develop a formal mathematical model of spiking neural computation for satellite workloads, deriving closed-form expressions for energy consumption as a function of network sparsity, firing rate, and synaptic fan-out. Second, we propose three original algorithms -- a Spiking Kalman-Complementary Filter (SKCF) for attitude estimation, a Temporal Contrast Anomaly Detector (TCAD) for spacecraft health monitoring, and a Spike-Based Adaptive Modulation and Coding (SAMC) scheme for cognitive satellite links -- each with pseudocode and analytical complexity bounds. Third, we introduce a quantitative Energy-Accuracy-Latency (EAL) trade-off surface and populate it with numerical estimates derived from published sili
    
[^259]: 关于AI时间视野估计与有效性的研究——从统计学视角审视METR图表

    On the estimation and validity of AI time horizons---a statistical look at the METR plot

    [https://arxiv.org/abs/2610.12466](https://arxiv.org/abs/2610.12466)

    本文用样条函数和项目反应理论重新估计了METR的AI时间视野，发现人类时间与AI难度的关系并非全局线性（在2至30分钟区间近乎平坦），因此同样10倍的提升，从3分钟到30分钟远比从30分钟到5小时容易实现。

    

    METR的50%时间视野衡量的是AI以50%概率成功解决的软件任务所需的人类完成时间，从而能以可解释的单位表达AI能力。本文基于228个任务和26个AI，使用样条函数和项目反应理论重新计算时间视野，放宽了“任务对AI的难度与人类时间的对数呈线性关系”这一假设。我们拟合出的样条可被解释为将人类时间转换为AI难度的函数；它在2至30分钟的区间内近乎平坦，但在其他区域接近线性。因此，尽管倍数同为10倍，时间视野从3分钟跃升至30分钟远比从30分钟跃升至5小时容易。总体而言，我们贡献了在一组经交叉验证的适当评分规则下表现更优的时间视野点估计，以及用于评估时间视野构建效度的诊断图。我们建议时间视野应结合……（原文此处截断）

    arXiv:2610.12466v1 Announce Type: new  Abstract: METR's 50\% time horizon measures the human completion time of software tasks that an AI solves with 50\% probability, allowing AI capabilities to be expressed in interpretable units. On 228 tasks and 26 AIs, we recompute the time horizons using splines and item-response theory to relax the assumption that the AI difficulty of a task depends linearly on the log of human time. Our fitted spline can be interpreted as a function that \emph{converts} human time to AI difficulty; it is nearly flat in a region from 2--30 min but close to linear elsewhere. Hence, a time-horizon jump from 3 min to 30 min is much easier than one from 30 min to 5 hours despite the same multiplier of $10 \times$. Overall, we contribute time-horizon point estimates that perform better under a cross-validated suite of proper scoring rules, as well as diagnostic plots for assessing time horizons' construct validity. We suggest that time horizons be interpreted togethe
    
[^260]: asdex：JAX 中的自动稀疏微分

    asdex: Automatic Sparse Differentiation in JAX

    [https://arxiv.org/abs/2610.12336](https://arxiv.org/abs/2610.12336)

    本文提出了 JAX 中的自动稀疏微分工具 asdex，通过稀疏模式检测、图着色、压缩微分与解压缩四个步骤，利用雅可比矩阵的稀疏结构，使自动微分传递次数与问题维度无关，从而大幅提升计算效率。

    

    科学计算和机器学习中的许多任务都需要函数的雅可比矩阵或海森矩阵。自动微分（AD）能够以机器精度计算这些导数，但要生成一个稠密的 m×n 雅可比矩阵，需要进行 n 次前向模式或 m 次反向模式的 AD 传递，即每个列或行各一次。对于一大类函数而言，每个输出仅依赖于少数几个输入，这使得导数矩阵是稀疏的。自动稀疏微分（ASD）通过四个步骤利用这一结构特性：检测与输入无关的稀疏模式、对图进行着色以将可共享一次 AD 传递的列或行分组、通过每个颜色一次 AD 传递的压缩微分计算压缩导数矩阵，最后将其解压缩还原至原始稀疏模式。颜色数量（即 AD 传递次数）通常与问题的维度无关：例如，具有 b 个连续带的带状雅可比矩阵只需要……

    arXiv:2610.12336v1 Announce Type: cross  Abstract: Many tasks in scientific computing and machine learning require the Jacobian or Hessian matrix of a function. Automatic differentiation (AD) computes these derivatives to machine precision, but materializing a dense $m \times n$ Jacobian requires $n$ forward-mode or $m$ reverse-mode AD passes, one per column or row. For a large class of functions, each output depends on only a few inputs, making the derivative matrix sparse. Automatic sparse differentiation (ASD) exploits this structure in four steps: detection of the input-agnostic sparsity pattern, coloring of a graph to group columns or rows that can share an AD pass, compressed differentiation to compute a compressed derivative matrix with one AD pass per color, and finally decompression into the original sparsity pattern. The number of colors, and hence of AD passes, is often independent of the problem dimension: a banded Jacobian with $b$ contiguous bands, for instance, only ever
    
[^261]: 排练一切，记住无物：Attic-KV 只排练将被读取的内容

    Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read

    [https://arxiv.org/abs/2610.12133](https://arxiv.org/abs/2610.12133)

    该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。

    

    许多键值（KV）缓存会在无人知晓将来要被问及什么之前就被压缩：为检索而缓存的文档、跨请求共享的提示前缀、长对话的记忆。主流方法通过“排练”来为KV条目打分：模型重新阅读上下文并保留其关注的条目，并假设缓存对其上下文排练得越完整，就记得越牢。我们证明，在紧张的预算下这一假设会适得其反：排练一切，记住无物。在3%的保留率下，重读整个上下文在RULER上仅能保留96.5分中的31.5分，而在LongBench的自然文本任务上，其表现甚至低于完全不排练的方法。原因在于缓存会保留它所排练的内容：重读会将预算分散到整个上下文，导致答案自身的条目仅以略高于随机的概率得以保留。正如考试前的学生，缓存通过自我测试比通过重读能记得更多。

    arXiv:2610.12133v1 Announce Type: new  Abstract: Many key-value (KV) caches are compressed before anyone knows what will be asked of them: a document cached for retrieval, a prompt prefix shared across requests, the memory of a long conversation. The prevailing approach scores KV entries by rehearsal: the model rereads the context and keeps the entries it attends to, assuming that the more completely a cache rehearses its context, the better it remembers it. We show that under tight budgets this assumption backfires: rehearse everything, remember nothing. At a 3% keep ratio, rereading the whole context keeps 31.5 of 96.5 points on RULER, and on LongBench's natural-text tasks it falls below methods that rehearse nothing at all. The cause is that a cache keeps what it rehearses: rereading spreads the budget across the whole context, so the answer's own entries survive at little more than chance. Like a student before an exam, a cache remembers more by testing itself than by rereading. Tw
    
[^262]: 基于距离草图的高效二次熵计算

    Efficient quadratic entropy with distance sketches

    [https://arxiv.org/abs/2610.11976](https://arxiv.org/abs/2610.11976)

    本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。

    

    我们详细介绍了用于近似任意分布 $p$ 和负型常见距离 $d$ 下二次熵 $p^T d p$ 的可扩展方法。我们聚焦于欧几里得距离和球面测地距离两种情形，二者均在简单的框架内利用随机特征嵌入和投影来显著改善计算复杂度。通过摊销单次大型矩阵乘法以及控制变量技术，在 $d$ 保持不变而 $p$ 变化的场景下，该方法进一步实现了低内存占用和短运行时间的大规模计算。我们通过与直接配对采样方法的对比，以及在 Open Graph Benchmark 数据集上的文献计量学/科学计量学示例验证了该方法，仅凭引用和文本特征即可揭示跨学科影响力特别窄或特别宽的论文、领域和机构。

    arXiv:2610.11976v1 Announce Type: cross  Abstract: We detail scalable methods for approximating the quadratic entropy $p^T d p$ for arbitrary distributions $p$ and common distances $d$ of negative type. We focus on the Euclidean and spherical geodesic cases, which both use random feature embeddings and projections to dramatically improve computational complexity within a simple framework. Amortization of a single large matrix multiplication and control variates further enable computation at large scale with low memory and runtime in situations where $d$ is held constant while $p$ varies. We demonstrate this with a comparison against direct pair sampling and bibliometric/scientometric examples on Open Graph Benchmark datasets, revealing papers, fields, and institutions with both particularly narrow and broad interdisciplinary reach from their citations and text features alone.
    
[^263]: 超越QAOA：人工智能与量子计算在自适应组合优化中的综述

    Beyond QAOA: A Review of AI and Quantum Computing for Adaptive Combinatorial Optimization

    [https://arxiv.org/abs/2610.11759](https://arxiv.org/abs/2610.11759)

    本综述提出“自适应量子优化”的概念，按被学习的决策类型系统梳理了AI赋能量子优化、量子赋能AI优化和AI-量子协同优化三种范式，并通过对119篇论文的分析发现AI辅助量子优化的证据显著强于量子辅助AI优化。

    

    近期的量子组合优化方法受到量子比特数量、电路保真度、采样成本以及约束编码难度的限制，而机器学习正日益被用于配置和控制量子优化工作流程。我们将这类工作流程称为“自适应的”：传统上预先固定的决策——从问题形式化和惩罚项设置到采样预算、后端选择，乃至是否调用量子处理器本身——都由学习到的策略来做出，这些策略能够响应问题实例、求解过程的进展或硬件状态。本综述考察了三种范式：AI赋能量子优化、量子赋能AI驱动的优化，以及AI-量子协同优化，并按照被学习的决策而非应用领域来组织文献。对119篇论文的结构化综述（其中67篇进行了详细编码分析）表明，AI辅助量子优化的证据明显强于相反方向。

    arXiv:2610.11759v1 Announce Type: cross  Abstract: Near-term quantum approaches to combinatorial optimization are limited by qubit counts, circuit fidelity, sampling cost, and the difficulty of encoding constraints, while machine learning is increasingly used to configure and control quantum optimization workflows. We call such workflows adaptive: decisions conventionally fixed in advance, from formulation and penalties to shot budgets, backends, and whether to invoke a quantum processor at all, are made by learned policies that respond to the instance, the progress of the solve, or the hardware. This review examines three paradigms, AI for quantum optimization, quantum for AI-driven optimization, and AI-quantum co-optimization, and organizes the literature by the decision being learned rather than by application. A structured review of 119 papers, 67 coded in detail, shows that the evidence is considerably stronger for AI-assisted quantum optimization than for the reverse direction: l
    
[^264]: 平滑稀疏混合专家模型的Top-k暴露边界

    Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2610.11575](https://arxiv.org/abs/2610.11575)

    提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。

    

    稀疏混合专家模型在保持每个token固定计算预算的同时，能够高效地扩展参数容量。然而，传统的训练范式强制执行静态的top-k专家选择，这将连续的路由分布转换成了刚性的阶跃函数。这一约束引入了一个脆弱的边界，使得高度竞争的专家仅因微小的分数波动就被任意地划分为完全监督区域和零反馈区域。为了解决这个问题，我们提出了弹性专家路由，它从以k为中心的局部离散分布中随机采样激活专家的数量。在多次训练迭代中，该机制将尖锐的阈值软化成渐进的概率分布。由于采样邻域保持对称，该方法在匹配确定性训练的期望计算成本的同时，保留了推理预算。大量实验表明……

    arXiv:2610.11575v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts models scale parameter capacity efficiently while maintaining a fixed compute budget per token. However, traditional training paradigms enforce a static choice of top-$k$ experts, which converts a continuous routing distribution into a rigid step function. This constraint introduces a brittle boundary where highly competitive experts are arbitrarily separated into full-supervision and zero-feedback zones based on minor score fluctuations. To address this issue, we propose Elastic Expert Routing, which stochastically samples the active expert budget from a localized discrete distribution centered at $k$. Over multiple training iterations, this mechanism softens the sharp threshold into a gradual probability distribution. Because the sampling neighborhood remains symmetric, this approach matches the expected computational cost of deterministic training, while preserving the inference budget. Extensive experiments 
    
[^265]: Sera：面向可靠且可解释电池健康预测的语义表示聚合

    Sera: Semantic Representation Aggregation for Reliable and Interpretable Battery Health Forecasting

    [https://arxiv.org/abs/2610.11567](https://arxiv.org/abs/2610.11567)

    提出语义表示聚合框架 Sera，通过融合基于规则的知识与大语言模型解读所构建的退化语义表示来补充时序建模，从而实现更可靠、更可解释的电池健康状态预测。

    

    电池健康状态预测对电池管理至关重要，但由于电池退化的非线性以及不同电池之间的异质性，该任务仍然充满挑战。现有的数据驱动方法主要依靠时序模型从数值型电池时间序列中学习，而更高层次的退化特征往往没有被显式地表示出来。然而，这些特征能够提供退化方面的指导，从而支持可靠的预测，并使退化的影响更具可解释性。在本文中，我们提出了 Sera，一个语义表示聚合框架，通过退化语义来补充时序建模。在电池领域专业知识的指导下，Sera 从时间序列中提取退化语义，并利用基于规则的知识和基于大语言模型（LLM）的解读构建两种互补的表示，这些表示随后被独立编码（摘要在此处被截断）。

    arXiv:2610.11567v1 Announce Type: cross  Abstract: Battery state of health (SoH) forecasting is important for battery management, but remains challenging due to nonlinear degradation and heterogeneity across batteries. Existing data-driven approaches primarily use temporal models to learn from numerical battery time series, and higher-level degradation characteristics are often not explicitly represented. These characteristics, however, can provide degradation guidance to support reliable forecasting and make the influence of degradation more interpretable. In this paper, we propose \textsc{Sera}, a \underline{se}mantic \underline{r}epresentation \underline{a}ggregation framework that complements temporal modelling with degradation semantics. Guided by battery domain expertise, \textsc{Sera} extracts degradation semantics from time series and constructs two complementary representations using rule-based knowledge and LLM-based interpretation. The representations are independently encod
    
[^266]: 何时干预？联邦强化学习中的状态感知稀疏操纵

    When to Intervene? State-Aware Sparse Manipulation in Federated Reinforcement Learning

    [https://arxiv.org/abs/2610.11523](https://arxiv.org/abs/2610.11523)

    该论文首次将“何时干预”确立为联邦强化学习拜占庭攻击的一个独立攻击维度，提出了利用本地策略不确定性选择稀疏干预状态并施加包络约束行为引导的V-BSA攻击方法，证明干预时机的选择会实质性影响攻击效果。

    

    联邦强化学习（FRL）使分布式智能体能够协同训练决策策略，但其去中心化的训练过程也使全局策略学习暴露于拜占庭操纵的风险之中。现有的投毒攻击主要关注如何构造恶意更新，而轨迹层面的干预时机在很大程度上仍是隐式的。然而，在序贯决策中，干预施加在何处会改变后续轨迹和学习信号。通过受控实验，我们发现即使恶意更新的构造保持不变，改变所选择的轨迹状态也会实质性改变攻击效果。因此，我们将“何时”确定为一个独立的攻击维度，并提出了可行性约束行为引导攻击（V-BSA），该方法利用本地策略不确定性来选择稀疏干预状态，并施加包络约束的行为引导。（原文摘要在此处截断）

    arXiv:2610.11523v1 Announce Type: new  Abstract: Federated reinforcement learning (FRL) enables distributed agents to collaboratively train decision-making policies, but its decentralized training process also exposes global policy learning to Byzantine manipulation. Existing poisoning attacks primarily focus on how to construct malicious updates, while trajectory-level intervention timing remains largely implicit. In sequential decision making, however, where an intervention is applied can alter subsequent trajectories and learning signals. Through controlled experiments, we find that changing the selected trajectory states materially alters attack efficacy even when the malicious-update construction is fixed. We therefore identify when as a distinct attack dimension and introduce the Viability-constrained Behavioral Steering Attack (V-BSA), which uses local policy uncertainty to select sparse intervention states and applies envelope-constrained behavioral steering. Across discrete-ac
    
[^267]: 利用势函数估计自回归语言模型下的广义期望

    Estimating great expectations under autoregressive language models with potentials

    [https://arxiv.org/abs/2610.11399](https://arxiv.org/abs/2610.11399)

    本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。

    

    语言模型的许多应用所依赖的并非单个样本，而是模型下某个检验泛函的期望。可靠地估计这类期望在计算上可能非常昂贵。在本文中，我们展示了如何利用采样过程中作为副产品自然获得的下一词元条件概率来使估计更加高效。我们通过势函数来实现这一点：势函数是定义在前缀上的实值函数，能够对检验泛函进行加性分解。我们构造了一个估计器，其方差取决于所选择的势函数，并推导出势函数能够降低该方差的充分条件。随后，我们为若干估计对象和应用开发了实用的势函数，并在计算成本相当的情况下，在多个估计对象上展示了显著的方差缩减效果。

    arXiv:2610.11399v1 Announce Type: new  Abstract: Many applications of language models hinge not on individual samples but on the expectation of a test functional under the model. Estimating such expectations reliably can be computationally expensive. In this paper, we show how to make estimation more efficient by exploiting the next-token conditional probabilities which are available as a by-product of sampling. We do so through potentials: real-valued functions on prefixes that decompose the test functional additively. We construct an estimator whose variance depends on the chosen potential, and derive conditions under which a potential reduces this variance. We then develop practical potentials for several estimands and applications, and demonstrate substantial variance reductions across several estimands at comparable computational cost.
    
[^268]: 收缩海森矩阵：面向多模态扩散Transformer的秩4 W4A4量化

    Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers

    [https://arxiv.org/abs/2610.11315](https://arxiv.org/abs/2610.11315)

    该论文提出一个统一框架，将低秩辅助的W4A4训练后量化建模为耦合校准问题，通过“收缩海森矩阵”对已被低秩分量捕获的残差误差进行折扣，并结合激活噪声代理抑制激活量化误差，从而仅用秩4即可实现多模态扩散Transformer的高精度4比特量化。

    

    在扩散Transformer中，低秩分支可以通过将每个权重分解为低比特残差和一个高精度的低秩分量，来缓解4比特权重-激活（W4A4）训练后量化（PTQ）带来的损失。然而，现有的低秩PTQ方法要么将低秩补偿和残差量化分开优化，通常需要更高的秩；要么依赖二阶权重更新，而没有显式建模激活量化误差，这种误差在4比特量化下尤为显著。为了解决这些局限，我们提出了一个统一框架，将低秩辅助的W4A4 PTQ建模为一个耦合校准问题，并从联合目标出发推导出基于优化的求解器。通过消除输出侧的低秩因子，得到一个“收缩海森矩阵”，它可以对已被低秩分量捕获的残差误差进行折扣，同时引入激活噪声代理来抑制（激活量化噪声的影响）……

    arXiv:2610.11315v1 Announce Type: new  Abstract: In diffusion transformers, low-rank branches can mitigate 4-bit weight--activation (W4A4) post-training quantization (PTQ) loss by decomposing each weight into a low-bit residual and a high-precision low-rank component. Existing low-rank PTQ approaches, however, either optimize low-rank compensation and residual quantization separately, often requiring higher ranks, or rely on second-order weight updates without explicitly modeling activation quantization error, which becomes particularly pronounced under 4-bit quantization. To address these limitations, we present \method{}, a unified framework modeling low-rank-assisted W4A4 PTQ as a coupled calibration problem and deriving optimization-based solvers from the joint objective. Eliminating the output-side low-rank factor yields a \emph{deflated Hessian} that discounts residual errors already captured by the low-rank component, while an activation-noise surrogate is incorporated to suppre
    
[^269]: 剖析视觉Transformer中的表征结构：一项严谨的架构研究

    Dissecting Representation Structure in Vision Transformers: A Rigorous Architectural Study

    [https://arxiv.org/abs/2610.11205](https://arxiv.org/abs/2610.11205)

    该论文首次严谨分析了视觉Transformer跨架构尺度的特征信息，发现初始化时的特征坍缩问题并提出缓解方案，同时证明熵和最小特征值可作为泛化预测的可靠指标，用以指导高效的ViT设计。

    

    表征结构对于理解视觉Transformer（ViT）架构及其泛化行为至关重要。然而，先前的研究既没有隔离和分析模块级别的特征，也没有研究这些特征之间的交互如何促进性能估计。在这项工作中，我们首次对跨多种架构尺度的特征信息进行了严谨分析，实证揭示了ViT表征与泛化行为之间的关系，并利用这些见解来指导高效的ViT设计。我们的贡献有五个方面：在多样的架构尺度上，1）我们识别了初始化时的特征坍缩现象，该现象导致特征冗余，并提出了一种缩减方案来缓解这一问题。2）我们使用熵和最小特征值来量化特征信息，证明这些指标可以作为泛化预测的可靠指标。3）我们表明特征在token...（原文摘要在此处截断）

    arXiv:2610.11205v1 Announce Type: cross  Abstract: Representation structure is crucial for understanding Vision Transformer (ViT) architectures and their generalization behavior. However, prior studies neither isolate nor analyze module-level features nor investigate how their interactions contribute to performance estimation. In this work, we conduct the first rigorous analysis of feature information across diverse architectural scales, empirically uncover the relationship between ViT representation and generalization behavior, and leverage these insights to guide efficient ViT design. Our contributions are fivefold: Across diverse architectural scales, 1) We identify feature collapse at initialization, which leads to redundancy, and propose a reduction scheme to mitigate this issue. 2) We quantify feature information using entropy and the minimum eigenvalue, demonstrating that these metrics serve as reliable indicators for generalization prediction. 3) We show that feature in the tok
    
[^270]: 面向信贷风险建模的排序先验对齐：外部先验何时起作用？

    Ranking Prior Alignment for Credit Risk Modeling: When Do External Priors Matter?

    [https://arxiv.org/abs/2610.11146](https://arxiv.org/abs/2610.11146)

    提出了一种模型无关的“排序先验对齐”框架，通过温度缩放的KL散度损失将来自领域专家、教师模型或大语言模型的外部排序先验统一蒸馏进神经或树类评分模型，以缓解标注数据稀缺的冷启动信贷评分难题。

    

    冷启动信贷评分——即在标注数据稀缺、特征薄弱或模型容量极小的情况下部署模型——是金融机器学习中反复出现的难题。当一个新贷款产品上线时，标注的违约数据稀缺、特征流水线不成熟，模型必须以最小容量部署以避免过拟合。标准防御手段都在同样有限的同一份数据上运行；我们真正需要的是一种扎根于领域知识的外部正则化来源。我们提出了排序先验对齐，这是一个模型无关的框架，通过温度缩放的KL散度损失，将外部排序先验（来自领域专家、教师模型或大语言模型）蒸馏到任何评分模型中。该框架通过一个统一公式将神经网络（MIL注意力）架构和树模型（XGBoost自定义目标函数）架构统一起来：L = L_task + gamma(t) * KL(P_agent || P_model)，其中 gamma(t) 遵循指数衰减调度。该方法不需要外部……（原文摘要在此处截断）

    arXiv:2610.11146v1 Announce Type: new  Abstract: Cold-start credit scoring -- deploying models with scarce labeled data, weak features, or minimal capacity -- is a recurring problem in financial machine learning. When a new lending product launches, labeled default data is scarce, feature pipelines are immature, and models must be deployed with minimal capacity to avoid overfitting. Standard defenses operate on the same limited data; what is needed is a source of external regularization grounded in domain knowledge.   We propose Ranking Prior Alignment, a model-agnostic framework that distills external ranking priors (from domain experts, teacher models, or LLMs) into any scoring model via a temperature-scaled KL divergence loss. The framework unifies neural (MIL attention) and tree-based (XGBoost custom objective) architectures through a single formulation: L = L_task + gamma(t) * KL(P_agent || P_model), where gamma(t) follows an exponential decay schedule. The method requires no exte
    
[^271]: 动力学即代码：基于动力系统的模型压缩

    Dynamics as Code: On Model Compression via Dynamic System

    [https://arxiv.org/abs/2610.11115](https://arxiv.org/abs/2610.11115)

    该论文提出“动力学即代码”的新型模型压缩范式，证明了在丢番图条件下，无理缠绕的有限轨迹可构成权重空间上的ε-网，从而以可预测的方式将状态分辨率、解压误差与压缩比联系起来。

    

    预训练神经网络规模的不断攀升，使得模型压缩成为在严苛内存与计算约束下部署的先决条件。以无理缠绕为例，先前工作引入了一种动力系统（DS）范式，将压缩重新定义为紧凑的权重表示：高维参数由动力系统产生的轨迹的索引进行编码，解压时再从该轨迹中恢复原始向量。这一机制与剪枝、量化、知识蒸馏以及低秩分解有着本质区别。沿着这一方向，我们证明了在丢番图条件下，无理缠绕中由 \(M = O(\epsilon^{-(d+\nu)})\) 个状态构成的有限轨迹可以在 \(d\) 维权重空间上构成一个 \(\epsilon\)-网，从而以可预测的方式将状态分辨率、解压误差和压缩比联系了起来。此外，我们进一步……

    arXiv:2610.11115v1 Announce Type: new  Abstract: The escalating size of pretrained neural networks has rendered model compression a prerequisite for deployment under stringent memory and compute constraints. With the irrational winding as an example, earlier work introduced a dynamic system (DS) paradigm that reconceptualizes compression as compact weight representation: high-dimensional parameters are encoded by the index of a trajectory produced by a dynamic system, from which the vector is recovered during decompression. This mechanism is fundamentally distinct from pruning, quantization, knowledge distillation, and low-rank decomposition. Along this direction, we prove that under a Diophantine condition, a finite trajectory of \(M = O(\epsilon^{-(d+\nu)})\) states in the irrational winding constitutes an \(\epsilon\)-net over the \(d\)-dimensional weight space, thereby linking state resolution, decompression error, and compression ratio in a predictable manner. Furthermore, we prop
    
[^272]: 测量与缓解RLVR中的解模式坍缩

    Measuring and Mitigating Solution Mode Collapse in RLVR

    [https://arxiv.org/abs/2610.11064](https://arxiv.org/abs/2610.11064)

    本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。

    

    语言模型通常可以用多种方式回答同一个问题，但基于可验证奖励的强化学习（RLVR）对于模型产生哪个正确答案并不加以区分。无论一个解是某个熟悉答案的第千次重复，还是模型从未产生过的新解，它都会获得相同的奖励。然而，在训练过程中让模型保留多个正确解具有潜在价值。例如，多种模式可以为用户提供选择，并提供有助于提升模型整体性能的问题解决策略。在此，我们介绍了ModeBench，这是一个多解任务基准，其中验证器会同时返回正确性和所发现的模式。我们随后利用ModeBench来衡量RLVR后训练过程中解多样性的变化。我们发现，即使准确率保持不变甚至有所提升，RLVR后训练仍会将概率集中到更少的正确模式上，而且前沿模型几乎……

    arXiv:2610.11064v1 Announce Type: cross  Abstract: A language model (LM) can usually answer the same question in more than one way, but reinforcement learning with verifiable rewards (RLVR) is indifferent to which correct answer a model produces. A solution will earn the same reward whether it is the thousandth copy of a familiar answer or one the model has never produced before. Yet, there is potential value in having the model retain multiple correct solutions as it is trained. For instance, multiple modes may give users a choice and provide problem-solving strategies that improve overall model performance. Here, we introduce ModeBench, a benchmark of multi-solution tasks in which the verifier returns both correctness and mode discovered. We then use ModeBench to measure how solution diversity changes under RLVR post-training. We find that RLVR post-training concentrates probability onto fewer correct modes even as accuracy holds or improves, and moreover, that frontier models are al
    
[^273]: 使用CoMoE在消费级GPU上实现MoE推理的普及化

    Democratizing MoE inference on commodity GPUs with CoMoE

    [https://arxiv.org/abs/2610.09424](https://arxiv.org/abs/2610.09424)

    CoMoE通过新颖的以主机为中心的路由机制，将主机转变为主动路由枢纽，解决了消费级GPU互连带宽受限的通信瓶颈，使MoE模型推理能够在低成本的商品级GPU上高效部署。

    

    部署混合专家模型严重依赖专家并行，这会产生密集的GPU间通信。因此，最先进的推理系统需要数据中心GPU中配备高带宽的点对点互连（如NVLink）来处理海量的令牌路由，这使得部署成本高得令人望而却步。消费级GPU以显著更低的价格提供了相当的算力，有望让个人用户也能负担得起MoE推理，并实现保护隐私的本地部署。然而，消费级GPU带宽受限（仅有较弱的PCIe总线带宽）且互连需经主机中转（不支持点对点通信），从而引入了严重的通信瓶颈。我们提出了CoMoE，一个通信高效的MoE推理系统，通过新颖的以主机为中心的路由方式解决了这一不匹配问题。我们的关键洞察是，独特的通信拓扑结构提供了将主机提升为主动路由枢纽的机会，这可以从根本上

    arXiv:2610.09424v1 Announce Type: cross  Abstract: Deploying Mixture-of-Experts (MoE) models relies heavily on Expert Parallelism, which generates intense inter-GPU communication. Consequently, state-of-the-art inference systems require high-bandwidth, P2P interconnects (e.g., NVLink) in datacenter GPUs to handle massive token routing, making deployment prohibitively expensive. Consumer GPUs offer comparable compute power at significantly lower cost, promising to democratize MoE inference for individuals and enable privacy-preserving local deployments. However, their bandwidth-limited (only weak PCIe bus bandwidth) and host-mediated interconnects (no P2P support) introduce severe communication bottlenecks.   We present CoMoE, a communication-efficient MoE inference system that resolves this mismatch through novel host-centric routing. Our key insight is that the unique communication topology provides the opportunity to elevate the host to an active routing hub, which can fundamentally 
    
[^274]: Workhorse：从人类数据学习鲁棒的人形机器人全身移动操作

    Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data

    [https://arxiv.org/abs/2610.09117](https://arxiv.org/abs/2610.09117)

    提出Workhorse框架，通过在同一套无需重定向的真人演示数据上分别训练视觉规划器和强化学习全身跟踪器，并以互相模仿对方部署误差的方式增强训练数据，使Unitree G1人形机器人能从第一人称视觉实现鲁棒的全身移动操作并具备抗干扰恢复能力。

    

    人形机器人在仅依靠第一人称视角RGB图像和本体感觉来规划接触丰富的全身操作方面仍然面临困难。Workhorse从无需机器人参与的真人演示数据中学习此类操作。视觉规划器预测五个连杆目标：躯干、双手腕和双脚的姿态。强化学习全身跟踪器控制机器人跟随这些目标。两个策略在同一录制的真人姿态数据上分别训练，无需重定向。我们通过数据增强模仿对方策略在部署时可能产生的误差来增强每个策略的训练数据。在真实的Unitree G1机器人上，Workhorse能够用手和踢腿动作分拣箱子、接住抛来的箱子、推倒并攀爬行李箱。在箱子分拣过程中，我们展示了在人推动机器人或拿走箱子后系统的恢复能力。在演示房间的仿真环境中，系统在77%的回合中成功完成箱子分拣任务，在40 N·s推力干扰下成功率为64%。

    arXiv:2610.09117v1 Announce Type: cross  Abstract: Humanoid robots still struggle to plan contact-rich whole-body manipulation from egocentric RGB and proprioception. Workhorse learns such manipulation from robot-free human demonstrations. A visual planner predicts five-link targets: the poses of the torso, both wrists, and both feet. A reinforcement-learning whole-body tracker follows them on the robot. Both policies train separately on the same recorded human poses, without retargeting. We augment the training data of each policy to imitate the errors that the other makes at deployment. On a real Unitree G1, Workhorse sorts boxes with its hands and a kick, catches a thrown box, and topples and climbs a suitcase. During box sorting, we show recoveries after a person pushes the robot or takes the box away. In a simulated copy of the demonstration room, the system completes box sorting in 77% of episodes, and in 64% under 40 N.s pushes. With both policies retrained from the same demonst
    
[^275]: 趁它们沉睡时工作：利用评估延迟实现全贝叶斯优化

    Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization

    [https://arxiv.org/abs/2610.08969](https://arxiv.org/abs/2610.08969)

    该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。

    

    黑盒优化问题在科学与工程领域无处不在，通常涉及代价高昂的目标函数。这种目标评估延迟在优化过程中带来两个后果：（i）目标函数评估主导了整个执行时间；（ii）样本高效的算法对于加速开发、避免资源浪费至关重要。贝叶斯优化（BO）方法是规划器为推荐下一个尝试点的事实上的标准选择。标准的BO采用点估计来拟合代理模型的超参数；而完全贝叶斯方法则通过模型平均来考虑超参数的不确定性，从而获得更好的不确定性估计——这在BO中普遍存在的低数据场景下非常有用。然而，该方法往往计算代价过高，因此很少被使用。在这项工作中，我们提出ELF-BO，一种利用目标函数评估延迟来提前启动全贝叶斯代理模型计算的算法。

    arXiv:2610.08969v1 Announce Type: new  Abstract: Black-box optimization problems are ubiquitous across science and engineering, often dealing with expensive objective functions. This objective latency has two consequences during optimization: (i) the objective evaluation dominates execution time, and (ii) sample-efficient algorithms are crucial to accelerate development and avoid wasting resources. Bayesian optimization (BO) methods are the \textit{de facto} choice of planners for suggesting the next point to try. Standard BO fits the surrogate model's hyperparameters with a point estimate. Alternatively, a fully Bayesian approach uses model averaging to account for uncertainty over the hyperparameters, leading to better uncertainty estimates---useful in the low-data regime that is pervasive in BO. However, it is often prohibitively expensive and thus rarely used. In this work, we propose ELF-BO, an algorithm that uses the objective evaluation latency to headstart the computation of th
    
[^276]: TRACE：面向MoE语言模型FP4强化学习的Rollout引导量化感知训练

    TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models

    [https://arxiv.org/abs/2610.07767](https://arxiv.org/abs/2610.07767)

    TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。

    

    对大语言模型（LLM）进行后训练的强化学习（RL）在rollout生成过程中会产生大量的计算和内存开销，这促使人们采用低精度rollout来实现高效的RL训练。然而，现有的FP4 RL方法存在一个关键局限：它们主要在训练路径和rollout路径上分别独立地优化量化精度，而不是直接减少两条量化执行路径之间的差异。在本工作中，我们提出了TRACE（Train-Rollout Quantization Alignment via Compact GuidancE，通过紧凑引导实现训练-Rollout量化对齐），这是一个面向混合专家（MoE）语言模型RL训练的FP4量化框架，解决了现有FP4 RL方法的局限性。TRACE引入了rollout引导的量化感知训练，利用rollout侧的量化结果来指导训练侧的FP4舍入决策，从而直接减少训练与rollout之间的差异。此外，TRACE采用了高效的量化……

    arXiv:2610.07767v1 Announce Type: cross  Abstract: Reinforcement learning (RL) for post-training large language models (LLMs) incurs substantial computation and memory overhead during rollout generation, which motivates low-precision rollout for efficient RL training. However, existing FP4 RL methods suffer from a key limitation: they primarily optimize quantization accuracy on the training and rollout paths independently rather than directly reducing the discrepancy between the two quantized execution paths. In this work, we propose TRACE (Train-Rollout Quantization Alignment via Compact GuidancE), an FP4 quantization framework for RL training of Mixture-of-Experts (MoE) language models that addresses the limitation of existing FP4 RL methods. TRACE incorporates rollout-guided quantization-aware training that uses rollout-side quantization outcomes to guide training-side FP4 rounding decisions, directly reducing train-rollout discrepancy. Moreover, TRACE adopts an efficient quantizati
    
[^277]: 超越 $T^{2/3}$ 的序贯校准问题的显式渐近界

    Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$

    [https://arxiv.org/abs/2610.07623](https://arxiv.org/abs/2610.07623)

    该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。

    

    当预测概率与经验结果频率相匹配时，概率预测被称为校准的：在被赋予概率 $p$ 的事件中，我们希望正结果的占比接近 $p$。我们研究二元结果的序贯预测问题。Foster 和 Vohra 建立的关于期望累积 $\ell_1$ 校准误差的经典 $O(T^{2/3})$ 界保持了二十多年，直到 Dagan 等人将指数 $2/3$ 降低了一个未具体指明的常数。我们为“符号保持-复用”博弈建立了一种新的两阶段递归标记策略，对于所有空间和时间的选择都能得到 $O(n^{\alpha}t^\beta)$ 的界。随后，我们通过修改 Dagan 等人的等价性，使其仅使用 $O(\log T)$ 个“符号保持-复用”博弈实例，从而锐化了从符号保持上界到校准的归约。这使我们能够建立一个显式的界 $O(T^{0.662942288})$。

    arXiv:2610.07623v1 Announce Type: cross  Abstract: Probability forecasts are calibrated when predicted probabilities match empirical outcome frequencies: among events assigned a probability $p$, we'd hope that the fraction of positive outcomes is close to $p$. We study the problem of sequential forecasting of binary outcomes. The classical $O(T^{2/3})$ bound on expected cumulative $\ell_1$-calibration error established by Foster and Vohra stood for over two decades until Dagan et al. reduced the exponent $2/3$ by an unspecified constant.   We establish a new two-phase recursive labeling strategy for the sign-preservation-with-reuse game that yields the bound $O(n^{\alpha}t^\beta)$ for all choices of space and time. We then sharpen the reduction from upper bounds on sign preservation to calibration by modifying the equivalence of Dagan et al. to use only $O(\log T)$ instances of the sign-preservation-with-reuse game. This lets us establish an explicit bound of $O(T^{0.662942288})$, the 
    
[^278]: BiGym 2.0：人形机器人家庭操作中学习型与智能体开发策略的基准测试

    BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation

    [https://arxiv.org/abs/2610.07594](https://arxiv.org/abs/2610.07594)

    BiGym 2.0是一个针对宇树G1人形机器人20个家庭操作任务的全身控制基准测试平台，实验表明视觉-语言-动作微调方法总体表现最佳，而编码智能体开发的程序优于所有演示驱动的强化学习基线。

    

    人形机器人家庭操作需要手臂在身体保持平衡、迈步和改变姿势的同时进行动作。我们提出了BiGym 2.0，这是BiGym针对宇树G1机器人的适配版本，涵盖20个家庭任务，使用统一的全身控制器进行演示和评估。该测试套件为每个任务提供60个原生人类虚拟现实演示，包含同步的多相机视角和全身执行记录。我们对视觉-语言-动作微调、模仿学习、演示驱动的强化学习，以及给定在线强化学习交互预算的冷启动编码智能体进行了基准测试。在为每种方法提供相同的机载视角、本体感觉和全身控制器的条件下，视觉-语言-动作微调在九项任务的平均值上表现最佳，而智能体开发的程序在该平均值上超越了所有演示驱动的强化学习基线，并在双手抓取任务上处于领先地位。跨工作空间堆叠任务仍未被解决，π₀.

    arXiv:2610.07594v1 Announce Type: cross  Abstract: Humanoid household manipulation requires the arms to act while the body balances, steps and changes posture. We present BiGym 2.0, an adaptation of BiGym for the Unitree G1 across 20 household tasks using a unified whole-body controller for demonstration and evaluation. The suite provides 60 native human virtual-reality demonstrations per task with synchronised multi-camera views and full-body execution records. We benchmark vision-language-action fine-tuning, imitation learning, demo-driven reinforcement learning, and cold-start coding agents given the interaction budget of online reinforcement learning. With the same onboard views, proprioception and whole-body controller for every method, vision-language-action fine-tuning has the highest nine-task mean, and agent-developed programs outperform every demo-driven reinforcement learning baseline on this mean and lead on bimanual reaching. Cross-workspace stacking remains open, $\pi_{0.
    
[^279]: TRIAGE：面向原生NVFP4强化学习的方向感知失配稳定化方法

    TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning

    [https://arxiv.org/abs/2610.07043](https://arxiv.org/abs/2610.07043)

    提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。

    

    低精度执行可以大幅加速大语言模型的强化学习（RL），但学习器与采样器执行之间的差异（失配）可能会破坏策略优化的稳定性。在本文中，我们刻画了失配与策略梯度方向之间的相互作用，区分了局部放大型与收缩型的更新贡献，而仅凭失配幅度无法识别这些差异。在原生NVFP4运行中，我们观察到两个放大区域之间的早期不平衡，偏向于负优势、负差距的更新。在失配全局扩散之前，这些尾部分词会集中在一小部分响应片段中。基于这些发现，我们提出了TRIAGE，一种方向感知的稳定化方法，它利用片段级诊断来选择性地重新平衡策略梯度更新，并对残余的严重失配应用有界修复。TRIAGE在修改优化目标的同时……

    arXiv:2610.07043v1 Announce Type: cross  Abstract: Low-precision execution can substantially accelerate reinforcement learning (RL) for large language models, but discrepancies between learner and sampler execution can destabilize policy optimization. In this paper, we characterize the interaction between mismatch and the policy-gradient direction, distinguishing locally amplifying from contracting update contributions that mismatch magnitude alone cannot identify. In native NVFP4 runs, we observe an early imbalance between the two amplifying regions, favoring negative-advantage, negative-gap updates. Their tail tokens become concentrated in a small fraction of response segments before mismatch spreads globally. Motivated by these findings, we introduce TRIAGE, a direction-aware stabilization method that uses segment-level diagnosis to selectively rebalance policy-gradient updates and applies bounded repair to residual severe mismatch. TRIAGE modifies the optimization objective while r
    
[^280]: 超越分布内性能保持：通过面向脆弱性的调优恢复量化视觉-语言-动作模型的泛化能力

    Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning

    [https://arxiv.org/abs/2610.05745](https://arxiv.org/abs/2610.05745)

    该研究揭示量化VLA模型在环境扰动下虽保持分布内性能却变得脆弱，发现动作差异集中于少数脆弱状态且教师指导在高差异状态提升泛化、在低差异状态反而损害泛化，据此提出面向脆弱状态的选择性调优方法以恢复量化模型的泛化能力。

    

    后训练量化已被证明能在标准评估条件下保持VLA（视觉-语言-动作模型）的性能，但其在多大程度上保留了全精度模型的鲁棒性和泛化能力仍缺乏深入研究。在本研究中，我们系统研究了量化后VLA策略在环境扰动下的鲁棒性与泛化能力。实证结果表明，量化策略尽管保持了相当的分布内性能，但在面对细微的环境变化时可能变得脆弱。我们进一步观察到，动作差异集中于少量回放状态，而教师指导的效果因差异程度而异：在高差异状态下能提升泛化能力，但在低差异状态下反而可能损害泛化能力。这些发现表明，有效的量化后恢复需要对脆弱状态进行选择性干预，而非对学生模型进行全局蒸馏。

    arXiv:2610.05745v2 Announce Type: replace  Abstract: Post-training quantization has been shown to preserve VLA performance under standard evaluation conditions, but whether it preserves the full-precision model's robustness and generalization remains underexplored. In this study, we systematically study the robustness and generalization of post-quantized VLA policies under environmental disturbances. Empirical results show that quantized policies can become fragile to subtle environmental variations despite retaining comparable in-distribution performance. We further observe that action discrepancies are concentrated in a small subset of rollout states, while teacher guidance has opposite effects depending on discrepancy: it improves generalization at high-discrepancy states but can degrade it at low-discrepancy states. These findings reveal that effective post-quantization recovery requires selectively intervening on vulnerable states rather than globally distilling the student. We th
    
[^281]: 格林路由神经算子：物理决定网络读取的位置

    Green-Routed Neural Operators: Physics Determines Where the Network Reads

    [https://arxiv.org/abs/2610.05337](https://arxiv.org/abs/2610.05337)

    提出格林路由神经算子（GRNO），利用控制方程决定潜在特征的采样位置，在五个PDE系统中的四个上取得最低的自回归预测误差。

    

    我们发现许多偏微分方程（PDE）中输运场的物理角色与其在神经算子中的常见用法之间存在不匹配：PDE利用输运场来选择读取坐标，而神经算子通常仅将其视为输入值。我们提出格林路由神经算子（GRNO）来解决这一不匹配问题，该方法利用控制方程来确定潜在特征的采样位置。一个无参数的方程适配器评估诊断关系并构建偏移图，其数值即为读取坐标。一个多尺度编码器-解码器结合潜在特征的中心读取与路由读取，以学习完整的有限时间更新。在五个二维和三维PDE系统中，GRNO在40步自回归评估下于其中四个系统上取得了最低的平均最终相对L²误差，并在Keller-Segel系统上保持竞争力。固定权重的路由干预实验表明，模型对读取方向和空间对齐具有强烈的依赖性。

    arXiv:2610.05337v2 Announce Type: replace-cross  Abstract: We identify a mismatch between the physical role of transport fields in many PDEs and their usual role in neural operators: PDEs use them to select read coordinates, whereas neural operators typically treat them only as input values. We address this mismatch with the Green-Routed Neural Operator (GRNO), which uses the governing equation to determine where latent features are sampled. A parameter-free equation adapter evaluates the diagnostic relation and constructs a departure map whose values are the read coordinates. A multiscale encoder-decoder combines centered and routed reads of latent features to learn the complete finite-time update. Across five two- and three-dimensional PDE systems, GRNO achieves the lowest mean final relative $L^2$ error on four under 40-step autoregressive evaluation and remains competitive on Keller-Segel. Fixed-weight route interventions reveal strong dependence on direction and spatial alignment 
    
[^282]: 面向图像分类的混合量子-经典架构演化

    Evolving Hybrid Quantum-Classical Architectures for Image Classification

    [https://arxiv.org/abs/2610.03220](https://arxiv.org/abs/2610.03220)

    该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。

    

    混合量子-经典神经网络将参数化量子电路（PQC）与成熟的深度学习架构相结合，但其性能在很大程度上取决于量子电路架构的选择，而这一选择目前仍主要依赖人工完成。现有的大多数方法依赖于手工设计或固定的电路拟设，需要预先指定电路结构、门组合和量子比特连接方式，且无法保证这些设计适合特定任务。这一限制在图像分类任务中尤为突出，因为量子电路既要对经典网络提取的特征进行变换，又要保持足够紧凑以便于实际训练，而通用的、与任务无关的拟设很难同时满足这些要求。我们将EXAQC——一个用于自动化量子电路发现的演化框架——扩展到图像分类任务。EXAQC将参数化量子电路作为中间处理模块进行演化，同时保留经典……（摘要原文在此处截断）

    arXiv:2610.03220v1 Announce Type: cross  Abstract: Hybrid quantum classical neural networks integrate parameterized quantum circuits (PQCs) with established deep learning architectures, but their performance depends strongly on the choice of quantum circuit architecture, a choice that remains largely manual. Most existing approaches rely on hand-designed or fixed circuit ans\"atze, requiring circuit structure, gate composition, and qubit connectivity to be specified in advance with no guarantee that they suit the task. This limitation is especially acute in image classification, where quantum circuits must transform features extracted by classical networks while remaining compact enough for practical training, requirements that generic, task-agnostic ans\"atze are unlikely to satisfy simultaneously. We extend EXAQC, an evolutionary framework for automated quantum circuit discovery, to image classification. EXAQC evolves PQCs as intermediate processing modules while retaining classical 
    
[^283]: 哈密顿量局域性测试与认证无法达到海森堡极限

    Hamiltonian locality testing and certification do not achieve the Heisenberg limit

    [https://arxiv.org/abs/2610.03205](https://arxiv.org/abs/2610.03205)

    该论文证明了哈密顿量k-局域性测试和哈密顿量认证问题都需要Ω(1/ε²)的总演化时间，首次为哈密顿量学习与测试中的自然问题建立了排除1/ε海森堡极限标度的下界。

    

    我们在只能访问时间演化算符但无法访问其逆算符的条件下，为哈密顿量性质测试建立了下界。每个实验可以多次查询时间演化算符，哈密顿量之间的距离用归一化Frobenius范数来度量。在该模型中，我们证明了判断一个哈密顿量是k-局域的还是与所有k-局域哈密顿量都相距ε-远，需要Ω(1/ε²)的总演化时间，这与Kallaugher和Liang（TQC'25）的上界相匹配。我们还证明了判断一个未知哈密顿量是否等于某个目标哈密顿量或与其相距ε-远，需要Ω(1/ε²)的总演化时间，这与Sinha和Tong（2025）的上界相匹配。这些是哈密顿量学习与测试领域自然问题的首个下界，排除了1/ε的海森堡极限标度。作为第三个结果，我们证明了振幅估计（摘要在此处截断）

    arXiv:2610.03205v1 Announce Type: cross  Abstract: We establish lower bounds for Hamiltonian property testing with access to the time-evolution operator but not its inverse. Each experiment may query the time-evolution operator multiple times, and distances between Hamiltonians are measured in the normalized Frobenius norm.   In this model, we show that testing whether a Hamiltonian is $k$-local or $\varepsilon$-far from every $k$-local Hamiltonian requires $\Omega(1/\varepsilon^2)$ total evolution time, matching the upper bound of Kallaugher and Liang (TQC'25). We also prove that testing whether an unknown Hamiltonian equals a target Hamiltonian or is $\varepsilon$-far from it requires $\Omega(1/\varepsilon^2)$ total evolution time, matching the upper bound of Sinha and Tong (2025). These are the first lower bounds for natural problems in Hamiltonian learning and testing that rule out Heisenberg-limited scaling of $1/\varepsilon$.   As a third result, we show that amplitude estimation
    
[^284]: 想象未来，内化要旨：通过内化时空想象实现高效的VLA推理

    Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination

    [https://arxiv.org/abs/2610.02626](https://arxiv.org/abs/2610.02626)

    提出IG-VLA框架，通过在视觉表征空间中直接想象任务相关的未来场景演变，并将其内化为紧凑的场景要旨标记，使VLA模型在推理时无需昂贵的显式未来推演即可获得未来推理带来的动作预测增益，实现高效的机器人操作。

    

    视觉-语言-动作（VLA）模型日益融入中间推理以提升机器人操作能力，然而现有方法主要针对观测到的状态进行推理，并未显式地预测未来场景的演变。而若在每次推理步骤中都将推理扩展到显式的未来推演，则会引入大量计算开销。我们提出IG-VLA，一个使模型能够想象未来并内化其要旨的VLA推理框架。我们的潜在时空推理学会直接在视觉表征空间中想象与任务相关的未来场景演变，从而引导动作预测，而无需代价高昂的像素级视频生成。为进一步降低推理开销，我们引入了场景要旨记忆，它将推理所得的场景-行为关联内化为一个紧凑的场景要旨标记，在保留未来推理优势的同时，在推理阶段绕过显式的未来想象。

    arXiv:2610.02626v2 Announce Type: cross  Abstract: Vision-language-action (VLA) models increasingly incorporate intermediate reasoning to improve robotic manipulation, yet existing approaches primarily reason about observed states without explicitly anticipating future scene evolution. Extending such reasoning to explicit future rollouts at every inference step, however, introduces substantial computational overhead. We propose IG-VLA, a VLA reasoning framework that enables models to imagine the future and internalize the gist. Our Latent Spatiotemporal Reasoning learns to imagine task-relevant future scene evolution directly in visual representation space, guiding action prediction without costly pixel-level video generation. To further reduce inference overhead, we introduce Scene Gist Memory, which internalizes reasoning-derived scene-behavior associations into a compact Scene Gist Token, preserving the benefits of future reasoning while bypassing explicit future imagination at infe
    
[^285]: 基于选择的结构化推理：迈向高效的多模态搜索智能体

    Selection-Based Structured Reasoning: Toward Efficient Multimodal Search Agents

    [https://arxiv.org/abs/2610.01892](https://arxiv.org/abs/2610.01892)

    提出基于选择的结构化推理框架SSR，将开放式推理生成转化为从预定义推理候选中的似然选择，并借助共享上下文KV缓存实现并行评分，从而大幅提升小模型多模态搜索智能体的推理效率。

    

    多模态智能体通常在每次执行动作前生成自由形式的推理。对于小模型而言，有限的模型容量会导致冗长的推理，这不仅对动作生成几乎提供不了有用的指导，还会带来巨大的推理开销。为应对这一挑战，我们提出了基于选择的结构化推理框架（Selection-based Structured Reasoning，SSR），该框架将推理重新表述为选择，而非开放式生成。SSR将反复出现的高层推理表示为预先指定的、可复用的自然语言候选。在每一轮中，模型基于当前上下文下的似然从这些推理候选中进行选择，无需任何辅助任务头。使用预先指定的推理轨迹使得并行评分成为可能，即通过教师强制预填充（teacher-forced prefilling）并借助共享的上下文KV缓存，在候选内部和候选之间并发计算token似然。我们在七个多模态搜索基准上使用2B和……的模型对SSR进行了评估。

    arXiv:2610.01892v1 Announce Type: cross  Abstract: Multimodal agents commonly generate free-form reasoning before each action. For small models, limited model capacity can result in lengthy reasoning that provides little useful guidance for action generation while incurring substantial inference cost. To address this challenge, we introduce Selection-based Structured Reasoning (SSR), a framework that reformulates reasoning as selection instead of open-ended generation. SSR represents recurring high-level reasoning as pre-specified, reusable natural-language candidates. At each turn, the model selects from these reasoning candidates based on their likelihoods given the current context, without requiring an auxiliary task head. Using pre-specified reasoning traces enables parallel scoring, where teacher-forced prefilling computes token likelihoods concurrently within and across candidates using a shared context KV cache. We evaluate SSR on seven multimodal search benchmarks using 2B and 
    
[^286]: 探索更多，推理更优：面向扩散语言模型的逐步风险敏感GRPO

    Exploring More, Reasoning Better: Stepwise Risk-Sensitive GRPO for Diffusion Language Models

    [https://arxiv.org/abs/2610.00661](https://arxiv.org/abs/2610.00661)

    提出StepRS-GRPO方法，通过在去噪状态间动态调整群组优势变换的风险系数，使扩散语言模型在数学推理任务上同时提升pass@1准确率、pass@k覆盖率和答案多样性。

    

    扩散大语言模型通过去噪一个序列或连续的多个块来生成文本，能够并行地揭示多个 token。基于可验证奖励的强化学习（RLVR）会在这些决策中重复使用终端反馈，即使决策所依赖的上下文已经发生变化。我们提出逐步风险敏感的 GRPO（StepRS-GRPO），它在不同去噪状态之间改变群组优势变换的风险系数，同时保留底层训练器。对于二元奖励，我们证明了这种变换恰好是对中心化结果优势进行的、依赖于提示和状态的重新缩放。基于能力的校准给出了系数的合适量级，而端点与插值消融实验则指导调度方案的选择。在多个 dLLM 骨干模型和数学推理基准上，StepRS-GRPO 相比中心化 GRPO 提升了 pass@1 准确率和 pass@k 覆盖率，同时增加了答案多样性。

    arXiv:2610.00661v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) generate text by denoising a sequence or successive blocks, allowing several tokens to be revealed in parallel. Reinforcement learning with verifiable rewards (RLVR) reuses terminal feedback across these decisions, even as their conditioning context changes. We propose stepwise risk-sensitive GRPO (StepRS-GRPO), which varies the risk coefficient of the group-advantage transformation across denoising states while retaining the underlying trainer. For binary rewards, we show that this transformation is exactly a prompt- and state-dependent rescaling of centered outcome advantages. A capability-based calibration suggests a coefficient scale, while endpoint and interpolation ablations guide schedule selection. Across multiple dLLM backbones and mathematical reasoning benchmarks, StepRS-GRPO improves both pass@1 accuracy and pass@k coverage over centered GRPO, while increasing answer diversity. In our
    
[^287]: 对谁有用？样本价值只有相对于学习器才有意义

    Useful to Whom? Sample Value Is Defined Only Relative to the Learner

    [https://arxiv.org/abs/2610.00221](https://arxiv.org/abs/2610.00221)

    该论文通过冻结样本子集、只改变学习器的受控实验证明，核心集选择中“易样本优先”与“几何覆盖”准则之间的交叉边界并非由数据固定，而是随学习器容量（如ResNet-18宽度、输入网格分辨率）而移动，说明样本的价值只有相对于特定学习器才有意义。

    

    arXiv:2610.00221v1 通告类型：cross 摘要：一个模型需要什么样的数据才能学习？核心集选择（coreset selection）使这个问题变得具体：在预算约束下，保留对训练最有用的样本。“易样本优先”与“几何覆盖”两类准则在不同的预算区间内各有胜负，二者之间由一个交叉边界分隔。我们追问：这个边界是由数据本身固定，还是会随目标学习器而改变？受控实验冻结所选子集，仅改变训练所用的学习器。在低分辨率ImageNet-100上，将ResNet-18的宽度加倍会使交叉点从每类57个样本移动到85个样本：学习器改变了同一样本的相对价值。更大范围的扫描揭示了输入网格与模型容量之间的交互作用：在保留相同图像信息的前提下扩大输入网格会使边界左移，且这种偏移随宽度增大而减弱。步长控制实验在不改变输入网格的情况下再现并逆转了网格效应；仅移除最后一个下采样……（摘要原文在此处截断）

    arXiv:2610.00221v1 Announce Type: cross  Abstract: What kind of data does a model need in order to learn? Coreset selection makes this question concrete: under a budget, keep the samples most useful for training. Easy-first and geometric coverage criteria can win in different budget regimes, separated by a crossover boundary. We ask whether this boundary is fixed by the data or changes with the target learner. Controlled experiments freeze the selected subsets and manipulate only the training learner. On low-resolution ImageNet-100, doubling ResNet-18's width moves the crossover from 57 to 85 samples per class: the learner changes the relative value of the same samples. A wider sweep reveals an interaction between input grid and capacity. Enlarging the grid while retaining the same image information shifts the boundary left, and this shift weakens as width increases. Stride controls reproduce and reverse the grid effect without changing the input grid; removing only the last downsampli
    
[^288]: 马尔可夫数据下异步TD（时序差分）学习的精确统计速率

    Sharp Statistical Rates for Asynchronous TD Learning with Markovian Data

    [https://arxiv.org/abs/2609.38880](https://arxiv.org/abs/2609.38880)

    该论文为基于单条马尔可夫轨迹的表格型TD学习建立了精确的末次迭代样本复杂度界 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$，且该界对常数步长与递减步长调度均成立。

    

    我们研究了基于有限马尔可夫奖励过程单条轨迹的标准表格型时序差分（TD）学习的末次迭代（last iterate）。对于折扣因子 $\gamma$，记 $H=(1-\gamma)^{-1}$，并令 $\mu_{\min}$ 和 $t_{\operatorname{mix}}$ 分别表示最小平稳概率和全变差混合时间。我们证明，对于 $0<\varepsilon\leq1$，末次迭代TD以高概率达到至多 $\varepsilon$ 的sup范数误差，所需转移步数为 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$。该速率既适用于为目标精度所选的常数步长，也适用于与目标精度和终止时间无关的递减步长调度。后者在超过某个显式瞬态阈值之后的所有时刻上提供了一致的同时保证。统计项保留了同步TD中三次方（立方）的有效视界依赖关系，而附加的混合过渡项则是不可省略的。

    arXiv:2609.38880v1 Announce Type: new  Abstract: We study the last iterate of standard tabular temporal-difference (TD) learning from a single trajectory of a finite Markov reward process. For discount factor $\gamma$, write $H=(1-\gamma)^{-1}$, and let $\mu_{\min}$ and $t_{\operatorname{mix}}$ denote the minimum stationary probability and total-variation mixing time. We prove that last-iterate TD achieves sup-norm error at most $\varepsilon$ with high probability using$\widetilde O\left(   \frac{H^3}{\mu_{\min}\varepsilon^2}   +\frac{t_{\operatorname{mix}}}{\mu_{\min}} \right)$ transitions, for $0<\varepsilon\leq1$. This rate holds both for a constant step size selected for the target accuracy and for a decreasing schedule independent of the target accuracy and terminal time. The latter gives a simultaneous guarantee over all times beyond an explicit transient threshold. The statistical term retains the cubic effective-horizon dependence of synchronous TD, and the additive mixing tran
    
[^289]: OpenCollab：一个具有可编程协作与可控运行时的多智能体编程框架

    OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime

    [https://arxiv.org/abs/2609.38345](https://arxiv.org/abs/2609.38345)

    提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。

    

    多智能体编程系统旨在通过协作来解决复杂的软件工程任务。然而，现有的评估通常假设所配置的组织结构会被忠实地遵循，而实际情况并非如此。这种行为差距，再加上底层系统组件的差异，使得观察到的性能提升难以进行明确的归因。为此，我们提出了OpenCollab，这是一个多智能体编程框架，为可编程协作与可控运行时提供了统一的基础设施。具体而言，OpenCollab统一了组织设计，在共享运行时上强制实施实验控制，并通过细粒度的事件流来跟踪执行过程。在此基础上，我们定义了Adherence（遵循度）指标，用于量化所声明的组织结构是否真正得以实现。我们的实验表明，智能体在不同配置下的协作方式差异很大：改变任何一个单一维度都会使Adherence发生变化，从47.2%到高达97%以上。

    arXiv:2609.38345v1 Announce Type: cross  Abstract: Multi-agent coding systems are designed to tackle complex software engineering tasks through collaboration. However, existing evaluations typically assume configured organizations are followed faithfully, whereas reality differs. This behavioral gap, combined with differences in underlying system components, prevents clear attribution of observed gains. To this end, we introduce OpenCollab, a multi-agent coding framework that provides a unified infrastructure for programmable collaboration and controllable runtime. Specifically, OpenCollab unifies organization design, enforces experimental control on a shared runtime, and tracks execution through fine-grained event streams. On this basis, we define Adherence to quantify whether the declared organization is actually realized. Our experiments reveal that agents collaborate very differently across configurations: changing any single dimension shifts Adherence, from 47.2% to as high as 97.
    
[^290]: 鲁棒LassoNet：通过鲁棒损失函数增强神经网络中的特征选择

    Robust LassoNet: Enhancing Feature Selection in Neural Networks via Robust Loss Functions

    [https://arxiv.org/abs/2609.38263](https://arxiv.org/abs/2609.38263)

    本文提出Robust LassoNet，通过在LassoNet中引入Huber、Cauchy、Tukey双平方等鲁棒损失函数，在保留原始优化框架的同时，显著提升了神经网络在数据污染情况下的特征选择准确性与预测性能。

    

    神经网络中的特征选择仍然是一个具有挑战性的问题，尤其是在存在噪声或被污染数据的情况下。LassoNet是近期提出的一种应对该问题的方法，它将神经网络与分层稀疏性约束相结合，能够同时进行预测和变量选择。然而，其标准形式依赖于均方误差（MSE）损失，而MSE损失对异常值高度敏感是众所周知的。在本文中，我们提出了Robust LassoNet，这是LassoNet的一种扩展，它引入了鲁棒损失函数，如Huber损失、Cauchy损失、Tukey双平方损失和非负Garrote损失，以减轻极端观测值的影响。所提出的方法在保留原始优化框架的同时，提高了数据被污染情况下的稳定性。通过在合成数据集和真实数据集上的实验，我们证明了鲁棒LassoNet在预测性能和特征选择准确性方面均有显著提升。

    arXiv:2609.38263v1 Announce Type: new  Abstract: Feature selection in neural networks remains a challenging problem, particularly in the presence of noisy or contaminated data. LassoNet is a recent approach that addresses this issue by combining neural networks with hierarchical sparsity constraints, enabling simultaneous prediction and variable selection. However, its standard formulation relies on the mean squared error (MSE) loss, which is known to be highly sensitive to outliers. In this paper, we present Robust LassoNet, an extension of LassoNet that incorporates robust loss functions, such as Huber, Cauchy, Tukey's bisquare, and Nonnegative Garrote, to mitigate the effect of extreme observations. The proposed approach preserves the original optimization framework while improving stability under data contamination. Through experiments on synthetic and real datasets, we show that robust LassoNet significantly improves both predictive performance and feature selection accuracy in th
    
[^291]: 面向受限疫情政策规划的图世界模型

    Graph World Models for Constrained Epidemic Policy Planning

    [https://arxiv.org/abs/2609.35545](https://arxiv.org/abs/2609.35545)

    EpiMind是一个用于受限疫情政策规划的图世界模型框架，通过图因子化循环状态空间模型和图-时间ADMM实现跨区域协调干预，在保证共享资源预算可行性的同时将入院预测误差降低29%。

    

    疫情政策规划通常需要在地理区域之间进行协调，既要考虑人口流动带来的跨区域溢出效应，又要考虑如何利用有限的资源。现有方法要么缺乏针对耦合动力学的动作条件模型，要么无法保证每个时期的可行性。我们提出了EpiMind，一个用于跨区域受限疫情政策规划的图世界模型框架。图因子化的循环状态空间模型从区域潜在信念生成联合的策略条件推演，而图-时间ADMM优化区域干预措施，通过投影确保共享资源的可行性，并在学习到的模型下评估时间规范。相对于无图结构的动力学建模，EpiMind将入院人数RMSE降低了29%；在保证共享预算可行性的前提下，其规划结果与最佳可行恒定政策相差在1-5%以内；并且在三种资源预算下优于所有可部署的基线方法。

    arXiv:2609.35545v2 Announce Type: replace-cross  Abstract: Epidemic policy planning often requires coordination between geographical regions, taking into account mobility-driven spillovers and how to make use of limited resources. Existing methods either lack action-conditioned models of coupled dynamics or cannot guarantee per-period feasibility. We present EpiMind, a graph world model framework for constrained epidemic policy planning across regions. A graph-factored recurrent state-space model generates joint policy-conditioned rollouts from regional latent beliefs, while graph-temporal ADMM optimizes regional interventions, enforces shared-resource feasibility through projection, and evaluates temporal specifications under the learned model. EpiMind reduces admission RMSE by 29% relative to graph-free dynamics modeling, plans within 1-5% of the best feasible constant policy with guaranteed shared-budget feasibility, and outperforms all deployable baselines across three resource bud
    
[^292]: ThinkNet：用于高效运动想象脑电分类的进化式深度集成框架

    ThinkNet: Evolutionary Deep Ensembles for Efficient Motor-Imagery EEG Classification

    [https://arxiv.org/abs/2609.33967](https://arxiv.org/abs/2609.33967)

    ThinkNet提出了一种验证控制的框架，通过仅训练数据归一化、验证引导的进化搜索和验证门控推理，在严格防止测试受试者信息泄露的前提下，实现面向新用户的高效、紧凑的运动想象脑电分类。

    

    实用的辅助与康复类脑机接口需要受试者无关的运动想象脑电（MI-EEG）解码器，能够在目标用户数据有限且计算资源受限的条件下泛化到新用户。然而，当测试受试者的信息影响了预处理、模型选择或集成选择时，留出受试者的性能可能被高估。我们提出了ThinkNet，这是一个验证控制的框架，结合仅基于训练数据的归一化、验证引导的进化搜索和验证门控推理，为留出受试者识别紧凑的解码器与推理策略。我们在BCI Competition IV-2a（T会话）四分类解码任务上进行评估，采用九个留一受试者交叉验证折、三个随机种子、七个固定解码器条目，并对十个代表性解码器家族进行更广泛的搜索；留出受试者从不被用于归一化、超参数、架构或集成策略的选择。

    arXiv:2609.33967v2 Announce Type: replace-cross  Abstract: Practical assistive and rehabilitative brain--computer interfaces require subject-independent motor-imagery EEG (MI-EEG) decoders that generalize to new users under limited target-user data and constrained compute. However, held-out-subject performance can be overstated when test-subject information influences preprocessing, model selection, or ensemble selection. We present \textit{ThinkNet}, a validation-controlled framework that combines train-only normalization, validation-guided evolutionary search, and validation-gated inference to identify compact decoders and inference policies for held-out subjects. We evaluate four-class BCI Competition IV-2a (session T) decoding with nine Leave-One-Subject-Out (LOSO) folds, three seeds, seven fixed decoder entries, and a broader search over ten representative decoder families; the held-out subject is never used for normalization, hyperparameter, architecture, or ensemble-policy selec
    
[^293]: LLM4Trust：探索大语言模型在信任评估中的能力

    LLM4Trust: Exploring the Capabilities of Large Language Models for Trust Evaluation

    [https://arxiv.org/abs/2609.33521](https://arxiv.org/abs/2609.33521)

    本文提出首个系统性基准框架LLM4Trust，通过构建信任图建模五种基本信任属性，并在九种提示方法下评估八个代表性大语言模型的属性理解能力，系统探索了LLM在信任评估中的潜力。

    

    信任评估通过支持风险缓解和决策制定，在网络安全中发挥着关键作用。目前已提出多种信任评估方法，其中基于学习的方法具有高准确性和自动化的优势。然而，这些方法通常需要大量的真实标注数据，训练效率低下，缺乏对基本信任属性的支持，且可解释性有限。大语言模型（LLM）凭借其强大的零样本/少样本推理能力和广泛的知识储备，提供了一种极具吸引力的替代方案。为此，我们提出了LLM4Trust，这是首个系统探索大语言模型信任评估能力的基准框架。我们首先构建了多样化的信任图来建模五种基本信任属性，并设计了相应的属性理解任务。随后，我们评估了八个代表性大语言模型在九种提示方法下理解这些属性的能力。基于这一探索…

    arXiv:2609.33521v2 Announce Type: replace  Abstract: Trust evaluation plays a critical role in cybersecurity by supporting risk mitigation and decision-making. A variety of trust evaluation methods have been proposed, with learning-based approaches offering high accuracy and automation. However, they often require substantial ground truth, suffer from low training efficiency, lack support for basic trust properties, and provide limited explainability. Large Language Models (LLMs) offer a compelling alternative due to their strong zero-/few-shot reasoning abilities and broad knowledge. To this end, we propose LLM4Trust, the first benchmark framework that systematically explores the capabilities of LLMs for trust evaluation. We first construct diverse trust graphs to model five basic trust properties and design corresponding property understanding tasks. We then assess the ability of eight representative LLMs to understand these properties under nine prompt methods. Based on this explora
    
[^294]: 表格基础模型的注意力机制基准测试

    Benchmarking Attention for Tabular Foundation Models

    [https://arxiv.org/abs/2609.31306](https://arxiv.org/abs/2609.31306)

    该论文针对表格基础模型中独特的二维行/列注意力模式构建了可复现的基准测试，并在多种注意力后端上系统评估其性能，填补了高效注意力研究在二维表格场景中的空白。

    

    诸如 TabPFN、Mitra 或 ConTextTab 等表格上下文学习器依赖于对潜在嵌入的二维序列交替进行行注意力和列注意力处理。这些注意力模式与语言模型中的一维情形存在显著差异：行注意力涉及更长的序列，而列注意力则作用于短得多的序列，且表格数据的跨步内存布局使得生成连续张量的开销很高。此外，与近期的语言模型相比，当前模型所使用的隐藏维度较小。然而，高效注意力机制的研究大多集中在一维序列上，二维表格场景尚未被探索。为此，我们构建了一个可复现的基准测试环境，并在多个后端——Torch SDPA（高效版和 cuDNN 版）、FlashAttention-2/3/4，以及仅用于推理的后端 vLLM 和 SageAttention——上研究了表格注意力的独特特性，测量了前向和后向吞吐……

    arXiv:2609.31306v2 Announce Type: new  Abstract: Tabular in-context learners such as TabPFN, Mitra, or ConTextTab rely on alternating row and column attention over 2D sequences of latent embeddings. These attention patterns differ markedly from the one-dimensional case in language models: row attention involves longer sequences while column attention operates on much shorter ones, and the strided memory layout of tabular data makes producing contiguous tensors costly. Moreover, the hidden dimensions used in current models are small compared to recent language models. Yet efficient attention has been studied mostly for one-dimensional sequences, leaving the two-dimensional tabular setting unexplored. To this end, we create a reproducible benchmarking setup and study the unique characteristics of tabular attention across several backends -- Torch SDPA (efficient and cuDNN), FlashAttention-2/3/4, and the inference-only backends vLLM and SageAttention -- measuring forward and backward thro
    
[^295]: RLVR²：基于可验证评分量规排序的强化学习

    RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking

    [https://arxiv.org/abs/2609.23457](https://arxiv.org/abs/2609.23457)

    本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。

    

    基于可验证奖励的强化学习（RLVR）正在从具有明确定义正确性信号的任务（如数学和代码）扩展到由多维度评分量规指定的多方面质量要求。由于策略优化在每次 rollout 中只消耗一个标量，基于评分量规的流程必须将多个标准的得分聚合为标量奖励。这种聚合通常被视为简单的分数缩放，但它实际上隐式地决定了训练过程中各质量维度之间的权衡方式。目前主流的做法是对每个标准进行归一化后取线性组合，这隐含地假设了基数分数差异在各标准之间是可比的，并且某一标准上的收益可以补偿另一标准上的失败；当各标准在语义上异质时，这两种假设都不可靠。我们提出了基于可验证评分量规排序的强化学习（RLVR²），这是一种面向基于评分量规的 RLVR 的可验证排序范式。对于每个标准……

    arXiv:2609.23457v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is expanding from tasks with well-defined correctness signals, such as mathematics and code, toward multifaceted quality requirements specified by multi-dimensional rubrics. Since policy optimization consumes one scalar per rollout, rubric-based pipelines must map multiple criterion scores into a scalar reward. This aggregation is often treated as score scaling, but it implicitly determines how quality dimensions trade off during training. The prevailing practice, normalizing each criterion and taking a linear combination, assumes that cardinal score differences are comparable across criteria and that gains on one criterion compensate for failures on another; both assumptions are unreliable when criteria are semantically heterogeneous. We propose Reinforcement Learning with Verifiable Rubric-based Ranking (RLVR$^2$), a verifiable ranking paradigm for rubric-based RLVR. For each criter
    
[^296]: 信任，但要验证工具：在自建安全回归代理上审计AI生成的RTL验证计划

    Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies

    [https://arxiv.org/abs/2609.19844](https://arxiv.org/abs/2609.19844)

    论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。

    

    AI生成的RTL验证计划可能满足提供商的模式，但在与可信执行的边界处失败。我们提出了SecTB-RTL，一个涵盖31个任务和124个自建硬件安全回归测试的可审计框架。确定性非AI基线在递增的资源限制下分别杀死了36、75和78个突变体。第一次确认性运行（C1-R2）在模型执行之前就失败了，因为提供商拒绝了其响应模式。在未查看结果的情况下进行仅模式修复后，单独冻结的后续运行（C1-R3）完成了1,860次调用。提供商接受了1,857个响应，但只有九个通过了生产语义验证器。生成规则和执行规则并不匹配。因此，我们将该运行保留为工具验证事件，不报告任何提示效果估计。这一事件表明，提供商或模式的接受并不能证明执行的有效性。编译和覆盖率仅是诊断指标。

    arXiv:2609.19844v1 Announce Type: cross  Abstract: AI-generated RTL verification plans can satisfy a provider schema yet fail at the boundary to trusted execution. We present SecTB-RTL, an auditable framework covering 31 tasks and 124 authored hardware-security regressions. A deterministic non-AI baseline killed 36, 75, and 78 mutants at increasing resource limits. The first confirmatory run (C1-R2) failed before model execution because the provider rejected its response schema. After a schema-only repair made without viewing outcomes, a separately frozen follow-up run (C1-R3) completed 1,860 calls. The provider accepted 1,857 responses, but only nine passed the production semantic validator. The generation and execution rules did not match. We therefore preserve the run as an instrument-validation incident and report no prompt-effect estimate. This incident shows that provider or schema acceptance does not establish execution validity. Compilation and coverage are only diagnostics; th
    
[^297]: 一种用于循证自闭症谱系障碍筛查的多模态大语言模型

    A multimodal large language model for evidence-based autism spectrum disorder screening

    [https://arxiv.org/abs/2609.16464](https://arxiv.org/abs/2609.16464)

    本研究提出ASDchat——一个采用双分支架构的多模态大语言模型，能够基于视频、音频和对话进行自闭症筛查，同时生成与ADOS-2临床标准对齐的可追溯、带时间戳的行为证据，在中国27个站点1,035名参与者的数据集上达到0.953的AUC。

    

    自闭症谱系障碍（ASD）的临床管理在早期筛查环节面临瓶颈，主要原因在于训练有素的专家稀缺，且传统评估工具具有主观性。在此，我们介绍ASDchat，一个专为循证ASD筛查设计的多模态大语言模型，它以视频、音频和对话作为输入。ASDchat采用双分支架构，其中决策分支生成筛查概率，证据分支生成与标准化临床标准（ADOS-2）对齐的、可追溯的、带有时间戳的行为证据。该模型在中国27个站点共1,035名参与者的数据集上进行训练和评估，数据涵盖典型发育（TD）儿童、ASD儿童以及其他障碍儿童。在ASD与TD的区分任务中，ASDchat达到了0.953 ± 0.021的受试者工作特征曲线下面积（AUC）。在9个未参与训练的留存站点上……

    arXiv:2609.16464v1 Announce Type: cross  Abstract: The clinical management of autism spectrum disorder (ASD) faces a bottleneck in early screening, mainly because trained specialists are scarce and conventional assessment tools are subjective. Here, we introduce ASDchat, a multimodal large language model designed for evidence-based ASD screening, which takes video, audio, and dialogue as input. ASDchat adopts a dual-branch architecture, where the decision branch generates screening probabilities and the evidence branch generates traceable, timestamped behavioral evidence aligned with standardized clinical criteria (ADOS-2). The model was trained and evaluated on a dataset of 1,035 participants from 27 sites in China, which covered typically developing (TD) children, children with ASD, and children with other disorders. For ASD versus TD, ASDchat reached an area under the receiver operating characteristic curve (AUC) of 0.953 $\pm$ 0.021. On 9 held-out sites that were not used for train
    
[^298]: 使用机器学习预测社交媒体互动度

    Predicting Social Media Engagement using Machine Learning

    [https://arxiv.org/abs/2609.16082](https://arxiv.org/abs/2609.16082)

    本研究通过提取家具公司Facebook图片帖子的视觉、文本和时间特征，并运用随机森林、LightGBM等机器学习模型来预测社交媒体互动度。

    

    社交媒体平台因其庞大的用户群体和便捷的访问方式，成为传播信息的热门渠道。企业也将社交媒体视为广告流程中的重要组成部分。通过发布高质量的帖子，企业可以提升其互动指标并增加粉丝数量。尽管已有越来越多的研究关注社交媒体互动，但很少有研究综合考察图片帖子的视觉、文本和时间特征，尽管这些特征共同决定了内容在社交媒体上的表现。为了理解社交媒体互动的关键驱动因素，我们收集了家具公司在Facebook上发布的图片帖子，并运用文本和图像分析方法从中提取视觉、时间和文本特征。我们评估了多种机器学习模型——包括随机森林、轻量梯度提升机（LightGBM）和极限梯度提升等。

    arXiv:2609.16082v1 Announce Type: cross  Abstract: Social media platforms are popular channels for disseminating information, owing to their large user bases and ease of access. Companies also use social media as an important aspect of the advertising process. By creating high-quality posts, companies can strengthen their engagement metrics and increase their follower count. While a growing body of research has examined social media engagement, fewer studies have jointly examined the visual, textual, and temporal features of image posts, even though these features collectively determine the performance of content on social media. To understand the important drivers of social media engagement, we collect image posts of furniture firms on Facebook and extract visual, temporal, and textual features from them using text and image analytics methods. We evaluate several machine learning models - including Random Forest, Light Gradient Boosting Machine (LightGBM), and eXtreme Gradient Boostin
    
[^299]: 基于正交化动量的非光滑优化

    Nonsmooth Optimization via Orthogonalized Momentum

    [https://arxiv.org/abs/2609.13677](https://arxiv.org/abs/2609.13677)

    本文首次在与反向传播兼容的广义导数框架下证明：对于任意固定动量因子β∈[0,1)，正交化动量优化器Muon在凸Lipschitz非光滑目标上几乎从任何初始化出发都可能无法收敛到全局最优解，揭示了其超越光滑优化的根本局限。

    

    现代实际应用问题涉及矩阵值参数，然而传统优化器将其视为向量，这促使了利用输入输出几何结构的矩阵感知方法的出现，例如Muon——它在参数更新之前对动量矩阵进行正交化。其经验上的成功引发了一个概念性问题：正交化动量在光滑优化之外是否仍然有效？本文在与反向传播兼容的广义导数框架下，针对局部Lipschitz函数研究了这一问题。我们的第一个贡献是识别出一个关键局限性：对于每个固定的动量因子β∈[0,1)，当步长适应于完整的梯度历史时，Muon几乎从任何初始化出发都可能无法逼近凸Lipschitz目标函数的全局最优解。这种失败甚至在迭代序列有界的情况下也可能发生。我们的例子受Parshakova等人工作的启发，但后者仅覆盖β<1的特殊情形。

    arXiv:2609.13677v1 Announce Type: cross  Abstract: Modern real application problems involve matrix-valued parameters, yet conventional optimizers treat them as vectors, thereby motivating matrix-aware methods that exploit input-output geometry, such as Muon which orthogonalizes the momentum matrices before parameter updates. Its empirical success raises a conceptual question: can orthogonalized momentum remain effective beyond smooth optimization? This paper studies this question for locally Lipschitz functions using a generalized derivative framework compatible with backpropagation. Our first contribution is to identify a key limitation: for every fixed momentum factor $\beta\in[0,1)$, Muon can fail to approach the global optimal solution of a convex Lipschitz objective from almost every initialization, when step sizes adapt to the full gradient history. The failure can occur even along bounded iterates. Our example is inspired by the one of Parshakova et al. which only covers $\beta\
    
[^300]: 是什么使对抗样本能够跨深度伪造检测器迁移？

    What Makes Adversarial Examples Transfer Across Deepfake Detectors?

    [https://arxiv.org/abs/2609.10002](https://arxiv.org/abs/2609.10002)

    该研究通过对涵盖六种骨干网络、两种预训练方案和五种训练数据配置的60个深度伪造检测器进行受控评估，发现当攻击者的源模型与目标模型共享相同的骨干网络、架构家族、预训练方案或训练数据时，黑盒对抗迁移攻击的成功率显著更高，且这种兼容性效应因攻击方法而异。

    

    深度伪造检测器仍然容易受到基于迁移的黑盒攻击的影响，在这类攻击中，对抗样本在源替代模型上生成，随后迁移到攻击者无法访问的目标模型。然而，源模型与目标模型之间的兼容性如何影响攻击成功率，目前仍缺乏深入理解。先前的研究仅评估了有限的检测器集合，且很少将架构因素与训练因素解耦开来。我们对对抗样本的可迁移性进行了受控评估，涵盖60个检测器，涉及六种骨干网络、两种预训练方案和五种训练数据配置，并使用了两种攻击方法：AutoAttack（AA）和结合变换期望的Carlini-Wagner攻击（CW-EOT）。匹配比较结果表明，当源模型与目标模型共享完全相同的骨干网络、架构家族、预训练方案或训练数据时，攻击的迁移成功率显著更高。这种兼容性结构依赖于攻击方法：完全相同的骨干网络兼容性具有最……（原文此处截断）

    arXiv:2609.10002v1 Announce Type: cross  Abstract: Deepfake detectors remain vulnerable to transfer-based black-box attacks, in which adversarial examples are generated on a source surrogate model and transferred to a target model, unknown to the attacker. Yet how source--target compatibility shapes attack success remains poorly understood. Prior studies evaluate limited detector pools and rarely disentangle architectural from training factors. We conduct a controlled evaluation of adversarial transferability across 60 detectors spanning six backbones, two pretraining regimes, and five training-data configurations, using two attack procedures: AutoAttack (AA) and the Carlini--Wagner attack with Expectation over Transformation (CW--EOT). Matched comparisons reveal significantly higher transfer when source and target share an exact backbone, architecture family, pretraining regime, or training data. This compatibility structure is attack-dependent: exact backbone compatibility has the la
    
[^301]: SUN：在强化学习中追求新颖性

    SUN: Reaching for Novelty in Reinforcement Learning

    [https://arxiv.org/abs/2609.08642](https://arxiv.org/abs/2609.08642)

    本文提出基于后继价值函数的SUN指标，可同时衡量目标的新颖性与可达性，该可达性感知的目标选择框架能无缝集成到任意离策略强化学习算法中，并具有恢复计数奖励、约束命中概率和拒绝不可达目标的理论保证。

    

    强化学习（RL）中的探索仍然是一个根本性的挑战。近来的目标条件强化学习策略（通过选择目标来鼓励更广泛的状态覆盖）已展现出有希望的结果，但没有一种方法能够同时依据新颖性和可达性对目标进行评分：这两个信号要么被手动权衡，要么被依次应用，或者其中之一被完全忽视。在本文中，我们引入了一个可达性感知的目标选择框架，它明确地整合了这两个方面，并且可以无缝地融入任何离策略（off-policy）强化学习算法中。为此，我们提出了SUccessor-to-Novelty（SUN），这是一个由后继价值函数（successor value functions）导出的指标，用于识别既新颖又可达的目标。我们证明SUN在极限情况下可恢复基于计数的奖励，能够约束短时域命中概率，并且可以可证明地拒绝不可达的目标。我们进一步提出了一种利用这些特性的自适应目标选择策略，以及一个准确的……（摘要在此处截断）

    arXiv:2609.08642v1 Announce Type: cross  Abstract: Exploration in reinforcement learning (RL) remains a fundamental challenge. Recent goal-conditioned RL strategies (which select goals to encourage broader state coverage) have shown promising results, but none scores a goal by novelty and reachability jointly: the two signals are traded off by hand, applied in sequence, or one is neglected outright. In this paper, we introduce a reachability-aware goal-selection framework that explicitly integrates these two aspects, and that can be seamlessly incorporated into any off-policy RL algorithm. To this aim, we propose SUccessor-to-Novelty (SUN), an indicator derived from successor value functions to identify goals that are both novel and reachable. We prove that SUN recovers count-based bonuses in the limit, bounds short-horizon hitting probabilities, and provably rejects unreachable goals. We further present an adaptive goal-selection strategy that leverages these properties, and an accura
    
[^302]: 测试协议设置对跨 Trust-Hub 家族基于特征的硬件木马检测的影响

    Protocol effects on feature-based hardware-Trojan detection across Trust-Hub families

    [https://arxiv.org/abs/2609.07199](https://arxiv.org/abs/2609.07199)

    该论文发现基于特征的硬件木马检测性能严重依赖测试集划分协议——当测试门电路来自训练中已见过的宿主电路时性能被高估，而按宿主家族留出时 F1/AP 会从约 0.91/0.98 骤降至约 0.46/0.58。

    

    Trust-Hub 复用宿主电路：多个文件之间的主要区别仅在于插入的木马。当同一家族兄弟变体的门电路同时进入训练集和测试集时，检测器可以从已经见过的宿主逻辑中获益。我们测量的是这种效应，而不是提出又一种分类器。语料库包含来自 16 个网表的 49,124 个门电路，分为五个宿主家族。我们保持解析器、36 个门级特征、类别权重、模型设置、阈值和家族级聚合方式不变，仅改变一个选择：测试集边界。三种设置分别从合并语料库中抽取测试门电路、留出一个完整网表、或留出一个宿主的所有变体。这一选择至关重要：随机森林在合并门电路下 F1/AP 为 0.914/0.978，留出一个完整网表时降至 0.636/0.851，留出一个宿主家族时仅为 0.460/0.577；在同样对比下，XGBoost 从 0.946/0.976 下降到 0.464/0.544；逻辑回归损失了 AP，尽管其固定阈值……

    arXiv:2609.07199v1 Announce Type: cross  Abstract: Trust-Hub reuses host circuits: several files differ mainly in the inserted Trojan. When gates from sibling variants enter both training and test folds, a detector can benefit from host logic it has already seen. We measure that effect instead of proposing another classifier. The corpus contains 49,124 gates from 16 netlists grouped into five host families. We left the parser, 36 gate features, class weighting, model settings, threshold, and family-level aggregation unchanged and altered one choice: the test boundary. The three settings draw test gates from the pooled corpus, withhold a complete netlist, or withhold every variant of one host. The choice matters. Random forest records F1/AP of 0.914/0.978 with pooled gates, 0.636/0.851 with one netlist held out, and 0.460/0.577 with a host family held out. XGBoost falls from 0.946/0.976 to 0.464/0.544 across the same comparison. Logistic regression loses AP, although its fixed-threshold
    
[^303]: 基于可微分混合建模从实验数据中学习与优化化学传输过程

    Differentiable Hybrid Modelling for Learning and Optimising Chemical Transport Processes from Experimental Data

    [https://arxiv.org/abs/2609.04011](https://arxiv.org/abs/2609.04011)

    提出一个通用的可微分混合建模框架，将JAX有限体积群体平衡求解器与可学习的神经网络组件相结合，能够从真实实验数据中同时发现本构定律并拟合初始条件，用于学习和优化化学传输过程。

    

    可靠的传输模型在模拟和优化众多化学工程过程中至关重要，然而，大多数模型采用人工挑选的本构定律，这可能无法反映真实情况，并且通常假设初始条件是完全已知的。这两种限制都可能显著偏倚模型预测，并在预测和控制应用中导致系统性误差。黑箱神经代理替代建模方法能够更好地匹配真实的实验数据，但仅局限于其训练所用的任务，且无法对其进行物理一致性检验。在此，我们提出了一个面向传输过程的通用可微分混合建模框架，特别针对群体平衡方程的情形。我们的框架将JAX有限体积群体平衡求解器与可学习的神经网络组件相集成，这些组件经训练后既能发现本构定律，又能从真实实验数据中拟合初始条件。

    arXiv:2609.04011v1 Announce Type: cross  Abstract: Reliable transport models are essential when modelling and optimising many chemical engineering processes, yet, most models assume hand-picked constitutive laws which may not reflect reality, and often assume initial conditions are known exactly. Both restrictions can significantly bias model predictions and lead to systematic error when used in predictive and control settings. Black-box neural surrogate alternatives for modelling can better match real example data, but are confined to the task they were trained on and cannot be interrogated for physical consistency. Here we introduce a general-purpose differentiable hybrid modelling framework for transport processes, specifically for the case of population balance equations. Our framework integrates a JAX finite volume population balance solver with learnable neural network components which are trained to both discover constitutive laws and fit initial conditions from real experimenta
    
[^304]: DASC：面向混合线性注意力服务的衰减感知状态压缩

    DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving

    [https://arxiv.org/abs/2608.30386](https://arxiv.org/abs/2608.30386)

    提出衰减感知状态压缩（DASC），通过从模型权重中推导不同头和通道的“保留视野”来压缩混合线性注意力模型的持久状态检查点，从而缓解内存压力、减少缓存驱逐和重复预填充。

    

    混合线性注意力架构最近已扩展至大型开源权重模型，在提供与全注意力相当的质量的同时，大幅减少了键/值（KV）缓存的增长。然而，其原位递归状态更新使缓存管理变得复杂：前缀重用需要在全注意力KV之外额外保存状态检查点，而完整存储状态检查点又会增加内存压力，导致更多的缓存驱逐和重复预填充。通过分析门控DeltaNet（GDN）和Kimi Delta注意力（KDA）的衰减结构，我们发现不同的头和通道在显著不同的时间尺度上保留前缀信息，我们将其称为“保留视野”。这种差异表明持久状态检查点存在巨大的压缩潜力。基于这一观察，我们提出了衰减感知状态压缩（DASC），它从模型权重中推导保留视野，选择长视野状态……（摘要在此处截断）

    arXiv:2608.30386v1 Announce Type: cross  Abstract: Hybrid linear-attention architectures have recently scaled to large open-weight models, offering quality competitive with full attention while substantially reducing key/value (KV) cache growth. However, their in-place recurrent-state updates complicate cache management: prefix reuse requires state checkpoints alongside full-attention KV, while storing state checkpoints in full increases memory pressure, leading to more evictions and repeated prefill. By analyzing the decay structure of Gated DeltaNet (GDN) and Kimi Delta Attention (KDA), we find that different heads and channels retain prefix information over markedly different timescales, which we term \emph{retention horizons}. This variation suggests substantial compression potential in persistent state checkpoints. Building on this observation, we introduce \emph{Decay-Aware State Compression} (DASC), which derives retention horizons from model weights, selects long-horizon state 
    
[^305]: 面向策略优化的偏好引出方法及其在心脏移植与人类价值对齐中的应用

    Preference Elicitation for Policy Optimization and Application to Aligning Heart Transplantation with Human Values

    [https://arxiv.org/abs/2608.28620](https://arxiv.org/abs/2608.28620)

    提出一种两阶段的线性效用偏好引出算法，通过成对比较学习切割平面并可证明收敛到用户效用函数，从而直接从分配结果学习效用函数用于策略优化，并成功应用于心脏移植分配的价值对齐问题。

    

    偏好引出对于将人工智能系统与人类价值观对齐至关重要。以往的方法（例如用于器官分配的方法）通常要求利益相关者比较算法做出的决策（例如患者A与患者B孰先孰后）。这种决策层面的方法混淆了手段与目的。与此不同，我们直接针对分配结果引出偏好，以学习用于策略优化的效用函数。我们构建了一种新颖的面向线性效用的偏好引出算法，其在实践中优于以往的技术。我们的算法分为两个阶段：第一阶段通过成对比较学习切割平面，快速缩小可能的属性权重空间，并通过剔除被支配区域为第二阶段热身；第二阶段则可证明地收敛到用户的效用函数。我们将该技术应用于心脏移植分配问题，其中分配策略必须平衡移植后结局与等待期死亡率等相互竞争的目标。

    arXiv:2608.28620v1 Announce Type: new  Abstract: Preference elicitation is essential for aligning AI systems with human values. Prior approaches (e.g., for organ allocation) often ask stakeholders to compare the decisions of an algorithm (e.g., patient A vs. patient B). Such a decision-level approach conflates the means with the ends. Instead, we elicit preferences directly over allocation outcomes to learn a utility function for policy optimization. We construct a novel preference elicitation algorithm for linear utilities that outperforms prior techniques in practice. Our algorithm has two phases. The first phase learns cutting planes through pairwise comparisons to rapidly shrink the space of possible attribute weights and warm-starts the second phase by eliminating dominated regions. The second phase then provably converges to the user's utility function. We apply our technique to heart transplant allocation where a policy must balance competing objectives such as post-transplant o
    
[^306]: 基于上下文感知语义嵌入的表格学习器增强方法

    Enhancing Tabular Learners with Context-Aware Semantic Embeddings

    [https://arxiv.org/abs/2608.03565](https://arxiv.org/abs/2608.03565)

    提出CASE框架，通过用代表性行样本预填充基于Gemma 3的表格语言模型的KV缓存来为表格行嵌入提供数据集级语义上下文，从而弥合大语言模型语义理解与表格学习器统计能力之间的差距，解决文本特征的语义歧义问题。

    

    虽然现代表格学习器擅长捕捉统计模式，但它们常常在“语义真空”中运行，将文本特征视为离散符号，忽略了特征名称或单元格条目中蕴含的丰富语义。我们提出了CASE（上下文感知语义嵌入），这是一个新颖的框架，旨在弥合大型语言模型（LLM）的语义理解能力与表格学习器统计能力之间的鸿沟。与现有方法将行独立嵌入不同，CASE采用了一种上下文化策略：我们使用具有代表性的行样本预填充一个基于Gemma 3定制训练的表格语言模型的KV缓存，从而建立数据集语义的持久锚点。这确保了生成的行嵌入能够被动态上下文化，解决语义歧义，并将表示锚定在特定领域的上下文中。我们在多个基准测试（CARTE、TextTab等）上进行了实验……

    arXiv:2608.03565v2 Announce Type: replace  Abstract: While modern tabular learners excel at capturing statistical patterns, they frequently operate in a semantic vacuum, treating textual features as discrete symbols, ignoring the rich semantics inherent in feature names or cell entries. We propose CASE (Context-Aware Semantic Embeddings), a novel framework that bridges the gap between the semantic understanding of Large Language Models (LLMs) and the statistical capabilities of tabular learners. Unlike existing methods that embed rows in isolation, CASE utilizes a contextualization strategy: we pre-fill the KV cache of a custom-trained Gemma 3-based Tabular Language Model with a representative sample of rows to establish a persistent anchor of the dataset's semantics. This ensures that generated row embeddings are dynamically contextualized, resolving semantic ambiguities and anchoring representations in domain-specific context. Our experiments across several benchmarks (CARTE, TextTab
    
[^307]: DRIFT：利用对抗性贴片攻击使流匹配VLA模型的去噪轨迹脱轨

    DRIFT: Derailing Denoising Trajectories of Flow-Matching VLAs with Adversarial Patch Attack

    [https://arxiv.org/abs/2608.03207](https://arxiv.org/abs/2608.03207)

    本文提出DRIFT通用对抗性贴片攻击，揭示流匹配VLA模型（如pi0）的鲁棒性只是错觉——仅对第一个去噪步骤施加攻击即可高效地使现成策略失效。

    

    诸如pi0等流匹配视觉-语言-动作（VLA）模型通过积分学习到的去噪速度场来生成机器人动作，据报道这类模型能够抵抗那些轻易欺骗自回归VLA的对抗性扰动。我们证明这种鲁棒性在很大程度上是一种错觉：它源于先前的攻击方法忽略了多步去噪常微分方程（ODE）。我们提出DRIFT（通过输入扰动实现流匹配轨迹的去噪重定向），这是一种放置在机器人夹爪上的测试时通用对抗性贴片，能够攻击现成策略的去噪速度场。我们的核心发现是反直觉的：仅攻击第一个去噪步骤比攻击更宽的步骤窗口既更有效又更廉价，我们通过输入空间优化中特有的梯度冲突来解释这一现象，这与训练时后门攻击的情形恰好相反。在pi0和pi0.5模型上的四个LIBERO套件实验中，DRIFT……

    arXiv:2608.03207v2 Announce Type: replace-cross  Abstract: Flow-matching vision-language-action (VLA) models such as pi0 generate robot actions by integrating a learned denoising velocity field, and have been reported to resist adversarial perturbations that readily fool autoregressive VLAs. We show that this robustness is largely illusory: it stems from prior attacks ignoring the multi-step denoising ODE. We introduce DRIFT (Denoising Redirection via Input perturbation of the Flow-matching Trajectory), a test-time universal adversarial patch placed on the robot's gripper that attacks the denoising velocity field of an off-the-shelf policy. Our central finding is counterintuitive: attacking only the first denoising step is both stronger and cheaper than attacking a wider window of steps, which we explain through a gradient conflict unique to input-space optimization and which is exactly opposite to the training-time backdoor regime. On pi0 and pi0.5 across four LIBERO suites, DRIFT bre
    
[^308]: 基于粒子的广义随机优化

    Particle-based Generalised Stochastic Optimisation

    [https://arxiv.org/abs/2608.02844](https://arxiv.org/abs/2608.02844)

    本文提出了针对不可解梯度损失函数的基于扩散的广义随机粒子优化框架，通过平均场动力学及其交互粒子近似统一了多种现有算法，证明了指数收敛性与非渐近误差界，并据此构建了高阶Langevin和动量优化器。

    

    我们为具有难以求解梯度的损失函数开发了一类基于扩散的通用随机粒子优化方法。具体而言，我们考虑损失梯度是关于一个依赖于参数的分布的积分这一问题结构，该结构涵盖了训练生成模型、微调以及学习潜变量模型等场景。我们引入了平均场动力学及其交互粒子近似，其中包含多种现有算法作为特例，并为构建新方法提供了途径。在适定性与联合压缩性假设下，我们证明了指数收敛性，并表明连续时间粒子系统满足非渐近误差界。我们通过开发高阶Langevin和动量优化器，并在最大边际似然估计和基于能量的模型训练上进行评估，展示了该框架的应用。

    arXiv:2608.02844v2 Announce Type: replace-cross  Abstract: We develop a class of general diffusion-based stochastic particle optimisation methods for loss functions with intractable gradients. Specifically, we consider problems in which the loss gradient is an integral with respect to a parameter-dependent distribution, a structure that includes training generative models, fine-tuning, and learning latent-variable models. We introduce mean-field dynamics and its interacting-particle approximations, which contain several existing algorithms as special cases and provide a route to constructing new methods. Under well-posedness and joint contractivity assumptions, we prove exponential convergence and show that the continuous-time particle system admits a non-asymptotic error bound. We illustrate our framework by developing higher-order Langevin and momentum optimisers and evaluating them on maximum marginal-likelihood estimation and energy-based-model training.
    
[^309]: 第四象限：良性失配的风格化视角

    The Fourth Quadrant: A Stylized View of Benign Misfitting

    [https://arxiv.org/abs/2608.01032](https://arxiv.org/abs/2608.01032)

    该论文揭示了“良性失配”现象——在单尖峰线性回归模型的某个训练集大小范围内，任何泛化良好的线性预测器对训练数据的拟合都必须比零预测器更差，从而刻画了泛化能力与训练拟合之间一种反直觉的权衡关系。

    

    训练误差是我们在训练集上能够观察到的量，而测试误差才是我们真正关心的量。我们研究了确定性 $(d+1)$ 维单尖峰模型中的平方误差线性回归。每个训练向量的尖峰坐标为 $\sqrt{\gamma}$（其中 $\gamma>1$），并具有等范数、相互正交的干扰分量。训练标签为 $1$。新的测试点服从中心化高斯分布，协方差为 $\operatorname{diag}(\gamma,1,\ldots,1)$，其无噪声标签为 $x[1]/\sqrt{\gamma}$。我们计算了训练张成空间中线性预测器的精确训练-测试权衡。当尖峰方向未知时，允许使用所有线性预测器并不能改善这一权衡，无论在与数据一致的方向上取平均还是取最坏情况都是如此。存在一个训练集大小的范围，在该范围内，每一个泛化良好的张成空间预测器对训练数据的拟合都必须比零预测器更差。我们将这种现象称为良性失配，或第四象限。

    arXiv:2608.01032v2 Announce Type: replace  Abstract: Training error is what we can observe on a training set; test error is the quantity we care about. We study squared-error linear regression in a deterministic $(d+1)$-dimensional single-spike model. Each training vector has spike coordinate $\sqrt{\gamma}$, $\gamma>1$, and equal-norm, mutually orthogonal nuisance components. Training labels are $1$. Fresh test points are centered Gaussian with covariance $\operatorname{diag}(\gamma,1,\ldots,1)$ and noise-free label $x[1]/\sqrt{\gamma}$. We compute the exact train--test tradeoff for linear predictors in the training span. When the spike direction is unknown, allowing all linear predictors does not improve this tradeoff, on average or in the worst case over directions consistent with the data. There is a range of training-set sizes where every span predictor that generalizes well must fit the training data worse than the zero predictor. We call this benign misfitting, or the fourth qua
    
[^310]: 一阶上下文内沙普利：面向大语言模型对齐与评估的可扩展数据审计

    First-Order In-Context Shapley: Scalable Data Auditing for LLM Alignment and Evaluation

    [https://arxiv.org/abs/2607.22766](https://arxiv.org/abs/2607.22766)

    该论文提出了一种仅需推理、无需重训练的一阶上下文内 Shapley 数据估值方法，可在 O(k) 次前向传播内完成对大语言模型对齐数据的可扩展审计，以 19 倍更少的计算量实现了与蒙特卡洛上下文内 Shapley 在全部目标样本上一致的估值效果。

    

    大语言模型的后训练对齐正日益受到数据质量的制约：偏好数据与指令微调语料库中不断累积标签噪声、安全反转以及标注者之间的矛盾，而去重方法无法发现这些问题，全语料库的大模型评审又成本高昂难以负担。虽然 Data Shapley 提供了一个有原则性的数据估值框架，但精确计算需要进行难以承受的重训练，而诸如 KNN-Shapley 之类的嵌入代理方法则无法区分高效用与低效用的演示样本（与生成式效用的相关性仅为 ρ = 0.0003）。我们提出了一种仅基于推理的估值流水线，它在语义 k 近邻邻域上计算上下文内合作博弈的一阶项，仅需 O(k) 次前向传播且无需任何参数更新；该估值在 100% 的目标样本上与多次采样蒙特卡洛上下文内 Shapley 呈正相关，同时计算量减少 19 倍。通过将成对对数似然变化投影到有向影响图上，可以刻画每个……（摘要在此处被截断）

    arXiv:2607.22766v2 Announce Type: replace-cross  Abstract: Post-training alignment of Large Language Models is increasingly bottlenecked by data quality: preference and instruction-tuning corpora accumulate label noise, safety inversions, and inter-annotator contradictions that deduplication misses and full-corpus LLM judging cannot afford. While Data Shapley offers a principled valuation framework, exact computation requires intractable retraining, and embedding surrogates like KNN-Shapley fail to separate high- from low-utility demonstrations ($\rho = 0.0003$ with generative utility). We introduce an inference-only valuation pipeline that computes the first-order term of an in-context cooperative game over semantic $k$-NN neighborhoods in $O(k)$ forward passes without parameter updates, correlating positively with multi-shot Monte-Carlo in-context Shapley on 100% of targets at 19x less compute. Projecting pairwise log-likelihood shifts onto a directed influence graph characterizes ea
    
[^311]: 在长程智能体任务中通过首次访问覆盖从全失败组中学习

    Learning from All-Failed Groups via First-Visit Coverage in Long-Horizon Agentic Tasks

    [https://arxiv.org/abs/2607.22724](https://arxiv.org/abs/2607.22724)

    提出CovGPO方法，通过在整组rollout全部失败时引入首次访问观测覆盖，为欠采样的状态改变动作提供学习信号，从而打破长程智能体任务中采样不平衡导致的自我延续信用陷阱。

    

    基于组的策略优化（Group-based policy optimization）已被越来越多地用于通过比较组内的轨迹或步骤，从稀疏的结果奖励中训练大语言模型（LLM）智能体。然而，我们发现，在困难的长程智能体任务上，这种比较方式存在采样不平衡问题：重复的或低效用的动作主导了策略的高概率区域，使得能够改变状态的有用动作被欠采样。这种不平衡导致产生大量全失败的rollout组，而这类组内部的相对优势无法为策略更新提供任何方向。由此，采样不平衡与全失败组共同坍缩为一个自我延续的信用陷阱：以失败为主的采样无法产生学习信号，从而使重复的低效用动作得以持续存在。为了打破这一陷阱，我们提出了覆盖条件化组策略优化（Coverage-conditioned Group Policy Optimization, CovGPO），它仅当所有采样……（摘要原文在此处截断）

    arXiv:2607.22724v2 Announce Type: replace-cross  Abstract: Group-based policy optimization has been increasingly used to train large language model (LLM) agents from sparse outcome rewards by comparing trajectories or steps within a group. However, on difficult long-horizon agentic tasks, we found this comparison suffers from a sampling imbalance: repeated or low-effect actions dominate the high-probability region of the policy, leaving useful state-changing actions under-sampled. This imbalance produces many all-failed rollout groups, where the relative advantages within such a group provide no direction for updating the policy. Consequently, the sampling imbalance and the all-failed groups collapse into a self-perpetuating credit trap: failure-dominated sampling yields no learning signal, allowing repeated low-effect actions to persist. To break this trap, we propose Coverage-conditioned Group Policy Optimization (CovGPO), which uses first-visit observation coverage only when all sam
    
[^312]: MIRROR：从另一视角学习以实现多模态推理

    MIRROR: Learning from the Other View for Multi-Modal Reasoning

    [https://arxiv.org/abs/2607.21552](https://arxiv.org/abs/2607.21552)

    本文提出MIRROR强化学习方法，通过在同一问题的不同模态视角中选取模型表现最佳者作为教师，实现跨视角自蒸馏，从而提升多模态推理能力。

    

    视觉-语言模型在处理同一个底层问题时，若问题以文本形式或视觉形式呈现，其表现可能截然不同。这种不一致性表明，不同的视角会暴露出不同的推理路径和失败模式，而标准的多模态后训练方法并未充分加以利用。为了研究和利用这一现象，我们构建了ODA-Data，一个高质量的成对多模态几何数据集，其中同一问题具有文本主导、图像主导以及图像+文本组合三种视角，并附带用于训练和评估视角相关推理行为的数据划分。随后，我们提出了模态感知互惠推理优化，这是一种强化学习方法，它从模型自身的模态依赖能力中提取跨视角自蒸馏信号。对于每个问题，MIRROR会在所有视角下评估模型，选择表现最佳的视角作为教师，并对其余视角进行训练……

    arXiv:2607.21552v2 Announce Type: replace  Abstract: Vision-language models (VLMs) can behave very differently depending on whether the same underlying problem is presented textually or visually. This inconsistency suggests that different views expose different reasoning paths and failure modes that standard multimodal post-training does not fully exploit. To study and exploit this phenomenon, we construct ODA-Data, a high-quality paired multimodal geometry dataset with text-dominant, image-dominant, and combined image+text views of the same problems, together with splits for training and evaluating view dependent reasoning behaviors. We then develop Modality-Informed Reciprocal Reasoning Optimization (MIRROR), a reinforcement learning approach that derives a cross-view self-distillation signal from the model's own modality-dependent capabilities. For each problem, MIRROR evaluates the model under all views, selects the best-performing view as a teacher, and trains the remaining views 
    
[^313]: 标签监督下医学基础模型的表征收敛程度更低

    Medical foundation models converge less under label supervision

    [https://arxiv.org/abs/2607.20274](https://arxiv.org/abs/2607.20274)

    该论文通过对75万余张医学图像、18个公开模型和101个自训练模型的大规模分析发现，医学基础模型——尤其是经过标签监督训练的模型——对相同图像的表征彼此差异巨大，其近邻重叠度甚至接近随机初始化的水平，因此不同基础模型的嵌入不可互换使用。

    

    诊断分类器和影像生物标志物通常拟合在医学基础模型的嵌入表示之上，当新版本的模型出现时，这些基础模型就会被替换。这种做法隐含了一个假设：不同模型对图像的表征是相似的。我们用来自5种成像模态、14个数据集的超过75万张图像、18个公开模型，以及101个在胸片和组织病理学图像上训练的模型（这些模型在预训练目标、标签类型、模型规模、随机种子、初始化、训练患者或训练配方上各不相同）检验了这一假设。模型间的一致性用相互k近邻重叠度来衡量，即一张图像的最近邻中有多少是两个模型所共有的。两个公开模型在胸片图像的10个最近邻中平均仅有0.091的重叠，而随机初始化的网络与它们也有0.036的重叠。在其他四种成像模态中，重叠度至多为0.300。在胸片上，一致性更多取决于源数据集和……（原文在此处截断）

    arXiv:2607.20274v3 Announce Type: replace-cross  Abstract: Diagnostic classifiers and imaging biomarkers are fitted on the embeddings of medical foundation models. These models are replaced as new versions appear. This practice assumes that different models represent images alike. We tested this assumption with more than 750,000 images from 14 datasets in 5 imaging modalities, 18 public models, and 101 models trained on chest radiographs and histopathology that differ in pretraining objective, label type, size, random seed, initialization, training patients, or training recipe. Agreement was measured as the mutual k-nearest-neighbor overlap, the share of an image's nearest neighbors common to two models. Two public models shared on average 0.091 of the 10 nearest neighbors of a chest radiograph. A randomly initialized network shared 0.036 with them. In the four other modalities, they shared at most 0.300. On chest radiographs, agreement depended more on the source dataset and the radio
    
[^314]: 基于Transformer的规划：计算链与结构化上下文窗口

    Planning with Transformers: Chain of Computation and Structured Context Windows

    [https://arxiv.org/abs/2607.17710](https://arxiv.org/abs/2607.17710)

    提出计算链（COC）计算架构，通过将基于Transformer的语言模型置于迭代循环中并利用结构化上下文窗口作为外部记忆，以弥合LLMs理论计算能力与实际规划性能之间的差距。

    

    大语言模型（LLMs）已在机器学习的许多领域产生了显著影响。然而，近期研究表明，它们难以可靠地解决规划问题，尤其是与经典规划器相比时。即使推理导向模型最近取得了进展，LLMs能否真正执行规划仍然是一个悬而未决的问题。与此同时，理论结果表明，作为现代LLMs核心架构的transformer是图灵完备的。在这项工作中，我们研究了LLMs的理论计算能力与其实证规划性能之间这一明显的差距。我们提出了计算链（COC），这是一种计算架构，它将基于transformer的语言模型置于迭代循环中，并利用其作为模式匹配系统的优势。COC使用结构化上下文窗口（SCW）作为外部存储器，使模型只能访问其中的部分内容。

    arXiv:2607.17710v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have had a remarkable impact across many areas of machine learning. However, recent studies have shown that they struggle to reliably solve planning problems, particularly when compared with classical planners. Even with recent advances in reasoning-oriented models, the extent to which LLMs can genuinely perform planning remains an open question. At the same time, theoretical results have shown that transformers, the core architecture underlying modern LLMs, are Turing-complete. In this work, we investigate this apparent gap between the theoretical computational power of LLMs and their empirical planning performance. We propose Chain of Computation (COC), a computational architecture that places a transformer-based LM inside an iterative loop and leverages its strength as a pattern-matching system. COC uses a Structured Context Window (SCW) as external memory, allowing the model to access only the portion
    
[^315]: 公平性研究中的平等、公平与因果性：对Cheng（2026）的评论

    Equality, Equity, and Causality in Fairness Research: A Commentary on Cheng (2026)

    [https://arxiv.org/abs/2607.17679](https://arxiv.org/abs/2607.17679)

    该评论在Cheng（2026）对心理测量学与AI/ML公平性系统比较的基础上，进一步探讨了平等与公平的概念区别以及因果性在公平性研究中的作用，为跨学科公平性研究的未来方向提供了指引。

    

    这是一篇受邀撰写的评论，针对《心理测量学》重点文章《心理测量学与AI/ML中的公平性问题与评估：我们能从彼此领域学到什么？》（作者Ying Cheng，2026）。Cheng对长期存在的测验公平性与现代算法公平性进行了系统性比较。她将整个测验工作流程（而不仅仅是最终的选拔阶段）映射到AI/ML公平性范式上，这是对跨学科公平性研究的一项重要贡献。本评论通过探讨两个概念性问题来扩展她的讨论：平等与公平之间的区别，以及因果性在公平性研究中的作用。这篇重点文章和本评论共同为心理测量学界与AI/ML学界未来的公平性研究指明了方向。

    arXiv:2607.17679v2 Announce Type: replace-cross  Abstract: This is an invited commentary on the Psychometrika focus article "Fairness Issues and Evaluation in Psychometrics and AI/ML: What Can We Learn from Each Field?" by Ying Cheng (2026, doi:10.1017/psy.2026.10110). Cheng offers a systematic comparison between long-standing test fairness and modern algorithmic fairness. Her mapping of the entire testing workflow onto the AI/ML fairness paradigm, rather than only the final selection stage, is a crucial contribution to interdisciplinary fairness research. This commentary extends her discussion by examining two conceptual issues: the distinction between equality and equity, and the role of causality in fairness research. The focus article and this commentary point to directions for future fairness research across the psychometrics and AI/ML communities.
    
[^316]: H-OPD：置信度感知的异构多教师多模态在线蒸馏

    H-OPD: Confidence Aware Heterogeneous Multi-Teacher Multimodal On-policy Distillation

    [https://arxiv.org/abs/2607.02592](https://arxiv.org/abs/2607.02592)

    提出 H-OPD 框架，以置信度感知的 token 级异构多教师仲裁取代传统静态的任务/样本级教师路由，并通过视觉到语言的描述迁移让纯文本教师获取关键视觉语义，从而显著提升多模态推理在线蒸馏的效果。

    

    在线蒸馏近期已成为一种有效的后训练范式，它通过在学生模型生成的轨迹上提供监督信号来实现训练。然而，现有的面向多模态推理的在线蒸馏方法通常依赖于静态的教师路由，即根据模态或任务类型将每个样本分配给单一教师。这种方式忽略了视觉定位和抽象推理可能在不同解码步骤中轮流占主导地位，使得单一教师难以胜任完整的轨迹生成。为此，本文提出了 H-OPD，一种面向多模态推理的置信度感知异构多教师在线蒸馏框架。通过验证异构教师在同一推理过程中的互补性，H-OPD 用沿共享学生轨迹进行的 token 级教师仲裁，取代了任务级或样本级的教师路由。H-OPD 采用视觉到语言的描述迁移技术，使纯文本教师能够获取关键的视觉语义，并使用……（原文摘要在此处截断）

    arXiv:2607.02592v3 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) has recently emerged as an effective post-training paradigm by providing supervision on student-generated trajectories. However, existing OPD methods for multimodal reasoning usually rely on a static teacher routing, assigning each sample to a single teacher based on modality or task type. This ignores that visual grounding and abstract reasoning may dominate different decoding steps, making a single teacher insufficient for the full trajectory. To this end, H-OPD is proposed as a confidence-aware heterogeneous multi-teacher OPD framework for multimodal reasoning. By verifying the complementarity of heterogeneous teachers in the same reasoning process, H-OPD replaces task or sample level teacher routing with token-level teacher arbitration along the shared student trajectory. H-OPD employs vision-to-language description transfer to enable text-only teachers to access key visual semantics, and uses a
    
[^317]: 通过对抗蒸馏提升认证鲁棒性

    Improving Certified Robustness via Adversarial Distillation

    [https://arxiv.org/abs/2606.31653](https://arxiv.org/abs/2606.31653)

    本文提出了结合对抗蒸馏的认证训练方法AD-CERT，通过将对抗训练目标与IBP松弛界相结合，在标准准确率与认证鲁棒性之间取得更好的权衡。

    

    认证训练旨在产生预测结果能够针对对抗扰动进行形式化验证的模型，通常通过在允许的扰动集上优化最坏情况损失的上界来实现。对于神经网络而言，纯粹基于紧致松弛界的认证训练方法所产生的网络虽然易于认证，但会牺牲标准准确率。相反，对抗训练通常能带来更强的经验鲁棒性和标准准确率，但由此产生的模型通常难以用神经网络验证器进行认证。最近有研究表明，通过将对抗训练目标与基于区间界传播（IBP）的松散过近似相结合，可以实现更好的标准准确率与认证准确率之间的权衡，有效地在最坏情况损失的下界与上界之间进行插值。在此基础上，我们提出了AD-CERT，一种认证训练方法（摘要在此处截断）

    arXiv:2606.31653v2 Announce Type: replace-cross  Abstract: Certified training aims to produce models whose predictions can be formally verified against adversarial perturbations, typically by optimising upper bounds on the worst-case loss over an allowed perturbation set. For neural networks, certified training methods based purely on tight relaxation bounds produce networks that are amenable to certification, but sacrifice standard accuracy. Conversely, adversarial training often yields stronger empirical robustness and standard accuracy, but the resulting models are generally difficult to certify with neural network verifiers. Recently, the literature has shown that better standard-certified accuracy trade-offs can be achieved by combining adversarial training objectives with loose over-approximations based on Interval Bound Propagation (IBP), effectively interpolating between lower and upper bounds of the worst-case loss. Building on this, we introduce AD-CERT, a certified training 
    
[^318]: FlowAWR：基于优势加权矫正的在线自适应流强化学习

    FlowAWR: Online Adaptive Flow Reinforcement via Advantage-Weighted Rectification

    [https://arxiv.org/abs/2606.30376](https://arxiv.org/abs/2606.30376)

    FlowAWR提出将连续生成流模型的策略优化重构为朝向理论最优速度场的监督回归范式，通过优势加权矫正实现幅度感知的自适应优化，解决了现有方法中训练-推理不一致及固定幅度校正的局限。

    

    通过在线强化学习在连续空间上对齐生成流模型受到难以处理的轨迹似然的限制。现有的密度近似策略梯度方法依赖随机SDE采样器来构建可处理的转移核，这引入了训练与推理之间的不一致性，并且需要依赖无分类器引导（CFG）。而以DiffusionNFT为代表的隐式框架虽然直接优化前向过程的速度场，但其启发式的固定幅度校正无法使优化强度与组内相对质量相关联。我们提出了流优势加权矫正（FlowAWR），这一范式将连续生成策略优化重新表述为朝向理论最优速度场的监督回归。从KL约束奖励最大化的最优策略出发，FlowAWR推导出一种具有幅度感知、自适应……（原文摘要在此处截断）特性的最优速度场。

    arXiv:2606.30376v2 Announce Type: replace  Abstract: Aligning generative flow models on continuous spaces via online reinforcement learning is constrained by intractable trajectory likelihoods. Existing density-approximated policy gradient methods rely on stochastic SDE samplers to construct tractable transition kernels, which introduce training-inference inconsistencies and necessitates Classifier-Free Guidance (CFG).   While implicit frameworks such as DiffusionNFT directly optimize forward-process velocity fields, its heuristic fixed-magnitude corrections prevent optimization strength from relative intra-group quality.   We propose \textit{Flow Advantage-Weighted Rectification} (\textbf{FlowAWR}), a paradigm that recasts continuous generative policy optimization as supervised regression toward a theoretically optimal velocity field.   Starting from the optimal policy of a KL-constrained reward maximization, FlowAWR derives the optimal velocity field that admits a magnitude-aware, ad
    
[^319]: 分层实验智能体

    Hierarchical Experimentalist Agents

    [https://arxiv.org/abs/2606.29315](https://arxiv.org/abs/2606.29315)

    提出分层实验智能体（HExA），一种以实验为中心的上下文内自我改进框架，使智能体能够通过主动实验收集查询相关证据、发现未知环境的一般规律，并将交互经验转化为可复用技能。

    

    arXiv:2606.29315v2 公告类型：替换 摘要：大语言模型（LLM）越来越多地被用于在现实世界中执行任务和辅助人类决策，然而大多数系统依赖于通过模仿获得的参数化知识，并可选择地通过后训练、检索或在固定数据上的搜索来加以改进。这种范式在 novel 领域或需要先验知识之外信息的复杂查询上表现不佳；例如，仅知道物理定律本身并不能保证在复杂物理系统中进行可靠的推理或长时程行动。我们提出，智能体需要主动实验的能力：即收集针对特定查询的目标性证据、发现未知环境的一般原理，并将交互经验转化为可复用技能。我们提出了分层实验智能体（HExA），这是一个以实验为中心的上下文内自我改进框架，它（1）迭代地设计和完善与查询相关的实验，（2）增量……

    arXiv:2606.29315v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to act in the real world and augment human decision-making, yet most systems rely on parametric knowledge acquired through imitation, optionally improved by post-training, retrieval, or search over fixed data. This paradigm struggles in novel domains or on complex queries that require information unavailable from prior knowledge alone; knowing the laws of physics, for example, does not by itself enable reliable reasoning or long-horizon actions in a complex physical system. We argue that agents need active experimentation: the ability to gather targeted query-specific evidence, discover general principles of unseen environments, and convert interaction experience into reusable skills. We introduce Hierarchical Experimentalist Agents (HExA), an in-context, experiment-centric self-improvement framework that (1) iteratively designs and refines query-relevant experiments, (2) incremental
    
[^320]: 已被抹除，但并未消失：输出层面的遗忘并非真正的遗忘

    Erased, but Not Gone: Output Forgetting Is Not True Forgetting

    [https://arxiv.org/abs/2606.25001](https://arxiv.org/abs/2606.25001)

    该论文提出以重训练模型群体作为参考分布的“重训练一致的表示遗忘”评估范式，揭示现有机器遗忘方法虽在输出层面看似遗忘成功，但在表示空间中仍与理想遗忘存在结构性偏差，即输出遗忘并不等于真正的遗忘。

    

    机器遗忘（MU）通常通过输出层面的遗忘效果来评判，例如遗忘集上的低准确率或成员推断攻击效果的减弱。但如果输出层面的成功可能与表示空间中重训练不一致的残余共存，那么当前的评估究竟在认证何种意义上的“遗忘”？我们通过“重训练一致的表示遗忘”来研究这一问题，使用一组重训练模型（即从头开始训练且不包含待遗忘数据的模型）的总体作为理想遗忘的可操作参考分布。在多种遗忘方法、数据集和模型上，我们的理论分析与实证结果表明，标准的输出层面评估会系统性地高估遗忘的成功程度。在这种更严格的视角下，当前方法往往在输出层表现得像已遗忘，但在表示层面却与重训练结果存在结构性的不匹配。它们在遗忘集上仅部分对齐于重训练……

    arXiv:2606.25001v2 Announce Type: replace-cross  Abstract: Machine unlearning (MU) is commonly judged by output forgetting, such as low forget-set accuracy or reduced membership inference. But if output-level success can coexist with retraining-inconsistent residuals in representation space, what kind of forgetting are current evaluations actually certifying? We study this question through retraining-consistent representation forgetting, using a population of retrained models (i.e., trained from scratch without the forget data) as an operational reference distribution for ideal forgetting. Across multiple unlearning methods, datasets, and models, our theoretical analysis and empirical results show that standard output-level evaluation can systematically overestimate the success of unlearning. Under this stronger lens, current methods often appear forgotten at the output layer while exhibiting a structured mismatch relative to retraining. They partially align with retraining on forget s
    
[^321]: RoboMME-Interference：干扰下的机器人记忆基准测试

    RoboMME-Interference: Benchmarking Robot Memory Under Interference

    [https://arxiv.org/abs/2606.22338](https://arxiv.org/abs/2606.22338)

    本文提出了RoboMME-Interference，一个跨会话机器人记忆基准，用于评估长上下文和干扰下VLA模型的记忆性能，并发现现有感知记忆变体在干扰下表现有限。

    

    arXiv:2606.22338v3 公告类型：替换-交叉 摘要：部署在现实环境中的机器人会在其部署期间跨多个会话和任务积累经验。机器人的任务往往需要它记住多个会话之前的信息，这使得长上下文机器人记忆对于实际部署至关重要。然而，目前大多数机器人记忆基准测试基于单次情节或短上下文。为了衡量当前机器人记忆系统在更长会话和更多干扰下的表现，我们引入了RoboMME-Interference，这是一个基于RoboMME（Dai等人，2026）构建的跨会话基准测试。对于每个查询情节，我们使用查询相关的先前演示构建一个会话历史，随后加入受控数量的无关会话，并将其作为记忆提供给视觉-语言-动作模型（VLA），然后测量准确性。通过在该基准测试上运行RoboMME发布的未经修改的记忆增强π0.5变体，我们发现感知记忆变体虽然有所改进，但成功……（摘要截断）

    arXiv:2606.22338v3 Announce Type: replace-cross  Abstract: Robots deployed in realistic settings will accumulate experience across many sessions and tasks over their deployment. The robot's tasks may often require it to remember information from multiple sessions ago, making long-context robot memory important for real-world deployments. However, most robot-memory benchmarks today are based on single episodes or a short context. To measure how current robot memory systems perform on longer sessions with more distractions, we introduce RoboMME-Interference, a cross-session benchmark built on RoboMME (Dai et al., 2026). For each query episode, we construct a session history using the query's relevant prior demonstration followed by a controlled number of unrelated sessions, which we provide to the VLA as memory and measure accuracy. Running RoboMME's released memory-augmented $\pi_{0.5}$ variants unmodified through this benchmark, we find that while perceptual memory variants improve suc
    
[^322]: 视觉-语言-动作模型中陈述性知识与程序性知识的解耦

    Decoupling the Declarative from the Procedural in Vision-Language-Action Models

    [https://arxiv.org/abs/2606.21496](https://arxiv.org/abs/2606.21496)

    该论文指出现有视觉-语言-动作模型无法将陈述性知识（概念与实体语义）与程序性知识（操作方法）解耦，导致其对空间、语义和任务变化十分脆弱、难以泛化到训练物体之外，这是通用机器人技能迁移的根本瓶颈。

    

    在现实世界中部署通用机器人智能体需要可迁移的技能。具体而言，通过物体特定演示训练来克隆行为的策略必须能够超越该物体进行泛化，否则数据收集需求将变得难以承受。近年来，对预训练的数十亿参数视觉-语言模型（VLM）进行微调——首先在大规模机器人数据集上进行，然后在较少的场景特定演示上进行——已成为设计视觉-语言-动作（VLA）模型的主流范式。虽然这些策略在分布内实现了最先进的操作性能，但它们对微小的空间、语义和任务变化仍然十分脆弱。在这项工作中，我们针对当前模型无法将参数中编码的陈述性知识（即概念和实体语义）与程序性知识（即如何执行某项操作）解耦的问题进行研究，这是……的一个根本性瓶颈。

    arXiv:2606.21496v2 Announce Type: replace-cross  Abstract: Deploying generalist robotic agents in the real world requires transferable skills. Specifically, a policy trained to clone a behavior from object-specific demonstrations must generalize beyond that object, otherwise data collection requirements become intractable. Recently, fine-tuning of pre-trained billion-parameter Vision-Language Models (VLMs), initially on large-scale robot datasets and then on fewer scenario-specific demonstrations, has emerged as the predominant paradigm for designing Vision-Language-Action (VLA) models. While these policies achieve state-of-the-art manipulation performance in-distribution, they remain brittle to minor spatial, semantic, and task variations. In this work, we address the inability of current models to decouple the declarative (i.e., concepts and entity semantics) from the procedural knowledge (i.e., how to do something) encoded in their parameters, which is a fundamental bottleneck for z
    
[^323]: VLA-FAIL：面向微调视觉-语言-动作模型的高效任务失败检测

    VLA-FAIL: Efficient Task Failure Detection for Finetuned Vision-Language-Action Models

    [https://arxiv.org/abs/2606.21386](https://arxiv.org/abs/2606.21386)

    提出VLA-FAIL框架，通过最后一层马氏距离（LLMD）和动作块一致性（ACC）两个轻量级检测器，在无需失败数据和最小开销的情况下高效检测视觉-语言-动作模型的运行时任务失败。

    

    视觉-语言-动作模型（VLA）在许多机器人操作任务上取得了最先进的性能，但在分布外场景中仍可能表现出不可预测的行为。因此，运行时失败检测对于VLA在现实世界中的安全部署至关重要。然而，现有的任务失败检测器要么需要计算成本高昂的动作采样，要么基于限制其适用于VLA的架构假设，要么需要访问失败回放数据。我们提出VLA-FAIL，这是一个轻量级且广泛适用的VLA失败检测框架，它结合了两个新颖的失败检测器，开销极小且无需失败数据。第一个是最后一层马氏距离（LLMD），通过测量最后一层特征中相对于训练数据的逐token偏差来检测分布外状态。第二个是动作块一致性（ACC），利用滚动时域动作分块所带来的时间重叠来检测失败。

    arXiv:2606.21386v2 Announce Type: replace  Abstract: Vision-language-action models (VLAs) achieve state-of-the-art performance on many robotic manipulation tasks, yet they can still behave unpredictably in out-of-distribution scenarios. Runtime failure detection is therefore essential for the safe real-world deployment of VLAs.However, existing task failure detectors require computationally expensive action sampling, are based on architectural assumptions that limit their applicability to VLAs, or need access to failure rollouts. We propose VLA-FAIL, a lightweight and broadly applicable failure detection framework for VLAs that combines two novel failure detectors with minimal overhead, without requiring failure data. The first, last-layer Mahalanobis distance (LLMD), detects out-of-distribution states by measuring token-wise deviations in last-layer features relative to the training data. The second, action chunk consistency (ACC), exploits the temporal overlap induced by receding-hor
    
[^324]: 基于开放数据的城市电网拓扑与层级识别

    Urban Power Grid Topology and Hierarchy Identification from Open Data

    [https://arxiv.org/abs/2606.21352](https://arxiv.org/abs/2606.21352)

    本文提出一个基于开放数据的框架，通过整合公共基础设施数据与地理空间机器学习聚类建筑负荷，实现了从高压输电到单个建筑连接的城市电网完整拓扑与层级识别。

    

    理解城市电网复杂的拓扑结构和层级关系对于能源预测、潮流管理和系统韧性分析至关重要。然而，详细的电网信息在很大程度上仍属于专有数据。尽管如此，这些信息对于分析连接单个建筑物的“最后一公里”配电网络尤为重要。本文通过开发一个基于开放数据的框架来应对这一挑战，以识别从高压输电到单个建筑连接的城市电网拓扑。我们整合公共基础设施数据（电力线、变电站、变压器、电线杆）来绘制高中压电网骨架。随后，我们利用地理空间机器学习技术对OpenStreetMap建筑数据进行处理，对建筑能耗需求的负荷进行聚类。在聚类的基础上，我们构建了连接聚类建筑与低压配电线路的物理拓扑。

    arXiv:2606.21352v2 Announce Type: replace  Abstract: Understanding the complex topology and hierarchy of urban power grid is crucial for energy prognosis, power flow management, and system resilience analysis. However, detailed grid information remains largely proprietary. This information, nevertheless, is crucial especially for analyzing the last-mile distribution networks connecting individual buildings. This paper addresses this challenge by developing an open-data-driven framework to identify urban power grid topology, from high-voltage transmission down to individual building connections. We integrate public infrastructure data (power-lines, substations, transformers, poles) to map the high and medium-voltage grid skeleton. We then leverage geospatial machine learning on OpenStreetMap building data to cluster loads of building energy demand. Upon the clustering, we develop the physical topology of the final distribution lines linking the clustered buildings and low-voltage distri
    
[^325]: TelcoAgent：具有基于3GPP可解释性的可扩展5G多KPM预测

    TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability

    [https://arxiv.org/abs/2606.19821](https://arxiv.org/abs/2606.19821)

    提出TelcoAgent框架，通过自动构建3GPP知识图谱、零样本时间序列基础模型预测和领域可解释推理三个组件，实现了无需站点特定训练的准确、可扩展且可解释的5G多KPM预测。

    

    关键性能测量（KPM）预测对于5G及下一代电信网络的主动式网络管理至关重要。然而，现有的机器学习方法在可扩展性和可解释性方面存在显著局限，限制了其在实际部署中的有效性。我们提出了TelcoAgent，这是一个基于基础模型的框架，无需针对特定站点进行训练，即可对跨多个网络小区的多个KPM实现准确、可扩展且可解释的预测。具体而言，该框架包含三个关键组件：(i) 一个自动化三智能体流水线，可直接从3GPP规范文档构建第三代合作伙伴计划（3GPP）知识图谱；(ii) 一个基于时间序列基础模型（TSFM）的可扩展预测流水线，可提供准确的零样本预测；(iii) 一个推理与解释流水线，可提供可操作的、基于领域知识的解释。

    arXiv:2606.19821v2 Announce Type: replace  Abstract: Key Performance Measurement (KPM) forecasting is essential for proactive network management of 5G and next-generation telecom networks. However, existing machine learning (ML) approaches face significant limitations in scalability and explainability, restricting their effectiveness in real-world deployments. We propose TelcoAgent, a foundation model-based framework that enables accurate, scalable, and explainable forecasting of multiple KPMs across diverse network cells without the need for site-specific training. Specifically, the framework comprises three key components: (i) an automated three-agent pipeline that constructs a 3rd Generation Partnership Project (3GPP) knowledge graph directly from specification documents, (ii) a scalable, time-series foundation model (TSFM)-based prediction pipeline to deliver accurate, zero-shot forecasting, and finally (iii) a reasoning and explanation pipeline that provides actionable, domain-gro
    
[^326]: 构造性双稳态：以墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制

    Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence

    [https://arxiv.org/abs/2606.19386](https://arxiv.org/abs/2606.19386)

    本文勘误并重新审视了先前报告的“状态饱和陷阱”（实为动作间 dt=0 导致指数衰减从未运行的纯累加器结果），揭示出以采样时间校准与以墙钟时间校准的监测器动态在固定速率流上等价、但在动作间隔差异巨大的智能体流上根本不同，因此墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制。

    

    面向自主智能体的运行时监测器通常对累积的内部状态设定阈值——可以是行为基线、漂移统计量，或在我们先前工作中的建模情感状态。我们曾报告过一种“状态饱和陷阱”：在连续情感引擎之上基于状态阈值的触发机制，会在 SWE-bench 调试智能体上变成近乎持续不断的报警（Modgil 2026）。发布后的审计发现，该引擎在动作之间接收到的 dt=0，因此其指数衰减从未运行：先前发表的陷阱实际上是一个纯累加器的结果。我们更正了记录（勘误，v2），并将这一缺陷视为一次实验。它所揭示的关键变量是：监测器的动态是以采样时间（按每次观测，如 CUSUM）校准，还是以墙钟时间（以秒为单位的半衰期，如情感模型和 EMA 基线）校准。在固定速率的数据流上二者重合；而在动作间隔时间相差数个数量级的智能体数据流上，二者并不重合。一项预先注册的……

    arXiv:2606.19386v2 Announce Type: replace-cross  Abstract: Runtime monitors for autonomous agents commonly threshold an accumulated internal state - a behavioural baseline, a drift statistic, or, in our prior work, a modelled affective state. We previously reported a State Saturation Trap: threshold-on-state triggers over a continuous affect engine become near-constant alarms on SWE-bench debugging agents (Modgil 2026). A post-release audit found the engine received dt=0 between actions, so its exponential decay never operated: the published trap is a pure-accumulator result. We correct the record (erratum, v2) and treat the flaw as an experiment. The key variable it exposes is whether a monitor's dynamics are calibrated in sample time (per observation, as in CUSUM) or wall-clock time (half-lives in seconds, as in affect models and EMA baselines). On fixed-rate streams these coincide; on agent streams, where inter-action time varies by orders of magnitude, they do not. A pre-registered
    
[^327]: 面向大语言模型解释的可扩展电路学习方法

    Scalable Circuit Learning for Interpreting Large Language Models

    [https://arxiv.org/abs/2606.16939](https://arxiv.org/abs/2606.16939)

    提出基于稀疏线性回归的CircuitLasso方法，以极低的计算成本学习到与最先进干预方法结构准确性相当的电路，并能高效揭示人类可理解的SAE语义特征在大语言模型中的传播及其对预测的影响。

    

    机制可解释性研究的一个重要方向是在大语言模型的组件之上学习稀疏电路，以揭示这些组件如何共同产生模型行为。然而，原始神经元具有多义性，使得学习到的电路难以解释。稀疏自编码器（SAE）特征缓解了这一问题，但其高维度使得现有的基于干预的电路学习方法在计算上难以承受。我们提出了CircuitLasso，一种基于稀疏线性回归的可扩展电路学习方法。在基准数据上，CircuitLasso恢复的电路在结构准确性上与最先进的基于干预的方法相当，而计算成本仅为后者的一小部分。在可解释性方面，CircuitLasso能够高效地揭示SAE特征之间的关系，展示人类可理解的语义特征如何在模型中传播并影响其预测。最后，我们验证了所学习电路的实用性。

    arXiv:2606.16939v2 Announce Type: replace-cross  Abstract: A prominent research direction in mechanistic interpretability is learning sparse circuits over LLM components to reveal how they jointly produce model behavior. However, raw neurons are polysemantic, making learned circuits hard to interpret. Sparse autoencoder (SAE) features alleviate this, but their high dimensionality makes existing intervention-based circuit learning methods computationally prohibitive. We propose CircuitLasso, a scalable circuit learning approach based on sparse linear regression. CircuitLasso recovers circuits whose structural accuracy matches that of state-of-the-art intervention-based methods on the benchmark data, at a fraction of the computational cost. For interpretability, CircuitLasso efficiently uncovers relationships among SAE features, showing how human-interpretable semantic features propagate through the model and influence its predictions. Finally, we validate the utility of our learned circ
    
[^328]: PACT：面向多轮工具使用智能体的特权轨迹协同训练框架

    PACT: Privileged Trace Co-Training for Multi-Turn Tool-Use Agents

    [https://arxiv.org/abs/2606.16215](https://arxiv.org/abs/2606.16215)

    PACT框架提出将专家轨迹仅作为训练时的优化信号而非推理提示，通过轨迹条件化强化学习代理目标和带退火过程监督的组件感知SFT损失两个互补信号进行协同训练，同时保持推理时仅使用提示词生成，从而兼顾强化学习的推理一致性与监督微调的密集过程监督优势。

    

    多轮工具使用智能体需要在多轮交互过程中进行推理、调用工具并适应观测结果。对此类智能体进行后训练极具挑战性：强化学习虽然与仅提示词的推理设置相匹配，但常常面临奖励稀疏和信用分配能力弱的问题；而基于专家轨迹的监督微调虽然能提供密集的过程监督，却可能过度约束模型，使其拘泥于固定轨迹。为解决这一问题，我们提出了PACT——一个面向多轮工具使用智能体的特权轨迹协同训练框架。其核心思想是仅将专家轨迹用作训练时的优化信号，而非推理时的提示。PACT在生成rollout时保持仅提示词模式，随后通过两个互补的信号利用专家轨迹来引导优化：一是轨迹条件化的强化学习代理目标，它在专家轨迹的上下文中评估仅提示词的rollout；二是组件感知的监督微调损失，它提供退火式的过程监督

    arXiv:2606.16215v2 Announce Type: replace-cross  Abstract: Multi-turn tool-use agents must reason, call tools, and adapt to observations across several interaction turns. Post-training such agents is challenging, as reinforcement learning often suffers from sparse rewards and weak credit assignment despite matching the prompt-only inference setting, while supervised fine-tuning on expert traces provides dense process supervision but can over-constrain the model to fixed trajectories. To tackle this, we propose PACT, a Privileged trAce Co-Training framework for multi-turn tool-use agents. The key idea is to use expert traces only as training-time optimization signals rather than rollout-time hints. PACT keeps rollout generation prompt-only, then uses expert traces to guide optimization through two complementary signals: a trace-conditioned RL surrogate that evaluates prompt-only rollouts under expert-trace context, and a component-aware SFT loss that provides annealed process supervisio
    
[^329]: Lesion-DDPM：面向多发性硬化症MRI合成的病灶增强3D扩散模型

    Lesion-DDPM: Lesion-Enhanced 3D Diffusion for MS MRI Synthesis

    [https://arxiv.org/abs/2606.15457](https://arxiv.org/abs/2606.15457)

    提出Lesion-DDPM，一种结合多层次解剖掩码注入与病灶加权重建损失的3D条件扩散框架，能够在合成多发性硬化症FLAIR MRI时有效保留小而稀疏的病灶。

    

    3D FLAIR MRI被广泛推荐作为多发性硬化症（MS）脑成像的标准MRI序列之一，但公开可用的MS数据集规模仍然相对较小，且在不同扫描仪、采集协议和病灶模式之间存在差异。这种数据稀缺性和变异性阻碍了鲁棒的神经影像机器学习模型的开发，对于旨在合成图像同时保留小而稀疏病灶的生成模型来说尤其具有挑战性。我们提出了Lesion-DDPM，这是一个用于病灶感知FLAIR合成的3D条件扩散框架，它结合了多层次解剖掩码注入与病灶加权重建损失，在强调病灶体素的同时保持整体大脑结构。基于MSLesSeg数据集的一个精选子集，我们将Lesion-DDPM与具有代表性的最先进GAN和扩散模型进行比较，评估了图像生成指标以及下游3D（任务表现）。

    arXiv:2606.15457v2 Announce Type: replace-cross  Abstract: 3D FLAIR MRI is widely recommended as one of the standard MRI sequences for brain imaging in multiple sclerosis (MS), but publicly available MS datasets remain relatively small and vary across scanners, acquisition protocols, and lesion patterns. This scarcity and variability hinder the development of robust neuroimaging machine learning models and are particularly challenging for generative models that aim to synthesize images while preserving small, sparse lesions. We propose Lesion-DDPM, a 3D conditional diffusion framework for lesion-aware FLAIR synthesis that incorporates multi-level anatomical mask injection together with a lesion-weighted reconstruction loss to emphasize lesion voxels while maintaining global brain structure. Using a curated subset of the MSLesSeg dataset, we compare Lesion-DDPM with representative state-of-the-art GAN- and diffusion-based models, assessing both image-generation metrics and downstream 3D
    
[^330]: 学习迁移：神经不变量的Kan扩展

    Learning Transfers: Kan Extensions for Neural Invariants

    [https://arxiv.org/abs/2606.07627](https://arxiv.org/abs/2606.07627)

    本文通过范畴论中的Kan扩展形式化定义了表示迁移所需保持的结构，并提出了一种新的迁移差异度量，以任务改变所强制的不变量为目标进行评分。

    

    arXiv:2606.07627v3 公告类型：替换 摘要：如果表示在任务改变后仍然可用，则称其具有迁移性。标准评估报告目标准确性或数据分布之间的距离，但两者都没有说明表示的哪种结构旨在存续。在这里，我们使这种结构明确且可计算。任务是一个小范畴，任务的改变是一个函子，表示是进入不变量范畴的函子。目标必须展现的结构是源函子沿任务改变得到的左Kan扩展。我们的迁移差异是目标对象上该扩展与观测到的目标不变量之间距离的上确界，因此我们根据任务改变所强制的不变量来评分迁移，而非根据源。我们证明了在逗号范畴、链复形和持久模块中扩展的余核表示。在有限类型的一参数模块上，我们展示了该差异。

    arXiv:2606.07627v3 Announce Type: replace  Abstract: A representation transfers if it stays usable once the task has changed. Standard evaluations report target accuracy or a distance between data distributions, but neither says which structure of the representation is meant to survive. Here we make that structure explicit and computable. A task is a small category, a change of task is a functor, and a representation is a functor into a category of invariants. The structure the target has to exhibit is the left Kan extension of the source functor along the change of task. Our transfer discrepancy is the supremum over target objects of the distance between that extension and the observed target invariant, so we score transfer against the invariant the change of task forces rather than against the source. We prove cokernel presentations of the extension over comma categories, in chain complexes and in persistence modules. On one-parameter modules of finite type we show the discrepancy is
    
[^331]: 智能体的最后考试

    Agents' Last Exam

    [https://arxiv.org/abs/2606.05405](https://arxiv.org/abs/2606.05405)

    本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。

    

    arXiv:2606.05405v3 公告类型：替换 摘要：近年来的人工智能系统已在各类基准测试中取得了优异的成绩，但这些成果并未转化为在众多专业领域中的经济价值意义上的实际部署。我们认为，这一差距在很大程度上是一个评估问题：广泛使用的基准测试缺乏对真实的、具有经济价值的工作流程的持续性性能测量。本文提出了 Agents' Last Exam（ALE），这是一个旨在评估AI智能体在长周期、具有经济价值、结果可验证的真实世界任务上表现的基准。该基准与250多位行业专家合作开发，参考O*NET / SOC 2018（美国联邦职业分类体系）定义了非实体行业，并围绕一个任务分类体系组织，该体系包含55个子领域、归为13个行业集群，涵盖1千多项任务。当前结果显示，最难的任务层级仍远未达到饱和：在主流的运行框架和底层模型配置下，平均……

    arXiv:2606.05405v3 Announce Type: replace  Abstract: Recent AI systems have achieved strong results on a wide range of benchmarks, yet these gains have not translated into economically meaningful deployment across many professional domains. We argue that this gap is largely an evaluation problem: widely used benchmarks lack sustained performance measurement on real and economically valuable workflows. This paper introduces Agents' Last Exam (ALE), a benchmark designed to evaluate AI agents on long horizon, economically valuable, real world tasks with verifiable outcomes. Developed in collaboration with 250+ industry experts, ALE covers non-physical industries defined with reference to O*NET / SOC 2018 (the U.S. federal occupational taxonomy). It is organized around a task taxonomy with 55 sub fields grouped into 13 industry clusters covering 1K+ tasks. Current results show that the hardest tier remains far from saturated: across mainstream harness and backbone configurations, the avera
    
[^332]: 物理学的无形之手：当视频扩散模型知道的比它们展示的更多时

    The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show

    [https://arxiv.org/abs/2606.05328](https://arxiv.org/abs/2606.05328)

    该论文通过逆转扩散采样过程恢复模型内部轨迹，首次证明视频扩散模型内部确实线性编码了物理合理性信息，其可从扩散Transformer状态中以约81.27%的准确率解码出来，超越了专用表征学习基线。

    

    现代视频扩散模型能够生成越来越逼真且时间上连贯的视频，这促使它们被用作候选的世界模拟器。然而，这些模型究竟是在内部编码了物理结构，还是仅仅复现了训练期间见过的运动模式，目前仍不清楚。我们通过在对应于具有已知物理合理性真实视频的潜空间轨迹上探测视频扩散模型来研究这一问题。为了获得这些轨迹，我们通过将学习到的速度场从干净的视频潜表示反向积分至噪声，近似地逆转确定性采样过程，从而获得模型的中间状态和注意力图。利用这些恢复的轨迹，我们证明物理合理性可以从扩散Transformer的状态中被线性解码出来，在IntPhys和InfLevel数据集上达到约81.27%的平均准确率，并优于专用的表征学习基线方法（如……

    arXiv:2606.05328v2 Announce Type: replace-cross  Abstract: Modern video diffusion models generate increasingly realistic and temporally coherent videos, motivating their use as candidate world simulators. Yet it remains unclear whether these models internally encode physical structure, or merely reproduce motion patterns seen during training. We study this question by probing video diffusion models along latent trajectories corresponding to real videos with known physical plausibility. To obtain such trajectories, we approximately invert the deterministic sampling process by integrating the learned velocity field backward from a clean video latent to noise, giving access to the model's intermediate states and attention maps. Using these recovered trajectories, we show that physical plausibility is linearly decodable from diffusion transformer states across IntPhys and InfLevel, reaching around 81.27% average accuracy and outperforming dedicated representation-learning baselines such as
    
[^333]: 教师何时应该移动？自我在线策略蒸馏中的时间耦合与稳定性

    When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation

    [https://arxiv.org/abs/2606.03532](https://arxiv.org/abs/2606.03532)

    本研究发现在自我在线策略蒸馏中，教师更新的隔离期（更新之间完全冻结教师）而非教师年龄是实现稳定学习的关键结构属性，并提出诊断框架揭示了状态遗忘崩溃现象——短程最优的固定更新时间表在长程训练中会因时钟驱动的刷新将漂移的学生一次性不可逆地复制进教师而导致灾难性失败。

    

    自我在线策略蒸馏通过让学生策略与一个源自其自身参数历史的教师进行训练，然而教师的更新时间表——它决定了教师与学生之间的时间耦合——尚未被系统地研究作为稳定性变量。通过在Qwen3-8B上进行受控的时间表扫描，我们确立了隔离期（定义为更新之间教师完全冻结）是实现稳定学习的关键结构属性，而非教师年龄。为了刻画这些底层的训练动态，我们引入了一个包含时间KL结构、刷新冲击和长度尾部风险的诊断框架。该框架进一步揭示了状态遗忘崩溃现象：最优的短程固定时间表在长程训练中会灾难性地失败，因为时钟驱动的刷新可以在单个不可逆的步骤中将瞬态漂移的学生复制到教师中。这一失败

    arXiv:2606.03532v2 Announce Type: replace-cross  Abstract: Self on-policy distillation trains a student policy against a teacher derived from its own parameter history, yet the teacher's update schedule -- which governs the \emph{temporal coupling} between teacher and student -- has not been systematically studied as a stability variable. Through a controlled schedule sweep on Qwen3-8B, we establish that \emph{isolation periods}, defined as complete teacher freezing between updates, are the key structural property enabling stable learning, not teacher age. To characterize these underlying training dynamics, we introduce a diagnostic framework of temporal KL structure, refresh shock, and length-tail risk. This framework further uncovers \emph{state-oblivious collapse}: optimal short-horizon fixed schedules catastrophically fail under long-horizon training because a clock-driven refresh can copy a transiently drifting student into the teacher in a single, irreversible step. This failure 
    
[^334]: 动态医疗治疗强化学习的统一基准

    A Unified Benchmark for Dynamic Medical Treatment Reinforcement Learning

    [https://arxiv.org/abs/2606.01028](https://arxiv.org/abs/2606.01028)

    该论文提出了MedGym——一个基于物理信息神经网络、在连续时间框架下由临床数据构建的可配置动态治疗推荐强化学习基准，用于评估强化学习方法处理不规则测量间隔、个体化治疗反应和测量点间安全性的能力。

    

    医疗治疗推荐给强化学习带来了诸多挑战：患者生理状态在连续时间中演变，测量与干预以不规则的时间间隔进行，且治疗效果在个体间存在显著差异。然而，现有的强化学习建模方式和模拟环境均基于具有固定决策间隔的离散时间马尔可夫决策过程（MDP）。因此，评估强化学习方法能否处理与时间间隔相关的疾病进展、个性化治疗反应以及连续测量点之间的安全性问题仍然十分困难。为填补这一空白，我们提出了MedGym，一个面向动态治疗推荐的基准环境。MedGym在连续时间框架下对患者的纵向演化进行建模，并利用物理信息神经网络从临床数据构建了一个可配置的医疗强化学习基准。由此得到的基准使得离散…（摘要在此处截断）

    arXiv:2606.01028v2 Announce Type: replace  Abstract: Medical treatment recommendation poses several challenges to reinforcement learning (RL): patient physiology evolves in continuous time, measurements and interventions are performed at irregular intervals, and treatment effects vary substantially across individuals. Existing RL formulations and simulated environments, however, are based on discrete-time MDPs with fixed decision intervals. Thus, it remains difficult to evaluate whether RL methods can handle time-interval-dependent disease progression, personalized treatment response, and safety between consecutive measurement points. To address this gap, we introduce MedGym, a benchmark environment for dynamic treatment recommendation. MedGym models longitudinal patient evolution in a continuous-time framework and constructs a configurable medical RL benchmark from clinical data by using Physics-Informed Neural Networks. The resulting benchmark enables direct comparison between discre
    
[^335]: 微调过程中伪相关的无监督识别与消除

    Unsupervised Identification and Removal of Spurious Correlations During Fine-Tuning

    [https://arxiv.org/abs/2605.27676](https://arxiv.org/abs/2605.27676)

    本文证明可以在无监督条件下从朴素LoRA微调的权重中识别出微调引入的伪相关潜在因素，并提出应消除伪相关本身而非潜在因素，以避免破坏预训练模型依赖该因素获得的有效任务信号。

    

    在经过人工筛选的数据集上对预训练语言模型进行微调时，可能会在微调任务与非预期的潜在因素之间产生伪相关——例如失调的人格倾向或政治立场偏向——这些因素是数据筛选过程与任务纠缠在一起的。模型可能会依赖这些伪相关，从而导致偏见和分布外泛化能力下降。我们证明，在对任务复杂度和伪相关性做出合理假设的前提下，这类潜在因素可以在无监督的情况下，从朴素LoRA微调的权重中被识别出来。现有的消除偏见的方法（如激活引导）是在推理阶段或训练过程中从残差流激活中移除已识别的因素。然而，我们认为目标应该是消除伪相关本身，而不是潜在因素本身，因为预训练模型可能依赖该因素来获取真实的任务信号。为此，我们提出……

    arXiv:2605.27676v2 Announce Type: replace-cross  Abstract: Fine-tuning a pretrained language model on a curated dataset can produce spurious correlations between the fine-tuning task and unintended latent factors---such as misaligned personas or political slant---that the curation procedure has entangled with the task. The model can latch onto these spurious correlations, leading to bias and reduced out-of-distribution generalisation. We prove that under reasonable assumptions on task complexity and the spurious correlation, such latent factors can be identified, without supervision, from the weights of a naive LoRA fine-tune. Existing approaches to removing bias, such as activation steering, remove identified factors from residual-stream activations, either at inference or during training. We argue, however, that the goal should be to remove the spurious correlation, not the latent factor itself, as the pretrained model may rely on it for genuine task signal. To enable this, we propos
    
[^336]: TabPFN是保险定价的灵丹妙药吗？

    Is TabPFN the Silver Bullet for Insurance Pricing?

    [https://arxiv.org/abs/2605.22892](https://arxiv.org/abs/2605.22892)

    本文首次实证评估了TabPFN在机动车保险定价中的表现，发现它并非“灵丹妙药”——在索赔频率预测上落后于GLM且表现依赖上下文训练集规模，仅在索赔强度预测上具有竞争力。

    

    非寿险定价中的索赔频率与索赔强度建模主要依赖于广义线性模型（GLM），而梯度提升机则是领先的机器学习替代方案。表格基础模型（TFM）提供了一种根本不同的推断范式：通过在大量合成数据集上进行预训练，TFM能够通过上下文学习（in-context learning）对新数据进行推断，无需任何针对特定数据集的预处理、拟合或超参数调优。本文首次对TabPFN在机动车保险定价中的表现进行了实证评估，在两个公开可用的机动车第三方责任险（MTPL）数据集上，将两个版本的TabPFN与GLM和XGBoost进行了基准比较。结果表明，TabPFN并不能始终如一地超越既有基线：在索赔频率预测上它落后于GLM，且其表现依赖于上下文训练集的规模；但在索赔强度预测上具有竞争力。

    arXiv:2605.22892v3 Announce Type: replace-cross  Abstract: Modelling claim frequency and severity for non-life insurance pricing predominantly relies on generalised linear models (GLMs), with gradient-boosted machines as the leading machine learning alternative. Tabular foundation models (TFMs) present a fundamentally different inference paradigm. By pre-training on large collections of synthetic datasets, TFMs enable inference on new data through in-context learning, without any dataset-specific preprocessing, fitting, or hyperparameter tuning. This paper presents a first empirical evaluation of TabPFN for motor insurance pricing, benchmarking two versions of TabPFN against GLM and XGBoost on two publicly available motor third-party liability (MTPL) datasets. Our results show that TabPFN does not consistently outperform established baselines: it falls behind the GLM on claim frequency, where its performance depends on the size of the in-context training set, but it is competitive on c
    
[^337]: 深度集成真的能捕获图神经网络中的不确定性吗？

    Do Deep Ensembles Actually Capture Uncertainty in Graph Neural Networks?

    [https://arxiv.org/abs/2605.22593](https://arxiv.org/abs/2605.22593)

    研究发现深度集成在图神经网络中并不能有效捕获认知不确定性，其边际改进仅来自平均化优化噪声，而独立训练的网络因“认知坍缩”现象收敛到过于相似的预测，缺乏集成方法所需的多样性。

    

    尽管深度集成被广泛认为是深度学习中不确定性量化的默认方法，但其在图结构数据上的有效性往往仅是基于计算机视觉等领域取得的成功而简单假定的。我们专门针对消息传递图神经网络研究了标准深度集成方法。通过在八个代表不同任务和复杂度的数据集上进行基准测试，我们发现集成方法相较于单一模型仅能带来边际改进。这些边际收益主要源于对点预测中优化噪声的平均化消除，而非来自实质上更好的不确定性估计。通过偶然不确定性-认知不确定性的分解分析，我们识别出一种“认知坍缩”现象：独立训练的网络总是收敛到过于相似的预测。由于分歧是集成方法捕获认知不确定性的基本机制，这种多样性的缺失使得该方法失效（摘要在此处被截断）。

    arXiv:2605.22593v2 Announce Type: replace  Abstract: While deep ensembles are widely considered to be the default method for uncertainty quantification in deep learning, their effectiveness for graph-structured data is often simply assumed based on successes in domains like computer vision. We investigate standard deep ensembles specifically for message-passing graph neural networks. Benchmarking across eight datasets representing varied tasks and complexities, we reveal that ensembles provide only marginal improvement over a single model. These marginal gains stem primarily from averaging out optimization noise in point predictions rather than from meaningfully better uncertainty estimates. Through an aleatoric-epistemic decomposition, we identify epistemic collapse: independently trained networks consistently converge to overly similar predictions. Because disagreement is the fundamental mechanism through which ensembles capture epistemic uncertainty, this lack of diversity neutraliz
    
[^338]: 随机MeanFlow策略：基于熵镜像下降的单步生成式控制

    Stochastic MeanFlow Policies: One-Step Generative Control with Entropic Mirror Descent

    [https://arxiv.org/abs/2605.21282](https://arxiv.org/abs/2605.21282)

    SMFP是一种将MeanFlow噪声到动作映射与高斯重参数化相结合的单步生成式策略，通过解析熵下界实现熵正则化镜像下降强化学习，在七个MuJoCo基准上优于高斯策略和其他生成式策略。

    

    在在线离策略强化学习中，将熵正则化与镜像下降相结合，需要既具备高效采样能力又能显式控制熵的表达性策略。高斯策略难以表示多模态目标，而生成式策略通常需要迭代采样或缺乏可处理的熵估计。我们提出了随机MeanFlow策略，这是一种单步生成式策略类别，它将基于MeanFlow的噪声到动作映射与高斯重参数化相结合。潜在映射能够捕捉多样化的动作模式，而可学习的条件高斯尺度则提供了具有解析形式的熵下界。我们利用该下界来调控随机性，并结合直接价值引导和优势加权的MeanFlow回归，从而实现基于单步策略的熵正则化镜像下降学习。在七个MuJoCo基准测试中，SMFP相较于高斯策略和生成式策略取得了强劲的性能表现。

    arXiv:2605.21282v3 Announce Type: replace-cross  Abstract: Combining entropy regularisation with mirror descent in online off-policy reinforcement learning requires expressive policies with efficient sampling and explicit entropy control. Gaussian policies struggle to represent multimodal targets, whereas generative policies often require iterative sampling or lack tractable entropy estimates. We introduce Stochastic MeanFlow Policies (SMFP), a one-step generative policy class that combines MeanFlow-based noise-to-action mappings with Gaussian reparameterisation. The latent mapping captures diverse action modes, while a learnable conditional Gaussian scale provides an analytic entropy lower bound. We use this bound to regulate stochasticity alongside direct value guidance and advantage-weighted MeanFlow regression, enabling entropy-regularised mirror-descent learning with one-step policies. Across seven MuJoCo benchmarks, SMFP achieves strong performance against Gaussian and generative
    
[^339]: SURF：调控标量化权重以均匀遍历帕累托前沿

    SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front

    [https://arxiv.org/abs/2605.20619](https://arxiv.org/abs/2605.20619)

    该论文提出SURF方法，通过将权重变化时解在帕累托前沿上的移动速度积分得到的累积弧长归一化为CDF，并利用其反函数调控标量化权重的采样，从而实现对帕累托前沿的均匀覆盖。

    

    标量化方法因其简单性和可扩展性在多目标优化中被广泛使用。在许多应用中，目标是生成能够代表多样化用户偏好的解，理想情况下这些解应均匀地覆盖帕累托前沿（PF）。然而，对标量化权重进行均匀采样通常会导致帕累托前沿的非均匀覆盖。我们通过标量化路径的几何结构解释了这种不匹配现象：随着权重的变化，相应的解通常以不均匀的速度遍历帕累托前沿。对该速度进行积分可得累积弧长，经总弧长归一化后，它定义了一个关于权重的累积分布函数（CDF），其反函数可用于选取等弧长间隔的点。基于这一见解，我们提出了SURF（沿帕累托前沿均匀采样）方法。对于包括双目标老虎机在内的结构化问题，我们推导了该CDF映射的解析表达式以及由此产生的……（摘要被截断）

    arXiv:2605.20619v2 Announce Type: replace  Abstract: Scalarization is widely used in multi-objective optimization owing to its simplicity and scalability. In many applications, the goal is to generate solutions that represent diverse user preferences, ideally with uniform coverage of the Pareto front (PF). However, uniformly sampling scalarization weights usually induces non-uniform coverage of the PF. We explain this mismatch through the geometry of the scalarization path. As the weight varies, the corresponding solutions traverse the PF at a generally nonuniform speed. Integrating this speed gives cumulative arc length. After normalization by the total length, it defines a cumulative distribution function (CDF) on weights whose inverse selects points at equal arc-length intervals. Building on this insight, we propose SURF (Sampling Uniformly along the Pareto Front). For structured problems, including bi-objective bandits, we derive analytical expressions for this CDF map and the resu
    
[^340]: 面向神经网络可达性分析的二次型刻画方法

    Quadratic Characterizations for Reachability Analysis of Neural Networks

    [https://arxiv.org/abs/2605.20482](https://arxiv.org/abs/2605.20482)

    本文提出一个通过凸二次规划局部生成候选二次不等式、再用平方和证书全局验证的框架，为二维平面上的标量关系构建可靠过近似的二次刻画，可直接应用于基于二次约束和积分二次约束的神经网络可达性分析。

    

    二次约束被广泛用于刻画非线性特性与不确定性，但通用的解析刻画在有限域上可能过于保守。本文提出了一个用于构建二维实平面上标量关系的经验证的二次型刻画框架。候选二次不等式通过求解凸二次规划在局部生成，利用了来自该关系的样本点以及外部样本点；随后借助平方和证书，在精确半代数描述上进行全局验证，对于非多项式关系，则在松弛的多项式描述上进行验证。由此得到的经验证约束在所考虑的域上定义了标量关系的可靠过近似。这些约束与现有的基于二次约束和逐点积分二次约束的静态非线性分析框架直接兼容。

    arXiv:2605.20482v2 Announce Type: replace  Abstract: Quadratic constraints (QCs) are widely used to characterize nonlinearities and uncertainties, but generic analytical characterizations can be conservative on bounded domains. This paper develops a framework for constructing verified quadratic characterizations of scalar relations in the two-dimensional real plane. Candidate quadratic inequalities are locally generated by solving convex quadratic programs using samples from the relation and exterior sample points. They are then verified globally using sum-of-squares certificates over an exact semialgebraic description or, in the case of nonpolynomial relations, over relaxed polynomial descriptions. The resulting verified constraints define a sound overapproximation of the scalar relations over the considered domains. These constraints are directly compatible with existing analysis frameworks based on QCs and pointwise integral quadratic constraints (IQCs) for static nonlinearities and
    
[^341]: 跨视图注意力融合网络：一种用于从短期PPG信号估计心输出量的先验引导双视图表示学习

    Cross-View Attention Fusion Net: A Prior-Guided Dual-View Representation Learning for Cardiac Output Estimation from Short-Term PPG Signals

    [https://arxiv.org/abs/2605.19666](https://arxiv.org/abs/2605.19666)

    提出跨视图注意力融合网络CVAF-Net，通过跨视图注意力将原始PPG的时间视图与特征序列图的先验引导视图相融合，实现从短期PPG信号准确估计心输出量。

    

    基于光电容积脉搏波（PPG）准确估计心输出量（CO）对于无创血流动力学监测具有重要意义，但由于心输出量由心脏功能和血管张力共同决定，该任务仍然具有挑战性。传统的基于特征的模型使用具有生理意义的PPG描述符，但依赖于准确的脉搏检测，且可能遗漏潜在的时间关系。相比之下，完全端到端的深度学习模型直接从原始PPG中学习，但往往未能充分利用已有的PPG衍生先验信息。本文提出了跨视图注意力融合网络（CVAF-Net），这是一种先验引导的双视图深度学习模型，用于从短的固定长度PPG片段中估计心输出量。CVAF-Net将原始PPG作为时间视图处理，将特征序列图（FSM）作为结构化的先验引导视图，并通过跨视图注意力机制融合这两种表示。该模型分别使用5秒、15秒和30秒的片段进行了独立评估。

    arXiv:2605.19666v2 Announce Type: replace-cross  Abstract: Accurate cardiac output (CO) estimation from photoplethysmography (PPG) is promising for unobtrusive hemodynamic monitoring, but remains difficult since CO is jointly determined by cardiac function and vascular tone. Conventional feature-based models use physiologically meaningful PPG descriptors, yet depend on accurate pulse detection and may miss latent temporal relationships. In contrast, fully end-to-end deep learning models learn directly from raw PPG but often underuse established PPG-derived prior information. Here, we introduce the Cross-View Attention Fusion Network (CVAF-Net), a prior-guided dual-view deep learning model for CO estimation from short, fixed-length PPG segments. CVAF-Net processes raw PPG as a temporal view and a feature sequence map (FSM) as a structured prior-guided view, and fuses the two representations through cross-view attention. The model was independently evaluated using 5-, 15-, and 30-s segme
    
[^342]: 基于采样的安全强化学习

    Sampling-Based Safe Reinforcement Learning

    [https://arxiv.org/abs/2605.19469](https://arxiv.org/abs/2605.19469)

    该论文提出基于采样的安全强化学习算法SBSRL，通过在有限动力学样本上联合施加约束来近似最坏情况优化，并结合认知不确定性约束的探索策略，提供了学习全过程的高概率安全保证和有限时间样本复杂度界，在仿真与真实机器人上实现了安全高效的探索。

    

    安全探索仍然是强化学习（RL）中的一个根本性挑战，它限制了RL智能体在现实世界中的部署。我们提出了基于采样的安全强化学习（SBSRL），这是一种基于模型的RL算法，通过在有限的动力学样本集合上联合实施约束，在整个学习过程中保持安全性。这一表述方式近似了针对不确定动力学的难以处理的最坏情况优化，并使连续域中的实用安全保证成为可能。我们进一步引入了一种基于约束认知不确定性的探索策略，从而无需显式的探索奖励。在正则性条件下，我们推导出了整个学习过程中安全性的高概率保证，以及恢复近最优策略的有限时间样本复杂度界。实验表明，SBSRL在仿真和真实机器人硬件上都实现了安全且高效的探索。

    arXiv:2605.19469v2 Announce Type: replace-cross  Abstract: Safe exploration remains a fundamental challenge in reinforcement learning (RL), limiting the deployment of RL agents in the real world. We propose Sampling-Based Safe Reinforcement Learning (SBSRL), a model-based RL algorithm that maintains safety throughout the learning process by enforcing constraints jointly across a finite set of dynamics samples. This formulation approximates an intractable worst-case optimization over uncertain dynamics and enables practical safety guarantees in continuous domains. We further introduce an exploration strategy based on constraining epistemic uncertainty, eliminating the need for explicit exploration bonuses. Under regularity conditions, we derive high-probability guarantees of safety throughout learning and a finite-time sample complexity bound for recovering a near-optimal policy. Empirically, SBSRL achieves safe and efficient exploration both in simulation and in real robotic hardware, 
    
[^343]: 基于混合动量Frank--Wolfe的随机复合优化

    Stochastic Compositional Optimization via Hybrid Momentum Frank--Wolfe

    [https://arxiv.org/abs/2605.15350](https://arxiv.org/abs/2605.15350)

    该论文提出了混合动量随机Frank--Wolfe这一无投影算法，首次系统性地处理不可微的凸外层函数（如最大值、CVaR和范数）的随机复合优化问题，在有限方差噪声下对非凸目标达到K^{-1/4}收敛率、在模型下界条件下达到K^{-1/3}的末次迭代误差，并将分析扩展到无需截断的重尾噪声情形。

    

    随机复合优化研究在紧凸集上最小化 F(f(x), x)，其中内层映射 f 只能通过带噪样本获得。我们提出混合动量随机Frank--Wolfe方法，这是一种无投影方法，适用于联合凸且Lipschitz连续但未必可微的外层函数 F，例如最大值、CVaR（条件风险价值）和范数。该方法利用动量跟踪 f 的雅可比矩阵与函数值（对函数值采用泰勒修正），并将跟踪到的线性化模型传递给广义线性极小化预言机。在有限方差噪声下，对于非凸目标，该方法可达到阶为 K^{-1/4} 的期望广义Frank--Wolfe间隙；当线性化模型构成目标的下界时，可达到阶为 K^{-1/3} 的末次迭代原始误差。该分析还可扩展到重尾噪声情形而无需梯度截断，且对函数值与雅可比矩阵分别采用不同的矩阶数要求。当使用精确预言机时，该方法可退化为……（摘要原文在此处截断）

    arXiv:2605.15350v2 Announce Type: replace-cross  Abstract: Stochastic compositional optimization minimizes $F(f(x), x)$ over a compact convex set when the inner map $f$ is available only through noisy samples. We propose Hybrid Momentum Stochastic Frank--Wolfe, a projection-free method for outer functions $F$ that are jointly convex and Lipschitz but not necessarily differentiable, such as maxima, CVaR, and norms. The method tracks the Jacobian and the value of $f$ with momentum, using a Taylor correction for the value, and passes the tracked linearization to a generalized linear minimization oracle. Under finite-variance noise, it attains an expected generalized Frank--Wolfe gap of order $K^{-1/4}$ for non-convex objectives and, when the linearized models minorize the objective, a last-iterate primal error of order $K^{-1/3}$. The analysis extends to heavy-tailed noise without clipping, with separate moment orders for values and Jacobians. With an exact oracle, the method reduces to t
    
[^344]: 基于高斯过程的贝叶斯非参数混合效应常微分方程模型

    Bayesian Nonparametric Mixed-Effect ODEs with Gaussian Processes

    [https://arxiv.org/abs/2605.13088](https://arxiv.org/abs/2605.13088)

    提出MEGPODE模型，利用高斯过程先验将每个受试者的ODE向量场分解为共享种群成分与个体偏差，并通过状态空间高斯过程和单一变分目标避免重复求解ODE，从而在稀疏不规则观测下实现灵活且稳健的非参数动力学建模。

    

    药物计量学、系统生物学和流行病学中的动力学建模通常涉及具有相关但又各不相同动力学的受试者，且观测数据在时间上稀疏且不规则。经典的非线性混合效应常微分方程（ODE）模型能够表示这种异质性，但依赖于参数化的向量场，因而容易受到结构性误设的影响。现有的非参数ODE方法通常假设动力学是共享的，而非显式的混合效应层次结构。我们提出了MEGPODE，这是一种贝叶斯非参数混合效应ODE模型，它将每个受试者的向量场分解为共享的种群成分和受试者特异的偏差，两者均赋予高斯过程先验。为避免训练过程中重复求解ODE，我们使用状态空间高斯过程先验来表示潜在轨迹，并在有限时间网格上惩罚其导数与向量场之间的差异。单一的变分目标使得（摘要在此处被截断）

    arXiv:2605.13088v2 Announce Type: replace  Abstract: Dynamical modelling in pharmacometrics, systems biology, and epidemiology often involves subjects with related but distinct dynamics, observed through temporally sparse and irregular measurements. Classical nonlinear mixed-effects ODE models represent this heterogeneity but rely on parametric vector fields, making them vulnerable to structural misspecification. Existing nonparametric ODE methods typically assume shared dynamics rather than an explicit mixed-effect hierarchy. We propose MEGPODE, a Bayesian nonparametric mixed-effect ODE model that decomposes each subject's vector field into a shared population component and a subject-specific deviation, both endowed with Gaussian process priors. To avoid repeated ODE solves during training, we represent latent trajectories with state-space GP priors and penalize discrepancies between their derivatives and the vector fields on a finite time grid. A single variational objective enables 
    
[^345]: 关于矩阵乘积算符玻恩机器（MPO Born Machines）的近似复杂性

    On the Approximation Complexity of Matrix Product Operator (MPO) Born Machines

    [https://arxiv.org/abs/2605.11471](https://arxiv.org/abs/2605.11471)

    本文证明了用多项式键维MPO玻恩机器近似任意连续目标是NP难的，但同时表明当损失诱导哈密顿量局部且具有常数谱隙时，满足对数Sobolev不等式的目标可获得带可证明KL保证的多项式键维MPO-BM近似，且多项式数量的局部分数查询足以维持这些保证。

    

    矩阵乘积算符玻恩机器（MPO-BMs）是一类易于处理的用于概率建模的张量网络模型，然而其近似能力在理论上仍缺乏深入理解。本文从负面和正面两个角度刻画了这些近似能力。作为负面结果，我们证明了为任意连续目标构造具有有界KL误差的多项式键维MPO-BM近似是NP难的。在正面结果方面，对于基于分数的变分推断，我们证明了当损失诱导的哈密顿量是局部的且具有常数谱隙时，满足对数Sobolev不等式的目标容许具有可证明KL保证的多项式键维MPO-BM近似。此外，在相同条件下，我们构造了局部重要性采样估计器，并证明了多项式数量的局部分数查询足以保持KL保证。

    arXiv:2605.11471v2 Announce Type: replace  Abstract: Matrix product operator Born machines (MPO-BMs) are tractable tensor-network models for probabilistic modeling, yet their approximation capabilities remain poorly understood in theory. In this paper, we characterize these capabilities from both negative and positive perspectives. As a negative result, we prove that constructing a polynomial-bond MPO-BM approximation to an arbitrary continuous target with bounded KL error is NP-hard. On the positive side, for score-based variational inference, we prove that targets satisfying a log-Sobolev inequality admit polynomial-bond MPO-BM approximations with provable KL guarantees when the loss-induced Hamiltonian is local and has a constant spectral gap. Furthermore, under the same conditions, we construct local importance-sampling estimators and prove that polynomially many local score queries suffice to preserve the KL guarantees.
    
[^346]: Vision2Code：用于评估图像到代码生成的多领域基准测试

    Vision2Code: A Multi-Domain Benchmark for Evaluating Image-to-Code Generation

    [https://arxiv.org/abs/2605.11307](https://arxiv.org/abs/2605.11307)

    提出了Vision2Code——一个覆盖15个数据集共539个样例、无需参考源代码的多领域图像到代码生成基准，通过执行生成代码并结合领域定制评分标准进行评估，其评分与人类评分的相关性显著优于通用评估指标。

    

    我们研究视觉语言模型（VLM）能否通过编写可执行代码来重构结构化的、基于图形的图像。现有的图像到代码生成基准通常在覆盖领域上存在局限，依赖参考源代码进行评估，且缺乏量化领域特定问题的指标。为解决这些问题，我们推出了Vision2Code——一个多领域、无需参考代码的基准，包含来自15个不同数据集的539个测试样例，涵盖图表与绘图、几何、图结构、科学示意图、文档以及3D空间场景。评估方式是执行生成的程序，并使用针对数据集和领域定制的评分标准由VLM充当评判者，将其渲染输出与源图像进行比较。我们发现，通过这种方式获得的最终评分与人类评分的相关性显著高于使用通用VLM评分标准或标准像素级指标所获得的评分。

    arXiv:2605.11307v2 Announce Type: replace-cross  Abstract: We study whether vision-language models (VLMs) can reconstruct structured, graphics-based images by writing executable code to generate them. Existing benchmarks on image-to-code generation are generally limited in the domains they cover, rely on reference source code for evaluation, and do not provide metrics to quantify domain-specific issues. To address this, we introduce Vision2Code, a multi-domain, reference-code-free benchmark containing 539 test examples from 15 different datasets, spanning charts and plots, geometry, graphs, scientific diagrams, documents and 3D spatial scenes. Evaluation is done by executing the generated programs and comparing their rendered outputs to the source images using dataset- and domain-specific rubrics for a VLM judge. We find that the final scores obtained in this manner have substantially higher correlation with human ratings than those obtained via generic VLM rubrics or standard pixel-ba
    
[^347]: 模型容量通过记忆化与泛化速度的竞争决定顿悟现象

    Model Capacity Determines Grokking through Competing Memorisation and Generalisation Speeds

    [https://arxiv.org/abs/2605.09724](https://arxiv.org/abs/2605.09724)

    该论文提出一个信息论解释框架，指出顿悟现象并非在模型容量足以记住训练集时立即发生，而是由记忆化速度与泛化速度这两个随参数量变化的可测量时间尺度的相对竞争所决定。

    

    现有关于顿悟（grokking）现象的解释大多基于电路效率或从懒惰到富饶转变等机制性框架。然而，尽管已知顿悟与模型规模之间存在依赖关系，模型容量如何塑造顿悟仍是一个悬而未决的问题。我们在模运算任务上对这一关系给出了信息论层面的解释，表明当模型变得足够大以至于能够记住训练集时，顿悟并不会立即出现，而是作为两个可测量时间尺度之间竞争的结果而涌现：记忆化速度 $T_{\text{mem}}(P)$ 与泛化速度 $T_{\text{gen}}(P)$，二者均为模型参数量 $P$ 的函数。借助并改进 Morris 等人（2025）的信息容量框架，我们在同等复杂度的随机标签数据上估计 $T_{\text{mem}}(P)$，在模运算任务本身上估计 $T_{\text{gen}}(P)$，并表明顿悟在……时涌现（原文在此处截断）。

    arXiv:2605.09724v2 Announce Type: replace  Abstract: Existing accounts of grokking explain the phenomena in terms of mechanistic frameworks such as circuit efficiency or lazy-to-rich transitions. However, despite a known dependence between grokking and model size, how model capacity shapes grokking remains an open question. We give an information-theoretic account of this relationship on the task of modular arithmetic, showing that grokking does not immediately occur when a model becomes large enough to memorise the training set, but rather emerges as the outcome of a competition between two measurable timescales: a memorisation speed $T_{\text{mem}}(P)$ and a generalisation speed $T_{\text{gen}}(P)$, both of which are functions of model parameter count $P$. Adapting the information capacity framework of Morris et al. (2025), we estimate $T_{\text{mem}}(P)$ on random-label data of equivalent complexity and $T_{\text{gen}}(P)$ on the modular task itself, and show that grokking emerges c
    
[^348]: FactoryNet：面向工业时间序列基础模型的大规模数据集

    FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models

    [https://arxiv.org/abs/2605.09081](https://arxiv.org/abs/2605.09081)

    FactoryNet是一个包含1.13亿数据点、覆盖7种机器形态和4项任务的大规模工业时间序列预训练数据集，通过统一的S-E-F-C控制理论模式使指令意图与实际响应可在不同机器间比较，为工业时间序列基础模型奠定了基础。

    

    工业机器是受驱动的系统，然而大多数开放的时间序列语料库仅记录传感器的输出结果；少数记录了指令信号的语料库则以供应商特定的格式呈现，且不同机器之间各不相同，因此指令意图与测量响应无法在不同机器形态之间进行比较。我们提出了FactoryNet，一个面向工业时间序列的预训练语料库：包含1.13亿个数据点，涵盖7种机器形态、4项任务共5.7万次端到端任务执行（其中1.5万次为真实数据，4.2万次为合成数据），并附带27种标注的异常类型，以及健康基线和反事实配对数据。每个来源中的每个信号都被映射到一个统一的控制理论模式——设定值、作用量、反馈、上下文，该模式以共同的表征框架表达任何受驱动系统，并使指令与实际执行之间的动态差异可以作为显式的预测残差被读取。利用该语料库，我们发现一个仅在仿真数据上训练的任务阶段模型，在没有……

    arXiv:2605.09081v5 Announce Type: replace-cross  Abstract: Industrial machines are actuated systems, yet most open time-series corpora record sensor outcomes alone; the few that do log commanded signals expose them in vendor-specific forms that differ from machine to machine, so commanded intent and measured response cannot be compared across embodiments. We introduce FactoryNet, a pretraining corpus for industrial time-series: 113M datapoints across 57k end-to-end task executions (15k real, 42k synthetic) on seven embodiments and four tasks, with 27 annotated anomaly types alongside healthy baselines and counterfactual pairs. Every signal in every source is mapped into a single control-theoretic schema, Setpoint, Effort, Feedback, Context (S-E-F-C), which expresses any actuated system in a common representational frame and makes commanded-versus-realized dynamics readable as explicit prediction residuals. Using the corpus, we find that a task-phase model trained on simulation, with no
    
[^349]: UxSID：面向超长序列的语义感知用户兴趣建模

    UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence

    [https://arxiv.org/abs/2605.09040](https://arxiv.org/abs/2605.09040)

    UxSID 通过语义 ID 与双层注意力机制构建语义组共享兴趣记忆，为超长用户行为序列建模提供兼顾效率与效果的第三条路径，实现 SOTA 性能并在大规模广告 A/B 测试中带来 0.337% 的收入提升。

    

    对超长用户行为序列进行建模需要在效率与效果之间进行艰难的权衡。现有范式要么依赖特定物品的检索，要么采用与物品无关的压缩方式，我们提出 UxSID 框架，探索第三条路径：语义组共享兴趣记忆。通过利用语义 ID（SIDs）和双层注意力策略，UxSID 无需承担特定物品模型的沉重开销，即可捕捉目标感知的用户偏好。这一端到端架构在计算简洁性与语义感知之间取得平衡，实现了最先进的性能，并在大规模广告 A/B 测试中带来 0.337% 的收入提升。

    arXiv:2605.09040v4 Announce Type: replace  Abstract: Modeling ultra-long user sequences involves a difficult trade-off between efficiency and effectiveness. While current paradigms rely on either item-specific search or item-agnostic compression, we propose UxSID, a framework exploring a third path: semantic-group shared interest memory. By utilizing Semantic IDs (SIDs) and a dual-level attention strategy, UxSID captures target-aware preferences without the heavy cost of item-specific models. This end-to-end architecture balances computational parsimony with semantic awareness, achieving state-of-the-art performance and a 0.337% revenue lift in large-scale advertising A/B test.
    
[^350]: 面向逆向轮廓设计与紧凑平行四边形四联剪纸原型快速激光切割的强化学习方法

    Reinforcement learning for inverse silhouette design and rapid laser cutting of compact parallelogram quad kirigami prototypes

    [https://arxiv.org/abs/2605.08098](https://arxiv.org/abs/2605.08098)

    本文提出RL-Kirigami框架，将最优传输条件流匹配与强化学习（GRPO）相结合，实现紧凑平行四边形四联剪纸的逆向轮廓设计，单个样本即达到94.2%的sIoU并优于求解器基线，可支持快速激光切割制作原型。

    

    剪纸是一种日益实用的制造方法，可用于生产形状可编程的超材料结构。然而，逆向设计仍然困难，原因在于展开过程是非线性的，且可行的切割布局必须满足离散的兼容性规则、避免重叠，并将目标形状映射到有效设计。我们提出了RL-Kirigami，这是一个逆向设计框架，它将最优传输条件流匹配（OT-CFM）与强化学习相结合，为紧凑可重构的平行四边形四联剪纸生成兼容的比率场。行进解码器强制执行全局几何兼容性，群体相对策略优化（GRPO）则使生成器与轮廓匹配、可行性以及比率场正则性等不可微奖励对齐。在程序化生成的目标形状实例中，来自预训练OT-CFM先验的单个样本即达到94.2%的sIoU，优于求解器基线方法，同时……

    arXiv:2605.08098v2 Announce Type: replace  Abstract: Kirigami is an increasingly useful fabrication method to produce shape-programmable metamaterial structures. However, inverse design remains difficult because deployment is nonlinear, and feasible cut layouts must satisfy discrete compatibility rules, avoid overlap, and map one target shape to valid designs. We present RL-Kirigami, an inverse design framework that combines optimal-transport conditional flow matching (OT-CFM) with reinforcement learning to generate compatible ratio fields for compact reconfigurable parallelogram quad kirigami. A marching decoder enforces global geometric compatibility, and Group Relative Policy Optimization (GRPO) aligns the generator with nondifferentiable rewards for silhouette matching, feasibility, and ratio-field regularity. Across procedurally generated target shape instances, a single sample from the pretrained OT-CFM prior reached $94.2\%~\mathrm{sIoU}$ and outperformed solver baselines while 
    
[^351]: 超越成对比较：你的语言模型正在暗中优化一张偏好图

    Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph

    [https://arxiv.org/abs/2605.08037](https://arxiv.org/abs/2605.08037)

    本文提出 GraphDPO，将 DPO 从成对偏好比较推广到有向无环偏好图结构，充分利用多个 rollout 之间的传递性偏好关系进行聚合监督，从而实现更稳定、更充分的语言模型对齐。

    

    直接偏好优化（DPO）通过成对偏好比较来对齐语言模型，为基于人类反馈的强化学习（RLHF）提供了一种简单有效的替代方案。然而，在许多实际场景中，训练数据由每个提示的多个 rollout 组成，这蕴含了丰富的偏好结构，而成对 DPO 未能充分利用这些结构。将此类数据压缩为独立配对会丢弃传递性，引入冗余或冲突的监督信号，并可能导致优化不稳定。我们提出了图直接偏好优化（GraphDPO），这是 DPO 的一种结构化推广方法，作用于由 rollout 排名诱导的有向无环偏好图。GraphDPO 将支配关系编码为图中的边，并优化一种受 Plackett-Luce 启发的图结构化目标函数，该函数在 dominated 邻域上聚合监督信号，传播输入偏好的传递结构，同时恢复（注：原文摘要在此处截断）

    arXiv:2605.08037v2 Announce Type: replace-cross  Abstract: Direct Preference Optimization (DPO) aligns language models using pairwise preference comparisons, offering a simple and effective alternative to Reinforcement Learning (RL) from human feedback. However, in many practical settings, training data consists of multiple rollouts per prompt, inducing rich preference structure that pairwise DPO fails to exploit. Collapsing such data into independent pairs discards transitivity, introduces redundant or conflicting supervision, and can lead to unstable optimization. We propose Graph Direct Preference Optimization (GraphDPO), a structured generalization of DPO that operates over directed acyclic preference graphs induced by rollout rankings. GraphDPO encodes dominance relations as edges and optimizes a graph-structured Plackett--Luce-inspired objective that aggregates supervision over dominated neighborhoods, propagating the transitive structure of the input preferences while recovering
    
[^352]: 具有全局最优与近似保证的脉冲神经网络凸优化

    Convex Optimization of Spiking Neural Networks with Global and Approximation Guarantees

    [https://arxiv.org/abs/2605.08022](https://arxiv.org/abs/2605.08022)

    该论文将脉冲神经网络（包括LIF-SNN）的训练问题首次等价转化为凸优化问题，证明了零对偶间隙与全局最优保证，并提出具有高概率近似保证的随机化替代方法以解决凸规划变量指数级增长的问题。

    

    脉冲神经网络已被提出作为深度神经网络的节能替代方案。其训练问题是非凸的，且涉及不可微的脉冲函数，因此现有方法无法保证达到全局最优。我们通过将脉冲神经网络优化问题表述为一个等价的凸优化问题来解决这一难题。具体而言，我们证明了一类阈值循环神经网络（包括泄漏积分发放型脉冲神经网络）在带权重衰减正则化条件下的训练问题具有零对偶间隙，且与某个凸规划具有相同的最优值。由于所得的凸规划的变量数量可能随数据量呈指数级增长，我们进一步引入了一种实用的随机化替代方法，并为其近似质量建立了高概率保证。从实证结果来看，表现最好的凸优化变体在加法（addition）和首尾XOR（first-last-XOR）任务上优于代理梯度训练方法，在序列MNIST任务上与其性能相当。

    arXiv:2605.08022v3 Announce Type: replace-cross  Abstract: Spiking neural networks (SNNs) have been proposed as energy-efficient alternatives to deep neural networks. Their training is non-convex and involves a non-differentiable spike function, so existing methods offer no guarantee of reaching a global optimum. We address this issue by formulating SNN optimization as an equivalent convex optimization problem. Specifically, we show that the weight-decay-regularized training problem for a class of threshold RNNs, including leaky integrate-and-fire SNNs, has zero duality gap and the same optimal value as a convex program. Since the resulting convex programs can have exponentially many variables in the amount of data, we introduce a practical randomized surrogate and establish a high-probability guarantee on its approximation quality. Empirically, the best convex variant outperforms surrogate-gradient training on addition and first-last-XOR, matches its performance on sequential MNIST, a
    
[^353]: FactoryBench：评估工业机器理解能力

    FactoryBench: Evaluating Industrial Machine Understanding

    [https://arxiv.org/abs/2605.07675](https://arxiv.org/abs/2605.07675)

    FactoryBench基于珀尔因果阶梯构建了四个因果层级的问答基准，包含7万余个问答项，用于评估时间序列模型和大语言模型对工业机器人遥测数据的理解能力，零样本评测显示当前前沿大语言模型在结构化任务上不超过50%、在决策任务上不超过18%。

    

    我们推出了FactoryBench，一个用于评估时间序列模型和大语言模型（LLM）在工业机器人遥测数据上机器理解能力的基准。问答对按照四个因果层级（状态、干预、反事实、决策）组织，实例化了珀尔的因果阶梯，并涵盖五种答案格式：其中四种结构化格式采用确定性评分，自由形式答案则通过“大模型作为裁判”的投票协议进行评分。我们提出了一个围绕结构化问题模板构建的可扩展问答生成框架，介绍了FactoryWave（一个从UR3协作机器人和KUKA KR10工业机械臂采集的密集、多任务、多变量传感器数据集），并基于来自FactoryWave、AURSAD和voraus-AD的约1.5万个规范化回合，构建了包含超过7万个问答项的大规模基准FactoryBench。对六个前沿大语言模型的零样本评估表明，没有任何模型在结构化层级上超过50%，在决策任务上不超过18%，揭示了……

    arXiv:2605.07675v2 Announce Type: replace  Abstract: We introduce FactoryBench, a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present FactoryWave (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a large-scale benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% on structured levels or 18% on decision-making, revealing 
    
[^354]: 面向离散选择模型的约束三头价格优化网络（C3PO）

    Constrained Triple-Head Price Optimization (C3PO) Network for Discrete-Choice Models

    [https://arxiv.org/abs/2605.06941](https://arxiv.org/abs/2605.06941)

    提出了基于Transformer的约束三头价格优化网络C3PO，通过融合价格模仿学习、多任务收入预测、上下文弹性学习以及轻量级可微分约束层，在离散选择环境下实现无需重训练、可执行定价约束的跨领域实时定价优化。

    

    我们提出了一种约束三头价格优化网络（C3PO），用于离散选择环境中的实时双层决策。在这些场景中，卖方选择产品或服务方案以实现商业目标，而异质用户则基于自身偏好做出个性化的接受或拒绝决策，最终决定卖方的收益结果。现有的在合成数据上微调预训练表格基础模型的方法，受限于其模仿学习范式，且无法强制执行常见的定价规则，包括跨产品约束和全局约束。C3PO通过一种新颖的基于Transformer的架构解决了这些局限，该架构联合整合了价格模仿学习、多任务收入预测和上下文弹性学习，无需重新训练即可跨领域生成稳健的定价建议。一个轻量级的可微分层（用于约束执行）……

    arXiv:2605.06941v2 Announce Type: replace  Abstract: We propose a Constrained Triple-Head Price Optimization (C3PO) network for real-time bi-level decision making in discrete-choice environments. In these settings, a seller selects product or service offerings to achieve business objectives, while heterogeneous users make personalized acceptance or rejection decisions based on their preferences, ultimately determining seller outcomes. Existing approaches that fine-tune pre-trained tabular foundation models on synthetic data are limited by their imitation-learning paradigm and inability to enforce common pricing rules, including cross-product and global constraints. C3PO addresses these limitations through a novel transformer-based architecture that jointly integrates price imitation learning, multi-task revenue prediction, and in-context elasticity learning to generate robust pricing recommendations across domains without retraining. A lightweight differentiable constraint layer enable
    
[^355]: 逆向玩转网络：一个博弈论归因框架

    Playing the network backward: A Game Theoretic Attribution Framework

    [https://arxiv.org/abs/2605.06212](https://arxiv.org/abs/2605.06212)

    该论文提出将神经网络的反向归因方法（如梯度法和LRP）统一重构为扩展网络图上的双人博弈框架，使各类归因图成为特定均衡下博弈轨迹的积分投影，并可通过策略正则化、风险规避等博弈论概念定制期望的解释性质。

    

    归因方法用于解释哪些输入特征驱动了模型的预测，这使它们成为模型调试和机制可解释性的核心。然而，包括梯度法、LRP（层级相关性传播）以及针对Transformer的特定规则在内的反向归因方法，缺乏一个可用来比较底层反向计算的统一框架。我们通过将反向归因重新构建为扩展网络图上的双人博弈，在Gaubert和Vlassopoulos的ReLU网络博弈基础上引入了这样一个框架。梯度和完整的alpha-beta-LRP族在特定均衡下作为博弈轨迹上的积分而出现，因此归因图成为轨迹分布的投影而非首要对象。期望的解释性质，如定位聚焦、对输入噪声的鲁棒性或稳定的注意力路由，可以被指定为博弈论概念，包括策略正则化、风险规避和扩展动作集等。

    arXiv:2605.06212v2 Announce Type: replace  Abstract: Attribution methods explain which input features drive a model's prediction, making them central to model debugging and mechanistic interpretability. Yet backward attribution methods, including gradients, LRP, and transformer-specific rules, lack a shared framework in which to compare the underlying backward calculations. We introduce such a framework by recasting backward attribution as a two-player game on an extended network graph, building on Gaubert and Vlassopoulos' ReLU Net Game. Gradients and the full alpha-beta-LRP family arise as integrals over game trajectories under specific equilibria, so attribution maps become projections of trajectory distributions rather than the primary object. Desired explanation properties, such as localisation focus, robustness to input noise, or stable attention routing, can be specified as game-theoretic concepts, including policy regularization, risk aversion, and extended action sets, and tra
    
[^356]: 通过最优传输训练不可微分网络

    Training Non-Differentiable Networks via Optimal Transport

    [https://arxiv.org/abs/2605.01928](https://arxiv.org/abs/2605.01928)

    本文提出PolyStep方法，通过最优传输理论仅利用前向传播即可训练具有跳跃性损失的非可微网络，解决了反向传播不适用的问题。

    

    我们优化那些具有跳跃性的损失函数：尖峰阈值、量化层和离散路由在前向传播中引入跳跃，反向传播在此不适用。有限差分方法失效：在导数估计半径下，量化网络上99.5%的探测对损失结果完全相同，而平滑控制仅有1.6%。在跳跃点，Clarke和保守平稳性未定义，任何半径为h的平滑化会以1/h的速率陡增，因此不存在消失半径来定义导数。固定分辨率平稳性仍然成立。PolyStep仅通过前向传播即可达到该平稳性，它在旋转多胞体上对探测点进行排序，并沿其方向的softmax加权平均值步进，这是KL惩罚传输程序的λ=0端点。第二个不可能性固定了框架：在比探测范围更宽的平台上，任何重新加权单一成本行的规则（包括softmax）都会冻结，只有列约束得以逃脱。在期望中，

    arXiv:2605.01928v2 Announce Type: replace  Abstract: We optimize losses that jump: spiking thresholds, quantized layers, and discrete routing put jumps in the forward pass, where backpropagation does not apply. Finite differences fail: at a derivative-estimating radius, 99.5% of probe pairs on a quantized network leave the loss bit-identical, against 1.6% on a smooth control. At a jump, Clarke and conservative stationarity are undefined, and any radius-$h$ smoothing steepens like $1/h$, so no vanishing radius defines a derivative. Fixed-resolution stationarity survives. PolyStep attains it from forward passes alone, ranking probe points on a rotated polytope and stepping along a softmax-weighted average of their directions, the $\lambda=0$ endpoint of a KL-penalized transport program. A second impossibility fixes the frame: on a plateau wider than the probes reach, every rule that reweights one cost row freezes, softmax included, and only the column constraint escapes. In expectation, 
    
[^357]: 面向NR-U/Wi-Fi共存的公平性引导DQN传输机会（TXOP）控制

    Fairness-Guided DQN-Based TXOP Control for NR-U/Wi-Fi Coexistence

    [https://arxiv.org/abs/2605.00457](https://arxiv.org/abs/2605.00457)

    本文提出一种公平性引导的深度Q网络方法，通过根据每节点吞吐量比率动态调整NR-U的传输机会（TXOP）时长，有效改善NR-U与Wi-Fi在非授权频谱中共存的公平性与吞吐量性能。

    

    arXiv:2605.00457v5 公告类型：replace-cross 摘要：工作在非授权频谱的新空口（NR-U）与Wi-Fi采用不同的竞争参数和信道占用时长，这可能导致严重的共存失衡。本文提出一种公平性引导的深度Q网络（DQN）方法，利用观测到的每节点吞吐量比率来调整NR-U的传输机会（TXOP）。两种奖励设计Q1和Q2展现出互补的工作模式：Q1促进严格的每节点吞吐量均衡，而Q2允许更宽的无线接入技术间公平性区域，并在中等公平性区间对吞吐量提升给予奖励。这些策略与第三代合作伙伴计划（3GPP）默认的先听后说（LBT）机制、一种改进的多臂老虎机方法、一种基于效用公平性的表格型Q学习对比方法，以及一个以默认配置为参考的固定TXOP基线进行了比较。实验在四个优先级类别和非对称节点规模下报告了聚合……（原文摘要在此处截断）

    arXiv:2605.00457v5 Announce Type: replace-cross  Abstract: New Radio operating in unlicensed spectrum (NR-U) and Wi-Fi employ different contention parameters and channel-occupancy durations, which can produce severe coexistence imbalance. This correspondence develops a fairness-guided deep Q-network (DQN) that adjusts the NR-U transmission opportunity (TXOP) using the observed per-node throughput ratio. Two reward designs, Q1 and Q2, expose complementary operating regimes. Q1 promotes strict per-node throughput balance, whereas Q2 admits a broader inter-radio-access-technology fairness region and rewards throughput improvement in an intermediate fairness band. The policies are compared with the default Third Generation Partnership Project (3GPP) listen-before-talk (LBT), an adapted multi-armed-bandit method, a utility-fairness-based tabular Q-learning comparator, and a default-referenced fixed-TXOP baseline. Experiments across four priority classes and asymmetric populations report agg
    
[^358]: 奥卡姆剃刀的锋利程度取决于你的ELBO

    Occam's Razor is Only as Sharp as Your ELBO

    [https://arxiv.org/abs/2604.25984](https://arxiv.org/abs/2604.25984)

    本文证明在贝叶斯线性回归中，基于ELBO的超参数学习会因高斯近似后验协方差结构的不同而产生系统性偏差：对角或低秩加对角协方差会低估信噪比导致欠拟合，而低秩协方差会高估信噪比导致过拟合。

    

    边际似然，也被称为证据，被视为奥卡姆剃刀的数学体现，使超参数选择能够避免过拟合。来自变分推断的证据下界目标函数也被用于类似的目的。在本文中，我们针对贝叶斯线性回归定义了一个信噪比，该信噪比由控制信号方差和噪声方差的关键超参数表示。随后，我们研究了基于ELBO的超参数学习如何根据高斯近似后验中所假设的协方差矩阵结构而低估或高估信噪比。我们证明了假设对角协方差矩阵或低秩加对角协方差矩阵会低估信噪比，而低秩协方差矩阵会高估信噪比。低估信噪比会导致训练集上的均方误差高于最优值（欠拟合），而高估信噪比则会导致误差更低（过拟合）。

    arXiv:2604.25984v2 Announce Type: replace-cross  Abstract: The marginal likelihood, also known as the evidence, is regarded as a mathematical embodiment of Occam's razor, enabling hyperparameter selection that avoids overfitting. The evidence lower bound (ELBO) objective from variational inference has also been used for similar purposes. In this paper, we define a signal-to-noise ratio (SNR) for Bayesian linear regression in terms of key hyperparameters controlling signal and noise variance. We then study how ELBO-based hyperparameter learning can under or overestimate SNR depending on the assumed covariance matrix structure in a Gaussian approximate posterior. We prove that assuming a diagonal or low-rank plus diagonal covariance matrix underestimates the SNR, while a low-rank covariance matrix overestimates the SNR. Underestimating the SNR leads to higher than optimal mean squared error on the training set (underfitting), while overestimating SNR leads to lower error (overfitting). B
    
[^359]: 高维空间中利用两阶段结构化Hadamard旋转逼近均匀随机旋转

    Approximating Uniform Random Rotations by Two-Stage Structured Hadamard Rotations in High Dimensions

    [https://arxiv.org/abs/2604.23418](https://arxiv.org/abs/2604.23418)

    该论文为两阶段结构化Hadamard旋转逼近高维均匀随机旋转建立了首个非渐近理论保证，证明了坐标分布距离的 $\sqrt{\log d/d}$ 阶一致上界以及 $d^{-1/2}$ 阶下界。

    

    均匀随机旋转在快速Johnson--Lindenstrauss嵌入、核近似、通信高效学习以及近期AI压缩流程中非常有用，但在高维空间中生成和应用这些旋转的计算成本很高。一种实用的替代方法是重复使用带有随机符号对角矩阵的Walsh--Hadamard变换。将该结构化旋转应用两次已在实证中显示出优势，但支持该做法的理论仍然有限。我们研究了两阶段结构化Hadamard旋转的逼近质量，其中 $d$ 是被旋转向量的维度。从积极方面来看，我们证明了在Kolmogorov距离和Wasserstein-1距离下，对于经过 $\sqrt d$ 缩放的对应坐标，在所有单位输入和坐标上一致成立的显式非渐近上界，其阶为 $\sqrt{\log d/d}$。对于输入 $u=e_1$，两种距离在每个坐标处也存在阶为 $d^{-1/2}$ 的下界。从消极方面来看，我们证明了（摘要在此处不完整）

    arXiv:2604.23418v2 Announce Type: replace  Abstract: Uniform random rotations are useful in fast Johnson--Lindenstrauss embeddings, kernel approximation, communication-efficient learning, and recent AI compression pipelines, but are computationally expensive to generate and apply in high dimensions. A practical alternative repeats Walsh--Hadamard transforms with random sign diagonals.   Applying this structured rotation twice has shown empirical benefits, but supporting theory remains limited. We study the approximation quality of the two-stage structured Hadamard rotation. Here $d$ is the dimension of the rotated vector. On the positive side, we prove explicit, nonasymptotic upper bounds of order $\sqrt{\log d/d}$ in both Kolmogorov and Wasserstein-1 distance for corresponding coordinates scaled by $\sqrt d$, uniformly over unit inputs and coordinates. For the input $u=e_1$, both distances also have lower bounds of order $d^{-1/2}$ at every coordinate. On the negative side, we prove a
    
[^360]: 基于成分自适应与病灶级监督的脑部MRI小结构分割改进方法

    Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI

    [https://arxiv.org/abs/2604.08015](https://arxiv.org/abs/2604.08015)

    提出CATMIL训练目标，通过成分自适应Tversky加权和病灶级多示例学习两个辅助损失项，在不修改网络架构的前提下显著提升脑部MRI中小病灶的分割召回率。

    

    脑部MRI中的小病灶难以分割，因为它们仅占体积的极小部分，并且在体素级优化中被背景和较大病灶所主导，导致模型即使达到较高的Dice相似系数（DSC），仍可能遗漏许多小病灶。我们提出CATMIL，一种在不改变网络架构的情况下，向标准nnU-Net的Dice和交叉熵损失添加两个辅助项的训练目标。成分自适应Tversky（CAT）项根据病灶连通域大小的倒数对病灶体素进行加权，使每个病灶无论体积大小都贡献近乎相等的损失。病灶级多示例学习（MIL）项将每个病灶视为一个体素包，并对没有任何体素被检测出来的病灶施加惩罚。在MSLesSeg数据集的多发性硬化病灶分割任务中，CATMIL取得了最高的小病灶召回率（0.873，而Dice+CE为0.796；差异的95%置信区间为+0.030至+0.157，在所有六个测试中均更高）（摘要文本在此处截断）。

    arXiv:2604.08015v3 Announce Type: replace-cross  Abstract: Small lesions in brain MRI are hard to segment because they occupy a tiny fraction of the volume and are dominated by background and larger lesions during voxel-wise optimization, so a model can reach a high Dice similarity coefficient (DSC) while missing many of them. We propose CATMIL, a training objective that adds two auxiliary terms to the standard nnU-Net Dice and cross-entropy loss without changing the architecture. The Component-Adaptive Tversky (CAT) term weights lesion voxels by the inverse size of their connected component, so each lesion contributes nearly equally regardless of volume. The lesion-level Multiple Instance Learning (MIL) term treats each lesion as a bag of voxels and penalizes lesions with no detected voxel. For multiple sclerosis lesion segmentation on MSLesSeg, CATMIL achieves the highest small-lesion recall (0.873 vs. 0.796 for Dice+CE; 95% CI of the difference +0.030 to +0.157, higher in all six te
    
[^361]: 基于Koopman算子的涡扇发动机非线性辨识与模型预测控制

    Koopman-Based Nonlinear Identification and Model Predictive Control of a Turbofan Engine

    [https://arxiv.org/abs/2604.01730](https://arxiv.org/abs/2604.01730)

    本文提出基于元启发式扩展动态模态分解的Koopman模型辨识方法，并据此开发了带扰动观测器的自适应Koopman模型预测控制器，实现涡扇发动机的多变量控制，且该单一模型可跨多种控制策略复用。

    

    本文研究了基于Koopman算子的双转子涡扇发动机多变量控制方法。建立了基于物理的部件级模型，用于生成训练数据并验证控制器。采用了一种元启发式扩展动态模态分解方法，并设计了代价函数以准确捕获转子转速动态和发动机压比（EPR），从而构建出一个可在多种控制策略中复用的单一Koopman模型。基于辨识得到的时变Koopman模型，开发了一种带扰动观测器的自适应Koopman模型预测控制器（AKMPC），并将其与基于Koopman的反馈线性化控制器（K-FBLC）及其积分器增强版本（K-FBLC-I）进行了比较。Koopman表示还使得非线性的燃气涡轮发动机输出限制器（如转子加速度和涡轮进口温度限制）能够被表示为...

    arXiv:2604.01730v3 Announce Type: replace  Abstract: This paper investigates Koopman operator-based approaches for multivariable control of a two-spool turbofan engine. A physics-based component-level model is developed to generate training data and validate the controllers. A meta-heuristic extended dynamic mode decomposition is adapted, with a cost function designed to accurately capture both spool-speed dynamics and the engine pressure ratio (EPR), enabling the construction of a single Koopman model that can be reused across multiple control strategies. Using the identified time-varying Koopman model, an adaptive Koopman-based model predictive controller (AKMPC) with a disturbance observer is developed and compared with a Koopman-based feedback linearization controller (K-FBLC) and its integrator-augmented version (K-FBLC-I). The Koopman representation further enables nonlinear GTE output limiters, such as rotor-acceleration and turbine-inlet-temperature limits, to be expressed as l
    
[^362]: MVNN：一种用于从粒子数据学习McKean-Vlasov动力学的测度值神经网络

    MVNN: A Measure-Valued Neural Network for Learning McKean-Vlasov Dynamics from Particle Data

    [https://arxiv.org/abs/2604.00333](https://arxiv.org/abs/2604.00333)

    该论文提出一种测度值神经网络MVNN，可直接从粒子轨迹数据中学习McKean-Vlasov动力学中依赖测度的相互作用项，并提供了适定性、混沌传播以及万能逼近与定量逼近速率等理论保证。

    

    由相互作用产生的集体行为是众多生物系统的基础。为了从观测数据中学习这类相互作用力，我们提出了一种测度值神经网络，可直接从粒子轨迹观测中推断依赖测度的相互作用（漂移）项。所提出的架构通过学习柱状特征，将标准神经网络推广到可在概率测度上运算，并使用嵌入网络产生可扩展的分布到向量表示。在理论方面，我们建立了所得动力学的适定性，并证明了相应相互作用粒子系统的混沌传播性质。我们进一步在低维测度依赖性假设下证明了万能逼近性以及定量逼近速率。数值实验在一阶和二阶系统上进行，包括确定性与随机性的Motsch-Tadmor动力学、二维（摘要在此处被截断）

    arXiv:2604.00333v3 Announce Type: replace-cross  Abstract: Collective behaviors that emerge from interactions are fundamental to numerous biological systems. To learn such interacting forces from observations, we introduce a measure-valued neural network that infers measure-dependent interaction (drift) terms directly from particle-trajectory observations. The proposed architecture generalizes standard neural networks to operate on probability measures by learning cylindrical features, using an embedding network that produces scalable distribution-to-vector representations. On the theory side, we establish well-posedness of the resulting dynamics and prove propagation-of-chaos for the associated interacting-particle system. We further show universal approximation and quantitative approximation rates under a low-dimensional measure-dependence assumption. Numerical experiments on first and second order systems, including deterministic and stochastic Motsch-Tadmor dynamics, two-dimensiona
    
[^363]: 面向在线生成模型选择的混合贪心策略：在多样性感知的多臂老虎机中UCB是否必要？

    Mixture-Greedy for Online Generative Model Selection: Is UCB Necessary in Diversity-Aware Multi-Armed Bandits?

    [https://arxiv.org/abs/2603.21716](https://arxiv.org/abs/2603.21716)

    本文提出Mixture-Greedy策略，并发现在多样性感知的多臂老虎机生成模型选择问题中，UCB探索奖励并非必要——去除UCB反而收敛更快、样本效率更高、性能更好。

    

    在现代生成式AI中，如何在多个生成模型之间进行高效选择变得越来越重要，因为从次优模型中采样的代价很高。该问题可以被视为一个多臂老虎机（MAB）任务。在多样性感知的评估分数下，生成器的非退化混合可以优于任何单个模型，这使得该MAB设置与经典的最佳臂识别问题区别开来。先前的方法在混合目标中引入了置信上界（UCB）探索奖励。然而，在多个数据集和评估指标上，我们观察到UCB项始终会减慢收敛速度并降低样本效率。相比之下，一种简单的、不包含显式UCB类乐观主义的Mixture-Greedy（混合贪心）策略收敛更快，并取得了更好的性能，尤其是对于FID和Vendi这类广泛使用但难以构造紧致置信界的指标。我们提供了理论见解……

    arXiv:2603.21716v2 Announce Type: replace-cross  Abstract: Efficient selection among multiple generative models is increasingly important in modern generative AI, where sampling from suboptimal models is costly. This problem can be viewed as a multi-armed bandit (MAB) task. Under diversity-aware evaluation scores, a non-degenerate mixture of generators can outperform any individual model, distinguishing this MAB setting from classical best-arm identification. Prior approaches incorporate an Upper Confidence Bound (UCB) exploration bonus into the mixture objective. However, across multiple datasets and evaluation metrics, we observe that the UCB term consistently slows convergence and reduces sample efficiency. In contrast, a simple Mixture-Greedy strategy without explicit UCB-type optimism converges faster and achieves even better performance, particularly for widely used metrics such as FID and Vendi where tight confidence bounds are difficult to construct. We provide theoretical insi
    
[^364]: 权重聚类揭示大语言模型中相对排序的重要性

    Weight Clustering Reveals the Importance of Relative Rank in Large Language Models

    [https://arxiv.org/abs/2603.17917](https://arxiv.org/abs/2603.17917)

    利用标量 K 均值聚类将大语言模型权重压缩为 16-64 个共享值并可在不重训的情况下保持性能，并通过扰动实验揭示保持权重间的相对排序比保持精确数值更重要。

    

    大语言模型包含数十亿参数，但许多精确的权重值并非必需。我们对预训练模型应用标量 K 均值聚类，仅用 16-64 个共享值替换每个权重矩阵。对于 Llama 3.1-8B-Instruct 和 SmolLM2-135M，该方法在无需重新训练的情况下保持了较高的准确率，为在磁盘上压缩大语言模型提供了一种简单途径。仅对聚类中心进行微调，可进一步恢复 30-40% 的剩余精度损失。随后，我们将聚类作为分析工具，固定聚类分配，仅扰动共享值，以研究其哪些性质最为关键。我们发现，保持这些值的相对排序（即哪些连接比其他连接更强或更弱）比保持它们的精确数值更为重要。破坏这种排序会显著增加困惑度，而改变数值但保持排序在许多中间层几乎不造成损失。

    arXiv:2603.17917v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) contain billions of parameters, but many exact weight values are not essential. We apply scalar $K$-means clustering to pretrained models, replacing each weight matrix with only 16--64 shared values. For Llama~3.1-8B-Instruct and SmolLM2-135M, this preserves strong accuracy without retraining, providing a simple way to compress LLMs on disk. Fine-tuning only the cluster centroids further recovers 30--40\% of the remaining accuracy loss. Using clustering as an analysis tool, we then keep the cluster assignments fixed and perturb only the shared values, to study which of their properties matter most. We find that preserving their \emph{relative rank} (which connections are stronger or weaker than others) is more important than preserving their exact magnitudes. Breaking this ordering greatly increases perplexity, while changing the values but keeping the ordering causes little loss in many middle and 
    
[^365]: 用于预测细胞对基因扰动响应的检索增强生成

    Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation

    [https://arxiv.org/abs/2603.07233](https://arxiv.org/abs/2603.07233)

    提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。

    

    预测基因扰动的转录响应是功能基因组学和治疗发现的基础。最近的深度学习模型在单细胞扰动响应预测方面展现出了前景，但它们通常孤立地生成每个响应，而没有显式地利用相关的扰动信息。我们提出了PT-RAG（扰动感知的两阶段检索增强生成），这是一个用于生成式细胞扰动响应的即插即用检索与条件化模块。PT-RAG通过学习访问相关的扰动上下文来增强现有的扰动-响应骨干模型。关键挑战在于，在这种设置下相关性并非固定不变：功能相关的基因在不同细胞类型中可能引发不同的效应。PT-RAG通过两阶段检索机制解决这一问题：首先基于GenePT的语义检索识别K个候选扰动，随后由一个可微分的Gumbel-Softmax选择器…

    arXiv:2603.07233v2 Announce Type: replace  Abstract: Predicting transcriptional responses to genetic perturbations is fundamental to functional genomics and therapeutic discovery. Recent deep learning models have shown promise in single-cell perturbation response prediction, but they typically generate each response in isolation, without explicitly leveraging related perturbations. We introduce PT-RAG (Perturbation-aware Two-stage Retrieval-Augmented Generation), a plug-in retrieval-and-conditioning module for generative cellular perturbation response. PT-RAG augments an existing perturbation-response backbone with learned access to related perturbation contexts. The key challenge is that relevance is not fixed in this setting: functionally related genes may elicit different effects across cell types. PT-RAG addresses this with a two-stage retrieval mechanism: GenePT-based semantic retrieval first identifies K candidate perturbations, after which a differentiable Gumbel-Softmax selecto
    
[^366]: 关于非线性神经网络中梯度下降收敛速度的研究：对抗鲁棒性视角

    On the Rate of Convergence of GD in Non-linear Neural Networks: An Adversarial Robustness Perspective

    [https://arxiv.org/abs/2603.02095](https://arxiv.org/abs/2603.02095)

    该论文在双神经元ReLU网络的最简二分类设置中证明，梯度下降虽能收敛到最优对抗鲁棒性边际，但收敛速度极慢（严格为 Θ(1/ln(t))），并首次建立了非线性模型中鲁棒性边际收敛速度的显式下界。

    

    我们在一个极简的二分类设置中研究梯度下降（GD）的收敛动态，该设置由一个双神经元ReLU网络和两个训练样本组成。我们证明，即使在这些强简化假设下，虽然GD能够成功收敛到最优鲁棒性边际，有效地最大化决策边界与训练点之间的距离，但这一收敛以极其缓慢的速率发生，严格以 Θ(1/ln(t)) 的速度缩放。据我们所知，这建立了非线性模型中鲁棒性边际收敛速度的首个显式下界。通过经验模拟，我们进一步证明这种固有的失效模式是普遍存在的，在多个自然网络初始化下均表现出完全相同的紧致收敛速率。我们的理论保证是通过对GD在不同激活机制下轨迹的严格分析得出的。

    arXiv:2603.02095v2 Announce Type: replace-cross  Abstract: We study the convergence dynamics of Gradient Descent (GD) in a minimal binary classification setting, consisting of a two-neuron ReLU network and two training instances. We prove that even under these strong simplifying assumptions, while GD successfully converges to an optimal robustness margin, effectively maximizing the distance between the decision boundary and the training points, this convergence occurs at a prohibitively slow rate, scaling strictly as $\Theta(1/\ln(t))$. To the best of our knowledge, this establishes the first explicit lower bound on the convergence rate of the robustness margin in a non-linear model. Through empirical simulations, we further demonstrate that this inherent failure mode is pervasive, exhibiting the exact same tight convergence rate across multiple natural network initializations. Our theoretical guarantees are derived via a rigorous analysis of the GD trajectories across the distinct act
    
[^367]: 桥匹配采样器：通过广义不动点扩散匹配实现可扩展采样

    Bridge Matching Sampler: Scalable Sampling via Generalized Fixed-Point Diffusion Matching

    [https://arxiv.org/abs/2603.00530](https://arxiv.org/abs/2603.00530)

    本文提出桥匹配采样器（BMS），将最小二乘匹配方法统一推广为基于Nelson关系的不动点迭代，以单一、稳定且可扩展的目标函数学习任意先验与目标分布间的随机传输映射，并通过阻尼正则化变体缓解模式崩溃、实现前所未有规模的采样。

    

    使用扩散模型从非归一化密度中采样已成为一种强大的范式。然而，尽管近期采用最小二乘“匹配”目标的方法提升了可扩展性，但它们往往需要重大的权衡，例如限制先验分布或依赖不稳定的优化方案。通过将这些方法推广为植根于Nelson关系的不动点迭代的特殊形式，我们开发了一种解决上述局限的新方法，称为桥匹配采样器（BMS）。我们的方法能够通过单一、可扩展且稳定的目标函数，学习任意先验分布与目标分布之间的随机传输映射。此外，我们引入了该迭代的阻尼变体，通过加入正则化项来缓解模式崩溃并进一步稳定训练。实证结果表明，我们的方法能够在前所未有的规模下进行采样，同时保持模式覆盖。

    arXiv:2603.00530v2 Announce Type: replace  Abstract: Sampling from unnormalized densities using diffusion models has emerged as a powerful paradigm. However, while recent approaches that use least-squares `matching' objectives have improved scalability, they often necessitate significant trade-offs, such as restricting prior distributions or relying on unstable optimization schemes. By generalizing these methods as special forms of fixed-point iterations rooted in Nelson's relation, we develop a new method that addresses these limitations, called Bridge Matching Sampler (BMS). Our approach enables learning a stochastic transport map between arbitrary prior and target distributions with a single, scalable, and stable objective. Furthermore, we introduce a damped variant of this iteration that incorporates a regularization term to mitigate mode collapse and further stabilize training. Empirically, we demonstrate that our method enables sampling at unprecedented scales while preserving mo
    
[^368]: 超越逐状态镜像下降：基于参数化策略的离线策略优化

    Beyond State-Wise Mirror Descent: Offline Policy Optimization with Parametric Policies

    [https://arxiv.org/abs/2602.23811](https://arxiv.org/abs/2602.23811)

    该工作将离线强化学习的理论保证从有限小动作空间扩展到大/连续动作空间上的参数化策略类，识别出将镜像下降推广至参数化策略时的核心难点“上下文耦合”，并据此提出了计算上高效（oracle高效）的独立参数化离线策略优化算法。

    

    我们研究了在一般函数逼近下离线强化学习（RL）的理论问题。尽管先前的工作（如 Xie et al., 2021）已通过悲观原则确立了从离线数据中学习良好策略的理论基础，但现有在计算上可行的算法（通常在 oracle 高效的意义上），例如 PSPI，仅适用于有限且较小的动作空间。此外，这些算法依赖于逐状态的镜像下降，并要求策略执行者（actor）从评论者（critic）函数中隐式导出，因而无法适应实践中普遍存在的独立策略参数化方式。在本工作中，我们解决了这些局限性，将理论保证扩展到大动作空间或连续动作空间上的参数化策略类。在将镜像下降扩展到参数化策略时，我们识别出上下文耦合是其中的核心难点，并展示了如何通过将镜像下降与……（原文摘要在此处截断）

    arXiv:2602.23811v5 Announce Type: replace-cross  Abstract: We investigate the theoretical aspects of offline reinforcement learning (RL) under general function approximation. While prior works (e.g., Xie et al., 2021) have established the theoretical foundations of learning a good policy from offline data via pessimism, existing algorithms that are computationally tractable (often in an oracle-efficient sense), such as PSPI, only apply to finite and small action spaces. Moreover, these algorithms rely on state-wise mirror descent and require actors to be implicitly induced from the critic functions, failing to accommodate standalone policy parameterization which is ubiquitous in practice. In this work, we address these limitations and extend the theoretical guarantees to parameterized policy classes over large or continuous action spaces. When extending mirror descent to parameterized policies, we identify contextual coupling as the core difficulty, and show how connecting mirror desce
    
[^369]: 本地隐私保护的变点检测参数化方法

    Locally Private Parametric Methods for Change-Point Detection

    [https://arxiv.org/abs/2602.13619](https://arxiv.org/abs/2602.13619)

    本文研究了本地差分隐私下的参数化变点检测问题，建立了以Chernoff信息为根本限制的有限样本极小极大上下界，并据此设计了分别适用于低隐私和高隐私机制的最优本地隐私变点检测算法，从而精确量化了隐私保护带来的统计代价。

    

    我们研究本地差分隐私下的参数化变点检测问题，重点关注对独立观测序列中单一变点的离线定位。在无隐私保护的设定下，我们利用鞅方法为广义似然比估计量推导出改进的有限样本精度保证。我们在无隐私和本地隐私两种设定下均建立了有限样本极小极大下界，表明定位精度从根本上受变点前后分布之间Chernoff信息的限制。受此启发，我们设计了基于聚合随机响应机制和二元机制的本地隐私变点检测算法，这两种算法分别在低隐私和高隐私机制下最大化诱导输出分布之间的Chernoff信息，并推导出其有限样本保证。我们的上下界共同量化了隐私保护所带来的统计代价。

    arXiv:2602.13619v2 Announce Type: replace-cross  Abstract: We study parametric change-point detection under local differential privacy, focusing on the offline localization of a single change point in a sequence of independent observations.~In the non-private setting, we derive improved finite-sample accuracy guarantees for a generalized likelihood-ratio estimator using martingale methods.~We establish finite-sample minimax lower bounds in both the non-private and locally private settings, showing that localization accuracy is fundamentally limited by the Chernoff information between the pre- and post-change distributions.~Guided by this, we design locally private CPD algorithms based on aggregated randomized response and binary mechanisms, which maximize the Chernoff information between the induced output distributions in low- and high-privacy regimes, respectively, and derive their finite-sample guarantees. Together, our upper and lower bounds quantify the statistical cost of privacy
    
[^370]: 基于图注意力实现跨客户端可解释性的非线性时间动态联邦学习

    Federated Learning of Nonlinear Temporal Dynamics with Graph Attention-based Cross-Client Interpretability

    [https://arxiv.org/abs/2602.13485](https://arxiv.org/abs/2602.13485)

    本文提出一种垂直联邦学习框架，在原始数据不可共享、客户端观测异构且各客户端模型固定不可重训的约束下，利用图注意力机制实现跨客户端的非线性时间动态一步预测，并具备跨客户端可解释性。

    

    现代工业系统的网络日益由分布式传感器进行监测，每个系统由多个子系统组成，这些子系统生成高维时间序列。这些子系统之间往往相互依赖，因此理解在一个子系统中观察到的时间模式与其他子系统中观察到的时间模式之间的关系至关重要。这一问题在去中心化环境中尤其具有挑战性，因为原始测量数据无法共享，且客户端的观测数据是异构的。此外，在实际应用中，每个子系统（客户端）运行着一个固定的、专有的模型，该模型无法被修改或重新训练，这限制了现有方法的适用性。另外，非线性动态使得跨客户端的时间相互依赖关系难以刻画，因为它们隐式地嵌入在非线性状态转移函数之中。本文提出了一种垂直联邦学习框架，用于学习一步前瞻的跨客户端时间动态……

    arXiv:2602.13485v3 Announce Type: replace  Abstract: Networks of modern industrial systems are increasingly monitored by distributed sensors, where each system comprises multiple subsystems that generate high-dimensional time-series. These subsystems are often interdependent, making it important to understand how temporal patterns observed at one subsystem relate to those observed across others. This problem is particularly challenging in decentralized settings where raw measurements cannot be shared, and client observations are heterogeneous. Moreover, in practice, each subsystem (client) operates a fixed, proprietary model that cannot be modified or retrained, limiting the applicability of existing approaches. Separately, nonlinear dynamics make cross-client temporal interdependencies difficult to characterize, as they are embedded implicitly within nonlinear state-transition functions. This paper presents a vertical federated learning framework for learning one-step-ahead cross-clie
    
[^371]: ERIS：通过联邦分片聚合增强联邦学习中的隐私-效用-可扩展性权衡

    ERIS: Enhancing the Privacy-Utility-Scalability Trade-Off in Federated Learning via Federated Shard Aggregation

    [https://arxiv.org/abs/2602.08617](https://arxiv.org/abs/2602.08617)

    ERIS提出联邦分片聚合（FSA）机制，将客户端更新分割为互不重叠的分片并分散至多个客户端聚合器进行聚合，从而在十亿参数规模的联邦学习中同时改善隐私、可扩展性与模型效用之间的权衡。

    

    将联邦学习（FL）扩展到十亿参数级别的模型，会在隐私、可扩展性和模型效用之间带来一个具有挑战性的权衡。现有解决方案往往孤立地应对这些挑战，要么牺牲精度，要么依赖昂贵的密码学工具，要么引入影响收敛的通信和优化低效问题。我们提出了ERIS，一个以联邦分片聚合（FSA）为核心的联邦学习框架。FSA是一种新颖的机制，它将每个客户端的更新分割成互不重叠的分片，并将这些分片的聚合任务分散到多个客户端聚合器上执行。FSA消除了中央聚合瓶颈，限制了任何单一观察者可见的信息，并在重新组装后保持与集中式联邦学习更新的一致性。此外，ERIS可以便捷地集成分布式移位压缩（DSC），以减少传输负载和暴露的坐标数量。我们证明了在标准假设下，ERIS能够保持收敛性……

    arXiv:2602.08617v3 Announce Type: replace  Abstract: Scaling Federated Learning (FL) to billion-parameter models forces a challenging trade-off between privacy, scalability, and model utility. Existing solutions often tackle these challenges in isolation, sacrificing accuracy, relying on costly cryptographic tools, or introducing communication and optimization inefficiencies that affect convergence. We introduce ERIS, an FL framework centered on Federated Shard Aggregation (FSA), a novel mechanism that partitions each client update into non-overlapping shards whose aggregation is distributed across multiple client-side aggregators. FSA removes the central aggregation bottleneck, limits the information visible to any single observer, and preserves the centralized FL update after reassembly. ERIS can further readily integrate Distributed Shifted Compression (DSC) to reduce transmitted payloads and exposed coordinates. We prove that ERIS preserves convergence under standard assumptions an
    
[^372]: RanSOM：用于约束与无约束优化的随机化缩放二阶动量方法

    RanSOM: Second-Order Momentum with Randomized Scaling for Constrained and Unconstrained Optimization

    [https://arxiv.org/abs/2602.06824](https://arxiv.org/abs/2602.06824)

    该论文提出RanSOM方法，通过将优化步长随机化（指数或Beta分布），利用Stein型恒等式使新迭代点处单次Hessian-向量积（且其前向传播可与下一次梯度计算共享）成为梯度变化的无偏估计，从而在无需Lipschitz Hessian或额外前向反向传播的情况下，对约束（Frank-Wolfe）和无约束优化均达到最优的oracle复杂度。

    

    arXiv:2602.06824v3 公告类型： replace-cross 摘要：二阶动量通过Hessian-向量积（HVP）来修正动量落后于当前梯度的滞后问题，但现有的修正方法要么需要Hessian满足Lipschitz条件，要么需要在辅助点进行额外的前向和反向传播。我们证明，将步长随机化可以消除这一权衡：如果步长服从均值为η的指数分布，一个Stein型恒等式使得新迭代点处的一次HVP乘以η成为梯度变化的无偏估计（对该随机步长的期望而言），并且该HVP的前向传播可以与下一次梯度计算共享。由此得到的方法RanSOM-E适用于归一化步长、符号步长和Muon型步长；而Beta分布变体RanSOM-B则能保持Frank-Wolfe迭代的可行性。对于梯度Lipschitz连续且梯度噪声和Hessian噪声均为重尾分布（尾指数p, q ∈ (1,2]）的目标函数，RanSOM-E对每一对都达到了最优的预言机复杂度指数，包括p...（原文在此截断）

    arXiv:2602.06824v3 Announce Type: replace-cross  Abstract: Second-order momentum corrects the lag of momentum behind the current gradient with a Hessian-vector product (HVP), but existing corrections need either a Lipschitz Hessian or extra forward and backward passes at an auxiliary point. We show that randomizing the step length removes this trade-off: if the step is exponentially distributed with mean $\eta$, a Stein-type identity makes $\eta$ times one HVP at the new iterate an estimate of the gradient change that is unbiased in expectation over the step, and this HVP shares its forward pass with the next gradient. The resulting method, RanSOM-E, works with normalized, sign, and Muon-type steps, and a Beta-distributed variant, RanSOM-B, keeps Frank-Wolfe iterates feasible. For objectives with Lipschitz gradient and heavy-tailed gradient and Hessian noise with tail indices $p, q \in (1,2]$, RanSOM-E attains the optimal oracle-complexity exponents for every pair $(p,q)$, including $p
    
[^373]: 非平稳加权风险最小化的快速收敛率

    Fast Rates for Nonstationary Weighted Risk Minimization

    [https://arxiv.org/abs/2602.05742](https://arxiv.org/abs/2602.05742)

    本文将分布漂移下的超额风险分解为学习误差与漂移误差，对非平稳加权经验风险最小化建立了在任意权重类上一致成立的快速oracle界，并在结构突变与平滑漂移过程以及线性模型、基近似和神经网络等回归问题中刻画了最优权重。

    

    加权经验风险最小化是分布漂移下预测问题的一种常用方法。本文研究了非平稳条件下该方法的样本外预测误差。我们将超额风险给出了一个一般性的分解：学习项与分布漂移相关的误差项，并在混合条件下证明了学习误差的oracle不等式。该学习界对任意权重类一致成立，并考虑了由权重向量诱导的有效样本量、权重类与假设类的复杂度以及潜在的数据依赖性。我们还分析了平方损失设定下的漂移误差，并针对结构突变过程和平滑漂移过程给出了显式界。随后，我们将学习分析与漂移分析相结合，应用于线性模型、基函数近似和神经网络的（自）回归问题。在所有情形下，最优权重都得到了刻画。

    arXiv:2602.05742v3 Announce Type: replace-cross  Abstract: Weighted empirical risk minimization is a common approach to prediction under distribution drift. This article studies its out-of-sample prediction error under nonstationarity. We provide a general decomposition of the excess risk into a learning term and an error term associated with distribution drift, and prove oracle inequalities for the learning error under mixing conditions. The learning bound holds uniformly over arbitrary weight classes and accounts for the effective sample size induced by the weight vector, the complexity of the weight and hypothesis classes, and potential data dependence. We also analyze the drift error in the square loss setting and provide explicit bounds for structural-break and smoothly drifting processes. We then combine the learning and drift analysis in (auto-) regression problems with linear models, basis approximations, and neural networks. In all cases, the optimal weights are characterized 
    
[^374]: COBALT：基于离线轨迹的上下文老虎机学习用于多轮代码生成

    COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation

    [https://arxiv.org/abs/2602.03806](https://arxiv.org/abs/2602.03806)

    COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。

    

    训练大语言模型（LLM）进行多轮代码生成通常依赖在线强化学习（RL），即模型通过反复生成的在线 rollout 进行学习。虽然有效，但在线强化学习成本高昂且不稳定。本文提出了基于离线轨迹的上下文老虎机学习方法（COBALT），这是一种简单的方法，通过重用离线多轮轨迹来高效训练 LLM 进行多轮代码生成。基于多轮代码生成是一个单步可恢复马尔可夫决策过程的观察，COBALT 首先使用一个参考 LLM 收集代码生成轨迹，并将其划分为部分轨迹。然后，在在线老虎机学习过程中，训练 LLM 通过单步代码生成来补全每个部分轨迹。在 LiveCodeBench 上，COBALT 使 R1-Distill 8B 和 Qwen3 8B 的绝对 Pass@1 分数最多分别提升 9.0 和 6.2 个百分点，并匹敌了（原摘要在此处截断）。

    arXiv:2602.03806v2 Announce Type: replace-cross  Abstract: Training large language models (LLMs) for multi-turn code generation typically relies on online reinforcement learning (RL), in which models learn from repeatedly generated online rollouts. Although effective, online RL can be costly and unstable. In this paper, we present contextual bandit learning with offline trajectories (COBALT), a simple method that reuses offline multi-turn trajectories to efficiently train LLMs for multi-turn code generation. Building on the observation that multi-turn code generation is a one-step recoverable Markov decision process, COBALT first collects code generation trajectories using a reference LLM and divides them into partial trajectories. Then, during online bandit learning, the LLM is trained to complete each partial trajectory through single-step code generation. On LiveCodeBench, COBALT substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 points and matche
    
[^375]: HumanX：从人类视频迈向敏捷且可泛化的人形机器人交互技能

    HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos

    [https://arxiv.org/abs/2602.02473](https://arxiv.org/abs/2602.02473)

    HumanX通过协同设计的XGen数据生成流水线与XMimic交互优先学习框架，将人类视频直接转化为无需任务定制设计、可泛化的人形机器人敏捷交互技能。

    

    让人形机器人能够执行敏捷且自适应的交互，长期以来一直是机器人学中的核心挑战。当前方法要么受限于真实交互数据的稀缺，要么需要对特定任务进行细致的奖励工程设计，这限制了其可扩展性。为缩小这一差距，我们提出了HumanX——一个统一框架，能够将人类视频直接编译为人形机器人可泛化的真实世界交互技能，且无需针对每个任务进行专门设计。HumanX集成了两个协同设计的组件：XGen，一个以锚点为中心的数据生成流水线，可从单目视频中生成物理上合理的机器人交互数据，并通过数据增强构建多样化、可扩展的训练数据集；以及XMimic，一个交互优先的学习框架，能够仅凭几秒钟的演示就习得可泛化的交互技能。该方法在五个不同领域进行了评估——篮球、足球、羽毛球、货物拾取以及反应性（摘要在此处被截断）。

    arXiv:2602.02473v2 Announce Type: replace-cross  Abstract: Enabling humanoid robots to perform agile and adaptive interactions has long been a core challenge in robotics. Current approaches are bottlenecked by either the scarcity of realistic interaction data or the need for meticulous, task-specific reward engineering, which limits their scalability. To narrow this gap, we present HumanX, a unified framework that compiles human video into generalizable, real-world interaction skills for humanoids, without per-task design. HumanX integrates two co-designed components: XGen, an anchor-centric data generation pipeline that yields physically plausible robot interaction data from monocular video and augments it to create a diverse, scalable training dataset; and XMimic, an interaction-prioritized learning framework to achieve generalizable interaction skills from seconds of demonstration. Evaluated across five distinct domains--basketball, football, badminton, cargo pickup, and reactive fi
    
[^376]: 超越预测集大小：通过信度集合几何量化共形不确定性

    Beyond Prediction-Set Size: Quantifying Conformal Uncertainty through Credal Geometry

    [https://arxiv.org/abs/2602.01667](https://arxiv.org/abs/2602.01667)

    该论文提出以信度集合几何作为超越预测集大小的共形不确定性量化新视角，为多种信度不确定性度量给出解析刻画与排序后线性时间的精确算法，通过 Tsallis 熵集合大小惩罚将其与经典共形摘要统一，并证明了常见分离式共形回归中不确定性摘要无法区分输入的不可能性定理。

    

    预测集大小只是对共形不确定性的一种粗略概括。我们通过信度集合的几何视角发展出一种更丰富的刻画方式：信度集合是与共形预测（CP）区域相容的概率分布族。我们为分离式共形预测建立了这一表示，扩展了最近在全共形预测中发现的对应关系。基于该表示，我们为几种广泛使用的信度不确定性度量推导出解析刻画和精确算法，在共形 p 值排序之后，可将潜在的指数级计算降低至线性时间。我们进一步通过引入基于 Tsallis 熵的、跨越不同误差水平的集合大小惩罚，将其中若干度量与经典共形摘要统一起来。我们还证明了一个不可能性定理：对于常见的分离式共形回归构造，所诱导信度集合的不确定性摘要无法区分不同的输入。分类实验……（摘要原文在此处截断）

    arXiv:2602.01667v2 Announce Type: replace  Abstract: Prediction-set size offers a coarse summary of conformal uncertainty. We develop a richer perspective through the geometry of credal sets: families of probability distributions compatible with conformal prediction (CP) regions. We establish this representation for split CP, extending a correspondence recently identified for full CP. Building on this representation, we derive analytical characterisations and exact algorithms for several widely used credal uncertainty measures, reducing potentially exponential computations to linear time after sorting the conformal p-values. We further unify several of these measures with classical conformal summaries by integrating set-size penalties based on Tsallis entropies across error levels. We also prove an impossibility theorem: for common split-conformal regression constructions, uncertainty summaries of the induced credal sets cannot distinguish between inputs. Classification experiments on 
    
[^377]: 一种面向半哈斯勒热电材料的可泛化机器学习框架

    A Generalizable Machine Learning Framework for Half-Heusler Thermoelectrics

    [https://arxiv.org/abs/2602.01149](https://arxiv.org/abs/2602.01149)

    本文提出了一种可泛化的机器学习框架，通过基于PCA的数据划分策略、贝叶斯超参数优化与特征选择，能够准确预测半哈斯勒化合物（包括未见成分）的热电品质因数zT，并将RMSE降低16-27%。

    

    机器学习（ML）已成为加速热电（TE）材料发现的强大方法，然而许多模型的可泛化性有限。本文提出了一个用于预测半哈斯勒（hH）化合物热电品质因数（zT）的稳健机器学习框架，重点在于对先前未见成分的预测。我们引入了一种基于主成分分析（PCA）的训练/测试集划分策略，以生成化学空间中具有代表性的分区。针对随机森林、XGBoost和神经网络模型，将贝叶斯超参数优化与基于相关性的k最优特征选择相结合，并辅以可解释的SISSO符号回归方法。相对于全部118个输入特征，优化后的特征子集将测试集均方根误差（RMSE）降低了16-27%。SHAP和SISSO分析确定温度、A位掺杂浓度和A位汽化热为关键特征。

    arXiv:2602.01149v2 Announce Type: replace-cross  Abstract: Machine learning (ML) has emerged as a powerful approach for accelerating the discovery of thermoelectric (TE) materials, yet many models suffer from limited generalizability. Here we present a robust ML framework for predicting the thermoelectric figure of merit (zT) in half-Heusler (hH) compounds, emphasizing prediction of previously unseen compositions. A principal component analysis (PCA)-based train/test splitting strategy is introduced to generate representative partitions of chemical space. Bayesian hyperparameter optimization is combined with correlation-based k-best feature selection for Random Forest, XGBoost, and Neural Network models, together with the interpretable SISSO symbolic-regression method. Relative to the full set of 118 input features, the optimized feature subsets reduce the test RMSE by 16-27%. SHAP and SISSO analyses identify temperature, A-site dopant concentration, and A-site heat of vaporization as 
    
[^378]: RPP：面向数据集不平衡场景下后门攻击的认证型中毒样本检测框架

    RPP: A Certified Poisoned-Sample Detection Framework for Backdoor Attacks under Dataset Imbalance

    [https://arxiv.org/abs/2602.00183](https://arxiv.org/abs/2602.00183)

    本文首次揭示了数据集不平衡会加剧后门攻击的脆弱性，并提出了一种仅在黑盒设置下利用模型输出概率运行的认证型中毒样本检测框架RPP，该框架可提供可证明的域内可检测性保证和误报率的概率上界。

    

    深度神经网络极易受到后门攻击，然而迄今为止大多数防御方法都依赖于平衡数据，忽视了现实场景中普遍存在的类别不平衡问题，而这种不平衡会进一步放大后门威胁。本文首次深入研究了数据集不平衡如何放大后门攻击的脆弱性，研究结果表明：（i）不平衡会引起多数类偏差，从而增加模型对后门攻击的易感性；（ii）随着不平衡程度的加剧，传统防御方法的性能会显著下降。为解决这一问题，我们提出了随机化概率扰动，这是一种认证型中毒样本检测框架，可在黑盒设置下仅利用模型输出概率进行操作。对于任意被检测的样本，RPP能够判断该输入是否遭受了后门操纵，同时提供可证明的域内可检测性保证以及误报率的概率上界。在……上进行了大量实验（原文此处截断）

    arXiv:2602.00183v2 Announce Type: replace-cross  Abstract: Deep neural networks are highly susceptible to backdoor attacks, yet most defense methods to date rely on balanced data, overlooking the pervasive class imbalance in real-world scenarios that can amplify backdoor threats. This paper presents the first in-depth investigation of how the dataset imbalance amplifies backdoor vulnerability, showing that (i) the imbalance induces a majority-class bias that increases susceptibility and (ii) conventional defenses degrade significantly as the imbalance grows. To address this, we propose Randomized Probability Perturbation (RPP), a certified poisoned-sample detection framework that operates in a black-box setting using only model output probabilities. For any inspected sample, RPP determines whether the input has been backdoor-manipulated, while offering provable within-domain detectability guarantees and a probabilistic upper bound on the false positive rate. Extensive experiments on fi
    
[^379]: 基于偏好条件的重型车辆战术决策多目标强化学习

    Preference-Conditioned Multi-Objective Reinforcement Learning for Tactical Decision Making in Heavy-Duty Vehicles

    [https://arxiv.org/abs/2601.18783](https://arxiv.org/abs/2601.18783)

    提出一种基于偏好条件的多目标强化学习框架，结合广义策略改进与近端策略优化，在安全性、能效和时间三个冲突目标之间逼近帕累托前沿，从而实现重型车辆的战术决策优化。

    

    将强化学习应用于自动驾驶等现实世界场景的一个核心难点在于，通过聚合多个通常相互冲突的目标而得到的传统标量奖励设计，可能会掩盖这些目标之间权衡关系的结构。这些挑战对于重型车辆的决策制定尤为突出，因为重型车辆需要在严苛的商业需求下平衡安全性、效率和运营成本。我们提出了一种基于偏好条件的多目标强化学习框架，该框架显式地表示这些目标，并利用广义策略改进和近端策略优化来学习能够捕捉这些目标之间不同权衡策略组合的策略。所提出的方法在三个相互冲突的目标之间逼近帕累托前沿：安全性（以碰撞和成功完成任务来量化）、能效（以能耗成本来量化）以及时间（原文摘要此处截断）。

    arXiv:2601.18783v3 Announce Type: replace-cross  Abstract: A central difficulty in applying reinforcement learning to real-world applications such as autonomous driving is that conventional scalar reward formulations obtained by aggregating multiple often competing objectives may obscure the structure of their trade-offs. These challenges are particularly relevant to decision-making for heavy-duty vehicles, which requires balancing safety, efficiency, and operational costs under demanding commercial requirements. We present a preference-conditioned multi-objective reinforcement learning framework that explicitly represents these objectives and utilizes Generalized Policy Improvement and Proximal Policy Optimization to learn policies capturing different trade-offs between them. The proposed method approximates the Pareto front across three conflicting objectives: safety, quantified in terms of collisions and successful completion; energy efficiency, quantified using energy cost; and tim
    
[^380]: 学习在保证收敛性的前提下加速Krasnosel'skii-Mann不动点迭代

    Learning to accelerate Krasnosel'skii-Mann fixed-point iterations with guarantees

    [https://arxiv.org/abs/2601.07665](https://arxiv.org/abs/2601.07665)

    本文提出一个学习优化框架，通过向Krasnosel'skii-Mann不动点迭代注入可求和扰动来提升其平均情况性能，同时在理论上保证局部线性收敛性，并可推广用于加速多种算子分裂方法。

    

    我们提出了一个有原则的“学习优化”（Learning to Optimize, L2O）框架，用于求解涉及一般非扩张映射的不动点问题。我们的想法是向标准的Krasnosel'skii-Mann迭代中刻意注入可求和的扰动，以提升其在特定问题分布上的平均情况性能，同时保留其收敛性保证。在度量次正则性假设下，我们证明了所提出的参数化仅包含那些在局部实现线性收敛的迭代（直至一个消失的偏差项），并且它涵盖了所有以足够快速率实现这一目标的迭代。随后，我们展示了如何利用该框架来增强几种广泛使用的算子分裂方法，以加速结构化单调包含问题的求解，并通过一个L2O增强的Douglas-Rachford分裂算法在最佳逼近问题上验证了我们的方法。

    arXiv:2601.07665v2 Announce Type: replace-cross  Abstract: We introduce a principled learning to optimize (L2O) framework for solving fixed-point problems involving general nonexpansive mappings. Our idea is to deliberately inject summable perturbations into a standard Krasnosel'skii-Mann iteration to improve its average-case performance over a specific distribution of problems while retaining its convergence guarantees. Under a metric sub-regularity assumption, we prove that the proposed parametrization includes only iterations that locally achieve linear convergence-up to a vanishing bias term-and that it encompasses all iterations that do so at a sufficiently fast rate. We then demonstrate how our framework can be used to augment several widely-used operator splitting methods to accelerate the solution of structured monotone inclusion problems, and validate our approach on a best approximation problem using an L2O-augmented Douglas-Rachford splitting algorithm.
    
[^381]: 概率多智能体飞机着陆时间预测

    Probabilistic Multi-Agent Aircraft Landing Time Prediction

    [https://arxiv.org/abs/2512.08281](https://arxiv.org/abs/2512.08281)

    提出一个概率多智能体着陆时间预测框架，将多架飞机的着陆时间以概率分布形式输出，同时兼顾轨迹不确定性及空域中飞机间的交互影响。

    

    准确可靠的飞机着陆时间预测对于空中交通管理中有效的资源配置至关重要。然而，飞机轨迹和交通流的固有不确定性对预测精度和可信度都构成了重大挑战。因此，预测模型不仅应提供飞机着陆时间的点估计，还应提供与这些预测相关的不确定性。此外，飞机轨迹经常通过雷达引导等空中交通管制干预措施受到附近飞机存在的影响。因此，着陆时间预测模型必须考虑空域中的多智能体交互。在这项工作中，我们提出了一个概率多智能体飞机着陆时间预测框架，该框架将多架飞机的着陆时间以分布形式提供。我们使用空中交通监控数据评估了所提出的框架。

    arXiv:2512.08281v2 Announce Type: replace-cross  Abstract: Accurate and reliable aircraft landing time prediction is essential for effective resource allocation in air traffic management. However, the inherent uncertainty of aircraft trajectories and traffic flows poses significant challenges to both prediction accuracy and trustworthiness. Therefore, prediction models should not only provide point estimates of aircraft landing times but also the uncertainties associated with these predictions. Furthermore, aircraft trajectories are frequently influenced by the presence of nearby aircraft through air traffic control interventions such as radar vectoring. Consequently, landing time prediction models must account for multi-agent interactions in the airspace. In this work, we propose a probabilistic multi-agent aircraft landing time prediction framework that provides the landing times of multiple aircraft as distributions. We evaluate the proposed framework using an air traffic surveillan
    
[^382]: MetaJoin：面向异构表的自适应连接选择

    MetaJoin: Adaptive Join Selections for Heterogeneous Tables

    [https://arxiv.org/abs/2512.02444](https://arxiv.org/abs/2512.02444)

    MetaJoin提出了一个无需标签的自适应连接选择框架，能够在执行后依据专家的输出证据动态决定保留、比较、升级或修复连接结果，并引入基于互最近邻覆盖率搜索字符串转换的MNN Join专家。

    

    连接异构表需要不同形式的对应关系，包括完全一致、句法转换、语义相似性以及实体级匹配。合适的连接方法往往事先未知，而运行所有候选方法会产生不必要的开销。我们提出了MetaJoin，一个无需标签的自适应连接选择框架。MetaJoin利用输入证据选择一个初始专家，然后利用该专家特有的输出证据来决定其结果应当被保留、与备选方案比较、升级处理还是进行修复。这使得连接选择能够在执行后进行自适应调整，而不是一次性做出固定的路由决策。我们还提出了MNN Join，一种转换专家，它通过最大化两个未对齐列之间的互最近邻覆盖率来搜索字符串转换，并利用相同的互惠结构来验证其输出。我们对Me...（摘要原文在此处截断）

    arXiv:2512.02444v2 Announce Type: replace-cross  Abstract: Joining heterogeneous tables requires different forms of correspondence, including exact agreement, syntactic transformation, semantic similarity, and entity-level matching. The appropriate join method is often unknown in advance, while running every candidate method incurs unnecessary cost. We introduce MetaJoin, a label-free framework for adaptive join selection. MetaJoin uses input evidence to choose an initial specialist, then uses specialist-specific output evidence to determine whether that result should be retained, compared with an alternative, escalated, or repaired. This allows join selection to adapt after execution rather than committing to a one-shot routing decision. We also introduce MNN Join, a transformation specialist that searches for string transformations by maximizing mutual-nearest-neighbor coverage between two unaligned columns and uses the same reciprocal structure to validate its output. We evaluate Me
    
[^383]: 不完整前瞻信息下多用途批处理过程的贝叶斯动态调度

    Bayesian dynamic scheduling of multipurpose batch processes under incomplete look-ahead information

    [https://arxiv.org/abs/2512.01093](https://arxiv.org/abs/2512.01093)

    该论文提出了一种基于从扰动概率分布中学习贝叶斯网络的动态调度方法，使多用途批处理过程能够在仅拥有不完整前瞻信息的现实条件下有效应对扰动，从而最小化长期成本和系统紧张性。

    

    多用途批处理过程因其能够适应低产量、高价值产品和不断变化的需求，在制造业中日益受到欢迎。这些过程通常在动态环境中运行，面临诸如加工延迟和需求变化等扰动。为了最小化长期成本和系统紧张性（即对调度产生的破坏性改变），调度员必须设计重调度策略以有效应对此类扰动。现有方法通常假设在整个调度范围内拥有完整的前瞻信息。这一假设与现实情况不符——实际中调度员只能获取不完整的前瞻信息。沿用现有方法可能导致次优的长期成本和高水平的系统紧张性。在本工作中，我们提出了一种贝叶斯动态调度方法。我们的方法依赖于从扰动的概率分布中学习贝叶斯网络。

    arXiv:2512.01093v2 Announce Type: replace  Abstract: Multipurpose batch processes become increasingly popular in manufacturing industries since they adapt to low-volume, high-value products and shifting demands. These processes often operate in a dynamic environment, which faces disturbances such as processing delays and demand changes. To minimise long-term cost and system nervousness (i.e., disruptive changes to schedules), schedulers must design rescheduling strategies to address such disturbances effectively. Existing methods often assume complete look-ahead information over the scheduling horizon. This assumption contrasts with realistic situations where schedulers can only access incomplete look-ahead information. Sticking with existing methods may lead to suboptimal long-term costs and high-level system nervousness. In this work we propose a Bayesian dynamic scheduling method. Our method relies on learning a Bayesian Network from the probability distribution of disturbances. Spe
    
[^384]: 无穷小算子世界模型：离线强化学习的有限样本保证

    Infinitesimal Operator World Models: Finite-Sample Guarantees for Offline Reinforcement Learning

    [https://arxiv.org/abs/2511.10383](https://arxiv.org/abs/2511.10383)

    该论文将强化学习与HJB方程相联系，提出基于受控扩散过程无穷小生成元（在再生核希尔伯特空间中学习）的算子理论世界模型，在生成元近似误差足够小时证明了值函数的全局收敛性，并为离线强化学习提供了有限样本保证。

    

    连续时间随机过程是许多自然系统与工程系统的基础。在医疗保健、自动驾驶和工业控制等领域，与环境的直接交互往往是不安全或不切实际的，这促使人们从历史数据中进行离线强化学习。然而，对于从离线数据集中学习策略所固有的近似误差，目前的统计学理解仍然有限。我们通过将强化学习与哈密顿-雅可比-贝尔曼（HJB）方程联系起来来解决这一问题，并提出了一种基于简单动态规划递归的算子理论算法。具体而言，我们以受控扩散过程的无穷小生成元来表示世界模型，并在再生核希尔伯特空间中对其进行学习。通过融合统计学习方法与算子理论，我们在生成元近似误差足够小的条件下建立了值函数的全局收敛性，并由此推导出有限样本保证……

    arXiv:2511.10383v2 Announce Type: replace-cross  Abstract: Continuous-time stochastic processes underlie many natural and engineered systems. In healthcare, autonomous driving, and industrial control, direct interaction with the environment is often unsafe or impractical, motivating offline reinforcement learning from historical data. However, there is limited statistical understanding of the approximation errors inherent in learning policies from offline datasets. We address this by linking reinforcement learning to the Hamilton-Jacobi-Bellman equation and proposing an operator-theoretic algorithm based on a simple dynamic programming recursion. Specifically, we represent our world model in terms of the infinitesimal generator of controlled diffusion processes learned in a reproducing kernel Hilbert space. By integrating statistical learning methods and operator theory, we establish global convergence of the value function under sufficiently small generator-approximation error and der
    
[^385]: 信息力学理论：通过不确定性测量实现泛化（学习即测量）

    A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)

    [https://arxiv.org/abs/2510.22809](https://arxiv.org/abs/2510.22809)

    提出了一种基于信息论不确定性（惊奇度）的“无模型”学习框架，将学习视为对不确定性的测量，可直接从原始数据进行推理，支持数据的编辑与删除，并在异常检测等任务上达到最先进性能。

    

    传统机器学习依赖显式模型和领域假设，这常常限制了灵活性和可解释性。我们引入了一种“无模型”的基于实例的学习框架，利用惊奇度（surprisal，即信息论中的不确定性）直接对原始数据进行分析和推理，从而消除了分布建模的需求，减少了偏差，并支持高效的更新操作，包括直接编辑和删除训练数据。通过以不确定性来量化相关性，该方法能够在多种任务上实现可泛化的推理，包括生成式推理、因果发现、异常检测和时间序列预测。我们的方法强调可追溯性、可解释性和数据驱动的决策，为机器学习提供了一个统一的、人类可理解的框架，几乎可以适用于所有机器学习任务，在所有任务上均表现出色，并在异常检测上达到了最先进的性能。

    arXiv:2510.22809v2 Announce Type: replace-cross  Abstract: Traditional machine learning relies on explicit models and domain assumptions, often limiting flexibility and interpretability. We introduce a "model-free" framework of instance-based learning using surprisal (information theoretic uncertainty) to directly analyze and perform inferences from raw data, eliminating distribution modeling, reducing bias, and enabling efficient updates including direct edits and deletion of training data. By quantifying relevance through uncertainty, the approach enables generalizable inference across tasks including generative inference, causal discovery, anomaly detection, and time series forecasting. Our approach emphasizes traceability, interpretability, and data-driven decision making, offering a unified, human-understandable framework for machine learning, can be adapted to virtually all machine learning tasks, achieving competency across all and state-of-the-art performance on anomaly detecti
    
[^386]: 基于异步曲率信息的二阶优化

    Second-Order Optimization with Asynchronous Curvature

    [https://arxiv.org/abs/2510.15714](https://arxiv.org/abs/2510.15714)

    提出Split-Client异步任务并行设计，使Hessian分解在后台进行而不阻塞梯度更新，并证明了依赖曲率平均延迟（而非最大延迟）的一阶驻点收敛界，其墙钟时间性能可与最优调参的Lazy Hessian方法相媲美且无需调参重用周期。

    

    二阶方法每次迭代进展迅速，但构造和分解Hessian矩阵往往占据了大部分实际运行时间（wall-clock time）。我们研究了Split-Client（分裂客户端），一种任务并行设计：曲率客户端在后台对Hessian进行分解，同时梯度客户端使用最近发布的矩阵执行三次正则化步长，从而使曲率计算的成本表现为延迟而非阻塞时间。对于延迟的精确Hessian，我们证明了一阶驻点界，该界取决于曲率的平均延迟（age）而非最大延迟，证明中使用了重叠计数和加权Hardy不等式，并允许任意有界误差的初始矩阵；该分析还扩展到了子采样曲率和一致不精确曲率的情形。在固定成本模型下，所得到的实际运行时间界与经过最优调节的Lazy Hessian方法的扩展性相匹配，且无需选择重用周期。在两个LIBSVM数据集上的实验中，该实现降低了……（摘要原文此处截断）

    arXiv:2510.15714v4 Announce Type: replace-cross  Abstract: Second-order methods make fast progress per iteration, but forming and factorizing the Hessian often dominates their wall-clock time. We study \emph{Split-Client}, a task-parallel design in which a curvature client factorizes Hessians in the background while a gradient client takes cubic-regularized steps with the most recently published matrix, so that curvature cost appears as a delay rather than as blocking time. For delayed exact Hessians, we prove a first-order stationarity bound that depends on the average age of the curvature rather than its maximum, using an overlap count and a weighted Hardy inequality that allows any bounded-error initial matrix; the analysis extends to subsampled and uniformly inexact curvature. Under a fixed-cost model, the resulting wall-clock bound matches the scaling of optimally tuned Lazy Hessian without choosing a reuse period. In experiments on two LIBSVM datasets, an implementation decreases
    
[^387]: ABLEIST：LLM生成的招聘场景中的交叉性残障偏见

    ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios

    [https://arxiv.org/abs/2510.10998](https://arxiv.org/abs/2510.10998)

    该论文提出了ABLEIST评估框架，通过五个健全主义伤害指标和三个交叉性伤害指标，对六个LLM在2,820个招聘场景中的表现进行审计，揭示了针对残障候选人（尤其在性别、种姓等交叉身份维度）的显著偏见伤害，而许多先进模型无法检测到这些伤害。

    

    大型语言模型（LLM）因在招聘等高风险领域延续基于身份的歧视而日益受到审视，尤其是针对残障人士（PwD）的歧视。然而，现有研究仍主要聚焦于西方视角，忽视了性别和种姓等交叉性边缘化形式如何影响全球南方残障人士的经历。我们对六个LLM在2,820个招聘场景中进行了全面审计，这些场景涵盖多样的残障、性别、国籍和种姓特征。为了捕捉微妙的交叉性伤害与偏见，我们引入了ABLEIST（健全主义、励志化、超人化与象征化）框架，这是一套基于残障研究文献构建的评估体系，包含五个健全主义特定伤害指标和三个交叉性伤害指标。我们的结果显示，针对残障候选人的ABLEIST伤害显著增加——而许多最先进的模型未能检测到这些伤害。（摘要在此处截断）

    arXiv:2510.10998v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly under scrutiny for perpetuating identity-based discrimination in high-stakes domains such as hiring, particularly against people with disabilities (PwD). However, existing research remains largely Western-centric, overlooking how intersecting forms of marginalization--such as gender and caste--shape experiences of PwD in the Global South. We conduct a comprehensive audit of six LLMs across 2,820 hiring scenarios spanning diverse disability, gender, nationality, and caste profiles. To capture subtle intersectional harms and biases, we introduce ABLEIST (Ableism, Inspiration, Superhumanization, and Tokenism), a set of five ableism-specific and three intersectional harm metrics grounded in disability studies literature. Our results reveal significant increases in ABLEIST harms towards disabled candidates--harms that many state-of-the-art models failed to detect. These harms were furthe
    
[^388]: 面向空间预测的代理图划分

    Surrogate Graph Partitioning for Spatial Prediction

    [https://arxiv.org/abs/2510.07832](https://arxiv.org/abs/2510.07832)

    本文提出将代理图划分问题建模为混合整数二次规划，用于构建使分段内预测方差之和最小的空间分段，并针对其计算复杂度过高的问题开发了高效的近似求解方案。

    

    空间预测是指从空间分布的观测值中估计未观测到的值。尽管近期的进展提升了对多种观测类型建模的能力，但在需要可解释性的行业中，实际应用仍然有限。为弥合这一差距，能够解释黑盒预测器的代理模型为实现可解释决策提供了一条有前景的路径。在本研究中，我们提出一个图划分问题，用于构建空间分段，使各分段内个体预测值的方差之和最小。将数据点分配到分段的问题可以被表述为一个混合整数二次规划问题。虽然这种表述有可能识别出精确的分段，但随着数据点数量的增加，其计算复杂度会变得难以承受。受此挑战的启发，我们开发了一种利用该结构的近似求解方案……

    arXiv:2510.07832v2 Announce Type: replace-cross  Abstract: Spatial prediction refers to the estimation of unobserved values from spatially distributed observations. Although recent advances have improved the capacity to model diverse observation types, adoption in practice remains limited in industries that demand interpretability. To mitigate this gap, surrogate models that explain black-box predictors provide a promising path toward interpretable decision making. In this study, we propose a graph partitioning problem to construct spatial segments that minimize the sum of within-segment variances of individual predictions. The assignment of data points to segments can be formulated as a mixed-integer quadratic programming problem. While this formulation potentially enables the identification of exact segments, its computational complexity becomes prohibitive as the number of data points increases. Motivated by this challenge, we develop an approximation scheme that leverages the struc
    
[^389]: 大量零计算专家的混合：高率量化理论的视角

    Mixture of Many Zero-Compute Experts: A High-Rate Quantization Theory Perspective

    [https://arxiv.org/abs/2510.03151](https://arxiv.org/abs/2510.03151)

    本文借助高率量化理论，分析了一种由推理时零计算的单参数常数专家构成的MoE模型，推导了其在一维与多维输入下的测试误差逼近结果以及专家参数的统计学习性质。

    

    本文利用经典的高率量化理论，为回归任务中的专家混合模型提供了新的见解。我们的MoE模型通过将输入空间分割为多个区域来定义，每个区域由一个单参数专家负责，该专家在推理时充当零计算的常数预测器。受高率量化理论假设的启发，我们假设专家的数量足够多，使得它们对应的输入空间区域非常小。这使我们能够研究该MoE模型类的逼近误差：(i) 对于一维输入，我们给出了测试误差的公式及其最小化所需的分割方式与专家；(ii) 对于多维输入，我们给出了测试误差的上界并研究其最小化问题。此外，我们考虑在给定输入空间分割的情况下，从训练数据集中学习专家参数，并阐述了其统计学习特性。这使我们得出定理……

    arXiv:2510.03151v2 Announce Type: replace  Abstract: This paper uses classical high-rate quantization theory to provide new insights into mixture-of-experts (MoE) models for regression tasks. Our MoE is defined by a segmentation of the input space to regions, each with a single-parameter expert that acts as a constant predictor with zero-compute at inference. Motivated by high-rate quantization theory assumptions, we assume that the number of experts is sufficiently large to make their input-space regions very small. This lets us study the approximation error of our MoE model class: (i) for one-dimensional inputs, we formulate the test error and its minimizing segmentation and experts; (ii) for multidimensional inputs, we formulate an upper bound for the test error and study its minimization. Moreover, we consider the learning of the expert parameters from a training dataset, given an input-space segmentation, and formulate their statistical learning properties. This leads us to theore
    
[^390]: 贝叶斯最优实验设计中微分熵的近似

    Approximation of differential entropy in Bayesian optimal experimental design

    [https://arxiv.org/abs/2510.00734](https://arxiv.org/abs/2510.00734)

    该论文针对贝叶斯最优实验设计提出了一种高效计算方法：当似然熵与设计无关或可显式计算时，期望信息增益的嵌套积分可简化为单积分，并通过蒙特卡洛或拟蒙特卡洛代理近似证据密度来高效估计微分熵。

    

    贝叶斯最优实验设计为选择能够最大化所获信息的实验配置提供了一个原则性框架。在本工作中，我们考虑似然熵与设计无关或可被显式计算的情形，这适用于例如具有设计无关或可显式计算噪声熵的加性噪声模型。这将问题简化为最大熵估计，其中定义期望信息增益的嵌套积分塌缩为单个积分。我们的研究受到大规模推断问题（如逆问题）的启发，此类问题的计算成本主要由昂贵的似然评估所主导。在加性高斯噪声模型下，我们提出了一种计算方法：证据密度由蒙特卡洛或拟蒙特卡洛代理进行近似，而微分熵则使用标准方法进行评估，无需……

    arXiv:2510.00734v2 Announce Type: replace-cross  Abstract: Bayesian optimal experimental design provides a principled framework for selecting experimental configurations that maximize obtained information. In this work, we consider settings where the likelihood entropy is independent of the design or can be evaluated explicitly, which applies, e.g. to additive noise models with design-independent or explicitly computable noise entropy. This reduces the problem to maximum entropy estimation, where the nested integral defining the expected information gain collapses into a single integral.   Our study is motivated by large-scale inference problems, such as inverse problems, where the computational cost is dominated by expensive likelihood evaluations. Under additive Gaussian noise model, we propose a computational approach in which the evidence density is approximated by a Monte Carlo or quasi-Monte Carlo surrogate, while the differential entropy is evaluated using standard methods witho
    
[^391]: 工作记忆任务表现的多维贝叶斯主动机器学习

    Multidimensional Bayesian Active Machine Learning of Working Memory Task Performance

    [https://arxiv.org/abs/2510.00375](https://arxiv.org/abs/2510.00375)

    该研究提出一种基于高斯过程后验不确定性的双轴贝叶斯主动分类方法，在沉浸式虚拟环境中同时控制空间负荷与特征绑定负荷两个维度来评估工作记忆表现，其效果与传统单维阶梯式自适应方法相当。

    

    arXiv:2510.00375v3 公告类型：替换 摘要：尽管适应性实验设计已经超越了一维的、基于阶梯式的自适应方法，但大多数认知实验仍然只控制单一因素，并用标量来总结表现。相比之下，我们在此验证了一种贝叶斯双轴主动分类方法，应用于沉浸式虚拟测试环境中，并以5×5空间工作记忆重建任务作为应用领域。该方法控制两个变量：空间负荷L（被占用的方格数量）和项目的特征绑定负荷K（不同颜色的数量）。刺激采集由非参数高斯过程（GP）概率分类器的后验不确定性引导，该分类器输出的是关于空间上的一个曲面，而非单一阈值或最大广度值。在年轻成年人群体中，我们将GP驱动的自适应模式与传统自适应阶梯式的经典模式进行比较，后者仅在K=3时改变L。两种方法达到了相当的水平……

    arXiv:2510.00375v3 Announce Type: replace  Abstract: While adaptive experimental design has outgrown one-dimensional, staircase-based adaptations, most cognitive experiments still control a single factor and summarize performance with a scalar. In contrast, we validate here a Bayesian, two-axis, active-classification approach in an immersive virtual testing environment, using a 5-by-5 spatial working-memory reconstruction task as an application domain. Two variables are controlled: spatial load L (number of occupied tiles) and feature-binding load K (number of distinct colors) of items. Stimulus acquisition is guided by posterior uncertainty of a nonparametric Gaussian Process (GP) probabilistic classifier, which outputs a surface over (L, K) rather than a single threshold or maximum span value. In a young adult population, we compare GP-driven Adaptive Mode (AM) with a traditional adaptive staircase Classic Mode (CM), which varies L only at K = 3. Parity between the methods is achieve
    
[^392]: 发现学习到的哈密顿系统中的相空间结构

    Discovering Phase Space Structure in Learned Hamiltonian Systems

    [https://arxiv.org/abs/2509.23003](https://arxiv.org/abs/2509.23003)

    提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。

    

    力学启发的神经动力学模型提升了可解释性和物理一致性，但它们通常假设构型空间已知、物理参数固定，并且从给定初始条件出发进行确定性预测。我们提出结构化相空间生成对抗网络（SPS-GAN），这是一种条件生成模型，能够同时从笛卡尔轨迹和视频中学习结构化动力学。SPS-GAN 将端口哈密顿骨干网络与对抗式条件生成相结合，使保守系统、耗散系统和受迫系统能够在单一架构中得到表示。为了从原始观测中推断潜在的力学结构，我们采用循环坐标损失，在无需访问真实相空间的情况下同时恢复系统的自由度和正则坐标。在模拟系统和真实世界系统上的实验表明，SPS-GAN 在轨迹精度上优于监督动力学基线方法，并提升了视频一致性。

    arXiv:2509.23003v3 Announce Type: replace-cross  Abstract: Mechanics-inspired neural dynamical models improve interpretability and physical consistency, but they usually assume a known configuration space, fixed physical parameters, and deterministic prediction from a prescribed initial condition. We introduce Structured Phase Space GAN (SPS-GAN), a conditional generative model that learns structured dynamics from both Cartesian trajectories and video. SPS-GAN combines a port-Hamiltonian backbone with adversarial conditional generation, allowing conservative, dissipative, and forced systems to be represented in a single architecture. To infer latent mechanical structure from raw observations, we use a cyclic-coordinate loss that recovers both the system degrees of freedom and canonical coordinates without access to the true phase space. Experiments on simulated and real-world systems show that SPS-GAN improves trajectory accuracy over supervised dynamics baselines and video consistency
    
[^393]: 统一全局与局部时间信息的用户轨迹预测

    User Trajectory Prediction Unifying Global and Local Temporal Information

    [https://arxiv.org/abs/2508.02161](https://arxiv.org/abs/2508.02161)

    该论文提出一种融合MLP、多尺度卷积神经网络和交叉注意力的轨迹预测模型，统一提取轨迹数据中的全局与局部时间信息，在保持可接受推理时间的同时降低预测误差。

    

    轨迹预测对于制定预测用户移动性并支持提前准备的主动策略至关重要。因此，如何在可接受的推理时间内减少用户轨迹预测中的预测误差成为一个值得关注的问题。然而，轨迹数据同时包含全局和局部时间信息，这使得提取完整的时间模式变得复杂。此外，用户行为发生在不同的时间尺度上，进一步增加了捕捉行为模式的难度。为了应对这些挑战，本文提出了一种基于多层感知机（MLP）、多尺度卷积神经网络（MSCNN）和交叉注意力（CA）的轨迹预测模型。具体而言，MLP用于提取每个特征的全局时间信息；与此同时，MSCNN通过建模局部时间范围内特征之间的交互来提取局部时间信息。

    arXiv:2508.02161v2 Announce Type: replace  Abstract: Trajectory prediction is essential for formulating proactive strategies that anticipate user mobility and support advance preparation. Therefore, how to reduce the forecasting error in user trajectory prediction within an acceptable inference time arises as an interesting issue. However, trajectory data contains both global and local temporal information, complicating the extraction of the complete temporal pattern. Moreover, user behavior occurs over different time scales, increasing the difficulty of capturing behavioral patterns. To address these challenges, a trajectory prediction model based on multilayer perceptron (MLP), multi-scale convolutional neural network (MSCNN), and cross-attention (CA) is proposed. Specifically, MLP is used to extract the global temporal information of each feature. In parallel, MSCNN is employed to extract the local temporal information by modeling interactions among features within a local temporal 
    
[^394]: 针对Shapley值与概率值的回归调整蒙特卡洛估计器

    Regression-adjusted Monte Carlo Estimators for Shapley Values and Probabilistic Values

    [https://arxiv.org/abs/2506.11849](https://arxiv.org/abs/2506.11849)

    本文提出一种将蒙特卡洛采样与回归调整相结合的新方法来估计Shapley值等概率值，允许用任意可高效计算概率值的函数族（如XGBoost）替代线性回归，并通过交叉拟合实现无偏且性能领先的概率值估计。

    

    源于博弈论的概率值（如Shapley值、Banzhaf值和半值）已成为可解释人工智能领域的核心工具，被广泛应用于特征归因、数据归因、数据估值等任务。由于精确计算可能需要指数级时间，研究工作一直聚焦于两类高效近似技术：蒙特卡洛采样和线性回归建模。在本工作中，我们提出了一种将这两种技术相结合的新方法。与以往的算法相比，我们的方法更加灵活，允许将线性回归替换为任何其概率值可以被高效计算的函数族。这使得我们能够利用XGBoost等基于树的模型的准确性；通过交叉拟合，该方法还能产生无偏估计。在八个数据集上的实验表明，我们的方法在估计概率值方面达到了最先进的性能。

    arXiv:2506.11849v3 Announce Type: replace-cross  Abstract: With origins in game theory, probabilistic values like Shapley values, Banzhaf values, and semi-values have emerged as a central tool in explainable AI. They are used for feature attribution, data attribution, data valuation, and more. Since exact computation can require exponential time, research has focused on efficient approximation methods using two techniques: Monte Carlo sampling and linear regression formulations. In this work, we present a new way of combining both of these techniques. Our approach is more flexible than prior algorithms, allowing for linear regression to be replaced with any function family whose probabilistic values can be computed efficiently. This allows us to harness the accuracy of tree-based models like XGBoost; with cross-fitting, the method also produces unbiased estimates. From experiments across eight datasets, we find that our methods give state-of-the-art performance for estimating probabili
    
[^395]: UniConFlow：一种用于可认证运动规划的统一约束流匹配框架

    UniConFlow: A Unified Constrained Flow-Matching Framework for Certified Motion Planning

    [https://arxiv.org/abs/2506.02955](https://arxiv.org/abs/2506.02955)

    本文提出UniConFlow，一个统一的约束流匹配轨迹生成框架，通过新颖的预设时间归零函数系统性地处理等式与不等式约束，实现可认证的机器人运动规划。

    

    生成式模型已成为机器人运动生成中日益强大的工具，能够在各类任务中实现灵活且多模态的轨迹生成。然而，大多数现有方法在处理多种类型的约束（如避碰、执行器限制和动力学一致性）方面仍然有限，这些约束通常被单独或以启发式方式处理。在这项工作中，我们提出了UniConFlow，一个基于统一约束流匹配的轨迹生成框架，它系统性地融合了等式约束和不等式约束。此外，UniConFlow引入了一种新颖的预设时间归零函数，用于在推理过程中塑造时变引导场，使生成过程能够适应不同的系统模型和任务需求。进一步地，为了解决长时域和高维轨迹生成的计算挑战，我们提出了两种实用的……（原文在此处截断）

    arXiv:2506.02955v3 Announce Type: replace-cross  Abstract: Generative models have become increasingly powerful tools for robot motion generation, enabling flexible and multimodal trajectory generation across various tasks. Yet, most existing approaches remain limited in handling multiple types of constraints, such as collision avoidance, actuation limits, and dynamic consistency, which are typically addressed individually or heuristically. In this work, we propose UniConFlow, a unified constrained flow matching-based framework for trajectory generation that systematically incorporates both equality and inequality constraints. Moreover, UniConFlow introduces a novel prescribed-time zeroing function that shapes a time-varying guidance field during inference, allowing the generation process to adapt to varying system models and task requirements. Furthermore, to further address the computational challenges of long-horizon and high-dimensional trajectory generation, we propose two practica
    
[^396]: 基于真实硬件噪声量子扩散的物理启发生成式人工智能模型

    Physics-inspired Generative AI models via real hardware-based noisy quantum diffusion

    [https://arxiv.org/abs/2505.22193](https://arxiv.org/abs/2505.22193)

    本文提出两种受物理启发的量子扩散模型方案——利用量子随机行走中量子与经典动力学的相互作用降低生成图像的FID，以及仅用四个量子比特借助真实IBM量子硬件的固有噪声生成图像——为可扩展的量子生成式AI算法开辟了新路径。

    

    量子扩散模型（QDMs）是生成式人工智能中的一种新兴范式，旨在利用量子特性来提升其经典对应模型的性能。然而，由于近期量子设备的限制，现有算法难以扩展。在我们此前关于量子扩散模型工作的基础上，本文提出并实现了两种受物理启发的协议。在第一种方法中，我们采用量子随机行走的形式体系，表明正向过程中量子动力学与经典动力学的特定相互作用能够产生统计上更稳健的模型，其生成的MNIST图像集合的Fréchet Inception Distance（FID）低于完全使用经典动力学的情形。在第二种方法中，我们实现了一种算法，仅使用四个量子比特，通过利用真实IBM量子硬件的固有噪声来生成图像。我们的工作可以为大规模量子算法的新应用场景铺平道路。

    arXiv:2505.22193v2 Announce Type: replace-cross  Abstract: Quantum Diffusion Models (QDMs) are an emerging paradigm in Generative AI that aims to use quantum properties to improve the performances of their classical counterparts. However, existing algorithms are not easily scalable due to the limitations of near-term quantum devices. Following our previous work on QDMs, here we propose and implement two physics-inspired protocols. In the first, we use the formalism of quantum stochastic walks, showing that a specific interplay of quantum and classical dynamics in the forward process produces statistically more robust models generating sets of MNIST images with lower Fr\'echet Inception Distance (FID) than using totally classical dynamics. In the second approach, we realize an algorithm to generate images by exploiting the intrinsic noise of real IBM quantum hardware with only four qubits. Our work could be a starting point to pave the way for new scenarios for large-scale algorithms in
    
[^397]: 用于集成异构模型输出的可优化模型平均方法

    Optimizable Model Averaging for Ensembling Heterogeneous Model Outputs

    [https://arxiv.org/abs/2505.21857](https://arxiv.org/abs/2505.21857)

    提出可优化模型平均（OMA），一种黑盒集成方法，通过最小化正则化预测熵来学习非负组合权重以融合异构架构模型的预测概率，无需微调或参数合并即可改善性能与校准。

    

    基础模型集成可以提高准确性和鲁棒性，但当候选模型具有不同的架构时，完全微调和参数空间合并通常代价高昂甚至无法实现。我们提出了可优化模型平均（OMA），这是一种黑盒方法，它通过学习非负的模型组合权重（每个模型对应一个标量系数）来平均候选模型的预测概率，同时保持所有模型参数冻结，并通过最小化正则化预测熵来实现。一个核心挑战是最小化预测熵可能偏向那些过度自信但错误的模型：置信度的提升并不必然意味着准确性或校准的改善。因此，我们研究了正则化作为控制适应性的一种机制，并通过实验评估了其对校准的影响，观察到在选定配置中期望校准误差更低，同时相比模型输出平均方法具有更好的性能-校准权衡。

    arXiv:2505.21857v2 Announce Type: replace  Abstract: Foundation model ensembles can improve accuracy and robustness, but full fine-tuning and parameter-space merging are often expensive or impossible when candidate models have different architectures. We propose Optimizable Model Averaging (OMA), a black-box approach that learns nonnegative model-combination weights, one scalar coefficient per model, to average the candidate models' predictive probabilities, leaving all model parameters frozen, by minimizing regularized predictive entropy. A central challenge is that minimizing predictive entropy can favour confidently incorrect models: increased confidence need not imply improved accuracy or calibration. We therefore examine regularization as a mechanism for controlling adaptation and empirically evaluate its effect on calibration, observing lower expected calibration error in selected configurations, alongside better performance-calibration trade-offs than model output averaging. As 
    
[^398]: 在提示注入防御到达用户之前对其进行审计：强大对手揭示了什么

    Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal

    [https://arxiv.org/abs/2505.15738](https://arxiv.org/abs/2505.15738)

    该论文主张，由于基于微调的提示注入防御缺乏形式化保证且可能带来虚假安全感，开发者应在防御部署给用户之前用强大的对抗性攻击对其进行压力测试。

    

    大型语言模型越来越多地被部署在面向用户的产品中，这些产品在处理敏感个人数据的同时还需与不可信的外部内容交互。这使得它们成为提示注入攻击的首要目标——攻击者在外部内容中注入恶意指令以劫持模型。为缓解此类攻击，研究者提出了基于微调的防御方法，通过训练模型遵循良性的用户指令并忽略注入的指令。由于缺乏形式化保证，这些防御方法仅依据当时可获得的最有效的现实攻击进行实证评估，往往报告近乎为零的攻击成功率——这给开发者以及依赖这些防御的用户带来了虚假的安全感，同时也留下了未来更强大的攻击可能绕过这些防御、并在广泛部署的系统中造成严重后果的隐患。在本文中，我们主张开发者应对其防御……

    arXiv:2505.15738v3 Announce Type: replace-cross  Abstract: Large language models are increasingly deployed in user-facing products that handle sensitive personal data while interacting with untrusted external content. This makes them a prime target for prompt injection attacks, in which an adversary injects malicious instructions in that content to hijack the model. Fine-tuning-based defenses have been proposed to mitigate these attacks by training models to follow benign user instructions and ignore injected ones. Lacking formal guarantees, these defenses are evaluated empirically against the most effective realistic attacks available at the time, often reporting near-zero attack success rates -- giving developers, and the users who depend on them, a false sense of security and leaving open the possibility that future, more capable attacks could circumvent them and cause severe consequences in widely deployed systems. In this paper, we argue that developers should stress-test their de
    
[^399]: 面向最优且高效联邦学习的异构感知客户端采样

    Heterogeneity-Aware Client Sampling for Optimal and Efficient Federated Learning

    [https://arxiv.org/abs/2505.11304](https://arxiv.org/abs/2505.11304)

    本文首次对一般异构联邦学习进行了统一理论分析，揭示了通信异构性与计算异构性各自导致目标不一致的不同机制，并据此提出异构感知的客户端采样策略，以实现最优且高效的联邦学习。

    

    联邦学习（FL）通常涉及具有多样化通信和计算能力的客户端。这种异构性会显著扭曲优化动态并导致目标不一致，即全局模型收敛到一个可能远离所追求最优解的错误驻点。尽管其影响至关重要，但由于通信与计算异构性之间相互作用的内在复杂性，两者的联合效应在很大程度上仍未被探索。在本文中，我们揭示了异构通信和异构计算驱动联邦学习中不一致性的截然不同的机制。据我们所知，这是首个针对一般异构联邦学习的统一理论分析，为这两种异构性如何在任意局部求解器选择下共同扭曲优化轨迹提供了原则性的理解。基于这些见解……

    arXiv:2505.11304v4 Announce Type: replace-cross  Abstract: Federated learning (FL) commonly involves clients with diverse communication and computational capabilities. Such heterogeneity can significantly distort the optimization dynamics and lead to objective inconsistency, where the global model converges to an incorrect stationary point potentially far from the pursued optimum. Despite its critical impact, the joint effect of communication and computation heterogeneity has remained largely unexplored, due to the intrinsic complexity of their interaction. In this paper, we reveal the fundamentally distinct mechanisms through which heterogeneous communication and computation drive inconsistency in FL. To the best of our knowledge, this is the first unified theoretical analysis of general heterogeneous FL, offering a principled understanding of how these two forms of heterogeneity jointly distort the optimization trajectory under arbitrary choices of local solvers. Motivated by these i
    
[^400]: EmbodiTTA：面向具身视觉系统的资源高效测试时自适应方法

    EmbodiTTA: Resource-Efficient Test-Time Adaptation for Embodied Visual Systems

    [https://arxiv.org/abs/2505.00986](https://arxiv.org/abs/2505.00986)

    本文提出按需测试时自适应框架OD-TTA，仅在检测到显著域偏移时才触发自适应，并通过轻量级域偏移检测、源域选择和解耦BN更新三项技术，在边缘设备上实现了精准且资源高效的自适应。

    

    持续测试时自适应（CTTA）会在每一批新到达的数据上持续对已部署的模型进行自适应调整。尽管能够达到最优精度，现有的CTTA方法由于存在巨大的内存开销和能量消耗，在资源受限的边缘设备上实际应用性较差。在本工作中，我们首先提出了一种新颖的范式——按需测试时自适应，即仅在检测到显著的域偏移时才触发自适应过程。随后，我们提出了OD-TTA，一个用于在边缘设备上实现精准且高效自适应的按需TTA框架。OD-TTA包含三项创新技术：1）一种轻量级的域偏移检测机制，仅在需要时激活TTA，从而大幅降低整体计算开销；2）一个源域选择模块，为自适应选择合适的源模型，确保高且稳健的精度；3）一种解耦的批归一化（BN）更新方案，以实现内存高效的……

    arXiv:2505.00986v4 Announce Type: replace  Abstract: Continual Test-time adaptation (CTTA) continuously adapts the deployed model on every incoming batch of data. While achieving optimal accuracy, existing CTTA approaches present poor real-world applicability on resource-constrained edge devices, due to the substantial memory overhead and energy consumption. In this work, we first introduce a novel paradigm -- on-demand TTA -- which triggers adaptation only when a significant domain shift is detected. Then, we present OD-TTA, an on-demand TTA framework for accurate and efficient adaptation on edge devices. OD-TTA comprises three innovative techniques: 1) a lightweight domain shift detection mechanism to activate TTA only when it is needed, drastically reducing the overall computation overhead, 2) a source domain selection module that chooses an appropriate source model for adaptation, ensuring high and robust accuracy, 3) a decoupled Batch Normalization (BN) update scheme to enable mem
    
[^401]: 快速局地城市微气候模拟器：未见城市形态下的风场与温度场预测

    Fast localized urban microclimate emulator: wind and temperatures in unseen urban morphologies

    [https://arxiv.org/abs/2503.19708](https://arxiv.org/abs/2503.19708)

    本文提出基于傅里叶神经算子的快速局地城市微气候模拟器FLUME，通过引入多方向距离特征（MDDF）编码建筑几何信息，实现了数据高效、可泛化到未见城市形态的三维风场与温度场快速预测。

    

    城市微气候，包括由建筑几何形状塑造的风场和温度场，强烈影响着建筑能耗、行人风舒适度、污染物扩散、城市热岛效应以及公共健康。传统计算流体力学（CFD）方法在快速评估方面计算成本过高，而许多深度学习方法则需要大量训练数据，且难以泛化到未见过的城市配置。在本工作中，我们提出了快速局地城市微气候模拟器（FLUME），这是一个基于傅里叶神经算子的快速、数据高效且感知几何信息的框架，用于预测三维风场和温度场。FLUME假设在给定入流条件和热强迫下，这些平均场的空间变化很大程度上由周围建筑几何形状所控制。该框架引入了多方向距离特征（MDDF），用于在……范围内捕获建筑几何信息。

    arXiv:2503.19708v3 Announce Type: replace-cross  Abstract: Urban microclimate, encompassing wind and temperature fields shaped by building geometry, strongly affects building energy use, pedestrian wind comfort, pollutant dispersion, the urban heat island, and public health. Conventional Computational Fluid Dynamics (CFD) is computationally prohibitive for rapid assessments, while many deep learning approaches require extensive training data and struggle with generalization to unseen configurations. In this work, we present the Fast Localized Urban Microclimate Emulator (FLUME), a rapid, data-efficient, and geometry-aware framework based on a Fourier neural operator for predicting three-dimensional wind and temperature fields. FLUME assumes that, under prescribed inflow and thermal forcing, much of the spatial variation of these mean fields is controlled by surrounding building geometry. The framework introduces a Multi-Directional Distance Feature (MDDF), which captures geometry over 
    
[^402]: 解耦总方差与信噪比以改进扩散模型

    Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models

    [https://arxiv.org/abs/2502.08598](https://arxiv.org/abs/2502.08598)

    提出了一种总方差/信噪比解耦（TV/SNR）框架，使扩散模型噪声调度中的总方差和信噪比可被独立控制，并发现恒定总方差调度能够持续改进指数爆炸总方差的调度。

    

    arXiv:2502.08598v3 公告类型：替换 摘要：扩散模型作为一种强大的生成框架已被广泛应用于众多科学领域，但其固有的迭代采样过程仍然是一个显著的计算瓶颈。虽然通常可以通过减少反向扩散步数来缓解这一问题，但样本质量会对所选的噪声调度（即在每个步骤中引入噪声和降低信号的方式）变得高度敏感。尽管先前的工作在原始的方差保持和方差爆炸调度基础上有所改进，但这些方法只是被动地调整总方差，而无法对其进行直接的独立控制。在这项工作中，我们提出了一种新颖的总方差/信噪比解耦（TV/SNR）框架，使总方差和信噪比能够被独立控制。我们的方法揭示，通过采用恒定总方差调度并保持（摘要原文在此处截断）……

    arXiv:2502.08598v3 Announce Type: replace  Abstract: Diffusion models have been widely adopted as a powerful generative framework across numerous scientific domains, yet their inherently iterative sampling process remains a significant computational bottleneck. While this is typically mitigated by reducing the number of reverse diffusion steps, sample quality becomes highly sensitive to the chosen noise schedule, i.e. the manner in which noise is introduced and the signal is reduced at each step. Although prior work has improved on the original variance-preserving and variance-exploding schedules, these approaches passively adjust the total variance, without direct independent control over it. In this work, we propose a novel total-variance/signal-to-noise-ratio disentangled (TV/SNR) framework, where TV and SNR can be controlled independently. Our approach reveals that schedules with exponentially exploding TV can be consistently improved by adopting a constant TV schedule while preser
    
[^403]: 带任意时刻背包约束的老虎机问题

    Bandits with Anytime Knapsacks

    [https://arxiv.org/abs/2501.18560](https://arxiv.org/abs/2501.18560)

    本文提出了SUAK算法，用于解决带任意时刻成本约束的背包老虎机这一新问题，通过战略性地低利用预算并在必要时自适应跳过轮次，实现了与经典背包老虎机问题相同的 $O(K \log T)$ 问题相关遗憾上界。

    

    我们研究了带任意时刻背包的老虎机问题，这是背包老虎机问题的一个新变体，其中存在一个“任意时刻”的成本约束，而非总成本预算。这一问题设定带来了额外的复杂性，因为它要求在整个决策过程中始终遵守该约束。我们提出了SUAK（针对任意时刻背包的战略性低利用算法），该算法利用置信上界来识别最优的臂组合，同时在探索与利用之间保持平衡。SUAK是一种自适应算法，它在决策过程中战略性地利用每一轮的可用预算，并在可能违反任意时刻成本约束时跳过该轮。特别地，SUAK略微降低了对可用成本预算的利用，以减少跳过轮次的需求。我们证明SUAK达到了与经典问题相同的问题相关遗憾上界 $O(K \log T)$。

    arXiv:2501.18560v2 Announce Type: replace  Abstract: We consider bandits with anytime knapsacks (BwAK), a novel version of the BwK problem where there is an \textit{anytime} cost constraint instead of a total cost budget. This problem setting introduces additional complexities as it mandates adherence to the constraint throughout the decision-making process. We propose SUAK (Strategic Under-utilization for Anytime Knapsacks), an algorithm that utilizes upper confidence bounds to identify the optimal mixture of arms while maintaining a balance between exploration and exploitation. SUAK is an adaptive algorithm that strategically utilizes the available budget in each round in the decision-making process and skips a round when it is possible to violate the anytime cost constraint. In particular, SUAK slightly under-utilizes the available cost budget to reduce the need for skipping rounds. We show that SUAK attains the same problem-dependent regret upper bound of $ O(K \log T)$ established
    
[^404]: 一种面向离散时间分数阶动力系统辨识的采样复杂度感知框架

    A Sampling Complexity-aware Framework for Discrete-time Fractional-Order Dynamical System Identification

    [https://arxiv.org/abs/2501.17499](https://arxiv.org/abs/2501.17499)

    本文提出了一种采样复杂度感知的离散时间分数阶动力系统辨识算法，通过将辨识问题转化为最小二乘估计并避免求解高阶多项式方程，有效解决了噪声数据下的系统参数辨识问题，并给出了参数估计误差的理论界。

    

    许多复杂的生物、自然和人造系统都表现出非马尔可夫动力学特性，这些特性可以通过分数阶微分方程进行建模，然而，我们目前缺乏考虑样本复杂度的系统辨识策略。为此，我们提出了一种仿射离散时间分数阶动力系统（FoDS）辨识算法，并提供了详细的样本复杂度分析。该算法有效解决了存在噪声数据情况下的FoDS辨识难题。所提出的算法包含两个关键步骤：首先，它避免了求解高阶多项式方程，否则将导致分数阶出现多个潜在解；其次，将辨识问题重新表述为最小二乘估计问题，从而使我们能够推断系统参数。在假设具有先验知识的条件下，我们推导出了FoDS参数估计误差的期望界和概率界。

    arXiv:2501.17499v1 Announce Type: cross  Abstract: A variety of complex biological, natural and man-made systems exhibit non-Markovian dynamics that can be modeled through fractional order differential equations, yet, we lack sample comlexity aware system identification strategies. Towards this end, we propose an affine discrete-time fractional order dynamical system (FoDS) identification algorithm and provide a detailed sample complexity analysis. The algorithm effectively addresses the challenges of FoDS identification in the presence of noisy data. The proposed algorithm consists of two key steps. Firstly, it avoids solving higher-order polynomial equations, which would otherwise result in multiple potential solutions for the fractional orders. Secondly, the identification problem is reformulated as a least squares estimation, allowing us to infer the system parameters. We derive the expectation and probabilistic bounds for the FoDS parameter estimation error, assuming prior knowled
    
[^405]: 通过反演理解基于Transformer的视觉模型

    Understanding Transformer-based Vision Models through Inversion

    [https://arxiv.org/abs/2412.06534](https://arxiv.org/abs/2412.06534)

    本文提出一种新颖的模块化特征反演方法，能更高效地从大规模Transformer视觉模型的中间表示重建语义更连贯的图像，并借此揭示了DETR与ViT等模型内部机制随网络深度的差异。

    

    理解深度神经网络的内部机制仍然是机器学习中的核心挑战。在计算机视觉领域，一种有前景但仅被初步探索的方法是通过逆向网络进行特征反演，即使用训练好的逆向网络从中间表示重建图像。在本研究中，我们重新审视了通过逆向网络进行特征反演的方法，提出了一种新颖的模块化变体，使该技术在计算上更加高效，并且在某些情况下能够生成语义上更加连贯的图像重建结果。我们将该方法应用于大规模基于Transformer的视觉模型，具体包括检测Transformer（Detection Transformer）、视觉Transformer（Vision Transformer）、Swin Transformer和数据高效图像Transformer（Data-Efficient Image Transformer），并分析了重建结果随网络深度的变化。我们的主要分析比较了检测Transformer和视觉Transformer，二者展现出最具信息量的差异。

    arXiv:2412.06534v5 Announce Type: replace-cross  Abstract: Understanding the internal mechanisms of deep neural networks remains a central challenge in machine learning. In computer vision, one promising yet only preliminarily explored approach is feature inversion via inverse networks, which reconstructs images from intermediate representations using trained inverse networks. In this study, we revisit feature inversion via inverse networks, introducing a novel, modular variant that enables a computationally more efficient application of the technique while, in some cases, producing semantically more coherent image reconstructions. We apply our method to large-scale transformer-based vision models, specifically Detection Transformer, Vision Transformer, Swin Transformer, and Data-Efficient Image Transformer, analyzing the resulting reconstructions across network depth. Our main analysis compares Detection Transformer and Vision Transformer, which exhibit the most informative difference
    
[^406]: 条件Vendi分数：面向生成式AI模型与大语言模型的提示感知多样性评估

    Conditional Vendi Score: Prompt-Aware Diversity Evaluation for Generative AI Models and LLMs

    [https://arxiv.org/abs/2411.02817](https://arxiv.org/abs/2411.02817)

    该论文提出了基于条件熵的条件Vendi和条件RKE两种新度量方法，能够在提示引导的生成任务中分离出模型自身引起的多样性，为生成式AI模型和大语言模型提供了提示感知的多样性评估工具。

    

    由文本提示引导的生成式模型在保真度和提示对齐方面已被广泛评估，但其产生多样化输出的能力仍未得到充分研究。现有的多样性指标（如基于核矩阵的冯·诺依曼熵和Rényi熵的Vendi与RKE）是为无条件模型设计的，无法区分由提示引起的变异与模型自身引起的变异。我们通过引入条件Vendi（Conditional-Vendi）和条件RKE（Conditional-RKE）来填补这一空白，这两种多样性度量源自半正定矩阵的条件熵。这些分数能够在提示引导的生成中分离出由模型引起的多样性，其中条件RKE具有O(1/√n)的收敛速率。对于条件Vendi，我们提出了一种截断谱近似方法，可提供可扩展且一致的估计。在文本到图像、图像描述生成以及大语言模型任务上的实验表明，条件分数能够……（原文摘要在此处截断）

    arXiv:2411.02817v3 Announce Type: replace-cross  Abstract: Generative models guided by text prompts are widely evaluated for fidelity and prompt alignment, yet their ability to produce diverse outputs remains underexplored. Existing diversity metrics such as Vendi and RKE, which are based on the von Neumann and R\'enyi entropies of kernel matrices, were developed for unconditional models and cannot distinguish prompt-induced from model-induced variability. We address this gap by introducing \textit{Conditional-Vendi} and \textit{Conditional-RKE}, diversity measures derived from the conditional entropy of positive semidefinite matrices. These scores isolate model-induced diversity in prompt-guided generation, with Conditional-RKE enjoying an $O(1/\sqrt{n})$ convergence rate. For Conditional-Vendi, we introduce a truncated-spectrum approximation that yields scalable and consistent estimates. Experiments on text-to-image, image-captioning, and LLM tasks show that the conditional scores re
    
[^407]: 具有时间一致性的深度端到端生存分析

    Deep End-to-End Survival Analysis with Temporal Consistency

    [https://arxiv.org/abs/2410.06786](https://arxiv.org/abs/2410.06786)

    该论文提出了一种受深度Q网络启发的新型生存分析框架，通过延迟目标网络实现时间一致性约束下的端到端训练，能够高效处理大规模纵向数据并建模复杂的时间依赖关系。

    

    在本研究中，我们提出了一种新颖的生存分析算法，旨在高效处理大规模纵向数据。我们的方法受到强化学习原理的启发，特别是深度Q网络（Deep Q-Network）范式，将时序学习概念扩展到生存回归中。我们方法的核心论点依赖于时间一致性，即相邻状态之间生存预测的递归一致性条件。我们的框架引入了一个延迟目标网络，将时间一致性训练目标的生成与模型更新解耦，从而提供更稳定的训练信号，同时支持小批量优化和端到端训练。此外，该框架支持任意复杂的网络架构，能够建模复杂的时间依赖关系。通过大量实验，我们提供了实证证据，证明了我们的框架利用时间一致性的能力。

    arXiv:2410.06786v2 Announce Type: replace  Abstract: In this study, we present a novel Survival Analysis algorithm designed to efficiently handle large-scale longitudinal data. Our approach draws inspiration from Reinforcement Learning principles, particularly the Deep Q-Network paradigm, extending Temporal Learning concepts to Survival Regression. A central argument in our method relies on temporal consistency, a recursive consistency condition between survival predictions made at consecutive states. Our framework introduces a delayed target network that decouples the generation of temporally consistent training targets from model updates, providing a more stable training signal while enabling mini-batch optimization and end-to-end training. Additionally, it supports arbitrarily complex architectures, enabling the modeling of intricate temporal dependencies. Through numerous experiments we provide empirical evidence demonstrating our framework's ability to exploit temporal consistency
    
[^408]: 实例级难度：机器遗忘中被忽视的视角

    Instance-Level Difficulty: A Missing Perspective in Machine Unlearning

    [https://arxiv.org/abs/2410.03043](https://arxiv.org/abs/2410.03043)

    本文从实例级角度研究机器遗忘的难度问题，总结出使遗忘某个数据点困难的四个因素，并证明这些因素仅取决于目标模型及其训练数据，而与具体的遗忘算法无关。

    

    当前关于深度机器遗忘的研究主要集中于提升或评估遗忘方法的整体有效性，而忽视了遗忘单个训练样本时难度的差异。因此，机器遗忘更广泛的可行性仍然未被充分探索。本文通过对多种遗忘算法和数据集进行彻底的实例级遗忘性能分析，研究了使机器遗忘变得困难的关键因素。特别地，我们总结了导致遗忘某个数据点困难的四个因素，并通过实验表明这些因素与特定的遗忘算法无关，而仅与目标模型及其训练数据相关。基于这些发现，我们认为机器遗忘研究应当关注实例级的遗忘难度。

    arXiv:2410.03043v3 Announce Type: replace  Abstract: Current research on deep machine unlearning primarily focuses on improving or evaluating the overall effectiveness of unlearning methods while overlooking the varying difficulty of unlearning individual training samples. As a result, the broader feasibility of machine unlearning remains under-explored. This paper studies the cruxes that make machine unlearning difficult through a thorough instance-level unlearning performance analysis over various unlearning algorithms and datasets. In particular, we summarize four factors that make unlearning a data point difficult, and we empirically show that these factors are independent of a specific unlearning algorithm but only relevant to the target model and its training data. Given these findings, we argue that machine unlearning research should pay attention to the instance-level difficulty of unlearning.
    
[^409]: 基于杠杆分数采样的可证明精确的Shapley值估计

    Provably Accurate Shapley Value Estimation via Leverage Score Sampling

    [https://arxiv.org/abs/2410.01917](https://arxiv.org/abs/2410.01917)

    提出Leverage SHAP，通过对Kernel SHAP的轻量级改进，利用杠杆分数采样将模型评估次数从指数级降至O(n log n)，实现了可证明精确的Shapley值估计。

    

    Shapley值最初源于博弈论，如今已成为可解释机器学习中的核心工具，用于将模型预测归因于特定的输入特征。然而，精确计算Shapley值的代价十分高昂：对于一个具有 $n$ 个特征的一般模型，需要进行 $O(2^n)$ 次模型评估。为解决这一问题，近似算法被广泛采用，其中最流行的算法之一是Kernel SHAP，该算法与具体模型无关，且在实践中表现出色。在我们最初提交论文时，Kernel SHAP尚缺乏强有力的非渐近复杂度保证。为解决这一问题，我们提出了Leverage SHAP，这是对Kernel SHAP的一种轻量级改进，仅需 $O(n\log n)$ 次模型评估即可提供可证明精确的Shapley值估计。我们的方法利用了Shapley值估计与不可知主动学习之间的联系，通过采用……

    arXiv:2410.01917v3 Announce Type: replace-cross  Abstract: Originally introduced in game theory, Shapley values have emerged as a central tool in explainable machine learning, where they are used to attribute model predictions to specific input features. However, computing Shapley values exactly is expensive: for a general model with $n$ features, $O(2^n)$ model evaluations are necessary. To address this issue, approximation algorithms are widely used. One of the most popular is the Kernel SHAP algorithm, which is model agnostic and remarkably effective in practice. At the time of our original submission, no strong non-asymptotic complexity guarantees were known for Kernel SHAP. We address this issue by introducing Leverage SHAP, a lightweight modification of Kernel SHAP that provides provably accurate Shapley value estimates with just $O(n\log n)$ model evaluations. Our approach takes advantage of a connection between Shapley value estimation and agnostic active learning by employing 
    
[^410]: 再生核希尔伯特空间中的数据驱动随机最优控制

    Data-Driven Stochastic Optimal Control in Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2407.16407](https://arxiv.org/abs/2407.16407)

    该论文提出一种完全数据驱动的方法，在动力学和成本函数均未知的情况下，通过在再生核希尔伯特空间中学习马尔可夫转移算子，并与随状态维度线性扩展的凸算子理论HJB递归相结合，实现非线性随机扩散系统的最优控制。

    

    本文提出了一种完全数据驱动的方法，用于由随机扩散所描述的非线性仿射控制系统的最优控制。研究聚焦于非线性动力学和阶段成本函数均未知、仅提供控制惩罚函数和约束条件的场景。为此，我们将状态概率密度嵌入到再生核希尔伯特空间（RKHS）中，以利用算子回归领域的最新进展，从而识别与受控扩散过程相关联的马尔可夫转移算子。这种算子学习方法与随状态维度线性扩展的凸算子理论哈密顿-雅可比-贝尔曼（HJB）递归自然融合，能够有效求解广泛的非线性最优控制问题。数值结果证明了该方法处理多种非线性控制任务的能力，包括自主水下航行器的深度调节。

    arXiv:2407.16407v3 Announce Type: replace-cross  Abstract: This paper proposes a fully data-driven approach for optimal control of nonlinear control-affine systems represented by a stochastic diffusion. The focus is on the scenario where both the nonlinear dynamics and stage cost functions are unknown, while only a control penalty function and constraints are provided. To this end, we embed state probability densities into a reproducing kernel Hilbert space (RKHS) to leverage recent advances in operator regression, thereby identifying Markov transition operators associated with controlled diffusion processes. This operator learning approach integrates naturally with convex operator-theoretic Hamilton-Jacobi-Bellman recursions that scale linearly with state dimensionality, effectively solving a wide range of nonlinear optimal control problems. Numerical results demonstrate its ability to address diverse nonlinear control tasks, including the depth regulation of an autonomous underwater 
    
[^411]: TernaryLLM：三值化大语言模型

    TernaryLLM: Ternarized Large Language Model

    [https://arxiv.org/abs/2406.07177](https://arxiv.org/abs/2406.07177)

    本文提出TernaryLLM，通过双重可学习三值化（DLT）和对离群值友好的特征知识蒸馏（OFF）两项创新技术，克服了权重与激活中离群值的挑战，实现了大语言模型的高效三值化，从而大幅降低计算成本和内存需求。

    

    大语言模型（LLM）在自然语言处理（NLP）任务上取得了卓越的性能，但其高昂的计算成本和内存需求制约了其发展。三值化作为一种极端形式的量化，通过减少内存占用并支持高能效的浮点加法运算，提供了一种解决方案。然而，将三值化应用于大语言模型时，会面临权重和激活值中离群值带来的挑战。在本工作中，通过观察权重中不对称的离群值和非零均值现象，我们提出了双重可学习三值化（Dual Learnable Ternarization, DLT），使尺度和偏移参数均可学习。我们还提出了对离群值友好的特征知识蒸馏（Outlier-Friendly Feature Knowledge Distillation, OFF），以恢复极低位宽量化中丢失的信息。所提出的OFF能够融入语义信息，且对离群值不敏感。OFF的核心在于最大化三值化模型与浮点模型中特征之间的互信息。

    arXiv:2406.07177v2 Announce Type: replace  Abstract: Large language models (LLMs) have achieved remarkable performance on Natural Language Processing (NLP) tasks, but they are hindered by high computational costs and memory requirements. Ternarization, an extreme form of quantization, offers a solution by reducing memory usage and enabling energy-efficient floating-point additions. However, applying ternarization to LLMs faces challenges stemming from outliers in both weights and activations. In this work, observing asymmetric outliers and non-zero means in weights, we introduce Dual Learnable Ternarization (DLT), which enables both scales and shifts to be learnable. We also propose Outlier-Friendly Feature Knowledge Distillation (OFF) to recover the information lost in extremely low-bit quantization. The proposed OFF can incorporate semantic information and is insensitive to outliers. At the core of OFF is maximizing the mutual information between features in ternarized and floating-p
    
[^412]: 具有近似保证的线性化Wasserstein降维方法

    Linearized Wasserstein dimensionality reduction with approximation guarantees

    [https://arxiv.org/abs/2302.07373](https://arxiv.org/abs/2302.07373)

    该论文提出LOT Wassmap算法，利用Sinkhorn距离和线性化最优传输等近似方案避免计算两两Wasserstein距离矩阵，从而高效揭示概率测度数据集在Wasserstein空间中的低维结构，并为此类近似下的嵌入质量提供了理论保证。

    

    我们提出了LOT Wassmap，一种计算上可行的算法，用于揭示Wasserstein空间中的低维结构。该算法的动机来自这样一个观察：许多数据集天然地应被理解为概率测度而非$\mathbb{R}^n$中的点，而为此类数据集寻找低维描述需要在Wasserstein空间中使用流形学习算法。现有的大多数算法都依赖于计算两两Wasserstein距离矩阵，这对于高维大规模数据集而言在计算上极具挑战性。我们的算法利用Sinkhorn距离和线性化最优传输等近似方案来加速计算，尤其避免了计算两两距离矩阵。我们为在这种近似下的嵌入质量提供了理论保证，包括当概率测度的显式表达不可用、而必须处理从测度中采样得到的样本的情形。

    arXiv:2302.07373v2 Announce Type: replace  Abstract: We introduce LOT Wassmap, a computationally feasible algorithm to uncover low-dimensional structures in the Wasserstein space. The algorithm is motivated by the observation that many datasets are naturally interpreted as probability measures rather than points in $\mathbb{R}^n$, and that finding low-dimensional descriptions of such datasets requires manifold learning algorithms in the Wasserstein space. Most available algorithms are based on computing the pairwise Wasserstein distance matrix, which can be computationally challenging for large datasets in high dimensions. Our algorithm leverages approximation schemes such as Sinkhorn distances and linearized optimal transport to speed-up computations, and in particular, avoids computing a pairwise distance matrix. We provide guarantees on the embedding quality under such approximations, including when explicit descriptions of the probability measures are not available and one must dea
    

